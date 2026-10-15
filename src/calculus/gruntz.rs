//! Gruntz algorithm for computing symbolic limits.
//!
//! Implements the algorithm from Dominik Gruntz's PhD thesis (ETH Zürich, 1996)
//! for computing limits of expressions as a variable approaches infinity.
//! Any limit `lim(x→a)` is first converted to `lim(x→∞)` via substitution.
//!
//! # Algorithm Overview
//!
//! 1. Find the **MRV (Most Rapidly Varying)** set — subexpressions growing fastest
//! 2. If `x` is in the MRV set, "move up" by replacing `x → exp(x)`
//! 3. Pick `ω` from the MRV set such that `ω → 0` as `x → ∞`
//! 4. Rewrite the expression in terms of `ω`
//! 5. Extract the leading term `c₀ · ω^e₀`
//! 6. If `e₀ > 0`: limit is 0. If `e₀ < 0`: limit is ±∞. If `e₀ = 0`: recurse on `c₀`.
//!
//! # References
//!
//! - Gruntz, D. "On Computing Limits in a Symbolic Manipulation System"
//!   PhD Thesis, ETH Zürich, 1996.
//! - The structure (`SubsSet`, `mrv`, `sign`, `rewrite`) follows SymPy's
//!   `sympy/series/gruntz.py` (BSD-3-Clause; notice in
//!   `THIRD-PARTY-NOTICES.md`).

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{Signed, ToPrimitive, Zero};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::base::arena::Arena;
use crate::base::budget::{BudgetKind, Work};
use crate::base::func::{Elementary, Family, Func, InternalFn};
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::base::truth::Truth;
use crate::calculus::calculus_util::{ZeroTest, settle_constant};

/// Maximum recursion depth for the Gruntz algorithm.  The depth grows with
/// the nesting of the expression as well as with the algorithm's own
/// recursion (`mrv` → `compare` → `limitinf` → …); at 15 the towers of
/// Gruntz's thesis (`exp(exp(exp(x + e⁻ˣ)))/exp(exp(exp(x)))`, `Γ(x + e⁻ˣ)
/// − Γ(x)`, `ψ(ψ(ψ(x)))`, 8.2–8.18) were refused.  The work budget
/// still bounds the breadth; the deepest of those needs under 256 KiB of
/// stack in release and 1 MiB in debug builds.
const MAX_DEPTH: usize = 30;

/// Total work units one top-level Gruntz invocation may spend before it
/// gives up. Every recursive step of the algorithm (`limitinf`, `mrv`,
/// `leadterm`, `compare`, `sign_at_inf`, `rewrite`) costs one unit and every
/// series expansion costs [`SERIES_COST`]. The battery of ~100 standard
/// limits in `tests/v02_transforms_limits.rs` uses well under 2 000 units
/// per invocation; the cap is set an order of magnitude higher so that
/// legitimate limits never hit it while pathological inputs are cut off
/// within a fraction of a second.
const MAX_WORK: u64 = 20_000;

/// Work charged for one series expansion (see [`MAX_WORK`]), on top of a
/// size-proportional charge.
const SERIES_COST: u32 = 25;

/// Work the failed last resorts of one top-level Gruntz invocation may
/// spend in all ([`Budget::resort`]).
const MAX_RESORT_WORK: u64 = 5_000;

/// Largest denominator of a fractional power of `ω` that
/// [`leadterm_last_resort`] ramifies (`ω = u^q`).
const MAX_RAMIFICATION: i64 = 30;

/// Largest order (in `u`) of a ramified expansion in
/// [`leadterm_last_resort`].
const MAX_RAMIFIED_ORDER: i64 = 60;

/// Largest expression *tree* (nodes with multiplicity) the series-based
/// fallbacks will expand. Nested function series substituted into each
/// other grow exponentially; beyond this size a single `expand` can take
/// seconds, so the attempt is abandoned instead.
const MAX_TREE_SIZE: usize = 4_000;

/// Work budget shared by one top-level Gruntz invocation.
///
/// The Gruntz algorithm is recursive in several mutually dependent ways
/// (`mrv` calls `limitinf` on exponents, `compare` calls `limitinf` on log
/// ratios, `leadterm` falls back to series expansion, …). Each recursion is
/// depth-limited, but the *breadth* of the search is not, so a pathological
/// input can still take a very long time without any single recursion
/// exceeding its depth limit. The budget bounds the total amount of work
/// ([`Work`] of kind [`BudgetKind::LimitSteps`]): once a charge does not
/// fit, it and every later one fail with
/// [`BudgetExhausted`](crate::base::errors::SymplexError::BudgetExhausted)
/// (up to 0.49 `ComputationFailed`), the error propagates to the top, and
/// the caller returns an unevaluated `Limit` node.
///
/// The budget also hands out the fresh dummy symbols (`__gw0`, `__gw1`, …)
/// used for MRV substitutions.
#[derive(Debug)]
pub(crate) struct Budget {
    dummies: u32,
    work: Spent,
    /// Work spent by failed last resorts ([`leadterm_last_resort`]), which
    /// is given back to `work` so that a refusal there leaves the rest of
    /// the computation as it was; bounded by [`MAX_RESORT_WORK`].
    resort: Spent,
}

/// A [`Work`] that stays spent once a charge did not fit: the computation
/// it bounds is abandoned, also where a caller catches the error and tries
/// another method.  `attempted` counts every unit asked for, those that
/// did not fit included, which is what a failed last resort gives back
/// ([`leadterm_last_resort`]).
#[derive(Clone, Copy, Debug)]
struct Spent {
    work: Work,
    attempted: u64,
}

impl Spent {
    const fn new(cap: u64) -> Self {
        Spent {
            work: Work::new(BudgetKind::LimitSteps, cap),
            attempted: 0,
        }
    }

    /// Charge `units`; `Err` if they do not fit or an earlier charge did
    /// not.
    fn charge(&mut self, units: u64) -> Result<(), crate::base::errors::SymplexError> {
        self.attempted = self.attempted.saturating_add(units);
        if self.overrun() {
            return Err(crate::base::errors::SymplexError::BudgetExhausted {
                operation: "gruntz",
                kind: BudgetKind::LimitSteps,
            });
        }
        self.work.charge(units, "gruntz")
    }

    /// Has a charge not fitted?
    const fn overrun(&self) -> bool {
        self.attempted > self.work.cap()
    }
}

impl Budget {
    pub(crate) fn new() -> Self {
        Self {
            dummies: 0,
            work: Spent::new(MAX_WORK),
            resort: Spent::new(MAX_RESORT_WORK),
        }
    }

    /// Charge `weight` units of work; `Err` once the budget is exhausted.
    fn tick(&mut self, weight: u32) -> Result<(), crate::base::errors::SymplexError> {
        let charged = self.work.charge(u64::from(weight));
        if charged.is_err() {
            tracing::warn!(work = self.work.attempted, "gruntz: work budget exhausted");
        }
        charged
    }

    /// Charge for an expensive operation on `e` proportionally to its tree
    /// size; `Err` if `e` is larger than [`MAX_TREE_SIZE`] or the budget is
    /// exhausted.
    fn charge_size(
        &mut self,
        arena: &Arena,
        e: ExprId,
    ) -> Result<(), crate::base::errors::SymplexError> {
        let size = crate::transforms::pattern::tree_size_capped(arena, e, MAX_TREE_SIZE + 1);
        if size > MAX_TREE_SIZE {
            tracing::warn!(size, "gruntz: expression too large for series fallback");
            return Err(crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz",
                reason: "intermediate expression too large for the limit engine".into(),
            });
        }
        self.tick((size / 8) as u32)
    }

    /// Work units spent so far (for diagnostics and tests).
    #[cfg(test)]
    fn work(&self) -> u64 {
        self.work.attempted
    }
}

fn fresh_dummy(arena: &mut Arena, budget: &mut Budget) -> ExprId {
    let n = budget.dummies;
    budget.dummies += 1;
    arena.symbol(&format!("__gw{n}"))
}

/// A fresh dummy declared positive, for the variable `x → +∞` of
/// `limitinf`.  The limit is taken along the positive real axis, so the
/// real-variable identities that simplification only applies to known-real
/// arguments (`exp(x)^(1/x) = e`, `√(x²) = x`) hold for it — as in
/// SymPy's `limitinf`, which substitutes a positive dummy the same way.
/// (`__gwp…` is a separate name family from the MRV dummies `__gw…`,
/// which carry no assumptions.)
fn fresh_positive_dummy(arena: &mut Arena, budget: &mut Budget) -> ExprId {
    let n = budget.dummies;
    budget.dummies += 1;
    arena.positive_symbol(&format!("__gwp{n}"))
}

/// Maclaurin series of `expr` in `var` to `order` terms, charged against
/// the work budget.
///
/// This is a guarded re-implementation of the Taylor loop in
/// [`series`](crate::calculus::series::series):
///
/// * every derivative is size-checked before it is differentiated again
///   (derivatives of nested functions grow exponentially, and one
///   unguarded `series` of `sin(sin x) − tan(tan x)` to order 10 takes
///   seconds);
/// * the value at `var = 0` is obtained with
///   [`safe_substitute`](crate::calculus::limit::safe_substitute), so a
///   pole hidden by canonicalization (`0·zoo → 0`) is reported as an
///   error instead of silently producing a wrong coefficient.
fn budgeted_series(
    arena: &mut Arena,
    budget: &mut Budget,
    expr: ExprId,
    var: ExprId,
    order: u32,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    budget.tick(SERIES_COST)?;
    let zero = arena.zero();
    let pole = || crate::base::errors::SymplexError::ComputationFailed {
        operation: "gruntz::series",
        reason: "pole detected at expansion point".into(),
    };

    let mut terms: Vec<ExprId> = Vec::with_capacity(order as usize);
    let mut deriv = expr;
    let mut factorial = Ratio::from_integer(BigInt::from(1));
    for k in 0..order {
        budget.charge_size(arena, deriv)?;
        // A formal derivative (`d re(u)/dx` for a variable not declared
        // real) has no value to substitute.
        if crate::base::walk::has_unevaluated(arena, deriv) {
            return Err(crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz::series",
                reason: "a derivative of the expression stays formal".into(),
            });
        }
        let value = if crate::base::walk::contains(arena, deriv, var) {
            crate::calculus::limit::safe_substitute(arena, deriv, var, zero).ok_or_else(pole)?
        } else {
            crate::transforms::eval::eval(arena, deriv)
        };
        if is_infinite(arena, value) || value == arena.nan() || contains_singular_atom(arena, value)
        {
            return Err(pole());
        }
        if !arena.is_zero_structural(value) {
            let term = if k == 0 {
                value
            } else {
                let k_id = arena.int(i64::from(k));
                let power = arena.pow(var, k_id);
                let coeff_nid =
                    arena.intern_num(Ratio::from_integer(BigInt::from(1)) / factorial.clone());
                let coeff = arena.intern(ExprNode::Num(coeff_nid));
                arena.mul(&[value, coeff, power])
            };
            terms.push(term);
        }
        if k + 1 < order {
            if !crate::base::walk::contains(arena, deriv, var) {
                break;
            }
            deriv = crate::transforms::diff::diff(arena, deriv, var);
            deriv = crate::transforms::eval::eval(arena, deriv);
            factorial *= Ratio::from_integer(BigInt::from(i64::from(k) + 1));
        }
    }

    Ok(match terms.len() {
        0 => zero,
        1 => terms[0],
        _ => arena.add(&terms),
    })
}

/// Leading term `(c, e)` of `f ≈ c·ω^e` as `ω → 0⁺` from the truncated
/// Laurent engine of [`series`](crate::calculus::series), log-extended with
/// `ln ω = logw` (SymPy's `logx`), charged against the work budget.
///
/// The engine tracks the exact precision of every intermediate series, so a
/// cancellation deeper than the truncation is never mistaken for a leading
/// term, and it knows the expansions of `Γ`, `ψ`, `ζ`, `ln Γ`, `Ei`, `Ci`,
/// `Chi`, `li`, `Si`, `Shi`, `K₀`, `W` at the singular points of their
/// arguments, which the differentiation of [`budgeted_series`] cannot
/// evaluate (`Γ(ω) − 1/ω`, `Ei(ω) − ln ω`, `W(ω)/ω`).  Its own
/// differentiation fallback takes no limits here.
fn leadterm_by_tseries(
    arena: &mut Arena,
    budget: &mut Budget,
    f: ExprId,
    w: ExprId,
    logw: ExprId,
) -> Result<(ExprId, ExprId), crate::base::errors::SymplexError> {
    let f = combine_exponentials(arena, f, w);
    let l = log_omega_symbol(arena);
    let f = variable_as_log_omega(arena, f, logw, l);
    for order in [4i64, 10] {
        budget.tick(SERIES_COST)?;
        budget.charge_size(arena, f)?;
        let ts = crate::calculus::series::expand_leading(arena, f, w, order, l)?;
        for e in ts.shift()..ts.known() {
            let c = ts.coefficient(arena, e);
            if arena.is_zero_structural(c) {
                continue;
            }
            let c = with_logw(arena, c, l, logw);
            let c = settled_coefficient(arena, c)?;
            if arena.is_zero_structural(c) || is_rational_zero(arena, c) {
                continue;
            }
            if crate::base::walk::contains(arena, c, w) || contains_singular_atom(arena, c) {
                break;
            }
            tracing::debug!(f = %arena.display(f), c = %arena.display(c), e, "gruntz::leadterm_by_tseries");
            return Ok((c, arena.int(e)));
        }
        if ts.undecided() {
            break; // a longer expansion meets the same undecided coefficient
        }
    }
    Err(crate::base::errors::SymplexError::ComputationFailed {
        operation: "gruntz::leadterm",
        reason: "the series expansion has no non-zero term within its precision".into(),
    })
}

/// `f` with the variable `x` written through `ln ω` when `logw = c·x` (the
/// usual `logw = −x` after a move up): `x = l/c` for the engine's symbol `l`
/// of `ln ω`.  The rewrite leaves `x` explicit beside `ω` (`e^{ψ(x)} − x`
/// at `∞` becomes `−ω⁻¹ + e^{−x + ψ(1/ω)}/ω`), and the engine, which
/// writes `ln ω` in `ψ(1/ω) = −ln ω − ω/2 + …` as `l`, did not see `−x − l
/// = 0` (SymPy passes `logx = logw` and gets it): the leading term came out
/// `e^{−x}·ω⁻²` and the limit `∞` (it is `−1/2`).
fn variable_as_log_omega(arena: &mut Arena, f: ExprId, logw: ExprId, l: ExprId) -> ExprId {
    let (c, s) = match arena.node(logw).clone() {
        ExprNode::Symbol(_) => (Q::from_integer(BigInt::from(1)), logw),
        ExprNode::Neg(a) if matches!(arena.node(a), ExprNode::Symbol(_)) => {
            (Q::from_integer(BigInt::from(-1)), a)
        }
        ExprNode::Mul(cs) if cs.len() == 2 => {
            match (arena.as_num(cs[0]).cloned(), arena.node(cs[1])) {
                (Some(c), ExprNode::Symbol(_)) if !c.is_zero() => (c, cs[1]),
                _ => return f,
            }
        }
        _ => return f,
    };
    if !crate::base::walk::contains(arena, f, s) {
        return f;
    }
    let inv = arena.intern_num(Q::from_integer(BigInt::from(1)) / c);
    let inv = arena.intern(ExprNode::Num(inv));
    let x_of_l = arena.mul(&[inv, l]);
    let g = crate::transforms::subs::subs(arena, f, s, x_of_l);
    crate::transforms::eval::eval(arena, g)
}

/// The symbol for `ln ω` in the expansions of the series engine (its
/// `log_var`, SymPy's `logx`), declared negative as `ln ω` is for
/// `ω → 0⁺`; [`with_logw`] puts Gruntz's `logw` (an expression in `x`,
/// `−x` after a move up) in its place in a coefficient.
///
/// The engine recognises `ln ω` in a coefficient by this symbol: `e^{m·l +
/// r} = ωᵐ·eʳ`, and no Taylor expansion of `f(u₀ + w)` about a `u₀` that
/// grows like `ln ω`.  Given the expression `logw` itself it saw `ln ω`
/// only where that expression stood literally, took `e^{−logw} = e^x` for a
/// constant coefficient of `ω⁰` (it is `ω⁻¹`), and `Γ(x + 1)/Γ(x) − x`
/// at `∞`, rewritten as `−x + e^{ln Γ(x + 1) − ln Γ(x)}`, came out `−∞`
/// (it is `0`).
const LOG_OMEGA: &str = "__gruntz_lnw";

fn log_omega_symbol(arena: &mut Arena) -> ExprId {
    let l = arena.symbol(LOG_OMEGA);
    if let ExprNode::Symbol(sid) = *arena.node(l)
        && arena.symbol_assumptions(sid) == crate::base::assumptions::Assumptions::default()
    {
        let mut a = crate::base::assumptions::Assumptions::default();
        a.assert_true(crate::base::assumptions::Props::NEGATIVE);
        a.forward_chain();
        arena.set_symbol_assumptions(sid, a);
    }
    l
}

/// `c` with `logw` for the symbol `l` of [`log_omega_symbol`].
fn with_logw(arena: &mut Arena, c: ExprId, l: ExprId, logw: ExprId) -> ExprId {
    if !crate::base::walk::contains(arena, c, l) {
        return c;
    }
    let r = crate::transforms::subs::subs(arena, c, l, logw);
    crate::transforms::eval::eval(arena, r)
}

/// Gruntz's last resort for the leading term `(c, e)` of `f ≈ c·ω^e`, once
/// [`leadterm`] has failed: the truncated Laurent engine with the
/// expansions a series makes beside cuts, at branch points and at infinity
/// ([`expand_leading_last_resort`](crate::calculus::series::expand_leading_last_resort):
/// `atan(1/ω) = π/2 − ω + …`, `ln Γ(−ω) = −ln ω − iπ + …`), and with
/// the fractional powers of `ω` ramified: `ω = u^q` for the least common
/// denominator `q` of their exponents, so that a Puiseux series in `ω` is
/// a Laurent series in `u` (`(e^{ω^{1/3}} − 1)/ω = u⁻² + u⁻¹/2 + …`,
/// leading exponent `−2/3`).  The engine tracks its precision, so a
/// cancellation deeper than the truncation is never taken for a leading
/// term.  Before, `k·(e^{k^{−1/3}} − 1)` and `k·(π/2 − atan k)` at `∞`
/// were refused ("leading coefficients cancel"; SymPy: `oo`, `1`).
///
/// A failure gives its work back to the budget (counting it in
/// [`Budget::resort`] instead), so a refusal here changes nothing else.
fn leadterm_last_resort(
    arena: &mut Arena,
    budget: &mut Budget,
    f: ExprId,
    w: ExprId,
    logw: ExprId,
) -> Result<(ExprId, ExprId), crate::base::errors::SymplexError> {
    let refused = || crate::base::errors::SymplexError::ComputationFailed {
        operation: "gruntz::leadterm",
        reason: "the last-resort series expansion found no leading term".into(),
    };
    if budget.resort.overrun() {
        return Err(refused());
    }
    let before = budget.work;
    let r = last_resort_series(arena, budget, f, w, logw);
    if let Err(e) = &r {
        tracing::debug!(f = %arena.display(f), error = %e, "gruntz::leadterm_last_resort failed");
        let spent = budget.work.attempted.saturating_sub(before.attempted);
        // Past its cap the resort count only refuses the next last resort.
        let _ = budget.resort.charge(spent);
        budget.work = before;
    }
    r.map_err(|_| refused())
}

/// The expansion of [`leadterm_last_resort`].
fn last_resort_series(
    arena: &mut Arena,
    budget: &mut Budget,
    f: ExprId,
    w: ExprId,
    logw: ExprId,
) -> Result<(ExprId, ExprId), crate::base::errors::SymplexError> {
    let f = combine_exponentials(arena, f, w);
    budget.charge_size(arena, f)?;
    let q = ramification_index(arena, f, w);
    let lw = log_omega_symbol(arena);
    let f = if q == 1 {
        variable_as_log_omega(arena, f, logw, lw)
    } else {
        f
    };
    let (g, var, log_var) = if q == 1 {
        (f, w, logw)
    } else {
        // `u > 0`, so that `(u^q)^{p/q} = u^p` (as `ω → 0⁺` is positive).
        let u = fresh_positive_dummy(arena, budget);
        let qe = arena.int(q);
        let uq = arena.pow(u, qe);
        let g = crate::transforms::subs::subs(arena, f, w, uq);
        let g = crate::transforms::eval::eval(arena, g);
        let lv = arena.div(logw, qe);
        let lv = crate::transforms::eval::eval(arena, lv);
        (g, u, lv)
    };
    // The orders of the first attempt (`ω⁴`, `ω¹⁰`), after a cheap one in
    // `u`; a long expansion costs more of the budget.
    let orders = if q == 1 {
        [4, 10, 10]
    } else {
        [8, 4 * q, 10 * q]
    };
    // (`l` stands for `ln var`, which is `log_var`.)
    let l = log_omega_symbol(arena);
    let mut tried = 0;
    for order in orders {
        let order = order.min(MAX_RAMIFIED_ORDER);
        if order <= tried {
            continue;
        }
        tried = order;
        budget.tick(SERIES_COST.saturating_mul(u32::try_from(order / 10).unwrap_or(1).max(1)))?;
        budget.charge_size(arena, g)?;
        // A short expansion may not reach a sub-expression's leading term
        // (`acos(1 − u¹⁰)`): the next order decides.
        let ts = crate::calculus::series::expand_leading_last_resort(arena, g, var, order, l);
        tracing::trace!(g = %arena.display(g), q, order, ok = ts.is_ok(), "gruntz::last_resort_series");
        let Ok(ts) = ts else {
            continue;
        };
        for e in ts.shift()..ts.known() {
            let c = ts.coefficient(arena, e);
            tracing::trace!(e, c = %arena.display(c), "gruntz::last_resort_series coefficient");
            if arena.is_zero_structural(c) {
                continue;
            }
            let c = with_logw(arena, c, l, log_var);
            let c = settled_coefficient(arena, c)?;
            if arena.is_zero_structural(c) || is_rational_zero(arena, c) {
                continue;
            }
            if crate::base::walk::contains(arena, c, var) || contains_singular_atom(arena, c) {
                break;
            }
            tracing::debug!(f = %arena.display(f), c = %arena.display(c), e, q, "gruntz::leadterm_last_resort");
            return Ok((c, arena.rational(e, q)));
        }
        if ts.undecided() {
            break; // a longer expansion meets the same undecided coefficient
        }
    }
    Err(crate::base::errors::SymplexError::ComputationFailed {
        operation: "gruntz::leadterm",
        reason: "the series expansion has no non-zero term within its precision".into(),
    })
}

/// The least common denominator of the fractional exponents of the powers
/// in `f` whose base depends on `w`, times 2 when the series of a function
/// of `f` at a branch point has half-integer powers of a `w`-dependent
/// argument (`acos(−1 + s) = π − √(2s) + …`, `acosh`, `asin`, `W` at
/// `−1/e`, which the engine expands only when `s` has an even valuation);
/// `1` when there are none, or when it exceeds [`MAX_RAMIFICATION`].
fn ramification_index(arena: &Arena, f: ExprId, w: ExprId) -> i64 {
    let mut q = 1i64;
    for id in crate::base::walk::post_order_ids(arena, f) {
        let d = match *arena.node(id) {
            ExprNode::Asin(a) | ExprNode::Acos(a) | ExprNode::Acosh(a) | ExprNode::LambertW(a)
                if crate::base::walk::contains(arena, a, w) =>
            {
                2
            }
            ExprNode::Pow(b, g) => {
                let Some(r) = arena.as_num(g) else {
                    continue;
                };
                if r.is_integer() || !crate::base::walk::contains(arena, b, w) {
                    continue;
                }
                let Some(d) = r.denom().to_i64() else {
                    return 1;
                };
                d
            }
            _ => continue,
        };
        q = num_integer::Integer::lcm(&q, &d);
        if q > MAX_RAMIFICATION {
            return 1;
        }
    }
    q
}

/// `e^a·e^b → e^{a+b}` in every product of `f` with two or more
/// `ω`-dependent exponential factors (an identity on all of ℂ).  The
/// rewrite of Gruntz's `Γ(z) = e^{ln Γ(z)}` leaves `Γ(x + 1)/Γ(x)` as
/// `e^{−ln Γ(1/ω)}·e^{ln Γ(1/ω + 1)}`, two essential singularities whose
/// product is `1/ω`: only the combined exponent has a series (Stirling's
/// terms cancel in it).  `limit(Γ(x + 1)/Γ(x) − x, x, ∞)` and
/// `limit(x·(Γ(x + 1/2)/(Γ(x)·√x) − 1), x, ∞)` were refused (SymPy: `0`,
/// `−1/8`).
fn combine_exponentials(arena: &mut Arena, f: ExprId, w: ExprId) -> ExprId {
    let post = crate::base::walk::post_order_ids(arena, f);
    let w_exp = |arena: &Arena, c: ExprId| matches!(arena.node(c), ExprNode::Exp(a) if crate::base::walk::contains(arena, *a, w));
    let several = |arena: &Arena, id: ExprId| match arena.node(id) {
        ExprNode::Mul(cs) => cs.iter().filter(|&&c| w_exp(arena, c)).count() >= 2,
        _ => false,
    };
    if !post.iter().any(|&id| several(arena, id)) {
        return f;
    }
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    for &id in &post {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = if several(arena, rebuilt) {
            let ExprNode::Mul(cs) = arena.node(rebuilt).clone() else {
                continue;
            };
            let mut args: SmallVec<[ExprId; 4]> = SmallVec::new();
            let mut rest: SmallVec<[ExprId; 6]> = SmallVec::new();
            for c in cs {
                match arena.node(c) {
                    ExprNode::Exp(a) if crate::base::walk::contains(arena, *a, w) => args.push(*a),
                    _ => rest.push(c),
                }
            }
            let sum = arena.add(&args);
            let sum = crate::transforms::eval::eval(arena, sum);
            rest.push(arena.exp(sum));
            arena.mul(&rest)
        } else {
            rebuilt
        };
        cache.insert(id, new);
    }
    cache.get(&f).copied().unwrap_or(f)
}

/// Is the coefficient `c` — a rational function of `ln ω` (and `x`) — zero
/// over a common denominator?  `1/x + (x − 1)/x − 1` is; taken for a
/// leading coefficient it made `x!·eˣ/xˣ` at `∞` an "indeterminate
/// `0·∞`".
fn is_rational_zero(arena: &mut Arena, c: ExprId) -> bool {
    if crate::base::walk::free_symbols(arena, c).is_empty()
        || crate::transforms::pattern::tree_size_capped(arena, c, 201) > 200
    {
        return false;
    }
    let t = crate::poly::polybridge::together(arena, c);
    let (num, _) = crate::poly::polybridge::as_numer_denom(arena, t);
    let num = crate::transforms::expand::expand(arena, num);
    let num = crate::transforms::eval::eval(arena, num);
    arena.is_zero_structural(num)
}

/// A coefficient of a leading-term search after the zero test
/// ([`settle_parametric`](crate::calculus::series::settle_parametric)):
/// `0` when proved 0, the value when proved nonzero (or with parameters),
/// an error when it is 0 to the precision reached and proved neither — it
/// can be neither skipped nor taken for the leading term.
fn settled_coefficient(
    arena: &mut Arena,
    c: ExprId,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    match crate::calculus::series::settle_parametric(arena, c) {
        ZeroTest::Undecided(v) => Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::leadterm",
            reason: format!(
                "cannot decide whether the coefficient {} is 0",
                arena.display(v)
            ),
        }),
        settled => Ok(settled.value(arena)),
    }
}

/// `Some(K)` if the truncated Laurent engine shows `f = O(ωᴷ)` with
/// `K > 0` — every coefficient below its precision `K` vanishes — although
/// no leading term can be named (`ln Γ(z + 1) − ln Γ(z) − ln z`, which is
/// identically zero).  That suffices where only the *limit* of `f` matters:
/// `f → 0` and `e^f → 1`.  (A leading term `(0, K)` would not do: a
/// product or quotient with `f` needs its true order.)
fn vanishes_to_positive_order(
    arena: &mut Arena,
    budget: &mut Budget,
    f: ExprId,
    w: ExprId,
    logw: ExprId,
) -> Option<i64> {
    budget.tick(SERIES_COST).ok()?;
    let f = combine_exponentials(arena, f, w);
    budget.charge_size(arena, f).ok()?;
    let l = log_omega_symbol(arena);
    let f = variable_as_log_omega(arena, f, logw, l);
    // Is a coefficient below the precision of `ts` non-zero?
    let nonzero_below_known = |arena: &mut Arena, ts: &crate::calculus::series::TSeries| {
        (ts.shift()..ts.known()).any(|e| {
            let c = ts.coefficient(arena, e);
            if arena.is_zero_structural(c) {
                return false;
            }
            let c = with_logw(arena, c, l, logw);
            // An undecided coefficient is not shown to vanish.
            match crate::calculus::series::settle_parametric(arena, c) {
                ZeroTest::Zero => false,
                ZeroTest::Nonzero(v) => !is_rational_zero(arena, v),
                ZeroTest::Undecided(_) => true,
            }
        })
    };
    // A short expansion first: the usual answer, a non-zero coefficient,
    // shows there at a fraction of the cost of ten terms (those of
    // `atanh(xˣ)` and `cos(xˣ)` near `x = π`, large constants in `π^π`,
    // took 3 s of `lim_{x→π⁺} (atanh(xˣ) − atanh(π^π))/(cos(xˣ) − cos(π^π))`).
    if let Ok(short) = crate::calculus::series::expand_leading(arena, f, w, 3, l) {
        if nonzero_below_known(arena, &short) {
            return None;
        }
        // Cut short by an undecided coefficient: ten terms meet it too.
        if short.undecided() {
            return (short.known() > 0).then_some(short.known());
        }
    }
    let ts = crate::calculus::series::expand_leading(arena, f, w, 10, l).ok()?;
    if ts.known() <= 0 || nonzero_below_known(arena, &ts) {
        return None;
    }
    Some(ts.known())
}

// ═══════════════════════════════════════════════════════════════════════════
// SubsSet — the core data structure
// ═══════════════════════════════════════════════════════════════════════════

/// Tracks MRV expressions, their dummy substitutions, and inter-MRV rewrites.
///
/// Mirrors SymPy's `SubsSet`. Maps `original_expr → dummy_variable`.
/// The `rewrites` dict maps `dummy → rewritten_form` for MRV elements
/// that contain other MRV elements as subexpressions.
///
/// Example: for MRV = {exp(x), exp(-x), exp(x - exp(-x))}
///   exprs = {exp(x): d1, exp(-x): d2, exp(x - exp(-x)): d3}
///   rewrites = {d3: exp(x - d2)}  ← d3 contains d2 as subexpression
#[derive(Clone, Debug)]
struct SubsSet {
    /// Maps: original MRV expression → dummy variable
    exprs: FxHashMap<ExprId, ExprId>,
    /// Maps: dummy variable → rewritten form (in terms of other dummies)
    rewrites: FxHashMap<ExprId, ExprId>,
}

impl SubsSet {
    fn new() -> Self {
        Self {
            exprs: FxHashMap::default(),
            rewrites: FxHashMap::default(),
        }
    }

    fn is_empty(&self) -> bool {
        self.exprs.is_empty()
    }

    /// One member of the set, standing for its growth class (the callers
    /// have already returned on an empty set; the `Err` names the
    /// invariant instead of unwrapping).
    ///
    /// The member is chosen deterministically, preferring one free of MRV
    /// dummies and then the smallest (SymPy sorts by height): taking the
    /// first key of the hash map made the choice depend on the order in
    /// which expression nodes had been created, and a key still holding a
    /// dummy (`exp(x − ω₂)`) led `compare` to a leading coefficient with
    /// that dummy — `exp(x − e⁻ˣ) − eˣ` at `∞` failed or succeeded
    /// depending on what had been computed before in the same arena.
    fn representative_in(
        &self,
        arena: &Arena,
        x: ExprId,
    ) -> Result<ExprId, crate::base::errors::SymplexError> {
        self.exprs
            .iter()
            .map(|(&k, _)| {
                let dirty = contains_foreign_dummy(arena, k, x);
                let size = crate::transforms::pattern::tree_size_capped(arena, k, 10_000);
                (dirty, size, k)
            })
            .min_by(|a, b| {
                (a.0, a.1)
                    .cmp(&(b.0, b.1))
                    .then_with(|| arena.canonical_cmp(a.2, b.2))
            })
            .map(|(_, _, k)| k)
            .ok_or_else(|| crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz",
                reason: "internal: MRV set unexpectedly empty".into(),
            })
    }

    fn len(&self) -> usize {
        self.exprs.len()
    }

    fn contains_key(&self, e: &ExprId) -> bool {
        self.exprs.contains_key(e)
    }

    /// Get or create a dummy for the given expression.
    /// Like SymPy's `SubsSet.__getitem__`: auto-creates on first access.
    fn get_or_create_dummy(
        &mut self,
        expr: ExprId,
        arena: &mut Arena,
        budget: &mut Budget,
    ) -> ExprId {
        if let Some(&d) = self.exprs.get(&expr) {
            return d;
        }
        let d = fresh_dummy(arena, budget);
        self.exprs.insert(expr, d);
        d
    }

    /// Apply all substitutions (expr → dummy) to an expression.
    ///
    /// Larger expressions are replaced first: with `{exp(x − e⁻ˣ): ω₁,
    /// e⁻ˣ: ω₂}`, replacing `e⁻ˣ` first left `exp(x − ω₂)` behind, a
    /// stale dummy that ended as a "bounded oscillation" in a leading
    /// coefficient.  (The hash-map order made this depend on the order in
    /// which nodes had been created.)
    fn do_subs(&self, arena: &mut Arena, e: ExprId) -> ExprId {
        let mut entries: Vec<(usize, ExprId, ExprId)> = self
            .exprs
            .iter()
            .map(|(&orig, &dummy)| {
                let size = crate::transforms::pattern::tree_size_capped(arena, orig, 10_000);
                (size, orig, dummy)
            })
            .collect();
        entries.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| arena.canonical_cmp(a.1, b.1)));
        let mut result = e;
        for (_, orig, dummy) in entries {
            result = crate::transforms::subs::subs(arena, result, orig, dummy);
        }
        result
    }

    /// Undo all substitutions (dummy → expr) in an expression.
    ///
    /// Used when this set loses an MRV comparison: its elements are no
    /// longer "most rapidly varying" and must reappear as ordinary
    /// sub-expressions of `x` rather than as opaque dummies.
    fn undo_subs(&self, arena: &mut Arena, e: ExprId) -> ExprId {
        let mut result = e;
        for (&orig, &dummy) in &self.exprs {
            result = crate::transforms::subs::subs(arena, result, dummy, orig);
        }
        result
    }

    /// Union two SubsSets (merge b into self).
    fn union_with(&mut self, other: &SubsSet) {
        for (&expr, &dummy) in &other.exprs {
            self.exprs.entry(expr).or_insert(dummy);
        }
        for (&dummy, &rewrite) in &other.rewrites {
            self.rewrites.entry(dummy).or_insert(rewrite);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Growth comparison
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GrowthOrder {
    Less,
    Equal,
    Greater,
}

/// Compare the growth rates of `a` and `b` as `x → ∞`.
fn compare(
    arena: &mut Arena,
    a: ExprId,
    b: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<GrowthOrder, crate::base::errors::SymplexError> {
    tracing::debug!(depth, "gruntz::compare: comparing growth rates");
    budget.tick(1)?;

    let la = log_of(arena, a);
    let lb = log_of(arena, b);
    let ratio = arena.div(la, lb);

    tracing::trace!("gruntz::compare: computing limitinf of log ratio");
    let c = limitinf(arena, ratio, x, depth + 1, budget)?;

    if arena.is_zero_structural(c) {
        tracing::debug!("gruntz::compare → Less (a grows slower)");
        Ok(GrowthOrder::Less)
    } else if is_infinite(arena, c) {
        tracing::debug!("gruntz::compare → Greater (a grows faster)");
        Ok(GrowthOrder::Greater)
    } else {
        tracing::debug!("gruntz::compare → Equal (same comparability class)");
        Ok(GrowthOrder::Equal)
    }
}

/// Compute `log(|e|)` — if `e` is `exp(f)`, return `f` directly.
fn log_of(arena: &mut Arena, e: ExprId) -> ExprId {
    match arena.node(e).clone() {
        ExprNode::Exp(inner) => {
            tracing::trace!("gruntz::log_of: exp(f) → f");
            inner
        }
        _ => {
            tracing::trace!("gruntz::log_of: general → ln(|e|)");
            let abs_e = arena.abs(e);
            arena.ln(abs_e)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sign determination
// ═══════════════════════════════════════════════════════════════════════════

/// Determine the *eventual* sign of `e` for large `x`: +1, -1, or 0.
///
/// Mirrors SymPy's `gruntz.sign`: `e ≈ c₀·ω^{e₀}` with `ω → 0⁺`, so the
/// sign of `e` for large `x` is the sign of `c₀` (recursively). Unlike a
/// limit-based test this is correct for expressions that tend to zero
/// (`1/x` has eventual sign `+1`, `−1/x` has `−1`).
fn sign_at_inf(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<i32, crate::base::errors::SymplexError> {
    if depth > MAX_DEPTH {
        tracing::warn!("gruntz::sign_at_inf: max depth exceeded");
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::sign_at_inf",
            reason: "maximum recursion depth exceeded".into(),
        });
    }
    budget.tick(1)?;

    let e = crate::transforms::eval::eval(arena, e);

    if !crate::base::walk::contains(arena, e, x) {
        tracing::trace!("gruntz::sign_at_inf: constant expression");
        return sign_of_constant(arena, e);
    }

    // x itself → +∞ as x → ∞, so sign is +1.
    // This is the critical base case that prevents infinite recursion
    // in the limitinf → mrv_leadterm → rewrite → sign_at_inf chain.
    if e == x {
        tracing::trace!("gruntz::sign_at_inf: e == x → +1 (x → +∞)");
        return Ok(1);
    }

    // A polynomial in x: the sign of its leading coefficient.  (Through
    // `mrv_leadterm` each `x − 1` cost several levels of the recursion
    // depth, which the nested comparisons of `Γ(x + 1)/Γ(x)` ran out of.)
    if matches!(arena.node(e), ExprNode::Add(_))
        && let Some(coeffs) = crate::poly::polybridge::poly_coefficients(arena, e, x)
        && let Some(&lead) = coeffs.iter().rev().find(|&&c| !arena.is_zero_structural(c))
        && !crate::base::walk::contains(arena, lead, x)
        && let Ok(s) = sign_of_constant(arena, lead)
        && s != 0
    {
        return Ok(s);
    }

    match arena.node(e).clone() {
        // Pow(x, c) with a real constant exponent: positive for large x > 0
        ExprNode::Pow(base, exp) if base == x && arena.as_num(exp).is_some() => {
            tracing::trace!("gruntz::sign_at_inf: x^c → +1");
            return Ok(1);
        }
        ExprNode::Exp(_) => {
            tracing::trace!("gruntz::sign_at_inf: exp() is always positive → +1");
            return Ok(1);
        }
        ExprNode::Neg(inner) => {
            return sign_at_inf(arena, inner, x, depth + 1, budget).map(|s| -s);
        }
        ExprNode::Mul(children) => {
            let mut result = 1i32;
            for c in children {
                let s = sign_at_inf(arena, c, x, depth + 1, budget)?;
                if s == 0 {
                    return Ok(0);
                }
                result *= s;
            }
            return Ok(result);
        }
        _ => {}
    }

    tracing::trace!("gruntz::sign_at_inf: computing leading term to determine sign");
    let (c0, e0) = mrv_leadterm(arena, e, x, depth + 1, budget, false)?;
    if contains_foreign_dummy(arena, c0, x) {
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::sign_at_inf",
            reason: "leading coefficient is not a genuine constant".into(),
        });
    }
    // `c₀·ω^{e₀}` has the sign of `c₀` only for a real `e₀`.
    if exponent_turns(arena, e0, x, depth, budget) {
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::sign_at_inf",
            reason: "the leading term has a turning phase".into(),
        });
    }
    if arena.is_zero_structural(c0) {
        return Ok(0);
    }
    let result = sign_at_inf(arena, c0, x, depth + 1, budget);
    tracing::debug!(sign = ?result, "gruntz::sign_at_inf result");
    result
}

/// The sum `c` (with `x` in it) over a common denominator when that form
/// is smaller (`a/D + b/D = (a + b)/D`, and `1` when `a + b = D`);
/// otherwise `c`.
fn common_denominator_if_smaller(arena: &mut Arena, c: ExprId, x: ExprId) -> ExprId {
    if !matches!(arena.node(c), ExprNode::Add(_)) || !crate::base::walk::contains(arena, c, x) {
        return c;
    }
    let cap = 200;
    let size =
        |arena: &Arena, e: ExprId| crate::transforms::pattern::tree_size_capped(arena, e, cap + 1);
    let before = size(arena, c);
    if before > cap {
        return c;
    }
    let t = crate::poly::polybridge::together(arena, c);
    let t = crate::transforms::eval::eval(arena, t);
    if size(arena, t) < before { t } else { c }
}

/// Determine the sign of a constant expression (no free variable x).
///
/// Consults symbol assumptions (`a > 0`) through
/// [`limit::const_sign`](crate::calculus::limit::const_sign).
fn sign_of_constant(
    arena: &mut Arena,
    e: ExprId,
) -> Result<i32, crate::base::errors::SymplexError> {
    let sign = crate::calculus::limit::const_sign(arena, e).or_else(|| {
        // A real constant written with `i` (the leading coefficient of an
        // imaginary part `(f − f̄)/(2i)`: `(a + bi)·i + (−a + bi)·i = −2b`).
        if !crate::base::walk::contains(arena, e, arena.i_unit()) {
            return None;
        }
        let x = crate::transforms::expand::expand(arena, e);
        let x = crate::transforms::eval::eval(arena, x);
        (x != e && !crate::base::walk::contains(arena, x, arena.i_unit()))
            .then(|| crate::calculus::limit::const_sign(arena, x))
            .flatten()
    });
    match sign {
        Some(s) => Ok(s),
        None => Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::sign_of_constant",
            reason: format!("cannot determine the sign of {}", arena.display(e)),
        }),
    }
}

/// `true` if `e` contains an internal dummy symbol other than `x` itself.
///
/// Used to detect leading coefficients that still depend on the Gruntz
/// `ω` variable — a sign that the leading-term extraction gave up.
fn contains_foreign_dummy(arena: &Arena, e: ExprId, x: ExprId) -> bool {
    let mut stack = vec![e];
    let mut visited = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        if id != x
            && let ExprNode::Symbol(sid) = arena.node(id)
        {
            let name = arena.symbol_name(*sid);
            if name.starts_with("__gw") || name.starts_with("__lim") {
                return true;
            }
        }
        arena.node(id).for_each_child(|c| stack.push(c));
    }
    false
}

// ═══════════════════════════════════════════════════════════════════════════
// MRV (Most Rapidly Varying) set computation
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the MRV set of `e` with respect to `x`.
///
/// Returns `(SubsSet, rewritten_expr)` where the rewritten expression has
/// all MRV elements replaced by their dummy variables.
fn mrv(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<(SubsSet, ExprId), crate::base::errors::SymplexError> {
    if depth > MAX_DEPTH {
        tracing::warn!("gruntz::mrv: max depth exceeded");
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::mrv",
            reason: "maximum recursion depth exceeded".into(),
        });
    }
    budget.tick(1)?;

    // Simplify first: SymPy's `powsimp(e, deep=True, combine='exp')`.
    // Combining `e^a·e^b → e^{a+b}` matters: `rewrite` expands, which splits
    // `exp(ln Γ(x+1) − x ln x)` back into `exp(ln Γ(x+1))·exp(−x ln x)`,
    // the two members of one class it had just combined, and `x!/xˣ` looped
    // (the leading coefficient was the expression itself) until the
    // recursion depth ran out.
    let e = crate::transforms::eval::eval(arena, e);
    let e = if count_exp_nodes(arena, e) >= 2 {
        let p = crate::simplify::powsimp::powsimp(arena, e);
        crate::transforms::eval::eval(arena, p)
    } else {
        e
    };

    // Base case: e doesn't depend on x
    if !crate::base::walk::contains(arena, e, x) {
        tracing::trace!("gruntz::mrv: constant → empty MRV set");
        return Ok((SubsSet::new(), e));
    }

    // Base case: e IS x
    if e == x {
        tracing::trace!("gruntz::mrv: e == x → MRV = {{x}}");
        let mut s = SubsSet::new();
        let d = s.get_or_create_dummy(x, arena, budget);
        return Ok((s, d));
    }

    let node = arena.node(e).clone();

    match node {
        // ── Add or Mul: merge children's MRV sets ──
        ExprNode::Add(ref children) | ExprNode::Mul(ref children) => {
            let is_add = matches!(node, ExprNode::Add(_));
            let children_vec: SmallVec<[ExprId; 6]> = children.clone();
            tracing::trace!(
                n = children_vec.len(),
                kind = if is_add { "Add" } else { "Mul" },
                "gruntz::mrv: processing n-ary node"
            );
            mrv_nary(arena, e, &children_vec, x, depth, is_add, budget)
        }

        // ── Pow: handle x^n vs x^f(x) ──
        ExprNode::Pow(base, exp) => {
            tracing::trace!("gruntz::mrv: Pow node");
            if crate::base::walk::contains(arena, exp, x) {
                // x^f(x) → rewrite as exp(f(x)*ln(x))
                tracing::debug!("gruntz::mrv: Pow with x-dependent exponent → exp(exp*ln(base))");
                let ln_base = arena.ln(base);
                let product = arena.mul(&[exp, ln_base]);
                let as_exp = arena.exp(product);
                mrv(arena, as_exp, x, depth + 1, budget)
            } else {
                // x^const: MRV is just MRV of base
                let (s, rw_base) = mrv(arena, base, x, depth + 1, budget)?;
                let rebuilt = arena.pow(rw_base, exp);
                Ok((s, rebuilt))
            }
        }

        // ── Exp: the critical case ──
        ExprNode::Exp(arg) => {
            tracing::debug!("gruntz::mrv: Exp node — checking if exponent → ±∞");

            // SymPy: if exp(log(...)), simplify to avoid non-termination
            if let ExprNode::Ln(inner) = arena.node(arg).clone() {
                tracing::trace!("gruntz::mrv: exp(ln(f)) → mrv(f)");
                return mrv(arena, inner, x, depth + 1, budget);
            }

            // Check if the exponent goes to ±∞
            let li = limitinf(arena, arg, x, depth + 1, budget)?;
            let li_is_inf = is_infinite(arena, li);

            // `exp(g)` with `g ~ c·ln x` is a power of `x`, not a new class:
            // `exp(g) = x^c·exp(g − c·ln x)` exactly (`x > 0`), the second
            // factor's exponent finite.  Taken for exponents with `ln Γ`
            // (from `Γ` at `∞`, rewritten as `exp(ln Γ)`): as a member of the
            // class of `x` the rewrite left dummies of `x` inside it, and
            // `Γ(x + 1)/(x·Γ(x))` failed.
            if li_is_inf && contains_log_gamma(arena, arg) {
                let ln_x = arena.ln(x);
                let ratio = arena.div(arg, ln_x);
                if let Ok(c) = limitinf(arena, ratio, x, depth + 1, budget)
                    && arena.as_num(c).is_some_and(|r| !r.is_zero())
                {
                    let c_ln_x = arena.mul(&[c, ln_x]);
                    let rest = arena.sub(arg, c_ln_x);
                    let rest = crate::transforms::eval::eval(arena, rest);
                    if let Ok(lr) = limitinf(arena, rest, x, depth + 1, budget)
                        && !is_infinite(arena, lr)
                        && !contains_singular_atom(arena, lr)
                    {
                        let xc = arena.pow(x, c);
                        let er = arena.exp(rest);
                        let split = arena.mul(&[xc, er]);
                        let evaluated = crate::transforms::eval::eval(arena, split);
                        if crate::base::walk::contains(arena, evaluated, er)
                            || !crate::base::walk::contains(arena, evaluated, e)
                        {
                            return mrv(arena, split, x, depth + 1, budget);
                        }
                        // Evaluation folded `e^{−c·ln x + …}` back into
                        // `x^{−c}·e^{…}` (`exp(k·ln z) = z^k` for a
                        // symbolic `z`), `e` itself again, and `mrv` of
                        // the product would return here until the depth
                        // ran out.  The two factors' sets instead, merged
                        // as for a product: `x^c` in the class of `x`,
                        // `e^{rest}` (`rest` finite) in that of `rest`.
                        let (s1, r1) = mrv(arena, x, x, depth + 1, budget)?;
                        let (s2, r2) = mrv(arena, rest, x, depth + 1, budget)?;
                        let r1 = arena.pow(r1, c);
                        let r2 = arena.exp(r2);
                        let (s, a, b) = mrv_max1(arena, &s1, r1, &s2, r2, x, depth, budget)?;
                        return Ok((s, arena.mul(&[a, b])));
                    }
                }
            }

            // The representative `ω = e^{−g}` must be positive (`ω → 0⁺`): a
            // constant non-real term of the exponent made it complex, and
            // `e^{x − i}·|√x − 3i|` had the leading coefficient 1 (its limit
            // was `oo`; it is `e^{−i}·oo`).  Constant terms come out as
            // `e^a·(cos b + i·sin b)` when the varying part is real.  A
            // non-real constant need not be written with `i`: up to 0.49
            // `(1 + z)·e^{−z}` (`Γ(2, z)`), `z = (−3)^{−1/3} − e·x`, kept
            // `(−1)^{2/3}` in `ω` and its limit at `∞` was `−∞` (it is
            // `−e^{i·3^{1/6}/2}·∞`), and `e^{x + asin 2}` tended to `∞`.
            let split = if li_is_inf {
                exponent_phase_split(arena, arg, x, depth, budget)?
            } else {
                None
            };
            if let Some((real, im)) = split {
                let e_real = arena.exp(real);
                let cos = arena.cos(im);
                let sin = arena.sin(im);
                let i_sin = arena.mul(&[arena.i_unit(), sin]);
                let phase = arena.add(&[cos, i_sin]);
                let split = arena.mul(&[e_real, phase]);
                return mrv(arena, split, x, depth + 1, budget);
            }

            if li_is_inf {
                tracing::debug!("gruntz::mrv: exp(arg) with arg → ∞ — new comparability class");
                // exp(arg) creates a new comparability class
                let mut s1 = SubsSet::new();
                let e1 = s1.get_or_create_dummy(e, arena, budget);

                // Also compute MRV of the exponent
                let (s2, e2) = mrv(arena, arg, x, depth + 1, budget)?;

                // Record the rewrite: dummy_for_exp(arg) = exp(rewritten_arg)
                let exp_e2 = arena.exp(e2);

                // Merge using mrv_max3 logic
                mrv_max3(arena, s1, e1, s2, exp_e2, x, depth, budget)
            } else {
                tracing::debug!("gruntz::mrv: exp(arg) with arg → finite — same class as arg");
                let (s, rw_arg) = mrv(arena, arg, x, depth + 1, budget)?;
                let rebuilt = arena.exp(rw_arg);
                Ok((s, rebuilt))
            }
        }

        // ── Ln: always in a lower comparability class ──
        ExprNode::Ln(inner) => {
            tracing::trace!("gruntz::mrv: Ln node — recurse into argument");
            let (s, rw) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.ln(rw)))
        }

        // ── Neg ──
        ExprNode::Neg(inner) => {
            let (s, rw) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.neg(rw)))
        }

        // ── Unary functions: recurse into argument ──
        ExprNode::Sin(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.sin(r)))
        }
        ExprNode::Cos(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.cos(r)))
        }
        ExprNode::Tan(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.tan(r)))
        }
        ExprNode::Asin(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.asin(r)))
        }
        ExprNode::Acos(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.acos(r)))
        }
        ExprNode::Atan(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.atan(r)))
        }
        ExprNode::Sinh(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.sinh(r)))
        }
        ExprNode::Cosh(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.cosh(r)))
        }
        ExprNode::Tanh(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.tanh(r)))
        }
        ExprNode::Abs(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.abs(r)))
        }
        ExprNode::Sign(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.sign(r)))
        }
        ExprNode::Asinh(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.asinh(r)))
        }
        ExprNode::Acosh(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.acosh(r)))
        }
        ExprNode::Atanh(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.atanh(r)))
        }
        // Special functions: the MRV set is that of the argument (the
        // function itself is handled by `unary_leadterm`, which knows the
        // asymptotic behaviour or rejects it). Treating `W(exp(x))` as an
        // opaque function of `x` would lose the inner `exp(x)` class.
        ExprNode::Erf(inner)
        | ExprNode::Erfc(inner)
        | ExprNode::Gamma(inner)
        | ExprNode::LogGamma(inner)
        | ExprNode::Digamma(inner)
        | ExprNode::LambertW(inner)
        | ExprNode::Factorial(inner)
        | ExprNode::Heaviside(inner) => {
            check_growth_known(arena, &node, x, depth, budget)?;
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, apply_unary(arena, &node, r)))
        }
        ExprNode::Internal(f, ref args) if args.len() == 1 => {
            let (s, r) = mrv(arena, args[0], x, depth + 1, budget)?;
            Ok((s, arena.func(f.into(), &[r])?))
        }
        // `Iν`, `Kν`, `Γ(s, ·)` of the asymptotic rewrite: a constant parameter.
        ExprNode::Internal(f, ref args)
            if args.len() == 2 && !crate::base::walk::contains(arena, args[0], x) =>
        {
            let (s, r) = mrv(arena, args[1], x, depth + 1, budget)?;
            Ok((s, arena.func(f.into(), &[args[0], r])?))
        }

        // ── Fallback: treat as containing x somewhere ──
        _ => {
            check_growth_known(arena, &node, x, depth, budget)?;
            tracing::trace!("gruntz::mrv: fallback — treating expression as atomic with x");
            let mut s = SubsSet::new();
            let d = s.get_or_create_dummy(x, arena, budget);
            // Substitute x → d in the whole expression
            let rw = crate::transforms::subs::subs(arena, e, x, d);
            Ok((s, rw))
        }
    }
}

/// Refuse a function application whose growth class is unknown.
///
/// The MRV set places a function it does not analyse in the class of `x`
/// and, when a faster class dominates, leaves it in the leading
/// coefficient as if it were of lower order.  That is sound only if the
/// function grows (and approaches its limit) no faster than a power of
/// `x`: `Chi(x)·x·e⁻ˣ` was `0` (it is `1/2`: `Chi x ~ eˣ/(2x)`), and so
/// were `Shi(x)·x·e⁻ˣ`, `I₀(x)·√x·e⁻ˣ` (`1/√(2π)`),
/// `erfi(x)·x·e^{−x²}` (`1/√π`); `Γ(x)/eˣ` was `0` (it is `∞`) before
/// `Γ` at `∞` was rewritten.  So every argument must tend to a finite
/// value (the function is then analytic, or has a pole or a logarithmic
/// singularity there — power-like), or the function must be one known to be
/// power-like at `±∞` (`Si`, `Ci`, `Jₙ`, `Yₙ`, Fresnel integrals, and at
/// `+∞` `ln Γ`, `ψ`, `ψ⁽ᵐ⁾`, `W`); otherwise the limit is refused.
/// (`erf`, `erfc`, `Ei`, `li`, `Γ` at `±∞` reach here only if the
/// tractable rewrite could not decide their argument's limit.)
fn check_growth_known(
    arena: &mut Arena,
    node: &ExprNode,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<(), crate::base::errors::SymplexError> {
    use crate::base::libfn::LibFn;
    let function_like = node.func_head().is_some_and(growth_checked)
        || matches!(
            node,
            ExprNode::Apply(..)
            // An unevaluated sum, product or integral with a bound tending
            // to infinity grows as its terms do, which the algorithm does
            // not see: `2^{−n}·Σ_{k=1}^{n} 2^k/k` was `0` at `∞` (it is `2`),
            // the sum taken for a coefficient of lower class.
            | ExprNode::Sum(..)
            | ExprNode::Product_(..)
            | ExprNode::Integral(..)
            | ExprNode::DefiniteIntegral(..)
            | ExprNode::Derivative(..)
            | ExprNode::Limit(..)
        );
    if !function_like {
        return Ok(());
    }
    let tame_at = |arena: &Arena, plus: bool| -> bool {
        match node {
            ExprNode::Si(_) | ExprNode::Ci(_) => true,
            ExprNode::LogGamma(_) | ExprNode::Digamma(_) | ExprNode::LambertW(_) => plus,
            ExprNode::Polygamma(m, _) => plus && !crate::base::walk::contains(arena, *m, x),
            ExprNode::Lib(f, _) => matches!(
                f,
                LibFn::BesselJ | LibFn::BesselY | LibFn::FresnelS | LibFn::FresnelC
            ),
            _ => false,
        }
    };
    let mut children: SmallVec<[ExprId; 4]> = SmallVec::new();
    node.for_each_child(|c| children.push(c));
    for c in children {
        if !crate::base::walk::contains(arena, c, x) {
            continue;
        }
        let unknown = || crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::mrv",
            reason: "a function of unknown growth at the limit of its argument".into(),
        };
        match limitinf(arena, c, x, depth + 1, budget) {
            Ok(l) if !is_infinite(arena, l) && !contains_singular_atom(arena, l) => {}
            Ok(l) if l == arena.infinity() || l == arena.neg_infinity() => {
                // The Fresnel integrals are power-like along the reals only:
                // `sin(πu²/2)` grows like `e^{π·k·|Im c|·x}` for `u = c + k·x`
                // (`Si`, `Ci`, `Jₙ`, `Yₙ` stay bounded on such a line), and
                // `eˣ·fresnelc(√5·i − 2x)` at `−∞` was `0` (it has no limit).
                let fresnel = matches!(node, ExprNode::Lib(LibFn::FresnelS | LibFn::FresnelC, _));
                if !tame_at(arena, l == arena.infinity())
                    || (fresnel && !eventually_real(arena, c, x, depth + 1, budget))
                {
                    return Err(unknown());
                }
            }
            Err(e) if is_budget_error(&e) => return Err(e),
            _ => return Err(unknown()),
        }
    }
    Ok(())
}

/// The functions whose growth [`check_growth_known`] checks: the special
/// functions but `C(n, k)`, and every library and internal function.
fn growth_checked(f: Func) -> bool {
    f.family() != Family::Node || (f.is_special() && f != Func::Binomial)
}

/// MRV for n-ary (Add/Mul) nodes.
///
/// Processes children pairwise: `mrv_max1(mrv(a), mrv(b), ...)`.
fn mrv_nary(
    arena: &mut Arena,
    _original: ExprId,
    children: &[ExprId],
    x: ExprId,
    depth: usize,
    is_add: bool,
    budget: &mut Budget,
) -> Result<(SubsSet, ExprId), crate::base::errors::SymplexError> {
    if children.is_empty() {
        return Ok((
            SubsSet::new(),
            if is_add { arena.zero() } else { arena.one() },
        ));
    }

    // Split into x-independent and x-dependent parts
    let mut indep: SmallVec<[ExprId; 6]> = SmallVec::new();
    let mut dep: SmallVec<[ExprId; 6]> = SmallVec::new();

    for &child in children {
        if crate::base::walk::contains(arena, child, x) {
            dep.push(child);
        } else {
            indep.push(child);
        }
    }

    if dep.is_empty() {
        return Ok((
            SubsSet::new(),
            if is_add {
                arena.add(children)
            } else {
                arena.mul(children)
            },
        ));
    }

    // Process dependent children pairwise
    let (mut combined_set, mut rw_first) = mrv(arena, dep[0], x, depth + 1, budget)?;

    for &child in &dep[1..] {
        let (child_set, rw_child) = mrv(arena, child, x, depth + 1, budget)?;

        // Merge the two MRV sets
        let (merged, rw_a, rw_b) = mrv_max1(
            arena,
            &combined_set,
            rw_first,
            &child_set,
            rw_child,
            x,
            depth,
            budget,
        )?;
        combined_set = merged;

        // Rebuild the pair using the new rewritten forms
        rw_first = if is_add {
            arena.add(&[rw_a, rw_b])
        } else {
            arena.mul(&[rw_a, rw_b])
        };
    }

    // Re-attach independent parts
    if !indep.is_empty() {
        if is_add {
            indep.push(rw_first);
            rw_first = arena.add(&indep);
        } else {
            indep.push(rw_first);
            rw_first = arena.mul(&indep);
        }
    }

    Ok((combined_set, rw_first))
}

/// Merge two MRV sets, keeping the faster-growing one.
///
/// Returns `(merged_set, rewritten_e1, rewritten_e2)`.
/// When one set dominates, the other's expressions are rewritten using
/// the dominating set's dummies.
#[allow(clippy::too_many_arguments)]
fn mrv_max1(
    arena: &mut Arena,
    s1: &SubsSet,
    e1: ExprId,
    s2: &SubsSet,
    e2: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<(SubsSet, ExprId, ExprId), crate::base::errors::SymplexError> {
    if s1.is_empty() {
        return Ok((s2.clone(), e1, e2));
    }
    if s2.is_empty() {
        return Ok((s1.clone(), e1, e2));
    }

    let a_rep = s1.representative_in(arena, x)?;
    let b_rep = s2.representative_in(arena, x)?;

    // Same representative — union and unify dummies so e2 uses s1's dummies
    if a_rep == b_rep {
        let (merged, rw_e2) = merge_unified(arena, s1, s2, e2);
        return Ok((merged, e1, rw_e2));
    }

    tracing::debug!("gruntz::mrv_max1: comparing MRV representatives");
    match compare(arena, a_rep, b_rep, x, depth + 1, budget)? {
        GrowthOrder::Greater => {
            tracing::debug!(
                "gruntz::mrv_max1: s1 grows faster — keeping s1, rewriting e2 with s1 dummies"
            );
            // s2's elements are no longer MRV: restore them, then apply s1.
            let restored = s2.undo_subs(arena, e2);
            let rw_e2 = s1.do_subs(arena, restored);
            Ok((s1.clone(), e1, rw_e2))
        }
        GrowthOrder::Less => {
            tracing::debug!(
                "gruntz::mrv_max1: s2 grows faster — keeping s2, rewriting e1 with s2 dummies"
            );
            let restored = s1.undo_subs(arena, e1);
            let rw_e1 = s2.do_subs(arena, restored);
            Ok((s2.clone(), rw_e1, e2))
        }
        GrowthOrder::Equal => {
            tracing::debug!("gruntz::mrv_max1: same class — merging both sets");
            let (merged, rw_e2) = merge_unified(arena, s1, s2, e2);
            Ok((merged, e1, rw_e2))
        }
    }
}

/// The union of two sets of one class, with `s2`'s dummy for a key both
/// hold renamed to `s1`'s — in `e2` and in `s2`'s recorded rewrites.  The
/// rewrites kept the old dummy before: the rewrite of `exp(ln Γ(x + 1) −
/// ln Γ(x))` still named a dummy for `x` that no longer belonged to the
/// set, and it survived into the leading coefficient (`Γ(x + 1)/(x·Γ(x))`
/// at `∞` failed as a "bounded oscillation").
fn merge_unified(arena: &mut Arena, s1: &SubsSet, s2: &SubsSet, e2: ExprId) -> (SubsSet, ExprId) {
    let mut renames: Vec<(ExprId, ExprId)> = Vec::new();
    for (&expr, &d1_dummy) in &s1.exprs {
        if let Some(&d2_dummy) = s2.exprs.get(&expr)
            && d1_dummy != d2_dummy
        {
            tracing::trace!("gruntz::mrv_max1: unifying dummy for shared key");
            renames.push((d2_dummy, d1_dummy));
        }
    }
    let mut rw_e2 = e2;
    let mut s2 = s2.clone();
    for &(d2, d1) in &renames {
        rw_e2 = crate::transforms::subs::subs(arena, rw_e2, d2, d1);
        let dummies: Vec<ExprId> = s2.rewrites.keys().copied().collect();
        for d in dummies {
            if let Some(&r) = s2.rewrites.get(&d) {
                let r = crate::transforms::subs::subs(arena, r, d2, d1);
                s2.rewrites.insert(d, r);
            }
        }
    }
    let mut merged = s1.clone();
    merged.union_with(&s2);
    (merged, rw_e2)
}

/// Three-way MRV merge for the Exp case.
///
/// s1 contains exp(arg) itself, s2 contains MRV of arg.
/// Returns the merged set with the correct rewrite tracked.
#[allow(clippy::too_many_arguments)]
fn mrv_max3(
    arena: &mut Arena,
    mut s1: SubsSet, // {exp(arg): d1}
    e1: ExprId,      // d1
    s2: SubsSet,     // MRV of arg
    exp_e2: ExprId,  // exp(rewritten_arg)
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<(SubsSet, ExprId), crate::base::errors::SymplexError> {
    if s2.is_empty() {
        return Ok((s1, e1));
    }

    let a_rep = s1.representative_in(arena, x)?;
    let b_rep = s2.representative_in(arena, x)?;

    tracing::debug!("gruntz::mrv_max3: comparing exp vs arg MRV");
    match compare(arena, a_rep, b_rep, x, depth + 1, budget)? {
        GrowthOrder::Greater => {
            tracing::debug!("gruntz::mrv_max3: exp dominates — keeping exp as MRV");
            Ok((s1, e1))
        }
        GrowthOrder::Less => {
            tracing::debug!("gruntz::mrv_max3: arg's MRV dominates — using that");
            Ok((s2, exp_e2))
        }
        GrowthOrder::Equal => {
            tracing::debug!("gruntz::mrv_max3: same class — merging, recording rewrite");
            // Record that d1 (the dummy for the exp) rewrites to exp(rewritten_arg)
            s1.rewrites.insert(e1, exp_e2);
            s1.union_with(&s2);
            Ok((s1, e1))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// moveup: replace x → exp(x) in a SubsSet
// ═══════════════════════════════════════════════════════════════════════════

/// Replace `x` with `exp(x)` in a single expression.
fn moveup_expr(arena: &mut Arena, e: ExprId, x: ExprId) -> ExprId {
    let exp_x = arena.exp(x);
    crate::transforms::subs::subs(arena, e, x, exp_x)
}

/// Move up a SubsSet: replace x with exp(x) in all keys and rewrite values.
fn moveup_subsset(arena: &mut Arena, s: &SubsSet, x: ExprId) -> SubsSet {
    let mut result = SubsSet::new();
    for (&expr, &dummy) in &s.exprs {
        let moved = moveup_expr(arena, expr, x);
        result.exprs.insert(moved, dummy);
    }
    for (&dummy, &rw) in &s.rewrites {
        result.rewrites.insert(dummy, moveup_expr(arena, rw, x));
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Rewrite in terms of ω
// ═══════════════════════════════════════════════════════════════════════════

/// Largest denominator [`hidden_rational`] tries.
const MAX_HIDDEN_DENOMINATOR: i64 = 10_000;

/// The rational `p/q` a symbolic constant `c` equals although no canonical
/// form shows it (`ln 3/ln 27 = 1/3`), found from its continued fraction
/// (denominators up to [`MAX_HIDDEN_DENOMINATOR`]) and confirmed by a
/// proof that `c − p/q` is 0
/// ([`hidden_zero`](crate::calculus::calculus_util::hidden_zero), exact or
/// certified).  Up to 0.48 a zero to about 80 digits relative to its terms
/// confirmed it.
///
/// The exponent ratios of the rewrite must be exact: with `c = ln 3/ln 27`
/// the powers `ω^{3c}` and `ω^{−1}` of `27^{ln n/ln 3 + 1}/n³` never
/// cancelled, the leading exponent `3c − 1` was taken for a non-zero one
/// and the limit (`27`) came out `∞`.
pub(crate) fn hidden_rational(arena: &mut Arena, c: ExprId) -> Option<ExprId> {
    if arena.as_num(c).is_some()
        || !crate::base::walk::free_symbols(arena, c).is_empty()
        || crate::base::walk::has_unevaluated(arena, c)
        || contains_singular_atom(arena, c)
    {
        return None;
    }
    let v = crate::transforms::evalf::evalf_f64(arena, c).ok()?;
    if !v.is_finite() || v.abs() > 1e9 {
        return None;
    }
    // Continued-fraction convergents of v.
    let (mut h0, mut h1) = (0i64, 1i64);
    let (mut k0, mut k1) = (1i64, 0i64);
    let mut r = v;
    for _ in 0..40 {
        let a = r.floor();
        if !a.is_finite() || a.abs() > 1e12 {
            return None;
        }
        let a = a as i64;
        let h2 = a.checked_mul(h1)?.checked_add(h0)?;
        let k2 = a.checked_mul(k1)?.checked_add(k0)?;
        if k2 > MAX_HIDDEN_DENOMINATOR {
            return None;
        }
        (h0, h1, k0, k1) = (h1, h2, k1, k2);
        if (v - h1 as f64 / k1 as f64).abs() <= 1e-12 * v.abs().max(1.0) {
            let q = arena.rational(h1, k1);
            let d = arena.sub(c, q);
            let d = crate::transforms::eval::eval(arena, d);
            return (arena.is_zero_structural(d)
                || crate::calculus::calculus_util::hidden_zero(arena, d)
                    .truth_at_least(crate::base::truth::Evidence::Certified)
                    .is_true())
            .then_some(q);
        }
        let frac = r - a as f64;
        if frac.abs() < 1e-15 {
            return None;
        }
        r = 1.0 / frac;
    }
    None
}

/// Rewrite `exps` (the dummy-substituted expression) in terms of `w → 0`.
///
/// All elements of `omega` must be `Exp(something)` at this point
/// (guaranteed by moveup). Returns `(rewritten_f, log_w)`.
fn rewrite(
    arena: &mut Arena,
    exps: ExprId,
    omega: &SubsSet,
    x: ExprId,
    wsym: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<(ExprId, ExprId), crate::base::errors::SymplexError> {
    if omega.is_empty() {
        return Ok((exps, arena.zero()));
    }
    budget.tick(1)?;

    let exps_display = arena.display(exps).to_string();
    tracing::debug!(mrv_size = omega.len(), expr = %exps_display, "gruntz::rewrite: starting rewrite");

    // Collect all MRV entries: (original_expr, dummy)
    let entries: Vec<(ExprId, ExprId)> = omega.exprs.iter().map(|(&e, &d)| (e, d)).collect();

    // All entries must be Exp nodes.  The representative g is the simplest
    // (SymPy sorts by the height of the rewrite tree and takes the last):
    // the smallest tree, ties in canonical order.  It was the last entry in hash-map
    // order, which changed with the contents of the arena: `(ln(e + 1/x) −
    // 1)^(−√(e + 1/x))` at `∞` (`∞`) failed whenever `exp(−√(…)·ln(…))`
    // came last instead of `exp(x)`.
    let Some((_, g_expr)) = entries
        .iter()
        .map(|&(e, _)| (crate::base::walk::post_order_ids(arena, e).len(), e))
        .min_by(|a, b| a.0.cmp(&b.0).then_with(|| arena.canonical_cmp(a.1, b.1)))
    else {
        return Ok((exps, arena.zero()));
    };

    // Get g's exponent
    let g_exp = match arena.node(g_expr).clone() {
        ExprNode::Exp(inner) => inner,
        _ => {
            tracing::warn!("gruntz::rewrite: MRV element is not Exp after moveup!");
            return Err(crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz::rewrite",
                reason: "MRV element is not Exp (moveup may have failed)".into(),
            });
        }
    };

    // Determine sign of g's exponent (g_exp → ±∞ by construction of the
    // MRV set, so this decides whether ω = g or ω = 1/g).
    let sig = sign_at_inf(arena, g_exp, x, depth + 1, budget)?;
    tracing::debug!(
        sign = sig,
        "gruntz::rewrite: sign of representative's exponent"
    );

    // If sig > 0 (g → ∞), use w = 1/g, so log(w) = -g_exp
    // If sig < 0 (g → 0), use w = g, so log(w) = g_exp
    let w_actual = if sig >= 0 {
        tracing::debug!("gruntz::rewrite: g → ∞, using ω = 1/g (so ω → 0)");
        arena.div(arena.one(), wsym) // w → 1/wsym effectively makes wsym = 1/w
    } else {
        tracing::debug!("gruntz::rewrite: g → 0, using ω = g (already → 0)");
        wsym
    };
    let _ = w_actual; // used conceptually

    // Build the rewrite table: for each (f = exp(f_exp), dummy_f) in omega:
    //   c = lim(f_exp / g_exp)
    //   rewrite dummy_f → exp(f_exp - c*g_exp) * wsym^(±c)
    let mut subs_table: Vec<(ExprId, ExprId)> = Vec::new();
    let mut held_exps: Vec<(ExprId, ExprId)> = Vec::new();

    for &(f_expr, f_dummy) in &entries {
        // Get f's exponent (either from the Exp directly, or from rewrites)
        let f_exp = if let Some(&rewrite_form) = omega.rewrites.get(&f_dummy) {
            // f_dummy rewrites to exp(rewritten_arg), get the arg
            match arena.node(rewrite_form).clone() {
                ExprNode::Exp(inner) => inner,
                _ => rewrite_form, // shouldn't happen
            }
        } else {
            match arena.node(f_expr).clone() {
                ExprNode::Exp(inner) => inner,
                _ => {
                    tracing::warn!("gruntz::rewrite: non-Exp MRV element without rewrite");
                    continue;
                }
            }
        };

        // Compute c = lim(f_exp / g_exp, x → ∞), from the original
        // exponent (SymPy's `f.exp`): the rewritten one holds the dummies of
        // other members (`x − ω₂` for `exp(x − e⁻ˣ)`), which a limit takes
        // for constants — the leading coefficient `1 − ω₂/x` then counted
        // as an oscillation and the whole limit failed.
        let f_exp_orig = match arena.node(f_expr).clone() {
            ExprNode::Exp(inner) => inner,
            _ => f_exp,
        };
        let ratio = arena.div(f_exp_orig, g_exp);
        let c = limitinf(arena, ratio, x, depth + 1, budget)?;
        let c = hidden_rational(arena, c).unwrap_or(c);
        let c_display = arena.display(c).to_string();
        let f_exp_display = arena.display(f_exp).to_string();
        let g_exp_display = arena.display(g_exp).to_string();
        tracing::debug!(f_exp = %f_exp_display, g_exp = %g_exp_display, c = %c_display, "gruntz::rewrite: c = limitinf(f_exp/g_exp)");

        // Compute remainder = f_exp - c * g_exp
        let c_times_gexp = arena.mul(&[c, g_exp]);
        let remainder = arena.sub(f_exp, c_times_gexp);
        let remainder = crate::transforms::eval::eval(arena, remainder);

        // f = exp(remainder) * w^(±c).  The exponential of the remainder
        // stands for a fresh symbol through the simplification below, whose
        // `expand` would split it (`e^{a + b} = e^a·e^b`) into factors of the
        // class of ω that the series engine takes for constants:
        // `e^{ψ(x)} − x` at `∞` became `−ω⁻¹ + e^{−x}·e^{ψ(1/ω)}/ω` with the
        // leading term `e^{−x}·ω⁻²` and the limit `∞` (it is `−1/2`).  SymPy
        // keeps `exp(f.exp − c·g.exp)` whole too.
        let exp_remainder = if arena.is_zero_structural(remainder) {
            arena.one()
        } else {
            let held = fresh_dummy(arena, budget);
            let e = arena.exp(remainder);
            held_exps.push((held, e));
            held
        };

        let w_power = if sig >= 0 { arena.neg(c) } else { c };
        let w_to_c = arena.pow(wsym, w_power);
        let replacement = arena.mul(&[exp_remainder, w_to_c]);
        let replacement = crate::transforms::eval::eval(arena, replacement);

        subs_table.push((f_dummy, replacement));
    }

    // Apply all substitutions to the expression.  A replacement may hold
    // another member's dummy (the rewrite of `exp(x − e⁻ˣ)` holds that of
    // `e⁻ˣ`), so the table is applied until no dummy of it is left (SymPy
    // orders it by the height of the rewrite tree); in the hash-map order a
    // dummy could survive into the leading term.
    let mut f = exps;
    for _ in 0..=subs_table.len() {
        let before = f;
        for &(dummy, replacement) in &subs_table {
            f = crate::transforms::subs::subs(arena, f, dummy, replacement);
        }
        if f == before {
            break;
        }
    }

    // Simplify
    let pre_simplify = arena.display(f).to_string();
    f = simplify_positive_powers(arena, f, wsym);
    f = crate::transforms::eval::eval(arena, f);
    f = crate::transforms::expand::expand(arena, f);
    f = crate::transforms::eval::eval(arena, f);
    // The remainders may hold the dummies of other members (a DAG: the
    // remainder of `exp(x − e⁻ˣ)` holds that of `e⁻ˣ`).
    if !held_exps.is_empty() {
        for _ in 0..=2 * (subs_table.len() + held_exps.len()) {
            let before = f;
            for &(dummy, replacement) in held_exps.iter().chain(subs_table.iter()) {
                f = crate::transforms::subs::subs(arena, f, dummy, replacement);
            }
            if f == before {
                break;
            }
        }
        f = crate::transforms::eval::eval(arena, f);
    }
    let post_simplify = arena.display(f).to_string();
    tracing::debug!(before = %pre_simplify, after = %post_simplify, "gruntz::rewrite: simplification of rewritten expression");

    // Compute log(w)
    let logw = if sig >= 0 {
        arena.neg(g_exp) // w = exp(-g_exp), so log(w) = -g_exp
    } else {
        g_exp // w = exp(g_exp), so log(w) = g_exp
    };

    tracing::debug!("gruntz::rewrite: rewrite complete");
    Ok((f, logw))
}

// ═══════════════════════════════════════════════════════════════════════════
// Leading term extraction
// ═══════════════════════════════════════════════════════════════════════════

/// Extract the leading term of `f` as a power of `w`.
///
/// Returns `(c0, e0)` where `f ≈ c0 * w^e0` as `w → 0`.
/// `c0` may depend on other variables (including `x`) but NOT on `w`.
///
/// `x`, `depth` and `budget` are threaded through so that sign decisions
/// for divergent arguments (`atan(g)` with `g → ±∞`) can use
/// [`sign_at_inf`].
#[allow(clippy::too_many_arguments)]
fn leadterm(
    arena: &mut Arena,
    f: ExprId,
    w: ExprId,
    logw: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<(ExprId, ExprId), crate::base::errors::SymplexError> {
    if depth > MAX_DEPTH {
        tracing::warn!("gruntz::leadterm: max depth exceeded");
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::leadterm",
            reason: "maximum recursion depth exceeded".into(),
        });
    }
    budget.tick(1)?;

    // If f doesn't depend on w: (f, 0) — unless f is an exponential of
    // the class of ω left over by the rewrite (`e^{x}` beside `ω = e^{−x}`,
    // from splitting `e^{remainder}`): `e^{k·ln ω + a} = ωᵏ·eᵃ`.
    if !crate::base::walk::contains(arena, f, w) {
        if let ExprNode::Exp(arg) = *arena.node(f)
            && crate::base::walk::contains(arena, arg, x)
            && let Some((k, a)) = log_w_part(arena, arg, logw, x, depth, budget)?
        {
            let coeff = arena.exp(a);
            let coeff = crate::transforms::eval::eval(arena, coeff);
            return Ok((coeff, k));
        }
        tracing::trace!("gruntz::leadterm: constant wrt ω → (f, 0)");
        return Ok((f, arena.zero()));
    }

    // If f IS w: (1, 1)
    if f == w {
        tracing::trace!("gruntz::leadterm: f == ω → (1, 1)");
        return Ok((arena.one(), arena.one()));
    }

    let node = arena.node(f).clone();

    // ── Unary functions of an ω-dependent argument (none is matched by an
    //    arm of the `match` below) ──
    if let Some(inner) = node.unary_arg_if(leadterm_unary)
        && crate::base::walk::contains(arena, inner, w)
    {
        return unary_leadterm(arena, f, &node, inner, w, logw, x, depth, budget).or_else(|e| {
            if is_budget_error(&e) {
                return Err(e);
            }
            leadterm_by_tseries(arena, budget, f, w, logw).map_err(|_| e)
        });
    }

    match node {
        ExprNode::Mul(ref children) => {
            tracing::trace!(n = children.len(), "gruntz::leadterm: Mul");
            // `e^a·e^b = e^{a+b}`: the rewrite's `expand` splits the
            // exponential of a sum into factors that may each diverge while
            // their product does not (`e^{−ln Γ(1/ω)}·e^{ln Γ(1/ω + 1)}`).
            let exp_args: SmallVec<[ExprId; 4]> = children
                .iter()
                .filter_map(|&c| match arena.node(c) {
                    ExprNode::Exp(a) if crate::base::walk::contains(arena, *a, w) => Some(*a),
                    _ => None,
                })
                .collect();
            if exp_args.len() >= 2 {
                let mut rest: SmallVec<[ExprId; 6]> = children
                    .iter()
                    .copied()
                    .filter(|&c| {
                        !matches!(arena.node(c), ExprNode::Exp(a) if crate::base::walk::contains(arena, *a, w))
                    })
                    .collect();
                let sum = arena.add(&exp_args);
                let sum = crate::transforms::eval::eval(arena, sum);
                rest.push(arena.exp(sum));
                let combined = arena.mul(&rest);
                if combined != f {
                    return leadterm(arena, combined, w, logw, x, depth + 1, budget);
                }
            }
            let mut total_coeff = arena.one();
            let mut total_exp = arena.zero();
            for &child in children {
                let (c, e) = leadterm(arena, child, w, logw, x, depth + 1, budget)?;
                total_coeff = arena.mul(&[total_coeff, c]);
                total_exp = arena.add(&[total_exp, e]);
            }
            total_coeff = crate::transforms::eval::eval(arena, total_coeff);
            total_exp = crate::transforms::eval::eval(arena, total_exp);
            Ok((total_coeff, total_exp))
        }

        ExprNode::Pow(base, exp) if !crate::base::walk::contains(arena, exp, w) => {
            tracing::trace!("gruntz::leadterm: Pow with constant exponent");
            let (c_b, e_b) = leadterm(arena, base, w, logw, x, depth + 1, budget)?;
            if crate::base::walk::contains(arena, c_b, w) {
                // A bounded-oscillation marker stays bounded only under
                // positive integer powers (`sin(1/ω)⁻¹` is unbounded).
                match arena.as_num(exp) {
                    Some(r) if r.is_integer() && r.is_positive() => {}
                    _ => return Err(oscillation_err()),
                }
            }
            // A fractional power of a base tending to a negative constant
            // is taken on its branch cut: the principal value there is the
            // limit only for a real base (a complex one may approach from
            // below the cut: `√(u² + 1)` inside `asinh u` for
            // `u = atanh(1 − x) → iπ/2`).
            // The same for a base tending to `0` or `∞` along the negative
            // axis (`resolve_negative_leading` continues those it can).
            if arena.as_num(exp).is_none_or(|r| !r.is_integer())
                && crate::calculus::limit::const_sign(arena, c_b) == Some(-1)
                && !crate::calculus::limit::inner_known_real(arena, base)
            {
                return Err(crate::base::errors::SymplexError::ComputationFailed {
                    operation: "gruntz::leadterm",
                    reason: "a non-real base approaches the branch cut of a power".into(),
                });
            }
            let new_coeff = arena.pow(c_b, exp);
            let new_exp = arena.mul(&[e_b, exp]);
            Ok((
                crate::transforms::eval::eval(arena, new_coeff),
                crate::transforms::eval::eval(arena, new_exp),
            ))
        }

        // b^g with ω-dependent exponent: b^g = exp(g·ln b).
        ExprNode::Pow(base, exp) => {
            tracing::trace!("gruntz::leadterm: Pow with ω-dependent exponent → exp(g·ln b)");
            let ln_b = arena.ln(base);
            let prod = arena.mul(&[exp, ln_b]);
            let as_exp = arena.exp(prod);
            leadterm(arena, as_exp, w, logw, x, depth + 1, budget)
        }

        ExprNode::Add(ref children) => {
            tracing::trace!(
                n = children.len(),
                "gruntz::leadterm: Add — finding dominant term"
            );
            if children.is_empty() {
                return Ok((arena.zero(), arena.zero()));
            }

            // Find the term with the SMALLEST exponent (it dominates as w→0).
            // Exponents are real constants, not always rational (`(1/ω)^π`):
            // an irrational one used to count as 0, and `x^π − ln x` had
            // the "leading term" `(1 − x)·ω^(−π)` (its limit at ∞ was −∞).
            let mut raw: Vec<(ExprId, ExprId)> = Vec::new();
            for &child in children {
                let (c, e) = leadterm(arena, child, w, logw, x, depth + 1, budget)?;
                let e_eval = crate::transforms::eval::eval(arena, e);
                raw.push((c, e_eval));
            }
            // Exponents that are symbolic (`−n`, `5 − n` for `xⁿ − xⁿ⁻⁵`) are
            // ordered by their differences from the first one when those are
            // constants; the leading exponent then stays symbolic (before:
            // "cannot order the exponent −n", SymPy: `∞`).
            let mut symbolic = false;
            let mut terms: Vec<(ExprId, ExprId, ExpKey)> = Vec::with_capacity(raw.len());
            for &(c, e) in &raw {
                match exponent_key(arena, e) {
                    Some(key) => terms.push((c, e, key)),
                    None => {
                        symbolic = true;
                        break;
                    }
                }
            }
            if symbolic {
                terms.clear();
                let e_ref = raw[0].1;
                for &(c, e) in &raw {
                    let d = arena.sub(e, e_ref);
                    let d = crate::transforms::eval::eval(arena, d);
                    let key = exponent_key(arena, d).ok_or_else(|| {
                        crate::base::errors::SymplexError::ComputationFailed {
                            operation: "gruntz::leadterm",
                            reason: format!(
                                "cannot order the exponent {} of a leading term",
                                arena.display(e)
                            ),
                        }
                    })?;
                    terms.push((c, e, key));
                }
            }

            // Sort by exponent (ascending) — smallest exponent dominates
            terms.sort_by(|a, b| a.2.cmp(&b.2));

            // Group terms with the same leading exponent
            let min_key = terms[0].2.clone();
            let mut coeff_sum = arena.zero();
            let leading_exp_id = terms[0].1;
            for (c, e_id, key) in &terms {
                let same = *e_id == leading_exp_id || key.exactly_equal(&min_key);
                if !same && key.close_to(&min_key) {
                    // Numerically equal but not provably so: undecided.
                    let d = arena.sub(*e_id, leading_exp_id);
                    if !matches!(settle_constant(arena, d), ZeroTest::Zero) {
                        return Err(crate::base::errors::SymplexError::ComputationFailed {
                            operation: "gruntz::leadterm",
                            reason: "exponents of leading terms too close to order".into(),
                        });
                    }
                }
                if same || key.close_to(&min_key) {
                    coeff_sum = arena.add(&[coeff_sum, *c]);
                }
            }
            let min_exp = match &min_key {
                ExpKey::Exact(r) => r.clone(),
                ExpKey::Approx(_) => {
                    // leadterm_add_by_series shifts by a rational exponent.
                    // A coefficient not proved nonzero may cancel too.
                    let needs = crate::base::walk::contains(arena, coeff_sum, w)
                        || !matches!(settle_constant(arena, coeff_sum), ZeroTest::Nonzero(_));
                    if needs {
                        return Err(crate::base::errors::SymplexError::ComputationFailed {
                            operation: "gruntz::leadterm",
                            reason: "cancellation at an irrational exponent".into(),
                        });
                    }
                    Ratio::from_integer(BigInt::from(0))
                }
            };
            // Two spellings of one constant (`asinh 2` beside the
            // `ln(2 + √5)` of the tractable rewrite) cancel only
            // numerically; a structural test took their sum for the leading
            // coefficient (`lim_{x→0} x/(asinh(x + 2) − asinh 2)` was 0).
            coeff_sum = settled_coefficient(arena, coeff_sum)?;

            // If the leading coefficients cancel to 0 OR the coefficient
            // still depends on w (meaning further decomposition is needed),
            // fall back to series expansion.
            //
            // Example: (exp(w) - 1)/w after expansion is exp(w)/w - 1/w.
            // Both terms have exponent -1. Coefficient sum is exp(w) - 1,
            // which still contains w. The TRUE leading term requires knowing
            // that exp(w) - 1 ≈ w, so the product is w * w^(-1) = w^0.
            // Only series expansion can resolve this.
            // A sum of bounded oscillations (`cos(π/ω) + i·sin(π/ω)`, the
            // phase [`phase_split`] splits off) is a bounded oscillation: the
            // marker for `limitinf` (`0` times a vanishing factor, no limit
            // otherwise).  It may vanish identically, which then only costs
            // an answer, never a wrong one.
            if !arena.is_zero_structural(coeff_sum)
                && crate::base::walk::contains(arena, coeff_sum, w)
                && only_bounded_markers(arena, coeff_sum, w)
            {
                return Ok((coeff_sum, leading_exp_id));
            }
            let needs_series = arena.is_zero_structural(coeff_sum)
                || crate::base::walk::contains(arena, coeff_sum, w)
                || is_rational_zero(arena, coeff_sum);

            if needs_series && symbolic {
                return Err(crate::base::errors::SymplexError::ComputationFailed {
                    operation: "gruntz::leadterm",
                    reason: "leading terms cancel at a symbolic exponent".into(),
                });
            }
            if needs_series {
                let coeff_display = arena.display(coeff_sum).to_string();
                let f_display = arena.display(f).to_string();
                tracing::debug!(
                    coeff = %coeff_display,
                    full_expr = %f_display,
                    "gruntz::leadterm: Add coefficients cancel or depend on ω, using series expansion"
                );
                return leadterm_add_by_series(arena, f, w, logw, &min_exp, x, depth, budget);
            }

            Ok((coeff_sum, leading_exp_id))
        }

        ExprNode::Neg(inner) => {
            tracing::trace!("gruntz::leadterm: Neg");
            let (c, e) = leadterm(arena, inner, w, logw, x, depth + 1, budget)?;
            Ok((arena.neg(c), e))
        }

        ExprNode::Exp(arg) if crate::base::walk::contains(arena, arg, w) => {
            tracing::trace!("gruntz::leadterm: Exp(arg) where arg depends on ω");
            let (c_arg, e_arg) = match leadterm(arena, arg, w, logw, x, depth + 1, budget) {
                Ok(r) => r,
                Err(e) if is_budget_error(&e) => return Err(e),
                Err(e) => {
                    // arg = O(ωᴷ), K > 0: e^arg = 1 + O(ωᴷ).
                    if vanishes_to_positive_order(arena, budget, arg, w, logw).is_some() {
                        return Ok((arena.one(), arena.zero()));
                    }
                    return Err(e);
                }
            };
            let e_eval = crate::transforms::eval::eval(arena, e_arg);
            if let Some(r) = arena.as_num(e_eval) {
                let r = r.clone();
                if r.is_positive() {
                    // arg → 0 as w → 0 → exp(0) = 1
                    return Ok((arena.one(), arena.zero()));
                }
                if r.is_zero() {
                    // arg = k·ln ω + a + o(1): e^arg = ωᵏ·eᵃ·(1 + o(1)).
                    if let Some((k, a)) = log_w_part(arena, c_arg, logw, x, depth, budget)? {
                        let coeff = arena.exp(a);
                        let coeff = crate::transforms::eval::eval(arena, coeff);
                        return Ok((coeff, k));
                    }
                    // arg → c_arg → exp(c_arg) as coefficient
                    let coeff = arena.exp(c_arg);
                    return Ok((crate::transforms::eval::eval(arena, coeff), arena.zero()));
                }
            }
            // Negative exponent: exp(g) with g → ±∞ is its own comparability
            // class and should have been rewritten in terms of ω by the MRV
            // machinery.  Reaching here means the analysis is inconsistent.
            Err(crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz::leadterm",
                reason: "exp of a divergent argument was not captured by the MRV set".into(),
            })
        }

        ExprNode::Ln(arg) if crate::base::walk::contains(arena, arg, w) => {
            tracing::trace!("gruntz::leadterm: Ln(arg) where arg depends on ω");
            // ln(c * w^e) = ln(c) + e*ln(w) = ln(c) + e*logw
            let (c_arg, e_arg) = leadterm(arena, arg, w, logw, x, depth + 1, budget)?;
            if crate::base::walk::contains(arena, c_arg, w) {
                // ln of a bounded oscillation is unbounded (and complex).
                return Err(oscillation_err());
            }
            let e_eval = crate::transforms::eval::eval(arena, e_arg);
            // ln at a point of its cut (the negative reals) is the limit only
            // along the cut: `ln(−1 − i/x) → −iπ`, not `ln(−1) = iπ`.
            // So is `ln(c·ωᵉ)` with `c < 0` for `e ≠ 0` (see
            // `resolve_negative_leading`): `ln(−x − i)` is `ln x − iπ`.
            let on_cut = if arena.is_zero_structural(e_eval) {
                crate::calculus::limit::on_branch_cut(
                    arena,
                    crate::calculus::limit::BranchCut::NegativeReals,
                    c_arg,
                )
            } else {
                crate::calculus::limit::const_sign(arena, c_arg) == Some(-1)
            };
            if on_cut && !crate::calculus::limit::inner_known_real(arena, arg) {
                return Err(crate::base::errors::SymplexError::ComputationFailed {
                    operation: "gruntz::leadterm",
                    reason: "a non-real argument approaches the branch cut of ln".into(),
                });
            }
            if arena.is_zero_structural(e_eval) {
                // arg → c_arg as w → 0, so ln(arg) → ln(c_arg)
                let coeff = crate::calculus::series::ln_of_constant(arena, c_arg);
                let coeff = crate::transforms::eval::eval(arena, coeff);
                let one = arena.one();
                let c_minus_one = arena.sub(c_arg, one);
                let c_is_one = arena.is_zero_structural(coeff)
                    || match settle_constant(arena, c_minus_one) {
                        ZeroTest::Zero => true,
                        ZeroTest::Nonzero(_) => false,
                        ZeroTest::Undecided(_) => {
                            return Err(crate::base::errors::SymplexError::ComputationFailed {
                                operation: "gruntz::leadterm",
                                reason: format!(
                                    "cannot decide whether {} is 1",
                                    arena.display(c_arg)
                                ),
                            });
                        }
                    };
                if !c_is_one {
                    return Ok((coeff, arena.zero()));
                }
                // ln(c_arg) = 0, meaning c_arg = 1 (or equivalent).
                // We have ln(1 + δ) where δ = arg − c_arg → 0 as w → 0.
                // Taylor: ln(1 + δ) = δ − δ²/2 + δ³/3 − …
                // The leading term of ln(1 + δ) equals the leading term of δ.
                let delta = arena.sub(arg, c_arg);
                let delta = crate::transforms::eval::eval(arena, delta);
                if !arena.is_zero_structural(delta) && crate::base::walk::contains(arena, delta, w)
                {
                    tracing::debug!("gruntz::leadterm: ln(arg) with arg→1, using ln(1+δ) ≈ δ");
                    return leadterm(arena, delta, w, logw, x, depth + 1, budget);
                }
                return Ok((arena.zero(), arena.zero()));
            }
            // ln(c * w^e) = ln(c) + e*logw — treat as coefficient with power 0
            // since logw doesn't involve w (it involves x)
            let ln_c = crate::calculus::series::ln_of_constant(arena, c_arg);
            let e_logw = arena.mul(&[e_arg, logw]);
            let coeff = arena.add(&[ln_c, e_logw]);
            Ok((crate::transforms::eval::eval(arena, coeff), arena.zero()))
        }

        // x! = Γ(x + 1): reuse the Gamma pole and asymptotic rules (the
        // poles of x! at the negative integers are Γ's at 0, −1, …).
        ExprNode::Factorial(inner) if crate::base::walk::contains(arena, inner, w) => {
            let one = arena.one();
            let shifted = arena.add(&[inner, one]);
            let g = arena.gamma(shifted);
            leadterm(arena, g, w, logw, x, depth + 1, budget)
        }

        // Fallback: try series expansion
        _ => {
            tracing::debug!("gruntz::leadterm: fallback — trying series expansion");
            match leadterm_by_tseries(arena, budget, f, w, logw) {
                Ok(r) => return Ok(r),
                Err(e) if is_budget_error(&e) => return Err(e),
                Err(_) => {}
            }
            if let Ok(series) = budgeted_series(arena, budget, f, w, 4) {
                let evaled = crate::transforms::eval::eval(arena, series);
                if evaled != f && !crate::base::walk::contains(arena, evaled, w) {
                    return Ok((evaled, arena.zero()));
                }
                if evaled != f {
                    return leadterm(arena, evaled, w, logw, x, depth + 1, budget);
                }
            }
            Err(crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz::leadterm",
                reason: format!(
                    "cannot determine the leading behaviour of {}",
                    arena.display(f)
                ),
            })
        }
    }
}

/// `(k, a)` with `c = k·ln ω + a`, `k` a non-zero rational and `a` of a
/// lower comparability class than `ln ω` (`a/ln ω → 0`), for the leading
/// coefficient `c` of an exponent (`ln ω` is the expression `logw` in `x`);
/// `None` when `c` is itself of a lower class (`k = 0`, a constant `c`) or
/// its ratio to `ln ω` has no such limit.
///
/// The leading coefficient of `e^g` is `e^c` only for such a lower-class
/// `c`: `e^{k·ln ω}` is `ωᵏ`, a power of `ω`, which taken for a coefficient
/// gave the leading term the wrong exponent.  The rewrite meets such an
/// exponential once `exp(k·ln z) = z^k` is folded for its dummy:
/// `exp(−ln(1/ω) + ln Γ(1/ω + 1) − ln Γ(1/ω))` becomes
/// `ω·exp(ln Γ(1/ω + 1) − ln Γ(1/ω))`, whose second factor's exponent
/// tends to infinity like `−ln ω` — and `lim Γ(x + 1)/(x·Γ(x))` at `∞`
/// was `0` (it is `1`).  Without that fold, `x·e^{ln Γ(x) − ln Γ(x + 1)}`
/// at `∞` (rewritten as `ω⁻¹·e^{ln Γ(1/ω) − ln Γ(1/ω + 1)}`) was `∞`
/// (it is `1`).
fn log_w_part(
    arena: &mut Arena,
    c: ExprId,
    logw: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<Option<(ExprId, ExprId)>, crate::base::errors::SymplexError> {
    if !crate::base::walk::contains(arena, c, x) || !crate::base::walk::contains(arena, logw, x) {
        return Ok(None);
    }
    let ratio = arena.div(c, logw);
    let k = match limitinf(arena, ratio, x, depth + 1, budget) {
        Ok(k) => k,
        Err(e) if is_budget_error(&e) => return Err(e),
        Err(_) => return Ok(None),
    };
    // `k` a non-zero real constant: rational, or one of known sign
    // (`−√e`; before, only a rational `k` was taken, and `e^{√e·x}` stayed
    // in the coefficient of `(1/x)^{1 − √(e + 1/x)}`, whose limit at `∞`
    // came out `0` (it is `∞`)).
    let k = hidden_rational(arena, k).unwrap_or(k);
    let real_constant = match arena.as_num(k) {
        Some(kq) => !kq.is_zero(),
        None => {
            crate::base::walk::free_symbols(arena, k).is_empty()
                && !contains_singular_atom(arena, k)
                && !may_leave_reals(arena, k)
                && matches!(crate::calculus::limit::const_sign(arena, k), Some(s) if s != 0)
        }
    };
    if !real_constant {
        return Ok(None);
    }
    let k_logw = arena.mul(&[k, logw]);
    let a = arena.sub(c, k_logw);
    let a = crate::transforms::eval::eval(arena, a);
    Ok(Some((k, a)))
}

/// A real constant exponent of a leading term, for ordering: exact when
/// rational, otherwise its value (`π`, `√2`).
#[derive(Clone, Debug)]
enum ExpKey {
    Exact(Q),
    Approx(f64),
}

impl ExpKey {
    fn value(&self) -> f64 {
        match self {
            ExpKey::Exact(r) => num_traits::ToPrimitive::to_f64(r).unwrap_or(f64::NAN),
            ExpKey::Approx(v) => *v,
        }
    }

    fn cmp(&self, other: &ExpKey) -> std::cmp::Ordering {
        match (self, other) {
            (ExpKey::Exact(a), ExpKey::Exact(b)) => a.cmp(b),
            _ => self.value().total_cmp(&other.value()),
        }
    }

    fn exactly_equal(&self, other: &ExpKey) -> bool {
        matches!((self, other), (ExpKey::Exact(a), ExpKey::Exact(b)) if a == b)
    }

    /// Within rounding of each other (and not both exact).
    fn close_to(&self, other: &ExpKey) -> bool {
        if let (ExpKey::Exact(_), ExpKey::Exact(_)) = (self, other) {
            return false;
        }
        let (a, b) = (self.value(), other.value());
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
    }
}

/// The ordering key of the evaluated exponent `e`: `None` when it is not
/// a real constant.
fn exponent_key(arena: &mut Arena, e: ExprId) -> Option<ExpKey> {
    if let Some(r) = arena.as_num(e) {
        return Some(ExpKey::Exact(r.clone()));
    }
    if !crate::base::walk::free_symbols(arena, e).is_empty() {
        return None;
    }
    // A hidden rational (`3·ln 3/ln 27`) orders exactly.
    if let Some(q) = hidden_rational(arena, e)
        && let Some(r) = arena.as_num(q)
    {
        return Some(ExpKey::Exact(r.clone()));
    }
    let v = crate::transforms::evalf::eval_const_f64(arena, e)?;
    v.is_finite().then_some(ExpKey::Approx(v))
}

/// Error for a bounded-oscillation marker (`sin(1/ω)`, …) reaching an
/// operation under which it is no longer bounded.
fn oscillation_err() -> crate::base::errors::SymplexError {
    crate::base::errors::SymplexError::ComputationFailed {
        operation: "gruntz::leadterm",
        reason: "unbounded function of a bounded oscillation has no limit".into(),
    }
}

/// `true` if `F` is bounded and continuous on the whole real line, so that
/// `F(bounded oscillation)` is again a bounded oscillation.
fn unary_is_bounded(template: &ExprNode) -> bool {
    matches!(
        template,
        ExprNode::Sin(_)
            | ExprNode::Cos(_)
            | ExprNode::Atan(_)
            | ExprNode::Tanh(_)
            | ExprNode::Erf(_)
            | ExprNode::Erfc(_)
            | ExprNode::Abs(_)
            | ExprNode::Sign(_)
            | ExprNode::Heaviside(_)
    )
}

/// Rebuild a unary function node of the same kind as `template` applied to
/// `arg`.
fn apply_unary(arena: &mut Arena, template: &ExprNode, arg: ExprId) -> ExprId {
    // The one-argument function of `template`, built by its constructor
    // (`Arena::func`); until 0.49 a list of 24 arms, one per function the
    // callers (`mrv`, `unary_leadterm`) pass.
    match template.func(arena) {
        Some((f, args)) if args.len() == 1 => arena.func(f, &[arg]).unwrap_or(arg),
        _ => arg,
    }
}

/// `true` if `F(c)` is a pole/undefined point for the function kind.
fn unary_is_singular_at(arena: &Arena, template: &ExprNode, c: ExprId) -> bool {
    let r = arena.as_num(c);
    match template {
        ExprNode::Gamma(_) | ExprNode::LogGamma(_) | ExprNode::Digamma(_) => match r {
            Some(r) => r.is_integer() && !r.is_positive(),
            None => false,
        },
        ExprNode::Factorial(_) => match r {
            Some(r) => r.is_integer() && r.is_negative(),
            None => false,
        },
        ExprNode::Atanh(_) => match r {
            Some(r) => r.abs() == Ratio::from_integer(BigInt::from(1)),
            None => false,
        },
        ExprNode::Sign(_) | ExprNode::Heaviside(_) => r.is_some_and(|r| r.is_zero()),
        _ => false,
    }
}

/// The one-argument functions [`unary_leadterm`] expands at an
/// ω-dependent argument: the trigonometric and hyperbolic functions and
/// their inverses, `erf`, `erfc`, `Γ`, `ln Γ`, `ψ`, `W`, `|·|`, `sign`, `H`.
fn leadterm_unary(f: Func) -> bool {
    matches!(f.elementary(), Some(e) if !matches!(e, Elementary::Exp | Elementary::Ln))
        || matches!(
            f,
            Func::Erf
                | Func::Erfc
                | Func::Gamma
                | Func::LogGamma
                | Func::Digamma
                | Func::LambertW0
                | Func::Abs
                | Func::Sign
                | Func::Heaviside
        )
}

/// Leading term of `F(inner)` for a unary function `F`.
///
/// * `inner → 0`: expand `F` at 0 and take the first non-vanishing term.
/// * `inner → c` (finite): `F(c)` if nonzero, otherwise expand `F(c + δ)`.
/// * `inner → ±∞`: use the known asymptotic value of `F` (via the eventual
///   sign of the argument); bounded oscillating functions are treated as
///   `O(1)` and unbounded ones are rejected.
#[allow(clippy::too_many_arguments)]
fn unary_leadterm(
    arena: &mut Arena,
    f: ExprId,
    template: &ExprNode,
    inner: ExprId,
    w: ExprId,
    logw: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<(ExprId, ExprId), crate::base::errors::SymplexError> {
    let (c_in, e_in) = leadterm(arena, inner, w, logw, x, depth + 1, budget)?;
    let e_eval = crate::transforms::eval::eval(arena, e_in);
    let Some(r) = arena.as_num(e_eval).cloned() else {
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::leadterm",
            reason: "non-numeric exponent in function argument".into(),
        });
    };
    let zero = arena.zero();

    if r.is_positive() {
        // inner → 0.
        if unary_is_singular_at(arena, template, zero) {
            if crate::base::walk::contains(arena, c_in, w) {
                // 1/(bounded oscillation) is unbounded.
                return Err(oscillation_err());
            }
            // Γ(t) ≈ 1/t and ψ(t) ≈ −1/t near the origin.
            if matches!(template, ExprNode::Gamma(_) | ExprNode::Digamma(_)) {
                let neg_e = arena.neg(e_in);
                let inv_c = arena.pow(c_in, arena.neg_one());
                let inv_c = crate::transforms::eval::eval(arena, inv_c);
                let coeff = if matches!(template, ExprNode::Gamma(_)) {
                    inv_c
                } else {
                    arena.neg(inv_c)
                };
                return Ok((coeff, crate::transforms::eval::eval(arena, neg_e)));
            }
            return Err(crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz::leadterm",
                reason: "function has a pole at the limit of its argument".into(),
            });
        }
        tracing::trace!("gruntz::leadterm: F(inner) with inner → 0, expanding F at 0");
        let t = fresh_dummy(arena, budget);
        let ft = apply_unary(arena, template, t);
        let p = budgeted_series(arena, budget, ft, t, 6)?;
        let p0 = crate::transforms::subs::subs(arena, p, t, zero);
        let p0 = crate::transforms::eval::eval(arena, p0);
        if !arena.is_zero_structural(p0) {
            return Ok((p0, zero));
        }
        let p_inner = crate::transforms::subs::subs(arena, p, t, inner);
        let p_inner = crate::transforms::eval::eval(arena, p_inner);
        if arena.is_zero_structural(p_inner) || p_inner == f {
            return Err(crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz::leadterm",
                reason: "series expansion of function did not resolve the leading term".into(),
            });
        }
        return leadterm(arena, p_inner, w, logw, x, depth + 1, budget);
    }

    if r.is_zero() {
        // inner → c_in (a constant, possibly depending on x).
        if crate::base::walk::contains(arena, c_in, w) && !unary_is_bounded(template) {
            // `F(bounded oscillation)` is only a bounded oscillation when
            // `F` itself is bounded (ln, Γ, tan, … are not).
            return Err(oscillation_err());
        }
        if unary_is_singular_at(arena, template, c_in) {
            // Γ(−n + δ) ≈ (−1)ⁿ / (n!·δ) at its poles.
            if let ExprNode::Gamma(_) = template
                && let Some(cn) = arena.as_num(c_in).cloned()
                && cn.is_integer()
                && !cn.is_positive()
            {
                let n = (-cn.to_integer()).to_u64().unwrap_or(0);
                let mut fact = BigInt::from(1u64);
                for i in 2..=n {
                    fact *= BigInt::from(i);
                }
                let sign = if n % 2 == 0 { 1 } else { -1 };
                let coeff_rat = Ratio::from_integer(BigInt::from(sign)) / Ratio::from_integer(fact);
                let coeff = {
                    let nid = arena.intern_num(coeff_rat);
                    arena.intern(ExprNode::Num(nid))
                };
                let delta = arena.sub(inner, c_in);
                let delta = crate::transforms::eval::eval(arena, delta);
                if arena.is_zero_structural(delta) {
                    return Err(crate::base::errors::SymplexError::ComputationFailed {
                        operation: "gruntz::leadterm",
                        reason: "Γ evaluated exactly at a pole".into(),
                    });
                }
                let inv_delta = arena.pow(delta, arena.neg_one());
                let approx = arena.mul(&[coeff, inv_delta]);
                return leadterm(arena, approx, w, logw, x, depth + 1, budget);
            }
            return Err(crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz::leadterm",
                reason: "function has a pole at the limit of its argument".into(),
            });
        }
        // On a branch cut the value there is the limit only along the cut
        // (a real argument) or from the side the principal value is
        // continuous with: `asin(2 + iω)` tends to `π − asin 2`, not
        // `asin 2` (a branch function whose argument tends to its cut is
        // continued across the cut by `rewrite_tractable`; this is the
        // guard for one it did not see).
        if let Some((_, cut)) = crate::calculus::limit::branch_cut_of(arena, f)
            && crate::calculus::limit::on_branch_cut(arena, cut, c_in)
            && (cut == crate::calculus::limit::BranchCut::ImaginaryAxis
                || !crate::calculus::limit::inner_known_real(arena, inner))
        {
            return Err(crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz::leadterm",
                reason: "a non-real argument approaches the branch cut of a function".into(),
            });
        }
        let v = apply_unary(arena, template, c_in);
        // `cos` of a hidden `−π/2` (`3·sec(x/3 − 4πx)` at `3π/(24π − 2)`:
        // before, the pole was the finite value `3/cos(…)`), or a value that
        // vanishes in the parameters, is a zero.
        let v = crate::transforms::eval::eval(arena, v);
        let v = settled_coefficient(arena, v)?;
        if is_infinite(arena, v) || v == arena.nan() {
            return Err(crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz::leadterm",
                reason: "function value is infinite at the limit of its argument".into(),
            });
        }
        if !arena.is_zero_structural(v) {
            return Ok((v, zero));
        }
        // F(c) = 0: expand F(c + δ) with δ = inner − c → 0.
        let delta = arena.sub(inner, c_in);
        let delta = crate::transforms::eval::eval(arena, delta);
        if arena.is_zero_structural(delta) {
            return Ok((zero, zero));
        }
        tracing::trace!("gruntz::leadterm: F(c) = 0, expanding F(c + δ)");
        let t = fresh_dummy(arena, budget);
        let arg = arena.add(&[c_in, t]);
        let ft = apply_unary(arena, template, arg);
        let p = budgeted_series(arena, budget, ft, t, 6)?;
        let p_delta = crate::transforms::subs::subs(arena, p, t, delta);
        let p_delta = crate::transforms::eval::eval(arena, p_delta);
        if arena.is_zero_structural(p_delta) || p_delta == f {
            return Err(crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz::leadterm",
                reason: "series expansion of function did not resolve the leading term".into(),
            });
        }
        return leadterm(arena, p_delta, w, logw, x, depth + 1, budget);
    }

    // `sin`, `cos` of a real argument tending to `±∞` are a bounded
    // oscillation whichever the side: `cos(m·x)/x → 0` at `∞` for a real
    // `m` of unknown sign (it was refused, "cannot determine the sign of
    // m"; SymPy: 0).
    if matches!(template, ExprNode::Sin(_) | ExprNode::Cos(_)) && {
        // Real for real values of its symbols, all of which are real (the
        // variables, `ω > 0`, or parameters declared real).
        let mut cache = crate::base::assumptions::AssumptionCache::new();
        !may_leave_reals(arena, inner)
            && crate::base::walk::free_symbols(arena, inner)
                .into_iter()
                .all(|s| {
                    s == x
                        || s == w
                        || Truth::from(cache.query(arena, s, crate::base::assumptions::Props::REAL))
                            .is_true()
                })
    } {
        return Ok((f, zero));
    }
    // inner → ±∞: sign of the leading coefficient decides the side.
    let s = if crate::base::walk::contains(arena, c_in, x) {
        sign_at_inf(arena, c_in, x, depth + 1, budget)?
    } else {
        sign_of_constant(arena, c_in)?
    };
    let pos = s > 0;
    let unbounded = || {
        Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::leadterm",
            reason: "unbounded function of a divergent argument".into(),
        })
    };
    match template {
        ExprNode::Atan(_) => {
            let two = arena.int(2);
            let pi = arena.pi();
            let half_pi = arena.div(pi, two);
            let v = if pos { half_pi } else { arena.neg(half_pi) };
            Ok((v, zero))
        }
        ExprNode::Erf(_) | ExprNode::Tanh(_) | ExprNode::Sign(_) => {
            let v = if pos { arena.one() } else { arena.neg_one() };
            Ok((v, zero))
        }
        ExprNode::Erfc(_) => {
            if pos {
                // erfc(+∞) decays like exp(−t²): not a power of ω.
                unbounded()
            } else {
                Ok((arena.int(2), zero))
            }
        }
        ExprNode::Heaviside(_) => Ok((if pos { arena.one() } else { zero }, zero)),
        ExprNode::Abs(_) => {
            if s == 0 {
                return unbounded();
            }
            let c = if pos { c_in } else { arena.neg(c_in) };
            Ok((crate::transforms::eval::eval(arena, c), e_in))
        }
        // Bounded oscillation: O(1), keep the function as the coefficient.
        ExprNode::Sin(_) | ExprNode::Cos(_) => Ok((f, zero)),
        // W(u) ~ ln u as u → +∞ (leading order only).
        ExprNode::LambertW(_) if pos => {
            let ln_inner = arena.ln(inner);
            leadterm(arena, ln_inner, w, logw, x, depth + 1, budget)
        }
        // `asin`, `acos` of a real argument along the infinite ends of their
        // cut, by the principal values there (mpmath `asin(10²⁰)` = π/2 −
        // 46.74i): `asin u = π/2 − i·acosh u` (`u → +∞`), `−π/2 + i·acosh(−u)`
        // (`u → −∞`), `acos u = π/2 − asin u`, with `acosh v = ln(v +
        // √(v + 1)·√(v − 1))`.  (A non-real argument is continued by
        // `resolve_branch_cut` first.)  Before, `asin(x) + i·ln x` at `∞` was
        // refused (it is `π/2 − i·ln 2`).
        ExprNode::Asin(_) | ExprNode::Acos(_)
            if s != 0 && crate::calculus::limit::inner_known_real(arena, inner) =>
        {
            let v = if pos { inner } else { arena.neg(inner) };
            let one = arena.one();
            let vp = arena.add(&[v, one]);
            let vm = arena.sub(v, one);
            let rp = arena.sqrt(vp);
            let rm = arena.sqrt(vm);
            let root = arena.mul(&[rp, rm]);
            let sum = arena.add(&[v, root]);
            let ach = arena.ln(sum);
            let i = arena.i_unit();
            let pi = arena.pi();
            let half = arena.rational(1, 2);
            let half_pi = arena.mul(&[half, pi]);
            let i_ach = arena.mul(&[i, ach]);
            let asin = if pos {
                arena.sub(half_pi, i_ach)
            } else {
                let nh = arena.neg(half_pi);
                arena.add(&[nh, i_ach])
            };
            let g = if matches!(template, ExprNode::Asin(_)) {
                asin
            } else {
                arena.sub(half_pi, asin)
            };
            let g = crate::transforms::eval::eval(arena, g);
            leadterm(arena, g, w, logw, x, depth + 1, budget)
        }
        _ => unbounded(),
    }
}

/// Resolve an `Add` whose leading coefficients cancel (or still depend on
/// `w`) by series expansion.
///
/// The sum is first *normalised* so that every pole in `w` is explicit
/// (`(A)^k → w^{ek}·(A·w^{−e})^k`, `ln A → e·logw + ln(A·w^{−e})`, `b^g →
/// exp(g·ln b)`), then multiplied by `w^{−e_min}` to make it regular at
/// `w = 0`, and finally expanded as a Taylor series in `w`.
#[allow(clippy::too_many_arguments)]
fn leadterm_add_by_series(
    arena: &mut Arena,
    f: ExprId,
    w: ExprId,
    logw: ExprId,
    min_exp: &Q,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<(ExprId, ExprId), crate::base::errors::SymplexError> {
    let min_exp_id = {
        let nid = arena.intern_num(min_exp.clone());
        arena.intern(ExprNode::Num(nid))
    };
    budget.charge_size(arena, f)?;

    // Strategy 0: the truncated Laurent engine, which keeps track of its
    // precision (see `leadterm_by_tseries`).
    match leadterm_by_tseries(arena, budget, f, w, logw) {
        Ok(r) => return Ok(r),
        Err(e) if is_budget_error(&e) => return Err(e),
        Err(_) => {}
    }
    // When the engine has shown `f = O(ωᴷ)` (every coefficient below its
    // precision `K` vanishes), a "leading term" of lower order from the
    // strategies below — which substitute truncated series and can
    // manufacture one — is spurious and is discarded: for the identically
    // zero `ln Γ(1/ω + 1) − ln Γ(1/ω) − ln(1/ω)` they produced a term of
    // negative order, and `Γ(x + 1)/(x·Γ(x))` at `∞` failed.
    let floor = vanishes_to_positive_order(arena, budget, f, w, logw);
    let below_floor = |arena: &Arena, e: ExprId| -> bool {
        floor.is_some_and(|k| {
            arena
                .as_num(e)
                .is_some_and(|r| *r < Ratio::from_integer(BigInt::from(k)))
        })
    };

    // Strategy 1: normalise poles and expand as a regular Taylor series.
    if let Ok(normalized) = normalize_poles(arena, f, w, logw, x, depth, budget) {
        let neg_min = {
            let nid = arena.intern_num(-min_exp.clone());
            arena.intern(ExprNode::Num(nid))
        };
        let w_shift = arena.pow(w, neg_min);
        let g = arena.mul(&[normalized, w_shift]);
        let g = simplify_positive_powers(arena, g, w);
        let g = crate::transforms::eval::eval(arena, g);
        budget.charge_size(arena, g)?;
        let g = crate::transforms::expand::expand(arena, g);
        let g = crate::transforms::eval::eval(arena, g);
        let g_display = arena.display(g).to_string();
        tracing::debug!(regular = %g_display, "gruntz::leadterm_add_by_series: normalised expression");

        for order in [3u32, 5, 8, 12] {
            match budgeted_series(arena, budget, g, w, order) {
                Ok(s) => {
                    let h = crate::transforms::expand::expand(arena, s);
                    let h = crate::transforms::eval::eval(arena, h);
                    if arena.is_zero_structural(h) {
                        continue;
                    }
                    let h_display = arena.display(h).to_string();
                    tracing::debug!(order, series = %h_display, "gruntz::leadterm_add_by_series: series");
                    let (c, e) = leadterm(arena, h, w, logw, x, depth + 1, budget)?;
                    if arena.is_zero_structural(c) {
                        continue;
                    }
                    let e_total = arena.add(&[e, min_exp_id]);
                    let e_total = crate::transforms::eval::eval(arena, e_total);
                    if below_floor(arena, e_total) {
                        break;
                    }
                    return Ok((c, e_total));
                }
                Err(_) => break,
            }
        }
    }

    // Strategy 2: expand individual functions as Taylor series, substitute
    // back, simplify algebraically, and retry.
    //
    // Substituting a *truncated* series into an expression with poles can
    // manufacture a spurious leading term when the true cancellation runs
    // deeper than the truncation order, so the result is only trusted when
    // two different orders agree.
    let mut agreed: Option<(ExprId, ExprId)> = None;
    for order in [6u32, 10, 14] {
        let func_expanded = expand_functions_as_series(arena, budget, f, w, order)?;
        if func_expanded == f {
            break;
        }
        let simplified = crate::transforms::eval::eval(arena, func_expanded);
        budget.charge_size(arena, simplified)?;
        let simplified = crate::transforms::expand::expand(arena, simplified);
        let simplified = crate::transforms::eval::eval(arena, simplified);
        budget.charge_size(arena, simplified)?;
        if simplified == f {
            break;
        }
        if arena.is_zero_structural(simplified) {
            // Everything cancelled to this order: the true leading term is
            // of higher order, try again with more terms.
            agreed = None;
            continue;
        }
        let Ok((c, e)) = leadterm(arena, simplified, w, logw, x, depth + 1, budget) else {
            break;
        };
        let c = crate::transforms::eval::eval(arena, c);
        let e = crate::transforms::eval::eval(arena, e);
        if below_floor(arena, e) {
            break;
        }
        match agreed {
            Some((c0, e0)) if c0 == c && e0 == e => {
                tracing::debug!(
                    "gruntz::leadterm_add_by_series: function-series strategy agreed at two consecutive orders"
                );
                return Ok((c, e));
            }
            _ => agreed = Some((c, e)),
        }
    }

    // Strategy 3: full series expansion of f itself (works when f is
    // regular at w = 0).
    for order in 2..10 {
        if let Ok(series) = budgeted_series(arena, budget, f, w, order) {
            let expanded = crate::transforms::expand::expand(arena, series);
            let evaled = crate::transforms::eval::eval(arena, expanded);
            if evaled != f && !arena.is_zero_structural(evaled) {
                let r = leadterm(arena, evaled, w, logw, x, depth + 1, budget)?;
                if below_floor(arena, r.1) {
                    break;
                }
                return Ok(r);
            }
        } else {
            break;
        }
    }

    Err(crate::base::errors::SymplexError::ComputationFailed {
        operation: "gruntz::leadterm",
        reason: "leading coefficients cancel and series expansion failed".into(),
    })
}

/// `(ω^a)^b → ω^{ab}` and `(c·ω^a)^b → c^b·ω^{ab}` (for `c` a product of
/// positive numbers and exponentials, or integer `b`).
///
/// These identities hold because `ω → 0⁺` is positive; canonicalization
/// cannot apply them to a general symbol (`√(1/ω) ≠ 1/√ω` for `ω < 0`).
/// Without them expressions such as `√(1/ω)·√(1/ω + 1) − 1/ω` never become
/// regular enough for series expansion.
fn simplify_positive_powers(arena: &mut Arena, f: ExprId, w: ExprId) -> ExprId {
    let post = crate::base::walk::post_order_ids(arena, f);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post {
        if !crate::base::walk::contains(arena, id, w) {
            cache.insert(id, id);
            continue;
        }
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match arena.node(rebuilt).clone() {
            ExprNode::Pow(base, b) if arena.as_num(b).is_some() => match arena.node(base).clone() {
                ExprNode::Pow(inner, a) if inner == w && arena.as_num(a).is_some() => {
                    let ab = arena.mul(&[a, b]);
                    let ab = crate::transforms::eval::eval(arena, ab);
                    arena.pow(w, ab)
                }
                ExprNode::Mul(children) => {
                    let mut w_exp = Ratio::from_integer(BigInt::from(0));
                    let mut rest: SmallVec<[ExprId; 4]> = SmallVec::new();
                    for &c in &children {
                        if c == w {
                            w_exp += Ratio::from_integer(BigInt::from(1));
                        } else if let ExprNode::Pow(inner, a) = arena.node(c).clone()
                            && inner == w
                            && let Some(r) = arena.as_num(a)
                        {
                            w_exp += r.clone();
                        } else {
                            rest.push(c);
                        }
                    }
                    let b_int = arena.as_num(b).is_some_and(|r| r.is_integer());
                    let rest_positive = rest.iter().all(|&c| {
                        arena.as_num(c).is_some_and(|r| r.is_positive())
                            || matches!(arena.node(c), ExprNode::Exp(_))
                    });
                    if w_exp.is_zero() || !(b_int || rest_positive) {
                        rebuilt
                    } else {
                        let w_exp_id = {
                            let nid = arena.intern_num(w_exp);
                            arena.intern(ExprNode::Num(nid))
                        };
                        let ab = arena.mul(&[w_exp_id, b]);
                        let ab = crate::transforms::eval::eval(arena, ab);
                        let w_part = arena.pow(w, ab);
                        let rest_mul = arena.mul(&rest);
                        let rest_part = arena.pow(rest_mul, b);
                        arena.mul(&[w_part, rest_part])
                    }
                }
                _ => rebuilt,
            },
            _ => rebuilt,
        };
        cache.insert(id, new);
    }

    cache.get(&f).copied().unwrap_or(f)
}

/// Make every pole in `w` explicit so that `f · w^{−e_min}` is regular at
/// `w = 0` and amenable to Taylor expansion. See [`leadterm_add_by_series`].
#[allow(clippy::too_many_arguments)]
fn normalize_poles(
    arena: &mut Arena,
    f: ExprId,
    w: ExprId,
    logw: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    let post = crate::base::walk::post_order_ids(arena, f);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post {
        if !crate::base::walk::contains(arena, id, w) {
            cache.insert(id, id);
            continue;
        }
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let node = arena.node(rebuilt).clone();
        let new = match node {
            ExprNode::Pow(base, k)
                if base != w
                    && !crate::base::walk::contains(arena, k, w)
                    && crate::base::walk::contains(arena, base, w)
                    && matches!(arena.node(base), ExprNode::Add(_)) =>
            {
                let (_, e) = leadterm(arena, base, w, logw, x, depth + 1, budget)?;
                let e = crate::transforms::eval::eval(arena, e);
                if arena.as_num(e).is_some() && !arena.is_zero_structural(e) {
                    let neg_e = arena.neg(e);
                    let w_neg_e = arena.pow(w, neg_e);
                    let scaled = arena.mul(&[base, w_neg_e]);
                    let scaled = crate::transforms::expand::expand(arena, scaled);
                    let scaled = crate::transforms::eval::eval(arena, scaled);
                    let ek = arena.mul(&[e, k]);
                    let w_ek = arena.pow(w, ek);
                    let regular_pow = arena.pow(scaled, k);
                    arena.mul(&[w_ek, regular_pow])
                } else {
                    rebuilt
                }
            }
            ExprNode::Pow(base, k) if crate::base::walk::contains(arena, k, w) => {
                let ln_b = arena.ln(base);
                let prod = arena.mul(&[k, ln_b]);
                arena.exp(prod)
            }
            ExprNode::Ln(arg) => {
                let (_, e) = leadterm(arena, arg, w, logw, x, depth + 1, budget)?;
                let e = crate::transforms::eval::eval(arena, e);
                if arena.as_num(e).is_some() && !arena.is_zero_structural(e) {
                    let neg_e = arena.neg(e);
                    let w_neg_e = arena.pow(w, neg_e);
                    let scaled = arena.mul(&[arg, w_neg_e]);
                    let scaled = crate::transforms::expand::expand(arena, scaled);
                    let scaled = crate::transforms::eval::eval(arena, scaled);
                    let e_logw = arena.mul(&[e, logw]);
                    let ln_scaled = arena.ln(scaled);
                    arena.add(&[e_logw, ln_scaled])
                } else {
                    rebuilt
                }
            }
            _ => rebuilt,
        };
        cache.insert(id, new);
    }

    Ok(cache.get(&f).copied().unwrap_or(f))
}

// ═══════════════════════════════════════════════════════════════════════════
// Core: mrv_leadterm and limitinf
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the leading term of `e` as `x → ∞`.
///
/// Returns `(c0, e0)` where `e ≈ c0 · ω^e0` and `ω → 0`.  With
/// `limit_only` (the caller is [`limitinf`], which needs only the limit),
/// an `e` that vanishes to a positive order without a nameable leading term
/// gives `(0, K)` (see [`vanishes_to_positive_order`]).
fn mrv_leadterm(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
    limit_only: bool,
) -> Result<(ExprId, ExprId), crate::base::errors::SymplexError> {
    if depth > MAX_DEPTH {
        tracing::warn!("gruntz::mrv_leadterm: max depth exceeded");
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::mrv_leadterm",
            reason: "maximum recursion depth exceeded".into(),
        });
    }
    budget.tick(1)?;

    if !crate::base::walk::contains(arena, e, x) {
        return Ok((e, arena.zero()));
    }

    let e_display = arena.display(e).to_string();
    tracing::debug!(depth, expr = %e_display, "gruntz::mrv_leadterm: Step 1 — computing MRV set");

    // Step 1: Compute MRV set
    let (omega, exps) = mrv(arena, e, x, depth + 1, budget)?;

    if omega.is_empty() {
        return Ok((exps, arena.zero()));
    }

    let exps_display = arena.display(exps).to_string();
    let mrv_keys: Vec<String> = omega
        .exprs
        .keys()
        .map(|k| arena.display(*k).to_string())
        .collect();
    tracing::debug!(
        mrv_size = omega.len(),
        mrv_keys = ?mrv_keys,
        rewritten = %exps_display,
        "gruntz::mrv_leadterm: MRV set computed"
    );

    // Step 2: If x is in the MRV set, move up (x → exp(x))
    // CRITICAL: do NOT recompute MRV! Just transform the existing set.
    let (omega, exps) = if omega.contains_key(&x) {
        tracing::debug!("gruntz::mrv_leadterm: Step 2 — x in MRV, moving up (x → exp(x))");
        let omega_up = moveup_subsset(arena, &omega, x);
        let exps_up = moveup_expr(arena, exps, x);
        let up_display = arena.display(exps_up).to_string();
        let up_keys: Vec<String> = omega_up
            .exprs
            .keys()
            .map(|k| arena.display(*k).to_string())
            .collect();
        tracing::debug!(moved_expr = %up_display, moved_mrv = ?up_keys, "gruntz::mrv_leadterm: after moveup");
        (omega_up, exps_up)
    } else {
        tracing::trace!("gruntz::mrv_leadterm: x not in MRV, no moveup needed");
        (omega, exps)
    };

    if omega.is_empty() {
        return Ok((exps, arena.zero()));
    }

    // Step 3: Rewrite in terms of w
    tracing::debug!("gruntz::mrv_leadterm: Step 3 — rewriting in terms of ω");
    let w = fresh_dummy(arena, budget);
    let (f, logw) = rewrite(arena, exps, &omega, x, w, depth, budget)?;

    let f_display = arena.display(f).to_string();
    let w_display = arena.display(w).to_string();
    let logw_display = arena.display(logw).to_string();
    tracing::debug!(rewritten_f = %f_display, w = %w_display, logw = %logw_display, "gruntz::mrv_leadterm: Step 4 — extracting leading term from rewritten expression");

    // Step 4: Extract leading term (by the expansions of a series as a last
    // resort, see `leadterm_last_resort`).
    let (c0, e0) = match leadterm(arena, f, w, logw, x, depth + 1, budget) {
        Ok(r) => r,
        Err(err) if is_budget_error(&err) => return Err(err),
        Err(err) => {
            let vanishes = if limit_only {
                vanishes_to_positive_order(arena, budget, f, w, logw)
            } else {
                None
            };
            match vanishes {
                Some(k) => (arena.zero(), arena.int(k)),
                None => leadterm_last_resort(arena, budget, f, w, logw).map_err(|_| err)?,
            }
        }
    };

    let c0_display = arena.display(c0).to_string();
    let e0_display = arena.display(e0).to_string();
    tracing::debug!(c0 = %c0_display, e0 = %e0_display, "gruntz::mrv_leadterm: leading term extracted");

    // `c₀·ω^{e₀}` with `e₀ < 0` (or undecided) needs the phase of `ω`, which a
    // member of unknown phase does not give ([`phase_unknown`]); `e₀ ≥ 0`
    // does not (`|ω| → 0`).  Up to 0.49 `Γ((−1)^{1/7} − x/2)·sinh(√5·i +
    // 2x)` at `−∞` was `(−cos √5 + i·sin √5)·∞/2` (its phase turns).
    if !arena.as_num(e0).is_some_and(|r| !r.is_negative())
        && !matches!(crate::calculus::limit::const_sign(arena, e0), Some(0 | 1))
    {
        let members: Vec<ExprId> = omega.exprs.keys().copied().collect();
        for m in members {
            if phase_unknown(arena, m, x, depth, budget)? {
                return Err(crate::base::errors::SymplexError::ComputationFailed {
                    operation: "gruntz::mrv_leadterm",
                    reason: format!(
                        "the phase of {} is not known (a non-real divergent exponent)",
                        arena.display(m)
                    ),
                });
            }
        }
    }

    // Replace log(w) → logw in c0 (SymPy does this as the last step)
    let ln_w = arena.ln(w);
    let c0 = crate::transforms::subs::subs(arena, c0, ln_w, logw);

    tracing::debug!("gruntz::mrv_leadterm: done");
    Ok((c0, e0))
}

/// Compute `lim(x→∞) e` using the Gruntz algorithm: a finite value or
/// `±∞` (see [`limitinf_directed`] for the limit as a value).
pub(crate) fn limitinf(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    limitinf_directed(arena, e, x, depth, budget, false)
}

/// [`limitinf`]; with `directed` (the limit itself, not a step of the
/// algorithm) an infinite limit whose leading coefficient is not real is
/// the directed infinity of [`directed_infinity`] (`d·∞`, or `±i·∞`),
/// where it was refused ("cannot determine the sign of 1 + I" for
/// `x + i·x`).  The steps of the algorithm (comparisons, the limits of
/// exponents and arguments) keep asking for finite or real infinite
/// values: `e^{(1 + i)x}` has no limit.
fn limitinf_directed(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
    directed: bool,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    if depth > MAX_DEPTH {
        tracing::warn!(depth, "gruntz::limitinf: max recursion depth exceeded");
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::limitinf",
            reason: "maximum recursion depth exceeded".into(),
        });
    }
    budget.tick(1)?;

    // Simplify
    let e = crate::transforms::eval::eval(arena, e);

    // Base case: doesn't depend on x
    if !crate::base::walk::contains(arena, e, x) {
        tracing::trace!("gruntz::limitinf: constant → returning as-is");
        return Ok(e);
    }

    // Rewrite into a "tractable" form: tan → sin/cos, hyperbolic → exp,
    // inverse hyperbolic → ln, and resolve sign-dependent functions
    // (abs, sign, H, floor, piecewise, min, max) by their eventual sign.
    let e = rewrite_tractable(arena, e, x, depth, budget)?;
    if !crate::base::walk::contains(arena, e, x) {
        let v = crate::transforms::eval::eval(arena, e);
        return Ok(v);
    }

    let e_display = arena.display(e).to_string();
    let x_display = arena.display(x).to_string();
    tracing::debug!(depth, expr = %e_display, var = %x_display, "gruntz::limitinf: computing limit at infinity");

    // Compute leading term
    let (c0, e0) = mrv_leadterm(arena, e, x, depth, budget, true)?;
    let e0_eval = crate::transforms::eval::eval(arena, e0);
    let e0_eval = hidden_rational(arena, e0_eval).unwrap_or(e0_eval);

    let c0_display = arena.display(c0).to_string();
    let e0_display = arena.display(e0_eval).to_string();
    tracing::debug!(c0 = %c0_display, e0 = %e0_display, "gruntz::limitinf: leading term extracted, checking exponent sign");

    // A coefficient in `x` that is a sum of quotients, over a common
    // denominator when that is smaller: the phases of `e^{x²/2 + ix}` in
    // `tanh(ix + x²/2)` leave `i·sin 2x/D + cos 2x/D` with `D = i·sin 2x +
    // cos 2x`, which is `1`.  Up to 0.51 its sign was sought as that of a
    // bounded oscillation and `limit(Γ(x)·tanh(ix + x²/2), x, ∞)` was
    // refused (0.48: `∞`).
    let c0 = common_denominator_if_smaller(arena, c0, x);

    // f ≈ 0·ω^e0 → the expression vanishes identically.
    if arena.is_zero_structural(c0) {
        return Ok(arena.zero());
    }
    // An infinite or undefined leading coefficient (a zero taken for a
    // non-zero one further down, `1/(−sin π)`) decides nothing.
    if is_infinite(arena, c0) || contains_singular_atom(arena, c0) {
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::limitinf",
            reason: format!(
                "the leading coefficient {} is not finite",
                arena.display(c0)
            ),
        });
    }

    // Determine sign of e0
    let e0_sign = if let Some(r) = arena.as_num(e0_eval) {
        let r = r.clone();
        if r.is_positive() {
            1
        } else if r.is_negative() {
            -1
        } else {
            0
        }
    } else {
        sign_at_inf(arena, e0_eval, x, depth + 1, budget)?
    };

    // A coefficient that still contains ω (or another dummy) is a bounded
    // oscillation marker (`sin`/`cos` of a divergent argument).  Multiplied
    // by something that vanishes it still gives 0; otherwise there is no
    // limit — refuse rather than leak internal symbols into the answer.
    if contains_foreign_dummy(arena, c0, x) {
        if e0_sign > 0 {
            tracing::info!("gruntz::limitinf: bounded × vanishing → limit is 0");
            return Ok(arena.zero());
        }
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::limitinf",
            reason: "expression has no limit (bounded oscillation)".into(),
        });
    }

    match e0_sign {
        s if s > 0 => {
            tracing::info!("gruntz::limitinf: e0 > 0 → limit is 0");
            Ok(arena.zero())
        }
        s if s < 0 => {
            let c0_sign = match sign_at_inf(arena, c0, x, depth + 1, budget) {
                Ok(s) => s,
                Err(err) if directed && !is_budget_error(&err) => {
                    if let Some(v) = directed_infinity(arena, c0, x, depth + 1, budget) {
                        return Ok(v);
                    }
                    // A leading coefficient in the parameters of unknown sign
                    // (`a` in `a·x`, `y` in `y/(x·ln x)` at `0⁺`): the
                    // infinity in its direction, `sign(c₀)·∞`, generic in the
                    // parameters as every limit here (`limit_dir` adds the
                    // case `c₀ = 0`).  Before: refused (SymPy `oo*sign(a)`).
                    // A coefficient `k·g` with `k` such a constant and `g`
                    // of known eventual sign: `sign(k)·sign(g)·∞`
                    // (`−y/x` for `y/(x·ln x)` at `0⁺`).
                    let (k, g) = match arena.node(c0).clone() {
                        ExprNode::Mul(cs) => {
                            let (ks, gs): (Vec<ExprId>, Vec<ExprId>) = cs
                                .iter()
                                .partition(|&&c| !crate::base::walk::contains(arena, c, x));
                            (arena.mul(&ks), arena.mul(&gs))
                        }
                        _ if !crate::base::walk::contains(arena, c0, x) => (c0, arena.one()),
                        _ => return Err(err),
                    };
                    if !contains_foreign_dummy(arena, k, x)
                        && !crate::base::walk::free_symbols(arena, k).is_empty()
                        && !contains_singular_atom(arena, k)
                        && !contains_foreign_dummy(arena, g, x)
                    {
                        let gs = if g == arena.one() {
                            1
                        } else {
                            match sign_at_inf(arena, g, x, depth + 1, budget) {
                                Ok(s) if s != 0 => s,
                                _ => return Err(err),
                            }
                        };
                        let s = arena.sign(k);
                        let gs = arena.int(i64::from(gs));
                        let v = arena.mul(&[gs, s, arena.infinity()]);
                        return Ok(crate::transforms::eval::eval(arena, v));
                    }
                    return Err(err);
                }
                Err(err) => return Err(err),
            };
            tracing::info!(c0_sign, "gruntz::limitinf: e0 < 0 → limit is ±∞");
            if c0_sign > 0 {
                Ok(arena.infinity())
            } else if c0_sign < 0 {
                Ok(arena.neg_infinity())
            } else {
                Err(crate::base::errors::SymplexError::ComputationFailed {
                    operation: "gruntz::limitinf",
                    reason: "indeterminate 0·∞ form in leading term".into(),
                })
            }
        }
        _ => {
            tracing::debug!("gruntz::limitinf: e0 = 0 → recursing on coefficient c0");
            limitinf_directed(arena, c0, x, depth + 1, budget, directed)
        }
    }
}

/// The infinite limit of `f ≈ c₀·ω^{e₀}` (`e₀ < 0`, `ω → 0⁺`) for a
/// leading coefficient `c₀` whose sign [`sign_at_inf`] cannot give because
/// it is not real: `f/|f| → d/|d|` for the constant `d` reached by
/// following leading coefficients (`c₀ ≈ c₁·ω₁^{e₁}`, `ω₁ > 0`, …), as
/// `sign_at_inf` does for a real `c₀`.  The value is written as limits are
/// (see `limit.rs`): `±∞` and `±i·∞` on the axes (the caller completes
/// the perpendicular component), `d·∞` off them (`x + i·x → (1 + i)·∞`,
/// `e^{x − i} − eˣ → (e^{−i} − 1)·∞`; SymPy: `oo*sign(1 + I)`).  `None`
/// when `d` is not a constant whose real and imaginary parts have certified
/// signs, not both zero.
fn directed_infinity(
    arena: &mut Arena,
    c0: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Option<ExprId> {
    let mut d = crate::transforms::eval::eval(arena, c0);
    let mut depth = depth;
    while crate::base::walk::contains(arena, d, x) {
        if depth > MAX_DEPTH {
            return None;
        }
        // `ω^{e}` keeps its direction only for a real `e`: `ω^{3i}` turns.
        let (c, e) = mrv_leadterm(arena, d, x, depth + 1, budget, false).ok()?;
        if contains_foreign_dummy(arena, c, x)
            || arena.is_zero_structural(c)
            || exponent_turns(arena, e, x, depth, budget)
        {
            return None;
        }
        d = crate::transforms::eval::eval(arena, c);
        depth += 1;
    }
    if !crate::base::walk::free_symbols(arena, d).is_empty()
        || crate::base::walk::has_unevaluated(arena, d)
        || contains_singular_atom(arena, d)
    {
        return None;
    }
    let signs = crate::calculus::limit::direction_signs(arena, d)?;
    let oo = arena.infinity();
    let i = arena.i_unit();
    let v = match (signs.re, signs.im) {
        (0, 0) => return None,
        (1, 0) => oo,
        (_, 0) => arena.neg_infinity(),
        (0, s) => {
            let unit = if s > 0 { i } else { arena.neg(i) };
            arena.mul(&[unit, oo])
        }
        _ => arena.mul(&[d, oo]),
    };
    tracing::debug!(direction = %arena.display(d), value = %arena.display(v), "gruntz::directed_infinity");
    Some(crate::transforms::eval::eval(arena, v))
}

// ═══════════════════════════════════════════════════════════════════════════
// Tractable rewriting
// ═══════════════════════════════════════════════════════════════════════════

/// The functions [`needs_tractable_rewrite`] always rewrites: `tan`, the
/// hyperbolic functions and their inverses, and the piecewise-defined
/// `|·|`, `sign`, `H`, `δ`, `⌊·⌋`, `⌈·⌉`, `min`, `max`, `atan2`, `arg`.
fn always_rewritten(f: Func) -> bool {
    f == Func::Tan
        || matches!(
            f.elementary(),
            Some(Elementary::Hyperbolic | Elementary::InverseHyperbolic)
        )
        || matches!(
            f,
            Func::Abs
                | Func::Sign
                | Func::Heaviside
                | Func::DiracDelta
                | Func::Floor
                | Func::Ceiling
                | Func::Min
                | Func::Max
                | Func::Atan2
                | Func::Arg
        )
}

/// `true` if `e` contains a node that [`rewrite_tractable`] would touch
/// (including a branch function of an argument that may leave the reals,
/// which [`resolve_branch_cut`] continues across its cut).
fn needs_tractable_rewrite(arena: &Arena, e: ExprId, x: ExprId) -> bool {
    let mut stack = vec![e];
    let mut visited = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        if let Some((arg, _)) = crate::calculus::limit::branch_cut_of(arena, id)
            && crate::base::walk::contains(arena, arg, x)
            && may_leave_reals(arena, arg)
        {
            return true;
        }
        let node = arena.node(id);
        if special_asymptotic_arg(arena, node, x)
            .is_some_and(|u| crate::base::walk::contains(arena, u, x))
        {
            return true;
        }
        if node.func_head().is_some_and(always_rewritten) || matches!(node, ExprNode::Piecewise(_))
        {
            return true;
        }
        if matches!(node, ExprNode::Binomial(..)) && crate::base::walk::contains(arena, id, x) {
            return true;
        }
        if let ExprNode::Lib(
            crate::base::libfn::LibFn::Fibonacci | crate::base::libfn::LibFn::Lucas,
            args,
        ) = node
            && args
                .iter()
                .any(|&a| crate::base::walk::contains(arena, a, x))
        {
            return true;
        }
        if matches!(
            node,
            ExprNode::Erf(_)
                | ExprNode::Erfc(_)
                | ExprNode::Ei(_)
                | ExprNode::Li(_)
                | ExprNode::Gamma(_)
                | ExprNode::Factorial(_)
        ) {
            return true;
        }
        if let ExprNode::Pow(b, _) = node
            && arena.is_zero_structural(*b)
        {
            return true;
        }
        if let ExprNode::Exp(a) = node
            && !exp_log_terms(arena, *a).0.is_empty()
        {
            return true;
        }
        // A power with a varying exponent may have a non-real or negative
        // base, an exponential a non-real exponent ([`phase_split`]).
        if let ExprNode::Pow(b, g) = node
            && crate::base::walk::contains(arena, *g, x)
            && !obviously_positive(arena, *b, x)
        {
            return true;
        }
        if let ExprNode::Exp(a) = node
            && crate::base::walk::contains(arena, *a, x)
            && may_leave_reals(arena, *a)
        {
            return true;
        }
        node.for_each_child(|c| stack.push(c));
    }
    false
}

/// `e^g` with the phase of a non-real exponent split off when its
/// imaginary part diverges: `e^{Re g}·(cos(Im g) + i·sin(Im g))`, the
/// factor in parentheses a bounded oscillation (`x` is real, `x → +∞`).
/// `None` when `g` is eventually real or its imaginary part converges
/// (`e^{x + 2i}` keeps its direction `e^{2i}`).
///
/// The Gruntz algorithm orders real growth: it took `e^{x·ln(−x)} =
/// e^{x ln x}·e^{iπx}` for the growth class of `e^{x ln x}` and found
/// `lim (−x)^x = ∞` (it has no direction: the phase `πx` turns; SymPy
/// `zoo`), the same for `(1 − x/2)^{3x}`; and it refused `(−1/4)^x`
/// ("cannot determine the sign of πi + ln(1/4)"), which tends to `0`:
/// `4^{−x}` times a bounded oscillation.
fn phase_split(
    arena: &mut Arena,
    g: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<Option<ExprId>, crate::base::errors::SymplexError> {
    let (gr, gi) = arena.as_real_imag_expr(g);
    let gi = crate::transforms::eval::eval(arena, gi);
    if !crate::base::walk::contains(arena, gi, x)
        || walk_has(arena, gi, |n| {
            matches!(n, ExprNode::Re(_) | ExprNode::Im(_))
        })
    {
        return Ok(None);
    }
    match limitinf(arena, gi, x, depth + 1, budget) {
        Ok(l) if l == arena.infinity() || l == arena.neg_infinity() => {}
        Err(e) if is_budget_error(&e) => return Err(e),
        _ => return Ok(None),
    }
    let gr = crate::transforms::eval::eval(arena, gr);
    let modulus = arena.exp(gr);
    let c = arena.cos(gi);
    let s = arena.sin(gi);
    let is = arena.mul(&[arena.i_unit(), s]);
    let phase = arena.add(&[c, is]);
    Ok(Some(arena.mul(&[modulus, phase])))
}

/// `expr` rewritten by the tractable rewrite for `var → +∞` (`−∞` when
/// `negative`) — the special functions of a divergent argument in their
/// asymptotic forms with internal heads the series engine expands (see
/// [`special_asymptotic_rewrite`]) — for the asymptotic series at
/// infinity; `None` when nothing changes.
pub(crate) fn tractable_at_infinity(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    negative: bool,
) -> Option<ExprId> {
    let x = arena.positive_symbol("__gzt_x");
    let along = if negative { arena.neg(x) } else { x };
    let e = crate::transforms::subs::subs(arena, expr, var, along);
    let e = crate::transforms::eval::eval(arena, e);
    let mut budget = Budget::new();
    let r = rewrite_tractable(arena, e, x, 0, &mut budget).ok()?;
    // `e^{u}·e^{−u}` (from the rewrite, or `Γ(2, x)·eˣ = (x + 1)·e^{−x}·eˣ`)
    // cancels only once combined.
    let r = crate::transforms::expand::expand(arena, r);
    let r = crate::transforms::eval::eval(arena, r);
    let r = combine_exponentials(arena, r, x);
    if r == e {
        return None;
    }
    let back = if negative { arena.neg(var) } else { var };
    let r = crate::transforms::subs::subs(arena, r, x, back);
    Some(crate::transforms::eval::eval(arena, r))
}

/// A base that is positive by its form (a positive number or constant, the
/// positive variable `x`, an exponential of a real argument): no phase.
fn obviously_positive(arena: &Arena, b: ExprId, x: ExprId) -> bool {
    match arena.node(b) {
        ExprNode::Num(_) => arena.as_num(b).is_some_and(|r| r.is_positive()),
        ExprNode::Pi
        | ExprNode::E
        | ExprNode::EulerGamma
        | ExprNode::Catalan
        | ExprNode::GoldenRatio => true,
        ExprNode::Exp(a) => !may_leave_reals(arena, *a),
        _ => b == x,
    }
}

/// `|u|` for a power `u = b^g` with an eventually real exponent (`|b|^g`)
/// or an exponential `u = e^g` (`e^{Re g}`), whose phase may turn; `None`
/// for any other `u`.
fn modulus_of_power(
    arena: &mut Arena,
    u: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Option<ExprId> {
    let node = arena.node(u).clone();
    match node {
        ExprNode::Pow(b, g)
            if crate::base::walk::contains(arena, g, x)
                && eventually_real(arena, g, x, depth + 1, budget)
                && !eventually_real(arena, u, x, depth + 1, budget) =>
        {
            let ab = arena.abs(b);
            Some(arena.pow(ab, g))
        }
        // `|b^g| = b^{Re g}` for an eventually positive real base (`|x^{ln(−2)
        // + x/2}| = x^{x/2 + ln 2}`).
        ExprNode::Pow(b, g)
            if crate::base::walk::contains(arena, g, x)
                && !eventually_real(arena, g, x, depth + 1, budget)
                && eventually_real(arena, b, x, depth + 1, budget)
                && matches!(sign_at_inf(arena, b, x, depth + 1, budget), Ok(1)) =>
        {
            let (gr, _) = arena.as_real_imag_expr(g);
            let gr = crate::transforms::eval::eval(arena, gr);
            (!walk_has(arena, gr, |n| {
                matches!(n, ExprNode::Re(_) | ExprNode::Im(_))
            }))
            .then(|| arena.pow(b, gr))
        }
        ExprNode::Exp(g)
            if crate::base::walk::contains(arena, g, x) && may_leave_reals(arena, g) =>
        {
            let (gr, _) = arena.as_real_imag_expr(g);
            let gr = crate::transforms::eval::eval(arena, gr);
            (!walk_has(arena, gr, |n| {
                matches!(n, ExprNode::Re(_) | ExprNode::Im(_))
            }))
            .then(|| arena.exp(gr))
        }
        _ => None,
    }
}

/// Does `c` depend on `w` only through `sin`/`cos` factors (bounded
/// oscillation markers), combined by sums, products and non-negative
/// integer powers?
fn only_bounded_markers(arena: &Arena, c: ExprId, w: ExprId) -> bool {
    let mut stack = vec![c];
    while let Some(id) = stack.pop() {
        if !crate::base::walk::contains(arena, id, w) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Sin(_) | ExprNode::Cos(_) => {}
            ExprNode::Add(cs) | ExprNode::Mul(cs) => stack.extend(cs.iter().copied()),
            ExprNode::Neg(a) => stack.push(*a),
            ExprNode::Pow(b, e)
                if arena
                    .as_num(*e)
                    .is_some_and(|r| r.is_integer() && !r.is_negative()) =>
            {
                stack.push(*b)
            }
            _ => return false,
        }
    }
    true
}

/// Does any node of `e` satisfy `pred`?
fn walk_has(arena: &Arena, e: ExprId, pred: impl Fn(&ExprNode) -> bool) -> bool {
    crate::base::walk::post_order_ids(arena, e)
        .into_iter()
        .any(|id| pred(arena.node(id)))
}

/// The analytic functions real at every real argument: `exp`, the
/// trigonometric and hyperbolic functions, `atan`, `asinh`, `erf`, `erfc`
/// (`eventually_real` reads them through their argument).
fn real_analytic_on_reals(f: Func) -> bool {
    matches!(
        f.elementary(),
        Some(Elementary::Trig | Elementary::Hyperbolic | Elementary::Exp)
    ) || matches!(f, Func::Atan | Func::Asinh | Func::Erf | Func::Erfc)
}

/// The functions real at every real argument that [`may_leave_reals`]
/// lets through: [`real_analytic_on_reals`] and `|·|`, `sign`, `H`,
/// `⌊·⌋`, `⌈·⌉`.
fn real_on_reals(f: Func) -> bool {
    real_analytic_on_reals(f)
        || matches!(
            f,
            Func::Abs | Func::Sign | Func::Heaviside | Func::Floor | Func::Ceiling
        )
}

/// The one-argument functions `expand_functions_as_series` expands at an
/// ω-dependent argument: the elementary functions but `acos`, `acosh`, and
/// `erf`.
fn expanded_as_series(f: Func) -> bool {
    (f.elementary().is_some() && !matches!(f, Func::Acos | Func::Acosh)) || f == Func::Erf
}

/// Can `e` take non-real values for real values of its symbols?  `false`
/// when it is built only from numbers, real constants, symbols, sums,
/// products, integer powers and functions that are real on the reals
/// (`exp`, the trigonometric and hyperbolic functions, `atan`, `asinh`,
/// `|·|`, `erf`, …) — a cheap structural screen before the branch-cut
/// analysis of [`resolve_branch_cut`].
fn may_leave_reals(arena: &Arena, e: ExprId) -> bool {
    let mut stack = vec![e];
    let mut visited = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        let node = arena.node(id);
        let real_preserving = match node {
            ExprNode::Pow(_, g) => arena.as_num(*g).is_some_and(|r| r.is_integer()),
            _ => {
                matches!(
                    node,
                    ExprNode::Num(_)
                        | ExprNode::Pi
                        | ExprNode::E
                        | ExprNode::EulerGamma
                        | ExprNode::Catalan
                        | ExprNode::GoldenRatio
                        | ExprNode::Symbol(_)
                        | ExprNode::Add(_)
                        | ExprNode::Mul(_)
                        | ExprNode::Neg(_)
                ) || node.func_head().is_some_and(real_on_reals)
            }
        };
        if !real_preserving {
            return true;
        }
        node.for_each_child(|c| stack.push(c));
    }
    false
}

/// The terms `k·ln u` (`k` a rational number) of the exponent `a`, as
/// `(u, k)`, and the remaining terms.
fn exp_log_terms(arena: &Arena, a: ExprId) -> (Vec<(ExprId, ExprId)>, Vec<ExprId>) {
    let terms: Vec<ExprId> = match arena.node(a) {
        ExprNode::Add(cs) => cs.to_vec(),
        _ => vec![a],
    };
    let mut logs = Vec::new();
    let mut rest = Vec::new();
    for t in terms {
        match arena.node(t) {
            ExprNode::Ln(u) => logs.push((*u, arena.one())),
            ExprNode::Mul(cs)
                if cs.len() == 2
                    && arena.as_num(cs[0]).is_some()
                    && matches!(arena.node(cs[1]), ExprNode::Ln(_)) =>
            {
                if let ExprNode::Ln(u) = arena.node(cs[1]) {
                    logs.push((*u, cs[0]));
                }
            }
            _ => rest.push(t),
        }
    }
    (logs, rest)
}

/// Rewrite `e` into a form the Gruntz machinery handles well, using the
/// fact that `x → +∞`:
///
/// * `tan u → sin u / cos u`
/// * `sinh u, cosh u, tanh u → ` exponentials
/// * `asinh u, acosh u, atanh u → ` logarithms
/// * `|u| → ±u`, `sign u → ±1`, `H(u) → 0/1`, `δ(u) → 0` by the eventual
///   sign of `u`
/// * `⌊u⌋, ⌈u⌉ → n` when `u` converges to a finite value
/// * `min/max → ` the eventually smallest/largest argument
/// * `Piecewise → ` the branch whose condition eventually holds
/// * a branch function whose argument tends to a point of its cut → its
///   continuation across the cut from the side the argument approaches
///   from ([`resolve_branch_cut`]); the continuation's own functions are
///   rewritten by a further pass
///
/// Nodes whose eventual sign cannot be determined are left untouched.
fn rewrite_tractable(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    let mut e = e;
    for _ in 0..MAX_TRACTABLE_PASSES {
        let (r, continued) = rewrite_tractable_pass(arena, e, x, depth, budget)?;
        e = r;
        if !continued {
            break;
        }
    }
    resolve_negative_leading(arena, e, x, depth, budget)
}

/// `ln u` or a fractional power `u^r` whose argument `u` is not real and
/// has a negative real leading coefficient while it tends to `−∞` or `0`
/// (`1 − x − i`, `−1/x + i/x²`), continued across the cut from the side
/// `u` approaches from: `ln u = ln(−u) ± iπ`, `u^r = (−u)^r·e^{±iπr}` with
/// the eventual sign of `Im u` ([`cut_side`]; `+` for a real `u`, on the
/// cut).  The leading term `c·ωᵉ` with `c < 0` keeps no side, and
/// `leadterm` took the principal `ln c = ln|c| + iπ` from both sides:
/// `limit(ln(−x − i) − ln x, x, ∞)` was `iπ` (it is `−iπ`; mpmath
/// `log(−10²⁰ − i)`), and `atanh(x + i)`, rewritten as
/// `(ln(1 + x + i) − ln(1 − x − i))/2`, tended to `−iπ/2` (it is `+iπ/2`).
/// (`resolve_branch_cut` handles arguments tending to a finite point of
/// a cut.)  `Err` when the side is undecided.
fn resolve_negative_leading(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    let candidate = |arena: &Arena, id: ExprId| -> Option<ExprId> {
        let u = match *arena.node(id) {
            ExprNode::Ln(u) => u,
            ExprNode::Pow(u, r) if arena.as_num(r).is_some_and(|q| !q.is_integer()) => u,
            _ => return None,
        };
        (crate::base::walk::contains(arena, u, x) && may_leave_reals(arena, u)).then_some(u)
    };
    let post = crate::base::walk::post_order_ids(arena, e);
    if !post.iter().any(|&id| candidate(arena, id).is_some()) {
        return Ok(e);
    }
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    let mut changed = false;
    for &id in &post {
        if !crate::base::walk::contains(arena, id, x) {
            cache.insert(id, id);
            continue;
        }
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match candidate(arena, rebuilt) {
            Some(u) => match negative_leading_side(arena, u, x, depth, budget)? {
                Some(side) => {
                    let i_pi = arena.mul(&[arena.i_unit(), arena.pi()]);
                    let shift = if side >= 0 { i_pi } else { arena.neg(i_pi) };
                    let neg_u = arena.neg(u);
                    match arena.node(rebuilt).clone() {
                        ExprNode::Pow(_, r) => {
                            let p = arena.pow(neg_u, r);
                            let phase = arena.mul(&[shift, r]);
                            let rot = arena.exp(phase);
                            arena.mul(&[p, rot])
                        }
                        _ => {
                            let ln = arena.ln(neg_u);
                            arena.add(&[ln, shift])
                        }
                    }
                }
                None => rebuilt,
            },
            None => rebuilt,
        };
        changed |= new != rebuilt;
        cache.insert(id, new);
    }
    if !changed {
        return Ok(e);
    }
    let result = cache.get(&e).copied().unwrap_or(e);
    let result = crate::transforms::eval::eval(arena, result);
    tracing::debug!(
        rewritten = %arena.display(result).to_string(),
        "gruntz: ln/power of a non-real argument with a negative leading term continued"
    );
    Ok(result)
}

/// The side for [`resolve_negative_leading`]: `Some(s)` (the eventual sign
/// of `Im u`, `0` on the cut) when `u` is not eventually real, its leading
/// coefficient is a negative real number and `u → −∞` or `u → 0`; `None`
/// when the function needs no continuation; `Err` for an undecided side.
fn negative_leading_side(
    arena: &mut Arena,
    u: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<Option<i32>, crate::base::errors::SymplexError> {
    if eventually_real(arena, u, x, depth + 1, budget) {
        return Ok(None);
    }
    match sign_at_inf(arena, u, x, depth + 1, budget) {
        Ok(-1) => {}
        Err(e) if is_budget_error(&e) => return Err(e),
        _ => return Ok(None),
    }
    let l = match limitinf(arena, u, x, depth + 1, budget) {
        Ok(l) => l,
        Err(e) if is_budget_error(&e) => return Err(e),
        Err(_) => return Ok(None),
    };
    if l != arena.neg_infinity() && !arena.is_zero_structural(l) {
        return Ok(None);
    }
    let zero = arena.zero();
    match cut_side(
        arena,
        u,
        zero,
        crate::calculus::limit::BranchCut::NegativeReals,
        x,
        depth,
        budget,
    ) {
        Some(s) => Ok(Some(s)),
        None => Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::resolve_negative_leading",
            reason: "a non-real argument approaches the cut of ln from an undecided side".into(),
        }),
    }
}

/// Most passes of [`rewrite_tractable`]: one more after each pass that
/// continued a function across its cut (whose continuation has functions
/// of its own to rewrite, `acosh(−u) + iπ`).
const MAX_TRACTABLE_PASSES: usize = 3;

/// One pass of [`rewrite_tractable`]; `true` when a function was continued
/// across its branch cut.
fn rewrite_tractable_pass(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<(ExprId, bool), crate::base::errors::SymplexError> {
    if !needs_tractable_rewrite(arena, e, x) {
        return Ok((e, false));
    }

    let post = crate::base::walk::post_order_ids(arena, e);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    let mut continued = false;

    for &id in &post {
        if !crate::base::walk::contains(arena, id, x) {
            cache.insert(id, id);
            continue;
        }
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let node = arena.node(rebuilt).clone();
        if let Some(g) = resolve_branch_cut(arena, &node, rebuilt, x, depth, budget)? {
            continued = true;
            cache.insert(id, g);
            continue;
        }
        // `|e^g| = e^{Re g}` and `|b^g| = |b|^g` for a real `g` (the modulus
        // of a power whose phase [`phase_split`] turns): `|(−x/(1 + x))^x|`
        // tends to `e^{−1}` at `∞` (SymPy; before: refused).  Taken from the
        // argument before this pass rewrote it.
        if let ExprNode::Abs(u) = *arena.node(id)
            && let Some(m) = modulus_of_power(arena, u, x, depth, budget)
        {
            continued = true;
            cache.insert(id, m);
            continue;
        }
        let new = match node {
            ExprNode::Tan(u) => {
                let s = arena.sin(u);
                let c = arena.cos(u);
                arena.div(s, c)
            }
            ExprNode::Sinh(u) => {
                let neg_u = arena.neg(u);
                let ep = arena.exp(u);
                let em = arena.exp(neg_u);
                let diff = arena.sub(ep, em);
                let two = arena.int(2);
                arena.div(diff, two)
            }
            ExprNode::Cosh(u) => {
                let neg_u = arena.neg(u);
                let ep = arena.exp(u);
                let em = arena.exp(neg_u);
                let sum = arena.add(&[ep, em]);
                let two = arena.int(2);
                arena.div(sum, two)
            }
            ExprNode::Tanh(u) => {
                let two = arena.int(2);
                let two_u = arena.mul(&[two, u]);
                let e2u = arena.exp(two_u);
                let one = arena.one();
                let num = arena.sub(e2u, one);
                let den = arena.add(&[e2u, one]);
                arena.div(num, den)
            }
            ExprNode::Asinh(u) => {
                let two = arena.int(2);
                let u2 = arena.pow(u, two);
                let one = arena.one();
                let inner = arena.add(&[u2, one]);
                let root = arena.sqrt(inner);
                let sum = arena.add(&[u, root]);
                arena.ln(sum)
            }
            // `acosh u = ln(u + √(u + 1)·√(u − 1))` on all of ℂ (SymPy's
            // `acosh._eval_rewrite_as_log`).  Up to 0.33 this was
            // `ln(u + √(u² − 1))`, which is `acosh u` only for `Re u > 0`: for
            // `u → −∞` it is `−acosh(−u)`, and `limit(acosh(x) − ln(−x), x,
            // −∞)` came out `−∞` (it is `ln 2 + iπ`); at `u = acosh 0 = iπ/2`
            // the radicand `u² − 1` is a negative real the evaluator cannot
            // place on a side of the cut, and the nightly `fuzz_calculus`
            // limit `x/(acosh(acosh x) − acosh(acosh 0))` at `0⁺` came out `0`
            // (it is `−i·√(iπ/2 − 1)·√(iπ/2 + 1) ≈ −1.862`, 2026-10-03).
            ExprNode::Acosh(u) => {
                let one = arena.one();
                let up = arena.add(&[u, one]);
                let um = arena.sub(u, one);
                let rp = arena.sqrt(up);
                let rm = arena.sqrt(um);
                let root = arena.mul(&[rp, rm]);
                let sum = arena.add(&[u, root]);
                arena.ln(sum)
            }
            ExprNode::Atanh(u) => {
                let one = arena.one();
                let p = arena.add(&[one, u]);
                let m = arena.sub(one, u);
                let lp = arena.ln(p);
                let lm = arena.ln(m);
                let diff = arena.sub(lp, lm);
                let two = arena.int(2);
                arena.div(diff, two)
            }
            // At ±∞ the error functions and `Ei`, `li` separate their
            // exponential factor from an internal function with a power
            // series in `1/z` (SymPy's `_erfs`, `_eis` in the "tractable"
            // rewrite): `erfc z = e^{−z²}·erfcx(z)`, `Ei z = e^z·(e^{−z} Ei z)`.
            // Before, `erfc(x)·x·e^{x²}` had no limit at `∞` (it is `1/√π`).
            // `Γ(z) = exp(ln Γ(z))` for `z → +∞` (SymPy's tractable rewrite):
            // Stirling's series of `ln Γ` is known to the series engine.
            ExprNode::Gamma(u) | ExprNode::Factorial(u) => {
                match limitinf(arena, u, x, depth + 1, budget) {
                    Ok(l) if l == arena.infinity() => {
                        let z = if matches!(node, ExprNode::Factorial(_)) {
                            let one = arena.one();
                            arena.add(&[u, one])
                        } else {
                            u
                        };
                        let lg = arena.log_gamma(z);
                        arena.exp(lg)
                    }
                    Err(e) if is_budget_error(&e) => return Err(e),
                    _ => rebuilt,
                }
            }
            // `Ei` of a non-real argument whose real part tends to `−∞`: its
            // value tends to `±iπ` (the side of the cut), which the
            // asymptotic series of `e^{−z} Ei(z)` does not carry; before,
            // `limit(Ei(−x + i), x, ∞)` was `0` (it is `iπ`; mpmath
            // `ei(−10 + i) = −1.9·10⁻⁶ + 3.14159i`).
            ExprNode::Ei(u) | ExprNode::Li(u) => match limitinf(arena, u, x, depth + 1, budget) {
                Ok(l) if l == arena.infinity() => {
                    asymptotic_rewrite(arena, &node, u, true).unwrap_or(rebuilt)
                }
                // `li u = Ei(ln u)` (principal branches; on the cut `(−∞, 0)`
                // both take the values from above) at the branch point `0`,
                // where `ln u → −∞` and the rule for `Ei` applies (in the next
                // pass): `li(−x) → iπ` at `0⁺`, `li x → 0`.
                Ok(l) if matches!(node, ExprNode::Li(_)) && arena.is_zero_structural(l) => {
                    continued = true;
                    let lu = arena.ln(u);
                    arena.ei(lu)
                }
                Ok(l) if l == arena.neg_infinity() => {
                    if eventually_real(arena, u, x, depth + 1, budget) {
                        asymptotic_rewrite(arena, &node, u, false).unwrap_or(rebuilt)
                    } else if let (ExprNode::Ei(_), Some(side)) = (
                        &node,
                        cut_side(
                            arena,
                            u,
                            arena.zero(),
                            crate::calculus::limit::BranchCut::NegativeReals,
                            x,
                            depth,
                            budget,
                        ),
                    ) && side != 0
                    {
                        let r = asymptotic_rewrite(arena, &node, u, false).unwrap_or(rebuilt);
                        let i_pi = arena.mul(&[arena.i_unit(), arena.pi()]);
                        let shift = if side > 0 { i_pi } else { arena.neg(i_pi) };
                        arena.add(&[r, shift])
                    } else {
                        rebuilt
                    }
                }
                Err(e) if is_budget_error(&e) => return Err(e),
                _ => rebuilt,
            },
            ExprNode::Erf(u) | ExprNode::Erfc(u) => {
                match limitinf(arena, u, x, depth + 1, budget) {
                    Ok(l) if l == arena.infinity() || l == arena.neg_infinity() => {
                        asymptotic_rewrite(arena, &node, u, l == arena.infinity())
                            .unwrap_or(rebuilt)
                    }
                    Err(e) if is_budget_error(&e) => return Err(e),
                    _ => rebuilt,
                }
            }
            ExprNode::Zeta(_) | ExprNode::Lib(..)
                if special_asymptotic_arg(arena, &node, x).is_some() =>
            {
                let Some(u) = special_asymptotic_arg(arena, &node, x) else {
                    continue;
                };
                match limitinf(arena, u, x, depth + 1, budget) {
                    Ok(l)
                        if (l == arena.infinity() || l == arena.neg_infinity())
                            && eventually_real(arena, u, x, depth + 1, budget) =>
                    {
                        special_asymptotic_rewrite(arena, &node, u, l == arena.infinity(), e)
                            .unwrap_or(rebuilt)
                    }
                    Err(e) if is_budget_error(&e) => return Err(e),
                    _ => rebuilt,
                }
            }
            // The eventual sign decides |u|, sign u, H(u) only for a real u:
            // `(−2)^(−1/x)` tends to 1 but is not real, and taking it for
            // positive made `lim_{x→0⁺} |(−2)^(−x)|^(1/x)` equal −1/2.
            // `⌊u⌋`, `⌈u⌉`, `δ(u)`, `min`, `max` have no value at all off the
            // reals: `ln(x)^(x²)` tends to 1 at `0⁺` with a positive
            // imaginary part, and `lim ceiling(ln(x)^(x²))` was 2.
            ExprNode::Abs(u) | ExprNode::Sign(u) | ExprNode::Heaviside(u)
                if !eventually_real(arena, u, x, depth + 1, budget) =>
            {
                rebuilt
            }
            ExprNode::Floor(u) | ExprNode::Ceiling(u) | ExprNode::DiracDelta(u)
                if !real_for_generic_parameters(arena, u, x, depth, budget) =>
            {
                rebuilt
            }
            ExprNode::Min(ref args) | ExprNode::Max(ref args)
                if !args
                    .clone()
                    .into_iter()
                    .all(|a| min_max_argument_real(arena, a, x, depth, budget)) =>
            {
                rebuilt
            }
            // A power with a varying exponent and an eventually negative or
            // non-real base, an exponential with a non-real exponent: the
            // phase split off ([`phase_split`]).
            ExprNode::Pow(b, g)
                if crate::base::walk::contains(arena, g, x)
                    && (!eventually_real(arena, b, x, depth + 1, budget)
                        || matches!(sign_at_inf(arena, b, x, depth + 1, budget), Ok(-1))) =>
            {
                let lb = arena.ln(b);
                let ge = arena.mul(&[g, lb]);
                let ge = crate::transforms::eval::eval(arena, ge);
                match phase_split(arena, ge, x, depth, budget)? {
                    Some(r) => {
                        continued = true;
                        r
                    }
                    None => rebuilt,
                }
            }
            ExprNode::Exp(g)
                if crate::base::walk::contains(arena, g, x) && may_leave_reals(arena, g) =>
            {
                match phase_split(arena, g, x, depth, budget)? {
                    Some(r) => {
                        continued = true;
                        r
                    }
                    None => rebuilt,
                }
            }
            ExprNode::Abs(u) => match sign_at_inf(arena, u, x, depth + 1, budget) {
                Ok(s) if s > 0 => u,
                Ok(s) if s < 0 => arena.neg(u),
                _ => rebuilt,
            },

            ExprNode::Sign(u) => match sign_at_inf(arena, u, x, depth + 1, budget) {
                Ok(s) => arena.int(s as i64),
                Err(_) => rebuilt,
            },
            ExprNode::Heaviside(u) => match sign_at_inf(arena, u, x, depth + 1, budget) {
                Ok(s) if s > 0 => arena.one(),
                Ok(s) if s < 0 => arena.zero(),
                _ => rebuilt,
            },
            ExprNode::DiracDelta(u) => match sign_at_inf(arena, u, x, depth + 1, budget) {
                Ok(s) if s != 0 => arena.zero(),
                _ => rebuilt,
            },
            // `C(a, b) = Γ(a + 1)/(Γ(b + 1)·Γ(a − b + 1))`, whose gammas the
            // next pass writes as `exp(ln Γ)` (SymPy's tractable rewrite of
            // `binomial`): `C(n, n/2)`, `16ᵏ/(k·C(2k, k)²) → π`, `n^{−z}·C(n + z, n)
            // → 1/Γ(z + 1)` at `∞` were refused.
            ExprNode::Binomial(a, b) => {
                continued = true;
                let one = arena.one();
                let a1 = arena.add(&[a, one]);
                let b1 = arena.add(&[b, one]);
                let amb = arena.sub(a, b);
                let amb1 = arena.add(&[amb, one]);
                let ga = arena.gamma(a1);
                let gb = arena.gamma(b1);
                let gab = arena.gamma(amb1);
                let den = arena.mul(&[gb, gab]);
                arena.div(ga, den)
            }
            // Binet's formula, extended to real `u` by `cos(πu)` for `(−1)^u`:
            // `F(u) = (φ^u − cos(πu)·φ^{−u})/√5`, `L(u) = φ^u + cos(πu)·φ^{−u}`
            // (SymPy's real extension of `fibonacci`; `F(n + 1)/F(n) → φ`).
            ExprNode::Lib(
                f @ (crate::base::libfn::LibFn::Fibonacci | crate::base::libfn::LibFn::Lucas),
                ref args,
            ) if args.len() == 1 => {
                let u = args[0];
                let phi = arena.golden_ratio();
                let pu = arena.pow(phi, u);
                let nu = arena.neg(u);
                let pnu = arena.pow(phi, nu);
                let piu = arena.mul(&[arena.pi(), u]);
                let c = arena.cos(piu);
                let tail = arena.mul(&[c, pnu]);
                if f == crate::base::libfn::LibFn::Fibonacci {
                    let d = arena.sub(pu, tail);
                    let five = arena.int(5);
                    let s5 = arena.sqrt(five);
                    arena.div(d, s5)
                } else {
                    arena.add(&[pu, tail])
                }
            }
            // `arg u` of an eventually real `u` (the imaginary part of
            // `ln u` in [`phase_split`]): `0` or `π` by its eventual sign.
            ExprNode::Arg(u) if eventually_real(arena, u, x, depth + 1, budget) => {
                match sign_at_inf(arena, u, x, depth + 1, budget) {
                    Ok(1) => arena.zero(),
                    Ok(-1) => arena.pi(),
                    Err(e) if is_budget_error(&e) => return Err(e),
                    _ => rebuilt,
                }
            }
            // `atan2(a, b)` of real arguments (the argument of a complex
            // number in the decomposition `cut_side` uses) by the eventual
            // signs: `atan(a/b)` for `b > 0`, `atan(a/b) ± π` for `b < 0`.
            ExprNode::Atan2(a, b)
                if eventually_real(arena, a, x, depth + 1, budget)
                    && eventually_real(arena, b, x, depth + 1, budget) =>
            {
                let sb = sign_at_inf(arena, b, x, depth + 1, budget);
                let sa = sign_at_inf(arena, a, x, depth + 1, budget);
                let pi = arena.pi();
                let ratio = arena.div(a, b);
                let at = arena.atan(ratio);
                match (sb, sa) {
                    (Ok(1), _) => at,
                    (Ok(-1), Ok(1)) => arena.add(&[at, pi]),
                    (Ok(-1), Ok(0)) => pi,
                    (Ok(-1), Ok(-1)) => arena.sub(at, pi),
                    (Ok(0), Ok(s)) if s != 0 => {
                        let half = arena.rational(s as i64, 2);
                        arena.mul(&[half, pi])
                    }
                    (Err(e), _) | (_, Err(e)) if is_budget_error(&e) => return Err(e),
                    _ => rebuilt,
                }
            }
            ExprNode::Floor(u) | ExprNode::Ceiling(u) => {
                let is_floor = matches!(node, ExprNode::Floor(_));
                match limitinf(arena, u, x, depth + 1, budget) {
                    Ok(l) if !is_infinite(arena, l) => match arena.as_num(l).cloned() {
                        Some(r) if r.is_integer() => {
                            let n_id = l;
                            let diff = arena.sub(u, n_id);
                            match sign_at_inf(arena, diff, x, depth + 1, budget) {
                                Ok(s) => {
                                    let one = arena.one();
                                    if s > 0 {
                                        if is_floor {
                                            n_id
                                        } else {
                                            arena.add(&[n_id, one])
                                        }
                                    } else if s < 0 {
                                        if is_floor { arena.sub(n_id, one) } else { n_id }
                                    } else {
                                        n_id
                                    }
                                }
                                Err(_) => rebuilt,
                            }
                        }
                        Some(r) => {
                            let v = if is_floor { r.floor() } else { r.ceil() };
                            let nid = arena.intern_num(v);
                            arena.intern(ExprNode::Num(nid))
                        }
                        None => rebuilt,
                    },
                    _ => rebuilt,
                }
            }
            ExprNode::Min(ref args) | ExprNode::Max(ref args) => {
                let is_min = matches!(node, ExprNode::Min(_));
                let args = args.clone();
                let mut best = args[0];
                let mut ok = true;
                for &a in &args[1..] {
                    let diff = arena.sub(a, best);
                    match sign_at_inf(arena, diff, x, depth + 1, budget) {
                        Ok(s) => {
                            if (is_min && s < 0) || (!is_min && s > 0) {
                                best = a;
                            }
                        }
                        Err(_) => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok { best } else { rebuilt }
            }
            ExprNode::Piecewise(ref pairs) => {
                let pairs = pairs.clone();
                let mut chosen = None;
                for &(val, cond) in &pairs {
                    match eventually_true(arena, cond, x, depth, budget) {
                        Truth::True => {
                            chosen = Some(val);
                            break;
                        }
                        Truth::False => continue,
                        Truth::Unknown => break,
                    }
                }
                chosen.unwrap_or(rebuilt)
            }
            _ => rebuilt,
        };
        cache.insert(id, new);
    }

    let result = cache.get(&e).copied().unwrap_or(e);
    let result = crate::transforms::eval::eval(arena, result);
    let result = simplify_exp_log(arena, result);
    let result = resolve_zero_powers(arena, result, x, depth, budget)?;
    if result != e {
        let r_display = arena.display(result).to_string();
        tracing::debug!(rewritten = %r_display, "gruntz::rewrite_tractable");
    }
    Ok((result, continued))
}

/// `f(z)` of an error function, `Ei` or `li` with `z → +∞` (`positive`) or
/// `−∞`, written with its exponential factor explicit and an internal
/// function ([`InternalFn::ErfcxAsym`], [`InternalFn::EiAsym`]) whose
/// asymptotic series the series engine knows:
/// `erfc z = e^{−z²}E(z)`, `erfc z = 2 − e^{−z²}E(−z)`, `erf z = 1 − erfc z`,
/// `Ei z = e^z F(z)`, `li z = Ei(ln z) = z·F(ln z)` (`z → +∞` only).
///
/// Follows SymPy's `_eval_rewrite_as_tractable` of `erf`, `erfc`, `Ei`
/// and the helper functions `_erfs`, `_eis` in
/// `sympy/functions/special/error_functions.py` (BSD-3-Clause; notice in
/// `THIRD-PARTY-NOTICES.md`); the asymptotic series are DLMF 7.12.1 and
/// 6.12.2.
fn asymptotic_rewrite(
    arena: &mut Arena,
    node: &ExprNode,
    u: ExprId,
    positive: bool,
) -> Option<ExprId> {
    let internal =
        |arena: &mut Arena, f: InternalFn, arg: ExprId| arena.func(f.into(), &[arg]).ok();
    match *node {
        ExprNode::Erf(_) | ExprNode::Erfc(_) => {
            let z = if positive { u } else { arena.neg(u) };
            let two = arena.int(2);
            let u2 = arena.pow(u, two);
            let nu2 = arena.neg(u2);
            let damp = arena.exp(nu2);
            // `erfc(√x)` (and `Γ(1/2, x) = √π·erfc(√x)`): `E(z)` would have a
            // series in `x^{−1/2}`; `E₂(z²)/z` has one in `1/x`.
            let u2e = crate::transforms::eval::eval(arena, u2);
            let e = if has_fractional_power(arena, u) && !has_fractional_power(arena, u2e) {
                let e2 = internal(arena, InternalFn::Erfcx2Asym, u2e)?;
                let inv = arena.pow(z, arena.neg_one());
                arena.mul(&[e2, inv])
            } else {
                internal(arena, InternalFn::ErfcxAsym, z)?
            };
            // t = erfc(z), z → +∞
            let t = arena.mul(&[damp, e]);
            let one = arena.one();
            let r = match (matches!(node, ExprNode::Erfc(_)), positive) {
                (true, true) => t,
                (true, false) => arena.sub(two, t),
                (false, true) => arena.sub(one, t),
                (false, false) => arena.sub(t, one),
            };
            Some(r)
        }
        ExprNode::Ei(_) => {
            let e = arena.exp(u);
            let f = internal(arena, InternalFn::EiAsym, u)?;
            Some(arena.mul(&[e, f]))
        }
        ExprNode::Li(_) if positive => {
            let l = arena.ln(u);
            let f = internal(arena, InternalFn::EiAsym, l)?;
            Some(arena.mul(&[u, f]))
        }
        _ => None,
    }
}

/// The argument of a special function that [`special_asymptotic_rewrite`]
/// expands at `±∞` (`Chi`, `Shi`, `erfi`, `Iν`, `Kν`, `Eₙ`, `Γ(s, ·)`,
/// `γ(s, ·)` with a parameter free of the limit variable, `ζ`), if `node` is
/// one.
fn special_asymptotic_arg(arena: &Arena, node: &ExprNode, x: ExprId) -> Option<ExprId> {
    use crate::base::libfn::LibFn;
    match *node {
        ExprNode::Zeta(u) => Some(u),
        ExprNode::Lib(f, ref args) => match (f, args.as_slice()) {
            (LibFn::Chi | LibFn::Shi | LibFn::Erfi, &[u]) => Some(u),
            (_, &[p, _]) if crate::base::walk::contains(arena, p, x) => None,
            (
                LibFn::BesselI
                | LibFn::BesselK
                | LibFn::ExpInt
                | LibFn::UpperGamma
                | LibFn::LowerGamma,
                &[_, u],
            ) => Some(u),
            _ => None,
        },
        _ => None,
    }
}

/// `f(u)` for `u → +∞` (`positive`) or `−∞`, `u` real, of a function of
/// [`special_asymptotic_arg`], with its exponential factor taken out and an
/// internal function whose expansion in `1/u` the series engine knows
/// ([`InternalFn::EiAsym`], `F`, and the
/// [`PowerAsymptotic`](crate::calculus::series::PowerAsymptotic) heads):
///
/// * `Chi u = (e^u F(u) + e^{−u} F(−u))/2` (`+ iπ` for `u < 0`), `Shi u =
///   (e^u F(u) − e^{−u} F(−u))/2` — `Chi`, `Shi` are `(Ei(u) ± Ei(−u))/2`
///   exactly (DLMF 6.2.15–16),
/// * `erfi u = e^{u²} E(u²)/u` (odd on both sides; `E` of `u²` keeps the series
///   of `erfi(√x)` in integer powers of `1/x`),
/// * `Iν(u) = e^u (2πu)^{−1/2} I(ν, u)`, `Kν(u) = e^{−u} (π/(2u))^{1/2} K(ν, u)`
///   (`u → +∞`),
/// * `Γ(s, u) = u^{s−1} e^{−u} G(s, u)`, `Eₙ(u) = e^{−u} u^{−1} G(1 − n, u)`,
///   `γ(s, u) = Γ(s) − Γ(s, u)` (`u → +∞`),
/// * `ζ(u) = 1 + 2^{−u} Z(u)` (`u → +∞`).
///
/// SymPy's `_eval_aseries` of `erfi`, `besseli`, `besselk`, `expint` and
/// `Chi`/`Shi` via `Ei` (SymPy expands `uppergamma`, `ζ` by other means);
/// the series are DLMF 7.12.1, 10.40.1–2, 8.11.2, 8.20.2, 25.2.1.  `Iν` is
/// not rewritten beside an `I₋ν` (anywhere in `whole`): the two have the
/// same expansion to every order in `1/u` and differ by the exponentially
/// smaller `(2/π) sin(νπ) Kν(u)`, which the expansion cannot see.  Before,
/// all of these were refused ("a function of unknown growth"):
/// `limit(Chi(x)·x·e^{−x}, x, ∞)` is `1/2`.
fn special_asymptotic_rewrite(
    arena: &mut Arena,
    node: &ExprNode,
    u: ExprId,
    positive: bool,
    whole: ExprId,
) -> Option<ExprId> {
    use crate::base::libfn::LibFn;
    use crate::calculus::series::PowerAsymptotic;
    let internal =
        |arena: &mut Arena, f: InternalFn, args: &[ExprId]| arena.func(f.into(), args).ok();
    let one = arena.one();
    let two = arena.int(2);
    let half = arena.rational(1, 2);
    let neg_u = arena.neg(u);
    let e_u = arena.exp(u);
    let e_neg_u = arena.exp(neg_u);
    match *node {
        ExprNode::Zeta(_) if positive => {
            let z = internal(arena, PowerAsymptotic::Zeta.head(), &[u])?;
            let p = arena.pow(two, neg_u);
            let t = arena.mul(&[p, z]);
            Some(arena.add(&[one, t]))
        }
        ExprNode::Lib(f, ref args) => match (f, args.as_slice()) {
            (LibFn::Chi | LibFn::Shi, _) => {
                let fp = internal(arena, InternalFn::EiAsym, &[u])?;
                let fm = internal(arena, InternalFn::EiAsym, &[neg_u])?;
                let ei_p = arena.mul(&[e_u, fp]);
                let ei_m = arena.mul(&[e_neg_u, fm]);
                let chi = f == LibFn::Chi;
                let ei_m = if chi { ei_m } else { arena.neg(ei_m) };
                let s = arena.add(&[ei_p, ei_m]);
                let r = arena.mul(&[half, s]);
                Some(if chi && !positive {
                    let i_pi = arena.mul(&[arena.i_unit(), arena.pi()]);
                    arena.add(&[r, i_pi])
                } else {
                    r
                })
            }
            // erfi u = e^{u²}·E(u²)/u for u → ±∞ (both sides odd).
            (LibFn::Erfi, _) => {
                let u2 = arena.pow(u, two);
                let u2 = crate::transforms::eval::eval(arena, u2);
                let g = arena.exp(u2);
                let e = internal(arena, PowerAsymptotic::Erfi.head(), &[u2])?;
                let inv = arena.pow(u, arena.neg_one());
                Some(arena.mul(&[g, e, inv]))
            }
            (LibFn::BesselI, &[nu, _]) if positive => {
                // Not beside an `I₋ν` (see above).
                for id in crate::base::walk::post_order_ids(arena, whole) {
                    if let ExprNode::Lib(LibFn::BesselI, gargs) = arena.node(id).clone()
                        && gargs.len() == 2
                        && gargs[0] != nu
                    {
                        let s = arena.add(&[gargs[0], nu]);
                        let s = crate::transforms::eval::eval(arena, s);
                        if arena.is_zero_structural(s) {
                            return None;
                        }
                    }
                }
                let i = internal(arena, PowerAsymptotic::BesselI.head(), &[nu, u])?;
                let two_pi_u = arena.mul(&[two, arena.pi(), u]);
                let m_half = arena.rational(-1, 2);
                let pre = arena.pow(two_pi_u, m_half);
                Some(arena.mul(&[e_u, pre, i]))
            }
            (LibFn::BesselK, &[nu, _]) if positive => {
                let k = internal(arena, PowerAsymptotic::BesselK.head(), &[nu, u])?;
                let two_u = arena.mul(&[two, u]);
                let q = arena.div(arena.pi(), two_u);
                let pre = arena.pow(q, half);
                Some(arena.mul(&[e_neg_u, pre, k]))
            }
            (LibFn::UpperGamma | LibFn::LowerGamma, &[s, _]) if positive => {
                let g = internal(arena, PowerAsymptotic::UpperGamma.head(), &[s, u])?;
                let sm1 = arena.sub(s, one);
                let p = arena.pow(u, sm1);
                let upper = arena.mul(&[p, e_neg_u, g]);
                if f == LibFn::UpperGamma {
                    return Some(upper);
                }
                // γ(s, u) = Γ(s) − Γ(s, u), for Γ(s) finite.
                if arena
                    .as_num(s)
                    .is_some_and(|r| r.is_integer() && !r.is_positive())
                {
                    return None;
                }
                let gs = arena.gamma(s);
                Some(arena.sub(gs, upper))
            }
            (LibFn::ExpInt, &[n, _]) if positive => {
                let s = arena.sub(one, n);
                let g = internal(arena, PowerAsymptotic::UpperGamma.head(), &[s, u])?;
                let inv = arena.pow(u, arena.neg_one());
                Some(arena.mul(&[e_neg_u, inv, g]))
            }
            _ => None,
        },
        _ => None,
    }
}

/// Does `e` contain a power with a non-integer exponent?
fn has_fractional_power(arena: &Arena, e: ExprId) -> bool {
    crate::base::walk::post_order_ids(arena, e).into_iter().any(|id| {
        matches!(arena.node(id), ExprNode::Pow(_, g) if arena.as_num(*g).is_none_or(|r| !r.is_integer()))
    })
}

/// Does `e` contain one of the internal asymptotic functions of
/// [`asymptotic_rewrite`]?
fn contains_internal_asymptotic(arena: &Arena, e: ExprId) -> bool {
    crate::base::walk::post_order_ids(arena, e)
        .into_iter()
        .any(|id| matches!(arena.node(id), ExprNode::Internal(..)))
}

/// `0^g → 0` where `g` is eventually positive; `Err` where it is not
/// (complex infinity, or no limit).  Left alone, `0^g` became
/// `exp(g·ln 0)` with `ln 0` taken for a finite coefficient:
/// `lim_{x→0⁺} 0^x` was 1, and so was `lim_{x→0⁺} |acosh(sign x)|^x`,
/// whose base becomes 0 only once `sign x` is resolved — hence a pass
/// after the rewrites.
fn resolve_zero_powers(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    let post = crate::base::walk::post_order_ids(arena, e);
    if !post.iter().any(|&id| {
        matches!(arena.node(id), ExprNode::Pow(b, g)
            if arena.is_zero_structural(*b) && crate::base::walk::contains(arena, *g, x))
    }) {
        return Ok(e);
    }
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    for &id in &post {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match arena.node(rebuilt).clone() {
            ExprNode::Pow(b, g)
                if arena.is_zero_structural(b) && crate::base::walk::contains(arena, g, x) =>
            {
                match sign_at_inf(arena, g, x, depth + 1, budget) {
                    Ok(s) if s > 0 => arena.zero(),
                    _ => {
                        return Err(crate::base::errors::SymplexError::ComputationFailed {
                            operation: "gruntz",
                            reason: "0^g with g not eventually positive has no limit".into(),
                        });
                    }
                }
            }
            _ => rebuilt,
        };
        cache.insert(id, new);
    }
    let r = cache.get(&e).copied().unwrap_or(e);
    Ok(crate::transforms::eval::eval(arena, r))
}

/// `exp(a + k·ln u) → e^a·u^k` throughout `e` (the principal `u^k` is
/// `exp(k·Log u)`, so this holds for every `u ≠ 0`).
///
/// Gruntz's theory assumes `exp(ln …)` simplified (SymPy's `mrv` does it
/// "for termination"); left alone, `exp(−ln(1/x))` — from `sinh(ln x)`
/// rewritten into exponentials — joined the MRV set beside `x`, and
/// `lim_{x→0} x·sinh(ln x)` came out `0` instead of `−1/2`.
fn simplify_exp_log(arena: &mut Arena, e: ExprId) -> ExprId {
    let post = crate::base::walk::post_order_ids(arena, e);
    if !post.iter().any(
        |&id| matches!(arena.node(id), ExprNode::Exp(a) if !exp_log_terms(arena, *a).0.is_empty()),
    ) {
        return e;
    }
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    for &id in &post {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match arena.node(rebuilt).clone() {
            ExprNode::Exp(a) if !exp_log_terms(arena, a).0.is_empty() => {
                let (logs, rest) = exp_log_terms(arena, a);
                let mut factors = Vec::with_capacity(logs.len() + 1);
                for (u, k) in logs {
                    factors.push(arena.pow(u, k));
                }
                if !rest.is_empty() {
                    let r = arena.add(&rest);
                    factors.push(arena.exp(r));
                }
                arena.mul(&factors)
            }
            _ => rebuilt,
        };
        cache.insert(id, new);
    }
    let r = cache.get(&e).copied().unwrap_or(e);
    crate::transforms::eval::eval(arena, r)
}

/// A branch function `f(u)` (`ln`, a fractional power, `asin`, `acos`,
/// `atanh`, `acosh`, `asinh`, `atan`, `Ei`, `Ci`, `Chi`, `ln Γ`) whose
/// argument tends to a point `l` of the function's cut, continued across
/// the cut from the side `u` approaches from ([`continuation_across_cut`]):
/// the leading terms of Gruntz's expansion do not keep that side.  (The
/// other functions with a cut, `W`, `polylog`, `Kν`, …, are refused there.)  The side is the eventual sign
/// of `Im(u − l)` (`Re(u − l)` for the cuts on the imaginary axis), from
/// its exact imaginary part ([`cut_side`]) — as SymPy decides it from the
/// first non-real term of the argument's series (`log._eval_nseries`) or
/// its direction (`asin._eval_nseries`); a real `u` moves along the cut.
/// `Ok(None)` when `f(u)` needs no continuation (its argument does not
/// tend to the cut, or is real and stays on it: there the principal values
/// are continuous — except for `acosh`, whose logarithmic form would put
/// `ln` on its own cut, `acosh(−1 − 1/x) = acosh(1 + 1/x) + iπ`), `Err`
/// when the side is undecided.
///
/// Up to 0.36 such a function was refused (`limit(x/(ln(−2 + ix) −
/// ln(−2)), x, 0, '+')` is `2i`, SymPy agrees; from the left `0`), and one
/// whose argument the rewrite did not visit took the value on the cut as
/// its leading term: `limit(asin(2 + ix), x, 0, '+')` was `asin 2`; it is
/// `π − asin 2` (SymPy).
///
/// [`continuation_across_cut`]: crate::calculus::limit::continuation_across_cut
fn resolve_branch_cut(
    arena: &mut Arena,
    node: &ExprNode,
    id: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<Option<ExprId>, crate::base::errors::SymplexError> {
    use crate::calculus::limit::BranchCut;
    let Some((arg, cut)) = crate::calculus::limit::branch_cut_of(arena, id) else {
        return Ok(None);
    };
    if !crate::base::walk::contains(arena, arg, x) {
        return Ok(None);
    }
    let acosh = matches!(node, ExprNode::Acosh(_));
    let beyond_one = matches!(node, ExprNode::Asin(_) | ExprNode::Acos(_));
    let real = eventually_real(arena, arg, x, depth + 1, budget);
    if real && !acosh && !beyond_one {
        return Ok(None);
    }
    let Ok(l) = limitinf(arena, arg, x, depth + 1, budget) else {
        return Ok(None);
    };
    if real && beyond_one && !is_infinite(arena, l) {
        return Ok(None);
    }
    // `asin`, `acos` of an argument tending to `±∞`: along (or beside) the
    // infinite ends of the cut, where the leading terms are unbounded
    // (`asin(x + i) = π/2 + i·acosh(x + i)`; before, "unbounded function of
    // a divergent argument").  (`atanh`, `acosh` are written as logarithms,
    // whose side `resolve_negative_leading` decides.)
    let infinite_end = beyond_one && (l == arena.infinity() || l == arena.neg_infinity());
    if !infinite_end
        && (is_infinite(arena, l) || !crate::calculus::limit::on_branch_cut(arena, cut, l))
    {
        return Ok(None);
    }
    let undecided = || crate::base::errors::SymplexError::ComputationFailed {
        operation: "gruntz",
        reason:
            "a non-real argument approaches the branch cut of a function from an undecided side"
                .into(),
    };
    let side = if real && cut != BranchCut::ImaginaryAxis {
        0
    } else {
        // The side of a real infinite end: the sign of `Im arg` itself.
        let base = if infinite_end { arena.zero() } else { l };
        cut_side(arena, arg, base, cut, x, depth, budget).ok_or_else(undecided)?
    };
    // Along the cut of `acosh` to its branch point `−1`: from which end.
    let along = if acosh && side == 0 {
        let d = arena.sub(arg, l);
        sign_at_inf(arena, d, x, depth + 1, budget).unwrap_or(0)
    } else {
        0
    };
    match crate::calculus::limit::continuation_across_cut(arena, id, l, side, along) {
        Some(g) => {
            tracing::debug!(
                side,
                continued = %arena.display(g).to_string(),
                "gruntz: a branch function continued across its cut"
            );
            Ok(Some(g))
        }
        None => Err(undecided()),
    }
}

/// The side of the cut `cut` from which `u → l` (a point of the cut)
/// approaches as `x → ∞`: the eventual sign of the component of `u − l`
/// perpendicular to the cut (`Im` for the cuts on the real axis, `Re` for
/// those on the imaginary axis), `0` when that component is identically
/// zero.  `None` when the component is not known exactly (an opaque
/// `Re`/`Im` of a sub-expression) or its sign is undecided.
fn cut_side(
    arena: &mut Arena,
    u: ExprId,
    l: ExprId,
    cut: crate::calculus::limit::BranchCut,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Option<i32> {
    let d = arena.sub(u, l);
    let d = crate::transforms::eval::eval(arena, d);
    let imaginary_axis = cut == crate::calculus::limit::BranchCut::ImaginaryAxis;
    let mut parts = crate::base::complex::decompose(arena, d);
    if !parts.exact
        && let Some(d2) = continue_real_on_cut(arena, d, x, depth + 1, budget)
        && d2 != d
    {
        // `√(1 − x)` is opaque to the decomposition, `i·√(x − 1)` is not
        // (and `√(1 − x)·√(−1 − x) = −√(x − 1)·√(x + 1)` is real).
        if !imaginary_axis && eventually_real(arena, d2, x, depth + 1, budget) {
            return Some(0);
        }
        let p2 = crate::base::complex::decompose(arena, d2);
        if p2.exact {
            parts = p2;
        }
    }
    if !parts.exact {
        // `Re ln(2 − x)` is opaque to the decomposition: the leading term
        // `c·ωᵉ` of `u − l` decides when `c` is a constant off the line
        // of the cut (`(ln(2 − x) − ln x − iπ)/2 ~ −1/x`).
        let (c, _) = mrv_leadterm(arena, d, x, depth + 1, budget, false).ok()?;
        let c = if crate::base::walk::contains(arena, c, x) {
            c
        } else {
            match settle_constant(arena, c) {
                ZeroTest::Undecided(_) => return None,
                settled => settled.value(arena),
            }
        };
        if arena.is_zero_structural(c) {
            // `u ≡ l` (as in `limitinf`): on the cut.
            return Some(0);
        }
        if !crate::base::walk::free_symbols(arena, c).is_empty() {
            return None;
        }
        // The certified sign of the component itself: a component of an
        // evaluated `c` that is exactly 0 can be rounding noise there
        // (`Im(−2i·acos 2) = −2·Re(acos 2) = 0` came out negative, and
        // `sqrt(1 + (acos 2 − ix)²)` was continued from below the cut).
        let p = if imaginary_axis {
            arena.re(c)
        } else {
            arena.im(c)
        };
        let p = crate::transforms::eval::eval(arena, p);
        return match crate::calculus::limit::const_sign(arena, p) {
            Some(s) if s != 0 => Some(s),
            _ => None,
        };
    }
    let p = if imaginary_axis { parts.re } else { parts.im };
    let p = crate::transforms::eval::eval(arena, p);
    if arena.is_zero_structural(p) {
        return Some(0);
    }
    let expanded = crate::transforms::expand::expand(arena, p);
    let expanded = crate::transforms::eval::eval(arena, expanded);
    if arena.is_zero_structural(expanded) {
        return Some(0);
    }
    // The decomposition writes `|v|` for a modulus (`−ln|1/x| + ln(x⁻²)/2`,
    // from `ln(x²)/2 − ln x` at `0⁻`): resolved by the eventual sign of `v`
    // first; logarithms of positive quantities that cancel only once
    // expanded are zero, which the leading-term search cannot see.  A zero
    // leading coefficient from `sign_at_inf` is an identically vanishing
    // component, as in `limitinf`.
    let p = rewrite_tractable(arena, p, x, depth + 1, budget).ok()?;
    let expanded = crate::simplify::log_expand::expand_log(arena, p);
    let expanded = crate::transforms::eval::eval(arena, expanded);
    if arena.is_zero_structural(p) || arena.is_zero_structural(expanded) {
        return Some(0);
    }
    sign_at_inf(arena, p, x, depth + 1, budget).ok()
}

/// `e` with every function whose argument is eventually real and lies on
/// the function's branch cut written by its principal value off the cut
/// ([`continuation_across_cut`] from the cut itself: `ln u = ln(−u) + iπ`,
/// `u^r = (−u)^r·e^{iπr}`, `acosh u = acosh(−u) + iπ` below `−1`, `asin u
/// = π/2 − i·acosh u` beyond `1`, …): the same values, but the complex
/// decomposition sees through them and conjugation commutes with every
/// function left.  `None` when a continuation is not known.
///
/// [`continuation_across_cut`]: crate::calculus::limit::continuation_across_cut
fn continue_real_on_cut(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Option<ExprId> {
    use crate::calculus::limit::BranchCut;
    if depth > MAX_DEPTH {
        return None;
    }
    let post = crate::base::walk::post_order_ids(arena, e);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    for &id in &post {
        if !crate::base::walk::contains(arena, id, x) {
            cache.insert(id, id);
            continue;
        }
        let r = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match crate::calculus::limit::branch_cut_of(arena, r) {
            Some((u, cut))
                if cut != BranchCut::ImaginaryAxis
                    && crate::base::walk::contains(arena, u, x)
                    && eventually_real(arena, u, x, depth + 1, budget)
                    && !real_off_cut(arena, u, cut, x, depth, budget) =>
            {
                let l = limitinf(arena, u, x, depth + 1, budget).ok()?;
                let along = if is_infinite(arena, l) {
                    0
                } else {
                    let d = arena.sub(u, l);
                    sign_at_inf(arena, d, x, depth + 1, budget).unwrap_or(0)
                };
                crate::calculus::limit::continuation_across_cut(arena, r, l, 0, along)?
            }
            _ => r,
        };
        cache.insert(id, new);
    }
    let result = cache.get(&e).copied().unwrap_or(e);
    Some(crate::transforms::eval::eval(arena, result))
}

/// Is the eventually real `u` eventually off the branch cut `cut` (by the
/// eventual sign of its distance to the cut)?
fn real_off_cut(
    arena: &mut Arena,
    u: ExprId,
    cut: crate::calculus::limit::BranchCut,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> bool {
    use crate::calculus::limit::BranchCut;
    let one = arena.one();
    let d = match cut {
        BranchCut::None | BranchCut::ImaginaryAxis => return true,
        BranchCut::NegativeReals => u,
        BranchCut::BelowOne => arena.sub(u, one),
        BranchCut::AboveOne => arena.sub(one, u),
        BranchCut::BeyondOne => {
            let two = arena.int(2);
            let u2 = arena.pow(u, two);
            arena.sub(one, u2)
        }
        BranchCut::BelowMinusInvE => {
            let m1 = arena.neg_one();
            let inv_e = arena.exp(m1);
            arena.add(&[u, inv_e])
        }
    };
    matches!(sign_at_inf(arena, d, x, depth + 1, budget), Ok(1))
}

/// [`continue_real_on_cut`] for the positive variable `x → ∞`, with a
/// budget of its own.
pub(crate) fn continue_real_on_cut_at_inf(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
) -> Option<ExprId> {
    let mut budget = Budget::new();
    continue_real_on_cut(arena, e, x, 0, &mut budget)
}

/// [`eventually_real`] under the generic reading of the parameters: every
/// symbol other than `x` without declared assumptions taken real.  The
/// guard of [`rewrite_tractable`] for `⌊·⌋`, `⌈·⌉`, `δ`, `min`, `max`,
/// which have no value off the reals: `δ(x + y) → 0` and `min(x + y, …)`
/// are read for a generic (real) `y`, as `δ(y) = 0` is elsewhere in the
/// limit engine; `⌈ln(x)^{x²}⌉` and `⌊1/2 + i·x⌋` are not real for any
/// value of a parameter.
fn real_for_generic_parameters(
    arena: &mut Arena,
    u: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> bool {
    use crate::base::assumptions::{Assumptions, Props};
    if known_real(arena, u, x, depth, budget) {
        return true;
    }
    // Last, the values along the approach: exactly real at the samples, as
    // a function the rules above do not know (`δ(erfinv x)`, `H(F(5 | x))`
    // at 0); a certified imaginary digit refuses.
    let r = real_at_samples(arena, u, x);
    if r.is_known() {
        return r.is_true();
    }
    let params: Vec<ExprId> = crate::base::walk::free_symbols(arena, u)
        .into_iter()
        .filter(|&s| {
            s != x
                && matches!(*arena.node(s), ExprNode::Symbol(sid)
                    if arena.symbol_assumptions(sid) == Assumptions::default())
        })
        .collect();
    if params.is_empty() {
        return false;
    }
    let mut v = u;
    for (k, &s) in params.iter().enumerate() {
        let r = arena.symbol(&format!("__gz_real{k}"));
        if let ExprNode::Symbol(sid) = *arena.node(r) {
            let mut a = Assumptions::default();
            a.assert_true(Props::REAL);
            a.forward_chain();
            arena.set_symbol_assumptions(sid, a);
        }
        v = crate::transforms::subs::subs(arena, v, s, r);
    }
    let v = crate::transforms::eval::eval(arena, v);
    known_real(arena, v, x, depth, budget)
}

/// [`eventually_real`], with `W` and the internal asymptotic functions of
/// real arguments as real symbols ([`real_special_values_as_symbols`]),
/// or by the assumption system (`x` is positive), which knows the library
/// functions by their `real_on` facet (`⌈J₀(x)⌉` at `0`) and that `re`,
/// `im` are real.  Not rules of `eventually_real` itself, which also
/// routes `limit.rs` (a real product or sum goes to the Gruntz algorithm,
/// which does not expand `fresnels(x)/x` at `∞`).
fn known_real(arena: &mut Arena, v: ExprId, x: ExprId, depth: usize, budget: &mut Budget) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    if eventually_real(arena, v, x, depth + 1, budget)
        || Truth::from(AssumptionCache::new().query(arena, v, Props::REAL)).is_true()
    {
        return true;
    }
    let w = real_special_values_as_symbols(arena, v, x, depth, budget);
    w != v && eventually_real(arena, w, x, depth + 1, budget)
}

/// Is `u` (a function of the positive `x → ∞`, free of other symbols) real
/// at the samples `x = 10², 10⁴` by a certified evaluation: `False` for a
/// certified nonzero imaginary part at one of them, `True` for an exactly
/// real value at both, `Unknown` otherwise (a value `evalf` cannot give).  The reading of [`real_for_generic_parameters`] for functions
/// its rules do not know, as `nonzero_at_a_sample` is for the parameters
/// of `δ`.
fn real_at_samples(arena: &mut Arena, u: ExprId, x: ExprId) -> Truth {
    use crate::transforms::evalf::{ZeroSearch, evalf_verdict};
    if crate::base::walk::free_symbols(arena, u)
        .into_iter()
        .any(|s| s != x)
    {
        return Truth::Unknown;
    }
    for sample in [100_i64, 10_000] {
        let s = arena.int(sample);
        let v = crate::transforms::subs::subs(arena, u, x, s);
        let v = crate::transforms::eval::eval(arena, v);
        if crate::base::walk::has_unevaluated(arena, v) || contains_singular_atom(arena, v) {
            return Truth::Unknown;
        }
        let Some(z) = evalf_verdict(arena, v, 20, ZeroSearch::Cap)
            .ok()
            .and_then(crate::base::truth::Verdict::proven)
        else {
            return Truth::Unknown;
        };
        if !z.1.is_zero() {
            return Truth::False;
        }
    }
    Truth::True
}

/// Is the argument `a` of `min` or `max` real near the limit point: its
/// terms that vary with `x` by [`real_for_generic_parameters`], those free
/// of `x` unless they are proved not real (`min(−3y, Γ(x))`,
/// `min(ψ^{(1/2)}(−3/5), 3 − W(x))` keep their generic reading; `min(0, x +
/// i)` and `min(x, Ci(−2))` have no value)?
fn min_max_argument_real(
    arena: &mut Arena,
    a: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    let terms: Vec<ExprId> = match arena.node(a) {
        ExprNode::Add(cs) => cs.to_vec(),
        _ => vec![a],
    };
    let (consts, deps): (Vec<ExprId>, Vec<ExprId>) = terms
        .into_iter()
        .partition(|&t| !crate::base::walk::contains(arena, t, x));
    let consts_ok = consts.into_iter().all(|c| {
        !Truth::from(AssumptionCache::new().query(arena, c, Props::REAL)).is_false()
            && !(crate::base::walk::free_symbols(arena, c).is_empty() && {
                let mut reals = AssumptionCache::new();
                Truth::from(crate::transforms::realness::constant_realness_as_declared(
                    arena, c, 20, &mut reals,
                ))
                .is_false()
            })
    });
    consts_ok && {
        let d = arena.add(&deps);
        deps.is_empty() || real_for_generic_parameters(arena, d, x, depth, budget)
    }
}

/// `v` with the values of `W` and of the internal asymptotic functions of
/// [`special_asymptotic_rewrite`] that are real replaced by real symbols:
/// `W(a)` for a real `a` with `a + 1/e` eventually positive, an internal
/// function (a series in `1/u`) of real arguments.  So `⌊3/2 + W(−x)⌋` and
/// `⌈1 + li x⌉` at `0⁺` pass [`real_for_generic_parameters`].  (Not rules
/// of [`eventually_real`]: `limit.rs` sends a real product or sum to the
/// Gruntz algorithm, and `atan2(√2 − x, W(x))` at 0 then had no series.)
fn real_special_values_as_symbols(
    arena: &mut Arena,
    v: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> ExprId {
    use crate::base::assumptions::{Assumptions, Props};
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    let mut k = 0usize;
    for id in crate::base::walk::post_order_ids(arena, v) {
        let r = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        // `Some(true)`: positive (`W(a)` of a positive `a`), `Some(false)`:
        // real.
        let real = match arena.node(r).clone() {
            ExprNode::LambertW(a) if eventually_real(arena, a, x, depth + 1, budget) => {
                if matches!(sign_at_inf(arena, a, x, depth + 1, budget), Ok(1)) {
                    Some(true)
                } else {
                    let e_inv = arena.pow(arena.e_const(), arena.neg_one());
                    let d = arena.add(&[a, e_inv]);
                    matches!(sign_at_inf(arena, d, x, depth + 1, budget), Ok(1)).then_some(false)
                }
            }
            // Positive when its series in `1/u` is (`Ei u = e^u·F(u)`, `F > 0`
            // for `u → +∞`: `Ei(x)^x` at `∞`).
            ExprNode::Internal(_, args) => {
                if args
                    .iter()
                    .all(|&c| eventually_real(arena, c, x, depth + 1, budget))
                {
                    Some(matches!(sign_at_inf(arena, r, x, depth + 1, budget), Ok(1)))
                } else {
                    None
                }
            }
            _ => None,
        };
        let new = match real {
            Some(true) => {
                k += 1;
                arena.positive_symbol(&format!("__gz_posv{k}"))
            }
            Some(false) => {
                let s = arena.symbol(&format!("__gz_realv{k}"));
                k += 1;
                if let ExprNode::Symbol(sid) = *arena.node(s) {
                    let mut a = Assumptions::default();
                    a.assert_true(Props::REAL);
                    a.forward_chain();
                    arena.set_symbol_assumptions(sid, a);
                }
                s
            }
            None => r,
        };
        cache.insert(id, new);
    }
    cache.get(&v).copied().unwrap_or(v)
}

/// Is `e` real for all sufficiently large `x` (`x` itself is real and
/// positive)?  Structural and conservative: rational constants, `π`, `e`,
/// `x`; sums, products, `exp`, trigonometric and hyperbolic functions,
/// `atan`, `asinh`, `|·|`, `sign` of real arguments; `b^n` for an integer
/// `n`, `b^g` and `ln b` for an eventually positive real `b`; `asin`, `acos`,
/// `atanh` of a real `u` with `1 − u²` eventually positive, `acosh u` with
/// `u − 1` eventually positive; `Γ`, `ψ`, `x!`, `ζ`, `Si`, `Ei` of a real
/// argument, `ln Γ`, `Ci`, `li` of an eventually positive one.  Anything
/// else is not known to be real.
///
/// The sign-based rewrites of [`rewrite_tractable`] are valid for real
/// arguments only.
fn eventually_real(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> bool {
    if depth > MAX_DEPTH {
        return false;
    }
    let post = crate::base::walk::post_order_ids(arena, e);
    let mut real: FxHashMap<ExprId, bool> = FxHashMap::default();
    for &id in &post {
        let node = arena.node(id).clone();
        let r = |c: ExprId| real.get(&c).copied().unwrap_or(false);
        let v = match node {
            ExprNode::Num(_) | ExprNode::Pi | ExprNode::E | ExprNode::EulerGamma => true,
            ExprNode::Catalan | ExprNode::GoldenRatio => true,
            ExprNode::Symbol(_) => {
                id == x || {
                    let mut cache = crate::base::assumptions::AssumptionCache::new();
                    Truth::from(cache.query(arena, id, crate::base::assumptions::Props::REAL))
                        .is_true()
                }
            }
            ExprNode::Add(ref cs) | ExprNode::Mul(ref cs) => cs.iter().all(|&c| r(c)),
            ExprNode::Neg(a) => r(a),
            ref n if n.func_head().is_some_and(real_analytic_on_reals) => {
                n.unary_arg_if(real_analytic_on_reals).is_some_and(r)
            }
            ExprNode::Abs(_) | ExprNode::Sign(_) | ExprNode::Heaviside(_) => true,
            // `re c`, `im c` of an opaque constant ([`opaque_constants_split`]).
            ExprNode::Re(a) | ExprNode::Im(a) | ExprNode::Arg(a) => {
                !crate::base::walk::contains(arena, a, x)
            }
            ExprNode::Floor(a) | ExprNode::Ceiling(a) => r(a),
            ExprNode::Pow(b, g) => {
                r(b) && r(g) && {
                    let integer = arena.as_num(g).is_some_and(|q| q.is_integer());
                    integer || matches!(sign_at_inf(arena, b, x, depth + 1, budget), Ok(1))
                }
            }
            ExprNode::Ln(b) => r(b) && matches!(sign_at_inf(arena, b, x, depth + 1, budget), Ok(1)),
            ExprNode::Asin(u) | ExprNode::Acos(u) | ExprNode::Atanh(u) => {
                r(u) && {
                    let one = arena.one();
                    let two = arena.int(2);
                    let u2 = arena.pow(u, two);
                    let d = arena.sub(one, u2);
                    matches!(sign_at_inf(arena, d, x, depth + 1, budget), Ok(1))
                }
            }
            ExprNode::Acosh(u) => {
                r(u) && {
                    let one = arena.one();
                    let d = arena.sub(u, one);
                    matches!(sign_at_inf(arena, d, x, depth + 1, budget), Ok(1))
                }
            }
            // Real on the reals: `Γ`, `ψ`, `x!`, `ζ`, `Si`, and `Ei` (whose
            // principal value on its cut, the negative reals, is the mean of
            // its two sides); on the positive reals: `ln Γ`, `Ci`, and `li`
            // (on `(0, 1)` also the mean of its sides).  Before, `li(1 + x)`
            // at `0⁻` (real, `−∞`) was refused: the imaginary part of its
            // infinite limit was not determined.
            ExprNode::Gamma(a)
            | ExprNode::Digamma(a)
            | ExprNode::Factorial(a)
            | ExprNode::Zeta(a)
            | ExprNode::Si(a)
            | ExprNode::Ei(a) => r(a),
            ExprNode::LogGamma(a) | ExprNode::Ci(a) | ExprNode::Li(a) => {
                r(a) && matches!(sign_at_inf(arena, a, x, depth + 1, budget), Ok(1))
            }
            _ => false,
        };
        // A real constant the rules above do not see (`cosh(acos 4)`),
        // certified numerically (an unassumed symbol is complex), or a
        // Bessel function of an even order at an imaginary argument
        // (`J₀(acos π)`, whose value has no certified imaginary part 0).
        let v = v || {
            !crate::base::walk::contains(arena, id, x)
                && !matches!(arena.node(id), ExprNode::Symbol(_))
                && crate::base::walk::free_symbols(arena, id).is_empty()
                && ((!crate::base::walk::contains(arena, id, arena.i_unit()) && {
                    let mut reals = crate::base::assumptions::AssumptionCache::new();
                    Truth::from(crate::transforms::realness::constant_realness_as_declared(
                        arena, id, 20, &mut reals,
                    ))
                    .is_true()
                }) || crate::calculus::limit::real_bessel_at_imaginary_argument(arena, id))
        };
        real.insert(id, v);
    }
    real.get(&e).copied().unwrap_or(false)
}

/// `(Re g, Im g)` for the exponent `g → ±∞` of an exponential that is not
/// eventually real, when `Re g` is (so that `e^g = e^{Re g}·(cos(Im g) +
/// i·sin(Im g))` has a positive representative `e^{−Re g}`); `None` for a
/// real exponent or one not split exactly.
///
/// - A non-real constant term `c`, the varying part real: `Re c`, `Im c`
///   by the exact decomposition, of the logarithmic forms of the inverse
///   functions in it if needed (`atanh 2 = ln 3/2 − iπ/2`), else, for a
///   constant certified non-real, the parts `re c`, `im c` themselves.
/// - A non-real varying part whose imaginary part tends to a nonzero
///   constant or diverges: the exact decomposition of `g`, `Im g` in its
///   tractable form (`e^{−z·ln(2z + atanh 2)}`, from `(atanh 2 − 2x)^x` at
///   `−∞`: `Im g → π/4`; up to 0.49 its `ω` was not positive, and
///   `sign((atanh 2 − 2x)^x)` tended to `1`).  A diverging `Im g` leaves
///   the bounded oscillation `cos(Im g) + i·sin(Im g)`, as in
///   [`phase_split`] (which the tractable rewrite applies before; this
///   exponential was built later, from a power, or by combining two:
///   `(√5·i + 2x)^x·x^{2 − 3i + ex}` at `∞` was `e^{i√5/2}·∞`, its phase
///   `−3·ln x` lost).  An imaginary part tending to 0 turns the phase of
///   `ω` to 1 and needs no split.
///
/// A non-real varying part without an exact decomposition is left as it
/// is: [`mrv_leadterm`] refuses a leading term that needs the phase of such
/// a member ([`phase_unknown`]).
fn exponent_phase_split(
    arena: &mut Arena,
    arg: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<Option<(ExprId, ExprId)>, crate::base::errors::SymplexError> {
    use crate::transforms::eval::eval;
    let (c, d) = constant_and_varying_terms(arena, arg, x);
    let consts = if arena.is_zero_structural(c) {
        Vec::new()
    } else {
        vec![c]
    };
    let c_real = consts.is_empty() || constant_known_real(arena, c, x, depth, budget);
    let d_real = !may_leave_reals(arena, d) || known_real(arena, d, x, depth, budget);
    if c_real && d_real {
        return Ok(None);
    }
    let (re, im) = if d_real {
        let mut p = crate::base::complex::decompose(arena, c);
        if !p.exact {
            let logs = crate::calculus::limit::inverse_functions_as_logs(arena, c);
            if logs != c {
                let logs = eval(arena, logs);
                let q = crate::base::complex::decompose(arena, logs);
                if q.exact {
                    p = q;
                }
            }
        }
        let (re, im) = if p.exact {
            (p.re, p.im)
        } else if crate::base::walk::free_symbols(arena, c).is_empty() && {
            let mut reals = crate::base::assumptions::AssumptionCache::new();
            Truth::from(crate::transforms::realness::constant_realness_as_declared(
                arena, c, 20, &mut reals,
            ))
            .is_false()
        } {
            (arena.re(c), arena.im(c))
        } else {
            return Ok(None);
        };
        let re = eval(arena, re);
        (arena.add(&[d, re]), im)
    } else {
        let Some((re, im)) = varying_parts(arena, arg, x) else {
            return Ok(None);
        };
        if !eventually_real(arena, re, x, depth + 1, budget) {
            return Ok(None);
        }
        // `Im g` in its tractable form (`atan(π/(x/2 + ln 2))`, not
        // `atan2(π, x/2 + ln 2)`, which the series engine does not expand).
        match imaginary_part_limit(arena, im, x, depth, budget)? {
            Some((im, l)) if !arena.is_zero_structural(l) => (re, im),
            _ => return Ok(None),
        }
    };
    let im = eval(arena, im);
    Ok((!arena.is_zero_structural(im)).then_some((re, im)))
}

/// Most nodes of an exponent [`constant_and_varying_terms`] multiplies out.
const MAX_SPLIT_EXPAND_SIZE: usize = 64;

/// The terms of the exponent `g` free of `x` and those with `x`, summed;
/// a small `g` that is not a sum multiplied out first (`(c + re x)·ln √2`
/// has the constant term `c·ln √2`).
fn constant_and_varying_terms(arena: &mut Arena, g: ExprId, x: ExprId) -> (ExprId, ExprId) {
    let g = if matches!(arena.node(g), ExprNode::Add(_))
        || crate::transforms::pattern::tree_size_capped(arena, g, MAX_SPLIT_EXPAND_SIZE + 1)
            > MAX_SPLIT_EXPAND_SIZE
    {
        g
    } else {
        let e = crate::transforms::expand::expand(arena, g);
        crate::transforms::eval::eval(arena, e)
    };
    let terms: Vec<ExprId> = match arena.node(g) {
        ExprNode::Add(cs) => cs.to_vec(),
        _ => vec![g],
    };
    let (consts, deps): (Vec<ExprId>, Vec<ExprId>) = terms
        .into_iter()
        .partition(|&t| !crate::base::walk::contains(arena, t, x));
    (arena.add(&consts), arena.add(&deps))
}

/// `(Re g, Im g)` of an exponent `g` that varies with `x`, by the exact
/// decomposition of `g` with its inverse functions as logarithms, and with
/// its opaque constants written `re c + i·im c` if that is needed (`W(−1)`
/// in `−(πx + W(−1))²`, from `erfc(W(−1) − πx)`).  `None` when the
/// decomposition is not exact.
fn varying_parts(arena: &mut Arena, g: ExprId, x: ExprId) -> Option<(ExprId, ExprId)> {
    use crate::transforms::eval::eval;
    let logs = crate::calculus::limit::inverse_functions_as_logs(arena, g);
    let g = if logs == g { g } else { eval(arena, logs) };
    let mut p = crate::base::complex::decompose(arena, g);
    if !p.exact {
        let split = opaque_constants_split(arena, g, x);
        if split != g {
            p = crate::base::complex::decompose(arena, split);
        }
    }
    if !p.exact {
        return None;
    }
    let re = eval(arena, p.re);
    let im = eval(arena, p.im);
    Some((re, im))
}

/// The imaginary part `im` of an exponent in its tractable form, and its
/// limit at `∞` (`±∞` or a finite value); `None` when it is not found.
fn imaginary_part_limit(
    arena: &mut Arena,
    im: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<Option<(ExprId, ExprId)>, crate::base::errors::SymplexError> {
    let Ok(im) = rewrite_tractable(arena, im, x, depth + 1, budget) else {
        return Ok(None);
    };
    match limitinf(arena, im, x, depth + 1, budget) {
        Ok(l) if l == arena.infinity() || l == arena.neg_infinity() => Ok(Some((im, l))),
        // A constant limit that is 0 by its certified value (`im c` of a
        // real `c` the decomposition did not see: `laguerre(−1, 3)`).
        Ok(l) if !contains_singular_atom(arena, l) => {
            let zero = arena.is_zero_structural(l)
                || (crate::base::walk::free_symbols(arena, l).is_empty()
                    && crate::calculus::calculus_util::hidden_zero(arena, l).proven()
                        == Some(crate::base::truth::Truth::True));
            Ok(Some((im, if zero { arena.zero() } else { l })))
        }
        Err(e) if is_budget_error(&e) => Err(e),
        _ => Ok(None),
    }
}

/// Is the member `m = e^g` of an MRV set one whose phase is not known: the
/// terms of `g` that vary with `x` not known to be real, and their
/// imaginary part not shown to tend to 0 ([`exponent_phase_split`] leaves
/// such a member whole, so its representative `ω` need not be positive)?
/// `Γ(c − x/2)` at `−∞` for a non-real `c` is `e^{ln Γ(c + t/2)}`, whose
/// phase `Im c·ln t` turns.  A constant term is read generically (one
/// proved non-real has been split off).
fn phase_unknown(
    arena: &mut Arena,
    m: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<bool, crate::base::errors::SymplexError> {
    let ExprNode::Exp(g) = *arena.node(m) else {
        return Ok(false);
    };
    let (_, d) = constant_and_varying_terms(arena, g, x);
    // With parameters the generic reading stands (`(y/x)^{asin x + 42} →
    // sign(y)⁴²·∞` at `0`).
    if crate::base::walk::free_symbols(arena, d)
        .into_iter()
        .any(|s| s != x)
        || !may_leave_reals(arena, d)
        || known_real(arena, d, x, depth, budget)
        || real_at_samples(arena, d, x).is_true()
    {
        return Ok(false);
    }
    let Some((_, im)) = varying_parts(arena, d, x) else {
        return Ok(true);
    };
    Ok(!imaginary_part_limit(arena, im, x, depth, budget)?
        .is_some_and(|(_, l)| arena.is_zero_structural(l)))
}

/// `g` with every subexpression free of `x` and of symbols whose complex
/// decomposition is not exact (`W(−1)`, `acos(3)²`) written `re c +
/// i·im c`, with the real nodes `re c`, `im c`.
fn opaque_constants_split(arena: &mut Arena, g: ExprId, x: ExprId) -> ExprId {
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    let mut changed = false;
    for id in crate::base::walk::post_order_ids(arena, g) {
        let opaque = !crate::base::walk::contains(arena, id, x)
            && arena.as_num(id).is_none()
            && crate::base::walk::free_symbols(arena, id).is_empty()
            && !crate::base::complex::decompose(arena, id).exact;
        let new = if opaque {
            let re = arena.re(id);
            let im = arena.im(id);
            let i_im = arena.mul(&[arena.i_unit(), im]);
            arena.add(&[re, i_im])
        } else {
            crate::base::walk::rebuild_with_cache(arena, id, &cache)
        };
        changed |= new != id;
        cache.insert(id, new);
    }
    if changed {
        cache.get(&g).copied().unwrap_or(g)
    } else {
        g
    }
}

/// Is the leading exponent `e` (of `c·ω^e`, `ω → 0⁺`) a constant free of
/// symbols that is not proved real, so that `ω^e = e^{e·ln ω}` has a
/// phase turning with `ln ω`?  Up to 0.49 the leading coefficients were
/// followed regardless: `(√5·i + 2x)^x·x^{−3i}` at `∞` was `e^{i√5/2}·∞`
/// and `e^{−x}·(e^i − πx)^i` at `−∞` was `π^i·∞` (neither has a
/// direction).  An exponent with parameters keeps the generic reading.
fn exponent_turns(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> bool {
    arena.as_num(e).is_none()
        && !crate::base::walk::contains(arena, e, x)
        && crate::base::walk::free_symbols(arena, e).is_empty()
        && !constant_known_real(arena, e, x, depth, budget)
}

/// Is the constant (free of `x`) `c` real: by the assumptions (cheap,
/// structural), else by [`eventually_real`] (its rules, then a certified
/// evaluation of a constant free of symbols)?
fn constant_known_real(
    arena: &mut Arena,
    c: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> bool {
    let mut cache = crate::base::assumptions::AssumptionCache::new();
    Truth::from(cache.query(arena, c, crate::base::assumptions::Props::REAL)).is_true()
        || eventually_real(arena, c, x, depth + 1, budget)
}

/// Decide whether a boolean condition holds for all sufficiently large `x`.
fn eventually_true(
    arena: &mut Arena,
    cond: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Truth {
    if depth > MAX_DEPTH {
        return Truth::Unknown;
    }
    let node = arena.node(cond).clone();
    match node {
        ExprNode::BoolTrue => Truth::True,
        ExprNode::BoolFalse => Truth::False,
        ExprNode::Gt(a, b) | ExprNode::Ge(a, b) | ExprNode::Eq_(a, b) | ExprNode::Ne(a, b) => {
            let diff = arena.sub(a, b);
            let Ok(s) = sign_at_inf(arena, diff, x, depth + 1, budget) else {
                return Truth::Unknown;
            };
            Truth::from_bool(match node {
                ExprNode::Gt(..) => s > 0,
                ExprNode::Ge(..) => s >= 0,
                ExprNode::Eq_(..) => s == 0,
                _ => s != 0,
            })
        }
        ExprNode::And(children) => Truth::all(
            children
                .into_iter()
                .map(|c| eventually_true(arena, c, x, depth + 1, budget)),
        ),
        ExprNode::Or(children) => Truth::any(
            children
                .into_iter()
                .map(|c| eventually_true(arena, c, x, depth + 1, budget)),
        ),
        ExprNode::Not(inner) => eventually_true(arena, inner, x, depth + 1, budget).not(),
        _ => Truth::Unknown,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public entry points
// ═══════════════════════════════════════════════════════════════════════════

/// Validate a Gruntz result: it must not contain the variable or any
/// internal dummy symbol.
fn validate_result(
    arena: &Arena,
    r: ExprId,
    x: ExprId,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    if crate::base::walk::contains(arena, r, x)
        || contains_foreign_dummy(arena, r, x)
        || contains_internal_asymptotic(arena, r)
    {
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz",
            reason: "expression has no limit or the limit could not be determined".into(),
        });
    }
    Ok(r)
}

/// Compute `lim(x→+∞) e` using the Gruntz algorithm, with result validation.
///
/// This is the entry point used by the one-sided limit machinery in
/// [`limit`](crate::calculus::limit), after the substitution `x = a ± 1/w`.
pub(crate) fn limit_pos_inf(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    let mut budget = Budget::new();
    let r = limitinf_directed(arena, e, x, 0, &mut budget, true)?;
    let r = validate_result(arena, r, x)?;
    // `e^{½ ln 2 + ½ ln π}` (from Stirling's constant) is `√2·√π`.
    Ok(simplify_exp_log(arena, r))
}

/// [`real_for_generic_parameters`] for the positive variable `x → ∞`, with
/// a budget of its own.
pub(crate) fn real_for_generic_parameters_at_inf(arena: &mut Arena, e: ExprId, x: ExprId) -> bool {
    let mut budget = Budget::new();
    real_for_generic_parameters(arena, e, x, 0, &mut budget)
}

/// Is `e` real for all sufficiently large values of the positive variable
/// `x` ([`eventually_real`], with a budget of its own)?
pub(crate) fn eventually_real_at_inf(arena: &mut Arena, e: ExprId, x: ExprId) -> bool {
    let mut budget = Budget::new();
    eventually_real(arena, e, x, 0, &mut budget)
}

/// The eventual sign of `e` as the positive variable `x → ∞`
/// ([`sign_at_inf`]: the sign of the leading coefficient, so `−1` for
/// `−x + i`), with a budget of its own; `None` when undecided.
pub(crate) fn eventual_sign_at_inf(arena: &mut Arena, e: ExprId, x: ExprId) -> Option<i32> {
    let mut budget = Budget::new();
    sign_at_inf(arena, e, x, 0, &mut budget).ok()
}

/// Compute `lim(z → z0) e` using the Gruntz algorithm.
///
/// For a finite `z0` this computes the *right-hand* limit `z → z0⁺` via the
/// substitution `z = z0 + 1/x`, `x → +∞`.
pub(crate) fn gruntz(
    arena: &mut Arena,
    e: ExprId,
    z: ExprId,
    z0: ExprId,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    let mut budget = Budget::new();
    gruntz_with_budget(arena, e, z, z0, &mut budget)
}

/// [`gruntz`] with an explicit work [`Budget`].
fn gruntz_with_budget(
    arena: &mut Arena,
    e: ExprId,
    z: ExprId,
    z0: ExprId,
    budget: &mut Budget,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    tracing::info!("gruntz: entry point");

    if z0 == arena.infinity() {
        tracing::debug!("gruntz: limit at +∞, substituting a positive z");
        let x = fresh_positive_dummy(arena, budget);
        let e_sub = crate::transforms::subs::subs(arena, e, z, x);
        let e_sub = crate::transforms::eval::eval(arena, e_sub);
        let r = limitinf_directed(arena, e_sub, x, 0, budget, true)?;
        let r = validate_result(arena, r, x)?;
        let r = simplify_exp_log(arena, r);
        return validate_result(arena, r, z);
    }

    if z0 == arena.neg_infinity() {
        tracing::debug!("gruntz: limit at -∞, substituting z = -x");
        let x = fresh_positive_dummy(arena, budget);
        let neg_x = arena.neg(x);
        let e_sub = crate::transforms::subs::subs(arena, e, z, neg_x);
        let r = limitinf_directed(arena, e_sub, x, 0, budget, true)?;
        let r = validate_result(arena, r, x)?;
        let r = simplify_exp_log(arena, r);
        return validate_result(arena, r, z);
    }

    let e_display = arena.display(e).to_string();
    let z0_display = arena.display(z0).to_string();
    tracing::debug!(expr = %e_display, z0 = %z0_display, "gruntz: finite-point limit, substituting z = z0 + 1/x");
    let x = fresh_positive_dummy(arena, budget);
    let one = arena.one();
    let inv_x = arena.div(one, x);
    let z0_plus_inv_x = arena.add(&[z0, inv_x]);
    let e_sub = crate::transforms::subs::subs(arena, e, z, z0_plus_inv_x);
    let sub_display = arena.display(e_sub).to_string();
    tracing::debug!(after_sub = %sub_display, "gruntz: after z = z0 + 1/x substitution");

    // Clear nested fractions, simplify.
    let e_together = crate::poly::polybridge::together(arena, e_sub);
    let e_cancelled = crate::poly::polybridge::cancel(arena, e_together, x);
    let e_simplified = crate::transforms::eval::eval(arena, e_cancelled);
    let e_simplified = crate::transforms::expand::expand(arena, e_simplified);
    let e_simplified = crate::transforms::eval::eval(arena, e_simplified);

    let simplified_display = arena.display(e_simplified).to_string();
    tracing::debug!(simplified = %simplified_display, "gruntz: finite-point expression simplified, calling limitinf");
    let r = limitinf_directed(arena, e_simplified, x, 0, budget, true)?;
    let r = validate_result(arena, r, x)?;
    let r = simplify_exp_log(arena, r);
    validate_result(arena, r, z)
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Replace exp(arg), sin(arg), cos(arg), … nodes (where arg depends on w)
/// with their Taylor series in w around w=0.
///
/// This avoids pole issues with full series expansion on expressions like
/// `(exp(w)-1)/w`. By expanding `exp(w) → 1 + w + w²/2 + ...` and
/// substituting back, the algebraic simplification handles the rest:
/// `(1 + w + w²/2 - 1)/w = 1 + w/2 + ...`
///
/// Functions whose series cannot be formed (poles at `w = 0`) are left in
/// place; the only error is an exhausted work budget.
fn expand_functions_as_series(
    arena: &mut Arena,
    budget: &mut Budget,
    expr: ExprId,
    w: ExprId,
    order: u32,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    // Collect all function nodes that depend on w
    let post_order = crate::base::walk::post_order_ids(arena, expr);
    let mut result = expr;

    for &id in &post_order {
        budget.tick(1)?;
        let should_expand = match arena.node(id).clone() {
            // Non-integer powers of a regular base, e.g. `(1 + ω)^{1/2}`.
            ExprNode::Pow(base, e) => {
                crate::base::walk::contains(arena, base, w)
                    && arena.as_num(e).is_some_and(|r| !r.is_integer())
            }
            node => node
                .unary_arg_if(expanded_as_series)
                .is_some_and(|arg| crate::base::walk::contains(arena, arg, w)),
        };

        if should_expand && let Ok(series) = budgeted_series(arena, budget, id, w, order) {
            let expanded = crate::transforms::expand::expand(arena, series);
            let evaled = crate::transforms::eval::eval(arena, expanded);
            // `series` only rejects a value that *is* ±∞/zoo/NaN; a function
            // with a pole at ω = 0 (`ln(1/ω + 1)`) can still yield a
            // "series" containing one. Skip such nodes.
            if contains_singular_atom(arena, evaled) {
                continue;
            }
            let old_display = arena.display(id).to_string();
            let new_display = arena.display(evaled).to_string();
            tracing::trace!(
                original = %old_display,
                series = %new_display,
                "gruntz::expand_functions_as_series: expanded function"
            );
            result = crate::transforms::subs::subs(arena, result, id, evaled);
            budget.charge_size(arena, result)?;
        }
    }

    Ok(result)
}

/// Does `e` contain `ln Γ`?
fn contains_log_gamma(arena: &Arena, e: ExprId) -> bool {
    crate::base::walk::post_order_ids(arena, e)
        .into_iter()
        .any(|id| matches!(arena.node(id), ExprNode::LogGamma(_)))
}

/// Number of distinct `exp` nodes in `e` (the DAG is walked once).
fn count_exp_nodes(arena: &Arena, e: ExprId) -> usize {
    crate::base::walk::post_order_ids(arena, e)
        .into_iter()
        .filter(|&id| matches!(arena.node(id), ExprNode::Exp(_)))
        .count()
}

/// Is `e` the budget's refusal (exhausted work, or an expression too large
/// for the series fallbacks)?  Those end the whole computation; other
/// failures of one strategy let the next one try.
fn is_budget_error(e: &crate::base::errors::SymplexError) -> bool {
    matches!(
        e,
        crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz",
            ..
        } | crate::base::errors::SymplexError::BudgetExhausted {
            kind: BudgetKind::LimitSteps,
            ..
        }
    )
}

fn is_infinite(arena: &Arena, e: ExprId) -> bool {
    e == arena.infinity() || e == arena.neg_infinity() || e == arena.complex_infinity()
}

/// `true` if `e` contains `∞`, `−∞`, `zoo`, or `NaN` anywhere.
fn contains_singular_atom(arena: &Arena, e: ExprId) -> bool {
    crate::base::walk::contains(arena, e, arena.infinity())
        || crate::base::walk::contains(arena, e, arena.neg_infinity())
        || crate::base::walk::contains(arena, e, arena.complex_infinity())
        || crate::base::walk::contains(arena, e, arena.nan())
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }
    fn display(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    // ── Constant ──

    #[test]
    fn gruntz_constant() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let inf = a.infinity();
        let result = gruntz(&mut a, five, x, inf).unwrap();
        assert_eq!(display(&a, result), "5");
    }

    // ── Rational functions at infinity ──

    #[test]
    fn gruntz_1_over_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let expr = a.div(one, x);
        let inf = a.infinity();
        let result = gruntz(&mut a, expr, x, inf).unwrap();
        assert_eq!(display(&a, result), "0", "lim(1/x, x→∞) = 0");
    }

    #[test]
    fn gruntz_x_over_x_plus_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let denom = a.add(&[x, one]);
        let expr = a.div(x, denom);
        let inf = a.infinity();
        let result = gruntz(&mut a, expr, x, inf).unwrap();
        assert_eq!(display(&a, result), "1", "lim(x/(x+1), x→∞) = 1");
    }

    #[test]
    fn gruntz_3x2_plus_1_over_x2_plus_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let three = a.int(3);
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let three_x_sq = a.mul(&[three, x_sq]);
        let numer = a.add(&[three_x_sq, one]);
        let denom = a.add(&[x_sq, one]);
        let expr = a.div(numer, denom);
        let inf = a.infinity();
        let result = gruntz(&mut a, expr, x, inf).unwrap();
        assert_eq!(display(&a, result), "3", "lim((3x²+1)/(x²+1), x→∞) = 3");
    }

    #[test]
    fn gruntz_x_over_x2_plus_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let denom = a.add(&[x_sq, one]);
        let expr = a.div(x, denom);
        let inf = a.infinity();
        let result = gruntz(&mut a, expr, x, inf).unwrap();
        assert_eq!(display(&a, result), "0", "lim(x/(x²+1), x→∞) = 0");
    }

    #[test]
    fn gruntz_2x_plus_1_over_x_plus_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        let numer = a.add(&[two_x, one]);
        let denom = a.add(&[x, one]);
        let expr = a.div(numer, denom);
        let inf = a.infinity();
        let result = gruntz(&mut a, expr, x, inf).unwrap();
        assert_eq!(display(&a, result), "2", "lim((2x+1)/(x+1), x→∞) = 2");
    }

    // ── Exponential limits ──

    #[test]
    fn gruntz_exp_neg_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let neg_x = a.neg(x);
        let expr = a.exp(neg_x);
        let inf = a.infinity();
        let result = gruntz(&mut a, expr, x, inf).unwrap();
        assert_eq!(display(&a, result), "0", "lim(exp(-x), x→∞) = 0");
    }

    #[test]
    fn gruntz_x_times_exp_neg_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let neg_x = a.neg(x);
        let exp_neg_x = a.exp(neg_x);
        let expr = a.mul(&[x, exp_neg_x]);
        let inf = a.infinity();
        let result = gruntz(&mut a, expr, x, inf).unwrap();
        assert_eq!(display(&a, result), "0", "lim(x·exp(-x), x→∞) = 0");
    }

    #[test]
    fn gruntz_exp_x_over_x2() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let numer = a.exp(x);
        let denom = a.pow(x, two);
        let expr = a.div(numer, denom);
        let inf = a.infinity();
        let result = gruntz(&mut a, expr, x, inf).unwrap();
        assert!(
            is_infinite(&a, result),
            "lim(exp(x)/x², x→∞) = ∞, got: {}",
            display(&a, result)
        );
    }

    // ── Logarithmic limits ──

    #[test]
    fn gruntz_ln_x_over_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let ln_x = a.ln(x);
        let expr = a.div(ln_x, x);
        let inf = a.infinity();
        let result = gruntz(&mut a, expr, x, inf).unwrap();
        assert_eq!(display(&a, result), "0", "lim(ln(x)/x, x→∞) = 0");
    }

    // ── Finite-point limits via z0 + 1/x substitution ──

    #[test]
    fn gruntz_sin_x_over_x_at_0() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let expr = a.div(sin_x, x);
        let zero = a.zero();
        let result = gruntz(&mut a, expr, x, zero).unwrap();
        assert_eq!(display(&a, result), "1", "lim(sin(x)/x, x→0) = 1");
    }

    #[test]
    fn gruntz_1_minus_cos_over_x2_at_0() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let cos_x = a.cos(x);
        let numer = a.sub(one, cos_x);
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let expr = a.div(numer, x_sq);
        let zero = a.zero();
        let result = gruntz(&mut a, expr, x, zero).unwrap();
        assert_eq!(display(&a, result), "1/2", "lim((1-cos(x))/x², x→0) = 1/2");
    }

    #[test]
    fn gruntz_exp_minus_1_over_x_at_0() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let exp_x = a.exp(x);
        let numer = a.sub(exp_x, one);
        let expr = a.div(numer, x);
        let zero = a.zero();
        let result = gruntz(&mut a, expr, x, zero).unwrap();
        assert_eq!(display(&a, result), "1", "lim((exp(x)-1)/x, x→0) = 1");
    }

    #[test]
    fn gruntz_exp_minus_1_minus_x_over_x2_at_0() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let exp_x = a.exp(x);
        let exp_m1 = a.sub(exp_x, one);
        let numer = a.sub(exp_m1, x);
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let expr = a.div(numer, x_sq);
        let zero = a.zero();
        let result = gruntz(&mut a, expr, x, zero).unwrap();
        assert_eq!(
            display(&a, result),
            "1/2",
            "lim((exp(x)-1-x)/x², x→0) = 1/2"
        );
    }

    #[test]
    fn gruntz_x2_minus_1_over_x_minus_1_at_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let numer = a.sub(x_sq, one);
        let denom = a.sub(x, one);
        let expr = a.div(numer, denom);
        let result = gruntz(&mut a, expr, x, one).unwrap();
        assert_eq!(display(&a, result), "2", "lim((x²-1)/(x-1), x→1) = 2");
    }

    // ── Negative infinity ──

    #[test]
    fn gruntz_exp_x_at_neg_inf() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.exp(x);
        let neg_inf = a.neg_infinity();
        let result = gruntz(&mut a, expr, x, neg_inf).unwrap();
        assert_eq!(display(&a, result), "0", "lim(exp(x), x→-∞) = 0");
    }

    // ── Direct substitution ──

    #[test]
    fn gruntz_polynomial_at_2() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let two_c = a.int(2);
        let x_sq = a.pow(x, two_c);
        let expr = a.add(&[x_sq, one]);
        let two = a.int(2);
        let result = gruntz(&mut a, expr, x, two).unwrap();
        assert_eq!(display(&a, result), "5", "lim(x²+1, x→2) = 5");
    }

    // ── Work budget ──

    /// Representative hard limits must stay far below the work cap, so that
    /// the cap only ever triggers on pathological inputs.
    #[test]
    fn budget_headroom_on_hard_limits() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let inf = a.infinity();
        let zero = a.zero();
        let one = a.one();
        let two = a.int(2);
        let three = a.int(3);

        // (tan x − sin x)/x³ at 0 → 1/2 (series-heavy)
        let tan_x = a.tan(x);
        let sin_x = a.sin(x);
        let num = a.sub(tan_x, sin_x);
        let x3 = a.pow(x, three);
        let e1 = a.div(num, x3);
        // exp(x − exp(−x)) − exp(x) at ∞ → −1 (nested MRV classes)
        let neg_x = a.neg(x);
        let exp_neg_x = a.exp(neg_x);
        let arg = a.sub(x, exp_neg_x);
        let e_arg = a.exp(arg);
        let exp_x = a.exp(x);
        let e2 = a.sub(e_arg, exp_x);
        // (1/x² − 1/sin²x) at 0 → −1/3 (cancellation of poles)
        let x2 = a.pow(x, two);
        let inv_x2 = a.div(one, x2);
        let sin2 = a.pow(sin_x, two);
        let inv_sin2 = a.div(one, sin2);
        let e3 = a.sub(inv_x2, inv_sin2);

        for (e, p, want) in [(e1, zero, "1/2"), (e2, inf, "-1"), (e3, zero, "-1/3")] {
            let mut budget = Budget::new();
            let r = gruntz_with_budget(&mut a, e, x, p, &mut budget).unwrap();
            assert_eq!(display(&a, r), want);
            assert!(
                budget.work() < MAX_WORK / 4,
                "work {} too close to the cap {MAX_WORK} for {}",
                budget.work(),
                display(&a, e)
            );
        }
    }

    /// An exhausted budget aborts with `BudgetExhausted` instead of
    /// spinning (up to 0.49 `ComputationFailed`).
    #[test]
    fn exhausted_budget_fails_cleanly() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let inf = a.infinity();
        let neg_x = a.neg(x);
        let exp_neg_x = a.exp(neg_x);
        let e = a.mul(&[x, exp_neg_x]);
        let mut budget = Budget::new();
        budget.work.charge(MAX_WORK).unwrap();
        let err = gruntz_with_budget(&mut a, e, x, inf, &mut budget).unwrap_err();
        assert!(
            matches!(
                err,
                crate::base::errors::SymplexError::BudgetExhausted {
                    kind: BudgetKind::LimitSteps,
                    ..
                }
            ),
            "{err}"
        );
        assert!(err.to_string().contains("budget"), "{err}");
    }

    /// The function lists of the limit engine select exactly the functions
    /// of their 0.49 or-patterns (W6d: each list became a predicate over
    /// `Func`, the library and internal functions included).
    #[test]
    fn function_predicates_are_the_old_lists() {
        use crate::base::func::{Family, Func::*, selected};
        use std::collections::HashSet;
        let set = |fs: &[crate::base::func::Func]| fs.iter().copied().collect::<HashSet<_>>();
        let mut growth = set(&[
            Si, Ci, Ei, Li, Zeta, Polygamma, Beta, Erf, Erfc, Gamma, LogGamma, Digamma, LambertW0,
            Factorial,
        ]);
        growth.extend(selected(|f| f.family() != Family::Node));
        assert_eq!(selected(growth_checked), growth);
        assert_eq!(
            selected(leadterm_unary),
            set(&[
                Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh, Erf, Erfc,
                Gamma, LogGamma, Digamma, LambertW0, Abs, Sign, Heaviside,
            ])
        );
        assert_eq!(
            selected(always_rewritten),
            set(&[
                Tan, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh, Abs, Sign, Heaviside, DiracDelta,
                Floor, Ceiling, Min, Max, Atan2, Arg,
            ])
        );
        assert_eq!(
            selected(real_on_reals),
            set(&[
                Exp, Sin, Cos, Tan, Sinh, Cosh, Tanh, Atan, Asinh, Abs, Sign, Heaviside, Floor,
                Ceiling, Erf, Erfc,
            ])
        );
        assert_eq!(
            selected(real_analytic_on_reals),
            set(&[Exp, Sin, Cos, Tan, Sinh, Cosh, Tanh, Atan, Asinh, Erf, Erfc])
        );
        assert_eq!(
            selected(expanded_as_series),
            set(&[
                Exp, Ln, Sin, Cos, Tan, Asin, Atan, Sinh, Cosh, Tanh, Asinh, Atanh, Erf
            ])
        );
    }
}
