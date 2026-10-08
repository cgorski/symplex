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
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::calculus::calculus_util::settle_constant;

/// Maximum recursion depth for the Gruntz algorithm.
const MAX_DEPTH: usize = 15;

/// Total work units one top-level Gruntz invocation may spend before it
/// gives up. Every recursive step of the algorithm (`limitinf`, `mrv`,
/// `leadterm`, `compare`, `sign_at_inf`, `rewrite`) costs one unit and every
/// series expansion costs [`SERIES_COST`]. The battery of ~100 standard
/// limits in `tests/v02_transforms_limits.rs` uses well under 2 000 units
/// per invocation; the cap is set an order of magnitude higher so that
/// legitimate limits never hit it while pathological inputs are cut off
/// within a fraction of a second.
const MAX_WORK: u32 = 20_000;

/// Work charged for one series expansion (see [`MAX_WORK`]), on top of a
/// size-proportional charge.
const SERIES_COST: u32 = 25;

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
/// exceeding its depth limit. The budget bounds the total amount of work:
/// once it is exhausted every step fails with `ComputationFailed`, the
/// error propagates to the top, and the caller returns an unevaluated
/// `Limit` node.
///
/// The budget also hands out the fresh dummy symbols (`__gw0`, `__gw1`, …)
/// used for MRV substitutions.
#[derive(Debug)]
pub(crate) struct Budget {
    dummies: u32,
    work: u32,
}

impl Budget {
    pub(crate) fn new() -> Self {
        Self {
            dummies: 0,
            work: 0,
        }
    }

    /// Charge `weight` units of work; `Err` once the budget is exhausted.
    fn tick(&mut self, weight: u32) -> Result<(), crate::base::errors::SymplexError> {
        self.work = self.work.saturating_add(weight);
        if self.work > MAX_WORK {
            tracing::warn!(work = self.work, "gruntz: work budget exhausted");
            return Err(crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz",
                reason: "work budget exhausted (expression too complex for the limit engine)"
                    .into(),
            });
        }
        Ok(())
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
    fn work(&self) -> u32 {
        self.work
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
    for order in [4i64, 10] {
        budget.tick(SERIES_COST)?;
        budget.charge_size(arena, f)?;
        let ts = crate::calculus::series::expand_leading(arena, f, w, order, logw)?;
        for e in ts.shift()..ts.known() {
            let c = ts.coefficient(arena, e);
            if arena.is_zero_structural(c) {
                continue;
            }
            let c = settle_constant(arena, c);
            if arena.is_zero_structural(c) || is_rational_zero(arena, c) {
                continue;
            }
            if crate::base::walk::contains(arena, c, w) || contains_singular_atom(arena, c) {
                break;
            }
            tracing::debug!(f = %arena.display(f), c = %arena.display(c), e, "gruntz::leadterm_by_tseries");
            return Ok((c, arena.int(e)));
        }
    }
    Err(crate::base::errors::SymplexError::ComputationFailed {
        operation: "gruntz::leadterm",
        reason: "the series expansion has no non-zero term within its precision".into(),
    })
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
    budget.charge_size(arena, f).ok()?;
    let ts = crate::calculus::series::expand_leading(arena, f, w, 10, logw).ok()?;
    if ts.known() <= 0 {
        return None;
    }
    for e in ts.shift()..ts.known() {
        let c = ts.coefficient(arena, e);
        if arena.is_zero_structural(c) {
            continue;
        }
        let settled = settle_constant(arena, c);
        if !arena.is_zero_structural(settled) && !is_rational_zero(arena, settled) {
            return None;
        }
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
            .min()
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
        entries.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
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
    let (c0, _e0) = mrv_leadterm(arena, e, x, depth + 1, budget, false)?;
    if contains_foreign_dummy(arena, c0, x) {
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::sign_at_inf",
            reason: "leading coefficient is not a genuine constant".into(),
        });
    }
    if arena.is_zero_structural(c0) {
        return Ok(0);
    }
    let result = sign_at_inf(arena, c0, x, depth + 1, budget);
    tracing::debug!(sign = ?result, "gruntz::sign_at_inf result");
    result
}

/// Determine the sign of a constant expression (no free variable x).
///
/// Consults symbol assumptions (`a > 0`) through
/// [`limit::const_sign`](crate::calculus::limit::const_sign).
fn sign_of_constant(
    arena: &mut Arena,
    e: ExprId,
) -> Result<i32, crate::base::errors::SymplexError> {
    match crate::calculus::limit::const_sign(arena, e) {
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
                        return mrv(arena, split, x, depth + 1, budget);
                    }
                }
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
        ExprNode::Apply(f, ref args) if args.len() == 1 && is_internal_asymptotic(arena, f) => {
            let (s, r) = mrv(arena, args[0], x, depth + 1, budget)?;
            Ok((s, arena.intern(ExprNode::Apply(f, smallvec::smallvec![r]))))
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
    let function_like = matches!(
        node,
        ExprNode::Apply(..)
            | ExprNode::Si(_)
            | ExprNode::Ci(_)
            | ExprNode::Ei(_)
            | ExprNode::Li(_)
            | ExprNode::Zeta(_)
            | ExprNode::Polygamma(..)
            | ExprNode::Beta(..)
            | ExprNode::Erf(_)
            | ExprNode::Erfc(_)
            | ExprNode::Gamma(_)
            | ExprNode::LogGamma(_)
            | ExprNode::Digamma(_)
            | ExprNode::LambertW(_)
            | ExprNode::Factorial(_)
    );
    if !function_like {
        return Ok(());
    }
    let tame_at = |arena: &Arena, plus: bool| -> bool {
        match node {
            ExprNode::Si(_) | ExprNode::Ci(_) => true,
            ExprNode::LogGamma(_) | ExprNode::Digamma(_) | ExprNode::LambertW(_) => plus,
            ExprNode::Polygamma(m, _) => plus && !crate::base::walk::contains(arena, *m, x),
            ExprNode::Apply(f, _) => matches!(
                arena.lib_fn(*f),
                Some(LibFn::BesselJ | LibFn::BesselY | LibFn::FresnelS | LibFn::FresnelC)
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
                if !tame_at(arena, l == arena.infinity()) {
                    return Err(unknown());
                }
            }
            Err(e) if is_budget_error(&e) => return Err(e),
            _ => return Err(unknown()),
        }
    }
    Ok(())
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

    // All entries must be Exp nodes. Pick the last one as g (representative).
    // (SymPy sorts by expression tree height; we just pick one.)
    let (g_expr, _g_dummy) = entries[entries.len() - 1];

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
        let c_display = arena.display(c).to_string();
        let f_exp_display = arena.display(f_exp).to_string();
        let g_exp_display = arena.display(g_exp).to_string();
        tracing::debug!(f_exp = %f_exp_display, g_exp = %g_exp_display, c = %c_display, "gruntz::rewrite: c = limitinf(f_exp/g_exp)");

        // Compute remainder = f_exp - c * g_exp
        let c_times_gexp = arena.mul(&[c, g_exp]);
        let remainder = arena.sub(f_exp, c_times_gexp);
        let remainder = crate::transforms::eval::eval(arena, remainder);

        // f = exp(remainder) * w^(±c)
        let exp_remainder = if arena.is_zero_structural(remainder) {
            arena.one()
        } else {
            arena.exp(remainder)
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

    // If f doesn't depend on w: (f, 0)
    if !crate::base::walk::contains(arena, f, w) {
        tracing::trace!("gruntz::leadterm: constant wrt ω → (f, 0)");
        return Ok((f, arena.zero()));
    }

    // If f IS w: (1, 1)
    if f == w {
        tracing::trace!("gruntz::leadterm: f == ω → (1, 1)");
        return Ok((arena.one(), arena.one()));
    }

    let node = arena.node(f).clone();

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
            if arena.as_num(exp).is_none_or(|r| !r.is_integer())
                && arena.is_zero_structural(e_b)
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
            let mut terms: Vec<(ExprId, ExprId, ExpKey)> = Vec::new();
            for &child in children {
                let (c, e) = leadterm(arena, child, w, logw, x, depth + 1, budget)?;
                let e_eval = crate::transforms::eval::eval(arena, e);
                let key = exponent_key(arena, e_eval).ok_or_else(|| {
                    crate::base::errors::SymplexError::ComputationFailed {
                        operation: "gruntz::leadterm",
                        reason: format!(
                            "cannot order the exponent {} of a leading term",
                            arena.display(e_eval)
                        ),
                    }
                })?;
                terms.push((c, e_eval, key));
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
                    let d = settle_constant(arena, d);
                    if !arena.is_zero_structural(d) {
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
                    let settled = settle_constant(arena, coeff_sum);
                    let needs = crate::base::walk::contains(arena, coeff_sum, w)
                        || arena.is_zero_structural(settled);
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
            coeff_sum = settle_constant(arena, coeff_sum);

            // If the leading coefficients cancel to 0 OR the coefficient
            // still depends on w (meaning further decomposition is needed),
            // fall back to series expansion.
            //
            // Example: (exp(w) - 1)/w after expansion is exp(w)/w - 1/w.
            // Both terms have exponent -1. Coefficient sum is exp(w) - 1,
            // which still contains w. The TRUE leading term requires knowing
            // that exp(w) - 1 ≈ w, so the product is w * w^(-1) = w^0.
            // Only series expansion can resolve this.
            let needs_series = arena.is_zero_structural(coeff_sum)
                || crate::base::walk::contains(arena, coeff_sum, w)
                || is_rational_zero(arena, coeff_sum);

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
            if arena.is_zero_structural(e_eval)
                && crate::calculus::limit::on_branch_cut(
                    arena,
                    crate::calculus::limit::BranchCut::NegativeReals,
                    c_arg,
                )
                && !crate::calculus::limit::inner_known_real(arena, arg)
            {
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
                let settled = settle_constant(arena, c_minus_one);
                let c_is_one = arena.is_zero_structural(coeff) || arena.is_zero_structural(settled);
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

        // ── Unary functions of an ω-dependent argument ──
        ExprNode::Sin(inner)
        | ExprNode::Cos(inner)
        | ExprNode::Tan(inner)
        | ExprNode::Asin(inner)
        | ExprNode::Acos(inner)
        | ExprNode::Atan(inner)
        | ExprNode::Sinh(inner)
        | ExprNode::Cosh(inner)
        | ExprNode::Tanh(inner)
        | ExprNode::Asinh(inner)
        | ExprNode::Acosh(inner)
        | ExprNode::Atanh(inner)
        | ExprNode::Erf(inner)
        | ExprNode::Erfc(inner)
        | ExprNode::Gamma(inner)
        | ExprNode::LogGamma(inner)
        | ExprNode::Digamma(inner)
        | ExprNode::LambertW(inner)
        | ExprNode::Abs(inner)
        | ExprNode::Sign(inner)
        | ExprNode::Heaviside(inner)
            if crate::base::walk::contains(arena, inner, w) =>
        {
            unary_leadterm(arena, f, &node, inner, w, logw, x, depth, budget).or_else(|e| {
                if is_budget_error(&e) {
                    return Err(e);
                }
                leadterm_by_tseries(arena, budget, f, w, logw).map_err(|_| e)
            })
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
    match template {
        ExprNode::Sin(_) => arena.sin(arg),
        ExprNode::Cos(_) => arena.cos(arg),
        ExprNode::Tan(_) => arena.tan(arg),
        ExprNode::Asin(_) => arena.asin(arg),
        ExprNode::Acos(_) => arena.acos(arg),
        ExprNode::Atan(_) => arena.atan(arg),
        ExprNode::Sinh(_) => arena.sinh(arg),
        ExprNode::Cosh(_) => arena.cosh(arg),
        ExprNode::Tanh(_) => arena.tanh(arg),
        ExprNode::Asinh(_) => arena.asinh(arg),
        ExprNode::Acosh(_) => arena.acosh(arg),
        ExprNode::Atanh(_) => arena.atanh(arg),
        ExprNode::Erf(_) => arena.erf(arg),
        ExprNode::Erfc(_) => arena.erfc(arg),
        ExprNode::Gamma(_) => arena.gamma(arg),
        ExprNode::LogGamma(_) => arena.log_gamma(arg),
        ExprNode::Digamma(_) => arena.digamma(arg),
        ExprNode::LambertW(_) => arena.lambertw(arg),
        ExprNode::Factorial(_) => arena.factorial(arg),
        ExprNode::Abs(_) => arena.abs(arg),
        ExprNode::Sign(_) => arena.sign(arg),
        ExprNode::Heaviside(_) => arena.heaviside(arg),
        ExprNode::Exp(_) => arena.exp(arg),
        ExprNode::Ln(_) => arena.ln(arg),
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
        let v = apply_unary(arena, template, c_in);
        let v = crate::transforms::eval::eval(arena, v);
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

    // Step 4: Extract leading term
    let (c0, e0) = match leadterm(arena, f, w, logw, x, depth + 1, budget) {
        Ok(r) => r,
        Err(err) if limit_only && !is_budget_error(&err) => {
            match vanishes_to_positive_order(arena, budget, f, w, logw) {
                Some(k) => (arena.zero(), arena.int(k)),
                None => return Err(err),
            }
        }
        Err(err) => return Err(err),
    };

    let c0_display = arena.display(c0).to_string();
    let e0_display = arena.display(e0).to_string();
    tracing::debug!(c0 = %c0_display, e0 = %e0_display, "gruntz::mrv_leadterm: leading term extracted");

    // Replace log(w) → logw in c0 (SymPy does this as the last step)
    let ln_w = arena.ln(w);
    let c0 = crate::transforms::subs::subs(arena, c0, ln_w, logw);

    tracing::debug!("gruntz::mrv_leadterm: done");
    Ok((c0, e0))
}

/// Compute `lim(x→∞) e` using the Gruntz algorithm.
pub(crate) fn limitinf(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
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

    let c0_display = arena.display(c0).to_string();
    let e0_display = arena.display(e0_eval).to_string();
    tracing::debug!(c0 = %c0_display, e0 = %e0_display, "gruntz::limitinf: leading term extracted, checking exponent sign");

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
            let c0_sign = sign_at_inf(arena, c0, x, depth + 1, budget)?;
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
            limitinf(arena, c0, x, depth + 1, budget)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tractable rewriting
// ═══════════════════════════════════════════════════════════════════════════

/// `true` if `e` contains a node that [`rewrite_tractable`] would touch.
fn needs_tractable_rewrite(arena: &Arena, e: ExprId) -> bool {
    let mut stack = vec![e];
    let mut visited = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        let node = arena.node(id);
        if matches!(
            node,
            ExprNode::Tan(_)
                | ExprNode::Sinh(_)
                | ExprNode::Cosh(_)
                | ExprNode::Tanh(_)
                | ExprNode::Asinh(_)
                | ExprNode::Acosh(_)
                | ExprNode::Atanh(_)
                | ExprNode::Abs(_)
                | ExprNode::Sign(_)
                | ExprNode::Heaviside(_)
                | ExprNode::DiracDelta(_)
                | ExprNode::Floor(_)
                | ExprNode::Ceiling(_)
                | ExprNode::Piecewise(_)
                | ExprNode::Min(_)
                | ExprNode::Max(_)
        ) {
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
///
/// Nodes whose eventual sign cannot be determined are left untouched.
fn rewrite_tractable(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    if !needs_tractable_rewrite(arena, e) {
        return Ok(e);
    }

    let post = crate::base::walk::post_order_ids(arena, e);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post {
        if !crate::base::walk::contains(arena, id, x) {
            cache.insert(id, id);
            continue;
        }
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let node = arena.node(rebuilt).clone();
        check_branch_cut_approach(arena, &node, x, depth, budget)?;
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
            ExprNode::Erf(u) | ExprNode::Erfc(u) | ExprNode::Ei(u) | ExprNode::Li(u) => {
                match limitinf(arena, u, x, depth + 1, budget) {
                    Ok(l) if l == arena.infinity() || l == arena.neg_infinity() => {
                        asymptotic_rewrite(arena, &node, u, l == arena.infinity())
                            .unwrap_or(rebuilt)
                    }
                    Err(e) if is_budget_error(&e) => return Err(e),
                    _ => rebuilt,
                }
            }
            // The eventual sign decides |u|, sign u, H(u) only for a real u:
            // `(−2)^(−1/x)` tends to 1 but is not real, and taking it for
            // positive made `lim_{x→0⁺} |(−2)^(−x)|^(1/x)` equal −1/2.
            ExprNode::Abs(u) | ExprNode::Sign(u) | ExprNode::Heaviside(u)
                if !eventually_real(arena, u, x, depth + 1, budget) =>
            {
                rebuilt
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
                        Some(true) => {
                            chosen = Some(val);
                            break;
                        }
                        Some(false) => continue,
                        None => break,
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
    Ok(result)
}

/// `f(z)` of an error function, `Ei` or `li` with `z → +∞` (`positive`) or
/// `−∞`, written with its exponential factor explicit and an internal
/// function ([`ERFCX_ASYMPTOTIC`](crate::calculus::series::ERFCX_ASYMPTOTIC),
/// [`EI_ASYMPTOTIC`](crate::calculus::series::EI_ASYMPTOTIC)) whose
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
    use crate::calculus::series::{EI_ASYMPTOTIC, ERFCX_ASYMPTOTIC};
    let internal = |arena: &mut Arena, name: &str, arg: ExprId| -> ExprId {
        let head = arena.symbol(name);
        let ExprNode::Symbol(sid) = *arena.node(head) else {
            return head;
        };
        arena.intern(ExprNode::Apply(sid, smallvec::smallvec![arg]))
    };
    match *node {
        ExprNode::Erf(_) | ExprNode::Erfc(_) => {
            let z = if positive { u } else { arena.neg(u) };
            let two = arena.int(2);
            let u2 = arena.pow(u, two);
            let nu2 = arena.neg(u2);
            let damp = arena.exp(nu2);
            let e = internal(arena, ERFCX_ASYMPTOTIC, z);
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
            let f = internal(arena, EI_ASYMPTOTIC, u);
            Some(arena.mul(&[e, f]))
        }
        ExprNode::Li(_) if positive => {
            let l = arena.ln(u);
            let f = internal(arena, EI_ASYMPTOTIC, l);
            Some(arena.mul(&[u, f]))
        }
        _ => None,
    }
}

/// Is `f` the head of one of the internal asymptotic functions of
/// [`asymptotic_rewrite`]?
fn is_internal_asymptotic(arena: &Arena, f: crate::base::node::SymbolId) -> bool {
    use crate::calculus::series::{EI_ASYMPTOTIC, ERFCX_ASYMPTOTIC};
    matches!(arena.symbol_name(f), ERFCX_ASYMPTOTIC | EI_ASYMPTOTIC)
}

/// Does `e` contain one of the internal asymptotic functions?
fn contains_internal_asymptotic(arena: &Arena, e: ExprId) -> bool {
    crate::base::walk::post_order_ids(arena, e).into_iter().any(
        |id| matches!(arena.node(id), ExprNode::Apply(f, _) if is_internal_asymptotic(arena, *f)),
    )
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

/// Refuse a branch function (`ln`, a fractional power, `asin`, `acos`,
/// `atanh`, `acosh`, `asinh`, `atan`) whose argument is not eventually
/// real and tends to a point of the function's cut: the limit then
/// depends on the side from which the argument approaches, which the
/// leading terms of Gruntz's expansion do not keep (`asinh(atanh(1 − x))`
/// gave `asinh(iπ/2)`, the value on the cut, as its limit at `∞`;
/// the argument approaches from `Re < 0`, where the limit is
/// `−acosh(π/2) + iπ/2`).  A real argument moves along the cut, where
/// the principal values are continuous.
fn check_branch_cut_approach(
    arena: &mut Arena,
    node: &ExprNode,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<(), crate::base::errors::SymplexError> {
    use crate::calculus::limit::BranchCut;
    let (arg, cut) = match *node {
        ExprNode::Ln(a) => (a, BranchCut::NegativeReals),
        ExprNode::Pow(b, e) if arena.as_num(e).is_some_and(|r| !r.is_integer()) => {
            (b, BranchCut::NegativeReals)
        }
        ExprNode::Asin(a) | ExprNode::Acos(a) | ExprNode::Atanh(a) => (a, BranchCut::BeyondOne),
        ExprNode::Acosh(a) => (a, BranchCut::BelowOne),
        ExprNode::Asinh(a) | ExprNode::Atan(a) => (a, BranchCut::ImaginaryAxis),
        _ => return Ok(()),
    };
    if !crate::base::walk::contains(arena, arg, x)
        || eventually_real(arena, arg, x, depth + 1, budget)
    {
        return Ok(());
    }
    let Ok(l) = limitinf(arena, arg, x, depth + 1, budget) else {
        return Ok(());
    };
    if !is_infinite(arena, l) && crate::calculus::limit::on_branch_cut(arena, cut, l) {
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz",
            reason: "a non-real argument approaches the branch cut of a function".into(),
        });
    }
    Ok(())
}

/// Is `e` real for all sufficiently large `x` (`x` itself is real and
/// positive)?  Structural and conservative: rational constants, `π`, `e`,
/// `x`; sums, products, `exp`, trigonometric and hyperbolic functions,
/// `atan`, `asinh`, `|·|`, `sign` of real arguments; `b^n` for an integer
/// `n`, `b^g` and `ln b` for an eventually positive real `b`; `asin`, `acos`,
/// `atanh` of a real `u` with `1 − u²` eventually positive, `acosh u` with
/// `u − 1` eventually positive.  Anything else is not known to be real.
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
                    cache.query(arena, id, crate::base::assumptions::Props::REAL) == Some(true)
                }
            }
            ExprNode::Add(ref cs) | ExprNode::Mul(ref cs) => cs.iter().all(|&c| r(c)),
            ExprNode::Neg(a)
            | ExprNode::Exp(a)
            | ExprNode::Sin(a)
            | ExprNode::Cos(a)
            | ExprNode::Tan(a)
            | ExprNode::Sinh(a)
            | ExprNode::Cosh(a)
            | ExprNode::Tanh(a)
            | ExprNode::Atan(a)
            | ExprNode::Asinh(a)
            | ExprNode::Erf(a)
            | ExprNode::Erfc(a) => r(a),
            ExprNode::Abs(_) | ExprNode::Sign(_) | ExprNode::Heaviside(_) => true,
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
            _ => false,
        };
        real.insert(id, v);
    }
    real.get(&e).copied().unwrap_or(false)
}

/// Decide whether a boolean condition holds for all sufficiently large `x`.
fn eventually_true(
    arena: &mut Arena,
    cond: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Option<bool> {
    if depth > MAX_DEPTH {
        return None;
    }
    let node = arena.node(cond).clone();
    match node {
        ExprNode::BoolTrue => Some(true),
        ExprNode::BoolFalse => Some(false),
        ExprNode::Gt(a, b) | ExprNode::Ge(a, b) | ExprNode::Eq_(a, b) | ExprNode::Ne(a, b) => {
            let diff = arena.sub(a, b);
            let s = sign_at_inf(arena, diff, x, depth + 1, budget).ok()?;
            Some(match node {
                ExprNode::Gt(..) => s > 0,
                ExprNode::Ge(..) => s >= 0,
                ExprNode::Eq_(..) => s == 0,
                _ => s != 0,
            })
        }
        ExprNode::And(children) => {
            let mut all_true = true;
            for c in children {
                match eventually_true(arena, c, x, depth + 1, budget) {
                    Some(true) => {}
                    Some(false) => return Some(false),
                    None => all_true = false,
                }
            }
            if all_true { Some(true) } else { None }
        }
        ExprNode::Or(children) => {
            let mut all_false = true;
            for c in children {
                match eventually_true(arena, c, x, depth + 1, budget) {
                    Some(true) => return Some(true),
                    Some(false) => {}
                    None => all_false = false,
                }
            }
            if all_false { Some(false) } else { None }
        }
        ExprNode::Not(inner) => eventually_true(arena, inner, x, depth + 1, budget).map(|b| !b),
        _ => None,
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
    let r = limitinf(arena, e, x, 0, &mut budget)?;
    let r = validate_result(arena, r, x)?;
    // `e^{½ ln 2 + ½ ln π}` (from Stirling's constant) is `√2·√π`.
    Ok(simplify_exp_log(arena, r))
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
        let r = limitinf(arena, e_sub, x, 0, budget)?;
        let r = validate_result(arena, r, x)?;
        let r = simplify_exp_log(arena, r);
        return validate_result(arena, r, z);
    }

    if z0 == arena.neg_infinity() {
        tracing::debug!("gruntz: limit at -∞, substituting z = -x");
        let x = fresh_positive_dummy(arena, budget);
        let neg_x = arena.neg(x);
        let e_sub = crate::transforms::subs::subs(arena, e, z, neg_x);
        let r = limitinf(arena, e_sub, x, 0, budget)?;
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
    let r = limitinf(arena, e_simplified, x, 0, budget)?;
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
            ExprNode::Exp(arg)
            | ExprNode::Ln(arg)
            | ExprNode::Sin(arg)
            | ExprNode::Cos(arg)
            | ExprNode::Tan(arg)
            | ExprNode::Asin(arg)
            | ExprNode::Atan(arg)
            | ExprNode::Sinh(arg)
            | ExprNode::Cosh(arg)
            | ExprNode::Tanh(arg)
            | ExprNode::Asinh(arg)
            | ExprNode::Atanh(arg)
            | ExprNode::Erf(arg) => crate::base::walk::contains(arena, arg, w),
            // Non-integer powers of a regular base, e.g. `(1 + ω)^{1/2}`.
            ExprNode::Pow(base, e) => {
                crate::base::walk::contains(arena, base, w)
                    && arena.as_num(e).is_some_and(|r| !r.is_integer())
            }
            _ => false,
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

    /// An exhausted budget aborts with `ComputationFailed` instead of spinning.
    #[test]
    fn exhausted_budget_fails_cleanly() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let inf = a.infinity();
        let neg_x = a.neg(x);
        let exp_neg_x = a.exp(neg_x);
        let e = a.mul(&[x, exp_neg_x]);
        let mut budget = Budget {
            dummies: 0,
            work: MAX_WORK,
        };
        let err = gruntz_with_budget(&mut a, e, x, inf, &mut budget).unwrap_err();
        assert!(err.to_string().contains("budget"), "{err}");
    }
}
