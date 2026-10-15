//! Symbolic equation solving.
//!
//! This module implements [`solutions`] and [`solve_classified`], which find
//! the values of a variable that make an expression equal to zero.
//!
//! # Supported equation types
//!
//! - **Linear:** `a*x + b = 0` → `x = -b/a` (numeric or symbolic `a`, `b`)
//! - **Quadratic:** `a*x² + b*x + c = 0` → quadratic formula (numeric or
//!   symbolic coefficients)
//! - **Cubic / quartic:** Cardano and Ferrari on the irreducible factors
//! - **Binomial:** `a*xⁿ + b = 0` → all `n` complex roots via roots of unity
//! - **Higher degree:** exact factorisation over ℤ (Berlekamp–Zassenhaus);
//!   rational roots from the linear factors, radicals for the quadratic to
//!   quartic ones, `RootOf` placeholders only for irreducible factors of
//!   degree ≥ 5
//! - **Transcendental:** `exp`, `ln`, `sin`, `cos`, `tan`, hyperbolic and
//!   inverse functions, `|·|`, constant-base exponentials, all by inversion
//!   peeling (principal branches, or full periodic families in
//!   [`solve_general`])
//! - **Change of variable:** equations polynomial in `f(x)` for some `f`
//! - **Rational equations:** `P(x)/Q(x) = 0` (numerator roots that are not
//!   poles; over ℚ the factors shared with `Q` are divided out exactly)
//! - **Radicals:** rational powers of one linear `a·x + b`
//!   (`√(x + 1) = x − 1`), square roots of one or two polynomials
//!   (`√(x + 5) + √x = 3`, by the norm over the sign choices, SymPy's
//!   `unrad`), every candidate checked
//! - **Symbolic coefficients:** polynomials factored over `ℤ[x, params]`
//! - **Substitutions:** exponentials of one rate or a common base
//!   (`eˣ + e⁻ˣ = 3`, `4ˣ − 5·2ˣ + 6 = 0`), hyperbolic functions through
//!   exponentials (`sinh x + cosh x = 3`), trigonometric functions of one
//!   angle by `t = tan(x/2)` (`sin x + cos x = 1`, `sin 2x = sin x`,
//!   `tan 2x = tan x`), sums of logarithms and one logarithm of a
//!   non-linear argument (exponentiated, candidates checked); a
//!   substituted polynomial with algebraic coefficients is solved over
//!   the number field they generate (`sin 4x = √3·sin x`, `cos³x =
//!   √2·cos x`, [`algebraic_roots`])
//! - **Number fields:** polynomials whose coefficients generate a field
//!   beyond square roots (`(x − ∛2)·(x − i)²`), or with an irreducible
//!   factor of degree ≥ 5 over one (`x⁵ + √2·x + 1`, `RootOf`s of its norm)
//! - **Lambert W:** mixed polynomial–exponential and logarithmic forms
//!   (`x·eˣ = c`, `eˣ = 2x + π`, `x·ln x = c`, `xˣ = c`), a power times
//!   an exponential (`x² = 2ˣ`, `x²·eˣ = 3`, `(x + 1)·eˣ = 2`), also of a
//!   power of the unknown (`x·e^(x²) = c`, `x²·e^(√x/2) = 2`), powers
//!   with a logarithm (`x²·ln x = 1`, `x·ln²x = c`), every real branch
//!   (`W₀`, `W₋₁`), exact where `W_k(y·e^y) = y` (`W(√2·ln(2)/2) = ln(2)/2`)
//! - **Polynomial denominators:** the numerator of `P(x)/Q(x)` for a
//!   transcendental `P` (`ln(x)/x = 1/4`), its candidates checked
//!
//! # Design
//!
//! The solver works by:
//! 1. Classifying degenerate cases (identity `0 = 0`, contradiction `c = 0`).
//! 2. Converting the expression to a [`Poly`] in the given variable.
//! 3. Applying degree-specific solvers (linear, quadratic, …).
//! 4. Falling back to symbolic-coefficient and transcendental strategies.
//!
//! Every entry point answers with [`Solutions`]: every root, some of them,
//! or nothing claimed, over no root (a proof), every value, points or
//! periodic families (see [`outcome`]).  [`known_real_roots`] is the
//! lossy view of the crate's root searches (the roots claimed, or none).
//!
//! [`outcome`]: self::outcome

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode, SymbolId};
use crate::base::numeric::Q;
use crate::poly::Poly;
use crate::poly::polybridge;

mod algebraic_roots;
mod context;
mod outcome;

pub(crate) use context::{Domain, Params, SolveCx};
pub(crate) use outcome::{Guarded, Missing, Reason, RootSet, Solutions};

/// A solution to an equation, with the variable and its value.
#[derive(Clone, Debug)]
pub struct Solution {
    /// The value of the variable that satisfies the equation.
    pub value: ExprId,
}

// ═══════════════════════════════════════════════════════════════════════════
// Public entry points
// ═══════════════════════════════════════════════════════════════════════════

/// The roots of `expr = 0` in `var` taken real that a principal solve
/// claims ([`Solutions::claimed_points`]): every root found, each
/// **distinct** root once whatever its multiplicity (`(x − 1)² = 0` gives
/// `[1]`), principal branches only; **none** when it proves there is none,
/// finds every value a root, or finds nothing.  Up to 0.48 this was `solve`
/// (a `Vec` from `SolveOutcome::into_solutions`).
///
/// The policy of the crate's root searches (poles and critical points in
/// integration, limits, residues, ODEs, summation), which read "nothing
/// found" as "no root here"; a caller for which a refusal must not pass
/// for "no root" uses [`solve_classified_real`] and matches the outcome
/// (or [`Solutions::points_if_complete`]).  The range restrictions of
/// transcendental inversions take the unknown real (`eˣ = −1`, `sin x =
/// 2` have no solution), unlike [`solve_classified`].
///
/// A root that holds only for some values of the parameters is taken as a
/// candidate here ([`SolveCx::conditional_roots_as_candidates`]), where
/// [`solve_classified_real`] refuses.
pub(crate) fn known_real_roots(arena: &mut Arena, expr: ExprId, var: ExprId) -> Vec<ExprId> {
    solutions(
        arena,
        expr,
        var,
        &SolveCx::principal(Domain::Real).as_candidates(),
    )
    .claimed_points()
}

/// Why the public solvers (`Ex::solve`, `Ex::solve_general`) return no
/// list of roots ([`roots_or_refusal`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// Every value is a root.
    Identity,
    /// There is no root, for the reason given.
    NoSolution(String),
    /// The roots were not found, or not all of them (the reason given).
    Failed(String),
}

/// The roots of a complete outcome (`Ok`; none when nothing was found,
/// which the public solvers decide further), or why there is no list: a
/// partial outcome is a failure naming the roots found — up to 0.48 they
/// came back as if they were all (`|x| = 1` for a complex `x`: `[1, −1]`,
/// where the solutions are the unit circle).
pub(crate) fn roots_or_refusal(arena: &Arena, outcome: Solutions) -> Result<Vec<ExprId>, Refusal> {
    match outcome {
        Solutions::Complete(RootSet::All) => Err(Refusal::Identity),
        Solutions::Complete(RootSet::Empty(reason)) => {
            Err(Refusal::NoSolution(reason.text().into_owned()))
        }
        Solutions::Refused(Reason::NotFound) => Ok(Vec::new()),
        Solutions::Refused(reason) => Err(Refusal::Failed(reason.text_in(arena).into_owned())),
        Solutions::Partial(set, Missing::Alternative(reason)) => {
            let shown: Vec<String> = set
                .roots()
                .unwrap_or(&[])
                .iter()
                .map(|&r| arena.display(r).to_string())
                .collect();
            let why = match reason {
                Reason::NotFound => {
                    "an alternative (a branch of an inverse function, a factor) was not solved"
                        .into()
                }
                other => other.text_in(arena),
            };
            Err(Refusal::Failed(format!(
                "found the roots {} but not necessarily all: {why}",
                shown.join(", ")
            )))
        }
        complete => Ok(complete.claimed_points()),
    }
}

/// The principal roots of `expr = 0` in `var` taken real (for the solvers
/// on the real line, e.g. the equations of `reduce_inequalities`).
pub(crate) fn solve_classified_real(arena: &mut Arena, expr: ExprId, var: ExprId) -> Solutions {
    solutions(arena, expr, var, &SolveCx::principal(Domain::Real))
}

/// What `expr = 0` has for roots in `var`, in the context `cx` (the domain
/// of `var`, principal roots or families; see [`SolveCx`]): all of them,
/// some, or nothing claimed ([`Solutions`]).
pub(crate) fn solutions(arena: &mut Arena, expr: ExprId, var: ExprId, cx: &SolveCx) -> Solutions {
    solve_impl(arena, expr, var, cx)
}

/// The principal roots of `expr = 0` in `var` that `Ex::solve` returns.
///
/// `var` takes its declared domain, ℂ for a plain symbol: the range
/// restrictions of the inversions (`exp f = c ≤ 0`, `sin f = c` and `cos f
/// = c` with `|c| > 1`, `cosh f = c < 1`) prove "no solution" only for an
/// `f` real over that domain, and otherwise the principal complex roots
/// are returned — `eˣ + 1 = 0` gives `[iπ]`, `cosh x = 0` gives `[±iπ/2]`,
/// `sin x = 2` gives `[asin 2, π − asin 2]` (SymPy's `solve` likewise);
/// up to 0.47 these were "no real solutions" for a plain `x`, while
/// `Ex::solve_as_set` gave the complex families.  For a variable declared
/// real nothing changes.
///
/// The other symbols are taken as declared ([`Params::AsDeclared`]): a
/// plain parameter may be complex, so for a real `x` the restriction
/// does not apply to `e^(a·x) = −1` (root `iπ/a`, real for `a = iπ`), and
/// `|a·x| = 1` is not `±1/a`.  Up to 0.49 plain parameters were taken real
/// here ("no solution", `[1/a, −1/a]`).
pub(crate) fn solve_classified(arena: &mut Arena, expr: ExprId, var: ExprId) -> Solutions {
    let cx = SolveCx::principal(Domain::Complex).with_params(Params::AsDeclared);
    solutions(arena, expr, var, &cx)
}

/// Solve `expr = 0` for `var`, returning **general** solution families for
/// periodic functions.
///
/// `param` must be a fresh symbol (ideally carrying the *integer*
/// assumption); it is used as the free integer parameter `n` in
/// `asin(c) + 2πn`, `±acos(c) + 2πn`, `atan(c) + πn`, etc.  Equations
/// without periodic structure return the same results as
/// [`solve_classified_real`].  `var` is taken real: the real zeros of a
/// periodic function (sign charts, extrema); `Ex::solve_general` is
/// [`solve_general_declared`] (up to 0.49 it was this one, and `eˣ + 1 =
/// 0` had "no solution" for a plain `x`).
pub(crate) fn solve_general(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    param: ExprId,
) -> Solutions {
    solve_impl(arena, expr, var, &SolveCx::general(Domain::Real, param))
}

/// Every solution of `expr = 0` in `var` over its declared domain, the
/// other symbols as declared ([`Params::AsDeclared`], see
/// [`solve_classified`]), periodic families in the integer parameter
/// `param`: `Ex::solve_general`.  For a `var` declared real this is the
/// real solve of [`solve_general`] with declared parameters (its
/// substituted unknowns, `t = eˣ`, taken real: `e^(x²) = −1` has no
/// solution), otherwise [`solve_general_over_c`] (`eˣ = −1` gives
/// `2nπi + iπ`; `|x| = 1`, a circle, is refused).  Up to 0.49
/// `Ex::solve_general` took a plain `x` and plain parameters real.
pub(crate) fn solve_general_declared(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    param: ExprId,
) -> Solutions {
    let unknown = Domain::declared(arena, var);
    let cx = SolveCx::general(unknown, param).with_params(Params::AsDeclared);
    solve_impl(arena, expr, var, &cx)
}

/// [`solve_general`] with `var` ranging over its declared domain, ℂ for a
/// plain symbol, and the other symbols as declared ([`Params::AsDeclared`],
/// see [`solve_classified`]): the solution *set* of `Ex::solve_as_set`.
///
/// [`solve_classified_real`] and [`solve_general`] take the variable of a transcendental
/// inversion to be real: `sin(f) = 2`, `exp(f) = −1`, `cosh(f) = 1/2` have
/// no solution and `exp(f) = c` the one branch `f = ln c` when `f` is real
/// for a real `var`, and `x·eˣ = c` the real branches of Lambert W
/// ([`solve_classified`] gives the principal complex roots of the first
/// three for a plain `var`).  Here
/// those hold only for a variable declared real: `exp(x) = 1` gives the
/// family `2nπi`, `sin(x) = 2` the families `asin 2 + 2nπ`, `π − asin 2 +
/// 2nπ`, `|x| = 1` (a circle) and `x·eˣ = 1` (the branches `W_k(1)`) are
/// not solved.
pub(crate) fn solve_general_over_c(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    param: ExprId,
) -> Solutions {
    let cx = SolveCx::general(Domain::Complex, param).with_params(Params::AsDeclared);
    solve_impl(arena, expr, var, &cx)
}

/// Shared implementation of [`solve_classified`] and [`solve_general`].
///
/// A general solve that met a periodic function of a periodic family
/// ([`Reason::NestedPeriods`], from any of its strategies) is refused: up to
/// 0.46 both layers took the same parameter, a strict subset of the
/// solutions — for a real `x`, `sin(cos x) = 1/2` gave `±acos(2nπ + π/6) +
/// 2nπ` cut to the real line, i.e. `±acos(π/6)` without `±acos(π/6) + 2mπ`
/// (now listed: only `n = 0` puts `π/6 + 2nπ` in the range of `cos`).  The
/// refusal is in words from here on: an outer solve that called this one
/// may still succeed by another strategy.  (Up to 0.48 the thread-local
/// flag `PERIOD_CONFLICT`, reset and restored around each general solve.)
/// The roots of a general solve come back as [`RootSet::Families`].
/// Parameters pinned to a value by their assumptions are that value
/// ([`with_pinned_parameters`]).
fn solve_impl(arena: &mut Arena, expr: ExprId, var: ExprId, cx: &SolveCx) -> Solutions {
    let expr = with_pinned_parameters(arena, expr, var);
    let outcome = solve_impl_checked(arena, expr, var, cx);
    if outcome.is_nested_periods() {
        return Solutions::refused(Reason::said(Reason::NestedPeriods.text()));
    }
    match cx.period() {
        Some(n) => without_congruent_families(arena, outcome.with_families(n)),
        None => outcome,
    }
}

/// The outcome of a general solve without the families congruent to an
/// earlier one: `a + d·n` and `b + d·n` with `(a − b)/d` an integer are the
/// same set.  Before 0.50 `solve_general(sinh(x)² + 1)` listed `3πi/2 +
/// 2nπi` beside `−πi/2 + 2nπi` (the two branches of `sinh x = −i`).
fn without_congruent_families(arena: &mut Arena, outcome: Solutions) -> Solutions {
    let (roots, n, missing) = match outcome {
        Solutions::Complete(RootSet::Families { roots, n }) => (roots, n, None),
        Solutions::Partial(RootSet::Families { roots, n }, m) => (roots, n, Some(m)),
        other => return other,
    };
    // (root, its offset and step when it is linear in `n`)
    let mut kept: Vec<(ExprId, Option<(ExprId, ExprId)>)> = Vec::with_capacity(roots.len());
    for r in roots {
        let linear = match symbolic_poly_coeffs(arena, r, n) {
            Some(c) if c.len() == 2 && !arena.is_zero_structural(c[1]) => Some((c[0], c[1])),
            _ => None,
        };
        if let Some((a, d)) = linear
            && kept
                .iter()
                .any(|&(_, k)| k.is_some_and(|(b, e)| e == d && integer_quotient(arena, a, b, d)))
        {
            continue;
        }
        kept.push((r, linear));
    }
    let roots = kept.into_iter().map(|(r, _)| r).collect();
    let set = RootSet::Families { roots, n };
    match missing {
        None => Solutions::Complete(set),
        Some(m) => Solutions::Partial(set, m),
    }
}

/// `expr` with every parameter (a free symbol other than `var`) whose
/// assumptions pin it to one value
/// ([`Assumptions::pinned_value`](crate::base::assumptions::Assumptions::pinned_value):
/// `0` for `zero`, `2` for an even prime, `1` for a positive integer neither
/// prime nor composite) replaced by that value.  A factor or a coefficient
/// in such parameters is then decided like a number: up to 0.54
/// `solve(x·(n − 2), x)` for an even prime `n` and `solve(x·z, x)` for a `z`
/// declared zero gave `[0]` (every `x` is a root: the factor is 0, which the
/// product rule took as nonzero by the generic convention), `solve(n − 2,
/// x)` and `solve(z·eˣ, x)` "no solution", and `(n − 2)·x² + x − 1` lost
/// its root `1` to the quadratic formula over a leading coefficient 0.
/// The unknown keeps its name (its own pinned value is its domain's one
/// point, which the declared-domain check keeps).  `expr` itself when no
/// parameter is pinned or when it has a binder (whose bound variable a
/// substitution could capture).
fn with_pinned_parameters(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    if arena.free_symbol_mask(expr) == 0 || arena.facts(expr).has_binder() {
        return expr;
    }
    let mut out = expr;
    for s in crate::base::walk::free_symbols(arena, expr) {
        if s == var {
            continue;
        }
        if let ExprNode::Symbol(sid) = *arena.node(s)
            && let Some(v) = arena.symbol_assumptions(sid).pinned_value()
        {
            let value = arena.int(v);
            out = crate::transforms::subs::subs(arena, out, s, value);
        }
    }
    out
}

/// Is `(a − b)/d` an integer?
fn integer_quotient(arena: &mut Arena, a: ExprId, b: ExprId, d: ExprId) -> bool {
    let diff = arena.sub(a, b);
    let q = arena.div(diff, d);
    let q = crate::transforms::eval::eval(arena, q);
    arena.as_num(q).is_some_and(|q| q.is_integer())
}

/// [`solve_impl`] without the check of nested periodic families.
///
/// A candidate whose check settles nothing because its own value is not
/// certified, or is a real value written with `i`
/// ([`Residual::Undecided`]), is not claimed: the outcome is partial.  Up to 0.51
/// it was kept, and `i·x^(3/2) − 3·x^(−1/3) = 0` had the third root
/// `−3^(6/11)` (written with `sin`, `cos` of `5π/11`, a hidden-zero
/// imaginary part), where the residual is `3^(9/11)·e^(iπ/3)`: on the
/// branch cut of both powers its value was not found.
fn solve_impl_checked(arena: &mut Arena, expr: ExprId, var: ExprId, cx: &SolveCx) -> Solutions {
    let raw = solve_raw(arena, expr, var, cx);
    let raw = without_guarded_non_roots(arena, expr, var, raw, cx);
    match raw.into_found() {
        Ok((candidates, missing)) => {
            let (s, undecided) = match drop_certain_non_roots(arena, expr, var, candidates, cx) {
                Ok(checked) => checked,
                Err(reason) => return Solutions::refused(Reason::said(reason)),
            };
            let undecided = first_unaccounted(arena, var, &s, undecided).map(|c| {
                Missing::Alternative(Reason::said(format!(
                    "cannot decide whether {} solves the equation (its residual does not evaluate)",
                    arena.display(c)
                )))
            });
            let missing = Missing::merged(missing, undecided);
            if s.is_empty() {
                // A refused alternative may hold roots (see `Solutions::union`).
                if missing.is_some() {
                    return Solutions::with_missing(s, missing);
                }
                // The principal candidates all fail; a member of a periodic
                // family through them may not (`tan(x)/x = 0`: `0` is a
                // pole, `π` a root).  "No solution" only when none can.
                if cx.period().is_none()
                    && let Some(outcome) = other_branches_outcome(arena, expr, var, cx)
                {
                    return outcome;
                }
                return Solutions::none(Reason::said(
                    "every candidate solution fails the equation (a principal branch does not reach the right-hand side)",
                ));
            }
            finalize_solutions(arena, s, missing)
        }
        // `1/tan(x) = 0` has no solution: `1/tan x` is undefined at the poles
        // of `tan` (SymPy 1.14: `solveset(1/tan(x))` and `solve` are empty).
        // From 0.40 to 0.51 the poles were added as roots, `g/(tan u − c)`
        // at `cos u = 0` where `g` is finite.
        Err(other) => other,
    }
}

/// Has `expr` a power `b^e` with `var` in `b` and an exponent `e` that is
/// not a number (undefined where `b = 0` when `re e ≤ 0`)?
fn has_symbolic_power_of(arena: &Arena, expr: ExprId, var: ExprId) -> bool {
    crate::base::walk::post_order_ids(arena, expr)
        .into_iter()
        .any(|id| {
            matches!(*arena.node(id), ExprNode::Pow(b, e)
                if arena.as_num(e).is_none() && expr_contains_var(arena, b, var))
        })
}

/// Is every power `b^e` of `var` with an exponent that is not a number (see
/// [`has_symbolic_power_of`]) defined at the roots of the factors of a
/// product, for every parameter value under which the root is claimed?  At
/// an unconditional root its base must be proved nonzero; at a root under
/// the condition `cond`, a base that vanishes needs `re e > 0` (`e > 0` for
/// a real `e`, as [`power_equals_zero`] writes it), which `cond` must be or
/// contain.  `(4 − x)·x^(y + i) = 0` has the root `4` (`4^(y + i)` is
/// defined for every `y`) and `0` when `re y > 0`; up to 0.53 such a product
/// was solved as a whole and refused.  `x·∛(x^y)` at `0` (undefined for
/// `re y ≤ 0`) is not.
fn symbolic_powers_defined_at(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    roots: &[ExprId],
    groups: &[Guarded],
) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    use crate::base::truth::Truth;
    let powers: Vec<ExprId> = crate::base::walk::post_order_ids(arena, expr)
        .into_iter()
        .filter(|&id| {
            matches!(*arena.node(id), ExprNode::Pow(b, e)
                if arena.as_num(e).is_none() && expr_contains_var(arena, b, var))
        })
        .collect();
    let base_nonzero_at = |arena: &mut Arena, b: ExprId, r: ExprId| -> bool {
        let v = crate::transforms::subs::subs(arena, b, var, r);
        let v = crate::transforms::eval::eval(arena, v);
        if crate::base::walk::free_symbols(arena, v).is_empty() {
            Truth::from(crate::transforms::eval::constant_zero_decision(arena, v)).is_false()
        } else {
            Truth::from(AssumptionCache::new().query(arena, v, Props::NONZERO)).is_true()
        }
    };
    for &r in roots {
        for &p in &powers {
            let ExprNode::Pow(b, _) = *arena.node(p) else {
                return false;
            };
            if !base_nonzero_at(arena, b, r) {
                return false;
            }
        }
    }
    for g in groups {
        let RootSet::Points(rs) = &g.roots else {
            return false;
        };
        let conjuncts: Vec<ExprId> = match arena.node(g.cond) {
            ExprNode::And(ch) => ch.to_vec(),
            _ => vec![g.cond],
        };
        for &r in rs {
            for &p in &powers {
                let ExprNode::Pow(b, e) = *arena.node(p) else {
                    return false;
                };
                if base_nonzero_at(arena, b, r) {
                    continue;
                }
                if expr_contains_var(arena, e, var) {
                    return false;
                }
                let zero = arena.zero;
                let real =
                    Truth::from(AssumptionCache::new().query(arena, e, Props::REAL)).is_true();
                let lhs = if real {
                    e
                } else {
                    let re = arena.re(e);
                    crate::transforms::eval::eval(arena, re)
                };
                let needed = arena.gt(lhs, zero);
                if !conjuncts.contains(&needed) {
                    return false;
                }
            }
        }
    }
    true
}

/// `outcome` without the conditional roots ([`Reason::Conditional`]) that
/// are no root for any value of the parameters: a part of the equation at
/// a pole there (`(√x − y)/(x − y²)` at `y²`), or a residual that
/// certainly is not 0.  Conditional roots with parameters are otherwise
/// not checked (their check is the condition).
fn without_guarded_non_roots(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    outcome: Solutions,
    cx: &SolveCx,
) -> Solutions {
    // (`None`: a root with an infinity inside a function, not known finite,
    // which no condition makes a claim: the outcome is refused)
    let check = |arena: &mut Arena, groups: Vec<Guarded>| -> Option<Vec<Guarded>> {
        let mut out = Vec::with_capacity(groups.len());
        for g in groups {
            let roots = match g.roots {
                RootSet::Points(rs) => {
                    let mut kept = Vec::with_capacity(rs.len());
                    for r in rs {
                        // (evaluated as `finalize_solutions` does the others:
                        // `−e^{ln y}/3` is `−y/3`)
                        let r = crate::transforms::eval::eval(arena, r);
                        if not_known_finite(arena, r) {
                            return None;
                        }
                        if !kept.contains(&r)
                            && !part_at_a_pole(arena, expr, var, r)
                            && !certainly_not_a_root(arena, expr, var, r)
                        {
                            kept.push(r);
                        }
                    }
                    if kept.is_empty() {
                        continue;
                    }
                    RootSet::Points(kept)
                }
                other => other,
            };
            out.push(Guarded {
                cond: g.cond,
                roots,
            });
        }
        Some(out)
    };
    let not_finite = || {
        Solutions::refused(Reason::said(
            "a conditional root holds an infinity inside a function and is not known to be finite",
        ))
    };
    match outcome {
        Solutions::Refused(Reason::Conditional(groups)) => {
            let Some(groups) = check(arena, groups) else {
                return not_finite();
            };
            // (a proof only without periodic functions, whose principal
            // candidates are not all the roots: `tan(x^y)/C(x)` at `x = √π`)
            if groups.is_empty() && has_periodic_atom(arena, expr, var, cx) {
                Solutions::refused(Reason::said(
                    "the principal candidates are poles, and other branches are not searched",
                ))
            } else if groups.is_empty() {
                Solutions::none(Reason::said(
                    "the only candidates are no root for any value of the parameters",
                ))
            } else {
                Solutions::Refused(Reason::Conditional(groups))
            }
        }
        Solutions::Partial(set, Missing::Alternative(Reason::Conditional(groups))) => {
            let Some(groups) = check(arena, groups) else {
                let unknown = Missing::Alternative(Reason::said(
                    "a conditional root holds an infinity inside a function and is not known to be \
                     finite",
                ));
                return Solutions::Partial(set, unknown);
            };
            if groups.is_empty() {
                Solutions::Complete(set)
            } else {
                Solutions::Partial(set, Missing::Alternative(Reason::Conditional(groups)))
            }
        }
        other => other,
    }
}

/// A refusal, naming a root, when the principal candidates of `expr = 0`
/// all failed but a member of a periodic family through them solves it.
fn other_branches_outcome(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Option<Solutions> {
    let member = root_off_principal_branch(arena, expr, var, cx)?;
    let shown = arena.display(member).to_string();
    Some(Solutions::refused(Reason::said(format!(
        "every principal-branch candidate fails the equation (a pole or a branch the \
         right-hand side is not on), but other branches solve it (e.g. {shown}); \
         see solve_general"
    ))))
}

/// A fresh symbol `_n`, `_n_1`, … declared integer, for the periodic
/// families the solver builds internally; one with other assumptions, or
/// occurring in `avoid`, is skipped.
fn branch_parameter(arena: &mut Arena, avoid: &[ExprId]) -> ExprId {
    use crate::base::assumptions::{Assumption, Assumptions};
    let mut declared = Assumptions::default().with(Assumption::Integer);
    declared.normalize_declared();
    let mut k = 0usize;
    loop {
        let name = if k == 0 {
            "_n".to_owned()
        } else {
            format!("_n_{k}")
        };
        k += 1;
        match arena.symbols.get(&name) {
            None => {
                let s = arena.symbol(&name);
                if let ExprNode::Symbol(sid) = *arena.node(s) {
                    arena.set_symbol_assumptions(sid, declared);
                }
                return s;
            }
            Some(sid) if arena.symbol_assumptions(sid) == declared => {
                let s = arena.intern(ExprNode::Symbol(sid));
                if !avoid
                    .iter()
                    .any(|&e| crate::base::walk::contains(arena, e, s))
                {
                    return s;
                }
            }
            Some(_) => {}
        }
    }
}

/// Unchecked candidates of the general solve of `expr = 0`: the points
/// free of the integer parameter and the families `s(param)` through it.
/// A member may fail the equation (a pole: `0` in `2nπ` for `sin(x)/x`);
/// callers check the members they use with [`certainly_not_a_root`].
pub(crate) struct PeriodicCandidates {
    /// Candidates without the parameter.
    pub(crate) plain: Vec<ExprId>,
    /// Families, each linear in `param`.
    pub(crate) families: Vec<ExprId>,
    /// The fresh integer parameter of the families.
    pub(crate) param: ExprId,
}

/// The candidates of the general solve of `expr = 0` (see
/// [`PeriodicCandidates`]), in the context [`SolveCx::families`] of the
/// solve `cx`.  `None` when it gives no candidates, or an identity /
/// contradiction, or (unless `partial_ok`: a search for one root) when
/// the candidates may not be all (a refused branch).
fn raw_families(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
    partial_ok: bool,
) -> Option<PeriodicCandidates> {
    let expr = with_pinned_parameters(arena, expr, var);
    let n = branch_parameter(arena, &[expr, var]);
    // As in `solve_impl`: the families of a general solve, refused when
    // they nest periodic families ([`Reason::NestedPeriods`]); partial
    // candidates only when `partial_ok`.
    let Ok((cands, missing)) = solve_raw(arena, expr, var, &cx.families(n)).into_found() else {
        return None;
    };
    if missing.is_some() && !partial_ok {
        return None;
    }
    let mut plain = Vec::new();
    let mut families = Vec::new();
    for c in cands {
        let v = crate::transforms::eval::eval(arena, c);
        if crate::base::walk::contains(arena, v, n) {
            families.push(v);
        } else {
            plain.push(v);
        }
    }
    Some(PeriodicCandidates {
        plain,
        families,
        param: n,
    })
}

/// The general-solve candidates of `expr = 0` when they include a periodic
/// family (`sin(x) = 0`: `2nπ`, `2nπ + π`), for callers that need every
/// real solution in a bounded window rather than the principal ones.
pub(crate) fn periodic_candidates(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Option<PeriodicCandidates> {
    raw_families(arena, expr, var, &SolveCx::principal(Domain::Real), false)
        .filter(|c| !c.families.is_empty())
}

/// A root of `expr = 0` that is a member (`n = ±1, ±2, ±3`) of a periodic
/// family through the principal candidates, when the principal solve lost
/// all its candidates.
fn root_off_principal_branch(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Option<ExprId> {
    let PeriodicCandidates {
        families, param: n, ..
    } = raw_families(arena, expr, var, cx, true)?;
    for fam in families {
        for k in [1i64, -1, 2, -2, 3, -3] {
            let kk = arena.int(k);
            let member = crate::transforms::subs::subs(arena, fam, n, kk);
            let member = crate::transforms::eval::eval(arena, member);
            if member_is_root(arena, expr, var, member) {
                return Some(member);
            }
        }
    }
    None
}

/// The first candidate of `undecided` (whose check settled nothing) that
/// may be a root besides `kept`: one with the value of a kept root (another
/// form of it, which [`finalize_solutions`] would drop) or outside the
/// declared domain of `var` (which the public solvers drop) is not
/// missing.  For a positive `x`, `x^(−1/3)·(231361/10000 + π²/x²)/6 =
/// π²/(2·x^(7/3))` has `100√2·π/481` and a form of it with `i` whose
/// residual does not evaluate.
fn first_unaccounted(
    arena: &mut Arena,
    var: ExprId,
    kept: &[ExprId],
    undecided: Vec<ExprId>,
) -> Option<ExprId> {
    use crate::base::truth::Truth;
    'cand: for c in undecided {
        if Truth::from(assumptions_hold(arena, var, c)).is_false() {
            continue;
        }
        if let Some(z) = constant_value_f64(arena, c) {
            for &k in kept {
                if constant_value_f64(arena, k).is_some_and(|w| near_values(w, z))
                    && same_constant(arena, k, c)
                {
                    continue 'cand;
                }
            }
        }
        return Some(c);
    }
    None
}

/// Is `candidate` (free of symbols) certainly a root: the equation at it
/// evaluates to zero (structurally, or zero to the precision of the zero
/// search), and is not undefined ([`residual_at`]).
fn member_is_root(arena: &mut Arena, expr: ExprId, var: ExprId, candidate: ExprId) -> bool {
    crate::base::walk::free_symbols(arena, candidate).is_empty()
        && residual_at(arena, expr, var, candidate) == Residual::Zero
}

/// What substituting a candidate into the equation shows ([`residual_at`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Residual {
    /// Zero: structurally, certified, or to the precision of the zero
    /// search.
    Zero,
    /// Certainly not zero: a certified nonzero digit, or undefined there
    /// (`zoo`, `nan`, or a part of the equation at a pole).
    Nonzero,
    /// The evaluation settles nothing, and the candidate is a constant
    /// whose own value is not certified, or a real value written with `i`
    /// (`−3^(6/11)` as `−3^(6/11)·(−i)^(6/11)·(cos(5π/11) + …)`): on the
    /// real axis, where principal powers and logarithms have their cuts, a
    /// value known as a complex ball cannot say on which side it is.
    Undecided,
    /// Not checked: the candidate is not fully evaluated, or the
    /// substituted equation keeps symbols, or its evaluation settles
    /// nothing at a certified point (`acos(atan(tan 1))`: `acos` at its
    /// branch point 1, where it is continuous and 0).
    Unchecked,
}

/// `expr` at `var = candidate` (see [`Residual`]).
fn residual_at(arena: &mut Arena, expr: ExprId, var: ExprId, candidate: ExprId) -> Residual {
    use crate::base::walk;
    if walk::has_unevaluated(arena, candidate) {
        return Residual::Unchecked;
    }
    let at = crate::transforms::subs::subs(arena, expr, var, candidate);
    let at = crate::transforms::eval::eval(arena, at);
    let unless_a_pole = |arena: &mut Arena, r: Residual| {
        if part_at_a_pole(arena, expr, var, candidate) {
            Residual::Nonzero
        } else {
            r
        }
    };
    if arena.is_zero_structural(at) {
        return unless_a_pole(arena, Residual::Zero);
    }
    if !walk::free_symbols(arena, at).is_empty() || walk::has_unevaluated(arena, at) {
        return Residual::Unchecked;
    }
    if is_undefined_value(arena, at) {
        return Residual::Nonzero;
    }
    use crate::transforms::evalf::{Settled, ZeroSearch};
    match crate::transforms::evalf::evalf_settled(arena, at, ROOT_CHECK_DIGITS, ZeroSearch::Cap) {
        Ok((z, Settled::Certified)) if !(z.0.is_zero() && z.1.is_zero()) => Residual::Nonzero,
        Ok(_) => unless_a_pole(arena, Residual::Zero),
        Err(_) => {
            // An exact cancellation needs no value (`s·(x + e) + i·x` at `x =
            // −s·e/(s + i)`, with an `s` that does not evaluate)
            let together = polybridge::together(arena, at);
            let (num, _) = polybridge::as_numer_denom(arena, together);
            let num = crate::transforms::expand::expand(arena, num);
            let num = crate::transforms::eval::eval(arena, num);
            if arena.is_zero_structural(num) {
                return unless_a_pole(arena, Residual::Zero);
            }
            let uncertain_point = arena.is_symbol_free(candidate) && {
                let s = part_signs(arena, candidate);
                !s.certified()
                    || (s.im.is_zero().is_true()
                        && crate::base::walk::contains(arena, candidate, arena.i_unit))
            };
            let r = if uncertain_point {
                Residual::Undecided
            } else {
                Residual::Unchecked
            };
            unless_a_pole(arena, r)
        }
    }
}

/// May a denominator of `f` vanish on a set of positive measure: a power
/// with a negative or symbolic exponent whose base holds `var` and a
/// function that is 0 on whole intervals (`δ`, `H`, `⌊·⌋`, `⌈·⌉`, `min`,
/// `max`, the Kronecker `δ`, a piecewise function; `re`, `im`, `arg`, the
/// conjugate on parts of the line)?  An analytic base not identically 0, and
/// its modulus or sign, vanishes only at isolated points (the generic
/// convention's poles).
fn denominator_vanishes_on_intervals(arena: &Arena, f: ExprId, var: ExprId) -> bool {
    let piecewise = |arena: &Arena, b: ExprId| {
        crate::base::walk::post_order_ids(arena, b)
            .into_iter()
            .any(|id| {
                matches!(
                    arena.node(id),
                    ExprNode::DiracDelta(_)
                        | ExprNode::Heaviside(_)
                        | ExprNode::Floor(_)
                        | ExprNode::Ceiling(_)
                        | ExprNode::Min(_)
                        | ExprNode::Max(_)
                        | ExprNode::Re(_)
                        | ExprNode::Im(_)
                        | ExprNode::Arg(_)
                        | ExprNode::Conjugate(_)
                        | ExprNode::KroneckerDelta(..)
                        | ExprNode::Piecewise(..)
                )
            })
    };
    crate::base::walk::post_order_ids(arena, f)
        .into_iter()
        .any(|id| match *arena.node(id) {
            ExprNode::Pow(b, e) => {
                !arena.as_num(e).is_some_and(|q| !q.is_negative())
                    && expr_contains_var(arena, b, var)
                    && piecewise(arena, b)
            }
            _ => false,
        })
}

/// Does `v` hold `∞`, `−∞`, `zoo` or `nan` outside the formal integrals,
/// sums, products and limits (whose bounds they may be)?  Unlike
/// [`has_infinite_term`] it looks inside function arguments:
/// `chebyshev_t(2, y·∞)`, `Γ(y·∞)`, `sin ∞`.
fn holds_infinity(arena: &Arena, v: ExprId) -> bool {
    let mut stack = vec![v];
    while let Some(e) = stack.pop() {
        match arena.node(e) {
            ExprNode::Infinity
            | ExprNode::NegInfinity
            | ExprNode::ComplexInfinity
            | ExprNode::NaN => return true,
            ExprNode::Integral(..)
            | ExprNode::DefiniteIntegral(..)
            | ExprNode::Limit(..)
            | ExprNode::Sum(..)
            | ExprNode::Product_(..)
            | ExprNode::Series(..) => {}
            _ => stack.extend(arena.children(e)),
        }
    }
    false
}

/// Is `v` (evaluated) a value that holds an infinity inside a function
/// argument ([`holds_infinity`]) and is not known to be finite?  Such a
/// value is no number the solver can claim: `−chebyshev_t(2, y·∞)` is `−∞`
/// for every real `y ≠ 0` and undefined at `y = 0`, `−sin ∞` has no value.
/// The value `eval` folds (`atan ∞ = π/2`, `e^{−y·∞} = 0` for a positive
/// `y`) has no infinity left.
fn not_known_finite(arena: &mut Arena, v: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    holds_infinity(arena, v)
        && !AssumptionCache::new()
            .query_public(arena, v, Props::FINITE)
            .is_true()
}

/// Does the value `at` hold `∞`, `−∞`, `zoo` or `nan`?
fn is_undefined_value(arena: &Arena, at: ExprId) -> bool {
    crate::base::walk::post_order_ids(arena, at)
        .into_iter()
        .any(|id| {
            matches!(
                arena.node(id),
                ExprNode::Infinity
                    | ExprNode::NegInfinity
                    | ExprNode::ComplexInfinity
                    | ExprNode::NaN
            )
        })
}

/// Is a part of `expr` at a pole at `var = candidate`, where `expr` is
/// undefined although the substituted whole may evaluate to a number: a
/// `tan u` at a zero of `cos u`, a `tanh u` at a zero of `cosh u`, or a
/// negative power of a part that vanishes there (decided exactly).
/// `cos(3x)/(tan(−x) − 2)` at `π/2` evaluates to `0·(1/zoo) = 0`, and
/// `1/tan x` to `1/zoo = 0`: from 0.40 to 0.51 such poles were roots (SymPy
/// 1.14's `solveset` excludes them from `1/tan(x)` and `1/(tan(−x) − 2)`,
/// on bounded intervals also from `cos(3x)/(tan(−x) − 2)`;
/// `reduce_inequalities` excluded them since 0.50).  The natural domain of
/// an expression is where every part of it is defined.  `acot z` and
/// `acoth z` are written `atan(1/z)`, `atanh(1/z)` and are defined at `z =
/// 0` (`π/2`, `iπ/2`; SymPy's `acot(0)`, `acoth(0)`): a reciprocal that is
/// only the argument of `atan` or `atanh` is not a pole.
fn part_at_a_pole(arena: &mut Arena, expr: ExprId, var: ExprId, candidate: ExprId) -> bool {
    let ids = crate::base::walk::post_order_ids(arena, expr);
    let mut outside_atan: rustc_hash::FxHashSet<ExprId> = rustc_hash::FxHashSet::default();
    outside_atan.insert(expr);
    for &id in &ids {
        if !matches!(arena.node(id), ExprNode::Atan(_) | ExprNode::Atanh(_)) {
            outside_atan.extend(arena.children(id));
        }
    }
    let mut vanishing: Vec<ExprId> = Vec::new();
    for id in ids {
        let base = match *arena.node(id) {
            ExprNode::Tan(u) if expr_contains_var(arena, u, var) => arena.cos(u),
            ExprNode::Tanh(u) if expr_contains_var(arena, u, var) => arena.cosh(u),
            ExprNode::Pow(b, e)
                if arena.as_num(e).is_some_and(|r| r.is_negative())
                    && expr_contains_var(arena, b, var)
                    && outside_atan.contains(&id) =>
            {
                b
            }
            _ => continue,
        };
        if !vanishing.contains(&base) {
            vanishing.push(base);
        }
    }
    vanishing.into_iter().any(|base| {
        let v = crate::transforms::subs::subs(arena, base, var, candidate);
        let v = crate::transforms::eval::eval(arena, v);
        arena.is_zero_structural(v) || proved_zero_constant(arena, v)
    })
}

/// Remove the candidates that certainly do not solve `expr = 0`.
///
/// Inversion peeling takes principal branches (`asin(f) = c → f = sin c`,
/// `ln f = c → f = e^c`, `f^(p/q) = c → f = u^q`), and a branch reproduces
/// `c` only when `c` lies in the range of the inverted function: `asin x = π`
/// gave `x = 0`, `√x = −2` gave `4`, `atan x = 2` gave `tan 2`,
/// `ln(atan x) = 1` gave `tan e`.  Rather than a range rule per function
/// (and per composition), every candidate of a non-polynomial equation is
/// substituted back, as SymPy's `solve` does with `checksol`: one at which
/// the equation evaluates to a certified non-zero number, or is undefined
/// (`|x|/x = 0` at `x = 0`, `1/tan x = 0` at `x = π/2`: [`part_at_a_pole`]),
/// is dropped.  Candidates with free parameters are kept, and so are
/// constants at a certified point whose residual `evalf` cannot decide (a
/// branch point: `acos` at 1); constants whose own value is not certified
/// and whose residual does not evaluate are returned apart (the second
/// value; up to 0.51 they were kept).  Polynomial equations over ℚ are
/// solved exactly and are not re-checked.
///
/// A certified digit decides whatever the magnitude of the residual: the
/// candidates are exact.  Before 0.31 a residual counted only above
/// `10⁻¹⁰`, and `√x = −10⁻¹²` came back with `x = 10⁻²⁴` (where
/// `√x = −10⁻¹²` came back with `x = 10⁻²⁴` (where
/// `√x + 10⁻¹² = 2·10⁻¹²`), `asin x = π/2 + 10⁻¹¹` with
/// `sin(π/2 + 10⁻¹¹)`.
///
/// A periodic family is checked by [`check_family`]; `Err` carries the
/// reason when a family mixes solutions and non-solutions that no family
/// over all integers can separate.
fn drop_certain_non_roots(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    solutions: Vec<ExprId>,
    cx: &SolveCx,
) -> Result<(Vec<ExprId>, Vec<ExprId>), String> {
    if solutions.is_empty() || polybridge::expr_to_poly(arena, expr, var).is_some() {
        return Ok((solutions, Vec::new()));
    }
    let mut out = Vec::with_capacity(solutions.len());
    let mut undecided = Vec::new();
    for s in solutions {
        match cx.period() {
            Some(n) if crate::base::walk::contains(arena, s, n) => {
                out.extend(check_family(arena, expr, var, s, n)?);
            }
            _ => match residual_at(arena, expr, var, s) {
                Residual::Nonzero => {}
                Residual::Undecided => undecided.push(s),
                Residual::Zero | Residual::Unchecked => out.push(s),
            },
        }
    }
    Ok((out, undecided))
}

/// Classes of members of `candidate(n)` (`n` an integer) that solve
/// `expr = 0`, as families: the family itself, sub-families, or none.
///
/// * When `expr` depends on `var` only through `sin`, `cos`, `tan` (and
///   `exp(i·…)`) of rational multiples of `var` it is periodic, with a
///   period `T` ([`structural_period`]); for a family `s₀ + n·d` with
///   `T = m·d` the members `s₀ + j·d`, `j = 0..m`, decide their classes
///   `s₀ + j·d + n·T` exactly.  `sin 3x/sin x = 0`: the family `2nπ/3`
///   has the poles `n ≡ 0 (mod 3)`, and gives `2π/3 + 2nπ`, `4π/3 + 2nπ`.
/// * Otherwise the members `n = 0, ±1, ±2, ±3`, and the members at the
///   singular points of the non-periodic part (`x = 10π` for
///   `sin(x)/(x − 10π)`), are checked: a family all of whose sampled
///   members certainly fail is dropped (`exp(√x) = −1/2` gives
///   `√x = ln(1/2) + (2n + 1)πi`, with a negative real part that no
///   principal square root has), one with no failing member is kept, and
///   one with both is an error: `sin(x)/x = 0` has the family `2nπ`
///   without `0`, which SymPy writes `ℤ·2π \ {0}` (before 0.39 the family
///   was returned whole, `0` included).
fn check_family(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    candidate: ExprId,
    n: ExprId,
) -> Result<Vec<ExprId>, String> {
    let member = |arena: &mut Arena, k: i64| -> ExprId {
        let kk = arena.int(k);
        let m = crate::transforms::subs::subs(arena, candidate, n, kk);
        crate::transforms::eval::eval(arena, m)
    };
    // The step `d` of a family linear in `n`, as a rational multiple of π.
    let step_over_pi = {
        let s0 = member(arena, 0);
        let s1 = member(arena, 1);
        let s2 = member(arena, 2);
        let d = arena.sub(s1, s0);
        let d = crate::transforms::eval::eval(arena, d);
        let d2 = arena.sub(s2, s1);
        let d2 = crate::transforms::eval::eval(arena, d2);
        let ratio = arena.div(d, arena.pi);
        let ratio = crate::transforms::eval::eval(arena, ratio);
        if d == d2 {
            arena.as_num(ratio).cloned()
        } else {
            None
        }
    };
    if let (Some(t), Some(r)) = (structural_period(arena, expr, var), step_over_pi)
        && !r.is_zero()
    {
        let m = t / r.abs();
        if m.is_integer() && m.is_positive() && m <= Ratio::from_integer(BigInt::from(12)) {
            let m = m.to_integer();
            let m_i64: i64 = num_traits::ToPrimitive::to_i64(&m).unwrap_or(1);
            let mut keep = Vec::new();
            for j in 0..m_i64 {
                let mj = member(arena, j);
                if certainly_not_a_root(arena, expr, var, mj) {
                    continue;
                }
                if m_i64 == 1 {
                    keep.push(candidate);
                    continue;
                }
                // s(m·n + j)
                let m_id = arena.big_int(m.clone());
                let mn = arena.mul(&[m_id, n]);
                let j_id = arena.int(j);
                let shifted = arena.add(&[mn, j_id]);
                let sub = crate::transforms::subs::subs(arena, candidate, n, shifted);
                let sub = crate::transforms::expand::expand(arena, sub);
                keep.push(crate::transforms::eval::eval(arena, sub));
            }
            return Ok(keep);
        }
    }
    let mut failing: Option<ExprId> = None;
    let mut passing = false;
    for k in [0i64, 1, -1, 2, -2, 3, -3] {
        let mk = member(arena, k);
        if certainly_not_a_root(arena, expr, var, mk) {
            failing.get_or_insert(mk);
        } else {
            passing = true;
        }
    }
    if !passing {
        return Ok(Vec::new());
    }
    if failing.is_none() {
        failing = singular_member(arena, expr, var, candidate, n)?;
    }
    match failing {
        None => Ok(vec![candidate]),
        Some(bad) => Err(format!(
            "the solution family {} contains {}, which is not a solution (a pole, or a branch the \
             equation is not on); a family over all integers {} cannot exclude it",
            arena.display(candidate),
            arena.display(bad),
            arena.display(n),
        )),
    }
}

/// `T/π` for a period `T` of `expr` in `var`, when `var` occurs only in
/// `sin`, `cos`, `tan` of `a·var + b` and in `exp(i·a·var + b)` with
/// rational `a` (the least common multiple of their periods `2π/|a|`,
/// `π/|a|`).  `None` when `var` occurs anywhere else.
fn structural_period(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<Q> {
    let mut period: Option<Q> = None;
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        if !expr_contains_var(arena, id, var) {
            continue;
        }
        if id == var {
            return None;
        }
        let node = arena.node(id).clone();
        let atom = match node {
            ExprNode::Sin(arg) | ExprNode::Cos(arg) => Some((arg, 2i64)),
            ExprNode::Tan(arg) => Some((arg, 1i64)),
            ExprNode::Exp(arg) => Some((arg, -2i64)),
            _ => None,
        };
        match atom {
            Some((arg, half_turns)) => {
                let (a, _) = linear_rate(arena, arg, var)?;
                // exp(i·a·x + b) has the period 2π/|a|; exp(a·x + b), a real, none
                let a = if half_turns < 0 {
                    let i = arena.i_unit;
                    let r = arena.div(a, i);
                    crate::transforms::eval::eval(arena, r)
                } else {
                    a
                };
                let half_turns = half_turns.abs();
                let a = arena.as_num(a)?.clone();
                if a.is_zero() {
                    return None;
                }
                let p = Ratio::from_integer(BigInt::from(half_turns)) / a.abs();
                period = Some(match period {
                    None => p,
                    Some(q) => rational_lcm(&q, &p),
                });
            }
            None => stack.extend(arena.children(id)),
        }
    }
    period
}

/// The least common multiple of two positive rationals:
/// `lcm(a/b, c/d) = lcm(a, c)/gcd(b, d)` in lowest terms.
fn rational_lcm(x: &Q, y: &Q) -> Q {
    use num_integer::Integer;
    let num = x.numer().lcm(y.numer());
    let den = x.denom().gcd(y.denom());
    Ratio::new(num, den)
}

/// A member of the family `candidate(n)` at a singular point of `expr`
/// outside its periodic part that certainly fails the equation: the zeros
/// of polynomial denominators and logarithm arguments (`x = 10π` for
/// `sin(x)/(x − 10π)`, the member `n = 5` of `2nπ`).  The index `(p −
/// s₀)/d` of a constant singular point `p` is decided exactly (a rational,
/// or a certified non-real value, or certified floor and ceiling); an error
/// when it is not decided whether `p` is a member.  A point with parameters
/// is a member only for parameter values of measure zero and is skipped.
///
/// Up to 0.53 the index was the `f64` of a 20-digit value, and indices
/// beyond `10¹²` were skipped: `solve_as_set(sin(x)/(x − 2·10¹³π))` was the
/// family `nπ` with its pole `2·10¹³π`.
fn singular_member(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    candidate: ExprId,
    n: ExprId,
) -> Result<Option<ExprId>, String> {
    use crate::base::truth::SignSet;
    let mut points: Vec<ExprId> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        let base = match *arena.node(id) {
            ExprNode::Pow(b, e) if arena.as_num(e).is_some_and(|r| r.is_negative()) => b,
            ExprNode::Ln(b) => b,
            _ => continue,
        };
        if !expr_contains_var(arena, base, var) {
            continue;
        }
        if let Some(p) = polybridge::expr_to_poly(arena, base, var) {
            if !p.is_constant() {
                points.extend(
                    solve_rational_poly(arena, var, &p)
                        .into_iter()
                        .map(|s| s.value),
                );
            }
            continue;
        }
        let Some(coeffs) = symbolic_poly_coeffs(arena, base, var) else {
            continue;
        };
        if coeffs.len() > 3 || coeffs.iter().any(|&c| expr_contains_var(arena, c, var)) {
            continue;
        }
        if let Some(sols) = solve_symbolic_coeffs(arena, &coeffs) {
            points.extend(sols.into_iter().map(|s| s.value));
        }
    }
    let k0 = {
        let z = arena.zero;
        let m = crate::transforms::subs::subs(arena, candidate, n, z);
        crate::transforms::eval::eval(arena, m)
    };
    let k1 = {
        let o = arena.one;
        let m = crate::transforms::subs::subs(arena, candidate, n, o);
        crate::transforms::eval::eval(arena, m)
    };
    let d = arena.sub(k1, k0);
    let d = crate::transforms::eval::eval(arena, d);
    let k2 = {
        let two = arena.int(2);
        let m = crate::transforms::subs::subs(arena, candidate, n, two);
        crate::transforms::eval::eval(arena, m)
    };
    let d2 = arena.sub(k2, k1);
    let d2 = crate::transforms::eval::eval(arena, d2);
    // (the index is `(p − s₀)/d` only for a family linear in `n`)
    if d != d2 || has_infinite_term(arena, d) || arena.is_zero_structural(d) {
        return Ok(None);
    }
    let undecided = |arena: &Arena, p: ExprId| {
        format!(
            "cannot decide whether the singular point {} is a member of the solution family {}",
            arena.display(p),
            arena.display(candidate)
        )
    };
    let integer = |arena: &Arena, id: ExprId| {
        arena
            .as_num(id)
            .filter(|q| q.is_integer())
            .map(|q| q.to_integer())
    };
    for p in points {
        // the index n = (p − s₀)/d of `p` in the family, exactly
        let ratio = arena.sub(p, k0);
        let ratio = arena.div(ratio, d);
        let ratio = crate::transforms::eval::eval(arena, ratio);
        if !crate::base::walk::free_symbols(arena, ratio).is_empty()
            || has_infinite_term(arena, ratio)
        {
            continue;
        }
        let k = if let Some(q) = arena.as_num(ratio) {
            if !q.is_integer() {
                continue;
            }
            q.to_integer()
        } else {
            match crate::transforms::realness::imaginary_part_sign(arena, ratio) {
                SignSet::ZERO => {}
                s if s.is_nonzero().is_true() => continue,
                _ => return Err(undecided(arena, p)),
            }
            let f = arena.floor(ratio);
            let f = crate::transforms::eval::eval(arena, f);
            let c = arena.ceiling(ratio);
            let c = crate::transforms::eval::eval(arena, c);
            match (integer(arena, f), integer(arena, c)) {
                (Some(f), Some(c)) if f == c => f,
                // strictly between two members
                (Some(_), Some(_)) => continue,
                _ => {
                    // within its error of an integer: the nearest one
                    let half = arena.rational(1, 2);
                    let shifted = arena.add(&[ratio, half]);
                    let r = arena.floor(shifted);
                    let r = crate::transforms::eval::eval(arena, r);
                    let Some(k) = integer(arena, r) else {
                        return Err(undecided(arena, p));
                    };
                    let kk = arena.big_int(k);
                    let m = crate::transforms::subs::subs(arena, candidate, n, kk);
                    let m = crate::transforms::eval::eval(arena, m);
                    if certainly_not_a_root(arena, expr, var, m) {
                        return Ok(Some(m));
                    }
                    return Err(undecided(arena, p));
                }
            }
        };
        let kk = arena.big_int(k);
        let m = crate::transforms::subs::subs(arena, candidate, n, kk);
        let m = crate::transforms::eval::eval(arena, m);
        if certainly_not_a_root(arena, expr, var, m) {
            return Ok(Some(m));
        }
    }
    Ok(None)
}

/// Does `expr` at `var = candidate` certainly not vanish?  `true` when the
/// substituted equation, free of symbols, is undefined (`zoo`, `NaN`, `±∞`,
/// or a part of it at a pole: [`part_at_a_pole`]) or evaluates to a number
/// with a certified nonzero digit; `false` when it is zero (to the
/// precision of the zero search) or cannot be decided ([`residual_at`]).
/// A candidate with parameters is decided when the substituted equation
/// is free of them: up to 0.50 it was kept unchecked, and `(x + y)^(1/6)
/// = −(1 + √2)/3` gave `x = −y + 70√2/729 + 11/81` (the sixth root of
/// `(1 + √2)⁶/729` is positive), and `(x − y)^(3/2) = −√2/3` the three
/// roots `y + k`, `k³ = 2/9`, with the real `k`, whose `k^(3/2)` is `+√2/3`.
pub(crate) fn certainly_not_a_root(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    candidate: ExprId,
) -> bool {
    residual_at(arena, expr, var, candidate) == Residual::Nonzero
}

/// Digits at which [`certainly_not_a_root`] evaluates a residual.
const ROOT_CHECK_DIGITS: u32 = 20;

/// Strategy dispatch without the final clean-up pass (see [`solve_impl`]).
///
/// The strategies are tried in turn; one that finds roots (all, or some
/// with a refused alternative) answers, and so does one that proves there
/// is none, except the range restrictions of the inversion peeling, which
/// let the substitutions and the Lambert forms try first (they may find
/// roots a range restriction of a variable taken real does not see).  A
/// [`Reason::NestedPeriods`] refusal from any strategy of a general solve
/// answers at once: the general solve refuses as a whole ([`solve_impl`]).
fn solve_raw(arena: &mut Arena, expr: ExprId, var: ExprId, cx: &SolveCx) -> Solutions {
    // Step 0: degenerate cases.
    if arena.is_zero_structural(expr) {
        return Solutions::Complete(RootSet::All);
    }
    if !expr_contains_var(arena, expr, var) {
        return classify_constant(arena, expr, var);
    }
    // `C(f, k)` for a nonnegative integer `k` is the polynomial
    // `f(f − 1)⋯(f − k + 1)/k!` (refused before 0.47; SymPy: `solve(binomial(x,
    // 2) - 3)` → `[-2, 3]`, by `expand_func`).
    if let Some(rewritten) = binomials_as_polynomials(arena, expr, var) {
        return solve_raw(arena, rewritten, var, cx);
    }

    // Pre-check: if expr is a Mul, solve each var-dependent factor
    // independently.  x*(x-1)*(x+2) = 0 → union of factor solutions.
    // The union is the answer only when every factor was solved or shown
    // to have no solution: an empty list from a factor means its roots are
    // unknown, not absent.  Before 0.36 such a factor was dropped and the
    // roots of the others came back alone — `solve(√x·(x³ + √2·x + 1))`
    // and `solve(x²·(x − cos x))` were `[0]`; now the product is treated
    // as a whole (and refused when that fails too, as SymPy raises
    // `NotImplementedError` for the second).
    //
    // A factor that is a polynomial over ℚ is solved with the factors it
    // shares with the polynomial denominators (`q(x)^(−k)` factors) divided
    // out, so that no pole comes back as a root: before 0.39
    // `(x⁴ − 2x³ − 3x² + 7x − 2)·(x³ − 3x + 1)^(−1)` gave the three roots
    // of the cubic (poles, `0/0`) besides `2`, the substitution check not
    // recognising the zero of the cubic at a nested radical.
    //
    // A rational function with parameters (`(x² − a²)·(x − a)⁻¹`) is left
    // to the rational path, which divides out common factors over
    // ℚ[x, parameters].  Otherwise a root of a factor is tested against the
    // other factors' denominators exactly (`(x − √3)·((x − √3)(x + 1))⁻¹`
    // has the pole `√3`, which the numerator factor solves).
    let parametric_rational = matches!(arena.node(expr), ExprNode::Mul(_))
        && is_rational_function_of(arena, expr, var)
        && crate::base::walk::free_symbols(arena, expr)
            .iter()
            .any(|&s| s != var)
        && rational_over_q_in_all_symbols(arena, expr);
    if let ExprNode::Mul(ref children) = arena.node(expr).clone()
        && !parametric_rational
    {
        let mut denominator = Poly::one();
        for &child in children {
            if let ExprNode::Pow(b, e) = *arena.node(child)
                && arena
                    .as_num(e)
                    .is_some_and(|r| r.is_integer() && r.is_negative())
                && let Some(q) = polybridge::expr_to_poly(arena, b, var)
                && !q.is_constant()
            {
                denominator = denominator.mul(&q);
            }
        }
        // Every factor with the variable has no solution (each one proved
        // so): neither has the product (`(e^(x/2) + 1)·(e^(x/2) + 2)`; refused
        // before 0.44).  A factor that vanishes identically makes every
        // value a root.
        // Constant factors: the canonicalizer already folds literal zeros,
        // and a factor with parameters is taken nonzero (the generic
        // convention; parameters pinned to a value are substituted by
        // `solve_impl`).  A symbol-free one, and one with `±∞`, `zoo` or
        // `nan` inside (`e^{−y·∞}` is 0 for every `y > 0`: no generic
        // value), must be decided: proved nonzero, the other factors' roots
        // are the roots; proved 0 (`euler_number(5)`), every value is a root
        // (refused up to 0.54); without a value (`δ(i)`) or with an
        // undecided zero test the product is undefined or undecided at every
        // point (up to 0.52 `δ(i)·x = 0` gave `[0]`), whatever the other
        // factors.  (The crate's root searches take candidates: there a
        // factor stays taken nonzero, as before.)
        if !cx.conditional_roots_as_candidates() {
            let mut vanishing = false;
            // (a factor with an infinity in it makes a vanishing one `0·∞`,
            // undefined: `(x + ∞)·euler_number(5)` has no root)
            // (and one with a denominator that vanishes on intervals is
            // undefined there: `x/δ(x)·stirling2(0, 8)` is undefined for every
            // `x`, not 0)
            let mut infinite_part = false;
            for &child in children {
                infinite_part |= holds_infinity(arena, child)
                    || (expr_contains_var(arena, child, var)
                        && denominator_vanishes_on_intervals(arena, child, var));
                if expr_contains_var(arena, child, var)
                    || arena.as_num(child).is_some()
                    || (!crate::base::walk::free_symbols(arena, child).is_empty()
                        && !holds_infinity(arena, child))
                {
                    continue;
                }
                match classify_constant(arena, child, var) {
                    Solutions::Complete(RootSet::All) => vanishing = true,
                    c if c.is_none_proved() => {}
                    _ => {
                        return Solutions::refused(Reason::said(format!(
                            "cannot decide whether the constant factor {} is nonzero (or has a value)",
                            arena.display(child)
                        )));
                    }
                }
            }
            if vanishing && infinite_part {
                return Solutions::refused(Reason::said(
                    "a constant factor is 0 and another factor holds an infinity or a denominator \
                     that may vanish on intervals (0·∞ is undefined)",
                ));
            }
            if vanishing {
                return Solutions::Complete(RootSet::All);
            }
        }
        let mut product = Solutions::none(Reason::said("no factor of the product has a solution"));
        for &child in children {
            if !expr_contains_var(arena, child, var) {
                continue;
            }
            let outcome = match polybridge::expr_to_poly(arena, child, var) {
                Some(p) if !denominator.is_constant() && !p.is_constant() => {
                    let p = without_common_factors(p, &denominator);
                    if p.is_constant() {
                        Solutions::none(Reason::said("every root of the factor is a pole"))
                    } else {
                        Solutions::found(values(solve_rational_poly(arena, var, &p)))
                    }
                }
                _ => solve_raw(arena, child, var, cx),
            };
            product = product.union(outcome);
        }
        match product.into_found() {
            Ok((solutions, None)) => {
                // (Every root a pole: the whole product decides, below — a
                // principal root of a periodic factor is not all its roots.)
                match drop_poles_of_product(arena, expr, var, solutions) {
                    Ok(s) if !s.is_empty() => return Solutions::found(s),
                    Ok(_) => {
                        // Every candidate is a pole, as `solve_impl` decides
                        // when its check drops them all.
                        if cx.period().is_none()
                            && let Some(outcome) = other_branches_outcome(arena, expr, var, cx)
                        {
                            return outcome;
                        }
                        return Solutions::none(Reason::said(
                            "every root of a factor is a pole of another factor",
                        ));
                    }
                    Err(reason) => return Solutions::refused(Reason::said(reason)),
                }
            }
            // Some factor with roots under a condition on the parameters
            // only: the roots of the others and those, exactly — unless a
            // factor is a power with an exponent that is not a number, which
            // may be undefined at another factor's root for some parameter
            // values (`x·∛(x^y)` at `0` for `re y ≤ 0`, where `drop_poles_of_product`
            // sees no pole): then the product as a whole, below.
            Ok((solutions, Some(Missing::Alternative(Reason::Conditional(groups)))))
                if !has_symbolic_power_of(arena, expr, var)
                    || symbolic_powers_defined_at(arena, expr, var, &solutions, &groups) =>
            {
                if let Ok(s) = drop_poles_of_product(arena, expr, var, solutions) {
                    let m = Missing::Alternative(Reason::Conditional(groups));
                    return Solutions::with_missing(s, Some(m));
                }
            }
            // Some factor unsolved: the product as a whole, below.
            Ok((_, Some(_))) => {}
            Err(decided @ Solutions::Complete(_)) => return decided,
            Err(nested) if nested.is_nested_periods() => return nested,
            Err(conditional) if conditional.is_conditional() => return conditional,
            Err(_) => {}
        }
        // Otherwise fall through and treat the product as a whole.
    }

    // Step 1: Convert to polynomial with rational coefficients.
    if let Some(poly) = polybridge::expr_to_poly(arena, expr, var) {
        if poly.is_zero() {
            return Solutions::Complete(RootSet::All);
        }
        if poly.is_constant() {
            let c = arena.display(expr).to_string();
            return Solutions::none(Reason::said(format!("equation reduces to {c} = 0")));
        }
        return Solutions::found(values(solve_rational_poly(arena, var, &poly)));
    }

    // Step 2: Not polynomial over ℚ.  Try symbolic linear solver first:
    // handles a*x + b = 0 where a, b are symbolic expressions.
    if let Some(solutions) = try_solve_linear_symbolic(arena, expr, var)
        && !solutions.is_empty()
    {
        return Solutions::found(values(solutions));
    }

    // Step 2b: polynomial in `var` with symbolic coefficients (degree ≤ 2,
    // or binomial a·xⁿ + b).
    if let Some(solutions) = try_solve_symbolic_poly(arena, expr, var)
        && !solutions.is_empty()
    {
        return Solutions::found(values(solutions));
    }

    // Step 2b'': a polynomial of degree ≥ 3 whose coefficients are
    // algebraic numbers beyond square roots, solved over the field they
    // generate (`(x − ∛2)·(x − i)²`, refused before 0.45), and one over
    // square roots with an irreducible factor of degree ≥ 5 over their
    // field (`x⁵ + √2·x + 1`: `RootOf`s of its norm, refused before 0.45);
    // `Ex::solve` writes the roots of the others in radicals itself.
    let beyond = algebraic_roots::beyond_square_roots(arena, expr, var);
    let min_coeffs = if beyond { 4 } else { 6 };
    if symbolic_poly_coeffs(arena, expr, var).is_some_and(|c| {
        c.len() >= min_coeffs
            && c.iter()
                .all(|&k| crate::base::walk::free_symbols(arena, k).is_empty())
    }) && let Some(roots) = algebraic_roots::number_field_roots(arena, expr, var, !beyond)
    {
        return Solutions::found(roots);
    }

    // Step 2b': a polynomial with symbolic coefficients of higher degree,
    // factored over ℤ[x, parameters].
    let factors = try_solve_parametric_factors(arena, expr, var, cx);
    if factors.has_roots() || factors.is_nested_periods() {
        return factors;
    }

    // Step 2c: rational equations P(x)/Q(x) = 0.
    match try_solve_rational(arena, expr, var, cx) {
        Solutions::Refused(Reason::NotFound) => {}
        decided => return decided,
    }

    // Step 3: transcendental solving via inversion peeling.
    // Handles: exp(x)=c, ln(x)=c, sin(x)=c, sqrt(x)=c, etc.
    let inversion = try_solve_by_inversion(arena, expr, var, cx);
    if inversion.has_roots() || inversion.is_nested_periods() {
        return inversion;
    }
    // Roots under a condition on the parameters are the answer only when
    // no later strategy decides (`(y + 2)^(−1/2) = (1/x)^(−1/2)`: the
    // radical substitution finds `1/x − 2` in the principal sector for every
    // `x`), and never over a proof that there is none.
    let mut conditional = inversion.is_conditional().then_some(inversion.clone());
    // peeling proved "no real solution"
    let mut domain_empty = inversion.is_none_proved();

    // Step 3b: trigonometric functions of rational multiples of `var` with
    // a common factor `r ≠ 1` (`sin 300x + sin 600x`, `cos(x/5) + cos(x/15)`):
    // in `u = r·x` the multiples are coprime integers.
    match try_scale_trig_arguments(arena, expr, var, cx) {
        Solutions::Refused(reason) if reason != Reason::NestedPeriods => {}
        decided => return decided,
    }

    // Step 4: change-of-variable: if expression is polynomial in f(x) for
    // some f, substitute t = f(x), solve the polynomial, back-substitute.
    let changed = try_change_of_variable(arena, expr, var, cx);
    if changed.has_roots() || changed.is_nested_periods() || changed.is_none_proved() {
        return changed;
    }

    // Step 4b: radicals of the variable itself (`√x = x − 2`), or of one
    // linear radicand (`√(x + 1) = x − 1`).
    let radical = try_radical_substitution(arena, expr, var, cx);
    if radical.has_roots() {
        return radical;
    }
    if conditional.is_none() && radical.is_conditional() {
        conditional = Some(radical);
    }
    // Square roots of one or two polynomials (`√(x + 5) + √x = 3`).
    match try_square_root_elimination(arena, expr, var) {
        Some(Unradicated::Roots(solutions)) if !solutions.is_empty() => {
            return Solutions::found(solutions);
        }
        // The norm over the sign choices is a nonzero constant: no root
        // (`x + √(x² − 5) = 0`: the norm is 5; refused before 0.47).
        Some(Unradicated::NormConstant) => {
            return Solutions::none(Reason::said(
                "the radicals cancel to a nonzero constant under every choice of signs",
            ));
        }
        _ => {}
    }
    // Several absolute values of rational functions (`|x + 3| = 2·|x − 3|`).
    let abs_cases = try_abs_cases(arena, expr, var, cx);
    if abs_cases.has_roots() || abs_cases.is_none_proved() || abs_cases.is_nested_periods() {
        return abs_cases;
    }

    // Step 4c: exponentials of one rate (`e^x + e^(−x) = 3`,
    // `4^x − 5·2^x + 6 = 0`), trigonometric functions of one angle
    // (`sin x + cos x = 1`, `sin 2x = sin x`), sums of logarithms.
    for hyperbolic in [false, true] {
        let outcome = if hyperbolic {
            // Not after peeling proved there is no real solution (`cosh x =
            // 1/2`), which the exponentials would answer with complex roots.
            if domain_empty {
                continue;
            }
            try_hyperbolic_substitution(arena, expr, var, cx)
        } else {
            try_exponential_substitution(arena, expr, var, cx)
        };
        if outcome.has_roots() || outcome.is_nested_periods() {
            return outcome;
        }
        domain_empty |= outcome.is_none_proved();
    }
    // Exponentials of different bases in two groups (`5^(x/2) = 2^(x/3)`,
    // `7^(x² − 80) = 49^x`): logarithms of both sides.
    if !domain_empty {
        let split = try_exponential_log_split(arena, expr, var, cx);
        if split.has_roots() || split.is_nested_periods() {
            return split;
        }
        domain_empty = split.is_none_proved();
    }
    let half_angle = try_half_angle_substitution(arena, expr, var, cx);
    if half_angle.has_roots() || half_angle.is_nested_periods() {
        return half_angle;
    }
    let log_sum = try_log_sum(arena, expr, var, cx);
    if log_sum.has_roots() || log_sum.is_nested_periods() {
        return log_sum;
    }

    // Step 5: LambertW for mixed polynomial-exponential equations:
    // x·exp(x) = c, x·exp(a·x) = c, exp(x) + x = c, etc.
    let var_sym_opt = match arena.node(var) {
        ExprNode::Symbol(sid) => Some(*sid),
        _ => None,
    };
    if let Some(var_sym) = var_sym_opt
        && !cx.variable_may_be_nonreal(arena, var)
    {
        let lambert = try_solve_lambert(arena, expr, var, var_sym, cx);
        if lambert.has_roots() || lambert.is_nested_periods() {
            return lambert;
        }
        let power_exp = try_solve_power_exp(arena, expr, var, cx);
        if power_exp.has_roots() || power_exp.is_nested_periods() {
            return power_exp;
        }
        let power_log = try_lambert_power_log(arena, expr, var, cx);
        if power_log.has_roots() {
            return power_log;
        }
        let power_of_log = try_power_of_log(arena, expr, var, cx);
        if power_of_log.has_roots() {
            return power_of_log;
        }
        let monomial = try_monomial_substitution(arena, expr, var, cx);
        if monomial.has_roots() || monomial.is_nested_periods() {
            return monomial;
        }
    }

    if domain_empty {
        // (for a complex unknown the restriction is one that holds for
        // every argument, `e^f = 0`, `|f| = −1`, or for an argument real
        // by its form: up to 0.52 the text said "no real solutions" there)
        return Solutions::none(Reason::said(if cx.unknown_may_be_nonreal(arena, var) {
            "equation has no solutions (range restriction: an inverted function does not take \
             the value)"
        } else {
            "equation has no real solutions (range restriction of exp/sin/cos/cosh/abs)"
        }));
    }

    // Step 5a: the unknown inside Lambert W and outside it (`e^{W(u)} =
    // u/W(u)`; `W(f) = g` gives `f = g·e^g`).
    let lambert_inverse = try_lambert_inverse(arena, expr, var, cx);
    if lambert_inverse.has_roots()
        || lambert_inverse.is_none_proved()
        || lambert_inverse.is_nested_periods()
    {
        return lambert_inverse;
    }

    // Step 5b: the unknown only through linear forms with one common shift
    // (`sin(2x − 1) + cos(2x − 1) + 1/2`), solved in the shifted unknown.
    let shifted = try_shifted_argument(arena, expr, var, cx);
    if shifted.has_roots() || shifted.is_none_proved() || shifted.is_nested_periods() {
        return shifted;
    }
    if conditional.is_none() && shifted.is_conditional() {
        conditional = Some(shifted);
    }

    // Step 6: the numerator over a polynomial denominator (`ln(x)/x = 1/4`
    // as `ln x = x/4`, `(eˣ + 1/3)/x = 0` as `eˣ = −1/3`): every root of
    // `expr` is one of the numerator, whose candidates `solve_impl` checks
    // against `expr` (the poles are dropped), and a numerator without
    // solutions proves `expr` has none.  The numerator has no polynomial
    // denominator left, so this does not repeat.  Before 0.44 these were
    // refused unless the whole was a product.
    if let Some(num) = numerator_over_polynomial(arena, expr, var)
        && numerator_over_polynomial(arena, num, var).is_none()
    {
        let outcome = solve_raw(arena, num, var, cx);
        if outcome.has_roots() || outcome.is_none_proved() || outcome.is_nested_periods() {
            return outcome;
        }
        if conditional.is_none() && outcome.is_conditional() {
            conditional = Some(outcome);
        }
    }
    conditional.unwrap_or(Solutions::Refused(Reason::NotFound))
}

/// The argument `z` of a Lambert W value `W_k(z)` (any branch, `k` an
/// integer), or `None`.
fn lambert_argument(arena: &Arena, w: ExprId) -> Option<ExprId> {
    match arena.node(w) {
        ExprNode::LambertW(z) => Some(*z),
        ExprNode::Lib(crate::base::libfn::LibFn::LambertW, args) => match args[..] {
            [z] => Some(z),
            [z, k] if arena.as_num(k).is_some_and(|q| q.is_integer()) => Some(z),
            _ => None,
        },
        _ => None,
    }
}

/// `e` with each Lambert W value `W₀(z)` or `W₋₁(z)` at a constant `z = g·e^g`
/// replaced by `g`, for a real `g` in the range of the branch (`g ≥ −1` for
/// `W₀`, `g ≤ −1` for `W₋₁`, DLMF 4.13): `W(e) = 1`, `W(e^{3 + e³}) = e³`,
/// `W(2·ln 2) = ln 2`.  The candidates `g` are the factors of `z` and the
/// terms of the exponent of an exponential factor of it; `z·e^{−g} − g`
/// (with `e^{−g}` taken into that exponential) must be proved 0
/// ([`constant_zero_decision`](crate::transforms::eval::constant_zero_decision)),
/// `g` proved real and on the side of `−1` the branch takes.  `None` when
/// nothing is replaced.  For the check of a fitted initial-value solution,
/// whose residual at the initial point holds such values (`C1 − W(e^{C1}) =
/// 3` gives `C1 = 3 + e³`, and the residual `e³ − W(e^{3 + e³})`).
pub(crate) fn lambert_values(arena: &mut Arena, e: ExprId) -> Option<ExprId> {
    use crate::base::truth::{SignSet, Truth};
    // (branch, argument) of a W node with a symbol-free argument
    let branch_of = |arena: &Arena, w: ExprId| -> Option<(i8, ExprId)> {
        let (k, z) = match arena.node(w) {
            ExprNode::LambertW(z) => (0, *z),
            ExprNode::Lib(crate::base::libfn::LibFn::LambertW, args) => match args[..] {
                [z] => (0, z),
                [z, k] if k == arena.zero => (0, z),
                [z, k] if k == arena.neg_one => (-1, z),
                _ => return None,
            },
            _ => return None,
        };
        crate::base::walk::free_symbols(arena, z)
            .is_empty()
            .then_some((k, z))
    };
    let value = |arena: &mut Arena, k: i8, z: ExprId| -> Option<ExprId> {
        let factors: Vec<ExprId> = match arena.node(z) {
            ExprNode::Mul(c) => c.to_vec(),
            _ => vec![z],
        };
        let mut candidates: Vec<ExprId> = Vec::new();
        for &f in &factors {
            if let ExprNode::Exp(s) = *arena.node(f) {
                match arena.node(s) {
                    ExprNode::Add(terms) => candidates.extend(terms.iter().copied()),
                    _ => candidates.push(s),
                }
            } else if f == arena.e_const {
                candidates.push(arena.one);
            } else if arena.as_num(f).is_none() {
                candidates.push(f);
            }
        }
        for g in candidates {
            // `z·e^{−g} = g`, the `e^{−g}` taken into the first exponential
            // factor of `z` (`e^{1 + e}·e^{−e} = e`, which the zero tests do not
            // combine with the atom `e`)
            let mut quotient: Vec<ExprId> = Vec::with_capacity(factors.len() + 1);
            let mut absorbed = false;
            for &f in &factors {
                match *arena.node(f) {
                    ExprNode::Exp(s) if !absorbed => {
                        absorbed = true;
                        let d = arena.sub(s, g);
                        quotient.push(arena.exp(d));
                    }
                    _ => quotient.push(f),
                }
            }
            if !absorbed {
                let mg = arena.neg(g);
                quotient.push(arena.exp(mg));
            }
            let q = arena.mul(&quotient);
            let d = arena.sub(q, g);
            let d = crate::transforms::eval::eval(arena, d);
            let zero = arena.is_zero_structural(d)
                || Truth::from(crate::transforms::eval::constant_zero_decision(arena, d)).is_true();
            if !zero || crate::transforms::realness::imaginary_part_sign(arena, g) != SignSet::ZERO
            {
                continue;
            }
            let one = arena.one;
            let shifted = arena.add(&[g, one]);
            let side = part_signs(arena, shifted);
            if side.im == SignSet::ZERO
                && ((k == 0 && side.re.is_subset(SignSet::NONNEGATIVE))
                    || (k == -1 && side.re.is_subset(SignSet::NONPOSITIVE)))
            {
                return Some(g);
            }
        }
        None
    };
    let post = crate::base::walk::post_order_ids(arena, e);
    let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    let mut changed = false;
    for &id in &post {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match branch_of(arena, rebuilt) {
            Some((k, z)) => match value(arena, k, z) {
                Some(g) => {
                    changed = true;
                    g
                }
                None => rebuilt,
            },
            None => rebuilt,
        };
        if new != id {
            cache.insert(id, new);
        }
    }
    if !changed {
        return None;
    }
    let out = cache.get(&e).copied().unwrap_or(e);
    Some(crate::transforms::eval::eval(arena, out))
}

/// `e` with every `e^{k·W(z) + s}` (`k` an integer up to 8, `W` any branch
/// of Lambert W, `z` with `var`) written `(z/W(z))^k·e^s`, and the
/// arguments `z` so rewritten; `None` when there is none.  Every branch
/// solves `W·e^W = z`, so `e^{W(z)} = z/W(z)` wherever `W(z) ≠ 0`, that
/// is off `z = 0` (only `W₀(0)` is 0).  (The same rewrite as the ODE
/// check's, `calculus::ode`, for the unknown inside `W`.)
fn exp_lambert_to_ratio(
    arena: &mut Arena,
    e: ExprId,
    var: ExprId,
) -> Option<(ExprId, Vec<ExprId>)> {
    const MAX_POWER: i64 = 8;
    let post = crate::base::walk::post_order_ids(arena, e);
    let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    let mut args: Vec<ExprId> = Vec::new();
    for &id in &post {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match *arena.node(rebuilt) {
            ExprNode::Exp(arg) => {
                let terms: Vec<ExprId> = match arena.node(arg) {
                    ExprNode::Add(c) => c.to_vec(),
                    _ => vec![arg],
                };
                let mut factors = Vec::new();
                let mut rest = Vec::new();
                for t in terms {
                    let (k, term) = arena.as_coeff_term(t);
                    let power = k
                        .is_integer()
                        .then(|| num_traits::ToPrimitive::to_i64(&k.to_integer()))
                        .flatten()
                        .filter(|p| p.abs() <= MAX_POWER);
                    match (lambert_argument(arena, term), power) {
                        (Some(z), Some(p)) if expr_contains_var(arena, z, var) => {
                            let ratio = arena.div(z, term);
                            let pe = arena.int(p);
                            factors.push(arena.pow(ratio, pe));
                            if !args.contains(&z) {
                                args.push(z);
                            }
                        }
                        _ => rest.push(t),
                    }
                }
                if factors.is_empty() {
                    rebuilt
                } else {
                    if !rest.is_empty() {
                        let r = arena.add(&rest);
                        factors.push(arena.exp(r));
                    }
                    arena.mul(&factors)
                }
            }
            _ => rebuilt,
        };
        cache.insert(id, new);
    }
    if args.is_empty() {
        return None;
    }
    let out = cache.get(&e).copied().unwrap_or(e);
    Some((crate::transforms::eval::eval(arena, out), args))
}

/// The sum `e` with the factors common to all its terms (non-numeric, the
/// same node, with multiplicity) taken out as a product: `2x·W(2x)⁻¹ − x`
/// is `x·(2·W(2x)⁻¹ − 1)`.  `e` itself when there are none.
fn with_common_factors_out(arena: &mut Arena, e: ExprId) -> ExprId {
    let ExprNode::Add(terms) = arena.node(e).clone() else {
        return e;
    };
    let factors_of = |arena: &Arena, t: ExprId| -> Vec<ExprId> {
        match arena.node(t) {
            ExprNode::Mul(ch) => ch
                .iter()
                .copied()
                .filter(|&f| arena.as_num(f).is_none())
                .collect(),
            ExprNode::Num(_) => Vec::new(),
            _ => vec![t],
        }
    };
    let mut common = factors_of(arena, terms[0]);
    for &t in &terms[1..] {
        let mut fs = factors_of(arena, t);
        common.retain(|c| match fs.iter().position(|f| f == c) {
            Some(i) => {
                fs.swap_remove(i);
                true
            }
            None => false,
        });
        if common.is_empty() {
            return e;
        }
    }
    let product = arena.mul(&common);
    let rest: Vec<ExprId> = terms
        .iter()
        .map(|&t| {
            let q = arena.div(t, product);
            crate::transforms::eval::eval(arena, q)
        })
        .collect();
    let rest = arena.add(&rest);
    arena.mul(&[product, rest])
}

/// Equations in which the unknown is inside Lambert W and also outside
/// it, which peeling cannot isolate:
///
/// - `e^{W(u)}` is written `u/W(u)` ([`exp_lambert_to_ratio`]) and the
///   rewritten equation is solved, with the roots of `u = 0` (where the
///   identity fails) as further candidates; every candidate is checked
///   against the equation itself.  `C1·e^{W(2/C1)} = 2` becomes `2/W(2/C1)
///   = 2`, so `C1 = 2/e`.
/// - `a·W_k(f) + b = 0` with `a` free of the unknown: `W_k(f) = g` for `g =
///   −b/a` implies `f = g·e^g`, i.e. `f·e^{−g} = g`, whose roots are
///   candidates (checked: `g` must be a value of the branch).  `C1 −
///   W(e^{C1}) = 3` gives `e^{C1}·e^{3 − C1} = C1 − 3`, so `C1 = 3 + e³`.
///   A candidate with parameters is not checked, and refused in a public
///   solve.
///
/// Up to 0.52 both were refused, so the initial values of Lambert-W
/// families of ODEs could not be fitted (`y′ = (3y − x)/(x + y)`, `y(1) =
/// 3`, on `C1·e^{W(2x/C1)} + x`).
fn try_lambert_inverse(arena: &mut Arena, expr: ExprId, var: ExprId, cx: &SolveCx) -> Solutions {
    let not_found = || Solutions::refused(Reason::NotFound);
    if let Some((rewritten, args)) = exp_lambert_to_ratio(arena, expr, var)
        && rewritten != expr
    {
        // (`2x/W(2x) − x` as `x·(2/W(2x) − 1)`, which the product rule solves)
        let rewritten = with_common_factors_out(arena, rewritten);
        let mut outcome = solve_raw(arena, rewritten, var, cx).without_conditions();
        for z in args {
            outcome = outcome.union(solve_raw(arena, z, var, cx).without_conditions());
        }
        let outcome = checked_candidates(arena, expr, var, outcome);
        if outcome.has_roots() || outcome.is_none_proved() || outcome.is_nested_periods() {
            return outcome;
        }
    }
    // the one Lambert W of the unknown, linearly
    let ws: Vec<ExprId> = crate::base::walk::post_order_ids(arena, expr)
        .into_iter()
        .filter(|&id| lambert_argument(arena, id).is_some_and(|z| expr_contains_var(arena, z, var)))
        .collect();
    let Some(&w) = ws.first() else {
        return not_found();
    };
    if ws.iter().any(|&v| v != w) {
        return not_found();
    }
    let Some(f) = lambert_argument(arena, w) else {
        return not_found();
    };
    let s = crate::transforms::subs::fresh_symbol(arena, "__w_lambert", &[expr]);
    let in_s = arena.subs_structural(expr, w, s);
    if !expr_contains_var(arena, in_s, var) {
        return not_found();
    }
    let (b, a) = match symbolic_poly_coeffs(arena, in_s, s).as_deref() {
        Some(&[b, a]) => (b, a),
        _ => return not_found(),
    };
    if expr_contains_var(arena, a, var)
        || expr_contains_var(arena, a, s)
        || expr_contains_var(arena, b, s)
        || arena.is_zero_structural(a)
    {
        return not_found();
    }
    let neg_b = arena.neg(b);
    let g = arena.div(neg_b, a);
    let g = crate::transforms::eval::eval(arena, g);
    // `f·e^{−g}` with the exponentials of `f` merged into one (`e^{C1}·e^{3 −
    // C1}` is not combined by the canonical form; `e^{C1 + 3 − C1} = e³`)
    let mut exponent = vec![arena.neg(g)];
    let mut rest: Vec<ExprId> = Vec::new();
    let f_factors: Vec<ExprId> = match arena.node(f) {
        ExprNode::Mul(ch) => ch.to_vec(),
        _ => vec![f],
    };
    for factor in f_factors {
        match *arena.node(factor) {
            ExprNode::Exp(h) => exponent.push(h),
            _ => rest.push(factor),
        }
    }
    let exponent = arena.add(&exponent);
    let exponent = crate::transforms::expand::expand(arena, exponent);
    let exponent = crate::transforms::eval::eval(arena, exponent);
    rest.push(arena.exp(exponent));
    let fe = arena.mul(&rest);
    let eq = arena.sub(fe, g);
    let eq = crate::transforms::eval::eval(arena, eq);
    let eq = crate::transforms::expand::expand(arena, eq);
    let eq = crate::transforms::eval::eval(arena, eq);
    if crate::base::walk::post_order_ids(arena, eq)
        .into_iter()
        .any(|id| lambert_argument(arena, id).is_some())
    {
        return not_found();
    }
    let outcome = solve_raw(arena, eq, var, cx).without_conditions();
    let parametric = outcome.roots().is_some_and(|rs| {
        rs.iter().any(|&r| {
            crate::base::walk::free_symbols(arena, r)
                .into_iter()
                .any(|sym| Some(sym) != cx.period())
        })
    });
    if parametric && !cx.conditional_roots_as_candidates() {
        return not_found();
    }
    let outcome = checked_candidates(arena, expr, var, outcome);
    if outcome.has_roots() || outcome.is_none_proved() || outcome.is_nested_periods() {
        return outcome;
    }
    not_found()
}

/// `outcome`, the candidates of a derived equation, checked against `expr`
/// itself (not the whole equation, which may have other factors with
/// parameters, where the final check cannot evaluate: `(W(x) + 2x)·g(y)`
/// got the candidate `(ln 2 − iπ)/2` of `x·(e^{2x} + 2) = 0`, where `W₀`
/// is not `−ln 2 + iπ`, which is the value of another branch).  A root is
/// dropped when its residual certainly is not 0, makes the outcome partial
/// when undecided ([`Residual::Undecided`]), and is kept otherwise, as in
/// [`drop_certain_non_roots`]; with every candidate dropped
/// and none missing there is no root (the derived equation has every root
/// of `expr`).  Candidates with parameters or with the integer parameter of
/// a family are left to the final check.
fn checked_candidates(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    outcome: Solutions,
) -> Solutions {
    let (roots, missing) = match outcome.into_found() {
        Ok(found) => found,
        Err(other) => return other,
    };
    let mut kept: Vec<ExprId> = Vec::with_capacity(roots.len());
    let mut undecided = false;
    for r in roots {
        if !crate::base::walk::free_symbols(arena, r).is_empty() {
            kept.push(r);
            continue;
        }
        // (as `drop_certain_non_roots`)
        match residual_at(arena, expr, var, r) {
            Residual::Zero | Residual::Unchecked => kept.push(r),
            Residual::Nonzero => {}
            Residual::Undecided => undecided = true,
        }
    }
    let missing = if undecided {
        Missing::merged(missing, Some(Missing::Alternative(Reason::NotFound)))
    } else {
        missing
    };
    // Every candidate failing is no proof: they are the principal roots of
    // the derived equation (`W₋₁(x) = −x` gives `x·(eˣ + 1) = 0`, whose
    // principal roots `0`, `iπ` fail while mpmath's `W₋₁(iπ) = −iπ`
    // convention makes another member a root).
    if kept.is_empty() && missing.is_none() {
        return Solutions::refused(Reason::said(
            "every principal candidate fails the equation (a branch of Lambert W does not reach \
             the value); the other roots of the derived equation are not searched",
        ));
    }
    Solutions::with_missing(kept, missing)
}

/// The values of a list of [`Solution`]s.
fn values(solutions: Vec<Solution>) -> Vec<ExprId> {
    solutions.into_iter().map(|s| s.value).collect()
}

/// Post-process a list of candidate solutions: evaluate exact special
/// values (`asin(1/2)` → `π/6`), drop infinite / undefined candidates
/// (e.g. `1/x = 0` → `zoo`), and deduplicate; `missing` is what the
/// candidates may lack (a partial outcome).
fn finalize_solutions(
    arena: &mut Arena,
    solutions: Vec<ExprId>,
    missing: Option<Missing>,
) -> Solutions {
    let had_candidates = !solutions.is_empty();
    let mut out: Vec<ExprId> = Vec::with_capacity(solutions.len());
    // The 16-digit values of the constant solutions kept, for the test of
    // equal values written differently.
    let mut values: Vec<Option<num_complex::Complex64>> = Vec::with_capacity(solutions.len());
    // A candidate with an infinity inside a function and not known finite is
    // not claimed, nor is it dropped as infinite: up to 0.54 `x +
    // chebyshev_t(2, y·∞) = 0` gave `[−chebyshev_t(2, y·∞)]` (0.53's "no
    // solution" came from `eval` expanding `T₂`), `x + sin ∞ = 0` `[−sin ∞]`.
    let mut not_finite: Option<ExprId> = None;
    for s in solutions {
        let v = crate::transforms::eval::eval(arena, s);
        if has_infinite_term(arena, v) {
            continue;
        }
        if not_known_finite(arena, v) {
            not_finite.get_or_insert(v);
            continue;
        }
        if out.contains(&v) {
            continue;
        }
        // One value in several forms is one solution: up to 0.46 `t³` for
        // the three roots `t` of a cubic in `t = x^(1/3)` came back as
        // `100√2·π/481` and two radical forms of the same number.
        let z = constant_value_f64(arena, v);
        if let Some(z) = z
            && out
                .iter()
                .zip(&values)
                .any(|(&o, w)| w.is_some_and(|w| near_values(w, z)) && same_constant(arena, o, v))
        {
            continue;
        }
        values.push(z);
        out.push(v);
    }
    let missing = match not_finite {
        Some(v) => Missing::merged(
            missing,
            Some(Missing::Alternative(Reason::said(format!(
                "the candidate {} holds an infinity inside a function and is not known to be finite",
                arena.display(v)
            )))),
        ),
        None => missing,
    };
    if had_candidates && out.is_empty() && missing.is_none() {
        return Solutions::none(Reason::said(
            "every candidate solution is infinite or undefined (e.g. 1/x = 0)",
        ));
    }
    Solutions::with_missing(out, missing)
}

/// Is `v` infinite or undefined: `∞`, `−∞`, `zoo`, `nan`, or a sum or
/// product with one of them (a directed infinity `i·∞`, a family `nπ +
/// i·∞`)?  `tan z = i` inverts to `atan(i) = i·∞`: up to 0.47 `solve`
/// returned `[I*oo]` and `solve_as_set` the family `nπ + I*oo`, while `tan
/// z = ±i` has no solution (`e^(2iz) = 0`; SymPy: `[]`).
fn has_infinite_term(arena: &Arena, v: ExprId) -> bool {
    let mut stack = vec![v];
    while let Some(e) = stack.pop() {
        match arena.node(e) {
            ExprNode::Infinity
            | ExprNode::NegInfinity
            | ExprNode::ComplexInfinity
            | ExprNode::NaN => return true,
            ExprNode::Add(ch) | ExprNode::Mul(ch) => stack.extend(ch.iter().copied()),
            _ => {}
        }
    }
    false
}

/// The 16-digit value of a constant solution with an algebraic or
/// transcendental form (not a rational, not a `RootOf`: those are
/// distinct by construction), for [`finalize_solutions`].
fn constant_value_f64(arena: &mut Arena, v: ExprId) -> Option<num_complex::Complex64> {
    if arena.as_num(v).is_some()
        || !crate::base::walk::free_symbols(arena, v).is_empty()
        || crate::base::walk::post_order_ids(arena, v)
            .into_iter()
            .any(|id| matches!(arena.node(id), ExprNode::RootOf(..)))
    {
        return None;
    }
    crate::transforms::evalf::evalf_complex64(arena, v)
        .ok()
        .filter(|z| z.re.is_finite() && z.im.is_finite())
}

/// Are two 16-digit values ([`constant_value_f64`]) close enough to be
/// compared exactly by [`same_constant`]?  A guide only: `same_constant`
/// decides.
fn near_values(w: num_complex::Complex64, z: num_complex::Complex64) -> bool {
    (w - z).norm() <= 1e-9 * (1.0 + z.norm())
}

/// Are the constants `a` and `b` exactly equal (their difference decided
/// to be 0)?  An algebraic multiple of `π` is decided as the algebraic
/// number `(a − b)/π` (`15√3πi·∛20·20000^(1/6)/1924 + … = 100√2π/481`).
fn same_constant(arena: &mut Arena, a: ExprId, b: ExprId) -> bool {
    let d = arena.sub(a, b);
    if crate::transforms::eval::constant_zero_decision(arena, d) == Some(true) {
        return true;
    }
    let d = crate::transforms::expand::expand(arena, d);
    let q = arena.div(d, arena.pi);
    let q = crate::transforms::expand::expand(arena, q);
    let q = crate::transforms::eval::eval(arena, q);
    !crate::base::walk::contains(arena, q, arena.pi)
        && crate::poly::algebraic::is_zero_checked(arena, q) == Some(true)
}

/// Dispatch a nonconstant polynomial with rational coefficients to the
/// degree-specific solvers.
fn solve_rational_poly(arena: &mut Arena, var: ExprId, poly: &Poly) -> Vec<Solution> {
    let degree = poly.degree().unwrap_or(0);
    tracing::debug!(degree = degree, "polynomial degree determined");
    match degree {
        0 => Vec::new(),
        1 => solve_linear(arena, poly),
        2 => solve_quadratic(arena, poly),
        3 => solve_cubic(arena, var, poly),
        4 => solve_quartic(arena, var, poly),
        _ => solve_rational_roots(arena, var, poly),
    }
}

/// Classify an expression that does **not** depend on the solve variable.
///
/// Decides whether `expr = 0` is an identity, a contradiction, or
/// undecidable (symbolic constant).  Symbolic constants that are not
/// provably zero are reported as `NoSolution` with an explanatory reason
/// — the equation holds only for special values of the parameters.
fn classify_constant(arena: &mut Arena, expr: ExprId, var: ExprId) -> Solutions {
    let var_name = arena.display(var).to_string();
    let evaled = crate::transforms::eval::eval(arena, expr);
    if arena.is_zero_structural(evaled) {
        return Solutions::Complete(RootSet::All);
    }
    if let Some(r) = arena.as_num(evaled)
        && !r.is_zero()
    {
        return Solutions::none(Reason::said(format!("equation reduces to {r} = 0")));
    }
    // Try harder: expand + eval.
    let expanded = crate::transforms::expand::expand(arena, evaled);
    let expanded = crate::transforms::eval::eval(arena, expanded);
    if arena.is_zero_structural(expanded) {
        return Solutions::Complete(RootSet::All);
    }
    if let Some(r) = arena.as_num(expanded)
        && !r.is_zero()
    {
        return Solutions::none(Reason::said(format!("equation reduces to {r} = 0")));
    }
    // Purely numeric constant (no free symbols): the certified zero test
    // (a certified digit, or the minimal polynomial of an algebraic
    // number).  Before 0.31 `|evalf(c, 20)| < 10⁻¹⁵` was an identity:
    // `solve(exp(−40), x)` reported every `x` as a solution.  A constant
    // the test does not decide (one without a value, `δ(i)`, or with a
    // hidden zero in a denominator, `1/max(Γ(1/3)Γ(2/3) − 2π/√3, 0)`) is
    // refused: up to 0.51 it was "no solution" (`solve_as_set`:
    // `EmptySet`).  An infinite one is not zero.
    if crate::base::walk::free_symbols(arena, expanded).is_empty() {
        match crate::poly::algebraic::is_zero_checked(arena, expanded) {
            Some(false) => {
                let shown = arena.display(expr).to_string();
                return Solutions::none(Reason::said(format!(
                    "equation reduces to the nonzero constant {shown} = 0"
                )));
            }
            Some(true) => return Solutions::Complete(RootSet::All),
            // infinite: not zero (below)
            None if has_infinite_term(arena, expanded)
                && !matches!(arena.node(expanded), ExprNode::NaN) => {}
            None => {
                // What `Ex::is_zero` decides from the structure and the
                // value (a reciprocal is never 0; a known sign)
                use crate::base::assumptions::{AssumptionCache, Props};
                use crate::base::truth::Truth;
                let zero = AssumptionCache::new()
                    .query(arena, expr, Props::ZERO)
                    .or_else(|| {
                        crate::transforms::eval::constant_facts(arena, expr).query(Props::ZERO)
                    });
                match Truth::from(zero) {
                    Truth::True => return Solutions::Complete(RootSet::All),
                    Truth::False => {
                        let shown = arena.display(expr).to_string();
                        return Solutions::none(Reason::said(format!(
                            "equation reduces to the nonzero constant {shown} = 0"
                        )));
                    }
                    Truth::Unknown => {}
                }
                // Zero to the precision of the deep zero search (1,024 bits
                // beyond the working precision) but not provably so: an
                // identity, the tolerance the old `10⁻¹⁵` stood for.
                use crate::transforms::evalf::{Settled, ZeroSearch};
                if let Ok((_, Settled::ZeroToPrecision)) =
                    crate::transforms::evalf::evalf_settled(arena, expanded, 20, ZeroSearch::Deep)
                {
                    return Solutions::Complete(RootSet::All);
                }
                let shown = arena.display(expr).to_string();
                return Solutions::refused(Reason::said(format!(
                    "cannot decide whether the constant {shown} is zero (it does not evaluate)"
                )));
            }
        }
    }
    // With parameters: zero only on a set of measure zero of their values
    // (the generic convention), unless an infinity inside a function makes
    // the value constant on pieces: `e^{−y·∞}` is 0 for every `y > 0`, and
    // `chebyshev_t(2, y·∞)` has no finite value (up to 0.54 both were "no
    // solution").  An infinite term makes the value infinite or undefined
    // for every value of the parameters (`y·∞ + 1`): no solution.
    if holds_infinity(arena, expanded) && !has_infinite_term(arena, expanded) {
        use crate::base::assumptions::{AssumptionCache, Props};
        use crate::base::truth::Truth;
        let shown = arena.display(expr).to_string();
        let mut cache = AssumptionCache::new();
        let zero = Truth::from(cache.query(arena, expanded, Props::ZERO));
        // (not finite wherever it is defined: infinite, never 0)
        let infinite = Truth::from(cache.query(arena, expanded, Props::FINITE)).is_false();
        return match zero {
            Truth::True => Solutions::Complete(RootSet::All),
            Truth::False => Solutions::none(Reason::said(format!(
                "equation reduces to {shown} = 0, which is nonzero"
            ))),
            Truth::Unknown if infinite => Solutions::none(Reason::said(format!(
                "equation reduces to {shown} = 0, which is infinite"
            ))),
            Truth::Unknown => Solutions::refused(Reason::said(format!(
                "cannot decide whether {shown} (an infinity inside a function) is zero for the \
                 values of the parameters"
            ))),
        };
    }
    let shown = arena.display(expr).to_string();
    Solutions::none(Reason::said(format!(
        "expression {shown} does not depend on {var_name} and is not identically zero"
    )))
}

// ═══════════════════════════════════════════════════════════════════════════
// Local helper: check if an expression contains a given variable
// ═══════════════════════════════════════════════════════════════════════════

fn expr_contains_var(arena: &Arena, expr: ExprId, var: ExprId) -> bool {
    crate::base::walk::contains(arena, expr, var)
}

// ═══════════════════════════════════════════════════════════════════════════
// Symbolic-coefficient polynomial solving
// ═══════════════════════════════════════════════════════════════════════════

/// Extract ascending coefficients of `expr` viewed as a polynomial in
/// `var`, allowing arbitrary var-free symbolic coefficients.
///
/// Returns `None` if `var` appears in a non-polynomial position.
pub(crate) fn symbolic_poly_coeffs(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Option<Vec<ExprId>> {
    let expanded = crate::transforms::expand::expand(arena, expr);
    let terms: Vec<ExprId> = match arena.node(expanded).clone() {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![expanded],
    };
    let mut buckets: Vec<Vec<ExprId>> = Vec::new();
    for term in terms {
        let (deg, coeff) = term_degree_coeff(arena, term, var)?;
        if buckets.len() <= deg {
            buckets.resize_with(deg + 1, Vec::new);
        }
        buckets[deg].push(coeff);
    }
    let mut coeffs = Vec::with_capacity(buckets.len());
    for bucket in buckets {
        let c = match bucket.len() {
            0 => arena.zero,
            1 => bucket[0],
            _ => arena.add(&bucket),
        };
        coeffs.push(crate::transforms::eval::eval(arena, c));
    }
    // Trim leading zeros.
    while coeffs.len() > 1 && arena.is_zero_structural(*coeffs.last()?) {
        coeffs.pop();
    }
    Some(coeffs)
}

/// Split a single product term into `(degree in var, var-free coefficient)`.
fn term_degree_coeff(arena: &mut Arena, term: ExprId, var: ExprId) -> Option<(usize, ExprId)> {
    if term == var {
        return Some((1, arena.one));
    }
    if !expr_contains_var(arena, term, var) {
        return Some((0, term));
    }
    match arena.node(term).clone() {
        ExprNode::Pow(base, exp) if base == var => {
            let n = arena.as_num(exp)?.clone();
            if !n.is_integer() || n.is_negative() {
                return None;
            }
            let d: usize = n.to_integer().try_into().ok()?;
            Some((d, arena.one))
        }
        ExprNode::Neg(inner) => {
            let (d, c) = term_degree_coeff(arena, inner, var)?;
            Some((d, arena.neg(c)))
        }
        ExprNode::Mul(children) => {
            let mut deg = 0usize;
            let mut consts: Vec<ExprId> = Vec::new();
            for &child in &children {
                if !expr_contains_var(arena, child, var) {
                    consts.push(child);
                } else if child == var {
                    deg += 1;
                } else if let ExprNode::Pow(base, exp) = arena.node(child).clone()
                    && base == var
                {
                    let n = arena.as_num(exp)?.clone();
                    if !n.is_integer() || n.is_negative() {
                        return None;
                    }
                    let d: usize = n.to_integer().try_into().ok()?;
                    deg += d;
                } else {
                    return None;
                }
            }
            let c = match consts.len() {
                0 => arena.one,
                1 => consts[0],
                _ => arena.mul(&consts),
            };
            Some((deg, c))
        }
        _ => None,
    }
}

/// Solve a polynomial in `var` whose coefficients are symbolic but free
/// of `var`.  Handles degree 1, degree 2 (quadratic formula) and the
/// binomial form `a·xⁿ + b` (roots of unity).
fn try_solve_symbolic_poly(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<Vec<Solution>> {
    let coeffs = symbolic_poly_coeffs(arena, expr, var)?;
    solve_symbolic_coeffs(arena, &coeffs)
}

/// Solve from ascending symbolic coefficients (see [`try_solve_symbolic_poly`]).
///
/// The linear root `-b/a` and the quadratic discriminant are put into
/// rational normal form with `ratsimp`, so parametric coefficients that are
/// themselves fractions (`(3r - 1)/(j + 1) - (r + 1)/(2j)`) yield a single
/// cancelled fraction rather than a fraction of fractions.
///
/// A constant leading coefficient or discriminant that is 0 without being
/// written 0 is decided exactly ([`proved_zero_constant`]): canonical forms
/// keep `4^(3/4)` beside `2√2`, and before 0.51 the discriminant `8√2 −
/// 4·4^(3/4)` of `4^(3/4)·t² + 2·2^(3/4)·t + 1` (from `4^(x + 3/4) +
/// 2·2^(x + 3/4) + 1 = 0`, `t = 2ˣ`) split the double root `t = −2^(−3/4)`
/// into two forms whose sign no evaluation decides, and a real `x` got two
/// logarithms of negative numbers as roots (the left side is positive).
fn solve_symbolic_coeffs(arena: &mut Arena, coeffs: &[ExprId]) -> Option<Vec<Solution>> {
    let mut len = coeffs.len();
    while len > 1 && proved_zero_constant(arena, coeffs[len - 1]) {
        len -= 1;
    }
    let coeffs = &coeffs[..len];
    let degree = coeffs.len().checked_sub(1)?;
    match degree {
        0 => None,
        1 => {
            let a = coeffs[1];
            let b = coeffs[0];
            let neg_b = arena.neg(b);
            let v = arena.div(neg_b, a);
            let v = crate::transforms::eval::eval(arena, v);
            let v = crate::simplify::ratsimp::ratsimp(arena, v);
            Some(vec![Solution { value: v }])
        }
        2 => {
            let a = coeffs[2];
            let b = coeffs[1];
            let c = coeffs[0];
            if arena.is_zero_structural(b) {
                // a·x² + c = 0 → x = ±sqrt(-c/a)
                let neg_c = arena.neg(c);
                let ratio = arena.div(neg_c, a);
                let ratio = crate::transforms::eval::eval(arena, ratio);
                let root = arena.sqrt(ratio);
                let root = crate::transforms::eval::eval(arena, root);
                let neg_root = arena.neg(root);
                return Some(vec![Solution { value: root }, Solution { value: neg_root }]);
            }
            // x = (-b ± sqrt(b² - 4ac)) / (2a)
            let two = arena.int(2);
            let four = arena.int(4);
            let b_sq = arena.pow(b, two);
            let four_ac = arena.mul(&[four, a, c]);
            let disc = arena.sub(b_sq, four_ac);
            let disc = crate::transforms::eval::eval(arena, disc);
            let disc = crate::simplify::ratsimp::ratsimp(arena, disc);
            if proved_zero_constant(arena, disc) {
                let neg_b = arena.neg(b);
                let two_a = arena.mul(&[two, a]);
                let root = arena.div(neg_b, two_a);
                let root = crate::transforms::eval::eval(arena, root);
                return Some(vec![Solution { value: root }]);
            }
            let sqrt_disc = arena.sqrt(disc);
            let neg_b = arena.neg(b);
            let two_a = arena.mul(&[two, a]);
            let num1 = arena.add(&[neg_b, sqrt_disc]);
            let num2 = arena.sub(neg_b, sqrt_disc);
            let x1 = arena.div(num1, two_a);
            let x2 = arena.div(num2, two_a);
            let x1 = crate::transforms::eval::eval(arena, x1);
            let x2 = crate::transforms::eval::eval(arena, x2);
            if x1 == x2 {
                Some(vec![Solution { value: x1 }])
            } else {
                Some(vec![Solution { value: x1 }, Solution { value: x2 }])
            }
        }
        _ => {
            // Binomial a·xⁿ + b = 0.
            let middle_zero = coeffs[1..degree]
                .iter()
                .all(|&c| arena.is_zero_structural(c));
            if !middle_zero {
                return None;
            }
            let a = coeffs[degree];
            let b = coeffs[0];
            let neg_b = arena.neg(b);
            let ratio = arena.div(neg_b, a);
            let ratio = crate::transforms::eval::eval(arena, ratio);
            Some(binomial_roots(arena, ratio, degree))
        }
    }
}

/// Is `c` a constant (no symbols) that is 0 without being written 0, by an
/// exact zero test ([`constant_zero_decision`]: the minimal polynomial, the
/// square-root tower, sums of arctangents; a value only zero to the working
/// precision is not a proof)?  `false` for a number, a structural 0 and a
/// constant with parameters.
///
/// [`constant_zero_decision`]: crate::transforms::eval::constant_zero_decision
fn proved_zero_constant(arena: &mut Arena, c: ExprId) -> bool {
    if arena.as_num(c).is_some() || !crate::base::walk::free_symbols(arena, c).is_empty() {
        return false;
    }
    crate::base::truth::Truth::from(crate::transforms::eval::constant_zero_decision(arena, c))
        .is_true()
}

/// All `n` complex solutions of `xⁿ = c`: `c^(1/n) · e^{2πik/n}` for
/// `k = 0..n`, with the roots of unity written as `cos + i·sin`.
fn binomial_roots(arena: &mut Arena, c: ExprId, n: usize) -> Vec<Solution> {
    if arena.is_zero_structural(c) {
        return vec![Solution { value: arena.zero }];
    }
    // For a negative real constant use |c|^(1/n)·e^{iπ(2k+1)/n} so that the
    // roots come out as explicit real/imaginary combinations instead of
    // an unevaluated principal root of a negative number.
    let negative_real = arena.as_num(c).is_some_and(|r| r.is_negative());
    let (radicand, angle_offset) = if negative_real {
        (arena.neg(c), 1i64)
    } else {
        (c, 0i64)
    };
    let inv_n = arena.rational(1, n as i64);
    let magnitude = arena.pow(radicand, inv_n);
    let magnitude = crate::transforms::eval::eval(arena, magnitude);
    let pi = arena.pi;
    let i_unit = arena.i_unit;
    let mut roots: Vec<Solution> = Vec::with_capacity(n);
    for k in 0..n {
        // angle = π·(2k + offset)/n
        let numer = 2 * k as i64 + angle_offset;
        let root = if numer == 0 {
            magnitude
        } else {
            let frac = arena.rational(numer, n as i64);
            let angle = arena.mul(&[frac, pi]);
            let cos_a = arena.cos(angle);
            let sin_a = arena.sin(angle);
            let i_sin = arena.mul(&[i_unit, sin_a]);
            let omega = arena.add(&[cos_a, i_sin]);
            let prod = arena.mul(&[magnitude, omega]);
            let prod = crate::transforms::eval::eval(arena, prod);
            let prod = crate::transforms::expand::expand(arena, prod);
            crate::transforms::eval::eval(arena, prod)
        };
        if !roots.iter().any(|r| r.value == root) {
            roots.push(Solution { value: root });
        }
    }
    roots
}

/// Binomial shortcut for rational polynomials `a·xⁿ + b` (n ≥ 3): returns
/// all `n` roots via roots of unity.  `None` if the polynomial has any
/// middle terms.
fn try_solve_binomial_rational(arena: &mut Arena, poly: &Poly) -> Option<Vec<Solution>> {
    let n = poly.degree()?;
    if n < 3 {
        return None;
    }
    for i in 1..n {
        if !poly.coeff(i).is_zero() {
            return None;
        }
    }
    let a = poly.coeff(n);
    let b = poly.coeff(0);
    if a.is_zero() {
        return None;
    }
    let ratio = -b / a;
    let c = arena.num_ratio(ratio.clone());
    Some(binomial_roots(arena, c, n))
}

// ═══════════════════════════════════════════════════════════════════════════
// Transcendental solving via inversion peeling
// ═══════════════════════════════════════════════════════════════════════════

/// `2π·i·n` for the integer parameter `n`: the period of `exp`.
fn two_pi_i_n(arena: &mut Arena, n: ExprId) -> ExprId {
    let two = arena.int(2);
    let pi = arena.pi;
    let i = arena.i_unit;
    arena.mul(&[two, pi, i, n])
}

/// Try to solve `expr = 0` by algebraic inversion.
/// Restructures as `f(x) = c` and inverts `f`.
///
/// No root ([`Reason::NoPreimage`]) when a range restriction proves there
/// is no real solution (e.g. `exp(x) = -1`), refused when the structure is
/// not invertible.
fn try_solve_by_inversion(arena: &mut Arena, expr: ExprId, var: ExprId, cx: &SolveCx) -> Solutions {
    if !expr_contains_var(arena, expr, var) {
        return Solutions::refused(Reason::NotFound);
    }
    let zero = arena.zero;
    solve_by_peeling(arena, expr, zero, var, cx)
}

/// Peel layers off `lhs = rhs` to isolate `var`.
///
/// `rhs` never contains `var`.  In a general solve ([`SolveCx::period`])
/// periodic inversions (sin, cos, tan) emit full solution families in terms
/// of the integer parameter `n`; otherwise only principal branches are
/// produced.  A range restriction is used when [`SolveCx::range_restricted`]
/// allows it, and gives no root ([`Reason::NoPreimage`]); a structure that
/// is not inverted is refused, and a periodic layer whose right-hand side
/// is already a family ([`period_in_use`]) refuses with
/// [`Reason::NestedPeriods`] unless it handles it.
fn solve_by_peeling(
    arena: &mut Arena,
    lhs: ExprId,
    rhs: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    let not_inverted = || Solutions::refused(Reason::NotFound);
    let no_preimage = || Solutions::none(Reason::NoPreimage);
    let period = cx.period();
    // Base case: lhs IS the variable
    if lhs == var {
        return Solutions::found(vec![rhs]);
    }

    let node = arena.node(lhs).clone();
    match node {
        // f(x) + c = rhs → f(x) = rhs - c ; several var-terms → polynomial fallback
        ExprNode::Add(ref children) => {
            let mut dep = Vec::new();
            let mut indep = Vec::new();
            for &child in children {
                if expr_contains_var(arena, child, var) {
                    dep.push(child);
                } else {
                    indep.push(child);
                }
            }
            if dep.len() == 1 {
                let new_rhs = if indep.is_empty() {
                    rhs
                } else {
                    let sum_indep = arena.add(&indep);
                    arena.sub(rhs, sum_indep)
                };
                return solve_by_peeling(arena, dep[0], new_rhs, var, cx);
            }
            // Multiple var-dependent terms: lhs - rhs = 0 may be polynomial.
            let diff = arena.sub(lhs, rhs);
            let diff = crate::transforms::eval::eval(arena, diff);
            if let Some(poly) = polybridge::expr_to_poly(arena, diff, var) {
                if poly.is_zero() || poly.is_constant() {
                    return not_inverted();
                }
                return Solutions::exactly(
                    values(solve_rational_poly(arena, var, &poly)),
                    Reason::NoPreimage,
                );
            }
            if let Some(sols) = try_solve_linear_symbolic(arena, diff, var) {
                return Solutions::exactly(values(sols), Reason::NoPreimage);
            }
            match try_solve_symbolic_poly(arena, diff, var) {
                Some(sols) => Solutions::exactly(values(sols), Reason::NoPreimage),
                None => not_inverted(),
            }
        }
        // c * f(x) = rhs → f(x) = rhs/c
        ExprNode::Mul(ref children) => {
            let mut dep = Vec::new();
            let mut indep = Vec::new();
            for &child in children {
                if expr_contains_var(arena, child, var) {
                    dep.push(child);
                } else {
                    indep.push(child);
                }
            }
            if dep.len() != 1 && is_rational_in(arena, lhs, var) {
                // A rational function (`(x − 3)/x = 0` under `√`, `(x + 1)(x − 1)
                // = 1` under `ln`; refused before 0.47), by the polynomial and
                // rational solvers directly (not `solve_raw`, which peels).
                let diff = arena.sub(lhs, rhs);
                let diff = crate::transforms::eval::eval(arena, diff);
                if let Some(poly) = polybridge::expr_to_poly(arena, diff, var) {
                    if poly.is_zero() || poly.is_constant() {
                        return not_inverted();
                    }
                    return Solutions::exactly(
                        values(solve_rational_poly(arena, var, &poly)),
                        Reason::NoPreimage,
                    );
                }
                if is_rational_function_of(arena, diff, var) {
                    let outcome = try_solve_rational(arena, diff, var, cx);
                    return if outcome.has_roots()
                        || outcome.is_none_proved()
                        || outcome.is_nested_periods()
                    {
                        outcome
                    } else {
                        not_inverted()
                    };
                }
                return match try_solve_linear_symbolic(arena, diff, var)
                    .filter(|s| !s.is_empty())
                    .or_else(|| try_solve_symbolic_poly(arena, diff, var))
                {
                    Some(sols) => Solutions::exactly(values(sols), Reason::NoPreimage),
                    None => not_inverted(),
                };
            }
            if dep.len() != 1 || indep.is_empty() {
                return not_inverted();
            }
            let coeff = arena.mul(&indep);
            let mut new_rhs = arena.div(rhs, coeff);
            // `i·x = 2πin + iπ` → `x = 2πn + π`, not `−(2πin + iπ)·i`.
            if crate::base::walk::contains(arena, coeff, arena.i_unit) {
                let e = crate::transforms::eval::eval(arena, new_rhs);
                let e = crate::transforms::expand::expand(arena, e);
                new_rhs = crate::transforms::eval::eval(arena, e);
            }
            solve_by_peeling(arena, dep[0], new_rhs, var, cx)
        }
        // exp(f(x)) = rhs → f(x) = ln(rhs) (+ 2πin for a non-real f)
        // Domain check: exp never vanishes; exp(f) > 0 for a real f, so
        // then rhs must be strictly positive.
        ExprNode::Exp(inner) => {
            if rhs == arena.zero {
                tracing::debug!("solve_by_peeling: exp domain error, rhs = 0");
                return no_preimage();
            }
            let real_inner = cx.real_for_real_var(arena, inner, var);
            if real_inner
                && let Some(c) = arena.as_num(rhs)
                && !c.is_positive()
                && cx.range_restricted(arena, inner, var)
            {
                tracing::debug!("solve_by_peeling: exp domain error, rhs <= 0");
                return no_preimage();
            }
            let ln_rhs = principal_ln(arena, rhs);
            // A non-real argument (`exp(i·x) = c`) reaches every branch of
            // the logarithm with real values of `var`: the general solution
            // carries the period `2πi`.
            if !real_inner && period_in_use(arena, rhs, cx) {
                return Solutions::refused(Reason::NestedPeriods);
            }
            let new_rhs = match period {
                Some(n) if !real_inner => {
                    let p = two_pi_i_n(arena, n);
                    arena.add(&[ln_rhs, p])
                }
                _ => ln_rhs,
            };
            solve_by_peeling(arena, inner, new_rhs, var, cx)
        }
        // ln(f(x)) = rhs → f(x) = exp(rhs)
        ExprNode::Ln(inner) => {
            let new_rhs = arena.exp(rhs);
            peel_principal_inverse(arena, PrincipalInverse::Ln, inner, rhs, new_rhs, var, cx)
        }
        // sin(f(x)) = rhs → f(x) ∈ {asin(rhs), π - asin(rhs)} (+ 2πn)
        ExprNode::Sin(inner) | ExprNode::Cos(inner) if period_in_use(arena, rhs, cx) => {
            // `sin f = cₙ` for a family `cₙ` (an outer periodic layer): over
            // the reals only the members with `|cₙ| ≤ 1` can be reached, each
            // giving families in a parameter of their own.  One refused
            // member refuses the whole.
            let Some(n) = period else {
                return not_inverted();
            };
            let members = if cx.real_for_real_var(arena, inner, var) {
                bounded_family_members(arena, rhs, n)
            } else {
                None
            };
            let Some(members) = members else {
                return Solutions::refused(Reason::NestedPeriods);
            };
            let mut out = no_preimage();
            for m in members {
                match solve_by_peeling(arena, lhs, m, var, cx) {
                    refused @ Solutions::Refused(_) => return refused,
                    member => out = out.union(member),
                }
            }
            out
        }
        ExprNode::Sin(inner) => {
            // Domain check: sin(x) = c has no real solutions when |c| > 1
            if let Some(c) = arena.as_num(rhs)
                && c.abs() > Ratio::one()
                && cx.range_restricted(arena, inner, var)
            {
                tracing::debug!("solve_by_peeling: sin domain error, |c| > 1");
                return no_preimage();
            }
            tracing::debug!("solve_by_peeling: inverting sin, two branches");
            let asin_rhs = arena.asin(rhs);
            let pi = arena.pi;
            let pi_minus_asin = arena.sub(pi, asin_rhs);
            let (b1, b2) = match period {
                Some(n) => {
                    let two = arena.int(2);
                    let two_pi_n = arena.mul(&[two, pi, n]);
                    (
                        arena.add(&[asin_rhs, two_pi_n]),
                        arena.add(&[pi_minus_asin, two_pi_n]),
                    )
                }
                None => (asin_rhs, pi_minus_asin),
            };
            peel_two_branches(arena, inner, b1, b2, var, cx)
        }
        // cos(f(x)) = rhs → f(x) ∈ {acos(rhs), -acos(rhs)} (+ 2πn)
        ExprNode::Cos(inner) => {
            // Domain check: cos(x) = c has no real solutions when |c| > 1
            if let Some(c) = arena.as_num(rhs)
                && c.abs() > Ratio::one()
                && cx.range_restricted(arena, inner, var)
            {
                tracing::debug!("solve_by_peeling: cos domain error, |c| > 1");
                return no_preimage();
            }
            tracing::debug!("solve_by_peeling: inverting cos, two branches");
            let acos_rhs = arena.acos(rhs);
            let neg_acos = arena.neg(acos_rhs);
            let (b1, b2) = match period {
                Some(n) => {
                    let two = arena.int(2);
                    let pi = arena.pi;
                    let two_pi_n = arena.mul(&[two, pi, n]);
                    (
                        arena.add(&[acos_rhs, two_pi_n]),
                        arena.add(&[neg_acos, two_pi_n]),
                    )
                }
                None => (acos_rhs, neg_acos),
            };
            peel_two_branches(arena, inner, b1, b2, var, cx)
        }
        // tan(f(x)) = rhs → f(x) = atan(rhs) (+ πn)
        ExprNode::Tan(inner) => {
            if period_in_use(arena, rhs, cx) {
                return Solutions::refused(Reason::NestedPeriods);
            }
            let atan_rhs = arena.atan(rhs);
            let new_rhs = match period {
                Some(n) => {
                    let pi = arena.pi;
                    let pi_n = arena.mul(&[pi, n]);
                    arena.add(&[atan_rhs, pi_n])
                }
                None => atan_rhs,
            };
            solve_by_peeling(arena, inner, new_rhs, var, cx)
        }
        // f(x)^n = rhs → f(x) = rhs^(1/n)
        // When n is a positive even integer, also consider f(x) = -(rhs^(1/n))
        //
        // a^f(x) = rhs → f(x) = ln(rhs) / ln(a)  (constant base, variable exponent)
        ExprNode::Pow(inner_base, inner_exp) => {
            if let Some(n) = arena.as_num(inner_exp) {
                let n = n.clone();
                if !n.is_zero() {
                    return match power_preimages(arena, rhs, &n, cx) {
                        None => not_inverted(),
                        Some(Preimages::Values(branches)) => {
                            peel_branches(arena, inner_base, &branches, var, cx)
                        }
                        Some(Preimages::Conditional { value, cond }) => {
                            peel_branches(arena, inner_base, &[value], var, cx).under(arena, cond)
                        }
                    };
                }
            }

            // f^c = 0 for an exponent free of `var`: `f = 0` exactly when
            // `re c > 0` (`0^c` is `0` there, undefined or `1` elsewhere, and
            // `f^c = e^(c·ln f) ≠ 0` for `f ≠ 0`).  Up to 0.52 the candidate
            // `0^(1/c)` was left unevaluated (`x^(1/erfi 3) = 0`: "cannot
            // decide whether 0^(1/erfi(3)) solves the equation").
            if arena.is_zero_structural(rhs)
                && expr_contains_var(arena, inner_base, var)
                && !expr_contains_var(arena, inner_exp, var)
            {
                return power_equals_zero(arena, inner_base, inner_exp, var, cx);
            }

            // a^f(x) = rhs where a is a constant (no var) and f(x) contains var.
            // Strategy: f(x) = ln(rhs) / ln(a).
            //
            // Integer shortcut: if a and rhs are positive integers and a^k == rhs
            // for some k, solve f(x) = k directly (gives exact answer like x = 3
            // instead of x = ln(8)/ln(2)).
            if !expr_contains_var(arena, inner_base, var)
                && expr_contains_var(arena, inner_exp, var)
            {
                tracing::debug!("solve_by_peeling: constant-base exponential a^f(x) = rhs");

                // a^f = exp(f·ln a) reaches every branch of the logarithm
                // with real values of `var` unless `f·ln a` is real (a > 0
                // and a real f): the general solution is then
                // f = (ln rhs + 2πin)/ln a.  Before 0.31 `(−1)^x = 1` gave
                // only `x = 0` from `solve_general` (the family is `2n`),
                // `(−1)^x = −1` only `1`.
                if let Some(n) = period {
                    let ln_base = arena.ln(inner_base);
                    let ln_base = crate::transforms::eval::eval(arena, ln_base);
                    let prod = arena.mul(&[inner_exp, ln_base]);
                    if !cx.real_for_real_var(arena, prod, var) {
                        if period_in_use(arena, rhs, cx) {
                            return Solutions::refused(Reason::NestedPeriods);
                        }
                        let ln_rhs = principal_ln(arena, rhs);
                        let p = two_pi_i_n(arena, n);
                        let num = arena.add(&[ln_rhs, p]);
                        let new_rhs = arena.div(num, ln_base);
                        let new_rhs = crate::transforms::expand::expand(arena, new_rhs);
                        let new_rhs = crate::transforms::eval::eval(arena, new_rhs);
                        return solve_by_peeling(arena, inner_exp, new_rhs, var, cx);
                    }
                }

                // Integer shortcut: try to find k such that base^k == rhs
                if let (Some(b), Some(r)) = (
                    arena.as_num(inner_base).cloned(),
                    arena.as_num(rhs).cloned(),
                ) && b.is_integer()
                    && r.is_integer()
                    && b > Ratio::one()
                    && r.is_positive()
                {
                    let b_int = b.to_integer();
                    let r_int = r.to_integer();
                    // Try small powers: b^1, b^2, ... up to b^64
                    let mut power = BigInt::one();
                    for k in 0u32..65 {
                        if power == r_int {
                            let k_expr = arena.int(k as i64);
                            tracing::debug!(
                                "solve_by_peeling: integer log shortcut, base^{k} = rhs"
                            );
                            return solve_by_peeling(arena, inner_exp, k_expr, var, cx);
                        }
                        if power > r_int {
                            break;
                        }
                        power *= &b_int;
                    }
                }

                // General case: f(x) = ln(rhs) / ln(base)
                let ln_base = arena.ln(inner_base);
                let ln_rhs = principal_ln(arena, rhs);
                let new_rhs = arena.div(ln_rhs, ln_base);
                return solve_by_peeling(arena, inner_exp, new_rhs, var, cx);
            }

            // f^c = rhs for a constant irrational real c (`x^π = 2`): the
            // principal f = rhs^(1/c), the one real solution of a real f
            // (`f^c = exp(c·ln f)` is real only for f ≥ 0; refused before
            // 0.47, SymPy: `[2**(1/pi)]`).  Other branches for a complex f.
            if expr_contains_var(arena, inner_base, var)
                && !expr_contains_var(arena, inner_exp, var)
                && crate::base::walk::free_symbols(arena, inner_exp).is_empty()
                && arena.as_num(inner_exp).is_none()
                && crate::transforms::realness::constant_realness_as_declared(
                    arena,
                    inner_exp,
                    ROOT_CHECK_DIGITS,
                    &mut crate::base::assumptions::AssumptionCache::new(),
                ) == Some(true)
            {
                if period.is_some() && !cx.real_for_real_var(arena, inner_base, var) {
                    return not_inverted();
                }
                let one = arena.one;
                let inv = arena.div(one, inner_exp);
                let new_rhs = arena.pow(rhs, inv);
                let new_rhs = crate::transforms::eval::eval(arena, new_rhs);
                return solve_by_peeling(arena, inner_base, new_rhs, var, cx);
            }

            not_inverted()
        }
        // Neg(-f(x)) = rhs → f(x) = -rhs
        ExprNode::Neg(inner) => {
            let new_rhs = arena.neg(rhs);
            solve_by_peeling(arena, inner, new_rhs, var, cx)
        }
        // ── Inverse trig peeling ──────────────────────────────────
        // asin(f(x)) = rhs → f(x) = sin(rhs)
        ExprNode::Asin(inner) => {
            let new_rhs = arena.sin(rhs);
            peel_principal_inverse(arena, PrincipalInverse::Asin, inner, rhs, new_rhs, var, cx)
        }
        // acos(f(x)) = rhs → f(x) = cos(rhs)
        ExprNode::Acos(inner) => {
            let new_rhs = arena.cos(rhs);
            peel_principal_inverse(arena, PrincipalInverse::Acos, inner, rhs, new_rhs, var, cx)
        }
        // atan(f(x)) = rhs → f(x) = tan(rhs)
        ExprNode::Atan(inner) => {
            let new_rhs = arena.tan(rhs);
            peel_principal_inverse(arena, PrincipalInverse::Atan, inner, rhs, new_rhs, var, cx)
        }
        // ── Inverse hyperbolic peeling ────────────────────────────
        // sinh(f(x)) = rhs → f(x) = asinh(rhs); for a non-real f the
        // general solution is asinh(rhs) + 2πin, iπ − asinh(rhs) + 2πin.
        ExprNode::Sinh(inner) => {
            let asinh_rhs = arena.asinh(rhs);
            match period {
                Some(_)
                    if !cx.real_for_real_var(arena, inner, var)
                        && period_in_use(arena, rhs, cx) =>
                {
                    Solutions::refused(Reason::NestedPeriods)
                }
                Some(n) if !cx.real_for_real_var(arena, inner, var) => {
                    let p = two_pi_i_n(arena, n);
                    let i_pi = arena.mul(&[arena.i_unit, arena.pi]);
                    let other = arena.sub(i_pi, asinh_rhs);
                    let b1 = arena.add(&[asinh_rhs, p]);
                    let b2 = arena.add(&[other, p]);
                    peel_two_branches(arena, inner, b1, b2, var, cx)
                }
                _ => solve_by_peeling(arena, inner, asinh_rhs, var, cx),
            }
        }
        // cosh(f(x)) = rhs → f(x) = ±acosh(rhs) (+ 2πin for a non-real f)
        // Domain check: cosh(f) >= 1 for a real f, so rhs must be >= 1.
        ExprNode::Cosh(inner) => {
            let real_inner = cx.real_for_real_var(arena, inner, var);
            if real_inner
                && let Some(c) = arena.as_num(rhs)
                && *c < Ratio::one()
                && cx.range_restricted(arena, inner, var)
            {
                tracing::debug!("solve_by_peeling: cosh domain error, rhs < 1");
                return no_preimage();
            }
            if !real_inner && period_in_use(arena, rhs, cx) {
                return Solutions::refused(Reason::NestedPeriods);
            }
            let acosh_rhs = arena.acosh(rhs);
            let neg_acosh = arena.neg(acosh_rhs);
            let (b1, b2) = match period {
                Some(n) if !real_inner => {
                    let p = two_pi_i_n(arena, n);
                    (arena.add(&[acosh_rhs, p]), arena.add(&[neg_acosh, p]))
                }
                _ => (acosh_rhs, neg_acosh),
            };
            peel_two_branches(arena, inner, b1, b2, var, cx)
        }
        // tanh(f(x)) = rhs → f(x) = atanh(rhs) (+ iπn for a non-real f)
        ExprNode::Tanh(inner) => {
            if !cx.real_for_real_var(arena, inner, var) && period_in_use(arena, rhs, cx) {
                return Solutions::refused(Reason::NestedPeriods);
            }
            let atanh_rhs = arena.atanh(rhs);
            let new_rhs = match period {
                Some(n) if !cx.real_for_real_var(arena, inner, var) => {
                    let i_pi_n = arena.mul(&[arena.i_unit, arena.pi, n]);
                    arena.add(&[atanh_rhs, i_pi_n])
                }
                _ => atanh_rhs,
            };
            solve_by_peeling(arena, inner, new_rhs, var, cx)
        }
        // asinh(f) = rhs → f = sinh(rhs), acosh(f) = rhs → f = cosh(rhs),
        // atanh(f) = rhs → f = tanh(rhs): valid when rhs lies in the range
        // of the principal branch, which the final check of every candidate
        // decides (`acosh x = −1` has no solution: `cosh(−1)` gives
        // `acosh(cosh 1) = 1`).  Without these the kinks of `|asinh x|`
        // were invisible to the definite integrator.
        ExprNode::Asinh(inner) => {
            let new_rhs = arena.sinh(rhs);
            peel_principal_inverse(arena, PrincipalInverse::Asinh, inner, rhs, new_rhs, var, cx)
        }
        ExprNode::Acosh(inner) => {
            let new_rhs = arena.cosh(rhs);
            peel_principal_inverse(arena, PrincipalInverse::Acosh, inner, rhs, new_rhs, var, cx)
        }
        ExprNode::Atanh(inner) => {
            let new_rhs = arena.tanh(rhs);
            peel_principal_inverse(arena, PrincipalInverse::Atanh, inner, rhs, new_rhs, var, cx)
        }
        // ── Abs peeling ───────────────────────────────────────────
        // ── Abs peeling ───────────────────────────────────────────────────────────
        // erf(f) = rhs → f = erfinv(rhs), erfc(f) = rhs → f = erfcinv(rhs),
        // the principal solutions (erf is entire and takes each value
        // infinitely often).  A rhs outside the range (−1, 1) of erf on the
        // reals, (0, 2) of erfc — a real one beyond the ends, or a non-real
        // one — is reached by a complex f only, where the inverse has no
        // principal value: no solution when f is real in the domain of the
        // unknown, refused otherwise (a constant rhs not decided either way
        // too).  Up to 0.48 a plain `x` was taken real here (alone among the
        // range restrictions), and `Ex::solve` answered `erf(x) = 2` with "no
        // solution", while `erf(0.8694… + 1.3406…i) = 2` (mpmath
        // `findroot`); up to 0.51 a non-real rhs was inverted, and `erf(2x +
        // 1/3) = −2i` gave the one root `−erfinv(2i)/2 − 1/6` of infinitely
        // many.  Refused before 0.47 (SymPy: `solve(erf(x) - y)` →
        // `[erfinv(y)]`).
        ExprNode::Erf(inner) | ExprNode::Erfc(inner) => {
            use crate::base::truth::Truth;
            let is_erf = matches!(arena.node(lhs), ExprNode::Erf(_));
            let (lo, hi) = if is_erf {
                (-Q::one(), Q::one())
            } else {
                (Q::zero(), Q::from_integer(BigInt::from(2)))
            };
            // a constant `rhs` by its certified sign against the ends (`ln 3`:
            // up to 0.50 the real solve of `exp(erf x) = 3` gave `erfinv(ln 3)`,
            // which is not real); a value with parameters is inverted
            let in_range = match arena.as_num(rhs) {
                Some(c) => Truth::from(*c > lo && *c < hi),
                None if crate::base::walk::free_symbols(arena, rhs).is_empty() => {
                    let (lo_id, hi_id) = (arena.num_ratio(lo.clone()), arena.num_ratio(hi.clone()));
                    let above = arena.sub(rhs, hi_id);
                    let below = arena.sub(rhs, lo_id);
                    let above = crate::poly::algebraic::sign_checked(arena, above);
                    let below = crate::poly::algebraic::sign_checked(arena, below);
                    if above == Some(-1) && below == Some(1) {
                        Truth::True
                    } else if matches!(above, Some(0 | 1))
                        || matches!(below, Some(0 | -1))
                        || part_signs(arena, rhs).nonreal()
                    {
                        Truth::False
                    } else {
                        Truth::Unknown
                    }
                }
                None => Truth::True,
            };
            // `erf` and `erfc` are entire: never infinite (`π^(erfc x − 1) =
            // 0`); a `nan` (`0^(erfc f) = 0`) proves nothing
            if has_infinite_term(arena, rhs) && !matches!(arena.node(rhs), ExprNode::NaN) {
                return no_preimage();
            }
            let real_inner = cx.range_restricted(arena, inner, var);
            match in_range {
                Truth::True => {}
                Truth::False if real_inner => return no_preimage(),
                Truth::False | Truth::Unknown => return not_inverted(),
            }
            if period.is_some() && !real_inner {
                // the other preimages of a complex argument are not known
                return not_inverted();
            }
            let f = if is_erf {
                crate::base::libfn::LibFn::ErfInv
            } else {
                crate::base::libfn::LibFn::ErfcInv
            };
            let new_rhs = arena.lib_apply(f, &[rhs]);
            let new_rhs = crate::transforms::eval::eval(arena, new_rhs);
            solve_by_peeling(arena, inner, new_rhs, var, cx)
        }
        // W(f) = rhs → f = rhs·e^rhs (the principal branch reaches rhs only
        // in its range, which the check of the candidate decides; refused
        // before 0.47, SymPy: `solve(LambertW(2*x) - y)` → `[y*exp(y)/2]`).
        //
        // A rhs with parameters, which that check cannot evaluate, is a value
        // of `W₀` when its assumptions show `rhs ≥ −1`; a real one otherwise
        // exactly under that condition (`W₀(y·e^y) = y` for real `y` iff `y ≥
        // −1`), and a complex one is refused (the range of `W₀` is a region
        // bounded by `−η·cot η + iη`).  Up to 0.52 `W(2x) = y` gave `[y·e^y/2]`
        // for every `y`, wrong for `y = −5` (SymPy 1.14 likewise).
        ExprNode::LambertW(inner) => {
            use crate::base::assumptions::Props;
            use crate::base::truth::Truth;
            let e = arena.exp(rhs);
            let new_rhs = arena.mul(&[rhs, e]);
            let new_rhs = crate::transforms::eval::eval(arena, new_rhs);
            let parametric = crate::base::walk::free_symbols(arena, rhs)
                .into_iter()
                .any(|s| Some(s) != period);
            if parametric && !cx.conditional_roots_as_candidates() {
                let minus_one = arena.neg_one;
                let one = arena.one;
                let shifted = arena.add(&[rhs, one]);
                let mut cache = cx.parameter_cache(arena, rhs);
                let in_range = Truth::from(cache.query(arena, shifted, Props::NONNEGATIVE));
                let real = Truth::from(cache.query(arena, rhs, Props::REAL)).is_true();
                let with_period =
                    period.is_some_and(|p| crate::base::walk::contains(arena, rhs, p));
                match in_range {
                    Truth::True => {}
                    Truth::False if real => return no_preimage(),
                    _ if real && !with_period => {
                        let cond = arena.ge(rhs, minus_one);
                        return solve_by_peeling(arena, inner, new_rhs, var, cx).under(arena, cond);
                    }
                    _ => return not_inverted(),
                }
            }
            solve_by_peeling(arena, inner, new_rhs, var, cx)
        }
        // W_k(f) = rhs → f = rhs·e^rhs for the other branches too, when rhs is
        // a value of W_k — `W_k(rhs·e^rhs) = rhs`, which depends on rhs alone
        // and is decided here (the candidate check skips an equation with
        // parameters): exactly, else no preimage when certified otherwise
        // (`W₁(x) = 0`: `W₁(0) = −∞`; `W₋₁(x) = −asin(1/π)`: `W₋₁` is real
        // only ≤ −1), else refused (`W₋₁(x) = y`) unless roots may hold for
        // some parameter values only ([`SolveCx::conditional_roots_as_candidates`]).
        // `W₋₁(x) = −2` → `−2e⁻²`; refused before 0.52, so an initial value
        // on `W₋₁` could not be fitted (SymPy 1.14 raises NotImplementedError).
        ExprNode::Lib(crate::base::libfn::LibFn::LambertW, ref args)
            if args.len() == 2 && !expr_contains_var(arena, args[1], var) =>
        {
            let (inner, k) = (args[0], args[1]);
            let e = arena.exp(rhs);
            let new_rhs = arena.mul(&[rhs, e]);
            let new_rhs = crate::transforms::eval::eval(arena, new_rhs);
            let back = arena.lambertw_branch(new_rhs, k);
            let gap = arena.sub(back, rhs);
            let gap = crate::transforms::eval::eval(arena, gap);
            if !arena.is_zero_structural(gap) {
                // `gap` is free of `var`: the substitution is the identity.
                if certainly_not_a_root(arena, gap, var, var) {
                    return Solutions::none(Reason::said(
                        "the right-hand side is not a value of this branch of Lambert W",
                    ));
                }
                if !cx.conditional_roots_as_candidates() {
                    return not_inverted();
                }
            }
            solve_by_peeling(arena, inner, new_rhs, var, cx)
        }
        // atan2(f, c) = rhs for a constant c > 0 is atan(f/c) = rhs: f =
        // c·tan(rhs) (in the range when |rhs| < π/2, checked; refused before
        // 0.47, SymPy: `solve(atan2(x, 2) - pi/3)` → `[2*sqrt(3)]`).
        ExprNode::Atan2(f, c)
            if !expr_contains_var(arena, c, var)
                && crate::base::walk::free_symbols(arena, c).is_empty()
                && crate::poly::algebraic::sign_checked(arena, c) == Some(1) =>
        {
            let t = arena.tan(rhs);
            let new_rhs = arena.mul(&[c, t]);
            let new_rhs = crate::transforms::eval::eval(arena, new_rhs);
            solve_by_peeling(arena, f, new_rhs, var, cx)
        }
        // erfinv(f) = rhs → f = erf(rhs), erfcinv(f) = rhs → f = erfc(rhs)
        // (the inverses are injective; candidates are checked).
        ExprNode::Lib(
            f @ (crate::base::libfn::LibFn::ErfInv | crate::base::libfn::LibFn::ErfcInv),
            ref args,
        ) if args.len() == 1 => {
            let inner = args[0];
            let new_rhs = if f == crate::base::libfn::LibFn::ErfInv {
                arena.intern(ExprNode::Erf(rhs))
            } else {
                arena.intern(ExprNode::Erfc(rhs))
            };
            let new_rhs = crate::transforms::eval::eval(arena, new_rhs);
            solve_by_peeling(arena, inner, new_rhs, var, cx)
        }
        // |f(x)| = rhs → f(x) = rhs OR f(x) = -rhs (when rhs ≥ 0)
        ExprNode::Abs(inner) => {
            // |f(x)| = negative has no solutions
            if let Some(r) = arena.as_num(rhs)
                && r.is_negative()
            {
                return no_preimage();
            }
            // |f| = 0 exactly when f = 0, for any complex `f`.
            if arena.is_zero_structural(rhs) {
                return solve_by_peeling(arena, inner, rhs, var, cx);
            }
            // `f = ±rhs` are all the solutions only for a real `f`: a
            // complex `f` has modulus `rhs` on a whole circle.  Before 0.31
            // `|i·x| = 2` gave `x = ±2i` and missed the real `x = ±2`;
            // up to 0.48 a plain `x` was taken real here under `Ex::solve`
            // (`|x| = 1` gave `[1, −1]`; see `SolveCx::real`).
            if !cx.real(arena, inner, var) {
                // … but none when `rhs` is a constant that is not a
                // nonnegative real (`|x|² = −4` peels to `|x| = ±2i`).
                let not_a_modulus = crate::base::walk::free_symbols(arena, rhs).is_empty()
                    && (crate::transforms::realness::constant_realness_as_declared(
                        arena,
                        rhs,
                        ROOT_CHECK_DIGITS,
                        &mut crate::base::assumptions::AssumptionCache::new(),
                    ) == Some(false)
                        || crate::poly::algebraic::sign_checked(arena, rhs) == Some(-1));
                if not_a_modulus {
                    return no_preimage();
                }
                // A real unknown: `|f|² = (Re f)² + (Im f)² = rhs²`, an
                // equivalent real equation for `rhs ≥ 0` (`|x + i| = 2`:
                // `x² + 1 = 4`, `x = ±√3`; up to 0.49 refused).
                if !cx.unknown_may_be_nonreal(arena, var)
                    && crate::base::walk::free_symbols(arena, rhs).is_empty()
                    && crate::poly::algebraic::sign_checked(arena, rhs).is_some_and(|s| s >= 0)
                    && let Some(squared) = modulus_squared(arena, inner, var)
                {
                    let two = arena.int(2);
                    let rhs2 = arena.pow(rhs, two);
                    let eq = arena.sub(squared, rhs2);
                    let eq = crate::transforms::expand::expand(arena, eq);
                    let eq = crate::transforms::eval::eval(arena, eq);
                    if expr_contains_var(arena, eq, var) {
                        return solve_raw(arena, eq, var, cx);
                    }
                }
                return not_inverted();
            }
            let neg_rhs = arena.neg(rhs);
            peel_two_branches(arena, inner, rhs, neg_rhs, var, cx)
        }
        _ => not_inverted(),
    }
}

/// `|f|² = (Re f)² + (Im f)²` for a real `var`, expanded (`|x + i|² =
/// x² + 1`, `|a·x|² = (re(a)² + im(a)²)·x²`); `None` when a part of `f`
/// is not found free of `re`, `im`, `arg`, `conjugate` of `var`.
fn modulus_squared(arena: &mut Arena, f: ExprId, var: ExprId) -> Option<ExprId> {
    use crate::base::assumptions::{Assumptions, Props};
    // the parts for a real variable: a real stand-in for `var`
    let t = crate::transforms::subs::fresh_symbol(arena, "__abs_real", &[f]);
    if let ExprNode::Symbol(sid) = *arena.node(t) {
        let mut real = Assumptions::default();
        real.assert_true(Props::REAL);
        arena.set_symbol_assumptions(sid, real);
    }
    let f_t = crate::transforms::subs::subs(arena, f, var, t);
    let f_t = crate::transforms::expand::expand(arena, f_t);
    let (re, im) = crate::base::complex::as_real_imag(arena, f_t);
    let mut parts = Vec::with_capacity(2);
    for p in [re, im] {
        let p = crate::transforms::eval::eval(arena, p);
        let opaque = crate::base::walk::post_order_ids(arena, p)
            .into_iter()
            .any(|id| match *arena.node(id) {
                ExprNode::Re(u) | ExprNode::Im(u) | ExprNode::Arg(u) | ExprNode::Conjugate(u) => {
                    crate::base::walk::contains(arena, u, t)
                }
                _ => false,
            });
        if opaque {
            return None;
        }
        parts.push(p);
    }
    let two = arena.int(2);
    let re2 = arena.pow(parts[0], two);
    let im2 = arena.pow(parts[1], two);
    let sum = arena.add(&[re2, im2]);
    let sum = crate::transforms::subs::subs(arena, sum, t, var);
    let sum = crate::transforms::expand::expand(arena, sum);
    Some(crate::transforms::eval::eval(arena, sum))
}

/// Does `rhs` hold the integer parameter of a general solve (a family from
/// an outer periodic layer)?
fn period_in_use(arena: &Arena, rhs: ExprId, cx: &SolveCx) -> bool {
    cx.period()
        .is_some_and(|n| crate::base::walk::contains(arena, rhs, n))
}

/// Most members of a family listed by [`bounded_family_members`].
const MAX_BOUNDED_MEMBERS: usize = 64;

/// The members `c(k)` of the family `c(n) = a + b·n` (constants `a`, `b`
/// proved real, `b ≠ 0`) that may lie in `[−1, 1]`, the range of `sin` and
/// `cos` on the reals: the integers `k` with `|a + b·k| ≤ 1`, and one more
/// on either side (a member just outside is answered with no real solution
/// by the next layer), from [`lattice_indices`].  `None` when `c` is not of
/// that form, the range is not decided or there are too many.
///
/// Up to 0.53 the range came from `f64` values of `a` and `b` (an imaginary
/// part below `10⁻¹²` read as 0) and was cast with `as i64`: for `tan(sin x
/// − 10²⁰) = 1` (`sin x = 10²⁰ + π/4 + kπ`, the member `k ≈ −3.18·10¹⁹` is
/// `0.0840…`) the index saturated, no member was listed, and a real `x` had
/// "no solution".
fn bounded_family_members(arena: &mut Arena, c: ExprId, n: ExprId) -> Option<Vec<ExprId>> {
    use crate::base::truth::SignSet;
    let expanded = crate::transforms::expand::expand(arena, c);
    let coeffs = symbolic_poly_coeffs(arena, expanded, n)?;
    let [a, b] = coeffs[..] else {
        return None;
    };
    if !crate::base::walk::free_symbols(arena, a).is_empty()
        || !crate::base::walk::free_symbols(arena, b).is_empty()
    {
        return None;
    }
    for part in [a, b] {
        if crate::transforms::realness::imaginary_part_sign(arena, part) != SignSet::ZERO {
            return None;
        }
    }
    let (lo, hi) = (arena.neg_one, arena.one);
    let indices = lattice_indices(arena, a, b, lo, hi, MAX_BOUNDED_MEMBERS)?;
    let mut out = Vec::with_capacity(indices.len());
    for k in indices {
        let kk = arena.big_int(k);
        let m = crate::transforms::subs::subs(arena, c, n, kk);
        out.push(crate::transforms::eval::eval(arena, m));
    }
    Some(out)
}

/// The integers `k` from `⌊min(q_lo, q_hi)⌋ − 1` to `⌈max(q_lo, q_hi)⌉ + 1`,
/// `q_end = (end − offset)/step`: every index of the lattice `offset +
/// step·k` (real constants, `step ≠ 0`) whose member lies between the real
/// constants `lo` and `hi`, and one more on either side.  The floors and
/// ceilings are exact (`eval`: of a rational, or the integers a certified
/// value lies strictly between), so the range is right however far the
/// window is from the lattice's origin.  `None` when one of them is not
/// decided or the range holds more than `max` integers.
///
/// Up to 0.53 such ranges were computed from 16-digit `f64` values and cast
/// with `as i64`: an index beyond `2⁵³` lost its last digits and one beyond
/// `2⁶³` saturated, and the members in the window were missed
/// (`solve_as_set(sin(x − 10²⁰)) ∩ [0, 10]` was empty).
pub(crate) fn lattice_indices(
    arena: &mut Arena,
    offset: ExprId,
    step: ExprId,
    lo: ExprId,
    hi: ExprId,
    max: usize,
) -> Option<Vec<BigInt>> {
    let mut quotients = [lo, hi];
    for q in &mut quotients {
        let d = arena.sub(*q, offset);
        *q = arena.div(d, step);
    }
    integers_around(arena, quotients, max)
}

/// The integers from `⌊min q⌋ − 1` to `⌈max q⌉ + 1` over the real constants
/// `q`, the floors and ceilings exact (see [`lattice_indices`]).  A
/// quotient within its error of an integer (`⌊q⌋` open: `π²/π² − 1` not
/// simplified, an index at a window's end) is bracketed by its certified
/// 20-digit value, widened by one more on either side.  `None` when a
/// quotient has no value or there are more than `max + 1` integers.
pub(crate) fn integers_around(
    arena: &mut Arena,
    quotients: [ExprId; 2],
    max: usize,
) -> Option<Vec<BigInt>> {
    let integer = |arena: &Arena, id: ExprId| {
        arena
            .as_num(id)
            .filter(|q| q.is_integer())
            .map(|q| q.to_integer())
    };
    let mut floors: Vec<BigInt> = Vec::with_capacity(2);
    let mut ceilings: Vec<BigInt> = Vec::with_capacity(2);
    for q in quotients {
        let q = crate::transforms::eval::eval(arena, q);
        let f = arena.floor(q);
        let f = crate::transforms::eval::eval(arena, f);
        let c = arena.ceiling(q);
        let c = crate::transforms::eval::eval(arena, c);
        match (integer(arena, f), integer(arena, c)) {
            (Some(f), Some(c)) => {
                floors.push(f);
                ceilings.push(c);
            }
            _ => {
                // `eval` leaves the floor open: the certified 20-digit value
                // places `q` within one of an integer (far below 2⁴⁰, its
                // error is far below 1), and the range is widened by one more
                let v = crate::transforms::evalf::evalf(arena, q, 20)
                    .ok()
                    .and_then(|s| s.trim().parse::<f64>().ok())
                    .filter(|v| v.abs() < 1_099_511_627_776.0)?;
                let k = crate::base::cast::f64_to_i64(v.floor())?;
                floors.push(BigInt::from(k) - BigInt::one());
                ceilings.push(BigInt::from(k) + BigInt::from(2));
            }
        }
    }
    let first = floors.into_iter().min()? - BigInt::one();
    let last = ceilings.into_iter().max()? + BigInt::one();
    let count = num_traits::ToPrimitive::to_usize(&(&last - &first)).filter(|&c| c <= max)?;
    Some((0..=count).map(|i| &first + BigInt::from(i)).collect())
}

/// Largest `|p|` for which `f^(p/q) = c` with a numeric `c` is inverted
/// through all `|p|` roots of `u^p = c` (beyond it, the principal
/// candidate only).
const MAX_POWER_PREIMAGES: i64 = 24;

/// The values `f` with `f^n = c` (`n ≠ 0` rational) that peeling continues
/// with; `Values(vec![])` when there is none (`f^(−k) = 0`), `None` when
/// refused.
///
/// * Integer `n`: every root of `f^|n| = c^(sign n)` (`binomial_roots`), as
///   the polynomial route returns all roots of `xⁿ = c`.  Before 0.30 only
///   `c^(1/n)` (and its negative for a positive even `n`) came back:
///   `x^(−2) = 3` lost `−1/√3`, `(x²)^(−3) = 1/2` lost `−2^(1/6)`.
/// * `n = p/q` in lowest terms, `q > 1`: `f^(p/q) = (f^(1/q))^p` with the
///   principal `u = f^(1/q)` in the sector `−π/q < arg u ≤ π/q`, so the
///   candidates are `u^q` for the roots `u` of `u^p = c`; those outside the
///   sector are spurious (`√x = −2` has none, `x^(1/3) = −3/4` neither)
///   and are dropped by the check of every candidate against the equation
///   ([`certainly_not_a_root`]).  For a `c` with free symbols that check
///   cannot run, and only the principal candidate `c^(q/p)` can be a root
///   ([`principal_power_preimage`] decides whether it is one): it is kept
///   when it is, there is no preimage when `c` is nonzero and outside the
///   range of `f^(p/q)` (`√x = −√y`, `y > 0`), and otherwise the public
///   solvers get it under its exact condition (below) while the crate's
///   root searches keep the candidate
///   ([`SolveCx::conditional_roots_as_candidates`]).  Before 0.51 it was
///   always kept: `√(x + ∛x) + (x − y)^(1/5) = 0` gave `y = x + (x +
///   ∛x)^(5/2)`, a root nowhere but at `x + ∛x = 0` (`−√w` is never a
///   principal fifth root), as in SymPy 1.14.
///
/// In a general solve a `c` whose only symbol is the integer parameter
/// stands for constants, each member of the family of candidates checked
/// by [`check_family`] (`e^√x = −1`: `√x = (2n + 1)πi`, `x = −(2n +
/// 1)²π²`, all roots although half the `√x` are not principal).
///
/// When the sector test with parameters is undecided, a public solver
/// gets the principal candidate under its exact condition on the
/// parameters ([`Preimages::Conditional`], [`principal_power_condition`]):
/// `√x = y` is `x = y²` when `y ≥ 0` for a real `y`.  Up to 0.52 it
/// refused (since 0.51), and a `c` with the integer parameter of a general
/// solve still is (the condition would depend on the member).
fn power_preimages(arena: &mut Arena, c: ExprId, n: &Q, cx: &SolveCx) -> Option<Preimages> {
    let p = n.numer().clone();
    let q = n.denom().clone();
    let c_zero = arena.is_zero_structural(c);
    if c_zero {
        // f^n = 0: f = 0 for n > 0, nothing for n < 0.
        return Some(Preimages::Values(if n.is_positive() {
            vec![arena.zero]
        } else {
            vec![]
        }));
    }
    let symbolic = !crate::base::walk::free_symbols(arena, c).is_empty();
    let p_abs: i64 = num_traits::ToPrimitive::to_i64(&p.abs()).unwrap_or(i64::MAX);
    if q.is_one() || (!symbolic && p_abs <= MAX_POWER_PREIMAGES) {
        // u^|p| = c^(sign p), then f = u^q.
        let base = if p.is_negative() {
            let m1 = arena.neg_one;
            let inv = arena.pow(c, m1);
            crate::transforms::eval::eval(arena, inv)
        } else {
            c
        };
        if p_abs > MAX_INT_POWER_ROOTS {
            return None;
        }
        let roots = binomial_roots(arena, base, p_abs as usize);
        // The `k`-th root of `u^|p| = c^(sign p)` has argument
        // `π·(t + 2k)/|p|` with `t·π = arg c^(sign p)` (`binomial_roots`
        // multiplies the principal root by `e^(2πik/|p|)`, and writes a
        // negative real `c` as `|c|·e^(iπ)`).  When `t` is known exactly the
        // sector test is exact; otherwise the check of every candidate
        // against the equation decides.
        let exact_arg = if q.is_one() {
            None
        } else {
            gaussian_rational_arg(arena, c).map(|t| {
                if p.is_negative() && t != Ratio::one() {
                    -t
                } else {
                    t
                }
            })
        };
        // `binomial_roots` lists one root per `k = 0..|p|` (distinct for
        // `c ≠ 0`), so the position of a root is its `k`.
        let aligned = roots.len() as i64 == p_abs;
        let q_id = arena.big_int(q.clone());
        let mut out = Vec::with_capacity(roots.len());
        for (k, r) in roots.into_iter().enumerate() {
            if let Some(t) = &exact_arg
                && aligned
                && !in_principal_sector(t, k, p_abs, &q)
            {
                continue;
            }
            let f = arena.pow(r.value, q_id);
            let f = crate::transforms::expand::expand(arena, f);
            let f = crate::transforms::eval::eval(arena, f);
            if !out.contains(&f) {
                out.push(f);
            }
        }
        return Some(Preimages::Values(out));
    }
    // Principal candidate only: f = c^(1/n).
    let parametric = crate::base::walk::free_symbols(arena, c)
        .into_iter()
        .any(|s| Some(s) != cx.period());
    let inv_n = Ratio::one() / n.clone();
    let inv_n_id = arena.num_ratio(inv_n);
    if parametric {
        match principal_power_preimage(arena, c, n) {
            crate::base::truth::Truth::True => {}
            crate::base::truth::Truth::False => return Some(Preimages::Values(Vec::new())),
            crate::base::truth::Truth::Unknown if cx.conditional_roots_as_candidates() => {}
            crate::base::truth::Truth::Unknown => {
                if cx
                    .period()
                    .is_some_and(|p| crate::base::walk::contains(arena, c, p))
                {
                    return None;
                }
                let cond = principal_power_condition(arena, c, n)?;
                let value = arena.pow(c, inv_n_id);
                return Some(Preimages::Conditional { value, cond });
            }
        }
    }
    Some(Preimages::Values(vec![arena.pow(c, inv_n_id)]))
}

/// A principal inverse function peeled by its forward function, `inv(f) =
/// c → f = fwd(c)` ([`peel_principal_inverse`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrincipalInverse {
    Ln,
    Asin,
    Acos,
    Atan,
    Asinh,
    Acosh,
    Atanh,
}

/// `inv(inner) = rhs` peeled as `inner = new_rhs` (`= fwd(rhs)`), which
/// holds exactly when `rhs` is a value of the principal `inv`.  For a
/// constant `rhs` the check of every candidate decides; one with
/// parameters, which that check cannot evaluate, is decided by its
/// assumptions where they suffice, and otherwise a public solver gets the
/// root under the exact condition ([`Solutions::under`]): `ln f = y` needs
/// `−π < im y ≤ π` (always for a real `y`), and for a real `y` `asin`
/// needs `|y| ≤ π/2`, `acos` `0 ≤ y ≤ π`, `atan` `|y| < π/2`, `acosh` `y ≥
/// 0`, `asinh` and `atanh` nothing; a complex `y` is refused for the
/// inverse trigonometric and hyperbolic functions (their ranges have
/// boundaries that depend on the branch-cut conventions).  Up to 0.52 the
/// root was returned for every value: `ln x = y` gave `[e^y]` (wrong for
/// `y = 4i`: `ln(e^{4i}) = (4 − 2π)i`), `asin x = y` for a real `y` gave
/// `[sin y]` (wrong for `y = 4`), as SymPy 1.14 does.
#[allow(clippy::too_many_arguments)]
fn peel_principal_inverse(
    arena: &mut Arena,
    inverse: PrincipalInverse,
    inner: ExprId,
    rhs: ExprId,
    new_rhs: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    use crate::base::assumptions::Props;
    let parametric = crate::base::walk::free_symbols(arena, rhs)
        .into_iter()
        .any(|s| Some(s) != cx.period());
    if !parametric || cx.conditional_roots_as_candidates() {
        return solve_by_peeling(arena, inner, new_rhs, var, cx);
    }
    // (plain parameters are real under `Params::Real`: `reduce_inequalities`)
    let mut cache = cx.parameter_cache(arena, rhs);
    let real = crate::base::truth::Truth::from(cache.query(arena, rhs, Props::REAL)).is_true();
    let (subject, bounds): (ExprId, Vec<RangeBound>) = if real {
        let pi = arena.pi;
        let half = arena.rational(1, 2);
        let half_pi = arena.mul(&[half, pi]);
        let minus_half_pi = arena.neg(half_pi);
        let zero = arena.zero;
        let b = |value: ExprId, lower: bool, strict: bool| RangeBound {
            value,
            lower,
            strict,
        };
        let bounds = match inverse {
            PrincipalInverse::Ln | PrincipalInverse::Asinh | PrincipalInverse::Atanh => Vec::new(),
            PrincipalInverse::Asin => {
                vec![b(minus_half_pi, true, false), b(half_pi, false, false)]
            }
            PrincipalInverse::Acos => vec![b(zero, true, false), b(pi, false, false)],
            PrincipalInverse::Atan => vec![b(minus_half_pi, true, true), b(half_pi, false, true)],
            PrincipalInverse::Acosh => vec![b(zero, true, false)],
        };
        (rhs, bounds)
    } else if inverse == PrincipalInverse::Ln {
        let im = arena.im(rhs);
        let im = crate::transforms::eval::eval(arena, im);
        // a principal argument is in `(−π, π]` (`ln(log(−3x, y)) = 0`:
        // `rhs = ln y`, `im = arg y`)
        if matches!(arena.node(im), ExprNode::Arg(_)) {
            return solve_by_peeling(arena, inner, new_rhs, var, cx);
        }
        let pi = arena.pi;
        let minus_pi = arena.neg(pi);
        let bounds = vec![
            RangeBound {
                value: minus_pi,
                lower: true,
                strict: true,
            },
            RangeBound {
                value: pi,
                lower: false,
                strict: false,
            },
        ];
        (im, bounds)
    } else {
        return Solutions::refused(Reason::NotFound);
    };
    // the bounds the assumptions do not decide
    let mut open: Vec<ExprId> = Vec::new();
    for bound in bounds {
        let (hi, lo) = if bound.lower {
            (subject, bound.value)
        } else {
            (bound.value, subject)
        };
        let gap = arena.sub(hi, lo);
        let gap = crate::transforms::eval::eval(arena, gap);
        let prop = if bound.strict {
            Props::POSITIVE
        } else {
            Props::NONNEGATIVE
        };
        match crate::base::truth::Truth::from(cache.query(arena, gap, prop)) {
            crate::base::truth::Truth::True => {}
            crate::base::truth::Truth::False => return Solutions::none(Reason::NoPreimage),
            crate::base::truth::Truth::Unknown => open.push(if bound.strict {
                arena.gt(hi, lo)
            } else {
                arena.ge(hi, lo)
            }),
        }
    }
    let peeled = solve_by_peeling(arena, inner, new_rhs, var, cx);
    if open.is_empty() {
        return peeled;
    }
    if cx
        .period()
        .is_some_and(|p| crate::base::walk::contains(arena, rhs, p))
    {
        return Solutions::refused(Reason::NotFound);
    }
    let cond = arena.and(&open);
    peeled.under(arena, cond)
}

/// One end of the range of a principal inverse ([`peel_principal_inverse`]).
struct RangeBound {
    /// The end.
    value: ExprId,
    /// A lower end (the value must be above it).
    lower: bool,
    /// The end itself is excluded.
    strict: bool,
}

/// The roots of `base^c = 0` for an exponent `c` free of `var` (not a
/// number): those of `base = 0` when `re c > 0`, none when `re c ≤ 0` —
/// decided by a certified sign for a constant `c`, by the assumptions of
/// its symbols otherwise, and else, in a public solver, under the
/// condition `re c > 0` (`c > 0` for a real `c`; [`Solutions::under`]).
fn power_equals_zero(
    arena: &mut Arena,
    base: ExprId,
    c: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    use crate::base::assumptions::{AssumptionCache, Props};
    use crate::base::truth::Truth;
    let zero = arena.zero;
    let re = arena.re(c);
    let re = crate::transforms::eval::eval(arena, re);
    let constant = crate::base::walk::free_symbols(arena, c).is_empty();
    let positive = if constant {
        part_signs(arena, re).re.is_positive()
    } else {
        Truth::from(AssumptionCache::new().query(arena, re, Props::POSITIVE))
    };
    match positive {
        Truth::True => solve_by_peeling(arena, base, zero, var, cx),
        Truth::False => Solutions::none(Reason::NoPreimage),
        // (the crate's root searches keep refusing here: the integrator
        // would take `0` as a breakpoint of `x·δ(x^y)` for every `y`)
        Truth::Unknown
            if !constant
                && !cx.conditional_roots_as_candidates()
                && !cx
                    .period()
                    .is_some_and(|p| crate::base::walk::contains(arena, c, p)) =>
        {
            let at_zero = solve_by_peeling(arena, base, zero, var, cx);
            let real = Truth::from(AssumptionCache::new().query(arena, c, Props::REAL)).is_true();
            let cond = arena.gt(if real { c } else { re }, zero);
            at_zero.under(arena, cond)
        }
        Truth::Unknown => Solutions::refused(Reason::NotFound),
    }
}

/// The values `f` with `f^n = c` ([`power_preimages`]).
enum Preimages {
    /// Exactly these (none: no preimage).
    Values(Vec<ExprId>),
    /// `value` when `cond`, a condition on the parameters, holds, and none
    /// otherwise.
    Conditional {
        /// The principal candidate `c^(1/n)`.
        value: ExprId,
        /// The exact condition ([`principal_power_condition`]).
        cond: ExprId,
    },
}

/// The exact condition on the symbols of `c` under which the principal
/// candidate `c^(q/p)` solves `f^(p/q) = c` (`n = p/q` in lowest terms,
/// `|p| < q`; see [`principal_power_preimage`]): `arg c` in the range of
/// `f^(p/q)` — `(−|p|π/q, |p|π/q]` for `p > 0`, with `c = 0` (whose
/// preimage is `0`), `[−|p|π/q, |p|π/q)` without it for `p < 0`.  Written
/// `c ≥ 0` (`c > 0` for `p < 0`) for a `c` real by its symbols'
/// assumptions, `re c > 0 ∨ (re c = 0 ∧ im c ≥ 0)` (`im c < 0` for `p < 0`)
/// for `|p|/q = 1/2`, and with `arg c` otherwise.  `None` for `|p| ≥ q`
/// (every `c` has its preimage).
fn principal_power_condition(arena: &mut Arena, c: ExprId, n: &Q) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    let (p, q) = (n.numer().abs(), n.denom().clone());
    if p >= q {
        return None;
    }
    let positive = n.is_positive();
    let zero = arena.zero;
    let real = crate::base::truth::Truth::from(AssumptionCache::new().query(arena, c, Props::REAL))
        .is_true();
    if real {
        return Some(if positive {
            arena.ge(c, zero)
        } else {
            arena.gt(c, zero)
        });
    }
    let b = Ratio::new(p, q);
    if b == Ratio::new(BigInt::one(), BigInt::from(2)) {
        let re = arena.re(c);
        let re = crate::transforms::eval::eval(arena, re);
        let im = arena.im(c);
        let im = crate::transforms::eval::eval(arena, im);
        let right = arena.gt(re, zero);
        let on_axis = arena.eq_(re, zero);
        let side = if positive {
            arena.ge(im, zero)
        } else {
            arena.gt(zero, im)
        };
        let edge = arena.and(&[on_axis, side]);
        return Some(arena.or(&[right, edge]));
    }
    let arg = arena.arg(c);
    let arg = crate::transforms::eval::eval(arena, arg);
    let b_id = arena.num_ratio(b);
    let pi = arena.pi;
    let hi = arena.mul(&[b_id, pi]);
    let hi = crate::transforms::eval::eval(arena, hi);
    let lo = arena.neg(hi);
    let lo = crate::transforms::eval::eval(arena, lo);
    Some(if positive {
        let above = arena.gt(arg, lo);
        let below = arena.ge(hi, arg);
        let sector = arena.and(&[above, below]);
        let at_zero = arena.eq_(c, zero);
        arena.or(&[sector, at_zero])
    } else {
        let above = arena.ge(arg, lo);
        let below = arena.gt(hi, arg);
        let nonzero = arena.ne_(c, zero);
        arena.and(&[above, below, nonzero])
    })
}

/// Does the principal candidate `F = c^(q/p)` solve `f^(p/q) = c` (`n =
/// p/q` in lowest terms, `q > 1`) for every value of the symbols in `c`?
///
/// `F^(p/q) = c` exactly when `θ·q/p` (`θ = arg c ∈ (−π, π]`) needs no
/// reduction into `(−π, π]`: always for `|p| > q`, and for `|p| < q` when
/// `θ ∈ (−|p|π/q, |p|π/q]` (`p > 0`) or `[−|p|π/q, |p|π/q)` (`p < 0`) — the
/// range of `f^(p/q)` — or `c = 0`.  No other preimage exists then (a root
/// `u` of `u^p = c` in the sector `|arg u| ≤ π/q` of principal `q`-th roots
/// has `|arg(u^p)| < π`).  `True` when the argument range of `c`
/// ([`arg_range`]) lies in that range, `False` when it misses it and `c` is
/// not 0, `Unknown` otherwise: `√x = y` for a plain `y` (`x = y²` only for
/// `−π/2 < arg y ≤ π/2`), or `−√w` when `w` may be 0.
fn principal_power_preimage(arena: &mut Arena, c: ExprId, n: &Q) -> crate::base::truth::Truth {
    use crate::base::truth::Truth;
    let (p, q) = (n.numer().abs(), n.denom().clone());
    if p > q {
        return Truth::True;
    }
    let Some(range) = arg_range(arena, c) else {
        return Truth::Unknown;
    };
    let b = Ratio::new(p, q);
    let positive = n.is_positive();
    let (lo, hi) = (&range.center - &range.half, &range.center + &range.half);
    let point = range.half.is_zero();
    // the range of `f^(p/q)` shifted by `2k` turns: `(a, a + 2b]` for
    // `p > 0`, `[a, a + 2b)` for `p < 0`
    let meets = |k: i64| {
        let shift = Ratio::from_integer(BigInt::from(2 * k));
        let (a, e) = (&shift - &b, &shift + &b);
        match (point, positive) {
            (true, true) => a < lo && lo <= e,
            (true, false) => a <= lo && lo < e,
            (false, true) => lo < e && a < hi,
            (false, false) => a <= hi && lo < e,
        }
    };
    // in `(−b, b]` (`p > 0`) or `[−b, b)` (`p < 0`); an interval open
    // at `lo` may start at `−b`
    let minus_b = -b.clone();
    let inside = match (point, positive) {
        (true, true) => minus_b < lo && hi <= b,
        (false, true) => minus_b <= lo && hi <= b,
        (_, false) => minus_b <= lo && hi < b,
    };
    if inside {
        Truth::True
    } else if !range.maybe_zero && !(-1..=1).any(meets) {
        Truth::False
    } else {
        Truth::Unknown
    }
}

/// The arguments a value can take, in turns of `π`: `arg/π ∈ (center −
/// half, center + half]` (modulo 2; the point `center` when `half` is 0),
/// or the value is 0 (`maybe_zero`).
struct ArgRange {
    center: Q,
    half: Q,
    maybe_zero: bool,
}

/// The [`ArgRange`] of a product of factors whose arguments are known
/// exactly from the principal branch and the declared assumptions: a
/// nonzero rational or Gaussian rational with a rational argument
/// ([`gaussian_rational_arg`]), a positive or negative factor (`0`, `1`;
/// nonnegative and nonpositive ones may also be 0), `B^r` with a negative
/// `B` (`r`) and, for `0 < r < 1`, `B^r` with `B` of unknown sign (`r·arg B
/// ∈ (−rπ, rπ]`, or 0).  `None` when a factor is not of these kinds or the
/// range is the whole circle.  Exact: no evaluation decides an argument.
fn arg_range(arena: &mut Arena, value: ExprId) -> Option<ArgRange> {
    use crate::base::assumptions::{AssumptionCache, Props};
    let mut factors: Vec<ExprId> = vec![value];
    let mut out = ArgRange {
        center: Q::zero(),
        half: Q::zero(),
        maybe_zero: false,
    };
    let mut cache = AssumptionCache::new();
    while let Some(f) = factors.pop() {
        match arena.node(f) {
            ExprNode::Mul(ch) => {
                factors.extend(ch.iter().copied());
                continue;
            }
            ExprNode::Neg(inner) => {
                factors.push(*inner);
                out.center += Q::one();
                continue;
            }
            _ => {}
        }
        if crate::base::walk::free_symbols(arena, f).is_empty()
            && let Some(t) = gaussian_rational_arg(arena, f)
        {
            out.center += t;
            continue;
        }
        let mut known = |arena: &Arena, e: ExprId, p: Props| {
            crate::base::truth::Truth::from(cache.query(arena, e, p)).is_true()
        };
        if known(arena, f, Props::POSITIVE) {
            continue;
        }
        if known(arena, f, Props::NEGATIVE) {
            out.center += Q::one();
            continue;
        }
        if known(arena, f, Props::NONNEGATIVE) {
            out.maybe_zero = true;
            continue;
        }
        if known(arena, f, Props::NONPOSITIVE) {
            out.center += Q::one();
            out.maybe_zero = true;
            continue;
        }
        let ExprNode::Pow(base, e) = *arena.node(f) else {
            return None;
        };
        let r = arena.as_num(e)?.clone();
        if known(arena, base, Props::NEGATIVE) {
            out.center += r;
        } else if r.is_positive() && r < Q::one() {
            out.half += &r;
            out.maybe_zero |= !known(arena, base, Props::NONZERO);
        } else {
            return None;
        }
    }
    if out.half >= Q::one() {
        return None;
    }
    let two = Q::from_integer(BigInt::from(2));
    while out.center > Q::one() {
        out.center -= &two;
    }
    while out.center <= -Q::one() {
        out.center += &two;
    }
    Some(out)
}

/// Largest integer exponent `|n|` whose `n` roots are listed when peeling
/// `f^n = c`.
const MAX_INT_POWER_ROOTS: i64 = 64;

/// The principal logarithm of `c`, folded for a Gaussian rational whose
/// argument is a rational multiple of `π` (`ln i = iπ/2`,
/// `ln(1 + i) = ln 2/2 + iπ/4`), which `eval` leaves as `ln(i)`; a
/// candidate such as `(ln(i) − 2πi)²` (`exp(√x) = i`) is otherwise a
/// negative real the residual check cannot place on the branch cut of `√`.
fn principal_ln(arena: &mut Arena, c: ExprId) -> ExprId {
    let Some(t) = gaussian_rational_arg(arena, c) else {
        return arena.ln(c);
    };
    if t.is_zero() || t.is_one() {
        return arena.ln(c);
    }
    let (re, im) = crate::base::complex::as_real_imag(arena, c);
    let re = crate::transforms::eval::eval(arena, re);
    let im = crate::transforms::eval::eval(arena, im);
    let (Some(a), Some(b)) = (arena.as_num(re).cloned(), arena.as_num(im).cloned()) else {
        return arena.ln(c);
    };
    // |c| = |b| (a = 0) or |a|·√2 (|a| = |b|)
    let modulus_ln = if a.is_zero() {
        let m = arena.num_ratio(b.abs());
        arena.ln(m)
    } else {
        let m = arena.num_ratio(a.abs());
        let ln_m = arena.ln(m);
        let two = arena.int(2);
        let ln2 = arena.ln(two);
        let half = arena.rational(1, 2);
        let half_ln2 = arena.mul(&[half, ln2]);
        arena.add(&[ln_m, half_ln2])
    };
    let t_id = arena.num_ratio(t);
    let angle = arena.mul(&[t_id, arena.pi, arena.i_unit]);
    let sum = arena.add(&[modulus_ln, angle]);
    crate::transforms::eval::eval(arena, sum)
}

/// `arg(c)/π ∈ (−1, 1]` for a nonzero Gaussian rational `c = a + b·i` whose
/// argument is a rational multiple of `π` (`b = 0`, `a = 0` or `|a| = |b|`:
/// by Niven's theorem these are the only ones with a rational tangent).
fn gaussian_rational_arg(arena: &mut Arena, c: ExprId) -> Option<Q> {
    let (re, im) = crate::base::complex::as_real_imag(arena, c);
    let re = crate::transforms::eval::eval(arena, re);
    let im = crate::transforms::eval::eval(arena, im);
    let a = arena.as_num(re)?.clone();
    let b = arena.as_num(im)?.clone();
    let q = |n: i64, d: i64| Ratio::new(BigInt::from(n), BigInt::from(d));
    Some(match (a.signum(), b.signum()) {
        (sa, sb) if sa.is_zero() && sb.is_zero() => return None,
        (sa, sb) if sb.is_zero() => {
            if sa.is_positive() {
                q(0, 1)
            } else {
                q(1, 1)
            }
        }
        (sa, sb) if sa.is_zero() => {
            if sb.is_positive() {
                q(1, 2)
            } else {
                q(-1, 2)
            }
        }
        (sa, sb) if a.abs() == b.abs() => match (sa.is_positive(), sb.is_positive()) {
            (true, true) => q(1, 4),
            (false, true) => q(3, 4),
            (false, false) => q(-3, 4),
            (true, false) => q(-1, 4),
        },
        _ => return None,
    })
}

/// Is the `k`-th root `u` of `u^n = c` (argument `π·(t + 2k)/n`, `t·π =
/// arg c`) in the sector `−π/q < arg u ≤ π/q` of principal `q`-th roots?
/// Exactly then `(u^q)^(1/q) = u`, and `u^q` solves `f^(p/q) = c`.
fn in_principal_sector(t: &Q, k: usize, n: i64, q: &BigInt) -> bool {
    let two = Ratio::from_integer(BigInt::from(2));
    let mut s = (t + Ratio::from_integer(BigInt::from(2 * k as i64)))
        / Ratio::from_integer(BigInt::from(n));
    while s > Ratio::one() {
        s -= &two;
    }
    while s <= -Ratio::<BigInt>::one() {
        s += &two;
    }
    let bound = Ratio::new(BigInt::one(), q.clone());
    -bound.clone() < s && s <= bound
}

/// Continue peeling `inner` against each of several right-hand sides (the
/// preimages of a power; none for `f^(−k) = 0`, which has no root) and
/// merge the results by [`Solutions::union`]: no root needs every branch
/// to prove it, and roots next to a refused branch are partial.
fn peel_branches(
    arena: &mut Arena,
    inner: ExprId,
    rhss: &[ExprId],
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    let mut out = Solutions::none(Reason::NoPreimage);
    for &rhs in rhss {
        out = out.union(solve_by_peeling(arena, inner, rhs, var, cx));
    }
    out
}

/// Continue peeling `inner` against two alternative right-hand sides (`f =
/// c₁` or `f = c₂`) and merge the results by [`Solutions::union`]: up to
/// 0.48 roots next to a refused branch were all the roots (now partial),
/// and before that one branch proved empty next to a refused one gave "no
/// solution" — for a real `x`, `|x + i|² = 4` peels to `|x + i| = ±2`, the
/// branch `−2` is empty and `|x + i| = 2` (of a non-real argument) is
/// refused, while `±√3` solve it (SymPy's `solveset` too).
fn peel_two_branches(
    arena: &mut Arena,
    inner: ExprId,
    rhs1: ExprId,
    rhs2: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    let first = solve_by_peeling(arena, inner, rhs1, var, cx);
    first.union(solve_by_peeling(arena, inner, rhs2, var, cx))
}

// ═══════════════════════════════════════════════════════════════════════════
// Linear solver: a*x + b = 0 → x = -b/a
// ═══════════════════════════════════════════════════════════════════════════

fn solve_linear(arena: &mut Arena, poly: &Poly) -> Vec<Solution> {
    let a = poly.coeff(1); // coefficient of x
    let b = poly.coeff(0); // constant term

    if a.is_zero() {
        return Vec::new(); // Degenerate: 0*x + b = 0.
    }

    // x = -b / a
    let value = -b / a;
    let value_id = arena.num_ratio(value.clone());

    vec![Solution { value: value_id }]
}

// ═══════════════════════════════════════════════════════════════════════════
// Quadratic solver: a*x² + b*x + c = 0
// ═══════════════════════════════════════════════════════════════════════════

fn solve_quadratic(arena: &mut Arena, poly: &Poly) -> Vec<Solution> {
    let a = poly.coeff(2);
    let b = poly.coeff(1);
    let c = poly.coeff(0);

    if a.is_zero() {
        // Actually linear.
        let linear = Poly::from_coeffs(vec![c, b]);
        return solve_linear(arena, &linear);
    }

    // Discriminant: b² - 4ac
    let discriminant = &b * &b - Ratio::from_integer(BigInt::from(4)) * &a * &c;

    if discriminant.is_zero() {
        // Double root: x = -b / (2a)
        let two_a = Ratio::from_integer(BigInt::from(2)) * &a;
        let value = -b / two_a;
        let value_id = arena.num_ratio(value.clone());
        return vec![Solution { value: value_id }];
    }

    if discriminant.is_negative() {
        // Complex roots: x = (-b ± i√|Δ|) / (2a)
        let abs_disc = -discriminant;
        let neg_b = arena.num_ratio((-&b).clone());
        let abs_disc_id = arena.num_ratio(abs_disc.clone());

        // Check if |Δ| is a perfect square
        let sqrt_abs_disc = if let Some(s) = rational_sqrt(&abs_disc) {
            arena.num_ratio(s.clone())
        } else {
            arena.sqrt(abs_disc_id)
        };

        let i_sqrt = arena.mul(&[arena.i_unit, sqrt_abs_disc]);
        let two_a_val = Ratio::from_integer(BigInt::from(2)) * &a;
        let two_a_id = arena.num_ratio(two_a_val.clone());
        let two_a_inv = {
            let neg_one = arena.neg_one;
            arena.pow(two_a_id, neg_one)
        };

        // x1 = (-b + i√|Δ|) / (2a)
        let sum1 = arena.add(&[neg_b, i_sqrt]);
        let x1 = arena.mul(&[sum1, two_a_inv]);

        // x2 = (-b - i√|Δ|) / (2a)
        let neg_i_sqrt = arena.neg(i_sqrt);
        let sum2 = arena.add(&[neg_b, neg_i_sqrt]);
        let x2 = arena.mul(&[sum2, two_a_inv]);

        return vec![Solution { value: x1 }, Solution { value: x2 }];
    }

    // Check if the discriminant is a perfect square (rational root).
    if let Some(sqrt_disc) = rational_sqrt(&discriminant) {
        let two_a = Ratio::from_integer(BigInt::from(2)) * &a;

        // x = (-b ± √Δ) / (2a)
        let x1 = (-&b + &sqrt_disc) / &two_a;
        let x2 = (-&b - &sqrt_disc) / &two_a;

        let x1_id = arena.num_ratio(x1.clone());
        let x2_id = arena.num_ratio(x2.clone());

        if x1 == x2 {
            vec![Solution { value: x1_id }]
        } else {
            vec![Solution { value: x1_id }, Solution { value: x2_id }]
        }
    } else {
        // Discriminant is not a perfect square — roots are irrational.
        // Express as (-b ± sqrt(discriminant)) / (2*a) symbolically.
        let neg_b = {
            let neg_b_val = -&b;
            arena.num_ratio(neg_b_val.clone())
        };
        let sqrt_disc = {
            let disc_id = arena.num_ratio(discriminant.clone());
            arena.sqrt(disc_id)
        };
        let two_a_val = Ratio::from_integer(BigInt::from(2)) * &a;
        let two_a_id = arena.num_ratio(two_a_val.clone());
        let two_a_inv = {
            let neg_one = arena.neg_one;
            arena.pow(two_a_id, neg_one)
        };

        // x1 = (-b + sqrt(disc)) / (2a)
        let sum1 = arena.add(&[neg_b, sqrt_disc]);
        let x1 = arena.mul(&[sum1, two_a_inv]);

        // x2 = (-b - sqrt(disc)) / (2a)
        let neg_sqrt = arena.neg(sqrt_disc);
        let sum2 = arena.add(&[neg_b, neg_sqrt]);
        let x2 = arena.mul(&[sum2, two_a_inv]);

        vec![Solution { value: x1 }, Solution { value: x2 }]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cubic solver: a*x³ + b*x² + c*x + d = 0
// ═══════════════════════════════════════════════════════════════════════════

/// Solve a cubic polynomial. Tries rational roots first, then falls back
/// to Cardano's formula for irrational / complex roots.
fn solve_cubic(arena: &mut Arena, var: ExprId, poly: &Poly) -> Vec<Solution> {
    let a = poly.coeff(3);
    if a.is_zero() {
        let quadratic = Poly::from_coeffs(vec![poly.coeff(0), poly.coeff(1), poly.coeff(2)]);
        return solve_quadratic(arena, &quadratic);
    }

    // Try rational roots first — exact answers are preferable.
    let rational_attempt = solve_rational_roots(arena, var, poly);
    if !rational_attempt.is_empty() {
        return rational_attempt;
    }

    // No rational roots — use Cardano's formula.
    solve_cubic_cardano(arena, poly)
}

/// Pure Cardano's formula (no rational-root attempt) to avoid re-entry loops.
fn solve_cubic_cardano(arena: &mut Arena, poly: &Poly) -> Vec<Solution> {
    let a = poly.coeff(3);
    let b = poly.coeff(2);
    let c = poly.coeff(1);
    let d = poly.coeff(0);

    if a.is_zero() {
        let quadratic = Poly::from_coeffs(vec![d, c, b]);
        return solve_quadratic(arena, &quadratic);
    }

    // Binomial a·x³ + d = 0: cleaner roots-of-unity form than Cardano.
    if let Some(roots) = try_solve_binomial_rational(arena, poly) {
        return roots;
    }

    let a_id = arena.num_ratio(a.clone());
    let b_id = arena.num_ratio(b.clone());
    let c_id = arena.num_ratio(c.clone());
    let d_id = arena.num_ratio(d.clone());

    let three = arena.int(3);
    let nine = arena.int(9);
    let twenty_seven = arena.int(27);
    let two = arena.int(2);
    let four = arena.int(4);

    // Depress to t³ + pt + q = 0  via  x = t - b/(3a)
    //   p = (3ac - b²) / (3a²)
    //   q = (2b³ - 9abc + 27a²d) / (27a³)

    // p_num = 3ac - b²
    let tmp1 = arena.mul(&[three, a_id, c_id]);
    let tmp2 = arena.mul(&[b_id, b_id]);
    let p_num = arena.sub(tmp1, tmp2);
    let p_den = arena.mul(&[three, a_id, a_id]);
    let p = arena.div(p_num, p_den);

    // q_num = 2b³ - 9abc + 27a²d
    let tmp3 = arena.mul(&[two, b_id, b_id, b_id]);
    let tmp4 = arena.mul(&[nine, a_id, b_id, c_id]);
    let tmp4n = arena.neg(tmp4);
    let tmp5 = arena.mul(&[twenty_seven, a_id, a_id, d_id]);
    let q_num = arena.add(&[tmp3, tmp4n, tmp5]);
    let q_den = arena.mul(&[twenty_seven, a_id, a_id, a_id]);
    let q = arena.div(q_num, q_den);

    // Discriminant: Δ = q²/4 + p³/27
    let qq = arena.mul(&[q, q]);
    let qq_over4 = arena.div(qq, four);
    let ppp = arena.mul(&[p, p, p]);
    let ppp_over27 = arena.div(ppp, twenty_seven);
    let disc = arena.add(&[qq_over4, ppp_over27]);

    // √Δ
    let half = arena.rational(1, 2);
    let sqrt_disc = arena.pow(disc, half);

    // Cardano: t = cbrt(-q/2 + √Δ) + cbrt(-q/2 - √Δ)
    let neg_q = arena.neg(q);
    let neg_q_half = arena.div(neg_q, two);

    let u_arg = arena.add(&[neg_q_half, sqrt_disc]);
    let v_arg = arena.sub(neg_q_half, sqrt_disc);

    // Cardano's formula needs the *real* cube roots of the two radicands
    // (their product must be −p/3).  A `Pow(negative, 1/3)` node is
    // evaluated on the principal complex branch by `evalf`, which would
    // silently produce wrong roots, so a radicand that is provably
    // negative is written as −cbrt(|radicand|).  The signs follow from the
    // exact rational data: for Δ ≥ 0, √Δ ≥ |q|/2 exactly when p ≥ 0.
    let (p_rat, q_rat, disc_rat) = {
        let three_r = Ratio::from_integer(BigInt::from(3));
        let nine_r = Ratio::from_integer(BigInt::from(9));
        let two_r = Ratio::from_integer(BigInt::from(2));
        let four_r = Ratio::from_integer(BigInt::from(4));
        let twenty_seven_r = Ratio::from_integer(BigInt::from(27));
        let p_r = (&three_r * &a * &c - &b * &b) / (&three_r * &a * &a);
        let q_r = (&two_r * &b * &b * &b - &nine_r * &a * &b * &c + &twenty_seven_r * &a * &a * &d)
            / (&twenty_seven_r * &a * &a * &a);
        let disc_r = &q_r * &q_r / &four_r + &p_r * &p_r * &p_r / &twenty_seven_r;
        (p_r, q_r, disc_r)
    };
    let (u_sign, v_sign) = cardano_radicand_signs(&p_rat, &q_rat, &disc_rat);

    let u = real_cbrt_with_sign(arena, u_arg, u_sign);
    let v = real_cbrt_with_sign(arena, v_arg, v_sign);

    let t1 = arena.add(&[u, v]);

    // Shift back: x = t - b/(3a)
    let three_a = arena.mul(&[three, a_id]);
    let shift = arena.div(b_id, three_a);
    let x1 = arena.sub(t1, shift);

    // Other two roots via cube roots of unity:
    //   ω  = (-1 + i√3)/2
    //   ω² = (-1 - i√3)/2
    let omega_re = arena.rational(-1, 2);
    let sqrt3 = arena.pow(three, half);
    let sqrt3_half = arena.div(sqrt3, two);
    let i_unit = arena.i_unit;
    let omega_im = arena.mul(&[sqrt3_half, i_unit]);
    let omega = arena.add(&[omega_re, omega_im]);
    let omega2 = arena.sub(omega_re, omega_im);

    let ou = arena.mul(&[omega, u]);
    let o2v = arena.mul(&[omega2, v]);
    let t2 = arena.add(&[ou, o2v]);
    let o2u = arena.mul(&[omega2, u]);
    let ov = arena.mul(&[omega, v]);
    let t3 = arena.add(&[o2u, ov]);

    let x2 = arena.sub(t2, shift);
    let x3 = arena.sub(t3, shift);

    // Simplify all roots through eval
    let x1s = crate::transforms::eval::eval(arena, x1);
    let x2s = crate::transforms::eval::eval(arena, x2);
    let x3s = crate::transforms::eval::eval(arena, x3);

    vec![
        Solution { value: x1s },
        Solution { value: x2s },
        Solution { value: x3s },
    ]
}

/// Signs of the Cardano radicands `−q/2 + √Δ` and `−q/2 − √Δ` for the
/// depressed cubic `t³ + pt + q` with `Δ = q²/4 + p³/27`.
///
/// Returns `(sign_u, sign_v)` with values in `{-1, 0, 1}`; `0` is also used
/// when `Δ < 0` (complex radicands, principal branch is correct there).
fn cardano_radicand_signs(p: &Q, q: &Q, disc: &Q) -> (i8, i8) {
    use std::cmp::Ordering;
    if disc.is_negative() {
        return (0, 0);
    }
    let neg_q_sign: i8 = match q.cmp(&Ratio::zero()) {
        Ordering::Less => 1,
        Ordering::Equal => 0,
        Ordering::Greater => -1,
    };
    if disc.is_zero() {
        // Both radicands equal −q/2.
        return (neg_q_sign, neg_q_sign);
    }
    // Δ > 0: √Δ > |q|/2 ⇔ p > 0; √Δ = |q|/2 ⇔ p = 0; √Δ < |q|/2 ⇔ p < 0.
    let p_sign = match p.cmp(&Ratio::zero()) {
        Ordering::Less => -1i8,
        Ordering::Equal => 0,
        Ordering::Greater => 1,
    };
    let u_sign = if neg_q_sign >= 0 {
        // −q/2 ≥ 0 and √Δ > 0 ⇒ positive.
        1
    } else {
        // −q/2 < 0: sign decided by whether √Δ exceeds |q|/2.
        p_sign
    };
    let v_sign = if neg_q_sign <= 0 {
        -1
    } else {
        // −q/2 > 0: −q/2 − √Δ is positive iff √Δ < q/2 ⇔ p < 0.
        -p_sign
    };
    (u_sign, v_sign)
}

/// Build the real cube root of `arg` given its known sign: `cbrt(arg)` for
/// a positive radicand, `−cbrt(−arg)` for a negative one (so that no cube
/// root of a negative real is ever emitted), `0` for a zero radicand.  An
/// unknown sign (`0` for a complex radicand) falls back to the principal
/// branch `arg^(1/3)`.
fn real_cbrt_with_sign(arena: &mut Arena, arg: ExprId, sign: i8) -> ExprId {
    let third = arena.rational(1, 3);
    match sign {
        1 => arena.pow(arg, third),
        -1 => {
            let neg_arg = arena.neg(arg);
            let neg_arg = crate::transforms::eval::eval(arena, neg_arg);
            let root = arena.pow(neg_arg, third);
            arena.neg(root)
        }
        _ => {
            // Zero radicand (exactly), or complex radicand (principal branch).
            let ev = crate::transforms::eval::eval(arena, arg);
            if arena.is_zero_structural(ev) {
                return arena.zero;
            }
            arena.pow(arg, third)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Quartic solver: Ferrari's method  a*x⁴ + b*x³ + c*x² + d*x + e = 0
// ═══════════════════════════════════════════════════════════════════════════

/// Solve a quartic polynomial. Tries rational roots first, then falls back
/// to Ferrari's method.
fn solve_quartic(arena: &mut Arena, var: ExprId, poly: &Poly) -> Vec<Solution> {
    let a = poly.coeff(4);
    if a.is_zero() {
        let cubic = Poly::from_coeffs(vec![
            poly.coeff(0),
            poly.coeff(1),
            poly.coeff(2),
            poly.coeff(3),
        ]);
        return solve_cubic(arena, var, &cubic);
    }

    // Try rational roots first.
    let rational_attempt = solve_rational_roots(arena, var, poly);
    if !rational_attempt.is_empty() {
        return rational_attempt;
    }

    // No rational roots — use Ferrari's method.
    solve_quartic_ferrari(arena, poly)
}

/// Pure Ferrari's method (no rational-root attempt) to avoid re-entry loops.
fn solve_quartic_ferrari(arena: &mut Arena, poly: &Poly) -> Vec<Solution> {
    let a = poly.coeff(4);
    let b = poly.coeff(3);
    let c = poly.coeff(2);
    let d = poly.coeff(1);
    let e = poly.coeff(0);

    if a.is_zero() {
        let cubic = Poly::from_coeffs(vec![e.clone(), d, c, b]);
        return solve_cubic_cardano(arena, &cubic);
    }

    // Binomial a·x⁴ + e = 0: cleaner roots-of-unity form than Ferrari.
    if let Some(roots) = try_solve_binomial_rational(arena, poly) {
        return roots;
    }

    // Depress to t⁴ + pt² + qt + r = 0  via  x = t - b/(4a)
    //   p = (8ac - 3b²) / (8a²)
    //   q = (b³ - 4abc + 8a²d) / (8a³)
    //   r = (-3b⁴ + 256a³e - 64a²bd + 16ab²c) / (256a⁴)

    // Exact rational depressed coefficients (also used for the biquadratic
    // shortcut below and for the resolvent-root filter).
    let (p_rat, q_rat, r_rat) = {
        let r3 = Ratio::from_integer(BigInt::from(3));
        let r4 = Ratio::from_integer(BigInt::from(4));
        let r8 = Ratio::from_integer(BigInt::from(8));
        let r16 = Ratio::from_integer(BigInt::from(16));
        let r64 = Ratio::from_integer(BigInt::from(64));
        let r256 = Ratio::from_integer(BigInt::from(256));
        let a2 = &a * &a;
        let a3 = &a2 * &a;
        let a4 = &a3 * &a;
        let b2 = &b * &b;
        let p_r = (&r8 * &a * &c - &r3 * &b2) / (&r8 * &a2);
        let q_r = (&b2 * &b - &r4 * &a * &b * &c + &r8 * &a2 * &d) / (&r8 * &a3);
        let r_r = (-&r3 * &b2 * &b2 + &r256 * &a3 * &e - &r64 * &a2 * &b * &d
            + &r16 * &a * &b2 * &c)
            / (&r256 * &a4);
        (p_r, q_r, r_r)
    };

    // Biquadratic t⁴ + pt² + r = 0 (q = 0): Ferrari's factorisation
    // degenerates (k = √(2m − p) = 0 for the rational resolvent root
    // m = p/2, giving 0/0), so solve the quadratic in s = t² and take
    // t = ±√s instead.
    if q_rat.is_zero() {
        let quad = Poly::from_coeffs(vec![r_rat.clone(), p_rat.clone(), Ratio::one()]);
        let s_roots = solve_quadratic(arena, &quad);
        let four_a = Ratio::from_integer(BigInt::from(4)) * &a;
        let shift_rat = &b / &four_a;
        let shift = arena.num_ratio(shift_rat.clone());
        let half = arena.rational(1, 2);
        let mut out: Vec<Solution> = Vec::new();
        for s in s_roots {
            let s_ev = crate::transforms::eval::eval(arena, s.value);
            let t_pos = arena.pow(s_ev, half);
            let t_neg = arena.neg(t_pos);
            for t in [t_pos, t_neg] {
                let x = arena.sub(t, shift);
                let x = crate::transforms::eval::eval(arena, x);
                if !out.iter().any(|o| o.value == x) {
                    out.push(Solution { value: x });
                }
            }
        }
        return out;
    }

    let a_id = arena.num_ratio(a.clone());
    let b_id = arena.num_ratio(b.clone());
    let c_id = arena.num_ratio(c.clone());
    let d_id = arena.num_ratio(d.clone());
    let e_id = arena.num_ratio(e.clone());

    let two = arena.int(2);
    let three = arena.int(3);
    let four = arena.int(4);
    let eight = arena.int(8);
    let sixteen = arena.int(16);
    let sixty_four = arena.int(64);
    let two_fifty_six = arena.int(256);

    // p = (8ac - 3b²) / (8a²)
    let tmp_8ac = arena.mul(&[eight, a_id, c_id]);
    let tmp_3bb = arena.mul(&[three, b_id, b_id]);
    let p_num = arena.sub(tmp_8ac, tmp_3bb);
    let p_den = arena.mul(&[eight, a_id, a_id]);
    let p = arena.div(p_num, p_den);

    // q = (b³ - 4abc + 8a²d) / (8a³)
    let tmp_bbb = arena.mul(&[b_id, b_id, b_id]);
    let tmp_4abc = arena.mul(&[four, a_id, b_id, c_id]);
    let tmp_4abc_n = arena.neg(tmp_4abc);
    let tmp_8aad = arena.mul(&[eight, a_id, a_id, d_id]);
    let q_num = arena.add(&[tmp_bbb, tmp_4abc_n, tmp_8aad]);
    let q_den = arena.mul(&[eight, a_id, a_id, a_id]);
    let q = arena.div(q_num, q_den);

    // r = (-3b⁴ + 256a³e - 64a²bd + 16ab²c) / (256a⁴)
    let tmp_3b4 = arena.mul(&[three, b_id, b_id, b_id, b_id]);
    let tmp_3b4_n = arena.neg(tmp_3b4);
    let tmp_256a3e = arena.mul(&[two_fifty_six, a_id, a_id, a_id, e_id]);
    let tmp_64a2bd = arena.mul(&[sixty_four, a_id, a_id, b_id, d_id]);
    let tmp_64a2bd_n = arena.neg(tmp_64a2bd);
    let tmp_16ab2c = arena.mul(&[sixteen, a_id, b_id, b_id, c_id]);
    let r_num = arena.add(&[tmp_3b4_n, tmp_256a3e, tmp_64a2bd_n, tmp_16ab2c]);
    let r_den = arena.mul(&[two_fifty_six, a_id, a_id, a_id, a_id]);
    let r = arena.div(r_num, r_den);

    // Resolvent cubic:  8m³ - 4pm² - 8rm + (4pr - q²) = 0
    // We need the coefficients as rationals to build a Poly.
    let resolvent_c3 = eight;
    let neg_four = arena.neg(four);
    let resolvent_c2 = arena.mul(&[neg_four, p]);
    let neg_eight = arena.neg(eight);
    let resolvent_c1 = arena.mul(&[neg_eight, r]);
    let tmp_4pr = arena.mul(&[four, p, r]);
    let tmp_qq = arena.mul(&[q, q]);
    let tmp_qq_n = arena.neg(tmp_qq);
    let resolvent_c0 = arena.add(&[tmp_4pr, tmp_qq_n]);

    let resolvent_c3_e = crate::transforms::eval::eval(arena, resolvent_c3);
    let resolvent_c2_e = crate::transforms::eval::eval(arena, resolvent_c2);
    let resolvent_c1_e = crate::transforms::eval::eval(arena, resolvent_c1);
    let resolvent_c0_e = crate::transforms::eval::eval(arena, resolvent_c0);

    let rc3 = match arena.as_num(resolvent_c3_e) {
        Some(v) => v.clone(),
        None => return Vec::new(),
    };
    let rc2 = match arena.as_num(resolvent_c2_e) {
        Some(v) => v.clone(),
        None => return Vec::new(),
    };
    let rc1 = match arena.as_num(resolvent_c1_e) {
        Some(v) => v.clone(),
        None => return Vec::new(),
    };
    let rc0 = match arena.as_num(resolvent_c0_e) {
        Some(v) => v.clone(),
        None => return Vec::new(),
    };

    let resolvent_poly = Poly::from_coeffs(vec![rc0, rc1, rc2, rc3]);

    // We need 2m - p ≠ 0 for a non-degenerate Ferrari factorization; with
    // q ≠ 0 (guaranteed above) m = p/2 is never a resolvent root.
    debug_assert!(!q_rat.is_zero());

    // Try rational roots of the resolvent cubic first.
    // This avoids the *casus irreducibilis* problem where Cardano's formula
    // produces complex cube roots for real rational roots, yielding
    // unsimplifiable expressions like cbrt(±i·√(1/27)).
    let m = if let Some(m_rat) = find_preferred_resolvent_root(&resolvent_poly, &p_rat) {
        arena.num_ratio(m_rat.clone())
    } else {
        let m_solutions = solve_cubic_cardano(arena, &resolvent_poly);
        if m_solutions.is_empty() {
            return Vec::new();
        }
        m_solutions[0].value
    };

    // Factor into two quadratics via √(2m − p):
    //   t² + k·t + (m − q/(2k)) = 0
    //   t² − k·t + (m + q/(2k)) = 0
    // where k = √(2m − p)
    let half = arena.rational(1, 2);
    let two_m = arena.mul(&[two, m]);
    let two_m_minus_p = arena.sub(two_m, p);
    let k = arena.pow(two_m_minus_p, half);

    let two_k = arena.mul(&[two, k]);
    let q_over_2k = arena.div(q, two_k);

    // shift = b/(4a)
    let four_a = arena.mul(&[four, a_id]);
    let shift = arena.div(b_id, four_a);

    // Quadratic 1:  t² + kt + (m - q/(2k)) = 0
    let s1 = arena.sub(m, q_over_2k);
    let kk = arena.mul(&[k, k]);
    let four_s1 = arena.mul(&[four, s1]);
    let disc1 = arena.sub(kk, four_s1);
    let sqrt_disc1 = arena.pow(disc1, half);
    let neg_k = arena.neg(k);

    let sum1a = arena.add(&[neg_k, sqrt_disc1]);
    let t1a = arena.div(sum1a, two);
    let diff1b = arena.sub(neg_k, sqrt_disc1);
    let t1b = arena.div(diff1b, two);

    let x1 = arena.sub(t1a, shift);
    let x2 = arena.sub(t1b, shift);

    // Quadratic 2:  t² - kt + (m + q/(2k)) = 0
    let s2 = arena.add(&[m, q_over_2k]);
    let kk2 = arena.mul(&[k, k]);
    let four_s2 = arena.mul(&[four, s2]);
    let disc2 = arena.sub(kk2, four_s2);
    let sqrt_disc2 = arena.pow(disc2, half);

    let sum2a = arena.add(&[k, sqrt_disc2]);
    let t2a = arena.div(sum2a, two);
    let diff2b = arena.sub(k, sqrt_disc2);
    let t2b = arena.div(diff2b, two);

    let x3 = arena.sub(t2a, shift);
    let x4 = arena.sub(t2b, shift);

    // Simplify all roots
    let x1s = crate::transforms::eval::eval(arena, x1);
    let x2s = crate::transforms::eval::eval(arena, x2);
    let x3s = crate::transforms::eval::eval(arena, x3);
    let x4s = crate::transforms::eval::eval(arena, x4);

    vec![
        Solution { value: x1s },
        Solution { value: x2s },
        Solution { value: x3s },
        Solution { value: x4s },
    ]
}

/// Find a rational root of the resolvent cubic that yields a non-degenerate
/// Ferrari factorization (i.e. `2m − p ≠ 0`, so that `k = √(2m−p) ≠ 0`).
///
/// Uses the Rational Root Theorem: for a polynomial with integer coefficients,
/// every rational root `p/q` satisfies `p | a₀` and `q | aₙ`; when that
/// search finds no non-degenerate root (it enumerates no divisors of a
/// coefficient above 10⁶), the linear factors of the resolvent over ℤ.
///
/// Returns `None` if no rational root is found (Cardano fallback will be used).
fn find_preferred_resolvent_root(resolvent: &Poly, p_rat: &Q) -> Option<Q> {
    let found = rational_resolvent_root_by_divisors(resolvent, p_rat);
    if let Some((m, true)) = found {
        return Some(m);
    }
    // `divisors` gives up above 10⁶ (it returns `{1, n}`), so a resolvent
    // with large coefficients kept its rational root hidden and Cardano's
    // formula took over: the Rothstein–Trager polynomial `7168t⁴ + 7168t³ +
    // 96t² + 16t + 1` of `∫ x³/(x⁴ + 4x³ − 1) dx` (resolvent root `−9/224`)
    // had its roots in cube roots of `81/44957696·√21 ± …` instead of in
    // `ℚ(√2)` (up to 0.46).  Its linear factors over ℤ are its rational
    // roots.
    let two = Ratio::from_integer(BigInt::from(2));
    let (_, factors) = resolvent.factor_over_z();
    let mut fallback = found.map(|(m, _)| m);
    for (f, _) in &factors {
        if f.degree() == Some(1) {
            let m = -f.coeff(0) / f.coeff(1);
            if &two * &m != *p_rat {
                return Some(m);
            }
            fallback.get_or_insert(m);
        }
    }
    fallback
}

/// The rational-root search of [`find_preferred_resolvent_root`] over the
/// divisors of the end coefficients: a root and whether it is
/// non-degenerate (`2m ≠ p`).
fn rational_resolvent_root_by_divisors(resolvent: &Poly, p_rat: &Q) -> Option<(Q, bool)> {
    let (int_poly, _scale) = clear_denominators(resolvent);
    let a0 = int_poly.coeff(0).to_integer();
    let an = {
        let v = int_poly.leading_coeff()?;
        v.to_integer()
    };

    let two_r = Ratio::from_integer(BigInt::from(2));
    let mut fallback: Option<Q> = None;

    if a0.is_zero() {
        // m = 0 is a root.  Record it but keep looking for a non-degenerate one.
        let zero = Ratio::zero();
        if &two_r * &zero - p_rat != Ratio::zero() {
            return Some((zero, true));
        }
        fallback = Some(zero);

        // Divide out m and check the remaining quadratic for rational roots.
        let reduced_coeffs: Vec<Q> = resolvent.coeffs().iter().skip(1).cloned().collect();
        let reduced = Poly::from_coeffs(reduced_coeffs);
        let aq = reduced.coeff(2);
        let bq = reduced.coeff(1);
        let cq = reduced.coeff(0);
        if !aq.is_zero() {
            let disc = &bq * &bq - Ratio::from_integer(BigInt::from(4)) * &aq * &cq;
            if let Some(sqrt_d) = rational_sqrt(&disc) {
                let two_aq = Ratio::from_integer(BigInt::from(2)) * &aq;
                for candidate in [(-&bq + &sqrt_d) / &two_aq, (-&bq - &sqrt_d) / &two_aq] {
                    if &two_r * &candidate - p_rat != Ratio::zero() {
                        return Some((candidate, true));
                    }
                    if fallback.is_none() {
                        fallback = Some(candidate);
                    }
                }
            }
        }
    } else {
        let divs_a0 = divisors(&a0.abs());
        let divs_an = divisors(&an.abs());

        for p_div in &divs_a0 {
            for q_div in &divs_an {
                for &sign in &[1i64, -1i64] {
                    let candidate = Ratio::new(p_div * BigInt::from(sign), q_div.clone());
                    if resolvent.eval(&candidate).is_zero() {
                        // Prefer a root where 2m - p ≠ 0.
                        if &two_r * &candidate - p_rat != Ratio::zero() {
                            return Some((candidate, true));
                        }
                        if fallback.is_none() {
                            fallback = Some(candidate);
                        }
                    }
                }
            }
        }
    }

    fallback.map(|m| (m, false))
}

// ═══════════════════════════════════════════════════════════════════════════
// Exact factorisation over ℤ for higher-degree polynomials
// ═══════════════════════════════════════════════════════════════════════════

/// The roots of a polynomial over ℚ through its exact factorisation over
/// ℤ ([`Poly::factor_over_z`]: content, Yun's square-free decomposition,
/// Berlekamp–Zassenhaus).
///
/// The content is dropped first, so `c·p` has exactly the roots of `p` for
/// every rational `c ≠ 0`, and every irreducible factor is solved once,
/// whatever its multiplicity: a linear factor *is* a rational root, a
/// quadratic, cubic or quartic factor goes to the radical formulas, and a
/// `RootOf(f, k)` placeholder is emitted only for an irreducible `f` of
/// degree ≥ 5, with `k` running over the roots of *that factor* — never
/// over a reducible or non-square-free polynomial.  The result lists the
/// **distinct** roots.
///
/// This replaced a Rational Root Theorem search whose divisor enumeration
/// gave up (silently, returning `{1, |n|}`) for a coefficient above `10⁶`,
/// so that `4·p(x)` lost the rational roots `p(x)` had, and which divided a
/// found root out once only, leaving its multiplicity in the cofactor
/// handed to `RootOf`.
///
/// Used for degree ≥ 5 and as the first pass of the cubic and quartic
/// solvers.  A binomial `a·xⁿ + b` of degree ≥ 5 keeps its roots-of-unity
/// form (before factoring, so `x⁵ − 32` is not split into a linear and a
/// quartic factor).
fn solve_rational_roots(arena: &mut Arena, var: ExprId, poly: &Poly) -> Vec<Solution> {
    let degree = match poly.degree() {
        Some(d) if d >= 1 => d,
        _ => return Vec::new(),
    };

    // Binomial a·xⁿ + b = 0 (n ≥ 5): all n roots explicitly via roots of
    // unity.  Cubics/quartics get here via their own solvers, which fall
    // back to the binomial form only after rational-root extraction.
    if degree > 4
        && let Some(roots) = try_solve_binomial_rational(arena, poly)
    {
        return roots;
    }

    let (_content, factors) = poly.factor_over_z();
    let mut roots: Vec<Solution> = Vec::new();
    for (factor, _multiplicity) in &factors {
        let found = match factor.degree() {
            None | Some(0) => Vec::new(),
            Some(1) => solve_linear(arena, factor),
            Some(2) => solve_quadratic(arena, factor),
            // Irreducible over ℚ: no rational root to extract, straight to
            // the radical formulas (which are invariant under the scaling
            // that made the factor primitive).
            Some(3) => solve_cubic_cardano(arena, factor),
            Some(4) => solve_quartic_ferrari(arena, factor),
            Some(d) => {
                if let Some(explicit) = try_solve_binomial_rational(arena, factor) {
                    explicit
                } else {
                    let factor_expr = polybridge::poly_to_expr(arena, factor, var);
                    (0..d)
                        .map(|i| {
                            let idx = arena.int(i as i64);
                            Solution {
                                value: arena.intern(ExprNode::RootOf(factor_expr, var, idx)),
                            }
                        })
                        .collect()
                }
            }
        };
        for sol in found {
            if !roots.iter().any(|s| s.value == sol.value) {
                roots.push(sol);
            }
        }
    }

    tracing::debug!(
        factors = factors.len(),
        roots = roots.len(),
        "roots via factorisation over ℤ"
    );
    roots
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Try to compute the exact square root of a non-negative rational.
///
/// Returns `Some(√r)` if `r` is a perfect square (both numerator and
/// denominator are perfect squares), or `None` otherwise.
fn rational_sqrt(r: &Q) -> Option<Q> {
    if r.is_negative() {
        return None;
    }
    if r.is_zero() {
        return Some(Ratio::zero());
    }

    let n = r.numer().abs();
    let d = r.denom().abs();

    let sqrt_n = n.sqrt();
    let sqrt_d = d.sqrt();

    if &sqrt_n * &sqrt_n == n && &sqrt_d * &sqrt_d == d {
        Some(Ratio::new(sqrt_n, sqrt_d))
    } else {
        None
    }
}

/// Clear denominators of a polynomial's coefficients.
///
/// Returns the integer-coefficient polynomial and the scale factor.
fn clear_denominators(poly: &Poly) -> (Poly, BigInt) {
    if poly.is_zero() {
        return (Poly::zero(), BigInt::one());
    }

    // Compute LCM of all denominators.
    let mut lcm = BigInt::one();
    for c in poly.coeffs() {
        let d = c.denom().abs();
        lcm = num_integer::lcm(lcm, d);
    }

    // Multiply all coefficients by the LCM.
    let scale = Ratio::from_integer(lcm.clone());
    let scaled = poly.scale(&scale);

    (scaled, lcm)
}

/// Compute all positive divisors of `n` (a positive BigInt).
///
/// Returns them in ascending order.  For n=0, returns `[1]` as a
/// fallback.
fn divisors(n: &BigInt) -> Vec<BigInt> {
    if n.is_zero() {
        return vec![BigInt::one()];
    }

    let n_abs = n.abs();

    // For small numbers, brute-force trial division.
    // For large numbers, this would be slow — but our polynomials
    // typically have small coefficients.
    let limit: u64 = match (&n_abs).try_into() {
        Ok(v) if v <= 1_000_000u64 => v,
        _ => {
            // Very large coefficient — just return 1 and n.
            return vec![BigInt::one(), n_abs];
        }
    };

    let mut result = Vec::new();
    let mut i = 1u64;
    while i * i <= limit {
        if limit.is_multiple_of(i) {
            result.push(BigInt::from(i));
            if i * i != limit {
                result.push(BigInt::from(limit / i));
            }
        }
        i += 1;
    }

    result.sort();
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Change-of-variable solving
// ═══════════════════════════════════════════════════════════════════════════

/// Try solving by detecting that the expression is polynomial in some f(x).
/// E.g., `exp(2x) - 3*exp(x) + 2` is polynomial in `t = exp(x)`: `t² - 3t + 2`.
fn try_change_of_variable(arena: &mut Arena, expr: ExprId, var: ExprId, cx: &SolveCx) -> Solutions {
    let var_sym = match arena.node(var) {
        ExprNode::Symbol(sid) => *sid,
        _ => return Solutions::refused(Reason::NotFound),
    };

    // Collect candidate generators: inner functions applied to var, then
    // trigonometric functions and exponentials of one linear form `a·x + b`,
    // used only for a proof that there is no root (`cos(3x + 1)² + cos(3x +
    // 1) − 6` for a real `x`: up to 0.52 left to the half-angle substitution,
    // which found a partial list; the roots of the others are left to the
    // strategies that found them before).
    let mut candidates = collect_generators(arena, expr, var, var_sym);
    let bare = candidates.len();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if let ExprNode::Sin(u) | ExprNode::Cos(u) | ExprNode::Tan(u) | ExprNode::Exp(u) =
            *arena.node(id)
            && u != var
            && !candidates.contains(&id)
            && linear_rate(arena, u, var).is_some()
        {
            candidates.push(id);
        }
    }

    for (index, generator) in candidates.into_iter().enumerate() {
        let proofs_only = index >= bare;
        // Create a temporary variable for substitution
        let t = arena.symbol("__t_subst");

        // --- Simple case: direct structural substitution of generator → t ---
        // This handles e.g. sin(x)^2 - sin(x) where walk_and_rebuild
        // naturally replaces sin(x) inside Pow(sin(x), 2) as well.
        let substituted = arena.subs_structural(expr, generator, t);

        if !expr_contains_var(arena, substituted, var) {
            // All the roots in `t`, or none: a polynomial with radical
            // coefficients of which the solver finds some roots only is
            // solved over its number field, or this generator is skipped.
            // A periodic equation in `t` gives its families in a general
            // solve: up to 0.46 its principal roots alone, and for a real `x`
            // `cot(eˣ) = 2` had the solution set `{ln(2·atan(√5 − 2))}`
            // instead of `ln(acot 2 + nπ)` for `n ≥ 0`.
            let t_outcome = if cx.period().is_some() && has_periodic_atom(arena, substituted, t, cx)
            {
                solve_impl(arena, substituted, t, cx)
            } else {
                substitution_roots(arena, substituted, t, cx)
            };

            if let Ok((t_roots, t_missing)) = t_outcome.without_conditions().into_found() {
                // Back-substitute: for each t = c, solve generator(var) = c.
                // Use solve_by_peeling directly to avoid recursing into
                // try_change_of_variable again (which would infinite-loop).
                // Every root in `t` of a polynomial without a preimage is a
                // proof: for a real `x`, `cos(3x + 1)² + cos(3x + 1) − 6 = 0`
                // has `cos(3x + 1) ∈ {2, −3}`, outside `[−1, 1]` (up to 0.52
                // refused, after a half-angle substitution).  (The principal
                // roots of a periodic equation in `t` are not all its roots.)
                let back = back_substitute(arena, generator, t_roots, var, cx).lacking(t_missing);
                let polynomial_in_t = symbolic_poly_coeffs(arena, substituted, t)
                    .is_some_and(|cs| cs.iter().all(|&c| !expr_contains_var(arena, c, t)));

                if (polynomial_in_t && back.is_none_proved())
                    || (!proofs_only && (back.has_roots() || back.is_nested_periods()))
                {
                    return back;
                }
            }
        }

        // --- Advanced: detect exp(n*x) = exp(x)^n pattern ---
        // exp(2*x) is Exp(Mul([2, x])) which does NOT structurally
        // contain exp(x) = Exp(x), so simple substitution misses it.
        // We rewrite exp(k*x) → exp(x)^k first, then substitute.
        if let ExprNode::Exp(inner) = arena.node(generator).clone()
            && inner == var
        {
            let rewritten = rewrite_exp_powers(arena, expr, var, generator);
            if rewritten != expr {
                let substituted2 = arena.subs_structural(rewritten, generator, t);
                if !expr_contains_var(arena, substituted2, var)
                    && let Ok((t_roots, t_missing)) = substitution_roots(arena, substituted2, t, cx)
                        .without_conditions()
                        .into_found()
                {
                    let back =
                        back_substitute(arena, generator, t_roots, var, cx).lacking(t_missing);
                    if back.has_roots() || back.is_nested_periods() {
                        return back;
                    }
                }
            }
        }
    }

    Solutions::refused(Reason::NotFound)
}

/// The roots in `var` of `generator = tᵢ` for the roots `tᵢ` of a
/// substituted equation, by peeling, merged by [`Solutions::union`] (no
/// root when no `tᵢ` has a preimage; partial when one is refused).
fn back_substitute(
    arena: &mut Arena,
    generator: ExprId,
    t_roots: Vec<ExprId>,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    let mut out = Solutions::none(Reason::NoPreimage);
    for t in t_roots {
        out = out.union(solve_by_peeling(arena, generator, t, var, cx));
    }
    out
}

/// An equation in `sin`, `cos`, `tan` of `aᵢ·x + bᵢ` alone with rational
/// `aᵢ` whose gcd `r` (gcd of the numerators over the lcm of the
/// denominators) is not 1, solved in `u = r·x` (multiples `aᵢ/r`, coprime
/// integers) and mapped back by `x = u/r`: `sin 300x + sin 600x = 0` and
/// `cos(x/5) + cos(x/15) = 0` were refused before 0.47 (the half-angle
/// substitution of `sin 600x` has degree 1200), SymPy solves both.  `u`
/// carries the assumptions of `x` (`r > 0`).  `None` for another shape or
/// `r = 1`.
fn try_scale_trig_arguments(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    let not_found = || Solutions::refused(Reason::NotFound);
    let ExprNode::Symbol(var_sid) = *arena.node(var) else {
        return not_found();
    };
    // `var` only in the trigonometric atoms
    if walk_var_atoms(arena, expr, var, |id| {
        matches!(
            arena.node(id),
            ExprNode::Sin(_) | ExprNode::Cos(_) | ExprNode::Tan(_)
        )
    })
    .is_none()
    {
        return not_found();
    }
    let mut slopes: Vec<Q> = Vec::new();
    let mut ok = true;
    let atoms: Vec<ExprId> = crate::base::walk::post_order_ids(arena, expr)
        .into_iter()
        .filter(|&id| {
            matches!(arena.node(id), ExprNode::Sin(a) | ExprNode::Cos(a) | ExprNode::Tan(a) if expr_contains_var(arena, *a, var))
        })
        .collect();
    for id in atoms {
        let (ExprNode::Sin(a) | ExprNode::Cos(a) | ExprNode::Tan(a)) = *arena.node(id) else {
            continue;
        };
        match linear_rate(arena, a, var) {
            Some((rate, _)) => match arena.as_num(rate) {
                Some(q) if !q.is_zero() => slopes.push(q.abs()),
                _ => ok = false,
            },
            None => ok = false,
        }
    }
    if !ok || slopes.is_empty() {
        return not_found();
    }
    let mut gn = BigInt::zero();
    let mut gd = BigInt::one();
    for q in &slopes {
        gn = num_integer::Integer::gcd(&gn, q.numer());
        gd = num_integer::Integer::lcm(&gd, q.denom());
    }
    let r = Q::new(gn, gd);
    if r.is_zero() || r.is_one() {
        return not_found();
    }
    let u = arena.symbol("__u_scaled");
    if let ExprNode::Symbol(u_sid) = *arena.node(u) {
        let declared = arena.symbol_assumptions(var_sid);
        arena.set_symbol_assumptions(u_sid, declared);
    }
    let r_id = arena.num_ratio(r.clone());
    let x_of_u = arena.div(u, r_id);
    let sub = crate::transforms::subs::subs(arena, expr, var, x_of_u);
    let sub = crate::transforms::eval::eval(arena, sub);
    if expr_contains_var(arena, sub, var) {
        return not_found();
    }
    solve_raw(arena, sub, u, cx).map_roots(|v| {
        let x = arena.div(v, r_id);
        let x = crate::transforms::expand::expand(arena, x);
        crate::transforms::eval::eval(arena, x)
    })
}

/// An equation in which the unknown `x` occurs only through linear forms
/// `aᵢ·x + bᵢ` with one common shift `s = bᵢ/aᵢ ≠ 0`
/// (`sin(2x − 1) + cos(2x − 1) + 1/2`, `sinh(x + 1/3) − tanh(x + 1/3)`,
/// `cosh(2x + 1)² + 3·cosh(2x + 1) + 2`, `sin(2x − 1) + sin(4x − 2)`), solved in
/// `u = x + s`, in which every form is `aᵢ·u`, and mapped back by
/// `x = u − s` (a family `u₀ + n·P` becomes `u₀ − s + n·P`).  Before 0.50
/// these were refused while the same equations in `aᵢ·x` were solved: the
/// substitutions look for `sin(a·x)`, `e^(a·x)`, … of the unknown itself.
/// SymPy solves them.
///
/// `u` is real when `x` is declared real; the shift must be real (per the
/// context's [`Params`]), since a shift by
/// `i` would make `u` non-real for a real `x`.  Refused (`NotFound`) for
/// another shape (`x` bare or outside the forms: `x + sin(x + 1)`), for no
/// shift, or for two different shifts (`sin(x + 1) + cos(x + 2)`).  The
/// equation in `u` has no shifted form left, so this step does not repeat.
fn try_shifted_argument(arena: &mut Arena, expr: ExprId, var: ExprId, cx: &SolveCx) -> Solutions {
    use crate::base::assumptions::{Assumptions, Props};
    let not_found = || Solutions::refused(Reason::NotFound);
    if !matches!(arena.node(var), ExprNode::Symbol(_)) {
        return not_found();
    }
    let Some(forms) = linear_forms_of(arena, expr, var) else {
        return not_found();
    };
    let mut shift = None;
    let mut rates: Vec<ExprId> = Vec::with_capacity(forms.len());
    for &form in &forms {
        let Some((a, b)) = linear_rate(arena, form, var) else {
            return not_found();
        };
        let s = arena.div(b, a);
        let s = crate::transforms::eval::eval(arena, s);
        match shift {
            None => shift = Some(s),
            Some(t) if t == s => {}
            Some(_) => return not_found(),
        }
        rates.push(a);
    }
    let Some(s) = shift else {
        return not_found();
    };
    if arena.is_zero_structural(s) || expr_contains_var(arena, s, var) || !cx.real(arena, s, var) {
        return not_found();
    }
    let u = crate::transforms::subs::fresh_symbol(arena, "__u_shifted", &[expr]);
    if let ExprNode::Symbol(u_sid) = *arena.node(u) {
        let mut declared = Assumptions::default();
        if Domain::declared(arena, var) == Domain::Real {
            declared.assert_true(Props::REAL);
        }
        arena.set_symbol_assumptions(u_sid, declared);
    }
    let mut sub = expr;
    for (&form, &a) in forms.iter().zip(&rates) {
        let au = arena.mul(&[a, u]);
        sub = crate::transforms::subs::subs(arena, sub, form, au);
    }
    if expr_contains_var(arena, sub, var) {
        return not_found();
    }
    let sub = crate::transforms::eval::eval(arena, sub);
    let outcome = solve_raw(arena, sub, u, cx);
    // A candidate at which the evaluated equation in `u` is undefined is
    // dropped here: evaluation can expose an infinite factor that the
    // original hides in an unevaluated function (`betainc_regularized(1, 1,
    // −u, ∞) = u + ∞`), where the root of another factor is `∞·0`, and the
    // check against the original (`solve_impl`), whose substitution makes
    // that product `0`, would pass it.  The other candidates are left to that
    // check.
    let outcome = match outcome.into_found() {
        Ok((roots, missing)) => {
            let mut kept = Vec::with_capacity(roots.len());
            for v in roots {
                let at = crate::transforms::subs::subs(arena, sub, u, v);
                let at = crate::transforms::eval::eval(arena, at);
                let undefined = matches!(
                    arena.node(at),
                    ExprNode::NaN
                        | ExprNode::ComplexInfinity
                        | ExprNode::Infinity
                        | ExprNode::NegInfinity
                );
                if !undefined {
                    kept.push(v);
                }
            }
            if kept.is_empty() {
                return not_found();
            }
            Solutions::with_missing(kept, missing)
        }
        Err(other) => other,
    };
    if !(outcome.has_roots()
        || outcome.is_none_proved()
        || outcome.is_nested_periods()
        || outcome.is_conditional())
    {
        return not_found();
    }
    tracing::debug!("solve: solved in the shifted unknown");
    // A reason in words names the unknown it was found for: `x`, not `u`.
    let (u_name, x_name) = (arena.display(u).to_string(), arena.display(var).to_string());
    let renamed = |reason: Reason| match reason {
        Reason::Said(text) if text.contains(&u_name) => {
            Reason::said(text.replace(&u_name, &x_name))
        }
        other => other,
    };
    let outcome = match outcome {
        Solutions::Complete(RootSet::Empty(reason)) => Solutions::none(renamed(reason)),
        Solutions::Partial(set, Missing::Alternative(reason)) => {
            Solutions::Partial(set, Missing::Alternative(renamed(reason)))
        }
        other => other,
    };
    outcome.map_roots(|v| {
        let x = arena.sub(v, s);
        let x = crate::transforms::expand::expand(arena, x);
        crate::transforms::eval::eval(arena, x)
    })
}

/// The maximal sums `a·x + b` (`x = var`, `a` and `b ≠ 0` free of it: an
/// `Add` of terms free of `var`, `var` and products of `var` with factors
/// free of it) through which `var` occurs in `expr`, or `None` when it
/// occurs elsewhere too (bare, or inside no such sum).
fn linear_forms_of(arena: &Arena, expr: ExprId, var: ExprId) -> Option<Vec<ExprId>> {
    let linear_term = |t: ExprId| {
        t == var
            || matches!(arena.node(t), ExprNode::Mul(fs)
                if fs.iter().filter(|&&f| f == var).count() == 1
                    && fs.iter().all(|&f| f == var || !expr_contains_var(arena, f, var)))
    };
    let mut forms = Vec::new();
    let mut stack = vec![expr];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) || !expr_contains_var(arena, id, var) {
            continue;
        }
        if id == var {
            return None;
        }
        if let ExprNode::Add(terms) = arena.node(id)
            && terms.iter().any(|&t| !expr_contains_var(arena, t, var))
            && terms
                .iter()
                .all(|&t| !expr_contains_var(arena, t, var) || linear_term(t))
        {
            forms.push(id);
            continue;
        }
        stack.extend(arena.children(id));
    }
    (!forms.is_empty()).then_some(forms)
}

/// Does `e` hold a function of `t` that is periodic for the variable of the
/// solve: `sin`, `cos`, `tan` always, exponentials and hyperbolic
/// functions for a variable that may be non-real
/// ([`SolveCx::variable_may_be_nonreal`]: [`solve_general_over_c`]).
fn has_periodic_atom(arena: &Arena, e: ExprId, t: ExprId, cx: &SolveCx) -> bool {
    let complex = cx.variable_may_be_nonreal(arena, t);
    crate::base::walk::post_order_ids(arena, e)
        .into_iter()
        .any(|id| match *arena.node(id) {
            ExprNode::Sin(a) | ExprNode::Cos(a) | ExprNode::Tan(a) => {
                crate::base::walk::contains(arena, a, t)
            }
            ExprNode::Exp(a) | ExprNode::Sinh(a) | ExprNode::Cosh(a) | ExprNode::Tanh(a) => {
                complex && crate::base::walk::contains(arena, a, t)
            }
            ExprNode::Pow(b, x) => {
                complex
                    && !crate::base::walk::contains(arena, b, t)
                    && crate::base::walk::contains(arena, x, t)
            }
            _ => false,
        })
}

/// Equations in `x` and rational powers `x^(p/q)` of `x` itself
/// (`√x = x − 2`, `x^(2/3) + x^(1/3) = 2`): with `L` the lcm of the
/// denominators and `t = x^(1/L)` (principal), `x^(p/q) = t^(pL/q)` and
/// `x = t^L`, so the equation becomes one in `t` alone; each root `t₀`
/// gives the candidate `x = t₀^L`.  The identity holds only for `t` in the
/// principal sector (`−π/L < arg t ≤ π/L`), so a root outside it yields a
/// spurious `x` (`t = −1` gives `x = 1`, where `√1 = 1 ≠ 1 − 2`); the
/// check of every candidate against the equation removes those.  A root
/// `t₀` with parameters, which that check cannot evaluate, is decided by
/// its argument range ([`principal_power_preimage`]): kept in the sector,
/// dropped outside it, and otherwise kept by a public solver under the
/// exact sector condition ([`principal_power_condition`],
/// [`Reason::Conditional`]; partial up to 0.52, and when the root has the
/// integer parameter of a family) or as a candidate
/// ([`SolveCx::conditional_roots_as_candidates`]).  Before 0.51
/// it was kept: `√x + y + 4 = 0` gave `x = (y + 4)²` (no root for `y = 0`:
/// `√x = −4`), `x^(2/3) = y` gave `±y^(3/2)` (`−y^(3/2)` is no root for `y =
/// 1`: `(−1)^(2/3) = e^(2πi/3)`).
///
/// `None` when `x` occurs other than bare or as the base of a rational
/// power, or when no power has a denominator.
///
/// The radicand may also be one linear `u = a·x + b` (`√(x + 1) = x − 1`,
/// `x − 2√(x − 1) = 0`): `t = u^(1/L)` and `x = (t^L − b)/a`.  SymPy:
/// `solve(sqrt(x + 1) - (x - 1))` → `[3]` (the candidate `0` of the squared
/// equation fails), `solve(x - 2*sqrt(x - 1))` → `[2]`; both were refused
/// before 0.39.
fn try_radical_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    match radical_substitution(arena, expr, var) {
        Some(sub) => sub.solve(arena, expr, var, cx),
        None => Solutions::refused(Reason::NotFound),
    }
}

/// The equation of [`try_radical_substitution`] in `t`, and `x` as a
/// function of `t`.
struct RadicalSubstitution {
    /// The new unknown `t`.
    t: ExprId,
    /// The equation in `t`.
    substituted: ExprId,
    /// `x = t^L`, or `(t^L − b)/a` for the radicand `a·x + b`.
    x_of_t: ExprId,
    /// The radicand is `a·x + b` rather than `x`: the candidates are
    /// checked strictly.
    linear: bool,
    /// `1/L`: `t` is the principal `L`-th root of the radicand.
    index: Q,
}

impl RadicalSubstitution {
    /// The roots in `x` from those of the equation in `t`.
    fn solve(&self, arena: &mut Arena, expr: ExprId, var: ExprId, cx: &SolveCx) -> Solutions {
        let Ok((t_roots, missing)) = solve_impl(arena, self.substituted, self.t, &cx.substituted())
            .without_conditions()
            .into_found()
        else {
            return Solutions::refused(Reason::NotFound);
        };
        let mut out: Vec<ExprId> = Vec::new();
        let mut off_branch = false;
        let mut undecided = false;
        // roots `t₀` in the principal sector only for some parameter values
        let mut guarded: Vec<Guarded> = Vec::new();
        for ts in t_roots {
            let parametric = crate::base::walk::free_symbols(arena, ts)
                .into_iter()
                .any(|s| Some(s) != cx.period());
            let mut cond = None;
            if parametric {
                match principal_power_preimage(arena, ts, &self.index) {
                    crate::base::truth::Truth::True => {}
                    crate::base::truth::Truth::False => {
                        off_branch = true;
                        continue;
                    }
                    crate::base::truth::Truth::Unknown if cx.conditional_roots_as_candidates() => {}
                    crate::base::truth::Truth::Unknown => {
                        let with_period = cx
                            .period()
                            .is_some_and(|p| crate::base::walk::contains(arena, ts, p));
                        match principal_power_condition(arena, ts, &self.index) {
                            Some(c) if !with_period => cond = Some(c),
                            _ => {
                                undecided = true;
                                continue;
                            }
                        }
                    }
                }
            }
            let x0 = crate::transforms::subs::subs(arena, self.x_of_t, self.t, ts);
            let x0 = crate::transforms::expand::expand(arena, x0);
            let x0 = crate::transforms::eval::eval(arena, x0);
            if let Some(cond) = cond {
                guarded.push(Guarded {
                    cond,
                    roots: RootSet::Points(vec![x0]),
                });
            } else if !out.contains(&x0) {
                out.push(x0);
            }
        }
        // no other root: every root in `t` was taken
        let only_guarded = out.is_empty() && missing.is_none() && !undecided;
        let outcome = self.unconditional(arena, expr, var, out, missing, off_branch, undecided);
        if guarded.is_empty() {
            return outcome;
        }
        let outcome = if only_guarded {
            Solutions::none(Reason::NoPreimage)
        } else {
            outcome
        };
        outcome.union(Solutions::refused(Reason::Conditional(guarded)))
    }

    /// The outcome of the roots `out` that are principal for every value of
    /// the parameters (see [`solve`](Self::solve)).
    #[allow(clippy::too_many_arguments)]
    fn unconditional(
        &self,
        arena: &mut Arena,
        expr: ExprId,
        var: ExprId,
        out: Vec<ExprId>,
        missing: Option<Missing>,
        off_branch: bool,
        undecided: bool,
    ) -> Solutions {
        let missing = if undecided {
            Missing::merged(
                missing,
                Some(Missing::Alternative(Reason::said(
                    "a root of the equation in the radical is a principal root only for some \
                     values of the parameters",
                ))),
            )
        } else {
            missing
        };
        if out.is_empty() && missing.is_none() && off_branch {
            return Solutions::none(Reason::said(
                "no root of the equation in the radical is a principal root",
            ));
        }
        if self.linear {
            return match strictly_checked(arena, expr, var, out) {
                Some(roots) => Solutions::with_missing(roots, missing),
                None => Solutions::refused(Reason::NotFound),
            };
        }
        Solutions::with_missing(out, missing)
    }
}

/// The substitution of [`try_radical_substitution`], or `None` for another
/// shape.
fn radical_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Option<RadicalSubstitution> {
    let post = crate::base::walk::post_order_ids(arena, expr);
    // The one base of the fractional powers.
    let mut radicand: Option<ExprId> = None;
    for &id in &post {
        if let ExprNode::Pow(b, e) = *arena.node(id)
            && expr_contains_var(arena, b, var)
        {
            let r = arena.as_num(e)?;
            if !r.is_integer() {
                match radicand {
                    None => radicand = Some(b),
                    Some(u) if u == b => {}
                    Some(_) => return None,
                }
            }
        }
    }
    let u = radicand?;
    let rate = if u == var {
        None
    } else {
        Some(linear_rate(arena, u, var)?)
    };
    let mut lcm = BigInt::one();
    let mut powers: Vec<(ExprId, Q)> = Vec::new();
    for &id in &post {
        if !expr_contains_var(arena, id, var) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Pow(b, e) if *b == u => {
                let r = arena.as_num(*e)?.clone();
                if !r.is_integer() {
                    lcm = num_integer::Integer::lcm(&lcm, r.denom());
                }
                powers.push((id, r));
            }
            // `x` elsewhere must be bare: a power's exponent or a function
            // argument that is not `x` or `x^(p/q)` is not handled here.
            ExprNode::Pow(_, e) if expr_contains_var(arena, *e, var) => return None,
            ExprNode::Symbol(_)
            | ExprNode::Add(_)
            | ExprNode::Mul(_)
            | ExprNode::Neg(_)
            | ExprNode::Pow(..) => {}
            _ => return None,
        }
    }
    if lcm.is_one() || lcm > BigInt::from(MAX_RADICAL_INDEX) {
        return None;
    }
    let t = arena.symbol("__t_radical");
    let l_id = arena.big_int(lcm.clone());
    let mut substituted = expr;
    for (pow_id, r) in &powers {
        let k = r * Ratio::from_integer(lcm.clone());
        let k_id = arena.num_ratio(k);
        let tk = arena.pow(t, k_id);
        substituted = arena.subs_structural(substituted, *pow_id, tk);
    }
    let t_l = arena.pow(t, l_id);
    // x = t^L, or (t^L − b)/a for the radicand a·x + b
    let x_of_t = match rate {
        None => t_l,
        Some((a, b)) => {
            let num = arena.sub(t_l, b);
            arena.div(num, a)
        }
    };
    let index = Ratio::new(BigInt::one(), lcm);
    substituted = crate::transforms::subs::subs(arena, substituted, var, x_of_t);
    if rate.is_some() {
        substituted = crate::transforms::expand::expand(arena, substituted);
    }
    let substituted = crate::transforms::eval::eval(arena, substituted);
    if expr_contains_var(arena, substituted, var) {
        return None;
    }
    Some(RadicalSubstitution {
        t,
        substituted,
        x_of_t,
        linear: rate.is_some(),
        index,
    })
}

/// Largest root index `L` of [`try_radical_substitution`].
const MAX_RADICAL_INDEX: i64 = 12;

/// Equations polynomial in `x` and in the square roots `√A`, `√B` of one
/// or two polynomials (`√(x + 5) + √x = 3`, `√(x² + 1) = x + 2`), as
/// SymPy's `unrad`: with `s = √A`, `r = √B` the equation is
/// `P = a + b·s + c·r + d·s·r = 0` (`a, b, c, d` polynomials, `s² = A`,
/// `r² = B`), and every solution is a root of the norm
/// `P(s, r)·P(−s, −r)·P(−s, r)·P(s, −r) = E² − F²·A·B` with
/// `E = a² + d²AB − b²A − c²B`, `F = 2(ad − bc)`.  Its roots are the
/// candidates; those of the other sign choices fail the equation and are
/// removed by the check of every candidate.  SymPy:
/// `solve(sqrt(x + 5) + sqrt(x) - 3)` → `[4/9]`; refused before 0.39.
fn try_square_root_elimination(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Option<Unradicated> {
    // The radicands, and the half-integer powers of them.
    let mut radicands: Vec<ExprId> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if !expr_contains_var(arena, id, var) {
            continue;
        }
        match arena.node(id).clone() {
            ExprNode::Pow(b, e) => {
                let r = arena.as_num(e)?.clone();
                if r.is_integer() {
                    continue;
                }
                if *r.denom() != BigInt::from(2) || expr_contains_var(arena, e, var) {
                    return None;
                }
                polybridge::expr_to_poly(arena, b, var)?;
                if !radicands.contains(&b) {
                    radicands.push(b);
                }
            }
            ExprNode::Symbol(_) | ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => {}
            _ => return None,
        }
    }
    if radicands.len() > 2 && radicands.len() <= MAX_SQRT_RADICANDS {
        return eliminate_many_square_roots(arena, expr, var, &radicands);
    }
    if radicands.is_empty() || radicands.len() > 2 {
        return None;
    }
    // s, r stand for √A, √B; A^(k/2) = A^((k−1)/2)·s for odd k.
    let s = arena.symbol("__s_unrad");
    let r = arena.symbol("__r_unrad");
    let syms = [s, r];
    let mut replaced = expr;
    for id in crate::base::walk::post_order_ids(arena, expr) {
        let ExprNode::Pow(b, e) = *arena.node(id) else {
            continue;
        };
        let Some(k) = arena.as_num(e).cloned() else {
            continue;
        };
        let Some(which) = radicands.iter().position(|&u| u == b) else {
            continue;
        };
        if k.is_integer() {
            continue;
        }
        // k = m + 1/2 with m = floor(k)
        let m = k.floor();
        let m_id = arena.num_ratio(m);
        let bm = arena.pow(b, m_id);
        let repl = arena.mul(&[bm, syms[which]]);
        replaced = arena.subs_structural(replaced, id, repl);
    }
    let replaced = crate::transforms::expand::expand(arena, replaced);
    let replaced = crate::transforms::eval::eval(arena, replaced);
    let (a_rad, b_rad) = (
        radicands[0],
        radicands.get(1).copied().unwrap_or(arena.zero),
    );
    // Coefficients of 1, s, r, s·r.
    let by_s = symbolic_poly_coeffs(arena, replaced, s)?;
    if by_s.len() > 2 {
        return None;
    }
    let zero = arena.zero;
    let s0 = by_s.first().copied().unwrap_or(zero);
    let s1 = by_s.get(1).copied().unwrap_or(zero);
    let split_r = |arena: &mut Arena, e: ExprId| -> Option<(ExprId, ExprId)> {
        let c = symbolic_poly_coeffs(arena, e, r)?;
        if c.len() > 2 {
            return None;
        }
        Some((
            c.first().copied().unwrap_or(zero),
            c.get(1).copied().unwrap_or(zero),
        ))
    };
    let (a, c) = split_r(arena, s0)?;
    let (b, d) = split_r(arena, s1)?;
    for &coef in &[a, b, c, d] {
        if crate::base::walk::contains(arena, coef, s)
            || crate::base::walk::contains(arena, coef, r)
        {
            return None;
        }
    }
    // E = a² + d²AB − b²A − c²B, F = 2(ad − bc), N = E² − F²AB
    let two = arena.int(2);
    let sq = |arena: &mut Arena, e: ExprId| arena.pow(e, two);
    let ab = arena.mul(&[a_rad, b_rad]);
    let a2 = sq(arena, a);
    let d2 = sq(arena, d);
    let b2 = sq(arena, b);
    let c2 = sq(arena, c);
    let d2ab = arena.mul(&[d2, ab]);
    let b2a = arena.mul(&[b2, a_rad]);
    let c2b = arena.mul(&[c2, b_rad]);
    let neg_b2a = arena.neg(b2a);
    let neg_c2b = arena.neg(c2b);
    let e_big = arena.add(&[a2, d2ab, neg_b2a, neg_c2b]);
    let ad = arena.mul(&[a, d]);
    let bc = arena.mul(&[b, c]);
    let ad_bc = arena.sub(ad, bc);
    let f_big = arena.mul(&[two, ad_bc]);
    let e2 = sq(arena, e_big);
    let f2 = sq(arena, f_big);
    let f2ab = arena.mul(&[f2, ab]);
    let norm = arena.sub(e2, f2ab);
    let norm = crate::transforms::expand::expand(arena, norm);
    let norm = crate::transforms::eval::eval(arena, norm);
    let poly = polybridge::expr_to_poly(arena, norm, var)?;
    if poly.is_constant() {
        if !poly.is_zero() {
            return Some(Unradicated::NormConstant);
        }
        return None;
    }
    let candidates = values(solve_rational_poly(arena, var, &poly));
    strictly_checked(arena, expr, var, candidates).map(Unradicated::Roots)
}

/// What [`try_square_root_elimination`] found.
enum Unradicated {
    /// The candidates that certainly solve the equation (possibly none:
    /// not taken as a proof).
    Roots(Vec<ExprId>),
    /// The norm of the equation over the sign choices is a nonzero
    /// constant: the equation has no root.  (Up to 0.48 the thread-local
    /// flag `SQRT_NORM_CONSTANT` next to an empty list.)
    NormConstant,
}

/// Most square roots of polynomials eliminated by
/// [`eliminate_many_square_roots`].
const MAX_SQRT_RADICANDS: usize = 4;

/// Largest degree of the norm of [`eliminate_many_square_roots`].
const MAX_SQRT_NORM_DEGREE: usize = 64;

/// [`try_square_root_elimination`] for three or four radicands `Aᵢ`: with
/// `sᵢ = √Aᵢ` the equation is a polynomial `P` in `x` and the `sᵢ`, of
/// degree one in each after `sᵢ² = Aᵢ`; writing `P = P₀ + P₁·sₖ`, the
/// product with its conjugate `P₀ − P₁·sₖ` is `P₀² − P₁²·Aₖ`, free of `sₖ`
/// — repeated for every `sᵢ`, the norm over all sign choices, a polynomial
/// in `x` that every solution is a root of.  Its roots are the candidates,
/// each checked strictly.  `√(2x + 9) = √(x + 1) + √(x + 4)` gives `0`
/// (refused before 0.47; SymPy: `[0]`).
fn eliminate_many_square_roots(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    radicands: &[ExprId],
) -> Option<Unradicated> {
    let syms: Vec<ExprId> = (0..radicands.len())
        .map(|i| arena.symbol(&format!("__s_unrad_{i}")))
        .collect();
    let mut replaced = expr;
    for id in crate::base::walk::post_order_ids(arena, expr) {
        let ExprNode::Pow(b, e) = *arena.node(id) else {
            continue;
        };
        let Some(k) = arena.as_num(e).cloned() else {
            continue;
        };
        let Some(which) = radicands.iter().position(|&u| u == b) else {
            continue;
        };
        if k.is_integer() {
            continue;
        }
        let m = k.floor();
        let m_id = arena.num_ratio(m);
        let bm = arena.pow(b, m_id);
        let repl = arena.mul(&[bm, syms[which]]);
        replaced = arena.subs_structural(replaced, id, repl);
    }
    // `p` with every `sⱼ^m` reduced by `sⱼ² = Aⱼ` (degree ≤ 1 in each).
    let reduce = |arena: &mut Arena, p: ExprId| -> Option<ExprId> {
        let mut p = crate::transforms::expand::expand(arena, p);
        for (j, &s) in syms.iter().enumerate() {
            if !crate::base::walk::contains(arena, p, s) {
                continue;
            }
            let coeffs = symbolic_poly_coeffs(arena, p, s)?;
            let mut terms: Vec<ExprId> = Vec::with_capacity(coeffs.len());
            for (m, &c) in coeffs.iter().enumerate() {
                let half = arena.int((m / 2) as i64);
                let am = arena.pow(radicands[j], half);
                let t = if m % 2 == 1 {
                    arena.mul(&[c, am, s])
                } else {
                    arena.mul(&[c, am])
                };
                terms.push(t);
            }
            let sum = arena.add(&terms);
            p = crate::transforms::expand::expand(arena, sum);
        }
        Some(crate::transforms::eval::eval(arena, p))
    };
    let mut p = reduce(arena, replaced)?;
    for (i, &s) in syms.iter().enumerate().rev() {
        let coeffs = symbolic_poly_coeffs(arena, p, s)?;
        if coeffs.len() > 2 {
            return None;
        }
        let zero = arena.zero;
        let p0 = coeffs.first().copied().unwrap_or(zero);
        let p1 = coeffs.get(1).copied().unwrap_or(zero);
        let two = arena.int(2);
        let p0_2 = arena.pow(p0, two);
        let p1_2 = arena.pow(p1, two);
        let p1_2a = arena.mul(&[p1_2, radicands[i]]);
        let n = arena.sub(p0_2, p1_2a);
        p = reduce(arena, n)?;
        if crate::base::walk::contains(arena, p, s) {
            return None;
        }
    }
    let poly = polybridge::expr_to_poly(arena, p, var)?;
    if poly.is_constant() && !poly.is_zero() {
        return Some(Unradicated::NormConstant);
    }
    if poly.is_constant() || poly.degree().unwrap_or(0) > MAX_SQRT_NORM_DEGREE {
        return None;
    }
    let candidates = values(solve_rational_poly(arena, var, &poly));
    strictly_checked(arena, expr, var, candidates).map(Unradicated::Roots)
}

/// The candidates that certainly solve `expr = 0` (whose only symbol is
/// `var`), or `None` when one of them can be decided neither way: the
/// candidates of squaring away radicals include roots of the other sign
/// choices, and one whose residual sits on a branch cut (`√` of a negative
/// number written in nested radicals) must not be returned unverified.
/// (No candidate left is not taken as a proof that there is no root.)
fn strictly_checked(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    candidates: Vec<ExprId>,
) -> Option<Vec<ExprId>> {
    if crate::base::walk::free_symbols(arena, expr)
        .iter()
        .any(|&s| s != var)
    {
        return None;
    }
    let mut out = Vec::new();
    for c in candidates {
        if member_is_root(arena, expr, var, c) {
            out.push(c);
        } else if !certainly_not_a_root(arena, expr, var, c) {
            return None;
        }
    }
    Some(out)
}

// ═══════════════════════════════════════════════════════════════════════════
// Rational equations, exponential and half-angle substitutions, log sums
// ═══════════════════════════════════════════════════════════════════════════
//
// Each strategy reduces the equation to one it solves already and returns
// candidates; every candidate of a non-polynomial equation is then checked
// against the original by `drop_certain_non_roots` (a certified nonzero or
// undefined residual removes it), as SymPy's `checksol`.

/// Walk `expr` top-down through `Add`, `Mul`, `Neg` and integer powers of
/// subexpressions containing `var`, calling `atom` on every other node that
/// contains `var`; `None` as soon as `atom` rejects one.
fn walk_var_atoms(
    arena: &Arena,
    expr: ExprId,
    var: ExprId,
    mut atom: impl FnMut(ExprId) -> bool,
) -> Option<()> {
    let mut stack = vec![expr];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) || !expr_contains_var(arena, id, var) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Add(ch) | ExprNode::Mul(ch) => stack.extend(ch.iter().copied()),
            ExprNode::Neg(b) => stack.push(*b),
            ExprNode::Pow(b, e)
                if !expr_contains_var(arena, *e, var)
                    && arena.as_num(*e).is_some_and(|r| r.is_integer()) =>
            {
                stack.push(*b)
            }
            _ => {
                if !atom(id) {
                    return None;
                }
            }
        }
    }
    Some(())
}

/// Rational equations `P(x)/Q(x) = 0` (the variable only in sums, products
/// and integer powers, at least one of them negative): the numerator of
/// the combined fraction is solved, and roots of the denominator are
/// dropped.  (`x + 1/x = 3` was refused before 0.31; SymPy:
/// `solve(x + 1/x - 3)` → `[3/2 − √5/2, √5/2 + 3/2]`.)
///
/// When `P` and `Q` have rational coefficients, every factor `P` shares
/// with `Q` is divided out of `P` first (`P/gcd(P, Q)` until the gcd is
/// constant): the roots of what is left are exactly the solutions, and none
/// of them is a pole.  Before 0.39 a root of `P` was tested against `Q` by
/// substitution, and a common root in nested radicals or a `RootOf` was
/// not recognised as a zero of `Q`: `(x⁴ − 2x³ − 3x² + 7x − 2)/(x³ − 3x + 1)`
/// (`= x − 2` off the poles) gave `2` and the three poles.  A numerator
/// left constant proves there is no solution (`(x² − 1)/(x − 1) = 2`;
/// SymPy: `[]`).
fn try_solve_rational(arena: &mut Arena, expr: ExprId, var: ExprId, cx: &SolveCx) -> Solutions {
    let not_found = || Solutions::refused(Reason::NotFound);
    if !is_rational_function_of(arena, expr, var) {
        return not_found();
    }
    let (num, den) = arena.as_numer_denom_expr(expr);
    let num = crate::transforms::expand::expand(arena, num);
    let num = crate::transforms::eval::eval(arena, num);
    if !expr_contains_var(arena, num, var) {
        // A numerator that is a nonzero constant: no root.  Up to 0.51 not
        // found, and the half-angle form `((1 − t²)² + 2(1 − t²)(1 + t²) +
        // (1 + t²)²)/(1 + t²)²` of `cos(−x)² + 2cos(−x) + 1` (numerator 4)
        // left `solve` partial (`[π]` and an unsolved alternative).
        if arena.is_symbol_free(num)
            && crate::base::truth::Truth::from(crate::poly::algebraic::is_zero_checked(arena, num))
                .is_false()
        {
            return Solutions::none(Reason::said(
                "the equation is a fraction with a nonzero constant numerator",
            ));
        }
        return not_found();
    }
    // The numerator is a polynomial: no negative power of `var` is left.
    let still_rational = crate::base::walk::post_order_ids(arena, num)
        .into_iter()
        .any(|id| {
            matches!(arena.node(id), ExprNode::Pow(b, e)
            if expr_contains_var(arena, *b, var)
                && arena.as_num(*e).is_some_and(|r| r.is_negative()))
        });
    if still_rational {
        return not_found();
    }
    if let (Some(p), Some(q)) = (
        polybridge::expr_to_poly(arena, num, var),
        polybridge::expr_to_poly(arena, den, var),
    ) && !q.is_zero()
        && !p.is_zero()
    {
        let p = without_common_factors(p, &q);
        if p.is_constant() {
            return Solutions::none(Reason::said(
                "every root of the numerator is a pole (a root of the denominator)",
            ));
        }
        return Solutions::found(values(solve_rational_poly(arena, var, &p)));
    }
    // Coefficients with parameters: the common factors divided out over
    // ℚ[x, parameters] (`(x² − a²)/(x − a) = 0` is `x + a = 0`; SymPy
    // `[-a]`, before 0.39 `±√(a²)`, the pole `a` among them).
    let num = match without_common_factors_multi(arena, num, den, var) {
        Some(reduced) if !expr_contains_var(arena, reduced, var) => {
            return Solutions::none(Reason::said(
                "every root of the numerator is a pole (a root of the denominator)",
            ));
        }
        Some(reduced) => reduced,
        None => num,
    };
    // (No root of the numerator, or none found: nothing found here; the
    // next strategies may still answer.)
    let (candidates, missing) = match solve_raw(arena, num, var, cx).into_found() {
        Ok(found) => found,
        Err(nested) if nested.is_nested_periods() => return nested,
        // (the poles among conditional roots: `without_guarded_non_roots`)
        Err(conditional) if conditional.is_conditional() => return conditional,
        Err(_) => return not_found(),
    };
    let mut out = Vec::with_capacity(candidates.len());
    for s in candidates {
        let d = crate::transforms::subs::subs(arena, den, var, s);
        let d = crate::transforms::expand::expand(arena, d);
        let d = crate::transforms::eval::eval(arena, d);
        if arena.is_zero_structural(d) {
            continue;
        }
        // A constant denominator value is tested exactly (minimal
        // polynomial of an algebraic number): `(x² − 2x − 1)/(x − 1 − √2)`
        // has the pole `1 + √2` among the roots of its numerator.
        if crate::base::walk::free_symbols(arena, d).is_empty() {
            match crate::poly::algebraic::is_zero_checked(arena, d) {
                Some(true) => continue,
                Some(false) => {}
                None => {
                    return Solutions::refused(Reason::said(format!(
                        "cannot decide whether the root {} of the numerator is a pole",
                        arena.display(s)
                    )));
                }
            }
        }
        out.push(s);
    }
    Solutions::with_missing(out, missing)
}

/// Polynomials in `var` whose coefficients are polynomials in other
/// symbols, of degree ≥ 3 in `var`: factored over ℤ in all the symbols
/// (`a·x³ − (2a² + a)x² + … = a·(x − a)·(x − a − 1)·(x + 1)`), and the
/// answer is the union of the roots of the factors when every factor
/// containing `var` is solved.  SymPy's `solve` factors the same way;
/// before 0.39 the expanded form was refused ("coefficients not all in
/// ℚ(√p…, i)").  `None` when the polynomial does not split or a factor
/// is not solved.
fn try_solve_parametric_factors(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    let not_found = || Solutions::refused(Reason::NotFound);
    let Some(coeffs) = symbolic_poly_coeffs(arena, expr, var) else {
        return not_found();
    };
    if coeffs.len() < 4 || coeffs.len() > 9 {
        return not_found();
    }
    let factored = arena.factor_expr(expr, var);
    if factored == expr {
        return not_found();
    }
    let parts: Vec<ExprId> = match arena.node(factored) {
        ExprNode::Mul(ch) => ch.to_vec(),
        _ => vec![factored],
    };
    let mut with_var: Vec<ExprId> = Vec::new();
    for p in parts {
        let base = match *arena.node(p) {
            ExprNode::Pow(b, e)
                if arena
                    .as_num(e)
                    .is_some_and(|r| r.is_integer() && r.is_positive()) =>
            {
                b
            }
            _ => p,
        };
        if expr_contains_var(arena, base, var) && !with_var.contains(&base) {
            with_var.push(base);
        }
    }
    // Every factor solved (roots found; a refused branch makes the whole
    // partial), or nothing.
    let mut out: Option<Solutions> = None;
    for f in with_var {
        if f == expr {
            return not_found();
        }
        let factor = solve_raw(arena, f, var, cx);
        if factor.is_nested_periods() {
            return factor;
        }
        if !factor.has_roots() {
            return not_found();
        }
        out = Some(match out {
            Some(roots) => roots.union(factor),
            None => factor,
        });
    }
    out.unwrap_or_else(not_found)
}

/// Is the product `expr` a quotient of polynomials over ℚ in all its
/// symbols (at most four), so that [`without_common_factors_multi`] applies?
fn rational_over_q_in_all_symbols(arena: &Arena, expr: ExprId) -> bool {
    let ExprNode::Mul(children) = arena.node(expr) else {
        return false;
    };
    let mut syms = crate::base::walk::free_symbols(arena, expr);
    syms.sort_by(|&a, &b| arena.canonical_cmp(a, b));
    syms.dedup();
    if syms.len() > 4 {
        return false;
    }
    children.iter().all(|&c| {
        let base = match arena.node(c) {
            ExprNode::Pow(b, e) if arena.as_num(*e).is_some_and(|r| r.is_integer()) => *b,
            _ => c,
        };
        polybridge::expr_to_multipoly(arena, base, &syms).is_some()
    })
}

/// The candidates (roots of factors of the product `expr`) that are not
/// poles: each denominator factor (`b^(−k)` containing `var`) at a
/// candidate free of symbols is tested exactly (`is_zero_checked`: certified
/// digits, then the minimal polynomial of an algebraic number); one that is
/// infinite there (`tan(−x) − 2` at `π/2`) leaves `expr` undefined.  Up to
/// 0.51 that was undecided: `solve(cos(3x)/(tan(−x) − 2))` was refused
/// ("cannot decide whether 2*atan(-1) is a pole").  So is each argument `b`
/// of a Lambert W branch `W_k(b)`, `k ≠ 0` (at any candidate: `W_k(0) =
/// −∞`).  `Err` when a test is undecided.
fn drop_poles_of_product(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    candidates: Vec<ExprId>,
) -> Result<Vec<ExprId>, String> {
    if !matches!(arena.node(expr), ExprNode::Mul(_)) {
        return Ok(candidates);
    }
    // Every denominator, also inside a factor: up to 0.46 the roots of
    // `d = x⁵ − x + 1` came back from `d·(1 + 1/d) = 0` (`0·∞` there; SymPy
    // returns the roots of `d + 1` only).
    let mut dens: Vec<ExprId> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if let ExprNode::Pow(b, e) = *arena.node(id)
            && arena.as_num(e).is_some_and(|r| r.is_negative())
            && expr_contains_var(arena, b, var)
            && !dens.contains(&b)
        {
            dens.push(b);
        }
    }
    let lambert_args = lambert_branch_poles(arena, expr, var);
    if dens.is_empty() && lambert_args.is_empty() {
        return Ok(candidates);
    }
    let mut out = Vec::with_capacity(candidates.len());
    'cand: for s in candidates {
        // A Lambert argument with parameters left is nonzero generically, as
        // a denominator of a rational function with parameters (`x + y` at
        // `x = 0`); only an exact zero makes the candidate a pole.
        for &(b, direct) in &lambert_args {
            let d = crate::transforms::subs::subs(arena, b, var, s);
            let d = crate::transforms::expand::expand(arena, d);
            let d = crate::transforms::eval::eval(arena, d);
            let zero = if arena.is_zero_structural(d) {
                Some(true)
            } else if arena.is_symbol_free(d) {
                crate::poly::algebraic::is_zero_checked(arena, d)
            } else {
                Some(false)
            };
            match zero {
                Some(true) if direct => continue 'cand,
                Some(false) => {}
                _ => {
                    return Err(format!(
                        "cannot decide whether {} is a pole",
                        arena.display(s)
                    ));
                }
            }
        }
        if !crate::base::walk::free_symbols(arena, s).is_empty() {
            out.push(s);
            continue;
        }
        for &b in &dens {
            // `RootOf(q, k)` (of an irreducible `q`) is a zero of the
            // polynomial `b` exactly when `q` divides it.
            if let ExprNode::RootOf(q, qv, _) = *arena.node(s)
                && let (Some(qp), Some(bp)) = (
                    polybridge::expr_to_poly(arena, q, qv),
                    polybridge::expr_to_poly(arena, b, var),
                )
            {
                let g = Poly::gcd(&qp, &bp).degree().unwrap_or(0);
                if g > 0 && Some(g) == qp.degree() {
                    continue 'cand;
                }
                if g == 0 {
                    continue;
                }
            }
            let d = crate::transforms::subs::subs(arena, b, var, s);
            let d = crate::transforms::expand::expand(arena, d);
            let d = crate::transforms::eval::eval(arena, d);
            if arena.is_zero_structural(d) || is_undefined_value(arena, d) {
                continue 'cand;
            }
            match crate::poly::algebraic::is_zero_checked(arena, d) {
                Some(true) => continue 'cand,
                Some(false) => {}
                None => {
                    return Err(format!(
                        "cannot decide whether {} is a pole",
                        arena.display(s)
                    ));
                }
            }
        }
        out.push(s);
    }
    Ok(out)
}

/// The arguments `b` (with `var`) of the Lambert W branches `W_k(b)`, `k ≠
/// 0`, in `expr`, each with whether it is reached through sums, products
/// and positive powers only.  Where `b = 0` the branch is `−∞`, which the
/// canonical product `0·W_k(0) = 0` does not show (`canon_mul` folds
/// `W_k(0)` only later, in `eval`): reached so, the term is infinite there
/// and the point is a pole; inside a function or an exponent the infinity
/// may vanish (`e^{W₁(0)} = 0`) or not, undecided.  Since 0.52, when `W_k`
/// is inverted, `x·W₁(x) = 0` would otherwise have the root 0.
fn lambert_branch_poles(arena: &Arena, expr: ExprId, var: ExprId) -> Vec<(ExprId, bool)> {
    let mut out: Vec<(ExprId, bool)> = Vec::new();
    let mut stack = vec![(expr, true)];
    while let Some((id, direct)) = stack.pop() {
        match arena.node(id) {
            ExprNode::Add(c) | ExprNode::Mul(c) => {
                stack.extend(c.iter().map(|&ch| (ch, direct)));
            }
            ExprNode::Neg(inner) => stack.push((*inner, direct)),
            ExprNode::Pow(b, e) => {
                let positive = arena.as_num(*e).is_some_and(|q| q.is_positive());
                stack.push((*b, direct && positive));
                stack.push((*e, false));
            }
            node => {
                if let ExprNode::Lib(crate::base::libfn::LibFn::LambertW, args) = node
                    && let [b, k] = args[..]
                    && !arena.as_num(k).is_some_and(|q| q.is_zero())
                    && expr_contains_var(arena, b, var)
                {
                    match out.iter_mut().find(|(seen, _)| *seen == b) {
                        Some(entry) => entry.1 |= direct,
                        None => out.push((b, direct)),
                    }
                }
                node.for_each_child(|c| stack.push((c, false)));
            }
        }
    }
    out
}

/// Does `var` occur in `expr` only in sums, products and integer powers,
/// at least one of them negative (a rational function of `var` that is
/// not a polynomial)?
fn is_rational_function_of(arena: &Arena, expr: ExprId, var: ExprId) -> bool {
    let mut negative = false;
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if !expr_contains_var(arena, id, var) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Symbol(_) | ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => {}
            ExprNode::Pow(_, e) => match arena.as_num(*e) {
                Some(r) if r.is_integer() => negative |= r.is_negative(),
                _ => return false,
            },
            _ => return false,
        }
    }
    negative
}

/// [`without_common_factors`] for polynomials in `var` and up to three
/// parameters with rational coefficients (multivariate gcd over ℤ).  `None`
/// when they are not such polynomials.
fn without_common_factors_multi(
    arena: &mut Arena,
    num: ExprId,
    den: ExprId,
    var: ExprId,
) -> Option<ExprId> {
    let mut syms = crate::base::walk::free_symbols(arena, num);
    syms.extend(crate::base::walk::free_symbols(arena, den));
    syms.sort_by(|&a, &b| arena.canonical_cmp(a, b));
    syms.dedup();
    if !syms.contains(&var) || syms.len() < 2 || syms.len() > 4 {
        return None;
    }
    let mut n = polybridge::expr_to_multipoly(arena, num, &syms)?;
    let d = polybridge::expr_to_multipoly(arena, den, &syms)?;
    if d.is_zero() || n.is_zero() {
        return None;
    }
    let mut changed = false;
    for _ in 0..64 {
        let g = crate::poly::multipoly::MultiPoly::gcd(&n, &d);
        if g.total_degree().unwrap_or(0) == 0 {
            break;
        }
        n = n.div_exact(&g)?;
        changed = true;
    }
    changed.then(|| polybridge::multipoly_to_expr(arena, &n, &syms))
}

/// `p` with every irreducible factor it shares with `q` removed (all of
/// its multiplicity): `p / gcd(p, q)` repeated until the gcd is constant.
/// The roots of the result are the roots of `p` that are not roots of `q`.
fn without_common_factors(mut p: Poly, q: &Poly) -> Poly {
    if q.is_zero() {
        return p;
    }
    loop {
        let g = Poly::gcd(&p, q);
        if g.degree().unwrap_or(0) == 0 || p.is_zero() {
            return p;
        }
        p = p.div(&g);
    }
}

/// `(a, b)` with `arg = a·var + b`, `a ≠ 0` and `b` free of `var`.
fn linear_rate(arena: &mut Arena, arg: ExprId, var: ExprId) -> Option<(ExprId, ExprId)> {
    let coeffs = symbolic_poly_coeffs(arena, arg, var)?;
    if coeffs.len() != 2 || arena.is_zero_structural(coeffs[1]) {
        return None;
    }
    Some((coeffs[1], coeffs[0]))
}

/// `c = r^k` with `r ≥ 2` not a perfect power (`c` a rational > 0 that
/// is an integer or the reciprocal of one; `k` negative for `c < 1`).
fn perfect_power_base(c: &Q) -> Option<(BigInt, i64)> {
    if !c.is_positive() || c.is_one() {
        return None;
    }
    let (n, sign) = if c.is_integer() {
        (c.to_integer(), 1i64)
    } else if c.numer().is_one() {
        (c.denom().clone(), -1i64)
    } else {
        return None;
    };
    let bits = n.bits();
    for k in (2..=bits.min(64)).rev() {
        let k32 = u32::try_from(k).ok()?;
        let r = n.nth_root(k32);
        if num_traits::Pow::pow(&r, k32) == n {
            return Some((r, sign * i64::try_from(k).ok()?));
        }
    }
    Some((n, sign))
}

/// Equations in exponentials of one rate: every occurrence of the variable
/// is in `exp(a·x + b)` or `c^(a·x + b)` (`c` a rational base), the rates
/// `a` rational multiples of one constant `a₀` (`i` in `e^(ix) + e^(−ix)`).
/// With `t = exp(g·a₀·x)` (or `r^(g·a₀·x)` for the common base `r` of the
/// `c`, e.g. 2 for 4 and 8) and `g` the gcd of the rational ratios, each
/// exponential is a constant times an integer power of `t`; the equation in
/// `t` is solved (a Laurent polynomial by [`try_solve_rational`]), and `t`
/// is peeled back.  SymPy: `solve(exp(x) + exp(-x) - 3)` →
/// `[log(3/2 − √5/2), log(√5/2 + 3/2)]`, `solve(2**(2*x) - 5*2**x + 6)`
/// → `[1, log(3)/log(2)]`; both were refused before 0.31.
///
/// `NoSolution` when the equation in `t` is a polynomial over ℚ (its roots
/// all found) and no root has a real preimage (`eˣ + 3e^(x/2) + 2 = 0`:
/// `t = −1, −2`, `exp` of a real argument is positive), as peeling proves
/// `eˣ = −1` has no real solution; before 0.44 such equations were
/// refused.
fn try_exponential_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    exponential_substitution(arena, expr, var, cx)
        .unwrap_or_else(|| Solutions::refused(Reason::NotFound))
}

/// [`try_exponential_substitution`]; `None` when the equation does not
/// have its shape.
fn exponential_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Option<Solutions> {
    let mut atoms: Vec<ExprId> = Vec::new();
    walk_var_atoms(arena, expr, var, |id| {
        let ok = match arena.node(id) {
            ExprNode::Exp(_) => true,
            ExprNode::Pow(b, _) => !expr_contains_var(arena, *b, var),
            _ => false,
        };
        if ok && !atoms.contains(&id) {
            atoms.push(id);
        }
        ok
    })?;
    // (atom, rate in units of a₀ and the base's log, constant factor)
    let mut rated: Vec<(ExprId, Q, ExprId)> = Vec::with_capacity(atoms.len());
    let mut unit: Option<ExprId> = None; // a₀
    let mut base: Option<Option<BigInt>> = None; // Some(None): e; Some(Some(r)): r
    for &id in &atoms {
        let (this_base, arg, scale, cbase) = match arena.node(id).clone() {
            ExprNode::Exp(arg) => (None, arg, Q::one(), None),
            ExprNode::Pow(b, arg) => {
                let c = arena.as_num(b)?.clone();
                let (r, k) = perfect_power_base(&c)?;
                (Some(r), arg, Q::from_integer(BigInt::from(k)), Some(b))
            }
            _ => return None,
        };
        match &base {
            None => base = Some(this_base.clone()),
            Some(b) if *b == this_base => {}
            Some(_) => return None,
        }
        let (a, b0) = linear_rate(arena, arg, var)?;
        let a0 = *unit.get_or_insert(a);
        let ratio = arena.div(a, a0);
        let ratio = crate::transforms::eval::eval(arena, ratio);
        let ratio = arena.as_num(ratio)?.clone();
        // exp(a·x + b0) = e^b0·exp(a·x); c^(a·x + b0) = c^b0·r^(k·a·x).
        let factor = match cbase {
            None => arena.exp(b0),
            Some(cb) => arena.pow(cb, b0),
        };
        rated.push((id, ratio * scale, factor));
    }
    let unit = unit?;
    let base = base?;
    if rated.is_empty() {
        return None;
    }
    // g = gcd of the rates (gcd of numerators over lcm of denominators).
    let mut gn = BigInt::zero();
    let mut gd = BigInt::one();
    for (_, r, _) in &rated {
        gn = num_integer::Integer::gcd(&gn, r.numer());
        gd = num_integer::Integer::lcm(&gd, r.denom());
    }
    let g = Q::new(gn, gd);
    if g.is_zero() {
        return None;
    }
    let t = arena.symbol("__t_exp");
    let mut sub = expr;
    for (id, r, factor) in &rated {
        let n = (r / &g).to_integer();
        if n.abs() > BigInt::from(MAX_SUBSTITUTION_DEGREE) {
            return None;
        }
        let n_id = arena.big_int(n);
        let tn = arena.pow(t, n_id);
        let repl = arena.mul(&[*factor, tn]);
        sub = arena.subs_structural(sub, *id, repl);
    }
    let sub = crate::transforms::eval::eval(arena, sub);
    if expr_contains_var(arena, sub, var) {
        return None;
    }
    let g_id = arena.num_ratio(g);
    let gx = arena.mul(&[g_id, unit, var]);
    let generator = match base {
        None => arena.exp(gx),
        Some(r) => {
            let r_id = arena.big_int(r);
            arena.pow(r_id, gx)
        }
    };
    // A rational equation in `t` with a constant numerator has no root at
    // all: neither has the equation in `x` (`tanh(x − 3)·tanh(x + 3) = 1`;
    // refused before 0.47, SymPy: `[]`).
    {
        let num = full_numerator(arena, sub);
        if polybridge::expr_to_poly(arena, num, t).is_some_and(|p| p.is_constant() && !p.is_zero())
        {
            return Some(Solutions::none(Reason::said(
                "the equation in the exponential has no root",
            )));
        }
    }
    let (t_roots, t_missing) = solutions_unless_identity(arena, sub, t, cx)?
        .without_conditions()
        .into_found()
        .ok()?;
    let back = back_substitute(arena, generator, t_roots, var, cx).lacking(t_missing);
    if back.is_none_proved() {
        let (num, _) = polybridge::fraction_parts(arena, sub);
        let num = crate::transforms::expand::expand(arena, num);
        let num = crate::transforms::eval::eval(arena, num);
        if polybridge::expr_to_poly(arena, num, t).is_some() {
            return Some(Solutions::none(Reason::said(
                "no root of the equation in the exponential has a real preimage (exp of a real argument is positive)",
            )));
        }
        return None;
    }
    (back.has_roots() || back.is_nested_periods()).then_some(back)
}

/// Sums `Σ cᵢ·e^(Sᵢ(x))` of exponentials (`e^L`, `b^L` with constant
/// positive bases `b`, products of them; `S = Σ L·ln b`) whose exponents
/// fall into two groups up to constants: `C₁·e^(S₁) + C₂·e^(S₂) = 0` is
/// `S₁ − S₂ = ln(−C₂/C₁)`, an equation without exponentials (SymPy's
/// `_solve_exponential`).  For a real `S₁ − S₂`: no solution when
/// `−C₂/C₁` is certainly not positive.  `5^(x/2) = 2^(x/3)` gives `x = 0`,
/// `4^(x+1) + 4^(x−1) + 4^(x+2) = 3^(x+2) + 3^(x+3)` one root, `5^(x/2) =
/// 2^(3/x)` the two `±√(6·ln 2/ln 5)`; all refused up to 0.46 (bases
/// with no common power).  Otherwise a general solve takes every branch
/// `ln(−C₂/C₁) + 2nπi` (`2ˣ + 3ˣ = 0` for a plain `x`: `(2n + 1)πi/ln(2/3)`)
/// and a principal solve the principal one.  Up to 0.49 a real unknown was
/// taken to make `S₁ − S₂` real (`eˣ + e^(x + ix) = 0` for a real `x` had
/// "no solution", but `x = π` solves it) and a general solve over ℂ did
/// not split.  Candidates are checked by the caller.
fn try_exponential_log_split(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    exponential_log_split(arena, expr, var, cx)
        .unwrap_or_else(|| Solutions::refused(Reason::NotFound))
}

/// [`try_exponential_log_split`]; `None` when the equation does not have
/// its shape.
fn exponential_log_split(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Option<Solutions> {
    let ExprNode::Add(terms) = arena.node(expr).clone() else {
        return None;
    };
    // (coefficient, exponent S) per term
    let mut parts: Vec<(ExprId, ExprId)> = Vec::with_capacity(terms.len());
    for t in terms.iter().copied() {
        let factors: Vec<ExprId> = match arena.node(t) {
            ExprNode::Mul(ch) => ch.to_vec(),
            _ => vec![t],
        };
        let mut coeff: Vec<ExprId> = Vec::new();
        let mut logs: Vec<ExprId> = Vec::new();
        for f in factors {
            if !expr_contains_var(arena, f, var) {
                coeff.push(f);
                continue;
            }
            let node = arena.node(f).clone();
            match node {
                ExprNode::Exp(l) => logs.push(l),
                ExprNode::Pow(b, l)
                    if !expr_contains_var(arena, b, var)
                        && crate::base::walk::free_symbols(arena, b).is_empty()
                        && crate::poly::algebraic::sign_checked(arena, b) == Some(1) =>
                {
                    let ln_b = arena.ln(b);
                    logs.push(arena.mul(&[l, ln_b]));
                }
                _ => return None,
            }
        }
        if logs.is_empty() {
            return None;
        }
        let c = arena.mul(&coeff);
        let s = arena.add(&logs);
        let s = crate::transforms::expand::expand(arena, s);
        let s = crate::transforms::eval::eval(arena, s);
        parts.push((c, s));
    }
    // Group the exponents equal up to a constant: (S, Σ cᵢ·e^(Sᵢ − S)).
    let mut groups: Vec<(ExprId, Vec<ExprId>)> = Vec::new();
    for (c, s) in parts {
        let mut placed = false;
        for (s0, coeffs) in groups.iter_mut() {
            let d = arena.sub(s, *s0);
            let d = crate::transforms::expand::expand(arena, d);
            let d = crate::transforms::eval::eval(arena, d);
            if !expr_contains_var(arena, d, var) {
                let e = arena.exp(d);
                coeffs.push(arena.mul(&[c, e]));
                placed = true;
                break;
            }
        }
        if !placed {
            groups.push((s, vec![c]));
        }
    }
    let [(s1, c1), (s2, c2)] = &groups[..] else {
        return None;
    };
    let c1 = arena.add(c1);
    let c1 = crate::transforms::eval::eval(arena, c1);
    let c2 = arena.add(c2);
    let c2 = crate::transforms::eval::eval(arena, c2);
    if arena.is_zero_structural(c1) || arena.is_zero_structural(c2) {
        return None;
    }
    let ratio = arena.div(c2, c1);
    let ratio = arena.neg(ratio);
    let ratio = crate::transforms::eval::eval(arena, ratio);
    let d = arena.sub(*s1, *s2);
    // `e^(S₁ − S₂) = ratio ≤ 0`: no solution for a real `S₁ − S₂`;
    // otherwise `S₁ − S₂ = ln(ratio)`, all its branches in a general solve
    // (as the arm of `exp` in `solve_by_peeling`).
    let sign = if crate::base::walk::free_symbols(arena, ratio).is_empty() {
        crate::poly::algebraic::sign_checked(arena, ratio)
    } else {
        None
    };
    let real_d = cx.range_restricted(arena, d, var);
    if sign == Some(0) || (sign == Some(-1) && real_d) {
        return Some(Solutions::none(Reason::said(
            "the exponentials of a real argument are positive and the coefficients do not allow it",
        )));
    }
    let ln_ratio = principal_ln(arena, ratio);
    let branch = match cx.period() {
        Some(n) if !real_d => {
            if period_in_use(arena, ratio, cx) {
                return Some(Solutions::refused(Reason::NestedPeriods));
            }
            let p = two_pi_i_n(arena, n);
            arena.add(&[ln_ratio, p])
        }
        _ => ln_ratio,
    };
    let eq = arena.sub(d, branch);
    let eq = crate::transforms::expand::expand(arena, eq);
    let eq = crate::transforms::eval::eval(arena, eq);
    if !expr_contains_var(arena, eq, var) || has_exponential_of(arena, eq, var) {
        return None;
    }
    Some(solve_raw(arena, eq, var, cx))
}

/// Most absolute values split by [`try_abs_cases`] (`2ⁿ` sign patterns).
const MAX_ABS_CASES: usize = 4;

/// What an equation with absolute values says of a complex unknown
/// ([`abs_unknown_shape`]).
enum AbsUnknown {
    /// Every solution is real: the numerator is `A·x + B(|uᵢ|)` with a
    /// nonzero real constant `A` and `B` real for real moduli.
    Real,
    /// The unknown occurs only in the one modulus given (`1 + C(|x|, 4)`).
    ModulusOnly(ExprId),
    /// Neither: `|x| = 1` is a circle.
    Other,
}

/// The shape of `expr` in the absolute values `|uᵢ|` of `var` (each `uᵢ`
/// rational in `var`) for a complex `var` ([`AbsUnknown`]): `|x − π| + x
/// = 0` forces `x = −|x − π|` real, as do `|x|/x + 3 = 0` and `|x| + c −
/// x = 0`; up to 0.48 these were split as for a real `x` because a plain
/// `x` was taken real, and in 0.49 refused.
fn abs_unknown_shape(arena: &mut Arena, expr: ExprId, var: ExprId) -> AbsUnknown {
    use crate::base::assumptions::{AssumptionCache, Assumptions, Props};
    let mut atoms: Vec<ExprId> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if let ExprNode::Abs(u) = *arena.node(id)
            && expr_contains_var(arena, u, var)
        {
            if !is_rational_in(arena, u, var) {
                return AbsUnknown::Other;
            }
            if !atoms.contains(&id) {
                atoms.push(id);
            }
        }
    }
    if atoms.is_empty() || atoms.len() > MAX_ABS_CASES {
        return AbsUnknown::Other;
    }
    // each modulus a real stand-in `sᵢ`
    let mut replaced = expr;
    for (i, &a) in atoms.iter().enumerate() {
        let s =
            crate::transforms::subs::fresh_symbol(arena, &format!("__abs_modulus_{i}"), &[expr]);
        if let ExprNode::Symbol(sid) = *arena.node(s) {
            let mut real = Assumptions::default();
            real.assert_true(Props::REAL);
            arena.set_symbol_assumptions(sid, real);
        }
        replaced = arena.subs_structural(replaced, a, s);
    }
    if !expr_contains_var(arena, replaced, var) {
        return match atoms[..] {
            [atom] => AbsUnknown::ModulusOnly(atom),
            _ => AbsUnknown::Other,
        };
    }
    let num = full_numerator(arena, replaced);
    let Some(coeffs) = symbolic_poly_coeffs(arena, num, var) else {
        return AbsUnknown::Other;
    };
    let [b, a] = coeffs[..] else {
        return AbsUnknown::Other;
    };
    let a_real_nonzero = crate::base::walk::free_symbols(arena, a).is_empty()
        && matches!(crate::poly::algebraic::sign_checked(arena, a), Some(s) if s != 0);
    if a_real_nonzero
        && !expr_contains_var(arena, b, var)
        && AssumptionCache::new().query(arena, b, Props::REAL) == Some(true)
    {
        AbsUnknown::Real
    } else {
        AbsUnknown::Other
    }
}

/// The roots of `expr = 0` in a complex `var` that occurs only in the
/// modulus `atom = |u|` (real and nonnegative): no root when no real
/// `s ≥ 0` solves `expr(|u| = s) = 0` (`1 + C(|x|, 4)`: `C(s, 4) ≥ −1/24`),
/// the roots of `u = 0` when `s = 0` is the only one; `None` when a
/// positive `s` solves it (`|u| = s` is a curve: `tan |x| = 0` holds on
/// the circles `|x| = nπ`) or nothing is decided.
fn modulus_only_roots(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    atom: ExprId,
    cx: &SolveCx,
) -> Option<Solutions> {
    use crate::base::assumptions::{Assumptions, Props};
    let ExprNode::Abs(u) = *arena.node(atom) else {
        return None;
    };
    let s = crate::transforms::subs::fresh_symbol(arena, "__abs_modulus", &[expr]);
    if let ExprNode::Symbol(sid) = *arena.node(s) {
        let mut real = Assumptions::default();
        real.assert_true(Props::REAL);
        arena.set_symbol_assumptions(sid, real);
    }
    let in_s = arena.subs_structural(expr, atom, s);
    // every real root of the equation in `s` (a general solve: the
    // principal roots of `tan s = 0` are `[0]`, its roots `nπ`)
    let n = branch_parameter(arena, &[expr, var]);
    let points = match solve_impl(arena, in_s, s, &SolveCx::general(Domain::Real, n)) {
        Solutions::Complete(RootSet::Empty(_)) => Vec::new(),
        complete @ Solutions::Complete(RootSet::Points(_) | RootSet::Families { .. }) => {
            complete.claimed_points()
        }
        _ => return None,
    };
    // a family of `s` has positive members (`|x| = π`: a circle)
    if points
        .iter()
        .any(|&p| crate::base::walk::contains(arena, p, n))
    {
        return None;
    }
    let mut zero = false;
    for p in points {
        if crate::transforms::realness::constant_realness_as_declared(
            arena,
            p,
            ROOT_CHECK_DIGITS,
            &mut crate::base::assumptions::AssumptionCache::new(),
        ) == Some(false)
        {
            continue;
        }
        match crate::poly::algebraic::sign_checked(arena, p) {
            Some(s) if s < 0 => {}
            Some(0) => zero = true,
            _ => return None,
        }
    }
    if zero {
        return Some(solve_raw(arena, u, var, cx));
    }
    Some(Solutions::none(Reason::said(
        "no nonnegative real value of the absolute value solves the equation",
    )))
}

/// An equation with absolute values `|uᵢ|` of rational functions `uᵢ` (with
/// rational coefficients) of a variable taken real: for each choice of
/// signs `|uᵢ| = ±uᵢ` the equation without them (a rational one, all its
/// roots found), and the real candidates of all the choices — the caller
/// checks every candidate against the equation, which drops those of a
/// choice the signs at them contradict.  Every real root is found by the
/// choice of the signs of the `uᵢ` at it.  `|x + 3| = 2·|x − 3|` gives `1, 9`,
/// `|x + 1| + |x² − 7| = 9` gives `(√61 − 1)/2, (1 − √69)/2`, refused
/// before 0.47 (SymPy gives the same for a real `x`).  `None` for another shape, a
/// variable that may be non-real, or more than [`MAX_ABS_CASES`] absolute
/// values; `NoSolution` when no choice has a real root.
fn try_abs_cases(arena: &mut Arena, expr: ExprId, var: ExprId, cx: &SolveCx) -> Solutions {
    abs_cases(arena, expr, var, cx).unwrap_or_else(|| Solutions::refused(Reason::NotFound))
}

/// [`try_abs_cases`]; `None` when the equation does not have its shape, or
/// a choice of signs is refused or vanishes identically.
fn abs_cases(arena: &mut Arena, expr: ExprId, var: ExprId, cx: &SolveCx) -> Option<Solutions> {
    if !crate::base::walk::free_symbols(arena, expr)
        .iter()
        .all(|&s| s == var)
    {
        return None;
    }
    // (Up to 0.48 a plain `x` was taken real under `Ex::solve`: `|x + 3| =
    // 2·|x − 3|` gave `[9, 1]`, while the Apollonius circle solves it.  A
    // complex unknown is split only when the equation makes it real.)
    let real_cx;
    let cx = if cx.unknown_may_be_nonreal(arena, var) {
        match abs_unknown_shape(arena, expr, var) {
            AbsUnknown::Real => {
                real_cx = cx.with_real_unknown();
                &real_cx
            }
            AbsUnknown::ModulusOnly(atom) => return modulus_only_roots(arena, expr, var, atom, cx),
            AbsUnknown::Other => return None,
        }
    } else {
        cx
    };
    let mut abs_atoms: Vec<ExprId> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if let ExprNode::Abs(u) = *arena.node(id)
            && expr_contains_var(arena, u, var)
        {
            // `|u| = ±u` covers every case only for a real `u` (an imaginary
            // variable has `|i·t| = |t|`).
            if !is_rational_in(arena, u, var)
                || !cx.real(arena, u, var)
                || crate::base::walk::post_order_ids(arena, u)
                    .into_iter()
                    .any(|j| matches!(arena.node(j), ExprNode::Abs(_)))
            {
                return None;
            }
            if !abs_atoms.contains(&id) {
                abs_atoms.push(id);
            }
        }
    }
    if abs_atoms.is_empty() || abs_atoms.len() > MAX_ABS_CASES {
        return None;
    }
    let mut out: Vec<ExprId> = Vec::new();
    for mask in 0..(1usize << abs_atoms.len()) {
        let mut branch = expr;
        for (i, &a) in abs_atoms.iter().enumerate() {
            let ExprNode::Abs(u) = *arena.node(a) else {
                return None;
            };
            let signed = if mask & (1 << i) != 0 {
                arena.neg(u)
            } else {
                u
            };
            branch = arena.subs_structural(branch, a, signed);
        }
        let branch = crate::transforms::eval::eval(arena, branch);
        if !is_rational_in(arena, branch, var) {
            return None;
        }
        // Each choice is a rational equation with rational coefficients,
        // whose roots are all found: one refused (or partial) answers
        // nothing; `0 = 0` on a whole choice is an interval of solutions.
        match solve_raw(arena, branch, var, cx).into_found() {
            Ok((roots, None)) => {
                for v in roots {
                    let v = crate::transforms::eval::eval(arena, v);
                    if crate::transforms::realness::constant_realness_as_declared(
                        arena,
                        v,
                        ROOT_CHECK_DIGITS,
                        &mut crate::base::assumptions::AssumptionCache::new(),
                    ) == Some(false)
                    {
                        continue;
                    }
                    if !out.contains(&v) {
                        out.push(v);
                    }
                }
            }
            Err(Solutions::Complete(RootSet::Empty(_))) => {}
            Err(nested) if nested.is_nested_periods() => return Some(nested),
            Ok((_, Some(_))) | Err(_) => return None,
        }
    }
    Some(Solutions::exactly(
        out,
        Reason::said("no choice of the signs of the absolute values has a real root"),
    ))
}

/// Is `expr` a polynomial or a rational function of `var` (integer powers
/// only), with any coefficients free of it?
fn is_rational_in(arena: &Arena, expr: ExprId, var: ExprId) -> bool {
    crate::base::walk::post_order_ids(arena, expr)
        .into_iter()
        .filter(|&id| expr_contains_var(arena, id, var))
        .all(|id| match arena.node(id) {
            ExprNode::Symbol(_) | ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => true,
            ExprNode::Pow(_, e) => arena.as_num(*e).is_some_and(|r| r.is_integer()),
            _ => false,
        })
}

/// Largest `k` of a `C(f, k)` written as a polynomial.
const MAX_BINOMIAL_POLY: i64 = 32;

/// `expr` with each `C(f, k)` (`f` involving `var`, `k` an integer in
/// `0..=MAX_BINOMIAL_POLY`) replaced by `f(f − 1)⋯(f − k + 1)/k!`; `None`
/// when there is none.
fn binomials_as_polynomials(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<ExprId> {
    let mut out = expr;
    let mut any = false;
    for id in crate::base::walk::post_order_ids(arena, expr) {
        let ExprNode::Binomial(f, k) = *arena.node(id) else {
            continue;
        };
        if !expr_contains_var(arena, f, var) {
            continue;
        }
        let Some(kk) = arena
            .as_num(k)
            .filter(|q| q.is_integer())
            .map(|q| q.to_integer())
        else {
            continue;
        };
        let Some(kk) =
            num_traits::ToPrimitive::to_i64(&kk).filter(|v| (0..=MAX_BINOMIAL_POLY).contains(v))
        else {
            continue;
        };
        let mut factors: Vec<ExprId> = Vec::with_capacity(kk as usize + 1);
        let mut fact = BigInt::one();
        for j in 0..kk {
            let jj = arena.int(j);
            factors.push(arena.sub(f, jj));
            fact *= BigInt::from(j + 1);
        }
        let inv = arena.num_ratio(Q::new(BigInt::one(), fact));
        factors.push(inv);
        let poly = arena.mul(&factors);
        out = arena.subs_structural(out, id, poly);
        any = true;
    }
    if !any {
        return None;
    }
    let out = crate::transforms::eval::eval(arena, out);
    Some(out)
}

/// Does `e` hold an exponential (`e^u`, `b^u`) whose exponent involves `var`?
fn has_exponential_of(arena: &Arena, e: ExprId, var: ExprId) -> bool {
    crate::base::walk::post_order_ids(arena, e)
        .into_iter()
        .any(|id| match *arena.node(id) {
            ExprNode::Exp(u) => expr_contains_var(arena, u, var),
            ExprNode::Pow(b, u) => {
                !expr_contains_var(arena, b, var) && expr_contains_var(arena, u, var)
            }
            _ => false,
        })
}

/// Equations in `sinh`, `cosh`, `tanh` (and exponentials) of the variable:
/// rewritten in exponentials (`sinh u = (eᵘ − e⁻ᵘ)/2`, …) and solved by
/// [`try_exponential_substitution`]; the candidates are checked against the
/// original equation.  SymPy: `solve(sinh(x) + cosh(x) - 3)` → `[log(3)]`,
/// `solve(sinh(2*x) - sinh(x))` → `[0, -I*pi/3, I*pi/3, I*pi]`; both were
/// refused before 0.44 (the single inversions `sinh x = 2`, `cosh x = 2`
/// were solved).
fn try_hyperbolic_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    let mut hyperbolic = false;
    let shape = walk_var_atoms(arena, expr, var, |id| match arena.node(id) {
        ExprNode::Sinh(_) | ExprNode::Cosh(_) | ExprNode::Tanh(_) => {
            hyperbolic = true;
            true
        }
        ExprNode::Exp(_) => true,
        ExprNode::Pow(b, _) => !expr_contains_var(arena, *b, var),
        _ => false,
    });
    if shape.is_none() || !hyperbolic {
        return Solutions::refused(Reason::NotFound);
    }
    let rewritten = arena.rewrite_as_exp_expr(expr);
    let rewritten = crate::transforms::eval::eval(arena, rewritten);
    if rewritten == expr {
        return Solutions::refused(Reason::NotFound);
    }
    try_exponential_substitution(arena, rewritten, var, cx)
}

/// The numerator of `expr` over a common denominator when that
/// denominator is a polynomial in `var` and the numerator is not
/// (`ln(x)/x − 1/4` → `4·ln x − x`): every root of `expr` is one of the
/// numerator, and the callers check the numerator's roots against `expr`.
fn numerator_over_polynomial(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<ExprId> {
    if !matches!(arena.node(expr), ExprNode::Add(_) | ExprNode::Mul(_)) {
        return None;
    }
    let combined = polybridge::together(arena, expr);
    let (num, den) = polybridge::as_numer_denom(arena, combined);
    if !expr_contains_var(arena, den, var) || symbolic_poly_coeffs(arena, den, var).is_none() {
        return None;
    }
    let num = crate::transforms::expand::expand(arena, num);
    let num = crate::transforms::eval::eval(arena, num);
    if num == expr
        || !expr_contains_var(arena, num, var)
        || polybridge::expr_to_poly(arena, num, var).is_some()
    {
        return None;
    }
    Some(num)
}

/// The solutions of the substituted equation `sub = 0` in `t`, or `None`
/// when it is an identity: the original equation then holds wherever the
/// substitution is defined, which no finite list (or family) of points
/// describes.  Before 0.39 `sin(x)/cos(x) − tan(x) = 0` (`0 = 0` in
/// `t = tan(x/2)`) gave only `x = π` (the value `t = ∞` checked apart),
/// and `solve_general` the family `π + 2nπ`.  Also `None` when the roots
/// of its numerator cannot all be found ([`numerator_roots`]).
fn solutions_unless_identity(
    arena: &mut Arena,
    sub: ExprId,
    t: ExprId,
    cx: &SolveCx,
) -> Option<Solutions> {
    let combined = polybridge::together(arena, sub);
    let (num, _) = polybridge::as_numer_denom(arena, combined);
    let num = crate::transforms::expand::expand(arena, num);
    let num = crate::transforms::eval::eval(arena, num);
    if arena.is_zero_structural(num) {
        return None;
    }
    // (`together` leaves nested quotients: the whole combined fraction)
    let full_num = full_numerator(arena, sub);
    match numerator_roots(arena, full_num, t, cx) {
        NumeratorRoots::Solver => {}
        NumeratorRoots::Found(roots) => return Some(Solutions::found(roots)),
        NumeratorRoots::Unknown => return None,
    }
    match solve_impl(arena, sub, t, &cx.substituted()) {
        Solutions::Complete(RootSet::All) => None,
        other => Some(other),
    }
}

/// The numerator of `sub` over one denominator (nested quotients
/// combined), expanded.
fn full_numerator(arena: &mut Arena, sub: ExprId) -> ExprId {
    let (full_num, _) = polybridge::fraction_parts(arena, sub);
    let full_num = crate::transforms::expand::expand(arena, full_num);
    crate::transforms::eval::eval(arena, full_num)
}

/// How the roots of the numerator of a substituted equation in `t` are
/// found ([`numerator_roots`]).
enum NumeratorRoots {
    /// The solver finds them all: solve the substituted equation.
    Solver,
    /// Every root of the numerator, found over a number field; candidates
    /// that are poles of the substituted equation are left to the check of
    /// the original equation.
    Found(Vec<ExprId>),
    /// Not all found: the substitution must not answer.
    Unknown,
}

/// Does the solver find every root of the numerator `num` of a substituted
/// equation in `t`?  Over ℚ always (exact factorisation; `RootOf` beyond
/// degree 4), and for degree ≤ 2; otherwise (coefficients with radicals or
/// parameters) only when it finds as many distinct roots as the degree.
/// The solver returns the roots it finds of such a polynomial, which
/// `Ex::solve` checks against the count of distinct roots, but a
/// substitution took them as all its roots: before 0.44 `sin 4x = √3·sin x`
/// (in `t = tan(x/2)` a degree-9 polynomial over ℚ(√3)) gave `[π]`, without
/// the root `0` and the six others.  When the solver does not find them
/// all, a polynomial with algebraic coefficients is solved over the field
/// they generate ([`algebraic_roots::number_field_roots`]): in 0.44 the
/// substitution then refused, `sin 4x = √3·sin x` and `cos³x = √2·cos x`
/// were "not polynomial".  `Solver` when `num` is not a polynomial in `t`
/// (nothing to count).
fn numerator_roots(arena: &mut Arena, num: ExprId, t: ExprId, cx: &SolveCx) -> NumeratorRoots {
    if polybridge::expr_to_poly(arena, num, t).is_some() {
        return NumeratorRoots::Solver;
    }
    let Some(coeffs) = symbolic_poly_coeffs(arena, num, t) else {
        return NumeratorRoots::Solver;
    };
    let degree = coeffs.len().saturating_sub(1);
    if degree <= 2
        || solve_impl(arena, num, t, &cx.substituted())
            .roots()
            .map_or(0, <[ExprId]>::len)
            >= degree
    {
        return NumeratorRoots::Solver;
    }
    match algebraic_roots::number_field_roots(arena, num, t, false) {
        Some(roots) => NumeratorRoots::Found(roots),
        None => NumeratorRoots::Unknown,
    }
}

/// The roots in `t` of the substituted equation `sub = 0` of a change of
/// variable (refused when they cannot all be found, [`numerator_roots`]).
fn substitution_roots(arena: &mut Arena, sub: ExprId, t: ExprId, cx: &SolveCx) -> Solutions {
    let full_num = full_numerator(arena, sub);
    match numerator_roots(arena, full_num, t, cx) {
        NumeratorRoots::Solver => solve_impl(arena, sub, t, &cx.substituted()),
        NumeratorRoots::Found(roots) => Solutions::found(roots),
        NumeratorRoots::Unknown => Solutions::refused(Reason::NotFound),
    }
}

/// Largest power of the new unknown a substitution may create.
const MAX_SUBSTITUTION_DEGREE: i64 = 64;

/// Trigonometric equations in `sin(k·x)`, `cos(k·x)`, `tan(k·x)` (integer
/// `k`) of one variable: the multiple angles are expanded, and the
/// half-angle substitution `t = tan(x/2)` (`sin x = 2t/(1 + t²)`,
/// `cos x = (1 − t²)/(1 + t²)`, `tan x = 2t/(1 − t²)`) makes the
/// equation rational in `t`; each root gives `x = 2·atan(t)`.  The value
/// `t = ∞` (`x = π`) is not reached by the substitution and is added when
/// the equation holds there exactly.  SymPy: `solve(sin(x) + cos(x) - 1)`
/// → `[0, pi/2]`, `solve(sin(2*x) - sin(x))` → `[0, -pi/3, pi/3]` (SymPy
/// misses the root `π` there, which this adds); refused before 0.31.
fn try_half_angle_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    half_angle_substitution(arena, expr, var, cx)
        .filter(|s| s.has_roots() || s.is_nested_periods())
        .unwrap_or_else(|| Solutions::refused(Reason::NotFound))
}

/// [`try_half_angle_substitution`]; `None` when the equation does not have
/// its shape.
fn half_angle_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Option<Solutions> {
    let mut atoms: Vec<ExprId> = Vec::new();
    let mut multiple = false;
    walk_var_atoms(arena, expr, var, |id| {
        let arg = match arena.node(id) {
            ExprNode::Sin(a) | ExprNode::Cos(a) | ExprNode::Tan(a) => *a,
            _ => return false,
        };
        let ok = if arg == var {
            true
        } else if let ExprNode::Mul(ch) = arena.node(arg)
            && ch.len() == 2
            && ch.contains(&var)
            && ch
                .iter()
                .any(|&c| arena.as_num(c).is_some_and(|r| r.is_integer()))
        {
            multiple = true;
            true
        } else {
            false
        };
        if ok && !atoms.contains(&id) {
            atoms.push(id);
        }
        ok
    })?;
    if atoms.len() < 2 && !multiple {
        return None;
    }
    let expanded = if multiple {
        // `tan(k·x)` as `sin(k·x)/cos(k·x)`, which `expand_trig` expands
        // (it leaves `tan(2x)` alone): `tan 2x = tan x` was refused before
        // 0.44.  Not as a rational function of `tan x`, which loses the
        // roots at the poles of `tan x` (`tan 2x = 0` at `x = π/2`).
        let mut e = expr;
        for &id in &atoms {
            if let ExprNode::Tan(arg) = *arena.node(id)
                && arg != var
            {
                let s = arena.sin(arg);
                let c = arena.cos(arg);
                let q = arena.div(s, c);
                e = arena.subs_structural(e, id, q);
            }
        }
        let e = arena.expand_trig_expr(e);
        crate::transforms::eval::eval(arena, e)
    } else {
        expr
    };
    // After the expansion only sin x, cos x, tan x may carry the variable.
    let (s, c, tn) = (arena.sin(var), arena.cos(var), arena.tan(var));
    walk_var_atoms(arena, expanded, var, |id| id == s || id == c || id == tn)?;
    let t = arena.symbol("__t_half");
    let one = arena.one;
    let two = arena.int(2);
    let t2 = arena.pow(t, two);
    let one_p = arena.add(&[one, t2]);
    let one_m = arena.sub(one, t2);
    let two_t = arena.mul(&[two, t]);
    let sin_t = arena.div(two_t, one_p);
    let cos_t = arena.div(one_m, one_p);
    let tan_t = arena.div(two_t, one_m);
    let mut sub = arena.subs_structural(expanded, s, sin_t);
    sub = arena.subs_structural(sub, c, cos_t);
    sub = arena.subs_structural(sub, tn, tan_t);
    let sub = crate::transforms::eval::eval(arena, sub);
    if expr_contains_var(arena, sub, var) {
        return None;
    }
    let half = arena.rational(1, 2);
    let half_x = arena.mul(&[half, var]);
    let generator = arena.tan(half_x);
    // (No root in `t`, or none found: `x = π` below may still be one.)
    let (t_roots, t_missing) = match solutions_unless_identity(arena, sub, t, cx)?
        .without_conditions()
        .into_found()
    {
        Ok(found) => found,
        Err(Solutions::Refused(reason)) => (Vec::new(), Some(Missing::Alternative(reason))),
        Err(_) => (Vec::new(), None),
    };
    let mut out = back_substitute(arena, generator, t_roots, var, cx);
    // x = π (t = ∞), when the equation holds there exactly.
    let pi = arena.pi;
    let at_pi = crate::transforms::subs::subs(arena, expr, var, pi);
    let at_pi = crate::transforms::eval::eval(arena, at_pi);
    let zero_at_pi = arena.is_zero_structural(at_pi)
        || (crate::base::walk::free_symbols(arena, at_pi).is_empty()
            && crate::poly::algebraic::is_zero_checked(arena, at_pi) == Some(true));
    if zero_at_pi {
        let value = match cx.period() {
            Some(n) => {
                let two_pi_n = arena.mul(&[two, pi, n]);
                arena.add(&[pi, two_pi_n])
            }
            None => pi,
        };
        out = out.union(Solutions::found(vec![value]));
    }
    Some(out.lacking(t_missing))
}

/// Sums of logarithms `Σ cᵢ·ln fᵢ(x) + g = 0` (rational `cᵢ`, `g` free of
/// `x`): exponentiated, `∏ fᵢ^cᵢ = e^(−g)`, solved, and every candidate
/// checked against the original equation (principal logarithms: the
/// candidate `x = −3` of `ln x + ln(x + 1) = ln 6` has the residual `2πi`
/// and is dropped).  SymPy: `solve(log(x) + log(x + 1) - log(6))` → `[2]`;
/// refused before 0.31.
fn try_log_sum(arena: &mut Arena, expr: ExprId, var: ExprId, cx: &SolveCx) -> Solutions {
    match log_sum_equation(arena, expr, var) {
        // (no root of the exponentiated equation is not taken as a proof)
        Some(eq) => match solve_raw(arena, eq, var, cx) {
            found if found.has_roots() || found.is_nested_periods() => found,
            _ => Solutions::refused(Reason::NotFound),
        },
        None => Solutions::refused(Reason::NotFound),
    }
}

/// The exponentiated equation of [`try_log_sum`], or `None` when the
/// equation does not have its shape.
fn log_sum_equation(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<ExprId> {
    let terms: Vec<ExprId> = match arena.node(expr) {
        ExprNode::Add(ch) => ch.to_vec(),
        _ => return None,
    };
    let mut factors: Vec<ExprId> = Vec::new();
    let mut logs: Vec<ExprId> = Vec::new();
    let mut rest: Vec<ExprId> = Vec::new();
    let mut var_rest: Vec<ExprId> = Vec::new();
    for term in terms {
        if !expr_contains_var(arena, term, var) {
            rest.push(term);
            continue;
        }
        let (c, f) = match arena.node(term).clone() {
            ExprNode::Ln(f) => (arena.one, f),
            ExprNode::Mul(ch) if ch.len() == 2 => {
                let (k, l) = if arena.as_num(ch[0]).is_some() {
                    (ch[0], ch[1])
                } else {
                    (ch[1], ch[0])
                };
                match (arena.as_num(k), arena.node(l)) {
                    (Some(_), ExprNode::Ln(f)) => (k, *f),
                    _ => {
                        var_rest.push(term);
                        continue;
                    }
                }
            }
            _ => {
                var_rest.push(term);
                continue;
            }
        };
        factors.push(arena.pow(f, c));
        logs.push(f);
    }
    if var_rest.is_empty() {
        if factors.len() < 2 {
            return None;
        }
    } else {
        // One logarithm of a non-linear argument beside other terms in `x`
        // (`ln(x²) = 3x` as `x² = e^(3x)`; SymPy `solve(log(x**2) - 3*x)`
        // → `[-2*LambertW(-3/2)/3, -2*LambertW(3/2)/3]`, refused before
        // 0.44).  A linear argument is left to the Lambert forms
        // (`ln x = 2x + 1`), and the other terms must be free of logarithms
        // of `x`.
        let log_free = !var_rest.iter().any(|&r| {
            crate::base::walk::post_order_ids(arena, r).into_iter().any(
                |id| matches!(arena.node(id), ExprNode::Ln(a) if expr_contains_var(arena, *a, var)),
            )
        });
        if factors.len() != 1 || !log_free || linear_rate(arena, logs[0], var).is_some() {
            return None;
        }
        rest.extend(var_rest);
    }
    let lhs = arena.mul(&factors);
    let g = arena.add(&rest);
    let minus_g = arena.neg(g);
    let rhs = arena.exp(minus_g);
    let eq = arena.sub(lhs, rhs);
    let eq = crate::transforms::eval::eval(arena, eq);
    if !expr_contains_var(arena, eq, var) {
        return None;
    }
    Some(eq)
}

/// Collect candidate generator functions: exp(x), sin(x), cos(x), ln(x), x^(k), etc.
fn collect_generators(arena: &Arena, expr: ExprId, var: ExprId, var_sym: SymbolId) -> Vec<ExprId> {
    let mut generators = Vec::new();
    let mut visited = std::collections::HashSet::new();
    collect_gens_recursive(arena, expr, var, var_sym, &mut generators, &mut visited);
    generators
}

#[allow(clippy::only_used_in_recursion)]
fn collect_gens_recursive(
    arena: &Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    gens: &mut Vec<ExprId>,
    visited: &mut std::collections::HashSet<ExprId>,
) {
    if !visited.insert(expr) {
        return;
    }
    match arena.node(expr).clone() {
        ExprNode::Exp(inner) if inner == var && !gens.contains(&expr) => {
            gens.push(expr);
        }
        ExprNode::Sin(inner) | ExprNode::Cos(inner) | ExprNode::Tan(inner)
            if inner == var && !gens.contains(&expr) =>
        {
            gens.push(expr);
        }
        ExprNode::Ln(inner) if inner == var && !gens.contains(&expr) => {
            gens.push(expr);
        }
        ExprNode::Pow(base, exp)
            if base == var && !expr_contains_var(arena, exp, var)
            // x^(1/n) or x^k type
            && !gens.contains(&expr) =>
        {
            gens.push(expr);
        }
        ExprNode::Add(ref children) | ExprNode::Mul(ref children) => {
            for &c in children {
                collect_gens_recursive(arena, c, var, var_sym, gens, visited);
            }
        }
        ExprNode::Pow(base, exp) => {
            collect_gens_recursive(arena, base, var, var_sym, gens, visited);
            collect_gens_recursive(arena, exp, var, var_sym, gens, visited);
        }
        ExprNode::Neg(inner)
        | ExprNode::Exp(inner)
        | ExprNode::Ln(inner)
        | ExprNode::Sin(inner)
        | ExprNode::Cos(inner)
        | ExprNode::Tan(inner) => {
            collect_gens_recursive(arena, inner, var, var_sym, gens, visited);
        }
        _ => {}
    }
}

/// Rewrite `exp(k*x)` as `exp(x)^k` for small integer k throughout the expression.
fn rewrite_exp_powers(arena: &mut Arena, expr: ExprId, var: ExprId, gen_exp_x: ExprId) -> ExprId {
    let mut result = expr;
    // Check small positive integer multiples: exp(k*x) → exp(x)^k
    for k in 2i64..=6 {
        let k_id = arena.int(k);
        let k_var = arena.mul(&[k_id, var]);
        let exp_k_var = arena.exp(k_var);
        let gen_pow_k = arena.pow(gen_exp_x, k_id);
        result = arena.subs_structural(result, exp_k_var, gen_pow_k);
    }
    // Also handle negative multiples: exp(-k*x) → exp(x)^(-k)
    for k in [-1i64, -2, -3] {
        let k_id = arena.int(k);
        let k_var = arena.mul(&[k_id, var]);
        let exp_k_var = arena.exp(k_var);
        let gen_pow_k = arena.pow(gen_exp_x, k_id);
        result = arena.subs_structural(result, exp_k_var, gen_pow_k);
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// LambertW solving for mixed polynomial-exponential equations
// ═══════════════════════════════════════════════════════════════════════════

/// Try to solve equations involving mixed polynomial and exponential or
/// logarithmic terms with the Lambert W function (after SymPy's
/// `_solve_lambert` / `_lambert`, `sympy/solvers/bivariate.py`, BSD):
///
/// - `A·x·exp(B·x) + D = 0`  →  `x = W_k(−B·D/A)/B`
/// - `A·exp(B·x) + C·x + D = 0`  →  `x = −W_k(A·B/(C·exp(B·D/C)))/B − D/C`
/// - `A·x·ln(B·x) + C·x + D = 0`  →  `x = exp(W_k(−(B·D/A)·exp(C/A)) − C/A)/B`
/// - `A·ln(B·x) + C·x + D = 0`  →  `x = (A/C)·W_k((C/(A·B))·exp(−D/A))`
/// - `A·x^x + D = 0`  →  `x = exp(W_k(ln(−D/A)))`
///
/// (with `ln(B·x)` for a constant `B` solved in `B·x`).  The branches
/// `W_k` returned are those of [`lambert_branches`]: the principal one, and
/// `W₋₁` as well when the argument lies in `(−1/e, 0)`, where both are real
/// and the equation has two real roots.  Before 0.30 only `W₀` was taken:
/// `solve(exp(x) − 2x − π, x)` gave `−1.4540` and missed `1.9526`
/// (`−π/2 − W₋₁(−e^{−π/2}/2)`), and the logarithmic forms and `x^x` were
/// not recognised at all.
fn try_solve_lambert(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    cx: &SolveCx,
) -> Solutions {
    candidates_found(lambert_candidates(arena, expr, var, var_sym, cx))
}

/// The candidates of [`try_solve_lambert`]; `None` when the equation does
/// not have one of its shapes.
fn lambert_candidates(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    _var_sym: SymbolId,
    cx: &SolveCx,
) -> Option<Vec<Solution>> {
    tracing::debug!("solve: trying LambertW");

    // Get additive terms
    let terms: Vec<ExprId> = match arena.node(expr).clone() {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![expr],
    };

    // Classify each term into categories
    let mut constant_terms: Vec<ExprId> = Vec::new();
    let mut linear_coeffs: Vec<ExprId> = Vec::new(); // A for A*var
    let mut exp_terms: Vec<LambertExp> = Vec::new(); // A*exp(B*var)
    let mut var_exp_terms: Vec<LambertExp> = Vec::new(); // A*var*exp(B*var)
    let mut var_ln_terms: Vec<LambertLn> = Vec::new(); // A*var*ln(B*var)
    let mut ln_terms: Vec<LambertLn> = Vec::new(); // A*ln(B*var)
    let mut pow_terms: Vec<ExprId> = Vec::new(); // A for A*var^var

    for &term in &terms {
        if !expr_contains_var(arena, term, var) {
            constant_terms.push(term);
            continue;
        }

        match classify_lambert_term(arena, term, var) {
            Some(LambertTermClass::Linear(coeff)) => linear_coeffs.push(coeff),
            Some(LambertTermClass::ExpVar(e)) => exp_terms.push(e),
            Some(LambertTermClass::VarExpVar(e)) => var_exp_terms.push(e),
            Some(LambertTermClass::VarLnVar(ln)) => var_ln_terms.push(ln),
            Some(LambertTermClass::LnVar(ln)) => ln_terms.push(ln),
            Some(LambertTermClass::PowVarVar(coeff)) => pow_terms.push(coeff),
            None => return None,
        }
    }

    // Build constant sum (D)
    let d = match constant_terms.len() {
        0 => arena.zero,
        1 => constant_terms[0],
        _ => arena.add(&constant_terms),
    };
    let c = match linear_coeffs.len() {
        0 => None,
        1 => Some(linear_coeffs[0]),
        _ => Some(arena.add(&linear_coeffs)),
    };
    let counts = (
        exp_terms.len(),
        var_exp_terms.len(),
        var_ln_terms.len(),
        ln_terms.len(),
        pow_terms.len(),
    );

    let solutions: Vec<ExprId> = match (counts, c) {
        // ── A*var*exp(B*var) + D = 0 ─────────────────────────────────
        //   B*var*exp(B*var) = -B*D/A  →  var = W(-B*D/A)/B
        ((0, 1, 0, 0, 0), None) => {
            let LambertExp { coeff: a, rate: b } = var_exp_terms[0];
            let neg_d = arena.neg(d);
            let neg_d_over_a = arena.div(neg_d, a);
            let b_arg = arena.mul(&[b, neg_d_over_a]);
            lambert_branches(arena, b_arg, cx)
                .into_iter()
                .map(|w| arena.div(w, b))
                .collect()
        }
        // ── A*exp(B*var) + C*var + D = 0 ─────────────────────────────
        //   u = -(B*var + B*D/C):  u·exp(u) = A·B/(C·exp(B·D/C))
        //   var = -W(…)/B - D/C
        ((1, 0, 0, 0, 0), Some(c)) => {
            let LambertExp {
                coeff: a_exp,
                rate: b,
            } = exp_terms[0];
            let b_d = arena.mul(&[b, d]);
            let b_d_over_c = arena.div(b_d, c);
            let exp_bd_c = arena.exp(b_d_over_c);
            let c_exp_bd_c = arena.mul(&[c, exp_bd_c]);
            let a_b = arena.mul(&[a_exp, b]);
            let w_arg = arena.div(a_b, c_exp_bd_c);
            let d_over_c = arena.div(d, c);
            lambert_branches(arena, w_arg, cx)
                .into_iter()
                .map(|w| {
                    let neg_w = arena.neg(w);
                    let neg_w_over_b = arena.div(neg_w, b);
                    arena.sub(neg_w_over_b, d_over_c)
                })
                .collect()
        }
        // ── A*var*ln(B*var) + C*var + D = 0 ──────────────────────────
        //   B*var = e^t:  (t + C/A)·e^{t + C/A} = -(B·D/A)·e^{C/A}
        //   var = exp(W(-(B·D/A)·e^{C/A}) - C/A)/B
        ((0, 0, 1, 0, 0), c) if arena.is_zero_structural(var_ln_terms[0].shift) => {
            let LambertLn {
                coeff: a, scale: b, ..
            } = var_ln_terms[0];
            let c_over_a = match c {
                Some(c) => arena.div(c, a),
                None => arena.zero,
            };
            let neg_bd = arena.mul(&[arena.neg_one, b, d]);
            let neg_bd_over_a = arena.div(neg_bd, a);
            let exp_c_a = arena.exp(c_over_a);
            let w_arg = arena.mul(&[neg_bd_over_a, exp_c_a]);
            lambert_branches(arena, w_arg, cx)
                .into_iter()
                .map(|w| {
                    let t = arena.sub(w, c_over_a);
                    let y = arena.exp(t);
                    arena.div(y, b)
                })
                .collect()
        }
        // ── A*ln(B*var + E) + C*var + D = 0 ──────────────────────────
        //   u = B·var + E:  A·ln u + C′·u + D′ = 0,  C′ = C/B,  D′ = D − C·E/B
        //   u = (A/C′)·W((C′/A)·e^{-D′/A}),  var = (u − E)/B
        //   (E = 0: var = (A/C)·W((C/(A·B))·e^{-D/A}); a shift E was
        //   refused before 0.44: `ln(4x − 2) = 2x − 2`.)
        ((0, 0, 0, 1, 0), Some(c)) => {
            let LambertLn {
                coeff: a,
                scale: b,
                shift: e,
            } = ln_terms[0];
            let c1 = arena.div(c, b);
            let c1 = crate::transforms::eval::eval(arena, c1);
            let ce = arena.mul(&[c1, e]);
            let d1 = arena.sub(d, ce);
            let c1_over_a = arena.div(c1, a);
            let a_over_c1 = arena.div(a, c1);
            let neg_d = arena.neg(d1);
            let neg_d_over_a = arena.div(neg_d, a);
            let exp_part = arena.exp(neg_d_over_a);
            let w_arg = arena.mul(&[c1_over_a, exp_part]);
            lambert_branches(arena, w_arg, cx)
                .into_iter()
                .map(|w| {
                    let u = arena.mul(&[a_over_c1, w]);
                    let shifted = arena.sub(u, e);
                    arena.div(shifted, b)
                })
                .collect()
        }
        // ── A*var^var + D = 0 ────────────────────────────────────────
        //   var·ln(var) = ln(-D/A)  →  var = exp(W(ln(-D/A)))
        //   Besides, the integers with n^n = r: n ≥ 2 exactly (`x^x = 27`:
        //   `3`, not `exp(W(ln 27))`), the negative ones (`(−2)^(−2) = 1/4`,
        //   `(−1)^(−1) = −1`, real values of x^x off the positive axis that no
        //   real branch of W reaches; SymPy misses them), and `0` for
        //   `r = 1` (`0⁰ = 1`, as SymPy's `[0, 1]`).
        ((0, 0, 0, 0, 1), None) => {
            let a = pow_terms[0];
            let neg_d = arena.neg(d);
            let r = arena.div(neg_d, a);
            let r = crate::transforms::eval::eval(arena, r);
            let q = arena.as_num(r).cloned();
            let positive = q
                .as_ref()
                .filter(|q| q.is_integer() && q.is_positive())
                .and_then(|q| self_power_root(&q.to_integer()))
                .filter(|&n| n >= 2);
            let mut sols: Vec<ExprId> = match positive {
                Some(n) => vec![arena.int(i64::from(n))],
                None => {
                    let ln_r = arena.ln(r);
                    lambert_branches(arena, ln_r, cx)
                        .into_iter()
                        .map(|w| arena.exp(w))
                        .collect()
                }
            };
            if let Some(q) = q {
                // (−m)^(−m) = (−1)^m/m^m
                if q.numer().abs().is_one()
                    && let Some(m) = self_power_root(q.denom())
                    && (m % 2 == 0) == q.is_positive()
                {
                    sols.push(arena.int(-i64::from(m)));
                }
                if q.is_one() {
                    sols.push(arena.zero);
                }
            }
            sols
        }
        _ => return None,
    };
    Some(
        solutions
            .into_iter()
            .map(|s| Solution {
                value: crate::transforms::eval::eval(arena, s),
            })
            .collect(),
    )
}

/// Logarithmic Lambert forms in a power of the variable: `var` only in
/// `ln(var)` and in powers `varⁿ` of one rational exponent `n ≠ 0, 1`
/// (`x²·ln x = 1`, `ln(x)/x² = c`, `ln x = x³`).  With `u = xⁿ` and
/// `ln x = ln(u)/n` (for `x > 0`; a negative `x` makes `ln x` non-real)
/// they are forms of [`try_solve_lambert`] in `u`, and `x = u^(1/n)`;
/// [`solve_impl`] checks the candidates.  A logarithm `ln(B·x)` of a
/// constant multiple is first rescaled, `x = y/B`.  SymPy:
/// `solve(x**2*log(x) - 1)` → `[exp(LambertW(2)/2)]`, `solve(x**5*log(2*x)
/// - 3)` → `[exp(LambertW(480)/5)/2]`; refused before 0.44.
fn try_lambert_power_log(arena: &mut Arena, expr: ExprId, var: ExprId, cx: &SolveCx) -> Solutions {
    candidates_found(power_log_candidates(arena, expr, var, cx))
}

/// The candidates of [`try_lambert_power_log`]; `None` when the equation
/// does not have its shape.
fn power_log_candidates(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Option<Vec<Solution>> {
    let logs: Vec<ExprId> = crate::base::walk::post_order_ids(arena, expr)
        .into_iter()
        .filter_map(|id| match *arena.node(id) {
            ExprNode::Ln(a) if expr_contains_var(arena, a, var) => Some(a),
            _ => None,
        })
        .collect();
    let [arg] = logs[..] else {
        return None;
    };
    if arg == var {
        return power_log_in(arena, expr, var, cx);
    }
    let scale = extract_var_coeff_in_product(arena, arg, var)?;
    // y = B·x: xⁿ = B⁻ⁿ·yⁿ for an integer n, or a positive B
    let powers: Vec<(ExprId, Q)> = crate::base::walk::post_order_ids(arena, expr)
        .into_iter()
        .filter_map(|id| match *arena.node(id) {
            ExprNode::Pow(b, e) if b == var => arena.as_num(e).map(|q| (id, q.clone())),
            _ => None,
        })
        .collect();
    let [(pow_node, ref n)] = powers[..] else {
        return None;
    };
    if !n.is_integer() && crate::poly::algebraic::sign_checked(arena, scale) != Some(1) {
        return None;
    }
    let y = arena.symbol("__y_lambert");
    let n_id = arena.num_ratio(n.clone());
    let neg_n = arena.num_ratio(-n);
    let y_n = arena.pow(y, n_id);
    let b_neg_n = arena.pow(scale, neg_n);
    let repl = arena.mul(&[b_neg_n, y_n]);
    let ln_arg = arena.ln(arg);
    let ln_y = arena.ln(y);
    let scaled = arena.subs_structural(expr, pow_node, repl);
    let scaled = arena.subs_structural(scaled, ln_arg, ln_y);
    let scaled = crate::transforms::eval::eval(arena, scaled);
    if expr_contains_var(arena, scaled, var) {
        return None;
    }
    let sols = power_log_in(arena, scaled, y, cx)?;
    Some(
        sols.into_iter()
            .map(|s| {
                let x = arena.div(s.value, scale);
                Solution {
                    value: crate::transforms::eval::eval(arena, x),
                }
            })
            .collect(),
    )
}

/// `K·x^a·(ln x)^m + d = 0` for an integer `m ≥ 2` and a rational `a ≠ 0`
/// (`x·ln²x = c`): for each `r` with `r^m = −d/K` ([`power_roots`]: both
/// `±√c` for an even `m`), `x^s·ln x = r` with `s = a/m`, and `y = s·ln x`
/// solves `y·e^y = s·r`: `x = exp(W_k(s·r)/s)` on the real branches.  Every
/// candidate is checked by [`solve_impl`].  SymPy: `solve(x*log(x)**2 -
/// 2)` → `[exp(2*LambertW(-sqrt(2)/2)), exp(2*LambertW(sqrt(2)/2))]`;
/// refused before 0.45.
fn try_power_of_log(arena: &mut Arena, expr: ExprId, var: ExprId, cx: &SolveCx) -> Solutions {
    candidates_found(power_of_log_candidates(arena, expr, var, cx))
}

/// The candidates of [`try_power_of_log`]; `None` when the equation does
/// not have its shape.
fn power_of_log_candidates(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Option<Vec<Solution>> {
    let terms: Vec<ExprId> = match arena.node(expr) {
        ExprNode::Add(ch) => ch.to_vec(),
        _ => vec![expr],
    };
    let mut constants: Vec<ExprId> = Vec::new();
    let mut var_term: Option<ExprId> = None;
    for t in terms {
        if !expr_contains_var(arena, t, var) {
            constants.push(t);
        } else if var_term.replace(t).is_some() {
            return None;
        }
    }
    let term = var_term?;
    let factors: Vec<ExprId> = match arena.node(term) {
        ExprNode::Mul(ch) => ch.to_vec(),
        _ => vec![term],
    };
    let ln_x = arena.ln(var);
    let mut k_parts: Vec<ExprId> = Vec::new();
    let mut a = Q::zero();
    let mut m: Option<Q> = None;
    for f in factors {
        if !expr_contains_var(arena, f, var) {
            k_parts.push(f);
            continue;
        }
        if f == var {
            a += Q::one();
            continue;
        }
        match *arena.node(f) {
            ExprNode::Pow(b, e) if b == var => a += arena.as_num(e)?.clone(),
            ExprNode::Pow(b, e) if b == ln_x && m.is_none() => {
                let q = arena.as_num(e)?.clone();
                if !q.is_integer() || q < Q::from_integer(BigInt::from(2)) {
                    return None;
                }
                m = Some(q);
            }
            _ => return None,
        }
    }
    let m = m?;
    if a.is_zero() || constants.is_empty() {
        return None;
    }
    let k = if k_parts.is_empty() {
        arena.one
    } else {
        arena.mul(&k_parts)
    };
    let d = arena.add(&constants);
    let neg_d = arena.neg(d);
    let c = arena.div(neg_d, k);
    let c = crate::transforms::eval::eval(arena, c);
    if arena.is_zero_structural(c) {
        return None;
    }
    let s = &a / &m;
    let s_id = arena.num_ratio(s);
    let mut out: Vec<Solution> = Vec::new();
    for r in power_roots(arena, c, &m)? {
        let arg = arena.mul(&[s_id, r]);
        for w in lambert_branches(arena, arg, cx) {
            let y = arena.div(w, s_id);
            let x = arena.exp(y);
            let x = crate::transforms::eval::eval(arena, x);
            if !out.iter().any(|o| o.value == x) {
                out.push(Solution { value: x });
            }
        }
    }
    Some(out)
}

/// Equations in which the unknown occurs only through `xⁿ` (`n ≥ 2`, the
/// gcd of its integer exponents): `w = xⁿ` is substituted, the equation in
/// `w` solved, and each root `w₀` gives the `x` with `xⁿ = w₀` (the real
/// ones, or all `n` when there is none, [`real_nth_roots`]).  The
/// substitution is exact, so no root is lost, and [`solve_impl`] checks
/// the candidates against the equation in `x`.  `(x² + 1)·e^(x²) = 3` is
/// `(w + 1)·e^w = 3`: `x = ±√(W(3e) − 1)`;
/// refused before 0.45 (SymPy 1.14 solves it too).  `None` for a
/// polynomial, for families in `w` (periodic ones are peeled directly),
/// or when the equation in `w` is not solved.
fn try_monomial_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    let not_found = || Solutions::refused(Reason::NotFound);
    let Some((w, sub, n)) = monomial_substitution(arena, expr, var) else {
        return not_found();
    };
    let (w_roots, missing) = match solve_raw(arena, sub, w, cx)
        .without_conditions()
        .into_found()
    {
        Ok(found) => found,
        Err(nested) if nested.is_nested_periods() => return nested,
        Err(_) => return not_found(),
    };
    let mut out: Vec<ExprId> = Vec::new();
    for r in w_roots {
        if cx
            .period()
            .is_some_and(|p| crate::base::walk::contains(arena, r, p))
        {
            return not_found();
        }
        for x in real_nth_roots(arena, r, n) {
            let x = crate::transforms::eval::eval(arena, x);
            if !out.contains(&x) {
                out.push(x);
            }
        }
    }
    Solutions::with_missing(out, missing)
}

/// The substitution `w = xⁿ` of [`try_monomial_substitution`]: `(w, the
/// equation in w, n)`, or `None` when the equation does not have its shape.
fn monomial_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Option<(ExprId, ExprId, u32)> {
    if polybridge::expr_to_poly(arena, expr, var).is_some() {
        return None;
    }
    // gcd of the exponents of `var`; a bare occurrence makes it 1
    let mut g = BigInt::zero();
    let mut powers: Vec<(ExprId, BigInt)> = Vec::new();
    let order = crate::base::walk::post_order_ids(arena, expr);
    let mut under_pow: rustc_hash::FxHashSet<ExprId> = rustc_hash::FxHashSet::default();
    for &id in &order {
        if let ExprNode::Pow(b, e) = *arena.node(id)
            && b == var
        {
            let q = arena.as_num(e)?;
            if !q.is_integer() {
                return None;
            }
            let k = q.to_integer();
            g = num_integer::Integer::gcd(&g, &k);
            powers.push((id, k));
            under_pow.insert(id);
        }
    }
    // `var` must not occur outside those powers
    for &id in &order {
        if under_pow.contains(&id) {
            continue;
        }
        if arena.node(id).children().contains(&var) && !under_pow.contains(&id) {
            return None;
        }
    }
    if expr == var || g < BigInt::from(2) {
        return None;
    }
    let n = num_traits::ToPrimitive::to_u32(&g)?;
    if u64::from(n) > MAX_LAMBERT_POWER {
        return None;
    }
    let w = arena.symbol("__w_monomial");
    let mut sub = expr;
    for (id, k) in &powers {
        let kk = arena.big_int(k / &g);
        let wk = arena.pow(w, kk);
        sub = arena.subs_structural(sub, *id, wk);
    }
    let sub = crate::transforms::eval::eval(arena, sub);
    if expr_contains_var(arena, sub, var) {
        return None;
    }
    Some((w, sub, n))
}

/// The outcome of a candidate search (Lambert W forms): the candidates
/// found, or refused when there are none or the shape does not apply —
/// such a search proves nothing.
fn candidates_found(candidates: Option<Vec<Solution>>) -> Solutions {
    match candidates {
        Some(found) => Solutions::found(values(found)),
        None => Solutions::refused(Reason::NotFound),
    }
}

/// The roots of a peeling inside a candidate search: `Some` (possibly
/// none) unless refused.  (The peelings of these searches invert `e^(b·x)`
/// or a power of `x` against a constant: they meet no periodic family.)
fn peeled_candidates(outcome: Solutions) -> Option<Vec<ExprId>> {
    match outcome.into_found() {
        Ok((roots, _)) => Some(roots),
        Err(Solutions::Complete(RootSet::Empty(_))) => Some(Vec::new()),
        Err(_) => None,
    }
}

/// [`try_lambert_power_log`] for `ln(var)` itself.
fn power_log_in(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Option<Vec<Solution>> {
    let ln_x = arena.ln(var);
    let mut power: Option<(ExprId, Q)> = None;
    let mut has_log = false;
    let mut stack = vec![expr];
    let mut seen: rustc_hash::FxHashSet<ExprId> = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) || !expr_contains_var(arena, id, var) {
            continue;
        }
        if id == ln_x {
            has_log = true;
            continue;
        }
        match arena.node(id) {
            ExprNode::Add(ch) | ExprNode::Mul(ch) => stack.extend(ch.iter().copied()),
            ExprNode::Neg(b) => stack.push(*b),
            ExprNode::Pow(b, e) if *b == var => {
                let q = arena.as_num(*e)?.clone();
                match &power {
                    Some((p, _)) if *p != id => return None,
                    _ => power = Some((id, q)),
                }
            }
            _ => return None,
        }
    }
    let (pow_node, n) = power?;
    // n = −1 (`ln(x)/x = c`) is left to the numerator `ln x − c·x`, which
    // gives SymPy's `−4·W(−1/4)` rather than the equal `exp(−W(−1/4))`.
    if !has_log || n.is_zero() || n.is_one() || n == -Q::one() {
        return None;
    }
    let u = arena.symbol("__u_lambert");
    let ExprNode::Symbol(u_sym) = *arena.node(u) else {
        return None;
    };
    let inv_n = arena.num_ratio(n.recip());
    let ln_u = arena.ln(u);
    let ln_u_over_n = arena.mul(&[inv_n, ln_u]);
    let sub = arena.subs_structural(expr, pow_node, u);
    let sub = arena.subs_structural(sub, ln_x, ln_u_over_n);
    let sub = crate::transforms::eval::eval(arena, sub);
    if expr_contains_var(arena, sub, var) {
        return None;
    }
    let sols = lambert_candidates(arena, sub, u, u_sym, cx)?;
    Some(
        sols.into_iter()
            .map(|s| {
                // exp(v)^(1/n) = exp(v/n) for a real v (`exp(W(2)/2)`, as
                // SymPy writes it, rather than `√(exp(W(2)))`)
                let node = arena.node(s.value).clone();
                let x = match node {
                    ExprNode::Exp(v)
                        if crate::base::walk::free_symbols(arena, v).is_empty()
                            && crate::poly::algebraic::sign_checked(arena, v).is_some() =>
                    {
                        let w = arena.mul(&[inv_n, v]);
                        arena.exp(w)
                    }
                    _ => arena.pow(s.value, inv_n),
                };
                Solution {
                    value: crate::transforms::eval::eval(arena, x),
                }
            })
            .collect(),
    )
}

/// A product with several `exp` factors with them merged into one
/// (`exp(−1)·exp(−3)` → `exp(−4)`, which the evaluator leaves apart).
fn merge_exp_factors(arena: &mut Arena, e: ExprId) -> ExprId {
    let ExprNode::Mul(factors) = arena.node(e).clone() else {
        return e;
    };
    let mut args: Vec<ExprId> = Vec::new();
    let mut rest: Vec<ExprId> = Vec::new();
    for f in factors.iter() {
        match *arena.node(*f) {
            ExprNode::Exp(a) => args.push(a),
            _ => rest.push(*f),
        }
    }
    if args.len() < 2 {
        return e;
    }
    let sum = arena.add(&args);
    let sum = crate::transforms::eval::eval(arena, sum);
    rest.push(arena.exp(sum));
    let merged = arena.mul(&rest);
    crate::transforms::eval::eval(arena, merged)
}

/// The integer `m ≥ 1` with `m^m = n`, if any.
fn self_power_root(n: &BigInt) -> Option<u32> {
    if !n.is_positive() {
        return None;
    }
    for m in 1u32..=64 {
        let p = num_traits::Pow::pow(BigInt::from(m), m);
        match p.cmp(n) {
            std::cmp::Ordering::Equal => return Some(m),
            std::cmp::Ordering::Greater => return None,
            std::cmp::Ordering::Less => {}
        }
    }
    None
}

/// The Lambert W values `W_k(arg)` a Lambert solution takes: the principal
/// branch, and `W₋₁(arg)` as well when `arg` is certainly in `(−1/e, 0)`,
/// where both branches are real (SymPy's `_lambert` keeps `k = −1` only
/// when `LambertW(arg, -1)` is known to be real; here the interval is
/// decided with certified signs).  At `arg = −1/e` the two coincide, and
/// for `arg ≥ 0` or a symbolic `arg` only `W₀` is taken — the other
/// branches give the complex roots, infinitely many, which `solve` does
/// not enumerate.
///
/// In a general solve (a solution set, [`SolveCx::all_branches`]) a
/// symbolic `arg` not known to be nonnegative also gives `W₋₁(arg)`, real for `−1/e ≤ arg < 0`: up to
/// 0.46 the real solutions of `a/x + e^(x/2) = 0` were `{2·W(−a/2)}`,
/// without `2·W₋₁(−a/2)` (for `0 < a < 2/e` both are real), and the set
/// cut to the real line drops it where it is not real.
fn lambert_branches(arena: &mut Arena, arg: ExprId, cx: &SolveCx) -> Vec<ExprId> {
    let arg = crate::transforms::eval::eval(arena, arg);
    let arg = merge_exp_factors(arena, arg);
    if cx.all_branches()
        && !crate::base::walk::free_symbols(arena, arg).is_empty()
        && crate::base::complex::query_prop(
            arena,
            arg,
            crate::base::assumptions::Props::NONNEGATIVE,
        ) != Some(true)
    {
        let principal = arena.lambertw(arg);
        let neg_one = arena.neg_one;
        let lower = arena.lambertw_branch(arg, neg_one);
        return vec![principal, lower];
    }
    if crate::base::walk::free_symbols(arena, arg).is_empty()
        && crate::poly::algebraic::sign_checked(arena, arg) == Some(-1)
    {
        let neg_one = arena.neg_one;
        let inv_e = arena.exp(neg_one);
        let shifted = arena.add(&[arg, inv_e]);
        match crate::poly::algebraic::sign_checked(arena, shifted) {
            Some(1) => {
                let principal = arena.lambertw(arg);
                let principal = exact_lambert_value(arena, arg, principal, true);
                let lower = arena.lambertw_branch(arg, neg_one);
                let lower = exact_lambert_value(arena, arg, lower, false);
                return vec![principal, lower];
            }
            // The branch point `−1/e` in a form the evaluator does not
            // recognise (`−√(4·e⁻²)/2`): `W(−1/e) = −1` on both branches,
            // rather than a `W` evaluated at its branch point.
            Some(0) => return vec![neg_one],
            _ => {}
        }
    }
    let principal = arena.lambertw(arg);
    vec![exact_lambert_value(arena, arg, principal, true)]
}

/// Largest denominator of the rational multiples `y = r·s` that
/// [`exact_lambert_value`] tries.
const LAMBERT_VALUE_MAX_DENOMINATOR: i64 = 12;

/// `w = W_k(arg)` (`principal`: `k = 0`, else `k = −1`) as the exact `y`
/// with `y·e^y = arg` when there is a simple one: `y = r·s` for a small
/// rational `r` and `s` either `1` or a logarithm `ln m` (`m` a positive
/// rational) in `arg`, `r` read off the value of `w` and then verified
/// exactly — `y·e^y − arg` is an exact zero, and `y ≥ −1` (`W₀`) or
/// `y ≤ −1` (`W₋₁`), the ranges on which `y ↦ y·e^y` is injective, so
/// `W_k(arg) = y`.  The evaluator knows `W(t·p^t·ln p)` for integers `t`,
/// `p`, not `W(√2·ln(2)/2) = ln(2)/2` or `W(−2·ln(5/2)/5) = −ln(5/2)`, and
/// before 0.45 such roots stayed as `W` values whose sign no evaluation
/// decides: `2·2ˣ + 4x − 2 = 0` for a positive `x` gave `[1/2 −
/// W(√2·ln(2)/2)/ln 2]`, which is `0`.  `w` itself when nothing is found.
fn exact_lambert_value(arena: &mut Arena, arg: ExprId, w: ExprId, principal: bool) -> ExprId {
    use crate::base::walk;
    if !matches!(
        arena.node(w),
        ExprNode::LambertW(_) | ExprNode::Apply(..) | ExprNode::Lib(..)
    ) || !walk::free_symbols(arena, arg).is_empty()
        || walk::has_unevaluated(arena, arg)
    {
        return w;
    }
    let Ok(value) = crate::transforms::evalf::evalf_complex64(arena, w) else {
        return w;
    };
    // (the values only guide the guess of `r`; `y` is verified exactly below)
    if value.im != 0.0 || !value.re.is_finite() {
        return w;
    }
    let mut scales: Vec<ExprId> = vec![arena.one];
    for id in walk::post_order_ids(arena, arg) {
        if let ExprNode::Ln(m) = *arena.node(id)
            && arena
                .as_num(m)
                .is_some_and(|q| q.is_positive() && !q.is_one())
            && !scales.contains(&id)
        {
            scales.push(id);
        }
    }
    for s in scales {
        let Ok(sv) = crate::transforms::evalf::evalf_complex64(arena, s) else {
            continue;
        };
        if sv.re == 0.0 || !sv.re.is_finite() {
            continue;
        }
        let Some(r) = small_rational_near(value.re / sv.re, LAMBERT_VALUE_MAX_DENOMINATOR) else {
            continue;
        };
        let r = arena.num_ratio(r);
        let y = arena.mul(&[r, s]);
        let y = crate::transforms::eval::eval(arena, y);
        let ey = arena.exp(y);
        let product = arena.mul(&[y, ey]);
        let diff = arena.sub(product, arg);
        let diff = crate::transforms::eval::eval(arena, diff);
        let exact = arena.is_zero_structural(diff)
            || crate::poly::algebraic::is_zero_checked(arena, diff) == Some(true);
        if !exact {
            continue;
        }
        let one = arena.one;
        let y_plus_one = arena.add(&[y, one]);
        let y_plus_one = crate::transforms::eval::eval(arena, y_plus_one);
        let on_branch = match crate::poly::algebraic::sign_checked(arena, y_plus_one) {
            Some(s) => (principal && s >= 0) || (!principal && s <= 0),
            None => false,
        };
        if on_branch {
            return y;
        }
    }
    w
}

/// The rational with denominator at most `max_den` within `10⁻¹²`
/// (relative) of `x`, the one with the least denominator (a guess that
/// [`exact_lambert_value`] verifies exactly).
fn small_rational_near(x: f64, max_den: i64) -> Option<Q> {
    if !x.is_finite() || x.abs() > 1e9 {
        return None;
    }
    for den in 1..=max_den {
        let num = (x * den as f64).round();
        if (x - num / den as f64).abs() <= 1e-12 * x.abs().max(1.0) {
            return Some(Q::new(BigInt::from(num as i64), BigInt::from(den)));
        }
    }
    None
}

/// `A·exp(B·var)` (and `A·var·exp(B·var)`): the coefficient `A` and the
/// rate `B`.
#[derive(Clone, Copy)]
struct LambertExp {
    coeff: ExprId,
    rate: ExprId,
}

/// `A·ln(B·var + E)` (and `A·var·ln(B·var + E)`): the coefficient `A`, the
/// scale `B` and the shift `E`.
#[derive(Clone, Copy)]
struct LambertLn {
    coeff: ExprId,
    scale: ExprId,
    shift: ExprId,
}

/// Classification of a single additive term for LambertW analysis.
enum LambertTermClass {
    /// `A * var` — linear in the solve variable.
    Linear(ExprId),
    /// `A * exp(B * var)` — exponential in the solve variable.
    ExpVar(LambertExp),
    /// `A * var * exp(B * var)` — mixed polynomial-exponential.
    VarExpVar(LambertExp),
    /// `A * var * ln(B * var)`.
    VarLnVar(LambertLn),
    /// `A * ln(B * var)`.
    LnVar(LambertLn),
    /// `A * var^var`.
    PowVarVar(ExprId),
}

/// Classify a single additive term (known to contain `var`) into a
/// LambertW-relevant category, or return `None` if unrecognizable.
fn classify_lambert_term(arena: &mut Arena, term: ExprId, var: ExprId) -> Option<LambertTermClass> {
    // `-(inner)`: classify `inner` and negate the coefficient.  This
    // handles cases like `-(x*exp(x))` that remain as Neg nodes rather than
    // being absorbed into a Mul with -1.
    let (term, negate) = match arena.node(term).clone() {
        ExprNode::Neg(inner) => (inner, true),
        _ => (term, false),
    };
    let factors: Vec<ExprId> = match arena.node(term).clone() {
        ExprNode::Mul(children) => children.to_vec(),
        _ => vec![term],
    };
    let mut const_factors: Vec<ExprId> = Vec::new();
    let mut has_var = false;
    let mut exp_inner: Option<ExprId> = None;
    let mut ln_inner: Option<ExprId> = None;
    let mut pow_var_var = false;
    for &child in &factors {
        if !expr_contains_var(arena, child, var) {
            const_factors.push(child);
            continue;
        }
        if child == var {
            if has_var {
                return None; // var appears twice → var²
            }
            has_var = true;
            continue;
        }
        match arena.node(child).clone() {
            ExprNode::Exp(inner) if exp_inner.is_none() => exp_inner = Some(inner),
            // a^f = exp(f·ln a) for a constant base (principal powers):
            // `2^x = 3x` as `exp(x·ln 2) = 3x` (SymPy: `solve(2**x - 3*x)` →
            // `[-LambertW(-log(2)/3)/log(2), -LambertW(-log(2)/3, -1)/log(2)]`).
            ExprNode::Pow(base, e)
                if exp_inner.is_none()
                    && !expr_contains_var(arena, base, var)
                    && !arena.is_zero_structural(base) =>
            {
                let ln_base = arena.ln(base);
                let ln_base = crate::transforms::eval::eval(arena, ln_base);
                exp_inner = Some(arena.mul(&[e, ln_base]));
            }
            ExprNode::Ln(inner) if ln_inner.is_none() => ln_inner = Some(inner),
            ExprNode::Pow(base, exp) if base == var && exp == var && !pow_var_var => {
                pow_var_var = true;
            }
            _ => return None, // unrecognized var-dependent factor
        }
    }
    let mut coeff = match const_factors.len() {
        0 => arena.one,
        1 => const_factors[0],
        _ => arena.mul(&const_factors),
    };
    if negate {
        coeff = arena.neg(coeff);
    }
    let class = match (has_var, exp_inner, ln_inner, pow_var_var) {
        (true, None, None, false) => LambertTermClass::Linear(coeff),
        (has_var, Some(inner), None, false) => {
            // exp(B·var + c) = e^c·exp(B·var) (`2^(x + 3) = 8·2^x`; refused
            // before 0.44)
            let rate = match extract_var_coeff_in_product(arena, inner, var) {
                Some(rate) => rate,
                None => {
                    let (rate, offset) = linear_rate(arena, inner, var)?;
                    let e_offset = arena.exp(offset);
                    coeff = arena.mul(&[coeff, e_offset]);
                    coeff = crate::transforms::eval::eval(arena, coeff);
                    rate
                }
            };
            let e = LambertExp { coeff, rate };
            if has_var {
                LambertTermClass::VarExpVar(e)
            } else {
                LambertTermClass::ExpVar(e)
            }
        }
        (has_var, None, Some(inner), false) => {
            let (scale, shift) = match extract_var_coeff_in_product(arena, inner, var) {
                Some(scale) => (scale, arena.zero),
                None => linear_rate(arena, inner, var)?,
            };
            let l = LambertLn {
                coeff,
                scale,
                shift,
            };
            if has_var {
                LambertTermClass::VarLnVar(l)
            } else {
                LambertTermClass::LnVar(l)
            }
        }
        (false, None, None, true) => LambertTermClass::PowVarVar(coeff),
        _ => return None,
    };
    Some(class)
}

// ═══════════════════════════════════════════════════════════════════════════
// Lambert W: a power times an exponential
// ═══════════════════════════════════════════════════════════════════════════

/// Largest `|p|` of a power `u^p` (integer `p`) whose `p`-th roots
/// [`try_solve_power_exp`] takes.
const MAX_LAMBERT_POWER: u64 = 12;

/// An additive term `K·L^a·e^(b·Mⁿ)` of [`try_solve_power_exp`]: `L` the
/// variable or a polynomial of degree 1 in it (`None` when `a = 0`), `a`
/// rational, `K` and the rate `b` free of the variable (`b = 0` without an
/// exponential), and the exponent `b·x` (`M = x`, `n = 1`) or `b·Mⁿ` with
/// `M = x + s` and `n ≥ 2` (`x·e^(x²)`, `(x + 1)·e^(−(x + 1)²)`).
#[derive(Clone)]
struct PowExpTerm {
    coeff: ExprId,
    base: Option<ExprId>,
    power: Q,
    rate: ExprId,
    /// `M`: the variable for a rate of degree 1.
    rate_base: ExprId,
    /// `n`: 1 for `e^(b·x)`, else an integer `≥ 2` or `1/m`.
    rate_power: Q,
}

/// `arg = b·Mⁿ + c₀` with `M = var + s` and `1 ≤ n ≤`
/// [`MAX_LAMBERT_POWER`] an integer or `n = 1/m` (`b·√(x + 1)`), `b ≠ 0`,
/// `b`, `s`, `c₀` free of `var`: `(b, c₀, M, n)`, with `M = var` for `n =
/// 1` (the linear rates of [`linear_rate`]).
fn monomial_rate(
    arena: &mut Arena,
    arg: ExprId,
    var: ExprId,
) -> Option<(ExprId, ExprId, ExprId, Q)> {
    let Some(coeffs) = symbolic_poly_coeffs(arena, arg, var) else {
        return radical_rate(arena, arg, var);
    };
    let n = coeffs.len().checked_sub(1)?;
    if n == 0 || n as u64 > MAX_LAMBERT_POWER {
        return None;
    }
    let lead = coeffs[n];
    if arena.is_zero_structural(lead) {
        return None;
    }
    if n == 1 {
        return Some((lead, coeffs[0], var, Q::one()));
    }
    // M = x + s with s = c_{n−1}/(n·b); every other coefficient must match.
    let n_id = arena.int(n as i64);
    let nb = arena.mul(&[n_id, lead]);
    let s = arena.div(coeffs[n - 1], nb);
    let s = crate::transforms::eval::eval(arena, s);
    let m = arena.add(&[var, s]);
    let m = crate::transforms::eval::eval(arena, m);
    let m_n = arena.pow(m, n_id);
    let model = arena.mul(&[lead, m_n]);
    let rest = arena.sub(arg, model);
    let rest = crate::transforms::expand::expand(arena, rest);
    let rest = crate::transforms::eval::eval(arena, rest);
    if expr_contains_var(arena, rest, var) {
        return None;
    }
    Some((lead, rest, m, Q::from_integer(BigInt::from(n))))
}

/// `arg = b·P^(1/m) + c₀` for one `P = α·(var + s)` with `α > 0` and an
/// integer `2 ≤ m ≤` [`MAX_LAMBERT_POWER`]: `(b·α^(1/m), c₀, var + s, 1/m)`
/// ([`monomial_rate`]).
fn radical_rate(
    arena: &mut Arena,
    arg: ExprId,
    var: ExprId,
) -> Option<(ExprId, ExprId, ExprId, Q)> {
    let mut radical: Option<(ExprId, ExprId, Q)> = None;
    for id in crate::base::walk::post_order_ids(arena, arg) {
        if let ExprNode::Pow(p, e) = *arena.node(id)
            && expr_contains_var(arena, p, var)
        {
            let r = arena.as_num(e)?.clone();
            match &radical {
                Some((node, _, _)) if *node != id => return None,
                _ => radical = Some((id, p, r)),
            }
        }
    }
    let (node, p, r) = radical?;
    if !r.numer().is_one() || r.denom() < &BigInt::from(2) {
        return None;
    }
    let m = num_traits::ToPrimitive::to_u64(r.denom())?;
    if m > MAX_LAMBERT_POWER {
        return None;
    }
    let (alpha, beta) = linear_rate(arena, p, var)?;
    if !crate::base::walk::free_symbols(arena, alpha).is_empty()
        || crate::poly::algebraic::sign_checked(arena, alpha) != Some(1)
    {
        return None;
    }
    let y = arena.symbol("__y_radical_rate");
    let sub = arena.subs_structural(arg, node, y);
    let sub = crate::transforms::eval::eval(arena, sub);
    if expr_contains_var(arena, sub, var) {
        return None;
    }
    let (b, c0) = linear_rate(arena, sub, y)?;
    let s = arena.div(beta, alpha);
    let rate_base = arena.add(&[var, s]);
    let rate_base = crate::transforms::eval::eval(arena, rate_base);
    let r_id = arena.num_ratio(r.clone());
    let alpha_r = arena.pow(alpha, r_id);
    let b = arena.mul(&[b, alpha_r]);
    let b = crate::transforms::eval::eval(arena, b);
    Some((b, c0, rate_base, r))
}

/// `b = content·L^m` with `L` of degree 1 in `var` (`var` itself, `2x + 1`,
/// `x + a`), or a polynomial over ℚ whose square-free decomposition is one
/// linear factor (`x² + 2x + 1 = (x + 1)²`).
fn linear_power_base(arena: &mut Arena, b: ExprId, var: ExprId) -> Option<(ExprId, u32, Q)> {
    if b == var {
        return Some((var, 1, Q::one()));
    }
    if let Some(p) = polybridge::expr_to_poly(arena, b, var) {
        match p.degree()? {
            0 => return None,
            1 => return Some((b, 1, Q::one())),
            _ => {}
        }
        let (content, factors) = p.sqf_list();
        let [(l, m)] = factors.as_slice() else {
            return None;
        };
        if l.degree() != Some(1) {
            return None;
        }
        let l_expr = polybridge::poly_to_expr(arena, l, var);
        let l_expr = crate::transforms::eval::eval(arena, l_expr);
        return Some((l_expr, *m, content));
    }
    linear_rate(arena, b, var)?;
    Some((b, 1, Q::one()))
}

/// Split a term containing `var` as `K·L^a·e^(b·x)` ([`PowExpTerm`]):
/// powers of the variable or of one linear polynomial, exponentials
/// `exp(a·x + c)` and constant-base powers `c^(a·x + d)` (`c` positive or
/// symbolic, as `exp((a·x + d)·ln c)`).  `None` for any other factor.
fn pow_exp_term(arena: &mut Arena, term: ExprId, var: ExprId) -> Option<PowExpTerm> {
    let (term, negate) = match *arena.node(term) {
        ExprNode::Neg(inner) => (inner, true),
        _ => (term, false),
    };
    let factors: Vec<ExprId> = match arena.node(term) {
        ExprNode::Mul(ch) => ch.to_vec(),
        _ => vec![term],
    };
    let mut consts: Vec<ExprId> = Vec::new();
    if negate {
        consts.push(arena.neg_one);
    }
    let mut base: Option<ExprId> = None;
    let mut power = Q::zero();
    let mut rates: Vec<ExprId> = Vec::new();
    // (M, n) of the exponents `b·Mⁿ`, one for all exponential factors
    let mut rate_shape: Option<(ExprId, Q)> = None;
    let mut same_shape = |shape: (ExprId, Q)| match &rate_shape {
        Some(s) => *s == shape,
        None => {
            rate_shape = Some(shape);
            true
        }
    };
    for f in factors {
        if !expr_contains_var(arena, f, var) {
            consts.push(f);
            continue;
        }
        let (b, e) = match arena.node(f).clone() {
            ExprNode::Exp(arg) => {
                let (a, c0, m, n) = monomial_rate(arena, arg, var)?;
                if !same_shape((m, n)) {
                    return None;
                }
                rates.push(a);
                consts.push(arena.exp(c0));
                continue;
            }
            ExprNode::Pow(c, e) if !expr_contains_var(arena, c, var) => {
                let positive = if crate::base::walk::free_symbols(arena, c).is_empty() {
                    crate::poly::algebraic::sign_checked(arena, c) == Some(1)
                } else {
                    true
                };
                if !positive {
                    return None;
                }
                let (a, c0, m, n) = monomial_rate(arena, e, var)?;
                if !same_shape((m, n)) {
                    return None;
                }
                let ln_c = arena.ln(c);
                let ln_c = crate::transforms::eval::eval(arena, ln_c);
                rates.push(arena.mul(&[a, ln_c]));
                consts.push(arena.pow(c, c0));
                continue;
            }
            ExprNode::Pow(b, e) if !expr_contains_var(arena, e, var) => {
                (b, arena.as_num(e)?.clone())
            }
            _ => (f, Q::one()),
        };
        let (l, m, content) = linear_power_base(arena, b, var)?;
        // (c·L^m)^e = c^e·L^(m·e) for an integer e only (principal powers)
        if (m != 1 || !content.is_one()) && !e.is_integer() {
            return None;
        }
        match base {
            Some(prev) if prev != l => return None,
            _ => base = Some(l),
        }
        power += &e * Q::from_integer(BigInt::from(m));
        if !content.is_one() {
            let c_id = arena.num_ratio(content);
            let e_id = arena.num_ratio(e);
            consts.push(arena.pow(c_id, e_id));
        }
    }
    let coeff = if consts.is_empty() {
        arena.one
    } else {
        let k = arena.mul(&consts);
        crate::transforms::eval::eval(arena, k)
    };
    let rate = if rates.is_empty() {
        arena.zero
    } else {
        let r = arena.add(&rates);
        crate::transforms::eval::eval(arena, r)
    };
    if power.is_zero() {
        base = None;
    }
    let (rate_base, rate_power) = rate_shape.unwrap_or((var, Q::one()));
    Some(PowExpTerm {
        coeff,
        base,
        power,
        rate,
        rate_base,
        rate_power,
    })
}

/// Merge the terms of one rate into one: `x·eˣ + eˣ = (x + 1)·eˣ`.  The
/// terms of a rate must then be `Kᵢ·xⁿⁱ` (nonnegative integers `nᵢ`,
/// rational ratios `Kᵢ/K₀`), and their polynomial `c·L^m` for one linear
/// `L`; `None` otherwise.
fn merge_by_rate(
    arena: &mut Arena,
    terms: Vec<PowExpTerm>,
    var: ExprId,
) -> Option<Vec<PowExpTerm>> {
    let mut groups: Vec<Vec<PowExpTerm>> = Vec::new();
    for t in terms {
        let same = |g: &&mut Vec<PowExpTerm>| {
            g[0].rate == t.rate
                && (arena.is_zero_structural(t.rate)
                    || (g[0].rate_base == t.rate_base && g[0].rate_power == t.rate_power))
        };
        match groups.iter_mut().find(same) {
            Some(g) => g.push(t),
            None => groups.push(vec![t]),
        }
    }
    let mut out = Vec::with_capacity(groups.len());
    for mut g in groups {
        if g.len() == 1 {
            out.push(g.pop()?);
            continue;
        }
        let k0 = g[0].coeff;
        let rate = g[0].rate;
        let (rate_base, rate_power) = (g[0].rate_base, g[0].rate_power.clone());
        if !rate_power.is_one() {
            return None;
        }
        let mut p = Poly::zero();
        for t in &g {
            if t.base.is_some_and(|b| b != var) || !t.power.is_integer() || t.power.is_negative() {
                return None;
            }
            let ratio = arena.div(t.coeff, k0);
            let ratio = crate::transforms::eval::eval(arena, ratio);
            let r = arena.as_num(ratio)?.clone();
            let deg = num_traits::ToPrimitive::to_usize(&t.power.to_integer())?;
            if deg > MAX_SUBSTITUTION_DEGREE as usize {
                return None;
            }
            p = p.add(&Poly::monomial(r, deg));
        }
        let (content, factors) = p.sqf_list();
        if content.is_zero() {
            return None;
        }
        let content_id = arena.num_ratio(content);
        let coeff = arena.mul(&[k0, content_id]);
        let coeff = crate::transforms::eval::eval(arena, coeff);
        match factors.as_slice() {
            [] => out.push(PowExpTerm {
                coeff,
                base: None,
                power: Q::zero(),
                rate,
                rate_base,
                rate_power: rate_power.clone(),
            }),
            [(l, m)] if l.degree() == Some(1) => {
                let l_expr = polybridge::poly_to_expr(arena, l, var);
                let l_expr = crate::transforms::eval::eval(arena, l_expr);
                out.push(PowExpTerm {
                    coeff,
                    base: Some(l_expr),
                    power: Q::from_integer(BigInt::from(*m)),
                    rate,
                    rate_base,
                    rate_power: rate_power.clone(),
                });
            }
            _ => return None,
        }
    }
    Some(out)
}

/// `K·L^a·e^(b·Mⁿ) = c` with `L = α·x + β`, `M = x + s`, `n ≥ 2` an
/// integer or `n = 1/m` ([`power_exp_monomial_solutions`]).
struct PowExpMonomial<'q> {
    k: ExprId,
    a: &'q Q,
    b: ExprId,
    c: ExprId,
    alpha: ExprId,
    beta: ExprId,
    rate_base: ExprId,
    n: &'q Q,
}

/// The candidates of `K·L^a·e^(b·Mⁿ) = c` ([`PowExpMonomial`]) when `L`
/// is a constant multiple of `M` (`x·e^(x²) = c`, `(2x + 2)²·e^(−(x +
/// 1)²) = c`): with `u = L`, `e^(b·Mⁿ) = e^(b′·uⁿ)` for `b′ = b/αⁿ`, and
/// [`power_exp_monomial_roots`] gives `u`; `x = (u − β)/α`.  For `a = 0`
/// the exponential alone is inverted.  `root_of_base` adds `L = 0` (the
/// two-term form with both powers positive).  Every candidate is checked
/// against the equation by [`solve_impl`].
fn power_exp_monomial_solutions(
    arena: &mut Arena,
    eq: &PowExpMonomial<'_>,
    root_of_base: bool,
    var: ExprId,
    cx: &SolveCx,
) -> Option<Vec<Solution>> {
    let mut out: Vec<Solution> = Vec::new();
    let mut push = |arena: &mut Arena, value: ExprId| {
        let value = crate::transforms::eval::eval(arena, value);
        if !out.iter().any(|s| s.value == value) {
            out.push(Solution { value });
        }
    };
    let n_id = arena.num_ratio(eq.n.clone());
    if eq.a.is_zero() {
        let m_n = arena.pow(eq.rate_base, n_id);
        let exponent = arena.mul(&[eq.b, m_n]);
        let generator = arena.exp(exponent);
        let c_over_k = arena.div(eq.c, eq.k);
        let c_over_k = crate::transforms::eval::eval(arena, c_over_k);
        for s in peeled_candidates(solve_by_peeling(arena, generator, c_over_k, var, cx))? {
            push(arena, s);
        }
    } else {
        // L = α·M exactly: β = α·s
        let s = arena.sub(eq.rate_base, var);
        let alpha_s = arena.mul(&[eq.alpha, s]);
        let gap = arena.sub(eq.beta, alpha_s);
        let gap = crate::transforms::expand::expand(arena, gap);
        let gap = crate::transforms::eval::eval(arena, gap);
        if !arena.is_zero_structural(gap) {
            return None;
        }
        // (α·u)^(1/m) = α^(1/m)·u^(1/m) for a positive α only
        if !eq.n.is_integer() && crate::poly::algebraic::sign_checked(arena, eq.alpha) != Some(1) {
            return None;
        }
        let alpha_n = arena.pow(eq.alpha, n_id);
        let b_u = arena.div(eq.b, alpha_n);
        let b_u = crate::transforms::eval::eval(arena, b_u);
        let c_u = arena.div(eq.c, eq.k);
        let c_u = crate::transforms::eval::eval(arena, c_u);
        for u in power_exp_monomial_roots(arena, eq.a, b_u, eq.n, c_u, cx)? {
            let shifted = arena.sub(u, eq.beta);
            let x = arena.div(shifted, eq.alpha);
            push(arena, x);
        }
    }
    if root_of_base {
        let neg_beta = arena.neg(eq.beta);
        let x0 = arena.div(neg_beta, eq.alpha);
        push(arena, x0);
    }
    Some(out)
}

/// The candidates `u` of `u^a·e^(b·uⁿ) = c` for a rational `a = p/q ≠ 0`
/// and an integer `n ≥ 2`.  Raised to the power `q·n` (integer powers, so
/// every root of the equation solves the result) it is `w^p·e^(b·q·n·w) =
/// c^(q·n)` in `w = uⁿ`, solved as in [`try_solve_power_exp`] (the values
/// `r` with `r^p = c^(q·n)` of [`power_roots`], then `w = (p/b″)·W_k((b″/p)·r)`
/// with `b″ = b·q·n`).  For `a = ±1` each `w` gives the one root `u =
/// (c·e^(−b·w))^(1/a)` (then `uⁿ = w` and `u^a·e^(b·uⁿ) = c` exactly):
/// SymPy's `solve(x*exp(x**2) - c)` → `[c*exp(-LambertW(2*c**2)/2)]`.
/// When `a = k·n` for an integer `k` no power is raised: `u^a = w^k` and
/// `w^k·e^(b·w) = c` is the linear form, every real `n`-th root of each
/// of its `w` a root (`x²·e^(x²) = c`: `±√W(c)`).  Otherwise the
/// candidates are the real `n`-th roots of `w` (both signs for an even
/// `n`), some of which may solve `u^a·e^(b·uⁿ) = ζ·c` for a root of unity
/// `ζ ≠ 1` instead; [`solve_impl`] checks them against the equation
/// (`x³·e^(x²) = 5` keeps `+√w` only), so that case needs a constant
/// `c` and `b` (`None` otherwise: a candidate with parameters is kept
/// unchecked).
///
/// For `n = 1/m` the unknown is `w = u^(1/m)` (principal), `u = w^m` and
/// `u^a = w^(m·a)` on the principal sector of `w`: `w^(m·a)·e^(b·w) = c`
/// is the linear form, `w = (m·a/b)·W_k((b/(m·a))·r)` for `r^(m·a) = c`,
/// and the candidates `u = w^m` are checked (`x²·e^(√x/2) = 2`); a `w`
/// outside the principal sector gives a non-root, so again `c` and `b`
/// must be constants.
///
/// Before 0.45 all these were refused ("not polynomial").
fn power_exp_monomial_roots(
    arena: &mut Arena,
    a: &Q,
    b: ExprId,
    n: &Q,
    c: ExprId,
    cx: &SolveCx,
) -> Option<Vec<ExprId>> {
    let constant = crate::base::walk::free_symbols(arena, b).is_empty()
        && crate::base::walk::free_symbols(arena, c).is_empty();
    if n.is_integer() && a.is_integer() {
        let n_int = n.to_integer();
        let a_int = a.to_integer();
        if !n_int.is_zero() && (&a_int % &n_int).is_zero() {
            // u^a = w^k exactly, k = a/n
            let k = Q::from_integer(&a_int / &n_int);
            let n = num_traits::ToPrimitive::to_u32(&n_int)?;
            let k_id = arena.num_ratio(k.clone());
            let b_over_k = arena.div(b, k_id);
            let k_over_b = arena.div(k_id, b);
            let mut out: Vec<ExprId> = Vec::new();
            for r in power_roots(arena, c, &k)? {
                let arg = arena.mul(&[b_over_k, r]);
                for wk in lambert_branches(arena, arg, cx) {
                    let w = arena.mul(&[k_over_b, wk]);
                    let w = crate::transforms::eval::eval(arena, w);
                    for u in real_nth_roots(arena, w, n) {
                        let u = crate::transforms::eval::eval(arena, u);
                        if !out.contains(&u) {
                            out.push(u);
                        }
                    }
                }
            }
            return Some(out);
        }
    }
    if !n.is_integer() {
        if !n.numer().is_one() || !constant {
            return None;
        }
        let m = n.denom().clone();
        let am = a * Q::from_integer(m.clone());
        let am_id = arena.num_ratio(am.clone());
        let b_over_am = arena.div(b, am_id);
        let am_over_b = arena.div(am_id, b);
        let m_id = arena.big_int(m);
        let mut out: Vec<ExprId> = Vec::new();
        for r in power_roots(arena, c, &am)? {
            let arg = arena.mul(&[b_over_am, r]);
            for wk in lambert_branches(arena, arg, cx) {
                let w = arena.mul(&[am_over_b, wk]);
                let u = arena.pow(w, m_id);
                let u = crate::transforms::eval::eval(arena, u);
                if !out.contains(&u) {
                    out.push(u);
                }
            }
        }
        return Some(out);
    }
    let n = num_traits::ToPrimitive::to_u32(n.numer())?;
    let p = num_traits::ToPrimitive::to_i64(a.numer())?;
    let q = num_traits::ToPrimitive::to_i64(a.denom())?;
    let qn = q.checked_mul(i64::from(n))?;
    if p == 0 || qn.unsigned_abs() > MAX_LAMBERT_POWER {
        return None;
    }
    let qn_id = arena.int(qn);
    let p_id = arena.int(p);
    let b2 = arena.mul(&[b, qn_id]);
    let b2 = crate::transforms::eval::eval(arena, b2);
    let c2 = arena.pow(c, qn_id);
    let c2 = crate::transforms::eval::eval(arena, c2);
    let b2_over_p = arena.div(b2, p_id);
    let p_over_b2 = arena.div(p_id, b2);
    let unit = q == 1 && p.abs() == 1;
    if !unit && !constant {
        return None;
    }
    let mut out: Vec<ExprId> = Vec::new();
    for r in power_roots(arena, c2, &Q::from_integer(BigInt::from(p)))? {
        let arg = arena.mul(&[b2_over_p, r]);
        for wk in lambert_branches(arena, arg, cx) {
            let w = arena.mul(&[p_over_b2, wk]);
            let w = crate::transforms::eval::eval(arena, w);
            let candidates: Vec<ExprId> = if unit {
                let bw = arena.mul(&[b, w]);
                let u = if p == 1 {
                    let minus_bw = arena.neg(bw);
                    let e = arena.exp(minus_bw);
                    arena.mul(&[c, e])
                } else {
                    let e = arena.exp(bw);
                    arena.div(e, c)
                };
                vec![u]
            } else {
                real_nth_roots(arena, w, n)
            };
            for u in candidates {
                let u = crate::transforms::eval::eval(arena, u);
                if !out.contains(&u) {
                    out.push(u);
                }
            }
        }
    }
    Some(out)
}

/// The `n`-th roots of `w` (`n ≥ 2`) that [`power_exp_monomial_roots`]
/// tries: the real ones of a real constant `w` that has some (`±w^(1/n)`
/// for a positive `w` and an even `n`, the one real root for an odd `n`),
/// all `n` complex ones of a constant without a real one (a negative `w`
/// and an even `n`, a non-real `w`), as [`power_roots`] does, and
/// `±w^(1/n)` (principal) for a symbolic `w`.
fn real_nth_roots(arena: &mut Arena, w: ExprId, n: u32) -> Vec<ExprId> {
    let even = n.is_multiple_of(2);
    let n_i64 = i64::from(n);
    let constant = crate::base::walk::free_symbols(arena, w).is_empty();
    let sign = if constant {
        crate::poly::algebraic::sign_checked(arena, w)
    } else {
        None
    };
    match sign {
        Some(0) => vec![arena.zero],
        Some(1) => {
            let r = positive_root(arena, w, n_i64);
            if even {
                let minus = arena.neg(r);
                vec![r, minus]
            } else {
                vec![r]
            }
        }
        Some(_) if !even => {
            let minus_w = arena.neg(w);
            let minus_w = crate::transforms::eval::eval(arena, minus_w);
            let r = positive_root(arena, minus_w, n_i64);
            vec![arena.neg(r)]
        }
        _ if constant => match usize::try_from(n) {
            Ok(k) => binomial_roots(arena, w, k)
                .into_iter()
                .map(|s| s.value)
                .collect(),
            Err(_) => Vec::new(),
        },
        _ => {
            let inv_n = arena.rational(1, n_i64);
            let r = arena.pow(w, inv_n);
            if even {
                let minus = arena.neg(r);
                vec![r, minus]
            } else {
                vec![r]
            }
        }
    }
}

/// The values `r` with `r^a = c` that [`try_solve_power_exp`] takes, for a
/// rational `a = p/q`: for `q = 1` every real `|p|`-th root of `c` (or of
/// `1/c` for `p < 0`) when `c` is a real constant with one (`±√c` for an
/// even `p` and `c > 0`, the real root for an odd `p`), all `|p|` complex
/// roots when it has none (`c < 0`, `p` even, or `c` not real), and `c^(1/p)`
/// (with `−c^(1/p)` for an even `p`) for a symbolic `c`; for `q > 1` the
/// principal `c^(1/a)`, the one positive value of a principal power.
/// `None` beyond [`MAX_LAMBERT_POWER`].
fn power_roots(arena: &mut Arena, c: ExprId, a: &Q) -> Option<Vec<ExprId>> {
    if !a.denom().is_one() {
        let inv = arena.num_ratio(a.recip());
        let r = arena.pow(c, inv);
        return Some(vec![crate::transforms::eval::eval(arena, r)]);
    }
    let p = num_traits::ToPrimitive::to_i64(a.numer())?;
    let m = p.unsigned_abs();
    if m == 0 || m > MAX_LAMBERT_POWER {
        return None;
    }
    let target = if p < 0 {
        let one = arena.one;
        arena.div(one, c)
    } else {
        c
    };
    let target = crate::transforms::eval::eval(arena, target);
    if m == 1 {
        return Some(vec![target]);
    }
    let m_i64 = i64::try_from(m).ok()?;
    let inv_m = arena.rational(1, m_i64);
    let even = m % 2 == 0;
    let constant = crate::base::walk::free_symbols(arena, target).is_empty();
    let sign = if constant {
        crate::poly::algebraic::sign_checked(arena, target)
    } else {
        None
    };
    let mut out = Vec::new();
    match sign {
        Some(1) => {
            let r = positive_root(arena, target, m_i64);
            out.push(r);
            if even {
                out.push(arena.neg(r));
            }
        }
        Some(-1) if !even => {
            let minus = arena.neg(target);
            let minus = crate::transforms::eval::eval(arena, minus);
            let r = positive_root(arena, minus, m_i64);
            out.push(arena.neg(r));
        }
        Some(0) => return None,
        _ if constant => {
            let n = usize::try_from(m).ok()?;
            out.extend(
                binomial_roots(arena, target, n)
                    .into_iter()
                    .map(|s| s.value),
            );
        }
        // A symbolic `c` and an odd `m ≥ 3`: every root (the real one of a
        // real `c` is not the principal one when `c < 0`): up to 0.46 the
        // real solutions of `e^(x/2)/4 + 2a/x³ = 0` were `{6·W(∛(−8a)/6)}`,
        // not real for `a > 0`, without the two real roots `6·Wₖ(−∛a/3)`
        // (SymPy's `solve` gives `W₀` of the three cube roots).
        _ if !even => {
            let n = usize::try_from(m).ok()?;
            out.extend(
                binomial_roots(arena, target, n)
                    .into_iter()
                    .map(|s| s.value),
            );
        }
        _ => {
            let r = arena.pow(target, inv_m);
            out.push(r);
            if even {
                out.push(arena.neg(r));
            }
        }
    }
    Some(
        out.into_iter()
            .map(|r| crate::transforms::eval::eval(arena, r))
            .collect(),
    )
}

/// `t^(1/m)` for a positive constant `t`, distributed over the factors of
/// a product of positive constants (`(4·e⁻²)^(1/2) = 2·e⁻¹`, which the
/// evaluator leaves as `√(4·e⁻²)`), `exp(u)^(1/m)` as `exp(u/m)`.
fn positive_root(arena: &mut Arena, t: ExprId, m: i64) -> ExprId {
    let inv_m = arena.rational(1, m);
    let factors: Vec<ExprId> = match arena.node(t) {
        ExprNode::Mul(ch) => ch.to_vec(),
        _ => vec![t],
    };
    let mut parts: Vec<ExprId> = Vec::with_capacity(factors.len());
    for f in factors {
        if crate::poly::algebraic::sign_checked(arena, f) != Some(1) {
            let r = arena.pow(t, inv_m);
            return crate::transforms::eval::eval(arena, r);
        }
        let node = arena.node(f).clone();
        parts.push(match node {
            ExprNode::Exp(u) => {
                let v = arena.mul(&[inv_m, u]);
                arena.exp(v)
            }
            ExprNode::Pow(b, e) if crate::poly::algebraic::sign_checked(arena, b) == Some(1) => {
                let v = arena.mul(&[inv_m, e]);
                arena.pow(b, v)
            }
            _ => arena.pow(f, inv_m),
        });
    }
    let r = arena.mul(&parts);
    crate::transforms::eval::eval(arena, r)
}

/// Equations `K·L^a·e^(b·x) = c` and `K₁·L^a₁·e^(b₁·x) + K₂·L^a₂·e^(b₂·x) =
/// 0` with `L` the variable or linear in it, `a` rational and `b ≠ 0` (an
/// exponential `e^(b·x)` or a power `p^x = e^(x·ln p)`): `x² = 2ˣ`,
/// `x²·eˣ = 3`, `xⁿ = c·eˣ`, `(x + 1)·eˣ = 2`, `√x = 2ˣ/4`, `x⁻²·eˣ = 3`.
/// Terms of one rate are merged first (`x·eˣ + eˣ = (x + 1)·eˣ`).  With
/// `u = L`, `u^a·e^(b′·u) = c′` gives `u·e^((b′/a)·u) = r` for each `r` with
/// `r^a = c′` ([`power_roots`]: both `±√c′` of `x²·eˣ = c′`), and
/// `u = (a/b′)·W_k((b′/a)·r)` on the branches of [`lambert_branches`] (SymPy's
/// `_solve_lambert` and `_lambert`, `sympy/solvers/bivariate.py`, BSD, take
/// the logarithm of both sides and the `p` roots of `t^p = e^(A)` the same
/// way).  In the two-term form the root `L = 0` is a candidate as well when
/// both powers are positive (`x²·eˣ = x³`).  Every candidate is checked
/// against the equation by [`solve_impl`].
///
/// Before 0.44 these were refused: `solve(x² − 2ˣ)` (SymPy `[2, 4,
/// −2·W(ln(2)/2)/ln 2]`), `solve(x²·eˣ − 3)` (SymPy `[2·W(−√3/2),
/// 2·W(√3/2)]`), `solve(x³ − 3ˣ)`, `solve((x + 1)·eˣ − 2)`.
fn try_solve_power_exp(arena: &mut Arena, expr: ExprId, var: ExprId, cx: &SolveCx) -> Solutions {
    candidates_found(power_exp_candidates(arena, expr, var, cx))
}

/// The candidates of [`try_solve_power_exp`]; `None` when the equation
/// does not have its shape.
fn power_exp_candidates(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Option<Vec<Solution>> {
    let terms: Vec<ExprId> = match arena.node(expr) {
        ExprNode::Add(ch) => ch.to_vec(),
        _ => vec![expr],
    };
    let mut constants: Vec<ExprId> = Vec::new();
    let mut parts: Vec<PowExpTerm> = Vec::new();
    for t in terms {
        if expr_contains_var(arena, t, var) {
            parts.push(pow_exp_term(arena, t, var)?);
        } else {
            constants.push(t);
        }
    }
    let parts = merge_by_rate(arena, parts, var)?;
    let d = match constants.len() {
        0 => arena.zero,
        _ => {
            let s = arena.add(&constants);
            crate::transforms::eval::eval(arena, s)
        }
    };
    let d_zero = arena.is_zero_structural(d);
    // K·L^a·e^(b·x) = c, and the root L = 0 of the two-term form
    let shape: (ExprId, Q);
    let (k, base, a, b, c, root_of_base) = match (parts.as_slice(), d_zero) {
        ([t], false) => {
            shape = (t.rate_base, t.rate_power.clone());
            (
                t.coeff,
                t.base,
                t.power.clone(),
                t.rate,
                arena.neg(d),
                false,
            )
        }
        ([t1, t2], true) => {
            let base = match (t1.base, t2.base) {
                (Some(p), Some(q)) if p != q => return None,
                (p, q) => p.or(q),
            };
            let s1 = (t1.rate_base, t1.rate_power.clone());
            let s2 = (t2.rate_base, t2.rate_power.clone());
            shape = match (
                arena.is_zero_structural(t1.rate),
                arena.is_zero_structural(t2.rate),
            ) {
                (true, _) => s2,
                (false, true) => s1,
                (false, false) if s1 == s2 => s1,
                _ => return None,
            };
            let both_vanish = t1.power.is_positive() && t2.power.is_positive();
            let rate = arena.sub(t1.rate, t2.rate);
            let rate = crate::transforms::eval::eval(arena, rate);
            let ratio = arena.div(t2.coeff, t1.coeff);
            let c = arena.neg(ratio);
            (arena.one, base, &t1.power - &t2.power, rate, c, both_vanish)
        }
        _ => return None,
    };
    let c = crate::transforms::eval::eval(arena, c);
    let base = base?;
    if arena.is_zero_structural(b) || arena.is_zero_structural(c) {
        return None;
    }
    let (alpha, beta) = linear_rate(arena, base, var)?;
    if !shape.1.is_one() {
        let eq = PowExpMonomial {
            k,
            a: &a,
            b,
            c,
            alpha,
            beta,
            rate_base: shape.0,
            n: &shape.1,
        };
        return power_exp_monomial_solutions(arena, &eq, root_of_base, var, cx);
    }
    let root_of_base = root_of_base.then(|| {
        let neg_beta = arena.neg(beta);
        let x0 = arena.div(neg_beta, alpha);
        crate::transforms::eval::eval(arena, x0)
    });
    if a.is_zero() {
        // L^a·(K₁·e^(b₁·x) + K₂·e^(b₂·x)) = 0: e^(b·x) = c, and L = 0
        // (`x²·2ˣ − x²/10`).
        let bx = arena.mul(&[b, var]);
        let generator = arena.exp(bx);
        let mut out = peeled_candidates(solve_by_peeling(arena, generator, c, var, cx))?;
        if let Some(x0) = root_of_base
            && !out.contains(&x0)
        {
            out.push(x0);
        }
        return Some(out.into_iter().map(|value| Solution { value }).collect());
    }
    // u = L = α·x + β:  u^a·e^(b′·u) = c′,  b′ = b/α,  c′ = (c/K)·e^(b·β/α)
    let b1 = arena.div(b, alpha);
    let b1 = crate::transforms::eval::eval(arena, b1);
    let shift = arena.mul(&[b, beta]);
    let shift = arena.div(shift, alpha);
    let e_shift = arena.exp(shift);
    let c_over_k = arena.div(c, k);
    let c1 = arena.mul(&[c_over_k, e_shift]);
    let c1 = crate::transforms::eval::eval(arena, c1);
    let a_id = arena.num_ratio(a.clone());
    let b1_over_a = arena.div(b1, a_id);
    let a_over_b1 = arena.div(a_id, b1);
    let mut out: Vec<Solution> = Vec::new();
    let mut push = |arena: &mut Arena, value: ExprId| {
        let value = crate::transforms::eval::eval(arena, value);
        if !out.iter().any(|s| s.value == value) {
            out.push(Solution { value });
        }
    };
    for r in power_roots(arena, c1, &a)? {
        let arg = arena.mul(&[b1_over_a, r]);
        for w in lambert_branches(arena, arg, cx) {
            let u = arena.mul(&[a_over_b1, w]);
            let shifted = arena.sub(u, beta);
            let x = arena.div(shifted, alpha);
            push(arena, x);
        }
    }
    if let Some(x0) = root_of_base {
        push(arena, x0);
    }
    Some(out)
}

// ═══════════════════════════════════════════════════════════════════════════
// Symbolic linear solver: a*x + b = 0 where a, b may be symbolic
// ═══════════════════════════════════════════════════════════════════════════

/// Try to solve a linear equation with symbolic coefficients.
///
/// Given `expr = 0`, solve for `var` when `expr` is linear in `var` but
/// the coefficients may be symbolic (not just numbers).
/// For example: `k*x - F = 0` → `x = F/k`.
///
/// Returns `Some(solutions)` if `expr` is linear in `var`, `None` otherwise.
pub(crate) fn try_solve_linear_symbolic(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Option<Vec<Solution>> {
    // Get the Add children (or treat expr as a single-term sum).
    let terms: Vec<ExprId> = match arena.node(expr).clone() {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![expr],
    };

    let mut coeff_parts: Vec<ExprId> = Vec::new(); // coefficients of var
    let mut const_parts: Vec<ExprId> = Vec::new(); // terms without var

    for &term in &terms {
        if !expr_contains_var(arena, term, var) {
            // Term doesn't contain var — it's part of the constant.
            const_parts.push(term);
            continue;
        }

        // Term contains var — try to extract a linear coefficient.
        {
            let coeff = extract_var_coeff_in_product(arena, term, var)?;
            coeff_parts.push(coeff);
        }
    }

    if coeff_parts.is_empty() {
        return None;
    }

    // Build total coefficient: sum of all var-coefficients.
    let coeff = if coeff_parts.len() == 1 {
        coeff_parts[0]
    } else {
        arena.add(&coeff_parts)
    };

    // Build constant: sum of all non-var terms.
    let constant = if const_parts.is_empty() {
        arena.zero
    } else if const_parts.len() == 1 {
        const_parts[0]
    } else {
        arena.add(&const_parts)
    };

    // Solution: var = -constant / coeff, as a single cancelled fraction
    // when the coefficients are themselves fractions.
    let neg_const = arena.neg(constant);
    let solution = arena.div(neg_const, coeff);
    let solution = crate::simplify::ratsimp::ratsimp(arena, solution);

    Some(vec![Solution { value: solution }])
}

fn extract_var_coeff_in_product(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<ExprId> {
    if expr == var {
        return Some(arena.one);
    }
    match arena.node(expr).clone() {
        ExprNode::Mul(children) => {
            let mut const_parts: Vec<ExprId> = Vec::new();
            let mut found_var = false;
            for &child in &children {
                if child == var {
                    if found_var {
                        return None;
                    }
                    found_var = true;
                } else if !expr_contains_var(arena, child, var) {
                    const_parts.push(child);
                } else {
                    return None;
                }
            }
            if !found_var {
                return None;
            }
            match const_parts.len() {
                0 => Some(arena.one),
                1 => Some(const_parts[0]),
                _ => Some(arena.mul(&const_parts)),
            }
        }
        _ => None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Declared domains of the unknown
// ═══════════════════════════════════════════════════════════════════════════
//
// The public solvers answer over the declared domain of the unknown, as
// SymPy's `solve` does with `check_assumptions`: a candidate that certainly
// violates an assumption of the variable is dropped (`x` positive:
// `x² − 4 = 0` gives `[2]`, `x + 3 = 0` and `x = 0` give `[]`; `m` integer:
// `2m − 3 = 0` gives `[]`), one whose membership is not decided is kept
// (SymPy keeps those too: `x − a` for a plain `a`).  Up to 0.42 the
// assumptions of the unknown were ignored.  The internal callers of
// [`known_real_roots`] (partial fractions, singularities, residues, …) want every
// root and are not filtered.

/// The properties of a value compared with the declared assumptions of the
/// unknown ([`assumptions_hold`]).  Not `commutative`, `complex`, `finite`,
/// `hermitian`, …: every candidate is a finite complex number.
const DOMAIN_PROPS: [crate::base::assumptions::Props; 17] = {
    use crate::base::assumptions::Props;
    [
        Props::REAL,
        Props::RATIONAL,
        Props::INTEGER,
        Props::ALGEBRAIC,
        Props::TRANSCENDENTAL,
        Props::IRRATIONAL,
        Props::IMAGINARY,
        Props::POSITIVE,
        Props::NEGATIVE,
        Props::NONNEGATIVE,
        Props::NONPOSITIVE,
        Props::ZERO,
        Props::NONZERO,
        Props::EVEN,
        Props::ODD,
        Props::PRIME,
        Props::COMPOSITE,
    ]
};

/// The declared assumptions of `var`, closed under the implication rules,
/// when they restrict its values (`None` for a plain symbol, one declared
/// only `complex`, `finite`, …, and for a non-symbol).
pub(crate) fn declared_domain(
    arena: &Arena,
    var: ExprId,
) -> Option<crate::base::assumptions::Assumptions> {
    let ExprNode::Symbol(sid) = *arena.node(var) else {
        return None;
    };
    let mut declared = arena.symbol_assumptions(sid);
    declared.forward_chain();
    DOMAIN_PROPS
        .iter()
        .any(|&p| declared.query(p).is_some())
        .then_some(declared)
}

/// The signs of the real and imaginary parts of a constant from a
/// certified evaluation: one atom each ([`SignSet::ZERO`]: the part is
/// exactly 0), or [`SignSet::REAL`] for both parts when the evaluation is
/// not certified (a part is a real number of unknown sign).  Up to 0.49 two
/// `Option<Ordering>`s, read by `== Some(Less)` and `matches!(…, Some(Less |
/// Greater))` at each use.
///
/// [`SignSet::ZERO`]: crate::base::truth::SignSet::ZERO
/// [`SignSet::REAL`]: crate::base::truth::SignSet::REAL
#[derive(Clone, Copy)]
struct PartSigns {
    re: crate::base::truth::SignSet,
    im: crate::base::truth::SignSet,
}

impl PartSigns {
    /// Certainly not real: a certified nonzero imaginary part.
    fn nonreal(self) -> bool {
        self.im.is_nonzero().is_true()
    }

    /// Certainly not 0: a certified nonzero part.
    fn nonzero(self) -> bool {
        self.nonreal() || self.re.is_nonzero().is_true()
    }

    /// Certainly 0: both parts certified 0.
    fn zero(self) -> bool {
        self.re.is_zero().is_true() && self.im.is_zero().is_true()
    }

    /// Are the signs certified?
    fn certified(self) -> bool {
        self.re.is_single() && self.im.is_single()
    }
}

/// Strict sign of a `BigFloat`: a zero of either sign is `Equal`.
fn bigfloat_sign(x: &astro_float::BigFloat) -> std::cmp::Ordering {
    if x.is_zero() {
        std::cmp::Ordering::Equal
    } else if x.is_negative() {
        std::cmp::Ordering::Less
    } else {
        std::cmp::Ordering::Greater
    }
}

fn part_signs(arena: &Arena, value: ExprId) -> PartSigns {
    use crate::base::truth::SignSet;
    use crate::transforms::evalf::{Settled, ZeroSearch};
    match crate::transforms::evalf::evalf_settled(arena, value, ROOT_CHECK_DIGITS, ZeroSearch::Cap)
    {
        Ok((z, Settled::Certified)) => PartSigns {
            re: SignSet::from_ordering(bigfloat_sign(&z.0)),
            im: SignSet::from_ordering(bigfloat_sign(&z.1)),
        },
        _ => PartSigns {
            re: SignSet::REAL,
            im: SignSet::REAL,
        },
    }
}

/// Is the constant `value` certainly not an integer: not real, or at a
/// certified nonzero distance below 1 from the integer `k` nearest to a
/// 16-digit approximation (so strictly between `k − 1` and `k + 1`, and
/// not `k`)?
fn certainly_not_integer(arena: &mut Arena, value: ExprId, signs: PartSigns) -> bool {
    if signs.nonreal() {
        return true;
    }
    let Ok(z) = crate::transforms::evalf::evalf_complex64(arena, value) else {
        return false;
    };
    if !z.re.is_finite() || z.re.abs() > 1e15 {
        return false;
    }
    let k = arena.int(z.re.round() as i64);
    let d = arena.sub(value, k);
    let d = crate::transforms::eval::eval(arena, d);
    if let Some(r) = arena.as_num(d) {
        return !r.is_integer();
    }
    if !part_signs(arena, d).re.is_nonzero().is_true() {
        return false;
    }
    crate::transforms::evalf::evalf_complex64(arena, d).is_ok_and(|w| w.re.abs() < 0.99)
}

/// The rational `q` with `e = q·π`, for a constant `e`.
fn rational_multiple_of_pi(arena: &mut Arena, e: ExprId) -> Option<Q> {
    if arena.is_zero_structural(e) {
        return Some(Q::zero());
    }
    let pi = arena.pi;
    let q = arena.div(e, pi);
    let q = crate::transforms::eval::eval(arena, q);
    arena.as_num(q).cloned()
}

/// Is the constant `value` certainly transcendental: a nonzero rational
/// multiple of `π`, `e^q`, `q·ln r` or `W(q)` for nonzero rationals `q`
/// and a positive rational `r ≠ 1` (Lindemann–Weierstrass: `e^a` is
/// transcendental for algebraic `a ≠ 0`, and `W(q) = a` would make
/// `e^a = q/a` algebraic)?
fn certainly_transcendental(arena: &mut Arena, value: ExprId) -> bool {
    if rational_multiple_of_pi(arena, value).is_some_and(|q| !q.is_zero()) {
        return true;
    }
    let nonzero_rational = |arena: &Arena, e: ExprId| arena.as_num(e).is_some_and(|q| !q.is_zero());
    let log_of_rational = |arena: &Arena, e: ExprId| match *arena.node(e) {
        ExprNode::Ln(r) => arena
            .as_num(r)
            .is_some_and(|r| r.is_positive() && !r.is_one()),
        _ => false,
    };
    match arena.node(value) {
        ExprNode::Exp(q) | ExprNode::LambertW(q) => nonzero_rational(arena, *q),
        // the other branches `W_k(q)`
        ExprNode::Lib(crate::base::libfn::LibFn::LambertW, args) if args.len() == 2 => {
            nonzero_rational(arena, args[0])
        }
        ExprNode::Ln(_) => log_of_rational(arena, value),
        ExprNode::Mul(ch) if ch.len() == 2 => {
            nonzero_rational(arena, ch[0]) && log_of_rational(arena, ch[1])
        }
        _ => false,
    }
}

/// Is `e^{is}` (so `sin s`, `cos s`) algebraic: `s` a sum of rational
/// multiples of `π`, `asin q`, `acos q`, `atan q` for algebraic `q`
/// (`e^{i·asin q} = √(1 − q²) + iq`, `e^{i·atan q} = (1 + iq)/√(1 + q²)`,
/// and a rational power of an algebraic number is algebraic)?
fn exp_i_is_algebraic(arena: &mut Arena, s: ExprId) -> bool {
    let terms: Vec<ExprId> = match arena.node(s) {
        ExprNode::Add(ch) => ch.to_vec(),
        _ => vec![s],
    };
    terms.into_iter().all(|t| {
        let atom = match arena.node(t) {
            ExprNode::Mul(ch) if ch.len() == 2 && arena.as_num(ch[0]).is_some() => ch[1],
            ExprNode::Neg(a) => *a,
            _ => t,
        };
        match *arena.node(atom) {
            ExprNode::Pi => true,
            ExprNode::Num(_) => arena.is_zero_structural(atom),
            ExprNode::Asin(q) | ExprNode::Acos(q) | ExprNode::Atan(q) => {
                arena.as_num(q).is_some()
                    || (crate::base::walk::free_symbols(arena, q).is_empty()
                        && crate::base::walk::post_order_ids(arena, q).len() <= MAX_MINPOLY_NODES
                        && crate::poly::algebraic::minimal_polynomial(arena, q).is_some())
            }
            _ => false,
        }
    })
}

/// Largest constant (in nodes) whose minimal polynomial
/// [`certainly_irrational`] looks for.
const MAX_MINPOLY_NODES: usize = 40;

/// Is the real constant `value` certainly irrational: a nonzero rational
/// multiple of `π`, or an algebraic number whose (verified) minimal
/// polynomial has degree above 1 (`√3`, `(1 + √5)/2`)?
fn certainly_irrational(arena: &mut Arena, value: ExprId) -> bool {
    if certainly_transcendental(arena, value) {
        return true;
    }
    // a root of an irreducible polynomial over ℚ of degree above 1
    if let ExprNode::RootOf(p, x, _) = *arena.node(value) {
        return polybridge::expr_to_poly(arena, p, x).is_some_and(|q| {
            q.degree().is_some_and(|d| d > 1)
                && crate::poly::factor_zassenhaus::is_irreducible_z(&q) == Some(true)
        });
    }
    if crate::base::walk::post_order_ids(arena, value).len() > MAX_MINPOLY_NODES {
        return false;
    }
    crate::poly::algebraic::minimal_polynomial(arena, value)
        .and_then(|p| p.degree())
        .is_some_and(|d| d > 1)
}

/// Does a certified evaluation show that the constant `value` lacks the
/// property `prop` (declared true of the unknown)?
fn violates_numerically(
    arena: &mut Arena,
    value: ExprId,
    prop: crate::base::assumptions::Props,
    s: PartSigns,
) -> bool {
    use crate::base::assumptions::Props;
    let zero = s.zero();
    let re_negative = s.re.is_negative().is_true();
    let re_positive = s.re.is_positive().is_true();
    let re_nonzero = s.re.is_nonzero().is_true();
    match prop {
        Props::REAL | Props::IRRATIONAL => s.nonreal(),
        Props::RATIONAL => s.nonreal() || certainly_irrational(arena, value),
        Props::ALGEBRAIC => certainly_transcendental(arena, value),
        Props::POSITIVE => s.nonreal() || re_negative || zero,
        Props::NEGATIVE => s.nonreal() || re_positive || zero,
        Props::NONNEGATIVE => s.nonreal() || re_negative,
        Props::NONPOSITIVE => s.nonreal() || re_positive,
        Props::ZERO => s.nonreal() || re_nonzero,
        Props::IMAGINARY => re_nonzero,
        // `1` is neither prime nor composite (the assumption system leaves
        // it open)
        Props::PRIME | Props::COMPOSITE
            if arena
                .as_num(value)
                .is_some_and(|r| r.is_integer() && *r <= Q::one()) =>
        {
            true
        }
        Props::INTEGER | Props::EVEN | Props::ODD | Props::PRIME | Props::COMPOSITE => {
            certainly_not_integer(arena, value, s)
        }
        _ => false,
    }
}

/// `id`, or `0` when it is a constant that is exactly 0 without being
/// written 0 (`atan(1/2) + atan(1/3) − π/4`, `cos(π/16) − √(2 + √(2 +
/// √2))/2`: the zero tests of
/// [`constant_zero_decision`](crate::transforms::eval::constant_zero_decision)).
/// Only a value whose `f64` evaluation is within `10⁻¹⁰` of 0 is tested.
pub(crate) fn exact_zero_form(arena: &mut Arena, id: ExprId) -> ExprId {
    let near_zero = crate::base::walk::free_symbols(arena, id).is_empty()
        && !arena.is_zero_structural(id)
        && crate::transforms::evalf::evalf_complex64(arena, id).is_ok_and(|z| z.norm() < 1e-10);
    if near_zero && crate::transforms::eval::constant_zero_decision(arena, id) == Some(true) {
        arena.zero
    } else {
        id
    }
}

/// Does `value` satisfy every assumption declared on the symbol `var`?
/// `Some(false)` when it certainly violates one: by the assumption system
/// (`−a` is not positive for a positive `a`, `3/2` is not an integer), or
/// for a constant by a certified evaluation (`−√2` is not positive, `√2`
/// not an integer, `asin 2` not real); `None` when it is not decided.
/// `Some(true)` for a non-symbol or a symbol without restrictions.
pub(crate) fn assumptions_hold(arena: &mut Arena, var: ExprId, value: ExprId) -> Option<bool> {
    match undecided_assumptions(arena, var, value) {
        None => Some(false),
        Some(open) if open.is_empty() => Some(true),
        Some(_) => None,
    }
}

/// The declared properties of `var` that `value` is not known to have
/// ([`assumptions_hold`]): empty when it has them all, `None` when it
/// certainly lacks one.
fn undecided_assumptions(
    arena: &mut Arena,
    var: ExprId,
    value: ExprId,
) -> Option<crate::base::assumptions::Props> {
    use crate::base::assumptions::AssumptionCache;
    let Some(declared) = declared_domain(arena, var) else {
        return Some(crate::base::assumptions::Props::empty());
    };
    let value = decision_form(arena, value);
    use crate::base::assumptions::Props;
    let mut cache = AssumptionCache::new();
    let constant = crate::base::walk::free_symbols(arena, value).is_empty()
        && !crate::base::walk::has_unevaluated(arena, value);
    // The constant whose signs decide those of a parametric `value`
    // ([`phase_constant`]), computed when first needed.
    let mut phase: Option<Option<Phase>> = None;
    let mut facts: Vec<(ExprId, PartSigns, Option<bool>)> = Vec::new();
    let mut undecided = Props::empty();
    for prop in DOMAIN_PROPS {
        let Some(want) = declared.query(prop) else {
            continue;
        };
        match cache.query(arena, value, prop) {
            Some(got) if got == want => continue,
            Some(_) => return None,
            None => {}
        }
        let sign_kind = matches!(
            prop,
            Props::REAL
                | Props::POSITIVE
                | Props::NEGATIVE
                | Props::NONNEGATIVE
                | Props::NONPOSITIVE
                | Props::ZERO
                | Props::NONZERO
                | Props::IMAGINARY
        );
        if !constant && want && implies_real(prop) && nonreal_logarithm(arena, value) {
            return None;
        }
        let subject = if constant {
            Some(value)
        } else if sign_kind {
            let ph = phase.get_or_insert_with(|| phase_of(arena, value)).clone();
            match ph {
                Some(p) if p.spread.is_zero() && !p.maybe_zero => Some(p.w),
                Some(p) => {
                    if want && arc_excludes(arena, &p, prop) {
                        return None;
                    }
                    None
                }
                None => None,
            }
        } else {
            None
        };
        if let Some(w) = subject {
            let (s, r) = match facts.iter().find(|f| f.0 == w) {
                Some(&(_, s, r)) => (s, r),
                None => {
                    let s = part_signs(arena, w);
                    let mut reals = AssumptionCache::new();
                    let r = crate::transforms::realness::constant_realness_as_declared(
                        arena,
                        w,
                        ROOT_CHECK_DIGITS,
                        &mut reals,
                    );
                    facts.push((w, s, r));
                    (s, r)
                }
            };
            if want && violates_numerically(arena, w, prop, s) {
                return None;
            }
            if holds_numerically(prop, want, s, r) {
                continue;
            }
        }
        undecided |= prop;
    }
    Some(undecided)
}

/// `value` in a form the decisions read better: `exp(q·ln m)` (rational
/// `q`, positive rational `m`) is `m^q` (`ln(x)/ln 10 = 2` gives
/// `exp(2·ln 10)`, which is `100`, odd for nobody).
fn decision_form(arena: &mut Arena, value: ExprId) -> ExprId {
    let ExprNode::Exp(arg) = *arena.node(value) else {
        return value;
    };
    let (q, ln) = match arena.node(arg) {
        ExprNode::Mul(ch) if ch.len() == 2 => match arena.as_num(ch[0]) {
            Some(q) => (q.clone(), ch[1]),
            None => return value,
        },
        ExprNode::Ln(_) => (Q::one(), arg),
        _ => return value,
    };
    let ExprNode::Ln(m) = *arena.node(ln) else {
        return value;
    };
    if !arena.as_num(m).is_some_and(|m| m.is_positive()) {
        return value;
    }
    let q = arena.num_ratio(q);
    let p = arena.pow(m, q);
    crate::transforms::eval::eval(arena, p)
}

/// Does the property imply that the value is real?
fn implies_real(prop: crate::base::assumptions::Props) -> bool {
    use crate::base::assumptions::Props;
    matches!(
        prop,
        Props::REAL
            | Props::POSITIVE
            | Props::NEGATIVE
            | Props::NONNEGATIVE
            | Props::NONPOSITIVE
            | Props::RATIONAL
            | Props::IRRATIONAL
            | Props::INTEGER
            | Props::EVEN
            | Props::ODD
            | Props::PRIME
            | Props::COMPOSITE
    )
}

/// Is `value` `q·ln B` (`q` a nonzero rational) with `B` negative, whose
/// imaginary part is `q·π`?  (`exp(x) = a` for a negative `a` gives
/// `ln a`, not real; SymPy 1.14 `log(a).is_real` is `False`.)
fn nonreal_logarithm(arena: &Arena, value: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    let ln = match arena.node(value) {
        ExprNode::Mul(ch) if ch.len() == 2 => match (arena.as_num(ch[0]), arena.node(ch[1])) {
            (Some(q), ExprNode::Ln(_)) if !q.is_zero() => ch[1],
            _ => return false,
        },
        ExprNode::Neg(inner) => *inner,
        _ => value,
    };
    let ExprNode::Ln(b) = *arena.node(ln) else {
        return false;
    };
    AssumptionCache::new().query(arena, b, Props::NEGATIVE) == Some(true)
}

/// The argument of a parametric value `|…|·w·e^{iθ}` ([`phase_of`]):
/// `θ ∈ (−spread·π, spread·π]`, and the value may be `0` (`maybe_zero`).
#[derive(Clone)]
struct Phase {
    w: ExprId,
    spread: Q,
    maybe_zero: bool,
}

/// For a `value` with parameters that is `c·∏ Bᵢ^rᵢ` (rational `rᵢ`, `c` a
/// constant, also as a sum of such terms with the same parametric part),
/// its argument from the principal branch: a positive `B` contributes `0`,
/// a negative one `r·π` (`B^r = |B|^r·e^{iπr}`), a nonnegative or
/// nonpositive one the same or the value `0`, and one of unknown sign with
/// `0 < r < 1` the range `(−rπ, rπ]` (or `0`).  With only strict signs
/// `w = c·e^{iπΣrᵢ[Bᵢ < 0]}` has the sign and realness of `value`: for a
/// negative `a`, `∛a` is not real (`w = e^{iπ/3}`; SymPy 1.14 also says
/// `False`), `(−1/2 + i√3/2)·∛a` is negative (`w = −1`); with a range,
/// [`arc_excludes`] decides what the value cannot be (`√(1 − b)` is never
/// negative, `−√(1 − b)` never positive).
fn phase_of(arena: &mut Arena, value: ExprId) -> Option<Phase> {
    use crate::base::assumptions::{AssumptionCache, Props};
    let first = match arena.node(value) {
        ExprNode::Add(ch) => *ch.first()?,
        _ => value,
    };
    let factors: Vec<ExprId> = match arena.node(first) {
        ExprNode::Mul(ch) => ch.to_vec(),
        _ => vec![first],
    };
    let parametric: Vec<ExprId> = factors
        .into_iter()
        .filter(|&f| !crate::base::walk::free_symbols(arena, f).is_empty())
        .collect();
    if parametric.is_empty() {
        return None;
    }
    let mut cache = AssumptionCache::new();
    let mut turns = Q::zero();
    let mut spread = Q::zero();
    let mut maybe_zero = false;
    for &f in &parametric {
        let (base, r) = match *arena.node(f) {
            ExprNode::Pow(b, e) => (b, arena.as_num(e)?.clone()),
            _ => (f, Q::one()),
        };
        let known =
            |cache: &mut AssumptionCache, p: Props| cache.query(arena, base, p) == Some(true);
        if known(&mut cache, Props::NEGATIVE) {
            turns += r;
        } else if known(&mut cache, Props::POSITIVE) {
            // argument 0
        } else if known(&mut cache, Props::NONPOSITIVE) {
            // `0^r` is `0` for `r > 0`, undefined (no value) for `r < 0`
            turns += r.clone();
            maybe_zero |= r.is_positive();
        } else if known(&mut cache, Props::NONNEGATIVE) {
            maybe_zero |= r.is_positive();
        } else if r.is_positive() && r < Q::one() {
            spread += r;
            maybe_zero |= !known(&mut cache, Props::NONZERO);
        } else {
            return None;
        }
    }
    if spread >= Q::one() {
        return None;
    }
    let m = arena.mul(&parametric);
    let c = arena.div(value, m);
    let c = crate::transforms::expand::expand(arena, c);
    let c = crate::transforms::eval::eval(arena, c);
    if !crate::base::walk::free_symbols(arena, c).is_empty() {
        return None;
    }
    let turns = arena.num_ratio(turns);
    let angle = arena.mul(&[turns, arena.pi, arena.i_unit]);
    let rotation = arena.exp(angle);
    let w = arena.mul(&[c, rotation]);
    let w = crate::transforms::eval::eval(arena, w);
    Some(Phase {
        w,
        spread,
        maybe_zero,
    })
}

/// Does the argument range of `p` show that the value cannot have the
/// property `prop` (declared true of the unknown)?  The range is
/// `arg w + (−spread·π, spread·π]`; an angle within `10⁻⁹` of an end is
/// taken to be in it.
fn arc_excludes(arena: &mut Arena, p: &Phase, prop: crate::base::assumptions::Props) -> bool {
    use crate::base::assumptions::Props;
    use std::f64::consts::PI;
    if !part_signs(arena, p.w).certified() {
        return false;
    }
    let Ok(w) = crate::transforms::evalf::evalf_complex64(arena, p.w) else {
        return false;
    };
    let r = w.norm();
    if r.is_nan() || r <= 0.0 || r.is_infinite() {
        return false;
    }
    let phi = w.arg();
    let half = num_traits::ToPrimitive::to_f64(&p.spread).unwrap_or(1.0) * PI;
    // is the angle `t` (mod 2π) possibly in `(phi − half, phi + half]`?
    let reaches = |t: f64| {
        let mut d = (t - phi) % (2.0 * PI);
        if d > PI {
            d -= 2.0 * PI;
        } else if d <= -PI {
            d += 2.0 * PI;
        }
        d.abs() <= half + 1e-9
    };
    let zero_ok = p.maybe_zero;
    match prop {
        Props::POSITIVE => !reaches(0.0),
        Props::NEGATIVE => !reaches(PI),
        Props::REAL => !zero_ok && !reaches(0.0) && !reaches(PI),
        Props::NONNEGATIVE => !zero_ok && !reaches(0.0),
        Props::NONPOSITIVE => !zero_ok && !reaches(PI),
        Props::IMAGINARY => !reaches(PI / 2.0) && !reaches(-PI / 2.0),
        _ => false,
    }
}

/// Does a certified evaluation (signs `s`, realness `real`) show that a
/// constant has the property `prop` (`want`) of the sign and realness
/// kind?  `false` when it does not decide.
fn holds_numerically(
    prop: crate::base::assumptions::Props,
    want: bool,
    s: PartSigns,
    real: Option<bool>,
) -> bool {
    use crate::base::assumptions::Props;
    let is_real = real == Some(true);
    let not_real = real == Some(false) || s.nonreal();
    let pos = is_real && s.re.is_positive().is_true();
    let neg = is_real && s.re.is_negative().is_true();
    let zero = s.zero();
    let re_nonzero = s.re.is_nonzero().is_true();
    let nonzero = s.nonzero();
    match (prop, want) {
        (Props::REAL, true) => is_real,
        (Props::REAL, false) => not_real,
        (Props::POSITIVE, true) => pos,
        (Props::POSITIVE, false) => neg || zero || not_real,
        (Props::NEGATIVE, true) => neg,
        (Props::NEGATIVE, false) => pos || zero || not_real,
        (Props::NONNEGATIVE, true) => pos || zero,
        (Props::NONNEGATIVE, false) => neg || not_real,
        (Props::NONPOSITIVE, true) => neg || zero,
        (Props::NONPOSITIVE, false) => pos || not_real,
        (Props::NONZERO, true) | (Props::ZERO, false) => nonzero,
        (Props::IMAGINARY, false) => is_real || re_nonzero,
        _ => false,
    }
}

/// The candidates in `values` that do not certainly violate the declared
/// assumptions of `var` ([`assumptions_hold`]): an undecided one with
/// parameters is `Err` when its membership may fail on a set of parameter
/// values of positive measure ([`in_domain_only_for_some_parameters`]),
/// and kept otherwise; an undecided constant is `Err` with the reason —
/// one that is 0 to the precision of the zero search
/// ([`zero_to_precision`]), or whose membership no exact or certified test
/// decides.  Up to 0.50 the latter was kept: a real `x` got the roots
/// `ln(⅕·4^(1/4)·(±√(8√2 − 4·4^(3/4)) − 2·2^(3/4)))/ln 2` of `4^(x + 3/4) +
/// 2·2^(x + 3/4) + 1 = 0`, logarithms of negative numbers whose value no
/// evaluation could settle (the radicand is 0); a root of a real unknown
/// must be proved real.
pub(crate) fn retain_admissible(
    arena: &mut Arena,
    var: ExprId,
    values: Vec<ExprId>,
) -> Result<Vec<ExprId>, String> {
    let Some(declared) = declared_domain(arena, var) else {
        return Ok(values);
    };
    let real_unknown =
        crate::base::truth::Truth::from(declared.query(crate::base::assumptions::Props::REAL))
            .is_true();
    let mut kept = Vec::with_capacity(values.len());
    for v in values {
        match assumptions_hold(arena, var, v) {
            Some(false) => {}
            Some(true) => kept.push(v),
            None => {
                if zero_to_precision(arena, v) {
                    let shown = arena.display(v).to_string();
                    let name = arena.display(var).to_string();
                    return Err(format!(
                        "cannot decide whether the solution {shown} lies in the declared domain \
                         of {name}: it is 0 to {ROOT_CHECK_DIGITS} digits, and not known to be 0"
                    ));
                }
                match constant_realness(arena, v) {
                    Some(crate::base::truth::Truth::False) if real_unknown => continue,
                    Some(crate::base::truth::Truth::Unknown) if real_unknown => {
                        let shown = arena.display(v).to_string();
                        let name = arena.display(var).to_string();
                        return Err(format!(
                            "cannot decide whether the solution {shown} is real (the declared \
                             domain of {name})"
                        ));
                    }
                    _ => {}
                }
                if in_domain_only_for_some_parameters(arena, var, v, &declared) {
                    let shown = arena.display(v).to_string();
                    let name = arena.display(var).to_string();
                    return Err(format!(
                        "the solution {shown} lies in the declared domain of {name} only for some \
                         values of the parameters, which their assumptions do not decide \
                         (solve_as_set gives the solution set)"
                    ));
                }
                kept.push(v);
            }
        }
    }
    Ok(kept)
}

/// The exact condition on its symbols under which the value `value` is
/// real, when it has the shape `A + C·u^(p/q)` with `A`, `C` and `u` real by
/// the assumptions of the symbols, `C` nonzero and `q > 1`: `u ≥ 0` (`u >
/// 0` for `p < 0`, where `u = 0` is a pole) — for `u < 0` the principal
/// power `|u|^(p/q)·e^(iπp/q)` is not real, since `p/q` is not an integer.
/// A radicand that is itself of that shape is decided the same way when
/// `|p| < q` (then `u^(p/q)` is real only for a real `u ≥ 0`): `√(a −
/// √b)` is real exactly when `b ≥ 0 ∧ a − √b ≥ 0`.  `None` for other
/// shapes (two radicals, a coefficient that may be 0).
pub(crate) fn realness_condition(arena: &mut Arena, value: ExprId) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    use crate::base::truth::Truth;
    const MAX_NESTING: usize = 4;
    let mut cache = AssumptionCache::new();
    let mut known =
        |arena: &Arena, e: ExprId, p: Props| Truth::from(cache.query(arena, e, p)).is_true();
    let mut conds: Vec<ExprId> = Vec::new();
    let mut current = value;
    for _ in 0..MAX_NESTING {
        if known(arena, current, Props::REAL) {
            return Some(arena.and(&conds));
        }
        // the one term not known real, `C·u^(p/q)`
        let terms: Vec<ExprId> = match arena.node(current) {
            ExprNode::Add(ch) => ch.to_vec(),
            _ => vec![current],
        };
        let open: Vec<ExprId> = terms
            .into_iter()
            .filter(|&t| !known(arena, t, Props::REAL))
            .collect();
        let [term] = open[..] else {
            return None;
        };
        let mut factors: Vec<ExprId> = Vec::new();
        let mut stack = vec![term];
        while let Some(f) = stack.pop() {
            match arena.node(f) {
                ExprNode::Mul(ch) => stack.extend(ch.iter().copied()),
                ExprNode::Neg(inner) => stack.push(*inner),
                _ => factors.push(f),
            }
        }
        let mut radical: Option<(ExprId, Q)> = None;
        for f in factors {
            if known(arena, f, Props::REAL) && known(arena, f, Props::NONZERO) {
                continue;
            }
            let ExprNode::Pow(u, e) = *arena.node(f) else {
                return None;
            };
            let r = arena.as_num(e)?.clone();
            if r.is_integer() || radical.is_some() {
                return None;
            }
            radical = Some((u, r));
        }
        let (u, r) = radical?;
        let zero = arena.zero;
        conds.push(if r.is_positive() {
            arena.ge(u, zero)
        } else {
            arena.gt(u, zero)
        });
        if known(arena, u, Props::REAL) {
            return Some(arena.and(&conds));
        }
        if r.numer().abs() >= *r.denom() {
            return None;
        }
        current = u;
    }
    None
}

/// Is the root `value`, with parameters, in the declared domain of `var`
/// only for some parameter values, so that a list holding it would be
/// wrong on a set of positive measure?  Undecided realness, sign,
/// integrality or rationality fail there (`x² = y` for a real `x` and a
/// real `y`: `±√y` is real only for `y ≥ 0`); an undecided `≠ 0`,
/// irrationality or "not an integer" fails on a set of measure zero only
/// (the generic convention: `x − a` for a nonzero `x` gives `a`).  Up to
/// 0.52 every root with parameters whose membership was undecided was
/// returned (the radical hunt of 0.51: 829 of 16,000 solves).
fn in_domain_only_for_some_parameters(
    arena: &mut Arena,
    var: ExprId,
    value: ExprId,
    declared: &crate::base::assumptions::Assumptions,
) -> bool {
    use crate::base::assumptions::Props;
    if crate::base::walk::free_symbols(arena, value).is_empty() {
        return false;
    }
    let Some(open) = undecided_assumptions(arena, var, value) else {
        return false;
    };
    open.iter().any(|p| {
        let generic = if crate::base::truth::Truth::from(declared.query(p)).is_true() {
            Props::NONZERO | Props::IRRATIONAL | Props::TRANSCENDENTAL
        } else {
            Props::ZERO
                | Props::INTEGER
                | Props::RATIONAL
                | Props::ALGEBRAIC
                | Props::EVEN
                | Props::ODD
                | Props::PRIME
                | Props::COMPOSITE
        };
        !generic.contains(p)
    })
}

/// Is the constant `value` real: by the assumption system or a certified
/// evaluation ([`constant_realness_as_declared`]), else by the facts `eval`
/// decides from its value ([`constant_facts`], with the exact zero tests:
/// `∛(1 − √(8√2 − 4·4^(3/4)))` is `1`)?  `None` for a value with symbols
/// or an unevaluated part (not decided here).
///
/// [`constant_realness_as_declared`]: crate::transforms::realness::constant_realness_as_declared
/// [`constant_facts`]: crate::transforms::eval::constant_facts
fn constant_realness(arena: &mut Arena, value: ExprId) -> Option<crate::base::truth::Truth> {
    use crate::base::truth::Truth;
    use crate::base::walk;
    if !walk::free_symbols(arena, value).is_empty() || walk::has_unevaluated(arena, value) {
        return None;
    }
    let mut reals = crate::base::assumptions::AssumptionCache::new();
    let by_value = Truth::from(crate::transforms::realness::constant_realness_as_declared(
        arena,
        value,
        ROOT_CHECK_DIGITS,
        &mut reals,
    ));
    Some(match by_value {
        Truth::Unknown => Truth::from(
            crate::transforms::eval::constant_facts(arena, value)
                .query(crate::base::assumptions::Props::REAL),
        ),
        decided => decided,
    })
}

/// Is the constant `value` zero to the precision of the zero search
/// without being certified zero or nonzero?  Such a member of a solution
/// list is undecided against a domain bounded at `0` (positive, negative,
/// nonzero, …) however many digits are computed: before 0.45 it was
/// kept, a root that may be `0` returned for a positive unknown.
fn zero_to_precision(arena: &mut Arena, value: ExprId) -> bool {
    use crate::base::walk;
    use crate::transforms::evalf::{Settled, ZeroSearch};
    if !walk::free_symbols(arena, value).is_empty() || walk::has_unevaluated(arena, value) {
        return false;
    }
    matches!(
        crate::transforms::evalf::evalf_settled(arena, value, ROOT_CHECK_DIGITS, ZeroSearch::Cap),
        Ok((_, Settled::ZeroToPrecision))
    )
}

/// The solutions `values` of `expr = 0` (from `solve`) restricted to the
/// declared domain of `var` ([`retain_admissible`]).  `Err` with the reason
/// when every one of them is outside the domain while a member of a
/// periodic family through them is in it (`x` negative: `sin x = 1/2`
/// has the principal solutions `π/6, 5π/6`, and `−7π/6`): the empty list
/// would claim there is no solution, as SymPy's `solve` does.
pub(crate) fn restrict_to_domain(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    values: Vec<ExprId>,
) -> Result<Vec<ExprId>, String> {
    if values.is_empty() || declared_domain(arena, var).is_none() {
        return Ok(values);
    }
    let kept = retain_admissible(arena, var, values)?;
    if !kept.is_empty() || polybridge::expr_to_poly(arena, expr, var).is_some() {
        return Ok(kept);
    }
    let Some(PeriodicCandidates {
        families, param: n, ..
    }) = raw_families(arena, expr, var, &SolveCx::principal(Domain::Real), true)
    else {
        return Ok(kept);
    };
    for fam in families {
        for k in [1i64, -1, 2, -2, 3, -3, 4, -4] {
            let kk = arena.int(k);
            let member = crate::transforms::subs::subs(arena, fam, n, kk);
            let member = crate::transforms::eval::eval(arena, member);
            if assumptions_hold(arena, var, member) == Some(true)
                && member_is_root(arena, expr, var, member)
            {
                let shown = arena.display(member).to_string();
                let name = arena.display(var).to_string();
                return Err(format!(
                    "every principal-branch solution is outside the declared domain of {name}, \
                     but other branches lie in it (e.g. {shown}); see solve_general"
                ));
            }
        }
    }
    Ok(kept)
}

/// Has `expr = 0` no solution at all in the declared domain of `var`?  For
/// `Ex::solve` when [`restrict_to_domain`] dropped every principal root:
/// decided by the general solution over that domain
/// ([`solve_general_declared`]), complete, each family without a member in
/// the domain ([`family_in_domain`]) and each other root certainly outside
/// it.  For a real `x`, `eˣ + e^(ix) = 0` has the principal root
/// `iπ/(1 − i)` and the solutions `(2n + 1)πi/(1 − i)`, none real; up to
/// 0.49 `Ex::solve` answered `Ok([])`, which also means "no principal
/// root in the domain, other branches not looked at".  `false` for a
/// polynomial (its empty list of roots in the domain is an answer) and
/// when not decided.
pub(crate) fn none_in_domain(arena: &mut Arena, expr: ExprId, var: ExprId) -> bool {
    if declared_domain(arena, var).is_none() || polybridge::expr_to_poly(arena, expr, var).is_some()
    {
        return false;
    }
    if let Some(coeffs) = symbolic_poly_coeffs(arena, expr, var)
        && coeffs.iter().all(|&c| !expr_contains_var(arena, c, var))
    {
        return false;
    }
    let n = branch_parameter(arena, &[expr, var]);
    let roots = match solve_general_declared(arena, expr, var, n) {
        Solutions::Complete(RootSet::Empty(_)) => return true,
        complete @ Solutions::Complete(RootSet::Points(_) | RootSet::Families { .. }) => {
            complete.claimed_points()
        }
        _ => return false,
    };
    // (the parameter of the rays of `family_in_domain`, never returned)
    let k = branch_parameter(arena, &[expr, var, n]);
    for r in roots {
        let outside = if crate::base::walk::contains(arena, r, n) {
            matches!(family_in_domain(arena, var, r, n, k), FamilyInDomain::Empty)
        } else {
            assumptions_hold(arena, var, r) == Some(false)
        };
        if !outside {
            return false;
        }
    }
    true
}

/// The members of a periodic family in the declared domain of the unknown
/// ([`family_in_domain`]).
pub(crate) enum FamilyInDomain {
    /// Every member.
    All,
    /// No member.
    Empty,
    /// Exactly one member.
    Point(ExprId),
    /// The members of the rays `family(n₀ + k)` or `family(n₀ − k)`, `k ≥ 0`,
    /// written with the nonnegative integer parameter `k` given.
    Rays(Vec<ExprId>),
    /// Not decided.
    Undecided,
}

/// The members of the periodic family `fam` (linear in the integer
/// parameter `n`, which ranges over ℤ) in the declared domain of `var`:
/// all or none when the assumption system decides the family as a whole
/// (`2nπ + π/6` is real, `2nπi + i` is not), otherwise for constant
/// families and a domain on the real line given by signs (positive,
/// nonnegative, negative, nonpositive, nonzero, real): the rays of a real
/// progression cut by `0` (`x` positive: `nπ` is `π + kπ`, `k ≥ 0`), the
/// one real member of a family with a non-real step, or none for a real
/// step and a non-real offset.  `Undecided` for discrete domains (integer,
/// rational, …), symbolic families and members not certified.
pub(crate) fn family_in_domain(
    arena: &mut Arena,
    var: ExprId,
    fam: ExprId,
    n: ExprId,
    k: ExprId,
) -> FamilyInDomain {
    use crate::base::assumptions::Props;
    let Some(declared) = declared_domain(arena, var) else {
        return FamilyInDomain::All;
    };
    match assumptions_hold(arena, var, fam) {
        Some(true) => return FamilyInDomain::All,
        Some(false) => return FamilyInDomain::Empty,
        None => {}
    }
    let member = |arena: &mut Arena, j: i64| -> ExprId {
        let jj = arena.int(j);
        let m = crate::transforms::subs::subs(arena, fam, n, jj);
        crate::transforms::eval::eval(arena, m)
    };
    let s0 = member(arena, 0);
    let s1 = member(arena, 1);
    let d = arena.sub(s1, s0);
    let d = crate::transforms::eval::eval(arena, d);
    if !crate::base::walk::free_symbols(arena, s0).is_empty()
        || !crate::base::walk::free_symbols(arena, d).is_empty()
    {
        return FamilyInDomain::Undecided;
    }
    // The decisions below read the family as the progression `s₀ + n·d`;
    // one that is not (`√(2nπ + π/6)` from `sin(x²) = 1/2`, whose members
    // for `n < 0` are imaginary) is left to the intersection with the
    // domain.  Before 0.45 the real `s₀` and `d` of `√(2nπ + π/6)` made the
    // whole family real, and the solution set of `sin(x²) = 1/2` for a real
    // `x` contained `i·√(11π/6)`.
    let nd = arena.mul(&[n, d]);
    let progression = arena.add(&[s0, nd]);
    let gap = arena.sub(fam, progression);
    let gap = crate::transforms::expand::expand(arena, gap);
    let gap = crate::transforms::eval::eval(arena, gap);

    if !arena.is_zero_structural(gap) {
        return FamilyInDomain::Undecided;
    }
    // A family `s₀ + n·c·π` (`c ≠ 0` rational) with `sin s₀` algebraic in a
    // domain of algebraic numbers (integer, rational, even, prime, …):
    // every member has an algebraic sine (`sin(s₀ + θ)` for `θ ∈ ℚπ`), and
    // the sine of a nonzero algebraic number is transcendental
    // (Lindemann–Weierstrass), so `0` is the only member that can be in
    // the domain (`nπ`, `2nπ + asin(1/3)`, `nπ + atan(1/2)` for an integer
    // `x`).
    let algebraic_domain = [Props::INTEGER, Props::RATIONAL, Props::ALGEBRAIC]
        .iter()
        .any(|&p| declared.query(p) == Some(true));
    if algebraic_domain
        && let Some(c) = rational_multiple_of_pi(arena, d)
        && !c.is_zero()
        && exp_i_is_algebraic(arena, s0)
    {
        // the only index at which the member could be `0`
        let zero_at = match rational_multiple_of_pi(arena, s0) {
            Some(r) => {
                let j = -r / c;
                Some(j.is_integer())
            }
            None => match crate::transforms::evalf::evalf_complex64(arena, s0) {
                Ok(z) if z.re.is_finite() => {
                    let t = -z.re
                        / (num_traits::ToPrimitive::to_f64(&c).unwrap_or(1.0)
                            * std::f64::consts::PI);
                    let m = member(arena, t.round() as i64);
                    if arena.is_zero_structural(m) {
                        Some(true)
                    } else {
                        part_signs(arena, m).nonzero().then_some(false)
                    }
                }
                _ => None,
            },
        };
        let zero = arena.zero;
        return match zero_at {
            Some(false) => FamilyInDomain::Empty,
            Some(true) => match assumptions_hold(arena, var, zero) {
                Some(true) => FamilyInDomain::Point(zero),
                Some(false) => FamilyInDomain::Empty,
                None => FamilyInDomain::Undecided,
            },
            None => FamilyInDomain::Undecided,
        };
    }
    let discrete = [
        Props::INTEGER,
        Props::RATIONAL,
        Props::ALGEBRAIC,
        Props::IRRATIONAL,
        Props::TRANSCENDENTAL,
        Props::EVEN,
        Props::ODD,
        Props::PRIME,
        Props::COMPOSITE,
    ]
    .iter()
    .any(|&p| declared.query(p).is_some());

    // (A discrete domain on the real line is decided below only for a
    // non-real step, when at most one member is real.)
    if discrete && declared.query(Props::REAL) != Some(true) {
        return FamilyInDomain::Undecided;
    }
    let (Ok(z0), Ok(zd)) = (
        crate::transforms::evalf::evalf_complex64(arena, s0),
        crate::transforms::evalf::evalf_complex64(arena, d),
    ) else {
        return FamilyInDomain::Undecided;
    };
    let mut reals = crate::base::assumptions::AssumptionCache::new();
    let d_real = crate::transforms::realness::constant_realness_as_declared(
        arena,
        d,
        ROOT_CHECK_DIGITS,
        &mut reals,
    );
    let s_real = crate::transforms::realness::constant_realness_as_declared(
        arena,
        s0,
        ROOT_CHECK_DIGITS,
        &mut reals,
    );
    // The index `t` of the member on the boundary (`Im = 0`, or `0`): the
    // members decided are those next to it.
    let near = |t: f64| t.is_finite() && t.abs() < 1e12;
    let holds = |p: Props| declared.query(p) == Some(true);
    let ray = |arena: &mut Arena, from: i64, up: bool| -> ExprId {
        let from = arena.int(from);
        let idx = if up {
            arena.add(&[from, k])
        } else {
            arena.sub(from, k)
        };
        let r = crate::transforms::subs::subs(arena, fam, n, idx);
        let r = crate::transforms::expand::expand(arena, r);
        crate::transforms::eval::eval(arena, r)
    };
    // The family without its member `0`: the rays on either side of it.
    let without_zero = |arena: &mut Arena| -> FamilyInDomain {
        let neg_s0 = arena.neg(s0);
        let q = arena.div(neg_s0, d);
        let q = crate::transforms::eval::eval(arena, q);
        let at = if let Some(r) = arena.as_num(q) {
            Some(r.is_integer().then(|| r.to_integer()))
        } else {
            match crate::transforms::evalf::evalf_complex64(arena, q) {
                Ok(z) if near(z.re) && near(z.im) => {
                    let j = arena.int(z.re.round() as i64);
                    let off = arena.sub(q, j);
                    let off = crate::transforms::eval::eval(arena, off);
                    part_signs(arena, off).nonzero().then_some(None)
                }
                _ => None,
            }
        };
        match at {
            Some(None) => FamilyInDomain::All,
            Some(Some(j)) => match i64::try_from(j) {
                Ok(j) => {
                    let up = ray(arena, j + 1, true);
                    let down = ray(arena, j - 1, false);
                    FamilyInDomain::Rays(vec![up, down])
                }
                Err(_) => FamilyInDomain::Undecided,
            },
            None => FamilyInDomain::Undecided,
        }
    };
    if holds(Props::IMAGINARY) || declared.query(Props::REAL) == Some(false) {
        return FamilyInDomain::Undecided;
    }
    if declared.query(Props::REAL) != Some(true) {
        // A domain in ℂ restricted only by `≠ 0`.
        let only_nonzero = DOMAIN_PROPS
            .iter()
            .all(|&p| p == Props::NONZERO || p == Props::ZERO || declared.query(p).is_none());
        return if only_nonzero && holds(Props::NONZERO) {
            without_zero(arena)
        } else {
            FamilyInDomain::Undecided
        };
    }
    if d_real == Some(false) {
        // A non-real step: at most one real member, at `n = −Im s₀ / Im d`.
        let t = -z0.im / zd.im;
        if !near(t) {
            return FamilyInDomain::Undecided;
        }
        // Every other member is at least half a step off the real line.
        let m = member(arena, t.round() as i64);
        return match assumptions_hold(arena, var, m) {
            Some(true) => FamilyInDomain::Point(m),
            Some(false) => FamilyInDomain::Empty,
            None => FamilyInDomain::Undecided,
        };
    }
    if discrete || d_real != Some(true) || !part_signs(arena, d).re.is_nonzero().is_true() {
        return FamilyInDomain::Undecided;
    }
    match s_real {
        Some(true) => {}
        Some(false) => return FamilyInDomain::Empty,
        None => return FamilyInDomain::Undecided,
    }
    // A real progression `s₀ + n·d`; the domain is a half-line, the line
    // without 0, or the line.
    let lower = holds(Props::POSITIVE) || holds(Props::NONNEGATIVE);
    let upper = holds(Props::NEGATIVE) || holds(Props::NONPOSITIVE);
    let t = -z0.re / zd.re;
    if !near(t) || (lower && upper) {
        return FamilyInDomain::Undecided;
    }
    if lower || upper {
        // The admissible members are `n ≥ n₀` or `n ≤ n₀`: find the
        // boundary between two decided neighbours near `t`.
        let increasing = (zd.re > 0.0) == lower;
        let base = t.floor() as i64;
        for j in (base - 2)..=(base + 3) {
            let (inside, outside) = if increasing { (j, j - 1) } else { (j, j + 1) };
            let mi = member(arena, inside);
            let mo = member(arena, outside);
            if assumptions_hold(arena, var, mi) == Some(true)
                && assumptions_hold(arena, var, mo) == Some(false)
            {
                return FamilyInDomain::Rays(vec![ray(arena, inside, increasing)]);
            }
        }
        return FamilyInDomain::Undecided;
    }
    if holds(Props::NONZERO) {
        return without_zero(arena);
    }
    // real (the family is real)
    FamilyInDomain::All
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::arena::Arena;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }

    fn display(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    /// The roots [`known_real_roots`] claims, as `Solution`s (the `solve`
    /// of 0.48 these tests were written against).
    fn solve(a: &mut Arena, expr: ExprId, var: ExprId) -> Vec<Solution> {
        known_real_roots(a, expr, var)
            .into_iter()
            .map(|value| Solution { value })
            .collect()
    }

    fn solution_strings(a: &Arena, solutions: &[Solution]) -> Vec<String> {
        solutions.iter().map(|s| display(a, s.value)).collect()
    }

    /// `Arena::solve_for` returned `[]` for "no root" (`eˣ + 1` for a real
    /// `x`), for "not solved" (`x − cos x`, root ≈ 0.739) and for an
    /// identity; `Arena::solve_roots` (0.50) tells them apart.
    #[test]
    fn arena_solve_roots_tells_refusal_from_no_root() {
        use crate::base::errors::SymplexError;
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let (two, four, one) = (a.int(2), a.int(4), a.one);
        let x2 = a.pow(x, two);
        let e = a.sub(x2, four);
        assert!(matches!(a.solve_roots(e, x), Ok(r) if r.len() == 2));
        let ex = a.exp(x);
        let e = a.add(&[ex, one]);
        assert!(matches!(
            a.solve_roots(e, x),
            Err(SymplexError::NoSolution { .. })
        ));
        let cx = a.cos(x);
        let e = a.sub(x, cx);
        assert!(matches!(
            a.solve_roots(e, x),
            Err(SymplexError::ComputationFailed { .. })
        ));
    }

    // ── Linear ──────────────────────────────────────────────────────

    #[test]
    fn solve_linear_simple() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x - 3 = 0 → x = 3
        let three = a.int(3);
        let expr = a.sub(x, three);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1);
        assert_eq!(display(&a, solutions[0].value), "3");
    }

    #[test]
    fn solve_linear_with_coefficient() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // 2*x - 6 = 0 → x = 3
        let two = a.int(2);
        let six = a.int(6);
        let two_x = a.mul(&[two, x]);
        let expr = a.sub(two_x, six);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1);
        assert_eq!(display(&a, solutions[0].value), "3");
    }

    #[test]
    fn solve_linear_rational_solution() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // 3*x - 1 = 0 → x = 1/3
        let three = a.int(3);
        let one = a.one;
        let three_x = a.mul(&[three, x]);
        let expr = a.sub(three_x, one);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1);
        assert_eq!(display(&a, solutions[0].value), "1/3");
    }

    #[test]
    fn solve_linear_negative_solution() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x + 5 = 0 → x = -5
        let five = a.int(5);
        let expr = a.add(&[x, five]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1);
        assert_eq!(display(&a, solutions[0].value), "-5");
    }

    // ── Quadratic ───────────────────────────────────────────────────

    #[test]
    fn solve_quadratic_two_roots() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 - 5x + 6 = 0 → x = 2, x = 3
        let two = a.int(2);
        let five = a.int(5);
        let six = a.int(6);
        let x_sq = a.pow(x, two);
        let five_x = a.mul(&[five, x]);
        let neg_five_x = a.neg(five_x);
        let expr = a.add(&[x_sq, neg_five_x, six]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 2);
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(
            vals.contains(&"2".to_string()),
            "should have root 2: {vals:?}"
        );
        assert!(
            vals.contains(&"3".to_string()),
            "should have root 3: {vals:?}"
        );
    }

    #[test]
    fn solve_quadratic_double_root() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 - 2x + 1 = 0 → x = 1 (double root)
        let two = a.int(2);
        let one = a.one;
        let x_sq = a.pow(x, two);
        let two_x = a.mul(&[two, x]);
        let neg_two_x = a.neg(two_x);
        let expr = a.add(&[x_sq, neg_two_x, one]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1);
        assert_eq!(display(&a, solutions[0].value), "1");
    }

    #[test]
    fn solve_quadratic_no_real_roots() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 + 1 = 0 → complex roots ±i
        let two = a.int(2);
        let one = a.one;
        let x_sq = a.pow(x, two);
        let expr = a.add(&[x_sq, one]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 2, "x²+1=0 should have 2 complex roots");
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(
            vals.iter().all(|v| v.contains("I")),
            "roots should contain I: {vals:?}"
        );
    }

    #[test]
    fn solve_quadratic_irrational_roots() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 - 2 = 0 → x = ±√2
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let expr = a.sub(x_sq, two);
        let solutions = solve(&mut a, expr, x);
        // Should return symbolic sqrt(2) and -sqrt(2).
        assert_eq!(solutions.len(), 2, "x²-2=0 should have 2 solutions");
        let vals: Vec<String> = solution_strings(&a, &solutions);
        // Check that one contains sqrt and the other is negative.
        let has_sqrt = vals.iter().any(|v| v.contains("sqrt"));
        assert!(has_sqrt, "should contain sqrt(2): {vals:?}");
    }

    #[test]
    fn solve_x_squared_minus_one() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 - 1 = 0 → x = 1, x = -1
        let two = a.int(2);
        let one = a.one;
        let x_sq = a.pow(x, two);
        let expr = a.sub(x_sq, one);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 2);
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(vals.contains(&"1".to_string()));
        assert!(vals.contains(&"-1".to_string()));
    }

    // ── Cubic via rational roots ────────────────────────────────────

    #[test]
    fn solve_cubic_all_rational() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
        let three = a.int(3);
        let six = a.int(6);
        let eleven = a.int(11);
        let two = a.int(2);

        let x3 = a.pow(x, three);
        let x2 = a.pow(x, two);
        let six_x2 = a.mul(&[six, x2]);
        let eleven_x = a.mul(&[eleven, x]);

        let neg_six_x2 = a.neg(six_x2);
        let neg_six = a.neg(six);
        let expr = a.add(&[x3, neg_six_x2, eleven_x, neg_six]);

        let solutions = solve(&mut a, expr, x);
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(
            vals.contains(&"1".to_string()),
            "should have root 1: {vals:?}"
        );
        assert!(
            vals.contains(&"2".to_string()),
            "should have root 2: {vals:?}"
        );
        assert!(
            vals.contains(&"3".to_string()),
            "should have root 3: {vals:?}"
        );
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn solve_constant_nonzero_no_solutions() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // 5 = 0 → no solutions.
        let five = a.int(5);
        let solutions = solve(&mut a, five, x);
        assert!(solutions.is_empty());
        assert!(solve_classified(&mut a, five, x).is_none_proved());
    }

    #[test]
    fn solve_zero_expression() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // 0 = 0 → infinite solutions (bare `solve` returns empty).
        let zero = a.zero;
        let solutions = solve(&mut a, zero, x);
        assert!(solutions.is_empty());
        assert!(matches!(
            solve_classified(&mut a, zero, x),
            Solutions::Complete(RootSet::All)
        ));
    }

    #[test]
    fn solve_classified_identity_after_eval() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // sin(0) + 0*x is independent of x and evaluates to 0.
        let zero = a.zero;
        let sin0 = a.sin(zero);
        assert!(matches!(
            solve_classified(&mut a, sin0, x),
            Solutions::Complete(RootSet::All)
        ));
    }

    #[test]
    fn solve_classified_exp_eq_zero_no_solution() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let e = a.exp(x);
        assert!(solve_classified(&mut a, e, x).is_none_proved());
    }

    #[test]
    fn solve_classified_polynomial_solutions() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let four = a.int(4);
        let expr = a.sub(x2, four);
        match solve_classified(&mut a, expr, x).points_if_complete() {
            Some(s) => assert_eq!(s.len(), 2),
            None => panic!("expected every root"),
        }
    }

    // ── General (periodic) solutions ────────────────────────────────────

    #[test]
    fn solve_general_sin_half_has_period_param() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let n = sym(&mut a, "n");
        let half = a.rational(1, 2);
        let sx = a.sin(x);
        let expr = a.sub(sx, half);
        let out = solve_general(&mut a, expr, x, n);
        assert!(matches!(out, Solutions::Complete(RootSet::Families { .. })));
        let sols = out.claimed_points();
        assert_eq!(sols.len(), 2, "two families expected");
        for &s in &sols {
            assert!(
                expr_contains_var(&a, s, n),
                "family should mention n: {}",
                display(&a, s)
            );
            assert!(display(&a, s).contains("pi"));
        }
    }

    #[test]
    fn solve_general_tan_has_pi_n() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let n = sym(&mut a, "n");
        let one = a.one;
        let tx = a.tan(x);
        let expr = a.sub(tx, one);
        let sols = solve_general(&mut a, expr, x, n).claimed_points();
        assert_eq!(sols.len(), 1);
        let s = display(&a, sols[0]);
        assert!(s.contains("n") && s.contains("pi"), "got {s}");
    }

    #[test]
    fn solve_general_polynomial_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let n = sym(&mut a, "n");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let one = a.one;
        let expr = a.sub(x2, one);
        let sols = solve_general(&mut a, expr, x, n).claimed_points();
        assert_eq!(sols.len(), 2);
        for &s in &sols {
            assert!(!expr_contains_var(&a, s, n));
        }
    }

    #[test]
    fn solve_sin_linear_argument() {
        // sin(2x + 1) = 1/2 requires peeling through an Add node.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let one = a.one;
        let two_x = a.mul(&[two, x]);
        let arg = a.add(&[two_x, one]);
        let s = a.sin(arg);
        let half = a.rational(1, 2);
        let expr = a.sub(s, half);
        let sols = solve(&mut a, expr, x);
        assert_eq!(sols.len(), 2, "got {:?}", solution_strings(&a, &sols));
    }

    #[test]
    fn solve_symbolic_quadratic_coefficients() {
        // x^2 - k = 0 with symbolic k → ±sqrt(k)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let k = sym(&mut a, "k");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let expr = a.sub(x2, k);
        let sols = solve(&mut a, expr, x);
        assert_eq!(sols.len(), 2, "got {:?}", solution_strings(&a, &sols));
        for s in &sols {
            assert!(expr_contains_var(&a, s.value, k));
        }
    }

    #[test]
    fn solve_binomial_quintic_roots_of_unity() {
        // x^5 - 2 = 0 → five explicit roots, no RootOf.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let x5 = a.pow(x, five);
        let two = a.int(2);
        let expr = a.sub(x5, two);
        let sols = solve(&mut a, expr, x);
        assert_eq!(sols.len(), 5);
        for s in &sols {
            assert!(!matches!(a.node(s.value), ExprNode::RootOf(..)));
        }
    }

    #[test]
    fn solve_sin_x_eq_zero_via_change_of_variable() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // sin(x) = 0 → change-of-variable with t = sin(x) finds t = 0,
        // then back-substitutes sin(x) = 0 → x ∈ {asin(0), π - asin(0)} = {0, π}.
        let expr = a.sin(x);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(
            solutions.len(),
            2,
            "sin(x)=0 should have 2 solutions (two branches), got {}",
            solutions.len()
        );
        let vals: Vec<String> = solutions.iter().map(|s| display(&a, s.value)).collect();
        // asin(0) is not auto-evaluated, so expect symbolic forms
        assert!(
            vals.iter().any(|v| v == "0" || v.contains("asin(0)")),
            "should have root 0 or asin(0): {vals:?}"
        );
        assert!(
            vals.iter()
                .any(|v| v == "pi" || v.contains("pi") || v.contains("asin")),
            "should have root involving pi or asin: {vals:?}"
        );
    }

    #[test]
    fn solve_with_zero_root() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 - x = x*(x-1) = 0 → x = 0, x = 1
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let expr = a.sub(x_sq, x);
        let solutions = solve(&mut a, expr, x);
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(
            vals.contains(&"0".to_string()),
            "should have root 0: {vals:?}"
        );
        assert!(
            vals.contains(&"1".to_string()),
            "should have root 1: {vals:?}"
        );
    }

    // ── Verification ────────────────────────────────────────────────

    #[test]
    fn solve_and_verify_quadratic() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 - 5x + 6 = 0
        let two = a.int(2);
        let five = a.int(5);
        let six = a.int(6);
        let x_sq = a.pow(x, two);
        let five_x = a.mul(&[five, x]);
        let neg_five_x = a.neg(five_x);
        let expr = a.add(&[x_sq, neg_five_x, six]);

        let solutions = solve(&mut a, expr, x);
        // Verify each solution by substitution.
        for sol in &solutions {
            let val = crate::transforms::subs::subs(&mut a, expr, x, sol.value);
            assert!(
                a.is_zero_structural(val),
                "substituting x={} should give 0, got {}",
                display(&a, sol.value),
                display(&a, val)
            );
        }
    }

    // ── Transcendental / inversion peeling ──────────────────────────

    #[test]
    fn solve_exp_x_eq_5() {
        // exp(x) - 5 = 0 → x = ln(5)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let exp_x = a.exp(x);
        let expr = a.sub(exp_x, five);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "exp(x)-5=0 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(
            val.contains("ln") || val.contains("log"),
            "solution should be ln(5): {val}"
        );
    }

    #[test]
    fn solve_ln_x_eq_2() {
        // ln(x) - 2 = 0 → x = exp(2) = e^2
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let ln_x = a.ln(x);
        let expr = a.sub(ln_x, two);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "ln(x)-2=0 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(
            val.contains("exp") || val.contains("e") || val.contains("E"),
            "solution should be exp(2): {val}"
        );
    }

    #[test]
    fn solve_sin_x_eq_half() {
        // sin(x) - 1/2 = 0 → x ∈ {asin(1/2), π - asin(1/2)}
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let half = {
            let nid = a.intern_num(Ratio::new(BigInt::from(1), BigInt::from(2)));
            a.intern(ExprNode::Num(nid))
        };
        let sin_x = a.sin(x);
        let expr = a.sub(sin_x, half);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(
            solutions.len(),
            2,
            "sin(x)-1/2=0 should have 2 solutions (two branches)"
        );
        // Solutions are evaluated: asin(1/2) → π/6 and π - asin(1/2) → 5π/6.
        let val0 = display(&a, solutions[0].value);
        let val1 = display(&a, solutions[1].value);
        assert!(
            val0.contains("pi") || val0.contains("asin"),
            "first solution should be pi/6 (or asin(1/2)): {val0}"
        );
        assert!(
            val1.contains("pi") || val1.contains("asin"),
            "second solution should be 5*pi/6 (or pi - asin(1/2)): {val1}"
        );
        assert_ne!(val0, val1);
    }

    #[test]
    fn solve_sqrt_x_eq_3() {
        // sqrt(x) - 3 = 0 → x = 9
        // sqrt(x) is x^(1/2)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let sqrt_x = a.sqrt(x);
        let expr = a.sub(sqrt_x, three);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "sqrt(x)-3=0 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert_eq!(val, "9", "solution should be 9: {val}");
    }

    #[test]
    fn solve_mul_factors() {
        // x*(x-1)*(x+2) = 0 → roots 0, 1, -2
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let two = a.int(2);
        let x_minus_1 = a.sub(x, one);
        let x_plus_2 = a.add(&[x, two]);
        let expr = a.mul(&[x, x_minus_1, x_plus_2]);
        let solutions = solve(&mut a, expr, x);
        assert!(
            solutions.len() >= 3,
            "x*(x-1)*(x+2)=0 should have 3 roots, got {}",
            solutions.len()
        );
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(
            vals.contains(&"0".to_string()),
            "should have root 0: {vals:?}"
        );
        assert!(
            vals.contains(&"1".to_string()),
            "should have root 1: {vals:?}"
        );
        assert!(
            vals.contains(&"-2".to_string()),
            "should have root -2: {vals:?}"
        );
    }

    #[test]
    fn solve_2_exp_x_minus_6() {
        // 2*exp(x) - 6 = 0 → exp(x) = 3 → x = ln(3)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let six = a.int(6);
        let exp_x = a.exp(x);
        let two_exp_x = a.mul(&[two, exp_x]);
        let expr = a.sub(two_exp_x, six);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "2*exp(x)-6=0 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(
            val.contains("ln") || val.contains("log"),
            "solution should be ln(3): {val}"
        );
    }

    #[test]
    fn solve_and_verify_cubic() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^3 - 6x^2 + 11x - 6 = 0
        let three = a.int(3);
        let six = a.int(6);
        let eleven = a.int(11);
        let two = a.int(2);

        let x3 = a.pow(x, three);
        let x2 = a.pow(x, two);
        let six_x2 = a.mul(&[six, x2]);
        let eleven_x = a.mul(&[eleven, x]);

        let neg_six_x2 = a.neg(six_x2);
        let neg_six = a.neg(six);
        let expr = a.add(&[x3, neg_six_x2, eleven_x, neg_six]);

        let solutions = solve(&mut a, expr, x);
        for sol in &solutions {
            let val = crate::transforms::subs::subs(&mut a, expr, x, sol.value);
            assert!(
                a.is_zero_structural(val),
                "substituting x={} should give 0, got {}",
                display(&a, sol.value),
                display(&a, val)
            );
        }
    }

    // ── Change-of-variable ──────────────────────────────────────────

    #[test]
    fn solve_exp_2x_minus_3_exp_x_plus_2() {
        // exp(2x) - 3*exp(x) + 2 = 0
        // Let t = exp(x): t² - 3t + 2 = (t-1)(t-2) = 0
        // t = 1 → x = ln(1) = 0
        // t = 2 → x = ln(2)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let three = a.int(3);

        let two_x = a.mul(&[two, x]);
        let exp_2x = a.exp(two_x);
        let exp_x = a.exp(x);
        let three_exp_x = a.mul(&[three, exp_x]);
        let neg_three_exp_x = a.neg(three_exp_x);
        let expr = a.add(&[exp_2x, neg_three_exp_x, two]);

        let solutions = solve(&mut a, expr, x);
        assert!(
            solutions.len() >= 2,
            "exp(2x)-3*exp(x)+2=0 should have 2 solutions, got {}: {:?}",
            solutions.len(),
            solution_strings(&a, &solutions)
        );
        let vals: Vec<String> = solution_strings(&a, &solutions);
        // ln(1) may or may not simplify to 0 depending on canonicalization
        assert!(
            vals.contains(&"0".to_string()) || vals.contains(&"ln(1)".to_string()),
            "should have root 0 or ln(1) (from exp(x)=1): {vals:?}"
        );
        let has_ln2 = vals.iter().any(|v| v.contains("ln") || v.contains("log"));
        assert!(has_ln2, "should have root ln(2): {vals:?}");
    }

    #[test]
    fn solve_sin_squared_minus_sin() {
        // sin(x)^2 - sin(x) = 0
        // Let t = sin(x): t² - t = t(t-1) = 0
        // t = 0 → sin(x) = 0 → x = asin(0) = 0
        // t = 1 → sin(x) = 1 → x = asin(1)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);

        let sin_x = a.sin(x);
        let sin_x_sq = a.pow(sin_x, two);
        let expr = a.sub(sin_x_sq, sin_x);

        let solutions = solve(&mut a, expr, x);
        assert!(
            solutions.len() >= 2,
            "sin(x)^2-sin(x)=0 should have ≥2 solutions, got {}: {:?}",
            solutions.len(),
            solution_strings(&a, &solutions)
        );
        let vals: Vec<String> = solution_strings(&a, &solutions);
        // One root should be 0 (from sin(x)=0 → x=asin(0)=0)
        let has_zero = vals.contains(&"0".to_string());
        // The other should involve asin (from sin(x)=1 → x=asin(1))
        let has_asin = vals
            .iter()
            .any(|v| v.contains("asin") || v.contains("arcsin"));
        assert!(
            has_zero || has_asin,
            "should have root 0 or asin(1): {vals:?}"
        );
    }

    #[test]
    fn solve_exp_quadratic_one_valid_root() {
        // exp(2x) - 5*exp(x) + 6 = 0
        // Let t = exp(x): t² - 5t + 6 = (t-2)(t-3) = 0
        // t = 2 → x = ln(2)
        // t = 3 → x = ln(3)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let five = a.int(5);
        let six = a.int(6);

        let two_x = a.mul(&[two, x]);
        let exp_2x = a.exp(two_x);
        let exp_x = a.exp(x);
        let five_exp_x = a.mul(&[five, exp_x]);
        let neg_five_exp_x = a.neg(five_exp_x);
        let expr = a.add(&[exp_2x, neg_five_exp_x, six]);

        let solutions = solve(&mut a, expr, x);
        assert!(
            solutions.len() >= 2,
            "exp(2x)-5*exp(x)+6=0 should have 2 solutions, got {}: {:?}",
            solutions.len(),
            solution_strings(&a, &solutions)
        );
        let vals: Vec<String> = solution_strings(&a, &solutions);
        let has_ln = vals.iter().all(|v| v.contains("ln") || v.contains("log"));
        assert!(has_ln, "all roots should involve ln: {vals:?}");
    }

    // ── Helper tests ────────────────────────────────────────────────

    #[test]
    fn rational_sqrt_perfect() {
        let r = Ratio::new(BigInt::from(9), BigInt::from(4));
        let s = rational_sqrt(&r).unwrap();
        assert_eq!(s, Ratio::new(BigInt::from(3), BigInt::from(2)));
    }

    #[test]
    fn rational_sqrt_not_perfect() {
        let r = Ratio::from_integer(BigInt::from(2));
        assert!(rational_sqrt(&r).is_none());
    }

    #[test]
    fn rational_sqrt_zero() {
        let r = Ratio::zero();
        let s = rational_sqrt(&r).unwrap();
        assert!(s.is_zero());
    }

    #[test]
    fn divisors_of_12() {
        let d = divisors(&BigInt::from(12));
        assert_eq!(
            d,
            vec![1, 2, 3, 4, 6, 12]
                .into_iter()
                .map(BigInt::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn divisors_of_1() {
        let d = divisors(&BigInt::from(1));
        assert_eq!(d, vec![BigInt::from(1)]);
    }

    #[test]
    fn clear_denominators_works() {
        let poly = Poly::from_coeffs(vec![
            Ratio::new(BigInt::from(1), BigInt::from(2)),
            Ratio::new(BigInt::from(1), BigInt::from(3)),
        ]);
        let (int_poly, lcm) = clear_denominators(&poly);
        assert_eq!(lcm, BigInt::from(6));
        // 1/2 * 6 = 3, 1/3 * 6 = 2
        assert_eq!(int_poly.coeff(0), Ratio::from_integer(BigInt::from(3)));
        assert_eq!(int_poly.coeff(1), Ratio::from_integer(BigInt::from(2)));
    }

    // ── Complex quadratic roots ─────────────────────────────────────

    #[test]
    fn solve_x2_plus_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x² + 1 = 0 → roots ±i
        let two = a.int(2);
        let one = a.one;
        let x_sq = a.pow(x, two);
        let expr = a.add(&[x_sq, one]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 2, "x²+1=0 should have 2 complex roots");
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(
            vals.iter().all(|v| v.contains("I")),
            "roots should contain I: {vals:?}"
        );
    }

    #[test]
    fn solve_x2_plus_2x_plus_5() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x² + 2x + 5 = 0 → roots -1 ± 2i
        let two = a.int(2);
        let five = a.int(5);
        let x_sq = a.pow(x, two);
        let two_x = a.mul(&[two, x]);
        let expr = a.add(&[x_sq, two_x, five]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 2, "x²+2x+5=0 should have 2 complex roots");
        let vals: Vec<String> = solution_strings(&a, &solutions);
        // roots are -1 ± 2i
        for v in &vals {
            assert!(v.contains("I"), "root should contain I: {v}");
        }
    }

    #[test]
    fn solve_x2_plus_4() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x² + 4 = 0 → roots ±2i
        let two = a.int(2);
        let four = a.int(4);
        let x_sq = a.pow(x, two);
        let expr = a.add(&[x_sq, four]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 2, "x²+4=0 should have 2 complex roots");
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(
            vals.iter().all(|v| v.contains("I")),
            "roots should contain I: {vals:?}"
        );
    }

    #[test]
    fn solve_even_power_peeling_both_roots() {
        // (2x + 1)^2 = 9  →  2x + 1 = ±3  →  x = 1 or x = -2
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let two = a.int(2);
        let nine = a.int(9);
        let two_x = a.mul(&[two, x]);
        let inner = a.add(&[two_x, one]); // 2x + 1
        let squared = a.pow(inner, two); // (2x + 1)^2
        let neg_nine = a.neg(nine);
        let expr = a.add(&[squared, neg_nine]); // (2x + 1)^2 - 9
        let solutions = solve(&mut a, expr, x);
        let mut vals: Vec<String> = solution_strings(&a, &solutions);
        vals.sort();
        assert_eq!(vals.len(), 2, "expected 2 solutions, got {vals:?}");
        assert_eq!(vals, vec!["-2", "1"], "solutions: {vals:?}");
    }

    // ── LambertW solver ─────────────────────────────────────────────

    #[test]
    fn solve_lambert_x_exp_x_eq_1() {
        // x·exp(x) - 1 = 0  →  x = W(1)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let exp_x = a.exp(x);
        let x_exp_x = a.mul(&[x, exp_x]);
        let expr = a.sub(x_exp_x, one);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "x·exp(x)=1 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(val.contains("W("), "solution should be W(1): {val}");
    }

    #[test]
    fn solve_lambert_x_exp_x_eq_5() {
        // x·exp(x) - 5 = 0  →  x = W(5)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let exp_x = a.exp(x);
        let x_exp_x = a.mul(&[x, exp_x]);
        let expr = a.sub(x_exp_x, five);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "x·exp(x)=5 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(val.contains("W("), "solution should be W(5): {val}");
    }

    #[test]
    fn solve_lambert_x_exp_x_eq_0() {
        // x·exp(x) = 0  →  factored as Mul([x, Exp(x)]);
        // the Mul pre-check solves x=0 from the x factor.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let exp_x = a.exp(x);
        let expr = a.mul(&[x, exp_x]);
        let solutions = solve(&mut a, expr, x);
        assert!(!solutions.is_empty(), "x·exp(x)=0 should have a solution");
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(
            vals.contains(&"0".to_string()),
            "should have root 0: {vals:?}"
        );
    }

    #[test]
    fn solve_lambert_2x_exp_x_eq_4() {
        // 2·x·exp(x) - 4 = 0  →  x·exp(x) = 2  →  x = W(2)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let four = a.int(4);
        let exp_x = a.exp(x);
        let two_x_exp_x = a.mul(&[two, x, exp_x]);
        let expr = a.sub(two_x_exp_x, four);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "2·x·exp(x)=4 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(val.contains("W("), "solution should involve W: {val}");
    }

    #[test]
    fn solve_lambert_x_exp_2x_eq_3() {
        // x·exp(2·x) - 3 = 0
        // Multiply by 2: 2·x·exp(2·x) = 6 → 2·x = W(6) → x = W(6)/2
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let three = a.int(3);
        let two_x = a.mul(&[two, x]);
        let exp_2x = a.exp(two_x);
        let x_exp_2x = a.mul(&[x, exp_2x]);
        let expr = a.sub(x_exp_2x, three);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "x·exp(2x)=3 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(val.contains("W("), "solution should involve W: {val}");
    }

    #[test]
    fn solve_lambert_exp_x_plus_x_eq_2() {
        // exp(x) + x - 2 = 0  →  Pattern 2 (A=1, B=1, C=1, D=-2)
        // Solution: x = 2 - W(exp(2))
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let exp_x = a.exp(x);
        let sum = a.add(&[exp_x, x]);
        let expr = a.sub(sum, two);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "exp(x)+x-2=0 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(val.contains("W("), "solution should involve W: {val}");
    }

    #[test]
    fn solve_lambert_neg_exp_x_minus_x_plus_2() {
        // -exp(x) - x + 2 = 0  is the same equation as exp(x) + x - 2 = 0
        // Pattern 2 with A=-1, B=1, C=-1, D=2
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let exp_x = a.exp(x);
        let neg_exp_x = a.neg(exp_x);
        let neg_x = a.neg(x);
        let expr = a.add(&[neg_exp_x, neg_x, two]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "-exp(x)-x+2=0 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(val.contains("W("), "solution should involve W: {val}");
    }

    #[test]
    fn solve_lambert_x_squared_exp_x() {
        // x²·exp(x) - 1 = 0: x·e^(x/2) = ±1, x = 2·W(±1/2) (SymPy 1.14:
        // `[2*LambertW(-1/2), 2*LambertW(1/2)]`; the first is complex).
        // Refused (empty) before 0.44, when this test pinned that.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let exp_x = a.exp(x);
        let x2_exp_x = a.mul(&[x2, exp_x]);
        let expr = a.sub(x2_exp_x, one);
        let solutions = solve(&mut a, expr, x);
        let mut got = solution_strings(&a, &solutions);
        got.sort();
        assert_eq!(got, ["2*W(-1/2)", "2*W(1/2)"]);
    }

    #[test]
    fn solve_lambert_x_exp_x_eq_e_gives_1() {
        // x·exp(x) - e = 0  →  x = W(e) = 1
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let e = a.e_const;
        let exp_x = a.exp(x);
        let x_exp_x = a.mul(&[x, exp_x]);
        let expr = a.sub(x_exp_x, e);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "x·exp(x)=e should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert_eq!(val, "1", "W(e) should evaluate to 1: {val}");
    }

    #[test]
    fn solve_lambert_preserves_existing_polynomial() {
        // x^2 - 1 = 0 should still be solved by the polynomial path, not LambertW
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let expr = a.sub(x2, one);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 2, "x²-1 should still give 2 roots");
    }

    #[test]
    fn solve_lambert_preserves_existing_transcendental() {
        // exp(x) - 5 = 0 should still be solved by inversion peeling
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let exp_x = a.exp(x);
        let expr = a.sub(exp_x, five);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "exp(x)-5=0 should still give 1 root");
        let val = display(&a, solutions[0].value);
        assert!(
            val.contains("ln"),
            "solution should be ln(5), not lambertw: {val}"
        );
    }

    // ── Symbolic linear ─────────────────────────────────────────────

    #[test]
    fn solve_symbolic_linear_kx_minus_f() {
        let mut a = Arena::new();
        let k = sym(&mut a, "k");
        let x = sym(&mut a, "x");
        let f = sym(&mut a, "F");
        // k*x - F = 0, solve for x → x = F/k
        let kx = a.mul(&[k, x]);
        let expr = a.sub(kx, f);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1);
        let s = display(&a, solutions[0].value);
        assert!(s.contains('F') && s.contains('k'), "Expected F/k, got: {s}");
    }

    #[test]
    fn solve_symbolic_linear_bare_var() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let c = sym(&mut a, "c");
        // x + c = 0, solve for x → x = -c
        let expr = a.add(&[x, c]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1);
        let s = display(&a, solutions[0].value);
        // Should be -c or (-1)*c or similar
        assert!(s.contains('c'), "Expected -c, got: {s}");
    }

    #[test]
    fn solve_symbolic_linear_multiple_terms() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let p = sym(&mut a, "a");
        let b = sym(&mut a, "b");
        let c = sym(&mut a, "c");
        // a*x + b*x + c = 0 → x = -c/(a+b)
        let ax = a.mul(&[p, x]);
        let bx = a.mul(&[b, x]);
        let expr = a.add(&[ax, bx, c]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1);
        let s = display(&a, solutions[0].value);
        assert!(s.contains('c'), "Expected -c/(a+b), got: {s}");
    }

    // ── Bug 21 regression: exp/range domain checks ──────────────────

    #[test]
    fn solve_exp_x_eq_zero_no_solution() {
        // exp(x) = 0 has no real solution (exp(x) > 0 for all real x).
        // Regression: previously returned [ln(0)] instead of [].
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let exp_x = a.exp(x);
        let solutions = solve(&mut a, exp_x, x);
        assert!(
            solutions.is_empty(),
            "exp(x)=0 should have no solutions, got: {:?}",
            solution_strings(&a, &solutions)
        );
    }

    #[test]
    fn solve_exp_x_eq_negative_no_solution() {
        // exp(x) + 3 = 0  →  exp(x) = -3, no real solution.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let exp_x = a.exp(x);
        let expr = a.add(&[exp_x, three]);
        let solutions = solve(&mut a, expr, x);
        assert!(
            solutions.is_empty(),
            "exp(x)=-3 should have no solutions, got: {:?}",
            solution_strings(&a, &solutions)
        );
    }

    #[test]
    fn solve_sin_x_eq_2_no_solution() {
        // sin(x) = 2 has no real solution (sin range is [-1, 1]).
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let sin_x = a.sin(x);
        let expr = a.sub(sin_x, two);
        let solutions = solve(&mut a, expr, x);
        assert!(
            solutions.is_empty(),
            "sin(x)=2 should have no solutions, got: {:?}",
            solution_strings(&a, &solutions)
        );
    }

    #[test]
    fn solve_ln_x_eq_zero() {
        // ln(x) = 0 → x = exp(0) = 1.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let ln_x = a.ln(x);
        let solutions = solve(&mut a, ln_x, x);
        assert_eq!(solutions.len(), 1, "ln(x)=0 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(
            val == "1" || val.contains("exp(0)"),
            "solution should be 1 or exp(0), got: {val}"
        );
    }

    #[test]
    fn solve_abs_x_plus_1_no_solution() {
        // |x| + 1 = 0 → |x| = -1, impossible since |x| >= 0.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let abs_x = a.abs(x);
        let expr = a.add(&[abs_x, one]);
        let solutions = solve(&mut a, expr, x);
        assert!(
            solutions.is_empty(),
            "|x|+1=0 should have no solutions, got: {:?}",
            solution_strings(&a, &solutions)
        );
    }
}
