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
use crate::base::value_class;
use crate::poly::Poly;
use crate::poly::polybridge;

mod algebraic_roots;
mod context;
mod outcome;

pub(crate) use context::{Domain, Params, SolveCx};
pub(crate) use outcome::{Guarded, Missing, Reason, RootSet, Solutions};

mod change_of_var;
mod closed_forms;
mod declared;
mod lambert;
mod peeling;
mod rational_exp;
use change_of_var::*;
use closed_forms::*;
pub(crate) use declared::*;
use lambert::*;
pub(crate) use peeling::*;
use rational_exp::*;

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

/// Does `expr` hold `e^u`, `a^u` (a constant base), `sinh u`, `cosh u` or
/// `tanh u` of an argument `u` with `var` not known to be real for a real
/// `var` (`(−1)^x = e^{iπx}`), whose real zeros a principal solve lists one
/// period of, as it does for `sin`, `cos`, `tan` (see
/// [`SolveCx::lists_every_real_root`])?  A caller that needs every real
/// zero in a window takes the families of the general solve for such an
/// expression: up to 0.57 `Ex::singularities(1/((−1)^x − 1), x, [0, 10])`
/// was `{0}`, the principal root, for `{0, 2, 4, 6, 8, 10}`.
pub(crate) fn has_periodic_exponential(arena: &mut Arena, expr: ExprId, var: ExprId) -> bool {
    let cx = SolveCx::principal(Domain::Real);
    for id in crate::base::walk::post_order_ids(arena, expr) {
        let periodic = match *arena.node(id) {
            ExprNode::Exp(u) | ExprNode::Sinh(u) | ExprNode::Cosh(u) | ExprNode::Tanh(u) => {
                expr_contains_var(arena, u, var) && !cx.real_for_real_var(arena, u, var)
            }
            ExprNode::Pow(b, u)
                if !expr_contains_var(arena, b, var) && expr_contains_var(arena, u, var) =>
            {
                let ln_b = arena.ln(b);
                let ln_b = crate::transforms::eval::eval(arena, ln_b);
                let exponent = arena.mul(&[u, ln_b]);
                !cx.real_for_real_var(arena, exponent, var)
            }
            _ => false,
        };
        if periodic {
            return true;
        }
    }
    false
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
                        // (or whose value meets an unevaluated pole,
                        // `evalf::meets_infinity`, as in `finalize_solutions`)
                        if not_known_finite(arena, r) || arena.meets_infinity(r) {
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

/// Is `v` (evaluated) a value that holds `±∞`, `zoo` or `nan` outside the
/// bounds of formal integrals, sums and limits
/// ([`value_class::holds_special_atom_outside_formal`]; then its
/// [`ValueClass`](value_class::ValueClass) is never `Finite`) and is not finite at every point by
/// the assumptions (the public reading)?  Such a value is no number the
/// solver can claim: `−chebyshev_t(2, y·∞)` is `−∞` for every real `y ≠ 0`
/// and undefined at `y = 0`, `−sin ∞` has no value.  The value `eval` folds
/// (`atan ∞ = π/2`, `e^{−y·∞} = 0` for a positive `y`) has no infinity
/// left.  A constant without a special atom keeps the generic reading
/// (finite): `ValueClass::of(..).is_finite_proved()` would refuse every
/// root `evalf` cannot certify (`erfinv(9/4)`, `besselj(1/2, π) = 0`; 229
/// golden roots).  An unevaluated pole (`elliptic_pi(π, 1)`) is refused by
/// its evaluation instead, which proves it infinite
/// (`evalf::meets_infinity`; read in [`finalize_solutions`]).
fn not_known_finite(arena: &mut Arena, v: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    value_class::holds_special_atom_outside_formal(arena, v)
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
                infinite_part |= value_class::holds_special_atom_outside_formal(arena, child)
                    || (expr_contains_var(arena, child, var)
                        && denominator_vanishes_on_intervals(arena, child, var));
                if expr_contains_var(arena, child, var)
                    || arena.as_num(child).is_some()
                    || (!crate::base::walk::free_symbols(arena, child).is_empty()
                        && !value_class::holds_special_atom_outside_formal(arena, child))
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
    // Nor is a candidate whose value meets an unevaluated pole (`evalf`'s
    // `SymplexError::Infinite`): up to 0.57 `x − elliptic_pi(π, 1)` gave
    // `[elliptic_pi(π, 1)]`, which is `−∞`, and `x·elliptic_pi(π, 1) − 1`
    // `[1/elliptic_pi(π, 1)]`.  A candidate `evalf` cannot evaluate for
    // another reason (`erfinv(9/4)`) is no evidence of an infinity.
    let mut infinite: Option<ExprId> = None;
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
        let evaluation = constant_evaluation(arena, v);
        if matches!(
            evaluation,
            Some(Err(crate::base::errors::SymplexError::Infinite { .. }))
        ) {
            infinite.get_or_insert(v);
            continue;
        }
        // One value in several forms is one solution: up to 0.46 `t³` for
        // the three roots `t` of a cubic in `t = x^(1/3)` came back as
        // `100√2·π/481` and two radical forms of the same number.
        let z = evaluation
            .and_then(Result::ok)
            .filter(|z| z.re.is_finite() && z.im.is_finite());
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
    let missing = match infinite {
        Some(v) => Missing::merged(
            missing,
            Some(Missing::Alternative(Reason::said(format!(
                "the value of the candidate {} meets an infinity (a function at a pole)",
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
    constant_evaluation(arena, v)?
        .ok()
        .filter(|z| z.re.is_finite() && z.im.is_finite())
}

/// The evaluation behind [`constant_value_f64`], with its error (an
/// infinity met, `SymplexError::Infinite`, is read by
/// [`finalize_solutions`]); `None` for a candidate it does not evaluate.
fn constant_evaluation(
    arena: &mut Arena,
    v: ExprId,
) -> Option<Result<num_complex::Complex64, crate::base::errors::SymplexError>> {
    if arena.as_num(v).is_some()
        || !crate::base::walk::free_symbols(arena, v).is_empty()
        || crate::base::walk::post_order_ids(arena, v)
            .into_iter()
            .any(|id| matches!(arena.node(id), ExprNode::RootOf(..)))
    {
        return None;
    }
    Some(crate::transforms::evalf::evalf_complex64(arena, v))
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
    if value_class::holds_special_atom_outside_formal(arena, expanded)
        && !has_infinite_term(arena, expanded)
    {
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

/// The highest degree [`symbolic_poly_coeffs`] lists coefficients up to
/// (`2¹⁶`): up to 0.56 the breakpoint scan of the density of `F(1.5·10¹⁰,
/// 3/5)` (a factor `x^(7.5·10⁹ − 1)`) asked for a 360 GB coefficient vector
/// and aborted the process.
const MAX_DENSE_POLY_DEGREE: usize = crate::base::cast::MAX_DENSE_DEGREE;

/// Extract ascending coefficients of `expr` viewed as a polynomial in
/// `var`, allowing arbitrary var-free symbolic coefficients.
///
/// Returns `None` if `var` appears in a non-polynomial position, or if its
/// degree exceeds `MAX_DENSE_POLY_DEGREE`.
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
        // A dense list of coefficients up to the degree: beyond a bound no
        // caller can use, refused.  Up to 0.56 `x^23258988544` (the density
        // of an F distribution with `d₁ = 46517977088`, met by the breakpoint
        // scan of a quadrature) asked for a 1.1 TB vector and aborted.
        if deg > MAX_DENSE_POLY_DEGREE {
            return None;
        }
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
