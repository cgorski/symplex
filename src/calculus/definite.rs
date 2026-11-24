//! Definite and improper integration: singularity checks, limits at endpoints, known-value tables.
//!
//! The entry point is [`integrate_definite`], which computes
//! `∫ₐᵇ f(x) dx` **without ever returning a silently wrong finite number**.
//! Where the naive rule `F(b) − F(a)` would give `−2` for `∫₋₁¹ dx/x²`,
//! this module detects the interior pole, splits the interval there, and
//! reports [`SymplexError::Divergent`].
//!
//! # Algorithm
//!
//! 1. **Bounds.** `a = b` gives `0`; `a > b` swaps the bounds and negates.
//! 2. **Special integrands.** `DiracDelta`, `Heaviside`, `Abs`, `Sign`,
//!    `Piecewise`, `Floor`/`Ceiling` are eliminated first by truncating or
//!    splitting the interval so that every remaining piece is smooth (the
//!    integral is refused when a step cannot be located, e.g. `floor` of
//!    a non-linear argument or on an infinite interval).
//! 3. **Known-value table.** Classical improper integrals (Gaussian, Gamma,
//!    Beta, Dirichlet, Wallis, Fresnel, Bose–Einstein, …) are matched
//!    structurally with symbolic parameters; any positivity (or realness:
//!    an unassumed parameter may be complex) conditions are checked
//!    through the assumption system and the entry is skipped when they
//!    cannot be established.
//!    Rational × `cos`/`sin` over the line (or an even/odd one over `[0, ∞)`)
//!    and `ln x` × rational over `[0, ∞)` are summed over their poles
//!    (Jordan's lemma, the keyhole contour).
//! 4. **Breakpoints.** A factor `B(x)^(−k)` with `B ∈ ℚ[x]` whose real
//!    root in the range the other factors do not cancel (they are enclosed
//!    away from 0 next to it) proves divergence at once, from the root's
//!    isolating interval alone — no closed form of the root is needed.
//!    Poles, `ln` zeros, `tan` poles, branch points of
//!    fractional powers and inverse-trig domain edges of the integrand are
//!    located with the breakpoint scanner in `calculus_util`.  Interior points
//!    split the interval; endpoint singularities and infinite bounds are
//!    handled as improper integrals.  A numeric sign-change guard on every
//!    denominator catches zeros the symbolic solver missed; if a
//!    singularity cannot be placed relative to symbolic bounds the
//!    integral is left unevaluated rather than guessed.
//! 5. **Fundamental theorem.** An antiderivative `F` is obtained from the
//!    indefinite integrator.  Its own discontinuities inside the interval
//!    (e.g. the `atan(tan(x/2))` jump produced by the Weierstrass
//!    substitution) are split as well, and so are the exact points where a
//!    linear complex argument of `ln`, a power, `Ei`, `Ci` or `Chi` crosses
//!    the cut `(−∞, 0]`, each side taking the branch continued from it
//!    (`located_crossing`, `side_value`); a crossing that is only sampled
//!    is refused.  Endpoint values are direct
//!    substitutions when `F` is continuous there, otherwise **one-sided
//!    limits**: the approach `x → c⁺` is rewritten as `x = c + 1/u`,
//!    `u → +∞` (and `x = c − 1/u` for `c⁻`), so that every limit is a
//!    limit at `+∞`, where the Gruntz algorithm is strongest.  Limits are
//!    evaluated compositionally first (using assumptions for parameters)
//!    and only then by the black-box limit engine, whose finite results are
//!    sanity-checked numerically.
//! 6. **Divergence.** An infinite one-sided limit of `F` proves
//!    divergence.  At an infinite end the comparison with `1/x` comes
//!    first, on the integrand (`F` may be far harder to take a limit of);
//!    without an antiderivative, comparison with `1/(x − c)` is attempted
//!    at the finite singular ends; if nothing can be decided the
//!    integral is returned unevaluated.
//!
//! The same file also provides the adaptive Gauss–Kronrod (G7/K15)
//! quadrature used by `Ex::integrate_numeric` ([`quadrature`], [`QuadOpts`],
//! [`QuadResult`]).

use std::cmp::Ordering;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, ToPrimitive, Zero};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::base::arena::Arena;
use crate::base::assumptions::{AssumptionCache, Props};
use crate::base::errors::SymplexError;
use crate::base::extended::Extended;
use crate::base::interval::Interval;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::base::truth::{SignSet, Truth};
use crate::base::value_class::{Sign, ValueClass};
use crate::base::walk;
use crate::calculus::calculus_util::{
    self, BreakKind, BreakScan, Breakpoint, LinearCoeffs, POINT_VALUE_ACCURACY, ScanRange,
};
use crate::calculus::residue;
use crate::transforms::{eval, evalf, expand, subs};

mod branch_cuts;
mod endpoints;
mod gauss_kronrod;
mod known_values;
mod special_integrands;
use branch_cuts::*;
use endpoints::*;
pub use gauss_kronrod::*;
use known_values::*;
use special_integrands::*;

/// Maximum nesting of interval splits before giving up.
const MAX_DEPTH: usize = 8;
/// Maximum number of sub-intervals produced by any single split.
const MAX_PIECES: usize = 512;
/// Sample count for the numeric sign-change guard.
const GUARD_SAMPLES: usize = 129;

// ═══════════════════════════════════════════════════════════════════════════
// Seam: unevaluated definite integral
// ═══════════════════════════════════════════════════════════════════════════

/// Build the unevaluated form of `∫ₐᵇ f dx`: an
/// [`ExprNode::DefiniteIntegral`] node with the bounds preserved.
///
/// Every "cannot evaluate" fallback in the definite-integration API routes
/// through this single function.  The node constructor applies only the
/// structural folds (`a == b`, constant integrand over a finite interval,
/// reversed numeric bounds); it never attempts integration, so this cannot
/// loop back into [`integrate_definite`].
pub(crate) fn unevaluated_definite(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    a: ExprId,
    b: ExprId,
) -> ExprId {
    arena.definite_integral(f, x, a, b)
}

/// Evaluate every `DefiniteIntegral` node inside `f`, innermost first
/// (the "doit" operation for formal definite integrals).
///
/// Nodes that still cannot be evaluated — no closed form, or a proven
/// divergence — are left in place.  Used by [`integrate_definite`] so that
/// an integrand containing a formal definite integral (the fallback result
/// of an earlier call, or the `∫ ∂f/∂t dx` term of a Leibniz derivative) is
/// resolved before the outer integration runs, and exposed as
/// `Ex::eval_integrals`.  Bounded: each node is visited once; the inner
/// evaluations go through [`integrate_range`], never back through this
/// function.
pub(crate) fn evaluate_inner_definite(arena: &mut Arena, f: ExprId) -> ExprId {
    if !contains_node(arena, f, |n| matches!(n, ExprNode::DefiniteIntegral(..))) {
        return f;
    }
    let post_order = walk::post_order_ids(arena, f);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    for &id in &post_order {
        let node = arena.node(id).clone();
        let rebuilt = match node {
            ExprNode::DefiniteIntegral(body, var, lo, hi) => {
                let nb = cache.get(&body).copied().unwrap_or(body);
                let nl = cache.get(&lo).copied().unwrap_or(lo);
                let nh = cache.get(&hi).copied().unwrap_or(hi);
                match integrate_range(arena, nb, var, nl, nh, 1) {
                    Ok(v) if !walk::contains(arena, v, var) => safe_eval(arena, v),
                    _ => arena.definite_integral(nb, var, nl, nh),
                }
            }
            _ if node.is_atom() => id,
            _ => walk::rebuild_with_cache(arena, id, &cache),
        };
        cache.insert(id, rebuilt);
    }
    cache.get(&f).copied().unwrap_or(f)
}

// ═══════════════════════════════════════════════════════════════════════════
// Public entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the definite integral `∫ₐᵇ f dx` on the arena.
///
/// If `f` is, or contains, an unevaluated [`ExprNode::DefiniteIntegral`]
/// (for instance the fallback result of an earlier call, or the
/// `∫ ∂f/∂t dx` term produced by differentiating one), those inner
/// integrals are evaluated first, innermost out.
///
/// Returns:
///
/// * `Ok(value)` — an exact closed form (possibly containing symbolic
///   parameters of the integrand or symbolic bounds),
/// * `Err(SymplexError::Divergent { .. })` — the integral was **proven** to
///   diverge (non-integrable singularity, or an antiderivative with an
///   infinite one-sided limit),
/// * `Err(SymplexError::ComputationFailed { .. })` — no closed form could be
///   established without guessing (unknown singularity location relative
///   to symbolic bounds, no antiderivative and no table match, …),
/// * `Err(SymplexError::InvalidArgument { .. })` — `x` is not a symbol.
///
/// # Examples
///
/// The arena is reached through [`Context::with_arena_mut`](crate::context::Context::with_arena_mut):
///
/// ```
/// use symplex::prelude::*;
/// use symplex::definite::integrate_definite;
///
/// let ctx = Context::new();
/// let (value, divergent) = ctx.with_arena_mut(|a| {
///     let x = a.symbol("x");
///     let two = a.int(2);
///     let x2 = a.pow(x, two);
///     let zero = a.zero();
///     let one = a.one();
///     // ∫₀¹ x² dx = 1/3
///     let v = integrate_definite(a, x2, x, zero, one).unwrap();
///     let value = a.display(v).to_string();
///     // ∫₋₁¹ x⁻² dx diverges — the naive F(b) − F(a) = −2 is never returned.
///     let neg_two = a.int(-2);
///     let inv_sq = a.pow(x, neg_two);
///     let neg_one = a.neg_one();
///     let divergent = matches!(
///         integrate_definite(a, inv_sq, x, neg_one, one),
///         Err(SymplexError::Divergent { .. })
///     );
///     (value, divergent)
/// });
/// assert_eq!(value, "1/3");
/// assert!(divergent);
/// ```
pub fn integrate_definite(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    a: ExprId,
    b: ExprId,
) -> Result<ExprId, SymplexError> {
    if !matches!(arena.node(x), ExprNode::Symbol(_)) {
        return Err(SymplexError::InvalidArgument {
            operation: "integrate_definite",
            reason: "integration variable must be a symbol".into(),
        });
    }
    let f = safe_eval(arena, f);
    let f = evaluate_inner_definite(arena, f);
    let a = safe_eval(arena, a);
    let b = safe_eval(arena, b);
    // A bound in the variable itself (`∫_{3x}^{5x} x dx`): the variable of
    // integration is a dummy, the one in the bounds is free (SymPy: `8·x²`).
    // Up to 0.46 the two were the same symbol and the result was refused
    // as "still depending on the integration variable".
    if walk::contains(arena, a, x) || walk::contains(arena, b, x) {
        let dummy = fresh_symbol(arena, "idv", 0);
        let f_d = crate::transforms::subs::subs(arena, f, x, dummy);
        let result = integrate_definite(arena, f_d, dummy, a, b)?;
        if walk::contains(arena, result, dummy) {
            return Err(SymplexError::ComputationFailed {
                operation: "integrate_definite",
                reason: "result still depends on the integration variable".into(),
            });
        }
        return Ok(result);
    }
    let result = integrate_range(arena, f, x, a, b, 0)?;
    let result = safe_eval(arena, result);
    if walk::contains(arena, result, x) {
        return Err(SymplexError::ComputationFailed {
            operation: "integrate_definite",
            reason: "result still depends on the integration variable".into(),
        });
    }
    // One numeric value for both checks (a special-function closed form
    // costs milliseconds to evaluate).
    let numeric = if walk::free_symbols(arena, result).is_empty() {
        evalf::evalf_complex64(arena, result).ok()
    } else {
        None
    };
    if non_real_for_real_integrand(arena, f, x, a, b, numeric) {
        return Err(failed(
            "the closed form is not real although the integrand is (an antiderivative that jumps along the path?)",
        ));
    }
    if contradicted_by_quadrature(arena, f, x, a, b, numeric) {
        return Err(failed(
            "the closed form disagrees with adaptive quadrature (an antiderivative that jumps inside the interval?)",
        ));
    }
    Ok(result)
}

/// `∫_lo^hi body d(var)` (all in `arena`) by [`integrate_definite`], for
/// `evalf` of a `DefiniteIntegral` node whose quadrature refused: the
/// integral is copied into a private arena and integrated there.
/// `Ok(Some((arena, value)))` for a constant closed form (no symbol, no
/// unevaluated node), `Err` with the proof when the integral diverges,
/// `Ok(None)` otherwise — also for a body with another free symbol, a
/// bound that is not constant, or a nested definite integral.
///
/// Up to 0.51 a quadrature that could not evaluate its integrand at the
/// nodes refused although the integral has a closed form:
/// `∫ |sin(x/3 + 1/2)| dx` over `[10¹⁰⁰ + 7/10, 10¹⁰⁰ + 100]` (an
/// argument of `3·10⁹⁹` needs 330 bits before its first fractional one)
/// was refused by `eval_f64`, `eval_decimal` and `integrate_numeric` — in
/// 16 s at 0.50 — while this route answers in milliseconds.
///
/// # Errors
///
/// [`SymplexError::Divergent`] when [`integrate_definite`] proves the
/// integral divergent.
pub(crate) fn closed_form(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
) -> Result<Option<(Arena, ExprId)>, SymplexError> {
    if !matches!(arena.node(var), ExprNode::Symbol(_))
        || walk::free_symbols(arena, body).iter().any(|&s| s != var)
        || !walk::free_symbols(arena, lo).is_empty()
        || !walk::free_symbols(arena, hi).is_empty()
        || [body, lo, hi]
            .into_iter()
            .any(|e| contains_node(arena, e, |n| matches!(n, ExprNode::DefiniteIntegral(..))))
    {
        return Ok(None);
    }
    let mut pa = Arena::with_config(arena.config.clone());
    let mut map: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    let mut copy = |pa: &mut Arena, id: ExprId| {
        crate::base::compact::transfer_subtree(arena, pa, id, &mut map)
    };
    let body_p = copy(&mut pa, body);
    let var_p = copy(&mut pa, var);
    let lo_p = copy(&mut pa, lo);
    let hi_p = copy(&mut pa, hi);
    let value = match integrate_definite(&mut pa, body_p, var_p, lo_p, hi_p) {
        Ok(v) => v,
        Err(e @ SymplexError::Divergent { .. }) => return Err(e),
        Err(_) => return Ok(None),
    };
    if !walk::free_symbols(&pa, value).is_empty()
        || walk::has_unevaluated(&pa, value)
        || contains_node(&pa, value, |n| matches!(n, ExprNode::DefiniteIntegral(..)))
    {
        return Ok(None);
    }
    Ok(Some((pa, value)))
}

/// Does adaptive quadrature contradict the closed form `value` of `∫ₐᵇ f dx`
/// with confidence?  Only for a parameter-free `f` that compiles, finite
/// real bounds, a real `value` and a quadrature that converges with every
/// segment resolved ([`quadrature_report`]): then a difference beyond
/// `10⁻⁶·max(1, |Q|)` and 100 error estimates is a wrong closed form — an
/// antiderivative that jumps where `f` is continuous, which the FTC
/// splitting does not see when the jump is not at a breakpoint of a node it
/// knows (up to 0.37 `∫₀¹ (x² + 1)³/((x² + 1)⁴ + 32x⁴) dx` was −0.53 for
/// 0.53: a `RootSum` of logarithms whose arguments cross the cut at `x =
/// 0`).  Anything else (no compiled form, a complex value, a quadrature
/// that does not converge or leaves a segment unresolved next to a
/// singularity, a `DiracDelta`, which samples as 0) says nothing.  Before
/// 0.44 the quadrature's smoothing retry cut the integrand off next to a
/// nonzero end and its estimate did not see it, so correct closed forms
/// were refused: `∫₀¹ (1 − x)^(−9/10) dx = 10` (the quadrature said
/// `9.77`), `∫₀¹ (1 − x)^(−99/100) dx = 100` (`31.5`), `∫₁² (x −
/// 1)^(−9/10) dx = 10`.
fn contradicted_by_quadrature(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    a: ExprId,
    b: ExprId,
    value: Option<num_complex::Complex64>,
) -> bool {
    if walk::free_symbols(arena, f).iter().any(|&s| s != x)
        || contains_node(arena, f, |n| matches!(n, ExprNode::DiracDelta(..)))
    {
        return false;
    }
    // The value as `evalf_f64` reads it: real when the imaginary part is
    // below the 16th digit of the real part.  (This check can only refuse
    // a closed form, so an `f64` reading is enough.)
    let Some(v) = value
        .filter(|z| z.im == 0.0 || z.im.abs() <= 1e-16 * z.re.abs())
        .map(|z| z.re)
    else {
        return false;
    };
    let (Ok(lo), Ok(hi)) = (evalf::evalf_f64(arena, a), evalf::evalf_f64(arena, b)) else {
        return false;
    };
    if !(lo.is_finite() && hi.is_finite() && v.is_finite()) {
        return false;
    }
    let ExprNode::Symbol(sid) = *arena.node(x) else {
        return false;
    };
    let name = arena.symbol_name(sid).to_string();
    let Ok(func) = crate::output::lambdify::compile(arena, f, &[&name]) else {
        return false;
    };
    let peak = std::cell::Cell::new(0.0f64);
    let g = |t: f64| {
        let v = func(&[t]);
        peak.set(peak.get().max(v.abs()));
        v
    };
    // Pieces between the integrand's breakpoints (kinks of `|·|`, poles,
    // …): over a whole range the error estimate missed the kinks of
    // `|x² − 3x|` on `[−15, 15]` and the right value 2259 was refused (0.46,
    // SymPy's test_integrals).
    let (a_ord, b_ord, ends) = if lo <= hi {
        (lo, hi, Interval::closed(a, b))
    } else {
        (hi, lo, Interval::closed(b, a))
    };
    if f64_narrow(a_ord, b_ord) {
        // A range of few `f64` values (`[10²⁰, 10²⁰ + 4]` is one): no
        // `f64` quadrature to compare with.
        return false;
    }
    let points = match quadrature_breakpoints(
        arena,
        f,
        x,
        Interval::closed(a_ord, b_ord),
        Some(ends),
        None,
    ) {
        Ok(points) => points,
        // A proof that the integral diverges contradicts every value.
        Err(SymplexError::Divergent { .. }) => return true,
        Err(_) => return false,
    };
    let mut cuts: Vec<f64> = vec![a_ord];
    cuts.extend(points.into_iter().map(|(_, v)| v));
    cuts.push(b_ord);
    let mut q = QuadResult {
        value: 0.0,
        error: 0.0,
    };
    for w in cuts.windows(2) {
        let Ok((part, true)) = quadrature_report(&g, w[0], w[1], &QuadOpts::default()) else {
            return false;
        };
        q.value += part.value;
        q.error += part.error;
    }
    if lo > hi {
        q.value = -q.value;
    }
    if !(q.value.is_finite() && q.error.is_finite()) {
        return false;
    }
    // The quadrature ran between the bounds rounded to `f64`: up to 0.49
    // it said 0 for `[10²⁰, 10²⁰ + 4]`, one `f64`.
    let moved = bound_rounding(arena, a, lo) + bound_rounding(arena, b, hi);
    let tol = (1e-6 * q.value.abs().max(1.0)).max(100.0 * q.error) + 2.0 * moved * peak.get();
    // Only a refusal rests on this comparison.
    let contradicted = (v - q.value).abs() > tol;
    if contradicted {
        tracing::debug!(
            closed_form = v,
            quadrature = q.value,
            error = q.error,
            "integrate_definite: closed form contradicted by quadrature"
        );
    }
    contradicted
}

/// An upper bound on `|c − v|` for the real constant `c` whose value
/// rounded to the nearest `f64` is `v`: 0 for a rational that `v`
/// represents, their difference for another rational, an ulp of `v`
/// otherwise (and 0 for `±∞` itself).
pub(crate) fn bound_rounding(arena: &Arena, c: ExprId, v: f64) -> f64 {
    if v.is_infinite() && is_infinite(arena, c) {
        return 0.0;
    }
    if !v.is_finite() {
        return f64::INFINITY;
    }
    if let Some(r) = arena.as_num(c)
        && let Some(rv) = crate::base::numeric::f64_to_ratio_exact(v)
    {
        let d = (r - rv).abs();
        return crate::base::numeric::ratio_to_f64(&d)
            .map_or(f64::INFINITY, |d| d * (1.0 + 4.0 * f64::EPSILON));
    }
    (v.abs() * f64::EPSILON).max(f64::from_bits(1))
}

/// `eval`, except that expressions containing a `Piecewise` are left
/// untouched: `eval` selects the first branch whose condition is literally
/// `True` even when earlier branches are undecided, which would collapse
/// `Piecewise((x², x < 1), (2 − x, True))` to `2 − x`.
fn safe_eval(arena: &mut Arena, f: ExprId) -> ExprId {
    if contains_node(arena, f, |n| matches!(n, ExprNode::Piecewise(_))) {
        return f;
    }
    eval::eval(arena, f)
}

/// Does any node of the tree satisfy `pred`?
fn contains_node(arena: &Arena, root: ExprId, pred: impl Fn(&ExprNode) -> bool) -> bool {
    let mut stack = vec![root];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let node = arena.node(id);
        if pred(node) {
            return true;
        }
        node.for_each_child(|c| stack.push(c));
    }
    false
}

/// Error helper.
fn failed(reason: impl Into<String>) -> SymplexError {
    SymplexError::computation_failed("integrate_definite", reason)
}

/// Error helper.
fn divergent(reason: impl Into<String>) -> SymplexError {
    SymplexError::Divergent {
        operation: "integrate_definite",
        reason: reason.into(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Orientation and the top-level driver
// ═══════════════════════════════════════════════════════════════════════════

/// `∫ₐᵇ f dx` with arbitrary bound order.
fn integrate_range(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    a: ExprId,
    b: ExprId,
    depth: usize,
) -> Result<ExprId, SymplexError> {
    if depth > MAX_DEPTH {
        return Err(failed("interval splitting exceeded the maximum depth"));
    }
    if a == b {
        return Ok(arena.zero());
    }
    match cmp_points(arena, a, b) {
        Some(Ordering::Equal) => return Ok(arena.zero()),
        Some(Ordering::Greater) => {
            let v = integrate_ordered(arena, f, x, b, a, depth)?;
            return Ok(arena.neg(v));
        }
        _ => {}
    }
    integrate_ordered(arena, f, x, a, b, depth)
}

/// `∫ₐᵇ f dx` assuming `a < b` (numerically, or by convention when the
/// bounds are symbolic and incomparable).
fn integrate_ordered(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    a: ExprId,
    b: ExprId,
    depth: usize,
) -> Result<ExprId, SymplexError> {
    let f = safe_eval(arena, f);
    if arena.is_zero_structural(f) {
        return Ok(arena.zero());
    }

    // ── Constant integrand ──────────────────────────────────────────
    if !walk::contains(arena, f, x) {
        return constant_over(arena, f, a, b);
    }

    // ── Special integrands (δ, H, |·|, sign, Piecewise, floor) ─────
    if let Some(r) = try_special_integrand(arena, f, x, a, b, depth) {
        return r;
    }

    // ── Known-value table on the whole interval ─────────────────────
    if let Some(v) = table_lookup(arena, f, x, a, b) {
        return Ok(v);
    }

    // ── Rational × trigonometric over ℝ by residues ─────────────────
    if let Some(v) = rational_trig_by_residues(arena, f, x, a, b) {
        return Ok(v);
    }
    // ── ln(x) × rational over [0, ∞) by the keyhole contour ──────────
    if let Some(v) = rational_log_by_keyhole(arena, f, x, a, b) {
        return Ok(v);
    }

    // ── Breakpoint analysis → pieces ────────────────────────────────
    let pieces = split_at_breakpoints(arena, f, x, a, b)?;
    let result = integrate_pieces(arena, f, x, &pieces, depth);
    if matches!(result, Err(SymplexError::ComputationFailed { .. }))
        && let Some(v) = log_rational_tail_inverted(arena, f, x, a, b, depth)
    {
        return Ok(v);
    }
    result
}

/// `f = c·ln(x)·R(x)` with `R` rational over ℚ: `(c, R)` (`c` the
/// `x`-free factors).
fn log_times_rational(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
) -> Option<(Vec<ExprId>, residue::RationalQ)> {
    if !contains_node(arena, f, |n| matches!(n, ExprNode::Ln(_))) {
        return None;
    }
    let (coeff, dep) = factorize(arena, f, x);
    let mut found_log = false;
    let mut rest: Vec<ExprId> = Vec::new();
    for d in dep {
        if matches!(arena.node(d), ExprNode::Ln(arg) if *arg == x) {
            if found_log {
                return None;
            }
            found_log = true;
        } else {
            rest.push(d);
        }
    }
    if !found_log {
        return None;
    }
    let rest = if rest.is_empty() {
        arena.one()
    } else {
        arena.mul(&rest)
    };
    let r = residue::rational_over_q(arena, rest, x)?;
    Some((coeff, r))
}

/// `∫₀^∞ R(x)·ln x dx = −½ Σ Res[R(z)·ln²(−z)]` over all poles of a
/// rational `R` over ℚ that is finite on `[0, ∞)` with `deg D ≥ deg N +
/// 2` (keyhole contour around `[0, ∞)`: on its two lips `ln(−z) = ln x ∓
/// iπ`, and `(ln x − iπ)² − (ln x + iπ)² = −4iπ ln x`).  Shown as the real
/// part when that is exact.  Before 0.46 only `ln x/(c x² + k)` was in the
/// table; `∫₀^∞ ln x/(x² + x + 1) dx` (`0`) and `∫₀^∞ ln x/((x + 1)(x + 2))
/// dx` (`ln² 2/2`) were refused, the antiderivatives having dilogarithms
/// whose limits at `∞` were unknown.  SymPy:
/// `integrate(log(x)/((x + 1)*(x + 2)), (x, 0, oo))` = `log(2)**2/2`.
fn rational_log_by_keyhole(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    a: ExprId,
    b: ExprId,
) -> Option<ExprId> {
    if classify_interval(arena, a, b) != IntervalShape::ZeroInf {
        return None;
    }
    let (coeff, r) = log_times_rational(arena, f, x)?;
    if r.degree_gap() < 2 || r.denom.coeff(0).is_zero() {
        return None;
    }
    let bound = crate::poly::sturm::cauchy_bound(&r.denom);
    let (_, sqf) = r.denom.sqf_list();
    if sqf.iter().any(|(p, _)| {
        p.degree().is_some_and(|d| d > 0)
            && crate::poly::sturm::SturmChain::new(p).count_roots_in_closed(&Q::zero(), &bound) > 0
    }) {
        return None;
    }
    let neg_x = arena.neg(x);
    let ln = arena.ln(neg_x);
    let two = arena.int(2);
    let weight = arena.pow(ln, two);
    let sum = residue::rational_residue_sum(arena, &r, weight, x, |_| true)?;
    let half = arena.rational(-1, 2);
    let raw = arena.mul(&[half, sum]);
    let raw = safe_eval(arena, raw);
    let parts = crate::base::complex::decompose(arena, raw);
    let mut value = raw;
    if parts.exact {
        let re = safe_eval(arena, parts.re);
        if !contains_node(arena, re, |n| {
            matches!(
                n,
                ExprNode::Re(_) | ExprNode::Im(_) | ExprNode::Conjugate(_) | ExprNode::Arg(_)
            )
        }) {
            value = re;
        }
    }
    Some(with_coeff(arena, &coeff, value))
}

/// `∫_c^∞ ln(x)·R(x) dx = −∫₀^{1/c} ln(u)·R(1/u)/u² du` (`x = 1/u`) for a
/// rational `R` over ℚ and a positive rational `c`, when the direct route
/// failed: the antiderivative's dilogarithms have unknown limits at `∞`,
/// while at `u → 0⁺` they are continuous.  Before 0.46 `∫₁^∞ ln x/(x² + 4)
/// dx` was refused.
fn log_rational_tail_inverted(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    a: ExprId,
    b: ExprId,
    depth: usize,
) -> Option<ExprId> {
    if ValueClass::special_atom(arena, b) != Some(ValueClass::PosInf) {
        return None;
    }
    let c = arena.as_num(a)?.clone();
    if !c.is_positive() {
        return None;
    }
    let (coeff, r) = log_times_rational(arena, f, x)?;
    let gap = r.degree_gap();
    if !(2..=64).contains(&gap) {
        return None;
    }
    // R(1/u)/u² = u^{gap−2}·Ñ(u)/D̃(u) with the reversed polynomials.
    let reversed = |p: &crate::poly::Poly| {
        let mut c = p.coeffs().to_vec();
        c.reverse();
        crate::poly::Poly::from_coeffs(c)
    };
    let u = fresh_symbol(arena, "definv", depth);
    let n_rev = crate::poly::polybridge::poly_to_expr(arena, &reversed(&r.numer), u);
    let d_rev = crate::poly::polybridge::poly_to_expr(arena, &reversed(&r.denom), u);
    let power = arena.int(gap - 2);
    let u_pow = arena.pow(u, power);
    let ln_u = arena.ln(u);
    let ratio = arena.div(n_rev, d_rev);
    let mut factors = coeff;
    factors.extend([arena.neg_one(), ln_u, u_pow, ratio]);
    let g = arena.mul(&factors);
    let g = safe_eval(arena, g);
    let zero = arena.zero();
    let hi = arena.num_ratio(Q::one() / c);
    let v = integrate_range(arena, g, u, zero, hi, depth + 1).ok()?;
    (!walk::contains(arena, v, u)).then_some(v)
}

/// `∫_{−∞}^{∞} R(x)·cos(kx) dx` and `∫_{−∞}^{∞} R(x)·sin(kx) dx` for a
/// rational `R` over ℚ and a real `k ≠ 0` of known sign (a number or a
/// parameter declared positive or negative), by residues
/// ([`residue::fourier_rational`]): with `J± = ∫ R(x) e^{±i|k|x} dx`
/// (Jordan's lemma, closing in the upper and the lower half-plane),
/// `∫ R cos = (J₊ + J₋)/2` and `∫ R sin = (J₊ − J₋)/(2i)`, shown as the real
/// and imaginary parts of `J₊` when those can be written without `re`/`im`
/// nodes.  Also `∫₀^∞` of an even `R·cos` or an odd `R·sin` (half the
/// line), and sums of such terms.
///
/// The integral converges (conditionally for `deg D = deg N + 1`) when
/// `deg D ≥ deg N + 1` and `D` has no real roots; `sin(kx)·R(x)` may have a
/// simple pole of `R` at `x = 0`, where the integrand is bounded and the
/// half residue of the principal value is its value (`∫ sin x/(x(x² + 1))
/// dx = π(1 − e⁻¹)`).  Anything else (`None`) is left to the other routes,
/// which also prove divergence.
///
/// Before 0.46 these were refused unless the half-line table matched: the
/// antiderivatives have `Si`/`Ci` at the complex poles, whose limits at
/// `±∞` the limit engine did not know (`∫_{−∞}^{∞} cos 2x/(x² + 2) dx`,
/// `∫ cos x/(x⁴ + 1) dx`, `∫ x sin 3x/(x² + 4x + 13) dx`).  SymPy evaluates
/// them with Meijer G-functions.
fn rational_trig_by_residues(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    a: ExprId,
    b: ExprId,
) -> Option<ExprId> {
    let half_line = match classify_interval(arena, a, b) {
        IntervalShape::FullLine => false,
        IntervalShape::ZeroInf => true,
        _ => return None,
    };
    if !contains_node(arena, f, |n| {
        matches!(n, ExprNode::Sin(_) | ExprNode::Cos(_))
    }) || walk::post_order_ids(arena, f).len() > MAX_RESIDUE_INTEGRAND_NODES
    {
        return None;
    }
    let f = expand::expand(arena, f);
    let f = safe_eval(arena, f);
    let terms: Vec<ExprId> = match arena.node(f) {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![f],
    };
    if terms.len() > 16 {
        return None;
    }
    let mut values = Vec::with_capacity(terms.len());
    for t in terms {
        values.push(rational_trig_term(arena, t, x, half_line)?);
    }
    let s = arena.add(&values);
    Some(safe_eval(arena, s))
}

/// Largest integrand (distinct nodes) that [`rational_trig_by_residues`]
/// expands.
const MAX_RESIDUE_INTEGRAND_NODES: usize = 400;

/// One term `c·R(x)·trig(kx)` of [`rational_trig_by_residues`].
fn rational_trig_term(arena: &mut Arena, t: ExprId, x: ExprId, half_line: bool) -> Option<ExprId> {
    let (coeff, dep) = factorize(arena, t, x);
    let mut trig: Option<(Trig, ExprId)> = None;
    let mut rest: Vec<ExprId> = Vec::new();
    for d in dep {
        match arena.node(d) {
            ExprNode::Sin(arg) | ExprNode::Cos(arg) => {
                if trig.is_some() {
                    return None;
                }
                let kind = if matches!(arena.node(d), ExprNode::Sin(_)) {
                    Trig::Sin
                } else {
                    Trig::Cos
                };
                trig = Some((kind, *arg));
            }
            _ => rest.push(d),
        }
    }
    let (kind, arg) = trig?;
    let k = pure_linear_coeff(arena, arg, x)?;
    if !is_real(arena, k) {
        return None;
    }
    let negative = match sign_of(arena, k)? {
        Ordering::Greater => false,
        Ordering::Less => true,
        Ordering::Equal => return None,
    };
    let k_abs = if negative { arena.neg(k) } else { k };
    let k_abs = safe_eval(arena, k_abs);
    let rest = if rest.is_empty() {
        arena.one()
    } else {
        arena.mul(&rest)
    };
    let r = residue::rational_over_q(arena, rest, x)?;
    if r.degree_gap() < 1 {
        return None;
    }
    // Real poles: none for cos; at most a simple one at 0 for sin.
    let zero_mult = r.denom.coeffs().iter().take_while(|c| c.is_zero()).count();
    let allowed = if kind == Trig::Sin { 1 } else { 0 };
    if zero_mult > allowed {
        return None;
    }
    let shifted = crate::poly::Poly::from_coeffs(r.denom.coeffs()[zero_mult..].to_vec());
    let (_, sqf) = shifted.sqf_list();
    if sqf.iter().any(|(p, _)| {
        p.degree().is_some_and(|d| d > 0)
            && crate::poly::sturm::SturmChain::new(p).count_real_roots() > 0
    }) {
        return None;
    }
    if half_line {
        // ∫₀^∞ = ½ ∫_{−∞}^{∞} for an even integrand.
        let want = if kind == Trig::Cos {
            residue::Parity::Even
        } else {
            residue::Parity::Odd
        };
        if r.parity()? != want {
            return None;
        }
    }
    let neg_k = arena.neg(k_abs);
    let neg_k = safe_eval(arena, neg_k);
    let j_plus = residue::fourier_rational(arena, &r, k_abs, SignSet::POSITIVE, x)?;
    let j_minus = residue::fourier_rational(arena, &r, neg_k, SignSet::NEGATIVE, x)?;
    let combined = match kind {
        Trig::Cos => {
            let s = arena.add(&[j_plus, j_minus]);
            let half = arena.rational(1, 2);
            arena.mul(&[half, s])
        }
        Trig::Sin => {
            let d = arena.sub(j_plus, j_minus);
            let i = arena.i_unit();
            let two = arena.int(2);
            let two_i = arena.mul(&[two, i]);
            arena.div(d, two_i)
        }
    };
    let combined = safe_eval(arena, combined);
    // N/D has rational coefficients and k is real, so J₋ = conj(J₊): the
    // value is re J₊ (cos) or im J₊ (sin), written out when exact.
    let parts = crate::base::complex::decompose(arena, j_plus);
    let mut value = combined;
    if parts.exact {
        let part = match kind {
            Trig::Cos => parts.re,
            Trig::Sin => parts.im,
        };
        let part = safe_eval(arena, part);
        if !contains_node(arena, part, |n| {
            matches!(
                n,
                ExprNode::Re(_) | ExprNode::Im(_) | ExprNode::Conjugate(_) | ExprNode::Arg(_)
            )
        }) {
            value = part;
        }
    }
    if negative && kind == Trig::Sin {
        value = arena.neg(value);
    }
    if half_line {
        let half = arena.rational(1, 2);
        value = arena.mul(&[half, value]);
    }
    Some(with_coeff(arena, &coeff, value))
}

/// Integrate a constant `c` over `[a, b]`.
fn constant_over(
    arena: &mut Arena,
    c: ExprId,
    a: ExprId,
    b: ExprId,
) -> Result<ExprId, SymplexError> {
    let a_inf = is_infinite(arena, a);
    let b_inf = is_infinite(arena, b);
    if a_inf || b_inf {
        return match sign_of(arena, c) {
            Some(Ordering::Equal) => Ok(arena.zero()),
            Some(_) => Err(divergent(
                "non-zero constant integrand over an infinite interval",
            )),
            None => Err(failed(
                "constant integrand of unknown sign over an infinite interval",
            )),
        };
    }
    let width = arena.sub(b, a);
    let v = arena.mul(&[c, width]);
    Ok(safe_eval(arena, v))
}

/// Sum the integrals over consecutive pieces.  A proven divergence on any
/// piece dominates a mere failure elsewhere.
fn integrate_pieces(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    pieces: &[Piece],
    depth: usize,
) -> Result<ExprId, SymplexError> {
    let mut total: Vec<ExprId> = Vec::with_capacity(pieces.len());
    let mut first_failure: Option<SymplexError> = None;
    for p in pieces {
        match integrate_piece(arena, f, x, p, depth) {
            Ok(v) => total.push(v),
            Err(e @ SymplexError::Divergent { .. }) => return Err(e),
            Err(e) => {
                if first_failure.is_none() {
                    first_failure = Some(e);
                }
            }
        }
    }
    if let Some(e) = first_failure {
        return Err(e);
    }
    let s = arena.add(&total);
    Ok(safe_eval(arena, s))
}

// ═══════════════════════════════════════════════════════════════════════════
// Point comparison, signs, finiteness
// ═══════════════════════════════════════════════════════════════════════════

/// Is `id` `+∞` or `−∞`?
fn is_infinite(arena: &Arena, id: ExprId) -> bool {
    real_infinity(arena, id).is_some()
}

/// The direction of `id` when it is the atom `+∞` or `−∞`.
fn real_infinity(arena: &Arena, id: ExprId) -> Option<Sign> {
    ValueClass::special_atom(arena, id).and_then(ValueClass::real_infinity)
}

/// Numeric value of a constant expression (±∞ allowed), or `None`.
fn point_value(arena: &mut Arena, id: ExprId) -> Option<f64> {
    match real_infinity(arena, id) {
        Some(Sign::Positive) => return Some(f64::INFINITY),
        Some(Sign::Negative) => return Some(f64::NEG_INFINITY),
        None => {}
    }
    if !walk::free_symbols(arena, id).is_empty() {
        return None;
    }
    if has_bad_atom(arena, id) {
        return None;
    }
    calculus_util::expr_to_f64(arena, id).filter(|v| v.is_finite())
}

/// Compare two points on the extended real line.
///
/// Exact for rationals and infinities; for other constants from their
/// 16-digit values when these are far enough apart to show the order
/// ([`separated`]), otherwise by the certified sign of the difference
/// ([`sign_checked`](crate::poly::algebraic::sign_checked)); assumption-based
/// for symbolic differences.  `None` when the order cannot be established.
///
/// Up to 0.49 two values within `10⁻¹³` of each other (relatively, of 1
/// below 1) were "equal": a zero of `sin x` less than 100 from an end at
/// `10¹⁵` was taken for the end, and `∫_{10¹⁵}^{10¹⁵+4} |sin x| dx` came out
/// as `∫ sin x dx = 1.4982` (truly `2.4718`).
fn cmp_points(arena: &mut Arena, p: ExprId, q: ExprId) -> Option<Ordering> {
    if p == q {
        return Some(Ordering::Equal);
    }
    if let (Some(rp), Some(rq)) = (arena.as_num(p), arena.as_num(q)) {
        return Some(rp.cmp(rq));
    }
    let pv = point_value(arena, p);
    let qv = point_value(arena, q);
    cmp_values(arena, (p, pv), (q, qv))
}

/// The relative gap beyond which two values known to
/// [`POINT_VALUE_ACCURACY`] (`2⁻⁴⁶`) are certainly in the order of the
/// points they stand for: `2⁻⁴⁰`, 32 times their joint error.
const SEPARATION: f64 = 64.0 * POINT_VALUE_ACCURACY;

/// The order of two points from their values (each within
/// [`POINT_VALUE_ACCURACY`] of its point), when the values are nonzero,
/// finite and further apart than [`SEPARATION`] relatively; `None` when the
/// values cannot show it.
fn separated(pv: f64, qv: f64) -> Option<Ordering> {
    // A value 0 shows no relative gap: the certified sign decides then.
    if pv == 0.0 || qv == 0.0 || !pv.is_finite() || !qv.is_finite() {
        return None;
    }
    if (pv - qv).abs() > SEPARATION * pv.abs().max(qv.abs()) {
        pv.partial_cmp(&qv)
    } else {
        None
    }
}

/// [`cmp_points`] for two points with their values ([`point_value`], or a
/// [`Breakpoint::value`]) known.
fn cmp_values(
    arena: &mut Arena,
    (p, pv): (ExprId, Option<f64>),
    (q, qv): (ExprId, Option<f64>),
) -> Option<Ordering> {
    if p == q {
        return Some(Ordering::Equal);
    }
    if let (Some(rp), Some(rq)) = (arena.as_num(p), arena.as_num(q)) {
        return Some(rp.cmp(rq));
    }
    if let (Some(pv), Some(qv)) = (pv, qv) {
        if pv.is_infinite() || qv.is_infinite() {
            return if pv == qv {
                Some(Ordering::Equal)
            } else {
                pv.partial_cmp(&qv)
            };
        }
        if let Some(order) = separated(pv, qv) {
            return Some(order);
        }
        // Closer than their values tell (or one of them 0): the certified
        // sign of the difference decides, equality included.
        let d = arena.sub(q, p);
        let d = safe_eval(arena, d);
        return match crate::poly::algebraic::sign_checked(arena, d)? {
            s if s > 0 => Some(Ordering::Less),
            s if s < 0 => Some(Ordering::Greater),
            _ => Some(Ordering::Equal),
        };
    }
    if is_infinite(arena, p) || is_infinite(arena, q) {
        // One infinite, other symbolic-but-finite.
        if real_infinity(arena, p) == Some(Sign::Positive)
            || real_infinity(arena, q) == Some(Sign::Negative)
        {
            return Some(Ordering::Greater);
        }
        return Some(Ordering::Less);
    }
    // Symbolic: sign of q − p via assumptions.
    let d = arena.sub(q, p);
    let d = safe_eval(arena, d);
    match sign_of(arena, d) {
        Some(Ordering::Greater) => Some(Ordering::Less),
        Some(Ordering::Less) => Some(Ordering::Greater),
        Some(Ordering::Equal) => Some(Ordering::Equal),
        None => None,
    }
}

/// Sign of a constant or symbolic expression: `Less` (negative), `Equal`
/// (zero), `Greater` (positive), or `None` if unknown.
fn sign_of(arena: &mut Arena, id: ExprId) -> Option<Ordering> {
    if let Some(r) = arena.as_num(id) {
        return Some(r.cmp(&Ratio::zero()));
    }
    match real_infinity(arena, id) {
        Some(Sign::Positive) => return Some(Ordering::Greater),
        Some(Sign::Negative) => return Some(Ordering::Less),
        None => {}
    }
    if walk::free_symbols(arena, id).is_empty() {
        if has_bad_atom(arena, id) {
            return None;
        }
        let v = calculus_util::expr_to_f64(arena, id)?;
        if !v.is_finite() {
            return None;
        }
        if v != 0.0 {
            // A printed digit of `evalf` is certified: so is its sign.
            return v.partial_cmp(&0.0);
        }
        // 0 to 16 digits, or below the `f64` range, is no proof of 0 (up to
        // 0.49 it was): the certified sign decides, or nothing does.
        return crate::poly::algebraic::sign_checked(arena, id).map(|s| s.cmp(&0));
    }
    let mut cache = AssumptionCache::new();
    if Truth::from(cache.query(arena, id, Props::POSITIVE)).is_true() {
        return Some(Ordering::Greater);
    }
    if Truth::from(cache.query(arena, id, Props::NEGATIVE)).is_true() {
        return Some(Ordering::Less);
    }
    if Truth::from(cache.query(arena, id, Props::ZERO)).is_true() {
        return Some(Ordering::Equal);
    }
    None
}

/// Is `id` provably positive?
fn is_positive(arena: &mut Arena, id: ExprId) -> bool {
    sign_of(arena, id) == Some(Ordering::Greater)
}

/// Is `id` provably real?  An unassumed parameter may be complex
/// (decision D4), so a table entry whose value needs a real coefficient
/// (`∫₀^∞ sin(bx)/x dx = (π/2) sign b` diverges for `Im b ≠ 0`) must ask.
fn is_real(arena: &mut Arena, id: ExprId) -> bool {
    if sign_of(arena, id).is_some() {
        return true;
    }
    let mut cache = AssumptionCache::new();
    Truth::from(cache.query(arena, id, Props::REAL)).is_true()
}

/// Does the tree contain `∞`, `−∞`, `zoo` or `NaN`, or a `ln(0)` /
/// `0⁻ⁿ` sub-term that `eval` leaves in place?
fn has_bad_atom(arena: &Arena, root: ExprId) -> bool {
    let mut stack = vec![root];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Infinity
            | ExprNode::NegInfinity
            | ExprNode::ComplexInfinity
            | ExprNode::NaN => {
                return true;
            }
            ExprNode::Ln(inner) if arena.is_zero_structural(*inner) => return true,
            // 0^e is only a plain finite value for a positive numeric e;
            // 0^{-n} is infinite and 0^{a} with symbolic a depends on the
            // sign of a (it may hide a divergence).
            ExprNode::Pow(base, exp)
                if arena.is_zero_structural(*base)
                    && !arena.as_num(*exp).is_some_and(|r| r.is_positive()) =>
            {
                return true;
            }
            node => node.for_each_child(|c| stack.push(c)),
        }
    }
    false
}

/// Is `v` an acceptable finite value: free of `x`, of infinities, of
/// unevaluated nodes, and (when constant) numerically finite?
fn is_finite_value(arena: &mut Arena, v: ExprId, x: ExprId) -> bool {
    finite_value(arena, v, x).is_some()
}

/// [`is_finite_value`] with the numeric value of a constant that is not a
/// literal number: `None` when `v` is not a finite value, `Some(Some(z))`
/// for such a constant, `Some(None)` otherwise.
fn finite_value(arena: &mut Arena, v: ExprId, x: ExprId) -> Option<Option<num_complex::Complex64>> {
    if walk::contains(arena, v, x) || has_bad_atom(arena, v) || walk::has_unevaluated(arena, v) {
        return None;
    }
    if walk::free_symbols(arena, v).is_empty() && arena.as_num(v).is_none() {
        // Constant: verify numerically (catches Γ(0), atan(zoo)-like junk).
        // A complex value is a value: an antiderivative with `Ci`, `ln` or
        // `Ei` of a negative argument is complex at each end and real only
        // in the difference.  (Before 0.46 the test for a negligible
        // imaginary part looked for `I` in a string printed with `i`, so
        // every complex endpoint value was refused: `∫₀¹ x·sin 2x/(x² − x −
        // 1) dx`, whose antiderivative has `Ci(2x − √5 − 1)`.)
        return evalf::evalf_complex64(arena, v)
            .ok()
            .filter(|z| z.re.is_finite() && z.im.is_finite())
            .map(Some);
    }
    Some(None)
}

/// Is the parameter-free closed form `value` of `∫ₐᵇ f dx` non-real while
/// `f` is real on the interval?  The integrand is sampled (compiled, at
/// the guard grid of [`guard_grid`]): when every sample is a finite real
/// number, the integral is real, and a value with an imaginary part beyond
/// `10⁻¹⁰·max(1, |value|)` comes from an antiderivative that is not
/// continuous along the path.  The quadrature check
/// ([`contradicted_by_quadrature`]) only sees real values.
fn non_real_for_real_integrand(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    a: ExprId,
    b: ExprId,
    value: Option<num_complex::Complex64>,
) -> bool {
    if walk::free_symbols(arena, f).iter().any(|&s| s != x) {
        return false;
    }
    let Some(z) = value else {
        return false;
    };
    // Only a refusal rests on it (a value clearly off the real axis).
    if !z.im.is_finite() || z.im.abs() <= 1e-10 * z.norm().max(1.0) {
        return false;
    }
    let (Some(lo), Some(hi)) = (point_value(arena, a), point_value(arena, b)) else {
        return false;
    };
    let ExprNode::Symbol(sid) = *arena.node(x) else {
        return false;
    };
    let name = arena.symbol_name(sid).to_string();
    let Ok(func) = crate::output::lambdify::compile(arena, f, &[&name]) else {
        return false;
    };
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    let samples: Vec<f64> = guard_grid(lo, hi).into_iter().step_by(8).collect();
    !samples.is_empty() && samples.iter().all(|&t| func(&[t]).is_finite())
}

/// Fresh dummy symbol.
fn fresh_symbol(arena: &mut Arena, base: &str, n: usize) -> ExprId {
    arena.symbol(&format!("__{base}{n}"))
}

// ═══════════════════════════════════════════════════════════════════════════
// Pieces and breakpoints
// ═══════════════════════════════════════════════════════════════════════════

/// A sub-interval together with flags marking endpoints where the
/// integrand (or antiderivative) is singular or infinite.
#[derive(Clone, Copy, Debug)]
struct Piece {
    lo: ExprId,
    hi: ExprId,
    lo_sing: bool,
    hi_sing: bool,
}

/// Where a point lies relative to `[a, b]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Position {
    Below,
    AtLower,
    Inside,
    AtUpper,
    Above,
    Unknown,
}

/// Locate `c` relative to `[a, b]`.
///
/// The bounds are assumed ordered `a < b` only when that order is
/// provable; for incomparable symbolic bounds a point is `Below`/`Above`
/// only if it is on the same side of *both* bounds.
fn position_in(arena: &mut Arena, c: ExprId, a: ExprId, b: ExprId) -> Position {
    let ca = cmp_points(arena, c, a);
    let cb = cmp_points(arena, c, b);
    match (ca, cb) {
        (Some(Ordering::Equal), _) => Position::AtLower,
        (_, Some(Ordering::Equal)) => Position::AtUpper,
        (Some(Ordering::Less), Some(Ordering::Less)) => Position::Below,
        (Some(Ordering::Greater), Some(Ordering::Greater)) => Position::Above,
        (Some(Ordering::Greater), Some(Ordering::Less)) => Position::Inside,
        // c < a and c > b: only possible when b < a, i.e. inside [b, a].
        (Some(Ordering::Less), Some(Ordering::Greater)) => Position::Inside,
        (Some(Ordering::Less), None) => {
            if cmp_points(arena, a, b) == Some(Ordering::Less) {
                Position::Below
            } else {
                Position::Unknown
            }
        }
        (None, Some(Ordering::Greater)) => {
            if cmp_points(arena, a, b) == Some(Ordering::Less) {
                Position::Above
            } else {
                Position::Unknown
            }
        }
        _ => Position::Unknown,
    }
}

/// Numeric `[a, b]` of the integration interval when both bounds are
/// constants.  The endpoints keep the integral's orientation (`lower` is
/// the value of `a`, `upper` of `b`) and may be infinite.
fn numeric_range(arena: &mut Arena, a: ExprId, b: ExprId) -> Option<Interval<f64>> {
    let av = point_value(arena, a)?;
    let bv = point_value(arena, b)?;
    Some(Interval::closed(av, bv))
}

/// The range of a breakpoint scan of `[a, b]` with the numeric `range` of
/// [`numeric_range`]: the exact bounds, when both are finite (periodic
/// families are only enumerated over a finite range).
fn scan_range(a: ExprId, b: ExprId, range: Option<Interval<f64>>) -> Option<ScanRange> {
    ScanRange::new(a, b, range?)
}

/// Split `[a, b]` at the breakpoints of `f`, marking singular endpoints.
fn split_at_breakpoints(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    a: ExprId,
    b: ExprId,
) -> Result<Vec<Piece>, SymplexError> {
    // A pole that is not integrable, proved without solving for it (the
    // comparisons and limits at a quartic's root in radicals took minutes).
    if let Some(reason) = pole_divergence(arena, f, x, a, b) {
        return Err(divergent(reason));
    }
    // `R(e^{λx})` with a rate that is not real: its singular points located
    // exactly, before the solver's (which may write `x = 1` as `(½ ln 2 +
    // iπ/4)/(iπ/4 + ln √2)`, a point the comparisons do not place).
    let range = numeric_range(arena, a, b);
    if let Some(exact) = exp_rational_scan(arena, f, x, scan_range(a, b, range)) {
        return build_pieces(arena, &exact, a, b);
    }
    let scan = calculus_util::scan_breakpoints(arena, f, x, scan_range(a, b, range));

    // Opaque nodes (unknown functions, formal integrals, …) cannot be
    // analysed at all — no numeric guard can vouch for them.
    if scan.opaque {
        return Err(failed(
            "the integrand contains a function whose singularities cannot be analysed",
        ));
    }

    // Numeric guard: unexplained sign changes of any denominator / log
    // argument / fractional-power base mean an unlocated singularity.
    match range {
        Some(r) => match numeric_guard(arena, f, x, r.lower, r.upper, &scan) {
            Truth::False => {
                return Err(failed(
                    "a denominator or function argument changes sign inside the interval at a point the solver could not locate",
                ));
            }
            Truth::Unknown if !scan.complete => {
                return Err(failed(
                    "could not determine all singularities of the integrand",
                ));
            }
            _ => {}
        },
        None => {
            if !scan.complete {
                return Err(failed(
                    "could not determine all singularities of the integrand",
                ));
            }
        }
    }

    build_pieces(arena, &scan, a, b)
}

/// Most roots of the denominator in `u` that [`exp_rational_scan`] examines.
const MAX_EXP_POLES: usize = 8;

/// Most members of a lattice of poles on the unit circle (an imaginary rate)
/// that [`exp_rational_scan`] lists in the range.
const MAX_CIRCLE_POLES: usize = 256;

/// The singular points of `f = R(e^{L₁}, …, e^{Lₖ})`, every `Lⱼ = rⱼ·α·x +
/// βⱼ` linear with a rational `rⱼ` and constants `α`, `βⱼ`, for a rate `α`
/// that is not real, where the solver gives up on a denominator such as
/// `e^{(1+i)x} + e^{−(1+i)x} − 3`: with `u = e^{λ·x}` (`λ = g·α`, `g` the
/// gcd of the `rⱼ`) `f` is rational in `u`, and `x` is singular exactly where
/// `u` meets a root `ρ ≠ 0` of its denominator.  For `re λ ≠ 0`, `|u| = |ρ|`
/// only at `x_ρ = ln|ρ|/re λ`, and `u = ρ` there exactly when `θ/2π` is an
/// integer, `θ = im λ·x_ρ − arg ρ` (an exact integer part `n` and the
/// certified sign of `θ/2π − n` decide it).  For an imaginary `λ = iω`, `u`
/// runs around the unit circle: no singular point when `|ρ| ≠ 1` is
/// certified, and for `|ρ| = 1` the lattice `x = (arg ρ + 2πk)/ω` over the
/// exact bounds of the finite `range` (`solve::lattice_indices`; up to 0.57
/// the solver's principal root alone: `∫₋₂⁰ dx/(e^{iπx} + 1)` was `2`, across
/// the pole at `x = −1`).  The scan is complete: these are all the singular
/// points.  `None` when `f` is not of that form, or a root, an integer part
/// or a sign does not settle, or the lattice has no finite range.  Up to
/// 0.55 `∫₀³ dx/(e^{(1+i)x} + e^{−(1+i)x} − 3)` was refused ("could not
/// determine all singularities").  A power `c^L` of a constant `c ≠ 0` is
/// `e^{L·ln c}` (the principal power): up to 0.57 `∫₀² (1+i)ˣ/((1+i)ˣ − 1 − i)
/// dx` (a pole at `x = 1`) and `∫₀³ dx/((1+i)ˣ + (1+i)⁻ˣ − 3)` were refused.
fn exp_rational_scan(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    range: Option<ScanRange>,
) -> Option<BreakScan> {
    use num_integer::Integer;
    let mut exps: Vec<(ExprId, LinearCoeffs<ExprId>)> = Vec::new();
    for id in walk::post_order_ids(arena, f) {
        let (arg, base) = match *arena.node(id) {
            ExprNode::Exp(arg) => (arg, None),
            ExprNode::Pow(c, g) if c != arena.e_const() => (g, Some(c)),
            _ => continue,
        };
        if !walk::contains(arena, arg, x) {
            continue;
        }
        let l = calculus_util::linear_coeffs_exact(arena, arg, x)?;
        if !walk::free_symbols(arena, l.slope).is_empty()
            || !walk::free_symbols(arena, l.intercept).is_empty()
        {
            return None;
        }
        let l = match base {
            None => l,
            Some(c) => {
                if !walk::free_symbols(arena, c).is_empty()
                    || !arena
                        .part_signs(c)
                        .is_some_and(|s| s.re.is_some() || s.im.is_some())
                {
                    return None;
                }
                let ln_c = arena.ln(c);
                let ln_c = eval::eval(arena, ln_c);
                let slope = arena.mul(&[l.slope, ln_c]);
                let intercept = arena.mul(&[l.intercept, ln_c]);
                LinearCoeffs {
                    slope: eval::eval(arena, slope),
                    intercept: eval::eval(arena, intercept),
                }
            }
        };
        exps.push((id, l));
    }
    let alpha = exps.first()?.1.slope;
    let mut ratios: Vec<Q> = Vec::with_capacity(exps.len());
    for (_, l) in &exps {
        ratios.push(crate::transforms::integrate::rational_ratio(
            arena, l.slope, alpha,
        )?);
    }
    let mut num_gcd = BigInt::zero();
    let mut den_lcm = BigInt::one();
    for r in &ratios {
        num_gcd = num_gcd.gcd(r.numer());
        den_lcm = den_lcm.lcm(r.denom());
    }
    if num_gcd.is_zero() {
        return None;
    }
    let g = Q::new(num_gcd, den_lcm);
    let g_id = arena.num_ratio(g.clone());
    let lambda = arena.mul(&[g_id, alpha]);
    let lambda = eval::eval(arena, lambda);
    if !crate::transforms::realness::imaginary_part_sign(arena, lambda)
        .is_nonzero()
        .is_true()
    {
        return None;
    }
    let u = arena.symbol("__exp_pole_u");
    let mut in_u = f;
    for ((node, l), r) in exps.iter().zip(&ratios) {
        let k = arena.num_ratio(r / &g);
        let u_k = arena.pow(u, k);
        let e_beta = arena.exp(l.intercept);
        let repl = arena.mul(&[e_beta, u_k]);
        in_u = arena.subs_structural(in_u, *node, repl);
    }
    if walk::contains(arena, in_u, x) {
        return None;
    }
    let in_u = eval::eval(arena, in_u);
    let in_u = crate::transforms::integrate::clear_nested_fractions(arena, in_u, u);
    let (n, d) = crate::poly::polybridge::as_numer_denom(arena, in_u);
    // Polynomials in `u` over the constants (`u − 1 − i` from `(1+i)ˣ`).
    if !arena.is_zero_structural(n) {
        calculus_util::poly_coeffs_symbolic(arena, n, u)?;
    }
    calculus_util::poly_coeffs_symbolic(arena, d, u)?;
    let mut scan = BreakScan {
        points: Vec::new(),
        complete: true,
        opaque: false,
        crowded: false,
        periodic: false,
    };
    if !walk::contains(arena, d, u) {
        return Some(scan);
    }
    let cx =
        crate::transforms::solve::SolveCx::principal(crate::transforms::solve::Domain::Complex);
    let roots = crate::transforms::solve::solutions(arena, d, u, &cx).points_if_complete()?;
    if roots.len() > MAX_EXP_POLES {
        return None;
    }
    let re_lambda = arena.re(lambda);
    let re_lambda = eval::eval(arena, re_lambda);
    let circle = crate::poly::algebraic::sign_checked(arena, re_lambda)? == 0;
    let im_lambda = arena.im(lambda);
    let im_lambda = eval::eval(arena, im_lambda);
    for rho in roots {
        if arena.is_zero_structural(rho) {
            continue;
        }
        let abs_rho = arena.abs(rho);
        let ln_abs_rho = arena.ln(abs_rho);
        let ln_abs_rho = eval::eval(arena, ln_abs_rho);
        if circle {
            if crate::poly::algebraic::sign_checked(arena, ln_abs_rho)? == 0 {
                // `u = e^{iωx}` meets `ρ` on the unit circle wherever `ωx = arg ρ
                // + 2πk`: every member of the lattice in the range.
                let r = range.as_ref()?;
                let arg_rho = arena.arg(rho);
                let offset = arena.div(arg_rho, im_lambda);
                let offset = eval::eval(arena, offset);
                let two = arena.int(2);
                let pi = arena.pi();
                let two_pi = arena.mul(&[two, pi]);
                let step = arena.div(two_pi, im_lambda);
                let step = eval::eval(arena, step);
                let ks = crate::transforms::solve::lattice_indices(
                    arena,
                    offset,
                    step,
                    r.exact.lower,
                    r.exact.upper,
                    MAX_CIRCLE_POLES,
                )?;
                for k in ks {
                    let k = arena.num_ratio(Q::from_integer(k));
                    let sk = arena.mul(&[step, k]);
                    let at = arena.add(&[offset, sk]);
                    let at = eval::eval(arena, at);
                    let value = evalf::eval_const_f64(arena, at).filter(|v| v.is_finite())?;
                    scan.points.push(Breakpoint {
                        point: at,
                        value: Some(value),
                        kind: BreakKind::Singular,
                    });
                }
            }
            continue;
        }
        let at = arena.div(ln_abs_rho, re_lambda);
        let mut at = eval::eval(arena, at);
        // `ln 2/ln √2` (`(1+i)ˣ = 2i`) is 2 once the logarithms of the
        // positive constants are expanded.
        if arena.as_num(at).is_none() {
            let n = crate::simplify::log_expand::expand_log(arena, ln_abs_rho);
            let d = crate::simplify::log_expand::expand_log(arena, re_lambda);
            let q = arena.div(n, d);
            let q = eval::eval(arena, q);
            if arena.as_num(q).is_some() {
                at = q;
            }
        }
        let phase = arena.mul(&[im_lambda, at]);
        let arg_rho = arena.arg(rho);
        let neg_arg = arena.neg(arg_rho);
        let theta = arena.add(&[phase, neg_arg]);
        let two = arena.int(2);
        let pi = arena.pi();
        let two_pi = arena.mul(&[two, pi]);
        let q = arena.div(theta, two_pi);
        let q = eval::eval(arena, q);
        let n = arena.floor(q);
        let n = eval::eval(arena, n);
        arena.as_num(n).filter(|v| v.is_integer())?;
        let frac = arena.sub(q, n);
        let frac = eval::eval(arena, frac);
        if crate::poly::algebraic::sign_checked(arena, frac)? == 0 {
            let value = evalf::eval_const_f64(arena, at).filter(|v| v.is_finite())?;
            scan.points.push(Breakpoint {
                point: at,
                value: Some(value),
                kind: BreakKind::Singular,
            });
        }
    }
    Some(scan)
}

/// `points` (each with its value, as [`cmp_values`] takes them) in
/// increasing order, equal ones merged; `None` when two of them cannot be
/// ordered.  A bottom-up merge sort: the comparison may fail, and the
/// standard sort wants a total order.
fn sort_points(
    arena: &mut Arena,
    points: Vec<(ExprId, Option<f64>)>,
) -> Option<Vec<(ExprId, Option<f64>)>> {
    let mut runs: Vec<Vec<(ExprId, Option<f64>)>> = points.into_iter().map(|p| vec![p]).collect();
    while runs.len() > 1 {
        let mut next = Vec::with_capacity(runs.len().div_ceil(2));
        let mut pairs = runs.into_iter();
        while let Some(left) = pairs.next() {
            let Some(right) = pairs.next() else {
                next.push(left);
                break;
            };
            let mut merged = Vec::with_capacity(left.len() + right.len());
            let mut l = left.into_iter().peekable();
            let mut r = right.into_iter().peekable();
            loop {
                match (l.peek().copied(), r.peek().copied()) {
                    (Some(p), Some(q)) => match cmp_values(arena, p, q)? {
                        Ordering::Less => {
                            merged.push(p);
                            l.next();
                        }
                        Ordering::Greater => {
                            merged.push(q);
                            r.next();
                        }
                        Ordering::Equal => {
                            merged.push(p);
                            l.next();
                            r.next();
                        }
                    },
                    (Some(_), None) => {
                        merged.extend(l.by_ref());
                        break;
                    }
                    (None, _) => {
                        merged.extend(r.by_ref());
                        break;
                    }
                }
            }
            next.push(merged);
        }
        runs = next;
    }
    Some(runs.pop().unwrap_or_default())
}

/// Is the parametric point `p` non-real for every value the assumptions
/// allow?  `r·acos(g) + c` and `r·asin(g) + c` (`r ≠ 0` rational, `c` a
/// real constant) with `g > 1` or `g < −1` provable: the zeros of `b + 1 +
/// cos x` for `b > 0`, `acos(−b − 1)` and its translates, are off the real
/// line, and `∫₀^{2π} dx/(b + 1 + cos x)` was refused because their place
/// relative to the interval was "undecided".
fn provably_off_the_real_line(arena: &mut Arena, p: ExprId) -> bool {
    let mut core: Option<ExprId> = None;
    let terms: Vec<ExprId> = match arena.node(p) {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![p],
    };
    for term in terms {
        if walk::free_symbols(arena, term).is_empty() {
            if point_value(arena, term).is_none() {
                return false;
            }
            continue;
        }
        if core.is_some() {
            return false;
        }
        let factors = factors_of(arena, term);
        let mut inverse_trig = None;
        for f in factors {
            if let Some(r) = arena.as_num(f) {
                if r.is_zero() {
                    return false;
                }
                continue;
            }
            if inverse_trig.is_some() {
                return false;
            }
            inverse_trig = Some(f);
        }
        core = inverse_trig;
    }
    let Some(core) = core else {
        return false;
    };
    let (ExprNode::Acos(g) | ExprNode::Asin(g)) = *arena.node(core) else {
        return false;
    };
    let one = arena.one();
    let above = arena.sub(g, one);
    let above = safe_eval(arena, above);
    if sign_of(arena, above) == Some(Ordering::Greater) {
        return true;
    }
    let below = arena.add(&[g, one]);
    let below = safe_eval(arena, below);
    sign_of(arena, below) == Some(Ordering::Less)
}

/// Sample grid on `[lo, hi]` (infinite ends mapped through `t/(1−t)`).
fn guard_grid(lo: f64, hi: f64) -> Vec<f64> {
    let n = GUARD_SAMPLES;
    (0..n)
        .map(|i| {
            let t = ((i as f64) + 0.5) / (n as f64);
            match (lo.is_finite(), hi.is_finite()) {
                (true, true) => lo + (hi - lo) * t,
                (true, false) => lo + t / (1.0 - t),
                (false, true) => hi - t / (1.0 - t),
                (false, false) => {
                    let s = 2.0 * t - 1.0;
                    s / (1.0 - s * s)
                }
            }
        })
        .collect()
}

/// Turn a breakpoint scan into ordered pieces of `[a, b]`.
fn build_pieces(
    arena: &mut Arena,
    scan: &BreakScan,
    a: ExprId,
    b: ExprId,
) -> Result<Vec<Piece>, SymplexError> {
    let mut lo_sing = is_infinite(arena, a);
    let mut hi_sing = is_infinite(arena, b);
    let mut interior: Vec<(ExprId, Option<f64>)> = Vec::new();

    for bp in &scan.points {
        if bp.value.is_none() && provably_off_the_real_line(arena, bp.point) {
            continue;
        }
        match position_in(arena, bp.point, a, b) {
            Position::Below | Position::Above => {}
            Position::AtLower => lo_sing = true,
            Position::AtUpper => hi_sing = true,
            Position::Inside => {
                if !interior.iter().any(|(p, _)| *p == bp.point) {
                    interior.push((bp.point, bp.value));
                }
            }
            Position::Unknown => {
                let shown = arena.display(bp.point).to_string();
                return Err(failed(format!(
                    "cannot decide whether the singular point {shown} lies inside the interval"
                )));
            }
        }
    }

    if interior.len() > MAX_PIECES {
        return Err(failed("too many singularities inside the interval"));
    }

    // Order the interior points and merge equal ones.  Up to 0.49 constant
    // points were sorted by their `f64` values and merged within `10⁻¹²`:
    // the zeros of `sin(10²⁰x)` near 1 were all one point.
    let interior =
        sort_points(arena, interior).ok_or_else(|| failed("cannot order the singular points"))?;

    let mut pieces = Vec::with_capacity(interior.len() + 1);
    let mut cur_lo = a;
    let mut cur_lo_sing = lo_sing;
    for (p, _) in &interior {
        pieces.push(Piece {
            lo: cur_lo,
            hi: *p,
            lo_sing: cur_lo_sing,
            hi_sing: true,
        });
        cur_lo = *p;
        cur_lo_sing = true;
    }
    pieces.push(Piece {
        lo: cur_lo,
        hi: b,
        lo_sing: cur_lo_sing,
        hi_sing,
    });
    Ok(pieces)
}

/// An end of the range of [`pole_divergence`], as a rational.
#[derive(Clone, Debug)]
enum RationalEnd {
    /// `±∞`.
    Unbounded,
    /// The end itself, or a rational inside the range next to it.
    At(Q),
}

/// The end `e` of a range as a [`RationalEnd`]: `±∞`, `e` itself when
/// rational, otherwise a rational on the inner side of its value (above it
/// for the `lower` end; see [`inner_rational`]).  `None` when `e` has no
/// real value.
fn rational_end(arena: &mut Arena, e: ExprId, lower: bool) -> Option<RationalEnd> {
    if is_infinite(arena, e) {
        return Some(RationalEnd::Unbounded);
    }
    if let Some(r) = arena.as_num(e) {
        return Some(RationalEnd::At(r.clone()));
    }
    let v = point_value(arena, e)?;
    inner_rational(v, lower).map(RationalEnd::At)
}

/// A rational on the inner side of an end whose value is `v` (within
/// [`POINT_VALUE_ACCURACY`] of it relatively, or rounded to 0 from below the
/// `f64` range): above it for a `lower` end, clear of that uncertainty.
fn inner_rational(v: f64, lower: bool) -> Option<Q> {
    if !v.is_finite() {
        return None;
    }
    let margin = (4.0 * POINT_VALUE_ACCURACY * v.abs()).max((-1000f64).exp2());
    let inner = if lower { v + margin } else { v - margin };
    crate::base::numeric::f64_to_ratio_exact(inner)
}

/// Cauchy's bound `1 + max |aᵢ/aₙ|` on the absolute values of the roots of
/// `p` (degree at least 1).
fn root_bound(p: &crate::poly::Poly) -> Option<Q> {
    let coeffs = p.coeffs();
    let lead = coeffs.last().filter(|c| !c.is_zero())?;
    let mut m = Q::zero();
    for c in &coeffs[..coeffs.len() - 1] {
        let r = (c / lead).abs();
        if r > m {
            m = r;
        }
    }
    Some(m + Q::one())
}

/// Most refinements of a root's isolating cell while the other factors of
/// the integrand are enclosed away from 0 ([`pole_divergence`]); each
/// narrows the cell 256-fold.
const POLE_REFINEMENTS: usize = 8;

/// A proof that `∫ f dx` over `[a, b]` (constant bounds, `a < b`, `±∞`
/// allowed) diverges at a pole, decided exactly and without the pole's
/// closed form; `None` when none is found (see [`pole_divergence_in`]).
///
/// The breakpoint scanner locates a pole by solving its denominator, and
/// the comparison test then takes the limit of `(x − c)·f` at that root
/// `c` — for a quartic, Ferrari's nested radicals, whose zero tests
/// (minimal polynomials of degree in the hundreds) took minutes:
/// `∫₁² dx/(x⁴ − 3x³ − 2x² + 3x + 2)`, which diverges at the root
/// `1.1488…`, ran for more than a minute up to 0.51 on every route
/// (`integrate_definite`, `integrate_numeric`, `eval_decimal`).
fn pole_divergence(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    a: ExprId,
    b: ExprId,
) -> Option<String> {
    if !has_rational_pole_factor(arena, f, x) {
        return None;
    }
    let lower = rational_end(arena, a, true)?;
    let upper = rational_end(arena, b, false)?;
    pole_divergence_in(arena, f, x, &lower, &upper)
}

/// Does `f` have a factor `B^(−k)` with `B` a polynomial in `x` (the cheap
/// test before [`pole_divergence_in`])?
fn has_rational_pole_factor(arena: &Arena, f: ExprId, x: ExprId) -> bool {
    factors_of(arena, f).into_iter().any(|g| {
        matches!(*arena.node(g), ExprNode::Pow(base, e)
            if walk::contains(arena, base, x)
                && arena.as_num(e).is_some_and(|r| r.is_integer() && r.is_negative()))
    })
}

/// [`pole_divergence`] between rational ends (`lower < upper`).
///
/// `f = G·B^(−k)` with `B ∈ ℚ[x]` and an integer `k ≥ 1`, `G` the other
/// factors (no parameter, no kink or step), and `c` a real root of `B` in
/// `(lower, upper]`, isolated exactly in a cell `(l, h]` (Descartes'
/// method, [`SturmChain`](crate::poly::sturm::SturmChain)) where `B` has no
/// other root.  When the values of `G` over `[l, h]` lie in a box that
/// excludes 0 ([`evalf::nonzero_on`], the cell narrowed until they do),
/// some direction `u` has `Re(ū·G) ≥ m > 0` there, while `B^(−k)` is real,
/// of one sign on `(l, c)` and at least `1/(K·|x − c|)` near `c` (`B(c) = 0`,
/// `|B| ≤ K·|x − c|`): `Re(ū·f)` keeps a sign and is not integrable at
/// `c`.  A root where `G` vanishes too (a removable or weaker
/// singularity), or that no cell separates from `G`'s own singularities,
/// proves nothing.
fn pole_divergence_in(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    lower: &RationalEnd,
    upper: &RationalEnd,
) -> Option<String> {
    if walk::free_symbols(arena, f).iter().any(|&s| s != x) || has_kink(arena, f, x) {
        return None;
    }
    let factors = factors_of(arena, f);
    for (i, &factor) in factors.iter().enumerate() {
        let ExprNode::Pow(base, e) = *arena.node(factor) else {
            continue;
        };
        if !arena
            .as_num(e)
            .is_some_and(|r| r.is_integer() && r.is_negative())
        {
            continue;
        }
        let Some(p) = crate::poly::polybridge::expr_to_poly(arena, base, x) else {
            continue;
        };
        if p.degree().is_none_or(|d| d == 0) {
            continue;
        }
        let chain = crate::poly::sturm::SturmChain::new(&p);
        if chain.has_no_real_roots() {
            continue;
        }
        let Some(bound) = root_bound(&p) else {
            continue;
        };
        let lo = match lower {
            RationalEnd::At(q) => q.clone(),
            RationalEnd::Unbounded => -bound.clone(),
        };
        let hi = match upper {
            RationalEnd::At(q) => q.clone(),
            RationalEnd::Unbounded => bound,
        };
        if lo >= hi {
            continue;
        }
        let cells = chain.isolate_roots_in(&lo, &hi, 64);
        if cells.is_empty() {
            continue;
        }
        let rest: Vec<ExprId> = factors
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, &g)| g)
            .collect();
        let g = if rest.is_empty() {
            arena.one()
        } else {
            let g = arena.mul(&rest);
            safe_eval(arena, g)
        };
        for cell in cells {
            let mut cell = cell;
            for _ in 0..POLE_REFINEMENTS {
                // One root in `(l, h]`, `l < h` (a rational root hit by the
                // bisection is left to the comparison test, which is cheap
                // at a rational point).
                if cell.lower >= cell.upper || chain.count_roots_in(&cell.lower, &cell.upper) != 1 {
                    break;
                }
                if evalf::nonzero_on(arena, g, x, &cell.lower, &cell.upper) {
                    return Some(format!(
                        "the integrand is not integrable at the zero of {} in ({}, {}]",
                        arena.display(base),
                        cell.lower,
                        cell.upper
                    ));
                }
                let width = (&cell.upper - &cell.lower) / Q::from_integer(BigInt::from(256));
                cell = chain.refine_interval(&cell, &width);
            }
        }
    }
    None
}

/// Sample every `var`-dependent denominator, logarithm argument and
/// fractional-power base of `f` on `[lo, hi]` and look for sign changes
/// that no known breakpoint explains.
///
/// Returns `True` if all sign changes are explained, `False` if an
/// unexplained one exists, `Unknown` if the check could not run (`f64`
/// samples: only a refusal may rest on `False`, never a claim).
fn numeric_guard(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    lo: f64,
    hi: f64,
    scan: &BreakScan,
) -> Truth {
    let mut watched: Vec<ExprId> = Vec::new();
    let mut stack = vec![f];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match arena.node(id).clone() {
            ExprNode::Pow(base, exp) if walk::contains(arena, base, x) => {
                let matters = arena
                    .as_num(exp)
                    .is_some_and(|r| r.is_negative() || !r.is_integer());
                if matters {
                    watched.push(base);
                }
            }
            ExprNode::Ln(inner) | ExprNode::Tan(inner) if walk::contains(arena, inner, x) => {
                let w = if matches!(arena.node(id), ExprNode::Tan(_)) {
                    arena.cos(inner)
                } else {
                    inner
                };
                watched.push(w);
            }
            _ => {}
        }
        arena.node(id).for_each_child(|c| stack.push(c));
    }
    let known: Vec<f64> = scan.points.iter().filter_map(|bp| bp.value).collect();
    sign_changes_explained(arena, &watched, x, lo, hi, &known)
}

/// Sample each of the `watched` functions of `x` on `[lo, hi]` and look
/// for a sign change that none of the `known` points explains.
///
/// Returns `True` if all sign changes are explained, `False` if an
/// unexplained one exists, `Unknown` if the check could not run (`f64`
/// samples: only a refusal may rest on `False`, never a claim).
fn sign_changes_explained(
    arena: &mut Arena,
    watched: &[ExprId],
    x: ExprId,
    lo: f64,
    hi: f64,
    known: &[f64],
) -> Truth {
    if watched.is_empty() {
        return Truth::True;
    }
    // `f64` samples cannot resolve a range this narrow next to its position
    // (`7t` rounds by about 1 at `t = 10¹⁵`): their sign changes are noise.
    if f64_narrow(lo, hi) {
        return Truth::Unknown;
    }
    let x_name = match arena.node(x) {
        ExprNode::Symbol(sid) => arena.symbol_name(*sid).to_string(),
        _ => return Truth::Unknown,
    };
    let grid = guard_grid(lo, hi);
    let mut all_ok = true;
    for &w in watched {
        let w_eval = safe_eval(arena, w);
        if !walk::free_symbols(arena, w_eval).iter().all(|&s| s == x) {
            return Truth::Unknown; // parametric — cannot sample
        }
        let Ok(func) = crate::output::lambdify::compile(arena, w_eval, &[&x_name]) else {
            return Truth::Unknown;
        };
        // The last sample with a nonzero value: a sample exactly on a zero
        // is skipped, and the sign change through it is judged between its
        // neighbours (up to 0.52 neither pair was compared, so a zero the
        // solver did not locate at a sample point passed the guard).
        let mut prev: Option<(f64, f64)> = None;
        for &t in &grid {
            let v = func(&[t]);
            if !v.is_finite() {
                return Truth::Unknown;
            }
            // A guide only (`f64` samples): the guard's `False` refuses.
            if v == 0.0 {
                continue;
            }
            if let Some((pt, pv)) = prev
                && (pv < 0.0) != (v < 0.0)
            {
                // The known values are within `2⁻⁴⁶` of their points: at
                // `10¹⁵` a zero of `sin` may show up to `0.06` off.
                let slack = 1e-9_f64.max(4.0 * POINT_VALUE_ACCURACY * pt.abs().max(t.abs()));
                let explained = known.iter().any(|&k| k >= pt - slack && k <= t + slack);
                if !explained {
                    all_ok = false;
                }
            }
            prev = Some((t, v));
        }
    }
    Truth::from_bool(all_ok)
}

// ═══════════════════════════════════════════════════════════════════════════
// Single smooth piece
// ═══════════════════════════════════════════════════════════════════════════

/// Integrate `f` over one piece whose interior is free of breakpoints.
fn integrate_piece(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    p: &Piece,
    depth: usize,
) -> Result<ExprId, SymplexError> {
    if depth > MAX_DEPTH {
        return Err(failed("interval splitting exceeded the maximum depth"));
    }
    let lo_is_neg_inf = real_infinity(arena, p.lo) == Some(Sign::Negative);
    let hi_is_pos_inf = real_infinity(arena, p.hi) == Some(Sign::Positive);

    // ── (−∞, ∞): parity, then split at 0 ────────────────────────────
    if lo_is_neg_inf && hi_is_pos_inf {
        let zero = arena.zero();
        let inf = arena.infinity();
        match parity(arena, f, x) {
            Some(Parity::Even) => {
                let half = integrate_range(arena, f, x, zero, inf, depth + 1)?;
                let two = arena.int(2);
                let v = arena.mul(&[two, half]);
                return Ok(safe_eval(arena, v));
            }
            Some(Parity::Odd) => {
                // Each half must converge on its own.
                integrate_range(arena, f, x, zero, inf, depth + 1)?;
                return Ok(arena.zero());
            }
            None => {}
        }
        let left = Piece {
            lo: p.lo,
            hi: zero,
            lo_sing: true,
            hi_sing: false,
        };
        let right = Piece {
            lo: zero,
            hi: inf,
            lo_sing: false,
            hi_sing: true,
        };
        return integrate_pieces(arena, f, x, &[left, right], depth + 1);
    }

    // ── Symmetric finite interval [−c, c] ───────────────────────────
    if !lo_is_neg_inf && !hi_is_pos_inf && is_negation_of(arena, p.lo, p.hi) {
        let zero = arena.zero();
        match parity(arena, f, x) {
            Some(Parity::Even) => {
                let half = Piece {
                    lo: zero,
                    hi: p.hi,
                    lo_sing: false,
                    hi_sing: p.hi_sing,
                };
                let v = integrate_piece(arena, f, x, &half, depth + 1)?;
                let two = arena.int(2);
                let v = arena.mul(&[two, v]);
                return Ok(safe_eval(arena, v));
            }
            Some(Parity::Odd) => {
                if !p.lo_sing && !p.hi_sing {
                    // Continuous odd function on a symmetric interval.
                    return Ok(arena.zero());
                }
                let half = Piece {
                    lo: zero,
                    hi: p.hi,
                    lo_sing: false,
                    hi_sing: p.hi_sing,
                };
                integrate_piece(arena, f, x, &half, depth + 1)?;
                return Ok(arena.zero());
            }
            None => {}
        }
    }

    // ── Table on this piece (also via reflection for (−∞, c]) ──────
    if let Some(v) = table_lookup(arena, f, x, p.lo, p.hi) {
        return Ok(v);
    }
    if lo_is_neg_inf {
        // x → −t maps (−∞, c] to [−c, ∞).
        let t = fresh_symbol(arena, "defr", depth);
        let neg_t = arena.neg(t);
        let g = subs::subs(arena, f, x, neg_t);
        let g = safe_eval(arena, g);
        let neg_hi = arena.neg(p.hi);
        let neg_hi = safe_eval(arena, neg_hi);
        let inf = arena.infinity();
        if let Some(v) = table_lookup(arena, g, t, neg_hi, inf) {
            return Ok(v);
        }
    }

    // ── A tail that is not integrable ─────────────────────────────────
    // The comparison test on the integrand before the antiderivative's
    // limits at the infinite end: these can be far harder (`∫₁^∞
    // e^{x/2}/(x³ − x + 1) dx`, whose antiderivative is a sum of `Ei` at the
    // cubic's roots in radicals, took 25 s to diverge up to 0.51; the
    // integrand grows like `e^{x/2}/x³`).
    for (end, side, infinite) in [
        (p.hi, Side::FromLeft, hi_is_pos_inf),
        (p.lo, Side::FromRight, lo_is_neg_inf),
    ] {
        if infinite && diverges_near(arena, f, x, end, side) {
            let shown = arena.display(end).to_string();
            return Err(divergent(format!(
                "the integrand is not integrable as x → {shown}"
            )));
        }
    }

    // ── Antiderivative ──────────────────────────────────────────────
    let anti = crate::transforms::integrate::integrate(arena, f, x);
    let anti = safe_eval(arena, anti);
    if walk::has_unevaluated(arena, anti) {
        let r = fallback_without_antiderivative(arena, f, x, p);
        return divergent_at_singular_end(arena, f, x, p, r);
    }

    // Parametric antiderivatives come wrapped in a Piecewise over the
    // parameter (e.g. ∫ xⁿ: n ≠ −1 vs n = −1).  Conditions that the
    // assumption system can decide are resolved first; the remaining
    // branches are evaluated individually.  A divergence in an undecided
    // branch is reported as "cannot compute" (the answer depends on the
    // parameter), not as a divergence of the whole integral.
    if let ExprNode::Piecewise(pairs) = arena.node(anti).clone()
        && !pairs.iter().any(|&(_, c)| walk::contains(arena, c, x))
    {
        let mut live: Vec<(ExprId, ExprId)> = Vec::new();
        for (val, cond) in pairs.iter() {
            match decide_condition(arena, *cond) {
                Truth::False => continue,
                Truth::True => {
                    live.push((*val, arena.bool_true()));
                    break;
                }
                Truth::Unknown => live.push((*val, *cond)),
            }
        }
        if live.len() == 1 {
            let r = evaluate_antiderivative(arena, f, live[0].0, x, p, depth);
            return divergent_at_singular_end(arena, f, x, p, r);
        }
        let mut out: Vec<(ExprId, ExprId)> = Vec::with_capacity(live.len());
        let mut all_divergent = !live.is_empty();
        for (val, cond) in live {
            match evaluate_antiderivative(arena, f, val, x, p, depth) {
                Ok(v) => {
                    all_divergent = false;
                    out.push((v, cond));
                }
                Err(SymplexError::Divergent { .. }) => {
                    if all_divergent {
                        continue;
                    }
                    return Err(failed(
                        "convergence depends on a parameter (some branches diverge)",
                    ));
                }
                Err(e) => return Err(e),
            }
        }
        if all_divergent {
            return Err(divergent("every parameter branch diverges"));
        }
        if out.len() < 2 {
            return Err(failed(
                "convergence depends on a parameter (some branches diverge)",
            ));
        }
        return Ok(arena.piecewise(&out));
    }

    let r = evaluate_antiderivative(arena, f, anti, x, p, depth);
    divergent_at_singular_end(arena, f, x, p, r)
}

/// `F(hi⁻) − F(lo⁺)` for a specific antiderivative `F`, after splitting
/// at any discontinuity of `F` inside the piece.  A discontinuity where
/// the argument of a principal branch crosses its cut
/// ([`check_branch_cuts`]) is not split but refused.
fn evaluate_antiderivative(
    arena: &mut Arena,
    f: ExprId,
    anti: ExprId,
    x: ExprId,
    p: &Piece,
    depth: usize,
) -> Result<ExprId, SymplexError> {
    if let Some(split) = piecewise_antiderivative(arena, f, anti, x, p, depth) {
        return split;
    }
    let anti = atan_of_complex_line_to_logs(arena, anti, x, p);
    let mut crossings = check_branch_cuts(arena, anti, x, p)?;
    // Discontinuities of F strictly inside (lo, hi).
    let range = numeric_range(arena, p.lo, p.hi);
    let scan = calculus_util::scan_breakpoints(arena, anti, x, scan_range(p.lo, p.hi, range));
    if let Some(r) = range
        && numeric_guard(arena, anti, x, r.lower, r.upper, &scan).is_false()
    {
        return Err(failed(
            "the antiderivative has a discontinuity inside the interval that could not be located",
        ));
    }
    let mut inner_points: Vec<(ExprId, Option<f64>)> = Vec::new();
    for bp in &scan.points {
        // Both poles and kinks (sign/Heaviside jumps) of F matter here.
        match position_in(arena, bp.point, p.lo, p.hi) {
            Position::Inside => match bp.value {
                Some(v) => {
                    if !inner_points.iter().any(|(q, _)| *q == bp.point) {
                        inner_points.push((bp.point, Some(v)));
                    }
                }
                None => {
                    return Err(failed(
                        "the antiderivative has a symbolic discontinuity inside the interval",
                    ));
                }
            },
            Position::Unknown => {
                let shown = arena.display(bp.point).to_string();
                return Err(failed(format!(
                    "cannot decide whether the antiderivative's discontinuity at {shown} lies inside the interval"
                )));
            }
            _ => {}
        }
    }
    // The points where an argument crosses a branch cut: `F` is continued
    // across each from either side ([`side_value`]), where it is otherwise
    // continuous.  At an end the value is taken from inside the piece; at a
    // singular end the limit decides (refused on a cut).
    crossings.retain(|c| !((c.at == p.lo && p.lo_sing) || (c.at == p.hi && p.hi_sing)));
    let jumps = inner_points.len();
    for c in &mut crossings {
        if c.at == p.lo || c.at == p.hi {
            continue;
        }
        let mut known = None;
        for (k, &(q, _)) in inner_points.iter().enumerate() {
            match cmp_points(arena, q, c.at) {
                Some(Ordering::Equal) if k < jumps => {
                    return Err(failed(
                        "the antiderivative is discontinuous where an argument crosses a branch cut",
                    ));
                }
                Some(Ordering::Equal) => known = Some(q),
                Some(_) => {}
                None => return Err(failed("cannot order the crossings of the branch cuts")),
            }
        }
        // Crossings at one point share its expression.
        match known {
            Some(q) => c.at = q,
            None => {
                let v = point_value(arena, c.at);
                inner_points.push((c.at, v));
            }
        }
    }
    // The arguments crossing at `at`, each with the side of the axis it lies
    // on within a piece that ends there (`from_below`) or starts there.
    let sides_at = |at: ExprId, from_below: bool| -> Vec<(ExprId, SignSet)> {
        crossings
            .iter()
            .filter(|c| c.at == at)
            .map(|c| (c.arg, if from_below { c.before } else { c.before.neg() }))
            .collect()
    };
    if !inner_points.is_empty() {
        if depth >= MAX_DEPTH || inner_points.len() > MAX_PIECES {
            return Err(failed("too many discontinuities of the antiderivative"));
        }
        let inner_points = sort_points(arena, inner_points)
            .ok_or_else(|| failed("cannot order the discontinuities of the antiderivative"))?;
        let mut sub_pieces = Vec::new();
        let mut cur_lo = p.lo;
        let mut cur_sing = p.lo_sing;
        for (pt, _) in &inner_points {
            // A crossing point is no singularity of `F`.
            let sing = !crossings.iter().any(|c| c.at == *pt);
            sub_pieces.push(Piece {
                lo: cur_lo,
                hi: *pt,
                lo_sing: cur_sing,
                hi_sing: sing,
            });
            cur_lo = *pt;
            cur_sing = sing;
        }
        sub_pieces.push(Piece {
            lo: cur_lo,
            hi: p.hi,
            lo_sing: cur_sing,
            hi_sing: p.hi_sing,
        });
        let mut total = Vec::new();
        for sp in &sub_pieces {
            let sides = [sides_at(sp.lo, false), sides_at(sp.hi, true)];
            match ftc(arena, anti, x, sp, &sides)? {
                Some(v) => total.push(v),
                None if sides.iter().any(|s| !s.is_empty()) => {
                    return Err(failed(
                        "the antiderivative has no value where an argument crosses a branch cut",
                    ));
                }
                None => return fallback_without_antiderivative(arena, f, x, sp),
            }
        }
        let s = arena.add(&total);
        return Ok(safe_eval(arena, s));
    }

    let sides = [sides_at(p.lo, false), sides_at(p.hi, true)];
    match ftc(arena, anti, x, p, &sides)? {
        Some(v) => Ok(v),
        None if sides.iter().any(|s| !s.is_empty()) => Err(failed(
            "the antiderivative has no value where an argument crosses a branch cut",
        )),
        None => fallback_without_antiderivative(arena, f, x, p),
    }
}

/// `F(hi⁻) − F(lo⁺)` for an antiderivative with `Piecewise` nodes whose
/// conditions compare `x` with constants `tⱼ` (`Piecewise((F₀, x < t₁),
/// …)`, as the integrator joins the pieces of `|g|`, `sign g` or of a
/// spiral `e^{λx}`): the piece split at the `tⱼ` inside it, every such node
/// resolved to its branch at a test point of each part, and the parts'
/// differences added (each branch is an antiderivative on its part, so this
/// holds whether or not the branches join continuously).  `None` when no
/// condition depends on `x`.  Up to 0.54 such a node reached the parameter
/// branches of [`integrate_piece`], which evaluated every branch over the
/// whole piece.
fn piecewise_antiderivative(
    arena: &mut Arena,
    f: ExprId,
    anti: ExprId,
    x: ExprId,
    p: &Piece,
    depth: usize,
) -> Option<Result<ExprId, SymplexError>> {
    let nodes: Vec<ExprId> = walk::post_order_ids(arena, anti)
        .into_iter()
        .filter(|&id| match arena.node(id) {
            ExprNode::Piecewise(pairs) => pairs.iter().any(|&(_, c)| walk::contains(arena, c, x)),
            _ => false,
        })
        .collect();
    if nodes.is_empty() {
        return None;
    }
    Some(split_piecewise_antiderivative(
        arena, f, anti, x, p, depth, &nodes,
    ))
}

/// [`piecewise_antiderivative`] for the `Piecewise` `nodes` of `anti`.
fn split_piecewise_antiderivative(
    arena: &mut Arena,
    f: ExprId,
    anti: ExprId,
    x: ExprId,
    p: &Piece,
    depth: usize,
    nodes: &[ExprId],
) -> Result<ExprId, SymplexError> {
    let unsplittable = || {
        failed(
            "the antiderivative is piecewise in the variable with conditions it cannot be split at",
        )
    };
    // The constants the conditions compare `x` with.
    let mut ends: Vec<ExprId> = Vec::new();
    for &id in nodes {
        let ExprNode::Piecewise(pairs) = arena.node(id).clone() else {
            continue;
        };
        for &(_, cond) in pairs.iter() {
            if !walk::contains(arena, cond, x) {
                continue;
            }
            let t = match *arena.node(cond) {
                ExprNode::Gt(a, b) | ExprNode::Ge(a, b)
                    if b == x && !walk::contains(arena, a, x) =>
                {
                    a
                }
                ExprNode::Gt(a, b) | ExprNode::Ge(a, b)
                    if a == x && !walk::contains(arena, b, x) =>
                {
                    b
                }
                _ => return Err(unsplittable()),
            };
            if !walk::free_symbols(arena, t).is_empty() {
                return Err(unsplittable());
            }
            if !ends.contains(&t) {
                ends.push(t);
            }
        }
    }
    let mut inner: Vec<(ExprId, Option<f64>)> = Vec::new();
    for t in ends {
        match position_in(arena, t, p.lo, p.hi) {
            Position::Inside => {
                let v = point_value(arena, t);
                inner.push((t, v));
            }
            Position::Unknown => return Err(unsplittable()),
            _ => {}
        }
    }
    if depth >= MAX_DEPTH || inner.len() > MAX_PIECES {
        return Err(failed("too many pieces of the antiderivative"));
    }
    let inner = sort_points(arena, inner).ok_or_else(unsplittable)?;
    let mut bounds: Vec<ExprId> = Vec::with_capacity(inner.len() + 2);
    bounds.push(p.lo);
    bounds.extend(inner.iter().map(|&(t, _)| t));
    bounds.push(p.hi);
    let last = bounds.len() - 2;
    let mut total = Vec::with_capacity(bounds.len() - 1);
    for (k, w) in bounds.windows(2).enumerate() {
        let piece = Piece {
            lo: w[0],
            hi: w[1],
            lo_sing: k == 0 && p.lo_sing,
            hi_sing: k == last && p.hi_sing,
        };
        let mid = test_point(arena, piece.lo, piece.hi).ok_or_else(unsplittable)?;
        // Outer nodes first: a branch may hold an inner one.
        let mut resolved = anti;
        for &id in nodes.iter().rev() {
            let ExprNode::Piecewise(pairs) = arena.node(id).clone() else {
                continue;
            };
            let mut chosen = None;
            for &(val, cond) in pairs.iter() {
                let at = subs::subs(arena, cond, x, mid);
                match decide_condition(arena, at) {
                    Truth::True => {
                        chosen = Some(val);
                        break;
                    }
                    Truth::False => {}
                    Truth::Unknown => return Err(unsplittable()),
                }
            }
            let val = chosen.ok_or_else(unsplittable)?;
            resolved = arena.subs_structural(resolved, id, val);
        }
        if piecewise_antiderivative_nodes_left(arena, resolved, x) {
            return Err(unsplittable());
        }
        let resolved = safe_eval(arena, resolved);
        total.push(evaluate_antiderivative(
            arena,
            f,
            resolved,
            x,
            &piece,
            depth + 1,
        )?);
    }
    let s = arena.add(&total);
    Ok(safe_eval(arena, s))
}

/// Does `e` still hold a `Piecewise` with a condition on `x`?
fn piecewise_antiderivative_nodes_left(arena: &Arena, e: ExprId, x: ExprId) -> bool {
    walk::post_order_ids(arena, e)
        .into_iter()
        .any(|id| match arena.node(id) {
            ExprNode::Piecewise(pairs) => pairs.iter().any(|&(_, c)| walk::contains(arena, c, x)),
            _ => false,
        })
}

// ═══════════════════════════════════════════════════════════════════════════
// Parity
// ═══════════════════════════════════════════════════════════════════════════

/// Symmetry of an integrand under `x → −x`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Parity {
    Even,
    Odd,
}

/// Is `lo` structurally or numerically `−hi`?
fn is_negation_of(arena: &mut Arena, lo: ExprId, hi: ExprId) -> bool {
    let neg_hi = arena.neg(hi);
    let neg_hi = safe_eval(arena, neg_hi);
    cmp_points(arena, lo, neg_hi) == Some(Ordering::Equal)
}

/// Determine whether `f(−x) = f(x)` (even) or `f(−x) = −f(x)` (odd).
fn parity(arena: &mut Arena, f: ExprId, x: ExprId) -> Option<Parity> {
    let neg_x = arena.neg(x);
    let g = subs::subs(arena, f, x, neg_x);
    let g = safe_eval(arena, g);
    let fe = safe_eval(arena, f);
    if g == fe {
        return Some(Parity::Even);
    }
    let neg_g = arena.neg(g);
    let neg_g = safe_eval(arena, neg_g);
    if neg_g == fe {
        return Some(Parity::Odd);
    }
    // Simplification-based check.
    let d = arena.sub(g, fe);
    let d = safe_eval(arena, d);
    if arena.is_zero_structural(d) || simplifies_to_zero(arena, d) {
        return Some(Parity::Even);
    }
    let s = arena.add(&[g, fe]);
    let s = safe_eval(arena, s);
    if arena.is_zero_structural(s) || simplifies_to_zero(arena, s) {
        return Some(Parity::Odd);
    }
    None
}

/// Run the unified simplifier and test for structural zero.
fn simplifies_to_zero(arena: &mut Arena, id: ExprId) -> bool {
    let expanded = expand::expand(arena, id);
    let expanded = safe_eval(arena, expanded);
    if arena.is_zero_structural(expanded) {
        return true;
    }
    let opts = crate::simplify::simplify_engine::SimplifyOpts::default();
    let r = crate::simplify::simplify_engine::unified_simplify(arena, expanded, &opts);
    arena.is_zero_structural(r.expr)
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

    fn show(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    fn f64_of(a: &mut Arena, id: ExprId) -> f64 {
        calculus_util::expr_to_f64(a, id).expect("numeric")
    }

    #[test]
    fn poly_0_to_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let (zero, one) = (a.zero(), a.one());
        let v = integrate_definite(&mut a, x2, x, zero, one).unwrap();
        assert_eq!(show(&a, v), "1/3");
    }

    #[test]
    fn reversed_bounds_negate() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let (zero, one) = (a.zero(), a.one());
        let v = integrate_definite(&mut a, x, x, one, zero).unwrap();
        assert_eq!(show(&a, v), "-1/2");
    }

    #[test]
    fn interior_pole_diverges() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let m2 = a.int(-2);
        let f = a.pow(x, m2);
        let (neg1, one) = (a.neg_one(), a.one());
        let r = integrate_definite(&mut a, f, x, neg1, one);
        assert!(matches!(r, Err(SymplexError::Divergent { .. })), "{r:?}");
    }

    #[test]
    fn one_over_x_symmetric_diverges() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let f = a.pow(x, a.neg_one());
        let (neg1, one) = (a.neg_one(), a.one());
        let r = integrate_definite(&mut a, f, x, neg1, one);
        assert!(matches!(r, Err(SymplexError::Divergent { .. })), "{r:?}");
    }

    #[test]
    fn ln_0_to_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let f = a.ln(x);
        let (zero, one) = (a.zero(), a.one());
        let v = integrate_definite(&mut a, f, x, zero, one).unwrap();
        assert_eq!(show(&a, v), "-1");
    }

    #[test]
    fn inv_sqrt_0_to_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let e = a.rational(-1, 2);
        let f = a.pow(x, e);
        let (zero, one) = (a.zero(), a.one());
        let v = integrate_definite(&mut a, f, x, zero, one).unwrap();
        assert_eq!(show(&a, v), "2");
    }

    #[test]
    fn exp_neg_x_0_to_inf() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let nx = a.neg(x);
        let f = a.exp(nx);
        let (zero, inf) = (a.zero(), a.infinity());
        let v = integrate_definite(&mut a, f, x, zero, inf).unwrap();
        assert_eq!(show(&a, v), "1");
    }

    #[test]
    fn gaussian_full_line() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let nx2 = a.neg(x2);
        let f = a.exp(nx2);
        let (ni, pi_) = (a.neg_infinity(), a.infinity());
        let v = integrate_definite(&mut a, f, x, ni, pi_).unwrap();
        let fv = f64_of(&mut a, v);
        assert!(
            (fv - std::f64::consts::PI.sqrt()).abs() < 1e-12,
            "{}",
            show(&a, v)
        );
    }

    #[test]
    fn lorentzian_full_line() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let one = a.one();
        let d = a.add(&[one, x2]);
        let f = a.pow(d, a.neg_one());
        let (ni, pi_) = (a.neg_infinity(), a.infinity());
        let v = integrate_definite(&mut a, f, x, ni, pi_).unwrap();
        assert_eq!(show(&a, v), "pi");
    }

    #[test]
    fn sinc_0_to_inf() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let s = a.sin(x);
        let f = a.div(s, x);
        let (zero, inf) = (a.zero(), a.infinity());
        let v = integrate_definite(&mut a, f, x, zero, inf).unwrap();
        assert_eq!(show(&a, v), "1/2*pi");
    }

    #[test]
    fn x_over_x_diverges_at_infinity() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let (one, inf) = (a.one(), a.infinity());
        let f = a.pow(x, a.neg_one());
        let r = integrate_definite(&mut a, f, x, one, inf);
        assert!(matches!(r, Err(SymplexError::Divergent { .. })), "{r:?}");
    }

    #[test]
    fn wallis_sin_4() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let s = a.sin(x);
        let four = a.int(4);
        let f = a.pow(s, four);
        let zero = a.zero();
        let half = a.rational(1, 2);
        let pi = a.pi();
        let hp = a.mul(&[half, pi]);
        let v = integrate_definite(&mut a, f, x, zero, hp).unwrap();
        assert_eq!(show(&a, v), "3/16*pi");
    }

    #[test]
    fn weierstrass_trap_is_not_wrong() {
        // ∫₀^{2π} dx/(2 + cos x) = 2π/√3; naive FTC through atan(tan(x/2)) gives 0.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let c = a.cos(x);
        let two = a.int(2);
        let d = a.add(&[two, c]);
        let f = a.pow(d, a.neg_one());
        let zero = a.zero();
        let pi = a.pi();
        let two_pi = a.mul(&[two, pi]);
        let r = integrate_definite(&mut a, f, x, zero, two_pi);
        match r {
            Ok(v) => {
                let fv = f64_of(&mut a, v);
                assert!(
                    (fv - 2.0 * std::f64::consts::PI / 3f64.sqrt()).abs() < 1e-10,
                    "{}",
                    show(&a, v)
                );
            }
            Err(SymplexError::ComputationFailed { .. }) => {}
            Err(e) => panic!("unexpected {e}"),
        }
    }

    #[test]
    fn heaviside_truncates() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one();
        let xm1 = a.sub(x, one);
        let h = a.heaviside(xm1);
        let f = a.mul(&[x, h]);
        let zero = a.zero();
        let two = a.int(2);
        let v = integrate_definite(&mut a, f, x, zero, two).unwrap();
        assert_eq!(show(&a, v), "3/2");
    }

    #[test]
    fn delta_sifts() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one();
        let xm1 = a.sub(x, one);
        let d = a.dirac_delta(xm1);
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let f = a.mul(&[x2, d]);
        let zero = a.zero();
        let five = a.int(5);
        let v = integrate_definite(&mut a, f, x, zero, five).unwrap();
        assert_eq!(show(&a, v), "1");
        let ten = a.int(10);
        let v = integrate_definite(&mut a, f, x, five, ten).unwrap();
        assert_eq!(show(&a, v), "0");
    }

    #[test]
    fn abs_splits() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let f = a.abs(x);
        let neg1 = a.neg_one();
        let two = a.int(2);
        let v = integrate_definite(&mut a, f, x, neg1, two).unwrap();
        assert_eq!(show(&a, v), "5/2");
    }

    #[test]
    fn symbolic_upper_bound() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let t = sym(&mut a, "t");
        let nx = a.neg(x);
        let f = a.exp(nx);
        let zero = a.zero();
        let v = integrate_definite(&mut a, f, x, zero, t).unwrap();
        assert_eq!(show(&a, v), "-exp(-t) + 1");
    }

    #[test]
    fn symbolic_bounds_with_pole_are_refused() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let t = sym(&mut a, "t");
        let f = a.pow(x, a.neg_one());
        let one = a.one();
        let r = integrate_definite(&mut a, f, x, one, t);
        assert!(
            matches!(r, Err(SymplexError::ComputationFailed { .. })),
            "{r:?}"
        );
    }

    #[test]
    fn quadrature_polynomial_exact() {
        let r = quadrature(
            &|x: f64| x.powi(5) - 3.0 * x,
            0.0,
            2.0,
            &QuadOpts::default(),
        )
        .unwrap();
        assert!((r.value - (64.0 / 6.0 - 6.0)).abs() < 1e-12);
        assert!(r.error < 1e-10);
    }

    #[test]
    fn quadrature_divergent_does_not_converge() {
        // Either an explicit error or a result whose error estimate is not
        // small relative to the value — never a confidently wrong number.
        match quadrature(&|x: f64| 1.0 / (x * x), -1.0, 1.0, &QuadOpts::default()) {
            Err(_) => {}
            Ok(r) => assert!(
                r.error > 1e-6 * r.value.abs(),
                "v={} err={}",
                r.value,
                r.error
            ),
        }
    }

    #[test]
    fn quadrature_endpoint_singularity() {
        let v = quadrature(&|x: f64| 1.0 / x.sqrt(), 0.0, 1.0, &QuadOpts::default())
            .unwrap()
            .value;
        assert!((v - 2.0).abs() < 1e-8, "{v}");
    }

    /// The guard of `integrate_definite`: `∫₃⁴ dx/(2 + cos x)` is
    /// 0.9150785657603947 (mpmath `quad`); `−2.7125201627080`, what
    /// `F(4) − F(3)` gave for 0.37's `2/√3·atan(tan(x/2)/√3)`, is refused,
    /// the right value accepted, and a parameter or a complex value says
    /// nothing.
    #[test]
    fn quadrature_contradicts_a_jumping_antiderivative() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let cos_x = a.cos(x);
        let two = a.int(2);
        let den = a.add(&[two, cos_x]);
        let minus_one = a.int(-1);
        let f = a.pow(den, minus_one);
        let (three, four) = (a.int(3), a.int(4));
        let wrong = a.num_ratio(Ratio::new(
            BigInt::from(-27_125_201_627_i64),
            BigInt::from(10_000_000_000_i64),
        ));
        let right = a.num_ratio(Ratio::new(
            BigInt::from(9_150_785_657_604_i64),
            BigInt::from(10_000_000_000_000_i64),
        ));
        let num = |a: &mut Arena, e: ExprId| evalf::evalf_complex64(a, e).ok();
        let (wrong, right) = (num(&mut a, wrong), num(&mut a, right));
        assert!(contradicted_by_quadrature(&mut a, f, x, three, four, wrong));
        assert!(!contradicted_by_quadrature(
            &mut a, f, x, three, four, right
        ));
        let p = sym(&mut a, "p");
        let g = a.mul(&[p, f]);
        assert!(!contradicted_by_quadrature(
            &mut a, g, x, three, four, wrong
        ));
        let complex = right.map(|z| z + num_complex::Complex64::new(0.0, 1.0));
        assert!(!contradicted_by_quadrature(
            &mut a, f, x, three, four, complex
        ));
    }

    #[test]
    fn gamma_exact_half_integers() {
        let mut a = Arena::new();
        let r = Ratio::new(BigInt::from(3), BigInt::from(2));
        let g = gamma_exact(&mut a, &r).unwrap();
        assert_eq!(show(&a, g), "1/2*sqrt(pi)");
        let r = Ratio::new(BigInt::from(-1), BigInt::from(2));
        let g = gamma_exact(&mut a, &r).unwrap();
        assert_eq!(show(&a, g), "-2*sqrt(pi)");
    }

    #[test]
    fn zeta_2_and_4() {
        let mut a = Arena::new();
        let z2 = zeta_even(&mut a, 2).unwrap();
        assert_eq!(show(&a, z2), "1/6*pi^2");
        let z4 = zeta_even(&mut a, 4).unwrap();
        assert_eq!(show(&a, z4), "1/90*pi^4");
    }
}
