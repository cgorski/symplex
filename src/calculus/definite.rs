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
//! 4. **Breakpoints.** Poles, `ln` zeros, `tan` poles, branch points of
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
//!    substitution) are split as well.  Endpoint values are direct
//!    substitutions when `F` is continuous there, otherwise **one-sided
//!    limits**: the approach `x → c⁺` is rewritten as `x = c + 1/u`,
//!    `u → +∞` (and `x = c − 1/u` for `c⁻`), so that every limit is a
//!    limit at `+∞`, where the Gruntz algorithm is strongest.  Limits are
//!    evaluated compositionally first (using assumptions for parameters)
//!    and only then by the black-box limit engine, whose finite results are
//!    sanity-checked numerically.
//! 6. **Divergence.** An infinite one-sided limit of `F` proves
//!    divergence.  Without an antiderivative, comparison with `1/(x − c)`
//!    (or `1/x` at infinity) is attempted; if nothing can be decided the
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
use crate::base::walk;
use crate::calculus::calculus_util::{
    self, BreakKind, BreakScan, Breakpoint, LinearCoeffs, POINT_VALUE_ACCURACY, ScanRange,
};
use crate::calculus::residue;
use crate::transforms::{eval, evalf, expand, subs};

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
    // below the 16th digit of the real part.
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
    let Ok(points) = quadrature_breakpoints(
        arena,
        f,
        x,
        Interval::closed(a_ord, b_ord),
        Some(ends),
        None,
    ) else {
        return false;
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
    if b != arena.infinity() {
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
    id == arena.infinity() || id == arena.neg_infinity()
}

/// Numeric value of a constant expression (±∞ allowed), or `None`.
fn point_value(arena: &mut Arena, id: ExprId) -> Option<f64> {
    if id == arena.infinity() {
        return Some(f64::INFINITY);
    }
    if id == arena.neg_infinity() {
        return Some(f64::NEG_INFINITY);
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
        if p == arena.infinity() || q == arena.neg_infinity() {
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
    if id == arena.infinity() {
        return Some(Ordering::Greater);
    }
    if id == arena.neg_infinity() {
        return Some(Ordering::Less);
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
    let range = numeric_range(arena, a, b);
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
        let mut prev: Option<(f64, f64)> = None;
        for &t in &grid {
            let v = func(&[t]);
            if !v.is_finite() {
                return Truth::Unknown;
            }
            if let Some((pt, pv)) = prev
                && pv != 0.0
                && v != 0.0
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
    let lo_is_neg_inf = p.lo == arena.neg_infinity();
    let hi_is_pos_inf = p.hi == arena.infinity();

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

    // ── Antiderivative ──────────────────────────────────────────────
    let anti = crate::transforms::integrate::integrate(arena, f, x);
    let anti = safe_eval(arena, anti);
    if walk::has_unevaluated(arena, anti) {
        return fallback_without_antiderivative(arena, f, x, p);
    }

    // Parametric antiderivatives come wrapped in a Piecewise over the
    // parameter (e.g. ∫ xⁿ: n ≠ −1 vs n = −1).  Conditions that the
    // assumption system can decide are resolved first; the remaining
    // branches are evaluated individually.  A divergence in an undecided
    // branch is reported as "cannot compute" (the answer depends on the
    // parameter), not as a divergence of the whole integral.
    if let ExprNode::Piecewise(pairs) = arena.node(anti).clone() {
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
            return evaluate_antiderivative(arena, f, live[0].0, x, p, depth);
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

    evaluate_antiderivative(arena, f, anti, x, p, depth)
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
    let anti = atan_of_complex_line_to_logs(arena, anti, x, p);
    check_branch_cuts(arena, anti, x, p)?;
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
            sub_pieces.push(Piece {
                lo: cur_lo,
                hi: *pt,
                lo_sing: cur_sing,
                hi_sing: true,
            });
            cur_lo = *pt;
            cur_sing = true;
        }
        sub_pieces.push(Piece {
            lo: cur_lo,
            hi: p.hi,
            lo_sing: cur_sing,
            hi_sing: p.hi_sing,
        });
        let mut total = Vec::new();
        for sp in &sub_pieces {
            match ftc(arena, anti, x, sp)? {
                Some(v) => total.push(v),
                None => return fallback_without_antiderivative(arena, f, x, sp),
            }
        }
        let s = arena.add(&total);
        return Ok(safe_eval(arena, s));
    }

    match ftc(arena, anti, x, p)? {
        Some(v) => Ok(v),
        None => fallback_without_antiderivative(arena, f, x, p),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Branch cuts of the antiderivative along the real path
// ═══════════════════════════════════════════════════════════════════════════

/// Where the principal branch of a function of `u` is discontinuous.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Cut {
    /// The real axis below a bound: `ln`, non-integer powers, `arg`,
    /// `loggamma`, `Ei`, `Ci`, `Chi`, `Eₙ`, `Yν`, `Kν`, incomplete gamma,
    /// `li` (0), `W` (−1/e), `acosh` (1).
    RealBelow(f64),
    /// The real axis above a bound: `polylog(s, u)` (1).
    RealAbove(f64),
    /// The real axis outside `[−1, 1]`: `asin`, `acos`, `atanh`.
    RealOutside,
    /// The imaginary axis outside `[−i, i]`: `atan`, `asinh`.
    ImaginaryOutside,
}

/// The arguments `u` (depending on `x`) of the principal-branch functions
/// in `anti`, with their cuts.
fn cut_arguments(arena: &Arena, anti: ExprId, x: ExprId) -> Vec<(ExprId, Cut)> {
    let mut out: Vec<(ExprId, Cut)> = Vec::new();
    let mut stack = vec![anti];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let node = arena.node(id);
        let found = match *node {
            ExprNode::Ln(u)
            | ExprNode::Arg(u)
            | ExprNode::LogGamma(u)
            | ExprNode::Ei(u)
            | ExprNode::Ci(u)
            | ExprNode::Li(u) => Some((u, Cut::RealBelow(0.0))),
            ExprNode::Pow(u, e) if !arena.as_num(e).is_some_and(|r| r.is_integer()) => {
                Some((u, Cut::RealBelow(0.0)))
            }
            ExprNode::LambertW(u) => Some((u, Cut::RealBelow(-(-1.0f64).exp()))),
            ExprNode::Acosh(u) => Some((u, Cut::RealBelow(1.0))),
            ExprNode::Asin(u) | ExprNode::Acos(u) | ExprNode::Atanh(u) => {
                Some((u, Cut::RealOutside))
            }
            ExprNode::Atan(u) | ExprNode::Asinh(u) => Some((u, Cut::ImaginaryOutside)),
            // Before 0.46 the library functions were not looked at: the
            // dilogarithms of `∫ ln(x)·R(x) dx` (cut `[1, ∞)`) and `Chi`, `Eₙ`
            // were taken as continuous whatever their arguments did.
            ExprNode::Lib(f, ref args) => {
                use crate::base::libfn::LibFn;
                match (f, args.as_slice()) {
                    (LibFn::Chi, [u]) => Some((*u, Cut::RealBelow(0.0))),
                    (
                        LibFn::ExpInt
                        | LibFn::BesselY
                        | LibFn::BesselK
                        | LibFn::UpperGamma
                        | LibFn::LowerGamma,
                        [_, u],
                    ) => Some((*u, Cut::RealBelow(0.0))),
                    (LibFn::PolyLog, [_, u]) => Some((*u, Cut::RealAbove(1.0))),
                    _ => None,
                }
            }
            _ => None,
        };
        if let Some((u, cut)) = found
            && walk::contains(arena, u, x)
            && !out.contains(&(u, cut))
        {
            out.push((u, cut));
        }
        node.for_each_child(|c| stack.push(c));
    }
    out
}

/// Is `e` real at every real `x` where it is defined, under the declared
/// assumptions of the other symbols?  `x`, constants that
/// [`constant_realness_as_declared`](crate::transforms::realness::constant_realness_as_declared)
/// proves real, and sums, products, integer powers, `exp`, `sin`, `cos`,
/// `tan`, `sinh`, `cosh`, `tanh`, `atan`, `|·|` of such.  Conservative.
fn real_for_real_x(arena: &mut Arena, e: ExprId, x: ExprId, reals: &mut AssumptionCache) -> bool {
    let mut stack = vec![e];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if id == x {
            continue;
        }
        if !walk::contains(arena, id, x) {
            if !Truth::from(crate::transforms::realness::constant_realness_as_declared(
                arena, id, 30, reals,
            ))
            .is_true()
            {
                return false;
            }
            continue;
        }
        match arena.node(id).clone() {
            ExprNode::Add(children) | ExprNode::Mul(children) => stack.extend(children),
            ExprNode::Pow(b, e) if arena.as_num(e).is_some_and(|r| r.is_integer()) => stack.push(b),
            ExprNode::Neg(u)
            | ExprNode::Exp(u)
            | ExprNode::Sin(u)
            | ExprNode::Cos(u)
            | ExprNode::Tan(u)
            | ExprNode::Sinh(u)
            | ExprNode::Cosh(u)
            | ExprNode::Tanh(u)
            | ExprNode::Atan(u)
            | ExprNode::Abs(u) => stack.push(u),
            _ => return false,
        }
    }
    true
}

/// Can the argument `u` of a function with cut `cut` provably not cross
/// that cut as `x` runs along the real axis over the piece `p`, whatever
/// the values of the parameters under their declared assumptions?
///
/// - A real `u` ([`real_for_real_x`]) moves along the real axis: it may run
///   along a cut on it, where the principal value is the limit from one
///   side and continuous, but never across one.
/// - For a cut on the real axis it suffices that `im u` does not depend on
///   `x`: `u = c + g(x)` with `g` real and `c` constant (`ln(x + a + i)`).
/// - A linear `u = α·x + β` runs along a straight segment, which meets the
///   axis of the cut (the real axis, or the imaginary one for `atan`) at
///   most once unless it lies in it.  If it meets it at an endpoint of the
///   piece, off the cut, it does not cross it inside: `∫₀¹ dx/(a·x + 1)` has
///   `u(0) = 1`, and `∫₀¹ dx/(1 + a²x²) = atan(a)/a` has `u(0) = 0`, for
///   every complex `a` (where the integral exists).
/// - For a cut `(−∞, 0]` (`ln`, powers): `u = c·g(x)` with `c` constant and
///   `g` real ([`constant_multiple_of_real`]) stays on the line through 0
///   in the direction of `c`, which meets the real axis only at the branch
///   point 0 (or lies in it).  Passing through 0 is a zero of `u`: a
///   singular point of the integrand, which the breakpoint scan splits at,
///   or a point where `F` is continuous (`u·ln u`, `u^{3/2}`).  So
///   `∫₀¹ ln(a·x + a) dx` and `∫₁² √(a²·x + a²) dx` are differences for
///   every complex `a`.
fn cut_crossing_excluded(
    arena: &mut Arena,
    u: ExprId,
    x: ExprId,
    cut: Cut,
    p: &Piece,
    reals: &mut AssumptionCache,
) -> bool {
    if real_for_real_x(arena, u, x, reals) {
        return true;
    }
    if cut != Cut::ImaginaryOutside && imaginary_part_constant_polynomial(arena, u, x, reals) {
        return true;
    }
    if cut != Cut::ImaginaryOutside
        && let ExprNode::Add(children) = arena.node(u).clone()
    {
        let dependent: Vec<ExprId> = children
            .into_iter()
            .filter(|&t| walk::contains(arena, t, x))
            .collect();
        if dependent
            .into_iter()
            .all(|t| real_for_real_x(arena, t, x, reals))
        {
            return true;
        }
    }
    // A line through 0 meets the real axis only at 0: the branch point of
    // a cut `(−∞, 0]`, off a cut `[b, ∞)` with `b > 0`.
    let through_origin_ok = match cut {
        Cut::RealBelow(b) => b == 0.0,
        Cut::RealAbove(b) => b > 0.0,
        _ => false,
    };
    if through_origin_ok && constant_multiple_of_real(arena, u, x, reals) {
        return true;
    }
    calculus_util::linear_coeffs_exact(arena, u, x).is_some()
        && [p.lo, p.hi]
            .into_iter()
            .any(|e| meets_axis_off_the_cut(arena, u, x, e, cut, reals))
}

/// Is `u` a polynomial in `x` whose coefficients other than the constant
/// term are real, so that `im u` does not depend on `x`?  (`(√3/3)·(x + a) +
/// i` is not a sum with a real `x`-dependent part as written.)
fn imaginary_part_constant_polynomial(
    arena: &mut Arena,
    u: ExprId,
    x: ExprId,
    reals: &mut AssumptionCache,
) -> bool {
    let Some(coeffs) = calculus_util::poly_coeffs_symbolic(arena, u, x) else {
        return false;
    };
    coeffs.len() >= 2
        && coeffs.into_iter().skip(1).all(|c| {
            Truth::from(crate::transforms::realness::constant_realness_as_declared(
                arena, c, 30, reals,
            ))
            .is_true()
        })
}

/// Is `u = c·g(x)` with `c` constant and `g` real for real `x`?  A product
/// whose `x`-dependent factors are real ([`real_for_real_x`]), or a
/// polynomial in `x` whose coefficients are real multiples of its leading
/// one (`a·x + a`, `a²·x² + 2a²`).
fn constant_multiple_of_real(
    arena: &mut Arena,
    u: ExprId,
    x: ExprId,
    reals: &mut AssumptionCache,
) -> bool {
    if let ExprNode::Mul(factors) = arena.node(u).clone() {
        let dependent: Vec<ExprId> = factors
            .into_iter()
            .filter(|&t| walk::contains(arena, t, x))
            .collect();
        if dependent
            .into_iter()
            .all(|t| real_for_real_x(arena, t, x, reals))
        {
            return true;
        }
    }
    let Some(coeffs) = calculus_util::poly_coeffs_symbolic(arena, u, x) else {
        return false;
    };
    let Some(&lead) = coeffs.last() else {
        return false;
    };
    if coeffs.len() < 2 || arena.is_zero_structural(lead) {
        return false;
    }
    coeffs.into_iter().all(|c| {
        let ratio = arena.div(c, lead);
        let ratio = eval::eval(arena, ratio);
        Truth::from(crate::transforms::realness::constant_realness_as_declared(
            arena, ratio, 30, reals,
        ))
        .is_true()
    })
}

/// Is `u(e)` (at a finite endpoint `e`) a constant on the axis of `cut`
/// but not on the cut itself (a branch point counts as off the cut)?
fn meets_axis_off_the_cut(
    arena: &mut Arena,
    u: ExprId,
    x: ExprId,
    e: ExprId,
    cut: Cut,
    reals: &mut AssumptionCache,
) -> bool {
    if is_infinite(arena, e) {
        return false;
    }
    let at = subs::subs(arena, u, x, e);
    let at = eval::eval(arena, at);
    if !walk::free_symbols(arena, at).is_empty() {
        return false;
    }
    // On the real axis: `u(e)` real; on the imaginary one: `i·u(e)` real.
    let on_axis = match cut {
        Cut::ImaginaryOutside => {
            let i = arena.i_unit();
            let rotated = arena.mul(&[i, at]);
            eval::eval(arena, rotated)
        }
        _ => at,
    };
    if !Truth::from(crate::transforms::realness::constant_realness_as_declared(
        arena, on_axis, 30, reals,
    ))
    .is_true()
    {
        return false;
    }
    let Some(v) = evalf::eval_const_f64(arena, on_axis) else {
        return false;
    };
    match cut {
        Cut::RealBelow(bound) => v >= bound,
        Cut::RealAbove(bound) => v <= bound,
        Cut::RealOutside | Cut::ImaginaryOutside => v.abs() <= 1.0,
    }
}

/// The first sample point of `[lo, hi]` ([`guard_grid`]) where the
/// parameter-free `u` crosses `cut`: its transversal coordinate (`im u`
/// for a cut on the real axis, `re u` for one on the imaginary axis)
/// changes sign between two samples, at a point (linear interpolation) of
/// the cut.  Coordinates within `1e-12·max(1, |u|)` of 0 count as 0 (no
/// sign), so a real `u` with a rounding residue does not cross.
fn numeric_cut_crossing(
    arena: &mut Arena,
    u: ExprId,
    x: ExprId,
    cut: Cut,
    lo: f64,
    hi: f64,
) -> Option<f64> {
    let mut prev: Option<(f64, f64, f64)> = None;
    for t in guard_grid(lo, hi) {
        let Some(q) = crate::base::numeric::f64_to_ratio_exact(t) else {
            prev = None;
            continue;
        };
        let pt = arena.num_ratio(q);
        let ut = subs::subs(arena, u, x, pt);
        let ut = eval::eval(arena, ut);
        let Ok(z) = evalf::evalf_complex64(arena, ut) else {
            prev = None;
            continue;
        };
        if !z.re.is_finite() || !z.im.is_finite() {
            prev = None;
            continue;
        }
        let tiny = 1e-12 * z.norm().max(1.0);
        let (s, c) = match cut {
            Cut::ImaginaryOutside => (z.re, z.im),
            _ => (z.im, z.re),
        };
        let s = if s.abs() <= tiny { 0.0 } else { s };
        if let Some((t0, s0, c0)) = prev
            && s0 != 0.0
            && s != 0.0
            && (s0 < 0.0) != (s < 0.0)
        {
            let w = s0 / (s0 - s);
            let along = c0 + w * (c - c0);
            let on_cut = match cut {
                Cut::RealBelow(bound) => along < bound,
                Cut::RealAbove(bound) => along > bound,
                Cut::RealOutside | Cut::ImaginaryOutside => along.abs() > 1.0,
            };
            if on_cut {
                return Some(t0 + w * (t - t0));
            }
        }
        prev = Some((t, s, c));
    }
    None
}

/// `F` with every `atan(u)` whose argument may cross the cut of `atan`
/// ([`cut_crossing_excluded`]) but is a line `u = α·x + β` with a real
/// slope `α` written as `(i/2)·(ln(u + i) − ln(u − i))`.  That has the same
/// derivative `u′/(1 + u²)`, and its logarithms have the imaginary parts
/// `im β ± 1`, independent of `x`, so they never cross their cuts: the
/// Newton–Leibniz difference is right for every complex `β`.  (This is
/// SymPy's form: `integrate(1/(x**2 + 2*a*x + a**2 + 1), (x, 0, 1))` →
/// `I*log(a - I)/2 - I*log(a + I)/2 - I*log(a + 1 - I)/2 +
/// I*log(a + 1 + I)/2`.)  For a real `u` nothing changes.
fn atan_of_complex_line_to_logs(arena: &mut Arena, anti: ExprId, x: ExprId, p: &Piece) -> ExprId {
    let mut reals = AssumptionCache::new();
    let mut out = anti;
    for (u, cut) in cut_arguments(arena, anti, x) {
        if cut != Cut::ImaginaryOutside || cut_crossing_excluded(arena, u, x, cut, p, &mut reals) {
            continue;
        }
        let atan_u = arena.atan(u);
        if !walk::contains(arena, out, atan_u) {
            continue;
        }
        let Some(LinearCoeffs { slope, .. }) = calculus_util::linear_coeffs_exact(arena, u, x)
        else {
            continue;
        };
        if !Truth::from(crate::transforms::realness::constant_realness_as_declared(
            arena, slope, 30, &mut reals,
        ))
        .is_true()
        {
            continue;
        }
        let i = arena.i_unit();
        let neg_i = arena.neg(i);
        let up = arena.add(&[u, i]);
        let um = arena.add(&[u, neg_i]);
        let ln_up = arena.ln(up);
        let ln_um = arena.ln(um);
        let diff = arena.sub(ln_up, ln_um);
        let half = arena.rational(1, 2);
        let logs = arena.mul(&[half, i, diff]);
        out = arena.subs_structural(out, atan_u, logs);
    }
    out
}

/// Refuse `F(hi) − F(lo)` when a principal branch in `F` may jump inside
/// the piece because its argument crosses the branch cut.  The
/// antiderivative is then discontinuous where the integrand is smooth, and
/// the Newton–Leibniz difference is off by the jump (a multiple of `2πi`
/// for `ln`, of `π` for `atan`).
///
/// This happens with complex constants: `∫₀¹ dx/(x² + 2a·x + a² + 1)` is
/// `atan(a + 1) − atan(a)` for a real `a`, but for `a = −1/2 + 3i/2` the
/// path `x + a` crosses the cut of `atan` at `x = 1/2` and the true value
/// (mpmath `quad`) differs from that by `π`.  An unassumed parameter is a
/// complex number in the crate's model (decision D4, 0.31), and up to 0.30
/// the difference was returned for every `a`.  (That one is now rewritten
/// first, [`atan_of_complex_line_to_logs`]; `∫₀¹ atan(x + a) dx`, whose
/// antiderivative also has `ln((x + a)² + 1)`, is refused.)
///
/// Each argument is first tried for a proof that no crossing can occur
/// ([`cut_crossing_excluded`]: real, with an imaginary part independent of
/// `x`, as `ln(x + a + i)`, or a line meeting the cut's axis off the cut at
/// an endpoint).  Otherwise an argument with parameters is refused (its
/// crossing depends on their values), and a parameter-free one is sampled
/// on the piece ([`numeric_cut_crossing`]), refused where it crosses and
/// when the piece has symbolic bounds.
fn check_branch_cuts(
    arena: &mut Arena,
    anti: ExprId,
    x: ExprId,
    p: &Piece,
) -> Result<(), SymplexError> {
    let arguments = cut_arguments(arena, anti, x);
    if arguments.is_empty() {
        return Ok(());
    }
    let mut reals = AssumptionCache::new();
    let mut range: Option<Option<Interval<f64>>> = None;
    for (u, cut) in arguments {
        if cut_crossing_excluded(arena, u, x, cut, p, &mut reals) {
            continue;
        }
        let shown = arena.display(u).to_string();
        if walk::free_symbols(arena, u).iter().any(|&s| s != x) {
            return Err(failed(format!(
                "the antiderivative's branch cut at {shown} may be crossed along the path for complex values of the parameters"
            )));
        }
        let r = *range.get_or_insert_with(|| numeric_range(arena, p.lo, p.hi));
        let Some(r) = r else {
            return Err(failed(format!(
                "cannot tell whether {shown} crosses the antiderivative's branch cut between symbolic bounds"
            )));
        };
        if let Some(t) = numeric_cut_crossing(arena, u, x, cut, r.lower, r.upper) {
            return Err(failed(format!(
                "the antiderivative is discontinuous near x = {t:.6}, where {shown} crosses a branch cut"
            )));
        }
    }
    Ok(())
}

/// `F(hi⁻) − F(lo⁺)`.  `Ok(None)` when an endpoint value is unknown.
fn ftc(
    arena: &mut Arena,
    anti: ExprId,
    x: ExprId,
    p: &Piece,
) -> Result<Option<ExprId>, SymplexError> {
    let hi_val = endpoint_value(arena, anti, x, p.hi, Side::FromLeft, p.hi_sing);
    if let Some(inf @ (Extended::PosInf | Extended::NegInf)) = hi_val {
        let shown = arena.display(p.hi).to_string();
        let dir = if inf == Extended::PosInf {
            "+∞"
        } else {
            "−∞"
        };
        return Err(divergent(format!(
            "the antiderivative tends to {dir} as x → {shown}"
        )));
    }
    let lo_val = endpoint_value(arena, anti, x, p.lo, Side::FromRight, p.lo_sing);
    if let Some(inf @ (Extended::PosInf | Extended::NegInf)) = lo_val {
        let shown = arena.display(p.lo).to_string();
        let dir = if inf == Extended::PosInf {
            "+∞"
        } else {
            "−∞"
        };
        return Err(divergent(format!(
            "the antiderivative tends to {dir} as x → {shown}"
        )));
    }
    match (hi_val, lo_val) {
        (Some(Extended::Finite(h)), Some(Extended::Finite(l))) => {
            let d = arena.sub(h, l);
            Ok(Some(safe_eval(arena, d)))
        }
        _ => Ok(None),
    }
}

/// Without an antiderivative: try to prove divergence by comparison,
/// otherwise report failure.
fn fallback_without_antiderivative(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    p: &Piece,
) -> Result<ExprId, SymplexError> {
    if p.hi_sing && diverges_near(arena, f, x, p.hi, Side::FromLeft) {
        let shown = arena.display(p.hi).to_string();
        return Err(divergent(format!(
            "the integrand is not integrable near x = {shown}"
        )));
    }
    if p.lo_sing && diverges_near(arena, f, x, p.lo, Side::FromRight) {
        let shown = arena.display(p.lo).to_string();
        return Err(divergent(format!(
            "the integrand is not integrable near x = {shown}"
        )));
    }
    Err(failed(
        "no antiderivative in closed form and no matching known integral",
    ))
}

/// Comparison test: near a finite `c`, `∫ f` diverges if
/// `(x − c)·f(x)` has a non-zero (or infinite) one-sided limit; at `±∞`
/// if `x·f(x)` does.
fn diverges_near(arena: &mut Arena, f: ExprId, x: ExprId, c: ExprId, side: Side) -> bool {
    let u = fresh_symbol(arena, "defd", 0);
    let h = if c == arena.infinity() {
        let xf = arena.mul(&[x, f]);
        subs::subs(arena, xf, x, u)
    } else if c == arena.neg_infinity() {
        let xf = arena.mul(&[x, f]);
        let neg_u = arena.neg(u);
        subs::subs(arena, xf, x, neg_u)
    } else {
        // x = c ± 1/u  ⇒  (x − c) f = ± f(c ± 1/u) / u
        let one = arena.one();
        let inv_u = arena.div(one, u);
        let repl = match side {
            Side::FromRight => arena.add(&[c, inv_u]),
            Side::FromLeft => arena.sub(c, inv_u),
        };
        let fu = subs::subs(arena, f, x, repl);
        let scaled = arena.mul(&[fu, inv_u]);
        match side {
            Side::FromRight => scaled,
            Side::FromLeft => arena.neg(scaled),
        }
    };
    let h = safe_eval(arena, h);
    match limit_pos_inf(arena, h, u) {
        Some(Extended::PosInf | Extended::NegInf) => true,
        Some(Extended::Finite(v)) => sign_of(arena, v).is_some_and(|s| s != Ordering::Equal),
        None => false,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Endpoint values and one-sided limits
// ═══════════════════════════════════════════════════════════════════════════

/// Direction of approach to an endpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    /// `x → c⁺` (lower endpoint).
    FromRight,
    /// `x → c⁻` (upper endpoint).
    FromLeft,
}

/// Does `F` contain a node with a jump (so a direct substitution at a
/// singular endpoint might not equal the one-sided limit)?
fn has_jump_node(arena: &Arena, root: ExprId) -> bool {
    let mut stack = vec![root];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Sign(_)
            | ExprNode::Heaviside(_)
            | ExprNode::Floor(_)
            | ExprNode::Ceiling(_)
            | ExprNode::Piecewise(_)
            | ExprNode::Tan(_)
            | ExprNode::Atan2(_, _) => return true,
            node => node.for_each_child(|c| stack.push(c)),
        }
    }
    false
}

/// Value of `F` at endpoint `c`, approached from `side`: finite, `±∞`, or
/// `None` when unknown.
fn endpoint_value(
    arena: &mut Arena,
    anti: ExprId,
    x: ExprId,
    c: ExprId,
    side: Side,
    singular: bool,
) -> Option<Extended<ExprId>> {
    if !is_infinite(arena, c) {
        // Direct substitution.  Valid when F is continuous at c; when c is
        // a singular endpoint we still accept a finite value provided F
        // has no jump-type nodes (its continuous extension equals the
        // one-sided limit for elementary F).
        let v = subs::subs(arena, anti, x, c);
        let v = safe_eval(arena, v);
        if let Some(numeric) = finite_value(arena, v, x)
            && (!singular || !has_jump_node(arena, anti))
        {
            // A complex value at a point of a cut may be the principal
            // value from the wrong side.
            if numeric.is_some_and(|z| z.im != 0.0) && endpoint_on_a_cut(arena, anti, x, c) {
                return None;
            }
            return Some(Extended::Finite(v));
        }
        // One-sided limit: x = c ± 1/u, u → +∞.
        let u = fresh_symbol(arena, "defu", 0);
        let one = arena.one();
        let inv_u = arena.div(one, u);
        let repl = match side {
            Side::FromRight => arena.add(&[c, inv_u]),
            Side::FromLeft => arena.sub(c, inv_u),
        };
        let g = subs::subs(arena, anti, x, repl);
        let g = safe_eval(arena, g);
        return limit_pos_inf(arena, g, u);
    }
    if c == arena.infinity() {
        return limit_pos_inf(arena, anti, x);
    }
    // c = −∞: x = −u.
    let u = fresh_symbol(arena, "defu", 1);
    let neg_u = arena.neg(u);
    let g = subs::subs(arena, anti, x, neg_u);
    let g = safe_eval(arena, g);
    limit_pos_inf(arena, g, u)
}

/// Does a principal-branch argument `u` of `F` lie on its cut at the
/// finite endpoint `c` while `u` leaves the cut's axis inside the
/// interval?  Then `F(c)` (the principal value, the limit from one side of
/// the cut) need not be the limit of `F` along the path, which may come
/// from the other side, and the endpoint value is unknown.  Arguments that
/// are real for real `x`, or whose imaginary part does not depend on `x`,
/// run along the axis (on the cut or parallel to it) and are fine; so are
/// arguments with parameters, which [`check_branch_cuts`] has vetted.
fn endpoint_on_a_cut(arena: &mut Arena, anti: ExprId, x: ExprId, c: ExprId) -> bool {
    let arguments = cut_arguments(arena, anti, x);
    if arguments.is_empty() {
        return false;
    }
    let mut reals = AssumptionCache::new();
    for (u, cut) in arguments {
        if real_for_real_x(arena, u, x, &mut reals)
            || (cut != Cut::ImaginaryOutside
                && imaginary_part_constant_polynomial(arena, u, x, &mut reals))
        {
            continue;
        }
        let at = subs::subs(arena, u, x, c);
        let at = safe_eval(arena, at);
        if !walk::free_symbols(arena, at).is_empty() {
            continue;
        }
        let Ok(z) = evalf::evalf_complex64(arena, at) else {
            return true;
        };
        let tiny = 1e-12 * z.norm().max(1.0);
        let on_cut = match cut {
            Cut::RealBelow(b) => z.im.abs() <= tiny && z.re < b,
            Cut::RealAbove(b) => z.im.abs() <= tiny && z.re > b,
            Cut::RealOutside => z.im.abs() <= tiny && z.re.abs() > 1.0,
            Cut::ImaginaryOutside => z.re.abs() <= tiny && z.im.abs() > 1.0,
        };
        if on_cut {
            return true;
        }
    }
    false
}

/// `lim_{u → +∞} g(u)`: finite, `±∞`, or `None` when unknown.
///
/// Compositional rules (with assumptions on parameters) first; then the
/// black-box limit engine for parameter-free expressions, whose finite
/// answers are checked against numeric samples.
fn limit_pos_inf(arena: &mut Arena, g: ExprId, u: ExprId) -> Option<Extended<ExprId>> {
    if !walk::contains(arena, g, u) {
        let v = safe_eval(arena, g);
        return if is_finite_value(arena, v, u) {
            Some(Extended::Finite(v))
        } else {
            None
        };
    }
    match compositional_limit(arena, g, u) {
        LimVal::Value(Extended::Finite(v)) => {
            if is_finite_value(arena, v, u) {
                return Some(Extended::Finite(v));
            }
        }
        LimVal::Value(infinite) => return Some(infinite),
        LimVal::Bounded | LimVal::Unknown => {}
    }
    if let Some(v) = log_polylog_limit(arena, g, u) {
        return Some(Extended::Finite(v));
    }

    // Black-box engine: only trusted when no parameters are present and
    // the expression has no unbounded oscillation (the engine can report
    // `∞` for `u·sin(u²)`, whose limit does not exist).
    let params: Vec<ExprId> = walk::free_symbols(arena, g)
        .into_iter()
        .filter(|&s| s != u)
        .collect();
    if params.is_empty() && !has_unbounded_oscillation(arena, g, u) {
        let inf = arena.infinity();
        if let Ok(l) = crate::calculus::limit::limit(arena, g, u, inf) {
            // The engine occasionally leaks its internal dummy symbols into
            // a half-finished result; a limit of a parameter-free
            // expression must itself be parameter-free.
            if !walk::free_symbols(arena, l).is_empty() {
                return None;
            }
            if l == arena.infinity() || l == arena.neg_infinity() {
                let sign: i8 = if l == arena.infinity() { 1 } else { -1 };
                if numeric_growth_sanity(arena, g, u, sign) {
                    return Some(if sign > 0 {
                        Extended::PosInf
                    } else {
                        Extended::NegInf
                    });
                }
                return None;
            }
            // A complex limit would escape the numeric sanity check (it
            // samples a compiled real function): only real ones are taken.
            if is_finite_value(arena, l, u)
                && calculus_util::expr_to_f64(arena, l).is_some()
                && numeric_limit_sanity(arena, g, u, l)
            {
                return Some(Extended::Finite(l));
            }
        }
    }
    None
}

/// `lim_{u→+∞} g` for a `g` that is (multiplied out) a sum of terms
/// `c·ln A`, `c·ln A·ln B`, `c·Li₂(C)` with `A`, `B`, `C` linear in `u` (numeric
/// slopes) and terms with finite limits: with `ℓ = ln u`, `ln(a·u + b) = ℓ +
/// λ + o(1)` ([`asymptotic_log_shift`]) and `Li₂(C) = −π²/6 − ½·ln²(−C) −
/// Li₂(1/C)` (DLMF 25.12.4, `C` off `[0, ∞)`), each term is `α·ℓ² + β·ℓ + γ +
/// o(1)`; when the `α` and the `β` add up to 0 the limit is the sum of the
/// `γ`.  The antiderivative of `ln(x + c)/D(x)` is such a sum of `Li₂` and
/// products of logarithms at the roots of `D`, whose `ln²u` terms cancel
/// only in the sum: `∫₀^∞ ln(x + 1)/(x² + 1) dx = π·ln 2/4 + G` was refused up
/// to 0.46.  The sums are decided exactly, or for a parameter-free `g` to
/// 30 digits (the closed form then faces the quadrature check of
/// [`integrate_definite`]).
fn log_polylog_limit(arena: &mut Arena, g: ExprId, u: ExprId) -> Option<ExprId> {
    let has_target = walk::post_order_ids(arena, g).into_iter().any(|id| {
        matches!(arena.node(id), ExprNode::Lib(crate::base::libfn::LibFn::PolyLog, args)
            if args.len() == 2)
    });
    if !has_target {
        return None;
    }
    let e = expand::expand(arena, g);
    let terms: Vec<ExprId> = match arena.node(e) {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![e],
    };
    let mut sq: Vec<ExprId> = Vec::new();
    let mut lin: Vec<ExprId> = Vec::new();
    let mut cst: Vec<ExprId> = Vec::new();
    let mut rest: Vec<ExprId> = Vec::new();
    let order_two = arena.int(2);
    for t in terms {
        if !walk::contains(arena, t, u) {
            cst.push(t);
            continue;
        }
        let factors: Vec<ExprId> = match arena.node(t) {
            ExprNode::Mul(fs) => fs.to_vec(),
            _ => vec![t],
        };
        let (dep, coef): (Vec<ExprId>, Vec<ExprId>) = factors
            .into_iter()
            .partition(|&f| walk::contains(arena, f, u));
        let c = arena.mul(&coef);
        let mut logs: Vec<ExprId> = Vec::new();
        let mut li2: Option<ExprId> = None;
        let mut ok = true;
        for &d in &dep {
            match arena.node(d).clone() {
                ExprNode::Ln(a) => logs.push(a),
                ExprNode::Pow(b, n)
                    if matches!(arena.node(b), ExprNode::Ln(_))
                        && arena
                            .as_num(n)
                            .is_some_and(|q| *q == Q::from_integer(2.into())) =>
                {
                    let ExprNode::Ln(a) = *arena.node(b) else {
                        return None;
                    };
                    logs.push(a);
                    logs.push(a);
                }
                ExprNode::Lib(crate::base::libfn::LibFn::PolyLog, args)
                    if args.len() == 2 && args[0] == order_two && li2.is_none() =>
                {
                    li2 = Some(args[1]);
                }
                _ => ok = false,
            }
        }
        if !ok || (li2.is_some() && !logs.is_empty()) || logs.len() > 2 {
            rest.push(t);
            continue;
        }
        if let Some(z) = li2 {
            // −π²/6 − ½(ℓ + μ)² with μ the shift of ln(−C).
            let neg_z = arena.neg(z);
            let neg_z = safe_eval(arena, neg_z);
            let mu = asymptotic_log_shift(arena, neg_z, u, true)?;
            let minus_half = arena.rational(-1, 2);
            sq.push(arena.mul(&[c, minus_half]));
            let minus_one = arena.int(-1);
            lin.push(arena.mul(&[c, minus_one, mu]));
            let pi = arena.pi();
            let two = arena.int(2);
            let pi2 = arena.pow(pi, two);
            let sixth = arena.rational(-1, 6);
            let mu2 = arena.pow(mu, two);
            let g0 = arena.mul(&[sixth, pi2]);
            let g1 = arena.mul(&[minus_half, mu2]);
            let g = arena.add(&[g0, g1]);
            cst.push(arena.mul(&[c, g]));
            continue;
        }
        let lambdas: Vec<ExprId> = logs
            .iter()
            .map(|&a| asymptotic_log_shift(arena, a, u, false))
            .collect::<Option<_>>()?;
        match lambdas.len() {
            1 => {
                lin.push(c);
                cst.push(arena.mul(&[c, lambdas[0]]));
            }
            2 => {
                sq.push(c);
                let s = arena.add(&[lambdas[0], lambdas[1]]);
                lin.push(arena.mul(&[c, s]));
                cst.push(arena.mul(&[c, lambdas[0], lambdas[1]]));
            }
            _ => rest.push(t),
        }
    }
    let vanishes = |arena: &mut Arena, parts: &[ExprId]| -> bool {
        let s = arena.add(parts);
        let s = safe_eval(arena, s);
        if arena.is_zero_structural(s) {
            return true;
        }
        let s = expand::expand(arena, s);
        let s = safe_eval(arena, s);
        arena.is_zero_structural(s)
            || (walk::free_symbols(arena, s).is_empty()
                && evalf::evalf_complex(arena, s, 30)
                    .ok()
                    .and_then(|v| evalf::abs_to_f64(&v))
                    .is_some_and(|m| m < 1e-25))
    };
    if !vanishes(arena, &sq) || !vanishes(arena, &lin) {
        return None;
    }
    if !rest.is_empty() {
        let r = arena.add(&rest);
        match compositional_limit(arena, r, u) {
            LimVal::Value(Extended::Finite(v)) => cst.push(v),
            _ => return None,
        }
    }
    let total = arena.add(&cst);
    let total = safe_eval(arena, total);
    is_finite_value(arena, total, u).then_some(total)
}

/// `λ` with `ln(a·u + b) = ln u + λ + o(1)` as `u → +∞`, for a numeric
/// slope `a ≠ 0` and a constant `b`: `ln a` off the negative axis; on it
/// `ln|a| ± iπ` above or below the axis by the sign of `im b` (`+iπ` on the
/// axis itself, the principal value).  With `avoid_positive_axis` (the
/// argument `−C` of a `Li₂(C)` inversion), `None` when `a·u + b` runs along
/// the negative axis: `C` on `Li₂`'s cut.
fn asymptotic_log_shift(
    arena: &mut Arena,
    arg: ExprId,
    u: ExprId,
    avoid_positive_axis: bool,
) -> Option<ExprId> {
    let LinearCoeffs { slope, intercept } = calculus_util::linear_coeffs_exact(arena, arg, u)?;
    if walk::contains(arena, slope, u)
        || walk::contains(arena, intercept, u)
        || !walk::free_symbols(arena, slope).is_empty()
    {
        return None;
    }
    let a = evalf::evalf_complex64(arena, slope).ok()?;
    if a.norm().is_nan() || a.norm() <= 0.0 {
        return None;
    }
    if a.re < 0.0 && a.im.abs() <= 1e-12 * a.norm() {
        if !walk::free_symbols(arena, intercept).is_empty() {
            return None;
        }
        let b = evalf::evalf_complex64(arena, intercept).ok()?;
        let on_axis = b.im.abs() <= 1e-12 * b.norm().max(1.0);
        if avoid_positive_axis && on_axis {
            return None;
        }
        let abs_a = arena.neg(slope);
        let ln_abs = arena.ln(abs_a);
        let pi = arena.pi();
        let i = arena.i_unit();
        let sg = arena.int(if b.im < 0.0 && !on_axis { -1 } else { 1 });
        let i_pi = arena.mul(&[sg, i, pi]);
        return Some(arena.add(&[ln_abs, i_pi]));
    }
    Some(arena.ln(slope))
}

/// Does `g` contain `sin`/`cos`/`tan` of an argument that does not
/// converge as `u → ∞`?
fn has_unbounded_oscillation(arena: &mut Arena, g: ExprId, u: ExprId) -> bool {
    let order = walk::post_order_ids(arena, g);
    for id in order {
        let arg = match arena.node(id) {
            ExprNode::Sin(a) | ExprNode::Cos(a) | ExprNode::Tan(a) => *a,
            _ => continue,
        };
        if !walk::contains(arena, arg, u) {
            continue;
        }
        match compositional_limit(arena, arg, u) {
            LimVal::Value(Extended::Finite(_)) => {}
            _ => return true,
        }
    }
    false
}

/// Check a claimed `±∞` limit of `g(u)` against samples: `|g|` must grow
/// monotonically with the claimed sign at `u = 10², 10⁴, 10⁶`.
fn numeric_growth_sanity(arena: &mut Arena, g: ExprId, u: ExprId, sign: i8) -> bool {
    let u_name = match arena.node(u) {
        ExprNode::Symbol(sid) => arena.symbol_name(*sid).to_string(),
        _ => return true,
    };
    let Ok(func) = crate::output::lambdify::compile(arena, g, &[&u_name]) else {
        return true;
    };
    let samples = [func(&[1e2]), func(&[1e4]), func(&[1e6])];
    if samples.iter().any(|v| v.is_nan()) {
        return true;
    }
    let mut prev = 0.0f64;
    for v in samples {
        if v.is_infinite() {
            return (v > 0.0) == (sign > 0);
        }
        if (v > 0.0) != (sign > 0) || v.abs() <= prev {
            return false;
        }
        prev = v.abs();
    }
    true
}

/// Check a claimed finite limit `l` of `g(u)` at `+∞` against samples at
/// `u = 10³` and `u = 10⁶`.  Accepts when either the far sample is close
/// or the discrepancy is clearly shrinking; rejects obvious contradictions.
fn numeric_limit_sanity(arena: &mut Arena, g: ExprId, u: ExprId, l: ExprId) -> bool {
    let u_name = match arena.node(u) {
        ExprNode::Symbol(sid) => arena.symbol_name(*sid).to_string(),
        _ => return true,
    };
    let Some(lv) = calculus_util::expr_to_f64(arena, l) else {
        return true;
    };
    let Ok(func) = crate::output::lambdify::compile(arena, g, &[&u_name]) else {
        return true;
    };
    let g3 = func(&[1e3]);
    let g6 = func(&[1e6]);
    if !g3.is_finite() || !g6.is_finite() {
        return true;
    }
    let scale = lv.abs().max(1.0);
    let d3 = (g3 - lv).abs();
    let d6 = (g6 - lv).abs();
    if d6 <= 1e-3 * scale {
        return true;
    }
    d6 < 0.5 * d3
}

/// Value classes for the compositional limit evaluator.
#[derive(Clone, Copy, Debug)]
enum LimVal {
    /// A limit on the extended line: a finite value or `±∞`.
    Value(Extended<ExprId>),
    /// Bounded but not convergent (e.g. `sin(u)`).
    Bounded,
    Unknown,
}

/// Bottom-up limit evaluation over the expression DAG (explicit
/// post-order, no recursion).  Handles sums, products, powers, exp, ln,
/// atan/erf/tanh saturation, and bounded oscillation.
fn compositional_limit(arena: &mut Arena, g: ExprId, u: ExprId) -> LimVal {
    let order = walk::post_order_ids(arena, g);
    let mut vals: FxHashMap<ExprId, LimVal> = FxHashMap::default();
    // Nodes that tend to 0 faster than any power of `u` grows
    // ([`exponentially_small`]).
    let mut small: FxHashSet<ExprId> = FxHashSet::default();

    // Leading terms `c·u^d` of the nodes that are algebraic in `u`.
    let mut lead: FxHashMap<ExprId, (Q, ExprId)> = FxHashMap::default();

    for id in order {
        let node = arena.node(id).clone();
        if exponentially_small(arena, &node, &vals, &small, u) {
            small.insert(id);
        }
        if let Some(t) = leading_term(arena, id, &node, &lead, u) {
            lead.insert(id, t);
        }
        let v = if id == u {
            LimVal::Value(Extended::PosInf)
        } else if !walk::contains(arena, id, u) {
            LimVal::Value(Extended::Finite(id))
        } else {
            match node {
                ExprNode::Mul(_) | ExprNode::Add(_) if small.contains(&id) => {
                    LimVal::Value(Extended::Finite(arena.zero()))
                }
                ExprNode::Add(children) => {
                    let mut finite: Vec<ExprId> = Vec::new();
                    let mut pos = 0usize;
                    let mut neg = 0usize;
                    let mut bounded = false;
                    let mut unknown = false;
                    let (grouped, group_value) = linear_log_group(arena, &children, u);
                    if let Some(v) = group_value {
                        finite.push(v);
                    }
                    for c in children.iter() {
                        if grouped.contains(c) {
                            continue;
                        }
                        match vals.get(c).copied().unwrap_or(LimVal::Unknown) {
                            LimVal::Value(Extended::Finite(v)) => finite.push(v),
                            LimVal::Value(Extended::PosInf) => pos += 1,
                            LimVal::Value(Extended::NegInf) => neg += 1,
                            LimVal::Bounded => bounded = true,
                            LimVal::Unknown => unknown = true,
                        }
                    }
                    if unknown || (pos > 0 && neg > 0) {
                        LimVal::Unknown
                    } else if pos > 0 {
                        LimVal::Value(Extended::PosInf)
                    } else if neg > 0 {
                        LimVal::Value(Extended::NegInf)
                    } else if bounded {
                        LimVal::Bounded
                    } else {
                        let s = arena.add(&finite);
                        LimVal::Value(Extended::Finite(safe_eval(arena, s)))
                    }
                }
                ExprNode::Mul(children) => {
                    let mut finite: Vec<ExprId> = Vec::new();
                    let mut infs = 0usize;
                    let mut neg_infs = 0usize;
                    let mut bounded = false;
                    let mut unknown = false;
                    for c in children.iter() {
                        match vals.get(c).copied().unwrap_or(LimVal::Unknown) {
                            LimVal::Value(Extended::Finite(v)) => finite.push(v),
                            LimVal::Value(Extended::PosInf) => infs += 1,
                            LimVal::Value(Extended::NegInf) => {
                                infs += 1;
                                neg_infs += 1;
                            }
                            LimVal::Bounded => bounded = true,
                            LimVal::Unknown => unknown = true,
                        }
                    }
                    if unknown {
                        LimVal::Unknown
                    } else {
                        let prod = if finite.is_empty() {
                            arena.one()
                        } else {
                            let m = arena.mul(&finite);
                            safe_eval(arena, m)
                        };
                        let psign = sign_of(arena, prod);
                        if infs > 0 {
                            match psign {
                                Some(Ordering::Equal) | None => LimVal::Unknown,
                                Some(s) => {
                                    if bounded {
                                        LimVal::Unknown
                                    } else {
                                        let negative = (s == Ordering::Less) ^ (neg_infs % 2 == 1);
                                        if negative {
                                            LimVal::Value(Extended::NegInf)
                                        } else {
                                            LimVal::Value(Extended::PosInf)
                                        }
                                    }
                                }
                            }
                        } else if bounded {
                            if psign == Some(Ordering::Equal) {
                                LimVal::Value(Extended::Finite(arena.zero()))
                            } else {
                                LimVal::Bounded
                            }
                        } else {
                            LimVal::Value(Extended::Finite(prod))
                        }
                    }
                }
                ExprNode::Neg(inner) => {
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Value(Extended::Finite(v)) => {
                            let n = arena.neg(v);
                            LimVal::Value(Extended::Finite(safe_eval(arena, n)))
                        }
                        LimVal::Value(Extended::PosInf) => LimVal::Value(Extended::NegInf),
                        LimVal::Value(Extended::NegInf) => LimVal::Value(Extended::PosInf),
                        other => other,
                    }
                }
                ExprNode::Pow(base, exp) => {
                    let bv = vals.get(&base).copied().unwrap_or(LimVal::Unknown);
                    let ev = vals.get(&exp).copied().unwrap_or(LimVal::Unknown);
                    pow_limit(arena, base, exp, bv, ev, u)
                }
                ExprNode::Exp(inner) => {
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Value(Extended::Finite(v)) => {
                            let e = arena.exp(v);
                            LimVal::Value(Extended::Finite(safe_eval(arena, e)))
                        }
                        LimVal::Value(Extended::PosInf) => LimVal::Value(Extended::PosInf),
                        LimVal::Value(Extended::NegInf) => {
                            LimVal::Value(Extended::Finite(arena.zero()))
                        }
                        LimVal::Bounded => LimVal::Bounded,
                        LimVal::Unknown => LimVal::Unknown,
                    }
                }
                ExprNode::Ln(inner) => match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                    LimVal::Value(Extended::Finite(v)) => match sign_of(arena, v) {
                        Some(Ordering::Equal) => LimVal::Value(Extended::NegInf),
                        Some(_) => {
                            let l = arena.ln(v);
                            LimVal::Value(Extended::Finite(safe_eval(arena, l)))
                        }
                        None => LimVal::Unknown,
                    },
                    LimVal::Value(Extended::PosInf) => LimVal::Value(Extended::PosInf),
                    _ => LimVal::Unknown,
                },
                ExprNode::Atan(inner) => match vals.get(&inner).copied().unwrap_or(LimVal::Unknown)
                {
                    LimVal::Value(Extended::Finite(v)) => {
                        let a = arena.atan(v);
                        LimVal::Value(Extended::Finite(safe_eval(arena, a)))
                    }
                    LimVal::Value(Extended::PosInf) => {
                        let half_pi = arena.rational(1, 2);
                        let pi = arena.pi();
                        LimVal::Value(Extended::Finite(arena.mul(&[half_pi, pi])))
                    }
                    LimVal::Value(Extended::NegInf) => {
                        let neg_half = arena.rational(-1, 2);
                        let pi = arena.pi();
                        LimVal::Value(Extended::Finite(arena.mul(&[neg_half, pi])))
                    }
                    _ => LimVal::Unknown,
                },
                ExprNode::Tanh(inner) | ExprNode::Erf(inner) => {
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Value(Extended::Finite(v)) => {
                            let r = if matches!(node, ExprNode::Tanh(_)) {
                                arena.tanh(v)
                            } else {
                                arena.erf(v)
                            };
                            LimVal::Value(Extended::Finite(safe_eval(arena, r)))
                        }
                        LimVal::Value(Extended::PosInf) => {
                            LimVal::Value(Extended::Finite(arena.one()))
                        }
                        LimVal::Value(Extended::NegInf) => {
                            LimVal::Value(Extended::Finite(arena.neg_one()))
                        }
                        _ => LimVal::Unknown,
                    }
                }
                ExprNode::Erfc(inner) => match vals.get(&inner).copied().unwrap_or(LimVal::Unknown)
                {
                    LimVal::Value(Extended::Finite(v)) => {
                        let r = arena.erfc(v);
                        LimVal::Value(Extended::Finite(safe_eval(arena, r)))
                    }
                    LimVal::Value(Extended::PosInf) => {
                        LimVal::Value(Extended::Finite(arena.zero()))
                    }
                    LimVal::Value(Extended::NegInf) => {
                        LimVal::Value(Extended::Finite(arena.int(2)))
                    }
                    _ => LimVal::Unknown,
                },
                ExprNode::Sin(inner) | ExprNode::Cos(inner) => {
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Value(Extended::Finite(v)) => {
                            let r = if matches!(node, ExprNode::Sin(_)) {
                                arena.sin(v)
                            } else {
                                arena.cos(v)
                            };
                            LimVal::Value(Extended::Finite(safe_eval(arena, r)))
                        }
                        LimVal::Value(Extended::PosInf)
                        | LimVal::Value(Extended::NegInf)
                        | LimVal::Bounded => LimVal::Bounded,
                        LimVal::Unknown => LimVal::Unknown,
                    }
                }
                ExprNode::Sinh(inner) | ExprNode::Asinh(inner) => {
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Value(Extended::Finite(v)) => {
                            let r = if matches!(node, ExprNode::Sinh(_)) {
                                arena.sinh(v)
                            } else {
                                arena.asinh(v)
                            };
                            LimVal::Value(Extended::Finite(safe_eval(arena, r)))
                        }
                        LimVal::Value(Extended::PosInf) => LimVal::Value(Extended::PosInf),
                        LimVal::Value(Extended::NegInf) => LimVal::Value(Extended::NegInf),
                        _ => LimVal::Unknown,
                    }
                }
                ExprNode::Cosh(inner) => match vals.get(&inner).copied().unwrap_or(LimVal::Unknown)
                {
                    LimVal::Value(Extended::Finite(v)) => {
                        let r = arena.cosh(v);
                        LimVal::Value(Extended::Finite(safe_eval(arena, r)))
                    }
                    LimVal::Value(Extended::PosInf) | LimVal::Value(Extended::NegInf) => {
                        LimVal::Value(Extended::PosInf)
                    }
                    _ => LimVal::Unknown,
                },
                ExprNode::Abs(inner) => {
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Value(Extended::Finite(v)) => {
                            let r = arena.abs(v);
                            LimVal::Value(Extended::Finite(safe_eval(arena, r)))
                        }
                        LimVal::Value(Extended::PosInf) | LimVal::Value(Extended::NegInf) => {
                            LimVal::Value(Extended::PosInf)
                        }
                        LimVal::Bounded => LimVal::Bounded,
                        LimVal::Unknown => LimVal::Unknown,
                    }
                }
                ExprNode::Sign(inner) => match vals.get(&inner).copied().unwrap_or(LimVal::Unknown)
                {
                    LimVal::Value(Extended::PosInf) => LimVal::Value(Extended::Finite(arena.one())),
                    LimVal::Value(Extended::NegInf) => {
                        LimVal::Value(Extended::Finite(arena.neg_one()))
                    }
                    LimVal::Value(Extended::Finite(v)) => match sign_of(arena, v) {
                        Some(Ordering::Greater) => LimVal::Value(Extended::Finite(arena.one())),
                        Some(Ordering::Less) => LimVal::Value(Extended::Finite(arena.neg_one())),
                        _ => LimVal::Unknown,
                    },
                    _ => LimVal::Unknown,
                },
                ExprNode::Acosh(inner) | ExprNode::Gamma(inner) => {
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Value(Extended::PosInf) => LimVal::Value(Extended::PosInf),
                        LimVal::Value(Extended::Finite(v)) if is_positive(arena, v) => {
                            let r = if matches!(node, ExprNode::Acosh(_)) {
                                arena.acosh(v)
                            } else {
                                arena.gamma(v)
                            };
                            LimVal::Value(Extended::Finite(safe_eval(arena, r)))
                        }
                        _ => LimVal::Unknown,
                    }
                }
                ExprNode::Asin(inner) | ExprNode::Acos(inner) | ExprNode::Atanh(inner) => {
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Value(Extended::Finite(v)) => {
                            let r = match node {
                                ExprNode::Asin(_) => arena.asin(v),
                                ExprNode::Acos(_) => arena.acos(v),
                                _ => arena.atanh(v),
                            };
                            let r = safe_eval(arena, r);
                            if is_finite_value(arena, r, u) {
                                LimVal::Value(Extended::Finite(r))
                            } else {
                                LimVal::Unknown
                            }
                        }
                        _ => LimVal::Unknown,
                    }
                }
                ExprNode::Si(inner) | ExprNode::Ci(inner) | ExprNode::Ei(inner) => {
                    let kind = match node {
                        ExprNode::Si(_) => SineIntegral::Si,
                        ExprNode::Ci(_) => SineIntegral::Ci,
                        _ => SineIntegral::Ei,
                    };
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Value(Extended::Finite(v)) => integral_function_at(arena, kind, v),
                        // A complex slope leaves the argument's limit
                        // unknown here; the linear form decides.
                        _ => integral_function_at_infinity(arena, kind, inner, u),
                    }
                }
                _ => LimVal::Unknown,
            }
        };
        let v = match (v, lead.get(&id)) {
            (LimVal::Unknown, Some((d, c))) => leading_term_limit(arena, d, *c),
            (v, _) => v,
        };
        vals.insert(id, v);
    }
    vals.get(&g).copied().unwrap_or(LimVal::Unknown)
}

/// The leading term `c·u^d` (`u → +∞`) of the node `id` (children done) when
/// it is algebraic in `u`: a constant, `u`, a sum (the top degree's
/// coefficients must not cancel structurally), a product, a constant
/// power (of a positive leading coefficient unless integer).
fn leading_term(
    arena: &mut Arena,
    id: ExprId,
    node: &ExprNode,
    lead: &FxHashMap<ExprId, (Q, ExprId)>,
    u: ExprId,
) -> Option<(Q, ExprId)> {
    if id == u {
        return Some((Q::one(), arena.one()));
    }
    if !walk::contains(arena, id, u) {
        return Some((Q::zero(), id));
    }
    match node {
        ExprNode::Add(children) => {
            let terms: Vec<(Q, ExprId)> = children
                .iter()
                .map(|c| lead.get(c).cloned())
                .collect::<Option<_>>()?;
            let top = terms.iter().map(|(d, _)| d.clone()).max()?;
            let coeffs: Vec<ExprId> = terms
                .iter()
                .filter(|(d, _)| *d == top)
                .map(|&(_, c)| c)
                .collect();
            let c = arena.add(&coeffs);
            let c = safe_eval(arena, c);
            if arena.is_zero_structural(c) || sign_of(arena, c) == Some(Ordering::Equal) {
                return None;
            }
            Some((top, c))
        }
        ExprNode::Mul(children) => {
            let mut d = Q::zero();
            let mut cs: Vec<ExprId> = Vec::with_capacity(children.len());
            for c in children.iter() {
                let (dc, cc) = lead.get(c).cloned()?;
                d += dc;
                cs.push(cc);
            }
            let c = arena.mul(&cs);
            Some((d, safe_eval(arena, c)))
        }
        ExprNode::Neg(inner) => {
            let (d, c) = lead.get(inner).cloned()?;
            let n = arena.neg(c);
            Some((d, safe_eval(arena, n)))
        }
        ExprNode::Pow(b, e) => {
            if walk::contains(arena, *e, u) {
                return None;
            }
            let r = arena.as_num(*e)?.clone();
            let (db, cb) = lead.get(b).cloned()?;
            if !r.is_integer() && sign_of(arena, cb) != Some(Ordering::Greater) {
                return None;
            }
            let p = arena.pow(cb, *e);
            Some((db * r, safe_eval(arena, p)))
        }
        _ => None,
    }
}

/// The limit at `+∞` of a leading term `c·u^d`.
fn leading_term_limit(arena: &mut Arena, d: &Q, c: ExprId) -> LimVal {
    if d.is_negative() {
        return LimVal::Value(Extended::Finite(arena.zero()));
    }
    if d.is_zero() {
        return LimVal::Value(Extended::Finite(c));
    }
    match sign_of(arena, c) {
        Some(Ordering::Greater) => LimVal::Value(Extended::PosInf),
        Some(Ordering::Less) => LimVal::Value(Extended::NegInf),
        _ => LimVal::Unknown,
    }
}

/// Does the node `node` (its children already visited) tend to 0 as `u → +∞`
/// faster than any power of `u` grows: `e^g` for an algebraic `g → −∞` (so
/// `|e^g| ≤ e^{−c·u^ε}`), a product of such a factor with factors that grow at
/// most like a power of `u` ([`at_most_polynomial`]), a sum of such terms?
/// Such a product tends to 0.  The compositional rules saw `∞·0` in
/// `x²·e^{−(x−μ)²/(2σ²)}` and in `x·(e^{−x}·sin x + e^{−x}·cos x)`, and the limit
/// engine is not consulted with parameters or oscillation:
/// `∫₀^∞ x³·e^{−x}·sin x dx` and the moments of a normal density with symbolic
/// `μ`, `σ` were refused up to 0.46 (SymPy's test_meijerint).
fn exponentially_small(
    arena: &Arena,
    node: &ExprNode,
    vals: &FxHashMap<ExprId, LimVal>,
    small: &FxHashSet<ExprId>,
    u: ExprId,
) -> bool {
    match node {
        ExprNode::Exp(g) => {
            matches!(vals.get(g), Some(LimVal::Value(Extended::NegInf)))
                && algebraic_in(arena, *g, u)
        }
        ExprNode::Mul(children) => {
            children.iter().any(|c| small.contains(c))
                && children.iter().all(|&c| {
                    small.contains(&c)
                        || !walk::contains(arena, c, u)
                        || at_most_polynomial(arena, c, u)
                })
        }
        ExprNode::Add(children) => children.iter().all(|c| small.contains(c)),
        ExprNode::Neg(c) => small.contains(c),
        _ => false,
    }
}

/// Is every node of `e` that depends on `u` a sum, product or constant
/// power of `u`?
fn algebraic_in(arena: &Arena, e: ExprId, u: ExprId) -> bool {
    walk::post_order_ids(arena, e).into_iter().all(|id| {
        !walk::contains(arena, id, u)
            || id == u
            || match arena.node(id) {
                ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => true,
                ExprNode::Pow(_, ex) => !walk::contains(arena, *ex, u),
                _ => false,
            }
    })
}

/// Is `|e|` bounded by a power of `u` as `u → +∞`: built from `u`, sums,
/// products, constant powers of algebraic bases, logarithms, and the
/// bounded functions `sin`, `cos`, `atan`, `tanh`, `erf`, `erfc`, `sign`,
/// `Si` (of anything), with no exponential?
fn at_most_polynomial(arena: &Arena, e: ExprId, u: ExprId) -> bool {
    walk::post_order_ids(arena, e).into_iter().all(|id| {
        if !walk::contains(arena, id, u) || id == u {
            return true;
        }
        match arena.node(id) {
            ExprNode::Add(_)
            | ExprNode::Mul(_)
            | ExprNode::Neg(_)
            | ExprNode::Sin(_)
            | ExprNode::Cos(_)
            | ExprNode::Atan(_)
            | ExprNode::Tanh(_)
            | ExprNode::Erf(_)
            | ExprNode::Erfc(_)
            | ExprNode::Sign(_)
            | ExprNode::Si(_)
            | ExprNode::Abs(_) => true,
            ExprNode::Ln(g) => algebraic_in(arena, *g, u),
            ExprNode::Pow(b, ex) => {
                !walk::contains(arena, *ex, u)
                    && (algebraic_in(arena, *b, u)
                        || arena.as_num(*ex).is_some_and(|r| !r.is_negative()))
            }
            _ => false,
        }
    })
}

/// The terms `cⱼ·ln(aⱼ·u + bⱼ)` among `children` (constants `cⱼ`, numeric
/// `aⱼ ≠ 0`, constant `bⱼ`), each of modulus tending to `∞`, and the limit
/// of their sum as `u → +∞` when the coefficients `cⱼ` add up to 0:
/// `ln(a·u + b) = ln a + ln u + o(1)` (principal branches, `u > 0`) off the
/// negative axis, and `ln|a| + ln u ± iπ` along it, above or below by the
/// sign of `im b` (`+iπ` on the cut itself), so the `ln u` cancel and the
/// limit is `Σ cⱼ·ln aⱼ` (with those `±iπ`).  The continuous antiderivative
/// of `∫ e^{−x}·sin x/(x² + 1) dx` holds `ln(x + i) − ln((−1 + i)·x − 1 − i)`,
/// each of whose limits is a complex infinity (0.46 refused
/// `∫₀^∞ e^{−x}·sin x/(x² + 1) dx`).  Empty and `None` otherwise.
fn linear_log_group(
    arena: &mut Arena,
    children: &[ExprId],
    u: ExprId,
) -> (Vec<ExprId>, Option<ExprId>) {
    let none = (Vec::new(), None);
    let mut members: Vec<ExprId> = Vec::new();
    let mut coeffs: Vec<ExprId> = Vec::new();
    let mut values: Vec<ExprId> = Vec::new();
    for &c in children {
        let (coeff, log) = match arena.node(c).clone() {
            ExprNode::Ln(_) => (arena.one(), c),
            ExprNode::Mul(fs) => {
                let (logs, rest): (Vec<ExprId>, Vec<ExprId>) = fs
                    .iter()
                    .partition(|&&f| matches!(arena.node(f), ExprNode::Ln(_)));
                if logs.len() != 1 || rest.iter().any(|&f| walk::contains(arena, f, u)) {
                    continue;
                }
                (arena.mul(&rest), logs[0])
            }
            _ => continue,
        };
        let ExprNode::Ln(arg) = *arena.node(log) else {
            continue;
        };
        // `ln|a·u + b|` (real `a`, `b`): `ln|a| + ln u + o(1)`.
        let (arg, absolute) = match *arena.node(arg) {
            ExprNode::Abs(inner) => (inner, true),
            _ => (arg, false),
        };
        let Some(LinearCoeffs { slope, intercept }) =
            calculus_util::linear_coeffs_exact(arena, arg, u)
        else {
            continue;
        };
        if absolute {
            if walk::contains(arena, slope, u)
                || walk::contains(arena, intercept, u)
                || !walk::free_symbols(arena, slope).is_empty()
                || !is_real(arena, slope)
                || !is_real(arena, intercept)
            {
                continue;
            }
            let abs_slope = arena.abs(slope);
            let value = arena.ln(abs_slope);
            members.push(c);
            coeffs.push(coeff);
            values.push(arena.mul(&[coeff, value]));
            continue;
        }
        if walk::contains(arena, slope, u)
            || walk::contains(arena, intercept, u)
            || !walk::free_symbols(arena, slope).is_empty()
        {
            continue;
        }
        let Ok(a) = evalf::evalf_complex64(arena, slope) else {
            continue;
        };
        if a.norm().is_nan() || a.norm() <= 0.0 {
            continue;
        }
        let tiny = 1e-12 * a.norm();
        let value = if a.re < 0.0 && a.im.abs() <= tiny {
            // Along the negative axis: ln|a| + ln u ± iπ by the side of b.
            let below = walk::free_symbols(arena, intercept).is_empty()
                && evalf::evalf_complex64(arena, intercept)
                    .is_ok_and(|b| b.im < -1e-12 * b.norm().max(1.0));
            let above_or_on = walk::free_symbols(arena, intercept).is_empty()
                && evalf::evalf_complex64(arena, intercept)
                    .is_ok_and(|b| b.im >= -1e-12 * b.norm().max(1.0));
            if !below && !above_or_on {
                return none;
            }
            let abs_a = arena.neg(slope);
            let ln_abs = arena.ln(abs_a);
            let pi = arena.pi();
            let i = arena.i_unit();
            let sg = arena.int(if below { -1 } else { 1 });
            let i_pi = arena.mul(&[sg, i, pi]);
            arena.add(&[ln_abs, i_pi])
        } else {
            arena.ln(slope)
        };
        members.push(c);
        coeffs.push(coeff);
        values.push(arena.mul(&[coeff, value]));
    }
    if members.len() < 2 {
        return none;
    }
    let total = arena.add(&coeffs);
    let total = safe_eval(arena, total);
    if !arena.is_zero_structural(total) {
        return none;
    }
    let sum = arena.add(&values);
    (members, Some(safe_eval(arena, sum)))
}

/// The exponential and trigonometric integrals that
/// [`compositional_limit`] knows at infinity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SineIntegral {
    Si,
    Ci,
    Ei,
}

/// `Si(v)`, `Ci(v)`, `Ei(v)` as the limit of `f(w)` for `w → v`: `Si` is
/// entire; `Ci` and `Ei` are continuous off their cut `(−∞, 0]`, so `v`
/// must be provably positive or provably non-real (a parameter-free value
/// whose imaginary part is not negligible).
fn integral_function_at(arena: &mut Arena, kind: SineIntegral, v: ExprId) -> LimVal {
    let off_cut = kind == SineIntegral::Si
        || sign_of(arena, v) == Some(Ordering::Greater)
        || (walk::free_symbols(arena, v).is_empty()
            && evalf::evalf_complex64(arena, v)
                .is_ok_and(|z| z.im.is_finite() && z.im.abs() > 1e-9 * z.norm().max(1.0)));
    if !off_cut {
        return LimVal::Unknown;
    }
    let r = match kind {
        SineIntegral::Si => arena.si(v),
        SineIntegral::Ci => arena.ci(v),
        SineIntegral::Ei => arena.ei(v),
    };
    LimVal::Value(Extended::Finite(safe_eval(arena, r)))
}

/// The sign of `im c` for a constant `c`: `0` when `c` is provably real,
/// otherwise from the assumptions on the imaginary part of the real/imaginary
/// split, or numerically for a parameter-free `c` whose imaginary part is
/// not negligible.
///
/// A single atom when decided, otherwise [`SignSet::REAL`] (an imaginary
/// part is real; its sign is unknown).
fn imaginary_sign(arena: &mut Arena, c: ExprId) -> SignSet {
    if is_real(arena, c) {
        return SignSet::ZERO;
    }
    if walk::free_symbols(arena, c).is_empty() {
        let Ok(z) = evalf::evalf_complex64(arena, c) else {
            return SignSet::REAL;
        };
        if !z.im.is_finite() || z.im.abs() <= 1e-9 * z.norm().max(1.0) {
            return SignSet::REAL;
        }
        return if z.im > 0.0 {
            SignSet::POSITIVE
        } else {
            SignSet::NEGATIVE
        };
    }
    let parts = crate::base::complex::decompose(arena, c);
    if !parts.exact {
        return SignSet::REAL;
    }
    let im = safe_eval(arena, parts.im);
    sign_of(arena, im).map_or(SignSet::REAL, SignSet::from_ordering)
}

/// `lim_{u → ∞} f(αu + β)` for `f = Si, Ci, Ei`, a real slope `α ≠ 0` of
/// known sign and a constant `β` (the argument runs to `±∞` along a
/// horizontal line `im w = im β`): `Si → ±π/2`; `Ci → 0` at `+∞` and `±iπ`
/// at `−∞` above/below the axis (`+iπ` on the cut itself, the principal
/// value `Ci(−x) = Ci(x) + iπ`); `Ei → ±iπ` at `−∞` above/below the axis and
/// `0` on it, `+∞` at `+∞` on the axis (a complex infinity off it:
/// unknown).  Before 0.46 these were unknown here and in the limit engine,
/// so `∫₀^∞ sin x/(x² + 1) dx` — an antiderivative with `Si(x ± i)`,
/// `Ci(x ± i)` — was refused.  mpmath: `ci(-200+0.1j)` = `−0.0044 +
/// 3.14135i`, `ei(-200-3j)` = `−iπ + O(10⁻⁸⁹)`.
fn integral_function_at_infinity(
    arena: &mut Arena,
    kind: SineIntegral,
    w: ExprId,
    u: ExprId,
) -> LimVal {
    let Some(LinearCoeffs { slope, intercept }) = calculus_util::linear_coeffs_exact(arena, w, u)
    else {
        return LimVal::Unknown;
    };
    if walk::contains(arena, slope, u) || walk::contains(arena, intercept, u) {
        return LimVal::Unknown;
    }
    if !is_real(arena, slope) {
        // `Ei(w)` along a ray with `Re w` not growing and `Im w → ±∞`:
        // `Ei(w) = −E₁(−w) ± iπ` there and `E₁(−w) → 0` (as for
        // `e^{−x}·sin x` over rational functions, `Ei((−1 ± i)x + β)`).
        if kind != SineIntegral::Ei {
            return LimVal::Unknown;
        }
        let parts = crate::base::complex::decompose(arena, slope);
        if !parts.exact {
            return LimVal::Unknown;
        }
        let re = safe_eval(arena, parts.re);
        let im = safe_eval(arena, parts.im);
        let re_sign = sign_of(arena, re);
        let im_sign = sign_of(arena, im);
        return match (re_sign, im_sign) {
            (
                Some(Ordering::Less | Ordering::Equal),
                Some(s @ (Ordering::Greater | Ordering::Less)),
            ) => {
                let pi = arena.pi();
                let i = arena.i_unit();
                let sg = arena.int(if s == Ordering::Greater { 1 } else { -1 });
                let v = arena.mul(&[sg, i, pi]);
                LimVal::Value(Extended::Finite(safe_eval(arena, v)))
            }
            _ => LimVal::Unknown,
        };
    }
    let up = match sign_of(arena, slope) {
        Some(Ordering::Greater) => true,
        Some(Ordering::Less) => false,
        _ => return LimVal::Unknown,
    };
    let pi = arena.pi();
    let i = arena.i_unit();
    let i_pi = arena.mul(&[i, pi]);
    let finite =
        |arena: &mut Arena, v: ExprId| LimVal::Value(Extended::Finite(safe_eval(arena, v)));
    match (kind, up) {
        (SineIntegral::Si, _) => {
            let half = arena.rational(if up { 1 } else { -1 }, 2);
            let v = arena.mul(&[half, pi]);
            finite(arena, v)
        }
        (SineIntegral::Ci, true) => LimVal::Value(Extended::Finite(arena.zero())),
        (SineIntegral::Ci, false) => match imaginary_sign(arena, intercept) {
            SignSet::NEGATIVE => {
                let v = arena.neg(i_pi);
                finite(arena, v)
            }
            SignSet::ZERO | SignSet::POSITIVE => finite(arena, i_pi),
            _ => LimVal::Unknown,
        },
        (SineIntegral::Ei, false) => match imaginary_sign(arena, intercept) {
            SignSet::NEGATIVE => {
                let v = arena.neg(i_pi);
                finite(arena, v)
            }
            SignSet::POSITIVE => finite(arena, i_pi),
            SignSet::ZERO => LimVal::Value(Extended::Finite(arena.zero())),
            _ => LimVal::Unknown,
        },
        (SineIntegral::Ei, true) => match imaginary_sign(arena, intercept) {
            SignSet::ZERO => LimVal::Value(Extended::PosInf),
            _ => LimVal::Unknown,
        },
    }
}

/// Limit of `base^exp` from the limits of its parts.
fn pow_limit(
    arena: &mut Arena,
    base: ExprId,
    exp: ExprId,
    bv: LimVal,
    ev: LimVal,
    u: ExprId,
) -> LimVal {
    let exp_has_u = walk::contains(arena, exp, u);
    let base_has_u = walk::contains(arena, base, u);
    if !exp_has_u {
        // Constant exponent.
        let Some(e_sign) = sign_of(arena, exp) else {
            return LimVal::Unknown;
        };
        let e_int_parity: Option<bool> = arena.as_num(exp).and_then(|r| {
            if r.is_integer() {
                Some(r.to_integer().to_i64().unwrap_or(1) % 2 == 0)
            } else {
                None
            }
        });
        return match bv {
            LimVal::Value(Extended::Finite(b)) => {
                let b_zero = sign_of(arena, b) == Some(Ordering::Equal);
                if b_zero {
                    // 0^e: 0 for e > 0, undefined/infinite otherwise.
                    return if e_sign == Ordering::Greater {
                        LimVal::Value(Extended::Finite(arena.zero()))
                    } else {
                        LimVal::Unknown
                    };
                }
                let p = arena.pow(b, exp);
                let p = safe_eval(arena, p);
                if is_finite_value(arena, p, u) {
                    LimVal::Value(Extended::Finite(p))
                } else {
                    LimVal::Unknown
                }
            }
            LimVal::Value(Extended::PosInf) => match e_sign {
                Ordering::Greater => LimVal::Value(Extended::PosInf),
                Ordering::Less => LimVal::Value(Extended::Finite(arena.zero())),
                Ordering::Equal => LimVal::Value(Extended::Finite(arena.one())),
            },
            LimVal::Value(Extended::NegInf) => match e_sign {
                Ordering::Greater => match e_int_parity {
                    Some(true) => LimVal::Value(Extended::PosInf),
                    Some(false) => LimVal::Value(Extended::NegInf),
                    None => LimVal::Unknown,
                },
                Ordering::Less => LimVal::Value(Extended::Finite(arena.zero())),
                Ordering::Equal => LimVal::Value(Extended::Finite(arena.one())),
            },
            LimVal::Bounded => {
                if e_sign == Ordering::Greater {
                    LimVal::Bounded
                } else {
                    LimVal::Unknown
                }
            }
            LimVal::Unknown => LimVal::Unknown,
        };
    }
    if !base_has_u {
        // Constant base c > 0: c^{g(u)}.
        let one = arena.one();
        let c_minus_1 = arena.sub(base, one);
        let c_minus_1 = safe_eval(arena, c_minus_1);
        let Some(cmp1) = sign_of(arena, c_minus_1) else {
            return LimVal::Unknown;
        };
        if !is_positive(arena, base) {
            return LimVal::Unknown;
        }
        return match (ev, cmp1) {
            (LimVal::Value(Extended::Finite(e)), _) => {
                let p = arena.pow(base, e);
                LimVal::Value(Extended::Finite(safe_eval(arena, p)))
            }
            (_, Ordering::Equal) => LimVal::Value(Extended::Finite(arena.one())),
            (LimVal::Value(Extended::PosInf), Ordering::Greater)
            | (LimVal::Value(Extended::NegInf), Ordering::Less) => LimVal::Value(Extended::PosInf),
            (LimVal::Value(Extended::PosInf), Ordering::Less)
            | (LimVal::Value(Extended::NegInf), Ordering::Greater) => {
                LimVal::Value(Extended::Finite(arena.zero()))
            }
            _ => LimVal::Unknown,
        };
    }
    LimVal::Unknown
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
// Special integrands: δ, H, |·|, sign, Piecewise, floor
// ═══════════════════════════════════════════════════════════════════════════

/// Top-level factor list of `f`.
fn factors_of(arena: &Arena, f: ExprId) -> Vec<ExprId> {
    match arena.node(f) {
        ExprNode::Mul(children) => children.to_vec(),
        _ => vec![f],
    }
}

/// Handle integrands containing distributions / piecewise constructs.
fn try_special_integrand(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    a: ExprId,
    b: ExprId,
    depth: usize,
) -> Option<Result<ExprId, SymplexError>> {
    let factors = factors_of(arena, f);

    // ── DiracDelta factor ───────────────────────────────────────────
    for (i, &fac) in factors.iter().enumerate() {
        if let ExprNode::DiracDelta(g) = arena.node(fac).clone()
            && walk::contains(arena, g, x)
        {
            let rest: Vec<ExprId> = factors
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &c)| c)
                .collect();
            let rest = if rest.is_empty() {
                arena.one()
            } else {
                arena.mul(&rest)
            };
            return Some(delta_integral(arena, g, rest, x, a, b));
        }
    }

    // ── Heaviside factor ────────────────────────────────────────────
    for (i, &fac) in factors.iter().enumerate() {
        if let ExprNode::Heaviside(g) = arena.node(fac).clone()
            && walk::contains(arena, g, x)
        {
            let rest: Vec<ExprId> = factors
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &c)| c)
                .collect();
            let rest = if rest.is_empty() {
                arena.one()
            } else {
                arena.mul(&rest)
            };
            return Some(heaviside_integral(arena, g, rest, x, a, b, depth));
        }
    }

    // ── Abs / Sign / Piecewise / Floor anywhere in the tree ─────────
    if contains_piecewise_like(arena, f, x) {
        return Some(split_piecewise_like(arena, f, x, a, b, depth));
    }
    None
}

/// `∫ₐᵇ rest(x)·δ(g(x)) dx` for linear `g`.
fn delta_integral(
    arena: &mut Arena,
    g: ExprId,
    rest: ExprId,
    x: ExprId,
    a: ExprId,
    b: ExprId,
) -> Result<ExprId, SymplexError> {
    let Some(LinearCoeffs {
        slope: alpha,
        intercept: beta,
    }) = calculus_util::linear_coeffs_exact(arena, g, x)
    else {
        return Err(failed("DiracDelta with a non-linear argument"));
    };
    if sign_of(arena, alpha) == Some(Ordering::Equal) {
        return Err(failed(
            "DiracDelta argument does not depend on the variable",
        ));
    }
    let neg_beta = arena.neg(beta);
    let c = arena.div(neg_beta, alpha);
    let c = safe_eval(arena, c);
    match position_in(arena, c, a, b) {
        Position::Inside => {
            let rc = subs::subs(arena, rest, x, c);
            let abs_alpha = arena.abs(alpha);
            let v = arena.div(rc, abs_alpha);
            Ok(safe_eval(arena, v))
        }
        Position::Below | Position::Above => Ok(arena.zero()),
        Position::AtLower | Position::AtUpper => Err(failed(
            "DiracDelta located exactly at an integration bound (value is convention-dependent)",
        )),
        Position::Unknown => Err(failed(
            "cannot decide whether the DiracDelta support lies inside the interval",
        )),
    }
}

/// `∫ₐᵇ rest(x)·H(g(x)) dx` for linear `g`: truncate the interval.
fn heaviside_integral(
    arena: &mut Arena,
    g: ExprId,
    rest: ExprId,
    x: ExprId,
    a: ExprId,
    b: ExprId,
    depth: usize,
) -> Result<ExprId, SymplexError> {
    let Some(LinearCoeffs {
        slope: alpha,
        intercept: beta,
    }) = calculus_util::linear_coeffs_exact(arena, g, x)
    else {
        return Err(failed("Heaviside with a non-linear argument"));
    };
    let alpha_sign = match sign_of(arena, alpha) {
        Some(Ordering::Equal) | None => {
            return Err(failed("Heaviside argument slope has unknown sign"));
        }
        Some(s) => s,
    };
    let neg_beta = arena.neg(beta);
    let c = arena.div(neg_beta, alpha);
    let c = safe_eval(arena, c);
    let pos = position_in(arena, c, a, b);
    let (lo, hi) = match (alpha_sign, pos) {
        // H(x − c): support is x > c.
        (Ordering::Greater, Position::Below | Position::AtLower) => (a, b),
        (Ordering::Greater, Position::Inside) => (c, b),
        (Ordering::Greater, Position::AtUpper | Position::Above) => return Ok(arena.zero()),
        // H(c − x): support is x < c.
        (Ordering::Less, Position::Above | Position::AtUpper) => (a, b),
        (Ordering::Less, Position::Inside) => (a, c),
        (Ordering::Less, Position::Below | Position::AtLower) => return Ok(arena.zero()),
        (_, Position::Unknown) => {
            return Err(failed(
                "cannot decide whether the Heaviside step lies inside the interval",
            ));
        }
        _ => return Err(failed("unexpected Heaviside configuration")),
    };
    integrate_range(arena, rest, x, lo, hi, depth + 1)
}

/// Does `f` contain `Abs`, `Sign`, `Piecewise`, `Floor`, `Ceiling`, or
/// `Heaviside`/`DiracDelta` nodes (not as top-level factors) depending on `x`?
fn contains_piecewise_like(arena: &Arena, f: ExprId, x: ExprId) -> bool {
    let mut stack = vec![f];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Abs(inner)
            | ExprNode::Sign(inner)
            | ExprNode::Floor(inner)
            | ExprNode::Ceiling(inner)
            | ExprNode::Heaviside(inner)
                if walk::contains(arena, *inner, x) =>
            {
                return true;
            }
            ExprNode::Piecewise(_) if walk::contains(arena, id, x) => {
                return true;
            }
            _ => {}
        }
        arena.node(id).for_each_child(|c| stack.push(c));
    }
    false
}

/// Split `[a, b]` at the kinks of `f` and rewrite each piece to a smooth
/// expression by resolving `Abs`, `Sign`, `Piecewise`, `Floor`, `Ceiling`
/// and `Heaviside` at a test point of the piece.
fn split_piecewise_like(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    a: ExprId,
    b: ExprId,
    depth: usize,
) -> Result<ExprId, SymplexError> {
    let range = numeric_range(arena, a, b);
    let scan = calculus_util::scan_breakpoints(arena, f, x, scan_range(a, b, range));
    if scan.opaque {
        return Err(failed(
            "the integrand contains a function whose breakpoints cannot be analysed",
        ));
    }

    // The pieces are where every argument of |·|, sign and H keeps its
    // sign: between its zeros (the kinks) and its own poles (`sign(1/x)`).
    // `solve` may miss zeros (it knew no inverse of `asinh`, and returns
    // one branch of `x·eˣ = c`); then each piece was resolved at its test
    // point as if the sign never changed, and `∫_{−1/2}^{3/2} |asinh x| dx`
    // came out as `∫ asinh x dx`.  The arguments are sampled as the
    // singularity path samples denominators, and an unexplained sign
    // change refuses the split.
    let args = kink_arguments(arena, f, x);
    let mut breaks: Vec<Breakpoint> = scan
        .points
        .iter()
        .filter(|bp| bp.kind == BreakKind::Kink)
        .cloned()
        .collect();
    let mut complete = scan.complete;
    for &g in &args {
        let s = calculus_util::scan_breakpoints(arena, g, x, scan_range(a, b, range));
        complete &= s.complete && !s.opaque;
        breaks.extend(
            s.points
                .into_iter()
                .filter(|bp| bp.kind == BreakKind::Singular),
        );
    }
    let known: Vec<f64> = breaks.iter().filter_map(|bp| bp.value).collect();
    let guard = match range {
        Some(r) => sign_changes_explained(arena, &args, x, r.lower, r.upper, &known),
        None => Truth::Unknown,
    };
    match guard {
        Truth::False => {
            return Err(failed(
                "an argument of |·|, sign or Heaviside changes sign inside the interval at a point the solver could not locate",
            ));
        }
        Truth::Unknown if !complete => {
            return Err(failed(
                "could not determine where the arguments of |·|, sign or Heaviside change sign",
            ));
        }
        _ => {}
    }
    // The guard watches the arguments of |·|, sign and H only: the steps of
    // floor/ceiling and the boundaries of Piecewise conditions count only
    // where the scan itself located all of them.  Up to 0.39 an incomplete
    // scan of `floor(x²)` (a non-linear argument) passed with an empty
    // watch list, the one piece was resolved at its midpoint, and
    // `∫₀² a·floor(x²) dx` came out as `2a` (it is `(5 − √2 − √3)·a`).
    if !steps_located(arena, f, x, scan_range(a, b, range)) {
        return Err(failed(
            "could not locate every step of floor, ceiling or a Piecewise condition inside the interval",
        ));
    }

    // Singularities other than sign changes of those arguments are handled
    // downstream.
    let mut points: Vec<(ExprId, Option<f64>)> = Vec::new();
    for bp in &breaks {
        match position_in(arena, bp.point, a, b) {
            Position::Inside if !points.iter().any(|(p, _)| *p == bp.point) => {
                points.push((bp.point, bp.value));
            }
            Position::Unknown => {
                return Err(failed(
                    "cannot decide whether a kink of the integrand lies inside the interval",
                ));
            }
            _ => {}
        }
    }
    if points.iter().any(|(_, v)| v.is_none()) && points.len() > 1 {
        return Err(failed("cannot order symbolic kinks of the integrand"));
    }
    if points.len() > MAX_PIECES {
        return Err(failed("too many kinks inside the interval"));
    }
    let points = sort_points(arena, points)
        .ok_or_else(|| failed("cannot order the kinks of the integrand"))?;

    let mut bounds: Vec<ExprId> = Vec::with_capacity(points.len() + 2);
    bounds.push(a);
    bounds.extend(points.iter().map(|(p, _)| *p));
    bounds.push(b);

    let mut total = Vec::new();
    for w in bounds.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        let mid = test_point(arena, lo, hi).ok_or_else(|| {
            failed("cannot choose a test point inside a piece with symbolic bounds")
        })?;
        let smooth = resolve_at(arena, f, x, mid)?;
        if smooth == f {
            return Err(failed(
                "could not resolve piecewise structure of the integrand",
            ));
        }
        let v = integrate_range(arena, smooth, x, lo, hi, depth + 1)?;
        total.push(v);
    }
    let s = arena.add(&total);
    Ok(safe_eval(arena, s))
}

/// Did the breakpoint scan locate every step of the `floor`/`ceiling` nodes
/// and every boundary of the `Piecewise` conditions of `f` depending on
/// `x` (the breakpoints [`sign_changes_explained`] does not check)?
fn steps_located(arena: &mut Arena, f: ExprId, x: ExprId, range: Option<ScanRange>) -> bool {
    let mut probes: Vec<ExprId> = Vec::new();
    for id in walk::post_order_ids(arena, f) {
        match arena.node(id) {
            ExprNode::Floor(g) | ExprNode::Ceiling(g) if walk::contains(arena, *g, x) => {
                probes.push(id);
            }
            ExprNode::Piecewise(pairs) => {
                for &(_, cond) in pairs.iter() {
                    if walk::contains(arena, cond, x) {
                        probes.push(cond);
                    }
                }
            }
            _ => {}
        }
    }
    // A periodic composition's boundaries are one period of a principal
    // solve (`Piecewise((1, cos x > 1/2), (0, True))` on `[0, 10]` has six,
    // not two).
    probes.into_iter().all(|p| {
        let s = calculus_util::scan_breakpoints(arena, p, x, range);
        s.complete && !s.opaque && !s.periodic
    })
}

/// The `x`-dependent arguments of the `|·|`, `sign` and `Heaviside` nodes
/// of `f`, whose sign changes split the interval in
/// [`split_piecewise_like`].
fn kink_arguments(arena: &Arena, f: ExprId, x: ExprId) -> Vec<ExprId> {
    let mut out = Vec::new();
    for id in walk::post_order_ids(arena, f) {
        if let ExprNode::Abs(g) | ExprNode::Sign(g) | ExprNode::Heaviside(g) = *arena.node(id)
            && walk::contains(arena, g, x)
            && !out.contains(&g)
        {
            out.push(g);
        }
    }
    out
}

/// A rational point strictly inside `(lo, hi)` (infinite ends allowed).
fn test_point(arena: &mut Arena, lo: ExprId, hi: ExprId) -> Option<ExprId> {
    // A symbolic end next to an infinite one, or two symbolic ends (whose
    // order the caller decided): `h − 1`, `l + 1`, the midpoint.  The
    // Laplace density `e^{−|x − μ|/b}` over ℝ with a real `μ` was refused up
    // to 0.46 for want of a test point in `(−∞, μ)` (SymPy's test_meijerint).
    let (lo_v, hi_v) = match (point_value(arena, lo), point_value(arena, hi)) {
        (Some(l), Some(h)) => (l, h),
        (l, h) => {
            let one = arena.one();
            return Some(match (l, h) {
                (Some(l), None) if l == f64::NEG_INFINITY => arena.sub(hi, one),
                (None, Some(h)) if h == f64::INFINITY => arena.add(&[lo, one]),
                (None, None) => {
                    let half = arena.rational(1, 2);
                    let s = arena.add(&[lo, hi]);
                    let m = arena.mul(&[half, s]);
                    safe_eval(arena, m)
                }
                _ => return None,
            });
        }
    };
    let m = match (lo_v.is_finite(), hi_v.is_finite()) {
        (true, true) => {
            if let (Some(rl), Some(rh)) = (arena.as_num(lo).cloned(), arena.as_num(hi).cloned()) {
                let two = Ratio::from_integer(BigInt::from(2));
                let mid = (rl + rh) / two;
                let nid = arena.intern_num(mid);
                return Some(arena.intern(ExprNode::Num(nid)));
            }
            let m = 0.5 * (lo_v + hi_v);
            // The binary midpoint of the values, when they show it strictly
            // inside; otherwise (ends closer than their `f64` values tell,
            // `[kπ/10²⁰, (k + 1)π/10²⁰]`) the exact midpoint.  Up to 0.49 the
            // binary one was taken anyway, outside such a piece.
            let above = |end: ExprId, v: f64, arena: &Arena| {
                separated(v, m) == Some(Ordering::Less)
                    || (arena.is_zero_structural(end) && m > 0.0)
            };
            let below = |end: ExprId, v: f64, arena: &Arena| {
                separated(m, v) == Some(Ordering::Less)
                    || (arena.is_zero_structural(end) && m < 0.0)
            };
            if !(above(lo, lo_v, arena) && below(hi, hi_v, arena)) {
                let half = arena.rational(1, 2);
                let s = arena.add(&[lo, hi]);
                let mid = arena.mul(&[half, s]);
                return Some(safe_eval(arena, mid));
            }
            m
        }
        (true, false) => lo_v + 1.0,
        (false, true) => hi_v - 1.0,
        (false, false) => 0.0,
    };
    let r = crate::base::numeric::f64_to_ratio_exact(m)?;
    let nid = arena.intern_num(r);
    Some(arena.intern(ExprNode::Num(nid)))
}

/// Replace every `Abs`, `Sign`, `Heaviside`, `Floor`, `Ceiling` and
/// `Piecewise` node depending on `x` by its smooth branch as determined at
/// the test point `mid`.
fn resolve_at(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    mid: ExprId,
) -> Result<ExprId, SymplexError> {
    let mut current = f;
    for _ in 0..64 {
        let order = walk::post_order_ids(arena, current);
        let mut target: Option<(ExprId, ExprId)> = None;
        for id in order {
            let node = arena.node(id).clone();
            let replacement = match node {
                ExprNode::Abs(inner) if walk::contains(arena, inner, x) => {
                    match sign_at(arena, inner, x, mid)? {
                        Ordering::Less => Some(arena.neg(inner)),
                        _ => Some(inner),
                    }
                }
                ExprNode::Sign(inner) if walk::contains(arena, inner, x) => {
                    match sign_at(arena, inner, x, mid)? {
                        Ordering::Less => Some(arena.neg_one()),
                        Ordering::Greater => Some(arena.one()),
                        Ordering::Equal => Some(arena.zero()),
                    }
                }
                ExprNode::Heaviside(inner) if walk::contains(arena, inner, x) => {
                    match sign_at(arena, inner, x, mid)? {
                        Ordering::Less => Some(arena.zero()),
                        Ordering::Greater => Some(arena.one()),
                        Ordering::Equal => Some(arena.rational(1, 2)),
                    }
                }
                ExprNode::Floor(inner) | ExprNode::Ceiling(inner)
                    if walk::contains(arena, inner, x) =>
                {
                    Some(integer_part_at(
                        arena,
                        inner,
                        x,
                        mid,
                        matches!(node, ExprNode::Ceiling(_)),
                    )?)
                }
                ExprNode::Piecewise(pairs) if walk::contains(arena, id, x) => {
                    let mut chosen: Option<ExprId> = None;
                    for (val, cond) in pairs.iter() {
                        match condition_at(arena, *cond, x, mid)? {
                            Truth::True => {
                                chosen = Some(*val);
                                break;
                            }
                            Truth::False => {}
                            Truth::Unknown => {
                                return Err(failed(
                                    "cannot decide a Piecewise condition at a test point",
                                ));
                            }
                        }
                    }
                    Some(chosen.unwrap_or_else(|| arena.zero()))
                }
                _ => None,
            };
            if let Some(r) = replacement {
                target = Some((id, r));
                break;
            }
        }
        match target {
            Some((old, new)) => {
                current = subs::subs(arena, current, old, new);
                current = safe_eval(arena, current);
            }
            None => return Ok(current),
        }
    }
    Err(failed("too many piecewise constructs in the integrand"))
}

/// Numeric value of `g` at `x = mid`.
fn value_at(arena: &mut Arena, g: ExprId, x: ExprId, mid: ExprId) -> Result<f64, SymplexError> {
    let gm = subs::subs(arena, g, x, mid);
    let gm = safe_eval(arena, gm);
    if !walk::free_symbols(arena, gm).is_empty() {
        return Err(failed(
            "piecewise structure depends on symbolic parameters; cannot resolve",
        ));
    }
    calculus_util::expr_to_f64(arena, gm)
        .filter(|v| v.is_finite())
        .ok_or_else(|| failed("could not evaluate a piecewise argument at a test point"))
}

/// `⌊g⌋` (`ceiling`: `⌈g⌉`) at `x = mid`, certified (`eval`'s integer part
/// of a constant).  Up to 0.49 it was the `f64` floor of the value, refused
/// beyond `10¹⁵`.
fn integer_part_at(
    arena: &mut Arena,
    g: ExprId,
    x: ExprId,
    mid: ExprId,
    ceiling: bool,
) -> Result<ExprId, SymplexError> {
    let gm = subs::subs(arena, g, x, mid);
    let gm = safe_eval(arena, gm);
    if !walk::free_symbols(arena, gm).is_empty() {
        return Err(failed(
            "piecewise structure depends on symbolic parameters; cannot resolve",
        ));
    }
    let part = if ceiling {
        arena.ceiling(gm)
    } else {
        arena.floor(gm)
    };
    let n = eval::eval(arena, part);
    match arena.as_num(n) {
        Some(r) if r.is_integer() => Ok(n),
        _ => Err(failed(
            "could not decide a floor or ceiling at a test point",
        )),
    }
}

/// Sign of `g` at `x = mid`: from its 16-digit value when that is not 0
/// (a printed digit is certified), otherwise from the certified sign test
/// (up to 0.49 a value 0 to 16 digits was a zero of `g`).
fn sign_at(arena: &mut Arena, g: ExprId, x: ExprId, mid: ExprId) -> Result<Ordering, SymplexError> {
    let gm = subs::subs(arena, g, x, mid);
    let gm = safe_eval(arena, gm);
    if !walk::free_symbols(arena, gm).is_empty() {
        return sign_of(arena, gm).ok_or_else(|| {
            failed("piecewise structure depends on symbolic parameters; cannot resolve")
        });
    }
    match value_at(arena, g, x, mid) {
        Ok(v) if v != 0.0 => return Ok(v.partial_cmp(&0.0).unwrap_or(Ordering::Equal)),
        Ok(_) | Err(_) => {}
    }
    match crate::poly::algebraic::sign_checked(arena, gm) {
        Some(s) => Ok(s.cmp(&0)),
        None => Err(failed(
            "could not decide the sign of a piecewise argument at a test point",
        )),
    }
}

/// Decide a parameter condition (no integration variable involved) with
/// the assumption system: `Unknown` if undecidable.
fn decide_condition(arena: &mut Arena, cond: ExprId) -> Truth {
    let order = walk::post_order_ids(arena, cond);
    let mut vals: FxHashMap<ExprId, Truth> = FxHashMap::default();
    for id in order {
        let node = arena.node(id).clone();
        let v = match node {
            ExprNode::BoolTrue => Truth::True,
            ExprNode::BoolFalse => Truth::False,
            ExprNode::Gt(a, b) | ExprNode::Ge(a, b) | ExprNode::Eq_(a, b) | ExprNode::Ne(a, b) => {
                let d = arena.sub(a, b);
                let d = safe_eval(arena, d);
                let s = sign_of(arena, d);
                Truth::from(match node {
                    ExprNode::Gt(..) => s.map(|s| s == Ordering::Greater),
                    ExprNode::Ge(..) => s.map(|s| s != Ordering::Less),
                    ExprNode::Eq_(..) => s.map(|s| s == Ordering::Equal),
                    _ => s.map(|s| s != Ordering::Equal),
                })
            }
            ExprNode::And(children) => Truth::all(children.iter().map(|c| truth_of(&vals, *c))),
            ExprNode::Or(children) => Truth::any(children.iter().map(|c| truth_of(&vals, *c))),
            ExprNode::Not(inner) => truth_of(&vals, inner).not(),
            _ => Truth::Unknown,
        };
        vals.insert(id, v);
    }
    truth_of(&vals, cond)
}

/// The truth value recorded for `id`, `Unknown` when there is none.
fn truth_of(vals: &FxHashMap<ExprId, Truth>, id: ExprId) -> Truth {
    vals.get(&id).copied().unwrap_or(Truth::Unknown)
}

/// Truth value of a boolean condition at `x = mid`, three-valued: the
/// relation of `p` and `q` there is decided by the exact or certified zero
/// and sign tests of `p − q`
/// ([`is_zero_checked`](crate::poly::algebraic::is_zero_checked),
/// [`sign_checked`](crate::poly::algebraic::sign_checked)), and is
/// `Unknown` when they do not decide it (a value 0 to the precision
/// reached, an order of non-real values); so is a connective of an
/// undecided part, by Kleene's rules.  Up to 0.48 the difference was an
/// `f64`, `Eq` held when it was within `10⁻¹²` of 0, and an undecided
/// sub-condition counted as `false`.
fn condition_at(
    arena: &mut Arena,
    cond: ExprId,
    x: ExprId,
    mid: ExprId,
) -> Result<Truth, SymplexError> {
    let order = walk::post_order_ids(arena, cond);
    let mut vals: FxHashMap<ExprId, Truth> = FxHashMap::default();
    for id in order {
        let node = arena.node(id).clone();
        let v = match node {
            ExprNode::BoolTrue => Truth::True,
            ExprNode::BoolFalse => Truth::False,
            ExprNode::Gt(p, q) | ExprNode::Ge(p, q) | ExprNode::Eq_(p, q) | ExprNode::Ne(p, q) => {
                let d = arena.sub(p, q);
                let d = subs::subs(arena, d, x, mid);
                let d = safe_eval(arena, d);
                if !walk::free_symbols(arena, d).is_empty() {
                    return Err(failed(
                        "piecewise structure depends on symbolic parameters; cannot resolve",
                    ));
                }
                match node {
                    ExprNode::Gt(..) | ExprNode::Ge(..) => {
                        match crate::poly::algebraic::sign_checked(arena, d) {
                            Some(s) if matches!(node, ExprNode::Gt(..)) => Truth::from_bool(s > 0),
                            Some(s) => Truth::from_bool(s >= 0),
                            None => Truth::Unknown,
                        }
                    }
                    ExprNode::Eq_(..) => {
                        Truth::from(crate::poly::algebraic::is_zero_checked(arena, d))
                    }
                    _ => Truth::from(crate::poly::algebraic::is_zero_checked(arena, d)).not(),
                }
            }
            ExprNode::And(children) => Truth::all(children.iter().map(|c| truth_of(&vals, *c))),
            ExprNode::Or(children) => Truth::any(children.iter().map(|c| truth_of(&vals, *c))),
            ExprNode::Not(inner) => truth_of(&vals, inner).not(),
            // Relational sub-terms (numbers, symbols) are not booleans.
            _ => {
                if id == cond {
                    return Err(failed("unsupported Piecewise condition"));
                }
                Truth::Unknown
            }
        };
        vals.insert(id, v);
    }
    vals.get(&cond)
        .copied()
        .ok_or_else(|| failed("unsupported Piecewise condition"))
}

// ═══════════════════════════════════════════════════════════════════════════
// Known-value table
// ═══════════════════════════════════════════════════════════════════════════

/// Canonical interval shapes recognised by the table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IntervalShape {
    ZeroInf,
    FullLine,
    ZeroOne,
    ZeroHalfPi,
    ZeroPi,
    ZeroTwoPi,
    NegPiPi,
    Other,
}

fn classify_interval(arena: &mut Arena, a: ExprId, b: ExprId) -> IntervalShape {
    let zero = arena.zero();
    let one = arena.one();
    let pi = arena.pi();
    let half = arena.rational(1, 2);
    let half_pi = arena.mul(&[half, pi]);
    let two = arena.int(2);
    let two_pi = arena.mul(&[two, pi]);
    let neg_pi = arena.neg(pi);
    let same =
        |arena: &mut Arena, p: ExprId, q: ExprId| cmp_points(arena, p, q) == Some(Ordering::Equal);

    if a == arena.neg_infinity() && b == arena.infinity() {
        return IntervalShape::FullLine;
    }
    if same(arena, a, zero) {
        if b == arena.infinity() {
            return IntervalShape::ZeroInf;
        }
        if same(arena, b, one) {
            return IntervalShape::ZeroOne;
        }
        if same(arena, b, half_pi) {
            return IntervalShape::ZeroHalfPi;
        }
        if same(arena, b, pi) {
            return IntervalShape::ZeroPi;
        }
        if same(arena, b, two_pi) {
            return IntervalShape::ZeroTwoPi;
        }
    }
    if same(arena, a, neg_pi) && same(arena, b, pi) {
        return IntervalShape::NegPiPi;
    }
    IntervalShape::Other
}

/// Split `f` into an `x`-free coefficient and `x`-dependent factors.
/// Fractional/negative powers of products are distributed over the
/// factors so that e.g. `(x(1−x))^{−1/2}` becomes `x^{−1/2}(1−x)^{−1/2}`.
fn factorize(arena: &mut Arena, f: ExprId, x: ExprId) -> (Vec<ExprId>, Vec<ExprId>) {
    let mut coeff = Vec::new();
    let mut dep = Vec::new();
    let mut stack: Vec<ExprId> = vec![f];
    while let Some(id) = stack.pop() {
        if !walk::contains(arena, id, x) {
            coeff.push(id);
            continue;
        }
        match arena.node(id).clone() {
            ExprNode::Mul(children) => stack.extend(children.iter().copied()),
            ExprNode::Neg(inner) => {
                coeff.push(arena.neg_one());
                stack.push(inner);
            }
            ExprNode::Pow(base, exp) if arena.as_num(exp).is_some() => {
                match arena.node(base).clone() {
                    ExprNode::Mul(children) => {
                        for c in children.iter() {
                            let p = arena.pow(*c, exp);
                            stack.push(p);
                        }
                    }
                    // (b^m)^n with numeric m, n → b^{mn} (real-valued
                    // context; the canon layer flattens only an integer n).
                    ExprNode::Pow(inner_base, inner_exp) if arena.as_num(inner_exp).is_some() => {
                        let m = arena.as_num(inner_exp).cloned().unwrap_or_else(Ratio::one);
                        let n = arena.as_num(exp).cloned().unwrap_or_else(Ratio::one);
                        let mn = ratio_expr(arena, &(m * n));
                        let flat = arena.pow(inner_base, mn);
                        stack.push(flat);
                    }
                    _ => dep.push(id),
                }
            }
            _ => dep.push(id),
        }
    }
    (coeff, dep)
}

/// Multiply a table value by the constant coefficient.
fn with_coeff(arena: &mut Arena, coeff: &[ExprId], v: ExprId) -> ExprId {
    let mut all: Vec<ExprId> = coeff.to_vec();
    all.push(v);
    let r = arena.mul(&all);
    safe_eval(arena, r)
}

/// `x^p` → `p` (as an expression); `x` → `1`.
fn as_x_power(arena: &Arena, e: ExprId, x: ExprId) -> Option<ExprId> {
    if e == x {
        return Some(arena.one());
    }
    if let ExprNode::Pow(base, exp) = arena.node(e)
        && *base == x
        && !walk::contains(arena, *exp, x)
    {
        return Some(*exp);
    }
    None
}

/// Rational value of a numeric expression.
fn as_ratio(arena: &Arena, e: ExprId) -> Option<Q> {
    arena.as_num(e).cloned()
}

/// `Pow(base, −m)` → `(base, m)` with `m` a positive number; a bare
/// non-power expression is not a reciprocal.
fn as_reciprocal(arena: &Arena, e: ExprId) -> Option<(ExprId, Q)> {
    if let ExprNode::Pow(base, exp) = arena.node(e)
        && let Some(r) = arena.as_num(*exp)
        && r.is_negative()
    {
        return Some((*base, -r.clone()));
    }
    None
}

/// Trig kind for table matching.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Trig {
    Sin,
    Cos,
}

/// `sin(arg)^n` / `cos(arg)^n` → `(kind, arg, n)`; bare `sin(arg)` has `n = 1`.
fn as_trig_power(arena: &Arena, e: ExprId) -> Option<(Trig, ExprId, Q)> {
    let (inner, n) = if let ExprNode::Pow(base, exp) = arena.node(e) {
        (*base, arena.as_num(*exp)?.clone())
    } else {
        (e, Ratio::one())
    };
    match arena.node(inner) {
        ExprNode::Sin(arg) => Some((Trig::Sin, *arg, n)),
        ExprNode::Cos(arg) => Some((Trig::Cos, *arg, n)),
        _ => None,
    }
}

/// `ln(arg)^n` → `(arg, n)`.
fn as_ln_power(arena: &Arena, e: ExprId) -> Option<(ExprId, Q)> {
    let (inner, n) = if let ExprNode::Pow(base, exp) = arena.node(e) {
        (*base, arena.as_num(*exp)?.clone())
    } else {
        (e, Ratio::one())
    };
    match arena.node(inner) {
        ExprNode::Ln(arg) => Some((*arg, n)),
        _ => None,
    }
}

/// Exact `Γ(r)` for integer and half-integer `r`; a `Gamma` node otherwise.
fn gamma_exact(arena: &mut Arena, r: &Q) -> Option<ExprId> {
    if r.is_integer() {
        if !r.is_positive() {
            return None;
        }
        let n = r.to_integer().to_u64()?;
        if n > 170 {
            return None;
        }
        let mut acc = BigInt::one();
        for k in 2..n {
            acc *= BigInt::from(k);
        }
        return Some(arena.big_int(acc));
    }
    let two = BigInt::from(2);
    if *r.denom() == two {
        // Γ(k + ½) = (2k)!/(4^k k!) √π  for k ≥ 0; use recurrence downwards.
        let twice = r.numer().clone(); // odd
        let k2 = (&twice - BigInt::one()) / &two; // r = k2 + 1/2
        let k = k2.to_i64()?;
        if k.abs() > 200 {
            return None;
        }
        // Compute rational factor c with Γ(k + ½) = c √π.
        let mut c = Ratio::one();
        if k >= 0 {
            for j in 0..k {
                // Γ(j + 3/2) = (j + 1/2) Γ(j + 1/2)
                c *= Ratio::new(BigInt::from(2 * j + 1), two.clone());
            }
        } else {
            for j in 0..(-k) {
                // Γ(−j − 1/2) = Γ(−j + 1/2) / (−j − 1/2)
                c /= Ratio::new(BigInt::from(-2 * j - 1), two.clone());
            }
        }
        let c_id = {
            let nid = arena.intern_num(c);
            arena.intern(ExprNode::Num(nid))
        };
        let pi = arena.pi();
        let sqrt_pi = arena.sqrt(pi);
        return Some(arena.mul(&[c_id, sqrt_pi]));
    }
    let r_id = {
        let nid = arena.intern_num(r.clone());
        arena.intern(ExprNode::Num(nid))
    };
    Some(arena.gamma(r_id))
}

/// `Γ(e)` for a general expression: exact when numeric, symbolic otherwise.
fn gamma_of(arena: &mut Arena, e: ExprId) -> Option<ExprId> {
    if let Some(r) = as_ratio(arena, e) {
        return gamma_exact(arena, &r);
    }
    Some(arena.gamma(e))
}

/// `B(p, q)`: `π / sin(πp)` when `p + q = 1`, exact gammas otherwise.
fn beta_of(arena: &mut Arena, p: ExprId, q: ExprId) -> Option<ExprId> {
    if let (Some(rp), Some(rq)) = (as_ratio(arena, p), as_ratio(arena, q)) {
        if !rp.is_positive() || !rq.is_positive() {
            return None;
        }
        if &rp + &rq == Ratio::one() {
            let pi = arena.pi();
            let pp = arena.mul(&[p, pi]);
            let s = arena.sin(pp);
            let r = arena.div(pi, s);
            return Some(safe_eval(arena, r));
        }
    }
    let gp = gamma_of(arena, p)?;
    let gq = gamma_of(arena, q)?;
    let pq = arena.add(&[p, q]);
    let pq = safe_eval(arena, pq);
    let gpq = gamma_of(arena, pq)?;
    let num = arena.mul(&[gp, gq]);
    let r = arena.div(num, gpq);
    Some(safe_eval(arena, r))
}

/// `ζ(2k)` exactly via Bernoulli numbers.
fn zeta_even(arena: &mut Arena, two_k: u32) -> Option<ExprId> {
    if two_k == 0 || !two_k.is_multiple_of(2) || two_k > 60 {
        return None;
    }
    let k = two_k / 2;
    let b = crate::base::bernoulli::bernoulli(two_k as usize);
    // ζ(2k) = (−1)^{k+1} B_{2k} (2π)^{2k} / (2 (2k)!)
    let mut fact = BigInt::one();
    for j in 2..=two_k {
        fact *= BigInt::from(j);
    }
    let sign = if k % 2 == 1 { 1 } else { -1 };
    let two_pow = BigInt::from(2).pow(two_k);
    let coeff = b * Ratio::from_integer(two_pow) * Ratio::from_integer(BigInt::from(sign))
        / Ratio::from_integer(fact * BigInt::from(2));
    let c_id = {
        let nid = arena.intern_num(coeff);
        arena.intern(ExprNode::Num(nid))
    };
    let pi = arena.pi();
    let n = arena.int(two_k as i64);
    let pi_pow = arena.pow(pi, n);
    Some(arena.mul(&[c_id, pi_pow]))
}

/// Coefficients `[c₀, c₁, c₂]` of a quadratic in `x` (zero-padded).
fn quadratic_coeffs(arena: &mut Arena, e: ExprId, x: ExprId) -> Option<[ExprId; 3]> {
    let c = calculus_util::poly_coeffs_symbolic(arena, e, x)?;
    if c.len() > 3 {
        return None;
    }
    let z = arena.zero();
    Some([
        c.first().copied().unwrap_or(z),
        c.get(1).copied().unwrap_or(z),
        c.get(2).copied().unwrap_or(z),
    ])
}

/// Decompose `k + c·x^n` (exactly two terms, `k`, `c` free of `x`).
fn as_binomial_in_x(arena: &mut Arena, e: ExprId, x: ExprId) -> Option<(ExprId, ExprId, Q)> {
    let terms: Vec<ExprId> = match arena.node(e).clone() {
        ExprNode::Add(ch) => ch.to_vec(),
        _ => return None,
    };
    if terms.len() != 2 {
        return None;
    }
    let mut k: Option<ExprId> = None;
    let mut cx: Option<(ExprId, Q)> = None;
    for t in terms {
        if !walk::contains(arena, t, x) {
            k = Some(t);
            continue;
        }
        // c · x^n
        let (coeff, dep) = factorize(arena, t, x);
        if dep.len() != 1 {
            return None;
        }
        let n = as_x_power(arena, dep[0], x)?;
        let n = as_ratio(arena, n)?;
        let c = if coeff.is_empty() {
            arena.one()
        } else {
            let m = arena.mul(&coeff);
            safe_eval(arena, m)
        };
        cx = Some((c, n));
    }
    let k = k?;
    let (c, n) = cx?;
    Some((k, c, n))
}

/// Match `f` against the known-value table for `∫ₐᵇ`.
fn table_lookup(arena: &mut Arena, f: ExprId, x: ExprId, a: ExprId, b: ExprId) -> Option<ExprId> {
    let iv = classify_interval(arena, a, b);
    if iv == IntervalShape::Other {
        return None;
    }
    let (coeff, dep) = factorize(arena, f, x);
    if dep.is_empty() || dep.len() > 4 {
        return None;
    }
    let v = match iv {
        IntervalShape::ZeroInf => table_zero_inf(arena, &dep, x),
        IntervalShape::FullLine => table_full_line(arena, &dep, x),
        IntervalShape::ZeroOne => table_zero_one(arena, &dep, x),
        IntervalShape::ZeroHalfPi
        | IntervalShape::ZeroPi
        | IntervalShape::ZeroTwoPi
        | IntervalShape::NegPiPi => table_trig(arena, &dep, x, iv),
        IntervalShape::Other => None,
    }?;
    Some(with_coeff(arena, &coeff, v))
}

/// Shape of a parsed `[0, ∞)`-type integrand.
#[derive(Default)]
struct Shape {
    /// exponent of the bare `x^p` factor (None if absent)
    x_pow: Option<ExprId>,
    /// argument of an `exp(...)` factor
    exp_arg: Option<ExprId>,
    /// `(kind, arg, n)` of trig factors (at most two)
    trig: Vec<(Trig, ExprId, Q)>,
    /// `(base, m)` of reciprocal factors
    recip: Vec<(ExprId, Q)>,
    /// `(arg, n)` of a logarithm factor
    ln: Option<(ExprId, Q)>,
    /// `(arg, m)` of `cosh(arg)^{-m}` / `sinh(arg)^{-m}` (kind: true = cosh)
    hyp_recip: Option<(bool, ExprId, Q)>,
    /// any factor that could not be classified
    other: bool,
}

/// Classify the dependent factors.
fn parse_shape(arena: &Arena, dep: &[ExprId], x: ExprId) -> Shape {
    let mut s = Shape::default();
    for &d in dep {
        if let Some(p) = as_x_power(arena, d, x) {
            if s.x_pow.is_some() {
                s.other = true;
            }
            s.x_pow = Some(p);
        } else if let ExprNode::Exp(arg) = arena.node(d) {
            if s.exp_arg.is_some() {
                s.other = true;
            }
            s.exp_arg = Some(*arg);
        } else if let Some(t) = as_trig_power(arena, d) {
            s.trig.push(t);
        } else if let Some((base, m)) = as_reciprocal(arena, d) {
            match arena.node(base) {
                ExprNode::Cosh(arg) => s.hyp_recip = Some((true, *arg, m)),
                ExprNode::Sinh(arg) => s.hyp_recip = Some((false, *arg, m)),
                _ => s.recip.push((base, m)),
            }
        } else if let Some(l) = as_ln_power(arena, d) {
            if s.ln.is_some() {
                s.other = true;
            }
            s.ln = Some(l);
        } else {
            s.other = true;
        }
    }
    s
}

/// Extract `α` from a purely linear argument `α·x` (no constant term).
fn pure_linear_coeff(arena: &mut Arena, arg: ExprId, x: ExprId) -> Option<ExprId> {
    let lin = calculus_util::linear_coeffs_exact(arena, arg, x)?;
    if !arena.is_zero_structural(lin.intercept) {
        return None;
    }
    Some(lin.slope)
}

/// Build `Num` from a rational.
fn ratio_expr(arena: &mut Arena, r: &Q) -> ExprId {
    let nid = arena.intern_num(r.clone());
    arena.intern(ExprNode::Num(nid))
}

/// A single `[0, ∞)` table entry.
type HalfLineEntry = fn(&mut Arena, &HalfLine, ExprId) -> Option<ExprId>;

/// Shared pre-processing for the `[0, ∞)` entries.
struct HalfLine {
    s: Shape,
    /// exponent `p` of the `x^p` factor (0 if absent)
    p: ExprId,
    /// `p + 1`, evaluated
    p1: ExprId,
}

/// Table for `∫₀^∞`: each entry is independent, so a failed structural
/// match inside one entry never prevents the next from being tried.
fn table_zero_inf(arena: &mut Arena, dep: &[ExprId], x: ExprId) -> Option<ExprId> {
    let s = parse_shape(arena, dep, x);
    if s.other {
        return None;
    }
    let one = arena.one();
    let p = s.x_pow.unwrap_or(arena.zero());
    let p1 = arena.add(&[p, one]);
    let p1 = safe_eval(arena, p1);
    let h = HalfLine { s, p, p1 };

    let entries: [HalfLineEntry; 10] = [
        entry_exp_linear,
        entry_exp_gaussian,
        entry_fresnel,
        entry_trig_over_power,
        entry_algebraic_beta,
        entry_trig_over_quadratic,
        entry_ln_over_quadratic,
        entry_bose_fermi,
        entry_hyperbolic_recip,
        entry_none,
    ];
    for entry in entries {
        if let Some(v) = entry(arena, &h, x) {
            return Some(v);
        }
    }
    None
}

/// Placeholder to keep the entry table length fixed.
fn entry_none(_arena: &mut Arena, _h: &HalfLine, _x: ExprId) -> Option<ExprId> {
    None
}

/// `x^p e^{−a x + c}` → `e^c Γ(p+1)/a^{p+1}`;
/// `e^{−a x} sin(bx)` → `b/(a²+b²)`;  `e^{−a x} cos(bx)` → `a/(a²+b²)`;
/// `e^{−a x} sin(bx)/x` → `atan(b/a)`   (all with `a > 0`).
fn entry_exp_linear(arena: &mut Arena, h: &HalfLine, x: ExprId) -> Option<ExprId> {
    let s = &h.s;
    let arg = s.exp_arg?;
    if !s.recip.is_empty() || s.ln.is_some() || s.hyp_recip.is_some() {
        return None;
    }
    let [c0, c1, c2] = quadratic_coeffs(arena, arg, x)?;
    if !arena.is_zero_structural(c2) {
        return None;
    }
    let e_c0 = arena.exp(c0);
    let a = arena.neg(c1);
    let a = safe_eval(arena, a);
    if !is_positive(arena, a) {
        return None;
    }
    if s.trig.is_empty() {
        if !is_positive(arena, h.p1) {
            return None;
        }
        let g = gamma_of(arena, h.p1)?;
        let neg_p1 = arena.neg(h.p1);
        let a_pow = arena.pow(a, neg_p1);
        return Some(arena.mul(&[e_c0, g, a_pow]));
    }
    if s.trig.len() != 1 {
        return None;
    }
    let (kind, targ, n) = s.trig[0].clone();
    if n != Ratio::one() {
        return None;
    }
    let bcoef = pure_linear_coeff(arena, targ, x)?;
    // e^{−ax} sin(bx) converges only for a > |Im b|: ask for a real b.
    if !is_real(arena, bcoef) {
        return None;
    }
    if arena.is_zero_structural(h.p) {
        let a2 = arena.mul(&[a, a]);
        let b2 = arena.mul(&[bcoef, bcoef]);
        let denom = arena.add(&[a2, b2]);
        let num = match kind {
            Trig::Sin => bcoef,
            Trig::Cos => a,
        };
        let r = arena.div(num, denom);
        return Some(arena.mul(&[e_c0, r]));
    }
    if kind == Trig::Sin && h.p == arena.neg_one() {
        let r = arena.div(bcoef, a);
        let at = arena.atan(r);
        return Some(arena.mul(&[e_c0, at]));
    }
    None
}

/// `x^p e^{−a x² + c}` → `e^c Γ((p+1)/2) / (2 a^{(p+1)/2})`;
/// `e^{−a x²} cos(bx)` → `½ √(π/a) e^{−b²/(4a)}`   (`a > 0`).
fn entry_exp_gaussian(arena: &mut Arena, h: &HalfLine, x: ExprId) -> Option<ExprId> {
    let s = &h.s;
    let arg = s.exp_arg?;
    if !s.recip.is_empty() || s.ln.is_some() || s.hyp_recip.is_some() {
        return None;
    }
    let [c0, c1, c2] = quadratic_coeffs(arena, arg, x)?;
    if arena.is_zero_structural(c2) || !arena.is_zero_structural(c1) {
        return None;
    }
    let e_c0 = arena.exp(c0);
    let a = arena.neg(c2);
    let a = safe_eval(arena, a);
    if !is_positive(arena, a) {
        return None;
    }
    let half = arena.rational(1, 2);
    if s.trig.is_empty() {
        if !is_positive(arena, h.p1) {
            return None;
        }
        let s_arg = arena.mul(&[half, h.p1]);
        let s_arg = safe_eval(arena, s_arg);
        let g = gamma_of(arena, s_arg)?;
        let neg_s = arena.neg(s_arg);
        let a_pow = arena.pow(a, neg_s);
        return Some(arena.mul(&[e_c0, half, g, a_pow]));
    }
    if s.trig.len() != 1 || !arena.is_zero_structural(h.p) {
        return None;
    }
    let (kind, targ, n) = s.trig[0].clone();
    if kind != Trig::Cos || n != Ratio::one() {
        return None;
    }
    let bcoef = pure_linear_coeff(arena, targ, x)?;
    let pi = arena.pi();
    let pi_over_a = arena.div(pi, a);
    let root = arena.sqrt(pi_over_a);
    let b2 = arena.mul(&[bcoef, bcoef]);
    let four = arena.int(4);
    let four_a = arena.mul(&[four, a]);
    let ratio = arena.div(b2, four_a);
    let neg = arena.neg(ratio);
    let e = arena.exp(neg);
    Some(arena.mul(&[e_c0, half, root, e]))
}

/// Fresnel: `sin(c x²)`, `cos(c x²)` → `½ √(π/(2|c|))` (sign of `c` for sin).
fn entry_fresnel(arena: &mut Arena, h: &HalfLine, x: ExprId) -> Option<ExprId> {
    let s = &h.s;
    if s.exp_arg.is_some() || !s.recip.is_empty() || s.ln.is_some() || s.hyp_recip.is_some() {
        return None;
    }
    if s.trig.len() != 1 || !arena.is_zero_structural(h.p) {
        return None;
    }
    let (kind, targ, n) = s.trig[0].clone();
    if n != Ratio::one() {
        return None;
    }
    let [c0, c1, c2] = quadratic_coeffs(arena, targ, x)?;
    if !arena.is_zero_structural(c0) || !arena.is_zero_structural(c1) {
        return None;
    }
    let c_sign = sign_of(arena, c2)?;
    if c_sign == Ordering::Equal {
        return None;
    }
    let abs_c = arena.abs(c2);
    let two = arena.int(2);
    let two_c = arena.mul(&[two, abs_c]);
    let pi = arena.pi();
    let ratio = arena.div(pi, two_c);
    let root = arena.sqrt(ratio);
    let half = arena.rational(1, 2);
    let v = arena.mul(&[half, root]);
    if kind == Trig::Sin && c_sign == Ordering::Less {
        return Some(arena.neg(v));
    }
    Some(v)
}

/// `sin(bx)/x` → `(π/2) sign(b)`;  `sin²(bx)/x²` → `π|b|/2`;
/// `x^p sin(bx)` → `Γ(p+1) sin(π(p+1)/2)/b^{p+1}` (−2 < p < 0, non-integer);
/// `x^p cos(bx)` → `Γ(p+1) cos(π(p+1)/2)/b^{p+1}` (−1 < p < 0, non-integer).
fn entry_trig_over_power(arena: &mut Arena, h: &HalfLine, x: ExprId) -> Option<ExprId> {
    let s = &h.s;
    if s.exp_arg.is_some() || !s.recip.is_empty() || s.ln.is_some() || s.hyp_recip.is_some() {
        return None;
    }
    if s.trig.len() != 1 {
        return None;
    }
    let (kind, targ, n) = s.trig[0].clone();
    let bcoef = pure_linear_coeff(arena, targ, x)?;
    // sin(bx) grows like e^{|Im b| x}: every entry needs a real b.
    if !is_real(arena, bcoef) {
        return None;
    }
    let two = Ratio::from_integer(BigInt::from(2));
    let pi = arena.pi();
    let half = arena.rational(1, 2);
    if kind == Trig::Sin && n == Ratio::one() && h.p == arena.neg_one() {
        let sg = arena.sign(bcoef);
        return Some(arena.mul(&[half, pi, sg]));
    }
    if kind == Trig::Sin && n == two && h.p == arena.int(-2) {
        let ab = arena.abs(bcoef);
        return Some(arena.mul(&[half, pi, ab]));
    }
    if n != Ratio::one() || !is_positive(arena, bcoef) {
        return None;
    }
    let pr = as_ratio(arena, h.p)?;
    if pr.is_integer() {
        return None;
    }
    let lower = if kind == Trig::Sin { -2 } else { -1 };
    if pr <= Ratio::from_integer(BigInt::from(lower)) || pr >= Ratio::zero() {
        return None;
    }
    let g = gamma_of(arena, h.p1)?;
    let ang = arena.mul(&[half, pi, h.p1]);
    let tr = match kind {
        Trig::Sin => arena.sin(ang),
        Trig::Cos => arena.cos(ang),
    };
    let neg_p1 = arena.neg(h.p1);
    let b_pow = arena.pow(bcoef, neg_p1);
    Some(arena.mul(&[g, tr, b_pow]))
}

/// `x^p (k + c x^n)^{−m}` → `(1/n) k^{s−m} c^{−s} B(s, m−s)` with
/// `s = (p+1)/n`, `0 < s < m`, `k, c > 0`.
fn entry_algebraic_beta(arena: &mut Arena, h: &HalfLine, x: ExprId) -> Option<ExprId> {
    let s = &h.s;
    if s.exp_arg.is_some() || !s.trig.is_empty() || s.ln.is_some() || s.hyp_recip.is_some() {
        return None;
    }
    if s.recip.len() != 1 {
        return None;
    }
    let (base, m) = s.recip[0].clone();
    let (k, c, n) = as_binomial_in_x(arena, base, x)?;
    if !n.is_positive() || !is_positive(arena, k) || !is_positive(arena, c) {
        return None;
    }
    let pr = as_ratio(arena, h.p)?;
    let s_val = (pr + Ratio::one()) / n.clone();
    if !s_val.is_positive() || s_val >= m {
        return None;
    }
    let s_id = ratio_expr(arena, &s_val);
    let m_minus_s = ratio_expr(arena, &(m.clone() - s_val.clone()));
    let beta = beta_of(arena, s_id, m_minus_s)?;
    let inv_n = ratio_expr(arena, &(Ratio::one() / n));
    let k_exp = ratio_expr(arena, &(s_val.clone() - m));
    let k_pow = arena.pow(k, k_exp);
    let c_exp = ratio_expr(arena, &(-s_val));
    let c_pow = arena.pow(c, c_exp);
    Some(arena.mul(&[inv_n, k_pow, c_pow, beta]))
}

/// `cos(bx)/(c x² + k)` → `π e^{−ab}/(2ac)`;  `x sin(bx)/(c x² + k)` →
/// `π e^{−ab}/(2c)` with `a = √(k/c)`, `b > 0`.
fn entry_trig_over_quadratic(arena: &mut Arena, h: &HalfLine, x: ExprId) -> Option<ExprId> {
    let s = &h.s;
    if s.exp_arg.is_some() || s.ln.is_some() || s.hyp_recip.is_some() {
        return None;
    }
    if s.recip.len() != 1 || s.trig.len() != 1 {
        return None;
    }
    let (base, m) = s.recip[0].clone();
    if m != Ratio::one() {
        return None;
    }
    let (k, c, n) = as_binomial_in_x(arena, base, x)?;
    let two = Ratio::from_integer(BigInt::from(2));
    if n != two || !is_positive(arena, k) || !is_positive(arena, c) {
        return None;
    }
    let (kind, targ, tn) = s.trig[0].clone();
    if tn != Ratio::one() {
        return None;
    }
    let bcoef = pure_linear_coeff(arena, targ, x)?;
    if !is_positive(arena, bcoef) {
        return None;
    }
    let k_over_c = arena.div(k, c);
    let a_val = arena.sqrt(k_over_c);
    let ab = arena.mul(&[a_val, bcoef]);
    let neg_ab = arena.neg(ab);
    let e = arena.exp(neg_ab);
    let half = arena.rational(1, 2);
    let pi = arena.pi();
    let neg_one = arena.neg_one();
    let inv_c = arena.pow(c, neg_one);
    let one = arena.one();
    match kind {
        Trig::Cos if arena.is_zero_structural(h.p) => {
            let inv_a = arena.pow(a_val, neg_one);
            Some(arena.mul(&[half, pi, e, inv_a, inv_c]))
        }
        Trig::Sin if h.p == one => Some(arena.mul(&[half, pi, e, inv_c])),
        _ => None,
    }
}

/// `ln(x)/(c x² + k)` → `π ln(k/c) / (4 √(kc))`.
fn entry_ln_over_quadratic(arena: &mut Arena, h: &HalfLine, x: ExprId) -> Option<ExprId> {
    let s = &h.s;
    let (larg, ln_n) = s.ln.clone()?;
    if s.exp_arg.is_some() || !s.trig.is_empty() || s.hyp_recip.is_some() {
        return None;
    }
    if s.recip.len() != 1 || !arena.is_zero_structural(h.p) || larg != x || ln_n != Ratio::one() {
        return None;
    }
    let (base, m) = s.recip[0].clone();
    if m != Ratio::one() {
        return None;
    }
    let (k, c, n) = as_binomial_in_x(arena, base, x)?;
    let two = Ratio::from_integer(BigInt::from(2));
    if n != two || !is_positive(arena, k) || !is_positive(arena, c) {
        return None;
    }
    let kc = arena.mul(&[k, c]);
    let root = arena.sqrt(kc);
    let four = arena.int(4);
    let denom = arena.mul(&[four, root]);
    let k_over_c = arena.div(k, c);
    let l = arena.ln(k_over_c);
    let pi = arena.pi();
    let num = arena.mul(&[pi, l]);
    Some(arena.div(num, denom))
}

/// Bose–Einstein / Fermi–Dirac: `x^s/(e^{ax} − 1)` → `Γ(s+1) ζ(s+1)/a^{s+1}`,
/// `x^s/(e^{ax} + 1)` → `(1 − 2^{−s}) Γ(s+1) ζ(s+1)/a^{s+1}` for odd `s`.
fn entry_bose_fermi(arena: &mut Arena, h: &HalfLine, x: ExprId) -> Option<ExprId> {
    let s = &h.s;
    if s.exp_arg.is_some() || !s.trig.is_empty() || s.ln.is_some() || s.hyp_recip.is_some() {
        return None;
    }
    if s.recip.len() != 1 {
        return None;
    }
    let (base, m) = s.recip[0].clone();
    if m != Ratio::one() {
        return None;
    }
    let terms: Vec<ExprId> = match arena.node(base).clone() {
        ExprNode::Add(ch) if ch.len() == 2 => ch.to_vec(),
        _ => return None,
    };
    let mut exp_arg = None;
    let mut const_term = None;
    for t in terms {
        if let ExprNode::Exp(arg) = arena.node(t).clone()
            && walk::contains(arena, arg, x)
        {
            exp_arg = Some(arg);
        } else if !walk::contains(arena, t, x) {
            const_term = Some(t);
        }
    }
    let (arg, kterm) = (exp_arg?, const_term?);
    let kr = as_ratio(arena, kterm)?;
    let fermi = if kr == Ratio::one() {
        true
    } else if kr == -Ratio::one() {
        false
    } else {
        return None;
    };
    let alpha = pure_linear_coeff(arena, arg, x)?;
    if !is_positive(arena, alpha) {
        return None;
    }
    let sr = as_ratio(arena, h.p)?;
    if !sr.is_integer() || !sr.is_positive() {
        return None;
    }
    let s_int = sr.to_integer().to_u32()?;
    let zeta = zeta_even(arena, s_int + 1)?;
    let g = gamma_exact(arena, &(sr + Ratio::one()))?;
    let neg_s1 = arena.neg(h.p1);
    let a_pow = arena.pow(alpha, neg_s1);
    let mut parts = vec![g, zeta, a_pow];
    if fermi {
        let two_pow = Ratio::new(BigInt::one(), BigInt::from(2).pow(s_int));
        let factor = ratio_expr(arena, &(Ratio::one() - two_pow));
        parts.push(factor);
    }
    Some(arena.mul(&parts))
}

/// `1/cosh(ax)` → `π/(2a)`;  `x/sinh(ax)` → `π²/(4a²)`   (`a > 0`).
fn entry_hyperbolic_recip(arena: &mut Arena, h: &HalfLine, x: ExprId) -> Option<ExprId> {
    let s = &h.s;
    let (is_cosh, harg, m) = s.hyp_recip.clone()?;
    if s.exp_arg.is_some() || !s.trig.is_empty() || !s.recip.is_empty() || s.ln.is_some() {
        return None;
    }
    if m != Ratio::one() {
        return None;
    }
    let alpha = pure_linear_coeff(arena, harg, x)?;
    if !is_positive(arena, alpha) {
        return None;
    }
    let pi = arena.pi();
    let one = arena.one();
    if is_cosh && arena.is_zero_structural(h.p) {
        let two = arena.int(2);
        let two_a = arena.mul(&[two, alpha]);
        return Some(arena.div(pi, two_a));
    }
    if !is_cosh && h.p == one {
        let pi2 = arena.mul(&[pi, pi]);
        let four = arena.int(4);
        let a2 = arena.mul(&[alpha, alpha]);
        let denom = arena.mul(&[four, a2]);
        return Some(arena.div(pi2, denom));
    }
    None
}

/// Table for `∫₋∞^∞`: general Gaussian; otherwise even integrands reduce
/// to twice the half-line table (odd ones are handled by the parity
/// shortcut before we get here).
fn table_full_line(arena: &mut Arena, dep: &[ExprId], x: ExprId) -> Option<ExprId> {
    let s = parse_shape(arena, dep, x);
    if !s.other
        && let Some(arg) = s.exp_arg
        && s.trig.is_empty()
        && s.recip.is_empty()
        && s.ln.is_none()
        && s.hyp_recip.is_none()
    {
        let p = s.x_pow.unwrap_or(arena.zero());
        let [c0, c1, c2] = quadratic_coeffs(arena, arg, x)?;
        let a = arena.neg(c2);
        let a = safe_eval(arena, a);
        if !is_positive(arena, a) {
            return None;
        }
        // ∫ e^{−a x² + c1 x + c0} = √(π/a) e^{c0 + c1²/(4a)}
        let pi = arena.pi();
        let pi_over_a = arena.div(pi, a);
        let root = arena.sqrt(pi_over_a);
        let c1_sq = arena.mul(&[c1, c1]);
        let four_a = {
            let four = arena.int(4);
            arena.mul(&[four, a])
        };
        let shift = arena.div(c1_sq, four_a);
        let e_arg = arena.add(&[c0, shift]);
        let e = arena.exp(e_arg);
        if arena.is_zero_structural(p) {
            return Some(arena.mul(&[root, e]));
        }
        // x e^{−a x² + c1 x + c0} = (c1/(2a)) √(π/a) e^{…}
        if p == arena.one() {
            let two_a = {
                let two = arena.int(2);
                arena.mul(&[two, a])
            };
            let mean = arena.div(c1, two_a);
            return Some(arena.mul(&[mean, root, e]));
        }
        return None;
    }
    None
}

/// Table for `∫₀¹`.
fn table_zero_one(arena: &mut Arena, dep: &[ExprId], x: ExprId) -> Option<ExprId> {
    let s = parse_shape(arena, dep, x);
    let p = s.x_pow.unwrap_or(arena.zero());
    let one = arena.one();
    let p_plus_1 = arena.add(&[p, one]);
    let p_plus_1 = safe_eval(arena, p_plus_1);

    // ── x^p ln(x)^n → (−1)^n n! / (p+1)^{n+1} ─────────────────────
    if let Some((larg, n)) = s.ln.clone()
        && larg == x
        && s.recip.is_empty()
        && s.trig.is_empty()
        && s.exp_arg.is_none()
        && s.hyp_recip.is_none()
        && !s.other
    {
        if !n.is_integer() || !n.is_positive() || !is_positive(arena, p_plus_1) {
            return None;
        }
        let ni = n.to_integer().to_u32()?;
        if ni > 40 {
            return None;
        }
        let mut fact = BigInt::one();
        for j in 2..=ni {
            fact *= BigInt::from(j);
        }
        let sign = if ni % 2 == 0 { 1 } else { -1 };
        let c = ratio_expr(arena, &Ratio::from_integer(fact * BigInt::from(sign)));
        let neg_n1 = arena.int(-(ni as i64) - 1);
        let denom = arena.pow(p_plus_1, neg_n1);
        return Some(arena.mul(&[c, denom]));
    }

    // ── Beta: x^p (1 − x^n)^q  (possibly with q = 1 as a bare Add) ──
    // Collect the non-power factors as (base, q).
    let mut beta_factors: Vec<(ExprId, ExprId)> = Vec::new();
    let mut ok = true;
    for &d in dep {
        if as_x_power(arena, d, x).is_some() {
            continue;
        }
        match arena.node(d).clone() {
            ExprNode::Pow(base, exp) if !walk::contains(arena, exp, x) => {
                beta_factors.push((base, exp))
            }
            ExprNode::Add(_) => beta_factors.push((d, one)),
            _ => ok = false,
        }
    }
    if ok && beta_factors.len() == 1 {
        let (base, q) = beta_factors[0];
        // base = β − β x^n  (β > 0)
        let coeffs = calculus_util::poly_coeffs_symbolic(arena, base, x)?;
        let n = coeffs.len().checked_sub(1)?;
        if n >= 1 && coeffs[1..n].iter().all(|&c| arena.is_zero_structural(c)) {
            let beta_c = coeffs[0];
            let lead = coeffs[n];
            let neg_lead = arena.neg(lead);
            let neg_lead = safe_eval(arena, neg_lead);
            if cmp_points(arena, beta_c, neg_lead) == Some(Ordering::Equal)
                && is_positive(arena, beta_c)
            {
                let q_plus_1 = arena.add(&[q, one]);
                let q_plus_1 = safe_eval(arena, q_plus_1);
                if !is_positive(arena, q_plus_1) || !is_positive(arena, p_plus_1) {
                    return None;
                }
                let n_r = Ratio::from_integer(BigInt::from(n));
                let inv_n = ratio_expr(arena, &(Ratio::one() / n_r.clone()));
                let s_arg = arena.mul(&[inv_n, p_plus_1]);
                let s_arg = safe_eval(arena, s_arg);
                let b = beta_of(arena, s_arg, q_plus_1)?;
                let beta_pow = arena.pow(beta_c, q);
                return Some(arena.mul(&[inv_n, beta_pow, b]));
            }
        }
        return None;
    }

    // ── Dilogarithm-type: ln(L₁)/L₂ with linear/quadratic L ────────
    // The denominator is either a reciprocal factor or a bare x⁻¹.
    let dilog_denominator: Option<ExprId> = if s.recip.len() == 1 && s.x_pow.is_none() {
        if s.recip[0].1 == Ratio::one() {
            Some(s.recip[0].0)
        } else {
            None
        }
    } else if s.recip.is_empty() && s.x_pow == Some(arena.neg_one()) {
        Some(x)
    } else {
        None
    };
    if let Some((larg, n)) = s.ln.clone()
        && n == Ratio::one()
        && let Some(base) = dilog_denominator
        && s.trig.is_empty()
        && s.exp_arg.is_none()
        && s.hyp_recip.is_none()
        && !s.other
    {
        // Classify the log argument: x, 1−x, 1+x.
        #[derive(PartialEq, Clone, Copy)]
        enum L {
            X,
            OneMinusX,
            OnePlusX,
        }
        let classify = |arena: &mut Arena, e: ExprId| -> Option<(L, ExprId)> {
            if e == x {
                return Some((L::X, arena.one()));
            }
            let c = calculus_util::poly_coeffs_symbolic(arena, e, x)?;
            match c.len() {
                2 => {
                    let c0 = as_ratio(arena, c[0])?;
                    let c1 = as_ratio(arena, c[1])?;
                    if c0.is_zero() {
                        // c1 x
                        return Some((L::X, c[1]));
                    }
                    if c0 == -c1.clone() {
                        return Some((L::OneMinusX, c[0]));
                    }
                    if c0 == c1 {
                        return Some((L::OnePlusX, c[0]));
                    }
                    None
                }
                _ => None,
            }
        };
        let (lk, lscale) = classify(arena, larg)?;
        if !arena.is_one_structural(lscale) && lk != L::X {
            return None; // ln(k(1−x)) splits into ln k + ln(1−x); skip
        }
        // Denominator: k·x, k(1−x), k(1+x), k(1−x²)
        let dc = calculus_util::poly_coeffs_symbolic(arena, base, x)?;
        let pi = arena.pi();
        let pi2 = arena.mul(&[pi, pi]);
        let value_over = |arena: &mut Arena, num: i64, den: i64, k: ExprId| -> Option<ExprId> {
            let r = arena.rational(num, den);
            let v = arena.mul(&[r, pi2]);
            let out = arena.div(v, k);
            Some(safe_eval(arena, out))
        };
        return match dc.len() {
            2 => {
                let d0 = as_ratio(arena, dc[0])?;
                let d1 = as_ratio(arena, dc[1])?;
                if d0.is_zero() {
                    // ln(L)/(k x)
                    let k = dc[1];
                    return match lk {
                        L::OneMinusX => value_over(arena, -1, 6, k),
                        L::OnePlusX => value_over(arena, 1, 12, k),
                        L::X => None, // divergent (ln²)
                    };
                }
                if d0 == -d1.clone() {
                    // k(1 − x)
                    let k = dc[0];
                    return match lk {
                        L::X => {
                            if arena.is_one_structural(lscale) {
                                value_over(arena, -1, 6, k)
                            } else {
                                None
                            }
                        }
                        L::OnePlusX => {
                            // ∫₀¹ ln(1+x)/(1−x) diverges
                            None
                        }
                        L::OneMinusX => None, // divergent
                    };
                }
                if d0 == d1 {
                    // k(1 + x)
                    let k = dc[0];
                    return match lk {
                        L::X if arena.is_one_structural(lscale) => value_over(arena, -1, 12, k),
                        _ => None,
                    };
                }
                None
            }
            3 => {
                let d0 = as_ratio(arena, dc[0])?;
                let d1 = as_ratio(arena, dc[1])?;
                let d2 = as_ratio(arena, dc[2])?;
                if d1.is_zero()
                    && d0 == -d2.clone()
                    && lk == L::X
                    && arena.is_one_structural(lscale)
                {
                    // ln(x)/(k(1 − x²)) → −π²/(8k)
                    let k = dc[0];
                    return value_over(arena, -1, 8, k);
                }
                None
            }
            _ => None,
        };
    }

    None
}

/// Double factorial `n!!` (with `(−1)!! = 0!! = 1`).
fn double_factorial(n: i64) -> BigInt {
    let mut acc = BigInt::one();
    let mut k = n;
    while k > 1 {
        acc *= BigInt::from(k);
        k -= 2;
    }
    acc
}

/// Wallis: `∫₀^{π/2} sin^m cos^n = ((m−1)!!(n−1)!!/(m+n)!!) · (π/2 if m, n even)`
/// for non-negative integers; `½ B((m+1)/2, (n+1)/2)` otherwise (m, n > −1).
fn wallis(arena: &mut Arena, m: &Q, n: &Q) -> Option<ExprId> {
    if m.is_integer() && n.is_integer() && !m.is_negative() && !n.is_negative() {
        let mi = m.to_integer().to_i64()?;
        let ni = n.to_integer().to_i64()?;
        if mi + ni > 400 {
            return None;
        }
        let num = double_factorial(mi - 1) * double_factorial(ni - 1);
        let den = double_factorial(mi + ni);
        let r = Ratio::new(num, den);
        let r_id = ratio_expr(arena, &r);
        if mi % 2 == 0 && ni % 2 == 0 {
            let half = arena.rational(1, 2);
            let pi = arena.pi();
            return Some(arena.mul(&[r_id, half, pi]));
        }
        return Some(r_id);
    }
    let one = Ratio::one();
    let two = Ratio::from_integer(BigInt::from(2));
    let mp = (m + &one) / &two;
    let np = (n + &one) / &two;
    if !mp.is_positive() || !np.is_positive() {
        return None;
    }
    let mp_id = ratio_expr(arena, &mp);
    let np_id = ratio_expr(arena, &np);
    let b = beta_of(arena, mp_id, np_id)?;
    let half = arena.rational(1, 2);
    Some(arena.mul(&[half, b]))
}

/// Table for trigonometric intervals `[0, π/2]`, `[0, π]`, `[0, 2π]`, `[−π, π]`.
fn table_trig(arena: &mut Arena, dep: &[ExprId], x: ExprId, iv: IntervalShape) -> Option<ExprId> {
    let s = parse_shape(arena, dep, x);
    if s.other
        || s.exp_arg.is_some()
        || s.ln.is_some()
        || s.hyp_recip.is_some()
        || s.x_pow.is_some()
    {
        return None;
    }

    // ── Powers of sin / cos with argument x, or a single power of
    //    sin(kx) / cos(kx) with integer k over a (half) period ────────
    let single_multiple: Option<i64> =
        if s.recip.is_empty() && s.trig.len() == 1 && s.trig[0].1 != x {
            let kc = pure_linear_coeff(arena, s.trig[0].1, x);
            kc.and_then(|k| as_ratio(arena, k))
                .filter(|k| k.is_integer() && !k.is_zero())
                .and_then(|k| k.to_integer().to_i64())
        } else {
            None
        };
    if s.recip.is_empty()
        && !s.trig.is_empty()
        && (s.trig.iter().all(|(_, arg, _)| *arg == x) || single_multiple.is_some())
    {
        let mut m = Ratio::zero();
        let mut n = Ratio::zero();
        for (kind, _, k) in &s.trig {
            match kind {
                Trig::Sin => m += k,
                Trig::Cos => n += k,
            }
        }
        let neg_one = -Ratio::one();
        if m <= neg_one || n <= neg_one {
            return None;
        }
        let both_int = m.is_integer() && n.is_integer();
        let m_odd = both_int && m.to_integer().to_i64()? % 2 != 0;
        let n_odd = both_int && n.to_integer().to_i64()? % 2 != 0;
        if let Some(k) = single_multiple {
            // sinⁿ(kx) / cosⁿ(kx): substitution u = kx.  Over a full
            // period the value equals the k = 1 value; over [0, π] as well,
            // except that odd powers of sin(kx) with even k cancel.  Only
            // integer powers are supported (sign changes otherwise).
            if !both_int || iv == IntervalShape::ZeroHalfPi {
                return None;
            }
            if iv == IntervalShape::ZeroPi && m_odd {
                // ∫₀^π sinᵐ(kx) = (1/k) Σⱼ (−1)ʲ ∫₀^π sinᵐ: zero for even k,
                // W_π/k for odd k.
                if k % 2 == 0 {
                    return Some(arena.zero());
                }
                let w = wallis(arena, &m, &n)?;
                let two_over_k = arena.rational(2, k);
                return Some(arena.mul(&[two_over_k, w]));
            }
        }
        return match iv {
            IntervalShape::ZeroHalfPi => wallis(arena, &m, &n),
            IntervalShape::ZeroPi => {
                // cos changes sign on (π/2, π): need integer n.
                if !n.is_integer() {
                    return None;
                }
                if n_odd {
                    return Some(arena.zero());
                }
                let w = wallis(arena, &m, &n)?;
                let two = arena.int(2);
                Some(arena.mul(&[two, w]))
            }
            IntervalShape::ZeroTwoPi | IntervalShape::NegPiPi => {
                if !both_int {
                    return None;
                }
                if m_odd || n_odd {
                    return Some(arena.zero());
                }
                let w = wallis(arena, &m, &n)?;
                let four = arena.int(4);
                Some(arena.mul(&[four, w]))
            }
            _ => None,
        };
    }

    // ── Orthogonality: sin(mx)sin(nx), cos(mx)cos(nx), sin(mx)cos(nx) ──
    if s.recip.is_empty() && s.trig.len() == 2 {
        let (k1, a1, n1) = s.trig[0].clone();
        let (k2, a2, n2) = s.trig[1].clone();
        if n1 != Ratio::one() || n2 != Ratio::one() {
            return None;
        }
        let m1 = pure_linear_coeff(arena, a1, x)?;
        let m2 = pure_linear_coeff(arena, a2, x)?;
        let m1 = as_ratio(arena, m1)?;
        let m2 = as_ratio(arena, m2)?;
        if !m1.is_integer() || !m2.is_integer() {
            return None;
        }
        let (mi, ni) = (m1.to_integer().to_i64()?, m2.to_integer().to_i64()?);
        if mi == 0 || ni == 0 {
            return None;
        }
        let pi = arena.pi();
        let same_kind = k1 == k2;
        return match iv {
            IntervalShape::ZeroTwoPi | IntervalShape::NegPiPi => {
                if !same_kind {
                    return Some(arena.zero());
                }
                if mi.abs() == ni.abs() {
                    // sin·sin: π·sign(mn); cos·cos: π
                    if k1 == Trig::Sin && (mi > 0) != (ni > 0) {
                        return Some(arena.neg(pi));
                    }
                    return Some(pi);
                }
                Some(arena.zero())
            }
            IntervalShape::ZeroPi => {
                if same_kind {
                    if mi.abs() == ni.abs() {
                        let half = arena.rational(1, 2);
                        let v = arena.mul(&[half, pi]);
                        if k1 == Trig::Sin && (mi > 0) != (ni > 0) {
                            return Some(arena.neg(v));
                        }
                        return Some(v);
                    }
                    return Some(arena.zero());
                }
                // sin(m x) cos(n x) on [0, π]: (1 − (−1)^{m+n}) m/(m² − n²)
                let (sm, cn) = if k1 == Trig::Sin { (mi, ni) } else { (ni, mi) };
                if sm.abs() == cn.abs() {
                    return Some(arena.zero());
                }
                if (sm + cn) % 2 == 0 {
                    return Some(arena.zero());
                }
                let r = Ratio::new(BigInt::from(2 * sm), BigInt::from(sm * sm - cn * cn));
                Some(ratio_expr(arena, &r))
            }
            _ => None,
        };
    }

    // ── 1/(a + b cos x)^m, 1/(a + b sin x)^m on a full period ───────
    if s.trig.is_empty() && s.recip.len() == 1 {
        let (base, m) = s.recip[0].clone();
        let terms: Vec<ExprId> = match arena.node(base).clone() {
            ExprNode::Add(ch) if ch.len() >= 2 => ch.to_vec(),
            _ => return None,
        };
        // The constant part may be a sum (`b + 1 + cos x`).
        let free: Vec<ExprId> = terms
            .iter()
            .copied()
            .filter(|&t| !walk::contains(arena, t, x))
            .collect();
        if free.is_empty() || terms.len() - free.len() != 1 {
            return None;
        }
        let a_c: Option<ExprId> = Some(arena.add(&free));
        let mut b_c: Option<(ExprId, Trig)> = None;
        for t in terms {
            if !walk::contains(arena, t, x) {
                continue;
            }
            let (coeff, d) = factorize(arena, t, x);
            if d.len() != 1 {
                return None;
            }
            let (kind, arg, n) = as_trig_power(arena, d[0])?;
            if arg != x || n != Ratio::one() {
                return None;
            }
            let bc = if coeff.is_empty() {
                arena.one()
            } else {
                let mm = arena.mul(&coeff);
                safe_eval(arena, mm)
            };
            b_c = Some((bc, kind));
        }
        let (a_c, (b_c, kind)) = (a_c?, b_c?);
        // Need a > |b|.
        let abs_b = arena.abs(b_c);
        let diff_ab = arena.sub(a_c, abs_b);
        let diff_ab = safe_eval(arena, diff_ab);
        if !is_positive(arena, diff_ab) {
            return None;
        }
        let a2 = arena.mul(&[a_c, a_c]);
        let b2 = arena.mul(&[b_c, b_c]);
        let disc = arena.sub(a2, b2);
        let pi = arena.pi();
        let full = match iv {
            IntervalShape::ZeroTwoPi | IntervalShape::NegPiPi => true,
            IntervalShape::ZeroPi if kind == Trig::Cos => false,
            _ => return None,
        };
        if m == Ratio::one() {
            // 2π/√(a²−b²)   (π/√(a²−b²) on [0, π] for cos)
            let root = arena.sqrt(disc);
            let c = if full { arena.int(2) } else { arena.one() };
            let num = arena.mul(&[c, pi]);
            return Some(arena.div(num, root));
        }
        if m == Ratio::from_integer(BigInt::from(2)) {
            // 2π a/(a²−b²)^{3/2}
            let e = arena.rational(-3, 2);
            let pw = arena.pow(disc, e);
            let c = if full { arena.int(2) } else { arena.one() };
            return Some(arena.mul(&[c, pi, a_c, pw]));
        }
        return None;
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Numeric quadrature: adaptive Gauss–Kronrod G7/K15
// ═══════════════════════════════════════════════════════════════════════════

/// Options for [`quadrature`] and `Ex::integrate_numeric_with`.
///
/// Iteration stops once the global error estimate drops below
/// `max(abs_tol, rel_tol · |value|)` or `max_subdivisions` bisections have
/// been performed.
///
/// ```
/// use symplex::definite::QuadOpts;
///
/// let opts = QuadOpts { rel_tol: 1e-8, ..QuadOpts::default() };
/// assert_eq!(opts.max_subdivisions, QuadOpts::default().max_subdivisions);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadOpts {
    /// Relative tolerance (default `1e-10`).
    pub rel_tol: f64,
    /// Absolute tolerance (default `1e-12`).
    pub abs_tol: f64,
    /// Maximum number of interval bisections (default `2000`).
    pub max_subdivisions: usize,
}

impl Default for QuadOpts {
    fn default() -> Self {
        Self {
            rel_tol: 1e-10,
            abs_tol: 1e-12,
            max_subdivisions: 2000,
        }
    }
}

/// The rule of the arbitrary-precision quadrature behind
/// `Ex::integrate_numeric_decimal` (mpmath's `quad(…, method=…)`).
/// `eval_decimal` of an unevaluated definite integral uses
/// [`QuadRule::TanhSinh`].
///
/// ```
/// use symplex::definite::QuadRule;
///
/// assert_eq!(QuadRule::default(), QuadRule::TanhSinh);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QuadRule {
    /// Tanh-sinh (double exponential): algebraic and logarithmic endpoint
    /// singularities and infinite ranges at full precision.  A finite piece
    /// it does not resolve (an oscillation) is tried by Gauss–Legendre, then
    /// bisected.
    #[default]
    TanhSinh,
    /// Gauss–Legendre of degree `3·2^(m−1)`: fewer nodes for a smooth
    /// integrand, slow at an endpoint singularity.  Infinite ranges are
    /// mapped as for tanh-sinh; a piece it does not resolve is bisected.
    GaussLegendre,
}

/// Result of [`quadrature`] and `Ex::integrate_numeric_with`: the integral
/// estimate and the estimated absolute error of that estimate.
///
/// ```
/// use symplex::definite::{quadrature, QuadOpts, QuadResult};
///
/// let QuadResult { value, error } =
///     quadrature(&|x: f64| x * x, 0.0, 1.0, &QuadOpts::default()).unwrap();
/// assert!((value - 1.0 / 3.0).abs() < 1e-12 && error < 1e-10);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadResult {
    /// The estimate of the integral.
    pub value: f64,
    /// Estimated absolute error of [`value`](Self::value): the sum of
    /// QUADPACK's estimates of the segments, a heuristic (a feature no node
    /// sees is not in it).  A segment the bisection cannot split — next to
    /// a singularity at a nonzero end or interior point, which `f64` nodes
    /// cannot approach closer than its rounding — counts at least its width
    /// times the largest value at its nodes, which covers the mass the nodes
    /// miss for `|x − c|^α` with `α` down to about `−0.99`.
    /// `Ex::integrate_numeric_with` serves an arbitrary-precision value
    /// instead when the estimate exceeds the tolerance.
    pub error: f64,
}

/// Kronrod-15 abscissae (positive half; index 7 is the centre).
const XGK: [f64; 8] = [
    0.9914553711208126,
    0.9491079123427585,
    0.8648644233597691,
    0.7415311855993945,
    0.5860872354676911,
    0.4058451513773972,
    0.20778495500789848,
    0.0,
];

/// Kronrod-15 weights.
const WGK: [f64; 8] = [
    0.022935322010529224,
    0.06309209262997856,
    0.10479001032225019,
    0.14065325971552592,
    0.1690047266392679,
    0.19035057806478542,
    0.20443294007529889,
    0.20948214108472782,
];

/// Gauss-7 weights (for abscissae `XGK[1], XGK[3], XGK[5], XGK[7]`).
const WG: [f64; 4] = [
    0.1294849661688697,
    0.27970539148927664,
    0.3818300505051189,
    0.4179591836734694,
];

/// One G7/K15 rule application on `[a, b]`.  Returns `(k15, err, peak)`;
/// `err` is `NaN` if the integrand was non-finite at a node.  The error
/// estimate is QUADPACK's `resasc·min(1, (200·|K15 − G7|/resasc)^1.5)`, or
/// with `raw` `|K15 − G7|` itself: the reduction (up to about 35-fold where
/// `|K15 − G7|` is small next to `resasc`) assumes the integrand smooth on
/// the segment, the Kronrod error far below the Gauss one.  The estimate is
/// at least `50ε` times the result, and without `raw` times `resabs`, the
/// rule applied to `|f|`, as in QUADPACK: the rounding of a sum that cancels
/// is that of its terms (before 0.44 `∫₋₂⁻¹ sin(2x/3 + 1) dx`, exactly 0,
/// was `3.9·10⁻¹⁷` with an estimate of `3·10⁻²³`).  The caller with
/// `raw` (`evalf`) adds the rounding of the values over the range itself.
/// Without `raw` the estimate also covers an end where the values grow like
/// `|x − end|^α`, `α < −¾`, between the two outermost nodes: twice the mass
/// between the outermost node and the end, `d₀|f₀|/(1 + α)`, extrapolated
/// from them, and is infinite for a faster rise (`α ≤ −1`: a divergence, or
/// an exponential the segment does not resolve, which is bisected until it
/// does) (QUADPACK leaves that to its ε-algorithm; its estimate, the
/// spread of the values, falls short of the error as `α → −1`: before 0.44
/// `∫₀¹ x^(−19/20)·sin(x/4 + 2)/(1 + 5x²) dx` was `3.7·10⁻⁹` off with an
/// estimate of `1.7·10⁻⁹`).  `peak` is `|b −
/// a|·max|f|` over the nodes, the error of a segment the bisection cannot
/// split ([`Floor`]).
fn gk15_with(f: &dyn Fn(f64) -> f64, a: f64, b: f64, raw: bool) -> (f64, f64, f64) {
    let centre = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let fc = f(centre);
    let mut k15 = fc * WGK[7];
    let mut g7 = fc * WG[3];
    let mut fv = [0.0f64; 15];
    fv[7] = fc;
    let mut nonfinite = !fc.is_finite();
    // Nodes that round onto an endpoint (possible for very narrow
    // segments) are nudged one ulp inward so endpoint singularities are
    // never sampled exactly.
    let inward = |x: f64| -> f64 {
        if x <= a {
            a.next_up()
        } else if x >= b {
            b.next_down()
        } else {
            x
        }
    };
    for j in 0..7 {
        let dx = half * XGK[j];
        let f1 = f(inward(centre - dx));
        let f2 = f(inward(centre + dx));
        nonfinite |= !f1.is_finite() || !f2.is_finite();
        fv[j] = f1;
        fv[14 - j] = f2;
        k15 += WGK[j] * (f1 + f2);
        if j % 2 == 1 {
            g7 += WG[j / 2] * (f1 + f2);
        }
    }
    if nonfinite {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let peak = fv.iter().fold(0.0f64, |m, v| m.max(v.abs())) * (b - a).abs();
    let mean = 0.5 * k15;
    let mut resasc = WGK[7] * (fc - mean).abs();
    for j in 0..7 {
        resasc += WGK[j] * ((fv[j] - mean).abs() + (fv[14 - j] - mean).abs());
    }
    let result = k15 * half;
    let resasc = resasc * half.abs();
    let mut err = ((k15 - g7) * half).abs();
    if !raw && resasc != 0.0 && err != 0.0 {
        err = resasc * (200.0 * err / resasc).powf(1.5).min(1.0);
    }
    let resabs = if raw {
        result.abs()
    } else {
        let mut s = WGK[7] * fc.abs();
        for j in 0..7 {
            s += WGK[j] * (fv[j].abs() + fv[14 - j].abs());
        }
        s * half.abs()
    };
    let resabs_floor = 50.0 * f64::EPSILON * resabs;
    if resabs_floor > err {
        err = resabs_floor;
    }
    // Nodes that round onto each other (a segment of a few hundred ulps of
    // its position): the rule sees fewer values than it weighs, K15 and G7
    // agree on them, and their difference says nothing.  Up to 0.49 such a
    // segment was "resolved": `∫_{10¹⁶}^{10¹⁶+4} |sin(x/3 + 1/2)| dx` was
    // `3.0537` (truly `2.8325`), its nodes on a grid of spacing 2.
    let nodes_distinct = (0..7).all(|j| {
        let (o, i) = (half * XGK[j], half * XGK[j + 1]);
        inward(centre - o) < inward(centre - i) && inward(centre + i) < inward(centre + o)
    });
    if !nodes_distinct {
        err = err.max(peak);
    }
    if !raw {
        // The outermost nodes lie at `(1 − XGK[0])` and `(1 − XGK[1])`
        // half-widths from the end.  A rise faster than `|x − end|⁻¹` (a
        // divergence, or an exponential the segment does not resolve, the
        // tail of `sech(10⁴x)`) bounds nothing: the segment is bisected.
        let ratio = (1.0 - XGK[1]) / (1.0 - XGK[0]);
        let d0 = half.abs() * (1.0 - XGK[0]);
        let tail = |f0: f64, f1: f64| -> f64 {
            let growth = (f0 / f1).abs();
            if f0 * f1 <= 0.0 || growth <= ratio.powf(0.75) {
                return 0.0;
            }
            let alpha = -growth.ln() / ratio.ln();
            if alpha <= -1.0 {
                return f64::INFINITY;
            }
            2.0 * d0 * f0.abs() / (1.0 + alpha)
        };
        err = err.max(tail(fv[0], fv[1])).max(tail(fv[14], fv[13]));
    }
    (result, err, peak)
}

/// Adaptive Gauss–Kronrod quadrature of `f` over `[a, b]`.
///
/// Infinite bounds are handled by the substitutions `x = a + t/(1−t)` and
/// `x = b − t/(1−t)` (the whole line split at 0); endpoint singularities
/// are absorbed by adaptive bisection (the rule never samples endpoints)
/// and, when that does not meet the tolerance, by a retry under an
/// endpoint-smoothing substitution at the ends that `f64` nodes can
/// approach: a finite end at 0 and an infinite end (see
/// [`QuadResult::error`] for what the estimate covers).
///
/// Returns the estimate and its error estimate as a [`QuadResult`], also
/// when the tolerance was not met.  If the integrand is non-finite on a
/// vanishing interval (a non-integrable interior singularity) or the
/// estimate is not a finite number, `Err(ComputationFailed)` is returned.
///
/// Before 0.44 the smoothing retry was applied to the whole range, so its
/// nodes crowded towards a nonzero finite end (and towards the end of the
/// map of an infinite range), which in `f64` they cannot approach closer
/// than half an ulp: the integrand was cut off at `1 − x ≈ 2⁻⁵³` (at `x ≈
/// 10¹⁶`) and the estimate did not see it.  `∫₀¹ dx/√(1 − x)` was
/// `1.9999999898808887` with an estimate of `3·10⁻¹⁰`, and so was
/// `∫₁^∞ x^(−3/2) dx`; `∫₀¹ log²(1 − x)/√(1 − x) dx` was `15.99998`
/// (truly 16) with an estimate of `3·10⁻⁷`.
///
/// ```
/// use symplex::definite::{quadrature, QuadOpts};
///
/// let r = quadrature(&|x: f64| x.sin(), 0.0, std::f64::consts::PI, &QuadOpts::default()).unwrap();
/// assert!((r.value - 2.0).abs() < 1e-12);
/// assert!(r.error < 1e-8);
///
/// // ∫₀^∞ e^{−x²} dx = √π / 2
/// let r = quadrature(&|x: f64| (-x * x).exp(), 0.0, f64::INFINITY, &QuadOpts::default()).unwrap();
/// assert!((r.value - std::f64::consts::PI.sqrt() / 2.0).abs() < 1e-10);
/// ```
pub fn quadrature(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    opts: &QuadOpts,
) -> Result<QuadResult, SymplexError> {
    quadrature_report(f, a, b, opts).map(|(r, _)| r)
}

/// [`quadrature`], and whether every segment was resolved: none left at the
/// narrowest width the bisection can make, where the estimate of a segment
/// is only `width·max|f|` (see [`QuadResult::error`]).
///
/// The plain bisection runs first, its segments too narrow when below
/// `10⁻¹²` of their own position ([`Floor::Relative`]: next to an end at 0
/// it goes as deep as `f64` allows).  When it is not resolved within the
/// tolerance, the retry of [`smoothed_pieces`] runs, and the result with
/// the smaller error estimate is kept.
///
/// # Errors
///
/// As [`quadrature`].
pub(crate) fn quadrature_report(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    opts: &QuadOpts,
) -> Result<(QuadResult, bool), SymplexError> {
    if a.is_nan() || b.is_nan() {
        return Err(SymplexError::InvalidArgument {
            operation: "integrate_numeric",
            reason: "integration bounds must not be NaN".into(),
        });
    }
    if a == b {
        return Ok((
            QuadResult {
                value: 0.0,
                error: 0.0,
            },
            true,
        ));
    }
    if a > b {
        let (r, resolved) = quadrature_report(f, b, a, opts)?;
        return Ok((
            QuadResult {
                value: -r.value,
                error: r.error,
            },
            resolved,
        ));
    }
    if a.is_infinite() && b.is_infinite() {
        // Split at 0 so that each half must converge on its own: a single
        // symmetric rule on t/(1−t²) would report ∫ x dx = 0 with zero
        // error (a numeric principal value).
        let half = QuadOpts {
            max_subdivisions: opts.max_subdivisions.div_ceil(2),
            ..*opts
        };
        let (left, l_ok) = quadrature_report(f, f64::NEG_INFINITY, 0.0, &half)?;
        let (right, r_ok) = quadrature_report(f, 0.0, f64::INFINITY, &half)?;
        return Ok((
            QuadResult {
                value: left.value + right.value,
                error: left.error + right.error,
            },
            l_ok && r_ok,
        ));
    }
    let plain = plain_report(f, a, b, opts, false, Floor::Relative(None));
    if let Ok((r, true)) = &plain
        && r.error <= opts.abs_tol.max(opts.rel_tol * r.value.abs())
    {
        return plain;
    }
    let Some(retry) = smoothed_pieces(f, a, b, opts, false) else {
        return plain;
    };
    match (plain, retry) {
        (Ok(p), Ok(m)) => Ok(if m.0.error < p.0.error { m } else { p }),
        (Ok(p), Err(_)) => Ok(p),
        (Err(_), Ok(m)) => Ok(m),
        (Err(e), Err(_)) => Err(e),
    }
}

/// The plain bisection of [`quadrature_report`] and [`quadrature_resolved`]
/// on a finite or half-infinite range `a < b`: on `[a, b]` itself, or on
/// `t ∈ [0, 1]` under `x = a + t/(1−t)` (`x = b − t/(1−t)`), the floor of
/// [`Floor::Relative`] judged in `x` too.
fn plain_report(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    opts: &QuadOpts,
    raw: bool,
    floor: Floor<'_>,
) -> Result<(QuadResult, bool), SymplexError> {
    let relative = matches!(floor, Floor::Relative(_));
    match (a.is_finite(), b.is_finite()) {
        (true, true) => quadrature_finite_report(f, a, b, opts, raw, floor),
        (true, false) => {
            // x = a + t/(1−t), t ∈ [0, 1), dx = dt/(1−t)²
            let g = move |t: f64| {
                let om = 1.0 - t;
                f(a + t / om) / (om * om)
            };
            let x = move |t: f64| a + t / (1.0 - t);
            let floor = if relative {
                Floor::Relative(Some(&x))
            } else {
                floor
            };
            quadrature_finite_report(&g, 0.0, 1.0, opts, raw, floor)
        }
        (false, true) => {
            // x = b − t/(1−t)
            let g = move |t: f64| {
                let om = 1.0 - t;
                f(b - t / om) / (om * om)
            };
            let x = move |t: f64| b - t / (1.0 - t);
            let floor = if relative {
                Floor::Relative(Some(&x))
            } else {
                floor
            };
            quadrature_finite_report(&g, 0.0, 1.0, opts, raw, floor)
        }
        (false, false) => Err(SymplexError::ComputationFailed {
            operation: "integrate_numeric",
            reason: "the whole real line is split at 0 before the bisection".into(),
        }),
    }
}

/// The retry of [`quadrature_report`] (and of [`quadrature_resolved`] with
/// `smoothing`): `[a, b]` split so that each end the `f64` nodes can
/// approach — a finite end at 0, an infinite end — is the end at 0 of a
/// piece in a variable of its own, integrated under the endpoint-smoothing
/// substitution of [`smoothed`]; a nonzero finite end is left to plain
/// bisection.  An infinite end becomes `u → 0⁺` under `x = e ± (1/u − 1)`
/// on the piece beyond `e ± 1`, where the nodes keep their relative
/// distance to it (under `x = e ± t/(1 − t)` they cannot come closer to `t =
/// 1` than `2⁻⁵³`, which cuts the integrand off at `x ≈ 10¹⁶`).  `None` when
/// no end is such (both finite and nonzero): the retry would be the plain
/// bisection again.
///
/// The pieces share the tolerance and the subdivisions.
fn smoothed_pieces(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    opts: &QuadOpts,
    raw: bool,
) -> Option<Result<(QuadResult, bool), SymplexError>> {
    let share = QuadOpts {
        rel_tol: 0.5 * opts.rel_tol,
        abs_tol: 0.5 * opts.abs_tol,
        max_subdivisions: opts.max_subdivisions.div_ceil(2),
    };
    // The floor judged in the piece's own variable too: where `s(t)`
    // underflows, the end at 0 is cut off (beyond `x ≈ 10³⁰⁸` for an
    // infinite end) and no estimate sees it.
    let smooth = |g: &dyn Fn(f64) -> f64, h: f64| {
        let s = smoothed(g, 0.0, h);
        let position = move |t: f64| smoothing_map(0.0, h, t);
        quadrature_finite_report(&s, 0.0, 1.0, &share, raw, Floor::Relative(Some(&position)))
    };
    let plain = |g: &dyn Fn(f64) -> f64, lo: f64, hi: f64, x: Option<&dyn Fn(f64) -> f64>| {
        quadrature_finite_report(g, lo, hi, &share, raw, Floor::Relative(x))
    };
    let (first, second) = match (a.is_finite(), b.is_finite()) {
        (true, true) if a == 0.0 => (smooth(f, 0.5 * b), plain(f, 0.5 * b, b, None)),
        (true, true) if b == 0.0 => {
            let mirrored = move |u: f64| f(-u);
            (plain(f, a, 0.5 * a, None), smooth(&mirrored, -0.5 * a))
        }
        (true, false) | (false, true) => {
            let (e, sign) = if a.is_finite() { (a, 1.0) } else { (b, -1.0) };
            // [e, e ± 1]: x = e ± t/(1 − t), t ∈ [0, ½].
            let near = move |t: f64| {
                let om = 1.0 - t;
                f(e + sign * (t / om)) / (om * om)
            };
            // Beyond e ± 1: x = e ± (1/u − 1), u ∈ (0, ½], dx = du/u².
            let far = move |u: f64| f(e + sign * (1.0 / u - 1.0)) / u / u;
            let x = move |t: f64| e + sign * (t / (1.0 - t));
            let near_run = if e == 0.0 {
                smooth(&near, 0.5)
            } else {
                plain(&near, 0.0, 0.5, Some(&x))
            };
            (near_run, smooth(&far, 0.5))
        }
        _ => return None,
    };
    Some(first.and_then(|(p, p_ok)| {
        second.map(|(q, q_ok)| {
            (
                QuadResult {
                    value: p.value + q.value,
                    error: p.error + q.error,
                },
                p_ok && q_ok,
            )
        })
    }))
}

/// [`quadrature`] for a caller that certifies digits with the result
/// (`evalf`): `None` unless every segment was resolved — none left at the
/// narrowest width the bisection can make, where the estimate of a segment
/// next to a singularity is no bound — and with the raw error estimate
/// `|K15 − G7|` of each segment ([`gk15_with`]).  Plain bisection (its
/// segments too narrow below `10⁻¹²` of the range, [`Floor::Range`]), or
/// with `smoothing` the pieces of [`smoothed_pieces`], the endpoint-smoothing
/// substitution at an end at 0 and an infinite end only (plain bisection
/// when neither end is such).  The caller judges the error estimate: next
/// to a nonzero end `b` no node comes closer to `b` than half an ulp, so the
/// computed integrand is wrong where a singularity at `b` matters and no
/// estimate sees it.  Before 0.44 the smoothed run on `[0, ∞)` crowded its
/// nodes towards the end `t = 1` of `x = t/(1 − t)`, which cut the integrand
/// off at `x ≈ 10¹⁶`.
///
/// Before 0.43 `evalf` took [`quadrature`]'s estimate as a bound, and that
/// is how it went wrong at an endpoint singularity:
/// `∫₀¹ x⁴e^{x/10}/(√(1 − x)·B(5, ½)) dx` was `1.0952385480452327` with an
/// estimate of `4·10⁻¹⁰` (truly `1.0952385618059876`, mpmath; the error,
/// `1.4·10⁻⁸`, is that of the integrand cut off at `1 − x = 2⁻⁵³`), and
/// `∫₀¹ dx/√(1 − x)` `1.9999999898808887`.
///
/// # Errors
///
/// As [`quadrature`].
pub(crate) fn quadrature_resolved(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    opts: &QuadOpts,
    smoothing: bool,
) -> Result<Option<QuadResult>, SymplexError> {
    if a.is_nan() || b.is_nan() {
        return Err(SymplexError::InvalidArgument {
            operation: "integrate_numeric",
            reason: "integration bounds must not be NaN".into(),
        });
    }
    if f64_narrow(a, b) {
        return Ok(None);
    }
    if a == b {
        return Ok(Some(QuadResult {
            value: 0.0,
            error: 0.0,
        }));
    }
    if a > b {
        return Ok(
            quadrature_resolved(f, b, a, opts, smoothing)?.map(|r| QuadResult {
                value: -r.value,
                error: r.error,
            }),
        );
    }
    if a.is_infinite() && b.is_infinite() {
        let half = QuadOpts {
            max_subdivisions: opts.max_subdivisions.div_ceil(2),
            ..*opts
        };
        let left = quadrature_resolved(f, f64::NEG_INFINITY, 0.0, &half, smoothing)?;
        let right = quadrature_resolved(f, 0.0, f64::INFINITY, &half, smoothing)?;
        return Ok(left.zip(right).map(|(l, r)| QuadResult {
            value: l.value + r.value,
            error: l.error + r.error,
        }));
    }
    let smoothed_run = if smoothing {
        smoothed_pieces(f, a, b, opts, true)
    } else {
        None
    };
    let run = match smoothed_run {
        Some(run) => run,
        None => plain_report(f, a, b, opts, true, Floor::Range),
    };
    run.map(|(r, ok)| ok.then_some(r))
}

/// The interior breakpoints of `f` (in `x`) strictly inside `(lo, hi)` for
/// numeric quadrature, sorted: the points the definite integrator's
/// breakpoint scanner locates (poles, branch points of fractional powers
/// and logarithms, `tan` poles, kinks of `|·|`, `sign`, `Heaviside`, steps
/// and `Piecewise` boundaries), and the centres of narrow peaks — the real
/// parts of complex poles next to the range, … ([`peak_points`]) — as a
/// real constant expression and its `f64` value.  Points without a numeric
/// value are skipped; an incomplete scan gives the points it found (the
/// quadrature checks its own convergence).
///
/// `range` holds the bounds' values (`lower ≤ upper`, `±∞` allowed) and
/// `exact`, when known, the finite bounds themselves: the periodic families
/// are enumerated between them and a point is inside when that is
/// certified ([`cmp_values`]); without them the `f64` bounds stand in, as
/// the binary numbers they are.  Up to 0.49 the families were enumerated
/// from the `f64` bounds and a point within `10⁻¹²` of one was dropped: no
/// zero of `sin x` was found on `[10²⁰, 10²⁰ + 4]`.
///
/// # Errors
///
/// [`SymplexError::ComputationFailed`] when a family of poles, kinks or
/// steps has more than 10,000 members in the range (`Γ(9·10¹⁸x)` on `[−1,
/// 1]`): no quadrature resolves them, and up to 0.49 the arbitrary-precision
/// one spent minutes trying.
pub(crate) fn quadrature_breakpoints(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    range: Interval<f64>,
    exact: Option<Interval<ExprId>>,
    locate: Option<u32>,
) -> Result<Vec<(ExprId, f64)>, SymplexError> {
    let (lo, hi) = (range.lower, range.upper);
    let ends = match exact {
        Some(e) => Some(e),
        None => ScanRange::from_f64(arena, range).map(|r| r.exact),
    }
    .filter(|_| lo.is_finite() && hi.is_finite());
    let scan_range = ends.and_then(|e| ScanRange::new(e.lower, e.upper, range));
    let scan = calculus_util::scan_breakpoints(arena, f, x, scan_range);
    if scan.crowded {
        return Err(failed(
            "the integrand has more than 10,000 poles, kinks or steps inside the interval",
        ));
    }
    // Kinks and steps the scan could not place (`sign(cos x²)`, `floor(x³ −
    // x)`): located numerically, or the quadrature is refused.  Up to 0.49
    // the quadrature ran across them, and its estimate, which assumes a
    // smooth integrand, missed the jumps: `∫₀³ sign(cos x² − 1/7) dx` was
    // `0.5178` from `eval_f64` (truly `0.5360`).
    // (On an infinite range the kinks are not sought: there may be
    // infinitely many, `|sin x|·e⁻ˣ` on `[0, ∞)`.)
    let mut located: Vec<(ExprId, f64)> = Vec::new();
    if let Some(bits) = locate
        && let Some(e) = ends.filter(|_| lo < hi)
        && (!scan.complete || scan.periodic)
    {
        let unplaced = unlocated_kink_arguments(arena, f, x, scan_range);
        if !unplaced.is_empty() {
            let mut budget = MAX_LOCATE_EVALS;
            for (g, crossing) in unplaced {
                located.extend(locate_crossings(
                    arena,
                    g,
                    crossing,
                    x,
                    e,
                    range,
                    bits,
                    &mut budget,
                )?);
                if located.len() > MAX_PIECES {
                    return Err(failed("too many kinks or steps inside the interval"));
                }
            }
        }
    }
    let mut points: Vec<(ExprId, Option<f64>)> = Vec::new();
    let candidates = scan
        .points
        .into_iter()
        .filter_map(|bp| bp.value.map(|v| (bp.point, v)))
        .chain(located)
        .chain(peak_points(arena, f, x, lo, hi));
    for (p, v) in candidates {
        if !v.is_finite() || !walk::free_symbols(arena, p).is_empty() {
            continue;
        }
        // Strictly inside, certified.
        let inside = match ends {
            Some(e) => {
                let (ev_lo, ev_hi) = (Some(lo), Some(hi));
                cmp_values(arena, (e.lower, ev_lo), (p, Some(v))) == Some(Ordering::Less)
                    && cmp_values(arena, (p, Some(v)), (e.upper, ev_hi)) == Some(Ordering::Less)
            }
            None => lo < v && v < hi,
        };
        if inside && !points.iter().any(|(q, _)| *q == p) {
            points.push((p, Some(v)));
        }
    }
    let points = sort_points(arena, points)
        .ok_or_else(|| failed("cannot order the breakpoints of the integrand"))?;
    Ok(points
        .into_iter()
        .filter_map(|(p, v)| v.map(|v| (p, v)))
        .collect())
}

/// What crossing of an argument `g` makes a kink or a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Crossing {
    /// `g` changes sign: `|g|`, `sign g`, `H(g)`, `min`/`max` (`g` a
    /// difference of two arguments), a condition `p > q` (`g = p − q`).
    Zero,
    /// `g` passes an integer: `floor g`, `ceiling g`.
    Integer,
}

/// The arguments, with their crossings, of the kink and step nodes of `f`
/// (depending on `x`) whose kinks a scan over `range` does not locate.
fn unlocated_kink_arguments(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    range: Option<ScanRange>,
) -> Vec<(ExprId, Crossing)> {
    let mut out: Vec<(ExprId, Crossing)> = Vec::new();
    for id in walk::post_order_ids(arena, f) {
        let args: Vec<(ExprId, Crossing)> = match arena.node(id).clone() {
            ExprNode::Abs(g) | ExprNode::Sign(g) | ExprNode::Heaviside(g) => {
                vec![(g, Crossing::Zero)]
            }
            ExprNode::Floor(g) | ExprNode::Ceiling(g) => vec![(g, Crossing::Integer)],
            ExprNode::Gt(p, q) | ExprNode::Ge(p, q) | ExprNode::Eq_(p, q) | ExprNode::Ne(p, q) => {
                vec![(arena.sub(p, q), Crossing::Zero)]
            }
            ExprNode::Min(args) | ExprNode::Max(args) => {
                let args = args.to_vec();
                let mut d = Vec::new();
                for (i, &p) in args.iter().enumerate() {
                    for &q in &args[i + 1..] {
                        d.push((arena.sub(p, q), Crossing::Zero));
                    }
                }
                d
            }
            _ => continue,
        };
        if !args.iter().any(|&(g, _)| walk::contains(arena, g, x)) {
            continue;
        }
        let s = calculus_util::scan_breakpoints(arena, id, x, range);
        if s.complete && !s.opaque && !s.periodic {
            continue;
        }
        for (g, c) in args {
            let g = safe_eval(arena, g);
            if walk::contains(arena, g, x) && !out.contains(&(g, c)) {
                out.push((g, c));
            }
        }
    }
    out
}

/// Most evaluations of kink arguments one [`quadrature_breakpoints`] call
/// spends locating crossings.
const MAX_LOCATE_EVALS: usize = 20_000;
/// Samples of a kink argument over the range.
const LOCATE_SAMPLES: usize = 256;
/// Samples of the refinement around a minimum of the distance to a level.
const LOCATE_REFINE: usize = 32;

/// The value of `h` at `x = t` for locating its crossings: the 16-digit
/// value, whose sign is certified when it is not 0; for a value that is 0
/// to 16 digits the certified sign test decides (`0.0` for an exact zero,
/// `±f64::MIN_POSITIVE` for a tiny one); `None` when `h` has no real value
/// there or its sign does not settle.
fn crossing_value(arena: &mut Arena, h: ExprId, x: ExprId, t: ExprId) -> Option<f64> {
    let v = subs::subs(arena, h, x, t);
    let v = safe_eval(arena, v);
    if !walk::free_symbols(arena, v).is_empty() {
        return None;
    }
    match calculus_util::expr_to_f64(arena, v) {
        Some(y) if y.is_finite() && y != 0.0 => Some(y),
        Some(0.0) => match crate::poly::algebraic::sign_checked(arena, v)? {
            0 => Some(0.0),
            s => Some(f64::from(s) * f64::MIN_POSITIVE),
        },
        _ => None,
    }
}

/// The crossings of `g` ([`Crossing`]) strictly inside the finite range
/// between the exact ends `ends` (values `range`, `lower < upper`), located
/// numerically: `g` sampled at [`LOCATE_SAMPLES`] points (and
/// [`LOCATE_REFINE`] more around each minimum of its distance to a level
/// that a parabola through three samples puts beyond the level) — by its
/// compiled `f64` form when it has one, a guide only —, each level passed
/// between two samples bracketed by certified signs ([`crossing_value`])
/// and narrowed (secant and bisection steps) to `2⁻ᵇⁱᵗˢ` of the range.  A
/// point returned is a crossing that the signs prove, to that accuracy; a
/// pair of crossings between two samples that no minimum reveals is
/// missed, the limit of every sampling (and of the quadrature's own
/// estimate).
///
/// # Errors
///
/// [`SymplexError::ComputationFailed`] when `g` has no real value at a
/// sample (or at an end), a sign does not settle, the samples cannot be
/// placed in `f64`, or the certified evaluations exceed `budget`.
#[allow(clippy::too_many_arguments)]
fn locate_crossings(
    arena: &mut Arena,
    g: ExprId,
    crossing: Crossing,
    x: ExprId,
    ends: Interval<ExprId>,
    range: Interval<f64>,
    bits: u32,
    budget: &mut usize,
) -> Result<Vec<(ExprId, f64)>, SymplexError> {
    let unlocatable = || failed("cannot locate the kinks or steps of the integrand");
    let (lo_f, hi_f) = (range.lower, range.upper);
    let width = hi_f - lo_f;
    // The samples: the exact ends and binary numbers between them.
    let mut ts: Vec<(ExprId, f64)> = vec![(ends.lower, lo_f)];
    for j in 1..LOCATE_SAMPLES {
        let t = lo_f + width * (j as f64) / (LOCATE_SAMPLES as f64);
        let Some(q) = crate::base::numeric::f64_to_ratio_exact(t) else {
            return Err(unlocatable());
        };
        if t <= ts.last().map_or(lo_f, |p| p.1) || t >= hi_f {
            return Err(unlocatable());
        }
        ts.push((arena.num_ratio(q), t));
    }
    ts.push((ends.upper, hi_f));
    let compiled = match arena.node(x) {
        ExprNode::Symbol(sid) => {
            let name = arena.symbol_name(*sid).to_string();
            crate::output::lambdify::compile(arena, g, &[&name]).ok()
        }
        _ => None,
    };
    // A guide to the value of `g` at the sample `(id, t)`: compiled, or
    // certified when that is not finite (or not available).
    let guide = |arena: &mut Arena, id: ExprId, t: f64, budget: &mut usize| {
        if let Some(func) = &compiled {
            let v = func(&[t]);
            if v.is_finite() {
                return Ok(v);
            }
        }
        spend(budget)?;
        crossing_value(arena, g, x, id).ok_or_else(unlocatable)
    };
    let mut vs: Vec<f64> = Vec::with_capacity(ts.len());
    for &(id, t) in &ts {
        vs.push(guide(arena, id, t, budget)?);
    }
    // The level nearest to a value.
    let nearest = |v: f64| match crossing {
        Crossing::Zero => 0.0,
        Crossing::Integer => v.round(),
    };
    // Refine around minima of the distance to the nearest level that a
    // parabola puts across it.
    let mut fine: Vec<(ExprId, f64, f64)> = Vec::new();
    for j in 1..ts.len() - 1 {
        let level = nearest(vs[j]);
        let d = |v: f64| (v - level).abs();
        if !(d(vs[j]) < d(vs[j - 1]) && d(vs[j]) < d(vs[j + 1])) {
            continue;
        }
        let side = |v: f64| v > level;
        if side(vs[j - 1]) != side(vs[j]) || side(vs[j]) != side(vs[j + 1]) {
            continue;
        }
        let (t0, t1, t2) = (ts[j - 1].1, ts[j].1, ts[j + 1].1);
        let (v0, v1, v2) = (vs[j - 1], vs[j], vs[j + 1]);
        let a = ((v2 - v1) / (t2 - t1) - (v1 - v0) / (t1 - t0)) / (t2 - t0);
        let b = (v1 - v0) / (t1 - t0) - a * (t0 + t1);
        if a == 0.0 || !a.is_finite() || !b.is_finite() {
            continue;
        }
        let tv = -b / (2.0 * a);
        let pv = v1 + (v1 - v0) / (t1 - t0) * (tv - t1) + a * (tv - t0) * (tv - t1);
        if !(t0 < tv && tv < t2) || side(pv) == side(v1) {
            continue;
        }
        for k in 1..LOCATE_REFINE {
            let t = t0 + (t2 - t0) * (k as f64) / (LOCATE_REFINE as f64);
            if t == t1 || t <= t0 || t >= t2 {
                continue;
            }
            let Some(q) = crate::base::numeric::f64_to_ratio_exact(t) else {
                continue;
            };
            let id = arena.num_ratio(q);
            let v = guide(arena, id, t, budget)?;
            fine.push((id, t, v));
        }
    }
    let mut all: Vec<(ExprId, f64, f64)> = ts
        .iter()
        .zip(&vs)
        .map(|(&(id, t), &v)| (id, t, v))
        .chain(fine)
        .collect();
    all.sort_by(|p, q| p.1.partial_cmp(&q.1).unwrap_or(Ordering::Equal));
    // Brackets: a level the guide puts between two neighbouring samples, or
    // close to one of them, confirmed by certified signs; or a sample
    // exactly on a level.
    let target = width * f64::powi(2.0, -i32::try_from(bits.min(1000)).unwrap_or(1000));
    let max_steps = 2 * bits.min(1000) + 128;
    let close = |v: f64, level: f64| (v - level).abs() <= 1e-6 * v.abs().max(level.abs()).max(1.0);
    let mut out: Vec<(ExprId, f64)> = Vec::new();
    for (k, w) in all.windows(2).enumerate() {
        let ((l, lt, lv), (r, rt, rv)) = (w[0], w[1]);
        let levels: Vec<f64> = match crossing {
            Crossing::Zero => vec![0.0],
            Crossing::Integer => {
                let (first, last) = (lv.min(rv).floor(), lv.max(rv).ceil());
                if !(first.is_finite() && last.is_finite()) || last - first > 64.0 {
                    return Err(unlocatable());
                }
                let mut ls = Vec::new();
                let mut n = first;
                while n <= last {
                    ls.push(n);
                    n += 1.0;
                }
                ls
            }
        };
        for level in levels {
            let between = (lv - level) * (rv - level) < 0.0;
            if !between && !close(lv, level) && !close(rv, level) {
                continue;
            }
            let h = if level == 0.0 {
                g
            } else {
                let Some(q) = crate::base::numeric::f64_to_ratio_exact(level) else {
                    return Err(unlocatable());
                };
                let n = arena.num_ratio(q);
                let h = arena.sub(g, n);
                safe_eval(arena, h)
            };
            spend(budget)?;
            spend(budget)?;
            let sl = crossing_value(arena, h, x, l).ok_or_else(unlocatable)?;
            let sr = crossing_value(arena, h, x, r).ok_or_else(unlocatable)?;
            if sl == 0.0 {
                // On the level: a crossing unless it is the lower end.
                if k > 0 && !out.iter().any(|&(p, _)| p == l) {
                    out.push((l, lt));
                }
                continue;
            }
            if sr == 0.0 || (sl < 0.0) == (sr < 0.0) {
                continue;
            }
            if let Some(p) = narrow_crossing(
                arena,
                h,
                x,
                (l, lt, sl),
                (r, rt, sr),
                target,
                max_steps,
                budget,
            )? {
                out.push(p);
            }
        }
    }
    Ok(out)
}

/// One evaluation of the budget of [`locate_crossings`].
fn spend(budget: &mut usize) -> Result<(), SymplexError> {
    if *budget == 0 {
        return Err(failed(
            "locating the kinks of the integrand took too many evaluations",
        ));
    }
    *budget -= 1;
    Ok(())
}

/// A crossing of `h` between `l` and `r` (each `(point, f64 value of the
/// point, f64 value of h)`, values of `h` of opposite certified signs),
/// narrowed until the bracket is narrower than `target` (at most
/// `max_steps` evaluations): secant steps (kept a sixteenth of the bracket
/// from its ends) alternating with bisections.  The midpoint of the last
/// bracket, with its value; `None` when the signs do not differ.
#[allow(clippy::too_many_arguments)]
fn narrow_crossing(
    arena: &mut Arena,
    h: ExprId,
    x: ExprId,
    (mut l, mut lt, mut lv): (ExprId, f64, f64),
    (mut r, mut rt, mut rv): (ExprId, f64, f64),
    target: f64,
    max_steps: u32,
    budget: &mut usize,
) -> Result<Option<(ExprId, f64)>, SymplexError> {
    if (lv < 0.0) == (rv < 0.0) || lv == 0.0 || rv == 0.0 {
        return Ok(None);
    }
    for step in 0..max_steps {
        let w = arena.sub(r, l);
        let w = safe_eval(arena, w);
        let wv = calculus_util::expr_to_f64(arena, w).unwrap_or(f64::INFINITY);
        if wv.is_finite() && wv.abs() <= target {
            break;
        }
        let frac = if step % 2 == 0 {
            (lv / (lv - rv)).clamp(1.0 / 16.0, 15.0 / 16.0)
        } else {
            0.5
        };
        let Some(q) = crate::base::numeric::f64_to_ratio_exact(frac) else {
            return Err(failed("cannot locate the kinks or steps of the integrand"));
        };
        let fq = arena.num_ratio(q);
        let shift = arena.mul(&[fq, w]);
        let m = arena.add(&[l, shift]);
        let m = safe_eval(arena, m);
        spend(budget)?;
        let Some(mv) = crossing_value(arena, h, x, m) else {
            return Err(failed("cannot locate the kinks or steps of the integrand"));
        };
        let mt = lt + (rt - lt) * frac;
        if mv == 0.0 {
            let value = calculus_util::expr_to_f64(arena, m).unwrap_or(mt);
            return Ok(Some((m, value)));
        }
        if (mv < 0.0) == (lv < 0.0) {
            (l, lt, lv) = (m, mt, mv);
        } else {
            (r, rt, rv) = (m, mt, mv);
        }
    }
    let half = arena.rational(1, 2);
    let s = arena.add(&[l, r]);
    let m = arena.mul(&[half, s]);
    let m = safe_eval(arena, m);
    let value = calculus_util::expr_to_f64(arena, m).unwrap_or(0.5 * (lt + rt));
    Ok(Some((m, value)))
}

/// The centres of the narrow peaks of `f` in `(lo, hi)`, features a rule
/// over the whole range may not see: the real parts of the complex poles
/// next to the range — zeros off the real line of a polynomial raised to a
/// negative power (degree 2 to [`MAX_POLE_SPLIT_DEGREE`]) —, the vertices
/// of Gaussians `exp(a₂x² + a₁x + a₀)`, `a₂ < 0`, and the zeros of the
/// linear arguments of `sech` (`cosh` under a negative power), whose half-width
/// `w` (imaginary part, `1/√(2|a₂|)`) is below a tenth of the range (of
/// `max(1, |centre|)` on an infinite one), each with `centre ± 10w` inside
/// the range, as exact binary numbers.  A split there makes a narrow peak (`1/(1 + 10⁶(x − c)²)`, `exp(−10⁶(x − c)²)`,
/// of width `10⁻³`) the end of two pieces, where the nodes crowd, instead
/// of a feature between the nodes: before 0.44 `integrate_numeric` of
/// `exp(−10⁶(x − 2/3)²)·(x + 1)²cos(4x/3)` on `[0, 2]` was `0` with an
/// error estimate of 0 (truly `0.0031`), and the arbitrary-precision
/// quadrature chased a Lorentzian of width `10⁻³` by bisection until its
/// budget ran out at 54 digits.
fn peak_points(arena: &mut Arena, f: ExprId, x: ExprId, lo: f64, hi: f64) -> Vec<(ExprId, f64)> {
    /// The polynomial a peak is read from.
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum Source {
        /// A denominator: its complex zeros.
        Pole,
        /// A Gaussian's exponent `a₂x² + a₁x + a₀`.
        Gaussian,
        /// The argument `l₁x + l₀` of `cosh` under a negative power
        /// (`sech(l₁x + l₀)` peaks at its zero, half-width `1/|l₁|`).
        Sech,
    }
    let mut parts: Vec<(ExprId, Source)> = walk::post_order_ids(arena, f)
        .into_iter()
        .filter_map(|id| match *arena.node(id) {
            ExprNode::Pow(base, exp)
                if arena.as_num(exp).is_some_and(Signed::is_negative)
                    && walk::contains(arena, base, x) =>
            {
                match *arena.node(base) {
                    ExprNode::Cosh(arg) => Some((arg, Source::Sech)),
                    _ => Some((base, Source::Pole)),
                }
            }
            ExprNode::Exp(arg) if walk::contains(arena, arg, x) => Some((arg, Source::Gaussian)),
            _ => None,
        })
        .collect();
    parts.sort_unstable_by(|a, b| arena.canonical_cmp(a.0, b.0).then(a.1.cmp(&b.1)));
    parts.dedup();
    let narrow = |centre: f64, half_width: f64| {
        let scale = if lo.is_finite() && hi.is_finite() {
            hi - lo
        } else {
            centre.abs().max(1.0)
        };
        half_width != 0.0 && centre > lo && centre < hi && half_width.abs() < 0.1 * scale
    };
    // A peak at `c` of half-width `w` splits at `c` and at `c ± 10w`: the
    // nodes of a piece ending at `c` stay at a distance from it (`0.0043`
    // of its width for Gauss–Kronrod), where a Gaussian's values are 0 and
    // nothing calls for a bisection; the piece `[c − 10w, c]` sees it.
    let mut centres: Vec<f64> = Vec::new();
    let mut push = |c: f64, w: f64| {
        centres.extend(
            [c - 10.0 * w.abs(), c, c + 10.0 * w.abs()]
                .into_iter()
                .filter(|&p| p > lo && p < hi),
        );
    };
    for (part, source) in parts {
        // Expanded only when small: `(x + 1)^5000 + 2` is not.
        let max_degree = match source {
            Source::Pole => MAX_POLE_SPLIT_DEGREE,
            Source::Gaussian => 2,
            Source::Sech => 1,
        };
        if !degree_bound(arena, part, x).is_some_and(|d| d <= max_degree) {
            continue;
        }
        let Some(p) = crate::poly::polybridge::expr_to_poly(arena, part, x) else {
            continue;
        };
        let degree = p.degree().unwrap_or(0);
        let coeff = |k: usize| {
            p.coeffs()
                .get(k)
                .and_then(crate::base::numeric::ratio_to_f64)
        };
        match source {
            Source::Gaussian => {
                let (Some(a1), Some(a2)) = (coeff(1), coeff(2)) else {
                    continue;
                };
                let (c, w) = (-a1 / (2.0 * a2), (-2.0 * a2).sqrt().recip());
                if degree == 2 && a2 < 0.0 && narrow(c, w) {
                    push(c, w);
                }
            }
            Source::Sech => {
                let (Some(l0), Some(l1)) = (coeff(0), coeff(1)) else {
                    continue;
                };
                let (c, w) = (-l0 / l1, l1.recip());
                if degree == 1 && narrow(c, w) {
                    push(c, w);
                }
            }
            Source::Pole if (2..=MAX_POLE_SPLIT_DEGREE).contains(&degree) => {
                for z in crate::poly::roots::nroots_f64(&p, 128) {
                    if narrow(z.re, z.im) {
                        push(z.re, z.im);
                    }
                }
            }
            Source::Pole => {}
        }
    }
    centres
        .into_iter()
        .filter(|c| c.is_finite())
        .filter_map(|c| {
            crate::base::numeric::f64_to_ratio_exact(c).map(|q| (arena.num_ratio(q), c))
        })
        .collect()
}

/// The points at which `Ex::integrate_numeric_with` splits its `f64`
/// quadrature of `f` over `(lo, hi)`, sorted: the breakpoints of
/// [`quadrature_breakpoints`] (with the [`peak_points`]).  At a kink or a
/// step at an irrational point (`|sin x|` at `π`) the error estimate of the
/// segment holding it understates its error (QUADPACK's reduction assumes a
/// smooth integrand); a segment across an interior pole may cancel it
/// exactly: up to 0.49, when only the integrands with kinks were scanned,
/// `∫₋₂² tan x dx` and `∫₋₁¹ x/(x² − 1/4) dx`, which diverge, were
/// `Ok(0.0)` (the nodes symmetric about 0, the integrand odd).  A piece
/// that ends at a singularity is left to the caller's arbitrary-precision
/// fallback.
///
/// `exact` holds the finite bounds when known (see
/// [`quadrature_breakpoints`], whose errors this returns).
pub(crate) fn quadrature_split_points(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    lo: f64,
    hi: f64,
    exact: Option<Interval<ExprId>>,
) -> Result<Vec<f64>, SymplexError> {
    let mut points: Vec<f64> =
        quadrature_breakpoints(arena, f, x, Interval::closed(lo, hi), exact, Some(64))?
            .into_iter()
            .map(|(_, v)| v)
            .collect();
    points.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    points.dedup_by(|a, b| (*a - *b).abs() <= 1e-14 * a.abs().max(1.0));
    Ok(points)
}

/// [`quadrature_report`] over `[a, b]` split at the sorted interior
/// `points` (`a < b`): the sum of the pieces, each with an equal share of
/// the absolute tolerance and of the subdivisions, and whether every piece
/// was resolved.
///
/// # Errors
///
/// As [`quadrature`].
pub(crate) fn quadrature_split(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    points: &[f64],
    opts: &QuadOpts,
) -> Result<(QuadResult, bool), SymplexError> {
    let inner: Vec<f64> = points.iter().copied().filter(|&p| a < p && p < b).collect();
    if inner.is_empty() {
        return quadrature_report(f, a, b, opts);
    }
    let n = inner.len() + 1;
    let share = QuadOpts {
        abs_tol: opts.abs_tol / n as f64,
        max_subdivisions: opts.max_subdivisions.div_ceil(n).max(1),
        ..*opts
    };
    let ends: Vec<f64> = std::iter::once(a)
        .chain(inner)
        .chain(std::iter::once(b))
        .collect();
    let mut total = QuadResult {
        value: 0.0,
        error: 0.0,
    };
    let mut resolved = true;
    for w in ends.windows(2) {
        let (r, ok) = quadrature_report(f, w[0], w[1], &share)?;
        total.value += r.value;
        total.error += r.error;
        resolved &= ok;
    }
    Ok((total, resolved))
}

/// Does `f` have a kink or a step depending on `x` (`|·|`, `sign`,
/// `Heaviside`, `floor`, `ceiling`, `min`, `max`, `Piecewise`)?
pub(crate) fn has_kink(arena: &Arena, f: ExprId, x: ExprId) -> bool {
    walk::post_order_ids(arena, f).into_iter().any(|id| {
        matches!(
            arena.node(id),
            ExprNode::Abs(_)
                | ExprNode::Sign(_)
                | ExprNode::Heaviside(_)
                | ExprNode::Floor(_)
                | ExprNode::Ceiling(_)
                | ExprNode::Min(_)
                | ExprNode::Max(_)
                | ExprNode::Piecewise(_)
        ) && walk::contains(arena, id, x)
    })
}

/// The pieces of the `f64` quadrature of `evalf` for a body `f` (in `x`,
/// both in `arena`) with a kink or a step on the finite range `[a, b]`
/// (`a < b`, the binary numbers themselves): the interior breakpoints of
/// [`quadrature_breakpoints`] (located to 64 bits where the scan cannot
/// place them) as `f64`s, and a bound on the error of integrating between
/// those instead of the breakpoints, per unit of the integrand's largest
/// value (the jump of a step is at most twice that).  `None` when not every
/// breakpoint is located, or the range is infinite or too narrow
/// ([`f64_narrow`]).
///
/// `evalf` reads the quadrature's error estimate as a bound, which across a
/// kink or a step it is not (QUADPACK's estimate assumes a smooth
/// integrand): up to 0.49 `eval_f64` of `∫_{100.7}^{130.7} frac(x) dx` was
/// `21` (truly 15), and of `∫_{10.7}^{410.7} |sin(x + 1/2)| dx` `254.736821`
/// (truly `254.736834`, mpmath).
pub(crate) fn f64_pieces(arena: &Arena, f: ExprId, x: ExprId, a: f64, b: f64) -> Option<F64Pieces> {
    if a.partial_cmp(&b) != Some(Ordering::Less)
        || !a.is_finite()
        || !b.is_finite()
        || f64_narrow(a, b)
    {
        return None;
    }
    let mut pa = Arena::with_config(arena.config.clone());
    let mut map: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    let f_p = crate::base::compact::transfer_subtree(arena, &mut pa, f, &mut map);
    let x_p = crate::base::compact::transfer_subtree(arena, &mut pa, x, &mut map);
    let points =
        quadrature_breakpoints(&mut pa, f_p, x_p, Interval::closed(a, b), None, Some(64)).ok()?;
    let mut cuts = Vec::with_capacity(points.len());
    let mut misplaced = 0.0;
    for (p, v) in points {
        // The nearest `f64` to the point, from its 16-digit value: within
        // two ulps (a located crossing is within `2⁻⁶⁴` of the range of
        // its own value).
        let v = calculus_util::expr_to_f64(&mut pa, p).unwrap_or(v);
        if !(a < v && v < b) {
            return None;
        }
        misplaced += 2.0 * (2.0 * f64::EPSILON * v.abs() + (b - a) * 2f64.powi(-60));
        cuts.push(v);
    }
    cuts.sort_by(|p, q| p.partial_cmp(q).unwrap_or(Ordering::Equal));
    cuts.dedup();
    Some(F64Pieces { cuts, misplaced })
}

/// The split of [`f64_pieces`].
#[derive(Clone, Debug)]
pub(crate) struct F64Pieces {
    /// The interior split points, increasing.
    pub cuts: Vec<f64>,
    /// The error of splitting at `cuts` rather than at the breakpoints, per
    /// unit of the integrand's largest value.
    pub misplaced: f64,
}

/// [`quadrature_resolved`] on `[a, b]` (`a < b`, finite) split at the
/// increasing interior `cuts`: the sum of the pieces, each with an equal
/// share of the absolute tolerance and of the subdivisions; `None` unless
/// every piece is resolved.
///
/// # Errors
///
/// As [`quadrature`].
pub(crate) fn quadrature_resolved_split(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    cuts: &[f64],
    opts: &QuadOpts,
) -> Result<Option<QuadResult>, SymplexError> {
    let n = cuts.len() + 1;
    let share = QuadOpts {
        abs_tol: opts.abs_tol / n as f64,
        max_subdivisions: opts.max_subdivisions.div_ceil(n).max(16),
        ..*opts
    };
    let ends: Vec<f64> = std::iter::once(a)
        .chain(cuts.iter().copied())
        .chain(std::iter::once(b))
        .collect();
    let mut total = QuadResult {
        value: 0.0,
        error: 0.0,
    };
    for w in ends.windows(2) {
        let Some(r) = quadrature_resolved(f, w[0], w[1], &share, false)? else {
            return Ok(None);
        };
        total.value += r.value;
        total.error += r.error;
    }
    Ok(Some(total))
}

/// Is the finite range between `a` and `b` narrower than `2⁻²⁰` of its
/// position, so that `f64` arithmetic cannot resolve it: its nodes are a
/// few thousand ulps apart at most, and an integrand that varies on the
/// scale of the range (`frac(10x)` near `10¹⁵`) is evaluated there with
/// errors of `2²⁰` ulps of its argument.  The `f64` quadratures leave such a
/// range to the arbitrary-precision one; up to 0.49 `∫_{10¹⁵+1/3}^{10¹⁵+1/3+40}
/// frac(10x) dx` was `0` from `integrate_numeric` (truly 20).
pub(crate) fn f64_narrow(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (b - a).abs() < NARROW * a.abs().max(b.abs())
}

/// The relative width below which [`f64_narrow`] holds: `2⁻²⁰`.
const NARROW: f64 = 9.5367431640625e-7;

/// The highest degree of a denominator whose complex zeros
/// [`peak_points`] computes.
const MAX_POLE_SPLIT_DEGREE: usize = 12;

/// An upper bound on the degree of `e` as a polynomial in `x`, from its
/// structure (sums, products, natural powers), without expanding it;
/// `None` when `e` is not visibly a polynomial in `x`.
fn degree_bound(arena: &Arena, e: ExprId, x: ExprId) -> Option<usize> {
    let mut deg: FxHashMap<ExprId, usize> = FxHashMap::default();
    for id in walk::post_order_ids(arena, e) {
        let d = if id == x {
            1
        } else if !walk::contains(arena, id, x) {
            0
        } else {
            match arena.node(id) {
                ExprNode::Add(children) => children
                    .iter()
                    .map(|c| deg.get(c).copied())
                    .try_fold(0usize, |m, d| d.map(|d| m.max(d)))?,
                ExprNode::Mul(children) => children
                    .iter()
                    .map(|c| deg.get(c).copied())
                    .try_fold(0usize, |s, d| d.map(|d| s.saturating_add(d)))?,
                ExprNode::Pow(base, exp) => {
                    let n = arena.as_num(*exp)?;
                    if !n.is_integer() || n.is_negative() {
                        return None;
                    }
                    let n = n.to_integer().to_usize()?;
                    deg.get(base)?.saturating_mul(n)
                }
                _ => return None,
            }
        };
        deg.insert(id, d);
    }
    deg.get(&e).copied()
}

/// `f` on `[a, b]` under the endpoint-smoothing substitution `x = a +
/// (b−a)·s(t)`, `s(t) = t³(10 − 15t + 6t²)`, `t ∈ [0, 1]`: the integrand of
/// the same integral over `[0, 1]`, whose derivative vanishes quadratically
/// at both ends.  The nodes crowd towards both ends, so it is used only on
/// a piece whose singular end is at 0 ([`smoothed_pieces`]).
fn smoothed<'a>(f: &'a dyn Fn(f64) -> f64, a: f64, b: f64) -> impl Fn(f64) -> f64 + 'a {
    let width = b - a;
    move |t: f64| {
        let ds = 30.0 * t * t * (1.0 - t) * (1.0 - t);
        if ds == 0.0 {
            return 0.0;
        }
        // Never exactly on an endpoint.
        let x = smoothing_map(a, b, t);
        let x = if t <= 0.5 {
            if x <= a { a.next_up() } else { x }
        } else if x >= b {
            b.next_down()
        } else {
            x
        };
        f(x) * width * ds
    }
}

/// The map of [`smoothed`], `a + (b−a)·s(t)`, evaluated from the nearer end
/// (`s(1−t) = 1 − s(t)`) so that `x` keeps full relative precision near
/// both ends.
fn smoothing_map(a: f64, b: f64, t: f64) -> f64 {
    let width = b - a;
    if t <= 0.5 {
        a + width * (t * t * t * (10.0 - 15.0 * t + 6.0 * t * t))
    } else {
        let r = 1.0 - t;
        b - width * (r * r * r * (10.0 - 15.0 * r + 6.0 * r * r))
    }
}

/// Work item for the adaptive loop.
#[derive(Clone, Copy, Debug)]
struct Segment {
    a: f64,
    b: f64,
    value: f64,
    err: f64,
    /// `|b − a|·max|f|` over its nodes ([`gk15_with`]).
    peak: f64,
}

impl Segment {
    fn new(a: f64, b: f64, (value, err, peak): (f64, f64, f64)) -> Segment {
        Segment {
            a,
            b,
            value: if value.is_finite() { value } else { 0.0 },
            err: if err.is_finite() { err } else { f64::INFINITY },
            peak: if peak.is_finite() {
                peak
            } else {
                f64::INFINITY
            },
        }
    }
}

/// When a segment of the adaptive loop is too narrow to bisect: its Kronrod
/// nodes would round onto each other or onto its ends.
#[derive(Clone, Copy)]
enum Floor<'a> {
    /// Narrower than `10⁻¹²` of the range (of its largest end, of 1): the
    /// plain bisection of [`quadrature_resolved`] (`evalf`).
    Range,
    /// Narrower than `10⁻¹²` of its own position — in the variable of the
    /// loop, and in `x` through the map when one is given — or than
    /// [`MIN_SEGMENT`]: towards an end at 0 the bisection goes as deep as
    /// `f64` allows, towards a nonzero end (in `x`) no deeper than before.
    Relative(Option<&'a dyn Fn(f64) -> f64>),
}

/// The narrowest segment [`Floor::Relative`] bisects: `10¹⁰` times the
/// smallest normal `f64`, so that its nodes stay normal numbers.
const MIN_SEGMENT: f64 = 1e10 * f64::MIN_POSITIVE;

/// Adaptive loop on a finite interval with a global error budget: the
/// estimate, and whether every segment was resolved (none left at the
/// narrowest width bisection can make, [`Floor`]).  The error of a segment
/// too narrow to bisect is at least its `peak`, `width·max|f|` at its
/// nodes: next to a singularity its estimate misses the mass between its
/// outermost node and its end, which `peak` covers for `|x − c|^α` with `α`
/// down to about `−0.99` (`0.0043^α ≥ 1/(1 + α)`, `0.0043` the relative
/// distance of the outermost Kronrod node to the end).
fn quadrature_finite_report(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    opts: &QuadOpts,
    raw: bool,
    floor: Floor<'_>,
) -> Result<(QuadResult, bool), SymplexError> {
    let nonfinite_err = |x: f64| SymplexError::ComputationFailed {
        operation: "integrate_numeric",
        reason: format!("integrand is not finite near x = {x:e}"),
    };
    // Segments narrower than this cannot be bisected meaningfully: the
    // Kronrod nodes would round onto the endpoints in `f64`.
    let min_width = 1e-12 * (b - a).abs().max(a.abs()).max(b.abs()).max(1.0);
    let too_narrow = |lo: f64, hi: f64| -> bool {
        let width = (hi - lo).abs();
        match floor {
            Floor::Range => width < min_width,
            Floor::Relative(map) => {
                let narrow =
                    |w: f64, l: f64, h: f64| w < MIN_SEGMENT || w < 1e-12 * l.abs().max(h.abs());
                narrow(width, lo, hi)
                    || map.is_some_and(|x| {
                        let (xl, xh) = (x(lo), x(hi));
                        xl.is_finite() && xh.is_finite() && narrow((xh - xl).abs(), xl, xh)
                    })
            }
        }
    };
    let mut segments: Vec<Segment> = vec![Segment::new(a, b, gk15_with(f, a, b, raw))];
    let mut total_value = segments[0].value;
    let mut total_err = segments[0].err;
    // A segment that could not be bisected further.
    let mut floor_hit = false;

    for _ in 0..opts.max_subdivisions {
        let tol = opts.abs_tol.max(opts.rel_tol * total_value.abs());
        if total_err.is_finite() && total_err <= tol {
            break;
        }
        // Pick the segment with the largest error.
        let (idx, _) = segments
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.err.partial_cmp(&y.1.err).unwrap_or(Ordering::Equal))
            .ok_or_else(|| SymplexError::ComputationFailed {
                operation: "integrate_numeric",
                reason: "empty work queue".into(),
            })?;
        let worst = segments.swap_remove(idx);
        if too_narrow(worst.a, worst.b) {
            if !worst.peak.is_finite() {
                return Err(nonfinite_err(0.5 * (worst.a + worst.b)));
            }
            // Cannot refine further: its estimate is no bound next to a
            // singularity, its `peak` is (a hundred of them for a rise
            // faster than the outermost nodes can extrapolate).
            let err = if worst.err.is_finite() {
                worst.err.max(worst.peak)
            } else {
                100.0 * worst.peak
            };
            segments.push(Segment { err, ..worst });
            floor_hit = true;
            total_err = segments.iter().map(|s| s.err).sum();
            break;
        }
        let mid = 0.5 * (worst.a + worst.b);
        segments.push(Segment::new(worst.a, mid, gk15_with(f, worst.a, mid, raw)));
        segments.push(Segment::new(mid, worst.b, gk15_with(f, mid, worst.b, raw)));
        total_value = segments.iter().map(|s| s.value).sum();
        total_err = segments.iter().map(|s| s.err).sum();
    }

    if !total_value.is_finite() || !total_err.is_finite() {
        return Err(SymplexError::ComputationFailed {
            operation: "integrate_numeric",
            reason:
                "quadrature did not produce a finite estimate (integrand non-finite or divergent)"
                    .into(),
        });
    }
    let resolved = !floor_hit;
    Ok((
        QuadResult {
            value: total_value,
            error: total_err,
        },
        resolved,
    ))
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
