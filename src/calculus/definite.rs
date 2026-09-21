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
//!    splitting the interval so that every remaining piece is smooth.
//! 3. **Known-value table.** Classical improper integrals (Gaussian, Gamma,
//!    Beta, Dirichlet, Wallis, Fresnel, Bose–Einstein, …) are matched
//!    structurally with symbolic parameters; any positivity conditions
//!    are checked through the assumption system and the entry is skipped
//!    when they cannot be established.
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
use rustc_hash::FxHashMap;

use crate::base::arena::Arena;
use crate::base::assumptions::{AssumptionCache, Props};
use crate::base::errors::SymplexError;
use crate::base::extended::Extended;
use crate::base::interval::Interval;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;
use crate::calculus::calculus_util::{self, BreakKind, BreakScan, LinearCoeffs};
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
    let result = integrate_range(arena, f, x, a, b, 0)?;
    let result = safe_eval(arena, result);
    if walk::contains(arena, result, x) {
        return Err(SymplexError::ComputationFailed {
            operation: "integrate_definite",
            reason: "result still depends on the integration variable".into(),
        });
    }
    Ok(result)
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

    // ── Breakpoint analysis → pieces ────────────────────────────────
    let pieces = split_at_breakpoints(arena, f, x, a, b)?;
    integrate_pieces(arena, f, x, &pieces, depth)
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
/// Exact for rationals and infinities; numeric (16 digits) for other
/// constants; assumption-based for symbolic differences.  `None` when the
/// order cannot be established.
fn cmp_points(arena: &mut Arena, p: ExprId, q: ExprId) -> Option<Ordering> {
    if p == q {
        return Some(Ordering::Equal);
    }
    if let (Some(rp), Some(rq)) = (arena.as_num(p), arena.as_num(q)) {
        return Some(rp.cmp(rq));
    }
    let pv = point_value(arena, p);
    let qv = point_value(arena, q);
    if let (Some(pv), Some(qv)) = (pv, qv) {
        if pv == qv {
            return Some(Ordering::Equal);
        }
        if pv.is_infinite() || qv.is_infinite() {
            return pv.partial_cmp(&qv);
        }
        let scale = pv.abs().max(qv.abs()).max(1.0);
        if (pv - qv).abs() <= 1e-13 * scale {
            return Some(Ordering::Equal);
        }
        return pv.partial_cmp(&qv);
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
        if v == 0.0 {
            return Some(Ordering::Equal);
        }
        return v.partial_cmp(&0.0);
    }
    let mut cache = AssumptionCache::new();
    if cache.query(arena, id, Props::POSITIVE) == Some(true) {
        return Some(Ordering::Greater);
    }
    if cache.query(arena, id, Props::NEGATIVE) == Some(true) {
        return Some(Ordering::Less);
    }
    if cache.query(arena, id, Props::ZERO) == Some(true) {
        return Some(Ordering::Equal);
    }
    None
}

/// Is `id` provably positive?
fn is_positive(arena: &mut Arena, id: ExprId) -> bool {
    sign_of(arena, id) == Some(Ordering::Greater)
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
/// unevaluated nodes, and (when constant) numerically finite and real?
fn is_finite_value(arena: &mut Arena, v: ExprId, x: ExprId) -> bool {
    if walk::contains(arena, v, x) || has_bad_atom(arena, v) || walk::has_unevaluated(arena, v) {
        return false;
    }
    if walk::free_symbols(arena, v).is_empty() && arena.as_num(v).is_none() {
        // Constant: verify numerically (catches Γ(0), atan(zoo)-like junk,
        // and complex values).
        return match evalf::evalf(arena, v, 16) {
            Ok(s) => match s.trim().parse::<f64>() {
                Ok(f) => f.is_finite(),
                Err(_) => {
                    // Complex-valued constant: allowed only if the
                    // imaginary part is negligible.
                    is_effectively_real(&s)
                }
            },
            Err(_) => false,
        };
    }
    true
}

/// Parse an `evalf` string of the form `a ± b*I` and check `|b| ≲ 0`.
fn is_effectively_real(s: &str) -> bool {
    let s = s.trim();
    if !s.contains('I') {
        return false;
    }
    // Find the split between real and imaginary parts.
    let bytes = s.as_bytes();
    let mut split = None;
    for i in (1..bytes.len()).rev() {
        if (bytes[i] == b'+' || bytes[i] == b'-') && bytes[i - 1] == b' ' {
            split = Some(i);
            break;
        }
    }
    let Some(i) = split else {
        return false;
    };
    let im = s[i..].trim().trim_end_matches("*I").trim();
    let im = im.trim_start_matches('+').trim();
    let im_val: f64 = match im.replace(' ', "").parse::<f64>() {
        Ok(v) => v,
        Err(_) => return false,
    };
    im_val.abs() < 1e-12
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

/// Finite numeric range for enumeration purposes (clamped for infinite
/// bounds: periodic families are only enumerated over a finite window).
fn finite_scan_range(range: Option<Interval<f64>>) -> Option<Interval<f64>> {
    let r = range?;
    if r.lower.is_finite() && r.upper.is_finite() {
        Some(r)
    } else {
        None
    }
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
    let scan = calculus_util::scan_breakpoints(arena, f, x, finite_scan_range(range));

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
            Some(false) => {
                return Err(failed(
                    "a denominator or function argument changes sign inside the interval at a point the solver could not locate",
                ));
            }
            None if !scan.complete => {
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

    // Order the interior points.
    if interior.iter().all(|(_, v)| v.is_some()) {
        interior.sort_by(|p, q| {
            p.1.unwrap_or(0.0)
                .partial_cmp(&q.1.unwrap_or(0.0))
                .unwrap_or(Ordering::Equal)
        });
        // Merge numerically coincident points.
        interior.dedup_by(|p, q| {
            let (pv, qv) = (p.1.unwrap_or(0.0), q.1.unwrap_or(0.0));
            (pv - qv).abs() <= 1e-12 * pv.abs().max(qv.abs()).max(1.0)
        });
    } else if interior.len() > 1 {
        // Symbolic interior points: sort with cmp_points, fail if unordered.
        let mut ordered: Vec<(ExprId, Option<f64>)> = Vec::new();
        for item in interior {
            let mut idx = ordered.len();
            for (i, other) in ordered.iter().enumerate() {
                match cmp_points(arena, item.0, other.0) {
                    Some(Ordering::Less) => {
                        idx = i;
                        break;
                    }
                    Some(Ordering::Equal) => {
                        idx = usize::MAX;
                        break;
                    }
                    Some(Ordering::Greater) => {}
                    None => return Err(failed("cannot order symbolic singular points")),
                }
            }
            if idx != usize::MAX {
                ordered.insert(idx, item);
            }
        }
        interior = ordered;
    }

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
/// Returns `Some(true)` if all sign changes are explained, `Some(false)`
/// if an unexplained one exists, `None` if the check could not run.
fn numeric_guard(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    lo: f64,
    hi: f64,
    scan: &BreakScan,
) -> Option<bool> {
    let x_name = match arena.node(x) {
        ExprNode::Symbol(sid) => arena.symbol_name(*sid).to_string(),
        _ => return None,
    };
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
    if watched.is_empty() {
        return Some(true);
    }
    let known: Vec<f64> = scan.points.iter().filter_map(|bp| bp.value).collect();
    let grid = guard_grid(lo, hi);
    let mut all_ok = true;
    for w in watched {
        let w_eval = safe_eval(arena, w);
        if !walk::free_symbols(arena, w_eval).iter().all(|&s| s == x) {
            return None; // parametric — cannot sample
        }
        let func = crate::output::lambdify::compile(arena, w_eval, &[&x_name]).ok()?;
        let mut prev: Option<(f64, f64)> = None;
        for &t in &grid {
            let v = func(&[t]);
            if !v.is_finite() {
                return None;
            }
            if let Some((pt, pv)) = prev
                && pv != 0.0
                && v != 0.0
                && (pv < 0.0) != (v < 0.0)
            {
                let explained = known.iter().any(|&k| k >= pt - 1e-9 && k <= t + 1e-9);
                if !explained {
                    all_ok = false;
                }
            }
            prev = Some((t, v));
        }
    }
    Some(all_ok)
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
                Some(false) => continue,
                Some(true) => {
                    live.push((*val, arena.bool_true()));
                    break;
                }
                None => live.push((*val, *cond)),
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
/// at any discontinuity of `F` inside the piece.
fn evaluate_antiderivative(
    arena: &mut Arena,
    f: ExprId,
    anti: ExprId,
    x: ExprId,
    p: &Piece,
    depth: usize,
) -> Result<ExprId, SymplexError> {
    // Discontinuities of F strictly inside (lo, hi).
    let range = numeric_range(arena, p.lo, p.hi);
    let scan = calculus_util::scan_breakpoints(arena, anti, x, finite_scan_range(range));
    if let Some(r) = range
        && numeric_guard(arena, anti, x, r.lower, r.upper, &scan) == Some(false)
    {
        return Err(failed(
            "the antiderivative has a discontinuity inside the interval that could not be located",
        ));
    }
    let mut inner_points: Vec<(ExprId, f64)> = Vec::new();
    for bp in &scan.points {
        // Both poles and kinks (sign/Heaviside jumps) of F matter here.
        match position_in(arena, bp.point, p.lo, p.hi) {
            Position::Inside => match bp.value {
                Some(v) => {
                    if !inner_points
                        .iter()
                        .any(|(_, w)| (w - v).abs() <= 1e-12 * v.abs().max(1.0))
                    {
                        inner_points.push((bp.point, v));
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
        inner_points.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
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
        if is_finite_value(arena, v, x) && (!singular || !has_jump_node(arena, anti)) {
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
        LimVal::Finite(v) => {
            if is_finite_value(arena, v, u) {
                return Some(Extended::Finite(v));
            }
        }
        LimVal::PosInf => return Some(Extended::PosInf),
        LimVal::NegInf => return Some(Extended::NegInf),
        LimVal::Bounded | LimVal::Unknown => {}
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
            if is_finite_value(arena, l, u) && numeric_limit_sanity(arena, g, u, l) {
                return Some(Extended::Finite(l));
            }
        }
    }
    None
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
            LimVal::Finite(_) => {}
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
    Finite(ExprId),
    PosInf,
    NegInf,
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

    for id in order {
        let node = arena.node(id).clone();
        let v = if id == u {
            LimVal::PosInf
        } else if !walk::contains(arena, id, u) {
            LimVal::Finite(id)
        } else {
            match node {
                ExprNode::Add(children) => {
                    let mut finite: Vec<ExprId> = Vec::new();
                    let mut pos = 0usize;
                    let mut neg = 0usize;
                    let mut bounded = false;
                    let mut unknown = false;
                    for c in children.iter() {
                        match vals.get(c).copied().unwrap_or(LimVal::Unknown) {
                            LimVal::Finite(v) => finite.push(v),
                            LimVal::PosInf => pos += 1,
                            LimVal::NegInf => neg += 1,
                            LimVal::Bounded => bounded = true,
                            LimVal::Unknown => unknown = true,
                        }
                    }
                    if unknown || (pos > 0 && neg > 0) {
                        LimVal::Unknown
                    } else if pos > 0 {
                        LimVal::PosInf
                    } else if neg > 0 {
                        LimVal::NegInf
                    } else if bounded {
                        LimVal::Bounded
                    } else {
                        let s = arena.add(&finite);
                        LimVal::Finite(safe_eval(arena, s))
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
                            LimVal::Finite(v) => finite.push(v),
                            LimVal::PosInf => infs += 1,
                            LimVal::NegInf => {
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
                                            LimVal::NegInf
                                        } else {
                                            LimVal::PosInf
                                        }
                                    }
                                }
                            }
                        } else if bounded {
                            if psign == Some(Ordering::Equal) {
                                LimVal::Finite(arena.zero())
                            } else {
                                LimVal::Bounded
                            }
                        } else {
                            LimVal::Finite(prod)
                        }
                    }
                }
                ExprNode::Neg(inner) => {
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Finite(v) => {
                            let n = arena.neg(v);
                            LimVal::Finite(safe_eval(arena, n))
                        }
                        LimVal::PosInf => LimVal::NegInf,
                        LimVal::NegInf => LimVal::PosInf,
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
                        LimVal::Finite(v) => {
                            let e = arena.exp(v);
                            LimVal::Finite(safe_eval(arena, e))
                        }
                        LimVal::PosInf => LimVal::PosInf,
                        LimVal::NegInf => LimVal::Finite(arena.zero()),
                        LimVal::Bounded => LimVal::Bounded,
                        LimVal::Unknown => LimVal::Unknown,
                    }
                }
                ExprNode::Ln(inner) => match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                    LimVal::Finite(v) => match sign_of(arena, v) {
                        Some(Ordering::Equal) => LimVal::NegInf,
                        Some(_) => {
                            let l = arena.ln(v);
                            LimVal::Finite(safe_eval(arena, l))
                        }
                        None => LimVal::Unknown,
                    },
                    LimVal::PosInf => LimVal::PosInf,
                    _ => LimVal::Unknown,
                },
                ExprNode::Atan(inner) => match vals.get(&inner).copied().unwrap_or(LimVal::Unknown)
                {
                    LimVal::Finite(v) => {
                        let a = arena.atan(v);
                        LimVal::Finite(safe_eval(arena, a))
                    }
                    LimVal::PosInf => {
                        let half_pi = arena.rational(1, 2);
                        let pi = arena.pi();
                        LimVal::Finite(arena.mul(&[half_pi, pi]))
                    }
                    LimVal::NegInf => {
                        let neg_half = arena.rational(-1, 2);
                        let pi = arena.pi();
                        LimVal::Finite(arena.mul(&[neg_half, pi]))
                    }
                    _ => LimVal::Unknown,
                },
                ExprNode::Tanh(inner) | ExprNode::Erf(inner) => {
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Finite(v) => {
                            let r = if matches!(node, ExprNode::Tanh(_)) {
                                arena.tanh(v)
                            } else {
                                arena.erf(v)
                            };
                            LimVal::Finite(safe_eval(arena, r))
                        }
                        LimVal::PosInf => LimVal::Finite(arena.one()),
                        LimVal::NegInf => LimVal::Finite(arena.neg_one()),
                        _ => LimVal::Unknown,
                    }
                }
                ExprNode::Erfc(inner) => match vals.get(&inner).copied().unwrap_or(LimVal::Unknown)
                {
                    LimVal::Finite(v) => {
                        let r = arena.erfc(v);
                        LimVal::Finite(safe_eval(arena, r))
                    }
                    LimVal::PosInf => LimVal::Finite(arena.zero()),
                    LimVal::NegInf => LimVal::Finite(arena.int(2)),
                    _ => LimVal::Unknown,
                },
                ExprNode::Sin(inner) | ExprNode::Cos(inner) => {
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Finite(v) => {
                            let r = if matches!(node, ExprNode::Sin(_)) {
                                arena.sin(v)
                            } else {
                                arena.cos(v)
                            };
                            LimVal::Finite(safe_eval(arena, r))
                        }
                        LimVal::PosInf | LimVal::NegInf | LimVal::Bounded => LimVal::Bounded,
                        LimVal::Unknown => LimVal::Unknown,
                    }
                }
                ExprNode::Sinh(inner) | ExprNode::Asinh(inner) => {
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Finite(v) => {
                            let r = if matches!(node, ExprNode::Sinh(_)) {
                                arena.sinh(v)
                            } else {
                                arena.asinh(v)
                            };
                            LimVal::Finite(safe_eval(arena, r))
                        }
                        LimVal::PosInf => LimVal::PosInf,
                        LimVal::NegInf => LimVal::NegInf,
                        _ => LimVal::Unknown,
                    }
                }
                ExprNode::Cosh(inner) => match vals.get(&inner).copied().unwrap_or(LimVal::Unknown)
                {
                    LimVal::Finite(v) => {
                        let r = arena.cosh(v);
                        LimVal::Finite(safe_eval(arena, r))
                    }
                    LimVal::PosInf | LimVal::NegInf => LimVal::PosInf,
                    _ => LimVal::Unknown,
                },
                ExprNode::Abs(inner) => {
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Finite(v) => {
                            let r = arena.abs(v);
                            LimVal::Finite(safe_eval(arena, r))
                        }
                        LimVal::PosInf | LimVal::NegInf => LimVal::PosInf,
                        LimVal::Bounded => LimVal::Bounded,
                        LimVal::Unknown => LimVal::Unknown,
                    }
                }
                ExprNode::Sign(inner) => match vals.get(&inner).copied().unwrap_or(LimVal::Unknown)
                {
                    LimVal::PosInf => LimVal::Finite(arena.one()),
                    LimVal::NegInf => LimVal::Finite(arena.neg_one()),
                    LimVal::Finite(v) => match sign_of(arena, v) {
                        Some(Ordering::Greater) => LimVal::Finite(arena.one()),
                        Some(Ordering::Less) => LimVal::Finite(arena.neg_one()),
                        _ => LimVal::Unknown,
                    },
                    _ => LimVal::Unknown,
                },
                ExprNode::Acosh(inner) | ExprNode::Gamma(inner) => {
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::PosInf => LimVal::PosInf,
                        LimVal::Finite(v) if is_positive(arena, v) => {
                            let r = if matches!(node, ExprNode::Acosh(_)) {
                                arena.acosh(v)
                            } else {
                                arena.gamma(v)
                            };
                            LimVal::Finite(safe_eval(arena, r))
                        }
                        _ => LimVal::Unknown,
                    }
                }
                ExprNode::Asin(inner) | ExprNode::Acos(inner) | ExprNode::Atanh(inner) => {
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Finite(v) => {
                            let r = match node {
                                ExprNode::Asin(_) => arena.asin(v),
                                ExprNode::Acos(_) => arena.acos(v),
                                _ => arena.atanh(v),
                            };
                            let r = safe_eval(arena, r);
                            if is_finite_value(arena, r, u) {
                                LimVal::Finite(r)
                            } else {
                                LimVal::Unknown
                            }
                        }
                        _ => LimVal::Unknown,
                    }
                }
                _ => LimVal::Unknown,
            }
        };
        vals.insert(id, v);
    }
    vals.get(&g).copied().unwrap_or(LimVal::Unknown)
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
            LimVal::Finite(b) => {
                let b_zero = sign_of(arena, b) == Some(Ordering::Equal);
                if b_zero {
                    // 0^e: 0 for e > 0, undefined/infinite otherwise.
                    return if e_sign == Ordering::Greater {
                        LimVal::Finite(arena.zero())
                    } else {
                        LimVal::Unknown
                    };
                }
                let p = arena.pow(b, exp);
                let p = safe_eval(arena, p);
                if is_finite_value(arena, p, u) {
                    LimVal::Finite(p)
                } else {
                    LimVal::Unknown
                }
            }
            LimVal::PosInf => match e_sign {
                Ordering::Greater => LimVal::PosInf,
                Ordering::Less => LimVal::Finite(arena.zero()),
                Ordering::Equal => LimVal::Finite(arena.one()),
            },
            LimVal::NegInf => match e_sign {
                Ordering::Greater => match e_int_parity {
                    Some(true) => LimVal::PosInf,
                    Some(false) => LimVal::NegInf,
                    None => LimVal::Unknown,
                },
                Ordering::Less => LimVal::Finite(arena.zero()),
                Ordering::Equal => LimVal::Finite(arena.one()),
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
            (LimVal::Finite(e), _) => {
                let p = arena.pow(base, e);
                LimVal::Finite(safe_eval(arena, p))
            }
            (_, Ordering::Equal) => LimVal::Finite(arena.one()),
            (LimVal::PosInf, Ordering::Greater) | (LimVal::NegInf, Ordering::Less) => {
                LimVal::PosInf
            }
            (LimVal::PosInf, Ordering::Less) | (LimVal::NegInf, Ordering::Greater) => {
                LimVal::Finite(arena.zero())
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
    let scan = calculus_util::scan_breakpoints(arena, f, x, finite_scan_range(range));

    // Only kinks matter here; singularities are handled downstream.
    let mut points: Vec<(ExprId, Option<f64>)> = Vec::new();
    for bp in &scan.points {
        if bp.kind != BreakKind::Kink {
            continue;
        }
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
    points.sort_by(|p, q| {
        p.1.unwrap_or(0.0)
            .partial_cmp(&q.1.unwrap_or(0.0))
            .unwrap_or(Ordering::Equal)
    });
    points.dedup_by(|p, q| match (p.1, q.1) {
        (Some(pv), Some(qv)) => (pv - qv).abs() <= 1e-12 * pv.abs().max(qv.abs()).max(1.0),
        _ => false,
    });

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

/// A rational point strictly inside `(lo, hi)` (infinite ends allowed).
fn test_point(arena: &mut Arena, lo: ExprId, hi: ExprId) -> Option<ExprId> {
    let lo_v = point_value(arena, lo)?;
    let hi_v = point_value(arena, hi)?;
    let m = match (lo_v.is_finite(), hi_v.is_finite()) {
        (true, true) => {
            if let (Some(rl), Some(rh)) = (arena.as_num(lo).cloned(), arena.as_num(hi).cloned()) {
                let two = Ratio::from_integer(BigInt::from(2));
                let mid = (rl + rh) / two;
                let nid = arena.intern_num(mid);
                return Some(arena.intern(ExprNode::Num(nid)));
            }
            0.5 * (lo_v + hi_v)
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
                    let v = value_at(arena, inner, x, mid)?;
                    let n = if matches!(node, ExprNode::Floor(_)) {
                        v.floor()
                    } else {
                        v.ceil()
                    };
                    if n.abs() > 1e15 {
                        return Err(failed("floor/ceiling argument too large"));
                    }
                    Some(arena.int(n as i64))
                }
                ExprNode::Piecewise(pairs) if walk::contains(arena, id, x) => {
                    let mut chosen: Option<ExprId> = None;
                    for (val, cond) in pairs.iter() {
                        if condition_at(arena, *cond, x, mid)? {
                            chosen = Some(*val);
                            break;
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

/// Sign of `g` at `x = mid`.
fn sign_at(arena: &mut Arena, g: ExprId, x: ExprId, mid: ExprId) -> Result<Ordering, SymplexError> {
    let v = value_at(arena, g, x, mid)?;
    Ok(if v == 0.0 {
        Ordering::Equal
    } else if v < 0.0 {
        Ordering::Less
    } else {
        Ordering::Greater
    })
}

/// Decide a parameter condition (no integration variable involved) with
/// the assumption system: `Some(true/false)` or `None` if undecidable.
fn decide_condition(arena: &mut Arena, cond: ExprId) -> Option<bool> {
    let order = walk::post_order_ids(arena, cond);
    let mut vals: FxHashMap<ExprId, Option<bool>> = FxHashMap::default();
    for id in order {
        let node = arena.node(id).clone();
        let v: Option<bool> = match node {
            ExprNode::BoolTrue => Some(true),
            ExprNode::BoolFalse => Some(false),
            ExprNode::Gt(a, b) | ExprNode::Ge(a, b) | ExprNode::Eq_(a, b) | ExprNode::Ne(a, b) => {
                let d = arena.sub(a, b);
                let d = safe_eval(arena, d);
                let s = sign_of(arena, d);
                match node {
                    ExprNode::Gt(..) => s.map(|s| s == Ordering::Greater),
                    ExprNode::Ge(..) => s.map(|s| s != Ordering::Less),
                    ExprNode::Eq_(..) => s.map(|s| s == Ordering::Equal),
                    _ => s.map(|s| s != Ordering::Equal),
                }
            }
            ExprNode::And(children) => {
                let parts: Vec<Option<bool>> = children
                    .iter()
                    .map(|c| vals.get(c).copied().flatten())
                    .collect();
                if parts.contains(&Some(false)) {
                    Some(false)
                } else if parts.iter().all(|p| *p == Some(true)) {
                    Some(true)
                } else {
                    None
                }
            }
            ExprNode::Or(children) => {
                let parts: Vec<Option<bool>> = children
                    .iter()
                    .map(|c| vals.get(c).copied().flatten())
                    .collect();
                if parts.contains(&Some(true)) {
                    Some(true)
                } else if parts.iter().all(|p| *p == Some(false)) {
                    Some(false)
                } else {
                    None
                }
            }
            ExprNode::Not(inner) => vals.get(&inner).copied().flatten().map(|b| !b),
            _ => None,
        };
        vals.insert(id, v);
    }
    vals.get(&cond).copied().flatten()
}

/// Truth value of a boolean condition at `x = mid`.
fn condition_at(
    arena: &mut Arena,
    cond: ExprId,
    x: ExprId,
    mid: ExprId,
) -> Result<bool, SymplexError> {
    let order = walk::post_order_ids(arena, cond);
    let mut vals: FxHashMap<ExprId, bool> = FxHashMap::default();
    for id in order {
        let node = arena.node(id).clone();
        let v = match node {
            ExprNode::BoolTrue => true,
            ExprNode::BoolFalse => false,
            ExprNode::Gt(p, q) | ExprNode::Ge(p, q) | ExprNode::Eq_(p, q) | ExprNode::Ne(p, q) => {
                let d = arena.sub(p, q);
                let dv = value_at(arena, d, x, mid)?;
                match node {
                    ExprNode::Gt(..) => dv > 0.0,
                    ExprNode::Ge(..) => dv >= 0.0,
                    ExprNode::Eq_(..) => dv.abs() <= 1e-12,
                    _ => dv.abs() > 1e-12,
                }
            }
            ExprNode::And(children) => children
                .iter()
                .all(|c| vals.get(c).copied().unwrap_or(false)),
            ExprNode::Or(children) => children
                .iter()
                .any(|c| vals.get(c).copied().unwrap_or(false)),
            ExprNode::Not(inner) => !vals.get(&inner).copied().unwrap_or(false),
            // Relational sub-terms (numbers, symbols) are not booleans.
            _ => {
                if id == cond {
                    return Err(failed("unsupported Piecewise condition"));
                }
                false
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
                    // context; the canon layer only flattens integers).
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
fn as_ratio(arena: &Arena, e: ExprId) -> Option<Ratio<BigInt>> {
    arena.as_num(e).cloned()
}

/// `Pow(base, −m)` → `(base, m)` with `m` a positive number; a bare
/// non-power expression is not a reciprocal.
fn as_reciprocal(arena: &Arena, e: ExprId) -> Option<(ExprId, Ratio<BigInt>)> {
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
fn as_trig_power(arena: &Arena, e: ExprId) -> Option<(Trig, ExprId, Ratio<BigInt>)> {
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
fn as_ln_power(arena: &Arena, e: ExprId) -> Option<(ExprId, Ratio<BigInt>)> {
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
fn gamma_exact(arena: &mut Arena, r: &Ratio<BigInt>) -> Option<ExprId> {
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
fn as_binomial_in_x(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
) -> Option<(ExprId, ExprId, Ratio<BigInt>)> {
    let terms: Vec<ExprId> = match arena.node(e).clone() {
        ExprNode::Add(ch) => ch.to_vec(),
        _ => return None,
    };
    if terms.len() != 2 {
        return None;
    }
    let mut k: Option<ExprId> = None;
    let mut cx: Option<(ExprId, Ratio<BigInt>)> = None;
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
    trig: Vec<(Trig, ExprId, Ratio<BigInt>)>,
    /// `(base, m)` of reciprocal factors
    recip: Vec<(ExprId, Ratio<BigInt>)>,
    /// `(arg, n)` of a logarithm factor
    ln: Option<(ExprId, Ratio<BigInt>)>,
    /// `(arg, m)` of `cosh(arg)^{-m}` / `sinh(arg)^{-m}` (kind: true = cosh)
    hyp_recip: Option<(bool, ExprId, Ratio<BigInt>)>,
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
fn ratio_expr(arena: &mut Arena, r: &Ratio<BigInt>) -> ExprId {
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
fn wallis(arena: &mut Arena, m: &Ratio<BigInt>, n: &Ratio<BigInt>) -> Option<ExprId> {
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
            ExprNode::Add(ch) if ch.len() == 2 => ch.to_vec(),
            _ => return None,
        };
        let mut a_c: Option<ExprId> = None;
        let mut b_c: Option<(ExprId, Trig)> = None;
        for t in terms {
            if !walk::contains(arena, t, x) {
                a_c = Some(t);
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
    /// Estimated absolute error of [`value`](Self::value).
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

/// One G7/K15 rule application on `[a, b]`.  Returns `(k15, err)`; `err`
/// is `NaN` if the integrand was non-finite at a node.
fn gk15(f: &dyn Fn(f64) -> f64, a: f64, b: f64) -> (f64, f64) {
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
        return (f64::NAN, f64::NAN);
    }
    let mean = 0.5 * k15;
    let mut resasc = WGK[7] * (fc - mean).abs();
    for j in 0..7 {
        resasc += WGK[j] * ((fv[j] - mean).abs() + (fv[14 - j] - mean).abs());
    }
    let result = k15 * half;
    let resasc = resasc * half.abs();
    let mut err = ((k15 - g7) * half).abs();
    if resasc != 0.0 && err != 0.0 {
        err = resasc * (200.0 * err / resasc).powf(1.5).min(1.0);
    }
    let resabs_floor = 50.0 * f64::EPSILON * result.abs();
    if resabs_floor > err {
        err = resabs_floor;
    }
    (result, err)
}

/// Adaptive Gauss–Kronrod quadrature of `f` over `[a, b]`.
///
/// Infinite bounds are handled by the substitutions `x = a + t/(1−t)`,
/// `x = b − t/(1−t)` and `x = t/(1−t²)`; endpoint singularities are
/// absorbed by adaptive bisection (the rule never samples endpoints).
///
/// Returns the estimate and its error estimate as a [`QuadResult`].  If the
/// integrand is non-finite on a vanishing interval (a non-integrable
/// interior singularity) or the estimate is not a finite number,
/// `Err(ComputationFailed)` is returned.
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
    if a.is_nan() || b.is_nan() {
        return Err(SymplexError::InvalidArgument {
            operation: "integrate_numeric",
            reason: "integration bounds must not be NaN".into(),
        });
    }
    if a == b {
        return Ok(QuadResult {
            value: 0.0,
            error: 0.0,
        });
    }
    if a > b {
        let r = quadrature(f, b, a, opts)?;
        return Ok(QuadResult {
            value: -r.value,
            error: r.error,
        });
    }
    // Transform infinite ranges onto finite ones.
    match (a.is_finite(), b.is_finite()) {
        (true, true) => quadrature_finite_adaptive(f, a, b, opts),
        (true, false) => {
            // x = a + t/(1−t), t ∈ [0, 1), dx = dt/(1−t)²
            let g = move |t: f64| {
                let om = 1.0 - t;
                f(a + t / om) / (om * om)
            };
            quadrature_finite_adaptive(&g, 0.0, 1.0, opts)
        }
        (false, true) => {
            // x = b − t/(1−t)
            let g = move |t: f64| {
                let om = 1.0 - t;
                f(b - t / om) / (om * om)
            };
            quadrature_finite_adaptive(&g, 0.0, 1.0, opts)
        }
        (false, false) => {
            // Split at 0 so that each half must converge on its own: a
            // single symmetric rule on t/(1−t²) would report ∫ x dx = 0
            // with zero error (a numeric principal value).
            let half = QuadOpts {
                max_subdivisions: opts.max_subdivisions.div_ceil(2),
                ..*opts
            };
            let left = quadrature(f, f64::NEG_INFINITY, 0.0, &half)?;
            let right = quadrature(f, 0.0, f64::INFINITY, &half)?;
            Ok(QuadResult {
                value: left.value + right.value,
                error: left.error + right.error,
            })
        }
    }
}

/// Finite-interval driver: plain adaptive G7/K15 first; if that does not
/// meet the tolerance (typically an algebraic or logarithmic endpoint
/// singularity, where bisection alone is limited by `f64` resolution near
/// the endpoint), retry under the endpoint-smoothing substitution
/// `x = a + (b−a)·s(t)`, `s(t) = t³(10 − 15t + 6t²)` whose derivative
/// vanishes quadratically at both ends, and keep the better estimate.
fn quadrature_finite_adaptive(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    opts: &QuadOpts,
) -> Result<QuadResult, SymplexError> {
    let plain = quadrature_finite(f, a, b, opts);
    if let Ok(r) = plain
        && r.error <= opts.abs_tol.max(opts.rel_tol * r.value.abs())
    {
        return Ok(r);
    }
    let width = b - a;
    let g = move |t: f64| {
        let ds = 30.0 * t * t * (1.0 - t) * (1.0 - t);
        if ds == 0.0 {
            return 0.0;
        }
        // Evaluate the map from the nearer endpoint (s(1−t) = 1 − s(t)) so
        // that x keeps full relative precision near both ends, and never
        // lands exactly on an endpoint.
        let x = if t <= 0.5 {
            let s = t * t * t * (10.0 - 15.0 * t + 6.0 * t * t);
            let x = a + width * s;
            if x <= a { a.next_up() } else { x }
        } else {
            let r = 1.0 - t;
            let s = r * r * r * (10.0 - 15.0 * r + 6.0 * r * r);
            let x = b - width * s;
            if x >= b { b.next_down() } else { x }
        };
        f(x) * width * ds
    };
    let mapped = quadrature_finite(&g, 0.0, 1.0, opts);
    match (plain, mapped) {
        (Ok(p), Ok(m)) => Ok(if m.error < p.error { m } else { p }),
        (Ok(p), Err(_)) => Ok(p),
        (Err(_), Ok(m)) => Ok(m),
        (Err(e), Err(_)) => Err(e),
    }
}

/// Work item for the adaptive loop.
#[derive(Clone, Copy, Debug)]
struct Segment {
    a: f64,
    b: f64,
    value: f64,
    err: f64,
}

/// Adaptive loop on a finite interval with a global error budget.
fn quadrature_finite(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    opts: &QuadOpts,
) -> Result<QuadResult, SymplexError> {
    let nonfinite_err = |x: f64| SymplexError::ComputationFailed {
        operation: "integrate_numeric",
        reason: format!("integrand is not finite near x = {x}"),
    };
    // Segments narrower than this cannot be bisected meaningfully: the
    // Kronrod nodes would round onto the endpoints in `f64`.
    let min_width = 1e-12 * (b - a).abs().max(a.abs()).max(b.abs()).max(1.0);
    let (v0, e0) = gk15(f, a, b);
    let mut segments: Vec<Segment> = vec![Segment {
        a,
        b,
        value: if v0.is_finite() { v0 } else { 0.0 },
        err: if e0.is_finite() { e0 } else { f64::INFINITY },
    }];
    let mut total_value = segments[0].value;
    let mut total_err = segments[0].err;

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
        if (worst.b - worst.a).abs() < min_width {
            if !worst.err.is_finite() {
                return Err(nonfinite_err(0.5 * (worst.a + worst.b)));
            }
            // Cannot refine further; keep it as is.
            segments.push(worst);
            break;
        }
        let mid = 0.5 * (worst.a + worst.b);
        let (v1, e1) = gk15(f, worst.a, mid);
        let (v2, e2) = gk15(f, mid, worst.b);
        let s1 = Segment {
            a: worst.a,
            b: mid,
            value: if v1.is_finite() { v1 } else { 0.0 },
            err: if e1.is_finite() { e1 } else { f64::INFINITY },
        };
        let s2 = Segment {
            a: mid,
            b: worst.b,
            value: if v2.is_finite() { v2 } else { 0.0 },
            err: if e2.is_finite() { e2 } else { f64::INFINITY },
        };
        segments.push(s1);
        segments.push(s2);
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
    Ok(QuadResult {
        value: total_value,
        error: total_err,
    })
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
