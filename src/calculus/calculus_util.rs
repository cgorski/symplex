//! Symbolic domain analysis utilities.
//!
//! This module provides functions for analysing the domain of continuity,
//! locating singularities, and estimating oscillation frequency of symbolic
//! expressions.
//!
//! All traversals use explicit stacks (no recursion) following symplex
//! Principle 5.

use crate::base::arena::Arena;
use crate::base::interval::Interval;
use crate::base::interval::IntervalKind;
use crate::base::node::{ExprId, ExprNode, SymbolId};
use crate::base::truth::Truth;
use crate::base::walk;
use crate::transforms::eval;
use crate::transforms::evalf;
use crate::transforms::inequalities::Relation;
use crate::transforms::solve;
use num_bigint::BigInt;
use num_rational::Ratio;

// ═══════════════════════════════════════════════════════════════════════════
// continuous_domain
// ═══════════════════════════════════════════════════════════════════════════

/// Returns the domain on which `expr` is continuous over `domain`.
///
/// Uses symbolic analysis of the expression tree to find singularities
/// and restricted domains (square roots, logarithms, inverse trig, etc.),
/// then intersects the valid region with the supplied `domain`.
#[allow(dead_code)]
pub(crate) fn continuous_domain(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    _var_sym: SymbolId,
    domain: ExprId,
) -> ExprId {
    tracing::debug!("continuous_domain: starting domain analysis");

    let post_order = walk::post_order_ids(arena, expr);
    let mut valid = domain;

    for &id in &post_order {
        let node = arena.node(id).clone();
        let constraint = compute_node_constraint(arena, &node, var);
        if let Some(c) = constraint {
            valid = arena.set_intersection(&[valid, c]);
        }
    }

    valid
}

/// Compute the domain constraint imposed by a single node.
///
/// Returns `Some(set)` if the node restricts the domain, `None` otherwise.
#[allow(dead_code)]
fn compute_node_constraint(arena: &mut Arena, node: &ExprNode, var: ExprId) -> Option<ExprId> {
    match node {
        // ── Pow(base, exp): negative or fractional exponents ────────
        ExprNode::Pow(base, exp) => {
            let base = *base;
            let exp = *exp;

            if !walk::contains(arena, base, var) {
                return None;
            }

            // Check if exponent is negative → exclude zeros of base
            if let Some(r) = arena.as_num(exp).cloned() {
                use num_traits::Signed;
                if r.is_negative() {
                    // base ≠ 0  →  ℝ \ {zeros of base}
                    return Some(domain_exclude_zeros(arena, base, var));
                }
                // Check if exponent is 1/2 (sqrt) → require base ≥ 0
                let half = num_rational::Ratio::new(
                    num_bigint::BigInt::from(1),
                    num_bigint::BigInt::from(2),
                );
                if r == half {
                    return solve_ge_zero(arena, base, var);
                }
            }

            // Check if exponent is Neg(something) meaning negative
            if let ExprNode::Neg(_) = arena.node(exp)
                && walk::contains(arena, base, var)
            {
                return Some(domain_exclude_zeros(arena, base, var));
            }

            None
        }

        // ── Ln(inner): require inner > 0 ───────────────────────────
        ExprNode::Ln(inner) => {
            let inner = *inner;
            if !walk::contains(arena, inner, var) {
                return None;
            }
            solve_gt_zero(arena, inner, var)
        }

        // ── Tan(inner): exclude where cos(inner) = 0 ──────────────
        ExprNode::Tan(inner) => {
            let inner = *inner;
            if !walk::contains(arena, inner, var) {
                return None;
            }
            // cos(inner) = 0  when inner = π/2 + n·π
            // For inner = a*x + b (linear), solve for x.
            let cos_inner = arena.cos(inner);
            Some(domain_exclude_zeros(arena, cos_inner, var))
        }

        // ── Asin / Acos: require -1 ≤ inner ≤ 1 ──────────────────
        ExprNode::Asin(inner) | ExprNode::Acos(inner) => {
            let inner = *inner;
            if !walk::contains(arena, inner, var) {
                return None;
            }
            // inner ≥ -1  AND  inner ≤ 1
            // i.e. inner - (-1) ≥ 0  AND  1 - inner ≥ 0
            let neg_one = arena.neg_one;
            let one = arena.one;
            // inner + 1 ≥ 0
            let shifted_low = arena.sub(inner, neg_one); // inner - (-1) = inner + 1
            let set_low = solve_ge_zero(arena, shifted_low, var);
            // 1 - inner ≥ 0
            let shifted_high = arena.sub(one, inner);
            let set_high = solve_ge_zero(arena, shifted_high, var);
            match (set_low, set_high) {
                (Some(a), Some(b)) => Some(arena.set_intersection(&[a, b])),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            }
        }

        // ── Acosh: require inner ≥ 1 ──────────────────────────────
        ExprNode::Acosh(inner) => {
            let inner = *inner;
            if !walk::contains(arena, inner, var) {
                return None;
            }
            let one = arena.one;
            let shifted = arena.sub(inner, one); // inner - 1
            solve_ge_zero(arena, shifted, var)
        }

        // ── Atanh: require -1 < inner < 1 ─────────────────────────
        ExprNode::Atanh(inner) => {
            let inner = *inner;
            if !walk::contains(arena, inner, var) {
                return None;
            }
            let neg_one = arena.neg_one;
            let one = arena.one;
            // inner + 1 > 0
            let shifted_low = arena.sub(inner, neg_one);
            let set_low = solve_gt_zero(arena, shifted_low, var);
            // 1 - inner > 0
            let shifted_high = arena.sub(one, inner);
            let set_high = solve_gt_zero(arena, shifted_high, var);
            match (set_low, set_high) {
                (Some(a), Some(b)) => Some(arena.set_intersection(&[a, b])),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            }
        }

        _ => None,
    }
}

/// Solve `expr > 0` for `var`, returning the solution set.
#[allow(dead_code)]
fn solve_gt_zero(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<ExprId> {
    arena.solve_inequality_expr(expr, var, Relation::Gt).ok()
}

/// Solve `expr >= 0` for `var`, returning the solution set.
#[allow(dead_code)]
fn solve_ge_zero(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<ExprId> {
    arena.solve_inequality_expr(expr, var, Relation::Ge).ok()
}

/// Build the domain that excludes the zeros of `expr` w.r.t. `var`.
///
/// Returns `ℝ \ {roots of expr}`.  If no roots are found, returns the
/// full real line (no restriction).
#[allow(dead_code)]
fn domain_exclude_zeros(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    let solutions = solve::known_real_roots(arena, expr, var);
    if solutions.is_empty() {
        // No zeros found → no restriction (denominator never zero for real x)
        return arena.interval(arena.neg_infinity, arena.infinity, IntervalKind::Open);
    }

    let roots_set = arena.finite_set(&solutions);
    let reals = arena.interval(arena.neg_infinity, arena.infinity, IntervalKind::Open);
    arena.set_complement(reals, roots_set)
}

// ═══════════════════════════════════════════════════════════════════════════
// singularities
// ═══════════════════════════════════════════════════════════════════════════

/// Returns the x-locations of singularities (poles, branch points) in
/// the closed numeric range `[range.lower, range.upper]`.
///
/// Walks the expression tree, finds denominators and constrained
/// functions (ln, sqrt, tan, …), and solves for zeros/boundaries
/// numerically in the given range.  Zeros of `sin`/`cos`/`tan` with a
/// linear argument are enumerated over the whole range (the generic
/// solver only reports one period).
pub(crate) fn singularities(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    _var_sym: SymbolId,
    range: Interval<f64>,
) -> Vec<f64> {
    tracing::debug!(
        range_lo = range.lower,
        range_hi = range.upper,
        "singularities: scanning for singularities"
    );

    let scan_range = ScanRange::from_f64(arena, range);
    let scan = scan_breakpoints(arena, expr, var, scan_range);
    let mut sing_points: Vec<f64> = Vec::new();

    for bp in scan.points {
        if bp.kind != BreakKind::Singular {
            continue;
        }
        if let Some(val) = bp.value
            && val.is_finite()
            && val >= range.lower
            && val <= range.upper
        {
            // Deduplicate (`f64` points for plots and quadrature splits: a
            // guide).
            if !sing_points.iter().any(|&v| (v - val).abs() < 1e-12) {
                sing_points.push(val);
            }
        }
    }

    sing_points.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sing_points
}

/// How a breakpoint affects the integrand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BreakKind {
    /// The integrand (or its antiderivative) may blow up or leave the
    /// real domain here: poles, `ln` at zero, `tan` poles, fractional
    /// powers of a sign-changing base, inverse-trig domain edges.
    Singular,
    /// The integrand is bounded but not smooth here: `abs`, `sign`,
    /// `Heaviside`, `floor`/`ceiling` steps, and `Piecewise` condition
    /// boundaries.
    Kink,
}

/// A candidate breakpoint of an integrand.
#[derive(Clone, Debug)]
pub(crate) struct Breakpoint {
    /// The location as a symbolic expression (may contain parameters).
    pub point: ExprId,
    /// Numeric value when the location is a real constant, within `2⁻⁴⁶` of
    /// it relatively ([`POINT_VALUE_ACCURACY`]): two values further apart
    /// than `2⁻⁴⁰` show the order of their points, closer ones do not.
    pub value: Option<f64>,
    /// Whether this is a genuine singularity or just a kink.
    pub kind: BreakKind,
}

/// Result of scanning an expression for breakpoints.
#[derive(Clone, Debug, Default)]
pub(crate) struct BreakScan {
    /// All candidate breakpoints (unsorted, may contain duplicates).
    pub points: Vec<Breakpoint>,
    /// `false` when some sub-expression depending on `var` could have
    /// zeros/poles that the solver was unable to locate (e.g. a
    /// transcendental denominator `solve` gave up on).  A numeric sampling
    /// guard may still vouch for such an expression.
    pub complete: bool,
    /// `true` when the expression contains a node whose behaviour cannot be
    /// analysed at all (unknown `Apply` functions, formal integrals /
    /// limits, `RootOf`, `LambertW`).  No numeric guard can compensate.
    pub opaque: bool,
    /// `true` when a periodic family (`sin`/`cos`/`tan` zeros, `Gamma`
    /// poles, `floor` steps) has more members in the range than the scan
    /// enumerates ([`MAX_FAMILY_POINTS`]); `complete` is then `false` too.
    pub crowded: bool,
    /// `true` when the zeros of a periodic composition (`cos(x²) − 1/7`,
    /// `sin x − cos x`) are the roots of a principal solve, one period of
    /// infinitely many: their translates may be missing.  The definite
    /// integrator's numeric guard samples such an expression for sign
    /// changes no point explains; the quadrature, which has no guard,
    /// locates its kinks numerically.
    pub periodic: bool,
}

/// The relative accuracy of [`Breakpoint::value`]: `2⁻⁴⁶`.  A 16-digit
/// evaluation has it, and so has the `f64` formula of a family member when
/// it cancels less than four bits.
pub(crate) const POINT_VALUE_ACCURACY: f64 = 1.4210854715202004e-14;

/// The most members of one periodic family a scan enumerates.
const MAX_FAMILY_POINTS: u32 = 10_001;

/// The range over which [`scan_breakpoints`] enumerates periodic families
/// (`sin`/`cos`/`tan` zeros, `Gamma` poles, `floor` steps): its exact ends,
/// real constants with finite values, in either order.
///
/// The members of a family `α·x + β = k·c + d` inside are those with `k`
/// from the certified ceiling and floor of `(α·end + β − d)/c` at the exact
/// ends, and each is an exact expression.  Up to 0.49 they were enumerated
/// from the `f64` ends with a tolerance of `10⁻¹²`: on `[1, 1 + 4·10⁻²⁰]`,
/// a single `f64`, the zeros of `sin(10²⁰x)` could not be told from the
/// ends, and `∫|sin(10²⁰x)| dx` there was `1.75·10⁻²⁰` (truly
/// `2.25·10⁻²⁰`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScanRange {
    /// The exact ends.
    pub exact: Interval<ExprId>,
}

impl ScanRange {
    /// The range between the real constants `lower` and `upper`, whose
    /// values are `approx`; `None` when a value is not finite.
    pub(crate) fn new(lower: ExprId, upper: ExprId, approx: Interval<f64>) -> Option<Self> {
        (approx.lower.is_finite() && approx.upper.is_finite()).then_some(ScanRange {
            exact: Interval::closed(lower, upper),
        })
    }

    /// The `f64` range `r` itself, its ends taken as the exact binary
    /// numbers they are; `None` when an end is not finite.
    pub(crate) fn from_f64(arena: &mut Arena, r: Interval<f64>) -> Option<Self> {
        let lower = crate::base::numeric::f64_to_ratio_exact(r.lower)?;
        let upper = crate::base::numeric::f64_to_ratio_exact(r.upper)?;
        let (lower, upper) = (arena.num_ratio(lower), arena.num_ratio(upper));
        ScanRange::new(lower, upper, r)
    }
}

/// Scan `expr` for every point where it (or an antiderivative of it) may
/// fail to be smooth with respect to `var`.
///
/// When `range` is given, periodic families (`sin`, `cos`, `tan` with a
/// linear argument, `Gamma` poles, `floor` steps) are enumerated inside
/// the closed range between its exact ends ([`ScanRange`]); without a range
/// only the solver's representative roots are reported and `complete` is
/// cleared for such families.
///
/// Uses an explicit post-order walk (no recursion).
pub(crate) fn scan_breakpoints(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    range: Option<ScanRange>,
) -> BreakScan {
    let post_order = walk::post_order_ids(arena, expr);
    let mut scan = BreakScan {
        points: Vec::new(),
        complete: true,
        opaque: false,
        crowded: false,
        periodic: false,
    };

    for &id in &post_order {
        let node = arena.node(id).clone();
        breakpoints_of_node(arena, &node, var, range, &mut scan);
    }

    scan
}

/// Push the real zeros of `g` (w.r.t. `var`) as breakpoints of `kind`.
///
/// Trigonometric zero families with a linear argument are enumerated over
/// `range`.  Returns `false` if `g` depends on `var` but no zero could be
/// determined at all (the caller records incompleteness).
fn push_zeros(
    arena: &mut Arena,
    g: ExprId,
    var: ExprId,
    kind: BreakKind,
    range: Option<ScanRange>,
    scan: &mut BreakScan,
) {
    if !walk::contains(arena, g, var) {
        return;
    }

    // Periodic families first: sin/cos/tan of a linear argument.
    match push_trig_zeros(arena, g, var, kind, range, scan) {
        TrigZeros::NotTrig => {}
        TrigZeros::Pushed => return,
        TrigZeros::Incomplete => {
            scan.complete = false;
            return;
        }
    }

    // Every real root, or the scan is incomplete (a root list that misses
    // a branch).  The principal solve lists one period of a periodic
    // composition (`cos(x²) = 1/7` gives `±√acos(1/7)` only): `periodic`.
    // Up to 0.49 nothing said so, and the quadrature of `sign(cos(x²) −
    // 1/7)` over `[0, 3]` ran across the zeros it lacked (`0.5178` for
    // `0.5360`).
    use solve::{RootSet, Solutions};
    let (solutions, complete) = match solve::solve_classified_real(arena, g, var).into_found() {
        Ok((roots, missing)) => {
            if periodic_in(arena, g, var) {
                scan.periodic = true;
            }
            (roots, missing.is_none())
        }
        Err(Solutions::Complete(RootSet::Empty(_))) => (Vec::new(), true),
        Err(_) => {
            // A polynomial with no real roots is fine (solve returns complex
            // roots explicitly).  Anything the solver gave up on is a hole in
            // our knowledge.
            let polynomial = crate::poly::polybridge::expr_to_poly(arena, g, var).is_some();
            (Vec::new(), polynomial)
        }
    };
    if !complete {
        scan.complete = false;
    }
    for s in solutions {
        push_point(arena, s, kind, scan);
    }
}

/// Does `g` contain `sin`, `cos` or `tan` of an argument depending on
/// `var` (whose principal roots are one period of infinitely many)?
fn periodic_in(arena: &Arena, g: ExprId, var: ExprId) -> bool {
    walk::post_order_ids(arena, g).into_iter().any(|id| {
        matches!(*arena.node(id), ExprNode::Sin(u) | ExprNode::Cos(u) | ExprNode::Tan(u)
            if walk::contains(arena, u, var))
    })
}

/// Push a single candidate point, filtering out non-real constants.
fn push_point(arena: &mut Arena, point: ExprId, kind: BreakKind, scan: &mut BreakScan) {
    if walk::free_symbols(arena, point).is_empty() {
        // Constant: keep only real, finite ones.
        match expr_to_f64(arena, point) {
            Some(v) if v.is_finite() => scan.points.push(Breakpoint {
                point,
                value: Some(v),
                kind,
            }),
            _ => {}
        }
    } else {
        // Parametric root — the caller must decide where it lies, unless
        // it is provably non-real (e.g. ±i·a for x² + a²).
        let mut cache = crate::base::assumptions::AssumptionCache::new();
        if Truth::from(cache.query(arena, point, crate::base::assumptions::Props::REAL)).is_false()
        {
            return;
        }
        // Structural non-reality: an explicit `I`, or an even root of a
        // provably negative quantity (e.g. √(−a²) from x² + a² = 0).
        // A reading for real parameters only: with a parameter that is not
        // declared real (decision D4, 0.31) `−a + i` is real where
        // `Im a = 1`, and a pole there may lie on the path.  The point used
        // to be dropped, and `∫₀¹ dx/(x + a − i)` came out finite for
        // `a = −1/2 + i`, where the integrand has a pole at `x = 1/2`.
        let declared_real = walk::free_symbols(arena, point).iter().all(|&s| {
            matches!(*arena.node(s), ExprNode::Symbol(sid)
                if crate::transforms::realness::symbol_declared_real(arena, sid))
        });
        let mut nonreal = false;
        let mut stack = if declared_real {
            vec![point]
        } else {
            Vec::new()
        };
        while let Some(id) = stack.pop() {
            match arena.node(id).clone() {
                ExprNode::ImaginaryUnit => {
                    nonreal = true;
                    break;
                }
                ExprNode::Pow(b, e) => {
                    if let Some(r) = arena.as_num(e)
                        && !r.is_integer()
                        && num_integer::Integer::is_even(r.denom())
                        && Truth::from(cache.query(
                            arena,
                            b,
                            crate::base::assumptions::Props::NEGATIVE,
                        ))
                        .is_true()
                    {
                        nonreal = true;
                        break;
                    }
                    stack.push(b);
                    stack.push(e);
                }
                node => node.for_each_child(|c| stack.push(c)),
            }
        }
        if nonreal
            && !Truth::from(cache.query(arena, point, crate::base::assumptions::Props::REAL))
                .is_true()
        {
            return;
        }
        scan.points.push(Breakpoint {
            point,
            value: None,
            kind,
        });
    }
}

/// If `g` is `sin(αx+β)`, `cos(αx+β)` or `tan(αx+β)` with constant α, β,
/// push its zeros in `range`, `x = ((k + d)·π − β)/α` (`d = ½` for `cos`),
/// exactly: `k` runs between the certified ceiling and floor of `(α·end +
/// β)/π − d` at the exact ends (see [`ScanRange`]).
fn push_trig_zeros(
    arena: &mut Arena,
    g: ExprId,
    var: ExprId,
    kind: BreakKind,
    range: Option<ScanRange>,
    scan: &mut BreakScan,
) -> TrigZeros {
    let (inner, half) = match arena.node(g).clone() {
        ExprNode::Sin(i) | ExprNode::Tan(i) => (i, false),
        ExprNode::Cos(i) => (i, true),
        _ => return TrigZeros::NotTrig,
    };
    if !walk::contains(arena, inner, var) {
        return TrigZeros::NotTrig;
    }
    let (Some(lin), Some(range)) = (linear_coeffs_exact(arena, inner, var), range) else {
        return TrigZeros::Incomplete;
    };
    // The family in the variable t = (αx + β)/π − d: the integers.
    let pi = arena.pi();
    let shift = if half {
        arena.rational(1, 2)
    } else {
        arena.zero()
    };
    let to_t = |arena: &mut Arena, u: ExprId| {
        let over_pi = arena.div(u, pi);
        arena.sub(over_pi, shift)
    };
    let Some(ks) = family_indices(arena, lin, range, to_t, scan) else {
        return TrigZeros::Incomplete;
    };
    let lin_f = linear_f64(arena, lin);
    for k in ks {
        // x = ((k + d)·π − β)/α.
        let k_d = if half {
            Ratio::new(BigInt::from(2) * &k + 1, BigInt::from(2))
        } else {
            Ratio::from_integer(k)
        };
        let t_f64 = num_traits::ToPrimitive::to_f64(&k_d).map(|t| t * std::f64::consts::PI);
        let k_ex = arena.num_ratio(k_d);
        let k_pi = arena.mul(&[k_ex, pi]);
        let num = arena.sub(k_pi, lin.intercept);
        let point = arena.div(num, lin.slope);
        let value = affine_point_f64(t_f64, lin_f).or_else(|| expr_to_f64(arena, point));
        scan.points.push(Breakpoint { point, value, kind });
    }
    TrigZeros::Pushed
}

/// What [`push_trig_zeros`] did (it was an `Option<bool>` up to 0.49, whose
/// `None` and `Some(false)` meant different things).
enum TrigZeros {
    /// `g` is not `sin`, `cos` or `tan` of an argument in the variable.
    NotTrig,
    /// The zeros in the range were pushed.
    Pushed,
    /// The family could not be enumerated (no range, more than
    /// [`MAX_FAMILY_POINTS`] members, or an integer part that does not
    /// settle): the scan is incomplete.
    Incomplete,
}

/// The integers `k` of a family `t(α·x + β) = k` (`to_t` builds `t(u)`, an
/// increasing affine map) with `x` in the closed `range`, from the certified
/// ceiling and floor of `t` at its exact ends; `None` (with `scan.crowded`
/// when that is the reason) when an integer part does not settle or there
/// are more than [`MAX_FAMILY_POINTS`].
fn family_indices(
    arena: &mut Arena,
    lin: LinearCoeffs<ExprId>,
    range: ScanRange,
    to_t: impl Fn(&mut Arena, ExprId) -> ExprId,
    scan: &mut BreakScan,
) -> Option<Vec<BigInt>> {
    if arena.is_zero_structural(lin.slope) {
        return None;
    }
    let t_at = |arena: &mut Arena, end: ExprId| {
        let ax = arena.mul(&[lin.slope, end]);
        let u = arena.add(&[ax, lin.intercept]);
        let t = to_t(arena, u);
        eval::eval(arena, t)
    };
    let ta = t_at(arena, range.exact.lower);
    let tb = t_at(arena, range.exact.upper);
    let (ca, fa) = (
        integer_part(arena, ta, true)?,
        integer_part(arena, ta, false)?,
    );
    let (cb, fb) = (
        integer_part(arena, tb, true)?,
        integer_part(arena, tb, false)?,
    );
    // Integers between t(a) and t(b), in whichever order they are.
    let (mut k, last) = (ca.min(cb), fa.max(fb));
    if &last - &k >= BigInt::from(MAX_FAMILY_POINTS) {
        scan.crowded = true;
        return None;
    }
    let mut ks = Vec::new();
    while k <= last {
        ks.push(k.clone());
        k += 1;
    }
    Some(ks)
}

/// `⌊t⌋` (`ceiling`: `⌈t⌉`) of the real constant `t`, certified (`eval`'s
/// integer part of a constant: exact, or the integer its certified value
/// lies strictly inside); `None` when it does not settle.
fn integer_part(arena: &mut Arena, t: ExprId, ceiling: bool) -> Option<BigInt> {
    let node = if ceiling {
        arena.ceiling(t)
    } else {
        arena.floor(t)
    };
    let n = eval::eval(arena, node);
    arena
        .as_num(n)
        .filter(|r| r.is_integer())
        .map(|r| r.to_integer())
}

/// The slope and intercept of `lin` as `f64`s, when both evaluate.
fn linear_f64(arena: &mut Arena, lin: LinearCoeffs<ExprId>) -> Option<LinearCoeffs<f64>> {
    // Guide values only (see [`Breakpoint::value`]).
    let slope = expr_to_f64(arena, lin.slope).filter(|v| v.is_finite() && *v != 0.0)?;
    let intercept = expr_to_f64(arena, lin.intercept).filter(|v| v.is_finite())?;
    Some(LinearCoeffs { slope, intercept })
}

/// `(t − β)/α` in `f64` for a family member `t` (`f64`, within `4·2⁻⁵³` of
/// its value), when that is within [`POINT_VALUE_ACCURACY`] of the exact
/// point: finite, nonzero, and `t − β` cancelling less than four bits.
fn affine_point_f64(t: Option<f64>, lin: Option<LinearCoeffs<f64>>) -> Option<f64> {
    let (t, lin) = (t?, lin?);
    let num = t - lin.intercept;
    let v = num / lin.slope;
    (t.is_finite()
        && num != 0.0
        && t.abs() + lin.intercept.abs() <= 16.0 * num.abs()
        && v.is_finite()
        && v != 0.0)
        .then_some(v)
}

/// The coefficients of a linear expression `slope·var + intercept`, as
/// returned by [`linear_coeffs_exact`] (exact `ExprId`s), and their values
/// ([`linear_f64`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LinearCoeffs<T> {
    /// Coefficient of `var` (`α` in `α·var + β`).
    pub slope: T,
    /// Constant term (`β` in `α·var + β`).
    pub intercept: T,
}

/// Exact `α·var + β` coefficients for `expr`, where both are free of
/// `var`.
pub(crate) fn linear_coeffs_exact(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Option<LinearCoeffs<ExprId>> {
    let coeffs = poly_coeffs_symbolic(arena, expr, var)?;
    if coeffs.len() != 2 {
        return None;
    }
    Some(LinearCoeffs {
        slope: coeffs[1],
        intercept: coeffs[0],
    })
}

/// Coefficients `[c₀, c₁, …, cₙ]` of `expr` viewed as a polynomial in
/// `var` with coefficients that may be arbitrary `var`-free expressions.
///
/// The expression is expanded first.  Returns `None` if some term is not
/// of the form `c · var^k` with `k` a non-negative integer, if the
/// polynomial is identically zero (the empty coefficient list would be
/// ambiguous), or if its degree exceeds [`MAX_DENSE_DEGREE`].
pub(crate) fn poly_coeffs_symbolic(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Option<Vec<ExprId>> {
    let expanded = crate::transforms::expand::expand(arena, expr);
    let expanded = eval::eval(arena, expanded);

    let terms: Vec<ExprId> = match arena.node(expanded).clone() {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![expanded],
    };

    let mut buckets: Vec<Vec<ExprId>> = Vec::new();
    for term in terms {
        let (k, coeff) = split_term_power(arena, term, var)?;
        if k > MAX_DENSE_DEGREE {
            return None;
        }
        while buckets.len() <= k {
            buckets.push(Vec::new());
        }
        buckets[k].push(coeff);
    }

    if buckets.is_empty() {
        return None;
    }
    let mut result = Vec::with_capacity(buckets.len());
    for bucket in buckets {
        let c = if bucket.is_empty() {
            arena.zero()
        } else {
            let s = arena.add(&bucket);
            eval::eval(arena, s)
        };
        result.push(c);
    }
    // Trim trailing zeros (keep at least one coefficient).
    while result.len() > 1 && arena.is_zero_structural(*result.last()?) {
        result.pop();
    }
    if result.len() == 1 && arena.is_zero_structural(result[0]) {
        return None;
    }
    Some(result)
}

/// The largest degree [`poly_coeffs_symbolic`] lays out densely (`2¹⁶`, as
/// `algebraic::MAX_POWER_BITS`): up to 0.56 the density of
/// `Weibull(51/8, 9·10¹⁰)` (a factor `x^(9·10¹⁰ − 1)`) asked for a
/// coefficient vector of `9·10¹⁰` entries and aborted the process.
const MAX_DENSE_DEGREE: usize = crate::base::cast::MAX_DENSE_DEGREE;

/// Split a single product term into `(k, coefficient)` such that
/// `term = coefficient · var^k` with the coefficient free of `var`.
fn split_term_power(arena: &mut Arena, term: ExprId, var: ExprId) -> Option<(usize, ExprId)> {
    if !walk::contains(arena, term, var) {
        return Some((0, term));
    }
    if term == var {
        return Some((1, arena.one()));
    }
    match arena.node(term).clone() {
        ExprNode::Pow(base, exp) if base == var => {
            let r = arena.as_num(exp)?.clone();
            if !r.is_integer() {
                return None;
            }
            use num_traits::{Signed, ToPrimitive};
            if r.is_negative() {
                return None;
            }
            let k = r.to_integer().to_usize()?;
            Some((k, arena.one()))
        }
        ExprNode::Mul(children) => {
            let mut k_total = 0usize;
            let mut coeff_parts: Vec<ExprId> = Vec::new();
            for c in children.iter() {
                if !walk::contains(arena, *c, var) {
                    coeff_parts.push(*c);
                    continue;
                }
                // Nested Mul is not canonical; only var or var^k allowed here.
                let (k, cc) = split_term_power(arena, *c, var)?;
                if !arena.is_one_structural(cc) {
                    return None;
                }
                k_total += k;
            }
            let coeff = if coeff_parts.is_empty() {
                arena.one()
            } else {
                arena.mul(&coeff_parts)
            };
            Some((k_total, coeff))
        }
        ExprNode::Neg(inner) => {
            let (k, c) = split_term_power(arena, inner, var)?;
            Some((k, arena.neg(c)))
        }
        _ => None,
    }
}

/// Collect breakpoint candidates contributed by a single node.
fn breakpoints_of_node(
    arena: &mut Arena,
    node: &ExprNode,
    var: ExprId,
    range: Option<ScanRange>,
    scan: &mut BreakScan,
) {
    use BreakKind::{Kink, Singular};
    match node {
        // Negative powers → poles at zeros of base.
        // Non-integer powers → the base must not change sign (branch
        // point); its zeros are treated as singular endpoints too.
        ExprNode::Pow(base, exp) => {
            let base = *base;
            let exp = *exp;
            if !walk::contains(arena, base, var) {
                return;
            }
            if walk::contains(arena, exp, var) {
                // f(x)^{g(x)}: only well-behaved for f > 0; we cannot
                // analyse this in general.
                scan.complete = false;
                return;
            }
            let matters = if let Some(r) = arena.as_num(exp).cloned() {
                use num_traits::Signed;
                r.is_negative() || !r.is_integer()
            } else {
                // Symbolic exponent: assume it can be negative.
                true
            };
            if matters {
                push_zeros(arena, base, var, Singular, range, scan);
            }
        }

        // ln(g) → singular where g = 0.
        ExprNode::Ln(inner) => push_zeros(arena, *inner, var, Singular, range, scan),

        // tan(g) → poles where cos(g) = 0.
        ExprNode::Tan(inner) => {
            let inner = *inner;
            if !walk::contains(arena, inner, var) {
                return;
            }
            let cos_inner = arena.cos(inner);
            push_zeros(arena, cos_inner, var, Singular, range, scan);
        }

        // Inverse trig / hyperbolic domain edges.
        ExprNode::Asin(inner) | ExprNode::Acos(inner) | ExprNode::Atanh(inner) => {
            let inner = *inner;
            if !walk::contains(arena, inner, var) {
                return;
            }
            let one = arena.one();
            let g_minus = arena.sub(inner, one);
            let g_plus = arena.add(&[inner, one]);
            push_zeros(arena, g_minus, var, Singular, range, scan);
            push_zeros(arena, g_plus, var, Singular, range, scan);
        }
        ExprNode::Acosh(inner) => {
            let inner = *inner;
            if !walk::contains(arena, inner, var) {
                return;
            }
            let one = arena.one();
            let g_minus = arena.sub(inner, one);
            push_zeros(arena, g_minus, var, Singular, range, scan);
        }

        // Γ(g), ψ(g), lnΓ(g): poles at non-positive integers.
        ExprNode::Gamma(inner) | ExprNode::Digamma(inner) | ExprNode::LogGamma(inner) => {
            let inner = *inner;
            if !walk::contains(arena, inner, var) {
                return;
            }
            push_integer_family(arena, inner, var, range, Singular, scan, |n| {
                !num_traits::Signed::is_positive(n)
            });
        }

        // Kinks: |g|, sign(g), H(g), δ(g) at zeros of g.
        ExprNode::Abs(inner)
        | ExprNode::Sign(inner)
        | ExprNode::Heaviside(inner)
        | ExprNode::DiracDelta(inner) => push_zeros(arena, *inner, var, Kink, range, scan),

        // Steps at integers.
        ExprNode::Floor(inner) | ExprNode::Ceiling(inner) => {
            let inner = *inner;
            if !walk::contains(arena, inner, var) {
                return;
            }
            // `sin u`, `cos u` reach the integers −1, 0, 1 only where `sin u`
            // or `cos u` is 0: the steps of `floor(sin(αx + β))` are among
            // the zeros of both (up to 0.49 they were not located).
            if let ExprNode::Sin(u) | ExprNode::Cos(u) = *arena.node(inner) {
                let (s, c) = (arena.sin(u), arena.cos(u));
                push_zeros(arena, s, var, Kink, range, scan);
                push_zeros(arena, c, var, Kink, range, scan);
                return;
            }
            push_integer_family(arena, inner, var, range, Kink, scan, |_| true);
        }

        // Relational conditions (from Piecewise): boundaries where a = b.
        ExprNode::Gt(a, b) | ExprNode::Ge(a, b) | ExprNode::Eq_(a, b) | ExprNode::Ne(a, b) => {
            let d = arena.sub(*a, *b);
            push_zeros(arena, d, var, Kink, range, scan);
        }

        // min/max: kinks where two arguments cross.  Up to 0.49 none was
        // pushed, and the quadrature of `Max(x − 1/3, 0)` was not split.
        ExprNode::Min(args) | ExprNode::Max(args)
            if args.iter().any(|&a| walk::contains(arena, a, var)) =>
        {
            let args = args.to_vec();
            for (i, &p) in args.iter().enumerate() {
                for &q in &args[i + 1..] {
                    let d = arena.sub(p, q);
                    push_zeros(arena, d, var, Kink, range, scan);
                }
            }
        }

        // Unknown functions and formal nodes: we cannot see inside.
        ExprNode::Apply(_, args) | ExprNode::Lib(_, args) | ExprNode::Internal(_, args)
            if args.iter().any(|&a| walk::contains(arena, a, var)) =>
        {
            scan.complete = false;
            scan.opaque = true;
        }
        ExprNode::Integral(body, _)
        | ExprNode::Derivative(body, _)
        | ExprNode::Limit(body, _, _)
        | ExprNode::Sum(body, _, _, _)
        | ExprNode::Product_(body, _, _, _)
        | ExprNode::Residue(body, _, _)
            if walk::contains(arena, *body, var) =>
        {
            scan.complete = false;
            scan.opaque = true;
        }
        ExprNode::RootOf(..)
        | ExprNode::RootSum(..)
        | ExprNode::LambertW(_)
        | ExprNode::Subs(..)
            if node
                .children()
                .iter()
                .any(|&c| walk::contains(arena, c, var)) =>
        {
            scan.complete = false;
            scan.opaque = true;
        }

        _ => {}
    }
}

/// Push the points where the linear expression `inner = αx+β` takes the
/// integer values selected by `keep`, restricted to `range`: `n` between
/// the certified ceiling and floor of `α·end + β` at the exact ends, at
/// most [`MAX_FAMILY_POINTS`] of them.  Before 0.49 `as i64` casts of the
/// `f64` ends saturated and the `± 1` and the length wrapped: the jumps of
/// `floor(10¹⁹x)` on `[−1, 1]` were skipped (integral 0) and
/// `floor(9·10¹⁸x)` looped over 1.8·10¹⁹ candidates; up to 0.49 the ends
/// were `f64`s and the steps of `floor(x)` beyond `2⁵³` were not found.
fn push_integer_family(
    arena: &mut Arena,
    inner: ExprId,
    var: ExprId,
    range: Option<ScanRange>,
    kind: BreakKind,
    scan: &mut BreakScan,
    keep: impl Fn(&BigInt) -> bool,
) {
    let (Some(lin), Some(range)) = (linear_coeffs_exact(arena, inner, var), range) else {
        scan.complete = false;
        return;
    };
    let Some(ns) = family_indices(arena, lin, range, |_, u| u, scan) else {
        scan.complete = false;
        return;
    };
    let lin_f = linear_f64(arena, lin);
    for n in ns {
        if !keep(&n) {
            continue;
        }
        let n_f64 = num_traits::ToPrimitive::to_f64(&n);
        let n_ex = arena.num_ratio(Ratio::from_integer(n));
        let num = arena.sub(n_ex, lin.intercept);
        let point = arena.div(num, lin.slope);
        let point = eval::eval(arena, point);
        let value = affine_point_f64(n_f64, lin_f).or_else(|| expr_to_f64(arena, point));
        scan.points.push(Breakpoint { point, value, kind });
    }
}

/// The value of the real constant `expr` as an `f64` (correctly rounded
/// from its 16-digit value, [`evalf::const_value`]); `None` when it does
/// not evaluate or is not real.
///
/// A constant whose value keeps an imaginary part is refused, however
/// small the part: `evalf` keeps a part next to a much larger one only when
/// its error ball excludes 0, so that part is certified nonzero.  Up to
/// 0.52 `1 + i·e⁻⁵⁰` was the real point `1` (the printed decimal omitted
/// the part): the pole of `1/(x − 1 − i·e⁻⁵⁰)` lay on `[0, 2]`, and the
/// integral, `iπ` to 21 digits, was `Divergent`; up to 0.57 the decimal was
/// printed, searched for `i` and parsed, and the value evaluated a second
/// time for the part.  An imaginary part that is only 0 to the precision
/// reached is not decided here (callers that need realness proved use
/// [`constant_realness_as_declared`](crate::transforms::realness::constant_realness_as_declared)).
pub(crate) fn expr_to_f64(arena: &mut Arena, expr: ExprId) -> Option<f64> {
    let evaled = eval::eval(arena, expr);
    // Try exact rational first
    if let Some(r) = arena.as_num(evaled).cloned() {
        use num_traits::ToPrimitive;
        return r.to_f64();
    }
    // Fall back to numerical evaluation
    evalf::const_value(arena, evaled, 16).ok()?.to_f64()
}

// ═══════════════════════════════════════════════════════════════════════════
// estimate_frequency
// ═══════════════════════════════════════════════════════════════════════════

/// Walk the expression tree to estimate the maximum oscillation frequency.
///
/// Looks for `Sin(inner)`, `Cos(inner)`, `Tan(inner)` where `inner`
/// is linear in `var`. Extracts the coefficient as the angular
/// frequency ω (rad/s) and returns `ω / (2π)` in Hz, or the raw ω
/// depending on convention. Returns the maximum angular frequency (rad/s)
/// found, or `None` if no trig terms are present.
pub(crate) fn estimate_frequency(
    arena: &Arena,
    expr: ExprId,
    var: ExprId,
    _var_sym: SymbolId,
) -> Option<f64> {
    tracing::debug!("estimate_frequency: scanning for trig terms");

    let post_order = walk::post_order_ids(arena, expr);
    let mut max_omega: Option<f64> = None;

    for &id in &post_order {
        let node = arena.node(id);
        let inner = match node {
            ExprNode::Sin(i) => Some(*i),
            ExprNode::Cos(i) => Some(*i),
            ExprNode::Tan(i) => Some(*i),
            _ => None,
        };

        if let Some(inner) = inner
            && let Some(omega) = extract_linear_coefficient(arena, inner, var)
        {
            let omega_abs = omega.abs();
            match max_omega {
                Some(cur) if cur >= omega_abs => {}
                _ => max_omega = Some(omega_abs),
            }
        }
    }

    max_omega
}

/// If `expr` is of the form `a*var + b` (linear in `var`), extract the
/// coefficient `a` as an f64.  Returns `None` if the expression is not
/// linear in `var` or the coefficient cannot be evaluated numerically.
fn extract_linear_coefficient(arena: &Arena, expr: ExprId, var: ExprId) -> Option<f64> {
    // Case 1: expr IS var → coefficient is 1
    if expr == var {
        return Some(1.0);
    }

    let node = arena.node(expr);
    match node {
        // a * x  or  a * x + b  (inside an Add)
        ExprNode::Mul(args) => {
            // Look for var among the factors; the rest is the coefficient
            let mut has_var = false;
            let mut coeff_ids: Vec<ExprId> = Vec::new();
            for &arg in args.iter() {
                if arg == var {
                    has_var = true;
                } else if walk::contains(arena, arg, var) {
                    // Non-linear in var
                    return None;
                } else {
                    coeff_ids.push(arg);
                }
            }
            if !has_var {
                return None;
            }
            if coeff_ids.is_empty() {
                return Some(1.0);
            }
            // Evaluate the remaining coefficient numerically
            // We need to read the numeric value directly
            if coeff_ids.len() == 1 {
                return num_value(arena, coeff_ids[0]);
            }
            None
        }

        ExprNode::Add(args) => {
            // Linear form: a*x + b.  Find the term containing var.
            let mut omega = None;
            for &arg in args.iter() {
                if walk::contains(arena, arg, var) {
                    // This term should be linear in var
                    omega = extract_linear_coefficient(arena, arg, var);
                }
            }
            omega
        }

        ExprNode::Neg(inner) => extract_linear_coefficient(arena, *inner, var).map(|c| -c),

        _ => None,
    }
}

/// Try to read a constant expression as an f64 from the arena (exact rational only).
fn num_value(arena: &Arena, expr: ExprId) -> Option<f64> {
    if let Some(r) = arena.as_num(expr) {
        use num_traits::ToPrimitive;
        r.to_f64()
    } else {
        match arena.node(expr) {
            ExprNode::Neg(inner) => num_value(arena, *inner).map(|v| -v),
            _ => None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Function analysis (0.9): continuity scan and periodicity
//
// Arena-level helpers behind `Ex::singularities`, `Ex::maximum`,
// `Ex::function_range` and `Ex::periodicity`
// (`src/api/expr_calculus_util_ext.rs`).
// ═══════════════════════════════════════════════════════════════════════════

/// What a structural walk of an expression says about its continuity in
/// one variable.
#[derive(Clone, Debug, Default)]
pub(crate) struct ContinuityScan {
    /// Sub-expressions `g` such that the expression is undefined wherever
    /// `g = 0` (SymPy's `singularities` rule set): bases of negative
    /// powers, arguments of `ln`, `cos(g)` for `tan(g)`, and `g ∓ 1` for
    /// `atanh(g)`.  Deduplicated; only sources depending on the variable.
    pub singular: Vec<ExprId>,
    /// Sub-expressions `g` such that the expression is continuous but not
    /// differentiable where `g = 0` (`|g|`).
    pub kinks: Vec<ExprId>,
    /// Name of the first node kind that makes the expression discontinuous
    /// (`floor`, `sign`, `Piecewise`, …), has poles the scan does not
    /// enumerate (`Gamma`, `zeta`, …), or cannot be looked into at all
    /// (unknown functions, formal integrals / limits, `RootOf`).
    pub opaque: Option<&'static str>,
}

/// Scan `expr` for the singular / kink sources and opaque nodes of
/// [`ContinuityScan`], with respect to `var`.
///
/// Explicit post-order walk (no recursion).  Only nodes that depend on
/// `var` contribute.
pub(crate) fn continuity_scan(arena: &mut Arena, expr: ExprId, var: ExprId) -> ContinuityScan {
    let mut scan = ContinuityScan::default();
    let mut assumptions = crate::base::assumptions::AssumptionCache::new();

    for id in walk::post_order_ids(arena, expr) {
        let node = arena.node(id).clone();
        let depends = |arena: &Arena, child: ExprId| walk::contains(arena, child, var);
        match node {
            ExprNode::Pow(base, exp) => {
                if !depends(arena, base) || depends(arena, exp) {
                    continue;
                }
                let negative = match arena.as_num(exp) {
                    Some(r) => {
                        use num_traits::Signed;
                        r.is_negative()
                    }
                    None => Truth::from(assumptions.query(
                        arena,
                        exp,
                        crate::base::assumptions::Props::NEGATIVE,
                    ))
                    .is_true(),
                };
                if negative {
                    push_unique(&mut scan.singular, base);
                }
            }
            ExprNode::Ln(g) if depends(arena, g) => push_unique(&mut scan.singular, g),
            ExprNode::Tan(g) if depends(arena, g) => {
                let cos_g = arena.cos(g);
                push_unique(&mut scan.singular, cos_g);
            }
            ExprNode::Atanh(g) if depends(arena, g) => {
                let one = arena.one();
                let minus = arena.sub(g, one);
                let plus = arena.add(&[g, one]);
                push_unique(&mut scan.singular, minus);
                push_unique(&mut scan.singular, plus);
            }
            ExprNode::Abs(g) if depends(arena, g) => push_unique(&mut scan.kinks, g),
            _ => {
                if scan.opaque.is_none()
                    && let Some(name) = opaque_node_name(&node)
                    && node.children().iter().any(|&c| depends(arena, c))
                {
                    scan.opaque = Some(name);
                }
            }
        }
    }
    scan
}

fn push_unique(list: &mut Vec<ExprId>, id: ExprId) {
    if !list.contains(&id) {
        list.push(id);
    }
}

/// Node kinds that the extremum / range analysis refuses to look through:
/// discontinuous functions, functions with poles outside the
/// [`ContinuityScan::singular`] rule set, and formal or unknown nodes.
fn opaque_node_name(node: &ExprNode) -> Option<&'static str> {
    Some(match node {
        ExprNode::Floor(_) => "floor",
        ExprNode::Ceiling(_) => "ceiling",
        ExprNode::Sign(_) => "sign",
        ExprNode::Heaviside(_) => "Heaviside",
        ExprNode::DiracDelta(_) => "DiracDelta",
        ExprNode::Piecewise(_) => "Piecewise",
        ExprNode::Min(_) => "min",
        ExprNode::Max(_) => "max",
        ExprNode::Re(_) => "re",
        ExprNode::Im(_) => "im",
        ExprNode::Conjugate(_) => "conjugate",
        ExprNode::Arg(_) => "arg",
        ExprNode::Atan2(..) => "atan2",
        ExprNode::Gamma(_) => "Gamma",
        ExprNode::LogGamma(_) => "loggamma",
        ExprNode::Digamma(_) => "digamma",
        ExprNode::Polygamma(..) => "polygamma",
        ExprNode::Zeta(_) => "zeta",
        ExprNode::Beta(..) => "Beta",
        ExprNode::Factorial(_) => "factorial",
        ExprNode::Binomial(..) => "binomial",
        ExprNode::KroneckerDelta(..) => "KroneckerDelta",
        ExprNode::LambertW(_) => "LambertW",
        ExprNode::Apply(..) | ExprNode::Lib(..) | ExprNode::Internal(..) => "an unknown function",
        ExprNode::Derivative(..) => "Derivative",
        ExprNode::Integral(..) => "Integral",
        ExprNode::DefiniteIntegral(..) => "DefiniteIntegral",
        ExprNode::Sum(..) => "Sum",
        ExprNode::Product_(..) => "Product",
        ExprNode::Limit(..) => "Limit",
        ExprNode::Series(..) => "Series",
        ExprNode::LaplaceTransform(..) => "LaplaceTransform",
        ExprNode::InverseLaplaceTransform(..) => "InverseLaplaceTransform",
        ExprNode::Residue(..) => "Residue",
        ExprNode::RootOf(..) => "RootOf",
        ExprNode::RootSum(..) => "RootSum",
        ExprNode::DSolve(..) => "DSolve",
        ExprNode::Subs(..) => "Subs",
        ExprNode::Gt(..)
        | ExprNode::Ge(..)
        | ExprNode::Eq_(..)
        | ExprNode::Ne(..)
        | ExprNode::And(_)
        | ExprNode::Or(_)
        | ExprNode::Not(_) => "a boolean",
        ExprNode::ConditionSet(..)
        | ExprNode::Interval(..)
        | ExprNode::FiniteSet(_)
        | ExprNode::SetUnion(_)
        | ExprNode::SetIntersection(_)
        | ExprNode::SetComplement(..)
        | ExprNode::ImageSet(..) => "a set",
        _ => return None,
    })
}

/// The distinct `sign(h)` nodes of `expr` whose argument `h` depends on
/// `var`, as `(sign(h), h)` pairs in post-order.  These are the factors
/// the derivative of `|h|` introduces; the extremum analysis resolves
/// each to `±1` on the regions where `h` has a fixed sign.
pub(crate) fn sign_nodes_of(arena: &Arena, expr: ExprId, var: ExprId) -> Vec<(ExprId, ExprId)> {
    let mut out: Vec<(ExprId, ExprId)> = Vec::new();
    for id in walk::post_order_ids(arena, expr) {
        if let ExprNode::Sign(h) = arena.node(id)
            && walk::contains(arena, *h, var)
            && !out.iter().any(|(s, _)| *s == id)
        {
            out.push((id, *h));
        }
    }
    out
}

/// Does `expr` contain `sin`, `cos` or `tan` of something depending on
/// `var`?  Decides whether the zeros of `expr` may form periodic
/// families (so `solve_general` rather than `solve` should be used).
pub(crate) fn has_trig_of(arena: &Arena, expr: ExprId, var: ExprId) -> bool {
    walk::post_order_ids(arena, expr)
        .into_iter()
        .any(|id| match arena.node(id) {
            ExprNode::Sin(g) | ExprNode::Cos(g) | ExprNode::Tan(g) => {
                walk::contains(arena, *g, var)
            }
            _ => false,
        })
}

// ── periodicity ────────────────────────────────────────────────────────────

/// A period of `expr` in `var` (SymPy's `periodicity`) — not necessarily
/// the fundamental one: composite expressions get the lcm of the periods
/// of their pieces, and identities that shorten the period are not
/// detected (`sin²x·cos²x` → `π`, although the fundamental period is
/// `π/2`).
///
/// * `Some(0)` when `expr` does not depend on `var`;
/// * `Some(p)` for `sin`/`cos`/`tan` of a linear argument `a·var + b`
///   (`2π/|a|`, `2π/|a|`, `π/|a|`), for sums, products, powers and
///   compositions of periodic pieces (lcm of the periods, which requires
///   pairwise rational ratios), with the half-period refinement for
///   products `sin(g)ᵖ·cos(g)ᵠ` whose exponent sum `p + q` is even
///   (`sin·cos`, `cos/sin`, `sin²` → `π/|a|`) and for `|sin g|`, `|cos g|`;
/// * `None` when some piece depending on `var` is not recognised as
///   periodic (a bare `var`, `sin(x²)`, `sin(√2·x) + sin(x)`, formal
///   nodes, …).
///
/// Explicit post-order walk with a per-node memo; no recursion.
pub(crate) fn periodicity(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<ExprId> {
    if !walk::contains(arena, expr, var) {
        return Some(arena.zero());
    }
    let mut memo: rustc_hash::FxHashMap<ExprId, Option<ExprId>> = rustc_hash::FxHashMap::default();
    for id in walk::post_order_ids(arena, expr) {
        if memo.contains_key(&id) {
            continue;
        }
        let period = if walk::contains(arena, id, var) {
            node_period(arena, id, var, &memo)
        } else {
            Some(arena.zero())
        };
        memo.insert(id, period);
    }
    memo.get(&expr).copied().flatten()
}

/// Period of one node (which depends on `var`) from the periods of its
/// children.
fn node_period(
    arena: &mut Arena,
    id: ExprId,
    var: ExprId,
    memo: &rustc_hash::FxHashMap<ExprId, Option<ExprId>>,
) -> Option<ExprId> {
    let node = arena.node(id).clone();
    match &node {
        // The variable itself is not periodic.
        ExprNode::Symbol(_) => None,

        ExprNode::Sin(g) | ExprNode::Cos(g) | ExprNode::Tan(g) => {
            if let Some(p) = trig_power_period(arena, id, var) {
                return Some(p);
            }
            // Composition `sin(h(x))` with `h` periodic.
            memo.get(g).copied().flatten()
        }

        // |sin g| and |cos g| have half the period of sin g / cos g.
        ExprNode::Abs(g) => {
            if let ExprNode::Sin(h) | ExprNode::Cos(h) = arena.node(*g).clone()
                && let Some(lin) = linear_coeffs_exact(arena, h, var)
            {
                return Some(pi_over_abs(arena, lin.slope, 1));
            }
            memo.get(g).copied().flatten()
        }

        ExprNode::Pow(_, exp) => {
            if !walk::contains(arena, *exp, var)
                && let Some(p) = trig_power_period(arena, id, var)
            {
                return Some(p);
            }
            lcm_of_children(arena, &node, memo)
        }

        ExprNode::Mul(children) => {
            let children = children.clone();
            mul_period(arena, &children, var, memo)
        }

        // Binders, formal nodes, piecewise, booleans and sets: not analysed.
        ExprNode::Integral(..)
        | ExprNode::DefiniteIntegral(..)
        | ExprNode::Sum(..)
        | ExprNode::Product_(..)
        | ExprNode::Limit(..)
        | ExprNode::Series(..)
        | ExprNode::LaplaceTransform(..)
        | ExprNode::InverseLaplaceTransform(..)
        | ExprNode::Residue(..)
        | ExprNode::RootOf(..)
        | ExprNode::RootSum(..)
        | ExprNode::DSolve(..)
        | ExprNode::Piecewise(_)
        | ExprNode::Gt(..)
        | ExprNode::Ge(..)
        | ExprNode::Eq_(..)
        | ExprNode::Ne(..)
        | ExprNode::And(_)
        | ExprNode::Or(_)
        | ExprNode::Not(_)
        | ExprNode::ConditionSet(..)
        | ExprNode::Interval(..)
        | ExprNode::FiniteSet(_)
        | ExprNode::SetUnion(_)
        | ExprNode::SetIntersection(_)
        | ExprNode::SetComplement(..)
        | ExprNode::ImageSet(..) => None,

        // Any other function of periodic arguments is periodic with the
        // lcm of their periods (`Add`, `Neg`, `exp`, `ln`, `sinh`, …).
        _ => lcm_of_children(arena, &node, memo),
    }
}

/// lcm of the children's periods; `None` if any child is not periodic.
fn lcm_of_children(
    arena: &mut Arena,
    node: &ExprNode,
    memo: &rustc_hash::FxHashMap<ExprId, Option<ExprId>>,
) -> Option<ExprId> {
    let mut acc = arena.zero();
    for c in node.children() {
        let p = memo.get(&c).copied().flatten()?;
        acc = lcm_periods(arena, acc, p)?;
    }
    Some(acc)
}

/// Which trigonometric function a factor is built from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TrigKind {
    SinCos,
    Tan,
}

/// View `factor` as `trig(g)^k` with an integer `k` (`k = 1` for a bare
/// `sin`/`cos`/`tan`).
fn as_trig_power(arena: &Arena, factor: ExprId) -> Option<(ExprId, TrigKind, i64)> {
    let (base, k) = match arena.node(factor) {
        ExprNode::Pow(base, exp) => {
            let r = arena.as_num(*exp)?;
            if !r.is_integer() {
                return None;
            }
            use num_traits::ToPrimitive;
            (*base, r.to_integer().to_i64()?)
        }
        _ => (factor, 1),
    };
    match arena.node(base) {
        ExprNode::Sin(g) | ExprNode::Cos(g) => Some((*g, TrigKind::SinCos, k)),
        ExprNode::Tan(g) => Some((*g, TrigKind::Tan, k)),
        _ => None,
    }
}

/// Period of a single `trig(a·var + b)^k` factor: `π/|a|` for `tan` and
/// for even powers of `sin`/`cos`, `2π/|a|` otherwise.  `None` if the
/// factor is not of that shape.
fn trig_power_period(arena: &mut Arena, factor: ExprId, var: ExprId) -> Option<ExprId> {
    let (g, kind, k) = as_trig_power(arena, factor)?;
    let a = linear_coeffs_exact(arena, g, var)?.slope;
    let halves = kind == TrigKind::Tan || k % 2 == 0;
    Some(pi_over_abs(arena, a, if halves { 1 } else { 2 }))
}

/// Period of a product: trigonometric factors sharing the same linear
/// argument are grouped so that `sin(g)ᵖ·cos(g)ᵠ·tan(g)ʳ` gets period
/// `π/|a|` when `p + q` is even (both `sin` and `cos` flip sign under a
/// half-period shift, `tan` does not); everything else contributes its own
/// period.  The result is the lcm of all contributions.
fn mul_period(
    arena: &mut Arena,
    children: &[ExprId],
    var: ExprId,
    memo: &rustc_hash::FxHashMap<ExprId, Option<ExprId>>,
) -> Option<ExprId> {
    // (inner argument g, coefficient a, exponent sum of sin/cos factors)
    let mut groups: Vec<(ExprId, ExprId, i64)> = Vec::new();
    let mut acc = arena.zero();
    for &c in children {
        if !walk::contains(arena, c, var) {
            continue;
        }
        if let Some((g, kind, k)) = as_trig_power(arena, c)
            && let Some(LinearCoeffs { slope: a, .. }) = linear_coeffs_exact(arena, g, var)
        {
            let weight = if kind == TrigKind::Tan { 0 } else { k };
            match groups.iter_mut().find(|(gg, _, _)| *gg == g) {
                Some(entry) => entry.2 += weight,
                None => groups.push((g, a, weight)),
            }
            continue;
        }
        let p = memo.get(&c).copied().flatten()?;
        acc = lcm_periods(arena, acc, p)?;
    }
    for (_, a, sum) in groups {
        let p = pi_over_abs(arena, a, if sum % 2 == 0 { 1 } else { 2 });
        acc = lcm_periods(arena, acc, p)?;
    }
    Some(acc)
}

/// `k·π/|a|`, evaluated.  `|a|` is resolved through the assumption system
/// when the sign of `a` is known (`|π| = π`, `|−3| = 3`).
fn pi_over_abs(arena: &mut Arena, a: ExprId, k: i64) -> ExprId {
    let k = arena.int(k);
    let pi = arena.pi();
    let num = arena.mul(&[k, pi]);
    let mut assumptions = crate::base::assumptions::AssumptionCache::new();
    let den = if Truth::from(assumptions.query(arena, a, crate::base::assumptions::Props::POSITIVE))
        .is_true()
    {
        a
    } else if Truth::from(assumptions.query(arena, a, crate::base::assumptions::Props::NEGATIVE))
        .is_true()
    {
        arena.neg(a)
    } else {
        arena.abs(a)
    };
    let q = arena.div(num, den);
    eval::eval(arena, q)
}

/// Least common multiple of two periods (`0` is the identity).  Requires
/// the ratio `a/b` to evaluate to a rational `m/n`; then `lcm = n·a`.
/// `None` when the ratio is irrational or cannot be decided.
fn lcm_periods(arena: &mut Arena, a: ExprId, b: ExprId) -> Option<ExprId> {
    if arena.is_zero_structural(a) {
        return Some(b);
    }
    if arena.is_zero_structural(b) || a == b {
        return Some(a);
    }
    let ratio = arena.div(a, b);
    let ratio = eval::eval(arena, ratio);
    let r = arena.as_num(ratio)?.clone();
    use num_traits::Signed;
    if !r.is_positive() {
        return None;
    }
    let n = arena.big_int(r.denom().clone());
    let l = arena.mul(&[a, n]);
    Some(eval::eval(arena, l))
}

// ═══════════════════════════════════════════════════════════════════════════
// Zero constants that canonicalisation does not see
// ═══════════════════════════════════════════════════════════════════════════

/// Digits at which [`hidden_zero`] evaluates a constant.
const HIDDEN_ZERO_DIGITS: u32 = 30;

/// Largest constant (in nodes) the exact tests of [`hidden_zero`] rewrite.
const MAX_EXACT_ZERO_SIZE: usize = 400;

/// Largest constant (in nodes) [`zero_by_identity`] simplifies.
const MAX_IDENTITY_SIZE: usize = 150;

/// Is the evaluated, symbol-free constant `c` zero — also when canonical
/// forms do not cancel it: two spellings of one number (`asinh 2` and
/// `ln(2 + √5)`, side by side once one of them has been rewritten or
/// expanded), `sin²1 + cos²1 − 1`, `asin(sin 1) − 1`?  And on what
/// evidence?
///
/// - A number, or a product of powers of the named constants (`π`, `e`,
///   `i`, …) and numbers, is decided by its canonical form (`Exact`): such
///   a product is 0 only when canonical forms make it 0.
/// - Otherwise the value is evaluated with certified digits: a certified
///   digit is `False`, an exact 0 `True` (`Certified`).  A function value
///   can vanish without a sum in sight (`sin(π·ln 8/ln 2)`).
/// - A value zero to the cap of the digits must be shown to be 0 exactly
///   ([`exactly_zero`]: minimal polynomials, exact rewrites, rational
///   functions of the generators) — `True`, `Exact`.  Failing that, the
///   deep zero search of `evalf` (about 1,024 bits more) certifies a tiny
///   value hidden by cancellation (`e^{−10⁻¹⁵⁰} − 1`): `False`,
///   `Certified`; then the costlier identities of
///   [`vanishes_by_identity`](crate::simplify::identically_zero::vanishes_by_identity)
///   are tried.
/// - A value still zero there is `True` on `Approximate` evidence only,
///   which is no proof.  A constant with free symbols or an unevaluated
///   part, or one `evalf` cannot evaluate (`legendre(3/2, e)`), is
///   `Unknown`: there is no evidence either way (see
///   [`ZeroTest::Nonzero`] for how the series engine reads that).  So is
///   one without a zero question, as for
///   [`Ex::is_zero`](crate::api::expr::Ex::is_zero): with `±∞`, `zoo` or
///   NaN in it, and one with a `δ` that no identity shows to be 0 (`δ` of
///   an argument not proved real has no value; it is never taken for 0).
///
/// Series coefficients and Gruntz's leading coefficients are decided this
/// way, and both readings of an unproved zero are claims: as "nonzero" it
/// makes a vanishing leading coefficient a leading term (`lim_{x→0}
/// x/(asinh(x + 2) − asinh 2)`, which is `√5`, was `0`, as was the series
/// of `x/(asinh(x + 2) − ln(2 + √5))`: `x/0`); as "zero", up to 0.48, a
/// value below about `10⁻⁸⁰` relative to its terms counted as 0: `c =
/// e^{−10⁻¹⁵⁰} − 1` (`−10⁻¹⁵⁰`) made `lim_{x→∞} c·x` equal `0` (it is
/// `−∞`).  Read the verdict with
/// [`Verdict::proven`](crate::base::truth::Verdict::proven); a caller
/// that meets `Unknown` tries its next method or refuses.
///
/// The verdict is memoised per arena
/// ([`FactKey::HiddenZero`](crate::base::arena::FactKey::HiddenZero)):
/// the series engine settles the same constant in every expansion that
/// meets it, and a constant that is 0 but not proved costs the whole
/// ladder above each time.  Up to 0.49 `lim_{x→∞} sin(asin(x + π/3)) +
/// cos(acos(x + π/3))` re-tested its undecided constants at every
/// occurrence before it refused (1.9 s; 1.6 s with the memo), and 142
/// limits and series of the golden corpus ran past 5 s (120 with it).
/// A memoised proof comes back as `Certified` (whether it was
/// exact is not kept; both are proofs, and callers read them alike); a
/// test without any evidence (a value `evalf` cannot compute) is not
/// memoised.  The ladder reads no work budget, so the verdict depends
/// only on the node and the arena's configuration, like the other
/// entries of [`Arena::constant_facts`](crate::base::arena::Arena).
pub(crate) fn hidden_zero(
    arena: &mut Arena,
    c: ExprId,
) -> crate::base::truth::Verdict<crate::base::truth::Truth> {
    use crate::base::arena::FactKey;
    use crate::base::truth::{Evidence, SignSet, Truth, Verdict};
    if let Some(r) = arena.as_num(c) {
        return Verdict::new(
            Truth::from_bool(num_traits::Zero::is_zero(r)),
            Evidence::Exact,
        );
    }
    // Only symbol-free constants without unevaluated parts are memoised.
    if !walk::free_symbols(arena, c).is_empty() || walk::has_unevaluated(arena, c) {
        return Verdict::new(Truth::Unknown, Evidence::Approximate);
    }
    // Past the allowance of an [`EvaluationScope`]: undecided, as a value 0
    // to the precision reached (never "no value", which reads as nonzero),
    // and not memoised.  Charged before the memo is read, so that the
    // outcome does not depend on what the arena has seen.
    if !EvaluationScope::charge(arena, c) {
        return Verdict::new(Truth::True, Evidence::Approximate);
    }
    if let Some(signs) = arena.constant_fact(c, FactKey::HiddenZero) {
        return match signs {
            SignSet::ZERO => Verdict::new(Truth::True, Evidence::Certified),
            SignSet::NONZERO => Verdict::new(Truth::False, Evidence::Certified),
            _ => Verdict::new(Truth::True, Evidence::Approximate),
        };
    }
    let verdict = hidden_zero_uncached(arena, c);
    let signs = match (verdict.proven(), verdict.value) {
        (Some(Truth::True), _) => Some(SignSet::ZERO),
        (Some(Truth::False), _) => Some(SignSet::NONZERO),
        (None, Truth::True) => Some(SignSet::UNKNOWN),
        (Some(Truth::Unknown), _) | (None, Truth::False | Truth::Unknown) => None,
    };
    if let Some(signs) = signs {
        arena.remember_constant_fact(c, FactKey::HiddenZero, signs);
    }
    verdict
}

thread_local! {
    /// The units of evaluation work charged by [`hidden_zero`] on this
    /// thread while an [`EvaluationScope`] is open, and the count at which
    /// the innermost scope stops them.
    static EVALUATIONS: std::cell::Cell<(u64, Option<u64>)> =
        const { std::cell::Cell::new((0, None)) };
}

/// Distinct nodes of a constant per unit of evaluation work
/// ([`EvaluationScope`]).
const NODES_PER_EVALUATION_UNIT: usize = 64;

/// A bound on the work of [`hidden_zero`] (each call a certified `evalf` of
/// a constant, or its memo: the costly step of the series engine's zero
/// tests) while the scope lives, in units of [`NODES_PER_EVALUATION_UNIT`]
/// distinct nodes of the constants tested (a smaller constant is free):
/// the tighter of `cap` and what an enclosing scope has left.  Every call
/// is charged, also one the memo answers, so the outcome does not depend on
/// what the arena has seen.  Past the bound `hidden_zero` answers
/// "undecided" without evaluating, so the caller refuses, and the scope
/// stays spent.  Gruntz's [`Budget`](crate::calculus::gruntz) holds one and
/// charges the units to its work at each step; up to 0.58 they were not
/// counted, and `limit(E1(digamma(x))^Ei(x), x, 0, −)` evaluated
/// coefficients of its series (polynomials in `ζ(k)` and `γ` of a thousand
/// distinct nodes, each `ζ` at 166 bits) for over 25 s.  Scopes nest
/// (dropped in the reverse order of entering, as locals are).
#[derive(Debug)]
pub(crate) struct EvaluationScope {
    /// The stop count of the enclosing scope, restored on drop.
    outer: Option<u64>,
    /// The count when the scope was entered.
    start: u64,
}

impl EvaluationScope {
    /// Open a scope allowing `cap` more units.
    pub(crate) fn enter(cap: u64) -> Self {
        let (made, outer) = EVALUATIONS.get();
        let end = made.saturating_add(cap);
        let end = outer.map_or(end, |o| o.min(end));
        EVALUATIONS.set((made, Some(end)));
        EvaluationScope { outer, start: made }
    }

    /// The units charged since the scope was entered (by it and by the
    /// scopes inside it).
    pub(crate) fn made(&self) -> u64 {
        EVALUATIONS.get().0.saturating_sub(self.start)
    }

    /// Charge the evaluation of the constant `c`; `false` when the
    /// innermost scope has no room for it (it is then spent).  Without an
    /// open scope nothing is counted.
    fn charge(arena: &Arena, c: ExprId) -> bool {
        let (made, end) = EVALUATIONS.get();
        let Some(end) = end else {
            return true;
        };
        let room = end.saturating_sub(made);
        // `evalf` evaluates each distinct node once: the cost is the size of
        // the DAG, not of the tree (which a shared sub-expression can make
        // exponentially larger).
        let size = walk::post_order_ids(arena, c).len();
        let units = (size / NODES_PER_EVALUATION_UNIT) as u64;
        if units > room {
            EVALUATIONS.set((end, Some(end)));
            return false;
        }
        EVALUATIONS.set((made + units, Some(end)));
        true
    }
}

impl Drop for EvaluationScope {
    fn drop(&mut self) {
        let (made, _) = EVALUATIONS.get();
        EVALUATIONS.set((made, self.outer));
    }
}

/// [`hidden_zero`] of a symbol-free constant without unevaluated parts,
/// not memoised.
fn hidden_zero_uncached(
    arena: &mut Arena,
    c: ExprId,
) -> crate::base::truth::Verdict<crate::base::truth::Truth> {
    use crate::base::truth::{Evidence, Truth, Verdict};
    // No zero question to answer, as for `Ex::is_zero`: up to 0.49 the
    // structural shortcut below called `zoo` and NaN nonzero.
    if without_zero_question(arena, c) {
        return Verdict::new(Truth::Unknown, Evidence::Approximate);
    }
    // `δ(a)` of a nonzero real `a` is 0 ([`dirac_delta_off_zero`]); `δ(0)`
    // has no value, nor has `δ` off the reals (`evalf` took `δ(i)` for 0 up
    // to 0.50; up to 0.49 the rewrites here did too, and `δ(i) + 1` was nonzero on
    // exact evidence): no claim unless an identity shows `c` to be 0
    // whatever the value of `δ`.
    if has_dirac_delta(arena, c) {
        let rewritten = exact_rewrites(arena, c);
        if exactly_zero(arena, c, rewritten) {
            return Verdict::new(Truth::True, Evidence::Exact);
        }
        return Verdict::new(Truth::Unknown, Evidence::Approximate);
    }
    if walk::post_order_ids(arena, c).into_iter().all(|id| {
        let node = arena.node(id);
        node.is_atom()
            || matches!(
                node,
                ExprNode::Mul(_) | ExprNode::Pow(..) | ExprNode::Neg(_)
            )
    }) {
        return Verdict::new(Truth::False, Evidence::Exact);
    }
    let certified = |z: &crate::base::bigcomplex::Complex| {
        Verdict::new(
            Truth::from_bool(z.0.is_zero() && z.1.is_zero()),
            Evidence::Certified,
        )
    };
    let cap = evalf::evalf_verdict(arena, c, HIDDEN_ZERO_DIGITS, evalf::ZeroSearch::Cap);
    match cap.map(crate::base::truth::Verdict::proven) {
        Ok(Some(z)) => return certified(&z),
        Ok(None) => {}
        Err(_) => return Verdict::new(Truth::Unknown, Evidence::Approximate),
    }
    let rewritten = exact_rewrites(arena, c);
    if exactly_zero(arena, c, rewritten) {
        return Verdict::new(Truth::True, Evidence::Exact);
    }
    if let Some(z) = evalf::evalf_verdict(arena, c, HIDDEN_ZERO_DIGITS, evalf::ZeroSearch::Deep)
        .ok()
        .and_then(crate::base::truth::Verdict::proven)
    {
        return certified(&z);
    }
    if zero_by_identity(arena, c, rewritten) {
        return Verdict::new(Truth::True, Evidence::Exact);
    }
    tracing::debug!(
        c = %arena.display(c),
        "hidden_zero: zero to the precision reached, not proved"
    );
    Verdict::new(Truth::True, Evidence::Approximate)
}

/// Is the constant `c` (`rewritten` by [`exact_rewrites`]) exactly 0, by a
/// cheap exact method: the minimal polynomial of an algebraic number
/// ([`exact_is_zero`](crate::poly::algebraic::exact_is_zero)), or a
/// rational function of its generators that vanishes
/// ([`vanishes_as_rational_function`](crate::simplify::ratsimp::vanishes_as_rational_function):
/// a polynomial in `π` multiplied out), before and after the rewrites, and
/// the minimal polynomial of `c` multiplied out.
/// With free parameters, `true` means 0 for all their values.  Only a
/// constant of at most [`MAX_EXACT_ZERO_SIZE`] nodes is tried.
pub(crate) fn exactly_zero(arena: &mut Arena, c: ExprId, rewritten: ExprId) -> bool {
    use crate::poly::algebraic::exact_is_zero;
    if arena.is_zero_structural(rewritten) {
        return true;
    }
    if walk::post_order_ids(arena, c).len() > MAX_EXACT_ZERO_SIZE {
        return false;
    }
    let constant = walk::free_symbols(arena, c).is_empty();
    if constant && Truth::from(exact_is_zero(arena, c)).is_true() {
        return true;
    }
    if crate::simplify::ratsimp::vanishes_as_rational_function(arena, c) {
        return true;
    }
    // Multiplied out, the transcendental parts may cancel and leave an
    // algebraic number (`(a − ζ(3))·i + (b + ζ(3))·i`, `a + b = 0`).
    let expanded = crate::transforms::expand::expand(arena, c);
    if arena.is_zero_structural(expanded)
        || (constant && expanded != c && Truth::from(exact_is_zero(arena, expanded)).is_true())
    {
        return true;
    }
    if rewritten == c {
        return false;
    }
    (constant && Truth::from(exact_is_zero(arena, rewritten)).is_true())
        || crate::simplify::ratsimp::vanishes_as_rational_function(arena, rewritten)
}

/// The costlier exact test of a constant that [`exactly_zero`] did not
/// decide: the identities of
/// [`vanishes_by_identity`](crate::simplify::identically_zero::vanishes_by_identity)
/// (`sin²1 + cos²1 − 1` as a rational function of `e^i`, then `simplify`),
/// of `c` and of its `rewritten` form, for a constant of at most
/// [`MAX_IDENTITY_SIZE`] nodes.
pub(crate) fn zero_by_identity(arena: &mut Arena, c: ExprId, rewritten: ExprId) -> bool {
    if walk::post_order_ids(arena, c).len() > MAX_IDENTITY_SIZE {
        return false;
    }
    crate::simplify::identically_zero::vanishes_by_identity(arena, c)
        || (rewritten != c
            && crate::simplify::identically_zero::vanishes_by_identity(arena, rewritten))
}

/// Largest prime factor [`ln_of_rational_in_primes`] divides out; a larger
/// cofactor stays one logarithm.
const MAX_LOG_PRIME: u64 = 1 << 20;

/// `c` rewritten by identities that let canonical forms and the rational
/// arithmetic of the generators see a zero, each exact where `c` is
/// defined:
///
/// - `f(f⁻¹(z))` and its relatives, algebraic in `z` (`cos(asin z) =
///   √(1 − z²)`, [`function_of_inverse`]);
/// - `f⁻¹(f(a)) = a` for a real constant `a` in the principal range of
///   `f⁻¹` (certified): `asin(sin 1) = 1` ([`inverse_of_value`]);
///   `acosh(a) = i·acos(a)` for a real `a` in `(−1, 1)`
///   ([`acosh_as_acos`]) and `asin(a) = π/2 − i·acosh(a)` for `a > 1`
///   ([`asin_on_cut`]);
/// - the inverse trigonometric and hyperbolic functions as logarithms of
///   their principal values on all of ℂ
///   ([`inverse_functions_as_logs`](crate::calculus::limit::inverse_functions_as_logs)),
///   the spelling of Gruntz's tractable rewrite: `asinh 2 − ln(2 + √5)`
///   becomes `0`;
/// - the logarithms and radicals of constants on an axis by their
///   principal values, `ln(i·(√3 − 2)) = ln(2 − √3) − iπ/2`, `√(1 − π²/4)
///   = i·√(π²/4 − 1)`, `√(e^{2i}) = e^i` ([`values_on_axes`]), so the
///   logarithmic forms of inverse functions on their cuts read as the
///   principal values: `acos(−2) = π + i·ln(2 − √3)`;
/// - a sum of logarithms of positive reals with one cofactor whose
///   arguments multiply to 1 dropped, `ln(2 + √3) + ln(2 − √3) = 0`
///   ([`logs_combined`]): `atan(z) + atan(1/z)` for `z = acos 2 = i·ln(2 +
///   √3)` is `π/2`; and `atan(z) + atan(1/z) = ±π/2` for `Re z ≷ 0`
///   ([`reciprocal_atans_paired`]);
/// - the logarithm of a positive rational as a sum of logarithms of its
///   prime factors, `ln 16 = 4·ln 2`, `ln(3/2) = ln 3 − ln 2`;
/// - the square root of a polynomial with its nonnegative square factors
///   taken out, `√(a² + 2a + 1) = a + 1` for `a > 0` ([`sqrt_of_squares`]);
/// - two radicals or two values of one function whose arguments are equal
///   by an identity made one node ([`like_arguments_unified`]);
/// - `Γ(p)·Γ(1 − p) = π/sin(πp)` ([`gamma_reflection`]), `exp(a)·exp(b) =
///   exp(a + b)` ([`exponentials_combined`]), Bessel functions of integer
///   order and `Eₙ` of order `n ≤ 0` in a normal form
///   ([`special_function_rewrite`]), `atan2(0, a)` for a real `a ≠ 0`,
///   `δ(a) = 0` for `a ≠ 0` ([`dirac_delta_off_zero`]), a binomial
///   coefficient that evaluates to exactly 0 ([`certified_zero_value`]),
///   `min`/`max` of real constants of certified order
///   ([`extreme_constant`]);
/// - a function argument with a quotient in it reduced as a rational
///   function of its generators ([`ratsimp`](crate::simplify::ratsimp::ratsimp)):
///   `cos((π − 12π²)/(24π − 2)) = cos(−π/2) = 0`.
pub(crate) fn exact_rewrites(arena: &mut Arena, c: ExprId) -> ExprId {
    if walk::post_order_ids(arena, c).len() > MAX_EXACT_ZERO_SIZE {
        return c;
    }
    let r = compositions_pass(arena, c);
    let r = exact_pass(arena, r);
    let r = reciprocal_atans_paired(arena, r);
    let logs = crate::calculus::limit::inverse_functions_as_logs(arena, r);
    let logs = values_on_axes(arena, logs);
    let r = if logs == r {
        r
    } else {
        exact_pass(arena, logs)
    };
    let r = logs_combined(arena, r);
    like_arguments_unified(arena, r)
}

/// The sum `c` with each pair of terms `q·K·atan(z) + q·K·atan(w)` (one
/// rational `q`, one cofactor `K`, `z·w = 1` exactly; or `q·K·atan(z) −
/// q·K·atan(w)` with `z·w = −1`) replaced by `±q·K·π/2` for a constant
/// `z` whose real part has a certified sign: the principal
/// values satisfy `atan z + atan(1/z) = π/2` for `Re z > 0` and `−π/2` for
/// `Re z < 0` (`atan(2 + i) + atan((2 − i)/5) = π/2`, mpmath
/// `1.5707963267949`).  On the imaginary axis the sum depends on `|z|` and
/// is left to the logarithmic forms ([`logs_combined`]).  `c` unchanged
/// otherwise.
fn reciprocal_atans_paired(arena: &mut Arena, c: ExprId) -> ExprId {
    use crate::base::truth::SignSet;
    if !arena.is_symbol_free(c) {
        return c;
    }
    let ExprNode::Add(terms) = arena.node(c).clone() else {
        return c;
    };
    // (index, q, K, z) of each term `q·K·atan(z)`.
    let mut atans: Vec<(usize, crate::base::numeric::Q, ExprId, ExprId)> = Vec::new();
    for (k, &t) in terms.iter().enumerate() {
        let (q, rest) = arena.as_coeff_term(t);
        let (z, cofactor) = match arena.node(rest).clone() {
            ExprNode::Atan(z) => (z, arena.one()),
            ExprNode::Mul(fs) => {
                let mut found = fs
                    .iter()
                    .filter(|&&f| matches!(arena.node(f), ExprNode::Atan(_)));
                let (Some(&a), None) = (found.next(), found.next()) else {
                    continue;
                };
                let ExprNode::Atan(z) = *arena.node(a) else {
                    continue;
                };
                let others: Vec<ExprId> = fs.iter().copied().filter(|&f| f != a).collect();
                (z, arena.mul(&others))
            }
            _ => continue,
        };
        atans.push((k, q, cofactor, z));
    }
    let mut replaced: Vec<(usize, usize, ExprId)> = Vec::new();
    let used = |replaced: &[(usize, usize, ExprId)], k: usize| {
        replaced.iter().any(|&(i, j, _)| i == k || j == k)
    };
    for (a, (i, qi, ki, zi)) in atans.iter().enumerate() {
        for (j, qj, kj, zj) in &atans[a + 1..] {
            // `q·atan z − q·atan w` with `w = −1/z` is the pair too (`atan` is
            // odd, and canonical forms take the sign out of `atan(−u)`).
            let product = if qi == qj {
                arena.one()
            } else if *qi == -qj.clone() {
                arena.neg_one()
            } else {
                continue;
            };
            if ki != kj || used(&replaced, *i) || used(&replaced, *j) {
                continue;
            }
            let p = arena.mul(&[*zi, *zj]);
            let p = arena.sub(p, product);
            let p = eval::eval(arena, p);
            if !(arena.is_zero_structural(p)
                || Truth::from(crate::poly::algebraic::exact_is_zero(arena, p)).is_true()
                || crate::simplify::ratsimp::vanishes_as_rational_function(arena, p))
            {
                continue;
            }
            let re = arena.re(*zi);
            let re = eval::eval(arena, re);
            let half = match real_constant_sign(arena, re) {
                SignSet::POSITIVE => arena.rational(1, 2),
                SignSet::NEGATIVE => arena.rational(-1, 2),
                _ => continue,
            };
            let q = arena.num_ratio(qi.clone());
            let pi = arena.pi();
            let value = arena.mul(&[q, *ki, half, pi]);
            replaced.push((*i, *j, value));
        }
    }
    if replaced.is_empty() {
        return c;
    }
    let mut kept: Vec<ExprId> = terms
        .iter()
        .enumerate()
        .filter(|&(k, _)| !used(&replaced, k))
        .map(|(_, &t)| t)
        .collect();
    kept.extend(replaced.iter().map(|&(_, _, v)| v));
    let s = arena.add(&kept);
    eval::eval(arena, s)
}

/// Most terms [`distributed`] multiplies a product out into.
const MAX_DISTRIBUTED_TERMS: usize = 16;

/// `c` with the logarithms and powers of constants whose value lies on an
/// axis written by their principal values, bottom-up (an explicit
/// post-order walk), then `eval`:
///
/// - `ln w = ln(−w) + iπ` for a negative real `w`, `ln(i·b) = ln b + iπ/2`
///   and `ln(−i·b) = ln b − iπ/2` for a positive real `b`, and `ln(e^z) =
///   z` for `|Im z| < π` ([`ln_on_axis`]);
/// - `w^q = (−1)^q·(−w)^q` for a negative real `w` and a rational `q` that
///   is not an integer (`√(1 − π²/4) = i·√(π²/4 − 1)`), and `(e^z)^q =
///   e^{qz}` for `|Im z| < π` (`√(e^{2i}) = e^i`) ([`power_on_axis`]);
/// - `√a·√b = √(ab)` for positive reals `a`, `b` whose product multiplies
///   out to a sum ([`positive_roots_merged`]);
/// - a product multiplied out over its factors that are sums
///   ([`distributed`]): canonical products distribute only a number, and
///   `−i·(ln b + iπ/2)` (the form `eval` gives the logarithm of `acos 2`)
///   must read `π/2 − i·ln b` for the nodes above it to see `i·ln b`.
///
/// The signs are certified ([`real_constant_sign`]) and each rewrite is an
/// identity of the principal values where both sides are finite; a result
/// more than four times [`MAX_EXACT_ZERO_SIZE`] is dropped.  So the
/// logarithmic form of `acos(−2)`
/// ([`inverse_functions_as_logs`](crate::calculus::limit::inverse_functions_as_logs))
/// reads `π + i·ln(2 − √3)`, and that of `asinh(iπ/2)` reads `ln(π/2 +
/// √(π²/4 − 1)) + iπ/2`: up to 0.51 the limits at the cuts whose
/// coefficients compare such values with their spellings by Gruntz's
/// rewrites were refused (`lim_{x→0⁺} x/(acos(−2 + ix) − acos(−2))`, which
/// is `−√3`; 0.48 answered them, on a zero test that proved nothing).
fn values_on_axes(arena: &mut Arena, c: ExprId) -> ExprId {
    if !arena.is_symbol_free(c) {
        return c;
    }
    let post = walk::post_order_ids(arena, c);
    let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    let mut changed = false;
    for &id in &post {
        let r = walk::rebuild_with_cache(arena, id, &cache);
        let r = match arena.node(r).clone() {
            ExprNode::Mul(fs) => {
                let r = positive_roots_merged(arena, &fs).unwrap_or(r);
                match arena.node(r).clone() {
                    ExprNode::Mul(fs) => distributed(arena, &fs).unwrap_or(r),
                    _ => r,
                }
            }
            _ => r,
        };
        let new = match *arena.node(r) {
            ExprNode::Ln(w) => ln_on_axis(arena, w).unwrap_or(r),
            ExprNode::Pow(b, q) => power_on_axis(arena, b, q).unwrap_or(r),
            _ => r,
        };
        changed |= new != id;
        cache.insert(id, new);
    }
    if !changed {
        return c;
    }
    let r = cache.get(&c).copied().unwrap_or(c);
    let r = eval::eval(arena, r);
    let cap = 4 * MAX_EXACT_ZERO_SIZE;
    if crate::transforms::pattern::tree_size_capped(arena, r, cap + 1) > cap {
        return c;
    }
    r
}

/// `POSITIVE` or `NEGATIVE` for a symbol-free constant `v` proved real
/// with a certified sign ([`eval::constant_facts`]: realness by the
/// assumptions or an exact imaginary part 0), `UNKNOWN` otherwise (a value
/// that is not real, 0, or not decided).
fn real_constant_sign(arena: &mut Arena, v: ExprId) -> crate::base::truth::SignSet {
    use crate::base::assumptions::Props;
    use crate::base::truth::SignSet;
    if !arena.is_symbol_free(v) || walk::has_unevaluated(arena, v) {
        return SignSet::UNKNOWN;
    }
    let facts = eval::constant_facts(arena, v);
    if Truth::from(facts.query(Props::POSITIVE)).is_true() {
        SignSet::POSITIVE
    } else if Truth::from(facts.query(Props::NEGATIVE)).is_true() {
        SignSet::NEGATIVE
    } else {
        SignSet::UNKNOWN
    }
}

/// `z` for the exponential `e^z` (`exp(z)` or `E^z`).
fn exponential_argument(arena: &Arena, b: ExprId) -> Option<ExprId> {
    match *arena.node(b) {
        ExprNode::Exp(z) => Some(z),
        ExprNode::Pow(e, z) if e == arena.e_const => Some(z),
        _ => None,
    }
}

/// Largest `|k|` of a shift `2πik` [`principal_exponent`] makes.
const MAX_EXPONENT_TURNS: f64 = 1e6;

/// The principal logarithm `ln(e^z) = z − 2πik` of `e^z` for a constant
/// `z`: `k` the integer that puts the imaginary part inside `(−π, π)`,
/// certified with a margin (`z` itself for `|Im z| < π`, `4i − 2πi` for
/// `4i`).  `None` when the imaginary part is not certified away from the
/// odd multiples of `π`.
fn principal_exponent(arena: &mut Arena, z: ExprId) -> Option<ExprId> {
    use std::f64::consts::PI;
    let im = arena.im(z);
    let im = eval::eval(arena, im);
    let k = (certified_real_value(arena, im)? / (2.0 * PI)).round();
    if k.abs() > MAX_EXPONENT_TURNS {
        return None;
    }
    let k = crate::base::cast::f64_to_i64(k)?;
    if k == 0 {
        return real_constant_within(arena, im, -PI, PI).then_some(z);
    }
    let pi = arena.pi();
    let two_k = arena.int(2 * k);
    let turns = arena.mul(&[two_k, pi]);
    let shifted_im = arena.sub(im, turns);
    let shifted_im = eval::eval(arena, shifted_im);
    if !real_constant_within(arena, shifted_im, -PI, PI) {
        return None;
    }
    let i = arena.i_unit();
    let shift = arena.mul(&[i, turns]);
    let shifted = arena.sub(z, shift);
    Some(eval::eval(arena, shifted))
}

/// `ln w` of a constant `w` on an axis by its principal value (`Arg w ∈
/// (−π, π]`): `ln(−w) + iπ` for a negative real `w`, `ln b ± iπ/2` for `w
/// = ±i·b` with a positive real `b`, and `z − 2πik` for `w = e^z`
/// ([`principal_exponent`]).  `None` for a positive real `w` (nothing to
/// do) and any other.
fn ln_on_axis(arena: &mut Arena, w: ExprId) -> Option<ExprId> {
    use crate::base::truth::SignSet;
    if let Some(z) = exponential_argument(arena, w) {
        return principal_exponent(arena, z);
    }
    let i = arena.i_unit();
    let (b, angle) = match real_constant_sign(arena, w) {
        SignSet::POSITIVE => return None,
        SignSet::NEGATIVE => (arena.neg(w), arena.one()),
        _ => {
            let ni = arena.neg(i);
            let b = match distributed(arena, &[ni, w]) {
                Some(b) => b,
                None => arena.mul(&[ni, w]),
            };
            match real_constant_sign(arena, b) {
                SignSet::POSITIVE => (b, arena.rational(1, 2)),
                SignSet::NEGATIVE => (arena.neg(b), arena.rational(-1, 2)),
                _ => return None,
            }
        }
    };
    let l = arena.ln(b);
    let pi = arena.pi();
    let phase = arena.mul(&[angle, i, pi]);
    Some(arena.add(&[l, phase]))
}

/// `b^q` for a rational `q` that is not an integer and a constant base on
/// an axis, by its principal value `e^{q·ln b}`: `(−1)^q·(−b)^q` for a
/// negative real `b`, `e^{q·(z − 2πik)}` for `b = e^z`
/// ([`principal_exponent`]).  `None` for a numeric base (canonical forms
/// write those) and any other.
fn power_on_axis(arena: &mut Arena, b: ExprId, q: ExprId) -> Option<ExprId> {
    use crate::base::truth::SignSet;
    if arena.as_num(q).is_none_or(|r| r.is_integer()) || arena.as_num(b).is_some() {
        return None;
    }
    if let Some(z) = exponential_argument(arena, b) {
        let z = principal_exponent(arena, z)?;
        let qz = arena.mul(&[q, z]);
        return Some(arena.exp(qz));
    }
    if real_constant_sign(arena, b) != SignSet::NEGATIVE {
        return None;
    }
    let m1 = arena.neg_one();
    let unit = arena.pow(m1, q);
    let nb = arena.neg(b);
    let root = arena.pow(nb, q);
    Some(arena.mul(&[unit, root]))
}

/// The product of the factors `fs` with its square roots `√a` of positive
/// real constants (certified) merged into one, `√a·√b = √(ab)`, when the
/// product of the radicands multiplies out to a sum: `√(π/2 − 1)·√(π/2 +
/// 1) = √(π²/4 − 1)`.  (A product of radicands that stays a product is
/// split again by [`power_of_positive_product`].)  `None` otherwise.
fn positive_roots_merged(arena: &mut Arena, fs: &[ExprId]) -> Option<ExprId> {
    use crate::base::truth::SignSet;
    let half = arena.rational(1, 2);
    let mut radicands: Vec<ExprId> = Vec::new();
    let mut rest: Vec<ExprId> = Vec::new();
    for &f in fs {
        let radicand = match *arena.node(f) {
            ExprNode::Pow(b, e) if e == half && arena.as_num(b).is_none() => Some(b),
            _ => None,
        };
        match radicand {
            Some(b) if real_constant_sign(arena, b) == SignSet::POSITIVE => {
                radicands.push(b);
            }
            _ => rest.push(f),
        }
    }
    if radicands.len() < 2 {
        return None;
    }
    let product = arena.mul(&radicands);
    let product = crate::transforms::expand::expand(arena, product);
    if !matches!(arena.node(product), ExprNode::Add(_)) {
        return None;
    }
    rest.push(arena.sqrt(product));
    Some(arena.mul(&rest))
}

/// The product of the factors `fs` multiplied out over the factors that
/// are sums (one level, at most [`MAX_DISTRIBUTED_TERMS`] terms):
/// `−i·(ln b + iπ/2) = π/2 − i·ln b`.  `None` without a sum factor or
/// past the bound.
fn distributed(arena: &mut Arena, fs: &[ExprId]) -> Option<ExprId> {
    if !fs
        .iter()
        .any(|&f| matches!(arena.node(f), ExprNode::Add(_)))
    {
        return None;
    }
    let mut products: Vec<Vec<ExprId>> = vec![Vec::new()];
    for &f in fs {
        match arena.node(f).clone() {
            ExprNode::Add(ts) => {
                if products.len() * ts.len() > MAX_DISTRIBUTED_TERMS {
                    return None;
                }
                products = products
                    .iter()
                    .flat_map(|p| {
                        ts.iter().map(move |&t| {
                            let mut q = p.clone();
                            q.push(t);
                            q
                        })
                    })
                    .collect();
            }
            _ => products.iter_mut().for_each(|p| p.push(f)),
        }
    }
    let terms: Vec<ExprId> = products.iter().map(|p| arena.mul(p)).collect();
    Some(arena.add(&terms))
}

/// Most logarithms [`logs_combined`] merges in one group.
const MAX_COMBINED_LOGS: usize = 8;

/// Largest exponent `|eⱼ|` of a factor of the product [`logs_combined`]
/// forms.
const MAX_LOG_EXPONENT: i64 = 6;

/// The sum `c` with each group of its terms `qⱼ·K·ln aⱼ` — rational `qⱼ`,
/// one cofactor `K`, positive real constants `aⱼ` (certified) — dropped
/// when `Π aⱼ^{eⱼ} = 1` for the integers `eⱼ = d·qⱼ` (`d` the common
/// denominator): the logarithm of a positive real is additive, so the
/// group is `(K/d)·ln 1 = 0`.  The product is shown to be 1 exactly, by
/// the minimal polynomial of an algebraic number
/// ([`exact_is_zero`](crate::poly::algebraic::exact_is_zero)) or as a
/// rational function of its generators
/// ([`vanishes_as_rational_function`](crate::simplify::ratsimp::vanishes_as_rational_function)):
/// `ln(2 + √3) + ln(2 − √3)`, and `ln(1 + L) − ln(L − 1) + ln(1 − 1/L) −
/// ln(1 + 1/L)` with `L = ln(2 + √3)` (from `atan(z) + atan(1/z)`, `z =
/// i·L`).  `c` unchanged otherwise.
fn logs_combined(arena: &mut Arena, c: ExprId) -> ExprId {
    use crate::base::truth::SignSet;
    use num_traits::{One, ToPrimitive};
    if !arena.is_symbol_free(c) || !matches!(arena.node(c), ExprNode::Add(_)) {
        return c;
    }
    // The terms multiplied out (`−½·(ln a − ln b)·i`, as `eval` leaves the
    // logarithmic form of `atan`).
    let c = values_on_axes(arena, c);
    let ExprNode::Add(terms) = arena.node(c).clone() else {
        return c;
    };
    // (cofactor K, [(qⱼ, aⱼ)], [index of the term])
    type Group = (ExprId, Vec<(crate::base::numeric::Q, ExprId)>, Vec<usize>);
    let mut groups: Vec<Group> = Vec::new();
    for (k, &t) in terms.iter().enumerate() {
        let (q, rest) = arena.as_coeff_term(t);
        let (a, cofactor) = match arena.node(rest).clone() {
            ExprNode::Ln(a) => (a, arena.one()),
            ExprNode::Mul(fs) => {
                let mut logs = fs
                    .iter()
                    .filter(|&&f| matches!(arena.node(f), ExprNode::Ln(_)));
                let (Some(&l), None) = (logs.next(), logs.next()) else {
                    continue;
                };
                let ExprNode::Ln(a) = *arena.node(l) else {
                    continue;
                };
                let others: Vec<ExprId> = fs.iter().copied().filter(|&f| f != l).collect();
                (a, arena.mul(&others))
            }
            _ => continue,
        };
        if real_constant_sign(arena, a) != SignSet::POSITIVE {
            continue;
        }
        match groups.iter_mut().find(|g| g.0 == cofactor) {
            Some(g) if g.1.len() < MAX_COMBINED_LOGS => {
                g.1.push((q, a));
                g.2.push(k);
            }
            Some(_) => {}
            None => groups.push((cofactor, vec![(q, a)], vec![k])),
        }
    }
    let mut dropped: Vec<usize> = Vec::new();
    for (_, logs, indices) in groups.into_iter().filter(|g| g.1.len() >= 2) {
        let d = logs.iter().fold(num_bigint::BigInt::one(), |d, (q, _)| {
            num_integer::Integer::lcm(&d, q.denom())
        });
        let mut factors: Vec<ExprId> = Vec::with_capacity(logs.len());
        for (q, a) in &logs {
            let e = (q * crate::base::numeric::Q::from_integer(d.clone()))
                .to_integer()
                .to_i64()
                .filter(|e| e.abs() <= MAX_LOG_EXPONENT);
            let Some(e) = e else {
                factors.clear();
                break;
            };
            let e = arena.int(e);
            factors.push(arena.pow(*a, e));
        }
        if factors.is_empty() {
            continue;
        }
        let product = arena.mul(&factors);
        let one = arena.one();
        let p1 = arena.sub(product, one);
        let p1 = eval::eval(arena, p1);
        if arena.is_zero_structural(p1)
            || Truth::from(crate::poly::algebraic::exact_is_zero(arena, p1)).is_true()
            || crate::simplify::ratsimp::vanishes_as_rational_function(arena, p1)
        {
            dropped.extend(indices);
        }
    }
    if dropped.is_empty() {
        return c;
    }
    let kept: Vec<ExprId> = terms
        .iter()
        .enumerate()
        .filter(|(k, _)| !dropped.contains(k))
        .map(|(_, &t)| t)
        .collect();
    let s = arena.add(&kept);
    eval::eval(arena, s)
}

/// Most nodes [`like_arguments_unified`] compares in one group.
const MAX_LIKE_NODES: usize = 12;

/// `c` with like nodes — radicals `b^q` of one exponent, applications of
/// one function of one argument — whose arguments are equal made one node:
/// the difference of two arguments is shown to be 0 by
/// [`vanishes_by_identity`](crate::simplify::identically_zero::vanishes_by_identity)
/// (a certified sample, then an exact confirmation: as rational functions
/// of exponentials for `tanh`, `sin`, …).  So `√((e^{−2/3} − 1)/(e^{−2/3}
/// + 1)) − √(−tanh(1/3))`, the constant term Gruntz's exponential form
/// of `√(tanh x)` leaves beside the original at `x = −1/3`, is 0.
fn like_arguments_unified(arena: &mut Arena, c: ExprId) -> ExprId {
    use rustc_hash::FxHashSet;
    // The kind of a like node: its variant, and the exponent of a radical.
    type Kind = (std::mem::Discriminant<ExprNode>, Option<ExprId>);
    // (node, argument, kind) for every radical and unary function.
    let mut like: Vec<(ExprId, ExprId, Kind)> = Vec::new();
    for id in walk::post_order_ids(arena, c) {
        let node = arena.node(id);
        let kind = std::mem::discriminant(node);
        let entry = match *node {
            ExprNode::Pow(b, q)
                if arena.as_num(q).is_some_and(|q| !q.is_integer())
                    && !matches!(arena.node(b), ExprNode::Num(_)) =>
            {
                Some((b, (kind, Some(q))))
            }
            // The elementary functions (`Func::elementary`).
            _ => match node.func(arena) {
                Some((f, args)) if f.elementary().is_some() => {
                    args.first().map(|&a| (a, (kind, None)))
                }
                _ => None,
            },
        };
        if let Some((arg, kind)) = entry {
            like.push((id, arg, kind));
        }
    }
    let mut out = c;
    let mut replaced: FxHashSet<ExprId> = FxHashSet::default();
    for i in 0..like.len() {
        let (node_i, arg_i, kind_i) = like[i];
        if replaced.contains(&node_i) {
            continue;
        }
        let mut compared = 0;
        for &(node_j, arg_j, kind_j) in &like[i + 1..] {
            if kind_j != kind_i || replaced.contains(&node_j) || arg_i == arg_j {
                continue;
            }
            compared += 1;
            if compared > MAX_LIKE_NODES {
                break;
            }
            let d = arena.sub(arg_i, arg_j);
            let d = eval::eval(arena, d);
            if arena.is_zero_structural(d)
                || crate::simplify::identically_zero::vanishes_by_identity(arena, d)
            {
                out = crate::transforms::subs::subs(arena, out, node_j, node_i);
                replaced.insert(node_j);
            }
        }
    }
    if out == c { c } else { eval::eval(arena, out) }
}

/// The compositions of [`function_of_inverse`] rewritten, bottom-up (before
/// [`exact_pass`] rewrites the inner function), then `eval`.
fn compositions_pass(arena: &mut Arena, c: ExprId) -> ExprId {
    let post = walk::post_order_ids(arena, c);
    let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    let mut changed = false;
    for &id in &post {
        let r = walk::rebuild_with_cache(arena, id, &cache);
        let node = arena.node(r).clone();
        let new = match &node {
            ExprNode::Sin(a)
            | ExprNode::Cos(a)
            | ExprNode::Tan(a)
            | ExprNode::Sinh(a)
            | ExprNode::Cosh(a)
            | ExprNode::Tanh(a) => function_of_inverse(arena, &node, *a).unwrap_or(r),
            _ => r,
        };
        changed |= new != r;
        cache.insert(id, new);
    }
    if !changed {
        return c;
    }
    let r = cache.get(&c).copied().unwrap_or(c);
    eval::eval(arena, r)
}

/// One bottom-up pass of the rewrites of [`exact_rewrites`] other than the
/// compositions and the logarithmic forms, then `eval`.  An explicit
/// post-order walk.
fn exact_pass(arena: &mut Arena, c: ExprId) -> ExprId {
    use rustc_hash::{FxHashMap, FxHashSet};
    let post = walk::post_order_ids(arena, c);
    let mut arguments: FxHashSet<ExprId> = FxHashSet::default();
    for &id in &post {
        let node = arena.node(id);
        if !node.is_atom()
            && !matches!(
                node,
                ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Pow(..) | ExprNode::Neg(_)
            )
        {
            node.for_each_child(|ch| {
                arguments.insert(ch);
            });
        }
    }
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    let mut changed = false;
    for &id in &post {
        let r = walk::rebuild_with_cache(arena, id, &cache);
        let node = arena.node(r).clone();
        let mut new = match &node {
            ExprNode::Ln(a) => ln_of_rational_in_primes(arena, *a).unwrap_or(r),
            ExprNode::Lib(f, args) if args.len() == 2 => {
                special_function_rewrite(arena, *f, args[0], args[1]).unwrap_or(r)
            }
            ExprNode::Atan2(y, a) if arena.is_zero_structural(*y) => {
                if real_constant_within(arena, *a, 0.0, f64::INFINITY) {
                    arena.zero
                } else if real_constant_within(arena, *a, f64::NEG_INFINITY, 0.0) {
                    arena.pi()
                } else {
                    r
                }
            }
            ExprNode::DiracDelta(a) => dirac_delta_off_zero(arena, *a).unwrap_or(r),
            ExprNode::Abs(a) => {
                if real_constant_within(arena, *a, 0.0, f64::INFINITY) {
                    *a
                } else if real_constant_within(arena, *a, f64::NEG_INFINITY, 0.0) {
                    arena.neg(*a)
                } else {
                    r
                }
            }
            ExprNode::Pow(m, q)
                if matches!(arena.node(*m), ExprNode::Mul(_)) && arena.as_num(*q).is_some() =>
            {
                power_of_positive_product(arena, *m, *q).unwrap_or(r)
            }
            ExprNode::Binomial(..) => certified_zero_value(arena, r).unwrap_or(r),
            ExprNode::Min(args) => extreme_constant(arena, args, true).unwrap_or(r),
            ExprNode::Max(args) => extreme_constant(arena, args, false).unwrap_or(r),
            ExprNode::Acosh(a) => inverse_of_value(arena, &node, *a)
                .or_else(|| acosh_as_acos(arena, *a))
                .unwrap_or(r),
            ExprNode::Mul(fs) => gamma_reflection(arena, fs)
                .or_else(|| exponentials_combined(arena, fs))
                .unwrap_or(r),
            ExprNode::Pow(b, e)
                if matches!(arena.node(*b), ExprNode::Add(_))
                    && arena.as_num(*e).is_some_and(|q| {
                        *q == crate::base::numeric::Q::new(1.into(), 2.into())
                    }) =>
            {
                sqrt_of_squares(arena, *b).unwrap_or(r)
            }
            ExprNode::Asin(a) => inverse_of_value(arena, &node, *a)
                .or_else(|| asin_on_cut(arena, *a))
                .unwrap_or(r),
            ExprNode::Acos(a) | ExprNode::Atan(a) | ExprNode::Asinh(a) | ExprNode::Atanh(a) => {
                inverse_of_value(arena, &node, *a).unwrap_or(r)
            }
            _ => r,
        };
        if arguments.contains(&id) && has_quotient(arena, new) {
            new = crate::simplify::ratsimp::ratsimp(arena, new);
        }
        changed |= new != r;
        cache.insert(id, new);
    }
    if !changed {
        return c;
    }
    let r = cache.get(&c).copied().unwrap_or(c);
    eval::eval(arena, r)
}

/// Is the symbol-free constant `c` one without a zero question (the
/// conventions of [`Ex::is_zero`](crate::api::expr::Ex::is_zero)): `±∞`,
/// `zoo` or NaN in it (NaN is not "nonzero"), or a value proved infinite?
fn without_zero_question(arena: &mut Arena, c: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    let singular = walk::post_order_ids(arena, c).into_iter().any(|id| {
        matches!(
            arena.node(id),
            ExprNode::Infinity | ExprNode::NegInfinity | ExprNode::ComplexInfinity | ExprNode::NaN
        )
    });
    singular || Truth::from(AssumptionCache::new().query(arena, c, Props::FINITE)).is_false()
}

/// Does the symbol-free constant `c` contain a `δ` whose argument is not
/// proved real (`δ(i)`: no value, so no zero or sign of `c` either)?
pub(crate) fn dirac_delta_off_reals(arena: &mut Arena, c: ExprId) -> bool {
    walk::post_order_ids(arena, c).into_iter().any(|id| {
        let ExprNode::DiracDelta(a) = *arena.node(id) else {
            return false;
        };
        !dirac_delta_argument_real(arena, a)
    })
}

/// Is the symbol-free argument `a` of a `δ` proved real?
fn dirac_delta_argument_real(arena: &mut Arena, a: ExprId) -> bool {
    Truth::from(
        crate::transforms::eval::constant_facts(arena, a)
            .query(crate::base::assumptions::Props::REAL),
    )
    .is_true()
}

/// Does `c` contain `δ`?
fn has_dirac_delta(arena: &Arena, c: ExprId) -> bool {
    walk::post_order_ids(arena, c)
        .into_iter()
        .any(|id| matches!(arena.node(id), ExprNode::DiracDelta(_)))
}

/// `0` for `δ(a)` when `a` is not 0: a real constant with a certified
/// nonzero digit (`δ(i)` has no value; up to 0.49 it was 0 here), or an
/// expression in parameters that is nonzero at a sample point (`δ(y) = 0`
/// for generic `y`, the generic-parameter reading of the limit and series
/// engines).  `None` otherwise.
fn dirac_delta_off_zero(arena: &mut Arena, a: ExprId) -> Option<ExprId> {
    let nonzero = if walk::free_symbols(arena, a).is_empty() {
        !has_dirac_delta(arena, a)
            && dirac_delta_argument_real(arena, a)
            && evalf::evalf_verdict(arena, a, HIDDEN_ZERO_DIGITS, evalf::ZeroSearch::Cap)
                .ok()
                .and_then(crate::base::truth::Verdict::proven)
                .is_some_and(|z| !(z.0.is_zero() && z.1.is_zero()))
    } else {
        crate::simplify::identically_zero::nonzero_at_a_sample(arena, a)
    };
    nonzero.then_some(arena.zero)
}

/// `0` for a symbol-free value `v` that a certified evaluation finds to be
/// exactly 0 (`binomial(−2, −1)`).
fn certified_zero_value(arena: &mut Arena, v: ExprId) -> Option<ExprId> {
    if !walk::free_symbols(arena, v).is_empty() || has_dirac_delta(arena, v) {
        return None;
    }
    let z = evalf::evalf_verdict(arena, v, HIDDEN_ZERO_DIGITS, evalf::ZeroSearch::Cap)
        .ok()?
        .proven()?;
    (z.0.is_zero() && z.1.is_zero()).then_some(arena.zero)
}

/// `(Π fᵢ)^q = Π fᵢ^q` when every factor `fᵢ` of the product `m` is a
/// positive real constant (certified), as for positive reals:
/// `√(3^(2/3)·e⁻²) = ∛3·e⁻¹`.  `None` otherwise.
fn power_of_positive_product(arena: &mut Arena, m: ExprId, q: ExprId) -> Option<ExprId> {
    let ExprNode::Mul(fs) = arena.node(m).clone() else {
        return None;
    };
    let mut powers: Vec<ExprId> = Vec::with_capacity(fs.len());
    for &f in &fs {
        if !real_constant_within(arena, f, 0.0, f64::INFINITY) {
            return None;
        }
        powers.push(arena.pow(f, q));
    }
    Some(arena.mul(&powers))
}

/// Largest order `|n|` [`special_function_rewrite`] reduces.
const MAX_SPECIAL_ORDER: i64 = 20;

/// Bessel functions of integer order and exponential integrals of
/// non-positive integer order in a normal form, by identities on all of
/// ℂ (DLMF 10.4.1, 10.11.1, 10.6.1; 10.27.1, 10.27.3, 10.29.1; 8.19.7 with
/// 8.4.8): `Jₙ`, `Yₙ`, `Iₙ`, `Kₙ` of `|n| ≤` [`MAX_SPECIAL_ORDER`] through
/// the orders 0 and 1 (`Jₙ`, `Iₙ` of an argument without a negative
/// coefficient), and
/// `E₋ₘ(z) = m!·e⁻ᶻ·Σₖ₌₀ᵐ zᵏ/k!/zᵐ⁺¹`.  So a combination the series
/// engine builds from derivatives (`J₁′ = (J₀ − J₂)/2`) cancels
/// structurally.  `None` for anything else.
fn special_function_rewrite(
    arena: &mut Arena,
    lib: crate::base::libfn::LibFn,
    order: ExprId,
    z: ExprId,
) -> Option<ExprId> {
    use crate::base::libfn::LibFn;
    use num_traits::ToPrimitive;
    let n = arena
        .as_num(order)
        .filter(|q| q.is_integer())?
        .to_integer()
        .to_i64()?;
    if n.abs() > MAX_SPECIAL_ORDER {
        return None;
    }
    match lib {
        LibFn::ExpInt if n <= 0 => {
            let m = -n;
            let mut terms: Vec<ExprId> = Vec::new();
            let mut factorial = num_bigint::BigInt::from(1);
            for k in 0..=m {
                if k > 0 {
                    factorial *= k;
                }
                let inv =
                    arena.num_ratio(crate::base::numeric::Q::new(1.into(), factorial.clone()));
                let k = arena.int(k);
                let zk = arena.pow(z, k);
                terms.push(arena.mul(&[inv, zk]));
            }
            let sum = arena.add(&terms);
            let m_fact = arena.big_int(factorial);
            let nz = arena.neg(z);
            let e = arena.exp(nz);
            let power = arena.int(-(m + 1));
            let den = arena.pow(z, power);
            Some(arena.mul(&[m_fact, e, sum, den]))
        }
        LibFn::BesselJ | LibFn::BesselI | LibFn::BesselY | LibFn::BesselK => {
            // J, Y: Jₖ = 2(k − 1)/z·Jₖ₋₁ − Jₖ₋₂ and J₋ₙ = (−1)ⁿ Jₙ;
            // I, K: Iₖ = Iₖ₋₂ − 2(k − 1)/z·Iₖ₋₁, Kₖ = Kₖ₋₂ + 2(k − 1)/z·Kₖ₋₁,
            // I₋ₙ = Iₙ, K₋ₙ = Kₙ.
            let bessel = matches!(lib, LibFn::BesselJ | LibFn::BesselY);
            let mut sign = if n < 0 && bessel && n % 2 != 0 { -1 } else { 1 };
            let n = n.abs();
            let (coeff, _) = arena.as_coeff_term(z);
            // Jₙ(−z) = (−1)ⁿ Jₙ(z), likewise Iₙ; not Yₙ, Kₙ (a cut on (−∞, 0]).
            let entire = matches!(lib, LibFn::BesselJ | LibFn::BesselI);
            let z = if entire && num_traits::Signed::is_negative(&coeff) {
                if n % 2 != 0 {
                    sign = -sign;
                }
                arena.neg(z)
            } else {
                z
            };
            let call = |arena: &mut Arena, k: i64| {
                let k = arena.int(k);
                arena.lib_apply(lib, &[k, z])
            };
            let (mut prev, mut cur) = (call(arena, 0), call(arena, 1));
            if n == 0 {
                cur = prev;
            }
            for k in 2..=n {
                // Jₖ = 2(k − 1)/z·Jₖ₋₁ − Jₖ₋₂; Iₖ = Iₖ₋₂ − 2(k − 1)/z·Iₖ₋₁.
                let c = arena.int(2 * (k - 1));
                let m1 = arena.neg_one();
                let zi = arena.pow(z, m1);
                let t = arena.mul(&[c, zi, cur]);
                let next = match lib {
                    LibFn::BesselJ | LibFn::BesselY => arena.sub(t, prev),
                    LibFn::BesselK => arena.add(&[prev, t]),
                    _ => arena.sub(prev, t),
                };
                (prev, cur) = (cur, next);
            }
            let s = arena.int(sign);
            Some(arena.mul(&[s, cur]))
        }
        _ => None,
    }
}

/// The product of the factors `fs` with its exponentials `exp(a)`, `e`,
/// `e^a` merged into one `exp(Σa)` (`exp(−1)·e = 1`, an identity on all of
/// ℂ).  `None` when there are fewer than two.
fn exponentials_combined(arena: &mut Arena, fs: &[ExprId]) -> Option<ExprId> {
    let mut exponents: Vec<ExprId> = Vec::new();
    let mut rest: Vec<ExprId> = Vec::new();
    for &f in fs {
        match *arena.node(f) {
            ExprNode::Exp(a) => exponents.push(a),
            ExprNode::E => exponents.push(arena.one()),
            ExprNode::Pow(b, a) if b == arena.e_const => exponents.push(a),
            _ => rest.push(f),
        }
    }
    if exponents.len() < 2 {
        return None;
    }
    let s = arena.add(&exponents);
    let e = arena.exp(s);
    rest.push(e);
    Some(arena.mul(&rest))
}

/// The product of the factors `fs` with a pair `Γ(p)·Γ(q)`, `p + q = 1`,
/// replaced by `π/sin(πp)` (the reflection formula, an identity wherever
/// both sides are finite): `Γ(1/3)·Γ(2/3) = 2π/√3`.  `None` when there
/// is no such pair.
fn gamma_reflection(arena: &mut Arena, fs: &[ExprId]) -> Option<ExprId> {
    let gammas: Vec<(usize, ExprId)> = fs
        .iter()
        .enumerate()
        .filter_map(|(i, &f)| match arena.node(f) {
            ExprNode::Gamma(a) => Some((i, *a)),
            _ => None,
        })
        .collect();
    for (j, &(i1, p)) in gammas.iter().enumerate() {
        for &(i2, q) in &gammas[j + 1..] {
            let s = arena.add(&[p, q]);
            if eval::eval(arena, s) != arena.one() {
                continue;
            }
            let mut rest: Vec<ExprId> = fs
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != i1 && i != i2)
                .map(|(_, &f)| f)
                .collect();
            let pi = arena.pi();
            let pi_p = arena.mul(&[pi, p]);
            let sine = arena.sin(pi_p);
            let m1 = arena.neg_one();
            let cosecant = arena.pow(sine, m1);
            rest.push(pi);
            rest.push(cosecant);
            return Some(arena.mul(&rest));
        }
    }
    None
}

/// `√b` with the square factors of the polynomial `b` that are nonnegative
/// (by the assumptions) taken out: `√(a² + 2a + 1) = a + 1` for `a > 0`
/// (`√(f²·g) = f·√g` for `f ≥ 0`, the principal root).  `None` when no
/// such factor exists or the content is not a positive rational.
fn sqrt_of_squares(arena: &mut Arena, b: ExprId) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    let (content, factors) = crate::simplify::factor::factor_list(arena, b, None);
    if factors.iter().all(|&(_, m)| m < 2)
        || !arena
            .as_num(content)
            .is_some_and(num_traits::Signed::is_positive)
    {
        return None;
    }
    let mut cache = AssumptionCache::new();
    let mut outside: Vec<ExprId> = Vec::new();
    let mut inside: Vec<ExprId> = vec![content];
    for (f, m) in factors {
        if m >= 2 && Truth::from(cache.query(arena, f, Props::NONNEGATIVE)).is_true() {
            let half = arena.int(i64::from(m / 2));
            let p = arena.pow(f, half);
            outside.push(p);
            if m % 2 == 1 {
                inside.push(f);
            }
        } else {
            let m = arena.int(i64::from(m));
            let p = arena.pow(f, m);
            inside.push(p);
        }
    }
    if outside.is_empty() {
        return None;
    }
    let rest = arena.mul(&inside);
    let root = arena.sqrt(rest);
    outside.push(root);
    Some(arena.mul(&outside))
}

/// Does `e` divide by something other than a number (a power with a
/// negative exponent of a non-atomic base)?
fn has_quotient(arena: &Arena, e: ExprId) -> bool {
    walk::post_order_ids(arena, e)
        .into_iter()
        .any(|id| match arena.node(id) {
            ExprNode::Pow(b, n) => {
                !arena.node(*b).is_atom()
                    && arena
                        .as_num(*n)
                        .is_some_and(num_traits::Signed::is_negative)
            }
            _ => false,
        })
}

/// `ln(p/q)` for a positive rational `p/q ≠ 1` as `Σ eᵢ·ln pᵢ` over the
/// prime factors of `p` and `q` (to [`MAX_LOG_PRIME`]; a larger cofactor
/// stays one logarithm) — exact, the logarithm of a positive real being
/// additive.  `None` for another argument, or when nothing splits.
fn ln_of_rational_in_primes(arena: &mut Arena, a: ExprId) -> Option<ExprId> {
    use num_traits::{One, Signed, ToPrimitive};
    let q = arena.as_num(a)?.clone();
    if !q.is_positive() || q.is_one() {
        return None;
    }
    let mut factors: Vec<(u64, i64)> = Vec::new();
    for (n, sign) in [(q.numer(), 1i64), (q.denom(), -1i64)] {
        let mut n = n.to_u64()?;
        let mut d = 2u64;
        while d <= MAX_LOG_PRIME && d.saturating_mul(d) <= n {
            let mut e = 0i64;
            while n % d == 0 {
                n /= d;
                e += 1;
            }
            if e > 0 {
                factors.push((d, sign * e));
            }
            d += 1;
        }
        if n > 1 {
            factors.push((n, sign));
        }
    }
    if factors.len() == 1 && factors[0].1 == 1 {
        return None; // a prime (or an unsplit cofactor) already
    }
    let terms: Vec<ExprId> = factors
        .into_iter()
        .map(|(p, e)| {
            let p = arena.big_int(num_bigint::BigInt::from(p));
            let l = arena.ln(p);
            let e = arena.int(e);
            arena.mul(&[e, l])
        })
        .collect();
    Some(arena.add(&terms))
}

/// `a` for `f⁻¹(f(a))` (`outer` the inverse function, `inner` its
/// argument `f(a)`) when `a` is a real constant (by the assumptions) whose
/// certified value lies inside the principal range of `f⁻¹`: `[−π/2, π/2]`
/// for `asin` and `atan`, `[0, π]` for `acos`, `[0, ∞)` for `acosh`, all
/// reals for `asinh` and `atanh`.  `None` otherwise.
fn inverse_of_value(arena: &mut Arena, outer: &ExprNode, inner: ExprId) -> Option<ExprId> {
    use std::f64::consts::{FRAC_PI_2, PI};
    let (a, lo, hi) = match (outer, arena.node(inner)) {
        (ExprNode::Asin(_), ExprNode::Sin(a)) | (ExprNode::Atan(_), ExprNode::Tan(a)) => {
            (*a, -FRAC_PI_2, FRAC_PI_2)
        }
        (ExprNode::Acos(_), ExprNode::Cos(a)) => (*a, 0.0, PI),
        (ExprNode::Acosh(_), ExprNode::Cosh(a)) => (*a, 0.0, f64::INFINITY),
        (ExprNode::Asinh(_), ExprNode::Sinh(a)) | (ExprNode::Atanh(_), ExprNode::Tanh(a)) => {
            (*a, f64::NEG_INFINITY, f64::INFINITY)
        }
        _ => return None,
    };
    real_constant_within(arena, a, lo, hi).then_some(a)
}

/// `i·acos(a)` for `acosh(a)` when `a` is a real constant (by the
/// assumptions) whose certified value lies in `(−1, 1)`, where the
/// principal values agree (`acosh(1/2) = iπ/3`; mpmath `acosh(0.5)` =
/// `1.0471975511966j`): the form in which Gruntz continues `acosh` along
/// its cut.
fn acosh_as_acos(arena: &mut Arena, a: ExprId) -> Option<ExprId> {
    if !real_constant_within(arena, a, -1.0, 1.0) {
        return None;
    }
    let acos = arena.acos(a);
    let i = arena.i_unit();
    Some(arena.mul(&[i, acos]))
}

/// `f(g(z))` for a trigonometric (hyperbolic) `f` of an inverse
/// trigonometric (hyperbolic) `g`, an algebraic function of `z` on all of
/// ℂ with the principal branches (as SymPy evaluates them): `sin(asin z) =
/// z`, `cos(asin z) = sin(acos z) = √(1 − z²)`, `tan(asin z) = z/√(1 −
/// z²)`, `sin(atan z) = z/√(1 + z²)`, `cosh(asinh z) = √(1 + z²)`, …
/// `None` for another pair.
fn function_of_inverse(arena: &mut Arena, outer: &ExprNode, inner: ExprId) -> Option<ExprId> {
    let one = arena.one();
    let two = arena.int(2);
    let half = arena.rational(1, 2);
    let m_half = arena.rational(-1, 2);
    // √(1 ∓ z²) and its reciprocal.
    let root = |arena: &mut Arena, z: ExprId, minus: bool, e: ExprId| {
        let z2 = arena.pow(z, two);
        let s = if minus {
            arena.sub(one, z2)
        } else {
            arena.add(&[one, z2])
        };
        arena.pow(s, e)
    };
    Some(match (outer, arena.node(inner).clone()) {
        (ExprNode::Sin(_), ExprNode::Asin(z))
        | (ExprNode::Cos(_), ExprNode::Acos(z))
        | (ExprNode::Tan(_), ExprNode::Atan(z))
        | (ExprNode::Sinh(_), ExprNode::Asinh(z))
        | (ExprNode::Cosh(_), ExprNode::Acosh(z))
        | (ExprNode::Tanh(_), ExprNode::Atanh(z)) => z,
        (ExprNode::Cos(_), ExprNode::Asin(z)) | (ExprNode::Sin(_), ExprNode::Acos(z)) => {
            root(arena, z, true, half)
        }
        (ExprNode::Tan(_), ExprNode::Asin(z)) => {
            let r = root(arena, z, true, m_half);
            arena.mul(&[z, r])
        }
        (ExprNode::Tan(_), ExprNode::Acos(z)) => {
            let r = root(arena, z, true, half);
            let m1 = arena.neg_one();
            let zi = arena.pow(z, m1);
            arena.mul(&[r, zi])
        }
        (ExprNode::Sin(_), ExprNode::Atan(z)) => {
            let r = root(arena, z, false, m_half);
            arena.mul(&[z, r])
        }
        (ExprNode::Cos(_), ExprNode::Atan(z)) => root(arena, z, false, m_half),
        (ExprNode::Cosh(_), ExprNode::Asinh(z)) => root(arena, z, false, half),
        (ExprNode::Sinh(_), ExprNode::Acosh(z)) => {
            let zm = arena.sub(z, one);
            let zp = arena.add(&[z, one]);
            let rm = arena.sqrt(zm);
            let rp = arena.sqrt(zp);
            arena.mul(&[rm, rp])
        }
        _ => return None,
    })
}

/// `asin(a)` for a real constant `a` beyond `±1` (certified), on the cut
/// of `asin`: `π/2 − i·acosh(a)` for `a > 1`, `−π/2 + i·acosh(−a)` for
/// `a < −1` — the principal values there (mpmath `asin(2)` =
/// `1.5708 − 1.3170j`), so that `asin(φ)` and the `acosh` written by
/// Gruntz's tractable rewrite meet.
fn asin_on_cut(arena: &mut Arena, a: ExprId) -> Option<ExprId> {
    let (t, sign) = if real_constant_within(arena, a, 1.0, f64::INFINITY) {
        (a, 1)
    } else if real_constant_within(arena, a, f64::NEG_INFINITY, -1.0) {
        (arena.neg(a), -1)
    } else {
        return None;
    };
    let half = arena.rational(1, 2);
    let pi = arena.pi();
    let half_pi = arena.mul(&[half, pi]);
    let i = arena.i_unit();
    let acosh = arena.acosh(t);
    let i_acosh = arena.mul(&[i, acosh]);
    let v = arena.sub(half_pi, i_acosh);
    let s = arena.int(sign);
    Some(arena.mul(&[s, v]))
}

/// Is `a` a real constant (by the assumptions) whose certified value lies
/// strictly inside `(lo, hi)`, with a margin far above its error?
fn real_constant_within(arena: &mut Arena, a: ExprId, lo: f64, hi: f64) -> bool {
    certified_real_value(arena, a).is_some_and(|v| {
        let margin = value_margin(v);
        v > lo + margin && v < hi - margin
    })
}

/// The certified value of a real constant `a` (real by the assumptions),
/// to [`HIDDEN_ZERO_DIGITS`] digits, as an `f64`; compare it with a
/// [`value_margin`].
fn certified_real_value(arena: &mut Arena, a: ExprId) -> Option<f64> {
    if !walk::free_symbols(arena, a).is_empty() {
        return None;
    }
    let mut cache = crate::base::assumptions::AssumptionCache::new();
    if !Truth::from(cache.query(arena, a, crate::base::assumptions::Props::REAL)).is_true() {
        return None;
    }
    let z = evalf::evalf_verdict(arena, a, HIDDEN_ZERO_DIGITS, evalf::ZeroSearch::Cap)
        .ok()?
        .proven()?;
    crate::base::bigfloat::to_f64(&z.0)
}

/// A margin far above the error of a [`certified_real_value`] `v` (30
/// certified digits, then rounded to `f64`).
fn value_margin(v: f64) -> f64 {
    1e-9 * v.abs().max(1.0)
}

/// The argument of `min(args)` (`max` for `min = false`) of real constants
/// whose certified value is below (above) every other by a margin:
/// `min(0, Catalan − 1/6, bernoulli(1/2)) = 0`.  `None` otherwise.
fn extreme_constant(arena: &mut Arena, args: &[ExprId], min: bool) -> Option<ExprId> {
    let mut vals: Vec<(f64, ExprId)> = Vec::with_capacity(args.len());
    for &a in args {
        let v = certified_real_value(arena, a)?;
        vals.push(if min { (v, a) } else { (-v, a) });
    }
    let &(best, arg) = vals.iter().min_by(|p, q| p.0.total_cmp(&q.0))?;
    let margin = value_margin(best);
    vals.iter()
        .all(|&(v, a)| a == arg || v > best + margin.max(value_margin(v)))
        .then_some(arg)
}

/// A constant after the zero test of [`settle_constant`], the first steps of
/// [`ZeroDecision::decide`] (without the identity test of the parameter
/// parts): a reading of a value to keep, drop or compare, never a divisor —
/// a divisor is a [`NonzeroExpr`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ZeroTest {
    /// Proved 0, exactly or by a certified evaluation.
    Zero,
    /// The evaluated constant, not 0: proved nonzero, or a constant with
    /// free parameters, which is taken as nonzero for generic values of
    /// them (the generic-solution convention; [`ZeroDecision::decide`]
    /// recognises the ones that vanish identically).  So is a constant
    /// without a value — one `evalf` cannot evaluate or with an
    /// unevaluated part (`legendre(3/2, e) − 1`): it is opaque like a
    /// parameter, and there is no numerical hint that it vanishes.
    Nonzero(ExprId),
    /// The evaluated constant, 0 to the precision reached (at the sample
    /// points of its parameters, if any) but proved neither 0 nor nonzero:
    /// a caller may keep it as a value, but may not divide by it, take it
    /// for a leading coefficient, or drop it.
    Undecided(ExprId),
}

impl ZeroTest {
    /// `0` for `Zero`, the evaluated constant otherwise.
    pub(crate) fn value(self, arena: &Arena) -> ExprId {
        match self {
            ZeroTest::Zero => arena.zero,
            ZeroTest::Nonzero(v) | ZeroTest::Undecided(v) => v,
        }
    }
}

/// `c` after `eval`, and whether it is 0 ([`hidden_zero`]; a constant with
/// free parameters or without a value is `Nonzero`, see
/// [`ZeroTest::Nonzero`]).  A piecewise one that may vanish on a region of
/// its parameters (`0^y`, `δ(1/2, y)`:
/// [`may_vanish_on_a_region`](crate::simplify::identically_zero::may_vanish_on_a_region))
/// is `Undecided`: the generic convention does not cover it.
pub(crate) fn settle_constant(arena: &mut Arena, c: ExprId) -> ZeroTest {
    let v = eval::eval(arena, c);
    if arena.is_zero_structural(v) {
        return ZeroTest::Zero;
    }
    if !walk::free_symbols(arena, v).is_empty() {
        if crate::simplify::identically_zero::may_vanish_on_a_region(arena, v) {
            return ZeroTest::Undecided(v);
        }
        return ZeroTest::Nonzero(v);
    }
    match zero_reading(hidden_zero(arena, v)) {
        crate::base::truth::Truth::True => ZeroTest::Zero,
        crate::base::truth::Truth::False => ZeroTest::Nonzero(v),
        crate::base::truth::Truth::Unknown => ZeroTest::Undecided(v),
    }
}

/// How the limit and series engines read a [`hidden_zero`] verdict: a
/// proof decides; a value 0 to the precision reached and not proved
/// (`True` on `Approximate` evidence) is `Unknown`; a constant without a
/// value (`Unknown`) is `False`, opaque like a parameter (see
/// [`ZeroTest::Nonzero`]).
pub(crate) fn zero_reading(
    verdict: crate::base::truth::Verdict<crate::base::truth::Truth>,
) -> crate::base::truth::Truth {
    use crate::base::truth::Truth;
    match (verdict.proven(), verdict.value) {
        (Some(t), _) => t,
        (None, Truth::True) => Truth::Unknown,
        (None, Truth::False | Truth::Unknown) => Truth::False,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Proof-carrying divisors
// ═══════════════════════════════════════════════════════════════════════════

/// An expression shown to be nonzero: the only divisor the series, limit
/// and Gruntz engines divide by ([`div_by`], [`recip`]).
///
/// A series coefficient, a leading term or a derivative at a point can be 0
/// without looking like it: `besselj(−1/2, π/2)` (0 to every digit, not
/// proved), `H(y)`, `0^y` or `⌊y⌋` (0 on a region of the parameter).  Each
/// site used to pick its own zero test before dividing, and some picked
/// none: up to 0.57 `series(ln(x + besselj(−1/2, π/2)), x)` was `ln
/// J + x/J − …` and `series(√(x + H(y)), x)` was `√H(y) + x/(2√H(y)) + …`
/// (both divided by a zero: `ln x` and `√x` have no Taylor series), as were
/// `series(Π(x | m))` for such an `m`, L'Hôpital's quotient by
/// `J_{1/2}(π)` and the series reciprocal of `besselj(−1/2, π/2 + x)` before
/// them.  The field is private: a value of this type exists only where a
/// decider below has shown the divisor nonzero, and the ratchet
/// `tests/unit/test_divisors.rs` counts the divisions those engines still
/// build by hand.
///
/// # Which constructor
///
/// | Divisor | Constructor | Cost |
/// |---|---|---|
/// | a coefficient, a leading term, a value at a point, a direction | [`ZeroDecision::decide`] | `eval`, then the zero test of [`hidden_zero`] (memoised) or of the parameters |
/// | a nonzero number (`2`, `−1/3`) | [`NonzeroExpr::number`] | O(1) |
/// | a product of powers of nonzero numbers and the named constants (`√π`, `2i`, `e⁻¹`) | [`NonzeroExpr::structural`] | one walk, no evaluation |
/// | the dummy symbol of a substitution `x = a + 1/t` (an indeterminate, not a value) | [`NonzeroExpr::variable`] | O(1) |
///
/// What "nonzero" means is the convention of [`ZeroTest::Nonzero`]: proved
/// for a constant without symbols that has a value; for a constant with
/// parameters, not identically zero in them (nonzero for generic values,
/// the generic-solution convention) and not piecewise — a part that may
/// vanish on a region (`0^y`, `δ`, `H`, `⌊·⌋`, `min`/`max`:
/// [`may_vanish_on_a_region`](crate::simplify::identically_zero::may_vanish_on_a_region))
/// is never generic.
///
/// Quotients that are not divisions by a value keep `Arena::div` with a
/// comment: a quotient of two functions of the variable that forms a new
/// limit problem (`f′/g′` before its limit, `ln a/ln b` in Gruntz's
/// comparison), and identities of functions (`tan u = sin u/cos u`,
/// `erfi u = e^{u²}·E(u²)/u` for `u → ∞`).
///
/// # Example
///
/// ```
/// use symplex::prelude::*;
///
/// let ctx = Context::new();
/// let x = ctx.symbol("x");
/// let zero = ctx.int(0);
/// // `H(y)` is 0 for `y < 0`: `ln(x + H(y))` has no series there.
/// let e = ctx.parse("log(x + Heaviside(y))").unwrap();
/// assert!(e.try_series(&x, &zero, 4).is_err());
/// // A generic parameter is a nonzero divisor.
/// let e = ctx.parse("log(x + y)").unwrap();
/// assert_eq!(
///     e.try_series(&x, &zero, 2).unwrap().to_string(),
///     "x/y + ln(y)"
/// );
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct NonzeroExpr(ExprId);

impl NonzeroExpr {
    /// The divisor.
    pub(crate) fn id(self) -> ExprId {
        self.0
    }

    /// `c` if it is a nonzero number.
    pub(crate) fn number(arena: &Arena, c: ExprId) -> Option<Self> {
        arena
            .as_num(c)
            .is_some_and(|r| !num_traits::Zero::is_zero(r))
            .then_some(NonzeroExpr(c))
    }

    /// `c` if its canonical form is a product of powers of nonzero numbers
    /// and the named constants `π`, `e`, `i`, `γ`, `G`, `φ` (`√π`, `2i`,
    /// `e⁻¹`, `√5`): nonzero without evaluation, the structural shortcut of
    /// [`hidden_zero`].
    pub(crate) fn structural(arena: &Arena, c: ExprId) -> Option<Self> {
        walk::post_order_ids(arena, c)
            .into_iter()
            .all(|id| match arena.node(id) {
                ExprNode::Num(_) => Self::number(arena, id).is_some(),
                ExprNode::Pi
                | ExprNode::E
                | ExprNode::ImaginaryUnit
                | ExprNode::EulerGamma
                | ExprNode::Catalan
                | ExprNode::GoldenRatio
                | ExprNode::Mul(_)
                | ExprNode::Pow(..)
                | ExprNode::Neg(_) => true,
                _ => false,
            })
            .then_some(NonzeroExpr(c))
    }

    /// The symbol `s` as an indeterminate: the dummy `t` of a substitution
    /// `x = a ± 1/t` or `x = 1/t`, the variable of an expansion.  Dividing by
    /// it builds a function of it, not a quotient by a value.  `None` for
    /// anything but a symbol.
    pub(crate) fn variable(arena: &Arena, s: ExprId) -> Option<Self> {
        matches!(arena.node(s), ExprNode::Symbol(_)).then_some(NonzeroExpr(s))
    }
}

/// `n/d` (`n·d⁻¹`, as `Arena::div` builds it) for a divisor shown nonzero.
pub(crate) fn div_by(arena: &mut Arena, n: ExprId, d: NonzeroExpr) -> ExprId {
    arena.div(n, d.0)
}

/// `1/d` (`d⁻¹`) for a divisor shown nonzero.
pub(crate) fn recip(arena: &mut Arena, d: NonzeroExpr) -> ExprId {
    let m1 = arena.neg_one();
    arena.pow(d.0, m1)
}

/// A constant (or a coefficient with parameters) after the zero test of
/// [`ZeroDecision::decide`]: the reading of [`ZeroTest`] with the nonzero
/// value carried as a divisor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ZeroDecision {
    /// Proved 0, exactly or by a certified evaluation, or identically 0 in
    /// its parameters.
    Zero,
    /// The evaluated constant, nonzero (see [`NonzeroExpr`] for the
    /// convention on parameters).
    Nonzero(NonzeroExpr),
    /// The evaluated constant, 0 to the precision reached (at the sample
    /// points of its parameters, if any) or piecewise in its parameters, but
    /// proved neither 0 nor nonzero: a caller may keep it as a value, but
    /// may not divide by it, take it for a leading coefficient, or drop it.
    Undecided(ExprId),
}

impl ZeroDecision {
    /// Is `c` zero?  The one entry point for a divisor, applied in order:
    ///
    /// 1. `eval`, and the structural test (`0` is `Zero`);
    /// 2. a constant without symbols: [`hidden_zero`], read by
    ///    [`zero_reading`] (a proof decides; 0 to the precision reached and
    ///    not proved is `Undecided`; a constant without a value is opaque
    ///    like a parameter, `Nonzero`);
    /// 3. with symbols, a piecewise part that may vanish on a region of
    ///    the parameters
    ///    ([`may_vanish_on_a_region`](crate::simplify::identically_zero::may_vanish_on_a_region):
    ///    `0^y`, `H(y)`, `δ(1/2, y)`, `⌊y⌋`) is `Undecided` — up to 0.56
    ///    `series(acosh(P₂(x))/(x² + H(y)), x)` divided by `H(y)`;
    /// 4. the parameter parts: the symbol-free coefficients `kₗ` of `c = Σ
    ///    kₗ·mₗ` are decided by step 2 and the groups proved 0 dropped
    ///    ([`parameter_parts`](crate::calculus::series::parameter_parts)): `ω·k`
    ///    is `Zero` when `k` is proved 0, `Undecided` when `k` is not
    ///    decided, never "nonzero";
    /// 5. two or more parts left: whether they cancel identically
    ///    ([`vanishes_identically`](crate::calculus::series::vanishes_identically):
    ///    sampled within the assumptions, confirmed exactly).
    ///
    /// Before steps 4–5, a constant with parameters is searched for hidden
    /// zeros inside its functions ([`hidden_parts`]): its maximal
    /// symbol-free parts (`J` in `sin(y·J)`) are decided by
    /// [`hidden_zero`]; one proved 0 is replaced by 0, and one 0 to the
    /// precision reached but not proved makes the constant `Nonzero` only
    /// when it is nonzero both as it stands and with that part replaced by
    /// 0 (`y + J` is; `sin(y·J)` is `Undecided`).  Up to 0.58 steps 4–5 saw
    /// only symbol-free *factors*: `cos(y·J + π/2) = −sin(y·J)` (`J =
    /// besselj(−1/2, π/2) = 0`) was nonzero for a generic `y`, and
    /// `limit(tan(π/2 + y·besselj(−1/2, π/2 + x)), x, 0)` was the value
    /// `tan(y·J + π/2)` at a pole.
    pub(crate) fn decide(arena: &mut Arena, c: ExprId) -> Self {
        Self::decide_with(arena, c, HIDDEN_PART_ROUNDS)
    }

    /// [`decide`](Self::decide), replacing hidden zeros of the parameter
    /// parts at most `rounds` more times.
    fn decide_with(arena: &mut Arena, c: ExprId, rounds: u8) -> Self {
        if arena.is_zero_structural(c) {
            return ZeroDecision::Zero;
        }
        if let Some(d) = NonzeroExpr::number(arena, c) {
            return ZeroDecision::Nonzero(d);
        }
        let v = match settle_constant(arena, c) {
            ZeroTest::Zero => return ZeroDecision::Zero,
            ZeroTest::Undecided(v) => return ZeroDecision::Undecided(v),
            // `settle_constant` has applied steps 1–3.
            ZeroTest::Nonzero(v) => v,
        };
        if arena.is_symbol_free(v) {
            return ZeroDecision::Nonzero(NonzeroExpr(v));
        }
        if rounds > 0 {
            match hidden_parts(arena, v) {
                HiddenParts::None => {}
                HiddenParts::Proved(w) => return Self::decide_with(arena, w, rounds - 1),
                // `v` is one of two values: as it stands (the part a nonzero
                // constant) or `w` (the part 0).
                HiddenParts::Undecided(w) => {
                    let at_zero = Self::decide_with(arena, w, 0);
                    return match (at_zero, Self::by_parameter_parts(arena, v)) {
                        (ZeroDecision::Nonzero(_), ZeroDecision::Nonzero(d)) => {
                            ZeroDecision::Nonzero(d)
                        }
                        (ZeroDecision::Zero, ZeroDecision::Zero) => ZeroDecision::Zero,
                        _ => ZeroDecision::Undecided(v),
                    };
                }
                HiddenParts::Several => return ZeroDecision::Undecided(v),
            }
        }
        Self::by_parameter_parts(arena, v)
    }

    /// Steps 4–5 of [`decide`](Self::decide) for the evaluated constant `v`
    /// with parameters.
    fn by_parameter_parts(arena: &mut Arena, v: ExprId) -> Self {
        use crate::calculus::series::{ParameterParts, parameter_parts, vanishes_identically};
        let sum = match parameter_parts(arena, v) {
            ParameterParts::Decided(t) => return Self::from_test(t),
            ParameterParts::Sum(sum) => sum,
        };
        match vanishes_identically(arena, sum) {
            Truth::True => ZeroDecision::Zero,
            Truth::False => ZeroDecision::Nonzero(NonzeroExpr(v)),
            Truth::Unknown => ZeroDecision::Undecided(sum),
        }
    }

    /// A result of the deciders behind [`decide`](Self::decide) as a
    /// decision.
    fn from_test(t: ZeroTest) -> Self {
        match t {
            ZeroTest::Zero => ZeroDecision::Zero,
            ZeroTest::Nonzero(v) => ZeroDecision::Nonzero(NonzeroExpr(v)),
            ZeroTest::Undecided(v) => ZeroDecision::Undecided(v),
        }
    }

    /// `True` for `Zero`, `False` for `Nonzero`, `Unknown` for `Undecided`.
    pub(crate) fn is_zero(self) -> Truth {
        match self {
            ZeroDecision::Zero => Truth::True,
            ZeroDecision::Nonzero(_) => Truth::False,
            ZeroDecision::Undecided(_) => Truth::Unknown,
        }
    }

    /// The divisor, when shown nonzero.
    pub(crate) fn nonzero(self) -> Option<NonzeroExpr> {
        match self {
            ZeroDecision::Nonzero(d) => Some(d),
            ZeroDecision::Zero | ZeroDecision::Undecided(_) => None,
        }
    }

    /// `0` for `Zero`, the evaluated constant otherwise.
    pub(crate) fn value(self, arena: &Arena) -> ExprId {
        match self {
            ZeroDecision::Zero => arena.zero,
            ZeroDecision::Nonzero(d) => d.0,
            ZeroDecision::Undecided(v) => v,
        }
    }
}

/// How many times [`ZeroDecision::decide`] replaces the hidden zeros of a
/// constant's parameter parts and decides again (a replaced part can
/// uncover another: `sin(y·(J + sin(y·J)))`).
const HIDDEN_PART_ROUNDS: u8 = 3;

/// The hidden zeros [`hidden_parts`] found in a constant with parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HiddenParts {
    /// No maximal symbol-free part is 0 or undecided.
    None,
    /// The constant with its parts proved 0 replaced by 0, evaluated.
    Proved(ExprId),
    /// The constant with its one part that is 0 to the precision reached,
    /// not proved, replaced by 0, evaluated.
    Undecided(ExprId),
    /// Two or more parts 0 to the precision reached, not proved.
    Several,
}

/// Most entries of [`HIDDEN_PARTS_MEMO`] before it is cleared.
const HIDDEN_PARTS_MEMO_ENTRIES: usize = 4096;

thread_local! {
    /// The results of [`hidden_parts`] by (arena serial, constant): the
    /// series engine decides the same coefficients in every product and sum
    /// it forms, and a part `evalf` cannot evaluate is not memoised by
    /// [`hidden_zero`].
    static HIDDEN_PARTS_MEMO: std::cell::RefCell<rustc_hash::FxHashMap<(u64, ExprId), HiddenParts>> =
        std::cell::RefCell::new(rustc_hash::FxHashMap::default());
}

/// The maximal symbol-free parts of the evaluated constant `v` with
/// parameters that are hidden zeros: each argument without symbols of a
/// node with symbols (`J` in `sin(y·J)`, `√6 − √2·√3` in `y^(√6 −
/// √2·√3)`), the symbol-free terms of a sum together, decided by
/// [`hidden_zero`] read by [`zero_reading`] (numbers, and products of
/// powers of numbers and named constants, are skipped: canonical forms
/// decide them).  Parts proved 0 are replaced first; otherwise one part
/// that is 0 to the precision reached but not proved.  Memoised per arena.
fn hidden_parts(arena: &mut Arena, v: ExprId) -> HiddenParts {
    let key = (arena.serial(), v);
    if let Some(h) = HIDDEN_PARTS_MEMO.with(|m| m.borrow().get(&key).copied()) {
        return h;
    }
    let h = hidden_parts_uncached(arena, v);
    HIDDEN_PARTS_MEMO.with(|m| {
        let mut m = m.borrow_mut();
        if m.len() >= HIDDEN_PARTS_MEMO_ENTRIES {
            m.clear();
        }
        m.insert(key, h);
    });
    h
}

/// [`hidden_parts`], not memoised.
fn hidden_parts_uncached(arena: &mut Arena, v: ExprId) -> HiddenParts {
    // (part, what replaces the part's node when the part is 0)
    let mut proved: Vec<(ExprId, ExprId)> = Vec::new();
    let mut undecided: Vec<(ExprId, ExprId)> = Vec::new();
    let mut seen: rustc_hash::FxHashSet<ExprId> = rustc_hash::FxHashSet::default();
    let mut stack = vec![v];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let mut parts: smallvec::SmallVec<[(ExprId, ExprId, ExprId); 4]> =
            smallvec::SmallVec::new();
        if let ExprNode::Add(ts) = arena.node(id).clone() {
            // The terms grouped by their factor with symbols (`1` for a
            // symbol-free term): the coefficient of a group of two or more
            // terms is a part (`asinh 2 − ln(2 + √5)` in `y·asinh 2 − y·ln(2
            // + √5)`), replaced by dropping the group; the coefficient of a
            // single term is met inside it.
            type Group = (ExprId, smallvec::SmallVec<[(ExprId, ExprId); 2]>);
            let mut groups: Vec<Group> = Vec::new();
            for &t in &ts {
                let (k, m) = if arena.is_symbol_free(t) {
                    (t, arena.one())
                } else if let ExprNode::Mul(fs) = arena.node(t).clone() {
                    let (k, m): (
                        smallvec::SmallVec<[ExprId; 4]>,
                        smallvec::SmallVec<[ExprId; 4]>,
                    ) = fs.iter().copied().partition(|&f| arena.is_symbol_free(f));
                    (arena.mul(&k), arena.mul(&m))
                } else {
                    (arena.one(), t)
                };
                match groups.iter_mut().find(|g| g.0 == m) {
                    Some(g) => g.1.push((t, k)),
                    None => groups.push((m, smallvec::smallvec![(t, k)])),
                }
            }
            for (m, members) in groups {
                let symbol_free = m == arena.one();
                if !symbol_free {
                    stack.extend(members.iter().map(|&(t, _)| t));
                    if members.len() < 2 {
                        continue;
                    }
                }
                let ks: smallvec::SmallVec<[ExprId; 2]> = members.iter().map(|&(_, k)| k).collect();
                let part = arena.add(&ks);
                let rest: Vec<ExprId> = ts
                    .iter()
                    .copied()
                    .filter(|t| !members.iter().any(|&(u, _)| u == *t))
                    .collect();
                let without = arena.add(&rest);
                let (node, replacement) = match members.as_slice() {
                    [(t, _)] => (*t, arena.zero),
                    _ => (id, without),
                };
                parts.push((part, node, replacement));
            }
        } else {
            let mut children: smallvec::SmallVec<[ExprId; 4]> = smallvec::SmallVec::new();
            arena.node(id).for_each_child(|c| children.push(c));
            for c in children {
                if arena.is_symbol_free(c) {
                    parts.push((c, c, arena.zero));
                } else {
                    stack.push(c);
                }
            }
        }
        for (part, node, replacement) in parts {
            if arena.as_num(part).is_some() || NonzeroExpr::structural(arena, part).is_some() {
                continue;
            }
            match zero_reading(hidden_zero(arena, part)) {
                Truth::True => proved.push((node, replacement)),
                Truth::Unknown => undecided.push((node, replacement)),
                Truth::False => {}
            }
        }
    }
    let replaced = |arena: &mut Arena, pairs: &[(ExprId, ExprId)]| {
        let w = crate::transforms::subs::subs_map(arena, v, pairs);
        eval::eval(arena, w)
    };
    if !proved.is_empty() {
        return HiddenParts::Proved(replaced(arena, &proved));
    }
    match undecided.as_slice() {
        [] => HiddenParts::None,
        [pair] => HiddenParts::Undecided(replaced(arena, &[*pair])),
        _ => HiddenParts::Several,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::arena::Arena;
    use crate::base::node::ExprNode;
    use std::f64::consts::PI;

    /// A constant of more than one unit of evaluation work is undecided
    /// past the allowance of an [`EvaluationScope`] (never "no value",
    /// which would read as nonzero), certified nonzero without one, and an
    /// inner scope's allowance ends with it.
    #[test]
    fn evaluation_scope_bounds_the_zero_tests() {
        let mut arena = Arena::new();
        let terms: Vec<ExprId> = (2..60)
            .map(|k| {
                let n = arena.int(k);
                let r = arena.sqrt(n);
                let l = arena.ln(n);
                arena.mul(&[r, l])
            })
            .collect();
        let c = arena.add(&terms);
        assert!(walk::post_order_ids(&arena, c).len() > 2 * NODES_PER_EVALUATION_UNIT);
        {
            let _outer = EvaluationScope::enter(1_000);
            {
                let inner = EvaluationScope::enter(1);
                let v = hidden_zero(&mut arena, c);
                assert_eq!(zero_reading(v), Truth::Unknown);
                assert!(inner.made() >= 1);
            }
            // The outer scope has room again (the inner one spent only its own).
            assert_eq!(zero_reading(hidden_zero(&mut arena, c)), Truth::False);
        }
        assert_eq!(zero_reading(hidden_zero(&mut arena, c)), Truth::False);
    }

    /// Helper: get the SymbolId for an ExprId that is a Symbol.
    fn sym_id(arena: &Arena, expr: ExprId) -> SymbolId {
        match arena.node(expr) {
            ExprNode::Symbol(sid) => *sid,
            _ => panic!("expected Symbol node"),
        }
    }

    /// Helper: check if a set (as displayed) contains an interval description.
    fn set_display(arena: &Arena, set: ExprId) -> String {
        // Use the arena's display infrastructure via a minimal formatter
        use crate::output::display;
        display::format_expr(arena, set)
    }

    // ── Test 1: domain_sqrt_x ──────────────────────────────────────

    #[test]
    fn domain_sqrt_x() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let x_sym = sym_id(&arena, x);

        // sqrt(x) = x^(1/2)
        let expr = arena.sqrt(x);

        // domain = ℝ = (-∞, ∞)
        let reals = arena.interval(arena.neg_infinity, arena.infinity, IntervalKind::Open);

        let result = continuous_domain(&mut arena, expr, x, x_sym, reals);
        let s = set_display(&arena, result);

        // Should be [0, ∞) or equivalent
        assert!(
            s.contains("0") && (s.contains("oo") || s.contains("∞")),
            "sqrt(x) domain should be [0, ∞), got: {s}"
        );
        // Should NOT be the empty set
        assert!(
            !s.contains("EmptySet"),
            "sqrt(x) domain should not be empty: {s}"
        );
    }

    // ── Test 2: domain_1_over_x ────────────────────────────────────

    #[test]
    fn domain_1_over_x() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let x_sym = sym_id(&arena, x);

        // 1/x = x^(-1)
        let expr = arena.div(arena.one, x);

        let reals = arena.interval(arena.neg_infinity, arena.infinity, IntervalKind::Open);
        let result = continuous_domain(&mut arena, expr, x, x_sym, reals);
        let s = set_display(&arena, result);

        // Should exclude x=0: (-∞, 0) ∪ (0, ∞) or ℝ \ {0}
        assert!(
            !s.contains("EmptySet"),
            "1/x domain should not be empty: {s}"
        );
        assert!(
            s.contains("0"),
            "1/x domain should reference 0 as excluded point: {s}"
        );
    }

    // ── Test 3: domain_ln_x ───────────────────────────────────────

    #[test]
    fn domain_ln_x() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let x_sym = sym_id(&arena, x);

        let expr = arena.ln(x);

        let reals = arena.interval(arena.neg_infinity, arena.infinity, IntervalKind::Open);
        let result = continuous_domain(&mut arena, expr, x, x_sym, reals);
        let s = set_display(&arena, result);

        // Should be (0, ∞)
        assert!(
            !s.contains("EmptySet"),
            "ln(x) domain should not be empty: {s}"
        );
        assert!(
            s.contains("0") && (s.contains("oo") || s.contains("∞")),
            "ln(x) domain should be (0, ∞), got: {s}"
        );
    }

    // ── Test 4: domain_sqrt_x_minus_2 over [-5, 5] ────────────────

    #[test]
    fn domain_sqrt_x_minus_2() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let x_sym = sym_id(&arena, x);

        // sqrt(x - 2)
        let two = arena.int(2);
        let inner = arena.sub(x, two);
        let expr = arena.sqrt(inner);

        // domain = [-5, 5]
        let neg5 = arena.int(-5);
        let five = arena.int(5);
        let domain = arena.interval(neg5, five, IntervalKind::Closed);

        let result = continuous_domain(&mut arena, expr, x, x_sym, domain);
        let s = set_display(&arena, result);

        // sqrt(x-2) requires x-2 ≥ 0 → x ≥ 2, intersected with [-5,5] → [2, 5]
        assert!(
            !s.contains("EmptySet"),
            "sqrt(x-2) on [-5,5] should not be empty: {s}"
        );
        assert!(
            s.contains("2") && s.contains("5"),
            "sqrt(x-2) on [-5,5] should give [2, 5], got: {s}"
        );
    }

    // ── Test 5: singularities_tan_x ────────────────────────────────

    #[test]
    fn singularities_tan_x() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let x_sym = sym_id(&arena, x);

        let expr = arena.tan(x);

        let sings = singularities(&mut arena, expr, x, x_sym, Interval::closed(0.0, 5.0));

        // tan(x) has singularities at π/2 ≈ 1.5708 and 3π/2 ≈ 4.7124
        // in [0, 5]
        assert!(
            !sings.is_empty(),
            "tan(x) should have singularities in [0, 5]"
        );

        // Check that π/2 is approximately present
        let has_pi_half = sings.iter().any(|&v| (v - PI / 2.0).abs() < 0.1);
        assert!(
            has_pi_half,
            "tan(x) singularities should include ≈π/2, got: {:?}",
            sings
        );
    }

    // ── Test 6: singularities_1_over_x ─────────────────────────────

    #[test]
    fn singularities_1_over_x() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let x_sym = sym_id(&arena, x);

        let expr = arena.div(arena.one, x);

        let sings = singularities(&mut arena, expr, x, x_sym, Interval::closed(-2.0, 2.0));

        assert_eq!(
            sings.len(),
            1,
            "1/x should have one singularity: {:?}",
            sings
        );
        assert!(
            sings[0].abs() < 1e-10,
            "1/x singularity should be at 0, got: {}",
            sings[0]
        );
    }

    // ── Test 7: frequency_sin_100x ─────────────────────────────────

    #[test]
    fn frequency_sin_100x() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let x_sym = sym_id(&arena, x);

        // sin(100*x)
        let hundred = arena.int(100);
        let inner = arena.mul(&[hundred, x]);
        let expr = arena.sin(inner);

        let freq = estimate_frequency(&arena, expr, x, x_sym);

        assert!(freq.is_some(), "sin(100*x) should have a frequency");
        let omega = freq.unwrap();
        assert!(
            (omega - 100.0).abs() < 1e-10,
            "sin(100*x) angular frequency should be 100 rad/s, got: {}",
            omega
        );
    }

    // ── Test 8: frequency_no_trig ──────────────────────────────────

    #[test]
    fn frequency_no_trig() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let x_sym = sym_id(&arena, x);

        // x^2 + 1
        let two = arena.int(2);
        let x2 = arena.pow(x, two);
        let one = arena.one;
        let expr = arena.add(&[x2, one]);

        let freq = estimate_frequency(&arena, expr, x, x_sym);

        assert!(
            freq.is_none(),
            "x^2 + 1 should have no frequency, got: {:?}",
            freq
        );
    }

    /// `hidden_zero` makes the claims `Ex::is_zero` makes: none for `zoo`,
    /// NaN, `±∞` (up to 0.49 the structural shortcut called `zoo·π` and
    /// NaN nonzero, on exact evidence), and none that takes `δ` of a
    /// non-real argument for 0 (`δ(i)` was 0, exactly); `δ` of a nonzero
    /// real is 0.
    #[test]
    fn hidden_zero_makes_no_claim_without_a_zero_question() {
        use crate::base::truth::Truth;
        let mut arena = Arena::new();
        let zoo = arena.complex_infinity;
        let zoo_pi = arena.mul(&[zoo, arena.pi]);
        let nan = arena.nan;
        let oo = arena.infinity;
        for c in [zoo, zoo_pi, nan, oo] {
            assert_eq!(hidden_zero(&mut arena, c).proven(), None);
        }
        let i = arena.i_unit;
        let delta_i = arena.dirac_delta(i);
        let one = arena.one;
        let delta_i_plus_one = arena.add(&[delta_i, one]);
        for c in [delta_i, delta_i_plus_one] {
            assert_eq!(hidden_zero(&mut arena, c).proven(), None);
        }
        let two = arena.int(2);
        let delta_two = arena.dirac_delta(two);
        assert_eq!(
            hidden_zero(&mut arena, delta_two).proven(),
            Some(Truth::True)
        );
    }
}
