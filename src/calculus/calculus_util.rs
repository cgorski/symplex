//! Symbolic domain analysis utilities.
//!
//! This module provides functions for analysing the domain of continuity,
//! locating singularities, and estimating oscillation frequency of symbolic
//! expressions.
//!
//! All traversals use explicit stacks (no recursion) following symplex
//! Principle 5.

use crate::base::arena::Arena;
use crate::base::cast::integer_span;
use crate::base::interval::Interval;
use crate::base::interval::IntervalKind;
use crate::base::node::{ExprId, ExprNode, SymbolId};
use crate::base::walk;
use crate::transforms::eval;
use crate::transforms::evalf;
use crate::transforms::inequalities::Relation;
use crate::transforms::solve;

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

    let scan = scan_breakpoints(arena, expr, var, Some(range));
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
            // Deduplicate
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
    /// Numeric value when the location is a real constant.
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
}

/// Scan `expr` for every point where it (or an antiderivative of it) may
/// fail to be smooth with respect to `var`.
///
/// When `range` is given, periodic families (`sin`, `cos`, `tan` with a
/// linear argument, `Gamma` poles, `floor` steps) are enumerated inside
/// the closed range `[lower, upper]`; without a range only the solver's
/// representative roots are reported and `complete` is cleared for such
/// families.
///
/// Uses an explicit post-order walk (no recursion).
pub(crate) fn scan_breakpoints(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    range: Option<Interval<f64>>,
) -> BreakScan {
    let post_order = walk::post_order_ids(arena, expr);
    let mut scan = BreakScan {
        points: Vec::new(),
        complete: true,
        opaque: false,
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
    range: Option<Interval<f64>>,
    scan: &mut BreakScan,
) {
    if !walk::contains(arena, g, var) {
        return;
    }

    // Periodic families first: sin/cos/tan of a linear argument.
    if let Some(done) = push_trig_zeros(arena, g, var, kind, range, scan) {
        if !done {
            scan.complete = false;
        }
        return;
    }

    let solutions = solve::known_real_roots(arena, g, var);
    if solutions.is_empty() {
        // A polynomial with no real roots is fine (solve returns complex
        // roots explicitly).  Anything the solver silently gave up on is
        // a hole in our knowledge.
        if crate::poly::polybridge::expr_to_poly(arena, g, var).is_none() {
            scan.complete = false;
        }
        return;
    }
    for s in solutions {
        push_point(arena, s, kind, scan);
    }
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
        if cache.query(arena, point, crate::base::assumptions::Props::REAL) == Some(false) {
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
                        && cache.query(arena, b, crate::base::assumptions::Props::NEGATIVE)
                            == Some(true)
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
        if nonreal && cache.query(arena, point, crate::base::assumptions::Props::REAL) != Some(true)
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

/// If `g` is `sin(αx+β)`, `cos(αx+β)` or `tan(αx+β)` with numeric α, β,
/// enumerate its zeros in `range`.  Returns `None` if `g` is not such a
/// form, `Some(true)` on success, `Some(false)` if the family could not
/// be enumerated (no range or non-numeric coefficients).
fn push_trig_zeros(
    arena: &mut Arena,
    g: ExprId,
    var: ExprId,
    kind: BreakKind,
    range: Option<Interval<f64>>,
    scan: &mut BreakScan,
) -> Option<bool> {
    // offset_k: zeros are at αx+β = offset + kπ
    let (inner, offset) = match arena.node(g).clone() {
        ExprNode::Sin(i) | ExprNode::Tan(i) => (i, 0.0),
        ExprNode::Cos(i) => (i, std::f64::consts::FRAC_PI_2),
        _ => return None,
    };
    if !walk::contains(arena, inner, var) {
        return None;
    }
    let LinearCoeffs {
        slope: alpha,
        intercept: beta,
    } = match linear_coeffs_f64(arena, inner, var) {
        Some(ab) => ab,
        None => return Some(false),
    };
    let (lo, hi) = match range {
        Some(r) => (r.lower, r.upper),
        None => return Some(false),
    };
    if alpha == 0.0 || !lo.is_finite() || !hi.is_finite() {
        return Some(false);
    }
    // x = (offset + kπ − β)/α.  Enumerate k so that x ∈ [lo, hi].
    let t_lo = alpha * lo + beta;
    let t_hi = alpha * hi + beta;
    let (t_min, t_max) = if t_lo <= t_hi {
        (t_lo, t_hi)
    } else {
        (t_hi, t_lo)
    };
    // `as i64` saturates for a huge range (`[−10³⁰⁰, 10³⁰⁰]`); the bounds and
    // their difference saturate too (up to 0.45 they overflowed: a panic in
    // debug builds, a wrapped count in release).
    let k_min = (((t_min - offset) / std::f64::consts::PI).floor() as i64).saturating_sub(1);
    let k_max = (((t_max - offset) / std::f64::consts::PI).ceil() as i64).saturating_add(1);
    if k_max.saturating_sub(k_min) > 10_000 {
        return Some(false);
    }
    // Build exact symbolic points: x = ((offset_frac + k)·π − β)/α.
    let LinearCoeffs {
        slope: alpha_ex,
        intercept: beta_ex,
    } = match linear_coeffs_exact(arena, inner, var) {
        Some(ab) => ab,
        None => return Some(false),
    };
    for k in k_min..=k_max {
        let t = offset + (k as f64) * std::f64::consts::PI;
        let xv = (t - beta) / alpha;
        if xv < lo - 1e-12 || xv > hi + 1e-12 {
            continue;
        }
        // Symbolic: ((k + offset/π)·π − β)/α
        let k_ex = if offset == 0.0 {
            arena.int(k)
        } else {
            arena.rational(2 * k + 1, 2)
        };
        let pi = arena.pi();
        let k_pi = arena.mul(&[k_ex, pi]);
        let num = arena.sub(k_pi, beta_ex);
        let point = arena.div(num, alpha_ex);
        scan.points.push(Breakpoint {
            point,
            value: Some(xv),
            kind,
        });
    }
    Some(true)
}

/// The coefficients of a linear expression `slope·var + intercept`, as
/// returned by [`linear_coeffs_exact`] (exact `ExprId`s) and
/// [`linear_coeffs_f64`] (numeric).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LinearCoeffs<T> {
    /// Coefficient of `var` (`α` in `α·var + β`).
    pub slope: T,
    /// Constant term (`β` in `α·var + β`).
    pub intercept: T,
}

/// Numeric `α·var + β` coefficients for `expr` with constant coefficients.
fn linear_coeffs_f64(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<LinearCoeffs<f64>> {
    let lin = linear_coeffs_exact(arena, expr, var)?;
    let af = expr_to_f64(arena, lin.slope)?;
    let bf = expr_to_f64(arena, lin.intercept)?;
    if af.is_finite() && bf.is_finite() {
        Some(LinearCoeffs {
            slope: af,
            intercept: bf,
        })
    } else {
        None
    }
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
/// of the form `c · var^k` with `k` a non-negative integer, or if the
/// polynomial is identically zero (the empty coefficient list would be
/// ambiguous).
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
    range: Option<Interval<f64>>,
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
            push_integer_family(arena, inner, var, range, Singular, scan, |n| n <= 0);
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
            push_integer_family(arena, inner, var, range, Kink, scan, |_| true);
        }

        // Relational conditions (from Piecewise): boundaries where a = b.
        ExprNode::Gt(a, b) | ExprNode::Ge(a, b) | ExprNode::Eq_(a, b) | ExprNode::Ne(a, b) => {
            let d = arena.sub(*a, *b);
            push_zeros(arena, d, var, Kink, range, scan);
        }

        // Unknown functions and formal nodes: we cannot see inside.
        ExprNode::Apply(_, args) if args.iter().any(|&a| walk::contains(arena, a, var)) => {
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
/// integer values selected by `keep`, restricted to `range`.
fn push_integer_family(
    arena: &mut Arena,
    inner: ExprId,
    var: ExprId,
    range: Option<Interval<f64>>,
    kind: BreakKind,
    scan: &mut BreakScan,
    keep: impl Fn(i64) -> bool,
) {
    let (Some(lin), Some(lin_ex), Some(r)) = (
        linear_coeffs_f64(arena, inner, var),
        linear_coeffs_exact(arena, inner, var),
        range,
    ) else {
        scan.complete = false;
        return;
    };
    let (alpha, beta) = (lin.slope, lin.intercept);
    let (alpha_ex, beta_ex) = (lin_ex.slope, lin_ex.intercept);
    let (lo, hi) = (r.lower, r.upper);
    if alpha == 0.0 || !lo.is_finite() || !hi.is_finite() {
        scan.complete = false;
        return;
    }
    let t_lo = alpha * lo + beta;
    let t_hi = alpha * hi + beta;
    let (t_min, t_max) = if t_lo <= t_hi {
        (t_lo, t_hi)
    } else {
        (t_hi, t_lo)
    };
    // At most 10 001 candidates (`n_max − n_min ≤ 10 000`).  Before 0.49 the
    // `as i64` casts saturated and the `± 1` and the length wrapped: the
    // jumps of `floor(10¹⁹x)` on `[−1, 1]` were skipped (integral 0) and
    // `floor(9·10¹⁸x)` looped over 1.8·10¹⁹ candidates.
    let Ok(span) = integer_span(t_min, t_max, 1, 10_001) else {
        scan.complete = false;
        return;
    };
    for n in span {
        if !keep(n) {
            continue;
        }
        let xv = ((n as f64) - beta) / alpha;
        if xv < lo - 1e-12 || xv > hi + 1e-12 {
            continue;
        }
        let n_ex = arena.int(n);
        let num = arena.sub(n_ex, beta_ex);
        let point = arena.div(num, alpha_ex);
        let point = eval::eval(arena, point);
        scan.points.push(Breakpoint {
            point,
            value: Some(xv),
            kind,
        });
    }
}

/// Try to evaluate an ExprId to f64.
pub(crate) fn expr_to_f64(arena: &mut Arena, expr: ExprId) -> Option<f64> {
    let evaled = eval::eval(arena, expr);
    // Try exact rational first
    if let Some(r) = arena.as_num(evaled).cloned() {
        use num_traits::ToPrimitive;
        return r.to_f64();
    }
    // Fall back to numerical evaluation
    let s = evalf::evalf(arena, evaled, 16).ok()?;
    if s.contains('I') || s.contains('i') {
        return None;
    }
    s.trim().parse::<f64>().ok()
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
                    None => {
                        assumptions.query(arena, exp, crate::base::assumptions::Props::NEGATIVE)
                            == Some(true)
                    }
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
        ExprNode::Apply(..) => "an unknown function",
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
    let den = if assumptions.query(arena, a, crate::base::assumptions::Props::POSITIVE)
        == Some(true)
    {
        a
    } else if assumptions.query(arena, a, crate::base::assumptions::Props::NEGATIVE) == Some(true) {
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
///   [`ZeroTest::Nonzero`] for how the series engine reads that).
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
pub(crate) fn hidden_zero(
    arena: &mut Arena,
    c: ExprId,
) -> crate::base::truth::Verdict<crate::base::truth::Truth> {
    use crate::base::truth::{Evidence, Truth, Verdict};
    if let Some(r) = arena.as_num(c) {
        return Verdict::new(
            Truth::from_bool(num_traits::Zero::is_zero(r)),
            Evidence::Exact,
        );
    }
    if !walk::free_symbols(arena, c).is_empty() || walk::has_unevaluated(arena, c) {
        return Verdict::new(Truth::Unknown, Evidence::Approximate);
    }
    // `evalf` takes `δ(x)` for 0 everywhere, `δ(0)` included: no value,
    // but `δ(a)` of a nonzero `a` is 0 ([`dirac_delta_off_zero`]).
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
    if constant && exact_is_zero(arena, c) == Some(true) {
        return true;
    }
    if crate::simplify::ratsimp::vanishes_as_rational_function(arena, c) {
        return true;
    }
    // Multiplied out, the transcendental parts may cancel and leave an
    // algebraic number (`(a − ζ(3))·i + (b + ζ(3))·i`, `a + b = 0`).
    let expanded = crate::transforms::expand::expand(arena, c);
    if arena.is_zero_structural(expanded)
        || (constant && expanded != c && exact_is_zero(arena, expanded) == Some(true))
    {
        return true;
    }
    if rewritten == c {
        return false;
    }
    (constant && exact_is_zero(arena, rewritten) == Some(true))
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
/// - the logarithm of a positive rational as a sum of logarithms of its
///   prime factors, `ln 16 = 4·ln 2`, `ln(3/2) = ln 3 − ln 2`;
/// - the square root of a polynomial with its nonnegative square factors
///   taken out, `√(a² + 2a + 1) = a + 1` for `a > 0` ([`sqrt_of_squares`]);
/// - `Γ(p)·Γ(1 − p) = π/sin(πp)` ([`gamma_reflection`]), `exp(a)·exp(b) =
///   exp(a + b)` ([`exponentials_combined`]), Bessel functions of integer
///   order and `Eₙ` of order `n ≤ 0` in a normal form
///   ([`special_function_rewrite`]), `atan2(0, a)` for a real `a ≠ 0`,
///   `δ(a) = 0` for `a ≠ 0` ([`dirac_delta_off_zero`]), a binomial
///   coefficient that evaluates to exactly 0 ([`certified_zero_value`]);
/// - a function argument with a quotient in it reduced as a rational
///   function of its generators ([`ratsimp`](crate::simplify::ratsimp::ratsimp)):
///   `cos((π − 12π²)/(24π − 2)) = cos(−π/2) = 0`.
pub(crate) fn exact_rewrites(arena: &mut Arena, c: ExprId) -> ExprId {
    if walk::post_order_ids(arena, c).len() > MAX_EXACT_ZERO_SIZE {
        return c;
    }
    let r = compositions_pass(arena, c);
    let r = exact_pass(arena, r);
    let logs = crate::calculus::limit::inverse_functions_as_logs(arena, r);
    if logs == r {
        r
    } else {
        exact_pass(arena, logs)
    }
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
            ExprNode::Apply(f, args) if args.len() == 2 => {
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

/// Does `c` contain `δ`?
fn has_dirac_delta(arena: &Arena, c: ExprId) -> bool {
    walk::post_order_ids(arena, c)
        .into_iter()
        .any(|id| matches!(arena.node(id), ExprNode::DiracDelta(_)))
}

/// `0` for `δ(a)` when `a` is not 0: a constant with a certified nonzero
/// digit, or an expression in parameters that is nonzero at a sample point
/// (`δ(y) = 0` for generic `y`, the generic-parameter reading of the limit
/// and series engines).  `None` otherwise.
fn dirac_delta_off_zero(arena: &mut Arena, a: ExprId) -> Option<ExprId> {
    let nonzero = if walk::free_symbols(arena, a).is_empty() {
        !has_dirac_delta(arena, a)
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
    f: crate::base::node::SymbolId,
    order: ExprId,
    z: ExprId,
) -> Option<ExprId> {
    use crate::base::libfn::LibFn;
    use num_traits::ToPrimitive;
    let lib = arena.lib_fn(f)?;
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
        if m >= 2 && cache.query(arena, f, Props::NONNEGATIVE) == Some(true) {
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
    if !walk::free_symbols(arena, a).is_empty() {
        return false;
    }
    let mut cache = crate::base::assumptions::AssumptionCache::new();
    if cache.query(arena, a, crate::base::assumptions::Props::REAL) != Some(true) {
        return false;
    }
    let Some(z) = evalf::evalf_verdict(arena, a, HIDDEN_ZERO_DIGITS, evalf::ZeroSearch::Cap)
        .ok()
        .and_then(crate::base::truth::Verdict::proven)
    else {
        return false;
    };
    let Some(v) = crate::base::bigfloat::to_f64(&z.0) else {
        return false;
    };
    // 30 certified digits: a margin far above the error of `v`.
    let margin = 1e-9 * v.abs().max(1.0);
    v > lo + margin && v < hi - margin
}

/// A constant after the zero test of [`settle_constant`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ZeroTest {
    /// Proved 0, exactly or by a certified evaluation.
    Zero,
    /// The evaluated constant, not 0: proved nonzero, or a constant with
    /// free parameters, which is taken as nonzero for generic values of
    /// them (the generic-solution convention; [`settle_parametric`]
    /// recognises the ones that vanish identically).  So is a constant
    /// without a value — one `evalf` cannot evaluate or with an
    /// unevaluated part (`legendre(3/2, e) − 1`): it is opaque like a
    /// parameter, and there is no numerical hint that it vanishes.
    ///
    /// [`settle_parametric`]: crate::calculus::series::settle_parametric
    Nonzero(ExprId),
    /// The evaluated constant, 0 to the precision reached (at the sample
    /// points of its parameters, if any) but proved neither 0 nor nonzero:
    /// a caller may keep it as a value, but may not divide by it, take it
    /// for a leading coefficient, or drop it.
    Undecided(ExprId),
}

impl ZeroTest {
    /// `True` for `Zero`, `False` for `Nonzero`, `Unknown` for `Undecided`.
    pub(crate) fn is_zero(self) -> crate::base::truth::Truth {
        use crate::base::truth::Truth;
        match self {
            ZeroTest::Zero => Truth::True,
            ZeroTest::Nonzero(_) => Truth::False,
            ZeroTest::Undecided(_) => Truth::Unknown,
        }
    }

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
/// [`ZeroTest::Nonzero`]).
pub(crate) fn settle_constant(arena: &mut Arena, c: ExprId) -> ZeroTest {
    let v = eval::eval(arena, c);
    if arena.is_zero_structural(v) {
        return ZeroTest::Zero;
    }
    if !walk::free_symbols(arena, v).is_empty() {
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
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::arena::Arena;
    use crate::base::node::ExprNode;
    use std::f64::consts::PI;

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
}
