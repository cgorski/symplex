//! Symbolic domain analysis utilities.
//!
//! This module provides functions for analysing the domain of continuity,
//! locating singularities, and estimating oscillation frequency of symbolic
//! expressions.
//!
//! All traversals use explicit stacks (no recursion) following symplex
//! Principle 5.

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode, INTERVAL_BOTH_OPEN, SymbolId};
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
    let solutions = solve::solve(arena, expr, var);
    if solutions.is_empty() {
        // No zeros found → no restriction (denominator never zero for real x)
        return arena.interval(arena.neg_infinity, arena.infinity, INTERVAL_BOTH_OPEN);
    }

    let root_ids: Vec<ExprId> = solutions.into_iter().map(|s| s.value).collect();
    let roots_set = arena.finite_set(&root_ids);
    let reals = arena.interval(arena.neg_infinity, arena.infinity, INTERVAL_BOTH_OPEN);
    arena.set_complement(reals, roots_set)
}

// ═══════════════════════════════════════════════════════════════════════════
// singularities
// ═══════════════════════════════════════════════════════════════════════════

/// Returns the x-locations of singularities (poles, branch points) in
/// the given numeric range.
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
    range: (f64, f64),
) -> Vec<f64> {
    tracing::debug!(
        range_lo = range.0,
        range_hi = range.1,
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
            && val >= range.0
            && val <= range.1
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
/// the range; without a range only the solver's representative roots are
/// reported and `complete` is cleared for such families.
///
/// Uses an explicit post-order walk (no recursion).
pub(crate) fn scan_breakpoints(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    range: Option<(f64, f64)>,
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
    range: Option<(f64, f64)>,
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

    let solutions = solve::solve(arena, g, var);
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
        push_point(arena, s.value, kind, scan);
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
        let mut nonreal = false;
        let mut stack = vec![point];
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
    range: Option<(f64, f64)>,
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
    let (alpha, beta) = match linear_coeffs_f64(arena, inner, var) {
        Some(ab) => ab,
        None => return Some(false),
    };
    let (lo, hi) = match range {
        Some(r) => r,
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
    let k_min = ((t_min - offset) / std::f64::consts::PI).floor() as i64 - 1;
    let k_max = ((t_max - offset) / std::f64::consts::PI).ceil() as i64 + 1;
    if k_max - k_min > 10_000 {
        return Some(false);
    }
    // Build exact symbolic points: x = ((offset_frac + k)·π − β)/α.
    let (alpha_ex, beta_ex) = match linear_coeffs_exact(arena, inner, var) {
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

/// Numeric `(α, β)` for `expr = α·var + β` with constant coefficients.
fn linear_coeffs_f64(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<(f64, f64)> {
    let (a, b) = linear_coeffs_exact(arena, expr, var)?;
    let af = expr_to_f64(arena, a)?;
    let bf = expr_to_f64(arena, b)?;
    if af.is_finite() && bf.is_finite() {
        Some((af, bf))
    } else {
        None
    }
}

/// Exact `(α, β)` expressions for `expr = α·var + β`, where both
/// coefficients are free of `var`.
pub(crate) fn linear_coeffs_exact(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Option<(ExprId, ExprId)> {
    let coeffs = poly_coeffs_symbolic(arena, expr, var)?;
    if coeffs.len() != 2 {
        return None;
    }
    Some((coeffs[1], coeffs[0]))
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
    range: Option<(f64, f64)>,
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
        ExprNode::RootOf(..) | ExprNode::RootSum(..) | ExprNode::LambertW(_)
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
    range: Option<(f64, f64)>,
    kind: BreakKind,
    scan: &mut BreakScan,
    keep: impl Fn(i64) -> bool,
) {
    let (Some((alpha, beta)), Some((alpha_ex, beta_ex)), Some((lo, hi))) = (
        linear_coeffs_f64(arena, inner, var),
        linear_coeffs_exact(arena, inner, var),
        range,
    ) else {
        scan.complete = false;
        return;
    };
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
    let n_min = t_min.floor() as i64 - 1;
    let n_max = t_max.ceil() as i64 + 1;
    if n_max - n_min > 10_000 {
        scan.complete = false;
        return;
    }
    for n in n_min..=n_max {
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
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::arena::Arena;
    use crate::base::node::{ExprNode, INTERVAL_BOTH_CLOSED, INTERVAL_BOTH_OPEN};
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
        let reals = arena.interval(arena.neg_infinity, arena.infinity, INTERVAL_BOTH_OPEN);

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

        let reals = arena.interval(arena.neg_infinity, arena.infinity, INTERVAL_BOTH_OPEN);
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

        let reals = arena.interval(arena.neg_infinity, arena.infinity, INTERVAL_BOTH_OPEN);
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
        let domain = arena.interval(neg5, five, INTERVAL_BOTH_CLOSED);

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

        let sings = singularities(&mut arena, expr, x, x_sym, (0.0, 5.0));

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

        let sings = singularities(&mut arena, expr, x, x_sym, (-2.0, 2.0));

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
