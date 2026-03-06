//! Symbolic domain analysis utilities.
//!
//! This module provides functions for analysing the domain of continuity,
//! locating singularities, and estimating oscillation frequency of symbolic
//! expressions.
//!
//! All traversals use explicit stacks (no recursion) following symplex
//! Principle 5.

use crate::arena::Arena;
use crate::eval;
use crate::evalf;
use crate::inequalities::Relation;
use crate::node::{ExprId, ExprNode, SymbolId, INTERVAL_BOTH_OPEN};
use crate::solve;
use crate::walk;

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
fn compute_node_constraint(
    arena: &mut Arena,
    node: &ExprNode,
    var: ExprId,
) -> Option<ExprId> {
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
    arena
        .solve_inequality_expr(expr, var, Relation::Gt)
        .ok()
}

/// Solve `expr >= 0` for `var`, returning the solution set.
#[allow(dead_code)]
fn solve_ge_zero(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<ExprId> {
    arena
        .solve_inequality_expr(expr, var, Relation::Ge)
        .ok()
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
        return arena.interval(
            arena.neg_infinity,
            arena.infinity,
            INTERVAL_BOTH_OPEN,
        );
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
/// numerically in the given range.
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

    let post_order = walk::post_order_ids(arena, expr);
    let mut sing_points: Vec<f64> = Vec::new();

    for &id in &post_order {
        let node = arena.node(id).clone();
        let candidates = singularity_candidates(arena, &node, var);
        for root_expr in candidates {
            if let Some(val) = expr_to_f64(arena, root_expr)
                && val.is_finite() && val >= range.0 && val <= range.1
            {
                // Deduplicate
                if !sing_points.iter().any(|&v| (v - val).abs() < 1e-12) {
                    sing_points.push(val);
                }
            }
        }
    }

    sing_points.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sing_points
}

/// Collect candidate singularity locations (as ExprIds) from a single node.
fn singularity_candidates(
    arena: &mut Arena,
    node: &ExprNode,
    var: ExprId,
) -> Vec<ExprId> {
    match node {
        // Negative powers → zeros of base are poles
        ExprNode::Pow(base, exp) => {
            let base = *base;
            let exp = *exp;
            if !walk::contains(arena, base, var) {
                return Vec::new();
            }
            let is_neg = if let Some(r) = arena.as_num(exp).cloned() {
                use num_traits::Signed;
                r.is_negative()
            } else {
                matches!(arena.node(exp), ExprNode::Neg(_))
            };
            if is_neg {
                solve::solve(arena, base, var)
                    .into_iter()
                    .map(|s| s.value)
                    .collect()
            } else {
                Vec::new()
            }
        }

        // ln(inner) → singularity at inner = 0 boundary
        ExprNode::Ln(inner) => {
            let inner = *inner;
            if !walk::contains(arena, inner, var) {
                return Vec::new();
            }
            solve::solve(arena, inner, var)
                .into_iter()
                .map(|s| s.value)
                .collect()
        }

        // tan(inner) → singularities where cos(inner) = 0
        ExprNode::Tan(inner) => {
            let inner = *inner;
            if !walk::contains(arena, inner, var) {
                return Vec::new();
            }
            let cos_inner = arena.cos(inner);
            solve::solve(arena, cos_inner, var)
                .into_iter()
                .map(|s| s.value)
                .collect()
        }

        _ => Vec::new(),
    }
}

/// Try to evaluate an ExprId to f64.
fn expr_to_f64(arena: &mut Arena, expr: ExprId) -> Option<f64> {
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

        ExprNode::Neg(inner) => {
            extract_linear_coefficient(arena, *inner, var).map(|c| -c)
        }

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
    use crate::arena::Arena;
    use crate::node::{ExprNode, INTERVAL_BOTH_CLOSED, INTERVAL_BOTH_OPEN};
    use std::f64::consts::PI;

    /// Helper: get the SymbolId for an ExprId that is a Symbol.
    fn sym_id(arena: &Arena, expr: ExprId) -> SymbolId {
        match arena.node(expr) {
            ExprNode::Symbol(sid) => *sid,
            _ => panic!("expected Symbol node"),
        }
    }

    /// Helper: evaluate an ExprId to f64 for assertions.
    fn to_f64(arena: &mut Arena, expr: ExprId) -> Option<f64> {
        expr_to_f64(arena, expr)
    }

    /// Helper: check if a set (as displayed) contains an interval description.
    fn set_display(arena: &Arena, set: ExprId) -> String {
        // Use the arena's display infrastructure via a minimal formatter
        use crate::display;
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

        assert_eq!(sings.len(), 1, "1/x should have one singularity: {:?}", sings);
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
