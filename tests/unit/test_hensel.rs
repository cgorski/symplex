//! Integration tests for polynomial factoring over ℤ.
//!
//! Tests the enhanced factoring pipeline: square-free decomposition,
//! rational root theorem, and Kronecker's method for higher-degree factors.

use super::common;

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helper: verify that factored form equals original at several points
// ═══════════════════════════════════════════════════════════════════════════

fn assert_values_match(original: &Ex, factored: &Ex, var: &Ex, label: &str) {
    for &pt in &[-5i64, -3, -2, -1, 0, 1, 2, 3, 5, 7] {
        let vo = common::eval_at_i64(original, var, pt);
        let vf = common::eval_at_i64(factored, var, pt);
        assert!(
            common::approx_eq(vo, vf, 1e-9),
            "{label}: value mismatch at {var}={pt}: original={vo}, factored={vf}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Difference of squares
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_difference_of_squares() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x² − 1 = (x − 1)(x + 1)
    let expr = &x.powi(2) - 1;
    let factored = expr.factor(&x);
    let s = format!("{factored}");
    assert!(!s.contains("x^2"), "should be factored (no x^2): {s}");
    assert_values_match(&expr, &factored, &x, "x²−1");
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. x⁴ − 1 → (x−1)(x+1)(x²+1)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_x4_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(4) - 1;
    let factored = expr.factor(&x);
    let s = format!("{factored}");
    assert!(!s.contains("x^4"), "should be factored (no x^4): {s}");
    assert!(!s.contains("x^3"), "should be factored (no x^3): {s}");
    // The x²+1 factor is irreducible over ℤ, so x^2 may appear inside
    // a factor — but not as the top-level polynomial.
    assert_values_match(&expr, &factored, &x, "x⁴−1");
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. x⁶ − 1 → (x−1)(x+1)(x²−x+1)(x²+x+1)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_x6_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(6) - 1;
    let factored = expr.factor(&x);
    let s = format!("{factored}");
    assert!(!s.contains("x^6"), "should be factored (no x^6): {s}");
    assert!(!s.contains("x^5"), "should be factored (no x^5): {s}");
    assert!(!s.contains("x^4"), "should be factored (no x^4): {s}");
    assert!(!s.contains("x^3"), "should be factored (no x^3): {s}");
    assert_values_match(&expr, &factored, &x, "x⁶−1");
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Perfect square: x² + 2x + 1 = (x+1)²
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_perfect_square() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(2) + &x * 2 + 1;
    let factored = expr.factor(&x);
    let s = format!("{factored}");
    // Should not have bare x^2 as the top-level expression.
    assert!(!s.contains("x^2 +"), "should be factored: {s}");
    assert!(!s.contains("+ x^2"), "should be factored: {s}");
    assert_values_match(&expr, &factored, &x, "(x+1)²");
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Cubic with three rational roots: x³ − 6x² + 11x − 6
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_cubic_with_rational_roots() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x³ − 6x² + 11x − 6 = (x−1)(x−2)(x−3)
    let expr = &x.powi(3) - &x.powi(2) * 6 + &x * 11 - 6;
    let factored = expr.factor(&x);
    let s = format!("{factored}");
    assert!(!s.contains("x^3"), "should be fully factored (no x^3): {s}");
    assert!(!s.contains("x^2"), "should be fully factored (no x^2): {s}");
    assert_values_match(&expr, &factored, &x, "cubic");
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Quadratic in x²: x⁴ + 5x² + 6 = (x²+2)(x²+3)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_quadratic_in_x_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x⁴ + 5x² + 6 — no rational roots, but factors into two quadratics.
    let expr = &x.powi(4) + &x.powi(2) * 5 + 6;
    let factored = expr.factor(&x);
    let s = format!("{factored}");
    assert!(!s.contains("x^4"), "should be factored (no x^4): {s}");
    assert_values_match(&expr, &factored, &x, "x⁴+5x²+6");
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Content extraction: 6x² + 12x + 6 = 6·(x+1)²
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_content_extraction() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(2) * 6 + &x * 12 + 6;
    let factored = expr.factor(&x);
    let s = format!("{factored}");
    // Should not contain a raw x^2 term in a sum.
    assert!(
        !s.contains("x^2 +") && !s.contains("+ x^2"),
        "should be factored: {s}"
    );
    assert_values_match(&expr, &factored, &x, "6x²+12x+6");
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Factored form preserves value at many test points
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_preserves_value() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let cases: Vec<Ex> = vec![
        // x² − 1
        &x.powi(2) - 1,
        // x⁴ − 1
        &x.powi(4) - 1,
        // x³ − 6x² + 11x − 6
        &x.powi(3) - &x.powi(2) * 6 + &x * 11 - 6,
        // x⁴ + 5x² + 6
        &x.powi(4) + &x.powi(2) * 5 + 6,
        // 2x² − 2
        &x.powi(2) * 2 - 2,
        // x² + 2x + 1
        &x.powi(2) + &x * 2 + 1,
    ];

    for (i, expr) in cases.iter().enumerate() {
        let factored = expr.factor(&x);
        assert_values_match(expr, &factored, &x, &format!("case {i}"));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. Irreducible polynomial stays unfactored
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_irreducible_stays() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x² + 1 is irreducible over ℤ.
    let expr = &x.powi(2) + 1;
    let factored = expr.factor(&x);
    let s = format!("{factored}");
    // Should still have x^2 — no factoring possible.
    assert!(s.contains("x^2"), "should remain unfactored: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Linear polynomial with content: 2x + 4 → 2·(x + 2)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_linear() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x * 2 + 4;
    let factored = expr.factor(&x);
    // The value must be preserved at all test points.
    assert_values_match(&expr, &factored, &x, "2x+4");
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional coverage
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_double_root_x2_minus_2x_plus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x² − 2x + 1 = (x − 1)²
    let expr = &x.powi(2) - &x * 2 + 1;
    let factored = expr.factor(&x);
    let s = format!("{factored}");
    assert!(
        !s.contains("x^2 +") && !s.contains("x^2 -") && !s.contains("- x^2"),
        "should be factored into (x-1)^2: {s}"
    );
    assert_values_match(&expr, &factored, &x, "(x-1)²");
}

#[test]
fn factor_with_content_2x2_minus_2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 2x² − 2 = 2·(x−1)(x+1)
    let expr = &x.powi(2) * 2 - 2;
    let factored = expr.factor(&x);
    let s = format!("{factored}");
    assert!(!s.contains("x^2"), "should be factored: {s}");
    assert_values_match(&expr, &factored, &x, "2x²−2");
}

#[test]
fn factor_non_polynomial_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin();
    let factored = expr.factor(&x);
    assert_eq!(format!("{factored}"), "sin(x)");
}

#[test]
fn factor_constant_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let factored = ctx.int(42).factor(&x);
    assert_eq!(format!("{factored}"), "42");
}

#[test]
fn factor_already_linear_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x + 1;
    let factored = expr.factor(&x);
    assert_eq!(format!("{factored}"), "x + 1");
}

#[test]
fn factor_x4_plus_x2_plus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x⁴ + x² + 1 = (x² + x + 1)(x² − x + 1)
    let expr = &x.powi(4) + &x.powi(2) + 1;
    let factored = expr.factor(&x);
    let s = format!("{factored}");
    assert!(!s.contains("x^4"), "should be factored (no x^4): {s}");
    assert_values_match(&expr, &factored, &x, "x⁴+x²+1");
}

#[test]
fn factor_difference_of_cubes() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x³ − 1 = (x − 1)(x² + x + 1)
    let expr = &x.powi(3) - 1;
    let factored = expr.factor(&x);
    let s = format!("{factored}");
    assert!(!s.contains("x^3"), "should be factored (no x^3): {s}");
    assert_values_match(&expr, &factored, &x, "x³−1");
}

#[test]
fn factor_sum_of_cubes() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x³ + 1 = (x + 1)(x² − x + 1)
    let expr = &x.powi(3) + 1;
    let factored = expr.factor(&x);
    let s = format!("{factored}");
    assert!(!s.contains("x^3"), "should be factored (no x^3): {s}");
    assert_values_match(&expr, &factored, &x, "x³+1");
}

#[test]
fn factor_x_cubed_minus_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x³ − x = x(x−1)(x+1)
    let expr = &x.powi(3) - &x;
    let factored = expr.factor(&x);
    let s = format!("{factored}");
    assert!(!s.contains("x^3"), "should be factored (no x^3): {s}");
    assert_values_match(&expr, &factored, &x, "x³−x");
}

#[test]
fn factor_quartic_two_double_roots() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x − 1)²(x + 1)² = x⁴ − 2x² + 1
    let expr = &x.powi(4) - &x.powi(2) * 2 + 1;
    let factored = expr.factor(&x);
    let s = format!("{factored}");
    assert!(!s.contains("x^4"), "should be factored: {s}");
    assert_values_match(&expr, &factored, &x, "(x−1)²(x+1)²");
}
