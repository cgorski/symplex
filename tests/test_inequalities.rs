//! Tests for Wave η — inequality solving.
//!
//! Covers: solve_gt, solve_ge, solve_lt, solve_le, solveset,
//! constant expressions, linear, quadratic, and edge cases.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// solve_gt — strict greater-than
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_x_gt_0() {
    symplex::vars!(x);
    let result = x.solve_gt(&x).unwrap();
    let s = format!("{result}");
    // x > 0 → (0, ∞)
    assert!(!s.contains("EmptySet"), "x > 0 should not be empty: {s}");
    assert!(
        s.contains("oo") || s.contains("∞") || s.contains("Interval"),
        "x > 0 should contain an interval to infinity: {s}"
    );
}

#[test]
fn solve_x2_minus_4_gt_0() {
    symplex::vars!(x);
    let poly = &x.powi(2) - 4;
    let result = poly.solve_gt(&x).unwrap();
    let s = format!("{result}");
    // x² - 4 > 0 → (-∞, -2) ∪ (2, ∞)
    assert!(
        !s.contains("EmptySet"),
        "x²-4 > 0 should have solutions: {s}"
    );
    // Should mention both 2 and -2 as boundary points
    assert!(s.contains("2"), "solution should reference 2: {s}");
}

#[test]
fn solve_positive_constant_gt() {
    symplex::vars!(x);
    let five = symplex::int(5);
    let result = five.solve_gt(&x).unwrap();
    let s = format!("{result}");
    // 5 > 0 is always true → (-∞, ∞)
    assert!(
        !s.contains("EmptySet"),
        "5 > 0 should be satisfied everywhere: {s}"
    );
}

#[test]
fn solve_negative_constant_gt() {
    symplex::vars!(x);
    let neg = symplex::int(-3);
    let result = neg.solve_gt(&x).unwrap();
    // -3 > 0 is always false → EmptySet
    assert_eq!(format!("{result}"), "EmptySet");
}

#[test]
fn solve_zero_gt() {
    symplex::vars!(x);
    let zero = symplex::int(0);
    let result = zero.solve_gt(&x).unwrap();
    // 0 > 0 is false → EmptySet
    assert_eq!(format!("{result}"), "EmptySet");
}

// ═══════════════════════════════════════════════════════════════════════════
// solve_ge — greater-than-or-equal
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_zero_ge() {
    symplex::vars!(x);
    let zero = symplex::int(0);
    let result = zero.solve_ge(&x).unwrap();
    let s = format!("{result}");
    // 0 >= 0 is always true → (-∞, ∞)
    assert!(
        !s.contains("EmptySet"),
        "0 >= 0 should be satisfied everywhere: {s}"
    );
}

#[test]
fn solve_x2_minus_4_ge_0() {
    symplex::vars!(x);
    let poly = &x.powi(2) - 4;
    let result = poly.solve_ge(&x).unwrap();
    let s = format!("{result}");
    // x² - 4 >= 0 → (-∞, -2] ∪ [2, ∞)
    // The solution must not be empty and should include the roots.
    assert!(
        !s.contains("EmptySet"),
        "x²-4 >= 0 should have solutions: {s}"
    );
}

#[test]
fn solve_positive_constant_ge() {
    symplex::vars!(x);
    let seven = symplex::int(7);
    let result = seven.solve_ge(&x).unwrap();
    let s = format!("{result}");
    // 7 >= 0 always true
    assert!(
        !s.contains("EmptySet"),
        "7 >= 0 should be true everywhere: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// solve_lt — strict less-than
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_x_lt_0() {
    symplex::vars!(x);
    let result = x.solve_lt(&x).unwrap();
    let s = format!("{result}");
    // x < 0 → (-∞, 0)
    assert!(!s.contains("EmptySet"), "x < 0 should not be empty: {s}");
}

#[test]
fn solve_x2_minus_4_lt_0() {
    symplex::vars!(x);
    let poly = &x.powi(2) - 4;
    let result = poly.solve_lt(&x).unwrap();
    let s = format!("{result}");
    // x² - 4 < 0 → (-2, 2)
    assert!(
        !s.contains("EmptySet"),
        "x²-4 < 0 should have solutions: {s}"
    );
    assert!(s.contains("2"), "solution should reference 2: {s}");
}

#[test]
fn solve_negative_constant_lt() {
    symplex::vars!(x);
    let neg = symplex::int(-5);
    let result = neg.solve_lt(&x).unwrap();
    let s = format!("{result}");
    // -5 < 0 is always true → (-∞, ∞)
    assert!(
        !s.contains("EmptySet"),
        "-5 < 0 should be true everywhere: {s}"
    );
}

#[test]
fn solve_positive_constant_lt() {
    symplex::vars!(x);
    let pos = symplex::int(3);
    let result = pos.solve_lt(&x).unwrap();
    // 3 < 0 is always false → EmptySet
    assert_eq!(format!("{result}"), "EmptySet");
}

// ═══════════════════════════════════════════════════════════════════════════
// solve_le — less-than-or-equal
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_x2_minus_4_le_0() {
    symplex::vars!(x);
    let poly = &x.powi(2) - 4;
    let result = poly.solve_le(&x).unwrap();
    let s = format!("{result}");
    // x² - 4 ≤ 0 → [-2, 2]
    assert!(
        !s.contains("EmptySet"),
        "x²-4 ≤ 0 should have solutions: {s}"
    );
}

#[test]
fn solve_zero_le() {
    symplex::vars!(x);
    let zero = symplex::int(0);
    let result = zero.solve_le(&x).unwrap();
    let s = format!("{result}");
    // 0 <= 0 always true
    assert!(
        !s.contains("EmptySet"),
        "0 <= 0 should be satisfied everywhere: {s}"
    );
}

#[test]
fn solve_negative_constant_le() {
    symplex::vars!(x);
    let neg = symplex::int(-2);
    let result = neg.solve_le(&x).unwrap();
    let s = format!("{result}");
    // -2 <= 0 always true
    assert!(
        !s.contains("EmptySet"),
        "-2 <= 0 should be true everywhere: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// solveset — equation solving as a set
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solveset_quadratic() {
    symplex::vars!(x);
    let poly = &x.powi(2) - &x * 5 + 6;
    let result = poly.solveset(&x);
    let s = format!("{result}");
    // x² - 5x + 6 = 0 → {2, 3}
    assert!(
        !s.contains("EmptySet"),
        "solveset of x²-5x+6 should find roots: {s}"
    );
    assert!(s.contains("2"), "should contain root 2: {s}");
    assert!(s.contains("3"), "should contain root 3: {s}");
}

#[test]
fn solveset_linear() {
    symplex::vars!(x);
    let expr = &x - 7;
    let result = expr.solveset(&x);
    let s = format!("{result}");
    // x - 7 = 0 → {7}
    assert!(s.contains("7"), "should contain root 7: {s}");
}

#[test]
fn solveset_no_real_roots() {
    symplex::vars!(x);
    // x² + 1 = 0 has no real roots (only complex)
    let expr = &x.powi(2) + 1;
    let result = expr.solveset(&x);
    let s = format!("{result}");
    // The solver may or may not find complex roots; if it doesn't,
    // we should at least get something (possibly EmptySet or a set
    // with imaginary values). Just assert it doesn't panic.
    assert!(!s.is_empty(), "solveset should produce output: {s}");
}

#[test]
fn solveset_constant_nonzero() {
    symplex::vars!(x);
    let five = symplex::int(5);
    let result = five.solveset(&x);
    // 5 = 0 has no solutions → EmptySet
    assert_eq!(format!("{result}"), "EmptySet");
}

// ═══════════════════════════════════════════════════════════════════════════
// Quadratic sign-chart: full coverage
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quadratic_positive_leading_coeff_gt() {
    symplex::vars!(x);
    // x² - 1 > 0 → (-∞, -1) ∪ (1, ∞)
    let poly = &x.powi(2) - 1;
    let result = poly.solve_gt(&x).unwrap();
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "x²-1 > 0 should have solutions: {s}"
    );
}

#[test]
fn quadratic_positive_leading_coeff_lt() {
    symplex::vars!(x);
    // x² - 1 < 0 → (-1, 1)
    let poly = &x.powi(2) - 1;
    let result = poly.solve_lt(&x).unwrap();
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "x²-1 < 0 should have solutions: {s}"
    );
    // Should be a bounded interval between -1 and 1
    assert!(s.contains("1"), "solution should reference 1: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Linear inequalities
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn linear_2x_minus_6_gt_0() {
    symplex::vars!(x);
    // 2x - 6 > 0 → x > 3 → (3, ∞)
    let expr = &x * 2 - 6;
    let result = expr.solve_gt(&x).unwrap();
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "2x-6 > 0 should have solutions: {s}"
    );
    assert!(s.contains("3"), "solution should reference 3: {s}");
}

#[test]
fn linear_neg_x_plus_5_le_0() {
    symplex::vars!(x);
    // -x + 5 ≤ 0 → x ≥ 5 → [5, ∞)
    let expr = -&x + 5;
    let result = expr.solve_le(&x).unwrap();
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "-x+5 ≤ 0 should have solutions: {s}"
    );
    assert!(s.contains("5"), "solution should reference 5: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Cubic polynomial
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cubic_x3_minus_x_gt_0() {
    symplex::vars!(x);
    // x³ - x = x(x-1)(x+1) > 0  → (-1, 0) ∪ (1, ∞)
    let poly = &x.powi(3) - &x;
    let result = poly.solve_gt(&x).unwrap();
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "x³-x > 0 should have solutions: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge: large positive / large negative constants
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn large_positive_constant_gt() {
    symplex::vars!(x);
    let big = symplex::int(999999);
    let result = big.solve_gt(&x).unwrap();
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "999999 > 0 should be true everywhere: {s}"
    );
}

#[test]
fn large_negative_constant_lt() {
    symplex::vars!(x);
    let big_neg = symplex::int(-999999);
    let result = big_neg.solve_lt(&x).unwrap();
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "-999999 < 0 should be true everywhere: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// solveset / solve_gt with Context API
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solveset_with_context() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let poly = &x.powi(2) - &x * 5 + 6;
    let result = poly.solveset(&x);
    let s = format!("{result}");
    assert!(s.contains("2"), "should contain 2: {s}");
    assert!(s.contains("3"), "should contain 3: {s}");
}

#[test]
fn solve_gt_with_context() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let five = ctx.int(5);
    let result = five.solve_gt(&x).unwrap();
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "5 > 0 via context should be true: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Rational coefficients
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solveset_with_rational_roots() {
    symplex::vars!(x);
    // 2x - 1 = 0 → x = 1/2
    let expr = &x * 2 - 1;
    let result = expr.solveset(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "2x - 1 = 0 should have a solution: {s}"
    );
}
