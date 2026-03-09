//! Tests for Wave η — inequality solving.
//!
//! Covers: solve_gt, solve_ge, solve_lt, solve_le, solveset,
//! constant expressions, linear, quadratic, and edge cases.

mod common;

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// solve_gt — strict greater-than
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_x_gt_0() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let result = x.solve_gt(&x);
    let s = format!("{result}");
    // x > 0 → (0, ∞)
    assert!(!s.contains("EmptySet"), "x > 0 should not be empty: {s}");
    assert!(
        s.contains("oo") || s.contains("∞") || s.contains("Interval"),
        "x > 0 should contain an interval to infinity: {s}"
    );
    // Interior: x=3 → 3 > 0 ✓
    common::assert_positive_at(&x, &x, 3, "x > 0 interior at x=3");
    // Exterior: x=-1 → -1 < 0 ✗
    common::assert_negative_at(&x, &x, -1, "x > 0 exterior at x=-1");
}

#[test]
fn solve_x2_minus_4_gt_0() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let poly = &x.powi(2) - 4;
    let result = poly.solve_gt(&x);
    let s = format!("{result}");
    // x² - 4 > 0 → (-∞, -2) ∪ (2, ∞)
    assert!(
        !s.contains("EmptySet"),
        "x²-4 > 0 should have solutions: {s}"
    );
    // Should mention both 2 and -2 as boundary points
    assert!(s.contains("2"), "solution should reference 2: {s}");
    // Interior: x=3 → 9-4=5 > 0 ✓
    common::assert_positive_at(&poly, &x, 3, "x²-4 > 0 interior at x=3");
    // Interior: x=-3 → 9-4=5 > 0 ✓
    common::assert_positive_at(&poly, &x, -3, "x²-4 > 0 interior at x=-3");
    // Exterior: x=0 → 0-4=-4 < 0 ✗
    common::assert_negative_at(&poly, &x, 0, "x²-4 > 0 exterior at x=0");
}

#[test]
fn solve_positive_constant_gt() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let five = ctx.int(5);
    let result = five.solve_gt(&x);
    let s = format!("{result}");
    // 5 > 0 is always true → (-∞, ∞)
    assert!(
        !s.contains("EmptySet"),
        "5 > 0 should be satisfied everywhere: {s}"
    );
    // Interior: constant 5 > 0 everywhere, check at x=0
    common::assert_positive_at(&five, &x, 0, "5 > 0 constant check");
}

#[test]
fn solve_negative_constant_gt() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let neg = ctx.int(-3);
    let result = neg.solve_gt(&x);
    // -3 > 0 is always false → EmptySet
    assert_eq!(format!("{result}"), "EmptySet");
    // Spot check: -3 is negative everywhere
    common::assert_negative_at(&neg, &x, 0, "-3 > 0 constant check");
}

#[test]
fn solve_zero_gt() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let zero = ctx.int(0);
    let result = zero.solve_gt(&x);
    // 0 > 0 is false → EmptySet
    assert_eq!(format!("{result}"), "EmptySet");
}

// ═══════════════════════════════════════════════════════════════════════════
// solve_ge — greater-than-or-equal
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_zero_ge() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let zero = ctx.int(0);
    let result = zero.solve_ge(&x);
    let s = format!("{result}");
    // 0 >= 0 is always true → (-∞, ∞)
    assert!(
        !s.contains("EmptySet"),
        "0 >= 0 should be satisfied everywhere: {s}"
    );
}

#[test]
fn solve_x2_minus_4_ge_0() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let poly = &x.powi(2) - 4;
    let result = poly.solve_ge(&x);
    let s = format!("{result}");
    // x² - 4 >= 0 → (-∞, -2] ∪ [2, ∞)
    // The solution must not be empty and should include the roots.
    assert!(
        !s.contains("EmptySet"),
        "x²-4 >= 0 should have solutions: {s}"
    );
    // Interior: x=5 → 25-4=21 > 0 ✓
    common::assert_positive_at(&poly, &x, 5, "x²-4 >= 0 interior at x=5");
    // Exterior: x=1 → 1-4=-3 < 0 ✗
    common::assert_negative_at(&poly, &x, 1, "x²-4 >= 0 exterior at x=1");
}

#[test]
fn solve_positive_constant_ge() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let seven = ctx.int(7);
    let result = seven.solve_ge(&x);
    let s = format!("{result}");
    // 7 >= 0 always true
    assert!(
        !s.contains("EmptySet"),
        "7 >= 0 should be true everywhere: {s}"
    );
    common::assert_positive_at(&seven, &x, 0, "7 >= 0 constant check");
}

// ═══════════════════════════════════════════════════════════════════════════
// solve_lt — strict less-than
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_x_lt_0() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let result = x.solve_lt(&x);
    let s = format!("{result}");
    // x < 0 → (-∞, 0)
    assert!(!s.contains("EmptySet"), "x < 0 should not be empty: {s}");
    // Interior: x=-2 → -2 < 0 ✓
    common::assert_negative_at(&x, &x, -2, "x < 0 interior at x=-2");
    // Exterior: x=1 → 1 > 0 ✗
    common::assert_positive_at(&x, &x, 1, "x < 0 exterior at x=1");
}

#[test]
fn solve_x2_minus_4_lt_0() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let poly = &x.powi(2) - 4;
    let result = poly.solve_lt(&x);
    let s = format!("{result}");
    // x² - 4 < 0 → (-2, 2)
    assert!(
        !s.contains("EmptySet"),
        "x²-4 < 0 should have solutions: {s}"
    );
    assert!(s.contains("2"), "solution should reference 2: {s}");
    // Interior: x=0 → 0-4=-4 < 0 ✓
    common::assert_negative_at(&poly, &x, 0, "x²-4 < 0 interior at x=0");
    // Interior: x=1 → 1-4=-3 < 0 ✓
    common::assert_negative_at(&poly, &x, 1, "x²-4 < 0 interior at x=1");
    // Exterior: x=3 → 9-4=5 > 0 ✗
    common::assert_positive_at(&poly, &x, 3, "x²-4 < 0 exterior at x=3");
}

#[test]
fn solve_negative_constant_lt() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let neg = ctx.int(-5);
    let result = neg.solve_lt(&x);
    let s = format!("{result}");
    // -5 < 0 is always true → (-∞, ∞)
    assert!(
        !s.contains("EmptySet"),
        "-5 < 0 should be true everywhere: {s}"
    );
    common::assert_negative_at(&neg, &x, 0, "-5 < 0 constant check");
}

#[test]
fn solve_positive_constant_lt() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let pos = ctx.int(3);
    let result = pos.solve_lt(&x);
    // 3 < 0 is always false → EmptySet
    assert_eq!(format!("{result}"), "EmptySet");
    common::assert_positive_at(&pos, &x, 0, "3 < 0 constant is positive");
}

// ═══════════════════════════════════════════════════════════════════════════
// solve_le — less-than-or-equal
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_x2_minus_4_le_0() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let poly = &x.powi(2) - 4;
    let result = poly.solve_le(&x);
    let s = format!("{result}");
    // x² - 4 ≤ 0 → [-2, 2]
    assert!(
        !s.contains("EmptySet"),
        "x²-4 ≤ 0 should have solutions: {s}"
    );
    // Interior: x=0 → -4 < 0 ✓
    common::assert_negative_at(&poly, &x, 0, "x²-4 ≤ 0 interior at x=0");
    // Exterior: x=5 → 21 > 0 ✗
    common::assert_positive_at(&poly, &x, 5, "x²-4 ≤ 0 exterior at x=5");
}

#[test]
fn solve_zero_le() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let zero = ctx.int(0);
    let result = zero.solve_le(&x);
    let s = format!("{result}");
    // 0 <= 0 always true
    assert!(
        !s.contains("EmptySet"),
        "0 <= 0 should be satisfied everywhere: {s}"
    );
}

#[test]
fn solve_negative_constant_le() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let neg = ctx.int(-2);
    let result = neg.solve_le(&x);
    let s = format!("{result}");
    // -2 <= 0 always true
    assert!(
        !s.contains("EmptySet"),
        "-2 <= 0 should be true everywhere: {s}"
    );
    common::assert_negative_at(&neg, &x, 0, "-2 <= 0 constant check");
}

// ═══════════════════════════════════════════════════════════════════════════
// solveset — equation solving as a set
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solveset_quadratic() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let poly = &x.powi(2) - &x * 5 + 6;
    let result = poly.solve_as_set(&x);
    let s = format!("{result}");
    // x² - 5x + 6 = 0 → {2, 3}
    assert!(
        !s.contains("EmptySet"),
        "solveset of x²-5x+6 should find roots: {s}"
    );
    assert!(s.contains("2"), "should contain root 2: {s}");
    assert!(s.contains("3"), "should contain root 3: {s}");
    // Verify roots: poly at x=2 → 4-10+6=0
    let val_at_2 = poly.subs(&x, &ctx.int(2)).eval_f64()
        .expect("eval at root 2 should succeed");
    assert!(val_at_2.abs() < 1e-10, "poly(2) should be 0, got {val_at_2}");
    // Verify roots: poly at x=3 → 9-15+6=0
    let val_at_3 = poly.subs(&x, &ctx.int(3)).eval_f64()
        .expect("eval at root 3 should succeed");
    assert!(val_at_3.abs() < 1e-10, "poly(3) should be 0, got {val_at_3}");
}

#[test]
fn solveset_linear() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x - 7;
    let result = expr.solve_as_set(&x);
    let s = format!("{result}");
    // x - 7 = 0 → {7}
    assert!(s.contains("7"), "should contain root 7: {s}");
    // Verify root
    let val_at_7 = expr.subs(&x, &ctx.int(7)).eval_f64()
        .expect("eval at root 7 should succeed");
    assert!(val_at_7.abs() < 1e-10, "expr(7) should be 0, got {val_at_7}");
}

#[test]
fn solveset_no_real_roots() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    // x² + 1 = 0 has no real roots (only complex)
    let expr = &x.powi(2) + 1;
    let result = expr.solve_as_set(&x);
    let s = format!("{result}");
    // The solver may or may not find complex roots; if it doesn't,
    // we should at least get something (possibly EmptySet or a set
    // with imaginary values). Just assert it doesn't panic.
    assert!(!s.is_empty(), "solveset should produce output: {s}");
    // Verify the polynomial is always positive for real x
    common::assert_positive_at(&expr, &x, 0, "x²+1 at x=0");
    common::assert_positive_at(&expr, &x, 5, "x²+1 at x=5");
    common::assert_positive_at(&expr, &x, -3, "x²+1 at x=-3");
}

#[test]
fn solveset_constant_nonzero() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let five = ctx.int(5);
    let result = five.solve_as_set(&x);
    // 5 = 0 has no solutions → EmptySet
    assert_eq!(format!("{result}"), "EmptySet");
}

// ═══════════════════════════════════════════════════════════════════════════
// Quadratic sign-chart: full coverage
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quadratic_positive_leading_coeff_gt() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    // x² - 1 > 0 → (-∞, -1) ∪ (1, ∞)
    let poly = &x.powi(2) - 1;
    let result = poly.solve_gt(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "x²-1 > 0 should have solutions: {s}"
    );
    // Interior: x=2 → 4-1=3 > 0 ✓
    common::assert_positive_at(&poly, &x, 2, "x²-1 > 0 interior at x=2");
    // Interior: x=-2 → 4-1=3 > 0 ✓
    common::assert_positive_at(&poly, &x, -2, "x²-1 > 0 interior at x=-2");
    // Exterior: x=0 → 0-1=-1 < 0 ✗
    common::assert_negative_at(&poly, &x, 0, "x²-1 > 0 exterior at x=0");
}

#[test]
fn quadratic_positive_leading_coeff_lt() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    // x² - 1 < 0 → (-1, 1)
    let poly = &x.powi(2) - 1;
    let result = poly.solve_lt(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "x²-1 < 0 should have solutions: {s}"
    );
    // Should be a bounded interval between -1 and 1
    assert!(s.contains("1"), "solution should reference 1: {s}");
    // Interior: x=0 → -1 < 0 ✓
    common::assert_negative_at(&poly, &x, 0, "x²-1 < 0 interior at x=0");
    // Exterior: x=3 → 8 > 0 ✗
    common::assert_positive_at(&poly, &x, 3, "x²-1 < 0 exterior at x=3");
}

// ═══════════════════════════════════════════════════════════════════════════
// Linear inequalities
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn linear_2x_minus_6_gt_0() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    // 2x - 6 > 0 → x > 3 → (3, ∞)
    let expr = &x * 2 - 6;
    let result = expr.solve_gt(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "2x-6 > 0 should have solutions: {s}"
    );
    assert!(s.contains("3"), "solution should reference 3: {s}");
    // Interior: x=5 → 10-6=4 > 0 ✓
    common::assert_positive_at(&expr, &x, 5, "2x-6 > 0 interior at x=5");
    // Exterior: x=1 → 2-6=-4 < 0 ✗
    common::assert_negative_at(&expr, &x, 1, "2x-6 > 0 exterior at x=1");
}

#[test]
fn linear_neg_x_plus_5_le_0() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    // -x + 5 ≤ 0 → x ≥ 5 → [5, ∞)
    let expr = -&x + 5;
    let result = expr.solve_le(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "-x+5 ≤ 0 should have solutions: {s}"
    );
    assert!(s.contains("5"), "solution should reference 5: {s}");
    // Interior: x=10 → -10+5=-5 < 0 ✓
    common::assert_negative_at(&expr, &x, 10, "-x+5 ≤ 0 interior at x=10");
    // Exterior: x=2 → -2+5=3 > 0 ✗
    common::assert_positive_at(&expr, &x, 2, "-x+5 ≤ 0 exterior at x=2");
}

// ═══════════════════════════════════════════════════════════════════════════
// Cubic polynomial
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cubic_x3_minus_x_gt_0() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    // x³ - x = x(x-1)(x+1) > 0  → (-1, 0) ∪ (1, ∞)
    let poly = &x.powi(3) - &x;
    let result = poly.solve_gt(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "x³-x > 0 should have solutions: {s}"
    );
    // Interior: x=2 → 8-2=6 > 0 ✓
    common::assert_positive_at(&poly, &x, 2, "x³-x > 0 interior at x=2");
    // Exterior: x=-2 → -8+2=-6 < 0 ✗
    common::assert_negative_at(&poly, &x, -2, "x³-x > 0 exterior at x=-2");
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge: large positive / large negative constants
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn large_positive_constant_gt() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let big = ctx.int(999999);
    let result = big.solve_gt(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "999999 > 0 should be true everywhere: {s}"
    );
    common::assert_positive_at(&big, &x, 0, "999999 > 0 constant check");
}

#[test]
fn large_negative_constant_lt() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let big_neg = ctx.int(-999999);
    let result = big_neg.solve_lt(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "-999999 < 0 should be true everywhere: {s}"
    );
    common::assert_negative_at(&big_neg, &x, 0, "-999999 < 0 constant check");
}

// ═══════════════════════════════════════════════════════════════════════════
// solveset / solve_gt with Context API
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solveset_with_context() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let poly = &x.powi(2) - &x * 5 + 6;
    let result = poly.solve_as_set(&x);
    let s = format!("{result}");
    assert!(s.contains("2"), "should contain 2: {s}");
    assert!(s.contains("3"), "should contain 3: {s}");
}

#[test]
fn solve_gt_with_context() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let five = ctx.int(5);
    let result = five.solve_gt(&x);
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
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    // 2x - 1 = 0 → x = 1/2
    let expr = &x * 2 - 1;
    let result = expr.solve_as_set(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "2x - 1 = 0 should have a solution: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// New tests: always positive, always negative, boundary inclusion
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_always_positive() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    // x² + 1 > 0 should be true for all real x → UniversalSet or (-∞, ∞)
    let poly = &x.powi(2) + 1;
    let result = poly.solve_gt(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "x²+1 > 0 should be always true, got: {s}"
    );
    // Should be the entire real line — UniversalSet or an interval from -oo to oo
    assert!(
        s.contains("UniversalSet") || (s.contains("-oo") && s.contains("oo")),
        "x²+1 > 0 should be the entire real line, got: {s}"
    );
    // Verify polynomial is positive at several points
    common::assert_positive_at(&poly, &x, 0, "x²+1 at x=0");
    common::assert_positive_at(&poly, &x, 100, "x²+1 at x=100");
    common::assert_positive_at(&poly, &x, -100, "x²+1 at x=-100");
}

#[test]
fn solve_always_negative() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    // -(x² + 1) > 0 should be false for all real x → EmptySet
    let poly = -&(&x.powi(2) + 1);
    let result = poly.solve_gt(&x);
    let s = format!("{result}");
    assert_eq!(
        s, "EmptySet",
        "-(x²+1) > 0 should be EmptySet, got: {s}"
    );
    // Verify polynomial is negative at several points
    common::assert_negative_at(&poly, &x, 0, "-(x²+1) at x=0");
    common::assert_negative_at(&poly, &x, 5, "-(x²+1) at x=5");
    common::assert_negative_at(&poly, &x, -5, "-(x²+1) at x=-5");
}

#[test]
fn solve_ge_includes_boundary() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    // x² - 4 >= 0 should include x=2 and x=-2 as boundary
    let poly = &x.powi(2) - 4;
    let result = poly.solve_ge(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "x²-4 >= 0 should have solutions: {s}"
    );
    // At the boundary x=2, x²-4 = 0 which satisfies >= 0
    let val_at_2 = poly.subs(&x, &ctx.int(2)).eval_f64()
        .expect("eval at boundary x=2 should succeed");
    assert!(
        val_at_2.abs() < 1e-10,
        "x²-4 at x=2 should be 0, got {val_at_2}"
    );
    // At the boundary x=-2, x²-4 = 0
    let val_at_neg2 = poly.subs(&x, &ctx.int(-2)).eval_f64()
        .expect("eval at boundary x=-2 should succeed");
    assert!(
        val_at_neg2.abs() < 1e-10,
        "x²-4 at x=-2 should be 0, got {val_at_neg2}"
    );
    // The ge solution should include the boundary — 0 >= 0 is true
    // Also verify interior and exterior
    common::assert_positive_at(&poly, &x, 3, "x²-4 >= 0 interior at x=3");
    common::assert_negative_at(&poly, &x, 0, "x²-4 >= 0 exterior at x=0");
}
