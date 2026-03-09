//! Integration tests for Sturm-based inequality solving and root analysis.

use symplex::prelude::*;
mod common;

// ═══════════════════════════════════════════════════════════════════════════
// x^2 - 4 > 0 — two real roots at ±2
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sturm_x2_minus_4_gt_0() {
    let ctx = Context::new();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; x);
    let poly = &x.powi(2) - 4;
    let result = poly.solve_gt(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "x²-4 > 0 should have solutions, got: {s}"
    );
    // Verify sign at specific points
    common::assert_positive_at(&poly, &x, 3, "x²-4 at x=3");
    common::assert_positive_at(&poly, &x, -3, "x²-4 at x=-3");
    common::assert_negative_at(&poly, &x, 0, "x²-4 at x=0");
    common::assert_negative_at(&poly, &x, 1, "x²-4 at x=1");
    common::assert_negative_at(&poly, &x, -1, "x²-4 at x=-1");
}

// ═══════════════════════════════════════════════════════════════════════════
// x^2 + 1 > 0 — no real roots, always positive → UniversalSet
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sturm_x2_plus_1_gt_0_universal() {
    let ctx = Context::new();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; x);
    let poly = &x.powi(2) + 1;
    let result = poly.solve_gt(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "x²+1 > 0 should be always true, got: {s}"
    );
    assert!(
        s.contains("UniversalSet") || (s.contains("-oo") && s.contains("oo")),
        "x²+1 > 0 should be the entire real line, got: {s}"
    );
    // Verify polynomial is positive everywhere
    common::assert_positive_at(&poly, &x, 0, "x²+1 at x=0");
    common::assert_positive_at(&poly, &x, 100, "x²+1 at x=100");
    common::assert_positive_at(&poly, &x, -100, "x²+1 at x=-100");
}

// ═══════════════════════════════════════════════════════════════════════════
// -(x^2 + 1) > 0 — no real roots, always negative → EmptySet
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sturm_neg_x2_plus_1_gt_0_empty() {
    let ctx = Context::new();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; x);
    let poly = -&(&x.powi(2) + 1);
    let result = poly.solve_gt(&x);
    let s = format!("{result}");
    assert_eq!(
        s, "EmptySet",
        "-(x²+1) > 0 should be EmptySet, got: {s}"
    );
    // Verify polynomial is negative everywhere
    common::assert_negative_at(&poly, &x, 0, "-(x²+1) at x=0");
    common::assert_negative_at(&poly, &x, 5, "-(x²+1) at x=5");
    common::assert_negative_at(&poly, &x, -5, "-(x²+1) at x=-5");
}

// ═══════════════════════════════════════════════════════════════════════════
// x^3 - x > 0 — roots at -1, 0, 1
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sturm_x3_minus_x_gt_0() {
    let ctx = Context::new();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; x);
    // x^3 - x = x(x-1)(x+1)
    let poly = &x.powi(3) - &x;
    let result = poly.solve_gt(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "x³-x > 0 should have solutions, got: {s}"
    );
    // x³ - x > 0 on (-1, 0) ∪ (1, ∞)
    common::assert_positive_at_rational(&poly, &x, -1, 2, "x³-x at x=-1/2");
    common::assert_positive_at(&poly, &x, 2, "x³-x at x=2");
    // negative on (-∞, -1) ∪ (0, 1)
    common::assert_negative_at(&poly, &x, -2, "x³-x at x=-2");
    common::assert_negative_at_rational(&poly, &x, 1, 2, "x³-x at x=1/2");
}

// ═══════════════════════════════════════════════════════════════════════════
// x^2 >= 0 — root at 0 but it's non-strict, so the whole real line
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sturm_x2_ge_0_universal() {
    let ctx = Context::new();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; x);
    let poly = x.powi(2);
    let result = poly.solve_ge(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "x² >= 0 should have solutions, got: {s}"
    );
    // x² is non-negative everywhere, so the solution should cover the whole line
    // Verify at specific points
    let val0 = poly.subs(&x, &ctx.int(0)).eval_f64()
        .expect("eval at x=0 should succeed");
    assert!(val0.abs() < 1e-10, "x² at x=0 should be 0, got {val0}");
    common::assert_positive_at(&poly, &x, 1, "x² at x=1");
    common::assert_positive_at(&poly, &x, -1, "x² at x=-1");
}

// ═══════════════════════════════════════════════════════════════════════════
// (x-1)(x-2)(x-3) > 0 — roots at 1, 2, 3
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sturm_cubic_factored_gt_0() {
    let ctx = Context::new();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; x);
    // (x-1)(x-2)(x-3) = x³ - 6x² + 11x - 6
    let poly = &(&(&x - 1) * &(&x - 2)) * &(&x - 3);
    let result = poly.solve_gt(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "(x-1)(x-2)(x-3) > 0 should have solutions, got: {s}"
    );
    // Positive on (1, 2) ∪ (3, ∞)
    common::assert_positive_at_rational(&poly, &x, 3, 2, "(x-1)(x-2)(x-3) at x=3/2");
    common::assert_positive_at(&poly, &x, 4, "(x-1)(x-2)(x-3) at x=4");
    common::assert_positive_at(&poly, &x, 10, "(x-1)(x-2)(x-3) at x=10");
    // Negative on (-∞, 1) ∪ (2, 3)
    common::assert_negative_at(&poly, &x, 0, "(x-1)(x-2)(x-3) at x=0");
    common::assert_negative_at(&poly, &x, -1, "(x-1)(x-2)(x-3) at x=-1");
    common::assert_negative_at_rational(&poly, &x, 5, 2, "(x-1)(x-2)(x-3) at x=5/2");
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sturm_constant_positive_gt() {
    let ctx = Context::new();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; x);
    let poly = ctx.int(7);
    let result = poly.solve_gt(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "7 > 0 should be always true, got: {s}"
    );
}

#[test]
fn sturm_constant_negative_gt() {
    let ctx = Context::new();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; x);
    let poly = ctx.int(-3);
    let result = poly.solve_gt(&x);
    let s = format!("{result}");
    assert_eq!(s, "EmptySet", "-3 > 0 should be EmptySet, got: {s}");
}

#[test]
fn sturm_linear_gt() {
    let ctx = Context::new();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; x);
    // 2x - 6 > 0 ↔ x > 3
    let poly = &(&x * 2) - 6;
    let result = poly.solve_gt(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "2x-6 > 0 should have solutions, got: {s}"
    );
    common::assert_positive_at(&poly, &x, 4, "2x-6 at x=4");
    common::assert_negative_at(&poly, &x, 2, "2x-6 at x=2");
}

#[test]
fn sturm_x2_plus_1_lt_0_empty() {
    let ctx = Context::new();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; x);
    // x² + 1 < 0 should have no solutions
    let poly = &x.powi(2) + 1;
    let result = poly.solve_lt(&x);
    let s = format!("{result}");
    assert_eq!(
        s, "EmptySet",
        "x²+1 < 0 should be EmptySet, got: {s}"
    );
}

#[test]
fn sturm_neg_x2_minus_1_le_0_universal() {
    let ctx = Context::new();
    let __vars_ctx = ctx.clone(); symplex::syms!(__vars_ctx; x);
    // -(x² + 1) ≤ 0 should be true for all x (always negative)
    let poly = -&(&x.powi(2) + 1);
    let result = poly.solve_le(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "-(x²+1) ≤ 0 should be always true, got: {s}"
    );
    assert!(
        s.contains("UniversalSet") || (s.contains("-oo") && s.contains("oo")),
        "-(x²+1) ≤ 0 should be the entire real line, got: {s}"
    );
}
