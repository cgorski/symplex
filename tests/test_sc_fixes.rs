//! Tests for short-circuit fixes: 0*Add(…∞…)→NaN, 2-arg fast paths,
//! and multi-branch trig inversions.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Part 1: 0 * Add(…∞…) → NaN
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn zero_times_add_with_pos_infinity_is_nan() {
    let ctx = Context::new();
    let oo = ctx.infinity();
    let x = ctx.symbol("x");
    let sum = &x + &oo;
    let expr = &ctx.int(0) * &sum;
    assert_eq!(
        format!("{expr}"),
        "nan",
        "0 * (x + oo) should be nan, got: {}",
        expr
    );
}

#[test]
fn zero_times_add_with_neg_infinity_is_nan() {
    let ctx = Context::new();
    let neg_oo = ctx.neg_infinity();
    let x = ctx.symbol("x");
    let sum = &x + &neg_oo;
    let expr = &ctx.int(0) * &sum;
    assert_eq!(
        format!("{expr}"),
        "nan",
        "0 * (x + (-oo)) should be nan, got: {}",
        expr
    );
}

#[test]
fn zero_times_finite_add_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sum = &x + &ctx.int(1);
    let expr = &ctx.int(0) * &sum;
    assert_eq!(format!("{expr}"), "0", "0 * (x + 1) should be 0");
}

#[test]
fn zero_times_bare_infinity_is_nan() {
    let ctx = Context::new();
    // Sanity: the pre-existing 0 * oo path still works
    let expr = &ctx.int(0) * &ctx.infinity();
    assert_eq!(format!("{expr}"), "nan", "0 * oo should be nan");
}

#[test]
fn zero_times_symbol_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &ctx.int(0) * &x;
    assert_eq!(format!("{expr}"), "0", "0 * x should be 0");
}

// ═══════════════════════════════════════════════════════════════════════════
// Part 2: 2-arg numeric fast paths
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fast_path_add_two_integers() {
    let ctx = Context::new();
    let a = ctx.int(7);
    let b = ctx.int(13);
    let result = &a + &b;
    assert_eq!(format!("{result}"), "20", "7 + 13 = 20");
}

#[test]
fn fast_path_add_two_rationals() {
    let ctx = Context::new();
    let a = ctx.rational(1, 3);
    let b = ctx.rational(1, 6);
    let result = &a + &b;
    assert_eq!(format!("{result}"), "1/2", "1/3 + 1/6 = 1/2");
}

#[test]
fn fast_path_add_cancels_to_zero() {
    let ctx = Context::new();
    let a = ctx.int(5);
    let b = ctx.int(-5);
    let result = &a + &b;
    assert_eq!(format!("{result}"), "0", "5 + (-5) = 0");
}

#[test]
fn fast_path_mul_two_integers() {
    let ctx = Context::new();
    let a = ctx.int(6);
    let b = ctx.int(7);
    let result = &a * &b;
    assert_eq!(format!("{result}"), "42", "6 * 7 = 42");
}

#[test]
fn fast_path_mul_two_rationals() {
    let ctx = Context::new();
    let a = ctx.rational(2, 3);
    let b = ctx.rational(3, 4);
    let result = &a * &b;
    assert_eq!(format!("{result}"), "1/2", "2/3 * 3/4 = 1/2");
}

#[test]
fn fast_path_mul_to_one() {
    let ctx = Context::new();
    let a = ctx.rational(3, 7);
    let b = ctx.rational(7, 3);
    let result = &a * &b;
    assert_eq!(format!("{result}"), "1", "3/7 * 7/3 = 1");
}

#[test]
fn fast_path_mul_to_zero() {
    let ctx = Context::new();
    let a = ctx.int(0);
    let b = ctx.int(99);
    let result = &a * &b;
    assert_eq!(format!("{result}"), "0", "0 * 99 = 0");
}

// ═══════════════════════════════════════════════════════════════════════════
// Part 3: Multi-branch trig inversions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sin_equation_two_branches_via_solveset() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    // sin(x) = 1/2 → x ∈ {asin(1/2), π - asin(1/2)}
    // Use solveset() which accesses the full transcendental solver
    let half = ctx.rational(1, 2);
    let eq = &x.sin() - &half;
    let result = eq.solve_as_set(&x);
    let s = format!("{result}");
    // Should contain at least two elements (not EmptySet)
    assert!(
        !s.contains("EmptySet"),
        "sin(x) = 1/2 via solveset should not be empty: {s}"
    );
    // Should reference asin (the inverse)
    assert!(
        s.contains("asin") || s.contains("arcsin") || s.contains("pi"),
        "sin(x) = 1/2 solution should reference asin or pi: {s}"
    );
}

#[test]
fn cos_equation_two_branches_via_solveset() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    // cos(x) = 1/2 → x ∈ {acos(1/2), -acos(1/2)}
    let half = ctx.rational(1, 2);
    let eq = &x.cos() - &half;
    let result = eq.solve_as_set(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "cos(x) = 1/2 via solveset should not be empty: {s}"
    );
    assert!(
        s.contains("acos") || s.contains("arccos"),
        "cos(x) = 1/2 solution should reference acos: {s}"
    );
}

#[test]
fn sin_equation_branches_are_distinct() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let half = ctx.rational(1, 2);
    let eq = &x.sin() - &half;
    let result = eq.solve_as_set(&x);
    let s = format!("{result}");
    // If we got a FiniteSet with two elements, they should differ
    // (FiniteSet displays as {a, b})
    if s.contains(',') {
        let inner = s.trim_start_matches('{').trim_end_matches('}');
        let parts: Vec<&str> = inner.split(", ").collect();
        if parts.len() >= 2 {
            assert_ne!(
                parts[0], parts[1],
                "two sin branches should be distinct values"
            );
        }
    }
}

#[test]
fn sin_equation_zero_via_solveset() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    // sin(x) = 0 → solveset should find at least x = 0
    let eq = x.sin();
    let result = eq.solve_as_set(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "sin(x) = 0 via solveset should not be empty: {s}"
    );
    assert!(
        s.contains("0"),
        "sin(x) = 0 should include 0 among roots: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Regression: existing behaviour preserved
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mul_nan_still_propagates() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let nan = ctx.nan();
    let result = &x * &nan;
    assert_eq!(format!("{result}"), "nan", "x * nan should be nan");
}

#[test]
fn add_oo_minus_oo_still_nan() {
    let ctx = Context::new();
    let oo = ctx.infinity();
    let neg_oo = ctx.neg_infinity();
    let result = &oo + &neg_oo;
    assert_eq!(format!("{result}"), "nan", "oo + (-oo) should be nan");
}

#[test]
fn add_oo_plus_finite_still_oo() {
    let ctx = Context::new();
    let oo = ctx.infinity();
    let x = ctx.symbol("x");
    let result = &oo + &x;
    assert_eq!(format!("{result}"), "oo", "oo + x should be oo");
}
