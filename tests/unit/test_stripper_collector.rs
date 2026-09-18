//! Integration tests for the stripper-collector sub-expression matching
//! optimisation in `apply_rules`.
//!
//! The stripper-collector algorithm replaces O(n²) pairwise enumeration
//! with O(k·n) scanning for k-term patterns inside n-term Add/Mul nodes.

use symplex::expr;
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Pythagorean identity inside sums of increasing size
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn stripper_collector_pythagorean_plus_constant() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = expr!(ctx, sin(x) ^ 2 + cos(x) ^ 2 + 5);
    let result = expr.simplify();
    assert_eq!(format!("{result}"), "6");
}

#[test]
fn stripper_collector_pythagorean_in_large_sum() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, a, b, c);
    // a + b + sin²(x) + cos²(x) + c → a + b + c + 1
    let expr = &a + &b + &x.sin().powi(2) + &x.cos().powi(2) + &c;
    let result = expr.simplify();
    let s = format!("{result}");
    assert!(!s.contains("sin"), "should not contain sin: {s}");
    assert!(!s.contains("cos"), "should not contain cos: {s}");
}

#[test]
fn stripper_collector_5_term_add() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    // 1 + 2 + sin²(x) + 3 + cos²(x) → 7
    let expr = expr!(ctx, 1 + 2 + sin(x) ^ 2 + 3 + cos(x) ^ 2);
    let result = expr.simplify();
    assert_eq!(format!("{result}"), "7");
}

#[test]
fn stripper_collector_6_term_two_symbols() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y, z);
    // y + z + sin²(x) + 10 + cos²(x) + y  →  2*y + z + 11
    let expr = &y + &z + &x.sin().powi(2) + 10 + &x.cos().powi(2) + &y;
    let result = expr.simplify();
    let s = format!("{result}");
    assert!(!s.contains("sin"), "sin should have been eliminated: {s}");
    assert!(!s.contains("cos"), "cos should have been eliminated: {s}");
    assert!(s.contains("11"), "constant part should be 11: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Hyperbolic Pythagorean in larger sums
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn stripper_collector_cosh_sinh_in_sum() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    // cosh²(x) - sinh²(x) + 3 → 4
    let expr = expr!(ctx, cosh(x) ^ 2 - sinh(x) ^ 2 + 3);
    let result = expr.simplify();
    assert_eq!(format!("{result}"), "4");
}

// ═══════════════════════════════════════════════════════════════════════════
// Mul sub-expression matching (exp combine in larger product)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn stripper_collector_exp_mul_in_product() {
    let ctx = Context::new();
    symplex::syms!(ctx; a, b, z);
    // z * exp(a) * exp(b) → z * exp(a + b)
    let expr = &z * &a.exp() * &b.exp();
    let result = expr.simplify();
    let s = format!("{result}");
    assert!(s.contains("exp("), "should still contain exp: {s}");
    assert!(s.contains("z"), "should still contain z: {s}");
    // There should be only one exp() call, not two
    let exp_count = s.matches("exp(").count();
    assert_eq!(
        exp_count, 1,
        "should have exactly one exp after combining: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Regression: 2-term sums still work via normal matching
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn regression_exact_2_term_pythagorean() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    // sin²(x) + cos²(x) → 1  (exact match, no sub-expression needed)
    let expr = expr!(ctx, sin(x) ^ 2 + cos(x) ^ 2);
    let result = expr.simplify();
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn regression_3_term_add_pythagorean() {
    let ctx = Context::new();
    // The original sub-expression matching handled 3-term sums.
    // Verify it still works.
    symplex::syms!(ctx; x);
    let expr = expr!(ctx, sin(x) ^ 2 + cos(x) ^ 2 + 3);
    let result = expr.simplify();
    assert_eq!(format!("{result}"), "4");
}

// ═══════════════════════════════════════════════════════════════════════════
// Non-matching cases: ensure no false positives
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn no_false_match_different_args() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    // sin²(x) + cos²(y) should NOT simplify via Pythagorean (different args)
    let expr = expr!(ctx, sin(x) ^ 2 + cos(y) ^ 2 + 3);
    let result = expr.simplify();
    let s = format!("{result}");
    assert!(s.contains("sin"), "sin should remain (different args): {s}");
    assert!(s.contains("cos"), "cos should remain (different args): {s}");
}

#[test]
fn no_false_match_sin_sin() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    // sin²(x) + sin²(x) → 2*sin²(x), not 1
    let expr = expr!(ctx, sin(x) ^ 2 + sin(x) ^ 2);
    let result = expr.simplify();
    let s = format!("{result}");
    assert!(s.contains("sin"), "should still contain sin: {s}");
    assert_ne!(s, "1", "should not reduce to 1");
}
