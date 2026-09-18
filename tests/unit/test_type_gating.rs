//! Tests for type-based strategy gating in smart_simplify and trigsimp.
//!
//! Verifies that irrelevant simplification strategies are skipped based on
//! expression node types, while still producing correct results.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// smart_simplify gating tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn smart_simplify_polynomial_unchanged() {
    // Pure polynomial — should skip trig/exp/logcombine strategies
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let big_poly = x.powi(10) + x.powi(5) + &x + 1;
    let result = big_poly.simplify();
    let s = format!("{result}");
    // Should still contain x^10 — polynomial identity is preserved
    assert!(
        s.contains("x"),
        "polynomial should still contain x, got: {s}"
    );
}

#[test]
fn smart_simplify_atom_is_identity() {
    // A single symbol is an atom — should early-exit immediately
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.simplify();
    assert_eq!(format!("{result}"), "x", "atom should be unchanged");
}

#[test]
fn smart_simplify_numeric_atom_is_identity() {
    // A single number is an atom — should early-exit immediately
    let ctx = Context::new();
    let five = ctx.int(5);
    let result = five.simplify();
    assert_eq!(format!("{result}"), "5", "numeric atom should be unchanged");
}

#[test]
fn smart_simplify_trig_still_works() {
    // Trig identity sin²(x) + cos²(x) = 1 must still simplify
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin().powi(2) + x.cos().powi(2);
    let result = expr.simplify();
    assert_eq!(format!("{result}"), "1", "sin²+cos² should simplify to 1");
}

#[test]
fn smart_simplify_exp_ln_still_works() {
    // exp(ln(x)) = x must still simplify even with gating
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.ln().exp();
    let result = expr.simplify();
    assert_eq!(format!("{result}"), "x", "exp(ln(x)) should simplify to x");
}

#[test]
fn smart_simplify_sin_zero() {
    // sin(0) = 0 — eval strategy (always run) should catch this
    let ctx = Context::new();
    let zero = ctx.int(0);
    let expr = zero.sin();
    let result = expr.simplify();
    assert_eq!(format!("{result}"), "0", "sin(0) should simplify to 0");
}

#[test]
fn smart_simplify_pure_product_no_bloat() {
    // x * y — no trig, no exp, no Add → many strategies skipped
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x * &y;
    let result = expr.simplify();
    let s = format!("{result}");
    assert!(
        s.contains("x") && s.contains("y"),
        "x*y should be preserved, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// trigsimp gating tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn trigsimp_on_polynomial_is_noop() {
    // trigsimp on a pure polynomial should be identity (early exit)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let poly = x.powi(3) + &x + 1;
    let result = poly.simplify_trig();
    let original_s = format!("{poly}");
    let result_s = format!("{result}");
    assert_eq!(
        result_s, original_s,
        "trigsimp on polynomial should be identity, got: {result_s}"
    );
}

#[test]
fn trigsimp_on_exp_is_noop() {
    // trigsimp on exp(x) — no trig nodes, should skip
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.exp();
    let result = expr.simplify_trig();
    let original_s = format!("{expr}");
    let result_s = format!("{result}");
    assert_eq!(
        result_s, original_s,
        "trigsimp on exp(x) should be identity, got: {result_s}"
    );
}

#[test]
fn trigsimp_still_works() {
    // sin²(x) + cos²(x) + x should simplify the trig part
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin().powi(2) + x.cos().powi(2) + &x;
    let result = expr.simplify_trig();
    let s = format!("{result}");
    assert!(
        !s.contains("sin") && !s.contains("cos"),
        "should simplify trig identity away: {s}"
    );
}

#[test]
fn trigsimp_pythagorean_identity() {
    // Classic: sin²(x) + cos²(x) = 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin().powi(2) + x.cos().powi(2);
    let result = expr.simplify_trig();
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn trigsimp_on_symbol_is_noop() {
    // Just a bare symbol — nothing to do
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.simplify_trig();
    assert_eq!(format!("{result}"), "x");
}

#[test]
fn trigsimp_on_number_is_noop() {
    // Just a number — nothing to do
    let ctx = Context::new();
    let n = ctx.int(42);
    let result = n.simplify_trig();
    assert_eq!(format!("{result}"), "42");
}

// ═══════════════════════════════════════════════════════════════════════════
// Regression: gating must not break cancel strategy
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn smart_simplify_rational_cancel() {
    // (x² - 1)/(x - 1) should still simplify to x + 1 via cancel
    // This expression has negative powers (division → Pow(_, -1))
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let numer = x.powi(2) - 1;
    let denom = &x - 1;
    let expr = numer / denom;
    let result = expr.simplify();
    let s = format!("{result}");
    assert!(
        !s.contains('/') && s.contains('x'),
        "(x²-1)/(x-1) should cancel to x+1, got: {s}"
    );
}
