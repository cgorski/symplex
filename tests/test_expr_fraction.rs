//! Integration tests for the `expr!` macro's fraction / rational literal handling.
//!
//! Verifies that `expr!(ctx, 1/2)` produces an exact rational (not Rust integer
//! division), that `expr!(ctx, -1/2)` is handled correctly, and that mixed
//! expressions involving fractions compose properly.

// ═══════════════════════════════════════════════════════════════════════════
// expr!(ctx, 1/2) should equal ctx.rational(1, 2)
// ═══════════════════════════════════════════════════════════════════════════

use symplex::prelude::*;
#[test]
fn expr_half_equals_rational() {
    let ctx = Context::new();
    let half_macro = symplex::expr!(ctx, 1 / 2);
    let half_fn = ctx.rational(1, 2);
    assert_eq!(
        format!("{half_macro}"),
        format!("{half_fn}"),
        "expr!(ctx, 1/2) should produce the same value as rational(1,2)"
    );
    assert_eq!(
        format!("{half_macro}"),
        "1/2",
        "expr!(ctx, 1/2) should display as 1/2"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// expr!(ctx, 1/3) should equal ctx.rational(1, 3)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_third_equals_rational() {
    let ctx = Context::new();
    let third_macro = symplex::expr!(ctx, 1 / 3);
    let third_fn = ctx.rational(1, 3);
    assert_eq!(
        format!("{third_macro}"),
        format!("{third_fn}"),
        "expr!(ctx, 1/3) should produce the same value as rational(1,3)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// expr!(ctx, 3/4 * x) should work — fraction times a variable
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_fraction_times_variable() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = symplex::expr!(ctx, 3 / 4 * x);
    let s = format!("{expr}");
    // The result should represent (3/4)*x in some canonical form.
    assert!(
        s.contains("3/4") || s.contains("3*x/4") || s.contains("x"),
        "expr!(ctx, 3/4 * x) should contain the fraction and variable: {s}"
    );
    // It should NOT be zero (which would happen if 3/4 were integer-divided).
    assert_ne!(s, "0", "expr!(ctx, 3/4 * x) should not be 0: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// expr!(ctx, x / 2) should still be division (left is not int literal)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_var_divided_by_int_is_division() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = symplex::expr!(ctx, x / 2);
    let s = format!("{expr}");
    // This should be x/2 or (1/2)*x — NOT rational(x, 2).
    assert!(
        s.contains('x'),
        "expr!(ctx, x/2) should still involve x: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// expr!(ctx, 1 / x) should still be division (right is not int literal)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_int_divided_by_var_is_division() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = symplex::expr!(ctx, 1 / x);
    let s = format!("{expr}");
    // Should be 1/x or x^(-1) — NOT rational(1, x).
    assert!(
        s.contains('x'),
        "expr!(ctx, 1/x) should still involve x: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// expr!(ctx, -1/2) should equal ctx.rational(-1, 2) (not 0)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_neg_half_equals_rational() {
    let ctx = Context::new();
    let neg_half_macro = symplex::expr!(ctx, -1 / 2);
    let neg_half_fn = ctx.rational(-1, 2);
    let s_macro = format!("{neg_half_macro}");
    let s_fn = format!("{neg_half_fn}");
    assert_eq!(
        s_macro, s_fn,
        "expr!(ctx, -1/2) should produce the same value as rational(-1,2)"
    );
    // Crucially, it should NOT be zero (the old bug).
    assert_ne!(
        s_macro, "0",
        "expr!(ctx, -1/2) must NOT be 0 (was a Rust integer division bug)"
    );
}

#[test]
fn expr_neg_half_is_negative() {
    let ctx = Context::new();
    let neg_half = symplex::expr!(ctx, -1 / 2);
    let s = format!("{neg_half}");
    // The display should be -1/2 or similar negative fraction.
    assert!(
        s.contains("-1/2") || s.starts_with('-'),
        "expr!(ctx, -1/2) should display as a negative fraction: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// expr!(ctx, 1/2 + 1/3) should equal rational(1,2) + rational(1,3) = 5/6
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_half_plus_third_equals_five_sixths() {
    let ctx = Context::new();
    let sum = symplex::expr!(ctx, 1 / 2 + 1 / 3);
    let s = format!("{sum}");
    assert_eq!(
        s, "5/6",
        "expr!(ctx, 1/2 + 1/3) should be 5/6, got {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// expr!(ctx, -3/4) should work
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_neg_three_quarters() {
    let ctx = Context::new();
    let val = symplex::expr!(ctx, -3 / 4);
    let expected = ctx.rational(-3, 4);
    assert_eq!(
        format!("{val}"),
        format!("{expected}"),
        "expr!(ctx, -3/4) should equal rational(-3, 4)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Fractions inside larger expressions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_fraction_in_addition() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = symplex::expr!(ctx, x + 1 / 2);
    let s = format!("{expr}");
    assert!(
        s.contains("1/2") || s.contains("x"),
        "expr!(ctx, x + 1/2) should contain x and 1/2: {s}"
    );
    // Substituting x = 1/2 should give 1.
    let half = ctx.rational(1, 2);
    let result = expr.subs(&x, &half);
    assert_eq!(
        format!("{result}"),
        "1",
        "x + 1/2 with x=1/2 should be 1"
    );
}

#[test]
fn expr_fraction_in_subtraction() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = symplex::expr!(ctx, x - 1 / 2);
    // Substituting x = 1/2 should give 0.
    let half = ctx.rational(1, 2);
    let result = expr.subs(&x, &half);
    assert_eq!(
        format!("{result}"),
        "0",
        "x - 1/2 with x=1/2 should be 0"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Auto-reduction: 2/4 should reduce to 1/2
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_auto_reduces_fraction() {
    let ctx = Context::new();
    let val = symplex::expr!(ctx, 2 / 4);
    let s = format!("{val}");
    assert_eq!(s, "1/2", "expr!(ctx, 2/4) should auto-reduce to 1/2, got {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Integer division should still be exact: 6/3 = 2
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_exact_integer_division() {
    let ctx = Context::new();
    let val = symplex::expr!(ctx, 6 / 3);
    let s = format!("{val}");
    assert_eq!(s, "2", "expr!(ctx, 6/3) should be exactly 2, got {s}");
}
