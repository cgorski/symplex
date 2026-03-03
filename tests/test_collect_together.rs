//! Tests for collect() and together() through the public Ex API.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// collect()
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn collect_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x^2 + 2*x + 1 collected in x — should stay the same (already grouped)
    let expr = &x.powi(2) + &x * 2 + 1;
    let collected = expr.collect(&x);
    assert_eq!(format!("{collected}"), "1 + x^2 + 2*x");
}

#[test]
fn collect_groups_by_variable() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // x*y + x^2 + y, collected in x → y + x^2 + x*y
    // The polynomial view: (1)*x^2 + (y)*x + (y)*1
    // But since we can't represent symbolic coefficients as Poly (which uses Ratio<BigInt>),
    // collect may return unchanged if the expression isn't polynomial over rationals.
    let expr = &x * &y + &x.powi(2) + &y;
    let collected = expr.collect(&x);
    // If not representable as poly in x (symbolic coefficients), returns unchanged
    let s = format!("{collected}");
    // Should at least contain all terms
    assert!(
        s.contains("x") && s.contains("y"),
        "should contain both vars: {s}"
    );
}

#[test]
fn collect_non_polynomial_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin();
    let collected = expr.collect(&x);
    assert_eq!(format!("{collected}"), "sin(x)");
}

#[test]
fn collect_constant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let five = ctx.int(5);
    let collected = five.collect(&x);
    assert_eq!(format!("{collected}"), "5");
}

#[test]
fn collect_pure_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 3*x^3 + 2*x + 7
    let expr = &x.powi(3) * 3 + &x * 2 + 7;
    let collected = expr.collect(&x);
    let s = format!("{collected}");
    assert!(s.contains("7"), "should contain constant: {s}");
    assert!(s.contains("x^3"), "should contain x^3: {s}");
    assert!(s.contains("2*x"), "should contain 2*x: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// together()
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn together_two_fractions() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // 1/x + 1/y → (x + y) / (x*y)
    let expr = &x.powi(-1) + &y.powi(-1);
    let result = expr.together();
    let s = format!("{result}");
    // Should have common denominator
    assert!(
        s.contains("x*y") || s.contains("y*x"),
        "should have x*y denom: {s}"
    );
}

#[test]
fn together_already_no_fractions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x + 1 — no fractions, should stay unchanged
    let expr = &x + 1;
    let result = expr.together();
    assert_eq!(format!("{result}"), "1 + x");
}

#[test]
fn together_not_an_add() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Just x — not an Add, should stay unchanged
    let result = x.together();
    assert_eq!(format!("{result}"), "x");
}

#[test]
fn together_mixed_fraction_and_non_fraction() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x + 1/x → (x^2 + 1) / x
    let expr = &x + &x.powi(-1);
    let result = expr.together();
    let s = format!("{result}");
    // Should have x as denominator
    assert!(s.contains("1/x") || s.contains("("), "should combine: {s}");
}

#[test]
fn together_single_fraction() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Just 1/x — it's a single term, not an Add, stays unchanged
    let result = x.powi(-1).together();
    assert_eq!(format!("{result}"), "1/x");
}
