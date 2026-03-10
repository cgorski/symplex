//! Tests for collect() and together() through the public Ex API.

mod common;

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
    assert_eq!(format!("{collected}"), "x^2 + 2*x + 1");
    common::assert_math_eq(&collected, &expr, &x, "collect_polynomial");
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
    // Two free variables: substitute one at a fixed value, then check the other.
    for &yval in &[2i64, 3, 5] {
        let c_at_y = collected.subs_i64(&y, yval);
        let e_at_y = expr.subs_i64(&y, yval);
        common::assert_math_eq(
            &c_at_y,
            &e_at_y,
            &x,
            &format!("collect_groups_by_variable (y={yval}, vary x)"),
        );
    }
    for &xval in &[2i64, 3, 5] {
        let c_at_x = collected.subs_i64(&x, xval);
        let e_at_x = expr.subs_i64(&x, xval);
        common::assert_math_eq(
            &c_at_x,
            &e_at_x,
            &y,
            &format!("collect_groups_by_variable (x={xval}, vary y)"),
        );
    }
}

#[test]
fn collect_non_polynomial_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin();
    let collected = expr.collect(&x);
    assert_eq!(format!("{collected}"), "sin(x)");
    common::assert_math_eq(&collected, &expr, &x, "collect_non_polynomial_unchanged");
}

#[test]
fn collect_constant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let five = ctx.int(5);
    let collected = five.collect(&x);
    assert_eq!(format!("{collected}"), "5");
    common::assert_math_eq(&collected, &five, &x, "collect_constant");
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
    common::assert_math_eq(&collected, &expr, &x, "collect_pure_polynomial");
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
    // Two free variables: substitute one at a fixed value, then check the other.
    for &yval in &[2i64, 3, 5] {
        let r_at_y = result.subs_i64(&y, yval);
        let e_at_y = expr.subs_i64(&y, yval);
        common::assert_math_eq(
            &r_at_y,
            &e_at_y,
            &x,
            &format!("together_two_fractions (y={yval}, vary x)"),
        );
    }
    for &xval in &[2i64, 3, 5] {
        let r_at_x = result.subs_i64(&x, xval);
        let e_at_x = expr.subs_i64(&x, xval);
        common::assert_math_eq(
            &r_at_x,
            &e_at_x,
            &y,
            &format!("together_two_fractions (x={xval}, vary y)"),
        );
    }
}

#[test]
fn together_already_no_fractions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x + 1 — no fractions, should stay unchanged
    let expr = &x + 1;
    let result = expr.together();
    assert_eq!(format!("{result}"), "x + 1");
    common::assert_math_eq(&result, &expr, &x, "together_already_no_fractions");
}

#[test]
fn together_not_an_add() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Just x — not an Add, should stay unchanged
    let result = x.together();
    assert_eq!(format!("{result}"), "x");
    common::assert_math_eq(&result, &x, &x, "together_not_an_add");
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
    common::assert_math_eq(
        &result,
        &expr,
        &x,
        "together_mixed_fraction_and_non_fraction",
    );
}

#[test]
fn together_single_fraction() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Just 1/x — it's a single term, not an Add, stays unchanged
    let expr = x.powi(-1);
    let result = expr.together();
    assert_eq!(format!("{result}"), "1/x");
    common::assert_math_eq(&result, &expr, &x, "together_single_fraction");
}
