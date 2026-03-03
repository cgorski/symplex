//! Public API tests for Ex::series() and Ex::factor().

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// series()
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn series_exp_x_order_4() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    // exp(x) around 0, order 4: 1 + x + x^2/2 + x^3/6
    let s = x.exp_fn().series(&x, &zero, 4).unwrap();
    let result = s.expand().eval();
    let text = format!("{result}");
    assert!(text.contains("1"), "constant term 1: {text}");
    assert!(text.contains("x"), "x term: {text}");
    assert!(text.contains("x^2"), "x^2 term: {text}");
    assert!(text.contains("x^3"), "x^3 term: {text}");
}

#[test]
fn series_sin_x_order_4() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    // sin(x) around 0, order 4: x - x^3/6
    let s = x.sin().series(&x, &zero, 4).unwrap();
    let result = s.expand().eval();
    let text = format!("{result}");
    assert!(text.contains("x"), "should have x term: {text}");
    assert!(text.contains("x^3"), "should have x^3 term: {text}");
}

#[test]
fn series_cos_x_order_4() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    // cos(x) around 0, order 4: 1 - x^2/2
    let s = x.cos().series(&x, &zero, 4).unwrap();
    let result = s.expand().eval();
    let text = format!("{result}");
    assert!(text.contains("1"), "constant term 1: {text}");
    assert!(text.contains("x^2"), "should have x^2 term: {text}");
}

#[test]
fn series_polynomial_is_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    // series(x^2, x, 0, 5) should give exactly x^2
    let s = x.powi(2).series(&x, &zero, 5).unwrap();
    assert_eq!(format!("{s}"), "x^2");
}

#[test]
fn series_constant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let s = ctx.int(7).series(&x, &zero, 3).unwrap();
    assert_eq!(format!("{s}"), "7");
}

#[test]
fn series_order_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let s = x.series(&x, &zero, 0).unwrap();
    assert_eq!(format!("{s}"), "0");
}

#[test]
fn series_x_around_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let s = x.series(&x, &zero, 3).unwrap();
    assert_eq!(format!("{s}"), "x");
}

// ═══════════════════════════════════════════════════════════════════════════
// factor()
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_x_squared_minus_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(2) - 1;
    let factored = expr.factor(&x);
    let s = format!("{factored}");
    // Should be factored — no x^2
    assert!(!s.contains("x^2"), "should be factored (no x^2): {s}");
}

#[test]
fn factor_quadratic_two_roots() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x^2 - 5x + 6 = (x-2)(x-3)
    let expr = &x.powi(2) - &x * 5 + 6;
    let factored = expr.factor(&x);
    let s = format!("{factored}");
    assert!(!s.contains("x^2"), "should be factored: {s}");
}

#[test]
fn factor_with_content() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 2*x^2 - 2 = 2*(x-1)*(x+1)
    let expr = &x.powi(2) * 2 - 2;
    let factored = expr.factor(&x);
    let s = format!("{factored}");
    assert!(!s.contains("x^2"), "should be factored: {s}");
    assert!(s.contains("2"), "should have content factor 2: {s}");
}

#[test]
fn factor_no_rational_roots() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x^2 + 1 — no real rational roots
    let expr = &x.powi(2) + 1;
    let factored = expr.factor(&x);
    let s = format!("{factored}");
    assert!(s.contains("x^2"), "should stay unfactored: {s}");
}

#[test]
fn factor_already_linear() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x + 1;
    let factored = expr.factor(&x);
    assert_eq!(format!("{factored}"), "1 + x");
}

#[test]
fn factor_cubic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
    let expr = &x.powi(3) - &x.powi(2) * 6 + &x * 11 - 6;
    let factored = expr.factor(&x);
    let s = format!("{factored}");
    assert!(!s.contains("x^3"), "should be fully factored: {s}");
}

#[test]
fn factor_non_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin();
    let factored = expr.factor(&x);
    assert_eq!(format!("{factored}"), "sin(x)");
}

#[test]
fn factor_constant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let factored = ctx.int(42).factor(&x);
    assert_eq!(format!("{factored}"), "42");
}

#[test]
fn factor_double_root() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x^2 - 2x + 1 = (x-1)^2
    let expr = &x.powi(2) - &x * 2 + 1;
    let factored = expr.factor(&x);
    let s = format!("{factored}");
    // Should show the double root — either as (x-1)^2 or (-1+x)^2 or (-1+x)*(-1+x)
    assert!(!s.contains("x^2"), "should be factored: {s}");
}
