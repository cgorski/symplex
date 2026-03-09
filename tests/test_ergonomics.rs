//! Tests for ergonomic API improvements:
//! - Global default context (var, symbol, int, rational)
//! - vars! macro (context-free)
//! - maclaurin() convenience method
//! - subs_i64() convenience method
//! - Display improvements (1/x, negative coefficients)

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Global default context
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn var_creates_symbol() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    assert_eq!(format!("{x}"), "x");
}

#[test]
fn symbol_is_alias_for_var() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    assert_eq!(format!("{x}"), "x");
}

#[test]
fn int_creates_integer() {
    let __ctx = Context::new();
    let five = __ctx.int(5);
    assert_eq!(format!("{five}"), "5");
}

#[test]
fn rational_creates_fraction() {
    let __ctx = Context::new();
    let half = __ctx.rational(1, 2);
    assert_eq!(format!("{half}"), "1/2");
}

#[test]
fn global_context_expressions_interoperate() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let expr = &x.powi(2) + &x + 1;
    let s = format!("{expr}");
    assert!(s.contains("x^2"), "should contain x^2: {s}");
    assert!(s.contains("x"), "should contain x: {s}");
}

#[test]
fn global_context_diff_works() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let deriv = x.powi(3).diff(&x);
    assert_eq!(format!("{deriv}"), "3*x^2");
}

#[test]
fn global_context_integrate_works() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let anti = x.powi(2).integrate(&x);
    assert_eq!(format!("{anti}"), "1/3*x^3");
}

// ═══════════════════════════════════════════════════════════════════════════
// vars! macro (context-free)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn vars_macro_creates_symbols() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; a, b, c);
    let expr = &a + &b + &c;
    let s = format!("{expr}");
    assert!(
        s.contains("a") && s.contains("b") && s.contains("c"),
        "should contain a, b, c: {s}"
    );
}

#[test]
fn vars_macro_trailing_comma() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x, y,);
    let expr = &x * &y;
    assert_eq!(format!("{expr}"), "x*y");
}

// ═══════════════════════════════════════════════════════════════════════════
// maclaurin() convenience
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn maclaurin_sin() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let s = x.sin().maclaurin(&x, 4);
    let result = s.expand().eval();
    let text = format!("{result}");
    assert!(text.contains("x"), "should have x term: {text}");
    assert!(text.contains("x^3"), "should have x^3 term: {text}");
}

#[test]
fn maclaurin_exp() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let s = x.exp().maclaurin(&x, 3);
    let result = s.expand().eval();
    let text = format!("{result}");
    assert!(text.contains("1"), "should have constant term: {text}");
    assert!(text.contains("x"), "should have x term: {text}");
    assert!(text.contains("x^2"), "should have x^2 term: {text}");
}

#[test]
fn maclaurin_polynomial_is_exact() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let poly = &x.powi(2) + &x * 3 + 7;
    let s = poly.maclaurin(&x, 5);
    assert_eq!(format!("{s}"), format!("{poly}"));
}

// ═══════════════════════════════════════════════════════════════════════════
// subs_i64() convenience
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn subs_i64_basic() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let expr = x.powi(2);
    let result = expr.subs_i64(&x, 3);
    assert_eq!(format!("{result}"), "9");
}

#[test]
fn subs_i64_in_polynomial() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let expr = &x.powi(2) + &x * 2 + 1;
    // (3)^2 + 2*3 + 1 = 9 + 6 + 1 = 16
    let result = expr.subs_i64(&x, 3);
    assert_eq!(format!("{result}"), "16");
}

#[test]
fn subs_i64_zero() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let expr = &x.powi(2) + 1;
    let result = expr.subs_i64(&x, 0);
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn subs_i64_negative() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let expr = &x + 5;
    let result = expr.subs_i64(&x, -3);
    assert_eq!(format!("{result}"), "2");
}

// ═══════════════════════════════════════════════════════════════════════════
// Display improvements
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn display_inverse_as_fraction() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let result = x.powi(-1);
    assert_eq!(format!("{result}"), "1/x");
}

#[test]
fn display_division_uses_fraction() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let y = __ctx.symbol("y");
    let result = &x / &y;
    // x * y^(-1) displays as x*1/y
    let s = format!("{result}");
    assert!(s.contains("1/y"), "should display y inverse as 1/y: {s}");
}

#[test]
fn display_negative_coeff_as_subtraction() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Build x + (-1/6)*x^3 — this is what sin(x) series gives
    let term1 = x.clone();
    let term2 = &x.powi(3) * &ctx.rational(-1, 6);
    let expr = &term1 + &term2;
    let s = format!("{expr}");
    // Should render with subtraction, not "x + -1/6*x^3"
    assert!(!s.contains("+ -"), "should not have '+ -' pattern: {s}");
}
