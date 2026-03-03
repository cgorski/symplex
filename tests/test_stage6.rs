//! Stage 6 integration tests for symplex.
//!
//! Tests symbolic differentiation through the public `Ex` API:
//! `ex.diff(&var)`.

use symplex::prelude::*;
use symplex::syms;

// ─── Constants differentiate to zero ──────────────────────────────────────

#[test]
fn diff_integer_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let five = ctx.int(5);
    let result = five.diff(&x);
    assert!(result.is_zero_structural(), "d/dx(5) should be 0");
}

#[test]
fn diff_pi_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = ctx.pi().diff(&x);
    assert!(result.is_zero_structural(), "d/dx(pi) should be 0");
}

#[test]
fn diff_e_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = ctx.e().diff(&x);
    assert!(result.is_zero_structural(), "d/dx(e) should be 0");
}

#[test]
fn diff_other_symbol_is_zero() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let result = y.diff(&x);
    assert!(result.is_zero_structural(), "d/dx(y) should be 0");
}

// ─── d/dx(x) = 1 ─────────────────────────────────────────────────────────

#[test]
fn diff_x_wrt_x_is_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.diff(&x);
    assert!(result.is_one_structural(), "d/dx(x) should be 1");
}

// ─── Linearity ────────────────────────────────────────────────────────────

#[test]
fn diff_sum_is_sum_of_derivatives() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // d/dx(x + 3) = 1
    let expr = &x + 3;
    let result = expr.diff(&x);
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn diff_2x_is_2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x * 2;
    let result = expr.diff(&x);
    assert_eq!(format!("{result}"), "2");
}

#[test]
fn diff_3x_plus_5_is_3() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x * 3 + 5;
    let result = expr.diff(&x);
    assert_eq!(format!("{result}"), "3");
}

// ─── Power rule ───────────────────────────────────────────────────────────

#[test]
fn diff_x_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(2).diff(&x);
    assert_eq!(format!("{result}"), "2*x");
}

#[test]
fn diff_x_cubed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(3).diff(&x);
    assert_eq!(format!("{result}"), "3*x**2");
}

#[test]
fn diff_x_to_the_fourth() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(4).diff(&x);
    assert_eq!(format!("{result}"), "4*x**3");
}

#[test]
fn diff_x_to_the_minus_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(-1).diff(&x);
    // d/dx(x^(-1)) = -1 * x^(-2) = -x^(-2)
    let s = format!("{result}");
    assert!(
        s.contains("-1") || s.contains("-x"),
        "d/dx(1/x) should be negative, got: {s}"
    );
    assert!(s.contains("x"), "should contain x, got: {s}");
}

// ─── Product rule ─────────────────────────────────────────────────────────

#[test]
fn diff_x_times_y() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    // d/dx(x*y) = y
    let expr = &x * &y;
    let result = expr.diff(&x);
    assert_eq!(format!("{result}"), "y");
}

#[test]
fn diff_x_times_sin_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // d/dx(x * sin(x)) = sin(x) + x*cos(x)
    let expr = &x * &x.sin();
    let result = expr.diff(&x);
    let s = format!("{result}");
    assert!(s.contains("sin"), "should contain sin(x), got: {s}");
    assert!(s.contains("cos"), "should contain cos(x), got: {s}");
}

#[test]
fn diff_product_three_symbols() {
    let ctx = Context::new();
    syms!(ctx; x, y, z);
    // d/dx(x*y*z) = y*z
    let expr = &(&x * &y) * &z;
    let result = expr.diff(&x);
    assert_eq!(format!("{result}"), "y*z");
}

// ─── Chain rule with trig ─────────────────────────────────────────────────

#[test]
fn diff_sin_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.sin().diff(&x);
    assert_eq!(format!("{result}"), "cos(x)");
}

#[test]
fn diff_cos_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.cos().diff(&x);
    assert_eq!(format!("{result}"), "-sin(x)");
}

#[test]
fn diff_tan_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.tan().diff(&x);
    let s = format!("{result}");
    // Should be 1 + tan(x)^2  (or sec(x)^2)
    assert!(
        s.contains("tan"),
        "d/dx(tan(x)) should involve tan, got: {s}"
    );
}

#[test]
fn diff_sin_of_x_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // d/dx(sin(x^2)) = 2*x*cos(x^2)
    let result = x.powi(2).sin().diff(&x);
    let s = format!("{result}");
    assert!(s.contains("2"), "should contain 2, got: {s}");
    assert!(s.contains("cos"), "should contain cos, got: {s}");
    assert!(s.contains("x"), "should contain x, got: {s}");
}

#[test]
fn diff_cos_of_3x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // d/dx(cos(3*x)) = -3*sin(3*x)
    let three_x = &x * 3;
    let result = three_x.cos().diff(&x);
    let s = format!("{result}");
    assert!(s.contains("3"), "should contain 3, got: {s}");
    assert!(s.contains("sin"), "should contain sin, got: {s}");
}

// ─── Exponential and logarithm ────────────────────────────────────────────

#[test]
fn diff_exp_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.exp_fn().diff(&x);
    assert_eq!(format!("{result}"), "exp(x)");
}

#[test]
fn diff_ln_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.ln().diff(&x);
    // d/dx(ln(x)) = 1/x = x^(-1)
    assert_eq!(format!("{result}"), "x**(-1)");
}

#[test]
fn diff_exp_of_2x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // d/dx(exp(2*x)) = 2*exp(2*x)
    let two_x = &x * 2;
    let result = two_x.exp_fn().diff(&x);
    let s = format!("{result}");
    assert!(s.contains("2"), "should contain 2, got: {s}");
    assert!(s.contains("exp"), "should contain exp, got: {s}");
}

#[test]
fn diff_ln_of_x_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // d/dx(ln(x^2)) = 2*x / x^2 = 2/x = 2*x^(-1)
    let result = x.powi(2).ln().diff(&x);
    let s = format!("{result}");
    assert!(s.contains("2"), "should contain 2, got: {s}");
    assert!(s.contains("x"), "should contain x, got: {s}");
}

// ─── Sqrt ─────────────────────────────────────────────────────────────────

#[test]
fn diff_sqrt_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.sqrt().diff(&x);
    let s = format!("{result}");
    // d/dx(sqrt(x)) = 1/(2*sqrt(x))
    assert!(
        s.contains("sqrt") || s.contains("1/2"),
        "should involve sqrt or 1/2, got: {s}"
    );
}

// ─── Negation ─────────────────────────────────────────────────────────────

#[test]
fn diff_neg_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = (-&x).diff(&x);
    assert_eq!(format!("{result}"), "-1");
}

#[test]
fn diff_neg_x_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // d/dx(-x^2) = -2*x
    let result = (-&x.powi(2)).diff(&x);
    assert_eq!(format!("{result}"), "-2*x");
}

// ─── Polynomials ──────────────────────────────────────────────────────────

#[test]
fn diff_quadratic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // d/dx(x^2 + 2*x + 1) = 2*x + 2
    let expr = &x.powi(2) + &x * 2 + 1;
    let result = expr.diff(&x);
    let s = format!("{result}");
    assert!(s.contains("2"), "should contain 2, got: {s}");
    assert!(s.contains("x"), "should contain x, got: {s}");
}

#[test]
fn diff_cubic_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // d/dx(x^3 + 2*x^2 + x + 5) = 3*x^2 + 4*x + 1
    let expr = &x.powi(3) + &x.powi(2) * 2 + &x + 5;
    let result = expr.diff(&x);
    let s = format!("{result}");
    assert!(s.contains("3*x**2"), "should contain 3*x^2, got: {s}");
    assert!(s.contains("4*x"), "should contain 4*x, got: {s}");
    assert!(s.contains('1'), "should contain 1, got: {s}");
}

// ─── Higher-order derivatives ─────────────────────────────────────────────

#[test]
fn second_derivative_of_x_cubed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // d²/dx²(x^3) = 6*x
    let first = x.powi(3).diff(&x);
    let second = first.diff(&x);
    assert_eq!(format!("{second}"), "6*x");
}

#[test]
fn third_derivative_of_x_cubed_is_constant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // d³/dx³(x^3) = 6
    let first = x.powi(3).diff(&x);
    let second = first.diff(&x);
    let third = second.diff(&x);
    assert_eq!(format!("{third}"), "6");
}

#[test]
fn fourth_derivative_of_x_cubed_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // d⁴/dx⁴(x^3) = 0
    let result = x.powi(3).diff(&x).diff(&x).diff(&x).diff(&x);
    assert!(
        result.is_zero_structural(),
        "d⁴/dx⁴(x³) should be 0, got: {result}"
    );
}

// ─── Multivariate ─────────────────────────────────────────────────────────

#[test]
fn partial_derivative_x() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    // ∂/∂x(x^2 * y) = 2*x*y
    let expr = &x.powi(2) * &y;
    let result = expr.diff(&x);
    assert_eq!(format!("{result}"), "2*x*y");
}

#[test]
fn partial_derivative_y() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    // ∂/∂y(x^2 * y) = x^2
    let expr = &x.powi(2) * &y;
    let result = expr.diff(&y);
    assert_eq!(format!("{result}"), "x**2");
}

#[test]
fn mixed_partial_derivative() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    // ∂²/∂x∂y(x^2 * y) = 2*x
    let expr = &x.powi(2) * &y;
    let result = expr.diff(&y).diff(&x);
    assert_eq!(format!("{result}"), "2*x");
}

// ─── Combined operations ──────────────────────────────────────────────────

#[test]
fn diff_then_substitute() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // d/dx(x^2) = 2*x, then evaluate at x=3 → 6
    let deriv = x.powi(2).diff(&x);
    let result = deriv.subs(&x, &ctx.int(3));
    assert_eq!(format!("{result}"), "6");
}

#[test]
fn diff_then_substitute_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // d/dx(x^3 + x) = 3*x^2 + 1, evaluate at x=2 → 13
    let expr = &x.powi(3) + &x;
    let deriv = expr.diff(&x);
    let result = deriv.subs(&x, &ctx.int(2));
    assert_eq!(format!("{result}"), "13");
}

// ─── Special cases ────────────────────────────────────────────────────────

#[test]
fn diff_zero_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    assert!(zero.diff(&x).is_zero_structural());
}

#[test]
fn diff_one_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    assert!(one.diff(&x).is_zero_structural());
}

#[test]
fn diff_constant_sum_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    syms!(ctx; y, z);
    let expr = &y + &z + 5;
    assert!(
        expr.diff(&x).is_zero_structural(),
        "d/dx(y + z + 5) should be 0"
    );
}

#[test]
fn diff_preserves_assumptions() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive, Assumption::Real]);
    // d/dx(x^2) = 2*x — the result should still know x is positive.
    let deriv = x.powi(2).diff(&x);
    // The result is 2*x, which should be positive since x is positive.
    assert_eq!(
        ctx.query(&deriv, Props::POSITIVE),
        Some(true),
        "2*x should be positive when x is positive"
    );
}
