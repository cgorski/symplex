//! Public API tests for Ex::integrate().

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Power rule
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.integrate(&x);
    assert_eq!(format!("{result}"), "1/2*x^2");
}

#[test]
fn integrate_x_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(2).integrate(&x);
    assert_eq!(format!("{result}"), "1/3*x^3");
}

#[test]
fn integrate_x_cubed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(3).integrate(&x);
    assert_eq!(format!("{result}"), "1/4*x^4");
}

#[test]
fn integrate_x_inverse() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(-1).integrate(&x);
    assert_eq!(format!("{result}"), "ln(abs(x))");
}

#[test]
fn integrate_x_neg2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(-2).integrate(&x);
    // ∫ x^(-2) dx = x^(-1) / (-1) = -x^(-1)
    let s = format!("{result}");
    assert!(s.contains("1/x"), "should contain 1/x: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_constant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = ctx.int(5).integrate(&x);
    assert_eq!(format!("{result}"), "5*x");
}

#[test]
fn integrate_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = ctx.int(0).integrate(&x);
    // 0 * x should canonicalize to 0
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn integrate_other_symbol() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // ∫ y dx = y*x
    let result = y.integrate(&x);
    assert_eq!(format!("{result}"), "x*y");
}

// ═══════════════════════════════════════════════════════════════════════════
// Trig and exp
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_sin_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.sin().integrate(&x);
    assert_eq!(format!("{result}"), "-cos(x)");
}

#[test]
fn integrate_cos_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.cos().integrate(&x);
    assert_eq!(format!("{result}"), "sin(x)");
}

#[test]
fn integrate_exp_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.exp().integrate(&x);
    assert_eq!(format!("{result}"), "exp(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Linearity and constant factor
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_sum() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ∫ (x + 1) dx = x^2/2 + x
    let result = (&x + 1).integrate(&x);
    let s = format!("{result}");
    assert!(s.contains("x^2"), "should have x^2 term: {s}");
    assert!(s.contains("x"), "should have x term: {s}");
}

#[test]
fn integrate_constant_times_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ∫ 3x dx = 3/2 * x^2
    let result = (&x * 3).integrate(&x);
    assert_eq!(format!("{result}"), "3/2*x^2");
}

// ═══════════════════════════════════════════════════════════════════════════
// Unevaluated
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_tan_unevaluated() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.tan().integrate(&x);
    assert_eq!(format!("{result}"), "Integral(tan(x), x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Verification: differentiate the integral should give back the integrand
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_then_diff_power() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // d/dx(∫ x^3 dx) should give x^3
    let anti = x.powi(3).integrate(&x);
    let back = anti.diff(&x);
    assert_eq!(format!("{back}"), "x^3");
}

#[test]
fn integrate_then_diff_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // d/dx(∫ sin(x) dx) = d/dx(-cos(x)) = sin(x)
    let anti = x.sin().integrate(&x);
    let back = anti.diff(&x);
    assert_eq!(format!("{back}"), "sin(x)");
}
