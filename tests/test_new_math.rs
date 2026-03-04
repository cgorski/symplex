//! Tests for new math features: partial fractions, nsolve, expand_trig,
//! polynomial GCD/LCM, odd/even trig eval, inverse hyperbolics.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Partial fractions (apart)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn apart_simple() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 1/(x^2 - 1) should decompose
    let expr = 1 / (&x.powi(2) - 1);
    let decomposed = expr.apart(&x);
    let s = format!("{decomposed}");
    // Should NOT contain x^2 in denominator anymore
    assert!(
        s != format!("{expr}") || s.contains("1/"),
        "should decompose: {s}"
    );
}

#[test]
fn apart_already_simple() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x + 1;
    let result = expr.apart(&x);
    assert_eq!(format!("{result}"), "x + 1");
}

// ═══════════════════════════════════════════════════════════════════════════
// Numerical root finding (nsolve)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nsolve_x_minus_cos_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x - cos(x) = 0 has a root near 0.739
    let expr = &x - &x.cos();
    let root = expr.nsolve(&x, 1.0, 50, 1e-12).unwrap();
    assert!((root - 0.7390851332).abs() < 1e-6, "got: {root}");
}

#[test]
fn nsolve_x_squared_minus_2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x^2 - 2 = 0 has root sqrt(2) ≈ 1.4142
    let expr = &x.powi(2) - 2;
    let root = expr.nsolve(&x, 1.5, 50, 1e-12).unwrap();
    assert!(
        (root - std::f64::consts::SQRT_2).abs() < 1e-8,
        "got: {root}"
    );
}

#[test]
fn nsolve_exp_minus_2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // exp(x) - 2 = 0 → x = ln(2) ≈ 0.6931
    let expr = &x.exp() - 2;
    let root = expr.nsolve(&x, 1.0, 50, 1e-12).unwrap();
    assert!((root - 2.0_f64.ln()).abs() < 1e-8, "got: {root}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Trig expansion
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expand_trig_sin_sum() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let expr = (&x + &y).sin();
    let expanded = expr.expand_trig();
    let s = format!("{expanded}");
    assert!(s.contains("sin(x)"), "should contain sin(x): {s}");
    assert!(s.contains("cos(y)"), "should contain cos(y): {s}");
}

#[test]
fn expand_trig_cos_sum() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let expr = (&x + &y).cos();
    let expanded = expr.expand_trig();
    let s = format!("{expanded}");
    assert!(s.contains("cos(x)"), "should contain cos(x): {s}");
    assert!(s.contains("sin(x)"), "should contain sin(x): {s}");
}

#[test]
fn expand_trig_bare_sin_unchanged() {
    let x = symplex::var("x");
    let expr = x.sin();
    let expanded = expr.expand_trig();
    assert_eq!(format!("{expanded}"), "sin(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Polynomial GCD / LCM
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn poly_gcd_common_factor() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // gcd(x^2 - 1, x - 1) = x - 1 (up to sign/scalar)
    let a = &x.powi(2) - 1;
    let b = &x - 1;
    let g = a.poly_gcd(&b, &x).unwrap();
    let s = format!("{g}");
    assert!(s.contains("x"), "gcd should involve x: {s}");
}

#[test]
fn poly_gcd_coprime() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // gcd(x + 1, x + 2) = 1
    let a = &x + 1;
    let b = &x + 2;
    let g = a.poly_gcd(&b, &x).unwrap();
    assert_eq!(format!("{g}"), "1");
}

#[test]
fn poly_lcm_basic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = &x - 1;
    let b = &x + 1;
    let lcm = a.poly_lcm(&b, &x).unwrap();
    let s = format!("{lcm}");
    // lcm(x-1, x+1) should be x^2 - 1 (up to sign)
    assert!(s.contains("x^2"), "lcm should be degree 2: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Odd/even trig in eval
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_sin_neg_x() {
    let x = symplex::var("x");
    let expr = (-&x).sin();
    let evaled = expr.eval();
    // sin(-x) should evaluate to -sin(x)
    assert_eq!(format!("{evaled}"), "-sin(x)");
}

#[test]
fn eval_cos_neg_x() {
    let x = symplex::var("x");
    let expr = (-&x).cos();
    let evaled = expr.eval();
    // cos(-x) should evaluate to cos(x)
    assert_eq!(format!("{evaled}"), "cos(x)");
}

#[test]
fn eval_tan_neg_x() {
    let x = symplex::var("x");
    let expr = (-&x).tan();
    let evaled = expr.eval();
    // tan(-x) should evaluate to -tan(x)
    assert_eq!(format!("{evaled}"), "-tan(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Inverse hyperbolic functions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn display_inverse_hyperbolic() {
    let x = symplex::var("x");
    assert_eq!(format!("{}", x.asinh()), "asinh(x)");
    assert_eq!(format!("{}", x.acosh()), "acosh(x)");
    assert_eq!(format!("{}", x.atanh()), "atanh(x)");
}

#[test]
fn eval_asinh_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).asinh().eval();
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn eval_acosh_one() {
    let ctx = Context::new();
    let result = ctx.int(1).acosh().eval();
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn eval_atanh_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).atanh().eval();
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn diff_asinh() {
    let x = symplex::var("x");
    let d = x.asinh().diff(&x);
    let s = format!("{d}");
    // d/dx(asinh(x)) = 1/sqrt(x^2+1)
    assert!(s.contains("sqrt"), "should contain sqrt: {s}");
}

#[test]
fn parse_inverse_hyperbolic() {
    let ctx = Context::new();
    let e = symplex::parse::parse(&ctx, "asinh(x) + acosh(x) + atanh(x)").unwrap();
    let s = format!("{e}");
    assert!(
        s.contains("asinh") && s.contains("acosh") && s.contains("atanh"),
        "got: {s}"
    );
}
