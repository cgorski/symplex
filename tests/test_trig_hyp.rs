//! Tests for inverse trig (asin, acos, atan) and hyperbolic (sinh, cosh, tanh) functions.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Construction and display
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn display_inverse_trig() {
    let x = symplex::var("x");
    assert_eq!(format!("{}", x.asin()), "asin(x)");
    assert_eq!(format!("{}", x.acos()), "acos(x)");
    assert_eq!(format!("{}", x.atan()), "atan(x)");
}

#[test]
fn display_hyperbolic() {
    let x = symplex::var("x");
    assert_eq!(format!("{}", x.sinh()), "sinh(x)");
    assert_eq!(format!("{}", x.cosh()), "cosh(x)");
    assert_eq!(format!("{}", x.tanh()), "tanh(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Differentiation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_asin() {
    let x = symplex::var("x");
    let d = x.asin().diff(&x);
    let s = format!("{d}");
    assert!(s.contains("sqrt"), "d/dx(asin(x)) should contain sqrt: {s}");
}

#[test]
fn diff_acos() {
    let x = symplex::var("x");
    let d = x.acos().diff(&x);
    let s = format!("{d}");
    assert!(s.contains("sqrt"), "d/dx(acos(x)) should contain sqrt: {s}");
}

#[test]
fn diff_atan() {
    let x = symplex::var("x");
    let d = x.atan().diff(&x);
    let s = format!("{d}");
    // d/dx(atan(x)) = 1/(1+x^2)
    assert!(s.contains("x^2"), "d/dx(atan(x)) should contain x^2: {s}");
}

#[test]
fn diff_sinh() {
    let x = symplex::var("x");
    let d = x.sinh().diff(&x);
    assert_eq!(format!("{d}"), "cosh(x)");
}

#[test]
fn diff_cosh() {
    let x = symplex::var("x");
    let d = x.cosh().diff(&x);
    assert_eq!(format!("{d}"), "sinh(x)");
}

#[test]
fn diff_tanh() {
    let x = symplex::var("x");
    let d = x.tanh().diff(&x);
    let s = format!("{d}");
    assert!(s.contains("tanh"), "d/dx(tanh(x)) should contain tanh: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Eval special values
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_asin_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).asin().eval();
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn eval_asin_one() {
    let ctx = Context::new();
    let result = ctx.int(1).asin().eval();
    // asin(1) = pi/2
    let s = format!("{result}");
    assert!(s.contains("pi"), "asin(1) should contain pi: {s}");
}

#[test]
fn eval_acos_one() {
    let ctx = Context::new();
    let result = ctx.int(1).acos().eval();
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn eval_atan_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).atan().eval();
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn eval_sinh_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).sinh().eval();
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn eval_cosh_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).cosh().eval();
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn eval_tanh_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).tanh().eval();
    assert_eq!(format!("{result}"), "0");
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_sinh() {
    let x = symplex::var("x");
    let result = x.sinh().integrate(&x);
    assert_eq!(format!("{result}"), "cosh(x)");
}

#[test]
fn integrate_cosh() {
    let x = symplex::var("x");
    let result = x.cosh().integrate(&x);
    assert_eq!(format!("{result}"), "sinh(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Numerical evaluation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn evalf_sinh_zero() {
    let ctx = Context::new();
    let val = ctx.int(0).sinh().evalf_f64().unwrap();
    assert!((val - 0.0).abs() < 1e-10);
}

#[test]
fn evalf_cosh_zero() {
    let ctx = Context::new();
    let val = ctx.int(0).cosh().evalf_f64().unwrap();
    assert!((val - 1.0).abs() < 1e-10);
}

#[test]
fn evalf_atan_one() {
    let ctx = Context::new();
    let val = ctx.int(1).atan().evalf_f64().unwrap();
    assert!((val - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// Parser integration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn parse_inverse_trig() {
    let ctx = Context::new();
    let e = symplex::parse::parse(&ctx, "asin(x) + acos(x) + atan(x)").unwrap();
    let s = format!("{e}");
    assert!(
        s.contains("asin") && s.contains("acos") && s.contains("atan"),
        "got: {s}"
    );
}

#[test]
fn parse_hyperbolic() {
    let ctx = Context::new();
    let e = symplex::parse::parse(&ctx, "sinh(x) + cosh(x) + tanh(x)").unwrap();
    let s = format!("{e}");
    assert!(
        s.contains("sinh") && s.contains("cosh") && s.contains("tanh"),
        "got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// LaTeX output
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn latex_inverse_trig() {
    let x = symplex::var("x");
    assert!(x.asin().to_latex().contains("\\arcsin"));
    assert!(x.acos().to_latex().contains("\\arccos"));
    assert!(x.atan().to_latex().contains("\\arctan"));
}

#[test]
fn latex_hyperbolic() {
    let x = symplex::var("x");
    assert!(x.sinh().to_latex().contains("\\sinh"));
    assert!(x.cosh().to_latex().contains("\\cosh"));
    assert!(x.tanh().to_latex().contains("\\tanh"));
}
