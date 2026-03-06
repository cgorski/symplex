//! Integration tests for math depth improvements (Cycle 6+).

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Integration: sinh/cosh u-substitution
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_sinh_2x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two_x = &x * 2;
    let result = two_x.sinh().integrate(&x);
    let s = format!("{result}");
    assert!(
        s.contains("cosh"),
        "∫ sinh(2x) dx should involve cosh, got: {s}"
    );
}

#[test]
fn integrate_cosh_3x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let three_x = &x * 3;
    let result = three_x.cosh().integrate(&x);
    let s = format!("{result}");
    assert!(
        s.contains("sinh"),
        "∫ cosh(3x) dx should involve sinh, got: {s}"
    );
}

#[test]
fn integrate_sinh_x_basic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.sinh().integrate(&x);
    let s = format!("{result}");
    assert!(
        s.contains("cosh"),
        "∫ sinh(x) dx should be cosh(x), got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Simplification: forward inverse function rules
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_sin_of_asin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.asin().sin();
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "x");
}

#[test]
fn simplify_cos_of_acos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.acos().cos();
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "x");
}

#[test]
fn simplify_tan_of_atan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.atan().tan();
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "x");
}

#[test]
fn simplify_sinh_of_asinh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.asinh().sinh();
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "x");
}

#[test]
fn simplify_cosh_of_acosh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.acosh().cosh();
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "x");
}

#[test]
fn simplify_tanh_of_atanh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.atanh().tanh();
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "x");
}

// ═══════════════════════════════════════════════════════════════════════════
// Eval: ln of negative numbers
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_ln_neg_one_is_i_pi() {
    let ctx = Context::new();
    let result = ctx.int(-1).ln().eval();
    let s = format!("{result}");
    assert!(
        s.contains("I") && s.contains("pi"),
        "ln(-1) should be i*pi, got: {s}"
    );
}

#[test]
fn eval_ln_neg_two() {
    let ctx = Context::new();
    let result = ctx.int(-2).ln().eval();
    let s = format!("{result}");
    assert!(
        s.contains("ln") && s.contains("I"),
        "ln(-2) should be ln(2)+i*pi, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Eval: inverse trig special values
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_asin_half() {
    let ctx = Context::new();
    let result = ctx.rational(1, 2).asin().eval();
    let s = format!("{result}");
    assert!(s.contains("pi"), "asin(1/2) should be π/6, got: {s}");
}

#[test]
fn eval_acos_half() {
    let ctx = Context::new();
    let result = ctx.rational(1, 2).acos().eval();
    let s = format!("{result}");
    assert!(s.contains("pi"), "acos(1/2) should be π/3, got: {s}");
}

#[test]
fn eval_asin_neg_half() {
    let ctx = Context::new();
    let result = ctx.rational(-1, 2).asin().eval();
    let s = format!("{result}");
    assert!(s.contains("pi"), "asin(-1/2) should be -π/6, got: {s}");
}

#[test]
fn eval_acos_neg_half() {
    let ctx = Context::new();
    let result = ctx.rational(-1, 2).acos().eval();
    let s = format!("{result}");
    assert!(s.contains("pi"), "acos(-1/2) should be 2π/3, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Cross-cutting: complex + simplify roundtrips
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn i_squared_in_expression() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let x = ctx.symbol("x");
    // x + i^2 should canonicalize to x - 1
    let expr = &x + &i.powi(2);
    let s = format!("{expr}");
    assert_eq!(
        s, "x - 1",
        "x + i² should be x - 1, got: {s}"
    );
}

#[test]
fn euler_identity_zero() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let euler = &(&i * &ctx.pi()).exp().eval() + 1;
    assert_eq!(format!("{euler}"), "0", "e^(iπ) + 1 should be 0");
}
