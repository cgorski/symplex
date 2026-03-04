//! Tests for math rules sprint: power-of-power, inverse hyperbolic compositions,
//! nth root evaluation, irrational trig values, hyperbolic odd/even, log expansion.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Power of power simplification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_pow_pow_integers() {
    let x = symplex::var("x");
    // (x^2)^3 → x^6
    let expr = x.powi(2).powi(3);
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "x^6");
}

#[test]
fn simplify_pow_pow_in_expression() {
    let x = symplex::var("x");
    // (x^2)^3 + 1 → x^6 + 1
    let expr = &x.powi(2).powi(3) + 1;
    let simplified = expr.simplify();
    let s = format!("{simplified}");
    assert!(s.contains("x^6"), "should contain x^6: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Inverse hyperbolic composition rules
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_asinh_sinh() {
    let x = symplex::var("x");
    let expr = x.sinh().asinh();
    assert_eq!(format!("{}", expr.simplify()), "x");
}

#[test]
fn simplify_acosh_cosh() {
    let x = symplex::var("x");
    let expr = x.cosh().acosh();
    assert_eq!(format!("{}", expr.simplify()), "abs(x)");
}

#[test]
fn simplify_atanh_tanh() {
    let x = symplex::var("x");
    let expr = x.tanh().atanh();
    assert_eq!(format!("{}", expr.simplify()), "x");
}

// ═══════════════════════════════════════════════════════════════════════════
// Perfect nth root evaluation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_cube_root_8() {
    let ctx = Context::new();
    let expr = ctx.int(8).cbrt();
    assert_eq!(format!("{}", expr.eval()), "2");
}

#[test]
fn eval_cube_root_27() {
    let ctx = Context::new();
    let expr = ctx.int(27).cbrt();
    assert_eq!(format!("{}", expr.eval()), "3");
}

#[test]
fn eval_fourth_root_16() {
    let ctx = Context::new();
    let expr = ctx.int(16).nthroot(4);
    assert_eq!(format!("{}", expr.eval()), "2");
}

#[test]
fn eval_cube_root_non_perfect() {
    let ctx = Context::new();
    let expr = ctx.int(7).cbrt();
    // 7^(1/3) is not a perfect cube — should stay symbolic
    let s = format!("{}", expr.eval());
    assert!(s.contains("7"), "should stay as 7^(1/3): {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Irrational trig special values
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_sin_pi_over_4() {
    let ctx = Context::new();
    let expr = (&ctx.pi() / 4).sin();
    let result = expr.eval();
    let s = format!("{result}");
    // sin(π/4) = √2/2
    assert!(s.contains("sqrt(2)") || s.contains("2^(1/2)"), "got: {s}");
}

#[test]
fn eval_cos_pi_over_4() {
    let ctx = Context::new();
    let expr = (&ctx.pi() / 4).cos();
    let result = expr.eval();
    let s = format!("{result}");
    assert!(s.contains("sqrt(2)") || s.contains("2^(1/2)"), "got: {s}");
}

#[test]
fn eval_sin_pi_over_3() {
    let ctx = Context::new();
    let expr = (&ctx.pi() / 3).sin();
    let result = expr.eval();
    let s = format!("{result}");
    // sin(π/3) = √3/2
    assert!(s.contains("sqrt(3)") || s.contains("3^(1/2)"), "got: {s}");
}

#[test]
fn eval_cos_pi_over_6() {
    let ctx = Context::new();
    let expr = (&ctx.pi() / 6).cos();
    let result = expr.eval();
    let s = format!("{result}");
    assert!(s.contains("sqrt(3)") || s.contains("3^(1/2)"), "got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Hyperbolic odd/even in eval
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_sinh_neg_x() {
    let x = symplex::var("x");
    let expr = (-&x).sinh();
    let evaled = expr.eval();
    assert_eq!(format!("{evaled}"), "-sinh(x)");
}

#[test]
fn eval_cosh_neg_x() {
    let x = symplex::var("x");
    let expr = (-&x).cosh();
    let evaled = expr.eval();
    assert_eq!(format!("{evaled}"), "cosh(x)");
}

#[test]
fn eval_tanh_neg_x() {
    let x = symplex::var("x");
    let expr = (-&x).tanh();
    let evaled = expr.eval();
    assert_eq!(format!("{evaled}"), "-tanh(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Logarithm expansion
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expand_log_product() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let expr = (&x * &y).ln();
    let expanded = expr.expand_log();
    let s = format!("{expanded}");
    assert!(s.contains("ln(x)") && s.contains("ln(y)"), "got: {s}");
}

#[test]
fn expand_log_power() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2).ln();
    let expanded = expr.expand_log();
    let s = format!("{expanded}");
    assert!(s.contains("2") && s.contains("ln(x)"), "got: {s}");
}

#[test]
fn expand_log_quotient() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let expr = (&x / &y).ln();
    let expanded = expr.expand_log();
    let s = format!("{expanded}");
    assert!(s.contains("ln(x)") && s.contains("ln(y)"), "got: {s}");
}

#[test]
fn expand_log_bare_unchanged() {
    let x = symplex::var("x");
    let expr = x.ln();
    let expanded = expr.expand_log();
    assert_eq!(format!("{expanded}"), "ln(x)");
}
