//! Tests for simplification rules:
//! inverse-trig compositions are NOT simplified (correctness),
//! cosh²-sinh²→1

use symplex::prelude::*;
#[test]
fn simplify_asin_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin().asin();
    // asin(sin(x)) does NOT simplify to x (not valid for all x)
    assert_eq!(format!("{}", expr.simplify()), "asin(sin(x))");
}

#[test]
fn simplify_acos_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.cos().acos();
    // acos(cos(x)) does NOT simplify to x (not valid for all x)
    assert_eq!(format!("{}", expr.simplify()), "acos(cos(x))");
}

#[test]
fn simplify_atan_tan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.tan().atan();
    // atan(tan(x)) does NOT simplify to x (not valid for all x)
    assert_eq!(format!("{}", expr.simplify()), "atan(tan(x))");
}

#[test]
fn simplify_cosh_sinh_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // cosh²(x) - sinh²(x) = 1
    let expr = &x.cosh().powi(2) - &x.sinh().powi(2);
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "1");
}

#[test]
fn simplify_cosh_sinh_in_larger_sum() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 5 + cosh²(x) - sinh²(x) = 6
    let expr = &x.cosh().powi(2) - &x.sinh().powi(2) + 5;
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "6");
}

#[test]
fn simplify_inverse_trig_nested() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // asin(sin(x)) + 1 stays unsimplified (rule removed for correctness)
    let expr = &x.sin().asin() + 1;
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "asin(sin(x)) + 1");
}

#[test]
fn full_simplify_inverse_trig() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin().asin();
    // full_simplify also does not cancel asin(sin(x)) (correctness)
    assert_eq!(format!("{}", expr.full_simplify()), "asin(sin(x))");
}
