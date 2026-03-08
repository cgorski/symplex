//! Public API tests for Ex::limit().

use symplex::prelude::*;

#[test]
fn limit_polynomial_direct() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // lim_{x→2} x^2 = 4
    let result = x.powi(2).limit(&x, &ctx.int(2)).unwrap();
    assert_eq!(format!("{result}"), "4");
}

#[test]
fn limit_sin_x_over_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // lim_{x→0} sin(x)/x = 1
    let expr = &x.sin() / &x;
    let result = expr.limit(&x, &ctx.int(0)).unwrap();
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn limit_x_sq_minus_1_over_x_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // lim_{x→1} (x^2-1)/(x-1) = 2
    let expr = (&x.powi(2) - 1) / (&x - 1);
    let result = expr.limit(&x, &ctx.int(1)).unwrap();
    assert_eq!(format!("{result}"), "2");
}

#[test]
fn limit_constant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = ctx.int(5).limit(&x, &ctx.int(0)).unwrap();
    assert_eq!(format!("{result}"), "5");
}

#[test]
fn limit_direct_substitution() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // lim_{x→3} (x+1) = 4
    let result = (&x + 1).limit(&x, &ctx.int(3)).unwrap();
    assert_eq!(format!("{result}"), "4");
}

#[test]
fn limit_cos_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.cos().limit(&x, &ctx.int(0)).unwrap();
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn limit_lhopital_0_over_0() {
    // lim(x→0) (e^x - 1) / x = 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let numer = x.exp() - 1;
    let expr = numer / &x;
    let result = expr.limit(&x, &ctx.int(0)).unwrap();
    let val = result.eval_f64().expect("limit should evaluate to f64");
    assert!((val - 1.0).abs() < 1e-8, "lim should be 1, got {}", val);
}

#[test]
fn limit_lhopital_repeated() {
    // lim(x→0) (e^x - 1 - x) / x² = 1/2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let numer = x.exp() - 1 - &x;
    let denom = x.powi(2);
    let expr = numer / denom;
    let result = expr.limit(&x, &ctx.int(0)).unwrap();
    let val = result.eval_f64().expect("limit should evaluate to f64");
    assert!((val - 0.5).abs() < 1e-8, "lim should be 0.5, got {}", val);
}
