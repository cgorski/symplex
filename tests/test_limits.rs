//! Public API tests for Ex::limit().

use symplex::prelude::*;

#[test]
fn limit_polynomial_direct() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // lim_{x→2} x^2 = 4
    let result = x.powi(2).limit(&x, &ctx.int(2));
    assert_eq!(format!("{result}"), "4");
}

#[test]
fn limit_sin_x_over_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // lim_{x→0} sin(x)/x = 1
    let expr = &x.sin() / &x;
    let result = expr.limit(&x, &ctx.int(0));
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn limit_x_sq_minus_1_over_x_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // lim_{x→1} (x^2-1)/(x-1) = 2
    let expr = (&x.powi(2) - 1) / (&x - 1);
    let result = expr.limit(&x, &ctx.int(1));
    assert_eq!(format!("{result}"), "2");
}

#[test]
fn limit_constant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = ctx.int(5).limit(&x, &ctx.int(0));
    assert_eq!(format!("{result}"), "5");
}

#[test]
fn limit_direct_substitution() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // lim_{x→3} (x+1) = 4
    let result = (&x + 1).limit(&x, &ctx.int(3));
    assert_eq!(format!("{result}"), "4");
}

#[test]
fn limit_cos_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.cos().limit(&x, &ctx.int(0));
    assert_eq!(format!("{result}"), "1");
}
