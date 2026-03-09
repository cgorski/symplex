//! Tests for limits at infinity.

use symplex::prelude::*;
#[test]
fn limit_one_over_x_at_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / &x;
    let r = expr.limit(&x, &ctx.infinity());
    assert_eq!(format!("{r}"), "0");
}

#[test]
fn limit_constant_at_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = ctx.int(5).limit(&x, &ctx.infinity());
    assert_eq!(format!("{r}"), "5");
}

#[test]
fn limit_x_squared_at_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = x.powi(2).limit(&x, &ctx.infinity());
    let s = format!("{r}");
    assert!(s.contains("oo") || s.contains("∞"), "x²→∞: {s}");
}

#[test]
fn limit_rational_function_same_degree() {
    let ctx = Context::new();
    // lim(x→∞) (3x²+1)/(x²-x) = 3
    let x = ctx.symbol("x");
    let numer = &x.powi(2) * 3 + 1;
    let denom = &x.powi(2) - &x;
    let expr = &numer / &denom;
    let r = expr.limit(&x, &ctx.infinity());
    assert_eq!(format!("{r}"), "3", "lim (3x²+1)/(x²-x) = 3");
}

#[test]
fn limit_rational_lower_numer_degree() {
    let ctx = Context::new();
    // lim(x→∞) x/(x²+1) = 0
    let x = ctx.symbol("x");
    let expr = &x / &(&x.powi(2) + 1);
    let r = expr.limit(&x, &ctx.infinity());
    assert_eq!(format!("{r}"), "0");
}

#[test]
fn limit_one_over_x_at_neg_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / &x;
    let neg_inf = -ctx.infinity();
    let r = expr.limit(&x, &neg_inf);
    assert_eq!(format!("{r}"), "0");
}

#[test]
fn limit_exp_neg_x_at_infinity() {
    let ctx = Context::new();
    // lim(x→∞) exp(-x) = 0
    let x = ctx.symbol("x");
    let expr = (-&x).exp();
    let r = expr.limit(&x, &ctx.infinity());
    assert_eq!(format!("{r}"), "0", "lim exp(-x) = 0");
}

#[test]
fn limit_one_over_x_squared_at_infinity() {
    let ctx = Context::new();
    // lim(x→∞) 1/x² = 0
    let x = ctx.symbol("x");
    let expr = 1 / &x.powi(2);
    let r = expr.limit(&x, &ctx.infinity());
    assert_eq!(format!("{r}"), "0");
}

#[test]
fn limit_rational_higher_numer_degree() {
    let ctx = Context::new();
    // lim(x→∞) x³/(x+1) = ∞
    let x = ctx.symbol("x");
    let expr = &x.powi(3) / &(&x + 1);
    let r = expr.limit(&x, &ctx.infinity());
    let s = format!("{r}");
    assert!(s.contains("oo") || s.contains("∞"), "x³/(x+1)→∞: {s}");
}

#[test]
fn limit_constant_plus_decay_at_infinity() {
    let ctx = Context::new();
    // lim(x→∞) (3 + 1/x) = 3
    let x = ctx.symbol("x");
    let expr = ctx.int(3) + 1 / &x;
    let r = expr.limit(&x, &ctx.infinity());
    assert_eq!(format!("{r}"), "3");
}

#[test]
fn limit_neg_leading_coeff_at_pos_inf() {
    let ctx = Context::new();
    // lim(x→+∞) -x³/(x+1) = -∞
    let x = ctx.symbol("x");
    let neg_x3 = -x.powi(3);
    let denom = &x + 1;
    let expr = &neg_x3 / &denom;
    let result = expr.limit(&x, &ctx.infinity());
    assert!(!result.has_unevaluated(), "limit should succeed");
    assert_eq!(format!("{result}"), "-oo");
}

#[test]
fn limit_neg_leading_coeff_at_neg_inf() {
    let ctx = Context::new();
    // lim(x→-∞) -x³/(x+1) = -∞
    // Leading coeff ratio is -1/1 = negative; degree diff = 2 (even), no sign flip at -∞
    let x = ctx.symbol("x");
    let neg_x3 = -x.powi(3);
    let denom = &x + 1;
    let expr = &neg_x3 / &denom;
    let neg_inf = -ctx.infinity();
    let result = expr.limit(&x, &neg_inf);
    assert!(!result.has_unevaluated(), "limit should succeed");
    assert_eq!(format!("{result}"), "-oo");
}

#[test]
fn limit_neg_over_neg_at_pos_inf() {
    let ctx = Context::new();
    // lim(x→+∞) -x²/(-x+1) = +∞ (negatives cancel)
    let x = ctx.symbol("x");
    let numer = -x.powi(2);
    let denom = -&x + 1;
    let expr = &numer / &denom;
    let result = expr.limit(&x, &ctx.infinity());
    assert!(!result.has_unevaluated(), "limit should succeed");
    assert_eq!(format!("{result}"), "oo");
}

#[test]
fn limit_even_degree_diff_neg_inf() {
    let ctx = Context::new();
    // lim(x→-∞) x⁴/(x²+1) = +∞ (even degree diff, no sign flip)
    let x = ctx.symbol("x");
    let numer = x.powi(4);
    let denom = x.powi(2) + 1;
    let expr = &numer / &denom;
    let neg_inf = -ctx.infinity();
    let result = expr.limit(&x, &neg_inf);
    assert!(!result.has_unevaluated(), "limit should succeed");
    assert_eq!(format!("{result}"), "oo");
}

#[test]
fn limit_odd_degree_diff_neg_inf() {
    let ctx = Context::new();
    // lim(x→-∞) x³/(x²+1) = -∞ (odd degree diff flips at -∞)
    let x = ctx.symbol("x");
    let numer = x.powi(3);
    let denom = x.powi(2) + 1;
    let expr = &numer / &denom;
    let neg_inf = -ctx.infinity();
    let result = expr.limit(&x, &neg_inf);
    assert!(!result.has_unevaluated(), "limit should succeed");
    assert_eq!(format!("{result}"), "-oo");
}

#[test]
fn limit_rational_coeffs_at_pos_inf() {
    let ctx = Context::new();
    // lim(x→+∞) (3x²+1)/(-2x+5) = -∞ (positive/negative leading coeffs)
    let x = ctx.symbol("x");
    let numer = &ctx.int(3) * &x.powi(2) + 1;
    let denom = &ctx.int(-2) * &x + 5;
    let expr = &numer / &denom;
    let result = expr.limit(&x, &ctx.infinity());
    assert!(!result.has_unevaluated(), "limit should succeed");
    assert_eq!(format!("{result}"), "-oo");
}
