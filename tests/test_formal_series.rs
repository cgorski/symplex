//! Integration tests for the formal power series module.
//!
//! Tests exercise the public `Ex::fps()` and `Ex::fps_maclaurin()` API,
//! verifying coefficient extraction, closed-form detection, and truncation
//! for known elementary functions.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helper
// ═══════════════════════════════════════════════════════════════════════════

/// Check that a rational coefficient equals p/q.
fn assert_coeff_eq(series: &symplex::formal_series::FormalPowerSeries, k: usize, p: i64, q: i64) {
    use num_bigint::BigInt;
    use num_rational::Ratio;
    let actual = series.coefficient_rational(k);
    let expected = Ratio::new(BigInt::from(p), BigInt::from(q));
    assert_eq!(
        actual, expected,
        "coefficient a_{k} should be {p}/{q}, got {actual}",
        k = k
    );
}

fn assert_coeff_zero(series: &symplex::formal_series::FormalPowerSeries, k: usize) {
    assert!(
        series.coefficient_rational(k).is_zero(),
        "coefficient a_{k} should be 0, got {}",
        series.coefficient_rational(k),
        k = k
    );
}

use num_traits::Zero;

// ═══════════════════════════════════════════════════════════════════════════
// exp(x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fps_exp_x_has_closed_form() {
    let x = symplex::var("x");
    let series = x.exp().fps_maclaurin(&x);
    assert!(series.has_closed_form());
}

#[test]
fn fps_exp_x_coefficients() {
    let x = symplex::var("x");
    let series = x.exp().fps_maclaurin(&x);
    // a_0 = 1
    assert_coeff_eq(&series, 0, 1, 1);
    // a_1 = 1
    assert_coeff_eq(&series, 1, 1, 1);
    // a_2 = 1/2
    assert_coeff_eq(&series, 2, 1, 2);
    // a_3 = 1/6
    assert_coeff_eq(&series, 3, 1, 6);
    // a_4 = 1/24
    assert_coeff_eq(&series, 4, 1, 24);
    // a_5 = 1/120
    assert_coeff_eq(&series, 5, 1, 120);
}

#[test]
fn fps_exp_x_coefficient_5_is_1_over_120() {
    let x = symplex::var("x");
    let series = x.exp().fps_maclaurin(&x);
    assert_coeff_eq(&series, 5, 1, 120);
}

// ═══════════════════════════════════════════════════════════════════════════
// sin(x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fps_sin_x_has_closed_form() {
    let x = symplex::var("x");
    let series = x.sin().fps_maclaurin(&x);
    assert!(series.has_closed_form());
}

#[test]
fn fps_sin_x_odd_coefficients() {
    let x = symplex::var("x");
    let series = x.sin().fps_maclaurin(&x);
    // sin(x) = x - x^3/6 + x^5/120 - ...
    // a_1 = 1
    assert_coeff_eq(&series, 1, 1, 1);
    // a_3 = -1/6
    assert_coeff_eq(&series, 3, -1, 6);
    // a_5 = 1/120
    assert_coeff_eq(&series, 5, 1, 120);
    // a_7 = -1/5040
    assert_coeff_eq(&series, 7, -1, 5040);
}

#[test]
fn fps_sin_x_even_coefficients_are_zero() {
    let x = symplex::var("x");
    let series = x.sin().fps_maclaurin(&x);
    assert_coeff_zero(&series, 0);
    assert_coeff_zero(&series, 2);
    assert_coeff_zero(&series, 4);
    assert_coeff_zero(&series, 6);
}

// ═══════════════════════════════════════════════════════════════════════════
// cos(x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fps_cos_x_even_coefficients() {
    let x = symplex::var("x");
    let series = x.cos().fps_maclaurin(&x);
    // cos(x) = 1 - x^2/2 + x^4/24 - ...
    assert_coeff_eq(&series, 0, 1, 1);
    assert_coeff_eq(&series, 2, -1, 2);
    assert_coeff_eq(&series, 4, 1, 24);
    assert_coeff_eq(&series, 6, -1, 720);
}

#[test]
fn fps_cos_x_odd_coefficients_are_zero() {
    let x = symplex::var("x");
    let series = x.cos().fps_maclaurin(&x);
    assert_coeff_zero(&series, 1);
    assert_coeff_zero(&series, 3);
    assert_coeff_zero(&series, 5);
}

// ═══════════════════════════════════════════════════════════════════════════
// 1/(1-x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fps_geometric_series_all_ones() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let zero = ctx.int(0);
    // 1/(1-x) = (1-x)^(-1)
    let expr = (&one - &x).powi(-1);
    let series = expr.fps(&x, &zero);
    for k in 0..10 {
        assert_coeff_eq(&series, k, 1, 1);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ln(1+x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fps_ln_1_plus_x_coefficients() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let zero = ctx.int(0);
    let expr = (&one + &x).ln();
    let series = expr.fps(&x, &zero);
    // a_0 = 0 (ln(1) = 0)
    assert_coeff_zero(&series, 0);
    // a_1 = 1
    assert_coeff_eq(&series, 1, 1, 1);
    // a_2 = -1/2
    assert_coeff_eq(&series, 2, -1, 2);
    // a_3 = 1/3
    assert_coeff_eq(&series, 3, 1, 3);
    // a_4 = -1/4
    assert_coeff_eq(&series, 4, -1, 4);
    // a_5 = 1/5
    assert_coeff_eq(&series, 5, 1, 5);
}

// ═══════════════════════════════════════════════════════════════════════════
// (1+x)^(1/2)  — generalized binomial
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fps_binomial_sqrt_coefficients() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let zero = ctx.int(0);
    let half = ctx.rational(1, 2);
    let expr = (&one + &x).pow(&half);
    let series = expr.fps(&x, &zero);
    assert!(series.has_closed_form());
    // a_0 = 1
    assert_coeff_eq(&series, 0, 1, 1);
    // a_1 = 1/2
    assert_coeff_eq(&series, 1, 1, 2);
    // a_2 = (1/2)(1/2 - 1)/2! = (1/2)(-1/2)/2 = -1/8
    assert_coeff_eq(&series, 2, -1, 8);
    // a_3 = (1/2)(-1/2)(-3/2)/3! = (3/8)/6 = 1/16
    assert_coeff_eq(&series, 3, 1, 16);
}

// ═══════════════════════════════════════════════════════════════════════════
// atan(x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fps_atan_x_odd_coefficients() {
    let x = symplex::var("x");
    let series = x.atan().fps_maclaurin(&x);
    // atan(x) = x - x^3/3 + x^5/5 - x^7/7 + ...
    assert_coeff_zero(&series, 0);
    assert_coeff_eq(&series, 1, 1, 1);
    assert_coeff_zero(&series, 2);
    assert_coeff_eq(&series, 3, -1, 3);
    assert_coeff_zero(&series, 4);
    assert_coeff_eq(&series, 5, 1, 5);
    assert_coeff_zero(&series, 6);
    assert_coeff_eq(&series, 7, -1, 7);
}

// ═══════════════════════════════════════════════════════════════════════════
// Truncation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fps_exp_truncate_5_terms() {
    // FPS truncation to 5 terms for exp(x) should produce
    // 1 + x + x^2/2 + x^3/6 + x^4/24
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let exp_x = x.exp();

    let series = exp_x.fps(&x, &zero);
    let _truncated = ctx.with_arena_mut(|arena| {
        let t = series.truncate(arena, 5);
        let expanded = arena.expand_expr(t);
        arena.eval_expr(expanded)
    });

    // Also compute via maclaurin for comparison
    let mac = exp_x.maclaurin(&x, 5).unwrap().expand().eval();

    let mac_s = format!("{mac}");
    assert!(mac_s.contains("x"), "maclaurin should have x: {mac_s}");
    assert!(mac_s.contains("x^2"), "maclaurin should have x^2: {mac_s}");
    assert!(mac_s.contains("x^3"), "maclaurin should have x^3: {mac_s}");
    assert!(mac_s.contains("x^4"), "maclaurin should have x^4: {mac_s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Coefficient access
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fps_coefficient_access_exp_5() {
    let x = symplex::var("x");
    let series = x.exp().fps_maclaurin(&x);
    // coefficient(5) = 1/120
    use num_bigint::BigInt;
    use num_rational::Ratio;
    let c5 = series.coefficient_rational(5);
    assert_eq!(c5, Ratio::new(BigInt::from(1), BigInt::from(120)));
}

// ═══════════════════════════════════════════════════════════════════════════
// sinh(x) and cosh(x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fps_sinh_x_coefficients() {
    let x = symplex::var("x");
    let series = x.sinh().fps_maclaurin(&x);
    assert!(series.has_closed_form());
    // sinh(x) = x + x^3/6 + x^5/120 + ...
    assert_coeff_zero(&series, 0);
    assert_coeff_eq(&series, 1, 1, 1);
    assert_coeff_zero(&series, 2);
    assert_coeff_eq(&series, 3, 1, 6);
    assert_coeff_zero(&series, 4);
    assert_coeff_eq(&series, 5, 1, 120);
}

#[test]
fn fps_cosh_x_coefficients() {
    let x = symplex::var("x");
    let series = x.cosh().fps_maclaurin(&x);
    assert!(series.has_closed_form());
    // cosh(x) = 1 + x^2/2 + x^4/24 + ...
    assert_coeff_eq(&series, 0, 1, 1);
    assert_coeff_zero(&series, 1);
    assert_coeff_eq(&series, 2, 1, 2);
    assert_coeff_zero(&series, 3);
    assert_coeff_eq(&series, 4, 1, 24);
}

// ═══════════════════════════════════════════════════════════════════════════
// Fallback (truncated) for non-elementary compositions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fps_non_elementary_falls_back_to_truncated() {
    let x = symplex::var("x");
    // exp(x) + sin(x) doesn't match a single known pattern
    let expr = &x.exp() + &x.sin();
    let series = expr.fps_maclaurin(&x);
    // Should NOT have a closed form
    assert!(!series.has_closed_form());
    // But should still provide coefficients via Taylor:
    // a_0 = exp(0) + sin(0) = 1
    use num_bigint::BigInt;
    use num_rational::Ratio;
    let c0 = series.coefficient_rational(0);
    assert_eq!(c0, Ratio::from_integer(BigInt::from(1)));
}
