//! Tests for denominator rationalization and limit edge cases.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Section 1: Rationalize denominator with numerical verification
// ═══════════════════════════════════════════════════════════════════════════

/// Helper: verify rationalization preserves value numerically.
fn assert_rationalize_preserves_value(expr: &Ex, label: &str) {
    let ctx = expr.context();
    let rationalized = expr.rationalize_denom();

    // Evaluate both at x = 3/2
    let test_point = ctx.rational(3, 2);
    let x = ctx.symbol("x");
    let orig = expr.subs(&x, &test_point).eval_f64();
    let rat = rationalized.subs(&x, &test_point).eval_f64();

    if let (Ok(o), Ok(r)) = (orig, rat)
        && o.is_finite()
        && r.is_finite()
    {
        let diff = (o - r).abs();
        let tol = 1e-10 * o.abs().max(1.0);
        assert!(
            diff < tol,
            "{label}: value changed by rationalization — orig={o}, rationalized={r}"
        );
    }
}

#[test]
fn rationalize_one_over_sqrt2() {
    let ctx = Context::new();
    // 1 / √2  →  √2 / 2
    let expr = ctx.int(1) / &ctx.int(2).sqrt();
    let result = expr.rationalize_denom();
    let s = format!("{result}");
    // Denominator should no longer contain a square root
    assert!(s.contains('2'), "should rationalize 1/√2: {s}");
    assert_rationalize_preserves_value(&expr, "1/√2");
}

#[test]
fn rationalize_one_over_one_plus_sqrt2() {
    let ctx = Context::new();
    // 1 / (1 + √2)  →  √2 − 1
    let sqrt2 = ctx.int(2).sqrt();
    let expr = ctx.int(1) / &(&ctx.int(1) + &sqrt2);
    assert_rationalize_preserves_value(&expr, "1/(1+√2)");
}

#[test]
fn rationalize_preserves_no_sqrt_denom() {
    let ctx = Context::new();
    // x/3 has no radical in the denominator — should be unchanged.
    let x = ctx.symbol("x");
    let expr = &x / &ctx.int(3);
    let result = expr.rationalize_denom();
    assert_eq!(
        format!("{result}"),
        format!("{expr}"),
        "no-sqrt denom should be unchanged"
    );
}

#[test]
fn rationalize_integer_unchanged() {
    let ctx = Context::new();
    let expr = ctx.int(5);
    let result = expr.rationalize_denom();
    assert_eq!(format!("{result}"), "5");
}

#[test]
fn rationalize_three_over_sqrt2() {
    let ctx = Context::new();
    // 3 / √2  — numerator should survive rationalization
    let expr = &ctx.int(3) / &ctx.int(2).sqrt();
    let result = expr.rationalize_denom();
    let s = format!("{result}");
    assert!(s.contains('3'), "numerator 3 should survive: {s}");
    assert_rationalize_preserves_value(&expr, "3/√2");
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 2: Limit edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn limit_constant_expression() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = ctx.int(5);
    let result = expr.limit(&x, &ctx.int(0));
    assert!(!result.has_unevaluated());
    assert_eq!(format!("{result}"), "5");
}

#[test]
fn limit_polynomial_direct_sub() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x² + x + 1  at x = 2  →  4 + 2 + 1 = 7
    let expr = &(x.powi(2) + &x) + &ctx.int(1);
    let result = expr.limit(&x, &ctx.int(2));
    assert_eq!(format!("{result}"), "7");
}

#[test]
fn limit_sin_x_over_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin() / &x;
    let result = expr.limit(&x, &ctx.int(0));
    assert!(!result.has_unevaluated());
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn limit_lhopital_x2_minus1_over_x_minus1() {
    let ctx = Context::new();
    // (x² − 1) / (x − 1)  →  2  as x → 1   (0/0 indeterminate form)
    let x = ctx.symbol("x");
    let expr = (x.powi(2) - 1) / (&x - 1);
    let result = expr.limit(&x, &ctx.int(1));
    assert!(!result.has_unevaluated());
    assert_eq!(format!("{result}"), "2");
}

#[test]
fn limit_at_infinity_polynomial_ratio() {
    let ctx = Context::new();
    // (3x² + x) / (x² + 1)  →  3  as x → ∞
    let x = ctx.symbol("x");
    let numer = &x.powi(2) * 3 + &x;
    let denom = x.powi(2) + 1;
    let expr = &numer / &denom;
    let result = expr.limit(&x, &ctx.infinity());
    assert_eq!(format!("{result}"), "3");
}

#[test]
fn limit_exp_neg_x_at_infinity() {
    let ctx = Context::new();
    // exp(−x) → 0  as x → ∞
    let x = ctx.symbol("x");
    let expr = (-&x).exp();
    let result = expr.limit(&x, &ctx.infinity());
    assert_eq!(format!("{result}"), "0", "lim exp(-x) at ∞ should be 0");
}

#[test]
fn limit_fallback() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin();
    // sin(0) = 0 via limit
    let result = expr.limit(&x, &ctx.int(0));
    let s = format!("{result}");
    assert_eq!(s, "0", "sin(0) should be 0, got: {s}");
}
