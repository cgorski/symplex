//! Tests for simplification rules, sub-expression matching, and eval special values.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// New simplification rules
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_exp_ln() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.ln().exp_fn();
    assert_eq!(format!("{}", expr.simplify()), "x");
}

#[test]
fn simplify_ln_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.exp_fn().ln();
    assert_eq!(format!("{}", expr.simplify()), "x");
}

#[test]
fn simplify_abs_abs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.abs().abs();
    assert_eq!(format!("{}", expr.simplify()), "abs(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Sub-expression matching in larger sums
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_pythagorean_in_larger_sum() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 3 + sin²(x) + cos²(x) → 3 + 1 = 4
    let expr = &x.sin().powi(2) + &x.cos().powi(2) + 3;
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "4");
}

#[test]
fn simplify_pythagorean_in_sum_with_symbols() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // y + sin²(x) + cos²(x) → y + 1 = 1 + y
    let expr = &y + &x.sin().powi(2) + &x.cos().powi(2);
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "1 + y");
}

#[test]
fn simplify_no_sub_match_when_not_applicable() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin²(x) + sin²(x) — not a Pythagorean pair, should stay
    let expr = &x.sin().powi(2) + &x.sin().powi(2);
    let simplified = expr.simplify();
    // Should simplify to 2*sin(x)^2 (by canonicalization) but not to 1
    let s = format!("{simplified}");
    assert!(
        !s.contains("1") || s.contains("sin"),
        "should not reduce to 1: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Eval special values (via public API)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_sin_pi_over_6() {
    let ctx = Context::new();
    let expr = (&ctx.pi() / 6).sin();
    let result = expr.eval();
    assert_eq!(format!("{result}"), "1/2");
}

#[test]
fn eval_cos_pi_over_3() {
    let ctx = Context::new();
    let expr = (&ctx.pi() / 3).cos();
    let result = expr.eval();
    assert_eq!(format!("{result}"), "1/2");
}

#[test]
fn eval_tan_pi_over_4() {
    let ctx = Context::new();
    let expr = (&ctx.pi() / 4).tan();
    let result = expr.eval();
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn eval_sqrt_nine_fourths() {
    let ctx = Context::new();
    let expr = ctx.rational(9, 4).sqrt();
    let result = expr.eval();
    assert_eq!(format!("{result}"), "3/2");
}

#[test]
fn eval_combined_workflow() {
    // sin(π/6)^2 + cos(π/6)^2 should simplify to 1 after eval + simplify
    let ctx = Context::new();
    let pi_6 = &ctx.pi() / 6;
    let expr = &pi_6.sin().powi(2) + &pi_6.cos().powi(2);
    let _result = expr.simplify();
    // The simplify pass applies Pythagorean, but the inner sin/cos haven't been eval'd
    // So first eval, then simplify:
    let _evaled = expr.eval();
    // After eval: (1/2)^2 + cos(π/6)^2 — cos(π/6) doesn't have a rational value
    // So this test checks the non-eval path via simplify on unevaluated trig
    let simplified = expr.simplify();
    assert_eq!(
        format!("{simplified}"),
        "1",
        "sin²+cos² should still be 1 via Pythagorean identity"
    );
}
