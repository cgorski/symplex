//! Integration tests for Stages 7–8: expand, eval, evalf, and simplify
//! through the public `Ex` API.

use symplex::prelude::*;
use symplex::syms;

// ═══════════════════════════════════════════════════════════════════════════
// expand()
// ═══════════════════════════════════════════════════════════════════════════

// ── Basic distribution ──────────────────────────────────────────────────

#[test]
fn expand_symbol_times_sum() {
    let ctx = Context::new();
    syms!(ctx; x, y, z);
    let expr = &z * &(&x + &y);
    assert_eq!(format!("{}", expr.expand()), "x*z + y*z");
}

#[test]
fn expand_sum_times_sum() {
    let ctx = Context::new();
    syms!(ctx; a, b, c, d);
    let expr = &(&a + &b) * &(&c + &d);
    let expanded = expr.expand();
    let s = format!("{expanded}");
    assert!(s.contains("a*c"), "should contain a*c, got: {s}");
    assert!(s.contains("a*d"), "should contain a*d, got: {s}");
    assert!(s.contains("b*c"), "should contain b*c, got: {s}");
    assert!(s.contains("b*d"), "should contain b*d, got: {s}");
}

#[test]
fn expand_numeric_coefficient_already_distributed() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    // 2*(x + y) is already distributed by Number*Add rule in canon_mul.
    let expr = &ctx.int(2) * &(&x + &y);
    let s = format!("{expr}");
    assert!(s.contains('+'), "should already be distributed: {s}");
}

#[test]
fn expand_atom_is_noop() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expanded = x.expand();
    assert_eq!(x, expanded);
}

#[test]
fn expand_sum_is_noop() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let expr = &x + &y;
    let expanded = expr.expand();
    assert_eq!(expr, expanded);
}

// ── Power expansion ─────────────────────────────────────────────────────

#[test]
fn expand_x_plus_1_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x + 1).powi(2);
    assert_eq!(format!("{expr}"), "(x + 1)^2");
    assert_eq!(format!("{}", expr.expand()), "x^2 + 2*x + 1");
}

#[test]
fn expand_x_plus_1_cubed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x + 1).powi(3);
    assert_eq!(format!("{}", expr.expand()), "x^3 + 3*x^2 + 3*x + 1");
}

#[test]
fn expand_x_plus_y_squared() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let expr = (&x + &y).powi(2);
    let expanded = format!("{}", expr.expand());
    assert!(expanded.contains("x^2"), "should contain x^2: {expanded}");
    assert!(expanded.contains("y^2"), "should contain y^2: {expanded}");
    assert!(
        expanded.contains("2*x*y") || expanded.contains("2*y*x"),
        "should contain 2xy: {expanded}"
    );
}

#[test]
fn expand_power_zero_is_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x + 1).powi(0);
    // (x+1)^0 canonicalizes to 1 immediately.
    assert!(expr.is_one_structural());
}

#[test]
fn expand_power_one_is_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x + 1).powi(1);
    // (x+1)^1 canonicalizes to x+1.
    assert_eq!(format!("{expr}"), "x + 1");
    assert_eq!(format!("{}", expr.expand()), "x + 1");
}

#[test]
fn expand_negative_power_not_expanded() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x + 1).powi(-2);
    let expanded = expr.expand();
    // Negative powers of sums should NOT be expanded.
    assert_eq!(format!("{expanded}"), "(x + 1)^(-2)");
}

#[test]
fn expand_non_sum_base_not_expanded() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(5);
    let expanded = expr.expand();
    assert_eq!(expr, expanded, "x^5 should be unchanged by expand");
}

// ── Nested expansion ────────────────────────────────────────────────────

#[test]
fn expand_product_of_power() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    // y * (x + 1)^2 → y*x^2 + 2*x*y + y
    let expr = &y * &(&x + 1).powi(2);
    let expanded = expr.expand();
    let s = format!("{expanded}");
    assert!(!s.contains("^2)"), "power should be expanded: {s}");
    assert!(s.contains('y'), "should contain y: {s}");
}

// ── Idempotence ─────────────────────────────────────────────────────────

#[test]
fn expand_is_idempotent() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x + 1).powi(3);
    let first = expr.expand();
    let second = first.expand();
    assert_eq!(first, second, "expand should be idempotent");
}

// ── Correctness via substitution ────────────────────────────────────────

#[test]
fn expand_evaluates_same_as_original() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x + 1).powi(3);
    let expanded = expr.expand();
    // Evaluate both at x=4: (4+1)^3 = 125
    let at_4 = expr.subs(&x, &ctx.int(4));
    let expanded_at_4 = expanded.subs(&x, &ctx.int(4));
    assert_eq!(at_4, expanded_at_4);
    assert_eq!(format!("{at_4}"), "125");
}

#[test]
fn expand_x_plus_y_squared_evaluates_correctly() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let expr = (&x + &y).powi(2);
    let expanded = expr.expand();
    // Evaluate at x=2, y=3: (2+3)^2 = 25
    let val = expr.subs(&x, &ctx.int(2)).subs(&y, &ctx.int(3));
    let exp_val = expanded.subs(&x, &ctx.int(2)).subs(&y, &ctx.int(3));
    assert_eq!(val, exp_val);
    assert_eq!(format!("{val}"), "25");
}

// ═══════════════════════════════════════════════════════════════════════════
// eval()
// ═══════════════════════════════════════════════════════════════════════════

// ── Trig special values ─────────────────────────────────────────────────

#[test]
fn eval_sin_zero() {
    let ctx = Context::new();
    let expr = ctx.int(0).sin();
    assert!(expr.eval().is_zero_structural(), "sin(0) should eval to 0");
}

#[test]
fn eval_sin_pi() {
    let ctx = Context::new();
    let expr = ctx.pi().sin();
    assert!(expr.eval().is_zero_structural(), "sin(π) should eval to 0");
}

#[test]
fn eval_sin_pi_over_2() {
    let ctx = Context::new();
    let expr = (&ctx.rational(1, 2) * &ctx.pi()).sin();
    assert!(expr.eval().is_one_structural(), "sin(π/2) should eval to 1");
}

#[test]
fn eval_cos_zero() {
    let ctx = Context::new();
    let expr = ctx.int(0).cos();
    assert!(expr.eval().is_one_structural(), "cos(0) should eval to 1");
}

#[test]
fn eval_cos_pi() {
    let ctx = Context::new();
    let expr = ctx.pi().cos();
    assert_eq!(format!("{}", expr.eval()), "-1", "cos(π) should be -1");
}

#[test]
fn eval_cos_pi_over_2() {
    let ctx = Context::new();
    let expr = (&ctx.rational(1, 2) * &ctx.pi()).cos();
    assert!(
        expr.eval().is_zero_structural(),
        "cos(π/2) should eval to 0"
    );
}

#[test]
fn eval_tan_zero() {
    let ctx = Context::new();
    let expr = ctx.int(0).tan();
    assert!(expr.eval().is_zero_structural(), "tan(0) should eval to 0");
}

// ── Exp / Ln special values ─────────────────────────────────────────────

#[test]
fn eval_exp_zero() {
    let ctx = Context::new();
    let expr = ctx.int(0).exp();
    assert!(expr.eval().is_one_structural(), "exp(0) should eval to 1");
}

#[test]
fn eval_exp_one_gives_e() {
    let ctx = Context::new();
    let expr = ctx.int(1).exp();
    let result = expr.eval();
    assert_eq!(format!("{result}"), "E", "exp(1) should eval to E");
}

#[test]
fn eval_ln_one() {
    let ctx = Context::new();
    let expr = ctx.int(1).ln();
    assert!(expr.eval().is_zero_structural(), "ln(1) should eval to 0");
}

#[test]
fn eval_ln_e() {
    let ctx = Context::new();
    let expr = ctx.e().ln();
    assert!(expr.eval().is_one_structural(), "ln(E) should eval to 1");
}

// ── Sqrt special values ─────────────────────────────────────────────────

#[test]
fn eval_sqrt_zero() {
    let ctx = Context::new();
    let expr = ctx.int(0).sqrt();
    assert!(expr.eval().is_zero_structural(), "sqrt(0) should eval to 0");
}

#[test]
fn eval_sqrt_one() {
    let ctx = Context::new();
    let expr = ctx.int(1).sqrt();
    assert!(expr.eval().is_one_structural(), "sqrt(1) should eval to 1");
}

#[test]
fn eval_sqrt_perfect_square() {
    let ctx = Context::new();
    let expr = ctx.int(9).sqrt();
    assert_eq!(format!("{}", expr.eval()), "3", "sqrt(9) should eval to 3");
}

#[test]
fn eval_sqrt_non_perfect_stays() {
    let ctx = Context::new();
    let expr = ctx.int(2).sqrt();
    let evaled = expr.eval();
    assert_eq!(
        format!("{evaled}"),
        "sqrt(2)",
        "sqrt(2) should stay unevaluated"
    );
}

// ── Abs special values ──────────────────────────────────────────────────

#[test]
fn eval_abs_positive() {
    let ctx = Context::new();
    let expr = ctx.int(5).abs();
    assert_eq!(format!("{}", expr.eval()), "5");
}

#[test]
fn eval_abs_negative() {
    let ctx = Context::new();
    let expr = ctx.int(-3).abs();
    assert_eq!(format!("{}", expr.eval()), "3");
}

#[test]
fn eval_abs_zero() {
    let ctx = Context::new();
    let expr = ctx.int(0).abs();
    assert!(expr.eval().is_zero_structural());
}

#[test]
fn eval_abs_symbolic_stays() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.abs();
    let evaled = expr.eval();
    assert_eq!(
        format!("{evaled}"),
        "abs(x)",
        "abs(x) should stay unevaluated"
    );
}

// ── No auto-evaluation in constructors ──────────────────────────────────

#[test]
fn constructors_do_not_eval_sin_zero() {
    let ctx = Context::new();
    let expr = ctx.int(0).sin();
    assert_eq!(
        format!("{expr}"),
        "sin(0)",
        "sin(0) should NOT auto-evaluate"
    );
}

#[test]
fn constructors_do_not_eval_cos_pi() {
    let ctx = Context::new();
    let expr = ctx.pi().cos();
    assert_eq!(
        format!("{expr}"),
        "cos(pi)",
        "cos(pi) should NOT auto-evaluate"
    );
}

#[test]
fn eval_then_display_cos_pi() {
    let ctx = Context::new();
    let expr = ctx.pi().cos();
    let evaled = expr.eval();
    assert_eq!(
        format!("{evaled}"),
        "-1",
        "cos(pi).eval() should display as -1"
    );
}

// ── Eval in compound expressions ────────────────────────────────────────

#[test]
fn eval_nested_sin_zero_in_sum() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x + sin(0) → x + 0 → x
    let expr = &x + &ctx.int(0).sin();
    let evaled = expr.eval();
    assert_eq!(evaled, x, "x + sin(0) should eval to x");
}

#[test]
fn eval_exp_zero_in_product() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x * exp(0) → x * 1 → x
    let expr = &x * &ctx.int(0).exp();
    let evaled = expr.eval();
    assert_eq!(evaled, x, "x * exp(0) should eval to x");
}

#[test]
fn eval_multiple_special_values() {
    let ctx = Context::new();
    // sin(0) + cos(0) + ln(1) = 0 + 1 + 0 = 1
    let expr = &ctx.int(0).sin() + &ctx.int(0).cos() + &ctx.int(1).ln();
    let evaled = expr.eval();
    assert!(evaled.is_one_structural(), "0 + 1 + 0 should be 1");
}

// ── Eval idempotence ────────────────────────────────────────────────────

#[test]
fn eval_is_idempotent() {
    let ctx = Context::new();
    let expr = ctx.pi().sin();
    let first = expr.eval();
    let second = first.eval();
    assert_eq!(first, second, "eval should be idempotent");
}

// ═══════════════════════════════════════════════════════════════════════════
// simplify()
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_pythagorean_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    assert_eq!(format!("{}", expr.simplify()), "1");
}

#[test]
fn simplify_pythagorean_different_symbol() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let expr = &t.sin().powi(2) + &t.cos().powi(2);
    assert_eq!(format!("{}", expr.simplify()), "1");
}

#[test]
fn simplify_no_match_returns_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin();
    let simplified = expr.simplify();
    assert_eq!(expr, simplified, "sin(x) alone should not simplify");
}

#[test]
fn simplify_pythagorean() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    let result = expr.simplify();
    assert_eq!(format!("{result}"), "1");
}

// ═══════════════════════════════════════════════════════════════════════════
// evalf()
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn evalf_integer() {
    let ctx = Context::new();
    let five = ctx.int(5);
    let result = five.eval_decimal(10).unwrap();
    assert_eq!(result, "5");
}

#[test]
fn evalf_rational() {
    let ctx = Context::new();
    let half = ctx.rational(1, 2);
    let result = half.eval_decimal(10).unwrap();
    assert_eq!(result, "0.5");
}

#[test]
fn evalf_pi_15_digits() {
    let ctx = Context::new();
    let result = ctx.pi().eval_decimal(15).unwrap();
    assert!(
        result.starts_with("3.14159265358979"),
        "pi to 15 digits: {result}"
    );
}

#[test]
fn evalf_pi_50_digits() {
    let ctx = Context::new();
    let result = ctx.pi().eval_decimal(50).unwrap();
    assert!(
        result.starts_with("3.1415926535897932384626433832795"),
        "pi to 50 digits: {result}"
    );
}

#[test]
fn evalf_e_15_digits() {
    let ctx = Context::new();
    let result = ctx.e().eval_decimal(15).unwrap();
    // e = 2.718281828459045…; the 15th digit rounds up.
    assert!(
        result.starts_with("2.71828182845905"),
        "e to 15 digits: {result}"
    );
}

#[test]
fn evalf_sqrt_2() {
    let ctx = Context::new();
    let expr = ctx.int(2).sqrt();
    let result = expr.eval_decimal(20).unwrap();
    assert!(
        result.starts_with("1.4142135623730950488"),
        "sqrt(2): {result}"
    );
}

#[test]
fn evalf_sin_zero_is_zero() {
    let ctx = Context::new();
    let expr = ctx.int(0).sin();
    let result = expr.eval_decimal(10).unwrap();
    assert_eq!(result, "0");
}

#[test]
fn evalf_cos_zero_is_one() {
    let ctx = Context::new();
    let expr = ctx.int(0).cos();
    let result = expr.eval_decimal(10).unwrap();
    assert_eq!(result, "1");
}

#[test]
fn evalf_exp_one_is_e() {
    let ctx = Context::new();
    let expr = ctx.int(1).exp();
    let result = expr.eval_decimal(15).unwrap();
    assert!(result.starts_with("2.71828182845905"), "exp(1): {result}");
}

#[test]
fn evalf_ln_one_is_zero() {
    let ctx = Context::new();
    let expr = ctx.int(1).ln();
    let result = expr.eval_decimal(10).unwrap();
    assert_eq!(result, "0");
}

#[test]
fn evalf_two_pow_ten() {
    let ctx = Context::new();
    let expr = ctx.int(2).powi(10);
    let result = expr.eval_decimal(10).unwrap();
    assert!(result.starts_with("1024"), "2^10: {result}");
}

#[test]
fn evalf_negative_pi() {
    let ctx = Context::new();
    let expr = -&ctx.pi();
    let result = expr.eval_decimal(10).unwrap();
    assert!(result.starts_with("-3.14159265"), "-pi: {result}");
}

#[test]
fn evalf_free_symbol_is_error() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.eval_decimal(10);
    assert!(result.is_err(), "evalf of free symbol should error");
}

#[test]
fn evalf_infinity_is_error() {
    let ctx = Context::new();
    let result = ctx.infinity().eval_decimal(10);
    assert!(result.is_err(), "evalf of infinity should error");
}

#[test]
fn evalf_nan_is_error() {
    let ctx = Context::new();
    let result = ctx.nan().eval_decimal(10);
    assert!(result.is_err(), "evalf of NaN should error");
}

#[test]
fn evalf_after_subs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x^2 + 1 at x=3 → 10
    let expr = &x.powi(2) + 1;
    let at_3 = expr.subs(&x, &ctx.int(3));
    let result = at_3.eval_decimal(10).unwrap();
    assert!(result.starts_with("10"), "3^2 + 1 = 10, got: {result}");
}

#[test]
fn evalf_sin_pi_near_zero() {
    let ctx = Context::new();
    let expr = ctx.pi().sin();
    let result = expr.eval_decimal(20).unwrap();
    let val: f64 = result.parse().unwrap_or(999.0);
    assert!(val.abs() < 1e-15, "sin(pi) should be ~0, got: {result}");
}

#[test]
fn evalf_pythagorean_identity_near_one() {
    let ctx = Context::new();
    let one = ctx.int(1);
    // sin^2(1) + cos^2(1) should be ~1
    let expr = &one.sin().powi(2) + &one.cos().powi(2);
    let result = expr.eval_decimal(15).unwrap();
    let val: f64 = result.parse().unwrap_or(0.0);
    assert!(
        (val - 1.0).abs() < 1e-12,
        "sin^2(1) + cos^2(1) should be ~1, got: {result}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Combined workflows
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn workflow_diff_subs_evalf() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // d/dx(x^3) at x=2 → 3*4 = 12
    let deriv = x.powi(3).diff(&x);
    let at_2 = deriv.subs(&x, &ctx.int(2));
    let result = at_2.eval_decimal(10).unwrap();
    assert!(result.starts_with("12"), "got: {result}");
}

#[test]
fn workflow_expand_diff_subs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // expand((x+1)^2) = x^2 + 2x + 1
    // d/dx = 2x + 2
    // at x=3 → 8
    let expanded = (&x + 1).powi(2).expand();
    let deriv = expanded.diff(&x);
    let at_3 = deriv.subs(&x, &ctx.int(3));
    assert_eq!(format!("{at_3}"), "8");
}

#[test]
fn workflow_eval_then_evalf() {
    let ctx = Context::new();
    // cos(pi) → eval → -1 → evalf → "-1"
    let expr = ctx.pi().cos();
    let evaled = expr.eval();
    assert_eq!(format!("{evaled}"), "-1");
    let result = evaled.eval_decimal(10).unwrap();
    assert_eq!(result, "-1");
}

#[test]
fn workflow_simplify_then_eval() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin^2(x) + cos^2(x) → simplify → 1 → eval → 1
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    let simplified = expr.simplify();
    assert!(simplified.is_one_structural());
    let evaled = simplified.eval();
    assert!(evaled.is_one_structural());
}

// ═══════════════════════════════════════════════════════════════════════════
// Assumption interaction
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn assumptions_through_expand() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]);
    let expr = (&x + 1).powi(2).expand();
    // x^2 + 2*x + 1 — all terms positive when x > 0.
    assert_eq!(
        ctx.query(&expr, Props::POSITIVE),
        Some(true),
        "expanded (x+1)^2 should be positive for positive x"
    );
}

#[test]
fn assumptions_survive_diff() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive, Assumption::Real]);
    let deriv = x.powi(2).diff(&x);
    // 2*x — should be positive since x > 0.
    assert_eq!(
        ctx.query(&deriv, Props::POSITIVE),
        Some(true),
        "2*x should be positive when x is positive"
    );
}

#[test]
fn assumptions_survive_subs() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let expr = &x + &y;
    // Substitute x → 5 (known positive integer).
    let result = expr.subs(&x, &ctx.int(5));
    // 5 + y — is it integer? Only if y is integer. We don't know, so None.
    assert_eq!(
        ctx.query(&result, Props::INTEGER),
        None,
        "5 + y with unknown y should have unknown integer status"
    );
    // But the 5 part is positive.
    let five = ctx.int(5);
    assert_eq!(ctx.query(&five, Props::POSITIVE), Some(true));
}
