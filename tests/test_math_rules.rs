//! Tests for math rules sprint: power-of-power, inverse hyperbolic compositions,
//! nth root evaluation, irrational trig values, hyperbolic odd/even, log expansion.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Power of power simplification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_pow_pow_integers() {
    let x = symplex::default_context().symbol("x");
    // (x^2)^3 → x^6
    let expr = x.powi(2).powi(3);
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "x^6");
}

#[test]
fn simplify_pow_pow_in_expression() {
    let x = symplex::default_context().symbol("x");
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
    let x = symplex::default_context().symbol("x");
    let expr = x.sinh().asinh();
    assert_eq!(format!("{}", expr.simplify()), "x");
}

#[test]
fn simplify_acosh_cosh() {
    let x = symplex::default_context().symbol("x");
    let expr = x.cosh().acosh();
    assert_eq!(format!("{}", expr.simplify()), "abs(x)");
}

#[test]
fn simplify_atanh_tanh() {
    let x = symplex::default_context().symbol("x");
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
    let x = symplex::default_context().symbol("x");
    let expr = (-&x).sinh();
    let evaled = expr.eval();
    assert_eq!(format!("{evaled}"), "-sinh(x)");
}

#[test]
fn eval_cosh_neg_x() {
    let x = symplex::default_context().symbol("x");
    let expr = (-&x).cosh();
    let evaled = expr.eval();
    assert_eq!(format!("{evaled}"), "cosh(x)");
}

#[test]
fn eval_tanh_neg_x() {
    let x = symplex::default_context().symbol("x");
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
    let x = symplex::default_context().symbol("x");
    let expr = x.ln();
    let expanded = expr.expand_log();
    assert_eq!(format!("{expanded}"), "ln(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Test helper macros for ergonomic rule testing
// ═══════════════════════════════════════════════════════════════════════════

/// Assert that simplifying an expression produces the expected string output.
macro_rules! assert_simplifies_to {
    ($expr:expr, $expected:expr) => {
        let result = ($expr).simplify();
        assert_eq!(
            format!("{result}"),
            $expected,
            "simplify({}) should give '{}', got '{result}'",
            $expr,
            $expected
        );
    };
}

/// Assert that simplifying an expression leaves it unchanged.
macro_rules! assert_simplify_unchanged {
    ($expr:expr) => {
        let input_str = format!("{}", $expr);
        let result = ($expr).simplify();
        let result_str = format!("{result}");
        assert_eq!(
            input_str, result_str,
            "simplify({input_str}) should be unchanged, got '{result_str}'"
        );
    };
}

/// Assert that simplifying preserves the mathematical value at a given point.
macro_rules! assert_simplify_preserves_value {
    ($expr:expr, $var:expr, $point:expr) => {
        let result = ($expr).simplify();
        let val_in = ($expr).subs(&$var, &$point).eval_f64();
        let val_out = result.subs(&$var, &$point).eval_f64();
        if let (Ok(v1), Ok(v2)) = (val_in, val_out) {
            assert!(
                (v1 - v2).abs() < 1e-8,
                "simplify changed value at point: {v1} vs {v2}"
            );
        }
    };
}

// ═══════════════════════════════════════════════════════════════════════════
// Negative tests — verify rules DON'T fire when they shouldn't
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn neg_exp_ln_different_structure() {
    // exp(ln(x) + 1) should NOT simplify to x (the +1 prevents matching)
    let x = symplex::default_context().symbol("x");
    let expr = (&x.ln() + 1).exp();
    assert_simplify_unchanged!(expr);
}

#[test]
fn neg_sqrt_sq_wrong_exponent() {
    // sqrt(x^3) should NOT simplify to |x| (exponent is 3, not 2)
    // It correctly becomes x^(3/2) instead.
    let x = symplex::default_context().symbol("x");
    let expr = x.powi(3).sqrt();
    let result = format!("{}", expr.simplify());
    assert_ne!(result, "abs(x)", "sqrt(x^3) must not simplify to abs(x)");
    assert_ne!(result, "x", "sqrt(x^3) must not simplify to x");
    assert_eq!(result, "x^(3/2)");
}

#[test]
fn neg_sin_div_cos_different_args() {
    // sin(x)/cos(y) should NOT become tan (different arguments)
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let expr = &x.sin() / &y.cos();
    assert_simplify_unchanged!(expr);
}

#[test]
fn neg_exp_mul_not_both_exp() {
    // exp(x) * sin(x) should NOT trigger exp combining
    let x = symplex::default_context().symbol("x");
    let expr = &x.exp() * &x.sin();
    assert_simplify_unchanged!(expr);
}

#[test]
fn neg_pow_pow_both_fractional() {
    // (x^(1/2))^(1/3) should NOT become x^(1/6) (no integer exponent)
    let x = symplex::default_context().symbol("x");
    let ctx = symplex::default_context();
    let half = ctx.rational(1, 2);
    let third = ctx.rational(1, 3);
    let expr = x.pow(&half).pow(&third);
    assert_simplify_unchanged!(expr);
}

#[test]
fn neg_abs_not_positive() {
    // abs(x) should NOT simplify when x has no positivity assumption
    let x = symplex::default_context().symbol("x");
    let expr = x.abs();
    assert_simplify_unchanged!(expr);
}

#[test]
fn neg_pythagorean_wrong_functions() {
    // sinh(x)^2 + cos(x)^2 should NOT simplify (mixed sinh/cos)
    let x = symplex::default_context().symbol("x");
    let expr = &x.sinh().powi(2) + &x.cos().powi(2);
    assert_simplify_unchanged!(expr);
}

#[test]
fn neg_cosh_sinh_wrong_sign() {
    // cosh(x)^2 + sinh(x)^2 should NOT simplify to 1 (wrong sign, identity is cosh²-sinh²)
    let x = symplex::default_context().symbol("x");
    let expr = &x.cosh().powi(2) + &x.sinh().powi(2);
    assert_simplify_unchanged!(expr);
}

#[test]
fn neg_asin_sin_removed() {
    // asin(sin(x)) should NOT simplify to x (rule removed for correctness)
    let x = symplex::default_context().symbol("x");
    assert_simplifies_to!(x.sin().asin(), "asin(sin(x))");
}

#[test]
fn neg_acos_cos_removed() {
    let x = symplex::default_context().symbol("x");
    assert_simplifies_to!(x.cos().acos(), "acos(cos(x))");
}

#[test]
fn neg_atan_tan_removed() {
    let x = symplex::default_context().symbol("x");
    assert_simplifies_to!(x.tan().atan(), "atan(tan(x))");
}

// ═══════════════════════════════════════════════════════════════════════════
// Positive tests using macros — verify rules DO fire correctly
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn pos_pythagorean_with_macro() {
    let x = symplex::default_context().symbol("x");
    assert_simplifies_to!(&x.sin().powi(2) + &x.cos().powi(2), "1");
}

#[test]
fn pos_exp_ln_with_macro() {
    let x = symplex::default_context().symbol("x");
    assert_simplifies_to!(x.ln().exp(), "x");
}

#[test]
fn pos_acosh_cosh_gives_abs() {
    let x = symplex::default_context().symbol("x");
    assert_simplifies_to!(x.cosh().acosh(), "abs(x)");
}

#[test]
fn pos_sin_asin_still_works() {
    let x = symplex::default_context().symbol("x");
    assert_simplifies_to!(x.asin().sin(), "x");
}

#[test]
fn pos_cos_acos_still_works() {
    let x = symplex::default_context().symbol("x");
    assert_simplifies_to!(x.acos().cos(), "x");
}

#[test]
fn pos_asinh_sinh_still_works() {
    let x = symplex::default_context().symbol("x");
    assert_simplifies_to!(x.sinh().asinh(), "x");
}

// ═══════════════════════════════════════════════════════════════════════════
// Value-preservation tests using macro
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn value_pythagorean() {
    let x = symplex::default_context().symbol("x");
    let point = symplex::default_context().rational(7, 10);
    assert_simplify_preserves_value!(&x.sin().powi(2) + &x.cos().powi(2), x, point);
}

#[test]
fn value_exp_ln() {
    let x = symplex::default_context().symbol("x");
    let point = symplex::default_context().int(3);
    assert_simplify_preserves_value!(x.ln().exp(), x, point);
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests for newly added rules and features
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_exp_log_denest() {
    let x = symplex::default_context().symbol("x");
    assert_simplifies_to!((&x.ln() * 3).exp(), "x^3");
}

#[test]
fn rule_exp_log_denest_symbolic() {
    let x = symplex::default_context().symbol("x");
    let a = symplex::default_context().symbol("a");
    assert_simplifies_to!((&x.ln() * &a).exp(), "x^a");
}

#[test]
fn expand_trig_sin_2x() {
    let x = symplex::default_context().symbol("x");
    let expr = (&x * 2).sin();
    let expanded = expr.expand_trig();
    let s = format!("{expanded}");
    assert!(
        s.contains("sin") && s.contains("cos"),
        "sin(2x) should expand to involve both sin and cos, got: {s}"
    );
}

#[test]
fn expand_trig_cos_2x() {
    let x = symplex::default_context().symbol("x");
    let expr = (&x * 2).cos();
    let expanded = expr.expand_trig();
    let s = format!("{expanded}");
    assert!(
        s.contains("sin") || s.contains("cos"),
        "cos(2x) should expand, got: {s}"
    );
}

#[test]
fn expand_trig_sin_3x() {
    let x = symplex::default_context().symbol("x");
    let expr = (&x * 3).sin();
    let expanded = expr.expand_trig();
    let s = format!("{expanded}");
    // Should be fully expanded — no remaining sin(2x) or sin(3x)
    assert!(
        !s.contains("3*x"),
        "sin(3x) should be fully expanded, got: {s}"
    );
}

#[test]
fn trig_combine_double_angle() {
    let x = symplex::default_context().symbol("x");
    let expr = &(&x.sin() * &x.cos()) * 2;
    let combined = expr.trig_combine();
    let s = format!("{combined}");
    assert!(!s.contains("sin(0)"), "should not contain sin(0), got: {s}");
}

#[test]
fn together_with_lcm() {
    let x = symplex::default_context().symbol("x");
    let a = symplex::default_context().symbol("a");
    let b = symplex::default_context().symbol("b");
    // a/(x-1) + b/(x-1)^2 should have denom (x-1)^2, not (x-1)^3
    let x_minus_1 = &x - 1;
    let frac1 = &a / &x_minus_1;
    let frac2 = &b / &x_minus_1.powi(2);
    let sum = &frac1 + &frac2;
    let result = sum.together();
    let s = format!("{result}");
    // Should NOT contain ^3 (which would indicate product-based denom)
    assert!(!s.contains("^3"), "together should use LCM, got: {s}");
}

#[test]
fn cos_div_sin_rule() {
    let x = symplex::default_context().symbol("x");
    let expr = &x.cos() / &x.sin();
    let simplified = expr.simplify();
    let s = format!("{simplified}");
    assert!(
        s.contains("tan"),
        "cos(x)/sin(x) should simplify to involve tan, got: {s}"
    );
}
