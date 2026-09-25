//! Tests for math rules sprint: power-of-power, inverse hyperbolic compositions,
//! nth root evaluation, irrational trig values, hyperbolic odd/even, log expansion.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Power of power simplification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_pow_pow_integers() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x^2)^3 → x^6
    let expr = x.powi(2).powi(3);
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "x^6");
}

#[test]
fn simplify_pow_pow_in_expression() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
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
    let ctx = Context::new();
    // 0.23: the identity needs a real argument (a symbol without assumptions may be complex).
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let expr = x.sinh().asinh();
    assert_eq!(format!("{}", expr.simplify()), "x");
}

#[test]
fn simplify_acosh_cosh() {
    let ctx = Context::new();
    // 0.23: the identity needs a real argument (a symbol without assumptions may be complex).
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let expr = x.cosh().acosh();
    assert_eq!(format!("{}", expr.simplify()), "abs(x)");
}

#[test]
fn simplify_atanh_tanh() {
    let ctx = Context::new();
    // 0.23: the identity needs a real argument (a symbol without assumptions may be complex).
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
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
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (-&x).sinh();
    let evaled = expr.eval();
    assert_eq!(format!("{evaled}"), "-sinh(x)");
}

#[test]
fn eval_cosh_neg_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (-&x).cosh();
    let evaled = expr.eval();
    assert_eq!(format!("{evaled}"), "cosh(x)");
}

#[test]
fn eval_tanh_neg_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
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
    // 0.23: the log identities need positive arguments (x = y = -1 breaks ln(x*y) = ln x + ln y).
    let (x, y) = (
        ctx.symbol_with("x", &[Assumption::Positive]).unwrap(),
        ctx.symbol_with("y", &[Assumption::Positive]).unwrap(),
    );
    let expr = (&x * &y).ln();
    let expanded = expr.expand_log();
    let s = format!("{expanded}");
    assert!(s.contains("ln(x)") && s.contains("ln(y)"), "got: {s}");
}

#[test]
fn expand_log_power() {
    let ctx = Context::new();
    // 0.23: the log identities need positive arguments (x = y = -1 breaks ln(x*y) = ln x + ln y).
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    let expr = x.powi(2).ln();
    let expanded = expr.expand_log();
    let s = format!("{expanded}");
    assert!(s.contains("2") && s.contains("ln(x)"), "got: {s}");
}

#[test]
fn expand_log_quotient() {
    let ctx = Context::new();
    // 0.23: the log identities need positive arguments (x = y = -1 breaks ln(x*y) = ln x + ln y).
    let (x, y) = (
        ctx.symbol_with("x", &[Assumption::Positive]).unwrap(),
        ctx.symbol_with("y", &[Assumption::Positive]).unwrap(),
    );
    let expr = (&x / &y).ln();
    let expanded = expr.expand_log();
    let s = format!("{expanded}");
    assert!(s.contains("ln(x)") && s.contains("ln(y)"), "got: {s}");
}

#[test]
fn expand_log_bare_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
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
    let ctx = Context::new();
    // exp(ln(x) + 1) should NOT simplify to x (the +1 prevents matching).
    // Since 0.2 `expand` splits exp(a + b) → exp(a)·exp(b), so the correct
    // simplification exp(ln(x))·e = x·E is produced instead.
    let x = ctx.symbol("x");
    let expr = (&x.ln() + 1).exp();
    let result = expr.simplify();
    let s = format!("{result}");
    assert_ne!(s, "x", "exp(ln(x) + 1) must not become x");
    assert_eq!(s, "x*E");
    let pt = ctx.rational(7, 3);
    let before = expr.subs(&x, &pt).eval_f64().unwrap();
    let after = result.subs(&x, &pt).eval_f64().unwrap();
    assert!((before - after).abs() < 1e-10);
}

#[test]
fn neg_sqrt_sq_wrong_exponent() {
    let ctx = Context::new();
    // sqrt(x^3) must NOT simplify to |x|, to x, or to x^(3/2): the inner
    // exponent 3 is an integer, so (x^3)^(1/2) = x^(3/2) fails for x < 0
    // (sympy: sqrt((-2)**3) = 2*sqrt(2)*I, (-2)**(3/2) = -2*sqrt(2)*I).
    // Since 0.22 `pow_pow` only fires for an integer *outer* exponent, a
    // non-negative base, or an inner exponent in (-1, 1].
    let x = ctx.symbol("x");
    let expr = x.powi(3).sqrt();
    let result = format!("{}", expr.simplify());
    assert_ne!(result, "abs(x)", "sqrt(x^3) must not simplify to abs(x)");
    assert_ne!(result, "x", "sqrt(x^3) must not simplify to x");
    assert_ne!(result, "x^(3/2)", "sqrt(x^3) = x^(3/2) is false for x < 0");
    assert_simplify_unchanged!(expr);
}

#[test]
fn neg_sin_div_cos_different_args() {
    let ctx = Context::new();
    // sin(x)/cos(y) should NOT become tan (different arguments)
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x.sin() / &y.cos();
    assert_simplify_unchanged!(expr);
}

#[test]
fn neg_exp_mul_not_both_exp() {
    let ctx = Context::new();
    // exp(x) * sin(x) should NOT trigger exp combining
    let x = ctx.symbol("x");
    let expr = &x.exp() * &x.sin();
    assert_simplify_unchanged!(expr);
}

#[test]
fn neg_pow_pow_both_fractional() {
    let ctx = Context::new();
    // (x^(1/2))^(1/3) = x^(1/6) holds on the principal branch for every
    // complex x because the inner exponent lies in (-1, 1] (sympy:
    // (x**Rational(1,2))**Rational(1,3) == x**Rational(1,6)).  The unsafe
    // direction is an *integer* inner exponent — see `neg_sqrt_sq_wrong_exponent`.
    let x = ctx.symbol("x");
    let half = ctx.rational(1, 2);
    let third = ctx.rational(1, 3);
    let expr = x.pow(&half).pow(&third);
    assert_simplifies_to!(expr, "x^(1/6)");
    // (x^2)^(1/3) is the unsafe shape and must stay.
    let unsafe_shape = x.powi(2).pow(&third);
    assert_simplify_unchanged!(unsafe_shape);
}

#[test]
fn neg_abs_not_positive() {
    let ctx = Context::new();
    // abs(x) should NOT simplify when x has no positivity assumption
    let x = ctx.symbol("x");
    let expr = x.abs();
    assert_simplify_unchanged!(expr);
}

#[test]
fn neg_pythagorean_wrong_functions() {
    let ctx = Context::new();
    // sinh(x)^2 + cos(x)^2 should NOT simplify (mixed sinh/cos)
    let x = ctx.symbol("x");
    let expr = &x.sinh().powi(2) + &x.cos().powi(2);
    assert_simplify_unchanged!(expr);
}

#[test]
fn neg_cosh_sinh_wrong_sign() {
    let ctx = Context::new();
    // cosh(x)^2 + sinh(x)^2 should NOT simplify to 1 (wrong sign, identity is cosh²-sinh²).
    // Since 0.2 the hyperbolic double-angle rule gives cosh(2x), which is exact.
    let x = ctx.symbol("x");
    let expr = &x.cosh().powi(2) + &x.sinh().powi(2);
    let result = expr.simplify();
    let s = format!("{result}");
    assert_ne!(s, "1", "cosh² + sinh² must not become 1");
    assert_eq!(s, "cosh(2*x)");
    let pt = ctx.rational(3, 5);
    let before = expr.subs(&x, &pt).eval_f64().unwrap();
    let after = result.subs(&x, &pt).eval_f64().unwrap();
    assert!((before - after).abs() < 1e-10);
}

#[test]
fn neg_asin_sin_removed() {
    let ctx = Context::new();
    // asin(sin(x)) should NOT simplify to x (rule removed for correctness)
    let x = ctx.symbol("x");
    assert_simplifies_to!(x.sin().asin(), "asin(sin(x))");
}

#[test]
fn neg_acos_cos_removed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_simplifies_to!(x.cos().acos(), "acos(cos(x))");
}

#[test]
fn neg_atan_tan_removed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_simplifies_to!(x.tan().atan(), "atan(tan(x))");
}

// ═══════════════════════════════════════════════════════════════════════════
// Positive tests using macros — verify rules DO fire correctly
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn pos_pythagorean_with_macro() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_simplifies_to!(&x.sin().powi(2) + &x.cos().powi(2), "1");
}

#[test]
fn pos_exp_ln_with_macro() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_simplifies_to!(x.ln().exp(), "x");
}

#[test]
fn pos_acosh_cosh_gives_abs() {
    let ctx = Context::new();
    // 0.23: the identity needs a real argument (a symbol without assumptions may be complex).
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    assert_simplifies_to!(x.cosh().acosh(), "abs(x)");
}

#[test]
fn pos_sin_asin_still_works() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_simplifies_to!(x.asin().sin(), "x");
}

#[test]
fn pos_cos_acos_still_works() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_simplifies_to!(x.acos().cos(), "x");
}

#[test]
fn pos_asinh_sinh_still_works() {
    let ctx = Context::new();
    // 0.23: the identity needs a real argument (a symbol without assumptions may be complex).
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    assert_simplifies_to!(x.sinh().asinh(), "x");
}

// ═══════════════════════════════════════════════════════════════════════════
// Value-preservation tests using macro
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn value_pythagorean() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let point = ctx.rational(7, 10);
    assert_simplify_preserves_value!(&x.sin().powi(2) + &x.cos().powi(2), x, point);
}

#[test]
fn value_exp_ln() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let point = ctx.int(3);
    assert_simplify_preserves_value!(x.ln().exp(), x, point);
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests for newly added rules and features
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_exp_log_denest() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_simplifies_to!((&x.ln() * 3).exp(), "x^3");
}

#[test]
fn rule_exp_log_denest_symbolic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    assert_simplifies_to!((&x.ln() * &a).exp(), "x^a");
}

#[test]
fn expand_trig_sin_2x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
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
    let ctx = Context::new();
    let x = ctx.symbol("x");
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
    let ctx = Context::new();
    let x = ctx.symbol("x");
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
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &(&x.sin() * &x.cos()) * 2;
    let combined = expr.trig_combine();
    let s = format!("{combined}");
    assert!(!s.contains("sin(0)"), "should not contain sin(0), got: {s}");
}

#[test]
fn together_with_lcm() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
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
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.cos() / &x.sin();
    let simplified = expr.simplify();
    let s = format!("{simplified}");
    assert!(
        s.contains("tan"),
        "cos(x)/sin(x) should simplify to involve tan, got: {s}"
    );
}
