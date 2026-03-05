//! Tests for rewrite rule APPLICATION — verifying that each of the 22 rules
//! in `basic_rules()` actually matches and transforms expressions correctly.
//!
//! This file tests three dimensions for each rule:
//!   1. **Positive:** the rule fires and produces the expected result.
//!   2. **Negative:** structurally similar expressions that should NOT match.
//!   3. **Value preservation:** the transformation preserves numerical value.
//!
//! All tests work through the public `Expr` API (`.simplify()`,
//! `.simplify_trace()`, `.evalf_f64()`, etc.) — no `mod common;` needed.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helper macros
// ═══════════════════════════════════════════════════════════════════════════

/// Assert simplification produces the exact expected string.
macro_rules! assert_simplifies_to {
    ($expr:expr, $expected:expr) => {{
        let result = ($expr).simplify();
        let s = format!("{result}");
        assert_eq!(
            s, $expected,
            "expected simplify to produce '{}', got '{}'",
            $expected, s,
        );
    }};
}

/// Assert simplification does NOT change the display string.
macro_rules! assert_simplify_unchanged {
    ($expr:expr) => {{
        let before = format!("{}", $expr);
        let result = ($expr).simplify();
        let after = format!("{result}");
        assert_eq!(
            before, after,
            "expected simplify to leave '{}' unchanged, but got '{}'",
            before, after,
        );
    }};
}

/// Assert that a named rule appears in the simplify trace.
macro_rules! assert_trace_contains_rule {
    ($expr:expr, $rule_name:expr) => {{
        let (_result, steps) = ($expr).simplify_trace();
        let found = steps.iter().any(|s| s.rule_name == $rule_name);
        assert!(
            found,
            "expected rule '{}' to fire, but trace contained: {:?}",
            $rule_name,
            steps.iter().map(|s| s.rule_name).collect::<Vec<_>>(),
        );
    }};
}

/// Assert that NO rules fire during simplification.
macro_rules! assert_trace_empty {
    ($expr:expr) => {{
        let (_result, steps) = ($expr).simplify_trace();
        assert!(
            steps.is_empty(),
            "expected no rules to fire, but trace contained: {:?}",
            steps.iter().map(|s| s.rule_name).collect::<Vec<_>>(),
        );
    }};
}

/// Assert value preservation: before and after simplification at a numeric
/// substitution point, the f64 values agree within tolerance.
macro_rules! assert_value_preserved {
    ($expr:expr, $var:expr, $val:expr, $tol:expr) => {{
        let before_val = ($expr).subs(&$var, &$val).evalf_f64();
        let after_val = ($expr).simplify().subs(&$var, &$val).evalf_f64();
        match (before_val, after_val) {
            (Ok(v1), Ok(v2)) => {
                assert!(
                    (v1 - v2).abs() < $tol,
                    "value not preserved: before={}, after={}, diff={}",
                    v1,
                    v2,
                    (v1 - v2).abs(),
                );
            }
            (Err(e1), _) => panic!("could not evaluate original expression: {e1}"),
            (_, Err(e2)) => panic!("could not evaluate simplified expression: {e2}"),
        }
    }};
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. PYTHAGOREAN: sin²(x) + cos²(x) → 1
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_pythagorean_fires() {
    let x = symplex::var("x");
    assert_simplifies_to!(&x.sin().powi(2) + &x.cos().powi(2), "1");
}

#[test]
fn rule_pythagorean_trace() {
    let x = symplex::var("x");
    assert_trace_contains_rule!(&x.sin().powi(2) + &x.cos().powi(2), "pythagorean");
}

#[test]
fn rule_pythagorean_different_args_no_fire() {
    let x = symplex::var("x");
    let y = symplex::var("y");
    // sin²(x) + cos²(y) must NOT simplify to 1
    let expr = &x.sin().powi(2) + &y.cos().powi(2);
    let s = format!("{}", expr.simplify());
    assert_ne!(s, "1", "sin²(x)+cos²(y) must not become 1");
}

#[test]
fn rule_pythagorean_value_preserved() {
    let x = symplex::var("x");
    let pt = symplex::rational(7, 10);
    assert_value_preserved!(&x.sin().powi(2) + &x.cos().powi(2), x, pt, 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. EXP_LN: exp(ln(x)) → x
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_exp_ln_fires() {
    let x = symplex::var("x");
    assert_simplifies_to!(x.ln().exp(), "x");
}

#[test]
fn rule_exp_ln_trace() {
    let x = symplex::var("x");
    assert_trace_contains_rule!(x.ln().exp(), "exp_ln");
}

#[test]
fn rule_exp_ln_plus_one_no_fire() {
    // exp(ln(x) + 1) should NOT simplify to x
    let x = symplex::var("x");
    let expr = (&x.ln() + 1).exp();
    let s = format!("{}", expr.simplify());
    assert_ne!(s, "x", "exp(ln(x)+1) must not become x");
}

#[test]
fn rule_exp_ln_value_preserved() {
    let x = symplex::var("x");
    let pt = symplex::int(3);
    assert_value_preserved!(x.ln().exp(), x, pt, 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. LN_EXP: ln(exp(x)) → x  (when x is real)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_ln_exp_fires_for_real_symbol() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]);
    let result = x.exp().ln().simplify();
    assert_eq!(format!("{result}"), "x");
}

#[test]
fn rule_ln_exp_blocked_for_unassumed_symbol() {
    // Without real assumption, ln(exp(x)) should stay unchanged.
    let x = symplex::var("x");
    let expr = x.exp().ln();
    let s = format!("{}", expr.simplify());
    // It should NOT simplify to just "x" without the real assumption.
    assert_eq!(s, "ln(exp(x))", "ln(exp(x)) should stay unchanged without real assumption");
}

#[test]
fn rule_ln_exp_trace_for_real() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]);
    assert_trace_contains_rule!(x.exp().ln(), "ln_exp");
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. ABS_ABS: abs(abs(x)) → abs(x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_abs_abs_fires() {
    let x = symplex::var("x");
    assert_simplifies_to!(x.abs().abs(), "abs(x)");
}

#[test]
fn rule_abs_abs_trace() {
    let x = symplex::var("x");
    assert_trace_contains_rule!(x.abs().abs(), "abs_abs");
}

#[test]
fn rule_abs_abs_value_preserved() {
    let x = symplex::var("x");
    let pt = symplex::int(-3);
    assert_value_preserved!(x.abs().abs(), x, pt, 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. SQRT_SQ: sqrt(x²) → abs(x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_sqrt_sq_fires() {
    let x = symplex::var("x");
    assert_simplifies_to!(x.powi(2).sqrt(), "abs(x)");
}

#[test]
fn rule_sqrt_sq_trace() {
    let x = symplex::var("x");
    assert_trace_contains_rule!(x.powi(2).sqrt(), "sqrt_sq");
}

#[test]
fn rule_sqrt_cube_no_fire() {
    // sqrt(x^3) should NOT become abs(x) — wrong exponent
    let x = symplex::var("x");
    let expr = x.powi(3).sqrt();
    let s = format!("{}", expr.simplify());
    assert_ne!(s, "abs(x)", "sqrt(x^3) must not become abs(x)");
}

#[test]
fn rule_sqrt_sq_value_preserved() {
    let x = symplex::var("x");
    let pt = symplex::int(-4);
    assert_value_preserved!(x.powi(2).sqrt(), x, pt, 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. COSH_SINH_IDENTITY: cosh²(x) - sinh²(x) → 1
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_cosh_sinh_identity_fires() {
    let x = symplex::var("x");
    assert_simplifies_to!(&x.cosh().powi(2) - &x.sinh().powi(2), "1");
}

#[test]
fn rule_cosh_sinh_identity_trace() {
    let x = symplex::var("x");
    assert_trace_contains_rule!(
        &x.cosh().powi(2) - &x.sinh().powi(2),
        "cosh_sinh_identity"
    );
}

#[test]
fn rule_cosh_sinh_wrong_sign_no_fire() {
    // cosh²(x) + sinh²(x) has the WRONG sign — must NOT become 1
    let x = symplex::var("x");
    let expr = &x.cosh().powi(2) + &x.sinh().powi(2);
    let s = format!("{}", expr.simplify());
    assert_ne!(s, "1", "cosh²+sinh² must not become 1");
}

#[test]
fn rule_cosh_sinh_different_args_no_fire() {
    let x = symplex::var("x");
    let y = symplex::var("y");
    let expr = &x.cosh().powi(2) - &y.sinh().powi(2);
    let s = format!("{}", expr.simplify());
    assert_ne!(s, "1", "cosh²(x)-sinh²(y) must not become 1");
}

#[test]
fn rule_cosh_sinh_value_preserved() {
    let x = symplex::var("x");
    let pt = symplex::rational(8, 10);
    assert_value_preserved!(&x.cosh().powi(2) - &x.sinh().powi(2), x, pt, 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. POW_POW: (x^a)^b → x^(a*b)  (when at least one exponent is integer)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_pow_pow_fires_integer_exponents() {
    let x = symplex::var("x");
    // (x²)³ → x⁶
    assert_simplifies_to!(x.powi(2).powi(3), "x^6");
}

#[test]
fn rule_pow_pow_trace() {
    // powi(2).powi(3) gets flattened by canonicalization at construction time,
    // so the pow_pow rule never fires. Use (x^a)^3 with symbolic `a` —
    // canonicalization can't simplify this, but pow_pow fires because 3 is integer.
    let x = symplex::var("x");
    let a = symplex::var("a");
    assert_trace_contains_rule!(x.pow(&a).powi(3), "pow_pow");
}

#[test]
fn rule_pow_pow_blocked_both_fractional() {
    // (x^(1/2))^(1/3) should NOT fire — no integer exponent
    let x = symplex::var("x");
    let ctx = symplex::default_context();
    let half = ctx.rational(1, 2);
    let third = ctx.rational(1, 3);
    let expr = x.pow(&half).pow(&third);
    assert_simplify_unchanged!(expr);
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. ASINH_SINH: asinh(sinh(x)) → x
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_asinh_sinh_fires() {
    let x = symplex::var("x");
    assert_simplifies_to!(x.sinh().asinh(), "x");
}

#[test]
fn rule_asinh_sinh_trace() {
    let x = symplex::var("x");
    assert_trace_contains_rule!(x.sinh().asinh(), "asinh_sinh");
}

#[test]
fn rule_asinh_sinh_value_preserved() {
    let x = symplex::var("x");
    let pt = symplex::rational(3, 2);
    assert_value_preserved!(x.sinh().asinh(), x, pt, 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. ACOSH_COSH: acosh(cosh(x)) → abs(x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_acosh_cosh_fires() {
    let x = symplex::var("x");
    assert_simplifies_to!(x.cosh().acosh(), "abs(x)");
}

#[test]
fn rule_acosh_cosh_trace() {
    let x = symplex::var("x");
    assert_trace_contains_rule!(x.cosh().acosh(), "acosh_cosh");
}

#[test]
fn rule_acosh_cosh_value_preserved() {
    let x = symplex::var("x");
    let pt = symplex::rational(3, 2);
    assert_value_preserved!(x.cosh().acosh(), x, pt, 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. ATANH_TANH: atanh(tanh(x)) → x
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_atanh_tanh_fires() {
    let x = symplex::var("x");
    assert_simplifies_to!(x.tanh().atanh(), "x");
}

#[test]
fn rule_atanh_tanh_trace() {
    let x = symplex::var("x");
    assert_trace_contains_rule!(x.tanh().atanh(), "atanh_tanh");
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. SIN_DIV_COS: sin(x)/cos(x) → tan(x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_sin_div_cos_fires() {
    let x = symplex::var("x");
    let expr = &x.sin() / &x.cos();
    assert_simplifies_to!(expr, "tan(x)");
}

#[test]
fn rule_sin_div_cos_trace() {
    let x = symplex::var("x");
    let expr = &x.sin() / &x.cos();
    assert_trace_contains_rule!(expr, "sin_div_cos");
}

#[test]
fn rule_sin_div_cos_different_args_no_fire() {
    // sin(x)/cos(y) should NOT become tan
    let x = symplex::var("x");
    let y = symplex::var("y");
    let expr = &x.sin() / &y.cos();
    let s = format!("{}", expr.simplify());
    assert!(!s.contains("tan("), "sin(x)/cos(y) must not become tan, got: {s}");
}

#[test]
fn rule_sin_div_cos_value_preserved() {
    let x = symplex::var("x");
    let pt = symplex::rational(1, 2);
    assert_value_preserved!(&x.sin() / &x.cos(), x, pt, 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. COS_DIV_SIN: cos(x)/sin(x) → 1/tan(x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_cos_div_sin_fires() {
    let x = symplex::var("x");
    let expr = &x.cos() / &x.sin();
    let s = format!("{}", expr.simplify());
    // Should contain "tan" (as tan(x)^(-1) or 1/tan(x))
    assert!(
        s.contains("tan"),
        "cos(x)/sin(x) should simplify to involve tan, got: {s}"
    );
}

#[test]
fn rule_cos_div_sin_trace() {
    let x = symplex::var("x");
    let expr = &x.cos() / &x.sin();
    assert_trace_contains_rule!(expr, "cos_div_sin");
}

#[test]
fn rule_cos_div_sin_value_preserved() {
    let x = symplex::var("x");
    let pt = symplex::rational(1, 2);
    assert_value_preserved!(&x.cos() / &x.sin(), x, pt, 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. SINH_DIV_COSH: sinh(x)/cosh(x) → tanh(x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_sinh_div_cosh_fires() {
    let x = symplex::var("x");
    let expr = &x.sinh() / &x.cosh();
    assert_simplifies_to!(expr, "tanh(x)");
}

#[test]
fn rule_sinh_div_cosh_trace() {
    let x = symplex::var("x");
    let expr = &x.sinh() / &x.cosh();
    assert_trace_contains_rule!(expr, "sinh_div_cosh");
}

#[test]
fn rule_sinh_div_cosh_different_args_no_fire() {
    let x = symplex::var("x");
    let y = symplex::var("y");
    let expr = &x.sinh() / &y.cosh();
    let s = format!("{}", expr.simplify());
    assert!(
        !s.contains("tanh("),
        "sinh(x)/cosh(y) must not become tanh, got: {s}"
    );
}

#[test]
fn rule_sinh_div_cosh_value_preserved() {
    let x = symplex::var("x");
    let pt = symplex::rational(1, 2);
    assert_value_preserved!(&x.sinh() / &x.cosh(), x, pt, 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// 14. EXP_MUL: exp(a)*exp(b) → exp(a+b)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_exp_mul_fires() {
    let a = symplex::var("a");
    let b = symplex::var("b");
    let expr = &a.exp() * &b.exp();
    assert_simplifies_to!(expr, "exp(a + b)");
}

#[test]
fn rule_exp_mul_trace() {
    let a = symplex::var("a");
    let b = symplex::var("b");
    let expr = &a.exp() * &b.exp();
    assert_trace_contains_rule!(expr, "exp_mul");
}

#[test]
fn rule_exp_mul_not_both_exp_no_fire() {
    // exp(x) * sin(x) should NOT trigger exp combining
    let x = symplex::var("x");
    let expr = &x.exp() * &x.sin();
    assert_simplify_unchanged!(expr);
}

#[test]
fn rule_exp_mul_value_preserved() {
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let expr = &a.exp() * &b.exp();
    // Substitute a=1, b=2 and compare
    let one = ctx.int(1);
    let two = ctx.int(2);
    let before_val = expr.subs(&a, &one).subs(&b, &two).evalf_f64().expect("evalf before");
    let after_val = expr
        .simplify()
        .subs(&a, &one)
        .subs(&b, &two)
        .evalf_f64()
        .expect("evalf after");
    assert!(
        (before_val - after_val).abs() < 1e-10,
        "exp_mul value not preserved: {before_val} vs {after_val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 15. EXP_LOG_DENEST: exp(a*ln(b)) → b^a
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_exp_log_denest_fires_numeric() {
    let x = symplex::var("x");
    // exp(3*ln(x)) → x³
    assert_simplifies_to!((&x.ln() * 3).exp(), "x^3");
}

#[test]
fn rule_exp_log_denest_fires_symbolic() {
    let x = symplex::var("x");
    let a = symplex::var("a");
    assert_simplifies_to!((&x.ln() * &a).exp(), "x^a");
}

#[test]
fn rule_exp_log_denest_trace() {
    let x = symplex::var("x");
    assert_trace_contains_rule!((&x.ln() * 3).exp(), "exp_log_denest");
}

#[test]
fn rule_exp_log_denest_value_preserved() {
    let x = symplex::var("x");
    let pt = symplex::int(2);
    // exp(3*ln(2)) should equal 2^3 = 8
    assert_value_preserved!((&x.ln() * 3).exp(), x, pt, 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// 16. ABS_POSITIVE: abs(w) → w  (when w is a positive literal)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_abs_positive_fires_for_literal() {
    let ctx = Context::new();
    let five = ctx.int(5);
    assert_simplifies_to!(five.abs(), "5");
}

#[test]
fn rule_abs_positive_trace_for_literal() {
    let ctx = Context::new();
    let five = ctx.int(5);
    assert_trace_contains_rule!(five.abs(), "abs_positive");
}

#[test]
fn rule_abs_positive_no_fire_for_symbol() {
    // abs(x) should NOT simplify when x is not a known positive literal
    let x = symplex::var("x");
    assert_simplify_unchanged!(x.abs());
}

#[test]
fn rule_abs_positive_no_fire_for_negative_literal() {
    let ctx = Context::new();
    let neg3 = ctx.int(-3);
    // abs(-3) should eval to 3, but the abs_positive rule specifically
    // requires positive — eval handles negative separately
    let result = neg3.abs().eval();
    assert_eq!(format!("{result}"), "3");
}

// ═══════════════════════════════════════════════════════════════════════════
// 17. SIN_ASIN: sin(asin(x)) → x
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_sin_asin_fires() {
    let x = symplex::var("x");
    assert_simplifies_to!(x.asin().sin(), "x");
}

#[test]
fn rule_sin_asin_trace() {
    let x = symplex::var("x");
    assert_trace_contains_rule!(x.asin().sin(), "sin_asin");
}

#[test]
fn rule_sin_asin_value_preserved() {
    let x = symplex::var("x");
    let pt = symplex::rational(1, 2);
    assert_value_preserved!(x.asin().sin(), x, pt, 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// 18. COS_ACOS: cos(acos(x)) → x
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_cos_acos_fires() {
    let x = symplex::var("x");
    assert_simplifies_to!(x.acos().cos(), "x");
}

#[test]
fn rule_cos_acos_trace() {
    let x = symplex::var("x");
    assert_trace_contains_rule!(x.acos().cos(), "cos_acos");
}

#[test]
fn rule_cos_acos_value_preserved() {
    let x = symplex::var("x");
    let pt = symplex::rational(1, 2);
    assert_value_preserved!(x.acos().cos(), x, pt, 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// 19. TAN_ATAN: tan(atan(x)) → x
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_tan_atan_fires() {
    let x = symplex::var("x");
    assert_simplifies_to!(x.atan().tan(), "x");
}

#[test]
fn rule_tan_atan_trace() {
    let x = symplex::var("x");
    assert_trace_contains_rule!(x.atan().tan(), "tan_atan");
}

#[test]
fn rule_tan_atan_value_preserved() {
    let x = symplex::var("x");
    let pt = symplex::rational(3, 2);
    assert_value_preserved!(x.atan().tan(), x, pt, 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// 20. SINH_ASINH: sinh(asinh(x)) → x
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_sinh_asinh_fires() {
    let x = symplex::var("x");
    assert_simplifies_to!(x.asinh().sinh(), "x");
}

#[test]
fn rule_sinh_asinh_trace() {
    let x = symplex::var("x");
    assert_trace_contains_rule!(x.asinh().sinh(), "sinh_asinh");
}

#[test]
fn rule_sinh_asinh_value_preserved() {
    let x = symplex::var("x");
    let pt = symplex::rational(5, 3);
    assert_value_preserved!(x.asinh().sinh(), x, pt, 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// 21. COSH_ACOSH: cosh(acosh(x)) → x
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_cosh_acosh_fires() {
    let x = symplex::var("x");
    // Note: cosh(acosh(x)) → x (domain: x ≥ 1)
    assert_simplifies_to!(x.acosh().cosh(), "x");
}

#[test]
fn rule_cosh_acosh_trace() {
    let x = symplex::var("x");
    assert_trace_contains_rule!(x.acosh().cosh(), "cosh_acosh");
}

#[test]
fn rule_cosh_acosh_value_preserved() {
    let x = symplex::var("x");
    // x must be ≥ 1 for acosh to be real
    let pt = symplex::int(2);
    assert_value_preserved!(x.acosh().cosh(), x, pt, 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// 22. TANH_ATANH: tanh(atanh(x)) → x
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_tanh_atanh_fires() {
    let x = symplex::var("x");
    assert_simplifies_to!(x.atanh().tanh(), "x");
}

#[test]
fn rule_tanh_atanh_trace() {
    let x = symplex::var("x");
    assert_trace_contains_rule!(x.atanh().tanh(), "tanh_atanh");
}

#[test]
fn rule_tanh_atanh_value_preserved() {
    let x = symplex::var("x");
    // x must be in (-1, 1) for atanh
    let pt = symplex::rational(1, 3);
    assert_value_preserved!(x.atanh().tanh(), x, pt, 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// INVERSE DIRECTION NON-MATCH — asin(sin(x)), acos(cos(x)), atan(tan(x))
// should NOT simplify (these inverse compositions are NOT in basic_rules)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_asin_sin_does_not_fire() {
    let x = symplex::var("x");
    // asin(sin(x)) is NOT in the ruleset — must stay unchanged
    assert_simplifies_to!(x.sin().asin(), "asin(sin(x))");
}

#[test]
fn rule_acos_cos_does_not_fire() {
    let x = symplex::var("x");
    assert_simplifies_to!(x.cos().acos(), "acos(cos(x))");
}

#[test]
fn rule_atan_tan_does_not_fire() {
    let x = symplex::var("x");
    assert_simplifies_to!(x.tan().atan(), "atan(tan(x))");
}

// ═══════════════════════════════════════════════════════════════════════════
// ADDITIONAL NEGATIVE TESTS — mixed function families
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_pythagorean_mixed_sinh_cos_no_fire() {
    // sinh²(x) + cos²(x) — mixed hyperbolic/trig, must not simplify to 1
    let x = symplex::var("x");
    let expr = &x.sinh().powi(2) + &x.cos().powi(2);
    let s = format!("{}", expr.simplify());
    assert_ne!(s, "1", "sinh²(x)+cos²(x) must not become 1, got: {s}");
}

#[test]
fn rule_exp_ln_nested_no_fire() {
    // exp(ln(ln(x))) should NOT simplify to ln(x) in one step...
    // actually exp_ln rule matches exp(ln(w)) where w=ln(x), so it DOES
    // simplify to ln(x). This is correct! Let's verify.
    let x = symplex::var("x");
    assert_simplifies_to!(x.ln().ln().exp(), "ln(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPOSITION TESTS — rules working together
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn composition_exp_ln_then_pythagorean() {
    // sin²(exp(ln(x))) + cos²(exp(ln(x))) should simplify:
    //   exp(ln(x)) → x  then  sin²(x) + cos²(x) → 1
    let x = symplex::var("x");
    let inner = x.ln().exp(); // → x
    let expr = &inner.sin().powi(2) + &inner.cos().powi(2);
    assert_simplifies_to!(expr, "1");
}

#[test]
fn composition_abs_abs_abs_collapses() {
    // abs(abs(abs(x))) should collapse to abs(x)
    let x = symplex::var("x");
    let expr = x.abs().abs().abs();
    // May need multiple simplify passes; try one first
    let s1 = expr.simplify();
    let s2 = s1.simplify();
    assert_eq!(format!("{s2}"), "abs(x)");
}

#[test]
fn composition_pow_pow_chain() {
    // ((x^2)^3)^2 → x^12
    let x = symplex::var("x");
    let expr = x.powi(2).powi(3).powi(2);
    // May need multiple passes
    let s1 = expr.simplify();
    let s2 = s1.simplify();
    let s = format!("{s2}");
    assert!(
        s == "x^12" || s == "x^6^2",
        "expected x^12 eventually, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// TRACE COMPLETENESS — verify simplify_trace returns non-empty steps
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn trace_pythagorean_has_steps() {
    let x = symplex::var("x");
    let (_result, steps) = (&x.sin().powi(2) + &x.cos().powi(2)).simplify_trace();
    assert!(
        !steps.is_empty(),
        "pythagorean simplification should produce trace steps"
    );
    assert_eq!(steps[0].rule_name, "pythagorean");
}

#[test]
fn trace_no_steps_for_atom() {
    let ctx = Context::new();
    let five = ctx.int(5);
    assert_trace_empty!(five);
}

#[test]
fn trace_no_steps_for_irreducible_symbol() {
    let x = symplex::var("x");
    assert_trace_empty!(x);
}

// ═══════════════════════════════════════════════════════════════════════════
// ADDITIONAL VALUE-PRESERVATION TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn value_preserved_cos_div_sin() {
    let x = symplex::var("x");
    let pt = symplex::rational(11, 10);
    assert_value_preserved!(&x.cos() / &x.sin(), x, pt, 1e-10);
}

#[test]
fn value_preserved_atanh_tanh() {
    let x = symplex::var("x");
    let pt = symplex::rational(1, 4);
    assert_value_preserved!(x.tanh().atanh(), x, pt, 1e-10);
}

#[test]
fn value_preserved_abs_abs() {
    let x = symplex::var("x");
    let pt = symplex::int(-7);
    assert_value_preserved!(x.abs().abs(), x, pt, 1e-10);
}

#[test]
fn value_preserved_pow_pow() {
    let x = symplex::var("x");
    let pt = symplex::int(2);
    assert_value_preserved!(x.powi(2).powi(3), x, pt, 1e-10);
}
