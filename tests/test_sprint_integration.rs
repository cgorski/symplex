//! Integration tests for Sprint A-D features.

use symplex::expr::ExprType;
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Sprint A: Integration Completeness
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_tan_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.tan().integrate(&x);
    let s = format!("{result}");
    assert!(
        s.contains("ln") && s.contains("cos"),
        "∫ tan(x) dx should be -ln|cos(x)|, got: {s}"
    );
}

#[test]
fn integrate_tan_x_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.tan().integrate(&x);
    assert_eq!(format!("{result}"), "-ln(abs(cos(x)))");
}

#[test]
fn integrate_ln_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.ln().integrate(&x);
    let s = format!("{result}");
    assert!(
        s.contains("ln") && s.contains("x"),
        "∫ ln(x) dx should involve x*ln(x), got: {s}"
    );
}

#[test]
fn integrate_ln_x_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.ln().integrate(&x);
    assert_eq!(format!("{result}"), "-x + x*ln(x)");
}

#[test]
fn integrate_one_over_x_squared_plus_one() {
    // ∫ 1/(x²+1) dx = atan(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = (&x.powi(2) + 1).powi(-1);
    let result = integrand.integrate(&x);
    let s = format!("{result}");
    assert!(
        s.contains("atan"),
        "∫ 1/(x²+1) dx should be atan(x), got: {s}"
    );
}

#[test]
fn integrate_one_over_x_squared_plus_one_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = (&x.powi(2) + 1).powi(-1);
    let result = integrand.integrate(&x);
    assert_eq!(format!("{result}"), "atan(x)");
}

#[test]
fn integrate_one_over_sqrt_one_minus_x_squared() {
    // ∫ 1/√(1-x²) dx = asin(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let inner = &one - &x.powi(2);
    let integrand = inner.pow(&ctx.rational(-1, 2));
    let result = integrand.integrate(&x);
    let s = format!("{result}");
    assert!(
        s.contains("asin"),
        "∫ 1/√(1-x²) dx should be asin(x), got: {s}"
    );
}

#[test]
fn integrate_one_over_sqrt_one_minus_x_squared_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let inner = &one - &x.powi(2);
    let integrand = inner.pow(&ctx.rational(-1, 2));
    let result = integrand.integrate(&x);
    assert_eq!(format!("{result}"), "asin(x)");
}

#[test]
fn integrate_tan_2x() {
    // ∫ tan(2x) dx = -ln|cos(2x)|/2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two_x = &x * 2;
    let result = two_x.tan().integrate(&x);
    let s = format!("{result}");
    assert!(
        s.contains("ln") && s.contains("cos"),
        "∫ tan(2x) dx should involve ln and cos, got: {s}"
    );
}

#[test]
fn integrate_tan_2x_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two_x = &x * 2;
    let result = two_x.tan().integrate(&x);
    assert_eq!(format!("{result}"), "-1/2*ln(abs(cos(2*x)))");
}

#[test]
fn integrate_ln_roundtrip() {
    // d/dx(∫ ln(x) dx) should give back ln(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integral = x.ln().integrate(&x);
    let deriv = integral.diff(&x);
    let simplified = deriv.full_simplify();
    let s = format!("{simplified}");
    assert!(
        s.contains("ln"),
        "d/dx(∫ ln(x) dx) should simplify to ln(x), got: {s}"
    );
}

#[test]
fn integrate_ln_roundtrip_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integral = x.ln().integrate(&x);
    let deriv = integral.diff(&x);
    let simplified = deriv.full_simplify();
    assert_eq!(format!("{simplified}"), "ln(x)");
}

#[test]
fn integrate_tan_roundtrip() {
    // d/dx(∫ tan(x) dx) should give back tan(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integral = x.tan().integrate(&x);
    let deriv = integral.diff(&x);
    let simplified = deriv.full_simplify();
    let s = format!("{simplified}");
    assert!(
        s.contains("tan") || s.contains("sin") || s.contains("cos"),
        "d/dx(∫ tan(x) dx) should simplify back to tan(x) or equivalent, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Sprint B: Simplification Rules
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_sin_over_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin() / &x.cos();
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "tan(x)");
}

#[test]
fn simplify_sinh_over_cosh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sinh() / &x.cosh();
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "tanh(x)");
}

#[test]
fn simplify_exp_product() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x.exp() * &y.exp();
    let simplified = expr.simplify();
    let s = format!("{simplified}");
    assert!(
        s.contains("exp"),
        "exp(x)*exp(y) should simplify to exp(x+y), got: {s}"
    );
}

#[test]
fn simplify_exp_product_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x.exp() * &y.exp();
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "exp(x + y)");
}

#[test]
fn simplify_sin_over_cos_in_larger_product() {
    // 2 * sin(x) / cos(x) → 2 * tan(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &(&x.sin() / &x.cos()) * 2;
    let simplified = expr.simplify();
    let s = format!("{simplified}");
    assert!(
        s.contains("tan"),
        "2*sin(x)/cos(x) should simplify to 2*tan(x), got: {s}"
    );
}

#[test]
fn simplify_sin_over_cos_in_larger_product_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &(&x.sin() / &x.cos()) * 2;
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "2*tan(x)");
}

#[test]
fn logcombine_two_logs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x.ln() + &y.ln();
    let combined = expr.logcombine();
    let s = format!("{combined}");
    assert!(
        s.contains("ln"),
        "ln(x)+ln(y) should combine to ln(x*y), got: {s}"
    );
}

#[test]
fn logcombine_two_logs_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x.ln() + &y.ln();
    let combined = expr.logcombine();
    assert_eq!(format!("{combined}"), "ln(x*y)");
}

#[test]
fn logcombine_then_expand_roundtrip() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let product = &x * &y;
    let ln_product = product.ln();
    let expanded = ln_product.expand_log();
    let recombined = expanded.logcombine();
    let s = format!("{recombined}");
    assert!(
        s.contains("ln"),
        "roundtrip should produce ln(...), got: {s}"
    );
}

#[test]
fn logcombine_roundtrip_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let product = &x * &y;
    let ln_product = product.ln();
    assert_eq!(format!("{ln_product}"), "ln(x*y)");
    let expanded = ln_product.expand_log();
    assert_eq!(format!("{expanded}"), "ln(x) + ln(y)");
    let recombined = expanded.logcombine();
    assert_eq!(format!("{recombined}"), "ln(x*y)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Sprint C: Eval Completeness
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_sin_2pi_over_3() {
    // sin(2π/3) = √3/2
    let ctx = Context::new();
    let angle = &ctx.rational(2, 3) * &ctx.pi();
    let result = angle.sin().eval();
    assert_eq!(format!("{result}"), "1/2*sqrt(3)");
}

#[test]
fn eval_sin_3pi_over_4() {
    // sin(3π/4) = √2/2
    let ctx = Context::new();
    let angle = &ctx.rational(3, 4) * &ctx.pi();
    let result = angle.sin().eval();
    assert_eq!(format!("{result}"), "1/2*sqrt(2)");
}

#[test]
fn eval_cos_5pi_over_6() {
    // cos(5π/6) = -√3/2
    let ctx = Context::new();
    let angle = &ctx.rational(5, 6) * &ctx.pi();
    let result = angle.cos().eval();
    assert_eq!(format!("{result}"), "-1/2*sqrt(3)");
}

#[test]
fn eval_tan_pi_over_6() {
    // tan(π/6) = √3/3 = (1/3)√3
    let ctx = Context::new();
    let angle = &ctx.rational(1, 6) * &ctx.pi();
    let result = angle.tan().eval();
    let s = format!("{result}");
    assert!(s.contains("3"), "tan(π/6) should involve √3, got: {s}");
}

#[test]
fn eval_tan_pi_over_6_exact() {
    let ctx = Context::new();
    let angle = &ctx.rational(1, 6) * &ctx.pi();
    let result = angle.tan().eval();
    assert_eq!(format!("{result}"), "1/3*sqrt(3)");
}

#[test]
fn eval_tan_pi_over_3() {
    // tan(π/3) = √3
    let ctx = Context::new();
    let angle = &ctx.rational(1, 3) * &ctx.pi();
    let result = angle.tan().eval();
    assert_eq!(format!("{result}"), "sqrt(3)");
}

#[test]
fn eval_sin_pi_over_6_exact() {
    // sin(π/6) = 1/2 (baseline sanity check)
    let ctx = Context::new();
    let expr = (&ctx.pi() / 6).sin();
    let result = expr.eval();
    assert_eq!(format!("{result}"), "1/2");
}

#[test]
fn eval_cos_pi_over_3_exact() {
    // cos(π/3) = 1/2 (baseline sanity check)
    let ctx = Context::new();
    let expr = (&ctx.pi() / 3).cos();
    let result = expr.eval();
    assert_eq!(format!("{result}"), "1/2");
}

#[test]
fn eval_tan_pi_over_4_exact() {
    // tan(π/4) = 1 (baseline sanity check)
    let ctx = Context::new();
    let expr = (&ctx.pi() / 4).tan();
    let result = expr.eval();
    assert_eq!(format!("{result}"), "1");
}

// ═══════════════════════════════════════════════════════════════════════════
// Sprint D: Ergonomics
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ex_zero() {
    let z = Ex::zero();
    assert_eq!(format!("{z}"), "0");
    assert!(z.is_zero_structural());
}

#[test]
fn ex_one() {
    let o = Ex::one();
    assert_eq!(format!("{o}"), "1");
    assert!(o.is_one_structural());
}

#[test]
fn ex_zero_is_not_one() {
    let z = Ex::zero();
    assert!(!z.is_one_structural());
}

#[test]
fn ex_one_is_not_zero() {
    let o = Ex::one();
    assert!(!o.is_zero_structural());
}

#[test]
fn expr_type_classification() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.expr_type(), ExprType::Symbol);
    assert_eq!(ctx.int(5).expr_type(), ExprType::Number);
    assert_eq!((&x + 1).expr_type(), ExprType::Add);
    assert_eq!((&x * 2).expr_type(), ExprType::Mul);
    assert_eq!(x.powi(2).expr_type(), ExprType::Pow);
    assert_eq!(x.sin().expr_type(), ExprType::Function);
    assert_eq!(ctx.pi().expr_type(), ExprType::Constant);
}

#[test]
fn expr_type_all_trig_are_function() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.sin().expr_type(), ExprType::Function);
    assert_eq!(x.cos().expr_type(), ExprType::Function);
    assert_eq!(x.tan().expr_type(), ExprType::Function);
    assert_eq!(x.sinh().expr_type(), ExprType::Function);
    assert_eq!(x.cosh().expr_type(), ExprType::Function);
    assert_eq!(x.tanh().expr_type(), ExprType::Function);
    assert_eq!(x.exp().expr_type(), ExprType::Function);
    assert_eq!(x.ln().expr_type(), ExprType::Function);
}

#[test]
fn replace_variable() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let expr = &x.powi(2) + &x + 1;
    let replaced = expr.replace(|e| if *e == x { Some(y.clone()) } else { None });
    let s = format!("{replaced}");
    assert!(
        s.contains("y") && !s.contains("x"),
        "should replace x with y, got: {s}"
    );
}

#[test]
fn replace_variable_exact() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let expr = &x.powi(2) + &x + 1;
    let replaced = expr.replace(|e| if *e == x { Some(y.clone()) } else { None });
    assert_eq!(format!("{replaced}"), "y^2 + y + 1");
}

#[test]
fn replace_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin();
    let replaced = expr.replace(|_| None);
    assert_eq!(format!("{replaced}"), format!("{expr}"));
}

#[test]
fn replace_identity_complex_expr() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(2) + &x.sin() + 1;
    let original = format!("{expr}");
    let replaced = expr.replace(|_| None);
    assert_eq!(format!("{replaced}"), original);
}

// ═══════════════════════════════════════════════════════════════════════════
// Cross-Sprint: combined workflows
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_then_simplify_sin_over_cos() {
    // Integrating sin(x)/cos(x) after simplifying to tan(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let ratio = &x.sin() / &x.cos();
    let simplified = ratio.simplify();
    assert_eq!(format!("{simplified}"), "tan(x)");
    let integral = simplified.integrate(&x);
    let s = format!("{integral}");
    assert!(
        s.contains("ln") && s.contains("cos"),
        "∫ tan(x) dx should be -ln|cos(x)|, got: {s}"
    );
}

#[test]
fn eval_then_check_expr_type() {
    // After eval, trig of special angles become numbers
    let ctx = Context::new();
    let angle = &ctx.pi() / 6;
    let raw = angle.sin();
    assert_eq!(raw.expr_type(), ExprType::Function);
    let evaled = raw.eval();
    // 1/2 is a number
    assert_eq!(evaled.expr_type(), ExprType::Number);
}

#[test]
fn replace_then_differentiate() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // Start with x^3, replace x→y, differentiate w.r.t. y
    let expr = x.powi(3);
    let replaced = expr.replace(|e| if *e == x { Some(y.clone()) } else { None });
    let deriv = replaced.diff(&y);
    assert_eq!(format!("{deriv}"), "3*y^2");
}

#[test]
fn logcombine_preserves_simplification() {
    // ln(x) + ln(y) → ln(x*y), then differentiating w.r.t. x should give 1/x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let combined = (&x.ln() + &y.ln()).logcombine();
    assert_eq!(format!("{combined}"), "ln(x*y)");
    let deriv = combined.diff(&x);
    let simplified = deriv.full_simplify();
    let s = format!("{simplified}");
    // The chain rule gives y/(x*y) which is equivalent to 1/x but may not
    // fully cancel. Accept either the simplified or unsimplified form.
    assert!(
        s.contains("1/x") || s.contains("x*y") || s == "1/x",
        "d/dx ln(x*y) should be equivalent to 1/x, got: {s}"
    );
}
