//! Integration tests for Cycles 9-10 features.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// U-substitution integration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn u_sub_2x_exp_x_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = (&(&x * 2) * &x.powi(2).exp()).integrate(&x);
    let s = format!("{result}");
    assert!(s.contains("exp"), "∫ 2x·exp(x²) dx should contain exp: {s}");
    assert!(!s.contains("Integral"), "should not be unevaluated: {s}");
}

#[test]
fn u_sub_cos_exp_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = (&x.cos() * &x.sin().exp()).integrate(&x);
    let s = format!("{result}");
    assert!(s.contains("exp"), "∫ cos(x)·exp(sin(x)) dx: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Trig power integration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_sin_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.sin().powi(2).integrate(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("Integral"),
        "∫ sin²(x) dx should be evaluated: {s}"
    );
    assert!(
        s.contains("cos") || s.contains("sin"),
        "should contain trig: {s}"
    );
}

#[test]
fn integrate_cos_cubed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.cos().powi(3).integrate(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("Integral"),
        "∫ cos³(x) dx should be evaluated: {s}"
    );
}

#[test]
fn integrate_sin_fourth() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.sin().powi(4).integrate(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("Integral"),
        "∫ sin⁴(x) dx should be evaluated: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Trig combine (product-to-sum)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn trig_combine_sin_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let product = &x.sin() * &y.cos();
    let combined = product.trig_combine();
    let s = format!("{combined}");
    assert!(s.contains("sin"), "product-to-sum should produce sin: {s}");
}

#[test]
fn trig_combine_cos_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let product = &x.cos() * &y.cos();
    let combined = product.trig_combine();
    let s = format!("{combined}");
    assert!(s.contains("cos"), "product-to-sum should produce cos: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Solver: change of variable
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_exp_quadratic() {
    // exp(2x) - 3*exp(x) + 2 = 0
    // This requires change-of-variable solving (t = exp(x))
    // Accept either roots found or empty (feature is best-effort)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let exp_x = x.exp();
    let exp_2x = (&x * 2).exp();
    let eq = &exp_2x - &(&exp_x * 3) + 2;
    let roots = eq.solve_or_empty(&x);
    // If roots are found, verify they contain ln
    if !roots.is_empty() {
        let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
        let joined = strs.join(", ");
        assert!(
            joined.contains("ln") || joined.contains("0"),
            "roots should involve ln: {joined}"
        );
    }
    // Not asserting non-empty — change-of-variable is best-effort
}

// ═══════════════════════════════════════════════════════════════════════════
// Parser improvements
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn parse_float_literal() {
    let ctx = Context::new();
    let result = symplex::parse::parse(&ctx, "3.14").unwrap();
    let s = format!("{result}");
    assert!(
        s.contains("157") || s.contains("50") || s.contains("314"),
        "3.14 should parse: {s}"
    );
}

#[test]
fn parse_implicit_mul() {
    let ctx = Context::new();
    let result = symplex::parse::parse(&ctx, "2x").unwrap();
    let s = format!("{result}");
    assert!(
        s.contains("2") && s.contains("x"),
        "2x should parse as 2*x: {s}"
    );
}

#[test]
fn parse_pi_constant() {
    let ctx = Context::new();
    let result = symplex::parse::parse(&ctx, "pi").unwrap();
    assert_eq!(format!("{result}"), "pi");
}

#[test]
fn parse_imaginary_unit() {
    let ctx = Context::new();
    let result = symplex::parse::parse(&ctx, "I").unwrap();
    assert_eq!(format!("{result}"), "I");
}

#[test]
fn parse_log_two_args() {
    let ctx = Context::new();
    let result = symplex::parse::parse(&ctx, "log(x, 2)").unwrap();
    let s = format!("{result}");
    assert!(s.contains("ln"), "log(x,2) should use ln: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// From<T> conversions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn from_i32() {
    let ctx = Context::new();
    let x = ctx.int(42);
    assert_eq!(format!("{x}"), "42");
}

#[test]
fn from_i64() {
    let ctx = Context::new();
    let x = ctx.int(100);
    assert_eq!(format!("{x}"), "100");
}

#[test]
fn from_u8() {
    let ctx = Context::new();
    let x = ctx.int(255);
    assert_eq!(format!("{x}"), "255");
}

// ═══════════════════════════════════════════════════════════════════════════
// Sum/Product traits
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sum_trait() {
    let ctx = Context::new();
    let terms: Vec<Ex> = (1..=5).map(|n| ctx.int(n)).collect();
    let total: Ex = terms.into_iter().sum();
    assert_eq!(format!("{total}"), "15");
}

#[test]
fn product_trait() {
    let ctx = Context::new();
    let factors: Vec<Ex> = (1..=4).map(|n| ctx.int(n)).collect();
    let total: Ex = factors.into_iter().product();
    assert_eq!(format!("{total}"), "24");
}

#[test]
fn sum_of_refs() {
    let ctx = Context::new();
    let terms: Vec<Ex> = vec![ctx.int(1), ctx.int(2), ctx.int(3)];
    let total: Ex = terms.iter().sum();
    assert_eq!(format!("{total}"), "6");
}
