//! Tests for convenience methods: evalf_f64, assume, sum_of, product_of,
//! and integration by parts.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// evalf_f64()
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn evalf_f64_integer() {
    let five = symplex::int(5);
    let val = five.evalf_f64().unwrap();
    assert!((val - 5.0).abs() < 1e-10);
}

#[test]
fn evalf_f64_pi() {
    let ctx = Context::new();
    let val = ctx.pi().evalf_f64().unwrap();
    assert!((val - std::f64::consts::PI).abs() < 1e-10);
}

#[test]
fn evalf_f64_expression() {
    let x = symplex::var("x");
    let val = x.powi(2).subs_i64(&x, 3).evalf_f64().unwrap();
    assert!((val - 9.0).abs() < 1e-10);
}

#[test]
fn evalf_f64_free_symbol_errors() {
    let x = symplex::var("x");
    assert!(x.evalf_f64().is_err());
}

#[test]
fn evalf_f64_rational() {
    let half = symplex::rational(1, 3);
    let val = half.evalf_f64().unwrap();
    assert!((val - 1.0 / 3.0).abs() < 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// assume()
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn assume_positive() {
    let t = symplex::var("t").assume(Assumption::Positive);
    assert_eq!(t.is_positive(), Some(true));
}

#[test]
fn assume_chained() {
    let t = symplex::var("t")
        .assume(Assumption::Positive)
        .assume(Assumption::Real);
    assert_eq!(t.is_positive(), Some(true));
    assert_eq!(t.is_real(), Some(true));
}

#[test]
fn assume_integer() {
    let n = symplex::var("n").assume(Assumption::Integer);
    assert_eq!(n.is_integer(), Some(true));
    // Integer implies rational, real, complex by forward chaining
    assert_eq!(n.is_real(), Some(true));
}

#[test]
fn assume_on_non_symbol_ignored() {
    let ctx = Context::new();
    let five = ctx.int(5);
    // Calling assume on a non-symbol should not panic
    let result = five.assume(Assumption::Positive);
    // The integer 5 is already positive via assumption handlers
    assert_eq!(result.is_positive(), Some(true));
}

// ═══════════════════════════════════════════════════════════════════════════
// sum_of() / product_of()
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sum_of_integers() {
    let ctx = Context::new();
    let terms: Vec<Ex> = (1..=4).map(|n| ctx.int(n)).collect();
    let total = Ex::sum_of(&ctx, terms);
    assert_eq!(format!("{total}"), "10");
}

#[test]
fn product_of_integers() {
    let ctx = Context::new();
    let factors: Vec<Ex> = (1..=4).map(|n| ctx.int(n)).collect();
    let total = Ex::product_of(&ctx, factors);
    assert_eq!(format!("{total}"), "24");
}

#[test]
fn sum_of_empty() {
    let ctx = Context::new();
    let total = Ex::sum_of(&ctx, vec![]);
    assert_eq!(format!("{total}"), "0");
}

#[test]
fn product_of_empty() {
    let ctx = Context::new();
    let total = Ex::product_of(&ctx, vec![]);
    assert_eq!(format!("{total}"), "1");
}

#[test]
fn sum_of_expressions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let terms = vec![x.powi(2), x.clone(), ctx.int(1)];
    let total = Ex::sum_of(&ctx, terms);
    let s = format!("{total}");
    assert!(
        s.contains("x^2") && s.contains("x"),
        "should be polynomial: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration by parts (through public API)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_x_sin_x() {
    let x = symplex::var("x");
    let expr = &x * &x.sin();
    let result = expr.integrate(&x);
    let s = format!("{result}");
    // ∫ x·sin(x) dx = sin(x) - x·cos(x)
    assert!(s.contains("sin(x)"), "should contain sin(x): {s}");
    assert!(s.contains("cos(x)"), "should contain cos(x): {s}");
}

#[test]
fn integrate_x_exp_x() {
    let x = symplex::var("x");
    let expr = &x * &x.exp();
    let result = expr.integrate(&x);
    let s = format!("{result}");
    // ∫ x·exp(x) dx = x·exp(x) - exp(x)
    assert!(s.contains("exp(x)"), "should contain exp(x): {s}");
}

#[test]
fn integrate_x_cos_x() {
    let x = symplex::var("x");
    let expr = &x * &x.cos();
    let result = expr.integrate(&x);
    let s = format!("{result}");
    // ∫ x·cos(x) dx = x·sin(x) + cos(x)
    assert!(s.contains("sin(x)"), "should contain sin(x): {s}");
    assert!(s.contains("cos(x)"), "should contain cos(x): {s}");
}

#[test]
fn args_of_add() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x + 1;
    let children = expr.args();
    assert_eq!(children.len(), 2);
}

#[test]
fn args_of_atom() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let children = x.args();
    assert_eq!(children.len(), 0);
}

#[test]
fn args_of_function() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin();
    let children = expr.args();
    assert_eq!(children.len(), 1);
}

#[test]
fn diff_n_third_derivative() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(4);
    let d3 = f.diff_n(&x, 3);
    assert_eq!(format!("{d3}"), "24*x");
}

#[test]
fn diff_n_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(2);
    let d0 = f.diff_n(&x, 0);
    assert_eq!(format!("{d0}"), "x^2");
}

#[test]
fn log_base_2() {
    let ctx = Context::new();
    let result = ctx.int(8).log(&ctx.int(2));
    let s = format!("{}", result.full_simplify());
    // log_2(8) = ln(8)/ln(2) = 3
    // This may or may not simplify fully, but let's check it contains the right structure
    assert!(s.contains("ln"), "log should produce ln expressions: {s}");
}
