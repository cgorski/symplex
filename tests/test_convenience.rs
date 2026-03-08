//! Tests for convenience methods: evalf_f64, assume, sum_of, product_of,
//! and integration by parts.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// evalf_f64()
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn evalf_f64_integer() {
    let five = symplex::default_context().int(5);
    let val = five.eval_f64().unwrap();
    assert!((val - 5.0).abs() < 1e-10);
}

#[test]
fn evalf_f64_pi() {
    let ctx = Context::new();
    let val = ctx.pi().eval_f64().unwrap();
    assert!((val - std::f64::consts::PI).abs() < 1e-10);
}

#[test]
fn evalf_f64_expression() {
    let x = symplex::default_context().symbol("x");
    let val = x.powi(2).subs_i64(&x, 3).eval_f64().unwrap();
    assert!((val - 9.0).abs() < 1e-10);
}

#[test]
fn evalf_f64_free_symbol_errors() {
    let x = symplex::default_context().symbol("x");
    assert!(x.eval_f64().is_err());
}

#[test]
fn evalf_f64_rational() {
    let half = symplex::default_context().rational(1, 3);
    let val = half.eval_f64().unwrap();
    assert!((val - 1.0 / 3.0).abs() < 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// assume()
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn assume_positive() {
    let t = symplex::default_context().symbol("t").assume(Assumption::Positive);
    assert_eq!(t.is_positive(), Some(true));
}

#[test]
fn assume_chained() {
    let t = symplex::default_context().symbol("t")
        .assume(Assumption::Positive)
        .assume(Assumption::Real);
    assert_eq!(t.is_positive(), Some(true));
    assert_eq!(t.is_real(), Some(true));
}

#[test]
fn assume_integer() {
    let n = symplex::default_context().symbol("n").assume(Assumption::Integer);
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

    // Verify that assume actually works on symbols by checking a non-positive value.
    // A symbol with no assumptions should return None for is_positive().
    let y = ctx.symbol("y");
    assert_eq!(y.is_positive(), None, "bare symbol should not be known positive");
    // After assuming positive, it should be Some(true).
    let y_pos = y.assume(Assumption::Positive);
    assert_eq!(y_pos.is_positive(), Some(true), "assumed-positive symbol should be positive");
    // And a negative integer should be known not positive.
    let neg = ctx.int(-3);
    assert_eq!(neg.is_positive(), Some(false), "-3 should not be positive");
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
    let x = symplex::default_context().symbol("x");
    let expr = &x * &x.sin();
    let result = expr.integrate(&x);
    let s = format!("{result}");
    // ∫ x·sin(x) dx = sin(x) - x·cos(x)
    assert!(s.contains("sin(x)"), "should contain sin(x): {s}");
    assert!(s.contains("cos(x)"), "should contain cos(x): {s}");

    // Numerical FTC check: ∫₁² x·sin(x) dx ≈ F(2) - F(1)
    // where F is the antiderivative we just computed.
    let f_at_2 = result.subs_i64(&x, 2).eval().eval_f64()
        .expect("F(2) should evaluate");
    let f_at_1 = result.subs_i64(&x, 1).eval().eval_f64()
        .expect("F(1) should evaluate");
    let ftc_value = f_at_2 - f_at_1;

    // Reference: numerical integration of x*sin(x) from 1 to 2
    // ∫₁² x·sin(x) dx = [sin(x) - x·cos(x)]₁² = sin(2) - 2·cos(2) - sin(1) + cos(1)
    let expected = 2.0_f64.sin() - 2.0 * 2.0_f64.cos() - 1.0_f64.sin() + 1.0_f64.cos();
    assert!(
        (ftc_value - expected).abs() < 1e-9,
        "FTC check failed: F(2)-F(1) = {ftc_value}, expected {expected}"
    );
}

#[test]
fn integrate_x_exp_x() {
    let x = symplex::default_context().symbol("x");
    let expr = &x * &x.exp();
    let result = expr.integrate(&x);
    let s = format!("{result}");
    // ∫ x·exp(x) dx = x·exp(x) - exp(x)
    assert!(s.contains("exp(x)"), "should contain exp(x): {s}");
}

#[test]
fn integrate_x_cos_x() {
    let x = symplex::default_context().symbol("x");
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
    // log_2(8) = ln(8)/ln(2) = 3
    // The symbolic form may not fully simplify, so verify numerically.
    let val = result.eval().eval_f64()
        .expect("log_2(8) should evaluate to a float");
    assert!(
        (val - 3.0).abs() < 1e-9,
        "log_2(8) should equal 3, got {val}"
    );
    // Also check that the symbolic form at least contains ln (structural sanity)
    let s = format!("{}", result.full_simplify());
    assert!(
        s == "3" || s.contains("ln"),
        "log should simplify to 3 or produce ln expressions: {s}"
    );
}
