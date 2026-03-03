//! Tests for definite integrals, degree(), and coeffs().

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Definite integrals
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn definite_integral_x_squared_0_to_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ∫₀¹ x² dx = 1/3
    let result = x.powi(2).definite_integral(&x, &ctx.int(0), &ctx.int(1));
    assert_eq!(format!("{result}"), "1/3");
}

#[test]
fn definite_integral_constant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ∫₀³ 5 dx = 15
    let result = ctx.int(5).definite_integral(&x, &ctx.int(0), &ctx.int(3));
    assert_eq!(format!("{result}"), "15");
}

#[test]
fn definite_integral_linear() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ∫₀² x dx = 2
    let result = x.definite_integral(&x, &ctx.int(0), &ctx.int(2));
    assert_eq!(format!("{result}"), "2");
}

#[test]
fn definite_integral_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ∫₁² (x^2 + x) dx = [x^3/3 + x^2/2]₁² = (8/3 + 2) - (1/3 + 1/2) = 14/3 - 5/6 = 23/6
    let expr = &x.powi(2) + &x;
    let result = expr.definite_integral(&x, &ctx.int(1), &ctx.int(2));
    assert_eq!(format!("{result}"), "23/6");
}

#[test]
fn definite_integral_sin_0_to_pi() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ∫₀^π sin(x) dx = [-cos(x)]₀^π = -cos(π) - (-cos(0)) = 1 + 1 = 2
    let result = x.sin().definite_integral(&x, &ctx.int(0), &ctx.pi());
    let evaled = result.eval();
    assert_eq!(format!("{evaled}"), "2");
}

#[test]
fn definite_integral_same_bounds_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(2).definite_integral(&x, &ctx.int(3), &ctx.int(3));
    assert_eq!(format!("{result}"), "0");
}

// ═══════════════════════════════════════════════════════════════════════════
// degree()
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn degree_quadratic() {
    let x = symplex::var("x");
    let expr = &x.powi(2) + &x + 1;
    assert_eq!(expr.degree(&x), Some(2));
}

#[test]
fn degree_cubic() {
    let x = symplex::var("x");
    let expr = &x.powi(3) + 1;
    assert_eq!(expr.degree(&x), Some(3));
}

#[test]
fn degree_constant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(ctx.int(5).degree(&x), Some(0));
}

#[test]
fn degree_linear() {
    let x = symplex::var("x");
    assert_eq!((&x + 1).degree(&x), Some(1));
}

#[test]
fn degree_non_polynomial() {
    let x = symplex::var("x");
    assert_eq!(x.sin().degree(&x), None);
}

#[test]
fn degree_zero_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(ctx.int(0).degree(&x), None);
}

// ═══════════════════════════════════════════════════════════════════════════
// coeffs()
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn coeffs_quadratic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x^2 + 3*x + 5
    let expr = &x.powi(2) + &x * 3 + 5;
    let cs = expr.coeffs(&x).unwrap();
    let strs: Vec<String> = cs.iter().map(|c| format!("{c}")).collect();
    assert_eq!(strs, vec!["5", "3", "1"]);
}

#[test]
fn coeffs_linear() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x * 2 + 7;
    let cs = expr.coeffs(&x).unwrap();
    let strs: Vec<String> = cs.iter().map(|c| format!("{c}")).collect();
    assert_eq!(strs, vec!["7", "2"]);
}

#[test]
fn coeffs_constant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cs = ctx.int(42).coeffs(&x).unwrap();
    let strs: Vec<String> = cs.iter().map(|c| format!("{c}")).collect();
    assert_eq!(strs, vec!["42"]);
}

#[test]
fn coeffs_non_polynomial() {
    let x = symplex::var("x");
    assert!(x.sin().coeffs(&x).is_none());
}
