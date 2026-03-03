//! Tests for complex number support.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// i^n canonicalization
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn i_squared_is_neg_one() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = i.powi(2);
    assert_eq!(format!("{result}"), "-1");
}

#[test]
fn i_cubed_is_neg_i() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = i.powi(3);
    assert_eq!(format!("{result}"), "-I");
}

#[test]
fn i_fourth_is_one() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = i.powi(4);
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn i_to_100() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = i.powi(100);
    assert_eq!(format!("{result}"), "1"); // 100 mod 4 = 0
}

#[test]
fn i_to_neg_one() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = i.powi(-1);
    assert_eq!(format!("{result}"), "-I"); // i^(-1) = -i
}

#[test]
fn i_to_neg_two() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = i.powi(-2);
    assert_eq!(format!("{result}"), "-1"); // i^(-2) = -1
}

// ═══════════════════════════════════════════════════════════════════════════
// Complex algebra
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn one_plus_i_squared() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let expr = (&ctx.int(1) + &i).powi(2).expand();
    let s = format!("{expr}");
    // (1+i)^2 = 1 + 2i + i^2 = 1 + 2i - 1 = 2i
    assert_eq!(s, "2*I");
}

#[test]
fn i_times_i() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = &i * &i;
    assert_eq!(format!("{result}"), "-1");
}

#[test]
fn complex_addition() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let z1 = &ctx.int(2) + &(&ctx.int(3) * &i);
    let z2 = &ctx.int(4) + &(&ctx.int(5) * &i);
    let sum = &z1 + &z2;
    let s = format!("{sum}");
    // (2+3i) + (4+5i) = 6+8i
    assert_eq!(s, "6 + 8*I");
}

#[test]
fn diff_complex_expression() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let i = ctx.i_unit();
    // d/dx(i*x^2) = 2*i*x
    let expr = &i * &x.powi(2);
    let result = expr.diff(&x);
    let s = format!("{result}");
    assert!(
        s.contains("I") && s.contains("x"),
        "d/dx(i*x²) should be 2*i*x, got: {s}"
    );
}

#[test]
fn integrate_complex_expression() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let i = ctx.i_unit();
    // ∫ i*x dx = i*x²/2
    let expr = &i * &x;
    let result = expr.integrate(&x);
    let s = format!("{result}");
    assert!(
        s.contains("I") && s.contains("x"),
        "∫ i*x dx should involve I and x, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Assumptions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn i_is_imaginary() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    assert_eq!(i.query(Props::IMAGINARY), Some(true));
    assert_eq!(i.query(Props::REAL), Some(false));
    assert_eq!(i.query(Props::COMPLEX), Some(true));
}

#[test]
fn i_squared_is_real() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let i2 = i.powi(2);
    // i^2 canonicalizes to -1, which is real
    assert_eq!(i2.is_real(), Some(true));
    assert_eq!(i2.is_negative(), Some(true));
}
