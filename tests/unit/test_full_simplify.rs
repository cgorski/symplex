//! Tests for simplify(), sqrt(x^2)→abs(x) rule, and related simplifications.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// full_simplify()
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn full_simplify_expand_plus_cancel() {
    let ctx = Context::new();
    // (x+1)^2 - x^2 - 2*x should become 1 after expand + canonicalization
    let x = ctx.symbol("x");
    let expr = &(&x + 1).powi(2) - &x.powi(2) - &x * 2;
    assert_eq!(format!("{}", expr.simplify()), "1");
}

#[test]
fn full_simplify_trig_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    assert_eq!(format!("{}", expr.simplify()), "1");
}

#[test]
fn full_simplify_exp_ln() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.ln().exp();
    assert_eq!(format!("{}", expr.simplify()), "x");
}

#[test]
fn full_simplify_nested_eval_then_simplify() {
    // sin(0) + cos(0) should become 0 + 1 = 1 after eval + canonicalization
    let ctx = Context::new();
    let zero = ctx.int(0);
    let expr = &zero.sin() + &zero.cos();
    assert_eq!(format!("{}", expr.simplify()), "1");
}

#[test]
fn full_simplify_already_simple() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x + 1;
    assert_eq!(format!("{}", expr.simplify()), "x + 1");
}

#[test]
fn full_simplify_pythagorean_in_larger_sum() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(2) + &x.cos().powi(2) + 5;
    assert_eq!(format!("{}", expr.simplify()), "6");
}

// ═══════════════════════════════════════════════════════════════════════════
// sqrt(x^2) → abs(x) rule
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_sqrt_of_square() {
    let ctx = Context::new();
    // 0.23: the identity needs a real argument (a symbol without assumptions may be complex).
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let expr = x.powi(2).sqrt();
    assert_eq!(format!("{}", expr.simplify()), "abs(x)");
}

#[test]
fn simplify_sqrt_of_square_in_sum() {
    let ctx = Context::new();
    // 0.23: the identity needs a real argument (a symbol without assumptions may be complex).
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let expr = &x.powi(2).sqrt() + 1;
    let simplified = expr.simplify();
    let s = format!("{simplified}");
    assert!(s.contains("abs(x)"), "should contain abs(x): {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// simplify() on expressions that formerly tested full_simplify_trace()
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_pythagorean_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    let result = expr.simplify();
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn simplify_no_change_on_simple_expr() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x + 1;
    let result = expr.simplify();
    assert_eq!(format!("{result}"), "x + 1");
}
