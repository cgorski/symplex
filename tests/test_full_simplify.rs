//! Tests for full_simplify(), sqrt(x^2)→abs(x) rule, and full_simplify_trace().

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// full_simplify()
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn full_simplify_expand_plus_cancel() {
    // (x+1)^2 - x^2 - 2*x should become 1 after expand + canonicalization
    let x = symplex::var("x");
    let expr = &(&x + 1).powi(2) - &x.powi(2) - &x * 2;
    assert_eq!(format!("{}", expr.full_simplify()), "1");
}

#[test]
fn full_simplify_trig_identity() {
    let x = symplex::var("x");
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    assert_eq!(format!("{}", expr.full_simplify()), "1");
}

#[test]
fn full_simplify_exp_ln() {
    let x = symplex::var("x");
    let expr = x.ln().exp();
    assert_eq!(format!("{}", expr.full_simplify()), "x");
}

#[test]
fn full_simplify_nested_eval_then_simplify() {
    // sin(0) + cos(0) should become 0 + 1 = 1 after eval + canonicalization
    let ctx = Context::new();
    let zero = ctx.int(0);
    let expr = &zero.sin() + &zero.cos();
    assert_eq!(format!("{}", expr.full_simplify()), "1");
}

#[test]
fn full_simplify_already_simple() {
    let x = symplex::var("x");
    let expr = &x + 1;
    assert_eq!(format!("{}", expr.full_simplify()), "x + 1");
}

#[test]
fn full_simplify_pythagorean_in_larger_sum() {
    let x = symplex::var("x");
    let expr = &x.sin().powi(2) + &x.cos().powi(2) + 5;
    assert_eq!(format!("{}", expr.full_simplify()), "6");
}

// ═══════════════════════════════════════════════════════════════════════════
// sqrt(x^2) → abs(x) rule
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_sqrt_of_square() {
    let x = symplex::var("x");
    let expr = x.powi(2).sqrt();
    assert_eq!(format!("{}", expr.simplify()), "abs(x)");
}

#[test]
fn simplify_sqrt_of_square_in_sum() {
    let x = symplex::var("x");
    let expr = &x.powi(2).sqrt() + 1;
    let simplified = expr.simplify();
    let s = format!("{simplified}");
    assert!(s.contains("abs(x)"), "should contain abs(x): {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// full_simplify_trace()
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn full_simplify_trace_records_steps() {
    let x = symplex::var("x");
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    let (result, steps) = expr.full_simplify_trace();
    assert_eq!(format!("{result}"), "1");
    assert!(!steps.is_empty(), "should have recorded at least one step");
    // The pythagorean rule should have fired
    assert!(
        steps.iter().any(|s| s.rule_name == "pythagorean"),
        "should have fired pythagorean rule"
    );
}

#[test]
fn full_simplify_trace_empty_when_no_change() {
    let x = symplex::var("x");
    let expr = &x + 1;
    let (_result, steps) = expr.full_simplify_trace();
    assert!(steps.is_empty(), "no rules should fire on x + 1");
}
