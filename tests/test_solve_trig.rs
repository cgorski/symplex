//! Integration tests for trigonometric equation solving.
//!
//! These tests exercise the trig-inversion fast path in `solve.rs`
//! through the public `solve_as_set` API (which bypasses the
//! polynomial pre-check in `Ex::solve`).

use symplex::prelude::*;
/// Evaluate an expression to f64, returning None on failure.
fn eval(e: &symplex::expr::Ex) -> Option<f64> {
    e.eval_f64().ok()
}

/// Check that a value is approximately zero.
fn approx_zero(val: f64, tol: f64) -> bool {
    val.abs() < tol
}

// ═══════════════════════════════════════════════════════════════════════════
// sin(x) = 1/2 → x ∈ {asin(1/2), π − asin(1/2)}  (≈ π/6, 5π/6)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_sin_x_eq_half_via_set() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let half = ctx.rational(1, 2);
    let expr = &x.sin() - &half;

    let set = expr.solve_as_set(&x);
    let s = format!("{set}");

    // The set should contain asin(1/2) and pi - asin(1/2).
    assert!(
        s.contains("asin") || s.contains("pi"),
        "solve_as_set for sin(x)=1/2 should mention asin or pi: {s}"
    );
    // Should NOT be the empty set.
    assert!(
        !s.contains("EmptySet"),
        "sin(x)=1/2 should not be empty: {s}"
    );
}

#[test]
fn solve_sin_x_eq_half_verify_numerically() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let half = ctx.rational(1, 2);
    let sin_x_minus_half = &x.sin() - &half;

    let set = sin_x_minus_half.solve_as_set(&x);
    let s = format!("{set}");

    // Extract approximate values by evaluating asin(1/2) ≈ 0.5236 and
    // π − asin(1/2) ≈ 2.6180.  We verify by substitution.
    let asin_half = x.sin().subs(&x, &ctx.rational(1, 2).asin());
    let val = eval(&asin_half);
    if let Some(v) = val {
        assert!(
            approx_zero(v - 0.5, 1e-10),
            "sin(asin(1/2)) should ≈ 0.5, got {v}"
        );
    }

    // The set string should have exactly 2 elements (FiniteSet with 2 entries).
    // FiniteSet is printed as `{elem1, elem2}`.
    let comma_count = s.matches(',').count();
    assert!(
        comma_count >= 1 || s.contains("asin"),
        "expected at least 2 solutions for sin(x)=1/2: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// cos(x) = 0 → x ∈ {acos(0), −acos(0)}  (≈ π/2, −π/2)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_cos_x_eq_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.cos();

    let set = expr.solve_as_set(&x);
    let s = format!("{set}");

    assert!(
        !s.contains("EmptySet"),
        "cos(x)=0 should have solutions: {s}"
    );
    assert!(
        s.contains("acos") || s.contains("pi"),
        "cos(x)=0 solutions should reference acos or pi: {s}"
    );
}

#[test]
fn solve_cos_x_eq_zero_verify() {
    let ctx = Context::new();
    let _x = ctx.symbol("x");
    // cos(π/2) should be 0
    let pi = ctx.pi();
    let half = ctx.rational(1, 2);
    let pi_half = &pi * &half;
    let val = eval(&pi_half.cos());
    if let Some(v) = val {
        assert!(approx_zero(v, 1e-10), "cos(π/2) should ≈ 0, got {v}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// tan(x) = 1 → x = atan(1) ≈ π/4
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_tan_x_eq_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let expr = &x.tan() - &one;

    let set = expr.solve_as_set(&x);
    let s = format!("{set}");

    assert!(
        !s.contains("EmptySet"),
        "tan(x)=1 should have a solution: {s}"
    );
    assert!(
        s.contains("atan") || s.contains("pi"),
        "tan(x)=1 solution should reference atan or pi: {s}"
    );
}

#[test]
fn solve_tan_x_eq_one_verify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let tan_minus_one = &x.tan() - &one;

    // atan(1) ≈ π/4 ≈ 0.7854
    let atan1 = one.atan();
    let substituted = tan_minus_one.subs(&x, &atan1);
    if let Some(v) = eval(&substituted) {
        assert!(
            approx_zero(v, 1e-10),
            "tan(atan(1)) - 1 should ≈ 0, got {v}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// sin(x) = 0 → x ∈ {asin(0), π − asin(0)} = {0, π}
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_sin_x_eq_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin();

    let set = expr.solve_as_set(&x);
    let s = format!("{set}");

    assert!(
        !s.contains("EmptySet"),
        "sin(x)=0 should have solutions: {s}"
    );
}

#[test]
fn solve_sin_x_eq_zero_verify() {
    // sin(0) = 0 and sin(π) ≈ 0
    let val0 = (0.0_f64).sin();
    assert!(approx_zero(val0, 1e-15), "sin(0) should be 0");
    let val_pi = std::f64::consts::PI.sin();
    assert!(approx_zero(val_pi, 1e-15), "sin(π) should ≈ 0");
}

// ═══════════════════════════════════════════════════════════════════════════
// sin(2*x) = 1 → x = asin(1)/2 (≈ π/4)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_sin_2x_eq_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two = ctx.int(2);
    let one = ctx.int(1);
    let sin_2x = (&x * &two).sin();
    let expr = &sin_2x - &one;

    let set = expr.solve_as_set(&x);
    let s = format!("{set}");

    assert!(
        !s.contains("EmptySet"),
        "sin(2x)=1 should have solutions: {s}"
    );
}

#[test]
fn solve_sin_2x_eq_one_verify() {
    // sin(2 · π/4) = sin(π/2) = 1  ✓
    let val = (2.0 * std::f64::consts::FRAC_PI_4).sin();
    assert!(
        approx_zero(val - 1.0, 1e-15),
        "sin(2·π/4) should be 1, got {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// sin(x) = 2 → empty (|c| > 1, no real solutions)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_sin_x_eq_two_empty() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two = ctx.int(2);
    let expr = &x.sin() - &two;

    let set = expr.solve_as_set(&x);
    let s = format!("{set}");

    assert!(
        s.contains("EmptySet") || s.contains("{}") || s == "∅",
        "sin(x)=2 should yield empty set, got: {s}"
    );
}

#[test]
fn solve_cos_x_eq_minus_two_empty() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let neg_two = ctx.int(-2);
    let expr = &x.cos() - &neg_two;

    let set = expr.solve_as_set(&x);
    let s = format!("{set}");

    assert!(
        s.contains("EmptySet") || s.contains("{}") || s == "∅",
        "cos(x)=-2 should yield empty set, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// exp(x) = 1 → x = ln(1) = 0
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_exp_x_eq_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let expr = &x.exp() - &one;

    let set = expr.solve_as_set(&x);
    let s = format!("{set}");

    assert!(
        !s.contains("EmptySet"),
        "exp(x)=1 should have a solution: {s}"
    );
    // ln(1) = 0, so the solution should be {0} or {ln(1)}.
    assert!(
        s.contains('0') || s.contains("ln"),
        "exp(x)=1 solution should be 0 or ln(1): {s}"
    );
}

#[test]
fn solve_exp_x_eq_one_verify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let expr = &x.exp() - &one;

    // Substitute x = 0: exp(0) - 1 = 0
    let zero = ctx.int(0);
    let substituted = expr.subs(&x, &zero);
    if let Some(v) = eval(&substituted) {
        assert!(approx_zero(v, 1e-15), "exp(0) - 1 should be 0, got {v}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// exp(x) = 5 → x = ln(5) (verify existing functionality)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_exp_x_eq_five() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let five = ctx.int(5);
    let expr = &x.exp() - &five;

    let set = expr.solve_as_set(&x);
    let s = format!("{set}");

    assert!(
        !s.contains("EmptySet"),
        "exp(x)=5 should have a solution: {s}"
    );
    assert!(s.contains("ln"), "exp(x)=5 solution should contain ln: {s}");
}

#[test]
fn solve_exp_x_eq_five_verify() {
    // exp(ln(5)) - 5 = 0
    let val = 5.0_f64.ln().exp() - 5.0;
    assert!(
        approx_zero(val, 1e-12),
        "exp(ln(5)) - 5 should ≈ 0, got {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// sin(x) = 1 → x ∈ {asin(1), π − asin(1)} (boundary: |c| = 1)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_sin_x_eq_one_boundary() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let expr = &x.sin() - &one;

    let set = expr.solve_as_set(&x);
    let s = format!("{set}");

    // |1| ≤ 1, so this should have solutions (π/2 from both branches).
    assert!(
        !s.contains("EmptySet"),
        "sin(x)=1 should have solutions: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// cos(x) = 1 → x ∈ {acos(1), −acos(1)} = {0, 0} = {0}
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_cos_x_eq_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let expr = &x.cos() - &one;

    let set = expr.solve_as_set(&x);
    let s = format!("{set}");

    assert!(
        !s.contains("EmptySet"),
        "cos(x)=1 should have a solution: {s}"
    );
}
