//! Tests for full separable and variable-coefficient first-order linear ODE solvers.
//!
//! Full separable: y' = f(x)*g(y)
//!   - y' = x*y → y = C1*exp(x²/2)
//!   - y' = -x*y → y = C1*exp(-x²/2)
//!
//! Variable-coefficient linear: y' + P(x)*y = Q(x)
//!   - y' + 2x*y = 0 → y = C1*exp(-x²)
//!   - y' + y/x = x → y = (1/x)*(x³/3 + C1)
//!
//! Regression: existing ODE types (constant-coeff, simple separable, 2nd order) still work.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Numerically verify a first-order ODE solution by substitution.
fn verify_first_order(
    ode_expr: &Ex,
    solution: &Ex,
    constants: &[Ex],
    y: &Ex,
    x: &Ex,
    sample_x_num: i64,
    sample_x_den: i64,
) {
    let ctx = ode_expr.context();
    let one = ctx.int(1);
    let mut concrete_sol = solution.clone();
    for c in constants {
        concrete_sol = concrete_sol.subs(c, &one);
    }

    let sol_prime = concrete_sol.diff(x);

    let dy_formal = y.formal_diff(x);
    let residual = ode_expr.subs(&dy_formal, &sol_prime).subs(y, &concrete_sol);

    let sample_val = ctx.rational(sample_x_num, sample_x_den);
    let residual_at = residual.subs(x, &sample_val);

    let val = residual_at
        .eval_f64()
        .expect("residual should evaluate to f64 for numerical ODE verification");
    assert!(
        val.abs() < 1e-6,
        "First-order ODE residual should be ~0, got {val} at x={sample_x_num}/{sample_x_den}\n  \
         solution (C=1): {concrete_sol}\n  residual: {residual_at}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Full separable: y' = f(x)*g(y)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_full_separable_dy_dx_eq_xy() {
    let ctx = Context::new();
    // y' - xy = 0 → y' = xy → separable → y = C1·exp(x²/2)
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let ode = expr!(ctx, diff(y, x) - x * y);
    let sol = ode.solve_ode(&y, &x);
    assert!(!sol.has_unevaluated(), "y' = xy should be solvable");

    let s = format!("{sol}");
    assert!(s.contains("C1"), "solution should have C1: {s}");
    assert!(s.contains("exp"), "solution should contain exp: {s}");

    let c1 = ctx.symbol("C1");
    verify_first_order(&ode, &sol, &[c1], &y, &x, 1, 2);
}

#[test]
fn ode_full_separable_neg_xy() {
    let ctx = Context::new();
    // y' + xy = 0 → y' = -xy → separable with f(x) = -x, g(y) = y
    // Solution: y = C1·exp(-x²/2)
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let ode = expr!(ctx, diff(y, x) + x * y);
    let sol = ode.solve_ode(&y, &x);
    assert!(!sol.has_unevaluated(), "y' = -xy should be solvable");

    let s = format!("{sol}");
    assert!(s.contains("C1"), "solution should have C1: {s}");
    assert!(s.contains("exp"), "solution should contain exp: {s}");

    let c1 = ctx.symbol("C1");
    verify_first_order(&ode, &sol, &[c1], &y, &x, 1, 1);
}

#[test]
fn ode_full_separable_3xy() {
    let ctx = Context::new();
    // y' - 3*x*y = 0 → y' = 3xy → y = C1·exp(3x²/2)
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);
    let three = ctx.int(3);
    let ode = &dy - &(&three * &x * &y);
    let sol = ode.solve_ode(&y, &x);
    assert!(!sol.has_unevaluated(), "y' = 3xy should be solvable");

    let s = format!("{sol}");
    assert!(s.contains("C1"), "solution should have C1: {s}");
    assert!(s.contains("exp"), "solution should contain exp: {s}");

    let c1 = ctx.symbol("C1");
    verify_first_order(&ode, &sol, &[c1], &y, &x, 1, 4);
}

// ═══════════════════════════════════════════════════════════════════════════
// Variable-coefficient first-order linear: y' + P(x)*y = Q(x)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_variable_coeff_linear_homogeneous() {
    let ctx = Context::new();
    // y' + 2xy = 0 → y = C1·exp(-x²)
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let ode = expr!(ctx, diff(y, x) + 2 * x * y);
    let sol = ode.solve_ode(&y, &x);
    assert!(!sol.has_unevaluated(), "y' + 2xy = 0 should be solvable");

    let s = format!("{sol}");
    assert!(s.contains("C1"), "solution should have C1: {s}");
    assert!(s.contains("exp"), "solution should contain exp: {s}");

    let c1 = ctx.symbol("C1");
    verify_first_order(&ode, &sol, &[c1], &y, &x, 1, 2);
}

#[test]
fn ode_variable_coeff_linear_3x_squared_y() {
    let ctx = Context::new();
    // y' + 3x²y = 0 → P(x) = 3x², ∫P dx = x³ → y = C1·exp(-x³)
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);
    let three = ctx.int(3);
    let x_sq = x.powi(2);
    let ode = &dy + &(&three * &x_sq * &y);
    let sol = ode.solve_ode(&y, &x);
    assert!(!sol.has_unevaluated(), "y' + 3x²y = 0 should be solvable");

    let s = format!("{sol}");
    assert!(s.contains("C1"), "solution should have C1: {s}");
    assert!(s.contains("exp"), "solution should contain exp: {s}");

    let c1 = ctx.symbol("C1");
    verify_first_order(&ode, &sol, &[c1], &y, &x, 1, 3);
}

// ═══════════════════════════════════════════════════════════════════════════
// Regression: existing ODE types still work
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_existing_types_still_work() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    // Regression: y' + 2y = 0 (constant coeff) still works
    let ode = expr!(ctx, diff(y, x) + 2 * y);
    let result = ode.solve_ode(&y, &x);
    assert!(
        !result.has_unevaluated(),
        "y' + 2y = 0 should still be solvable"
    );

    // Regression: y'' + y = 0 still works (second-order CC)
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode2 = &d2y + &y;
    // Complex roots — should still return Some (even if trig/complex form)
    let result2 = ode2.solve_ode(&y, &x);
    assert!(
        !result2.has_unevaluated(),
        "y'' + y = 0 should return Some (second-order CC with complex roots)"
    );

    // Regression: y' = x still works (simple separable)
    let ode3 = expr!(ctx, diff(y, x) - x);
    assert!(
        !ode3.solve_ode(&y, &x).has_unevaluated(),
        "y' = x should still be solvable"
    );
}

#[test]
fn ode_constant_coeff_not_broken_by_new_dispatch() {
    let ctx = Context::new();
    // y' + 5y = 0 → y = C1·exp(-5x)
    // This should still be caught by the constant-coefficient path, even though
    // the variable-coefficient solver now runs first in dispatch order.
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let ode = expr!(ctx, diff(y, x) + 5 * y);
    let sol = ode.try_solve_ode(&y, &x).expect("y' + 5y = 0 should solve");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "solution should have C1: {s}");
    assert!(s.contains("exp"), "solution should contain exp: {s}");

    let c1 = ctx.symbol("C1");
    verify_first_order(&ode, &sol, &[c1], &y, &x, 1, 2);
}

#[test]
fn ode_no_y_dependence_still_simple_separable() {
    let ctx = Context::new();
    // y' = x² + 1 should still be handled by simple separable, not full separable
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);
    let ode = &dy - &(x.powi(2) + ctx.int(1));
    let sol = ode
        .try_solve_ode(&y, &x)
        .expect("y' = x² + 1 should be solvable");
    let s = format!("{sol}");
    assert!(s.contains("C1"), "solution should have C1: {s}");

    let c1 = ctx.symbol("C1");
    verify_first_order(&ode, &sol, &[c1], &y, &x, 1, 1);
}
