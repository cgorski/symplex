//! Integration tests for the finite difference module.
//!
//! Tests exercise `finite_diff_weights`, `apply_finite_diff`,
//! `equispaced_grid`, and the public `Ex::differentiate_finite()` API.
//!
//! (0.2: the free functions take `Ex` grids instead of `Arena` + `ExprId`,
//! and `differentiate_finite` takes an explicit stencil and order.)

use symplex::finite_diff::{apply_finite_diff, equispaced_grid, finite_diff_weights};
use symplex::prelude::*;

fn strs(v: &[Ex]) -> Vec<String> {
    v.iter().map(|e| e.to_string()).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Ex::differentiate_finite — public API
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn differentiate_finite_replaces_derivative_node() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let h = ctx.symbol("h");
    let stencil = equispaced_grid(&x, &h, 1);
    let expr = x.powi(2).formal_diff(&x);
    let finite = expr.differentiate_finite(&x, &stencil, 0);
    let s = format!("{finite}");
    assert!(
        !s.contains("Derivative"),
        "should not contain unevaluated Derivative: {s}"
    );
    assert!(s.contains('h'), "should contain step size symbol h: {s}");
    // Central difference of x² is exactly 2x.
    assert_eq!(finite.expand().to_string(), "2*x");
}

#[test]
fn differentiate_finite_x_cubed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let h = ctx.symbol("h");
    let stencil = equispaced_grid(&x, &h, 1);
    let expr = x.powi(3).formal_diff(&x);
    let finite = expr.differentiate_finite(&x, &stencil, 0);
    let s = format!("{finite}");
    assert!(
        !s.contains("Derivative"),
        "should not contain Derivative: {s}"
    );
    // (f(x+h) − f(x−h))/(2h) for x³ = 3x² + h²
    assert_eq!(finite.expand().to_string(), "h^2 + 3*x^2");
}

#[test]
fn differentiate_finite_sin_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let h = ctx.symbol("h");
    let stencil = equispaced_grid(&x, &h, 1);
    let expr = x.sin().formal_diff(&x);
    let finite = expr.differentiate_finite(&x, &stencil, 0);
    let s = format!("{finite}");
    assert!(
        !s.contains("Derivative"),
        "should not contain Derivative: {s}"
    );
    assert!(s.contains('h'), "should contain h: {s}");
    assert!(s.contains("sin"), "should reference sin: {s}");
    // Numerically: at x = 1, h = 1e-3 the central difference ≈ cos(1)
    let v = finite
        .subs_i64(&x, 1)
        .subs(&h, &ctx.rational(1, 1000))
        .eval_f64()
        .unwrap();
    assert!((v - 1f64.cos()).abs() < 1e-6, "{v}");
}

#[test]
fn differentiate_finite_preserves_non_derivative_expr() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let h = ctx.symbol("h");
    let stencil = equispaced_grid(&x, &h, 1);
    let expr = x.powi(2);
    let finite = expr.differentiate_finite(&x, &stencil, 0);
    assert_eq!(
        format!("{expr}"),
        format!("{finite}"),
        "non-derivative expression should be unchanged at order 0"
    );
}

#[test]
fn differentiate_finite_direct_order() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let h = ctx.symbol("h");
    let stencil = equispaced_grid(&x, &h, 1);
    // second central difference of x⁴: 12x² + 2h²
    let d2 = x.powi(4).differentiate_finite(&x, &stencil, 2).expand();
    assert_eq!(d2.to_string(), "2*h^2 + 12*x^2");
}

// ═══════════════════════════════════════════════════════════════════════════
// finite_diff_weights
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn forward_diff_weights_two_points() {
    let ctx = Context::new();
    let w = finite_diff_weights(1, &[ctx.int(0), ctx.int(1)], &ctx.int(0));
    assert_eq!(strs(&w), ["-1", "1"]);
}

#[test]
fn central_diff_first_derivative_weights() {
    let ctx = Context::new();
    let grid = [ctx.int(-1), ctx.int(0), ctx.int(1)];
    let w = finite_diff_weights(1, &grid, &ctx.int(0));
    assert_eq!(strs(&w), ["-1/2", "0", "1/2"]);
}

#[test]
fn central_diff_second_derivative_weights() {
    let ctx = Context::new();
    let grid = [ctx.int(-1), ctx.int(0), ctx.int(1)];
    let w = finite_diff_weights(2, &grid, &ctx.int(0));
    assert_eq!(strs(&w), ["1", "-2", "1"]);
}

#[test]
fn four_point_forward_first_deriv_weights() {
    let ctx = Context::new();
    let grid = [ctx.int(0), ctx.int(1), ctx.int(2), ctx.int(3)];
    let w = finite_diff_weights(1, &grid, &ctx.int(0));
    assert_eq!(strs(&w), ["-11/6", "3", "-3/2", "1/3"]);
}

#[test]
fn symbolic_grid_weights_are_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let h = ctx.symbol("h");
    let grid = equispaced_grid(&x, &h, 2);
    // five-point first derivative: [1/12, -2/3, 0, 2/3, -1/12] / h
    let w = finite_diff_weights(1, &grid, &x);
    let scaled: Vec<String> = w
        .iter()
        .map(|wi| (wi * &h).simplify().to_string())
        .collect();
    assert_eq!(scaled, ["1/12", "-2/3", "0", "2/3", "-1/12"]);
}

// ═══════════════════════════════════════════════════════════════════════════
// apply_finite_diff
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn apply_finite_diff_quadratic_first_deriv_at_zero() {
    let ctx = Context::new();
    let xs = [ctx.int(-1), ctx.int(0), ctx.int(1)];
    let ys = [ctx.int(1), ctx.int(0), ctx.int(1)];
    let r = apply_finite_diff(1, &xs, &ys, &ctx.int(0)).unwrap();
    assert_eq!(r.to_string(), "0");
}

#[test]
fn apply_finite_diff_quadratic_second_deriv() {
    let ctx = Context::new();
    let xs = [ctx.int(-1), ctx.int(0), ctx.int(1)];
    let ys = [ctx.int(1), ctx.int(0), ctx.int(1)];
    let r = apply_finite_diff(2, &xs, &ys, &ctx.int(0)).unwrap();
    assert_eq!(r.to_string(), "2");
}

#[test]
fn apply_finite_diff_linear_first_deriv() {
    let ctx = Context::new();
    let xs = [ctx.int(0), ctx.int(1)];
    let ys = [ctx.int(1), ctx.int(4)];
    let r = apply_finite_diff(1, &xs, &ys, &ctx.int(0)).unwrap();
    assert_eq!(r.to_string(), "3");
}

#[test]
fn apply_finite_diff_rejects_mismatched_lengths() {
    let ctx = Context::new();
    let xs = [ctx.int(0), ctx.int(1)];
    let ys = [ctx.int(1)];
    assert!(matches!(
        apply_finite_diff(1, &xs, &ys, &ctx.int(0)),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// equispaced_grid
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn equispaced_grid_correct_count() {
    let ctx = Context::new();
    let h = ctx.symbol("h");
    let grid = equispaced_grid(&ctx.int(0), &h, 2);
    assert_eq!(grid.len(), 5);
    assert_eq!(strs(&grid), ["-2*h", "-h", "0", "h", "2*h"]);
}

#[test]
fn equispaced_grid_single_point() {
    let ctx = Context::new();
    let h = ctx.symbol("h");
    let grid = equispaced_grid(&ctx.int(0), &h, 0);
    assert_eq!(grid.len(), 1);
}
