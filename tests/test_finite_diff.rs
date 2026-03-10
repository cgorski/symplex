//! Integration tests for the finite difference module.
//!
//! Tests exercise `finite_diff_weights`, `apply_finite_diff`,
//! `differentiate_finite`, and the public `Ex::differentiate_finite()` API.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Ex::differentiate_finite — public API
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn differentiate_finite_replaces_derivative_node() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2).formal_diff(&x);
    let finite = expr.differentiate_finite(&x);
    let s = format!("{finite}");
    // Should not contain a raw Derivative
    assert!(
        !s.contains("Derivative"),
        "should not contain unevaluated Derivative: {s}"
    );
    // Should mention the step size symbol _h
    assert!(s.contains("_h"), "should contain step size symbol _h: {s}");
}

#[test]
fn differentiate_finite_x_cubed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(3).formal_diff(&x);
    let finite = expr.differentiate_finite(&x);
    let s = format!("{finite}");
    assert!(
        !s.contains("Derivative"),
        "should not contain Derivative: {s}"
    );
    assert!(s.contains("_h"), "should contain _h: {s}");
}

#[test]
fn differentiate_finite_sin_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin().formal_diff(&x);
    let finite = expr.differentiate_finite(&x);
    let s = format!("{finite}");
    assert!(
        !s.contains("Derivative"),
        "should not contain Derivative: {s}"
    );
    assert!(s.contains("_h"), "should contain _h: {s}");
    // Should contain sin since the function body is sin
    assert!(s.contains("sin"), "should reference sin: {s}");
}

#[test]
fn differentiate_finite_preserves_non_derivative_expr() {
    let ctx = Context::new();
    // An expression with no Derivative nodes should pass through unchanged
    let x = ctx.symbol("x");
    let expr = x.powi(2);
    let finite = expr.differentiate_finite(&x);
    let original = format!("{expr}");
    let result = format!("{finite}");
    assert_eq!(
        original, result,
        "non-derivative expression should be unchanged"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Arena-level finite_diff_weights — accessed through with_arena_mut
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn forward_diff_weights_two_points() {
    // Forward difference: grid [0, 1], x0 = 0, 1st derivative
    // Expected weights: [-1, 1]
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let zero = arena.int(0);
        let one = arena.int(1);
        let x_list = vec![zero, one];

        let weights = symplex::finite_diff::finite_diff_weights(arena, 1, &x_list, zero);

        let w = &weights[1][1];
        assert_eq!(w.len(), 2);

        let w0_s = arena.display(w[0]).to_string();
        let w1_s = arena.display(w[1]).to_string();
        assert_eq!(w0_s, "-1", "first weight should be -1, got {w0_s}");
        assert_eq!(w1_s, "1", "second weight should be 1, got {w1_s}");
    });
}

#[test]
fn central_diff_first_derivative_weights() {
    // Central difference: grid [-1, 0, 1], x0 = 0
    // 1st derivative weights: [-1/2, 0, 1/2]
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let zero = arena.int(0);
        let neg_one = arena.int(-1);
        let one = arena.int(1);
        let x_list = vec![neg_one, zero, one];

        let weights = symplex::finite_diff::finite_diff_weights(arena, 1, &x_list, zero);

        let w = &weights[1][2];
        assert_eq!(w.len(), 3);

        let w0 = arena.display(w[0]).to_string();
        let w1 = arena.display(w[1]).to_string();
        let w2 = arena.display(w[2]).to_string();

        assert_eq!(w0, "-1/2", "w[0] should be -1/2, got {w0}");
        assert_eq!(w1, "0", "w[1] should be 0, got {w1}");
        assert_eq!(w2, "1/2", "w[2] should be 1/2, got {w2}");
    });
}

#[test]
fn central_diff_second_derivative_weights() {
    // Central difference: grid [-1, 0, 1], x0 = 0
    // 2nd derivative weights: [1, -2, 1]
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let zero = arena.int(0);
        let neg_one = arena.int(-1);
        let one = arena.int(1);
        let x_list = vec![neg_one, zero, one];

        let weights = symplex::finite_diff::finite_diff_weights(arena, 2, &x_list, zero);

        let w = &weights[2][2];
        assert_eq!(w.len(), 3);

        let w0 = arena.display(w[0]).to_string();
        let w1 = arena.display(w[1]).to_string();
        let w2 = arena.display(w[2]).to_string();

        assert_eq!(w0, "1", "w[0] should be 1, got {w0}");
        assert_eq!(w1, "-2", "w[1] should be -2, got {w1}");
        assert_eq!(w2, "1", "w[2] should be 1, got {w2}");
    });
}

#[test]
fn apply_finite_diff_quadratic_first_deriv_at_zero() {
    // f(x) = x^2, grid [-1, 0, 1], 1st derivative at 0
    // f(-1)=1, f(0)=0, f(1)=1
    // weights: [-1/2, 0, 1/2]
    // result = -1/2*1 + 0*0 + 1/2*1 = 0
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let zero = arena.int(0);
        let neg_one = arena.int(-1);
        let one = arena.int(1);

        let x_list = vec![neg_one, zero, one];
        let y_list = vec![one, zero, one]; // f(-1)=1, f(0)=0, f(1)=1

        let result = symplex::finite_diff::apply_finite_diff(arena, 1, &x_list, &y_list, zero);
        let result_eval = arena.eval_expr(result);
        assert!(
            arena.is_zero_structural(result_eval),
            "derivative of x^2 at 0 should be 0, got {}",
            arena.display(result_eval)
        );
    });
}

#[test]
fn apply_finite_diff_quadratic_second_deriv() {
    // f(x) = x^2, grid [-1, 0, 1], 2nd derivative at 0
    // f(-1)=1, f(0)=0, f(1)=1
    // weights: [1, -2, 1] → 1*1 + (-2)*0 + 1*1 = 2
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let zero = arena.int(0);
        let neg_one = arena.int(-1);
        let one = arena.int(1);

        let x_list = vec![neg_one, zero, one];
        let y_list = vec![one, zero, one];

        let result = symplex::finite_diff::apply_finite_diff(arena, 2, &x_list, &y_list, zero);
        let result_eval = arena.eval_expr(result);
        let two = arena.int(2);
        assert_eq!(
            result_eval,
            two,
            "2nd derivative of x^2 should be 2, got {}",
            arena.display(result_eval)
        );
    });
}

#[test]
fn apply_finite_diff_linear_first_deriv() {
    // f(x) = 3x + 1, grid = [0, 1], 1st derivative at x=0
    // weights: [-1, 1], y = [1, 4]
    // result = -1*1 + 1*4 = 3
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let zero = arena.int(0);
        let one = arena.int(1);
        let y0 = arena.int(1); // f(0) = 1
        let y1 = arena.int(4); // f(1) = 4

        let x_list = vec![zero, one];
        let y_list = vec![y0, y1];

        let result = symplex::finite_diff::apply_finite_diff(arena, 1, &x_list, &y_list, zero);
        let result_eval = arena.eval_expr(result);
        let three = arena.int(3);
        assert_eq!(
            result_eval,
            three,
            "derivative of 3x+1 should be 3, got {}",
            arena.display(result_eval)
        );
    });
}

#[test]
fn equispaced_grid_correct_count() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let zero = arena.int(0);
        let h = arena.symbol("h");
        let grid = symplex::finite_diff::equispaced_grid(arena, zero, h, 2);
        // half_width=2 → 5 points: [-2h, -h, 0, h, 2h]
        assert_eq!(
            grid.len(),
            5,
            "equispaced grid with half_width=2 should have 5 points"
        );
    });
}

#[test]
fn equispaced_grid_single_point() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let zero = arena.int(0);
        let h = arena.symbol("h");
        let grid = symplex::finite_diff::equispaced_grid(arena, zero, h, 0);
        // half_width=0 → 1 point (just x0)
        assert_eq!(
            grid.len(),
            1,
            "equispaced grid with half_width=0 should have 1 point"
        );
    });
}

#[test]
fn four_point_forward_first_deriv_weights() {
    // Forward-biased 4-point stencil: grid [0, 1, 2, 3], x0 = 0
    // Known 1st derivative weights: [-11/6, 3, -3/2, 1/3]
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let zero = arena.int(0);
        let one = arena.int(1);
        let two = arena.int(2);
        let three = arena.int(3);
        let x_list = vec![zero, one, two, three];

        let weights = symplex::finite_diff::finite_diff_weights(arena, 1, &x_list, zero);
        let w = &weights[1][3]; // 1st derivative, all 4 points

        assert_eq!(w.len(), 4);

        let w0 = arena.display(w[0]).to_string();
        let w1 = arena.display(w[1]).to_string();
        let w2 = arena.display(w[2]).to_string();
        let w3 = arena.display(w[3]).to_string();

        assert_eq!(w0, "-11/6", "w[0] should be -11/6, got {w0}");
        assert_eq!(w1, "3", "w[1] should be 3, got {w1}");
        assert_eq!(w2, "-3/2", "w[2] should be -3/2, got {w2}");
        assert_eq!(w3, "1/3", "w[3] should be 1/3, got {w3}");
    });
}
