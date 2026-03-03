//! Stage 5 integration tests for symplex.
//!
//! Tests structural substitution through the public `Ex` API:
//! `ex.subs(&old, &new)` and `ex.subs_map(&[...])`.

use symplex::prelude::*;
use symplex::syms;

// ─── Basic substitution ───────────────────────────────────────────────────

#[test]
fn subs_symbol_for_integer() {
    let ctx = Context::new();
    syms!(ctx; x);
    let expr = &x + 1;
    let result = expr.subs(&x, &ctx.int(3));
    assert_eq!(format!("{result}"), "4");
}

#[test]
fn subs_symbol_for_symbol() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let expr = &x + 1;
    let result = expr.subs(&x, &y);
    assert_eq!(format!("{result}"), "1 + y");
}

#[test]
fn subs_in_product() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let expr = &x * 2;
    let result = expr.subs(&x, &y);
    assert_eq!(format!("{result}"), "2*y");
}

#[test]
fn subs_in_power() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2);
    let result = expr.subs(&x, &ctx.int(3));
    assert_eq!(format!("{result}"), "9");
}

#[test]
fn subs_in_sin() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let expr = x.sin();
    let result = expr.subs(&x, &y);
    assert_eq!(format!("{result}"), "sin(y)");
}

#[test]
fn subs_in_cos() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let expr = x.cos();
    let result = expr.subs(&x, &y);
    assert_eq!(format!("{result}"), "cos(y)");
}

#[test]
fn subs_in_exp() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let expr = x.exp();
    let result = expr.subs(&x, &y);
    assert_eq!(format!("{result}"), "exp(y)");
}

#[test]
fn subs_in_ln() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let expr = x.ln();
    let result = expr.subs(&x, &y);
    assert_eq!(format!("{result}"), "ln(y)");
}

#[test]
fn subs_in_sqrt() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let expr = x.sqrt();
    let result = expr.subs(&x, &y);
    assert_eq!(format!("{result}"), "sqrt(y)");
}

#[test]
fn subs_in_abs() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let expr = x.abs();
    let result = expr.subs(&x, &y);
    assert_eq!(format!("{result}"), "abs(y)");
}

// ─── Polynomial substitution ──────────────────────────────────────────────

#[test]
fn subs_polynomial_evaluates() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x^2 + 2*x + 1, substitute x → 3
    let expr = &x.powi(2) + &x * 2 + 1;
    let result = expr.subs(&x, &ctx.int(3));
    // 9 + 6 + 1 = 16
    assert_eq!(format!("{result}"), "16");
}

#[test]
fn subs_polynomial_symbol_for_symbol() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let expr = &x.powi(2) + &x * 2 + 1;
    let result = expr.subs(&x, &y);
    assert_eq!(format!("{result}"), "1 + y^2 + 2*y");
}

#[test]
fn subs_into_zero() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let expr = &x * &y;
    let zero = ctx.int(0);
    let result = expr.subs(&y, &zero);
    assert!(
        result.is_zero_structural(),
        "x*0 should be zero, got: {result}"
    );
}

// ─── No-match cases ──────────────────────────────────────────────────────

#[test]
fn subs_no_match_returns_equal() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let expr = &x + 1;
    let result = expr.subs(&y, &ctx.int(99));
    assert_eq!(result, expr, "no match should return equal expression");
}

#[test]
fn subs_old_equals_new_is_noop() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x + 1;
    let result = expr.subs(&x, &x);
    assert_eq!(result, expr);
}

// ─── Structural correctness ──────────────────────────────────────────────

#[test]
fn subs_structural_does_not_match_algebraic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (1/x).subs(x^2, 1) should return 1/x unchanged.
    // Build actual x^(-1):
    let x_inv = x.powi(-1);
    let x_sq = x.powi(2);
    let one = ctx.int(1);
    let result = x_inv.subs(&x_sq, &one);
    // x^(-1) does not structurally contain x^2, so no change
    assert_eq!(
        format!("{result}"),
        format!("{x_inv}"),
        "structural subs should not match x^2 inside x^(-1)"
    );
}

#[test]
fn subs_replaces_all_occurrences() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    // x + x + x = 3*x, substitute x → y
    let expr = &x + &x + &x;
    let result = expr.subs(&x, &y);
    assert_eq!(format!("{result}"), "3*y");
}

// ─── Nested substitution ─────────────────────────────────────────────────

#[test]
fn subs_in_nested_function() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let expr = x.powi(2).sin();
    let result = expr.subs(&x, &y);
    assert_eq!(format!("{result}"), "sin(y^2)");
}

#[test]
fn subs_in_deeply_nested() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    // sin(cos(x^2))
    let expr = x.powi(2).cos().sin();
    let result = expr.subs(&x, &y);
    assert_eq!(format!("{result}"), "sin(cos(y^2))");
}

#[test]
fn subs_in_product_of_functions() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let expr = &x.sin() * &x.cos();
    let result = expr.subs(&x, &y);
    assert_eq!(format!("{result}"), "sin(y)*cos(y)");
}

// ─── Substitution with expressions ────────────────────────────────────────

#[test]
fn subs_symbol_for_expression() {
    let ctx = Context::new();
    syms!(ctx; x, y, z);
    let expr = x.powi(2);
    let replacement = &y + &z;
    let result = expr.subs(&x, &replacement);
    // (y + z)^2
    assert_eq!(format!("{result}"), "(y + z)^2");
}

#[test]
fn subs_subexpression_for_symbol() {
    let ctx = Context::new();
    syms!(ctx; x, y, w);
    let inner = &x + &y;
    let expr = inner.powi(2);
    let result = expr.subs(&inner, &w);
    assert_eq!(format!("{result}"), "w^2");
}

// ─── Simultaneous substitution ────────────────────────────────────────────

#[test]
fn subs_map_swap_symbols() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    // (x + y).subs({x→y, y→x}) = y + x = x + y (commutative)
    let expr = &x + &y;
    let result = expr.subs_map(&[(&x, &y), (&y, &x)]);
    assert_eq!(
        result, expr,
        "swapping x↔y in x+y should give x+y (commutative)"
    );
}

#[test]
fn subs_map_multiple_values() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let expr = &x + &y;
    let result = expr.subs_map(&[(&x, &ctx.int(2)), (&y, &ctx.int(3))]);
    assert_eq!(format!("{result}"), "5");
}

#[test]
fn subs_map_empty_is_noop() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x + 1;
    let result = expr.subs_map(&[]);
    assert_eq!(result, expr);
}

#[test]
fn subs_map_partial_match() {
    let ctx = Context::new();
    syms!(ctx; x, y, z);
    let expr = &x + &y;
    // Only x is replaced, y stays
    let result = expr.subs_map(&[(&x, &z)]);
    assert_eq!(format!("{result}"), "y + z");
}

// ─── Substitution preserves canonical form ────────────────────────────────

#[test]
fn subs_result_is_canonical() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    // After subs, like terms should be collected
    let expr = &x + &x; // = 2*x
    let result = expr.subs(&x, &y); // should give 2*y
    let expected = &y * 2;
    assert_eq!(result, expected, "subs result should be canonical");
}

#[test]
fn subs_numeric_pow_evaluates() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(3);
    let result = expr.subs(&x, &ctx.int(2));
    assert_eq!(format!("{result}"), "8");
}

#[test]
fn subs_subtraction_to_zero() {
    let ctx = Context::new();
    syms!(ctx; x);
    let expr = &x - &x; // already 0
    assert!(expr.is_zero_structural());
    // subs on zero should stay zero
    let result = expr.subs(&x, &ctx.int(5));
    assert!(result.is_zero_structural());
}

// ─── Constant and special value substitution ──────────────────────────────

#[test]
fn subs_into_pi() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x + &ctx.pi();
    let result = expr.subs(&x, &ctx.int(1));
    assert_eq!(format!("{result}"), "1 + pi");
}

#[test]
fn subs_pi_stays_when_not_target() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let expr = &x + &ctx.pi();
    let result = expr.subs(&x, &y);
    assert_eq!(format!("{result}"), "y + pi");
}

// ─── Edge cases ───────────────────────────────────────────────────────────

#[test]
fn subs_atom_for_itself() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.subs(&x, &x);
    assert_eq!(result, x);
}

#[test]
fn subs_in_negative_expression() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let expr = -&x;
    let result = expr.subs(&x, &y);
    assert_eq!(format!("{result}"), "-y");
}

#[test]
fn subs_in_division() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let expr = &x / &y;
    let result = expr.subs(&y, &ctx.int(2));
    assert_eq!(format!("{result}"), "1/2*x");
}

#[test]
fn subs_nan_stays_nan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let nan = ctx.nan();
    let result = nan.subs(&x, &ctx.int(1));
    assert_eq!(format!("{result}"), "nan");
}
