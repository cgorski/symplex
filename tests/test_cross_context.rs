//! Tests that cross-context expression mixing is caught at runtime.
//!
//! After the `Expr.id` field was made private, every cross-expression
//! `ExprId` access goes through `checked_id()`, which panics when
//! contexts differ.  These tests prove that the guard fires for
//! every major category of operation.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Arithmetic operators
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[should_panic(expected = "cannot combine expressions from different contexts")]
fn cross_context_add_panics() {
    let ctx_a = Context::new();
    let ctx_b = Context::new();
    let x = ctx_a.symbol("x");
    let y = ctx_b.symbol("y");
    let _ = &x + &y;
}

#[test]
#[should_panic(expected = "cannot combine expressions from different contexts")]
fn cross_context_sub_panics() {
    let ctx_a = Context::new();
    let ctx_b = Context::new();
    let x = ctx_a.symbol("x");
    let y = ctx_b.symbol("y");
    let _ = &x - &y;
}

#[test]
#[should_panic(expected = "cannot combine expressions from different contexts")]
fn cross_context_mul_panics() {
    let ctx_a = Context::new();
    let ctx_b = Context::new();
    let x = ctx_a.symbol("x");
    let y = ctx_b.symbol("y");
    let _ = &x * &y;
}

#[test]
#[should_panic(expected = "cannot combine expressions from different contexts")]
fn cross_context_div_panics() {
    let ctx_a = Context::new();
    let ctx_b = Context::new();
    let x = ctx_a.symbol("x");
    let y = ctx_b.symbol("y");
    let _ = &x / &y;
}

// ═══════════════════════════════════════════════════════════════════════════
// Math functions taking a second expression
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[should_panic(expected = "cannot combine expressions from different contexts")]
fn cross_context_pow_panics() {
    let ctx_a = Context::new();
    let ctx_b = Context::new();
    let x = ctx_a.symbol("x");
    let y = ctx_b.symbol("y");
    let _ = x.pow(&y);
}

// ═══════════════════════════════════════════════════════════════════════════
// Substitution
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[should_panic(expected = "cannot combine expressions from different contexts")]
fn cross_context_subs_old_panics() {
    let ctx_a = Context::new();
    let ctx_b = Context::new();
    let x = ctx_a.symbol("x");
    let val = ctx_b.int(5);
    let _ = x.subs(&x, &val);
}

#[test]
#[should_panic(expected = "cannot combine expressions from different contexts")]
fn cross_context_subs_new_panics() {
    let ctx_a = Context::new();
    let ctx_b = Context::new();
    let x = ctx_a.symbol("x");
    let five = ctx_b.int(5);
    let _ = x.subs(&x, &five);
}

#[test]
#[should_panic(expected = "cannot combine expressions from different contexts")]
fn cross_context_subs_i64_panics() {
    let ctx_a = Context::new();
    let ctx_b = Context::new();
    let x = ctx_a.symbol("x");
    let expr = ctx_b.symbol("x").powi(2);
    let _ = expr.subs_i64(&x, 3);
}

// ═══════════════════════════════════════════════════════════════════════════
// Calculus
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[should_panic(expected = "cannot combine expressions from different contexts")]
fn cross_context_diff_panics() {
    let ctx_a = Context::new();
    let ctx_b = Context::new();
    let expr = ctx_a.symbol("x").powi(2);
    let var = ctx_b.symbol("x");
    let _ = expr.diff(&var);
}

#[test]
#[should_panic(expected = "cannot combine expressions from different contexts")]
fn cross_context_integrate_panics() {
    let ctx_a = Context::new();
    let ctx_b = Context::new();
    let expr = ctx_a.symbol("x").powi(2);
    let var = ctx_b.symbol("x");
    let _ = expr.integrate(&var);
}

// ═══════════════════════════════════════════════════════════════════════════
// Solving
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[should_panic(expected = "cannot combine expressions from different contexts")]
fn cross_context_solve_panics() {
    let ctx_a = Context::new();
    let ctx_b = Context::new();
    let expr = ctx_a.symbol("x").powi(2) - ctx_a.int(1);
    let var = ctx_b.symbol("x");
    let _ = expr.solve(&var);
}

// ═══════════════════════════════════════════════════════════════════════════
// Contains
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[should_panic(expected = "cannot combine expressions from different contexts")]
fn cross_context_contains_panics() {
    let ctx_a = Context::new();
    let ctx_b = Context::new();
    let expr = ctx_a.symbol("x").powi(2);
    let needle = ctx_b.symbol("x");
    let _ = expr.contains(&needle);
}

// ═══════════════════════════════════════════════════════════════════════════
// Boolean operations
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[should_panic(expected = "cannot combine expressions from different contexts")]
fn cross_context_boolean_and_panics() {
    let ctx_a = Context::new();
    let ctx_b = Context::new();
    let a = ctx_a.symbol("x").gt(&ctx_a.int(0));
    let b = ctx_b.symbol("y").gt(&ctx_b.int(0));
    let _ = a.and(&b);
}

// ═══════════════════════════════════════════════════════════════════════════
// Set operations
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[should_panic(expected = "cannot combine expressions from different contexts")]
fn cross_context_set_union_panics() {
    let ctx_a = Context::new();
    let ctx_b = Context::new();
    let a = ctx_a.interval(&ctx_a.int(0), &ctx_a.int(1), false, false);
    let b = ctx_b.interval(&ctx_b.int(2), &ctx_b.int(3), false, false);
    let _ = a.union(&b);
}

// ═══════════════════════════════════════════════════════════════════════════
// Positive tests: same-context operations MUST still work
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn same_context_operations_work() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let one = ctx.int(1);

    // Arithmetic
    let _ = &x + &y;
    let _ = &x - &y;
    let _ = &x * &y;
    let _ = &x / &y;

    // Functions
    let _ = x.pow(&y);
    let _ = x.powi(2);
    let _ = x.sin();

    // Substitution
    let expr = x.powi(2) + &one;
    let _ = expr.subs(&x, &y);
    let _ = expr.subs_i64(&x, 3);

    // Calculus
    let _ = expr.diff(&x);
    let _ = expr.integrate(&x);

    // Solving
    let _ = expr.solve(&x);

    // Contains
    let _ = expr.contains(&x);
}

#[test]
fn default_context_operations_work() {
    // All expressions from the global default context should interoperate
    let x = symplex::var("x");
    let y = symplex::var("y");
    let one = symplex::int(1);

    let expr = x.powi(2) + &y + &one;
    let _ = expr.diff(&x);
    let _ = expr.subs(&x, &y);
    let _ = expr.integrate(&x);
}

#[test]
fn context_method_returns_usable_context() {
    // expr.context() should return a Context that creates compatible expressions
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2);

    let same_ctx = expr.context();
    let pt = same_ctx.rational(3, 10);

    // This must NOT panic — pt is in the same context as expr
    let _ = expr.subs(&x, &pt);
}
