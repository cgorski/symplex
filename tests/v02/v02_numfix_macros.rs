//! Regression tests for `expr!` with purely numeric (sub)expressions
//! (0.2 numfix): integer literals lowered to `i64`, so `expr!(ctx, 2^10)`
//! did not compile (`i64` has no `powi`) and `expr!(ctx, 2 + 3)` was an
//! `i64`, not an `Ex`.  Numeric-only operations now promote their left
//! operand to an `Ex` and a purely numeric macro yields an `Ex`.

use symplex::expr;
use symplex::prelude::*;

#[test]
fn expr_macro_numeric_power() {
    let ctx = Context::new();
    let e: Ex = expr!(ctx, 2 ^ 10);
    assert_eq!(e, ctx.int(1024));
    let e: Ex = expr!(ctx, 2 ^ -3);
    assert_eq!(e, ctx.rational(1, 8));
    let e: Ex = expr!(ctx, 2 ^ (3 + 1));
    assert_eq!(e, ctx.int(16));
}

#[test]
fn expr_macro_numeric_arithmetic_is_an_ex() {
    let ctx = Context::new();
    let e: Ex = expr!(ctx, 2 + 3);
    assert_eq!(e, ctx.int(5));
    let e: Ex = expr!(ctx, (2 * 3) - 1);
    assert_eq!(e, ctx.int(5));
    let e: Ex = expr!(ctx, -(2 * 3));
    assert_eq!(e, ctx.int(-6));
    let e: Ex = expr!(ctx, 7);
    assert_eq!(e, ctx.int(7));
    let e: Ex = expr!(ctx, -7);
    assert_eq!(e, ctx.int(-7));
}

#[test]
fn expr_macro_numeric_rationals_stay_exact() {
    let ctx = Context::new();
    assert_eq!(expr!(ctx, 1 / 2), ctx.rational(1, 2));
    assert_eq!(expr!(ctx, -1 / 2), ctx.rational(-1, 2));
    let e: Ex = expr!(ctx, (1 + 2) / 4);
    assert_eq!(e, ctx.rational(3, 4));
    let e: Ex = expr!(ctx, 2 ^ 10 / 3);
    assert_eq!(e, ctx.rational(1024, 3));
}

#[test]
fn expr_macro_numeric_subexpressions_inside_symbolic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = expr!(ctx, 2 ^ 10 * x + 3 * 4);
    assert_eq!(e, &x * 1024 + 12);
    let e = expr!(ctx, x ^ (2 + 1));
    assert_eq!(e, x.powi(3));
    let e = expr!(ctx, sin(2 ^ 3 * x));
    assert_eq!(e, (&x * 8).sin());
}

#[test]
fn expr_macro_symbolic_forms_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    assert_eq!(
        expr!(ctx, 3 * x ^ 2 + 2 * x + 1),
        &x.powi(2) * 3 + &x * 2 + 1
    );
    assert_eq!(expr!(ctx, -x), -&x);
    assert_eq!(expr!(ctx, x / y), &x / &y);
    assert_eq!(expr!(ctx, 2 * pi), ctx.pi() * 2);
    assert_eq!(expr!(ctx, x ^ -1), x.powi(-1));
}
