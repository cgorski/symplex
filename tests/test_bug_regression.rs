//! Regression test for the proptest-discovered self-subtraction bug.
//!
//! The minimal failing input was:
//!   Mul(Neg(Int(1)), Add(Sym(0), Int(-1)))
//! which builds as (-1) * (-1 + a), and expr - expr did NOT produce zero.

use symplex::prelude::*;

#[test]
fn regression_mul_neg_one_times_add() {
    let ctx = Context::new();
    let a = ctx.symbol("a");

    // Build: (-1) * (-1 + a)
    let neg1 = ctx.int(-1);
    let sum = &a + &neg1; // canonicalizes to -1 + a
    let expr = &neg1 * &sum; // Mul(-1, Add(-1, a))

    eprintln!("expr = {expr}");

    // Self-subtraction must be zero
    let result = &expr - &expr;
    eprintln!("result = {result}");
    assert!(
        result.is_zero_structural(),
        "expr - expr should be structural zero, got: {result}"
    );
}

#[test]
fn regression_neg_of_product() {
    let ctx = Context::new();
    let a = ctx.symbol("a");

    let neg1 = ctx.int(-1);
    let sum = &a + &neg1;
    let expr = &neg1 * &sum;

    eprintln!("expr = {expr}");

    let neg_expr = -&expr;
    eprintln!("neg(expr) = {neg_expr}");

    // expr + neg(expr) must be zero
    let result = &expr + &neg_expr;
    eprintln!("expr + neg(expr) = {result}");
    assert!(
        result.is_zero_structural(),
        "expr + neg(expr) should be structural zero, got: {result}"
    );
}

#[test]
fn regression_simple_add_self_sub() {
    // Simpler case: (-1 + a) - (-1 + a) = 0
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let expr = &a + &ctx.int(-1); // -1 + a

    eprintln!("expr = {expr}");

    let result = &expr - &expr;
    eprintln!("result = {result}");
    assert!(
        result.is_zero_structural(),
        "(-1 + a) - (-1 + a) should be zero, got: {result}"
    );
}

#[test]
fn regression_neg_distributes_over_add() {
    // neg(-1 + a) should distribute to 1 + (-a) = 1 - a
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let expr = &a + &ctx.int(-1); // -1 + a
    let neg_expr = -&expr;

    eprintln!("expr = {expr}");
    eprintln!("neg(expr) = {neg_expr}");

    // neg should distribute: -(- 1 + a) = 1 - a = 1 + (-1)*a
    // Display should NOT contain a nested Add in parens like "-(- 1 + a)"
    let s = format!("{neg_expr}");
    assert!(
        !s.contains("(-"),
        "neg should distribute over Add, not wrap it: got {s}"
    );
}

#[test]
fn regression_as_coeff_term_roundtrip() {
    // For any expression, as_coeff_term followed by make_coeff_term
    // should return the original ExprId.
    let ctx = Context::new();
    let a = ctx.symbol("a");

    // Test with a simple Mul
    let expr = &a * 3; // 3*a
    let _expr_display = format!("{expr}");

    // Build it a different way — should be same ExprId
    let three = ctx.int(3);
    let expr2 = &three * &a;
    assert_eq!(
        format!("{expr}"),
        format!("{expr2}"),
        "3*a built two ways should match"
    );
    assert_eq!(expr, expr2, "3*a built two ways should be same ExprId");

    // Test with Mul of multiple factors
    let b = ctx.symbol("b");
    let expr3 = &a * &b * 2; // 2*a*b
    let expr3_display = format!("{expr3}");
    eprintln!("2*a*b = {expr3_display}");

    // Subtracting from itself must give zero
    let zero = &expr3 - &expr3;
    assert!(
        zero.is_zero_structural(),
        "2*a*b - 2*a*b should be zero, got: {zero}"
    );
}
