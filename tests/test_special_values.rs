//! Comprehensive tests for special-value canonicalization in symplex.
//!
//! These tests cover ComplexInfinity (zoo), regular infinities (oo, -oo),
//! NaN propagation, and indeterminate power forms. They verify the recent
//! canonicalization fixes produce correct results for every combination
//! of special values in addition and exponentiation.

use symplex::prelude::*;
use symplex::tree::ExprTree;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Build a ComplexInfinity (`zoo`) expression in the given context.
///
/// There is no public convenience method on `Context` for zoo, so we
/// round-trip through the serializable `ExprTree` representation.
fn zoo(ctx: &Context) -> Ex {
    ctx.from_tree(&ExprTree::ComplexInfinity)
}

/// Build a NaN expression in the given context.
fn nan(ctx: &Context) -> Ex {
    ctx.nan()
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. ComplexInfinity (zoo) in addition
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn zoo_plus_finite_integer_is_zoo() {
    let ctx = Context::new();
    let z = zoo(&ctx);
    let five = ctx.int(5);
    let result = &z + &five;
    assert_eq!(
        format!("{result}"),
        "zoo",
        "zoo + 5 should be zoo, got: {result}"
    );
}

#[test]
fn zoo_plus_zero_is_zoo() {
    let ctx = Context::new();
    let z = zoo(&ctx);
    let zero = ctx.int(0);
    let result = &z + &zero;
    assert_eq!(
        format!("{result}"),
        "zoo",
        "zoo + 0 should be zoo, got: {result}"
    );
}

#[test]
fn zoo_plus_negative_integer_is_zoo() {
    let ctx = Context::new();
    let z = zoo(&ctx);
    let neg3 = ctx.int(-3);
    let result = &z + &neg3;
    assert_eq!(
        format!("{result}"),
        "zoo",
        "zoo + (-3) should be zoo, got: {result}"
    );
}

#[test]
fn zoo_plus_rational_is_zoo() {
    let ctx = Context::new();
    let z = zoo(&ctx);
    let half = ctx.rational(1, 2);
    let result = &z + &half;
    assert_eq!(
        format!("{result}"),
        "zoo",
        "zoo + 1/2 should be zoo, got: {result}"
    );
}

#[test]
fn finite_integer_plus_zoo_is_zoo() {
    let ctx = Context::new();
    let z = zoo(&ctx);
    let seven = ctx.int(7);
    let result = &seven + &z;
    assert_eq!(
        format!("{result}"),
        "zoo",
        "7 + zoo should be zoo, got: {result}"
    );
}

#[test]
fn zoo_plus_symbol_is_zoo() {
    let ctx = Context::new();
    let z = zoo(&ctx);
    let x = ctx.symbol("x");
    let result = &z + &x;
    assert_eq!(
        format!("{result}"),
        "zoo",
        "zoo + x should be zoo (zoo dominates finite symbolic terms), got: {result}"
    );
}

#[test]
fn symbol_plus_zoo_is_zoo() {
    let ctx = Context::new();
    let z = zoo(&ctx);
    let x = ctx.symbol("x");
    let result = &x + &z;
    assert_eq!(
        format!("{result}"),
        "zoo",
        "x + zoo should be zoo (zoo dominates finite symbolic terms), got: {result}"
    );
}

#[test]
fn zoo_plus_expression_is_zoo() {
    let ctx = Context::new();
    let z = zoo(&ctx);
    let x = ctx.symbol("x");
    let expr = &x.powi(2) + &ctx.int(1); // x^2 + 1
    let result = &z + &expr;
    assert_eq!(
        format!("{result}"),
        "zoo",
        "zoo + (x^2 + 1) should be zoo, got: {result}"
    );
}

#[test]
fn zoo_plus_zoo_is_nan() {
    let ctx = Context::new();
    let z1 = zoo(&ctx);
    let z2 = zoo(&ctx);
    let result = &z1 + &z2;
    assert_eq!(
        format!("{result}"),
        "nan",
        "zoo + zoo should be nan (indeterminate), got: {result}"
    );
}

#[test]
fn zoo_plus_positive_infinity_is_nan() {
    let ctx = Context::new();
    let z = zoo(&ctx);
    let inf = ctx.infinity();
    let result = &z + &inf;
    assert_eq!(
        format!("{result}"),
        "nan",
        "zoo + oo should be nan (indeterminate), got: {result}"
    );
}

#[test]
fn zoo_plus_negative_infinity_is_nan() {
    let ctx = Context::new();
    let z = zoo(&ctx);
    let neg_inf = ctx.neg_infinity();
    let result = &z + &neg_inf;
    assert_eq!(
        format!("{result}"),
        "nan",
        "zoo + (-oo) should be nan (indeterminate), got: {result}"
    );
}

#[test]
fn positive_infinity_plus_zoo_is_nan() {
    let ctx = Context::new();
    let z = zoo(&ctx);
    let inf = ctx.infinity();
    let result = &inf + &z;
    assert_eq!(
        format!("{result}"),
        "nan",
        "oo + zoo should be nan (indeterminate), got: {result}"
    );
}

#[test]
fn negative_infinity_plus_zoo_is_nan() {
    let ctx = Context::new();
    let z = zoo(&ctx);
    let neg_inf = ctx.neg_infinity();
    let result = &neg_inf + &z;
    assert_eq!(
        format!("{result}"),
        "nan",
        "(-oo) + zoo should be nan (indeterminate), got: {result}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Regular infinity in addition (verify existing behavior preserved)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn positive_infinity_plus_positive_infinity_is_infinity() {
    let __ctx = Context::new();
    let inf1 = __ctx.infinity();
    let inf2 = __ctx.infinity();
    let result = &inf1 + &inf2;
    assert_eq!(
        format!("{result}"),
        "oo",
        "oo + oo should be oo, got: {result}"
    );
}

#[test]
fn negative_infinity_plus_negative_infinity_is_negative_infinity() {
    let __ctx = Context::new();
    let neg1 = __ctx.neg_infinity();
    let neg2 = __ctx.neg_infinity();
    let result = &neg1 + &neg2;
    assert_eq!(
        format!("{result}"),
        "-oo",
        "(-oo) + (-oo) should be -oo, got: {result}"
    );
}

#[test]
fn positive_infinity_plus_negative_infinity_is_nan() {
    let __ctx = Context::new();
    let inf = __ctx.infinity();
    let neg_inf = __ctx.neg_infinity();
    let result = &inf + &neg_inf;
    assert_eq!(
        format!("{result}"),
        "nan",
        "oo + (-oo) should be nan (indeterminate), got: {result}"
    );
}

#[test]
fn negative_infinity_plus_positive_infinity_is_nan() {
    let __ctx = Context::new();
    let neg_inf = __ctx.neg_infinity();
    let inf = __ctx.infinity();
    let result = &neg_inf + &inf;
    assert_eq!(
        format!("{result}"),
        "nan",
        "(-oo) + oo should be nan (indeterminate), got: {result}"
    );
}

#[test]
fn positive_infinity_plus_finite_is_infinity() {
    let __ctx = Context::new();
    let inf = __ctx.infinity();
    let ten = __ctx.int(10);
    let result = &inf + &ten;
    assert_eq!(
        format!("{result}"),
        "oo",
        "oo + 10 should be oo, got: {result}"
    );
}

#[test]
fn finite_plus_positive_infinity_is_infinity() {
    let __ctx = Context::new();
    let inf = __ctx.infinity();
    let ten = __ctx.int(10);
    let result = &ten + &inf;
    assert_eq!(
        format!("{result}"),
        "oo",
        "10 + oo should be oo, got: {result}"
    );
}

#[test]
fn negative_infinity_plus_finite_is_negative_infinity() {
    let __ctx = Context::new();
    let neg_inf = __ctx.neg_infinity();
    let ten = __ctx.int(10);
    let result = &neg_inf + &ten;
    assert_eq!(
        format!("{result}"),
        "-oo",
        "(-oo) + 10 should be -oo, got: {result}"
    );
}

#[test]
fn finite_plus_negative_infinity_is_negative_infinity() {
    let __ctx = Context::new();
    let neg_inf = __ctx.neg_infinity();
    let ten = __ctx.int(10);
    let result = &ten + &neg_inf;
    assert_eq!(
        format!("{result}"),
        "-oo",
        "10 + (-oo) should be -oo, got: {result}"
    );
}

#[test]
fn positive_infinity_plus_symbol_is_infinity() {
    let __ctx = Context::new();
    let inf = __ctx.infinity();
    let x = __ctx.symbol("x");
    let result = &inf + &x;
    assert_eq!(
        format!("{result}"),
        "oo",
        "oo + x should be oo (infinity dominates finite terms), got: {result}"
    );
}

#[test]
fn negative_infinity_plus_symbol_is_negative_infinity() {
    let __ctx = Context::new();
    let neg_inf = __ctx.neg_infinity();
    let x = __ctx.symbol("x");
    let result = &neg_inf + &x;
    assert_eq!(
        format!("{result}"),
        "-oo",
        "(-oo) + x should be -oo (neg-infinity dominates finite terms), got: {result}"
    );
}

#[test]
fn positive_infinity_plus_zero_is_infinity() {
    let __ctx = Context::new();
    let inf = __ctx.infinity();
    let zero = __ctx.int(0);
    let result = &inf + &zero;
    assert_eq!(
        format!("{result}"),
        "oo",
        "oo + 0 should be oo, got: {result}"
    );
}

#[test]
fn negative_infinity_plus_zero_is_negative_infinity() {
    let __ctx = Context::new();
    let neg_inf = __ctx.neg_infinity();
    let zero = __ctx.int(0);
    let result = &neg_inf + &zero;
    assert_eq!(
        format!("{result}"),
        "-oo",
        "(-oo) + 0 should be -oo, got: {result}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Indeterminate power forms
// ═══════════════════════════════════════════════════════════════════════════

// ── Infinity^0 forms (all indeterminate → NaN) ────────────────────────

#[test]
fn positive_infinity_to_the_zero_is_nan() {
    let __ctx = Context::new();
    let inf = __ctx.infinity();
    let result = inf.powi(0);
    assert_eq!(
        format!("{result}"),
        "nan",
        "oo^0 should be nan (indeterminate form), got: {result}"
    );
}

#[test]
fn negative_infinity_to_the_zero_is_nan() {
    let __ctx = Context::new();
    let neg_inf = __ctx.neg_infinity();
    let result = neg_inf.powi(0);
    assert_eq!(
        format!("{result}"),
        "nan",
        "(-oo)^0 should be nan (indeterminate form), got: {result}"
    );
}

#[test]
fn zoo_to_the_zero_is_nan() {
    let ctx = Context::new();
    let z = zoo(&ctx);
    let result = z.powi(0);
    assert_eq!(
        format!("{result}"),
        "nan",
        "zoo^0 should be nan (indeterminate form), got: {result}"
    );
}

// ── 0^0 convention ────────────────────────────────────────────────────

#[test]
fn zero_to_the_zero_is_one() {
    let __ctx = Context::new();
    let zero = __ctx.int(0);
    let result = zero.powi(0);
    assert_eq!(
        format!("{result}"),
        "1",
        "0^0 should be 1 (by convention), got: {result}"
    );
}

// ── Ordinary x^0 = 1 ─────────────────────────────────────────────────

#[test]
fn symbol_to_the_zero_is_one() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let result = x.powi(0);
    assert_eq!(
        format!("{result}"),
        "1",
        "x^0 should be 1 (for any symbol), got: {result}"
    );
}

#[test]
fn integer_to_the_zero_is_one() {
    let __ctx = Context::new();
    let five = __ctx.int(5);
    let result = five.powi(0);
    assert_eq!(format!("{result}"), "1", "5^0 should be 1, got: {result}");
}

#[test]
fn negative_integer_to_the_zero_is_one() {
    let __ctx = Context::new();
    let neg7 = __ctx.int(-7);
    let result = neg7.powi(0);
    assert_eq!(
        format!("{result}"),
        "1",
        "(-7)^0 should be 1, got: {result}"
    );
}

#[test]
fn rational_to_the_zero_is_one() {
    let ctx = Context::new();
    let r = ctx.rational(3, 7);
    let result = r.powi(0);
    assert_eq!(
        format!("{result}"),
        "1",
        "(3/7)^0 should be 1, got: {result}"
    );
}

#[test]
fn expression_to_the_zero_is_one() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let expr = &x.powi(2) + &__ctx.int(1); // x^2 + 1
    let result = expr.powi(0);
    assert_eq!(
        format!("{result}"),
        "1",
        "(x^2 + 1)^0 should be 1, got: {result}"
    );
}

#[test]
fn pi_to_the_zero_is_one() {
    let __ctx = Context::new();
    let p = __ctx.pi();
    let result = p.powi(0);
    assert_eq!(format!("{result}"), "1", "pi^0 should be 1, got: {result}");
}

// ── NaN^0 is NaN (NaN propagation beats the x^0 rule) ────────────────

#[test]
fn nan_to_the_zero_is_nan() {
    let ctx = Context::new();
    let n = nan(&ctx);
    let result = n.powi(0);
    assert_eq!(
        format!("{result}"),
        "nan",
        "nan^0 should be nan (NaN propagation), got: {result}"
    );
}

// ── 0^(positive) = 0 ─────────────────────────────────────────────────

#[test]
fn zero_to_the_positive_integer_is_zero() {
    let __ctx = Context::new();
    let zero = __ctx.int(0);
    let result = zero.powi(3);
    assert_eq!(format!("{result}"), "0", "0^3 should be 0, got: {result}");
}

#[test]
fn zero_to_the_one_is_zero() {
    let __ctx = Context::new();
    let zero = __ctx.int(0);
    let result = zero.powi(1);
    assert_eq!(format!("{result}"), "0", "0^1 should be 0, got: {result}");
}

#[test]
fn zero_to_large_positive_is_zero() {
    let __ctx = Context::new();
    let zero = __ctx.int(0);
    let result = zero.powi(100);
    assert_eq!(format!("{result}"), "0", "0^100 should be 0, got: {result}");
}

#[test]
fn zero_to_positive_rational_is_zero() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let half = ctx.rational(1, 2);
    let result = zero.pow(&half);
    assert_eq!(
        format!("{result}"),
        "0",
        "0^(1/2) should be 0, got: {result}"
    );
}

// ── oo^1 = oo ─────────────────────────────────────────────────────────

#[test]
fn positive_infinity_to_the_one_is_infinity() {
    let __ctx = Context::new();
    let inf = __ctx.infinity();
    let result = inf.powi(1);
    assert_eq!(
        format!("{result}"),
        "oo",
        "oo^1 should be oo, got: {result}"
    );
}

#[test]
fn negative_infinity_to_the_one_is_negative_infinity() {
    let __ctx = Context::new();
    let neg_inf = __ctx.neg_infinity();
    let result = neg_inf.powi(1);
    assert_eq!(
        format!("{result}"),
        "-oo",
        "(-oo)^1 should be -oo, got: {result}"
    );
}

// ── oo^2 stays as oo^2 (no further canonicalization) ──────────────────

#[test]
fn positive_infinity_to_the_two() {
    let __ctx = Context::new();
    let inf = __ctx.infinity();
    let result = inf.powi(2);
    let s = format!("{result}");
    // oo^2 is valid — it might stay as oo^2 or simplify to oo.
    // Both are mathematically acceptable; verify it's one of them.
    assert!(
        s == "oo^2" || s == "oo",
        "oo^2 should be oo^2 or oo, got: {s}"
    );
}

// ── NaN propagation in powers ─────────────────────────────────────────

#[test]
fn nan_to_the_one_is_nan() {
    let ctx = Context::new();
    let n = nan(&ctx);
    let result = n.powi(1);
    assert_eq!(
        format!("{result}"),
        "nan",
        "nan^1 should be nan, got: {result}"
    );
}

#[test]
fn nan_to_positive_integer_is_nan() {
    let ctx = Context::new();
    let n = nan(&ctx);
    let result = n.powi(5);
    assert_eq!(
        format!("{result}"),
        "nan",
        "nan^5 should be nan, got: {result}"
    );
}

#[test]
fn nan_to_negative_integer_is_nan() {
    let ctx = Context::new();
    let n = nan(&ctx);
    let result = n.powi(-2);
    assert_eq!(
        format!("{result}"),
        "nan",
        "nan^(-2) should be nan, got: {result}"
    );
}

#[test]
fn nan_to_symbol_is_nan() {
    let ctx = Context::new();
    let n = nan(&ctx);
    let x = ctx.symbol("x");
    let result = n.pow(&x);
    assert_eq!(
        format!("{result}"),
        "nan",
        "nan^x should be nan, got: {result}"
    );
}

#[test]
fn symbol_to_the_nan_is_nan() {
    let ctx = Context::new();
    let n = nan(&ctx);
    let x = ctx.symbol("x");
    let result = x.pow(&n);
    assert_eq!(
        format!("{result}"),
        "nan",
        "x^nan should be nan (NaN in exponent propagates), got: {result}"
    );
}

#[test]
fn integer_to_the_nan_is_nan() {
    let ctx = Context::new();
    let n = nan(&ctx);
    let five = ctx.int(5);
    let result = five.pow(&n);
    assert_eq!(
        format!("{result}"),
        "nan",
        "5^nan should be nan, got: {result}"
    );
}

#[test]
fn infinity_to_the_nan_is_nan() {
    let ctx = Context::new();
    let n = nan(&ctx);
    let inf = ctx.infinity();
    let result = inf.pow(&n);
    assert_eq!(
        format!("{result}"),
        "nan",
        "oo^nan should be nan, got: {result}"
    );
}

#[test]
fn nan_to_nan_is_nan() {
    let ctx = Context::new();
    let n1 = nan(&ctx);
    let n2 = nan(&ctx);
    let result = n1.pow(&n2);
    assert_eq!(
        format!("{result}"),
        "nan",
        "nan^nan should be nan, got: {result}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. NaN propagation in addition
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nan_plus_integer_is_nan() {
    let ctx = Context::new();
    let n = nan(&ctx);
    let five = ctx.int(5);
    let result = &n + &five;
    assert_eq!(
        format!("{result}"),
        "nan",
        "nan + 5 should be nan, got: {result}"
    );
}

#[test]
fn integer_plus_nan_is_nan() {
    let ctx = Context::new();
    let n = nan(&ctx);
    let five = ctx.int(5);
    let result = &five + &n;
    assert_eq!(
        format!("{result}"),
        "nan",
        "5 + nan should be nan, got: {result}"
    );
}

#[test]
fn nan_plus_symbol_is_nan() {
    let ctx = Context::new();
    let n = nan(&ctx);
    let x = ctx.symbol("x");
    let result = &n + &x;
    assert_eq!(
        format!("{result}"),
        "nan",
        "nan + x should be nan, got: {result}"
    );
}

#[test]
fn nan_plus_infinity_is_nan() {
    let ctx = Context::new();
    let n = nan(&ctx);
    let inf = ctx.infinity();
    let result = &n + &inf;
    assert_eq!(
        format!("{result}"),
        "nan",
        "nan + oo should be nan, got: {result}"
    );
}

#[test]
fn nan_plus_negative_infinity_is_nan() {
    let ctx = Context::new();
    let n = nan(&ctx);
    let neg_inf = ctx.neg_infinity();
    let result = &n + &neg_inf;
    assert_eq!(
        format!("{result}"),
        "nan",
        "nan + (-oo) should be nan, got: {result}"
    );
}

#[test]
fn nan_plus_zoo_is_nan() {
    let ctx = Context::new();
    let n = nan(&ctx);
    let z = zoo(&ctx);
    let result = &n + &z;
    assert_eq!(
        format!("{result}"),
        "nan",
        "nan + zoo should be nan, got: {result}"
    );
}

#[test]
fn nan_plus_nan_is_nan() {
    let ctx = Context::new();
    let n1 = nan(&ctx);
    let n2 = nan(&ctx);
    let result = &n1 + &n2;
    assert_eq!(
        format!("{result}"),
        "nan",
        "nan + nan should be nan, got: {result}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Negation of special values
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn negation_of_zoo_is_zoo() {
    let ctx = Context::new();
    let z = zoo(&ctx);
    let result = -&z;
    assert_eq!(
        format!("{result}"),
        "zoo",
        "-zoo should be zoo (complex infinity has no direction), got: {result}"
    );
}

#[test]
fn negation_of_infinity_is_negative_infinity() {
    let __ctx = Context::new();
    let inf = __ctx.infinity();
    let result = -&inf;
    assert_eq!(
        format!("{result}"),
        "-oo",
        "-oo should display as -oo, got: {result}"
    );
}

#[test]
fn negation_of_negative_infinity_is_infinity() {
    let __ctx = Context::new();
    let neg_inf = __ctx.neg_infinity();
    let result = -&neg_inf;
    assert_eq!(
        format!("{result}"),
        "oo",
        "-(-oo) should be oo, got: {result}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Subtraction involving special values (derived from addition + negation)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn infinity_minus_infinity_is_nan() {
    let __ctx = Context::new();
    let inf1 = __ctx.infinity();
    let inf2 = __ctx.infinity();
    let result = &inf1 - &inf2;
    assert_eq!(
        format!("{result}"),
        "nan",
        "oo - oo should be nan, got: {result}"
    );
}

#[test]
fn zoo_minus_zoo_is_nan() {
    let ctx = Context::new();
    let z1 = zoo(&ctx);
    let z2 = zoo(&ctx);
    let result = &z1 - &z2;
    assert_eq!(
        format!("{result}"),
        "nan",
        "zoo - zoo should be nan (indeterminate), got: {result}"
    );
}

#[test]
fn zoo_minus_finite_is_zoo() {
    let ctx = Context::new();
    let z = zoo(&ctx);
    let three = ctx.int(3);
    let result = &z - &three;
    assert_eq!(
        format!("{result}"),
        "zoo",
        "zoo - 3 should be zoo, got: {result}"
    );
}

#[test]
fn finite_minus_zoo_is_zoo() {
    let ctx = Context::new();
    let z = zoo(&ctx);
    let three = ctx.int(3);
    // 3 - zoo: negation of zoo is zoo, so 3 + zoo = zoo
    let result = &three - &z;
    assert_eq!(
        format!("{result}"),
        "zoo",
        "3 - zoo should be zoo (since -zoo = zoo), got: {result}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Display sanity checks for special atoms
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn display_positive_infinity() {
    let __ctx = Context::new();
    let inf = __ctx.infinity();
    assert_eq!(format!("{inf}"), "oo", "infinity should display as oo");
}

#[test]
fn display_negative_infinity() {
    let __ctx = Context::new();
    let neg_inf = __ctx.neg_infinity();
    assert_eq!(
        format!("{neg_inf}"),
        "-oo",
        "negative infinity should display as -oo"
    );
}

#[test]
fn display_complex_infinity() {
    let ctx = Context::new();
    let z = zoo(&ctx);
    assert_eq!(
        format!("{z}"),
        "zoo",
        "complex infinity should display as zoo"
    );
}

#[test]
fn display_nan() {
    let ctx = Context::new();
    let n = nan(&ctx);
    assert_eq!(format!("{n}"), "nan", "NaN should display as nan");
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Mixed three-operand sums with special values
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn zoo_plus_finite_plus_finite_is_zoo() {
    let ctx = Context::new();
    let z = zoo(&ctx);
    let a = ctx.int(2);
    let b = ctx.int(3);
    let result = &(&z + &a) + &b;
    assert_eq!(
        format!("{result}"),
        "zoo",
        "zoo + 2 + 3 should be zoo, got: {result}"
    );
}

#[test]
fn finite_plus_zoo_plus_finite_is_zoo() {
    let ctx = Context::new();
    let z = zoo(&ctx);
    let a = ctx.int(1);
    let b = ctx.int(4);
    let result = &(&a + &z) + &b;
    assert_eq!(
        format!("{result}"),
        "zoo",
        "1 + zoo + 4 should be zoo, got: {result}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. x^1 identity for special values
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn zoo_to_the_one_is_zoo() {
    let ctx = Context::new();
    let z = zoo(&ctx);
    let result = z.powi(1);
    assert_eq!(
        format!("{result}"),
        "zoo",
        "zoo^1 should be zoo, got: {result}"
    );
}

#[test]
fn nan_identity_through_power_of_one() {
    let ctx = Context::new();
    let n = nan(&ctx);
    let result = n.powi(1);
    assert_eq!(
        format!("{result}"),
        "nan",
        "nan^1 should be nan (NaN propagation), got: {result}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. 1^x for special exponents
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn one_to_the_infinity_is_one() {
    let __ctx = Context::new();
    // In symplex, 1^x → 1 unconditionally (the canon rule).
    let one = __ctx.int(1);
    let inf = __ctx.infinity();
    let result = one.pow(&inf);
    assert_eq!(format!("{result}"), "1", "1^oo should be 1, got: {result}");
}

#[test]
fn one_to_the_symbol_is_one() {
    let __ctx = Context::new();
    let one = __ctx.int(1);
    let x = __ctx.symbol("x");
    let result = one.pow(&x);
    assert_eq!(format!("{result}"), "1", "1^x should be 1, got: {result}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. Edge: combining special values from the global default context
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn global_context_infinity_operations() {
    let __ctx = Context::new();
    // Verify the free-standing convenience functions work correctly
    // with special value arithmetic.
    let inf = __ctx.infinity();
    let neg_inf = __ctx.neg_infinity();
    let five = __ctx.int(5);

    assert_eq!(format!("{}", &inf + &five), "oo", "oo + 5 = oo");
    assert_eq!(format!("{}", &neg_inf + &five), "-oo", "(-oo) + 5 = -oo");
    assert_eq!(format!("{}", &inf + &neg_inf), "nan", "oo + (-oo) = nan");
}
