//! Tests for Floor, Ceiling, Min, Max, Sum, and Product_ variants.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Floor tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn floor_of_integer() {
    let __ctx = Context::new();
    let result = __ctx.int(3).floor().eval();
    assert_eq!(format!("{result}"), "3");
}

#[test]
fn floor_of_positive_rational() {
    let __ctx = Context::new();
    let result = __ctx.rational(7, 2).floor().eval();
    assert_eq!(format!("{result}"), "3");
}

#[test]
fn floor_of_negative_rational() {
    let __ctx = Context::new();
    let result = __ctx.rational(-7, 2).floor().eval();
    assert_eq!(format!("{result}"), "-4");
}

#[test]
fn floor_of_zero() {
    let __ctx = Context::new();
    let result = __ctx.int(0).floor().eval();
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn floor_of_negative_integer() {
    let __ctx = Context::new();
    let result = __ctx.int(-5).floor().eval();
    assert_eq!(format!("{result}"), "-5");
}

#[test]
fn floor_symbolic_stays_unevaluated() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let fl = x.floor();
    let s = format!("{fl}");
    assert!(s.contains("floor"), "expected 'floor' in display, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Ceiling tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ceiling_of_integer() {
    let __ctx = Context::new();
    let result = __ctx.int(3).ceiling().eval();
    assert_eq!(format!("{result}"), "3");
}

#[test]
fn ceiling_of_positive_rational() {
    let __ctx = Context::new();
    let result = __ctx.rational(7, 2).ceiling().eval();
    assert_eq!(format!("{result}"), "4");
}

#[test]
fn ceiling_of_negative_rational() {
    let __ctx = Context::new();
    let result = __ctx.rational(-7, 2).ceiling().eval();
    assert_eq!(format!("{result}"), "-3");
}

#[test]
fn ceiling_of_zero() {
    let __ctx = Context::new();
    let result = __ctx.int(0).ceiling().eval();
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn ceiling_symbolic_stays_unevaluated() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let cl = x.ceiling();
    let s = format!("{cl}");
    assert!(
        s.contains("ceiling"),
        "expected 'ceiling' in display, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Frac (fractional part) tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn frac_of_positive_rational() {
    let __ctx = Context::new();
    let result = __ctx.rational(7, 2).frac().eval();
    assert_eq!(format!("{result}"), "1/2");
}

#[test]
fn frac_of_integer_is_zero() {
    let __ctx = Context::new();
    let result = __ctx.int(5).frac().eval();
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn frac_of_negative_rational() {
    let __ctx = Context::new();
    // frac(-7/2) = -7/2 - floor(-7/2) = -7/2 - (-4) = 1/2
    let result = __ctx.rational(-7, 2).frac().eval();
    assert_eq!(format!("{result}"), "1/2");
}

// ═══════════════════════════════════════════════════════════════════════════
// Min / Max tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn min_of_two_integers() {
    let __ctx = Context::new();
    let result = __ctx.int(3).min_with(&__ctx.int(5)).eval();
    assert_eq!(format!("{result}"), "3");
}

#[test]
fn max_of_two_integers() {
    let __ctx = Context::new();
    let result = __ctx.int(3).max_with(&__ctx.int(5)).eval();
    assert_eq!(format!("{result}"), "5");
}

#[test]
fn min_of_negative_integers() {
    let __ctx = Context::new();
    let result = __ctx.int(-10).min_with(&__ctx.int(-3)).eval();
    assert_eq!(format!("{result}"), "-10");
}

#[test]
fn max_of_negative_integers() {
    let __ctx = Context::new();
    let result = __ctx.int(-10).max_with(&__ctx.int(-3)).eval();
    assert_eq!(format!("{result}"), "-3");
}

#[test]
fn min_symbolic_stays_unevaluated() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let y = __ctx.symbol("y");
    let m = x.min_with(&y);
    let s = format!("{m}");
    assert!(s.contains("min"), "expected 'min' in display, got: {s}");
}

#[test]
fn max_symbolic_stays_unevaluated() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let y = __ctx.symbol("y");
    let m = x.max_with(&y);
    let s = format!("{m}");
    assert!(s.contains("max"), "expected 'max' in display, got: {s}");
}

#[test]
fn min_of_rationals() {
    let __ctx = Context::new();
    let a = __ctx.rational(1, 3);
    let b = __ctx.rational(1, 2);
    let result = a.min_with(&b).eval();
    assert_eq!(format!("{result}"), "1/3");
}

#[test]
fn max_of_rationals() {
    let __ctx = Context::new();
    let a = __ctx.rational(1, 3);
    let b = __ctx.rational(1, 2);
    let result = a.max_with(&b).eval();
    assert_eq!(format!("{result}"), "1/2");
}

// ═══════════════════════════════════════════════════════════════════════════
// Sum tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sum_1_to_10() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let one = ctx.int(1);
    let ten = ctx.int(10);
    let s = Ex::symbolic_sum(&k, &k, &one, &ten);
    let result = s.eval();
    assert_eq!(format!("{result}"), "55");
}

#[test]
fn sum_squares_1_to_5() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = k.powi(2);
    let one = ctx.int(1);
    let five = ctx.int(5);
    let s = Ex::symbolic_sum(&body, &k, &one, &five);
    let result = s.eval();
    // 1 + 4 + 9 + 16 + 25 = 55
    assert_eq!(format!("{result}"), "55");
}

#[test]
fn sum_display_format() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let one = ctx.int(1);
    let n = ctx.symbol("n");
    let s = Ex::symbolic_sum(&k, &k, &one, &n);
    let display = format!("{s}");
    assert!(
        display.contains("Sum"),
        "expected 'Sum' in display, got: {display}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Product_ tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn product_factorial_5() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let one = ctx.int(1);
    let five = ctx.int(5);
    let p = Ex::symbolic_product(&k, &k, &one, &five);
    let result = p.eval();
    // 1 * 2 * 3 * 4 * 5 = 120
    assert_eq!(format!("{result}"), "120");
}

#[test]
fn product_display_format() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let one = ctx.int(1);
    let n = ctx.symbol("n");
    let p = Ex::symbolic_product(&k, &k, &one, &n);
    let display = format!("{p}");
    assert!(
        display.contains("Product"),
        "expected 'Product' in display, got: {display}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Derivative tests for floor/ceiling
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_floor_is_zero() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let fl = x.floor();
    let d = fl.diff(&x);
    assert_eq!(format!("{d}"), "0");
}

#[test]
fn diff_ceiling_is_zero() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let cl = x.ceiling();
    let d = cl.diff(&x);
    assert_eq!(format!("{d}"), "0");
}
