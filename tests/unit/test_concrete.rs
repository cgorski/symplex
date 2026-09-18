//! Concrete evaluation tests for Floor, Ceiling, Min, Max, Sum, and Product_ variants.
//!
//! These tests focus on numeric evaluation (`.eval()` and `.eval_decimal()`) to verify
//! that the new variants produce correct results for concrete inputs.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Floor concrete evaluation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn floor_large_rational() {
    let ctx = Context::new();
    // floor(100/7) = floor(14.2857...) = 14
    let result = ctx.rational(100, 7).floor().eval();
    assert_eq!(format!("{result}"), "14");
}

#[test]
fn floor_just_below_integer() {
    let ctx = Context::new();
    // floor(99/10) = floor(9.9) = 9
    let result = ctx.rational(99, 10).floor().eval();
    assert_eq!(format!("{result}"), "9");
}

#[test]
fn floor_negative_just_above_integer() {
    let ctx = Context::new();
    // floor(-99/10) = floor(-9.9) = -10
    let result = ctx.rational(-99, 10).floor().eval();
    assert_eq!(format!("{result}"), "-10");
}

// ═══════════════════════════════════════════════════════════════════════════
// Ceiling concrete evaluation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ceiling_large_rational() {
    let ctx = Context::new();
    // ceil(100/7) = ceil(14.2857...) = 15
    let result = ctx.rational(100, 7).ceiling().eval();
    assert_eq!(format!("{result}"), "15");
}

#[test]
fn ceiling_negative_just_below_integer() {
    let ctx = Context::new();
    // ceil(-99/10) = ceil(-9.9) = -9
    let result = ctx.rational(-99, 10).ceiling().eval();
    assert_eq!(format!("{result}"), "-9");
}

#[test]
fn ceiling_exact_integer_unchanged() {
    let ctx = Context::new();
    // ceil(10/2) = ceil(5) = 5
    let result = ctx.rational(10, 2).ceiling().eval();
    assert_eq!(format!("{result}"), "5");
}

// ═══════════════════════════════════════════════════════════════════════════
// Rem (remainder) tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rem_positive_integers() {
    let ctx = Context::new();
    // 7 rem 3 = 7 - 3*floor(7/3) = 7 - 3*2 = 1
    let result = ctx.int(7).rem(&ctx.int(3)).eval();
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn rem_negative_dividend() {
    let ctx = Context::new();
    // (-7) rem 3 = -7 - 3*floor(-7/3) = -7 - 3*(-3) = -7 + 9 = 2
    let result = ctx.int(-7).rem(&ctx.int(3)).eval();
    assert_eq!(format!("{result}"), "2");
}

// ═══════════════════════════════════════════════════════════════════════════
// Sum concrete evaluation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sum_cubes_1_to_4() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = k.powi(3);
    let one = ctx.int(1);
    let four = ctx.int(4);
    let s = Ex::symbolic_sum(&body, &k, &one, &four);
    let result = s.eval();
    // 1 + 8 + 27 + 64 = 100
    assert_eq!(format!("{result}"), "100");
}

#[test]
fn sum_constants() {
    // Sum(2, k=1..5) = 2+2+2+2+2 = 10
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = ctx.int(2);
    let one = ctx.int(1);
    let five = ctx.int(5);
    let s = Ex::symbolic_sum(&body, &k, &one, &five);
    let result = s.eval();
    assert_eq!(format!("{result}"), "10");
}

#[test]
fn sum_empty_range() {
    // Sum(k, k=5..1) — upper < lower, so empty sum = 0
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let five = ctx.int(5);
    let one = ctx.int(1);
    let s = Ex::symbolic_sum(&k, &k, &five, &one);
    let result = s.eval();
    assert_eq!(format!("{result}"), "0");
}

// ═══════════════════════════════════════════════════════════════════════════
// Product_ concrete evaluation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn product_2_to_4() {
    // Product(k, k=2..4) = 2*3*4 = 24
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let two = ctx.int(2);
    let four = ctx.int(4);
    let p = Ex::symbolic_product(&k, &k, &two, &four);
    let result = p.eval();
    assert_eq!(format!("{result}"), "24");
}

#[test]
fn product_squares_1_to_3() {
    // Product(k^2, k=1..3) = 1 * 4 * 9 = 36
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = k.powi(2);
    let one = ctx.int(1);
    let three = ctx.int(3);
    let p = Ex::symbolic_product(&body, &k, &one, &three);
    let result = p.eval();
    assert_eq!(format!("{result}"), "36");
}

#[test]
fn product_empty_range_is_one() {
    // Product(k, k=5..1) — empty product = 1
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let five = ctx.int(5);
    let one = ctx.int(1);
    let p = Ex::symbolic_product(&k, &k, &five, &one);
    let result = p.eval();
    assert_eq!(format!("{result}"), "1");
}

// ═══════════════════════════════════════════════════════════════════════════
// Tree round-trip tests for new variants
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn tree_roundtrip_floor() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let fl = x.floor();
    let tree = fl.to_tree();
    let json = serde_json::to_string(&tree).unwrap();
    let tree2: symplex::tree::ExprTree = serde_json::from_str(&json).unwrap();
    assert_eq!(tree, tree2);
}

#[test]
fn tree_roundtrip_min_max() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let m = x.min_with(&y);
    let tree = m.to_tree();
    let json = serde_json::to_string(&tree).unwrap();
    let tree2: symplex::tree::ExprTree = serde_json::from_str(&json).unwrap();
    assert_eq!(tree, tree2);
}
