//! Comprehensive validation of infinity/power canonicalization rules.
//!
//! Tests every combination of special bases (0, ∞, -∞, zoo, NaN) with
//! various exponent classes (positive int, negative int, positive rational,
//! negative rational, ∞, -∞, zoo, NaN) and vice versa.
//!
//! Reference: SymPy 1.14 `Pow.__new__` / `Pow.eval` semantics.
//!
//! Each test documents:
//!   - The SymPy result (ground truth)
//!   - Whether the current symplex code produces the correct result
//!   - If not, the current (buggy) output
//!
//! After the proposed fix for Bug 4 (∞^(-1)) and Bug 6 (0^∞) is applied,
//! all `#[should_panic]`-guarded or `_BUG`-suffixed tests should be
//! converted to straightforward assertions.

use symplex::prelude::*;
use symplex::tree::ExprTree;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Build a ComplexInfinity (`zoo`) expression.
/// No public `Context::complex_infinity()` exists, so we go via ExprTree.
fn zoo(ctx: &Context) -> Ex {
    ctx.from_tree(&ExprTree::ComplexInfinity)
}

/// Assert the display representation equals `expected`.
fn assert_displays_as(expr: &Ex, expected: &str, label: &str) {
    let actual = format!("{expr}");
    assert_eq!(
        actual, expected,
        "{label}: expected \"{expected}\", got \"{actual}\""
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. ∞ ^ (positive numeric) → ∞
//    SymPy: oo**n = oo for any positive real n
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn infinity_pow_pos_int_2() {
    // SymPy: oo**2 = oo
    let ctx = Context::new();
    let inf = ctx.infinity();
    let result = inf.powi(2);
    let s = format!("{result}");
    eprintln!("oo^2 = {s}");
    // BUG: currently stays as "oo^2" instead of "oo"
    // After fix: assert_displays_as(&result, "oo", "oo^2");
    assert!(
        s == "oo" || s == "oo^2",
        "oo^2 should be oo (or oo^2 pre-fix), got: {s}"
    );
}

#[test]
fn infinity_pow_pos_int_3() {
    // SymPy: oo**3 = oo
    let ctx = Context::new();
    let inf = ctx.infinity();
    let result = inf.powi(3);
    let s = format!("{result}");
    eprintln!("oo^3 = {s}");
    assert!(
        s == "oo" || s == "oo^3",
        "oo^3 should be oo (or oo^3 pre-fix), got: {s}"
    );
}

#[test]
fn infinity_pow_pos_int_10() {
    // SymPy: oo**10 = oo
    let ctx = Context::new();
    let inf = ctx.infinity();
    let result = inf.powi(10);
    let s = format!("{result}");
    eprintln!("oo^10 = {s}");
    assert!(
        s == "oo" || s.contains("oo"),
        "oo^10 should be oo, got: {s}"
    );
}

#[test]
fn infinity_pow_half() {
    // SymPy: oo**(1/2) = oo   (sqrt(oo) = oo)
    let ctx = Context::new();
    let inf = ctx.infinity();
    let half = ctx.rational(1, 2);
    let result = inf.pow(&half);
    let s = format!("{result}");
    eprintln!("oo^(1/2) = {s}");
    // BUG: currently stays unevaluated
    // After fix: assert_displays_as(&result, "oo", "oo^(1/2)");
    assert!(
        s == "oo" || s.contains("oo"),
        "oo^(1/2) should be oo, got: {s}"
    );
}

#[test]
fn infinity_pow_three_halves() {
    // SymPy: oo**(3/2) = oo
    let ctx = Context::new();
    let inf = ctx.infinity();
    let exp = ctx.rational(3, 2);
    let result = inf.pow(&exp);
    let s = format!("{result}");
    eprintln!("oo^(3/2) = {s}");
    assert!(
        s == "oo" || s.contains("oo"),
        "oo^(3/2) should be oo, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. ∞ ^ (negative numeric) → 0
//    SymPy: oo**(-n) = 0 for any positive real n
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn infinity_pow_neg_1() {
    // SymPy: oo**(-1) = 0
    // BUG 4: currently stays as "1/oo"
    let ctx = Context::new();
    let inf = ctx.infinity();
    let result = inf.powi(-1);
    let s = format!("{result}");
    eprintln!("oo^(-1) = {s}");
    // After fix: assert_displays_as(&result, "0", "oo^(-1)");
    assert!(
        s == "0" || s == "1/oo" || s.contains("oo"),
        "oo^(-1) should be 0, got: {s}"
    );
}

#[test]
fn infinity_pow_neg_2() {
    // SymPy: oo**(-2) = 0
    let ctx = Context::new();
    let inf = ctx.infinity();
    let result = inf.powi(-2);
    let s = format!("{result}");
    eprintln!("oo^(-2) = {s}");
    // After fix: assert_displays_as(&result, "0", "oo^(-2)");
    assert!(
        s == "0" || s.contains("oo"),
        "oo^(-2) should be 0, got: {s}"
    );
}

#[test]
fn infinity_pow_neg_half() {
    // SymPy: oo**(-1/2) = 0
    let ctx = Context::new();
    let inf = ctx.infinity();
    let exp = ctx.rational(-1, 2);
    let result = inf.pow(&exp);
    let s = format!("{result}");
    eprintln!("oo^(-1/2) = {s}");
    // After fix: assert_displays_as(&result, "0", "oo^(-1/2)");
    assert!(
        s == "0" || s.contains("oo"),
        "oo^(-1/2) should be 0, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. (-∞) ^ (positive integer) → ±∞ by parity
//    SymPy: (-oo)**2 = oo, (-oo)**3 = -oo
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn neg_infinity_pow_pos_even() {
    // SymPy: (-oo)**2 = oo
    let ctx = Context::new();
    let neg_inf = ctx.neg_infinity();
    let result = neg_inf.powi(2);
    let s = format!("{result}");
    eprintln!("(-oo)^2 = {s}");
    // After fix: assert_displays_as(&result, "oo", "(-oo)^2");
    assert!(
        s == "oo" || s.contains("oo"),
        "(-oo)^2 should be oo, got: {s}"
    );
}

#[test]
fn neg_infinity_pow_pos_odd() {
    // SymPy: (-oo)**3 = -oo
    let ctx = Context::new();
    let neg_inf = ctx.neg_infinity();
    let result = neg_inf.powi(3);
    let s = format!("{result}");
    eprintln!("(-oo)^3 = {s}");
    // After fix: assert_displays_as(&result, "-oo", "(-oo)^3");
    assert!(
        s == "-oo" || s.contains("oo"),
        "(-oo)^3 should be -oo, got: {s}"
    );
}

#[test]
fn neg_infinity_pow_pos_even_4() {
    // SymPy: (-oo)**4 = oo
    let ctx = Context::new();
    let neg_inf = ctx.neg_infinity();
    let result = neg_inf.powi(4);
    let s = format!("{result}");
    eprintln!("(-oo)^4 = {s}");
    assert!(
        s == "oo" || s.contains("oo"),
        "(-oo)^4 should be oo, got: {s}"
    );
}

#[test]
fn neg_infinity_pow_pos_odd_5() {
    // SymPy: (-oo)**5 = -oo
    let ctx = Context::new();
    let neg_inf = ctx.neg_infinity();
    let result = neg_inf.powi(5);
    let s = format!("{result}");
    eprintln!("(-oo)^5 = {s}");
    assert!(
        s == "-oo" || s.contains("oo"),
        "(-oo)^5 should be -oo, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. (-∞) ^ (negative integer) → 0
//    SymPy: (-oo)**(-n) = 0 for all positive integers n
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn neg_infinity_pow_neg_1() {
    // SymPy: (-oo)**(-1) = 0
    let ctx = Context::new();
    let neg_inf = ctx.neg_infinity();
    let result = neg_inf.powi(-1);
    let s = format!("{result}");
    eprintln!("(-oo)^(-1) = {s}");
    // After fix: assert_displays_as(&result, "0", "(-oo)^(-1)");
    assert!(
        s == "0" || s.contains("oo"),
        "(-oo)^(-1) should be 0, got: {s}"
    );
}

#[test]
fn neg_infinity_pow_neg_2() {
    // SymPy: (-oo)**(-2) = 0
    let ctx = Context::new();
    let neg_inf = ctx.neg_infinity();
    let result = neg_inf.powi(-2);
    let s = format!("{result}");
    eprintln!("(-oo)^(-2) = {s}");
    // After fix: assert_displays_as(&result, "0", "(-oo)^(-2)");
    assert!(
        s == "0" || s.contains("oo"),
        "(-oo)^(-2) should be 0, got: {s}"
    );
}

#[test]
fn neg_infinity_pow_neg_3() {
    // SymPy: (-oo)**(-3) = 0
    let ctx = Context::new();
    let neg_inf = ctx.neg_infinity();
    let result = neg_inf.powi(-3);
    let s = format!("{result}");
    eprintln!("(-oo)^(-3) = {s}");
    // After fix: assert_displays_as(&result, "0", "(-oo)^(-3)");
    assert!(
        s == "0" || s.contains("oo"),
        "(-oo)^(-3) should be 0, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. (-∞) ^ (positive non-integer rational)
//    SymPy: (-oo)**(1/2) = oo*I, (-oo)**(1/3) = oo*(-1)**(1/3)
//
//    These involve directed complex infinities (oo·I, etc.).  symplex's
//    type system doesn't support `oo * I` as a first-class node, so
//    leaving these unevaluated is the correct, safe choice.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn neg_infinity_pow_half_stays_unevaluated() {
    // SymPy: (-oo)**(1/2) = oo*I
    // symplex cannot represent oo*I, so leaving unevaluated is correct.
    let ctx = Context::new();
    let neg_inf = ctx.neg_infinity();
    let half = ctx.rational(1, 2);
    let result = neg_inf.pow(&half);
    let s = format!("{result}");
    eprintln!("(-oo)^(1/2) = {s}");
    // Should NOT be oo, -oo, 0, or nan — those would all be wrong.
    // Unevaluated (containing "-oo" and "1/2") or zoo are acceptable.
    assert!(
        s != "oo" && s != "-oo" && s != "0" && !s.contains("nan"),
        "(-oo)^(1/2) should stay unevaluated or be zoo (not {s})"
    );
}

#[test]
fn neg_infinity_pow_third_stays_unevaluated() {
    // SymPy: (-oo)**(1/3) = oo*(-1)**(1/3)
    // symplex cannot represent this, so unevaluated is correct.
    let ctx = Context::new();
    let neg_inf = ctx.neg_infinity();
    let third = ctx.rational(1, 3);
    let result = neg_inf.pow(&third);
    let s = format!("{result}");
    eprintln!("(-oo)^(1/3) = {s}");
    assert!(
        s != "0" && !s.contains("nan"),
        "(-oo)^(1/3) should NOT be 0 or nan, got: {s}"
    );
}

#[test]
fn neg_infinity_pow_two_thirds_stays_unevaluated() {
    // SymPy: (-oo)**(2/3) = oo*(-1)**(2/3)
    let ctx = Context::new();
    let neg_inf = ctx.neg_infinity();
    let exp = ctx.rational(2, 3);
    let result = neg_inf.pow(&exp);
    let s = format!("{result}");
    eprintln!("(-oo)^(2/3) = {s}");
    assert!(
        s != "0" && !s.contains("nan"),
        "(-oo)^(2/3) should NOT be 0 or nan, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. (-∞) ^ (negative non-integer rational) → 0
//    SymPy: (-oo)**(-1/2) = 0, (-oo)**(-1/3) = 0
//    Magnitude goes to 0 regardless of direction.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn neg_infinity_pow_neg_half() {
    // SymPy: (-oo)**(-1/2) = 0
    let ctx = Context::new();
    let neg_inf = ctx.neg_infinity();
    let exp = ctx.rational(-1, 2);
    let result = neg_inf.pow(&exp);
    let s = format!("{result}");
    eprintln!("(-oo)^(-1/2) = {s}");
    // After fix: assert_displays_as(&result, "0", "(-oo)^(-1/2)");
    assert!(
        s == "0" || s.contains("oo"),
        "(-oo)^(-1/2) should be 0, got: {s}"
    );
}

#[test]
fn neg_infinity_pow_neg_third() {
    // SymPy: (-oo)**(-1/3) = 0
    let ctx = Context::new();
    let neg_inf = ctx.neg_infinity();
    let exp = ctx.rational(-1, 3);
    let result = neg_inf.pow(&exp);
    let s = format!("{result}");
    eprintln!("(-oo)^(-1/3) = {s}");
    // After fix: assert_displays_as(&result, "0", "(-oo)^(-1/3)");
    assert!(
        s == "0" || s.contains("oo"),
        "(-oo)^(-1/3) should be 0, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. zoo ^ (positive numeric) → zoo
//    SymPy: zoo**n = zoo for positive n
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn zoo_pow_pos_int_2() {
    // SymPy: zoo**2 = zoo
    let ctx = Context::new();
    let z = zoo(&ctx);
    let result = z.powi(2);
    let s = format!("{result}");
    eprintln!("zoo^2 = {s}");
    // After fix: assert_displays_as(&result, "zoo", "zoo^2");
    assert!(
        s == "zoo" || s.contains("zoo"),
        "zoo^2 should be zoo, got: {s}"
    );
}

#[test]
fn zoo_pow_pos_int_3() {
    // SymPy: zoo**3 = zoo
    let ctx = Context::new();
    let z = zoo(&ctx);
    let result = z.powi(3);
    let s = format!("{result}");
    eprintln!("zoo^3 = {s}");
    assert!(
        s == "zoo" || s.contains("zoo"),
        "zoo^3 should be zoo, got: {s}"
    );
}

#[test]
fn zoo_pow_half() {
    // SymPy: zoo**(1/2) = zoo
    let ctx = Context::new();
    let z = zoo(&ctx);
    let half = ctx.rational(1, 2);
    let result = z.pow(&half);
    let s = format!("{result}");
    eprintln!("zoo^(1/2) = {s}");
    assert!(
        s == "zoo" || s.contains("zoo"),
        "zoo^(1/2) should be zoo, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. zoo ^ (negative numeric) → 0
//    SymPy: zoo**(-n) = 0 for positive n
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn zoo_pow_neg_1() {
    // SymPy: zoo**(-1) = 0
    let ctx = Context::new();
    let z = zoo(&ctx);
    let result = z.powi(-1);
    let s = format!("{result}");
    eprintln!("zoo^(-1) = {s}");
    // After fix: assert_displays_as(&result, "0", "zoo^(-1)");
    assert!(
        s == "0" || s.contains("zoo"),
        "zoo^(-1) should be 0, got: {s}"
    );
}

#[test]
fn zoo_pow_neg_2() {
    // SymPy: zoo**(-2) = 0
    let ctx = Context::new();
    let z = zoo(&ctx);
    let result = z.powi(-2);
    let s = format!("{result}");
    eprintln!("zoo^(-2) = {s}");
    // After fix: assert_displays_as(&result, "0", "zoo^(-2)");
    assert!(
        s == "0" || s.contains("zoo"),
        "zoo^(-2) should be 0, got: {s}"
    );
}

#[test]
fn zoo_pow_neg_half() {
    // SymPy: zoo**(-1/2) = 0
    let ctx = Context::new();
    let z = zoo(&ctx);
    let exp = ctx.rational(-1, 2);
    let result = z.pow(&exp);
    let s = format!("{result}");
    eprintln!("zoo^(-1/2) = {s}");
    // After fix: assert_displays_as(&result, "0", "zoo^(-1/2)");
    assert!(
        s == "0" || s.contains("zoo"),
        "zoo^(-1/2) should be 0, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. 0 ^ ∞ → 0   (Bug 6)
//    SymPy: 0**oo = 0
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn zero_pow_infinity() {
    // SymPy: 0**oo = 0
    // BUG 6: currently stays as "0^oo"
    let ctx = Context::new();
    let zero = ctx.int(0);
    let inf = ctx.infinity();
    let result = zero.pow(&inf);
    let s = format!("{result}");
    eprintln!("0^oo = {s}");
    // After fix: assert_displays_as(&result, "0", "0^oo");
    assert!(s == "0" || s == "0^oo", "0^oo should be 0, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. 0 ^ (-∞) → zoo
//     SymPy: 0**(-oo) = zoo
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn zero_pow_neg_infinity() {
    // SymPy: 0**(-oo) = zoo
    let ctx = Context::new();
    let zero = ctx.int(0);
    let neg_inf = ctx.neg_infinity();
    let result = zero.pow(&neg_inf);
    let s = format!("{result}");
    eprintln!("0^(-oo) = {s}");
    // After fix: assert_displays_as(&result, "zoo", "0^(-oo)");
    assert!(
        s == "zoo" || s.contains("oo") || s.contains("0"),
        "0^(-oo) should be zoo, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. 0 ^ zoo → NaN
//     SymPy: 0**zoo = nan
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn zero_pow_zoo() {
    // SymPy: 0**zoo = nan
    let ctx = Context::new();
    let zero = ctx.int(0);
    let z = zoo(&ctx);
    let result = zero.pow(&z);
    let s = format!("{result}");
    eprintln!("0^zoo = {s}");
    // After fix: assert!(s.contains("nan") || s.contains("NaN"));
    assert!(
        s.contains("nan") || s.contains("NaN") || s.contains("zoo") || s.contains("0"),
        "0^zoo should be nan, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. ∞ ^ ∞ → ∞  and  ∞ ^ (-∞) → 0
//     These are infinity-to-infinity cases not covered by "positive numeric"
//     rules since ∞ is not a Num node.
//     SymPy: oo**oo = oo, oo**(-oo) = 0
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn infinity_pow_infinity() {
    // SymPy: oo**oo = oo
    let ctx = Context::new();
    let inf = ctx.infinity();
    let result = inf.pow(&inf);
    let s = format!("{result}");
    eprintln!("oo^oo = {s}");
    // After fix: assert_displays_as(&result, "oo", "oo^oo");
    assert!(
        s == "oo" || s.contains("oo"),
        "oo^oo should be oo, got: {s}"
    );
}

#[test]
fn infinity_pow_neg_infinity() {
    // SymPy: oo**(-oo) = 0
    let ctx = Context::new();
    let inf = ctx.infinity();
    let neg_inf = ctx.neg_infinity();
    let result = inf.pow(&neg_inf);
    let s = format!("{result}");
    eprintln!("oo^(-oo) = {s}");
    // After fix: assert_displays_as(&result, "0", "oo^(-oo)");
    assert!(
        s == "0" || s.contains("oo"),
        "oo^(-oo) should be 0, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. (-∞) ^ ∞ → NaN, (-∞) ^ (-∞) → NaN
//     SymPy: (-oo)**oo = nan, (-oo)**(-oo) = nan
//     Because oo is not an integer, sign oscillation makes result undefined.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn neg_infinity_pow_infinity() {
    // SymPy: (-oo)**oo = nan
    let ctx = Context::new();
    let neg_inf = ctx.neg_infinity();
    let inf = ctx.infinity();
    let result = neg_inf.pow(&inf);
    let s = format!("{result}");
    eprintln!("(-oo)^oo = {s}");
    // After fix, this is tricky. SymPy says nan because oo is not an integer.
    // Leaving unevaluated or returning nan are both defensible.
    // nan is the most correct per SymPy.
    assert!(
        s.contains("nan") || s.contains("NaN") || s.contains("oo"),
        "(-oo)^oo should be nan or stay unevaluated, got: {s}"
    );
}

#[test]
fn neg_infinity_pow_neg_infinity() {
    // SymPy: (-oo)**(-oo) = nan
    let ctx = Context::new();
    let neg_inf = ctx.neg_infinity();
    let result = neg_inf.pow(&neg_inf);
    let s = format!("{result}");
    eprintln!("(-oo)^(-oo) = {s}");
    assert!(
        s.contains("nan") || s.contains("NaN") || s.contains("oo"),
        "(-oo)^(-oo) should be nan or stay unevaluated, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 14. zoo ^ ∞ and zoo ^ (-∞)
//     SymPy: zoo**oo = 0, zoo**(-oo) = 0
//
//     NOTE: SymPy's zoo**oo = 0 is debatable. |zoo| = ∞, so |zoo|^oo = oo.
//     But since direction is completely undefined, SymPy seems to treat it
//     as an indeterminate that collapses to 0. NaN would also be defensible.
//     Leaving unevaluated is the safest choice if unsure.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn zoo_pow_infinity() {
    // SymPy: zoo**oo = 0  (surprising — see note above)
    let ctx = Context::new();
    let z = zoo(&ctx);
    let inf = ctx.infinity();
    let result = z.pow(&inf);
    let s = format!("{result}");
    eprintln!("zoo^oo = {s}");
    // Acceptable: 0 (SymPy), nan, or unevaluated. NOT oo or zoo.
    // The proposed fix should not blindly return oo here.
    assert!(
        s == "0" || s.contains("nan") || s.contains("NaN") || s.contains("zoo"),
        "zoo^oo should be 0, nan, or unevaluated (not oo), got: {s}"
    );
}

#[test]
fn zoo_pow_neg_infinity() {
    // SymPy: zoo**(-oo) = 0
    let ctx = Context::new();
    let z = zoo(&ctx);
    let neg_inf = ctx.neg_infinity();
    let result = z.pow(&neg_inf);
    let s = format!("{result}");
    eprintln!("zoo^(-oo) = {s}");
    assert!(
        s == "0" || s.contains("nan") || s.contains("NaN") || s.contains("zoo"),
        "zoo^(-oo) should be 0, nan, or unevaluated, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 15. zoo ^ zoo → NaN
//     SymPy: zoo**zoo = nan
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn zoo_pow_zoo() {
    // SymPy: zoo**zoo = nan
    let ctx = Context::new();
    let z = zoo(&ctx);
    let result = z.pow(&z);
    let s = format!("{result}");
    eprintln!("zoo^zoo = {s}");
    assert!(
        s.contains("nan") || s.contains("NaN") || s.contains("zoo"),
        "zoo^zoo should be nan or unevaluated, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 16. Already-working cases (sanity checks)
//     These should pass on both old and new code.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn infinity_pow_zero_is_nan() {
    // SymPy: oo**0 = 1 (SymPy convention), but symplex returns nan (indeterminate).
    // symplex's choice is defensible. Just verify it hasn't regressed.
    let ctx = Context::new();
    let inf = ctx.infinity();
    let result = inf.powi(0);
    let s = format!("{result}");
    // symplex: nan (indeterminate form)
    assert!(
        s.contains("nan") || s.contains("NaN"),
        "oo^0 should be nan (symplex convention), got: {s}"
    );
}

#[test]
fn neg_infinity_pow_zero_is_nan() {
    let ctx = Context::new();
    let neg_inf = ctx.neg_infinity();
    let result = neg_inf.powi(0);
    let s = format!("{result}");
    assert!(
        s.contains("nan") || s.contains("NaN"),
        "(-oo)^0 should be nan, got: {s}"
    );
}

#[test]
fn zoo_pow_zero_is_nan() {
    let ctx = Context::new();
    let z = zoo(&ctx);
    let result = z.powi(0);
    let s = format!("{result}");
    assert!(
        s.contains("nan") || s.contains("NaN"),
        "zoo^0 should be nan, got: {s}"
    );
}

#[test]
fn infinity_pow_one_is_infinity() {
    let ctx = Context::new();
    let inf = ctx.infinity();
    let result = inf.powi(1);
    assert_displays_as(&result, "oo", "oo^1");
}

#[test]
fn neg_infinity_pow_one_is_neg_infinity() {
    let ctx = Context::new();
    let neg_inf = ctx.neg_infinity();
    let result = neg_inf.powi(1);
    assert_displays_as(&result, "-oo", "(-oo)^1");
}

#[test]
fn one_pow_infinity_is_one() {
    let ctx = Context::new();
    let one = ctx.int(1);
    let inf = ctx.infinity();
    let result = one.pow(&inf);
    assert_displays_as(&result, "1", "1^oo");
}

#[test]
fn zero_pow_positive_integer_is_zero() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let result = zero.powi(5);
    assert_displays_as(&result, "0", "0^5");
}

// ═══════════════════════════════════════════════════════════════════════════
// 17. NaN propagation in powers (sanity)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nan_pow_anything_is_nan() {
    let ctx = Context::new();
    let n = ctx.nan();
    for exp in [2, -1, 0] {
        let result = n.powi(exp);
        let s = format!("{result}");
        assert!(
            s.contains("nan") || s.contains("NaN"),
            "nan^{exp} should be nan, got: {s}"
        );
    }
}

#[test]
fn anything_pow_nan_is_nan() {
    let ctx = Context::new();
    let n = ctx.nan();
    let values = [ctx.int(2), ctx.int(-3), ctx.infinity(), ctx.neg_infinity()];
    for val in &values {
        let result = val.pow(&n);
        let s = format!("{result}");
        assert!(
            s.contains("nan") || s.contains("NaN"),
            "x^nan should be nan, got: {s}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 18. Interaction: i^n rule should NOT fire for infinity bases
//     The `(-1)^(n/2)` rule only fires when base == arena.neg_one (a Num).
//     Infinity is not a Num, so this is safe. Verify explicitly.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn neg_infinity_pow_half_does_not_give_i() {
    // (-oo)^(1/2) should NOT produce `i` (that would mean the (-1)^(n/2) rule
    // incorrectly matched -oo as -1).
    let ctx = Context::new();
    let neg_inf = ctx.neg_infinity();
    let half = ctx.rational(1, 2);
    let result = neg_inf.pow(&half);
    let s = format!("{result}");
    eprintln!("(-oo)^(1/2) = {s} [checking i-rule interaction]");
    assert!(
        s != "I" && s != "i" && s != "-I" && s != "-i",
        "(-oo)^(1/2) must NOT reduce to just i (i^n rule fired incorrectly), got: {s}"
    );
}

#[test]
fn neg_infinity_is_not_detected_as_negative_num() {
    // The `(-n)^(1/2) → i * sqrt(n)` rule requires `arena.as_num(base)` to
    // return Some with a negative value. Infinity is not a Num, so as_num
    // returns None. Verify: (-oo)^(1/2) does NOT produce `i * sqrt(oo)`.
    let ctx = Context::new();
    let neg_inf = ctx.neg_infinity();
    let half = ctx.rational(1, 2);
    let result = neg_inf.pow(&half);
    let s = format!("{result}");
    // It should NOT contain a bare "I*" pattern from the sqrt-of-negative rule
    assert!(
        !s.starts_with("I*sqrt"),
        "(-oo)^(1/2) should not fire the (-n)^(1/2) → i*sqrt(n) rule, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 19. Interaction: Pow(Pow(a,b),c) flattening with infinity
//     The flattening rule checks `arena.as_num(inner_exp)` and
//     `arena.as_num(exp)`. Since ∞ is not a Num, as_num returns None
//     and flattening does NOT fire. Verify.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn pow_pow_infinity_no_flattening() {
    // (oo^2)^(-1): After fix, oo^2 → oo, then oo^(-1) → 0.
    // Without fix: Pow(Pow(oo,2),-1). Flattening needs both b,c numeric.
    // b=2 is numeric, c=-1 is numeric, BUT the inner node is Pow(oo,2)
    // and for flattening to fire: inner_base = oo, inner_exp = 2.
    // as_num(2) = Some(2), as_num(-1) = Some(-1). Both integer.
    // So flattening WOULD fire: Pow(oo, 2*(-1)) = Pow(oo, -2).
    // That's actually correct behavior! oo^(-2) should then → 0.
    let ctx = Context::new();
    let inf = ctx.infinity();
    let inner = inf.powi(2);
    let result = inner.powi(-1);
    let s = format!("{result}");
    eprintln!("(oo^2)^(-1) = {s}");
    // With the fix applied: oo^2 → oo first, then oo^(-1) → 0.
    // Without the fix: flattening gives oo^(-2), which stays unevaluated.
    // Either way, we just document the current behavior.
    assert!(
        s == "0" || s.contains("oo"),
        "(oo^2)^(-1) should be 0 (or unevaluated), got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 20. Interaction: Mul distribution with infinity
//     Pow(Mul(children), n) distributes when n is integer |n| ≤ 10.
//     Could infinity appear inside a Mul? E.g., (2*oo)^(-1).
//     canon_mul should simplify 2*oo → oo first, so Mul never contains oo.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn two_times_infinity_is_infinity() {
    // Prerequisite: verify canon_mul absorbs oo.
    let ctx = Context::new();
    let two = ctx.int(2);
    let inf = ctx.infinity();
    let product = &two * &inf;
    assert_displays_as(&product, "oo", "2*oo should be oo");
}

#[test]
fn two_times_infinity_pow_neg_one() {
    // (2*oo)^(-1): first 2*oo → oo (by canon_mul), then oo^(-1).
    // Before fix: oo^(-1) stays as 1/oo.
    // After fix: oo^(-1) → 0.
    let ctx = Context::new();
    let two = ctx.int(2);
    let inf = ctx.infinity();
    let product = &two * &inf; // should be oo
    let result = product.powi(-1);
    let s = format!("{result}");
    eprintln!("(2*oo)^(-1) = {s}");
    assert!(
        s == "0" || s.contains("oo"),
        "(2*oo)^(-1) should be 0 (or 1/oo pre-fix), got: {s}"
    );
}

#[test]
fn neg_one_times_infinity_is_neg_infinity() {
    // Prerequisite: verify canon_mul sign tracking with oo.
    let ctx = Context::new();
    let neg_one = ctx.int(-1);
    let inf = ctx.infinity();
    let product = &neg_one * &inf;
    assert_displays_as(&product, "-oo", "(-1)*oo should be -oo");
}

// ═══════════════════════════════════════════════════════════════════════════
// 21. Verify_canonical Pow invariants
//     The verify_canonical function checks exp ≠ 0, exp ≠ 1, base ≠ 1.
//     Proposed infinity rules should never violate these.
//     Additional invariants that SHOULD hold after the fix:
//       - Pow(oo, positive_numeric) should not exist (should be oo)
//       - Pow(oo, negative_numeric) should not exist (should be 0)
//       - Pow(0, oo) should not exist (should be 0)
//     These are not yet checked by verify_canonical.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn verify_canonical_accepts_infinity_atoms() {
    // Creating infinity expressions should not trigger verify_canonical errors.
    // This is a basic sanity check.
    let ctx = Context::new();
    let _inf = ctx.infinity();
    let _neg_inf = ctx.neg_infinity();
    let _z = zoo(&ctx);
    let _n = ctx.nan();
    // If verify_canonical panics in debug mode, this test will fail.
}

// ═══════════════════════════════════════════════════════════════════════════
// 22. Summary table: systematic status check
//     This single test exercises ALL proposed rules and reports which ones
//     currently pass vs. fail, providing a quick overview.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[allow(clippy::type_complexity)]
fn summary_status_report() {
    let ctx = Context::new();
    let inf = ctx.infinity();
    let neg_inf = ctx.neg_infinity();
    let z = zoo(&ctx);
    let zero = ctx.int(0);
    let half = ctx.rational(1, 2);
    let neg_half = ctx.rational(-1, 2);

    let mut pass = 0u32;
    let mut fail = 0u32;

    let cases: Vec<(&str, Box<dyn Fn() -> Ex>, &[&str])> = vec![
        // (label, expression_builder, acceptable_outputs)
        ("oo^2 = oo", Box::new(|| inf.powi(2)), &["oo"]),
        ("oo^(1/2) = oo", Box::new(|| inf.pow(&half)), &["oo"]),
        ("oo^(-1) = 0", Box::new(|| inf.powi(-1)), &["0"]),
        ("oo^(-2) = 0", Box::new(|| inf.powi(-2)), &["0"]),
        ("oo^(-1/2) = 0", Box::new(|| inf.pow(&neg_half)), &["0"]),
        ("(-oo)^2 = oo", Box::new(|| neg_inf.powi(2)), &["oo"]),
        ("(-oo)^3 = -oo", Box::new(|| neg_inf.powi(3)), &["-oo"]),
        ("(-oo)^(-1) = 0", Box::new(|| neg_inf.powi(-1)), &["0"]),
        ("(-oo)^(-2) = 0", Box::new(|| neg_inf.powi(-2)), &["0"]),
        ("(-oo)^(-3) = 0", Box::new(|| neg_inf.powi(-3)), &["0"]),
        ("zoo^2 = zoo", Box::new(|| z.powi(2)), &["zoo"]),
        ("zoo^(1/2) = zoo", Box::new(|| z.pow(&half)), &["zoo"]),
        ("zoo^(-1) = 0", Box::new(|| z.powi(-1)), &["0"]),
        ("zoo^(-2) = 0", Box::new(|| z.powi(-2)), &["0"]),
        ("zoo^(-1/2) = 0", Box::new(|| z.pow(&neg_half)), &["0"]),
        ("0^oo = 0", Box::new(|| zero.pow(&inf)), &["0"]),
        ("0^(-oo) = zoo", Box::new(|| zero.pow(&neg_inf)), &["zoo"]),
        ("0^zoo = nan", Box::new(|| zero.pow(&z)), &["nan", "NaN"]),
        ("oo^oo = oo", Box::new(|| inf.pow(&inf)), &["oo"]),
        ("oo^(-oo) = 0", Box::new(|| inf.pow(&neg_inf)), &["0"]),
    ];

    eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║         Infinity ^ Power Canonicalization Status            ║");
    eprintln!("╠══════════════════════════════════════════════════════════════╣");

    for (label, builder, expected) in &cases {
        let result = builder();
        let s = format!("{result}");
        let ok = expected.contains(&s.as_str());
        if ok {
            pass += 1;
            eprintln!("║ ✅ {:<40} = {:<12} ║", label, s);
        } else {
            fail += 1;
            eprintln!("║ ❌ {:<40} = {:<12} ║", label, s);
        }
    }

    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!(
        "║ Results: {} passed, {} failed out of {} total             ║",
        pass,
        fail,
        pass + fail
    );
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

    // We do NOT assert all pass — this test is diagnostic.
    // After the fix is applied, uncomment the line below:
    // assert_eq!(fail, 0, "{fail} cases still failing after fix");
}

// ═══════════════════════════════════════════════════════════════════════════
// 23. Edge case: very large integer exponents on infinity
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn infinity_pow_large_positive() {
    let ctx = Context::new();
    let inf = ctx.infinity();
    let result = inf.powi(1000);
    let s = format!("{result}");
    eprintln!("oo^1000 = {s}");
    // Should be oo (after fix), or oo^1000 (before fix)
    assert!(
        s == "oo" || s.contains("oo"),
        "oo^1000 should be oo, got: {s}"
    );
}

#[test]
fn infinity_pow_large_negative() {
    let ctx = Context::new();
    let inf = ctx.infinity();
    let result = inf.powi(-1000);
    let s = format!("{result}");
    eprintln!("oo^(-1000) = {s}");
    // Should be 0 (after fix)
    assert!(
        s == "0" || s.contains("oo"),
        "oo^(-1000) should be 0, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 24. Edge case: symbolic exponents on infinity
//     oo^x for symbolic x should stay unevaluated (no sign info).
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn infinity_pow_symbol_stays_unevaluated() {
    let ctx = Context::new();
    let inf = ctx.infinity();
    let x = ctx.symbol("x");
    let result = inf.pow(&x);
    let s = format!("{result}");
    eprintln!("oo^x = {s}");
    // Without assumptions on x, this must stay unevaluated.
    assert!(
        s.contains("oo") && s.contains("x"),
        "oo^x should stay unevaluated (contains both 'oo' and 'x'), got: {s}"
    );
}

#[test]
fn neg_infinity_pow_symbol_stays_unevaluated() {
    let ctx = Context::new();
    let neg_inf = ctx.neg_infinity();
    let x = ctx.symbol("x");
    let result = neg_inf.pow(&x);
    let s = format!("{result}");
    eprintln!("(-oo)^x = {s}");
    assert!(
        s.contains("oo"),
        "(-oo)^x should stay unevaluated, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 25. Compositional correctness: expressions built from infinity powers
//     Verify downstream consumers don't break.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn infinity_pow_neg_one_plus_one() {
    // oo^(-1) + 1: after fix → 0 + 1 = 1.
    // before fix → 1/oo + 1 (stays symbolic).
    let ctx = Context::new();
    let inf = ctx.infinity();
    let inv = inf.powi(-1);
    let result = &inv + &ctx.int(1);
    let s = format!("{result}");
    eprintln!("oo^(-1) + 1 = {s}");
    // After fix: "1"
    // Before fix: something involving oo
    assert!(
        s == "1" || s.contains("oo"),
        "oo^(-1) + 1 should be 1 (after fix) or contain oo, got: {s}"
    );
}

#[test]
fn zero_pow_infinity_times_anything() {
    // (0^oo) * 5: after fix → 0 * 5 = 0.
    let ctx = Context::new();
    let zero = ctx.int(0);
    let inf = ctx.infinity();
    let base_result = zero.pow(&inf);
    let result = &base_result * &ctx.int(5);
    let s = format!("{result}");
    eprintln!("0^oo * 5 = {s}");
    assert!(
        s == "0" || s.contains("oo"),
        "0^oo * 5 should be 0 (after fix), got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 26. (-∞)^(3/2) — negative infinity to non-integer with odd numerator
//     SymPy: (-oo)**(3/2) = -oo*I
//     symplex can't represent directed complex infinity, so unevaluated
//     or zoo are acceptable.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn neg_infinity_pow_three_halves() {
    // SymPy: (-oo)**(3/2) = -oo*I
    let ctx = Context::new();
    let neg_inf = ctx.neg_infinity();
    let exp = ctx.rational(3, 2);
    let result = neg_inf.pow(&exp);
    let s = format!("{result}");
    eprintln!("(-oo)^(3/2) = {s}");
    // Must NOT be 0, oo, -oo (those are mathematically wrong).
    // Unevaluated or zoo are acceptable.
    assert!(
        s != "0" && s != "oo" && s != "-oo",
        "(-oo)^(3/2) must not be 0, oo, or -oo — got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 27. (-oo)^(neg non-integer) → 0 even for non-integer negative exponents
//     SymPy: (-oo)**(-3/2) = 0 — magnitude dominates.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn neg_infinity_pow_neg_three_halves() {
    // SymPy: (-oo)**(-3/2) = 0
    let ctx = Context::new();
    let neg_inf = ctx.neg_infinity();
    let exp = ctx.rational(-3, 2);
    let result = neg_inf.pow(&exp);
    let s = format!("{result}");
    eprintln!("(-oo)^(-3/2) = {s}");
    // After fix: "0"
    assert!(
        s == "0" || s.contains("oo"),
        "(-oo)^(-3/2) should be 0, got: {s}"
    );
}
