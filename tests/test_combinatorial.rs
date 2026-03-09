//! Tests for combinatorial functions (Wave R).
//!
//! Each test verifies exact evaluation of a named combinatorial function
//! at known integer arguments.

use symplex::prelude::*;

// ─── Helper ────────────────────────────────────────────────────────────────

fn check(expr: &Ex, expected: &str) {
    let s = format!("{expr}");
    assert_eq!(s, expected, "expected '{expected}', got '{s}'");
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Double factorial  n!!
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factorial2_zero() {
    let __ctx = Context::new();
    check(&__ctx.int(0).factorial2().eval(), "1");
}

#[test]
fn factorial2_one() {
    let __ctx = Context::new();
    check(&__ctx.int(1).factorial2().eval(), "1");
}

#[test]
fn factorial2_neg_one() {
    let __ctx = Context::new();
    check(&__ctx.int(-1).factorial2().eval(), "1");
}

#[test]
fn factorial2_small_odd() {
    let __ctx = Context::new();
    // 5!! = 5 * 3 * 1 = 15
    check(&__ctx.int(5).factorial2().eval(), "15");
}

#[test]
fn factorial2_small_even() {
    let __ctx = Context::new();
    // 6!! = 6 * 4 * 2 = 48
    check(&__ctx.int(6).factorial2().eval(), "48");
}

#[test]
fn factorial2_ten() {
    let __ctx = Context::new();
    // 10!! = 10 * 8 * 6 * 4 * 2 = 3840
    check(&__ctx.int(10).factorial2().eval(), "3840");
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Subfactorial  !n  (derangements)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn subfactorial_known_values() {
    let __ctx = Context::new();
    check(&__ctx.int(0).subfactorial().eval(), "1");
    check(&__ctx.int(1).subfactorial().eval(), "0");
    check(&__ctx.int(2).subfactorial().eval(), "1");
    check(&__ctx.int(3).subfactorial().eval(), "2");
    check(&__ctx.int(4).subfactorial().eval(), "9");
    check(&__ctx.int(5).subfactorial().eval(), "44");
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Rising factorial  (x)_n
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rising_factorial_known_values() {
    let __ctx = Context::new();
    let n0 = __ctx.int(0);
    let n3 = __ctx.int(3);
    let n4 = __ctx.int(4);
    let n5 = __ctx.int(5);

    // (x)_0 = 1 for any x
    check(&n5.rising_factorial(&n0).eval(), "1");

    // (1)_4 = 1 * 2 * 3 * 4 = 24
    check(&__ctx.int(1).rising_factorial(&n4).eval(), "24");

    // (3)_3 = 3 * 4 * 5 = 60
    check(&n3.rising_factorial(&n3).eval(), "60");

    // (5)_3 = 5 * 6 * 7 = 210
    check(&n5.rising_factorial(&n3).eval(), "210");
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Falling factorial  x^(n)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn falling_factorial_known_values() {
    let __ctx = Context::new();
    let n0 = __ctx.int(0);
    let n3 = __ctx.int(3);
    let n5 = __ctx.int(5);

    // x^(0) = 1 for any x
    check(&n5.falling_factorial(&n0).eval(), "1");

    // 5^(3) = 5 * 4 * 3 = 60
    check(&n5.falling_factorial(&n3).eval(), "60");

    // 3^(3) = 3 * 2 * 1 = 6
    check(&n3.falling_factorial(&n3).eval(), "6");

    // 7^(4) = 7 * 6 * 5 * 4 = 840
    check(
        &__ctx.int(7).falling_factorial(&__ctx.int(4)).eval(),
        "840",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Fibonacci  F(n)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fibonacci_known_values() {
    let __ctx = Context::new();
    check(&__ctx.int(0).fibonacci().eval(), "0");
    check(&__ctx.int(1).fibonacci().eval(), "1");
    check(&__ctx.int(2).fibonacci().eval(), "1");
    check(&__ctx.int(3).fibonacci().eval(), "2");
    check(&__ctx.int(4).fibonacci().eval(), "3");
    check(&__ctx.int(5).fibonacci().eval(), "5");
    check(&__ctx.int(10).fibonacci().eval(), "55");
    check(&__ctx.int(20).fibonacci().eval(), "6765");
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Lucas  L(n)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn lucas_known_values() {
    let __ctx = Context::new();
    check(&__ctx.int(0).lucas().eval(), "2");
    check(&__ctx.int(1).lucas().eval(), "1");
    check(&__ctx.int(2).lucas().eval(), "3");
    check(&__ctx.int(3).lucas().eval(), "4");
    check(&__ctx.int(4).lucas().eval(), "7");
    check(&__ctx.int(5).lucas().eval(), "11");
    check(&__ctx.int(10).lucas().eval(), "123");
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Bernoulli  B(n)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bernoulli_known_values() {
    let __ctx = Context::new();
    check(&__ctx.int(0).bernoulli_number().eval(), "1");
    check(&__ctx.int(1).bernoulli_number().eval(), "-1/2");
    check(&__ctx.int(2).bernoulli_number().eval(), "1/6");
    check(&__ctx.int(3).bernoulli_number().eval(), "0");
    check(&__ctx.int(4).bernoulli_number().eval(), "-1/30");
    check(&__ctx.int(6).bernoulli_number().eval(), "1/42");
    check(&__ctx.int(8).bernoulli_number().eval(), "-1/30");
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Harmonic  H(n)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn harmonic_known_values() {
    let __ctx = Context::new();
    check(&__ctx.int(0).harmonic().eval(), "0");
    check(&__ctx.int(1).harmonic().eval(), "1");
    // H(2) = 1 + 1/2 = 3/2
    check(&__ctx.int(2).harmonic().eval(), "3/2");
    // H(3) = 1 + 1/2 + 1/3 = 11/6
    check(&__ctx.int(3).harmonic().eval(), "11/6");
    // H(4) = 1 + 1/2 + 1/3 + 1/4 = 25/12
    check(&__ctx.int(4).harmonic().eval(), "25/12");
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. Catalan  C(n)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn catalan_known_values() {
    let __ctx = Context::new();
    check(&__ctx.int(0).catalan_number().eval(), "1");
    check(&__ctx.int(1).catalan_number().eval(), "1");
    check(&__ctx.int(2).catalan_number().eval(), "2");
    check(&__ctx.int(3).catalan_number().eval(), "5");
    check(&__ctx.int(4).catalan_number().eval(), "14");
    check(&__ctx.int(5).catalan_number().eval(), "42");
    check(&__ctx.int(10).catalan_number().eval(), "16796");
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Bell  B(n)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bell_known_values() {
    let __ctx = Context::new();
    check(&__ctx.int(0).bell().eval(), "1");
    check(&__ctx.int(1).bell().eval(), "1");
    check(&__ctx.int(2).bell().eval(), "2");
    check(&__ctx.int(3).bell().eval(), "5");
    check(&__ctx.int(4).bell().eval(), "15");
    check(&__ctx.int(5).bell().eval(), "52");
    check(&__ctx.int(6).bell().eval(), "203");
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. Euler number  E(n)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn euler_number_known_values() {
    let __ctx = Context::new();
    check(&__ctx.int(0).euler_number().eval(), "1");
    check(&__ctx.int(1).euler_number().eval(), "0");
    check(&__ctx.int(2).euler_number().eval(), "-1");
    check(&__ctx.int(3).euler_number().eval(), "0");
    check(&__ctx.int(4).euler_number().eval(), "5");
    check(&__ctx.int(6).euler_number().eval(), "-61");
    check(&__ctx.int(8).euler_number().eval(), "1385");
}

// ═══════════════════════════════════════════════════════════════════════════
// Symbolic arguments remain unevaluated
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn combinatorial_symbolic_unchanged() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    // These should remain unevaluated since x is symbolic
    let fib = x.fibonacci().eval();
    let s = format!("{fib}");
    assert!(
        s.contains("fibonacci"),
        "expected unevaluated fibonacci, got: {s}"
    );

    let cat = x.catalan_number().eval();
    let s = format!("{cat}");
    assert!(
        s.contains("catalan"),
        "expected unevaluated catalan, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Rising factorial equals factorial for x = 1
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rising_factorial_from_one_is_factorial() {
    let __ctx = Context::new();
    // (1)_n = n!
    let one = __ctx.int(1);
    for n in 0..=7 {
        let rf = one.rising_factorial(&__ctx.int(n)).eval();
        let f = __ctx.int(n).factorial().eval();
        assert_eq!(format!("{rf}"), format!("{f}"), "(1)_{n} should equal {n}!");
    }
}
