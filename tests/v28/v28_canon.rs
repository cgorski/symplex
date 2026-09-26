//! Canonical forms of rational powers at the digit guard.

use symplex::prelude::*;

/// Before: `2/9991999999^589.99919999` overflowed the stack while parsing
/// (the nightly `fuzz_parser` crash of 2026-09-26).  `n^(−a/b)` is
/// rationalised as `n^(−k−1)·n^((b−s)/b)`; when the digit guard refuses to
/// fold `n^(k+1)` (`9991999999^590` has ~5,900 digits), both factors stay
/// powers of `n`, the product adds the exponents back to `−a/b`, and the
/// rewrite recursed without end.  It now rationalises only when the
/// coefficient folds.
#[test]
fn a_negative_rational_power_beyond_the_digit_guard_is_left_alone() {
    let ctx = Context::new();
    for s in ["2/9991999999^589.99919999", "9991999999^(-58999/100)"] {
        let e = ctx.parse(s).unwrap();
        let back = ctx.parse(&e.to_string()).unwrap();
        assert_eq!(back, e, "{s}");
    }
    // The crash input itself (control bytes removed): parses, and its
    // display parses again, as `fuzz_parser` requires.
    let s = "49999^584/5599/9991999999^589.99919999/99919999^584/5959/9999/99919999^584/5959/99";
    let e = ctx.parse(s).unwrap();
    assert!(ctx.parse(&e.to_string()).is_ok());
    // Within the guard the denominator is still rationalised: 2^(-1/2) = √2/2.
    assert_eq!(
        ctx.parse("2^(-1/2)").unwrap(),
        ctx.parse("sqrt(2)/2").unwrap()
    );
}

/// Before: `(10¹⁰⁰⁰ + 7)^(1999/2)` computed the million-digit integer part
/// `n⁹⁹⁹` of `n^999·√n` (15.9 s in a release build) — the digit guard that
/// stops every other power was not consulted.  A power whose integer part
/// would exceed the guard now keeps its exponent, as one with an exponent
/// beyond `max_pow_exponent` does.
#[test]
fn a_rational_power_whose_integer_part_exceeds_the_digit_guard_is_not_expanded() {
    let ctx = Context::new();
    let n = ctx.parse("10^999 + 7").unwrap();
    let e = n.pow(&ctx.rational(1999, 2));
    assert_eq!(e.to_string(), format!("{n}^(1999/2)"));
    // Small integer parts still split off: 15^(3/2) = 15·√15, 12^(5/2) = 288·√3.
    assert_eq!(
        ctx.parse("15^(3/2)").unwrap(),
        ctx.parse("15*sqrt(15)").unwrap()
    );
    assert_eq!(
        ctx.parse("12^(5/2)").unwrap(),
        ctx.parse("288*sqrt(3)").unwrap()
    );
}
