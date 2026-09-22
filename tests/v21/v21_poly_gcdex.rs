//! 0.22.2 — `Poly::extended_gcd` / `Ex::poly_gcdex` through the primitive
//! PRS in `ℤ[x]` (the `Field::poly_extended_gcd` hook), no longer Euclid
//! over ℚ.  Reference values: SymPy 1.14 `gcdex(f, g, x)`, which returns
//! `(s, t, h)` with `s·f + t·g = h` and `h` monic — the crate's
//! `ExtendedGcd { x, y, gcd }`.  Equality with the Euclidean reference on
//! random inputs is pinned in-crate (`zpoly::tests`).

use std::time::Instant;

use symplex::prelude::*;

fn assert_same(label: &str, got: &Ex, want: &Ex) {
    assert_eq!(
        (got - want).expand(),
        got.context().int(0),
        "{label}: got {got}, want {want}"
    );
}

/// SymPy: `gcdex(x**4 - 2*x**3 - 6*x**2 + 12*x + 15, x**3 + x**2 - 4*x - 4, x)`
/// = `(3/5 - x/5, x**2/5 - 6*x/5 + 2, x + 1)` (the SymPy documentation example).
#[test]
fn gcdex_matches_sympy_on_the_documentation_example() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(4) - 2 * x.powi(3) - 6 * x.powi(2) + 12 * &x + 15;
    let g = x.powi(3) + x.powi(2) - 4 * &x - 4;
    let e = f.poly_gcdex(&g, &x).unwrap();
    assert_same("gcd", &e.gcd, &(&x + 1));
    assert_same("x", &e.x, &(ctx.rational(3, 5) - &x / 5));
    assert_same("y", &e.y, &(x.powi(2) / 5 - ctx.rational(6, 5) * &x + 2));
}

/// Rational coefficients and `deg f < deg g`: SymPy
/// `gcdex(x**2/2 - Rational(1,2), 2*x**3/3 + Rational(2,3), x)` =
/// `(-2*x, 3/2, x + 1)`.
#[test]
fn gcdex_with_rational_coefficients_and_lower_degree_first() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(2) / 2 - ctx.rational(1, 2);
    let g = ctx.rational(2, 3) * x.powi(3) + ctx.rational(2, 3);
    let e = f.poly_gcdex(&g, &x).unwrap();
    assert_same("gcd", &e.gcd, &(&x + 1));
    assert_same("x", &e.x, &(-2 * &x));
    assert_same("y", &e.y, &ctx.rational(3, 2));
}

/// The 0.21 audit's timing input: degree 11 × 12 with 40-digit
/// coefficients took 4.3 s through Euclid over ℚ (release build) against
/// 5 ms for `gcd`.  The Bézout identity must hold exactly and the call must
/// stay well under a second even in a debug build.
#[test]
fn gcdex_of_forty_digit_coefficients_is_fast_and_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let big = |k: i64| ctx.int(10).pow(&ctx.int(40)) * k + (k * k - 7);
    let common = x.powi(3) - 2 * &x + 5;
    let a = (0..9).fold(ctx.int(0), |acc, k| acc + big(k + 1) * x.powi(k));
    let b = (0..10).fold(ctx.int(0), |acc, k| acc + big(3 * k - 11) * x.powi(k));
    let (a, b) = ((a * &common).expand(), (b * &common).expand());
    let started = Instant::now();
    let e = a.poly_gcdex(&b, &x).unwrap();
    let elapsed = started.elapsed();
    let bezout = (&e.x * &a + &e.y * &b).expand();
    assert_same("bezout", &bezout, &e.gcd);
    assert_eq!(
        e.gcd.poly_rem(&common, &x).unwrap(),
        ctx.int(0),
        "{}",
        e.gcd
    );
    assert!(elapsed.as_millis() < 1500, "gcdex took {elapsed:?}");
}
