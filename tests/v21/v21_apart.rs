//! 0.22 — apart regressions from the 0.21 differential audit.
//!
//! The numeric "log-to-real" route of `partial_fractions` evaluated the
//! poles of an irreducible denominator to `f64`, ran `nsimplify`, and
//! accepted whatever came back — when nothing matched, that was the
//! 12-digit rational the float had been rounded to, emitted as an *exact*
//! pole (`x − 310631195679/250000000000`).  Every `nsimplify` candidate is
//! now verified exactly against the polynomials (root, quadratic factor,
//! residue / numerator) and rejected otherwise; the pre-existing exact
//! symbolic fallback then takes over.
//!
//! Reference values cite sympy 1.14 (`symplex/.venv/bin/python`) by the
//! call that produced them (float literals are the shortest round-trip
//! `f64` of the quoted 17-digit sympy value, as clippy requires).  Nothing
//! below was computed by hand.

use symplex::prelude::*;

/// `partial_fractions` must agree with the input to f64 precision at `pt`
/// (`reference` is sympy's exact value of the input there).
fn assert_exact_at(orig: &Ex, pf: &Ex, x: &Ex, pt: &Ex, reference: f64) {
    let o = orig
        .subs(x, pt)
        .eval_f64()
        .unwrap_or_else(|err| panic!("{orig} at {pt} did not evaluate: {err}"));
    let p = pf
        .subs(x, pt)
        .eval_f64()
        .unwrap_or_else(|err| panic!("{pf} at {pt} did not evaluate: {err}"));
    let scale = reference.abs();
    assert!(
        (o - reference).abs() <= 1e-15 * scale,
        "input {orig} at {pt}: got {o}, sympy says {reference}"
    );
    assert!(
        (p - reference).abs() <= 1e-15 * scale,
        "partial_fractions at {pt}: got {p}, sympy says {reference} (relative error {:e})",
        (p - reference).abs() / scale
    );
}

/// The longest run of decimal digits in the display string — a
/// float-rounded "exact" rational shows up as a 12-digit denominator.
fn longest_digit_run(s: &str) -> usize {
    let mut best = 0;
    let mut cur = 0;
    for c in s.chars() {
        if c.is_ascii_digit() {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }
    best
}

#[test]
fn apart_irrational_cubic_poles_are_not_rounded_floats() {
    // 1/(-91/64·x³ + 3x − 1): three real irrational poles.
    // sympy: apart(1/(Rational(-91,64)*x**3 + 3*x - 1), x) == -64/(91*x**3 - 192*x + 64)
    //   (unchanged); f(3) = -64/1945 = -0.032904884318766067,
    //   f(5/2) = -512/8047 = -0.063626196097924692,
    //   f(-7) = 64/29805 = 0.0021472907230330481.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = ctx.int(1) / (ctx.rational(-91, 64) * x.powi(3) + ctx.int(3) * &x - ctx.int(1));
    let pf = e.partial_fractions(&x);
    let s = pf.to_string();
    assert!(
        longest_digit_run(&s) <= 6,
        "partial_fractions emitted a float-rounded rational (integer literal > 10^6): {s}"
    );
    assert_exact_at(&e, &pf, &x, &ctx.int(3), -0.032904884318766064);
    assert_exact_at(&e, &pf, &x, &ctx.rational(5, 2), -0.06362619609792469);
    assert_exact_at(&e, &pf, &x, &ctx.int(-7), 0.002147290723033048);
}

#[test]
fn apart_x_cubed_minus_two_is_exact() {
    // sympy: apart(1/(x**3 - 2), x) == 1/(x**3 - 2) (unchanged over Q);
    //   (1/(x**3-2)).subs(x, 3) == 1/25 = 0.04,
    //   (1/(x**3-2)).subs(x, Rational(5,2)) == 8/109 = 0.073394495412844037
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = ctx.int(1) / (x.powi(3) - ctx.int(2));
    let pf = e.partial_fractions(&x);
    let s = pf.to_string();
    assert!(
        longest_digit_run(&s) <= 6,
        "partial_fractions emitted a float-rounded rational: {s}"
    );
    assert_exact_at(&e, &pf, &x, &ctx.int(3), 0.04);
    assert_exact_at(&e, &pf, &x, &ctx.rational(5, 2), 0.07339449541284404);
}

// ── Genuinely rational / quadratic-surd poles still decompose ───────────────

#[test]
fn apart_difference_of_squares_still_decomposes() {
    // sympy: apart(1/(x**2 - 1), x) == -1/(2*(x + 1)) + 1/(2*(x - 1))
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = ctx.int(1) / (x.powi(2) - ctx.int(1));
    let pf = e.partial_fractions(&x);
    let expected = ctx.rational(1, 2) / (&x - ctx.int(1)) - ctx.rational(1, 2) / (&x + ctx.int(1));
    assert_ne!(pf.id(), e.id(), "1/(x^2 - 1) should be decomposed");
    assert_eq!(
        pf.equals(&expected),
        Some(true),
        "got {pf}, sympy says {expected}"
    );
}

#[test]
fn apart_with_zero_pole_still_decomposes() {
    // sympy: apart((x + 1)/(x**3 - x), x) == 1/(x - 1) - 1/x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = (&x + ctx.int(1)) / (x.powi(3) - &x);
    let pf = e.partial_fractions(&x);
    let expected = ctx.int(1) / (&x - ctx.int(1)) - ctx.int(1) / &x;
    assert_ne!(pf.id(), e.id(), "(x + 1)/(x^3 - x) should be decomposed");
    assert_eq!(
        pf.equals(&expected),
        Some(true),
        "got {pf}, sympy says {expected}"
    );
}

#[test]
fn apart_irreducible_quadratic_unchanged() {
    // sympy: apart(1/(x**2 + 1), x) == 1/(x**2 + 1)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = ctx.int(1) / (x.powi(2) + ctx.int(1));
    let pf = e.partial_fractions(&x);
    assert_eq!(
        pf.equals(&e),
        Some(true),
        "1/(x^2 + 1) should stay: got {pf}"
    );
}

#[test]
fn apart_x4_plus_1_verified_sqrt2_factors_still_decompose() {
    // The legitimate use of the numeric route: x⁴ + 1 = (x² + √2x + 1)(x² − √2x + 1).
    // The nsimplified √2 coefficients verify exactly, so the decomposition
    // is kept and must be exact.
    // sympy: (1/(x**4 + 1)).subs(x, 3) == 1/82 = 0.012195121951219512
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = ctx.int(1) / (x.powi(4) + ctx.int(1));
    let pf = e.partial_fractions(&x);
    let s = pf.to_string();
    assert_ne!(pf.id(), e.id(), "1/(x^4 + 1) should decompose over Q(√2)");
    assert!(s.contains("sqrt(2)"), "expected √2 factors: {s}");
    assert!(
        longest_digit_run(&s) <= 6,
        "partial_fractions emitted a float-rounded rational: {s}"
    );
    assert_exact_at(&e, &pf, &x, &ctx.int(3), 0.012195121951219513);
}
