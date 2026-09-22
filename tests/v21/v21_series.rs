//! 0.22 — series regressions from the 0.21 differential audit.
//!
//! The series engine had no rule for `|·|` and fell back to Taylor
//! coefficients by differentiation, where `d/dx |g| = g'·sign(g)` is `0` at
//! every zero of `g` — so `series(|x²|)` was the constant `0`.  The same
//! fallback made `series(x^(5/2), x, 0, 3)` return `0` while order 6
//! correctly refused.  `|g|` is now expanded as `±g` when the sign of `g`
//! near the point is known (even leading exponent, or one-sided), odd
//! leading exponents are cross-checked from both sides (`cos|x|` is fine,
//! `e^|x|` is not), and Puiseux exponents are a definite refusal at every
//! order.
//!
//! Reference values cite sympy 1.14 (`symplex/.venv/bin/python`) by the
//! call that produced them (float literals are the shortest round-trip
//! `f64` of the quoted 17-digit sympy value, as clippy requires).  Nothing
//! below was computed by hand.

use symplex::prelude::*;

/// `series(e, x, 0, 6)` must be a genuine expansion equal to `expected`
/// (structurally after subtraction, and numerically at `x = 1/10`).
fn assert_series_is(e: &Ex, x: &Ex, expected: &Ex, at_tenth: f64) {
    let ctx = e.context();
    let s = e.series(x, &ctx.int(0), 6);
    assert!(
        !s.has_unevaluated(),
        "series({e}) came back unevaluated: {s}"
    );
    assert_eq!(
        s.equals(expected),
        Some(true),
        "series({e}, x, 0, 6) = {s}, sympy says {expected}"
    );
    let v = s
        .subs(x, &ctx.rational(1, 10))
        .eval_f64()
        .unwrap_or_else(|err| panic!("{s} at 1/10 did not evaluate: {err}"));
    assert!(
        (v - at_tenth).abs() <= 1e-12,
        "series({e}) = {s} at 1/10 gives {v}, sympy says {at_tenth}"
    );
}

// ── |g| with even leading exponent: |g| = ±g ────────────────────────────────

#[test]
fn series_abs_x_squared() {
    // sympy: series(Abs(x**2), x, 0, 6) == x**2; at x = 1/10: 1/100
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_series_is(&x.powi(2).abs(), &x, &x.powi(2), 0.01);
}

#[test]
fn series_cos_abs_x_is_cos_series() {
    // Odd leading exponent inside, but cos is even: both one-sided
    // expansions agree.
    // sympy: series(cos(Abs(x)), x, 0, 6) == x**4/24 - x**2/2 + 1;
    //        at x = 1/10: 238801/240000 = 0.99500416666666667
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expected = x.powi(4) * ctx.rational(1, 24) - x.powi(2) * ctx.rational(1, 2) + ctx.int(1);
    assert_series_is(&x.abs().cos(), &x, &expected, 0.9950041666666667);
}

#[test]
fn series_abs_atan_squared() {
    // sympy: series(Abs(atan(x)**2), x, 0, 6) == -2*x**4/3 + x**2;
    //        at x = 1/10: 149/15000 = 0.0099333333333333333
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expected = x.powi(2) - x.powi(4) * ctx.rational(2, 3);
    assert_series_is(&x.atan().powi(2).abs(), &x, &expected, 0.009933333333333334);
}

#[test]
fn series_abs_x_minus_one_at_zero() {
    // k = 0 with a negative constant term: |x − 1| = 1 − x near 0.
    // sympy: series(Abs(x - 1), x, 0, 6) == 1 - x; at x = 1/10: 9/10
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expected = ctx.int(1) - &x;
    assert_series_is(&(&x - ctx.int(1)).abs(), &x, &expected, 0.9);
}

#[test]
fn series_abs_x_squared_via_even_power_of_abs() {
    // |x|² has a two-sided expansion even though |x| does not.
    // sympy: series(Abs(x)**2, x, 0, 6) == x**2; at x = 1/10: 1/100
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_series_is(&x.abs().powi(2), &x, &x.powi(2), 0.01);
}

// ── |g| with odd leading exponent: no two-sided expansion ───────────────────

#[test]
fn series_abs_x_is_unevaluated() {
    // sympy's default is the right-sided series x; two-sided there is none.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = x.abs().series(&x, &ctx.int(0), 6);
    assert!(
        s.has_unevaluated(),
        "series(|x|) must not be a polynomial: {s}"
    );
    assert!(!s.is_zero_structural(), "series(|x|) must not be 0");
}

#[test]
fn series_abs_sin_x_is_unevaluated() {
    // sympy (right-sided) gives x - x**3/6 + x**5/120; from the left the
    // sign flips, so no two-sided expansion exists.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = x.sin().abs().series(&x, &ctx.int(0), 6);
    assert!(
        s.has_unevaluated(),
        "series(|sin x|) must not be a polynomial: {s}"
    );
}

#[test]
fn series_exp_abs_x_is_unevaluated() {
    // sympy (right-sided) gives 1 + x + x**2/2 + …; e^|x| has a kink at 0.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = x.abs().exp().series(&x, &ctx.int(0), 6);
    assert!(
        s.has_unevaluated(),
        "series(exp|x|) must not be a polynomial: {s}"
    );
    assert!(
        !s.equals(&ctx.int(1)).unwrap_or(false),
        "series(exp|x|) must not be the constant 1"
    );
}

#[test]
fn series_abs_x_at_infinity_is_one_sided() {
    // sympy: series(Abs(x), x, oo, 3) == x; series(Abs(x), x, -oo, 3) == -x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let plus = x.abs().series(&x, &ctx.infinity(), 3);
    assert_eq!(
        plus.id(),
        x.id(),
        "series(|x|, x, oo) should be x; got {plus}"
    );
    let minus = x.abs().series(&x, &ctx.neg_infinity(), 3);
    assert_eq!(
        minus.id(),
        (-&x).id(),
        "series(|x|, x, -oo) should be -x; got {minus}"
    );
}

// ── Puiseux exponents: refused at every order, never a silent 0 ─────────────

#[test]
fn series_x_to_5_2_unevaluated_at_order_3() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.pow(&ctx.rational(5, 2));
    let s = e.series(&x, &ctx.int(0), 3);
    assert!(
        !s.is_zero_structural(),
        "series(x^(5/2), x, 0, 3) must not be 0"
    );
    assert!(
        s.has_unevaluated(),
        "series(x^(5/2), x, 0, 3) should be unevaluated: {s}"
    );
}

#[test]
fn series_x_to_5_2_unevaluated_at_order_6() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.pow(&ctx.rational(5, 2));
    let s = e.series(&x, &ctx.int(0), 6);
    assert!(
        !s.is_zero_structural(),
        "series(x^(5/2), x, 0, 6) must not be 0"
    );
    assert!(
        s.has_unevaluated(),
        "series(x^(5/2), x, 0, 6) should be unevaluated: {s}"
    );
}

#[test]
fn series_sqrt_x_times_x_not_zero_at_order_3() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.sqrt() * &x;
    let s = e.series(&x, &ctx.int(0), 3);
    assert!(
        !s.is_zero_structural(),
        "series(sqrt(x)*x, x, 0, 3) must not be 0"
    );
    assert!(
        s.has_unevaluated(),
        "series(sqrt(x)*x, x, 0, 3) should be unevaluated: {s}"
    );
}

#[test]
fn series_sqrt_of_even_valuation_still_expands() {
    // v = 4, α = 1/2: v/denom(α) = 2 is even, so no |x| appears.
    // sympy: series(sqrt(x**6 + x**4), x, 0, 6) == x**4/2 + x**2;
    //        at x = 1/10: 201/20000 = 0.01005
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = (x.powi(4) + x.powi(6)).sqrt();
    let expected = x.powi(2) + x.powi(4) * ctx.rational(1, 2);
    assert_series_is(&e, &x, &expected, 0.01005);
}
