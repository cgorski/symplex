//! After 0.28.0 — the integrator's self-check decides on evidence, and the
//! Lazard–Rioboo–Trager remainder sequence is a subresultant PRS.
//!
//! - The numeric check of a candidate antiderivative counted "`F′`
//!   evaluates at no sample point" as a pass.  Since 0.26 `evalf` refuses a
//!   quotient by a difference that cancels to 0 (`PrecisionExhausted`)
//!   instead of returning noise, so such a candidate passed unchecked.
//! - `∫ atan(√x − x⁹) dx` spent 24 s (release) in Euclid's remainder
//!   sequence over `ℚ(t)` for its degree-36 resultant.
//!
//! Each antiderivative is judged by `F′ = f` at sample points (complex
//! values allowed, as the Rubi harness does).

use std::time::{Duration, Instant};

use symplex::prelude::*;

/// `F′ = f` at `x = p/q` for each point, relatively to `1e-10`.
fn assert_derivative_matches(f: &Ex, big_f: &Ex, x: &Ex, ctx: &Context, points: &[(i64, i64)]) {
    let df = big_f.diff(x);
    for &(p, q) in points {
        let v = ctx.rational(p, q);
        let a = df.subs(x, &v).eval_complex64().unwrap();
        let b = f.subs(x, &v).eval_complex64().unwrap();
        assert!(
            (a - b).norm() <= 1e-10 * b.norm().max(1.0),
            "∫ {f} dx = {big_f}: F′ = {a} but f = {b} at x = {p}/{q}"
        );
    }
}

#[test]
fn a_candidate_with_a_vanishing_denominator_is_not_returned() {
    // `k = sin²1 + cos²1 − 1` is 0 without being so structurally (SymPy
    // 1.14: `simplify(sin(1)**2 + cos(1)**2 - 1)` → `0`), so the integrand
    // is `x`.  Up to 0.28.0 the result was `(x − ln|k·x + 1|/k)/k`: a quotient
    // by `k` that evaluates nowhere, which the self-check passed because no
    // sample point evaluated.  Rejecting it exposed a second route that
    // divided by the same hidden zero (`u = k·x + 1`, `du = k dx`) and
    // returned `zoo`.  Now the integral stays unevaluated — or, should a
    // later version see that `k = 0`, is a closed form that evaluates.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("x/((sin(1)^2 + cos(1)^2 - 1)*x + 1)").unwrap();
    let big_f = f.integrate(&x);
    if !big_f.has_unevaluated() {
        let third = ctx.rational(1, 3);
        let value = big_f.subs(&x, &third).eval_complex64();
        assert!(value.is_ok(), "∫ {f} dx = {big_f} does not evaluate at 1/3");
        // f(1/3) = 1/3 (SymPy 1.14: `simplify(f.subs(x, Rational(1, 3)))`).
        let slope = big_f.diff(&x).subs(&x, &third).eval_f64().unwrap();
        assert!(
            (slope - 1.0 / 3.0).abs() < 1e-12,
            "{big_f}: F′(1/3) = {slope}"
        );
    }
}

#[test]
fn atan_of_sqrt_x_minus_x9_integrates_in_seconds() {
    // Up to 0.28.0: 28 s in a release build (24 s of it in the remainder
    // sequence of the degree-36 denominator, `lrt_prs`) and 71 s in a debug
    // build; now 0.2 s and 2.3 s.  The answer is unchanged, character for
    // character: `x·atan(√x − x⁹)` minus a `RootSum` over the degree-36
    // resultant.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("atan(sqrt(x) - x^9)").unwrap();
    let t = Instant::now();
    let big_f = f.integrate(&x);
    let elapsed = t.elapsed();
    assert!(!big_f.has_unevaluated(), "∫ {f} dx = {big_f}");
    assert_derivative_matches(&f, &big_f, &x, &ctx, &[(1, 3), (7, 5), (-5, 7)]);
    // Generous for a loaded machine.
    assert!(
        elapsed < Duration::from_secs(20),
        "∫ {f} dx took {elapsed:?}"
    );
}

#[test]
fn smaller_members_of_the_family_keep_their_closed_forms() {
    // `∫ atan(√x − x³) dx` (resultant of degree 12, found by
    // `fuzz_integrate` in 0.25) and `x⁵` (degree 20): the subresultant
    // sequence gives the same log arguments as Euclid's did.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in ["atan(sqrt(x) - x^3)", "atan(sqrt(x) - x^5)"] {
        let f = ctx.parse(src).unwrap();
        let big_f = f.integrate(&x);
        assert!(!big_f.has_unevaluated(), "∫ {f} dx = {big_f}");
        assert_derivative_matches(&f, &big_f, &x, &ctx, &[(1, 3), (7, 5), (-5, 7)]);
    }
}
