//! After 0.28.0 — the integrator's self-check decides on evidence, and the
//! Lazard–Rioboo–Trager remainder sequence is a subresultant PRS.
//!
//! - The numeric check of a candidate antiderivative counted "`F′`
//!   evaluates at no sample point" as a pass.  Since 0.26 `evalf` refuses a
//!   quotient by a difference that cancels to 0 (`PrecisionExhausted`)
//!   instead of returning noise, so such a candidate passed unchecked.
//! - `∫ atan(√x − x⁹) dx` spent 24 s (release) in Euclid's remainder
//!   sequence over `ℚ(t)` for its degree-36 resultant.
//! - Only some routes checked their candidates: by parts, among others,
//!   returned answers that divide by a hidden zero.  Now every stage's
//!   answer faces the check before `integrate` returns it.
//! - `ln|u|` was kept for every `u` without an explicit `i`, also for
//!   `u = x − √(1/2 − √5/2)` (imaginary constant) and `x·√(−a) + a`.
//! - The integrator's dependence test was structural: a `RootOf` whose
//!   polynomial is written in `x` (as `solve` writes it) counted as a
//!   function of `x`.
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
fn a_by_parts_answer_that_divides_by_a_hidden_zero_is_not_returned() {
    // `k = sin²1 + cos²1 − 1` is 0 (SymPy 1.14: `simplify(sin(1)**2 +
    // cos(1)**2 - 1)` → `0`), so the integrand is `x`.  Up to 0.28.0 by
    // parts returned `x·e^{kx}/k − e^{kx}/k²`, which evaluates nowhere
    // (its derivative, simplified, is `x·e^{kx}`): that route never ran the
    // self-check.  Now the answer is unevaluated — or, should a later
    // version see that `k = 0`, a closed form that evaluates.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("x*exp((sin(1)^2 + cos(1)^2 - 1)*x)").unwrap();
    let big_f = f.integrate(&x);
    if !big_f.has_unevaluated() {
        for (p, q) in [(1, 3), (7, 5)] {
            let v = ctx.rational(p, q);
            assert!(
                big_f.subs(&x, &v).eval_complex64().is_ok(),
                "∫ {f} dx = {big_f} does not evaluate at {p}/{q}"
            );
        }
        assert_derivative_matches(&f, &big_f, &x, &ctx, &[(1, 3), (7, 5)]);
    }
}

#[test]
fn ln_abs_is_kept_only_for_an_argument_known_to_be_real() {
    // `√(1/2 − √5/2)` is imaginary (SymPy 1.14:
    // `N(sqrt(Rational(1, 2) - sqrt(5)/2))` → `0.786151377757423*I`), but
    // has no explicit `i`: up to 0.28.0 the answer was `ln|x − c|`, whose
    // derivative is the conjugate of the integrand at every real `x`.  Now
    // `ln(x − c)` (SymPy 1.14: `integrate(1/(x - c), x)` →
    // `log(sqrt(2)*x - sqrt(1 - sqrt(5)))`, the same up to a constant).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("1/(x - sqrt(1/2 - sqrt(5)/2))").unwrap();
    let big_f = f.integrate(&x);
    assert!(!big_f.has_unevaluated(), "∫ {f} dx = {big_f}");
    assert_derivative_matches(&f, &big_f, &x, &ctx, &[(1, 3), (7, 5), (-5, 7)]);

    // Real constants keep `ln|·|`: `√2` by the assumption system, `√(3 −
    // √5)` numerically (SymPy 1.14: `sqrt(3 - sqrt(5)).is_real` → `True`,
    // `N(sqrt(3 - sqrt(5)))` → `0.874032048897642`).
    for src in ["1/(x - sqrt(2))", "1/(x - sqrt(3 - sqrt(5)))"] {
        let f = ctx.parse(src).unwrap();
        let big_f = f.integrate(&x);
        assert!(big_f.to_string().contains("abs"), "∫ {f} dx = {big_f}");
        assert_derivative_matches(&f, &big_f, &x, &ctx, &[(1, 3), (7, 5), (-5, 7)]);
    }
}

#[test]
fn a_parameter_under_a_square_root_is_not_assumed_real() {
    // `√(−a)` is imaginary for `a > 0`.  Up to 0.28.0 the answer was
    // `ln|x·√(−a) + a|/√(−a)`, whose derivative is the conjugate of the
    // integrand for every positive `a` (the Rubi harness listed it as
    // real-verified without a single agreeing point).  Now
    // `ln(x·√(−a) + a)/√(−a)`, right for either sign of `a` (SymPy 1.14:
    // `integrate(1/(a + x*sqrt(-a)), x)` → `log(a + x*sqrt(-a))/sqrt(-a)`).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let f = ctx.parse("1/(a + x*sqrt(-a))").unwrap();
    let big_f = f.integrate(&x);
    assert!(!big_f.has_unevaluated(), "∫ {f} dx = {big_f}");
    for (p, q) in [(6, 5), (-6, 5)] {
        let v = ctx.rational(p, q);
        let (f_a, big_f_a) = (f.subs(&a, &v), big_f.subs(&a, &v));
        assert_derivative_matches(&f_a, &big_f_a, &x, &ctx, &[(1, 3), (7, 5), (-5, 7)]);
    }
}

#[test]
fn a_root_of_a_polynomial_in_the_variable_is_a_constant() {
    // `RootOf(x⁵ − x + 1, k)` binds its `x`: it is a number (SymPy 1.14:
    // `CRootOf(x**5 - x + 1, 1).free_symbols` → `set()`).  Up to 0.28.0
    // the integrator's dependence test was structural, and both integrals
    // stayed unevaluated.  SymPy 1.14: `integrate(1/(x - CRootOf(x**5 - x
    // + 1, 1)), x)` → `log(x - CRootOf(x**5 - x + 1, 1))` (root 1 is
    // `-0.181232444469875 - 1.08395410131771*I`), `integrate(x*CRootOf(x**5
    // - x + 1, 0), x)` → `x**2*CRootOf(x**5 - x + 1, 0)/2`.  (`Ex::subs`
    // rewrites the bound `x` too, so `F′ = f` is checked symbolically.)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in ["1/(x - RootOf(x^5 - x + 1, 1))", "x*RootOf(x^5 - x + 1, 0)"] {
        let f = ctx.parse(src).unwrap();
        let big_f = f.integrate(&x);
        assert!(!big_f.has_unevaluated(), "∫ {f} dx = {big_f}");
        let residual = (big_f.diff(&x) - &f).simplify();
        assert!(
            residual.is_zero_structural(),
            "∫ {f} dx = {big_f}: F′ − f = {residual}"
        );
    }
    // The complex root keeps the analytic logarithm.
    let f = ctx.parse("1/(x - RootOf(x^5 - x + 1, 1))").unwrap();
    assert!(!f.integrate(&x).to_string().contains("abs"));
}

#[test]
fn partial_fractions_over_implicit_roots_do_not_replace_the_root_sum() {
    // Once `RootOf(x⁵ − x + 1, k)` counted as a constant, the heuristic
    // path of the rational integrator could integrate `apart`'s partial
    // fractions over the five implicit roots: the `RootSum` written out term
    // by term, with each root's polynomial in `x`, which `subs` of a sample
    // point then rewrote (the Rubi harness could no longer evaluate eleven
    // such answers).  Such a candidate is refused, and the `RootSum` stays.
    // f(1/3) = 0.706167258637391 (SymPy 1.14: `N(f.subs(x, Rational(1, 3)))`).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("1/((x^2 + 2)*(x^5 - x + 1))").unwrap();
    let big_f = f.integrate(&x);
    let shown = big_f.to_string();
    assert!(
        shown.contains("RootSum") && !shown.contains("RootOf"),
        "∫ {f} dx = {big_f}"
    );
    let third = ctx.rational(1, 3);
    let value = f.subs(&x, &third).eval_f64().unwrap();
    assert!((value - 0.706_167_258_637_391).abs() < 1e-12);
    assert_derivative_matches(&f, &big_f, &x, &ctx, &[(1, 3), (7, 5), (-5, 7)]);
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
