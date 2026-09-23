//! 0.23 — the evaluator takes the principal branch everywhere.
//!
//! `fuzz_simplify` compares complex values since 0.23, and its first finds
//! were in the oracle itself: the sign of a zero imaginary part picked the
//! side of a branch cut, `acosh`/`atanh`/`atan` used formulas that are the
//! other branch in part of the plane, real arguments outside the real domain
//! gave NaN, and trigonometric functions of huge arguments either printed
//! digits the argument did not determine or ran for minutes.
//!
//! Reference values are mpmath 1.3 at `mp.dps = 40` (`symplex/.venv`),
//! quoted at 17 significant digits; the call is next to each value.

// Reference values are quoted at the 17 digits mpmath printed them.
#![allow(clippy::excessive_precision)]

use std::f64::consts::{FRAC_PI_2, PI};
use std::time::{Duration, Instant};

use symplex::prelude::*;

/// `z` agrees with `re + im·i` to a relative `1e-14` of `|re + im·i|`.
fn close_c(z: Complex64, re: f64, im: f64, label: &str) {
    let want = Complex64::new(re, im);
    let err = (z - want).norm();
    assert!(
        err <= 1e-14 * want.norm().max(1.0),
        "{label}: got {z}, expected {want} (error {err:e})"
    );
}

fn value(e: &Ex, label: &str) -> Complex64 {
    e.eval_complex64()
        .unwrap_or_else(|err| panic!("{label}: {e} did not evaluate: {err}"))
}

// ── the negative real axis has argument +π, however it was reached ─────

#[test]
fn ln_of_a_negated_positive_is_on_the_upper_side() {
    // mpmath: log(-sin(mpf(1)/3)) = -1.117199882220614 + 3.1415926535897932j
    let ctx = Context::new();
    let s = ctx.rational(1, 3).sin();
    for (label, e) in [
        ("ln(-sin(1/3))", (-&s).ln()),
        ("ln(-1*sin(1/3))", (&s * -1).ln()),
        ("ln(sin(-1/3))", ctx.rational(-1, 3).sin().ln()),
    ] {
        close_c(value(&e, label), -1.117199882220614, PI, label);
    }
}

#[test]
fn sqrt_of_a_negated_positive_is_the_upper_root() {
    // mpmath: sqrt(-tan(mpf(1)/3)) = 0.58843313087433774j
    let ctx = Context::new();
    let e = (-ctx.rational(1, 3).tan()).sqrt();
    close_c(
        value(&e, "sqrt(-tan(1/3))"),
        0.0,
        0.58843313087433774,
        "sqrt(-tan(1/3))",
    );
}

#[test]
fn rational_power_of_a_negated_positive_is_principal() {
    // mpmath: (-sin(mpf(1)/3))**(mpf(-3)/2) = 5.3430669501077501j
    let ctx = Context::new();
    let e = (-ctx.rational(1, 3).sin()).pow(&ctx.rational(-3, 2));
    close_c(
        value(&e, "(-sin(1/3))^(-3/2)"),
        0.0,
        5.3430669501077501,
        "(-sin(1/3))^(-3/2)",
    );
}

// ── inverse functions: principal values across the whole plane ─────────

#[test]
fn acosh_in_the_left_half_plane() {
    // mpmath: acosh(mpc(-2, 0.5)) = 1.3618009008578458 + 2.8638383970320793j
    //         acosh(mpc(-3, -3))  = 2.1386220863162211 - 2.342314518916745j
    let ctx = Context::new();
    let i = ctx.i_unit();
    let a = (ctx.int(-2) + &i * ctx.rational(1, 2)).acosh();
    close_c(
        value(&a, "acosh(-2+i/2)"),
        1.3618009008578458,
        2.8638383970320793,
        "acosh(-2+i/2)",
    );
    let b = (ctx.int(-3) - &i * 3).acosh();
    close_c(
        value(&b, "acosh(-3-3i)"),
        2.1386220863162211,
        -2.342314518916745,
        "acosh(-3-3i)",
    );
}

#[test]
fn atanh_beyond_one_is_below_the_cut() {
    // mpmath: atanh(2)          = 0.54930614433405485 - 1.5707963267948966j
    //         atanh(mpf(3)/2)   = 0.80471895621705019 - 1.5707963267948966j
    let ctx = Context::new();
    close_c(
        value(&ctx.int(2).atanh(), "atanh(2)"),
        0.54930614433405485,
        -FRAC_PI_2,
        "atanh(2)",
    );
    close_c(
        value(&ctx.rational(3, 2).atanh(), "atanh(3/2)"),
        0.80471895621705019,
        -FRAC_PI_2,
        "atanh(3/2)",
    );
}

#[test]
fn atan_on_the_imaginary_cut() {
    // mpmath: atan(mpc(0, 2))  =  1.5707963267948966 + 0.54930614433405485j
    //         atan(mpc(0, -3)) = -1.5707963267948966 - 0.34657359027997265j
    let ctx = Context::new();
    let i = ctx.i_unit();
    close_c(
        value(&(&i * 2).atan(), "atan(2i)"),
        FRAC_PI_2,
        0.54930614433405485,
        "atan(2i)",
    );
    close_c(
        value(&(&i * -3).atan(), "atan(-3i)"),
        -FRAC_PI_2,
        -0.34657359027997265,
        "atan(-3i)",
    );
}

#[test]
fn real_arguments_outside_the_real_domain_are_complex() {
    // mpmath: asin(2)          =  1.5707963267948966 - 1.3169578969248167j
    //         asin(-3)         = -1.5707963267948966 + 1.7627471740390861j
    //         acos(2)          =  1.3169578969248167j
    //         acosh(mpf(1)/3)  =  1.2309594173407747j
    //         acosh(-2)        =  1.3169578969248167 + 3.1415926535897932j
    let ctx = Context::new();
    let cases: [(&str, Ex, f64, f64); 5] = [
        ("asin(2)", ctx.int(2).asin(), FRAC_PI_2, -1.3169578969248167),
        (
            "asin(-3)",
            ctx.int(-3).asin(),
            -FRAC_PI_2,
            1.7627471740390861,
        ),
        ("acos(2)", ctx.int(2).acos(), 0.0, 1.3169578969248167),
        (
            "acosh(1/3)",
            ctx.rational(1, 3).acosh(),
            0.0,
            1.2309594173407747,
        ),
        ("acosh(-2)", ctx.int(-2).acosh(), 1.3169578969248167, PI),
    ];
    for (label, e, re, im) in cases {
        close_c(value(&e, label), re, im, label);
    }
}

#[test]
fn negative_real_arguments_outside_the_domain_are_complex() {
    // mpmath: acos(-2)         =  3.1415926535897932 - 1.3169578969248167j
    //         asin(3)          =  1.5707963267948966 - 1.7627471740390861j
    //         acosh(-1)        =  3.1415926535897932j
    //         acosh(mpf(-1)/2) =  2.0943951023931955j
    //         atanh(-2)        = -0.54930614433405485 + 1.5707963267948966j
    let ctx = Context::new();
    let cases: [(&str, Ex, f64, f64); 5] = [
        ("acos(-2)", ctx.int(-2).acos(), PI, -1.3169578969248167),
        ("asin(3)", ctx.int(3).asin(), FRAC_PI_2, -1.7627471740390861),
        ("acosh(-1)", ctx.int(-1).acosh(), 0.0, PI),
        (
            "acosh(-1/2)",
            ctx.rational(-1, 2).acosh(),
            0.0,
            2.0943951023931955,
        ),
        (
            "atanh(-2)",
            ctx.int(-2).atanh(),
            -0.54930614433405485,
            FRAC_PI_2,
        ),
    ];
    for (label, e, re, im) in cases {
        close_c(value(&e, label), re, im, label);
    }
}

#[test]
fn half_integer_power_of_a_negative_real_is_purely_imaginary() {
    // (−r)^(n/2) = r^(n/2)·iⁿ with no real part, so the next branch cut
    // sees exactly the imaginary axis: acosh((−5/7)^(9/2)) = acosh(0.22…i).
    // mpmath: (mpf(-5)/7)**(mpf(9)/2) = 0.22000058692433272j
    //         acosh(_)                 = 0.21826348065624887 + 1.5707963267948966j
    let ctx = Context::new();
    let p = ctx.rational(-5, 7).pow(&ctx.rational(9, 2));
    let z = value(&p, "(-5/7)^(9/2)");
    assert_eq!(z.re, 0.0, "(-5/7)^(9/2) = {z}");
    close_c(z, 0.0, 0.22000058692433272, "(-5/7)^(9/2)");
    close_c(
        value(&p.acosh(), "acosh((-5/7)^(9/2))"),
        0.21826348065624887,
        FRAC_PI_2,
        "acosh((-5/7)^(9/2))",
    );
}

#[test]
fn real_arguments_outside_the_domain_are_not_real_for_eval_f64() {
    let ctx = Context::new();
    let err = ctx.int(2).asin().eval_f64().unwrap_err();
    assert!(
        matches!(err, SymplexError::ComputationFailed { .. }),
        "asin(2) is complex, so eval_f64 must refuse it: {err:?}"
    );
}

#[test]
fn asin_and_asinh_keep_their_digits_for_large_arguments() {
    // iz + √(1 − z²) cancels in half the plane for large |z|, leaving no
    // correct digit: fuzz_simplify found acos(cosh(x^(9/2))) at x = 1/3 + 4i
    // evaluated to 2.6412 + 346.70i; mpmath says 0.54760585478819298 - 476.67258128330411j.
    // mpmath (mp.dps = 40): asin(mpf(-1e10))        = -1.5707963267948966 + 23.718998110500402j
    //                       asin(mpc(-5e8, -1))      = -1.5707963247948966 - 20.723265836946411j
    //                       asinh(mpc(-3e9, 2e9))    = -22.698887696237125 + 0.58800260354756755j
    //                       asinh(mpc(0, -1e10))     = -23.718998110500402 - 1.5707963267948966j
    let ctx = Context::new();
    let i = ctx.i_unit();
    let cases: [(&str, Ex, f64, f64); 4] = [
        (
            "asin(-1e10)",
            ctx.int(-10_000_000_000).asin(),
            -FRAC_PI_2,
            23.718998110500402,
        ),
        (
            "asin(-5e8 - i)",
            (ctx.int(-500_000_000) - &i).asin(),
            -1.5707963247948966,
            -20.723265836946411,
        ),
        (
            "asinh(-3e9 + 2e9 i)",
            (ctx.int(-3_000_000_000) + &i * 2_000_000_000).asinh(),
            -22.698887696237125,
            0.58800260354756755,
        ),
        (
            "asinh(-1e10 i)",
            (&i * -10_000_000_000i64).asinh(),
            -23.718998110500402,
            -FRAC_PI_2,
        ),
    ];
    for (label, e, re, im) in cases {
        close_c(value(&e, label), re, im, label);
    }
}

// ── trigonometric functions of huge arguments ─────────────────────────

#[test]
fn sin_of_a_huge_exact_integer_is_reduced_exactly() {
    // mpmath (mp.dps = 80): sin(mpf(10)**100) = -0.37237612366127669
    let ctx = Context::new();
    let v = ctx.int(10).powi(100).sin().eval_f64().unwrap();
    assert!(
        (v - -0.37237612366127669).abs() < 1e-15,
        "sin(10^100) = {v}, expected -0.37237612366127669"
    );
}

#[test]
fn sin_of_a_huge_inexact_argument_is_refused() {
    // exp(100) ≈ 2^144 is rounded to the working precision, so it is known
    // only to within about 2^(144 − prec): not enough to reduce modulo 2π.
    // mpmath (mp.dps = 80): sin(exp(100)) = 0.14219812365823864.
    let ctx = Context::new();
    let e = ctx.int(100).exp().sin();
    for digits in [16, 30] {
        let r = e.eval_decimal(digits);
        assert!(
            matches!(r, Err(SymplexError::PrecisionExhausted { .. })),
            "sin(exp(100)) at {digits} digits: {r:?}"
        );
    }
    assert!(e.eval_f64().is_err());
}

#[test]
fn sin_of_a_moderate_inexact_argument_still_evaluates() {
    // mpmath (mp.dps = 60): sin(exp(40)) = 0.94808470848664736  (exp(40) ≈ 2^57.7)
    let ctx = Context::new();
    let v = ctx.int(40).exp().sin().eval_f64().unwrap();
    assert!(
        (v - 0.94808470848664736).abs() < 1e-12,
        "sin(exp(40)) = {v}"
    );
}

#[test]
fn nested_cosh_at_a_complex_point_is_refused_promptly() {
    // cosh(x²) at x = 1/3 + 4i is ≈ −3.5e6 − 1.8e6i; the next cosh has
    // magnitude e^(3.5e6), and the last one's imaginary part needs cos of
    // that — π to millions of bits.  (Found by fuzz_simplify as a timeout.)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(2).cosh().cosh().cosh();
    let at = ctx.rational(1, 3) + ctx.i_unit() * 4;
    let t = Instant::now();
    let r = f.subs(&x, &at).eval_decimal(30);
    assert!(r.is_err(), "cosh(cosh(cosh(x^2))) at 1/3+4i: {r:?}");
    assert!(
        t.elapsed() < Duration::from_secs(5),
        "took {:?}",
        t.elapsed()
    );
}
