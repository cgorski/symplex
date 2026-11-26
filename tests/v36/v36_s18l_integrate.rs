//! Tests of session 18, second block of 0.60.1. Lane SI: size bounds of two symbolic
//! integration rules.
//! Every test says what was wrong before and cites its oracle.

use std::time::Duration;

use symplex::prelude::*;
use symplex::stats::{Distribution, Support};

use crate::perf_assert::assert_fast;

/// The CPU time a class-D call may take in a debug build.  A release build
/// needs 0.03 to 0.13 s for each (before: 5.7 to 41 s), a debug build about
/// twenty times that.
const BUDGET: Duration = Duration::from_secs(20);

/// `F` is unevaluated (the documented result of `integrate` when it finds no
/// closed form), or an antiderivative: then `F′ = f` at `x = 3` to 25 digits.
fn refused_or_right(ctx: &Context, f: &Ex, big_f: &Ex, x: &Ex) {
    if big_f.has_unevaluated() {
        return;
    }
    let three = ctx.int(3);
    let lhs = big_f.diff(x).subs(x, &three).eval_decimal(25).unwrap();
    let rhs = f.subs(x, &three).eval_decimal(25).unwrap();
    assert_eq!(lhs, rhs, "d/dx of the result of ∫ {f} at x = 3");
}

/// By parts with a polynomial `u` of a degree in the hundreds.  `∫ x^(759/2)·
/// e^(−x/2) dx` (the mean of a truncated `χ²(759)` law; `fuzz_stats`, class D)
/// took 14.2 s of CPU time in a release build before it came back unevaluated,
/// `x^(481/2)` (the second moment of `χ²(479)`) 5.7 s, in 0.60.0 and in the
/// 0.60.1 candidate alike: after `x = u²` by parts took `u⁷⁶⁰` for its `u`
/// and went down all twenty levels of the search — each level lowers the
/// degree by one, so the chain could not end — with every other strategy,
/// and its polynomial arithmetic at degree 760, at every level.  Beyond a
/// chain of 64 levels by parts no longer searches for `∫ v·du` when `dv` is
/// transcendental (it takes an integral at hand): 0.07 s and 0.03 s.
///
/// Oracle: the contract of `integrate` (no closed form found: the
/// unevaluated integral; never a guess), and for an answer `F′ = f`.
#[test]
fn a_half_integer_power_times_an_exponential_is_refused_at_once() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in [
        "x^(759/2)*exp(-x/2)",
        "x^(481/2)*exp(-x/2)",
        "x^(1999/2)*exp(-x/2)",
    ] {
        let f = ctx.parse(src).unwrap();
        let big_f = assert_fast!(BUDGET, f.integrate(&x), "∫ {src}");
        refused_or_right(&ctx, &f, &big_f, &x);
    }
}

/// The cyclic rule (`∫ eˣ·sin x dx`: by parts twice, then solve for the
/// integral) on a factor of a degree in the hundreds.  `∫ e^(−x/8)·x^(321/2)·
/// (1 + 323·x)^(−162) dx` (the moment generating function of `F(323, 1)` at
/// `−1/8`; `fuzz_stats`, class D) took 41 s before it came back unevaluated:
/// after `x = u²` the rule integrated the rational factor twice (Hermite's
/// reduction on a denominator of degree 324, 20 s each) to look for a cycle
/// that cannot be there.  The rule now declines a factor beyond degree 64:
/// 0.12 s.
///
/// Oracle: as above.
#[test]
fn an_exponential_times_a_rational_function_of_high_degree_is_refused_at_once() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let src = "exp(-x/8)*x^(321/2)/(1 + 323*x)^162";
    let f = ctx.parse(src).unwrap();
    let big_f = assert_fast!(BUDGET, f.integrate(&x), "∫ {src}");
    refused_or_right(&ctx, &f, &big_f, &x);
}

/// The two bounds take no integral away.  Below the bound of by parts: the
/// chain of twenty levels (`x²⁰·eˣ`, the longest of the Rubi suite) and the
/// half-integer powers the statistics module asks for (`x^(11/2)·e^(−x/2)`).
/// Beyond degree 64, where the bound does not apply or does not hurt: a
/// polynomial times a polynomial (`x¹²·(a + b·x¹³)¹²`, degree 156); a `u`
/// that is no polynomial (`x²⁰⁰·ln x`, `x¹⁰⁰·atan x`); a power of a sum, whose
/// chain is its exponent (`(1 + x⁸)¹²·ln x`, degree 96: a first version of
/// the bound, by the degree, lost it); a remaining integral that is at hand
/// (`x¹⁰⁰·e^(−x/2)/(1 + x)`: the terms of the quotient in rising order, each
/// `∫ v·du` two levels away; a first version that declined outright lost
/// it); one that is a multiple of the integrand (`x⁵⁰⁰·|x|`); and the cyclic
/// rule on factors of degree one (`e^(200x)·sin(201x)`).  All were
/// integrated before (0.60.0 and the candidate; the lane's hunt) and are now.
///
/// Oracle: `F′ = f` at `x = 3` (and `a = 2`, `b = 5`) to 25 digits.
#[test]
fn the_bounds_of_by_parts_take_no_integral_away() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (a, b) = (ctx.symbol("a"), ctx.symbol("b"));
    let at = |e: &Ex| {
        e.subs(&x, &ctx.int(3))
            .subs(&a, &ctx.int(2))
            .subs(&b, &ctx.int(5))
            .eval_decimal(25)
            .unwrap()
    };
    for src in [
        "x^20*exp(x)",
        "x^(11/2)*exp(-x/2)",
        "x^50*exp(-x/2)",
        "x^12*(a + b*x^13)^12",
        "x^200*ln(x)",
        "x^100*atan(x)",
        "(1 + x^8)^12*ln(x)",
        "x^100*exp(-x/2)/(1 + x)",
        "x^500*abs(x)",
        "exp(200*x)*sin(201*x)",
    ] {
        let f = ctx.parse(src).unwrap();
        let big_f = assert_fast!(BUDGET, f.integrate(&x), "∫ {src}");
        assert!(!big_f.has_unevaluated(), "∫ {src} = {big_f}");
        assert_eq!(at(&big_f.diff(&x)), at(&f), "d/dx of ∫ {src} at x = 3");
    }
}

/// The three statistics calls of class D return.  Before (release build,
/// 0.60.0 and the candidate): `mean()` of `χ²(759)` truncated to `[48.7,
/// 622.6]` 14.3 s, `mgf(−1/8)` of `F(323, 1)` 37 s, `variance()` of
/// `½·χ²(479) + ½·Triangular(0, 1, 0)` 16.8 s; each is an unevaluated
/// integral, before and now.
///
/// Oracles.  The mean: mpmath, `k·P(k/2 + 1, a/2, b/2)/P(k/2, a/2, b/2)` with
/// `P = gammainc(…, regularized=True)`, `k = 759`, at 30 and at 60 digits,
/// and `quad` of the density at both: `614.480687402481381436181558327`.
/// The variance: exact, from the moments of the components (`χ²(k)`: `k`,
/// `2k + k²`; `Triangular(0, 1, 0)`: `1/3`, `1/6`): `2079341/36`.
#[test]
fn the_statistics_calls_of_class_d_return() {
    let ctx = Context::new();
    let q = |s: &str| ctx.parse(s).unwrap();

    let truncated = Distribution::chi_squared(ctx.int(759))
        .truncated(&Support::interval(
            q("3428177688775441/70368744177664"),
            q("5476260035578373/8796093022208"),
        ))
        .unwrap();
    let mean = assert_fast!(BUDGET, truncated.mean(), "the mean of a truncated χ²(759)");
    let value = mean.eval_f64().unwrap();
    assert!(
        (value - 614.480_687_402_481_4).abs() < 1e-9 * 614.0,
        "mean = {value}"
    );

    let f_law = Distribution::f_distribution(ctx.int(323), ctx.int(1));
    let mgf = assert_fast!(BUDGET, f_law.mgf(&q("-1/8")), "mgf(−1/8) of F(323, 1)");
    assert!(mgf.has_unevaluated(), "mgf = {mgf}");

    let half = q("1/2");
    let mixture = Distribution::mixture(&[
        (half.clone(), Distribution::chi_squared(ctx.int(479))),
        (
            half,
            Distribution::triangular(ctx.int(0), ctx.int(1), ctx.int(0)),
        ),
    ])
    .unwrap();
    let variance = assert_fast!(
        BUDGET * 3,
        mixture.variance(),
        "the variance of ½·χ²(479) + ½·Triangular(0, 1, 0)"
    );
    let value = variance.eval_f64().unwrap();
    assert!(
        (value - 2_079_341.0 / 36.0).abs() < 1e-9 * 57_759.0,
        "variance = {value}"
    );
}
