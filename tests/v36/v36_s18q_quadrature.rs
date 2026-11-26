//! Tests of session 18, second block of 0.60.1 ("every statistics call returns"). Lane Q: the
//! numeric integrals (`integrate_numeric`, the certified quadrature) on densities with a
//! narrow component or an extreme scale.
//! Every test says what was wrong before and cites its oracle.
//!
//! "Before" is the release candidate `e9c50e9`, release build (the probe `zz_mix` of the lane's
//! scratch, `tests_q.txt`); 0.60.0 returns the same values.  The oracles are mpmath 1.3.0 at
//! `mp.dps = 40` and `80` (`oracle_mix.py` of the lane's scratch: closed forms of the
//! distribution functions) or an exact argument.

#![allow(unused_imports)]
use std::time::Duration;

use crate::perf_assert::CpuTimer;
use symplex::definite::{QuadOpts, QuadRule};
use symplex::prelude::*;
use symplex::stats::Distribution;

/// `∫ g dx` over `[lo, hi]` by `integrate_numeric`; a refusal is `None`.
fn fast(g: &Ex, x: &Ex, lo: &Ex, hi: &Ex) -> Option<f64> {
    g.integrate_numeric(x, lo, hi).ok()
}

/// The same by the certified digits of `integrate_numeric_decimal`; a refusal is `None`.
fn certified(g: &Ex, x: &Ex, lo: &Ex, hi: &Ex, digits: u32) -> Option<f64> {
    g.integrate_numeric_decimal(x, &[lo, hi], digits, QuadRule::default())
        .ok()
        .map(|s| s.parse::<f64>().unwrap())
}

/// A numeric integral is the right value or a refusal.
fn right_or_refused(what: &str, got: Option<f64>, want: f64, tol: f64) {
    if let Some(v) = got {
        assert!(
            (v - want).abs() <= tol,
            "{what} = {v:e}, the integral is {want}"
        );
    }
}

/// A numeric integral is the right value.
fn right(what: &str, got: Option<f64>, want: f64, tol: f64) {
    let Some(v) = got else {
        panic!("{what} is refused, the integral is {want}");
    };
    assert!(
        (v - want).abs() <= tol,
        "{what} = {v:e}, the integral is {want}"
    );
}

/// The mixture `a/2 + b/2`.
fn halves(ctx: &Context, a: Distribution, b: Distribution) -> Distribution {
    Distribution::mixture(&[(ctx.rational(1, 2), a), (ctx.rational(1, 2), b)]).unwrap()
}

// ── An estimate that is all error (class B of the fuzz findings) ─────────────

/// The density of a law of a huge scale over its support.  Before, `integrate_numeric` returned
/// an estimate smaller than its own error estimate, which passed the absolute tolerance
/// `10⁻¹²`: `Normal(0, 10⁸⁰)` over ℝ `Ok(6.130059011750917e-66)` (`integrate_numeric_with`: error
/// `4.8e-62`), `Normal(0, 10⁴⁰)` `Ok(6.1e-26)`, `Exponential(10⁻⁸⁰)` over `[0, ∞)`
/// `Ok(7.7e-66)`, `Gamma(2, 10⁸⁰)` `Ok(1.8e-131)`, `Weibull(4·10⁷⁶, 27/8)`
/// `Ok(2.6163788076620492e-250)`.  Now such an estimate is decided by the certified route or
/// refused; `integrate_numeric_with` does not return it with an error inside the tolerance.
///
/// Oracle: the definition (a density: total mass 1); mpmath, `quad` of the standard density in
/// `u = x/σ` = `1.0` (`mp.dps = 40`, `80`).
#[test]
fn an_estimate_not_larger_than_its_error_estimate_is_not_returned() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (minf, inf, zero) = (ctx.neg_infinity(), ctx.infinity(), ctx.int(0));
    let p = |s: &str| ctx.parse(s).unwrap();
    let laws = [
        (
            "Normal(0, 1e80)",
            Distribution::normal(ctx.int(0), p("10^80")),
            &minf,
        ),
        (
            "Normal(0, 1e40)",
            Distribution::normal(ctx.int(0), p("10^40")),
            &minf,
        ),
        (
            "Exponential(1e-80)",
            Distribution::exponential(p("1/10^80")),
            &zero,
        ),
        (
            "Gamma(2, 1e80)",
            Distribution::gamma(ctx.int(2), p("10^80")),
            &zero,
        ),
        (
            "Weibull(4e76, 27/8)",
            Distribution::weibull(p("4*10^76"), ctx.rational(27, 8)),
            &zero,
        ),
    ];
    let opts = QuadOpts::default();
    for (name, law, lo) in &laws {
        let f = law.density(&x);
        right_or_refused(
            &format!("integrate_numeric, {name}"),
            fast(&f, &x, lo, &inf),
            1.0,
            1e-9,
        );
        if let Ok(r) = f.integrate_numeric_with(&x, lo, &inf, &opts) {
            let tolerance = opts.abs_tol.max(opts.rel_tol * r.value.abs());
            assert!(
                r.error > tolerance || (r.value - 1.0).abs() <= 1e-9,
                "integrate_numeric_with, {name}: {:e} with an error estimate of {:e}",
                r.value,
                r.error
            );
        }
    }
}

/// Integrals that are 0 or tiny keep their values under the rule of the test above: the mean of
/// `Normal(0, 1)` is `Ok(0.0)` before and now (the certified 0, error `8.7e-19`), and a far
/// tail, whose estimate is far above its error estimate, is the same `f64`
/// (`7.619853027908694e-24`, error `5.2e-27`).  The two other integrals were not recorded
/// before; they are asserted as they are now.
///
/// Oracles: `∫ x·φ(x) dx` over ℝ, `∫₀^{2π} sin x dx` and `∫₋₁¹ x³ dx` are 0 exactly (odd
/// integrands, a full period); mpmath, `ncdf(-10)` = `7.619853024160526065973343e-24`.
#[test]
fn integrals_that_are_zero_or_tiny_are_still_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (minf, inf) = (ctx.neg_infinity(), ctx.infinity());
    let phi = Distribution::normal(ctx.int(0), ctx.int(1)).density(&x);
    right(
        "the mean of Normal(0, 1)",
        fast(&(&x * &phi), &x, &minf, &inf),
        0.0,
        1e-12,
    );
    right(
        "sin over a period",
        fast(&x.sin(), &x, &ctx.int(0), &(ctx.int(2) * ctx.pi())),
        0.0,
        1e-12,
    );
    right(
        "x^3 over [-1, 1]",
        fast(&x.powi(3), &x, &ctx.int(-1), &ctx.int(1)),
        0.0,
        1e-12,
    );
    let tail = fast(&phi, &x, &ctx.int(10), &inf);
    const TAIL: f64 = 7.619_853_024_160_526e-24;
    right("P(Z > 10)", tail, TAIL, 1e-8 * TAIL);
}

// ── A piece of the range that no node saw (the lost left part) ───────────────

/// `P(X ≤ 1)` for `X ~ 1/2·Normal(0, 1) + 1/2·FDistribution(10⁻²⁷⁰, 1)`, an integral over
/// `(−∞, 1]`.  The formula of the F density (used for `x > 0` only) has a pole at `x = −d₂/d₁ =
/// −10²⁷⁰`, which the breakpoint scan returns, so the range is split at `−10²⁷⁰` and at 0.  On
/// the piece `[−10²⁷⁰, 0]` every node gave 0 (none comes near 0, where the normal component
/// is) and the piece was summed as `0 ± 0`: before, `integrate_numeric` was
/// `Ok(0.17067237303427146)` and 12 certified digits `0.170672373034`, `(Φ(1) − 1/2)/2`, the
/// part on `[0, 1]` (the normal half's `0.25` left of 0 lost besides the F half).  The same for
/// `1/2·Laplace(0, 1/10) + 1/2·FDistribution(2⁻⁹⁰⁰, 1)` over ℝ: `Ok(0.25)`, certified `0.25`.
/// Now a piece that no node saw is not a 0: both are refused (the F half's mass lies where
/// `ln x ≈ −10²⁷⁰`, below every floating-point range: no quadrature can see it).
///
/// Oracle: mpmath, `ncdf(1)/2 + betainc(d1/2, mpf(1)/2, 0, d1/(d1 + 1), regularized=True)/2` =
/// `0.9206723730342714742926163` (`mp.dps = 40`, `80`); the definition for the second (1).
#[test]
fn a_piece_every_node_of_which_gave_zero_is_not_summed_as_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (minf, inf) = (ctx.neg_infinity(), ctx.infinity());
    let d = halves(
        &ctx,
        Distribution::normal(ctx.int(0), ctx.int(1)),
        Distribution::f_distribution(ctx.parse("1/10^270").unwrap(), ctx.int(1)),
    );
    let f = d.density(&x);
    const P: f64 = 0.920_672_373_034_271_5;
    right_or_refused(
        "integrate_numeric",
        fast(&f, &x, &minf, &ctx.int(1)),
        P,
        1e-7,
    );
    right_or_refused(
        "integrate_numeric_decimal, 12 digits",
        certified(&f, &x, &minf, &ctx.int(1), 12),
        P,
        1e-7,
    );
    let b = halves(
        &ctx,
        Distribution::laplace(ctx.int(0), ctx.rational(1, 10)),
        Distribution::f_distribution(ctx.parse("1/2^900").unwrap(), ctx.int(1)),
    );
    let g = b.density(&x);
    right_or_refused(
        "integrate_numeric, (b)",
        fast(&g, &x, &minf, &inf),
        1.0,
        1e-7,
    );
    right_or_refused(
        "integrate_numeric_decimal, 12 digits, (b)",
        certified(&g, &x, &minf, &inf, 12),
        1.0,
        1e-7,
    );
}

/// The pieces that ARE 0 keep their values, and their cost: the density of a mixture is a
/// `Piecewise` that is 0 left of a component's support, and every node of the piece `(−∞, 0]`
/// gives 0.  Such a piece counts as 0 when the integrand is an exact 0 next to its ends (or
/// below the exponent range: a far tail), without a quadrature of its own.  Before and now:
/// `1/2·Exponential(1) + 1/2·Normal(0, 1)` over ℝ is `1.0`, `P(X ≤ 1)` of `1/2·Gamma(5/2, 1) +
/// 1/2·Normal(0, 1)` is `0.4960998549919605` (the same `f64` values before and now, with the
/// same error estimates `3.98e-11` and `1.98e-11`), and the part of `1/2·Laplace(0, 1/10) + 1/2·FDistribution(10⁻⁵, 1)`
/// left of 0 (the F formula's pole at `−10⁵` is a cut of the range) is `0.25`.
///
/// Oracle: the definition (1); mpmath, `ncdf(1)/2 + gammainc(mpf(5)/2, 0, 1,
/// regularized=True)/2` = `0.496099854991966656179669715823` (`mp.dps = 40`, `80`); the
/// Laplace law is symmetric about 0 and the F law has no mass there: `1/4`.
#[test]
fn a_piece_that_is_zero_keeps_its_value_and_its_cost() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (minf, inf) = (ctx.neg_infinity(), ctx.infinity());
    let normal = || Distribution::normal(ctx.int(0), ctx.int(1));
    let timer = CpuTimer::start();
    let f = halves(&ctx, Distribution::exponential(ctx.int(1)), normal()).density(&x);
    right("the total mass", fast(&f, &x, &minf, &inf), 1.0, 1e-9);
    let g = halves(
        &ctx,
        Distribution::gamma(ctx.rational(5, 2), ctx.int(1)),
        normal(),
    )
    .density(&x);
    const P_ONE: f64 = 0.496_099_854_991_966_7;
    right("P(X <= 1)", fast(&g, &x, &minf, &ctx.int(1)), P_ONE, 1e-9);
    timer.assert_within(Duration::from_secs(20), "two integrals across a zero piece");
    let b = halves(
        &ctx,
        Distribution::laplace(ctx.int(0), ctx.rational(1, 10)),
        Distribution::f_distribution(ctx.parse("1/10^5").unwrap(), ctx.int(1)),
    );
    let h = b.density(&x);
    right("P(Y <= 0)", fast(&h, &x, &minf, &ctx.int(0)), 0.25, 1e-9);
    right(
        "P(Y <= 0), 12 digits",
        certified(&h, &x, &minf, &ctx.int(0), 12),
        0.25,
        1e-11,
    );
}

/// A tail cut off next to a narrow peak stays a value.  The range of `cos(x/2)·f(x)`, `f` the
/// density of `1/2·Normal(0, 10⁻⁴⁰) + 1/2·Exponential(1)`, is split next to the peak (ten widths
/// from it), and on the piece beyond the cut every node of the certified quadrature underflows
/// while the integrand has a value at the cut.  What such a piece holds next to its end is now
/// in the bound (to first order: `|f|/λ`, `λ` the rate of fall at the end) instead of counting
/// as nothing; the certified digits are as before: `0.9`, and `0.893071615781` for
/// `1/2·Normal(1/3, 10⁻⁶) + 1/2·Exponential(1)`, where `integrate_numeric` is
/// `0.8930716157694397` before and now.  (For the first law `integrate_numeric` is
/// `0.39999999999999997`, before and now: the peak at the cut 0 is hidden behind the
/// exponential component's values, the class of `a_narrow_component_of_a_mixture_…` in
/// `v36_s18h_fuzz_findings`, which is not fixed; it is not asserted here.)
///
/// Oracle: mpmath, `exp(-s*s*t*t/2)*cos(m*t)/2 + (1/(1 - 1j*t)).real/2` at `t = 1/2` =
/// `0.9` (`m = 0`, `s = 1e-40`) and `0.8930716157814008950133702` (`m = 1/3`, `s = 1e-6`).
#[test]
fn a_tail_cut_off_next_to_a_narrow_peak_is_still_certified() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (minf, inf) = (ctx.neg_infinity(), ctx.infinity());
    let half_x = &x / 2;
    let a = halves(
        &ctx,
        Distribution::normal(ctx.int(0), ctx.parse("1/10^40").unwrap()),
        Distribution::exponential(ctx.int(1)),
    );
    let g = half_x.cos() * a.density(&x);
    right(
        "12 digits, sd 1e-40",
        certified(&g, &x, &minf, &inf, 12),
        0.9,
        1e-11,
    );
    let b = halves(
        &ctx,
        Distribution::normal(ctx.rational(1, 3), ctx.parse("1/10^6").unwrap()),
        Distribution::exponential(ctx.int(1)),
    );
    let h = half_x.cos() * b.density(&x);
    const CF: f64 = 0.893_071_615_781_400_9;
    right(
        "12 digits, sd 1e-6",
        certified(&h, &x, &minf, &inf, 12),
        CF,
        1e-11,
    );
    right(
        "integrate_numeric, sd 1e-6",
        fast(&h, &x, &minf, &inf),
        CF,
        1e-9,
    );
}
