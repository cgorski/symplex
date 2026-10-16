//! Tests of session 15 (s15q4): slow paths found by the campaigns.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use std::time::Duration;

use symplex::prelude::*;
use symplex::stats::Distribution;

use crate::perf_assert::{CpuTimer, assert_fast};

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

/// `P(X = k³) = 1/n` for `k = 1..=n`.
fn cubes(ctx: &Context, n: i64) -> Distribution {
    let mass = ctx.rational(1, n);
    let table = (1..=n)
        .map(|k| (ctx.int(k * k * k), mass.clone()))
        .collect();
    Distribution::finite(ctx, table)
}

/// Up to 0.58 the CDF of a finite table asked the pmf once per atom, and
/// each lookup scanned the table (quadratic): `P(Y ≤ 5·10¹¹)` for `Y = X³`,
/// `X` a 100,000-sided die, took 3.4 s (the `fuzz_stats` input `Po4IAI5m`
/// 49.7 s); a rational table now has cumulative masses.
///
/// Oracle: counting, `k³ ≤ 10¹²` for `k ≤ 10⁴` of the 20,000 cubes.
#[test]
fn finite_table_cdf_reads_cumulative_masses() {
    let ctx = Context::new();
    let d = cubes(&ctx, 20_000);
    let at = ctx.int(1_000_000_000_000);
    let timer = CpuTimer::start();
    assert_eq!(d.cdf(&at), ctx.rational(1, 2));
    assert_eq!(d.cdf(&(&at - 1)), ctx.rational(9_999, 20_000));
    assert_eq!(d.sf(&at), ctx.rational(1, 2));
    assert_eq!(d.cdf(&ctx.rational(1, 2)), ctx.zero());
    assert_eq!(d.cdf(&ctx.int(10).powi(13)), ctx.one());
    // 9.7 s in a debug build up to 0.58.
    timer.assert_within(
        Duration::from_secs(3),
        "five CDF values of a 20,000-atom table",
    );
}

/// The pmf of a table was a scan per call (see above); it is now a lookup
/// by value, with the same answers: the mass at an atom, 0 at a rational
/// that is not one, 0 at an irrational number its certified value
/// separates from every atom.
///
/// Oracle: the table itself.
#[test]
fn finite_table_density_by_lookup() {
    let ctx = Context::new();
    let d = cubes(&ctx, 20_000);
    let atom = ctx.int(8_000 * 8_000 * 8_000);
    assert_eq!(d.density(&atom), ctx.rational(1, 20_000));
    assert_eq!(d.density(&(&atom + 1)), ctx.zero());
    assert_eq!(d.density(&ctx.int(2).sqrt()), ctx.zero());
}

/// A table of irrational values (`√X` of a die) keeps the indicator sum,
/// each comparison decided by the values (up to 0.58 one interned
/// difference and sign query per atom): the same exact answers.
///
/// Oracle: `√k ≤ 2` for `k ≤ 4`; `√5` is an atom.
#[test]
fn finite_table_of_roots_cdf() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let d = Distribution::die(ctx.int(6))
        .transformed(&x, &x.sqrt())
        .unwrap();
    assert_eq!(d.cdf(&ctx.int(2)), ctx.rational(2, 3));
    assert_eq!(d.cdf(&ctx.int(5).sqrt()), ctx.rational(5, 6));
    assert_eq!(d.cdf(&ctx.rational(21, 10)), ctx.rational(2, 3));
}

/// A die under `x³` (the fuzz case, smaller): the transformed table is
/// rational, and its CDF and survival function are exact.
///
/// Oracle: `k³ ≤ 10⁹` for `k ≤ 1000` of 5000 faces.
#[test]
fn die_under_cube_cdf_is_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = Distribution::die(ctx.int(5_000))
        .transformed(&x, &x.powi(3))
        .unwrap();
    let at = ctx.int(1_000_000_000);
    assert_eq!(y.cdf(&at), ctx.rational(1, 5));
    assert_eq!(y.sf(&at), ctx.rational(4, 5));
    assert_eq!(y.density(&at), ctx.rational(1, 5_000));
}

/// The `fuzz_stats` checks of the characteristic functions of
/// `StudentT(1/7)` and `StudentT(7/2)` took 10 s and 7.6 s: not the cf,
/// which is the closed form `K_{ν/2}(√ν|t|)(√ν|t|)^{ν/2}/(Γ(ν/2)2^{ν/2−1})`
/// (1.5 ms), but the harness's quadrature cross-check (a tail like
/// `|x|^{−8/7}`, and a sine part that is 0); the harness now skips
/// quadrature for tails heavier than `|x|⁻²` and requires a zero
/// imaginary part for an even density.  This pins the closed form.
///
/// Oracle: mpmath 1.3 at 30 and 60 digits, `besselk(nu/2, u)*u**(nu/2) /
/// (gamma(nu/2)*2**(nu/2 - 1))`, `u = sqrt(nu)*|t|`: `0.3641225979952596117`
/// (`ν = 1/7`, `t = −1/8`).
#[test]
fn student_t_characteristic_function_is_closed() {
    let ctx = Context::new();
    let d = Distribution::student_t(ctx.rational(1, 7));
    let cf = assert_fast!(
        Duration::from_secs(2),
        d.characteristic_function(&ctx.rational(-1, 8))
    );
    assert!(!cf.has_unevaluated(), "{cf}");
    let v = cf.eval_complex64().unwrap();
    assert!((v.re - 0.364_122_597_995_259_6).abs() < 1e-15, "{v}");
    assert_eq!(v.im, 0.0);
}

/// `F(10¹⁶, 10¹⁶).cdf(1)` is `I_{1/2}(5·10¹⁵, 5·10¹⁵)`: up to 0.58 the
/// continued fraction needed about 2·10⁶ steps there and was refused after
/// 20 s; the central route (Laplace's method with the convergent Taylor
/// series of the integrand) takes milliseconds.
///
/// Oracle: `I_{1/2}(a, a) = 1/2` by symmetry.
#[test]
fn f_cdf_at_the_centre_of_huge_shapes() {
    let ctx = Context::new();
    let big = ctx.int(10).powi(16);
    let d = Distribution::f_distribution(big.clone(), big);
    let c = d.cdf(&ctx.one());
    assert_eq!(
        assert_fast!(Duration::from_secs(5), c.eval_f64()).unwrap(),
        0.5
    );
    assert_eq!(c.eval_decimal(30).unwrap(), "0.5");
}

/// `I_x(10¹⁴, 3·10¹⁴)` a fraction of a standard deviation above the mode:
/// up to 0.58 a continued fraction of about 4·10⁵ steps (1.1 s; several
/// seconds in a debug build).  Both tails, each to its relative precision.
///
/// Oracle: mpmath 1.3 quadrature of `exp((a−1) ln(t/t₀) + (b−1)
/// ln((1−t)/(1−t₀)))` over `t₀ ± 45σ` at 200 and 300 bits:
/// `0.50184263658967740907595795721664900894936`.
#[test]
fn betainc_near_the_centre_of_shapes_1e14() {
    let ctx = Context::new();
    let a = ctx.int(10).powi(14);
    let b = ctx.int(3) * ctx.int(10).powi(14);
    let x = parse(&ctx, "1/4 + 10^(-10)");
    let e = x.betainc_regularized(&a, &b, &ctx.zero());
    let lower = assert_fast!(Duration::from_secs(5), e.eval_decimal(30)).unwrap();
    assert_eq!(lower, "0.501842636589677409075957957217");
    let upper = (ctx.one() - &e).eval_decimal(30).unwrap();
    assert_eq!(upper, "0.498157363410322590924042042783");
}

/// The far lower tail of `F(3·10¹²⁸, 1)`, at its `f64` quantile of
/// `10⁻³⁰⁰`: `I_z(1.5·10¹²⁸, ½)` with `1 − z = 4.6·10⁻¹²⁶`, which ran the
/// continued fraction through its 2·10⁶ steps (5.7 s) and then was refused
/// (`PrecisionExhausted`) in 0.57-era trees; computable since 0.58.
///
/// Oracle: `I_{1−y}(a, ½) = erfc(√(a·y))·(1 + O(1/a))` (`(1 − s)^{a−1} =
/// e^{−(a−1)s}` to `10⁻¹²³` on the mass), mpmath 1.3 at 300 and 400
/// digits: `9.9999999999999056923598638285378e-301`.
#[test]
fn f_cdf_far_lower_tail_of_a_huge_first_shape() {
    let ctx = Context::new();
    let d1 = ctx.int(3) * ctx.int(10).powi(128);
    let d = Distribution::f_distribution(d1, ctx.one());
    let at = parse(&ctx, "1678352823115841/2305843009213693952");
    let c = d.cdf(&at);
    assert_eq!(
        assert_fast!(Duration::from_secs(5), c.eval_decimal(25)).unwrap(),
        "9.999999999999905692359864e-301"
    );
}

/// `5/(3x·ln x·atanh(atanh x))` at 1 has a simple pole (`(x − 1)·f →
/// 10i/(3π)`) next to a function of `ln(1 − x)`: no expansion in powers.
/// Up to 0.58 the refusal tried all five pole retries `(x − 1)ᵏ·f` (0.3 s,
/// 4.4 s in a debug build); a finite `(x − 1)·f` makes only the first.
///
/// Oracle: `atanh(atanh x) → −iπ/2` as `x → 1`; `ln x/(x − 1) → 1`.
#[test]
fn series_refusal_at_a_simple_pole_retries_once() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = parse(&ctx, "5/(3*x*log(x)*atanh(atanh(x)))");
    let s = assert_fast!(Duration::from_secs(3), f.series(&x, &ctx.one(), 4));
    assert!(s.has_unevaluated(), "{s}");
}
