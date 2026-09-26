//! Quantiles at the edges of the floats: the far-subnormal branch of the
//! `numdist` log-quantile iterations, and `Distribution::quantile_f64` near
//! `p = 1` for families without a closed quantile.

use symplex::prelude::*;
use symplex::stats::Distribution;
use symplex::stats::numdist::{chi2, f, gamma};
use symplex::stats::order::order_statistic;

/// `|x − reference| ≤ ulps` units in the last place of `reference`
/// (subnormals included: one unit is then `5e-324`).
fn assert_ulps(x: f64, reference: f64, ulps: u64, what: &str) {
    let d = x.to_bits().abs_diff(reference.to_bits());
    assert!(d <= ulps, "{what}: {x:e} is {d} ulps from {reference:e}");
}

/// Before: `f::isf(3.274381818895057e-36, 7.972220621509184e-39,
/// 1.3716866187107116e7)` returned `0`, whose tail is `1` (the nightly
/// `fuzz_numdist` crash, run 36114672666).  Below `x = e^{−690}` the
/// iteration uses the leading term `ln I_z(a, b) = a(u − ln r) − ln(a B(a, b))`
/// and took the upper tail as `ln1p(−e^{ln_lower})`: `−∞` once the lower
/// tail `1 − O(a)` rounds to 1, so every subnormal `x` looked "past the
/// level" and the bracket closed on 0.  The upper tail is now
/// `ln(−expm1(ln_lower))`, and `ln(a B(a, b))` (of order `a`) is formed
/// without the cancellation of `ln a + ln B(a, b)`.
#[test]
fn f_isf_of_a_tiny_numerator_df_is_a_subnormal_not_zero() {
    let (q, d1, d2) = (
        3.274381818895057e-36,
        7.972220621509184e-39,
        1.3716866187107116e7,
    );
    let x = f::isf(q, d1, d2).unwrap();
    // mpmath (dps 80): sf = lambda x: betainc(d1/2, d2/2, d1*x/(d2 + d1*x), 1,
    //   regularized=True); exp(findroot(lambda t: log(sf(exp(t))) - log(q), -733))
    //   = 2.5031831288166893552e-319, a subnormal: nearest double
    //   repr(float(...)) = 2.5032e-319
    assert_ulps(x, 2.5032e-319, 1, "f::isf");
    let back = f::sf(x, d1, d2);
    assert!(((back - q) / q).abs() < 1e-9, "sf(isf(q)) = {back:e}");
}

/// Before: `gamma::isf(7.2e-37, 1e-39, 1)` was `2.69e-220`, whose tail is
/// `5.05e-37` (30 % off the level), and `chi2::isf` the same: the leading
/// term's `ln Γ(1 + k)` was `lgamma(1 + k)`, which rounds `1 + k` to 1, and
/// the upper tail `ln1p(−e^{ln_lower})` was `−∞` near the root.
#[test]
fn gamma_and_chi2_isf_of_a_tiny_shape_meet_their_level() {
    // mpmath (dps 60): s = lambda t: gammainc(k, exp(t)/th, inf, regularized=True);
    //   exp(findroot(lambda t: log(s(t)) - log(mpf(7.2e-37)), -720 + log(th)))
    //   k = mpf(1e-39), th = 1: 1.1410152568177858291e-313
    //   th = 2 (chi2 with df = 2e-39): 2.2820305136355716582e-313
    //   th = 1e10: 1.1410152568177858291e-303
    // (subnormals: the nearest doubles, repr(float(...)), are
    // 1.14101525683e-313 and 2.28203051366e-313)
    let x = gamma::isf(7.2e-37, 1e-39, 1.0).unwrap();
    assert_ulps(x, 1.141_015_256_83e-313, 1, "gamma::isf");
    let x = chi2::isf(7.2e-37, 2e-39).unwrap();
    assert_ulps(x, 2.282_030_513_66e-313, 1, "chi2::isf");
    // A normal-range answer: the tail moves only 1.4·10⁻³ relative per unit
    // relative change of x here (d ln Q/d ln x = −k·P/Q), so the quantile's
    // own conditioning is ~700ε = 1.6·10⁻¹³; its tail meets the level to
    // 10⁻¹³.
    let x = gamma::isf(7.2e-37, 1e-39, 1e10).unwrap();
    assert!(
        ((x - 1.141_015_256_817_785_8e-303) / x).abs() < 1e-12,
        "{x:e}"
    );
    assert!(
        ((x - 1.141_015_256_817_785_8e-303) / x).abs() < 1e-10,
        "{x:e}"
    );
    let back = gamma::sf(x, 1e-39, 1e10);
    assert!(((back - 7.2e-37) / 7.2e-37).abs() < 1e-13, "{back:e}");
}

/// `|x/reference − 1| ≤ tol`.
fn assert_rel(x: f64, reference: f64, tol: f64, what: &str) {
    assert!(
        (x / reference - 1.0).abs() <= tol,
        "{what}: {x:e} vs {reference:e}"
    );
}

/// Before: a density without a closed quantile or an `f64` kernel was
/// inverted by bracketing `F_classic(x) − p` to an absolute `2·10⁻¹²`.
/// Near `p = 1` the classic CDF rounds to 1 and the root landed anywhere
/// it did: at `1 − 2⁻⁵³` a mixture of exponentials gave `35.3537` (tail
/// `1.1067·10⁻¹⁶`), the maximum of five `Exp(1)` `38.529` (tail
/// `9.25·10⁻¹⁷`), `2·T₃ + 1` `524287` (tail `6.1·10⁻¹⁷`), and
/// `3·Gamma(5/2)` `255` (tail `7.3·10⁻³⁵`).  Small levels were decided to
/// the same absolute width: the minimum of five `Exp(1)` at `10⁻¹²` was
/// `0`.  The smaller tail is now solved against its level, relative to
/// it, through the non-cancelling forms; an increasing affine map carries
/// the inner family's quantile.
#[test]
fn quantile_f64_of_a_density_without_closed_quantile_keeps_the_far_tail() {
    let ctx = Context::new();
    let top = 1.0 - f64::EPSILON / 2.0; // q = 2⁻⁵³
    let e1 = Distribution::exponential(ctx.int(1));
    let e3 = Distribution::exponential(ctx.int(3));
    let mix = Distribution::mixture(&[(ctx.rational(1, 4), e1.clone()), (ctx.rational(3, 4), e3)])
        .unwrap();
    // mpmath (dps 50): findroot(lambda x: log(exp(-x)/4 + 3*exp(-3*x)/4)
    //   - log(mpf(2)**-53), 35) = 35.35050620855721078
    assert_rel(
        mix.quantile_f64(top).unwrap(),
        35.350_506_208_557_21,
        1e-14,
        "mixture",
    );
    let max5 = order_statistic(&e1, 5, 5).unwrap();
    // mpmath: findroot(lambda x: log(1 - (1 - exp(-x))**5) - log(mpf(2)**-53), 38)
    //   = 38.346238482111201729
    assert_rel(
        max5.quantile_f64(top).unwrap(),
        38.346_238_482_111_2,
        1e-14,
        "max of 5",
    );
    let min5 = order_statistic(&e1, 5, 1).unwrap();
    // mpmath: log(mpf(2)**53)/5 = 7.3473601139354202798;
    //   -log(1 - mpf('1e-12'))/5 = 2.000000000001e-13
    assert_rel(
        min5.quantile_f64(top).unwrap(),
        7.347_360_113_935_42,
        1e-14,
        "min of 5",
    );
    assert_rel(
        min5.quantile_f64(1e-12).unwrap(),
        2.000_000_000_001e-13,
        1e-14,
        "min, low",
    );
    let t3 = Distribution::student_t(ctx.int(3))
        .affine(ctx.int(2), ctx.int(1))
        .unwrap();
    // scipy: 2*stats.t.isf(2.0**-53, 3) + 1 = 429906.9961251591;
    //   2*stats.t.ppf(1e-12, 3) + 1 = -20661.216488584974
    assert_rel(
        t3.quantile_f64(top).unwrap(),
        429_906.996_125_159_1,
        1e-14,
        "2 T3 + 1",
    );
    assert_rel(
        t3.quantile_f64(1e-12).unwrap(),
        -20_661.216_488_584_974,
        1e-14,
        "2 T3 + 1 low",
    );
    let g = Distribution::gamma(ctx.rational(5, 2), ctx.int(1)).affine(ctx.int(3), ctx.int(0));
    // mpmath: findroot(lambda x: log(gammainc(mpf(5)/2, x/3, inf, regularized=True))
    //   - log(mpf(2)**-53), 126) = 126.29254835478196827
    assert_rel(
        g.unwrap().quantile_f64(top).unwrap(),
        126.292_548_354_781_97,
        1e-14,
        "3 Gamma",
    );
}
