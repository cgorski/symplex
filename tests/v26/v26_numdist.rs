//! 0.27 statistics bug hunt: numdist.
//!
//! Regressions from the 0.27 audit of `stats::numdist` against mpmath 1.3
//! (`symplex/.venv/bin/python`, `mp.dps = 50` unless stated, every `f64`
//! argument passed as `mpf(float)` — its exact binary value).  Each test
//! names the input, what 0.26 (or, where stated, the audit's intermediate
//! build) returned, and the oracle call.  Quantile references are the root
//! of `ln tail(x) = ln level` refined by Newton's method in mpmath from the
//! kernel's answer to 1e-35.  Nothing below was computed by hand.

// Reference values are quoted at the 17 digits mpmath printed them.
#![allow(clippy::excessive_precision)]

use std::time::{Duration, Instant};

use symplex::prelude::*;
use symplex::stats::numdist::{
    beta, betainc_regularized_f64, binom, chi2, f, gamma, gammainc_lower_regularized_f64,
    gammainc_upper_regularized_f64, poisson, t,
};

/// Relative closeness.
fn close(actual: f64, expected: f64, rel: f64, label: &str) {
    let scale = expected.abs().max(1e-300);
    assert!(
        (actual - expected).abs() <= rel * scale,
        "{label}: got {actual:e}, expected {expected:e} (relative error {:e})",
        (actual - expected).abs() / scale
    );
}

fn err_is_invalid<T: std::fmt::Debug>(r: Result<T, SymplexError>, label: &str) {
    assert!(
        matches!(r, Err(SymplexError::InvalidArgument { .. })),
        "{label}: expected an InvalidArgument error, got {r:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Quantiles at a subnormal level
// ═══════════════════════════════════════════════════════════════════════════

/// A subnormal level has a few significant bits, but it is an exact number
/// and its quantile is well conditioned.  0.26 solved `ln P(x) = ln p` with
/// `P` rounded to a subnormal: `t::ppf(4.4e-323, 111961533.3414874)` was
/// −38.410885323407236 (relative error 1.2e-5) and `chi2::ppf(5e-324,
/// 3421.755786450743)` 1139.3511613341434 (3.3e-5).
///
/// mpmath: the root of `ln(betainc(df/2, 1/2, 0, df/(df + x²))/2) = ln 4.4e-323`
/// is −38.410409280387276; of `ln gammainc(df/2, 0, x/2) = ln 5e-324`,
/// 1139.3892111622562.  Tolerance 1e-14: the logarithm of the level (≈ −744)
/// is rounded to 1.1e-13 absolute, and these quantiles move by less than
/// 1e-3 of that (`d ln x/d ln p < 10⁻³`).
#[test]
fn quantiles_at_subnormal_levels() -> Result<(), SymplexError> {
    close(
        t::ppf(4.4e-323, 111_961_533.341_487_4)?,
        -38.410409280387276,
        1e-14,
        "t.ppf(4.4e-323, 1.12e8)",
    );
    close(
        chi2::ppf(5e-324, 3421.755786450743)?,
        1139.3892111622562,
        1e-14,
        "chi2.ppf(5e-324, 3421.76)",
    );
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// The quantile solver returns the root it evaluated, corrected in x
// ═══════════════════════════════════════════════════════════════════════════

/// `gamma::ppf(1.3182698315284466e-235, 24076424999.43518, 0.11827182610906169)`:
/// 0.26 returned 2846961694.536568 (254 ulps low): after a Newton step
/// that already solved the equation it bisected once more and returned the
/// unevaluated midpoint.  mpmath: the root of `ln gammainc(k, 0, x/θ) = ln p`
/// is 2846961694.5366891862.
///
/// `gamma::ppf(1.4574149987357415e-306, 2.9819182959458717, 0.028407639328895947)`:
/// the audit's intermediate build returned 1.4042458145736066e-104 (45 ulps):
/// its objective was the difference of two logarithms near −704, whose
/// rounding (1.1e-13) hid the last digits.  mpmath root: 1.4042458145735926e-104.
///
/// `beta::ppf(7.152928814406449e-304, 6.43248805121942, 146936.90380242205)`:
/// 0.26 returned 1.6048752704356224e-52 (2.3e-14): the logit variable's
/// resolution (`|v| ≈ 119`) and Loader's prefactor far from the mean.
/// mpmath root of `ln betainc(a, b, 0, x) = ln p`: 1.6048752704355854e-52.
///
/// Tolerance 4e-15 (a few ulps): all three are well conditioned
/// (`d ln x/d ln p` is 1e-5, 0.34 and 0.16).
#[test]
fn quantile_solver_lands_on_the_root() -> Result<(), SymplexError> {
    close(
        gamma::ppf(
            1.3182698315284466e-235,
            24_076_424_999.435_18,
            0.118_271_826_109_061_69,
        )?,
        2846961694.5366891862,
        4e-15,
        "gamma.ppf(1.3e-235, 2.4e10, 0.118)",
    );
    close(
        gamma::ppf(
            1.4574149987357415e-306,
            2.9819182959458717,
            0.028407639328895947,
        )?,
        1.4042458145735926e-104,
        4e-15,
        "gamma.ppf(1.46e-306, 2.98, 0.0284)",
    );
    close(
        beta::ppf(
            7.152928814406449e-304,
            6.43248805121942,
            146_936.903_802_422_05,
        )?,
        1.6048752704355854e-52,
        4e-15,
        "beta.ppf(7.15e-304, 6.43, 1.47e5)",
    );
    Ok(())
}

/// The prefactor `xᵃyᵇ/B(a, b)` far from the mean: Loader's form carries
/// an exponent of −698 at `x = 1.6e-52` and loses 1e-13 to its rounding.
/// `beta::cdf(1.6048752704355853e-52, 6.43248805121942, 146936.90380242205)`
/// was 7.152928814404734e-304 before the fix (the audit's build; 0.26 used
/// the same prefactor); mpmath `betainc(a, b, 0, x, regularized=True)` =
/// 7.1529288144064464e-304.  Tolerance 1e-14.
#[test]
fn beta_prefactor_far_below_the_mean() {
    close(
        beta::cdf(
            1.6048752704355853e-52,
            6.43248805121942,
            146_936.903_802_422_05,
        ),
        7.1529288144064464e-304,
        1e-14,
        "beta.cdf(1.6e-52, 6.43, 1.47e5)",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Discrete quantiles: the smaller tail, compared in logarithms
// ═══════════════════════════════════════════════════════════════════════════

/// Levels near 1: `cdf(k) ≥ p` was tested on the rounded `cdf = 1 − sf`,
/// whose last bit decides at `p = 1 − 2⁻⁵²`.  0.26 returned
/// `poisson::isf(1 − 2⁻⁵², 1229036.676076251) = 1220000` and
/// `poisson::ppf(…) = 1238026`; `binom::ppf(1 − 2⁻⁵², 10391, 0.32936…) = 3814`,
/// `binom::isf(…) = 3035`.
///
/// mpmath, Poisson `P(X ≤ k) = gammainc(k + 1, λ, inf, regularized=True)`,
/// `P(X > k) = gammainc(k + 1, 0, λ, regularized=True)`, against 2⁻⁵² =
/// 2.22045e-16: `cdf(1220038) = 2.21214e-16 < 2⁻⁵² ≤ cdf(1220039) = 2.2287e-16`
/// (so `isf = 1220039`; scipy agrees); `sf(1238055) = 2.22914e-16 > 2⁻⁵² ≥
/// sf(1238056) = 2.21266e-16` (so `ppf = 1238056`; scipy says 1238055).
/// Binomial by summing the pmf `binomial(n, j) pʲ(1 − p)ⁿ⁻ʲ` at 60 digits:
/// `sf(3814) = 2.48278e-16 > 2⁻⁵² ≥ sf(3815) = 2.09658e-16` (`ppf = 3815`),
/// `cdf(3036) = 2.02899e-16 < 2⁻⁵² ≤ cdf(3037) = 2.419e-16` (`isf = 3037`).
#[test]
fn discrete_quantiles_at_levels_near_one() -> Result<(), SymplexError> {
    let p = 1.0 - 2f64.powi(-52);
    assert_eq!(p, 0.9999999999999998);
    assert_eq!(poisson::isf(p, 1_229_036.676_076_251)?, 1_220_039.0);
    assert_eq!(poisson::ppf(p, 1_229_036.676_076_251)?, 1_238_056.0);
    assert_eq!(binom::ppf(p, 10391.0, 0.32936050327658145)?, 3815.0);
    assert_eq!(binom::isf(p, 10391.0, 0.32936050327658145)?, 3037.0);
    Ok(())
}

/// Subnormal levels: a subnormal cdf has too few bits to compare with the
/// level.  0.26 returned `poisson::ppf(2e-323, 88143117.06551582) = 87782519`,
/// whose cdf 1.73e-323 rounds up to the level; and the audit's intermediate
/// build `poisson::isf(5e-324, 19827.03680102543) = 25483`, whose sf
/// 6.8e-324 rounds down to it.
///
/// mpmath (as above): `cdf(87782551) = 1.97343e-323 < 2e-323 (= 1.97626e-323)
/// ≤ cdf(87782552) = 1.98154e-323`; `sf(25484) = 5.3049e-324 > 5e-324
/// (= 4.94066e-324) ≥ sf(25485) = 4.12642e-324`.
#[test]
fn discrete_quantiles_at_subnormal_levels() -> Result<(), SymplexError> {
    assert_eq!(poisson::ppf(2e-323, 88_143_117.065_515_82)?, 87_782_552.0);
    assert_eq!(poisson::isf(5e-324, 19_827.036_801_025_43)?, 25_485.0);
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Tiny shapes
// ═══════════════════════════════════════════════════════════════════════════

/// The regularised incomplete gamma function for a shape below 1: 0.26
/// took `P` from the power series and `Q = 1 − P` for every `x < a + 1`,
/// with a prefactor whose `ln Γ(a) ≈ −ln a` cancelled.
/// `gammainc_upper_regularized_f64(1e-300, 0.9)` was −4.6407322429331543e-14
/// (a negative probability), `(1e-10, 0.5)` 5.598044250376688e-11 (5.5e-5
/// off), and `gammainc_lower_regularized_f64(1e-300, 0.9)` 1.0000000000000464.
/// mpmath `gammainc(a, x, inf, regularized=True)`: 2.6018393932599964e-301,
/// 5.5977359480549881e-11, and `gammainc(1e-300, 0, 0.9, …)` = 1.0 (to 50
/// digits).  `gammainc(1e-6, 0.9, inf, …)` = 2.6018418516621042e-7.
/// Tolerance 1e-14.
#[test]
fn incomplete_gamma_of_a_tiny_shape() {
    close(
        gammainc_upper_regularized_f64(1e-300, 0.9),
        2.6018393932599964e-301,
        1e-14,
        "Q(1e-300, 0.9)",
    );
    close(
        gammainc_upper_regularized_f64(1e-10, 0.5),
        5.5977359480549881e-11,
        1e-14,
        "Q(1e-10, 0.5)",
    );
    close(
        gammainc_upper_regularized_f64(1e-6, 0.9),
        2.6018418516621042e-7,
        1e-14,
        "Q(1e-6, 0.9)",
    );
    let p = gammainc_lower_regularized_f64(1e-300, 0.9);
    assert!(p <= 1.0 && p > 1.0 - 1e-15, "P(1e-300, 0.9) = {p:e}");
    // A subnormal shape: the series began with 1/a = ∞ and 0.26 returned
    // cdf = ∞, sf = −∞.  mpmath `gammainc(5e-324, 0.5, inf, regularized=True)`
    // = a·E₁(½) ≈ 2.77e-324, so P rounds to 1.
    assert_eq!(gamma::cdf(0.5, 5e-324, 1.0), 1.0);
    let s = gamma::sf(0.5, 5e-324, 1.0);
    assert!((0.0..=1e-323).contains(&s), "gamma.sf(0.5, 5e-324) = {s:e}");
}

/// χ² with 5.08e-144 degrees of freedom at `x = 5e-324`: the scaled
/// argument `x/2` underflows, and 0.26 took `sf = 1 − cdf` with `cdf`
/// rounded to 1 — `chi2::sf(5e-324, 5.08e-144) = 0`, so `chi2::isf` of any
/// level answered 5e-324.  mpmath `gammainc(df/2, 5e-324/2, inf,
/// regularized=True)` = 1.8914873437841932e-141; and the root of
/// `ln gammainc(df/2, x/2, inf) = ln 4.933159892566488e-308` is
/// 742.0826169820976.  Tolerance 1e-13 (a tail of order `df`, whose relative
/// error is that of `ln(x/2)`, 744).
#[test]
fn chi2_with_a_tiny_df_at_a_subnormal_x() -> Result<(), SymplexError> {
    let df = 5.0808463971888157e-144;
    close(
        chi2::sf(5e-324, df),
        1.8914873437841932e-141,
        1e-13,
        "chi2.sf",
    );
    close(
        chi2::isf(4.933159892566488e-308, df)?,
        742.0826169820976,
        1e-13,
        "chi2.isf",
    );
    Ok(())
}

/// The incomplete beta function with a tiny second shape (TOMS 708's
/// `bgrat`): 0.26 formed `1/Γ(1 + b) − 1` from `lgamma(1 + b)`, which is 0
/// for `b < ε`, and carried `ln Γ(b) ≈ −ln b` through an exponent.
/// `betainc_regularized_f64(1e6, 1e-20, 0.999999)` was 7.965995992864131e-21,
/// `(50, 1e-20, 0.99)` 1.1400280802357385e-20, and `(268.02537176493655,
/// 9.859247047821542e-113, 0.997453067054673)` 9.5029474464886e-113 (a
/// factor 2.5 high).  mpmath `betainc(a, b, 0, x, regularized=True)`:
/// 2.1938393438488033e-21, 5.6281241533420565e-21, 3.8120356063518139e-113.
/// Tolerance 1e-13 (the first is conditioned `x·f/I ≈ 1e6·10⁻⁶`, 1e-15 per
/// ulp of x; the kernel agrees to 4e-16).
#[test]
fn incomplete_beta_with_a_tiny_second_shape() {
    close(
        betainc_regularized_f64(1e6, 1e-20, 0.999999),
        2.1938393438488033e-21,
        1e-13,
        "I(1e6, 1e-20, 0.999999)",
    );
    close(
        betainc_regularized_f64(50.0, 1e-20, 0.99),
        5.6281241533420565e-21,
        1e-13,
        "I(50, 1e-20, 0.99)",
    );
    close(
        betainc_regularized_f64(
            268.02537176493655,
            9.859247047821542e-113,
            0.997453067054673,
        ),
        3.8120356063518139e-113,
        1e-13,
        "I(268, 9.9e-113, 0.9975)",
    );
}

/// Two tiny shapes: `ln(ab/(a + b))` in Loader's prefactor underflowed
/// (`a·b = 6.9e-394`) to −∞ and dropped the dominant term of `bup`.
/// `betainc_regularized_f64(1.2740278775145818e-111, 5.447155739457082e-283,
/// 0.9999999999999999)` was 1.8078598363689716e-281 before the fix (the
/// audit's build; the prefactor is 0.26's); mpmath at `mp.dps = 120`:
/// 4.2755388917262821e-172 (≈ b/(a + b)).  Tolerance 1e-13.
#[test]
fn incomplete_beta_with_two_tiny_shapes() {
    close(
        betainc_regularized_f64(
            1.2740278775145818e-111,
            5.447155739457082e-283,
            0.9999999999999999,
        ),
        4.2755388917262821e-172,
        1e-13,
        "I(1.3e-111, 5.4e-283, 1 - 2^-53)",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// The F distribution beyond the normal range of its beta arguments
// ═══════════════════════════════════════════════════════════════════════════

/// `y = d₂/(d₁x + d₂)` underflows long before the upper tail does.
/// `f::sf(7.196363823012135e239, 103469.01858561047, 6.361254171612904e-154)`
/// was 0 in 0.26 (and `cdf` 1): the lower tail is the small one.  mpmath at
/// `mp.dps = 300`, `1 − betainc(d₂/2, d₁/2, 0, d₂/(d₁x + d₂), regularized=True)`
/// = 2.878961251086338e-151.
///
/// `f::sf(2.209895240751112e-279, 4.205702632851584e-283, 1.8940012656387664)`:
/// 0.26 subtracted `1 − cdf` with `cdf ≈ 1` and returned 1.1368683772161603e-13;
/// mpmath at `mp.dps = 700`, `1 − betainc(d₁/2, d₂/2, 0, d₁x/(d₁x + d₂))` =
/// 2.7180464709183697e-280.
///
/// `f::sf(3.601178276708252e123, 3.2629542587305496e-121, 1956.1757798194587)`:
/// the audit's build multiplied `(b/a)·yᵃ` into a subnormal mantissa and
/// returned 0; mpmath `betainc(d₂/2, d₁/2, 0, y, regularized=True)` =
/// 6.5716827933481018e-324 (cross-checked with `yᵇ(1 − y)ᵃ/(b B(b, a))·
/// hyp2f1(a + b, 1, b + 1, y)`), which rounds to 5e-324.
/// Tolerance 1e-13 for the normal values.
#[test]
fn f_tails_beyond_the_normal_range_of_the_beta_arguments() {
    let (x, d1, d2) = (
        7.196363823012135e239,
        103_469.018_585_610_47,
        6.361254171612904e-154,
    );
    close(
        f::cdf(x, d1, d2),
        2.878961251086338e-151,
        1e-13,
        "f.cdf(7.2e239, ...)",
    );
    assert_eq!(f::sf(x, d1, d2), 1.0);
    close(
        f::sf(
            2.209895240751112e-279,
            4.205702632851584e-283,
            1.8940012656387664,
        ),
        2.7180464709183697e-280,
        1e-13,
        "f.sf(2.2e-279, 4.2e-283, 1.89)",
    );
    assert_eq!(
        f::sf(
            3.601178276708252e123,
            3.2629542587305496e-121,
            1956.1757798194587
        ),
        5e-324
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Where the quantile lies outside the positive floats
// ═══════════════════════════════════════════════════════════════════════════

/// An upper level with a tiny first shape: every positive `x` already has
/// `P(X > x) < 10⁻²⁹⁷`, so the quantile over the floats is the smallest
/// one.  0.26 returned `NaN` for this class of inputs (e.g.
/// `f::isf(0.3, 1e-300, 1e-100)`).  mpmath: `f.sf(5e-324, 1e-300, 1e10) =
/// betainc(d₂/2, d₁/2, 0, d₂/(d₁x + d₂), regularized=True)` = 7.1766576566767671e-298.
#[test]
fn quantile_below_every_positive_float() -> Result<(), SymplexError> {
    let tiniest = f64::from_bits(1);
    assert_eq!(f::isf(1e-10, 1e-300, 1e10)?, tiniest);
    assert_eq!(f::isf(0.3, 1e-300, 1e-100)?, tiniest);
    assert_eq!(beta::isf(0.3, 1e-300, 1e-100)?, tiniest);
    Ok(())
}

/// `beta::isf(0.1, 2, 1e20)`: the normal start `x₀ ≈ 3.8e-20` went through
/// `ln(1 − x₀) − …` with `1 − x₀ = 1`, the iteration started at `−∞`, and
/// the answer was 0 (the audit's build; 0.26 had the same start).  mpmath
/// root of `ln betainc(a, b, x, 1) = ln 0.1`: 3.889720169867429e-20.
#[test]
fn beta_upper_quantile_with_a_huge_second_shape() -> Result<(), SymplexError> {
    close(
        beta::isf(0.1, 2.0, 1e20)?,
        3.889720169867429e-20,
        1e-14,
        "beta.isf(0.1, 2, 1e20)",
    );
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Huge shapes
// ═══════════════════════════════════════════════════════════════════════════

/// Beyond `k ≈ 10³⁰` the gamma distribution is narrower than the resolution
/// of `ln x`.  0.26 returned `gamma::ppf(0.3, 1e30) = e²·10³⁰` and failed to
/// converge for `k = 10³⁰⁰` (9 ms).  scipy agrees with the new answers:
/// `stats.gamma.ppf(0.3, 1e30) = 9.999999999999995e+29`,
/// `stats.gamma.ppf(0.5, 1e300) = 1e+300`; the Cornish–Fisher expansion
/// `k + z√k + (z² − 1)/3 + (z³ − 7z)/(36√k)` at 50 digits, `z = −√2·erfinv(0.4)`
/// (mpmath), is 9.999999999999994756e29, which rounds to 9.999999999999995e29.
#[test]
fn gamma_quantiles_for_huge_shapes() -> Result<(), SymplexError> {
    assert_eq!(gamma::ppf(0.3, 1e30, 1.0)?, 9.999999999999995e29);
    assert_eq!(gamma::ppf(0.5, 1e300, 1.0)?, 1e300);
    let start = Instant::now();
    for _ in 0..100 {
        let _ = gamma::ppf(0.5, 1e300, 1.0)?;
    }
    assert!(
        start.elapsed() < Duration::from_secs(1),
        "{:?}",
        start.elapsed()
    );
    Ok(())
}

/// Near `f64::MAX`: `k + m` overflowed in `bd0`, which then took its
/// small-difference series for any pair — `gamma::sf(f64::MAX, 1e300)` was
/// 2.2191901102808667e-159 (the audit's build; 0.26's `bd0`), truly
/// `e^{−1.8·10³⁰⁸}` = 0.  And the continued fraction for `x ≫ a` never met
/// its tolerance: `chi2::cdf(f64::MAX, 1e300)` took 48 ms in 0.26 (ten
/// million iterations).
#[test]
fn gamma_tails_near_f64_max() {
    assert_eq!(gamma::sf(f64::MAX, 1e300, 1.0), 0.0);
    let start = Instant::now();
    for _ in 0..100 {
        assert_eq!(chi2::cdf(f64::MAX, 1e300), 1.0);
    }
    assert!(
        start.elapsed() < Duration::from_secs(1),
        "{:?}",
        start.elapsed()
    );
}

/// `t::ppf(1e-300, 1e100)`: 0.26 failed ("did not converge in 200 steps") —
/// a Newton step from a poor start went to `u = −5.8e102` and bisection ran
/// out of budget.  mpmath at `mp.dps = 160`, the root of
/// `ln(betainc(df/2, 1/2, 0, df/(df + x²))/2) = ln 1e-300`: −37.047096299361199.
#[test]
fn t_quantile_with_1e100_degrees_of_freedom() -> Result<(), SymplexError> {
    close(
        t::ppf(1e-300, 1e100)?,
        -37.047096299361199,
        1e-15,
        "t.ppf(1e-300, 1e100)",
    );
    Ok(())
}

/// A distribution narrower than one float spacing: Beta(10¹⁰⁰, 10¹⁰⁰) has
/// standard deviation `σ = 1/(2√(2·10¹⁰⁰ + 1)) ≈ 3.5·10⁻⁵¹`, so every
/// quantile of a level in `[10⁻³⁰⁰, ½]` lies within `38σ ≈ 1.3·10⁻⁴⁹` of ½
/// and its nearest double is ½.  The audit's build (0.26's log-variable
/// iteration) answered `beta::isf(1e-300, 1e100, 1e100)` with
/// 0.2887654057724061.
#[test]
fn quantile_of_a_distribution_narrower_than_a_float() -> Result<(), SymplexError> {
    assert_eq!(beta::isf(1e-300, 1e100, 1e100)?, 0.5);
    assert_eq!(beta::isf(0.3, 1e100, 1e100)?, 0.5);
    assert_eq!(beta::ppf(0.3, 1e100, 1e100)?, 0.5);
    Ok(())
}

/// `α + β` must be a double: every route of the incomplete beta forms it.
/// The audit's build answered `beta::ppf(0.3, f64::MAX, f64::MAX)` with 0
/// and `beta::isf(1e-300, f64::MAX, f64::MAX)` with 1 (the mean is ½); now
/// the tails are `NaN` (an invalid parameter, as documented) and the
/// quantiles an `InvalidArgument`.
#[test]
fn beta_with_an_overflowing_shape_sum() {
    assert!(beta::cdf(0.5, f64::MAX, f64::MAX).is_nan());
    err_is_invalid(
        beta::ppf(0.3, f64::MAX, f64::MAX),
        "beta.ppf(0.3, MAX, MAX)",
    );
    err_is_invalid(
        beta::isf(1e-300, f64::MAX, f64::MAX),
        "beta.isf(1e-300, MAX, MAX)",
    );
}

/// A very flat tail: F with `d₂ = 2.1e-4` has `P(X > x) ∝ x^{−d₂/2}`, and
/// its upper quantile moves 9538 times faster than the level
/// (`d ln x/d ln q`, mpmath).  A fuzz find: the audit's build (0.26's rule
/// "a small step that does not reduce |g| means the rounding noise has been
/// reached") stopped at 2.1338525018870388e219, whose tail is 0.9476 — the
/// small step came from a meaningless slope at an overflowed `eᵘ`.  mpmath
/// root of `ln betainc(d₂/2, d₁/2, 0, d₂/(d₁x + d₂)) = ln q`:
/// 2.3289227604580066e298.  Tolerance 1e-12: the conditioning times the
/// tail's own accuracy (≈ 1e-16).
#[test]
fn f_upper_quantile_of_a_very_flat_tail() -> Result<(), SymplexError> {
    close(
        f::isf(
            9.296646118164063e-1,
            3.606674002987266e6,
            2.0969297752037127e-4,
        )?,
        2.3289227604580066e298,
        1e-12,
        "f.isf(0.93, 3.6e6, 2.1e-4)",
    );
    Ok(())
}

/// Shapes near 10²⁴ put one float step of the quantile at 1e-4 of the tail;
/// the answer must then be one of the two floats between which the level
/// is crossed (a fuzz find: the audit's build returned a float a step
/// outside that pair, `gamma::isf(3.77655029296875e-3, 2.699106113000985e24,
/// 1.7529231428024996)`, whose tail and its neighbour's were both below the
/// level).  Checked against the kernel's own tails, which is what the pair
/// means: `x` and a neighbour bracket `q`.
#[test]
fn quantile_is_one_of_the_floats_where_the_level_is_crossed() -> Result<(), SymplexError> {
    let (q, k, s) = (
        3.77655029296875e-3,
        2.699106113000985e24,
        1.7529231428024996,
    );
    let x = gamma::isf(q, k, s)?;
    let (below, here, above) = (
        gamma::sf(x.next_down(), k, s),
        gamma::sf(x, k, s),
        gamma::sf(x.next_up(), k, s),
    );
    assert!(
        (below > q && here <= q) || (here > q && above <= q),
        "sf around {x:e}: {below:e}, {here:e}, {above:e}; q = {q:e}"
    );
    Ok(())
}
