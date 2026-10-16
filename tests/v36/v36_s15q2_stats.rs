//! Tests of session 15 (s15q2): numdist kernels at extreme shapes, exactness claims from floats.
//! Every test says what was wrong before and cites its oracle.
//!
//! The kernel references are mpmath quadratures of the densities at the exact
//! rational values of the `f64` inputs, each smaller tail as a prefactor times
//! a standardised integral, at 30 and 50 digits (agreeing to 50):
//! `stats-hunt-scratch/session10/main/s15` Q2's `kor2.py` (N2's `kor.py` plus the
//! normal family).

use symplex::errors::Degeneracy;
use symplex::prelude::*;
use symplex::stats::cox::{CoxOpts, cox_ph};
use symplex::stats::data::{self, Ddof, Dependent, Q, from_i64};
use symplex::stats::numdist::{beta, binom, f, gamma, poisson};
use symplex::stats::regression::{LogitOpts, logit};
use symplex::stats::survival::Observation;

fn rel(actual: f64, expected: f64) -> f64 {
    ((actual - expected) / expected).abs()
}

#[track_caller]
fn close_rel(actual: f64, expected: f64, tol: f64) {
    assert!(
        rel(actual, expected) <= tol,
        "got {actual:e}, expected {expected:e} (relative {:e} > {tol:e})",
        rel(actual, expected)
    );
}

#[track_caller]
fn zero_variance<T: std::fmt::Debug>(r: Result<T, SymplexError>) {
    match r {
        Err(SymplexError::DegenerateData {
            kind: Degeneracy::ZeroVariance,
            ..
        }) => {}
        other => panic!("expected DegenerateData(ZeroVariance), got {other:?}"),
    }
}

#[track_caller]
fn invalid<T: std::fmt::Debug>(r: Result<T, SymplexError>) {
    match r {
        Err(SymplexError::InvalidArgument { .. }) => {}
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. numdist at shapes up to 1e300
// ═══════════════════════════════════════════════════════════════════════════

/// `f::cdf(1, 3.0069839314266605e156, 3.4695270304868914e166)` was `0` (and
/// `sf` `1`; N2's finding): `x = 1` puts the beta argument at the mean of
/// `Beta(d₁/2, d₂/2)`, whose standard deviation is `10⁻⁷⁸` of it, so the
/// rounding of `z = d₁x/(d₁x + d₂)` moved the tail by `10⁶²` standard
/// deviations and no shift by the density could bring it back.  Now the
/// two-large-shapes expansion takes the exact distance from the mean, `λ = 0`.
/// mpmath (30/50 digits): both tails `0.5` (to `10⁻⁷⁸`).
#[test]
fn f_at_the_mean_of_shapes_near_1e160() {
    let (d1, d2) = (3.0069839314266605e156, 3.4695270304868914e166);
    close_rel(f::cdf(1.0, d1, d2), 0.5, 1e-15);
    close_rel(f::sf(1.0, d1, d2), 0.5, 1e-15);
    // Shapes on either side, up to 1e300 (base: 0 or NaN).
    for (d1, d2) in [
        (2.8656675442598885e280, 4.0415059117941074e40),
        (1e300, 1e300),
        (1e30, 3e45),
    ] {
        close_rel(f::cdf(1.0, d1, d2), 0.5, 1e-13);
        close_rel(f::sf(1.0, d1, d2), 0.5, 1e-13);
    }
}

/// N2's second F case: `F(2.0775102050888564e198, 3.017164624654962e27)` at
/// `x = 0.9999999999995147` was `1.513017887534444e-79`, `2·10⁻⁷` off (one ulp
/// of the argument moves the tail by 10 % there; the second-order shift left
/// `O(δ³)`).  mpmath (30/50 digits): `1.5130175878292053e-79`.
#[test]
fn f_far_from_the_mean_where_one_ulp_moves_the_tail_by_ten_percent() {
    let p = f::cdf(
        0.9999999999995147,
        2.0775102050888564e198,
        3.017164624654962e27,
    );
    close_rel(p, 1.5130175878292053e-79, 1e-13);
}

/// `f::cdf(4.122471161501623e-5, 1.9793271373960822e91, 1.7596591811994766e121)`
/// was `NaN` (47 of 493 F cases of the 1e300 hunt): the tail underflowed to 0
/// and the argument's correction moved it as `0·e^s` with `e^s = ∞`.  The tail
/// is `3.16·10^(−3.9·10⁸⁹)` (mpmath), so `cdf = 0`, `sf = 1`.
#[test]
fn f_tail_below_the_range_is_zero_not_nan() {
    let (x, d1, d2) = (
        4.122471161501623e-5,
        1.9793271373960822e91,
        1.7596591811994766e121,
    );
    assert_eq!(f::cdf(x, d1, d2), 0.0);
    assert_eq!(f::sf(x, d1, d2), 1.0);
    let (x, d1, d2) = (
        1.0000000000000002,
        1.334459913591267e44,
        1.1927505454587417e83,
    );
    assert_eq!(f::cdf(x, d1, d2), 1.0);
    assert_eq!(f::sf(x, d1, d2), 0.0);
}

/// `binom::cdf(7.433941030081419e16, 1.3453467376340822e17, 0.552566944005582)`
/// was `4.38·10⁻⁸` off: beyond `n = 2⁵³` the shape `n − k` of `I_{1−p}(n − k,
/// k + 1)` is rounded by up to 4, `ε·√n` standard deviations.  The distance
/// from the mean `(n + 1)p − (k + 1)` is now formed from `n`, `k`, `p`.
/// mpmath (30/50 digits): `4.700698755402783e-71`.
#[test]
fn binomial_beyond_2_to_the_53() {
    let p = binom::cdf(
        7.433941030081419e16,
        1.3453467376340822e17,
        0.552566944005582,
    );
    close_rel(p, 4.700698755402783e-71, 2e-13);
}

/// `poisson::cdf(2.6587481374085252e16, 2.6587484297296892e16)` was `1.1·10⁻⁷`
/// off: `P(X ≤ k) = Q(k + 1, λ)` and `k + 1` rounds to `k` there, one unit,
/// which is `|z|/√λ = 1.1·10⁻⁷` of a tail 18 standard deviations out.
/// mpmath (30/50 digits): `3.592026944282755e-72`.
#[test]
fn poisson_beyond_2_to_the_53() {
    let p = poisson::cdf(2.6587481374085252e16, 2.6587484297296892e16);
    close_rel(p, 3.592026944282755e-72, 2e-13);
}

/// Two logarithms of size `ln b` cancelled at a shape near `10²⁴¹`–`10²⁸³`
/// next to one below 100: `f::sf(4.227745016084576, 5.041802347844082e283,
/// 78.18752489986493)` was `6.4·10⁻¹³` off (the recurrence's term ratio
/// `ln x + ln((a + b + j)/(a + 1 + j))`, 39 steps), `f::cdf(0.024501717730109118,
/// 105.0081531152995, 8.154460572146583e161)` `1.0·10⁻¹²` (Loader's
/// `ln(a/(n·x))` as `ln(a/n) − ln x`).  mpmath (30/50 digits):
/// `2.5248411233930486e-13`, `2.6331609887748345e-64`.
#[test]
fn one_huge_shape_next_to_a_small_one() {
    close_rel(
        f::sf(4.227745016084576, 5.041802347844082e283, 78.18752489986493),
        2.5248411233930486e-13,
        5e-14,
    );
    close_rel(
        f::cdf(
            0.024501717730109118,
            105.0081531152995,
            8.154460572146583e161,
        ),
        2.6331609887748345e-64,
        2e-13,
    );
}

/// `gamma::cdf(2.2136721344863363e35, 2.8197943828123953e37, 2.257976338387281)`
/// was `NaN` (the wave-14 shift of the tails by the density, at a tail whose
/// logarithm is `−1.3·10³⁸`: `ln f − ln T` was rounding noise and the step
/// infinite).  `x/θ = 0.0035k`, so by the Chernoff bound `P(k, u) ≤
/// e^{−k(ρ − 1 − ln ρ)}`, `ρ = u/k`, the tail is below `e^{−10³⁷}`: `cdf = 0`.
#[test]
fn gamma_far_tail_of_a_huge_shape_is_zero_not_nan() {
    for (x, k, theta) in [
        (
            2.2136721344863363e35,
            2.8197943828123953e37,
            2.257976338387281,
        ),
        (
            1.0926938839609979e151,
            1.0920030461835123e157,
            1.0006326335626075e-3,
        ),
    ] {
        assert_eq!(gamma::cdf(x, k, theta), 0.0);
        assert_eq!(gamma::sf(x, k, theta), 1.0);
    }
}

/// The quantile is a float next to the crossing of the level: the tails of
/// its neighbours bracket the level (the fuzz target's check).  Beyond shapes
/// of `10³⁰` the whole distribution lies within an ulp of its mean and the
/// solver's root can be thousands of floats from the crossing:
/// `beta::ppf(0.9574508666992188, 1.1450475699384058e67, 4.6277474460145246e297)`
/// was `2.474308685372763e-231`, whose cdf and its lower neighbour's were
/// already `1`; `f::ppf(0.37500762939453125, 1.18e180, 8.36e30)` the same;
/// `beta::isf(0.4234…, 1.03e22, 1.40e274)` was a float whose tail was
/// `0.41746`, 1.4 % below its level (the fuzz campaign with the ranges
/// extended to `10³⁰⁰`; the kernels' tails are checked against mpmath above).
#[test]
fn quantiles_where_the_distribution_lies_within_an_ulp() {
    #[track_caller]
    fn bracketed(tail: impl Fn(f64) -> f64, x: f64, level: f64) {
        let (below, above) = (tail(x.next_down()), tail(x.next_up()));
        let (lo, hi) = if below <= above {
            (below, above)
        } else {
            (above, below)
        };
        assert!(
            lo <= level * (1.0 + 1e-9) && hi >= level * (1.0 - 1e-9),
            "x = {x:e}: neighbours' tails {lo:e}, {hi:e} do not bracket {level:e}"
        );
    }
    let (a, b, p) = (
        1.1450475699384058e67,
        4.6277474460145246e297,
        0.9574508666992188,
    );
    bracketed(|x| beta::cdf(x, a, b), beta::ppf(p, a, b).unwrap(), p);
    let (d1, d2, p) = (
        1.1837045176194425e180,
        8.363881549642393e30,
        0.37500762939453125,
    );
    bracketed(|x| f::cdf(x, d1, d2), f::ppf(p, d1, d2).unwrap(), p);
    let (a, b, q) = (
        1.0332154468364324e22,
        1.3967326915254346e274,
        0.42340850830078125,
    );
    bracketed(|x| beta::sf(x, a, b), beta::isf(q, a, b).unwrap(), q);
}

/// The beta kernel itself at shapes `10³⁰⁰` next to the mean (its argument is
/// exact, so it was right before too): `I_x(a, a)` at `x = ½` is `½`.
#[test]
fn beta_at_the_mean_of_shapes_1e300() {
    close_rel(beta::cdf(0.5, 1e300, 1e300), 0.5, 1e-15);
    close_rel(beta::sf(0.5, 1e300, 1e300), 0.5, 1e-15);
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. No exactness claim from an f64 statistic
// ═══════════════════════════════════════════════════════════════════════════

/// The likelihood-ratio test of a logistic fit returned its `f64` statistic as
/// the dyadic rational of the float, and `statistic_exact()` gave it back as
/// if exact (`Some(…/2⁵⁰)` for `3.2913151402020766`; P4's finding 9).  Now the
/// statistic is `exp(ℓ)` with `ℓ` the `f64` logarithm and both accessors are
/// `None`; the values are unchanged.  scipy: `chi2.sf(3.2913151402020766, 1)
/// = 0.0696472158708212` (= statsmodels `llr_pvalue`).
#[test]
fn logit_llr_test_claims_no_exact_statistic() {
    let ctx = Context::new();
    let y: Vec<bool> = (0..20).map(|i| matches!(i, 7..=9 | 13..=19)).collect();
    let x: Vec<Vec<f64>> = (0..20)
        .map(|i| vec![if i < 10 { 0.0 } else { 1.0 }])
        .collect();
    let fit = logit(&y, &x, true, &LogitOpts::default()).unwrap();
    let t = fit.llr_test(&ctx).unwrap();
    assert_eq!(t.statistic_exact(), None);
    assert_eq!(t.p_value_exact(), None);
    close_rel(t.statistic_f64().unwrap(), fit.llr(), 4.0 * f64::EPSILON);
    close_rel(t.statistic_f64().unwrap(), 3.291_315_140_202_076_6, 1e-6);
    close_rel(t.p_value_f64().unwrap(), 0.069_647_215_870_821_2, 1e-7);
}

/// The Cox model's likelihood-ratio, Wald and score tests: the same `f64`
/// statistics read as exact rationals before.  statsmodels (`PHReg`, ten
/// subjects): LR `7.286928045319378`, Wald `5.28884651560578`, score
/// `7.355733707724492`; `chi2.sf(·, 1)` `0.006945814500177098`,
/// `0.02146243134870459`, `0.006684922420334704`.
#[test]
fn cox_test_trio_claims_no_exact_statistic() {
    let ctx = Context::new();
    let obs = Observation::from_i64(
        &[4, 7, 2, 9, 12, 5, 15, 3, 11, 8],
        &[
            true, true, true, false, true, true, false, true, true, false,
        ],
    );
    let x: Vec<Vec<f64>> = [3.0, 1.0, 5.0, 2.0, 0.0, 4.0, 1.0, 6.0, 2.0, 3.0]
        .iter()
        .map(|&v| vec![v])
        .collect();
    let fit = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    let cases = [
        (
            fit.llr_test(&ctx).unwrap(),
            fit.llr(),
            7.286928045319378,
            0.006945814500177098,
        ),
        (
            fit.wald_test(&ctx).unwrap(),
            fit.wald_statistic(),
            5.28884651560578,
            0.02146243134870459,
        ),
        (
            fit.score_test(&ctx).unwrap(),
            fit.score_statistic(),
            7.355733707724492,
            0.006684922420334704,
        ),
    ];
    for (t, s, statistic, p) in cases {
        assert_eq!(t.statistic_exact(), None);
        assert_eq!(t.p_value_exact(), None);
        close_rel(t.statistic_f64().unwrap(), s, 4.0 * f64::EPSILON);
        close_rel(t.statistic_f64().unwrap(), statistic, 1e-8);
        close_rel(t.p_value_f64().unwrap(), p, 1e-8);
    }
}

/// The exact tests keep their exact statistics (McNemar's `χ² = (b − c)²/(b +
/// c) = 64/10` for `b = 1`, `c = 9` without the continuity correction: exact
/// arithmetic).
#[test]
fn exact_tests_keep_exact_statistics() {
    let ctx = Context::new();
    let r = symplex::stats::hypothesis::mcnemar_test(&ctx, 1, 9, false, false).unwrap();
    assert_eq!(r.statistic_exact(), Some(Q::new(32.into(), 5.into())));
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Degenerate data in the descriptive statistics
// ═══════════════════════════════════════════════════════════════════════════

/// `data::pearson` (and so `spearman`), `kendall_tau`, `skewness`,
/// `kurtosis`, `zscores`, `mad_outliers` reported a constant sample (a zero
/// MAD) as `InvalidArgument`; the tests and the models report it as
/// `DegenerateData { kind: ZeroVariance }` since 0.57–0.59.  Shape errors stay
/// `InvalidArgument`.  (A constant sample has zero variance: an exact
/// identity.)
#[test]
fn constant_samples_are_degenerate_data() {
    let ctx = Context::new();
    let c = from_i64(&[4, 4, 4, 4]);
    let v = from_i64(&[1, 2, 3, 5]);
    zero_variance(data::pearson(&ctx, &c, &v));
    zero_variance(data::pearson(&ctx, &v, &c));
    zero_variance(data::spearman(&ctx, &c, &v));
    zero_variance(data::kendall_tau(&ctx, &c, &v));
    zero_variance(data::skewness(&ctx, &c));
    zero_variance(data::kurtosis(&c));
    zero_variance(data::zscores(&ctx, &c, Ddof::Sample));
    zero_variance(data::mad_outliers(
        &from_i64(&[2, 2, 2, 9]),
        &Q::new(7.into(), 2.into()),
    ));
    // Shape errors are unchanged.
    invalid(data::pearson(&ctx, &v, &v[..3]));
    invalid(data::kendall_tau(&ctx, &v[..1], &v[..1]));
    invalid(data::pearson(&ctx, &[], &[]));
    // The values themselves are unchanged: statistics.correlation([1, 2, 3, 5], [2, 4, 5, 9])
    // = 0.9944903161976938 (= 15/√227.5).
    let r = data::pearson(&ctx, &v, &from_i64(&[2, 4, 5, 9])).unwrap();
    close_rel(r.eval_f64().unwrap(), 0.994_490_316_197_693_8, 1e-14);
}

/// Goodman–Kruskal's γ, Somers' D and Stuart's τ-c when every pair is tied on
/// the variable they divide by: `InvalidArgument` before, `DegenerateData {
/// kind: ZeroVariance }` now (`C + D = 0`, `T` counts of the pairs: exact).
#[test]
fn every_pair_tied_is_degenerate_data() {
    let xc = from_i64(&[2, 2, 2, 2, 2]);
    let yv = from_i64(&[1, 2, 2, 3, 1]);
    zero_variance(data::goodman_kruskal_gamma(&xc, &yv));
    zero_variance(data::somers_d(&xc, &yv, Dependent::Y));
    zero_variance(data::kendall_tau_c(&xc, &yv));
    invalid(data::somers_d(&xc[..1], &yv[..1], Dependent::Y));
}
