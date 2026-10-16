//! Tests of session 15 (s15l3): hypothesis tests, confidence intervals, estimation.
//! Every test says what was wrong before and cites its oracle.
//!
//! The values were found by the differential hunter of wave 12 (symplex
//! against scipy 1.18.1 / statsmodels 0.15.0, every disagreement settled by
//! an exact computation or mpmath 1.3 at 40–90 digits).  Where symplex is
//! right and the library wrong, the test records the library's value.

#![allow(unused_imports)]
use std::time::Duration;

use symplex::linprog::{q, qi};
use symplex::prelude::*;
use symplex::stats::data::{Q, from_i64};
use symplex::stats::estimation::{
    CLOPPER_PEARSON_EXACT_MAX_TRIALS, IntervalMethod, confidence_interval_mean,
    proportion_interval_exact,
};
use symplex::stats::hypothesis::*;

use crate::perf_assert::CpuTimer;

fn rel_close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs()
}

fn is_computation_failed<T: std::fmt::Debug>(r: &Result<T, SymplexError>) -> bool {
    matches!(r, Err(SymplexError::ComputationFailed { .. }))
}

/// The G-test of a 5 × 5 table (`ν = 16`): reading its p-value hung before
/// 0.57 (other tables of the hunt exhausted a 3 GB heap): `eval` rewrites
/// `uppergamma(8, G/2)/Γ(8)` into its closed form and *expands* it, and
/// `(G/2)^k` with `G/2` a sum of some 25 logarithms has millions of terms.
/// The p-value is now written out as the finite sum `e^{−x} Σ x^k/k!`.
/// Oracle: mpmath (40 digits) `G = 2 Σ O ln(O N/(R C))
/// = 14.411159933779008562404`, `gammainc(8, G/2, ∞, regularized=True)
/// = 0.56811220108836705508867543`; scipy 1.18.1 `chi2_contingency(t,
/// correction=False, lambda_='log-likelihood')` → (14.411159933779008,
/// 0.5681122010883675).
#[test]
fn g_test_five_by_five_table_evaluates() {
    let ctx = Context::new();
    let t = counts(&[
        &[3, 1, 4, 0, 3],
        &[1, 2, 1, 3, 4],
        &[1, 1, 2, 3, 1],
        &[1, 1, 4, 2, 1],
        &[2, 2, 1, 1, 1],
    ]);
    let r = g_test(&ctx, &t).unwrap();
    assert_eq!(r.df, Some(ctx.int(16)));
    assert!(rel_close(
        r.statistic_f64().unwrap(),
        14.411_159_933_779_008,
        1e-14
    ));
    assert!(rel_close(
        r.p_value_f64().unwrap(),
        0.568_112_201_088_367,
        1e-15
    ));
    assert_eq!(r.p_value_decimal(20).unwrap(), "0.56811220108836705509");
    assert!((r.p_value_log10().unwrap() - 0.568_112_201_088_367_f64.log10()).abs() < 1e-14);
}

/// The finite-sum tail for odd and small `ν` (the `erfc(√x)` branch).
/// mpmath (40 digits): `[[12, 5], [3, 9]]` (`ν = 1`) p =
/// 0.013711696120324077319052357; `[[10, 20, 30, 7], [6, 9, 17, 25]]`
/// (`ν = 3`) p = 0.00029140868491073933561495; scipy 1.18.1 gives
/// 0.0137116961203241 and 0.0002914086849107396.
#[test]
fn g_test_odd_degrees_of_freedom() {
    let ctx = Context::new();
    let r = g_test(&ctx, &counts(&[&[12, 5], &[3, 9]])).unwrap();
    assert!(rel_close(
        r.p_value_f64().unwrap(),
        0.013_711_696_120_324_077,
        1e-15
    ));
    let r = g_test(&ctx, &counts(&[&[10, 20, 30, 7], &[6, 9, 17, 25]])).unwrap();
    assert_eq!(r.df, Some(ctx.int(3)));
    assert!(rel_close(
        r.p_value_f64().unwrap(),
        2.914_086_849_107_393_4e-4,
        1e-15
    ));
}

/// The pooled two-sample t-test with a single observation in one sample:
/// `InvalidArgument` ("the second sample needs at least 2 observations")
/// before 0.57, though the pooled variance is defined (`df = n₁ + n₂ − 2`).
/// scipy 1.18.1 `ttest_ind([5], [1, 2, 3], alternative=…)`: statistic
/// 2.598076211353316 (= 3√3/2), p 0.12168993434632014 / 0.9391550328268399
/// (less) / 0.06084496717316007 (greater); `ttest_ind([5], [1, 2])`:
/// 4.041451884327381, p 0.1544209583112665.  mpmath: `betainc(1, ½, 0,
/// 2/(2 + 27/4), regularized=True) = 0.12168993434632013858`.
#[test]
fn pooled_t_test_accepts_a_single_observation() {
    let ctx = Context::new();
    let (x, y) = (from_i64(&[5]), from_i64(&[1, 2, 3]));
    for (alt, p) in [
        (Alternative::TwoSided, 0.121_689_934_346_320_14),
        (Alternative::Less, 0.939_155_032_826_839_9),
        (Alternative::Greater, 0.060_844_967_173_160_07),
    ] {
        let r = t_test_two_sample(&ctx, &x, &y, true, alt).unwrap();
        assert!(rel_close(
            r.statistic_f64().unwrap(),
            2.598_076_211_353_316,
            1e-15
        ));
        assert!(rel_close(r.p_value_f64().unwrap(), p, 1e-14), "{alt:?}");
    }
    let r = t_test_two_sample(&ctx, &x, &from_i64(&[1, 2]), true, Alternative::TwoSided).unwrap();
    assert_eq!(r.df, Some(ctx.int(1)));
    assert!(rel_close(
        r.p_value_f64().unwrap(),
        0.154_420_958_311_266_5,
        1e-14
    ));
    // Two observations in all leave no degree of freedom; Welch needs both variances.
    assert!(t_test_two_sample(&ctx, &x, &from_i64(&[7]), true, Alternative::TwoSided).is_err());
    assert!(t_test_two_sample(&ctx, &x, &y, false, Alternative::TwoSided).is_err());
}

/// Sample sizes beyond `2⁴⁰` were refused ("the required sample size
/// exceeds 2^40") before 0.57, though they are ordinary answers for close
/// proportions.  Oracle: mpmath (50 digits) with `h = 2 asin √p₁ − 2 asin
/// √p₂ = −2.0000000000588446624e-6` at the binary values of the inputs:
/// the two-sided power crosses 0.8 at `n = 3924430254432.168`, so the
/// answer is 3924430254433.  Better than statsmodels 0.15.0:
/// `NormalIndPower().solve_power(proportion_effectsize(0.5, 0.500001),
/// alpha=0.05, power=0.8, ratio=1) = 3924430254437.4`, because its
/// `proportion_effectsize` = −2.0000000000575113e-06 loses `6.7·10⁻¹³` to
/// cancellation (symplex uses the non-cancelling form of Cohen's `h`).
#[test]
fn sample_size_two_proportions_beyond_two_to_the_forty() {
    let n = sample_size_two_proportions(0.5, 0.500_001, 0.05, 0.8).unwrap();
    assert!(n > 1 << 40);
    assert_eq!(n, 3_924_430_254_433);
}

/// The exact Clopper–Pearson interval names its ends as roots of degree-`n`
/// polynomials; at `n = 435 511` it built that polynomial and hung (60 s
/// timeout of the hunter) before 0.57.  Now more than
/// `CLOPPER_PEARSON_EXACT_MAX_TRIALS` trials are refused at once when
/// `0 < k < n`, the boundary cases have closed forms (oracle: the Beta
/// quantile `1 − (α/2)^{1/n}` written with `expm1`), and small
/// `n` still agree with scipy 1.18.1 `beta.ppf(0.025, 3, 8) =
/// 0.06673951117773447`, `beta.ppf(0.975, 4, 7) = 0.6524528500599973`.
#[test]
fn clopper_pearson_exact_refuses_large_n_at_once() {
    let ctx = Context::new();
    let start = CpuTimer::start();
    let r = proportion_interval_exact(
        &ctx,
        41_220,
        435_511,
        &q(9, 10),
        IntervalMethod::ClopperPearson,
    );
    assert!(is_computation_failed(&r));
    let r = proportion_interval_exact(
        &ctx,
        3,
        CLOPPER_PEARSON_EXACT_MAX_TRIALS + 1,
        &q(95, 100),
        IntervalMethod::ClopperPearson,
    );
    assert!(is_computation_failed(&r));
    start.assert_within(Duration::from_secs(5), "two refusals");
    // k = 0 / k = n at any n: the closed forms 1 − (α/2)^{1/n} / (α/2)^{1/n}
    // (RootOf nodes before 0.57, which at n = 183 took more than a minute).
    let ci = proportion_interval_exact(
        &ctx,
        0,
        435_511,
        &q(95, 100),
        IntervalMethod::ClopperPearson,
    )
    .unwrap();
    let expected = -(0.025_f64.ln() / 435_511.0).exp_m1();
    assert!(rel_close(ci.upper.eval_f64().unwrap(), expected, 1e-13));
    let ci = proportion_interval_exact(&ctx, 200, 200, &q(95, 100), IntervalMethod::ClopperPearson)
        .unwrap();
    assert!(rel_close(
        ci.lower.eval_f64().unwrap(),
        (0.025_f64.ln() / 200.0).exp(),
        1e-14
    ));
    start.assert_within(Duration::from_secs(10), "two refusals and two closed forms");
    let ci = proportion_interval_exact(&ctx, 3, 10, &q(95, 100), IntervalMethod::ClopperPearson)
        .unwrap();
    assert!(rel_close(
        ci.lower.eval_f64().unwrap(),
        0.066_739_511_177_734_47,
        1e-14
    ));
    assert!(rel_close(
        ci.upper.eval_f64().unwrap(),
        0.652_452_850_059_997_3,
        1e-14
    ));
}

/// Exact Mann–Whitney with ties: `RankMethod::Exact` was an error ("the
/// exact method needs a pooled sample without ties") before 0.57.  It is
/// now the exact distribution of `U` given the ties.  Oracle: enumeration
/// of all `C(11, 5) = 462` splits of the pooled average ranks (Python
/// fractions): two-sided 40/231, less 20/231, greater 31/33.  scipy 1.18.1
/// `mannwhitneyu(x, y, method='exact')` applies the no-ties distribution:
/// 0.17748917748917747 (= 41/231), 0.08874458874458874, 0.9372294372294372;
/// its asymptotic method gives 0.1620516143696369.
#[test]
fn mann_whitney_exact_with_ties() {
    let ctx = Context::new();
    let (x, y) = (from_i64(&[1, 2, 2, 3, 5]), from_i64(&[2, 3, 3, 4, 6, 6]));
    for (alt, p) in [
        (Alternative::TwoSided, q(40, 231)),
        (Alternative::Less, q(20, 231)),
        (Alternative::Greater, q(31, 33)),
    ] {
        let r = mann_whitney_u(&ctx, &x, &y, alt, RankMethod::Exact).unwrap();
        assert_eq!(r.statistic_exact(), Some(qi(7)));
        assert_eq!(r.p_value_exact(), Some(p), "{alt:?}");
        // Swapping the samples swaps the one-sided alternatives.
        let flipped = match alt {
            Alternative::Less => Alternative::Greater,
            Alternative::Greater => Alternative::Less,
            a => a,
        };
        let s = mann_whitney_u(&ctx, &y, &x, flipped, RankMethod::Exact).unwrap();
        assert_eq!(s.p_value, r.p_value);
    }
    // An asymmetric conditional distribution (U = 7/2): enumeration of the
    // C(13, 6) = 1716 splits gives 1/66 two-sided (doubled smaller tail),
    // 1/132 less, 1715/1716 greater; scipy 1.18.1 exact: 0.013986013986013986.
    let (x, y) = (
        from_i64(&[1, 1, 2, 2, 2, 3]),
        from_i64(&[2, 3, 3, 4, 4, 4, 5]),
    );
    let r = mann_whitney_u(&ctx, &x, &y, Alternative::TwoSided, RankMethod::Exact).unwrap();
    assert_eq!(r.statistic_exact(), Some(q(7, 2)));
    assert_eq!(r.p_value_exact(), Some(q(1, 66)));
    let r = mann_whitney_u(&ctx, &x, &y, Alternative::Greater, RankMethod::Exact).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(1715, 1716)));
}

/// Exact Wilcoxon signed-rank with ties in `|d|`: an error before 0.57
/// ("the exact method needs |differences| without ties").  Oracle:
/// enumeration of the `2⁸` sign patterns of the average ranks: `T⁺ = 53/2`,
/// two-sided 1/4, less 113/128, greater 1/8.  scipy 1.18.1 `wilcoxon(d,
/// method='exact')` applies the no-ties distribution: 0.3125, 0.90234375,
/// 0.15625.  Second sample: two-sided 1/32, greater 1/64 (scipy
/// 0.02734375, 0.013671875).
#[test]
fn wilcoxon_exact_with_ties() {
    let ctx = Context::new();
    let d = from_i64(&[1, -2, 2, 3, 3, 3, -4, 5]);
    let r = wilcoxon_signed_rank(&ctx, &d, None, Alternative::TwoSided, RankMethod::Exact).unwrap();
    assert_eq!(r.statistic_exact(), Some(q(19, 2)));
    assert_eq!(r.p_value_exact(), Some(q(1, 4)));
    let r = wilcoxon_signed_rank(&ctx, &d, None, Alternative::Less, RankMethod::Exact).unwrap();
    assert_eq!(r.statistic_exact(), Some(q(53, 2)));
    assert_eq!(r.p_value_exact(), Some(q(113, 128)));
    let r = wilcoxon_signed_rank(&ctx, &d, None, Alternative::Greater, RankMethod::Exact).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(1, 8)));
    let d = from_i64(&[2, 2, -2, 4, 4, 6, -1, 7, 7]);
    let r = wilcoxon_signed_rank(&ctx, &d, None, Alternative::TwoSided, RankMethod::Exact).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(1, 32)));
    let r = wilcoxon_signed_rank(&ctx, &d, None, Alternative::Greater, RankMethod::Exact).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(1, 64)));
}

/// Beyond the work budget the exact tied distribution is refused with
/// `ComputationFailed`, not computed for minutes.
#[test]
fn exact_rank_tests_with_ties_have_a_budget() {
    let ctx = Context::new();
    let x: Vec<i64> = (0..400).map(|i| i % 7).collect();
    let y: Vec<i64> = (0..400).map(|i| i % 5).collect();
    let r = mann_whitney_u(
        &ctx,
        &from_i64(&x),
        &from_i64(&y),
        Alternative::TwoSided,
        RankMethod::Exact,
    );
    assert!(is_computation_failed(&r));
}

/// Better than scipy: data far from the origin.  scipy 1.18.1
/// `ttest_1samp(10¹⁵ + [1, 3, 4, 8, 9, 13, 14, 15, 21, 22, 30, 31], 10¹⁵ + 10)`
/// → statistic 1.5084049993023805, pvalue 0.15962398163985522: the
/// floating mean and variance lose the digits that matter (7 % error in
/// `p`).  Exact: `t = 1.4654314396145468741`, mpmath `p =
/// 0.17079471996613417532`.
#[test]
fn t_test_far_from_origin_is_exact() {
    let ctx = Context::new();
    let base = 1_000_000_000_000_000_i64;
    let x = from_i64(&[1, 3, 4, 8, 9, 13, 14, 15, 21, 22, 30, 31].map(|v| base + v));
    let r = t_test_one_sample(&ctx, &x, &qi(base + 10), Alternative::TwoSided).unwrap();
    assert!(rel_close(
        r.statistic_f64().unwrap(),
        1.465_431_439_614_547,
        1e-15
    ));
    assert!(rel_close(
        r.p_value_f64().unwrap(),
        0.170_794_719_966_134_18,
        1e-15
    ));
}

/// Better than scipy: a t interval at confidence `1 − 10⁻⁹`.  scipy 1.18.1
/// `t.interval(0.999999999, 6, loc=mean, scale=sem)` = (−88.69393126050197,
/// 105.83678660126519): it forms `(1 + c)/2` and loses the tail.  mpmath
/// (the quantile at the exact binary value of `c`): upper end
/// 105.83678840335911008.
#[test]
fn t_interval_at_extreme_confidence() {
    let ci = confidence_interval_mean(&from_i64(&[3, 5, 6, 9, 11, 12, 14]), 0.999_999_999).unwrap();
    assert!(rel_close(ci.upper, 105.836_788_403_359_11, 1e-12));
    assert!(rel_close(ci.lower, -88.693_931_260_501_97, 1e-12));
}

/// Better than scipy: a p-value below the `f64` range.  scipy 1.18.1
/// `binomtest(0, 2000, 0.5, alternative='less').pvalue = 0.0`; the exact
/// value is `2⁻²⁰⁰⁰` and `log10 p = −2000 log10 2 =
/// −602.05999132796239043` (mpmath).  (Not a change of this session: a
/// record of the comparison.)
#[test]
fn binomial_far_tail_is_exact() {
    let ctx = Context::new();
    let r = binomial_test(&ctx, 0, 2000, &q(1, 2), Alternative::Less).unwrap();
    let two = num_bigint::BigInt::from(2);
    assert_eq!(
        r.p_value_exact(),
        Some(Q::new(1.into(), num_traits::Pow::pow(&two, 2000u32)))
    );
    assert_eq!(r.p_value_f64().unwrap(), 0.0);
    assert!((r.p_value_log10().unwrap() + 602.059_991_327_962_4).abs() < 1e-10);
    assert!(r.p_value_decimal(12).unwrap().ends_with("e-603"));
}

/// Better than scipy: an exactly linear relation.  scipy 1.18.1
/// `pearsonr(range(1, 12), 2x + 1)` → (0.9999999999999996,
/// 4.797671568894085e-69); the exact `r` is 1 and the two-sided `p` is 0.
#[test]
fn pearson_exactly_linear_has_p_zero() {
    let ctx = Context::new();
    let x: Vec<i64> = (1..=11).collect();
    let y: Vec<i64> = x.iter().map(|v| 2 * v + 1).collect();
    let r = pearson_test(&ctx, &from_i64(&x), &from_i64(&y), Alternative::TwoSided).unwrap();
    assert_eq!(r.statistic, ctx.one());
    assert_eq!(r.p_value, ctx.zero());
}
