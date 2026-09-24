//! 0.27 statistics bug hunt: hypothesis.
//!
//! Every reference value cites the oracle call that produced it, run with
//! `symplex/.venv/bin/python` (scipy 1.18.1, statsmodels 0.15.0, mpmath
//! 1.3.0 at `mp.dps` 50–150 with exact rational inputs, Python
//! `fractions.Fraction` for exact sums).  Each test says what symplex
//! returned before the fix.

// Oracle values are quoted to all the digits mpmath printed.
#![allow(clippy::excessive_precision)]

use num_bigint::BigInt;
use num_traits::One;
use symplex::linprog::{Q, q, qi};
use symplex::prelude::*;
use symplex::stats::data::from_i64;
use symplex::stats::hypothesis::*;

// ── Helpers ──────────────────────────────────────────────────────────

fn close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol,
        "got {actual}, expected {expected} (tol {tol})"
    );
}

fn rel_close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol * expected.abs(),
        "got {actual}, expected {expected} (relative tol {tol})"
    );
}

/// `10^-k` as an exact rational.
fn ten_to_minus(k: u32) -> Q {
    Q::new(BigInt::one(), BigInt::from(10).pow(k))
}

/// `2^-k` as an exact rational.
fn two_to_minus(k: usize) -> Q {
    Q::new(BigInt::one(), BigInt::one() << k)
}

// ── Tiny p-values: the log accessors ─────────────────────────────────

#[test]
fn p_value_ln_is_finite_for_chi_square_tails_at_non_dyadic_statistics() {
    // Before: `p_value_ln` / `p_value_log10` failed with PrecisionExhausted
    // whenever the χ² statistic was not dyadic and p < ~1e-108 (the
    // evaluator's bound for exp(−x) is absolute, so ln(exp(−2601/10)) could
    // not be certified), although `p_value_f64` / `p_value_decimal` were fine.
    let ctx = Context::new();
    // mpmath: s = Fraction(90301, 151) (GOF of [301, 1, 0]);
    //   log(gammainc(1, s/2, inf, regularized=True)) = -299.00993377483443709
    let r = chi_square_goodness_of_fit(&ctx, &from_i64(&[301, 1, 0]), None, 0).unwrap();
    assert_eq!(r.statistic, q(90_301, 151));
    rel_close(r.p_value_ln().unwrap(), -299.009_933_774_834_437_09, 1e-13);
    rel_close(
        r.p_value_f64().unwrap(),
        1.385_593_149_008_449_861_7e-130,
        1e-13,
    );
    // [301, 1, 0, 0] (df 3, the tail is an erfc plus an exp): mpmath:
    //   log(gammainc(3/2, Fraction(135602, 151)/2, inf, regularized=True)) = -445.8378249034929647
    let r = chi_square_goodness_of_fit(&ctx, &from_i64(&[301, 1, 0, 0]), None, 0).unwrap();
    rel_close(r.p_value_ln().unwrap(), -445.837_824_903_492_964_7, 1e-13);
    rel_close(
        r.p_value_log10().unwrap(),
        -445.837_824_903_492_964_7 / std::f64::consts::LN_10,
        1e-13,
    );
    // [301, 1, 0, 0, 0, 0] (df 5): mpmath: log(gammainc(5/2, Fraction(226204, 151)/2, inf,
    //   regularized=True)) = -739.37440020943768715;  p = 7.8302914743742790685e-322 (subnormal)
    let r = chi_square_goodness_of_fit(&ctx, &from_i64(&[301, 1, 0, 0, 0, 0]), None, 0).unwrap();
    rel_close(r.p_value_ln().unwrap(), -739.374_400_209_437_687_15, 1e-13);
    assert_eq!(r.p_value_decimal(20).unwrap(), "7.8302914743742790685e-322");
}

#[test]
fn p_value_ln_of_kruskal_and_three_by_three_independence() {
    let ctx = Context::new();
    // Before: both `p_value_ln` calls were Err(PrecisionExhausted).
    // Kruskal–Wallis on 0..1000, 1000..2000, 2000..3000: H = 8000000/3001 (Fraction), df 2,
    //   p = exp(−H/2): mpmath: -Fraction(4000000, 3001) = -1332.889036987670776407864
    let groups: Vec<Vec<Q>> = (0..3)
        .map(|g| (1000 * g..1000 * (g + 1)).map(qi).collect())
        .collect();
    let r = kruskal_wallis(&ctx, &groups).unwrap();
    assert_eq!(r.statistic_exact(), Some(q(8_000_000, 3001)));
    rel_close(r.p_value_ln().unwrap(), -1_332.889_036_987_670_776_4, 1e-13);
    // χ² independence of [[9000, 10, 10], [10, 9000, 10], [10, 10, 9000]]: X² = 24246030/451, df 4;
    //   mpmath: p = gammainc(2, X²/2, inf, regularized=True):
    //   log p = -26870.1001486797124647351, log10 p = -11669.53622275934562856518,
    //   p = 2.9092245290302460052e-11670
    let t = counts(&[&[9000, 10, 10], &[10, 9000, 10], &[10, 10, 9000]]);
    let r = chi_square_independence(&ctx, &t, false).unwrap();
    assert_eq!(r.statistic, q(24_246_030, 451));
    rel_close(
        r.p_value_ln().unwrap(),
        -26_870.100_148_679_712_464_7,
        1e-13,
    );
    rel_close(
        r.p_value_log10().unwrap(),
        -11_669.536_222_759_345_628_6,
        1e-13,
    );
    assert_eq!(
        r.p_value_decimal(20).unwrap(),
        "2.9092245290302460052e-11670"
    );
    assert_eq!(r.p_value_f64().unwrap(), 0.0);
}

#[test]
fn tiny_student_and_normal_tails_through_every_accessor() {
    // Regression guard for the 0.26 evaluation change (no bug found here):
    // the log forms stay finite where p_value_f64 underflows, and the
    // complementary one-sided p (1 − ½·tiny) is 1 without error.
    let ctx = Context::new();
    let x: Vec<Q> = (0..200).map(|i| qi(1000 + (i % 2))).collect();
    // mpmath: x = [1000, 1001] * 100 (Fractions), t² = mean²/(s²/n), ν = 199;
    //   betainc(ν/2, 1/2, 0, ν/(ν + t²), regularized=True) = 6.3648262459479613688e-659,
    //   log = -1515.5527893497978721
    let r = t_test_one_sample(&ctx, &x, &qi(0), Alternative::TwoSided).unwrap();
    assert_eq!(r.p_value_f64().unwrap(), 0.0);
    assert_eq!(r.p_value_decimal(20).unwrap(), "6.3648262459479613688e-659");
    rel_close(r.p_value_ln().unwrap(), -1_515.552_789_349_797_872_1, 1e-13);
    let r = t_test_one_sample(&ctx, &x, &qi(0), Alternative::Less).unwrap();
    assert_eq!(r.p_value_f64().unwrap(), 1.0);
    assert_eq!(r.p_value_ln().unwrap(), 0.0);
    assert_eq!(r.p_value_decimal(20).unwrap(), "1");
    // mpmath: erfc(sqrt(100000/2)) = 4.7625610525528077177e-21718, log = -50005.982264084879854
    let r = z_test_proportion(&ctx, 100_000, 100_000, &q(1, 2), Alternative::TwoSided).unwrap();
    assert_eq!(
        r.p_value_decimal(20).unwrap(),
        "4.7625610525528077177e-21718"
    );
    rel_close(r.p_value_ln().unwrap(), -50_005.982_264_084_879_854, 1e-13);
}

// ── Exact binomial family: speed and exactness ───────────────────────

#[test]
fn binomial_family_is_fast_at_thousands_of_trials() {
    // Before: the pmf was a table of reduced rationals added one by one —
    // binomial_test(0, 5000, 1/2) took over 25 s (debug), the sign test on
    // 3000 observations ~10 s per alternative, mcnemar_test(0, 2000, exact)
    // 4 s.  Now each is a single pass of integer additions (milliseconds).
    let ctx = Context::new();
    let half = q(1, 2);
    // P(X = 0) = P(X = 5000) = 2^-5000; two-sided = 2·2^-5000 (symmetric).
    let r = binomial_test(&ctx, 0, 5000, &half, Alternative::TwoSided).unwrap();
    assert_eq!(r.p_value_exact(), Some(two_to_minus(4999)));
    let r = binomial_test(&ctx, 0, 5000, &half, Alternative::Less).unwrap();
    assert_eq!(r.p_value_exact(), Some(two_to_minus(5000)));
    let r = binomial_test(&ctx, 5000, 5000, &half, Alternative::Less).unwrap();
    assert_eq!(r.p_value_exact(), Some(qi(1)));
    let r = binomial_test(&ctx, 5000, 5000, &half, Alternative::Greater).unwrap();
    assert_eq!(r.p_value_exact(), Some(two_to_minus(5000)));
    // Sign test: 3000 observations all above 0 → 2·2^-3000.
    let x: Vec<Q> = (1..=3000).map(qi).collect();
    let r = sign_test(&ctx, &x, &qi(0), Alternative::TwoSided).unwrap();
    assert_eq!(r.p_value_exact(), Some(two_to_minus(2999)));
    // McNemar exact, b = 0, c = 2000: 2·P(Binomial(2000, ½) ≤ 0) = 2^-1999.
    let r = mcnemar_test(&ctx, 0, 2000, true, true).unwrap();
    assert_eq!(r.p_value_exact(), Some(two_to_minus(1999)));
}

#[test]
fn binomial_test_general_p0_matches_fraction_sums() {
    let ctx = Context::new();
    // Fraction: sum(comb(20000, i)·3^i·17^(20000−i), i ≤ 2600) / 20^20000
    //   → float 3.726658942859049e-16; mpmath log = -35.525849381426108052
    // (the 86 432-bit rational: p_value_f64 of it took 12 s before 0.27).
    // (and p_value_decimal of it took 12 s: mpmath prints 3.7266589428590489823e-16).
    let r = binomial_test(&ctx, 2600, 20_000, &q(3, 20), Alternative::Less).unwrap();
    assert_eq!(r.p_value_f64().unwrap(), 3.726_658_942_859_049e-16);
    assert_eq!(r.p_value_decimal(20).unwrap(), "3.7266589428590489823e-16");
    rel_close(r.p_value_ln().unwrap(), -35.525_849_381_426_108_05, 1e-14);
}

#[test]
fn binomial_two_sided_compares_the_pmf_exactly() {
    // Documented deliberate difference from scipy (not a regression):
    // P(X = 278) exceeds P(X = 198) by a relative 2.97e-8 for n = 950,
    // p₀ = 1/4, so the exact rule excludes x = 278 where scipy's 1e-7 slack
    // includes it.
    // Fraction: w = [comb(950, i)·3^(950−i)]; sum(v for v in w if v <= w[198]) / 4^950
    //   = 0.002718010549105766079 (mpmath, 20 digits);
    // scipy: binomtest(198, 950, 0.25).pvalue = 0.0030516140249515723
    let ctx = Context::new();
    let r = binomial_test(&ctx, 198, 950, &q(1, 4), Alternative::TwoSided).unwrap();
    rel_close(
        r.p_value_f64().unwrap(),
        0.002_718_010_549_105_766_079,
        1e-15,
    );
}

// ── McNemar ──────────────────────────────────────────────────────────

#[test]
fn mcnemar_continuity_correction_stops_at_zero() {
    // Before: b = c = 5 gave χ² = (|0| − 1)²/10 = 1/10, p = 0.7518
    // (statsmodels' value): more significant than the uncorrected test
    // (χ² = 0, p = 1) and the exact test (statsmodels:
    // mcnemar([[1, 5], [5, 1]], exact=True).pvalue = 1.0).  The correction
    // now moves |b − c| towards 0 and stops there, as scipy's Yates
    // correction does (scipy: chi2_contingency([[5, 5], [5, 5]]) →
    // statistic 0.0, pvalue 1.0) and R's mcnemar.test.
    let ctx = Context::new();
    let r = mcnemar_test(&ctx, 5, 5, false, true).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(0)));
    assert_eq!(r.p_value, ctx.one());
    // |b − c| = 1: statsmodels: mcnemar([[1, 7], [8, 1]], exact=False) → statistic 0.0, pvalue 1.0
    let r = mcnemar_test(&ctx, 7, 8, false, true).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(0)));
    // Unchanged away from b = c: statsmodels: mcnemar([[1, 1500], [1600, 1]], exact=False)
    //   → statistic 3.1616129032258065 (= 99²/3100), pvalue 0.0753886656726421
    let r = mcnemar_test(&ctx, 1500, 1600, false, true).unwrap();
    assert_eq!(r.statistic_exact(), Some(q(9801, 3100)));
    close(r.p_value_f64().unwrap(), 0.075_388_665_672_642_1, 1e-12);
}

// ── Effect sizes ─────────────────────────────────────────────────────

#[test]
fn cohens_h_does_not_cancel() {
    // Before: h = 2 asin √p₁ − 2 asin √p₂ evaluated to exactly 0 for
    // (½, ½ + 1e-100) and failed with PrecisionExhausted for (1 − 1e-30, 1).
    let ctx = Context::new();
    let half = q(1, 2);
    // mpmath (dps 150): 2*asin(sqrt(1/2)) - 2*asin(sqrt(1/2 + 1/10**100)) = -2.0e-100
    //   (−2ε(1 + O(ε)), the derivative of 2 asin √p at ½ being 2)
    let h = cohens_h(&ctx, &half, &(&half + ten_to_minus(100))).unwrap();
    rel_close(h.eval_f64().unwrap(), -2e-100, 1e-14);
    // mpmath: 2*asin(sqrt(1 - 1/10**30)) - pi = -2.0e-15 (to 25 digits)
    let h = cohens_h(&ctx, &(Q::one() - ten_to_minus(30)), &qi(1)).unwrap();
    rel_close(h.eval_f64().unwrap(), -2e-15, 1e-14);
    // mpmath: 2*asin(sqrt(1/10**30)) - 2*asin(sqrt(1 - 1/10**30)) = -3.141592653589789238462643
    let h = cohens_h(&ctx, &ten_to_minus(30), &(Q::one() - ten_to_minus(30))).unwrap();
    rel_close(h.eval_f64().unwrap(), -3.141_592_653_589_789_238_5, 1e-15);
    assert_eq!(cohens_h(&ctx, &q(1, 3), &q(1, 3)).unwrap(), ctx.zero());
    assert_eq!(cohens_h(&ctx, &qi(1), &qi(1)).unwrap(), ctx.zero());
}

// ── Power and sample size ────────────────────────────────────────────

#[test]
fn sample_size_for_proportion_errors_instead_of_saturating() {
    // Before: margin 1e-200 (n = ∞) and 1e-10 (n = 9.6e19) both returned
    // 18446744073709551615 (the saturated `as usize` cast).
    for margin in [1e-200, 1e-10] {
        assert!(matches!(
            sample_size_for_proportion(margin, 0.95, 0.5),
            Err(SymplexError::ComputationFailed { .. })
        ));
    }
    // statsmodels: samplesize_confint_proportion(0.5, 1e-9) = 9.603647051735316e+17
    let n = sample_size_for_proportion(1e-9, 0.95, 0.5).unwrap() as f64;
    rel_close(n, 9.603_647_051_735_316e17, 1e-14);
}

#[test]
fn power_t_test_is_accurate_and_total_at_large_n() {
    // Before: the χ² density's log form cancelled terms of size ν ln ν, so
    // at d = 0 the power drifted from α (by −1.65e-6 at n = 1e9), and
    // `2 * n_per_group - 2` overflowed (a panic) for n > usize::MAX / 2.
    // At d = 0 the power is α by definition.
    for n in [1_000usize, 10_000_000, 1_000_000_000] {
        close(power_t_test_two_sample(0.0, n, 0.05).unwrap(), 0.05, 1e-11);
    }
    // mpmath (dps 40): the two-sided noncentral-t power, quad over the χ²
    //   mixing density (target/scratch nct_power2.py) at d = 1e-4, n = 1569775619 → 0.8000008784137525983
    close(
        power_t_test_two_sample(1e-4, 1_569_775_619, 0.05).unwrap(),
        0.800_000_878_413_752_6,
        1e-9,
    );
    assert_eq!(power_t_test_two_sample(0.5, usize::MAX, 0.05).unwrap(), 1.0);
}

#[test]
fn power_t_test_where_scipy_nct_returns_nan() {
    // scipy: nct.sf(t_c, 190, δ) + nct.cdf(−t_c, 190, δ) is NaN for d = 1.0874119952634231,
    // n = 96, α = 0.01 (so is statsmodels' TTestIndPower().power); mpmath (dps 40, quad over
    // the χ² mixing density) = 0.99999949964090282069.  symplex was already right.
    close(
        power_t_test_two_sample(1.087_411_995_263_423_1, 96, 0.01).unwrap(),
        0.999_999_499_640_902_820_69,
        1e-12,
    );
}

// ── Counts that overflow usize ───────────────────────────────────────

#[test]
fn count_arguments_that_overflow_usize() {
    // Before: `b + c`, `a + b` and `k₁ + k₂` were plain usize additions —
    // panics ("attempt to add with overflow") in a debug build.
    let ctx = Context::new();
    assert!(matches!(
        mcnemar_test(&ctx, usize::MAX, 1, false, true),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        relative_risk([[usize::MAX, 1], [1, 1]], 0.95),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // n₁ = k₁ = usize::MAX = N, k₂ = 0, n₂ = 1: p̂ = N/(N + 1), p̂(1 − p̂)(1/N + 1) = 1/(N + 1),
    // so z = √(N + 1) (= 2^32 on a 64-bit target).
    let r =
        z_test_two_proportions(&ctx, usize::MAX, usize::MAX, 0, 1, Alternative::TwoSided).unwrap();
    let n_plus_1 = (usize::MAX as f64) + 1.0;
    assert_eq!(r.statistic_f64().unwrap(), n_plus_1.sqrt());
    assert!(phi_coefficient(&ctx, [[usize::MAX, 1], [1, 1]]).is_ok());
}

// ── Multiplicity ─────────────────────────────────────────────────────

#[test]
fn bonferroni_and_holm_compare_like_statsmodels() {
    // Before: symplex rejected where fl(m·p) ≤ α; statsmodels rejects where
    // p ≤ α/m, and at p = 0.05/11 the two disagree in the last bit.
    // statsmodels: multipletests([0.05/11] + [0.9]*10, alpha=0.05, method=m)
    //   → reject[0] True, pvals_corrected[0] 0.05000000000000001, for m in ('bonferroni', 'holm')
    let mut p = vec![0.05 / 11.0];
    p.extend([0.9; 10]);
    for adj in [bonferroni(&p, 0.05).unwrap(), holm(&p, 0.05).unwrap()] {
        assert!(adj.reject[0]);
        assert!(!adj.reject[1]);
        assert_eq!(adj.p_adjusted[0], 0.050_000_000_000_000_01);
    }
}

// ── Kendall, n = 2 ───────────────────────────────────────────────────

#[test]
fn kendall_asymptotic_with_two_pairs() {
    // Documented (not a change): scipy 1.18 raises ZeroDivisionError here
    // (kendalltau([47, 31], [1047, 1031], method='asymptotic')).  The
    // variance is n(n−1)(2n+5)/18 = 1, z = 1: mpmath erfc(1/sqrt(2)) =
    // 0.31731050786291410283 (two-sided), half of it greater.
    let ctx = Context::new();
    let (x, y) = (from_i64(&[47, 31]), from_i64(&[1047, 1031]));
    let r = kendall_test(&ctx, &x, &y, Alternative::TwoSided, false).unwrap();
    assert_eq!(r.statistic, ctx.one());
    close(
        r.p_value_f64().unwrap(),
        0.317_310_507_862_914_102_83,
        1e-15,
    );
    let r = kendall_test(&ctx, &x, &y, Alternative::Greater, false).unwrap();
    close(r.p_value_f64().unwrap(), 0.158_655_253_931_457_051_4, 1e-15);
}
