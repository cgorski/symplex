//! Independent verification of `symplex::stats` (0.17 audit, track a):
//! `hypothesis`, `agreement`, `aggregation`, `data`.
//!
//! Every reference value below was produced by the cited oracle call in
//! `symplex/.venv/bin/python` (scipy 1.18.1, statsmodels 0.15.0,
//! krippendorff 0.8, pingouin 0.6.1, numpy 2.5.3, Python `statistics`);
//! exact rationals were derived with `fractions.Fraction` arithmetic on
//! the integer data (never `Fraction(float).limit_denominator()`) and
//! `float(frac)` was checked against the oracle float to ≤ 1e-12 before
//! being written here.  None of the datasets appears in `tests/v13` or
//! `tests/v14`.
//!
//! Datasets (all new):
//! * `T1`  = [12, 15, 11, 19, 14, 17, 13, 16, 18, 10, 20]      (one-sample t, CI)
//! * `X8`  = [7, 11, 9, 14, 12, 8, 10, 13], `Y6` = [15, 9, 18, 12, 20, 11]
//! * `BEF` = [88, 92, 79, 85, 90, 84, 77, 95], `AFT` = [85, 90, 79, 80, 91, 78, 76, 90]
//! * `MWX` = [14, 3, 27, 9, 18, 31], `MWY` = [7, 22, 12, 5]    (no ties)
//! * `MWXT` = [5, 5, 8, 3, 9, 5], `MWYT` = [4, 8, 2, 8, 1]     (ties)
//! * `WD`  = [6, -2, 9, 4, -7, 3, 11, -1, 8, 5, 10]           (Wilcoxon, no ties)
//! * `WPX` = [10, 12, 9, 15, 11, 14, 13, 8, 16, 12], `WPY` = [8, 12, 11, 12, 10, 10, 13, 9, 12, 9]
//! * `KW`  = [[3, 5, 5, 8], [6, 9, 9, 12, 7], [2, 4, 3], [11, 9, 14, 6]]
//! * `FR`  = [[7, 9, 6, 9], [5, 8, 5, 7], [8, 8, 6, 10], [6, 9, 7, 7], [7, 10, 5, 8]]
//! * `SPX` = [3, 1, 4, 1, 5, 9, 2, 6], `SPY` = [2, 7, 1, 8, 2, 8, 1, 8]
//! * `T33` = [[12, 5, 9], [3, 14, 6], [7, 8, 15]], `T22` = [[17, 6], [4, 13]]
//! * `G4`  = [[23, 25, 21, 27], [30, 28, 33], [19, 22, 20, 24, 21], [26, 29]]
//! * `P7`  = [0.001, 0.02, 0.02, 0.5, 0.0, 1.0, 0.049]
//! * `RA`, `RB`: two raters × 24 items, categories 0..3 (rater B never uses 3)
//! * `D5`: 5 raters × 10 items × 3 categories (complete)
//! * `EK`, `FK`: 3 raters × 9 items with missing cells (values 0.5, −1, …)
//! * `G53`: 5 targets × 3 judges (ICC), `G2C`: 4 × 3 with a constant judge
//! * `R8` = [−3/2, 7/4, 2, −1/2, 5, 5, 3/4, 0], `S8` = [2, −1, 0, 3, 1, 1, 4, −2]

use num_bigint::BigInt;
use symplex::Interval;
use symplex::linprog::{Q, q, qi};
use symplex::prelude::*;
use symplex::stats::aggregation::*;
use symplex::stats::agreement::*;
use symplex::stats::anova::anova_one_way;
use symplex::stats::data::*;
use symplex::stats::estimation::{IntervalMethod, confidence_interval_mean, proportion_interval};
use symplex::stats::hypothesis::*;
use symplex::stats::{Distribution, Rng};

// ── Helpers ──────────────────────────────────────────────────────────

fn close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() < tol,
        "got {actual}, expected {expected} (tol {tol})"
    );
}

fn p_of(r: &TestResult) -> f64 {
    r.p_value_f64().unwrap()
}

fn s_of(r: &TestResult) -> f64 {
    r.statistic_f64().unwrap()
}

fn is_invalid<T: std::fmt::Debug>(r: Result<T, SymplexError>) {
    match r {
        Err(SymplexError::InvalidArgument { .. }) => {}
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

fn raters(cols: &[&[Option<i64>]]) -> RatingTable {
    RatingTable::from_raters_i64(cols).unwrap()
}

fn t1() -> Vec<Q> {
    from_i64(&[12, 15, 11, 19, 14, 17, 13, 16, 18, 10, 20])
}

fn x8y6() -> (Vec<Q>, Vec<Q>) {
    (
        from_i64(&[7, 11, 9, 14, 12, 8, 10, 13]),
        from_i64(&[15, 9, 18, 12, 20, 11]),
    )
}

fn mw() -> (Vec<Q>, Vec<Q>) {
    (from_i64(&[14, 3, 27, 9, 18, 31]), from_i64(&[7, 22, 12, 5]))
}

fn mw_ties() -> (Vec<Q>, Vec<Q>) {
    (from_i64(&[5, 5, 8, 3, 9, 5]), from_i64(&[4, 8, 2, 8, 1]))
}

fn wd() -> Vec<Q> {
    from_i64(&[6, -2, 9, 4, -7, 3, 11, -1, 8, 5, 10])
}

fn g4() -> Vec<Vec<Q>> {
    vec![
        from_i64(&[23, 25, 21, 27]),
        from_i64(&[30, 28, 33]),
        from_i64(&[19, 22, 20, 24, 21]),
        from_i64(&[26, 29]),
    ]
}

fn t33() -> Vec<Vec<Q>> {
    counts(&[&[12, 5, 9], &[3, 14, 6], &[7, 8, 15]])
}

const P7: [f64; 7] = [0.001, 0.02, 0.02, 0.5, 0.0, 1.0, 0.049];

fn asym(continuity: bool) -> RankMethod {
    RankMethod::Asymptotic { continuity }
}

// ═══════════════════════════════════════════════════════════════════════
// Regression tests for the bugs fixed in this audit
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn regression_phi_coefficient_large_counts_does_not_overflow() {
    // Margins of 2e5 → (a+b)(c+d)(a+c)(b+d) = 1.6e21 > u64::MAX: used to
    // panic with "attempt to multiply with overflow" (debug) / wrap (release).
    // numpy: corrcoef of the indicator vectors of [[100000, 100000], [100000, 100000]] = 0.0
    let ctx = Context::new();
    let phi = phi_coefficient(&ctx, [[100_000, 100_000], [100_000, 100_000]]).unwrap();
    close(phi.eval_f64().unwrap(), 0.0, 1e-15);
    // A non-trivial large table: [[300000, 100000], [100000, 300000]] → φ = 1/2 exactly.
    let phi = phi_coefficient(&ctx, [[300_000, 100_000], [100_000, 300_000]]).unwrap();
    assert_eq!(phi, ctx.rational(1, 2));
}

#[test]
fn regression_odds_ratio_and_fisher_large_counts_do_not_overflow() {
    // a·d = 2.5e19 > u64::MAX used to panic in `odds_ratio` and `fisher_exact`'s statistic.
    let r = odds_ratio([[5_000_000_000, 1], [1, 5_000_000_000]], 0.95).unwrap();
    assert_eq!(
        r.estimate,
        Q::from_integer(BigInt::from(5_000_000_000u64) * BigInt::from(5_000_000_000u64))
    );
    // statsmodels: Table2x2([[100000, 100000], [100000, 100000]]).oddsratio = 1.0,
    //              .oddsratio_confint(0.05) = (0.9876806120560633, 1.0124730482643487)
    let r = odds_ratio([[100_000, 100_000], [100_000, 100_000]], 0.95).unwrap();
    assert_eq!(r.estimate, qi(1));
    close(r.ci.lower, 0.987_680_612_056_063_3, 1e-12);
    close(r.ci.upper, 1.012_473_048_264_348_7, 1e-12);
    // Fisher's statistic with a·d = 2.5e19 overflowing usize (the small first
    // row keeps the hypergeometric support at 7 terms).
    let ctx = Context::new();
    let r = fisher_exact(
        &ctx,
        [[5, 1], [1, 5_000_000_000_000_000_000]],
        Alternative::TwoSided,
    )
    .unwrap();
    assert_eq!(
        r.statistic,
        ctx.from_ratio(Q::from_integer(BigInt::from(
            25_000_000_000_000_000_000u128
        )))
    );
}

#[test]
fn regression_yates_correction_is_clamped_like_scipy() {
    // scipy ≥ 1.7 (gh-13875) moves each observed count at most |O − E|
    // towards its expectation; the crate used to compute (|O−E| − ½)² even
    // when |O − E| < ½, giving 13/144 here instead of 0.
    // scipy: chi2_contingency([[3, 3], [3, 4]], correction=True) → statistic 0.0, pvalue 1.0
    let ctx = Context::new();
    let r = chi_square_independence(&ctx, &counts(&[&[3, 3], &[3, 4]]), true).unwrap();
    assert_eq!(r.statistic, qi(0));
    close(r.p_value_f64().unwrap(), 1.0, 1e-15);
    // scipy: chi2_contingency([[5, 3], [4, 3]], correction=True) → statistic 0.0, pvalue 1.0
    //        (|O − E| = 0.2 in every cell); correction=False → 0.04464285714285722 = 5/112
    let r = chi_square_independence(&ctx, &counts(&[&[5, 3], &[4, 3]]), true).unwrap();
    assert_eq!(r.statistic, qi(0));
    let r = chi_square_independence(&ctx, &counts(&[&[5, 3], &[4, 3]]), false).unwrap();
    assert_eq!(r.statistic, q(5, 112));
    // Unaffected when |O − E| ≥ ½:
    // scipy: chi2_contingency([[17, 6], [4, 13]], correction=True)
    //        → statistic 8.032613503067132 (= 417720/52003), pvalue 0.004594250126503651
    let r = chi_square_independence(&ctx, &counts(&[&[17, 6], &[4, 13]]), true).unwrap();
    assert_eq!(r.statistic, q(417_720, 52_003));
    close(r.p_value_f64().unwrap(), 0.004_594_250_126_503_651, 1e-12);
}

// ═══════════════════════════════════════════════════════════════════════
// hypothesis: exact discrete tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn binomial_test_rational_null_all_alternatives() {
    let ctx = Context::new();
    // Fraction: pmf of Binomial(12, 1/3); scipy: binomtest(3, 12, 1/3).pvalue = 0.761553963657302,
    //   alternative='less' 0.39307467809220625, 'greater' 0.8188773542124148
    let r = binomial_test(&ctx, 3, 12, &q(1, 3), Alternative::TwoSided).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(44_969, 59_049)));
    assert_eq!(r.statistic_exact(), Some(q(1, 4)));
    let r = binomial_test(&ctx, 3, 12, &q(1, 3), Alternative::Less).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(69_632, 177_147)));
    let r = binomial_test(&ctx, 3, 12, &q(1, 3), Alternative::Greater).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(435_185, 531_441)));
    // scipy: binomtest(0, 6, 0.5).pvalue = 0.03125 = 1/32
    let r = binomial_test(&ctx, 0, 6, &q(1, 2), Alternative::TwoSided).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(1, 32)));
}

#[test]
fn binomial_test_degenerate_null_proportions() {
    let ctx = Context::new();
    // scipy: binomtest(0, 5, 0.0).pvalue = 1.0; binomtest(2, 5, 0.0).pvalue = 0.0
    let r = binomial_test(&ctx, 0, 5, &qi(0), Alternative::TwoSided).unwrap();
    assert_eq!(r.p_value_exact(), Some(qi(1)));
    let r = binomial_test(&ctx, 2, 5, &qi(0), Alternative::TwoSided).unwrap();
    assert_eq!(r.p_value_exact(), Some(qi(0)));
    // scipy: binomtest(5, 5, 1.0).pvalue = 1.0; binomtest(3, 5, 1.0).pvalue = 0.0
    let r = binomial_test(&ctx, 5, 5, &qi(1), Alternative::TwoSided).unwrap();
    assert_eq!(r.p_value_exact(), Some(qi(1)));
    let r = binomial_test(&ctx, 3, 5, &qi(1), Alternative::TwoSided).unwrap();
    assert_eq!(r.p_value_exact(), Some(qi(0)));
    is_invalid(binomial_test(&ctx, 0, 0, &q(1, 2), Alternative::TwoSided));
    is_invalid(binomial_test(&ctx, 6, 5, &q(1, 2), Alternative::TwoSided));
    is_invalid(binomial_test(&ctx, 1, 5, &q(3, 2), Alternative::TwoSided));
    is_invalid(binomial_test(&ctx, 1, 5, &q(-1, 2), Alternative::TwoSided));
}

#[test]
fn fisher_exact_new_table_all_alternatives_and_zero_cells() {
    let ctx = Context::new();
    // Fraction: hypergeometric pmf of [[3, 7], [9, 2]];
    // scipy: fisher_exact([[3, 7], [9, 2]]) → statistic 0.09523809523809523, pvalue 0.02997312285237981;
    //   alternative='less' 0.024172422005239336, 'greater' 0.9982819038546593
    let r = fisher_exact(&ctx, [[3, 7], [9, 2]], Alternative::TwoSided).unwrap();
    assert_eq!(r.statistic_exact(), Some(q(2, 21)));
    assert_eq!(r.p_value_exact(), Some(q(881, 29_393)));
    let r = fisher_exact(&ctx, [[3, 7], [9, 2]], Alternative::Less).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(203, 8_398)));
    let r = fisher_exact(&ctx, [[3, 7], [9, 2]], Alternative::Greater).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(58_685, 58_786)));
    // Zero cell: scipy: fisher_exact([[0, 5], [6, 3]]) → statistic 0.0, pvalue 0.030969030969030965
    //   (= 31/1001); alternative='less' 0.02797202797202797 (= 4/143)
    let r = fisher_exact(&ctx, [[0, 5], [6, 3]], Alternative::TwoSided).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(0)));
    assert_eq!(r.p_value_exact(), Some(q(31, 1_001)));
    let r = fisher_exact(&ctx, [[0, 5], [6, 3]], Alternative::Less).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(4, 143)));
    // Diagonal table: scipy: fisher_exact([[4, 0], [0, 4]]) → statistic inf, pvalue 0.028571428571428567 (= 1/35)
    let r = fisher_exact(&ctx, [[4, 0], [0, 4]], Alternative::TwoSided).unwrap();
    assert_eq!(r.statistic, ctx.infinity());
    assert_eq!(r.p_value_exact(), Some(q(1, 35)));
    // Empty row / column: scipy returns (nan, 1.0); the crate errors as documented.
    is_invalid(fisher_exact(&ctx, [[0, 0], [3, 4]], Alternative::TwoSided));
    is_invalid(fisher_exact(&ctx, [[3, 0], [4, 0]], Alternative::TwoSided));
}

#[test]
fn mcnemar_test_new_counts_and_degenerate_discordance() {
    let ctx = Context::new();
    // statsmodels: mcnemar([[30, 12], [4, 40]], exact=True) → statistic 4.0, pvalue 0.076812744140625
    //   (= 2 Σ_{i≤4} C(16, i)/2^16 = 2517/32768)
    let r = mcnemar_test(&ctx, 12, 4, true, true).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(4)));
    assert_eq!(r.p_value_exact(), Some(q(2_517, 32_768)));
    // statsmodels: mcnemar(..., exact=False, correction=True) → statistic 3.0625 (= 49/16), pvalue 0.08011831372763438
    let r = mcnemar_test(&ctx, 12, 4, false, true).unwrap();
    assert_eq!(r.statistic_exact(), Some(q(49, 16)));
    close(p_of(&r), 0.080_118_313_727_634_38, 1e-12);
    // statsmodels: mcnemar(..., exact=False, correction=False) → statistic 4.0, pvalue 0.045500263896358445
    let r = mcnemar_test(&ctx, 12, 4, false, false).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(4)));
    close(p_of(&r), 0.045_500_263_896_358_445, 1e-12);
    // b = c: statsmodels: mcnemar([[1, 5], [5, 1]], exact=True).pvalue = 1.0;
    //   exact=False, correction=True → statistic 0.1, pvalue 0.7518296340458492
    let r = mcnemar_test(&ctx, 5, 5, true, true).unwrap();
    assert_eq!(r.p_value_exact(), Some(qi(1)));
    let r = mcnemar_test(&ctx, 5, 5, false, true).unwrap();
    assert_eq!(r.statistic_exact(), Some(q(1, 10)));
    close(p_of(&r), 0.751_829_634_045_849_2, 1e-12);
    // b = 0: statsmodels: mcnemar([[1, 0], [7, 1]], exact=True) → statistic 0.0, pvalue 0.015625 (= 1/64)
    let r = mcnemar_test(&ctx, 0, 7, true, true).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(0)));
    assert_eq!(r.p_value_exact(), Some(q(1, 64)));
    is_invalid(mcnemar_test(&ctx, 0, 0, true, true));
}

#[test]
fn sign_test_drops_ties_and_matches_statsmodels() {
    let ctx = Context::new();
    let x = from_i64(&[2, 9, 4, 4, 7, 1, 8, 6, 3, 5, 4]);
    // statsmodels: sign_test(x, mu0=4) = (1.0, 0.7265625)   (5 above, 3 below, 3 ties; 0.7265625 = 93/128)
    let r = sign_test(&ctx, &x, &qi(4), Alternative::TwoSided).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(1)));
    assert_eq!(r.p_value_exact(), Some(q(93, 128)));
    // Fraction: P(Bin(8, ½) ≥ 5) = 93/256, P(Bin(8, ½) ≤ 5) = 219/256
    let r = sign_test(&ctx, &x, &qi(4), Alternative::Greater).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(93, 256)));
    let r = sign_test(&ctx, &x, &qi(4), Alternative::Less).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(219, 256)));
    // Consistency with the binomial test of n₊ out of n₊ + n₋.
    let b = binomial_test(&ctx, 5, 8, &q(1, 2), Alternative::TwoSided).unwrap();
    assert_eq!(b.p_value_exact(), Some(q(93, 128)));
    is_invalid(sign_test(
        &ctx,
        &from_i64(&[4, 4]),
        &qi(4),
        Alternative::TwoSided,
    ));
    is_invalid(sign_test(&ctx, &[], &qi(4), Alternative::TwoSided));
}

// ═══════════════════════════════════════════════════════════════════════
// hypothesis: categorical
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn chi_square_independence_3x3_exact_statistic_and_expected() {
    let ctx = Context::new();
    // scipy: chi2_contingency([[12, 5, 9], [3, 14, 6], [7, 8, 15]], correction=False)
    //   → statistic 14.223330105214162 (= 75784463/5328180), dof 4, pvalue 0.0066153544706033345,
    //   expected[0][0] = 572/79, expected[1][2] = 690/79
    let r = chi_square_independence(&ctx, &t33(), true).unwrap();
    assert_eq!(r.statistic, q(75_784_463, 5_328_180));
    assert_eq!(r.df, 4);
    assert_eq!(r.expected[0][0], q(572, 79));
    assert_eq!(r.expected[1][2], q(690, 79));
    close(r.p_value_f64().unwrap(), 0.006_615_354_470_603_334_5, 1e-12);
    // `correction` is ignored for df > 1.
    let r2 = chi_square_independence(&ctx, &t33(), false).unwrap();
    assert_eq!(r2.statistic, r.statistic);
    // 1 × k and k × 1 tables are rejected (scipy returns dof 0, statistic 0).
    is_invalid(chi_square_independence(&ctx, &counts(&[&[3, 4, 5]]), false));
    is_invalid(chi_square_independence(
        &ctx,
        &counts(&[&[3], &[4], &[5]]),
        false,
    ));
    // Independence exactly: statistic 0, p = 1.
    let r = chi_square_independence(&ctx, &counts(&[&[2, 4], &[3, 6]]), false).unwrap();
    assert_eq!(r.statistic, qi(0));
    close(r.p_value_f64().unwrap(), 1.0, 1e-15);
}

#[test]
fn chi_square_goodness_of_fit_ddof_rational_expected_and_zero_counts() {
    let ctx = Context::new();
    let obs = from_i64(&[18, 22, 30, 15, 25]);
    // scipy: chisquare([18, 22, 30, 15, 25], ddof=1) → statistic 6.2727272727272725 (= 69/11), pvalue 0.09906964623852726
    let r = chi_square_goodness_of_fit(&ctx, &obs, None, 1).unwrap();
    assert_eq!(r.statistic, q(69, 11));
    assert_eq!(r.df, 3);
    close(r.p_value_f64().unwrap(), 0.099_069_646_238_527_26, 1e-12);
    // scipy: chisquare(obs, f_exp=[20, 20, 30, 20, 20]) → statistic 2.9 (= 29/10), pvalue 0.5746972058298043
    let r =
        chi_square_goodness_of_fit(&ctx, &obs, Some(&from_i64(&[20, 20, 30, 20, 20])), 0).unwrap();
    assert_eq!(r.statistic, q(29, 10));
    close(r.p_value_f64().unwrap(), 0.574_697_205_829_804_3, 1e-12);
    // Rational expected counts: scipy: chisquare([7, 5], f_exp=[4.5, 7.5])
    //   → statistic 2.2222222222222223 (= 20/9), pvalue 0.13603712811414329
    let r = chi_square_goodness_of_fit(&ctx, &from_i64(&[7, 5]), Some(&[q(9, 2), q(15, 2)]), 0)
        .unwrap();
    assert_eq!(r.statistic, q(20, 9));
    close(r.p_value_f64().unwrap(), 0.136_037_128_114_143_29, 1e-12);
    // A zero observed count is fine: scipy: chisquare([0, 6, 6]) → statistic 6.0, pvalue 0.04978706836786395
    let r = chi_square_goodness_of_fit(&ctx, &from_i64(&[0, 6, 6]), None, 0).unwrap();
    assert_eq!(r.statistic, qi(6));
    close(r.p_value_f64().unwrap(), 0.049_787_068_367_863_95, 1e-12);
    // Validation: ddof leaves no df; totals differ; zero expected; all-zero observed.
    is_invalid(chi_square_goodness_of_fit(&ctx, &obs, None, 4));
    is_invalid(chi_square_goodness_of_fit(
        &ctx,
        &obs,
        Some(&from_i64(&[20, 20, 30, 20, 21])),
        0,
    ));
    is_invalid(chi_square_goodness_of_fit(
        &ctx,
        &from_i64(&[7, 5]),
        Some(&[qi(12), qi(0)]),
        0,
    ));
    is_invalid(chi_square_goodness_of_fit(
        &ctx,
        &from_i64(&[0, 0]),
        None,
        0,
    ));
}

#[test]
fn g_test_3x3_and_zero_cell_agree_with_scipy_and_chi2_direction() {
    let ctx = Context::new();
    // scipy: chi2_contingency(T33, correction=False, lambda_='log-likelihood')
    //   → statistic 13.61814875846871, pvalue 0.008618985093683458
    let g = g_test(&ctx, &t33()).unwrap();
    close(s_of(&g), 13.618_148_758_468_71, 1e-12);
    close(p_of(&g), 0.008_618_985_093_683_458, 1e-12);
    assert_eq!(g.df, Some(ctx.int(4)));
    // Zero cell contributes 0: scipy: chi2_contingency([[0, 5], [6, 3]], correction=False,
    //   lambda_='log-likelihood') → statistic 7.664171902306574, pvalue 0.005632810705711467;
    //   Pearson (correction=False) → statistic 5.833333333333333 (= 35/6), pvalue 0.015725299754505352
    let t = counts(&[&[0, 5], &[6, 3]]);
    let g = g_test(&ctx, &t).unwrap();
    close(s_of(&g), 7.664_171_902_306_574, 1e-12);
    close(p_of(&g), 0.005_632_810_705_711_467, 1e-12);
    let c = chi_square_independence(&ctx, &t, false).unwrap();
    assert_eq!(c.statistic, q(35, 6));
    close(c.p_value_f64().unwrap(), 0.015_725_299_754_505_352, 1e-12);
    // Both vanish together under exact independence.
    let g = g_test(&ctx, &counts(&[&[2, 4], &[3, 6]])).unwrap();
    assert_eq!(g.statistic, ctx.zero());
    assert_eq!(g.p_value, ctx.one());
    is_invalid(g_test(&ctx, &counts(&[&[1, 2, 3]])));
}

#[test]
fn cramers_v_and_phi_new_tables_including_negative_association() {
    let ctx = Context::new();
    // scipy: association(T33, method='cramer', correction=False) = 0.300035125635782
    let v = cramers_v(&ctx, &t33()).unwrap();
    close(v.eval_f64().unwrap(), 0.300_035_125_635_782, 1e-12);
    // scipy: association([[10, 4, 6, 8], [3, 9, 7, 2]], method='cramer', correction=False) = 0.41756313817071355
    let v = cramers_v(&ctx, &counts(&[&[10, 4, 6, 8], &[3, 9, 7, 2]])).unwrap();
    close(v.eval_f64().unwrap(), 0.417_563_138_170_713_55, 1e-12);
    // numpy: corrcoef of the indicators of [[17, 6], [4, 13]] = 0.4987597511956934 (= 197/√156009)
    let phi = phi_coefficient(&ctx, [[17, 6], [4, 13]]).unwrap();
    close(phi.eval_f64().unwrap(), 0.498_759_751_195_693_4, 1e-12);
    close(
        (&phi * &phi).eval_f64().unwrap(),
        38_809.0 / 156_009.0,
        1e-15,
    );
    // numpy: corrcoef of the indicators of [[2, 9], [8, 3]] = -0.5477225575051661 (= −66/√14520)
    let phi = phi_coefficient(&ctx, [[2, 9], [8, 3]]).unwrap();
    close(phi.eval_f64().unwrap(), -0.547_722_557_505_166_1, 1e-12);
    // φ² = χ²/N without correction (= 3/10 here).
    let chi = chi_square_independence(&ctx, &counts(&[&[2, 9], &[8, 3]]), false).unwrap();
    assert_eq!(chi.statistic / qi(22), q(3, 10));
    close((&phi * &phi).eval_f64().unwrap(), 0.3, 1e-15);
    is_invalid(phi_coefficient(&ctx, [[0, 0], [1, 2]]));
    is_invalid(phi_coefficient(&ctx, [[1, 0], [2, 0]]));
}

#[test]
fn odds_ratio_and_relative_risk_at_90_percent() {
    // statsmodels: Table2x2([[17, 6], [4, 13]]).oddsratio = 9.208333333333334 (= 221/24),
    //              .oddsratio_confint(0.10) = (2.711710126882576, 31.269346209676755)
    let r = odds_ratio([[17, 6], [4, 13]], 0.90).unwrap();
    assert_eq!(r.estimate, q(221, 24));
    close(r.ci.lower, 2.711_710_126_882_576, 1e-9);
    close(r.ci.upper, 31.269_346_209_676_755, 1e-9);
    assert_eq!(r.confidence, 0.90);
    // scipy: relative_risk(17, 23, 4, 17).relative_risk = 3.141304347826087 (= 289/92),
    //        .confidence_interval(0.90) = (1.4875640616325736, 6.633524740333777)
    let r = relative_risk([[17, 6], [4, 13]], 0.90).unwrap();
    assert_eq!(r.estimate, q(289, 92));
    close(r.ci.lower, 1.487_564_061_632_573_6, 1e-9);
    close(r.ci.upper, 6.633_524_740_333_777, 1e-9);
    // Zero cells and confidence at the boundary.
    is_invalid(odds_ratio([[0, 6], [4, 13]], 0.95));
    is_invalid(odds_ratio([[17, 6], [4, 13]], 1.0));
    is_invalid(odds_ratio([[17, 6], [4, 13]], 0.0));
    is_invalid(relative_risk([[0, 6], [4, 13]], 0.95));
    // A zero non-case is allowed by relative_risk (a/(a+b) = 1).
    assert_eq!(
        relative_risk([[5, 0], [4, 13]], 0.95).unwrap().estimate,
        q(17, 4)
    );
}

#[test]
fn cohens_h_new_proportions_and_extremes() {
    let ctx = Context::new();
    // statsmodels: proportion_effectsize(0.35, 0.6) = -0.5060505748057285
    let h = cohens_h(&ctx, &q(7, 20), &q(3, 5)).unwrap();
    close(h.eval_f64().unwrap(), -0.506_050_574_805_728_5, 1e-12);
    // statsmodels: proportion_effectsize(0.0, 1.0) = -3.141592653589793 (= −π exactly)
    let h = cohens_h(&ctx, &qi(0), &qi(1)).unwrap();
    close(h.eval_f64().unwrap(), -std::f64::consts::PI, 1e-15);
    close(
        cohens_h(&ctx, &q(1, 3), &q(1, 3))
            .unwrap()
            .eval_f64()
            .unwrap(),
        0.0,
        1e-15,
    );
    is_invalid(cohens_h(&ctx, &q(4, 3), &q(1, 3)));
}

// ═══════════════════════════════════════════════════════════════════════
// hypothesis: means and proportions
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn t_test_one_sample_t1_all_alternatives_rational_and_negative_data() {
    let ctx = Context::new();
    // scipy: ttest_1samp(T1, 13) → statistic 2.0, pvalue 0.07338803477074037, df 10;
    //   alternative='less' 0.9633059826146299, 'greater' 0.03669401738537018
    let r = t_test_one_sample(&ctx, &t1(), &qi(13), Alternative::TwoSided).unwrap();
    assert_eq!(r.statistic, ctx.int(2));
    close(p_of(&r), 0.073_388_034_770_740_37, 1e-12);
    assert_eq!(r.df, Some(ctx.int(10)));
    let r = t_test_one_sample(&ctx, &t1(), &qi(13), Alternative::Less).unwrap();
    close(p_of(&r), 0.963_305_982_614_629_9, 1e-12);
    let r = t_test_one_sample(&ctx, &t1(), &qi(13), Alternative::Greater).unwrap();
    close(p_of(&r), 0.036_694_017_385_370_18, 1e-12);
    // Rational data: x = [1/2, 3/4, 5/4, 2, 7/4, 1/4], μ₀ = 1 → t² = 5/59;
    // scipy: ttest_1samp([.5, .75, 1.25, 2, 1.75, .25], 1) → statistic 0.2911112548697908, pvalue 0.7826557839201314
    let x = vec![q(1, 2), q(3, 4), q(5, 4), qi(2), q(7, 4), q(1, 4)];
    let r = t_test_one_sample(&ctx, &x, &qi(1), Alternative::TwoSided).unwrap();
    assert_eq!(
        (&r.statistic * &r.statistic).simplify(),
        ctx.rational(5, 59)
    );
    close(s_of(&r), 0.291_111_254_869_790_8, 1e-12);
    close(p_of(&r), 0.782_655_783_920_131_4, 1e-12);
    // Negative data and null: scipy: ttest_1samp([-3, -7, -2, -9, -5, -4, -8], -4)
    //   → statistic -1.43345544770249, pvalue 0.20171364634723996
    let r = t_test_one_sample(
        &ctx,
        &from_i64(&[-3, -7, -2, -9, -5, -4, -8]),
        &qi(-4),
        Alternative::TwoSided,
    )
    .unwrap();
    close(s_of(&r), -1.433_455_447_702_49, 1e-12);
    close(p_of(&r), 0.201_713_646_347_239_96, 1e-12);
}

#[test]
fn t_test_two_sample_unbalanced_student_and_welch_all_alternatives() {
    let ctx = Context::new();
    let (x, y) = x8y6();
    // scipy: ttest_ind(X8, Y6, equal_var=True) → statistic -2.040634002046935, df 12;
    //   pvalue two-sided 0.06391931826596217, 'less' 0.031959659132981086, 'greater' 0.968040340867019
    let r = t_test_two_sample(&ctx, &x, &y, true, Alternative::TwoSided).unwrap();
    close(s_of(&r), -2.040_634_002_046_935, 1e-12);
    close(p_of(&r), 0.063_919_318_265_962_17, 1e-12);
    assert_eq!(r.df, Some(ctx.int(12)));
    let r = t_test_two_sample(&ctx, &x, &y, true, Alternative::Less).unwrap();
    close(p_of(&r), 0.031_959_659_132_981_086, 1e-12);
    let r = t_test_two_sample(&ctx, &x, &y, true, Alternative::Greater).unwrap();
    close(p_of(&r), 0.968_040_340_867_019, 1e-12);
    // scipy: ttest_ind(X8, Y6, equal_var=False) → statistic -1.8864844365675968, df 7.457033589826292
    //   (= 161840/21703); pvalue 0.09862271481294474, 'less' 0.04931135740647237, 'greater' 0.9506886425935276
    let r = t_test_two_sample(&ctx, &x, &y, false, Alternative::TwoSided).unwrap();
    close(s_of(&r), -1.886_484_436_567_596_8, 1e-12);
    close(p_of(&r), 0.098_622_714_812_944_74, 1e-12);
    assert_eq!(r.df, Some(ctx.from_ratio(q(161_840, 21_703))));
    let r = t_test_two_sample(&ctx, &x, &y, false, Alternative::Less).unwrap();
    close(p_of(&r), 0.049_311_357_406_472_37, 1e-12);
    let r = t_test_two_sample(&ctx, &x, &y, false, Alternative::Greater).unwrap();
    close(p_of(&r), 0.950_688_642_593_527_6, 1e-12);
}

#[test]
fn t_test_two_sample_one_constant_sample_and_very_unbalanced_sizes() {
    let ctx = Context::new();
    let xc = from_i64(&[5, 5, 5, 5]);
    let yc = from_i64(&[6, 8, 7, 9, 10]);
    // One constant sample is allowed (only both-constant is degenerate).
    // scipy: ttest_ind([5]*4, [6, 8, 7, 9, 10], equal_var=False) → statistic -4.242640687119285,
    //   pvalue 0.01323559956368269, df 4.0
    let r = t_test_two_sample(&ctx, &xc, &yc, false, Alternative::TwoSided).unwrap();
    close(s_of(&r), -4.242_640_687_119_285, 1e-12);
    close(p_of(&r), 0.013_235_599_563_682_69, 1e-12);
    assert_eq!(r.df, Some(ctx.int(4)));
    // scipy: ttest_ind(..., equal_var=True) → statistic -3.7416573867739413, pvalue 0.007246989820287885, df 7.0
    let r = t_test_two_sample(&ctx, &xc, &yc, true, Alternative::TwoSided).unwrap();
    close(s_of(&r), -3.741_657_386_773_941_3, 1e-12);
    close(p_of(&r), 0.007_246_989_820_287_885, 1e-12);
    assert_eq!(r.df, Some(ctx.int(7)));
    // 2 vs 20: scipy: ttest_ind([3, 4], range(1, 21), equal_var=False) → statistic -4.949747468305833,
    //   pvalue 0.00010545521827078808, df 17.88235294117647 (= 304/17)
    let y20: Vec<Q> = (1..=20).map(qi).collect();
    let r =
        t_test_two_sample(&ctx, &from_i64(&[3, 4]), &y20, false, Alternative::TwoSided).unwrap();
    close(s_of(&r), -4.949_747_468_305_833, 1e-12);
    close(p_of(&r), 0.000_105_455_218_270_788_08, 1e-12);
    assert_eq!(r.df, Some(ctx.from_ratio(q(304, 17))));
    is_invalid(t_test_two_sample(
        &ctx,
        &xc,
        &from_i64(&[7, 7]),
        false,
        Alternative::TwoSided,
    ));
}

#[test]
fn t_test_paired_with_a_zero_difference() {
    let ctx = Context::new();
    let bef = from_i64(&[88, 92, 79, 85, 90, 84, 77, 95]);
    let aft = from_i64(&[85, 90, 79, 80, 91, 78, 76, 90]);
    // scipy: ttest_rel(BEF, AFT) → statistic 2.900248979258397, pvalue 0.02297776913201142;
    //   alternative='greater' 0.01148888456600571
    let r = t_test_paired(&ctx, &bef, &aft, Alternative::TwoSided).unwrap();
    close(s_of(&r), 2.900_248_979_258_397, 1e-12);
    close(p_of(&r), 0.022_977_769_132_011_42, 1e-12);
    assert_eq!(r.df, Some(ctx.int(7)));
    let r = t_test_paired(&ctx, &bef, &aft, Alternative::Greater).unwrap();
    close(p_of(&r), 0.011_488_884_566_005_71, 1e-12);
    // Equals the one-sample test of the differences against 0.
    let d: Vec<Q> = bef.iter().zip(&aft).map(|(a, b)| a - b).collect();
    let one = t_test_one_sample(&ctx, &d, &qi(0), Alternative::Greater).unwrap();
    assert_eq!(one, r);
    is_invalid(t_test_paired(&ctx, &bef, &aft[..7], Alternative::TwoSided));
    is_invalid(t_test_paired(&ctx, &bef, &bef, Alternative::TwoSided));
}

#[test]
fn z_tests_for_proportions_new_counts_all_alternatives() {
    let ctx = Context::new();
    // statsmodels: proportions_ztest(37, 120, value=0.25, prop_var=0.25, alternative=…)
    //   → z 1.4757295747452441; two-sided 0.14001650319716888, 'smaller' 0.9299917484014155, 'larger' 0.07000825159858444
    let r = z_test_proportion(&ctx, 37, 120, &q(1, 4), Alternative::TwoSided).unwrap();
    close(s_of(&r), 1.475_729_574_745_244_1, 1e-12);
    close(p_of(&r), 0.140_016_503_197_168_88, 1e-12);
    assert_eq!(r.df, None);
    let r = z_test_proportion(&ctx, 37, 120, &q(1, 4), Alternative::Less).unwrap();
    close(p_of(&r), 0.929_991_748_401_415_5, 1e-12);
    let r = z_test_proportion(&ctx, 37, 120, &q(1, 4), Alternative::Greater).unwrap();
    close(p_of(&r), 0.070_008_251_598_584_44, 1e-12);
    // statsmodels: proportions_ztest([18, 33], [40, 55]) → z -1.4476146708617306;
    //   two-sided 0.14772484583052478, 'smaller' 0.07386242291526239, 'larger' 0.9261375770847377
    let r = z_test_two_proportions(&ctx, 18, 40, 33, 55, Alternative::TwoSided).unwrap();
    close(s_of(&r), -1.447_614_670_861_730_6, 1e-12);
    close(p_of(&r), 0.147_724_845_830_524_78, 1e-12);
    let r = z_test_two_proportions(&ctx, 18, 40, 33, 55, Alternative::Less).unwrap();
    close(p_of(&r), 0.073_862_422_915_262_39, 1e-12);
    let r = z_test_two_proportions(&ctx, 18, 40, 33, 55, Alternative::Greater).unwrap();
    close(p_of(&r), 0.926_137_577_084_737_7, 1e-12);
    // k₁ = 0 with a non-degenerate pooled proportion:
    // statsmodels: proportions_ztest([0, 5], [10, 10]) → (-2.581988897471611, 0.009823274507519247)
    let r = z_test_two_proportions(&ctx, 0, 10, 5, 10, Alternative::TwoSided).unwrap();
    close(s_of(&r), -2.581_988_897_471_611, 1e-12);
    close(p_of(&r), 0.009_823_274_507_519_247, 1e-12);
    // Validation: p₀ at the boundary, pooled 0 or 1, k > n.
    is_invalid(z_test_proportion(
        &ctx,
        3,
        10,
        &qi(0),
        Alternative::TwoSided,
    ));
    is_invalid(z_test_proportion(
        &ctx,
        3,
        10,
        &qi(1),
        Alternative::TwoSided,
    ));
    is_invalid(z_test_two_proportions(
        &ctx,
        0,
        10,
        0,
        10,
        Alternative::TwoSided,
    ));
    is_invalid(z_test_two_proportions(
        &ctx,
        10,
        10,
        10,
        10,
        Alternative::TwoSided,
    ));
    is_invalid(z_test_two_proportions(
        &ctx,
        11,
        10,
        5,
        10,
        Alternative::TwoSided,
    ));
}

#[test]
fn anova_one_way_unbalanced_four_groups_exact() {
    let ctx = Context::new();
    // scipy: f_oneway(*G4) → statistic 11.144811997922975 (= 364870/32739), pvalue 0.0015739107298782685
    // Fraction: SS_between 36487/210, SS_within 1559/30, η² 36487/47400, df (3, 10)
    let r = anova_one_way(&ctx, &g4()).unwrap();
    assert_eq!(r.f, q(364_870, 32_739));
    assert_eq!(r.ss_between, q(36_487, 210));
    assert_eq!(r.ss_within, q(1_559, 30));
    assert_eq!(r.eta_squared, q(36_487, 47_400));
    assert_eq!((r.df_between, r.df_within), (3, 10));
    close(r.p_value_f64().unwrap(), 0.001_573_910_729_878_268_5, 1e-12);
    assert_eq!(eta_squared(&g4()).unwrap(), r.eta_squared);
    // Singleton groups are fine while N > k: scipy: f_oneway([5], [7], [4, 9])
    //   → statistic 0.09 (= 9/100), pvalue 0.9205746178983234
    let r = anova_one_way(&ctx, &[from_i64(&[5]), from_i64(&[7]), from_i64(&[4, 9])]).unwrap();
    assert_eq!(r.f, q(9, 100));
    close(r.p_value_f64().unwrap(), 0.920_574_617_898_323_4, 1e-12);
    // Equal group means: scipy: f_oneway([1, 3], [2, 2], [0, 4]) → statistic 0.0, pvalue 1.0
    let r = anova_one_way(
        &ctx,
        &[from_i64(&[1, 3]), from_i64(&[2, 2]), from_i64(&[0, 4])],
    )
    .unwrap();
    assert_eq!(r.f, qi(0));
    assert_eq!(r.eta_squared, qi(0));
    close(r.p_value_f64().unwrap(), 1.0, 1e-15);
    // η² alone rejects an all-identical sample; ANOVA rejects zero within variance.
    is_invalid(eta_squared(&[from_i64(&[2, 2]), from_i64(&[2])]));
    is_invalid(anova_one_way(&ctx, &[from_i64(&[1, 1]), from_i64(&[3, 3])]));
}

#[test]
fn confidence_interval_mean_at_several_levels_and_n_equals_2() {
    // scipy: t.interval(c, 10, loc=mean(T1), scale=sem(T1)):
    //   0.99 → (11.83072732738305, 18.16927267261695); 0.5 → (14.300187938687568, 15.699812061312432);
    //   0.8 → (13.627816358889664, 16.372183641110336)
    let ci = confidence_interval_mean(&t1(), 0.99).unwrap();
    close(ci.lower, 11.830_727_327_383_05, 1e-9);
    close(ci.upper, 18.169_272_672_616_95, 1e-9);
    let ci = confidence_interval_mean(&t1(), 0.5).unwrap();
    close(ci.lower, 14.300_187_938_687_568, 1e-9);
    close(ci.upper, 15.699_812_061_312_432, 1e-9);
    let ci = confidence_interval_mean(&t1(), 0.8).unwrap();
    close(ci.lower, 13.627_816_358_889_664, 1e-9);
    close(ci.upper, 16.372_183_641_110_336, 1e-9);
    // n = 2 (df = 1, the Cauchy quantile): scipy: t.interval(0.95, 1, loc=3.5, scale=sem([3, 4]))
    //   = (-2.853102368087347, 9.853102368087347)
    let ci = confidence_interval_mean(&from_i64(&[3, 4]), 0.95).unwrap();
    close(ci.lower, -2.853_102_368_087_347, 1e-9);
    close(ci.upper, 9.853_102_368_087_347, 1e-9);
    is_invalid(confidence_interval_mean(&t1(), 1.0));
    is_invalid(confidence_interval_mean(&t1(), 0.0));
}

// ═══════════════════════════════════════════════════════════════════════
// hypothesis: rank tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn mann_whitney_exact_new_samples_all_alternatives() {
    let ctx = Context::new();
    let (x, y) = mw();
    // Fraction: exact distribution of U over C(10, 6) arrangements; U₁ = 16.
    // scipy: mannwhitneyu(MWX, MWY, method='exact') → (16.0, 0.47619047619047616) (= 10/21);
    //   alternative='less' 0.8238095238095238 (= 173/210), 'greater' 0.23809523809523808 (= 5/21)
    let r = mann_whitney_u(&ctx, &x, &y, Alternative::TwoSided, RankMethod::Exact).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(16)));
    assert_eq!(r.p_value_exact(), Some(q(10, 21)));
    let r = mann_whitney_u(&ctx, &x, &y, Alternative::Less, RankMethod::Exact).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(173, 210)));
    let r = mann_whitney_u(&ctx, &x, &y, Alternative::Greater, RankMethod::Exact).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(5, 21)));
    // Swapping the samples: U₂ = n₁n₂ − U₁ and the one-sided tails swap.
    let s = mann_whitney_u(&ctx, &y, &x, Alternative::Less, RankMethod::Exact).unwrap();
    assert_eq!(s.statistic_exact(), Some(qi(8)));
    assert_eq!(s.p_value_exact(), Some(q(5, 21)));
}

#[test]
fn mann_whitney_asymptotic_with_and_without_continuity_and_with_ties() {
    let ctx = Context::new();
    let (x, y) = mw();
    // scipy: mannwhitneyu(MWX, MWY, method='asymptotic', alternative=…) (use_continuity=True):
    //   two-sided 0.45554509378932995, 'less' 0.8313221741118193, 'greater' 0.22777254689466497
    let r = mann_whitney_u(&ctx, &x, &y, Alternative::TwoSided, asym(true)).unwrap();
    close(p_of(&r), 0.455_545_093_789_329_95, 1e-12);
    let r = mann_whitney_u(&ctx, &x, &y, Alternative::Less, asym(true)).unwrap();
    close(p_of(&r), 0.831_322_174_111_819_3, 1e-12);
    let r = mann_whitney_u(&ctx, &x, &y, Alternative::Greater, asym(true)).unwrap();
    close(p_of(&r), 0.227_772_546_894_664_97, 1e-12);
    // use_continuity=False: two-sided 0.3937686346429927, 'less' 0.8031156826785036, 'greater' 0.19688431732149636
    let r = mann_whitney_u(&ctx, &x, &y, Alternative::TwoSided, asym(false)).unwrap();
    close(p_of(&r), 0.393_768_634_642_992_7, 1e-12);
    let r = mann_whitney_u(&ctx, &x, &y, Alternative::Less, asym(false)).unwrap();
    close(p_of(&r), 0.803_115_682_678_503_6, 1e-12);
    let r = mann_whitney_u(&ctx, &x, &y, Alternative::Greater, asym(false)).unwrap();
    close(p_of(&r), 0.196_884_317_321_496_36, 1e-12);
    // Ties: scipy: mannwhitneyu(MWXT, MWYT, method='asymptotic') → U 20.0;
    //   two-sided 0.4026252422331532, 'less' 0.8468292810861445, 'greater' 0.2013126211165766;
    //   use_continuity=False two-sided 0.3524045256120931
    let (xt, yt) = mw_ties();
    let r = mann_whitney_u(&ctx, &xt, &yt, Alternative::TwoSided, asym(true)).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(20)));
    close(p_of(&r), 0.402_625_242_233_153_2, 1e-12);
    let r = mann_whitney_u(&ctx, &xt, &yt, Alternative::Less, asym(true)).unwrap();
    close(p_of(&r), 0.846_829_281_086_144_5, 1e-12);
    let r = mann_whitney_u(&ctx, &xt, &yt, Alternative::Greater, asym(true)).unwrap();
    close(p_of(&r), 0.201_312_621_116_576_6, 1e-12);
    let r = mann_whitney_u(&ctx, &xt, &yt, Alternative::TwoSided, asym(false)).unwrap();
    close(p_of(&r), 0.352_404_525_612_093_1, 1e-12);
    is_invalid(mann_whitney_u(
        &ctx,
        &xt,
        &yt,
        Alternative::TwoSided,
        RankMethod::Exact,
    ));
}

#[test]
fn mann_whitney_degenerate_sizes_and_central_statistic() {
    let ctx = Context::new();
    // U₁ = U₂ = n₁n₂/2: scipy: mannwhitneyu([1, 4], [2, 3], method='asymptotic') = (2.0, 1.0); exact (2.0, 1.0)
    let (a, b) = (from_i64(&[1, 4]), from_i64(&[2, 3]));
    let r = mann_whitney_u(&ctx, &a, &b, Alternative::TwoSided, asym(true)).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(2)));
    close(p_of(&r), 1.0, 1e-15);
    let r = mann_whitney_u(&ctx, &a, &b, Alternative::TwoSided, RankMethod::Exact).unwrap();
    assert_eq!(r.p_value_exact(), Some(qi(1)));
    // One observation each: scipy: mannwhitneyu([1], [2], method='exact') = (0.0, 1.0); asymptotic (0.0, 1.0)
    let r = mann_whitney_u(
        &ctx,
        &from_i64(&[1]),
        &from_i64(&[2]),
        Alternative::TwoSided,
        RankMethod::Exact,
    )
    .unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(0)));
    assert_eq!(r.p_value_exact(), Some(qi(1)));
    let r = mann_whitney_u(
        &ctx,
        &from_i64(&[1]),
        &from_i64(&[2]),
        Alternative::TwoSided,
        asym(true),
    )
    .unwrap();
    close(p_of(&r), 1.0, 1e-15);
    // 1 vs 9: scipy: mannwhitneyu([10], range(1, 10), method='exact') = (9.0, 0.2); 'greater' 0.1
    let y9: Vec<Q> = (1..=9).map(qi).collect();
    let r = mann_whitney_u(
        &ctx,
        &from_i64(&[10]),
        &y9,
        Alternative::TwoSided,
        RankMethod::Exact,
    )
    .unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(9)));
    assert_eq!(r.p_value_exact(), Some(q(1, 5)));
    let r = mann_whitney_u(
        &ctx,
        &from_i64(&[10]),
        &y9,
        Alternative::Greater,
        RankMethod::Exact,
    )
    .unwrap();
    assert_eq!(r.p_value_exact(), Some(q(1, 10)));
    // All identical: the asymptotic variance vanishes.
    is_invalid(mann_whitney_u(
        &ctx,
        &from_i64(&[2, 2]),
        &from_i64(&[2]),
        Alternative::TwoSided,
        asym(true),
    ));
    is_invalid(mann_whitney_u(
        &ctx,
        &[],
        &from_i64(&[2]),
        Alternative::TwoSided,
        asym(true),
    ));
}

#[test]
fn kruskal_wallis_with_two_groups_is_the_squared_uncorrected_mann_whitney_z() {
    let ctx = Context::new();
    let (x, y) = mw();
    // scipy: kruskal(MWX, MWY) → statistic 0.7272727272727195, pvalue 0.3937686346429954
    let h = kruskal_wallis(&ctx, &[x.clone(), y.clone()]).unwrap();
    assert_eq!(h.statistic_exact(), Some(q(8, 11)));
    close(p_of(&h), 0.393_768_634_642_995_4, 1e-12);
    // z² of the tie-corrected normal approximation without continuity: (U₁ − n₁n₂/2)² / σ² = 8/11.
    let z = mann_whitney_u(&ctx, &x, &y, Alternative::TwoSided, asym(false)).unwrap();
    close(p_of(&z), p_of(&h), 1e-13);
    // With ties too: kruskal(MWXT, MWYT) equals the squared tie-corrected z.
    let (xt, yt) = mw_ties();
    let h = kruskal_wallis(&ctx, &[xt.clone(), yt.clone()]).unwrap();
    let z = mann_whitney_u(&ctx, &xt, &yt, Alternative::TwoSided, asym(false)).unwrap();
    close(p_of(&z), p_of(&h), 1e-13);
}

#[test]
fn wilcoxon_one_sample_exact_and_asymptotic_all_alternatives() {
    let ctx = Context::new();
    let d = wd();
    // Fraction: T⁺ = 56 over the 2¹¹ sign patterns; two-sided statistic min(T⁺, T⁻) = 10.
    // scipy: wilcoxon(WD, method='exact') → (10.0, 0.0419921875) (= 43/1024);
    //   alternative='less' (56.0, 0.98388671875) (= 2015/2048), 'greater' (56.0, 0.02099609375) (= 43/2048)
    let r = wilcoxon_signed_rank(&ctx, &d, None, Alternative::TwoSided, RankMethod::Exact).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(10)));
    assert_eq!(r.p_value_exact(), Some(q(43, 1_024)));
    let r = wilcoxon_signed_rank(&ctx, &d, None, Alternative::Less, RankMethod::Exact).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(56)));
    assert_eq!(r.p_value_exact(), Some(q(2_015, 2_048)));
    let r = wilcoxon_signed_rank(&ctx, &d, None, Alternative::Greater, RankMethod::Exact).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(43, 2_048)));
    // scipy: wilcoxon(WD, method='approx', correction=True): two-sided 0.0454469460731259,
    //   'less' 0.9816643527175838, 'greater' 0.02272347303656295;
    //   correction=False: 0.04085984371280629, 0.9795700781435969, 0.020429921856403147
    let r = wilcoxon_signed_rank(&ctx, &d, None, Alternative::TwoSided, asym(true)).unwrap();
    close(p_of(&r), 0.045_446_946_073_125_9, 1e-12);
    let r = wilcoxon_signed_rank(&ctx, &d, None, Alternative::Less, asym(true)).unwrap();
    close(p_of(&r), 0.981_664_352_717_583_8, 1e-12);
    let r = wilcoxon_signed_rank(&ctx, &d, None, Alternative::Greater, asym(true)).unwrap();
    close(p_of(&r), 0.022_723_473_036_562_95, 1e-12);
    let r = wilcoxon_signed_rank(&ctx, &d, None, Alternative::TwoSided, asym(false)).unwrap();
    close(p_of(&r), 0.040_859_843_712_806_29, 1e-12);
    let r = wilcoxon_signed_rank(&ctx, &d, None, Alternative::Less, asym(false)).unwrap();
    close(p_of(&r), 0.979_570_078_143_596_9, 1e-12);
    let r = wilcoxon_signed_rank(&ctx, &d, None, Alternative::Greater, asym(false)).unwrap();
    close(p_of(&r), 0.020_429_921_856_403_147, 1e-12);
}

#[test]
fn wilcoxon_paired_with_zeros_and_ties_matches_the_difference_form_and_sign_counts() {
    let ctx = Context::new();
    let x = from_i64(&[10, 12, 9, 15, 11, 14, 13, 8, 16, 12]);
    let y = from_i64(&[8, 12, 11, 12, 10, 10, 13, 9, 12, 9]);
    // scipy: wilcoxon(WPX, WPY, method='approx', correction=True) → (5.0, 0.07857855114584317);
    //   alternative='greater' → (31.0, 0.039289275572921584); correction=False → (5.0, 0.06734665140786797)
    let r = wilcoxon_signed_rank(&ctx, &x, Some(&y), Alternative::TwoSided, asym(true)).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(5)));
    close(p_of(&r), 0.078_578_551_145_843_17, 1e-12);
    let r = wilcoxon_signed_rank(&ctx, &x, Some(&y), Alternative::Greater, asym(true)).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(31)));
    close(p_of(&r), 0.039_289_275_572_921_584, 1e-12);
    let r = wilcoxon_signed_rank(&ctx, &x, Some(&y), Alternative::TwoSided, asym(false)).unwrap();
    close(p_of(&r), 0.067_346_651_407_867_97, 1e-12);
    // Paired form ≡ one-sample form of the differences.
    let d: Vec<Q> = x.iter().zip(&y).map(|(a, b)| a - b).collect();
    let one = wilcoxon_signed_rank(&ctx, &d, None, Alternative::TwoSided, asym(false)).unwrap();
    assert_eq!(one, r);
    // The sign structure: 6 positive, 2 negative, 2 zero differences → sign test M = 2,
    // scipy: binomtest(6, 8, 0.5).pvalue = 0.2890625 (= 37/128).
    let s = sign_test(&ctx, &d, &qi(0), Alternative::TwoSided).unwrap();
    assert_eq!(s.statistic_exact(), Some(qi(2)));
    assert_eq!(s.p_value_exact(), Some(q(37, 128)));
    // Ties in |d| forbid the exact method; every difference zero is an error.
    is_invalid(wilcoxon_signed_rank(
        &ctx,
        &x,
        Some(&y),
        Alternative::TwoSided,
        RankMethod::Exact,
    ));
    is_invalid(wilcoxon_signed_rank(
        &ctx,
        &x,
        Some(&x),
        Alternative::TwoSided,
        asym(true),
    ));
    is_invalid(wilcoxon_signed_rank(
        &ctx,
        &x,
        Some(&y[..9]),
        Alternative::TwoSided,
        asym(true),
    ));
}

#[test]
fn wilcoxon_degenerate_cases_single_difference_and_central_rank_sum() {
    let ctx = Context::new();
    // T⁺ = n(n+1)/4 exactly: scipy: wilcoxon([3, -1, -2], method='approx', correction=True) = (3.0, 1.0); exact (3.0, 1.0)
    let d = from_i64(&[3, -1, -2]);
    let r = wilcoxon_signed_rank(&ctx, &d, None, Alternative::TwoSided, asym(true)).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(3)));
    close(p_of(&r), 1.0, 1e-15);
    let r = wilcoxon_signed_rank(&ctx, &d, None, Alternative::TwoSided, RankMethod::Exact).unwrap();
    assert_eq!(r.p_value_exact(), Some(qi(1)));
    // A single non-zero difference: scipy: wilcoxon([5], method='exact') = (0.0, 1.0); 'greater' (1.0, 0.5)
    let r = wilcoxon_signed_rank(
        &ctx,
        &from_i64(&[5]),
        None,
        Alternative::TwoSided,
        RankMethod::Exact,
    )
    .unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(0)));
    assert_eq!(r.p_value_exact(), Some(qi(1)));
    let r = wilcoxon_signed_rank(
        &ctx,
        &from_i64(&[5]),
        None,
        Alternative::Greater,
        RankMethod::Exact,
    )
    .unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(1)));
    assert_eq!(r.p_value_exact(), Some(q(1, 2)));
    let r = wilcoxon_signed_rank(
        &ctx,
        &from_i64(&[5]),
        None,
        Alternative::TwoSided,
        asym(true),
    )
    .unwrap();
    close(p_of(&r), 1.0, 1e-15);
}

#[test]
fn kruskal_wallis_four_unbalanced_groups_with_ties() {
    let ctx = Context::new();
    let g = vec![
        from_i64(&[3, 5, 5, 8]),
        from_i64(&[6, 9, 9, 12, 7]),
        from_i64(&[2, 4, 3]),
        from_i64(&[11, 9, 14, 6]),
    ];
    // scipy: kruskal(*KW) → statistic 10.22696879643388 (= 27531/2692), pvalue 0.01673214929829152
    let r = kruskal_wallis(&ctx, &g).unwrap();
    assert_eq!(r.statistic_exact(), Some(q(27_531, 2_692)));
    close(p_of(&r), 0.016_732_149_298_291_52, 1e-12);
    assert_eq!(r.df, Some(ctx.int(3)));
    // Two singletons: H = 1 for [1] vs [2] (ranks 1, 2), P(χ²₁ ≥ 1) = 0.3173105078629141.
    let r = kruskal_wallis(&ctx, &[from_i64(&[1]), from_i64(&[2])]).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(1)));
    close(p_of(&r), 0.317_310_507_862_914_1, 1e-12);
    is_invalid(kruskal_wallis(&ctx, &[from_i64(&[4, 4]), from_i64(&[4])]));
}

#[test]
fn friedman_five_blocks_four_treatments_with_ties() {
    let ctx = Context::new();
    let blocks = vec![
        from_i64(&[7, 9, 6, 9]),
        from_i64(&[5, 8, 5, 7]),
        from_i64(&[8, 8, 6, 10]),
        from_i64(&[6, 9, 7, 7]),
        from_i64(&[7, 10, 5, 8]),
    ];
    // scipy: friedmanchisquare(*columns of FR) → statistic 11.086956521739133 (= 255/23), pvalue 0.011264842525771727
    let r = friedman(&ctx, &blocks).unwrap();
    assert_eq!(r.statistic_exact(), Some(q(255, 23)));
    close(p_of(&r), 0.011_264_842_525_771_727, 1e-12);
    assert_eq!(r.df, Some(ctx.int(3)));
    // A single block: scipy: friedmanchisquare([1], [2], [3]) → statistic 2.0, pvalue 0.36787944117144245
    let r = friedman(&ctx, &[from_i64(&[1, 2, 3])]).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(2)));
    close(p_of(&r), 0.367_879_441_171_442_45, 1e-12);
    is_invalid(friedman(
        &ctx,
        &[from_i64(&[1, 1, 1]), from_i64(&[5, 5, 5])],
    ));
    is_invalid(friedman(&ctx, &[]));
}

#[test]
fn spearman_test_with_ties_agrees_with_data_spearman_and_pearson_of_ranks() {
    let ctx = Context::new();
    let x = from_i64(&[3, 1, 4, 1, 5, 9, 2, 6]);
    let y = from_i64(&[2, 7, 1, 8, 2, 8, 1, 8]);
    // scipy: spearmanr(SPX, SPY) → statistic 0.19885368120992467 (ρ² = 128/3237 exactly),
    //   pvalue 0.6368617833253285; alternative='less' 0.6815691083373357, 'greater' 0.31843089166266425
    let r = spearman_test(&ctx, &x, &y, Alternative::TwoSided).unwrap();
    close(s_of(&r), 0.198_853_681_209_924_67, 1e-12);
    close(p_of(&r), 0.636_861_783_325_328_5, 1e-12);
    assert_eq!(
        (&r.statistic * &r.statistic).simplify(),
        ctx.rational(128, 3_237)
    );
    assert_eq!(r.df, Some(ctx.int(6)));
    let r = spearman_test(&ctx, &x, &y, Alternative::Less).unwrap();
    close(p_of(&r), 0.681_569_108_337_335_7, 1e-12);
    let r = spearman_test(&ctx, &x, &y, Alternative::Greater).unwrap();
    close(p_of(&r), 0.318_430_891_662_664_25, 1e-12);
    // Three routes to ρ agree exactly.
    let rho = spearman(&ctx, &x, &y).unwrap();
    assert_eq!(r.statistic, rho);
    assert_eq!(pearson(&ctx, &ranks(&x), &ranks(&y)).unwrap(), rho);
    // No ties, n = 5: scipy: spearmanr([10, 20, 30, 40, 50], [3, 1, 4, 5, 2]) → statistic 0.19999999999999998
    //   (= 1/5), pvalue 0.7470600781046621
    let r = spearman_test(
        &ctx,
        &from_i64(&[10, 20, 30, 40, 50]),
        &from_i64(&[3, 1, 4, 5, 2]),
        Alternative::TwoSided,
    )
    .unwrap();
    assert_eq!(r.statistic, ctx.rational(1, 5));
    close(p_of(&r), 0.747_060_078_104_662_1, 1e-12);
}

#[test]
fn spearman_test_perfect_negative_and_minimal_n() {
    let ctx = Context::new();
    let (x, y) = (from_i64(&[1, 2, 3]), from_i64(&[3, 2, 1]));
    // scipy: spearmanr([1, 2, 3], [3, 2, 1], alternative='greater') → (-1.0, 1.0); 'less' → (-1.0, 0.0)
    let r = spearman_test(&ctx, &x, &y, Alternative::Greater).unwrap();
    assert_eq!(r.statistic, ctx.int(-1));
    assert_eq!(r.p_value, ctx.one());
    let r = spearman_test(&ctx, &x, &y, Alternative::Less).unwrap();
    assert_eq!(r.p_value, ctx.zero());
    // scipy: spearmanr([1, 2, 3], [1, 3, 2]) → statistic 0.5, pvalue 0.6666666666666666
    let r = spearman_test(&ctx, &x, &from_i64(&[1, 3, 2]), Alternative::TwoSided).unwrap();
    assert_eq!(r.statistic, ctx.rational(1, 2));
    close(p_of(&r), 0.666_666_666_666_666_6, 1e-12);
    is_invalid(spearman_test(
        &ctx,
        &from_i64(&[1, 2]),
        &from_i64(&[2, 1]),
        Alternative::TwoSided,
    ));
    is_invalid(spearman_test(
        &ctx,
        &from_i64(&[2, 2, 2]),
        &x,
        Alternative::TwoSided,
    ));
}

#[test]
fn kendall_exact_n6_positive_and_negative_tau_all_alternatives() {
    let ctx = Context::new();
    let x = from_i64(&[1, 2, 3, 4, 5, 6]);
    let y = from_i64(&[3, 1, 2, 6, 4, 5]);
    // Fraction: 4 discordant of 15 pairs → τ = 7/15; Mahonian numbers of n = 6.
    // scipy: kendalltau(x, y, method='exact') → (0.4666666666666666, 0.2722222222222222) (= 49/180);
    //   alternative='less' 0.9319444444444445 (= 671/720), 'greater' 0.1361111111111111 (= 49/360)
    let r = kendall_test(&ctx, &x, &y, Alternative::TwoSided, true).unwrap();
    assert_eq!(r.statistic, ctx.rational(7, 15));
    assert_eq!(r.p_value_exact(), Some(q(49, 180)));
    let r = kendall_test(&ctx, &x, &y, Alternative::Less, true).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(671, 720)));
    let r = kendall_test(&ctx, &x, &y, Alternative::Greater, true).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(49, 360)));
    // scipy: kendalltau(x, y, method='asymptotic'): two-sided 0.18848603806737496,
    //   'less' 0.9057569809663125, 'greater' 0.09424301903368748
    let r = kendall_test(&ctx, &x, &y, Alternative::TwoSided, false).unwrap();
    close(p_of(&r), 0.188_486_038_067_374_96, 1e-12);
    let r = kendall_test(&ctx, &x, &y, Alternative::Less, false).unwrap();
    close(p_of(&r), 0.905_756_980_966_312_5, 1e-12);
    let r = kendall_test(&ctx, &x, &y, Alternative::Greater, false).unwrap();
    close(p_of(&r), 0.094_243_019_033_687_48, 1e-12);
    // Negative τ: y = [6, 4, 5, 1, 3, 2] → 12 discordant, τ = −3/5.
    // scipy: kendalltau(x, y2, method='exact') → (-0.6, 0.1361111111111111) (= 49/360);
    //   'less' 0.06805555555555555 (= 49/720), 'greater' 0.9722222222222222 (= 35/36)
    let y2 = from_i64(&[6, 4, 5, 1, 3, 2]);
    let r = kendall_test(&ctx, &x, &y2, Alternative::TwoSided, true).unwrap();
    assert_eq!(r.statistic, ctx.rational(-3, 5));
    assert_eq!(r.p_value_exact(), Some(q(49, 360)));
    let r = kendall_test(&ctx, &x, &y2, Alternative::Less, true).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(49, 720)));
    let r = kendall_test(&ctx, &x, &y2, Alternative::Greater, true).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(35, 36)));
}

#[test]
fn kendall_ties_asymptotic_and_exact_boundary_cases() {
    let ctx = Context::new();
    // scipy: kendalltau([1, 1, 2, 3, 3, 4, 5], [2, 1, 1, 3, 4, 4, 5]) → statistic 0.7894736842105262,
    //   pvalue 0.01845650965747274; alternative='greater' 0.00922825482873637
    let xt = from_i64(&[1, 1, 2, 3, 3, 4, 5]);
    let yt = from_i64(&[2, 1, 1, 3, 4, 4, 5]);
    let r = kendall_test(&ctx, &xt, &yt, Alternative::TwoSided, false).unwrap();
    close(s_of(&r), 0.789_473_684_210_526_2, 1e-12);
    close(p_of(&r), 0.018_456_509_657_472_74, 1e-12);
    assert_eq!(r.statistic, kendall_tau(&ctx, &xt, &yt).unwrap());
    let r = kendall_test(&ctx, &xt, &yt, Alternative::Greater, false).unwrap();
    close(p_of(&r), 0.009_228_254_828_736_37, 1e-12);
    is_invalid(kendall_test(&ctx, &xt, &yt, Alternative::TwoSided, true));
    // n = 2: scipy: kendalltau([1, 2], [2, 1], method='exact') = (-1.0, 1.0)
    //   (scipy's asymptotic branch raises ZeroDivisionError; the crate gives var = 1, p = erfc(1/√2)).
    let r = kendall_test(
        &ctx,
        &from_i64(&[1, 2]),
        &from_i64(&[2, 1]),
        Alternative::TwoSided,
        true,
    )
    .unwrap();
    assert_eq!(r.statistic, ctx.int(-1));
    assert_eq!(r.p_value_exact(), Some(qi(1)));
    let r = kendall_test(
        &ctx,
        &from_i64(&[1, 2]),
        &from_i64(&[2, 1]),
        Alternative::TwoSided,
        false,
    )
    .unwrap();
    close(p_of(&r), 0.317_310_507_862_914_1, 1e-12);
    // τ = 1, n = 4: scipy: kendalltau(range(4), range(4), method='exact') → pvalue 0.08333333333333333 (= 1/12);
    //   'less' 1.0, 'greater' 0.041666666666666664 (= 1/24)
    let i4 = from_i64(&[1, 2, 3, 4]);
    let r = kendall_test(&ctx, &i4, &i4, Alternative::TwoSided, true).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(1, 12)));
    let r = kendall_test(&ctx, &i4, &i4, Alternative::Less, true).unwrap();
    assert_eq!(r.p_value_exact(), Some(qi(1)));
    let r = kendall_test(&ctx, &i4, &i4, Alternative::Greater, true).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(1, 24)));
    // τ = 0, n = 4: scipy: kendalltau(range(4), [3, 1, 4, 2], method='exact') → (0.0, 1.0); 'greater' 0.625, 'less' 0.625
    let y0 = from_i64(&[3, 1, 4, 2]);
    let r = kendall_test(&ctx, &i4, &y0, Alternative::TwoSided, true).unwrap();
    assert_eq!(r.statistic, ctx.zero());
    assert_eq!(r.p_value_exact(), Some(qi(1)));
    let r = kendall_test(&ctx, &i4, &y0, Alternative::Greater, true).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(5, 8)));
    let r = kendall_test(&ctx, &i4, &y0, Alternative::Less, true).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(5, 8)));
    is_invalid(kendall_test(
        &ctx,
        &from_i64(&[2, 2, 2]),
        &i4[..3],
        Alternative::TwoSided,
        false,
    ));
}

#[test]
fn ks_one_sample_exponential_and_uniform_all_alternatives() {
    let ctx = Context::new();
    let x = from_f64(&[0.4, 1.7, 3.2, 0.9, 2.5, 5.1, 1.1, 0.2, 4.4, 2.0]).unwrap();
    let expo = Distribution::exponential(ctx.rational(1, 2));
    // scipy: ks_1samp(x, expon(scale=2).cdf, method='asymp') → statistic 0.17258506805127327, pvalue 0.9270093849648348;
    //   alternative='less' → (0.17258506805127327, 0.4961609490867956); 'greater' → (0.07808166600115318, 0.8463919361799129)
    let r = ks_one_sample(&x, &expo, Alternative::TwoSided).unwrap();
    close(r.statistic, 0.172_585_068_051_273_27, 1e-12);
    close(r.p_value, 0.927_009_384_964_834_8, 1e-9);
    let r = ks_one_sample(&x, &expo, Alternative::Less).unwrap();
    close(r.statistic, 0.172_585_068_051_273_27, 1e-12);
    close(r.p_value, 0.496_160_949_086_795_6, 1e-9);
    let r = ks_one_sample(&x, &expo, Alternative::Greater).unwrap();
    close(r.statistic, 0.078_081_666_001_153_18, 1e-12);
    close(r.p_value, 0.846_391_936_179_912_9, 1e-9);
    // Uniform(0, 1), n = 5: scipy: ks_1samp([.1, .35, .6, .62, .95], uniform.cdf, method='asymp')
    //   → (0.19999999999999996, 0.9882610776435244); 'less' (0.2, 0.5852800000000001); 'greater' (0.18, 0.6510200031999998)
    let u = from_f64(&[0.1, 0.35, 0.6, 0.62, 0.95]).unwrap();
    let unif = Distribution::uniform(ctx.int(0), ctx.int(1));
    let r = ks_one_sample(&u, &unif, Alternative::TwoSided).unwrap();
    close(r.statistic, 0.2, 1e-12);
    close(r.p_value, 0.988_261_077_643_524_4, 1e-9);
    let r = ks_one_sample(&u, &unif, Alternative::Less).unwrap();
    close(r.p_value, 0.585_28, 1e-9);
    let r = ks_one_sample(&u, &unif, Alternative::Greater).unwrap();
    close(r.statistic, 0.18, 1e-12);
    close(r.p_value, 0.651_020_003_2, 1e-9);
}

#[test]
fn ks_one_sample_single_observation_and_extreme_statistics() {
    let ctx = Context::new();
    let unif = Distribution::uniform(ctx.int(0), ctx.int(1));
    let normal = Distribution::normal(ctx.int(0), ctx.int(1));
    // scipy: ks_1samp([0.3], uniform.cdf, method='asymp') → (0.7, 0.7112351950296893);
    //   alternative='greater' → (0.7, 0.30000000000000004); 'less' → (0.3, 0.7)
    let one = from_f64(&[0.3]).unwrap();
    let r = ks_one_sample(&one, &unif, Alternative::TwoSided).unwrap();
    close(r.statistic, 0.7, 1e-12);
    close(r.p_value, 0.711_235_195_029_689_3, 1e-9);
    let r = ks_one_sample(&one, &unif, Alternative::Greater).unwrap();
    close(r.statistic, 0.7, 1e-12);
    close(r.p_value, 0.3, 1e-9);
    let r = ks_one_sample(&one, &unif, Alternative::Less).unwrap();
    close(r.statistic, 0.3, 1e-12);
    close(r.p_value, 0.7, 1e-9);
    // D ≈ 1 (the sample lies far in the tail): scipy: ks_1samp([5, 6, 7, 8], norm.cdf, method='asymp')
    //   → (0.9999997133484281, 0.0006709283329347779)
    let r = ks_one_sample(&from_i64(&[5, 6, 7, 8]), &normal, Alternative::TwoSided).unwrap();
    close(r.statistic, 0.999_999_713_348_428_1, 1e-12);
    close(r.p_value, 0.000_670_928_332_934_777_9, 1e-9);
    // √n·D < 1 exercises the Jacobi-form branch of Kolmogorov's distribution:
    // scipy: ks_1samp([-0.6744897501960817, 0, 0.6744897501960817], norm.cdf, method='asymp') → (0.25, 0.9919638849668337)
    let r = ks_one_sample(
        &from_f64(&[-0.674_489_750_196_081_7, 0.0, 0.674_489_750_196_081_7]).unwrap(),
        &normal,
        Alternative::TwoSided,
    )
    .unwrap();
    close(r.statistic, 0.25, 1e-12);
    close(r.p_value, 0.991_963_884_966_833_7, 1e-9);
    is_invalid(ks_one_sample(&[], &normal, Alternative::TwoSided));
}

// ═══════════════════════════════════════════════════════════════════════
// hypothesis: effect sizes
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn effect_sizes_x8_y6_exact_squares_match_numpy() {
    let ctx = Context::new();
    let (x, y) = x8y6();
    // numpy: pooled d = -1.1020683189683733 (d² = 968/797); unpooled d = -1.0548192632678448 (d² = 484/435);
    //   Hedges g = d·(1 − 3/(4·14 − 9)) = -1.031723532651243; Glass Δ = -0.8602680757063524 (Δ² = 242/327)
    let d = cohens_d(&ctx, &x, &y, true).unwrap();
    close(d.eval_f64().unwrap(), -1.102_068_318_968_373_3, 1e-12);
    assert_eq!((&d * &d).simplify(), ctx.rational(968, 797));
    let d2 = cohens_d(&ctx, &x, &y, false).unwrap();
    close(d2.eval_f64().unwrap(), -1.054_819_263_267_844_8, 1e-12);
    assert_eq!((&d2 * &d2).simplify(), ctx.rational(484, 435));
    let g = hedges_g(&ctx, &x, &y).unwrap();
    close(g.eval_f64().unwrap(), -1.031_723_532_651_243, 1e-12);
    close(
        g.eval_f64().unwrap(),
        d.eval_f64().unwrap() * 44.0 / 47.0,
        1e-14,
    );
    let gl = glass_delta(&ctx, &x, &y).unwrap();
    close(gl.eval_f64().unwrap(), -0.860_268_075_706_352_4, 1e-12);
    assert_eq!((&gl * &gl).simplify(), ctx.rational(242, 327));
    // Glass's Δ needs only one treatment observation; a constant control is rejected.
    assert!(glass_delta(&ctx, &from_i64(&[9]), &y).is_ok());
    is_invalid(glass_delta(&ctx, &x, &from_i64(&[4, 4])));
    is_invalid(cohens_d(
        &ctx,
        &from_i64(&[1, 1]),
        &from_i64(&[2, 2]),
        false,
    ));
}

#[test]
fn cliffs_delta_and_rank_biserial_agree_with_and_without_ties() {
    // MWX vs MWY: #{x > y} − #{x < y} = 16 − 8 → δ = 1/3; U₁ = 16 → r = 32/24 − 1 = 1/3.
    let (x, y) = mw();
    assert_eq!(cliffs_delta(&x, &y).unwrap(), q(1, 3));
    assert_eq!(rank_biserial(&qi(16), 6, 4).unwrap(), q(1, 3));
    assert_eq!(cliffs_delta(&y, &x).unwrap(), q(-1, 3));
    // With ties (MWXT, MWYT): U₁ = 20 (scipy), δ = (#{>} − #{<})/30 = 1/3 = 2·20/30 − 1.
    let (xt, yt) = mw_ties();
    assert_eq!(cliffs_delta(&xt, &yt).unwrap(), q(1, 3));
    assert_eq!(rank_biserial(&qi(20), 6, 5).unwrap(), q(1, 3));
    // Boundaries of U.
    assert_eq!(rank_biserial(&qi(0), 3, 4).unwrap(), qi(-1));
    assert_eq!(rank_biserial(&qi(12), 3, 4).unwrap(), qi(1));
    is_invalid(rank_biserial(&qi(13), 3, 4));
    is_invalid(rank_biserial(&qi(-1), 3, 4));
    is_invalid(rank_biserial(&qi(0), 0, 4));
    is_invalid(cliffs_delta(&[], &x));
}

// ═══════════════════════════════════════════════════════════════════════
// hypothesis: multiple comparisons
// ═══════════════════════════════════════════════════════════════════════

fn assert_adjusted(a: &Adjusted, p: &[f64], reject: &[bool]) {
    assert_eq!(a.p_adjusted.len(), p.len());
    for (i, (got, exp)) in a.p_adjusted.iter().zip(p).enumerate() {
        assert!(
            (got - exp).abs() < 1e-12,
            "p_adjusted[{i}] = {got}, expected {exp}"
        );
    }
    assert_eq!(a.reject, reject);
}

#[test]
fn multiple_comparisons_seven_pvalues_with_ties_zero_and_one() {
    // statsmodels: multipletests(P7, alpha=0.05, method=…):
    //   bonferroni → [0.007, 0.14, 0.14, 1.0, 0.0, 1.0, 0.343], reject [T, F, F, F, T, F, F]
    //   holm       → [0.006, 0.1, 0.1, 1.0, 0.0, 1.0, 0.14700000000000002], reject [T, F, F, F, T, F, F]
    //   fdr_bh     → [0.0035, 0.035, 0.035, 0.5833333333333334, 0.0, 1.0, 0.0686], reject [T, T, T, F, T, F, F]
    //   fdr_by     → [0.009075, 0.09075, 0.09075, 1.0, 0.0, 1.0, 0.17786999999999997], reject [T, F, F, F, T, F, F]
    let r = [true, false, false, false, true, false, false];
    assert_adjusted(
        &bonferroni(&P7, 0.05).unwrap(),
        &[0.007, 0.14, 0.14, 1.0, 0.0, 1.0, 0.343],
        &r,
    );
    assert_adjusted(
        &holm(&P7, 0.05).unwrap(),
        &[0.006, 0.1, 0.1, 1.0, 0.0, 1.0, 0.147_000_000_000_000_02],
        &r,
    );
    assert_adjusted(
        &benjamini_hochberg(&P7, 0.05).unwrap(),
        &[
            0.0035,
            0.035,
            0.035,
            0.583_333_333_333_333_4,
            0.0,
            1.0,
            0.0686,
        ],
        &[true, true, true, false, true, false, false],
    );
    assert_adjusted(
        &benjamini_yekutieli(&P7, 0.05).unwrap(),
        &[
            0.009_075,
            0.090_75,
            0.090_75,
            1.0,
            0.0,
            1.0,
            0.177_869_999_999_999_97,
        ],
        &r,
    );
    // alpha = 0.1: statsmodels holm reject [T, T, T, F, T, F, F]; fdr_bh reject [T, T, T, F, T, F, T]
    assert_eq!(
        holm(&P7, 0.1).unwrap().reject,
        vec![true, true, true, false, true, false, false]
    );
    assert_eq!(
        benjamini_hochberg(&P7, 0.1).unwrap().reject,
        vec![true, true, true, false, true, false, true]
    );
    // A single p-value is returned unchanged by every procedure.
    for f in [bonferroni, holm, benjamini_hochberg, benjamini_yekutieli] {
        let a = f(&[0.03], 0.05).unwrap();
        assert_eq!((a.p_adjusted, a.reject), (vec![0.03], vec![true]));
        is_invalid(f(&[], 0.05));
        is_invalid(f(&[0.5, 1.5], 0.05));
        is_invalid(f(&[0.5], 1.0));
        is_invalid(f(&[0.5, f64::NAN], 0.05));
    }
}

// ═══════════════════════════════════════════════════════════════════════
// hypothesis: resampling
// ═══════════════════════════════════════════════════════════════════════

fn mean_f64(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

#[test]
fn bootstrap_is_reproducible_and_basic_reflects_percentile() {
    let data = [3.1, 4.7, 2.2, 5.9, 4.4, 3.8, 6.1, 2.9, 5.2, 4.0];
    let a = bootstrap_ci(
        &data,
        mean_f64,
        999,
        0.9,
        &mut Rng::new(2024),
        BootstrapMethod::Percentile,
    )
    .unwrap();
    let b = bootstrap_ci(
        &data,
        mean_f64,
        999,
        0.9,
        &mut Rng::new(2024),
        BootstrapMethod::Percentile,
    )
    .unwrap();
    assert_eq!(a, b);
    let c = bootstrap_ci(
        &data,
        mean_f64,
        999,
        0.9,
        &mut Rng::new(2025),
        BootstrapMethod::Percentile,
    )
    .unwrap();
    assert_ne!(a, c);
    // Basic = (2θ̂ − q_hi, 2θ̂ − q_lo) from the same resamples.
    let basic = bootstrap_ci(
        &data,
        mean_f64,
        999,
        0.9,
        &mut Rng::new(2024),
        BootstrapMethod::Basic,
    )
    .unwrap();
    let theta = mean_f64(&data);
    close(basic.lower + a.upper, 2.0 * theta, 1e-12);
    close(basic.upper + a.lower, 2.0 * theta, 1e-12);
    assert!(a.lower < theta && theta < a.upper);
    // Degenerate but valid: one observation, one resample.
    let one = bootstrap_ci(
        &[3.0],
        mean_f64,
        5,
        0.95,
        &mut Rng::new(1),
        BootstrapMethod::Percentile,
    )
    .unwrap();
    assert_eq!((one.lower, one.upper), (3.0, 3.0));
    let single = bootstrap_ci(
        &data,
        mean_f64,
        1,
        0.95,
        &mut Rng::new(1),
        BootstrapMethod::Percentile,
    )
    .unwrap();
    assert_eq!(single.lower, single.upper);
    // Validation: no resamples, confidence boundary, non-finite data / statistic.
    is_invalid(bootstrap_ci(
        &data,
        mean_f64,
        0,
        0.95,
        &mut Rng::new(1),
        BootstrapMethod::Percentile,
    ));
    is_invalid(bootstrap_ci(
        &data,
        mean_f64,
        10,
        1.0,
        &mut Rng::new(1),
        BootstrapMethod::Percentile,
    ));
    is_invalid(bootstrap_ci(
        &[1.0, f64::INFINITY],
        mean_f64,
        10,
        0.95,
        &mut Rng::new(1),
        BootstrapMethod::Basic,
    ));
    assert!(matches!(
        bootstrap_ci(
            &data,
            |_| f64::NAN,
            10,
            0.95,
            &mut Rng::new(1),
            BootstrapMethod::Basic
        ),
        Err(SymplexError::ComputationFailed { .. })
    ));
}

#[test]
fn permutation_test_tails_are_complementary_and_reproducible() {
    let x = [2.1, 3.4, 1.9, 2.8, 3.0];
    let y = [3.9, 4.2, 3.1, 4.8, 3.6, 4.4];
    let diff = |a: &[f64], b: &[f64]| mean_f64(a) - mean_f64(b);
    let g = permutation_test(&x, &y, diff, 1999, &mut Rng::new(11), Alternative::Greater).unwrap();
    let l = permutation_test(&x, &y, diff, 1999, &mut Rng::new(11), Alternative::Less).unwrap();
    let t = permutation_test(&x, &y, diff, 1999, &mut Rng::new(11), Alternative::TwoSided).unwrap();
    assert_eq!(g.statistic, diff(&x, &y));
    // Every permuted statistic is ≥ or ≤ the observed one, so the counts sum to at least N.
    assert!(g.p_value + l.p_value >= 1.0 + 1.0 / 2000.0 - 1e-15);
    close(t.p_value, (2.0 * g.p_value.min(l.p_value)).min(1.0), 1e-15);
    assert!(l.p_value < 0.01, "shift should be detected: {}", l.p_value);
    assert_eq!(
        permutation_test(&x, &y, diff, 1999, &mut Rng::new(11), Alternative::Less).unwrap(),
        l
    );
    // The smallest attainable p-value is 1/(N + 1).
    close(l.p_value * 2000.0, (l.p_value * 2000.0).round(), 1e-9);
    // One permutation, one observation each: the statistic can only be ±1.
    let r = permutation_test(
        &[1.0],
        &[2.0],
        diff,
        1,
        &mut Rng::new(1),
        Alternative::TwoSided,
    )
    .unwrap();
    assert_eq!(r.statistic, -1.0);
    assert_eq!(r.p_value, 1.0);
    is_invalid(permutation_test(
        &x,
        &y,
        diff,
        0,
        &mut Rng::new(1),
        Alternative::TwoSided,
    ));
    is_invalid(permutation_test(
        &[],
        &y,
        diff,
        10,
        &mut Rng::new(1),
        Alternative::TwoSided,
    ));
    assert!(matches!(
        permutation_test(
            &x,
            &y,
            |_, _| f64::INFINITY,
            10,
            &mut Rng::new(1),
            Alternative::TwoSided
        ),
        Err(SymplexError::ComputationFailed { .. })
    ));
}

// ═══════════════════════════════════════════════════════════════════════
// hypothesis: power and sample size
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn sample_size_for_proportion_new_margins_and_levels() {
    // statsmodels: samplesize_confint_proportion(0.2, 0.05, alpha=0.10) = 173.1547810621066 → 174
    assert_eq!(sample_size_for_proportion(0.05, 0.90, 0.2).unwrap(), 174);
    // statsmodels: samplesize_confint_proportion(0.5, 0.01, alpha=0.01) = 16587.24150255304 → 16588
    assert_eq!(sample_size_for_proportion(0.01, 0.99, 0.5).unwrap(), 16_588);
    is_invalid(sample_size_for_proportion(0.0, 0.95, 0.5));
    is_invalid(sample_size_for_proportion(0.05, 0.95, 1.0));
    is_invalid(sample_size_for_proportion(0.05, 1.0, 0.5));
    is_invalid(sample_size_for_proportion(f64::NAN, 0.95, 0.5));
}

#[test]
fn power_and_sample_size_two_proportions_new_effect() {
    // statsmodels: NormalIndPower().power(proportion_effectsize(0.3, 0.45), nobs1=80, alpha=0.01, ratio=1) = 0.2720335190509904
    close(
        power_two_proportions(0.3, 0.45, 80, 0.01).unwrap(),
        0.272_033_519_050_990_4,
        1e-9,
    );
    // nobs1=1, alpha=0.05 → 0.05557071985895796
    close(
        power_two_proportions(0.3, 0.45, 1, 0.05).unwrap(),
        0.055_570_719_858_957_96,
        1e-9,
    );
    // Symmetric in (p1, p2); equal proportions give power = alpha.
    close(
        power_two_proportions(0.45, 0.3, 80, 0.01).unwrap(),
        power_two_proportions(0.3, 0.45, 80, 0.01).unwrap(),
        1e-15,
    );
    close(
        power_two_proportions(0.4, 0.4, 50, 0.05).unwrap(),
        0.05,
        1e-12,
    );
    // statsmodels: NormalIndPower().solve_power(proportion_effectsize(0.3, 0.45), alpha=0.01, power=0.9, ratio=1)
    //   = 306.98623627325765 → 307
    assert_eq!(
        sample_size_two_proportions(0.3, 0.45, 0.01, 0.9).unwrap(),
        307
    );
    // A tiny effect needs a large n (statsmodels: solve_power(proportion_effectsize(0.5, 0.51), 0.05, 0.8) = 39239.07 → 39240).
    assert_eq!(
        sample_size_two_proportions(0.5, 0.51, 0.05, 0.8).unwrap(),
        39_240
    );
    is_invalid(power_two_proportions(0.3, 0.45, 0, 0.05));
    is_invalid(power_two_proportions(0.0, 0.45, 10, 0.05));
    is_invalid(sample_size_two_proportions(0.4, 0.4, 0.05, 0.8));
    is_invalid(sample_size_two_proportions(0.3, 0.45, 0.5, 0.4));
}

#[test]
fn power_and_sample_size_t_test_new_effect_and_edges() {
    // statsmodels: TTestIndPower().power(0.8, nobs1=12, alpha=0.05, ratio=1) = 0.46575928830209745
    close(
        power_t_test_two_sample(0.8, 12, 0.05).unwrap(),
        0.465_759_288_302_097_45,
        1e-8,
    );
    // The sign of the effect does not matter: power(-0.8, 12, 0.05) = 0.46575928830209745
    close(
        power_t_test_two_sample(-0.8, 12, 0.05).unwrap(),
        0.465_759_288_302_097_45,
        1e-8,
    );
    // n = 2 per group (df = 2): TTestIndPower().power(1.5, nobs1=2, alpha=0.05, ratio=1) = 0.148691579150247
    close(
        power_t_test_two_sample(1.5, 2, 0.05).unwrap(),
        0.148_691_579_150_247,
        1e-8,
    );
    // d = 0: power equals alpha (0.05000000000000001).
    close(power_t_test_two_sample(0.0, 20, 0.05).unwrap(), 0.05, 1e-8);
    // statsmodels: TTestIndPower().solve_power(0.8, alpha=0.05, power=0.9, ratio=1) = 33.82554234416029 → 34
    assert_eq!(sample_size_t_test_two_sample(0.8, 0.05, 0.9).unwrap(), 34);
    is_invalid(power_t_test_two_sample(0.8, 1, 0.05));
    is_invalid(power_t_test_two_sample(f64::INFINITY, 10, 0.05));
    is_invalid(sample_size_t_test_two_sample(0.0, 0.05, 0.8));
    is_invalid(sample_size_t_test_two_sample(0.5, 0.05, 0.05));
}

// ═══════════════════════════════════════════════════════════════════════
// agreement
// ═══════════════════════════════════════════════════════════════════════

fn ra_rb() -> (Vec<Q>, Vec<Q>) {
    (
        from_i64(&[
            0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 0,
        ]),
        from_i64(&[
            0, 1, 2, 2, 0, 1, 1, 2, 0, 2, 2, 2, 0, 1, 1, 1, 2, 0, 2, 1, 0, 1, 2, 0,
        ]),
    )
}

/// D5: 5 raters × 10 items, categories 0..2, as items × raters.
fn d5() -> RatingTable {
    RatingTable::from_i64(&[
        &[0, 0, 0, 0, 0],
        &[0, 0, 1, 1, 1],
        &[1, 2, 2, 2, 2],
        &[0, 1, 2, 2, 2],
        &[0, 0, 0, 0, 1],
        &[1, 1, 1, 1, 1],
        &[0, 0, 0, 1, 1],
        &[2, 2, 2, 2, 2],
        &[0, 0, 1, 1, 2],
        &[0, 1, 1, 1, 1],
    ])
    .unwrap()
}

fn d5_counts() -> Vec<Vec<usize>> {
    vec![
        vec![5, 0, 0],
        vec![2, 3, 0],
        vec![0, 1, 4],
        vec![1, 1, 3],
        vec![4, 1, 0],
        vec![0, 5, 0],
        vec![3, 2, 0],
        vec![0, 0, 5],
        vec![2, 2, 1],
        vec![1, 4, 0],
    ]
}

/// FK: 3 raters × 9 items (columns are raters), values 1/2, 1, 2, 3, 4, missing cells.
fn fk() -> RatingTable {
    let h = Some(q(1, 2));
    let r = |v: i64| Some(qi(v));
    RatingTable::new(vec![
        vec![r(1), r(1), r(2)],
        vec![r(2), r(2), None],
        vec![r(3), r(3), r(3)],
        vec![h.clone(), h.clone(), h],
        vec![None, r(2), r(2)],
        vec![r(2), r(2), r(3)],
        vec![r(4), r(4), r(4)],
        vec![r(3), r(2), r(3)],
        vec![r(1), None, None],
    ])
    .unwrap()
}

#[test]
fn cohen_kappa_four_categories_one_unused_by_a_rater() {
    let (a, b) = ra_rb();
    // Confusion matrix [[6, 1, 0, 0], [0, 5, 1, 0], [1, 1, 4, 0], [0, 1, 4, 0]].
    // statsmodels: cohens_kappa(table).kappa = 0.49176470588235294 (= 209/425); p_o = 5/8, p_e = 151/576
    let k = cohen_kappa(&a, &b).unwrap();
    assert_eq!(k.kappa, q(209, 425));
    assert_eq!(k.observed, q(5, 8));
    assert_eq!(k.expected, q(151, 576));
    assert_eq!(percent_agreement(&a, &b).unwrap(), k.observed);
    let table = confusion_matrix(&a, &b, &from_i64(&[0, 1, 2, 3])).unwrap();
    assert_eq!(
        table,
        vec![
            vec![6, 1, 0, 0],
            vec![0, 5, 1, 0],
            vec![1, 1, 4, 0],
            vec![0, 1, 4, 0]
        ]
    );
    assert_eq!(
        kappa_from_confusion(&table, &Weights::Unweighted).unwrap(),
        k
    );
    // statsmodels: cohens_kappa(table, wt='linear').kappa = 0.5875 (= 47/80), observed 61/72, expected 17/27;
    //   wt='quadratic' → 0.6842105263157895 (= 13/19), observed 67/72, expected 337/432
    let lin = weighted_kappa(&a, &b, &Weights::Linear).unwrap();
    assert_eq!(
        (lin.kappa, lin.observed, lin.expected),
        (q(47, 80), q(61, 72), q(17, 27))
    );
    let quad = weighted_kappa(&a, &b, &Weights::Quadratic).unwrap();
    assert_eq!(
        (quad.kappa, quad.observed, quad.expected),
        (q(13, 19), q(67, 72), q(337, 432))
    );
    // statsmodels: cohens_kappa(table, weights=[[0,1,2,5],[1,0,1,2],[2,1,0,1],[5,2,1,0]]).kappa = 0.6281690140845071 (= 223/355)
    let w = Weights::Custom(counts(&[
        &[0, 1, 2, 5],
        &[1, 0, 1, 2],
        &[2, 1, 0, 1],
        &[5, 2, 1, 0],
    ]));
    assert_eq!(weighted_kappa(&a, &b, &w).unwrap().kappa, q(223, 355));
    // Wrong-sized custom weights, and Fleiss/Scott for two raters:
    is_invalid(weighted_kappa(
        &a,
        &b,
        &Weights::Custom(counts(&[&[0, 1], &[1, 0]])),
    ));
    // statsmodels: fleiss_kappa(aggregate_raters([a, b].T)[0]) = 0.48014440433212996 (= 133/277)
    assert_eq!(scott_pi(&a, &b).unwrap(), q(133, 277));
    let t = RatingTable::new(
        a.iter()
            .zip(&b)
            .map(|(x, y)| vec![Some(x.clone()), Some(y.clone())])
            .collect(),
    )
    .unwrap();
    assert_eq!(fleiss_kappa_ratings(&t).unwrap(), q(133, 277));
    // krippendorff.alpha(reliability_data=[a, b], level_of_measurement=…): nominal 0.49097472924187713 (= 136/277),
    //   interval 0.6839982070820261 (= 1526/2231), ordinal 0.719098165426558 (= 292253/406416)
    assert_eq!(krippendorff_alpha(&t, Level::Nominal).unwrap(), q(136, 277));
    assert_eq!(
        krippendorff_alpha(&t, Level::Interval).unwrap(),
        q(1_526, 2_231)
    );
    assert_eq!(
        krippendorff_alpha(&t, Level::Ordinal).unwrap(),
        q(292_253, 406_416)
    );
}

#[test]
fn agreement_with_a_constant_rater_and_perfect_agreement() {
    // Rater B always says 0: confusion [[3, 0], [3, 0]].
    // statsmodels: cohens_kappa([[3, 0], [3, 0]]).kappa = 0.0; fleiss_kappa (Scott's π) = -0.3333333333333333;
    // krippendorff.alpha([[0,1,0,1,0,1], [0]*6], 'nominal') = -0.2222222222222221 (= −2/9)
    let a = from_i64(&[0, 1, 0, 1, 0, 1]);
    let b = from_i64(&[0, 0, 0, 0, 0, 0]);
    let k = cohen_kappa(&a, &b).unwrap();
    assert_eq!((k.kappa, k.observed, k.expected), (qi(0), q(1, 2), q(1, 2)));
    assert_eq!(scott_pi(&a, &b).unwrap(), q(-1, 3));
    let t = raters(&[
        &[Some(0), Some(1), Some(0), Some(1), Some(0), Some(1)],
        &[Some(0); 6],
    ]);
    assert_eq!(krippendorff_alpha(&t, Level::Nominal).unwrap(), q(-2, 9));
    // Perfect agreement: κ = α = 1 (statsmodels cohens_kappa([[3, 0], [0, 4]]).kappa = 1.0; krippendorff 1.0).
    let a = from_i64(&[0, 1, 0, 1, 1, 0, 1]);
    assert_eq!(cohen_kappa(&a, &a).unwrap().kappa, qi(1));
    let t = RatingTable::new(
        a.iter()
            .map(|x| vec![Some(x.clone()), Some(x.clone())])
            .collect(),
    )
    .unwrap();
    assert_eq!(krippendorff_alpha(&t, Level::Nominal).unwrap(), qi(1));
    assert_eq!(krippendorff_alpha(&t, Level::Interval).unwrap(), qi(1));
    assert_eq!(fleiss_kappa_ratings(&t).unwrap(), qi(1));
    assert_eq!(gwet_ac1(&t, None).unwrap(), qi(1));
    // Both raters constant on the same value: every coefficient is undefined.
    let c = from_i64(&[2, 2, 2]);
    is_invalid(cohen_kappa(&c, &c));
    is_invalid(scott_pi(&c, &c));
    is_invalid(kappa_from_confusion(&[vec![5]], &Weights::Unweighted));
    is_invalid(weighted_kappa(&c, &c, &Weights::Linear));
    is_invalid(krippendorff_alpha(
        &RatingTable::from_i64(&[&[2, 2], &[2, 2]]).unwrap(),
        Level::Nominal,
    ));
    is_invalid(percent_agreement(&c, &c[..2]));
    is_invalid(percent_agreement(&[], &[]));
}

#[test]
fn fleiss_kappa_five_raters_ten_items_and_relatives() {
    // statsmodels: fleiss_kappa(D5 counts) = 0.4076549210206561 (= 671/1646)
    assert_eq!(fleiss_kappa(&d5_counts()).unwrap(), q(671, 1_646));
    assert_eq!(fleiss_kappa_ratings(&d5()).unwrap(), q(671, 1_646));
    assert_eq!(d5().count_table(&d5().categories()).unwrap(), d5_counts());
    // Krippendorff on the same complete data (krippendorff.alpha): nominal 0.41950182260024305 (= 1381/3292),
    //   ordinal 0.5511314843068875 (= 2022873/3670400), interval 0.5662295081967212 (= 1727/3050)
    assert_eq!(
        krippendorff_alpha(&d5(), Level::Nominal).unwrap(),
        q(1_381, 3_292)
    );
    assert_eq!(
        krippendorff_alpha(&d5(), Level::Ordinal).unwrap(),
        q(2_022_873, 3_670_400)
    );
    assert_eq!(
        krippendorff_alpha(&d5(), Level::Interval).unwrap(),
        q(1_727, 3_050)
    );
    // Fraction (Gwet 2008): p_a = mean Σ r_ik(r_ik−1)/20, π_k = mean r_ik/5, p_e = Σ π(1−π)/2 → AC₁ = 18/43
    assert_eq!(gwet_ac1(&d5(), None).unwrap(), q(18, 43));
    // Fraction: mean pairwise percent agreement over the 10 rater pairs = 61/100
    assert_eq!(pairwise_percent_agreement(&d5()).unwrap(), q(61, 100));
    // A single item: statsmodels: fleiss_kappa([[2, 1]]) = -0.5000000000000001
    assert_eq!(fleiss_kappa(&[vec![2, 1]]).unwrap(), q(-1, 2));
    is_invalid(fleiss_kappa(&[vec![2, 1], vec![1, 1]]));
    is_invalid(fleiss_kappa(&[vec![1, 0], vec![0, 1]]));
    is_invalid(fleiss_kappa(&[vec![3, 0], vec![3, 0]]));
    is_invalid(fleiss_kappa(&[]));
    is_invalid(fleiss_kappa_ratings(&fk()));
}

#[test]
fn krippendorff_alpha_missing_cells_rational_values_every_level() {
    // krippendorff.alpha(reliability_data=FK (columns), level_of_measurement=…): nominal 0.6519337016574586 (= 118/181),
    //   ordinal 0.8570796215348863 (= 15490/18073), interval 0.8871473354231975 (= 283/319),
    //   ratio 0.8879788753076042 (= 350741/394988); item 9 has one rating and drops out.
    let t = fk();
    assert!(!t.is_complete());
    assert_eq!(krippendorff_alpha(&t, Level::Nominal).unwrap(), q(118, 181));
    assert_eq!(
        krippendorff_alpha(&t, Level::Ordinal).unwrap(),
        q(15_490, 18_073)
    );
    assert_eq!(
        krippendorff_alpha(&t, Level::Interval).unwrap(),
        q(283, 319)
    );
    assert_eq!(
        krippendorff_alpha(&t, Level::Ratio).unwrap(),
        q(350_741, 394_988)
    );
    // Replacing 4 by −1 (a negative value) leaves nominal / ordinal unchanged and changes interval:
    // krippendorff.alpha(EK, 'interval') = 0.9239360096589194 (= 3061/3313)
    let mut rows = t.rows().to_vec();
    rows[6] = vec![Some(qi(-1)); 3];
    let e = RatingTable::new(rows).unwrap();
    assert_eq!(krippendorff_alpha(&e, Level::Nominal).unwrap(), q(118, 181));
    assert_eq!(
        krippendorff_alpha(&e, Level::Ordinal).unwrap(),
        q(15_490, 18_073)
    );
    assert_eq!(
        krippendorff_alpha(&e, Level::Interval).unwrap(),
        q(3_061, 3_313)
    );
    // Fraction (Gwet with missing cells, K = 5 observed categories): AC₁ = 1637/2366; with a sixth
    // unobserved category passed explicitly: 1747/2476.
    assert_eq!(gwet_ac1(&t, None).unwrap(), q(1_637, 2_366));
    let six = [q(1, 2), qi(1), qi(2), qi(3), qi(4), qi(7)];
    assert_eq!(gwet_ac1(&t, Some(&six)).unwrap(), q(1_747, 2_476));
    is_invalid(gwet_ac1(&t, Some(&[qi(1), qi(2)])));
    is_invalid(gwet_ac1(
        &t,
        Some(&[qi(1), qi(1), qi(2), qi(3), qi(4), q(1, 2)]),
    ));
    // Degenerate patterns: only single ratings; two values but only one pairable.
    is_invalid(krippendorff_alpha(
        &RatingTable::from_i64_missing(&[&[Some(1), None], &[None, Some(2)]]).unwrap(),
        Level::Nominal,
    ));
    is_invalid(krippendorff_alpha(
        &RatingTable::from_i64_missing(&[&[Some(1), Some(1)], &[Some(2), None]]).unwrap(),
        Level::Nominal,
    ));
    is_invalid(gwet_ac1(
        &RatingTable::from_i64_missing(&[&[Some(1), None], &[None, Some(2)]]).unwrap(),
        None,
    ));
    // One item, two raters disagreeing: α = 0 (D_o = D_e); two items with swapped values: −1/2.
    assert_eq!(
        krippendorff_alpha(&RatingTable::from_i64(&[&[1, 2]]).unwrap(), Level::Nominal).unwrap(),
        qi(0)
    );
    assert_eq!(
        krippendorff_alpha(
            &RatingTable::from_i64(&[&[1, 2], &[2, 1]]).unwrap(),
            Level::Nominal
        )
        .unwrap(),
        q(-1, 2)
    );
}

#[test]
fn icc_six_forms_five_targets_three_judges_match_pingouin() {
    let t = RatingTable::from_i64(&[&[4, 5, 6], &[7, 6, 9], &[2, 3, 3], &[8, 9, 7], &[5, 5, 8]])
        .unwrap();
    // Fraction: MSR = 199/15, MSC = 13/5, MSE = 19/15, MSW = 23/15.
    let a = icc_anova(&t).unwrap();
    assert_eq!(
        (a.msr, a.msc, a.mse, a.msw),
        (q(199, 15), q(13, 5), q(19, 15), q(23, 15))
    );
    // pingouin: intraclass_corr(targets, raters, ratings).ICC: ICC(1,1) 0.7183673469387755 (= 176/245),
    //   ICC(A,1) 0.7228915662650603 (= 60/83), ICC(C,1) 0.759493670886076 (= 60/79),
    //   ICC(1,k) 0.8844221105527639 (= 176/199), ICC(A,k) 0.8866995073891627 (= 180/203), ICC(C,k) 0.9045226130653267 (= 180/199)
    assert_eq!(icc(&t, IccForm::Icc1).unwrap(), q(176, 245));
    assert_eq!(icc(&t, IccForm::Icc2Single).unwrap(), q(60, 83));
    assert_eq!(icc(&t, IccForm::Icc3Single).unwrap(), q(60, 79));
    assert_eq!(icc(&t, IccForm::Icc1Average).unwrap(), q(176, 199));
    assert_eq!(icc(&t, IccForm::Icc2Average).unwrap(), q(180, 203));
    assert_eq!(icc(&t, IccForm::Icc3Average).unwrap(), q(180, 199));
    // Kendall's W of the same table (raters rank the 5 items; judge 2 ties two items):
    // scipy: friedmanchisquare(*rows).statistic / (3·4) = 0.8418079096045196 (= 149/177)
    assert_eq!(kendall_w(&t).unwrap(), q(149, 177));
    // A judge who gives every target the same score: pingouin ICC(1,1) 0.28947368421052616 (= 11/38),
    //   ICC(A,1) 0.357142857142857 (= 5/14), ICC(C,1) 0.4999999999999997 (= 1/2),
    //   ICC(1,k) 0.5499999999999998 (= 11/20), ICC(A,k) 0.6249999999999999 (= 5/8), ICC(C,k) 0.7499999999999999 (= 3/4)
    let c = RatingTable::from_i64(&[&[1, 2, 2], &[3, 2, 4], &[5, 2, 6], &[7, 2, 8]]).unwrap();
    assert_eq!(icc(&c, IccForm::Icc1).unwrap(), q(11, 38));
    assert_eq!(icc(&c, IccForm::Icc2Single).unwrap(), q(5, 14));
    assert_eq!(icc(&c, IccForm::Icc3Single).unwrap(), q(1, 2));
    assert_eq!(icc(&c, IccForm::Icc1Average).unwrap(), q(11, 20));
    assert_eq!(icc(&c, IccForm::Icc2Average).unwrap(), q(5, 8));
    assert_eq!(icc(&c, IccForm::Icc3Average).unwrap(), q(3, 4));
    // Identical items: MSR = 0 → every average form is undefined; incomplete tables are rejected.
    let flat = RatingTable::from_i64(&[&[1, 3], &[1, 3]]).unwrap();
    is_invalid(icc(&flat, IccForm::Icc3Average));
    is_invalid(icc(&flat, IccForm::Icc1Average));
    is_invalid(icc(&fk(), IccForm::Icc1));
    is_invalid(kendall_w(&fk()));
    is_invalid(kendall_w(
        &RatingTable::from_i64(&[&[1, 1], &[1, 1]]).unwrap(),
    ));
    // Two raters ranking in exact reverse: W = 0.
    assert_eq!(
        kendall_w(&RatingTable::from_i64(&[&[1, 4], &[2, 3], &[3, 2], &[4, 1]]).unwrap()).unwrap(),
        qi(0)
    );
}

#[test]
fn rating_table_queries_and_validation() {
    let t = fk();
    assert_eq!((t.n_items(), t.n_raters()), (9, 3));
    assert_eq!(t.categories(), vec![q(1, 2), qi(1), qi(2), qi(3), qi(4)]);
    assert_eq!(t.get(3, 0), Some(&q(1, 2)));
    assert_eq!(t.get(1, 2), None);
    assert_eq!(t.get(9, 0), None);
    assert_eq!(t.item(9), None);
    assert_eq!(t.rater(3), None);
    assert_eq!(t.rater(2).unwrap()[1], None);
    // Raters 1 and 2 share the items 0, 2, 3, 4, 5, 6, 7 (item 1: rater 2 missing; item 8: both missing)
    // and agree on 2, 3, 4, 6.
    let (a, b) = t.paired_ratings(1, 2).unwrap();
    assert_eq!(a.len(), 7);
    assert_eq!(percent_agreement(&a, &b).unwrap(), q(4, 7));
    is_invalid(t.paired_ratings(0, 3));
    is_invalid(t.complete_rows());
    is_invalid(t.count_table(&[qi(1), qi(2)]));
    is_invalid(t.count_table(&[qi(1), qi(1)]));
    assert_eq!(
        category_frequencies(&t),
        vec![(q(1, 2), 3), (qi(1), 3), (qi(2), 8), (qi(3), 6), (qi(4), 3)]
    );
    // Constructors reject empty / ragged input.
    is_invalid(RatingTable::new(vec![]));
    is_invalid(RatingTable::from_i64(&[&[]]));
    is_invalid(RatingTable::from_i64(&[&[1, 2], &[1]]));
    is_invalid(RatingTable::from_raters_i64(&[]));
    is_invalid(RatingTable::from_raters_i64(&[&[]]));
    is_invalid(RatingTable::from_raters_i64(&[
        &[Some(1)],
        &[Some(1), Some(2)],
    ]));
    is_invalid(pairwise_percent_agreement(
        &RatingTable::from_i64(&[&[1], &[2]]).unwrap(),
    ));
    is_invalid(pairwise_percent_agreement(
        &RatingTable::from_i64_missing(&[&[Some(1), None], &[None, Some(2)]]).unwrap(),
    ));
    is_invalid(confusion_matrix(
        &from_i64(&[1, 2]),
        &from_i64(&[1, 3]),
        &from_i64(&[1, 2]),
    ));
}

// ═══════════════════════════════════════════════════════════════════════
// aggregation
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn votes_edge_cases_empty_all_missing_ties_and_thresholds() {
    let v = majority_vote(&[]);
    assert_eq!((v.winner, v.tied, v.counts), (None, vec![], vec![]));
    let v = majority_vote(&[None, None]);
    assert_eq!((v.winner, v.tied, v.counts), (None, vec![], vec![]));
    let v = majority_vote(&[Some(3)]);
    assert_eq!(
        (v.winner, v.tied, v.counts),
        (Some(3), vec![3], vec![0, 0, 0, 1])
    );
    let three_way = [Some(0), Some(1), Some(2), None];
    assert_eq!(majority_vote(&three_way).tied, vec![0, 1, 2]);
    // Plurality: threshold 0 always accepts a unique top; threshold 1 needs unanimity.
    let labels = [Some(0), Some(0), Some(0), Some(1)];
    assert_eq!(plurality(&labels, &qi(0)).unwrap().winner, Some(0));
    assert_eq!(plurality(&labels, &q(3, 4)).unwrap().winner, Some(0));
    assert_eq!(plurality(&labels, &qi(1)).unwrap().winner, None);
    assert_eq!(
        plurality(&[Some(2), Some(2)], &qi(1)).unwrap().winner,
        Some(2)
    );
    is_invalid(plurality(&labels, &q(5, 4)));
    is_invalid(plurality(&labels, &q(-1, 4)));
    // Weighted: zero total weight → no winner; exact rational tie → no winner.
    let v = weighted_vote(&[Some(0), Some(1)], &[qi(0), qi(0)]).unwrap();
    assert_eq!((v.winner, v.tied), (None, vec![]));
    let v = weighted_vote(&[Some(0), Some(1), Some(1)], &[q(2, 3), q(1, 3), q(1, 3)]).unwrap();
    assert_eq!(
        (v.winner, v.tied, v.scores),
        (None, vec![0, 1], vec![q(2, 3), q(2, 3)])
    );
    let v = weighted_vote(&[Some(0), None, Some(2)], &[q(1, 2), qi(9), q(3, 4)]).unwrap();
    assert_eq!(
        (v.winner, v.scores),
        (Some(2), vec![q(1, 2), qi(0), q(3, 4)])
    );
    is_invalid(weighted_vote(&[Some(0), Some(1)], &[qi(1)]));
    is_invalid(weighted_vote(&[Some(0), Some(1)], &[qi(1), qi(-1)]));
    // Label tables: counts always run to n_categories.
    let t = LabelTable::from_rows(&[&[None, None, None], &[Some(2), Some(2), Some(0)]], 4).unwrap();
    let v = majority_votes(&t);
    assert_eq!((v[0].winner, v[0].counts.clone()), (None, vec![0, 0, 0, 0]));
    assert_eq!(
        (v[1].winner, v[1].counts.clone()),
        (Some(2), vec![1, 0, 2, 0])
    );
    is_invalid(LabelTable::from_rows(&[&[Some(0)]], 0));
    is_invalid(LabelTable::from_rows(&[], 2));
    is_invalid(LabelTable::from_rows(&[&[]], 2));
    is_invalid(LabelTable::from_rows(&[&[Some(0), Some(1)], &[Some(0)]], 2));
    is_invalid(LabelTable::complete(&[&[0, 2]], 2));
}

fn ds_rows() -> Vec<Vec<Option<usize>>> {
    vec![
        vec![Some(0), Some(0), Some(1)],
        vec![Some(1), Some(1), None],
        vec![Some(2), Some(2), Some(2)],
        vec![Some(0), Some(1), Some(0)],
        vec![None, Some(2), Some(2)],
        vec![Some(1), Some(0), Some(1)],
        vec![Some(0), Some(0), Some(0)],
        vec![Some(2), Some(1), Some(2)],
    ]
}

/// The M-step of the documentation, re-implemented independently.
fn ds_m_step(
    counts: &[Vec<Vec<usize>>],
    t: &[Vec<f64>],
    j: usize,
    smoothing: f64,
) -> (Vec<f64>, Vec<Vec<Vec<f64>>>) {
    let n_items = counts.len() as f64;
    let priors: Vec<f64> = (0..j)
        .map(|c| t.iter().map(|r| r[c]).sum::<f64>() / n_items)
        .collect();
    let confusion = (0..counts[0].len())
        .map(|k| {
            (0..j)
                .map(|c| {
                    let num: Vec<f64> = (0..j)
                        .map(|l| {
                            counts
                                .iter()
                                .zip(t)
                                .map(|(item, r)| r[c] * item[k][l] as f64)
                                .sum::<f64>()
                                + smoothing
                        })
                        .collect();
                    let den: f64 = num.iter().sum();
                    if den > 0.0 {
                        num.iter().map(|v| v / den).collect()
                    } else {
                        vec![1.0 / j as f64; j]
                    }
                })
                .collect()
        })
        .collect();
    (priors, confusion)
}

/// The E-step of the documentation, re-implemented independently.
fn ds_e_step(
    counts: &[Vec<Vec<usize>>],
    priors: &[f64],
    confusion: &[Vec<Vec<f64>>],
) -> Vec<Vec<f64>> {
    counts
        .iter()
        .map(|item| {
            let mut w: Vec<f64> = priors.to_vec();
            for (k, labels) in item.iter().enumerate() {
                for (l, &c) in labels.iter().enumerate() {
                    for (j, v) in w.iter_mut().enumerate() {
                        *v *= confusion[k][j][l].powi(c as i32);
                    }
                }
            }
            let s: f64 = w.iter().sum();
            w.iter().map(|v| v / s).collect()
        })
        .collect()
}

#[test]
fn dawid_skene_with_missing_labels_is_a_fixed_point_and_matches_a_python_em() {
    let t = LabelTable::new(ds_rows(), 3).unwrap();
    let counts = t.to_counts();
    // Python EM (same rules: majority-vote init, tol 1e-10 on the posteriors), smoothing 0:
    //   converged after 1 iteration, labels [0, 1, 2, 0, 2, 1, 0, 2], priors [0.375, 0.25, 0.375],
    //   log-likelihood -15.772486116083346
    let ds = dawid_skene(&t, &DawidSkeneOpts::default()).unwrap();
    assert!(ds.converged);
    assert_eq!(ds.iterations, 1);
    assert_eq!(ds.labels(), vec![0, 1, 2, 0, 2, 1, 0, 2]);
    for (p, e) in ds.priors.iter().zip([0.375, 0.25, 0.375]) {
        close(*p, e, 1e-12);
    }
    close(ds.log_likelihood, -15.772_486_116_083_346, 1e-9);
    // Fixed point: M-step(posteriors) reproduces the returned parameters and E-step reproduces the posteriors.
    let (pri, conf) = ds_m_step(&counts, &ds.posteriors, 3, 0.0);
    for (a, b) in pri.iter().zip(&ds.priors) {
        close(*a, *b, 1e-14);
    }
    for (k, mk) in conf.iter().enumerate() {
        for (j, row) in mk.iter().enumerate() {
            for (l, v) in row.iter().enumerate() {
                close(*v, ds.confusion[k][j][l], 1e-14);
            }
        }
    }
    let post = ds_e_step(&counts, &ds.priors, &ds.confusion);
    for (a, b) in post.iter().flatten().zip(ds.posteriors.iter().flatten()) {
        close(*a, *b, 1e-9);
    }
    // smoothing 0.5: Python EM converged after 52 iterations, labels [0, 0, 2, 0, 2, 0, 0, 2],
    //   priors [0.5782585385320876, 2.4395444050267883e-11, 0.42174146144351693], log-likelihood -19.681329620531834,
    //   posteriors[0] = [0.9902942703394358, 1.7918867250253823e-11, 0.009705729642645365]
    let opts = DawidSkeneOpts {
        smoothing: 0.5,
        ..DawidSkeneOpts::default()
    };
    let ds = dawid_skene(&t, &opts).unwrap();
    assert!(ds.converged);
    assert_eq!(ds.iterations, 52);
    assert_eq!(ds.labels(), vec![0, 0, 2, 0, 2, 0, 0, 2]);
    close(ds.priors[0], 0.578_258_538_532_087_6, 1e-8);
    close(ds.priors[2], 0.421_741_461_443_516_93, 1e-8);
    assert!(ds.priors[1] < 1e-9);
    close(ds.log_likelihood, -19.681_329_620_531_834, 1e-8);
    close(ds.posteriors[0][0], 0.990_294_270_339_435_8, 1e-8);
    close(ds.posteriors[0][2], 0.009_705_729_642_645_365, 1e-8);
    let (pri, conf) = ds_m_step(&counts, &ds.posteriors, 3, 0.5);
    for (a, b) in pri.iter().zip(&ds.priors) {
        close(*a, *b, 1e-14);
    }
    for (a, b) in conf
        .iter()
        .flatten()
        .flatten()
        .zip(ds.confusion.iter().flatten().flatten())
    {
        close(*a, *b, 1e-14);
    }
    let post = ds_e_step(&counts, &ds.priors, &ds.confusion);
    for (a, b) in post.iter().flatten().zip(ds.posteriors.iter().flatten()) {
        close(*a, *b, 1e-9);
    }
}

#[test]
fn dawid_skene_options_and_degenerate_tables() {
    let t = LabelTable::new(ds_rows(), 3).unwrap();
    // max_iter = 1 with a non-trivial start reports non-convergence honestly.
    let opts = DawidSkeneOpts {
        max_iter: 1,
        smoothing: 0.5,
        ..DawidSkeneOpts::default()
    };
    let ds = dawid_skene(&t, &opts).unwrap();
    assert!(!ds.converged);
    assert_eq!(ds.iterations, 1);
    // An explicit start that is the same for every item carries no information about which item
    // is which class: the M-step gives every rater identical confusion rows, and EM collapses
    // onto a single class (a sharp edge, not a bug — the start must break the symmetry).
    let flat = DawidSkeneInit::Posteriors(vec![vec![2.0, 1.0, 1.0]; 8]);
    let ds = dawid_skene(
        &t,
        &DawidSkeneOpts {
            init: flat,
            ..DawidSkeneOpts::default()
        },
    )
    .unwrap();
    assert!(ds.converged);
    assert_eq!(ds.labels(), vec![0; 8]);
    // Explicit posteriors are normalised (rows sum to 4 here) and a soft, informative start
    // reaches the same fixed point as the majority vote.
    let soft: Vec<Vec<f64>> = ds_rows()
        .iter()
        .map(|row| {
            let mut w = vec![1.0; 3];
            for l in row.iter().flatten() {
                w[*l] += 1.0;
            }
            w
        })
        .collect();
    let ds = dawid_skene(
        &t,
        &DawidSkeneOpts {
            init: DawidSkeneInit::Posteriors(soft),
            ..DawidSkeneOpts::default()
        },
    )
    .unwrap();
    assert!(ds.converged);
    assert_eq!(ds.labels(), vec![0, 1, 2, 0, 2, 1, 0, 2]);
    is_invalid(dawid_skene(
        &t,
        &DawidSkeneOpts {
            init: DawidSkeneInit::Posteriors(vec![vec![1.0; 3]; 7]),
            ..DawidSkeneOpts::default()
        },
    ));
    is_invalid(dawid_skene(
        &t,
        &DawidSkeneOpts {
            init: DawidSkeneInit::Posteriors(vec![vec![1.0, -1.0, 1.0]; 8]),
            ..DawidSkeneOpts::default()
        },
    ));
    is_invalid(dawid_skene(
        &t,
        &DawidSkeneOpts {
            init: DawidSkeneInit::Posteriors(vec![vec![0.0; 3]; 8]),
            ..DawidSkeneOpts::default()
        },
    ));
    is_invalid(dawid_skene(
        &t,
        &DawidSkeneOpts {
            max_iter: 0,
            ..DawidSkeneOpts::default()
        },
    ));
    is_invalid(dawid_skene(
        &t,
        &DawidSkeneOpts {
            tol: 0.0,
            ..DawidSkeneOpts::default()
        },
    ));
    is_invalid(dawid_skene(
        &t,
        &DawidSkeneOpts {
            smoothing: -0.1,
            ..DawidSkeneOpts::default()
        },
    ));
    is_invalid(dawid_skene(
        &t,
        &DawidSkeneOpts {
            smoothing: f64::NAN,
            ..DawidSkeneOpts::default()
        },
    ));
    // A rater who never labels gets uniform confusion rows; an item nobody labels keeps a uniform posterior.
    let t = LabelTable::from_rows(&[&[Some(0), None], &[Some(1), None], &[None, None]], 2).unwrap();
    let ds = dawid_skene(&t, &DawidSkeneOpts::default()).unwrap();
    for row in &ds.confusion[1] {
        assert_eq!(row, &vec![0.5, 0.5]);
    }
    close(ds.posteriors[2][0], 0.5, 1e-12);
    // Two categories minimum; ragged counts rejected.
    is_invalid(dawid_skene_counts(
        &[vec![vec![1]]],
        1,
        &DawidSkeneOpts::default(),
    ));
    is_invalid(dawid_skene_counts(
        &[vec![vec![1, 0]], vec![vec![1, 0], vec![0, 1]]],
        2,
        &DawidSkeneOpts::default(),
    ));
    is_invalid(dawid_skene_counts(&[], 2, &DawidSkeneOpts::default()));
}

#[test]
fn bradley_terry_five_players_matches_scipy_mle_and_the_stationary_condition() {
    // W = [[0,4,1,3,2],[2,0,3,1,1],[3,1,0,2,4],[1,2,1,0,3],[0,2,1,2,0]].
    // scipy: minimize(negative BT log-likelihood, method='BFGS', gtol=1e-13), normalised:
    //   [0.2955552194891264, 0.1644353110294351, 0.27758207284398334, 0.1591196264163072, 0.10330777022114797];
    // Hunter's MM iterated to 1e-16 (56 iterations):
    //   [0.29555519818046944, 0.16443531306965964, 0.2775820760387081, 0.15911963499925924, 0.10330777771190368]
    let w = vec![
        vec![0, 4, 1, 3, 2],
        vec![2, 0, 3, 1, 1],
        vec![3, 1, 0, 2, 4],
        vec![1, 2, 1, 0, 3],
        vec![0, 2, 1, 2, 0],
    ];
    let bt = bradley_terry(&w, &BradleyTerryOpts::default()).unwrap();
    assert!(bt.converged);
    let mm = [
        0.295_555_198_180_469_44,
        0.164_435_313_069_659_64,
        0.277_582_076_038_708_1,
        0.159_119_634_999_259_24,
        0.103_307_777_711_903_68,
    ];
    let mle = [
        0.295_555_219_489_126_4,
        0.164_435_311_029_435_1,
        0.277_582_072_843_983_34,
        0.159_119_626_416_307_2,
        0.103_307_770_221_147_97,
    ];
    for ((p, m), l) in bt.strengths.iter().zip(mm).zip(mle) {
        close(*p, m, 1e-9);
        close(*p, l, 1e-6);
    }
    close(bt.strengths.iter().sum::<f64>(), 1.0, 1e-12);
    // Stationarity: pᵢ = Wᵢ / Σⱼ Nᵢⱼ/(pᵢ + pⱼ).
    let n = w.len();
    for (i, row) in w.iter().enumerate() {
        let wins: f64 = row.iter().sum::<usize>() as f64;
        let denom: f64 = (0..n)
            .filter(|&j| j != i)
            .map(|j| (row[j] + w[j][i]) as f64 / (bt.strengths[i] + bt.strengths[j]))
            .sum();
        close(bt.strengths[i], wins / denom, 1e-9);
    }
    // Two players 7–3: P(A beats B) = 0.7 exactly at the MLE.
    let bt = bradley_terry(&[vec![0, 7], vec![3, 0]], &BradleyTerryOpts::default()).unwrap();
    close(bt.strengths[0], 0.7, 1e-12);
    close(bt.strengths[1], 0.3, 1e-12);
    // Validation: unbeaten player (not strongly connected), isolated player, self-play, options.
    is_invalid(bradley_terry(
        &[vec![0, 2], vec![0, 0]],
        &BradleyTerryOpts::default(),
    ));
    is_invalid(bradley_terry(
        &[vec![0, 1, 0], vec![1, 0, 0], vec![0, 0, 0]],
        &BradleyTerryOpts::default(),
    ));
    is_invalid(bradley_terry(
        &[vec![1, 1], vec![1, 0]],
        &BradleyTerryOpts::default(),
    ));
    is_invalid(bradley_terry(&[vec![0]], &BradleyTerryOpts::default()));
    is_invalid(bradley_terry(
        &[vec![0, 1], vec![1, 0]],
        &BradleyTerryOpts {
            max_iter: 0,
            tol: 1e-12,
        },
    ));
    is_invalid(bradley_terry(
        &[vec![0, 1], vec![1, 0]],
        &BradleyTerryOpts {
            max_iter: 10,
            tol: f64::NAN,
        },
    ));
    let beat = |winner, loser| PairwiseOutcome { winner, loser };
    assert_eq!(
        wins_matrix(&[beat(1, 0), beat(1, 0), beat(0, 1)], 2).unwrap(),
        vec![vec![0, 1], vec![2, 0]]
    );
    is_invalid(wins_matrix(&[beat(0, 0)], 2));
    is_invalid(wins_matrix(&[beat(0, 2)], 2));
}

#[test]
fn worker_quality_metrics_with_missing_answers_and_absent_categories() {
    let labels = [Some(0), Some(2), None, Some(1), Some(1), Some(0), None];
    let gold = [0, 1, 1, 1, 0, 2, 2];
    let acc = worker_accuracy(&labels, &gold).unwrap();
    assert_eq!(
        (acc.correct, acc.answered, acc.accuracy),
        (2, 5, Some(q(2, 5)))
    );
    let none = worker_accuracy(&[None, None], &[0, 1]).unwrap();
    assert_eq!((none.correct, none.answered, none.accuracy), (0, 0, None));
    is_invalid(worker_accuracy(&labels, &gold[..6]));
    // Per category over the 5 answered items (gold 0,1,1,0,2 vs answered 0,2,1,1,0):
    //   cat 0: tp 1, fp 1, fn 1 → P 1/2, R 1/2, F1 1/2; cat 1: tp 1, fp 1, fn 1; cat 2: tp 0, fp 1, fn 1 → P 0, R 0, F1 0;
    //   cat 3 never occurs → all None.
    let m = category_metrics(&labels, &gold, 4).unwrap();
    assert_eq!(
        (
            m[0].true_positives,
            m[0].false_positives,
            m[0].false_negatives
        ),
        (1, 1, 1)
    );
    assert_eq!(
        (m[0].precision.clone(), m[0].recall.clone(), m[0].f1.clone()),
        (Some(q(1, 2)), Some(q(1, 2)), Some(q(1, 2)))
    );
    assert_eq!(
        (m[2].precision.clone(), m[2].recall.clone(), m[2].f1.clone()),
        (Some(qi(0)), Some(qi(0)), Some(qi(0)))
    );
    assert_eq!(
        (m[3].precision.clone(), m[3].recall.clone(), m[3].f1.clone()),
        (None, None, None)
    );
    assert_eq!(m[3].category, 3);
    is_invalid(category_metrics(&labels, &gold, 2));
    is_invalid(category_metrics(&labels, &gold[..6], 4));
    // Gold screening: threshold 0 passes every rater who answered a gold item; 1 needs perfection.
    let t = LabelTable::from_rows(
        &[
            &[Some(0), Some(1), None],
            &[Some(1), Some(1), None],
            &[Some(0), Some(0), Some(0)],
            &[Some(1), None, Some(0)],
        ],
        2,
    )
    .unwrap();
    let gold = [Some(0), Some(1), None, Some(1)];
    assert_eq!(
        gold_screening(&t, &gold, &qi(0)).unwrap(),
        vec![Some(true), Some(true), Some(true)]
    );
    assert_eq!(
        gold_screening(&t, &gold, &qi(1)).unwrap(),
        vec![Some(true), Some(false), Some(false)]
    );
    assert_eq!(
        gold_screening(&t, &gold, &q(1, 2)).unwrap(),
        vec![Some(true), Some(true), Some(false)]
    );
    assert_eq!(
        gold_screening(&t, &[None; 4], &q(1, 2)).unwrap(),
        vec![None, None, None]
    );
    is_invalid(gold_screening(&t, &gold[..3], &q(1, 2)));
    is_invalid(gold_screening(&t, &[Some(2), None, None, None], &q(1, 2)));
    is_invalid(gold_screening(&t, &gold, &q(3, 2)));
}

#[test]
fn proportion_intervals_new_counts_and_levels_match_statsmodels() {
    use IntervalMethod::*;
    let ci = |k, n, c, m| proportion_interval(k, n, c, m).unwrap();
    let check = |i: Interval<f64>, lo: f64, hi: f64| {
        close(i.lower, lo, 1e-12);
        close(i.upper, hi, 1e-12);
    };
    // statsmodels: proportion_confint(1, 30, alpha=0.01, method=…): wilson (0.003925688565395338, 0.2317757164381748)
    //   beta (0.00016707076957579573, 0.22275106502443376) agresti_coull (0.0, 0.2550673260831378) normal (0.0, 0.11775116571085345)
    check(
        ci(1, 30, 0.99, Wilson),
        0.003_925_688_565_395_338,
        0.231_775_716_438_174_8,
    );
    check(
        ci(1, 30, 0.99, ClopperPearson),
        0.000_167_070_769_575_795_73,
        0.222_751_065_024_433_76,
    );
    check(ci(1, 30, 0.99, AgrestiCoull), 0.0, 0.255_067_326_083_137_8);
    check(ci(1, 30, 0.99, Wald), 0.0, 0.117_751_165_710_853_45);
    // proportion_confint(29, 30, alpha=0.10): wilson (0.8635958578481829, 0.9925281212350577) beta (0.8514039313408869, 0.9982916843555343)
    //   agresti_coull (0.8537456558733094, 1.0) normal (0.9127597646936846, 1.0)
    check(
        ci(29, 30, 0.90, Wilson),
        0.863_595_857_848_182_9,
        0.992_528_121_235_057_7,
    );
    check(
        ci(29, 30, 0.90, ClopperPearson),
        0.851_403_931_340_886_9,
        0.998_291_684_355_534_3,
    );
    check(ci(29, 30, 0.90, AgrestiCoull), 0.853_745_655_873_309_4, 1.0);
    check(ci(29, 30, 0.90, Wald), 0.912_759_764_693_684_6, 1.0);
    // proportion_confint(250, 1000, alpha=0.05): wilson (0.2241530989836914, 0.27776028025908617) beta (0.2234304062646804, 0.2780500062237555)
    //   agresti_coull (0.2241360963459787, 0.27777728289679887) normal (0.22316208784424263, 0.27683791215575737)
    check(
        ci(250, 1000, 0.95, Wilson),
        0.224_153_098_983_691_4,
        0.277_760_280_259_086_17,
    );
    check(
        ci(250, 1000, 0.95, ClopperPearson),
        0.223_430_406_264_680_4,
        0.278_050_006_223_755_5,
    );
    check(
        ci(250, 1000, 0.95, AgrestiCoull),
        0.224_136_096_345_978_7,
        0.277_777_282_896_798_87,
    );
    check(
        ci(250, 1000, 0.95, Wald),
        0.223_162_087_844_242_63,
        0.276_837_912_155_757_37,
    );
    // proportion_confint(2, 7, alpha=0.20): wilson (0.12533599053745428, 0.5275371834434032) beta (0.07882344616014128, 0.5961797278480441)
    //   agresti_coull (0.12202373128311073, 0.5308494426977466) normal (0.06689327207483031, 0.5045352993537411)
    check(
        ci(2, 7, 0.80, Wilson),
        0.125_335_990_537_454_28,
        0.527_537_183_443_403_2,
    );
    check(
        ci(2, 7, 0.80, ClopperPearson),
        0.078_823_446_160_141_28,
        0.596_179_727_848_044_1,
    );
    check(
        ci(2, 7, 0.80, AgrestiCoull),
        0.122_023_731_283_110_73,
        0.530_849_442_697_746_6,
    );
    check(
        ci(2, 7, 0.80, Wald),
        0.066_893_272_074_830_31,
        0.504_535_299_353_741_1,
    );
    // proportion_confint(500, 1000, alpha=0.001): wilson (0.4482516044524614, 0.5517483955475386) beta (0.4476294198586042, 0.5523705801413958)
    check(
        ci(500, 1000, 0.999, Wilson),
        0.448_251_604_452_461_4,
        0.551_748_395_547_538_6,
    );
    check(
        ci(500, 1000, 0.999, ClopperPearson),
        0.447_629_419_858_604_2,
        0.552_370_580_141_395_8,
    );
}

#[test]
fn proportion_intervals_single_trial_and_containment() {
    use IntervalMethod::*;
    let ci = |k, n, c, m| proportion_interval(k, n, c, m).unwrap();
    // statsmodels: proportion_confint(0, 1, 0.05): wilson (0.0, 0.7934506856227627) beta (0.0, 0.975)
    //   agresti_coull (0.0, 0.832500514520587) normal (0.0, 0.0);
    //   proportion_confint(1, 1, 0.05): wilson (0.2065493143772374, 1.0) beta (0.025, 1.0) agresti_coull (0.167499485479413, 1.0) normal (1.0, 1.0)
    let i = ci(0, 1, 0.95, Wilson);
    close(i.lower, 0.0, 1e-15);
    close(i.upper, 0.793_450_685_622_762_7, 1e-12);
    let i = ci(0, 1, 0.95, ClopperPearson);
    close(i.lower, 0.0, 1e-15);
    close(i.upper, 0.975, 1e-12);
    close(
        ci(0, 1, 0.95, AgrestiCoull).upper,
        0.832_500_514_520_587,
        1e-12,
    );
    assert_eq!(
        (ci(0, 1, 0.95, Wald).lower, ci(0, 1, 0.95, Wald).upper),
        (0.0, 0.0)
    );
    close(ci(1, 1, 0.95, Wilson).lower, 0.206_549_314_377_237_4, 1e-12);
    close(ci(1, 1, 0.95, ClopperPearson).lower, 0.025, 1e-12);
    close(
        ci(1, 1, 0.95, AgrestiCoull).lower,
        0.167_499_485_479_413,
        1e-12,
    );
    assert_eq!(
        (ci(1, 1, 0.95, Wald).lower, ci(1, 1, 0.95, Wald).upper),
        (1.0, 1.0)
    );
    // Every interval contains p̂ when 0 < k < n; at 95% Clopper–Pearson contains Wilson for these
    // counts (statsmodels agrees).  NB: this containment is *not* universal — at 99% and (1, 30)
    // the Clopper–Pearson upper limit 0.2228 lies below Wilson's 0.2318.
    for (k, n) in [(1usize, 30usize), (3, 10), (29, 30), (500, 1000)] {
        let p = k as f64 / n as f64;
        for m in [Wilson, ClopperPearson, AgrestiCoull, Wald] {
            let i = ci(k, n, 0.95, m);
            assert!(i.lower < p && p < i.upper, "{m:?} {k}/{n}: {i:?}");
            assert!(0.0 <= i.lower && i.upper <= 1.0);
        }
        let w = ci(k, n, 0.95, Wilson);
        let c = ci(k, n, 0.95, ClopperPearson);
        assert!(
            c.lower <= w.lower && w.upper <= c.upper,
            "{k}/{n}: {c:?} vs {w:?}"
        );
    }
    let w = ci(1, 30, 0.99, Wilson);
    let c = ci(1, 30, 0.99, ClopperPearson);
    assert!(c.upper < w.upper);
    is_invalid(proportion_interval(1, 1, 1.0, Wilson));
    is_invalid(proportion_interval(1, 1, f64::NAN, Wilson));
}

// ═══════════════════════════════════════════════════════════════════════
// data
// ═══════════════════════════════════════════════════════════════════════

fn r8() -> Vec<Q> {
    vec![
        q(-3, 2),
        q(7, 4),
        qi(2),
        q(-1, 2),
        qi(5),
        qi(5),
        q(3, 4),
        qi(0),
    ]
}

fn s8() -> Vec<Q> {
    from_i64(&[2, -1, 0, 3, 1, 1, 4, -2])
}

#[test]
fn data_moments_of_a_rational_sample_with_negatives_and_a_tie() {
    let r = r8();
    // statistics: mean(R8) = 25/16, pvariance = 1299/256, variance = 1299/224
    assert_eq!(mean(&r).unwrap(), q(25, 16));
    assert_eq!(variance(&r, Ddof::Population).unwrap(), q(1_299, 256));
    assert_eq!(variance(&r, Ddof::Sample).unwrap(), q(1_299, 224));
    assert_eq!(sum_of_squares(&r).unwrap(), q(1_299, 32));
    assert_eq!(sum(&r), q(25, 2));
    // Fraction central moments: m₂ = 1299/256, m₃ = 10107/2048, m₄ = 3209205/65536;
    // scipy: skew(R8, bias=True) = 0.4317561523017507 (skew² = 15133548/81182737),
    //        kurtosis(R8, fisher=True, bias=True) = -1.0981373129445817 (= −617666/562467)
    assert_eq!(central_moment(&r, 2).unwrap(), q(1_299, 256));
    assert_eq!(central_moment(&r, 3).unwrap(), q(10_107, 2_048));
    assert_eq!(central_moment(&r, 4).unwrap(), q(3_209_205, 65_536));
    assert_eq!(central_moment(&r, 1).unwrap(), qi(0));
    assert_eq!(central_moment(&r, 0).unwrap(), qi(1));
    let ctx = Context::new();
    let sk = skewness(&ctx, &r).unwrap();
    close(sk.eval_f64().unwrap(), 0.431_756_152_301_750_7, 1e-12);
    assert_eq!((&sk * &sk).simplify(), ctx.rational(15_133_548, 81_182_737));
    assert_eq!(kurtosis(&r).unwrap(), q(-617_666, 562_467));
    // Constant / tiny samples.
    is_invalid(variance(&[qi(3)], Ddof::Sample));
    is_invalid(variance(&[], Ddof::Population));
    assert_eq!(variance(&[qi(3)], Ddof::Population).unwrap(), qi(0));
    is_invalid(skewness(&ctx, &from_i64(&[4, 4, 4])));
    is_invalid(kurtosis(&[qi(3)]));
    is_invalid(mean(&[]));
    is_invalid(central_moment(&[], 2));
}

#[test]
fn data_covariance_and_correlations_exact() {
    let (r, s) = (r8(), s8());
    let ctx = Context::new();
    // statistics: covariance(R8, S8) = -0.8214285714285714 (= −23/28); population −23/32
    assert_eq!(covariance(&r, &s, Ddof::Sample).unwrap(), q(-23, 28));
    assert_eq!(covariance(&r, &s, Ddof::Population).unwrap(), q(-23, 32));
    // statistics: correlation(R8, S8) = -0.17055295274128673; r² = 529/18186 exactly
    let p = pearson(&ctx, &r, &s).unwrap();
    close(p.eval_f64().unwrap(), -0.170_552_952_741_286_73, 1e-12);
    assert_eq!((&p * &p).simplify(), ctx.rational(529, 18_186));
    // scipy: spearmanr(R8, S8).statistic = -0.27710843373493976 (ρ² = 529/6889, ρ = −23/83)
    let rho = spearman(&ctx, &r, &s).unwrap();
    assert_eq!(rho, ctx.rational(-23, 83));
    assert_eq!(pearson(&ctx, &ranks(&r), &ranks(&s)).unwrap(), rho);
    // scipy: kendalltau(R8, S8).statistic = -0.037037037037037035 (= −1/√729 = −1/27)
    let tau = kendall_tau(&ctx, &r, &s).unwrap();
    assert_eq!(tau, ctx.rational(-1, 27));
    // scipy: rankdata(R8) = [1, 5, 6, 2, 7.5, 7.5, 4, 3]
    assert_eq!(
        ranks(&r),
        vec![qi(1), qi(5), qi(6), qi(2), q(15, 2), q(15, 2), qi(4), qi(3)]
    );
    assert_eq!(tie_sizes(&r), vec![2]);
    assert_eq!(tie_sizes(&s), vec![2]);
    assert!(tie_sizes(&from_i64(&[1, 2, 3])).is_empty());
    assert!(ranks(&[]).is_empty());
    is_invalid(covariance(&r, &s[..7], Ddof::Sample));
    is_invalid(pearson(&ctx, &r, &from_i64(&[1; 8])));
    is_invalid(kendall_tau(&ctx, &r, &from_i64(&[1; 8])));
    is_invalid(kendall_tau(&ctx, &r[..1], &s[..1]));
    is_invalid(spearman(&ctx, &r, &s[..7]));
}

#[test]
fn data_order_statistics_quantiles_and_modes() {
    let r = r8();
    // statistics: median(R8) = 5/4; quantiles(R8, n=4) = [-3/8, 5/4, 17/4]; method='inclusive' → [-1/8, 5/4, 11/4]
    assert_eq!(median(&r).unwrap(), q(5, 4));
    assert_eq!(
        quantiles(&r, 4, QuantileMethod::Exclusive).unwrap(),
        vec![q(-3, 8), q(5, 4), q(17, 4)]
    );
    assert_eq!(
        quantiles(&r, 4, QuantileMethod::Inclusive).unwrap(),
        vec![q(-1, 8), q(5, 4), q(11, 4)]
    );
    assert_eq!(iqr(&r, QuantileMethod::Exclusive).unwrap(), q(37, 8));
    // scipy: iqr(R8) = 2.875 (= 23/8, the inclusive / numpy 'linear' method)
    assert_eq!(iqr(&r, QuantileMethod::Inclusive).unwrap(), q(23, 8));
    // Extremes: p = 0 and p = 1 hit the minimum / maximum in both methods.
    for m in [QuantileMethod::Exclusive, QuantileMethod::Inclusive] {
        assert_eq!(quantile(&r, &qi(0), m).unwrap(), q(-3, 2));
        assert_eq!(quantile(&r, &qi(1), m).unwrap(), qi(5));
        assert_eq!(quantile(&[qi(7)], &q(1, 3), m).unwrap(), qi(7));
    }
    // Exclusive at p = 1/10 with n = 8: position 0.9 < 1 clamps to the minimum (numpy 'weibull' gives -1.5;
    // Python's statistics.quantiles(R8, n=10)[0] extrapolates to -8/5 instead).
    assert_eq!(
        quantile(&r, &q(1, 10), QuantileMethod::Exclusive).unwrap(),
        q(-3, 2)
    );
    // numpy: quantile(R8, 0.1) (linear) = -0.7999999999999999 (= −4/5)
    assert_eq!(
        quantile(&r, &q(1, 10), QuantileMethod::Inclusive).unwrap(),
        q(-4, 5)
    );
    // statistics: multimode(R8) = [5]; everything distinct → every value is a mode.
    assert_eq!(modes(&r), vec![qi(5)]);
    assert_eq!(modes(&from_i64(&[3, 1, 2])), from_i64(&[1, 2, 3]));
    assert!(modes(&[]).is_empty());
    assert_eq!(frequencies(&r)[6], (qi(5), 2));
    assert_eq!(frequencies(&r).len(), 7);
    let mm = min_max(&r).unwrap();
    assert_eq!((mm.lower, mm.upper), (q(-3, 2), qi(5)));
    assert_eq!(sorted(&r)[0], q(-3, 2));
    is_invalid(quantile(&r, &q(5, 4), QuantileMethod::Exclusive));
    is_invalid(quantile(&[], &q(1, 2), QuantileMethod::Exclusive));
    is_invalid(quantiles(&r, 1, QuantileMethod::Exclusive));
    is_invalid(median(&[]));
    is_invalid(min_max(&[]));
}

#[test]
fn data_robust_statistics_means_and_outliers() {
    let r = r8();
    let ctx = Context::new();
    // scipy: median_abs_deviation(R8) = 1.5
    assert_eq!(median_abs_deviation(&r).unwrap(), q(3, 2));
    assert_eq!(median_abs_deviation(&[qi(3)]).unwrap(), qi(0));
    // scipy: zscore(R8, ddof=1)[4] = 1.4274540609868562, squared 21175/10392
    let z = zscores(&ctx, &r, Ddof::Sample).unwrap();
    close(z[4].eval_f64().unwrap(), 1.427_454_060_986_856_2, 1e-12);
    assert_eq!((&z[4] * &z[4]).simplify(), ctx.rational(21_175, 10_392));
    close(z[0].eval_f64().unwrap(), -1.271_731_799_788_290_2, 1e-12);
    is_invalid(zscores(&ctx, &from_i64(&[2, 2]), Ddof::Sample));
    // P = [1/2, 2, 9/4, 8, 3]: scipy gmean = 2.22064303492292 (= 54^(1/5)); statistics.harmonic_mean = 72/49
    let p = vec![q(1, 2), qi(2), q(9, 4), qi(8), qi(3)];
    let g = geometric_mean(&ctx, &p).unwrap();
    close(g.eval_f64().unwrap(), 2.220_643_034_922_92, 1e-12);
    assert_eq!(g.pow(&ctx.int(5)).simplify(), ctx.int(54));
    assert_eq!(harmonic_mean(&p).unwrap(), q(72, 49));
    is_invalid(geometric_mean(&ctx, &from_i64(&[1, 0])));
    is_invalid(harmonic_mean(&from_i64(&[1, -2])));
    is_invalid(geometric_mean(&ctx, &[]));
    // scipy: trim_mean([1..9, 100], 0.1) = 5.5; 0.25 → 5.5; 0.05 (⌊0.5⌋ = 0) → 14.5
    let t = from_i64(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 100]);
    assert_eq!(trimmed_mean(&t, &q(1, 10)).unwrap(), q(11, 2));
    assert_eq!(trimmed_mean(&t, &q(1, 4)).unwrap(), q(11, 2));
    assert_eq!(trimmed_mean(&t, &q(1, 20)).unwrap(), q(29, 2));
    assert_eq!(
        trimmed_mean(&from_i64(&[1, 5, 9]), &q(49, 100)).unwrap(),
        qi(5)
    );
    is_invalid(trimmed_mean(&t, &q(1, 2)));
    is_invalid(trimmed_mean(&[], &qi(0)));
    // Tukey fences (k = 3/2): exclusive Q₁ = 11/4, Q₃ = 33/4 → upper fence 33/2; inclusive 13/4, 31/4 → 29/2.
    assert_eq!(
        iqr_outliers(&t, &q(3, 2), QuantileMethod::Exclusive).unwrap(),
        vec![9]
    );
    assert_eq!(
        iqr_outliers(&t, &q(3, 2), QuantileMethod::Inclusive).unwrap(),
        vec![9]
    );
    assert!(
        iqr_outliers(&t, &qi(100), QuantileMethod::Inclusive)
            .unwrap()
            .is_empty()
    );
    assert!(
        iqr_outliers(&from_i64(&[2, 2, 2]), &q(3, 2), QuantileMethod::Exclusive)
            .unwrap()
            .is_empty()
    );
    // Modified z-scores: median 11/2, MAD 5/2; 0.6745·(100 − 5.5)/2.5 = 254961/10000 > 7/2; the rest are below.
    assert_eq!(mad_outliers(&t, &q(7, 2)).unwrap(), vec![9]);
    assert_eq!(
        mad_outliers(&t, &q(254_961, 10_000)).unwrap(),
        Vec::<usize>::new()
    );
    assert_eq!(mad_outliers(&t, &qi(1)).unwrap(), vec![0, 9]);
    is_invalid(mad_outliers(&from_i64(&[2, 2, 2, 9]), &q(7, 2)));
}

#[test]
fn data_conversions_and_binomials() {
    // Fraction(0.1) = 3602879701896397/36028797018963968; Fraction(-2.5) = -5/2
    let v = from_f64(&[0.1, -2.5, 0.0]).unwrap();
    assert_eq!(
        v[0],
        Q::new(
            BigInt::from(3_602_879_701_896_397i64),
            BigInt::from(36_028_797_018_963_968i64)
        )
    );
    assert_eq!(v[1], q(-5, 2));
    assert_eq!(v[2], qi(0));
    assert_eq!(to_f64(&v), vec![0.1, -2.5, 0.0]);
    // Subnormal: Fraction(1e-320) = 253 / 2^1074 round-trips too.
    let tiny = from_f64(&[1e-320]).unwrap();
    assert_eq!(*tiny[0].numer(), BigInt::from(253));
    assert_eq!(to_f64(&tiny), vec![1e-320]);
    is_invalid(from_f64(&[f64::NAN]));
    is_invalid(from_f64(&[f64::NEG_INFINITY]));
    // A ratio whose numerator and denominator both exceed f64::MAX but whose value is small:
    // (10^400 + 1) / 10^399 ≈ 10.
    let ten = BigInt::from(10);
    let huge = Q::new(
        num_traits::pow(ten.clone(), 400) + 1,
        num_traits::pow(ten, 399),
    );
    let f = to_f64(&[huge])[0];
    assert!(
        f.is_finite(),
        "to_f64 of a large-numerator/denominator ratio must not be NaN, got {f}"
    );
    close(f, 10.0, 1e-12);
    // C(30, 15) = 155117520; C(5, 0) = 1; C(3, 5) = 0; C(0, 0) = 1
    assert_eq!(binomial_q(30, 15), qi(155_117_520));
    assert_eq!(binomial_q(5, 0), qi(1));
    assert_eq!(binomial_q(3, 5), qi(0));
    assert_eq!(binomial_q(0, 0), qi(1));
    assert_eq!(
        counts(&[&[1, 2], &[3, 4]]),
        vec![from_i64(&[1, 2]), from_i64(&[3, 4])]
    );
}
