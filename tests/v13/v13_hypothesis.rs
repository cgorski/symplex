//! symplex 0.13 — hypothesis.  Reference values cite scipy 1.18 / statsmodels 0.15 /
//! SymPy 1.14 (`symplex/.venv/bin/python`).  Every reference number below was
//! produced by the call quoted next to it; nothing was computed by hand.

use symplex::linprog::{Q, q, qi};
use symplex::prelude::*;
use symplex::stats::data::{from_f64, from_i64};
use symplex::stats::hypothesis::*;
use symplex::stats::{Distribution, Rng};

fn close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() < tol,
        "expected {expected}, got {actual} (|Δ| = {:e})",
        (actual - expected).abs()
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

// ═══════════════════════════════════════════════════════════════════════════
// 1. Exact discrete tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn binomial_two_sided_is_scipy_minlike_mass_exactly() {
    let ctx = Context::new();
    // scipy: binomtest(7, 10, 0.5).pvalue = 0.34375
    let r = binomial_test(&ctx, 7, 10, &q(1, 2), Alternative::TwoSided).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(11, 32)));
    assert_eq!(r.statistic_exact(), Some(q(7, 10)));
    assert!(r.df.is_none());
    // scipy: binomtest(4, 20, 1/3).pvalue = 0.24340663241369162 (Fraction sum: 848706449/3486784401)
    let r = binomial_test(&ctx, 4, 20, &q(1, 3), Alternative::TwoSided).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(848_706_449, 3_486_784_401)));
    close(p_of(&r), 0.243_406_632_413_691_62, 1e-12);
    // scipy: binomtest(18, 20, 0.7).pvalue = 0.052627948729727106
    let r = binomial_test(&ctx, 18, 20, &q(7, 10), Alternative::TwoSided).unwrap();
    close(p_of(&r), 0.052_627_948_729_727_106, 1e-12);
    // scipy: binomtest(0, 5, 0.3).pvalue = 0.33115000000000006 (= 6623/20000)
    let r = binomial_test(&ctx, 0, 5, &q(3, 10), Alternative::TwoSided).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(6623, 20000)));
    // scipy: binomtest(5, 5, 0.3).pvalue = 0.0024299999999999994 (= 243/100000)
    let r = binomial_test(&ctx, 5, 5, &q(3, 10), Alternative::TwoSided).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(243, 100_000)));
    // scipy: binomtest(3, 12, 0.25).pvalue = 1.0  (k = n·p exactly)
    let r = binomial_test(&ctx, 3, 12, &q(1, 4), Alternative::TwoSided).unwrap();
    assert_eq!(r.p_value_exact(), Some(qi(1)));
}

#[test]
fn binomial_one_sided_tails_are_exact() {
    let ctx = Context::new();
    // scipy: binomtest(9, 12, 0.5, alternative='greater').pvalue = 0.072998046875 (= 299/4096)
    let r = binomial_test(&ctx, 9, 12, &q(1, 2), Alternative::Greater).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(299, 4096)));
    // scipy: binomtest(2, 12, 0.5, alternative='less').pvalue = 0.019287109375 (= 79/4096)
    let r = binomial_test(&ctx, 2, 12, &q(1, 2), Alternative::Less).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(79, 4096)));
    assert_eq!(r.alternative, Alternative::Less);
}

#[test]
fn binomial_rejects_bad_input() {
    let ctx = Context::new();
    is_invalid(binomial_test(&ctx, 1, 0, &q(1, 2), Alternative::TwoSided));
    is_invalid(binomial_test(&ctx, 11, 10, &q(1, 2), Alternative::TwoSided));
    is_invalid(binomial_test(&ctx, 1, 10, &q(3, 2), Alternative::TwoSided));
    is_invalid(binomial_test(&ctx, 1, 10, &q(-1, 2), Alternative::TwoSided));
}

#[test]
fn fisher_exact_all_alternatives_match_scipy_exactly() {
    let ctx = Context::new();
    let t = [[8, 2], [1, 5]];
    // scipy: fisher_exact([[8, 2], [1, 5]]) → (20.0, 0.034965034965034975)  = 5/143
    let r = fisher_exact(&ctx, t, Alternative::TwoSided).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(20)));
    assert_eq!(r.p_value_exact(), Some(q(5, 143)));
    close(p_of(&r), 0.034_965_034_965_034_975, 1e-15);
    // scipy: fisher_exact(t, alternative='greater').pvalue = 0.024475524475524483 = 7/286
    let r = fisher_exact(&ctx, t, Alternative::Greater).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(7, 286)));
    // scipy: fisher_exact(t, alternative='less').pvalue = 0.9991258741258742 = 1143/1144
    let r = fisher_exact(&ctx, t, Alternative::Less).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(1143, 1144)));
    // scipy: fisher_exact([[3, 1], [1, 3]]) → (9.0, 0.48571428571428565) = 17/35
    let r = fisher_exact(&ctx, [[3, 1], [1, 3]], Alternative::TwoSided).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(9)));
    assert_eq!(r.p_value_exact(), Some(q(17, 35)));
    // scipy: fisher_exact([[2, 7], [8, 2]]) → (0.07142857142857142, 0.02301413756522116) = 1/14, 1063/46189
    let r = fisher_exact(&ctx, [[2, 7], [8, 2]], Alternative::TwoSided).unwrap();
    assert_eq!(r.statistic_exact(), Some(q(1, 14)));
    assert_eq!(r.p_value_exact(), Some(q(1063, 46189)));
    // scipy: fisher_exact([[5, 5], [5, 5]]) → (1.0, 1.0)
    let r = fisher_exact(&ctx, [[5, 5], [5, 5]], Alternative::TwoSided).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(1)));
    assert_eq!(r.p_value_exact(), Some(qi(1)));
}

#[test]
fn fisher_infinite_odds_ratio_and_degenerate_tables() {
    let ctx = Context::new();
    // scipy: fisher_exact([[10, 0], [3, 5]]) → (inf, 0.006535947712418301) = 1/153
    let r = fisher_exact(&ctx, [[10, 0], [3, 5]], Alternative::TwoSided).unwrap();
    assert_eq!(r.statistic, ctx.infinity());
    assert_eq!(r.p_value_exact(), Some(q(1, 153)));
    // An empty row or column: scipy reports (nan, 1.0); here an error.
    is_invalid(fisher_exact(&ctx, [[0, 0], [3, 5]], Alternative::TwoSided));
    is_invalid(fisher_exact(&ctx, [[4, 0], [3, 0]], Alternative::TwoSided));
}

#[test]
fn fisher_less_tail_matches_hypergeometric_cdf() {
    let ctx = Context::new();
    let [[a, b], [c, d]] = [[8usize, 2], [1, 5]];
    let r = fisher_exact(&ctx, [[a, b], [c, d]], Alternative::Less).unwrap();
    // X ~ Hypergeometric(N = 16, successes = a + b = 10, draws = a + c = 9); P(X ≤ 8).
    let h = Distribution::hypergeometric(ctx.int(16), ctx.int(10), ctx.int(9));
    assert_eq!(r.p_value, h.cdf(&ctx.int(8)));
}

#[test]
fn mcnemar_exact_and_chi_squared_match_statsmodels() {
    let ctx = Context::new();
    // statsmodels: mcnemar([[100, 5], [15, 100]], exact=True) → (5.0, 0.04138946533203125) = 5425/131072
    let r = mcnemar_test(&ctx, 5, 15, true, true).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(5)));
    assert_eq!(r.p_value_exact(), Some(q(5425, 131_072)));
    // statsmodels: mcnemar(..., exact=False, correction=True) → (4.05, 0.0441713449084427)
    let r = mcnemar_test(&ctx, 5, 15, false, true).unwrap();
    assert_eq!(r.statistic_exact(), Some(q(81, 20)));
    assert_eq!(r.df, Some(ctx.int(1)));
    close(p_of(&r), 0.044_171_344_908_442_7, 1e-12);
    // statsmodels: mcnemar(..., exact=False, correction=False) → (5.0, 0.025347318677468252)
    let r = mcnemar_test(&ctx, 5, 15, false, false).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(5)));
    close(p_of(&r), 0.025_347_318_677_468_252, 1e-12);
    // statsmodels: mcnemar([[100, 10], [10, 100]], exact=True).pvalue = 1.0
    let r = mcnemar_test(&ctx, 10, 10, true, true).unwrap();
    assert_eq!(r.p_value_exact(), Some(qi(1)));
    // statsmodels: mcnemar([[100, 10], [10, 100]], exact=False, correction=True) → (0.05, 0.8230632737581214)
    let r = mcnemar_test(&ctx, 10, 10, false, true).unwrap();
    assert_eq!(r.statistic_exact(), Some(q(1, 20)));
    close(p_of(&r), 0.823_063_273_758_121_4, 1e-12);
    // statsmodels: mcnemar([[100, 10], [10, 100]], exact=False, correction=False) → (0.0, 1.0)
    let r = mcnemar_test(&ctx, 10, 10, false, false).unwrap();
    assert_eq!(r.p_value, ctx.one());
    // statsmodels: mcnemar([[100, 3], [9, 100]], exact=True).pvalue = 0.14599609375 = 299/2048
    let r = mcnemar_test(&ctx, 3, 9, true, true).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(299, 2048)));
    // statsmodels: mcnemar([[100, 3], [9, 100]], exact=False, correction=True) → (2.0833333333333335, 0.14891467317876161)
    let r = mcnemar_test(&ctx, 3, 9, false, true).unwrap();
    assert_eq!(r.statistic_exact(), Some(q(25, 12)));
    close(p_of(&r), 0.148_914_673_178_761_61, 1e-12);
    // statsmodels: mcnemar([[100, 3], [9, 100]], exact=False, correction=False) → (3.0, 0.08326451666355052)
    let r = mcnemar_test(&ctx, 3, 9, false, false).unwrap();
    close(p_of(&r), 0.083_264_516_663_550_52, 1e-12);
    is_invalid(mcnemar_test(&ctx, 0, 0, true, true));
}

#[test]
fn sign_test_matches_statsmodels() {
    let ctx = Context::new();
    let x = from_i64(&[3, 5, 7, 8, 9, 11, 12, 15, 2, 6]);
    // statsmodels: sign_test(x, mu0=5) = (2.5, 0.1796875)   → M = (7 − 2)/2, p = 23/128
    let r = sign_test(&ctx, &x, &qi(5), Alternative::TwoSided).unwrap();
    assert_eq!(r.statistic_exact(), Some(q(5, 2)));
    assert_eq!(r.p_value_exact(), Some(q(23, 128)));
    // scipy: binomtest(7, 9, 0.5, alternative='greater').pvalue = 0.08984375 = 23/256
    let r = sign_test(&ctx, &x, &qi(5), Alternative::Greater).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(23, 256)));
    is_invalid(sign_test(
        &ctx,
        &from_i64(&[5, 5, 5]),
        &qi(5),
        Alternative::TwoSided,
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Categorical
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn chi_square_independence_2x3_matches_scipy() {
    let ctx = Context::new();
    let t = counts(&[&[10, 20, 30], &[6, 9, 17]]);
    // scipy: chi2_contingency(t) → statistic 0.27157465150403504, dof 2, pvalue 0.873028283380073,
    //        expected_freq[0][0] = 10.434782608695652 = 240/23
    let r = chi_square_independence(&ctx, &t, true).unwrap();
    assert_eq!(r.df, 2);
    assert_eq!(r.expected[0][0], q(240, 23));
    assert_eq!(r.expected.len(), 2);
    close(
        ctx.from_ratio(r.statistic.clone()).eval_f64().unwrap(),
        0.271_574_651_504_035_04,
        1e-12,
    );
    close(r.p_value_f64().unwrap(), 0.873_028_283_380_073, 1e-12);
    // df = 2: the correction flag is irrelevant.
    let r2 = chi_square_independence(&ctx, &t, false).unwrap();
    assert_eq!(r2.statistic, r.statistic);
}

#[test]
fn chi_square_independence_yates_correction_matches_scipy() {
    let ctx = Context::new();
    let t = counts(&[&[20, 15], &[30, 35]]);
    // scipy: chi2_contingency(t, correction=True) → statistic 0.7032967032967032 (= 64/91), pvalue 0.4016781664697727
    let r = chi_square_independence(&ctx, &t, true).unwrap();
    assert_eq!(r.statistic, q(64, 91));
    assert_eq!(r.df, 1);
    close(r.p_value_f64().unwrap(), 0.401_678_166_469_772_7, 1e-12);
    // scipy: chi2_contingency(t, correction=False) → statistic 1.098901098901099 (= 100/91), pvalue 0.29450739368010714
    let r = chi_square_independence(&ctx, &t, false).unwrap();
    assert_eq!(r.statistic, q(100, 91));
    close(r.p_value_f64().unwrap(), 0.294_507_393_680_107_14, 1e-12);
    // Validation.
    is_invalid(chi_square_independence(&ctx, &counts(&[&[1, 2, 3]]), true));
    is_invalid(chi_square_independence(
        &ctx,
        &counts(&[&[1, 2], &[3]]),
        true,
    ));
    is_invalid(chi_square_independence(
        &ctx,
        &counts(&[&[0, 0], &[3, 4]]),
        true,
    ));
    is_invalid(chi_square_independence(
        &ctx,
        &counts(&[&[1, -2], &[3, 4]]),
        true,
    ));
}

#[test]
fn chi_square_p_value_is_the_chi_squared_survival_function() {
    let ctx = Context::new();
    let t = counts(&[&[20, 15], &[30, 35]]);
    let r = chi_square_independence(&ctx, &t, false).unwrap();
    // The expression is the upper regularised gamma, not a float.
    assert!(format!("{}", r.p_value).contains("uppergamma"));
    let via_dist =
        ctx.one() - Distribution::chi_squared(ctx.int(1)).cdf(&ctx.from_ratio(r.statistic.clone()));
    close(
        r.p_value_f64().unwrap(),
        via_dist.eval_f64().unwrap(),
        1e-14,
    );
}

#[test]
fn chi_square_goodness_of_fit_matches_scipy() {
    let ctx = Context::new();
    let obs = from_i64(&[16, 18, 16, 14, 12, 12]);
    // scipy: chisquare(obs) → (2.0, 0.8491450360846096)
    let r = chi_square_goodness_of_fit(&ctx, &obs, None, 0).unwrap();
    assert_eq!(r.statistic, qi(2));
    assert_eq!(r.df, 5);
    // Uniform expected counts: 88/6 = 44/3 each.
    assert_eq!(r.expected, vec![vec![q(44, 3); 6]]);
    close(r.p_value_f64().unwrap(), 0.849_145_036_084_609_6, 1e-12);
    // scipy: chisquare(obs, f_exp=[16, 16, 16, 16, 16, 8]) → (3.5, 0.6233876277495822)
    let e = from_i64(&[16, 16, 16, 16, 16, 8]);
    let r = chi_square_goodness_of_fit(&ctx, &obs, Some(&e), 0).unwrap();
    assert_eq!(r.statistic, q(7, 2));
    close(r.p_value_f64().unwrap(), 0.623_387_627_749_582_2, 1e-12);
    // scipy: chisquare(obs, ddof=1) → (2.0, 0.7357588823428847)
    let r = chi_square_goodness_of_fit(&ctx, &obs, None, 1).unwrap();
    assert_eq!(r.df, 4);
    close(r.p_value_f64().unwrap(), 0.735_758_882_342_884_7, 1e-12);
    // Totals must agree (scipy raises too).
    is_invalid(chi_square_goodness_of_fit(
        &ctx,
        &obs,
        Some(&from_i64(&[16, 16, 16, 16, 16, 16])),
        0,
    ));
    is_invalid(chi_square_goodness_of_fit(&ctx, &obs, None, 5));
    is_invalid(chi_square_goodness_of_fit(&ctx, &from_i64(&[5]), None, 0));
}

#[test]
fn g_test_matches_scipy_log_likelihood() {
    let ctx = Context::new();
    // scipy: chi2_contingency(t, correction=False, lambda_='log-likelihood') → (0.2740265420246606, 0.8719586542812721)
    let r = g_test(&ctx, &counts(&[&[10, 20, 30], &[6, 9, 17]])).unwrap();
    close(s_of(&r), 0.274_026_542_024_660_6, 1e-12);
    close(p_of(&r), 0.871_958_654_281_272_1, 1e-12);
    assert_eq!(r.df, Some(ctx.int(2)));
    // scipy: chi2_contingency([[20, 15], [30, 35]], correction=False, lambda_='log-likelihood')
    //        → (1.1017309005114981, 0.2938865575980739)
    let r = g_test(&ctx, &counts(&[&[20, 15], &[30, 35]])).unwrap();
    close(s_of(&r), 1.101_730_900_511_498_1, 1e-12);
    close(p_of(&r), 0.293_886_557_598_073_9, 1e-12);
    // A perfectly independent table: G = 0, p = 1.
    let r = g_test(&ctx, &counts(&[&[10, 20], &[20, 40]])).unwrap();
    assert_eq!(r.statistic, ctx.zero());
    assert_eq!(r.p_value, ctx.one());
}

#[test]
fn cramers_v_and_phi_match_scipy() {
    let ctx = Context::new();
    // scipy: association([[10, 20, 30], [6, 9, 17]], method='cramer', correction=False) = 0.05433137570422292
    let v = cramers_v(&ctx, &counts(&[&[10, 20, 30], &[6, 9, 17]])).unwrap();
    close(v.eval_f64().unwrap(), 0.054_331_375_704_222_92, 1e-12);
    // scipy: association([[20, 15], [30, 35]], method='cramer', correction=False) = 0.10482848367219183
    let v = cramers_v(&ctx, &counts(&[&[20, 15], &[30, 35]])).unwrap();
    close(v.eval_f64().unwrap(), 0.104_828_483_672_191_83, 1e-12);
    // scipy: association([[20, 10], [5, 15]], method='cramer', correction=False) = 0.408248290463863
    let v = cramers_v(&ctx, &counts(&[&[20, 10], &[5, 15]])).unwrap();
    close(v.eval_f64().unwrap(), 0.408_248_290_463_863, 1e-12);
    // numpy: corrcoef(indicator rows, indicator cols) of [[20, 10], [5, 15]] = 0.40824829046386296
    let phi = phi_coefficient(&ctx, [[20, 10], [5, 15]]).unwrap();
    close(phi.eval_f64().unwrap(), 0.408_248_290_463_862_96, 1e-12);
    // φ = (ad − bc)/√(...) = 250/√(30·20·25·25) = 1/√6: exact form.
    assert_eq!(phi.powi(2).simplify(), ctx.rational(1, 6));
    is_invalid(phi_coefficient(&ctx, [[0, 0], [5, 15]]));
}

#[test]
fn odds_ratio_and_relative_risk_match_statsmodels_and_scipy() {
    // statsmodels: Table2x2([[20, 10], [5, 15]]).oddsratio = 6.0,
    //              .oddsratio_confint(0.05) = (1.6931795592741443, 21.261773332199304)
    let r = odds_ratio([[20, 10], [5, 15]], 0.95).unwrap();
    assert_eq!(r.estimate, qi(6));
    close(r.ci.lower, 1.693_179_559_274_144_3, 1e-9);
    close(r.ci.upper, 21.261_773_332_199_304, 1e-9);
    // statsmodels: Table2x2([[12, 28], [30, 30]]).oddsratio = 0.42857142857142855 (= 3/7),
    //              .oddsratio_confint(0.10) = (0.21094863132132893, 0.8707023517397149)
    let r = odds_ratio([[12, 28], [30, 30]], 0.90).unwrap();
    assert_eq!(r.estimate, q(3, 7));
    close(r.ci.lower, 0.210_948_631_321_328_93, 1e-9);
    close(r.ci.upper, 0.870_702_351_739_714_9, 1e-9);
    // scipy: relative_risk(20, 30, 5, 20) → 2.6666666666666665, CI(0.95) = (1.198028521436089, 5.935677643623166)
    let r = relative_risk([[20, 10], [5, 15]], 0.95).unwrap();
    assert_eq!(r.estimate, q(8, 3));
    close(r.ci.lower, 1.198_028_521_436_089, 1e-9);
    close(r.ci.upper, 5.935_677_643_623_166, 1e-9);
    // scipy: relative_risk(12, 40, 30, 60) → 0.6, CI(0.90) = (0.38240028924790753, 0.9414218820493997)
    let r = relative_risk([[12, 28], [30, 30]], 0.90).unwrap();
    assert_eq!(r.estimate, q(3, 5));
    close(r.ci.lower, 0.382_400_289_247_907_53, 1e-9);
    close(r.ci.upper, 0.941_421_882_049_399_7, 1e-9);
    is_invalid(odds_ratio([[0, 10], [5, 15]], 0.95));
    is_invalid(relative_risk([[0, 10], [5, 15]], 0.95));
    is_invalid(odds_ratio([[20, 10], [5, 15]], 1.0));
}

#[test]
fn cohens_h_matches_statsmodels() {
    let ctx = Context::new();
    // statsmodels: proportion_effectsize(0.5, 0.4) = 0.20135792079033088
    let h = cohens_h(&ctx, &q(1, 2), &q(2, 5)).unwrap();
    close(h.eval_f64().unwrap(), 0.201_357_920_790_330_88, 1e-12);
    // statsmodels: proportion_effectsize(0.2, 0.35) = -0.3388084547778868
    let h = cohens_h(&ctx, &q(1, 5), &q(7, 20)).unwrap();
    close(h.eval_f64().unwrap(), -0.338_808_454_777_886_8, 1e-12);
    is_invalid(cohens_h(&ctx, &q(3, 2), &q(1, 2)));
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Means and proportions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn t_test_one_sample_all_alternatives_match_scipy() {
    let ctx = Context::new();
    let x = from_i64(&[5, 7, 8, 9, 10, 12]);
    // scipy: ttest_1samp(x, 6) → statistic 2.521097420448054, pvalue 0.053103491751797835
    let r = t_test_one_sample(&ctx, &x, &qi(6), Alternative::TwoSided).unwrap();
    close(s_of(&r), 2.521_097_420_448_054, 1e-12);
    close(p_of(&r), 0.053_103_491_751_797_835, 1e-12);
    assert_eq!(r.df, Some(ctx.int(5)));
    // scipy: ttest_1samp(x, 6, alternative='greater').pvalue = 0.026551745875898917
    let r = t_test_one_sample(&ctx, &x, &qi(6), Alternative::Greater).unwrap();
    close(p_of(&r), 0.026_551_745_875_898_917, 1e-12);
    // scipy: ttest_1samp(x, 6, alternative='less').pvalue = 0.9734482541241011
    let r = t_test_one_sample(&ctx, &x, &qi(6), Alternative::Less).unwrap();
    close(p_of(&r), 0.973_448_254_124_101_1, 1e-12);
    // scipy: ttest_1samp(x, 10, alternative='greater') → (-1.5126584522688322, 0.9046120242444489)
    let r = t_test_one_sample(&ctx, &x, &qi(10), Alternative::Greater).unwrap();
    close(s_of(&r), -1.512_658_452_268_832_2, 1e-12);
    close(p_of(&r), 0.904_612_024_244_448_9, 1e-12);
}

#[test]
fn t_statistic_and_p_value_are_exact_expressions() {
    let ctx = Context::new();
    // x = 1..10, μ₀ = 5: x̄ = 11/2, s² = 55/6, t = (1/2)·√10/√(55/6) → t² = 3/11.
    let x = from_i64(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let r = t_test_one_sample(&ctx, &x, &qi(5), Alternative::TwoSided).unwrap();
    assert_eq!(r.statistic.powi(2).simplify(), ctx.rational(3, 11));
    assert!(format!("{}", r.p_value).contains("betainc_regularized"));
    // scipy: ttest_1samp(range(1, 11), 5) → (0.5222329678670935, 0.6141172548083939)
    close(s_of(&r), 0.522_232_967_867_093_5, 1e-12);
    close(p_of(&r), 0.614_117_254_808_393_9, 1e-12);
    // The p-value agrees with the Student-t CDF of the same family: 2(1 − F(|t|)).
    let t = ctx.from_f64(s_of(&r)).unwrap();
    let via_cdf = ctx.int(2) * (ctx.one() - Distribution::student_t(ctx.int(9)).cdf(&t));
    close(p_of(&r), via_cdf.eval_f64().unwrap(), 1e-12);
}

#[test]
fn t_test_two_sample_student_and_welch_match_scipy() {
    let ctx = Context::new();
    let x = from_i64(&[20, 22, 19, 20, 22, 20, 21]);
    let y = from_i64(&[28, 32, 36, 24, 29, 32]);
    // scipy: ttest_ind(x, y, equal_var=True) → (-5.945966283974979, 9.648307874547567e-05, df 11)
    let r = t_test_two_sample(&ctx, &x, &y, true, Alternative::TwoSided).unwrap();
    close(s_of(&r), -5.945_966_283_974_979, 1e-12);
    close(p_of(&r), 9.648_307_874_547_567e-5, 1e-15);
    assert_eq!(r.df, Some(ctx.int(11)));
    // scipy: ttest_ind(x, y, equal_var=True, alternative='less').pvalue = 4.824153937273783e-05
    let r = t_test_two_sample(&ctx, &x, &y, true, Alternative::Less).unwrap();
    close(p_of(&r), 4.824_153_937_273_783e-5, 1e-15);
    // scipy: ttest_ind(x, y, equal_var=False) → (-5.529270507777645, 0.0017938799124542811, df 5.650760744239527)
    let r = t_test_two_sample(&ctx, &x, &y, false, Alternative::TwoSided).unwrap();
    close(s_of(&r), -5.529_270_507_777_645, 1e-12);
    close(p_of(&r), 0.001_793_879_912_454_281_1, 1e-12);
    // Fraction arithmetic: the Welch–Satterthwaite ν is exactly 3527433605/624240481.
    assert_eq!(r.df, Some(ctx.from_ratio(q(3_527_433_605, 624_240_481))));
    close(
        r.df.clone().unwrap().eval_f64().unwrap(),
        5.650_760_744_239_527,
        1e-12,
    );
    // scipy: ttest_ind(x, y, equal_var=False, alternative='greater').pvalue = 0.9991030600437728
    let r = t_test_two_sample(&ctx, &x, &y, false, Alternative::Greater).unwrap();
    close(p_of(&r), 0.999_103_060_043_772_8, 1e-12);
}

#[test]
fn t_test_paired_matches_scipy_and_the_one_sample_test_of_differences() {
    let ctx = Context::new();
    let before = from_i64(&[200, 190, 210, 180, 195, 205]);
    let after = from_i64(&[190, 185, 200, 182, 190, 195]);
    // scipy: ttest_rel(before, after) → (3.258473117707668, 0.022483670687634263)
    let r = t_test_paired(&ctx, &before, &after, Alternative::TwoSided).unwrap();
    close(s_of(&r), 3.258_473_117_707_668, 1e-12);
    close(p_of(&r), 0.022_483_670_687_634_263, 1e-12);
    // scipy: ttest_rel(before, after, alternative='greater').pvalue = 0.011241835343817131
    let r = t_test_paired(&ctx, &before, &after, Alternative::Greater).unwrap();
    close(p_of(&r), 0.011_241_835_343_817_131, 1e-12);
    let d: Vec<Q> = before.iter().zip(&after).map(|(a, b)| a - b).collect();
    let one = t_test_one_sample(&ctx, &d, &qi(0), Alternative::Greater).unwrap();
    assert_eq!(one, r);
}

#[test]
fn t_tests_reject_degenerate_input() {
    let ctx = Context::new();
    is_invalid(t_test_one_sample(
        &ctx,
        &from_i64(&[3, 3, 3]),
        &qi(0),
        Alternative::TwoSided,
    ));
    is_invalid(t_test_one_sample(
        &ctx,
        &from_i64(&[3]),
        &qi(0),
        Alternative::TwoSided,
    ));
    is_invalid(t_test_two_sample(
        &ctx,
        &from_i64(&[1, 1]),
        &from_i64(&[2, 2]),
        true,
        Alternative::TwoSided,
    ));
    is_invalid(t_test_two_sample(
        &ctx,
        &from_i64(&[1]),
        &from_i64(&[2, 3]),
        false,
        Alternative::TwoSided,
    ));
    is_invalid(t_test_paired(
        &ctx,
        &from_i64(&[1, 2]),
        &from_i64(&[2, 3, 4]),
        Alternative::TwoSided,
    ));
    is_invalid(t_test_paired(
        &ctx,
        &from_i64(&[1, 2, 3]),
        &from_i64(&[2, 3, 4]),
        Alternative::TwoSided,
    ));
    // One constant sample is fine when the other varies.
    assert!(
        t_test_two_sample(
            &ctx,
            &from_i64(&[1, 1, 1]),
            &from_i64(&[2, 3, 5]),
            false,
            Alternative::TwoSided
        )
        .is_ok()
    );
}

#[test]
fn z_tests_for_proportions_match_statsmodels() {
    let ctx = Context::new();
    // statsmodels: proportions_ztest(60, 100, value=0.5, prop_var=0.5) = (2.0, 0.04550026389635844)
    let r = z_test_proportion(&ctx, 60, 100, &q(1, 2), Alternative::TwoSided).unwrap();
    assert_eq!(r.statistic, ctx.int(2));
    assert!(format!("{}", r.p_value).contains("erfc"));
    close(p_of(&r), 0.045_500_263_896_358_44, 1e-12);
    // statsmodels: proportions_ztest(35, 100, value=0.5, prop_var=0.5, alternative='smaller') = (-3.0, 0.001349898031630093)
    let r = z_test_proportion(&ctx, 35, 100, &q(1, 2), Alternative::Less).unwrap();
    assert_eq!(r.statistic, ctx.int(-3));
    close(p_of(&r), 0.001_349_898_031_630_093, 1e-12);
    // statsmodels: proportions_ztest(12, 40, value=0.2, prop_var=0.2, alternative='larger') = (1.581138830084189, 0.056923149003329086)
    let r = z_test_proportion(&ctx, 12, 40, &q(1, 5), Alternative::Greater).unwrap();
    close(s_of(&r), 1.581_138_830_084_189, 1e-12);
    close(p_of(&r), 0.056_923_149_003_329_086, 1e-12);
    // statsmodels: proportions_ztest([45, 30], [100, 100]) = (2.1908902300206647, 0.028459736916310555)
    let r = two_proportion_z_test(&ctx, 45, 100, 30, 100, Alternative::TwoSided).unwrap();
    close(s_of(&r), 2.190_890_230_020_664_7, 1e-12);
    close(p_of(&r), 0.028_459_736_916_310_555, 1e-12);
    // statsmodels: proportions_ztest([45, 30], [100, 100], alternative='larger').pvalue = 0.014229868458155277
    let r = two_proportion_z_test(&ctx, 45, 100, 30, 100, Alternative::Greater).unwrap();
    close(p_of(&r), 0.014_229_868_458_155_277, 1e-12);
    // statsmodels: proportions_ztest([12, 20], [50, 60]) = (-1.0731772461757665, 0.28319159771497027)
    let r = two_proportion_z_test(&ctx, 12, 50, 20, 60, Alternative::TwoSided).unwrap();
    close(s_of(&r), -1.073_177_246_175_766_5, 1e-12);
    close(p_of(&r), 0.283_191_597_714_970_27, 1e-12);
    is_invalid(z_test_proportion(
        &ctx,
        3,
        10,
        &qi(1),
        Alternative::TwoSided,
    ));
    is_invalid(two_proportion_z_test(
        &ctx,
        0,
        10,
        0,
        10,
        Alternative::TwoSided,
    ));
}

#[test]
fn anova_one_way_matches_scipy_with_exact_sums_of_squares() {
    let ctx = Context::new();
    let g = [
        from_i64(&[6, 8, 4, 5, 3, 4]),
        from_i64(&[8, 12, 9, 11, 6, 8]),
        from_i64(&[13, 9, 11, 8, 7, 12]),
    ];
    // scipy: f_oneway(*g) → statistic 9.264705882352942 (= 315/34), pvalue 0.0023987773293929083
    let r = anova_one_way(&ctx, &g).unwrap();
    assert_eq!(r.f, q(315, 34));
    assert_eq!((r.df_between, r.df_within), (2, 15));
    assert_eq!(r.ss_between, qi(84));
    assert_eq!(r.ss_within, qi(68));
    assert_eq!(r.eta_squared, q(84, 152));
    close(r.p_value_f64().unwrap(), 0.002_398_777_329_392_908_3, 1e-12);
    // The p-value agrees with the F distribution of the same family.
    let via_cdf = ctx.one()
        - Distribution::f_distribution(ctx.int(2), ctx.int(15)).cdf(&ctx.from_ratio(r.f.clone()));
    close(r.p_value_f64().unwrap(), via_cdf.eval_f64().unwrap(), 1e-14);
    // scipy: f_oneway([1, 2, 3, 4], [2, 3, 4, 5, 6], [9, 8, 7]) → (14.294117647058824 = 243/17, 0.0016082648751593702)
    let g2 = [
        from_i64(&[1, 2, 3, 4]),
        from_i64(&[2, 3, 4, 5, 6]),
        from_i64(&[9, 8, 7]),
    ];
    let r = anova_one_way(&ctx, &g2).unwrap();
    assert_eq!(r.f, q(243, 17));
    close(r.p_value_f64().unwrap(), 0.001_608_264_875_159_370_2, 1e-12);
    // Validation.
    is_invalid(anova_one_way(&ctx, &[from_i64(&[1, 2])]));
    is_invalid(anova_one_way(&ctx, &[from_i64(&[1, 2]), vec![]]));
    is_invalid(anova_one_way(&ctx, &[from_i64(&[1, 1]), from_i64(&[2, 2])]));
    is_invalid(anova_one_way(&ctx, &[from_i64(&[1]), from_i64(&[2])]));
}

#[test]
fn confidence_interval_mean_matches_scipy_t_interval() {
    let ctx = Context::new();
    // scipy: t.interval(0.95, 5, loc=mean(x), scale=sem(x)) = (5.9509296876164886, 11.049070312383511)
    let x = from_i64(&[5, 7, 8, 9, 10, 12]);
    let ci = confidence_interval_mean(&ctx, &x, 0.95).unwrap();
    close(ci.lower, 5.950_929_687_616_488_6, 1e-9);
    close(ci.upper, 11.049_070_312_383_511, 1e-9);
    // scipy: t.interval(0.99, 6, loc=mean(a), scale=sem(a)) = (18.982530848003655, 22.16032629485349)
    let a = from_i64(&[20, 22, 19, 20, 22, 20, 21]);
    let ci = confidence_interval_mean(&ctx, &a, 0.99).unwrap();
    close(ci.lower, 18.982_530_848_003_655, 1e-9);
    close(ci.upper, 22.160_326_294_853_49, 1e-9);
    is_invalid(confidence_interval_mean(&ctx, &a, 1.0));
    is_invalid(confidence_interval_mean(&ctx, &from_i64(&[1]), 0.95));
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Rank tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mann_whitney_exact_all_alternatives_match_scipy() {
    let ctx = Context::new();
    let males = from_i64(&[19, 22, 16, 29, 24]);
    let females = from_i64(&[20, 11, 17, 12]);
    // scipy: mannwhitneyu(males, females, method='exact') = (17.0, 0.1111111111111111)
    let r = mann_whitney_u(
        &ctx,
        &males,
        &females,
        Alternative::TwoSided,
        RankMethod::Exact,
    )
    .unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(17)));
    assert_eq!(r.p_value_exact(), Some(q(1, 9)));
    // scipy: mannwhitneyu(..., alternative='greater', method='exact').pvalue = 0.05555555555555555 = 1/18
    let r = mann_whitney_u(
        &ctx,
        &males,
        &females,
        Alternative::Greater,
        RankMethod::Exact,
    )
    .unwrap();
    assert_eq!(r.p_value_exact(), Some(q(1, 18)));
    // scipy: mannwhitneyu(..., alternative='less', method='exact').pvalue = 0.9682539682539683 = 61/63
    let r = mann_whitney_u(&ctx, &males, &females, Alternative::Less, RankMethod::Exact).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(61, 63)));
    // Symmetry: 'less' for (x, y) is 'greater' for (y, x).
    let swapped = mann_whitney_u(
        &ctx,
        &females,
        &males,
        Alternative::Greater,
        RankMethod::Exact,
    )
    .unwrap();
    assert_eq!(swapped.p_value, r.p_value);
    assert_eq!(swapped.statistic_exact(), Some(qi(3)));
}

#[test]
fn mann_whitney_asymptotic_matches_scipy() {
    let ctx = Context::new();
    let males = from_i64(&[19, 22, 16, 29, 24]);
    let females = from_i64(&[20, 11, 17, 12]);
    let cc = RankMethod::Asymptotic { continuity: true };
    let nocc = RankMethod::Asymptotic { continuity: false };
    // scipy: mannwhitneyu(males, females, method='asymptotic').pvalue = 0.11134688653314039
    let r = mann_whitney_u(&ctx, &males, &females, Alternative::TwoSided, cc).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(17)));
    close(p_of(&r), 0.111_346_886_533_140_39, 1e-12);
    // scipy: mannwhitneyu(..., method='asymptotic', use_continuity=False).pvalue = 0.08641073297370001
    let r = mann_whitney_u(&ctx, &males, &females, Alternative::TwoSided, nocc).unwrap();
    close(p_of(&r), 0.086_410_732_973_700_01, 1e-12);
    // scipy: ... alternative='greater': cc 0.05567344326657019, no cc 0.04320536648685001
    let r = mann_whitney_u(&ctx, &males, &females, Alternative::Greater, cc).unwrap();
    close(p_of(&r), 0.055_673_443_266_570_19, 1e-12);
    let r = mann_whitney_u(&ctx, &males, &females, Alternative::Greater, nocc).unwrap();
    close(p_of(&r), 0.043_205_366_486_850_01, 1e-12);
    // scipy: ... alternative='less': cc 0.9669037101389033, no cc 0.95679463351315
    let r = mann_whitney_u(&ctx, &males, &females, Alternative::Less, cc).unwrap();
    close(p_of(&r), 0.966_903_710_138_903_3, 1e-12);
    let r = mann_whitney_u(&ctx, &males, &females, Alternative::Less, nocc).unwrap();
    close(p_of(&r), 0.956_794_633_513_15, 1e-12);
}

#[test]
fn mann_whitney_with_ties_uses_the_tie_correction() {
    let ctx = Context::new();
    let x = from_i64(&[1, 2, 2, 3, 5, 6, 6, 8]);
    let y = from_i64(&[2, 4, 6, 7, 7, 9, 10]);
    // scipy: mannwhitneyu(x, y, method='asymptotic') = (14.0, 0.11524971218079373)
    let r = mann_whitney_u(
        &ctx,
        &x,
        &y,
        Alternative::TwoSided,
        RankMethod::Asymptotic { continuity: true },
    )
    .unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(14)));
    close(p_of(&r), 0.115_249_712_180_793_73, 1e-12);
    // scipy: mannwhitneyu(x, y, method='asymptotic', use_continuity=False, alternative='less').pvalue = 0.05119627590718705
    let r = mann_whitney_u(
        &ctx,
        &x,
        &y,
        Alternative::Less,
        RankMethod::Asymptotic { continuity: false },
    )
    .unwrap();
    close(p_of(&r), 0.051_196_275_907_187_05, 1e-12);
    is_invalid(mann_whitney_u(
        &ctx,
        &x,
        &y,
        Alternative::TwoSided,
        RankMethod::Exact,
    ));
    is_invalid(mann_whitney_u(
        &ctx,
        &from_i64(&[1, 1]),
        &from_i64(&[1]),
        Alternative::TwoSided,
        RankMethod::Asymptotic { continuity: true },
    ));
    is_invalid(mann_whitney_u(
        &ctx,
        &[],
        &y,
        Alternative::TwoSided,
        RankMethod::Exact,
    ));
}

#[test]
fn wilcoxon_exact_all_alternatives_match_scipy() {
    let ctx = Context::new();
    let x = from_i64(&[125, 115, 130, 140, 140, 115, 140, 125, 140, 135]);
    let y = from_i64(&[110, 122, 125, 120, 140, 124, 123, 137, 134, 145]);
    // scipy: wilcoxon(x, y, method='exact') = (18.0, 0.65234375) = 167/256  (one zero dropped, n = 9)
    let r =
        wilcoxon_signed_rank(&ctx, &x, Some(&y), Alternative::TwoSided, RankMethod::Exact).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(18)));
    assert_eq!(r.p_value_exact(), Some(q(167, 256)));
    // scipy: wilcoxon(x, y, alternative='greater', method='exact') = (27.0, 0.326171875) = 167/512
    let r =
        wilcoxon_signed_rank(&ctx, &x, Some(&y), Alternative::Greater, RankMethod::Exact).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(27)));
    assert_eq!(r.p_value_exact(), Some(q(167, 512)));
    // scipy: wilcoxon(x, y, alternative='less', method='exact') = (27.0, 0.71484375) = 183/256
    let r = wilcoxon_signed_rank(&ctx, &x, Some(&y), Alternative::Less, RankMethod::Exact).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(183, 256)));
}

#[test]
fn wilcoxon_asymptotic_matches_scipy() {
    let ctx = Context::new();
    let x = from_i64(&[125, 115, 130, 140, 140, 115, 140, 125, 140, 135]);
    let y = from_i64(&[110, 122, 125, 120, 140, 124, 123, 137, 134, 145]);
    let cc = RankMethod::Asymptotic { continuity: true };
    let nocc = RankMethod::Asymptotic { continuity: false };
    // scipy: wilcoxon(x, y, method='asymptotic', correction=True/False) two-sided: 0.6355861222787667 / 0.5939546753269146
    let r = wilcoxon_signed_rank(&ctx, &x, Some(&y), Alternative::TwoSided, cc).unwrap();
    close(p_of(&r), 0.635_586_122_278_766_7, 1e-12);
    let r = wilcoxon_signed_rank(&ctx, &x, Some(&y), Alternative::TwoSided, nocc).unwrap();
    close(p_of(&r), 0.593_954_675_326_914_6, 1e-12);
    // greater: 0.3177930611393833 / 0.2969773376634573
    let r = wilcoxon_signed_rank(&ctx, &x, Some(&y), Alternative::Greater, cc).unwrap();
    close(p_of(&r), 0.317_793_061_139_383_3, 1e-12);
    let r = wilcoxon_signed_rank(&ctx, &x, Some(&y), Alternative::Greater, nocc).unwrap();
    close(p_of(&r), 0.296_977_337_663_457_3, 1e-12);
    // less: 0.7231915040171097 / 0.7030226623365428
    let r = wilcoxon_signed_rank(&ctx, &x, Some(&y), Alternative::Less, cc).unwrap();
    close(p_of(&r), 0.723_191_504_017_109_7, 1e-12);
    let r = wilcoxon_signed_rank(&ctx, &x, Some(&y), Alternative::Less, nocc).unwrap();
    close(p_of(&r), 0.703_022_662_336_542_8, 1e-12);
}

#[test]
fn wilcoxon_one_sample_ties_and_validation() {
    let ctx = Context::new();
    // scipy: wilcoxon([1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30], method='exact') = (0.0, 0.00390625) = 1/256
    let d = from_f64(&[1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30]).unwrap();
    let r = wilcoxon_signed_rank(&ctx, &d, None, Alternative::TwoSided, RankMethod::Exact).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(0)));
    assert_eq!(r.p_value_exact(), Some(q(1, 256)));
    let dt = from_i64(&[3, -1, 2, 2, -3, 4, 5, -2, 1]);
    // scipy: wilcoxon(dt, method='asymptotic', correction=True) = (12.0, 0.23366038889458673)
    let r = wilcoxon_signed_rank(
        &ctx,
        &dt,
        None,
        Alternative::TwoSided,
        RankMethod::Asymptotic { continuity: true },
    )
    .unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(12)));
    close(p_of(&r), 0.233_660_388_894_586_73, 1e-12);
    // scipy: wilcoxon(dt, method='asymptotic', correction=False, alternative='greater') = (33.0, 0.10555267284240172)
    let r = wilcoxon_signed_rank(
        &ctx,
        &dt,
        None,
        Alternative::Greater,
        RankMethod::Asymptotic { continuity: false },
    )
    .unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(33)));
    close(p_of(&r), 0.105_552_672_842_401_72, 1e-12);
    is_invalid(wilcoxon_signed_rank(
        &ctx,
        &dt,
        None,
        Alternative::TwoSided,
        RankMethod::Exact,
    ));
    is_invalid(wilcoxon_signed_rank(
        &ctx,
        &from_i64(&[0, 0]),
        None,
        Alternative::TwoSided,
        RankMethod::Exact,
    ));
    is_invalid(wilcoxon_signed_rank(
        &ctx,
        &dt,
        Some(&from_i64(&[1])),
        Alternative::TwoSided,
        RankMethod::Exact,
    ));
}

#[test]
fn kruskal_wallis_matches_scipy() {
    let ctx = Context::new();
    let g = [
        from_i64(&[1, 3, 5, 7, 9]),
        from_i64(&[2, 4, 6, 8, 10]),
        from_i64(&[11, 12, 13, 14, 15]),
    ];
    // scipy: kruskal(*g) → (9.5, 0.008651695203120634)
    let r = kruskal_wallis(&ctx, &g).unwrap();
    assert_eq!(r.statistic_exact(), Some(q(19, 2)));
    assert_eq!(r.df, Some(ctx.int(2)));
    close(p_of(&r), 0.008_651_695_203_120_634, 1e-12);
    // scipy: kruskal([2.9, 3.0, 2.5, 2.6, 3.2], [3.8, 2.7, 4.0, 2.4], [2.8, 3.4, 3.7, 2.2, 2.0]) → (0.7714285714285722, 0.6799647735788936)
    let g2 = [
        from_f64(&[2.9, 3.0, 2.5, 2.6, 3.2]).unwrap(),
        from_f64(&[3.8, 2.7, 4.0, 2.4]).unwrap(),
        from_f64(&[2.8, 3.4, 3.7, 2.2, 2.0]).unwrap(),
    ];
    let r = kruskal_wallis(&ctx, &g2).unwrap();
    close(s_of(&r), 0.771_428_571_428_572_2, 1e-12);
    close(p_of(&r), 0.679_964_773_578_893_6, 1e-12);
    // scipy: kruskal([1, 1, 2, 3], [2, 3, 3, 4], [4, 5, 5, 1]) → (3.7300000000000058 = 373/100, 0.1548962098849066)
    let g3 = [
        from_i64(&[1, 1, 2, 3]),
        from_i64(&[2, 3, 3, 4]),
        from_i64(&[4, 5, 5, 1]),
    ];
    let r = kruskal_wallis(&ctx, &g3).unwrap();
    assert_eq!(r.statistic_exact(), Some(q(373, 100)));
    close(p_of(&r), 0.154_896_209_884_906_6, 1e-12);
    is_invalid(kruskal_wallis(
        &ctx,
        &[from_i64(&[1, 1]), from_i64(&[1, 1])],
    ));
    is_invalid(kruskal_wallis(&ctx, &[from_i64(&[1, 2])]));
    is_invalid(kruskal_wallis(&ctx, &[from_i64(&[1, 2]), vec![]]));
}

#[test]
fn friedman_matches_scipy() {
    let ctx = Context::new();
    // scipy: friedmanchisquare([10, 9, 7, 8], [9, 8, 5, 6], [8, 6, 4, 3]) → (8.0, 0.018315638888734182)
    let blocks = [
        from_i64(&[10, 9, 8]),
        from_i64(&[9, 8, 6]),
        from_i64(&[7, 5, 4]),
        from_i64(&[8, 6, 3]),
    ];
    let r = friedman(&ctx, &blocks).unwrap();
    assert_eq!(r.statistic_exact(), Some(qi(8)));
    assert_eq!(r.df, Some(ctx.int(2)));
    close(p_of(&r), 0.018_315_638_888_734_182, 1e-12);
    // scipy: friedmanchisquare([7, 8, 6, 9, 5], [5, 8, 4, 7, 6], [6, 6, 5, 8, 7]) → (2.6315789473684212 = 50/19, 0.2682624534699609)
    let blocks = [
        from_i64(&[7, 5, 6]),
        from_i64(&[8, 8, 6]),
        from_i64(&[6, 4, 5]),
        from_i64(&[9, 7, 8]),
        from_i64(&[5, 6, 7]),
    ];
    let r = friedman(&ctx, &blocks).unwrap();
    assert_eq!(r.statistic_exact(), Some(q(50, 19)));
    close(p_of(&r), 0.268_262_453_469_960_9, 1e-12);
    is_invalid(friedman(&ctx, &[from_i64(&[1, 2]), from_i64(&[2, 1])]));
    is_invalid(friedman(&ctx, &[from_i64(&[1, 2, 3]), from_i64(&[2, 1])]));
    is_invalid(friedman(
        &ctx,
        &[from_i64(&[1, 1, 1]), from_i64(&[2, 2, 2])],
    ));
}

#[test]
fn spearman_test_matches_scipy() {
    let ctx = Context::new();
    let x = from_i64(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let y = from_i64(&[2, 1, 4, 3, 7, 8, 5, 6]);
    // scipy: spearmanr(x, y) → (0.7619047619047621 = 16/21, 0.028004939153071815)
    let r = spearman_test(&ctx, &x, &y, Alternative::TwoSided).unwrap();
    assert_eq!(r.statistic, ctx.from_ratio(q(16, 21)));
    assert_eq!(r.df, Some(ctx.int(6)));
    close(p_of(&r), 0.028_004_939_153_071_815, 1e-12);
    // scipy: spearmanr(x, y, alternative='greater').pvalue = 0.014002469576535908; 'less' 0.985997530423464
    let r = spearman_test(&ctx, &x, &y, Alternative::Greater).unwrap();
    close(p_of(&r), 0.014_002_469_576_535_908, 1e-12);
    let r = spearman_test(&ctx, &x, &y, Alternative::Less).unwrap();
    close(p_of(&r), 0.985_997_530_423_464, 1e-12);
    // scipy: spearmanr([12, 2, 1, 12, 2], [1, 4, 7, 1, 0]) → (-0.5407380704358752, 0.3467146139768868)  (ties)
    let r = spearman_test(
        &ctx,
        &from_i64(&[12, 2, 1, 12, 2]),
        &from_i64(&[1, 4, 7, 1, 0]),
        Alternative::TwoSided,
    )
    .unwrap();
    close(s_of(&r), -0.540_738_070_435_875_2, 1e-12);
    close(p_of(&r), 0.346_714_613_976_886_8, 1e-12);
    // scipy: spearmanr([1..5], [5..1]) → (-0.9999999999999999, 1.4042654220543602e-24): here exactly ρ = −1, p = 0.
    let r = spearman_test(
        &ctx,
        &from_i64(&[1, 2, 3, 4, 5]),
        &from_i64(&[5, 4, 3, 2, 1]),
        Alternative::TwoSided,
    )
    .unwrap();
    assert_eq!(r.statistic, ctx.int(-1));
    assert_eq!(r.p_value, ctx.zero());
    let r = spearman_test(
        &ctx,
        &from_i64(&[1, 2, 3, 4, 5]),
        &from_i64(&[5, 4, 3, 2, 1]),
        Alternative::Greater,
    )
    .unwrap();
    assert_eq!(r.p_value, ctx.one());
    is_invalid(spearman_test(
        &ctx,
        &from_i64(&[1, 1, 1]),
        &from_i64(&[1, 2, 3]),
        Alternative::TwoSided,
    ));
    is_invalid(spearman_test(
        &ctx,
        &from_i64(&[1, 2]),
        &from_i64(&[1, 2]),
        Alternative::TwoSided,
    ));
}

#[test]
fn kendall_test_exact_and_asymptotic_match_scipy() {
    let ctx = Context::new();
    let x = from_i64(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let y = from_i64(&[2, 1, 4, 3, 7, 8, 5, 6]);
    // scipy: kendalltau(x, y, method='exact') → (0.5714285714285714 = 4/7, 0.06101190476190476 = 41/672)
    let r = kendall_test(&ctx, &x, &y, Alternative::TwoSided, true).unwrap();
    assert_eq!(r.statistic, ctx.from_ratio(q(4, 7)));
    assert_eq!(r.p_value_exact(), Some(q(41, 672)));
    // scipy: ... alternative='greater' 0.03050595238095238 = 41/1344; 'less' 0.9844246031746032 = 9923/10080
    let r = kendall_test(&ctx, &x, &y, Alternative::Greater, true).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(41, 1344)));
    let r = kendall_test(&ctx, &x, &y, Alternative::Less, true).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(9923, 10080)));
    // scipy: kendalltau(x, y, method='asymptotic'): two-sided 0.04776124267510368, greater 0.02388062133755184, less 0.9761193786624481
    let r = kendall_test(&ctx, &x, &y, Alternative::TwoSided, false).unwrap();
    close(p_of(&r), 0.047_761_242_675_103_68, 1e-12);
    let r = kendall_test(&ctx, &x, &y, Alternative::Greater, false).unwrap();
    close(p_of(&r), 0.023_880_621_337_551_84, 1e-12);
    let r = kendall_test(&ctx, &x, &y, Alternative::Less, false).unwrap();
    close(p_of(&r), 0.976_119_378_662_448_1, 1e-12);
    // scipy: kendalltau([12, 2, 1, 12, 2], [1, 4, 7, 1, 0]) → (-0.4714045207910316, 0.28274545993277467)  (ties → asymptotic)
    let xt = from_i64(&[12, 2, 1, 12, 2]);
    let yt = from_i64(&[1, 4, 7, 1, 0]);
    let r = kendall_test(&ctx, &xt, &yt, Alternative::TwoSided, false).unwrap();
    close(s_of(&r), -0.471_404_520_791_031_6, 1e-12);
    close(p_of(&r), 0.282_745_459_932_774_67, 1e-12);
    is_invalid(kendall_test(&ctx, &xt, &yt, Alternative::TwoSided, true));
    is_invalid(kendall_test(
        &ctx,
        &from_i64(&[1, 1, 1]),
        &from_i64(&[1, 2, 3]),
        Alternative::TwoSided,
        false,
    ));
    // scipy: kendalltau(xk, yk, method='exact') → (0.7333333333333333, 0.002212852733686067 = 803/362880);
    //        alternative='greater' → 0.0011064263668430336 = 803/725760
    let xk = from_f64(&[1.2, 2.3, 3.1, 4.8, 5.0, 6.2, 7.7, 8.1, 9.4, 10.0]).unwrap();
    let yk = from_f64(&[2.1, 1.9, 4.5, 3.3, 7.2, 5.5, 8.8, 9.9, 10.5, 8.0]).unwrap();
    let r = kendall_test(&ctx, &xk, &yk, Alternative::TwoSided, true).unwrap();
    assert_eq!(r.statistic, ctx.from_ratio(q(11, 15)));
    assert_eq!(r.p_value_exact(), Some(q(803, 362_880)));
    let r = kendall_test(&ctx, &xk, &yk, Alternative::Greater, true).unwrap();
    assert_eq!(r.p_value_exact(), Some(q(803, 725_760)));
}

#[test]
fn ks_one_sample_matches_scipy_asymptotic() {
    let ctx = Context::new();
    let x = from_f64(&[-1.2, -0.3, 0.1, 0.4, 0.9, 1.5, 2.2, -0.7]).unwrap();
    let normal = Distribution::normal(ctx.int(0), ctx.int(1));
    // scipy: ks_1samp(x, norm.cdf, method='asymp') → (0.19093987465324047, 0.9324475218943081)
    let r = ks_one_sample(&x, &normal, Alternative::TwoSided).unwrap();
    close(r.statistic, 0.190_939_874_653_240_47, 1e-12);
    close(r.p_value, 0.932_447_521_894_308_1, 1e-9);
    // scipy: ... alternative='greater' → (0.01390344751349859, 0.984685645357474)
    let r = ks_one_sample(&x, &normal, Alternative::Greater).unwrap();
    close(r.statistic, 0.013_903_447_513_498_59, 1e-12);
    close(r.p_value, 0.984_685_645_357_474, 1e-9);
    // scipy: ... alternative='less' → (0.19093987465324047, 0.49892930907674393)
    let r = ks_one_sample(&x, &normal, Alternative::Less).unwrap();
    close(r.statistic, 0.190_939_874_653_240_47, 1e-12);
    close(r.p_value, 0.498_929_309_076_743_93, 1e-9);
    // scipy: ks_1samp([0.1, 0.2, 0.25, 0.3, 0.9], uniform.cdf, method='asymp') → (0.5, 0.1640791977266521)
    let u = from_f64(&[0.1, 0.2, 0.25, 0.3, 0.9]).unwrap();
    let r = ks_one_sample(
        &u,
        &Distribution::uniform(ctx.int(0), ctx.int(1)),
        Alternative::TwoSided,
    )
    .unwrap();
    close(r.statistic, 0.5, 1e-15);
    close(r.p_value, 0.164_079_197_726_652_1, 1e-9);
    // scipy: ks_1samp([0.1, 0.2, 0.25, 0.3, 0.9], expon.cdf, method='asymp') → (0.5408182206817179, 0.10732958650768845)
    let r = ks_one_sample(
        &u,
        &Distribution::exponential(ctx.int(1)),
        Alternative::TwoSided,
    )
    .unwrap();
    close(r.statistic, 0.540_818_220_681_717_9, 1e-12);
    close(r.p_value, 0.107_329_586_507_688_45, 1e-9);
    is_invalid(ks_one_sample(&[], &normal, Alternative::TwoSided));
    is_invalid(ks_one_sample(
        &u,
        &Distribution::binomial(ctx.int(3), ctx.rational(1, 2)),
        Alternative::TwoSided,
    ));
}

// ── Identities across alternatives and argument order ──────────────────────────

#[test]
fn one_sided_exact_tails_are_complementary_up_to_the_point_mass() {
    let ctx = Context::new();
    // Binomial: P(X ≤ k) + P(X ≥ k) = 1 + P(X = k), with P(X = 7) = C(10, 7)/2¹⁰ = 15/128.
    let less = binomial_test(&ctx, 7, 10, &q(1, 2), Alternative::Less).unwrap();
    let greater = binomial_test(&ctx, 7, 10, &q(1, 2), Alternative::Greater).unwrap();
    assert_eq!(
        less.p_value_exact().unwrap() + greater.p_value_exact().unwrap(),
        qi(1) + q(15, 128)
    );
    // Fisher: the same identity with the hypergeometric point mass at a = 8.
    let t = [[8, 2], [1, 5]];
    let less = fisher_exact(&ctx, t, Alternative::Less).unwrap();
    let greater = fisher_exact(&ctx, t, Alternative::Greater).unwrap();
    // scipy: hypergeom.pmf(8, 16, 10, 9) = 0.023601398601398604 = 27/1144
    let h = Distribution::hypergeometric(ctx.int(16), ctx.int(10), ctx.int(9));
    let at = (h.cdf(&ctx.int(8)) - h.cdf(&ctx.int(7)))
        .simplify()
        .as_rational()
        .unwrap();
    assert_eq!(at, q(27, 1144));
    assert_eq!(
        less.p_value_exact().unwrap() + greater.p_value_exact().unwrap(),
        qi(1) + at
    );
    // Continuous references: P(T ≥ t) + P(T ≤ t) = 1 and P(Z ≥ z) + P(Z ≤ z) = 1.
    let x = from_i64(&[5, 7, 8, 9, 10, 12]);
    let g = t_test_one_sample(&ctx, &x, &qi(6), Alternative::Greater).unwrap();
    let l = t_test_one_sample(&ctx, &x, &qi(6), Alternative::Less).unwrap();
    close(p_of(&g) + p_of(&l), 1.0, 1e-15);
    let g = z_test_proportion(&ctx, 12, 40, &q(1, 5), Alternative::Greater).unwrap();
    let l = z_test_proportion(&ctx, 12, 40, &q(1, 5), Alternative::Less).unwrap();
    close(p_of(&g) + p_of(&l), 1.0, 1e-15);
}

#[test]
fn two_sample_tests_are_symmetric_under_swapping_the_samples() {
    let ctx = Context::new();
    let x = from_i64(&[125, 115, 130, 140, 140, 115, 140, 125, 140, 135]);
    let y = from_i64(&[110, 122, 125, 120, 140, 124, 123, 137, 134, 145]);
    for method in [
        RankMethod::Exact,
        RankMethod::Asymptotic { continuity: true },
    ] {
        let g = wilcoxon_signed_rank(&ctx, &x, Some(&y), Alternative::Greater, method).unwrap();
        let l = wilcoxon_signed_rank(&ctx, &y, Some(&x), Alternative::Less, method).unwrap();
        assert_eq!(g.p_value, l.p_value);
        let two = wilcoxon_signed_rank(&ctx, &x, Some(&y), Alternative::TwoSided, method).unwrap();
        let two_swapped =
            wilcoxon_signed_rank(&ctx, &y, Some(&x), Alternative::TwoSided, method).unwrap();
        assert_eq!(two.p_value, two_swapped.p_value);
        assert_eq!(two.statistic, two_swapped.statistic);
    }
    let a = from_i64(&[20, 22, 19, 20, 22, 20, 21]);
    let b = from_i64(&[28, 32, 36, 24, 29, 32]);
    for equal_var in [true, false] {
        let ab = t_test_two_sample(&ctx, &a, &b, equal_var, Alternative::Less).unwrap();
        let ba = t_test_two_sample(&ctx, &b, &a, equal_var, Alternative::Greater).unwrap();
        assert_eq!(ab.p_value, ba.p_value);
        assert_eq!(ab.df, ba.df);
        assert_eq!((-&ab.statistic).simplify(), ba.statistic);
    }
    let g = two_proportion_z_test(&ctx, 45, 100, 30, 100, Alternative::Greater).unwrap();
    let l = two_proportion_z_test(&ctx, 30, 100, 45, 100, Alternative::Less).unwrap();
    assert_eq!(g.p_value, l.p_value);
}

#[test]
fn discrete_exact_p_values_are_rationals_and_continuous_ones_are_symbolic() {
    let ctx = Context::new();
    let males = from_i64(&[19, 22, 16, 29, 24]);
    let females = from_i64(&[20, 11, 17, 12]);
    let exact = mann_whitney_u(
        &ctx,
        &males,
        &females,
        Alternative::TwoSided,
        RankMethod::Exact,
    )
    .unwrap();
    assert!(exact.p_value_exact().is_some());
    let asymptotic = mann_whitney_u(
        &ctx,
        &males,
        &females,
        Alternative::TwoSided,
        RankMethod::Asymptotic { continuity: true },
    )
    .unwrap();
    assert!(asymptotic.p_value_exact().is_none());
    assert!(format!("{}", asymptotic.p_value).contains("erfc"));
    let kw = kruskal_wallis(&ctx, &[males.clone(), females.clone(), from_i64(&[30, 31])]).unwrap();
    assert!(kw.p_value_exact().is_none());
    assert!(format!("{}", kw.p_value).contains("uppergamma"));
    // An F tail with d₁ = 2 and even d₂ folds to an exact rational: I_z(d₂/2, 1) is a
    // polynomial in the rational z.  Group sizes 4, 4, 3 give df = (2, 8).
    let folded = anova_one_way(
        &ctx,
        &[
            from_i64(&[1, 2, 3, 4]),
            from_i64(&[2, 3, 4, 5]),
            from_i64(&[9, 8, 7]),
        ],
    )
    .unwrap();
    assert_eq!((folded.df_between, folded.df_within), (2, 8));
    let exact = folded.p_value.eval().as_rational();
    assert!(exact.is_some(), "{}", folded.p_value.eval());
    // scipy: f_oneway([1, 2, 3, 4], [2, 3, 4, 5], [9, 8, 7]) → (18.848484848484837 = 622/33, 0.0009393130181095339)
    assert_eq!(folded.f, q(622, 33));
    close(
        folded.p_value_f64().unwrap(),
        0.000_939_313_018_109_533_9,
        1e-12,
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Effect sizes
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cohens_d_hedges_g_glass_delta_match_numpy() {
    let ctx = Context::new();
    let x = from_i64(&[20, 22, 19, 20, 22, 20, 21]);
    let y = from_i64(&[28, 32, 36, 24, 29, 32]);
    // numpy: (mean(x) − mean(y)) / sqrt(((n1−1)var(x, ddof=1) + (n2−1)var(y, ddof=1))/(n1+n2−2)) = -3.3080302571461795
    let d = cohens_d(&ctx, &x, &y, true).unwrap();
    close(d.eval_f64().unwrap(), -3.308_030_257_146_179_5, 1e-12);
    // numpy: (mean(x) − mean(y)) / sqrt((var(x, ddof=1) + var(y, ddof=1))/2) = -3.1762230367995814
    let d = cohens_d(&ctx, &x, &y, false).unwrap();
    close(d.eval_f64().unwrap(), -3.176_223_036_799_581_4, 1e-12);
    // numpy: d · (1 − 3/(4·13 − 9)) = -3.077237448508074
    let g = hedges_g(&ctx, &x, &y).unwrap();
    close(g.eval_f64().unwrap(), -3.077_237_448_508_074, 1e-12);
    // numpy: (mean(x) − mean(y)) / std(y, ddof=1) = -2.329471985471971
    let g = glass_delta(&ctx, &x, &y).unwrap();
    close(g.eval_f64().unwrap(), -2.329_471_985_471_971, 1e-12);
    // Exactness: d² is rational.
    let d = cohens_d(&ctx, &x, &y, true).unwrap();
    assert!(d.powi(2).simplify().as_rational().is_some());
    is_invalid(cohens_d(&ctx, &from_i64(&[1, 1]), &from_i64(&[2, 2]), true));
    is_invalid(glass_delta(&ctx, &x, &from_i64(&[2, 2])));
}

#[test]
fn rank_biserial_equals_cliffs_delta() {
    let ctx = Context::new();
    let males = from_i64(&[19, 22, 16, 29, 24]);
    let females = from_i64(&[20, 11, 17, 12]);
    let u = mann_whitney_u(
        &ctx,
        &males,
        &females,
        Alternative::TwoSided,
        RankMethod::Exact,
    )
    .unwrap();
    let r = rank_biserial(&u.statistic_exact().unwrap(), 5, 4).unwrap();
    assert_eq!(r, q(7, 10));
    assert_eq!(cliffs_delta(&males, &females).unwrap(), q(7, 10));
    // With ties U counts each tie as ½, so 2U/(n₁n₂) − 1 is still P(X > Y) − P(X < Y).
    let x = from_i64(&[1, 2, 2, 3, 5, 6, 6, 8]);
    let y = from_i64(&[2, 4, 6, 7, 7, 9, 10]);
    let u = mann_whitney_u(
        &ctx,
        &x,
        &y,
        Alternative::TwoSided,
        RankMethod::Asymptotic { continuity: true },
    )
    .unwrap();
    assert_eq!(
        rank_biserial(&u.statistic_exact().unwrap(), 8, 7).unwrap(),
        cliffs_delta(&x, &y).unwrap()
    );
    is_invalid(rank_biserial(&qi(21), 5, 4));
    is_invalid(cliffs_delta(&[], &y));
}

#[test]
fn eta_squared_matches_anova() {
    let g = [
        from_i64(&[6, 8, 4, 5, 3, 4]),
        from_i64(&[8, 12, 9, 11, 6, 8]),
        from_i64(&[13, 9, 11, 8, 7, 12]),
    ];
    assert_eq!(eta_squared(&g).unwrap(), q(84, 152));
    is_invalid(eta_squared(&[from_i64(&[1, 1]), from_i64(&[1, 1])]));
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Multiple comparisons
// ═══════════════════════════════════════════════════════════════════════════

fn assert_adjusted(a: &Adjusted, p: &[f64], reject: &[bool]) {
    assert_eq!(a.p_adjusted.len(), p.len());
    for (i, (got, want)) in a.p_adjusted.iter().zip(p).enumerate() {
        assert!(
            (got - want).abs() < 1e-12,
            "p_adjusted[{i}]: expected {want}, got {got}"
        );
    }
    assert_eq!(a.reject, reject);
}

#[test]
fn multiple_comparisons_small_example_matches_statsmodels() {
    let p = [0.01, 0.04, 0.03, 0.2];
    // statsmodels: multipletests(p, method='bonferroni') → [0.04, 0.16, 0.12, 0.8], reject [T, F, F, F]
    assert_adjusted(
        &bonferroni(&p, 0.05).unwrap(),
        &[0.04, 0.16, 0.12, 0.8],
        &[true, false, false, false],
    );
    // statsmodels: multipletests(p, method='holm') → [0.04, 0.09, 0.09, 0.2], reject [T, F, F, F]
    assert_adjusted(
        &holm(&p, 0.05).unwrap(),
        &[0.04, 0.09, 0.09, 0.2],
        &[true, false, false, false],
    );
    // statsmodels: multipletests(p, method='fdr_bh') → [0.04, 0.05333333333333334, 0.05333333333333334, 0.2], reject [T, F, F, F]
    assert_adjusted(
        &benjamini_hochberg(&p, 0.05).unwrap(),
        &[
            0.04,
            0.053_333_333_333_333_34,
            0.053_333_333_333_333_34,
            0.2,
        ],
        &[true, false, false, false],
    );
    // statsmodels: multipletests(p, method='fdr_by') → [0.08333333333333331, 0.1111111111111111, 0.1111111111111111, 0.41666666666666663], reject all F
    assert_adjusted(
        &benjamini_yekutieli(&p, 0.05).unwrap(),
        &[
            0.083_333_333_333_333_31,
            0.111_111_111_111_111_1,
            0.111_111_111_111_111_1,
            0.416_666_666_666_666_63,
        ],
        &[false; 4],
    );
}

#[test]
fn multiple_comparisons_25_pvalues_match_statsmodels() {
    let p = [
        0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216, 0.222, 0.251, 0.269,
        0.275, 0.34, 0.341, 0.384, 0.569, 0.594, 0.696, 0.762, 0.94, 0.942, 0.975, 0.986,
    ];
    let mut only_first = [false; 25];
    only_first[0] = true;
    // statsmodels: multipletests(p, method='bonferroni')
    let mut want = [1.0; 25];
    want[..3].copy_from_slice(&[0.025, 0.2, 0.975]);
    assert_adjusted(&bonferroni(&p, 0.05).unwrap(), &want, &only_first);
    // statsmodels: multipletests(p, method='holm')
    let mut want = [1.0; 25];
    want[..5].copy_from_slice(&[0.025, 0.192, 0.897, 0.902, 0.902]);
    assert_adjusted(&holm(&p, 0.05).unwrap(), &want, &only_first);
    // statsmodels: multipletests(p, method='fdr_bh')
    let want = [
        0.025,
        0.1,
        0.21,
        0.21,
        0.21,
        0.25,
        0.264_285_714_285_714_23,
        0.491_071_428_571_428_55,
        0.491_071_428_571_428_55,
        0.491_071_428_571_428_55,
        0.491_071_428_571_428_55,
        0.491_071_428_571_428_55,
        0.491_071_428_571_428_55,
        0.491_071_428_571_428_55,
        0.532_812_5,
        0.532_812_5,
        0.564_705_882_352_941_2,
        0.781_578_947_368_421,
        0.781_578_947_368_421,
        0.869_999_999_999_999_9,
        0.907_142_857_142_857_1,
        0.986,
        0.986,
        0.986,
        0.986,
    ];
    assert_adjusted(&benjamini_hochberg(&p, 0.05).unwrap(), &want, &only_first);
    // statsmodels: multipletests(p, method='fdr_by')
    let mut want = [1.0; 25];
    want[..6].copy_from_slice(&[
        0.095_398_954_443_837_67,
        0.381_595_817_775_350_7,
        0.801_351_217_328_236_4,
        0.801_351_217_328_236_4,
        0.801_351_217_328_236_4,
        0.953_989_544_438_376_7,
    ]);
    assert_adjusted(&benjamini_yekutieli(&p, 0.05).unwrap(), &want, &[false; 25]);
}

#[test]
fn multiple_comparisons_handle_ties_and_validate() {
    let p = [0.02, 0.02, 0.5, 0.001];
    // statsmodels: multipletests(p, method='holm') → [0.06, 0.06, 0.5, 0.004], reject [F, F, F, T]
    assert_adjusted(
        &holm(&p, 0.05).unwrap(),
        &[0.06, 0.06, 0.5, 0.004],
        &[false, false, false, true],
    );
    // statsmodels: multipletests(p, method='fdr_bh') → [0.02666666666666667, 0.02666666666666667, 0.5, 0.004], reject [T, T, F, T]
    assert_adjusted(
        &benjamini_hochberg(&p, 0.05).unwrap(),
        &[
            0.026_666_666_666_666_67,
            0.026_666_666_666_666_67,
            0.5,
            0.004,
        ],
        &[true, true, false, true],
    );
    is_invalid(bonferroni(&[], 0.05));
    is_invalid(holm(&[0.5, 1.5], 0.05));
    is_invalid(benjamini_hochberg(&[0.5], 0.0));
    is_invalid(benjamini_yekutieli(&[0.5, f64::NAN], 0.05));
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Resampling
// ═══════════════════════════════════════════════════════════════════════════

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

#[test]
fn bootstrap_ci_is_deterministic_and_covers_the_true_mean() {
    let ctx = Context::new();
    // Normal(10, 2), n = 400 — the 95% interval of the mean should contain 10 and be about ±0.2 wide.
    let normal = Distribution::normal(ctx.int(10), ctx.int(2));
    let data = normal.sample(400, &mut Rng::new(2024)).unwrap();
    let ci = bootstrap_ci(
        &data,
        mean,
        2000,
        0.95,
        &mut Rng::new(1),
        BootstrapMethod::Percentile,
    )
    .unwrap();
    assert!(ci.lower < 10.0 && 10.0 < ci.upper, "{ci}");
    assert!(
        ci.width() > 0.25 && ci.width() < 0.55,
        "width {}",
        ci.width()
    );
    // Deterministic under a fixed seed.
    let again = bootstrap_ci(
        &data,
        mean,
        2000,
        0.95,
        &mut Rng::new(1),
        BootstrapMethod::Percentile,
    )
    .unwrap();
    assert_eq!(ci, again);
    // The basic interval is the percentile interval reflected about the estimate.
    let basic = bootstrap_ci(
        &data,
        mean,
        2000,
        0.95,
        &mut Rng::new(1),
        BootstrapMethod::Basic,
    )
    .unwrap();
    let m = mean(&data);
    close(basic.lower, 2.0 * m - ci.upper, 1e-12);
    close(basic.upper, 2.0 * m - ci.lower, 1e-12);
}

#[test]
fn bootstrap_interval_width_shrinks_like_inverse_sqrt_n() {
    let ctx = Context::new();
    let normal = Distribution::normal(ctx.int(0), ctx.int(1));
    let small = normal.sample(100, &mut Rng::new(5)).unwrap();
    let large = normal.sample(1600, &mut Rng::new(6)).unwrap();
    let small_ci = bootstrap_ci(
        &small,
        mean,
        4000,
        0.9,
        &mut Rng::new(3),
        BootstrapMethod::Percentile,
    )
    .unwrap();
    let large_ci = bootstrap_ci(
        &large,
        mean,
        4000,
        0.9,
        &mut Rng::new(4),
        BootstrapMethod::Percentile,
    )
    .unwrap();
    let ratio = small_ci.width() / large_ci.width();
    // 16× the data → about 4× narrower.
    assert!(ratio > 3.0 && ratio < 5.2, "ratio {ratio}");
}

#[test]
fn bootstrap_validates_input() {
    let mut rng = Rng::new(0);
    is_invalid(bootstrap_ci(
        &[],
        mean,
        10,
        0.95,
        &mut rng,
        BootstrapMethod::Percentile,
    ));
    is_invalid(bootstrap_ci(
        &[1.0, 2.0],
        mean,
        0,
        0.95,
        &mut rng,
        BootstrapMethod::Percentile,
    ));
    is_invalid(bootstrap_ci(
        &[1.0, 2.0],
        mean,
        10,
        1.0,
        &mut rng,
        BootstrapMethod::Percentile,
    ));
    is_invalid(bootstrap_ci(
        &[1.0, f64::NAN],
        mean,
        10,
        0.95,
        &mut rng,
        BootstrapMethod::Percentile,
    ));
    // A single observation gives a degenerate but valid interval.
    assert_eq!(
        bootstrap_ci(&[3.0], mean, 10, 0.95, &mut rng, BootstrapMethod::Basic).unwrap(),
        Interval {
            lower: 3.0,
            upper: 3.0
        }
    );
}

#[test]
fn permutation_test_detects_a_shift_and_is_deterministic() {
    let diff = |a: &[f64], b: &[f64]| mean(a) - mean(b);
    let x = [1.1, 2.3, 1.9, 2.8, 2.2, 1.7];
    let y = [4.9, 5.2, 4.4, 5.8, 5.1, 4.7];
    // scipy: permutation_test((x, y), mean difference, n_resamples=100000).pvalue = 0.0021645021645021645
    // (the exact p is 2/C(12,6) = 1/462 ≈ 0.00216; a randomised test with 5000 draws lands near it)
    let r = permutation_test(&x, &y, diff, 5000, &mut Rng::new(11), Alternative::TwoSided).unwrap();
    close(r.statistic, mean(&x) - mean(&y), 1e-12);
    assert!(r.p_value < 0.01, "{}", r.p_value);
    let again =
        permutation_test(&x, &y, diff, 5000, &mut Rng::new(11), Alternative::TwoSided).unwrap();
    assert_eq!(r, again);
    // One-sided in the wrong direction: p ≈ 1.
    let r = permutation_test(&x, &y, diff, 2000, &mut Rng::new(11), Alternative::Greater).unwrap();
    assert!(r.p_value > 0.99, "{}", r.p_value);
    // Two draws from the same distribution: nothing to detect.
    let ctx = Context::new();
    let normal = Distribution::normal(ctx.int(0), ctx.int(1));
    let a = normal.sample(40, &mut Rng::new(21)).unwrap();
    let b = normal.sample(40, &mut Rng::new(22)).unwrap();
    let r = permutation_test(&a, &b, diff, 3000, &mut Rng::new(12), Alternative::TwoSided).unwrap();
    assert!(r.p_value > 0.05, "{}", r.p_value);
    is_invalid(permutation_test(
        &[],
        &y,
        diff,
        10,
        &mut Rng::new(0),
        Alternative::TwoSided,
    ));
    is_invalid(permutation_test(
        &x,
        &y,
        diff,
        0,
        &mut Rng::new(0),
        Alternative::TwoSided,
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Power and sample size
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sample_size_for_proportion_matches_statsmodels() {
    // statsmodels: samplesize_confint_proportion(0.5, 0.03) = 1067.0718946372576 → 1068
    assert_eq!(sample_size_for_proportion(0.03, 0.95, 0.5).unwrap(), 1068);
    // statsmodels: samplesize_confint_proportion(0.2, 0.05, alpha=0.01) = 424.6333824653579 → 425
    assert_eq!(sample_size_for_proportion(0.05, 0.99, 0.2).unwrap(), 425);
    is_invalid(sample_size_for_proportion(0.0, 0.95, 0.5));
    is_invalid(sample_size_for_proportion(0.05, 0.95, 1.0));
    is_invalid(sample_size_for_proportion(0.05, 1.5, 0.5));
}

#[test]
fn power_two_proportions_matches_statsmodels_normal_ind_power() {
    // statsmodels: NormalIndPower().power(proportion_effectsize(0.5, 0.4), nobs1=200, alpha=0.05, ratio=1) = 0.5214145419211713
    close(
        power_two_proportions(0.5, 0.4, 200, 0.05).unwrap(),
        0.521_414_541_921_171_3,
        1e-12,
    );
    // statsmodels: ... nobs1=50 → 0.17175567262143845
    close(
        power_two_proportions(0.5, 0.4, 50, 0.05).unwrap(),
        0.171_755_672_621_438_45,
        1e-12,
    );
    // statsmodels: NormalIndPower().power(proportion_effectsize(0.2, 0.35), nobs1=100, alpha=0.01, ratio=1) = 0.4285406038297726
    close(
        power_two_proportions(0.2, 0.35, 100, 0.01).unwrap(),
        0.428_540_603_829_772_6,
        1e-12,
    );
    is_invalid(power_two_proportions(0.5, 0.4, 0, 0.05));
    is_invalid(power_two_proportions(0.0, 0.4, 10, 0.05));
}

#[test]
fn sample_size_two_proportions_matches_statsmodels_solve_power() {
    // statsmodels: NormalIndPower().solve_power(proportion_effectsize(0.5, 0.4), alpha=0.05, power=0.8, ratio=1) = 387.1677468578098 → 388
    let n = sample_size_two_proportions(0.5, 0.4, 0.05, 0.8).unwrap();
    assert_eq!(n, 388);
    assert!(power_two_proportions(0.5, 0.4, n - 1, 0.05).unwrap() < 0.8);
    assert!(power_two_proportions(0.5, 0.4, n, 0.05).unwrap() >= 0.8);
    // statsmodels: NormalIndPower().solve_power(proportion_effectsize(0.2, 0.35), alpha=0.01, power=0.9, ratio=1) = 259.2427151562742 → 260
    assert_eq!(
        sample_size_two_proportions(0.2, 0.35, 0.01, 0.9).unwrap(),
        260
    );
    is_invalid(sample_size_two_proportions(0.4, 0.4, 0.05, 0.8));
    is_invalid(sample_size_two_proportions(0.5, 0.4, 0.05, 0.04));
}

#[test]
fn power_t_test_two_sample_matches_statsmodels_noncentral_t() {
    // statsmodels: TTestIndPower().power(0.5, nobs1=30, alpha=0.05, ratio=1) = 0.47789652076016464
    close(
        power_t_test_two_sample(0.5, 30, 0.05).unwrap(),
        0.477_896_520_760_164_64,
        1e-9,
    );
    // statsmodels: TTestIndPower().power(0.8, nobs1=10, alpha=0.05, ratio=1) = 0.39506921211364077
    close(
        power_t_test_two_sample(0.8, 10, 0.05).unwrap(),
        0.395_069_212_113_640_77,
        1e-9,
    );
    // statsmodels: TTestIndPower().power(0.2, nobs1=200, alpha=0.01, ratio=1) = 0.27955921792884386
    close(
        power_t_test_two_sample(0.2, 200, 0.01).unwrap(),
        0.279_559_217_928_843_86,
        1e-9,
    );
    // statsmodels: TTestIndPower().power(0.5, nobs1=2, alpha=0.05, ratio=1) = 0.06150785655602509
    close(
        power_t_test_two_sample(0.5, 2, 0.05).unwrap(),
        0.061_507_856_556_025_09,
        1e-9,
    );
    // Two-sided: the sign of the effect is irrelevant.
    close(
        power_t_test_two_sample(-0.5, 30, 0.05).unwrap(),
        0.477_896_520_760_164_64,
        1e-9,
    );
    is_invalid(power_t_test_two_sample(0.5, 1, 0.05));
    is_invalid(power_t_test_two_sample(f64::INFINITY, 10, 0.05));
}

#[test]
fn sample_size_t_test_two_sample_matches_statsmodels_solve_power() {
    // statsmodels: TTestIndPower().solve_power(0.5, alpha=0.05, power=0.8, ratio=1) = 63.765610588911635 → 64
    let n = sample_size_t_test_two_sample(0.5, 0.05, 0.8).unwrap();
    assert_eq!(n, 64);
    assert!(power_t_test_two_sample(0.5, 63, 0.05).unwrap() < 0.8);
    assert!(power_t_test_two_sample(0.5, 64, 0.05).unwrap() >= 0.8);
    // statsmodels: TTestIndPower().solve_power(1.0, alpha=0.05, power=0.95, ratio=1) = 26.989203442388852 → 27
    assert_eq!(sample_size_t_test_two_sample(1.0, 0.05, 0.95).unwrap(), 27);
    // statsmodels: TTestIndPower().solve_power(0.2, alpha=0.01, power=0.9, ratio=1) = 745.6299677192546 → 746
    assert_eq!(sample_size_t_test_two_sample(0.2, 0.01, 0.9).unwrap(), 746);
    is_invalid(sample_size_t_test_two_sample(0.0, 0.05, 0.8));
    is_invalid(sample_size_t_test_two_sample(0.5, 0.05, 0.05));
}
