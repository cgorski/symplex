//! 0.18 track: layout.  The consistency pass over `symplex::stats` moved
//! items to the module whose rule names them, kept a re-export at every old
//! path for one release, and tightened the type conventions (`usize`
//! counts, `&Q` parameters, `ctx` only when an `Ex` comes back).
//!
//! One test per moved item calls it through its **new** path; one test
//! checks that every old path still compiles (so 0.19 can drop the
//! re-exports knowing exactly which names it removes); the rest cover the
//! new `counts_usize` helper and the `&Q` `Sprt` constructors.

use symplex::linprog::{Q, q, qi};
use symplex::prelude::*;
use symplex::stats::agreement::{
    RatingTable, cochrans_q, cohen_kappa_ci, cohen_kappa_maximum, confusion_matrix,
    kappa_ci_from_confusion, kappa_maximum_from_confusion, kappa_test, kappa_test_from_confusion,
};
use symplex::stats::anova::{AnovaResult, anova_one_way};
use symplex::stats::data::{
    ConcordanceCounts, Dependent, concordance_counts, from_i64, goodman_kruskal_gamma,
    kendall_tau_c, somers_d,
};
use symplex::stats::estimation::{
    IntervalMethod, confidence_interval_mean, confidence_interval_mean_z, fisher_z, pearson_ci,
    proportion_interval, proportion_interval_exact, proportion_interval_symbolic, z_for_confidence,
};
use symplex::stats::hypothesis::{
    Alternative, adjusted_residuals, chi_square_independence, chi2_contributions,
    compare_two_correlations, counts, counts_usize, expected_counts, pearson_t_statistic,
    pearson_test, standardized_residuals, z_test_two_proportions,
};
use symplex::stats::sequential::{Decision, Sprt};

fn close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() < tol,
        "expected {expected}, got {actual} (|Δ| = {:e})",
        (actual - expected).abs()
    );
}

/// Ten items rated by two raters on an ordinal scale (shared by several
/// tests below).
fn xy() -> (Vec<Q>, Vec<Q>) {
    (
        from_i64(&[1, 2, 2, 3, 3, 3, 4, 4, 5, 1, 2, 4]),
        from_i64(&[1, 1, 2, 2, 3, 2, 4, 3, 5, 2, 3, 4]),
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// hypothesis → anova
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn anova_one_way_lives_in_anova() {
    let ctx = Context::new();
    let g = [
        from_i64(&[6, 8, 4, 5, 3, 4]),
        from_i64(&[8, 12, 9, 11, 6, 8]),
        from_i64(&[13, 9, 11, 8, 7, 12]),
    ];
    // scipy: f_oneway(*g) → statistic 9.264705882352942 (= 315/34), pvalue 0.0023987773293929083
    let r: AnovaResult = anova_one_way(&ctx, &g).unwrap();
    assert_eq!(r.f, q(315, 34));
    assert_eq!((r.df_between, r.df_within), (2, 15));
    close(r.p_value_f64().unwrap(), 0.002_398_777_329_392_908_3, 1e-12);
}

// ═══════════════════════════════════════════════════════════════════════════
// reliability → agreement
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn kappa_ci_lives_in_agreement() {
    let ctx = Context::new();
    // statsmodels: cohens_kappa([[20, 5], [10, 15]], return_results=True)
    let ci = kappa_ci_from_confusion(&ctx, &[vec![20, 5], vec![10, 15]], 0.95).unwrap();
    assert_eq!((ci.kappa, ci.variance), (q(2, 5), q(252, 15625)));
    close(ci.ci.lower, 0.151_092_290_476_661, 1e-12);
    close(ci.ci.upper, 0.648_907_709_523_339, 1e-12);
    let a = from_i64(&[0, 0, 1, 1, 1, 0]);
    let b = from_i64(&[0, 1, 1, 1, 0, 0]);
    let from_ratings = cohen_kappa_ci(&ctx, &a, &b, 0.95).unwrap();
    assert_eq!(from_ratings.kappa, q(1, 3));
}

#[test]
fn kappa_test_lives_in_agreement() {
    let ctx = Context::new();
    // statsmodels: z_value 2.886751345948128, pvalue_two_sided 0.003892417122779
    let r = kappa_test_from_confusion(&ctx, &[vec![20, 5], vec![10, 15]], Alternative::TwoSided)
        .unwrap();
    close(r.statistic_f64().unwrap(), 2.886_751_345_948_128, 1e-12);
    close(r.p_value_f64().unwrap(), 0.003_892_417_122_779, 1e-12);
    let a = from_i64(&[0, 0, 1, 1, 1, 0]);
    let b = from_i64(&[0, 1, 1, 1, 0, 0]);
    assert!(kappa_test(&ctx, &a, &b, Alternative::Greater).is_ok());
}

#[test]
fn kappa_maximum_lives_in_agreement() {
    // statsmodels: cohens_kappa([[20, 5], [10, 15]], return_results=True).kappa_max = 0.8
    assert_eq!(
        kappa_maximum_from_confusion(&[vec![20, 5], vec![10, 15]]).unwrap(),
        q(4, 5)
    );
    let a = from_i64(&[0, 0, 1, 1, 1, 0]);
    let b = from_i64(&[0, 1, 1, 1, 0, 0]);
    assert_eq!(cohen_kappa_maximum(&a, &b).unwrap(), qi(1));
}

#[test]
fn cochrans_q_lives_in_agreement() {
    let ctx = Context::new();
    let t = RatingTable::from_i64(&[
        &[1, 1, 0],
        &[1, 1, 0],
        &[1, 0, 0],
        &[1, 1, 1],
        &[0, 1, 0],
        &[1, 0, 0],
        &[1, 1, 0],
        &[1, 1, 0],
        &[0, 0, 0],
        &[1, 1, 1],
        &[1, 0, 0],
        &[1, 1, 0],
    ])
    .unwrap();
    // statsmodels: cochrans_q(x) → statistic 104/9, pvalue 0.003095586852365, df 2
    let r = cochrans_q(&ctx, &t).unwrap();
    assert_eq!(r.statistic_exact(), Some(q(104, 9)));
    assert_eq!(r.df, Some(ctx.int(2)));
    close(r.p_value_f64().unwrap(), 0.003_095_586_852_365, 1e-12);
}

// ═══════════════════════════════════════════════════════════════════════════
// reliability → hypothesis
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn pearson_test_lives_in_hypothesis() {
    let ctx = Context::new();
    let x = from_i64(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let y = from_i64(&[2, 1, 4, 3, 7, 8, 5, 6, 10, 9]);
    // scipy: pearsonr(x, y) → statistic 13/15, pvalue 0.001173538180155
    let r = pearson_test(&ctx, &x, &y, Alternative::TwoSided).unwrap();
    assert_eq!(r.statistic, ctx.from_ratio(q(13, 15)));
    close(r.p_value_f64().unwrap(), 0.001_173_538_180_155, 1e-12);
}

#[test]
fn pearson_t_statistic_lives_in_hypothesis() {
    let ctx = Context::new();
    let x = from_i64(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let y = from_i64(&[2, 1, 4, 3, 7, 8, 5, 6, 10, 9]);
    // r = 13/15, t² = 169/7
    let t = pearson_t_statistic(&ctx, &x, &y).unwrap();
    close(t.eval_f64().unwrap(), 4.913_538_149_119_947, 1e-12);
}

#[test]
fn compare_two_correlations_lives_in_hypothesis() {
    let ctx = Context::new();
    // scipy.stats.norm: z = 2.251706268343729, two-sided p = 0.024340840282246236
    let r = compare_two_correlations(&ctx, 0.7, 50, 0.4, 60, Alternative::TwoSided).unwrap();
    close(r.statistic_f64().unwrap(), 2.251_706_268_343_729, 1e-12);
    close(r.p_value_f64().unwrap(), 0.024_340_840_282_246_236, 1e-12);
}

#[test]
fn expected_counts_lives_in_hypothesis() {
    let t = counts(&[&[10, 20, 30], &[6, 9, 17]]);
    // statsmodels: Table(t).fittedvalues[0][0] = 240/23
    assert_eq!(expected_counts(&t).unwrap()[0][0], q(240, 23));
}

#[test]
fn chi2_contributions_lives_in_hypothesis() {
    let ctx = Context::new();
    let t = counts(&[&[10, 20, 30], &[6, 9, 17]]);
    // statsmodels: Table(t).chi2_contribs[1][1] = 625/5336
    let c = chi2_contributions(&t).unwrap();
    assert_eq!(c[1][1], q(625, 5336));
    // They sum to the uncorrected χ².
    let total = c.iter().flatten().fold(q(0, 1), |acc, v| acc + v);
    assert_eq!(
        total,
        chi_square_independence(&ctx, &t, false).unwrap().statistic
    );
}

#[test]
fn standardized_residuals_live_in_hypothesis() {
    let ctx = Context::new();
    let t = counts(&[&[10, 20, 30], &[6, 9, 17]]);
    // statsmodels: Table(t).resid_pearson[0][1] = 0.24993752342773828
    let r = standardized_residuals(&ctx, &t).unwrap();
    close(r[0][1].eval_f64().unwrap(), 0.249_937_523_427_738_28, 1e-12);
}

#[test]
fn adjusted_residuals_live_in_hypothesis() {
    let ctx = Context::new();
    let t = counts(&[&[10, 20, 30], &[6, 9, 17]]);
    // statsmodels: Table(t).standardized_resids[0][1] = 0.5121226989905664
    let r = adjusted_residuals(&ctx, &t).unwrap();
    close(r[0][1].eval_f64().unwrap(), 0.512_122_698_990_566_4, 1e-12);
}

// ═══════════════════════════════════════════════════════════════════════════
// reliability → estimation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn pearson_ci_lives_in_estimation() {
    // scipy: pearsonr(range(1, 6), [1, 3, 2, 5, 4]).confidence_interval(0.95)
    let ci = pearson_ci(0.8, 5, 0.95).unwrap();
    close(ci.lower, -0.279_640_041_969_355, 1e-12);
    close(ci.upper, 0.986_196_193_301_271_4, 1e-12);
}

#[test]
fn fisher_z_lives_in_estimation() {
    let ctx = Context::new();
    // atanh(0.8) = ln 3
    let z = fisher_z(&ctx.rational(4, 5));
    close(z.eval_f64().unwrap(), 1.098_612_288_668_109_8, 1e-12);
}

// ═══════════════════════════════════════════════════════════════════════════
// reliability → data
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn concordance_counts_live_in_data() {
    let x = from_i64(&[1, 2, 2, 3]);
    let y = from_i64(&[1, 1, 2, 3]);
    let c = concordance_counts(&x, &y).unwrap();
    assert_eq!(
        c,
        ConcordanceCounts {
            concordant: 4,
            discordant: 0,
            ties_x: 1,
            ties_y: 1,
            ties_both: 0,
        }
    );
    assert_eq!(c.pairs(), 6);
}

#[test]
fn goodman_kruskal_gamma_lives_in_data() {
    let (x, y) = xy();
    // C = 44, D = 3
    assert_eq!(goodman_kruskal_gamma(&x, &y).unwrap(), q(41, 47));
}

#[test]
fn somers_d_lives_in_data() {
    let (x, y) = xy();
    // scipy: somersd(x, y) = 41/56; somersd(y, x) = 41/55
    assert_eq!(somers_d(&x, &y, Dependent::Y).unwrap(), q(41, 56));
    assert_eq!(somers_d(&x, &y, Dependent::X).unwrap(), q(41, 55));
}

#[test]
fn kendall_tau_c_lives_in_data() {
    let (x, y) = xy();
    // scipy: kendalltau(x, y, variant='c') = 205/288
    assert_eq!(kendall_tau_c(&x, &y).unwrap(), q(205, 288));
}

// ═══════════════════════════════════════════════════════════════════════════
// aggregation → estimation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn proportion_interval_lives_in_estimation() {
    // statsmodels: proportion_confint(3, 10, alpha=0.05, method='wilson')
    let ci = proportion_interval(3, 10, 0.95, IntervalMethod::Wilson).unwrap();
    close(ci.lower, 0.107_791_267_406_301_04, 1e-12);
    close(ci.upper, 0.603_221_852_538_854_6, 1e-12);
}

#[test]
fn proportion_interval_symbolic_lives_in_estimation() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let ci = proportion_interval_symbolic(&ctx, 3, 10, &z, IntervalMethod::Wilson).unwrap();
    assert!(format!("{}", ci.upper).contains("z^2"));
}

#[test]
fn proportion_interval_exact_lives_in_estimation() {
    let ctx = Context::new();
    // n = 1: the tail is linear and the endpoint is a rational.
    let ci =
        proportion_interval_exact(&ctx, 1, 1, &q(9, 10), IntervalMethod::ClopperPearson).unwrap();
    assert_eq!(ci.lower, ctx.rational(1, 20));
    assert_eq!(ci.upper, ctx.int(1));
}

#[test]
fn z_for_confidence_lives_in_estimation() {
    let ctx = Context::new();
    // scipy: norm.ppf(0.975) = 1.959963984540054
    let z = z_for_confidence(&ctx, &q(95, 100)).unwrap();
    assert_eq!(format!("{z}"), "sqrt(2)*erfinv(19/20)");
    close(z.eval_f64().unwrap(), 1.959_963_984_540_054, 1e-12);
}

// ═══════════════════════════════════════════════════════════════════════════
// hypothesis → estimation, and the ctx rule
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn confidence_interval_mean_lives_in_estimation_without_ctx() {
    let x = from_i64(&[5, 7, 8, 9, 10, 12]);
    // scipy: t.interval(0.95, 5, loc=mean(x), scale=sem(x)) = (5.9509296876164886, 11.049070312383511)
    let ci = confidence_interval_mean(&x, 0.95).unwrap();
    close(ci.lower, 5.950_929_687_616_488_6, 1e-9);
    close(ci.upper, 11.049_070_312_383_511, 1e-9);
}

#[test]
fn confidence_interval_mean_z_takes_no_ctx() {
    // scipy: norm.interval(0.95, 2, 2/sqrt(3)) = (-0.2631714681523438, 4.263171468152343)
    let ci = confidence_interval_mean_z(&from_i64(&[1, 2, 3]), &qi(2), 0.95).unwrap();
    close(ci.lower, -0.263_171_468_152_343_8, 1e-9);
    close(ci.upper, 4.263_171_468_152_343, 1e-9);
}

#[test]
fn ols_intervals_take_no_ctx() {
    use symplex::stats::regression::ols;
    let y = from_i64(&[2, 4, 5, 4, 5, 7, 8]);
    let x: Vec<Vec<Q>> = (1..=7).map(|v| vec![qi(v)]).collect();
    let fit = ols(&y, &x, true).unwrap();
    let ci = fit.conf_int(0.95).unwrap();
    assert_eq!(ci.len(), 2);
    assert!(ci[1].lower < ci[1].upper);
    let mean = fit
        .confidence_interval_mean_response(&[qi(8)], 0.95)
        .unwrap();
    let obs = fit.prediction_interval(&[qi(8)], 0.95).unwrap();
    assert!(obs.lower < mean.lower && mean.upper < obs.upper);
}

#[test]
fn spearman_brown_takes_the_context_of_r() {
    use symplex::stats::reliability::spearman_brown;
    let ctx = Context::new();
    assert_eq!(
        spearman_brown(&ctx.rational(3, 5), 2).simplify(),
        ctx.rational(3, 4)
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Renames, counts, `&Q` parameters
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn z_test_two_proportions_is_the_new_name() {
    let ctx = Context::new();
    // statsmodels: proportions_ztest([45, 30], [100, 100]) = (2.1908902300206647, 0.028459736916310555)
    let r = z_test_two_proportions(&ctx, 45, 100, 30, 100, Alternative::TwoSided).unwrap();
    close(r.statistic_f64().unwrap(), 2.190_890_230_020_664_7, 1e-12);
    close(r.p_value_f64().unwrap(), 0.028_459_736_916_310_555, 1e-12);
}

#[test]
fn counts_usize_composes_confusion_matrix_with_chi_square() {
    let ctx = Context::new();
    let a = from_i64(&[1, 2, 3, 1, 2, 3, 1, 1, 2, 3]);
    let b = from_i64(&[1, 2, 3, 1, 3, 3, 1, 2, 2, 3]);
    let m = confusion_matrix(&a, &b, &from_i64(&[1, 2, 3])).unwrap();
    let table = counts_usize(&m);
    assert_eq!(table, counts(&[&[3, 1, 0], &[0, 2, 1], &[0, 0, 3]]));
    let r = chi_square_independence(&ctx, &table, false).unwrap();
    assert_eq!(r.df, 4);
    // Row sums 4, 3, 3 and column sums 3, 3, 4 over N = 10.
    assert_eq!(r.expected[0][0], q(6, 5));
    assert_eq!(counts_usize(&[]), Vec::<Vec<Q>>::new());
}

#[test]
fn estimation_counts_are_usize() {
    use symplex::stats::Beta;
    use symplex::stats::estimation::{
        FamilyKind, beta_binomial_posterior, dirichlet_multinomial_posterior, fit,
        gamma_poisson_posterior, posterior_predictive_beta_binomial,
    };
    let ctx = Context::new();
    let n: usize = 7;
    let post = beta_binomial_posterior(&ctx, &qi(2), &qi(3), n, 3).unwrap();
    let b = post.downcast_ref::<Beta>().unwrap();
    assert_eq!((b.alpha.clone(), b.beta.clone()), (ctx.int(9), ctx.int(6)));
    let counts: Vec<usize> = vec![3, 5, 4];
    assert!(gamma_poisson_posterior(&ctx, &qi(2), &q(1, 2), &counts).is_ok());
    assert_eq!(
        dirichlet_multinomial_posterior(&[qi(1), qi(1), qi(1)], &counts).unwrap(),
        vec![q(4, 15), q(6, 15), q(5, 15)]
    );
    let pred = posterior_predictive_beta_binomial(&ctx, &qi(2), &qi(3), 4usize).unwrap();
    assert_eq!(pred.density(&ctx.int(2)).simplify(), ctx.rational(9, 35));
    let trials: usize = 10;
    assert!(
        fit(
            &ctx,
            FamilyKind::Binomial { n: trials },
            &from_i64(&[3, 5, 7, 4])
        )
        .is_ok()
    );
}

#[test]
fn sprt_takes_parameters_by_reference() {
    let mut test = Sprt::bernoulli(&q(1, 2), &q(3, 4), 0.05, 0.1).unwrap();
    assert_eq!(test.observe(&qi(1)).unwrap(), Decision::Continue);
    assert!(test.observe(&q(1, 2)).is_err());
    let (p0, p1) = (q(1, 2), q(3, 4));
    let again = Sprt::bernoulli(&p0, &p1, 0.05, 0.1).unwrap();
    // The parameters are still ours to use.
    assert_eq!(p0 + p1, q(5, 4));
    assert_eq!(again.observations(), 0);
    let mut normal = Sprt::normal_mean(&qi(0), &qi(1), &qi(2), 0.05, 0.05).unwrap();
    assert_eq!(normal.observe(&q(3, 2)).unwrap(), Decision::Continue);
    // Levels are checked with the shared wording.
    assert!(Sprt::bernoulli(&q(1, 2), &q(3, 4), 1.0, 0.1).is_err());
    assert!(Sprt::bernoulli(&q(1, 2), &q(3, 4), 0.1, 0.0).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// The old paths still compile (dropped in 0.19)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[allow(deprecated)]
fn old_paths_still_compile() {
    use symplex::stats::aggregation as agg;
    use symplex::stats::hypothesis as hyp;
    use symplex::stats::reliability as rel;

    let ctx = Context::new();
    // hypothesis → anova / estimation
    let g = [from_i64(&[1, 2, 3]), from_i64(&[2, 3, 5])];
    let _: hyp::AnovaResult = hyp::anova_one_way(&ctx, &g).unwrap();
    assert!(hyp::confidence_interval_mean(&from_i64(&[5, 7, 8, 9]), 0.95).is_ok());
    // the renamed test
    let old = hyp::two_proportion_z_test(&ctx, 45, 100, 30, 100, Alternative::TwoSided).unwrap();
    let new = z_test_two_proportions(&ctx, 45, 100, 30, 100, Alternative::TwoSided).unwrap();
    assert_eq!(old, new);
    // reliability → agreement
    let table = [vec![20, 5], vec![10, 15]];
    let _: rel::KappaCi = rel::kappa_ci_from_confusion(&ctx, &table, 0.95).unwrap();
    assert!(rel::kappa_test_from_confusion(&ctx, &table, Alternative::TwoSided).is_ok());
    assert_eq!(rel::kappa_maximum_from_confusion(&table).unwrap(), q(4, 5));
    let a = from_i64(&[0, 0, 1, 1, 1, 0]);
    let b = from_i64(&[0, 1, 1, 1, 0, 0]);
    assert!(rel::cohen_kappa_ci(&ctx, &a, &b, 0.95).is_ok());
    assert!(rel::kappa_test(&ctx, &a, &b, Alternative::TwoSided).is_ok());
    assert_eq!(rel::cohen_kappa_maximum(&a, &b).unwrap(), qi(1));
    let t = RatingTable::from_i64(&[&[1, 1, 0], &[1, 0, 0], &[0, 1, 1], &[1, 1, 0]]).unwrap();
    assert!(rel::cochrans_q(&ctx, &t).is_ok());
    // reliability → hypothesis
    let (x, y) = xy();
    assert!(rel::pearson_test(&ctx, &x, &y, Alternative::TwoSided).is_ok());
    assert!(rel::pearson_t_statistic(&ctx, &x, &y).is_ok());
    assert!(rel::compare_two_correlations(&ctx, 0.7, 50, 0.4, 60, Alternative::TwoSided).is_ok());
    let c = counts(&[&[10, 20, 30], &[6, 9, 17]]);
    assert!(rel::expected_counts(&c).is_ok());
    assert!(rel::chi2_contributions(&c).is_ok());
    assert!(rel::standardized_residuals(&ctx, &c).is_ok());
    assert!(rel::adjusted_residuals(&ctx, &c).is_ok());
    // reliability → estimation
    assert!(rel::pearson_ci(0.8, 5, 0.95).is_ok());
    let _ = rel::fisher_z(&ctx.rational(4, 5));
    // reliability → data
    let _: rel::ConcordanceCounts = rel::concordance_counts(&x, &y).unwrap();
    assert!(rel::goodman_kruskal_gamma(&x, &y).is_ok());
    assert!(rel::somers_d(&x, &y, rel::Dependent::Symmetric).is_ok());
    assert!(rel::kendall_tau_c(&x, &y).is_ok());
    // aggregation → estimation
    assert!(agg::proportion_interval(3, 10, 0.95, agg::IntervalMethod::Wilson).is_ok());
    let z = agg::z_for_confidence(&ctx, &q(95, 100)).unwrap();
    assert!(agg::proportion_interval_symbolic(&ctx, 3, 10, &z, agg::IntervalMethod::Wald).is_ok());
    assert!(
        agg::proportion_interval_exact(&ctx, 1, 1, &q(9, 10), agg::IntervalMethod::ClopperPearson)
            .is_ok()
    );
}
