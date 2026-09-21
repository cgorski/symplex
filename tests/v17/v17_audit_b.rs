//! Independent verification of `symplex::stats` (0.17 audit, track b):
//! `regression`, `reliability`, `anova`, `survival`, `sequential`.
//!
//! Every dataset here is new (none is shared with `tests/v14/*` or
//! `tests/v17/v17_anova.rs`).  Reference values were computed with
//! `symplex/.venv/bin/python` (scipy 1.18.1, statsmodels 0.15.0, pingouin
//! 0.6.1, numpy 2.5.3); exact rationals come from `fractions.Fraction`
//! arithmetic on the integer data (never from `Fraction(float)`).  The
//! exact oracle call is quoted in a comment above each assertion.

use symplex::linprog::{q, qi};
use symplex::num_traits::{One, Signed, Zero};
use symplex::prelude::*;
use symplex::stats::Rng;
use symplex::stats::agreement::RatingTable;
use symplex::stats::anova::{
    Adjustment, SsType, TwoWayData, anova_repeated_measures, anova_two_way, anova_two_way_with,
    pairwise_t_tests, studentized_range_cdf, studentized_range_quantile, studentized_range_sf,
    tukey_hsd,
};
use symplex::stats::data::from_i64;
use symplex::stats::hypothesis::{Alternative, anova_one_way, counts, t_test_two_sample};
use symplex::stats::regression::{
    Design, LogitOpts, hat_matrix, logit, ols, polyfit, r_squared_from_correlation,
    simple_linear_regression, slope_from_correlation, vif, wls,
};
use symplex::stats::reliability::{
    Dependent, SplitHalf, adjusted_residuals, alpha_if_deleted, average_inter_item_correlation,
    chi2_contributions, cochrans_q, cohen_kappa_ci, cohen_kappa_maximum, compare_two_correlations,
    concordance_counts, corrected_item_total_correlation, cronbach_alpha, cronbach_alpha_complete,
    expected_counts, fisher_z, goodman_kruskal_gamma, guttman_lambda2, item_difficulty,
    item_discrimination_index, item_response_summary, item_total_correlation,
    kappa_ci_from_confusion, kappa_maximum_from_confusion, kappa_test, kappa_test_from_confusion,
    kendall_tau_c, kr20, pearson_ci, pearson_t_statistic, pearson_test, point_biserial, somers_d,
    spearman_brown, split_half, split_half_correlation, standardized_alpha, standardized_residuals,
    total_scores,
};
use symplex::stats::sequential::{
    Decision, Sprt, expected_sample_size_bernoulli, operating_characteristic_bernoulli,
    wald_boundaries,
};
use symplex::stats::survival::{
    CiMethod, KaplanMeier, Observation, exponential_rate, log_rank_test, mean_event_time,
};

type R = Result<(), SymplexError>;

fn close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() < tol,
        "got {actual}, expected {expected} (tol {tol})"
    );
}

fn qf(x: &Q) -> f64 {
    symplex::stats::data::to_f64(std::slice::from_ref(x))[0]
}

fn col(x: &[i64]) -> Vec<Vec<Q>> {
    x.iter().map(|&v| vec![qi(v)]).collect()
}

fn qs(entries: &[(i64, i64)]) -> Vec<Q> {
    entries.iter().map(|&(n, d)| q(n, d)).collect()
}

fn is_invalid<T: std::fmt::Debug>(r: Result<T, SymplexError>) {
    assert!(
        matches!(r, Err(SymplexError::InvalidArgument { .. })),
        "expected InvalidArgument, got {r:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Regression datasets
// ═══════════════════════════════════════════════════════════════════════════

// R1: simple regression, n = 9.
//   x = [2, 4, 5, 7, 8, 10, 11, 13, 14]; y = [5, 8, 7, 12, 11, 15, 14, 19, 18]
//   statsmodels: OLS(y, add_constant(x)).fit()
fn r1() -> (Vec<Q>, Vec<Q>) {
    (
        from_i64(&[2, 4, 5, 7, 8, 10, 11, 13, 14]),
        from_i64(&[5, 8, 7, 12, 11, 15, 14, 19, 18]),
    )
}

// R2: two regressors, n = 10.
//   x1 = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]; x2 = [2, 7, 1, 8, 2, 8, 1, 8, 2, 8]
//   y  = [10, 12, 9, 14, 11, 22, 7, 19, 12, 16]
fn r2() -> (Vec<Vec<Q>>, Vec<Q>) {
    let x1 = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
    let x2 = [2, 7, 1, 8, 2, 8, 1, 8, 2, 8];
    (
        x1.iter()
            .zip(&x2)
            .map(|(&a, &b)| vec![qi(a), qi(b)])
            .collect(),
        from_i64(&[10, 12, 9, 14, 11, 22, 7, 19, 12, 16]),
    )
}

// R3: WLS on R2 with weights [1, 1, 2, 2, 3, 3, 4, 4, 5, 5].
fn r3_weights() -> Vec<Q> {
    from_i64(&[1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Regression — R1
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn r1_simple_regression_exact_and_matches_linregress() -> R {
    let (x, y) = r1();
    let fit = simple_linear_regression(&x, &y)?;
    // Fraction Gauss–Jordan on the normal equations: β = [812/305, 701/610]
    // scipy: linregress(x, y) → intercept 2.662295081967212, slope 1.1491803278688526
    assert_eq!(fit.coefficients, vec![q(812, 305), q(701, 610)]);
    // (812/305 = 2.662295081967213…; scipy's float intercept is 1.3e-15 low.)
    close(qf(&fit.coefficients[0]), 2.662_295_081_967_212, 1e-14);
    close(qf(&fit.coefficients[1]), 1.149_180_327_868_852_6, 1e-15);
    // Fraction: ssr 3011/305, tss 1700/9, R² 491401/518500, adj 853177/907375, σ̂² 3011/2135
    assert_eq!(fit.ssr, q(3011, 305));
    assert_eq!(fit.tss, q(1700, 9));
    assert_eq!(fit.ess, q(491_401, 2745));
    assert_eq!(fit.r_squared, q(491_401, 518_500));
    assert_eq!(fit.adjusted_r_squared, q(853_177, 907_375));
    assert_eq!(fit.mse_resid, q(3011, 2135));
    assert_eq!(
        (fit.df_model, fit.df_resid, fit.nobs(), fit.n_params()),
        (1, 7, 9, 2)
    );
    // linregress rvalue = 0.9735172193021161 → rvalue² = R²
    close(
        qf(&fit.r_squared),
        0.973_517_219_302_116_1_f64.powi(2),
        1e-14,
    );
    // Fraction residuals.
    assert_eq!(
        fit.residuals,
        qs(&[
            (12, 305),
            (226, 305),
            (-859, 610),
            (789, 610),
            (-261, 305),
            (258, 305),
            (-159, 122),
            (853, 610),
            (-229, 305)
        ])
    );
    // Residuals sum to zero and are orthogonal to x (normal equations).
    let sum: Q = fit.residuals.iter().fold(Q::zero(), |a, e| a + e);
    assert!(sum.is_zero());
    let dot: Q = fit
        .residuals
        .iter()
        .zip(&x)
        .fold(Q::zero(), |a, (e, xi)| a + e * xi);
    assert!(dot.is_zero());
    Ok(())
}

#[test]
fn r1_covariance_standard_errors_t_p_and_conf_int() -> R {
    let ctx = Context::new();
    let (x, y) = r1();
    let fit = simple_linear_regression(&x, &y)?;
    // Fraction: (XᵀX)⁻¹ = [[186/305, -37/610], [-37/610, 9/1220]]
    assert_eq!(
        fit.normalized_cov_params().to_rows(),
        vec![qs(&[(186, 305), (-37, 610)]), qs(&[(-37, 610), (9, 1220)])]
    );
    // cov_params = σ̂² (XᵀX)⁻¹ = [[560046/651175, -111407/1302350], [., 27099/2604700]]
    assert_eq!(
        fit.cov_params.to_rows(),
        vec![
            qs(&[(560_046, 651_175), (-111_407, 1_302_350)]),
            qs(&[(-111_407, 1_302_350), (27_099, 2_604_700)])
        ]
    );
    // statsmodels: bse = [0.927391242591253, 0.10199943766655643]
    let se = fit.standard_errors(&ctx);
    close(se[0].eval_f64()?, 0.927_391_242_591_253, 1e-12);
    close(se[1].eval_f64()?, 0.101_999_437_666_556_43, 1e-12);
    // tvalues = [2.87073563, 11.26653592]; t² exact = [2307704/280023, 3439807/27099]
    let t = fit.t_statistics(&ctx)?;
    close(t[1].eval_f64()?, (3_439_807.0_f64 / 27_099.0).sqrt(), 1e-12);
    close(
        t[0].eval_f64()?,
        (2_307_704.0_f64 / 280_023.0).sqrt(),
        1e-12,
    );
    // pvalues = [2.39667901e-02, 9.69782517e-06]; linregress pvalue 9.697825165130998e-06
    let p = fit.p_values(&ctx)?;
    close(p[1].eval_f64()?, 9.697_825_165_130_998e-6, 1e-15);
    close(p[0].eval_f64()?, 2.396_679_01e-2, 1e-9);
    let tests = fit.coefficient_tests(&ctx)?;
    assert_eq!(tests.len(), 2);
    assert_eq!(tests[1].df, Some(ctx.int(7)));
    assert_eq!(tests[1].alternative, Alternative::TwoSided);
    close(tests[1].p_value_f64()?, 9.697_825_165_130_998e-6, 1e-15);
    // conf_int(0.05) = [[0.46936326, 4.8552269], [0.90798998, 1.39037067]]
    let ci = fit.conf_int(&ctx, 0.95)?;
    close(ci[0].lower, 0.469_363_26, 1e-7);
    close(ci[0].upper, 4.855_226_9, 1e-7);
    close(ci[1].lower, 0.907_989_98, 1e-7);
    close(ci[1].upper, 1.390_370_67, 1e-7);
    // conf_int(0.10) = [[0.90527948, 4.41931069], [0.95593438, 1.34242628]]
    let ci90 = fit.conf_int(&ctx, 0.90)?;
    close(ci90[0].lower, 0.905_279_48, 1e-7);
    close(ci90[1].upper, 1.342_426_28, 1e-7);
    Ok(())
}

#[test]
fn r1_f_test_anova_table_and_information_criteria() -> R {
    let ctx = Context::new();
    let (x, y) = r1();
    let fit = simple_linear_regression(&x, &y)?;
    // Fraction: F = ESS / σ̂² = 3439807/27099; statsmodels fvalue 126.93483154359929,
    // f_pvalue 9.697825165131174e-06 (= the slope's p-value, one regressor)
    assert_eq!(fit.f_statistic()?, q(3_439_807, 27_099));
    let f = fit.f_test(&ctx)?;
    close(f.statistic_f64()?, 126.934_831_543_599_29, 1e-11);
    close(f.p_value_f64()?, 9.697_825_165_131_174e-6, 1e-15);
    assert_eq!(f.df, Some(ctx.int(7)));
    let t = fit.anova_table()?;
    assert_eq!((t.df_model, t.df_resid, t.df_total), (1, 7, 8));
    assert_eq!(&t.ss_model + &t.ss_resid, t.ss_total);
    assert_eq!(t.ms_model, q(491_401, 2745));
    assert_eq!(t.ms_resid, q(3011, 2135));
    assert_eq!(t.f, q(3_439_807, 27_099));
    // statsmodels: llf -13.18665708426352, aic 30.37331416852704, bic 30.76776332319948
    close(
        fit.log_likelihood(&ctx)?.eval_f64()?,
        -13.186_657_084_263_52,
        1e-12,
    );
    close(fit.aic(&ctx)?.eval_f64()?, 30.373_314_168_527_04, 1e-12);
    close(fit.bic(&ctx)?.eval_f64()?, 30.767_763_323_199_48, 1e-12);
    close(
        fit.residual_standard_error(&ctx).eval_f64()?,
        (3011.0_f64 / 2135.0).sqrt(),
        1e-14,
    );
    Ok(())
}

#[test]
fn r1_prediction_and_intervals_at_x_9() -> R {
    let ctx = Context::new();
    let (x, y) = r1();
    let fit = simple_linear_regression(&x, &y)?;
    // ŷ(9) = 812/305 + 9·701/610 = 7933/610; statsmodels get_prediction([[1, 9]]).summary_frame():
    //   mean 13.004918032786891, mean_ci (12.050259148715563, 13.95957691685822),
    //   obs_ci (10.038941402823053, 15.97089466275073)
    assert_eq!(fit.predict(&[qi(9)])?, q(7933, 610));
    let mean = fit.confidence_interval_mean_response(&ctx, &[qi(9)], 0.95)?;
    close(mean.lower, 12.050_259_148_715_563, 1e-7);
    close(mean.upper, 13.959_576_916_858_22, 1e-7);
    let obs = fit.prediction_interval(&ctx, &[qi(9)], 0.95)?;
    close(obs.lower, 10.038_941_402_823_053, 1e-7);
    close(obs.upper, 15.970_894_662_750_73, 1e-7);
    // The prediction interval contains the mean-response interval.
    assert!(obs.lower < mean.lower && mean.upper < obs.upper);
    // Wrong row length and confidence outside (0, 1) are rejected.
    is_invalid(fit.predict(&[qi(9), qi(1)]));
    is_invalid(fit.prediction_interval(&ctx, &[qi(9)], 1.0));
    is_invalid(fit.confidence_interval_mean_response(&ctx, &[qi(9)], 0.0));
    is_invalid(fit.conf_int(&ctx, 1.0));
    Ok(())
}

#[test]
fn r1_leverage_cooks_durbin_watson_and_hat_matrix() -> R {
    let (x, y) = r1();
    let fit = simple_linear_regression(&x, &y)?;
    // Fraction leverages; statsmodels hat_matrix_diag = [0.39672131, 0.24262295, 0.18770492, …]
    let lev = fit.leverage();
    assert_eq!(
        lev,
        qs(&[
            (121, 305),
            (74, 305),
            (229, 1220),
            (149, 1220),
            (34, 305),
            (41, 305),
            (41, 244),
            (341, 1220),
            (109, 305)
        ])
    );
    assert_eq!(lev.iter().fold(Q::zero(), |a, h| a + h), qi(2));
    // Fraction Cook's distances; statsmodels cooks_distance[0] = [0.00059823, 0.08233451, 0.20000083,
    //   0.093998, 0.03665883, 0.0455169, 0.14618854, 0.37327764, 0.17295878]
    let cooks = fit.cooks_distance()?;
    assert_eq!(cooks[0], q(7623, 12_742_552));
    assert_eq!(cooks[2], q(1_182_823_243, 5_914_091_782));
    assert_eq!(cooks[7], q(1_736_802_683, 4_652_844_102));
    close(qf(&cooks[7]), 0.373_277_64, 1e-8);
    close(qf(&cooks[4]), 0.036_658_83, 1e-8);
    // Fraction: DW = 678366/183671; statsmodels durbin_watson(resid) = 3.693375655383811
    assert_eq!(fit.durbin_watson()?, q(678_366, 183_671));
    close(qf(&fit.durbin_watson()?), 3.693_375_655_383_811, 1e-13);
    // The hat matrix is the symmetric idempotent projection with trace p, and the
    // free function on the design agrees with the method.
    let h = fit.hat_matrix()?;
    assert_eq!(h, hat_matrix(fit.design())?);
    assert_eq!(h, h.transpose());
    assert_eq!(h.matmul(&h)?, h);
    assert_eq!(h.trace()?, qi(2));
    assert_eq!(h.diagonal(), lev);
    Ok(())
}

#[test]
fn r1_design_builder_ols_and_simple_regression_are_identical() -> R {
    let (x, y) = r1();
    let a = simple_linear_regression(&x, &y)?;
    let b = Design::new().intercept().column(&x).fit(&y)?;
    let c = ols(&y, &col(&[2, 4, 5, 7, 8, 10, 11, 13, 14]), true)?;
    assert_eq!(a, b);
    assert_eq!(a, c);
    let d = Design::new().intercept().column(&x);
    assert!(d.has_intercept());
    assert_eq!(d.n_columns(), 1);
    assert_eq!(d.rows(9)?.len(), 9);
    // Zero rows: `rows(0)` is empty, fitting an empty response is rejected.
    assert_eq!(Design::new().intercept().rows(0)?, Vec::<Vec<Q>>::new());
    is_invalid(Design::new().intercept().fit(&[]));
    is_invalid(d.rows(8));
    is_invalid(d.fit(&y[..8]));
    // Weighted fit through the builder equals `wls`.
    let w = from_i64(&[1, 2, 1, 2, 1, 2, 1, 2, 1]);
    assert_eq!(
        d.fit_weighted(&y, &w)?,
        wls(&y, &col(&[2, 4, 5, 7, 8, 10, 11, 13, 14]), &w, true)?
    );
    Ok(())
}

#[test]
fn r1_correlation_bridges_recover_r_squared_and_slope() -> R {
    let ctx = Context::new();
    let (x, y) = r1();
    let fit = simple_linear_regression(&x, &y)?;
    // r² = 491401/518500 → r = √(491401/518500) = 701/√518500 … as an expression
    let r = ctx.from_ratio(q(491_401, 518_500)).sqrt();
    assert_eq!(
        r_squared_from_correlation(&r).simplify(),
        ctx.from_ratio(q(491_401, 518_500))
    );
    // b = r · s_y / s_x with the population standard deviations: Fraction
    //   s_x² = 1220/81·… — checked numerically: slope 1.1491803278688526
    let sd_x = symplex::stats::data::variance(&x, symplex::stats::data::Ddof::Population)?;
    let sd_y = symplex::stats::data::variance(&y, symplex::stats::data::Ddof::Population)?;
    let b = slope_from_correlation(
        &r,
        &ctx.from_ratio(sd_x).sqrt(),
        &ctx.from_ratio(sd_y).sqrt(),
    )?;
    close(b.eval_f64()?, qf(&fit.coefficients[1]), 1e-13);
    is_invalid(slope_from_correlation(&r, &ctx.zero(), &ctx.one()));
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Regression — R2 (two regressors) and R3 (WLS)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn r2_two_regressors_exact_fit_statistics() -> R {
    let ctx = Context::new();
    let (x, y) = r2();
    let fit = ols(&y, &x, true)?;
    // Fraction: β = [26845/6658, 56195/53264, 57269/53264]
    // statsmodels params [4.031991589065785, 1.0550277861219581, 1.0751914989486333]
    assert_eq!(
        fit.coefficients,
        qs(&[(26_845, 6658), (56_195, 53_264), (57_269, 53_264)])
    );
    close(qf(&fit.coefficients[0]), 4.031_991_589_065_785, 1e-14);
    // Fraction: ssr 86043/53264, tss 968/5, R² 51129337/51559552, adj 357044929/360916864,
    //   σ̂² 86043/372848, ESS 51129337/266320
    assert_eq!(fit.ssr, q(86_043, 53_264));
    assert_eq!(fit.tss, q(968, 5));
    assert_eq!(fit.ess, q(51_129_337, 266_320));
    assert_eq!(fit.r_squared, q(51_129_337, 51_559_552));
    assert_eq!(fit.adjusted_r_squared, q(357_044_929, 360_916_864));
    assert_eq!(fit.mse_resid, q(86_043, 372_848));
    assert_eq!((fit.df_model, fit.df_resid), (2, 7));
    // Fraction cov_params diagonal: 158921421/1241210992, 84408183/19859375872, 47237607/19859375872
    // statsmodels bse [0.35782313249027453, 0.065194277688097, 0.04877094228506145]
    let cov = fit.cov_params.to_rows();
    assert_eq!(cov[0][0], q(158_921_421, 1_241_210_992));
    assert_eq!(cov[1][1], q(84_408_183, 19_859_375_872));
    assert_eq!(cov[2][2], q(47_237_607, 19_859_375_872));
    assert_eq!(cov[1][2], q(-946_473, 2_837_053_696));
    let se = fit.standard_errors(&ctx);
    close(se[0].eval_f64()?, 0.357_823_132_490_274_53, 1e-14);
    close(se[2].eval_f64()?, 0.048_770_942_285_061_45, 1e-14);
    // statsmodels tvalues [11.26811327430144, 16.182828056925956, 22.045739708374768]
    //   pvalues [9.688767859574868e-06, 8.370354654659853e-07, 9.981096028181609e-08]
    let t = fit.t_statistics(&ctx)?;
    close(t[1].eval_f64()?, 16.182_828_056_925_956, 1e-12);
    let p = fit.p_values(&ctx)?;
    close(p[0].eval_f64()?, 9.688_767_859_574_868e-6, 1e-17);
    close(p[1].eval_f64()?, 8.370_354_654_659_853e-7, 1e-17);
    close(p[2].eval_f64()?, 9.981_096_028_181_609e-8, 1e-17);
    // Fraction: F = 357905359/860430; statsmodels fvalue 415.96104157223726, f_pvalue 5.3066133714138176e-08
    assert_eq!(fit.f_statistic()?, q(357_905_359, 860_430));
    close(
        fit.f_test(&ctx)?.p_value_f64()?,
        5.306_613_371_413_817_6e-8,
        1e-17,
    );
    // statsmodels llf -5.074392319388945, aic 16.14878463877789, bic 17.056539917760027
    close(
        fit.log_likelihood(&ctx)?.eval_f64()?,
        -5.074_392_319_388_945,
        1e-12,
    );
    close(fit.aic(&ctx)?.eval_f64()?, 16.148_784_638_777_89, 1e-12);
    close(fit.bic(&ctx)?.eval_f64()?, 17.056_539_917_760_027, 1e-12);
    // conf_int(0.05): [[3.185874332198384, 4.878108845933186], [0.9008678160356096, 1.2091877562083067],
    //   [0.959866546048345, 1.1905164518489215]]
    let ci = fit.conf_int(&ctx, 0.95)?;
    close(ci[0].lower, 3.185_874_332_198_384, 1e-9);
    close(ci[1].upper, 1.209_187_756_208_306_7, 1e-9);
    close(ci[2].lower, 0.959_866_546_048_345, 1e-9);
    // get_prediction([[1, 4, 5]]).summary_frame(alpha=0.10): mean 13.628060228296784,
    //   mean_ci (13.338779768589674, 13.917340688003893), obs_ci (12.67306082937617, 14.583059627217397)
    close(
        qf(&fit.predict(&[qi(4), qi(5)])?),
        13.628_060_228_296_784,
        1e-13,
    );
    let m = fit.confidence_interval_mean_response(&ctx, &[qi(4), qi(5)], 0.90)?;
    close(m.lower, 13.338_779_768_589_674, 1e-9);
    close(m.upper, 13.917_340_688_003_893, 1e-9);
    let o = fit.prediction_interval(&ctx, &[qi(4), qi(5)], 0.90)?;
    close(o.lower, 12.673_060_829_376_17, 1e-9);
    close(o.upper, 14.583_059_627_217_397, 1e-9);
    Ok(())
}

#[test]
fn r2_influence_diagnostics_and_vif() -> R {
    let (x, y) = r2();
    let fit = ols(&y, &x, true)?;
    // Fraction leverages: 9749/53264, 4377/13316, …, 12557/53264 (sum = 3)
    let lev = fit.leverage();
    assert_eq!(lev[0], q(9749, 53_264));
    assert_eq!(lev[1], q(4377, 13_316));
    assert_eq!(lev[5], q(34_229, 53_264));
    assert_eq!(lev[9], q(12_557, 53_264));
    assert_eq!(lev.iter().fold(Q::zero(), |a, h| a + h), qi(3));
    // Fraction Cook's: 82440891250907/488781516674025, 129769405425/327396569143, …
    // statsmodels cooks_distance[0] = [0.16866613903873992, 0.3963676399074282, …, 0.00098635425134591, …]
    let cooks = fit.cooks_distance()?;
    assert_eq!(cooks[0], q(82_440_891_250_907, 488_781_516_674_025));
    assert_eq!(cooks[1], q(129_769_405_425, 327_396_569_143));
    assert_eq!(cooks[7], q(23_832_613_343, 24_162_326_375_625));
    close(qf(&cooks[0]), 0.168_666_139_038_739_92, 1e-14);
    close(qf(&cooks[7]), 0.000_986_354_251_345_91, 1e-14);
    // Fraction: DW = 191287095/95479049; statsmodels durbin_watson 2.0034457507007626
    assert_eq!(fit.durbin_watson()?, q(191_287_095, 95_479_049));
    close(qf(&fit.durbin_watson()?), 2.003_445_750_700_762_6, 1e-13);
    // VIF: two regressors share one R², 1/(1 − r²₁₂) = 538569/532640;
    // statsmodels variance_inflation_factor(add_constant(X), j) = 1.0111313457494742 for both
    let v = vif(&x)?;
    assert_eq!(v, vec![q(538_569, 532_640), q(538_569, 532_640)]);
    close(qf(&v[0]), 1.011_131_345_749_474_2, 1e-14);
    // A lone column has VIF 1; an exact linear combination is rejected.
    assert_eq!(vif(&col(&[1, 2, 4, 7]))?, vec![Q::one()]);
    let dup: Vec<Vec<Q>> = x
        .iter()
        .map(|r| vec![r[0].clone(), &r[0] * qi(3)])
        .collect();
    is_invalid(vif(&dup));
    Ok(())
}

#[test]
fn r3_weighted_least_squares_exact() -> R {
    let ctx = Context::new();
    let (x, y) = r2();
    let w = r3_weights();
    let fit = wls(&y, &x, &w, true)?;
    assert_eq!(fit.weights(), Some(&w[..]));
    // Fraction: β = [2593266/640579, 671005/640579, 696443/640579]
    // statsmodels WLS params [4.048315664422336, 1.0474976544657255, 1.087208603466551]
    assert_eq!(
        fit.coefficients,
        qs(&[(2_593_266, 640_579), (671_005, 640_579), (696_443, 640_579)])
    );
    close(qf(&fit.coefficients[2]), 1.087_208_603_466_551, 1e-14);
    // Fraction: ssr 2344548/640579, tss 6223/10, R² 3962877637/3986323117, adj 27693252499/27904261819,
    //   scale 2344548/4484053; statsmodels ssr 3.6600450529911166, centered_tss 622.3,
    //   rsquared 0.9941185199212741, rsquared_adj 0.9924380970416381, scale 0.5228635789987309
    assert_eq!(fit.ssr, q(2_344_548, 640_579));
    assert_eq!(fit.tss, q(6223, 10));
    assert_eq!(fit.ess, q(3_962_877_637, 6_405_790));
    assert_eq!(fit.r_squared, q(3_962_877_637, 3_986_323_117));
    assert_eq!(fit.adjusted_r_squared, q(27_693_252_499, 27_904_261_819));
    assert_eq!(fit.mse_resid, q(2_344_548, 4_484_053));
    close(qf(&fit.ssr), 3.660_045_052_991_116_6, 1e-13);
    close(qf(&fit.adjusted_r_squared), 0.992_438_097_041_638_1, 1e-14);
    // Fraction cov diagonal: 300799647030/2872390186687, 10739202114/2872390186687, 5129871024/2872390186687
    // statsmodels bse [0.3236062821755174, 0.06114547045531395, 0.0422601958992233]
    let cov = fit.cov_params.to_rows();
    assert_eq!(cov[0][0], q(300_799_647_030, 2_872_390_186_687));
    assert_eq!(cov[1][1], q(10_739_202_114, 2_872_390_186_687));
    assert_eq!(cov[2][2], q(5_129_871_024, 2_872_390_186_687));
    let se = fit.standard_errors(&ctx);
    close(se[1].eval_f64()?, 0.061_145_470_455_313_95, 1e-14);
    // Fraction: F = 27740143459/46890960; statsmodels fvalue 591.5883031398812, f_pvalue 1.560282387615572e-08
    //   pvalues [4.8064597942598461e-06, 5.667578181073303e-07, 3.4268020006730285e-08]
    assert_eq!(fit.f_statistic()?, q(27_740_143_459, 46_890_960));
    close(
        fit.f_test(&ctx)?.p_value_f64()?,
        1.560_282_387_615_572e-8,
        1e-17,
    );
    let p = fit.p_values(&ctx)?;
    close(p[0].eval_f64()?, 4.806_459_794_259_846e-6, 1e-17);
    close(p[2].eval_f64()?, 3.426_802_000_673_028_5e-8, 1e-17);
    // statsmodels llf -4.376345408784089, aic 14.752690817568178, bic 15.660446096550316
    close(
        fit.log_likelihood(&ctx)?.eval_f64()?,
        -4.376_345_408_784_089,
        1e-12,
    );
    close(fit.aic(&ctx)?.eval_f64()?, 14.752_690_817_568_178, 1e-12);
    close(fit.bic(&ctx)?.eval_f64()?, 15.660_446_096_550_316, 1e-12);
    // conf_int(0.05) = [[3.2831084016223304, 4.813522927222342], [0.9029115921520401, 1.192083716779411],
    //   [0.9872791193661856, 1.1871380875669162]]
    let ci = fit.conf_int(&ctx, 0.95)?;
    close(ci[0].lower, 3.283_108_401_622_330_4, 1e-9);
    close(ci[2].upper, 1.187_138_087_566_916_2, 1e-9);
    Ok(())
}

#[test]
fn r3_wls_influence_prediction_and_durbin_watson() -> R {
    let ctx = Context::new();
    let (x, y) = r2();
    let fit = wls(&y, &x, &r3_weights(), true)?;
    // Fraction leverages wᵢxᵢᵀ(XᵀWX)⁻¹xᵢ: 40943/640579, 90527/640579, …, 284215/640579 (sum 3)
    // statsmodels OLSInfluence(wls).hat_matrix_diag = [0.06391561384310139, 0.14132058653187202, …]
    let lev = fit.leverage();
    assert_eq!(lev[0], q(40_943, 640_579));
    assert_eq!(lev[1], q(90_527, 640_579));
    assert_eq!(lev[9], q(284_215, 640_579));
    assert_eq!(lev.iter().fold(Q::zero(), |a, h| a + h), qi(3));
    close(qf(&lev[0]), 0.063_915_613_843_101_39, 1e-15);
    // Fraction Cook's wᵢeᵢ²hᵢᵢ/(pσ̂²(1−hᵢᵢ)²): 5265250915737281/281004497358943936, …
    // Reference: OLS on the √w-whitened data (what R's cooks.distance does for lm(weights=)):
    //   OLS(√w·y, √w·X).fit().get_influence().cooks_distance[0] = [0.01873724785625692, 0.06095082090859835,
    //   0.03078680860625308, …, 0.00166310596386287, 0.7119571751468676, 0.05683524538742996].
    //   (statsmodels' OLSInfluence applied to the WLS result itself is *not* a valid oracle: its
    //   hat_matrix_diag comes out as hᵢ/√wᵢ and sums to 1.725, not p = 3.)
    let cooks = fit.cooks_distance()?;
    assert_eq!(cooks[0], q(5_265_250_915_737_281, 281_004_497_358_943_936));
    assert_eq!(cooks[1], q(225_187_850_411_489, 3_694_582_731_694_051));
    close(qf(&cooks[2]), 0.030_786_808_606_253_08, 1e-14);
    close(qf(&cooks[7]), 0.001_663_105_963_862_87, 1e-14);
    close(qf(&cooks[8]), 0.711_957_175_146_867_6, 1e-13);
    close(qf(&cooks[9]), 0.056_835_245_387_429_96, 1e-12);
    // fitted / resid are the plain Xβ̂ and y − Xβ̂: statsmodels fittedvalues[0] 9.365225834752614,
    //   resid[1] -0.7062735431539195
    close(qf(&fit.fitted[0]), 9.365_225_834_752_614, 1e-13);
    close(qf(&fit.residuals[1]), -0.706_273_543_153_919_5, 1e-13);
    // statsmodels durbin_watson(resid) = 2.003108258475236 (unweighted residuals, as documented)
    close(qf(&fit.durbin_watson()?), 2.003_108_258_475_236, 1e-13);
    // get_prediction([[1, 4, 5]]).summary_frame(alpha=0.05): mean 13.674349299617994,
    //   mean_ci (13.358362578260236, 13.990336020975752), obs_ci (11.93555315189151, 15.413145447344478)
    close(
        qf(&fit.predict(&[qi(4), qi(5)])?),
        13.674_349_299_617_994,
        1e-13,
    );
    let m = fit.confidence_interval_mean_response(&ctx, &[qi(4), qi(5)], 0.95)?;
    close(m.lower, 13.358_362_578_260_236, 1e-9);
    close(m.upper, 13.990_336_020_975_752, 1e-9);
    let o = fit.prediction_interval(&ctx, &[qi(4), qi(5)], 0.95)?;
    close(o.lower, 11.935_553_151_891_51, 1e-9);
    close(o.upper, 15.413_145_447_344_478, 1e-9);
    // The hat matrix reproduces the fitted values: ŷ = H y.
    let h = fit.hat_matrix()?;
    let hy = h.matmul(&QMatrix::col_vector(y.clone()))?;
    assert_eq!(hy.col(0), fit.fitted);
    assert_eq!(h.diagonal(), lev);
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Regression — constant detection and designs without an added intercept
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ols_explicit_constant_column_uses_centred_tss() -> R {
    // X = [3, x], x = 1..6, y = [7, 9, 8, 12, 15, 14], no added intercept.
    // Fraction: β = [77/45, 57/35], ssr 884/105, centred tss 329/6, R² 9747/11515
    // statsmodels OLS(y, X): k_constant 1, params [1.7111111111111104, 1.62857142857143],
    //   rsquared 0.846461137646548, centered_tss 54.83333333333333, df_model 1
    let y = from_i64(&[7, 9, 8, 12, 15, 14]);
    let x: Vec<Vec<Q>> = (1..=6).map(|v| vec![qi(3), qi(v)]).collect();
    let fit = ols(&y, &x, false)?;
    assert!(fit.has_constant());
    assert_eq!(fit.coefficients, vec![q(77, 45), q(57, 35)]);
    assert_eq!(fit.ssr, q(884, 105));
    assert_eq!(fit.tss, q(329, 6));
    assert_eq!(fit.r_squared, q(9747, 11_515));
    assert_eq!((fit.df_model, fit.df_resid), (1, 4));
    close(qf(&fit.r_squared), 0.846_461_137_646_548, 1e-14);
    // Without an added intercept, `predict` takes the full row.
    assert_eq!(fit.predict(&[qi(3), qi(7)])?, q(77, 15) + q(57, 5));
    Ok(())
}

#[test]
fn ols_full_dummy_set_is_an_implicit_constant() -> R {
    // Three group indicators, three observations each, no intercept:
    //   y = [4, 5, 6 | 7, 9, 8 | 12, 10, 11]; β = group means [5, 8, 11].
    // Fraction: ssr 6, centred tss 60, R² 9/10, adj 13/15, F 27
    // statsmodels: k_constant 1, rsquared 0.9, rsquared_adj 0.8666666666666667, df_model 2, fvalue 27.0
    let y = from_i64(&[4, 5, 6, 7, 9, 8, 12, 10, 11]);
    let x: Vec<Vec<Q>> = (0..9)
        .map(|i| (0..3).map(|g| qi(i64::from(i / 3 == g))).collect())
        .collect();
    let fit = ols(&y, &x, false)?;
    assert!(fit.has_constant());
    assert_eq!(fit.coefficients, vec![qi(5), qi(8), qi(11)]);
    assert_eq!(fit.ssr, qi(6));
    assert_eq!(fit.tss, qi(60));
    assert_eq!(fit.r_squared, q(9, 10));
    assert_eq!(fit.adjusted_r_squared, q(13, 15));
    assert_eq!((fit.df_model, fit.df_resid), (2, 6));
    assert_eq!(fit.f_statistic()?, qi(27));
    // The same F as the one-way ANOVA of the three groups.
    let ctx = Context::new();
    let groups = [
        from_i64(&[4, 5, 6]),
        from_i64(&[7, 9, 8]),
        from_i64(&[12, 10, 11]),
    ];
    assert_eq!(anova_one_way(&ctx, &groups)?.f, qi(27));
    Ok(())
}

#[test]
fn ols_through_the_origin_uses_uncentred_tss() -> R {
    let ctx = Context::new();
    // x = 1..6, y = [3, 5, 8, 9, 12, 13], no constant anywhere.
    // Fraction: β = 211/91, ssr 251/91, uncentred tss 492, R² 44521/44772, adj 37059/37310,
    //   σ̂² 251/455, F 222605/251, cov 251/41405
    // statsmodels OLS(y, x): params 2.3186813186813184, rsquared 0.9943938175645493,
    //   rsquared_adj 0.9932725810774591, fvalue 886.8725099601592, f_pvalue 8.006052695603976e-07,
    //   bse 0.07785929487436638, llf -6.182133089388673, aic 14.364266178777346, bic 14.1560256480054
    let y = from_i64(&[3, 5, 8, 9, 12, 13]);
    let fit = ols(&y, &col(&[1, 2, 3, 4, 5, 6]), false)?;
    assert!(!fit.has_constant());
    assert_eq!(fit.coefficients, vec![q(211, 91)]);
    assert_eq!(fit.ssr, q(251, 91));
    assert_eq!(fit.tss, qi(492));
    assert_eq!(fit.r_squared, q(44_521, 44_772));
    assert_eq!(fit.adjusted_r_squared, q(37_059, 37_310));
    assert_eq!(fit.mse_resid, q(251, 455));
    assert_eq!(fit.cov_params.to_rows(), vec![vec![q(251, 41_405)]]);
    assert_eq!((fit.df_model, fit.df_resid), (1, 5));
    assert_eq!(fit.f_statistic()?, q(222_605, 251));
    close(
        fit.f_test(&ctx)?.p_value_f64()?,
        8.006_052_695_603_976e-7,
        1e-17,
    );
    close(
        fit.p_values(&ctx)?[0].eval_f64()?,
        8.006_052_695_603_976e-7,
        1e-17,
    );
    close(
        fit.standard_errors(&ctx)[0].eval_f64()?,
        0.077_859_294_874_366_38,
        1e-14,
    );
    close(
        fit.log_likelihood(&ctx)?.eval_f64()?,
        -6.182_133_089_388_673,
        1e-12,
    );
    close(fit.aic(&ctx)?.eval_f64()?, 14.364_266_178_777_346, 1e-12);
    close(fit.bic(&ctx)?.eval_f64()?, 14.156_025_648_005_4, 1e-12);
    // conf_int(0.05) = [2.118537629541821, 2.518825007820816]; prediction at x = 7:
    //   mean 16.23076923076923, mean_ci (14.829763406792745, 17.631775054745717),
    //   obs_ci (13.862637164702916, 18.598901296835546)
    let ci = fit.conf_int(&ctx, 0.95)?;
    close(ci[0].lower, 2.118_537_629_541_821, 1e-9);
    close(ci[0].upper, 2.518_825_007_820_816, 1e-9);
    assert_eq!(fit.predict(&[qi(7)])?, q(211, 13));
    let m = fit.confidence_interval_mean_response(&ctx, &[qi(7)], 0.95)?;
    close(m.lower, 14.829_763_406_792_745, 1e-9);
    let o = fit.prediction_interval(&ctx, &[qi(7)], 0.95)?;
    close(o.upper, 18.598_901_296_835_546, 1e-9);
    Ok(())
}

#[test]
fn regression_error_paths_are_documented_errors_not_panics() -> R {
    let ctx = Context::new();
    let (x, y) = r2();
    // n = p: two observations, intercept + regressor → InvalidArgument (df_resid would be 0).
    is_invalid(ols(&y[..3], &x[..3], true));
    is_invalid(simple_linear_regression(
        &x[..2].iter().map(|r| r[0].clone()).collect::<Vec<_>>(),
        &y[..2],
    ));
    // Perfectly collinear columns (x2 = 2·x1 + 1) → singular normal equations.
    let collinear: Vec<Vec<Q>> = x
        .iter()
        .map(|r| vec![r[0].clone(), qi(2) * &r[0] + qi(1)])
        .collect();
    match ols(&y, &collinear, true) {
        Err(SymplexError::InvalidArgument { reason, .. }) => {
            assert!(reason.contains("rank deficient"), "{reason}");
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
    is_invalid(hat_matrix(&QMatrix::from_i64(&[
        &[1, 2],
        &[2, 4],
        &[3, 6],
    ])?));
    // Constant response.
    is_invalid(ols(&vec![qi(5); 10], &x, true));
    // WLS: a zero weight, a negative weight, a wrong length.
    let mut w = r3_weights();
    w[3] = Q::zero();
    is_invalid(wls(&y, &x, &w, true));
    w[3] = qi(-1);
    is_invalid(wls(&y, &x, &w, true));
    is_invalid(wls(&y, &x, &w[..9], true));
    // Intercept-only fit: F, ANOVA table are undefined (df_model = 0) but the mean is returned.
    let only = ols(&y, &vec![vec![]; 10], true)?;
    assert_eq!(only.coefficients, vec![q(66, 5)]);
    is_invalid(only.f_statistic());
    is_invalid(only.anova_table());
    is_invalid(only.f_test(&ctx));
    // A perfect fit keeps the exact pieces and errors on the undefined ones.
    let perfect = ols(&from_i64(&[1, 3, 5, 7]), &col(&[0, 1, 2, 3]), true)?;
    assert!(perfect.ssr.is_zero());
    is_invalid(perfect.p_values(&ctx));
    is_invalid(perfect.cooks_distance());
    is_invalid(perfect.durbin_watson());
    is_invalid(perfect.bic(&ctx));
    Ok(())
}

#[test]
fn polyfit_with_repeated_abscissae_and_exact_cubic() -> R {
    // x = [1, 1, 2, 2, 3, 3], y = [2, 4, 5, 7, 10, 12]: three distinct abscissae.
    // Fraction: degree 1 → [-4/3, 4]; degree 2 → [2, 0, 1] (numpy polyfit, reversed:
    //   [4.000000000000002, -1.3333333333333375]; [1.0000000000000016, -7.4e-15, 2.000000000000004])
    let x = from_i64(&[1, 1, 2, 2, 3, 3]);
    let y = from_i64(&[2, 4, 5, 7, 10, 12]);
    assert_eq!(polyfit(&x, &y, 1)?, vec![q(-4, 3), qi(4)]);
    assert_eq!(polyfit(&x, &y, 2)?, vec![qi(2), qi(0), qi(1)]);
    // Degree 3 needs four distinct abscissae: the Vandermonde system is singular.
    is_invalid(polyfit(&x, &y, 3));
    // Fewer points than degree + 1, mismatched lengths.
    is_invalid(polyfit(&x[..2], &y[..2], 2));
    is_invalid(polyfit(&x, &y[..5], 1));
    // Exact cubic through six points: y = 1 + x + x³ on x = -2..3
    // (numpy polyfit(…, 3) = [0.9999999999999992, 6.7e-16, 0.999999999999998, 0.9999999999999992])
    let x3 = from_i64(&[-2, -1, 0, 1, 2, 3]);
    let y3 = from_i64(&[-9, -1, 1, 3, 11, 31]);
    assert_eq!(polyfit(&x3, &y3, 3)?, vec![qi(1), qi(1), qi(0), qi(1)]);
    // Degree 0 is the mean.
    assert_eq!(polyfit(&x, &y, 0)?, vec![q(20, 3)]);
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Regression — logistic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn logit_single_regressor_matches_statsmodels() -> R {
    // L4: x = 1..14, y = [0,0,1,0,0,1,0,1,1,0,1,1,1,1]
    // statsmodels Logit(y, add_constant(x)).fit(): params [-2.3721818074823027, 0.37421386830492154],
    //   bse [1.513510176268291, 0.20049535284080267], tvalues [-1.5673378644411573, 1.866446593413339],
    //   pvalues [0.11703573929072965, 0.06197890948002244], llf -6.963840195940476,
    //   llnull -9.560713465857104, prsquared 0.2716191923532165, llr 5.193746539833256,
    //   aic 17.92768039188095, bic 19.20579505111147
    let x: Vec<Vec<f64>> = (1..=14).map(|i| vec![f64::from(i)]).collect();
    let y = [0u8, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1];
    let fit = logit(&y, &x, true, &LogitOpts::default())?;
    assert!(fit.converged);
    assert_eq!(
        (fit.nobs, fit.n_params(), fit.df_model, fit.df_resid),
        (14, 2, 1, 12)
    );
    close(fit.coefficients[0], -2.372_181_807_482_302_7, 1e-9);
    close(fit.coefficients[1], 0.374_213_868_304_921_54, 1e-9);
    close(fit.standard_errors[0], 1.513_510_176_268_291, 1e-8);
    close(fit.standard_errors[1], 0.200_495_352_840_802_67, 1e-9);
    close(fit.z_values[1], 1.866_446_593_413_339, 1e-8);
    close(fit.p_values[0], 0.117_035_739_290_729_65, 1e-9);
    close(fit.p_values[1], 0.061_978_909_480_022_44, 1e-9);
    close(fit.log_likelihood, -6.963_840_195_940_476, 1e-9);
    // llnull is the closed form 8 ln(4/7) + 6 ln(3/7) = -9.560713465806604 (numpy); statsmodels fits the
    // null model numerically and lands 5e-11 away, so only 1e-9 is claimed against it.
    close(fit.null_log_likelihood, -9.560_713_465_806_604, 1e-12);
    close(fit.null_log_likelihood, -9.560_713_465_857_104, 1e-9);
    close(fit.pseudo_r_squared, 0.271_619_192_353_216_5, 1e-9);
    close(fit.llr(), 5.193_746_539_833_256, 1e-9);
    close(fit.aic(), 17.927_680_391_880_95, 1e-9);
    close(fit.bic(), 19.205_795_051_111_47, 1e-9);
    close(fit.deviance, 2.0 * 6.963_840_195_940_476, 1e-9);
    // conf_int(0.05) = [[-5.338607243203022, 0.5942436282384165], [-0.01874980233070211, 0.7671775389405452]]
    let ci = fit.conf_int(0.95)?;
    close(ci[0].lower, -5.338_607_243_203_022, 1e-8);
    close(ci[1].upper, 0.767_177_538_940_545_2, 1e-8);
    // predict([[1, 7.5]]) = 0.6069291542126406; cov_params [[2.290713053667673, -0.2729081173790412], [., 0.04019838651075795]]
    close(fit.predict_proba(&[7.5])?, 0.606_929_154_212_640_6, 1e-9);
    close(
        fit.predict_log_odds(&[7.5])?,
        -2.372_181_807_482_302_7 + 7.5 * 0.374_213_868_304_921_54,
        1e-8,
    );
    close(fit.cov_params[0][1], -0.272_908_117_379_041_2, 1e-8);
    close(fit.cov_params[1][1], 0.040_198_386_510_757_95, 1e-9);
    assert_eq!(fit.fitted_probabilities.len(), 14);
    Ok(())
}

#[test]
fn logit_two_regressors_matches_statsmodels() -> R {
    // L5: x1 binary alternating, x2 = [2.5, 1, 3.5, 4, 1.5, 2, 5, 3, 4.5, 6, 0.5, 5.5, 2, 1.5, 3, 4.5, 6.5, 2.5],
    //   y = [0,1,0,1,0,0,1,1,0,1,0,1,1,0,0,1,1,0]
    // statsmodels: params [-3.8005143339180916, 1.9237036872777253, 0.8765940905886599],
    //   bse [1.880427813365127, 1.3048822056314568, 0.44700774972093893],
    //   tvalues [-2.021090257709423, 1.4742355125816198, 1.9610266066660054],
    //   pvalues [0.04327042266713722, 0.14041819132186797, 0.04987591923113645],
    //   llf -8.50844212672478, llnull -12.476649250079015, prsquared 0.3180507076713007,
    //   aic 23.01688425344956, bic 25.687999527138054, llr 7.93641424670847, iterations 7
    let x2 = [
        2.5, 1.0, 3.5, 4.0, 1.5, 2.0, 5.0, 3.0, 4.5, 6.0, 0.5, 5.5, 2.0, 1.5, 3.0, 4.5, 6.5, 2.5,
    ];
    let x: Vec<Vec<f64>> = x2
        .iter()
        .enumerate()
        .map(|(i, &v)| vec![f64::from(u8::from(i % 2 == 1)), v])
        .collect();
    let y = [
        false, true, false, true, false, false, true, true, false, true, false, true, true, false,
        false, true, true, false,
    ];
    let fit = logit(&y, &x, true, &LogitOpts::default())?;
    assert!(fit.converged);
    close(fit.coefficients[0], -3.800_514_333_918_091_6, 1e-9);
    close(fit.coefficients[1], 1.923_703_687_277_725_3, 1e-9);
    close(fit.coefficients[2], 0.876_594_090_588_659_9, 1e-9);
    close(fit.standard_errors[1], 1.304_882_205_631_456_8, 1e-8);
    close(fit.z_values[2], 1.961_026_606_666_005_4, 1e-8);
    close(fit.p_values[0], 0.043_270_422_667_137_22, 1e-9);
    close(fit.p_values[2], 0.049_875_919_231_136_45, 1e-9);
    close(fit.log_likelihood, -8.508_442_126_724_78, 1e-9);
    close(fit.null_log_likelihood, -12.476_649_250_079_015, 1e-12);
    close(fit.pseudo_r_squared, 0.318_050_707_671_300_7, 1e-9);
    close(fit.llr(), 7.936_414_246_708_47, 1e-9);
    close(fit.aic(), 23.016_884_253_449_56, 1e-9);
    close(fit.bic(), 25.687_999_527_138_054, 1e-9);
    // exp(params) = [0.0223592687678939, 6.846268006769803, 2.4027023680375357]
    let or = fit.odds_ratios();
    close(or[0], 0.022_359_268_767_893_9, 1e-9);
    close(or[1], 6.846_268_006_769_803, 1e-8);
    // conf_int(0.10) = [[-6.893542842952146, -0.7074858248840368], [-0.22263654139961297, 4.070043915955064],
    //   [0.14133177218475768, 1.611856408992562]]
    let ci = fit.conf_int(0.90)?;
    close(ci[0].lower, -6.893_542_842_952_146, 1e-8);
    close(ci[1].upper, 4.070_043_915_955_064, 1e-8);
    close(ci[2].lower, 0.141_331_772_184_757_68, 1e-8);
    // predict([[1, 1, 3]]) = 0.6798258563572843, log-odds 0.7529716251256133; fitted[10] = 0.03349736361861215
    close(
        fit.predict_proba(&[1.0, 3.0])?,
        0.679_825_856_357_284_3,
        1e-9,
    );
    close(
        fit.predict_log_odds(&[1.0, 3.0])?,
        0.752_971_625_125_613_3,
        1e-8,
    );
    close(fit.fitted_probabilities[10], 0.033_497_363_618_612_15, 1e-9);
    assert_eq!(fit.iterations, 7);
    Ok(())
}

#[test]
fn logit_separation_constant_predictor_and_invalid_input() -> R {
    // Complete separation: y = 0 for x < 4, 1 otherwise → ComputationFailed (statsmodels only warns
    // and returns params ≈ [-141.9, 40.6]).
    let x: Vec<Vec<f64>> = (0..8).map(|i| vec![f64::from(i)]).collect();
    let y = [0i64, 0, 0, 0, 1, 1, 1, 1];
    assert!(matches!(
        logit(&y, &x, true, &LogitOpts::default()),
        Err(SymplexError::ComputationFailed { .. })
    ));
    // A constant predictor plus the intercept is collinear.
    let yc = [1u8, 0, 0, 1, 1, 0, 1, 1, 1, 0];
    let konst = vec![vec![2.0]; 10];
    is_invalid(logit(&yc, &konst, true, &LogitOpts::default()));
    // The same constant predictor without an intercept is an intercept-only model:
    // statsmodels Logit(y, 2·1).fit().params = 0.2027325540540823 (= ln(6/4)/2), llf -6.730116670092564
    let fit = logit(&yc, &konst, false, &LogitOpts::default())?;
    close(fit.coefficients[0], 1.5_f64.ln() / 2.0, 1e-9);
    close(fit.log_likelihood, -6.730_116_670_092_564, 1e-9);
    close(fit.log_likelihood, fit.null_log_likelihood, 1e-9);
    assert_eq!((fit.df_model, fit.df_resid), (0, 9));
    // Invalid input: non-binary y, constant y, n ≤ p, empty, non-finite x, max_iter 0, tol ≤ 0.
    is_invalid(logit(&[0u8, 2, 1], &x[..3], true, &LogitOpts::default()));
    is_invalid(logit(&[1u8; 8], &x, true, &LogitOpts::default()));
    is_invalid(logit(&[0u8, 1], &x[..2], true, &LogitOpts::default()));
    is_invalid(logit::<u8>(&[], &[], true, &LogitOpts::default()));
    let mut nan = x.clone();
    nan[2][0] = f64::NAN;
    is_invalid(logit(&y, &nan, true, &LogitOpts::default()));
    is_invalid(logit(
        &yc,
        &konst,
        false,
        &LogitOpts {
            max_iter: 0,
            tol: 1e-10,
        },
    ));
    is_invalid(logit(
        &yc,
        &konst,
        false,
        &LogitOpts {
            max_iter: 10,
            tol: 0.0,
        },
    ));
    is_invalid(fit.predict_proba(&[]));
    is_invalid(fit.conf_int(1.0));
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Reliability — response tables
// ═══════════════════════════════════════════════════════════════════════════

/// T1: seven respondents × five Likert items.
fn t1() -> RatingTable {
    RatingTable::from_i64(&[
        &[3, 4, 3, 5, 4],
        &[2, 2, 3, 2, 3],
        &[5, 4, 5, 4, 5],
        &[1, 2, 1, 2, 2],
        &[4, 3, 4, 4, 3],
        &[3, 3, 2, 3, 4],
        &[5, 5, 4, 5, 4],
    ])
    .unwrap()
}

/// T2: eight respondents × four dichotomous items; totals [3, 2, 4, 1, 2, 1, 3, 0].
fn t2() -> RatingTable {
    RatingTable::from_i64(&[
        &[1, 1, 0, 1],
        &[1, 0, 0, 1],
        &[1, 1, 1, 1],
        &[0, 0, 0, 1],
        &[1, 1, 0, 0],
        &[0, 1, 0, 0],
        &[1, 1, 1, 0],
        &[0, 0, 0, 0],
    ])
    .unwrap()
}

#[test]
fn t1_cronbach_alpha_and_alpha_if_deleted_exact() -> R {
    let t = t1();
    // Fraction: α = 5/4 · (1 − Σ s²ᵢ / s²_X) = 815/872; pingouin cronbach_alpha = 0.9346330275229358
    assert_eq!(cronbach_alpha(&t)?, q(815, 872));
    close(qf(&cronbach_alpha(&t)?), 0.934_633_027_522_935_8, 1e-15);
    assert_eq!(cronbach_alpha_complete(&t)?, q(815, 872));
    // Fraction α without item j: [980/1089, 301/330, 194/209, 392/425, 448/481]
    assert_eq!(
        alpha_if_deleted(&t)?,
        qs(&[(980, 1089), (301, 330), (194, 209), (392, 425), (448, 481)])
    );
    // No item raises α when dropped.
    for a in alpha_if_deleted(&t)? {
        assert!(a < q(815, 872));
    }
    // A missing cell is rejected by `cronbach_alpha` but dropped by `_complete`.
    let mut rows: Vec<Vec<Option<Q>>> = t.rows().to_vec();
    rows.push(vec![
        Some(qi(3)),
        None,
        Some(qi(4)),
        Some(qi(4)),
        Some(qi(3)),
    ]);
    let with_gap = RatingTable::new(rows)?;
    is_invalid(cronbach_alpha(&with_gap));
    assert_eq!(cronbach_alpha_complete(&with_gap)?, q(815, 872));
    Ok(())
}

#[test]
fn t1_standardized_alpha_and_mean_inter_item_correlation() -> R {
    let ctx = Context::new();
    let t = t1();
    // numpy: mean of the 10 off-diagonal corrcoef entries r̄ = 0.754413508172773,
    //   5r̄/(1 + 4r̄) = 0.938873160847369
    close(
        average_inter_item_correlation(&ctx, &t)?.eval_f64()?,
        0.754_413_508_172_773,
        1e-12,
    );
    close(
        standardized_alpha(&ctx, &t)?.eval_f64()?,
        0.938_873_160_847_369,
        1e-12,
    );
    // Spearman–Brown of r̄ with k = 5 is exactly the standardized α.
    let r = average_inter_item_correlation(&ctx, &t)?;
    close(
        spearman_brown(&ctx, &r, 5).eval_f64()?,
        standardized_alpha(&ctx, &t)?.eval_f64()?,
        1e-14,
    );
    assert_eq!(
        spearman_brown(&ctx, &ctx.rational(1, 3), 3).simplify(),
        ctx.rational(3, 5)
    );
    Ok(())
}

#[test]
fn t1_guttman_lambda2_exceeds_alpha() -> R {
    let ctx = Context::new();
    let t = t1();
    // Fraction: s²_X = 218/7, Σ s²ᵢ = 55/7, k/(k−1) Σᵢ≠ⱼ s²ᵢⱼ = 42655/1176;
    // numpy: (218/7 − 55/7 + √(42655/1176)) / (218/7) = 0.9410914549211506
    let l2 = guttman_lambda2(&ctx, &t)?;
    close(l2.eval_f64()?, 0.941_091_454_921_150_6, 1e-12);
    assert!(l2.eval_f64()? > qf(&cronbach_alpha(&t)?));
    // The exact form: (163/7 + √(42655/1176)) / (218/7)
    let expect = (ctx.rational(163, 7) + ctx.rational(42_655, 1176).sqrt()) / ctx.rational(218, 7);
    close(l2.eval_f64()?, expect.eval_f64()?, 1e-15);
    Ok(())
}

#[test]
fn t1_split_half_with_an_odd_number_of_items() -> R {
    let ctx = Context::new();
    let t = t1();
    // OddEven: items {0, 2, 4} vs {1, 3}. Fraction r² = 10647/17480;
    //   numpy r = 0.7804460966907432, 2r/(1+r) = 0.8766860149726888
    let r = split_half_correlation(&ctx, &t, &SplitHalf::OddEven)?;
    close(r.eval_f64()?, 0.780_446_096_690_743_2, 1e-12);
    close((&r * &r).simplify().eval_f64()?, 10_647.0 / 17_480.0, 1e-14);
    close(
        split_half(&ctx, &t, &SplitHalf::OddEven)?.eval_f64()?,
        0.876_686_014_972_688_8,
        1e-12,
    );
    // FirstLast: items {0, 1} vs {2, 3, 4}. Fraction r² = 24649/27456;
    //   numpy r = 0.9475039285610877, 2r/(1+r) = 0.9730444336111306
    let r2 = split_half_correlation(&ctx, &t, &SplitHalf::FirstLast)?;
    close(r2.eval_f64()?, 0.947_503_928_561_087_7, 1e-12);
    close(
        split_half(&ctx, &t, &SplitHalf::FirstLast)?.eval_f64()?,
        0.973_044_433_611_130_6,
        1e-12,
    );
    // The same split expressed as a custom mask gives the same value.
    let custom = SplitHalf::Custom(vec![true, true, false, false, false]);
    assert_eq!(
        split_half(&ctx, &t, &custom)?,
        split_half(&ctx, &t, &SplitHalf::FirstLast)?
    );
    // Degenerate masks: all in one half, wrong length.
    is_invalid(split_half(&ctx, &t, &SplitHalf::Custom(vec![true; 5])));
    is_invalid(split_half(&ctx, &t, &SplitHalf::Custom(vec![false; 5])));
    is_invalid(split_half(&ctx, &t, &SplitHalf::Custom(vec![true, false])));
    Ok(())
}

#[test]
fn t1_item_total_correlations_exact_squares() -> R {
    let ctx = Context::new();
    let t = t1();
    // Fraction r² of item j with the total: [28561/30738, 1200/1417, 18769/24852, 69169/88944, 37249/52320]
    let it = item_total_correlation(&ctx, &t)?;
    let want = qs(&[
        (28_561, 30_738),
        (1200, 1417),
        (18_769, 24_852),
        (69_169, 88_944),
        (37_249, 52_320),
    ]);
    for (r, w) in it.iter().zip(&want) {
        assert_eq!((r * r).simplify(), ctx.from_ratio(w.clone()));
        assert!(r.eval_f64()? > 0.0);
    }
    // Corrected (item vs rest): r² = [14884/17061, 2209/2860, 891/1444, 1521/2312, 23409/38480];
    // numpy corrcoef(item 0, total − item 0) = 0.9340230397283211, item 4: 0.7799628169611649
    let cit = corrected_item_total_correlation(&ctx, &t)?;
    let want = qs(&[
        (14_884, 17_061),
        (2209, 2860),
        (891, 1444),
        (1521, 2312),
        (23_409, 38_480),
    ]);
    for (r, w) in cit.iter().zip(&want) {
        assert_eq!((r * r).simplify(), ctx.from_ratio(w.clone()));
    }
    close(cit[0].eval_f64()?, 0.934_023_039_728_321_1, 1e-12);
    close(cit[4].eval_f64()?, 0.779_962_816_961_164_9, 1e-12);
    // Corrected correlations are below the uncorrected ones.
    for (a, b) in cit.iter().zip(&it) {
        assert!(a.eval_f64()? < b.eval_f64()?);
    }
    Ok(())
}

#[test]
fn t1_totals_difficulty_discrimination_and_summary() -> R {
    let ctx = Context::new();
    let t = t1();
    // Fraction: totals [19, 12, 23, 8, 18, 15, 23]; item means [23/7, 23/7, 22/7, 25/7, 25/7]
    assert_eq!(total_scores(&t)?, from_i64(&[19, 12, 23, 8, 18, 15, 23]));
    assert_eq!(
        item_difficulty(&t)?,
        qs(&[(23, 7), (23, 7), (22, 7), (25, 7), (25, 7)])
    );
    // Thirds rule with n = 7: g = 2; upper = rows {2, 6} (both total 23), lower = rows {3, 1}.
    // D = upper mean − lower mean = [7/2, 5/2, 5/2, 5/2, 2]
    assert_eq!(
        item_discrimination_index(&t)?,
        qs(&[(7, 2), (5, 2), (5, 2), (5, 2), (2, 1)])
    );
    let s = item_response_summary(&ctx, &t)?;
    assert_eq!(s.len(), 5);
    assert_eq!(s[3].difficulty, q(25, 7));
    assert_eq!(s[0].discrimination, q(7, 2));
    assert_eq!(s[2].alpha_if_deleted, Some(q(194, 209)));
    assert_eq!(
        s[1].corrected_item_total
            .clone()
            .map(|r| (&r * &r).simplify()),
        Some(ctx.from_ratio(q(2209, 2860)))
    );
    assert!(s.iter().all(|i| i.item_total.is_some()));
    Ok(())
}

#[test]
fn t2_kr20_equals_alpha_and_rejects_non_binary_scores() -> R {
    let t = t2();
    // Fraction: KR-20 = 4/3 · (1 − Σ pᵢqᵢ / σ²_X) = 19/36 = α (0.5277777777777778)
    assert_eq!(kr20(&t)?, q(19, 36));
    assert_eq!(kr20(&t)?, cronbach_alpha(&t)?);
    // Fraction: item difficulties [5/8, 5/8, 1/4, 1/2]
    assert_eq!(item_difficulty(&t)?, qs(&[(5, 8), (5, 8), (1, 4), (1, 2)]));
    // KR-20 is only defined for 0/1 items: a Likert table is rejected.
    is_invalid(kr20(&t1()));
    let two = RatingTable::from_i64(&[&[1, 2], &[0, 1], &[1, 0]])?;
    is_invalid(kr20(&two));
    Ok(())
}

#[test]
fn t2_discrimination_breaks_boundary_ties_by_row_order() -> R {
    // Totals [3, 2, 4, 1, 2, 1, 3, 0], g = 2.  Upper: row 2 (4), then rows 0 and 6 tie at 3 → row 0.
    // Lower: row 7 (0), then rows 3 and 5 tie at 1 → row 3.  Fraction D = [1, 1, 1/2, 1/2].
    let t = t2();
    assert_eq!(
        item_discrimination_index(&t)?,
        qs(&[(1, 1), (1, 1), (1, 2), (1, 2)])
    );
    // Fewer than three respondents: no thirds.
    let tiny = RatingTable::from_i64(&[&[1, 0], &[0, 1]])?;
    is_invalid(item_discrimination_index(&tiny));
    Ok(())
}

#[test]
fn t2_cochrans_q_matches_statsmodels_and_degenerate_tables() -> R {
    let ctx = Context::new();
    // statsmodels cochrans_q(T2): statistic 3.6 (= 18/5), pvalue 0.308022171558994, df 3
    let r = cochrans_q(&ctx, &t2())?;
    assert_eq!(r.statistic_exact(), Some(q(18, 5)));
    assert_eq!(r.df, Some(ctx.int(3)));
    close(r.p_value_f64()?, 0.308_022_171_558_994, 1e-12);
    // Five identical rows [1, 0, 1]: statsmodels cochrans_q → statistic 10.0, pvalue 0.006737946999085468
    let row: &[i64] = &[1, 0, 1];
    let same = RatingTable::from_i64(&[row; 5])?;
    let r = cochrans_q(&ctx, &same)?;
    assert_eq!(r.statistic_exact(), Some(qi(10)));
    close(r.p_value_f64()?, 0.006_737_946_999_085_468, 1e-12);
    // Every subject constant → zero denominator; a Likert score is rejected.
    let flat = RatingTable::from_i64(&[&[1, 1, 1], &[0, 0, 0], &[1, 1, 1], &[0, 0, 0]])?;
    is_invalid(cochrans_q(&ctx, &flat));
    is_invalid(cochrans_q(&ctx, &t1()));
    Ok(())
}

#[test]
fn t2_point_biserial_and_lambda2() -> R {
    let ctx = Context::new();
    let t = t2();
    let totals = total_scores(&t)?;
    let item0 = from_i64(&[1, 1, 1, 0, 1, 0, 1, 0]);
    // scipy pointbiserialr(item 0, totals).statistic = 0.843274042711568; Fraction r² = 32/45
    let r = point_biserial(&ctx, &item0, &totals)?;
    close(r.eval_f64()?, 0.843_274_042_711_568, 1e-12);
    assert_eq!((&r * &r).simplify(), ctx.from_ratio(q(32, 45)));
    assert_eq!(item_total_correlation(&ctx, &t)?[0], r);
    // Fraction: s²_X = 12/7, Σ s²ᵢ = 29/28, k/(k−1) Σᵢ≠ⱼ s²ᵢⱼ = 51/392;
    //   numpy λ₂ = 0.6062396862158766
    close(
        guttman_lambda2(&ctx, &t)?.eval_f64()?,
        0.606_239_686_215_876_6,
        1e-12,
    );
    Ok(())
}

#[test]
fn equal_variance_items_make_alpha_equal_standardized_alpha() -> R {
    let ctx = Context::new();
    // T3: four dichotomous items each answered correctly by exactly four of eight respondents,
    // so every item has the same variance and α coincides with the standardized α.
    // Fraction α = 4/7; numpy r̄ = 0.25, 4r̄/(1 + 3r̄) = 0.5714285714285713
    let t3 = RatingTable::from_i64(&[
        &[1, 1, 1, 1],
        &[1, 1, 1, 0],
        &[1, 1, 0, 1],
        &[1, 0, 1, 0],
        &[0, 1, 0, 1],
        &[0, 0, 1, 1],
        &[0, 0, 0, 0],
        &[0, 0, 0, 0],
    ])?;
    assert_eq!(cronbach_alpha(&t3)?, q(4, 7));
    assert_eq!(
        standardized_alpha(&ctx, &t3)?.simplify(),
        ctx.rational(4, 7)
    );
    assert_eq!(
        average_inter_item_correlation(&ctx, &t3)?.simplify(),
        ctx.rational(1, 4)
    );
    Ok(())
}

#[test]
fn alpha_degenerate_inputs() -> R {
    let ctx = Context::new();
    // One item, one respondent: rejected.
    is_invalid(cronbach_alpha(&RatingTable::from_i64(&[&[1], &[2], &[3]])?));
    is_invalid(cronbach_alpha(&RatingTable::from_i64(&[&[1, 2, 3]])?));
    is_invalid(alpha_if_deleted(&RatingTable::from_i64(&[
        &[1, 2],
        &[2, 3],
        &[3, 5],
    ])?));
    // A constant item contributes zero variance; α is still defined.
    // Fraction: α of [[1,3,2],[1,4,3],[1,2,2],[1,5,4],[1,3,3]] = 51/74; pingouin 0.6891891891891891
    let konst =
        RatingTable::from_i64(&[&[1, 3, 2], &[1, 4, 3], &[1, 2, 2], &[1, 5, 4], &[1, 3, 3]])?;
    assert_eq!(cronbach_alpha(&konst)?, q(51, 74));
    // …but its correlations are undefined: the standardized α errors, the summary uses None.
    is_invalid(standardized_alpha(&ctx, &konst));
    let summary = item_response_summary(&ctx, &konst)?;
    assert!(summary[0].item_total.is_none());
    assert!(summary[1].item_total.is_some());
    // Negatively covarying items give a negative α: Fraction α([[1,5],[2,4],[3,3],[4,1],[5,2]]) = -18
    // (pingouin -18.0).
    let neg = RatingTable::from_i64(&[&[1, 5], &[2, 4], &[3, 3], &[4, 1], &[5, 2]])?;
    assert_eq!(cronbach_alpha(&neg)?, qi(-18));
    // Perfectly anti-correlated items: every total is 5, α undefined.
    let anti = RatingTable::from_i64(&[&[1, 4], &[2, 3], &[3, 2], &[4, 1]])?;
    is_invalid(cronbach_alpha(&anti));
    is_invalid(guttman_lambda2(&ctx, &anti));
    is_invalid(kr20(&RatingTable::from_i64(&[&[1, 0], &[0, 1], &[1, 0]])?));
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Reliability — κ inference, ordinal association, contingency diagnostics
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn kappa_3x3_confusion_matches_statsmodels() -> R {
    let ctx = Context::new();
    let k1 = vec![vec![15, 3, 2], vec![4, 12, 4], vec![1, 5, 14]];
    // statsmodels cohens_kappa(K1, return_results=True): kappa 0.525, var_kappa 0.008114583333333331,
    //   kappa_low 0.34844451942210664, kappa_upp 0.7015554805778934, z_value 5.751086853804245,
    //   pvalue_one_sided 4.433577189202342e-09, pvalue_two_sided 8.867154378404684e-09, kappa_max 1.0,
    //   var_kappa0 0.008333333333333331
    // Fraction: κ = 21/40, Var = 779/96000, Var₀ = 1/120, κ_max = 1
    let ci = kappa_ci_from_confusion(&ctx, &k1, 0.95)?;
    assert_eq!(ci.kappa, q(21, 40));
    assert_eq!(ci.variance, q(779, 96_000));
    close(
        ci.se.eval_f64()?,
        0.008_114_583_333_333_331_f64.sqrt(),
        1e-15,
    );
    close(ci.ci.lower, 0.348_444_519_422_106_64, 1e-12);
    close(ci.ci.upper, 0.701_555_480_577_893_4, 1e-12);
    assert_eq!(ci.confidence, 0.95);
    let t = kappa_test_from_confusion(&ctx, &k1, Alternative::TwoSided)?;
    assert_eq!(
        (&t.statistic * &t.statistic).simplify(),
        ctx.from_ratio(q(21, 40) * q(21, 40) * qi(120))
    );
    close(t.statistic_f64()?, 5.751_086_853_804_245, 1e-12);
    close(t.p_value_f64()?, 8.867_154_378_404_684e-9, 1e-20);
    let g = kappa_test_from_confusion(&ctx, &k1, Alternative::Greater)?;
    close(g.p_value_f64()?, 4.433_577_189_202_342e-9, 1e-20);
    let l = kappa_test_from_confusion(&ctx, &k1, Alternative::Less)?;
    close(l.p_value_f64()?, 1.0 - 4.433_577_189_202_342e-9, 1e-15);
    assert_eq!(kappa_maximum_from_confusion(&k1)?, Q::one());
    Ok(())
}

#[test]
fn kappa_with_an_empty_category_equals_the_reduced_table() -> R {
    let ctx = Context::new();
    // K2 has an unused third category (its row and column are zero).
    // statsmodels cohens_kappa(K2): kappa 0.5, var_kappa 0.037125, z_value 2.247332874877474, kappa_max 0.9
    // Fraction: κ = 1/2, Var = 297/8000, Var₀ = 99/2000, κ_max = 9/10 — identical to the 2×2 table.
    let k2 = vec![vec![8, 2, 0], vec![3, 7, 0], vec![0, 0, 0]];
    let k2b = vec![vec![8, 2], vec![3, 7]];
    let a = kappa_ci_from_confusion(&ctx, &k2, 0.95)?;
    let b = kappa_ci_from_confusion(&ctx, &k2b, 0.95)?;
    assert_eq!(a.kappa, q(1, 2));
    assert_eq!(a.variance, q(297, 8000));
    assert_eq!((a.kappa.clone(), a.variance.clone()), (b.kappa, b.variance));
    assert_eq!(a.ci, b.ci);
    let ta = kappa_test_from_confusion(&ctx, &k2, Alternative::Greater)?;
    let tb = kappa_test_from_confusion(&ctx, &k2b, Alternative::Greater)?;
    assert_eq!(ta.statistic, tb.statistic);
    close(ta.statistic_f64()?, 2.247_332_874_877_474, 1e-12);
    assert_eq!(kappa_maximum_from_confusion(&k2)?, q(9, 10));
    // Non-square, empty, and single-category tables are rejected.
    is_invalid(kappa_ci_from_confusion(
        &ctx,
        &[vec![1, 2, 3], vec![4, 5, 6]],
        0.95,
    ));
    is_invalid(kappa_ci_from_confusion(
        &ctx,
        &[vec![0, 0], vec![0, 0]],
        0.95,
    ));
    is_invalid(kappa_ci_from_confusion(&ctx, &[vec![7]], 0.95));
    is_invalid(kappa_ci_from_confusion(&ctx, &k2b, 1.0));
    Ok(())
}

#[test]
fn kappa_4x4_with_a_zero_diagonal_cell() -> R {
    let ctx = Context::new();
    let k3 = vec![
        vec![6, 1, 0, 1],
        vec![2, 0, 3, 1],
        vec![0, 2, 7, 1],
        vec![1, 1, 0, 4],
    ];
    // statsmodels: kappa 0.41087613293051356, var_kappa 0.013365679877302335,
    //   kappa_low 0.184284630752614, kappa_upp 0.6374676351084131, z_value 3.8397045268403383,
    //   pvalue_two_sided 0.00012318247281197994, kappa_max 0.9093655589123867
    // Fraction: κ = 136/331, Var = 160436445/12003612721, Var₀ = 18818/1643415, κ_max = 301/331
    let ci = kappa_ci_from_confusion(&ctx, &k3, 0.95)?;
    assert_eq!(ci.kappa, q(136, 331));
    assert_eq!(ci.variance, q(160_436_445, 12_003_612_721));
    close(ci.ci.lower, 0.184_284_630_752_614, 1e-12);
    close(ci.ci.upper, 0.637_467_635_108_413_1, 1e-12);
    let t = kappa_test_from_confusion(&ctx, &k3, Alternative::TwoSided)?;
    close(t.statistic_f64()?, 3.839_704_526_840_338_3, 1e-12);
    close(t.p_value_f64()?, 0.000_123_182_472_811_979_94, 1e-15);
    assert_eq!(kappa_maximum_from_confusion(&k3)?, q(301, 331));
    Ok(())
}

#[test]
fn kappa_from_ratings_with_a_99_percent_interval() -> R {
    let ctx = Context::new();
    // Two raters, 15 items, categories 0..2 → confusion [[3, 1, 1], [0, 4, 1], [0, 1, 4]].
    // statsmodels: kappa 0.6, var_kappa 0.02848, kappa_low 0.2692361156151281, kappa_upp 0.9307638843848716,
    //   z_value 3.3541019662496834, pvalue_one_sided 0.0003981150787954078, pvalue_two_sided 0.0007962301575908156
    // Fraction: κ = 3/5, Var = 89/3125, Var₀ = 4/125, κ_max = 4/5; scipy norm.isf(0.005): 99 % CI
    //   (0.1653025705193505, 1.0346974294806495)
    let a = from_i64(&[0, 1, 2, 1, 0, 2, 2, 1, 0, 0, 1, 2, 1, 2, 0]);
    let b = from_i64(&[0, 1, 2, 2, 0, 2, 1, 1, 0, 1, 1, 2, 1, 2, 2]);
    let ci = cohen_kappa_ci(&ctx, &a, &b, 0.95)?;
    assert_eq!(ci.kappa, q(3, 5));
    assert_eq!(ci.variance, q(89, 3125));
    close(ci.ci.lower, 0.269_236_115_615_128_1, 1e-12);
    close(ci.ci.upper, 0.930_763_884_384_871_6, 1e-12);
    let ci99 = cohen_kappa_ci(&ctx, &a, &b, 0.99)?;
    close(ci99.ci.lower, 0.165_302_570_519_350_5, 1e-12);
    close(ci99.ci.upper, 1.034_697_429_480_649_5, 1e-12);
    let t = kappa_test(&ctx, &a, &b, Alternative::Greater)?;
    close(t.statistic_f64()?, 3.354_101_966_249_683_4, 1e-12);
    close(t.p_value_f64()?, 0.000_398_115_078_795_407_8, 1e-15);
    close(
        kappa_test(&ctx, &a, &b, Alternative::TwoSided)?.p_value_f64()?,
        2.0 * t.p_value_f64()?,
        1e-18,
    );
    assert_eq!(cohen_kappa_maximum(&a, &b)?, q(4, 5));
    is_invalid(cohen_kappa_ci(&ctx, &a, &b[..14], 0.95));
    is_invalid(kappa_test(&ctx, &[], &[], Alternative::TwoSided));
    Ok(())
}

#[test]
fn ordinal_association_o1_matches_scipy() -> R {
    // O1: x = [2,3,1,4,4,2,5,3,1,5,2,4,3,1,5], y = [1,3,2,4,3,1,5,2,1,4,3,5,3,2,4]
    // Fraction pair counts: C 71, D 7, T_x 11, T_y 12, T_xy 4 (105 pairs);
    //   γ = 32/39, D_{Y|X} = 32/45, D_{X|Y} = 64/89, D_sym = 128/179, τ-c = 32/45 (m = 5, n = 15)
    // scipy: somersd(x, y) 0.7111111111111111, somersd(y, x) 0.7191011235955056,
    //   kendalltau(x, y, variant='c') 0.7111111111111111
    let x = from_i64(&[2, 3, 1, 4, 4, 2, 5, 3, 1, 5, 2, 4, 3, 1, 5]);
    let y = from_i64(&[1, 3, 2, 4, 3, 1, 5, 2, 1, 4, 3, 5, 3, 2, 4]);
    let c = concordance_counts(&x, &y)?;
    assert_eq!(
        (c.concordant, c.discordant, c.ties_x, c.ties_y, c.ties_both),
        (71, 7, 11, 12, 4)
    );
    assert_eq!(c.pairs(), 105);
    assert_eq!(goodman_kruskal_gamma(&x, &y)?, q(32, 39));
    assert_eq!(somers_d(&x, &y, Dependent::Y)?, q(32, 45));
    assert_eq!(somers_d(&x, &y, Dependent::X)?, q(64, 89));
    assert_eq!(somers_d(&x, &y, Dependent::Symmetric)?, q(128, 179));
    assert_eq!(kendall_tau_c(&x, &y)?, q(32, 45));
    close(
        qf(&somers_d(&x, &y, Dependent::X)?),
        0.719_101_123_595_505_6,
        1e-15,
    );
    // Swapping the roles swaps the two asymmetric D's; symmetric and τ-c are invariant.
    assert_eq!(somers_d(&y, &x, Dependent::Y)?, q(64, 89));
    assert_eq!(somers_d(&y, &x, Dependent::Symmetric)?, q(128, 179));
    assert_eq!(kendall_tau_c(&y, &x)?, q(32, 45));
    Ok(())
}

#[test]
fn ordinal_measures_on_all_tied_and_perfectly_ordered_data() -> R {
    // Constant x against a varying y: every pair is tied on x (8 on x only, 2 on both).
    let xc = from_i64(&[2, 2, 2, 2, 2]);
    let yv = from_i64(&[1, 2, 2, 3, 1]);
    let c = concordance_counts(&xc, &yv)?;
    assert_eq!(
        (c.concordant, c.discordant, c.ties_x, c.ties_y, c.ties_both),
        (0, 0, 8, 0, 2)
    );
    is_invalid(goodman_kruskal_gamma(&xc, &yv));
    is_invalid(somers_d(&xc, &yv, Dependent::Y));
    is_invalid(kendall_tau_c(&xc, &yv));
    // With x as the dependent variable the denominator is T_x = 8: D = 0 (scipy returns nan here).
    assert_eq!(somers_d(&xc, &yv, Dependent::X)?, Q::zero());
    assert_eq!(somers_d(&xc, &yv, Dependent::Symmetric)?, Q::zero());
    // Both constant: every pair tied on both.
    is_invalid(somers_d(&xc, &xc, Dependent::Symmetric));
    // Perfect negative association with ties: x = [1,1,2,2,3,3], y = [3,3,2,2,1,1]
    // scipy: kendalltau(variant='c') -1.0, somersd -1.0; Fraction: C 0, D 12, T_xy 3
    let xn = from_i64(&[1, 1, 2, 2, 3, 3]);
    let yn = from_i64(&[3, 3, 2, 2, 1, 1]);
    assert_eq!(goodman_kruskal_gamma(&xn, &yn)?, qi(-1));
    assert_eq!(somers_d(&xn, &yn, Dependent::Y)?, qi(-1));
    assert_eq!(kendall_tau_c(&xn, &yn)?, qi(-1));
    is_invalid(concordance_counts(&xn, &yn[..5]));
    is_invalid(concordance_counts(&xn[..1], &yn[..1]));
    Ok(())
}

#[test]
fn contingency_3x3_diagnostics_match_statsmodels_table() -> R {
    let ctx = Context::new();
    let t = counts(&[&[12, 7, 3], &[4, 9, 10], &[6, 5, 14]]);
    // Fraction expected: [[242/35, 33/5, 297/35], [253/35, 69/10, 621/70], [55/7, 15/2, 135/14]]
    // statsmodels Table(t).fittedvalues[1][2] = 8.87142857142857
    let e = expected_counts(&t)?;
    assert_eq!(e[0], qs(&[(242, 35), (33, 5), (297, 35)]));
    assert_eq!(e[1], qs(&[(253, 35), (69, 10), (621, 70)]));
    assert_eq!(e[2], qs(&[(55, 7), (15, 2), (135, 14)]));
    // chi2_contribs: Fraction [[15842/4235, 4/165, 4096/1155], [12769/8855, 147/230, 6241/43470],
    //   [169/385, 5/6, 3721/1890]]; sum = 1600138/125235; scipy chi2_contingency(correction=False)
    //   statistic 12.777083083802452
    let c = chi2_contributions(&t)?;
    assert_eq!(c[0], qs(&[(15_842, 4235), (4, 165), (4096, 1155)]));
    assert_eq!(c[1], qs(&[(12_769, 8855), (147, 230), (6241, 43_470)]));
    assert_eq!(c[2], qs(&[(169, 385), (5, 6), (3721, 1890)]));
    let total = c.iter().flatten().fold(Q::zero(), |a, v| a + v);
    assert_eq!(total, q(1_600_138, 125_235));
    close(qf(&total), 12.777_083_083_802_452, 1e-13);
    // resid_pearson[0][0] = 1.9340972041956555, [2][1] = -0.9128709291752769; squares are the contributions
    let r = standardized_residuals(&ctx, &t)?;
    close(r[0][0].eval_f64()?, 1.934_097_204_195_655_5, 1e-12);
    close(r[2][1].eval_f64()?, -0.912_870_929_175_276_9, 1e-12);
    assert_eq!(
        (&r[1][2] * &r[1][2]).simplify(),
        ctx.from_ratio(q(6241, 43_470))
    );
    // standardized_resids (adjusted) [0][2] = -2.901564141215233, [1][2] = 0.589993401421159;
    //   Fraction: variance of cell (1, 2) = 1255041/343000, numerator 79/70
    let a = adjusted_residuals(&ctx, &t)?;
    close(a[0][2].eval_f64()?, -2.901_564_141_215_233, 1e-12);
    close(a[1][2].eval_f64()?, 0.589_993_401_421_159, 1e-12);
    assert_eq!(
        (&a[1][2] * &a[1][2]).simplify(),
        ctx.from_ratio(q(79, 70) * q(79, 70) / q(1_255_041, 343_000))
    );
    // A single-row table: expected = observed, adjusted residuals undefined; an empty column errors.
    assert_eq!(
        expected_counts(&counts(&[&[4, 6, 10]]))?,
        counts(&[&[4, 6, 10]])
    );
    is_invalid(adjusted_residuals(&ctx, &counts(&[&[4, 6, 10]])));
    is_invalid(expected_counts(&counts(&[&[4, 0], &[6, 0]])));
    Ok(())
}

#[test]
fn pearson_inference_on_new_data_matches_scipy() -> R {
    let ctx = Context::new();
    // x = [3, 7, 2, 9, 5, 8, 1, 6], y = [4, 8, 3, 7, 6, 9, 2, 5]
    // Fraction: r² = 1183/1413, t² = 3549/115; scipy pearsonr: statistic 0.9150004157335883,
    //   pvalue 0.001439079569732105, alternative='greater' 0.0007195397848660525, 'less' 0.999280460215134;
    //   t = 5.555256030572965
    let x = from_i64(&[3, 7, 2, 9, 5, 8, 1, 6]);
    let y = from_i64(&[4, 8, 3, 7, 6, 9, 2, 5]);
    let r = pearson_test(&ctx, &x, &y, Alternative::TwoSided)?;
    assert_eq!(
        (&r.statistic * &r.statistic).simplify(),
        ctx.from_ratio(q(1183, 1413))
    );
    close(r.statistic_f64()?, 0.915_000_415_733_588_3, 1e-14);
    close(r.p_value_f64()?, 0.001_439_079_569_732_105, 1e-14);
    assert_eq!(r.df, Some(ctx.int(6)));
    close(
        pearson_test(&ctx, &x, &y, Alternative::Greater)?.p_value_f64()?,
        0.000_719_539_784_866_052_5,
        1e-14,
    );
    close(
        pearson_test(&ctx, &x, &y, Alternative::Less)?.p_value_f64()?,
        0.999_280_460_215_134,
        1e-14,
    );
    let t = pearson_t_statistic(&ctx, &x, &y)?;
    close(t.eval_f64()?, 5.555_256_030_572_965, 1e-12);
    assert_eq!((&t * &t).simplify(), ctx.from_ratio(q(3549, 115)));
    // pearsonr(x, y).confidence_interval(0.95) = (0.5920982270486052, 0.9847379351580114);
    //   (0.90) = (0.6760551616947962, 0.9798191882801129)
    let ci = pearson_ci(0.915_000_415_733_588_3, 8, 0.95)?;
    close(ci.lower, 0.592_098_227_048_605_2, 1e-12);
    close(ci.upper, 0.984_737_935_158_011_4, 1e-12);
    let ci90 = pearson_ci(0.915_000_415_733_588_3, 8, 0.90)?;
    close(ci90.lower, 0.676_055_161_694_796_2, 1e-12);
    close(ci90.upper, 0.979_819_188_280_112_9, 1e-12);
    Ok(())
}

#[test]
fn pearson_ci_is_the_inverse_fisher_z_and_fisher_z_is_atanh() -> R {
    let ctx = Context::new();
    // atanh(0.6) = ½ ln(1.6/0.4) = ln 2 = 0.6931471805599453 (numpy arctanh)
    let z = fisher_z(&ctx.rational(3, 5));
    close(z.eval_f64()?, std::f64::consts::LN_2, 1e-15);
    close(z.eval_f64()?, ctx.int(2).ln().eval_f64()?, 1e-15);
    // pearson_ci(r, n, c) = tanh(atanh(r) ∓ z_{(1+c)/2}/√(n−3)); for r = -0.35, n = 30, 99 %:
    //   scipy: (-0.6968560524345225, 0.12954280759140732)
    let ci = pearson_ci(-0.35, 30, 0.99)?;
    close(ci.lower, -0.696_856_052_434_522_5, 1e-12);
    close(ci.upper, 0.129_542_807_591_407_32, 1e-12);
    // Invert: atanh of the limits differs from atanh(r) by exactly ±z/√27.
    let zc = 2.575_829_303_548_900_4; // scipy norm.isf(0.005)
    close(
        ci.lower.atanh(),
        (-0.35_f64).atanh() - zc / 27.0_f64.sqrt(),
        1e-12,
    );
    close(
        ci.upper.atanh(),
        (-0.35_f64).atanh() + zc / 27.0_f64.sqrt(),
        1e-12,
    );
    is_invalid(pearson_ci(1.0, 30, 0.95));
    is_invalid(pearson_ci(0.3, 3, 0.95));
    is_invalid(pearson_ci(0.3, 30, 1.0));
    is_invalid(pearson_ci(f64::NAN, 30, 0.95));
    Ok(())
}

#[test]
fn pearson_negative_correlation_and_comparing_two_correlations() -> R {
    let ctx = Context::new();
    // x = [3, 7, 2, 9, 5, 8, 1, 6], y = [9, 3, 8, 2, 6, 3, 10, 4]: Fraction r² = 235225/248217, s_xy = -485/64
    // scipy pearsonr: statistic -0.9734776329539765, pvalue 4.5719096326151415e-05,
    //   'less' 2.2859548163075708e-05, 'greater' 0.9999771404518369
    let x = from_i64(&[3, 7, 2, 9, 5, 8, 1, 6]);
    let y = from_i64(&[9, 3, 8, 2, 6, 3, 10, 4]);
    let r = pearson_test(&ctx, &x, &y, Alternative::TwoSided)?;
    assert!(r.statistic_f64()? < 0.0);
    assert_eq!(
        (&r.statistic * &r.statistic).simplify(),
        ctx.from_ratio(q(235_225, 248_217))
    );
    close(r.statistic_f64()?, -0.973_477_632_953_976_5, 1e-14);
    close(r.p_value_f64()?, 4.571_909_632_615_141_5e-5, 1e-17);
    close(
        pearson_test(&ctx, &x, &y, Alternative::Less)?.p_value_f64()?,
        2.285_954_816_307_570_8e-5,
        1e-17,
    );
    close(
        pearson_test(&ctx, &x, &y, Alternative::Greater)?.p_value_f64()?,
        0.999_977_140_451_836_9,
        1e-14,
    );
    assert!(pearson_t_statistic(&ctx, &x, &y)?.eval_f64()? < 0.0);
    // compare_two_correlations(0.25, 40, 0.6, 35): z = (atanh 0.25 − atanh 0.6)/√(1/37 + 1/32)
    //   scipy.stats.norm: z -1.813267812321463, two-sided 0.06979052523058814, less 0.03489526261529407,
    //   greater 0.9651047373847059
    let c = compare_two_correlations(&ctx, 0.25, 40, 0.6, 35, Alternative::TwoSided)?;
    close(c.statistic_f64()?, -1.813_267_812_321_463, 1e-12);
    close(c.p_value_f64()?, 0.069_790_525_230_588_14, 1e-12);
    close(
        compare_two_correlations(&ctx, 0.25, 40, 0.6, 35, Alternative::Less)?.p_value_f64()?,
        0.034_895_262_615_294_07,
        1e-12,
    );
    close(
        compare_two_correlations(&ctx, 0.25, 40, 0.6, 35, Alternative::Greater)?.p_value_f64()?,
        0.965_104_737_384_705_9,
        1e-12,
    );
    is_invalid(compare_two_correlations(
        &ctx,
        0.25,
        3,
        0.6,
        35,
        Alternative::TwoSided,
    ));
    is_invalid(compare_two_correlations(
        &ctx,
        -1.0,
        40,
        0.6,
        35,
        Alternative::TwoSided,
    ));
    // Constant samples and |r| = 1.
    is_invalid(pearson_test(
        &ctx,
        &x,
        &vec![qi(2); 8],
        Alternative::TwoSided,
    ));
    is_invalid(pearson_t_statistic(&ctx, &x, &x));
    let perfect = pearson_test(&ctx, &x, &x, Alternative::Less)?;
    assert_eq!(perfect.p_value, ctx.one());
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. ANOVA
// ═══════════════════════════════════════════════════════════════════════════

// BAL2: balanced 3 × 2 design, two replicates per cell (N = 12).
//   df = DataFrame(A=[0,0,0,0,1,1,1,1,2,2,2,2], B=[0,0,1,1]*3, y=[7,9,12,10,8,6,15,13,11,13,9,11])
fn bal2() -> TwoWayData {
    TwoWayData::from_i64(&[
        &[&[7, 9], &[12, 10]],
        &[&[8, 6], &[15, 13]],
        &[&[11, 13], &[9, 11]],
    ])
    .unwrap()
}

// UNB2: unbalanced 2 × 2 design, cell sizes 3, 2 / 4, 3 (N = 12).
//   A=[0,0,0,0,0,1,1,1,1,1,1,1], B=[0,0,0,1,1,0,0,0,0,1,1,1], y=[3,5,4,8,6,7,9,8,10,12,11,15]
fn unb2() -> TwoWayData {
    TwoWayData::from_i64(&[&[&[3, 5, 4], &[8, 6]], &[&[7, 9, 8, 10], &[12, 11, 15]]]).unwrap()
}

// RM6: six subjects × three conditions.
fn rm6() -> Vec<Vec<Q>> {
    vec![
        from_i64(&[8, 10, 13]),
        from_i64(&[6, 7, 9]),
        from_i64(&[9, 12, 11]),
        from_i64(&[5, 8, 10]),
        from_i64(&[7, 7, 12]),
        from_i64(&[10, 13, 15]),
    ]
}

// G4: four groups of sizes 5, 4, 6, 4.
fn g4() -> Vec<Vec<Q>> {
    vec![
        from_i64(&[12, 15, 11, 14, 13]),
        from_i64(&[18, 16, 20, 17]),
        from_i64(&[10, 9, 13, 12, 11, 10]),
        from_i64(&[21, 19, 22, 20]),
    ]
}

#[test]
fn bal2_two_way_all_types_coincide_and_are_exact() -> R {
    let ctx = Context::new();
    // statsmodels anova_lm(ols('y ~ C(A) * C(B)'), typ=1/2) and typ=3 with Sum contrasts:
    //   C(A) 4.666666666666667 df 2 F 1.1666666666666667, C(B) 21.333333333333336 df 1 F 10.666666666666668,
    //   C(A):C(B) 40.66666666666667 df 2 F 10.166666666666666, Residual 12.0 df 6
    // Fraction: SS_A 14/3, SS_B 64/3, SS_AB 122/3, SS_res 12, SS_total 236/3, F 7/6, 32/3, 61/6
    let d = bal2();
    assert!(d.is_balanced());
    assert_eq!((d.a_levels(), d.b_levels(), d.n_obs()), (3, 2, 12));
    let t2 = anova_two_way(&ctx, &d)?;
    for ss in [SsType::TypeI, SsType::TypeII, SsType::TypeIII] {
        let t = anova_two_way_with(&ctx, &d, ss)?;
        assert_eq!(t.ss_type, ss);
        assert_eq!(t.factor_a.ss, q(14, 3));
        assert_eq!(t.factor_b.ss, q(64, 3));
        assert_eq!(t.interaction.ss, q(122, 3));
        assert_eq!(t.residual.ss, qi(12));
        assert_eq!(t.total.ss, q(236, 3));
        assert_eq!(t.factor_a.f, Some(q(7, 6)));
        assert_eq!(t.factor_b.f, Some(q(32, 3)));
        assert_eq!(t.interaction.f, Some(q(61, 6)));
        assert_eq!(
            (
                t.factor_a.df,
                t.factor_b.df,
                t.interaction.df,
                t.residual.df,
                t.total.df
            ),
            (2, 1, 2, 6, 11)
        );
        if ss != SsType::TypeII {
            assert_eq!(t.factor_a.p_value, t2.factor_a.p_value);
        }
    }
    // PR(>F): 0.3732479999999989, 0.01712000000000003, 0.011828678867189038
    close(t2.factor_a.p_value_f64()?, 0.373_247_999_999_998_9, 1e-12);
    close(t2.factor_b.p_value_f64()?, 0.017_120_000_000_000_03, 1e-12);
    close(
        t2.interaction.p_value_f64()?,
        0.011_828_678_867_189_038,
        1e-12,
    );
    // Effect sizes: η² = SS/SS_total, partial η² = SS/(SS + SS_res)
    assert_eq!(t2.interaction.eta_squared, Some(q(122, 3) / q(236, 3)));
    assert_eq!(
        t2.interaction.partial_eta_squared,
        Some(q(122, 3) / (q(122, 3) + qi(12)))
    );
    assert_eq!(t2.grand_mean, q(31, 3));
    assert_eq!(t2.cell_means[1], vec![qi(7), qi(14)]);
    assert_eq!(t2.residual.ms, Some(qi(2)));
    assert!(t2.total.ms.is_none() && t2.residual.f.is_none());
    is_invalid(t2.residual.p_value_f64());
    Ok(())
}

#[test]
fn bal2_type1_sums_match_the_ols_anova_table_on_the_dummy_design() -> R {
    let ctx = Context::new();
    let d = bal2();
    let t1 = anova_two_way_with(&ctx, &d, SsType::TypeI)?;
    // Treatment-coded full model: intercept, A1, A2, B1, A1B1, A2B1.
    let mut y = Vec::new();
    let mut rows = Vec::new();
    for (a, row) in d.cells().iter().enumerate() {
        for (b, cell) in row.iter().enumerate() {
            for v in cell {
                y.push(v.clone());
                let a1 = qi(i64::from(a == 1));
                let a2 = qi(i64::from(a == 2));
                let b1 = qi(i64::from(b == 1));
                rows.push(vec![
                    a1.clone(),
                    a2.clone(),
                    b1.clone(),
                    &a1 * &b1,
                    &a2 * &b1,
                ]);
            }
        }
    }
    let fit = ols(&y, &rows, true)?;
    let table = fit.anova_table()?;
    // statsmodels: ess 66.66666666666667, ssr 12.0, fvalue 6.666666666666667, f_pvalue 0.019421043826744488
    assert_eq!(
        table.ss_model,
        &t1.factor_a.ss + &t1.factor_b.ss + &t1.interaction.ss
    );
    assert_eq!(table.ss_model, qi(200) / qi(3));
    assert_eq!(table.ss_resid, t1.residual.ss);
    assert_eq!(table.ss_total, t1.total.ss);
    assert_eq!((table.df_model, table.df_resid), (5, 6));
    assert_eq!(table.f, q(20, 3));
    close(
        fit.f_test(&ctx)?.p_value_f64()?,
        0.019_421_043_826_744_488,
        1e-12,
    );
    // The fitted values are the cell means.
    assert_eq!(fit.fitted[2], t1.cell_means[0][1]);
    assert_eq!(fit.fitted[11], t1.cell_means[2][1]);
    Ok(())
}

#[test]
fn unb2_type_i_ii_iii_sums_of_squares_exact() -> R {
    let ctx = Context::new();
    let d = unb2();
    assert!(!d.is_balanced());
    // statsmodels typ=1: C(A) 75.4380952380952 F 34.160646900269526 p 0.0003850800188029954,
    //   C(B) 39.601120448179266 F 17.932582844458537 p 0.0028580580418120574,
    //   C(A):C(B) 0.9607843137254969 F 0.435072142064376 p 0.528038204258251, Residual 17.666666666666664 df 8
    // Fraction: SS_A 7921/105, SS_B|A 70688/1785, SS_AB 49/51, SS_res 53/3, SS_total 401/3
    let t1 = anova_two_way_with(&ctx, &d, SsType::TypeI)?;
    assert_eq!(t1.factor_a.ss, q(7921, 105));
    assert_eq!(t1.factor_b.ss, q(70_688, 1785));
    assert_eq!(t1.interaction.ss, q(49, 51));
    assert_eq!(t1.residual.ss, q(53, 3));
    assert_eq!(t1.total.ss, q(401, 3));
    assert_eq!(t1.residual.df, 8);
    close(
        t1.factor_a.p_value_f64()?,
        0.000_385_080_018_802_995_4,
        1e-13,
    );
    close(
        t1.factor_b.p_value_f64()?,
        0.002_858_058_041_812_057_4,
        1e-13,
    );
    close(t1.interaction.p_value_f64()?, 0.528_038_204_258_251, 1e-12);
    assert_eq!(t1.interaction.f, Some(q(392, 901)));
    // typ=2: C(A) 72.28683473389346 F 32.7336610115744 p 0.00044335298903804807; Fraction SS_A|B 129032/1785
    let t2 = anova_two_way_with(&ctx, &d, SsType::TypeII)?;
    assert_eq!(t2.factor_a.ss, q(129_032, 1785));
    assert_eq!(t2.factor_b.ss, q(70_688, 1785));
    assert_eq!(t2.interaction.ss, q(49, 51));
    close(
        t2.factor_a.p_value_f64()?,
        0.000_443_352_989_038_048_07,
        1e-13,
    );
    assert_eq!(anova_two_way(&ctx, &d)?, t2);
    // typ=3 (Sum contrasts): C(A, Sum) 72.96078431372563 F 33.038845726970095 p 0.00043000956044663086,
    //   C(B, Sum) 36.25490196078422 F 16.417314095449456 p 0.0036748709688684713; Fraction 3721/51, 1849/51
    let t3 = anova_two_way_with(&ctx, &d, SsType::TypeIII)?;
    assert_eq!(t3.factor_a.ss, q(3721, 51));
    assert_eq!(t3.factor_b.ss, q(1849, 51));
    assert_eq!(t3.interaction.ss, q(49, 51));
    close(
        t3.factor_a.p_value_f64()?,
        0.000_430_009_560_446_630_86,
        1e-13,
    );
    close(
        t3.factor_b.p_value_f64()?,
        0.003_674_870_968_868_471_3,
        1e-13,
    );
    // Cell means [[4, 7], [17/2, 38/3]], grand mean 49/6.
    assert_eq!(
        t3.cell_means,
        vec![vec![qi(4), qi(7)], vec![q(17, 2), q(38, 3)]]
    );
    assert_eq!(t3.grand_mean, q(49, 6));
    Ok(())
}

#[test]
fn unb2_type1_decomposes_the_total_and_agrees_with_one_way_on_cells() -> R {
    let ctx = Context::new();
    let d = unb2();
    let t1 = anova_two_way_with(&ctx, &d, SsType::TypeI)?;
    let t2 = anova_two_way_with(&ctx, &d, SsType::TypeII)?;
    // Only the sequential sums add up to the total.
    assert_eq!(
        &t1.factor_a.ss + &t1.factor_b.ss + &t1.interaction.ss + &t1.residual.ss,
        t1.total.ss
    );
    assert_ne!(
        &t2.factor_a.ss + &t2.factor_b.ss + &t2.interaction.ss + &t2.residual.ss,
        t2.total.ss
    );
    // Type I SS_A is the one-way ANOVA between-groups SS on A alone.
    let by_a: Vec<Vec<Q>> = (0..2)
        .map(|a| d.cells()[a].iter().flatten().cloned().collect())
        .collect();
    assert_eq!(anova_one_way(&ctx, &by_a)?.ss_between, t1.factor_a.ss);
    // The four cells as groups: within = residual, between = SS_A + SS_B|A + SS_AB.
    let cells: Vec<Vec<Q>> = d.cells().iter().flatten().cloned().collect();
    let one = anova_one_way(&ctx, &cells)?;
    assert_eq!(one.ss_within, t1.residual.ss);
    assert_eq!(
        one.ss_between,
        &t1.factor_a.ss + &t1.factor_b.ss + &t1.interaction.ss
    );
    assert_eq!(one.df_within, t1.residual.df);
    // The OLS on the treatment-coded design agrees: statsmodels ess 116.0, ssr 17.666666666666664
    let mut y = Vec::new();
    let mut rows = Vec::new();
    for (a, row) in d.cells().iter().enumerate() {
        for (b, cell) in row.iter().enumerate() {
            for v in cell {
                y.push(v.clone());
                let (ai, bi) = (qi(a as i64), qi(b as i64));
                rows.push(vec![ai.clone(), bi.clone(), &ai * &bi]);
            }
        }
    }
    let table = ols(&y, &rows, true)?.anova_table()?;
    assert_eq!(table.ss_model, qi(116));
    assert_eq!(table.ss_resid, q(53, 3));
    Ok(())
}

#[test]
fn two_way_error_paths() -> R {
    let ctx = Context::new();
    // One level of a factor.
    is_invalid(TwoWayData::from_i64(&[&[&[1, 2], &[3, 4]]]));
    is_invalid(TwoWayData::from_i64(&[&[&[1, 2]], &[&[3, 4]]]));
    // An empty cell, ragged rows.
    is_invalid(TwoWayData::from_i64(&[
        &[&[1, 2], &[]],
        &[&[3, 4], &[5, 6]],
    ]));
    is_invalid(TwoWayData::from_i64(&[&[&[1, 2], &[3]], &[&[5, 6]]]));
    is_invalid(TwoWayData::from_long(&[]));
    // One replicate per cell: the interaction saturates the model (df_resid = 0).
    let saturated = TwoWayData::from_i64(&[&[&[1], &[2], &[4]], &[&[3], &[5], &[6]]])?;
    is_invalid(anova_two_way(&ctx, &saturated));
    // Constant response; zero within-cell variance with a real effect.
    let flat = TwoWayData::from_i64(&[&[&[3, 3], &[3, 3]], &[&[3, 3], &[3, 3]]])?;
    is_invalid(anova_two_way(&ctx, &flat));
    let zero_within = TwoWayData::from_i64(&[&[&[1, 1], &[2, 2]], &[&[4, 4], &[9, 9]]])?;
    is_invalid(anova_two_way_with(&ctx, &zero_within, SsType::TypeIII));
    Ok(())
}

#[test]
fn rm6_repeated_measures_exact_and_matches_statsmodels_pingouin() -> R {
    let ctx = Context::new();
    let r = anova_repeated_measures(&ctx, &rm6())?;
    // statsmodels AnovaRM: F Value 21.9159, Num DF 2, Den DF 10; pingouin rm_anova(detailed=True):
    //   SS cond 52.111111111111, SS Error 11.888888888888, F 21.91588785046726, p_unc 0.00022121087382799
    // Fraction: SS_cond 469/9, SS_subj 562/9, SS_err 107/9, SS_total 1138/9, F 2345/107
    assert_eq!((r.n_subjects, r.n_conditions), (6, 3));
    assert_eq!(r.conditions.ss, q(469, 9));
    assert_eq!(r.subjects.ss, q(562, 9));
    assert_eq!(r.error.ss, q(107, 9));
    assert_eq!(r.total.ss, q(1138, 9));
    assert_eq!(r.f, q(2345, 107));
    assert_eq!(r.conditions.f, Some(q(2345, 107)));
    assert_eq!(
        (r.conditions.df, r.subjects.df, r.error.df, r.total.df),
        (2, 5, 10, 17)
    );
    close(r.p_value_f64()?, 0.000_221_210_873_827_993_6, 1e-14);
    // pingouin ng2 = SS_cond / SS_total = 0.41212653778558872 (the row's η²)
    assert_eq!(r.conditions.eta_squared, Some(q(469, 1138)));
    close(qf(&q(469, 1138)), 0.412_126_537_785_588_7, 1e-15);
    assert_eq!(r.conditions.partial_eta_squared, Some(q(469, 576)));
    assert_eq!(r.condition_means, vec![q(15, 2), q(19, 2), q(35, 3)]);
    assert_eq!(r.grand_mean, q(86, 9));
    assert_eq!(r.subject_means[0], q(31, 3));
    // pingouin epsilon(wide, 'gg') = 0.7446988422011275, ('hf') = 0.9879196620470071
    // Fraction: ε̂ = 11449/15374, ε̃ = 13330/13493
    assert_eq!(r.epsilon_gg, q(11_449, 15_374));
    assert_eq!(r.epsilon_hf, Some(q(13_330, 13_493)));
    close(qf(&r.epsilon_gg), 0.744_698_842_201_127_5, 1e-13);
    // pingouin sphericity: W 0.6571752991527652, chi2 1.6792179140115608, dof 2, pval 0.4318793738165263
    // Fraction: W = 7524/11449, ρ = 4/5 (d = 2, n = 6)
    let m = r.mauchly.as_ref().unwrap();
    assert_eq!(m.w, q(7524, 11_449));
    assert_eq!(m.df, 2);
    close(m.chi_squared_f64()?, 1.679_217_914_011_560_8, 1e-12);
    close(m.p_value_f64()?, 0.431_879_373_816_526_3, 1e-12);
    Ok(())
}

#[test]
fn rm6_sphericity_corrected_p_values() -> R {
    let ctx = Context::new();
    let r = anova_repeated_measures(&ctx, &rm6())?;
    // scipy f.sf(2345/107, 2ε, 10ε): GG 0.0011174925749680333 (pingouin p_GG_corr 0.00111749257496798),
    //   HF 0.000238713626315027
    close(r.p_value_gg_f64()?, 0.001_117_492_574_968_033_3, 1e-13);
    close(
        r.p_value_hf_f64()?.unwrap(),
        0.000_238_713_626_315_027,
        1e-13,
    );
    // Corrections can only make the test more conservative than the sphericity-assuming p.
    assert!(r.p_value_gg_f64()? > r.p_value_f64()?);
    assert!(r.p_value_hf_f64()?.unwrap() > r.p_value_f64()?);
    assert!(r.p_value_gg_f64()? > r.p_value_hf_f64()?.unwrap());
    assert_eq!(r.rows().len(), 4);
    Ok(())
}

#[test]
fn rm_ss_conditions_equals_the_one_way_between_groups_ss() -> R {
    let ctx = Context::new();
    let data = rm6();
    let r = anova_repeated_measures(&ctx, &data)?;
    // The conditions as independent groups: same SS_between (n Σ (ȳⱼ − ȳ)²), different F because
    // the one-way error keeps the subject variation.  scipy f_oneway: F 5.257847533632289
    let groups: Vec<Vec<Q>> = (0..3)
        .map(|j| data.iter().map(|s| s[j].clone()).collect())
        .collect();
    let one = anova_one_way(&ctx, &groups)?;
    assert_eq!(one.ss_between, r.conditions.ss);
    assert_eq!(one.ss_within, &r.subjects.ss + &r.error.ss);
    assert!(one.f < r.f);
    close(qf(&one.f), 5.257_847_533_632_289, 1e-13);
    Ok(())
}

#[test]
fn rm_two_conditions_has_unit_epsilon_and_tiny_designs_error() -> R {
    let ctx = Context::new();
    // k = 2, n = 3: ε̂ = 1 exactly, ε̃ = (3·1·1 − 2)/(1·(2 − 1)) = 1, no Mauchly test.
    // pingouin epsilon(wide, 'gg') = 1.0, ('hf') = 1.0; rm_anova: SS 8.166667 / 2.333333, F 7.0
    let y = vec![from_i64(&[3, 5]), from_i64(&[4, 8]), from_i64(&[6, 7])];
    let r = anova_repeated_measures(&ctx, &y)?;
    assert_eq!(r.epsilon_gg, Q::one());
    assert_eq!(r.epsilon_hf, Some(Q::one()));
    assert!(r.mauchly.is_none());
    assert_eq!(r.p_value_gg, r.p_value);
    assert_eq!(r.p_value_hf, Some(r.p_value.clone()));
    // Fraction: SS_cond = 3·((13/3 − 11/2)² + (20/3 − 11/2)²) = 49/6, SS_subj 7, SS_err 7/3
    assert_eq!((r.conditions.df, r.error.df), (1, 2));
    assert_eq!(r.conditions.ss, q(49, 6));
    assert_eq!(r.subjects.ss, qi(7));
    assert_eq!(r.error.ss, q(7, 3));
    // The F of k = 2 is the squared paired t: d = [2, 4, 1], d̄ = 7/3, s_d² = 7/3 → t² = (7/3)²/((7/3)/3) = 7
    assert_eq!(r.f, qi(7));
    // n = 1 subject, k = 1 condition, ragged rows.
    is_invalid(anova_repeated_measures(&ctx, &[from_i64(&[1, 2, 3])]));
    is_invalid(anova_repeated_measures(
        &ctx,
        &[from_i64(&[1]), from_i64(&[2])],
    ));
    is_invalid(anova_repeated_measures(
        &ctx,
        &[from_i64(&[1, 2]), from_i64(&[2])],
    ));
    Ok(())
}

#[test]
fn studentized_range_cdf_stays_accurate_for_large_df() -> R {
    // As df → ∞ the studentized range tends to the range of k standard normals:
    //   k ∫ φ(z) (Φ(z + q) − Φ(z))^{k−1} dz  (scipy uses this asymptotic form for df ≥ 100000)
    // scipy studentized_range.cdf: (3.0, 3, 1e4) 0.9144062561780589; (3.0, 3, 1e6) 0.9144574283450421
    //   (asymptotic; quad of the normal-range integral gives 0.9144574283450422); (3.5, 4, 1e8) 0.9361236661430967;
    //   (2.5, 2, 1e5) 0.9229001282564583 = erf(1.25)
    close(
        studentized_range_cdf(3.0, 3, 1e4)?,
        0.914_406_256_178_058_9,
        1e-9,
    );
    // Finite-df values approach the limit from below with an O(1/df) gap.
    let limit_3 = 0.914_457_428_345_042_1;
    let at_1e5 = studentized_range_cdf(3.0, 3, 1e5)?;
    let at_1e6 = studentized_range_cdf(3.0, 3, 1e6)?;
    assert!(at_1e5 < at_1e6 && at_1e6 < limit_3, "{at_1e5} {at_1e6}");
    close(at_1e6, limit_3, 1e-6);
    for df in [1e8, 1e10, 1e12, 1e15, 1e300] {
        close(studentized_range_cdf(3.0, 3, df)?, limit_3, 1e-8);
    }
    close(
        studentized_range_cdf(3.5, 4, 1e8)?,
        0.936_123_666_143_096_7,
        1e-8,
    );
    close(
        studentized_range_cdf(2.5, 2, 1e12)?,
        0.922_900_128_256_458_3,
        1e-8,
    );
    Ok(())
}

#[test]
fn studentized_range_extremes_and_exact_quantiles() -> R {
    // scipy: cdf(1e-3, 3, 10) = 2.7566440176364996e-07; cdf(30, 3, 10) = 0.9999999966875557;
    //   cdf(4.0, 6, 3) = 0.7317828681569625
    close(
        studentized_range_cdf(1e-3, 3, 10.0)?,
        2.756_644_017_636_499_6e-7,
        1e-12,
    );
    close(
        studentized_range_sf(30.0, 3, 10.0)?,
        1.0 - 0.999_999_996_687_555_7,
        1e-10,
    );
    close(
        studentized_range_cdf(4.0, 6, 3.0)?,
        0.731_782_868_156_962_5,
        1e-9,
    );
    // Q_{2,1} = √2 |T₁| has median √2: scipy ppf(0.5, 2, 1) = 1.4142135623730945
    close(
        studentized_range_quantile(0.5, 2, 1.0)?,
        std::f64::consts::SQRT_2,
        1e-7,
    );
    // scipy ppf(0.99, 6, 3) = 14.240704798363693
    close(
        studentized_range_quantile(0.99, 6, 3.0)?,
        14.240_704_798_363_693,
        1e-5,
    );
    assert_eq!(studentized_range_cdf(0.0, 4, 7.0)?, 0.0);
    is_invalid(studentized_range_cdf(1.0, 1, 7.0));
    is_invalid(studentized_range_cdf(1.0, 3, 0.999));
    is_invalid(studentized_range_quantile(0.0, 3, 7.0));
    Ok(())
}

#[test]
fn tukey_hsd_four_unequal_groups_matches_scipy() -> R {
    let ctx = Context::new();
    let g = g4();
    // Fraction: means [13, 71/4, 65/6, 41/2], MSE 83/36, df 15;
    //   pair (0,1): diff -19/4, var 83/160, stat² 3610/83; (2,3): diff -29/3, var 415/864, stat² 80736/415
    // scipy tukey_hsd(*g): pvalue[0,1] 1.5555801092128618e-03, [0,2] 1.2914605081957486e-01,
    //   [0,3] 1.2706841465481844e-05, [1,2] 2.0932536113682509e-05, [1,3] 9.0147953650792978e-02,
    //   [2,3] 3.2906550573308380e-07; confidence_interval(0.95).low[0,1] -7.685691611257397,
    //   high[0,1] -1.8143083887426035; ppf(0.95, 4, 15) = 4.075973736606698
    let pairs = tukey_hsd(&ctx, &g, 0.95)?;
    assert_eq!(pairs.len(), 6);
    assert_eq!((pairs[0].i, pairs[0].j), (0, 1));
    assert_eq!(pairs[0].diff, q(-19, 4));
    assert_eq!(pairs[0].se, ctx.from_ratio(q(83, 160)).sqrt());
    assert_eq!(
        (&pairs[0].statistic * &pairs[0].statistic).simplify(),
        ctx.from_ratio(q(3610, 83))
    );
    assert_eq!(pairs[5].diff, q(-29, 3));
    assert_eq!(
        (&pairs[5].statistic * &pairs[5].statistic).simplify(),
        ctx.from_ratio(q(80_736, 415))
    );
    let want = [
        1.555_580_109_212_861_8e-3,
        1.291_460_508_195_748_6e-1,
        1.270_684_146_548_184_4e-5,
        2.093_253_611_368_251e-5,
        9.014_795_365_079_298e-2,
        3.290_655_057_330_838e-7,
    ];
    for (p, w) in pairs.iter().zip(want) {
        close(p.p_adj, w, 1e-7);
    }
    close(pairs[0].ci.lower, -7.685_691_611_257_397, 1e-6);
    close(pairs[0].ci.upper, -1.814_308_388_742_603_5, 1e-6);
    // confidence_interval(0.90).low[2,3] -12.120003158963215, high -7.213330174370117
    let pairs90 = tukey_hsd(&ctx, &g, 0.90)?;
    close(pairs90[5].ci.lower, -12.120_003_158_963_215, 1e-6);
    close(pairs90[5].ci.upper, -7.213_330_174_370_117, 1e-6);
    // The 90 % interval is inside the 95 % one.
    assert!(pairs90[5].ci.lower > pairs[5].ci.lower && pairs90[5].ci.upper < pairs[5].ci.upper);
    Ok(())
}

#[test]
fn tukey_hsd_with_two_groups_is_the_pooled_t_test() -> R {
    let ctx = Context::new();
    // a = [5, 7, 6, 9, 8], b = [10, 12, 9, 13]: scipy tukey_hsd pvalue 0.009627924725379766,
    //   ttest_ind(equal_var=True) pvalue 0.009627924725379822, t -3.5276684147527875, statistic -4.0 = √2·t… no:
    //   the Tukey statistic |diff|/se equals √2 |t| (4.988876515698589); scipy's `statistic` is the raw diff.
    let a = from_i64(&[5, 7, 6, 9, 8]);
    let b = from_i64(&[10, 12, 9, 13]);
    let pairs = tukey_hsd(&ctx, &[a.clone(), b.clone()], 0.95)?;
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].diff, qi(-4));
    let t = t_test_two_sample(&ctx, &a, &b, true, Alternative::TwoSided)?;
    close(pairs[0].p_adj, t.p_value_f64()?, 1e-9);
    close(pairs[0].p_adj, 0.009_627_924_725_379_766, 1e-9);
    close(
        pairs[0].statistic.eval_f64()?,
        std::f64::consts::SQRT_2 * t.statistic_f64()?.abs(),
        1e-12,
    );
    close(pairs[0].statistic.eval_f64()?, 4.988_876_515_698_589, 1e-12);
    Ok(())
}

#[test]
fn tukey_hsd_p_values_are_monotone_in_the_difference_for_equal_sizes() -> R {
    let ctx = Context::new();
    let g = vec![
        from_i64(&[1, 2, 3, 4]),
        from_i64(&[3, 4, 5, 6]),
        from_i64(&[6, 7, 8, 9]),
        from_i64(&[10, 11, 12, 13]),
    ];
    let mut pairs = tukey_hsd(&ctx, &g, 0.95)?;
    // Equal sizes → one common se; sort by |diff| and require strictly decreasing p.
    pairs.sort_by_key(|a| a.diff.abs());
    let diffs: Vec<Q> = pairs.iter().map(|p| p.diff.abs()).collect();
    assert_eq!(diffs, from_i64(&[2, 3, 4, 5, 7, 9]));
    for w in pairs.windows(2) {
        assert!(w[0].p_adj > w[1].p_adj, "{} vs {}", w[0].p_adj, w[1].p_adj);
        assert_eq!(w[0].se, w[1].se);
    }
    // Every interval has the same half-width.
    let widths: Vec<f64> = pairs.iter().map(|p| p.ci.upper - p.ci.lower).collect();
    for w in widths.windows(2) {
        close(w[0], w[1], 1e-12);
    }
    is_invalid(tukey_hsd(&ctx, &g, 0.0));
    is_invalid(tukey_hsd(&ctx, &g[..1], 0.95));
    Ok(())
}

#[test]
fn pairwise_welch_tests_holm_is_never_above_bonferroni() -> R {
    let ctx = Context::new();
    let g = g4();
    // scipy ttest_ind(equal_var=False) per pair: p = [0.004628941024699772, 0.046432940370345566,
    //   0.00010574031632595973, 0.0006342044662385739, 0.04522464605965956, 9.41792547693385e-06],
    //   df[0,1] 6.302353651176825, t[0,1] -4.284382359616124
    // statsmodels multipletests(p, 0.05, 'holm')[1] = [0.013886823074099316, 0.09044929211931912,
    //   0.0005287015816297987, 0.0025368178649542955, 0.09044929211931912, 5.65075528616031e-05]
    //   'bonferroni' = [0.027773646148198633, 0.2785976422220734, 0.0006344418979557584,
    //   0.003805226797431443, 0.2713478763579574, 5.65075528616031e-05]; reject = [T, F, T, T, F, T]
    let holm = pairwise_t_tests(&ctx, &g, Adjustment::Holm, 0.05)?;
    let bonf = pairwise_t_tests(&ctx, &g, Adjustment::Bonferroni, 0.05)?;
    assert_eq!(holm.len(), 6);
    close(
        holm[0].test.p_value_f64()?,
        0.004_628_941_024_699_772,
        1e-14,
    );
    close(holm[0].test.statistic_f64()?, -4.284_382_359_616_124, 1e-12);
    close(
        holm[0].test.df.as_ref().unwrap().eval_f64()?,
        6.302_353_651_176_825,
        1e-12,
    );
    assert_eq!(holm[0].diff, q(-19, 4));
    let holm_want = [
        0.013_886_823_074_099_316,
        0.090_449_292_119_319_12,
        0.000_528_701_581_629_798_7,
        0.002_536_817_864_954_295_5,
        0.090_449_292_119_319_12,
        5.650_755_286_160_31e-5,
    ];
    let bonf_want = [
        0.027_773_646_148_198_633,
        0.278_597_642_222_073_4,
        0.000_634_441_897_955_758_4,
        0.003_805_226_797_431_443,
        0.271_347_876_357_957_4,
        5.650_755_286_160_31e-5,
    ];
    for i in 0..6 {
        close(holm[i].p_adj, holm_want[i], 1e-12);
        close(bonf[i].p_adj, bonf_want[i], 1e-12);
        assert!(holm[i].p_adj <= bonf[i].p_adj + 1e-15);
        assert!(holm[i].p_adj >= holm[i].test.p_value_f64()?);
        assert_eq!(holm[i].test, bonf[i].test);
    }
    let reject: Vec<bool> = holm.iter().map(|p| p.reject).collect();
    assert_eq!(reject, vec![true, false, true, true, false, true]);
    assert_eq!(bonf.iter().map(|p| p.reject).collect::<Vec<_>>(), reject);
    is_invalid(pairwise_t_tests(&ctx, &g, Adjustment::Holm, 1.0));
    is_invalid(pairwise_t_tests(
        &ctx,
        &[g[0].clone(), from_i64(&[5])],
        Adjustment::Holm,
        0.05,
    ));
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Survival
// ═══════════════════════════════════════════════════════════════════════════

/// S1: times [2, 3, 3, 5, 6, 6, 8, 9, 11, 12], events [1, 1, 0, 1, 1, 1, 0, 1, 0, 1]
/// (an event and a censoring at t = 3, two events at t = 6).
fn s1() -> Vec<Observation> {
    Observation::from_i64(
        &[2, 3, 3, 5, 6, 6, 8, 9, 11, 12],
        &[
            true, true, false, true, true, true, false, true, false, true,
        ],
    )
}

#[test]
fn km_s1_life_table_exact() -> R {
    let km = KaplanMeier::fit(&s1())?;
    assert_eq!(km.n(), 10);
    // statsmodels SurvfuncRight: surv_times [2, 3, 5, 6, 9, 12], n_risk [10, 9, 7, 6, 3, 1],
    //   n_events [1, 1, 1, 2, 1, 1], surv_prob [0.9, 0.8, 0.6857142857142857, 0.45714285714285724,
    //   0.30476190476190484, 0.0]
    // Fraction rows (time, at_risk, events, censored, S, Greenwood, Nelson–Aalen):
    //   (2, 10, 1, 0, 9/10, 9/1000, 1/10), (3, 9, 1, 1, 4/5, 2/125, 19/90), (5, 7, 1, 0, 24/35, 984/42875, 223/630),
    //   (6, 6, 2, 0, 16/35, 1184/42875, 433/630), (9, 3, 1, 0, 32/105, 32128/1157625, 643/630), (12, 1, 1, 0, 0, 0, 1273/630)
    assert_eq!(km.event_times(), from_i64(&[2, 3, 5, 6, 9, 12]));
    let t = km.table();
    let at_risk: Vec<usize> = t.iter().map(|r| r.at_risk).collect();
    let events: Vec<usize> = t.iter().map(|r| r.events).collect();
    let censored: Vec<usize> = t.iter().map(|r| r.censored).collect();
    assert_eq!(at_risk, vec![10, 9, 7, 6, 3, 1]);
    assert_eq!(events, vec![1, 1, 1, 2, 1, 1]);
    assert_eq!(censored, vec![0, 1, 0, 0, 0, 0]);
    let s: Vec<Q> = t.iter().map(|r| r.survival.clone()).collect();
    assert_eq!(
        s,
        qs(&[(9, 10), (4, 5), (24, 35), (16, 35), (32, 105), (0, 1)])
    );
    let v: Vec<Q> = t.iter().map(|r| r.variance.clone()).collect();
    assert_eq!(
        v,
        qs(&[
            (9, 1000),
            (2, 125),
            (984, 42_875),
            (1184, 42_875),
            (32_128, 1_157_625),
            (0, 1)
        ])
    );
    let h: Vec<Q> = t.iter().map(|r| r.cumulative_hazard.clone()).collect();
    assert_eq!(
        h,
        qs(&[
            (1, 10),
            (19, 90),
            (223, 630),
            (433, 630),
            (643, 630),
            (1273, 630)
        ])
    );
    // statsmodels surv_prob_se[3] = 0.16617809828570743 = √(1184/42875)
    close(qf(&v[3]).sqrt(), 0.166_178_098_285_707_43, 1e-14);
    Ok(())
}

#[test]
fn km_s1_product_limit_and_greenwood_recomputed_from_the_table() -> R {
    let km = KaplanMeier::fit(&s1())?;
    // Recompute S and Greenwood's sum from (at_risk, events) row by row.
    let mut s = Q::one();
    let mut g = Q::zero();
    let mut h = Q::zero();
    for r in km.table() {
        let (n, d) = (qi(r.at_risk as i64), qi(r.events as i64));
        s *= (&n - &d) / &n;
        if r.at_risk > r.events {
            g += &d / (&n * (&n - &d));
        }
        h += &d / &n;
        assert_eq!(r.survival, s);
        assert_eq!(r.variance, &s * &s * &g);
        assert_eq!(r.cumulative_hazard, h);
        assert_eq!(km.survival_at(&r.time), s);
        assert_eq!(km.variance_at(&r.time), r.variance);
        assert_eq!(km.cumulative_hazard_at(&r.time), h);
    }
    // Between event times the curve is flat; before the first event it is 1 with zero variance.
    assert_eq!(km.survival_at(&qi(1)), Q::one());
    assert_eq!(km.variance_at(&q(3, 2)), Q::zero());
    assert_eq!(km.cumulative_hazard_at(&qi(0)), Q::zero());
    assert_eq!(km.survival_at(&qi(4)), q(4, 5));
    assert_eq!(km.survival_at(&q(11, 2)), q(24, 35));
    assert_eq!(km.survival_at(&qi(8)), q(16, 35));
    assert_eq!(km.survival_at(&qi(11)), q(32, 105));
    assert_eq!(km.survival_at(&qi(100)), Q::zero());
    // The censoring at 3 leaves the risk set after the event at 3: n(5) = 7, not 8.
    assert_eq!(km.table()[2].at_risk, 7);
    Ok(())
}

#[test]
fn km_s1_quantiles_restricted_mean_and_intervals() -> R {
    let km = KaplanMeier::fit(&s1())?;
    // statsmodels quantile(p): 0.25 → 5, 0.5 → 6, 0.75 → 12, 0.9 → 12
    assert_eq!(km.quantile(&q(1, 4)), Some(qi(5)));
    assert_eq!(km.median(), Some(qi(6)));
    assert_eq!(km.quantile(&q(3, 4)), Some(qi(12)));
    assert_eq!(km.quantile(&q(9, 10)), Some(qi(12)));
    assert_eq!(km.quantile(&Q::one()), Some(qi(12)));
    // Fraction ∫₀^τ S: τ = 10 → 1441/210, τ = 12 → 523/70, τ = 3/2 → 3/2, τ = 6 → 363/70;
    //   beyond the last event the curve is 0, so τ = 20 gives the same 523/70.
    assert_eq!(km.restricted_mean(&qi(10)), q(1441, 210));
    assert_eq!(km.restricted_mean(&qi(12)), q(523, 70));
    assert_eq!(km.restricted_mean(&q(3, 2)), q(3, 2));
    assert_eq!(km.restricted_mean(&qi(6)), q(363, 70));
    assert_eq!(km.restricted_mean(&qi(20)), q(523, 70));
    assert_eq!(km.restricted_mean(&Q::zero()), Q::zero());
    // At t = 6: S = 16/35, Var = 1184/42875, z = 1.959963984540054:
    //   linear (0.13143976948351332, 0.7828459448022009); log-log (0.1429821627556124, 0.729779106015238)
    let lin = km.confidence_interval(&qi(6), 0.95, CiMethod::Linear)?;
    close(lin.lower, 0.131_439_769_483_513_32, 1e-9);
    close(lin.upper, 0.782_845_944_802_200_9, 1e-9);
    let ll = km.confidence_interval(&qi(6), 0.95, CiMethod::LogLog)?;
    close(ll.lower, 0.142_982_162_755_612_4, 1e-9);
    close(ll.upper, 0.729_779_106_015_238, 1e-9);
    // After the last event S = 0: the linear interval is clamped, the log-log one degenerate.
    let end = km.confidence_interval(&qi(12), 0.95, CiMethod::Linear)?;
    assert_eq!((end.lower, end.upper), (0.0, 0.0));
    let end = km.confidence_interval(&qi(12), 0.95, CiMethod::LogLog)?;
    assert_eq!((end.lower, end.upper), (0.0, 0.0));
    is_invalid(km.confidence_interval(&qi(6), 0.0, CiMethod::Linear));
    Ok(())
}

#[test]
fn km_edge_cases_all_censored_all_events_at_once_and_mixed_times() -> R {
    // All censored: no event, S ≡ 1, no quantile, restricted mean = τ.
    let cens = KaplanMeier::fit(&Observation::from_i64(&[1, 4, 6], &[false; 3]))?;
    assert!(cens.table().is_empty());
    assert_eq!(cens.survival_at(&qi(10)), Q::one());
    assert_eq!(cens.variance_at(&qi(10)), Q::zero());
    assert_eq!(cens.median(), None);
    assert_eq!(cens.quantile(&Q::zero()), None);
    assert_eq!(cens.restricted_mean(&qi(7)), qi(7));
    // Every subject fails at t = 4: Fraction row (4, 4, 4, 0, S 0, Var 0, H 1)
    let all = KaplanMeier::fit(&Observation::from_i64(&[4, 4, 4, 4], &[true; 4]))?;
    assert_eq!(all.table().len(), 1);
    let r = &all.table()[0];
    assert_eq!((r.at_risk, r.events, r.censored), (4, 4, 0));
    assert_eq!(
        (
            r.survival.clone(),
            r.variance.clone(),
            r.cumulative_hazard.clone()
        ),
        (Q::zero(), Q::zero(), Q::one())
    );
    assert_eq!(all.median(), Some(qi(4)));
    assert_eq!(all.restricted_mean(&qi(4)), qi(4));
    assert_eq!(all.restricted_mean(&qi(9)), qi(4));
    // Event and censoring at the same time (t = 2): the censored subject counts in the risk set.
    // Fraction: (1, 4, 1, 0, 3/4, 3/64, 1/4), (2, 3, 1, 1, 1/2, 1/16, 7/12)
    let mixed = KaplanMeier::fit(&Observation::from_i64(
        &[1, 2, 2, 3],
        &[true, true, false, false],
    ))?;
    let t = mixed.table();
    assert_eq!(t.len(), 2);
    assert_eq!((t[1].at_risk, t[1].events, t[1].censored), (3, 1, 1));
    assert_eq!(
        (t[1].survival.clone(), t[1].variance.clone()),
        (q(1, 2), q(1, 16))
    );
    assert_eq!(t[1].cumulative_hazard, q(7, 12));
    assert_eq!(
        (t[0].survival.clone(), t[0].variance.clone()),
        (q(3, 4), q(3, 64))
    );
    // The curve never drops below 1/2: the 0.75-quantile is undefined, the median is 2.
    assert_eq!(mixed.quantile(&q(3, 4)), None);
    assert_eq!(mixed.median(), Some(qi(2)));
    // Invalid input.
    is_invalid(KaplanMeier::fit(&[]));
    is_invalid(KaplanMeier::fit(&Observation::from_q(&[qi(-1)], &[true])));
    // `from_i64` / `from_q` zip to the shorter slice.
    assert_eq!(Observation::from_i64(&[1, 2, 3], &[true]).len(), 1);
    Ok(())
}

#[test]
fn km_quantile_uses_a_non_strict_inequality_unlike_statsmodels() -> R {
    // Four events at 1, 2, 3, 4: S(2) = 1/2 exactly.  The crate's documented rule (smallest t with
    // S(t) ≤ 1 − p) gives 2; statsmodels' SurvfuncRight.quantile(0.5) uses a strict `<` and gives 3.
    let km = KaplanMeier::fit(&Observation::from_i64(&[1, 2, 3, 4], &[true; 4]))?;
    assert_eq!(km.survival_at(&qi(2)), q(1, 2));
    assert_eq!(km.median(), Some(qi(2)));
    assert_eq!(km.quantile(&q(1, 4)), Some(qi(1)));
    assert_eq!(km.quantile(&Q::one()), Some(qi(4)));
    Ok(())
}

#[test]
fn log_rank_two_groups_matches_survdiff_exactly() -> R {
    let ctx = Context::new();
    // Group 0 = S1; group 1: times [4, 4, 7, 10, 10, 13, 14, 16, 19, 20], events [1,0,1,1,1,0,1,1,0,1]
    // statsmodels survdiff: (4.006685469693783, 0.04532016289995067); Fraction (O−E)²/V = 4995443074453/1246776946241
    let mut obs = s1();
    obs.extend(Observation::from_i64(
        &[4, 4, 7, 10, 10, 13, 14, 16, 19, 20],
        &[
            true, false, true, true, true, false, true, true, false, true,
        ],
    ));
    let groups: Vec<usize> = (0..20).map(|i| usize::from(i >= 10)).collect();
    let r = log_rank_test(&ctx, &obs, &groups)?;
    assert_eq!(
        r.statistic_exact(),
        Some(q(4_995_443_074_453, 1_246_776_946_241))
    );
    close(r.statistic_f64()?, 4.006_685_469_693_783, 1e-12);
    close(r.p_value_f64()?, 0.045_320_162_899_950_67, 1e-12);
    assert_eq!(r.df, Some(ctx.int(1)));
    // Relabelling the groups does not change the statistic.
    let swapped: Vec<usize> = groups.iter().map(|g| 1 - g).collect();
    assert_eq!(log_rank_test(&ctx, &obs, &swapped)?.statistic, r.statistic);
    Ok(())
}

#[test]
fn log_rank_identical_groups_three_groups_and_validation() -> R {
    let ctx = Context::new();
    // Two copies of S1 as two groups: O = E everywhere → statistic 0, p = 1 (survdiff (0.0, 1.0)).
    let mut obs = s1();
    obs.extend(s1());
    let groups: Vec<usize> = (0..20).map(|i| usize::from(i >= 10)).collect();
    let r = log_rank_test(&ctx, &obs, &groups)?;
    assert_eq!(r.statistic_exact(), Some(Q::zero()));
    assert_eq!(r.p_value_f64()?, 1.0);
    // Three groups: S1, the second group above, and [1, 5, 6, 9, 12, 15, 18] / [1,1,1,0,1,1,1]
    // statsmodels survdiff: (4.158554655467686, 0.12502052842873013), df 2
    let mut obs3 = s1();
    obs3.extend(Observation::from_i64(
        &[4, 4, 7, 10, 10, 13, 14, 16, 19, 20],
        &[
            true, false, true, true, true, false, true, true, false, true,
        ],
    ));
    obs3.extend(Observation::from_i64(
        &[1, 5, 6, 9, 12, 15, 18],
        &[true, true, true, false, true, true, true],
    ));
    let mut g3 = vec![0usize; 10];
    g3.extend(vec![1usize; 10]);
    g3.extend(vec![2usize; 7]);
    let r3 = log_rank_test(&ctx, &obs3, &g3)?;
    close(r3.statistic_f64()?, 4.158_554_655_467_686, 1e-12);
    close(r3.p_value_f64()?, 0.125_020_528_428_730_13, 1e-12);
    assert_eq!(r3.df, Some(ctx.int(2)));
    // One group, a label gap, a length mismatch, no events.
    is_invalid(log_rank_test(&ctx, &s1(), &[0; 10]));
    let mut gap = vec![0usize; 10];
    gap.extend(vec![2usize; 10]);
    is_invalid(log_rank_test(&ctx, &obs, &gap));
    is_invalid(log_rank_test(&ctx, &obs, &groups[..19]));
    let none = Observation::from_i64(&[1, 2, 3, 4], &[false; 4]);
    is_invalid(log_rank_test(&ctx, &none, &[0, 0, 1, 1]));
    Ok(())
}

#[test]
fn exponential_rate_and_mean_event_time_exact() -> R {
    let obs = s1();
    // Fraction: λ̂ = 7 events / 65 total time; mean of the event times (2, 3, 5, 6, 6, 9, 12) = 43/7
    assert_eq!(exponential_rate(&obs)?, q(7, 65));
    assert_eq!(mean_event_time(&obs)?, q(43, 7));
    // All censored: rate 0 but no mean event time; zero total time is rejected.
    let cens = Observation::from_i64(&[1, 4, 6], &[false; 3]);
    assert_eq!(exponential_rate(&cens)?, Q::zero());
    is_invalid(mean_event_time(&cens));
    is_invalid(exponential_rate(&Observation::from_i64(
        &[0, 0],
        &[true, false],
    )));
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Sequential (SPRT)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn wald_boundaries_new_rates_and_sprt_reports_the_same_pair() -> R {
    // α = 0.1, β = 0.2: A = ln(0.2/0.9) = -1.504077396776274, B = ln(0.8/0.1) = 2.0794415416798357 (numpy)
    let b = wald_boundaries(0.1, 0.2)?;
    close(b.lower, -1.504_077_396_776_274, 1e-15);
    close(b.upper, 2.079_441_541_679_835_7, 1e-15);
    let test = Sprt::bernoulli(q(1, 2), q(3, 4), 0.1, 0.2)?;
    assert_eq!(test.boundaries(), b);
    assert_eq!((test.alpha(), test.beta()), (0.1, 0.2));
    assert!(test.boundaries().contains(&0.0));
    assert!(!test.boundaries().contains(&b.upper));
    // Symmetric rates give symmetric boundaries: α = β = 0.05 → ±2.9444389791664403
    let s = wald_boundaries(0.05, 0.05)?;
    close(s.lower, -2.944_438_979_166_440_3, 1e-15);
    close(s.upper, 2.944_438_979_166_440_3, 1e-15);
    assert_eq!(
        Sprt::normal_mean(qi(0), qi(1), qi(2), 0.05, 0.05)?.boundaries(),
        s
    );
    Ok(())
}

#[test]
fn sprt_operating_characteristic_and_expected_n_new_design() -> R {
    // p₀ = 1/2, p₁ = 3/4, α = 0.1, β = 0.2 (scipy brentq on Wald's identity, then Wald's formulas):
    //   p = 0.5:  L 0.9 (= 1 − α, h = 1),  E[N] 7.9652200303354
    //   p = 0.75: L 0.2 (= β, h = −1),     E[N] 10.417525758882162
    //   p = 0.6:  L 0.6808653970598991,    E[N] 10.60784422391488   (h 0.23938717951024752)
    //   p = 0.3:  L 0.9963071998833086,    E[N] 4.100643222963972
    //   p = 0.9:  L 0.01395294369814303,   E[N] 6.865406871177556
    //   drift-zero p = ln 2 / ln 3 = 0.6309297535714575: L = B/(B−A) = 0.5802792108518123, E[N] = −AB/E[Z²] = 11.128533874054364
    let (p0, p1) = (q(1, 2), q(3, 4));
    let oc = |p: f64| operating_characteristic_bernoulli(p, &p0, &p1, 0.1, 0.2);
    let en = |p: f64| expected_sample_size_bernoulli(p, &p0, &p1, 0.1, 0.2);
    close(oc(0.5)?, 0.9, 1e-12);
    close(en(0.5)?, 7.965_220_030_335_4, 1e-9);
    close(oc(0.75)?, 0.2, 1e-12);
    close(en(0.75)?, 10.417_525_758_882_162, 1e-9);
    close(oc(0.6)?, 0.680_865_397_059_899_1, 1e-9);
    close(en(0.6)?, 10.607_844_223_914_88, 1e-8);
    close(oc(0.3)?, 0.996_307_199_883_308_6, 1e-9);
    close(en(0.3)?, 4.100_643_222_963_972, 1e-8);
    close(oc(0.9)?, 0.013_952_943_698_143_03, 1e-9);
    close(en(0.9)?, 6.865_406_871_177_556, 1e-8);
    let pz = 2.0_f64.ln() / 3.0_f64.ln();
    close(oc(pz)?, 0.580_279_210_851_812_3, 1e-9);
    close(en(pz)?, 11.128_533_874_054_364, 1e-8);
    // L is decreasing in p between the hypotheses.
    assert!(oc(0.5)? > oc(0.6)? && oc(0.6)? > oc(0.7)? && oc(0.7)? > oc(0.75)?);
    Ok(())
}

#[test]
fn sprt_reversed_hypotheses_operating_characteristic() -> R {
    // p₀ = 3/5 > p₁ = 2/5, α = β = 0.05: L(0.6) = 0.95, L(0.4) = 0.05, E[N] = 32.67846022083444 at both;
    //   at p = 0.5 the drift is zero: L = 0.5, E[N] = 52.73490184714365
    let (p0, p1) = (q(3, 5), q(2, 5));
    close(
        operating_characteristic_bernoulli(0.6, &p0, &p1, 0.05, 0.05)?,
        0.95,
        1e-12,
    );
    close(
        operating_characteristic_bernoulli(0.4, &p0, &p1, 0.05, 0.05)?,
        0.05,
        1e-12,
    );
    close(
        expected_sample_size_bernoulli(0.6, &p0, &p1, 0.05, 0.05)?,
        32.678_460_220_834_44,
        1e-9,
    );
    close(
        expected_sample_size_bernoulli(0.4, &p0, &p1, 0.05, 0.05)?,
        32.678_460_220_834_44,
        1e-9,
    );
    close(
        operating_characteristic_bernoulli(0.5, &p0, &p1, 0.05, 0.05)?,
        0.5,
        1e-12,
    );
    close(
        expected_sample_size_bernoulli(0.5, &p0, &p1, 0.05, 0.05)?,
        52.734_901_847_143_65,
        1e-9,
    );
    // A run of failures accepts H₁ (the lower success rate).
    let mut t = Sprt::bernoulli(p0, p1, 0.05, 0.05)?;
    let mut d = Decision::Continue;
    while d == Decision::Continue {
        d = t.update(false);
    }
    assert_eq!(d, Decision::AcceptH1);
    // ⌈B / ln(3/2)⌉ = ⌈2.9444 / 0.4055⌉ = 8 failures
    assert_eq!(t.observations(), 8);
    assert_eq!((t.successes(), t.failures()), (0, 8));
    Ok(())
}

#[test]
fn sprt_operating_characteristic_at_p_zero_and_one_are_the_limits() -> R {
    // p = 1: every trial succeeds, Λₙ = n·ln(p₁/p₀) → +∞ when p₁ > p₀: L = 0 and
    //   E[N] = B / ln(p₁/p₀) = 2.0794415416798357 / ln 1.5 = 5.128533874054364.
    // p = 0: Λₙ = n·ln((1−p₁)/(1−p₀)) → −∞: L = 1 and E[N] = A / ln((1−p₁)/(1−p₀)) = 1.504077396776274 / ln 2
    //   = 2.169925001442312.
    let (p0, p1) = (q(1, 2), q(3, 4));
    let l1 = operating_characteristic_bernoulli(1.0, &p0, &p1, 0.1, 0.2)?;
    let l0 = operating_characteristic_bernoulli(0.0, &p0, &p1, 0.1, 0.2)?;
    assert_eq!(l1, 0.0, "L(1)");
    assert_eq!(l0, 1.0, "L(0)");
    let n1 = expected_sample_size_bernoulli(1.0, &p0, &p1, 0.1, 0.2)?;
    let n0 = expected_sample_size_bernoulli(0.0, &p0, &p1, 0.1, 0.2)?;
    close(n1, 2.079_441_541_679_835_7 / 1.5_f64.ln(), 1e-12);
    close(n0, 1.504_077_396_776_274 / 2.0_f64.ln(), 1e-12);
    // Reversed hypotheses (p₁ < p₀): all successes accept H₀.
    let (r0, r1) = (q(3, 5), q(2, 5));
    assert_eq!(
        operating_characteristic_bernoulli(1.0, &r0, &r1, 0.05, 0.05)?,
        1.0
    );
    assert_eq!(
        operating_characteristic_bernoulli(0.0, &r0, &r1, 0.05, 0.05)?,
        0.0
    );
    close(
        expected_sample_size_bernoulli(1.0, &r0, &r1, 0.05, 0.05)?,
        2.944_438_979_166_440_3 / 1.5_f64.ln(),
        1e-12,
    );
    // The limits are continuous: p = 1 − 1e-9 is within 1e-6 of p = 1.
    close(
        operating_characteristic_bernoulli(1.0 - 1e-9, &p0, &p1, 0.1, 0.2)?,
        0.0,
        1e-6,
    );
    close(
        expected_sample_size_bernoulli(1.0 - 1e-9, &p0, &p1, 0.1, 0.2)?,
        n1,
        1e-6,
    );
    Ok(())
}

#[test]
fn sprt_rejects_invalid_designs_and_observations() -> R {
    // α + β ≥ 1, α or β outside (0, 1), p₀ = p₁, p outside (0, 1), σ ≤ 0.
    is_invalid(wald_boundaries(0.4, 0.6));
    is_invalid(wald_boundaries(0.3, 0.7));
    is_invalid(Sprt::bernoulli(q(1, 2), q(3, 4), 0.5, 0.5));
    is_invalid(Sprt::bernoulli(q(1, 2), q(3, 4), 1.0, 0.1));
    is_invalid(Sprt::bernoulli(q(1, 2), q(3, 4), 0.1, 0.0));
    is_invalid(Sprt::bernoulli(q(1, 2), q(1, 2), 0.1, 0.2));
    is_invalid(Sprt::bernoulli(q(1, 2), Q::one(), 0.1, 0.2));
    is_invalid(Sprt::bernoulli(Q::zero(), q(1, 2), 0.1, 0.2));
    is_invalid(Sprt::normal_mean(qi(0), qi(1), qi(0), 0.1, 0.2));
    is_invalid(Sprt::normal_mean(qi(1), qi(1), qi(1), 0.1, 0.2));
    is_invalid(operating_characteristic_bernoulli(
        0.5,
        &q(1, 2),
        &q(1, 2),
        0.1,
        0.2,
    ));
    is_invalid(operating_characteristic_bernoulli(
        1.5,
        &q(1, 2),
        &q(3, 4),
        0.1,
        0.2,
    ));
    is_invalid(operating_characteristic_bernoulli(
        -0.1,
        &q(1, 2),
        &q(3, 4),
        0.1,
        0.2,
    ));
    is_invalid(expected_sample_size_bernoulli(
        0.5,
        &q(1, 2),
        &q(3, 4),
        0.6,
        0.4,
    ));
    // A Bernoulli test only accepts 0 and 1 as observed values.
    let mut t = Sprt::bernoulli(q(1, 2), q(3, 4), 0.1, 0.2)?;
    is_invalid(t.observe(q(1, 2)));
    assert_eq!(t.observations(), 0);
    assert_eq!(t.observe(Q::one())?, Decision::Continue);
    assert_eq!(t.observe(Q::zero())?, Decision::Continue);
    assert_eq!(
        (t.observations(), t.successes(), t.failures(), t.sum()),
        (2, 1, 1, &Q::one())
    );
    Ok(())
}

#[test]
fn sprt_decision_is_recomputed_from_the_counts_after_a_boundary_is_reached() -> R {
    // The test is a plain value: observations after a decision keep being counted and the decision
    // is re-derived from the running log-likelihood ratio, so it can fall back to `Continue`.
    let ctx = Context::new();
    let mut t = Sprt::bernoulli(q(1, 2), q(3, 4), 0.1, 0.2)?;
    let mut d = Decision::Continue;
    while d == Decision::Continue {
        d = t.update(true);
    }
    assert_eq!(d, Decision::AcceptH1);
    // ⌈B / ln(3/2)⌉ = ⌈2.0794 / 0.4055⌉ = 6 successes; Λ = 6 ln(3/2) = 2.4327906486489868
    assert_eq!(t.observations(), 6);
    close(t.log_likelihood_ratio_f64(), 6.0 * 1.5_f64.ln(), 1e-14);
    assert_eq!(
        t.log_likelihood_ratio(&ctx)
            .equals(&(ctx.int(6) * ctx.rational(3, 2).ln())),
        Some(true)
    );
    // One failure subtracts ln 2 = 0.693: Λ = 1.7396 < B → back to Continue.
    assert_eq!(t.update(false), Decision::Continue);
    assert_eq!(t.decision(), Decision::Continue);
    assert_eq!(t.observations(), 7);
    // Reset forgets the data but not the design.
    let before = t.boundaries();
    t.reset();
    assert_eq!((t.observations(), t.successes()), (0, 0));
    assert_eq!(t.decision(), Decision::Continue);
    assert_eq!(t.boundaries(), before);
    assert_eq!(t.log_likelihood_ratio_f64(), 0.0);
    Ok(())
}

#[test]
fn sprt_normal_mean_llr_is_exact_and_decides() -> R {
    let ctx = Context::new();
    // H₀: μ = 0, H₁: μ = 1, σ = 2, α = β = 0.05.  Λ = (1/4)·Σx − n/8.
    let mut t = Sprt::normal_mean(qi(0), qi(1), qi(2), 0.05, 0.05)?;
    for x in [q(3, 2), qi(-1), q(5, 2), qi(2)] {
        t.observe(x)?;
    }
    // Σx = 5, n = 4: Λ = 5/4 − 1/2 = 3/4; positive observations counted as "successes" = 3
    assert_eq!(t.sum(), &qi(5));
    assert_eq!(t.log_likelihood_ratio(&ctx), ctx.rational(3, 4));
    assert_eq!(t.log_likelihood_ratio_f64(), 0.75);
    assert_eq!((t.observations(), t.successes()), (4, 3));
    assert_eq!(t.decision(), Decision::Continue);
    // Large observations push Λ past B = 2.9444: Σx = 5 + 12 = 17, n = 5: Λ = 17/4 − 5/8 = 29/8 = 3.625
    assert_eq!(t.observe(qi(12))?, Decision::AcceptH1);
    assert_eq!(t.log_likelihood_ratio(&ctx), ctx.rational(29, 8));
    // `update` on a normal-mean test records 1 or 0.
    let mut u = Sprt::normal_mean(qi(0), qi(1), qi(2), 0.05, 0.05)?;
    u.update(true);
    u.update(false);
    assert_eq!(u.sum(), &Q::one());
    assert_eq!(u.log_likelihood_ratio(&ctx), ctx.rational(0, 1));
    Ok(())
}

#[test]
fn sprt_expected_sample_size_against_a_simulation() -> R {
    // Wald's E[N] ignores the overshoot, so the simulated mean run length sits a little above it;
    // the simulated acceptance rate of H₀ at p₀ lies between 1 − α/(1−β) and 1 (Wald's bounds).
    let (p0, p1) = (q(1, 2), q(3, 4));
    let mut rng = Rng::new(0x5EED_2026_0921);
    for (p, l_wald, n_wald) in [
        (0.5, 0.9, 7.965_220_030_335_4),
        (0.75, 0.2, 10.417_525_758_882_162),
        (0.6, 0.680_865_397_059_899_1, 10.607_844_223_914_88),
    ] {
        let mut test = Sprt::bernoulli(p0.clone(), p1.clone(), 0.1, 0.2)?;
        let runs = 10_000;
        let mut total = 0usize;
        let mut accept_h0 = 0usize;
        for _ in 0..runs {
            test.reset();
            let mut d = Decision::Continue;
            while d == Decision::Continue && test.observations() < 10_000 {
                d = test.update(rng.next_f64() < p);
            }
            assert_ne!(d, Decision::Continue);
            total += test.observations();
            accept_h0 += usize::from(d == Decision::AcceptH0);
        }
        let mean_n = total as f64 / f64::from(runs);
        let l_sim = accept_h0 as f64 / f64::from(runs);
        assert!(
            mean_n > n_wald * 0.98 && mean_n < n_wald * 1.35,
            "p = {p}: simulated E[N] {mean_n} vs Wald {n_wald}"
        );
        // The realised error rates are at most α/(1−β) and β/(1−α); Monte-Carlo sd ≈ 0.004.
        assert!(
            (l_sim - l_wald).abs() < 0.06,
            "p = {p}: simulated L {l_sim} vs Wald {l_wald}"
        );
        close(
            expected_sample_size_bernoulli(p, &p0, &p1, 0.1, 0.2)?,
            n_wald,
            1e-8,
        );
    }
    Ok(())
}
