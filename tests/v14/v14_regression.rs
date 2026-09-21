//! symplex 0.14 — regression.  Reference values cite scipy 1.18 / statsmodels 0.15 /
//! numpy 2.5 (`symplex/.venv/bin/python`).  Exact rationals were computed with
//! `fractions.Fraction` Gauss–Jordan on the normal equations in the same script
//! and cross-checked against the statsmodels floats.

use symplex::linprog::{q, qi};
use symplex::num_traits::{One, Zero};
use symplex::prelude::*;
use symplex::stats::data::{self, Ddof, from_i64};
use symplex::stats::hypothesis::Alternative;
use symplex::stats::regression::{
    Design, LogitOpts, hat_matrix, logit, ols, polyfit, r_squared_from_correlation,
    simple_linear_regression, slope_from_correlation, vif, wls,
};

fn close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() < tol,
        "got {actual}, expected {expected} (tol {tol})"
    );
}

fn ev(e: &Ex) -> f64 {
    e.eval_f64().unwrap()
}

fn col(x: &[i64]) -> Vec<Vec<Q>> {
    x.iter().map(|&v| vec![qi(v)]).collect()
}

fn rows(r: &[&[i64]]) -> Vec<Vec<Q>> {
    r.iter().map(|row| from_i64(row)).collect()
}

fn qs(entries: &[(i64, i64)]) -> Vec<Q> {
    entries.iter().map(|&(n, d)| q(n, d)).collect()
}

// D1: x = 1..7, y = [2, 3, 5, 4, 6, 8, 9]  (statsmodels OLS(y, add_constant(x)))
fn d1() -> (Vec<Q>, Vec<Q>) {
    (
        from_i64(&[1, 2, 3, 4, 5, 6, 7]),
        from_i64(&[2, 3, 5, 4, 6, 8, 9]),
    )
}

// D2: two regressors, n = 8
fn d2() -> (Vec<Vec<Q>>, Vec<Q>) {
    (
        rows(&[
            &[1, 5],
            &[2, 3],
            &[3, 8],
            &[4, 1],
            &[5, 7],
            &[6, 2],
            &[7, 9],
            &[8, 4],
        ]),
        from_i64(&[6, 5, 10, 4, 12, 7, 15, 9]),
    )
}

// D3: no intercept, x = 1..5, y = [2, 3, 7, 8, 11]
fn d3() -> (Vec<Vec<Q>>, Vec<Q>) {
    (col(&[1, 2, 3, 4, 5]), from_i64(&[2, 3, 7, 8, 11]))
}

// D4: WLS on D1 with weights [1, 2, 3, 1, 2, 3, 1]
fn d4_weights() -> Vec<Q> {
    from_i64(&[1, 2, 3, 1, 2, 3, 1])
}

// ═══════════════════════════════════════════════════════════════════════════
// OLS — simple linear regression (D1)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ols_simple_coefficients_fitted_residuals_exact() {
    // statsmodels: OLS(y, add_constant(x)).fit().params = [0.7142857142857169, 1.1428571428571432]
    let (x, y) = d1();
    let fit = ols(&y, &col(&[1, 2, 3, 4, 5, 6, 7]), true).unwrap();
    assert_eq!(fit.coefficients, vec![q(5, 7), q(8, 7)]);
    assert_eq!(
        fit.fitted,
        qs(&[(13, 7), (3, 1), (29, 7), (37, 7), (45, 7), (53, 7), (61, 7)])
    );
    assert_eq!(
        fit.residuals,
        qs(&[(1, 7), (0, 1), (6, 7), (-9, 7), (-3, 7), (3, 7), (2, 7)])
    );
    assert_eq!(fit.nobs(), 7);
    assert_eq!(fit.n_params(), 2);
    assert!(fit.has_constant());
    assert_eq!(fit.response(), &y[..]);
    assert_eq!(fit.design().shape(), (7, 2));
    assert_eq!(fit.design().col(1), x);
}

#[test]
fn ols_simple_sums_of_squares_and_r_squared_exact() {
    // statsmodels: ssr 2.8571428571428568, ess 36.57142857142858, centered_tss 39.42857142857143,
    // rsquared 0.927536231884058, rsquared_adj 0.9130434782608696, mse_resid 0.5714285714285714
    let (_, y) = d1();
    let fit = ols(&y, &col(&[1, 2, 3, 4, 5, 6, 7]), true).unwrap();
    assert_eq!(fit.ssr, q(20, 7));
    assert_eq!(fit.ess, q(256, 7));
    assert_eq!(fit.tss, q(276, 7));
    assert_eq!(fit.r_squared, q(64, 69));
    assert_eq!(fit.adjusted_r_squared, q(21, 23));
    assert_eq!(fit.mse_resid, q(4, 7));
    assert_eq!((fit.df_model, fit.df_resid), (1, 5));
    close(
        data::to_f64(std::slice::from_ref(&fit.r_squared))[0],
        0.927_536_231_884_058,
        1e-15,
    );
}

#[test]
fn ols_simple_cov_params_and_standard_errors() {
    // statsmodels: normalized_cov_params = [[5/7, -1/7], [-1/7, 1/28]],
    // cov_params() = mse_resid * that = [[20/49, -4/49], [-4/49, 1/49]],
    // bse = [0.6388765649999398, 0.14285714285714288]
    let ctx = Context::new();
    let (_, y) = d1();
    let fit = ols(&y, &col(&[1, 2, 3, 4, 5, 6, 7]), true).unwrap();
    assert_eq!(
        fit.normalized_cov_params().to_rows(),
        vec![qs(&[(5, 7), (-1, 7)]), qs(&[(-1, 7), (1, 28)])]
    );
    assert_eq!(
        fit.cov_params.to_rows(),
        vec![qs(&[(20, 49), (-4, 49)]), qs(&[(-4, 49), (1, 49)])]
    );
    let se = fit.standard_errors(&ctx);
    close(ev(&se[0]), 0.638_876_564_999_939_8, 1e-12);
    // se₁ = √(1/49) = 1/7 exactly.
    assert_eq!(se[1].as_rational(), Some(q(1, 7)));
    // np.sqrt(mse_resid) = 0.7559289460184544
    close(
        ev(&fit.residual_standard_error(&ctx)),
        0.755_928_946_018_454_4,
        1e-12,
    );
}

#[test]
fn ols_simple_t_statistics_and_p_values() {
    // statsmodels: tvalues [1.118033988749899, 8.000000000000002],
    // pvalues [0.31437263764701545, 0.00049290666057244]
    let ctx = Context::new();
    let (_, y) = d1();
    let fit = ols(&y, &col(&[1, 2, 3, 4, 5, 6, 7]), true).unwrap();
    let t = fit.t_statistics(&ctx).unwrap();
    close(ev(&t[0]), 1.118_033_988_749_899, 1e-12);
    // t₀ = (5/7)/(2√5/7) = √5/2 — squared it is exactly 5/4.
    assert_eq!(t[0].powi(2).simplify().as_rational(), Some(q(5, 4)));
    assert_eq!(t[1].as_rational(), Some(qi(8)));
    let p = fit.p_values(&ctx).unwrap();
    close(ev(&p[0]), 0.314_372_637_647_015_45, 1e-12);
    close(ev(&p[1]), 0.000_492_906_660_572_44, 1e-12);
}

#[test]
fn ols_simple_coefficient_tests_are_two_sided_student_t() {
    let ctx = Context::new();
    let (_, y) = d1();
    let fit = ols(&y, &col(&[1, 2, 3, 4, 5, 6, 7]), true).unwrap();
    let tests = fit.coefficient_tests(&ctx).unwrap();
    assert_eq!(tests.len(), 2);
    for t in &tests {
        assert_eq!(t.alternative, Alternative::TwoSided);
        assert_eq!(t.df.as_ref().and_then(Ex::as_rational), Some(qi(5)));
    }
    close(tests[1].statistic_f64().unwrap(), 8.0, 1e-12);
    // scipy: linregress(x, y).pvalue = 0.0004929066605724445
    close(
        tests[1].p_value_f64().unwrap(),
        0.000_492_906_660_572_444_5,
        1e-12,
    );
}

#[test]
fn ols_simple_overall_f_test() {
    // statsmodels: fvalue 64.00000000000001, f_pvalue 0.0004929066605724437
    let ctx = Context::new();
    let (_, y) = d1();
    let fit = ols(&y, &col(&[1, 2, 3, 4, 5, 6, 7]), true).unwrap();
    assert_eq!(fit.f_statistic().unwrap(), qi(64));
    let f = fit.f_test(&ctx).unwrap();
    assert_eq!(f.statistic_exact(), Some(qi(64)));
    assert_eq!(f.alternative, Alternative::Greater);
    assert_eq!(f.df.as_ref().and_then(Ex::as_rational), Some(qi(5)));
    close(f.p_value_f64().unwrap(), 0.000_492_906_660_572_443_7, 1e-12);
    // With one regressor, F = t² and the p-values coincide.
    let p = fit.p_values(&ctx).unwrap();
    close(f.p_value_f64().unwrap(), ev(&p[1]), 1e-14);
}

#[test]
fn ols_simple_anova_table() {
    let (_, y) = d1();
    let fit = ols(&y, &col(&[1, 2, 3, 4, 5, 6, 7]), true).unwrap();
    let t = fit.anova_table().unwrap();
    assert_eq!(t.ss_model, q(256, 7));
    assert_eq!(t.df_model, 1);
    assert_eq!(t.ms_model, q(256, 7));
    assert_eq!(t.ss_resid, q(20, 7));
    assert_eq!(t.df_resid, 5);
    assert_eq!(t.ms_resid, q(4, 7));
    assert_eq!(t.ss_total, q(276, 7));
    assert_eq!(t.df_total, 6);
    assert_eq!(t.f, qi(64));
    assert_eq!(&t.ss_model + &t.ss_resid, t.ss_total);
}

#[test]
fn ols_simple_conf_int_matches_statsmodels() {
    // statsmodels: conf_int(0.05) = [[-0.9279987789168518, 2.3565702074882857],
    //                                [0.7756311663376696, 1.5100831193766169]]
    //              conf_int(0.10) = [[-0.5730814687780013, 2.0016528973494347],
    //                                [0.8549930895238542, 1.4307211961904323]]
    let ctx = Context::new();
    let (_, y) = d1();
    let fit = ols(&y, &col(&[1, 2, 3, 4, 5, 6, 7]), true).unwrap();
    let ci = fit.conf_int(&ctx, 0.95).unwrap();
    close(ci[0].0, -0.927_998_778_916_851_8, 1e-9);
    close(ci[0].1, 2.356_570_207_488_285_7, 1e-9);
    close(ci[1].0, 0.775_631_166_337_669_6, 1e-9);
    close(ci[1].1, 1.510_083_119_376_616_9, 1e-9);
    let ci = fit.conf_int(&ctx, 0.90).unwrap();
    close(ci[0].0, -0.573_081_468_778_001_3, 1e-9);
    close(ci[0].1, 2.001_652_897_349_434_7, 1e-9);
    close(ci[1].0, 0.854_993_089_523_854_2, 1e-9);
    close(ci[1].1, 1.430_721_196_190_432_3, 1e-9);
    assert!(fit.conf_int(&ctx, 1.0).is_err());
    assert!(fit.conf_int(&ctx, 0.0).is_err());
}

#[test]
fn ols_simple_predict_and_intervals() {
    // statsmodels: get_prediction([1, 8]).summary_frame(): mean 9.857142857142863,
    //   mean_ci (8.214858363940294, 11.499427350345432), obs_ci (7.312926660379568, 12.401359053906159)
    // get_prediction([1, 2.5]): mean 3.5714285714285747,
    //   mean_ci (2.653363630129891, 4.489493512727258), obs_ci (1.4222936441378704, 5.7205634987192795)
    let ctx = Context::new();
    let (_, y) = d1();
    let fit = ols(&y, &col(&[1, 2, 3, 4, 5, 6, 7]), true).unwrap();
    assert_eq!(fit.predict(&[qi(8)]).unwrap(), q(69, 7));
    let (lo, hi) = fit
        .confidence_interval_mean_response(&ctx, &[qi(8)], 0.95)
        .unwrap();
    close(lo, 8.214_858_363_940_294, 1e-9);
    close(hi, 11.499_427_350_345_432, 1e-9);
    let (lo, hi) = fit.prediction_interval(&ctx, &[qi(8)], 0.95).unwrap();
    close(lo, 7.312_926_660_379_568, 1e-9);
    close(hi, 12.401_359_053_906_159, 1e-9);

    assert_eq!(fit.predict(&[q(5, 2)]).unwrap(), q(25, 7));
    let (lo, hi) = fit
        .confidence_interval_mean_response(&ctx, &[q(5, 2)], 0.95)
        .unwrap();
    close(lo, 2.653_363_630_129_891, 1e-9);
    close(hi, 4.489_493_512_727_258, 1e-9);
    let (lo, hi) = fit.prediction_interval(&ctx, &[q(5, 2)], 0.95).unwrap();
    close(lo, 1.422_293_644_137_870_4, 1e-9);
    close(hi, 5.720_563_498_719_279_5, 1e-9);

    assert!(fit.predict(&[qi(1), qi(2)]).is_err());
    assert!(fit.prediction_interval(&ctx, &[qi(8)], 1.5).is_err());
}

#[test]
fn ols_simple_leverage_cooks_distance_durbin_watson_exact() {
    // statsmodels: get_influence().hat_matrix_diag = [0.46428571428571425, 0.2857142857142857,
    //   0.17857142857142852, 0.14285714285714288, 0.17857142857142866, 0.2857142857142858, 0.46428571428571447]
    // cooks_distance[0] = [2.8888888888887684e-02, ~0, 1.7013232514177534e-01, 2.8125000000000205e-01,
    //   4.2533081285445189e-02, 8.9999999999997929e-02, 1.1555555555555053e-01]
    // durbin_watson(resid) = 2.3928571428571432
    let (_, y) = d1();
    let fit = ols(&y, &col(&[1, 2, 3, 4, 5, 6, 7]), true).unwrap();
    let lev = fit.leverage();
    assert_eq!(
        lev,
        qs(&[(13, 28), (2, 7), (5, 28), (1, 7), (5, 28), (2, 7), (13, 28)])
    );
    assert_eq!(lev.iter().fold(Q::zero(), |a, b| a + b), qi(2)); // tr H = p
    assert_eq!(
        fit.cooks_distance().unwrap(),
        qs(&[
            (13, 450),
            (0, 1),
            (90, 529),
            (9, 32),
            (45, 1058),
            (9, 100),
            (26, 225)
        ])
    );
    assert_eq!(fit.durbin_watson().unwrap(), q(67, 28));
    close(
        data::to_f64(&[q(67, 28)])[0],
        2.392_857_142_857_143_2,
        1e-15,
    );
}

#[test]
fn ols_simple_hat_matrix_is_symmetric_idempotent_projection() {
    let (_, y) = d1();
    let fit = ols(&y, &col(&[1, 2, 3, 4, 5, 6, 7]), true).unwrap();
    let h = fit.hat_matrix().unwrap();
    assert_eq!(h.shape(), (7, 7));
    assert!(h.is_symmetric());
    assert_eq!(h.matmul(&h).unwrap(), h);
    assert_eq!(h.trace().unwrap(), qi(2));
    assert_eq!(h.diagonal(), fit.leverage());
    // ŷ = H y
    let ycol = QMatrix::new(y.iter().map(|v| vec![v.clone()]).collect()).unwrap();
    assert_eq!(h.matmul(&ycol).unwrap().col(0), fit.fitted);
    // The free function on the design matrix agrees.
    assert_eq!(hat_matrix(fit.design()).unwrap(), h);
    // A rank-deficient design is rejected.
    let bad = QMatrix::from_i64(&[&[1, 2], &[2, 4], &[3, 6]]).unwrap();
    assert!(matches!(
        hat_matrix(&bad),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

#[test]
fn ols_simple_log_likelihood_aic_bic() {
    // statsmodels: llf -6.796261646484484, aic 17.59252329296897, bic 17.484343591079593
    let ctx = Context::new();
    let (_, y) = d1();
    let fit = ols(&y, &col(&[1, 2, 3, 4, 5, 6, 7]), true).unwrap();
    close(
        ev(&fit.log_likelihood(&ctx).unwrap()),
        -6.796_261_646_484_484,
        1e-12,
    );
    close(ev(&fit.aic(&ctx).unwrap()), 17.592_523_292_968_97, 1e-12);
    close(ev(&fit.bic(&ctx).unwrap()), 17.484_343_591_079_593, 1e-12);
}

#[test]
fn simple_linear_regression_matches_scipy_linregress() {
    // scipy: linregress(x, y) → slope 1.1428571428571428, intercept 0.7142857142857144,
    //   rvalue 0.9630868246861536, pvalue 0.0004929066605724445,
    //   stderr 0.14285714285714285, intercept_stderr 0.6388765649999399
    let ctx = Context::new();
    let (x, y) = d1();
    let fit = simple_linear_regression(&x, &y).unwrap();
    assert_eq!(fit.coefficients, vec![q(5, 7), q(8, 7)]);
    assert_eq!(fit.r_squared, q(64, 69));
    close(
        data::to_f64(std::slice::from_ref(&fit.r_squared))[0].sqrt(),
        0.963_086_824_686_153_6,
        1e-15,
    );
    let se = fit.standard_errors(&ctx);
    close(ev(&se[1]), 0.142_857_142_857_142_85, 1e-15);
    close(ev(&se[0]), 0.638_876_564_999_939_9, 1e-12);
    close(
        ev(&fit.p_values(&ctx).unwrap()[1]),
        0.000_492_906_660_572_444_5,
        1e-12,
    );
    assert!(simple_linear_regression(&x[..6], &y).is_err());
}

#[test]
fn correlation_bridges_recover_r_squared_and_slope() {
    // scipy: pearsonr(x, y).statistic = 0.9630868246861537; r² = 64/69; slope = r·s_y/s_x = 8/7
    let ctx = Context::new();
    let (x, y) = d1();
    let r = data::pearson(&ctx, &x, &y).unwrap();
    close(ev(&r), 0.963_086_824_686_153_7, 1e-15);
    let r2 = r_squared_from_correlation(&r);
    assert_eq!(r2.as_rational(), Some(q(64, 69)));
    let sx = data::std(&ctx, &x, Ddof::Sample).unwrap();
    let sy = data::std(&ctx, &y, Ddof::Sample).unwrap();
    let slope = slope_from_correlation(&r, &sx, &sy).unwrap();
    close(ev(&slope), 8.0 / 7.0, 1e-15);
    assert_eq!(slope.as_rational(), Some(q(8, 7)));
    // Population sds give the same slope.
    let sx0 = data::std(&ctx, &x, Ddof::Population).unwrap();
    let sy0 = data::std(&ctx, &y, Ddof::Population).unwrap();
    close(
        ev(&slope_from_correlation(&r, &sx0, &sy0).unwrap()),
        8.0 / 7.0,
        1e-15,
    );
    assert!(slope_from_correlation(&r, &ctx.zero(), &sy).is_err());
}

#[test]
fn polyfit_degree_one_returns_highest_power_first() {
    // numpy: polyfit(x, y, 1) = [1.1428571428571426, 0.7142857142857142]
    let (x, y) = d1();
    assert_eq!(polyfit(&x, &y, 1).unwrap(), vec![q(8, 7), q(5, 7)]);
    // Degree 0 is the mean.
    assert_eq!(polyfit(&x, &y, 0).unwrap(), vec![q(37, 7)]);
}

// ═══════════════════════════════════════════════════════════════════════════
// OLS — two regressors (D2)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ols_two_regressors_coefficients_and_residuals_exact() {
    // statsmodels: params [0.18705220061411953, 0.693909928352098, 1.0646878198567045]
    let (x, y) = d2();
    let fit = ols(&y, &x, true).unwrap();
    assert_eq!(
        fit.coefficients,
        qs(&[(731, 3908), (13559, 19540), (5201, 4885)])
    );
    assert_eq!(
        fit.fitted,
        qs(&[
            (60617, 9770),
            (18637, 3908),
            (52691, 4885),
            (15739, 3908),
            (108539, 9770),
            (126617, 19540),
            (71451, 4885),
            (195343, 19540)
        ])
    );
    assert_eq!(
        fit.residuals,
        qs(&[
            (-1997, 9770),
            (903, 3908),
            (-3841, 4885),
            (-107, 3908),
            (8701, 9770),
            (10163, 19540),
            (1824, 4885),
            (-19483, 19540)
        ])
    );
    // Residuals are orthogonal to every column of X (normal equations).
    for j in 0..3 {
        let c = fit.design().col(j);
        let s = c
            .iter()
            .zip(&fit.residuals)
            .fold(Q::zero(), |a, (u, e)| a + u * e);
        assert!(s.is_zero());
    }
}

#[test]
fn ols_two_regressors_fit_statistics_exact() {
    // statsmodels: ssr 2.9114124872057303, ess 95.08858751279428, rsquared 0.9702917093142273,
    //   rsquared_adj 0.9584083930399182, mse_resid 0.582282497441146, fvalue 81.65159345391909,
    //   f_pvalue 0.0001521227478835553, df_model 2, df_resid 5
    let ctx = Context::new();
    let (x, y) = d2();
    let fit = ols(&y, &x, true).unwrap();
    assert_eq!(fit.ssr, q(56889, 19540));
    assert_eq!(fit.tss, qi(98));
    assert_eq!(fit.ess, q(1858031, 19540));
    assert_eq!(fit.r_squared, q(37919, 39080));
    assert_eq!(fit.adjusted_r_squared, q(187273, 195400));
    assert_eq!(fit.mse_resid, q(56889, 97700));
    assert_eq!((fit.df_model, fit.df_resid), (2, 5));
    assert_eq!(fit.f_statistic().unwrap(), q(189595, 2322));
    let f = fit.f_test(&ctx).unwrap();
    close(f.statistic_f64().unwrap(), 81.651_593_453_919_09, 1e-12);
    close(f.p_value_f64().unwrap(), 0.000_152_122_747_883_555_3, 1e-12);
    let t = fit.anova_table().unwrap();
    assert_eq!(t.ms_model, q(1858031, 39080));
    assert_eq!(t.df_total, 7);
}

#[test]
fn ols_two_regressors_cov_params_exact_and_bse() {
    // statsmodels: bse [0.7330990583423428, 0.11847181498782183, 0.10006316304606439]
    let ctx = Context::new();
    let (x, y) = d2();
    let fit = ols(&y, &x, true).unwrap();
    assert_eq!(
        fit.cov_params.to_rows(),
        vec![
            qs(&[
                (205198623, 381811600),
                (-21674709, 381811600),
                (-1024002, 23863225)
            ]),
            qs(&[
                (-21674709, 381811600),
                (26794719, 1909058000),
                (-625779, 477264500)
            ]),
            qs(&[
                (-1024002, 23863225),
                (-625779, 477264500),
                (1194669, 119316125)
            ]),
        ]
    );
    assert!(fit.cov_params.is_symmetric());
    let se = fit.standard_errors(&ctx);
    close(ev(&se[0]), 0.733_099_058_342_342_8, 1e-12);
    close(ev(&se[1]), 0.118_471_814_987_821_83, 1e-12);
    close(ev(&se[2]), 0.100_063_163_046_064_39, 1e-12);
    // √mse_resid = 0.7630743721559167
    close(
        ev(&fit.residual_standard_error(&ctx)),
        0.763_074_372_155_916_7,
        1e-12,
    );
}

#[test]
fn ols_two_regressors_t_and_p_values() {
    // statsmodels: tvalues [0.25515269524022477, 5.85717310419721, 10.640157550951814]
    //   pvalues [8.087681377094662e-01, 2.055751582282303e-03, 1.268576692379607e-04]
    let ctx = Context::new();
    let (x, y) = d2();
    let fit = ols(&y, &x, true).unwrap();
    let t = fit.t_statistics(&ctx).unwrap();
    close(ev(&t[0]), 0.255_152_695_240_224_77, 1e-12);
    close(ev(&t[1]), 5.857_173_104_197_21, 1e-12);
    close(ev(&t[2]), 10.640_157_550_951_814, 1e-12);
    let p = fit.p_values(&ctx).unwrap();
    close(ev(&p[0]), 0.808_768_137_709_466_2, 1e-12);
    close(ev(&p[1]), 0.002_055_751_582_282_303, 1e-12);
    close(ev(&p[2]), 0.000_126_857_669_237_960_7, 1e-12);
    let tests = fit.coefficient_tests(&ctx).unwrap();
    assert_eq!(tests.len(), 3);
    close(
        tests[2].statistic_f64().unwrap(),
        10.640_157_550_951_814,
        1e-12,
    );
    close(
        tests[2].p_value_f64().unwrap(),
        0.000_126_857_669_237_960_7,
        1e-12,
    );
}

#[test]
fn ols_two_regressors_conf_int_95_and_99() {
    // statsmodels: conf_int(0.05) = [[-1.6974389224827937, 2.071543323711033],
    //   [0.3893684327095371, 0.9984514239946589], [0.8074672705141764, 1.3219083691992326]]
    // conf_int(0.01) = [[-2.768908023731902, 3.143012424960141],
    //   [0.21621463079989922, 1.1716052259042966], [0.6612188390681732, 1.4681568006452357]]
    let ctx = Context::new();
    let (x, y) = d2();
    let fit = ols(&y, &x, true).unwrap();
    let ci = fit.conf_int(&ctx, 0.95).unwrap();
    close(ci[0].0, -1.697_438_922_482_793_7, 1e-9);
    close(ci[0].1, 2.071_543_323_711_033, 1e-9);
    close(ci[1].0, 0.389_368_432_709_537_1, 1e-9);
    close(ci[1].1, 0.998_451_423_994_658_9, 1e-9);
    close(ci[2].0, 0.807_467_270_514_176_4, 1e-9);
    close(ci[2].1, 1.321_908_369_199_232_6, 1e-9);
    let ci = fit.conf_int(&ctx, 0.99).unwrap();
    close(ci[0].0, -2.768_908_023_731_902, 1e-9);
    close(ci[0].1, 3.143_012_424_960_141, 1e-9);
    close(ci[1].0, 0.216_214_630_799_899_22, 1e-9);
    close(ci[1].1, 1.171_605_225_904_296_6, 1e-9);
    close(ci[2].0, 0.661_218_839_068_173_2, 1e-9);
    close(ci[2].1, 1.468_156_800_645_235_7, 1e-9);
}

#[test]
fn ols_two_regressors_predict_and_intervals() {
    // statsmodels: get_prediction([1, 9, 6]).summary_frame(): mean 12.820368474923228,
    //   mean_ci (11.285745794051824, 14.354991155794632), obs_ci (10.329841215158392, 15.310895734688064)
    let ctx = Context::new();
    let (x, y) = d2();
    let fit = ols(&y, &x, true).unwrap();
    let x0 = [qi(9), qi(6)];
    assert_eq!(fit.predict(&x0).unwrap(), q(25051, 1954));
    let (lo, hi) = fit
        .confidence_interval_mean_response(&ctx, &x0, 0.95)
        .unwrap();
    close(lo, 11.285_745_794_051_824, 1e-9);
    close(hi, 14.354_991_155_794_632, 1e-9);
    let (lo, hi) = fit.prediction_interval(&ctx, &x0, 0.95).unwrap();
    close(lo, 10.329_841_215_158_392, 1e-9);
    close(hi, 15.310_895_734_688_064, 1e-9);
    // Predicting at an observed row returns its fitted value.
    assert_eq!(fit.predict(&x[2]).unwrap(), fit.fitted[2]);
}

#[test]
fn ols_two_regressors_leverage_cooks_durbin_watson_exact() {
    // statsmodels: hat_matrix_diag [0.4225179119754351, 0.31499488229273287, 0.36827021494370515,
    //   0.38050153531218023, 0.20388945752302962, 0.3407881269191401, 0.5218014329580347, 0.44723643807574176]
    // cooks_distance[0] [3.0302614452944410e-02, 2.0517633381207826e-02, 3.2659413050181152e-01,
    //   4.2548024357876626e-04, 1.4606368034250630e-01, 1.2144346654046742e-01, 1.8211858517738755e-01,
    //   8.3304012029125374e-01]; durbin_watson(resid) 1.6075322091523627
    let (x, y) = d2();
    let fit = ols(&y, &x, true).unwrap();
    assert_eq!(
        fit.leverage(),
        qs(&[
            (2064, 4885),
            (1231, 3908),
            (1799, 4885),
            (1487, 3908),
            (996, 4885),
            (6659, 19540),
            (2549, 4885),
            (8739, 19540)
        ])
    );
    let cooks = fit.cooks_distance().unwrap();
    assert_eq!(
        cooks,
        qs(&[
            (319040720, 10528488243),
            (1323325, 64496961),
            (18957966085, 58047479469),
            (425616575, 1000320417747),
            (2564781340, 17559336681),
            (3438926314855, 28317096117387),
            (18403780, 101053827),
            (1842896288095, 2212253939763)
        ])
    );
    // float(Fraction(1842896288095, 2212253939763)) = 0.8330401202912675; statsmodels'
    // 0.83304012029125374 carries ~1.4e-14 of its own rounding.
    close(
        data::to_f64(&[cooks[7].clone()])[0],
        0.833_040_120_291_267_5,
        1e-15,
    );
    close(
        data::to_f64(&[cooks[7].clone()])[0],
        0.833_040_120_291_253_7,
        1e-13,
    );
    assert_eq!(fit.durbin_watson().unwrap(), q(1786950583, 1111611060));
    close(
        data::to_f64(&[fit.durbin_watson().unwrap()])[0],
        1.607_532_209_152_362_7,
        1e-15,
    );
}

#[test]
fn ols_two_regressors_aic_bic() {
    // statsmodels: llf -7.308295515765012, aic 20.616591031530024, bic 20.85491565656953
    let ctx = Context::new();
    let (x, y) = d2();
    let fit = ols(&y, &x, true).unwrap();
    close(
        ev(&fit.log_likelihood(&ctx).unwrap()),
        -7.308_295_515_765_012,
        1e-12,
    );
    close(ev(&fit.aic(&ctx).unwrap()), 20.616_591_031_530_024, 1e-12);
    close(ev(&fit.bic(&ctx).unwrap()), 20.854_915_656_569_53, 1e-12);
}

#[test]
fn design_builder_matches_ols_rows() {
    let (x, y) = d2();
    let x1: Vec<Q> = x.iter().map(|r| r[0].clone()).collect();
    let x2: Vec<Q> = x.iter().map(|r| r[1].clone()).collect();
    let design = Design::new().intercept().column(&x1).column(&x2);
    assert!(design.has_intercept());
    assert_eq!(design.n_columns(), 2);
    assert_eq!(design.rows(8).unwrap(), x);
    assert_eq!(design.fit(&y).unwrap(), ols(&y, &x, true).unwrap());
    // Column-length mismatch is caught.
    assert!(Design::new().intercept().column(&x1[..5]).fit(&y).is_err());
    // Intercept-only design.
    let only = Design::new().intercept().fit(&y).unwrap();
    assert_eq!(only.coefficients, vec![qi(17) / qi(2)]);
    assert_eq!(only.r_squared, Q::zero());
    assert_eq!(only.df_model, 0);
    assert!(matches!(
        only.f_statistic(),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(only.anova_table().is_err());
}

#[test]
fn vif_two_regressors_exact() {
    // statsmodels: variance_inflation_factor(add_constant(X), 1) = 1.012384851586489 (and column 2)
    let (x, _) = d2();
    let v = vif(&x).unwrap();
    assert_eq!(v, vec![q(9891, 9770), q(9891, 9770)]);
    close(data::to_f64(&v)[0], 1.012_384_851_586_489, 1e-15);
}

// ═══════════════════════════════════════════════════════════════════════════
// OLS — no intercept (D3) and constant detection
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ols_no_intercept_uses_uncentered_tss() {
    // statsmodels: OLS(y, x).fit(): params [2.109090909090909], ssr 2.3454545454545457,
    //   ess 244.65454545454546, uncentered_tss 247, rsquared 0.9905042326094957,
    //   rsquared_adj 0.9881302907618696, mse_resid 0.5863636363636364, fvalue 417.2403100775193,
    //   df_model 1, df_resid 4
    let (x, y) = d3();
    let fit = ols(&y, &x, false).unwrap();
    assert!(!fit.has_constant());
    assert_eq!(fit.coefficients, vec![q(116, 55)]);
    assert_eq!(
        fit.fitted,
        qs(&[(116, 55), (232, 55), (348, 55), (464, 55), (116, 11)])
    );
    assert_eq!(
        fit.residuals,
        qs(&[(-6, 55), (-67, 55), (37, 55), (-24, 55), (5, 11)])
    );
    assert_eq!(fit.ssr, q(129, 55));
    assert_eq!(fit.tss, qi(247));
    assert_eq!(fit.ess, q(13456, 55));
    assert_eq!(fit.r_squared, q(13456, 13585));
    assert_eq!(fit.adjusted_r_squared, q(10739, 10868));
    assert_eq!(fit.mse_resid, q(129, 220));
    assert_eq!((fit.df_model, fit.df_resid), (1, 4));
    assert_eq!(fit.f_statistic().unwrap(), q(53824, 129));
    assert_eq!(fit.cov_params.to_rows(), vec![vec![q(129, 12100)]]);
}

#[test]
fn ols_no_intercept_inference_and_information_criteria() {
    // statsmodels: bse [0.10325287901455041], tvalues [20.426461026754474],
    //   pvalues [3.392120339091609e-05], f_pvalue 3.3921203390916105e-05,
    //   conf_int(0.05) [[1.8224149585533804, 2.3957668596284374]],
    //   conf_int(0.10) [[1.8889715907847653, 2.3292102273970525]],
    //   llf -5.202295932761115, aic 12.40459186552223, bic 12.01402977795633
    let ctx = Context::new();
    let (x, y) = d3();
    let fit = ols(&y, &x, false).unwrap();
    close(
        ev(&fit.standard_errors(&ctx)[0]),
        0.103_252_879_014_550_41,
        1e-12,
    );
    close(
        ev(&fit.t_statistics(&ctx).unwrap()[0]),
        20.426_461_026_754_474,
        1e-12,
    );
    close(
        ev(&fit.p_values(&ctx).unwrap()[0]),
        3.392_120_339_091_609e-5,
        1e-14,
    );
    close(
        fit.f_test(&ctx).unwrap().p_value_f64().unwrap(),
        3.392_120_339_091_610_5e-5,
        1e-14,
    );
    let ci = fit.conf_int(&ctx, 0.95).unwrap();
    close(ci[0].0, 1.822_414_958_553_380_4, 1e-9);
    close(ci[0].1, 2.395_766_859_628_437_4, 1e-9);
    let ci = fit.conf_int(&ctx, 0.90).unwrap();
    close(ci[0].0, 1.888_971_590_784_765_3, 1e-9);
    close(ci[0].1, 2.329_210_227_397_052_5, 1e-9);
    close(
        ev(&fit.log_likelihood(&ctx).unwrap()),
        -5.202_295_932_761_115,
        1e-12,
    );
    close(ev(&fit.aic(&ctx).unwrap()), 12.404_591_865_522_23, 1e-12);
    close(ev(&fit.bic(&ctx).unwrap()), 12.014_029_777_956_33, 1e-12);
}

#[test]
fn ols_no_intercept_prediction_and_influence() {
    // statsmodels: get_prediction([6]).summary_frame(): mean 12.654545454545453,
    //   mean_ci (10.934489751320282, 14.374601157770623), obs_ci (9.919831181333315, 15.38925972775759)
    // hat_matrix_diag [0.01818181818181817, 0.07272727272727272, 0.1636363636363636, 0.2909090909090909,
    //   0.4545454545454544]; cooks_distance[0] [3.8281175232079463e-04, 2.1406197377871938e-01,
    //   1.8055128148766916e-01, 1.8788128984908881e-01, 5.3832902670112026e-01]; durbin_watson 2.911768851303736
    let ctx = Context::new();
    let (x, y) = d3();
    let fit = ols(&y, &x, false).unwrap();
    assert_eq!(fit.predict(&[qi(6)]).unwrap(), q(696, 55));
    let (lo, hi) = fit
        .confidence_interval_mean_response(&ctx, &[qi(6)], 0.95)
        .unwrap();
    close(lo, 10.934_489_751_320_282, 1e-9);
    close(hi, 14.374_601_157_770_623, 1e-9);
    let (lo, hi) = fit.prediction_interval(&ctx, &[qi(6)], 0.95).unwrap();
    close(lo, 9.919_831_181_333_315, 1e-9);
    close(hi, 15.389_259_727_757_59, 1e-9);
    assert_eq!(
        fit.leverage(),
        qs(&[(1, 55), (4, 55), (9, 55), (16, 55), (5, 11)])
    );
    assert_eq!(
        fit.cooks_distance().unwrap(),
        qs(&[
            (4, 10449),
            (71824, 335529),
            (4107, 22747),
            (4096, 21801),
            (625, 1161)
        ])
    );
    assert_eq!(fit.durbin_watson().unwrap(), q(20659, 7095));
    assert_eq!(fit.hat_matrix().unwrap().trace().unwrap(), qi(1));
}

#[test]
fn ols_detects_user_supplied_constant_column() {
    // statsmodels: OLS(y, column_stack([ones, x])).fit(): k_constant 1, df_model 1, rsquared 0.927536231884058
    let (x, y) = d1();
    let with_ones: Vec<Vec<Q>> = x.iter().map(|v| vec![qi(1), v.clone()]).collect();
    let fit = ols(&y, &with_ones, false).unwrap();
    assert!(fit.has_constant());
    assert_eq!(fit.coefficients, vec![q(5, 7), q(8, 7)]);
    assert_eq!(fit.r_squared, q(64, 69));
    assert_eq!(fit.adjusted_r_squared, q(21, 23));
    assert_eq!(fit.df_model, 1);
    // Prediction rows now include the constant column explicitly.
    assert_eq!(fit.predict(&[qi(1), qi(8)]).unwrap(), q(69, 7));
    assert!(fit.predict(&[qi(8)]).is_err());
}

#[test]
fn ols_detects_implicit_constant_from_full_dummy_set() {
    // statsmodels: OLS(y, dummies).fit(): k_constant 1, params [3.0, 7.5], rsquared 0.8321917808219178,
    //   df_model 1, fvalue 24.795918367346946, f_pvalue 0.004177335830133132
    let ctx = Context::new();
    let x = rows(&[
        &[1, 0],
        &[1, 0],
        &[1, 0],
        &[0, 1],
        &[0, 1],
        &[0, 1],
        &[0, 1],
    ]);
    let y = from_i64(&[2, 3, 4, 6, 7, 8, 9]);
    let fit = ols(&y, &x, false).unwrap();
    assert!(fit.has_constant());
    assert_eq!(fit.coefficients, vec![qi(3), q(15, 2)]);
    assert_eq!(fit.tss, q(292, 7));
    assert_eq!(fit.r_squared, q(243, 292));
    assert_eq!(fit.df_model, 1);
    assert_eq!(fit.f_statistic().unwrap(), q(1215, 49));
    close(
        fit.f_test(&ctx).unwrap().p_value_f64().unwrap(),
        0.004_177_335_830_133_132,
        1e-12,
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// WLS (D4)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn wls_coefficients_and_weighted_sums_of_squares_exact() {
    // statsmodels: WLS(y, add_constant(x), weights=w).fit(): params [0.9354838709677429, 1.1290322580645162],
    //   ssr 4.516129032258058, ess 54.71464019851117, rsquared 0.9237536656891496,
    //   rsquared_adj 0.9085043988269795, mse_resid 0.9032258064516115, fvalue 60.57692307692317
    let (_, y) = d1();
    let fit = wls(&y, &col(&[1, 2, 3, 4, 5, 6, 7]), &d4_weights(), true).unwrap();
    assert_eq!(fit.weights(), Some(&d4_weights()[..]));
    assert_eq!(fit.coefficients, vec![q(29, 31), q(35, 31)]);
    assert_eq!(
        fit.fitted,
        qs(&[
            (64, 31),
            (99, 31),
            (134, 31),
            (169, 31),
            (204, 31),
            (239, 31),
            (274, 31)
        ])
    );
    assert_eq!(
        fit.residuals,
        qs(&[
            (-2, 31),
            (-6, 31),
            (21, 31),
            (-45, 31),
            (-18, 31),
            (9, 31),
            (5, 31)
        ])
    );
    assert_eq!(fit.ssr, q(140, 31));
    assert_eq!(fit.tss, q(770, 13));
    assert_eq!(fit.ess, q(22050, 403));
    assert_eq!(fit.r_squared, q(315, 341));
    assert_eq!(fit.adjusted_r_squared, q(1549, 1705));
    assert_eq!(fit.mse_resid, q(28, 31));
    assert_eq!(fit.f_statistic().unwrap(), q(1575, 26));
    assert_eq!((fit.df_model, fit.df_resid), (1, 5));
}

#[test]
fn wls_covariance_and_inference() {
    // statsmodels: bse [0.647486848056972, 0.14506169422830145], tvalues [1.4447920815303266, 7.7831178249415665],
    //   pvalues [0.20813207299978637, 0.00056057568027843], f_pvalue 0.0005605756802784246
    let ctx = Context::new();
    let (_, y) = d1();
    let fit = wls(&y, &col(&[1, 2, 3, 4, 5, 6, 7]), &d4_weights(), true).unwrap();
    assert_eq!(
        fit.normalized_cov_params().to_rows(),
        vec![qs(&[(259, 558), (-53, 558)]), qs(&[(-53, 558), (13, 558)])]
    );
    assert_eq!(
        fit.cov_params.to_rows(),
        vec![
            qs(&[(3626, 8649), (-742, 8649)]),
            qs(&[(-742, 8649), (182, 8649)])
        ]
    );
    let se = fit.standard_errors(&ctx);
    close(ev(&se[0]), 0.647_486_848_056_972, 1e-12);
    close(ev(&se[1]), 0.145_061_694_228_301_45, 1e-12);
    let t = fit.t_statistics(&ctx).unwrap();
    close(ev(&t[0]), 1.444_792_081_530_326_6, 1e-12);
    close(ev(&t[1]), 7.783_117_824_941_566_5, 1e-12);
    let p = fit.p_values(&ctx).unwrap();
    close(ev(&p[0]), 0.208_132_072_999_786_37, 1e-12);
    close(ev(&p[1]), 0.000_560_575_680_278_43, 1e-12);
    close(
        fit.f_test(&ctx).unwrap().p_value_f64().unwrap(),
        0.000_560_575_680_278_424_6,
        1e-12,
    );
    // √(28/31) = 0.9503819266229829
    close(
        ev(&fit.residual_standard_error(&ctx)),
        0.950_381_926_622_982_9,
        1e-12,
    );
}

#[test]
fn wls_log_likelihood_information_criteria_and_conf_int() {
    // statsmodels: llf -6.606918004945605, aic 17.21383600989121, bic 17.105656308001837,
    //   conf_int(0.05) [[-0.7289340594609197, 2.5999018013964053], [0.7561393018346153, 1.5019252142944173]]
    let ctx = Context::new();
    let (_, y) = d1();
    let fit = wls(&y, &col(&[1, 2, 3, 4, 5, 6, 7]), &d4_weights(), true).unwrap();
    close(
        ev(&fit.log_likelihood(&ctx).unwrap()),
        -6.606_918_004_945_605,
        1e-12,
    );
    close(ev(&fit.aic(&ctx).unwrap()), 17.213_836_009_891_21, 1e-12);
    close(ev(&fit.bic(&ctx).unwrap()), 17.105_656_308_001_837, 1e-12);
    let ci = fit.conf_int(&ctx, 0.95).unwrap();
    close(ci[0].0, -0.728_934_059_460_919_7, 1e-9);
    close(ci[0].1, 2.599_901_801_396_405_3, 1e-9);
    close(ci[1].0, 0.756_139_301_834_615_3, 1e-9);
    close(ci[1].1, 1.501_925_214_294_417_3, 1e-9);
}

#[test]
fn wls_leverage_cooks_prediction_and_durbin_watson() {
    // statsmodels (OLS on the √w-scaled data, the WLS hat matrix): hat_matrix_diag
    //   [0.29749103942652333, 0.35483870967742004, 0.3118279569892477, 0.07706093189964164,
    //    0.1935483870967742, 0.48924731182795694, 0.2759856630824372]
    //   cooks_distance[0] [0.00138893020765161, 0.0353571428571433, 0.501800537109374, 0.10552697844148543,
    //    0.11108571428571458, 0.262520775623265, 0.00758197725713156]
    // WLS get_prediction([1, 8]).summary_frame(): mean 9.967741935483872,
    //   mean_ci (8.355554097987461, 11.579929772980282), obs_ci (7.040701232480428, 12.894782638487316)
    // durbin_watson(resid) = 2.23944141689373
    let ctx = Context::new();
    let (_, y) = d1();
    let fit = wls(&y, &col(&[1, 2, 3, 4, 5, 6, 7]), &d4_weights(), true).unwrap();
    let lev = fit.leverage();
    assert_eq!(
        lev,
        qs(&[
            (83, 279),
            (11, 31),
            (29, 93),
            (43, 558),
            (6, 31),
            (91, 186),
            (77, 279)
        ])
    );
    assert_eq!(lev.iter().fold(Q::zero(), |a, b| a + b), qi(2));
    close(data::to_f64(&lev)[5], 0.489_247_311_827_956_94, 1e-15);
    assert_eq!(
        fit.cooks_distance().unwrap(),
        qs(&[
            (747, 537824),
            (99, 2800),
            (16443, 32768),
            (31347, 297052),
            (486, 4375),
            (9477, 36100),
            (2475, 326432)
        ])
    );
    // ŷ = H y also for the weighted projection X(XᵀWX)⁻¹XᵀW.
    let h = fit.hat_matrix().unwrap();
    assert_eq!(h.diagonal(), lev);
    let ycol = QMatrix::new(y.iter().map(|v| vec![v.clone()]).collect()).unwrap();
    assert_eq!(h.matmul(&ycol).unwrap().col(0), fit.fitted);
    assert_eq!(h.matmul(&h).unwrap(), h);

    assert_eq!(fit.predict(&[qi(8)]).unwrap(), q(309, 31));
    let (lo, hi) = fit
        .confidence_interval_mean_response(&ctx, &[qi(8)], 0.95)
        .unwrap();
    close(lo, 8.355_554_097_987_461, 1e-9);
    close(hi, 11.579_929_772_980_282, 1e-9);
    let (lo, hi) = fit.prediction_interval(&ctx, &[qi(8)], 0.95).unwrap();
    close(lo, 7.040_701_232_480_428, 1e-9);
    close(hi, 12.894_782_638_487_316, 1e-9);
    assert_eq!(fit.durbin_watson().unwrap(), q(6575, 2936));
    close(
        data::to_f64(&[q(6575, 2936)])[0],
        2.239_441_416_893_73,
        1e-14,
    );
}

#[test]
fn wls_with_unit_weights_equals_ols_and_rejects_bad_weights() {
    let (x, y) = d2();
    let ones = vec![Q::one(); 8];
    let w = wls(&y, &x, &ones, true).unwrap();
    let o = ols(&y, &x, true).unwrap();
    assert_eq!(w.coefficients, o.coefficients);
    assert_eq!(w.cov_params, o.cov_params);
    assert_eq!(w.r_squared, o.r_squared);
    assert!(matches!(
        wls(&y, &x, &ones[..7], true),
        Err(SymplexError::InvalidArgument { .. })
    ));
    let mut zero = ones.clone();
    zero[3] = Q::zero();
    assert!(matches!(
        wls(&y, &x, &zero, true),
        Err(SymplexError::InvalidArgument { .. })
    ));
    let mut neg = ones;
    neg[0] = qi(-1);
    assert!(wls(&y, &x, &neg, true).is_err());
    // The builder's weighted fit agrees.
    let x1: Vec<Q> = x.iter().map(|r| r[0].clone()).collect();
    let x2: Vec<Q> = x.iter().map(|r| r[1].clone()).collect();
    let wts = from_i64(&[1, 2, 1, 2, 1, 2, 1, 2]);
    assert_eq!(
        Design::new()
            .intercept()
            .column(&x1)
            .column(&x2)
            .fit_weighted(&y, &wts)
            .unwrap(),
        wls(&y, &x, &wts, true).unwrap()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// polyfit (D5) and VIF (D8)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn polyfit_exact_for_degrees_one_to_three() {
    // numpy: polyfit(x, y, 1) = [5.742857142857141, -2.8571428571428616]
    //        polyfit(x, y, 2) = [1.1785714285714293, -0.15000000000000532, 1.0714285714285816]
    //        polyfit(x, y, 3) = [0.05555555555555559, 0.7619047619047639, 0.6111111111111055, 0.90476190476191]
    let x = from_i64(&[0, 1, 2, 3, 4, 5]);
    let y = from_i64(&[1, 2, 6, 11, 19, 30]);
    assert_eq!(polyfit(&x, &y, 1).unwrap(), vec![q(201, 35), q(-20, 7)]);
    assert_eq!(
        polyfit(&x, &y, 2).unwrap(),
        vec![q(33, 28), q(-3, 20), q(15, 14)]
    );
    assert_eq!(
        polyfit(&x, &y, 3).unwrap(),
        vec![q(1, 18), q(16, 21), q(11, 18), q(19, 21)]
    );
    let c = polyfit(&x, &y, 2).unwrap();
    close(data::to_f64(&c)[0], 1.178_571_428_571_429_3, 1e-14);
}

#[test]
fn polyfit_interpolates_with_degree_plus_one_points_and_validates() {
    // numpy: polyfit([0, 1, 2], [1, 3, 9], 2) = [2.0, ~1e-15, 1.0]
    let x = from_i64(&[0, 1, 2]);
    let y = from_i64(&[1, 3, 9]);
    assert_eq!(polyfit(&x, &y, 2).unwrap(), vec![qi(2), Q::zero(), qi(1)]);
    // Too few points for the degree.
    assert!(matches!(
        polyfit(&x, &y, 3),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // Mismatched lengths.
    assert!(polyfit(&x, &y[..2], 1).is_err());
    // Only two distinct abscissae cannot determine a quadratic.
    let xd = from_i64(&[1, 1, 2, 2]);
    let yd = from_i64(&[1, 2, 3, 4]);
    assert!(matches!(
        polyfit(&xd, &yd, 2),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert_eq!(polyfit(&xd, &yd, 1).unwrap(), vec![qi(2), q(-1, 2)]);
}

#[test]
fn vif_three_regressors_exact() {
    // statsmodels: variance_inflation_factor(add_constant(X), j) for j = 1, 2, 3 =
    //   9.94779351515861, 6.3245263457132825, 13.72351632915837
    let x = rows(&[
        &[1, 5, 1],
        &[2, 3, 4],
        &[3, 8, 2],
        &[4, 1, 6],
        &[5, 7, 3],
        &[6, 2, 8],
        &[7, 9, 5],
        &[8, 4, 7],
    ]);
    let v = vif(&x).unwrap();
    assert_eq!(v, vec![q(84984, 8543), q(378213, 59801), q(117240, 8543)]);
    let f = data::to_f64(&v);
    close(f[0], 9.947_793_515_158_61, 1e-13);
    close(f[1], 6.324_526_345_713_282_5, 1e-13);
    close(f[2], 13.723_516_329_158_37, 1e-13);
    // A single regressor has VIF 1; an exact duplicate column is rejected.
    assert_eq!(vif(&col(&[1, 2, 3, 5])).unwrap(), vec![Q::one()]);
    let dup = rows(&[&[1, 2], &[2, 4], &[3, 6], &[5, 10]]);
    assert!(matches!(
        vif(&dup),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(vif(&[]).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// OLS — validation and degenerate fits
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ols_rejects_bad_shapes() {
    let (x, y) = d2();
    assert!(matches!(
        ols(&[], &[], true),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // Row count mismatch.
    assert!(matches!(
        ols(&y[..7], &x, true),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // Ragged rows.
    let mut ragged = x.clone();
    ragged[3].push(qi(1));
    assert!(matches!(
        ols(&y, &ragged, true),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // No columns at all.
    let empty_rows: Vec<Vec<Q>> = vec![vec![]; 8];
    assert!(matches!(
        ols(&y, &empty_rows, false),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // Intercept-only through empty rows is fine.
    assert_eq!(
        ols(&y, &empty_rows, true).unwrap().coefficients,
        vec![q(17, 2)]
    );
}

#[test]
fn ols_rejects_n_le_p_and_rank_deficiency() {
    // n = p = 3.
    let (x, y) = d2();
    assert!(matches!(
        ols(&y[..3], &x[..3], true),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // Duplicate column → singular XᵀX.
    let dup: Vec<Vec<Q>> = x
        .iter()
        .map(|r| vec![r[0].clone(), r[0].clone() * qi(2)])
        .collect();
    match ols(&y, &dup, true) {
        Err(SymplexError::InvalidArgument { reason, .. }) => {
            assert!(reason.contains("rank deficient"), "{reason}");
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
    // Constant regressor plus intercept is also collinear.
    let konst: Vec<Vec<Q>> = (0..8).map(|_| vec![qi(3)]).collect();
    assert!(ols(&y, &konst, true).is_err());
    // Constant response.
    let flat = vec![qi(4); 8];
    assert!(matches!(
        ols(&flat, &x, true),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

#[test]
fn ols_perfect_fit_has_zero_residual_variance() {
    // y = 2x + 1 exactly: statsmodels ssr ≈ 6e-29, rsquared 1.0, bse ≈ 1e-15 (all noise).
    let ctx = Context::new();
    let x = col(&[1, 2, 3, 4, 5]);
    let y = from_i64(&[3, 5, 7, 9, 11]);
    let fit = ols(&y, &x, true).unwrap();
    assert_eq!(fit.coefficients, vec![qi(1), qi(2)]);
    assert!(fit.ssr.is_zero());
    assert_eq!(fit.r_squared, Q::one());
    assert_eq!(fit.adjusted_r_squared, Q::one());
    assert!(fit.mse_resid.is_zero());
    assert!(fit.cov_params.is_zero());
    assert_eq!(fit.standard_errors(&ctx), vec![ctx.zero(), ctx.zero()]);
    assert!(matches!(
        fit.t_statistics(&ctx),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(fit.p_values(&ctx).is_err());
    assert!(fit.f_statistic().is_err());
    assert!(fit.cooks_distance().is_err());
    assert!(fit.durbin_watson().is_err());
    assert!(fit.log_likelihood(&ctx).is_err());
    assert!(fit.aic(&ctx).is_err());
    // Leverage and prediction remain exact.
    assert_eq!(fit.leverage().iter().fold(Q::zero(), |a, b| a + b), qi(2));
    assert_eq!(fit.predict(&[qi(10)]).unwrap(), qi(21));
}

// ═══════════════════════════════════════════════════════════════════════════
// Logistic regression
// ═══════════════════════════════════════════════════════════════════════════

// L1: two regressors, n = 16
fn l1() -> (Vec<u8>, Vec<Vec<f64>>) {
    let x = vec![
        vec![2.0, 1.0],
        vec![3.0, 4.0],
        vec![1.0, 2.0],
        vec![5.0, 3.0],
        vec![4.0, 6.0],
        vec![6.0, 2.0],
        vec![2.5, 5.0],
        vec![7.0, 4.0],
        vec![3.5, 1.5],
        vec![1.5, 6.5],
        vec![5.5, 5.5],
        vec![4.5, 2.5],
        vec![6.5, 6.0],
        vec![0.5, 3.0],
        vec![3.0, 3.0],
        vec![5.0, 1.0],
    ];
    let y = vec![0u8, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1];
    (y, x)
}

// L2: binary regressor, ten controls (3 successes) and ten treated (7 successes)
fn l2() -> (Vec<bool>, Vec<Vec<f64>>) {
    let y: Vec<bool> = (0..20).map(|i| matches!(i, 7..=9 | 13..=19)).collect();
    let x: Vec<Vec<f64>> = (0..20)
        .map(|i| vec![if i < 10 { 0.0 } else { 1.0 }])
        .collect();
    (y, x)
}

// L3: near-separated single regressor that statsmodels still fits
fn l3() -> (Vec<u8>, Vec<Vec<f64>>) {
    let x: Vec<Vec<f64>> = (1..=12).map(|i| vec![f64::from(i)]).collect();
    let y = vec![0u8, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1];
    (y, x)
}

#[test]
fn logit_two_regressors_matches_statsmodels() {
    // statsmodels: Logit(y, add_constant(X)).fit(): params [-8.542419056821446, 1.857755547762101, 0.449996209830459]
    //   bse [4.782605093317698, 0.9249008713975254, 0.5808415973556448]
    //   tvalues [-1.786143511777085, 2.008599629660887, 0.7747313757814935]
    //   pvalues [0.07407602453438336, 0.04457961063080209, 0.4384984068938963]
    //   llf -4.435039672866281, llnull -11.090354888959125, prsquared 0.6000993911131254
    //   llr 13.310630432185688, aic 14.870079345732561, bic 17.187845512451904, df_model 2
    let (y, x) = l1();
    let fit = logit(&y, &x, true, &LogitOpts::default()).unwrap();
    assert!(fit.converged);
    assert!(fit.iterations <= 20);
    assert_eq!((fit.nobs, fit.df_model, fit.df_resid), (16, 2, 13));
    close(fit.coefficients[0], -8.542_419_056_821_446, 1e-6);
    close(fit.coefficients[1], 1.857_755_547_762_101, 1e-6);
    close(fit.coefficients[2], 0.449_996_209_830_459, 1e-6);
    close(fit.standard_errors[0], 4.782_605_093_317_698, 1e-6);
    close(fit.standard_errors[1], 0.924_900_871_397_525_4, 1e-6);
    close(fit.standard_errors[2], 0.580_841_597_355_644_8, 1e-6);
    close(fit.z_values[0], -1.786_143_511_777_085, 1e-6);
    close(fit.z_values[1], 2.008_599_629_660_887, 1e-6);
    close(fit.z_values[2], 0.774_731_375_781_493_5, 1e-6);
    close(fit.p_values[0], 0.074_076_024_534_383_36, 1e-6);
    close(fit.p_values[1], 0.044_579_610_630_802_09, 1e-6);
    close(fit.p_values[2], 0.438_498_406_893_896_3, 1e-6);
    close(fit.log_likelihood, -4.435_039_672_866_281, 1e-6);
    close(fit.null_log_likelihood, -11.090_354_888_959_125, 1e-6);
    close(fit.pseudo_r_squared, 0.600_099_391_113_125_4, 1e-6);
    close(fit.llr(), 13.310_630_432_185_688, 1e-6);
    close(fit.aic(), 14.870_079_345_732_561, 1e-6);
    close(fit.bic(), 17.187_845_512_451_904, 1e-6);
}

#[test]
fn logit_two_regressors_predictions_deviance_and_covariance() {
    // statsmodels: predict()[:5] = [0.01240820194395873, 0.23700548210473507, 0.00306489890224918,
    //   0.8905476495757533, 0.8304162805865373]; predict([1, 4, 4]) = 0.6656527408597888;
    //   predict([1, 1, 1]) = 0.0019564461082377882; GLM(Binomial).deviance = 8.870079345732563
    let (y, x) = l1();
    let fit = logit(&y, &x, true, &LogitOpts::default()).unwrap();
    let expected = [
        0.012_408_201_943_958_73,
        0.237_005_482_104_735_07,
        0.003_064_898_902_249_18,
        0.890_547_649_575_753_3,
        0.830_416_280_586_537_3,
    ];
    for (p, e) in fit.fitted_probabilities.iter().zip(expected) {
        close(*p, e, 1e-6);
    }
    close(
        fit.predict_proba(&[4.0, 4.0]).unwrap(),
        0.665_652_740_859_788_8,
        1e-6,
    );
    close(
        fit.predict_proba(&[1.0, 1.0]).unwrap(),
        0.001_956_446_108_237_788_2,
        1e-6,
    );
    close(
        fit.predict_log_odds(&[4.0, 4.0]).unwrap(),
        (0.665_652_740_859_788_8_f64 / (1.0 - 0.665_652_740_859_788_8)).ln(),
        1e-6,
    );
    close(fit.deviance, 8.870_079_345_732_563, 1e-6);
    close(fit.deviance, -2.0 * fit.log_likelihood, 1e-12);
    // cov_params diagonal is bse².
    for j in 0..3 {
        close(fit.cov_params[j][j], fit.standard_errors[j].powi(2), 1e-9);
    }
    assert!(fit.predict_proba(&[4.0]).is_err());
    assert!(fit.predict_proba(&[4.0, f64::NAN]).is_err());
}

#[test]
fn logit_conf_int_and_odds_ratios() {
    // statsmodels: conf_int(0.05) = [[-17.91615279200196, 0.8313146783590657],
    //   [0.04498315055323898, 3.670527944970963], [-0.6884324017093202, 1.5884248213702383]]
    //   np.exp(params) = [1.9501792962715631e-04, 6.4093351689570266, 1.5683062413323574]
    let (y, x) = l1();
    let fit = logit(&y, &x, true, &LogitOpts::default()).unwrap();
    let ci = fit.conf_int(0.95).unwrap();
    close(ci[0].0, -17.916_152_792_001_96, 1e-6);
    close(ci[0].1, 0.831_314_678_359_065_7, 1e-6);
    close(ci[1].0, 0.044_983_150_553_238_98, 1e-6);
    close(ci[1].1, 3.670_527_944_970_963, 1e-6);
    close(ci[2].0, -0.688_432_401_709_320_2, 1e-6);
    close(ci[2].1, 1.588_424_821_370_238_3, 1e-6);
    let or = fit.odds_ratios();
    close(or[0], 1.950_179_296_271_563e-4, 1e-10);
    close(or[1], 6.409_335_168_957_027, 1e-6);
    close(or[2], 1.568_306_241_332_357_4, 1e-6);
    assert!(fit.conf_int(0.0).is_err());
    assert!(fit.conf_int(1.0).is_err());
}

#[test]
fn logit_binary_regressor_has_closed_form() {
    // statsmodels: params [-0.8472978603872037, 1.6945957207744073] (= ln(3/7), ln(49/9)),
    //   bse [0.6900655593423543, 0.9759000729485332], tvalues [-1.2278512511111188, 1.736443891898117],
    //   pvalues [0.21950281228300073, 0.08248537711586468], llf -12.217286041097868,
    //   llnull -13.862943611198906, prsquared 0.11870910076930752, llr 3.2913151402020766,
    //   aic 28.434572082195736, bic 30.426036629303717,
    //   conf_int(0.05) [[-2.1998015036697054, 0.505205782895298], [-0.2181332747147291, 3.607324716263544]]
    //   predict: 0.3 for controls, 0.7 for treated; exp(params) = [0.42857142857142855, 5.444444444444445]
    let (y, x) = l2();
    let fit = logit(&y, &x, true, &LogitOpts::default()).unwrap();
    assert!(fit.converged);
    close(fit.coefficients[0], (3.0_f64 / 7.0).ln(), 1e-9);
    close(fit.coefficients[1], (49.0_f64 / 9.0).ln(), 1e-9);
    close(fit.standard_errors[0], 0.690_065_559_342_354_3, 1e-6);
    close(fit.standard_errors[1], 0.975_900_072_948_533_2, 1e-6);
    close(fit.z_values[0], -1.227_851_251_111_118_8, 1e-6);
    close(fit.z_values[1], 1.736_443_891_898_117, 1e-6);
    close(fit.p_values[0], 0.219_502_812_283_000_73, 1e-6);
    close(fit.p_values[1], 0.082_485_377_115_864_68, 1e-6);
    close(fit.log_likelihood, -12.217_286_041_097_868, 1e-6);
    close(fit.null_log_likelihood, -13.862_943_611_198_906, 1e-6);
    close(fit.null_log_likelihood, 20.0 * 0.5_f64.ln(), 1e-12);
    close(fit.pseudo_r_squared, 0.118_709_100_769_307_52, 1e-6);
    close(fit.llr(), 3.291_315_140_202_076_6, 1e-6);
    close(fit.aic(), 28.434_572_082_195_736, 1e-6);
    close(fit.bic(), 30.426_036_629_303_717, 1e-6);
    let ci = fit.conf_int(0.95).unwrap();
    close(ci[0].0, -2.199_801_503_669_705_4, 1e-6);
    close(ci[0].1, 0.505_205_782_895_298, 1e-6);
    close(ci[1].0, -0.218_133_274_714_729_1, 1e-6);
    close(ci[1].1, 3.607_324_716_263_544, 1e-6);
    close(fit.predict_proba(&[0.0]).unwrap(), 0.3, 1e-9);
    close(fit.predict_proba(&[1.0]).unwrap(), 0.7, 1e-9);
    let or = fit.odds_ratios();
    close(or[0], 3.0 / 7.0, 1e-9);
    close(or[1], 49.0 / 9.0, 1e-9);
}

#[test]
fn logit_near_separated_data_still_fits() {
    // statsmodels: params [-8.498852464660226, 1.307515763793881], bse [5.526614625575412, 0.8318365844065271],
    //   tvalues [-1.537804431908504, 1.5718421001244214], pvalues [0.12409643964594493, 0.11598717514781305],
    //   llf -2.5105386774674776, llnull -8.317766166719343, prsquared 0.698171525004811,
    //   aic 9.021077354934956, bic 9.990890654510956, predict()[:5] = [0.00075251509687898,
    //   0.00277639710970661, 0.01018799289779641, 0.03665755482849433, 0.12332927597916724], predict([1, 6.5]) = 0.5
    let (y, x) = l3();
    let fit = logit(&y, &x, true, &LogitOpts::default()).unwrap();
    assert!(fit.converged);
    close(fit.coefficients[0], -8.498_852_464_660_226, 1e-6);
    close(fit.coefficients[1], 1.307_515_763_793_881, 1e-6);
    close(fit.standard_errors[0], 5.526_614_625_575_412, 1e-6);
    close(fit.standard_errors[1], 0.831_836_584_406_527_1, 1e-6);
    close(fit.z_values[0], -1.537_804_431_908_504, 1e-6);
    close(fit.z_values[1], 1.571_842_100_124_421_4, 1e-6);
    close(fit.p_values[0], 0.124_096_439_645_944_93, 1e-6);
    close(fit.p_values[1], 0.115_987_175_147_813_05, 1e-6);
    close(fit.log_likelihood, -2.510_538_677_467_477_6, 1e-6);
    close(fit.null_log_likelihood, -8.317_766_166_719_343, 1e-6);
    close(fit.pseudo_r_squared, 0.698_171_525_004_811, 1e-6);
    close(fit.aic(), 9.021_077_354_934_956, 1e-6);
    close(fit.bic(), 9.990_890_654_510_956, 1e-6);
    let expected = [
        0.000_752_515_096_878_98,
        0.002_776_397_109_706_61,
        0.010_187_992_897_796_41,
        0.036_657_554_828_494_33,
        0.123_329_275_979_167_24,
    ];
    for (p, e) in fit.fitted_probabilities.iter().zip(expected) {
        close(*p, e, 1e-6);
    }
    close(fit.predict_proba(&[6.5]).unwrap(), 0.5, 1e-6);
}

#[test]
fn logit_without_intercept_matches_statsmodels() {
    // statsmodels: Logit(y, X_centered).fit() (no constant): params [1.7773508899229844, 0.3895430068514461],
    //   bse [0.8808405476526124, 0.5265308077380182], tvalues [2.0177895927469716, 0.7398294670067397],
    //   pvalues [0.04361317923130994, 0.45940347662375014], llf -4.5724002646286985,
    //   llnull -11.090354888959125, prsquared 0.5877138008287995, df_model 1, df_resid 14,
    //   aic 13.144800529257397, bic 14.689977973736958, predict([0, 0]) = 0.5, predict([1, -1]) = 0.8002420535600436,
    //   conf_int(0.10) [[0.3284971203506639, 3.2262046594953047], [-0.47652310195812125, 1.2556091156610134]]
    let (y, x) = l1();
    let centered: Vec<Vec<f64>> = x.iter().map(|r| vec![r[0] - 4.0, r[1] - 3.5]).collect();
    let fit = logit(&y, &centered, false, &LogitOpts::default()).unwrap();
    assert!(fit.converged);
    assert_eq!(fit.n_params(), 2);
    assert_eq!((fit.df_model, fit.df_resid), (1, 14));
    close(fit.coefficients[0], 1.777_350_889_922_984_4, 1e-6);
    close(fit.coefficients[1], 0.389_543_006_851_446_1, 1e-6);
    close(fit.standard_errors[0], 0.880_840_547_652_612_4, 1e-6);
    close(fit.standard_errors[1], 0.526_530_807_738_018_2, 1e-6);
    close(fit.z_values[0], 2.017_789_592_746_971_6, 1e-6);
    close(fit.z_values[1], 0.739_829_467_006_739_7, 1e-6);
    close(fit.p_values[0], 0.043_613_179_231_309_94, 1e-6);
    close(fit.p_values[1], 0.459_403_476_623_750_14, 1e-6);
    close(fit.log_likelihood, -4.572_400_264_628_698_5, 1e-6);
    close(fit.null_log_likelihood, -11.090_354_888_959_125, 1e-6);
    close(fit.pseudo_r_squared, 0.587_713_800_828_799_5, 1e-6);
    close(fit.aic(), 13.144_800_529_257_397, 1e-6);
    close(fit.bic(), 14.689_977_973_736_958, 1e-6);
    close(fit.predict_proba(&[0.0, 0.0]).unwrap(), 0.5, 1e-12);
    close(
        fit.predict_proba(&[1.0, -1.0]).unwrap(),
        0.800_242_053_560_043_6,
        1e-6,
    );
    let ci = fit.conf_int(0.90).unwrap();
    close(ci[0].0, 0.328_497_120_350_663_9, 1e-6);
    close(ci[0].1, 3.226_204_659_495_304_7, 1e-6);
    close(ci[1].0, -0.476_523_101_958_121_25, 1e-6);
    close(ci[1].1, 1.255_609_115_661_013_4, 1e-6);
}

#[test]
fn logit_accepts_bool_u8_i64_and_f64_outcomes() {
    let (y_u8, x) = l1();
    let y_bool: Vec<bool> = y_u8.iter().map(|&v| v == 1).collect();
    let y_i64: Vec<i64> = y_u8.iter().map(|&v| i64::from(v)).collect();
    let y_f64: Vec<f64> = y_u8.iter().map(|&v| f64::from(v)).collect();
    let opts = LogitOpts::default();
    let a = logit(&y_u8, &x, true, &opts).unwrap();
    assert_eq!(logit(&y_bool, &x, true, &opts).unwrap(), a);
    assert_eq!(logit(&y_i64, &x, true, &opts).unwrap(), a);
    assert_eq!(logit(&y_f64, &x, true, &opts).unwrap(), a);
    // Non-binary codes are rejected.
    let mut bad = y_u8.clone();
    bad[0] = 2;
    assert!(matches!(
        logit(&bad, &x, true, &opts),
        Err(SymplexError::InvalidArgument { .. })
    ));
    let mut badf = y_f64.clone();
    badf[0] = 0.5;
    assert!(logit(&badf, &x, true, &opts).is_err());
}

#[test]
fn logit_detects_complete_separation() {
    // statsmodels: Logit(y, add_constant(x)).fit() → PerfectSeparationWarning + ConvergenceWarning,
    //   params [-193.95903897789725, 43.125626614294404], llf ≈ -8.7e-10 (no finite MLE).
    let x: Vec<Vec<f64>> = (1..=8).map(|i| vec![f64::from(i)]).collect();
    let y = vec![0u8, 0, 0, 0, 1, 1, 1, 1];
    match logit(&y, &x, true, &LogitOpts::default()) {
        Err(SymplexError::ComputationFailed { reason, .. }) => {
            assert!(reason.contains("separation"), "{reason}");
        }
        other => panic!("expected ComputationFailed, got {other:?}"),
    }
    // Still detected with a generous iteration budget.
    let opts = LogitOpts {
        max_iter: 2000,
        tol: 1e-12,
    };
    assert!(matches!(
        logit(&y, &x, true, &opts),
        Err(SymplexError::ComputationFailed { .. })
    ));
}

#[test]
fn logit_detects_quasi_complete_separation() {
    // statsmodels: Logit(y, add_constant(dummy)).fit(maxiter=100) → ConvergenceWarning, converged False,
    //   params [-0.5108256237562828, 27.031924507106936] (the slope diverges; every treated unit is a success).
    let x: Vec<Vec<f64>> = (0..14)
        .map(|i| vec![if i < 8 { 0.0 } else { 1.0 }])
        .collect();
    let y = vec![0u8, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    match logit(&y, &x, true, &LogitOpts::default()) {
        Err(SymplexError::ComputationFailed { reason, .. }) => {
            assert!(reason.contains("separation"), "{reason}");
        }
        other => panic!("expected ComputationFailed, got {other:?}"),
    }
    assert!(matches!(
        logit(
            &y,
            &x,
            true,
            &LogitOpts {
                max_iter: 5000,
                tol: 1e-10
            }
        ),
        Err(SymplexError::ComputationFailed { .. })
    ));
}

#[test]
fn logit_rejects_invalid_input() {
    let (y, x) = l1();
    let opts = LogitOpts::default();
    // Empty, constant, mismatched, ragged, non-finite, n ≤ p, collinear, bad options.
    assert!(matches!(
        logit::<u8>(&[], &[], true, &opts),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        logit(&[1u8; 16], &x, true, &opts),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        logit(&y[..15], &x, true, &opts),
        Err(SymplexError::InvalidArgument { .. })
    ));
    let mut ragged = x.clone();
    ragged[2].push(1.0);
    assert!(logit(&y, &ragged, true, &opts).is_err());
    let mut nan = x.clone();
    nan[2][0] = f64::NAN;
    assert!(logit(&y, &nan, true, &opts).is_err());
    assert!(matches!(
        logit(&y[..3], &x[..3], true, &opts),
        Err(SymplexError::InvalidArgument { .. })
    ));
    let dup: Vec<Vec<f64>> = x.iter().map(|r| vec![r[0], 2.0 * r[0]]).collect();
    match logit(&y, &dup, true, &opts) {
        Err(SymplexError::InvalidArgument { reason, .. }) => {
            assert!(reason.contains("rank deficient"), "{reason}");
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
    let empty_rows: Vec<Vec<f64>> = vec![vec![]; 16];
    assert!(logit(&y, &empty_rows, false, &opts).is_err());
    assert!(
        logit(
            &y,
            &x,
            true,
            &LogitOpts {
                max_iter: 0,
                tol: 1e-8
            }
        )
        .is_err()
    );
    assert!(
        logit(
            &y,
            &x,
            true,
            &LogitOpts {
                max_iter: 10,
                tol: 0.0
            }
        )
        .is_err()
    );
}

#[test]
fn logit_reports_non_convergence_when_iteration_budget_is_tiny() {
    // One Newton step from β = 0 is not the optimum for L1 (statsmodels needs 8).
    let (y, x) = l1();
    let fit = logit(
        &y,
        &x,
        true,
        &LogitOpts {
            max_iter: 1,
            tol: 1e-10,
        },
    )
    .unwrap();
    assert!(!fit.converged);
    assert_eq!(fit.iterations, 1);
    // The full fit has a strictly higher log-likelihood.
    let full = logit(&y, &x, true, &LogitOpts::default()).unwrap();
    assert!(full.log_likelihood > fit.log_likelihood);
}
