//! symplex 0.20 — multinomial (`mnlogit`) and ordinal (`ologit`) logistic
//! regression.  Reference values cite statsmodels 0.15 / scipy 1.18
//! (`symplex/.venv/bin/python`):
//!
//! * `sm.MNLogit(y, X).fit(method='newton', maxiter=200, tol=1e-12)` —
//!   `params.T`, `bse.T`, `tvalues.T`, `pvalues.T`, `llf`, `llnull`, `llr`,
//!   `llr_pvalue`, `prsquared`, `aic`, `bic`, `cov_params()`, `predict`,
//!   `conf_int(0.05)`; reference category 0, per-category rows.
//! * `OrderedModel(y, X, distr='logit').fit(method='newton', maxiter=200,
//!   tol=1e-12)` — `params` are `(β, θ₀, ln(θ₁ − θ₀), …)`;
//!   `transform_threshold_params(params)[1:-1]` gives the thresholds the
//!   crate reports.  Its `bse` come from a numerically differentiated
//!   Hessian, so they carry ~1e-6 relative noise; the delta method
//!   `J cov J.T` with `∂θ_j/∂α_m = exp(α_m)` (`1 ≤ m ≤ j`) moves them to the
//!   natural thresholds and is compared at 1e-5.  Exact standard errors
//!   (a 40-digit mpmath Hessian at the optimum) are cited where used.
//! * statsmodels' `llnull` is itself a Newton fit (≈ 3e-9 off the closed
//!   form `Σ n_j ln(n_j/n)`); the closed form is cited from mpmath.

use symplex::prelude::*;
use symplex::stats::hypothesis::Alternative;
use symplex::stats::regression::{LogitOpts, logit, mnlogit, ologit};

fn close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() < tol,
        "got {actual}, expected {expected} (tol {tol})"
    );
}

/// Relative comparison at `rel` (absolute below `1e-12`).
fn rel(actual: f64, expected: f64, rel: f64) {
    let tol = (rel * expected.abs()).max(1e-12);
    close(actual, expected, tol);
}

// M1: k = 3, x = 1..24 (one regressor + intercept)
fn m1() -> (Vec<usize>, Vec<Vec<f64>>) {
    let y = vec![
        0, 0, 1, 0, 0, 1, 2, 0, 1, 1, 2, 0, 1, 2, 1, 2, 1, 2, 2, 1, 2, 0, 2, 2,
    ];
    let x = (1..=24).map(|i| vec![f64::from(i)]).collect();
    (y, x)
}

// M2: k = 4, two regressors (x1 = 1..6 repeated, x2 integers), n = 30
fn m2() -> (Vec<usize>, Vec<Vec<f64>>) {
    let y = vec![
        0, 3, 1, 1, 2, 1, 2, 0, 3, 3, 0, 3, 0, 1, 1, 2, 2, 3, 0, 1, 2, 3, 3, 3, 0, 3, 2, 3, 2, 2,
    ];
    let x2 = [
        3.0, 1.0, 4.0, 1.0, 5.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0, 9.0, 7.0, 9.0, 3.0, 2.0, 3.0, 8.0,
        4.0, 6.0, 2.0, 6.0, 4.0, 3.0, 3.0, 8.0, 3.0, 2.0, 7.0, 9.0,
    ];
    let x = (0..30)
        .map(|i| vec![f64::from(i as u8 % 6 + 1), x2[i]])
        .collect();
    (y, x)
}

// L2 (v14): binary regressor, ten controls with 3 successes, ten treated with 7
fn l2() -> (Vec<usize>, Vec<Vec<f64>>) {
    let y = (0..20)
        .map(|i| usize::from(matches!(i, 7..=9 | 13..=19)))
        .collect();
    let x = (0..20)
        .map(|i| vec![if i < 10 { 0.0 } else { 1.0 }])
        .collect();
    (y, x)
}

// L1 (v14): two regressors, n = 16
fn l1() -> (Vec<usize>, Vec<Vec<f64>>) {
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
    let y = vec![0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1];
    (y, x)
}

// O1: k = 3 ordered, x = 0..19
fn o1() -> (Vec<usize>, Vec<Vec<f64>>) {
    let y = vec![0, 0, 1, 0, 1, 1, 2, 1, 2, 2, 0, 1, 2, 2, 1, 0, 0, 1, 2, 2];
    let x = (0..20).map(|i| vec![f64::from(i)]).collect();
    (y, x)
}

// O2: k = 4 ordered, the M2 design
fn o2() -> (Vec<usize>, Vec<Vec<f64>>) {
    let y = vec![
        1, 0, 2, 2, 2, 3, 2, 1, 1, 2, 3, 3, 2, 2, 3, 1, 3, 3, 1, 0, 0, 1, 3, 0, 1, 2, 1, 0, 1, 2,
    ];
    (y, m2().1)
}

fn scaled(x: &[Vec<f64>], factor: f64) -> Vec<Vec<f64>> {
    x.iter()
        .map(|r| r.iter().map(|v| v * factor).collect())
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// mnlogit
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mnlogit_m1_coefficients_match_statsmodels() {
    // statsmodels: MNLogit(y, add_constant(x)).fit(method='newton'): converged, 6 iterations,
    //   params.T [[-0.9175964754951935, 0.10985702371911016], [-2.875029245270399, 0.2536273262970427]]
    let (y, x) = m1();
    let fit = mnlogit(&y, &x, true, &LogitOpts::default()).unwrap();
    assert!(fit.converged);
    assert!(fit.iterations <= 10);
    assert_eq!((fit.n_categories, fit.n_params, fit.nobs), (3, 2, 24));
    assert_eq!((fit.df_model, fit.df_resid), (2, 20));
    assert_eq!(fit.coefficients.len(), 2);
    rel(fit.coefficients[0][0], -0.917_596_475_495_193_5, 1e-6);
    rel(fit.coefficients[0][1], 0.109_857_023_719_110_16, 1e-6);
    rel(fit.coefficients[1][0], -2.875_029_245_270_399, 1e-6);
    rel(fit.coefficients[1][1], 0.253_627_326_297_042_7, 1e-6);
}

#[test]
fn mnlogit_m1_standard_errors_z_and_p_values() {
    // statsmodels: bse.T [[1.0202923117484362, 0.09311682562597019], [1.4432124419606307, 0.1081682371602044]]
    //   tvalues.T [[-0.8993466528457352, 1.1797762969324326], [-1.99210397698943, 2.3447486337546914]]
    //   pvalues.T [[0.36846804499407193, 0.23808919923338745], [0.04635965070751905, 0.019039911083865633]]
    let (y, x) = m1();
    let fit = mnlogit(&y, &x, true, &LogitOpts::default()).unwrap();
    rel(fit.standard_errors[0][0], 1.020_292_311_748_436_2, 1e-6);
    rel(fit.standard_errors[0][1], 0.093_116_825_625_970_19, 1e-6);
    rel(fit.standard_errors[1][0], 1.443_212_441_960_630_7, 1e-6);
    rel(fit.standard_errors[1][1], 0.108_168_237_160_204_4, 1e-6);
    rel(fit.z_values[0][0], -0.899_346_652_845_735_2, 1e-6);
    rel(fit.z_values[0][1], 1.179_776_296_932_432_6, 1e-6);
    rel(fit.z_values[1][0], -1.992_103_976_989_43, 1e-6);
    rel(fit.z_values[1][1], 2.344_748_633_754_691_4, 1e-6);
    rel(fit.p_values[0][0], 0.368_468_044_994_071_93, 1e-6);
    rel(fit.p_values[0][1], 0.238_089_199_233_387_45, 1e-6);
    rel(fit.p_values[1][0], 0.046_359_650_707_519_05, 1e-6);
    rel(fit.p_values[1][1], 0.019_039_911_083_865_633, 1e-6);
}

#[test]
fn mnlogit_m1_likelihood_summary() {
    // statsmodels: llf -22.117379463121267, llnull -26.241367358680538 (Newton fit; closed form
    //   Σ n_j ln(n_j/n) = -26.24136735549884 by mpmath), llr 8.247975791118542,
    //   prsquared 0.15715598349699078, aic 52.234758926242534, bic 56.946974247634316
    let (y, x) = m1();
    let fit = mnlogit(&y, &x, true, &LogitOpts::default()).unwrap();
    close(fit.log_likelihood, -22.117_379_463_121_267, 1e-8);
    close(fit.null_log_likelihood, -26.241_367_355_498_84, 1e-8);
    close(fit.null_log_likelihood, -26.241_367_358_680_538, 1e-7);
    close(fit.llr(), 8.247_975_791_118_542, 1e-7);
    rel(fit.pseudo_r_squared, 0.157_155_983_496_990_78, 1e-7);
    close(fit.aic(), 52.234_758_926_242_534, 1e-7);
    close(fit.bic(), 56.946_974_247_634_316, 1e-7);
}

#[test]
fn mnlogit_m1_llr_test_is_chi_squared_with_slope_degrees_of_freedom() {
    // statsmodels: llr_pvalue 0.016179862014191373 with df_model 2 = (J − 1)(K − 1);
    //   scipy: chi2.sf(8.24797579111854, 2) = 0.016179862014191387
    let ctx = Context::new();
    let (y, x) = m1();
    let fit = mnlogit(&y, &x, true, &LogitOpts::default()).unwrap();
    let t = fit.llr_test(&ctx).unwrap();
    close(t.statistic_f64().unwrap(), 8.247_975_791_118_542, 1e-7);
    close(t.p_value_f64().unwrap(), 0.016_179_862_014_191_373, 1e-8);
    assert_eq!(t.df, Some(ctx.int(2)));
    assert_eq!(t.alternative, Alternative::Greater);
}

#[test]
fn mnlogit_m1_cov_params_is_flattened_in_statsmodels_order() {
    // statsmodels: cov_params() (Fortran order: category-1 params, then category-2 params)
    //   [[1.0409964014129682, -0.08031900522826174, 0.5913496891773338, -0.05084889610034827],
    //    [-0.08031900522826163, 0.008670743214657339, -0.05409719451709933, 0.006255448171693201],
    //    [0.5913496891773314, -0.054097194517099156, 2.082862152629967, -0.13873191218512748],
    //    [-0.05084889610034803, 0.006255448171693184, -0.1387319121851275, 0.011700367530346227]]
    let (y, x) = m1();
    let fit = mnlogit(&y, &x, true, &LogitOpts::default()).unwrap();
    let expected = [
        [
            1.040_996_401_412_968_2,
            -0.080_319_005_228_261_74,
            0.591_349_689_177_333_8,
            -0.050_848_896_100_348_27,
        ],
        [
            -0.080_319_005_228_261_63,
            0.008_670_743_214_657_339,
            -0.054_097_194_517_099_33,
            0.006_255_448_171_693_201,
        ],
        [
            0.591_349_689_177_331_4,
            -0.054_097_194_517_099_156,
            2.082_862_152_629_967,
            -0.138_731_912_185_127_48,
        ],
        [
            -0.050_848_896_100_348_03,
            0.006_255_448_171_693_184,
            -0.138_731_912_185_127_5,
            0.011_700_367_530_346_227,
        ],
    ];
    assert_eq!(fit.cov_params.len(), 4);
    for (r, row) in expected.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            rel(fit.cov_params[r][c], *v, 1e-6);
        }
    }
    // The diagonal is bse² in the same flat order.
    close(
        fit.cov_params[3][3],
        fit.standard_errors[1][1].powi(2),
        1e-12,
    );
}

#[test]
fn mnlogit_m1_predictions() {
    // statsmodels: predict()[:3] = [[0.6585161258968459, 0.2936091784319376, 0.0478746956712164],
    //   [0.6284063695426623, 0.3127188719947723, 0.05887475846256543], ...];
    //   predict([1, 12]) = [0.27200686543576275, 0.4060657594944313, 0.32192737506980595]
    let (y, x) = m1();
    let fit = mnlogit(&y, &x, true, &LogitOpts::default()).unwrap();
    assert_eq!(fit.fitted_probabilities.len(), 24);
    let expected = [
        [
            0.658_516_125_896_845_9,
            0.293_609_178_431_937_6,
            0.047_874_695_671_216_4,
        ],
        [
            0.628_406_369_542_662_3,
            0.312_718_871_994_772_3,
            0.058_874_758_462_565_43,
        ],
    ];
    for (row, exp) in fit.fitted_probabilities.iter().zip(expected) {
        for (p, e) in row.iter().zip(exp) {
            rel(*p, e, 1e-6);
        }
    }
    let at12 = fit.predict_proba(&[12.0]).unwrap();
    rel(at12[0], 0.272_006_865_435_762_75, 1e-6);
    rel(at12[1], 0.406_065_759_494_431_3, 1e-6);
    rel(at12[2], 0.321_927_375_069_805_95, 1e-6);
    close(at12.iter().sum::<f64>(), 1.0, 1e-12);
    assert_eq!(fit.predict(&[12.0]).unwrap(), 1);
    assert_eq!(fit.predict(&[1.0]).unwrap(), 0);
    assert_eq!(fit.predict(&[24.0]).unwrap(), 2);
    assert!(fit.predict_proba(&[1.0, 2.0]).is_err());
    assert!(fit.predict_proba(&[f64::NAN]).is_err());
    assert!(fit.predict(&[]).is_err());
}

#[test]
fn mnlogit_m1_conf_int_and_relative_risk_ratios() {
    // statsmodels: conf_int(0.05) = [[[-2.9173326602252416, 1.0821397092348544], [-0.07264860086248778, 0.29236264830070813]],
    //   [[-5.703673653553338, -0.04638483698745999], [0.04162147719185494, 0.4656331754022305]]]
    //   np.exp(params).T = [[0.39947804339649723, 1.1161184805809736], [0.05641448962264841, 1.2886914533503968]]
    let (y, x) = m1();
    let fit = mnlogit(&y, &x, true, &LogitOpts::default()).unwrap();
    let ci = fit.conf_int(0.95).unwrap();
    rel(ci[0][0].lower, -2.917_332_660_225_241_6, 1e-6);
    rel(ci[0][0].upper, 1.082_139_709_234_854_4, 1e-6);
    rel(ci[0][1].lower, -0.072_648_600_862_487_78, 1e-6);
    rel(ci[0][1].upper, 0.292_362_648_300_708_13, 1e-6);
    rel(ci[1][0].lower, -5.703_673_653_553_338, 1e-6);
    rel(ci[1][0].upper, -0.046_384_836_987_459_99, 1e-5);
    rel(ci[1][1].lower, 0.041_621_477_191_854_94, 1e-6);
    rel(ci[1][1].upper, 0.465_633_175_402_230_5, 1e-6);
    let rrr = fit.relative_risk_ratios();
    rel(rrr[0][0], 0.399_478_043_396_497_23, 1e-6);
    rel(rrr[0][1], 1.116_118_480_580_973_6, 1e-6);
    rel(rrr[1][0], 0.056_414_489_622_648_41, 1e-6);
    rel(rrr[1][1], 1.288_691_453_350_396_8, 1e-6);
    assert!(fit.conf_int(0.0).is_err());
    assert!(fit.conf_int(1.0).is_err());
}

#[test]
fn mnlogit_rescaling_a_column_scales_its_slopes_inversely() {
    // statsmodels: MNLogit(y, add_constant(x / 10)).fit(): params.T
    //   [[-0.9175964754951933, 1.0985702371911013], [-2.875029245270399, 2.5362732629704268]],
    //   bse.T [[1.0202923117484366, 0.9311682562597019], [1.4432124419606285, 1.0816823716020425]],
    //   llf -22.117379463121267 (unchanged), np.exp(params).T[1][1] = 12.632505113466499
    let (y, x) = m1();
    let fit = mnlogit(&y, &scaled(&x, 0.1), true, &LogitOpts::default()).unwrap();
    let full = mnlogit(&y, &x, true, &LogitOpts::default()).unwrap();
    rel(fit.coefficients[0][0], -0.917_596_475_495_193_3, 1e-6);
    rel(fit.coefficients[0][1], 1.098_570_237_191_101_3, 1e-6);
    rel(fit.coefficients[1][1], 2.536_273_262_970_426_8, 1e-6);
    rel(fit.standard_errors[0][1], 0.931_168_256_259_701_9, 1e-6);
    rel(fit.standard_errors[1][1], 1.081_682_371_602_042_5, 1e-6);
    close(fit.log_likelihood, -22.117_379_463_121_267, 1e-8);
    // Intercepts unchanged, slopes × 10, standard errors of slopes × 10, z and p unchanged.
    for j in 0..2 {
        close(fit.coefficients[j][0], full.coefficients[j][0], 1e-9);
        close(fit.coefficients[j][1], 10.0 * full.coefficients[j][1], 1e-9);
        close(
            fit.standard_errors[j][1],
            10.0 * full.standard_errors[j][1],
            1e-9,
        );
        close(fit.z_values[j][1], full.z_values[j][1], 1e-9);
        close(fit.p_values[j][1], full.p_values[j][1], 1e-9);
    }
    rel(
        fit.relative_risk_ratios()[1][1],
        12.632_505_113_466_499,
        1e-6,
    );
}

#[test]
fn mnlogit_without_intercept_matches_statsmodels() {
    // statsmodels: MNLogit(y, x).fit(method='newton') (no constant): params.T [[0.03517894143505059], [0.0652501991424487]]
    //   bse.T [[0.04388214859884475], [0.04067442828115374]], tvalues.T [[0.8016686182949738], [1.6042069157412595]]
    //   llf -24.819851584803594, aic 53.63970316960719, bic 55.99581083030308, df_model 0 (assumes a constant), llr_pvalue nan
    //   predict()[0] = [0.32224477247865535, 0.333782759659005, 0.34397246786233965]
    let ctx = Context::new();
    let (y, x) = m1();
    let fit = mnlogit(&y, &x, false, &LogitOpts::default()).unwrap();
    assert!(fit.converged);
    assert_eq!((fit.n_params, fit.df_model, fit.df_resid), (1, 0, 22));
    rel(fit.coefficients[0][0], 0.035_178_941_435_050_59, 1e-6);
    rel(fit.coefficients[1][0], 0.065_250_199_142_448_7, 1e-6);
    rel(fit.standard_errors[0][0], 0.043_882_148_598_844_75, 1e-6);
    rel(fit.standard_errors[1][0], 0.040_674_428_281_153_74, 1e-6);
    rel(fit.z_values[1][0], 1.604_206_915_741_259_5, 1e-6);
    close(fit.log_likelihood, -24.819_851_584_803_594, 1e-8);
    close(fit.aic(), 53.639_703_169_607_19, 1e-7);
    close(fit.bic(), 55.995_810_830_303_08, 1e-7);
    let p0 = fit.predict_proba(&[1.0]).unwrap();
    rel(p0[0], 0.322_244_772_478_655_35, 1e-6);
    rel(p0[2], 0.343_972_467_862_339_65, 1e-6);
    // df_model = 0: the LR test against the intercept-only model is undefined (statsmodels: nan).
    assert!(matches!(
        fit.llr_test(&ctx),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

#[test]
fn mnlogit_m2_four_categories_two_regressors() {
    // statsmodels: MNLogit(y, add_constant(X)).fit(method='newton'): 7 iterations, params.T
    //   [[-1.2966049385810385, 0.7943916776437226, -0.1504521237625517],
    //    [-2.1879140115272633, 1.0543715784335537, -0.09875454783618469],
    //    [-2.4065754359152534, 1.1255952242257463, -0.06728861998828524]]
    //   bse.T [[1.8540253561941988, 0.49817635775899954, 0.27427814840442205],
    //    [1.8875319497784881, 0.4991183989620144, 0.2656697711139698],
    //    [1.8757384592070452, 0.49495893954610765, 0.2597301671917575]]
    //   pvalues.T[2] = [0.19949151709939905, 0.0229588695842321, 0.79558026414905]
    let (y, x) = m2();
    let fit = mnlogit(&y, &x, true, &LogitOpts::default()).unwrap();
    assert!(fit.converged);
    assert_eq!((fit.n_categories, fit.n_params, fit.nobs), (4, 3, 30));
    assert_eq!((fit.df_model, fit.df_resid), (6, 21));
    let params = [
        [
            -1.296_604_938_581_038_5,
            0.794_391_677_643_722_6,
            -0.150_452_123_762_551_7,
        ],
        [
            -2.187_914_011_527_263_3,
            1.054_371_578_433_553_7,
            -0.098_754_547_836_184_69,
        ],
        [
            -2.406_575_435_915_253_4,
            1.125_595_224_225_746_3,
            -0.067_288_619_988_285_24,
        ],
    ];
    let bse = [
        [
            1.854_025_356_194_198_8,
            0.498_176_357_758_999_54,
            0.274_278_148_404_422_05,
        ],
        [
            1.887_531_949_778_488_1,
            0.499_118_398_962_014_4,
            0.265_669_771_113_969_8,
        ],
        [
            1.875_738_459_207_045_2,
            0.494_958_939_546_107_65,
            0.259_730_167_191_757_5,
        ],
    ];
    for j in 0..3 {
        for a in 0..3 {
            rel(fit.coefficients[j][a], params[j][a], 1e-6);
            rel(fit.standard_errors[j][a], bse[j][a], 1e-6);
        }
    }
    rel(fit.p_values[2][0], 0.199_491_517_099_399_05, 1e-6);
    rel(fit.p_values[2][1], 0.022_958_869_584_232_1, 1e-6);
    rel(fit.p_values[2][2], 0.795_580_264_149_05, 1e-6);
}

#[test]
fn mnlogit_m2_likelihood_covariance_and_prediction() {
    // statsmodels: llf -35.92131640627783, llnull -40.873424559558686 (closed form -40.873424555748855),
    //   llr 9.904216306561707, llr_pvalue 0.1287440648223197, prsquared 0.12115716279326905,
    //   aic 89.84263281255566, bic 102.45340924751505; cov_params()[0][:3] = [3.437410021411026,
    //   -0.6006461141680979, -0.34995453453426645], cov_params()[4][7] = 0.19648438279548727;
    //   predict([1, 3, 5]) = [0.1694899074768902, 0.23677323237793213, 0.27429971713854884, 0.3194371430066289]
    let ctx = Context::new();
    let (y, x) = m2();
    let fit = mnlogit(&y, &x, true, &LogitOpts::default()).unwrap();
    close(fit.log_likelihood, -35.921_316_406_277_83, 1e-8);
    close(fit.null_log_likelihood, -40.873_424_555_748_855, 1e-8);
    close(fit.llr(), 9.904_216_306_561_707, 1e-7);
    rel(fit.pseudo_r_squared, 0.121_157_162_793_269_05, 1e-7);
    close(fit.aic(), 89.842_632_812_555_66, 1e-7);
    close(fit.bic(), 102.453_409_247_515_05, 1e-7);
    let t = fit.llr_test(&ctx).unwrap();
    assert_eq!(t.df, Some(ctx.int(6)));
    close(t.p_value_f64().unwrap(), 0.128_744_064_822_319_7, 1e-8);
    assert_eq!(fit.cov_params.len(), 9);
    rel(fit.cov_params[0][0], 3.437_410_021_411_026, 1e-6);
    rel(fit.cov_params[0][1], -0.600_646_114_168_097_9, 1e-6);
    rel(fit.cov_params[0][2], -0.349_954_534_534_266_45, 1e-6);
    rel(fit.cov_params[4][7], 0.196_484_382_795_487_27, 1e-6);
    let pr = fit.predict_proba(&[3.0, 5.0]).unwrap();
    rel(pr[0], 0.169_489_907_476_890_2, 1e-6);
    rel(pr[1], 0.236_773_232_377_932_13, 1e-6);
    rel(pr[2], 0.274_299_717_138_548_84, 1e-6);
    rel(pr[3], 0.319_437_143_006_628_9, 1e-6);
    assert_eq!(fit.predict(&[3.0, 5.0]).unwrap(), 3);
}

#[test]
fn mnlogit_with_two_categories_is_logit() {
    // statsmodels: MNLogit(y, add_constant(x)).fit() on L2: params.T [[-0.8472978603872037, 1.6945957207744071]]
    //   (= Logit's [-0.8472978603872037, 1.6945957207744073] = ln(3/7), ln(49/9)), bse.T [[0.6900655593423543, 0.9759000729485332]],
    //   llf -12.21728604109787, cov_params [[0.47619047619047616, -0.47619047619047616], [-0.47619047619047616, 0.9523809523809523]]
    let (y, x) = l2();
    let mn = mnlogit(&y, &x, true, &LogitOpts::default()).unwrap();
    let yb: Vec<u8> = y.iter().map(|&v| v as u8).collect();
    let lg = logit(&yb, &x, true, &LogitOpts::default()).unwrap();
    assert_eq!(mn.coefficients.len(), 1);
    close(mn.coefficients[0][0], (3.0_f64 / 7.0).ln(), 1e-9);
    close(mn.coefficients[0][1], (49.0_f64 / 9.0).ln(), 1e-9);
    for a in 0..2 {
        close(mn.coefficients[0][a], lg.coefficients[a], 1e-9);
        close(mn.standard_errors[0][a], lg.standard_errors[a], 1e-9);
        close(mn.z_values[0][a], lg.z_values[a], 1e-9);
        close(mn.p_values[0][a], lg.p_values[a], 1e-9);
    }
    close(mn.log_likelihood, lg.log_likelihood, 1e-10);
    close(mn.log_likelihood, -12.217_286_041_097_87, 1e-8);
    close(mn.null_log_likelihood, lg.null_log_likelihood, 1e-10);
    close(mn.aic(), lg.aic(), 1e-9);
    close(mn.bic(), lg.bic(), 1e-9);
    close(mn.cov_params[0][0], 10.0 / 21.0, 1e-9);
    close(mn.cov_params[1][1], 20.0 / 21.0, 1e-9);
    // P(y = 1 | x) is Logit's fitted probability; column 0 its complement.
    for (row, p) in mn.fitted_probabilities.iter().zip(&lg.fitted_probabilities) {
        close(row[1], *p, 1e-9);
        close(row[0], 1.0 - p, 1e-9);
    }
    close(mn.predict_proba(&[1.0]).unwrap()[1], 0.7, 1e-9);
}

#[test]
fn mnlogit_with_two_categories_matches_logit_on_two_regressors() {
    // statsmodels: MNLogit on L1: params.T [[-8.542419056821442, 1.8577555477621004, 0.4499962098304588]],
    //   llf -4.435039672866282, llr_pvalue 0.0012871623450507405 (df 2)
    let ctx = Context::new();
    let (y, x) = l1();
    let mn = mnlogit(&y, &x, true, &LogitOpts::default()).unwrap();
    let yb: Vec<u8> = y.iter().map(|&v| v as u8).collect();
    let lg = logit(&yb, &x, true, &LogitOpts::default()).unwrap();
    rel(mn.coefficients[0][0], -8.542_419_056_821_442, 1e-6);
    rel(mn.coefficients[0][1], 1.857_755_547_762_100_4, 1e-6);
    rel(mn.coefficients[0][2], 0.449_996_209_830_458_8, 1e-6);
    for a in 0..3 {
        close(mn.coefficients[0][a], lg.coefficients[a], 1e-8);
        close(mn.standard_errors[0][a], lg.standard_errors[a], 1e-8);
    }
    close(mn.log_likelihood, -4.435_039_672_866_282, 1e-8);
    close(
        mn.llr_test(&ctx).unwrap().p_value_f64().unwrap(),
        0.001_287_162_345_050_740_5,
        1e-9,
    );
}

#[test]
fn mnlogit_detects_complete_separation() {
    // statsmodels: MNLogit(y, add_constant(1..9)).fit(method='newton', maxiter=200) → ConvergenceWarning,
    //   converged False, params.T [[-149.75229827065252, 42.824983793202], [-424.2180038128295, 85.06197939380296]],
    //   llf -2.3639250564238745e-09 (no finite MLE).
    let x: Vec<Vec<f64>> = (1..=9).map(|i| vec![f64::from(i)]).collect();
    let y = [0, 0, 0, 1, 1, 1, 2, 2, 2];
    match mnlogit(&y, &x, true, &LogitOpts::default()) {
        Err(SymplexError::ComputationFailed { reason, .. }) => {
            assert!(reason.contains("separation"), "{reason}");
            assert!(reason.contains("category"), "{reason}");
            assert!(reason.contains("covariate 0"), "{reason}");
        }
        other => panic!("expected ComputationFailed, got {other:?}"),
    }
    let generous = LogitOpts {
        max_iter: 2000,
        tol: 1e-12,
    };
    assert!(matches!(
        mnlogit(&y, &x, true, &generous),
        Err(SymplexError::ComputationFailed { .. })
    ));
}

#[test]
fn mnlogit_detects_quasi_complete_separation() {
    // statsmodels: category 2 occurs only when the dummy is 1: fit(method='newton', maxiter=100) → converged False,
    //   params.T [[1.03e-11, -2.06e-11], [-25.297932546464665, 25.991079727009147]] (the category-2 equation diverges).
    let x: Vec<Vec<f64>> = (0..16)
        .map(|i| vec![if i < 8 { 0.0 } else { 1.0 }])
        .collect();
    let y = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 2, 2, 2, 1, 0, 2];
    match mnlogit(&y, &x, true, &LogitOpts::default()) {
        Err(SymplexError::ComputationFailed { reason, .. }) => {
            assert!(reason.contains("separation"), "{reason}");
            assert!(reason.contains("category 2"), "{reason}");
        }
        other => panic!("expected ComputationFailed, got {other:?}"),
    }
}

#[test]
fn mnlogit_rejects_invalid_input() {
    let (y, x) = m1();
    let opts = LogitOpts::default();
    // Empty, constant (k < 2), category with no observations, mismatched, ragged,
    // non-finite, no columns, n ≤ (k − 1)p, collinear, bad options.
    assert!(matches!(
        mnlogit(&[], &[], true, &opts),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        mnlogit(&[0; 24], &x, true, &opts),
        Err(SymplexError::InvalidArgument { .. })
    ));
    let gap: Vec<usize> = y.iter().map(|&c| if c == 1 { 2 } else { c }).collect();
    match mnlogit(&gap, &x, true, &opts) {
        Err(SymplexError::InvalidArgument { reason, .. }) => {
            assert!(
                reason.contains("category 1 has no observations"),
                "{reason}"
            );
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
    assert!(matches!(
        mnlogit(&y[..23], &x, true, &opts),
        Err(SymplexError::InvalidArgument { .. })
    ));
    let mut ragged = x.clone();
    ragged[2].push(1.0);
    assert!(mnlogit(&y, &ragged, true, &opts).is_err());
    let mut nan = x.clone();
    nan[2][0] = f64::NAN;
    assert!(mnlogit(&y, &nan, true, &opts).is_err());
    let empty_rows: Vec<Vec<f64>> = vec![vec![]; 24];
    assert!(mnlogit(&y, &empty_rows, false, &opts).is_err());
    assert!(matches!(
        mnlogit(&[0, 1, 2, 0], &x[..4], true, &opts),
        Err(SymplexError::InvalidArgument { .. })
    ));
    let dup: Vec<Vec<f64>> = x.iter().map(|r| vec![r[0], 2.0 * r[0]]).collect();
    match mnlogit(&y, &dup, true, &opts) {
        Err(SymplexError::InvalidArgument { reason, .. }) => {
            assert!(reason.contains("rank deficient"), "{reason}");
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
    assert!(
        mnlogit(
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
        mnlogit(
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
fn mnlogit_reports_non_convergence_when_iteration_budget_is_tiny() {
    // One Newton step from β = 0 is not the optimum for M1 (statsmodels needs 6).
    let (y, x) = m1();
    let fit = mnlogit(
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
    let full = mnlogit(&y, &x, true, &LogitOpts::default()).unwrap();
    assert!(full.log_likelihood > fit.log_likelihood);
}

// ═══════════════════════════════════════════════════════════════════════════
// ologit
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ologit_o1_parameters_match_statsmodels_after_the_threshold_transform() {
    // statsmodels: OrderedModel(y, x, distr='logit').fit(method='newton'): converged, params
    //   [0.11402351915116841, 0.12118510043763872, 0.47652318845685754] = (β, θ₀, ln(θ₁ − θ₀));
    //   transform_threshold_params(params)[1:-1] = [0.12118510043763872, 1.731650472914306]
    //   (θ₁ = θ₀ + exp(0.47652318845685754)).
    let (y, x) = o1();
    let fit = ologit(&y, &x, &LogitOpts::default()).unwrap();
    assert!(fit.converged);
    assert!(fit.iterations <= 20);
    assert_eq!((fit.n_categories, fit.n_params(), fit.nobs), (3, 3, 20));
    assert_eq!((fit.df_model, fit.df_resid), (1, 17));
    assert_eq!(fit.coefficients.len(), 1);
    assert_eq!(fit.thresholds.len(), 2);
    rel(fit.coefficients[0], 0.114_023_519_151_168_41, 1e-6);
    rel(fit.thresholds[0], 0.121_185_100_437_638_72, 1e-6);
    rel(fit.thresholds[1], 1.731_650_472_914_306, 1e-6);
    // The crate's thresholds are statsmodels' after the log-difference transform.
    close(
        fit.thresholds[1],
        0.121_185_100_437_638_72 + 0.476_523_188_456_857_54_f64.exp(),
        1e-6,
    );
    assert!(fit.thresholds[0] < fit.thresholds[1]);
}

#[test]
fn ologit_o1_standard_errors_are_for_the_natural_thresholds() {
    // statsmodels: bse [0.07636599031948124, 0.8004024090884866, 0.3374747686396624] (β, θ₀, ln Δ; numerical Hessian);
    //   delta method J cov J.T → se_nat [0.07636599031948124, 0.8004024090884866, 0.8933257969028551],
    //   z_nat [1.4931191054308977, 0.15140521700284063, 1.9384310616775067],
    //   p_nat [0.13540601228098428, 0.8796560743328857, 0.052570653882098715].
    //   Exact (mpmath, 40 digits, analytic Hessian at the optimum, rounded to 17 significant digits):
    //   se [0.076365882208287882, 0.80040078804408812, 0.89332461197224442] → [0.07636588220828788, 0.8004007880440881, 0.8933246119722444].
    let (y, x) = o1();
    let fit = ologit(&y, &x, &LogitOpts::default()).unwrap();
    assert_eq!(fit.standard_errors.len(), 3);
    rel(fit.standard_errors[0], 0.076_365_990_319_481_24, 1e-5);
    rel(fit.standard_errors[1], 0.800_402_409_088_486_6, 1e-5);
    rel(fit.standard_errors[2], 0.893_325_796_902_855_1, 1e-5);
    rel(fit.standard_errors[0], 0.076_365_882_208_287_88, 1e-9);
    rel(fit.standard_errors[1], 0.800_400_788_044_088_1, 1e-9);
    rel(fit.standard_errors[2], 0.893_324_611_972_244_4, 1e-9);
    rel(fit.z_values[0], 1.493_119_105_430_897_7, 1e-5);
    rel(fit.z_values[2], 1.938_431_061_677_506_7, 1e-5);
    rel(fit.p_values[0], 0.135_406_012_280_984_28, 1e-5);
    rel(fit.p_values[1], 0.879_656_074_332_885_7, 1e-5);
    rel(fit.p_values[2], 0.052_570_653_882_098_715, 1e-5);
    // The log-difference standard error is not the threshold's.
    assert!((fit.standard_errors[2] - 0.337_474_768_639_662_4).abs() > 0.5);
    for j in 0..3 {
        close(fit.cov_params[j][j], fit.standard_errors[j].powi(2), 1e-12);
    }
}

#[test]
fn ologit_o1_cov_params_matches_the_delta_method() {
    // statsmodels: cov_nat = J cov_params J.T = [[0.005831764477475103, 0.047301204200038444, 0.057212050561999836],
    //   [0.04730120420003844, 0.640644016474653, 0.5716460312567045], [0.057212050561999836, 0.5716460312567047, 0.7980309794121211]]
    let (y, x) = o1();
    let fit = ologit(&y, &x, &LogitOpts::default()).unwrap();
    let expected = [
        [
            0.005_831_764_477_475_103,
            0.047_301_204_200_038_444,
            0.057_212_050_561_999_836,
        ],
        [
            0.047_301_204_200_038_44,
            0.640_644_016_474_653,
            0.571_646_031_256_704_5,
        ],
        [
            0.057_212_050_561_999_836,
            0.571_646_031_256_704_7,
            0.798_030_979_412_121_1,
        ],
    ];
    for (r, row) in expected.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            rel(fit.cov_params[r][c], *v, 1e-5);
            close(fit.cov_params[r][c], fit.cov_params[c][r], 1e-12);
        }
    }
}

#[test]
fn ologit_o1_likelihood_summary_and_llr_test() {
    // statsmodels: llf -20.751965417580625, llnull -21.921346568937107 (closed form -21.921346568937103),
    //   llr 2.338762302712965, llr_pvalue 0.12618978075284423 (df_model 1), prsquared 0.053344403259127926,
    //   aic 47.50393083516125, bic 50.49112765582322
    let ctx = Context::new();
    let (y, x) = o1();
    let fit = ologit(&y, &x, &LogitOpts::default()).unwrap();
    close(fit.log_likelihood, -20.751_965_417_580_625, 1e-8);
    close(fit.null_log_likelihood, -21.921_346_568_937_103, 1e-8);
    close(fit.llr(), 2.338_762_302_712_965, 1e-8);
    rel(fit.pseudo_r_squared, 0.053_344_403_259_127_926, 1e-7);
    close(fit.aic(), 47.503_930_835_161_25, 1e-8);
    close(fit.bic(), 50.491_127_655_823_22, 1e-8);
    let t = fit.llr_test(&ctx).unwrap();
    close(t.statistic_f64().unwrap(), 2.338_762_302_712_965, 1e-8);
    close(t.p_value_f64().unwrap(), 0.126_189_780_752_844_23, 1e-8);
    assert_eq!(t.df, Some(ctx.int(1)));
    assert_eq!(t.alternative, Alternative::Greater);
}

#[test]
fn ologit_o1_predictions() {
    // statsmodels: predict()[:2] = [[0.5302592523209494, 0.31936415978307753, 0.15037658789597308],
    //   [0.5017903876694705, 0.3326772099972721, 0.1655324023332574]];
    //   predict(which='cumprob')[:2] = [[0.5302592523209494, 0.8496234121040269, 1.0], [0.5017903876694705, 0.8344675976667426, 1.0]];
    //   predict(exog=[[7.0]]) = [0.33693577245431927, 0.38084617872902227, 0.28221804881665846]
    let (y, x) = o1();
    let fit = ologit(&y, &x, &LogitOpts::default()).unwrap();
    assert_eq!(fit.fitted_probabilities.len(), 20);
    rel(
        fit.fitted_probabilities[0][0],
        0.530_259_252_320_949_4,
        1e-6,
    );
    rel(
        fit.fitted_probabilities[0][1],
        0.319_364_159_783_077_53,
        1e-6,
    );
    rel(
        fit.fitted_probabilities[0][2],
        0.150_376_587_895_973_08,
        1e-6,
    );
    rel(
        fit.fitted_probabilities[1][1],
        0.332_677_209_997_272_1,
        1e-6,
    );
    let cum = fit.cumulative_proba(&[1.0]).unwrap();
    rel(cum[0], 0.501_790_387_669_470_5, 1e-6);
    rel(cum[1], 0.834_467_597_666_742_6, 1e-6);
    assert_eq!(cum[2], 1.0);
    let at7 = fit.predict_proba(&[7.0]).unwrap();
    rel(at7[0], 0.336_935_772_454_319_27, 1e-6);
    rel(at7[1], 0.380_846_178_729_022_27, 1e-6);
    rel(at7[2], 0.282_218_048_816_658_46, 1e-6);
    close(at7.iter().sum::<f64>(), 1.0, 1e-12);
    assert_eq!(fit.predict(&[7.0]).unwrap(), 1);
    assert_eq!(fit.predict(&[0.0]).unwrap(), 0);
    assert_eq!(fit.predict(&[19.0]).unwrap(), 2);
    assert!(fit.predict_proba(&[1.0, 2.0]).is_err());
    assert!(fit.cumulative_proba(&[f64::INFINITY]).is_err());
    assert!(fit.predict(&[]).is_err());
}

#[test]
fn ologit_o1_conf_int_and_odds_ratios() {
    // statsmodels: conf_int(0.05)[:1] = [[-0.03565107151874923, 0.26369810982108605]], np.exp(params[:1]) = [1.1207784843327753]
    let (y, x) = o1();
    let fit = ologit(&y, &x, &LogitOpts::default()).unwrap();
    let ci = fit.conf_int(0.95).unwrap();
    assert_eq!(ci.len(), 1);
    close(ci[0].lower, -0.035_651_071_518_749_23, 1e-6);
    close(ci[0].upper, 0.263_698_109_821_086_05, 1e-6);
    rel(fit.odds_ratios()[0], 1.120_778_484_332_775_3, 1e-6);
    assert!(fit.conf_int(0.0).is_err());
    assert!(fit.conf_int(1.0).is_err());
}

#[test]
fn ologit_rescaling_the_regressor_scales_its_slope_inversely() {
    // statsmodels: OrderedModel(y, x / 10).fit(method='newton'): params [1.1402351954070107, 0.12118510557733055, 0.4765231881211714],
    //   thresholds [0.12118510557733055, 1.731650477513387], llf -20.751965417580628, np.exp(β) = 3.127503853232749
    let (y, x) = o1();
    let fit = ologit(&y, &scaled(&x, 0.1), &LogitOpts::default()).unwrap();
    let full = ologit(&y, &x, &LogitOpts::default()).unwrap();
    rel(fit.coefficients[0], 1.140_235_195_407_010_7, 1e-6);
    rel(fit.thresholds[0], 0.121_185_105_577_330_55, 1e-6);
    rel(fit.thresholds[1], 1.731_650_477_513_387, 1e-6);
    close(fit.log_likelihood, -20.751_965_417_580_628, 1e-8);
    close(fit.coefficients[0], 10.0 * full.coefficients[0], 1e-9);
    close(fit.standard_errors[0], 10.0 * full.standard_errors[0], 1e-9);
    close(fit.thresholds[0], full.thresholds[0], 1e-9);
    close(fit.thresholds[1], full.thresholds[1], 1e-9);
    close(fit.z_values[0], full.z_values[0], 1e-9);
    rel(fit.odds_ratios()[0], 3.127_503_853_232_749, 1e-6);
}

#[test]
fn ologit_o2_four_categories_two_regressors() {
    // statsmodels: OrderedModel(y, X, distr='logit').fit(method='newton'): 19 iterations, params
    //   [0.41603460421737887, 0.2919361880851621, 0.8275090202482066, 0.5623517567211521, 0.5268263335386298];
    //   transform_threshold_params(params)[1:-1] = [0.8275090202482066, 2.5823035214352488, 4.275852532479732];
    //   se_nat (delta method) [0.21523210446360697, 0.1485340521523365, 0.9657629613448437, 1.0304316321060842, 1.2198095982836235];
    //   p_nat[4] = 0.00045601726859448327
    let (y, x) = o2();
    let fit = ologit(&y, &x, &LogitOpts::default()).unwrap();
    assert!(fit.converged);
    assert_eq!((fit.n_categories, fit.n_params(), fit.nobs), (4, 5, 30));
    assert_eq!((fit.df_model, fit.df_resid), (2, 25));
    rel(fit.coefficients[0], 0.416_034_604_217_378_87, 1e-6);
    rel(fit.coefficients[1], 0.291_936_188_085_162_1, 1e-6);
    rel(fit.thresholds[0], 0.827_509_020_248_206_6, 1e-6);
    rel(fit.thresholds[1], 2.582_303_521_435_248_8, 1e-6);
    rel(fit.thresholds[2], 4.275_852_532_479_732, 1e-6);
    let se_nat = [
        0.215_232_104_463_606_97,
        0.148_534_052_152_336_5,
        0.965_762_961_344_843_7,
        1.030_431_632_106_084_2,
        1.219_809_598_283_623_5,
    ];
    for (s, e) in fit.standard_errors.iter().zip(se_nat) {
        rel(*s, e, 1e-5);
    }
    rel(fit.p_values[4], 0.000_456_017_268_594_483_27, 1e-4);
    assert!(fit.thresholds.windows(2).all(|w| w[0] < w[1]));
}

#[test]
fn ologit_o2_likelihood_and_predictions() {
    // statsmodels: llf -36.29934259418187, llnull -40.817318452255016, llr 9.035951716146286, llr_pvalue 0.01091108691994756 (df 2),
    //   prsquared 0.1106877185809726, aic 82.59868518836375, bic 89.60467209667452;
    //   predict()[0] = [0.3859581484147373, 0.39826695008520085, 0.16762137431208324, 0.048153527187978606];
    //   predict(which='cumprob')[0] = [0.3859581484147373, 0.7842250984999382, 0.9518464728120214, 1.0];
    //   predict(exog=[[3, 5]]) = [0.13235720551146332, 0.3363135837378929, 0.35883719121087787, 0.1724920195397659];
    //   np.exp(β) = [1.5159383258223076, 1.3390175696379023]; conf_int(0.05)[:2] = [[-0.005812568848053368, 0.8378817772828111],
    //   [0.0008147953887884252, 0.5830575807815357]]
    let ctx = Context::new();
    let (y, x) = o2();
    let fit = ologit(&y, &x, &LogitOpts::default()).unwrap();
    close(fit.log_likelihood, -36.299_342_594_181_87, 1e-8);
    close(fit.null_log_likelihood, -40.817_318_452_255_016, 1e-8);
    close(fit.llr(), 9.035_951_716_146_286, 1e-8);
    rel(fit.pseudo_r_squared, 0.110_687_718_580_972_6, 1e-7);
    close(fit.aic(), 82.598_685_188_363_75, 1e-8);
    close(fit.bic(), 89.604_672_096_674_52, 1e-8);
    let t = fit.llr_test(&ctx).unwrap();
    assert_eq!(t.df, Some(ctx.int(2)));
    close(t.p_value_f64().unwrap(), 0.010_911_086_919_947_56, 1e-8);
    let first = &fit.fitted_probabilities[0];
    rel(first[0], 0.385_958_148_414_737_3, 1e-6);
    rel(first[1], 0.398_266_950_085_200_85, 1e-6);
    rel(first[2], 0.167_621_374_312_083_24, 1e-6);
    rel(first[3], 0.048_153_527_187_978_606, 1e-6);
    let cum = fit.cumulative_proba(&[1.0, 3.0]).unwrap();
    rel(cum[1], 0.784_225_098_499_938_2, 1e-6);
    rel(cum[2], 0.951_846_472_812_021_4, 1e-6);
    let pr = fit.predict_proba(&[3.0, 5.0]).unwrap();
    rel(pr[0], 0.132_357_205_511_463_32, 1e-6);
    rel(pr[1], 0.336_313_583_737_892_9, 1e-6);
    rel(pr[2], 0.358_837_191_210_877_87, 1e-6);
    rel(pr[3], 0.172_492_019_539_765_9, 1e-6);
    assert_eq!(fit.predict(&[3.0, 5.0]).unwrap(), 2);
    let or = fit.odds_ratios();
    rel(or[0], 1.515_938_325_822_307_6, 1e-6);
    rel(or[1], 1.339_017_569_637_902_3, 1e-6);
    let ci = fit.conf_int(0.95).unwrap();
    assert_eq!(ci.len(), 2);
    close(ci[0].lower, -0.005_812_568_848_053_368, 1e-5);
    close(ci[0].upper, 0.837_881_777_282_811_1, 1e-5);
    close(ci[1].lower, 0.000_814_795_388_788_425_2, 1e-5);
    close(ci[1].upper, 0.583_057_580_781_535_7, 1e-5);
}

#[test]
fn ologit_with_two_categories_is_logit_with_negated_intercept() {
    // statsmodels: OrderedModel(y, x, distr='logit') on L2 (Newton stops loosely: converged False after 200,
    //   params [1.6945957207417444, 0.8472978603949245] ≈ (ln(49/9), −ln(3/7)), bse [0.9759000871903722, 0.6900655779581232],
    //   llf -12.217286041097871).  P(y = 1 | x) = σ(xβ − θ₀): Logit's intercept is −θ₀.
    let (y, x) = l2();
    let ol = ologit(&y, &x, &LogitOpts::default()).unwrap();
    let yb: Vec<u8> = y.iter().map(|&v| v as u8).collect();
    let lg = logit(&yb, &x, true, &LogitOpts::default()).unwrap();
    assert!(ol.converged);
    close(ol.coefficients[0], (49.0_f64 / 9.0).ln(), 1e-9);
    close(ol.thresholds[0], -(3.0_f64 / 7.0).ln(), 1e-9);
    close(ol.coefficients[0], lg.coefficients[1], 1e-9);
    close(ol.thresholds[0], -lg.coefficients[0], 1e-9);
    close(ol.standard_errors[0], lg.standard_errors[1], 1e-9);
    close(ol.standard_errors[1], lg.standard_errors[0], 1e-9);
    close(ol.log_likelihood, lg.log_likelihood, 1e-10);
    close(ol.log_likelihood, -12.217_286_041_097_871, 1e-8);
    close(ol.null_log_likelihood, lg.null_log_likelihood, 1e-10);
    close(ol.aic(), lg.aic(), 1e-9);
    close(ol.bic(), lg.bic(), 1e-9);
    close(ol.predict_proba(&[1.0]).unwrap()[1], 0.7, 1e-9);
    close(ol.predict_proba(&[0.0]).unwrap()[1], 0.3, 1e-9);
    close(ol.cumulative_proba(&[1.0]).unwrap()[0], 0.3, 1e-9);
    close(ol.odds_ratios()[0], 49.0 / 9.0, 1e-8);
}

#[test]
fn ologit_detects_complete_separation() {
    // statsmodels: OrderedModel(y, 1..9, distr='logit').fit(method='newton', maxiter=200) → converged False,
    //   params [46.81201039455544, 163.57507634777056, 4.946650516133856], llf -2.7840429962404296e-10 (no finite MLE).
    let x: Vec<Vec<f64>> = (1..=9).map(|i| vec![f64::from(i)]).collect();
    let y = [0, 0, 0, 1, 1, 1, 2, 2, 2];
    match ologit(&y, &x, &LogitOpts::default()) {
        Err(SymplexError::ComputationFailed { reason, .. }) => {
            assert!(reason.contains("separation"), "{reason}");
            assert!(reason.contains("covariate 0"), "{reason}");
        }
        other => panic!("expected ComputationFailed, got {other:?}"),
    }
    let generous = LogitOpts {
        max_iter: 2000,
        tol: 1e-12,
    };
    assert!(matches!(
        ologit(&y, &x, &generous),
        Err(SymplexError::ComputationFailed { .. })
    ));
}

#[test]
fn ologit_rejects_a_category_with_no_observations() {
    // statsmodels: OrderedModel on a numpy y ∈ {0, 2} silently relabels with np.unique (k_levels 2, labels [0 2]);
    //   on a pandas ordered Categorical with categories [0, 1, 2] the start value ln(θ₁ − θ₀) = ln 0 = −inf,
    //   the fit ends with HessianInversionWarning + ConvergenceWarning and params [0, 0, −inf].
    let x: Vec<Vec<f64>> = (0..10).map(|i| vec![f64::from(i)]).collect();
    let y = [0, 0, 2, 0, 2, 2, 0, 2, 2, 0];
    match ologit(&y, &x, &LogitOpts::default()) {
        Err(SymplexError::InvalidArgument { reason, .. }) => {
            assert!(
                reason.contains("category 1 has no observations"),
                "{reason}"
            );
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
    // Category 0 unused is the same error: y must be 0..k.
    let shifted = [1, 1, 2, 1, 2, 2, 1, 2, 2, 1];
    match ologit(&shifted, &x, &LogitOpts::default()) {
        Err(SymplexError::InvalidArgument { reason, .. }) => {
            assert!(
                reason.contains("category 0 has no observations"),
                "{reason}"
            );
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

#[test]
fn ologit_rejects_invalid_input() {
    let (y, x) = o1();
    let opts = LogitOpts::default();
    // Empty, constant y, mismatched, ragged, non-finite, no columns, constant column,
    // n ≤ p + k − 1, collinear, bad options.
    assert!(matches!(
        ologit(&[], &[], &opts),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        ologit(&[0; 20], &x, &opts),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        ologit(&y[..19], &x, &opts),
        Err(SymplexError::InvalidArgument { .. })
    ));
    let mut ragged = x.clone();
    ragged[2].push(1.0);
    assert!(ologit(&y, &ragged, &opts).is_err());
    let mut nan = x.clone();
    nan[2][0] = f64::NAN;
    assert!(ologit(&y, &nan, &opts).is_err());
    let empty_rows: Vec<Vec<f64>> = vec![vec![]; 20];
    assert!(ologit(&y, &empty_rows, &opts).is_err());
    let with_const: Vec<Vec<f64>> = x.iter().map(|r| vec![1.0, r[0]]).collect();
    match ologit(&y, &with_const, &opts) {
        Err(SymplexError::InvalidArgument { reason, .. }) => {
            assert!(reason.contains("column 0 of x is constant"), "{reason}");
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
    assert!(matches!(
        ologit(&[0, 1, 2], &x[..3], &opts),
        Err(SymplexError::InvalidArgument { .. })
    ));
    let dup: Vec<Vec<f64>> = x.iter().map(|r| vec![r[0], 2.0 * r[0]]).collect();
    match ologit(&y, &dup, &opts) {
        Err(SymplexError::InvalidArgument { reason, .. }) => {
            assert!(reason.contains("rank deficient"), "{reason}");
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
    assert!(
        ologit(
            &y,
            &x,
            &LogitOpts {
                max_iter: 0,
                tol: 1e-8
            }
        )
        .is_err()
    );
    assert!(
        ologit(
            &y,
            &x,
            &LogitOpts {
                max_iter: 10,
                tol: f64::NAN
            }
        )
        .is_err()
    );
}

#[test]
fn ologit_reports_non_convergence_when_iteration_budget_is_tiny() {
    let (y, x) = o1();
    let fit = ologit(
        &y,
        &x,
        &LogitOpts {
            max_iter: 1,
            tol: 1e-10,
        },
    )
    .unwrap();
    assert!(!fit.converged);
    assert_eq!(fit.iterations, 1);
    let full = ologit(&y, &x, &LogitOpts::default()).unwrap();
    assert!(full.log_likelihood > fit.log_likelihood);
    // Even the one-step fit keeps the thresholds ordered and the probabilities proper.
    assert!(fit.thresholds[0] < fit.thresholds[1]);
    for row in &fit.fitted_probabilities {
        close(row.iter().sum::<f64>(), 1.0, 1e-12);
    }
}

#[test]
fn ologit_fitted_probabilities_are_differences_of_the_cumulative_curve() {
    let (y, x) = o2();
    let fit = ologit(&y, &x, &LogitOpts::default()).unwrap();
    for (row, pr) in x.iter().zip(&fit.fitted_probabilities) {
        let cum = fit.cumulative_proba(row).unwrap();
        close(pr[0], cum[0], 1e-12);
        close(pr[1], cum[1] - cum[0], 1e-12);
        close(pr[2], cum[2] - cum[1], 1e-12);
        close(pr[3], 1.0 - cum[2], 1e-12);
        assert_eq!(fit.predict_proba(row).unwrap(), *pr);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Book: guide/response-analysis.md, "Several answers, or ordered answers"
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn book_response_analysis_mnlogit_example() {
    // The numbers printed in the book subsection (statsmodels M1 oracle rounded to 4 places).
    let (y, x) = m1();
    let fit = mnlogit(&y, &x, true, &LogitOpts::default()).unwrap();
    assert_eq!(
        format!(
            "{:.4} {:.4}",
            fit.coefficients[1][0], fit.coefficients[1][1]
        ),
        "-2.8750 0.2536"
    );
    assert_eq!(format!("{:.4}", fit.relative_risk_ratios()[1][1]), "1.2887");
    let p = fit.predict_proba(&[12.0]).unwrap();
    assert_eq!(
        format!("{:.3} {:.3} {:.3}", p[0], p[1], p[2]),
        "0.272 0.406 0.322"
    );
    assert_eq!(fit.predict(&[12.0]).unwrap(), 1);
    assert_eq!(format!("{:.4}", fit.pseudo_r_squared), "0.1572");
}

#[test]
fn book_response_analysis_ologit_example() {
    let ctx = Context::new();
    let (y, x) = o1();
    let fit = ologit(&y, &x, &LogitOpts::default()).unwrap();
    assert_eq!(format!("{:.4}", fit.coefficients[0]), "0.1140");
    assert_eq!(
        format!("{:.4} {:.4}", fit.thresholds[0], fit.thresholds[1]),
        "0.1212 1.7317"
    );
    assert_eq!(format!("{:.4}", fit.odds_ratios()[0]), "1.1208");
    let p = fit.predict_proba(&[7.0]).unwrap();
    assert_eq!(
        format!("{:.3} {:.3} {:.3}", p[0], p[1], p[2]),
        "0.337 0.381 0.282"
    );
    assert_eq!(
        format!("{:.3}", fit.llr_test(&ctx).unwrap().p_value_f64().unwrap()),
        "0.126"
    );
}
