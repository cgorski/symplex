//! symplex 0.21 — the shared summaries of the maximum-likelihood fits
//! ([`LikelihoodFit`], [`WaldFit`]) and the `Family` boilerplate.
//!
//! Every fit value below was already documented against statsmodels 0.15 /
//! scipy 1.18 (`symplex/.venv/bin/python`) in an earlier track, and the
//! test that cites it is named at each block:
//!
//! * `Logit` on L2 — `tests/v14/v14_regression.rs`
//!   (`logit_binary_regressor_has_closed_form`), plus
//!   `llr_pvalue = chi2.sf(3.2913151402020766, 1) = 0.0696472158708212`
//!   (scipy, this track).
//! * `MnLogit` on M1 — `tests/v19/v19_mnlogit.rs`
//!   (`mnlogit_m1_standard_errors_z_and_p_values`, `mnlogit_m1_likelihood_summary`,
//!   `mnlogit_m1_llr_test_is_chi_squared_with_slope_degrees_of_freedom`,
//!   `mnlogit_m1_conf_int_and_relative_risk_ratios`,
//!   `mnlogit_without_intercept_matches_statsmodels`).
//! * `OrderedLogit` on O1 — `tests/v19/v19_mnlogit.rs`
//!   (`ologit_o1_standard_errors_are_for_the_natural_thresholds`,
//!   `ologit_o1_likelihood_summary_and_llr_test`, `ologit_o1_conf_int_and_odds_ratios`).
//! * `CoxModel` on D1 / D2 — `tests/v18/v18_cox.rs`
//!   (`d1_efron_coefficient_and_standard_error_match_statsmodels`,
//!   `d1_log_likelihoods_llr_and_aic_match_statsmodels`,
//!   `d1_hazard_ratios_and_wald_intervals_match_statsmodels`,
//!   `d1_likelihood_ratio_wald_and_score_tests_match_statsmodels`,
//!   `d2_efron_coefficients_match_statsmodels`,
//!   `d2_efron_log_likelihoods_and_the_test_trio_match_statsmodels`), plus
//!   D1's `tvalues [2.299749228852089]`, `pvalues [0.021462431348704628]`
//!   (`PHReg(...).fit()`, this track).
//!
//! `CoxModel::bic` is new.  statsmodels' `PHRegResults` has **no** `bic`
//! attribute (checked: `hasattr(r, "bic")` is `False`), so the oracle is
//! R's definition `BIC(coxph) = −2ℓ + p·ln(nevent)` (`logLik.coxph` sets
//! `nobs = nevent`), computed by hand from statsmodels' `llf` and the
//! event count the crate reports (`fit.n_events`, itself checked against
//! the data): D1 `-2(-8.465216276861835) + ln 7 = 18.87634270277898`,
//! D2 (Efron) `-2(-15.16722416403536) + 2 ln 8 = 34.49333141143039`.

use symplex::prelude::*;
use symplex::stats::cox::{CoxModel, CoxOpts, cox_ph};
use symplex::stats::hypothesis::Alternative;
use symplex::stats::order::order_statistic;
use symplex::stats::regression::{Logit, LogitOpts, MnLogit, OrderedLogit, logit, mnlogit, ologit};
use symplex::stats::survival::Observation;
use symplex::stats::{Distribution, Kind, LikelihoodFit, Piece, Support, WaldFit};

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

// ═══════════════════════════════════════════════════════════════════════════
// Data (copied from the citing test files)
// ═══════════════════════════════════════════════════════════════════════════

/// L2 (v14): binary regressor, ten controls with 3 successes, ten treated with 7.
fn l2() -> Logit {
    let y: Vec<bool> = (0..20).map(|i| matches!(i, 7..=9 | 13..=19)).collect();
    let x: Vec<Vec<f64>> = (0..20)
        .map(|i| vec![if i < 10 { 0.0 } else { 1.0 }])
        .collect();
    logit(&y, &x, true, &LogitOpts::default()).unwrap()
}

/// M1 (v19): k = 3, x = 1..24 (one regressor + intercept).
fn m1_data() -> (Vec<usize>, Vec<Vec<f64>>) {
    let y = vec![
        0, 0, 1, 0, 0, 1, 2, 0, 1, 1, 2, 0, 1, 2, 1, 2, 1, 2, 2, 1, 2, 0, 2, 2,
    ];
    let x = (1..=24).map(|i| vec![f64::from(i)]).collect();
    (y, x)
}

fn m1() -> MnLogit {
    let (y, x) = m1_data();
    mnlogit(&y, &x, true, &LogitOpts::default()).unwrap()
}

/// O1 (v19): k = 3 ordered, x = 0..19.
fn o1() -> OrderedLogit {
    let y = vec![0, 0, 1, 0, 1, 1, 2, 1, 2, 2, 0, 1, 2, 2, 1, 0, 0, 1, 2, 2];
    let x: Vec<Vec<f64>> = (0..20).map(|i| vec![f64::from(i)]).collect();
    ologit(&y, &x, &LogitOpts::default()).unwrap()
}

/// D1 (v18): ten subjects, one covariate, no tied event times, three censored.
fn d1() -> CoxModel {
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
    cox_ph(&obs, &x, &CoxOpts::default()).unwrap()
}

/// D2 (v18): twelve subjects, two covariates, tied event times.
fn d2() -> CoxModel {
    let obs = Observation::from_i64(
        &[3, 3, 5, 5, 5, 7, 8, 8, 10, 12, 12, 14],
        &[
            true, true, true, false, true, true, true, false, true, true, false, false,
        ],
    );
    let x = vec![
        vec![2.0, 1.0],
        vec![4.0, 0.0],
        vec![1.0, 1.0],
        vec![3.0, 0.0],
        vec![5.0, 1.0],
        vec![2.0, 0.0],
        vec![6.0, 1.0],
        vec![1.0, 0.0],
        vec![3.0, 1.0],
        vec![0.0, 0.0],
        vec![4.0, 1.0],
        vec![2.0, 0.0],
    ];
    cox_ph(&obs, &x, &CoxOpts::default()).unwrap()
}

// ═══════════════════════════════════════════════════════════════════════════
// Generic use: one function over every implementor
// ═══════════════════════════════════════════════════════════════════════════

/// A one-line summary written against the traits only.
fn report<F: LikelihoodFit + WaldFit>(f: &F) -> String {
    let z: Vec<String> = f.z_values().iter().map(|z| format!("{z:.3}")).collect();
    format!(
        "k={} n={} df={} ll={:.4} aic={:.4} bic={:.4} llr={:.4} r2={:.4} z=[{}]",
        f.n_params(),
        f.nobs(),
        f.df_model(),
        f.log_likelihood(),
        f.aic(),
        f.bic(),
        f.llr(),
        f.pseudo_r_squared(),
        z.join(", ")
    )
}

#[test]
fn report_is_usable_on_every_implementor() {
    // Logit L2: llf -12.217286041097868, aic 28.434572082195736, bic 30.426036629303717,
    //   llr 3.2913151402020766, prsquared 0.11870910076930752, tvalues [-1.2278512511111188, 1.736443891898117]
    assert_eq!(
        report(&l2()),
        "k=2 n=20 df=1 ll=-12.2173 aic=28.4346 bic=30.4260 llr=3.2913 r2=0.1187 z=[-1.228, 1.736]"
    );
    // MnLogit M1: (k − 1)p = 4 parameters; llf -22.117379463121267, aic 52.234758926242534,
    //   bic 56.946974247634316, llr 8.247975791118542, prsquared 0.15715598349699078,
    //   tvalues.T [[-0.8993466528457352, 1.1797762969324326], [-1.99210397698943, 2.3447486337546914]]
    assert_eq!(
        report(&m1()),
        "k=4 n=24 df=2 ll=-22.1174 aic=52.2348 bic=56.9470 llr=8.2480 r2=0.1572 z=[-0.899, 1.180, -1.992, 2.345]"
    );
    // OrderedLogit O1: p + k − 1 = 3; llf -20.751965417580625, aic 47.50393083516125, bic 50.49112765582322,
    //   llr 2.338762302712965, prsquared 0.053344403259127926, z_nat[0] 1.4931191054308977
    assert_eq!(
        report(&o1()),
        "k=3 n=20 df=1 ll=-20.7520 aic=47.5039 bic=50.4911 llr=2.3388 r2=0.0533 z=[1.493]"
    );
    // Cox D1: llf -8.465216276861835, aic 18.93043255372367, BIC(coxph) 18.87634270277898,
    //   LR 7.286928045319378, 1 − llf/llnull = 0.3008968717097652, tvalues [2.299749228852089]
    assert_eq!(
        report(&d1()),
        "k=1 n=10 df=1 ll=-8.4652 aic=18.9304 bic=18.8763 llr=7.2869 r2=0.3009 z=[2.300]"
    );
}

/// The trait's provided methods and the inherent methods of the same name
/// are the same numbers (bit for bit: the inherent ones delegate).
fn inherent_matches_trait<F: LikelihoodFit + WaldFit>(
    f: &F,
    inherent: (f64, f64, f64, Vec<Interval<f64>>),
) {
    let (llr, aic, bic, ci) = inherent;
    assert_eq!(LikelihoodFit::llr(f), llr);
    assert_eq!(LikelihoodFit::aic(f), aic);
    assert_eq!(LikelihoodFit::bic(f), bic);
    assert_eq!(WaldFit::conf_int(f, 0.95).unwrap(), ci);
}

#[test]
fn inherent_methods_delegate_to_the_traits() {
    let fit = l2();
    inherent_matches_trait(
        &fit,
        (fit.llr(), fit.aic(), fit.bic(), fit.conf_int(0.95).unwrap()),
    );
    let fit = o1();
    inherent_matches_trait(
        &fit,
        (fit.llr(), fit.aic(), fit.bic(), fit.conf_int(0.95).unwrap()),
    );
    let fit = d1();
    inherent_matches_trait(
        &fit,
        (fit.llr(), fit.aic(), fit.bic(), fit.conf_int(0.95).unwrap()),
    );
    // MnLogit's inherent `conf_int` is nested; the trait's is the same
    // intervals flattened in `cov_params` order.
    let fit = m1();
    assert_eq!(LikelihoodFit::llr(&fit), fit.llr());
    assert_eq!(LikelihoodFit::aic(&fit), fit.aic());
    assert_eq!(LikelihoodFit::bic(&fit), fit.bic());
    let nested = fit.conf_int(0.95).unwrap();
    let flat = WaldFit::conf_int(&fit, 0.95).unwrap();
    assert_eq!(nested.concat(), flat);
}

// ═══════════════════════════════════════════════════════════════════════════
// Logit (L2 — v14 `logit_binary_regressor_has_closed_form`)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn logit_likelihood_fit_matches_v14() {
    // llf -12.217286041097868, llnull -13.862943611198906, prsquared 0.11870910076930752,
    // llr 3.2913151402020766, aic 28.434572082195736, bic 30.426036629303717; nobs 20, df_model 1
    let fit = l2();
    assert_eq!(
        (
            fit.n_params(),
            LikelihoodFit::nobs(&fit),
            LikelihoodFit::df_model(&fit)
        ),
        (2, 20, 1)
    );
    close(
        LikelihoodFit::log_likelihood(&fit),
        -12.217_286_041_097_868,
        1e-6,
    );
    close(
        LikelihoodFit::null_log_likelihood(&fit),
        -13.862_943_611_198_906,
        1e-6,
    );
    close(
        LikelihoodFit::pseudo_r_squared(&fit),
        0.118_709_100_769_307_52,
        1e-6,
    );
    assert_eq!(LikelihoodFit::pseudo_r_squared(&fit), fit.pseudo_r_squared);
    close(LikelihoodFit::llr(&fit), 3.291_315_140_202_076_6, 1e-6);
    close(LikelihoodFit::aic(&fit), 28.434_572_082_195_736, 1e-6);
    close(LikelihoodFit::bic(&fit), 30.426_036_629_303_717, 1e-6);
}

#[test]
fn logit_llr_test_is_chi_squared_on_one_degree_of_freedom() {
    // scipy: chi2.sf(3.2913151402020766, 1) = 0.0696472158708212 (= statsmodels llr_pvalue)
    let ctx = Context::new();
    let fit = l2();
    let t = fit.llr_test(&ctx).unwrap();
    close(t.statistic_f64().unwrap(), 3.291_315_140_202_076_6, 1e-6);
    close(t.p_value_f64().unwrap(), 0.069_647_215_870_821_2, 1e-8);
    assert_eq!(t.df, Some(ctx.int(1)));
    assert_eq!(t.alternative, Alternative::Greater);
    assert_eq!(t, LikelihoodFit::llr_test(&fit, &ctx).unwrap());
}

#[test]
fn logit_wald_fit_matches_v14() {
    // params [-0.8472978603872037, 1.6945957207744073], bse [0.6900655593423543, 0.9759000729485332],
    // tvalues [-1.2278512511111188, 1.736443891898117], pvalues [0.21950281228300073, 0.08248537711586468],
    // conf_int(0.05) [[-2.1998015036697054, 0.505205782895298], [-0.2181332747147291, 3.607324716263544]]
    let fit = l2();
    close(
        WaldFit::coefficients(&fit)[0],
        -0.847_297_860_387_203_7,
        1e-6,
    );
    close(
        WaldFit::coefficients(&fit)[1],
        1.694_595_720_774_407_3,
        1e-6,
    );
    close(
        WaldFit::standard_errors(&fit)[0],
        0.690_065_559_342_354_3,
        1e-6,
    );
    close(
        WaldFit::standard_errors(&fit)[1],
        0.975_900_072_948_533_2,
        1e-6,
    );
    let z = fit.z_values();
    close(z[0], -1.227_851_251_111_118_8, 1e-6);
    close(z[1], 1.736_443_891_898_117, 1e-6);
    let p = fit.p_values();
    close(p[0], 0.219_502_812_283_000_73, 1e-6);
    close(p[1], 0.082_485_377_115_864_68, 1e-6);
    // The trait computes what the fields hold.
    assert_eq!(z, fit.z_values);
    assert_eq!(p, fit.p_values);
    let ci = WaldFit::conf_int(&fit, 0.95).unwrap();
    close(ci[0].lower, -2.199_801_503_669_705_4, 1e-6);
    close(ci[0].upper, 0.505_205_782_895_298, 1e-6);
    close(ci[1].lower, -0.218_133_274_714_729_1, 1e-6);
    close(ci[1].upper, 3.607_324_716_263_544, 1e-6);
    assert!(WaldFit::conf_int(&fit, 0.0).is_err());
    assert!(WaldFit::conf_int(&fit, 1.0).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// MnLogit (M1 — v19 `mnlogit_m1_*`)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mnlogit_likelihood_fit_matches_v19() {
    // llf -22.117379463121267, llnull (closed form) -26.24136735549884, llr 8.247975791118542,
    // prsquared 0.15715598349699078, aic 52.234758926242534, bic 56.946974247634316;
    // n_categories 3, n_params (per equation) 2, nobs 24, df_model 2
    let fit = m1();
    assert_eq!((fit.n_categories, fit.n_params, fit.nobs), (3, 2, 24));
    // The trait's parameter count is the total (k − 1)p the criteria charge for.
    assert_eq!(LikelihoodFit::n_params(&fit), 4);
    assert_eq!(LikelihoodFit::nobs(&fit), 24);
    assert_eq!(LikelihoodFit::df_model(&fit), 2);
    close(
        LikelihoodFit::log_likelihood(&fit),
        -22.117_379_463_121_267,
        1e-8,
    );
    close(
        LikelihoodFit::null_log_likelihood(&fit),
        -26.241_367_355_498_84,
        1e-8,
    );
    close(LikelihoodFit::llr(&fit), 8.247_975_791_118_542, 1e-7);
    rel(
        LikelihoodFit::pseudo_r_squared(&fit),
        0.157_155_983_496_990_78,
        1e-7,
    );
    assert_eq!(LikelihoodFit::pseudo_r_squared(&fit), fit.pseudo_r_squared);
    close(LikelihoodFit::aic(&fit), 52.234_758_926_242_534, 1e-7);
    close(LikelihoodFit::bic(&fit), 56.946_974_247_634_316, 1e-7);
}

#[test]
fn mnlogit_llr_test_matches_v19_and_rejects_df_model_zero() {
    // llr_pvalue 0.016179862014191373 with df_model 2 = (J − 1)(K − 1)
    let ctx = Context::new();
    let fit = m1();
    let t = LikelihoodFit::llr_test(&fit, &ctx).unwrap();
    close(t.statistic_f64().unwrap(), 8.247_975_791_118_542, 1e-7);
    close(t.p_value_f64().unwrap(), 0.016_179_862_014_191_373, 1e-8);
    assert_eq!(t.df, Some(ctx.int(2)));
    assert_eq!(t.alternative, Alternative::Greater);
    assert_eq!(t, fit.llr_test(&ctx).unwrap());
    // Without a constant statsmodels' df_model is 0 and llr_pvalue nan: an error here.
    let (y, x) = m1_data();
    let no_const = mnlogit(&y, &x, false, &LogitOpts::default()).unwrap();
    assert_eq!(LikelihoodFit::df_model(&no_const), 0);
    assert!(matches!(
        LikelihoodFit::llr_test(&no_const, &ctx),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // aic 53.63970316960719, bic 55.99581083030308 with (k − 1)p = 2 parameters.
    assert_eq!(LikelihoodFit::n_params(&no_const), 2);
    close(no_const.aic(), 53.639_703_169_607_19, 1e-7);
    close(no_const.bic(), 55.995_810_830_303_08, 1e-7);
}

#[test]
fn mnlogit_wald_fit_is_the_flattened_view() {
    // params.T [[-0.9175964754951935, 0.10985702371911016], [-2.875029245270399, 0.2536273262970427]]
    // bse.T [[1.0202923117484362, 0.09311682562597019], [1.4432124419606307, 0.1081682371602044]]
    // tvalues.T [[-0.8993466528457352, 1.1797762969324326], [-1.99210397698943, 2.3447486337546914]]
    // pvalues.T [[0.36846804499407193, 0.23808919923338745], [0.04635965070751905, 0.019039911083865633]]
    // conf_int(0.05) [[[-2.9173326602252416, 1.0821397092348544], [-0.07264860086248778, 0.29236264830070813]],
    //   [[-5.703673653553338, -0.04638483698745999], [0.04162147719185494, 0.4656331754022305]]]
    let fit = m1();
    let b = WaldFit::coefficients(&fit);
    let se = WaldFit::standard_errors(&fit);
    assert_eq!((b.len(), se.len()), (4, 4));
    // Flat index (j − 1)·p + a: category 1's intercept and slope, then category 2's.
    assert_eq!(b, fit.coefficients.concat());
    assert_eq!(se, fit.standard_errors.concat());
    rel(b[0], -0.917_596_475_495_193_5, 1e-6);
    rel(b[3], 0.253_627_326_297_042_7, 1e-6);
    rel(se[1], 0.093_116_825_625_970_19, 1e-6);
    rel(se[2], 1.443_212_441_960_630_7, 1e-6);
    let z = fit.z_values();
    rel(z[0], -0.899_346_652_845_735_2, 1e-6);
    rel(z[1], 1.179_776_296_932_432_6, 1e-6);
    rel(z[2], -1.992_103_976_989_43, 1e-6);
    rel(z[3], 2.344_748_633_754_691_4, 1e-6);
    assert_eq!(z, fit.z_values.concat());
    let p = fit.p_values();
    rel(p[0], 0.368_468_044_994_071_93, 1e-6);
    rel(p[3], 0.019_039_911_083_865_633, 1e-6);
    assert_eq!(p, fit.p_values.concat());
    let ci = WaldFit::conf_int(&fit, 0.95).unwrap();
    assert_eq!(ci.len(), 4);
    rel(ci[0].lower, -2.917_332_660_225_241_6, 1e-6);
    rel(ci[0].upper, 1.082_139_709_234_854_4, 1e-6);
    rel(ci[1].lower, -0.072_648_600_862_487_78, 1e-6);
    rel(ci[2].upper, -0.046_384_836_987_459_99, 1e-5);
    rel(ci[3].upper, 0.465_633_175_402_230_5, 1e-6);
    // The inherent version keeps the nested shape and the same numbers.
    let nested = fit.conf_int(0.95).unwrap();
    assert_eq!(nested.len(), 2);
    assert_eq!(nested[0].len(), 2);
    assert_eq!(nested[1][1], ci[3]);
    assert!(fit.conf_int(0.0).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// OrderedLogit (O1 — v19 `ologit_o1_*`)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ologit_likelihood_fit_matches_v19() {
    // llf -20.751965417580625, llnull -21.921346568937103, llr 2.338762302712965,
    // llr_pvalue 0.12618978075284423 (df_model 1), prsquared 0.053344403259127926,
    // aic 47.50393083516125, bic 50.49112765582322; n_params 3, nobs 20
    let ctx = Context::new();
    let fit = o1();
    assert_eq!(
        (
            fit.n_params(),
            LikelihoodFit::nobs(&fit),
            LikelihoodFit::df_model(&fit)
        ),
        (3, 20, 1)
    );
    close(
        LikelihoodFit::log_likelihood(&fit),
        -20.751_965_417_580_625,
        1e-8,
    );
    close(
        LikelihoodFit::null_log_likelihood(&fit),
        -21.921_346_568_937_103,
        1e-8,
    );
    close(LikelihoodFit::llr(&fit), 2.338_762_302_712_965, 1e-8);
    rel(
        LikelihoodFit::pseudo_r_squared(&fit),
        0.053_344_403_259_127_926,
        1e-7,
    );
    assert_eq!(LikelihoodFit::pseudo_r_squared(&fit), fit.pseudo_r_squared);
    close(LikelihoodFit::aic(&fit), 47.503_930_835_161_25, 1e-8);
    close(LikelihoodFit::bic(&fit), 50.491_127_655_823_22, 1e-8);
    let t = LikelihoodFit::llr_test(&fit, &ctx).unwrap();
    close(t.statistic_f64().unwrap(), 2.338_762_302_712_965, 1e-8);
    close(t.p_value_f64().unwrap(), 0.126_189_780_752_844_23, 1e-8);
    assert_eq!(t.df, Some(ctx.int(1)));
    assert_eq!(t.alternative, Alternative::Greater);
    assert_eq!(t, fit.llr_test(&ctx).unwrap());
}

#[test]
fn ologit_wald_fit_covers_the_slopes() {
    // β 0.11402351915116841; exact se[0] 0.07636588220828788 (mpmath); statsmodels' numerical-Hessian
    // z_nat[0] 1.4931191054308977, p_nat[0] 0.13540601228098428 (1e-5);
    // conf_int(0.05)[:1] = [[-0.03565107151874923, 0.26369810982108605]]
    let fit = o1();
    assert_eq!(WaldFit::coefficients(&fit).len(), 1);
    assert_eq!(WaldFit::standard_errors(&fit).len(), 1);
    // The fields carry the thresholds' Wald quantities too.
    assert_eq!(fit.standard_errors.len(), 3);
    rel(
        WaldFit::coefficients(&fit)[0],
        0.114_023_519_151_168_41,
        1e-6,
    );
    rel(
        WaldFit::standard_errors(&fit)[0],
        0.076_365_882_208_287_88,
        1e-9,
    );
    let z = fit.z_values();
    assert_eq!(z.len(), 1);
    rel(z[0], 1.493_119_105_430_897_7, 1e-5);
    assert_eq!(z[0], fit.z_values[0]);
    let p = fit.p_values();
    rel(p[0], 0.135_406_012_280_984_28, 1e-5);
    assert_eq!(p[0], fit.p_values[0]);
    let ci = WaldFit::conf_int(&fit, 0.95).unwrap();
    assert_eq!(ci.len(), 1);
    close(ci[0].lower, -0.035_651_071_518_749_23, 1e-6);
    close(ci[0].upper, 0.263_698_109_821_086_05, 1e-6);
    assert_eq!(ci, fit.conf_int(0.95).unwrap());
}

// ═══════════════════════════════════════════════════════════════════════════
// CoxModel (D1, D2 — v18 `d1_*`, `d2_efron_*`)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cox_likelihood_fit_matches_v18() {
    // llf -8.465216276861835, m.loglike(zeros) -12.108680299521524, LR 7.286928045319378,
    // AIC 18.93043255372367; nobs 10, n_events 7, p 1
    let fit = d1();
    assert_eq!((fit.n_params(), fit.nobs, fit.n_events), (1, 10, 7));
    assert_eq!(LikelihoodFit::n_params(&fit), 1);
    assert_eq!(LikelihoodFit::nobs(&fit), 10);
    assert_eq!(LikelihoodFit::df_model(&fit), 1);
    close(
        LikelihoodFit::log_likelihood(&fit),
        -8.465_216_276_861_835,
        1e-9,
    );
    close(
        LikelihoodFit::null_log_likelihood(&fit),
        -12.108_680_299_521_524,
        1e-9,
    );
    close(LikelihoodFit::llr(&fit), 7.286_928_045_319_378, 1e-8);
    close(LikelihoodFit::aic(&fit), 18.930_432_553_723_67, 1e-8);
    // 1 − llf/llnull = 0.3008968717097652 (the partial-likelihood McFadden ratio).
    close(
        LikelihoodFit::pseudo_r_squared(&fit),
        0.300_896_871_709_765_2,
        1e-9,
    );
}

#[test]
fn cox_bic_counts_events_as_r_does() {
    // R: BIC(coxph) = −2ℓ + p·ln(nevent) (logLik.coxph: nobs = nevent); statsmodels PHRegResults has no bic.
    // D1: 2 · 8.465216276861835 + ln 7 = 18.87634270277898  (ln 10 would give 19.233017646717716)
    let fit = d1();
    assert_eq!(fit.n_events, 7);
    close(fit.bic(), 18.876_342_702_778_98, 1e-8);
    close(
        fit.bic(),
        -2.0 * fit.log_likelihood + (fit.n_events as f64).ln(),
        1e-12,
    );
    assert!((fit.bic() - 19.233_017_646_717_716).abs() > 0.3);
    assert_eq!(LikelihoodFit::bic(&fit), fit.bic());
    // D2 (Efron): llf -15.16722416403536, 8 events, 2 covariates: 30.33444832807072 + 2 ln 8 = 34.49333141143039
    let fit = d2();
    assert_eq!((fit.n_params(), fit.n_events), (2, 8));
    close(fit.bic(), 34.493_331_411_430_39, 1e-8);
    close(fit.aic(), 34.334_448_328_070_72, 1e-8);
}

#[test]
fn cox_llr_test_keeps_its_two_sided_label_and_v18_values() {
    // LR 7.286928045319378; chi2.sf(LR, 1) = 0.006945814500177098
    let ctx = Context::new();
    let fit = d1();
    let lr = LikelihoodFit::llr_test(&fit, &ctx).unwrap();
    close(lr.statistic_f64().unwrap(), 7.286_928_045_319_378, 1e-8);
    close(lr.p_value_f64().unwrap(), 0.006_945_814_500_177_098, 1e-10);
    assert_eq!(lr.df, Some(ctx.int(1)));
    // Reported like the model's Wald and score tests.
    assert_eq!(lr.alternative, Alternative::TwoSided);
    assert_eq!(lr, fit.llr_test(&ctx).unwrap());
    assert_eq!(
        fit.wald_test(&ctx).unwrap().alternative,
        Alternative::TwoSided
    );
    assert_eq!(
        fit.score_test(&ctx).unwrap().alternative,
        Alternative::TwoSided
    );
    // D2: LR 0.8759273939052861, chi2.sf(LR, 2) = 0.6453492105749965
    let fit = d2();
    let lr = fit.llr_test(&ctx).unwrap();
    close(lr.statistic_f64().unwrap(), 0.875_927_393_905_286_1, 1e-8);
    close(lr.p_value_f64().unwrap(), 0.645_349_210_574_996_5, 1e-10);
    assert_eq!(lr.df, Some(ctx.int(2)));
}

#[test]
fn cox_wald_fit_matches_v18() {
    // D1: params [0.8759809887649096], bse [0.3809028296542367], tvalues [2.299749228852089],
    //   pvalues [0.021462431348704628], conf_int(0.05) [[0.12942516103321033, 1.6225368164966087]]
    let fit = d1();
    close(
        WaldFit::coefficients(&fit)[0],
        0.875_980_988_764_909_6,
        1e-9,
    );
    close(
        WaldFit::standard_errors(&fit)[0],
        0.380_902_829_654_236_7,
        1e-9,
    );
    close(fit.z_values()[0], 2.299_749_228_852_089, 1e-8);
    close(fit.p_values()[0], 0.021_462_431_348_704_628, 1e-9);
    assert_eq!(fit.z_values(), fit.z_values);
    assert_eq!(fit.p_values(), fit.p_values);
    let ci = WaldFit::conf_int(&fit, 0.95).unwrap();
    close(ci[0].lower, 0.129_425_161_033_210_33, 1e-8);
    close(ci[0].upper, 1.622_536_816_496_608_7, 1e-8);
    assert_eq!(ci, fit.conf_int(0.95).unwrap());
    assert!(matches!(
        WaldFit::conf_int(&fit, 1.0),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // D2 (Efron): tvalues [-0.14848179832312458, 0.7694226841966265], pvalues [0.8819625495604642, 0.4416424264729595]
    let fit = d2();
    let z = fit.z_values();
    close(z[0], -0.148_481_798_323_124_58, 1e-8);
    close(z[1], 0.769_422_684_196_626_5, 1e-8);
    let p = fit.p_values();
    close(p[0], 0.881_962_549_560_464_2, 1e-9);
    close(p[1], 0.441_642_426_472_959_5, 1e-9);
}

// ═══════════════════════════════════════════════════════════════════════════
// Family boilerplate: `Distribution::eq`, names and parameters unchanged
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn distribution_equality_is_structural_per_family() {
    let ctx = Context::new();
    let n01 = Distribution::normal(ctx.int(0), ctx.int(1));
    assert_eq!(n01, Distribution::normal(ctx.int(0), ctx.int(1)));
    assert_ne!(n01, Distribution::normal(ctx.int(0), ctx.int(2)));
    // A different family with the same parameters is not equal.
    assert_ne!(n01, Distribution::uniform(ctx.int(0), ctx.int(1)));
    assert_ne!(n01, Distribution::laplace(ctx.int(0), ctx.int(1)));
    let b = Distribution::binomial(ctx.int(5), ctx.rational(1, 3));
    assert_eq!(b, Distribution::binomial(ctx.int(5), ctx.rational(1, 3)));
    assert_ne!(b, Distribution::binomial(ctx.int(5), ctx.rational(1, 2)));
    assert_ne!(b, Distribution::poisson(ctx.int(5)));
    // Symbolic parameters compare structurally.
    let mu = ctx.symbol("mu");
    assert_eq!(
        Distribution::normal(mu.clone(), ctx.int(1)),
        Distribution::normal(ctx.symbol("mu"), ctx.int(1))
    );
    assert_ne!(
        Distribution::normal(mu, ctx.int(1)),
        Distribution::normal(ctx.symbol("nu"), ctx.int(1))
    );
    assert_eq!(n01.name(), "Normal");
    assert_eq!(b.name(), "Binomial");
    assert_eq!(
        b.parameters(),
        vec![("n", ctx.int(5)), ("p", ctx.rational(1, 3))]
    );
}

#[test]
fn finite_table_equality_and_parameters() {
    let ctx = Context::new();
    let table = || {
        vec![
            (ctx.int(1), ctx.rational(1, 4)),
            (ctx.int(2), ctx.rational(3, 4)),
        ]
    };
    let f = Distribution::finite(&ctx, table());
    assert_eq!(f, Distribution::finite(&ctx, table()));
    assert_ne!(
        f,
        Distribution::finite(
            &ctx,
            vec![
                (ctx.int(1), ctx.rational(1, 2)),
                (ctx.int(2), ctx.rational(1, 2)),
            ]
        )
    );
    assert_ne!(f, Distribution::die(ctx.int(2)));
    assert_eq!(f.name(), "Finite");
    assert_eq!(
        f.parameters(),
        vec![
            ("value", ctx.int(1)),
            ("probability", ctx.rational(1, 4)),
            ("value", ctx.int(2)),
            ("probability", ctx.rational(3, 4)),
        ]
    );
    assert_eq!(f.mean(), ctx.rational(7, 4));
}

#[test]
fn wrapper_equality_and_names() {
    let ctx = Context::new();
    let n01 = Distribution::normal(ctx.int(0), ctx.int(1));
    let positive = Support::from_pieces(
        Kind::Continuous,
        vec![Piece::Interval(Interval::open(ctx.int(0), ctx.infinity()))],
    );
    let above_one = Support::from_pieces(
        Kind::Continuous,
        vec![Piece::Interval(Interval::open(ctx.int(1), ctx.infinity()))],
    );
    let half = n01.truncated(&positive).unwrap();
    assert_eq!(half, n01.truncated(&positive).unwrap());
    assert_ne!(half, n01.truncated(&above_one).unwrap());
    assert_ne!(half, n01);
    assert_eq!(half.name(), "Truncated");
    // Parameters: the inner family's, then the normalising mass.
    let params = half.parameters();
    assert_eq!(params.len(), 3);
    assert_eq!(params[0], ("mean", ctx.int(0)));
    assert_eq!(params[1], ("std", ctx.int(1)));
    assert_eq!(params[2].0, "mass");
    assert_eq!(params[2].1.equals(&ctx.rational(1, 2)), Some(true));

    let a = n01.affine(ctx.int(2), ctx.int(1)).unwrap();
    assert_eq!(a, n01.affine(ctx.int(2), ctx.int(1)).unwrap());
    assert_ne!(a, n01.affine(ctx.int(2), ctx.int(0)).unwrap());
    assert_ne!(a, Distribution::normal(ctx.int(1), ctx.int(2)));
    assert_eq!(a.name(), "Affine");
    assert_eq!(a.parameters()[..2], [("a", ctx.int(2)), ("b", ctx.int(1))]);

    let x = ctx.symbol("x");
    let t = n01.transformed(&x, &x.exp()).unwrap();
    assert_eq!(t, n01.transformed(&x, &x.exp()).unwrap());
    assert_ne!(t, n01.transformed(&x, &x.powi(3)).unwrap());
    assert_eq!(t.name(), "Transformed");
    assert_eq!(t.parameters()[0], ("map", x.exp()));

    let m = Distribution::mixture(&[
        (ctx.rational(1, 2), n01.clone()),
        (
            ctx.rational(1, 2),
            Distribution::normal(ctx.int(3), ctx.int(1)),
        ),
    ])
    .unwrap();
    assert_eq!(
        m,
        Distribution::mixture(&[
            (ctx.rational(1, 2), n01.clone()),
            (
                ctx.rational(1, 2),
                Distribution::normal(ctx.int(3), ctx.int(1))
            ),
        ])
        .unwrap()
    );
    assert_ne!(
        m,
        Distribution::mixture(&[
            (ctx.rational(1, 4), n01.clone()),
            (
                ctx.rational(3, 4),
                Distribution::normal(ctx.int(3), ctx.int(1))
            ),
        ])
        .unwrap()
    );
    assert_eq!(m.name(), "Mixture");
    assert_eq!(m.parameters()[0], ("weight", ctx.rational(1, 2)));
    assert_eq!(m.parameters().len(), 6);

    let u = Distribution::uniform(ctx.int(0), ctx.int(1));
    let second_of_five = order_statistic(&u, 5, 2).unwrap();
    assert_eq!(second_of_five, order_statistic(&u, 5, 2).unwrap());
    assert_ne!(second_of_five, order_statistic(&u, 5, 1).unwrap());
    assert_ne!(second_of_five, order_statistic(&n01, 5, 2).unwrap());
    assert_eq!(second_of_five.name(), "OrderStatistic");
    assert_eq!(
        second_of_five.parameters(),
        vec![
            ("n", ctx.int(5)),
            ("k", ctx.int(2)),
            ("lo", ctx.int(0)),
            ("hi", ctx.int(1)),
        ]
    );
    assert_eq!(
        second_of_five.to_string(),
        "OrderStatistic(k=2, n=5, Uniform(0, 1))"
    );
}

#[test]
fn order_statistic_parent_cdf_goes_through_cdf_on_support() {
    let ctx = Context::new();
    // Uniform(0, 1): P(X_(2) ≤ 1/2) = I_{1/2}(2, 4) = 13/16, mean 2/6.
    let u = Distribution::uniform(ctx.int(0), ctx.int(1));
    let x2 = order_statistic(&u, 5, 2).unwrap();
    assert_eq!(x2.cdf(&ctx.rational(1, 2)), ctx.rational(13, 16));
    assert_eq!(x2.mean(), ctx.rational(1, 3));
    // Outside the support the clamps still decide first.
    assert_eq!(x2.cdf(&ctx.int(-1)), ctx.int(0));
    assert_eq!(x2.cdf(&ctx.int(2)), ctx.int(1));
    // Maximum of three exponentials: F(x)³ with F = 1 − e^{−x}; the mean is 1 + 1/2 + 1/3.
    let e = Distribution::exponential(ctx.int(1));
    let m3 = order_statistic(&e, 3, 3).unwrap();
    assert_eq!(m3.mean().simplify(), ctx.rational(11, 6));
    close(
        m3.cdf(&ctx.int(1)).eval_f64().unwrap(),
        (1.0 - (-1.0_f64).exp()).powi(3),
        1e-12,
    );
    // A lattice parent (Geometric on 1..∞): the minimum of two is Geometric(1 − (1 − p)²).
    let g = Distribution::geometric(ctx.rational(1, 3));
    let min2 = order_statistic(&g, 2, 1).unwrap();
    // P(min ≤ 1) = 1 − (2/3)² = 5/9, P(min ≤ 2) = 1 − (4/9)² = 65/81.
    assert_eq!(min2.cdf(&ctx.int(1)).eval(), ctx.rational(5, 9));
    assert_eq!(min2.cdf(&ctx.int(2)).eval(), ctx.rational(65, 81));
    // P(min = 2) = 65/81 − 5/9 = 20/81.
    assert_eq!(min2.density(&ctx.int(2)).eval(), ctx.rational(20, 81));
}

#[test]
fn accumulate_and_mass_below_keep_probabilities_exact() {
    let ctx = Context::new();
    // Closed-form route through `mass_below`: P(2 < X) for Binomial(5, 1/3) = 17/81 (the
    // lattice atom at the region's lower end 3 belongs to the region: F(5) − F(2)).
    let b = Distribution::binomial(ctx.int(5), ctx.rational(1, 3));
    let above_two = Support::from_pieces(
        Kind::Continuous,
        vec![Piece::Interval(Interval::open(ctx.int(2), ctx.infinity()))],
    );
    assert_eq!(b.probability_of(&above_two).unwrap(), ctx.rational(17, 81));
    // …and P(X ≥ 2) = 1 − F(1) = 1 − (32 + 80)/243 = 131/243.
    let at_least_two = Support::from_pieces(
        Kind::Continuous,
        vec![Piece::Interval(Interval::closed(
            ctx.int(2),
            ctx.infinity(),
        ))],
    );
    assert_eq!(
        b.probability_of(&at_least_two).unwrap(),
        ctx.rational(131, 243)
    );
    // Truncation uses the same `mass_below`: Binomial(5, 1/3) | X > 2 has mass 17/81 and
    // P(X = 3 | X > 2) = (40/243) / (17/81) = 40/51.
    let given = b.truncated(&above_two).unwrap();
    assert_eq!(given.density(&ctx.int(3)).simplify(), ctx.rational(40, 51));
    assert_eq!(given.cdf(&ctx.int(3)), ctx.rational(40, 51));
    // `mass_below` on a lattice family with a closed-form CDF: P(X > 2) for Geometric(1/2)
    // is 1 − F(2) = 1/4, and X | X > 2 has P(X ≤ 4 | X > 2) = (F(4) − F(2)) / (1/4) = 3/4.
    let g = Distribution::geometric(ctx.rational(1, 2));
    assert_eq!(g.probability_of(&above_two).unwrap(), ctx.rational(1, 4));
    let tail = g.truncated(&above_two).unwrap();
    assert_eq!(tail.cdf(&ctx.int(4)), ctx.rational(3, 4));
    assert_eq!(tail.cdf(&ctx.int(2)), ctx.int(0));
    // `mass_below` on a density: N(0, 1) | X > 0 has F(1) = (Φ(1) − Φ(0)) / (1/2) = erf(1/√2)
    // = 0.6826894921370859 (scipy: erf(1/sqrt(2))).
    let n01 = Distribution::normal(ctx.int(0), ctx.int(1));
    let positive = Support::from_pieces(
        Kind::Continuous,
        vec![Piece::Interval(Interval::open(ctx.int(0), ctx.infinity()))],
    );
    let half = n01.truncated(&positive).unwrap();
    close(
        half.cdf(&ctx.int(1)).eval_f64().unwrap(),
        0.682_689_492_137_085_9,
        1e-12,
    );
    // Integration route (`accumulate` on a density without a closed-form CDF): Z² is χ²(1),
    // whose mean 1 comes from integrating y · e^{−y/2}/√(2πy) over (0, ∞).
    let x = ctx.symbol("x");
    let sq = n01.transformed(&x, &x.powi(2)).unwrap();
    assert_eq!(sq.mean(), ctx.int(1));
    // P(X > 2) for Exponential(1) is e⁻² = 0.1353352832366127 (closed form through `mass_below`).
    let e = Distribution::exponential(ctx.int(1));
    close(
        e.probability_of(&above_two).unwrap().eval_f64().unwrap(),
        0.135_335_283_236_612_7,
        1e-12,
    );
}
