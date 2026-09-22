//! symplex 0.18 — Cox proportional hazards (`stats::cox`).  Every reference
//! value cites statsmodels 0.15 `PHReg` (`symplex/.venv/bin/python`),
//! scipy 1.18 (`chi2.sf`, `norm.sf`, `norm.isf`) or `fractions.Fraction`
//! over an explicit pair enumeration; the Python is quoted in each test.
//! `lifelines` is not installed and was not used.
//!
//! Common preamble of the quoted Python:
//! ```python
//! import numpy as np
//! from statsmodels.duration.hazard_regression import PHReg
//! from scipy.stats import chi2, norm
//! def fit(t, s, X, ties): return PHReg(np.array(t, float), np.array(X, float), status=np.array(s), ties=ties).fit(method="newton", maxiter=200, tol=1e-12)
//! ```

use symplex::linprog::{q, qi};
use symplex::prelude::*;
use symplex::stats::cox::{BaselineHazardRow, CoxModel, CoxOpts, Ties, cox_ph, cox_ph_stratified};
use symplex::stats::survival::{Observation, log_rank_test};

fn close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() < tol,
        "got {actual}, expected {expected} (tol {tol})"
    );
}

fn close_all(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len(), "lengths differ");
    for (a, e) in actual.iter().zip(expected) {
        close(*a, *e, tol);
    }
}

fn opts(ties: Ties) -> CoxOpts {
    CoxOpts {
        ties,
        ..CoxOpts::default()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// D1: ten subjects, one covariate, no tied event times, three censored.
//   t1 = [4, 7, 2, 9, 12, 5, 15, 3, 11, 8]; s1 = [1, 1, 1, 0, 1, 1, 0, 1, 1, 0]
//   x1 = [[3], [1], [5], [2], [0], [4], [1], [6], [2], [3]]
// ═══════════════════════════════════════════════════════════════════════════

fn d1() -> (Vec<Observation>, Vec<Vec<f64>>) {
    let obs = Observation::from_i64(
        &[4, 7, 2, 9, 12, 5, 15, 3, 11, 8],
        &[
            true, true, true, false, true, true, false, true, true, false,
        ],
    );
    let x = [3.0, 1.0, 5.0, 2.0, 0.0, 4.0, 1.0, 6.0, 2.0, 3.0]
        .iter()
        .map(|&v| vec![v])
        .collect();
    (obs, x)
}

#[test]
fn d1_efron_coefficient_and_standard_error_match_statsmodels() {
    // r = fit(t1, s1, x1, "efron"); r.params → [0.8759809887649096]; r.bse → [0.3809028296542367]
    let (obs, x) = d1();
    let fit = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    assert_eq!(fit.n_params(), 1);
    assert_eq!(fit.nobs, 10);
    assert_eq!(fit.n_events, 7);
    assert_eq!(fit.ties, Ties::Efron);
    close(fit.coefficients[0], 0.8759809887649096, 1e-9);
    close(fit.standard_errors[0], 0.3809028296542367, 1e-9);
    // cov_params = bse²
    close(
        fit.cov_params[0][0],
        0.3809028296542367 * 0.3809028296542367,
        1e-9,
    );
}

#[test]
fn d1_z_and_p_values_match_statsmodels() {
    // r.tvalues → [2.299749228852089]; r.pvalues → [0.021462431348704628]
    // scipy: 2 * norm.sf(2.299749228852089) = 0.021462431348704628
    let (obs, x) = d1();
    let fit = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    close(fit.z_values[0], 2.299749228852089, 1e-8);
    close(fit.p_values[0], 0.021462431348704628, 1e-9);
}

#[test]
fn d1_log_likelihoods_llr_and_aic_match_statsmodels() {
    // r.llf → -8.465216276861835; m.loglike(np.zeros(1)) → -12.108680299521524
    // 2 * (llf - llnull) = 7.286928045319378; -2 * llf + 2 = 18.93043255372367
    let (obs, x) = d1();
    let fit = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    close(fit.log_likelihood, -8.465216276861835, 1e-9);
    close(fit.null_log_likelihood, -12.108680299521524, 1e-9);
    close(fit.llr(), 7.286928045319378, 1e-8);
    close(fit.aic(), 18.93043255372367, 1e-8);
}

#[test]
fn d1_breslow_equals_efron_when_no_event_times_are_tied() {
    // fit(t1, s1, x1, "breslow").params → [0.8759809887649093], bse [0.38090282965423605],
    // llf -8.465216276861836 — the two corrections coincide without ties.
    let (obs, x) = d1();
    let efron = cox_ph(&obs, &x, &opts(Ties::Efron)).unwrap();
    let breslow = cox_ph(&obs, &x, &opts(Ties::Breslow)).unwrap();
    assert_eq!(breslow.ties, Ties::Breslow);
    close(breslow.coefficients[0], 0.8759809887649093, 1e-9);
    close(breslow.standard_errors[0], 0.38090282965423605, 1e-9);
    close(breslow.log_likelihood, -8.465216276861836, 1e-9);
    close(breslow.coefficients[0], efron.coefficients[0], 1e-12);
    close(breslow.log_likelihood, efron.log_likelihood, 1e-12);
    close(
        breslow.null_log_likelihood,
        efron.null_log_likelihood,
        1e-12,
    );
    close(breslow.score_statistic(), efron.score_statistic(), 1e-12);
}

#[test]
fn d1_hazard_ratios_and_wald_intervals_match_statsmodels() {
    // np.exp(r.params) → [2.401229718322006]
    // r.conf_int(alpha=0.05) → [[0.12942516103321033, 1.6225368164966087]]
    // np.exp(...) → [[1.1381739285144856, 5.065925352620126]]
    // r.conf_int(alpha=0.10) → [[0.24945158789205968, 1.5025103896377594]]
    // np.exp(...) → [[1.2833214346561057, 4.492953989892548]]
    let (obs, x) = d1();
    let fit = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    close(fit.hazard_ratios()[0], 2.401229718322006, 1e-8);
    let ci = fit.conf_int(0.95).unwrap();
    close(ci[0].lower, 0.12942516103321033, 1e-8);
    close(ci[0].upper, 1.6225368164966087, 1e-8);
    let hr = fit.hazard_ratio_conf_int(0.95).unwrap();
    close(hr[0].lower, 1.1381739285144856, 1e-8);
    close(hr[0].upper, 5.065925352620126, 1e-8);
    let ci90 = fit.conf_int(0.90).unwrap();
    close(ci90[0].lower, 0.24945158789205968, 1e-8);
    close(ci90[0].upper, 1.5025103896377594, 1e-8);
    let hr90 = fit.hazard_ratio_conf_int(0.90).unwrap();
    close(hr90[0].lower, 1.2833214346561057, 1e-8);
    close(hr90[0].upper, 4.492953989892548, 1e-8);
    assert!(matches!(
        fit.conf_int(1.0),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        fit.hazard_ratio_conf_int(0.0),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

#[test]
fn d1_likelihood_ratio_wald_and_score_tests_match_statsmodels() {
    // LR = 2 * (r.llf - llnull) = 7.286928045319378; chi2.sf(LR, 1) = 0.006945814500177098
    // Wald = params @ solve(cov_params, params) = 5.28884651560578; chi2.sf(Wald, 1) = 0.02146243134870459
    // Score = g0 @ solve(-H0, g0) with g0 = m.score(zeros), H0 = m.hessian(zeros):
    //   7.355733707724492; chi2.sf(Score, 1) = 0.006684922420334704
    let (obs, x) = d1();
    let fit = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    let ctx = Context::new();
    let lr = fit.llr_test(&ctx).unwrap();
    close(lr.statistic_f64().unwrap(), 7.286928045319378, 1e-8);
    close(lr.p_value_f64().unwrap(), 0.006945814500177098, 1e-10);
    assert_eq!(lr.df, Some(ctx.int(1)));
    let wald = fit.wald_test(&ctx).unwrap();
    close(fit.wald_statistic(), 5.28884651560578, 1e-8);
    close(wald.statistic_f64().unwrap(), 5.28884651560578, 1e-8);
    close(wald.p_value_f64().unwrap(), 0.02146243134870459, 1e-10);
    // The Wald test of a single coefficient is z².
    close(
        fit.wald_statistic(),
        fit.z_values[0] * fit.z_values[0],
        1e-9,
    );
    let score = fit.score_test(&ctx).unwrap();
    close(fit.score_statistic(), 7.355733707724492, 1e-8);
    close(score.statistic_f64().unwrap(), 7.355733707724492, 1e-8);
    close(score.p_value_f64().unwrap(), 0.006684922420334704, 1e-10);
    assert_eq!(score.df, Some(ctx.int(1)));
}

#[test]
fn d1_baseline_hazard_is_the_breslow_estimator() {
    // With eta = x1 @ r.params and w = np.exp(eta):
    //   uft = [2, 3, 4, 5, 7, 11, 12]
    //   h0 = [s1[t1 == u].sum() / w[t1 >= u].sum() for u in uft]
    //      → [0.002858840843475405, 0.0037042294909749073, 0.012776215488862453, 0.015521881601113358,
    //         0.03207232624718611, 0.1090853491676483, 0.2940113085020759]
    //   np.cumsum(h0) → [0.002858840843475405, 0.006563070334450312, 0.019339285823312763, 0.03486116742442612,
    //         0.06693349367161222, 0.1760188428392605, 0.4700301513413364]
    // and r.baseline_cumulative_hazard[0][1] (statsmodels' *exclusive* cumsum(h0) - h0)
    //      → [0.0, 0.002858840843475406, 0.006563070334450314, 0.019339285823312766, 0.034861167424426125,
    //         0.06693349367161223, 0.1760188428392605], i.e. the same increments.
    let (obs, x) = d1();
    let fit = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    let rows = fit.baseline_hazard();
    let times: Vec<Q> = rows.iter().map(|r| r.time.clone()).collect();
    assert_eq!(
        times,
        vec![qi(2), qi(3), qi(4), qi(5), qi(7), qi(11), qi(12)]
    );
    assert!(rows.iter().all(|r| r.stratum == 0));
    let hazard: Vec<f64> = rows.iter().map(|r| r.hazard).collect();
    close_all(
        &hazard,
        &[
            0.002858840843475405,
            0.0037042294909749073,
            0.012776215488862453,
            0.015521881601113358,
            0.03207232624718611,
            0.1090853491676483,
            0.2940113085020759,
        ],
        1e-9,
    );
    let cumulative: Vec<f64> = rows.iter().map(|r| r.cumulative).collect();
    close_all(
        &cumulative,
        &[
            0.002858840843475405,
            0.006563070334450312,
            0.019339285823312763,
            0.03486116742442612,
            0.06693349367161222,
            0.1760188428392605,
            0.4700301513413364,
        ],
        1e-9,
    );
    // Exclusive form: statsmodels' cumhaz[k] = cumulative[k] - hazard[k].
    close(
        rows[6].cumulative - rows[6].hazard,
        0.1760188428392605,
        1e-9,
    );
}

#[test]
fn d1_martingale_residuals_sum_to_zero_and_match_the_breslow_hazard() {
    // Lam = [h0[uft <= ti].sum() for ti in t1]; mart = s1 - w * Lam
    //   → [0.7322425513938701, 0.8392773058446068, 0.7717774642312979, -0.38593210961470864, 0.5299698486586636,
    //      -0.15898219811736802, -1.1286503679082072, -0.2580830654344388, -0.014907778192170484, -0.9267116508615444]
    // mart.sum() → 9.992007221626409e-16.  (statsmodels' r.martingale_residuals uses the exclusive cumulative
    // hazard and gives [0.9091325820493006, 0.916290328765069, 1.0, ...] — a different quantity.)
    let (obs, x) = d1();
    let fit = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    let m = fit.martingale_residuals();
    close_all(
        &m,
        &[
            0.7322425513938701,
            0.8392773058446068,
            0.7717774642312979,
            -0.38593210961470864,
            0.5299698486586636,
            -0.15898219811736802,
            -1.1286503679082072,
            -0.2580830654344388,
            -0.014907778192170484,
            -0.9267116508615444,
        ],
        1e-9,
    );
    close(m.iter().sum::<f64>(), 0.0, 1e-12);
}

#[test]
fn d1_schoenfeld_residuals_match_statsmodels() {
    // r.schoenfeld_residuals[s1 == 1] (events in observation order; censored rows are NaN there)
    //   → [[-0.11637757066347332], [-1.2258790109077378], [-0.12653024017379], [-0.7059886914979241],
    //      [0.8586123665150809], [0.8360534661376562], [0.4801096805901861]]
    let (obs, x) = d1();
    let fit = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    let s = fit.schoenfeld_residuals();
    assert_eq!(s.len(), 7);
    let flat: Vec<f64> = s.iter().map(|r| r[0]).collect();
    close_all(
        &flat,
        &[
            -0.11637757066347332,
            -1.2258790109077378,
            -0.12653024017379,
            -0.7059886914979241,
            0.8586123665150809,
            0.8360534661376562,
            0.4801096805901861,
        ],
        1e-9,
    );
    // The score equation: Schoenfeld residuals sum to zero at the MLE.
    close(flat.iter().sum::<f64>(), 0.0, 1e-9);
}

#[test]
fn d1_concordance_is_exactly_31_over_38() {
    // β̂ > 0, so η orders as x.  Usable pairs (t_i < t_j, event i), '+' concordant (x_i > x_j), '=' tie:
    //   i=0 (t=4, x=3): 1+ 3+ 4+ 5- 6+ 8+ 9=   → 7 usable, 5 concordant, 1 tied
    //   i=1 (t=7, x=1): 3- 4+ 6= 8- 9-         → 5 usable, 1 concordant, 1 tied
    //   i=2 (t=2, x=5): 0+ 1+ 3+ 4+ 5+ 6+ 7- 8+ 9+ → 9 usable, 8 concordant
    //   i=4 (t=12, x=0): 6-                    → 1 usable, 0 concordant
    //   i=5 (t=5, x=4): 1+ 3+ 4+ 6+ 8+ 9+      → 6 usable, 6 concordant
    //   i=7 (t=3, x=6): 0+ 1+ 3+ 4+ 5+ 6+ 8+ 9+ → 8 usable, 8 concordant
    //   i=8 (t=11, x=2): 4+ 6+                 → 2 usable, 2 concordant
    // Totals: 38 usable, 30 concordant, 2 tied → Fraction(30 + 1, 38) = 31/38.
    let (obs, x) = d1();
    let fit = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    assert_eq!(fit.concordance().unwrap(), q(31, 38));
}

#[test]
fn d1_predictions_are_exp_of_the_linear_predictor() {
    // np.exp(2 * r.params[0]) → 5.76590416015278; np.exp(3 * r.params[0]) → 13.845260422355341
    let (obs, x) = d1();
    let fit = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    close(
        fit.predict_partial_hazard(&[2.0]).unwrap(),
        5.76590416015278,
        1e-8,
    );
    close(
        fit.predict_partial_hazard(&[3.0]).unwrap(),
        13.845260422355341,
        1e-8,
    );
    close(fit.predict_partial_hazard(&[0.0]).unwrap(), 1.0, 1e-15);
    close(
        fit.predict_log_partial_hazard(&[2.0]).unwrap(),
        2.0 * 0.8759809887649096,
        1e-8,
    );
    let eta = fit.linear_predictors();
    assert_eq!(eta.len(), 10);
    close(eta[7], 6.0 * 0.8759809887649096, 1e-8);
    assert!(matches!(
        fit.predict_partial_hazard(&[1.0, 2.0]),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        fit.predict_partial_hazard(&[f64::NAN]),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

#[test]
fn d1_newton_converges_in_a_handful_of_steps() {
    let (obs, x) = d1();
    let fit = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    assert!(fit.converged);
    assert!(
        fit.iterations >= 2 && fit.iterations <= 10,
        "{}",
        fit.iterations
    );
    assert!(fit.strata().iter().all(|&s| s == 0));
}

// ═══════════════════════════════════════════════════════════════════════════
// D2: twelve subjects, two covariates (continuous, binary), tied event times
// at 3 (two events), 5 (two events + a censoring), 8 (event + censoring), 12.
//   t2 = [3, 3, 5, 5, 5, 7, 8, 8, 10, 12, 12, 14]; s2 = [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0]
//   x2 = [[2,1],[4,0],[1,1],[3,0],[5,1],[2,0],[6,1],[1,0],[3,1],[0,0],[4,1],[2,0]]
// ═══════════════════════════════════════════════════════════════════════════

fn d2() -> (Vec<Observation>, Vec<Vec<f64>>) {
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
    (obs, x)
}

#[test]
fn d2_efron_coefficients_match_statsmodels() {
    // r = fit(t2, s2, x2, "efron"): params [-0.04053799780180025, 0.767190075048902]
    //   bse [0.27301661388543985, 0.9970983320435169]; tvalues [-0.14848179832312458, 0.7694226841966265]
    //   pvalues [0.8819625495604642, 0.4416424264729595]; np.exp(params) [0.9602726755673672, 2.153705991128979]
    let (obs, x) = d2();
    let fit = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    assert_eq!(fit.n_events, 8);
    close_all(
        &fit.coefficients,
        &[-0.04053799780180025, 0.767190075048902],
        1e-9,
    );
    close_all(
        &fit.standard_errors,
        &[0.27301661388543985, 0.9970983320435169],
        1e-9,
    );
    close_all(
        &fit.z_values,
        &[-0.14848179832312458, 0.7694226841966265],
        1e-8,
    );
    close_all(
        &fit.p_values,
        &[0.8819625495604642, 0.4416424264729595],
        1e-9,
    );
    close_all(
        &fit.hazard_ratios(),
        &[0.9602726755673672, 2.153705991128979],
        1e-8,
    );
}

#[test]
fn d2_breslow_coefficients_match_statsmodels_and_differ_from_efron() {
    // r = fit(t2, s2, x2, "breslow"): params [-0.027362967515010373, 0.704881586783013]
    //   bse [0.2681669289501311, 0.9793059340341044]; np.exp(params) [0.9730080071234777, 2.023607048914013]
    let (obs, x) = d2();
    let fit = cox_ph(&obs, &x, &opts(Ties::Breslow)).unwrap();
    close_all(
        &fit.coefficients,
        &[-0.027362967515010373, 0.704881586783013],
        1e-9,
    );
    close_all(
        &fit.standard_errors,
        &[0.2681669289501311, 0.9793059340341044],
        1e-9,
    );
    close_all(
        &fit.hazard_ratios(),
        &[0.9730080071234777, 2.023607048914013],
        1e-8,
    );
    let efron = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    assert!((efron.coefficients[1] - fit.coefficients[1]).abs() > 0.05);
}

#[test]
fn d2_efron_log_likelihoods_and_the_test_trio_match_statsmodels() {
    // Efron: llf -15.16722416403536; llnull -15.605187860988003
    //   LR 0.8759273939052861, chi2.sf(LR, 2) = 0.6453492105749965
    //   Wald 0.8445471219237513, chi2.sf(Wald, 2) = 0.6555546806927786
    //   Score 0.8746958525820299, chi2.sf(Score, 2) = 0.6457467200601523
    //   AIC = -2 llf + 4 = 34.33444832807072
    let (obs, x) = d2();
    let fit = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    close(fit.log_likelihood, -15.16722416403536, 1e-9);
    close(fit.null_log_likelihood, -15.605187860988003, 1e-9);
    close(fit.aic(), 34.33444832807072, 1e-8);
    let ctx = Context::new();
    let lr = fit.llr_test(&ctx).unwrap();
    close(lr.statistic_f64().unwrap(), 0.8759273939052861, 1e-8);
    close(lr.p_value_f64().unwrap(), 0.6453492105749965, 1e-10);
    assert_eq!(lr.df, Some(ctx.int(2)));
    let wald = fit.wald_test(&ctx).unwrap();
    close(wald.statistic_f64().unwrap(), 0.8445471219237513, 1e-8);
    close(wald.p_value_f64().unwrap(), 0.6555546806927786, 1e-10);
    let score = fit.score_test(&ctx).unwrap();
    close(score.statistic_f64().unwrap(), 0.8746958525820299, 1e-8);
    close(score.p_value_f64().unwrap(), 0.6457467200601523, 1e-10);
}

#[test]
fn d2_breslow_log_likelihoods_and_the_test_trio_match_statsmodels() {
    // Breslow: llf -15.40104714743394; llnull -15.797559753635461 (the null differs from Efron's: ties)
    //   LR 0.793025212403041 p 0.6726617969376194; Wald 0.7644549033226957 p 0.6823398362245261
    //   Score 0.7896012415696874 p 0.673814370437338; AIC 34.802094294867885
    let (obs, x) = d2();
    let fit = cox_ph(&obs, &x, &opts(Ties::Breslow)).unwrap();
    close(fit.log_likelihood, -15.40104714743394, 1e-9);
    close(fit.null_log_likelihood, -15.797559753635461, 1e-9);
    close(fit.aic(), 34.802094294867885, 1e-8);
    let ctx = Context::new();
    let lr = fit.llr_test(&ctx).unwrap();
    close(lr.statistic_f64().unwrap(), 0.793025212403041, 1e-8);
    close(lr.p_value_f64().unwrap(), 0.6726617969376194, 1e-10);
    let wald = fit.wald_test(&ctx).unwrap();
    close(wald.statistic_f64().unwrap(), 0.7644549033226957, 1e-8);
    close(wald.p_value_f64().unwrap(), 0.6823398362245261, 1e-10);
    let score = fit.score_test(&ctx).unwrap();
    close(score.statistic_f64().unwrap(), 0.7896012415696874, 1e-8);
    close(score.p_value_f64().unwrap(), 0.673814370437338, 1e-10);
}

#[test]
fn d2_baseline_hazard_under_efron_fit_is_breslow_increments() {
    // With eta = x2 @ params (Efron fit), w = exp(eta), uft = [3, 5, 7, 8, 10, 12]:
    //   h0 = [s2[t2 == u].sum() / w[t2 >= u].sum() for u in uft]
    //      → [0.1191885803708173, 0.1434325536599256, 0.10832320390143992, 0.12034405692584277,
    //         0.17666187420488488, 0.2664218737891737]
    //   np.cumsum(h0) → [0.1191885803708173, 0.2626211340307429, 0.3709443379321828, 0.49128839485802556,
    //         0.6679502690629104, 0.9343721428520841]
    //   (r.baseline_cumulative_hazard[0][1] = cumsum(h0) - h0 = [0.0, 0.11918858037081728, 0.2626211340307429, ...])
    let (obs, x) = d2();
    let fit = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    let rows = fit.baseline_hazard();
    let times: Vec<Q> = rows.iter().map(|r| r.time.clone()).collect();
    assert_eq!(times, vec![qi(3), qi(5), qi(7), qi(8), qi(10), qi(12)]);
    let hazard: Vec<f64> = rows.iter().map(|r| r.hazard).collect();
    close_all(
        &hazard,
        &[
            0.1191885803708173,
            0.1434325536599256,
            0.10832320390143992,
            0.12034405692584277,
            0.17666187420488488,
            0.2664218737891737,
        ],
        1e-9,
    );
    close(rows[5].cumulative, 0.9343721428520841, 1e-9);
    close(rows[1].cumulative, 0.2626211340307429, 1e-9);
    // Two events at t = 3 among all twelve at risk: h0 = 2 / Σ w.
    let w: Vec<f64> = fit.linear_predictors().iter().map(|e| e.exp()).collect();
    close(rows[0].hazard, 2.0 / w.iter().sum::<f64>(), 1e-12);
}

#[test]
fn d2_martingale_residuals_match_under_both_tie_corrections() {
    // Efron fit: mart = s2 - w * Lam (Lam inclusive)
    //   → [0.7632934881255977, 0.8986525252377275, 0.456861410955425, -0.23254841622109187, 0.5381627646343794,
    //      0.65794346746227, 0.1703587272660827, -0.47177082140551335, -0.27383817248676734, 0.06562785714791586,
    //      -1.711136215919106, -0.8616066147969194]; sum -1.1102230246251565e-16
    // Breslow fit:
    //   → [0.7709537685175839, 0.8928407178740501, 0.4816320277631593, -0.2425184619824171, 0.535373670008344,
    //      0.6485318571637448, 0.15598098199216515, -0.4782378860801601, -0.24764586642167297, 0.0647893699677714,
    //      -1.696294581859579, -0.8854055969429889]; sum 5.551115123125783e-16
    let (obs, x) = d2();
    let efron = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    let m = efron.martingale_residuals();
    close_all(
        &m,
        &[
            0.7632934881255977,
            0.8986525252377275,
            0.456861410955425,
            -0.23254841622109187,
            0.5381627646343794,
            0.65794346746227,
            0.1703587272660827,
            -0.47177082140551335,
            -0.27383817248676734,
            0.06562785714791586,
            -1.711136215919106,
            -0.8616066147969194,
        ],
        1e-9,
    );
    close(m.iter().sum::<f64>(), 0.0, 1e-12);
    let breslow = cox_ph(&obs, &x, &opts(Ties::Breslow)).unwrap();
    let m = breslow.martingale_residuals();
    close_all(
        &m,
        &[
            0.7709537685175839,
            0.8928407178740501,
            0.4816320277631593,
            -0.2425184619824171,
            0.535373670008344,
            0.6485318571637448,
            0.15598098199216515,
            -0.4782378860801601,
            -0.24764586642167297,
            0.0647893699677714,
            -1.696294581859579,
            -0.8854055969429889,
        ],
        1e-9,
    );
    close(m.iter().sum::<f64>(), 0.0, 1e-12);
}

#[test]
fn d2_schoenfeld_residuals_match_statsmodels() {
    // Efron fit, r.schoenfeld_residuals[s2 == 1]:
    //   [[-0.9033273731480036, 0.33017155465191983], [1.0966726268519964, -0.6698284453480802],
    //    [-1.9651094463203922, 0.3363500559264935], [2.034890553679608, 0.3363500559264935],
    //    [-1.014368532653283, -0.5878822152931167], [2.8730648647682067, 0.3468792628466101],
    //    [0.36936493986541263, 0.3395659596506837], [-2.442964703992438, -0.4879042257853926]]
    let (obs, x) = d2();
    let fit = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    let s = fit.schoenfeld_residuals();
    let expected = [
        [-0.9033273731480036, 0.33017155465191983],
        [1.0966726268519964, -0.6698284453480802],
        [-1.9651094463203922, 0.3363500559264935],
        [2.034890553679608, 0.3363500559264935],
        [-1.014368532653283, -0.5878822152931167],
        [2.8730648647682067, 0.3468792628466101],
        [0.36936493986541263, 0.3395659596506837],
        [-2.442964703992438, -0.4879042257853926],
    ];
    assert_eq!(s.len(), 8);
    for (row, e) in s.iter().zip(&expected) {
        close_all(row, e, 1e-9);
    }
    // Tied events (subjects 0 and 1 at t = 3) share the risk-set mean: their residuals differ by x_0 - x_1.
    close(s[0][0] - s[1][0], 2.0 - 4.0, 1e-12);
    close(s[0][1] - s[1][1], 1.0 - 0.0, 1e-12);
}

#[test]
fn d2_concordance_is_exactly_55_over_96() {
    // eta (Efron) = [0.686, -0.162, 0.727, -0.122, 0.565, -0.081, 0.524, -0.041, 0.646, 0.0, 0.605, -0.081]
    // Usable pairs t_i < t_j, event i (equal times are not usable):
    //   i=0 (t=3):  10 usable, 9 concordant     i=1 (t=3):  10 usable, 0 concordant
    //   i=2 (t=5):   7 usable, 7 concordant     i=4 (t=5):   7 usable, 5 concordant
    //   i=5 (t=7):   6 usable, 0 concordant, 1 tied (η_5 = η_11: identical rows [2, 0])
    //   i=6 (t=8):   4 usable, 2 concordant     i=8 (t=10):  3 usable, 3 concordant
    //   i=9 (t=12):  1 usable, 1 concordant
    // Totals 48 usable, 27 concordant, 1 tied → Fraction(2*27 + 1, 2*48) = 55/96.
    let (obs, x) = d2();
    let fit = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    assert_eq!(fit.concordance().unwrap(), q(55, 96));
    // The Breslow fit orders η the same way.
    let breslow = cox_ph(&obs, &x, &opts(Ties::Breslow)).unwrap();
    assert_eq!(breslow.concordance().unwrap(), q(55, 96));
}

// ═══════════════════════════════════════════════════════════════════════════
// D3: two groups of eight, a single 0/1 covariate, no tied event times.
//   t3 = [3, 5, 6, 7, 8, 10, 12, 14, 4, 9, 11, 13, 15, 16, 18, 20]
//   s3 = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1]; x3 = [[0]]*8 + [[1]]*8
// ═══════════════════════════════════════════════════════════════════════════

fn d3() -> (Vec<Observation>, Vec<Vec<f64>>, Vec<usize>) {
    let obs = Observation::from_i64(
        &[3, 5, 6, 7, 8, 10, 12, 14, 4, 9, 11, 13, 15, 16, 18, 20],
        &[
            true, false, true, true, false, true, true, false, true, true, false, true, true, true,
            false, true,
        ],
    );
    let groups: Vec<usize> = (0..16).map(|i| usize::from(i >= 8)).collect();
    let x = groups.iter().map(|&g| vec![g as f64]).collect();
    (obs, x, groups)
}

#[test]
fn d3_two_group_fit_matches_statsmodels() {
    // fit(t3, s3, x3, "breslow"): params [-1.121832078408797]; bse [0.7527302842899486]
    //   tvalues [-1.4903506632086985]; pvalues [0.13613205773552367]; llf -20.092294330305474
    //   llnull -21.25288086899316; np.exp(params) [0.32568257170274145]
    //   conf_int(0.05) [[-2.597156325689692, 0.35349216887209844]]; exp [[0.07448508867153207, 1.4240318351322656]]
    //   LR 2.32117307737537 p 0.1276237477240988; Wald 2.2211450993266073 p 0.13613205773552361
    // (Efron gives the same numbers: no ties.)
    let (obs, x, _) = d3();
    for ties in [Ties::Breslow, Ties::Efron] {
        let fit = cox_ph(&obs, &x, &opts(ties)).unwrap();
        close(fit.coefficients[0], -1.121832078408797, 1e-9);
        close(fit.standard_errors[0], 0.7527302842899486, 1e-9);
        close(fit.z_values[0], -1.4903506632086985, 1e-8);
        close(fit.p_values[0], 0.13613205773552367, 1e-9);
        close(fit.log_likelihood, -20.092294330305474, 1e-9);
        close(fit.null_log_likelihood, -21.25288086899316, 1e-9);
        close(fit.hazard_ratios()[0], 0.32568257170274145, 1e-8);
        let hr = fit.hazard_ratio_conf_int(0.95).unwrap();
        close(hr[0].lower, 0.07448508867153207, 1e-8);
        close(hr[0].upper, 1.4240318351322656, 1e-8);
        close(fit.llr(), 2.32117307737537, 1e-8);
        close(fit.wald_statistic(), 2.2211450993266073, 1e-8);
    }
}

#[test]
fn d3_score_test_of_a_binary_covariate_is_the_log_rank_statistic() {
    // statsmodels.duration.survfunc.survdiff(t3, s3, groups) → (2.425426790810062, 0.11938072392739785)
    // Score (Breslow) = g0 @ solve(-H0, g0) → 2.425426790810061; chi2.sf(2.425426790810061, 1) = 0.11938072392739774
    // Without ties the hypergeometric variance factor (n − d)/(n − 1) is 1 at every event time, so the
    // log-rank χ² and the Cox score statistic are the same number.
    let (obs, x, groups) = d3();
    let ctx = Context::new();
    let fit = cox_ph(&obs, &x, &opts(Ties::Breslow)).unwrap();
    let score = fit.score_test(&ctx).unwrap();
    let log_rank = log_rank_test(&ctx, &obs, &groups).unwrap();
    let lr_stat = log_rank.statistic_f64().unwrap();
    close(lr_stat, 2.425426790810062, 1e-12);
    close(score.statistic_f64().unwrap(), lr_stat, 1e-10);
    close(fit.score_statistic(), 2.425426790810061, 1e-9);
    close(score.p_value_f64().unwrap(), 0.11938072392739774, 1e-10);
    close(
        score.p_value_f64().unwrap(),
        log_rank.p_value_f64().unwrap(),
        1e-10,
    );
    assert_eq!(score.df, log_rank.df);
}

#[test]
fn d3_concordance_is_exactly_107_over_170() {
    // β̂ < 0, so group 0 is the higher risk: a pair is concordant when i is in group 0 and j in group 1,
    // tied when both are in the same group.  Per event i (t_i, group):
    //   (3,0): 15 usable, 8 conc, 7 tied   (6,0): 12, 7, 5   (7,0): 11, 7, 4   (10,0): 8, 6, 2   (12,0): 6, 5, 1
    //   (4,1): 14, 0, 7   (9,1): 9, 0, 6   (13,1): 5, 0, 4   (15,1): 3, 0, 3   (16,1): 2, 0, 2   (20,1): 0
    // Totals 85 usable, 33 concordant, 41 tied → Fraction(2*33 + 41, 2*85) = 107/170.
    let (obs, x, _) = d3();
    let fit = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    assert_eq!(fit.concordance().unwrap(), q(107, 170));
}

// ═══════════════════════════════════════════════════════════════════════════
// Freireich et al. (1963) 6-MP trial, Gehan's version: 21 placebo (all
// relapsed) versus 21 6-MP (9 relapsed), many tied times.
// ═══════════════════════════════════════════════════════════════════════════

fn gehan() -> (Vec<Observation>, Vec<Vec<f64>>) {
    let ctrl_t = [
        1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 8, 8, 11, 11, 12, 12, 15, 17, 22, 23,
    ];
    let mp_t = [
        6, 6, 6, 6, 7, 9, 10, 10, 11, 13, 16, 17, 19, 20, 22, 23, 25, 32, 32, 34, 35,
    ];
    let mp_s = [
        true, true, true, false, true, false, true, false, false, true, true, false, false, false,
        true, true, false, false, false, false, false,
    ];
    let times: Vec<i64> = ctrl_t.iter().chain(&mp_t).copied().collect();
    let events: Vec<bool> = [true; 21].iter().chain(&mp_s).copied().collect();
    let obs = Observation::from_i64(&times, &events);
    let x = (0..42)
        .map(|i| vec![if i < 21 { 0.0 } else { 1.0 }])
        .collect();
    (obs, x)
}

#[test]
fn gehan_efron_matches_statsmodels_and_r() {
    // fit(t4, s4, x4, "efron"): params [-1.572125148829067]; bse [0.41239671770945563]
    //   tvalues [-3.8121669773731583]; pvalues [0.0001377537619867646]; llf -85.00842457737163
    //   llnull -93.18426999684775; LR 16.351690838952237 p 5.260920532886758e-05
    //   Wald 14.532617063374403; Score 17.246536795716576 p 3.282954131415533e-05
    //   np.exp(params) [0.20760352484153724]
    // (R: coxph(Surv(time, status) ~ group, gehan) coef -1.5721, se 0.4124, exp(coef) 0.2076.)
    let (obs, x) = gehan();
    let fit = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    assert_eq!(fit.n_events, 30);
    close(fit.coefficients[0], -1.572125148829067, 1e-9);
    close(fit.standard_errors[0], 0.41239671770945563, 1e-9);
    close(fit.z_values[0], -3.8121669773731583, 1e-8);
    close(fit.p_values[0], 0.0001377537619867646, 1e-10);
    close(fit.log_likelihood, -85.00842457737163, 1e-9);
    close(fit.null_log_likelihood, -93.18426999684775, 1e-9);
    close(fit.llr(), 16.351690838952237, 1e-8);
    close(fit.wald_statistic(), 14.532617063374403, 1e-8);
    close(fit.score_statistic(), 17.246536795716576, 1e-8);
    close(fit.hazard_ratios()[0], 0.20760352484153724, 1e-9);
    let ctx = Context::new();
    close(
        fit.llr_test(&ctx).unwrap().p_value_f64().unwrap(),
        5.260920532886758e-05,
        1e-12,
    );
    close(
        fit.score_test(&ctx).unwrap().p_value_f64().unwrap(),
        3.282954131415533e-05,
        1e-12,
    );
}

#[test]
fn gehan_breslow_matches_statsmodels_and_r() {
    // fit(t4, s4, x4, "breslow"): params [-1.5091914125858785]; bse [0.4095644063667413]
    //   llf -86.37962207114556; llnull -93.98505047824518; LR 15.210856814199246
    //   Wald 13.578263650923104; Score 15.930539564032713; np.exp(params) [0.2210886752236532]
    // (R: coxph(..., ties = "breslow") coef -1.5092, se 0.4096.)
    let (obs, x) = gehan();
    let fit = cox_ph(&obs, &x, &opts(Ties::Breslow)).unwrap();
    close(fit.coefficients[0], -1.5091914125858785, 1e-9);
    close(fit.standard_errors[0], 0.4095644063667413, 1e-9);
    close(fit.log_likelihood, -86.37962207114556, 1e-9);
    close(fit.null_log_likelihood, -93.98505047824518, 1e-9);
    close(fit.llr(), 15.210856814199246, 1e-8);
    close(fit.wald_statistic(), 13.578263650923104, 1e-8);
    close(fit.score_statistic(), 15.930539564032713, 1e-8);
    close(fit.hazard_ratios()[0], 0.2210886752236532, 1e-9);
}

#[test]
fn gehan_concordance_is_exactly_487_over_706() {
    // Fraction over the 706 usable pairs (t_i < t_j, event i): 332 concordant (i placebo, j 6-MP),
    // 310 tied (same group), 64 discordant → (2*332 + 310) / (2*706) = 974/1412 = 487/706.
    let (obs, x) = gehan();
    let fit = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    assert_eq!(fit.concordance().unwrap(), q(487, 706));
}

// ═══════════════════════════════════════════════════════════════════════════
// Stratified: D2 with strata [0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1] (no tied
// event times within a stratum, so Efron = Breslow).
// ═══════════════════════════════════════════════════════════════════════════

fn d2_strata() -> Vec<usize> {
    vec![0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]
}

#[test]
fn stratified_fit_matches_statsmodels() {
    // PHReg(t2, x2, status=s2, strata=strata, ties="efron").fit(): params [-0.08418688221739194, 0.6182396394612647]
    //   bse [0.271422880491644, 0.9364979337868439]; llf -10.445356103938222; m.loglike(zeros) -10.6735957742322
    // ties="breslow": params [-0.08418688221739183, 0.6182396394612645], llf -10.445356103938224 (same: no ties within strata)
    let (obs, x) = d2();
    let strata = d2_strata();
    let fit = cox_ph_stratified(&obs, &x, &strata, &CoxOpts::default()).unwrap();
    close_all(
        &fit.coefficients,
        &[-0.08418688221739194, 0.6182396394612647],
        1e-9,
    );
    close_all(
        &fit.standard_errors,
        &[0.271422880491644, 0.9364979337868439],
        1e-9,
    );
    close(fit.log_likelihood, -10.445356103938222, 1e-9);
    close(fit.null_log_likelihood, -10.6735957742322, 1e-9);
    assert_eq!(fit.strata(), &strata[..]);
    let breslow = cox_ph_stratified(&obs, &x, &strata, &opts(Ties::Breslow)).unwrap();
    close_all(&breslow.coefficients, &fit.coefficients, 1e-10);
    close(breslow.log_likelihood, fit.log_likelihood, 1e-10);
    // Stratifying changes the fit: the unstratified coefficients are different.
    let pooled = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    assert!((pooled.coefficients[1] - fit.coefficients[1]).abs() > 0.1);
}

#[test]
fn stratified_baseline_hazard_is_per_stratum() {
    // Stratum 0 (subjects 0, 3, 4, 7, 9, 10; events at 3, 5, 12):
    //   h0 → [0.14689914142872798, 0.1908655606993763, 0.43008800329456454]
    //   cumulative → [0.14689914142872798, 0.3377647021281043, 0.7678527054226688]
    // Stratum 1 (subjects 1, 2, 5, 6, 8, 11; events at 3, 5, 7, 8, 10):
    //   h0 → [0.14989686745507383, 0.16786510080579842, 0.2352203555920969, 0.29357411923803384, 0.43734385222001526]
    //   cumulative → [..., 1.2839002953110181]
    // (statsmodels r.baseline_cumulative_hazard[k][1], exclusive, → [0, 0.14689914142872798, 0.3377647021281043] and
    //  [0, 0.14989686745507383, 0.31776196826087233, 0.5529823238529692, 0.8465564430910031].)
    let (obs, x) = d2();
    let fit = cox_ph_stratified(&obs, &x, &d2_strata(), &CoxOpts::default()).unwrap();
    let rows = fit.baseline_hazard();
    assert_eq!(rows.len(), 8);
    let s0: Vec<&BaselineHazardRow> = rows.iter().filter(|r| r.stratum == 0).collect();
    let s1: Vec<&BaselineHazardRow> = rows.iter().filter(|r| r.stratum == 1).collect();
    assert_eq!(
        s0.iter().map(|r| r.time.clone()).collect::<Vec<_>>(),
        vec![qi(3), qi(5), qi(12)]
    );
    assert_eq!(
        s1.iter().map(|r| r.time.clone()).collect::<Vec<_>>(),
        vec![qi(3), qi(5), qi(7), qi(8), qi(10)]
    );
    close_all(
        &s0.iter().map(|r| r.hazard).collect::<Vec<_>>(),
        &[0.14689914142872798, 0.1908655606993763, 0.43008800329456454],
        1e-9,
    );
    close(s0[2].cumulative, 0.7678527054226688, 1e-9);
    close_all(
        &s1.iter().map(|r| r.hazard).collect::<Vec<_>>(),
        &[
            0.14989686745507383,
            0.16786510080579842,
            0.2352203555920969,
            0.29357411923803384,
            0.43734385222001526,
        ],
        1e-9,
    );
    close(s1[4].cumulative, 1.2839002953110181, 1e-9);
    close(s1[3].cumulative, 0.8465564430910031, 1e-9);
}

#[test]
fn stratified_residuals_match() {
    // Martingale (inclusive, within stratum): mart = s2 - w * Lam_stratum(t)
    //   → [0.7696471845966079, 0.8929602802222012, 0.4579515805349239, -0.2623786944540906, 0.5885634838874121,
    //      0.5327089614332798, 0.052056194403678924, -0.310493393226104, -0.8507325680414659, 0.2321472945773312,
    //      -1.0174858753811566, -1.0849444485526172]; sum 6.66e-16
    // r.schoenfeld_residuals[s2 == 1]:
    //   [[-0.61141010644074, 0.3960501548368319], [1.1541474483092475, -0.6396232028887724],
    //    [-1.7075041590628723, 0.2837047547852738], [2.205596903779755, 0.514586634971467],
    //    [-1.3926339277299005, -0.6024597549852624], [2.261879684157549, 0.2480812661771924],
    //    [0.36957214380870473, 0.3695721438087043], [-2.279647986821742, -0.5699119967054355]]
    let (obs, x) = d2();
    let fit = cox_ph_stratified(&obs, &x, &d2_strata(), &CoxOpts::default()).unwrap();
    let m = fit.martingale_residuals();
    close_all(
        &m,
        &[
            0.7696471845966079,
            0.8929602802222012,
            0.4579515805349239,
            -0.2623786944540906,
            0.5885634838874121,
            0.5327089614332798,
            0.052056194403678924,
            -0.310493393226104,
            -0.8507325680414659,
            0.2321472945773312,
            -1.0174858753811566,
            -1.0849444485526172,
        ],
        1e-9,
    );
    close(m.iter().sum::<f64>(), 0.0, 1e-12);
    let s = fit.schoenfeld_residuals();
    let expected = [
        [-0.61141010644074, 0.3960501548368319],
        [1.1541474483092475, -0.6396232028887724],
        [-1.7075041590628723, 0.2837047547852738],
        [2.205596903779755, 0.514586634971467],
        [-1.3926339277299005, -0.6024597549852624],
        [2.261879684157549, 0.2480812661771924],
        [0.36957214380870473, 0.3695721438087043],
        [-2.279647986821742, -0.5699119967054355],
    ];
    for (row, e) in s.iter().zip(&expected) {
        close_all(row, e, 1e-9);
    }
}

#[test]
fn stratified_concordance_counts_pairs_within_strata() {
    // Fraction over pairs with equal stratum, t_i < t_j, event i: 23 usable, 13 concordant, 1 tied
    // (subjects 5 and 11, both [2, 0] in stratum 1) → (2*13 + 1) / 46 = 27/46.
    let (obs, x) = d2();
    let fit = cox_ph_stratified(&obs, &x, &d2_strata(), &CoxOpts::default()).unwrap();
    assert_eq!(fit.concordance().unwrap(), q(27, 46));
}

#[test]
fn stratified_with_one_stratum_equals_the_plain_fit() {
    let (obs, x) = d2();
    let plain = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    let one = cox_ph_stratified(&obs, &x, &[7; 12], &CoxOpts::default()).unwrap();
    close_all(&one.coefficients, &plain.coefficients, 1e-12);
    close(one.log_likelihood, plain.log_likelihood, 1e-12);
    assert_eq!(one.baseline_hazard().len(), plain.baseline_hazard().len());
    assert!(one.baseline_hazard().iter().all(|r| r.stratum == 7));
    assert_eq!(one.concordance().unwrap(), plain.concordance().unwrap());
    assert!(matches!(
        cox_ph_stratified(&obs, &x, &[0; 11], &CoxOpts::default()),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// Failure modes.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn monotone_likelihood_is_reported_naming_the_covariate() {
    // Every subject with x = 1 fails before every subject with x = 0: β̂ → +∞, sup ℓ = −2 ln 4! = −6.3561.
    // statsmodels PHReg(t, x, status=s).fit(method="newton", maxiter=200, tol=1e-12) runs to
    // params [28.240022504020462], llf -6.356107660700423 — no finite maximiser.
    let obs = Observation::from_i64(&[1, 2, 3, 4, 5, 6, 7, 8], &[true; 8]);
    let x: Vec<Vec<f64>> = (0..8)
        .map(|i| vec![if i < 4 { 1.0 } else { 0.0 }])
        .collect();
    match cox_ph(&obs, &x, &CoxOpts::default()) {
        Err(SymplexError::ComputationFailed { operation, reason }) => {
            assert_eq!(operation, "cox_ph");
            assert!(reason.contains("monotone likelihood"), "{reason}");
            assert!(reason.contains("covariate 0"), "{reason}");
        }
        other => panic!("expected ComputationFailed, got {other:?}"),
    }
    // Two covariates, only the second separating: the message names covariate 1.
    let x2: Vec<Vec<f64>> = (0..8)
        .map(|i| {
            vec![
                [0.3, -1.0, 2.0, 0.7, 1.1, -0.4, 0.0, 1.6][i],
                if i < 4 { 1.0 } else { 0.0 },
            ]
        })
        .collect();
    match cox_ph(&obs, &x2, &CoxOpts::default()) {
        Err(SymplexError::ComputationFailed { reason, .. }) => {
            assert!(reason.contains("covariate 1"), "{reason}");
        }
        other => panic!("expected ComputationFailed, got {other:?}"),
    }
}

#[test]
fn collinear_covariates_are_reported_as_a_singular_information_matrix() {
    let (obs, x) = d1();
    let x2: Vec<Vec<f64>> = x.iter().map(|r| vec![r[0], 2.0 * r[0]]).collect();
    match cox_ph(&obs, &x2, &CoxOpts::default()) {
        Err(SymplexError::ComputationFailed { reason, .. }) => {
            assert!(reason.contains("singular"), "{reason}");
        }
        other => panic!("expected ComputationFailed, got {other:?}"),
    }
}

#[test]
fn rejects_invalid_input() {
    let (obs, x) = d1();
    let o = CoxOpts::default();
    // Empty, mismatched, ragged, no covariates, non-finite, constant column, no events, bad options.
    assert!(matches!(
        cox_ph(&[], &[], &o),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        cox_ph(&obs, &x[..9], &o),
        Err(SymplexError::InvalidArgument { .. })
    ));
    let mut ragged = x.clone();
    ragged[4] = vec![1.0, 2.0];
    assert!(matches!(
        cox_ph(&obs, &ragged, &o),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        cox_ph(&obs, &vec![vec![]; 10], &o),
        Err(SymplexError::InvalidArgument { .. })
    ));
    let mut nan = x.clone();
    nan[2][0] = f64::NAN;
    assert!(matches!(
        cox_ph(&obs, &nan, &o),
        Err(SymplexError::InvalidArgument { reason, .. }) if reason.contains("x[2][0]")
    ));
    assert!(matches!(
        cox_ph(&obs, &vec![vec![1.5]; 10], &o),
        Err(SymplexError::InvalidArgument { reason, .. }) if reason.contains("constant")
    ));
    let censored: Vec<Observation> = obs
        .iter()
        .map(|ob| Observation {
            time: ob.time.clone(),
            event: false,
        })
        .collect();
    assert!(matches!(
        cox_ph(&censored, &x, &o),
        Err(SymplexError::InvalidArgument { reason, .. }) if reason.contains("no events")
    ));
    assert!(matches!(
        cox_ph(&obs, &x, &CoxOpts { max_iter: 0, ..o }),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        cox_ph(&obs, &x, &CoxOpts { tol: -1e-9, ..o }),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

#[test]
fn exhausting_the_iteration_budget_is_an_error() {
    // One Newton step from β = 0 is not the optimum for D1 (five are needed).
    let (obs, x) = d1();
    match cox_ph(
        &obs,
        &x,
        &CoxOpts {
            max_iter: 1,
            ..CoxOpts::default()
        },
    ) {
        Err(SymplexError::ComputationFailed { reason, .. }) => {
            assert!(reason.contains("no convergence in 1"), "{reason}");
        }
        other => panic!("expected ComputationFailed, got {other:?}"),
    }
}

#[test]
fn concordance_needs_a_usable_pair() {
    // Two events at the last time, everyone else censored earlier: no pair has t_i < t_j with i an event,
    // yet the partial likelihood β − 2 ln(1 + e^β) has a finite maximiser (β̂ = 0).
    let obs = Observation::from_i64(&[5, 5, 3, 2], &[true, true, false, false]);
    let x = vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0]];
    let fit = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    close(fit.coefficients[0], 0.0, 1e-9);
    assert!(matches!(
        fit.concordance(),
        Err(SymplexError::ComputationFailed {
            operation: "concordance",
            ..
        })
    ));
}

#[test]
fn options_default_to_efron_fifty_steps_and_1e_minus_9() {
    let o = CoxOpts::default();
    assert_eq!(o.ties, Ties::Efron);
    assert_eq!(o.max_iter, 50);
    assert_eq!(o.tol, 1e-9);
    assert_eq!(Ties::default(), Ties::Efron);
}

#[test]
fn a_tighter_tolerance_reproduces_the_default_fit() {
    let (obs, x) = d2();
    let a = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    let b = cox_ph(
        &obs,
        &x,
        &CoxOpts {
            tol: 1e-13,
            ..CoxOpts::default()
        },
    )
    .unwrap();
    close_all(&a.coefficients, &b.coefficients, 1e-10);
    assert!(b.iterations >= a.iterations);
    // The model is `Clone + PartialEq`.
    let c: CoxModel = a.clone();
    assert_eq!(c, a);
}

#[test]
fn book_example_numbers() {
    // The `Cox proportional hazards` subsection of book/src/guide/statistics.md prints these
    // (all from fit(t1, s1, x1, "efron") and the pair enumeration above).
    let (obs, x) = d1();
    let fit = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    close(fit.coefficients[0], 0.8759809887649096, 1e-9);
    close(fit.hazard_ratios()[0], 2.401229718322006, 1e-8);
    close(fit.standard_errors[0], 0.3809028296542367, 1e-9);
    close(fit.p_values[0], 0.021462431348704628, 1e-9);
    close(fit.log_likelihood, -8.465216276861835, 1e-9);
    close(fit.null_log_likelihood, -12.108680299521524, 1e-9);
    let ctx = Context::new();
    let lr = fit.llr_test(&ctx).unwrap();
    close(lr.statistic_f64().unwrap(), 7.286928045319378, 1e-8);
    close(lr.p_value_f64().unwrap(), 0.006945814500177098, 1e-10);
    close(fit.score_statistic(), 7.355733707724492, 1e-8);
    assert_eq!(fit.concordance().unwrap(), q(31, 38));
    let rows = fit.baseline_hazard();
    assert_eq!(rows[0].stratum, 0);
    assert_eq!(rows[0].time, qi(2));
    close(rows[0].hazard, 0.002858840843475405, 1e-9);
    close(rows[0].cumulative, 0.002858840843475405, 1e-9);
    assert_eq!(rows[6].time, qi(12));
    close(rows[6].hazard, 0.2940113085020759, 1e-9);
    close(rows[6].cumulative, 0.4700301513413364, 1e-9);
}
