//! symplex 0.14 — time-to-event analysis (`stats::survival`).  Reference
//! values cite statsmodels 0.15 `SurvfuncRight` / `survdiff`, scipy 1.18
//! `logrank`, and `Fraction` arithmetic (`symplex/.venv/bin/python`).

use symplex::linprog::{q, qi};
use symplex::prelude::*;
use symplex::stats::Distribution;
use symplex::stats::survival::{
    CiMethod, KaplanMeier, Observation, exponential_rate, hazard_function, log_rank_test,
    mean_event_time, survival_function,
};

/// statsmodels: SurvfuncRight([3,5,6,7,8,10,12,12], [1,0,1,1,0,1,1,0])
fn sample() -> Vec<Observation> {
    Observation::from_i64(
        &[3, 5, 6, 7, 8, 10, 12, 12],
        &[true, false, true, true, false, true, true, false],
    )
}

#[test]
fn kaplan_meier_steps_are_exact() {
    let km = KaplanMeier::fit(&sample()).unwrap();
    assert_eq!(km.n(), 8);
    // surv_times [3, 6, 7, 10, 12]; surv_prob [0.875, 0.72916667, 0.58333333, 0.38888889, 0.19444444]
    // = 7/8, 35/48, 7/12, 7/18, 7/36 (Fraction).
    assert_eq!(km.event_times(), vec![qi(3), qi(6), qi(7), qi(10), qi(12)]);
    let s: Vec<_> = km.table().iter().map(|r| r.survival.clone()).collect();
    assert_eq!(s, vec![q(7, 8), q(35, 48), q(7, 12), q(7, 18), q(7, 36)]);
    // Step function: 1 before the first event, constant between events.
    assert_eq!(km.survival_at(&qi(0)), qi(1));
    assert_eq!(km.survival_at(&q(5, 2)), qi(1));
    assert_eq!(km.survival_at(&qi(5)), q(7, 8));
    assert_eq!(km.survival_at(&qi(9)), q(7, 12));
    assert_eq!(km.survival_at(&qi(100)), q(7, 36));
    // Risk sets and counts.
    let r = &km.table()[1];
    assert_eq!((r.at_risk, r.events, r.censored), (6, 1, 0));
    let last = &km.table()[4];
    assert_eq!((last.at_risk, last.events, last.censored), (2, 1, 1));
}

#[test]
fn greenwood_variance_matches_statsmodels_standard_errors() {
    let km = KaplanMeier::fit(&sample()).unwrap();
    // surv_prob_se² = [0.013671875, 0.0272171585648, 0.0344328703704, 0.0405092592593, 0.0290316358025]
    // = 7/512, 1505/55296, 119/3456, 35/864, 301/10368 (Fraction Greenwood sums).
    let v: Vec<_> = km.table().iter().map(|r| r.variance.clone()).collect();
    assert_eq!(
        v,
        vec![
            q(7, 512),
            q(1505, 55296),
            q(119, 3456),
            q(35, 864),
            q(301, 10368)
        ]
    );
    assert_eq!(km.variance_at(&qi(7)), q(119, 3456));
    assert_eq!(km.variance_at(&qi(1)), qi(0));
}

#[test]
fn nelson_aalen_cumulative_hazard() {
    let km = KaplanMeier::fit(&sample()).unwrap();
    // Fraction: Σ d/n = 1/8, 7/24, 59/120, 33/40, 53/40.
    let h: Vec<_> = km
        .table()
        .iter()
        .map(|r| r.cumulative_hazard.clone())
        .collect();
    assert_eq!(h, vec![q(1, 8), q(7, 24), q(59, 120), q(33, 40), q(53, 40)]);
    assert_eq!(km.cumulative_hazard_at(&qi(8)), q(59, 120));
}

#[test]
fn quantiles_and_median() {
    let km = KaplanMeier::fit(&sample()).unwrap();
    // statsmodels quantile(0.25) = 6, quantile(0.5) = 10, quantile(0.75) = 12.
    assert_eq!(km.quantile(&q(1, 4)), Some(qi(6)));
    assert_eq!(km.median(), Some(qi(10)));
    assert_eq!(km.quantile(&q(3, 4)), Some(qi(12)));
    // A curve that never reaches 1 − p has no such quantile.
    let heavy = Observation::from_i64(&[1, 2, 3, 4], &[true, false, false, false]);
    let km2 = KaplanMeier::fit(&heavy).unwrap();
    assert_eq!(km2.survival_at(&qi(4)), q(3, 4));
    assert_eq!(km2.median(), None);
}

#[test]
fn pointwise_confidence_intervals() {
    let km = KaplanMeier::fit(&sample()).unwrap();
    // At t = 7: S = 7/12, se = √(119/3456); z = scipy norm.ppf(0.975).
    // linear: (0.21964053218643598, 0.9470261344802308);
    // log-log: (0.1801888910128228, 0.8440686916574577).
    let (lo, hi) = km
        .confidence_interval(&qi(7), 0.95, CiMethod::Linear)
        .unwrap();
    assert!(
        (lo - 0.219_640_532_186_435_98).abs() < 1e-9 && (hi - 0.947_026_134_480_230_8).abs() < 1e-9
    );
    let (lo, hi) = km
        .confidence_interval(&qi(7), 0.95, CiMethod::LogLog)
        .unwrap();
    assert!(
        (lo - 0.180_188_891_012_822_8).abs() < 1e-9 && (hi - 0.844_068_691_657_457_7).abs() < 1e-9
    );
    assert!(
        km.confidence_interval(&qi(7), 1.5, CiMethod::Linear)
            .is_err()
    );
}

#[test]
fn restricted_mean_survival_time() {
    let km = KaplanMeier::fit(&sample()).unwrap();
    // Area under the step function up to τ = 12: 1279/144 (Fraction).
    assert_eq!(km.restricted_mean(&qi(12)), q(1279, 144));
    // Before the first event the curve is 1: area = τ.
    assert_eq!(km.restricted_mean(&qi(2)), qi(2));
}

#[test]
fn log_rank_two_groups_matches_scipy_and_statsmodels() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let obs = Observation::from_i64(
        &[3, 5, 6, 7, 8, 10, 12, 12, 4, 9, 11, 13, 15, 16, 18, 20],
        &[
            true, false, true, true, false, true, true, false, true, true, false, true, true, true,
            false, true,
        ],
    );
    let groups = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let r = log_rank_test(&ctx, &obs, &groups)?;
    // statsmodels survdiff: (3.0736117780824452, 0.07957250154977413);
    // scipy logrank: z = 1.7531719191461075, z² = 3.0736117780824457.
    assert_eq!(r.statistic, ctx.rational(149_059_681, 48_496_587));
    assert!((r.statistic_f64()? - 3.073_611_778_082_445_2).abs() < 1e-12);
    assert!((r.p_value_f64()? - 0.079_572_501_549_774_13).abs() < 1e-12);
    assert_eq!(r.df, Some(ctx.int(1)));
    Ok(())
}

#[test]
fn log_rank_three_groups() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let obs = Observation::from_i64(
        &[1, 2, 3, 4, 5, 6, 7, 8, 9],
        &[true, true, false, true, true, true, false, true, true],
    );
    let groups = [0, 0, 0, 1, 1, 1, 2, 2, 2];
    let r = log_rank_test(&ctx, &obs, &groups)?;
    // statsmodels survdiff: (8.215752580706997, 0.01644265690224056), df = 2.
    assert!(
        (r.statistic_f64()? - 8.215_752_580_706_997).abs() < 1e-12,
        "{}",
        r.statistic
    );
    assert!((r.p_value_f64()? - 0.016_442_656_902_240_56).abs() < 1e-12);
    assert_eq!(r.df, Some(ctx.int(2)));
    Ok(())
}

#[test]
fn log_rank_validation() {
    let ctx = Context::new();
    let obs = Observation::from_i64(&[1, 2, 3], &[true, true, true]);
    assert!(log_rank_test(&ctx, &obs, &[0, 0, 0]).is_err(), "one group");
    assert!(
        log_rank_test(&ctx, &obs, &[0, 1]).is_err(),
        "length mismatch"
    );
    assert!(
        log_rank_test(&ctx, &obs, &[0, 0, 2]).is_err(),
        "group 1 missing"
    );
    let censored = Observation::from_i64(&[1, 2], &[false, false]);
    assert!(
        log_rank_test(&ctx, &censored, &[0, 1]).is_err(),
        "no events"
    );
    assert!(KaplanMeier::fit(&[]).is_err());
}

#[test]
fn parametric_summaries() -> Result<(), SymplexError> {
    let obs = sample();
    // λ̂ = events / total time = 5 / 63.
    assert_eq!(exponential_rate(&obs)?, q(5, 63));
    // Mean of the five event times (3, 6, 7, 10, 12) = 38/5.
    assert_eq!(mean_event_time(&obs)?, q(38, 5));
    // The exponential's hazard is constant and its survival function e^{−λt}.
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let e = Distribution::exponential(ctx.rational(5, 63));
    assert_eq!(hazard_function(&e, &t), ctx.rational(5, 63));
    assert_eq!(survival_function(&e, &ctx.int(63)), (-ctx.int(5)).exp());
    Ok(())
}
