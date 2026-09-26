//! Audit round 2b of `symplex::stats`: `survival` and `cox`.
//!
//! 1. `cox_ph` reported a monotone likelihood (`β̂ = ±∞`) as a converged
//!    fit: on `t = [0, 3, 3]`, `δ = [1, 0, 1]`, `x = [2, 4, 4]` it returned
//!    `β = −18.60`, `se = 2.4 × 10⁷` (the score and information had become
//!    rounding noise, step-halving ran out, and the `2⁻³⁰` step passed the
//!    `|Δβ|` test), and on a separated design with the covariate in units
//!    of `10¹⁰` it returned `β·s = 22.1` "converged".  The divergence test
//!    was `|β| > 25`, which depends on the covariate's units: the same data
//!    in units of `10³` or `10⁸` failed as "singular", and a finite fit on
//!    a covariate in units of `10⁻²` (`β̂ = 87.6`) or `10⁻³` was refused as
//!    monotone.  Convergence and monotone likelihood are now judged on the
//!    Newton step's movement of the log hazard ratios and on the predicted
//!    gain, both invariant to the covariates' units and offsets.
//! 2. The information matrix `S2/S0 − (S1/S0)²` was accumulated on the raw
//!    covariates: `x + 10⁶` gave `se = 0.380925` (true `0.380903`), and
//!    `x + 10⁸` a "singular information matrix".  Covariates are centred.
//! 3. Risk-set sums were shifted by the global `max η`: any risk set whose
//!    members all sat ~745 below it underflowed (`S0 = 0`, `ℓ = −∞`), so a
//!    subject with an extreme covariate that fails first, or is censored
//!    before the first event, made the fit fail ("no convergence").  Each
//!    risk set now has its own shift (a running log-sum-exp).
//! 4. `build_table`, `martingale_residuals` and `concordance` were
//!    `O(n·T)` / `O(n²)` in big-rational comparisons: 7.5 s, 3.5 s and
//!    3.6 s for 6 000 subjects (debug build); now sort-and-sweep.
//! 5. `survival_function` was `1 − F(t)` on the support's closed form:
//!    `S(26) = 0` for `N(0, 1)`, `hazard_function` `PrecisionExhausted` from
//!    `t = 20`, and below the support `S(−1) = e^{5/63} > 1` for an
//!    exponential, `h(−1) = −2` for a Weibull, `S(3) = −½` for `U(0, 2)`.
//! 6. `KaplanMeier::quantile(−1)` was the first event time; levels outside
//!    `[0, 1]` now give `None`.  Documentation: `quantile_strict` is exact
//!    where statsmodels rounds, the Greenwood variance at `d = n`, the
//!    Schoenfeld residuals' sum under Efron ties, the log-rank errors.
//!
//! Reference values: statsmodels 0.15.0 and scipy 1.18.1 by the call
//! quoted; mpmath 1.3 at 50 digits for the Cox partial likelihood
//! (`target/scratch/cox_mp.py`: the Efron/Breslow log partial likelihood,
//! score and information summed directly over the risk sets, Newton to
//! `10⁻⁴⁰`; driven by `oracle_2b.py`, output `oracle_2b.out`), exact
//! log-rank statistics from `fractions.Fraction` + `sympy.Rational`.

// Reference values are quoted at the digits mpmath printed them.
#![allow(clippy::excessive_precision)]

use std::time::{Duration, Instant};

use num_bigint::BigInt;
use symplex::linprog::{Q, q, qi};
use symplex::prelude::*;
use symplex::stats::Distribution;
use symplex::stats::cox::{CoxModel, CoxOpts, Ties, cox_ph, cox_ph_stratified};
use symplex::stats::survival::{
    KaplanMeier, Observation, hazard_function, log_rank_test, survival_function,
};

type R = Result<(), SymplexError>;

/// `actual` within `rel` of `expected`, relatively.
fn close(actual: f64, expected: f64, rel: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= rel * expected.abs(),
        "{label}: got {actual:e}, expected {expected:e} (rel {:.1e})",
        (actual / expected - 1.0).abs()
    );
}

fn big(s: &str) -> BigInt {
    s.parse().expect("integer literal")
}

fn opts(ties: Ties) -> CoxOpts {
    CoxOpts {
        ties,
        ..CoxOpts::default()
    }
}

/// A `ComputationFailed` whose reason diagnoses a monotone likelihood and
/// names `covariate`.
fn assert_monotone(r: Result<CoxModel, SymplexError>, covariate: usize, label: &str) {
    match r {
        Err(SymplexError::ComputationFailed { reason, .. }) => {
            assert!(reason.contains("monotone likelihood"), "{label}: {reason}");
            assert!(
                reason.contains(&format!("covariate {covariate}")),
                "{label}: {reason}"
            );
        }
        other => panic!("{label}: expected a monotone likelihood, got {other:?}"),
    }
}

/// The ten-subject, one-covariate dataset of the `stats::cox` module docs.
fn d1_times() -> Vec<Observation> {
    Observation::from_i64(
        &[4, 7, 2, 9, 12, 5, 15, 3, 11, 8],
        &[
            true, true, true, false, true, true, false, true, true, false,
        ],
    )
}

const D1_X: [f64; 10] = [3.0, 1.0, 5.0, 2.0, 0.0, 4.0, 1.0, 6.0, 2.0, 3.0];

// mpmath cox_mp.fit(t1, s1, [[b] for b in x1], ties) at 50 digits — the same for
// 'efron' and 'breslow' (no tied event times), and for every offset and scale below
// once multiplied back (oracle_2b.py sections 1–3):
const D1_BETA: f64 = 0.875_980_988_764_909_55;
const D1_SE: f64 = 0.380_902_829_654_236_25;
const D1_LL: f64 = -8.465_216_276_861_834_9;
const D1_LL0: f64 = -12.108_680_299_521_524;
const D1_SCORE: f64 = 7.355_733_707_724_495_7;

/// Newton stops once its step moves the log hazard ratios by at most
/// `10⁻⁹` of their spread, and the error left after that step is quadratic
/// in it; what remains is rounding in the `O(n)` risk-set sums (a few
/// `10⁻¹⁶` on these data).  `1e-12` leaves room for both.
const FIT_REL: f64 = 1e-12;

// ═══════════════════════════════════════════════════════════════════════════
// 1. Monotone likelihood and scale-invariant convergence
// ═══════════════════════════════════════════════════════════════════════════

/// `ℓ(β) = −ln(1 + 2e^{2β}) − ln 2` on these three subjects increases to its
/// supremum `−ln 2` as `β → −∞` (mpmath `cox_mp.loglik` at `β = 0, −10,
/// −20, −40`: `−1.79175946923`, `−0.693147184682`, `−0.69314718056`,
/// `−0.69314718056`; score `−1.33`, `−8.2e-9`, `−1.7e-17`, `−7.2e-35`).
/// Before, both tie corrections returned `Ok` with `β = −18.601781810229255`,
/// `se = 23 726 566`: the score and information had become rounding noise,
/// step-halving exhausted its 30 halvings, and the `2⁻³⁰` step passed the
/// step-size test.  statsmodels `PHReg([0, 3, 3], [[2], [4], [4]], status=[1,
/// 0, 1]).fit()` also returns `params −14.1689`, `bse 503398` with only a
/// `ConvergenceWarning`.
#[test]
fn a_monotone_likelihood_is_never_reported_as_converged() {
    let obs = Observation::from_i64(&[0, 3, 3], &[true, false, true]);
    let x = vec![vec![2.0], vec![4.0], vec![4.0]];
    for ties in [Ties::Efron, Ties::Breslow] {
        assert_monotone(cox_ph(&obs, &x, &opts(ties)), 0, &format!("{ties:?}"));
    }
}

/// Every subject with `x = s` fails before every subject with `x = 0`: the
/// supremum `−2 ln 4! = −6.3561076606958912` (mpmath) is not attained.  The
/// diagnosis depended on the covariate's units: before, `s = 10¹⁰` and
/// `10¹²` returned `Ok` ("converged", `β·s = 22.110377195520147`,
/// `se·s = 21917.85`) and `s = 10³`, `10⁸` failed as "the information matrix
/// became singular".  statsmodels `PHReg(arange(1, 9), [[1e10]*4 + [0]*4],
/// status=ones(8)).fit()` returns `params 2.893668726248029e-10`, `bse
/// 1.6856128105105916e-10`, `llf −6.786336925709188` without a warning.
/// With a second, non-separating covariate the message names covariate 1
/// (before, at `s = 10¹⁰`: `Ok` with `β₁ = 2.2e-9`).
#[test]
fn separation_is_diagnosed_in_any_units() {
    let obs = Observation::from_i64(&[1, 2, 3, 4, 5, 6, 7, 8], &[true; 8]);
    for s in [1e-3, 1.0, 1e3, 1e8, 1e10, 1e12] {
        let x: Vec<Vec<f64>> = (0..8).map(|i| vec![if i < 4 { s } else { 0.0 }]).collect();
        for ties in [Ties::Efron, Ties::Breslow] {
            assert_monotone(cox_ph(&obs, &x, &opts(ties)), 0, &format!("s = {s:e}"));
        }
        let x2: Vec<Vec<f64>> = (0..8)
            .map(|i| {
                vec![
                    [0.3, -1.0, 2.0, 0.7, 1.1, -0.4, 0.0, 1.6][i],
                    if i < 4 { s } else { 0.0 },
                ]
            })
            .collect();
        assert_monotone(
            cox_ph(&obs, &x2, &CoxOpts::default()),
            1,
            &format!("two covariates, s = {s:e}"),
        );
        // A loose or a tight tolerance does not change the diagnosis.
        for tol in [1e-4, 1e-14] {
            assert_monotone(
                cox_ph(
                    &obs,
                    &x,
                    &CoxOpts {
                        tol,
                        ..CoxOpts::default()
                    },
                ),
                0,
                &format!("s = {s:e}, tol = {tol:e}"),
            );
        }
    }
}

/// The fit is invariant to the covariate's units: `β̂·s` and `se·s` are
/// D1's for every `s`.  Before, `s = 10⁻²` and `10⁻³` were refused as a
/// monotone likelihood (`β` passed 25 — at `81.745` and `817.448` — while
/// the Newton step was still large).  statsmodels `PHReg(t1, x1*1e-2,
/// status=s1).fit()`: `params 87.59809887649098`, `bse 38.0902829654236`.
#[test]
fn the_fit_does_not_depend_on_the_covariates_units() -> R {
    let obs = d1_times();
    for s in [1e-3, 1e-2, 1e3, 1e9, 1e12] {
        let x: Vec<Vec<f64>> = D1_X.iter().map(|&v| vec![v * s]).collect();
        let fit = cox_ph(&obs, &x, &CoxOpts::default())?;
        assert!(fit.converged);
        close(
            fit.coefficients[0] * s,
            D1_BETA,
            FIT_REL,
            &format!("β·s, s = {s:e}"),
        );
        close(
            fit.standard_errors[0] * s,
            D1_SE,
            FIT_REL,
            &format!("se·s, s = {s:e}"),
        );
        close(fit.log_likelihood, D1_LL, FIT_REL, &format!("ℓ, s = {s:e}"));
    }
    Ok(())
}

/// Two covariates in very different units in one fit: D1's covariate and
/// `i·10⁻⁹` for subject `i`.  Before: refused as a monotone likelihood
/// (`β₁` passed 25).  mpmath `cox_mp.fit(t1, s1, [[x1[i], i*1e-9]],
/// 'efron')`: `β̂ = [3.6451935742743265, −1428279528.3264786]`, `se =
/// [2.4755877013577324, 1084425607.5498671]`; statsmodels `PHReg(t1,
/// column_stack([x1, arange(10)*1e-9]), status=s1).fit()` stops at
/// `[0.876, −1448.4]` with a `ConvergenceWarning`.  The two coefficients
/// are strongly correlated (`se/β ≈ 0.7`), so `1e-10`.
#[test]
fn covariates_in_very_different_units_fit_together() -> R {
    let x: Vec<Vec<f64>> = D1_X
        .iter()
        .enumerate()
        .map(|(i, &v)| vec![v, i as f64 * 1e-9])
        .collect();
    for ties in [Ties::Efron, Ties::Breslow] {
        let fit = cox_ph(&d1_times(), &x, &opts(ties))?;
        for (got, want) in fit
            .coefficients
            .iter()
            .zip([3.645_193_574_274_326_5, -1_428_279_528.326_478_6])
        {
            close(*got, want, 1e-10, &format!("β̂, {ties:?}"));
        }
        for (got, want) in fit
            .standard_errors
            .iter()
            .zip([2.475_587_701_357_732_4, 1_084_425_607.549_867_1])
        {
            close(*got, want, 1e-10, &format!("se, {ties:?}"));
        }
    }
    Ok(())
}

/// Units whose squares leave the `f64` range (`x·10²⁰⁰`, `x·10⁻²⁰⁰`) make
/// the information matrix `inf`/`0`; before, that was reported as "a
/// covariate is collinear with the others".  It is still an error (the
/// information holds squared spreads; statsmodels returns `nan` for
/// `x*1e200` and raises `Singular matrix` for `x*1e-200`), now naming the
/// covariate and the remedy.
#[test]
fn covariates_whose_squares_leave_the_f64_range_are_named() {
    let obs = d1_times();
    for s in [1e200, 1e-200] {
        let x: Vec<Vec<f64>> = D1_X.iter().map(|&v| vec![v * s]).collect();
        match cox_ph(&obs, &x, &CoxOpts::default()) {
            Err(SymplexError::ComputationFailed { reason, .. }) => {
                assert!(
                    reason.contains("covariate 0") && reason.contains("rescale"),
                    "{reason}"
                );
            }
            other => panic!("s = {s:e}: expected ComputationFailed, got {other:?}"),
        }
    }
}

/// A finite but large `β̂` on a covariate in thousandths: forty subjects,
/// all but one inversion of a separated order.  Before: refused as a
/// monotone likelihood (`β` passed 25, at `2160.66`).  mpmath
/// `cox_mp.fit(t, s, x, 'efron')` (oracle_2b_more2.py): `β̂ =
/// 1926.7698538651934`, `se = 418.46162274241273`, `ℓ = −97.374521181042557`;
/// statsmodels `PHReg(...).fit()`: `1926.76985386519`, `418.4616227424129`.
/// The information is small (`se/β̂ ≈ 0.2`), so `1e-10`.
#[test]
fn a_near_separated_fit_on_a_small_scale_is_not_refused() -> R {
    let ones = [
        1usize, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 30,
    ];
    let t: Vec<i64> = (1..=40).collect();
    let e: Vec<bool> = (0..40).map(|i| i != 34).collect();
    let x: Vec<Vec<f64>> = (0..40)
        .map(|i| vec![if ones.contains(&i) { 0.001 } else { 0.0 }])
        .collect();
    let fit = cox_ph(&Observation::from_i64(&t, &e), &x, &CoxOpts::default())?;
    close(fit.coefficients[0], 1_926.769_853_865_193_4, 1e-10, "β̂");
    close(fit.standard_errors[0], 418.461_622_742_412_73, 1e-10, "se");
    close(fit.log_likelihood, -97.374_521_181_042_557, 1e-12, "ℓ");
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Covariate offsets and 3. per-risk-set shifts
// ═══════════════════════════════════════════════════════════════════════════

/// `x + c` changes every `η` by the same `cβ`, which cancels in the partial
/// likelihood.  Before, the information cancelled catastrophically:
/// `c = 10⁴` gave `se = 0.3809028275681563`, `c = 10⁶` `0.3809253888964465`
/// (both tie corrections), `c = 10⁸` and `10¹⁰` "the information matrix
/// became singular".  statsmodels on `x + 1e6` is also off:
/// `bse 0.38091189501986294` (efron), `0.38090177555344723` (breslow).
/// mpmath on the offset data itself (not just by invariance) gives D1's
/// values at every `c` (oracle_2b.py section 1).
#[test]
fn a_covariate_offset_costs_no_digits() -> R {
    let obs = d1_times();
    for c in [1e4, 1e6, 1e8, 1e10] {
        let x: Vec<Vec<f64>> = D1_X.iter().map(|&v| vec![v + c]).collect();
        for ties in [Ties::Efron, Ties::Breslow] {
            let fit = cox_ph(&obs, &x, &opts(ties))?;
            let label = format!("c = {c:e}, {ties:?}");
            close(
                fit.coefficients[0],
                D1_BETA,
                FIT_REL,
                &format!("β̂, {label}"),
            );
            close(
                fit.standard_errors[0],
                D1_SE,
                FIT_REL,
                &format!("se, {label}"),
            );
            close(fit.log_likelihood, D1_LL, FIT_REL, &format!("ℓ, {label}"));
            close(
                fit.null_log_likelihood,
                D1_LL0,
                FIT_REL,
                &format!("ℓ₀, {label}"),
            );
            close(
                fit.score_statistic(),
                D1_SCORE,
                FIT_REL,
                &format!("score, {label}"),
            );
            // The linear predictor keeps the caller's (uncentred) covariates.
            close(
                fit.linear_predictors()[0],
                (3.0 + c) * fit.coefficients[0],
                1e-15,
                &format!("η₀, {label}"),
            );
        }
    }
    Ok(())
}

/// A subject censored at `t = 1`, before the first event, with `x = 1000`
/// is in no risk set, yet its `η = 876` was the global shift, so every
/// risk set underflowed to `S0 = 0`.  Before: "no convergence in 50 Newton
/// steps" (for `x = 1000` and `2000`).  mpmath `cox_mp.fit` on the eleven
/// subjects: D1's `β̂`, `se`, `ℓ`; statsmodels (which drops the subject):
/// `params 0.8759809887649095`, `bse 0.3809028296542359`.
#[test]
fn a_subject_outside_every_risk_set_cannot_underflow_the_others() -> R {
    for extreme in [1000.0, 2000.0] {
        let mut obs = d1_times();
        obs.push(Observation {
            time: qi(1),
            event: false,
        });
        let mut x: Vec<Vec<f64>> = D1_X.iter().map(|&v| vec![v]).collect();
        x.push(vec![extreme]);
        let fit = cox_ph(&obs, &x, &CoxOpts::default())?;
        close(fit.coefficients[0], D1_BETA, FIT_REL, "β̂");
        close(fit.standard_errors[0], D1_SE, FIT_REL, "se");
        close(fit.log_likelihood, D1_LL, FIT_REL, "ℓ");
    }
    Ok(())
}

/// A subject with `x = 1000` fails first (`t = 1`): its risk set is
/// dominated by `e^{876}`, and every later risk set — without it — lies
/// ~870 below the global maximum.  Before: "no convergence in 50 Newton
/// steps"; statsmodels `PHReg(...).fit()` returns `nan`.  mpmath
/// `cox_mp.fit` (oracle_2b.py section 3): `β̂`, `se`, `ℓ` are D1's to 17
/// digits (the first factor is `1 − O(e^{−870})`), `ℓ₀ =
/// −14.506575572319895`, score `10.197758398168895`; the baseline hazard
/// at `t = 1` is `3.7e-381` (`0` in `f64`) and at `t = 2`
/// `1/Σ_{t ≥ 2} e^{xβ̂} = 0.0028588408434754058`.  The extreme subject
/// makes the centred covariates larger (mean ≈ 93.6), so `1e-11`.
#[test]
fn a_dominant_early_failure_does_not_underflow_the_later_risk_sets() -> R {
    let mut obs = d1_times();
    obs.push(Observation {
        time: qi(1),
        event: true,
    });
    let mut x: Vec<Vec<f64>> = D1_X.iter().map(|&v| vec![v]).collect();
    x.push(vec![1000.0]);
    let fit = cox_ph(&obs, &x, &CoxOpts::default())?;
    close(fit.coefficients[0], D1_BETA, 1e-11, "β̂");
    close(fit.standard_errors[0], D1_SE, 1e-11, "se");
    close(fit.log_likelihood, D1_LL, 1e-11, "ℓ");
    close(
        fit.null_log_likelihood,
        -14.506_575_572_319_895,
        1e-12,
        "ℓ₀",
    );
    close(
        fit.score_statistic(),
        10.197_758_398_168_895,
        1e-12,
        "score",
    );
    let rows = fit.baseline_hazard();
    assert_eq!(rows[0].time, qi(1));
    assert_eq!(rows[0].hazard, 0.0);
    assert_eq!(rows[1].time, qi(2));
    close(rows[1].hazard, 0.002_858_840_843_475_405_8, 1e-11, "ĥ₀(2)");
    let m = fit.martingale_residuals();
    assert!(m.iter().all(|v| v.is_finite()), "{m:?}");
    assert!(m.iter().sum::<f64>().abs() < 1e-12, "{m:?}");
    // The extreme subject is (to rounding) its whole risk set: residual 0.
    let s = fit.schoenfeld_residuals();
    assert!(s.iter().all(|r| r[0].is_finite()), "{s:?}");
    assert_eq!(s[7][0], 0.0);
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Risk-set table, martingale residuals and concordance in O(n log n)
// ═══════════════════════════════════════════════════════════════════════════

/// A deterministic sample with tied times (values mod `ties`), tied
/// linear predictors (a 0/1 and a small-integer covariate), censoring and
/// two strata.
fn sample(n: usize, seed: u64, ties: u64) -> (Vec<Observation>, Vec<Vec<f64>>, Vec<usize>) {
    let mut s = seed;
    let mut next = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        s >> 33
    };
    let mut obs = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    let mut strata = Vec::with_capacity(n);
    for _ in 0..n {
        obs.push(Observation {
            time: qi((next() % ties) as i64),
            event: next() % 3 != 0,
        });
        x.push(vec![(next() % 2) as f64, (next() % 5) as f64 - 2.0]);
        strata.push((next() % 2) as usize);
    }
    (obs, x, strata)
}

/// The concordance index and martingale residuals were computed pair by
/// pair / event time by subject; now by a Fenwick tree over `η` ranks and
/// a log-space cumulative hazard.  Oracle: the definitions, counted and
/// summed directly here (Harrell's usable/concordant/tied pairs within a
/// stratum; `Mᵢ = δᵢ − e^{ηᵢ} Λ̂₀(tᵢ)` with `Λ̂₀` read off
/// `baseline_hazard`, `tᵢ` included).
#[test]
fn concordance_and_martingale_residuals_follow_their_definitions() -> R {
    for (seed, ties) in [(1, 40), (2, 7), (3, 1000)] {
        let (obs, x, strata) = sample(300, seed, ties);
        for t in [Ties::Efron, Ties::Breslow] {
            let fit = cox_ph_stratified(&obs, &x, &strata, &opts(t))?;
            let eta = fit.linear_predictors();
            let (mut usable, mut twice) = (0i64, 0i64);
            for i in 0..obs.len() {
                for j in 0..obs.len() {
                    if obs[i].event && strata[i] == strata[j] && obs[i].time < obs[j].time {
                        usable += 1;
                        twice += if eta[i] > eta[j] {
                            2
                        } else {
                            i64::from(eta[i] == eta[j])
                        };
                    }
                }
            }
            assert_eq!(fit.concordance()?, q(twice, 2 * usable), "seed {seed}");
            let rows = fit.baseline_hazard();
            let m = fit.martingale_residuals();
            for i in 0..obs.len() {
                let cum = rows
                    .iter()
                    .filter(|r| r.stratum == strata[i] && r.time <= obs[i].time)
                    .map(|r| r.hazard)
                    .sum::<f64>();
                let want = f64::from(u8::from(obs[i].event)) - eta[i].exp() * cum;
                assert!(
                    (m[i] - want).abs() < 1e-12,
                    "seed {seed}, subject {i}: {} vs {want}",
                    m[i]
                );
            }
        }
    }
    Ok(())
}

/// Total times of `a` and `b` over `rounds` interleaved rounds.
fn interleaved_totals(
    rounds: usize,
    mut a: impl FnMut() -> Duration,
    mut b: impl FnMut() -> Duration,
) -> (Duration, Duration) {
    let (mut ta, mut tb) = (Duration::ZERO, Duration::ZERO);
    for _ in 0..rounds {
        ta += a();
        tb += b();
    }
    (ta, tb)
}

/// Fit, martingale residuals and concordance scaled like `n·T` or `n²`
/// (`T` distinct event times): 7.5 s + 3.5 s + 3.6 s for 6 000 subjects in
/// a debug build, so eight times the subjects cost 64 times as much.
/// Sort-and-sweep costs `n log n` (about 11 times here).  As in
/// `v28_perf`, the two workloads are timed in interleaved rounds so that
/// both see the same machine load, and the bound sits near the geometric
/// mean of 11 and 64.
#[test]
fn cox_accessors_scale_as_n_log_n() {
    let work = |n: usize| {
        let (obs, x, _) = sample(n, 11, 1_000_000);
        move || {
            let t = Instant::now();
            let fit = cox_ph(&obs, &x, &CoxOpts::default()).expect("fit");
            let m = fit.martingale_residuals();
            let c = fit.concordance().expect("usable pairs");
            assert!(m.iter().all(|v| v.is_finite()) && c > Q::from_integer(0.into()));
            t.elapsed()
        }
    };
    let (small, large) = interleaved_totals(5, work(400), work(3_200));
    assert!(
        large < small * 26,
        "3 200 subjects took {large:?}, 400 took {small:?} (ratio {:.1})",
        large.as_secs_f64() / small.as_secs_f64()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Documented semantics of the Schoenfeld residuals
// ═══════════════════════════════════════════════════════════════════════════

/// The doc said the Schoenfeld residuals "sum to zero over the events (the
/// score equation)".  They use the Breslow risk-set mean under both tie
/// corrections (as statsmodels does, and as `tests/v18` pins), so their
/// sum is the Breslow score: zero for a Breslow fit, not for an Efron fit
/// with ties.  D2 of `tests/v18/v18_cox.rs`: statsmodels
/// `PHReg(t2, x2, status=s2, ties='efron').fit().schoenfeld_residuals[s2 ==
/// 1].sum(0)` = `[0.0482229290511067, -0.0562979974243889]` =
/// `PHReg(..., ties='breslow').score(efron_params)`; the Breslow fit's sums
/// are `[-8.9e-16, 2.8e-16]`.  statsmodels' own fit converges to about
/// `1e-12`, so `1e-9` absolute.
#[test]
fn schoenfeld_residuals_sum_to_the_breslow_score() -> R {
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
    let column_sums = |fit: &CoxModel| -> Vec<f64> {
        let s = fit.schoenfeld_residuals();
        (0..2).map(|j| s.iter().map(|r| r[j]).sum()).collect()
    };
    let efron = column_sums(&cox_ph(&obs, &x, &opts(Ties::Efron))?);
    for (got, want) in efron
        .iter()
        .zip([0.048_222_929_051_106_7, -0.056_297_997_424_388_9])
    {
        assert!((got - want).abs() < 1e-9, "{got} vs {want}");
    }
    let breslow = column_sums(&cox_ph(&obs, &x, &opts(Ties::Breslow))?);
    assert!(breslow.iter().all(|v| v.abs() < 1e-12), "{breslow:?}");
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Survival and hazard functions of a distribution
// ═══════════════════════════════════════════════════════════════════════════

/// `survival_function` was `1 − F(t)`: `0` for `N(0, 1)` at `t = 26`, and
/// `hazard_function` `PrecisionExhausted` (`f/(1 − Φ)` is `0/0`) from
/// `t = 20`.  mpmath: `ncdf(-26) = 2.4760633155033892858e-149` (scipy
/// `norm.sf(26)` = `2.476063315503107e-149`); `npdf(t)/ncdf(-t)` =
/// `20.049753068527850542` at 20, `40.024968847207263723` at 40.  `erfc` in
/// the far tail is good to a few `1e-14`.
#[test]
fn survival_and_hazard_functions_use_the_upper_tail() -> R {
    let ctx = Context::new();
    let n = Distribution::normal(ctx.int(0), ctx.int(1));
    close(
        survival_function(&n, &ctx.int(26)).eval_f64()?,
        2.476_063_315_503_389_285_8e-149,
        1e-13,
        "S(26)",
    );
    close(
        hazard_function(&n, &ctx.int(20)).eval_f64()?,
        20.049_753_068_527_850_542,
        1e-13,
        "h(20)",
    );
    close(
        hazard_function(&n, &ctx.int(40)).eval_f64()?,
        40.024_968_847_207_263_723,
        1e-13,
        "h(40)",
    );
    Ok(())
}

/// Below the support the closed forms were extrapolated: `S(−1) = e^{5/63}`
/// and `h(−1) = 5/63` for `Exponential(5/63)`, `S(−1) = e^{−1}` and `h(−1)
/// = −2` for `Weibull(1, 2)`, `S(−1) = 3/2` and `S(3) = −1/2` for
/// `U(0, 2)`.  By definition `S = 1`, `h = 0` below the support and `S = 0`
/// at or above its upper end.  Symbolic `t` keeps the closed form on the
/// support (`h = λ`, `h = 2t`).
#[test]
fn survival_and_hazard_functions_outside_the_support() {
    let ctx = Context::new();
    let e = Distribution::exponential(ctx.rational(5, 63));
    let w = Distribution::weibull(ctx.int(1), ctx.int(2));
    let u = Distribution::uniform(ctx.int(0), ctx.int(2));
    let m1 = ctx.int(-1);
    for d in [&e, &w, &u] {
        assert_eq!(survival_function(d, &m1), ctx.one());
        assert_eq!(hazard_function(d, &m1), ctx.zero());
    }
    assert_eq!(survival_function(&u, &ctx.int(3)), ctx.zero());
    assert_eq!(survival_function(&u, &ctx.int(2)), ctx.zero());
    assert_eq!(
        survival_function(&u, &ctx.rational(1, 2)),
        ctx.rational(3, 4)
    );
    let t = ctx.symbol("t");
    assert_eq!(hazard_function(&e, &t), ctx.rational(5, 63));
    assert_eq!(hazard_function(&w, &t), ctx.int(2) * t.clone());
    assert_eq!(survival_function(&w, &t), (-(t.pow(&ctx.int(2)))).exp());
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Kaplan–Meier quantiles; the log-rank test
// ═══════════════════════════════════════════════════════════════════════════

/// `quantile(−1)` and `quantile_strict(−1)` were the first event time
/// (`S ≤ 2` everywhere); a level outside `[0, 1]` is not a quantile level.
#[test]
fn km_quantile_levels_outside_the_unit_interval_are_undefined() -> R {
    let km = KaplanMeier::fit(&Observation::from_i64(&[1, 2, 3, 4], &[true; 4]))?;
    for p in [qi(-1), q(-1, 1000), q(3, 2), qi(2)] {
        assert_eq!(km.quantile(&p), None, "{p}");
        assert_eq!(km.quantile_strict(&p), None, "{p}");
    }
    assert_eq!(km.quantile(&Q::from_integer(0.into())), Some(qi(1)));
    assert_eq!(km.quantile(&qi(1)), Some(qi(4)));
    assert_eq!(km.quantile_strict(&qi(1)), None);
    Ok(())
}

/// Not a behaviour change: the doc said `quantile_strict` *is*
/// `SurvfuncRight.quantile`, but statsmodels compares a rounded `S`.  Forty
/// events, seven at `t = 0`, thirteen at `1`, twenty at `2`: exactly
/// `S(1) = (33/40)(20/33) = 1/2`, while statsmodels
/// `SurvfuncRight([0]*7 + [1]*13 + [2]*20, ones(40))` has `surv_prob =
/// [0.825, 0.49999999999999994, 0.0]` and `quantile(0.5) = 1.0` — the
/// non-strict answer.  The exact strict rule gives `2`.
#[test]
fn km_quantile_strict_compares_exactly_where_statsmodels_rounds() -> R {
    let mut times = vec![0i64; 7];
    times.extend([1i64; 13]);
    times.extend([2i64; 20]);
    let km = KaplanMeier::fit(&Observation::from_i64(&times, &[true; 40]))?;
    assert_eq!(km.survival_at(&qi(1)), q(1, 2));
    assert_eq!(km.quantile_strict(&q(1, 2)), Some(qi(2)));
    assert_eq!(km.quantile(&q(1, 2)), Some(qi(1)));
    Ok(())
}

/// Not a bug before (the symbolic `1 − cdf` was rewritten by `simplify`
/// into `erfc`/`exp` terms); pinned now that the p-value is the upper tail
/// `Γ(ν/2, x/2)/Γ(ν/2)` itself.  Groups failing in consecutive blocks
/// (complete separation): exact statistics from `fractions` (oracle_2b.py
/// section 7), p-values from mpmath `gammainc(ν/2, x/2, inf,
/// regularized=True)`: `1.0414427666891065e-22` (k = 2, 40 per group),
/// `1.7575356741428869e-49` (3 × 40), `4.3033647832365176e-59` (4 × 30).
/// statsmodels `survdiff` returns the same statistics and p = `0.0`
/// (`1 − chi2.cdf`); scipy `chi2.sf` agrees with mpmath to `3e-15`.
#[test]
fn log_rank_p_values_far_below_1e_minus_16() -> R {
    let ctx = Context::new();
    for (k, per, stat, p) in [
        (
            2usize,
            40usize,
            96.194_300_486_728_368,
            1.041_442_766_689_106_5e-22,
        ),
        (3, 40, 224.525_513_827_958_96, 1.757_535_674_142_886_9e-49),
        (4, 30, 273.954_888_184_350_63, 4.303_364_783_236_517_6e-59),
    ] {
        let n = k * per;
        let times: Vec<i64> = (1..=n as i64).collect();
        let obs = Observation::from_i64(&times, &vec![true; n]);
        let groups: Vec<usize> = (0..n).map(|i| i / per).collect();
        let r = log_rank_test(&ctx, &obs, &groups)?;
        close(
            r.statistic_f64()?,
            stat,
            1e-15,
            &format!("statistic, k = {k}"),
        );
        close(r.p_value_f64()?, p, 1e-13, &format!("p, k = {k}"));
        if k == 2 {
            assert_eq!(
                r.statistic_exact(),
                Some(Q::new(
                    big("496398102583159017449721421503492515881293139229392501698887097078025"),
                    big("5160369170225896471942653553923167620306357722928288979545171482527"),
                ))
            );
        }
    }
    Ok(())
}

/// Documented now: a group whose members are all censored before the
/// first event has nobody at risk at any event time, its `O − E` is `0`
/// and the covariance matrix singular — a `ComputationFailed` (statsmodels
/// `survdiff([5, 6, 7, 1, 2], [1, 1, 0, 0, 0], [0, 0, 0, 1, 1])` raises
/// `LinAlgError: Singular matrix`).
#[test]
fn log_rank_with_a_group_never_at_risk_is_a_documented_error() {
    let ctx = Context::new();
    let obs = Observation::from_i64(&[5, 6, 7, 1, 2], &[true, true, false, false, false]);
    assert!(matches!(
        log_rank_test(&ctx, &obs, &[0, 0, 0, 1, 1]),
        Err(SymplexError::ComputationFailed { .. })
    ));
}
