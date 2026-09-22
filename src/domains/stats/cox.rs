//! Cox proportional-hazards regression: how covariates change the hazard
//! of an event whose time may be right-censored, without a model for the
//! baseline hazard itself.
//!
//! The model is `h(t | x) = h₀(t) · exp(xᵀβ)`.  `β` maximises Cox's
//! partial likelihood — a product over the distinct event times of the
//! probability that the subjects who failed at that time were the ones to
//! fail, given the risk set — by Newton–Raphson with the analytic score
//! and information matrix, the [Efron](Ties::Efron) (default) or
//! [Breslow](Ties::Breslow) correction for tied event times, and
//! step-halving.  Everything downstream of `β̂` follows: hazard ratios,
//! Wald standard errors and intervals, the likelihood-ratio / Wald / score
//! trio that `R`'s `summary(coxph)` prints, the Breslow baseline hazard,
//! Schoenfeld and martingale residuals, and Harrell's concordance index,
//! which is a ratio of pair counts and therefore an exact rational.
//! `statsmodels.duration.hazard_regression.PHReg` is the reference named
//! in the tests.
//!
//! ```
//! use symplex::prelude::*;
//! use symplex::stats::cox::{cox_ph, CoxOpts};
//! use symplex::stats::survival::Observation;
//!
//! // Ten subjects, one covariate, three censored.
//! let obs = Observation::from_i64(
//!     &[4, 7, 2, 9, 12, 5, 15, 3, 11, 8],
//!     &[true, true, true, false, true, true, false, true, true, false],
//! );
//! let x: Vec<Vec<f64>> = [3.0, 1.0, 5.0, 2.0, 0.0, 4.0, 1.0, 6.0, 2.0, 3.0]
//!     .iter().map(|&v| vec![v]).collect();
//! let fit = cox_ph(&obs, &x, &CoxOpts::default())?;
//! // statsmodels PHReg(t, x, status=s, ties='efron').fit(): params [0.8759809887649096],
//! // bse [0.3809028296542367], llf -8.465216276861835
//! assert!((fit.coefficients[0] - 0.875_980_988_764_909_6).abs() < 1e-8);
//! assert!((fit.standard_errors[0] - 0.380_902_829_654_236_7).abs() < 1e-8);
//! assert!((fit.log_likelihood - (-8.465_216_276_861_835)).abs() < 1e-9);
//! // Harrell's C: 31 concordant of 38 usable pairs, exactly.
//! assert_eq!(fit.concordance()?, symplex::linprog::q(31, 38));
//! # Ok::<(), SymplexError>(())
//! ```
//!
//! # Conventions
//!
//! * A subject censored at an event time is still **at risk** at that time
//!   (the convention of `coxph`, `PHReg` and this crate's
//!   [`KaplanMeier`](super::survival::KaplanMeier)).
//! * There is no intercept: it is absorbed by the baseline hazard.
//! * Covariates are `f64`; a fit is numerical, so the coefficients,
//!   standard errors and residuals are `f64`.  Counts, event times and the
//!   concordance index are exact.

use super::common::{
    check_confidence, chi_squared_sf, ex_usize, information_cholesky, invalid, qu, wald_summary,
    z_two_sided,
};
use super::data::Q;
use super::hypothesis::{Alternative, TestResult};
use super::survival::Observation;
use crate::api::context::Context;
use crate::base::dense_f64::{self, dot};
use crate::base::errors::SymplexError;
use crate::base::interval::Interval;

/// A coefficient beyond this magnitude with the Newton step not yet small
/// is taken as a monotone likelihood (`β → ±∞`): a hazard ratio of
/// `e^25 ≈ 7 × 10¹⁰`.
const DIVERGENCE_BOUND: f64 = 25.0;

/// Maximum number of step halvings per Newton iteration.
const MAX_HALVINGS: u32 = 30;

fn failed(op: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::computation_failed(op, reason)
}

/// How tied event times enter the partial likelihood.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Ties {
    /// Breslow's approximation: every subject failing at `t` competes with
    /// the whole risk set, `Π_t exp(Σ_{i∈D_t} xᵢᵀβ) / (Σ_{j∈R_t} e^{xⱼᵀβ})^{d_t}`.
    /// `PHReg(ties='breslow')`, `coxph(ties = "breslow")`.
    Breslow,
    /// Efron's approximation: the `l`-th of the `d_t` tied failures sees
    /// the risk-set sum reduced by `l/d_t` of the tied subjects' own
    /// weights.  Closer to the exact partial likelihood; the default of
    /// `PHReg` and `coxph`.
    #[default]
    Efron,
}

/// Options of [`cox_ph`]'s Newton–Raphson iteration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoxOpts {
    /// Tie correction (default [`Ties::Efron`]).
    pub ties: Ties,
    /// Maximum number of Newton steps (default `50`).
    pub max_iter: usize,
    /// Convergence when both `max_j |Δβ_j| ≤ tol · max(1, max_j |β_j|)`
    /// and `|Δℓ| ≤ tol · max(1, |ℓ|)` (default `1e-9`).
    pub tol: f64,
}

impl Default for CoxOpts {
    fn default() -> Self {
        Self {
            ties: Ties::Efron,
            max_iter: 50,
            tol: 1e-9,
        }
    }
}

/// One distinct event time of the Breslow baseline-hazard estimate.
#[derive(Clone, Debug, PartialEq)]
pub struct BaselineHazardRow {
    /// The stratum (`0` for an unstratified fit).
    pub stratum: usize,
    /// The event time.
    pub time: Q,
    /// `ĥ₀(t) = d_t / Σ_{j∈R_t} exp(xⱼᵀβ̂)`: the Breslow increment.
    pub hazard: f64,
    /// `Λ̂₀(t) = Σ_{s ≤ t} ĥ₀(s)` within the stratum, `t` included.
    pub cumulative: f64,
}

/// A fitted Cox proportional-hazards model (`PHReg(...).fit()`), in `f64`.
#[derive(Clone, Debug, PartialEq)]
pub struct CoxModel {
    /// `β̂`, the maximum partial-likelihood coefficients (`params`).
    pub coefficients: Vec<f64>,
    /// `√diag(I(β̂)⁻¹)` with `I` the observed information (`bse`).
    pub standard_errors: Vec<f64>,
    /// `z_j = β̂_j / se_j` (`tvalues`).
    pub z_values: Vec<f64>,
    /// Two-sided normal p-values `erfc(|z_j|/√2)` (`pvalues`).
    pub p_values: Vec<f64>,
    /// `ℓ(β̂)`, the maximised log partial likelihood (`llf`).
    pub log_likelihood: f64,
    /// `ℓ(0)`, the log partial likelihood with no covariate effect, under
    /// the same tie correction (`model.loglike(zeros)`).
    pub null_log_likelihood: f64,
    /// `I(β̂)⁻¹` (`cov_params()`).
    pub cov_params: Vec<Vec<f64>>,
    /// Number of observations.
    pub nobs: usize,
    /// Number of events (uncensored observations).
    pub n_events: usize,
    /// The tie correction used.
    pub ties: Ties,
    /// Newton steps taken.
    pub iterations: usize,
    /// Whether both convergence criteria were met.  `true` for every model
    /// [`cox_ph`] returns — a fit that exhausts `max_iter` is an error —
    /// and kept alongside `iterations` as the fit's diagnostic record.
    pub converged: bool,
    obs: Vec<Observation>,
    x: Vec<Vec<f64>>,
    strata: Vec<usize>,
    table: Vec<EventTime>,
    score_statistic: f64,
    wald_statistic: f64,
}

/// One distinct event time of one stratum, for the backward sweep over
/// risk sets.
#[derive(Clone, Debug, PartialEq)]
struct EventTime {
    stratum: usize,
    time: Q,
    /// Subjects entering the risk set when the sweep (descending in time)
    /// reaches this time: those with `time ≤ tᵢ < next larger event time`.
    enter: Vec<usize>,
    /// Subjects failing at this time.
    events: Vec<usize>,
}

/// The risk-set sums at one event time: `S0 = Σ_R wⱼ`, `S1 = Σ_R wⱼ xⱼ`,
/// `S2 = Σ_R wⱼ xⱼ xⱼᵀ`, and the same over the failures `D` at that time.
struct RiskSums {
    s0: f64,
    s1: Vec<f64>,
    s2: Vec<Vec<f64>>,
    d0: f64,
    d1: Vec<f64>,
    d2: Vec<Vec<f64>>,
}

/// `ℓ`, the score `∂ℓ/∂β` and the information `−∂²ℓ/∂β²` at one `β`.
struct Evaluation {
    ll: f64,
    score: Vec<f64>,
    info: Vec<Vec<f64>>,
}

/// The distinct event times of every stratum with their risk-set entries,
/// ordered by stratum and then by *descending* time (the sweep order).
fn build_table(obs: &[Observation], strata: &[usize]) -> Vec<EventTime> {
    let mut labels: Vec<usize> = strata.to_vec();
    labels.sort_unstable();
    labels.dedup();
    let mut table = Vec::new();
    for s in labels {
        let members: Vec<usize> = (0..obs.len()).filter(|&i| strata[i] == s).collect();
        let mut times: Vec<Q> = members
            .iter()
            .filter(|&&i| obs[i].event)
            .map(|&i| obs[i].time.clone())
            .collect();
        times.sort();
        times.dedup();
        // Descending: the risk set at t is everyone with tᵢ ≥ t, so a
        // subject enters at the largest event time ≤ tᵢ.
        for (k, t) in times.iter().enumerate().rev() {
            let upper = times.get(k + 1);
            let enter = members
                .iter()
                .copied()
                .filter(|&i| obs[i].time >= *t && upper.is_none_or(|u| obs[i].time < *u))
                .collect();
            let events = members
                .iter()
                .copied()
                .filter(|&i| obs[i].event && obs[i].time == *t)
                .collect();
            table.push(EventTime {
                stratum: s,
                time: t.clone(),
                enter,
                events,
            });
        }
    }
    table
}

/// Sweep the risk sets given the weights `w = exp(η − max η)`, one
/// [`RiskSums`] per table entry, in table order.
fn sweep(table: &[EventTime], x: &[Vec<f64>], w: &[f64], p: usize) -> Vec<RiskSums> {
    let mut out = Vec::with_capacity(table.len());
    let mut s0 = 0.0;
    let mut s1 = vec![0.0; p];
    let mut s2 = vec![vec![0.0; p]; p];
    let mut stratum = None;
    for entry in table {
        if stratum != Some(entry.stratum) {
            stratum = Some(entry.stratum);
            s0 = 0.0;
            s1.iter_mut().for_each(|v| *v = 0.0);
            s2.iter_mut().flatten().for_each(|v| *v = 0.0);
        }
        for &i in &entry.enter {
            let wi = w[i];
            s0 += wi;
            for a in 0..p {
                s1[a] += wi * x[i][a];
                for b in 0..p {
                    s2[a][b] += wi * x[i][a] * x[i][b];
                }
            }
        }
        let mut d0 = 0.0;
        let mut d1 = vec![0.0; p];
        let mut d2 = vec![vec![0.0; p]; p];
        for &i in &entry.events {
            let wi = w[i];
            d0 += wi;
            for a in 0..p {
                d1[a] += wi * x[i][a];
                for b in 0..p {
                    d2[a][b] += wi * x[i][a] * x[i][b];
                }
            }
        }
        out.push(RiskSums {
            s0,
            s1: s1.clone(),
            s2: s2.clone(),
            d0,
            d1,
            d2,
        });
    }
    out
}

/// The linear predictor and the hazard weights at one `β`.
struct LinearPredictor {
    /// `ηᵢ = xᵢᵀβ`.
    eta: Vec<f64>,
    /// `maxᵢ ηᵢ`.
    shift: f64,
    /// `wᵢ = exp(ηᵢ − shift)`: the shift leaves `ℓ`, the score and the
    /// information unchanged and keeps every risk-set sum finite.
    w: Vec<f64>,
}

fn linear_predictor(x: &[Vec<f64>], beta: &[f64]) -> LinearPredictor {
    let eta: Vec<f64> = x.iter().map(|row| dot(row, beta)).collect();
    let shift = eta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let w = eta.iter().map(|e| (e - shift).exp()).collect();
    LinearPredictor { eta, shift, w }
}

fn evaluate(table: &[EventTime], x: &[Vec<f64>], beta: &[f64], ties: Ties) -> Evaluation {
    let p = beta.len();
    let lp = linear_predictor(x, beta);
    let sums = sweep(table, x, &lp.w, p);
    let mut ll = 0.0;
    let mut score = vec![0.0; p];
    let mut info = vec![vec![0.0; p]; p];
    for (entry, s) in table.iter().zip(&sums) {
        let d = entry.events.len();
        for &i in &entry.events {
            // ηᵢ − shift: the shifts cancel against the ln S0 terms below.
            ll += lp.eta[i] - lp.shift;
            for a in 0..p {
                score[a] += x[i][a];
            }
        }
        // The l-th tied failure sees the risk-set sums reduced by a
        // fraction f_l of the failures' own sums: f_l = l/d for Efron,
        // f_l = 0 for Breslow.
        for l in 0..d {
            let f = match ties {
                Ties::Efron => l as f64 / d as f64,
                Ties::Breslow => 0.0,
            };
            let a0 = s.s0 - f * s.d0;
            ll -= a0.ln();
            let mean: Vec<f64> = (0..p).map(|a| (s.s1[a] - f * s.d1[a]) / a0).collect();
            for a in 0..p {
                score[a] -= mean[a];
                for b in 0..p {
                    info[a][b] += (s.s2[a][b] - f * s.d2[a][b]) / a0 - mean[a] * mean[b];
                }
            }
        }
    }
    Evaluation { ll, score, info }
}

fn validate(
    op: &'static str,
    obs: &[Observation],
    x: &[Vec<f64>],
    strata: &[usize],
    opts: &CoxOpts,
) -> Result<usize, SymplexError> {
    let n = obs.len();
    if n == 0 {
        return Err(invalid(op, "at least one observation is required"));
    }
    if x.len() != n {
        return Err(invalid(
            op,
            format!("{n} observations but x has {} rows", x.len()),
        ));
    }
    if strata.len() != n {
        return Err(invalid(
            op,
            format!("{n} observations but {} stratum labels", strata.len()),
        ));
    }
    let p = x.first().map_or(0, Vec::len);
    if p == 0 {
        return Err(invalid(op, "at least one covariate is required"));
    }
    if let Some((i, r)) = x.iter().enumerate().find(|(_, r)| r.len() != p) {
        return Err(invalid(
            op,
            format!("row {i} of x has {} entries, expected {p}", r.len()),
        ));
    }
    if let Some((i, j)) = x
        .iter()
        .enumerate()
        .find_map(|(i, r)| r.iter().position(|v| !v.is_finite()).map(|j| (i, j)))
    {
        return Err(invalid(op, format!("x[{i}][{j}] is not finite")));
    }
    if let Some(j) = (0..p).find(|&j| x.iter().all(|r| r[j] == x[0][j])) {
        return Err(invalid(
            op,
            format!("covariate {j} is constant: its coefficient is not identified"),
        ));
    }
    if !obs.iter().any(|o| o.event) {
        return Err(invalid(op, "no events were observed"));
    }
    if opts.max_iter == 0 {
        return Err(invalid(op, "max_iter must be positive"));
    }
    if opts.tol.is_nan() || opts.tol <= 0.0 {
        return Err(invalid(
            op,
            format!("tol must be positive, got {}", opts.tol),
        ));
    }
    Ok(p)
}

/// Fit `h(t | x) = h₀(t) exp(xᵀβ)` by Newton–Raphson on Cox's log partial
/// likelihood from `β = 0`: `β ← β + I(β)⁻¹ U(β)` with the analytic score
/// `U` and information `I`, the step halved while `ℓ` would decrease, until
/// both `max_j |Δβ_j| ≤ tol · max(1, max_j |β_j|)` and
/// `|Δℓ| ≤ tol · max(1, |ℓ|)`.  `x[i]` is the covariate row of `obs[i]`;
/// there is no intercept.  `PHReg(time, x, status=event, ties=…).fit()`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::cox::{cox_ph, CoxOpts, Ties};
/// use symplex::stats::survival::Observation;
///
/// // Two groups of eight, a single 0/1 covariate, no tied event times.
/// let obs = Observation::from_i64(
///     &[3, 5, 6, 7, 8, 10, 12, 14, 4, 9, 11, 13, 15, 16, 18, 20],
///     &[true, false, true, true, false, true, true, false, true, true, false, true, true, true, false, true],
/// );
/// let x: Vec<Vec<f64>> = (0..16).map(|i| vec![if i < 8 { 0.0 } else { 1.0 }]).collect();
/// let opts = CoxOpts { ties: Ties::Breslow, ..CoxOpts::default() };
/// let fit = cox_ph(&obs, &x, &opts)?;
/// // statsmodels PHReg(..., ties='breslow').fit(): params [-1.121832078408797], bse [0.7527302842899486]
/// assert!((fit.coefficients[0] - (-1.121_832_078_408_797)).abs() < 1e-8);
/// assert!((fit.hazard_ratios()[0] - 0.325_682_571_702_741_45).abs() < 1e-8);
/// // With no ties the score test is the log-rank statistic: survdiff → 2.425426790810062.
/// let ctx = Context::new();
/// assert!((fit.score_test(&ctx)?.statistic_f64()? - 2.425_426_790_810_062).abs() < 1e-9);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// - [`SymplexError::InvalidArgument`] for no observations, mismatched or
///   ragged `x`, no covariates, a non-finite covariate, a constant
///   covariate column, no events, or `max_iter = 0` / `tol ≤ 0`.
/// - [`SymplexError::ComputationFailed`] when the information matrix is
///   singular (collinear covariates), when the likelihood is monotone — a
///   coefficient passes `±25` while the Newton step is still large, as
///   when every subject with the larger covariate value fails before every
///   subject with the smaller one; `coxph` warns "beta may be infinite" —
///   naming the covariate, or when `max_iter` steps do not converge.
pub fn cox_ph(
    obs: &[Observation],
    x: &[Vec<f64>],
    opts: &CoxOpts,
) -> Result<CoxModel, SymplexError> {
    let strata = vec![0; obs.len()];
    fit("cox_ph", obs, x, &strata, opts)
}

/// A stratified fit: subjects are compared only within their stratum, so
/// each stratum has its own baseline hazard and the log partial
/// likelihood is the sum of the strata's.  `strata[i]` labels `obs[i]`;
/// a stratum without events contributes nothing.
/// `PHReg(time, x, status=event, strata=strata, ties=…).fit()`.
///
/// # Errors
///
/// As [`cox_ph`], plus [`SymplexError::InvalidArgument`] when `strata`
/// does not have one label per observation.
pub fn cox_ph_stratified(
    obs: &[Observation],
    x: &[Vec<f64>],
    strata: &[usize],
    opts: &CoxOpts,
) -> Result<CoxModel, SymplexError> {
    fit("cox_ph_stratified", obs, x, strata, opts)
}

fn fit(
    op: &'static str,
    obs: &[Observation],
    x: &[Vec<f64>],
    strata: &[usize],
    opts: &CoxOpts,
) -> Result<CoxModel, SymplexError> {
    let p = validate(op, obs, x, strata, opts)?;
    let table = build_table(obs, strata);
    let ties = opts.ties;

    let mut beta = vec![0.0; p];
    let mut current = evaluate(&table, x, &beta, ties);
    let null_log_likelihood = current.ll;
    let singular_at_start = || {
        failed(
            op,
            "the information matrix at β = 0 is singular: a covariate is collinear with the others or constant within every risk set",
        )
    };
    let l0 = information_cholesky(&current.info).ok_or_else(singular_at_start)?;
    let score_statistic = dense_f64::quadratic_form(&l0, p, &current.score);

    let mut iterations = 0;
    let mut converged = false;
    for iter in 1..=opts.max_iter {
        iterations = iter;
        let l = if iter == 1 {
            l0.clone()
        } else {
            information_cholesky(&current.info).ok_or_else(|| {
                failed(
                    op,
                    "the information matrix became singular: the partial likelihood has no finite maximiser",
                )
            })?
        };
        let direction = dense_f64::cholesky_solve(&l, p, &current.score);
        // Step-halving: shrink the Newton step while ℓ would decrease.
        let mut lambda = 1.0;
        let mut halvings = 0;
        let (next_beta, next) = loop {
            let candidate: Vec<f64> = beta
                .iter()
                .zip(&direction)
                .map(|(b, d)| b + lambda * d)
                .collect();
            let eval = evaluate(&table, x, &candidate, ties);
            let acceptable =
                eval.ll.is_finite() && eval.ll >= current.ll - 1e-12 * current.ll.abs();
            if acceptable || halvings >= MAX_HALVINGS {
                break (candidate, eval);
            }
            lambda *= 0.5;
            halvings += 1;
        };
        let max_step = direction
            .iter()
            .fold(0.0_f64, |m, d| m.max((lambda * d).abs()));
        let delta_ll = (next.ll - current.ll).abs();
        beta = next_beta;
        current = next;
        if beta.iter().any(|b| !b.is_finite()) || !current.ll.is_finite() {
            return Err(failed(
                op,
                "the coefficients diverged: the partial likelihood has no finite maximiser",
            ));
        }
        let scale = beta.iter().fold(1.0_f64, |m, b| m.max(b.abs()));
        let step_small = max_step <= opts.tol * scale;
        let runaway = (0..p)
            .filter(|&j| beta[j].abs() > DIVERGENCE_BOUND)
            .max_by(|&a, &b| beta[a].abs().total_cmp(&beta[b].abs()));
        if let (false, Some(j)) = (step_small, runaway) {
            return Err(failed(
                op,
                format!(
                    "monotone likelihood: the coefficient of covariate {j} passed {} (β = {:.3}) with the Newton step still large, so the partial likelihood has no finite maximiser (every subject with a larger value of covariate {j} fails before every subject with a smaller one, or the reverse; if the covariate is merely on a tiny scale, rescale it)",
                    DIVERGENCE_BOUND, beta[j]
                ),
            ));
        }
        if step_small && delta_ll <= opts.tol * current.ll.abs().max(1.0) {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(failed(
            op,
            format!(
                "no convergence in {} Newton steps (last |Δβ| criterion not met); raise max_iter or rescale the covariates",
                opts.max_iter
            ),
        ));
    }

    let wald = wald_summary(&current.info, &beta).ok_or_else(|| {
        failed(
            op,
            "the information matrix at the estimate is singular: the standard errors are undefined",
        )
    })?;
    let wald_statistic = dot(
        &beta,
        &dense_f64::matvec(&dense_f64::flatten(&current.info), p, p, &beta),
    );

    Ok(CoxModel {
        coefficients: beta,
        standard_errors: wald.se,
        z_values: wald.z,
        p_values: wald.p,
        log_likelihood: current.ll,
        null_log_likelihood,
        cov_params: wald.cov,
        nobs: obs.len(),
        n_events: obs.iter().filter(|o| o.event).count(),
        ties,
        iterations,
        converged,
        obs: obs.to_vec(),
        x: x.to_vec(),
        strata: strata.to_vec(),
        table,
        score_statistic,
        wald_statistic,
    })
}

impl CoxModel {
    /// Number of coefficients `p`.
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.coefficients.len()
    }

    /// The stratum label of each observation (`0` throughout for [`cox_ph`]).
    #[must_use]
    pub fn strata(&self) -> &[usize] {
        &self.strata
    }

    /// `exp(β̂_j)`: the multiplicative change in the hazard per unit of
    /// covariate `j`.
    #[must_use]
    pub fn hazard_ratios(&self) -> Vec<f64> {
        self.coefficients.iter().map(|b| b.exp()).collect()
    }

    /// Wald intervals for the coefficients, `β̂_j ± z_{(1+c)/2} · se_j`
    /// (`conf_int(alpha = 1 − c)`).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for `confidence ∉ (0, 1)`.
    pub fn conf_int(&self, confidence: f64) -> Result<Vec<Interval<f64>>, SymplexError> {
        check_confidence("conf_int", confidence)?;
        let z = z_two_sided(confidence);
        Ok(self
            .coefficients
            .iter()
            .zip(&self.standard_errors)
            .map(|(b, se)| Interval::closed(b - z * se, b + z * se))
            .collect())
    }

    /// Wald intervals for the hazard ratios: [`conf_int`](Self::conf_int)
    /// exponentiated end to end (`np.exp(conf_int())`).
    ///
    /// # Errors
    ///
    /// As [`conf_int`](Self::conf_int).
    pub fn hazard_ratio_conf_int(
        &self,
        confidence: f64,
    ) -> Result<Vec<Interval<f64>>, SymplexError> {
        Ok(self
            .conf_int(confidence)?
            .into_iter()
            .map(|iv| Interval::closed(iv.lower.exp(), iv.upper.exp()))
            .collect())
    }

    /// The likelihood-ratio statistic `2(ℓ(β̂) − ℓ(0))`, asymptotically
    /// `χ²_p`.
    #[must_use]
    pub fn llr(&self) -> f64 {
        2.0 * (self.log_likelihood - self.null_log_likelihood)
    }

    /// The Wald statistic `β̂ᵀ I(β̂) β̂` (`= β̂ᵀ cov_params⁻¹ β̂`),
    /// asymptotically `χ²_p`.
    #[must_use]
    pub fn wald_statistic(&self) -> f64 {
        self.wald_statistic
    }

    /// The score (log-rank type) statistic `U(0)ᵀ I(0)⁻¹ U(0)`,
    /// asymptotically `χ²_p`.  For a single `0/1` covariate under
    /// [`Ties::Breslow`] with no tied event times this *is* the log-rank
    /// statistic of [`log_rank_test`](super::survival::log_rank_test).
    #[must_use]
    pub fn score_statistic(&self) -> f64 {
        self.score_statistic
    }

    fn chi_squared_test(&self, ctx: &Context, statistic: f64) -> Result<TestResult, SymplexError> {
        let df = self.n_params();
        let statistic = ctx.from_f64(statistic)?;
        let p_value = if statistic.is_positive() == Some(true) {
            chi_squared_sf(ctx, df, &statistic)
        } else {
            ctx.one()
        };
        Ok(TestResult {
            statistic,
            p_value,
            df: Some(ex_usize(ctx, df)),
            alternative: Alternative::TwoSided,
        })
    }

    /// The likelihood-ratio test of `β = 0`: [`llr`](Self::llr) referred to
    /// `χ²_p` (`summary(coxph)`'s "Likelihood ratio test").
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the statistic is not a number.
    pub fn llr_test(&self, ctx: &Context) -> Result<TestResult, SymplexError> {
        self.chi_squared_test(ctx, self.llr())
    }

    /// The Wald test of `β = 0`: [`wald_statistic`](Self::wald_statistic)
    /// referred to `χ²_p` (`summary(coxph)`'s "Wald test").
    ///
    /// # Errors
    ///
    /// As [`llr_test`](Self::llr_test).
    pub fn wald_test(&self, ctx: &Context) -> Result<TestResult, SymplexError> {
        self.chi_squared_test(ctx, self.wald_statistic())
    }

    /// The score test of `β = 0`: [`score_statistic`](Self::score_statistic)
    /// referred to `χ²_p` (`summary(coxph)`'s "Score (logrank) test").
    ///
    /// # Errors
    ///
    /// As [`llr_test`](Self::llr_test).
    pub fn score_test(&self, ctx: &Context) -> Result<TestResult, SymplexError> {
        self.chi_squared_test(ctx, self.score_statistic)
    }

    /// `AIC = −2ℓ(β̂) + 2p`.
    #[must_use]
    pub fn aic(&self) -> f64 {
        -2.0 * self.log_likelihood + 2.0 * self.n_params() as f64
    }

    fn check_row(&self, op: &'static str, x_row: &[f64]) -> Result<(), SymplexError> {
        let p = self.n_params();
        if x_row.len() != p {
            return Err(invalid(
                op,
                format!("x_row has {} entries, expected {p}", x_row.len()),
            ));
        }
        if let Some(j) = x_row.iter().position(|v| !v.is_finite()) {
            return Err(invalid(op, format!("x_row[{j}] is not finite")));
        }
        Ok(())
    }

    /// The linear predictor `x₀ᵀβ̂` (the log partial hazard).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for a wrong length or a non-finite
    /// entry.
    pub fn predict_log_partial_hazard(&self, x_row: &[f64]) -> Result<f64, SymplexError> {
        self.check_row("predict_log_partial_hazard", x_row)?;
        Ok(dot(x_row, &self.coefficients))
    }

    /// The partial hazard `exp(x₀ᵀβ̂)`: the hazard relative to a subject
    /// with `x = 0` (`predict(x, pred_type='hr')`).
    ///
    /// # Errors
    ///
    /// As [`predict_log_partial_hazard`](Self::predict_log_partial_hazard).
    pub fn predict_partial_hazard(&self, x_row: &[f64]) -> Result<f64, SymplexError> {
        self.check_row("predict_partial_hazard", x_row)?;
        Ok(dot(x_row, &self.coefficients).exp())
    }

    /// `ηᵢ = xᵢᵀβ̂` for every fitted observation, in observation order.
    #[must_use]
    pub fn linear_predictors(&self) -> Vec<f64> {
        linear_predictor(&self.x, &self.coefficients).eta
    }

    /// Harrell's concordance index, exactly.  A pair `(i, j)` is *usable*
    /// when `tᵢ < tⱼ`, subject `i` had the event, and both are in the same
    /// stratum — `i` is then known to have failed first.  It is
    /// *concordant* when the model agrees, `ηᵢ > ηⱼ`; a tie `ηᵢ = ηⱼ`
    /// counts one half.  `C = (concordant + ½ tied) / usable`; `1/2` is
    /// chance, `1` perfect ranking.  Pairs with equal times are not usable
    /// (this is the definition of Harrell et al. 1982 without the
    /// censored-at-equal-time refinement of `survival::concordance`).
    ///
    /// # Errors
    ///
    /// [`SymplexError::ComputationFailed`] when no pair is usable (every
    /// event is at the last observed time of its stratum).
    pub fn concordance(&self) -> Result<Q, SymplexError> {
        let eta = self.linear_predictors();
        let n = self.obs.len();
        let mut usable = 0usize;
        let mut twice_concordant = 0usize;
        for i in 0..n {
            if !self.obs[i].event {
                continue;
            }
            for j in 0..n {
                if self.strata[i] != self.strata[j] || self.obs[i].time >= self.obs[j].time {
                    continue;
                }
                usable += 1;
                if eta[i] > eta[j] {
                    twice_concordant += 2;
                } else if eta[i] == eta[j] {
                    twice_concordant += 1;
                }
            }
        }
        if usable == 0 {
            return Err(failed(
                "concordance",
                "no usable pairs: every event occurs at the last observed time of its stratum",
            ));
        }
        Ok(qu(twice_concordant) / qu(2 * usable))
    }

    /// The Breslow estimate of the baseline hazard, one row per distinct
    /// event time per stratum, ascending in time within each stratum:
    /// `ĥ₀(t) = d_t / Σ_{j∈R_t} exp(xⱼᵀβ̂)` and its running sum.  This is
    /// the estimator of `PHReg.baseline_cumulative_hazard` (whose
    /// cumulative column is, however, the sum *before* each time) and of
    /// `basehaz(coxph(ties = "breslow"), centered = FALSE)`; under Efron
    /// ties `basehaz` uses an Efron-adjusted increment instead, which
    /// differs at tied event times.
    #[must_use]
    pub fn baseline_hazard(&self) -> Vec<BaselineHazardRow> {
        let p = self.n_params();
        let lp = linear_predictor(&self.x, &self.coefficients);
        // `w` is exp(η − shift); undo the shift so the hazard is absolute.
        let unshift = (-lp.shift).exp();
        let sums = sweep(&self.table, &self.x, &lp.w, p);
        // The table runs descending in time within each stratum; emit
        // ascending with the cumulative sum.
        let mut rows: Vec<BaselineHazardRow> = Vec::with_capacity(self.table.len());
        let mut start = 0;
        while start < self.table.len() {
            let stratum = self.table[start].stratum;
            let end = (start..self.table.len())
                .find(|&k| self.table[k].stratum != stratum)
                .unwrap_or(self.table.len());
            let mut cumulative = 0.0;
            for k in (start..end).rev() {
                let hazard = self.table[k].events.len() as f64 * unshift / sums[k].s0;
                cumulative += hazard;
                rows.push(BaselineHazardRow {
                    stratum,
                    time: self.table[k].time.clone(),
                    hazard,
                    cumulative,
                });
            }
            start = end;
        }
        rows
    }

    /// Schoenfeld residuals, one row per event in observation order (the
    /// rows of `obs` with `event = true`, in order), `p` columns:
    /// `xᵢ − x̄(tᵢ)` where `x̄(t) = Σ_{j∈R_t} wⱼ xⱼ / Σ_{j∈R_t} wⱼ` is the
    /// hazard-weighted covariate mean over the risk set at the event time,
    /// `wⱼ = exp(xⱼᵀβ̂)`.  They sum to zero over the events (the score
    /// equation) and, plotted against time, should show no trend if the
    /// hazards are proportional.  `PHReg.fit().schoenfeld_residuals` (its
    /// censored rows, `NaN` there, are omitted here).
    #[must_use]
    pub fn schoenfeld_residuals(&self) -> Vec<Vec<f64>> {
        let p = self.n_params();
        let lp = linear_predictor(&self.x, &self.coefficients);
        let sums = sweep(&self.table, &self.x, &lp.w, p);
        // Event index → table entry.
        let mut entry_of = vec![usize::MAX; self.obs.len()];
        for (k, entry) in self.table.iter().enumerate() {
            for &i in &entry.events {
                entry_of[i] = k;
            }
        }
        (0..self.obs.len())
            .filter(|&i| self.obs[i].event)
            .map(|i| {
                let s = &sums[entry_of[i]];
                (0..p).map(|a| self.x[i][a] - s.s1[a] / s.s0).collect()
            })
            .collect()
    }

    /// Martingale residuals `Mᵢ = δᵢ − exp(xᵢᵀβ̂) · Λ̂₀(tᵢ)`, one per
    /// observation, with `Λ̂₀` the Breslow cumulative baseline hazard of
    /// [`baseline_hazard`](Self::baseline_hazard) *including* the
    /// increment at `tᵢ`.  They sum to zero exactly (up to rounding).
    /// Note that `PHReg.fit().martingale_residuals` evaluates the
    /// cumulative hazard *before* `tᵢ` and so differs; `residuals(coxph,
    /// type = "martingale")` agrees under Breslow ties.
    #[must_use]
    pub fn martingale_residuals(&self) -> Vec<f64> {
        let p = self.n_params();
        let lp = linear_predictor(&self.x, &self.coefficients);
        let w = &lp.w;
        let sums = sweep(&self.table, &self.x, w, p);
        // exp(ηᵢ) · d_k / S0_k = wᵢ · d_k / S0_k(shifted): the shift cancels.
        let mut resid: Vec<f64> = self
            .obs
            .iter()
            .map(|o| if o.event { 1.0 } else { 0.0 })
            .collect();
        for (k, entry) in self.table.iter().enumerate() {
            let increment = entry.events.len() as f64 / sums[k].s0;
            for i in 0..self.obs.len() {
                if self.strata[i] == entry.stratum && self.obs[i].time >= entry.time {
                    resid[i] -= w[i] * increment;
                }
            }
        }
        resid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn table_risk_sets_include_censored_at_event_time() {
        let obs = Observation::from_i64(&[2, 2, 3, 5], &[true, false, true, false]);
        let table = build_table(&obs, &[0; 4]);
        // Descending: t = 3 (enter: 3 and 5), then t = 2 (enter: both 2s).
        assert_eq!(table.len(), 2);
        assert_eq!(table[0].time, qu(3));
        assert_eq!(table[0].enter, vec![2, 3]);
        assert_eq!(table[0].events, vec![2]);
        assert_eq!(table[1].time, qu(2));
        assert_eq!(table[1].enter, vec![0, 1]);
        assert_eq!(table[1].events, vec![0]);
    }

    #[test]
    fn efron_and_breslow_agree_without_ties() {
        let (obs, x) = d1();
        let table = build_table(&obs, &[0; 10]);
        let b = [0.3];
        let e = evaluate(&table, &x, &b, Ties::Efron);
        let br = evaluate(&table, &x, &b, Ties::Breslow);
        assert!((e.ll - br.ll).abs() < 1e-12);
        assert!((e.score[0] - br.score[0]).abs() < 1e-12);
        assert!((e.info[0][0] - br.info[0][0]).abs() < 1e-12);
    }

    #[test]
    fn score_is_the_derivative_of_the_log_likelihood() {
        let (obs, x) = d1();
        let table = build_table(&obs, &[0; 10]);
        let h = 1e-6;
        for ties in [Ties::Efron, Ties::Breslow] {
            let at = |b: f64| evaluate(&table, &x, &[b], ties);
            let e = at(0.4);
            let numeric = (at(0.4 + h).ll - at(0.4 - h).ll) / (2.0 * h);
            assert!((e.score[0] - numeric).abs() < 1e-6, "{ties:?}");
            let numeric_info = -(at(0.4 + h).score[0] - at(0.4 - h).score[0]) / (2.0 * h);
            assert!((e.info[0][0] - numeric_info).abs() < 1e-5, "{ties:?}");
        }
    }
}
