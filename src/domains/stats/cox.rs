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
//! in the tests.  [`CoxModel`] implements the shared
//! [`LikelihoodFit`] and [`WaldFit`] summaries of the regression module.
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

use super::common::{information_cholesky, invalid, qu, wald_summary};
use super::data::Q;
use super::hypothesis::{Alternative, TestResult};
use super::regression::{LikelihoodFit, WaldFit, chi_squared_test_result};
use super::survival::Observation;
use crate::api::context::Context;
use crate::base::dense_f64::{self, dot};
use crate::base::errors::SymplexError;
use crate::base::interval::Interval;

/// A Newton step that still moves the log hazard ratios of two subjects
/// by at least this much ([`Centred::spread`]) while the quadratic model
/// predicts a negligible gain in `ℓ` is a step along a flat direction.
/// Along a monotone likelihood `ℓ ≈ sup ℓ − A·e^{−g}` in the separating
/// gap `g`, so every Newton step moves `g` by `U/I → 1` and the spread by
/// at least that; at a finite maximum the step shrinks quadratically.
const FLAT_STEP: f64 = 0.5;

/// "Negligible gain" for [`FLAT_STEP`]: `½ UᵀI⁻¹U ≤ FLAT_GAIN · max(1, |ℓ|)`.
/// Fixed rather than the caller's `tol`, so that a loose tolerance cannot
/// turn two large early steps of a finite fit into a diagnosis.
const FLAT_GAIN: f64 = 1e-9;

/// Consecutive flat steps that diagnose a monotone likelihood.
const FLAT_RUN: usize = 2;

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
/// `S2 = Σ_R wⱼ xⱼ xⱼᵀ`, and the same over the failures `D` at that time,
/// with `wⱼ = exp(ηⱼ − shift)` and `shift = max_{j∈R} ηⱼ`: every sum is
/// scaled by the same `e^{−shift}`, so ratios of sums are exact and the
/// true `Σ_R e^{ηⱼ}` is `e^{shift} · S0`.
struct RiskSums {
    shift: f64,
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
///
/// One sort by (stratum, descending time), then one pass: the risk set at
/// `t` is everyone with `tᵢ ≥ t`, so a subject enters at the largest event
/// time `≤ tᵢ`; subjects below the first event time of their stratum never
/// enter.  (Filtering the sample once per event time was `O(n·T)`: seconds
/// for a few thousand subjects.)
fn build_table(obs: &[Observation], strata: &[usize]) -> Vec<EventTime> {
    let mut order: Vec<usize> = (0..obs.len()).collect();
    order.sort_by(|&a, &b| {
        strata[a]
            .cmp(&strata[b])
            .then_with(|| obs[b].time.cmp(&obs[a].time))
    });
    let mut table = Vec::new();
    let mut pending: Vec<usize> = Vec::new();
    let mut start = 0;
    while start < order.len() {
        let first = order[start];
        let end = order[start..]
            .iter()
            .position(|&i| strata[i] != strata[first] || obs[i].time != obs[first].time)
            .map_or(order.len(), |len| start + len);
        if start == 0 || strata[order[start - 1]] != strata[first] {
            pending.clear();
        }
        let group = &order[start..end];
        pending.extend_from_slice(group);
        if group.iter().any(|&i| obs[i].event) {
            // Index order, as a filter over the sample would give (the
            // risk-set sums are accumulated in this order).
            let mut enter = std::mem::take(&mut pending);
            enter.sort_unstable();
            let mut events: Vec<usize> = group.iter().copied().filter(|&i| obs[i].event).collect();
            events.sort_unstable();
            table.push(EventTime {
                stratum: strata[first],
                time: obs[first].time.clone(),
                enter,
                events,
            });
        }
        start = end;
    }
    table
}

/// Sweep the risk sets given the linear predictor `η`, one [`RiskSums`]
/// per table entry, in table order.
///
/// The weights are shifted by the maximum of `η` over *each* risk set, a
/// running log-sum-exp: the risk set only grows along the (descending)
/// sweep, so when an entrant raises the maximum the running sums are
/// rescaled by `e^{old − new}`.  A single global shift would underflow
/// every later risk set whose members all have `η` more than ~745 below
/// the global maximum (`S0 = 0`, `ℓ = −∞`), as happens with a subject
/// censored before the first event that carries an extreme covariate.
fn sweep(table: &[EventTime], x: &[Vec<f64>], eta: &[f64], p: usize) -> Vec<RiskSums> {
    let mut out = Vec::with_capacity(table.len());
    let mut shift = f64::NEG_INFINITY;
    let mut s0 = 0.0;
    let mut s1 = vec![0.0; p];
    let mut s2 = vec![vec![0.0; p]; p];
    let mut stratum = None;
    for entry in table {
        if stratum != Some(entry.stratum) {
            stratum = Some(entry.stratum);
            shift = f64::NEG_INFINITY;
            s0 = 0.0;
            s1.iter_mut().for_each(|v| *v = 0.0);
            s2.iter_mut().flatten().for_each(|v| *v = 0.0);
        }
        let top = entry
            .enter
            .iter()
            .map(|&i| eta[i])
            .fold(f64::NEG_INFINITY, f64::max);
        if top > shift {
            // e^{−shift_old} → e^{−top}: `exp(−∞) = 0` on the first entrant
            // of a stratum, when the sums are still empty.
            let r = (shift - top).exp();
            s0 *= r;
            s1.iter_mut().for_each(|v| *v *= r);
            s2.iter_mut().flatten().for_each(|v| *v *= r);
            shift = top;
        }
        for &i in &entry.enter {
            let wi = (eta[i] - shift).exp();
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
            let wi = (eta[i] - shift).exp();
            d0 += wi;
            for a in 0..p {
                d1[a] += wi * x[i][a];
                for b in 0..p {
                    d2[a][b] += wi * x[i][a] * x[i][b];
                }
            }
        }
        out.push(RiskSums {
            shift,
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

/// Counts by rank with `O(log n)` insertion and prefix sums.
struct Fenwick {
    tree: Vec<usize>,
}

impl Fenwick {
    fn new(n: usize) -> Self {
        Self {
            tree: vec![0; n + 1],
        }
    }

    /// Count one more at rank `r` (`0`-based).
    fn add(&mut self, r: usize) {
        let mut k = r + 1;
        while k < self.tree.len() {
            self.tree[k] += 1;
            k += k & k.wrapping_neg();
        }
    }

    /// How many were counted at ranks `< r`.
    fn prefix(&self, r: usize) -> usize {
        let mut k = r.min(self.tree.len() - 1);
        let mut sum = 0;
        while k > 0 {
            sum += self.tree[k];
            k &= k - 1;
        }
        sum
    }

    fn total(&self) -> usize {
        self.prefix(self.tree.len() - 1)
    }
}

/// `ln(e^a + e^b)` without overflow; `−∞` is the empty sum.
fn log_add_exp(a: f64, b: f64) -> f64 {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    if hi == f64::NEG_INFINITY {
        return hi;
    }
    hi + (lo - hi).exp().ln_1p()
}

/// `ηᵢ = xᵢᵀβ` for every row of `x`.
fn linear_predictor(x: &[Vec<f64>], beta: &[f64]) -> Vec<f64> {
    x.iter().map(|row| dot(row, beta)).collect()
}

/// The covariates centred at their mean over the *informative* subjects —
/// those in at least one risk set — and the list of those subjects.
///
/// Replacing `xᵢ` by `xᵢ − c` adds the same `−cᵀβ` to every `ηᵢ`, which
/// cancels between numerator and denominator of every factor of the
/// partial likelihood: `ℓ`, the score and the information are unchanged
/// *exactly*.  Numerically it is essential: the information is a sum of
/// risk-set covariances `S2/S0 − (S1/S0)(S1/S0)ᵀ`, which for a covariate
/// with a large offset (a calendar year, `10⁶ + small`) cancels to noise —
/// uncentred, `x + 10⁶` gave a standard error wrong in the fifth digit and
/// `x + 10⁸` a spurious singular information matrix.
struct Centred {
    x: Vec<Vec<f64>>,
    informative: Vec<usize>,
    /// `max − min` of each covariate over the informative subjects.
    range: Vec<f64>,
}

fn centre_covariates(x: &[Vec<f64>], table: &[EventTime], p: usize) -> Centred {
    let mut informative: Vec<usize> = table.iter().flat_map(|e| e.enter.iter().copied()).collect();
    informative.sort_unstable();
    informative.dedup();
    let m = informative.len().max(1) as f64;
    let centre: Vec<f64> = (0..p)
        .map(|j| informative.iter().map(|&i| x[i][j]).sum::<f64>() / m)
        .collect();
    let range = (0..p)
        .map(|j| {
            let (lo, hi) = informative
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &i| {
                    (lo.min(x[i][j]), hi.max(x[i][j]))
                });
            (hi - lo).max(0.0)
        })
        .collect();
    let x = x
        .iter()
        .map(|row| row.iter().zip(&centre).map(|(v, c)| v - c).collect())
        .collect();
    Centred {
        x,
        informative,
        range,
    }
}

impl Centred {
    /// `maxᵢ xᵢᵀv − minᵢ xᵢᵀv` over the informative subjects: how far apart
    /// the coefficient change `v` moves the log hazard ratios of any two
    /// subjects that share a risk set.  Invariant to rescaling or shifting
    /// a covariate, unlike `|v|` itself.
    fn spread(&self, v: &[f64]) -> f64 {
        let (lo, hi) =
            self.informative
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &i| {
                    let e = dot(&self.x[i], v);
                    (lo.min(e), hi.max(e))
                });
        (hi - lo).max(0.0)
    }

    /// The covariate contributing most to [`spread`](Self::spread)`(v)`.
    fn dominant(&self, v: &[f64]) -> usize {
        (0..v.len())
            .max_by(|&a, &b| (v[a].abs() * self.range[a]).total_cmp(&(v[b].abs() * self.range[b])))
            .unwrap_or(0)
    }
}

fn evaluate(table: &[EventTime], x: &[Vec<f64>], beta: &[f64], ties: Ties) -> Evaluation {
    let p = beta.len();
    let eta = linear_predictor(x, beta);
    let sums = sweep(table, x, &eta, p);
    let mut ll = 0.0;
    let mut score = vec![0.0; p];
    let mut info = vec![vec![0.0; p]; p];
    for (entry, s) in table.iter().zip(&sums) {
        let d = entry.events.len();
        for &i in &entry.events {
            // ηᵢ − shift: the d shifts cancel against the d ln S0 terms
            // below, whose sums are all scaled by e^{−shift}.
            ll += eta[i] - s.shift;
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
/// likelihood from `β = 0`: `β ← β + λ·d` with the Newton step
/// `d = I(β)⁻¹ U(β)` (analytic score `U` and information `I`) and `λ`
/// halved while `ℓ` would decrease.  `x[i]` is the covariate row of
/// `obs[i]`; there is no intercept.  Only the order of the times matters
/// (negative times are accepted).  `PHReg(time, x, status=event,
/// ties=…).fit()`.
///
/// **Convergence** is judged on the full Newton step `d` at the current
/// `β`, in units that do not depend on how the covariates are scaled or
/// shifted (as long as their squared spreads stay inside the `f64` range,
/// about `10±¹⁵⁴`): it is reached when `d` moves the log hazard ratios of the
/// subjects by at most `tol · max(1, spread of xᵢᵀβ)` — the spread being
/// `maxᵢ xᵢᵀv − minᵢ xᵢᵀv` over the subjects in some risk set — and the
/// quadratic model predicts a gain `½ UᵀI⁻¹U ≤ tol · max(1, |ℓ|)`; the
/// step is then taken and the fit returned.  Internally the covariates are
/// centred and the risk-set sums shifted per risk set (both leave the
/// partial likelihood unchanged exactly) so that large offsets and large
/// `|xᵀβ|` cost no accuracy.
///
/// **Monotone likelihood** (`β̂` infinite: the subjects are separated —
/// every subject with a larger value of some linear combination of the
/// covariates fails before every subject with a smaller one) is diagnosed
/// by its signature rather than by a size of `β`: on two successive
/// iterations the predicted gain falls below `10⁻⁹ · max(1, |ℓ|)` while
/// the Newton step still moves the log hazard ratios by `≥ ½` (at a
/// finite maximum the step shrinks quadratically; along a flat direction
/// it stays near `1`).  It is an error, naming the covariate that
/// dominates the diverging step, never a "converged" fit with a huge `β`
/// and standard error (`PHReg` returns one silently).  A quasi-separation
/// whose finite maximum lies further out than that gain can resolve is
/// reported the same way.
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
/// - [`SymplexError::ComputationFailed`] when the information matrix at
///   `β = 0` is singular (collinear covariates, or a covariate constant
///   within every risk set) or not representable (a covariate whose
///   squared spread overflows or underflows `f64`), when the likelihood is
///   monotone (see above;
///   the message names the covariate), or when `max_iter` steps do not
///   converge.
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
    let design = centre_covariates(x, &table, p);
    let xc = &design.x;

    let mut beta = vec![0.0; p];
    let mut current = evaluate(&table, xc, &beta, ties);
    let null_log_likelihood = current.ll;
    let singular_at_start = || {
        // The information holds squared spreads: name a covariate whose
        // square leaves the f64 range before blaming collinearity.
        let unrepresentable = design.range.iter().position(|&r| {
            let r2 = r * r;
            r > 0.0 && (r2 == 0.0 || !r2.is_finite())
        });
        match unrepresentable {
            Some(j) => failed(
                op,
                format!(
                    "the information matrix at β = 0 is not representable: the squared spread of covariate {j} ({:e}²) leaves the f64 range; rescale it",
                    design.range[j]
                ),
            ),
            None => failed(
                op,
                "the information matrix at β = 0 is singular: a covariate is collinear with the others or constant within every risk set",
            ),
        }
    };
    let l0 = information_cholesky(&current.info).ok_or_else(singular_at_start)?;
    let score_statistic = dense_f64::quadratic_form(&l0, p, &current.score);

    let monotone = |j: usize, beta_j: f64, detail: String| {
        failed(
            op,
            format!(
                "monotone likelihood: the partial likelihood has no finite maximiser ({detail}); the subjects are separated along a direction dominated by covariate {j} (β = {beta_j:.4e} and running off): every subject with a larger value of that combination of the covariates fails before every subject with a smaller one, or the reverse, so the estimate is infinite"
            ),
        )
    };

    let mut factor = Some(l0);
    let mut last_direction: Option<Vec<f64>> = None;
    let mut flat_run = 0usize;
    let mut last_step = f64::NAN;
    let mut last_gain = f64::NAN;
    let mut iterations = 0;
    let mut converged = false;
    for iter in 1..=opts.max_iter {
        iterations = iter;
        let l = match factor.take() {
            Some(l) => l,
            None => information_cholesky(&current.info).ok_or_else(|| {
                // Exact singularity does not depend on β (the weights are
                // positive), so a matrix that was regular at β = 0 and turns
                // singular has had its weights concentrate on separated
                // subjects as the coefficients ran off.
                let j = last_direction.as_deref().map_or(0, |d| design.dominant(d));
                monotone(
                    j,
                    beta[j],
                    "the information matrix became numerically singular as the coefficients grew"
                        .to_string(),
                )
            })?,
        };
        let direction = dense_f64::cholesky_solve(&l, p, &current.score);
        let step = design.spread(&direction);
        let gain = 0.5 * dot(&current.score, &direction);
        let ll_scale = current.ll.abs().max(1.0);
        let flat = gain <= FLAT_GAIN * ll_scale && step >= FLAT_STEP;
        // A step moving log hazard ratios by ½ or more is never negligible,
        // whatever `tol` says.
        let small = step < FLAT_STEP
            && step <= opts.tol * design.spread(&beta).max(1.0)
            && gain <= opts.tol * ll_scale;
        flat_run = if flat { flat_run + 1 } else { 0 };
        last_step = step;
        last_gain = gain;
        let flat_detail = || {
            format!(
                "each Newton step still moves the log hazard ratios by {step:.3} while the predicted gain in ℓ is {gain:.1e}"
            )
        };
        if flat_run >= FLAT_RUN {
            let j = design.dominant(&direction);
            return Err(monotone(j, beta[j], flat_detail()));
        }
        // Step-halving: shrink the Newton step while ℓ would decrease.
        let mut lambda = 1.0;
        let mut accepted = None;
        for _ in 0..=MAX_HALVINGS {
            let candidate: Vec<f64> = beta
                .iter()
                .zip(&direction)
                .map(|(b, d)| b + lambda * d)
                .collect();
            let eval = evaluate(&table, xc, &candidate, ties);
            if eval.ll.is_finite() && eval.ll >= current.ll - 1e-12 * current.ll.abs() {
                accepted = Some((candidate, eval));
                break;
            }
            lambda *= 0.5;
        }
        match accepted {
            Some((next_beta, next)) => {
                beta = next_beta;
                current = next;
            }
            // No step improves ℓ but the full step is negligible: β is at
            // the maximum to rounding.
            None if small => {}
            None if flat => {
                let j = design.dominant(&direction);
                return Err(monotone(j, beta[j], flat_detail()));
            }
            None => {
                return Err(failed(
                    op,
                    format!(
                        "no step along the Newton direction increases the partial likelihood (step {step:.3e} in log hazard ratio, predicted gain {gain:.3e}): the information matrix is too ill-conditioned for the maximiser to be located"
                    ),
                ));
            }
        }
        if beta.iter().any(|b| !b.is_finite()) {
            return Err(failed(
                op,
                "the coefficients diverged: the partial likelihood has no finite maximiser",
            ));
        }
        last_direction = Some(direction);
        if small {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(failed(
            op,
            format!(
                "no convergence in {} Newton steps (the last step moved the log hazard ratios by {last_step:.3e} with a predicted gain in ℓ of {last_gain:.3e}); raise max_iter",
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

impl LikelihoodFit for CoxModel {
    /// `ℓ(β̂)`, the maximised log *partial* likelihood.
    fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// `ℓ(0)`, under the same tie correction.
    fn null_log_likelihood(&self) -> f64 {
        self.null_log_likelihood
    }

    fn n_params(&self) -> usize {
        self.coefficients.len()
    }

    fn nobs(&self) -> usize {
        self.nobs
    }

    /// `p`: there is no intercept, so every coefficient is a slope.
    fn df_model(&self) -> usize {
        self.coefficients.len()
    }

    /// `BIC = −2ℓ(β̂) + p ln d` with `d` the number of **events**
    /// ([`n_events`](CoxModel::n_events)), not of observations: the
    /// partial likelihood has one factor per event, so the effective sample
    /// size is the event count (Volinsky & Raftery 2000).  This is R's
    /// `BIC(coxph)` (`logLik.coxph` reports `nobs = nevent`); statsmodels'
    /// `PHRegResults` has no `bic`.
    fn bic(&self) -> f64 {
        -2.0 * self.log_likelihood + self.coefficients.len() as f64 * (self.n_events as f64).ln()
    }

    /// [`llr`](LikelihoodFit::llr) referred to `χ²_p`, reported — like
    /// the model's Wald and score tests — with [`Alternative::TwoSided`].
    fn llr_test(&self, ctx: &Context) -> Result<TestResult, SymplexError> {
        self.chi_squared_test(ctx, LikelihoodFit::llr(self))
    }
}

impl WaldFit for CoxModel {
    fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    fn standard_errors(&self) -> &[f64] {
        &self.standard_errors
    }
}

impl CoxModel {
    /// Number of coefficients `p`.
    #[must_use]
    pub fn n_params(&self) -> usize {
        <Self as LikelihoodFit>::n_params(self)
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
    /// (`conf_int(alpha = 1 − c)`); [`WaldFit::conf_int`].
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for `confidence ∉ (0, 1)`.
    pub fn conf_int(&self, confidence: f64) -> Result<Vec<Interval<f64>>, SymplexError> {
        <Self as WaldFit>::conf_int(self, confidence)
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
    /// `χ²_p`; [`LikelihoodFit::llr`].
    #[must_use]
    pub fn llr(&self) -> f64 {
        <Self as LikelihoodFit>::llr(self)
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

    /// A statistic referred to `χ²_p`, as the trio below reports it.
    fn chi_squared_test(&self, ctx: &Context, statistic: f64) -> Result<TestResult, SymplexError> {
        chi_squared_test_result(ctx, statistic, self.n_params(), Alternative::TwoSided)
    }

    /// The likelihood-ratio test of `β = 0`: [`llr`](Self::llr) referred to
    /// `χ²_p` (`summary(coxph)`'s "Likelihood ratio test");
    /// [`LikelihoodFit::llr_test`].
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the statistic is not a number.
    pub fn llr_test(&self, ctx: &Context) -> Result<TestResult, SymplexError> {
        <Self as LikelihoodFit>::llr_test(self, ctx)
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

    /// `AIC = −2ℓ(β̂) + 2p`; [`LikelihoodFit::aic`].
    #[must_use]
    pub fn aic(&self) -> f64 {
        <Self as LikelihoodFit>::aic(self)
    }

    /// `BIC = −2ℓ(β̂) + p ln d`, `d` the number of events
    /// ([`n_events`](Self::n_events)) — R's `BIC(coxph)`; see
    /// [`LikelihoodFit::bic`] for why events rather than observations.
    #[must_use]
    pub fn bic(&self) -> f64 {
        <Self as LikelihoodFit>::bic(self)
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
        linear_predictor(&self.x, &self.coefficients)
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
        // `+ 0.0` maps −0 to +0, so that ranking by `total_cmp` agrees with `==`.
        let eta: Vec<f64> = self.linear_predictors().iter().map(|e| e + 0.0).collect();
        let n = self.obs.len();
        // Rank the linear predictors (equal values share a rank).
        let mut sorted = eta.clone();
        sorted.sort_by(f64::total_cmp);
        sorted.dedup();
        let rank: Vec<usize> = eta
            .iter()
            .map(|e| sorted.partition_point(|v| v.total_cmp(e).is_lt()))
            .collect();
        // Descending in time within each stratum; each block of equal times
        // queries the subjects with strictly larger times (a Fenwick tree of
        // counts by rank), then joins them: O(n log n), not all n² pairs.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            self.strata[a]
                .cmp(&self.strata[b])
                .then_with(|| self.obs[b].time.cmp(&self.obs[a].time))
        });
        let mut tree = Fenwick::new(sorted.len());
        let mut usable = 0usize;
        let mut twice_concordant = 0usize;
        let mut start = 0;
        while start < n {
            let first = order[start];
            if start > 0 && self.strata[order[start - 1]] != self.strata[first] {
                tree = Fenwick::new(sorted.len());
            }
            let end = order[start..]
                .iter()
                .position(|&i| {
                    self.strata[i] != self.strata[first] || self.obs[i].time != self.obs[first].time
                })
                .map_or(n, |len| start + len);
            let later = tree.total();
            for &i in order[start..end].iter().filter(|&&i| self.obs[i].event) {
                // Usable pairs (i, j): every j with tⱼ > tᵢ in the stratum;
                // concordant when ηⱼ < ηᵢ, half when ηⱼ = ηᵢ.
                let below = tree.prefix(rank[i]);
                let equal = tree.prefix(rank[i] + 1) - below;
                usable += later;
                twice_concordant += 2 * below + equal;
            }
            for &i in &order[start..end] {
                tree.add(rank[i]);
            }
            start = end;
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
        // Only S0 is needed (p = 0 skips S1, S2); η uncentred, since the
        // baseline is the hazard at x = 0.
        let eta = linear_predictor(&self.x, &self.coefficients);
        let sums = sweep(&self.table, &self.x, &eta, 0);
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
                // d / Σ_R e^η = d / (e^shift · S0), in logs so that neither
                // e^shift nor its reciprocal overflows on the way.
                let hazard =
                    ((self.table[k].events.len() as f64 / sums[k].s0).ln() - sums[k].shift).exp();
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
    /// `wⱼ = exp(xⱼᵀβ̂)` — under either tie correction, as
    /// `PHReg.fit().schoenfeld_residuals` computes them (its censored rows,
    /// `NaN` there, are omitted here).  Plotted against time they should
    /// show no trend if the hazards are proportional.  Their sum over the
    /// events is the *Breslow* score at `β̂`: zero for a [`Ties::Breslow`]
    /// fit or when no event times are tied, but not for a [`Ties::Efron`]
    /// fit with tied events, whose score equation averages the risk-set
    /// mean over the `d` Efron steps at each tied time.
    #[must_use]
    pub fn schoenfeld_residuals(&self) -> Vec<Vec<f64>> {
        let p = self.n_params();
        let design = centre_covariates(&self.x, &self.table, p);
        let eta = linear_predictor(&design.x, &self.coefficients);
        let sums = sweep(&self.table, &design.x, &eta, p);
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
                // xᵢ − x̄(tᵢ) in centred coordinates: the centre cancels.
                let s = &sums[entry_of[i]];
                (0..p).map(|a| design.x[i][a] - s.s1[a] / s.s0).collect()
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
        let design = centre_covariates(&self.x, &self.table, self.n_params());
        let eta = linear_predictor(&design.x, &self.coefficients);
        let sums = sweep(&self.table, &design.x, &eta, 0);
        // ln Λ̂₀(t_k) (centred), t_k included, per table entry: within each
        // stratum the table descends in time, so accumulate from its end.
        // In logs, since Λ̂₀ itself may overflow when η is far below 0;
        // ηᵢ + ln Λ̂₀(tᵢ) ≤ ln(events) for every subject at risk.
        let mut log_cum = vec![f64::NEG_INFINITY; self.table.len()];
        let mut running = f64::NEG_INFINITY;
        for k in (0..self.table.len()).rev() {
            if k + 1 == self.table.len() || self.table[k + 1].stratum != self.table[k].stratum {
                running = f64::NEG_INFINITY;
            }
            let term = (self.table[k].events.len() as f64 / sums[k].s0).ln() - sums[k].shift;
            running = log_add_exp(running, term);
            log_cum[k] = running;
        }
        let mut resid: Vec<f64> = self
            .obs
            .iter()
            .map(|o| if o.event { 1.0 } else { 0.0 })
            .collect();
        // A subject enters the risk sets at the largest event time ≤ tᵢ and
        // stays in every earlier one; subjects that never enter keep δᵢ.
        for (k, entry) in self.table.iter().enumerate() {
            for &i in &entry.enter {
                resid[i] -= (eta[i] + log_cum[k]).exp();
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
