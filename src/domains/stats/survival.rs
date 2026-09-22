//! Time-to-event analysis with right censoring, exactly: how long until a
//! respondent completes (or abandons) a task, how long a worker stays
//! active, how long an item takes to reach agreement.
//!
//! Each observation is a `(time, event)` pair: `event = true` means the
//! event happened at `time`, `false` that the subject was only observed
//! until `time` (censored).  The Kaplan–Meier estimator of the survival
//! function `S(t) = P(T > t)` is a step function whose steps are exact
//! rationals, as are Greenwood's variance and the Nelson–Aalen cumulative
//! hazard; the log-rank test compares the survival of groups with an exact
//! χ² statistic.  `statsmodels.duration.survfunc.{SurvfuncRight,
//! survdiff}` are the references named in the tests.
//!
//! ```
//! use symplex::linprog::q;
//! use symplex::stats::survival::{KaplanMeier, Observation};
//!
//! let obs = Observation::from_i64(&[3, 5, 6, 7, 8, 10, 12, 12], &[true, false, true, true, false, true, true, false]);
//! let km = KaplanMeier::fit(&obs)?;
//! // statsmodels SurvfuncRight: S(3) = 0.875, S(6) = 0.7291666…, S(7) = 0.5833…
//! assert_eq!(km.survival_at(&q(3, 1)), q(7, 8));
//! assert_eq!(km.survival_at(&q(6, 1)), q(35, 48));
//! assert_eq!(km.median(), Some(q(10, 1)));
//! # Ok::<(), symplex::prelude::SymplexError>(())
//! ```

use num_traits::{One, Zero};

use super::common::{check_confidence, ex_usize, invalid, norm_ppf, q_to_f64, qi, qu};
use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::errors::SymplexError;
use crate::base::interval::Interval;
use crate::domains::stats::data::Q;
use crate::domains::stats::family::Distribution;
use crate::domains::stats::hypothesis::{Alternative, TestResult};

/// One subject: the time observed and whether the event occurred then
/// (`true`) or the observation was censored (`false`).
#[derive(Clone, Debug, PartialEq)]
pub struct Observation {
    /// Time of the event or of censoring (`≥ 0`).
    pub time: Q,
    /// `true` if the event occurred at `time`, `false` if censored.
    pub event: bool,
}

impl Observation {
    /// Observations from integer times and event flags.
    pub fn from_i64(times: &[i64], events: &[bool]) -> Vec<Observation> {
        times
            .iter()
            .zip(events)
            .map(|(&t, &e)| Observation {
                time: qi(t),
                event: e,
            })
            .collect()
    }

    /// Observations from exact times and event flags.
    pub fn from_q(times: &[Q], events: &[bool]) -> Vec<Observation> {
        times
            .iter()
            .zip(events)
            .map(|(t, &e)| Observation {
                time: t.clone(),
                event: e,
            })
            .collect()
    }
}

/// One distinct event time in a life table.
#[derive(Clone, Debug, PartialEq)]
pub struct LifeTableRow {
    /// The time.
    pub time: Q,
    /// Subjects at risk just before `time` (not yet failed or censored).
    pub at_risk: usize,
    /// Events at `time`.
    pub events: usize,
    /// Subjects censored at `time` (leaving the risk set after it).
    pub censored: usize,
    /// `S(time)`: the Kaplan–Meier estimate just after `time`.
    pub survival: Q,
    /// Greenwood's variance of `S(time)`:
    /// `S² · Σ_{tᵢ ≤ t} dᵢ / (nᵢ (nᵢ − dᵢ))`.
    pub variance: Q,
    /// The Nelson–Aalen cumulative hazard `Σ_{tᵢ ≤ t} dᵢ / nᵢ`.
    pub cumulative_hazard: Q,
}

/// The Kaplan–Meier product-limit estimate of a survival function.
#[derive(Clone, Debug, PartialEq)]
pub struct KaplanMeier {
    rows: Vec<LifeTableRow>,
    n: usize,
}

fn validate(op: &'static str, obs: &[Observation]) -> Result<(), SymplexError> {
    if obs.is_empty() {
        return Err(invalid(op, "at least one observation is required"));
    }
    if obs.iter().any(|o| o.time < Q::zero()) {
        return Err(invalid(op, "times must be non-negative"));
    }
    Ok(())
}

/// The distinct times, ascending, with `(events, censored)` counts at each.
fn tally(obs: &[Observation]) -> Vec<(Q, usize, usize)> {
    let mut times: Vec<Q> = obs.iter().map(|o| o.time.clone()).collect();
    times.sort();
    times.dedup();
    times
        .into_iter()
        .map(|t| {
            let events = obs.iter().filter(|o| o.time == t && o.event).count();
            let censored = obs.iter().filter(|o| o.time == t && !o.event).count();
            (t, events, censored)
        })
        .collect()
}

impl KaplanMeier {
    /// Fit the estimator.  `S(t) = Π_{tᵢ ≤ t} (1 − dᵢ/nᵢ)` over the distinct
    /// event times `tᵢ` with `dᵢ` events among `nᵢ` at risk; subjects
    /// censored at a time leave the risk set after the events at that time
    /// (the standard convention, as in `SurvfuncRight`).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for an empty sample or a negative
    /// time.
    pub fn fit(obs: &[Observation]) -> Result<Self, SymplexError> {
        validate("KaplanMeier::fit", obs)?;
        let n = obs.len();
        let mut at_risk = n;
        let mut survival = Q::one();
        let mut greenwood = Q::zero();
        let mut hazard = Q::zero();
        let mut rows = Vec::new();
        for (time, events, censored) in tally(obs) {
            if events > 0 {
                let d = qu(events);
                let nn = qu(at_risk);
                survival *= (&nn - &d) / &nn;
                if at_risk > events {
                    greenwood += &d / (&nn * (&nn - &d));
                }
                hazard += &d / &nn;
                rows.push(LifeTableRow {
                    time,
                    at_risk,
                    events,
                    censored,
                    survival: survival.clone(),
                    variance: &survival * &survival * &greenwood,
                    cumulative_hazard: hazard.clone(),
                });
            }
            at_risk -= events + censored;
        }
        Ok(KaplanMeier { rows, n })
    }

    /// The number of subjects.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The life table: one row per distinct event time.
    pub fn table(&self) -> &[LifeTableRow] {
        &self.rows
    }

    /// The distinct event times.
    pub fn event_times(&self) -> Vec<Q> {
        self.rows.iter().map(|r| r.time.clone()).collect()
    }

    /// `S(t)`: `1` before the first event, then the step value at the last
    /// event time `≤ t`.
    pub fn survival_at(&self, t: &Q) -> Q {
        self.rows
            .iter()
            .rev()
            .find(|r| r.time <= *t)
            .map_or_else(Q::one, |r| r.survival.clone())
    }

    /// Greenwood's variance of `S(t)`.
    pub fn variance_at(&self, t: &Q) -> Q {
        self.rows
            .iter()
            .rev()
            .find(|r| r.time <= *t)
            .map_or_else(Q::zero, |r| r.variance.clone())
    }

    /// The Nelson–Aalen cumulative hazard `Ĥ(t) = Σ_{tᵢ ≤ t} dᵢ/nᵢ`.
    pub fn cumulative_hazard_at(&self, t: &Q) -> Q {
        self.rows
            .iter()
            .rev()
            .find(|r| r.time <= *t)
            .map_or_else(Q::zero, |r| r.cumulative_hazard.clone())
    }

    /// The `p`-quantile of the survival time: the smallest event time at
    /// which `S(t) ≤ 1 − p` (`None` when the curve never falls that far —
    /// the last observation was censored above it).  This is R's
    /// `survfit` convention; `SurvfuncRight.quantile` uses the strict
    /// `S(t) < 1 − p`, so the two differ only when the curve lands exactly
    /// on `1 − p` (four events at `1, 2, 3, 4`: median `2` here, `3` there).
    pub fn quantile(&self, p: &Q) -> Option<Q> {
        let target = Q::one() - p;
        self.rows
            .iter()
            .find(|r| r.survival <= target)
            .map(|r| r.time.clone())
    }

    /// The median survival time (the `½`-quantile).
    pub fn median(&self) -> Option<Q> {
        self.quantile(&Q::new(1.into(), 2.into()))
    }

    /// A pointwise confidence interval for `S(t)`, in `f64`: the plain
    /// (linear) interval `S ± z·√Var`, clamped to `[0, 1]`, or the
    /// log-log interval `S^{exp(±z·√Var / (S ln S))}` (Kalbfleisch–Prentice;
    /// `SurvfuncRight.simultaneous_cb`'s pointwise cousin), which stays
    /// inside `(0, 1)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for a confidence outside `(0, 1)`.
    pub fn confidence_interval(
        &self,
        t: &Q,
        confidence: f64,
        method: CiMethod,
    ) -> Result<Interval<f64>, SymplexError> {
        check_confidence("KaplanMeier::confidence_interval", confidence)?;
        let z = norm_ppf(0.5 + confidence / 2.0);
        let s = q_to_f64(&self.survival_at(t));
        let se = q_to_f64(&self.variance_at(t)).sqrt();
        Ok(match method {
            CiMethod::Linear => Interval::closed((s - z * se).max(0.0), (s + z * se).min(1.0)),
            CiMethod::LogLog => {
                if s <= 0.0 || s >= 1.0 {
                    Interval::closed(s, s)
                } else {
                    let theta = z * se / (s * s.ln());
                    let lo = s.powf(theta.exp());
                    let hi = s.powf((-theta).exp());
                    Interval::closed(lo.min(hi), lo.max(hi))
                }
            }
        })
    }

    /// The restricted mean survival time `∫₀^τ S(t) dt` (the area under the
    /// step function up to `τ`), exactly.
    pub fn restricted_mean(&self, tau: &Q) -> Q {
        let mut area = Q::zero();
        let mut prev_t = Q::zero();
        let mut prev_s = Q::one();
        for r in &self.rows {
            if r.time >= *tau {
                break;
            }
            area += &prev_s * (&r.time - &prev_t);
            prev_t = r.time.clone();
            prev_s = r.survival.clone();
        }
        if *tau > prev_t {
            area += &prev_s * (tau - &prev_t);
        }
        area
    }
}

/// Which pointwise confidence interval [`KaplanMeier::confidence_interval`] builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CiMethod {
    /// `S ± z·se`, clamped to `[0, 1]`.
    Linear,
    /// The complementary log-log transform (Kalbfleisch–Prentice).
    LogLog,
}

/// The log-rank (Mantel–Cox) test that `k` groups share one survival
/// function.  At each distinct event time `t` with `n` at risk in all
/// groups and `d` events, group `g` with `n_g` at risk expects
/// `e_g = d·n_g/n` events; the statistic is `(O − E)ᵀ V⁻¹ (O − E)` over
/// `k − 1` groups with the hypergeometric covariance
/// `V_{gh} = Σ_t d (n − d) / (n − 1) · (n_g/n) (δ_{gh} − n_h/n)`, an exact
/// rational, referred to `χ²(k − 1)`.  `statsmodels.duration.survfunc.survdiff`.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for fewer than two groups, mismatched
/// lengths, an empty group, or no events at all.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::survival::{Observation, log_rank_test};
///
/// let ctx = Context::new();
/// let obs = Observation::from_i64(
///     &[3, 5, 6, 7, 8, 10, 12, 12, 4, 9, 11, 13, 15, 16, 18, 20],
///     &[true, false, true, true, false, true, true, false, true, true, false, true, true, true, false, true],
/// );
/// let groups = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
/// let r = log_rank_test(&ctx, &obs, &groups)?;
/// // statsmodels survdiff: statistic 3.0736117780824452, p 0.07957250154977413
/// assert!((r.statistic_f64()? - 3.073_611_778_082_445_2).abs() < 1e-12);
/// assert!((r.p_value_f64()? - 0.079_572_501_549_774_13).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
pub fn log_rank_test(
    ctx: &Context,
    obs: &[Observation],
    groups: &[usize],
) -> Result<TestResult, SymplexError> {
    const OP: &str = "log_rank_test";
    validate(OP, obs)?;
    if obs.len() != groups.len() {
        return Err(invalid(OP, "one group label per observation is required"));
    }
    let k = groups.iter().copied().max().map_or(0, |m| m + 1);
    if k < 2 {
        return Err(invalid(OP, "the log-rank test needs at least two groups"));
    }
    for g in 0..k {
        if !groups.contains(&g) {
            return Err(invalid(OP, format!("group {g} has no observations")));
        }
    }
    // O − E and V over the first k − 1 groups.
    let m = k - 1;
    let mut diff = vec![Q::zero(); m];
    let mut v = vec![vec![Q::zero(); m]; m];
    let mut any_event = false;
    for (time, events, _) in tally(obs) {
        if events == 0 {
            continue;
        }
        any_event = true;
        let at_risk: Vec<usize> = (0..k)
            .map(|g| {
                obs.iter()
                    .zip(groups)
                    .filter(|(o, gg)| **gg == g && o.time >= time)
                    .count()
            })
            .collect();
        let n: usize = at_risk.iter().sum();
        let d = qu(events);
        let nn = qu(n);
        for g in 0..m {
            let observed = obs
                .iter()
                .zip(groups)
                .filter(|(o, gg)| **gg == g && o.time == time && o.event)
                .count();
            let expected = &d * qu(at_risk[g]) / &nn;
            diff[g] += qu(observed) - expected;
        }
        if n > 1 {
            // Hypergeometric variance of the events in group g at this time:
            // d (n − d)/(n − 1) · (n_g/n)(δ_gh − n_h/n).
            let scale = &d * (&nn - &d) / (&nn - Q::one());
            for g in 0..m {
                for h in 0..m {
                    let pg = qu(at_risk[g]) / &nn;
                    let ph = qu(at_risk[h]) / &nn;
                    let delta = if g == h { Q::one() } else { Q::zero() };
                    v[g][h] += &scale * &pg * (delta - ph);
                }
            }
        }
    }
    if !any_event {
        return Err(invalid(OP, "no events were observed"));
    }
    // (O − E)ᵀ V⁻¹ (O − E) through an exact solve.
    let vm = crate::prelude::QMatrix::new(v).map_err(|e| invalid(OP, e.to_string()))?;
    let rhs = crate::prelude::QMatrix::new(diff.iter().map(|d| vec![d.clone()]).collect())
        .map_err(|e| invalid(OP, e.to_string()))?;
    let x = vm.solve(&rhs).map_err(|_| {
        SymplexError::computation_failed(
            OP,
            "the log-rank covariance matrix is singular (a group has no risk set at every event time)",
        )
    })?;
    let mut stat = Q::zero();
    for (g, d) in diff.iter().enumerate() {
        stat += d * x.get(g, 0);
    }
    let statistic = ctx.from_ratio(stat);
    let chi = Distribution::chi_squared(ex_usize(ctx, m));
    let p_value = (ctx.one() - chi.cdf(&statistic)).simplify();
    Ok(TestResult {
        statistic,
        p_value,
        df: Some(ex_usize(ctx, m)),
        alternative: Alternative::TwoSided,
    })
}

/// The exponential hazard rate estimate `λ̂ = events / total time at risk`
/// (the MLE under right censoring), exactly, with the number of events.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for an empty sample or zero total time.
pub fn exponential_rate(obs: &[Observation]) -> Result<Q, SymplexError> {
    const OP: &str = "exponential_rate";
    validate(OP, obs)?;
    let total: Q = obs.iter().fold(Q::zero(), |acc, o| acc + &o.time);
    if total.is_zero() {
        return Err(invalid(OP, "the total time at risk is zero"));
    }
    let events = obs.iter().filter(|o| o.event).count();
    Ok(qu(events) / total)
}

/// The mean of the uncensored event times (a naive summary; biased under
/// censoring — prefer [`KaplanMeier::restricted_mean`]).
pub fn mean_event_time(obs: &[Observation]) -> Result<Q, SymplexError> {
    const OP: &str = "mean_event_time";
    validate(OP, obs)?;
    let events: Vec<Q> = obs
        .iter()
        .filter(|o| o.event)
        .map(|o| o.time.clone())
        .collect();
    if events.is_empty() {
        return Err(invalid(OP, "no events were observed"));
    }
    let n = events.len();
    Ok(events.into_iter().fold(Q::zero(), |a, t| a + t) / qu(n))
}

/// The survival function of a [`Distribution`] as an expression:
/// `S(t) = 1 − F(t)`, using the family's closed-form CDF on the support
/// when it has one (so `Exponential(λ)` gives `e^{−λt}`), else the
/// whole-line CDF.
pub fn survival_function(dist: &Distribution, t: &Ex) -> Ex {
    let cdf = dist.family().cdf(t).unwrap_or_else(|| dist.cdf(t));
    (dist.context().one() - cdf).simplify()
}

/// The hazard function of a continuous [`Distribution`]:
/// `h(t) = f(t) / S(t)`, as an expression valid on the support.
pub fn hazard_function(dist: &Distribution, t: &Ex) -> Ex {
    (dist.density(t) / survival_function(dist, t)).simplify()
}
