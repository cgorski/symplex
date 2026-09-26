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

use num_traits::{One, Signed, Zero};

use super::common::{
    check_confidence, chi_squared_sf_q, ex_usize, invalid, norm_ppf, q_to_f64, qi, qu,
};
use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::errors::SymplexError;
use crate::base::interval::Interval;
use crate::domains::stats::data::Q;
use crate::domains::stats::family::{Distribution, sign_of};
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
    /// Observations from integer times and event flags.  The two slices
    /// are **zipped to the shorter one**: extra times or flags are dropped
    /// silently.  Use [`try_from_i64`](Self::try_from_i64) to have a
    /// length mismatch reported instead.
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

    /// Observations from exact times and event flags, zipped to the
    /// shorter slice like [`from_i64`](Self::from_i64); see
    /// [`try_from_q`](Self::try_from_q).
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

    /// [`from_i64`](Self::from_i64) that insists on one flag per time.
    ///
    /// ```
    /// use symplex::stats::survival::Observation;
    ///
    /// assert_eq!(Observation::try_from_i64(&[1, 2], &[true, false])?.len(), 2);
    /// assert!(Observation::try_from_i64(&[1, 2, 3], &[true]).is_err());
    /// # Ok::<(), symplex::prelude::SymplexError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] when the slices differ in length.
    pub fn try_from_i64(times: &[i64], events: &[bool]) -> Result<Vec<Observation>, SymplexError> {
        check_paired("Observation::try_from_i64", times.len(), events.len())?;
        Ok(Self::from_i64(times, events))
    }

    /// [`from_q`](Self::from_q) that insists on one flag per time.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] when the slices differ in length.
    pub fn try_from_q(times: &[Q], events: &[bool]) -> Result<Vec<Observation>, SymplexError> {
        check_paired("Observation::try_from_q", times.len(), events.len())?;
        Ok(Self::from_q(times, events))
    }
}

fn check_paired(op: &'static str, times: usize, events: usize) -> Result<(), SymplexError> {
    if times != events {
        return Err(invalid(
            op,
            format!("{times} times but {events} event flags; one flag per time is needed"),
        ));
    }
    Ok(())
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
    /// Subjects censored at *this event time* (leaving the risk set after
    /// it).  Censorings at times without an event have no row here; ask
    /// [`KaplanMeier::censored_at`] or [`KaplanMeier::censoring_times`]
    /// for those.
    pub censored: usize,
    /// `S(time)`: the Kaplan–Meier estimate just after `time`.
    pub survival: Q,
    /// Greenwood's variance of `S(time)`:
    /// `S² · Σ_{tᵢ ≤ t} dᵢ / (nᵢ (nᵢ − dᵢ))`.  At a time where every
    /// subject at risk fails (`d = n`, the last event time) the term is
    /// infinite but `S = 0`, and the variance is taken as `0` (the term is
    /// skipped); `SurvfuncRight.surv_prob_se` reports `NaN` there.
    pub variance: Q,
    /// The Nelson–Aalen cumulative hazard `Σ_{tᵢ ≤ t} dᵢ / nᵢ`.
    pub cumulative_hazard: Q,
}

/// The Kaplan–Meier product-limit estimate of a survival function.
#[derive(Clone, Debug, PartialEq)]
pub struct KaplanMeier {
    rows: Vec<LifeTableRow>,
    n: usize,
    /// Every distinct censoring time, ascending, with its count — including
    /// the times at which no event occurred (which have no life-table row).
    censorings: Vec<Censoring>,
}

/// Censorings at one time.
#[derive(Clone, Debug, PartialEq)]
struct Censoring {
    time: Q,
    count: usize,
}

/// `0 ≤ p ≤ 1`.
fn unit_closed(p: &Q) -> bool {
    !p.is_negative() && *p <= Q::one()
}

/// `ln S` for an exact `0 < S < 1`, accurate to rounding however close `S`
/// is to `1`: `ln(1 − u)` with `u = 1 − S` formed exactly when `S > ½`
/// (rounding `S` to `f64` first would cost `1 − S` about `−log₁₀(1 − S)`
/// of its digits).  A positive Kaplan–Meier `S` is at least `1/n`, far
/// from the `f64` underflow.
fn ln_survival(s: &Q) -> f64 {
    let half = Q::new(1.into(), 2.into());
    if *s > half {
        (-q_to_f64(&(Q::one() - s))).ln_1p()
    } else {
        q_to_f64(s).ln()
    }
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

/// The distinct times, ascending, with `(events, censored)` counts at each:
/// one sort, then one pass (not a scan of the sample per distinct time).
fn tally(obs: &[Observation]) -> Vec<(Q, usize, usize)> {
    let mut order: Vec<usize> = (0..obs.len()).collect();
    order.sort_by(|&a, &b| obs[a].time.cmp(&obs[b].time));
    let mut out: Vec<(Q, usize, usize)> = Vec::new();
    for i in order {
        let o = &obs[i];
        let (e, c) = (usize::from(o.event), usize::from(!o.event));
        match out.last_mut() {
            Some(last) if last.0 == o.time => {
                last.1 += e;
                last.2 += c;
            }
            _ => out.push((o.time.clone(), e, c)),
        }
    }
    out
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
        let mut censorings = Vec::new();
        for (time, events, censored) in tally(obs) {
            if censored > 0 {
                censorings.push(Censoring {
                    time: time.clone(),
                    count: censored,
                });
            }
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
        Ok(KaplanMeier {
            rows,
            n,
            censorings,
        })
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

    /// The distinct censoring times, ascending — every time at which a
    /// subject left under observation without the event, whether or not
    /// an event also occurred then (the life table's `censored` column
    /// only covers the latter).
    ///
    /// ```
    /// use symplex::linprog::qi;
    /// use symplex::stats::survival::{KaplanMeier, Observation};
    ///
    /// // Censored at 5 (no event there), at 8 (no event) and at 12 (an event too).
    /// let obs = Observation::from_i64(&[3, 5, 6, 7, 8, 10, 12, 12], &[true, false, true, true, false, true, true, false]);
    /// let km = KaplanMeier::fit(&obs)?;
    /// assert_eq!(km.censoring_times(), vec![qi(5), qi(8), qi(12)]);
    /// assert_eq!(km.censored_at(&qi(5)), 1);
    /// assert_eq!(km.censored_at(&qi(6)), 0);
    /// assert_eq!(km.table().iter().map(|r| r.censored).sum::<usize>(), 1);   // only the one at 12
    /// # Ok::<(), symplex::prelude::SymplexError>(())
    /// ```
    pub fn censoring_times(&self) -> Vec<Q> {
        self.censorings.iter().map(|c| c.time.clone()).collect()
    }

    /// The number of subjects censored exactly at `t` (`0` when none was).
    pub fn censored_at(&self, t: &Q) -> usize {
        self.censorings
            .iter()
            .find(|c| c.time == *t)
            .map_or(0, |c| c.count)
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
    /// which `S(t) ≤ 1 − p`, i.e. the generalised inverse
    /// `inf{t : F̂(t) ≥ p}` of `F̂ = 1 − Ŝ` (Brookmeyer & Crowley 1982 for
    /// the median).  `None` when the curve never falls that far — the last
    /// observation was censored above it — and for `p` outside `[0, 1]`.
    /// `SurvfuncRight.quantile` uses the strict `S(t) < 1 − p`
    /// ([`quantile_strict`](Self::quantile_strict)), so the two differ only
    /// when the curve lands exactly on `1 − p` (four events at `1, 2, 3, 4`:
    /// median `2` here, `3` there); the sample-median convention would take
    /// the midpoint `5/2` of such a flat stretch.
    pub fn quantile(&self, p: &Q) -> Option<Q> {
        if !unit_closed(p) {
            return None;
        }
        let target = Q::one() - p;
        self.rows
            .iter()
            .find(|r| r.survival <= target)
            .map(|r| r.time.clone())
    }

    /// The `p`-quantile with the strict convention of
    /// `SurvfuncRight.quantile` (and SAS): the smallest event time at which
    /// `S(t) < 1 − p`, so it differs from [`quantile`](Self::quantile)
    /// exactly when the curve lands on `1 − p`; `None` for `p` outside
    /// `[0, 1]`.  The comparison here is exact.  statsmodels compares a
    /// floating-point `S = exp(Σ ln(1 − d/n))`, which can land a rounding
    /// below an exact `1 − p` (`0.49999999999999994` for `S = ½`), and then
    /// returns the non-strict answer of [`quantile`](Self::quantile).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::linprog::{q, qi};
    /// use symplex::stats::survival::{KaplanMeier, Observation};
    ///
    /// // Four events at 1, 2, 3, 4: S(2) = 1/2 exactly.
    /// let km = KaplanMeier::fit(&Observation::from_i64(&[1, 2, 3, 4], &[true; 4]))?;
    /// assert_eq!(km.quantile(&q(1, 2)), Some(qi(2)));          // R: survfit
    /// assert_eq!(km.quantile_strict(&q(1, 2)), Some(qi(3)));   // statsmodels: SurvfuncRight.quantile(0.5)
    /// # Ok::<(), symplex::prelude::SymplexError>(())
    /// ```
    pub fn quantile_strict(&self, p: &Q) -> Option<Q> {
        if !unit_closed(p) {
            return None;
        }
        let target = Q::one() - p;
        self.rows
            .iter()
            .find(|r| r.survival < target)
            .map(|r| r.time.clone())
    }

    /// The median survival time (the `½`-quantile).
    pub fn median(&self) -> Option<Q> {
        self.quantile(&Q::new(1.into(), 2.into()))
    }

    /// A pointwise confidence interval for `S(t)`, in `f64`, from Greenwood's
    /// variance ([`variance_at`](Self::variance_at)) and `z = Φ⁻¹((1 + c)/2)`:
    /// the plain (linear) interval `S ± z·√Var`, clamped to `[0, 1]`, or the
    /// log-log interval, `u = ln(−ln S) ± z·√Var / |S ln S|` mapped back by
    /// `S = exp(−e^u)`, i.e. `S^{exp(±z·√Var / (S ln S))}` (Kalbfleisch &
    /// Prentice 1980), which stays inside `(0, 1)`: the transform of
    /// `SurvfuncRight.quantile_ci(method="cloglog")` and of
    /// `simultaneous_cb(transform="log")`.  statsmodels 0.15 has no pointwise
    /// interval for `S(t)` itself.  `ln S` is taken from the exact `S` (as
    /// `ln(1 − (1 − S))` with `1 − S` exact when `S > ½`), so the interval
    /// keeps its digits as `S → 1`.  Where the variance is `0` — before the
    /// first event (`S = 1`) and once `S = 0` — both intervals are the
    /// point `[S, S]`.
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
        // z = −Φ⁻¹((1 − c)/2): `1 − c` is exact for c ≥ ½, while the
        // `0.5 + c/2` of the upper tail rounds away the digits of 1 − c.
        let z = -norm_ppf((1.0 - confidence) / 2.0);
        let exact = self.survival_at(t);
        let s = q_to_f64(&exact);
        let se = q_to_f64(&self.variance_at(t)).sqrt();
        Ok(match method {
            CiMethod::Linear => Interval::closed((s - z * se).max(0.0), (s + z * se).min(1.0)),
            CiMethod::LogLog => {
                if !exact.is_positive() || exact >= Q::one() || se == 0.0 {
                    Interval::closed(s, s)
                } else {
                    let ln_s = ln_survival(&exact);
                    let theta = z * se / (s * ln_s);
                    // S^{e^{±θ}} = exp(e^{±θ} ln S), without rounding S first.
                    let lo = (theta.exp() * ln_s).exp();
                    let hi = ((-theta).exp() * ln_s).exp();
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
/// rational, referred to `χ²(k − 1)`: the p-value is the upper tail
/// `Γ(ν/2, x/2)/Γ(ν/2)` itself, never `1 − cdf`, so it keeps its digits
/// far below `10⁻¹⁶`.  The statistic is `statsmodels.duration.survfunc.
/// survdiff`'s (whose p-value, `1 − chi2.cdf`, is `0` below about
/// `10⁻¹⁶`; compare `chi2.sf`).  Groups are labelled `0..k`.
///
/// # Errors
///
/// - [`SymplexError::InvalidArgument`] for an empty sample, a negative
///   time, fewer than two groups, mismatched lengths, a label in `0..k`
///   with no observations, or no events at all.
/// - [`SymplexError::ComputationFailed`] when the covariance matrix is
///   singular: some group has nobody at risk at any event time (all of it
///   censored before the first event), so its `O − E` is identically `0`
///   and the `χ²(k − 1)` reference does not apply (`survdiff` raises
///   `LinAlgError`); drop that group.
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
    // Per distinct time (descending): each group's events and its subjects
    // leaving after that time, so the risk sets are running sums.
    let mut order: Vec<usize> = (0..obs.len()).collect();
    order.sort_by(|&a, &b| obs[b].time.cmp(&obs[a].time));
    // O − E and V over the first k − 1 groups.
    let m = k - 1;
    let mut diff = vec![Q::zero(); m];
    let mut v = vec![vec![Q::zero(); m]; m];
    let mut any_event = false;
    let mut at_risk = vec![0usize; k];
    let mut start = 0;
    while start < order.len() {
        let time = &obs[order[start]].time;
        let end = order[start..]
            .iter()
            .position(|&i| obs[i].time != *time)
            .map_or(order.len(), |len| start + len);
        let mut observed = vec![0usize; k];
        for &i in &order[start..end] {
            at_risk[groups[i]] += 1;
            observed[groups[i]] += usize::from(obs[i].event);
        }
        start = end;
        let events: usize = observed.iter().sum();
        if events == 0 {
            continue;
        }
        any_event = true;
        let n: usize = at_risk.iter().sum();
        let d = qu(events);
        let nn = qu(n);
        for g in 0..m {
            let expected = &d * qu(at_risk[g]) / &nn;
            diff[g] += qu(observed[g]) - expected;
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
    let p_value = chi_squared_sf_q(ctx, m, &stat);
    let statistic = ctx.from_ratio(stat);
    Ok(TestResult {
        statistic,
        p_value,
        df: Some(ex_usize(ctx, m)),
        alternative: Alternative::TwoSided,
    })
}

/// The exponential hazard rate estimate `λ̂ = events / total time at risk`
/// (the MLE under right censoring), exactly.
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
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for an empty sample, a negative time,
/// or no events.
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

/// The survival function `S(t) = P(T > t)` of a [`Distribution`] as an
/// expression.
///
/// * For a numeric `t` it is [`Distribution::sf`]: `1` below the support,
///   `0` at or above its upper end, and on the support the family's
///   non-cancelling upper tail (`½ erfc(t/√2)` for a normal), so that
///   `S(26) = 2.476… × 10⁻¹⁴⁹` for `N(0, 1)` rather than the `0` of
///   `1 − Φ(26)`.
/// * For a symbolic `t` it is the closed form on the support: the family's
///   survival function when it has one (`e^{−λt}` for `Exponential(λ)`),
///   else `1 − F(t)` with the family's (or the whole-line) CDF.
pub fn survival_function(dist: &Distribution, t: &Ex) -> Ex {
    if t.free_symbols().is_empty() {
        return dist.sf(t);
    }
    if let Some(sf) = dist.family().sf(t) {
        return sf.simplify();
    }
    let cdf = dist.family().cdf(t).unwrap_or_else(|| dist.cdf(t));
    (dist.context().one() - cdf).simplify()
}

/// The hazard function `h(t) = f(t) / S(t)` of a continuous
/// [`Distribution`] as an expression, with `S` from
/// [`survival_function`].  A symbolic `t` gets the formula valid on the
/// support.  A numeric `t` gets `0` below the support (where `f = 0`,
/// `S = 1`) and the tail-safe ratio on it (`h(40) = 40.0249…` for
/// `N(0, 1)`, where `f/(1 − Φ)` is `0/0`); at or above the upper end of
/// a bounded support `S = 0` and the hazard is undefined.
pub fn hazard_function(dist: &Distribution, t: &Ex) -> Ex {
    if t.free_symbols().is_empty() {
        let support = dist.support();
        let below = support.contains(t) == Some(false)
            && support
                .as_interval()
                .and_then(|iv| sign_of(&(t - &iv.lower)))
                == Some(std::cmp::Ordering::Less);
        if below {
            return dist.context().zero();
        }
    }
    (dist.density(t) / survival_function(dist, t)).simplify()
}
