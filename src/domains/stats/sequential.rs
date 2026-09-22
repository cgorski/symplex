//! Wald's sequential probability ratio test (SPRT): decide between two
//! simple hypotheses one observation at a time, stopping as soon as the
//! evidence is strong enough — the tool for screening workers, raters or
//! models on the fly rather than after a fixed batch.
//!
//! After each observation the log-likelihood ratio
//! `Λₙ = Σᵢ ln(f₁(xᵢ)/f₀(xᵢ))` is compared with Wald's boundaries
//!
//! `A = ln(β / (1 − α))`,  `B = ln((1 − β) / α)`:
//!
//! `Λₙ ≥ B` accepts `H₁`, `Λₙ ≤ A` accepts `H₀`, and otherwise the test
//! continues.  `α` and `β` are the *nominal* type I and II error rates;
//! Wald's boundaries ignore the overshoot at stopping, so the realised
//! rates are at most about `α` and `β` (Wald, *Sequential Analysis*,
//! 1947, §3; Wetherill & Glazebrook, *Sequential Methods in Statistics*,
//! ch. 2).
//!
//! ```
//! use symplex::prelude::*;
//! use symplex::linprog::q;
//! use symplex::stats::sequential::{Decision, Sprt};
//!
//! // Is a worker's accuracy 90 % (H₁) rather than 70 % (H₀)?  α = 5 %, β = 10 %.
//! let mut test = Sprt::bernoulli(&q(7, 10), &q(9, 10), 0.05, 0.10)?;
//! let mut decision = Decision::Continue;
//! while decision == Decision::Continue {
//!     decision = test.update(true);          // every answer correct
//! }
//! assert_eq!(decision, Decision::AcceptH1);
//! assert_eq!(test.observations(), 12);      // ⌈B / ln(9/7)⌉ = ⌈2.8904 / 0.2513⌉
//! # Ok::<(), SymplexError>(())
//! ```
//!
//! The Bernoulli log-likelihood ratio is exact in the counts,
//! `Λ = s·ln(p₁/p₀) + f·ln((1−p₁)/(1−p₀))`, and is returned as an
//! expression by [`Sprt::log_likelihood_ratio`]; decisions compare its
//! `f64` value with the `f64` boundaries.
//!
//! Hypothesis parameters (`p₀`, `p₁`, `μ₀`, `μ₁`, `σ`) and observations are
//! `&Q`, as everywhere in `stats`; the error rates `α`, `β` are `f64`
//! levels.

use num_traits::{One, Signed, Zero};

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::errors::SymplexError;
use crate::base::interval::{Bounds, Interval};
use crate::domains::optimize::{RootOpts, bisect, grow_bracket};

use super::common::{check_alpha, check_unit_open, ex_usize, invalid, q_to_f64};
use super::data::Q;

fn check_rates(op: &'static str, alpha: f64, beta: f64) -> Result<(), SymplexError> {
    check_alpha(op, alpha)?;
    check_unit_open(op, "beta", beta)?;
    if alpha + beta >= 1.0 {
        return Err(invalid(
            op,
            format!(
                "alpha + beta must be below 1 for the boundaries to be ordered, got {alpha} + {beta}"
            ),
        ));
    }
    Ok(())
}

fn check_probability(op: &'static str, p: &Q, what: &str) -> Result<(), SymplexError> {
    if !p.is_positive() || p >= &Q::one() {
        return Err(invalid(
            op,
            format!("{what} must lie strictly in (0, 1), got {p}"),
        ));
    }
    Ok(())
}

/// Wald's continuation region `(A, B)` for nominal error rates `α`, `β`:
/// the test goes on while `A < Λ < B`.  `lower = A = ln(β/(1−α))` (the
/// log-likelihood ratio falling to it accepts `H₀`) and
/// `upper = B = ln((1−β)/α)` (rising to it accepts `H₁`); the region is
/// *open* because reaching a boundary decides.
///
/// ```
/// use symplex::stats::sequential::wald_boundaries;
///
/// let bounds = wald_boundaries(0.05, 0.10)?;
/// assert!((bounds.lower - (-2.251291798606495)).abs() < 1e-12);   // A = ln(0.1/0.95)
/// assert!((bounds.upper - 2.8903717578961645).abs() < 1e-12);     // B = ln(0.9/0.05)
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless `0 < α, β` and `α + β < 1`.
pub fn wald_boundaries(alpha: f64, beta: f64) -> Result<Interval<f64>, SymplexError> {
    check_rates("wald_boundaries", alpha, beta)?;
    Ok(Interval::open(
        (beta / (1.0 - alpha)).ln(),
        ((1.0 - beta) / alpha).ln(),
    ))
}

/// The state of a sequential test after an observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Decision {
    /// Neither boundary crossed: observe more.
    Continue,
    /// The log-likelihood ratio fell to `A`: accept `H₀`.
    AcceptH0,
    /// The log-likelihood ratio rose to `B`: accept `H₁`.
    AcceptH1,
}

/// The pair of simple hypotheses under test.
#[derive(Clone, Debug, PartialEq)]
enum Model {
    /// `H₀: X ~ Bernoulli(p₀)` against `H₁: X ~ Bernoulli(p₁)`.
    Bernoulli { p0: Q, p1: Q },
    /// `H₀: X ~ Normal(μ₀, σ)` against `H₁: X ~ Normal(μ₁, σ)`, `σ` known.
    NormalMean { mu0: Q, mu1: Q, sigma: Q },
}

/// A sequential probability ratio test in progress.  Feed observations
/// with [`update`](Self::update) (a success / failure) or
/// [`observe`](Self::observe) (a value); each returns the [`Decision`]
/// reached so far.  The test is a plain value: clone it to branch, and
/// [`reset`](Self::reset) it to start over with the same design.
///
/// **Decisions are sticky.**  A sequential test *stops* at the first
/// boundary crossing: once `update`/`observe` has returned `AcceptH0` or
/// `AcceptH1`, every later call returns that same decision, whatever the
/// new data do to the likelihood ratio, and [`decision`](Self::decision)
/// reports it too ([`is_decided`](Self::is_decided) says whether this has
/// happened).  The observations are still recorded — `observations()`,
/// `successes()`, `sum()` and the log-likelihood ratio keep counting, so
/// a batch fed after the fact can still be inspected — and
/// [`stopped_at`](Self::stopped_at) gives the sample size at which the
/// test stopped.  [`reset`](Self::reset) clears the decision with the
/// data.  (Before 0.18.1 `decision()` re-evaluated the current ratio, so a
/// test that had accepted `H₁` could report `Continue` again.)
#[derive(Clone, Debug, PartialEq)]
pub struct Sprt {
    model: Model,
    alpha: f64,
    beta: f64,
    /// Wald's continuation region `(A, B)`.
    boundaries: Interval<f64>,
    observations: usize,
    successes: usize,
    sum: Q,
    /// The first terminal decision and the observation count at which it
    /// was reached; `None` while the test continues.
    stopped: Option<Stop>,
}

/// Where a sequential test stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stop {
    decision: Decision,
    at: usize,
}

impl Sprt {
    /// A test of `H₀: p = p₀` against `H₁: p = p₁` for a Bernoulli success
    /// probability, with nominal error rates `α` (accepting `H₁` under
    /// `H₀`) and `β` (accepting `H₀` under `H₁`).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] unless `0 < p₀, p₁ < 1`,
    /// `p₀ ≠ p₁`, `0 < α, β` and `α + β < 1`.
    pub fn bernoulli(p0: &Q, p1: &Q, alpha: f64, beta: f64) -> Result<Self, SymplexError> {
        const OP: &str = "Sprt::bernoulli";
        check_probability(OP, p0, "p0")?;
        check_probability(OP, p1, "p1")?;
        if p0 == p1 {
            return Err(invalid(OP, "p0 and p1 must differ"));
        }
        check_rates(OP, alpha, beta)?;
        let boundaries = wald_boundaries(alpha, beta)?;
        Ok(Sprt {
            model: Model::Bernoulli {
                p0: p0.clone(),
                p1: p1.clone(),
            },
            alpha,
            beta,
            boundaries,
            observations: 0,
            successes: 0,
            sum: Q::zero(),
            stopped: None,
        })
    }

    /// A test of `H₀: μ = μ₀` against `H₁: μ = μ₁` for the mean of a
    /// normal distribution with known standard deviation `σ`.  The
    /// log-likelihood ratio after observations `x₁, …, xₙ` is
    /// `((μ₁ − μ₀)/σ²)·Σxᵢ − n(μ₁² − μ₀²)/(2σ²)`, exact in the data.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] unless `σ > 0`, `μ₀ ≠ μ₁`,
    /// `0 < α, β` and `α + β < 1`.
    pub fn normal_mean(
        mu0: &Q,
        mu1: &Q,
        sigma: &Q,
        alpha: f64,
        beta: f64,
    ) -> Result<Self, SymplexError> {
        const OP: &str = "Sprt::normal_mean";
        if !sigma.is_positive() {
            return Err(invalid(OP, format!("sigma must be positive, got {sigma}")));
        }
        if mu0 == mu1 {
            return Err(invalid(OP, "mu0 and mu1 must differ"));
        }
        check_rates(OP, alpha, beta)?;
        let boundaries = wald_boundaries(alpha, beta)?;
        Ok(Sprt {
            model: Model::NormalMean {
                mu0: mu0.clone(),
                mu1: mu1.clone(),
                sigma: sigma.clone(),
            },
            alpha,
            beta,
            boundaries,
            observations: 0,
            successes: 0,
            sum: Q::zero(),
            stopped: None,
        })
    }

    /// Record a success (`true`) or failure (`false`) and return the
    /// decision so far.  For a [`normal_mean`](Self::normal_mean) test this
    /// records the observation `1` or `0`; use [`observe`](Self::observe)
    /// for real-valued data.  Once a terminal decision has been reached it
    /// is returned unchanged (see the type's docs).
    pub fn update(&mut self, success: bool) -> Decision {
        self.observations += 1;
        if success {
            self.successes += 1;
            self.sum += Q::one();
        }
        self.settle()
    }

    /// Record an observed value and return the decision so far.  A
    /// Bernoulli test accepts only `0` and `1`.  Once a terminal decision
    /// has been reached it is returned unchanged (see the type's docs).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if a Bernoulli test is given a
    /// value other than `0` or `1`.
    pub fn observe(&mut self, x: &Q) -> Result<Decision, SymplexError> {
        match &self.model {
            Model::Bernoulli { .. } => {
                if x.is_one() {
                    Ok(self.update(true))
                } else if x.is_zero() {
                    Ok(self.update(false))
                } else {
                    Err(invalid(
                        "Sprt::observe",
                        format!("a Bernoulli observation must be 0 or 1, got {x}"),
                    ))
                }
            }
            Model::NormalMean { .. } => {
                self.observations += 1;
                if x.is_positive() {
                    self.successes += 1;
                }
                self.sum += x;
                Ok(self.settle())
            }
        }
    }

    /// After an observation: the sticky decision if there is one, else the
    /// current boundary test, recording the first terminal outcome.
    fn settle(&mut self) -> Decision {
        if let Some(stop) = self.stopped {
            return stop.decision;
        }
        let decision = self.current();
        if decision != Decision::Continue {
            self.stopped = Some(Stop {
                decision,
                at: self.observations,
            });
        }
        decision
    }

    /// The boundary test on the current ratio: `Λ ≥ B` accepts `H₁`,
    /// `Λ ≤ A` accepts `H₀`, otherwise continue.
    fn current(&self) -> Decision {
        let llr = self.log_likelihood_ratio_f64();
        if llr >= self.boundaries.upper {
            Decision::AcceptH1
        } else if llr <= self.boundaries.lower {
            Decision::AcceptH0
        } else {
            Decision::Continue
        }
    }

    /// The decision so far: the terminal decision once one has been
    /// reached (sticky, see the type's docs), otherwise the boundary test
    /// on the current ratio — `Λ ≥ B` accepts `H₁`, `Λ ≤ A` accepts `H₀`,
    /// and in between the test continues.
    pub fn decision(&self) -> Decision {
        match self.stopped {
            Some(stop) => stop.decision,
            None => self.current(),
        }
    }

    /// `true` once `update`/`observe` has returned `AcceptH0` or
    /// `AcceptH1`; the decision then stays until [`reset`](Self::reset).
    pub fn is_decided(&self) -> bool {
        self.stopped.is_some()
    }

    /// The number of observations at which the test stopped (Wald's `N`),
    /// or `None` while it continues.
    pub fn stopped_at(&self) -> Option<usize> {
        self.stopped.map(|s| s.at)
    }

    /// Forget every observation and any decision reached; the design
    /// (hypotheses, boundaries) stays.
    pub fn reset(&mut self) {
        self.observations = 0;
        self.successes = 0;
        self.sum = Q::zero();
        self.stopped = None;
    }

    /// Wald's open continuation region `(A, B)` (`lower = A` accepts `H₀`,
    /// `upper = B` accepts `H₁`; `contains(&llr)` is exactly "continue");
    /// see [`wald_boundaries`].
    pub fn boundaries(&self) -> Interval<f64> {
        self.boundaries
    }

    /// The nominal type I error rate.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// The nominal type II error rate.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Observations recorded so far.
    pub fn observations(&self) -> usize {
        self.observations
    }

    /// Successes recorded so far (for a normal-mean test: positive
    /// observations).
    pub fn successes(&self) -> usize {
        self.successes
    }

    /// Failures recorded so far: `observations − successes`.
    pub fn failures(&self) -> usize {
        self.observations - self.successes
    }

    /// The sum of the observed values (`= successes` for a Bernoulli test).
    pub fn sum(&self) -> &Q {
        &self.sum
    }

    /// The exact log-likelihood ratio `Λₙ = ln(L₁/L₀)` of the observations
    /// so far as an expression in `ctx`:
    /// `s·ln(p₁/p₀) + f·ln((1−p₁)/(1−p₀))` for a Bernoulli test, and the
    /// rational `((μ₁ − μ₀)/σ²)·Σxᵢ − n(μ₁² − μ₀²)/(2σ²)` for a normal-mean
    /// test.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::linprog::q;
    /// use symplex::stats::sequential::Sprt;
    ///
    /// let ctx = Context::new();
    /// let mut test = Sprt::bernoulli(&q(7, 10), &q(9, 10), 0.05, 0.10)?;
    /// for s in [true, true, false, true, false] {
    ///     test.update(s);
    /// }
    /// let expected = 3 * ctx.rational(9, 7).ln() + 2 * ctx.rational(1, 3).ln();
    /// assert_eq!(test.log_likelihood_ratio(&ctx).equals(&expected), Some(true));
    /// # Ok::<(), SymplexError>(())
    /// ```
    pub fn log_likelihood_ratio(&self, ctx: &Context) -> Ex {
        match &self.model {
            Model::Bernoulli { p0, p1 } => {
                let s = ex_usize(ctx, self.successes);
                let f = ex_usize(ctx, self.failures());
                let one = Q::one();
                s * ctx.from_ratio(p1 / p0).ln()
                    + f * ctx.from_ratio((&one - p1) / (&one - p0)).ln()
            }
            Model::NormalMean { .. } => ctx.from_ratio(self.normal_llr_exact()),
        }
    }

    /// The log-likelihood ratio as a float (what the decision compares
    /// with the boundaries).
    pub fn log_likelihood_ratio_f64(&self) -> f64 {
        match &self.model {
            Model::Bernoulli { p0, p1 } => {
                let one = Q::one();
                let ls = q_to_f64(&(p1 / p0)).ln();
                let lf = q_to_f64(&((&one - p1) / (&one - p0))).ln();
                self.successes as f64 * ls + self.failures() as f64 * lf
            }
            Model::NormalMean { .. } => q_to_f64(&self.normal_llr_exact()),
        }
    }

    /// `((μ₁ − μ₀)/σ²)·S − n(μ₁² − μ₀²)/(2σ²)`, exactly.
    fn normal_llr_exact(&self) -> Q {
        match &self.model {
            Model::NormalMean { mu0, mu1, sigma } => {
                let var = sigma * sigma;
                let n = Q::from_integer(self.observations.into());
                let two = Q::from_integer(2.into());
                (mu1 - mu0) / &var * &self.sum - n * (mu1 * mu1 - mu0 * mu0) / (&two * &var)
            }
            Model::Bernoulli { .. } => Q::zero(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Wald's approximations for the Bernoulli test
// ═══════════════════════════════════════════════════════════════════════════

/// The per-observation log-likelihood-ratio increments
/// `(ln(p₁/p₀), ln((1−p₁)/(1−p₀)))` and the expectation of one increment
/// under a true success probability `p`.
fn bernoulli_increments(
    op: &'static str,
    p: f64,
    p0: &Q,
    p1: &Q,
) -> Result<(f64, f64, f64), SymplexError> {
    check_probability(op, p0, "p0")?;
    check_probability(op, p1, "p1")?;
    if p0 == p1 {
        return Err(invalid(op, "p0 and p1 must differ"));
    }
    if !(0.0..=1.0).contains(&p) {
        return Err(invalid(op, format!("p must lie in [0, 1], got {p}")));
    }
    let one = Q::one();
    let ls = q_to_f64(&(p1 / p0)).ln();
    let lf = q_to_f64(&((&one - p1) / (&one - p0))).ln();
    Ok((ls, lf, p * ls + (1.0 - p) * lf))
}

/// The non-zero root `h` of `p·(p₁/p₀)ʰ + (1−p)·((1−p₁)/(1−p₀))ʰ = 1`
/// (Wald's fundamental identity), or `None` when `E_p[Z] = 0` (`h → 0`).
/// The caller excludes `p ∈ {0, 1}`, where the only root is `h = 0` (the
/// non-zero root has escaped to `±∞`).
fn wald_h(
    op: &'static str,
    p: f64,
    ls: f64,
    lf: f64,
    drift: f64,
) -> Result<Option<f64>, SymplexError> {
    if drift.abs() < 1e-13 {
        return Ok(None);
    }
    // g(h) = p e^{h ls} + (1−p) e^{h lf} − 1 is convex with g(0) = 0 and
    // g'(0) = drift, so the other root lies on the side where g' < 0 first.
    // Divide out the root at 0: φ(h) = g(h)/h is increasing, φ(0) = drift,
    // and its only zero is the root wanted (`exp_m1` keeps it accurate
    // near 0).
    let phi = |h: f64| {
        if h == 0.0 {
            drift
        } else {
            (p * (h * ls).exp_m1() + (1.0 - p) * (h * lf).exp_m1()) / h
        }
    };
    let side = if drift < 0.0 { 1.0 } else { -1.0 };
    let (a, b, bounds) = if side > 0.0 {
        (0.0, side, Bounds::at_least(0.0))
    } else {
        (side, 0.0, Bounds::at_most(0.0))
    };
    let bracket = grow_bracket(phi, a, b, bounds, 60).map_err(|_| {
        SymplexError::computation_failed(
            op,
            "operating characteristic: could not bracket the root of Wald's identity",
        )
    })?;
    // To the last bits, as the fixed 200-step bisection this replaces.
    let tight = RootOpts {
        xtol: 0.0,
        max_iter: 200,
        ..RootOpts::default()
    };
    bisect(phi, bracket.lower, bracket.upper, &tight)
        .map(Some)
        .map_err(|e| SymplexError::computation_failed(op, e.to_string()))
}

/// Wald's `L(p)` from the increments, the drift and the boundaries `(a, b)`.
fn wald_oc(
    op: &'static str,
    p: f64,
    ls: f64,
    lf: f64,
    drift: f64,
    a: f64,
    b: f64,
) -> Result<f64, SymplexError> {
    // Every trial has the same outcome, so `Λₙ = n·drift` marches to the
    // boundary on the side of the drift (the non-zero root of Wald's
    // identity has escaped to `±∞`, where the formula tends to 0 or 1).
    if p <= 0.0 || p >= 1.0 {
        return Ok(if drift < 0.0 { 1.0 } else { 0.0 });
    }
    match wald_h(op, p, ls, lf, drift)? {
        // (eᴮʰ − 1)/(eᴮʰ − eᴬʰ), i.e. ((1−β)/α)ʰ = e^{Bh}, (β/(1−α))ʰ = e^{Ah}
        Some(h) => Ok(((b * h).exp() - 1.0) / ((b * h).exp() - (a * h).exp())),
        // h → 0: L = B / (B − A)
        None => Ok(b / (b - a)),
    }
}

/// Wald's approximation to the operating characteristic `L(p)` of the
/// Bernoulli test — the probability of accepting `H₀` when the true
/// success probability is `p`:
///
/// `L(p) ≈ (((1−β)/α)ʰ − 1) / (((1−β)/α)ʰ − (β/(1−α))ʰ)`
///
/// with `h = h(p)` the non-zero root of `p(p₁/p₀)ʰ + (1−p)((1−p₁)/(1−p₀))ʰ = 1`
/// (`h = 1` at `p₀`, giving `1 − α`; `h = −1` at `p₁`, giving `β`).  At
/// `p = 0` and `p = 1` every trial has the same outcome and `L` is the
/// limit `1` or `0` (accepting the hypothesis the constant sequence
/// favours).  The approximation ignores the overshoot of the boundaries
/// (Wald 1947, §3.4).
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for invalid parameters;
/// [`SymplexError::ComputationFailed`] if the root cannot be bracketed.
pub fn operating_characteristic_bernoulli(
    p: f64,
    p0: &Q,
    p1: &Q,
    alpha: f64,
    beta: f64,
) -> Result<f64, SymplexError> {
    const OP: &str = "operating_characteristic_bernoulli";
    let (ls, lf, drift) = bernoulli_increments(OP, p, p0, p1)?;
    check_rates(OP, alpha, beta)?;
    let (a, b) = wald_boundaries(alpha, beta)?.into_pair();
    wald_oc(OP, p, ls, lf, drift, a, b)
}

/// Wald's approximation to the expected number of observations of the
/// Bernoulli test when the true success probability is `p`:
///
/// `E_p[N] ≈ (L(p)·A + (1 − L(p))·B) / E_p[Z]`,  `Z = ln(f₁(X)/f₀(X))`,
///
/// with `L(p)` the [`operating_characteristic_bernoulli`]; when
/// `E_p[Z] = 0` the limit `−A·B / E_p[Z²]` is used, and at `p = 0` or
/// `p = 1` the formula reduces to the boundary over the constant
/// increment (`B / ln(p₁/p₀)` for all successes when `p₁ > p₀`).  This is
/// an **approximation**: it treats `Λ` as landing exactly on a boundary, so
/// it underestimates the true expectation slightly (Wald 1947, §3.5;
/// Wetherill & Glazebrook §2.4).
///
/// ```
/// use symplex::linprog::q;
/// use symplex::stats::sequential::expected_sample_size_bernoulli;
///
/// // At p = p₀: ((1−α)A + αB) / (p₀ ln(p₁/p₀) + (1−p₀) ln((1−p₁)/(1−p₀))) = 12.9777…
/// let n = expected_sample_size_bernoulli(0.7, &q(7, 10), &q(9, 10), 0.05, 0.10)?;
/// assert!((n - 12.977756554177112).abs() < 1e-9);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// As [`operating_characteristic_bernoulli`].
pub fn expected_sample_size_bernoulli(
    p: f64,
    p0: &Q,
    p1: &Q,
    alpha: f64,
    beta: f64,
) -> Result<f64, SymplexError> {
    const OP: &str = "expected_sample_size_bernoulli";
    let (ls, lf, drift) = bernoulli_increments(OP, p, p0, p1)?;
    check_rates(OP, alpha, beta)?;
    let (a, b) = wald_boundaries(alpha, beta)?.into_pair();
    if drift.abs() < 1e-13 {
        let second_moment = p * ls * ls + (1.0 - p) * lf * lf;
        return Ok(-a * b / second_moment);
    }
    let l = wald_oc(OP, p, ls, lf, drift, a, b)?;
    Ok((l * a + (1.0 - l) * b) / drift)
}
