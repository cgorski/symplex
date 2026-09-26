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
//! continues.  (Wald names the likelihood-ratio thresholds the other way
//! round: his `A = (1−β)/α` is `e^B` here and his `B = β/(1−α)` is
//! `e^A`.)  `α` and `β` are the *nominal* type I and II error rates;
//! Wald's boundaries ignore the overshoot at stopping, and what holds
//! exactly for the realised rates `α′`, `β′` are Wald's inequalities
//! `α′ ≤ α/(1 − β)`, `β′ ≤ β/(1 − α)` and `α′ + β′ ≤ α + β`; in
//! practice both are usually below nominal (Wald, *Sequential Analysis*,
//! 1947, §3; Wetherill & Glazebrook, *Sequential Methods in Statistics*,
//! ch. 2).  [`operating_characteristic_bernoulli`] and
//! [`expected_sample_size_bernoulli`] are Wald's *approximations* to the
//! operating characteristic and the average sample number; their docs say
//! how far off they typically are.
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

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::errors::SymplexError;
use crate::base::interval::Interval;
use crate::domains::optimize::{RootOpts, bisect};

use super::common::{check_alpha, check_unit_open, ex_usize, invalid, q_to_f64};
use super::data::Q;

/// `ln(num/den)` for positive rationals, to a few ulps whatever their
/// size: near `1` it is `ln_1p` of the exact `(num − den)/den` (rounding
/// the quotient first would leave only `ε/|num/den − 1|` of relative
/// accuracy), and a quotient that over- or underflows `f64` is rescaled
/// by a power of two first (`ln(1 − 10⁻⁴⁰⁰)` and `ln(10⁻⁴⁰⁰)` are finite).
fn ln_ratio(num: &Q, den: &Q) -> f64 {
    let ratio = num / den;
    let x = q_to_f64(&ratio);
    if (0.5..=2.0).contains(&x) {
        return q_to_f64(&((num - den) / den)).ln_1p();
    }
    if x.is_finite() && x >= f64::MIN_POSITIVE {
        return x.ln();
    }
    let bits = |n: &BigInt| i64::try_from(n.bits()).unwrap_or(i64::MAX);
    let k = bits(ratio.numer()).saturating_sub(bits(ratio.denom()));
    let shift = usize::try_from(k.unsigned_abs()).unwrap_or(usize::MAX);
    let pow = Q::from_integer(BigInt::one() << shift);
    let scaled = if k >= 0 { ratio / pow } else { ratio * pow };
    q_to_f64(&scaled).ln() + k as f64 * std::f64::consts::LN_2
}

/// The exact value of a finite `f64` (`0` for a non-finite one, which the
/// callers have already refused).
fn exact(v: f64) -> Q {
    Q::from_float(v).unwrap_or_else(Q::zero)
}

/// `0 < α, β` and `α + β < 1` — the sum in `f64`, so that the decimal
/// `0.3 + 0.7` counts as `1` (its binary values sum to `1 − 5.6·10⁻¹⁷`).
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
    // From the exact values of `α`, `β`: both boundaries keep their
    // relative accuracy as `α + β → 1` squeezes them towards `0`.
    let (a, b, one) = (exact(alpha), exact(beta), Q::one());
    Ok(Interval::open(
        ln_ratio(&b, &(&one - &a)),
        ln_ratio(&(&one - &b), &a),
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
    /// with the boundaries).  The per-observation increments are the
    /// logarithms of exact ratios, finite for every `p₀, p₁` in `(0, 1)`
    /// (so `p₁ = 1 − 10⁻⁴⁰⁰` cannot turn a failure-free run into
    /// `0 · (−∞) = NaN`), and accurate when `p₁/p₀` is close to `1`.
    pub fn log_likelihood_ratio_f64(&self) -> f64 {
        match &self.model {
            Model::Bernoulli { p0, p1 } => {
                let inc = BernoulliIncrements::new(p0, p1);
                self.successes as f64 * inc.success + self.failures() as f64 * inc.failure
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

/// The log-likelihood-ratio increments of one success, `ln(p₁/p₀)`, and
/// of one failure, `ln((1−p₁)/(1−p₀))`, from the exact ratios (see
/// [`ln_ratio`]).
#[derive(Clone, Copy, Debug)]
struct BernoulliIncrements {
    success: f64,
    failure: f64,
}

impl BernoulliIncrements {
    fn new(p0: &Q, p1: &Q) -> Self {
        let one = Q::one();
        BernoulliIncrements {
            success: ln_ratio(p1, p0),
            failure: ln_ratio(&(&one - p1), &(&one - p0)),
        }
    }

    /// The increments for Wald's approximations at a true success
    /// probability `p`, after validating the design.
    fn checked(op: &'static str, p: f64, p0: &Q, p1: &Q) -> Result<Self, SymplexError> {
        check_probability(op, p0, "p0")?;
        check_probability(op, p1, "p1")?;
        if p0 == p1 {
            return Err(invalid(op, "p0 and p1 must differ"));
        }
        if !(0.0..=1.0).contains(&p) {
            return Err(invalid(op, format!("p must lie in [0, 1], got {p}")));
        }
        Ok(Self::new(p0, p1))
    }
}

/// `w·(eᶻ − 1)` for a weight `w ∈ (0, 1]`, finite whenever `w·eᶻ` is
/// (at the root of Wald's identity `w·eᶻ ≤ 1` however large `z` is, as
/// for `p = 10⁻³⁰⁰`).
fn w_expm1(w: f64, z: f64) -> f64 {
    if z <= 1.0 {
        w * z.exp_m1()
    } else if z <= 700.0 {
        w * z.exp() - w
    } else {
        (w.ln() + z).exp() - w
    }
}

/// `w·(e^{hl} − 1 − hl)/h²`, the second-order remainder of one term of
/// Wald's identity: non-negative, by the series `w·l²·Σₖ zᵏ/(k+2)!` for
/// `|z| = |hl| ≤ 1` (no cancellation, and `w·l²/2` at `h = 0`), directly
/// beyond (where `eᶻ − 1` and `z` no longer cancel).
fn remainder_over_h2(w: f64, l: f64, h: f64) -> f64 {
    let z = h * l;
    if z.abs() <= 1.0 {
        let (mut term, mut sum) = (0.5_f64, 0.5_f64);
        for k in 1..=30 {
            term *= z / f64::from(k + 2);
            sum += term;
            if term.abs() <= 1e-17 * sum {
                break;
            }
        }
        w * l * l * sum
    } else {
        (w_expm1(w, z) - w * z) / (h * h)
    }
}

/// The non-zero root `h` of Wald's fundamental identity
/// `p·e^{h·ls} + (1−p)·e^{h·lf} = 1` for `p ∈ (0, 1)`, or `0` when the
/// drift `E_p[Z] = p·ls + (1−p)·lf` is exactly `0` (the two roots merge).
///
/// `g(h) = p(e^{h·ls} − 1) + (1−p)(e^{h·lf} − 1)` is convex with
/// `g(0) = 0` and `g'(0) = E_p[Z]`, so `φ(h) = g(h)/h` is increasing,
/// `φ(0) = E_p[Z]`, and its only zero is the root wanted, on the side
/// opposite to the drift.  At that root the term whose increment `l` has
/// the sign of `h` is below `1` (the other term is positive), so the root
/// lies in `(0, −ln(w)/l]` for that term's weight `w`: an analytic
/// bracket, on which `φ` is finite, bisected to the last bits.  (A
/// doubling search from `±1` used to overflow `e^{h·l}` before reaching
/// the root for small `p`.)  No threshold decides "drift ≈ 0": the drift
/// is in log-likelihood units, so any fixed cut-off is wrong for close
/// hypotheses, and `h` is small and accurate when the drift is.
fn wald_h(
    op: &'static str,
    p: f64,
    inc: BernoulliIncrements,
    drift: f64,
) -> Result<f64, SymplexError> {
    if drift == 0.0 {
        return Ok(0.0);
    }
    let (ls, lf) = (inc.success, inc.failure);
    let g = |h: f64| w_expm1(p, h * ls) + w_expm1(1.0 - p, h * lf);
    let phi = |h: f64| if h == 0.0 { drift } else { g(h) / h };
    // `ls` and `lf` have opposite signs; `h` has the sign of `−drift`.
    let end = if (drift < 0.0) == (ls > 0.0) {
        -p.ln() / ls
    } else {
        -(-p).ln_1p() / lf
    };
    if !end.is_finite() || end == 0.0 {
        return Err(SymplexError::computation_failed(
            op,
            format!("the root of Wald's identity is outside the f64 range (bracket end {end})"),
        ));
    }
    // g(end) > 0 exactly; rounding can hide that only when the root is
    // within rounding of `end`.
    if g(end) <= 0.0 {
        return Ok(end);
    }
    let (lo, hi) = if end > 0.0 { (0.0, end) } else { (end, 0.0) };
    let tight = RootOpts {
        xtol: 0.0,
        max_iter: 200,
        ..RootOpts::default()
    };
    bisect(phi, lo, hi, &tight).map_err(|e| SymplexError::computation_failed(op, e.to_string()))
}

/// Wald's approximations at a true success probability `p`.
#[derive(Clone, Copy, Debug)]
struct WaldApprox {
    /// `L(p)`, the probability of accepting `H₀`.
    oc: f64,
    /// `E_p[N]`, the expected number of observations.
    asn: f64,
}

/// Wald's `L(p)` and `E_p[N]` for the boundaries `a = A < 0 < B = b`.
///
/// With `h` from [`wald_h`], `L = (e^{Bh} − 1)/(e^{Bh} − e^{Ah})` and
/// `E_p[N] = (L·A + (1 − L)·B)/E_p[Z]`.  Evaluated as written these
/// overflow (`inf/inf` once `Bh > 709`) and cancel (numerator and drift
/// both vanish like `h` near the `p` where the drift is `0`, so the ratio
/// came out as `−2.4·10⁷` for a true `23.57`).  Instead:
///
/// * `L` is a ratio of the positive magnitudes `|e^{Bh} − 1|` and
///   `|e^{Ah} − 1|` (the two have opposite signs), divided by the larger;
/// * Wald's identity gives the drift back as `E_p[Z] = −h·s₃`, with `s₃`
///   a sum of the non-negative remainders of [`remainder_over_h2`] (or
///   `−E_p[Z]/h` itself when that has the smaller rounding error);
/// * for `|h|·(B − A) ≤ 1`, `L·A + (1 − L)·B = h·A·B·s₁/s₂` exactly, with
///   `s₁ = R_B/B − R_A/A > 0` (remainders again) and
///   `s₂ = (e^{Bh} − e^{Ah})/h = B·exprel(Bh) − A·exprel(Ah) > 0`, so
///   `E_p[N] = −A·B·s₁/(s₂·s₃)`: a quotient of sums of positive terms,
///   equal to Wald's limit `−A·B/E_p[Z²]` at `h = 0`.  Beyond, the
///   numerator loses at most a bit and the direct form is used.
fn wald_approx(
    op: &'static str,
    p: f64,
    inc: BernoulliIncrements,
    a: f64,
    b: f64,
) -> Result<WaldApprox, SymplexError> {
    let (ls, lf) = (inc.success, inc.failure);
    if p <= 0.0 || p >= 1.0 {
        // Every trial has the same outcome, so Λₙ = n·z marches to the
        // boundary on the side of z (the non-zero root of Wald's identity
        // has escaped to ±∞, where the formulas tend to these limits).
        let z = if p >= 1.0 { ls } else { lf };
        return Ok(if z > 0.0 {
            WaldApprox {
                oc: 0.0,
                asn: b / z,
            }
        } else {
            WaldApprox {
                oc: 1.0,
                asn: a / z,
            }
        });
    }
    let drift = p * ls + (1.0 - p) * lf;
    let h = wald_h(op, p, inc, drift)?;
    // `s₃ = −E_p[Z]/h` two ways.  The remainders cannot cancel, but the
    // growing one carries `h`'s bisection error times `|h·l|` (up to ~745
    // for tiny `p`); the drift is exact to `ε` unless its two terms cancel
    // (near the merged root).  Take the one with the smaller error bound.
    let err_remainders = 4.0 * f64::EPSILON * (h * ls).abs().max((h * lf).abs());
    let err_drift = f64::EPSILON * ((p * ls).abs() + ((1.0 - p) * lf).abs()) / drift.abs();
    let s3 = if h != 0.0 && err_drift < err_remainders {
        -drift / h
    } else {
        remainder_over_h2(p, ls, h) + remainder_over_h2(1.0 - p, lf, h)
    };
    if h.abs() * (b - a) <= 1.0 {
        let exprel = |z: f64| if z == 0.0 { 1.0 } else { z.exp_m1() / z };
        let (xb, xa) = (b * exprel(b * h), -a * exprel(a * h));
        let s1 = remainder_over_h2(1.0, b, h) / b - remainder_over_h2(1.0, a, h) / a;
        let s2 = xb + xa;
        return Ok(WaldApprox {
            oc: xb / s2,
            asn: -a * b * s1 / (s2 * s3),
        });
    }
    let (eb, ea) = ((b * h).exp_m1().abs(), (a * h).exp_m1().abs());
    let (oc, oc_c) = if eb >= ea {
        let r = ea / eb;
        (1.0 / (1.0 + r), r / (1.0 + r))
    } else {
        let r = eb / ea;
        (r / (1.0 + r), 1.0 / (1.0 + r))
    };
    Ok(WaldApprox {
        oc,
        asn: (oc * a + oc_c * b) / (-h * s3),
    })
}

/// Wald's approximation to the operating characteristic `L(p)` of the
/// Bernoulli test — the probability of accepting `H₀` when the true
/// success probability is `p`:
///
/// `L(p) ≈ (((1−β)/α)ʰ − 1) / (((1−β)/α)ʰ − (β/(1−α))ʰ)`
///
/// with `h = h(p)` the non-zero root of `p(p₁/p₀)ʰ + (1−p)((1−p₁)/(1−p₀))ʰ = 1`
/// (`h = 1` at `p₀`, giving `1 − α`; `h = −1` at `p₁`, giving `β`; where
/// the expected increment `E_p[Z]` vanishes the roots merge, `h = 0`, and
/// `L = B/(B − A)`).  At `p = 0` and `p = 1` every trial has the same
/// outcome and `L` is the limit `1` or `0` (accepting the hypothesis the
/// constant sequence favours).
///
/// **An approximation** (Wald 1947, §3.4): it treats the log-likelihood
/// ratio as stopping exactly on a boundary, ignoring the overshoot, and
/// its error has no fixed sign.  The realised error rates are usually
/// *below* the nominal ones, so `L(p₀) ≈ 1 − α` understates the true
/// `L(p₀)` and `L(p₁) ≈ β` overstates the true `L(p₁)`: a 200 000-run
/// simulation of `p₀ = 0.7`, `p₁ = 0.9`, `α = 0.05`, `β = 0.10` gives
/// `L(p₀) = 0.952` and `L(p₁) = 0.067`.  What holds exactly are Wald's
/// inequalities for the realised rates `α′`, `β′`: `α′ ≤ α/(1 − β)`,
/// `β′ ≤ β/(1 − α)` and `α′ + β′ ≤ α + β`.
///
/// The formula itself is evaluated to full `f64` accuracy for every `p`
/// (see the source of `wald_approx`): no overflow for tiny `p` or narrow
/// designs, no cancellation where `E_p[Z] → 0`, and increments from the
/// exact ratios, so `p₁ − p₀ = 10⁻⁹` loses nothing.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for invalid parameters;
/// [`SymplexError::ComputationFailed`] if the root of Wald's identity is
/// outside the `f64` range (hypotheses closer than about `10⁻³⁰⁸`).
pub fn operating_characteristic_bernoulli(
    p: f64,
    p0: &Q,
    p1: &Q,
    alpha: f64,
    beta: f64,
) -> Result<f64, SymplexError> {
    const OP: &str = "operating_characteristic_bernoulli";
    let inc = BernoulliIncrements::checked(OP, p, p0, p1)?;
    check_rates(OP, alpha, beta)?;
    let bounds = wald_boundaries(alpha, beta)?;
    Ok(wald_approx(OP, p, inc, bounds.lower, bounds.upper)?.oc)
}

/// Wald's approximation to the expected number of observations of the
/// Bernoulli test when the true success probability is `p`:
///
/// `E_p[N] ≈ (L(p)·A + (1 − L(p))·B) / E_p[Z]`,  `Z = ln(f₁(X)/f₀(X))`,
///
/// with `L(p)` the [`operating_characteristic_bernoulli`]; when
/// `E_p[Z] = 0` the limit `−A·B / E_p[Z²]` applies (the function is
/// continuous there and evaluated without cancellation on either side),
/// and at `p = 0` or `p = 1` the formula reduces to the boundary over the
/// constant increment (`B / ln(p₁/p₀)` for all successes when `p₁ > p₀`;
/// the test itself stops after `⌈B / ln(p₁/p₀)⌉`).
///
/// **An approximation** (Wald 1947, §3.5; Wetherill & Glazebrook §2.4):
/// it treats `Λ` as landing exactly on a boundary, and the overshoot it
/// ignores made it *underestimate* the true expectation at every point we
/// simulated (four designs, 27 values of `p`, 100 000–200 000 runs each),
/// the more so the shorter the test: by 4–20 % for `p₀ = 0.7`,
/// `p₁ = 0.9`, `α = 0.05`, `β = 0.10` (`15.85`, `22.52`, `27.98`
/// observations at `p = 0.7, 0.9, 0.8` against `12.98`, `20.43`, `22.60`),
/// by 3–6 % for the longer `p₀ = 0.5`, `p₁ = 0.6`, `α = β = 0.05`, and by
/// 12–32 % for the short `p₀ = 0.9`, `p₁ = 0.5`, `α = β = 0.1`.
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
    let inc = BernoulliIncrements::checked(OP, p, p0, p1)?;
    check_rates(OP, alpha, beta)?;
    let bounds = wald_boundaries(alpha, beta)?;
    Ok(wald_approx(OP, p, inc, bounds.lower, bounds.upper)?.asn)
}
