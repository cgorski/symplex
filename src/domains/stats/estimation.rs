//! Parameter estimation from observed data: maximum likelihood, the method
//! of moments, and Bayesian conjugate updating — every estimate handed back
//! as a [`Distribution`] so it plugs into the rest of `stats`.
//!
//! Observations are exact rationals ([`Q`]).  Every estimate that is a
//! rational function of the data (means, rates, probabilities, moment
//! estimators) is exact; the few that involve a root or a logarithm (the
//! normal `σ̂`, the log-normal parameters) are exact expressions ([`Ex`]).
//!
//! ```
//! use symplex::prelude::*;
//! use symplex::stats::Normal;
//! use symplex::stats::estimation::fit_normal;
//! use symplex::linprog::{q, qi};
//!
//! let ctx = Context::new();
//! // scipy: stats.norm.fit([2, 3.5, 4, 5.5, 7]) = (4.4, 1.7146428199482247); 1.7146…² = 2.94
//! let d = fit_normal(&ctx, &[qi(2), q(7, 2), qi(4), q(11, 2), qi(7)])?;
//! let n = d.downcast_ref::<Normal>().ok_or_else(|| SymplexError::computation_failed("fit", "not normal"))?;
//! assert_eq!(n.mean, ctx.rational(22, 5));
//! assert_eq!(n.std.powi(2).simplify(), ctx.rational(147, 50));
//! # Ok::<(), SymplexError>(())
//! ```
//!
//! # Conventions
//!
//! * The MLE of a normal variance is the **biased** `(1/n) Σ (xᵢ − x̄)²`
//!   (`scipy.stats.norm.fit` returns that `σ̂`, not the `n − 1` estimate).
//! * The method of moments matches the *population* moments of the data
//!   (`Ddof::Population`), the textbook definition.
//! * Discrete families follow SymPy's conventions: `Geometric` counts the
//!   trials up to and including the first success (support `1, 2, …`),
//!   `NegativeBinomial(r, p)` counts failures before the `r`-th success.
//! * `Gamma(k, θ)` is shape/scale; the Poisson conjugate prior is given as
//!   shape/scale too.

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::errors::SymplexError;
use crate::base::interval::Interval;

use super::data::{self, Ddof, Q};
use super::family::Distribution;

const OP: &str = "stats::estimation";

fn invalid(reason: impl Into<String>) -> SymplexError {
    SymplexError::invalid_argument(OP, reason)
}

fn qu(n: usize) -> Q {
    Q::from_integer(BigInt::from(n))
}

fn q64(n: u64) -> Q {
    Q::from_integer(BigInt::from(n))
}

fn nonempty(data: &[Q], what: &str) -> Result<(), SymplexError> {
    if data.is_empty() {
        return Err(invalid(format!("{what} needs at least one observation")));
    }
    Ok(())
}

fn require_positive_data(data: &[Q], what: &str) -> Result<(), SymplexError> {
    nonempty(data, what)?;
    if let Some(x) = data.iter().find(|x| !x.is_positive()) {
        return Err(invalid(format!(
            "{what} needs positive observations, got {x}"
        )));
    }
    Ok(())
}

fn require_counts(data: &[Q], what: &str) -> Result<(), SymplexError> {
    nonempty(data, what)?;
    if let Some(x) = data.iter().find(|x| !x.is_integer() || x.is_negative()) {
        return Err(invalid(format!(
            "{what} needs non-negative integer observations, got {x}"
        )));
    }
    Ok(())
}

fn require_positive_param(x: &Q, what: &str) -> Result<(), SymplexError> {
    if !x.is_positive() {
        return Err(invalid(format!("{what} must be positive, got {x}")));
    }
    Ok(())
}

/// `√v` as an exact expression (simplified, so `√(147/50)` is `7√6/10`).
fn sqrt_q(ctx: &Context, v: Q) -> Ex {
    ctx.from_ratio(v).sqrt().simplify()
}

/// `√v` for a numeric (parameter-free) expression `v ≥ 0`, with the
/// `√(u²) = |u|` that the simplifier produces resolved to `u` or `−u` by
/// the sign of `u`, which is decidable numerically (`√(ln² 2) = ln 2`).
fn positive_root(v: &Ex) -> Ex {
    let s = v.sqrt().simplify();
    if let [u] = s.args().as_slice()
        && s == u.abs()
        && let Ok(value) = u.eval_f64()
    {
        if value > 0.0 {
            return u.clone();
        }
        if value < 0.0 {
            return (-u).simplify();
        }
    }
    s
}

/// `ln x` for a positive rational, expanded over the prime factors of the
/// numerator and denominator (`ln 8 = 3 ln 2`, `ln(3/4) = ln 3 − 2 ln 2`),
/// so that sums of logarithms of rational data combine into one canonical
/// linear form in `ln p`.  Factoring is bounded; a large composite left
/// over is kept as a single `ln`.
fn ln_q(ctx: &Context, x: &Q) -> Ex {
    let side = |n: &BigInt, sign: i64| -> Ex {
        let (factors, cofactor) = crate::domains::ntheory::factorint_bounded(n, 64);
        let mut acc = ctx.zero();
        for (p, e) in factors {
            acc += ctx.int(sign * i64::from(e)) * ctx.from_bigint(p).ln();
        }
        if cofactor > BigInt::one() {
            acc += ctx.int(sign) * ctx.from_bigint(cofactor).ln();
        }
        acc
    };
    side(x.numer(), 1) + side(x.denom(), -1)
}

// ═══════════════════════════════════════════════════════════════════════════
// Families and the unified entry points
// ═══════════════════════════════════════════════════════════════════════════

/// The families [`fit`] and [`method_of_moments`] know how to estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FamilyKind {
    /// `Normal(μ, σ)`.
    Normal,
    /// `Exponential(λ)`.
    Exponential,
    /// `Poisson(λ)`.
    Poisson,
    /// `Bernoulli(p)`.
    Bernoulli,
    /// `Binomial(n, p)` with the number of trials `n` known.
    Binomial {
        /// The (known) number of trials of every observation.
        n: u64,
    },
    /// `Geometric(p)` on `1, 2, …`.
    Geometric,
    /// `Uniform(a, b)`.
    Uniform,
    /// `LogNormal(μ, σ)`.
    LogNormal,
    /// `Gamma(k, θ)` (shape, scale).  No closed-form MLE.
    Gamma,
    /// `Beta(α, β)`.  No closed-form MLE.
    Beta,
    /// `NegativeBinomial(r, p)`.  No closed-form MLE.
    NegativeBinomial,
}

/// The maximum-likelihood estimate of `family` from `data`, for the
/// families whose MLE has a closed form.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for `Gamma`, `Beta` and
/// `NegativeBinomial` (their likelihood equations involve the digamma
/// function; use [`method_of_moments`]), and whatever the family's `fit_*`
/// function rejects (empty data, observations outside the support, a
/// degenerate estimate such as `σ̂ = 0`).
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::Poisson;
/// use symplex::stats::estimation::{fit, FamilyKind};
/// use symplex::linprog::qi;
///
/// let ctx = Context::new();
/// // λ̂ = x̄ = 12/6 = 2 (exact arithmetic).
/// let d = fit(&ctx, FamilyKind::Poisson, &[0, 1, 1, 2, 3, 5].map(qi))?;
/// assert_eq!(d.downcast_ref::<Poisson>().map(|p| p.rate.clone()), Some(ctx.int(2)));
/// # Ok::<(), SymplexError>(())
/// ```
pub fn fit(ctx: &Context, family: FamilyKind, data: &[Q]) -> Result<Distribution, SymplexError> {
    match family {
        FamilyKind::Normal => fit_normal(ctx, data),
        FamilyKind::Exponential => fit_exponential(ctx, data),
        FamilyKind::Poisson => fit_poisson(ctx, data),
        FamilyKind::Bernoulli => fit_bernoulli(ctx, data),
        FamilyKind::Binomial { n } => fit_binomial_p(ctx, n, data),
        FamilyKind::Geometric => fit_geometric(ctx, data),
        FamilyKind::Uniform => fit_uniform(ctx, data),
        FamilyKind::LogNormal => fit_log_normal(ctx, data),
        FamilyKind::Gamma | FamilyKind::Beta | FamilyKind::NegativeBinomial => {
            Err(invalid(format!(
                "the {family:?} maximum-likelihood estimate has no closed form; use method_of_moments"
            )))
        }
    }
}

/// The method-of-moments estimate of `family` from `data`: the parameters
/// that make the family's first (and, for two-parameter families, second)
/// moment equal the data's population moments.  Coincides with the MLE for
/// `Normal`, `Exponential`, `Poisson`, `Bernoulli`, `Binomial` and
/// `Geometric`.
///
/// For `Uniform`: `a, b = x̄ ∓ √(3 s²)`; for `LogNormal`:
/// `σ² = ln(1 + s²/x̄²)`, `μ = ln x̄ − σ²/2` (both as exact expressions).
///
/// # Errors
///
/// As the family's `fit_*` function.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::Gamma;
/// use symplex::stats::estimation::{method_of_moments, FamilyKind};
/// use symplex::linprog::qi;
///
/// let ctx = Context::new();
/// // x̄ = 3, s² = 7/2  ⇒  k̂ = x̄²/s² = 18/7, θ̂ = s²/x̄ = 7/6 (Fraction arithmetic).
/// let d = method_of_moments(&ctx, FamilyKind::Gamma, &[1, 2, 3, 6].map(qi))?;
/// let g = d.downcast_ref::<Gamma>().ok_or_else(|| SymplexError::computation_failed("fit", "not gamma"))?;
/// assert_eq!((g.shape.clone(), g.scale.clone()), (ctx.rational(18, 7), ctx.rational(7, 6)));
/// # Ok::<(), SymplexError>(())
/// ```
pub fn method_of_moments(
    ctx: &Context,
    family: FamilyKind,
    data: &[Q],
) -> Result<Distribution, SymplexError> {
    match family {
        FamilyKind::Normal => fit_normal(ctx, data),
        FamilyKind::Exponential => fit_exponential(ctx, data),
        FamilyKind::Poisson => fit_poisson(ctx, data),
        FamilyKind::Bernoulli => fit_bernoulli(ctx, data),
        FamilyKind::Binomial { n } => fit_binomial_p(ctx, n, data),
        FamilyKind::Geometric => fit_geometric(ctx, data),
        FamilyKind::Uniform => fit_uniform_moments(ctx, data),
        FamilyKind::LogNormal => fit_log_normal_moments(ctx, data),
        FamilyKind::Gamma => fit_gamma_moments(ctx, data),
        FamilyKind::Beta => fit_beta_moments(ctx, data),
        FamilyKind::NegativeBinomial => fit_negative_binomial_moments(ctx, data),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Maximum likelihood, closed forms
// ═══════════════════════════════════════════════════════════════════════════

/// MLE of `Normal(μ, σ)`: `μ̂ = x̄` (exact) and `σ̂ = √((1/n) Σ (xᵢ − x̄)²)`,
/// the biased root — an exact expression.  `scipy.stats.norm.fit(data)`
/// returns `(μ̂, σ̂)`.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] on empty or constant data (`σ̂ = 0`).
pub fn fit_normal(ctx: &Context, data: &[Q]) -> Result<Distribution, SymplexError> {
    nonempty(data, "fit_normal")?;
    let mean = data::mean(data)?;
    let var = data::variance(data, Ddof::Population)?;
    if var.is_zero() {
        return Err(invalid("fit_normal: constant data give σ̂ = 0"));
    }
    Distribution::try_normal(ctx.from_ratio(mean), sqrt_q(ctx, var))
}

/// MLE of `Exponential(λ)`: `λ̂ = 1/x̄ = n / Σ xᵢ`, exact.
/// `scipy.stats.expon.fit(data, floc=0)` returns `(0, 1/λ̂)`.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless every observation is positive.
pub fn fit_exponential(ctx: &Context, data: &[Q]) -> Result<Distribution, SymplexError> {
    require_positive_data(data, "fit_exponential")?;
    let rate = data::mean(data)?.recip();
    Distribution::try_exponential(ctx.from_ratio(rate))
}

/// MLE of `Poisson(λ)`: `λ̂ = x̄`, exact.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless the observations are
/// non-negative integers with a positive sum.
pub fn fit_poisson(ctx: &Context, data: &[Q]) -> Result<Distribution, SymplexError> {
    require_counts(data, "fit_poisson")?;
    let rate = data::mean(data)?;
    if rate.is_zero() {
        return Err(invalid("fit_poisson: all-zero counts give λ̂ = 0"));
    }
    Distribution::try_poisson(ctx.from_ratio(rate))
}

/// MLE of `Bernoulli(p)`: `p̂ = x̄`, the fraction of ones, exact.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless every observation is `0` or `1`.
pub fn fit_bernoulli(ctx: &Context, data: &[Q]) -> Result<Distribution, SymplexError> {
    nonempty(data, "fit_bernoulli")?;
    if let Some(x) = data.iter().find(|x| !x.is_zero() && !x.is_one()) {
        return Err(invalid(format!(
            "fit_bernoulli needs observations in {{0, 1}}, got {x}"
        )));
    }
    Distribution::try_bernoulli(ctx.from_ratio(data::mean(data)?))
}

/// MLE of the success probability of `Binomial(n, p)` with `n` known:
/// `p̂ = Σ xᵢ / (n · N)` for `N` observations, exact.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if `n = 0` or an observation is not an
/// integer in `0..=n`.
pub fn fit_binomial_p(ctx: &Context, n: u64, data: &[Q]) -> Result<Distribution, SymplexError> {
    if n == 0 {
        return Err(invalid("fit_binomial_p needs at least one trial"));
    }
    require_counts(data, "fit_binomial_p")?;
    let nq = q64(n);
    if let Some(x) = data.iter().find(|x| **x > nq) {
        return Err(invalid(format!(
            "fit_binomial_p: observation {x} exceeds the number of trials {n}"
        )));
    }
    let p = data::mean(data)? / &nq;
    Distribution::try_binomial(ctx.from_ratio(nq), ctx.from_ratio(p))
}

/// MLE of `Geometric(p)` (trials up to the first success, support `1, 2, …`):
/// `p̂ = 1/x̄`, exact.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless every observation is a positive
/// integer.
pub fn fit_geometric(ctx: &Context, data: &[Q]) -> Result<Distribution, SymplexError> {
    require_counts(data, "fit_geometric")?;
    if let Some(x) = data.iter().find(|x| x.is_zero()) {
        return Err(invalid(format!(
            "fit_geometric needs observations ≥ 1 (trials up to the first success), got {x}"
        )));
    }
    Distribution::try_geometric(ctx.from_ratio(data::mean(data)?.recip()))
}

/// MLE of `Uniform(a, b)`: the sample minimum and maximum, exact.
/// `scipy.stats.uniform.fit(data)` returns `(min, max − min)`.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] on empty or constant data.
pub fn fit_uniform(ctx: &Context, data: &[Q]) -> Result<Distribution, SymplexError> {
    nonempty(data, "fit_uniform")?;
    let range = data::min_max(data)?;
    if range.lower == range.upper {
        return Err(invalid(
            "fit_uniform: constant data give a zero-width interval",
        ));
    }
    Distribution::try_uniform(ctx.from_ratio(range.lower), ctx.from_ratio(range.upper))
}

/// MLE of `LogNormal(μ, σ)`: the normal MLE of `ln xᵢ`, so
/// `μ̂ = (1/n) Σ ln xᵢ` and `σ̂² = (1/n) Σ (ln xᵢ − μ̂)²`, as exact
/// expressions linear (resp. quadratic) in the logarithms of the primes
/// dividing the data (`ln 8 = 3 ln 2`).  `scipy.stats.lognorm.fit(data,
/// floc=0)` returns `(σ̂, 0, e^{μ̂})`.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless every observation is positive
/// and they are not all equal.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::LogNormal;
/// use symplex::stats::estimation::fit_log_normal;
/// use symplex::linprog::qi;
///
/// let ctx = Context::new();
/// // ln [1, 2, 4, 8] = [0, 1, 2, 3]·ln 2  ⇒  μ̂ = (3/2) ln 2, σ̂² = (5/4) ln² 2
/// // scipy: lognorm.fit([1,2,4,8], floc=0) = (0.7749621070721792, 0, 2.82842712474619)
/// let d = fit_log_normal(&ctx, &[1, 2, 4, 8].map(qi))?;
/// let l = d.downcast_ref::<LogNormal>().ok_or_else(|| SymplexError::computation_failed("fit", "not lognormal"))?;
/// assert!((l.mu.eval_f64()? - 1.0397207708399179).abs() < 1e-12);
/// assert!((l.sigma.eval_f64()? - 0.7749621070721792).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
pub fn fit_log_normal(ctx: &Context, data: &[Q]) -> Result<Distribution, SymplexError> {
    require_positive_data(data, "fit_log_normal")?;
    let range = data::min_max(data)?;
    if range.lower == range.upper {
        return Err(invalid("fit_log_normal: constant data give σ̂ = 0"));
    }
    let n = ctx.int(data.len() as i64);
    let logs: Vec<Ex> = data.iter().map(|x| ln_q(ctx, x)).collect();
    let mu = (logs.iter().fold(ctx.zero(), |acc, l| acc + l) / &n).simplify();
    let var = (logs
        .iter()
        .fold(ctx.zero(), |acc, l| acc + (l - &mu).powi(2))
        / &n)
        .simplify();
    Distribution::try_log_normal(mu, positive_root(&var))
}

// ═══════════════════════════════════════════════════════════════════════════
// Method of moments
// ═══════════════════════════════════════════════════════════════════════════

/// Mean and population variance, the two moments every estimator below
/// matches.
fn two_moments(data: &[Q], what: &str) -> Result<(Q, Q), SymplexError> {
    nonempty(data, what)?;
    let m = data::mean(data)?;
    let v = data::variance(data, Ddof::Population)?;
    if v.is_zero() {
        return Err(invalid(format!(
            "{what}: constant data have zero variance; the moment equations are degenerate"
        )));
    }
    Ok((m, v))
}

/// Method of moments for `Gamma(k, θ)`: `k̂ = x̄²/s²`, `θ̂ = s²/x̄` with the
/// population variance `s²`, exact.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless the observations are positive
/// and not all equal.
pub fn fit_gamma_moments(ctx: &Context, data: &[Q]) -> Result<Distribution, SymplexError> {
    require_positive_data(data, "fit_gamma_moments")?;
    let (m, v) = two_moments(data, "fit_gamma_moments")?;
    let shape = &m * &m / &v;
    let scale = &v / &m;
    Distribution::try_gamma(ctx.from_ratio(shape), ctx.from_ratio(scale))
}

/// Method of moments for `Beta(α, β)` on data in `(0, 1)`: with
/// `c = x̄(1 − x̄)/s² − 1`, `α̂ = x̄·c` and `β̂ = (1 − x̄)·c`, exact.
/// (`c > 0` always holds for non-constant data strictly inside `(0, 1)`:
/// the Bhatia–Davis bound `s² ≤ x̄(1 − x̄)` is strict off `{0, 1}`.)
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless the observations lie strictly
/// in `(0, 1)` and are not all equal.
pub fn fit_beta_moments(ctx: &Context, data: &[Q]) -> Result<Distribution, SymplexError> {
    nonempty(data, "fit_beta_moments")?;
    if let Some(x) = data.iter().find(|x| !x.is_positive() || **x >= Q::one()) {
        return Err(invalid(format!(
            "fit_beta_moments needs observations in (0, 1), got {x}"
        )));
    }
    let (m, v) = two_moments(data, "fit_beta_moments")?;
    let one_minus = Q::one() - &m;
    let c = &m * &one_minus / &v - Q::one();
    if !c.is_positive() {
        return Err(invalid(
            "fit_beta_moments: the variance is too large for a Beta law (x̄(1 − x̄)/s² ≤ 1)",
        ));
    }
    Distribution::try_beta(ctx.from_ratio(&m * &c), ctx.from_ratio(one_minus * c))
}

/// Method of moments for `NegativeBinomial(r, p)` (failures before the
/// `r`-th success; mean `r(1−p)/p`, variance `r(1−p)/p²`):
/// `p̂ = x̄/s²`, `r̂ = x̄²/(s² − x̄)`, exact.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless the observations are
/// non-negative integers with `s² > x̄ > 0` (over-dispersion).
pub fn fit_negative_binomial_moments(
    ctx: &Context,
    data: &[Q],
) -> Result<Distribution, SymplexError> {
    require_counts(data, "fit_negative_binomial_moments")?;
    let (m, v) = two_moments(data, "fit_negative_binomial_moments")?;
    if !m.is_positive() {
        return Err(invalid("fit_negative_binomial_moments: all-zero counts"));
    }
    if v <= m {
        return Err(invalid(format!(
            "fit_negative_binomial_moments needs over-dispersed counts (s² > x̄), got s² = {v}, x̄ = {m}"
        )));
    }
    let p = &m / &v;
    let r = &m * &m / (&v - &m);
    Distribution::try_negative_binomial(ctx.from_ratio(r), ctx.from_ratio(p))
}

/// Method of moments for `Uniform(a, b)`: `a, b = x̄ ∓ √(3 s²)`, exact
/// expressions (the MLE is [`fit_uniform`]).
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] on empty or constant data.
pub fn fit_uniform_moments(ctx: &Context, data: &[Q]) -> Result<Distribution, SymplexError> {
    let (m, v) = two_moments(data, "fit_uniform_moments")?;
    let half_width = sqrt_q(ctx, v * qu(3));
    let m = ctx.from_ratio(m);
    Distribution::try_uniform((&m - &half_width).simplify(), (m + half_width).simplify())
}

/// Method of moments for `LogNormal(μ, σ)`: `σ² = ln(1 + s²/x̄²)`,
/// `μ = ln x̄ − σ²/2`, exact expressions (the MLE is [`fit_log_normal`]).
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless the observations are positive
/// and not all equal.
pub fn fit_log_normal_moments(ctx: &Context, data: &[Q]) -> Result<Distribution, SymplexError> {
    require_positive_data(data, "fit_log_normal_moments")?;
    let (m, v) = two_moments(data, "fit_log_normal_moments")?;
    let var = ctx.from_ratio(Q::one() + &v / (&m * &m)).ln();
    let mu = (ctx.from_ratio(m).ln() - &var / ctx.int(2)).simplify();
    Distribution::try_log_normal(mu, positive_root(&var))
}

// ═══════════════════════════════════════════════════════════════════════════
// Likelihood and information criteria
// ═══════════════════════════════════════════════════════════════════════════

/// The log-likelihood `ℓ = Σᵢ ln f(xᵢ)` of `data` under `dist`, with the
/// logarithms expanded and the sum simplified.  With numeric parameters
/// this is an exact expression (`scipy.stats.<dist>.logpdf(data).sum()`
/// numerically); with symbolic parameters it is the log-likelihood
/// *function*, ready for [`diff`](Ex::diff) and [`solve`](Ex::solve).
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::Distribution;
/// use symplex::stats::estimation::log_likelihood;
/// use symplex::linprog::qi;
///
/// let ctx = Context::new();
/// // scipy: stats.expon.logpdf([1, 2, 3, 6], scale=3).sum() = -8.39444915467244 = 4 ln(1/3) − 4
/// let l = log_likelihood(&Distribution::exponential(ctx.rational(1, 3)), &[1, 2, 3, 6].map(qi));
/// assert!((l.eval_f64()? + 8.39444915467244).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
pub fn log_likelihood(dist: &Distribution, data: &[Q]) -> Ex {
    let ctx = dist.context();
    data.iter()
        .fold(ctx.zero(), |acc, x| {
            acc + dist.density(&ctx.from_ratio(x.clone())).ln()
        })
        .expand_log()
        .simplify()
}

/// Akaike's information criterion `AIC = 2k − 2ℓ` for a model with `k`
/// free parameters and maximised log-likelihood `ℓ`.
pub fn aic(log_lik: &Ex, k: usize) -> Ex {
    let ctx = log_lik.context();
    (ctx.int(2 * k as i64) - ctx.int(2) * log_lik).simplify()
}

/// The Bayesian information criterion `BIC = k ln n − 2ℓ` for `k` free
/// parameters, `n` observations and maximised log-likelihood `ℓ`.
pub fn bic(log_lik: &Ex, k: usize, n: usize) -> Ex {
    let ctx = log_lik.context();
    (ctx.int(k as i64) * ctx.int(n as i64).ln() - ctx.int(2) * log_lik).simplify()
}

// ═══════════════════════════════════════════════════════════════════════════
// Bayesian conjugate updating
// ═══════════════════════════════════════════════════════════════════════════

/// The Beta posterior of a Bernoulli/Binomial success probability:
/// `Beta(alpha, beta)` prior with `successes` and `failures` observed
/// gives `Beta(alpha + successes, beta + failures)` (definitional; exact).
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless `alpha, beta > 0`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::Beta;
/// use symplex::stats::estimation::beta_binomial_posterior;
/// use symplex::linprog::qi;
///
/// let ctx = Context::new();
/// // Beta(2, 3) prior, 7 successes and 3 failures: Beta(9, 6).
/// let post = beta_binomial_posterior(&ctx, &qi(2), &qi(3), 7, 3)?;
/// let b = post.downcast_ref::<Beta>().ok_or_else(|| SymplexError::computation_failed("post", "not beta"))?;
/// assert_eq!((b.alpha.clone(), b.beta.clone()), (ctx.int(9), ctx.int(6)));
/// assert_eq!(post.mean(), ctx.rational(3, 5));
/// # Ok::<(), SymplexError>(())
/// ```
pub fn beta_binomial_posterior(
    ctx: &Context,
    alpha: &Q,
    beta: &Q,
    successes: u64,
    failures: u64,
) -> Result<Distribution, SymplexError> {
    require_positive_param(alpha, "the prior α")?;
    require_positive_param(beta, "the prior β")?;
    Distribution::try_beta(
        ctx.from_ratio(alpha + q64(successes)),
        ctx.from_ratio(beta + q64(failures)),
    )
}

/// The Gamma posterior of a Poisson rate, in **shape/scale** form (the
/// convention of [`Distribution::gamma`]): a `Gamma(shape, scale)` prior
/// with counts `x₁, …, xₙ` observed gives
/// `Gamma(shape + Σ xᵢ, scale / (1 + n·scale))` (exact).  The `scale` is
/// the *reciprocal* of the rate `β` of the `Gamma(α, β)` rate
/// parameterisation, in which the same update reads `β → β + n`; pass
/// `&rate.recip()` as `scale` if a prior is given that way.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::Gamma;
/// use symplex::stats::estimation::gamma_poisson_posterior;
/// use symplex::linprog::{q, qi};
///
/// let ctx = Context::new();
/// // Gamma(shape 2, scale 1/2) prior, counts [3, 5, 4]: shape 14, scale (1/2)/(1 + 3/2) = 1/5.
/// let post = gamma_poisson_posterior(&ctx, &qi(2), &q(1, 2), &[3, 5, 4])?;
/// let g = post.downcast_ref::<Gamma>().ok_or_else(|| SymplexError::computation_failed("post", "not gamma"))?;
/// assert_eq!((g.shape.clone(), g.scale.clone()), (ctx.int(14), ctx.rational(1, 5)));
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless `shape, scale > 0`.
pub fn gamma_poisson_posterior(
    ctx: &Context,
    shape: &Q,
    scale: &Q,
    counts: &[u64],
) -> Result<Distribution, SymplexError> {
    require_positive_param(shape, "the prior shape")?;
    require_positive_param(scale, "the prior scale")?;
    let total: Q = counts.iter().fold(Q::zero(), |acc, &c| acc + q64(c));
    let n = qu(counts.len());
    let post_scale = scale / (Q::one() + n * scale);
    Distribution::try_gamma(ctx.from_ratio(shape + total), ctx.from_ratio(post_scale))
}

/// The Normal posterior of a mean with the observation standard deviation
/// `sigma` known: the prior `Normal(prior_mean, prior_sd)` (`μ₀`, `σ₀` — a
/// standard deviation, not a variance) and data `x₁, …, xₙ` give precision
/// `τ = 1/σ₀² + n/σ²`, mean `(μ₀/σ₀² + Σ xᵢ/σ²) / τ` (exact) and standard
/// deviation `√(1/τ)` (an exact expression).
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::Normal;
/// use symplex::stats::estimation::normal_known_variance_posterior;
/// use symplex::linprog::qi;
///
/// let ctx = Context::new();
/// // Prior N(0, 1), σ = 2, data [1, 2, 3]: τ = 1 + 3/4 = 7/4, mean (6/4)/(7/4) = 6/7, variance 4/7.
/// let post = normal_known_variance_posterior(&ctx, &qi(0), &qi(1), &qi(2), &[1, 2, 3].map(qi))?;
/// let n = post.downcast_ref::<Normal>().ok_or_else(|| SymplexError::computation_failed("post", "not normal"))?;
/// assert_eq!(n.mean, ctx.rational(6, 7));
/// assert_eq!(n.std.powi(2).simplify(), ctx.rational(4, 7));
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless `prior_sd, sigma > 0` and the
/// data are non-empty.
pub fn normal_known_variance_posterior(
    ctx: &Context,
    prior_mean: &Q,
    prior_sd: &Q,
    sigma: &Q,
    data: &[Q],
) -> Result<Distribution, SymplexError> {
    let (mu0, sigma0) = (prior_mean, prior_sd);
    require_positive_param(sigma0, "the prior standard deviation")?;
    require_positive_param(sigma, "the observation standard deviation")?;
    nonempty(data, "normal_known_variance_posterior")?;
    let prior_prec = (sigma0 * sigma0).recip();
    let obs_prec = (sigma * sigma).recip();
    let n = qu(data.len());
    let precision = &prior_prec + &n * &obs_prec;
    let mean = (mu0 * &prior_prec + data::sum(data) * &obs_prec) / &precision;
    Distribution::try_normal(ctx.from_ratio(mean), sqrt_q(ctx, precision.recip()))
}

/// The Dirichlet posterior parameters of multinomial probabilities:
/// prior `Dirichlet(α₁, …, αₖ)` with category counts `c₁, …, cₖ` gives
/// `Dirichlet(α₁ + c₁, …, αₖ + cₖ)` (exact).
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if the lengths differ, `k = 0`, or an
/// `αᵢ ≤ 0`.
pub fn dirichlet_posterior_alphas(prior: &[Q], counts: &[u64]) -> Result<Vec<Q>, SymplexError> {
    if prior.is_empty() {
        return Err(invalid("a Dirichlet prior needs at least one category"));
    }
    if prior.len() != counts.len() {
        return Err(invalid(format!(
            "the Dirichlet prior has {} categories but {} counts were given",
            prior.len(),
            counts.len()
        )));
    }
    for a in prior {
        require_positive_param(a, "every prior α")?;
    }
    Ok(prior.iter().zip(counts).map(|(a, &c)| a + q64(c)).collect())
}

/// The posterior mean of multinomial probabilities under a Dirichlet
/// prior: `(αᵢ + cᵢ) / Σⱼ (αⱼ + cⱼ)`, exact.
///
/// # Errors
///
/// As [`dirichlet_posterior_alphas`].
///
/// ```
/// use symplex::stats::estimation::dirichlet_multinomial_posterior;
/// use symplex::linprog::{q, qi};
///
/// // Flat prior, counts (3, 2, 5): posterior mean (4/13, 3/13, 6/13).
/// let m = dirichlet_multinomial_posterior(&[qi(1), qi(1), qi(1)], &[3, 2, 5]).unwrap();
/// assert_eq!(m, vec![q(4, 13), q(3, 13), q(6, 13)]);
/// ```
pub fn dirichlet_multinomial_posterior(
    prior: &[Q],
    counts: &[u64],
) -> Result<Vec<Q>, SymplexError> {
    let alphas = dirichlet_posterior_alphas(prior, counts)?;
    let total = data::sum(&alphas);
    Ok(alphas.into_iter().map(|a| a / &total).collect())
}

/// The equal-tailed credible interval of `dist` at level `confidence`:
/// `[F⁻¹((1 − c)/2), F⁻¹(1 − (1 − c)/2)]` through
/// [`Distribution::quantile_f64`].  `scipy.stats.<dist>.ppf`.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless `0 < confidence < 1`; the
/// quantile's errors otherwise.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::Distribution;
/// use symplex::stats::estimation::credible_interval;
///
/// let ctx = Context::new();
/// // scipy: stats.beta.ppf([0.025, 0.975], 9, 6) = (0.3513801106159917, 0.8233889100178821)
/// let ci = credible_interval(&Distribution::beta(ctx.int(9), ctx.int(6)), 0.95)?;
/// assert!((ci.lower - 0.3513801106159917).abs() < 1e-9);
/// assert!((ci.upper - 0.8233889100178821).abs() < 1e-9);
/// # Ok::<(), SymplexError>(())
/// ```
pub fn credible_interval(
    dist: &Distribution,
    confidence: f64,
) -> Result<Interval<f64>, SymplexError> {
    if !(confidence > 0.0 && confidence < 1.0) {
        return Err(invalid(format!(
            "the credible level must lie strictly between 0 and 1, got {confidence}"
        )));
    }
    let tail = (1.0 - confidence) / 2.0;
    Ok(Interval::closed(
        dist.quantile_f64(tail)?,
        dist.quantile_f64(1.0 - tail)?,
    ))
}

/// The posterior predictive of `n` further Bernoulli trials under a
/// `Beta(α, β)` posterior — the Beta-Binomial law
/// `P(K = k) = C(n, k) B(k + α, n − k + β) / B(α, β)` on `0..=n`, as an
/// exact [`Finite`](super::Finite) table (the Beta ratios reduce to
/// rising factorials `α⁽ᵏ⁾ β⁽ⁿ⁻ᵏ⁾ / (α + β)⁽ⁿ⁾`, so rational `α, β` give
/// rational masses).  `scipy.stats.betabinom.pmf(k, n, α, β)`.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless `α, β > 0`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::estimation::posterior_predictive_beta_binomial;
/// use symplex::linprog::qi;
///
/// let ctx = Context::new();
/// // scipy: betabinom.pmf([0..4], 4, 2, 3) = [3/14, 2/7, 9/35, 6/35, 1/14]
/// let pred = posterior_predictive_beta_binomial(&ctx, &qi(2), &qi(3), 4)?;
/// assert_eq!(pred.density(&ctx.int(2)).simplify(), ctx.rational(9, 35));
/// assert_eq!(pred.mean(), ctx.rational(8, 5)); // n α / (α + β)
/// # Ok::<(), SymplexError>(())
/// ```
pub fn posterior_predictive_beta_binomial(
    ctx: &Context,
    alpha: &Q,
    beta: &Q,
    n: u64,
) -> Result<Distribution, SymplexError> {
    require_positive_param(alpha, "α")?;
    require_positive_param(beta, "β")?;
    let n_usize = usize::try_from(n)
        .map_err(|_| invalid(format!("the number of trials {n} is too large")))?;
    let denom = rising_factorial(&(alpha + beta), n_usize);
    let table: Vec<(Ex, Ex)> = (0..=n_usize)
        .map(|k| {
            let mass = data::binomial_q(n_usize, k)
                * rising_factorial(alpha, k)
                * rising_factorial(beta, n_usize - k)
                / &denom;
            (ctx.int(k as i64), ctx.from_ratio(mass))
        })
        .collect();
    Distribution::try_finite(ctx, table)
}

/// `a⁽ᵐ⁾ = a (a + 1) ⋯ (a + m − 1)`, the rising factorial (`1` for `m = 0`).
fn rising_factorial(a: &Q, m: usize) -> Q {
    (0..m).fold(Q::one(), |acc, i| acc * (a + qu(i)))
}

// ═══════════════════════════════════════════════════════════════════════════
// Sampling-distribution helpers (known σ)
// ═══════════════════════════════════════════════════════════════════════════

/// The standard error of a sample mean with known observation standard
/// deviation: `σ / √n`, as an exact expression.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless `σ > 0` and `n ≥ 1`.
pub fn standard_error_mean(ctx: &Context, sigma: &Q, n: usize) -> Result<Ex, SymplexError> {
    require_positive_param(sigma, "σ")?;
    if n == 0 {
        return Err(invalid("the standard error needs at least one observation"));
    }
    Ok((ctx.from_ratio(sigma.clone()) / ctx.int(n as i64).sqrt()).simplify())
}

/// The z confidence interval for a population mean with known `σ`:
/// `x̄ ∓ z_{1 − (1 − c)/2} · σ/√n` (`scipy.stats.norm.interval(c, loc=x̄,
/// scale=σ/√n)`).  The t-based interval for unknown `σ` lives in the
/// hypothesis-testing module.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless the data are non-empty,
/// `σ > 0` and `0 < confidence < 1`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::estimation::confidence_interval_mean_z;
/// use symplex::linprog::qi;
///
/// let ctx = Context::new();
/// // x̄ = 2, σ/√n = 2/√3; scipy: norm.interval(0.95, 2, 2/sqrt(3)) = (-0.2631714681523438, 4.263171468152343)
/// let ci = confidence_interval_mean_z(&ctx, &[1, 2, 3].map(qi), &qi(2), 0.95)?;
/// assert!((ci.lower + 0.2631714681523438).abs() < 1e-9);
/// assert!((ci.upper - 4.263171468152343).abs() < 1e-9);
/// # Ok::<(), SymplexError>(())
/// ```
pub fn confidence_interval_mean_z(
    ctx: &Context,
    data: &[Q],
    sigma: &Q,
    confidence: f64,
) -> Result<Interval<f64>, SymplexError> {
    nonempty(data, "confidence_interval_mean_z")?;
    if !(confidence > 0.0 && confidence < 1.0) {
        return Err(invalid(format!(
            "the confidence level must lie strictly between 0 and 1, got {confidence}"
        )));
    }
    let se = standard_error_mean(ctx, sigma, data.len())?.eval_f64()?;
    let mean = ctx.from_ratio(data::mean(data)?).eval_f64()?;
    let z =
        Distribution::normal(ctx.zero(), ctx.one()).quantile_f64(1.0 - (1.0 - confidence) / 2.0)?;
    Ok(Interval::closed(mean - z * se, mean + z * se))
}
