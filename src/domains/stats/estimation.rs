//! Parameter estimation from observed data: maximum likelihood, the method
//! of moments, Bayesian conjugate updating — every point estimate handed
//! back as a [`Distribution`] so it plugs into the rest of `stats` — and
//! the confidence / credible intervals for a parameter (a mean with known
//! or unknown `σ`, a proportion, a correlation, a posterior).
//!
//! **Rule:** a function lives here iff it *estimates* a parameter — a point
//! estimate or an interval for it.  Tests belong to [`super::hypothesis`];
//! the intervals for a proportion, the t interval for a mean and Fisher's
//! z interval for a correlation moved in from `aggregation`, `hypothesis`
//! and `reliability` in 0.18.
//!
//! Observations are exact rationals ([`Q`]).  Every estimate that is a
//! rational function of the data (means, rates, probabilities, moment
//! estimators) is exact; the few that involve a root or a logarithm (the
//! normal `σ̂`, the log-normal parameters) are exact expressions ([`Ex`]).
//! Interval limits that need a numeric quantile are `f64`; the `_exact` /
//! `_symbolic` proportion intervals keep them as expressions.
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
//! * Counts (`n`, successes, failures, category counts) are `usize`, as
//!   everywhere in `stats`; a `ctx: &Context` is a parameter exactly when
//!   the result contains an [`Ex`] (a `Distribution`, a symbolic interval),
//!   so the `f64` intervals take none.

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::errors::SymplexError;
use crate::base::interval::Interval;

use super::common::{
    check_confidence, check_sample, ex_usize, invalid, q_to_f64, qu, standard_normal, z_two_sided,
};
use super::data::{self, Ddof, Q};
use super::family::{Distribution, fresh_symbol};
use super::numdist;

/// `data` non-empty, for the function `op`.
fn nonempty(op: &'static str, data: &[Q]) -> Result<(), SymplexError> {
    if data.is_empty() {
        return Err(invalid(op, format!("{op} needs at least one observation")));
    }
    Ok(())
}

fn require_positive_data(op: &'static str, data: &[Q]) -> Result<(), SymplexError> {
    nonempty(op, data)?;
    if let Some(x) = data.iter().find(|x| !x.is_positive()) {
        return Err(invalid(
            op,
            format!("{op} needs positive observations, got {x}"),
        ));
    }
    Ok(())
}

fn require_counts(op: &'static str, data: &[Q]) -> Result<(), SymplexError> {
    nonempty(op, data)?;
    if let Some(x) = data.iter().find(|x| !x.is_integer() || x.is_negative()) {
        return Err(invalid(
            op,
            format!("{op} needs non-negative integer observations, got {x}"),
        ));
    }
    Ok(())
}

fn require_positive_param(op: &'static str, x: &Q, what: &str) -> Result<(), SymplexError> {
    if !x.is_positive() {
        return Err(invalid(op, format!("{what} must be positive, got {x}")));
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
        n: usize,
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
        FamilyKind::Gamma | FamilyKind::Beta | FamilyKind::NegativeBinomial => Err(invalid(
            "fit",
            format!(
                "the {family:?} maximum-likelihood estimate has no closed form; use method_of_moments"
            ),
        )),
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
    const OP: &str = "fit_normal";
    nonempty(OP, data)?;
    let mean = data::mean(data)?;
    let var = data::variance(data, Ddof::Population)?;
    if var.is_zero() {
        return Err(invalid(OP, "constant data give σ̂ = 0"));
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
    require_positive_data("fit_exponential", data)?;
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
    const OP: &str = "fit_poisson";
    require_counts(OP, data)?;
    let rate = data::mean(data)?;
    if rate.is_zero() {
        return Err(invalid(OP, "all-zero counts give λ̂ = 0"));
    }
    Distribution::try_poisson(ctx.from_ratio(rate))
}

/// MLE of `Bernoulli(p)`: `p̂ = x̄`, the fraction of ones, exact.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless every observation is `0` or `1`.
pub fn fit_bernoulli(ctx: &Context, data: &[Q]) -> Result<Distribution, SymplexError> {
    const OP: &str = "fit_bernoulli";
    nonempty(OP, data)?;
    if let Some(x) = data.iter().find(|x| !x.is_zero() && !x.is_one()) {
        return Err(invalid(
            OP,
            format!("fit_bernoulli needs observations in {{0, 1}}, got {x}"),
        ));
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
pub fn fit_binomial_p(ctx: &Context, n: usize, data: &[Q]) -> Result<Distribution, SymplexError> {
    const OP: &str = "fit_binomial_p";
    if n == 0 {
        return Err(invalid(OP, "fit_binomial_p needs at least one trial"));
    }
    require_counts(OP, data)?;
    let nq = qu(n);
    if let Some(x) = data.iter().find(|x| **x > nq) {
        return Err(invalid(
            OP,
            format!("observation {x} exceeds the number of trials {n}"),
        ));
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
    const OP: &str = "fit_geometric";
    require_counts(OP, data)?;
    if let Some(x) = data.iter().find(|x| x.is_zero()) {
        return Err(invalid(
            OP,
            format!(
                "fit_geometric needs observations ≥ 1 (trials up to the first success), got {x}"
            ),
        ));
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
    const OP: &str = "fit_uniform";
    nonempty(OP, data)?;
    let range = data::min_max(data)?;
    if range.lower == range.upper {
        return Err(invalid(OP, "constant data give a zero-width interval"));
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
    const OP: &str = "fit_log_normal";
    require_positive_data(OP, data)?;
    let range = data::min_max(data)?;
    if range.lower == range.upper {
        return Err(invalid(OP, "constant data give σ̂ = 0"));
    }
    let n = ex_usize(ctx, data.len());
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
fn two_moments(op: &'static str, data: &[Q]) -> Result<(Q, Q), SymplexError> {
    nonempty(op, data)?;
    let m = data::mean(data)?;
    let v = data::variance(data, Ddof::Population)?;
    if v.is_zero() {
        return Err(invalid(
            op,
            "constant data have zero variance; the moment equations are degenerate",
        ));
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
    const OP: &str = "fit_gamma_moments";
    require_positive_data(OP, data)?;
    let (m, v) = two_moments(OP, data)?;
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
    const OP: &str = "fit_beta_moments";
    nonempty(OP, data)?;
    if let Some(x) = data.iter().find(|x| !x.is_positive() || **x >= Q::one()) {
        return Err(invalid(
            OP,
            format!("fit_beta_moments needs observations in (0, 1), got {x}"),
        ));
    }
    let (m, v) = two_moments(OP, data)?;
    let one_minus = Q::one() - &m;
    let c = &m * &one_minus / &v - Q::one();
    if !c.is_positive() {
        return Err(invalid(
            OP,
            "the variance is too large for a Beta law (x̄(1 − x̄)/s² ≤ 1)",
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
    const OP: &str = "fit_negative_binomial_moments";
    require_counts(OP, data)?;
    let (m, v) = two_moments(OP, data)?;
    if !m.is_positive() {
        return Err(invalid(OP, "all-zero counts"));
    }
    if v <= m {
        return Err(invalid(
            OP,
            format!(
                "fit_negative_binomial_moments needs over-dispersed counts (s² > x̄), got s² = {v}, x̄ = {m}"
            ),
        ));
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
    let (m, v) = two_moments("fit_uniform_moments", data)?;
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
    const OP: &str = "fit_log_normal_moments";
    require_positive_data(OP, data)?;
    let (m, v) = two_moments(OP, data)?;
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
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::estimation::aic;
///
/// let ctx = Context::new();
/// assert_eq!(aic(&ctx.int(-3), 2), ctx.int(10));
/// // 2 (2⁶⁴ − 1) + 6; 0.28 formed `2 * k as i64`, -2 for usize::MAX, and said 4.
/// assert_eq!(aic(&ctx.int(-3), usize::MAX).to_string(), "36893488147419103236");
/// ```
pub fn aic(log_lik: &Ex, k: usize) -> Ex {
    let ctx = log_lik.context();
    (ctx.int(2) * ex_usize(&ctx, k) - ctx.int(2) * log_lik).simplify()
}

/// The Bayesian information criterion `BIC = k ln n − 2ℓ` for `k` free
/// parameters, `n` observations and maximised log-likelihood `ℓ`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::estimation::bic;
///
/// let ctx = Context::new();
/// // 0.28 took `n as i64` (-1 for usize::MAX): `iπ + 6`.
/// let b = bic(&ctx.int(-3), 1, usize::MAX);
/// assert!(b.free_symbols().is_empty() && (b.eval_f64()? - (6.0 + (usize::MAX as f64).ln())).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
pub fn bic(log_lik: &Ex, k: usize, n: usize) -> Ex {
    let ctx = log_lik.context();
    (ex_usize(&ctx, k) * ex_usize(&ctx, n).ln() - ctx.int(2) * log_lik).simplify()
}

// ═══════════════════════════════════════════════════════════════════════════
// Bayesian conjugate updating
// ═══════════════════════════════════════════════════════════════════════════

/// The Beta posterior of a Bernoulli/Binomial success probability:
/// `Beta(prior_alpha, prior_beta)` prior with `successes` and `failures`
/// observed gives `Beta(prior_alpha + successes, prior_beta + failures)`
/// (definitional; exact).
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless `prior_alpha, prior_beta > 0`.
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
    prior_alpha: &Q,
    prior_beta: &Q,
    successes: usize,
    failures: usize,
) -> Result<Distribution, SymplexError> {
    const OP: &str = "beta_binomial_posterior";
    require_positive_param(OP, prior_alpha, "the prior α")?;
    require_positive_param(OP, prior_beta, "the prior β")?;
    Distribution::try_beta(
        ctx.from_ratio(prior_alpha + qu(successes)),
        ctx.from_ratio(prior_beta + qu(failures)),
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
    counts: &[usize],
) -> Result<Distribution, SymplexError> {
    const OP: &str = "gamma_poisson_posterior";
    require_positive_param(OP, shape, "the prior shape")?;
    require_positive_param(OP, scale, "the prior scale")?;
    let total: Q = counts.iter().fold(Q::zero(), |acc, &c| acc + qu(c));
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
    const OP: &str = "normal_known_variance_posterior";
    let (mu0, sigma0) = (prior_mean, prior_sd);
    require_positive_param(OP, sigma0, "the prior standard deviation")?;
    require_positive_param(OP, sigma, "the observation standard deviation")?;
    nonempty(OP, data)?;
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
pub fn dirichlet_posterior_alphas(prior: &[Q], counts: &[usize]) -> Result<Vec<Q>, SymplexError> {
    const OP: &str = "dirichlet_posterior_alphas";
    if prior.is_empty() {
        return Err(invalid(OP, "a Dirichlet prior needs at least one category"));
    }
    if prior.len() != counts.len() {
        return Err(invalid(
            OP,
            format!(
                "the Dirichlet prior has {} categories but {} counts were given",
                prior.len(),
                counts.len()
            ),
        ));
    }
    for a in prior {
        require_positive_param(OP, a, "every prior α")?;
    }
    Ok(prior.iter().zip(counts).map(|(a, &c)| a + qu(c)).collect())
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
    counts: &[usize],
) -> Result<Vec<Q>, SymplexError> {
    let alphas = dirichlet_posterior_alphas(prior, counts)?;
    let total = data::sum(&alphas);
    Ok(alphas.into_iter().map(|a| a / &total).collect())
}

/// The equal-tailed credible interval of `dist` at level `confidence`:
/// with `t = (1 − c)/2`, the lower end is `F⁻¹(t)`
/// ([`Distribution::quantile_f64`], `scipy.stats.<dist>.ppf(t)`) and the
/// upper end the `x` with `P(X > x) = t` (`scipy.stats.<dist>.isf(t)`).
///
/// The upper end is solved on the upper tail itself
/// ([`Distribution::isf_f64`]), never as the quantile at the rounded
/// level `1 − t`: that level carries an absolute error of up to `2⁻⁵⁴`,
/// a relative error of `1.1·10⁻⁴` in `t` at `c = 1 − 10⁻¹²`, and at
/// `c = 1 − 2⁻⁵³` it rounds to `1` (0.28 then returned an error; at
/// `1 − 10⁻¹²` its `Beta(9, 6)` upper end was `0.99764814506449` for
/// `0.9976481886981728`).  For a discrete law the upper end is the
/// smallest atom `k` with `P(X > k) ≤ t` (scipy's `isf` convention); a
/// [`Finite`](super::Finite) table, a die, a geometric law at
/// `c = 1 − 2⁻⁵³` were an error until 0.29.
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
/// // c = 1 - 1e-12; scipy: stats.beta.isf((1 - c)/2, 9, 6) = 0.9976481886981728
/// let ci = credible_interval(&Distribution::beta(ctx.int(9), ctx.int(6)), 1.0 - 1e-12)?;
/// assert!((ci.upper - 0.9976481886981728).abs() < 1e-15);
/// // c = 1 - 2^-53: t = 2^-54; a fair die has P(X > 5) = 1/6 > t >= P(X > 6) = 0
/// let ci = credible_interval(&Distribution::die(ctx.int(6)), 1.0 - f64::EPSILON / 2.0)?;
/// assert_eq!((ci.lower, ci.upper), (1.0, 6.0));
/// # Ok::<(), SymplexError>(())
/// ```
pub fn credible_interval(
    dist: &Distribution,
    confidence: f64,
) -> Result<Interval<f64>, SymplexError> {
    check_confidence("credible_interval", confidence)?;
    let tail = (1.0 - confidence) / 2.0;
    Ok(Interval::closed(
        dist.quantile_f64(tail)?,
        dist.isf_f64(tail)?,
    ))
}

/// The posterior predictive of `n` further Bernoulli trials under a
/// `Beta(prior_alpha, prior_beta)` posterior — the Beta-Binomial law
/// `P(K = k) = C(n, k) B(k + α, n − k + β) / B(α, β)` on `0..=n`, as an
/// exact [`Finite`](super::Finite) table (the Beta ratios reduce to
/// rising factorials `α⁽ᵏ⁾ β⁽ⁿ⁻ᵏ⁾ / (α + β)⁽ⁿ⁾`, so rational `α, β` give
/// rational masses).  `scipy.stats.betabinom.pmf(k, n, α, β)`.
///
/// The masses follow from `P(0) = β⁽ⁿ⁾ / (α + β)⁽ⁿ⁾` by the ratio
/// `P(k+1)/P(k) = (n − k)(α + k) / ((k + 1)(β + n − k − 1))`, `O(n)`
/// exact operations; the table (the distinct values `0..=n`, masses
/// summing to exactly `1`) is valid by construction and is not re-checked.
/// (0.28 formed three rising factorials per mass and then compared every
/// pair of values: 22 s at `n = 300`.)
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless `prior_alpha, prior_beta > 0`.
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
    prior_alpha: &Q,
    prior_beta: &Q,
    n: usize,
) -> Result<Distribution, SymplexError> {
    const OP: &str = "posterior_predictive_beta_binomial";
    require_positive_param(OP, prior_alpha, "α")?;
    require_positive_param(OP, prior_beta, "β")?;
    let mut mass =
        rising_factorial(prior_beta, n) / rising_factorial(&(prior_alpha + prior_beta), n);
    let mut table: Vec<(Ex, Ex)> = Vec::new();
    for k in 0..n {
        // One small factor per step, so the big mass is reduced once.
        let ratio =
            qu(n - k) * (prior_alpha + qu(k)) / ((qu(k) + Q::one()) * (prior_beta + qu(n - k - 1)));
        let next = &mass * ratio;
        table.push((
            ex_usize(ctx, k),
            ctx.from_ratio(std::mem::replace(&mut mass, next)),
        ));
    }
    table.push((ex_usize(ctx, n), ctx.from_ratio(mass)));
    Ok(Distribution::finite(ctx, table))
}

/// `a⁽ᵐ⁾ = a (a + 1) ⋯ (a + m − 1)`, the rising factorial (`1` for `m = 0`):
/// with `a = p/q`, `Π (p + i q) / qᵐ`, reduced once.
fn rising_factorial(a: &Q, m: usize) -> Q {
    let (p, q) = (a.numer(), a.denom());
    let numer = (0..m).fold(BigInt::one(), |acc, i| acc * (p + q * BigInt::from(i)));
    Q::new(numer, num_traits::pow(q.clone(), m))
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
    const OP: &str = "standard_error_mean";
    require_positive_param(OP, sigma, "σ")?;
    if n == 0 {
        return Err(invalid(
            OP,
            "the standard error needs at least one observation",
        ));
    }
    Ok((ctx.from_ratio(sigma.clone()) / ex_usize(ctx, n).sqrt()).simplify())
}

/// The `confidence` interval for the mean with unknown `σ`,
/// `x̄ ± t_{(1+c)/2, n−1} · s/√n`, with the critical value solved on the
/// upper tail, `P(T > t) = (1 − c)/2` ([`numdist::t`]'s `isf`,
/// `scipy.stats.t.isf`).  `scipy.stats.t.interval(c, n−1, loc=mean,
/// scale=sem)` agrees except near `c = 1`, where it takes the upper end
/// from the quantile at the rounded level `1 − (1 − c)/2` (and so did 0.28
/// for both ends: at `c = 1 − 10⁻¹²` the half-width below was `1.1·10⁻⁵`
/// relatively too small, and `c = 1 − 2⁻⁵³` was an error).  A constant
/// sample gives the degenerate interval `[x̄, x̄]`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::estimation::confidence_interval_mean;
///
/// let x = from_i64(&[5, 7, 8, 9, 10, 12]);
/// // scipy: t.interval(0.95, 5, loc=mean(x), scale=sem(x)) = (5.9509296876164886, 11.049070312383511)
/// let ci = confidence_interval_mean(&x, 0.95)?;
/// assert!((ci.lower - 5.950_929_687_616_488_6).abs() < 1e-9);
/// assert!((ci.upper - 11.049_070_312_383_511).abs() < 1e-9);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for fewer than two observations or
/// `confidence ∉ (0, 1)`; the quantile's error if it fails to converge.
pub fn confidence_interval_mean(x: &[Q], confidence: f64) -> Result<Interval<f64>, SymplexError> {
    const OP: &str = "confidence_interval_mean";
    check_sample(OP, "the sample", x, 2)?;
    check_confidence(OP, confidence)?;
    let n = x.len();
    let mean = q_to_f64(&data::mean(x)?);
    let sem = q_to_f64(&(data::variance(x, Ddof::Sample)? / qu(n))).sqrt();
    let t = numdist::t::isf((1.0 - confidence) / 2.0, (n - 1) as f64).map_err(renamed(OP))?;
    Ok(Interval::closed(mean - t * sem, mean + t * sem))
}

/// The z confidence interval for a population mean with known `σ`:
/// `x̄ ∓ z_{1 − (1 − c)/2} · σ/√n` (`scipy.stats.norm.interval(c, loc=x̄,
/// scale=σ/√n)`).  The t-based interval for unknown `σ` is
/// [`confidence_interval_mean`].
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
/// // x̄ = 2, σ/√n = 2/√3; scipy: norm.interval(0.95, 2, 2/sqrt(3)) = (-0.2631714681523438, 4.263171468152343)
/// let ci = confidence_interval_mean_z(&[1, 2, 3].map(qi), &qi(2), 0.95)?;
/// assert!((ci.lower + 0.2631714681523438).abs() < 1e-9);
/// assert!((ci.upper - 4.263171468152343).abs() < 1e-9);
/// # Ok::<(), SymplexError>(())
/// ```
pub fn confidence_interval_mean_z(
    data: &[Q],
    sigma: &Q,
    confidence: f64,
) -> Result<Interval<f64>, SymplexError> {
    const OP: &str = "confidence_interval_mean_z";
    nonempty(OP, data)?;
    check_confidence(OP, confidence)?;
    require_positive_param(OP, sigma, "σ")?;
    let se = q_to_f64(&(sigma * sigma / qu(data.len()))).sqrt();
    let mean = q_to_f64(&data::mean(data)?);
    let z = z_two_sided(confidence);
    Ok(Interval::closed(mean - z * se, mean + z * se))
}

/// The z confidence interval for a population mean with known `σ` as an
/// exact expression in `z`: `x̄ ∓ z · σ/√n` with the sample mean and
/// `σ/√n` exact ([`standard_error_mean`]) and `z` any expression — a
/// symbol for the textbook formula, or the exact quantile of a level from
/// [`z_for_confidence`] (which is what [`confidence_interval_mean_z_exact`]
/// passes).
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless the data are non-empty and
/// `σ > 0`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::estimation::confidence_interval_mean_z_symbolic;
/// use symplex::linprog::qi;
///
/// let ctx = Context::new();
/// let z = ctx.symbol("z");
/// // x̄ = 2, σ/√n = 2/√3
/// let ci = confidence_interval_mean_z_symbolic(&ctx, &[1, 2, 3].map(qi), &qi(2), &z)?;
/// assert_eq!(format!("{}", ci.upper), "2/3*z*sqrt(3) + 2");
/// assert_eq!(format!("{}", (&ci.upper - &ci.lower).simplify()), "4/3*z*sqrt(3)");
/// # Ok::<(), SymplexError>(())
/// ```
pub fn confidence_interval_mean_z_symbolic(
    ctx: &Context,
    data: &[Q],
    sigma: &Q,
    z: &Ex,
) -> Result<Interval<Ex>, SymplexError> {
    nonempty("confidence_interval_mean_z_symbolic", data)?;
    let se = standard_error_mean(ctx, sigma, data.len())?;
    let mean = ctx.from_ratio(data::mean(data)?);
    let half = z * se;
    Ok(Interval::closed(&mean - &half, &mean + &half))
}

/// The z confidence interval for a population mean with known `σ` at the
/// rational level `confidence`, with exact endpoints:
/// [`confidence_interval_mean_z_symbolic`] at `z = √2 · erfinv(confidence)`
/// ([`z_for_confidence`]).  The `f64` [`confidence_interval_mean_z`] rounds
/// the same numbers.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless the data are non-empty,
/// `σ > 0` and `0 < confidence < 1`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::estimation::confidence_interval_mean_z_exact;
/// use symplex::linprog::{q, qi};
///
/// let ctx = Context::new();
/// // x̄ = 2, σ/√n = 2/√3; scipy: norm.interval(0.95, 2, 2/sqrt(3)) = (-0.2631714681523438, 4.263171468152343)
/// let ci = confidence_interval_mean_z_exact(&ctx, &[1, 2, 3].map(qi), &qi(2), &q(95, 100))?;
/// assert!((ci.lower.eval_f64()? + 0.2631714681523438).abs() < 1e-12);
/// assert!((ci.upper.eval_f64()? - 4.263171468152343).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
pub fn confidence_interval_mean_z_exact(
    ctx: &Context,
    data: &[Q],
    sigma: &Q,
    confidence: &Q,
) -> Result<Interval<Ex>, SymplexError> {
    nonempty("confidence_interval_mean_z_exact", data)?;
    let z = z_for_confidence(ctx, confidence)?;
    confidence_interval_mean_z_symbolic(ctx, data, sigma, &z)
}

// ═══════════════════════════════════════════════════════════════════════════
// Confidence intervals for a proportion
// ═══════════════════════════════════════════════════════════════════════════

/// Confidence-interval methods for a binomial proportion
/// (`statsmodels.stats.proportion.proportion_confint(method=…)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntervalMethod {
    /// Wilson score interval (`'wilson'`).
    Wilson,
    /// Clopper–Pearson exact interval from Beta quantiles (`'beta'`).
    ClopperPearson,
    /// Agresti–Coull (`'agresti_coull'`).
    AgrestiCoull,
    /// Wald / normal approximation (`'normal'`).
    Wald,
    /// Jeffreys' equal-tailed Bayesian interval, the Beta quantiles of the
    /// posterior under the `Beta(½, ½)` prior (`'jeffreys'`).
    Jeffreys,
}

/// A [`numdist`] quantile's error renamed to the calling operation `op`.
fn renamed(op: &'static str) -> impl Fn(SymplexError) -> SymplexError {
    move |e| match e {
        SymplexError::InvalidArgument { reason, .. } => invalid(op, reason),
        SymplexError::ComputationFailed { reason, .. } => {
            SymplexError::computation_failed(op, reason)
        }
        other => other,
    }
}

fn check_trials(op: &'static str, successes: usize, trials: usize) -> Result<(), SymplexError> {
    if trials == 0 {
        return Err(invalid(op, "needs at least one trial"));
    }
    if successes > trials {
        return Err(invalid(op, "more successes than trials"));
    }
    Ok(())
}

/// `0 < confidence < 1` for a rational level.
fn check_confidence_q(op: &'static str, confidence: &Q) -> Result<(), SymplexError> {
    if !(confidence.is_positive() && *confidence < Q::one()) {
        return Err(invalid(
            op,
            format!("confidence must lie strictly between 0 and 1, got {confidence}"),
        ));
    }
    Ok(())
}

/// A two-sided confidence interval for the success probability behind
/// `successes` out of `trials`, at level `confidence` (e.g. `0.95`).
/// With `p̂ = k/n`, `α = 1 − confidence` and `z = Φ⁻¹(1 − α/2)`, computed
/// as the upper-tail quantile `Φ̄⁻¹(α/2)` (`scipy.stats.norm.isf`):
///
/// * Wald: `p̂ ± z √(p̂(1−p̂)/n)`;
/// * Wilson: `(p̂ + z²/2n ± z √(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n)`, the
///   two roots of `(p̂ − p)² = z² p(1 − p)/n`; the upper one is formed
///   as that sum and the lower one as `p̂² / ((1 + z²/n) · upper)` (the
///   product of the roots), since the difference cancels — to rounding
///   noise at `k = 0`, where the end is exactly `0`;
/// * Agresti–Coull: Wald around `p̃ = (k + z²/2)/(n + z²)` with `ñ = n + z²`;
/// * Clopper–Pearson (*Biometrika* 26 (1934) 404–413): the `p_L`, `p_U`
///   with `P(Bin(n, p_L) ≥ k) = α/2` and `P(Bin(n, p_U) ≤ k) = α/2`,
///   which are the Beta quantiles
///   `p_L = Beta(k, n−k+1)⁻¹(α/2)` and `p_U` with
///   `P(Beta(k+1, n−k) > p_U) = α/2` (`0` / `1` at `k = 0` / `k = n`),
///   through [`numdist::beta`]'s `ppf` and `isf` — statsmodels' `'beta'`
///   formulas, `O(1)` in `n`;
/// * Jeffreys (Brown, Cai & DasGupta, *Statist. Sci.* 16 (2001)
///   101–133): the quantiles `Beta(k+½, n−k+½)⁻¹(α/2)` and the `x` with
///   `P(Beta(k+½, n−k+½) > x) = α/2`, as statsmodels (no special case at
///   `k = 0` or `k = n`, where Brown, Cai & DasGupta put the end at `0` or
///   `1`).
///
/// Wald, Wilson and Agresti–Coull are clipped to `[0, 1]`, as in statsmodels.
///
/// 0.28 took `z` from the quantile at the rounded level `1 − α/2`
/// (relative error up to `2⁻⁵⁴/(α/2)` in the tail: the Wilson lower end of
/// `(5, 5)` at `1 − 10⁻¹⁵` was `0.071773` for `0.072013`, and
/// `c = 1 − 2⁻⁵³` was an error), and Clopper–Pearson by bisection on a
/// binomial sum over all `n + 1` terms per step (hours at `n = 10⁹`,
/// relative error `10⁻¹²` at `n = 10⁵` from the accumulated
/// `ln C(n, i)`).
///
/// ```
/// use symplex::stats::estimation::{IntervalMethod, proportion_interval};
///
/// // statsmodels: proportion_confint(3, 10, alpha=0.05, method='wilson')
/// //   = (0.10779126740630104, 0.6032218525388546)
/// let ci = proportion_interval(3, 10, 0.95, IntervalMethod::Wilson)?;
/// assert!((ci.lower - 0.10779126740630104).abs() < 1e-12);
/// assert!((ci.upper - 0.6032218525388546).abs() < 1e-12);
/// // statsmodels: proportion_confint(3, 10, alpha=0.05, method='jeffreys')
/// //   = (0.09269459393815319, 0.6058183181486713)
/// let ci = proportion_interval(3, 10, 0.95, IntervalMethod::Jeffreys)?;
/// assert!((ci.lower - 0.09269459393815319).abs() < 1e-15);
/// assert!((ci.upper - 0.6058183181486713).abs() < 1e-15);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for `trials = 0`,
/// `successes > trials` or a confidence outside `(0, 1)`.
pub fn proportion_interval(
    successes: usize,
    trials: usize,
    confidence: f64,
    method: IntervalMethod,
) -> Result<Interval<f64>, SymplexError> {
    const OP: &str = "proportion_interval";
    check_trials(OP, successes, trials)?;
    check_confidence(OP, confidence)?;
    // α/2: `1 − c` is exact for c ≥ ½ (Sterbenz), the halving always.
    let tail = (1.0 - confidence) / 2.0;
    // The failures `n − k`, as a double (`n − k + 1` overflows a usize).
    let (k, fails, n) = (successes as f64, (trials - successes) as f64, trials as f64);
    let beta_ppf = |a: f64, b: f64| numdist::beta::ppf(tail, a, b).map_err(renamed(OP));
    let beta_isf = |a: f64, b: f64| numdist::beta::isf(tail, a, b).map_err(renamed(OP));
    let z = || standard_normal().isf_f64(tail).map_err(renamed(OP));
    let p = k / n;
    let unit = Interval::closed(0.0, 1.0);
    let clip = |ci: Interval<f64>| ci.map(|v| unit.clamp_to_closure(v));
    Ok(match method {
        IntervalMethod::ClopperPearson => {
            let lo = if successes == 0 {
                0.0
            } else {
                beta_ppf(k, fails + 1.0)?
            };
            let hi = if successes == trials {
                1.0
            } else {
                beta_isf(k + 1.0, fails)?
            };
            Interval::closed(lo, hi)
        }
        IntervalMethod::Jeffreys => {
            let (a, b) = (k + 0.5, fails + 0.5);
            Interval::closed(beta_ppf(a, b)?, beta_isf(a, b)?)
        }
        IntervalMethod::Wald => {
            let half = z()? * (p * (1.0 - p) / n).sqrt();
            clip(Interval::closed(p - half, p + half))
        }
        IntervalMethod::Wilson => {
            let z = z()?;
            let z2 = z * z;
            let denom = 1.0 + z2 / n;
            let centre = (p + z2 / (2.0 * n)) / denom;
            let half = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denom;
            let upper = centre + half;
            clip(Interval::closed(p * p / (denom * upper), upper))
        }
        IntervalMethod::AgrestiCoull => {
            let z = z()?;
            let z2 = z * z;
            let n_t = n + z2;
            let p_t = (k + z2 / 2.0) / n_t;
            let half = z * (p_t * (1.0 - p_t) / n_t).sqrt();
            clip(Interval::closed(p_t - half, p_t + half))
        }
    })
}

/// The two-sided standard-normal quantile of a confidence level, exactly:
/// `z = Φ⁻¹(1 − α/2) = √2 · erfinv(confidence)` with `α = 1 − confidence`
/// (`scipy.stats.norm.ppf(1 - alpha/2)`).  This is the `z` of
/// [`proportion_interval_exact`] and [`confidence_interval_mean_z_exact`].
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::estimation::z_for_confidence;
/// use symplex::linprog::q;
///
/// let ctx = Context::new();
/// // scipy: norm.ppf(0.975) = 1.959963984540054
/// let z = z_for_confidence(&ctx, &q(95, 100))?;
/// assert_eq!(format!("{z}"), "sqrt(2)*erfinv(19/20)");
/// assert!((z.eval_f64()? - 1.959963984540054).abs() < 1e-12);
/// assert!(z_for_confidence(&ctx, &q(1, 1)).is_err());
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless `0 < confidence < 1`.
pub fn z_for_confidence(ctx: &Context, confidence: &Q) -> Result<Ex, SymplexError> {
    check_confidence_q("z_for_confidence", confidence)?;
    Ok(ctx.int(2).sqrt() * ctx.from_ratio(confidence.clone()).erfinv())
}

/// A confidence interval for the success probability behind `successes`
/// out of `trials` as an exact expression in `z`, the two-sided normal
/// quantile: the closed forms of [`proportion_interval`] with `p̂ = k/n`
/// exact and `z` any expression.  A symbol gives the textbook formula;
/// [`z_for_confidence`] gives the exact `z` of a level.  With `n` the
/// number of trials:
///
/// * Wald: `p̂ ± z √(p̂(1−p̂)/n)`;
/// * Wilson: `(p̂ + z²/2n ± z √(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n)`;
/// * Agresti–Coull: `p̃ ± z √(p̃(1−p̃)/ñ)` with `ñ = n + z²` and
///   `p̃ = (k + z²/2)/ñ`.
///
/// Unlike the `f64` function, the endpoints are **not** clipped to
/// `[0, 1]`: a Wald bound may fall outside it (`k = 1`, `n = 5` at 95 %
/// has a negative lower end), and for a symbolic `z` whether it does is
/// not decidable.  Clip the evaluated numbers yourself if statsmodels
/// parity is wanted.  Clopper–Pearson has no closed form in `z`; its exact
/// endpoints are in [`proportion_interval_exact`].  Jeffreys has none
/// either (its ends are Beta quantiles at half-integer shapes).  The lower
/// Wilson end is the difference `(centre − half)` here — exact, so
/// nothing cancels until it is evaluated.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::estimation::{
///     IntervalMethod, proportion_interval_symbolic, z_for_confidence,
/// };
/// use symplex::linprog::q;
///
/// let ctx = Context::new();
/// let z = ctx.symbol("z");
/// let ci = proportion_interval_symbolic(&ctx, 3, 10, &z, IntervalMethod::Wilson)?;
/// assert!(format!("{}", ci.upper).contains("z^2"));
///
/// // statsmodels: proportion_confint(3, 10, alpha=0.05, method='wilson')
/// //   = (0.10779126740630104, 0.6032218525388546)
/// let z95 = z_for_confidence(&ctx, &q(95, 100))?;
/// let ci = proportion_interval_symbolic(&ctx, 3, 10, &z95, IntervalMethod::Wilson)?;
/// assert!((ci.lower.eval_f64()? - 0.10779126740630104).abs() < 1e-12);
/// assert!((ci.upper.eval_f64()? - 0.6032218525388546).abs() < 1e-12);
/// assert!(proportion_interval_symbolic(&ctx, 3, 10, &z, IntervalMethod::ClopperPearson).is_err());
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for `trials = 0`,
/// `successes > trials` or `method` `ClopperPearson` or `Jeffreys`.
pub fn proportion_interval_symbolic(
    ctx: &Context,
    successes: usize,
    trials: usize,
    z: &Ex,
    method: IntervalMethod,
) -> Result<Interval<Ex>, SymplexError> {
    const OP: &str = "proportion_interval_symbolic";
    check_trials(OP, successes, trials)?;
    let n = ex_usize(ctx, trials);
    let p_hat = ctx.from_ratio(qu(successes) / qu(trials));
    let z2 = z.powi(2);
    // p̂(1 − p̂)/n
    let var = &p_hat * (ctx.one() - &p_hat) / &n;
    Ok(match method {
        IntervalMethod::Wald => {
            let half = z * var.sqrt();
            Interval::closed(&p_hat - &half, &p_hat + &half)
        }
        IntervalMethod::Wilson => {
            let denom = ctx.one() + &z2 / &n;
            let centre = (&p_hat + &z2 / (ctx.int(2) * &n)) / &denom;
            let half = z * (&var + &z2 / (ctx.int(4) * &n * &n)).sqrt() / &denom;
            Interval::closed(&centre - &half, &centre + &half)
        }
        IntervalMethod::AgrestiCoull => {
            let n_t = &n + &z2;
            let p_t = (ex_usize(ctx, successes) + &z2 / ctx.int(2)) / &n_t;
            let half = z * (&p_t * (ctx.one() - &p_t) / &n_t).sqrt();
            Interval::closed(&p_t - &half, &p_t + &half)
        }
        IntervalMethod::ClopperPearson => {
            return Err(invalid(
                OP,
                "Clopper–Pearson has no closed form in z; use proportion_interval_exact",
            ));
        }
        IntervalMethod::Jeffreys => {
            return Err(invalid(
                OP,
                "the Jeffreys interval has no closed form in z; use proportion_interval",
            ));
        }
    })
}

/// The `p ∈ (0, 1)` at which a binomial tail equals `half_alpha`, as an
/// exact algebraic number: the unique root in `(0, 1)` of the degree-`n`
/// polynomial `Σ_j C(n, j) pʲ (1 − p)ⁿ⁻ʲ − α/2`, summed over `j ≥ k`
/// (`upper_tail`, needs `k ≥ 1`) or `j ≤ k` (needs `k ≤ n − 1`).
///
/// The polynomial is built in a fresh symbol and expanded; Sturm's theorem
/// (`count_real_roots_in`) certifies that exactly one root lies in
/// `(0, 1)` — the tail is strictly monotone there — and counts the roots
/// below `0`, which is the root's index among the ascending real roots;
/// [`Ex::root_of`] then names it (a rational for a linear polynomial,
/// otherwise `RootOf` of the irreducible factor over ℤ).
fn binomial_tail_root(
    ctx: &Context,
    n: usize,
    k: usize,
    upper_tail: bool,
    half_alpha: &Q,
) -> Result<Ex, SymplexError> {
    let op = "proportion_interval_exact";
    let failed = |reason: String| SymplexError::computation_failed(op, reason);
    let p = fresh_symbol(ctx, "p", &[]);
    let one_minus_p = ctx.one() - &p;
    let range = if upper_tail { k..=n } else { 0..=k };
    let tail = range.fold(ctx.zero(), |acc, j| {
        acc + ctx.from_ratio(data::binomial_q(n, j))
            * p.powi(j as i64)
            * one_minus_p.powi((n - j) as i64)
    });
    let poly = (tail - ctx.from_ratio(half_alpha.clone())).expand();
    let (zero, one) = (ctx.zero(), ctx.one());
    // `[0, 1]` closed, but neither end is a root: the tail is 0 or 1 there
    // and 0 < α/2 < 1/2.
    let in_unit = poly
        .count_real_roots_in(&p, &zero, &one)
        .ok_or_else(|| failed("the binomial tail did not expand to a polynomial".into()))?;
    if in_unit != 1 {
        return Err(failed(format!(
            "expected exactly one root of the binomial tail in (0, 1) for n = {n}, k = {k}, found {in_unit}"
        )));
    }
    let below = poly
        .count_real_roots_in(&p, &ctx.neg_infinity(), &zero)
        .ok_or_else(|| failed("the binomial tail did not expand to a polynomial".into()))?;
    poly.root_of(&p, below).ok_or_else(|| {
        failed(format!(
            "could not name the root of the degree-{n} binomial tail polynomial \
             (its factorisation over ℤ was not certified or the RootOf index is unstable); \
             try a smaller number of trials"
        ))
    })
}

/// A two-sided confidence interval for the success probability behind
/// `successes` out of `trials` at the rational level `confidence`, with
/// exact endpoints (the `f64` [`proportion_interval`] rounds).  With
/// `α = 1 − confidence`:
///
/// * Wald, Wilson, Agresti–Coull: [`proportion_interval_symbolic`] at
///   `z = √2 · erfinv(confidence)` ([`z_for_confidence`]), an exact
///   expression — **not** clipped to `[0, 1]`;
/// * Clopper–Pearson: `p_L` is the root in `(0, 1)` of
///   `Σ_{j=k}^{n} C(n, j) pʲ (1 − p)ⁿ⁻ʲ − α/2` (`0` when `k = 0`) and `p_U`
///   the root in `(0, 1)` of `Σ_{j=0}^{k} C(n, j) pʲ (1 − p)ⁿ⁻ʲ − α/2`
///   (`1` when `k = n`) — the Beta quantiles `Beta(k, n−k+1)⁻¹(α/2)` and
///   `Beta(k+1, n−k)⁻¹(1−α/2)` as algebraic numbers.  Each is a rational
///   when its polynomial is linear (`n = 1`), otherwise a `RootOf` node
///   over the irreducible factor, which `Display`s as such and evaluates
///   to any precision with [`Ex::eval_decimal`].
///
/// The Clopper–Pearson endpoints cost a Sturm isolation and a
/// factorisation over ℤ of a degree-`n` polynomial: exact, but `O(n)`
/// degree root isolation with coefficients around `C(n, n/2) 2ⁿ`, so
/// tens of trials are the practical range (beyond `n ≈ 40` the crate may
/// not certify the factorisation and reports [`SymplexError::ComputationFailed`]).
/// For large `n` use the `f64` function.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::estimation::{IntervalMethod, proportion_interval_exact};
/// use symplex::linprog::q;
///
/// let ctx = Context::new();
/// // scipy: beta.ppf(0.025, 3, 8) = 0.06673951117773447,
/// //        beta.ppf(0.975, 4, 7) = 0.6524528500599973
/// let ci = proportion_interval_exact(&ctx, 3, 10, &q(95, 100), IntervalMethod::ClopperPearson)?;
/// assert!(format!("{}", ci.lower).starts_with("RootOf("));
/// assert!((ci.lower.eval_f64()? - 0.06673951117773447).abs() < 1e-12);
/// assert!((ci.upper.eval_f64()? - 0.6524528500599973).abs() < 1e-12);
/// // mpmath (50 dps): 0.066739511177734467114648056291648899
/// assert!(ci.lower.eval_decimal(30)?.starts_with("0.06673951117773446711464805629"));
///
/// // n = 1: the tail is linear and the endpoint is a rational.
/// let ci = proportion_interval_exact(&ctx, 1, 1, &q(9, 10), IntervalMethod::ClopperPearson)?;
/// assert_eq!(ci.lower, ctx.rational(1, 20));
/// assert_eq!(ci.upper, ctx.int(1));
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for `trials = 0`,
/// `successes > trials`, a confidence outside `(0, 1)` or
/// `method = Jeffreys` (its ends are Beta quantiles at half-integer
/// shapes, which are not algebraic numbers);
/// [`SymplexError::ComputationFailed`] if a Clopper–Pearson root cannot be
/// named (see above).
pub fn proportion_interval_exact(
    ctx: &Context,
    successes: usize,
    trials: usize,
    confidence: &Q,
    method: IntervalMethod,
) -> Result<Interval<Ex>, SymplexError> {
    const OP: &str = "proportion_interval_exact";
    check_trials(OP, successes, trials)?;
    check_confidence_q(OP, confidence)?;
    match method {
        IntervalMethod::ClopperPearson => {}
        IntervalMethod::Jeffreys => {
            return Err(invalid(
                OP,
                "the Jeffreys ends are Beta quantiles at half-integer shapes, not algebraic \
                 numbers; use proportion_interval",
            ));
        }
        IntervalMethod::Wald | IntervalMethod::Wilson | IntervalMethod::AgrestiCoull => {
            let z = z_for_confidence(ctx, confidence)?;
            return proportion_interval_symbolic(ctx, successes, trials, &z, method);
        }
    }
    let half_alpha = (Q::one() - confidence) / qu(2);
    let lower = if successes == 0 {
        ctx.zero()
    } else {
        binomial_tail_root(ctx, trials, successes, true, &half_alpha)?
    };
    let upper = if successes == trials {
        ctx.one()
    } else {
        binomial_tail_root(ctx, trials, successes, false, &half_alpha)?
    };
    Ok(Interval::closed(lower, upper))
}

// ═══════════════════════════════════════════════════════════════════════════
// Interval for a correlation
// ═══════════════════════════════════════════════════════════════════════════

/// Fisher's z-transform `z = atanh(r) = ½ ln((1 + r)/(1 − r))`, whose
/// sampling distribution is approximately normal with variance
/// `1/(n − 3)`.
///
/// ```
/// use symplex::prelude::*;
/// let ctx = Context::new();
/// // atanh(0.8) = 1.0986122886681098 (= ln 3)
/// let z = symplex::stats::estimation::fisher_z(&ctx.rational(4, 5));
/// assert!((z.eval_f64()? - 1.098_612_288_668_109_8).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
#[must_use]
pub fn fisher_z(r: &Ex) -> Ex {
    r.atanh()
}

/// A confidence interval for a population correlation by Fisher's z:
/// `tanh(atanh(r) ∓ z_{α/2} / √(n − 3))`.  `f64` throughout (a normal
/// quantile is involved).  `scipy.stats.pearsonr(x, y).confidence_interval
/// (confidence_level)`.
///
/// ```
/// use symplex::stats::estimation::pearson_ci;
///
/// // scipy: pearsonr(range(1, 6), [1, 3, 2, 5, 4]).confidence_interval(0.95)
/// //   → (-0.279640041969355, 0.9861961933012714); r = 0.8, n = 5
/// let ci = pearson_ci(0.8, 5, 0.95)?;
/// assert!((ci.lower + 0.279_640_041_969_355).abs() < 1e-12);
/// assert!((ci.upper - 0.986_196_193_301_271_4).abs() < 1e-12);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for `|r| ≥ 1` or non-finite `r`,
/// `n < 4`, or a confidence level outside `(0, 1)`.
pub fn pearson_ci(r: f64, n: usize, confidence: f64) -> Result<Interval<f64>, SymplexError> {
    const OP: &str = "pearson_ci";
    if !r.is_finite() || r.abs() >= 1.0 {
        return Err(invalid(
            OP,
            format!("the correlation must lie strictly between −1 and 1, got {r}"),
        ));
    }
    if n < 4 {
        return Err(invalid(
            OP,
            format!("Fisher's z interval needs at least four observations, got {n}"),
        ));
    }
    check_confidence(OP, confidence)?;
    let z = r.atanh();
    let se = 1.0 / ((n - 3) as f64).sqrt();
    let zc = z_two_sided(confidence);
    Ok(Interval::closed((z - zc * se).tanh(), (z + zc * se).tanh()))
}
