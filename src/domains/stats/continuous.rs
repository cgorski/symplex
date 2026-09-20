//! Continuous distribution families.  Each family stores its parameters as
//! expressions and provides support, density, and whatever closed forms it
//! has; the generic machinery in [`super::rv`] does the rest.
//!
//! To add a family: a variant of [`ContinuousFamily`] with documented
//! parameters, a constructor `Distribution::name(…)` that validates
//! numeric parameters (`Result`), and arms in each `match` below.  Every
//! closed form here is a textbook identity; cite it in the arm.
//!
//! Where a family returns `None` from a closed-form arm, the generic route
//! in [`RandomVariable`](super::RandomVariable) integrates the density
//! over the support instead; the per-arm docs say what that yields.

use std::fmt;

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::errors::SymplexError;

use super::rv::{Distribution, Support};

fn invalid(reason: impl Into<String>) -> SymplexError {
    SymplexError::InvalidArgument {
        operation: "stats",
        reason: reason.into(),
    }
}

/// Is a parameter known to be positive?  `None` for symbolic parameters
/// whose sign is not decided by assumptions.
fn known_positive(e: &Ex) -> Option<bool> {
    e.is_positive()
}

/// Reject a numeric parameter that is not positive; accept symbolic ones.
fn require_positive(e: &Ex, what: &str) -> Result<(), SymplexError> {
    match known_positive(e) {
        Some(false) => Err(invalid(format!("{what} must be positive, got `{e}`"))),
        _ => Ok(()),
    }
}

/// Reject a numeric parameter that is negative; accept symbolic ones.
fn require_nonnegative(e: &Ex, what: &str) -> Result<(), SymplexError> {
    match e.is_negative() {
        Some(true) => Err(invalid(format!("{what} must be non-negative, got `{e}`"))),
        _ => Ok(()),
    }
}

/// Does a moment that exists only for `param > bound` exist?  `false` only
/// when `param − bound` is *known* to be non-positive (a numeric parameter
/// that fails the condition); a symbolic parameter is the caller's promise
/// and the closed form is returned.
fn exceeds(param: &Ex, bound: u32) -> bool {
    (param - i64::from(bound)).is_positive() != Some(false)
}

/// `E[Xⁿ]` for a distribution symmetric about `mean` whose even central
/// moments are `central(2j)` (the odd ones vanish): the binomial expansion
/// `E[(μ + Y)ⁿ] = Σ_j C(n, 2j) μⁿ⁻²ʲ E[Y²ʲ]`.
fn raw_from_even_central(mean: &Ex, n: u32, ctx: &Context, central: impl Fn(u32) -> Ex) -> Ex {
    let mut acc = ctx.zero();
    for j in 0..=n / 2 {
        let binom = ctx.int(i64::from(n)).binomial(&ctx.int(i64::from(2 * j)));
        let c = if j == 0 { ctx.one() } else { central(2 * j) };
        acc += binom * mean.powi(i64::from(n - 2 * j)) * c;
    }
    acc.simplify()
}

/// `ChiSquared(k)` is `Gamma(k/2, 2)`; its arms delegate to this.
fn chi_squared_as_gamma(dof: &Ex) -> ContinuousFamily {
    let ctx = dof.context();
    ContinuousFamily::Gamma {
        shape: dof / ctx.int(2),
        scale: ctx.int(2),
    }
}

/// The continuous families.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ContinuousFamily {
    /// `Normal(μ, σ)`: density `e^{−(x−μ)²/(2σ²)} / (σ√(2π))` on ℝ.
    Normal {
        /// Mean `μ`.
        mean: Ex,
        /// Standard deviation `σ > 0`.
        std: Ex,
    },
    /// `Uniform(a, b)`: density `1/(b − a)` on `[a, b]`.
    Uniform {
        /// Lower end `a`.
        lo: Ex,
        /// Upper end `b > a`.
        hi: Ex,
    },
    /// `Exponential(λ)`: density `λ e^{−λx}` on `[0, ∞)`.
    Exponential {
        /// Rate `λ > 0` (the mean is `1/λ`).
        rate: Ex,
    },
    /// `Gamma(k, θ)`: density `x^{k−1} e^{−x/θ} / (Γ(k) θᵏ)` on `[0, ∞)`
    /// (shape–scale parametrisation, SymPy's `Gamma(k, theta)`).
    Gamma {
        /// Shape `k > 0`.
        shape: Ex,
        /// Scale `θ > 0` (the rate is `1/θ`).
        scale: Ex,
    },
    /// `ChiSquared(k)` = `Gamma(k/2, 2)`: density
    /// `x^{k/2−1} e^{−x/2} / (2^{k/2} Γ(k/2))` on `[0, ∞)`.
    ChiSquared {
        /// Degrees of freedom `k > 0`.
        dof: Ex,
    },
    /// `Beta(α, β)`: density `x^{α−1} (1−x)^{β−1} / B(α, β)` on `[0, 1]`.
    Beta {
        /// First shape `α > 0`.
        alpha: Ex,
        /// Second shape `β > 0`.
        beta: Ex,
    },
    /// `Cauchy(x₀, γ)`: density `1 / (πγ (1 + ((x−x₀)/γ)²))` on ℝ.  No
    /// moments of any order exist.
    Cauchy {
        /// Location `x₀` (the median).
        location: Ex,
        /// Scale `γ > 0` (half the interquartile range).
        scale: Ex,
    },
    /// `Laplace(μ, b)`: density `e^{−|x−μ|/b} / (2b)` on ℝ.
    Laplace {
        /// Location `μ` (mean and median).
        mean: Ex,
        /// Scale `b > 0`.
        scale: Ex,
    },
    /// `Logistic(μ, s)`: density `e^{−(x−μ)/s} / (s (1 + e^{−(x−μ)/s})²)`
    /// on ℝ.
    Logistic {
        /// Location `μ` (mean and median).
        mean: Ex,
        /// Scale `s > 0`.
        scale: Ex,
    },
    /// `LogNormal(μ, σ)`: `ln X ~ Normal(μ, σ)`; density
    /// `e^{−(ln x − μ)²/(2σ²)} / (xσ√(2π))` on `(0, ∞)`.
    LogNormal {
        /// Mean `μ` of `ln X`.
        mu: Ex,
        /// Standard deviation `σ > 0` of `ln X`.
        sigma: Ex,
    },
    /// `StudentT(ν)`: density
    /// `Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) · (1 + x²/ν)^{−(ν+1)/2}` on ℝ.
    StudentT {
        /// Degrees of freedom `ν > 0`.
        dof: Ex,
    },
    /// `Weibull(λ, k)`: density `(k/λ) (x/λ)^{k−1} e^{−(x/λ)ᵏ}` on
    /// `[0, ∞)`.  SymPy's `Weibull(alpha, beta)` has `alpha = λ` (scale)
    /// and `beta = k` (shape).
    Weibull {
        /// Scale `λ > 0`.
        scale: Ex,
        /// Shape `k > 0`.
        shape: Ex,
    },
    /// `Pareto(x_m, α)`: density `α x_mᵅ / x^{α+1}` on `[x_m, ∞)`.
    Pareto {
        /// Scale `x_m > 0` (the minimum).
        scale: Ex,
        /// Shape (tail index) `α > 0`.
        shape: Ex,
    },
    /// `Triangular(a, b, c)`: density rising linearly from `0` at `a` to
    /// `2/(b−a)` at the mode `c` and falling linearly to `0` at `b`, on
    /// `[a, b]` with `a ≤ c ≤ b`.
    Triangular {
        /// Lower end `a`.
        lo: Ex,
        /// Upper end `b > a`.
        hi: Ex,
        /// Mode `c ∈ [a, b]`.
        mode: Ex,
    },
}

impl Distribution {
    /// `Normal(μ, σ)` with mean `mean` and standard deviation `std`.
    /// SymPy: `Normal('X', mu, sigma)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `std` is a number `≤ 0`.
    pub fn try_normal(mean: Ex, std: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&std, "the standard deviation")?;
        Ok(Distribution::Continuous(ContinuousFamily::Normal {
            mean,
            std,
        }))
    }

    /// `Normal(μ, σ)`; a non-positive numeric `std` is a programming error
    /// and yields the distribution anyway (use [`try_normal`](Self::try_normal)
    /// to check).
    pub fn normal(mean: Ex, std: Ex) -> Distribution {
        Distribution::Continuous(ContinuousFamily::Normal { mean, std })
    }

    /// `Uniform(a, b)` on `[lo, hi]`.  SymPy: `Uniform('X', a, b)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `hi − lo` is a number `≤ 0`.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::Distribution;
    ///
    /// let ctx = Context::new();
    /// assert!(Distribution::try_uniform(ctx.int(0), ctx.int(1)).is_ok());
    /// assert!(Distribution::try_uniform(ctx.int(1), ctx.int(1)).is_err());
    /// ```
    pub fn try_uniform(lo: Ex, hi: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&(&hi - &lo), "the width `b − a`")?;
        Ok(Distribution::Continuous(ContinuousFamily::Uniform {
            lo,
            hi,
        }))
    }

    /// `Uniform(a, b)` without parameter validation (see
    /// [`try_uniform`](Self::try_uniform)).
    pub fn uniform(lo: Ex, hi: Ex) -> Distribution {
        Distribution::Continuous(ContinuousFamily::Uniform { lo, hi })
    }

    /// `Exponential(λ)` with rate `rate`.  SymPy: `Exponential('X', rate)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `rate` is a number `≤ 0`.
    pub fn try_exponential(rate: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&rate, "the rate")?;
        Ok(Distribution::Continuous(ContinuousFamily::Exponential {
            rate,
        }))
    }

    /// `Exponential(λ)` without parameter validation (see
    /// [`try_exponential`](Self::try_exponential)).
    pub fn exponential(rate: Ex) -> Distribution {
        Distribution::Continuous(ContinuousFamily::Exponential { rate })
    }

    /// `Gamma(k, θ)` with shape `shape` and scale `scale`.  SymPy:
    /// `Gamma('X', k, theta)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if either parameter is a number
    /// `≤ 0`.
    pub fn try_gamma(shape: Ex, scale: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&shape, "the shape")?;
        require_positive(&scale, "the scale")?;
        Ok(Distribution::Continuous(ContinuousFamily::Gamma {
            shape,
            scale,
        }))
    }

    /// `Gamma(k, θ)` without parameter validation (see
    /// [`try_gamma`](Self::try_gamma)).
    pub fn gamma(shape: Ex, scale: Ex) -> Distribution {
        Distribution::Continuous(ContinuousFamily::Gamma { shape, scale })
    }

    /// `ChiSquared(k)` with `dof` degrees of freedom.  SymPy:
    /// `ChiSquared('X', k)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `dof` is a number `≤ 0`.
    pub fn try_chi_squared(dof: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&dof, "the degrees of freedom")?;
        Ok(Distribution::Continuous(ContinuousFamily::ChiSquared {
            dof,
        }))
    }

    /// `ChiSquared(k)` without parameter validation (see
    /// [`try_chi_squared`](Self::try_chi_squared)).
    pub fn chi_squared(dof: Ex) -> Distribution {
        Distribution::Continuous(ContinuousFamily::ChiSquared { dof })
    }

    /// `Beta(α, β)`.  SymPy: `Beta('X', alpha, beta)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if either parameter is a number
    /// `≤ 0`.
    pub fn try_beta(alpha: Ex, beta: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&alpha, "the shape α")?;
        require_positive(&beta, "the shape β")?;
        Ok(Distribution::Continuous(ContinuousFamily::Beta {
            alpha,
            beta,
        }))
    }

    /// `Beta(α, β)` without parameter validation (see
    /// [`try_beta`](Self::try_beta)).
    pub fn beta(alpha: Ex, beta: Ex) -> Distribution {
        Distribution::Continuous(ContinuousFamily::Beta { alpha, beta })
    }

    /// `Cauchy(x₀, γ)` with location `location` and scale `scale`.  SymPy:
    /// `Cauchy('X', x0, gamma)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `scale` is a number `≤ 0`.
    pub fn try_cauchy(location: Ex, scale: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&scale, "the scale")?;
        Ok(Distribution::Continuous(ContinuousFamily::Cauchy {
            location,
            scale,
        }))
    }

    /// `Cauchy(x₀, γ)` without parameter validation (see
    /// [`try_cauchy`](Self::try_cauchy)).
    pub fn cauchy(location: Ex, scale: Ex) -> Distribution {
        Distribution::Continuous(ContinuousFamily::Cauchy { location, scale })
    }

    /// `Laplace(μ, b)` with location `mean` and scale `scale`.  SymPy:
    /// `Laplace('X', mu, b)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `scale` is a number `≤ 0`.
    pub fn try_laplace(mean: Ex, scale: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&scale, "the scale")?;
        Ok(Distribution::Continuous(ContinuousFamily::Laplace {
            mean,
            scale,
        }))
    }

    /// `Laplace(μ, b)` without parameter validation (see
    /// [`try_laplace`](Self::try_laplace)).
    pub fn laplace(mean: Ex, scale: Ex) -> Distribution {
        Distribution::Continuous(ContinuousFamily::Laplace { mean, scale })
    }

    /// `Logistic(μ, s)` with location `mean` and scale `scale`.  SymPy:
    /// `Logistic('X', mu, s)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `scale` is a number `≤ 0`.
    pub fn try_logistic(mean: Ex, scale: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&scale, "the scale")?;
        Ok(Distribution::Continuous(ContinuousFamily::Logistic {
            mean,
            scale,
        }))
    }

    /// `Logistic(μ, s)` without parameter validation (see
    /// [`try_logistic`](Self::try_logistic)).
    pub fn logistic(mean: Ex, scale: Ex) -> Distribution {
        Distribution::Continuous(ContinuousFamily::Logistic { mean, scale })
    }

    /// `LogNormal(μ, σ)`: `ln X ~ Normal(mu, sigma)`.  SymPy:
    /// `LogNormal('X', mu, sigma)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `sigma` is a number `≤ 0`.
    pub fn try_log_normal(mu: Ex, sigma: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&sigma, "the standard deviation of ln X")?;
        Ok(Distribution::Continuous(ContinuousFamily::LogNormal {
            mu,
            sigma,
        }))
    }

    /// `LogNormal(μ, σ)` without parameter validation (see
    /// [`try_log_normal`](Self::try_log_normal)).
    pub fn log_normal(mu: Ex, sigma: Ex) -> Distribution {
        Distribution::Continuous(ContinuousFamily::LogNormal { mu, sigma })
    }

    /// `StudentT(ν)` with `dof` degrees of freedom.  SymPy:
    /// `StudentT('X', nu)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `dof` is a number `≤ 0`.
    pub fn try_student_t(dof: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&dof, "the degrees of freedom")?;
        Ok(Distribution::Continuous(ContinuousFamily::StudentT { dof }))
    }

    /// `StudentT(ν)` without parameter validation (see
    /// [`try_student_t`](Self::try_student_t)).
    pub fn student_t(dof: Ex) -> Distribution {
        Distribution::Continuous(ContinuousFamily::StudentT { dof })
    }

    /// `Weibull(λ, k)` with scale `scale` and shape `shape`.  SymPy:
    /// `Weibull('X', alpha, beta)` with `alpha = scale`, `beta = shape`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if either parameter is a number
    /// `≤ 0`.
    pub fn try_weibull(scale: Ex, shape: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&scale, "the scale")?;
        require_positive(&shape, "the shape")?;
        Ok(Distribution::Continuous(ContinuousFamily::Weibull {
            scale,
            shape,
        }))
    }

    /// `Weibull(λ, k)` without parameter validation (see
    /// [`try_weibull`](Self::try_weibull)).
    pub fn weibull(scale: Ex, shape: Ex) -> Distribution {
        Distribution::Continuous(ContinuousFamily::Weibull { scale, shape })
    }

    /// `Pareto(x_m, α)` with minimum `scale` and tail index `shape`.
    /// SymPy: `Pareto('X', xm, alpha)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if either parameter is a number
    /// `≤ 0`.
    pub fn try_pareto(scale: Ex, shape: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&scale, "the scale x_m")?;
        require_positive(&shape, "the shape α")?;
        Ok(Distribution::Continuous(ContinuousFamily::Pareto {
            scale,
            shape,
        }))
    }

    /// `Pareto(x_m, α)` without parameter validation (see
    /// [`try_pareto`](Self::try_pareto)).
    pub fn pareto(scale: Ex, shape: Ex) -> Distribution {
        Distribution::Continuous(ContinuousFamily::Pareto { scale, shape })
    }

    /// `Triangular(a, b, c)` on `[lo, hi]` with mode `mode`.  SymPy:
    /// `Triangular('X', a, b, c)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the parameters are numbers
    /// with `hi ≤ lo` or `mode` outside `[lo, hi]`.
    pub fn try_triangular(lo: Ex, hi: Ex, mode: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&(&hi - &lo), "the width `b − a`")?;
        require_nonnegative(&(&mode - &lo), "`c − a` (the mode must lie in [a, b])")?;
        require_nonnegative(&(&hi - &mode), "`b − c` (the mode must lie in [a, b])")?;
        Ok(Distribution::Continuous(ContinuousFamily::Triangular {
            lo,
            hi,
            mode,
        }))
    }

    /// `Triangular(a, b, c)` without parameter validation (see
    /// [`try_triangular`](Self::try_triangular)).
    pub fn triangular(lo: Ex, hi: Ex, mode: Ex) -> Distribution {
        Distribution::Continuous(ContinuousFamily::Triangular { lo, hi, mode })
    }
}

impl ContinuousFamily {
    /// The family's name.
    pub fn name(&self) -> &'static str {
        match self {
            ContinuousFamily::Normal { .. } => "Normal",
            ContinuousFamily::Uniform { .. } => "Uniform",
            ContinuousFamily::Exponential { .. } => "Exponential",
            ContinuousFamily::Gamma { .. } => "Gamma",
            ContinuousFamily::ChiSquared { .. } => "ChiSquared",
            ContinuousFamily::Beta { .. } => "Beta",
            ContinuousFamily::Cauchy { .. } => "Cauchy",
            ContinuousFamily::Laplace { .. } => "Laplace",
            ContinuousFamily::Logistic { .. } => "Logistic",
            ContinuousFamily::LogNormal { .. } => "LogNormal",
            ContinuousFamily::StudentT { .. } => "StudentT",
            ContinuousFamily::Weibull { .. } => "Weibull",
            ContinuousFamily::Pareto { .. } => "Pareto",
            ContinuousFamily::Triangular { .. } => "Triangular",
        }
    }

    /// The support.
    pub fn support(&self) -> Support {
        let real_line = Support::Continuous { lo: None, hi: None };
        let half_line = |ctx: &Context| Support::Continuous {
            lo: Some(ctx.zero()),
            hi: None,
        };
        match self {
            ContinuousFamily::Normal { .. }
            | ContinuousFamily::Cauchy { .. }
            | ContinuousFamily::Laplace { .. }
            | ContinuousFamily::Logistic { .. }
            | ContinuousFamily::StudentT { .. } => real_line,
            ContinuousFamily::Exponential { rate } => half_line(&rate.context()),
            ContinuousFamily::Gamma { shape, .. } => half_line(&shape.context()),
            ContinuousFamily::ChiSquared { dof } => half_line(&dof.context()),
            ContinuousFamily::LogNormal { mu, .. } => half_line(&mu.context()),
            ContinuousFamily::Weibull { scale, .. } => half_line(&scale.context()),
            ContinuousFamily::Uniform { lo, hi } | ContinuousFamily::Triangular { lo, hi, .. } => {
                Support::Continuous {
                    lo: Some(lo.clone()),
                    hi: Some(hi.clone()),
                }
            }
            ContinuousFamily::Beta { alpha, .. } => {
                let ctx = alpha.context();
                Support::Continuous {
                    lo: Some(ctx.zero()),
                    hi: Some(ctx.one()),
                }
            }
            ContinuousFamily::Pareto { scale, .. } => Support::Continuous {
                lo: Some(scale.clone()),
                hi: None,
            },
        }
    }

    /// The density as an expression in `x` (valid on the support).
    pub fn density(&self, x: &Ex) -> Ex {
        let ctx = x.context();
        match self {
            ContinuousFamily::Normal { mean, std } => {
                let two = ctx.int(2);
                let z = (x - mean) / std;
                (-(z.powi(2)) / &two).exp() / (std * (&two * ctx.pi()).sqrt())
            }
            ContinuousFamily::Uniform { lo, hi } => ctx.one() / (hi - lo),
            ContinuousFamily::Exponential { rate } => rate * (-(rate * x)).exp(),
            ContinuousFamily::Gamma { shape, scale } => {
                x.pow(&(shape - 1)) * (-(x / scale)).exp() / (shape.gamma() * scale.pow(shape))
            }
            ContinuousFamily::ChiSquared { dof } => chi_squared_as_gamma(dof).density(x),
            ContinuousFamily::Beta { alpha, beta } => {
                x.pow(&(alpha - 1)) * (ctx.one() - x).pow(&(beta - 1)) / alpha.beta(beta)
            }
            ContinuousFamily::Cauchy { location, scale } => {
                ctx.one() / (ctx.pi() * scale * (ctx.one() + ((x - location) / scale).powi(2)))
            }
            ContinuousFamily::Laplace { mean, scale } => {
                (-((x - mean).abs() / scale)).exp() / (ctx.int(2) * scale)
            }
            ContinuousFamily::Logistic { mean, scale } => {
                let e = (-((x - mean) / scale)).exp();
                &e / (scale * (ctx.one() + &e).powi(2))
            }
            ContinuousFamily::LogNormal { mu, sigma } => {
                let two = ctx.int(2);
                (-((x.ln() - mu).powi(2)) / (&two * sigma.powi(2))).exp()
                    / (x * sigma * (&two * ctx.pi()).sqrt())
            }
            ContinuousFamily::StudentT { dof } => {
                let half = ctx.rational(1, 2);
                let nu1 = (dof + 1) * &half;
                nu1.gamma() / ((dof * ctx.pi()).sqrt() * (dof * &half).gamma())
                    * (ctx.one() + x.powi(2) / dof).pow(&(-nu1))
            }
            ContinuousFamily::Weibull { scale, shape } => {
                let z = x / scale;
                (shape / scale) * z.pow(&(shape - 1)) * (-(z.pow(shape))).exp()
            }
            ContinuousFamily::Pareto { scale, shape } => {
                shape * scale.pow(shape) / x.pow(&(shape + 1))
            }
            // Wikipedia, "Triangular distribution": 2(x−a)/((b−a)(c−a)) on
            // [a, c], 2(b−x)/((b−a)(b−c)) on (c, b].
            ContinuousFamily::Triangular { lo, hi, mode } => {
                let two = ctx.int(2);
                let width = hi - lo;
                let rising = &two * (x - lo) / (&width * (mode - lo));
                let falling = &two * (hi - x) / (&width * (hi - mode));
                Ex::piecewise(&[(&rising, &x.le(mode)), (&falling, &x.gt(mode))])
            }
        }
    }

    /// Closed-form mean.  `None` for `Cauchy` (no mean; the generic route
    /// returns the divergent integral unevaluated), and for `StudentT` with
    /// numeric `ν ≤ 1` / `Pareto` with numeric `α ≤ 1`, where it likewise
    /// does not exist.
    pub fn mean(&self, ctx: &Context) -> Option<Ex> {
        match self {
            ContinuousFamily::Normal { mean, .. } => Some(mean.clone()),
            ContinuousFamily::Uniform { lo, hi } => Some(((lo + hi) / ctx.int(2)).simplify()),
            ContinuousFamily::Exponential { rate } => Some(ctx.one() / rate),
            ContinuousFamily::Gamma { shape, scale } => Some((shape * scale).simplify()),
            ContinuousFamily::ChiSquared { dof } => Some(dof.clone()),
            ContinuousFamily::Beta { alpha, beta } => Some((alpha / (alpha + beta)).simplify()),
            ContinuousFamily::Cauchy { .. } => None,
            ContinuousFamily::Laplace { mean, .. } | ContinuousFamily::Logistic { mean, .. } => {
                Some(mean.clone())
            }
            // e^{μ + σ²/2}
            ContinuousFamily::LogNormal { mu, sigma } => {
                Some((mu + sigma.powi(2) / ctx.int(2)).exp())
            }
            // 0 for ν > 1.
            ContinuousFamily::StudentT { dof } => exceeds(dof, 1).then(|| ctx.zero()),
            // λ Γ(1 + 1/k)
            ContinuousFamily::Weibull { scale, shape } => {
                Some((scale * (ctx.one() + ctx.one() / shape).gamma()).simplify())
            }
            // α x_m / (α − 1) for α > 1.
            ContinuousFamily::Pareto { scale, shape } => {
                exceeds(shape, 1).then(|| (shape * scale / (shape - 1)).simplify())
            }
            ContinuousFamily::Triangular { lo, hi, mode } => {
                Some(((lo + hi + mode) / ctx.int(3)).simplify())
            }
        }
    }

    /// Closed-form variance.  `None` for `Cauchy`, and for `StudentT` with
    /// numeric `ν ≤ 2` / `Pareto` with numeric `α ≤ 2` (infinite variance).
    pub fn variance(&self, ctx: &Context) -> Option<Ex> {
        match self {
            ContinuousFamily::Normal { std, .. } => Some(std.powi(2)),
            // (b − a)² / 12
            ContinuousFamily::Uniform { lo, hi } => {
                Some(((hi - lo).powi(2) / ctx.int(12)).simplify())
            }
            ContinuousFamily::Exponential { rate } => Some(ctx.one() / rate.powi(2)),
            ContinuousFamily::Gamma { shape, scale } => Some((shape * scale.powi(2)).simplify()),
            ContinuousFamily::ChiSquared { dof } => Some((ctx.int(2) * dof).simplify()),
            // αβ / ((α+β)² (α+β+1))
            ContinuousFamily::Beta { alpha, beta } => {
                let s = alpha + beta;
                Some((alpha * beta / (s.powi(2) * (&s + 1))).simplify())
            }
            ContinuousFamily::Cauchy { .. } => None,
            ContinuousFamily::Laplace { scale, .. } => {
                Some((ctx.int(2) * scale.powi(2)).simplify())
            }
            // s²π²/3
            ContinuousFamily::Logistic { scale, .. } => {
                Some((scale.powi(2) * ctx.pi().powi(2) / ctx.int(3)).simplify())
            }
            // (e^{σ²} − 1) e^{2μ + σ²}
            ContinuousFamily::LogNormal { mu, sigma } => {
                let s2 = sigma.powi(2);
                Some(((s2.exp() - 1) * (ctx.int(2) * mu + &s2).exp()).simplify())
            }
            // ν / (ν − 2) for ν > 2.
            ContinuousFamily::StudentT { dof } => {
                exceeds(dof, 2).then(|| (dof / (dof - 2)).simplify())
            }
            // λ² [Γ(1 + 2/k) − Γ(1 + 1/k)²]
            ContinuousFamily::Weibull { scale, shape } => {
                let g1 = (ctx.one() + ctx.one() / shape).gamma();
                let g2 = (ctx.one() + ctx.int(2) / shape).gamma();
                Some((scale.powi(2) * (g2 - g1.powi(2))).simplify())
            }
            // x_m² α / ((α − 1)² (α − 2)) for α > 2.
            ContinuousFamily::Pareto { scale, shape } => exceeds(shape, 2)
                .then(|| (scale.powi(2) * shape / ((shape - 1).powi(2) * (shape - 2))).simplify()),
            // (a² + b² + c² − ab − ac − bc) / 18
            ContinuousFamily::Triangular { lo, hi, mode } => Some(
                ((lo.powi(2) + hi.powi(2) + mode.powi(2) - lo * hi - lo * mode - hi * mode)
                    / ctx.int(18))
                .simplify(),
            ),
        }
    }

    /// Closed-form raw moment `E[Xⁿ]`.  `None` for `Cauchy` (no moments),
    /// and for `StudentT` / `Pareto` when a numeric parameter says the
    /// moment does not exist (`n ≥ ν`, `n ≥ α`); the generic route then
    /// returns the divergent integral unevaluated.
    pub fn raw_moment(&self, n: u32, ctx: &Context) -> Option<Ex> {
        let n_ex = ctx.int(i64::from(n));
        match self {
            // E[Xⁿ] = Σ_{k=0}^{⌊n/2⌋} C(n, 2k) (2k−1)!! μⁿ⁻²ᵏ σ²ᵏ
            ContinuousFamily::Normal { mean, std } => {
                let mut acc = ctx.zero();
                for k in 0..=n / 2 {
                    let binom = ctx.int(i64::from(n)).binomial(&ctx.int(i64::from(2 * k)));
                    // (2k − 1)!! = (2k)! / (2ᵏ k!)
                    let dfact = ctx.int(i64::from(2 * k)).factorial()
                        / (ctx.int(2).powi(i64::from(k)) * ctx.int(i64::from(k)).factorial());
                    acc += binom
                        * dfact
                        * mean.powi(i64::from(n - 2 * k))
                        * std.powi(i64::from(2 * k));
                }
                Some(acc.simplify())
            }
            // (bⁿ⁺¹ − aⁿ⁺¹) / ((n+1)(b − a))
            ContinuousFamily::Uniform { lo, hi } => {
                let n1 = i64::from(n) + 1;
                Some(((hi.powi(n1) - lo.powi(n1)) / (ctx.int(n1) * (hi - lo))).simplify())
            }
            // n! / λⁿ
            ContinuousFamily::Exponential { rate } => {
                Some((n_ex.factorial() / rate.powi(i64::from(n))).simplify())
            }
            // θⁿ Γ(k+n)/Γ(k) = θⁿ (k)ₙ
            ContinuousFamily::Gamma { shape, scale } => {
                Some((scale.powi(i64::from(n)) * shape.rising_factorial(&n_ex)).simplify())
            }
            ContinuousFamily::ChiSquared { dof } => chi_squared_as_gamma(dof).raw_moment(n, ctx),
            // Π_{i<n} (α+i)/(α+β+i) = (α)ₙ / (α+β)ₙ
            ContinuousFamily::Beta { alpha, beta } => Some(
                (alpha.rising_factorial(&n_ex) / (alpha + beta).rising_factorial(&n_ex)).simplify(),
            ),
            ContinuousFamily::Cauchy { .. } => None,
            // Even central moments E[(X−μ)ᵏ] = k! bᵏ, odd ones 0.
            ContinuousFamily::Laplace { mean, scale } => {
                Some(raw_from_even_central(mean, n, ctx, |k| {
                    ctx.int(i64::from(k)).factorial() * scale.powi(i64::from(k))
                }))
            }
            // Even central moments E[(X−μ)ᵏ] = (−1)^{k/2+1} (2ᵏ − 2) B_k (πs)ᵏ
            // (from the standard logistic mgf πt/sin(πt)), odd ones 0.
            ContinuousFamily::Logistic { mean, scale } => {
                Some(raw_from_even_central(mean, n, ctx, |k| {
                    let sign = if (k / 2) % 2 == 1 { 1 } else { -1 };
                    let k_ex = ctx.int(i64::from(k));
                    ctx.int(sign)
                        * (ctx.int(2).powi(i64::from(k)) - 2)
                        * k_ex.bernoulli_number()
                        * (ctx.pi() * scale).powi(i64::from(k))
                }))
            }
            // e^{nμ + n²σ²/2}
            ContinuousFamily::LogNormal { mu, sigma } => {
                Some((&n_ex * mu + n_ex.powi(2) * sigma.powi(2) / ctx.int(2)).exp())
            }
            // Odd moments 0; E[X²ᵐ] = νᵐ Π_{i=1}^{m} (2i−1)/(ν−2i), for n < ν.
            ContinuousFamily::StudentT { dof } => {
                if !exceeds(dof, n) {
                    return None;
                }
                if n % 2 == 1 {
                    return Some(ctx.zero());
                }
                let m = n / 2;
                let mut acc = dof.powi(i64::from(m));
                for i in 1..=m {
                    acc *= ctx.int(i64::from(2 * i - 1)) / (dof - i64::from(2 * i));
                }
                Some(acc.simplify())
            }
            // λⁿ Γ(1 + n/k)
            ContinuousFamily::Weibull { scale, shape } => {
                Some((scale.powi(i64::from(n)) * (ctx.one() + &n_ex / shape).gamma()).simplify())
            }
            // α x_mⁿ / (α − n) for n < α.
            ContinuousFamily::Pareto { scale, shape } => exceeds(shape, n)
                .then(|| (shape * scale.powi(i64::from(n)) / (shape - &n_ex)).simplify()),
            // 2 [aⁿ⁺²(b−c) − bⁿ⁺²(a−c) + cⁿ⁺²(a−b)] / ((n+1)(n+2)(a−b)(a−c)(b−c))
            // (integrate xⁿ against the two linear pieces).
            ContinuousFamily::Triangular { lo, hi, mode } => {
                let e = i64::from(n) + 2;
                let num = ctx.int(2)
                    * (lo.powi(e) * (hi - mode) - hi.powi(e) * (lo - mode)
                        + mode.powi(e) * (lo - hi));
                let den = ctx.int(e - 1) * ctx.int(e) * (lo - hi) * (lo - mode) * (hi - mode);
                Some((num / den).simplify())
            }
        }
    }

    /// Closed-form CDF in `x`.  `None` for `Beta` (the regularised
    /// incomplete beta function is not in the crate; the generic route
    /// integrates the density, which closes for integer `α`, `β`) and
    /// `StudentT` (the generic route closes for odd integer `ν`).
    pub fn cdf(&self, x: &Ex) -> Option<Ex> {
        let ctx = x.context();
        let half = ctx.rational(1, 2);
        match self {
            // Φ((x−μ)/σ) = ½ + ½ erf((x−μ)/(σ√2))
            ContinuousFamily::Normal { mean, std } => {
                let arg = (x - mean) / (std * ctx.int(2).sqrt());
                Some(&half + &half * arg.erf())
            }
            ContinuousFamily::Uniform { lo, hi } => Some((x - lo) / (hi - lo)),
            // 1 − e^{−λx}
            ContinuousFamily::Exponential { rate } => Some(ctx.one() - (-(rate * x)).exp()),
            // γ(k, x/θ) / Γ(k); `eval` closes the incomplete gamma for
            // integer and half-integer `k`.
            ContinuousFamily::Gamma { shape, scale } => {
                Some(((x / scale).lowergamma(shape) / shape.gamma()).eval())
            }
            ContinuousFamily::ChiSquared { dof } => chi_squared_as_gamma(dof).cdf(x),
            ContinuousFamily::Beta { .. } | ContinuousFamily::StudentT { .. } => None,
            // ½ + atan((x−x₀)/γ)/π
            ContinuousFamily::Cauchy { location, scale } => {
                Some(&half + ((x - location) / scale).atan() / ctx.pi())
            }
            // ½ e^{(x−μ)/b} for x < μ, 1 − ½ e^{−(x−μ)/b} for x ≥ μ
            ContinuousFamily::Laplace { mean, scale } => {
                let z = (x - mean) / scale;
                let below = &half * z.exp();
                let above = ctx.one() - &half * (-z).exp();
                Some(Ex::piecewise(&[
                    (&below, &x.lt(mean)),
                    (&above, &x.ge(mean)),
                ]))
            }
            // 1 / (1 + e^{−(x−μ)/s})
            ContinuousFamily::Logistic { mean, scale } => {
                Some(ctx.one() / (ctx.one() + (-((x - mean) / scale)).exp()))
            }
            // Φ((ln x − μ)/σ)
            ContinuousFamily::LogNormal { mu, sigma } => {
                let arg = (x.ln() - mu) / (sigma * ctx.int(2).sqrt());
                Some(&half + &half * arg.erf())
            }
            // 1 − e^{−(x/λ)ᵏ}
            ContinuousFamily::Weibull { scale, shape } => {
                Some(ctx.one() - (-((x / scale).pow(shape))).exp())
            }
            // 1 − (x_m/x)^α
            ContinuousFamily::Pareto { scale, shape } => Some(ctx.one() - (scale / x).pow(shape)),
            // (x−a)²/((b−a)(c−a)) on [a, c], 1 − (b−x)²/((b−a)(b−c)) on (c, b]
            ContinuousFamily::Triangular { lo, hi, mode } => {
                let width = hi - lo;
                let rising = (x - lo).powi(2) / (&width * (mode - lo));
                let falling = ctx.one() - (hi - x).powi(2) / (&width * (hi - mode));
                Some(Ex::piecewise(&[
                    (&rising, &x.le(mode)),
                    (&falling, &x.gt(mode)),
                ]))
            }
        }
    }

    /// Closed-form moment generating function in `t`.  `None` where none
    /// exists in closed form: `Beta` (a ₁F₁), `Cauchy` (undefined),
    /// `LogNormal` (divergent for `t > 0`), `StudentT` (undefined),
    /// `Weibull` and `Pareto` (series / incomplete gamma only); the generic
    /// route then returns `E[e^{tX}]` as an integral.
    pub fn mgf(&self, t: &Ex) -> Option<Ex> {
        let ctx = t.context();
        match self {
            // exp(μt + σ²t²/2)
            ContinuousFamily::Normal { mean, std } => {
                Some((mean * t + std.powi(2) * t.powi(2) / ctx.int(2)).exp())
            }
            // (e^{bt} − e^{at}) / ((b−a) t)
            ContinuousFamily::Uniform { lo, hi } => {
                Some(((hi * t).exp() - (lo * t).exp()) / ((hi - lo) * t))
            }
            // λ / (λ − t)
            ContinuousFamily::Exponential { rate } => Some(rate / (rate - t)),
            // (1 − θt)^{−k}
            ContinuousFamily::Gamma { shape, scale } => {
                Some((ctx.one() - scale * t).pow(&(-shape)))
            }
            ContinuousFamily::ChiSquared { dof } => chi_squared_as_gamma(dof).mgf(t),
            ContinuousFamily::Beta { .. }
            | ContinuousFamily::Cauchy { .. }
            | ContinuousFamily::LogNormal { .. }
            | ContinuousFamily::StudentT { .. }
            | ContinuousFamily::Weibull { .. }
            | ContinuousFamily::Pareto { .. } => None,
            // e^{μt} / (1 − b²t²)
            ContinuousFamily::Laplace { mean, scale } => {
                Some((mean * t).exp() / (ctx.one() - scale.powi(2) * t.powi(2)))
            }
            // e^{μt} B(1 − st, 1 + st)
            ContinuousFamily::Logistic { mean, scale } => {
                let st = scale * t;
                Some((mean * t).exp() * (ctx.one() - &st).beta(&(ctx.one() + &st)))
            }
            // 2 [(b−c) e^{at} − (b−a) e^{ct} + (c−a) e^{bt}] / ((b−a)(c−a)(b−c) t²)
            ContinuousFamily::Triangular { lo, hi, mode } => {
                let num = ctx.int(2)
                    * ((hi - mode) * (lo * t).exp() - (hi - lo) * (mode * t).exp()
                        + (mode - lo) * (hi * t).exp());
                let den = (hi - lo) * (mode - lo) * (hi - mode) * t.powi(2);
                Some(num / den)
            }
        }
    }

    /// Closed-form quantile function at `p`.  `None` for `Gamma`,
    /// `ChiSquared`, `Beta` and `StudentT`, whose inverse CDFs have no
    /// elementary form (so [`RandomVariable::sample`](super::RandomVariable::sample)
    /// is not available for them).
    pub fn quantile(&self, p: &Ex) -> Option<Ex> {
        let ctx = p.context();
        let half = ctx.rational(1, 2);
        match self {
            // μ + σ√2 · erfinv(2p − 1)
            ContinuousFamily::Normal { mean, std } => {
                Some(mean + std * ctx.int(2).sqrt() * (ctx.int(2) * p - 1).erfinv())
            }
            // a + p(b − a)
            ContinuousFamily::Uniform { lo, hi } => Some(lo + p * (hi - lo)),
            // −ln(1 − p) / λ
            ContinuousFamily::Exponential { rate } => Some(-(ctx.one() - p).ln() / rate),
            ContinuousFamily::Gamma { .. }
            | ContinuousFamily::ChiSquared { .. }
            | ContinuousFamily::Beta { .. }
            | ContinuousFamily::StudentT { .. } => None,
            // x₀ + γ tan(π(p − ½))
            ContinuousFamily::Cauchy { location, scale } => {
                Some(location + scale * (ctx.pi() * (p - &half)).tan())
            }
            // μ − b sign(p − ½) ln(1 − 2|p − ½|)
            ContinuousFamily::Laplace { mean, scale } => {
                let d = p - &half;
                Some(mean - scale * d.sign() * (ctx.one() - ctx.int(2) * d.abs()).ln())
            }
            // μ + s ln(p/(1 − p))
            ContinuousFamily::Logistic { mean, scale } => {
                Some(mean + scale * (p / (ctx.one() - p)).ln())
            }
            // exp(μ + σ√2 · erfinv(2p − 1))
            ContinuousFamily::LogNormal { mu, sigma } => {
                Some((mu + sigma * ctx.int(2).sqrt() * (ctx.int(2) * p - 1).erfinv()).exp())
            }
            // λ (−ln(1 − p))^{1/k}
            ContinuousFamily::Weibull { scale, shape } => {
                Some(scale * (-(ctx.one() - p).ln()).pow(&(ctx.one() / shape)))
            }
            // x_m (1 − p)^{−1/α}
            ContinuousFamily::Pareto { scale, shape } => {
                Some(scale * (ctx.one() - p).pow(&(-(ctx.one() / shape))))
            }
            // a + √(p(b−a)(c−a)) for p < (c−a)/(b−a), else b − √((1−p)(b−a)(b−c))
            ContinuousFamily::Triangular { lo, hi, mode } => {
                let width = hi - lo;
                let threshold = (mode - lo) / &width;
                let rising = lo + (p * &width * (mode - lo)).sqrt();
                let falling = hi - ((ctx.one() - p) * &width * (hi - mode)).sqrt();
                Some(Ex::piecewise(&[
                    (&rising, &p.lt(&threshold)),
                    (&falling, &p.ge(&threshold)),
                ]))
            }
        }
    }

    /// Closed-form differential entropy `−∫ f ln f` in nats (Wikipedia's
    /// per-family tables; `ψ` is the digamma function, `γ` the
    /// Euler–Mascheroni constant).  Every family has one.
    pub fn entropy(&self, ctx: &Context) -> Option<Ex> {
        let half = ctx.rational(1, 2);
        let two_pi_e = ctx.int(2) * ctx.pi() * ctx.e();
        match self {
            // ½ ln(2πeσ²)
            ContinuousFamily::Normal { std, .. } => Some(&half * (two_pi_e * std.powi(2)).ln()),
            // ln(b − a)
            ContinuousFamily::Uniform { lo, hi } => Some((hi - lo).ln()),
            // 1 − ln λ
            ContinuousFamily::Exponential { rate } => Some(ctx.one() - rate.ln()),
            // k + ln θ + ln Γ(k) + (1 − k) ψ(k)
            ContinuousFamily::Gamma { shape, scale } => Some(
                shape + scale.ln() + shape.gamma().ln() + (ctx.one() - shape) * shape.digamma(),
            ),
            ContinuousFamily::ChiSquared { dof } => chi_squared_as_gamma(dof).entropy(ctx),
            // ln B(α, β) − (α−1) ψ(α) − (β−1) ψ(β) + (α+β−2) ψ(α+β)
            ContinuousFamily::Beta { alpha, beta } => {
                let s = alpha + beta;
                Some(
                    alpha.beta(beta).ln()
                        - (alpha - 1) * alpha.digamma()
                        - (beta - 1) * beta.digamma()
                        + (&s - 2) * s.digamma(),
                )
            }
            // ln(4πγ)
            ContinuousFamily::Cauchy { scale, .. } => Some((ctx.int(4) * ctx.pi() * scale).ln()),
            // 1 + ln(2b)
            ContinuousFamily::Laplace { scale, .. } => Some(ctx.one() + (ctx.int(2) * scale).ln()),
            // ln s + 2
            ContinuousFamily::Logistic { scale, .. } => Some(scale.ln() + 2),
            // μ + ½ ln(2πeσ²)
            ContinuousFamily::LogNormal { mu, sigma } => {
                Some(mu + &half * (two_pi_e * sigma.powi(2)).ln())
            }
            // (ν+1)/2 [ψ((ν+1)/2) − ψ(ν/2)] + ln(√ν B(ν/2, ½))
            ContinuousFamily::StudentT { dof } => {
                let a = (dof + 1) * &half;
                let b = dof * &half;
                Some(&a * (a.digamma() - b.digamma()) + (dof.sqrt() * b.beta(&half)).ln())
            }
            // γ(1 − 1/k) + ln(λ/k) + 1
            ContinuousFamily::Weibull { scale, shape } => {
                Some(ctx.euler_gamma() * (ctx.one() - ctx.one() / shape) + (scale / shape).ln() + 1)
            }
            // ln(x_m/α) + 1/α + 1
            ContinuousFamily::Pareto { scale, shape } => {
                Some((scale / shape).ln() + ctx.one() / shape + 1)
            }
            // ½ + ln((b − a)/2)
            ContinuousFamily::Triangular { lo, hi, .. } => {
                Some(&half + ((hi - lo) / ctx.int(2)).ln())
            }
        }
    }
}

impl fmt::Display for ContinuousFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContinuousFamily::Normal { mean, std } => write!(f, "Normal({mean}, {std})"),
            ContinuousFamily::Uniform { lo, hi } => write!(f, "Uniform({lo}, {hi})"),
            ContinuousFamily::Exponential { rate } => write!(f, "Exponential({rate})"),
            ContinuousFamily::Gamma { shape, scale } => write!(f, "Gamma({shape}, {scale})"),
            ContinuousFamily::ChiSquared { dof } => write!(f, "ChiSquared({dof})"),
            ContinuousFamily::Beta { alpha, beta } => write!(f, "Beta({alpha}, {beta})"),
            ContinuousFamily::Cauchy { location, scale } => {
                write!(f, "Cauchy({location}, {scale})")
            }
            ContinuousFamily::Laplace { mean, scale } => write!(f, "Laplace({mean}, {scale})"),
            ContinuousFamily::Logistic { mean, scale } => write!(f, "Logistic({mean}, {scale})"),
            ContinuousFamily::LogNormal { mu, sigma } => write!(f, "LogNormal({mu}, {sigma})"),
            ContinuousFamily::StudentT { dof } => write!(f, "StudentT({dof})"),
            ContinuousFamily::Weibull { scale, shape } => write!(f, "Weibull({scale}, {shape})"),
            ContinuousFamily::Pareto { scale, shape } => write!(f, "Pareto({scale}, {shape})"),
            ContinuousFamily::Triangular { lo, hi, mode } => {
                write!(f, "Triangular({lo}, {hi}, {mode})")
            }
        }
    }
}
