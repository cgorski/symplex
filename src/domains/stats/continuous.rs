//! Continuous distribution families.  Each family stores its parameters as
//! expressions and provides support, density, and whatever closed forms it
//! has; the generic machinery in [`super::rv`] does the rest.
//!
//! To add a family: a variant of [`ContinuousFamily`] with documented
//! parameters, a constructor `Distribution::name(…)` that validates
//! numeric parameters (`Result`), and arms in each `match` below.  Every
//! closed form here is a textbook identity; cite it in the arm.

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
}

impl ContinuousFamily {
    /// The family's name.
    pub fn name(&self) -> &'static str {
        match self {
            ContinuousFamily::Normal { .. } => "Normal",
        }
    }

    /// The support.
    pub fn support(&self) -> Support {
        match self {
            ContinuousFamily::Normal { .. } => Support::Continuous { lo: None, hi: None },
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
        }
    }

    /// Closed-form mean.
    pub fn mean(&self, _ctx: &Context) -> Option<Ex> {
        match self {
            ContinuousFamily::Normal { mean, .. } => Some(mean.clone()),
        }
    }

    /// Closed-form variance.
    pub fn variance(&self, _ctx: &Context) -> Option<Ex> {
        match self {
            ContinuousFamily::Normal { std, .. } => Some(std.powi(2)),
        }
    }

    /// Closed-form raw moment `E[Xⁿ]`.
    pub fn raw_moment(&self, n: u32, ctx: &Context) -> Option<Ex> {
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
        }
    }

    /// Closed-form CDF in `x`.
    pub fn cdf(&self, x: &Ex) -> Option<Ex> {
        let ctx = x.context();
        match self {
            // Φ((x−μ)/σ) = ½ + ½ erf((x−μ)/(σ√2))
            ContinuousFamily::Normal { mean, std } => {
                let half = ctx.rational(1, 2);
                let arg = (x - mean) / (std * ctx.int(2).sqrt());
                Some(&half + &half * arg.erf())
            }
        }
    }

    /// Closed-form moment generating function in `t`.
    pub fn mgf(&self, t: &Ex) -> Option<Ex> {
        let ctx = t.context();
        match self {
            // exp(μt + σ²t²/2)
            ContinuousFamily::Normal { mean, std } => {
                Some((mean * t + std.powi(2) * t.powi(2) / ctx.int(2)).exp())
            }
        }
    }

    /// Closed-form quantile function at `p`.
    pub fn quantile(&self, p: &Ex) -> Option<Ex> {
        let ctx = p.context();
        match self {
            // μ + σ√2 · erfinv(2p − 1)
            ContinuousFamily::Normal { mean, std } => {
                Some(mean + std * ctx.int(2).sqrt() * (ctx.int(2) * p - 1).erfinv())
            }
        }
    }
}

impl fmt::Display for ContinuousFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContinuousFamily::Normal { mean, std } => write!(f, "Normal({mean}, {std})"),
        }
    }
}
