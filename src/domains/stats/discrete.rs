//! Discrete distribution families (probability mass on integers).  Same
//! shape as [`super::continuous`]: parameters as expressions, support,
//! pmf, and closed forms where the family has them.

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

/// Reject a numeric probability outside `[0, 1]`; accept symbolic ones.
fn require_probability(p: &Ex, what: &str) -> Result<(), SymplexError> {
    if let Some(q) = p.eval().as_rational()
        && (q < num_rational::Ratio::from_integer(0.into())
            || q > num_rational::Ratio::from_integer(1.into()))
    {
        return Err(invalid(format!("{what} must lie in [0, 1], got `{p}`")));
    }
    Ok(())
}

/// Reject a numeric count that is not a non-negative integer; accept
/// symbolic ones.
fn require_count(n: &Ex, what: &str) -> Result<(), SymplexError> {
    if let Some(q) = n.eval().as_rational()
        && (!q.is_integer() || q < num_rational::Ratio::from_integer(0.into()))
    {
        return Err(invalid(format!(
            "{what} must be a non-negative integer, got `{n}`"
        )));
    }
    Ok(())
}

/// The discrete families.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum DiscreteFamily {
    /// `Binomial(n, p)`: `P(X = k) = C(n, k) pᵏ (1−p)ⁿ⁻ᵏ` on `0..=n`.
    Binomial {
        /// Number of trials `n`.
        n: Ex,
        /// Success probability `p ∈ [0, 1]`.
        p: Ex,
    },
}

impl Distribution {
    /// `Binomial(n, p)`.  SymPy: `Binomial('X', n, p)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `n` is a number that is not a
    /// non-negative integer or `p` a number outside `[0, 1]`.
    pub fn try_binomial(n: Ex, p: Ex) -> Result<Distribution, SymplexError> {
        require_count(&n, "the number of trials")?;
        require_probability(&p, "the success probability")?;
        Ok(Distribution::Discrete(DiscreteFamily::Binomial { n, p }))
    }

    /// `Binomial(n, p)` without parameter validation (see
    /// [`try_binomial`](Self::try_binomial)).
    pub fn binomial(n: Ex, p: Ex) -> Distribution {
        Distribution::Discrete(DiscreteFamily::Binomial { n, p })
    }
}

impl DiscreteFamily {
    /// The family's name.
    pub fn name(&self) -> &'static str {
        match self {
            DiscreteFamily::Binomial { .. } => "Binomial",
        }
    }

    /// The support.
    pub fn support(&self) -> Support {
        match self {
            DiscreteFamily::Binomial { n, .. } => Support::Discrete {
                lo: Some(n.context().zero()),
                hi: Some(n.clone()),
            },
        }
    }

    /// The probability mass function as an expression in `k` (valid on the
    /// support).
    pub fn pmf(&self, k: &Ex) -> Ex {
        let ctx = k.context();
        match self {
            DiscreteFamily::Binomial { n, p } => {
                let q = ctx.one() - p;
                n.binomial(k) * p.pow(k) * q.pow(&(n - k))
            }
        }
    }

    /// Closed-form mean.
    pub fn mean(&self, _ctx: &Context) -> Option<Ex> {
        match self {
            DiscreteFamily::Binomial { n, p } => Some((n * p).simplify()),
        }
    }

    /// Closed-form variance.
    pub fn variance(&self, ctx: &Context) -> Option<Ex> {
        match self {
            DiscreteFamily::Binomial { n, p } => Some((n * p * (ctx.one() - p)).simplify()),
        }
    }

    /// Closed-form raw moment `E[Xⁿ]` (`None` leaves it to the generic
    /// summation over the support).
    pub fn raw_moment(&self, _n: u32, _ctx: &Context) -> Option<Ex> {
        match self {
            DiscreteFamily::Binomial { .. } => None,
        }
    }

    /// Closed-form CDF in `k` (as a finite sum where no simpler form exists;
    /// `None` leaves it to the generic summation).
    pub fn cdf(&self, _k: &Ex) -> Option<Ex> {
        match self {
            DiscreteFamily::Binomial { .. } => None,
        }
    }

    /// Closed-form moment generating function in `t`.
    pub fn mgf(&self, t: &Ex) -> Option<Ex> {
        let ctx = t.context();
        match self {
            // (1 − p + p eᵗ)ⁿ
            DiscreteFamily::Binomial { n, p } => Some((ctx.one() - p + p * t.exp()).pow(n)),
        }
    }

    /// Closed-form quantile function at `p` (discrete families rarely have
    /// one).
    pub fn quantile(&self, _p: &Ex) -> Option<Ex> {
        match self {
            DiscreteFamily::Binomial { .. } => None,
        }
    }
}

impl fmt::Display for DiscreteFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiscreteFamily::Binomial { n, p } => write!(f, "Binomial({n}, {p})"),
        }
    }
}
