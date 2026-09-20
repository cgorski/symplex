//! Discrete distribution families (probability mass on integers).  Same
//! shape as [`super::continuous`]: parameters as expressions, support,
//! pmf, and closed forms where the family has them.
//!
//! To add a family: a variant of [`DiscreteFamily`] with documented
//! parameters, a constructor pair `Distribution::try_name(…)` (validates
//! numeric parameters, `Result`) / `Distribution::name(…)` (unchecked), and
//! arms in each `match` below.  Every closed form here is a textbook
//! identity; cite it in the arm.  Raw moments come from one of two exact
//! routes shared by several families: factorial moments through Stirling
//! numbers ([`raw_moment_from_factorial_moments`]) and derivatives of the
//! moment generating function at `0` ([`DiscreteFamily::moment_from_mgf`]).

use std::fmt;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::errors::SymplexError;
use crate::domains::combinatorics::stirling2;

use super::rv::{Distribution, Support};

type Rat = Ratio<BigInt>;

fn invalid(reason: impl Into<String>) -> SymplexError {
    SymplexError::InvalidArgument {
        operation: "stats",
        reason: reason.into(),
    }
}

/// The exact value of a numeric parameter; `None` for a symbolic one.
fn numeric(e: &Ex) -> Option<Rat> {
    e.eval().as_rational()
}

/// Reject a numeric probability outside `[0, 1]`; accept symbolic ones.
fn require_probability(p: &Ex, what: &str) -> Result<(), SymplexError> {
    if let Some(q) = numeric(p)
        && (q < Rat::zero() || q > Rat::one())
    {
        return Err(invalid(format!("{what} must lie in [0, 1], got `{p}`")));
    }
    Ok(())
}

/// Reject a numeric probability outside `(0, 1]` (`allow_one`) or `(0, 1)`;
/// accept symbolic ones.
fn require_probability_positive(p: &Ex, what: &str, allow_one: bool) -> Result<(), SymplexError> {
    if let Some(q) = numeric(p)
        && (q <= Rat::zero() || q > Rat::one() || (!allow_one && q == Rat::one()))
    {
        let range = if allow_one { "(0, 1]" } else { "(0, 1)" };
        return Err(invalid(format!("{what} must lie in {range}, got `{p}`")));
    }
    Ok(())
}

/// Reject a numeric count that is not a non-negative integer; accept
/// symbolic ones.
fn require_count(n: &Ex, what: &str) -> Result<(), SymplexError> {
    if let Some(q) = numeric(n)
        && (!q.is_integer() || q < Rat::zero())
    {
        return Err(invalid(format!(
            "{what} must be a non-negative integer, got `{n}`"
        )));
    }
    Ok(())
}

/// Reject a numeric value that is not an integer; accept symbolic ones.
fn require_integer(e: &Ex, what: &str) -> Result<(), SymplexError> {
    if let Some(q) = numeric(e)
        && !q.is_integer()
    {
        return Err(invalid(format!("{what} must be an integer, got `{e}`")));
    }
    Ok(())
}

/// Reject a parameter known not to be positive (a number `≤ 0`, or a symbol
/// whose assumptions decide it); accept the undecided.
fn require_positive(e: &Ex, what: &str) -> Result<(), SymplexError> {
    if e.is_positive() == Some(false) {
        return Err(invalid(format!("{what} must be positive, got `{e}`")));
    }
    Ok(())
}

/// Reject numeric `a`, `b` with `a > b`; accept when either is symbolic.
fn require_le(a: &Ex, b: &Ex, what: &str) -> Result<(), SymplexError> {
    if let (Some(x), Some(y)) = (numeric(a), numeric(b))
        && x > y
    {
        return Err(invalid(format!("{what}: `{a}` exceeds `{b}`")));
    }
    Ok(())
}

/// The falling factorial `x^{(k)} = x(x−1)⋯(x−k+1)` as an explicit product
/// (a polynomial in `x`, so symbolic parameters stay polynomial).
fn falling_factorial(x: &Ex, k: u32) -> Ex {
    let ctx = x.context();
    let mut acc = ctx.one();
    for j in 0..k {
        acc *= x - ctx.int(i64::from(j));
    }
    acc
}

/// `E[Xⁿ] = Σ_{k=1}^{n} S(n, k) · E[X^{(k)}]`: a raw moment from the
/// factorial moments `E[X^{(k)}] = E[X(X−1)⋯(X−k+1)]` through the Stirling
/// numbers of the second kind (`xⁿ = Σ_k S(n, k) x^{(k)}`).  `None` only if
/// a Stirling number is out of range (it is not, for `u32` arguments).
fn raw_moment_from_factorial_moments(
    n: u32,
    ctx: &Context,
    factorial_moment: impl Fn(u32) -> Ex,
) -> Option<Ex> {
    let mut acc = ctx.zero();
    for k in 1..=n {
        let s = ctx.from_bigint(stirling2(n, k)?);
        acc += s * factorial_moment(k);
    }
    Some(acc.simplify())
}

/// Largest numeric integer `r` for which the NegativeBinomial pmf spells the
/// binomial coefficient out as the polynomial `(k+1)⋯(k+r−1)/(r−1)!`.  The
/// crate's summation engine closes `Σ poly(k)·xᵏ` over infinite ranges but
/// not `Σ C(k+c, k)·xᵏ`, so the polynomial form is what makes
/// probabilities and expectations of a NegativeBinomial variable exact.
const NEGATIVE_BINOMIAL_POLYNOMIAL_MAX_R: i64 = 32;

/// The discrete families.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum DiscreteFamily {
    /// An explicit finite table `P(X = vᵢ) = pᵢ` (SymPy `FiniteRV`); the
    /// values need not be integers.
    Finite {
        /// `(value, probability)` pairs, in the order given.
        table: Vec<(Ex, Ex)>,
    },
    /// `Bernoulli(p)`: `P(X = 1) = p`, `P(X = 0) = 1 − p` on `{0, 1}`.
    Bernoulli {
        /// Success probability `p ∈ [0, 1]`.
        p: Ex,
    },
    /// `Binomial(n, p)`: `P(X = k) = C(n, k) pᵏ (1−p)ⁿ⁻ᵏ` on `0..=n`.
    Binomial {
        /// Number of trials `n`.
        n: Ex,
        /// Success probability `p ∈ [0, 1]`.
        p: Ex,
    },
    /// `Poisson(λ)`: `P(X = k) = λᵏ e^{−λ} / k!` on `0..∞`.
    Poisson {
        /// Rate `λ > 0`.
        rate: Ex,
    },
    /// `Geometric(p)`: the number of trials up to and including the first
    /// success, `P(X = k) = (1−p)^{k−1} p` on `1..∞` (SymPy's convention).
    Geometric {
        /// Success probability `p ∈ (0, 1]`.
        p: Ex,
    },
    /// `NegativeBinomial(r, p)`: the number of failures before the `r`-th
    /// success, `P(X = k) = C(k+r−1, k) pʳ (1−p)ᵏ` on `0..∞` (SymPy's
    /// convention).
    NegativeBinomial {
        /// Number of successes `r > 0`.
        r: Ex,
        /// Success probability `p ∈ (0, 1)`.
        p: Ex,
    },
    /// `Hypergeometric(N, m, n)`: the number of successes in `n` draws
    /// without replacement from a population of `N` containing `m`
    /// successes, `P(X = k) = C(m, k) C(N−m, n−k) / C(N, n)` on
    /// `max(0, n+m−N) ..= min(n, m)`.
    Hypergeometric {
        /// Population size `N`.
        population: Ex,
        /// Number of successes `m ≤ N` in the population.
        successes: Ex,
        /// Number of draws `n ≤ N`.
        draws: Ex,
    },
    /// `DiscreteUniform(a, b)`: `P(X = k) = 1/(b−a+1)` on the integers
    /// `a..=b`.  `Die(s)` is `DiscreteUniform(1, s)`.
    DiscreteUniform {
        /// Lowest value `a`.
        a: Ex,
        /// Highest value `b ≥ a`.
        b: Ex,
    },
}

impl Distribution {
    /// A finite distribution from an explicit `(value, probability)` table.
    /// SymPy: `FiniteRV('X', {v: p, …})`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the table is empty, a numeric
    /// probability is negative, two numeric values coincide, or all
    /// probabilities are numeric and do not sum to exactly `1`.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::{Distribution, RandomVariable};
    ///
    /// let ctx = Context::new();
    /// // A loaded coin: heads (1) with probability 2/3.
    /// let coin = Distribution::try_finite(vec![
    ///     (ctx.int(1), ctx.rational(2, 3)),
    ///     (ctx.int(0), ctx.rational(1, 3)),
    /// ])?;
    /// let c = RandomVariable::new(&ctx, "C", coin);
    /// assert_eq!(c.mean(), ctx.rational(2, 3));
    /// assert_eq!(c.variance(), ctx.rational(2, 9));
    /// assert_eq!(c.probability(&c.symbol().eq_expr(&ctx.int(1)))?, ctx.rational(2, 3));
    /// # Ok::<(), SymplexError>(())
    /// ```
    pub fn try_finite(table: Vec<(Ex, Ex)>) -> Result<Distribution, SymplexError> {
        use num_bigint::BigInt;
        use num_rational::Ratio;
        if table.is_empty() {
            return Err(invalid("a finite distribution needs at least one value"));
        }
        let mut total = Ratio::from_integer(BigInt::from(0));
        let mut all_numeric = true;
        for (v, p) in &table {
            match p.eval().as_rational() {
                Some(q) => {
                    if q < Ratio::from_integer(BigInt::from(0)) {
                        return Err(invalid(format!("probability of `{v}` is negative: `{p}`")));
                    }
                    total += q;
                }
                None => all_numeric = false,
            }
        }
        if all_numeric && total != Ratio::from_integer(BigInt::from(1)) {
            return Err(invalid(format!("the probabilities sum to {total}, not 1")));
        }
        for (i, (v, _)) in table.iter().enumerate() {
            for (w, _) in &table[i + 1..] {
                if v.equals(w) == Some(true) {
                    return Err(invalid(format!("the value `{v}` is listed twice")));
                }
            }
        }
        Ok(Distribution::Discrete(DiscreteFamily::Finite { table }))
    }

    /// A finite distribution from a table without validation (see
    /// [`try_finite`](Self::try_finite)).
    pub fn finite(table: Vec<(Ex, Ex)>) -> Distribution {
        Distribution::Discrete(DiscreteFamily::Finite { table })
    }

    /// `Bernoulli(p)`.  SymPy: `Bernoulli('X', p)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `p` is a number outside `[0, 1]`.
    pub fn try_bernoulli(p: Ex) -> Result<Distribution, SymplexError> {
        require_probability(&p, "the success probability")?;
        Ok(Distribution::Discrete(DiscreteFamily::Bernoulli { p }))
    }

    /// `Bernoulli(p)` without parameter validation (see
    /// [`try_bernoulli`](Self::try_bernoulli)).
    pub fn bernoulli(p: Ex) -> Distribution {
        Distribution::Discrete(DiscreteFamily::Bernoulli { p })
    }

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

    /// `Poisson(λ)` with rate `rate`.  SymPy: `Poisson('X', lamda)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `rate` is a number `≤ 0`.
    pub fn try_poisson(rate: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&rate, "the rate")?;
        Ok(Distribution::Discrete(DiscreteFamily::Poisson { rate }))
    }

    /// `Poisson(λ)` without parameter validation (see
    /// [`try_poisson`](Self::try_poisson)).
    pub fn poisson(rate: Ex) -> Distribution {
        Distribution::Discrete(DiscreteFamily::Poisson { rate })
    }

    /// `Geometric(p)` on `1..∞` (trials up to the first success).  SymPy:
    /// `Geometric('X', p)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `p` is a number outside `(0, 1]`.
    pub fn try_geometric(p: Ex) -> Result<Distribution, SymplexError> {
        require_probability_positive(&p, "the success probability", true)?;
        Ok(Distribution::Discrete(DiscreteFamily::Geometric { p }))
    }

    /// `Geometric(p)` without parameter validation (see
    /// [`try_geometric`](Self::try_geometric)).
    pub fn geometric(p: Ex) -> Distribution {
        Distribution::Discrete(DiscreteFamily::Geometric { p })
    }

    /// `NegativeBinomial(r, p)` on `0..∞` (failures before the `r`-th
    /// success).  SymPy: `NegativeBinomial('X', r, p)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `r` is a number `≤ 0` or `p` a
    /// number outside `(0, 1)`.
    pub fn try_negative_binomial(r: Ex, p: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&r, "the number of successes")?;
        require_probability_positive(&p, "the success probability", false)?;
        Ok(Distribution::Discrete(DiscreteFamily::NegativeBinomial {
            r,
            p,
        }))
    }

    /// `NegativeBinomial(r, p)` without parameter validation (see
    /// [`try_negative_binomial`](Self::try_negative_binomial)).
    pub fn negative_binomial(r: Ex, p: Ex) -> Distribution {
        Distribution::Discrete(DiscreteFamily::NegativeBinomial { r, p })
    }

    /// `Hypergeometric(N, m, n)`: `draws` without replacement from a
    /// population of `population` with `successes` successes.  SymPy:
    /// `Hypergeometric('X', N, m, n)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if a numeric parameter is not a
    /// non-negative integer, or `successes` / `draws` numerically exceed
    /// `population`.
    pub fn try_hypergeometric(
        population: Ex,
        successes: Ex,
        draws: Ex,
    ) -> Result<Distribution, SymplexError> {
        require_count(&population, "the population size")?;
        require_count(&successes, "the number of successes")?;
        require_count(&draws, "the number of draws")?;
        require_le(
            &successes,
            &population,
            "the number of successes must not exceed the population",
        )?;
        require_le(
            &draws,
            &population,
            "the number of draws must not exceed the population",
        )?;
        Ok(Distribution::Discrete(DiscreteFamily::Hypergeometric {
            population,
            successes,
            draws,
        }))
    }

    /// `Hypergeometric(N, m, n)` without parameter validation (see
    /// [`try_hypergeometric`](Self::try_hypergeometric)).
    pub fn hypergeometric(population: Ex, successes: Ex, draws: Ex) -> Distribution {
        Distribution::Discrete(DiscreteFamily::Hypergeometric {
            population,
            successes,
            draws,
        })
    }

    /// `DiscreteUniform(a, b)` on the integers `a..=b`.  SymPy:
    /// `DiscreteUniform('X', range(a, b + 1))`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `a` or `b` is a non-integer
    /// number, or both are numbers with `a > b`.
    pub fn try_discrete_uniform(a: Ex, b: Ex) -> Result<Distribution, SymplexError> {
        require_integer(&a, "the lowest value")?;
        require_integer(&b, "the highest value")?;
        require_le(&a, &b, "the lowest value must not exceed the highest")?;
        Ok(Distribution::Discrete(DiscreteFamily::DiscreteUniform {
            a,
            b,
        }))
    }

    /// `DiscreteUniform(a, b)` without parameter validation (see
    /// [`try_discrete_uniform`](Self::try_discrete_uniform)).
    pub fn discrete_uniform(a: Ex, b: Ex) -> Distribution {
        Distribution::Discrete(DiscreteFamily::DiscreteUniform { a, b })
    }

    /// A fair die with `sides` faces: `DiscreteUniform(1, sides)`.  SymPy:
    /// `Die('X', sides)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `sides` is a number that is not
    /// a positive integer.
    pub fn try_die(sides: Ex) -> Result<Distribution, SymplexError> {
        require_integer(&sides, "the number of sides")?;
        require_positive(&sides, "the number of sides")?;
        Ok(Distribution::die(sides))
    }

    /// A fair die with `sides` faces without parameter validation (see
    /// [`try_die`](Self::try_die)).
    pub fn die(sides: Ex) -> Distribution {
        let one = sides.context().one();
        Distribution::Discrete(DiscreteFamily::DiscreteUniform { a: one, b: sides })
    }
}

impl DiscreteFamily {
    /// The family's name.
    pub fn name(&self) -> &'static str {
        match self {
            DiscreteFamily::Finite { .. } => "Finite",
            DiscreteFamily::Bernoulli { .. } => "Bernoulli",
            DiscreteFamily::Binomial { .. } => "Binomial",
            DiscreteFamily::Poisson { .. } => "Poisson",
            DiscreteFamily::Geometric { .. } => "Geometric",
            DiscreteFamily::NegativeBinomial { .. } => "NegativeBinomial",
            DiscreteFamily::Hypergeometric { .. } => "Hypergeometric",
            DiscreteFamily::DiscreteUniform { .. } => "DiscreteUniform",
        }
    }

    /// The support.
    pub fn support(&self) -> Support {
        match self {
            DiscreteFamily::Finite { table } => {
                Support::Finite(table.iter().map(|(v, _)| v.clone()).collect())
            }
            DiscreteFamily::Bernoulli { p } => Support::Discrete {
                lo: Some(p.context().zero()),
                hi: Some(p.context().one()),
            },
            DiscreteFamily::Binomial { n, .. } => Support::Discrete {
                lo: Some(n.context().zero()),
                hi: Some(n.clone()),
            },
            DiscreteFamily::Poisson { rate } => Support::Discrete {
                lo: Some(rate.context().zero()),
                hi: None,
            },
            DiscreteFamily::Geometric { p } => Support::Discrete {
                lo: Some(p.context().one()),
                hi: None,
            },
            DiscreteFamily::NegativeBinomial { p, .. } => Support::Discrete {
                lo: Some(p.context().zero()),
                hi: None,
            },
            // max(0, n+m−N) ..= min(n, m); numeric parameters fold the
            // max/min to a number.
            DiscreteFamily::Hypergeometric {
                population,
                successes,
                draws,
            } => {
                let zero = population.context().zero();
                Support::Discrete {
                    lo: Some(zero.max_with(&(draws + successes - population)).simplify()),
                    hi: Some(draws.min_with(successes).simplify()),
                }
            }
            DiscreteFamily::DiscreteUniform { a, b } => Support::Discrete {
                lo: Some(a.clone()),
                hi: Some(b.clone()),
            },
        }
    }

    /// The probability mass function as an expression in `k` (valid on the
    /// support).
    pub fn pmf(&self, k: &Ex) -> Ex {
        let ctx = k.context();
        match self {
            // Piecewise((pᵢ, k = vᵢ), …, (0, True)): the value 0 off the table.
            DiscreteFamily::Finite { table } => {
                let zero = ctx.zero();
                let true_ = ctx.bool_true();
                let pairs: Vec<(Ex, crate::api::expr::BoolEx)> = table
                    .iter()
                    .map(|(v, p)| (p.clone(), k.eq_expr(v)))
                    .chain(std::iter::once((zero, true_)))
                    .collect();
                let refs: Vec<(&Ex, &crate::api::expr::BoolEx)> =
                    pairs.iter().map(|(a, b)| (a, b)).collect();
                Ex::piecewise(&refs)
            }
            // pᵏ (1−p)^{1−k} is p at k = 1 and 1 − p at k = 0.
            DiscreteFamily::Bernoulli { p } => p.pow(k) * (ctx.one() - p).pow(&(ctx.one() - k)),
            DiscreteFamily::Binomial { n, p } => {
                let q = ctx.one() - p;
                n.binomial(k) * p.pow(k) * q.pow(&(n - k))
            }
            DiscreteFamily::Poisson { rate } => rate.pow(k) * (-rate).exp() / k.factorial(),
            DiscreteFamily::Geometric { p } => (ctx.one() - p).pow(&(k - ctx.one())) * p,
            DiscreteFamily::NegativeBinomial { r, p } => {
                // C(k+r−1, k) = (k+1)(k+2)⋯(k+r−1)/(r−1)! for an integer
                // r ≥ 1: the polynomial form is the one the summation
                // engine can sum against (1−p)ᵏ.
                let coeff = match r.as_i64() {
                    Some(ri) if (1..=NEGATIVE_BINOMIAL_POLYNOMIAL_MAX_R).contains(&ri) => {
                        let mut num = ctx.one();
                        let mut den = BigInt::one();
                        for j in 1..ri {
                            num *= k + ctx.int(j);
                            den *= BigInt::from(j);
                        }
                        num / ctx.from_bigint(den)
                    }
                    _ => (k + r - ctx.one()).binomial(k),
                };
                coeff * p.pow(r) * (ctx.one() - p).pow(k)
            }
            DiscreteFamily::Hypergeometric {
                population,
                successes,
                draws,
            } => {
                successes.binomial(k) * (population - successes).binomial(&(draws - k))
                    / population.binomial(draws)
            }
            DiscreteFamily::DiscreteUniform { a, b } => ctx.one() / (b - a + ctx.one()),
        }
    }

    /// Closed-form mean.
    pub fn mean(&self, ctx: &Context) -> Option<Ex> {
        match self {
            DiscreteFamily::Finite { table } => Some(
                table
                    .iter()
                    .fold(ctx.zero(), |acc, (v, p)| acc + v * p)
                    .simplify(),
            ),
            DiscreteFamily::Bernoulli { p } => Some(p.clone()),
            DiscreteFamily::Binomial { n, p } => Some((n * p).simplify()),
            DiscreteFamily::Poisson { rate } => Some(rate.clone()),
            // 1/p
            DiscreteFamily::Geometric { p } => Some((ctx.one() / p).simplify()),
            // r(1−p)/p
            DiscreteFamily::NegativeBinomial { r, p } => Some((r * (ctx.one() - p) / p).simplify()),
            // nm/N
            DiscreteFamily::Hypergeometric {
                population,
                successes,
                draws,
            } => Some((draws * successes / population).simplify()),
            // (a+b)/2
            DiscreteFamily::DiscreteUniform { a, b } => Some(((a + b) / ctx.int(2)).simplify()),
        }
    }

    /// Closed-form variance.
    pub fn variance(&self, ctx: &Context) -> Option<Ex> {
        match self {
            DiscreteFamily::Finite { table } => {
                let mean = table.iter().fold(ctx.zero(), |acc, (v, p)| acc + v * p);
                let second = table
                    .iter()
                    .fold(ctx.zero(), |acc, (v, p)| acc + v.powi(2) * p);
                Some((second - mean.powi(2)).simplify())
            }
            // p(1−p)
            DiscreteFamily::Bernoulli { p } => Some((p * (ctx.one() - p)).simplify()),
            DiscreteFamily::Binomial { n, p } => Some((n * p * (ctx.one() - p)).simplify()),
            DiscreteFamily::Poisson { rate } => Some(rate.clone()),
            // (1−p)/p²
            DiscreteFamily::Geometric { p } => Some(((ctx.one() - p) / p.powi(2)).simplify()),
            // r(1−p)/p²
            DiscreteFamily::NegativeBinomial { r, p } => {
                Some((r * (ctx.one() - p) / p.powi(2)).simplify())
            }
            // n (m/N) ((N−m)/N) ((N−n)/(N−1))
            DiscreteFamily::Hypergeometric {
                population,
                successes,
                draws,
            } => Some(
                (draws
                    * (successes / population)
                    * ((population - successes) / population)
                    * ((population - draws) / (population - ctx.one())))
                .simplify(),
            ),
            // ((b−a+1)² − 1)/12
            DiscreteFamily::DiscreteUniform { a, b } => {
                Some((((b - a + ctx.one()).powi(2) - ctx.one()) / ctx.int(12)).simplify())
            }
        }
    }

    /// Closed-form raw moment `E[Xⁿ]` (`None` leaves it to the generic
    /// summation over the support).
    pub fn raw_moment(&self, n: u32, ctx: &Context) -> Option<Ex> {
        if n == 0 {
            return Some(ctx.one());
        }
        match self {
            DiscreteFamily::Finite { table } => Some(
                table
                    .iter()
                    .fold(ctx.zero(), |acc, (v, p)| acc + v.powi(i64::from(n)) * p)
                    .simplify(),
            ),
            // Xⁿ = X on {0, 1}, so E[Xⁿ] = p.
            DiscreteFamily::Bernoulli { p } => Some(p.clone()),
            // Factorial moments E[X^{(k)}] = n^{(k)} pᵏ.
            DiscreteFamily::Binomial { n: trials, p } => {
                raw_moment_from_factorial_moments(n, ctx, |k| {
                    falling_factorial(trials, k) * p.powi(i64::from(k))
                })
            }
            // Factorial moments E[X^{(k)}] = λᵏ (Touchard polynomial
            // E[Xⁿ] = Σ_k S(n, k) λᵏ).
            DiscreteFamily::Poisson { rate } => {
                raw_moment_from_factorial_moments(n, ctx, |k| rate.powi(i64::from(k)))
            }
            // Elementary mgf: E[Xⁿ] = M⁽ⁿ⁾(0).
            DiscreteFamily::Geometric { .. } | DiscreteFamily::NegativeBinomial { .. } => {
                self.moment_from_mgf(n, ctx)
            }
            // Factorial moments E[X^{(k)}] = n^{(k)} m^{(k)} / N^{(k)}.
            DiscreteFamily::Hypergeometric {
                population,
                successes,
                draws,
            } => raw_moment_from_factorial_moments(n, ctx, |k| {
                falling_factorial(draws, k) * falling_factorial(successes, k)
                    / falling_factorial(population, k)
            }),
            // Σ_{k=a}^{b} kⁿ/(b−a+1) is a Faulhaber sum the generic
            // summation closes.
            DiscreteFamily::DiscreteUniform { .. } => None,
        }
    }

    /// `E[Xⁿ] = M⁽ⁿ⁾(0)`: the `n`-th derivative of the moment generating
    /// function at `t = 0`, simplified.  An exact route for families whose
    /// mgf is elementary; `None` if the family has no closed-form mgf or
    /// the derivative did not evaluate.
    fn moment_from_mgf(&self, n: u32, ctx: &Context) -> Option<Ex> {
        let t = ctx.symbol("_t_mgf");
        let mut m = self.mgf(&t)?;
        for _ in 0..n {
            m = m.diff(&t);
        }
        let at_zero = m.subs(&t, &ctx.zero()).simplify();
        (!at_zero.has_unevaluated() && !at_zero.contains(&t)).then_some(at_zero)
    }

    /// Closed-form CDF `P(X ≤ k)` in `k` (as an expression valid on the
    /// support; `None` leaves it to the generic summation).
    pub fn cdf(&self, k: &Ex) -> Option<Ex> {
        let ctx = k.context();
        match self {
            // Left to the generic route (indicator sum over the table).
            DiscreteFamily::Finite { .. } => None,
            // 0 for k < 0, 1 − p for 0 ≤ k < 1, 1 for k ≥ 1.
            DiscreteFamily::Bernoulli { p } => {
                let zero = ctx.zero();
                let one = ctx.one();
                let q = &one - p;
                Some(Ex::piecewise(&[
                    (&zero, &k.lt(&zero)),
                    (&q, &k.lt(&one)),
                    (&one, &k.ge(&one)),
                ]))
            }
            // Regularised incomplete beta; no elementary closed form.
            DiscreteFamily::Binomial { .. } => None,
            // Γ(⌊k⌋+1, λ) / ⌊k⌋!
            DiscreteFamily::Poisson { rate } => {
                let kf = k.floor();
                Some(rate.uppergamma(&(&kf + ctx.one())) / kf.factorial())
            }
            // 1 − (1−p)^{⌊k⌋}
            DiscreteFamily::Geometric { p } => Some(ctx.one() - (ctx.one() - p).pow(&k.floor())),
            // Regularised incomplete beta I_p(r, ⌊k⌋+1); no elementary
            // closed form.
            DiscreteFamily::NegativeBinomial { .. } => None,
            DiscreteFamily::Hypergeometric { .. } => None,
            // (⌊k⌋ − a + 1)/(b − a + 1)
            DiscreteFamily::DiscreteUniform { a, b } => {
                Some((k.floor() - a + ctx.one()) / (b - a + ctx.one()))
            }
        }
    }

    /// Closed-form moment generating function in `t`.
    pub fn mgf(&self, t: &Ex) -> Option<Ex> {
        let ctx = t.context();
        match self {
            DiscreteFamily::Finite { table } => Some(
                table
                    .iter()
                    .fold(t.context().zero(), |acc, (v, p)| acc + p * (v * t).exp())
                    .simplify(),
            ),
            // 1 − p + p eᵗ
            DiscreteFamily::Bernoulli { p } => Some(ctx.one() - p + p * t.exp()),
            // (1 − p + p eᵗ)ⁿ
            DiscreteFamily::Binomial { n, p } => Some((ctx.one() - p + p * t.exp()).pow(n)),
            // exp(λ(eᵗ − 1))
            DiscreteFamily::Poisson { rate } => Some((rate * (t.exp() - ctx.one())).exp()),
            // p eᵗ / (1 − (1−p) eᵗ)
            DiscreteFamily::Geometric { p } => {
                Some(p * t.exp() / (ctx.one() - (ctx.one() - p) * t.exp()))
            }
            // (p / (1 − (1−p) eᵗ))ʳ
            DiscreteFamily::NegativeBinomial { r, p } => {
                Some((p / (ctx.one() - (ctx.one() - p) * t.exp())).pow(r))
            }
            // A ₂F₁ hypergeometric function; no elementary closed form.
            DiscreteFamily::Hypergeometric { .. } => None,
            // (e^{at} − e^{(b+1)t}) / ((b−a+1)(1 − eᵗ))
            DiscreteFamily::DiscreteUniform { a, b } => Some(
                ((a * t).exp() - ((b + ctx.one()) * t).exp())
                    / ((b - a + ctx.one()) * (ctx.one() - t.exp())),
            ),
        }
    }

    /// Closed-form quantile function at `p` (discrete families rarely have
    /// one).
    pub fn quantile(&self, p: &Ex) -> Option<Ex> {
        let ctx = p.context();
        match self {
            DiscreteFamily::Finite { .. } => None,
            DiscreteFamily::Bernoulli { .. }
            | DiscreteFamily::Binomial { .. }
            | DiscreteFamily::Poisson { .. }
            | DiscreteFamily::Geometric { .. }
            | DiscreteFamily::NegativeBinomial { .. }
            | DiscreteFamily::Hypergeometric { .. } => None,
            // Smallest k in a..=b with (k − a + 1)/(b − a + 1) ≥ p:
            // a + ⌈p (b − a + 1)⌉ − 1.
            DiscreteFamily::DiscreteUniform { a, b } => {
                Some((a + (p * (b - a + ctx.one())).ceiling() - ctx.one()).simplify())
            }
        }
    }
}

impl fmt::Display for DiscreteFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiscreteFamily::Finite { table } => {
                write!(f, "Finite({{")?;
                for (i, (v, p)) in table.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}: {p}")?;
                }
                write!(f, "}})")
            }
            DiscreteFamily::Bernoulli { p } => write!(f, "Bernoulli({p})"),
            DiscreteFamily::Binomial { n, p } => write!(f, "Binomial({n}, {p})"),
            DiscreteFamily::Poisson { rate } => write!(f, "Poisson({rate})"),
            DiscreteFamily::Geometric { p } => write!(f, "Geometric({p})"),
            DiscreteFamily::NegativeBinomial { r, p } => write!(f, "NegativeBinomial({r}, {p})"),
            DiscreteFamily::Hypergeometric {
                population,
                successes,
                draws,
            } => write!(f, "Hypergeometric({population}, {successes}, {draws})"),
            DiscreteFamily::DiscreteUniform { a, b } => write!(f, "DiscreteUniform({a}, {b})"),
        }
    }
}
