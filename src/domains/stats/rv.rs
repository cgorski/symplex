//! A random variable: a symbol with a [`Distribution`].  Expressions in the
//! symbol are random quantities; every query translates them (and events
//! in the symbol) into questions about the distribution.

use std::fmt;

use crate::api::context::Context;
use crate::api::expr::{BoolEx, Ex};
use crate::base::errors::SymplexError;

use super::events::event_region;
use super::family::Distribution;
use super::sample::Rng;
use super::support::Support;

/// A random variable: a symbol with a distribution.  Expressions in the
/// symbol are random quantities; the queries below evaluate them exactly.
///
/// See the [module docs](super) for an overview and examples.
#[derive(Clone, Debug)]
pub struct RandomVariable {
    symbol: Ex,
    dist: Distribution,
}

impl RandomVariable {
    /// A random variable named `name` with distribution `dist`.  The name
    /// becomes a symbol in `ctx`; use [`symbol`](Self::symbol) to build
    /// expressions in it.
    ///
    /// # Panics
    ///
    /// If `name` is empty, or if `dist`'s parameters live in another context
    /// than `ctx` (the crate's cross-context logic error, raised here at
    /// construction rather than deep inside a later query).
    /// [`try_new`](Self::try_new) reports both as errors instead.
    pub fn new(ctx: &Context, name: &str, dist: Distribution) -> Self {
        let symbol = ctx.symbol(name);
        // The cross-context guard: `checked_id` is the crate's one
        // documented panic for this logic error.
        if let Some((_, first)) = dist.parameters().first() {
            let _ = symbol.checked_id(first);
        }
        RandomVariable { symbol, dist }
    }

    /// A random variable, or an error when the name is empty or the
    /// distribution lives in another context than `ctx`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] on an empty name or a context
    /// mismatch.
    pub fn try_new(ctx: &Context, name: &str, dist: Distribution) -> Result<Self, SymplexError> {
        if dist.context().id != ctx.id {
            return Err(SymplexError::invalid_argument(
                "RandomVariable::new",
                "the distribution's parameters live in another context",
            ));
        }
        Ok(RandomVariable {
            symbol: ctx.try_symbol(name)?,
            dist,
        })
    }

    /// The same distribution under another symbol.
    pub(crate) fn with_symbol(symbol: Ex, dist: Distribution) -> Self {
        RandomVariable { symbol, dist }
    }

    /// The variable's symbol.
    pub fn symbol(&self) -> &Ex {
        &self.symbol
    }

    /// The distribution.
    pub fn distribution(&self) -> &Distribution {
        &self.dist
    }

    /// The context the variable lives in.
    pub fn context(&self) -> Context {
        self.symbol.context()
    }

    /// The support of the variable.
    pub fn support(&self) -> Support {
        self.dist.support()
    }

    /// The density (continuous) or probability mass function (discrete) as
    /// an expression in `var`, valid on the support.  SymPy: `density(X)(x)`.
    pub fn density(&self, var: &Ex) -> Ex {
        self.dist.density(var)
    }

    /// `E[X]`.  SymPy: `E(X)`.
    pub fn mean(&self) -> Ex {
        self.dist.mean()
    }

    /// `Var[X]`.  SymPy: `variance(X)`.
    pub fn variance(&self) -> Ex {
        self.dist.variance()
    }

    /// Standard deviation `√Var[X]`.  SymPy: `std(X)`.
    pub fn std(&self) -> Ex {
        self.dist.std()
    }

    /// The `n`-th raw moment `E[Xⁿ]`.  SymPy: `moment(X, n)`.
    pub fn moment(&self, n: u32) -> Ex {
        self.dist.moment(n)
    }

    /// The `n`-th central moment `E[(X − μ)ⁿ]`.  SymPy: `cmoment(X, n)`.
    pub fn central_moment(&self, n: u32) -> Ex {
        self.dist.central_moment(n)
    }

    /// Skewness `E[(X − μ)³] / σ³`.  SymPy: `skewness(X)`.
    pub fn skewness(&self) -> Ex {
        self.dist.skewness()
    }

    /// Kurtosis `E[(X − μ)⁴] / σ⁴` (not excess).  SymPy: `kurtosis(X)`.
    pub fn kurtosis(&self) -> Ex {
        self.dist.kurtosis()
    }

    /// Cumulative distribution function `P(X ≤ var)` as an expression in
    /// `var`, on the whole line (`0` below the support, `1` above it).
    /// SymPy: `cdf(X)(x)`.
    pub fn cdf(&self, var: &Ex) -> Ex {
        self.dist.cdf(var)
    }

    /// Moment generating function `E[e^{tX}]` in `t`.  SymPy:
    /// `moment_generating_function(X)(t)`.
    pub fn mgf(&self, t: &Ex) -> Ex {
        self.dist.mgf(t)
    }

    /// Characteristic function `E[e^{itX}]` in `t`.  SymPy:
    /// `characteristic_function(X)(t)`.
    pub fn characteristic_function(&self, t: &Ex) -> Ex {
        self.dist.characteristic_function(t)
    }

    /// Quantile function (inverse CDF) at `p`, when the family has a
    /// closed form.  SymPy: `quantile(X)(p)`.
    pub fn quantile(&self, p: &Ex) -> Option<Ex> {
        self.dist.quantile(p)
    }

    /// The median, when the quantile function has a closed form.  SymPy:
    /// `median(X)`.
    pub fn median(&self) -> Option<Ex> {
        self.dist.median()
    }

    /// Entropy in nats.  SymPy: `entropy(X)`.
    pub fn entropy(&self) -> Ex {
        self.dist.entropy()
    }

    /// `E[g(X)]` for an expression `g` in the variable's symbol.  SymPy:
    /// `E(expr)`.  See [`Distribution::expectation`].
    pub fn expectation(&self, g: &Ex) -> Ex {
        self.dist.expectation(g, &self.symbol)
    }

    /// The region of the line an event in the variable's symbol describes
    /// (see [`RandomVariable::probability`] for the accepted shapes).
    ///
    /// # Errors
    ///
    /// [`SymplexError::NotImplemented`] for an event of another shape.
    pub fn event_region(&self, event: &BoolEx) -> Result<Support, SymplexError> {
        let _ = self.symbol.checked_id(event);
        event_region(&self.symbol, event)
    }

    /// `P(event)` for an event in the variable's symbol, exactly.
    ///
    /// Accepted events: relations `X < a`, `X ≤ a`, `X > a`, `X ≥ a`,
    /// `X = a` with any bound `a`, and their conjunctions; and any boolean
    /// combination of relations in `X` with *numeric* bounds (`X² < 1`,
    /// `|X| > 2`, `X < −1 ∨ X > 1`), through the crate's inequality solver.
    /// The event's region is clipped to the support piece by piece and
    /// measured through the closed-form CDF when the family has one, else
    /// by exact integration / summation.  SymPy: `P(cond)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::NotImplemented`] for events of another shape
    /// (non-linear or disjunctive with symbolic bounds), or when a listed
    /// value's membership cannot be decided.
    pub fn probability(&self, event: &BoolEx) -> Result<Ex, SymplexError> {
        let region = self.event_region(event)?;
        self.dist.probability_of(&region)
    }

    /// The variable conditioned on an event in its own symbol: the same
    /// symbol with the [`Truncated`](super::Truncated) distribution
    /// `f / P(event)` on the event's region.  SymPy: `given(X, cond)`.
    ///
    /// # Errors
    ///
    /// As [`probability`](Self::probability); additionally
    /// [`SymplexError::InvalidArgument`] when the event has probability
    /// zero.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::{Distribution, RandomVariable};
    ///
    /// let ctx = Context::new();
    /// let x = RandomVariable::new(&ctx, "X", Distribution::normal(ctx.int(0), ctx.int(1)));
    /// let half = x.given(&x.symbol().gt(&ctx.int(0)))?;
    /// // E[X | X > 0] = √(2/π)
    /// assert_eq!(half.mean().equals(&(ctx.int(2) / ctx.pi()).sqrt()), Some(true));
    /// # Ok::<(), SymplexError>(())
    /// ```
    pub fn given(&self, event: &BoolEx) -> Result<RandomVariable, SymplexError> {
        let region = self.event_region(event)?;
        let dist = self.dist.truncated(&region)?;
        Ok(RandomVariable::with_symbol(self.symbol.clone(), dist))
    }

    /// A new variable `name` with the distribution of `g(X)` for an
    /// expression `g` in this variable's symbol (see
    /// [`Distribution::transformed`] for the recognised shapes).  SymPy:
    /// `Y = g(X)` used as a random expression.
    ///
    /// # Errors
    ///
    /// As [`Distribution::transformed`].
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::{Distribution, RandomVariable};
    ///
    /// let ctx = Context::new();
    /// let z = RandomVariable::new(&ctx, "Z", Distribution::normal(ctx.int(0), ctx.int(1)));
    /// let y = z.transform("Y", &(2 * z.symbol() + 1))?;
    /// assert_eq!(y.mean(), ctx.int(1));
    /// assert_eq!(y.variance(), ctx.int(4));
    /// # Ok::<(), SymplexError>(())
    /// ```
    pub fn transform(&self, name: &str, g: &Ex) -> Result<RandomVariable, SymplexError> {
        let _ = self.symbol.checked_id(g);
        let dist = self.dist.transformed(&self.symbol, g)?;
        Ok(RandomVariable::with_symbol(
            self.context().symbol(name),
            dist,
        ))
    }

    /// `n` samples as `f64`; see [`Distribution::sample`].  SymPy:
    /// `sample(X, size=n)`.
    ///
    /// # Errors
    ///
    /// As [`Distribution::sample`].
    pub fn sample(&self, n: usize, rng: &mut Rng) -> Result<Vec<f64>, SymplexError> {
        self.dist.sample(n, rng)
    }

    /// A single sample; see [`sample`](Self::sample).
    ///
    /// # Errors
    ///
    /// As [`sample`](Self::sample).
    pub fn sample_one(&self, rng: &mut Rng) -> Result<f64, SymplexError> {
        self.dist.sample_one(rng)
    }
}

impl fmt::Display for RandomVariable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ~ {}", self.symbol, self.dist)
    }
}
