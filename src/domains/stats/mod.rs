//! Symbolic probability and statistics (SymPy's `stats`): random variables
//! with named distributions, exact moments, probabilities of events,
//! densities and distribution functions — all as expressions.
//!
//! A [`RandomVariable`] is a symbol together with a [`Distribution`].  The
//! queries are exact where the distribution's parameters are exact:
//!
//! ```
//! use symplex::prelude::*;
//! use symplex::stats::{Distribution, RandomVariable};
//!
//! let ctx = Context::new();
//! let x = RandomVariable::new(&ctx, "X", Distribution::normal(ctx.int(0), ctx.int(1)));
//! assert_eq!(x.mean(), ctx.int(0));
//! assert_eq!(x.variance(), ctx.int(1));
//! // E[X² + 3X] for a standard normal.
//! assert_eq!(x.expectation(&(x.symbol().powi(2) + 3 * x.symbol())).simplify(), ctx.int(1));
//!
//! let y = RandomVariable::new(&ctx, "Y", Distribution::binomial(ctx.int(5), ctx.rational(1, 3)));
//! assert_eq!(y.mean(), ctx.rational(5, 3));
//! assert_eq!(y.probability(&y.symbol().gt(&ctx.int(2)))?, ctx.rational(17, 81));
//! # Ok::<(), SymplexError>(())
//! ```
//!
//! # Design
//!
//! * Every distribution family knows its **support**, its **density** (or
//!   probability mass function) as an expression in a free variable, and
//!   closed forms for the moments it has; the generic machinery
//!   ([`RandomVariable::expectation`], [`probability`](RandomVariable::probability))
//!   falls back to the crate's exact [`integrate_definite`](crate::api::expr::Ex::integrate_definite)
//!   / [`summation`](crate::api::expr::Ex::summation) over the support, so
//!   `E[g(X)]` works for any expression `g` the integrator can handle.
//! * Parameters are expressions: rational parameters give exact rational
//!   answers, symbolic parameters give symbolic answers (`E[X] = μ`).
//!   Parameter *validity* (`σ > 0`, `0 ≤ p ≤ 1`) is the caller's promise
//!   for symbolic parameters and is checked for numeric ones by the
//!   constructors that return `Result`.
//! * Nothing here is numerical by default: [`RandomVariable::sample`] is
//!   the only place a random number generator appears, and it is a plain
//!   deterministic-seed generator so tests are reproducible.
//!
//! Distribution families: see [`Distribution`].

mod continuous;
mod discrete;
mod joint;
mod rv;
mod sample;

pub use continuous::*;
pub use discrete::*;
pub use joint::*;
pub use rv::{Distribution, RandomVariable, Support};
pub use sample::Rng;
