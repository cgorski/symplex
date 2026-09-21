//! Symbolic probability and statistics (SymPy's `stats`): random variables
//! with named distributions, exact moments, probabilities of events,
//! densities and distribution functions — all as expressions — plus
//! conditioning, transformations and mixtures of distributions.
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
//! // Conditioning, and a transformed variable.
//! let half = x.given(&x.symbol().gt(&ctx.int(0)))?;          // X | X > 0
//! assert_eq!(half.mean().equals(&(ctx.int(2) / ctx.pi()).sqrt()), Some(true));
//! let w = x.transform("W", &(2 * x.symbol() + 1))?;          // 2X + 1 ~ Normal(1, 2)
//! assert_eq!(w.variance(), ctx.int(4));
//! # Ok::<(), SymplexError>(())
//! ```
//!
//! # Design
//!
//! * A distribution is a [`Family`]: its **support** ([`Support`], a
//!   finite union of intervals and points, continuous or on the integer
//!   lattice), its **density** (or probability mass function) as an
//!   expression in a free variable, and the **closed forms** it happens to
//!   have (moments, CDF, MGF, quantile, entropy), each optional.  The
//!   generic machinery on [`Distribution`] answers every query from those:
//!   it clips an event's region to the support and measures each piece
//!   through the CDF, or by exact
//!   [`integrate_definite`](crate::api::expr::Ex::integrate_definite) /
//!   [`summation`](crate::api::expr::Ex::summation) of the density, and it
//!   takes `E[g(X)]` through the raw moments when `g` is a polynomial.
//!   Implement [`Family`] to add a distribution of your own.
//! * Families **compose**: [`Truncated`] (conditioning), [`Affine`]
//!   (`aX + b`), [`Transformed`] (`g(X)` by the change-of-variables
//!   formula) and [`Mixture`] wrap other distributions and transport their
//!   closed forms exactly where the transport is exact.
//! * Parameters are expressions: rational parameters give exact rational
//!   answers, symbolic parameters give symbolic answers (`E[X] = μ`).
//!   Parameter *validity* (`σ > 0`, `0 ≤ p ≤ 1`) is the caller's promise
//!   for symbolic parameters and is checked for numeric ones by the
//!   `try_` constructors.
//! * Nothing here is numerical by default: [`Distribution::sample`] is the
//!   only place a random number generator appears, and it is a plain
//!   deterministic-seed generator so tests are reproducible.
//!
//! Distribution families: the structs re-exported below
//! ([`Normal`], [`Binomial`], …); each has a `Distribution::name(…)`
//! constructor and a `try_name` twin.

pub mod aggregation;
pub mod agreement;
pub mod anova;
#[allow(dead_code)] // 0.18 skeleton: every stats module switches to these helpers in the structure pass
pub(crate) mod common;
mod continuous;
pub mod data;
mod discrete;
pub mod estimation;
mod events;
mod family;
pub mod hypothesis;
pub mod information;
mod joint;
pub mod markov;
pub mod multivariate;
pub mod order;
pub mod regression;
pub mod reliability;
mod rv;
mod sample;
pub mod sequential;
mod support;
pub mod survival;
mod wrappers;

pub use continuous::*;
pub use discrete::*;
pub use family::{Distribution, Family, Sampler, same_family};
pub use hypothesis::PValue;
pub use joint::*;
pub use rv::RandomVariable;
pub use sample::Rng;
pub use support::{Kind, Piece, Support};
pub use wrappers::{Affine, Mixture, Transformed, Truncated};
