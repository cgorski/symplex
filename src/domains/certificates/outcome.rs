//! The result type every certificate search returns, and the trait the
//! certificate types have in common.
//!
//! Every prover in this module answers one of three ways, and the answer
//! has the same shape whatever the method: a certificate that was
//! re-verified with exact arithmetic before it was returned, an exact
//! counterexample, or an honest `Unknown` carrying what was tried.
//! [`Outcome`] is that shape; the type aliases
//! [`BoxOutcome`](super::BoxOutcome), [`HalfLineOutcome`](super::HalfLineOutcome),
//! [`PolyhedronOutcome`](super::PolyhedronOutcome) and
//! [`SosOutcome`](super::SosOutcome) fix the certificate and `Unknown`
//! payload types per method, so `PolyhedronOutcome::Proved(c)` reads as
//! before.

use std::fmt;

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::api::poly_ex::Poly;
use crate::base::errors::SymplexError;
use crate::domains::linprog::Q;
use crate::output::lean::LeanOpts;

/// The result of a certificate search: `Proved` with a verified certificate
/// of type `C`, `Refuted` with an exact point where the goal is negative,
/// or `Unknown` with a method-specific account `U` of what was tried.
///
/// `Refuted` is `#[non_exhaustive]`: match it with `Refuted { point, value,
/// .. }` so that a future field does not break the pattern.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::certificates::{prove_nonnegative_on_box, Outcome};
///
/// let ctx = Context::new();
/// let x = ctx.symbol("x");
/// let bounds = [(x.clone(), ctx.int(0), ctx.int(1))];
/// // x·(1 − x) ≥ 0 on [0, 1]: proved; x − 2 ≥ 0 there: refuted at x = 0.
/// assert!(prove_nonnegative_on_box(&(&x * (1 - &x)), &bounds, 2)?.is_proved());
/// match prove_nonnegative_on_box(&(&x - 2), &bounds, 1)? {
///     Outcome::Refuted { point, value, .. } => {
///         assert_eq!(point[0].0, x);
///         assert_eq!(value, symplex::linprog::qi(-2));
///     }
///     other => panic!("{other}"),
/// }
/// # Ok::<(), SymplexError>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome<C, U> {
    /// A certificate, re-verified with exact arithmetic before it was
    /// returned: the goal is non-negative on the set.
    Proved(C),
    /// The goal is negative at this exact point of the set — the claim is
    /// false.
    #[non_exhaustive]
    Refuted {
        /// `(variable, value)` pairs in the prover's generator order; for a
        /// parametric set the parameter is the last entry.
        point: Vec<(Ex, Q)>,
        /// The (negative) value of the goal at `point`.
        value: Q,
        /// The parameter value at which the counterexample lives, when the
        /// set depends on a parameter.
        param_value: Option<Q>,
    },
    /// Neither a certificate nor a counterexample was found within the
    /// search; `U` says how far the search went.  Never a wrong `Proved`.
    Unknown(U),
}

impl<C, U> Outcome<C, U> {
    /// `true` for [`Proved`](Self::Proved).
    pub fn is_proved(&self) -> bool {
        matches!(self, Outcome::Proved(_))
    }

    /// `true` for [`Refuted`](Self::Refuted).
    pub fn is_refuted(&self) -> bool {
        matches!(self, Outcome::Refuted { .. })
    }

    /// `true` for [`Unknown`](Self::Unknown).
    pub fn is_unknown(&self) -> bool {
        matches!(self, Outcome::Unknown(_))
    }

    /// The certificate, if proved.
    pub fn certificate(&self) -> Option<&C> {
        match self {
            Outcome::Proved(c) => Some(c),
            _ => None,
        }
    }

    /// The certificate, if proved, by value.
    pub fn into_certificate(self) -> Option<C> {
        match self {
            Outcome::Proved(c) => Some(c),
            _ => None,
        }
    }

    /// The counterexample `(point, value)`, if refuted.
    pub fn refutation(&self) -> Option<(&[(Ex, Q)], &Q)> {
        match self {
            Outcome::Refuted { point, value, .. } => Some((point.as_slice(), value)),
            _ => None,
        }
    }

    /// The account of the failed search, if unknown.
    pub fn unknown(&self) -> Option<&U> {
        match self {
            Outcome::Unknown(u) => Some(u),
            _ => None,
        }
    }

    /// Apply `f` to the certificate, leaving the other outcomes as they are
    /// — for wrapping one prover's certificate in another's.
    pub fn map_certificate<D>(self, f: impl FnOnce(C) -> D) -> Outcome<D, U> {
        match self {
            Outcome::Proved(c) => Outcome::Proved(f(c)),
            Outcome::Refuted {
                point,
                value,
                param_value,
            } => Outcome::Refuted {
                point,
                value,
                param_value,
            },
            Outcome::Unknown(u) => Outcome::Unknown(u),
        }
    }
}

impl<C: fmt::Display, U: fmt::Display> fmt::Display for Outcome<C, U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Outcome::Proved(c) => write!(f, "proved: {c}"),
            Outcome::Refuted {
                point,
                value,
                param_value,
            } => {
                write!(f, "refuted at (")?;
                for (i, (v, q)) in point.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v} = {q}")?;
                }
                write!(f, "): value {value}")?;
                if let Some(p) = param_value {
                    write!(f, " (parameter {p})")?;
                }
                Ok(())
            }
            Outcome::Unknown(u) => write!(f, "unknown: {u}"),
        }
    }
}

/// What every certificate type offers: the goal it certifies, exact
/// re-verification, Lean export and a JSON round trip.  Implemented by
/// [`BoxCertificate`](super::BoxCertificate),
/// [`HalfLineCertificate`](super::HalfLineCertificate),
/// [`RealLineCertificate`](super::RealLineCertificate),
/// [`PolyhedronCertificate`](super::PolyhedronCertificate) and
/// [`SosCertificate`](super::SosCertificate), whose inherent methods of the
/// same names do the work — the trait lets generic code (a prover that
/// falls back to another, a report that prints whatever it was given)
/// treat them alike.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::certificates::{prove_sos, Certificate, SosOpts};
///
/// fn check<C: Certificate>(c: &C) -> Result<String, SymplexError> {
///     assert!(c.verify());
///     c.to_lean("t")
/// }
///
/// let ctx = Context::new();
/// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
/// let c = prove_sos(&(x.powi(2) + y.powi(2)), &[x, y], &SosOpts::default())?
///     .into_certificate()
///     .expect("a sum of squares");
/// assert!(check(&c)?.contains("positivity"));
/// # Ok::<(), SymplexError>(())
/// ```
pub trait Certificate: fmt::Display {
    /// The polynomial whose non-negativity the certificate proves.
    fn goal(&self) -> &Poly;

    /// Recompute the certified identity with exact arithmetic and check
    /// every side condition (weights non-negative, …).
    fn verify(&self) -> bool;

    /// A Lean 4 / Mathlib theorem named `theorem_name` proving the goal,
    /// with explicit rendering options.
    ///
    /// # Errors
    ///
    /// [`SymplexError::NotImplemented`] if an expression cannot be rendered.
    fn to_lean_with(&self, theorem_name: &str, opts: &LeanOpts) -> Result<String, SymplexError>;

    /// [`to_lean_with`](Self::to_lean_with) with [`LeanOpts::default`].
    ///
    /// # Errors
    ///
    /// As [`to_lean_with`](Self::to_lean_with).
    fn to_lean(&self, theorem_name: &str) -> Result<String, SymplexError> {
        self.to_lean_with(theorem_name, &LeanOpts::default())
    }

    /// The certificate as JSON (exact rationals as `"p/q"` strings).
    ///
    /// # Errors
    ///
    /// [`SymplexError::ComputationFailed`] if serialisation fails.
    fn to_json(&self) -> Result<String, SymplexError>;

    /// Parse a certificate produced by [`to_json`](Self::to_json) into
    /// `ctx`, **re-verifying** it: a certificate that does not check out is
    /// an error, never a value.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for malformed input or a
    /// certificate that fails verification.
    fn from_json(ctx: &Context, json: &str) -> Result<Self, SymplexError>
    where
        Self: Sized;
}
