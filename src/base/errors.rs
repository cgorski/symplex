//! Error types for symplex.

use crate::base::budget::BudgetKind;

/// Errors that can occur during symbolic computation.
///
/// Code that acts on a failure matches its variant, never its message:
/// a budget that ran out is [`BudgetExhausted`](Self::BudgetExhausted), a
/// matrix whose entries outgrew the expression budget is
/// [`ExpressionSwell`](Self::ExpressionSwell), a matrix without a basis
/// of eigenvectors is [`NotDiagonalizable`](Self::NotDiagonalizable).
/// Up to 0.48 these were `ComputationFailed` with a reason the callers
/// searched for `"budget"`, `"expression swell"` or `"not
/// diagonalizable"`; the messages are unchanged.
///
/// ```
/// use symplex::errors::SymplexError;
/// use symplex::prelude::*;
/// use symplex::matrix::Matrix;
///
/// let ctx = Context::new();
/// let m = Matrix::from_i64(&ctx, &[&[1, 1], &[0, 1]]).unwrap();
/// let err = m.diagonalize().unwrap_err();
/// assert!(matches!(err, SymplexError::NotDiagonalizable { .. }));
/// assert!(err.to_string().contains("matrix is not diagonalizable"));
/// let copy = err.clone();
/// assert_eq!(copy.to_string(), err.to_string());
/// ```
#[derive(Clone, Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SymplexError {
    /// Contradictory assumptions were specified for a symbol.
    #[error("contradictory assumptions for '{symbol}': cannot be both {a} and {b}")]
    ContradictoryAssumptions {
        /// The symbol with conflicting assumptions.
        symbol: String,
        /// The first conflicting property.
        a: String,
        /// The second conflicting property.
        b: String,
    },

    /// Numerical evaluation could not achieve the requested precision.
    #[error("precision exhausted: requested {requested} digits, achieved {achieved}")]
    PrecisionExhausted {
        /// Number of digits requested.
        requested: u32,
        /// Number of digits achieved before giving up.
        achieved: u32,
    },

    /// Numerical evaluation failed because the expression contains a free
    /// (unbound) symbol.
    #[error("expression contains free symbol '{name}'")]
    FreeSymbol {
        /// The name of the unbound symbol.
        name: String,
    },

    /// Numerical evaluation failed because a node type cannot be evaluated
    /// to a finite number.
    #[error("cannot evaluate: {reason}")]
    Unevaluable {
        /// Human-readable description of why evaluation failed.
        reason: String,
    },

    /// Numerical evaluation met a value that is infinite: `∞`, `−∞`,
    /// `zoo`, or a function at a pole or singularity of exact arguments
    /// (`elliptic_pi(π, 1)`, `Ci(0)`, `polylog(1, 1)`), in the expression
    /// or inside it (`1/elliptic_pi(π, 1)`).  Up to 0.58 these were
    /// [`Unevaluable`](Self::Unevaluable) like a value that could not be
    /// computed, and `solve(x − elliptic_pi(π, 1), x)` claimed the root
    /// `elliptic_pi(π, 1) = −∞`; the message is unchanged.  A singular
    /// point met only by an argument that rounded onto it is no such claim.
    ///
    /// ```
    /// use symplex::errors::SymplexError;
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let pole = ctx.parse("elliptic_pi(pi, 1)").unwrap();
    /// assert!(matches!(pole.eval_decimal(15), Err(SymplexError::Infinite { .. })));
    /// let near = ctx.parse("elliptic_k(1 - 10^-60)").unwrap();
    /// assert!(near.eval_decimal(15).is_ok());
    /// ```
    #[error("cannot evaluate: {reason}")]
    Infinite {
        /// Which value is infinite.
        reason: String,
    },

    /// A symbolic computation could not produce a result.
    ///
    /// The `operation` field names the operation that failed (e.g., "limit",
    /// "solve", "series"). The `reason` field provides a human-readable
    /// explanation.
    #[error("{operation} could not be computed: {reason}")]
    ComputationFailed {
        /// The name of the operation that failed.
        operation: &'static str,
        /// Human-readable description of why the computation failed.
        reason: String,
    },

    /// A feature or operation is not yet implemented.
    #[error("not implemented: {0}")]
    NotImplemented(String),

    /// An integral, sum, product, or limit was shown to diverge.
    #[error("{operation} diverges: {reason}")]
    Divergent {
        /// The operation (e.g. "integrate", "summation").
        operation: &'static str,
        /// Why / how it diverges (e.g. "integrand has a non-integrable pole at x = 0").
        reason: String,
    },

    /// An equation or system is inconsistent — it has no solution at all.
    ///
    /// Distinct from `Ok(vec![])`, which some solvers use for "no roots in
    /// the requested domain"; this variant means the problem itself is
    /// contradictory (e.g. `x + y = 1, x + y = 2`).
    #[error("{operation}: no solution ({reason})")]
    NoSolution {
        /// The operation that detected the inconsistency.
        operation: &'static str,
        /// Human-readable explanation.
        reason: String,
    },

    /// An equation or system is satisfied by infinitely many values and
    /// the solver cannot (or was not asked to) return a parametric family.
    ///
    /// For example `solve` on the identity `0 = 0`, or an underdetermined
    /// linear system when a unique solution was requested.
    #[error("{operation}: infinitely many solutions ({reason})")]
    InfiniteSolutions {
        /// The operation.
        operation: &'static str,
        /// Human-readable explanation.
        reason: String,
    },

    /// A caller-supplied argument was invalid (wrong length, not a symbol,
    /// out of the function's domain, ...).
    #[error("{operation}: invalid argument: {reason}")]
    InvalidArgument {
        /// The operation that rejected its input.
        operation: &'static str,
        /// What was wrong with it.
        reason: String,
    },

    /// A deterministic work budget ran out before the computation ended
    /// (see [`BudgetKind`]).  The same input gives the same outcome on any
    /// machine and in any context.
    #[error("{operation} could not be computed: {}", kind.reason())]
    BudgetExhausted {
        /// The operation that stopped.
        operation: &'static str,
        /// Which budget ran out.
        kind: BudgetKind,
    },

    /// Intermediate expressions outgrew the expression budget (`nodes`
    /// tree nodes, [`EXPRESSION_BUDGET`](crate::config::EXPRESSION_BUDGET)
    /// for the matrix operations).
    #[error(
        "{operation} could not be computed: expression swell: intermediate result exceeds the \
         budget of {nodes} expression nodes; simplify the input or substitute numeric values \
         first"
    )]
    ExpressionSwell {
        /// The operation that stopped.
        operation: &'static str,
        /// The budget that was exceeded, in expression-tree nodes.
        nodes: usize,
    },

    /// The matrix has no basis of eigenvectors (a Jordan block of size
    /// greater than one), so it is not diagonalizable.
    #[error("{operation} could not be computed: matrix is not diagonalizable: {reason}")]
    NotDiagonalizable {
        /// The operation that needed a diagonalization.
        operation: &'static str,
        /// How the eigenvectors fell short.
        reason: String,
    },

    /// The data are degenerate for the model or statistic asked for: the
    /// estimate or statistic does not exist for them (it is infinite or
    /// undefined), and [`Degeneracy`] says why, so that a pipeline can
    /// react — drop an aliased regressor, switch to a penalised fit, skip a
    /// constant group — without reading the message.  Since 0.57 the
    /// regression, ANOVA, survival and Cox fits report these cases with
    /// this variant (up to 0.56 they were `InvalidArgument` or
    /// `ComputationFailed`; the reasons are unchanged).
    ///
    /// ```
    /// use symplex::errors::{Degeneracy, SymplexError};
    /// use symplex::stats::regression::{logit, LogitOpts};
    ///
    /// // Every x > 0 is a success: the likelihood has no finite maximiser.
    /// let y = [false, false, false, true, true, true];
    /// let x: Vec<Vec<f64>> = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0].iter().map(|&v| vec![v]).collect();
    /// let err = logit(&y, &x, true, &LogitOpts::default()).unwrap_err();
    /// assert!(matches!(err, SymplexError::DegenerateData { kind: Degeneracy::Separation, .. }));
    /// ```
    #[error("{operation}: degenerate data ({}): {reason}", kind.describe())]
    DegenerateData {
        /// The operation that met the data.
        operation: &'static str,
        /// How the data are degenerate.
        kind: Degeneracy,
        /// What exactly was found (which column, group or coefficient).
        reason: String,
    },
}

/// How data are degenerate for a model ([`SymplexError::DegenerateData`]).
///
/// Exact methods (least squares and ANOVA over ℚ, Kaplan–Meier, the
/// log-rank test) decide these exactly.  The `f64` maximum-likelihood
/// fits (logistic and Cox regression) decide [`Collinear`](Self::Collinear)
/// numerically, to the threshold each documents, and
/// [`Separation`](Self::Separation) by the signature of a diverging
/// Newton iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Degeneracy {
    /// The design is rank deficient: a regressor (covariate) is a linear
    /// combination of the others — a duplicated or constant column next to
    /// an intercept, a full set of dummies, a covariate constant within
    /// every risk set — so its coefficient is not identified.
    Collinear,
    /// The likelihood has no finite maximiser, so some coefficient is
    /// infinite: complete or quasi-complete separation of the outcomes in a
    /// logistic model, a monotone partial likelihood in a Cox model.
    Separation,
    /// The response takes a single value: there is no variation to model
    /// (a constant `y` in least squares, a single outcome class).
    ConstantResponse,
    /// A variance the statistic divides by is zero — a perfect fit, groups
    /// without within-group variation, a constant variable — so the
    /// statistic is undefined.
    ZeroVariance,
    /// No events were observed (every subject censored): a survival
    /// comparison or hazard model has no information.
    NoEvents,
}

impl Degeneracy {
    /// A short description, used in the error message.
    #[must_use]
    pub const fn describe(self) -> &'static str {
        match self {
            Degeneracy::Collinear => "collinear design",
            Degeneracy::Separation => "separation",
            Degeneracy::ConstantResponse => "constant response",
            Degeneracy::ZeroVariance => "zero variance",
            Degeneracy::NoEvents => "no events",
        }
    }
}

impl SymplexError {
    /// [`InvalidArgument`](Self::InvalidArgument) for `operation`: a
    /// caller-supplied value that fails validation (a shape, a parameter
    /// out of range, a symbol that is not one of the variables).
    pub fn invalid_argument(operation: &'static str, reason: impl Into<String>) -> Self {
        SymplexError::InvalidArgument {
            operation,
            reason: reason.into(),
        }
    }

    /// [`ComputationFailed`](Self::ComputationFailed) for `operation`: the
    /// algorithm could not complete on valid input (a budget ran out, a
    /// series did not converge, an internal invariant did not hold).
    pub fn computation_failed(operation: &'static str, reason: impl Into<String>) -> Self {
        SymplexError::ComputationFailed {
            operation,
            reason: reason.into(),
        }
    }

    /// [`DegenerateData`](Self::DegenerateData) for `operation`: the data
    /// admit no estimate or statistic of the kind asked for.
    pub fn degenerate(
        operation: &'static str,
        kind: Degeneracy,
        reason: impl Into<String>,
    ) -> Self {
        SymplexError::DegenerateData {
            operation,
            kind,
            reason: reason.into(),
        }
    }

    /// The [`Degeneracy`] of a [`DegenerateData`](Self::DegenerateData)
    /// error, `None` for every other variant.
    #[must_use]
    pub fn degeneracy(&self) -> Option<Degeneracy> {
        match self {
            SymplexError::DegenerateData { kind, .. } => Some(*kind),
            _ => None,
        }
    }
}
