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
}
