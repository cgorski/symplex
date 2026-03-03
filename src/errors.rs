//! Error types for symplex.

/// Errors that can occur during symbolic computation.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SymplexError {
    /// Contradictory assumptions were specified for a symbol.
    #[error("contradictory assumptions for '{symbol}': cannot be both {a} and {b}")]
    ContradictoryAssumptions {
        symbol: String,
        a: String,
        b: String,
    },

    /// Numerical evaluation could not achieve the requested precision.
    #[error("precision exhausted: requested {requested} digits, achieved {achieved}")]
    PrecisionExhausted { requested: u32, achieved: u32 },

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

    /// A feature or operation is not yet implemented.
    #[error("not implemented: {0}")]
    NotImplemented(String),
}
