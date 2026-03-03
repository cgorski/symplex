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

    /// A feature or operation is not yet implemented.
    #[error("not implemented: {0}")]
    NotImplemented(String),
}
