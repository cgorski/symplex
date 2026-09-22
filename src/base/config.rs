//! Configuration for controlling evaluation behavior.

/// Upper bound on the total expression-tree size (nodes, counted without
/// sharing) of the operands and intermediate results of the symbolic
/// algorithms that can swell: [`Matrix::det`], [`Matrix::inv`],
/// [`Matrix::solve`], [`Matrix::diagonalize`], [`Matrix::jordan_form`],
/// [`Matrix::matrix_exp`] and [`Matrix::qr`], and (at a quarter of it) the
/// closed forms of `rsolve`.
///
/// When the budget is exceeded these return
/// [`SymplexError::ComputationFailed`](crate::base::errors::SymplexError::ComputationFailed)
/// whose reason starts with `"expression swell"` instead of running for an
/// unbounded time.  The value is calibrated so that everything below it
/// finishes in well under a minute: a fully symbolic 7×7 determinant (5040
/// terms, ≈ 40 000 nodes) passes, an 8×8 one (≈ 360 000 nodes, many
/// minutes) is rejected.
///
/// [`Matrix::det`]: crate::matrix::Matrix::det
/// [`Matrix::inv`]: crate::matrix::Matrix::inv
/// [`Matrix::solve`]: crate::matrix::Matrix::solve
/// [`Matrix::diagonalize`]: crate::matrix::Matrix::diagonalize
/// [`Matrix::jordan_form`]: crate::matrix::Matrix::jordan_form
/// [`Matrix::matrix_exp`]: crate::matrix::Matrix::matrix_exp
/// [`Matrix::qr`]: crate::matrix::Matrix::qr
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
///
/// let ctx = Context::new();
/// // A tiny DAG whose *tree* is enormous: eₙ₊₁ = sin(eₙ) + cos(eₙ).
/// let mut e = ctx.symbol("x");
/// for _ in 0..16 {
///     e = &e.sin() + &e.cos();
/// }
/// let m = Matrix::new(vec![vec![e.clone(), ctx.int(1)], vec![ctx.int(1), e]]).unwrap();
/// let err = m.inv().unwrap_err();
/// assert!(err.to_string().contains("expression swell"), "{err}");
/// ```
pub const EXPRESSION_BUDGET: usize = 100_000;

/// Controls the bounds on automatic evaluation in constructors.
///
/// These guards prevent runaway computation from pathological inputs
/// like `7^(10^100)` which would allocate astronomical amounts of memory.
#[derive(Debug, Clone)]
pub struct EvalConfig {
    /// Maximum integer exponent for automatic evaluation of `pow()`.
    ///
    /// If the exponent exceeds this, `pow()` returns an unevaluated `Pow` node
    /// instead of computing the result.
    ///
    /// Default: 1000.
    pub max_pow_exponent: usize,

    /// Maximum digits in a numeric result before we refuse to auto-evaluate.
    ///
    /// Default: 5000.
    pub max_result_digits: usize,

    /// Maximum working precision (in bits) for `evalf()` before giving up.
    ///
    /// Default: 10,000.
    pub max_evalf_precision: u32,
}

impl Default for EvalConfig {
    fn default() -> Self {
        EvalConfig {
            max_pow_exponent: 1000,
            max_result_digits: 5000,
            max_evalf_precision: 10_000,
        }
    }
}
