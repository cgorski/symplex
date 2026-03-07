//! Configuration for controlling evaluation behavior.

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
