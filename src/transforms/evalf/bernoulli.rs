//! The even Bernoulli numbers `B₂ₖ` of the asymptotic series (`ln Γ`, `ψ`,
//! `ψ⁽ⁿ⁾` by Stirling's series), from the tangent numbers.
//!
//! The series need `B₂ₖ` up to `k ≈ 0.1·p` at a working precision of `p`
//! bits, and a value that is zero to the precision is pursued to about
//! 3,200 bits (`ZERO_SEARCH_BITS`): some 350 numbers.  The exact
//! rational recurrence of `base::bernoulli` (`B_m = −Σ C(m+1, k)·B_k/(m+1)`,
//! a sum of rationals per number) took 5–8 s to reach them.  The tangent
//! numbers `T₁ = 1, T₂ = 2, T₃ = 16, …` need only integer updates by small
//! factors, `O(n²)` of them for `T₁ … Tₙ` (Brent and Harvey, "Fast
//! computation of Bernoulli, Tangent and Secant numbers", 2011,
//! Algorithm TangentNumbers — the published algorithm, not code), and
//!
//! ```text
//! B₂ₖ = (−1)^(k−1) · 2k · Tₖ / (2^(2k) · (2^(2k) − 1)).
//! ```
//!
//! The exact evaluator's `bernoulli(n)` (`eval::exact_bernoulli`) takes its
//! even numbers from here too.  Since 0.40 the algorithm lives in
//! `base::bernoulli`, which every layer reaches (`ntheory::bernoulli`,
//! series and summation used the rational recurrence until then): one
//! implementation and one cache.

use astro_float::{BigFloat, Consts, RoundingMode};

use crate::base::errors::SymplexError;
use crate::base::numeric::Q;

/// `B₂ₖ` for `k ≥ 1` (`B₀ = 1` for `k = 0`), from the tangent numbers
/// ([`crate::base::bernoulli::even`]).
pub(crate) fn even(k: usize) -> Q {
    crate::base::bernoulli::even(k)
}

/// The Bernoulli function `B(s) = −s·ζ(1 − s)` of a real `s` (SymPy's
/// `bernoulli(s)` off the integers, which `evalf`s it so: `bernoulli(1/2)
/// = 0.730177…`, `bernoulli(−1) = ζ(2)`).  It is the integer `Bₙ` at
/// `n ≥ 2` and `n = 0`; at `s = 1` it is `+1/2`, and the exact `−1/2` of
/// this crate's convention is returned instead, as `eval` folds it.
/// (Before 0.40 `evalf` had no routine for `bernoulli`.)
pub(super) fn bernoulli_function(
    s: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    if s.is_nan() || s.is_inf() {
        return Err(SymplexError::Unevaluable {
            reason: "bernoulli of a non-finite argument".into(),
        });
    }
    if s.is_zero() {
        return Ok(BigFloat::from_i32(1, prec));
    }
    if *s == BigFloat::from_i32(1, 64) {
        return Ok(BigFloat::from_i32(-1, prec).div(&BigFloat::from_i32(2, prec), prec, rm));
    }
    // `1 − s` exactly: next to `s = 0` the pole of `ζ` at 1 is cancelled
    // by the factor `s`, both relative to `s`.
    let wp = prec + 32;
    let one_minus_s = BigFloat::from_i32(1, 64).sub(s, super::exact_bits(s, wp), rm);
    let zeta = super::arb_zeta(&one_minus_s, wp, rm, cc)?;
    let r = s.mul(&zeta, wp, rm).neg();
    Ok(super::round_to(r, prec, rm))
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_rational::Ratio;

    /// The first even Bernoulli numbers.
    #[test]
    fn first_values() {
        let q = |n: i64, d: i64| Ratio::new(BigInt::from(n), BigInt::from(d));
        assert_eq!(even(1), q(1, 6));
        assert_eq!(even(2), q(-1, 30));
        assert_eq!(even(3), q(1, 42));
        assert_eq!(even(6), q(691, -2730));
    }
}
