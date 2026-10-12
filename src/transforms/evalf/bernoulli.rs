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

/// Largest `|s|` the Euler function is evaluated at (`Γ(s + 1)` and the
/// Hurwitz sums of the order `−s` beyond).
const MAX_EULER_ORDER: f64 = 1.0e6;

/// The Euler function `E(s)` of a real `s` (SymPy's `euler(s)`, the Euler
/// numbers `Eₙ` at the non-negative integers), as SymPy's `euler._eval_evalf`
/// defines it (`sympy/functions/combinatorial/numbers.py`, BSD):
///
/// ```text
/// E(s) = 2^(s+1)·(ζ(−s, ½) − 2^(s+1)·ζ(−s, ¾)),   E(−1) = π/2.
/// ```
///
/// For `s ≥ ½` the Hurwitz functions of the negative order `−s` cancel about
/// `1.8·s` bits; Hurwitz's formula `ζ(−s, a) = 2Γ(s+1)/(2π)^(s+1)·Σₙ
/// cos(π(s+1)/2 − 2πna)/n^(s+1)` (DLMF 25.11.9) turns the combination into
///
/// ```text
/// E(s) = 2·cos(πs/2)·Γ(s+1)/(2π)^(s+1)·(ζ(s+1, ¼) − ζ(s+1, ¾))
/// ```
///
/// (`= 2Γ(s+1)(2/π)^(s+1)·cos(πs/2)·β(s+1)`, Dirichlet's `β`; `E(2) = −1`,
/// `E(4) = 5`), whose Hurwitz sums of positive order do not cancel.  The odd
/// integers are exact zeros (`cos(πs/2)` from an exact reduction), the even
/// ones up to [`crate::base::bernoulli::EXACT_EULER_N`] exact.  The loss of
/// the Hurwitz sums is measured and the evaluation repeated with that many
/// more bits, as for `polylog`.  (Before 0.44 `evalf` had no routine for
/// `euler_number`: `euler_number(1/2)` was refused.)
pub(super) fn euler_function(
    s: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    let unevaluable = |reason: &str| SymplexError::Unevaluable {
        reason: reason.into(),
    };
    if s.is_nan() || s.is_inf() {
        return Err(unevaluable("euler_number of a non-finite argument"));
    }
    let s_f = super::bigfloat_to_f64(s, rm, cc)?;
    if s_f.abs() > MAX_EULER_ORDER {
        return Err(unevaluable(
            "euler_number of an argument beyond 10^6 is not evaluated numerically",
        ));
    }
    if s.is_zero() {
        return Ok(BigFloat::from_i32(1, prec));
    }
    if s.is_int() && !s.is_negative() {
        let n = s_f as usize;
        if n % 2 == 1 {
            return Ok(BigFloat::new(prec));
        }
        if n <= crate::base::bernoulli::EXACT_EULER_N {
            let e = crate::base::combinatorics::euler_even(n / 2);
            return Ok(super::round_to(
                crate::base::numeric::bigint_to_bigfloat(&e, prec + 8),
                prec,
                rm,
            ));
        }
    }
    if *s == BigFloat::from_i32(-1, 64) {
        let pi = cc.pi(prec + 8, rm).clone();
        return Ok(super::round_to(
            pi.div(&BigFloat::from_i32(2, 64), prec + 8, rm),
            prec,
            rm,
        ));
    }
    let cap = super::cancellation_cap(prec);
    let size = s_f.abs() * (s_f.abs().ln().max(0.0) + 2.0) + 2.0;
    let mut extra = (32 + super::magnitude_bits(size)).min(cap);
    loop {
        let wp = prec + extra;
        let (v, largest) = if s_f >= 0.5 {
            euler_positive(s, wp, rm, cc)?
        } else {
            euler_negative(s, wp, rm, cc)?
        };
        // |v| ≥ 2^(e − 1).
        let lv = lg(&v) - 1.0;
        let lost = if lv.is_finite() && largest.is_finite() {
            ((largest - lv).max(0.0).ceil() as usize).min(cap + 1)
        } else {
            wp
        };
        if lost + 24 <= extra {
            return Ok(super::round_to(v, prec, rm));
        }
        if extra >= cap || lost + 24 > cap {
            return Err(super::special_exhausted(prec));
        }
        extra = (lost + 40).max(2 * extra).min(cap);
    }
}

/// An upper bound on `log₂|x|` (its binary exponent), `−∞` at 0.
fn lg(x: &BigFloat) -> f64 {
    if x.is_zero() {
        return f64::NEG_INFINITY;
    }
    x.exponent().map_or(f64::NEG_INFINITY, f64::from)
}

/// `E(s)` for `s ≥ ½` by Hurwitz's formula (see [`euler_function`]) at `wp`
/// bits, and `log₂` of the largest term of the Hurwitz sums scaled as the
/// value.
fn euler_positive(
    s: &BigFloat,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<(BigFloat, f64), SymplexError> {
    let one = BigFloat::from_i32(1, 64);
    let x = s.add(&one, super::exact_bits(s, wp), rm);
    let cos = super::sin_half_pi(&x, wp, rm, cc);
    if cos.is_zero() {
        return Ok((BigFloat::new(wp), f64::NEG_INFINITY));
    }
    let quarter = BigFloat::from_f64(0.25, 64);
    let three_quarters = BigFloat::from_f64(0.75, 64);
    let (h1, l1) = super::polylog::hurwitz_real(&x, &quarter, wp, rm, cc)?;
    let (h2, l2) = super::polylog::hurwitz_real(&x, &three_quarters, wp, rm, cc)?;
    // Γ(s+1)/(2π)^(s+1) = exp(ln Γ(x) − x·ln 2π), the exponent to an absolute
    // error of 2^−wp (the guard bits of `euler_function` cover its size).
    let lw = wp + 16;
    let two_pi = cc
        .pi(lw, rm)
        .clone()
        .mul(&BigFloat::from_i32(2, 64), lw, rm);
    let ln_gamma = super::stirling_log_gamma(&x, lw, rm, cc)?;
    let scale = ln_gamma
        .sub(&x.mul(&two_pi.ln(lw, rm, cc), lw, rm), lw, rm)
        .exp(wp, rm, cc)
        .mul(&cos, wp, rm)
        .mul(&BigFloat::from_i32(2, 64), wp, rm);
    let diff = h1.sub(&h2, wp, rm);
    Ok((scale.mul(&diff, wp, rm), lg(&scale) + l1.max(l2)))
}

/// `E(s)` for `s < ½`, `s ≠ −1`, by SymPy's formula (see
/// [`euler_function`]) at `wp` bits, and `log₂` of the largest term of the
/// Hurwitz sums scaled as the value.
fn euler_negative(
    s: &BigFloat,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<(BigFloat, f64), SymplexError> {
    let exact = super::exact_bits(s, wp);
    let sigma = s.neg();
    let half = BigFloat::from_f64(0.5, 64);
    let three_quarters = BigFloat::from_f64(0.75, 64);
    let (h1, l1) = super::polylog::hurwitz_real(&sigma, &half, wp, rm, cc)?;
    let (h2, l2) = super::polylog::hurwitz_real(&sigma, &three_quarters, wp, rm, cc)?;
    let s1 = s.add(&BigFloat::from_i32(1, 64), exact, rm);
    let p = super::bf_pow(&BigFloat::from_i32(2, 64), &s1, wp + 16, rm, cc);
    let lp = lg(&p);
    let inner = h1.sub(&p.mul(&h2, wp, rm), wp, rm);
    Ok((p.mul(&inner, wp, rm), lp + l1.max(lp + l2)))
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
