//! Checked conversions from `f64` to integers: [`f64_to_i64`] for one value,
//! [`integer_span`] for the integers covered by an `f64` range; and lengths
//! read from the data, bounded before they size an allocation:
//! [`bounded_len`], [`clamped_len`] and the caps they are used with.
//!
//! An `as` cast from `f64` to an integer type saturates: `1e19 as i64` is
//! `i64::MAX`, `-1e19 as i64` is `i64::MIN` and `NaN as i64` is `0`.  The
//! `± 1` that usually follows (`t.floor() as i64 - 1`) then wraps in a
//! release build and panics in a debug one, so a range estimated in `f64`
//! silently becomes empty, or `2⁶⁴` long.  0.48 had both outcomes in the
//! breakpoint scan of definite integration: `∫₋₁¹ floor(10¹⁹x)/10¹⁹ dx`
//! returned `0` (the true value is `−10⁻¹⁹`; the loop over the jumps was
//! empty) and `floor(9·10¹⁸x)/10¹⁹` never returned (the length guard
//! `n_max − n_min > 10 000` wrapped negative).  These functions return
//! `None` / an error instead of a saturated value, so the caller has to
//! decide what an out-of-range estimate means.
//!
//! The `tests/unit/test_float_casts.rs` ratchet counts the `as` casts of
//! `floor`/`ceil`/`round`/`trunc` results (and the `(f64 expr) as usize + …`
//! precision pattern) per file under `src/`; new code uses this module
//! instead.
//!
//! A length taken from the data — a degree, an exponent, a valuation, the
//! number of terms of a sum — that sizes a `vec![x; n]`, a
//! `Vec::with_capacity(n)` or a loop that pushes `n` items asks the
//! allocator for whatever the input says: up to 0.56 the dense coefficient
//! vector of `x^(9·10¹⁰)` (a Weibull shape) asked for 360 GB, a series up
//! to the valuation `N` of `exp(−x^N)` for 137 GB, and the process aborted
//! (an allocation failure is not a panic: nothing can catch it).  A cast
//! `n as usize` also wraps a negative `i64` to nearly `2⁶⁴`.  [`bounded_len`]
//! converts such a length and refuses it (`None`) beyond a cap, so the
//! routine returns its refusal — an error, `None`, the unevaluated form —
//! instead.  The `tests/unit/test_alloc_bounds.rs` ratchet counts per file
//! the `with_capacity`/`vec![_; n]`/`resize` calls whose length is a cast
//! outside these functions.
//!
//! # Which function
//!
//! | Need | Function |
//! |---|---|
//! | an integral `f64` (already `floor`ed, `ceil`ed, …) as `i64` | [`f64_to_i64`] |
//! | the integers `⌊t_min⌋ − w ..= ⌈t_max⌉ + w`, refused beyond a length | [`integer_span`] |
//! | a length from the data, refused when negative or beyond a cap | [`bounded_len`] |
//! | a length the caller may lower (a precision, a capacity hint) | [`clamped_len`] |
//!
//! # Caps
//!
//! A cap bounds what one routine allocates, so that an input too large for
//! it is refused before any work.  Domain-specific work bounds stay next
//! to their routine (`summation::MAX_ENUMERATION_TERMS`,
//! `ntheory::MAX_LISTED_ROOTS`, …) and are passed as the cap; the shared
//! ones are here:
//!
//! | Cap | Value | Bounds |
//! |---|---|---|
//! | [`MAX_DENSE_DEGREE`] | `2¹⁶` | the degree of a polynomial in one variable laid out densely by the internals (breakpoints, solver coefficients) |
//! | [`MAX_SERIES_TERMS`] | `2¹⁴` | the coefficients a series stores densely; beyond it the series is exact to a lower order |
//! | [`MAX_TERMS`] | `2²⁰` | a list built item by item from a count in the input (terms of a closed form, coefficients of a Fourier series) |
//! | [`MAX_DENSE_LIST`] | `2²⁴` | a dense list an exact operation the caller asked for builds (`Poly::all_coeffs`, the univariate images of a polynomial gcd) |
//! | [`MAX_POWER_BITS`] | `2¹⁶` | the bits of an exact power `aⁿ` a decider forms (`algebraic::rational_pow`) |
//!
//! # Corner cases
//!
//! - **Non-integral input.** [`f64_to_i64`] is exact: `2.5` is `None`, not
//!   `2`.  Round first, with the rounding the call site means.
//! - **Range.** Every `f64` in `[−2⁶³, 2⁶³)` that is integral converts; `2⁶³`
//!   itself (the `f64` nearest to `i64::MAX`) does not.  `-0.0` is `0`.
//! - **Precision.** An `f64` beyond `2⁵³` cannot tell adjacent integers
//!   apart.  [`integer_span`] does not know how far its arguments are from
//!   the values they estimate: the caller's `widen` must cover that error,
//!   and a caller whose estimate can be off by more (large magnitudes,
//!   cancellation) refuses before calling — as `enumerate_family` in
//!   `api/expr_calculus_util_ext.rs` does beyond `|k| > 10⁹`.
//! - **Reversed range.** `t_min > t_max` (beyond the widening) gives an
//!   empty range, which iterates nothing; it is not an error.
//! - **Lengths.** [`bounded_len`] accepts the integer types and exact
//!   integers ([`DataLen`]): a negative value, a non-integer rational, or a
//!   value beyond the cap is `None`; exactly the cap is allowed.  A cap is
//!   any integer type too (the work bounds of the routines are `i64`, `u32`
//!   or `u64`); a negative cap allows nothing.  [`clamped_len`] (of an
//!   `i64`) never refuses: negative is `0`, beyond the cap is the cap — use
//!   it only where the caller lowers its own bookkeeping to match (a series
//!   lowers its order) or the length is a capacity hint for a loop bounded
//!   elsewhere.

use std::ops::RangeInclusive;

use num_bigint::BigInt;
use num_traits::ToPrimitive;

use crate::base::numeric::Q;

/// The largest degree of a polynomial in one variable that the internals
/// lay out densely (`calculus_util::poly_coeffs_symbolic`,
/// `solve::symbolic_poly_coeffs`): `2¹⁶`.  Up to 0.56 the breakpoint scan of
/// `Weibull(51/8, 9·10¹⁰).mgf` asked for a coefficient vector of `9·10¹⁰`
/// entries and aborted the process.
pub(crate) const MAX_DENSE_DEGREE: usize = 1 << 16;

/// The most coefficients a series stores densely: `2¹⁴`.  Lowering the
/// order keeps a series exact below it; up to 0.56 the valuation `N` of
/// `x^N` (`N = 34278473728`) carried into `exp(−x^N)` made the series
/// engine allocate 137 GB and abort.
pub(crate) const MAX_SERIES_TERMS: usize = 1 << 14;

/// The longest list a routine builds item by item from a count in the
/// input — the terms of a closed form, the coefficients of a Fourier
/// series: `2²⁰`.  Routines with a tighter work bound of their own pass
/// that instead.
pub(crate) const MAX_TERMS: usize = 1 << 20;

/// The longest dense list an exact operation the caller asked for builds:
/// `2²⁴` entries (a few hundred MB) — the coefficients of
/// `Poly::all_coeffs`, the univariate images of `MultiPoly::gcd`.  Larger
/// than [`MAX_DENSE_DEGREE`], which bounds what the internals lay out on
/// their own initiative: `gcd(x^(10⁷) − 1, x − 1)` is a fair request
/// (2 s), `x^(2³¹) − 1` asked for 68 GB up to 0.57.
pub(crate) const MAX_DENSE_LIST: usize = 1 << 24;

/// The most bits an exact power `aⁿ` formed by a decider may have: `2¹⁶`
/// (`algebraic::rational_pow`, with `u32::try_from` for the exponent).  Up
/// to 0.51 the exponent was cast `as u32`: `2^(2³² + 1)` became `2`.
pub(crate) const MAX_POWER_BITS: u64 = 1 << 16;

/// A length, count or degree read from the data: what [`bounded_len`]
/// accepts (the primitive integers, and `&BigInt` / `&Q`).
pub(crate) trait DataLen {
    /// The value when it is a non-negative integer that fits a `u64`.
    fn non_negative(self) -> Option<u64>;
}

macro_rules! data_len_primitive {
    ($($t:ty),*) => {$(
        impl DataLen for $t {
            fn non_negative(self) -> Option<u64> {
                u64::try_from(self).ok()
            }
        }
    )*};
}

data_len_primitive!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);

impl DataLen for &BigInt {
    fn non_negative(self) -> Option<u64> {
        self.to_u64()
    }
}

impl DataLen for &Q {
    fn non_negative(self) -> Option<u64> {
        if self.is_integer() {
            self.numer().to_u64()
        } else {
            None
        }
    }
}

/// `n` as a `usize` when it is a non-negative integer no larger than `cap`;
/// `None` when it is negative, not an integer, or beyond `cap`.
///
/// Use it for every length taken from the data before it sizes an
/// allocation or bounds a loop that pushes: `Vec::with_capacity(n as
/// usize)` asks for whatever the input says (and wraps a negative `i64`).
/// On `None` the routine refuses (an error, `None`, the unevaluated form).
/// `cap` is any integer type, so a routine's own work bound passes as it
/// is; see the module documentation for the shared caps.
pub(crate) fn bounded_len(n: impl DataLen, cap: impl DataLen) -> Option<usize> {
    let n = n.non_negative()?;
    let cap = cap.non_negative()?;
    if n > cap {
        return None;
    }
    usize::try_from(n).ok()
}

/// `n` clamped to `0..=cap`, as a `usize`: negative is `0`, beyond `cap`
/// is `cap`.
///
/// Never refuses, so use it only where the caller lowers its own
/// bookkeeping to the result (a series stores fewer coefficients and
/// lowers its order) or the result is a capacity hint for a loop bounded
/// elsewhere.  A length that must be honoured exactly is [`bounded_len`].
pub(crate) fn clamped_len(n: i64, cap: usize) -> usize {
    usize::try_from(n.max(0)).map_or(cap, |n| n.min(cap))
}

/// `2⁶³`, exactly representable in `f64`: the first value above `i64::MAX`.
const TWO_POW_63: f64 = 9_223_372_036_854_775_808.0;

/// The `i64` equal to `x`, or `None` when `x` is NaN, `±∞`, not an integer,
/// or outside `[−2⁶³, 2⁶³)`.
///
/// Replaces `x as i64`, which saturates (`1e19 → i64::MAX`, `NaN → 0`) and
/// truncates (`2.5 → 2`) without saying so.
pub(crate) fn f64_to_i64(x: f64) -> Option<i64> {
    if x.is_finite() && x.trunc() == x && (-TWO_POW_63..TWO_POW_63).contains(&x) {
        // In range and integral: the cast is exact.
        Some(x as i64)
    } else {
        None
    }
}

/// Why [`integer_span`] refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SpanError {
    /// `t_min` or `t_max` is NaN or `±∞`.
    NotFinite,
    /// A widened end lies outside the `i64` range.
    OutOfRange,
    /// The span holds more than `max_len` integers.
    TooMany,
}

/// The integers `k` with `⌊t_min⌋ − widen ≤ k ≤ ⌈t_max⌉ + widen`, refused when
/// an end is not finite, falls outside `i64`, or the span holds more than
/// `max_len` integers.
///
/// `t_min` and `t_max` are usually `f64` estimates of exact values; `widen`
/// absorbs their error, which is the caller's to bound (see the module
/// documentation, "Precision").  `t_min > t_max` gives an empty range.
pub(crate) fn integer_span(
    t_min: f64,
    t_max: f64,
    widen: u32,
    max_len: u64,
) -> Result<RangeInclusive<i64>, SpanError> {
    if !t_min.is_finite() || !t_max.is_finite() {
        return Err(SpanError::NotFinite);
    }
    let widen = i64::from(widen);
    let lo = f64_to_i64(t_min.floor())
        .and_then(|n| n.checked_sub(widen))
        .ok_or(SpanError::OutOfRange)?;
    let hi = f64_to_i64(t_max.ceil())
        .and_then(|n| n.checked_add(widen))
        .ok_or(SpanError::OutOfRange)?;
    // At most 2⁶⁴ integers: no overflow in i128; ≤ 0 for a reversed range.
    let len = i128::from(hi) - i128::from(lo) + 1;
    if len > i128::from(max_len) {
        return Err(SpanError::TooMany);
    }
    Ok(lo..=hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_to_i64_is_exact_or_none() {
        assert_eq!(f64_to_i64(0.0), Some(0));
        assert_eq!(f64_to_i64(-0.0), Some(0));
        assert_eq!(f64_to_i64(-3.0), Some(-3));
        assert_eq!(f64_to_i64(2.5), None);
        assert_eq!(f64_to_i64(f64::NAN), None);
        assert_eq!(f64_to_i64(f64::INFINITY), None);
        assert_eq!(f64_to_i64(f64::NEG_INFINITY), None);
        // The ends of the range: −2⁶³ converts, 2⁶³ does not.
        assert_eq!(f64_to_i64(-TWO_POW_63), Some(i64::MIN));
        assert_eq!(f64_to_i64(TWO_POW_63), None);
        // The largest f64 below 2⁶³.
        let below = f64::from_bits(TWO_POW_63.to_bits() - 1);
        assert_eq!(f64_to_i64(below), Some(9_223_372_036_854_774_784));
        assert_eq!(f64_to_i64(1e19), None);
        assert_eq!(f64_to_i64(-1e19), None);
    }

    #[test]
    fn integer_span_widens_and_refuses() {
        assert_eq!(integer_span(0.5, 2.5, 1, 100), Ok(-1..=4));
        assert_eq!(integer_span(-2.0, 2.0, 0, 100), Ok(-2..=2));
        // Exactly `max_len` integers is allowed, one more is not.
        assert_eq!(integer_span(0.0, 9.0, 0, 10), Ok(0..=9));
        assert_eq!(integer_span(0.0, 10.0, 0, 10), Err(SpanError::TooMany));
        assert_eq!(
            integer_span(f64::NAN, 1.0, 1, 10),
            Err(SpanError::NotFinite)
        );
        assert_eq!(
            integer_span(0.0, f64::INFINITY, 1, 10),
            Err(SpanError::NotFinite)
        );
        // S4-B1: ⌊−10¹⁹⌋ is outside i64 (`as i64` saturated, `− 1` wrapped).
        assert_eq!(
            integer_span(-1e19, 1e19, 1, 10_001),
            Err(SpanError::OutOfRange)
        );
        // S4-B2: both ends fit, the length 1.8·10¹⁹ does not wrap here.
        assert_eq!(
            integer_span(-9e18, 9e18, 1, 10_001),
            Err(SpanError::TooMany)
        );
        // The widening itself must not overflow.
        assert_eq!(
            integer_span(-TWO_POW_63, 0.0, 1, u64::MAX),
            Err(SpanError::OutOfRange)
        );
        let full = integer_span(-TWO_POW_63, 9_223_372_036_854_774_784.0, 0, u64::MAX);
        assert_eq!(full, Ok(i64::MIN..=9_223_372_036_854_774_784));
        // A reversed range is empty, not an error.
        assert_eq!(integer_span(5.5, 1.5, 0, 10).map(Iterator::count), Ok(0));
    }

    #[test]
    fn bounded_len_refuses_negative_fractional_and_beyond_cap() {
        assert_eq!(bounded_len(0usize, 10usize), Some(0));
        assert_eq!(bounded_len(10u64, 10usize), Some(10));
        assert_eq!(bounded_len(11u32, 10usize), None);
        // `-1 as usize` is 2⁶⁴ − 1: refused, not wrapped.
        assert_eq!(bounded_len(-1i64, 10usize), None);
        assert_eq!(bounded_len(i64::MIN, u64::MAX), None);
        assert_eq!(bounded_len(u128::MAX, u64::MAX), None);
        // A routine's own bound of another integer type is the cap.
        assert_eq!(bounded_len(64i64, 64i64), Some(64));
        assert_eq!(bounded_len(65i64, 64i64), None);
        assert_eq!(bounded_len(1u8, -1i64), None);
        // Exact integers: `&BigInt`, and `&Q` only when integral.
        let big = BigInt::from(1u64 << 40);
        assert_eq!(bounded_len(&big, MAX_TERMS), None);
        assert_eq!(bounded_len(&BigInt::from(7), MAX_TERMS), Some(7));
        assert_eq!(bounded_len(&-BigInt::from(7), MAX_TERMS), None);
        let q = Q::from_integer(BigInt::from(5));
        assert_eq!(bounded_len(&q, MAX_TERMS), Some(5));
        let half = Q::new(BigInt::from(1), BigInt::from(2));
        assert_eq!(bounded_len(&half, MAX_TERMS), None);
        let neg = Q::from_integer(BigInt::from(-5));
        assert_eq!(bounded_len(&neg, MAX_TERMS), None);
        // 2³² + 1, which `as u32` read as 1.
        assert_eq!(bounded_len((1u64 << 32) + 1, MAX_DENSE_DEGREE), None);
    }

    #[test]
    fn clamped_len_saturates() {
        assert_eq!(clamped_len(-5, 10), 0);
        assert_eq!(clamped_len(0, 10), 0);
        assert_eq!(clamped_len(7, 10), 7);
        assert_eq!(clamped_len(i64::MAX, MAX_SERIES_TERMS), MAX_SERIES_TERMS);
        assert_eq!(clamped_len(i64::MIN, MAX_SERIES_TERMS), 0);
    }
}
