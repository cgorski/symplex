//! Checked conversions from `f64` to integers: [`f64_to_i64`] for one value,
//! [`integer_span`] for the integers covered by an `f64` range.
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
//! # Which function
//!
//! | Need | Function |
//! |---|---|
//! | an integral `f64` (already `floor`ed, `ceil`ed, …) as `i64` | [`f64_to_i64`] |
//! | the integers `⌊t_min⌋ − w ..= ⌈t_max⌉ + w`, refused beyond a length | [`integer_span`] |
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

use std::ops::RangeInclusive;

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
}
