//! Exact decimals kept unscaled: the digits `evalf` prints, without the
//! string.
//!
//! A [`Decimal`] is the exact value `±digits·10^exp` of a decimal number,
//! with an integer of digits and an `i64` exponent that is never multiplied
//! out.  Signs, magnitudes and comparisons are read from the digits and the
//! exponent; the exact rational is formed only on request and only for an
//! exponent within a cap the caller names ([`Decimal::to_ratio`]).
//!
//! # Design
//!
//! * **Built from the value, not from its printout.**  [`Decimal::of`]
//!   rounds a `BigFloat` part to the significant digits [`evalf`] prints,
//!   with the printer's own exact conversion and rounding
//!   (`bigfloat::to_decimal_digits`, then `round_decimal_digits`), so a
//!   decision on the digits is the decision a parse of the printed number
//!   made — without formatting and re-reading a string.  Up to 0.57 the
//!   inequality solver parsed `evalf` strings, and before 0.56 it turned
//!   them into exact rationals: the 30-digit value of `exp(1000³) ≈
//!   10^434294481` became a power of ten that never finished, and `exp(x³) ≥
//!   0` hung.
//! * **Literals are read with a cap on the written exponent**
//!   ([`Decimal::parse`]): `Context::decimal_str("1e999999999")` formed
//!   `10^999999999` before 0.58.
//! * **Comparisons are exact.**  [`Decimal::differs_relatively`] decides
//!   magnitudes two or more apart at once and aligns the digits only when
//!   they are close (the exponents then differ by about the difference of
//!   the digit counts).
//!
//! The numeric value of a constant for a decision is
//! [`ConstValue`](super::ConstValue); its decimal views are `Decimal`s.
//!
//! [`evalf`]: super::evalf

use astro_float::{BigFloat, Sign};
use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{Signed, Zero};

use crate::base::numeric::Q;

/// The exact value `±digits·10^exp` of a decimal number, kept unscaled (see
/// the module documentation).  Normalized: no trailing zeros in the digits,
/// and 0 is `+0·10⁰`, so equal values are equal decimals.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Decimal {
    negative: bool,
    /// Positive without trailing zeros; 0 for the value 0.
    digits: BigInt,
    exp: i64,
}

impl Decimal {
    /// The value 0.
    pub(crate) fn zero() -> Decimal {
        Decimal {
            negative: false,
            digits: BigInt::zero(),
            exp: 0,
        }
    }

    /// `x` rounded to `digits` significant digits (ties to even), exactly
    /// the digits [`evalf`](super::evalf) prints for it; `None` for NaN, an
    /// infinity, or a value whose decimal exponent does not fit.
    pub(crate) fn of(x: &BigFloat, digits: u32) -> Option<Decimal> {
        if x.is_nan() || x.is_inf() {
            return None;
        }
        if x.is_zero() {
            return Some(Decimal::zero());
        }
        let d = crate::base::bigfloat::to_decimal_digits(x).ok()?;
        let (rounded, carry) =
            super::round_decimal_digits(&d.digits, usize::try_from(digits).ok()?);
        // `0.r₁r₂…rₙ · 10^(exponent + carry)`, trailing zeros dropped.
        let kept = rounded.iter().rposition(|&r| r != 0).map_or(0, |p| p + 1);
        if kept == 0 {
            return Some(Decimal::zero());
        }
        let n = BigInt::from_radix_be(num_bigint::Sign::Plus, &rounded[..kept], 10)?;
        let exp = i64::from(d.exponent)
            .checked_add(carry)?
            .checked_sub(i64::try_from(kept).ok()?)?;
        Some(Decimal {
            negative: d.sign == Sign::Neg,
            digits: n,
            exp,
        })
    }

    /// The exact value of a decimal literal `[+|-] digits [. digits] [(e|E)
    /// [+|-] digits]` (surrounding whitespace ignored); `None` when `text` is
    /// not one or its written exponent exceeds `max_exponent` in absolute
    /// value.
    pub(crate) fn parse(text: &str, max_exponent: u64) -> Option<Decimal> {
        let t = text.trim();
        let (negative, t) = match t.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, t.strip_prefix('+').unwrap_or(t)),
        };
        let (mantissa, exp) = match t.find(['e', 'E']) {
            Some(pos) => (&t[..pos], t[pos + 1..].parse::<i64>().ok()?),
            None => (t, 0i64),
        };
        if exp.unsigned_abs() > max_exponent {
            return None;
        }
        let (int_part, frac_part) = match mantissa.split_once('.') {
            Some((i, f)) => (i, f),
            None => (mantissa, ""),
        };
        if int_part.is_empty() && frac_part.is_empty() {
            return None;
        }
        if !int_part.bytes().all(|b| b.is_ascii_digit())
            || !frac_part.bytes().all(|b| b.is_ascii_digit())
        {
            return None;
        }
        let mut digits: Vec<u8> = int_part
            .bytes()
            .chain(frac_part.bytes())
            .map(|b| b - b'0')
            .collect();
        // Trailing zeros go into the exponent (as in [`Decimal::of`]), so
        // equal values are equal decimals.
        let zeros = digits.iter().rev().take_while(|&&d| d == 0).count();
        digits.truncate(digits.len() - zeros);
        if digits.is_empty() {
            return Some(Decimal::zero());
        }
        let n = BigInt::from_radix_be(num_bigint::Sign::Plus, &digits, 10)?;
        let exp = exp
            .checked_sub(i64::try_from(frac_part.len()).ok()?)?
            .checked_add(i64::try_from(zeros).ok()?)?;
        Some(Decimal {
            negative,
            digits: n,
            exp,
        })
    }

    /// Is the value 0?
    pub(crate) fn is_zero(&self) -> bool {
        self.digits.is_zero()
    }

    /// Is the value negative?
    pub(crate) fn is_negative(&self) -> bool {
        self.negative && !self.is_zero()
    }

    /// The exponent `e` of `±digits·10^e`: `10^e` is a unit in the last
    /// nonzero digit (`1` for the value 0).
    pub(crate) fn exponent(&self) -> i64 {
        self.exp
    }

    /// `⌊log₁₀ |v|⌋` of a nonzero value (`|v| ∈ [10^m, 10^(m+1))`).
    pub(crate) fn magnitude(&self) -> i64 {
        let len = i64::try_from(self.digits.to_string().len()).unwrap_or(i64::MAX);
        self.exp.saturating_add(len).saturating_sub(1)
    }

    /// `(a, b)` with `a/b = |self|/|other|` up to a common power of ten.
    /// Called only for magnitudes at most one apart, where the exponents
    /// differ by at most the difference of the digit counts plus one, so the
    /// scale is small.
    fn aligned(&self, other: &Decimal) -> Option<[BigInt; 2]> {
        let shift = usize::try_from(self.exp.abs_diff(other.exp)).ok()?;
        let scale = num_traits::pow(BigInt::from(10), shift);
        Some(if self.exp >= other.exp {
            [&self.digits * scale, other.digits.clone()]
        } else {
            [self.digits.clone(), &other.digits * scale]
        })
    }

    /// `|self − other| > |self|·10^(−k)` for two nonzero values of one
    /// sign.  Magnitudes two or more apart differ by more than a tenth of
    /// the larger value.
    pub(crate) fn differs_relatively(&self, other: &Decimal, k: usize) -> bool {
        if self.magnitude().abs_diff(other.magnitude()) >= 2 {
            return true;
        }
        match self.aligned(other) {
            Some([a, b]) => {
                let diff = (&a - &b).abs();
                diff * num_traits::pow(BigInt::from(10), k) > a
            }
            None => true,
        }
    }

    /// The exact rational value, or `None` when `|exp| > max_exp`: the
    /// power of ten of a value at a far sample point cannot be formed
    /// (`exp(10⁹)` has a decimal exponent of 4.3·10⁸).
    pub(crate) fn to_ratio(&self, max_exp: u64) -> Option<Q> {
        if self.exp.unsigned_abs() > max_exp {
            return None;
        }
        let pow = num_traits::pow(
            BigInt::from(10),
            usize::try_from(self.exp.unsigned_abs()).ok()?,
        );
        let r = if self.exp >= 0 {
            Ratio::from_integer(&self.digits * pow)
        } else {
            Ratio::new(self.digits.clone(), pow)
        };
        Some(if self.negative { -r } else { r })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(p: i64, q: i64) -> Q {
        Ratio::new(BigInt::from(p), BigInt::from(q))
    }

    fn lit(s: &str) -> Decimal {
        Decimal::parse(s, u64::MAX).unwrap()
    }

    #[test]
    fn literals_are_exact() {
        let cases = [
            ("1.25", r(5, 4)),
            ("-0.5", r(-1, 2)),
            ("3", r(3, 1)),
            ("1.5e2", r(150, 1)),
            ("-12.5e-3", r(-1, 80)),
            (".5", r(1, 2)),
            ("+2.", r(2, 1)),
            ("-0", r(0, 1)),
        ];
        for (s, want) in cases {
            assert_eq!(lit(s).to_ratio(100), Some(want), "{s}");
        }
        for bad in ["abc", "", ".", "1e", "1e+", "1/2", "1.2.3", "--1", "1e5.5"] {
            assert!(Decimal::parse(bad, u64::MAX).is_none(), "{bad:?}");
        }
        assert!(!lit("-0").is_negative());
    }

    #[test]
    fn written_exponents_beyond_the_cap_are_refused() {
        assert!(Decimal::parse("1e10000", 10_000).is_some());
        assert!(Decimal::parse("1e10001", 10_000).is_none());
        assert!(Decimal::parse("1e-10001", 10_000).is_none());
        assert!(Decimal::parse("1e99999999999999999999", u64::MAX).is_none());
        // Nothing is formed until asked for, and then only within the cap.
        let huge = lit("1.6e434294481");
        assert_eq!(huge.magnitude(), 434_294_481);
        assert!(huge.to_ratio(1_000_000).is_none());
    }

    #[test]
    fn digits_of_a_bigfloat_are_the_printed_ones() {
        let rm = astro_float::RoundingMode::ToEven;
        let third = BigFloat::from_i32(1, 128).div(&BigFloat::from_i32(3, 128), 128, rm);
        assert_eq!(Decimal::of(&third, 5), Some(lit("0.33333")));
        let neg = third.neg();
        assert_eq!(Decimal::of(&neg, 3), Some(lit("-0.333")));
        // A carry out of the leading digit and trailing zeros.
        let x = crate::base::bigfloat::from_f64(9.9996, 128);
        assert_eq!(Decimal::of(&x, 3), Some(lit("10")));
        assert_eq!(
            Decimal::of(&BigFloat::from_i32(0, 128), 30),
            Some(Decimal::zero())
        );
        assert_eq!(
            Decimal::of(&BigFloat::from_i32(1500, 128), 30),
            Some(lit("1.5e3"))
        );
        assert!(Decimal::of(&BigFloat::nan(None), 30).is_none());
    }

    #[test]
    fn huge_decimals_compare_without_their_powers() {
        let big = lit("1.6e434294481");
        assert!(big.differs_relatively(&lit("2.0e434294400"), 6));
        assert!(!big.differs_relatively(&lit("1.6000000001e434294481"), 6));
        assert!(lit("1.000001e-30").differs_relatively(&lit("1e-30"), 7));
        assert!(!lit("1.0000001e-30").differs_relatively(&lit("1e-30"), 6));
        assert!(lit("-7e-3").is_negative());
        assert_eq!(lit("0.00150").magnitude(), -3);
    }
}
