//! The boundary to `astro_float::BigFloat`: [`pow`] instead of
//! `BigFloat::pow`, [`to_decimal_digits`] instead of
//! `BigFloat::convert_to_radix`, and [`to_f64`], the one `BigFloat → f64`
//! conversion.
//!
//! astro-float (0.9.6) is the only dependency whose defects reach answers.
//! `BigFloat::pow` and `BigFloat::log` never return when the exact result
//! is representable in binary and the rounding mode is anything but
//! `RoundingMode::None` (`4^(1/2)`, `(1/4)^(−1/2)`, `2.5²`, `log(8, 2)`,
//! `log(9, 3)`; upstream #31, open since 2024-07): their correct-rounding
//! loop adds precision until it can decide a rounding that never becomes
//! decidable.  The decimal conversion had a wrong exponent at 1024 bits
//! (upstream #33), and symplex prints every `evalf` result through it.
//! `clippy.toml` therefore lists `BigFloat::pow`, `BigFloat::log` and
//! `BigFloat::convert_to_radix` under `disallowed-methods`, and this module
//! holds the only `#[allow(clippy::disallowed_methods)]` (on
//! [`to_decimal_digits`]; [`pow`] calls none of them).  The canary crate
//! `probes/clippy_canary` (run by CI, expected to fail) notices a library
//! release under which those paths stop resolving and the lint goes quiet.
//!
//! # Which function
//!
//! | Need | Function |
//! |---|---|
//! | `b^e` for `b > 0`, or any `b` with an integer `e` | [`pow`] |
//! | `log_b(x)` | `x.ln(…) / b.ln(…)` (no wrapper: no caller needs one yet) |
//! | the decimal digits and exponent of a value, for printing | [`to_decimal_digits`] |
//! | the nearest `f64`, when finite | [`to_f64`] |
//!
//! # Corner cases
//!
//! - **Rounding modes.** Measured on 0.9.6 (W4 property test, exact
//!   rational reference): `Up`, `Down`, `ToZero` and `FromZero` are
//!   *directed* roundings (toward `+∞`, `−∞`, zero, away from zero),
//!   although the library documents them as "round half toward …";
//!   `ToEven` and `ToOdd` round to nearest; `None` truncates (for `exp` a
//!   truncated approximation: 3 of 25 000 results lay above the true
//!   value, all within one ulp).  `add`, `sub`,
//!   `mul`, `div`, `sqrt`, `powi` and `reciprocal` were correctly rounded in
//!   every mode (9.8 million cases); `exp`, `ln`, `sin`, `cos` and `atan`
//!   too, against mpmath (0.87 million cases: within half an ulp for
//!   `ToEven`/`ToOdd`, within one ulp on the mode's side for the directed
//!   modes).
//! - **[`pow`] accuracy.** Integer exponents below `10⁹` in magnitude go
//!   through `powi` (and one division for a negative exponent); the rest
//!   through `exp(e·ln b)` with 32 guard bits.  The result is not correctly
//!   rounded: measured within 1.5 ulps under `ToEven`/`ToOdd` and 2 ulps
//!   under the directed modes, *not* always on the mode's side, and an
//!   exactly representable power (`4^(1/2)`) may come back an ulp off.
//!   Callers that need exactness decide it symbolically first; callers that
//!   need a bound widen it themselves.  A negative base with a non-integer
//!   exponent is NaN.
//! - **[`to_f64`].** Rounds to nearest, ties to even, straight from the
//!   binary mantissa; values below the `f64` range go through the
//!   subnormals to `±0.0`.  NaN, `±∞` and values that overflow `f64` are
//!   `None`.
//! - **`BigFloat::from_f64`** halves every subnormal `f64` (0.9.6:
//!   `from_f64(5e-324)` is `2.47e-324`, `from_f64(-1e-310)` is `-5e-311`);
//!   normal values convert exactly.  There is no wrapper yet: pass only
//!   normal values (or zero), or scale by a power of two first.

use astro_float::{BigFloat, Consts, Error, Exponent, Radix, RoundingMode, Sign, WORD_BIT_SIZE};

/// `b^e` for real `b > 0` (or integer `e`), computed as `powi` for integer
/// exponents and `exp(e·ln b)` otherwise, at `wp` bits.
///
/// This deliberately avoids `BigFloat::pow`, whose correct-rounding loop
/// never terminates when the exact result is representable in binary
/// (e.g. `4^{1/2} = 2`), which is exactly what happens for the integer
/// bases used in series like Borwein's `ζ` algorithm.  See the module
/// documentation for the accuracy.
#[inline]
pub(crate) fn pow(
    b: &BigFloat,
    e: &BigFloat,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> BigFloat {
    if e.is_zero() {
        return BigFloat::from_i32(1, wp);
    }
    if e.is_int()
        && let Some(ef) = to_f64(e)
        && ef.abs() < 1.0e9
    {
        let n = ef.abs() as usize;
        let p = b.powi(n, wp, rm);
        return if ef < 0.0 {
            BigFloat::from_i32(1, wp).div(&p, wp, rm)
        } else {
            p
        };
    }
    let ln_b = b.ln(wp + 32, rm, cc);
    ln_b.mul(e, wp + 32, rm).exp(wp, rm, cc)
}

/// The decimal expansion of a finite value: `±0.d₁d₂d₃… × 10^exponent`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DecimalDigits {
    /// The sign of the value.
    pub(crate) sign: Sign,
    /// The digits `d₁, d₂, …`, each `0..=9`; `d₁` is nonzero for a nonzero
    /// value.
    pub(crate) digits: Vec<u8>,
    /// The power of ten of the digit string read as `0.d₁d₂…`.
    pub(crate) exponent: Exponent,
}

/// The decimal digits of `x` (astro-float's `convert_to_radix` with
/// `Radix::Dec`, the single allowed caller).
///
/// # Errors
///
/// astro-float's error: `InvalidArgument` for NaN and `±∞`,
/// `ExponentOverflow` when the decimal exponent does not fit, and
/// `MemoryAllocation`.
// The one call of a primitive `clippy.toml` bans.
#[allow(clippy::disallowed_methods)]
pub(crate) fn to_decimal_digits(
    x: &BigFloat,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<DecimalDigits, Error> {
    let (sign, digits, exponent) = x.convert_to_radix(Radix::Dec, rm, cc)?;
    Ok(DecimalDigits {
        sign,
        digits,
        exponent,
    })
}

/// The `f64` nearest to `x` (ties to even), or `None` when `x` is NaN,
/// `±∞` or beyond the `f64` range.  Zero maps to `+0.0`; a value below the
/// `f64` range rounds through the subnormals to `±0.0`.
pub(crate) fn to_f64(x: &BigFloat) -> Option<f64> {
    if x.is_zero() {
        return Some(0.0);
    }
    // NaN and ±∞ have no raw parts.
    let (words, _, sign, exponent, _) = x.as_raw_parts()?;
    let signed = |v: f64| if sign == Sign::Neg { -v } else { v };

    // The mantissa words are little-endian and the value is `0.m × 2^exponent`,
    // so the leading bit is worth `2^(exponent − 1)`.  Gather the 64 most
    // significant bits (the word size is 32 on wasm32) and remember whether
    // anything below them is set.
    let mut acc: u128 = 0;
    let mut filled = 0usize;
    let mut sticky = false;
    for &w in words.iter().rev() {
        if filled < 64 {
            acc = (acc << WORD_BIT_SIZE) | u128::from(w);
            filled += WORD_BIT_SIZE;
        } else if w != 0 {
            sticky = true;
        }
    }
    let top: u64 = if filled >= 64 {
        let extra = filled - 64;
        if extra > 0 && acc & ((1u128 << extra) - 1) != 0 {
            sticky = true;
        }
        (acc >> extra) as u64
    } else {
        (acc << (64 - filled)) as u64
    };
    if top & (1 << 63) == 0 {
        // Only astro-float's own subnormals (at its minimum exponent, far
        // below the f64 range) lack the leading bit.
        return Some(signed(0.0));
    }

    let mut unbiased = i64::from(exponent) - 1;
    if unbiased > 1023 {
        return None;
    }
    // Significand bits the result can hold: 53 for a normal number, fewer
    // for a subnormal (`2^-1074` is the last representable bit).
    let keep = if unbiased >= -1022 {
        53
    } else {
        unbiased + 1075
    };
    // `mant`: the kept bits; `above_half` / `exactly_half`: the dropped
    // remainder against half a unit in the last kept place.
    let (mant, above_half, exactly_half) = if keep <= 0 {
        // Below the smallest subnormal `2^-1074`; `keep == 0` puts the value
        // in `[2^-1075, 2^-1074)`, at or above half of it.
        let at_half = top == 1 << 63 && !sticky;
        (0u64, keep == 0 && !at_half, keep == 0 && at_half)
    } else {
        let shift = (64 - keep) as u32;
        let mant = top >> shift;
        let rem = top & ((1u64 << shift) - 1);
        let half = 1u64 << (shift - 1);
        (
            mant,
            rem > half || (rem == half && sticky),
            rem == half && !sticky,
        )
    };
    let round_up = above_half || (exactly_half && mant & 1 == 1);
    let mut mant = mant + u64::from(round_up);

    let bits = if keep == 53 {
        if mant == 1 << 53 {
            // Rounded up into the next binade.
            mant >>= 1;
            unbiased += 1;
            if unbiased > 1023 {
                return None;
            }
        }
        (((unbiased + 1023) as u64) << 52) | (mant & ((1 << 52) - 1))
    } else {
        // Subnormal: the value is `mant × 2^-1074` and the bit pattern is
        // `mant` itself (`mant == 2^52` is the smallest normal number).
        mant
    };
    Some(signed(f64::from_bits(bits)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_MODES: [RoundingMode; 7] = [
        RoundingMode::None,
        RoundingMode::Up,
        RoundingMode::Down,
        RoundingMode::ToZero,
        RoundingMode::FromZero,
        RoundingMode::ToEven,
        RoundingMode::ToOdd,
    ];

    fn consts() -> Consts {
        Consts::new().expect("astro-float constants cache")
    }

    /// The results `BigFloat::pow` never returns (`4^(1/2)`, `(1/4)^(−1/2)`,
    /// `2.5²`) come back, within an ulp, under every rounding mode.
    #[test]
    fn pow_terminates_on_exactly_representable_results() {
        let mut cc = consts();
        let p = 128;
        let cases = [
            (4.0, 0.5, 2.0),
            (0.25, -0.5, 2.0),
            (2.5, 2.0, 6.25),
            (9.0, 1.5, 27.0),
        ];
        for rm in ALL_MODES {
            for (b, e, want) in cases {
                let got = pow(
                    &BigFloat::from_f64(b, p),
                    &BigFloat::from_f64(e, p),
                    p,
                    rm,
                    &mut cc,
                );
                let err = got
                    .sub(&BigFloat::from_f64(want, p), p, RoundingMode::None)
                    .abs();
                let tol = BigFloat::from_f64(want * 2f64.powi(-120), p);
                assert!(err <= tol, "{b}^{e} under {rm:?} = {got}");
            }
        }
    }

    #[test]
    fn pow_integer_and_zero_exponents() {
        let mut cc = consts();
        let p = 160;
        let three = BigFloat::from_i32(3, p);
        let r = pow(
            &three,
            &BigFloat::from_i32(5, p),
            p,
            RoundingMode::ToEven,
            &mut cc,
        );
        assert_eq!(r, BigFloat::from_i32(243, p));
        let r = pow(
            &three,
            &BigFloat::from_i32(0, p),
            p,
            RoundingMode::ToEven,
            &mut cc,
        );
        assert_eq!(r, BigFloat::from_i32(1, p));
        let r = pow(
            &BigFloat::from_i32(2, p),
            &BigFloat::from_i32(-3, p),
            p,
            RoundingMode::ToEven,
            &mut cc,
        );
        assert_eq!(r, BigFloat::from_f64(0.125, p));
        // A negative base with an integer exponent keeps its sign.
        let r = pow(
            &BigFloat::from_i32(-2, p),
            &BigFloat::from_i32(3, p),
            p,
            RoundingMode::ToEven,
            &mut cc,
        );
        assert_eq!(r, BigFloat::from_i32(-8, p));
    }

    #[test]
    fn to_decimal_digits_reads_back() {
        let mut cc = consts();
        let d = to_decimal_digits(
            &BigFloat::from_f64(-12.5, 64),
            RoundingMode::ToEven,
            &mut cc,
        )
        .expect("finite value");
        assert_eq!(d.sign, Sign::Neg);
        assert_eq!(d.exponent, 2);
        assert_eq!(&d.digits[..3], &[1, 2, 5]);
        assert!(d.digits[3..].iter().all(|&x| x == 0));
        assert!(
            to_decimal_digits(
                &BigFloat::from_f64(f64::NAN, 64),
                RoundingMode::ToEven,
                &mut cc
            )
            .is_err()
        );
    }

    #[test]
    fn to_f64_round_trips_and_rounds_to_even() {
        // Normal values only: `BigFloat::from_f64` halves subnormals (module docs).
        for x in [1.0, -0.1, 1e300, -1e-300, f64::MAX, f64::MIN_POSITIVE] {
            assert_eq!(to_f64(&BigFloat::from_f64(x, 64)), Some(x), "{x}");
            assert_eq!(to_f64(&BigFloat::from_f64(x, 256)), Some(x), "{x}");
        }
        // Subnormal results, built exactly as 2^-k: 2^-1074 and 2^-1070 are
        // f64 values, 3·2^-1076 rounds up to 2^-1074, the tie 2^-1075 to even 0.
        let p = 128;
        let two = BigFloat::from_i32(2, p);
        let pow2 = |k: usize| {
            BigFloat::from_i32(1, p).div(
                &two.powi(k, p, RoundingMode::ToEven),
                p,
                RoundingMode::ToEven,
            )
        };
        assert_eq!(to_f64(&pow2(1074)), Some(5e-324));
        assert_eq!(to_f64(&pow2(1070)), Some(0.5f64.powi(1070)));
        let three_quarters = BigFloat::from_i32(3, p).mul(&pow2(1076), p, RoundingMode::ToEven);
        assert_eq!(to_f64(&three_quarters), Some(5e-324));
        assert_eq!(to_f64(&pow2(1075)), Some(0.0));
        for x in [5e-324, 0.5f64.powi(1070)] {
            // The library defect itself, so this test notices a fix.
            assert_eq!(to_f64(&BigFloat::from_f64(x, 64)), Some(x / 2.0), "{x}");
        }
        assert_eq!(to_f64(&BigFloat::from_f64(0.0, 64)), Some(0.0));
        assert_eq!(to_f64(&BigFloat::from_f64(f64::NAN, 64)), None);
        assert_eq!(to_f64(&BigFloat::from_f64(f64::INFINITY, 64)), None);
        assert_eq!(to_f64(&BigFloat::from_f64(f64::NEG_INFINITY, 64)), None);
        let p = 128;
        let one = BigFloat::from_i32(1, p);
        let ulp_half = BigFloat::from_f64(2f64.powi(-53), p);
        // 1 + 2⁻⁵³ is a tie between 1 and 1 + 2⁻⁵²: even is 1.
        assert_eq!(
            to_f64(&one.add(&ulp_half, p, RoundingMode::None)),
            Some(1.0)
        );
        // (1 + 2⁻⁵²) + 2⁻⁵³ is a tie: even is 1 + 2·2⁻⁵².
        let odd = BigFloat::from_f64(1.0 + f64::EPSILON, p);
        assert_eq!(
            to_f64(&odd.add(&ulp_half, p, RoundingMode::None)),
            Some(1.0 + 2.0 * f64::EPSILON)
        );
        // Just above the tie rounds up.
        let above = ulp_half.add(
            &BigFloat::from_f64(2f64.powi(-100), p),
            p,
            RoundingMode::None,
        );
        assert_eq!(
            to_f64(&one.add(&above, p, RoundingMode::None)),
            Some(1.0 + f64::EPSILON)
        );
        // Beyond the f64 range: None; far below it: zero with the sign.
        let two = BigFloat::from_i32(2, p);
        assert_eq!(to_f64(&two.powi(1024, p, RoundingMode::ToEven)), None);
        let tiny = one.div(
            &two.powi(1100, p, RoundingMode::ToEven),
            p,
            RoundingMode::ToEven,
        );
        assert_eq!(to_f64(&tiny), Some(0.0));
        assert_eq!(
            to_f64(&tiny.neg()).map(f64::to_bits),
            Some((-0.0f64).to_bits())
        );
    }
}
