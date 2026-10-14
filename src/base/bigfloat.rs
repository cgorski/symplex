//! The boundary to `astro_float::BigFloat`: [`pow`] instead of
//! `BigFloat::pow`, [`to_decimal_digits`] instead of
//! `BigFloat::convert_to_radix`, [`to_f64`] (and [`to_f64_or_inf`]), the one
//! `BigFloat → f64` conversion, [`from_f64`] instead of
//! `BigFloat::from_f64` for values that are not literals, and
//! [`is_strictly_positive`] / [`is_sign_positive`] instead of
//! `BigFloat::is_positive`.
//!
//! astro-float (0.9.6) is the only dependency whose defects reach answers.
//! `BigFloat::pow` and `BigFloat::log` never return when the exact result
//! is representable in binary and the rounding mode is anything but
//! `RoundingMode::None` (`4^(1/2)`, `(1/4)^(−1/2)`, `2.5²`, `log(8, 2)`,
//! `log(9, 3)`; upstream #31, open since 2024-07): their correct-rounding
//! loop adds precision until it can decide a rounding that never becomes
//! decidable.  The decimal conversion had a wrong exponent at 1024 bits
//! (upstream #33), and symplex prints every `evalf` result through
//! [`to_decimal_digits`], which therefore converts exactly from the binary
//! mantissa itself.  `BigFloat::is_positive` reads only the sign bit, so it
//! is true for `+0` (`evalf` once decided `a > a` that way).  `clippy.toml`
//! lists `BigFloat::pow`, `BigFloat::log`, `BigFloat::convert_to_radix` and
//! `BigFloat::is_positive` under `disallowed-methods`; no module of the
//! library calls them (the tests compare [`to_decimal_digits`] with
//! `convert_to_radix` under an `#[allow]`).  The canary crate
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
//! | the nearest `f64`, when finite | [`to_f64`]; `±∞` beyond the range: [`to_f64_or_inf`] |
//! | a computed `f64` as a `BigFloat` | [`from_f64`] |
//! | `x > 0` | [`is_strictly_positive`]; the sign bit ("not negative", `+0` included): [`is_sign_positive`] |
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
//! - **[`to_decimal_digits`].** As many digits as `convert_to_radix` gives
//!   (`⌊p·log₁₀ 2⌋ + 1` for a `p`-bit mantissa), correctly rounded, ties to
//!   even; it agreed with `convert_to_radix` on 10⁶ random values of 64 to
//!   4096 bits with exponents up to `±2³⁰` (W11).
//! - **[`to_f64`].** Rounds to nearest, ties to even, straight from the
//!   binary mantissa; values below the `f64` range go through the
//!   subnormals to `±0.0`.  NaN, `±∞` and values that overflow `f64` are
//!   `None` ([`to_f64_or_inf`]: `±∞`, and `None` only for NaN).
//! - **`BigFloat::from_f64`** halves every subnormal `f64` (0.9.6:
//!   `from_f64(5e-324)` is `2.47e-324`, `from_f64(-1e-310)` is `-5e-311`);
//!   normal values convert exactly.  [`from_f64`] scales subnormals into
//!   the normal range first; literal constants may still call the library.

use astro_float::{BigFloat, Consts, Error, Exponent, RoundingMode, Sign, WORD_BIT_SIZE};
use num_bigint::BigUint;
use num_traits::Zero;

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

/// The decimal digits of `x`, correctly rounded (ties to even) to as many
/// digits as astro-float's `convert_to_radix` gives for the precision of
/// `x` — `⌊p·log₁₀ 2⌋ + 1` for a `p`-bit mantissa — with trailing zeros
/// removed; an exact zero has no digits.
///
/// Computed from the binary mantissa and exponent with integer arithmetic:
/// the value is `m·2^q` exactly, and its leading digits are `⌊m·2^q/10^s⌋`
/// for a power `10^s` that leaves a few more digits than are kept, plus
/// whether anything is left below them.  For `|s|` up to
/// [`EXACT_DECIMAL_SCALE`] (or twice the precision) `5^|s|` is formed
/// exactly; beyond, between two bounds of `5^|s|` that the precision is
/// raised until the leading digits agree (the value is then not an
/// integer multiple of `10^s`, so something is always left over).  This
/// replaced `convert_to_radix`, whose decimal exponent was wrong at 1024
/// bits (astro-float #33, open).
///
/// # Errors
///
/// `Error::InvalidArgument` for NaN and `±∞` (as `convert_to_radix`).
pub(crate) fn to_decimal_digits(x: &BigFloat) -> Result<DecimalDigits, Error> {
    let Some((words, _, sign, exponent, _)) = x.as_raw_parts() else {
        return Err(Error::InvalidArgument);
    };
    let mut m = BigUint::zero();
    for &w in words.iter().rev() {
        m = (m << WORD_BIT_SIZE) | BigUint::from(w);
    }
    if m.is_zero() {
        return Ok(DecimalDigits {
            sign,
            digits: Vec::new(),
            exponent: 0,
        });
    }
    let p = words.len() * WORD_BIT_SIZE;
    // `convert_to_radix`'s digit count for a `p`-bit mantissa.
    let n = (p as u64 * 301_029_996 / 1_000_000_000) as usize + 1;
    // The value is `m·2^q`.
    let q = i64::from(exponent) - p as i64;
    let (lead, scale, rest) = leading_decimal_digits(&m, q, n + 2, p);
    let mut digits: Vec<u8> = lead.to_str_radix(10).bytes().map(|b| b - b'0').collect();
    let mut dexp = scale + digits.len() as i64;
    // Round to `n` digits, ties to even.
    if digits.len() > n {
        let next = digits[n];
        let below = rest || digits[n + 1..].iter().any(|&d| d != 0);
        let up = next > 5 || (next == 5 && (below || digits[n - 1] % 2 == 1));
        digits.truncate(n);
        if up {
            let mut carried = true;
            for d in digits.iter_mut().rev() {
                if *d == 9 {
                    *d = 0;
                } else {
                    *d += 1;
                    carried = false;
                    break;
                }
            }
            if carried {
                digits.insert(0, 1);
                digits.truncate(n);
                dexp += 1;
            }
        }
    }
    while digits.last() == Some(&0) {
        digits.pop();
    }
    let exponent = Exponent::try_from(dexp).map_err(|_| Error::ExponentOverflow(sign))?;
    Ok(DecimalDigits {
        sign,
        digits,
        exponent,
    })
}

/// Decimal scales `|s|` up to which [`to_decimal_digits`] forms `5^|s|`
/// exactly (at most a 150 000-bit integer).
const EXACT_DECIMAL_SCALE: i64 = 1 << 16;

/// `(N, s, rest)` with `N = ⌊m·2^q/10^s⌋` having at least `k` digits and
/// `rest` whether `m·2^q/10^s` is not an integer (`p`: the mantissa bits).
fn leading_decimal_digits(m: &BigUint, q: i64, k: usize, p: usize) -> (BigUint, i64, bool) {
    // 2^(b−1) ≤ m·2^(q) / 2^q < 2^b: the value lies in [2^(b+q−1), 2^(b+q)).
    let b = m.bits() as i64;
    let low_log10 =
        crate::base::cast::f64_to_i64(((b + q - 1) as f64 * std::f64::consts::LOG10_2).floor())
            .unwrap_or(0);
    // One digit of slack for the rounding of the logarithm.
    let s = low_log10 - k as i64 - 1;
    let limit = EXACT_DECIMAL_SCALE.max(2 * p as i64);
    if s.abs() <= limit {
        return scaled_exact(m, q, s);
    }
    // Bounds `lo·2^z ≤ 5^|s| ≤ hi·2^z` at `bits` bits, raised until the
    // leading digits agree.
    let mut bits = b as u64 + 4 * k as u64 + 64;
    loop {
        let (lo, hi, z) = pow5_bounds(s.unsigned_abs(), bits);
        let (n_lo, n_hi) = if s > 0 {
            // m·2^q/(2^s·5^s) between m·2^f/hi and m·2^f/lo, f = q − s − z.
            let f = q - s - z;
            (shifted_quotient(m, f, &hi), shifted_quotient(m, f, &lo))
        } else {
            // m·5^t·2^(q+t), t = −s, between m·lo·2^g and m·hi·2^g.
            let g = q - s + z;
            (shift_floor(&(m * &lo), g), shift_floor(&(m * &hi), g))
        };
        if n_lo == n_hi || bits > MAX_DECIMAL_BITS {
            // (Beyond the cap the leading digits are the lower bound's; never
            // reached: the bounds meet after a doubling or two.)
            return (n_lo, s, true);
        }
        bits *= 2;
    }
}

/// The working bits beyond which [`leading_decimal_digits`] stops raising
/// the precision of its bounds of `5^|s|`.
const MAX_DECIMAL_BITS: u64 = 1 << 26;

/// `⌊m·2^q/10^s⌋` and whether it is inexact, with `5^|s|` formed exactly.
fn scaled_exact(m: &BigUint, q: i64, s: i64) -> (BigUint, i64, bool) {
    let five = BigUint::from(5u32).pow(s.unsigned_abs() as u32);
    if s >= 0 {
        // m·2^(q−s)/5^s.
        let f = q - s;
        let (num, den) = if f >= 0 {
            (m << f.unsigned_abs(), five)
        } else {
            (m.clone(), five << f.unsigned_abs())
        };
        let n = &num / &den;
        let rest = !(&num % &den).is_zero();
        (n, s, rest)
    } else {
        // m·5^t·2^(q+t), t = −s.
        let g = q - s;
        let num = m * five;
        if g >= 0 {
            (num << g.unsigned_abs(), s, false)
        } else {
            let sh = g.unsigned_abs();
            let rest = num.trailing_zeros().is_none_or(|tz| tz < sh);
            (num >> sh, s, rest)
        }
    }
}

/// `⌊x·2^g⌋`.
fn shift_floor(x: &BigUint, g: i64) -> BigUint {
    if g >= 0 {
        x << g.unsigned_abs()
    } else {
        x >> g.unsigned_abs()
    }
}

/// `⌊m·2^f/d⌋`.
fn shifted_quotient(m: &BigUint, f: i64, d: &BigUint) -> BigUint {
    if f >= 0 {
        (m << f.unsigned_abs()) / d
    } else {
        m / (d << f.unsigned_abs())
    }
}

/// `(lo, hi, z)` with `lo·2^z ≤ 5^e ≤ hi·2^z`, `hi` of at most `bits`
/// bits: binary powering, every product truncated down for `lo` and up for
/// `hi`.
fn pow5_bounds(e: u64, bits: u64) -> (BigUint, BigUint, i64) {
    let trim = |lo: BigUint, hi: BigUint, z: i64| {
        let extra = hi.bits().saturating_sub(bits);
        if extra == 0 {
            (lo, hi, z)
        } else {
            (lo >> extra, (hi >> extra) + 1u32, z + extra as i64)
        }
    };
    let one = || BigUint::from(1u32);
    let (mut r_lo, mut r_hi, mut r_z) = (one(), one(), 0i64);
    let (mut b_lo, mut b_hi, mut b_z) = (BigUint::from(5u32), BigUint::from(5u32), 0i64);
    let mut e = e;
    while e > 0 {
        if e & 1 == 1 {
            (r_lo, r_hi, r_z) = trim(&r_lo * &b_lo, &r_hi * &b_hi, r_z + b_z);
        }
        e >>= 1;
        if e > 0 {
            (b_lo, b_hi, b_z) = trim(&b_lo * &b_lo, &b_hi * &b_hi, 2 * b_z);
        }
    }
    (r_lo, r_hi, r_z)
}

/// `x` at `p` bits (`p ≥ 53`), exactly, subnormal values included:
/// astro-float's `BigFloat::from_f64` halves every subnormal (0.9.6:
/// `from_f64(5e-324)` is `2.47e-324`), so those are converted from
/// `x·2⁶⁴` and scaled back by an exact division.  Use this for any `f64`
/// that is not a literal constant.
///
/// ```ignore
/// assert_eq!(to_f64(&from_f64(5e-324, 64)), Some(5e-324));
/// ```
pub(crate) fn from_f64(x: f64, p: usize) -> BigFloat {
    if x != 0.0 && x.abs() < f64::MIN_POSITIVE {
        let scale = 2f64.powi(64);
        return BigFloat::from_f64(x * scale, p).div(
            &BigFloat::from_f64(scale, p),
            p,
            RoundingMode::ToEven,
        );
    }
    BigFloat::from_f64(x, p)
}

/// Is `x > 0`?  (`+∞` is; `+0` and NaN are not.)  astro-float's
/// `is_positive` only reads the sign bit, so it is true for `+0`;
/// `clippy.toml` bans it in favour of this and [`is_sign_positive`].
pub(crate) fn is_strictly_positive(x: &BigFloat) -> bool {
    !x.is_nan() && !x.is_zero() && !x.is_negative()
}

/// Is the sign bit of `x` positive (`+0` and `+∞` included, NaN not)?
/// What astro-float's `is_positive` answers, for the places that mean
/// "not negative" and treat a zero as positive on purpose (`f64`'s
/// `is_sign_positive`).
pub(crate) fn is_sign_positive(x: &BigFloat) -> bool {
    !x.is_nan() && !x.is_negative()
}

/// The `f64` nearest to `x`, `±∞` beyond the `f64` range (the IEEE
/// conversion under round-to-nearest), or `None` for NaN: [`to_f64`] with
/// overflow made explicit.
pub(crate) fn to_f64_or_inf(x: &BigFloat) -> Option<f64> {
    if x.is_nan() {
        return None;
    }
    Some(to_f64(x).unwrap_or(if x.is_negative() {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    }))
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
        let d = to_decimal_digits(&BigFloat::from_f64(-12.5, 64)).expect("finite value");
        assert_eq!(d.sign, Sign::Neg);
        assert_eq!(d.exponent, 2);
        assert_eq!(d.digits, vec![1, 2, 5]);
        assert!(to_decimal_digits(&BigFloat::from_f64(f64::NAN, 64)).is_err());
        assert!(to_decimal_digits(&BigFloat::from_f64(f64::INFINITY, 64)).is_err());
        assert!(
            to_decimal_digits(&BigFloat::new(64))
                .expect("zero")
                .digits
                .is_empty()
        );
        // 1/3 at 64 bits is 12297829382473034411·2⁻⁶⁵ = 0.333…3334229…:
        // 20 digits (⌊64·log₁₀ 2⌋ + 1), the last rounded.
        let third =
            BigFloat::from_i32(1, 64).div(&BigFloat::from_i32(3, 64), 64, RoundingMode::ToEven);
        let d = to_decimal_digits(&third).expect("finite value");
        assert_eq!(d.exponent, 0);
        let mut want = vec![3; 19];
        want.push(4);
        assert_eq!(d.digits, want);
        // 2^-1100 (mpmath: 7.362151829022862675436866177145e-332) and
        // 2^3000 (mpmath: 1.2302319221611171769315588132768e+903).
        let two = BigFloat::from_i32(2, 128);
        let tiny = BigFloat::from_i32(1, 128).div(
            &two.powi(1100, 128, RoundingMode::ToEven),
            128,
            RoundingMode::ToEven,
        );
        let d = to_decimal_digits(&tiny).expect("finite value");
        assert_eq!(d.exponent, -331);
        assert_eq!(&d.digits[..12], &[7, 3, 6, 2, 1, 5, 1, 8, 2, 9, 0, 2]);
        let big = two.powi(3000, 128, RoundingMode::ToEven);
        let d = to_decimal_digits(&big).expect("finite value");
        assert_eq!(d.exponent, 904);
        assert_eq!(&d.digits[..12], &[1, 2, 3, 0, 2, 3, 1, 9, 2, 2, 1, 6]);
    }

    /// [`leading_decimal_digits`]'s two routes (exact `5^|s|`, and bounds of
    /// it) give the same digits.
    #[test]
    fn bounded_scale_matches_exact_scale() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..300 {
            let m = BigUint::from(next() | (1 << 63)) << 64u32 | BigUint::from(next());
            let q = (next() % 4001) as i64 - 2000;
            let k = 20 + (next() % 30) as usize;
            let b = m.bits() as i64;
            let s = crate::base::cast::f64_to_i64(
                ((b + q - 1) as f64 * std::f64::consts::LOG10_2).floor(),
            )
            .unwrap_or(0)
                - k as i64
                - 1;
            let exact = scaled_exact(&m, q, s);
            // The bounded route, forced.
            let mut bits = b as u64 + 4 * k as u64 + 64;
            let bounded = loop {
                let (lo, hi, z) = pow5_bounds(s.unsigned_abs(), bits);
                let (n_lo, n_hi) = if s > 0 {
                    let f = q - s - z;
                    (shifted_quotient(&m, f, &hi), shifted_quotient(&m, f, &lo))
                } else {
                    let g = q - s + z;
                    (shift_floor(&(&m * &lo), g), shift_floor(&(&m * &hi), g))
                };
                if n_lo == n_hi {
                    break n_lo;
                }
                bits *= 2;
            };
            assert_eq!(exact.0, bounded, "m = {m}, q = {q}, s = {s}");
        }
    }

    /// `convert_to_radix` (the library's conversion), for the comparison.
    #[allow(clippy::disallowed_methods)]
    fn library_digits(x: &BigFloat, cc: &mut Consts) -> Option<(Sign, Vec<u8>, Exponent)> {
        x.convert_to_radix(astro_float::Radix::Dec, RoundingMode::ToEven, cc)
            .ok()
    }

    /// [`to_decimal_digits`] against `convert_to_radix` on random values:
    /// mantissas of 64 to 4096 bits, exponents near 0, up to `±10⁵` and up
    /// to `±2³⁰`.  `W11_DECIMAL_CASES` raises the count (10⁶ for the
    /// W11 run); every disagreement is printed and must be one of the
    /// library's own errors at a 1024-bit mantissa (astro-float #33).
    #[test]
    fn to_decimal_digits_matches_the_library() {
        let cases: u64 = std::env::var("W11_DECIMAL_CASES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1500);
        let mut cc = consts();
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let precisions = [
            64usize, 128, 192, 256, 320, 512, 960, 1024, 1088, 2048, 4096,
        ];
        let mut mismatches = 0u64;
        // `W11_DECIMAL_DUMP=<file>`: the first 3000 cases, for an mpmath check.
        let mut dump = std::env::var("W11_DECIMAL_DUMP")
            .ok()
            .and_then(|path| std::fs::File::create(path).ok());
        for i in 0..cases {
            let p = precisions[(next() % precisions.len() as u64) as usize];
            let words = p / WORD_BIT_SIZE;
            let mut m: Vec<astro_float::Word> =
                (0..words).map(|_| next() as astro_float::Word).collect();
            if let Some(top) = m.last_mut() {
                *top |= 1 << (WORD_BIT_SIZE - 1);
            }
            // A short mantissa now and then (exact small values, ties).
            if i % 7 == 0 {
                let keep = 1 + (next() % (p as u64 - 1)) as usize;
                for (j, w) in m.iter_mut().enumerate() {
                    let lo = j * WORD_BIT_SIZE;
                    if lo + WORD_BIT_SIZE <= p - keep {
                        *w = 0;
                    } else if lo < p - keep {
                        *w &= !((1 << (p - keep - lo)) - 1);
                    }
                }
            }
            let e: i64 = match i % 3 {
                0 => (next() % 401) as i64 - 200,
                1 => (next() % 200_001) as i64 - 100_000,
                _ => (next() % (1 << 31)) as i64 - (1 << 30),
            };
            let sign = if next() & 1 == 0 {
                Sign::Pos
            } else {
                Sign::Neg
            };
            let x = BigFloat::from_raw_parts(&m, p, sign, e as Exponent, false);
            let ours = to_decimal_digits(&x).expect("finite value");
            if let Some(file) = dump.as_mut().filter(|_| i < 3000) {
                use std::io::Write;
                let mant: String = m.iter().rev().map(|w| format!("{w:016x}")).collect();
                let digits: String = ours.digits.iter().map(|d| char::from(b'0' + d)).collect();
                let neg = sign == Sign::Neg;
                let _ = writeln!(file, "{p} {neg} {mant} {e} {digits} {}", ours.exponent);
            }
            let Some((s, d, ex)) = library_digits(&x, &mut cc) else {
                continue;
            };
            if (s, &d, ex) != (ours.sign, &ours.digits, ours.exponent) {
                mismatches += 1;
                eprintln!(
                    "MISMATCH p={p} e={e} m={m:?} ours={:?}e{} lib={d:?}e{ex}",
                    ours.digits, ours.exponent
                );
                assert_eq!(p, 1024, "a disagreement away from the known library defect");
            }
        }
        eprintln!("to_decimal_digits: {cases} cases, {mismatches} disagreements");
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

    /// `BigFloat::from_f64` halves subnormals; [`from_f64`] is exact for them.
    #[test]
    fn from_f64_keeps_subnormals() {
        for x in [5e-324, -1e-310, 0.5f64.powi(1070), 2.2e-308, 1.0, -0.0, 0.0] {
            assert_eq!(to_f64(&from_f64(x, 64)), Some(x), "{x}");
            assert_eq!(to_f64(&from_f64(x, 192)), Some(x), "{x}");
        }
        // The library defect itself, so this test notices a fix.
        assert_eq!(to_f64(&BigFloat::from_f64(5e-324, 64)), Some(0.0));
    }

    /// `+0` is not positive; the sign bit is.
    #[test]
    fn strict_and_sign_positivity() {
        let p = 64;
        let zero = BigFloat::new(p);
        assert!(!is_strictly_positive(&zero));
        assert!(is_sign_positive(&zero));
        assert!(!is_strictly_positive(&zero.neg()));
        assert!(!is_sign_positive(&BigFloat::from_i32(-1, p)));
        assert!(is_strictly_positive(&BigFloat::from_i32(1, p)));
        assert!(is_strictly_positive(&BigFloat::from_f64(f64::INFINITY, p)));
        let nan = BigFloat::from_f64(f64::NAN, p);
        assert!(!is_strictly_positive(&nan) && !is_sign_positive(&nan));
        assert_eq!(to_f64_or_inf(&nan), None);
        let two = BigFloat::from_i32(2, p);
        let huge = two.powi(5000, p, RoundingMode::ToEven);
        assert_eq!(to_f64_or_inf(&huge), Some(f64::INFINITY));
        assert_eq!(to_f64_or_inf(&huge.neg()), Some(f64::NEG_INFINITY));
        assert_eq!(to_f64_or_inf(&two), Some(2.0));
    }
}
