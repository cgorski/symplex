//! Exact conversions between machine floats and rationals.
//!
//! Every finite `f64` is a dyadic rational `m · 2^e`, so it can be
//! converted to a [`Ratio<BigInt>`] *exactly* — no rounding, no chosen
//! denominator.  That is the right default for a library whose core
//! promise is exact arithmetic: `0.1_f64` really is
//! `3602879701896397/36028797018963968`, and pretending otherwise hides
//! error.
//!
//! When the caller *wants* the "nice" rational a human meant
//! (`0.1 → 1/10`), use [`f64_to_ratio_approx`], which returns the best
//! rational approximation with a bounded denominator (Stern–Brocot /
//! continued-fraction convergents), or the higher-level
//! `Context::from_f64_approx`.
//!
//! These two functions replace the several ad-hoc `(x * 10^k).round()`
//! helpers that used to be scattered through the crate, each with a
//! different scale and each silently saturating on large inputs.

use astro_float::{BigFloat, RoundingMode};
use num_bigint::{BigInt, Sign};
use num_rational::Ratio;
use num_traits::{One, Signed, ToPrimitive, Zero};

/// The exact rational number used by the exact linear algebra, linear
/// programming, polytope and certificate modules (`Ratio<BigInt>`).
/// Re-exported as `linprog::Q` and in the prelude.
pub type Q = Ratio<BigInt>;

/// The rational `n / d`.
///
/// # Panics
///
/// Panics if `d == 0` (a programming error, like a zero literal
/// denominator).
///
/// # Examples
///
/// ```
/// use symplex::linprog::q;
/// assert_eq!(q(2, 4), q(1, 2));
/// ```
pub fn q(n: i64, d: i64) -> Q {
    Ratio::new(BigInt::from(n), BigInt::from(d))
}

/// The integer `n` as a rational.
///
/// # Examples
///
/// ```
/// use symplex::linprog::{q, qi};
/// assert_eq!(qi(3), q(6, 2));
/// ```
pub fn qi(n: i64) -> Q {
    Ratio::from_integer(BigInt::from(n))
}

/// Convert a finite `f64` to the exact rational it represents.
///
/// Returns `None` for NaN and ±∞.  Negative zero maps to `0`.
///
/// # Examples
///
/// ```
/// use symplex::base::numeric::f64_to_ratio_exact;
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// assert_eq!(f64_to_ratio_exact(0.5), Some(Ratio::new(BigInt::from(1), BigInt::from(2))));
/// assert_eq!(f64_to_ratio_exact(-3.0), Some(Ratio::from_integer(BigInt::from(-3))));
/// // 0.1 is *not* 1/10 in binary floating point:
/// let tenth = f64_to_ratio_exact(0.1).unwrap();
/// assert_eq!(*tenth.denom(), BigInt::from(36028797018963968_u64));
/// assert!(f64_to_ratio_exact(f64::NAN).is_none());
/// assert!(f64_to_ratio_exact(f64::INFINITY).is_none());
/// ```
#[must_use]
pub fn f64_to_ratio_exact(x: f64) -> Option<Ratio<BigInt>> {
    if !x.is_finite() {
        return None;
    }
    if x == 0.0 {
        return Some(Ratio::zero());
    }

    let bits = x.to_bits();
    let sign = if bits >> 63 == 0 {
        Sign::Plus
    } else {
        Sign::Minus
    };
    let exponent = ((bits >> 52) & 0x7ff) as i64;
    let fraction = bits & 0x000f_ffff_ffff_ffff;

    // Subnormals have an implicit leading 0 and a fixed exponent of -1074;
    // normals have an implicit leading 1 and exponent (e - 1075) after
    // folding the 52-bit fraction into the mantissa.
    let (mantissa, exp2) = if exponent == 0 {
        (fraction, -1074_i64)
    } else {
        (fraction | (1_u64 << 52), exponent - 1075)
    };

    let mantissa = BigInt::from_biguint(sign, mantissa.into());
    let ratio = if exp2 >= 0 {
        Ratio::from_integer(mantissa << (exp2 as usize))
    } else {
        Ratio::new(mantissa, BigInt::one() << ((-exp2) as usize))
    };
    Some(ratio)
}

/// Best rational approximation to `x` with denominator at most `max_denom`.
///
/// Walks the continued-fraction convergents (and semiconvergents) of the
/// exact value of `x`, so the result is the *closest* rational with a
/// denominator not exceeding `max_denom` — this is what turns `0.1` into
/// `1/10`, `0.3333333333333333` into `1/3`, and `3.14159` into `355/113`
/// (for `max_denom = 1000`).
///
/// Returns `None` for NaN / ±∞ or `max_denom == 0`.
///
/// # Examples
///
/// ```
/// use symplex::base::numeric::f64_to_ratio_approx;
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// let r = |p: i64, q: i64| Ratio::new(BigInt::from(p), BigInt::from(q));
/// assert_eq!(f64_to_ratio_approx(0.1, 1_000_000), Some(r(1, 10)));
/// assert_eq!(f64_to_ratio_approx(1.0 / 3.0, 1_000_000), Some(r(1, 3)));
/// assert_eq!(f64_to_ratio_approx(0.3, 1_000_000), Some(r(3, 10)));
/// assert_eq!(f64_to_ratio_approx(std::f64::consts::PI, 1000), Some(r(355, 113)));
/// assert_eq!(f64_to_ratio_approx(-2.5, 10), Some(r(-5, 2)));
/// assert_eq!(f64_to_ratio_approx(7.0, 1), Some(r(7, 1)));
/// ```
#[must_use]
pub fn f64_to_ratio_approx(x: f64, max_denom: u64) -> Option<Ratio<BigInt>> {
    if max_denom == 0 {
        return None;
    }
    let exact = f64_to_ratio_exact(x)?;
    Some(best_rational_approx(&exact, &BigInt::from(max_denom)))
}

/// Best rational approximation to `target` with denominator `≤ max_denom`,
/// via continued-fraction convergents and the final semiconvergent.
///
/// `max_denom` must be `≥ 1`.
#[must_use]
pub fn best_rational_approx(target: &Ratio<BigInt>, max_denom: &BigInt) -> Ratio<BigInt> {
    debug_assert!(max_denom.is_positive());
    if target.denom() <= max_denom {
        return target.clone();
    }

    let negative = target.is_negative();
    let t = target.abs();

    // Standard convergent recurrence:  h_n = a_n h_{n-1} + h_{n-2}, same for k.
    let (mut h_prev, mut h) = (BigInt::one(), BigInt::zero()); // h_{-1}=1, h_{-2}=0
    let (mut k_prev, mut k) = (BigInt::zero(), BigInt::one()); // k_{-1}=0, k_{-2}=1
    let mut rem = t.clone();

    loop {
        let a = rem.floor().to_integer();
        let h_next = &a * &h_prev + &h;
        let k_next = &a * &k_prev + &k;

        if &k_next > max_denom {
            // The next convergent h_n/k_n overshoots the denominator bound.
            // Here h_prev/k_prev = h_{n-1}/k_{n-1} (last good convergent) and
            // h/k = h_{n-2}/k_{n-2}.  The semiconvergents are
            //     (h_{n-2} + m·h_{n-1}) / (k_{n-2} + m·k_{n-1}),  0 < m < a_n,
            // and the largest admissible m gives the closest one.  Pick
            // whichever of that and the last convergent is nearer.
            let m = (max_denom - &k) / &k_prev;
            let candidate = if m.is_zero() {
                None
            } else {
                Some(Ratio::new(&h + &m * &h_prev, &k + &m * &k_prev))
            };
            let conv = Ratio::new(h_prev.clone(), k_prev.clone());
            let best = match candidate {
                Some(semi) if (&semi - &t).abs() < (&conv - &t).abs() => semi,
                _ => conv,
            };
            return if negative { -best } else { best };
        }

        h = std::mem::replace(&mut h_prev, h_next);
        k = std::mem::replace(&mut k_prev, k_next);

        let frac = &rem - Ratio::from_integer(a);
        if frac.is_zero() {
            // Exact: the convergent is the target itself.
            let exact = Ratio::new(h_prev.clone(), k_prev.clone());
            return if negative { -exact } else { exact };
        }
        rem = frac.recip();
    }
}

/// Convert a rational to the nearest `f64`, returning `None` if it does
/// not fit (overflow to ±∞ is reported as `None`).
#[must_use]
pub fn ratio_to_f64(r: &Ratio<BigInt>) -> Option<f64> {
    // `to_f64` on Ratio<BigInt> performs a correctly-rounded division when
    // both parts fit in f64 range; otherwise fall back to scaling.
    if let Some(v) = r.to_f64()
        && v.is_finite()
    {
        return Some(v);
    }
    // Scale down by a common power of two until both fit.
    let shift = r.numer().bits().max(r.denom().bits()).saturating_sub(1000) as usize;
    let n = r.numer() >> shift;
    let d = r.denom() >> shift;
    if d.is_zero() {
        return None;
    }
    let v = n.to_f64()? / d.to_f64()?;
    v.is_finite().then_some(v)
}

/// Convert a `BigInt` to a `BigFloat`, correctly rounded (to nearest,
/// ties to even) to `prec` bits.
///
/// Values that fit in `i128` are converted directly.  Larger integers are
/// handed to astro-float as their limbs — the float `0.limbs · 2^(64·len)`
/// is the integer itself — and rounded once, in time linear in the size.
/// Before 0.29 the limbs were accumulated one at a time (`acc·2⁶⁴ + limb`)
/// at the full width of the integer, a full-width multiplication per limb:
/// quadratic, 12 s (debug) for the 86,000-bit p-value of a binomial test,
/// where `Ratio::to_f64` takes 70 µs.
pub(crate) fn bigint_to_bigfloat(n: &BigInt, prec: usize) -> BigFloat {
    if let Some(v) = n.to_i128() {
        // astro-float builds an `i128` only at 128 bits or more (NaN below).
        let mut out = BigFloat::from_i128(v, prec.max(128));
        if prec < 128 {
            let _ = out.set_precision(prec, RoundingMode::ToEven);
        }
        return out;
    }
    // astro-float's `Word` is 32 or 64 bits depending on the target (and
    // not always on `target_pointer_width` alone), so the words are packed
    // from 32-bit digits for whichever width it is, little-endian.
    let (sign, digits) = n.to_u32_digits();
    let per_word = astro_float::WORD_BIT_SIZE / 32;
    let words: Vec<astro_float::Word> = digits
        .chunks(per_word)
        .map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .fold(0 as astro_float::Word, |w, (i, &d)| {
                    w | (astro_float::Word::from(d) << (32 * i))
                })
        })
        .collect();
    let sign = if sign == Sign::Minus {
        astro_float::Sign::Neg
    } else {
        astro_float::Sign::Pos
    };
    let exponent = words
        .len()
        .checked_mul(astro_float::WORD_BIT_SIZE)
        .and_then(|b| astro_float::Exponent::try_from(b).ok());
    let Some(exponent) = exponent else {
        // Beyond astro-float's exponent range.
        return if sign == astro_float::Sign::Neg {
            astro_float::INF_NEG
        } else {
            astro_float::INF_POS
        };
    };
    let mut out = BigFloat::from_words(&words, sign, exponent);
    let _ = out.set_precision(prec, RoundingMode::ToEven);
    out
}

/// Convert a `Ratio<BigInt>` to a `BigFloat` at `prec` bits: numerator and
/// denominator are rounded to `prec + 64` bits (exact when they fit, as
/// they do for every rational of moderate size; see [`bigint_to_bigfloat`])
/// and divided with rounding mode `rm`, so the result is within
/// `2^(−prec)` relative of the rational however large its parts.
pub(crate) fn ratio_to_bigfloat(r: &Ratio<BigInt>, prec: usize, rm: RoundingMode) -> BigFloat {
    if r.is_zero() {
        return BigFloat::from_i32(0, prec);
    }
    if r.denom().is_one() {
        return bigint_to_bigfloat(r.numer(), prec);
    }
    let wp = prec + 64;
    let n = bigint_to_bigfloat(r.numer(), wp);
    let d = bigint_to_bigfloat(r.denom(), wp);
    n.div(&d, prec, rm)
}

// ── Rational arithmetic without quadratic gcds ──────────────────────────
//
// The operators of `Ratio<BigInt>` reduce every result with
// `BigUint::gcd`, which in num-bigint 0.4 is Stein's binary algorithm: a
// subtraction and a shift of the whole number per step, and a step per bit
// or two *even when the other operand is 1 or a single limb*.  So `n + 1`,
// `1·n` or `3·n` for an integer `n` of `b` bits cost `O(b²/64)` word
// operations where the arithmetic itself is linear (`gcd(n, 1)`: 0.14 ms
// at 5,000 digits, 35 ms at 100,000, 0.2 s at 200,000, release build), at
// every canonical sum or product that holds `n`; the product of twenty
// square roots of 5,000-digit integers spent most of its 3.3 s there.  The
// helpers below return the same values in the same (unique: lowest terms,
// positive denominator) representation, computed the way SymPy's
// `Rational.__add__`/`__mul__` do (an integer operand needs no gcd in a
// sum and only the one against the other denominator in a product) and,
// for two fractions, with Henrici's sum (Knuth, TAOCP vol. 2, 4.5.1);
// every gcd starts with Euclidean steps that bring the larger operand down
// to the size of the smaller ([`gcd_big`]).

/// `gcd(a, b)` on machine words.
fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// `gcd(|a|, |b|)` (`gcd(0, 0) = 0`).
///
/// Euclidean steps while the operands differ in size by more than a limb
/// (each a division linear in the larger), a word gcd once the smaller
/// fits in 64 bits, and num-bigint's binary gcd only for two operands of
/// about the same size, where its cost is that of the gcd itself.
pub(crate) fn gcd_big(a: &BigInt, b: &BigInt) -> BigInt {
    let (x, y) = (a.magnitude(), b.magnitude());
    let (mut big, mut small) = if x >= y {
        (x.clone(), y.clone())
    } else {
        (y.clone(), x.clone())
    };
    loop {
        if small.is_zero() {
            return BigInt::from(big);
        }
        if let Some(s) = small.to_u64() {
            let t = (&big % s).to_u64().unwrap_or(0);
            return BigInt::from(gcd_u64(s, t));
        }
        if big.bits() <= small.bits() + 64 {
            return BigInt::from(num_integer::Integer::gcd(&big, &small));
        }
        let r = &big % &small;
        big = small;
        small = r;
    }
}

/// `x / g` for a divisor `g` of `x` (skipped when `g = 1`).
fn div_by(x: &BigInt, g: &BigInt) -> BigInt {
    if g.is_one() { x.clone() } else { x / g }
}

/// `a + b`, equal to `a.clone() + b.clone()` (see the note above), for
/// operands in lowest terms with positive denominators, as `Ratio::new`
/// and the `Ratio` operators leave them and as every arena number is.
pub(crate) fn q_add(a: &Q, b: &Q) -> Q {
    if a.is_zero() {
        return b.clone();
    }
    if b.is_zero() {
        return a.clone();
    }
    match (a.is_integer(), b.is_integer()) {
        (true, true) => Ratio::from_integer(a.numer() + b.numer()),
        // p/q + k = (p + k·q)/q is in lowest terms: gcd(p + k·q, q) = gcd(p, q).
        (false, true) => Ratio::new_raw(a.numer() + b.numer() * a.denom(), a.denom().clone()),
        (true, false) => Ratio::new_raw(b.numer() + a.numer() * b.denom(), b.denom().clone()),
        (false, false) => {
            let (u, u1, v, v1) = (a.numer(), a.denom(), b.numer(), b.denom());
            let d1 = gcd_big(u1, v1);
            if d1.is_one() {
                return Ratio::new_raw(u * v1 + v * u1, u1 * v1);
            }
            let t = u * div_by(v1, &d1) + v * div_by(u1, &d1);
            if t.is_zero() {
                return Ratio::zero();
            }
            let d2 = gcd_big(&t, &d1);
            Ratio::new_raw(div_by(&t, &d2), div_by(u1, &d1) * div_by(v1, &d2))
        }
    }
}

/// `a · b`, equal to `a.clone() * b.clone()` (see the note above), for
/// operands in lowest terms with positive denominators (as for
/// [`q_add`]).
pub(crate) fn q_mul(a: &Q, b: &Q) -> Q {
    if a.is_zero() || b.is_zero() {
        return Ratio::zero();
    }
    if a.is_one() {
        return b.clone();
    }
    if b.is_one() {
        return a.clone();
    }
    if a.is_integer() && b.is_integer() {
        return Ratio::from_integer(a.numer() * b.numer());
    }
    // Cross-cancellation leaves the product in lowest terms.
    let g1 = gcd_big(a.numer(), b.denom());
    let g2 = gcd_big(b.numer(), a.denom());
    let n = div_by(a.numer(), &g1) * div_by(b.numer(), &g2);
    let d = div_by(a.denom(), &g2) * div_by(b.denom(), &g1);
    if d.is_negative() {
        Ratio::new_raw(-n, -d)
    } else {
        Ratio::new_raw(n, d)
    }
}

/// Integer square root helper for tests and callers that need exactness.
#[must_use]
pub fn is_perfect_square(n: &BigInt) -> bool {
    if n.is_negative() {
        return false;
    }
    let r = n.sqrt();
    &r * &r == *n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(p: i64, q: i64) -> Ratio<BigInt> {
        Ratio::new(BigInt::from(p), BigInt::from(q))
    }

    #[test]
    fn exact_round_trips_through_f64() {
        for &x in &[
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            0.1,
            1.0 / 3.0,
            1e300,
            -1e-300,
            f64::MIN_POSITIVE,
            f64::MAX,
            5e-324, // smallest subnormal
            123_456_789.987_654_3,
        ] {
            let ratio = f64_to_ratio_exact(x).unwrap();
            let back = ratio_to_f64(&ratio).unwrap();
            assert_eq!(back.to_bits(), (x + 0.0).to_bits(), "x = {x:e}");
        }
    }

    #[test]
    fn exact_rejects_non_finite() {
        assert!(f64_to_ratio_exact(f64::NAN).is_none());
        assert!(f64_to_ratio_exact(f64::INFINITY).is_none());
        assert!(f64_to_ratio_exact(f64::NEG_INFINITY).is_none());
    }

    #[test]
    fn exact_known_values() {
        assert_eq!(f64_to_ratio_exact(0.75), Some(r(3, 4)));
        assert_eq!(f64_to_ratio_exact(-1024.0), Some(r(-1024, 1)));
        assert_eq!(f64_to_ratio_exact(1.5e3), Some(r(1500, 1)));
    }

    #[test]
    fn approx_recovers_human_decimals() {
        let cases = [
            (0.1, 1, 10),
            (0.2, 1, 5),
            (0.3, 3, 10),
            (0.25, 1, 4),
            (0.125, 1, 8),
            (2.0 / 3.0, 2, 3),
            (0.142857142857, 1, 7),
        ];
        for &(x, p, q) in &cases {
            assert_eq!(f64_to_ratio_approx(x, 1_000_000), Some(r(p, q)), "x = {x}");
        }
        // √2 has no small rational form.  The f64 `SQRT_2` lies just below
        // the true √2, so among denominators ≤ 100 the closest rational to
        // *that float* is 140/99 (the next convergent 99/70 is above it).
        let best = f64_to_ratio_approx(std::f64::consts::SQRT_2, 100).unwrap();
        assert!(*best.denom() <= BigInt::from(100));
        let exact = f64_to_ratio_exact(std::f64::consts::SQRT_2).unwrap();
        for &(p, q) in &[(99_i64, 70_i64), (141, 100), (17, 12), (7, 5)] {
            assert!(
                (&best - &exact).abs() <= (&r(p, q) - &exact).abs(),
                "{best} is worse than {p}/{q}"
            );
        }
    }

    #[test]
    fn approx_respects_denominator_bound() {
        for &md in &[1u64, 2, 3, 7, 10, 100, 1000, 1_000_000] {
            for &x in &[
                0.1,
                0.7,
                std::f64::consts::PI,
                -std::f64::consts::E,
                12345.6789,
            ] {
                let a = f64_to_ratio_approx(x, md).unwrap();
                assert!(*a.denom() <= BigInt::from(md), "x={x}, md={md}, got {a}");
                // Must be at least as good as naive rounding with that denominator.
                let naive = Ratio::new(
                    BigInt::from((x * md as f64).round() as i64),
                    BigInt::from(md),
                );
                let exact = f64_to_ratio_exact(x).unwrap();
                assert!(
                    (&a - &exact).abs() <= (&naive - &exact).abs(),
                    "x={x}, md={md}: {a} worse than naive {naive}"
                );
            }
        }
    }

    #[test]
    fn approx_negative_and_integer() {
        assert_eq!(f64_to_ratio_approx(-0.5, 10), Some(r(-1, 2)));
        assert_eq!(f64_to_ratio_approx(-7.0, 10), Some(r(-7, 1)));
        assert_eq!(f64_to_ratio_approx(0.0, 10), Some(r(0, 1)));
        assert!(f64_to_ratio_approx(1.0, 0).is_none());
    }

    #[test]
    fn best_rational_exact_when_within_bound() {
        let t = r(22, 7);
        assert_eq!(best_rational_approx(&t, &BigInt::from(7)), t);
        assert_eq!(best_rational_approx(&t, &BigInt::from(1000)), t);
    }

    #[test]
    fn perfect_square() {
        assert!(is_perfect_square(&BigInt::from(0)));
        assert!(is_perfect_square(&BigInt::from(144)));
        assert!(!is_perfect_square(&BigInt::from(145)));
        assert!(!is_perfect_square(&BigInt::from(-4)));
    }

    /// `q_add`, `q_mul` and `gcd_big` against the `Ratio` operators and
    /// num-integer's gcd: same values, same (reduced) representation, on
    /// small, huge and mixed operands of both signs.
    #[test]
    fn fast_rational_arithmetic_matches_ratio() {
        use num_traits::Pow;
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let big = |k: u32, c: i64| Pow::pow(BigInt::from(10), k) + BigInt::from(c);
        let mut ints: Vec<BigInt> = vec![
            BigInt::from(0),
            BigInt::from(1),
            BigInt::from(-1),
            BigInt::from(2),
            BigInt::from(6),
            BigInt::from(-12),
            BigInt::from(u64::MAX),
            big(40, 7),
            big(40, 7) * 6,
            big(300, -3),
            Pow::pow(BigInt::from(2), 200u32) * 3,
            -big(25, 1),
        ];
        for _ in 0..12 {
            let v = BigInt::from(next()) * BigInt::from(next()) - BigInt::from(next());
            ints.push(if next() % 2 == 0 { v } else { -v });
        }
        let mut qs: Vec<Q> = Vec::new();
        for (i, n) in ints.iter().enumerate() {
            qs.push(Ratio::from_integer(n.clone()));
            let d = &ints[(i * 7 + 3) % ints.len()];
            if !d.is_zero() {
                qs.push(Ratio::new(n.clone(), d.clone()));
            }
        }
        for a in &qs {
            for b in &qs {
                let (sum, prod) = (q_add(a, b), q_mul(a, b));
                assert_eq!(sum, a.clone() + b.clone(), "{a} + {b}");
                assert_eq!(prod, a.clone() * b.clone(), "{a} * {b}");
                // The representation itself is the reduced one.
                for q in [&sum, &prod] {
                    assert!(q.denom().is_positive(), "{q}");
                    assert!(
                        num_integer::Integer::gcd(q.numer(), q.denom()).is_one(),
                        "{q}"
                    );
                }
            }
        }
        for a in &ints {
            for b in &ints {
                assert_eq!(
                    gcd_big(a, b),
                    num_integer::Integer::gcd(a, b),
                    "gcd({a}, {b})"
                );
            }
        }
    }
}
