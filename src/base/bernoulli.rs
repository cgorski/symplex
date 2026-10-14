//! Bernoulli number computation and caching.
//!
//! Provides exact rational Bernoulli numbers B_0, B_1, B_2, ... (`B₁ = +1/2`,
//! see [`bernoulli`])
//! computed from the tangent numbers (Brent–Harvey) and cached for reuse
//! across evaluations. Bernoulli numbers are the foundation for the
//! Stirling series used in arbitrary-precision Gamma function evaluation.
//!
//! Beyond `B₄₀₉₆` (the table would cost `O(n³)` bit operations and hold
//! every number below), a single `Bₙ` comes from the zeta function, as
//! mpmath's `bernfrac` computes it (`mpmath/libmp/gammazeta.py`, BSD):
//!
//! ```text
//! |Bₙ| = 2·n!·ζ(n)/(2π)ⁿ,   Bₙ = N/D,   D = ∏_{(p−1) | n} p
//! ```
//!
//! (the denominator by von Staudt–Clausen), with `1/ζ(n) = ∏_p (1 − p^(−n))`
//! (Euler's product, truncated where the rest is below the precision) and
//! every rounding counted, so that the numerator `N = |Bₙ|·D` is known to
//! within `2^(−60)` before it is rounded to the integer it is.  The Euler
//! numbers likewise from Dirichlet's `β`: `|Eₙ| = 2·n!·(2/π)^(n+1)·β(n + 1)`,
//! `1/β(s) = ∏_{p odd} (1 − χ₄(p)·p^(−s))` (DLMF 24.8.4, 25.15.1; Brent and
//! Harvey 2011, §2, use the same identities).

use astro_float::{BigFloat, Consts, RoundingMode, WORD_BIT_SIZE};
use num_bigint::{BigInt, BigUint};
use num_rational::Ratio;
use num_traits::{One, Zero};
use parking_lot::Mutex;

use crate::base::numeric::{Q, bigint_to_bigfloat};

/// Global cache of the even Bernoulli numbers `B₂, B₄, …, B₂ₙ`.
static BERNOULLI_CACHE: Mutex<Vec<Q>> = Mutex::new(Vec::new());

/// Largest `k` whose `B₂ₖ` comes from the tangent-number table ([`even`]);
/// beyond, each number is computed by itself ([`bernoulli_by_zeta`]).
const TABLE_MAX_K: usize = 2048;

/// Largest even index `n` of an Euler number `Eₙ` that callers compute
/// from the secant numbers (`base::combinatorics::euler_even`, `O(n³)` bit
/// operations); beyond, [`euler_by_beta`].
pub(crate) const EXACT_EULER_N: usize = 512;

/// Return B_n (the n-th Bernoulli number) as an exact rational, with
/// `B₁ = +1/2`: the crate's convention, `Bₙ = Bₙ(1) = −n·ζ(1 − n)`, so
/// that the Bernoulli function `bernoulli(s) = −s·ζ(1 − s)` of `evalf` is
/// continuous at 1 (SymPy's since 1.12; DLMF 24.2.1 and mpmath have
/// `B₁ = −1/2`, which was this crate's up to 0.49).
///
/// Odd Bernoulli numbers beyond B_1 are zero.  The even ones come from the
/// tangent numbers ([`even`]).  Results are cached globally; thread-safe
/// via `Mutex`.
#[must_use]
pub(crate) fn bernoulli(n: usize) -> Q {
    match n {
        0 => Ratio::one(),
        1 => Ratio::new(BigInt::from(1), BigInt::from(2)),
        _ if n % 2 == 1 => Ratio::zero(),
        _ => even(n / 2),
    }
}

/// `Bₙ` by itself (the `ntheory::bernoulli` of one index): from the table
/// when it already holds it or `n` is small, otherwise by the zeta function
/// ([`bernoulli_by_zeta`]), which costs far less than the table up to `n`
/// (`B₄₀₉₆`: 0.15 s against 44 s for the table, debug build).
#[must_use]
pub(crate) fn bernoulli_single(n: usize) -> Q {
    if n < SINGLE_FROM || !n.is_multiple_of(2) {
        return bernoulli(n);
    }
    let cached = BERNOULLI_CACHE.lock().get(n / 2 - 1).cloned();
    if let Some(b) = cached {
        return b;
    }
    bernoulli_by_zeta(n).unwrap_or_else(|| bernoulli(n))
}

/// The index from which [`bernoulli_single`] does not extend the table.
const SINGLE_FROM: usize = 512;

/// The tangent numbers `T₁ = 1, T₂ = 2, T₃ = 16, …, Tₙ` (`T₀ = 0` in
/// slot 0): Brent and Harvey, "Fast computation of Bernoulli, Tangent and
/// Secant numbers" (2011), Algorithm TangentNumbers — `O(n²)` updates by
/// small factors, in place.
fn tangent_numbers(n: usize) -> Vec<BigInt> {
    let mut t: Vec<BigInt> = vec![BigInt::zero(); n + 1];
    if n == 0 {
        return t;
    }
    t[1] = BigInt::one();
    for k in 2..=n {
        t[k] = &t[k - 1] * BigInt::from(k - 1);
    }
    for k in 2..=n {
        for j in k..=n {
            let v = &t[j - 1] * BigInt::from(j - k) + &t[j] * BigInt::from(j - k + 2);
            t[j] = v;
        }
    }
    t
}

/// `B₂ₖ` for `k ≥ 1` (`B₀ = 1` for `k = 0`), from the tangent numbers:
/// `B₂ₖ = (−1)^(k−1) · 2k · Tₖ / (2^(2k) · (2^(2k) − 1))`.
///
/// Up to 0.39 this module summed the rational recurrence `B_m = −Σ
/// C(m+1, k)·B_k/(m+1)`, a gcd per term: `ntheory::bernoulli(1000)` took
/// 9 s.  The values are the same.
#[must_use]
pub(crate) fn even(k: usize) -> Q {
    if k == 0 {
        return Q::one();
    }
    if k > TABLE_MAX_K {
        let cached = BERNOULLI_CACHE.lock().get(k - 1).cloned();
        if let Some(b) = cached {
            return b;
        }
        // Beyond the table: by itself (`O(n² log n)` bits of work, no
        // cache of the numbers below).  Up to 0.43 the table was extended,
        // and `ntheory::bernoulli` refused `n > 4096` for its cost.
        if let Some(b) = bernoulli_by_zeta(2 * k) {
            return b;
        }
    }
    let mut cache = BERNOULLI_CACHE.lock();
    if cache.len() < k {
        // Recomputed from scratch (the algorithm is in place), for at least
        // a quarter more numbers than before: a run of growing requests
        // costs a constant factor over the last one, and no request more
        // than about twice its own cost (the work is cubic in `n`).
        let n = k.max(cache.len() + cache.len() / 4).max(32);
        let t = tangent_numbers(n);
        let mut out = Vec::with_capacity(n);
        for (i, tk) in t.iter().enumerate().skip(1) {
            let two_k = 2 * i;
            let p = BigInt::one() << two_k;
            let den = &p * (&p - BigInt::one());
            let mut num = tk * BigInt::from(two_k);
            if i.is_multiple_of(2) {
                num = -num;
            }
            out.push(Ratio::new(num, den));
        }
        tracing::trace!(count = n, "bernoulli: tangent numbers computed");
        *cache = out;
    }
    cache[k - 1].clone()
}

/// The primes up to `limit` (Eratosthenes).
fn primes_to(limit: usize) -> Vec<usize> {
    let mut composite = vec![false; limit + 1];
    let mut out = Vec::new();
    for p in 2..=limit {
        if composite[p] {
            continue;
        }
        out.push(p);
        let mut m = p.saturating_mul(p);
        while m <= limit {
            composite[m] = true;
            m += p;
        }
    }
    out
}

/// Is `m` prime (trial division; `m` is at most an index plus one here)?
fn is_small_prime(m: usize) -> bool {
    if m < 2 {
        return false;
    }
    let mut d = 2usize;
    while d.saturating_mul(d) <= m {
        if m.is_multiple_of(d) {
            return false;
        }
        d += 1;
    }
    true
}

/// `log₂ n!` rounded up (Stirling with Robbins' bound `1/(12n)` on the
/// remainder, plus a margin for the `f64` arithmetic).
fn log2_factorial_up(n: usize) -> f64 {
    if n < 2 {
        return 0.0;
    }
    let x = n as f64;
    let ln = x * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI * x).ln() + 1.0 / (12.0 * x);
    ln * std::f64::consts::LOG2_E * (1.0 + 1e-12) + 1.0
}

/// `∏_{p ≤ P} (1 − χ(p)·p^(−s))` over the primes, `χ(p) ∈ {−1, 0, 1}`, at
/// `wp` bits, truncated where `Σ_{k > P} k^(−s) ≤ P^(1−s)` is below
/// `2^(−wp−8)`: its relative error is below `(primes + 2)·2^(−wp+1)`.  Each
/// `p^(−s)` is needed only to `wp − s·log₂ p` bits relative (it is that much
/// below 1), and so is the product that multiplies it.
fn euler_product(
    s: usize,
    wp: usize,
    chi: impl Fn(usize) -> i32,
    rm: RoundingMode,
) -> Option<BigFloat> {
    if s < 2 {
        return None;
    }
    let exponent = (wp as f64 + 8.0) / (s as f64 - 1.0);
    if exponent > 24.0 {
        return None; // a small `s`: millions of primes (callers do not ask)
    }
    let limit = (exponent.exp2().ceil() as usize).max(3);
    let mut prod = BigFloat::from_i32(1, wp);
    for p in primes_to(limit) {
        let c = chi(p);
        if c == 0 {
            continue;
        }
        let below = (s as f64) * (p as f64).log2();
        if below > wp as f64 + 8.0 {
            break;
        }
        let q = (wp as f64 - below.floor()) as usize + 32;
        let guard = usize::BITS as usize - s.leading_zeros() as usize + 16;
        let t = BigFloat::from_u64(p as u64, 64).powi(s, q + guard, rm);
        let t = BigFloat::from_i32(1, 64).div(&t, q, rm);
        let delta = prod.mul(&t, q, rm);
        prod = if c > 0 {
            prod.sub(&delta, wp, rm)
        } else {
            prod.add(&delta, wp, rm)
        };
    }
    Some(prod)
}

/// The integer nearest to a positive `x`, when `x` lies within `2^(−16)` of
/// it (`None` otherwise: the value was not known well enough).
fn nearest_integer(x: &BigFloat) -> Option<BigInt> {
    let (words, _, _, exponent, _) = x.as_raw_parts()?;
    let mut digits: Vec<u32> = Vec::with_capacity(words.len() * 2);
    for &w in words {
        let mut w = w as u128;
        for _ in 0..WORD_BIT_SIZE / 32 {
            digits.push((w & 0xFFFF_FFFF) as u32);
            w >>= 32;
        }
    }
    let mantissa = BigUint::new(digits);
    // x = mantissa · 2^(exponent − len).
    let len = (words.len() * WORD_BIT_SIZE) as i64;
    let shift = len - i64::from(exponent);
    if shift <= 0 {
        let shift = usize::try_from(-shift).ok()?;
        return Some(BigInt::from(mantissa << shift));
    }
    let shift = usize::try_from(shift).ok()?;
    if shift < 18 {
        return None;
    }
    let one = BigUint::one();
    let rounded = (&mantissa + (&one << (shift - 1))) >> shift;
    // The distance to the integer, in units of 2^(−shift).
    let back = &rounded << shift;
    let gap = if back >= mantissa {
        back - &mantissa
    } else {
        &mantissa - back
    };
    (gap < (one << (shift - 16))).then(|| BigInt::from(rounded))
}

/// `n!` exactly, by a product tree.
fn factorial(n: usize) -> BigInt {
    fn range(lo: usize, hi: usize) -> BigInt {
        if hi - lo <= 16 {
            return (lo..hi).fold(BigInt::one(), |acc, i| acc * BigInt::from(i));
        }
        let mid = lo + (hi - lo) / 2;
        range(lo, mid) * range(mid, hi)
    }
    range(1, n + 1)
}

/// Working precision for an integer of about `bits` bits computed from
/// `ops` roundings of relative size `2^(−wp)`: the error is below `2^(−60)`.
fn working_bits(bits: f64, ops: usize) -> usize {
    (bits.max(0.0).ceil() as usize) + 64 + (usize::BITS - ops.leading_zeros()) as usize + 8
}

/// `Bₙ` for an even `n ≥ 2` by the zeta function (see the module
/// documentation): `None` only if the computed numerator is not within
/// `2^(−16)` of an integer, which the error bound rules out.
#[must_use]
pub(crate) fn bernoulli_by_zeta(n: usize) -> Option<Q> {
    if n < 2 || !n.is_multiple_of(2) {
        return None;
    }
    // The denominator: the primes p with (p − 1) | n.
    let mut divisors = Vec::new();
    let mut d = 1usize;
    while d.saturating_mul(d) <= n {
        if n.is_multiple_of(d) {
            divisors.push(d);
            if d != n / d {
                divisors.push(n / d);
            }
        }
        d += 1;
    }
    let mut den = BigInt::one();
    for e in divisors {
        if is_small_prime(e + 1) {
            den *= BigInt::from(e + 1);
        }
    }
    let rm = RoundingMode::ToEven;
    let lg_den = den.bits() as f64;
    // |Bₙ| ≤ 2·n!·ζ(2)/(2π)ⁿ < 4·n!/(2π)ⁿ.
    let bits = 2.0 + log2_factorial_up(n) - n as f64 * (2.0 * std::f64::consts::PI).log2() + lg_den;
    let wp = working_bits(bits, 8 * n + 64);
    let mut cc = Consts::new().ok()?;
    let product = euler_product(n, wp, |_| 1, rm)?; // 1/ζ(n)
    let two_pi = cc.pi(wp, rm).mul(&BigFloat::from_i32(2, 64), wp, rm);
    let power = two_pi.powi(n, wp, rm);
    // 2·n!·D/((2π)ⁿ·(1/ζ(n))).
    let x = bigint_to_bigfloat(&factorial(n), wp)
        .mul(&bigint_to_bigfloat(&den, wp), wp, rm)
        .mul(&BigFloat::from_i32(2, 64), wp, rm)
        .div(&power.mul(&product, wp, rm), wp, rm);
    let num = nearest_integer(&x)?;
    let num = if n.is_multiple_of(4) { -num } else { num };
    // In lowest terms by von Staudt–Clausen (and a gcd of a 10⁵-digit
    // numerator would cost more than the number).
    Some(Ratio::new_raw(num, den))
}

/// The Euler number `Eₙ` for an even `n` by Dirichlet's `β` (see the module
/// documentation): `None` only if the computed value is not within
/// `2^(−16)` of an integer, which the error bound rules out.
#[must_use]
pub(crate) fn euler_by_beta(n: usize) -> Option<BigInt> {
    if !n.is_multiple_of(2) {
        return Some(BigInt::zero());
    }
    if n == 0 {
        return Some(BigInt::one());
    }
    let rm = RoundingMode::ToEven;
    let s = n + 1;
    // |Eₙ| ≤ 2·n!·(2/π)^(n+1) (β(s) < 1).
    let bits = 1.0 + log2_factorial_up(n) + s as f64 * (2.0 / std::f64::consts::PI).log2() + 1.0;
    let wp = working_bits(bits, 8 * n + 64);
    let mut cc = Consts::new().ok()?;
    // 1/β(s) = ∏_{p odd} (1 − χ₄(p) p^(−s)).
    let chi = |p: usize| match p % 4 {
        1 => 1,
        3 => -1,
        _ => 0,
    };
    let product = euler_product(s, wp, chi, rm)?;
    let pi = cc.pi(wp, rm).clone();
    let power = pi.powi(s, wp, rm);
    let mut two_pow = BigFloat::from_i32(1, 64);
    two_pow.set_exponent(i32::try_from(s + 2).ok()?); // 2^(s+1)
    let x = bigint_to_bigfloat(&factorial(n), wp)
        .mul(&two_pow, wp, rm)
        .div(&power.mul(&product, wp, rm), wp, rm);
    let e = nearest_integer(&x)?;
    Some(if n.is_multiple_of(4) { e } else { -e })
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::ToPrimitive;

    #[test]
    fn b0_is_one() {
        let b0 = bernoulli(0);
        assert_eq!(b0, Ratio::one(), "B_0 = 1");
    }

    #[test]
    fn b1_is_half() {
        // The crate's convention since 0.50: `B₁ = +1/2` (was `−1/2`).
        let b1 = bernoulli(1);
        let expected = Ratio::new(BigInt::from(1), BigInt::from(2));
        assert_eq!(b1, expected, "B_1 = +1/2");
    }

    #[test]
    fn b2_is_one_sixth() {
        let b2 = bernoulli(2);
        let expected = Ratio::new(BigInt::from(1), BigInt::from(6));
        assert_eq!(b2, expected, "B_2 = 1/6");
    }

    #[test]
    fn b4_is_neg_one_thirtieth() {
        let b4 = bernoulli(4);
        let expected = Ratio::new(BigInt::from(-1), BigInt::from(30));
        assert_eq!(b4, expected, "B_4 = -1/30");
    }

    #[test]
    fn b6_is_one_forty_second() {
        let b6 = bernoulli(6);
        let expected = Ratio::new(BigInt::from(1), BigInt::from(42));
        assert_eq!(b6, expected, "B_6 = 1/42");
    }

    #[test]
    fn b8_is_neg_one_thirtieth() {
        let b8 = bernoulli(8);
        let expected = Ratio::new(BigInt::from(-1), BigInt::from(30));
        assert_eq!(b8, expected, "B_8 = -1/30");
    }

    #[test]
    fn b10() {
        let b10 = bernoulli(10);
        let expected = Ratio::new(BigInt::from(5), BigInt::from(66));
        assert_eq!(b10, expected, "B_10 = 5/66");
    }

    #[test]
    fn b12() {
        let b12 = bernoulli(12);
        let expected = Ratio::new(BigInt::from(-691), BigInt::from(2730));
        assert_eq!(b12, expected, "B_12 = -691/2730");
    }

    #[test]
    fn odd_bernoulli_numbers_are_zero() {
        for n in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21] {
            let bn = bernoulli(n);
            assert!(bn.is_zero(), "B_{n} should be 0, got {bn}");
        }
    }

    #[test]
    fn b14() {
        let b14 = bernoulli(14);
        let expected = Ratio::new(BigInt::from(7), BigInt::from(6));
        assert_eq!(b14, expected, "B_14 = 7/6");
    }

    #[test]
    fn b16() {
        let b16 = bernoulli(16);
        let expected = Ratio::new(BigInt::from(-3617), BigInt::from(510));
        assert_eq!(b16, expected, "B_16 = -3617/510");
    }

    #[test]
    fn b18() {
        let b18 = bernoulli(18);
        let expected = Ratio::new(BigInt::from(43867), BigInt::from(798));
        assert_eq!(b18, expected, "B_18 = 43867/798");
    }

    #[test]
    fn b20() {
        let b20 = bernoulli(20);
        let expected = Ratio::new(BigInt::from(-174611), BigInt::from(330));
        assert_eq!(b20, expected, "B_20 = -174611/330");
    }

    #[test]
    fn cache_is_reused() {
        // Calling bernoulli twice for the same index should return identical values
        // (this implicitly tests the caching mechanism).
        let first = bernoulli(30);
        let second = bernoulli(30);
        assert_eq!(first, second, "cache should return identical values");
    }

    #[test]
    fn large_index_computable() {
        // Smoke test: compute B_50 without panicking.
        let b50 = bernoulli(50);
        // B_50 = 495057205241079648212477525/66
        assert!(!b50.is_zero(), "B_50 should be nonzero");
        // Verify the denominator
        assert_eq!(
            b50.denom(),
            &BigInt::from(66),
            "B_50 denominator should be 66"
        );
    }

    /// The tangent-number values equal the rational recurrence
    /// `B_m = −Σ_{k<m} C(m+1, k)·B_k/(m+1)` (the algorithm of this module up
    /// to 0.39) through `B₃₀₀` — which gives `B₁ = −1/2`, the other
    /// convention (the only number where they differ).
    #[test]
    fn tangent_numbers_match_the_rational_recurrence() {
        let mut b: Vec<Q> = vec![Ratio::one()];
        for m in 1..=300usize {
            let mut sum = Ratio::zero();
            let mut binom: Q = Ratio::one();
            for (k, bk) in b.iter().enumerate() {
                sum += &binom * bk;
                binom = binom * Ratio::from_integer(BigInt::from(m + 1 - k))
                    / Ratio::from_integer(BigInt::from(k + 1));
            }
            b.push(-sum / Ratio::from_integer(BigInt::from(m + 1)));
        }
        for (m, bm) in b.iter().enumerate() {
            let want = if m == 1 { -bm.clone() } else { bm.clone() };
            assert_eq!(bernoulli(m), want, "B_{m}");
        }
    }

    /// The zeta method (used beyond the table) agrees with the tangent
    /// numbers wherever both apply, and the beta method with the secant
    /// numbers.
    #[test]
    fn zeta_and_beta_methods_match_the_tables() {
        for n in (20..=240).step_by(2).chain([510, 512, 514]) {
            assert_eq!(bernoulli_by_zeta(n), Some(bernoulli(n)), "B_{n}");
        }
        for n in (16..=200).step_by(2).chain([512, 514]) {
            let want = crate::base::combinatorics::euler_even(n / 2);
            assert_eq!(euler_by_beta(n), Some(want), "E_{n}");
        }
        assert_eq!(bernoulli_single(514), bernoulli(514));
        assert_eq!(euler_by_beta(7), Some(BigInt::zero()));
    }

    #[test]
    fn alternating_signs_for_even() {
        // B_{2k} for k ≥ 1 alternate in sign:
        // B_2 > 0, B_4 < 0, B_6 > 0, B_8 < 0, ...
        for k in 1..=15 {
            let b = bernoulli(2 * k);
            let f = b.to_f64().unwrap();
            if k % 2 == 1 {
                assert!(f > 0.0, "B_{} should be positive, got {}", 2 * k, f);
            } else {
                assert!(f < 0.0, "B_{} should be negative, got {}", 2 * k, f);
            }
        }
    }
}
