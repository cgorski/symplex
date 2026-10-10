//! Bernoulli number computation and caching.
//!
//! Provides exact rational Bernoulli numbers B_0, B_1, B_2, ...
//! computed from the tangent numbers (Brent–Harvey) and cached for reuse
//! across evaluations. Bernoulli numbers are the foundation for the
//! Stirling series used in arbitrary-precision Gamma function evaluation.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};
use parking_lot::Mutex;

use crate::base::numeric::Q;

/// Global cache of the even Bernoulli numbers `B₂, B₄, …, B₂ₙ`.
static BERNOULLI_CACHE: Mutex<Vec<Q>> = Mutex::new(Vec::new());

/// Return B_n (the n-th Bernoulli number) as an exact rational, with
/// `B₁ = −1/2`.
///
/// Odd Bernoulli numbers beyond B_1 are zero.  The even ones come from the
/// tangent numbers ([`even`]).  Results are cached globally; thread-safe
/// via `Mutex`.
#[must_use]
pub(crate) fn bernoulli(n: usize) -> Q {
    match n {
        0 => Ratio::one(),
        1 => Ratio::new(BigInt::from(-1), BigInt::from(2)),
        _ if n % 2 == 1 => Ratio::zero(),
        _ => even(n / 2),
    }
}

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
    fn b1_is_neg_half() {
        let b1 = bernoulli(1);
        let expected = Ratio::new(BigInt::from(-1), BigInt::from(2));
        assert_eq!(b1, expected, "B_1 = -1/2");
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
    /// to 0.39) through `B₃₀₀`.
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
            assert_eq!(bernoulli(m), *bm, "B_{m}");
        }
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
