//! Bernoulli number computation and caching.
//!
//! Provides exact rational Bernoulli numbers B_0, B_1, B_2, ...
//! computed via the recurrence relation and cached for reuse across
//! evaluations. Bernoulli numbers are the foundation for the Stirling
//! series used in arbitrary-precision Gamma function evaluation.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};
use parking_lot::Mutex;

/// Global cache of Bernoulli numbers as exact rationals.
/// Lazily computed on demand via the standard recurrence.
static BERNOULLI_CACHE: Mutex<Vec<Ratio<BigInt>>> = Mutex::new(Vec::new());

/// Return B_n (the n-th Bernoulli number) as an exact rational.
///
/// Uses the recurrence: B_0 = 1, and for n ≥ 1:
///   B_n = -1/(n+1) · Σ_{k=0}^{n-1} C(n+1, k) · B_k
///
/// Odd Bernoulli numbers beyond B_1 are zero and are returned
/// immediately without computation. Results are cached globally.
/// Thread-safe via `Mutex`.
#[must_use]
pub(crate) fn bernoulli(n: usize) -> Ratio<BigInt> {
    let mut cache = BERNOULLI_CACHE.lock();

    // Extend cache if needed.
    while cache.len() <= n {
        let m = cache.len();
        if m == 0 {
            cache.push(Ratio::one()); // B_0 = 1
            continue;
        }
        if m == 1 {
            // B_1 = -1/2
            cache.push(Ratio::new(BigInt::from(-1), BigInt::from(2)));
            continue;
        }
        // B_{odd} = 0 for odd indices ≥ 3.
        if m >= 3 && m % 2 == 1 {
            cache.push(Ratio::zero());
            continue;
        }

        // Recurrence: B_m = -1/(m+1) · Σ_{k=0}^{m-1} C(m+1, k) · B_k
        let mut sum = Ratio::zero();
        let mut binom: Ratio<BigInt> = Ratio::one(); // C(m+1, 0) = 1
        for k in 0..m {
            sum += &binom * &cache[k];
            // C(m+1, k+1) = C(m+1, k) · (m+1-k) / (k+1)
            binom = binom * Ratio::from_integer(BigInt::from(m + 1 - k))
                / Ratio::from_integer(BigInt::from(k + 1));
        }
        let result = -sum / Ratio::from_integer(BigInt::from(m + 1));
        tracing::trace!(index = m, value = %result, "bernoulli: computed");
        cache.push(result);
    }

    cache[n].clone()
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
