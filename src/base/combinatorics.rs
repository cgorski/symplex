//! Exact integer combinatorics kernels — `n!`, `C(n, k)` and the
//! multinomial — shared by every layer.
//!
//! These are the single implementations behind
//! [`combinatorics::factorial`](crate::combinatorics::factorial),
//! [`combinatorics::binomial`](crate::combinatorics::binomial) and
//! [`combinatorics::multinomial`](crate::combinatorics::multinomial) (the
//! public spellings, re-exported from `domains`); they live in `base` so
//! that the series, summation, ODE and simplification engines below the
//! `domains` layer can call them without an upward dependency.

use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};

/// `n!` as an exact integer, by binary splitting: the product `1·2·…·n`
/// is split in halves recursively so that the big multiplications happen
/// between numbers of similar size (Borwein 1985; the same schedule
/// GMP's `mpz_fac_ui` and Python's `math.factorial` use for their
/// products).  `0! = 1! = 1`.
///
/// ```
/// use symplex::combinatorics::factorial;
/// use num_bigint::BigInt;
///
/// assert_eq!(factorial(0), BigInt::from(1));
/// assert_eq!(factorial(5), BigInt::from(120));
/// assert_eq!(factorial(20), BigInt::from(2_432_902_008_176_640_000u64));
/// assert_eq!(factorial(30).to_string(), "265252859812191058636308480000000");
/// ```
pub fn factorial(n: u64) -> BigInt {
    if n < 2 {
        return BigInt::one();
    }
    product_range(2, n)
}

/// `lo · (lo+1) · … · hi` for `lo ≤ hi`, by binary splitting.
fn product_range(lo: u64, hi: u64) -> BigInt {
    // Below this width the plain running product is faster than the
    // recursion.
    const LEAF: u64 = 16;
    if hi - lo < LEAF {
        let mut acc = BigInt::from(lo);
        for i in lo + 1..=hi {
            acc *= i;
        }
        return acc;
    }
    let mid = lo + (hi - lo) / 2;
    product_range(lo, mid) * product_range(mid + 1, hi)
}

/// Binomial coefficient `C(n, k)` for integer `n` (any sign) and `k ≥ 0`.
///
/// For negative `n` the generalised definition
/// `C(n, k) = n(n−1)⋯(n−k+1)/k!` is used, so
/// `C(−n, k) = (−1)ᵏ C(n+k−1, k)`.  Returns `0` for `k < 0` or
/// `0 ≤ n < k`.
///
/// # Examples
///
/// ```
/// use symplex::combinatorics::binomial;
/// use num_bigint::BigInt;
///
/// assert_eq!(binomial(10, 3), BigInt::from(120));
/// assert_eq!(binomial(5, 7), BigInt::from(0));
/// assert_eq!(binomial(-3, 2), BigInt::from(6));    // (−3)(−4)/2
/// assert_eq!(binomial(100, 50).to_string(), "100891344545564193334812497256");
/// ```
pub fn binomial(n: impl Into<BigInt>, k: impl Into<BigInt>) -> BigInt {
    let n: BigInt = n.into();
    let k: BigInt = k.into();
    if k.is_negative() {
        return BigInt::zero();
    }
    if !n.is_negative() && k > n {
        return BigInt::zero();
    }
    // Use symmetry for non-negative n to keep k small.
    let k = if !n.is_negative() && &k * 2 > n {
        &n - &k
    } else {
        k
    };
    let Some(k) = k.to_u64() else {
        return BigInt::zero();
    };
    let mut result = BigInt::one();
    for i in 0..k {
        result = result * (&n - BigInt::from(i)) / BigInt::from(i + 1);
    }
    result
}

/// Multinomial coefficient `n! / (k₁! · k₂! · … · kₘ!)`.
///
/// This is the number of ways to divide `n` objects into groups of sizes
/// `k₁, k₂, …, kₘ`.
///
/// Returns `Some(0)` if the `kᵢ` don't sum to `n` or if any `kᵢ` is negative.
/// Returns `None` if `n` or any `kᵢ` doesn't fit in `u64`.
///
/// The computation avoids computing full factorials by using incremental
/// products and divisions, keeping intermediate values small.
///
/// # Examples
///
/// ```
/// use symplex::combinatorics::multinomial;
/// use num_bigint::BigInt;
///
/// // 6! / (2! * 3! * 1!) = 60
/// assert_eq!(multinomial(6, &[2, 3, 1]), Some(BigInt::from(60)));
///
/// // Reduces to binomial: C(10, 3) = 120
/// assert_eq!(multinomial(10, &[3, 7]), Some(BigInt::from(120)));
/// ```
pub fn multinomial(n: impl Into<BigInt>, ks: &[impl Into<BigInt> + Clone]) -> Option<BigInt> {
    let n = n.into();
    if n.is_negative() {
        return Some(BigInt::zero());
    }

    let ks_big: Vec<BigInt> = ks.iter().map(|k| k.clone().into()).collect();

    // Check all k_i are non-negative and sum to n.
    let mut sum = BigInt::zero();
    for k in &ks_big {
        if k.is_negative() {
            return Some(BigInt::zero());
        }
        sum += k;
    }
    if sum != n {
        return Some(BigInt::zero());
    }

    let ks_u64: Vec<u64> = ks_big
        .iter()
        .map(|k| k.try_into().ok())
        .collect::<Option<Vec<u64>>>()?;

    Some(multinomial_u64(&ks_u64))
}

/// `(Σ kᵢ)! / Π kᵢ!` for parts that already sum to the total: the
/// product of binomials `C(n, k₁) · C(n − k₁, k₂) · …`, which keeps every
/// intermediate value bounded by one `C(n, kᵢ)`.
pub fn multinomial_u64(ks: &[u64]) -> BigInt {
    let mut result = BigInt::one();
    let mut remaining: u64 = ks.iter().sum();

    for &k in ks {
        if k == 0 {
            continue;
        }
        // Incremental binomial: C(remaining, k) = ∏_{i=0}^{k-1} (remaining - i) / (i + 1)
        let mut binom = BigInt::one();
        for i in 0..k {
            binom *= BigInt::from(remaining - i);
            binom /= BigInt::from(i + 1);
        }
        result *= binom;
        remaining -= k;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factorial_matches_the_running_product() {
        let mut acc = BigInt::one();
        assert_eq!(factorial(0), acc);
        for n in 1..=200u64 {
            acc *= n;
            assert_eq!(factorial(n), acc, "{n}!");
        }
    }

    #[test]
    fn binomial_symmetry_and_pascal() {
        for n in 0..30i64 {
            for k in 0..=n {
                assert_eq!(binomial(n, k), binomial(n, n - k));
                if k > 0 {
                    assert_eq!(
                        binomial(n + 1, k),
                        binomial(n, k - 1) + binomial(n, k),
                        "Pascal at ({n}, {k})"
                    );
                }
            }
        }
        assert_eq!(binomial(5, -1), BigInt::zero());
        assert_eq!(binomial(-1, 3), BigInt::from(-1));
        assert_eq!(binomial(-2, 3), BigInt::from(-4));
    }

    #[test]
    fn multinomial_parts() {
        assert_eq!(multinomial_u64(&[2, 3, 1]), BigInt::from(60));
        assert_eq!(multinomial_u64(&[0, 4]), BigInt::one());
        assert_eq!(multinomial(4, &[1, 1, 1, 1]), Some(BigInt::from(24)));
        assert_eq!(multinomial(4, &[1, 1, 1]), Some(BigInt::zero()));
        assert_eq!(multinomial(-1, &[1]), Some(BigInt::zero()));
        assert_eq!(multinomial(3, &[-1, 4]), Some(BigInt::zero()));
    }
}
