//! Exact integer combinatorics kernels — `n!`, `C(n, k)`, the multinomial,
//! the Fibonacci, Lucas and Euler numbers — shared by every layer.
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
/// With `j = min(k, n − k)` (after the reflection for a negative `n`), a
/// small `j` runs the product `n(n−1)⋯(n−j+1)/j!` term by term; a large
/// one (`n ≤ 2²⁸`) multiplies the prime powers of the result, the exponent
/// of `p` being the number of carries when `j` and `n − j` are added in
/// base `p` (Kummer, 1852), as a balanced product.  Up to 0.39 every `j`
/// ran the term-by-term product, quadratic in the result's size:
/// `catalan(300000)` took 16 s.  A result with `j ≥ 2⁶⁴` (more than `2⁶⁴`
/// bits) cannot be represented and gives `0`; up to 0.39 so did a negative
/// `n` with `k ≥ 2⁶⁴`, where `C(−1, k) = (−1)ᵏ`.
///
/// Like [`factorial`], this always computes: the result has at least
/// `j·log₂(n/j)` bits, and the caller bounds it (`eval` by its digit guard,
/// the `Option` functions of `combinatorics` by `MAX_RESULT_BITS`).
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
/// assert_eq!(binomial(-1, BigInt::from(1) << 64), BigInt::from(1));
/// ```
pub fn binomial(n: impl Into<BigInt>, k: impl Into<BigInt>) -> BigInt {
    let n: BigInt = n.into();
    let k: BigInt = k.into();
    if k.is_negative() {
        return BigInt::zero();
    }
    if n.is_negative() {
        // C(−m, k) = (−1)ᵏ·C(m + k − 1, k).
        let c = binomial_nonnegative(&(&k - &n - 1u32), &k);
        return if (&k % 2u32).is_one() { -c } else { c };
    }
    binomial_nonnegative(&n, &k)
}

/// Smallest `j = min(k, n − k)` for which [`binomial`] considers the
/// prime-power product.
const KUMMER_MIN_J: u64 = 256;

/// Largest `n` of [`binomial`]'s prime-power product (its sieve of the
/// odd numbers up to `n` takes `n/16` bytes: 16 MiB).
const KUMMER_MAX_N: u64 = 1 << 28;

/// `C(n, k)` for `n, k ≥ 0`.
fn binomial_nonnegative(n: &BigInt, k: &BigInt) -> BigInt {
    if k > n {
        return BigInt::zero();
    }
    let j = (n - k).min(k.clone());
    let Some(j) = j.to_u64() else {
        return BigInt::zero();
    };
    if let Some(n_u64) = n.to_u64()
        && j >= KUMMER_MIN_J
        && n_u64 <= KUMMER_MAX_N
    {
        // The term-by-term product costs about j·(bits of the result)/64
        // word operations, the prime-power product a sieve to n.
        let (jf, nf) = (j as f64, n_u64 as f64);
        let product_cost = jf * jf * (std::f64::consts::E * nf / jf).log2() / 64.0;
        if product_cost > nf {
            return binomial_by_prime_powers(n_u64, j);
        }
    }
    let mut result = BigInt::one();
    for i in 0..j {
        result = result * (n - BigInt::from(i)) / BigInt::from(i + 1);
    }
    result
}

/// `C(n, j)` for `j ≤ n ≤` [`KUMMER_MAX_N`] as `∏ p^{e_p}` over the primes
/// `p ≤ n`, `e_p = Σᵢ (⌊n/pⁱ⌋ − ⌊j/pⁱ⌋ − ⌊(n−j)/pⁱ⌋)` (Legendre's formula
/// for `n!/(j!·(n−j)!)`; Kummer: `p^{e_p} ≤ n`, so each prime power fits
/// in a word).
fn binomial_by_prime_powers(n: u64, j: u64) -> BigInt {
    let m = n - j;
    let exponent = |p: u64| -> u32 {
        let (mut e, mut q) = (0u32, p);
        loop {
            // Each term is 0 or 1 (a carry).
            e += u32::from(n / q - j / q - m / q > 0);
            match q.checked_mul(p) {
                Some(next) if next <= n => q = next,
                _ => return e,
            }
        }
    };
    let mut words: Vec<BigInt> = Vec::new();
    let mut acc: u64 = 1;
    let mut push = |p: u64| {
        let e = exponent(p);
        if e == 0 {
            return;
        }
        let f = p.pow(e);
        match acc.checked_mul(f) {
            Some(v) => acc = v,
            None => {
                words.push(BigInt::from(acc));
                acc = f;
            }
        }
    };
    push(2);
    // Sieve of Eratosthenes over the odd numbers: bit i stands for 2i + 1.
    let half = usize::try_from(n.div_ceil(2)).unwrap_or(0);
    let mut composite = vec![0u64; half.div_ceil(64)];
    let is_composite = |c: &[u64], i: usize| c[i / 64] >> (i % 64) & 1 == 1;
    for i in 1..half {
        if is_composite(&composite, i) {
            continue;
        }
        let p = 2 * i as u64 + 1;
        push(p);
        // Odd multiples from p² on: index (p² − 1)/2, step p.
        if let Some(sq) = p.checked_mul(p).filter(|&sq| sq <= n) {
            let mut idx = usize::try_from((sq - 1) / 2).unwrap_or(half);
            let step = usize::try_from(p).unwrap_or(half);
            while idx < half {
                composite[idx / 64] |= 1 << (idx % 64);
                idx += step;
            }
        }
    }
    words.push(BigInt::from(acc));
    // Balanced product: pairs of similar size.
    while words.len() > 1 {
        let mut next = Vec::with_capacity(words.len().div_ceil(2));
        let mut it = words.into_iter();
        while let Some(a) = it.next() {
            next.push(match it.next() {
                Some(b) => a * b,
                None => a,
            });
        }
        words = next;
    }
    words.pop().unwrap_or_else(BigInt::one)
}

/// Fibonacci number `Fₙ` (`F₀ = 0, F₁ = 1`) for any integer `n`, with
/// `F₋ₙ = (−1)ⁿ⁺¹·Fₙ`, by fast doubling (`O(log n)` multiplications).  The
/// single implementation behind `ntheory::fibonacci` and `eval`'s
/// `fibonacci(n)`; `|n|` beyond `u64` is taken as `u64::MAX` (a value of
/// `2⁶³` bits, which no caller reaches: they bound `n` first).
pub(crate) fn fibonacci(n: i128) -> BigInt {
    let m = u64::try_from(n.unsigned_abs()).unwrap_or(u64::MAX);
    let (f, _) = fibonacci_pair(m);
    if n < 0 && m.is_multiple_of(2) { -f } else { f }
}

/// Lucas number `Lₙ = 2Fₙ₊₁ − Fₙ` (`L₀ = 2, L₁ = 1`) for any integer `n`,
/// with `L₋ₙ = (−1)ⁿ·Lₙ`; see [`fibonacci`].
pub(crate) fn lucas(n: i128) -> BigInt {
    let m = u64::try_from(n.unsigned_abs()).unwrap_or(u64::MAX);
    let (f, f1) = fibonacci_pair(m);
    let l: BigInt = &f1 * 2 - &f;
    if n < 0 && m % 2 == 1 { -l } else { l }
}

/// `(Fₙ, Fₙ₊₁)` by fast doubling, iterating over the bits of `n`.
fn fibonacci_pair(n: u64) -> (BigInt, BigInt) {
    let mut a = BigInt::zero(); // F_k
    let mut b = BigInt::one(); // F_{k+1}
    let bits = 64 - n.leading_zeros();
    for i in (0..bits).rev() {
        // (F_k, F_{k+1}) → (F_{2k}, F_{2k+1})
        let a2 = &a * (&b * 2 - &a); // F_{2k} = F_k (2F_{k+1} − F_k)
        let b2 = &a * &a + &b * &b; // F_{2k+1} = F_k² + F_{k+1}²
        if (n >> i) & 1 == 1 {
            a = b2.clone();
            b = a2 + b2; // F_{2k+2} = F_{2k} + F_{2k+1}
        } else {
            a = a2;
            b = b2;
        }
    }
    (a, b)
}

/// The Euler number `E₂ₘ` (`1, −1, 5, −61, 1385, …`): `(−1)ᵐ` times the
/// secant number `Sₘ`, from Brent and Harvey's Algorithm SecantNumbers
/// ("Fast computation of Bernoulli, Tangent and Secant numbers", 2011):
/// `O(m²)` updates by small factors, in place.  No bound on `m` here: the
/// callers bound it (`ntheory::euler_number` by `m ≤ 2048`, `eval` by its
/// digit guard).
pub(crate) fn euler_even(m: usize) -> BigInt {
    let mut s: Vec<BigInt> = Vec::with_capacity(m + 1);
    s.push(BigInt::one());
    for k in 1..=m {
        let next = &s[k - 1] * BigInt::from(k);
        s.push(next);
    }
    for k in 1..=m {
        for j in k + 1..=m {
            s[j] = &s[j - 1] * BigInt::from(j - k) + &s[j] * BigInt::from(j - k + 1);
        }
    }
    let s_m = s.pop().unwrap_or_else(BigInt::one);
    if m.is_multiple_of(2) { s_m } else { -s_m }
}

/// Multinomial coefficient `n! / (k₁! · k₂! · … · kₘ!)`.
///
/// This is the number of ways to divide `n` objects into groups of sizes
/// `k₁, k₂, …, kₘ`.
///
/// Returns `Some(0)` if the `kᵢ` don't sum to `n` or if any `kᵢ` is negative.
/// Returns `None` if `n` or any `kᵢ` doesn't fit in `u64`, or if the result
/// would have more than
/// [`MAX_RESULT_BITS`](crate::combinatorics::MAX_RESULT_BITS) bits (decided
/// from the lower bound `C(m, j) ≥ (m/j)ʲ` of each binomial factor).
///
/// The coefficient is the product of the binomials `C(n, k₁)·C(n − k₁, k₂)⋯`
/// ([`binomial`]).  Up to 0.39 an `n` beyond `u64` overflowed the sum of
/// the parts (`multinomial(2⁶⁴ + 1, &[2⁶⁴ − 1, 2])` panicked in a debug
/// build), a result of `2⁴⁰` bits was attempted, and every binomial was a
/// term-by-term product (`multinomial(4·10⁶, &[2·10⁶, 2·10⁶])` ran for
/// minutes).
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
///
/// // n beyond u64, and a result of about 2⁴⁰ bits
/// let big = BigInt::from(u64::MAX) + 2;
/// assert_eq!(multinomial(big, &[BigInt::from(u64::MAX), BigInt::from(2)]), None);
/// assert_eq!(multinomial(1u64 << 40, &[1u64 << 39, 1u64 << 39]), None);
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

    let n_u64: u64 = (&n).try_into().ok()?;
    let ks_u64: Vec<u64> = ks_big
        .iter()
        .map(|k| k.try_into().ok())
        .collect::<Option<Vec<u64>>>()?;

    // The parts sum to n: each factor is C(m, k) with k ≤ m.
    let mut remaining = n_u64;
    let mut lower_bits = 0.0;
    for &k in &ks_u64 {
        let j = k.min(remaining - k);
        if j > 0 {
            lower_bits += j as f64 * (remaining as f64 / j as f64).log2();
        }
        remaining -= k;
    }
    if lower_bits > MULTINOMIAL_MAX_BITS {
        return None;
    }
    Some(multinomial_u64(&ks_u64))
}

/// `combinatorics::MAX_RESULT_BITS` (`2³²`), the bound of
/// [`multinomial`] (the public constant lives in `domains`).
const MULTINOMIAL_MAX_BITS: f64 = 4_294_967_296.0;

/// `(Σ kᵢ)! / Π kᵢ!` for parts that already sum to the total: the
/// product of binomials `C(n, k₁) · C(n − k₁, k₂) · …` ([`binomial`]).
/// The caller bounds the size of the result.
pub fn multinomial_u64(ks: &[u64]) -> BigInt {
    let mut result = BigInt::one();
    let mut remaining: u128 = ks.iter().map(|&k| u128::from(k)).sum();

    for &k in ks {
        if k == 0 {
            continue;
        }
        result *= binomial_nonnegative(&BigInt::from(remaining), &BigInt::from(k));
        remaining -= u128::from(k);
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

    /// The prime-power product equals the term-by-term one, across the
    /// switch between them (and for `j` near `n/2`, `n` odd and even).
    #[test]
    fn prime_power_product_matches_the_running_product() {
        let running = |n: u64, j: u64| -> BigInt {
            let mut r = BigInt::one();
            for i in 0..j {
                r = r * BigInt::from(n - i) / BigInt::from(i + 1);
            }
            r
        };
        for (n, j) in [
            (512u64, 256u64),
            (513, 256),
            (1000, 300),
            (2000, 1000),
            (2001, 1000),
            (4096, 2048),
            (10007, 5000),
            (65536, 300),
        ] {
            assert_eq!(binomial_by_prime_powers(n, j), running(n, j), "C({n}, {j})");
            assert_eq!(binomial(n, j), running(n, j), "C({n}, {j})");
            assert_eq!(binomial(n, n - j), running(n, j), "C({n}, {})", n - j);
        }
        // Negative n with a k beyond u64: C(−1, k) = (−1)ᵏ, C(−3, k) =
        // (−1)ᵏ·(k + 1)(k + 2)/2.
        let k = BigInt::from(1u8) << 64usize;
        assert_eq!(binomial(-1, k.clone()), BigInt::one());
        assert_eq!(binomial(-1, &k + 1u32), -BigInt::one());
        assert_eq!(binomial(-3, &k + 1u32), -((&k + 2u32) * (&k + 3u32) / 2u32));
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
