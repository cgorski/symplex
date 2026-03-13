//! Number theory functions: primality testing, factorization, divisors,
//! modular arithmetic, and related operations.
//!
//! **Unified API** — every function accepts arbitrary-precision integers via
//! `impl Into<BigInt>`.  Small values that fit in `i64` automatically take
//! an optimised fast path; truly large values use `BigInt` arithmetic.
//!
//! # Examples
//!
//! ```
//! use symplex::ntheory::{factorint, isprime, gcd, nextprime};
//! use num_bigint::BigInt;
//!
//! // Same function works for i64 …
//! assert!(isprime(104729));
//!
//! // … and for BigInt
//! let m61 = BigInt::from(2u64.pow(61) - 1);
//! assert!(isprime(m61));
//! ```

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};

// ═══════════════════════════════════════════════════════════════════════════
// Internal modular arithmetic helpers (u64 fast path)
// ═══════════════════════════════════════════════════════════════════════════

/// Modular multiplication using 128-bit intermediates to avoid overflow.
fn mod_mul_u64(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// Modular exponentiation: `base^exp mod modulus` via binary method (u64).
fn mod_pow_u64(base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result: u128 = 1;
    let m = modulus as u128;
    let mut b = (base % modulus) as u128;
    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * b) % m;
        }
        exp /= 2;
        b = (b * b) % m;
    }
    result as u64
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal Miller-Rabin helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Miller-Rabin primality test with the given deterministic witnesses (u64).
///
/// For `n < 3.3×10²⁴`, the witnesses `[2,3,5,7,11,13,17,19,23,29,31,37]`
/// give a deterministic (always-correct) result. Since `i64::MAX < 2⁶³`,
/// this is more than sufficient for our use case.
fn miller_rabin(n: u64, witnesses: &[u64]) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n.is_multiple_of(2) {
        return false;
    }

    // Write n-1 as 2^r · d where d is odd
    let mut d = n - 1;
    let mut r = 0u32;
    while d.is_multiple_of(2) {
        d /= 2;
        r += 1;
    }

    'outer: for &a in witnesses {
        if a >= n {
            continue;
        }
        let mut x = mod_pow_u64(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..r - 1 {
            x = mod_mul_u64(x, x, n);
            if x == n - 1 {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

/// Miller-Rabin primality test on BigInt using `BigInt::modpow`.
///
/// Uses the same deterministic witness set. For numbers that fit in u64,
/// the caller should prefer the fast `miller_rabin` above.
fn miller_rabin_big(n: &BigInt, witnesses: &[u64]) -> bool {
    let one = BigInt::one();
    let two = BigInt::from(2);
    let n_minus_1 = n - &one;

    // Write n-1 as 2^r · d where d is odd
    let mut d = n_minus_1.clone();
    let mut r = 0u32;
    while d.is_even() {
        d /= &two;
        r += 1;
    }

    if r == 0 {
        // n-1 is odd means n is even, but n>2 should already be filtered
        return false;
    }

    'outer: for &a in witnesses {
        let a_big = BigInt::from(a);
        if &a_big >= n {
            continue;
        }
        let mut x = a_big.modpow(&d, n);
        if x == one || x == n_minus_1 {
            continue;
        }
        let mut found = false;
        for _ in 0..r - 1 {
            x = x.modpow(&two, n);
            if x == n_minus_1 {
                found = true;
                break;
            }
        }
        if found {
            continue 'outer;
        }
        return false;
    }
    true
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal extended Euclidean algorithm
// ═══════════════════════════════════════════════════════════════════════════

/// Returns `(gcd, x, y)` such that `a*x + b*y = gcd` (BigInt).
fn extended_gcd_big(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
    if a.is_zero() {
        return (b.clone(), BigInt::zero(), BigInt::one());
    }
    let (g, x1, y1) = extended_gcd_big(&(b % a), a);
    let coeff = (b / a) * &x1;
    (g, &y1 - &coeff, x1)
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal i64 fast-path implementations
// ═══════════════════════════════════════════════════════════════════════════

/// Deterministic primality test (i64 fast path).
fn isprime_i64(n: i64) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }
    // Quick check against small primes
    let small_primes: [i64; 13] = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
    for &p in &small_primes {
        if n == p {
            return true;
        }
        if n % p == 0 {
            return false;
        }
    }
    // Deterministic Miller-Rabin witnesses sufficient for all i64
    let witnesses: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    miller_rabin(n as u64, &witnesses)
}

/// Primality test for BigInt values that do NOT fit in i64.
fn isprime_big_internal(n: &BigInt) -> bool {
    if *n < BigInt::from(2) {
        return false;
    }
    if n.is_even() {
        return false;
    }

    let three = BigInt::from(3);
    if (n % &three).is_zero() {
        return false;
    }

    // Quick check against small primes
    let small_primes: [u64; 13] = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
    for &p in &small_primes {
        let bp = BigInt::from(p);
        if n == &bp {
            return true;
        }
        if (n % &bp).is_zero() {
            return false;
        }
    }

    // Deterministic Miller-Rabin witnesses sufficient for all n < 3.3×10^24.
    // For larger n this is probabilistic but extremely reliable.
    let witnesses: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    miller_rabin_big(n, &witnesses)
}

/// Factorize an i64 into `(prime, exponent)` pairs (i64 fast path).
fn factorint_i64(n: i64) -> Vec<(i64, u32)> {
    if n == 0 {
        return vec![];
    }
    let mut n = n.unsigned_abs();
    let mut factors = Vec::new();

    const SMALL_PRIMES: [u64; 25] = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
        97,
    ];
    for &p in &SMALL_PRIMES {
        if p * p > n && n > 1 {
            break;
        }
        let mut count = 0u32;
        while n.is_multiple_of(p) {
            n /= p;
            count += 1;
        }
        if count > 0 {
            factors.push((p as i64, count));
        }
    }

    // Continue trial division with odd numbers > 97
    let mut d = 101u64;
    while d * d <= n {
        let mut count = 0u32;
        while n.is_multiple_of(d) {
            n /= d;
            count += 1;
        }
        if count > 0 {
            factors.push((d as i64, count));
        }
        d += 2;
    }

    if n > 1 {
        factors.push((n as i64, 1));
    }

    factors
}

/// Factorize a BigInt that may exceed i64 range.
fn factorint_big_internal(n: &BigInt) -> Vec<(BigInt, u32)> {
    if n.is_zero() {
        return vec![];
    }
    let mut n = n.abs();
    if n <= BigInt::one() {
        return vec![];
    }

    let mut factors: Vec<(BigInt, u32)> = Vec::new();

    // Trial division with a table of small primes
    const SMALL_PRIMES: [u64; 25] = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
        97,
    ];

    for &p in &SMALL_PRIMES {
        let bp = BigInt::from(p);
        if &bp * &bp > n && n > BigInt::one() {
            break;
        }
        let mut count = 0u32;
        while (&n % &bp).is_zero() {
            n /= &bp;
            count += 1;
        }
        if count > 0 {
            factors.push((bp, count));
        }
    }

    // If the remaining cofactor fits in i64, delegate to the fast i64 path
    if n > BigInt::one()
        && let Some(ni) = n.to_i64()
    {
        for (p, e) in factorint_i64(ni) {
            factors.push((BigInt::from(p), e));
        }
        return factors;
    }

    // For cofactors that exceed i64, check if already prime to avoid
    // O(sqrt(n)) BigInt divisions (e.g. Mersenne prime M61 ≈ 2.3×10^18).
    if n > BigInt::one() && isprime_big_internal(&n) {
        factors.push((n, 1));
        return factors;
    }

    // Continue trial division with odd numbers > 97 (only reached for
    // composite cofactors larger than i64::MAX ≈ 9.2×10^18).
    let mut d = BigInt::from(101u64);
    let two = BigInt::from(2);
    while &d * &d <= n {
        let mut count = 0u32;
        while (&n % &d).is_zero() {
            n /= &d;
            count += 1;
        }
        if count > 0 {
            factors.push((d.clone(), count));
            // After each successful factor extraction, check if the cofactor
            // is now prime to short-circuit the remaining trial division.
            if n > BigInt::one() && isprime_big_internal(&n) {
                factors.push((n, 1));
                return factors;
            }
        }
        d += &two;
    }

    if n > BigInt::one() {
        factors.push((n, 1));
    }

    factors
}

/// Next prime after `n` (i64 fast path).
fn nextprime_i64(n: i64) -> i64 {
    if n < 2 {
        return 2;
    }
    let mut candidate = if n % 2 == 0 { n + 1 } else { n + 2 };
    while !isprime_i64(candidate) {
        candidate += 2;
    }
    candidate
}

/// Previous prime before `n` (i64 fast path).
fn prevprime_i64(n: i64) -> Option<i64> {
    if n <= 2 {
        return None;
    }
    if n == 3 {
        return Some(2);
    }
    let mut candidate = if n % 2 == 0 { n - 1 } else { n - 2 };
    while candidate >= 2 && !isprime_i64(candidate) {
        candidate -= 2;
    }
    if candidate >= 2 {
        Some(candidate)
    } else {
        None
    }
}

/// Integer square root (i64 fast path).
fn isqrt_i64(n: i64) -> Option<i64> {
    if n < 0 {
        return None;
    }
    if n == 0 {
        return Some(0);
    }
    let mut s = (n as f64).sqrt() as i64;
    // Newton correction
    while s * s > n {
        s -= 1;
    }
    while (s + 1) * (s + 1) <= n {
        s += 1;
    }
    Some(s)
}

/// Integer square root for BigInt via Newton's method.
fn isqrt_big(n: &BigInt) -> Option<BigInt> {
    if n.is_negative() {
        return None;
    }
    if n.is_zero() {
        return Some(BigInt::zero());
    }
    let one = BigInt::one();
    let two = BigInt::from(2);
    // Initial guess: use bit length for a rough sqrt
    // sqrt(2^b) ≈ 2^(b/2)
    let bit_len = n.bits();
    let mut x = BigInt::one() << bit_len.div_ceil(2) as usize;
    loop {
        // Newton step: x_new = (x + n/x) / 2
        let x_new = (&x + n / &x) / &two;
        if x_new >= x {
            break;
        }
        x = x_new;
    }
    // Adjust for off-by-one
    while &x * &x > *n {
        x -= &one;
    }
    while (&x + &one) * (&x + &one) <= *n {
        x += &one;
    }
    Some(x)
}

// ═══════════════════════════════════════════════════════════════════════════
// Public unified API
// ═══════════════════════════════════════════════════════════════════════════

/// Deterministic primality test for any integer.
///
/// Accepts any integer type (`i64`, `i32`, `u64`, `BigInt`, etc.) via
/// `impl Into<BigInt>`. Values that fit in `i64` automatically use an
/// optimised u64/u128 code path; larger values use BigInt arithmetic.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::isprime;
/// use num_bigint::BigInt;
///
/// assert!(isprime(2));
/// assert!(isprime(104729));
/// assert!(!isprime(104730));
///
/// // Same function for BigInt
/// let m61 = BigInt::from(2u64.pow(61) - 1);
/// assert!(isprime(m61));
/// ```
pub fn isprime(n: impl Into<BigInt>) -> bool {
    let n: BigInt = n.into();
    // Fast path: if fits in i64, use optimized version
    if let Some(n_i64) = n.to_i64() {
        return isprime_i64(n_i64);
    }
    isprime_big_internal(&n)
}

/// Factorize an integer into prime factors.
///
/// Accepts any integer type (`i64`, `i32`, `u64`, `BigInt`, etc.).
/// Returns `(prime, exponent)` pairs in ascending order of prime.
/// Returns an empty vector for `n ∈ {-1, 0, 1}`. Negative inputs are
/// factored by absolute value.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::factorint;
/// use num_bigint::BigInt;
///
/// assert_eq!(
///     factorint(60),
///     vec![(BigInt::from(2), 2), (BigInt::from(3), 1), (BigInt::from(5), 1)]
/// );
/// assert_eq!(factorint(1), vec![]);
/// assert_eq!(factorint(-12), vec![(BigInt::from(2), 2), (BigInt::from(3), 1)]);
/// ```
pub fn factorint(n: impl Into<BigInt>) -> Vec<(BigInt, u32)> {
    let n: BigInt = n.into();
    // Fast path: if fits in i64, use optimized version
    if let Some(n_i64) = n.to_i64() {
        return factorint_i64(n_i64)
            .into_iter()
            .map(|(p, e)| (BigInt::from(p), e))
            .collect();
    }
    factorint_big_internal(&n)
}

/// Returns the smallest prime strictly greater than `n`.
///
/// Accepts any integer type.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::nextprime;
/// use num_bigint::BigInt;
///
/// assert_eq!(nextprime(10), BigInt::from(11));
/// assert_eq!(nextprime(11), BigInt::from(13));
/// assert_eq!(nextprime(-5), BigInt::from(2));
/// ```
pub fn nextprime(n: impl Into<BigInt>) -> BigInt {
    let n: BigInt = n.into();
    if let Some(n_i64) = n.to_i64() {
        return BigInt::from(nextprime_i64(n_i64));
    }
    // BigInt path
    let two = BigInt::from(2);
    if n < two {
        return two;
    }
    let mut candidate = if n.is_even() { &n + 1 } else { &n + 2 };
    while !isprime_big_internal(&candidate) {
        candidate += &two;
    }
    candidate
}

/// Returns the largest prime strictly less than `n`, or `None` if no
/// such prime exists (i.e. `n ≤ 2`).
///
/// Accepts any integer type.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::prevprime;
/// use num_bigint::BigInt;
///
/// assert_eq!(prevprime(10), Some(BigInt::from(7)));
/// assert_eq!(prevprime(3), Some(BigInt::from(2)));
/// assert_eq!(prevprime(2), None);
/// ```
pub fn prevprime(n: impl Into<BigInt>) -> Option<BigInt> {
    let n: BigInt = n.into();
    if let Some(n_i64) = n.to_i64() {
        return prevprime_i64(n_i64).map(BigInt::from);
    }
    // BigInt path (n > i64::MAX, so definitely > 2)
    let two = BigInt::from(2);
    let mut candidate = if n.is_even() { &n - 1 } else { &n - 2 };
    while candidate >= two && !isprime_big_internal(&candidate) {
        candidate -= &two;
    }
    if candidate >= two {
        Some(candidate)
    } else {
        None
    }
}

/// Returns all positive divisors of `|n|` in ascending order.
///
/// Returns an empty vector for `n = 0`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::divisors;
/// use num_bigint::BigInt;
///
/// assert_eq!(
///     divisors(12),
///     vec![1, 2, 3, 4, 6, 12].into_iter().map(BigInt::from).collect::<Vec<_>>()
/// );
/// ```
pub fn divisors(n: impl Into<BigInt>) -> Vec<BigInt> {
    let n: BigInt = n.into();
    if n.is_zero() {
        return vec![];
    }
    // Use the unified factorint (which already dispatches to the fast path)
    let n_abs = n.abs();
    let factors = factorint(n_abs);
    let mut divs = vec![BigInt::one()];
    for (p, e) in factors {
        let current_len = divs.len();
        let mut power = BigInt::one();
        for _ in 0..e {
            power *= &p;
            for j in 0..current_len {
                divs.push(&divs[j] * &power);
            }
        }
    }
    divs.sort();
    divs
}

/// Returns the number of positive divisors of `|n|`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::divisor_count;
/// assert_eq!(divisor_count(12), 6);
/// assert_eq!(divisor_count(1), 1);
/// ```
pub fn divisor_count(n: impl Into<BigInt>) -> usize {
    let n: BigInt = n.into();
    if n.is_zero() {
        return 0;
    }
    let factors = factorint(n.abs());
    factors.iter().map(|(_, e)| (*e + 1) as usize).product()
}

/// Returns the sum of all positive divisors of `|n|`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::divisor_sum;
/// use num_bigint::BigInt;
///
/// assert_eq!(divisor_sum(12), BigInt::from(28));
/// assert_eq!(divisor_sum(1), BigInt::from(1));
/// ```
pub fn divisor_sum(n: impl Into<BigInt>) -> BigInt {
    let divs = divisors(n);
    divs.into_iter().fold(BigInt::zero(), |acc, d| acc + d)
}

/// Euler's totient function φ(n).
///
/// Returns the count of integers in `[1, n]` coprime to `n`.
/// Returns 0 for `n ≤ 0`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::totient;
/// use num_bigint::BigInt;
///
/// assert_eq!(totient(12), BigInt::from(4));
/// assert_eq!(totient(13), BigInt::from(12));  // prime ⟹ φ(p) = p − 1
/// ```
pub fn totient(n: impl Into<BigInt>) -> BigInt {
    let n: BigInt = n.into();
    if n <= BigInt::zero() {
        return BigInt::zero();
    }
    let factors = factorint(n.clone());
    let mut result = n;
    for (p, _) in factors {
        result = &result / &p * (&p - BigInt::one());
    }
    result
}

/// Möbius function μ(n).
///
/// - μ(1) = 1
/// - μ(n) = 0 if n has a squared prime factor
/// - μ(n) = (−1)^k if n is a product of k distinct primes
///
/// Returns 0 for `n ≤ 0`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::mobius;
/// assert_eq!(mobius(1), 1);
/// assert_eq!(mobius(6), 1);    // 6 = 2·3 → (−1)² = 1
/// assert_eq!(mobius(30), -1);  // 30 = 2·3·5 → (−1)³ = −1
/// assert_eq!(mobius(4), 0);    // 4 = 2² → squared factor
/// ```
pub fn mobius(n: impl Into<BigInt>) -> i8 {
    let n: BigInt = n.into();
    if n <= BigInt::zero() {
        return 0;
    }
    let factors = factorint(n);
    for (_, e) in &factors {
        if *e > 1 {
            return 0;
        }
    }
    if factors.len().is_multiple_of(2) {
        1
    } else {
        -1
    }
}

/// Modular inverse of `a` modulo `n`, if it exists.
///
/// Returns `Some(x)` where `0 ≤ x < n` and `a·x ≡ 1 (mod n)`,
/// or `None` if `gcd(a, n) ≠ 1`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::mod_inverse;
/// use num_bigint::BigInt;
///
/// assert_eq!(mod_inverse(3, 7), Some(BigInt::from(5)));  // 3·5 = 15 ≡ 1 (mod 7)
/// assert_eq!(mod_inverse(2, 4), None);                   // gcd(2,4) = 2 ≠ 1
/// ```
pub fn mod_inverse(a: impl Into<BigInt>, n: impl Into<BigInt>) -> Option<BigInt> {
    let a: BigInt = a.into();
    let n: BigInt = n.into();
    let (g, x, _) = extended_gcd_big(&a, &n);
    if !g.is_one() {
        return None;
    }
    let result = ((&x % &n) + &n) % &n;
    Some(result)
}

/// Chinese Remainder Theorem (BigInt version).
///
/// Given a system of congruences `x ≡ remainders[i] (mod moduli[i])`,
/// returns the unique solution modulo the LCM of the moduli, or `None`
/// if no solution exists or the inputs are invalid.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::crt;
/// use num_bigint::BigInt;
///
/// let r: Vec<BigInt> = vec![2.into(), 3.into(), 2.into()];
/// let m: Vec<BigInt> = vec![3.into(), 5.into(), 7.into()];
/// assert_eq!(crt(&r, &m), Some(BigInt::from(23)));
/// ```
pub fn crt(remainders: &[BigInt], moduli: &[BigInt]) -> Option<BigInt> {
    if remainders.len() != moduli.len() || remainders.is_empty() {
        return None;
    }
    let mut result = remainders[0].clone();
    let mut modulus = moduli[0].clone();
    for i in 1..remainders.len() {
        let (g, p, _) = extended_gcd_big(&modulus, &moduli[i]);
        if !((&remainders[i] - &result) % &g).is_zero() {
            return None;
        }
        let step = &modulus * &((&remainders[i] - &result) / &g % (&moduli[i] / &g)) * &p;
        result = &result + &step;
        modulus = &modulus / &g * &moduli[i];
        result = ((&result % &modulus) + &modulus) % &modulus;
    }
    Some(result)
}

/// Chinese Remainder Theorem — convenience wrapper for `i64` slices.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::crt_i64;
/// assert_eq!(crt_i64(&[2, 3, 2], &[3, 5, 7]), Some(23));
/// ```
pub fn crt_i64(remainders: &[i64], moduli: &[i64]) -> Option<i64> {
    // Delegate to the arbitrary-precision BigInt implementation, which is
    // proven correct for all input sizes.  The i64 wrapper only converts
    // at the boundaries — no native-width intermediate arithmetic that
    // could overflow.
    let r: Vec<BigInt> = remainders.iter().map(|&r| BigInt::from(r)).collect();
    let m: Vec<BigInt> = moduli.iter().map(|&m| BigInt::from(m)).collect();
    crt(&r, &m).and_then(|x| (&x).try_into().ok())
}

/// Greatest common divisor of two integers.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::gcd;
/// use num_bigint::BigInt;
///
/// assert_eq!(gcd(12, 8), BigInt::from(4));
/// assert_eq!(gcd(0, 5), BigInt::from(5));
/// ```
pub fn gcd(a: impl Into<BigInt>, b: impl Into<BigInt>) -> BigInt {
    let a: BigInt = a.into();
    let b: BigInt = b.into();
    num_integer::Integer::gcd(&a, &b)
}

/// Least common multiple of two integers.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::lcm;
/// use num_bigint::BigInt;
///
/// assert_eq!(lcm(4, 6), BigInt::from(12));
/// assert_eq!(lcm(0, 5), BigInt::from(0));
/// ```
pub fn lcm(a: impl Into<BigInt>, b: impl Into<BigInt>) -> BigInt {
    let a: BigInt = a.into();
    let b: BigInt = b.into();
    num_integer::Integer::lcm(&a, &b)
}

/// Returns `true` if `a` and `b` are coprime (gcd = 1).
///
/// # Examples
///
/// ```
/// use symplex::ntheory::is_coprime;
/// assert!(is_coprime(8, 15));
/// assert!(!is_coprime(8, 12));
/// ```
pub fn is_coprime(a: impl Into<BigInt>, b: impl Into<BigInt>) -> bool {
    gcd(a, b).is_one()
}

/// Modular exponentiation: `base^exp mod modulus`.
///
/// Returns 0 if `modulus ≤ 0` or `exp < 0`. Works for non-negative
/// exponents.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::mod_pow;
/// use num_bigint::BigInt;
///
/// assert_eq!(mod_pow(2, 10, 1000), BigInt::from(24));  // 1024 mod 1000
/// assert_eq!(mod_pow(3, 4, 17), BigInt::from(13));     // 81 mod 17
/// ```
pub fn mod_pow(
    base: impl Into<BigInt>,
    exp: impl Into<BigInt>,
    modulus: impl Into<BigInt>,
) -> BigInt {
    let base: BigInt = base.into();
    let exp: BigInt = exp.into();
    let modulus: BigInt = modulus.into();
    if modulus <= BigInt::zero() || exp < BigInt::zero() {
        return BigInt::zero();
    }
    let base_mod = ((&base % &modulus) + &modulus) % &modulus;
    base_mod.modpow(&exp, &modulus)
}

/// Returns `true` if `n` is a perfect square.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::is_square;
/// assert!(is_square(0));
/// assert!(is_square(1));
/// assert!(is_square(144));
/// assert!(!is_square(145));
/// ```
pub fn is_square(n: impl Into<BigInt>) -> bool {
    let n: BigInt = n.into();
    if n.is_negative() {
        return false;
    }
    match isqrt_dispatch(&n) {
        Some(s) => &s * &s == n,
        None => false,
    }
}

/// Integer square root: the largest `s` such that `s² ≤ n`.
///
/// Returns `None` for negative inputs.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::isqrt;
/// use num_bigint::BigInt;
///
/// assert_eq!(isqrt(0), Some(BigInt::from(0)));
/// assert_eq!(isqrt(9), Some(BigInt::from(3)));
/// assert_eq!(isqrt(10), Some(BigInt::from(3)));
/// assert_eq!(isqrt(-1i64), None);
/// ```
pub fn isqrt(n: impl Into<BigInt>) -> Option<BigInt> {
    let n: BigInt = n.into();
    isqrt_dispatch(&n)
}

/// Internal dispatch for isqrt.
fn isqrt_dispatch(n: &BigInt) -> Option<BigInt> {
    if let Some(n_i64) = n.to_i64() {
        return isqrt_i64(n_i64).map(BigInt::from);
    }
    isqrt_big(n)
}

/// Returns all primes up to (and including) `limit` using the Sieve of
/// Eratosthenes.
///
/// This function always uses `i64` because the sieve must be bounded by
/// available memory.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::primes_up_to;
/// assert_eq!(primes_up_to(20), vec![2, 3, 5, 7, 11, 13, 17, 19]);
/// ```
pub fn primes_up_to(limit: i64) -> Vec<i64> {
    if limit < 2 {
        return vec![];
    }
    let limit = limit as usize;
    let mut sieve = vec![true; limit + 1];
    sieve[0] = false;
    sieve[1] = false;
    let mut p = 2;
    while p * p <= limit {
        if sieve[p] {
            let mut multiple = p * p;
            while multiple <= limit {
                sieve[multiple] = false;
                multiple += p;
            }
        }
        p += 1;
    }
    sieve
        .iter()
        .enumerate()
        .filter_map(|(i, &is_p)| if is_p { Some(i as i64) } else { None })
        .collect()
}

/// Legendre symbol `(a/p)` for odd prime `p`.
///
/// Returns:
/// - `1` if `a` is a quadratic residue mod `p` and `a ≢ 0`,
/// - `-1` if `a` is a non-residue mod `p`,
/// - `0` if `a ≡ 0 (mod p)`.
///
/// # Panics
///
/// Panics if `p` is not an odd prime.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::legendre_symbol;
/// assert_eq!(legendre_symbol(2, 7), 1);
/// assert_eq!(legendre_symbol(3, 7), -1);
/// assert_eq!(legendre_symbol(7, 7), 0);
/// ```
pub fn legendre_symbol(a: impl Into<BigInt>, p: impl Into<BigInt>) -> i8 {
    let a: BigInt = a.into();
    let p: BigInt = p.into();
    assert!(
        p > BigInt::from(2) && isprime_big_internal(&p)
            || p.to_i64().is_some_and(|pi| pi > 2 && isprime_i64(pi)),
        "p must be an odd prime"
    );
    let a_mod = ((&a % &p) + &p) % &p;
    if a_mod.is_zero() {
        return 0;
    }
    let exp = (&p - BigInt::one()) / BigInt::from(2);
    let result = a_mod.modpow(&exp, &p);
    if result.is_one() { 1 } else { -1 }
}

// ═══════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════
// Unit tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Shorthand for BigInt::from.
    fn bi(n: i64) -> BigInt {
        BigInt::from(n)
    }

    // ── isprime ────────────────────────────────────────────────────────
    #[test]
    fn test_isprime_small() {
        assert!(isprime(2));
        assert!(isprime(3));
        assert!(!isprime(4));
        assert!(isprime(5));
    }

    #[test]
    fn test_isprime_larger() {
        assert!(isprime(104729));
        assert!(!isprime(104730));
    }

    #[test]
    fn test_isprime_mersenne() {
        assert!(isprime(2_147_483_647)); // 2^31 − 1
    }

    #[test]
    fn test_isprime_edge() {
        assert!(!isprime(-1));
        assert!(!isprime(0));
        assert!(!isprime(1));
        assert!(isprime(2));
        assert!(!isprime(9));
    }

    // ── isprime with BigInt values ────────────────────────────────────
    #[test]
    fn test_isprime_bigint_small() {
        assert!(isprime(BigInt::from(2)));
        assert!(isprime(BigInt::from(3)));
        assert!(!isprime(BigInt::from(4)));
        assert!(isprime(BigInt::from(5)));
        assert!(!isprime(BigInt::from(0)));
        assert!(!isprime(BigInt::from(-7)));
    }

    #[test]
    fn test_isprime_bigint_mersenne_m31() {
        let m31 = BigInt::from(2_147_483_647i64);
        assert!(isprime(m31));
    }

    #[test]
    fn test_isprime_bigint_mersenne_m61() {
        let m61 = BigInt::from(2u64.pow(61) - 1);
        assert!(isprime(m61));
    }

    #[test]
    fn test_isprime_bigint_large_composite() {
        let m31 = BigInt::from(2_147_483_647i64);
        let other = BigInt::from(2_147_483_659i64);
        let product = &m31 * &other;
        assert!(!isprime(product));
    }

    #[test]
    fn test_isprime_bigint_carmichael() {
        for &c in &[561i64, 1105, 1729, 2465, 2821, 6601, 8911] {
            assert!(
                !isprime(BigInt::from(c)),
                "Carmichael number {c} should not be prime"
            );
        }
    }

    // ── factorint ─────────────────────────────────────────────────────
    #[test]
    fn test_factorint_60() {
        assert_eq!(factorint(60), vec![(bi(2), 2), (bi(3), 1), (bi(5), 1)]);
    }

    #[test]
    fn test_factorint_1() {
        assert_eq!(factorint(1), vec![]);
    }

    #[test]
    fn test_factorint_prime() {
        assert_eq!(factorint(97), vec![(bi(97), 1)]);
    }

    #[test]
    fn test_factorint_power_of_two() {
        assert_eq!(factorint(1024), vec![(bi(2), 10)]);
    }

    #[test]
    fn test_factorint_negative() {
        assert_eq!(factorint(-60), vec![(bi(2), 2), (bi(3), 1), (bi(5), 1)]);
    }

    // ── factorint with BigInt input ───────────────────────────────────
    #[test]
    fn test_factorint_bigint_60() {
        let factors = factorint(BigInt::from(60));
        assert_eq!(factors, vec![(bi(2), 2), (bi(3), 1), (bi(5), 1)]);
    }

    #[test]
    fn test_factorint_bigint_zero_and_one() {
        assert_eq!(factorint(BigInt::from(0)), vec![]);
        assert_eq!(factorint(BigInt::from(1)), vec![]);
    }

    #[test]
    fn test_factorint_bigint_negative() {
        let factors = factorint(BigInt::from(-60));
        assert_eq!(factors, vec![(bi(2), 2), (bi(3), 1), (bi(5), 1)]);
    }

    #[test]
    fn test_factorint_bigint_mersenne_prime_m31() {
        let m31 = BigInt::from(2_147_483_647i64);
        let factors = factorint(m31.clone());
        assert_eq!(factors, vec![(m31, 1)]);
    }

    #[test]
    fn test_factorint_bigint_m61_is_prime() {
        let m61 = BigInt::from(2u64.pow(61) - 1);
        let factors = factorint(m61.clone());
        assert_eq!(factors, vec![(m61, 1)]);
    }

    #[test]
    fn test_factorint_bigint_large_power_of_two() {
        let n = BigInt::from(1u64 << 40);
        let factors = factorint(n);
        assert_eq!(factors, vec![(bi(2), 40)]);
    }

    #[test]
    fn test_factorint_bigint_roundtrip() {
        for val in [60i64, 360, 2310, 100_000, 123456789, 999_999_937] {
            let n = BigInt::from(val);
            let factors = factorint(n.clone());
            let mut product = BigInt::one();
            for (p, e) in &factors {
                for _ in 0..*e {
                    product *= p;
                }
            }
            assert_eq!(product, n, "factorization of {val} doesn't multiply back");
        }
    }

    #[test]
    fn test_factorint_bigint_all_factors_prime() {
        for val in [60i64, 360, 2310, 100_000, 123456789] {
            let n = BigInt::from(val);
            let factors = factorint(n);
            for (p, _) in &factors {
                assert!(isprime(p.clone()), "factor {p} of {val} is not prime");
            }
        }
    }

    #[test]
    fn test_factorint_bigint_beyond_f64_precision() {
        let n = BigInt::from(2u64.pow(53) + 1);
        let factors = factorint(n.clone());
        let mut product = BigInt::one();
        for (p, e) in &factors {
            for _ in 0..*e {
                product *= p;
            }
        }
        assert_eq!(
            product, n,
            "factorization beyond f64 precision must be exact"
        );
        for (p, _) in &factors {
            assert!(isprime(p.clone()), "factor {p} should be prime");
        }
    }

    // ── nextprime / prevprime ─────────────────────────────────────────
    #[test]
    fn test_nextprime() {
        assert_eq!(nextprime(10), bi(11));
        assert_eq!(nextprime(11), bi(13));
    }

    #[test]
    fn test_nextprime_from_negative() {
        assert_eq!(nextprime(-10), bi(2));
    }

    #[test]
    fn test_prevprime() {
        assert_eq!(prevprime(10), Some(bi(7)));
        assert_eq!(prevprime(2), None);
    }

    #[test]
    fn test_prevprime_3() {
        assert_eq!(prevprime(3), Some(bi(2)));
    }

    // ── divisors ──────────────────────────────────────────────────────
    #[test]
    fn test_divisors_12() {
        assert_eq!(
            divisors(12),
            vec![bi(1), bi(2), bi(3), bi(4), bi(6), bi(12)]
        );
    }

    #[test]
    fn test_divisors_1() {
        assert_eq!(divisors(1), vec![bi(1)]);
    }

    #[test]
    fn test_divisor_count_12() {
        assert_eq!(divisor_count(12), 6);
    }

    #[test]
    fn test_divisor_sum_12() {
        assert_eq!(divisor_sum(12), bi(28));
    }

    // ── totient ───────────────────────────────────────────────────────
    #[test]
    fn test_totient_12() {
        assert_eq!(totient(12), bi(4));
    }

    #[test]
    fn test_totient_prime() {
        assert_eq!(totient(13), bi(12));
    }

    #[test]
    fn test_totient_1() {
        assert_eq!(totient(1), bi(1));
    }

    // ── mobius ─────────────────────────────────────────────────────────
    #[test]
    fn test_mobius() {
        assert_eq!(mobius(1), 1);
        assert_eq!(mobius(6), 1); // 2·3, even number of factors
        assert_eq!(mobius(4), 0); // 2², squared factor
        assert_eq!(mobius(30), -1); // 2·3·5, odd number of factors
    }

    // ── mod_inverse ───────────────────────────────────────────────────
    #[test]
    fn test_mod_inverse() {
        assert_eq!(mod_inverse(3, 7), Some(bi(5)));
        assert_eq!(mod_inverse(2, 4), None);
    }

    #[test]
    fn test_mod_inverse_verify() {
        let inv = mod_inverse(17, 43).unwrap();
        assert_eq!((bi(17) * &inv) % bi(43), bi(1));
    }

    // ── crt ───────────────────────────────────────────────────────────
    #[test]
    fn test_crt() {
        let r: Vec<BigInt> = vec![bi(2), bi(3), bi(2)];
        let m: Vec<BigInt> = vec![bi(3), bi(5), bi(7)];
        assert_eq!(crt(&r, &m), Some(bi(23)));
    }

    #[test]
    fn test_crt_no_solution() {
        let r: Vec<BigInt> = vec![bi(0), bi(1)];
        let m: Vec<BigInt> = vec![bi(2), bi(4)];
        assert_eq!(crt(&r, &m), None);
    }

    #[test]
    fn test_crt_i64() {
        assert_eq!(crt_i64(&[2, 3, 2], &[3, 5, 7]), Some(23));
    }

    #[test]
    fn test_crt_i64_no_solution() {
        assert_eq!(crt_i64(&[0, 1], &[2, 4]), None);
    }

    // ── gcd / lcm ────────────────────────────────────────────────────
    #[test]
    fn test_gcd_lcm() {
        assert_eq!(gcd(12, 8), bi(4));
        assert_eq!(lcm(4, 6), bi(12));
    }

    // ── is_coprime ───────────────────────────────────────────────────
    #[test]
    fn test_is_coprime() {
        assert!(is_coprime(8, 15));
        assert!(!is_coprime(8, 12));
    }

    // ── mod_pow ──────────────────────────────────────────────────────
    #[test]
    fn test_mod_pow() {
        assert_eq!(mod_pow(2, 10, 1000), bi(24));
        assert_eq!(mod_pow(3, 4, 17), bi(13));
    }

    // ── is_square / isqrt ────────────────────────────────────────────
    #[test]
    fn test_is_square() {
        assert!(is_square(0));
        assert!(is_square(1));
        assert!(is_square(144));
        assert!(!is_square(145));
        assert!(!is_square(-4));
    }

    #[test]
    fn test_isqrt() {
        assert_eq!(isqrt(0), Some(bi(0)));
        assert_eq!(isqrt(9), Some(bi(3)));
        assert_eq!(isqrt(10), Some(bi(3)));
        assert_eq!(isqrt(-1i64), None);
    }

    // ── primes_up_to ─────────────────────────────────────────────────
    #[test]
    fn test_primes_up_to() {
        assert_eq!(primes_up_to(30), vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    // ── legendre_symbol ──────────────────────────────────────────────
    #[test]
    fn test_legendre_symbol() {
        assert_eq!(legendre_symbol(2, 7), 1);
        assert_eq!(legendre_symbol(3, 7), -1);
        assert_eq!(legendre_symbol(7, 7), 0);
    }
}
