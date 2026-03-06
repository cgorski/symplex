//! Number theory functions: primality testing, factorization, divisors,
//! modular arithmetic, and related operations.
//!
//! All functions operate on exact integers. The primary API uses `i64` for
//! speed on common small inputs. BigInt-suffixed variants (`isprime_bigint`,
//! `factorint_bigint`) handle arbitrary-precision integers without loss.

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};

// ═══════════════════════════════════════════════════════════════════════════
// Modular arithmetic helpers (internal, i64/u64 fast path)
// ═══════════════════════════════════════════════════════════════════════════

/// Modular multiplication using 128-bit intermediates to avoid overflow.
fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// Modular exponentiation: `base^exp mod modulus` via binary method.
fn mod_pow(base: u64, mut exp: u64, modulus: u64) -> u64 {
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

/// Miller-Rabin primality test with the given deterministic witnesses.
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
    if n % 2 == 0 {
        return false;
    }

    // Write n-1 as 2^r · d where d is odd
    let mut d = n - 1;
    let mut r = 0u32;
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }

    'outer: for &a in witnesses {
        if a >= n {
            continue;
        }
        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..r - 1 {
            x = mod_mul(x, x, n);
            if x == n - 1 {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

// ═══════════════════════════════════════════════════════════════════════════
// BigInt modular arithmetic helpers (internal)
// ═══════════════════════════════════════════════════════════════════════════

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
// Extended Euclidean algorithm (internal)
// ═══════════════════════════════════════════════════════════════════════════

/// Returns `(gcd, x, y)` such that `a*x + b*y = gcd`.
fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if a == 0 {
        return (b, 0, 1);
    }
    let (g, x1, y1) = extended_gcd(b % a, a);
    (g, y1 - (b / a) * x1, x1)
}

// ═══════════════════════════════════════════════════════════════════════════
// Public API — BigInt versions
// ═══════════════════════════════════════════════════════════════════════════

/// Deterministic primality test for arbitrary-precision integers.
///
/// Uses trial division for tiny factors, then a deterministic Miller-Rabin
/// test with witnesses that are correct for all `n < 3.3×10²⁴`. For larger
/// numbers, the test is probabilistic but with negligible error probability
/// (the 12-witness set makes false positives astronomically unlikely).
///
/// # Examples
///
/// ```
/// use symplex::ntheory::isprime_bigint;
/// use num_bigint::BigInt;
///
/// assert!(isprime_bigint(&BigInt::from(2)));
/// assert!(isprime_bigint(&BigInt::from(2_147_483_647i64))); // 2^31 - 1
/// assert!(!isprime_bigint(&BigInt::from(4)));
/// ```
pub fn isprime_bigint(n: &BigInt) -> bool {
    if *n < BigInt::from(2) {
        return false;
    }

    // If it fits in i64, use the fast path
    if let Some(ni) = n.to_i64() {
        return isprime(ni);
    }

    // For BigInt values that don't fit in i64, use BigInt path
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

/// Returns the prime factorization of a BigInt `n` as `(prime, exponent)` pairs,
/// sorted by prime in ascending order.
///
/// Returns an empty vector for `n ∈ {-1, 0, 1}`. Negative inputs are
/// factored by absolute value.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::factorint_bigint;
/// use num_bigint::BigInt;
///
/// let factors = factorint_bigint(&BigInt::from(60));
/// assert_eq!(
///     factors,
///     vec![(BigInt::from(2), 2), (BigInt::from(3), 1), (BigInt::from(5), 1)]
/// );
/// ```
pub fn factorint_bigint(n: &BigInt) -> Vec<(BigInt, u32)> {
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
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
        89, 97,
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
    // which uses u64/u128 arithmetic instead of BigInt divisions.
    // This is the common case: most inputs or cofactors after small-prime
    // extraction are well within i64 range.
    if n > BigInt::one() {
        if let Some(ni) = n.to_i64() {
            for (p, e) in factorint(ni) {
                factors.push((BigInt::from(p), e));
            }
            return factors;
        }
    }

    // For cofactors that exceed i64, check if already prime to avoid
    // O(sqrt(n)) BigInt divisions (e.g. Mersenne prime M61 ≈ 2.3×10^18).
    if n > BigInt::one() && isprime_bigint(&n) {
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
            if n > BigInt::one() && isprime_bigint(&n) {
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

// ═══════════════════════════════════════════════════════════════════════════
// Public API — i64 versions (fast path)
// ═══════════════════════════════════════════════════════════════════════════

/// Deterministic primality test for any `i64`.
///
/// Uses trial division for tiny factors, then a deterministic Miller-Rabin
/// test with witnesses that are correct for all `n < 3.3×10²⁴`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::isprime;
/// assert!(isprime(2));
/// assert!(isprime(104729));
/// assert!(!isprime(104730));
/// ```
pub fn isprime(n: i64) -> bool {
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

/// Returns the prime factorization of `n` as `(prime, exponent)` pairs,
/// sorted by prime in ascending order.
///
/// Returns an empty vector for `n ∈ {-1, 0, 1}`. Negative inputs are
/// factored by absolute value.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::factorint;
/// assert_eq!(factorint(60), vec![(2, 2), (3, 1), (5, 1)]);
/// assert_eq!(factorint(1), vec![]);
/// assert_eq!(factorint(-12), vec![(2, 2), (3, 1)]);
/// ```
pub fn factorint(n: i64) -> Vec<(i64, u32)> {
    if n == 0 {
        return vec![];
    }
    let mut n = n.unsigned_abs();
    let mut factors = Vec::new();

    // Trial division with a table of small primes
    const SMALL_PRIMES: [u64; 25] = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
        89, 97,
    ];
    for &p in &SMALL_PRIMES {
        if p * p > n && n > 1 {
            break;
        }
        let mut count = 0u32;
        while n % p == 0 {
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
        while n % d == 0 {
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

/// Returns the smallest prime strictly greater than `n`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::nextprime;
/// assert_eq!(nextprime(10), 11);
/// assert_eq!(nextprime(11), 13);
/// assert_eq!(nextprime(-5), 2);
/// ```
pub fn nextprime(n: i64) -> i64 {
    if n < 2 {
        return 2;
    }
    let mut candidate = if n % 2 == 0 { n + 1 } else { n + 2 };
    while !isprime(candidate) {
        candidate += 2;
    }
    candidate
}

/// Returns the largest prime strictly less than `n`, or `None` if no
/// such prime exists (i.e. `n ≤ 2`).
///
/// # Examples
///
/// ```
/// use symplex::ntheory::prevprime;
/// assert_eq!(prevprime(10), Some(7));
/// assert_eq!(prevprime(3), Some(2));
/// assert_eq!(prevprime(2), None);
/// ```
pub fn prevprime(n: i64) -> Option<i64> {
    if n <= 2 {
        return None;
    }
    if n == 3 {
        return Some(2);
    }
    let mut candidate = if n % 2 == 0 { n - 1 } else { n - 2 };
    while candidate >= 2 && !isprime(candidate) {
        candidate -= 2;
    }
    if candidate >= 2 {
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
/// assert_eq!(divisors(12), vec![1, 2, 3, 4, 6, 12]);
/// assert_eq!(divisors(-12), vec![1, 2, 3, 4, 6, 12]);
/// ```
pub fn divisors(n: i64) -> Vec<i64> {
    if n == 0 {
        return vec![];
    }
    let n_abs = n.unsigned_abs() as i64;
    let factors = factorint(n_abs);
    let mut divs = vec![1i64];
    for (p, e) in factors {
        let current_len = divs.len();
        let mut power = 1i64;
        for _ in 0..e {
            power *= p;
            for j in 0..current_len {
                divs.push(divs[j] * power);
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
pub fn divisor_count(n: i64) -> usize {
    divisors(n).len()
}

/// Returns the sum of all positive divisors of `|n|`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::divisor_sum;
/// assert_eq!(divisor_sum(12), 28);
/// assert_eq!(divisor_sum(1), 1);
/// ```
pub fn divisor_sum(n: i64) -> i64 {
    divisors(n).iter().sum()
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
/// assert_eq!(totient(12), 4);
/// assert_eq!(totient(13), 12);  // prime ⟹ φ(p) = p − 1
/// ```
pub fn totient(n: i64) -> i64 {
    if n <= 0 {
        return 0;
    }
    let factors = factorint(n);
    let mut result = n;
    for (p, _) in factors {
        result = result / p * (p - 1);
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
pub fn mobius(n: i64) -> i8 {
    if n <= 0 {
        return 0;
    }
    let factors = factorint(n);
    for (_, e) in &factors {
        if *e > 1 {
            return 0;
        }
    }
    if factors.len() % 2 == 0 {
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
/// assert_eq!(mod_inverse(3, 7), Some(5));  // 3·5 = 15 ≡ 1 (mod 7)
/// assert_eq!(mod_inverse(2, 4), None);     // gcd(2,4) = 2 ≠ 1
/// ```
pub fn mod_inverse(a: i64, n: i64) -> Option<i64> {
    let (g, x, _) = extended_gcd(a, n);
    if g != 1 {
        return None;
    }
    Some(((x % n) + n) % n)
}

/// Chinese Remainder Theorem.
///
/// Given a system of congruences `x ≡ remainders[i] (mod moduli[i])`,
/// returns the unique solution modulo the LCM of the moduli, or `None`
/// if no solution exists or the inputs are invalid.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::crt;
/// // x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)  →  x = 23
/// assert_eq!(crt(&[2, 3, 2], &[3, 5, 7]), Some(23));
/// ```
pub fn crt(remainders: &[i64], moduli: &[i64]) -> Option<i64> {
    if remainders.len() != moduli.len() || remainders.is_empty() {
        return None;
    }
    let mut result = remainders[0];
    let mut modulus = moduli[0];
    for i in 1..remainders.len() {
        let (g, p, _) = extended_gcd(modulus, moduli[i]);
        if (remainders[i] - result) % g != 0 {
            return None;
        }
        result =
            result + modulus * ((remainders[i] - result) / g % (moduli[i] / g)) * p;
        modulus = modulus / g * moduli[i];
        result = ((result % modulus) + modulus) % modulus;
    }
    Some(result)
}

/// Greatest common divisor of two integers.
///
/// Wrapper around `num_integer::gcd`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::gcd_int;
/// assert_eq!(gcd_int(12, 8), 4);
/// assert_eq!(gcd_int(0, 5), 5);
/// ```
pub fn gcd_int(a: i64, b: i64) -> i64 {
    num_integer::gcd(a, b)
}

/// Least common multiple of two integers.
///
/// Wrapper around `num_integer::lcm`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::lcm_int;
/// assert_eq!(lcm_int(4, 6), 12);
/// assert_eq!(lcm_int(0, 5), 0);
/// ```
pub fn lcm_int(a: i64, b: i64) -> i64 {
    num_integer::lcm(a, b)
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
pub fn is_coprime(a: i64, b: i64) -> bool {
    gcd_int(a, b) == 1
}

/// Modular exponentiation: `base^exp mod modulus`.
///
/// Returns 0 if `modulus` is 0 or 1. Works for non-negative exponents.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::mod_pow_int;
/// assert_eq!(mod_pow_int(2, 10, 1000), 24);   // 1024 mod 1000
/// assert_eq!(mod_pow_int(3, 4, 17), 13);       // 81 mod 17
/// ```
pub fn mod_pow_int(base: i64, exp: i64, modulus: i64) -> i64 {
    if modulus <= 0 || exp < 0 {
        return 0;
    }
    mod_pow(
        ((base % modulus) + modulus) as u64 % modulus as u64,
        exp as u64,
        modulus as u64,
    ) as i64
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
pub fn is_square(n: i64) -> bool {
    if n < 0 {
        return false;
    }
    let s = (n as f64).sqrt() as i64;
    // Check s-1, s, s+1 to guard against floating-point imprecision
    for candidate in [s - 1, s, s + 1] {
        if candidate >= 0 && candidate * candidate == n {
            return true;
        }
    }
    false
}

/// Integer square root: the largest `s` such that `s² ≤ n`.
///
/// Returns `None` for negative inputs.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::isqrt;
/// assert_eq!(isqrt(0), Some(0));
/// assert_eq!(isqrt(9), Some(3));
/// assert_eq!(isqrt(10), Some(3));
/// assert_eq!(isqrt(-1), None);
/// ```
pub fn isqrt(n: i64) -> Option<i64> {
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

/// Returns all primes up to (and including) `limit` using the Sieve of
/// Eratosthenes.
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
        .filter_map(|(i, &is_prime)| if is_prime { Some(i as i64) } else { None })
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
pub fn legendre_symbol(a: i64, p: i64) -> i8 {
    assert!(p > 2 && isprime(p), "p must be an odd prime");
    let a_mod = ((a % p) + p) % p;
    if a_mod == 0 {
        return 0;
    }
    let result = mod_pow(a_mod as u64, ((p - 1) / 2) as u64, p as u64);
    if result == 1 {
        1
    } else {
        -1
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

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

    // ── isprime_bigint ────────────────────────────────────────────────
    #[test]
    fn test_isprime_bigint_small() {
        assert!(isprime_bigint(&BigInt::from(2)));
        assert!(isprime_bigint(&BigInt::from(3)));
        assert!(!isprime_bigint(&BigInt::from(4)));
        assert!(isprime_bigint(&BigInt::from(5)));
        assert!(!isprime_bigint(&BigInt::from(0)));
        assert!(!isprime_bigint(&BigInt::from(-7)));
    }

    #[test]
    fn test_isprime_bigint_mersenne_m31() {
        // 2^31 - 1 = 2147483647 is a Mersenne prime
        let m31 = BigInt::from(2_147_483_647i64);
        assert!(isprime_bigint(&m31));
    }

    #[test]
    fn test_isprime_bigint_mersenne_m61() {
        // 2^61 - 1 = 2305843009213693951 is a Mersenne prime
        let m61 = BigInt::from(2u64.pow(61) - 1);
        assert!(isprime_bigint(&m61));
    }

    #[test]
    fn test_isprime_bigint_large_composite() {
        // (2^31 - 1) * (2^31 + 11) — composite, both factors beyond i32
        let m31 = BigInt::from(2_147_483_647i64);
        let other = BigInt::from(2_147_483_659i64); // 2^31 + 11, which is prime
        let product = &m31 * &other;
        assert!(!isprime_bigint(&product));
    }

    #[test]
    fn test_isprime_bigint_carmichael() {
        // Carmichael numbers should not fool our test
        for &c in &[561i64, 1105, 1729, 2465, 2821, 6601, 8911] {
            assert!(
                !isprime_bigint(&BigInt::from(c)),
                "Carmichael number {c} should not be prime"
            );
        }
    }

    // ── factorint ─────────────────────────────────────────────────────
    #[test]
    fn test_factorint_60() {
        assert_eq!(factorint(60), vec![(2, 2), (3, 1), (5, 1)]);
    }

    #[test]
    fn test_factorint_1() {
        assert_eq!(factorint(1), vec![]);
    }

    #[test]
    fn test_factorint_prime() {
        assert_eq!(factorint(97), vec![(97, 1)]);
    }

    #[test]
    fn test_factorint_power_of_two() {
        assert_eq!(factorint(1024), vec![(2, 10)]);
    }

    #[test]
    fn test_factorint_negative() {
        assert_eq!(factorint(-60), vec![(2, 2), (3, 1), (5, 1)]);
    }

    // ── factorint_bigint ──────────────────────────────────────────────
    #[test]
    fn test_factorint_bigint_60() {
        let factors = factorint_bigint(&BigInt::from(60));
        assert_eq!(
            factors,
            vec![
                (BigInt::from(2), 2),
                (BigInt::from(3), 1),
                (BigInt::from(5), 1),
            ]
        );
    }

    #[test]
    fn test_factorint_bigint_zero_and_one() {
        assert_eq!(factorint_bigint(&BigInt::from(0)), vec![]);
        assert_eq!(factorint_bigint(&BigInt::from(1)), vec![]);
    }

    #[test]
    fn test_factorint_bigint_negative() {
        let factors = factorint_bigint(&BigInt::from(-60));
        assert_eq!(
            factors,
            vec![
                (BigInt::from(2), 2),
                (BigInt::from(3), 1),
                (BigInt::from(5), 1),
            ]
        );
    }

    #[test]
    fn test_factorint_bigint_mersenne_prime_m31() {
        // 2^31 - 1 = 2147483647 is prime, so factorization should be itself
        let m31 = BigInt::from(2_147_483_647i64);
        let factors = factorint_bigint(&m31);
        assert_eq!(factors, vec![(m31, 1)]);
    }

    #[test]
    fn test_factorint_bigint_m61_is_prime() {
        // 2^61 - 1 = 2305843009213693951 is a Mersenne prime
        let m61 = BigInt::from(2u64.pow(61) - 1);
        let factors = factorint_bigint(&m61);
        assert_eq!(factors, vec![(m61, 1)]);
    }

    #[test]
    fn test_factorint_bigint_large_power_of_two() {
        // 2^40 = 1099511627776
        let n = BigInt::from(1u64 << 40);
        let factors = factorint_bigint(&n);
        assert_eq!(factors, vec![(BigInt::from(2), 40)]);
    }

    #[test]
    fn test_factorint_bigint_roundtrip() {
        // Verify factorization multiplies back to the original
        for val in [60i64, 360, 2310, 100_000, 123456789, 999_999_937] {
            let n = BigInt::from(val);
            let factors = factorint_bigint(&n);
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
            let factors = factorint_bigint(&n);
            for (p, _) in &factors {
                assert!(
                    isprime_bigint(p),
                    "factor {p} of {val} is not prime"
                );
            }
        }
    }

    #[test]
    fn test_factorint_bigint_beyond_f64_precision() {
        // 2^53 + 1 is beyond f64 exact integer range — verifies no f64 truncation.
        // 2^53 + 1 = 9007199254740993 = 3 × 2336171 × 1285207
        // Let's just verify the roundtrip is exact.
        let n = BigInt::from(2u64.pow(53) + 1);
        let factors = factorint_bigint(&n);
        let mut product = BigInt::one();
        for (p, e) in &factors {
            for _ in 0..*e {
                product *= p;
            }
        }
        assert_eq!(product, n, "factorization beyond f64 precision must be exact");
        // All factors should be prime
        for (p, _) in &factors {
            assert!(isprime_bigint(p), "factor {p} should be prime");
        }
    }

    // ── nextprime / prevprime ─────────────────────────────────────────
    #[test]
    fn test_nextprime() {
        assert_eq!(nextprime(10), 11);
        assert_eq!(nextprime(11), 13);
    }

    #[test]
    fn test_nextprime_from_negative() {
        assert_eq!(nextprime(-10), 2);
    }

    #[test]
    fn test_prevprime() {
        assert_eq!(prevprime(10), Some(7));
        assert_eq!(prevprime(2), None);
    }

    #[test]
    fn test_prevprime_3() {
        assert_eq!(prevprime(3), Some(2));
    }

    // ── divisors ──────────────────────────────────────────────────────
    #[test]
    fn test_divisors_12() {
        assert_eq!(divisors(12), vec![1, 2, 3, 4, 6, 12]);
    }

    #[test]
    fn test_divisors_1() {
        assert_eq!(divisors(1), vec![1]);
    }

    #[test]
    fn test_divisor_count_12() {
        assert_eq!(divisor_count(12), 6);
    }

    #[test]
    fn test_divisor_sum_12() {
        assert_eq!(divisor_sum(12), 28);
    }

    // ── totient ───────────────────────────────────────────────────────
    #[test]
    fn test_totient_12() {
        assert_eq!(totient(12), 4);
    }

    #[test]
    fn test_totient_prime() {
        assert_eq!(totient(13), 12);
    }

    #[test]
    fn test_totient_1() {
        assert_eq!(totient(1), 1);
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
        assert_eq!(mod_inverse(3, 7), Some(5));
        assert_eq!(mod_inverse(2, 4), None);
    }

    #[test]
    fn test_mod_inverse_verify() {
        let inv = mod_inverse(17, 43).unwrap();
        assert_eq!((17 * inv) % 43, 1);
    }

    // ── crt ───────────────────────────────────────────────────────────
    #[test]
    fn test_crt() {
        assert_eq!(crt(&[2, 3, 2], &[3, 5, 7]), Some(23));
    }

    #[test]
    fn test_crt_no_solution() {
        // x ≡ 0 (mod 2) and x ≡ 1 (mod 4) has no solution
        assert_eq!(crt(&[0, 1], &[2, 4]), None);
    }

    // ── gcd / lcm ────────────────────────────────────────────────────
    #[test]
    fn test_gcd_lcm() {
        assert_eq!(gcd_int(12, 8), 4);
        assert_eq!(lcm_int(4, 6), 12);
    }

    // ── is_coprime ───────────────────────────────────────────────────
    #[test]
    fn test_is_coprime() {
        assert!(is_coprime(8, 15));
        assert!(!is_coprime(8, 12));
    }

    // ── mod_pow_int ──────────────────────────────────────────────────
    #[test]
    fn test_mod_pow_int() {
        assert_eq!(mod_pow_int(2, 10, 1000), 24);
        assert_eq!(mod_pow_int(3, 4, 17), 13);
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
        assert_eq!(isqrt(0), Some(0));
        assert_eq!(isqrt(9), Some(3));
        assert_eq!(isqrt(10), Some(3));
        assert_eq!(isqrt(-1), None);
    }

    // ── primes_up_to ─────────────────────────────────────────────────
    #[test]
    fn test_primes_up_to() {
        assert_eq!(
            primes_up_to(30),
            vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        );
    }

    // ── legendre_symbol ──────────────────────────────────────────────
    #[test]
    fn test_legendre_symbol() {
        assert_eq!(legendre_symbol(2, 7), 1);
        assert_eq!(legendre_symbol(3, 7), -1);
        assert_eq!(legendre_symbol(7, 7), 0);
    }
}
