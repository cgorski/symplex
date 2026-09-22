//! Number theory: primality testing, integer factorization, divisors,
//! modular arithmetic, quadratic residues, discrete logarithms, prime
//! counting, continued fractions and classical integer sequences.
//!
//! **Unified API** — every function accepts arbitrary-precision integers via
//! `impl Into<BigInt>`.  Small values that fit in `i64`/`u64` automatically
//! take an optimised machine-word fast path; larger values use `BigInt`
//! arithmetic.
//!
//! # Primality
//!
//! [`isprime`] is **deterministic** for `n < 3.3·10²⁴` (Miller–Rabin with
//! the first 13 prime bases) and uses the Baillie–PSW test (strong Fermat
//! base 2 + strong Lucas) beyond that.  BPSW has **no known
//! counterexamples** and none exist below 2⁶⁴ (exhaustively verified), but
//! it is not proven deterministic for arbitrary `n`.  [`is_probable_prime`]
//! exposes a plain randomized Miller–Rabin test with a caller-chosen round
//! count.
//!
//! # Factorization
//!
//! [`factorint`] combines trial division by all primes below 2¹⁶, a
//! perfect-power check, Pollard–Brent rho (machine words for `n < 2⁶⁴`),
//! and Lenstra's elliptic-curve method (Montgomery curves, stage 1) for
//! larger composites.  Every factor returned is prime (verified with
//! [`isprime`]); the algorithm never returns a composite as a "prime" —
//! for pathological inputs it may simply take a long time.
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
//!
//! // 2^64 + 1 = 274177 × 67280421310721
//! let f = factorint(BigInt::from(2u128.pow(64) + 1));
//! assert_eq!(f[0].0, BigInt::from(274177u64));
//! assert_eq!(f[1].0, BigInt::from(67280421310721u64));
//! ```

use std::sync::OnceLock;

use num_bigint::BigInt;
use num_integer::{ExtendedGcd, Integer, Roots};
use num_rational::Ratio;
use num_traits::{One, Signed, ToPrimitive, Zero};
use rustc_hash::FxHashMap;

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::errors::SymplexError;

// Combinatorial sequences live in `combinatorics`; re-export the classical
// ones here so that `ntheory::bell(5)` etc. work as users expect.
pub use crate::domains::combinatorics::{
    PartitionIter, bell, binomial, catalan, derangements, npartitions, partitions,
};

// ═══════════════════════════════════════════════════════════════════════════
// Internal modular arithmetic helpers (u64 fast path)
// ═══════════════════════════════════════════════════════════════════════════

/// Modular multiplication using 128-bit intermediates to avoid overflow.
#[inline]
fn mod_mul_u64(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// Modular addition without overflow.
#[inline]
fn mod_add_u64(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 + b as u128) % m as u128) as u64
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

/// Binary GCD on `u64`.
fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

use crate::base::numeric::Q;
/// The deterministic generator of the randomised algorithms here
/// (Miller–Rabin witnesses, Pollard–Brent), so that results are
/// reproducible run-to-run.  Root finding modulo `p` draws from the same
/// generator inside `PolyIn::roots`.
use crate::base::rng::XorShift64Star as XorShift;
use crate::poly::modpoly::PolyIn;

// ═══════════════════════════════════════════════════════════════════════════
// Small prime table
// ═══════════════════════════════════════════════════════════════════════════

/// Trial-division bound used by [`factorint`].
const TRIAL_DIVISION_LIMIT: u32 = 1 << 16;

/// Primes below [`TRIAL_DIVISION_LIMIT`] (6542 of them), computed once.
fn small_primes() -> &'static [u32] {
    static TABLE: OnceLock<Vec<u32>> = OnceLock::new();
    TABLE.get_or_init(|| sieve_u32(TRIAL_DIVISION_LIMIT))
}

/// Sieve of Eratosthenes returning all primes `< limit`.
fn sieve_u32(limit: u32) -> Vec<u32> {
    if limit < 3 {
        return if limit > 2 { vec![2] } else { vec![] };
    }
    let n = limit as usize;
    // Odd-only sieve: index i represents 2i + 1.
    let half = n / 2;
    let mut composite = vec![false; half];
    let mut primes = vec![2u32];
    let mut i = 1usize;
    while (2 * i + 1) * (2 * i + 1) < n {
        if !composite[i] {
            let p = 2 * i + 1;
            let mut j = (p * p) / 2;
            while j < half {
                composite[j] = true;
                j += p;
            }
        }
        i += 1;
    }
    for (i, &c) in composite.iter().enumerate().skip(1) {
        if !c {
            primes.push((2 * i + 1) as u32);
        }
    }
    primes
}

/// Odd-only bit-packed sieve of `[0, limit]`; returns `is_prime(k)` as a
/// closure-friendly bitset.
struct BitSieve {
    limit: u64,
    bits: Vec<u64>, // bit i set ⇒ 2i+1 is composite
}

impl BitSieve {
    fn new(limit: u64) -> Self {
        let half = (limit / 2 + 1) as usize;
        let mut bits = vec![0u64; half.div_ceil(64)];
        let mut i = 1u64;
        while (2 * i + 1) * (2 * i + 1) <= limit {
            if bits[(i / 64) as usize] & (1 << (i % 64)) == 0 {
                let p = 2 * i + 1;
                let mut j = (p * p) / 2;
                while j < half as u64 {
                    bits[(j / 64) as usize] |= 1 << (j % 64);
                    j += p;
                }
            }
            i += 1;
        }
        BitSieve { limit, bits }
    }

    fn is_prime(&self, k: u64) -> bool {
        if k < 2 || k > self.limit {
            return false;
        }
        if k == 2 {
            return true;
        }
        if k.is_multiple_of(2) {
            return false;
        }
        let i = k / 2;
        self.bits[(i / 64) as usize] & (1 << (i % 64)) == 0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal Miller-Rabin helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Deterministic Miller–Rabin witnesses: the first 13 primes suffice for
/// all `n < 3.3 × 10²⁴` (Sorenson–Webster 2015).
const DETERMINISTIC_WITNESSES: [u64; 13] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];

/// Miller-Rabin primality test with the given witnesses (u64).
///
/// With [`DETERMINISTIC_WITNESSES`] the result is exact for every `u64`.
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
        let a = a % n;
        if a == 0 {
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

/// Strong Fermat probable-prime test to the given base (BigInt).
fn strong_fermat_big(n: &BigInt, base: &BigInt) -> bool {
    let one = BigInt::one();
    let two = BigInt::from(2);
    let n_minus_1 = n - &one;
    let mut d = n_minus_1.clone();
    let mut r = 0u32;
    while d.is_even() {
        d /= &two;
        r += 1;
    }
    let a = base.mod_floor(n);
    if a.is_zero() {
        return true;
    }
    let mut x = a.modpow(&d, n);
    if x.is_one() || x == n_minus_1 {
        return true;
    }
    for _ in 1..r {
        x = (&x * &x) % n;
        if x == n_minus_1 {
            return true;
        }
    }
    false
}

/// Miller-Rabin primality test on BigInt with fixed witnesses.
fn miller_rabin_big(n: &BigInt, witnesses: &[u64]) -> bool {
    for &a in witnesses {
        let a_big = BigInt::from(a);
        if &a_big >= n {
            continue;
        }
        if !strong_fermat_big(n, &a_big) {
            return false;
        }
    }
    true
}

/// Strong Lucas probable-prime test with Selfridge's parameters
/// (method A): `D` is the first of `5, −7, 9, −11, …` with
/// `jacobi(D, n) = −1`, `P = 1`, `Q = (1 − D)/4`.
///
/// `n` must be odd, `> 2`, and not a perfect square.
fn strong_lucas_big(n: &BigInt) -> bool {
    // Choose D.
    let mut d_i: i64 = 5;
    let d = loop {
        let d_big = BigInt::from(d_i);
        match jacobi_big(&d_big, n) {
            -1 => break d_big,
            0 => {
                // gcd(D, n) > 1: composite unless n == |D|.
                return n.abs() == d_big.abs();
            }
            _ => {}
        }
        d_i = if d_i > 0 { -(d_i + 2) } else { -(d_i - 2) };
        if d_i.abs() > 1_000_000 {
            // Only possible for perfect squares, which callers exclude.
            return false;
        }
    };
    let p = BigInt::one();
    let q = (BigInt::one() - &d) / BigInt::from(4);
    let q = q.mod_floor(n);

    // n + 1 = k · 2^s with k odd.
    let n_plus_1 = n + BigInt::one();
    let mut k = n_plus_1.clone();
    let mut s = 0u32;
    while k.is_even() {
        k /= 2;
        s += 1;
    }

    // Compute U_k, V_k, Q^k mod n by the binary method.
    let half = |x: BigInt| -> BigInt {
        // x / 2 mod n for odd n
        let x = x.mod_floor(n);
        if x.is_even() { x / 2 } else { (x + n) / 2 }
    };
    let mut u = BigInt::one();
    let mut v = p.clone();
    let mut qk = q.clone();
    let bits = k.bits();
    for i in (0..bits - 1).rev() {
        // Double: U_{2m} = U_m V_m ; V_{2m} = V_m² − 2Q^m ; Q^{2m} = (Q^m)²
        u = (&u * &v).mod_floor(n);
        v = (&v * &v - BigInt::from(2) * &qk).mod_floor(n);
        qk = (&qk * &qk).mod_floor(n);
        if k.bit(i) {
            // Add one: U_{2m+1} = (P U + V)/2 ; V_{2m+1} = (D U + P V)/2
            let u_new = half(&p * &u + &v);
            let v_new = half(&d * &u + &p * &v);
            u = u_new;
            v = v_new;
            qk = (&qk * &q).mod_floor(n);
        }
    }

    if u.is_zero() || v.is_zero() {
        return true;
    }
    for _ in 1..s {
        v = (&v * &v - BigInt::from(2) * &qk).mod_floor(n);
        qk = (&qk * &qk).mod_floor(n);
        if v.is_zero() {
            return true;
        }
    }
    false
}

/// Baillie–PSW: strong Fermat base 2 + strong Lucas.  `n` odd, `> 2`.
fn bpsw_big(n: &BigInt) -> bool {
    if !strong_fermat_big(n, &BigInt::from(2)) {
        return false;
    }
    if is_square_big(n) {
        return false;
    }
    strong_lucas_big(n)
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal extended Euclidean algorithm
// ═══════════════════════════════════════════════════════════════════════════

/// Returns `(gcd, x, y)` such that `a*x + b*y = gcd` (BigInt, iterative).
fn extended_gcd_big(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
    let (mut old_r, mut r) = (a.clone(), b.clone());
    let (mut old_s, mut s) = (BigInt::one(), BigInt::zero());
    let (mut old_t, mut t) = (BigInt::zero(), BigInt::one());
    while !r.is_zero() {
        let q = &old_r / &r;
        let new_r = &old_r - &q * &r;
        old_r = std::mem::replace(&mut r, new_r);
        let new_s = &old_s - &q * &s;
        old_s = std::mem::replace(&mut s, new_s);
        let new_t = &old_t - &q * &t;
        old_t = std::mem::replace(&mut t, new_t);
    }
    if old_r.is_negative() {
        (-old_r, -old_s, -old_t)
    } else {
        (old_r, old_s, old_t)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal i64 / u64 fast-path implementations
// ═══════════════════════════════════════════════════════════════════════════

/// Deterministic primality test (u64).
fn isprime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for &p in &[2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] {
        if n == p {
            return true;
        }
        if n.is_multiple_of(p) {
            return false;
        }
    }
    miller_rabin(n, &DETERMINISTIC_WITNESSES)
}

/// Deterministic primality test (i64 fast path).
fn isprime_i64(n: i64) -> bool {
    if n < 2 {
        return false;
    }
    isprime_u64(n as u64)
}

/// Primality test for BigInt values that do NOT fit in u64.
fn isprime_big_internal(n: &BigInt) -> bool {
    if *n < BigInt::from(2) {
        return false;
    }
    if let Some(u) = n.to_u64() {
        return isprime_u64(u);
    }
    if n.is_even() {
        return false;
    }
    for &p in small_primes().iter().take(200) {
        if (n % p).is_zero() {
            return false;
        }
    }
    // Deterministic bound for the 13-witness Miller–Rabin test.
    static BOUND: OnceLock<BigInt> = OnceLock::new();
    let bound = BOUND.get_or_init(|| BigInt::from(33u64) * BigInt::from(10u64).pow(23));
    if n < bound {
        return miller_rabin_big(n, &DETERMINISTIC_WITNESSES);
    }
    bpsw_big(n)
}

/// Pollard–Brent rho on a `u64` composite.  Returns a non-trivial factor,
/// or `None` if this seed failed (caller retries with another seed).
fn pollard_brent_u64(n: u64, seed: u64) -> Option<u64> {
    if n.is_multiple_of(2) {
        return Some(2);
    }
    let c = 1 + seed % (n - 1);
    let f = |x: u64| mod_add_u64(mod_mul_u64(x, x, n), c, n);
    let mut y = (seed.wrapping_mul(6364136223846793005).wrapping_add(1)) % n;
    let mut x = y;
    let mut ys = y;
    let mut g = 1u64;
    let mut q = 1u64;
    let mut r = 1u64;
    let m = 128u64;
    while g == 1 {
        x = y;
        for _ in 0..r {
            y = f(y);
        }
        let mut k = 0u64;
        while k < r && g == 1 {
            ys = y;
            let lim = m.min(r - k);
            for _ in 0..lim {
                y = f(y);
                q = mod_mul_u64(q, x.abs_diff(y), n);
            }
            g = gcd_u64(q, n);
            k += lim;
        }
        r *= 2;
        if r > (1u64 << 40) {
            return None;
        }
    }
    if g == n {
        loop {
            ys = f(ys);
            g = gcd_u64(x.abs_diff(ys), n);
            if g > 1 {
                break;
            }
        }
    }
    if g == n { None } else { Some(g) }
}

/// Factor a `u64` with no prime factor below [`TRIAL_DIVISION_LIMIT`].
fn factor_u64_no_small(n: u64, out: &mut Vec<u64>) {
    let mut stack = vec![n];
    while let Some(m) = stack.pop() {
        if m == 1 {
            continue;
        }
        if isprime_u64(m) {
            out.push(m);
            continue;
        }
        // Perfect square shortcut (rho struggles with squares).
        let s = m.sqrt();
        if s * s == m {
            stack.push(s);
            stack.push(s);
            continue;
        }
        let mut seed = 1u64;
        let d = loop {
            if let Some(d) = pollard_brent_u64(m, seed) {
                break d;
            }
            seed += 1;
        };
        stack.push(d);
        stack.push(m / d);
    }
}

/// Factorize an i64 into `(prime, exponent)` pairs (i64 fast path).
fn factorint_i64(n: i64) -> Vec<(i64, u32)> {
    if n == 0 {
        return vec![];
    }
    let mut n = n.unsigned_abs();
    let mut factors: Vec<(i64, u32)> = Vec::new();

    for &p in small_primes() {
        let p = p as u64;
        if p * p > n {
            break;
        }
        if n.is_multiple_of(p) {
            let mut count = 0u32;
            while n.is_multiple_of(p) {
                n /= p;
                count += 1;
            }
            factors.push((p as i64, count));
        }
    }
    if n > 1 {
        if n < (TRIAL_DIVISION_LIMIT as u64) * (TRIAL_DIVISION_LIMIT as u64) {
            // Any remaining cofactor below 2^32 is prime.
            factors.push((n as i64, 1));
        } else {
            let mut rest = Vec::new();
            factor_u64_no_small(n, &mut rest);
            rest.sort_unstable();
            for p in rest {
                match factors.last_mut() {
                    Some((q, e)) if *q as u64 == p => *e += 1,
                    _ => factors.push((p as i64, 1)),
                }
            }
        }
    }
    factors.sort_by_key(|(p, _)| *p);
    factors
}

// ═══════════════════════════════════════════════════════════════════════════
// Modular rings for rho / ECM: Montgomery u128 fast path + BigInt fallback
// ═══════════════════════════════════════════════════════════════════════════

/// Arithmetic modulo a fixed odd `n`, abstracted so that Pollard rho and
/// ECM can run on machine words when `n < 2¹²⁷` and on `BigInt` otherwise.
///
/// Elements may be stored in any internal representation (e.g. Montgomery
/// form); only `gcd_with_n` needs to map back, and it is insensitive to
/// multiplication by units.
trait ModRing {
    /// Element representation.
    type El: Clone + PartialEq;
    fn embed(&self, x: &BigInt) -> Self::El;
    fn add(&self, a: &Self::El, b: &Self::El) -> Self::El;
    fn sub(&self, a: &Self::El, b: &Self::El) -> Self::El;
    fn mul(&self, a: &Self::El, b: &Self::El) -> Self::El;
    fn is_zero(&self, a: &Self::El) -> bool;
    /// `gcd(representative(a), n)` — the same for every representation
    /// that differs from the canonical one by a unit factor.
    fn gcd_with_n(&self, a: &Self::El) -> BigInt;
}

/// Plain `BigInt` arithmetic modulo `n`.
struct BigRing {
    n: BigInt,
}

impl ModRing for BigRing {
    type El = BigInt;
    fn embed(&self, x: &BigInt) -> BigInt {
        x.mod_floor(&self.n)
    }
    fn add(&self, a: &BigInt, b: &BigInt) -> BigInt {
        let s = a + b;
        if s >= self.n { s - &self.n } else { s }
    }
    fn sub(&self, a: &BigInt, b: &BigInt) -> BigInt {
        if a >= b { a - b } else { a + &self.n - b }
    }
    fn mul(&self, a: &BigInt, b: &BigInt) -> BigInt {
        (a * b) % &self.n
    }
    fn is_zero(&self, a: &BigInt) -> bool {
        a.is_zero()
    }
    fn gcd_with_n(&self, a: &BigInt) -> BigInt {
        a.gcd(&self.n)
    }
}

/// Montgomery arithmetic modulo an odd `n < 2¹²⁷` with `R = 2¹²⁸`.
struct Mont128 {
    n: u128,
    /// `−n⁻¹ mod 2¹²⁸`
    n_inv_neg: u128,
    /// `R² mod n` (for conversion into Montgomery form)
    r2: u128,
}

/// Full 256-bit product of two `u128`s as `(hi, lo)`.
#[inline]
fn mul_wide_u128(a: u128, b: u128) -> (u128, u128) {
    const M64: u128 = (1u128 << 64) - 1;
    let (a1, a0) = (a >> 64, a & M64);
    let (b1, b0) = (b >> 64, b & M64);
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let (mid, mid_carry) = p01.overflowing_add(p10);
    let (lo, lo_carry) = p00.overflowing_add(mid << 64);
    let hi = p11 + (mid >> 64) + ((mid_carry as u128) << 64) + (lo_carry as u128);
    (hi, lo)
}

fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

impl Mont128 {
    /// `n` must be odd and `< 2¹²⁷`.
    fn new(n: u128) -> Self {
        debug_assert!(n % 2 == 1 && n < (1u128 << 127));
        // Newton iteration for n⁻¹ mod 2¹²⁸ (7 steps double the precision
        // from the 1 correct bit of `inv = 1`).
        let mut inv: u128 = 1;
        for _ in 0..7 {
            inv = inv.wrapping_mul(2u128.wrapping_sub(n.wrapping_mul(inv)));
        }
        let n_inv_neg = 0u128.wrapping_sub(inv);
        // R² mod n via BigInt (done once).
        let nb = BigInt::from(n);
        let r2 = ((BigInt::one() << 256usize) % &nb).to_u128().unwrap_or(0);
        Mont128 { n, n_inv_neg, r2 }
    }

    /// Montgomery reduction of the 256-bit value `(hi, lo)`: returns
    /// `(hi·2¹²⁸ + lo) · R⁻¹ mod n`.
    #[inline]
    fn redc(&self, hi: u128, lo: u128) -> u128 {
        let m = lo.wrapping_mul(self.n_inv_neg);
        let (mn_hi, mn_lo) = mul_wide_u128(m, self.n);
        let (_, carry) = lo.overflowing_add(mn_lo);
        // hi + mn_hi + carry < 2n < 2^128 because n < 2^127.
        let t = hi + mn_hi + carry as u128;
        if t >= self.n { t - self.n } else { t }
    }

    #[inline]
    fn mont_mul(&self, a: u128, b: u128) -> u128 {
        let (hi, lo) = mul_wide_u128(a, b);
        self.redc(hi, lo)
    }

    fn to_mont(&self, x: u128) -> u128 {
        self.mont_mul(x % self.n, self.r2)
    }

    fn mont_to_plain(&self, x: u128) -> u128 {
        self.redc(0, x)
    }
}

impl ModRing for Mont128 {
    type El = u128;
    fn embed(&self, x: &BigInt) -> u128 {
        let r = x.mod_floor(&BigInt::from(self.n)).to_u128().unwrap_or(0);
        self.to_mont(r)
    }
    #[inline]
    fn add(&self, a: &u128, b: &u128) -> u128 {
        let s = a + b;
        if s >= self.n { s - self.n } else { s }
    }
    #[inline]
    fn sub(&self, a: &u128, b: &u128) -> u128 {
        if a >= b { a - b } else { a + self.n - b }
    }
    #[inline]
    fn mul(&self, a: &u128, b: &u128) -> u128 {
        self.mont_mul(*a, *b)
    }
    fn is_zero(&self, a: &u128) -> bool {
        *a == 0
    }
    fn gcd_with_n(&self, a: &u128) -> BigInt {
        BigInt::from(gcd_u128(self.mont_to_plain(*a), self.n))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BigInt factorization: rho + ECM
// ═══════════════════════════════════════════════════════════════════════════

/// Pollard–Brent rho with an iteration budget, over an abstract ring.
fn pollard_brent_ring<R: ModRing>(
    ring: &R,
    n: &BigInt,
    seed: u64,
    max_iters: u64,
) -> Option<BigInt> {
    let c = ring.embed(&BigInt::from(1 + seed));
    let f = |x: &R::El| -> R::El { ring.add(&ring.mul(x, x), &c) };
    let mut rng = XorShift::new(seed.wrapping_mul(0x1234_5678_9ABC_DEF1));
    let mut y = ring.embed(&rng.next_big_below(n));
    let mut x = y.clone();
    let mut ys = y.clone();
    let one = ring.embed(&BigInt::one());
    let mut g = BigInt::one();
    let mut q = one.clone();
    let mut r = 1u64;
    let m = 256u64;
    let mut iters = 0u64;
    while g.is_one() {
        x = y.clone();
        for _ in 0..r {
            y = f(&y);
        }
        let mut k = 0u64;
        while k < r && g.is_one() {
            ys = y.clone();
            let lim = m.min(r - k);
            for _ in 0..lim {
                y = f(&y);
                q = ring.mul(&q, &ring.sub(&x, &y));
            }
            g = ring.gcd_with_n(&q);
            k += lim;
            iters += lim;
        }
        r *= 2;
        if iters > max_iters {
            return None;
        }
    }
    if &g == n {
        // Backtrack one step at a time.
        let mut extra = 0u64;
        loop {
            ys = f(&ys);
            g = ring.gcd_with_n(&ring.sub(&x, &ys));
            if g > BigInt::one() {
                break;
            }
            extra += 1;
            if extra > max_iters {
                return None;
            }
        }
    }
    if &g == n || g.is_one() { None } else { Some(g) }
}

/// One ECM stage-1 attempt on a Montgomery curve chosen by Suyama's
/// parametrization with parameter `sigma`, over an abstract ring.
/// Returns a non-trivial factor of `n` on success.
fn ecm_stage1_ring<R: ModRing>(ring: &R, n: &BigInt, sigma: u64, b1: u64) -> Option<BigInt> {
    // Curve parameters computed once in BigInt, then moved into the ring.
    let sigma = BigInt::from(sigma);
    let u = (&sigma * &sigma - BigInt::from(5)).mod_floor(n);
    let v = (BigInt::from(4) * &sigma).mod_floor(n);
    let u3 = (&u * &u * &u).mod_floor(n);
    let v3 = (&v * &v * &v).mod_floor(n);
    // a24 = (A + 2)/4 = (v − u)³ (3u + v) / (16 u³ v)
    let vmu = (&v - &u).mod_floor(n);
    let numer = (&vmu * &vmu * &vmu * (BigInt::from(3) * &u + &v)).mod_floor(n);
    let denom = (BigInt::from(16) * &u3 * &v).mod_floor(n);
    let g = denom.gcd(n);
    if !g.is_one() {
        return if &g == n { None } else { Some(g) };
    }
    let (_, inv, _) = extended_gcd_big(&denom, n);
    let a24 = ring.embed(&(numer * inv).mod_floor(n));

    let mut x = ring.embed(&u3);
    let mut z = ring.embed(&v3);

    let dbl = |x: &R::El, z: &R::El| -> (R::El, R::El) {
        let s = ring.add(x, z);
        let d = ring.sub(x, z);
        let t1 = ring.mul(&s, &s);
        let t2 = ring.mul(&d, &d);
        let diff = ring.sub(&t1, &t2);
        let x2 = ring.mul(&t1, &t2);
        let z2 = ring.mul(&diff, &ring.add(&t2, &ring.mul(&a24, &diff)));
        (x2, z2)
    };
    let dadd = |x1: &R::El, z1: &R::El, x2: &R::El, z2: &R::El, x0: &R::El, z0: &R::El| {
        let a = ring.mul(&ring.sub(x1, z1), &ring.add(x2, z2));
        let b = ring.mul(&ring.add(x1, z1), &ring.sub(x2, z2));
        let s = ring.add(&a, &b);
        let d = ring.sub(&a, &b);
        let x3 = ring.mul(z0, &ring.mul(&s, &s));
        let z3 = ring.mul(x0, &ring.mul(&d, &d));
        (x3, z3)
    };
    let ladder = |k: u64, x: &R::El, z: &R::El| -> (R::El, R::El) {
        let (mut x1, mut z1) = (x.clone(), z.clone());
        let (mut x2, mut z2) = dbl(x, z);
        let bits = 64 - k.leading_zeros();
        for i in (0..bits - 1).rev() {
            if (k >> i) & 1 == 1 {
                let (nx1, nz1) = dadd(&x1, &z1, &x2, &z2, x, z);
                let (nx2, nz2) = dbl(&x2, &z2);
                x1 = nx1;
                z1 = nz1;
                x2 = nx2;
                z2 = nz2;
            } else {
                let (nx2, nz2) = dadd(&x1, &z1, &x2, &z2, x, z);
                let (nx1, nz1) = dbl(&x1, &z1);
                x1 = nx1;
                z1 = nz1;
                x2 = nx2;
                z2 = nz2;
            }
        }
        (x1, z1)
    };

    for &p in small_primes() {
        let p = p as u64;
        if p > b1 {
            break;
        }
        // Largest power of p not exceeding b1.
        let mut pk = p;
        while pk * p <= b1 {
            pk *= p;
        }
        let (nx, nz) = ladder(pk, &x, &z);
        x = nx;
        z = nz;
        if ring.is_zero(&z) {
            return None;
        }
    }
    // Primes between 2^16 and b1 (if b1 is that large) via a temporary sieve.
    if b1 > TRIAL_DIVISION_LIMIT as u64 {
        let sieve = BitSieve::new(b1);
        let mut p = TRIAL_DIVISION_LIMIT as u64 + 1;
        while p <= b1 {
            if sieve.is_prime(p) {
                let (nx, nz) = ladder(p, &x, &z);
                x = nx;
                z = nz;
            }
            p += 2;
        }
    }
    let g = ring.gcd_with_n(&z);
    if g.is_one() || &g == n { None } else { Some(g) }
}

/// ECM `B1` schedule: `(B1, number of curves)` tuned for roughly
/// 15 / 20 / 25 / 30-digit factors.
const ECM_SCHEDULE: [(u64, u64); 4] = [(2_000, 40), (11_000, 120), (50_000, 400), (250_000, 1_000)];

/// rho + ECM driver over a concrete ring.
fn split_with_ring<R: ModRing>(ring: &R, n: &BigInt) -> BigInt {
    // Cheap rho pass for small factors.
    for seed in 1..=3u64 {
        if let Some(d) = pollard_brent_ring(ring, n, seed, 1 << 14) {
            return d;
        }
    }
    let mut sigma = 6u64;
    for (b1, curves) in ECM_SCHEDULE {
        for _ in 0..curves {
            if let Some(d) = ecm_stage1_ring(ring, n, sigma, b1) {
                return d;
            }
            sigma += 1;
        }
    }
    // Last resort: unbounded rho (always terminates for composite n).
    let mut seed = 10u64;
    loop {
        if let Some(d) = pollard_brent_ring(ring, n, seed, u64::MAX) {
            return d;
        }
        seed += 1;
    }
}

/// Find a non-trivial factor of a BigInt composite that has no prime
/// factor below [`TRIAL_DIVISION_LIMIT`] and does not fit in `u64`.
fn split_big_composite(n: &BigInt) -> BigInt {
    if let Some((base, _)) = perfect_power_big(n) {
        return base;
    }
    if n.is_even() {
        return BigInt::from(2);
    }
    match n.to_u128() {
        Some(nu) if nu < (1u128 << 127) => split_with_ring(&Mont128::new(nu), n),
        _ => split_with_ring(&BigRing { n: n.clone() }, n),
    }
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

    // Trial division by all primes below 2^16.
    for &p in small_primes() {
        let pb = BigInt::from(p);
        if &pb * &pb > n {
            break;
        }
        if (&n % p).is_zero() {
            let mut count = 0u32;
            while (&n % p).is_zero() {
                n /= p;
                count += 1;
            }
            factors.push((pb, count));
        }
    }

    // Remaining cofactors on a work stack.
    let mut primes_found: Vec<BigInt> = Vec::new();
    let mut stack: Vec<BigInt> = vec![n];
    while let Some(m) = stack.pop() {
        if m.is_one() {
            continue;
        }
        if let Some(u) = m.to_u64() {
            if u < (TRIAL_DIVISION_LIMIT as u64) * (TRIAL_DIVISION_LIMIT as u64) {
                primes_found.push(m);
            } else {
                let mut out = Vec::new();
                factor_u64_no_small(u, &mut out);
                primes_found.extend(out.into_iter().map(BigInt::from));
            }
            continue;
        }
        if isprime_big_internal(&m) {
            primes_found.push(m);
            continue;
        }
        let d = split_big_composite(&m);
        let q = &m / &d;
        stack.push(d);
        stack.push(q);
    }

    primes_found.sort();
    for p in primes_found {
        match factors.last_mut() {
            Some((q, e)) if *q == p => *e += 1,
            _ => factors.push((p, 1)),
        }
    }
    factors.sort_by(|a, b| a.0.cmp(&b.0));
    factors
}

/// Next prime after `n` (i64 fast path).
fn nextprime_i64(n: i64) -> Option<i64> {
    if n < 2 {
        return Some(2);
    }
    let mut candidate = if n % 2 == 0 {
        n.checked_add(1)?
    } else {
        n.checked_add(2)?
    };
    while !isprime_i64(candidate) {
        candidate = candidate.checked_add(2)?;
    }
    Some(candidate)
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
    Some((n as u64).sqrt() as i64)
}

/// Integer square root for BigInt.
fn isqrt_big(n: &BigInt) -> Option<BigInt> {
    if n.is_negative() {
        return None;
    }
    Some(n.sqrt())
}

fn is_square_big(n: &BigInt) -> bool {
    if n.is_negative() {
        return false;
    }
    let s = n.sqrt();
    &s * &s == *n
}

/// Perfect power detection on `|n| ≥ 2`: returns `(base, exponent)` with
/// the exponent maximal.
fn perfect_power_big(n: &BigInt) -> Option<(BigInt, u32)> {
    let negative = n.is_negative();
    let m = n.abs();
    if m < BigInt::from(2) {
        return None;
    }
    let max_exp = m.bits() as u32; // 2^bits > m
    let mut result: Option<(BigInt, u32)> = None;
    // Test prime exponents from small to large; recurse on the base.
    for &q in small_primes() {
        if q > max_exp {
            break;
        }
        if negative && q == 2 {
            continue;
        }
        let r = m.nth_root(q);
        if r.pow(q) == m {
            let base = if negative { -r } else { r };
            let (b, e) = perfect_power_big(&base).unwrap_or((base, 1));
            result = Some((b, e * q));
            break;
        }
    }
    result
}

/// Jacobi symbol `(a/n)` for odd positive `n`.
fn jacobi_big(a: &BigInt, n: &BigInt) -> i8 {
    debug_assert!(n.is_positive() && n.is_odd());
    let mut a = a.mod_floor(n);
    let mut n = n.clone();
    let mut result: i8 = 1;
    let three = BigInt::from(3);
    let five = BigInt::from(5);
    let eight = BigInt::from(8);
    let four = BigInt::from(4);
    while !a.is_zero() {
        while a.is_even() {
            a /= 2;
            let r = n.mod_floor(&eight);
            if r == three || r == five {
                result = -result;
            }
        }
        std::mem::swap(&mut a, &mut n);
        if a.mod_floor(&four) == three && n.mod_floor(&four) == three {
            result = -result;
        }
        a = a.mod_floor(&n);
    }
    if n.is_one() { result } else { 0 }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public unified API — primality & factorization
// ═══════════════════════════════════════════════════════════════════════════

/// Primality test for any integer.
///
/// Deterministic (Miller–Rabin with the first 13 prime bases) for
/// `n < 3.3·10²⁴`; Baillie–PSW beyond that.  BPSW has no known
/// counterexamples but is not proven deterministic — see the module docs.
///
/// Accepts any integer type (`i64`, `i32`, `u64`, `BigInt`, etc.) via
/// `impl Into<BigInt>`.
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
/// // Mersenne prime M127 = 2^127 − 1 (BPSW range)
/// let m127 = (BigInt::from(1) << 127) - 1;
/// assert!(isprime(m127));
/// ```
pub fn isprime(n: impl Into<BigInt>) -> bool {
    let n: BigInt = n.into();
    // Fast path: if fits in i64, use optimized version
    if let Some(n_i64) = n.to_i64() {
        return isprime_i64(n_i64);
    }
    isprime_big_internal(&n)
}

/// Randomized Miller–Rabin probable-prime test with `rounds` random bases
/// (plus base 2).
///
/// A composite passes each round with probability at most `1/4`, so the
/// error probability is bounded by `4^(−rounds)`.  Bases are drawn from a
/// deterministic generator seeded by `n`, so results are reproducible.
/// For values below `3.3·10²⁴` this simply defers to the deterministic
/// [`isprime`].
///
/// # Examples
///
/// ```
/// use symplex::ntheory::is_probable_prime;
/// use num_bigint::BigInt;
///
/// let p = BigInt::parse_bytes(b"170141183460469231731687303715884105727", 10).unwrap(); // 2^127 − 1
/// assert!(is_probable_prime(p.clone(), 20));
/// assert!(!is_probable_prime(&p * 3, 20));
/// ```
pub fn is_probable_prime(n: impl Into<BigInt>, rounds: u32) -> bool {
    let n: BigInt = n.into();
    if n < BigInt::from(2) {
        return false;
    }
    if n.to_u64().is_some() {
        return isprime(n);
    }
    if n.is_even() {
        return false;
    }
    if !strong_fermat_big(&n, &BigInt::from(2)) {
        return false;
    }
    let seed = n.to_u64_digits().1.first().copied().unwrap_or(1);
    let mut rng = XorShift::new(seed);
    let n_minus_2 = &n - BigInt::from(2);
    for _ in 0..rounds {
        let a = rng.next_big_below(&n_minus_2) + BigInt::from(2);
        if !strong_fermat_big(&n, &a) {
            return false;
        }
    }
    true
}

/// Factorize an integer into prime factors.
///
/// Accepts any integer type (`i64`, `i32`, `u64`, `BigInt`, etc.).
/// Returns `(prime, exponent)` pairs in ascending order of prime.
/// Returns an empty vector for `n ∈ {-1, 0, 1}`. Negative inputs are
/// factored by absolute value.
///
/// Algorithm: trial division by all primes below 2¹⁶, perfect-power
/// detection, Pollard–Brent rho (machine-word arithmetic below 2⁶⁴), then
/// ECM stage 1 on Montgomery curves for larger composites.  Every reported
/// factor is verified prime with [`isprime`].
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
///
/// // 10^18 + 9: factors multiply back exactly and are all prime
/// let f = factorint(1_000_000_000_000_000_009i64);
/// let back: BigInt = f.iter().map(|(p, e)| p.pow(*e)).product();
/// assert_eq!(back, BigInt::from(1_000_000_000_000_000_009i64));
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

/// Bounded partial factorisation for radical extraction.
///
/// Returns `(factors, cofactor)` with `|n| = ∏ pᵢ^{eᵢ} · cofactor`.  The
/// work is bounded so the function is safe to call from hot construction
/// paths (`√n` canonicalisation): trial division by the primes below 2¹⁶,
/// then — for whatever remains — a primality test and a perfect-power
/// check, and a full [`factorint`] only when the remaining composite has
/// at most `max_bits` bits (Pollard rho on a `b`-bit semiprime costs
/// `~2^{b/4}` steps).  A larger composite is returned unfactored as
/// `cofactor` (which is then `> 1`); otherwise `cofactor == 1`.
///
/// Returns `(vec![], 1)` for `n ∈ {-1, 0, 1}`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::factorint_bounded;
/// use num_bigint::BigInt;
///
/// let (f, c) = factorint_bounded(&BigInt::from(3_912_608_155_036_063i64), 64);
/// assert_eq!(f, vec![(BigInt::from(7), 1), (BigInt::from(23_641_997), 2)]);
/// assert_eq!(c, BigInt::from(1));
///
/// // 2^127 − 1 is prime: recognised without any factoring effort.
/// let m127 = (BigInt::from(1) << 127) - 1;
/// let (f, c) = factorint_bounded(&m127, 64);
/// assert_eq!(f, vec![(m127, 1)]);
/// assert_eq!(c, BigInt::from(1));
///
/// // A 40-digit semiprime with 20-digit prime factors is left alone.
/// let p = BigInt::parse_bytes(b"18446744073709551629", 10).unwrap(); // > 2^64, prime
/// let q = BigInt::parse_bytes(b"18446744073709551653", 10).unwrap(); // prime
/// let (f, c) = factorint_bounded(&(&p * &q * 4), 64);
/// assert_eq!(f, vec![(BigInt::from(2), 2)]);
/// assert_eq!(c, &p * &q);
/// ```
pub fn factorint_bounded(n: &BigInt, max_bits: u64) -> (Vec<(BigInt, u32)>, BigInt) {
    let mut n = n.abs();
    if n <= BigInt::one() {
        return (vec![], BigInt::one());
    }
    let mut factors: Vec<(BigInt, u32)> = Vec::new();

    // Trial division by the primes below 2^16 (cheap on machine words).
    if let Some(mut small) = n.to_u64() {
        for &p in small_primes() {
            let p = p as u64;
            if p * p > small {
                break;
            }
            if small.is_multiple_of(p) {
                let mut count = 0u32;
                while small.is_multiple_of(p) {
                    small /= p;
                    count += 1;
                }
                factors.push((BigInt::from(p), count));
            }
        }
        n = BigInt::from(small);
    } else {
        for &p in small_primes() {
            if (&n % p).is_zero() {
                let mut count = 0u32;
                while (&n % p).is_zero() {
                    n /= p;
                    count += 1;
                }
                factors.push((BigInt::from(p), count));
            }
        }
    }

    if n.is_one() {
        return (factors, n);
    }

    // Any remainder below 2^32 has no prime factor below 2^16, so it is prime.
    if n < BigInt::from((TRIAL_DIVISION_LIMIT as u64) * (TRIAL_DIVISION_LIMIT as u64)) {
        merge_factors(&mut factors, vec![(n, 1)]);
        return (factors, BigInt::one());
    }
    // Known-cheap to factor: finish the job.
    if n.bits() <= max_bits {
        merge_factors(&mut factors, factorint(n));
        return (factors, BigInt::one());
    }

    // Large remainder: only the cheap structural checks.
    if isprime_big_internal(&n) {
        merge_factors(&mut factors, vec![(n, 1)]);
        return (factors, BigInt::one());
    }
    if let Some((base, e)) = perfect_power_big(&n) {
        let (inner, cof) = factorint_bounded(&base, max_bits);
        merge_factors(
            &mut factors,
            inner.into_iter().map(|(p, k)| (p, k * e)).collect(),
        );
        // An unfactored cofactor of the base contributes `cof^e`.
        return (factors, cof.pow(e));
    }
    (factors, n)
}

/// Merge `more` (sorted or not) into the sorted `(prime, exponent)` list.
fn merge_factors(factors: &mut Vec<(BigInt, u32)>, more: Vec<(BigInt, u32)>) {
    for (p, e) in more {
        match factors.iter_mut().find(|(q, _)| *q == p) {
            Some((_, k)) => *k += e,
            None => factors.push((p, e)),
        }
    }
    factors.sort_by(|a, b| a.0.cmp(&b.0));
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
        // Guard: skip i64 fast path near i64::MAX to avoid overflow in
        // candidate arithmetic (n+1, n+2, candidate+=2 can all wrap).
        if n_i64 <= i64::MAX - 1000
            && let Some(result) = nextprime_i64(n_i64)
        {
            return BigInt::from(result);
        }
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

/// The `n`-th prime (1-indexed: `prime(1) = 2`).
///
/// Uses a sieve with the Rosser–Schoenfeld upper bound.  Returns `None`
/// for `n = 0` or `n > 10⁷` (the sieve size cap).
///
/// # Examples
///
/// ```
/// use symplex::ntheory::prime;
/// use num_bigint::BigInt;
///
/// assert_eq!(prime(1), Some(BigInt::from(2)));
/// assert_eq!(prime(100), Some(BigInt::from(541)));
/// assert_eq!(prime(10_000), Some(BigInt::from(104_729)));
/// assert_eq!(prime(0), None);
/// ```
pub fn prime(n: impl Into<BigInt>) -> Option<BigInt> {
    let n: BigInt = n.into();
    let n = n.to_u64()?;
    if n == 0 || n > 10_000_000 {
        return None;
    }
    if n < 6 {
        return Some(BigInt::from([2u64, 3, 5, 7, 11][n as usize - 1]));
    }
    let nf = n as f64;
    let bound = (nf * (nf.ln() + nf.ln().ln())).ceil() as u64 + 10;
    let sieve = BitSieve::new(bound);
    let mut count = 0u64;
    let mut k = 2u64;
    while k <= bound {
        if sieve.is_prime(k) {
            count += 1;
            if count == n {
                return Some(BigInt::from(k));
            }
        }
        k += if k == 2 { 1 } else { 2 };
    }
    None
}

/// Prime-counting function `π(n)`: the number of primes `≤ n`.
///
/// Uses the Lucy_Hedgehog `O(n^{3/4})` algorithm.  Returns `None` for
/// `n > 10¹²` (time cap); negative `n` gives `Some(0)`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::primepi;
///
/// assert_eq!(primepi(10), Some(4));
/// assert_eq!(primepi(100), Some(25));
/// assert_eq!(primepi(1_000_000), Some(78_498));
/// assert_eq!(primepi(-5), Some(0));
/// ```
pub fn primepi(n: impl Into<BigInt>) -> Option<u64> {
    let n: BigInt = n.into();
    if n < BigInt::from(2) {
        return Some(0);
    }
    let n = n.to_u64()?;
    if n > 1_000_000_000_000 {
        return None;
    }
    Some(primepi_u64(n))
}

/// Lucy_Hedgehog prime counting.
fn primepi_u64(n: u64) -> u64 {
    if n < 2 {
        return 0;
    }
    let r = n.sqrt();
    // lo[v] = S(v) for v ≤ r ; hi[i] = S(n / i) for 1 ≤ i ≤ r
    let rs = r as usize;
    let mut lo: Vec<u64> = (0..=rs).map(|v| v.saturating_sub(1) as u64).collect();
    let mut hi: Vec<u64> = (0..=rs)
        .map(|i| if i == 0 { 0 } else { n / i as u64 - 1 })
        .collect();
    for p in 2..=rs {
        if lo[p] == lo[p - 1] {
            continue; // p is not prime
        }
        let sp = lo[p - 1];
        let p2 = (p * p) as u64;
        let pu = p as u64;
        // Update hi[i] for n/i ≥ p²  ⇔  i ≤ n / p²
        let i_max = (n / p2).min(r) as usize;
        for i in 1..=i_max {
            let ip = i as u64 * pu;
            let sub = if ip <= r {
                hi[ip as usize]
            } else {
                lo[(n / ip) as usize]
            };
            hi[i] -= sub - sp;
        }
        // Update lo[v] for v ≥ p², descending
        let mut v = rs;
        while v as u64 >= p2 {
            lo[v] -= lo[v / p] - sp;
            v -= 1;
        }
    }
    hi[1]
}

/// All primes in the half-open range `[a, b)`, in increasing order.
///
/// Uses a segmented sieve when `b ≤ 10¹⁴` and `b − a ≤ 10⁷`; otherwise
/// walks with [`nextprime`] (correct for any size, but slow for wide
/// ranges of huge numbers).
///
/// # Examples
///
/// ```
/// use symplex::ntheory::primerange;
/// use num_bigint::BigInt;
///
/// let ps: Vec<i64> = primerange(10, 30).iter().map(|p| p.try_into().unwrap()).collect();
/// assert_eq!(ps, vec![11, 13, 17, 19, 23, 29]);
/// assert!(primerange(5, 5).is_empty());
/// ```
pub fn primerange(a: impl Into<BigInt>, b: impl Into<BigInt>) -> Vec<BigInt> {
    let a: BigInt = a.into();
    let b: BigInt = b.into();
    let a = if a < BigInt::from(2) {
        BigInt::from(2)
    } else {
        a
    };
    if a >= b {
        return vec![];
    }
    let width = &b - &a;
    if let (Some(lo), Some(hi)) = (a.to_u64(), b.to_u64())
        && hi <= 100_000_000_000_000
        && width <= BigInt::from(10_000_000u64)
    {
        return segmented_sieve(lo, hi)
            .into_iter()
            .map(BigInt::from)
            .collect();
    }
    let mut out = Vec::new();
    let mut p = if isprime_big_internal(&a) {
        a.clone()
    } else {
        nextprime(a)
    };
    while p < b {
        out.push(p.clone());
        p = nextprime(p);
    }
    out
}

/// Primes in `[lo, hi)` by segmented sieving.
fn segmented_sieve(lo: u64, hi: u64) -> Vec<u64> {
    if hi <= 2 || lo >= hi {
        return vec![];
    }
    let root = (hi - 1).sqrt() + 1;
    let base_primes: Vec<u64> = sieve_u32((root + 1).min(u32::MAX as u64) as u32)
        .into_iter()
        .map(u64::from)
        .collect();
    let len = (hi - lo) as usize;
    let mut composite = vec![false; len];
    for &p in &base_primes {
        if p * p >= hi {
            break;
        }
        let mut start = lo.div_ceil(p) * p;
        if start < p * p {
            start = p * p;
        }
        let mut m = start;
        while m < hi {
            composite[(m - lo) as usize] = true;
            m += p;
        }
    }
    (0..len)
        .filter(|&i| {
            let v = lo + i as u64;
            v >= 2 && !composite[i]
        })
        .map(|i| lo + i as u64)
        .collect()
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
    divisor_sigma(n, 1)
}

/// Divisor function `σₖ(n) = Σ_{d | n} dᵏ` over the positive divisors of
/// `|n|`.  `σ₀` counts divisors, `σ₁` sums them.  Returns `0` for `n = 0`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::divisor_sigma;
/// use num_bigint::BigInt;
///
/// assert_eq!(divisor_sigma(12, 0), BigInt::from(6));
/// assert_eq!(divisor_sigma(12, 1), BigInt::from(28));
/// assert_eq!(divisor_sigma(12, 2), BigInt::from(210));  // 1+4+9+16+36+144
/// ```
pub fn divisor_sigma(n: impl Into<BigInt>, k: u32) -> BigInt {
    let n: BigInt = n.into();
    if n.is_zero() {
        return BigInt::zero();
    }
    let mut result = BigInt::one();
    for (p, e) in factorint(n.abs()) {
        // Σ_{i=0}^{e} p^{ik}
        let pk = p.pow(k);
        let mut term = BigInt::one();
        let mut acc = BigInt::one();
        for _ in 0..e {
            acc *= &pk;
            term += &acc;
        }
        result *= term;
    }
    result
}

/// Is `n` perfect (`σ(n) = 2n`)?  Requires `n ≥ 1`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::is_perfect;
/// assert!(is_perfect(6));
/// assert!(is_perfect(28));
/// assert!(!is_perfect(12));
/// ```
pub fn is_perfect(n: impl Into<BigInt>) -> bool {
    let n: BigInt = n.into();
    n.is_positive() && divisor_sigma(n.clone(), 1) == &n * 2
}

/// Is `n` abundant (`σ(n) > 2n`)?  Requires `n ≥ 1`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::is_abundant;
/// assert!(is_abundant(12));
/// assert!(!is_abundant(6));
/// ```
pub fn is_abundant(n: impl Into<BigInt>) -> bool {
    let n: BigInt = n.into();
    n.is_positive() && divisor_sigma(n.clone(), 1) > &n * 2
}

/// Is `n` deficient (`σ(n) < 2n`)?  Requires `n ≥ 1`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::is_deficient;
/// assert!(is_deficient(8));
/// assert!(!is_deficient(12));
/// ```
pub fn is_deficient(n: impl Into<BigInt>) -> bool {
    let n: BigInt = n.into();
    n.is_positive() && divisor_sigma(n.clone(), 1) < &n * 2
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

/// Carmichael's function λ(n): the exponent of the multiplicative group
/// `(ℤ/nℤ)×`, i.e. the smallest `m` with `aᵐ ≡ 1 (mod n)` for all `a`
/// coprime to `n`.  Returns 0 for `n ≤ 0`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::carmichael_lambda;
/// use num_bigint::BigInt;
///
/// assert_eq!(carmichael_lambda(1), BigInt::from(1));
/// assert_eq!(carmichael_lambda(8), BigInt::from(2));
/// assert_eq!(carmichael_lambda(15), BigInt::from(4));
/// assert_eq!(carmichael_lambda(561), BigInt::from(80));  // Carmichael number
/// ```
pub fn carmichael_lambda(n: impl Into<BigInt>) -> BigInt {
    let n: BigInt = n.into();
    if n <= BigInt::zero() {
        return BigInt::zero();
    }
    let mut result = BigInt::one();
    for (p, e) in factorint(n) {
        let lam = if p == BigInt::from(2) {
            match e {
                1 => BigInt::one(),
                2 => BigInt::from(2),
                _ => BigInt::one() << (e - 2) as usize,
            }
        } else {
            p.pow(e - 1) * (&p - BigInt::one())
        };
        result = result.lcm(&lam);
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

/// Perfect-power decomposition: `Some((b, k))` with `bᵏ = n`, `k ≥ 2`
/// maximal and `b` not itself a perfect power.  Negative `n` are handled
/// with odd `k`.  Returns `None` for `|n| < 2` and for non-powers.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::perfect_power;
/// use num_bigint::BigInt;
///
/// assert_eq!(perfect_power(64), Some((BigInt::from(2), 6)));
/// assert_eq!(perfect_power(36), Some((BigInt::from(6), 2)));
/// assert_eq!(perfect_power(-27), Some((BigInt::from(-3), 3)));
/// assert_eq!(perfect_power(10), None);
/// ```
pub fn perfect_power(n: impl Into<BigInt>) -> Option<(BigInt, u32)> {
    let n: BigInt = n.into();
    perfect_power_big(&n)
}

/// Is `n` a perfect power `bᵏ` with `k ≥ 2`?
///
/// # Examples
///
/// ```
/// use symplex::ntheory::is_perfect_power;
/// assert!(is_perfect_power(1024));
/// assert!(!is_perfect_power(1000_i64 + 1));
/// ```
pub fn is_perfect_power(n: impl Into<BigInt>) -> bool {
    perfect_power(n).is_some()
}

/// Lucas–Lehmer test: is the Mersenne number `2ᵖ − 1` prime?
///
/// Returns `false` for composite or non-positive `p` (a necessary
/// condition), `true` for `p = 2`, and otherwise runs the exact
/// Lucas–Lehmer iteration.  Deterministic.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::is_mersenne_prime;
/// assert!(is_mersenne_prime(2));
/// assert!(is_mersenne_prime(31));   // 2^31 − 1
/// assert!(is_mersenne_prime(127));  // 2^127 − 1
/// assert!(!is_mersenne_prime(11));  // 2047 = 23 · 89
/// assert!(!is_mersenne_prime(4));
/// ```
pub fn is_mersenne_prime(p: impl Into<BigInt>) -> bool {
    let p: BigInt = p.into();
    let Some(p) = p.to_u64() else {
        return false;
    };
    if !isprime_u64(p) {
        return false;
    }
    if p == 2 {
        return true;
    }
    let m = (BigInt::one() << p as usize) - BigInt::one();
    let mut s = BigInt::from(4);
    for _ in 0..(p - 2) {
        s = (&s * &s - BigInt::from(2)) % &m;
    }
    s.is_zero()
}

// ═══════════════════════════════════════════════════════════════════════════
// Modular arithmetic
// ═══════════════════════════════════════════════════════════════════════════

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
    if n <= BigInt::one() {
        return None;
    }
    let (g, x, _) = extended_gcd_big(&a, &n);
    if !g.is_one() {
        return None;
    }
    Some(x.mod_floor(&n))
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
    if moduli.iter().any(|m| !m.is_positive()) {
        return None;
    }
    let mut result = remainders[0].mod_floor(&moduli[0]);
    let mut modulus = moduli[0].clone();
    for i in 1..remainders.len() {
        let (g, p, _) = extended_gcd_big(&modulus, &moduli[i]);
        let diff = &remainders[i] - &result;
        if !(&diff % &g).is_zero() {
            return None;
        }
        let step = &modulus * ((&diff / &g).mod_floor(&(&moduli[i] / &g))) * &p;
        result = &result + &step;
        modulus = &modulus / &g * &moduli[i];
        result = result.mod_floor(&modulus);
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

/// Extended GCD: [`ExtendedGcd`]` { gcd, x, y }` with
/// `a·x + b·y = gcd = gcd(a, b) ≥ 0`.
///
/// This is [`num_integer::Integer::extended_gcd`] on `BigInt` — the same
/// iterative Euclidean algorithm and sign normalisation the crate always
/// used — behind the `impl Into<BigInt>` convenience of this module.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::gcdex;
/// use num_bigint::BigInt;
///
/// let g = gcdex(240, 46);
/// assert_eq!(g.gcd, BigInt::from(2));
/// assert_eq!(BigInt::from(240) * &g.x + BigInt::from(46) * &g.y, g.gcd);
/// ```
pub fn gcdex(a: impl Into<BigInt>, b: impl Into<BigInt>) -> ExtendedGcd<BigInt> {
    let a: BigInt = a.into();
    let b: BigInt = b.into();
    Integer::extended_gcd(&a, &b)
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
    let base_mod = base.mod_floor(&modulus);
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
    is_square_big(&n)
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

/// Integer `k`-th root: the largest `r` with `rᵏ ≤ n`.
///
/// Returns `None` for `k = 0`, or for negative `n` with even `k`.
/// Negative `n` with odd `k` gives the negative root (rounded toward `−∞`
/// in magnitude, i.e. `−⌈|n|^{1/k}⌉`… precisely: the largest `r` with
/// `rᵏ ≤ n`).
///
/// # Examples
///
/// ```
/// use symplex::ntheory::iroot;
/// use num_bigint::BigInt;
///
/// assert_eq!(iroot(1000, 3), Some(BigInt::from(10)));
/// assert_eq!(iroot(1001, 3), Some(BigInt::from(10)));
/// assert_eq!(iroot(-8, 3), Some(BigInt::from(-2)));
/// assert_eq!(iroot(-8, 2), None);
/// ```
pub fn iroot(n: impl Into<BigInt>, k: u32) -> Option<BigInt> {
    let n: BigInt = n.into();
    if k == 0 {
        return None;
    }
    if n.is_negative() {
        if k.is_multiple_of(2) {
            return None;
        }
        let r = n.abs().nth_root(k);
        // Largest r' with r'^k ≤ n (n negative): −r if r^k == |n|, else −(r+1).
        return if r.pow(k) == n.abs() {
            Some(-r)
        } else {
            Some(-(r + BigInt::one()))
        };
    }
    Some(n.nth_root(k))
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
    if limit < u32::MAX as i64 {
        return sieve_u32(limit as u32 + 1)
            .into_iter()
            .map(i64::from)
            .collect();
    }
    segmented_sieve(2, limit as u64 + 1)
        .into_iter()
        .map(|p| p as i64)
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Quadratic residues
// ═══════════════════════════════════════════════════════════════════════════

/// Legendre symbol `(a/p)` for odd prime `p`.
///
/// Returns:
/// - `1` if `a` is a quadratic residue mod `p` and `a ≢ 0`,
/// - `-1` if `a` is a non-residue mod `p`,
/// - `0` if `a ≡ 0 (mod p)`.
///
/// # Panics
///
/// Panics if `p` is not an odd prime.  Use [`jacobi_symbol`] for a
/// non-panicking generalisation.
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
        p > BigInt::from(2) && isprime_big_internal(&p),
        "p must be an odd prime"
    );
    jacobi_big(&a, &p)
}

/// Jacobi symbol `(a/n)` for odd positive `n`.
///
/// Generalises the Legendre symbol to composite odd moduli.  Note that
/// `(a/n) = 1` does **not** imply `a` is a quadratic residue when `n` is
/// composite.
///
/// # Errors
///
/// `InvalidArgument` if `n` is not a positive odd integer.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::jacobi_symbol;
/// assert_eq!(jacobi_symbol(1001, 9907).unwrap(), -1);
/// assert_eq!(jacobi_symbol(2, 15).unwrap(), 1);   // yet 2 is not a QR mod 15
/// assert_eq!(jacobi_symbol(6, 15).unwrap(), 0);
/// assert!(jacobi_symbol(3, 8).is_err());
/// ```
pub fn jacobi_symbol(a: impl Into<BigInt>, n: impl Into<BigInt>) -> Result<i8, SymplexError> {
    let a: BigInt = a.into();
    let n: BigInt = n.into();
    if !n.is_positive() || n.is_even() {
        return Err(SymplexError::InvalidArgument {
            operation: "jacobi_symbol",
            reason: format!("modulus {n} must be a positive odd integer"),
        });
    }
    Ok(jacobi_big(&a, &n))
}

/// Kronecker symbol `(a/n)`, defined for all integers `n`.
///
/// Extends the Jacobi symbol with `(a/2) = 0, 1, −1` for `a` even,
/// `a ≡ ±1 (mod 8)`, `a ≡ ±3 (mod 8)` respectively, `(a/−1) = sign(a)`,
/// and `(a/0) = 1` iff `a = ±1`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::kronecker_symbol;
/// assert_eq!(kronecker_symbol(3, 8), -1);
/// assert_eq!(kronecker_symbol(7, 8), 1);
/// assert_eq!(kronecker_symbol(2, 8), 0);
/// assert_eq!(kronecker_symbol(-5, -1), -1);
/// assert_eq!(kronecker_symbol(1001, 9907), -1);
/// ```
pub fn kronecker_symbol(a: impl Into<BigInt>, n: impl Into<BigInt>) -> i8 {
    let a: BigInt = a.into();
    let n: BigInt = n.into();
    if n.is_zero() {
        return i8::from(a.abs().is_one());
    }
    let mut result: i8 = 1;
    let mut n = n;
    if n.is_negative() {
        n = -n;
        if a.is_negative() {
            result = -result;
        }
    }
    // Factor out powers of two from n.
    let mut twos = 0u32;
    while n.is_even() {
        n /= 2;
        twos += 1;
    }
    if twos > 0 {
        if a.is_even() {
            return 0;
        }
        let r = a.mod_floor(&BigInt::from(8)).to_u32().unwrap_or(0);
        let k2: i8 = if r == 1 || r == 7 { 1 } else { -1 };
        if twos % 2 == 1 {
            result *= k2;
        }
    }
    if n.is_one() {
        return result;
    }
    result * jacobi_big(&a, &n)
}

/// Is `a` a quadratic residue modulo `n` (i.e. does `x² ≡ a (mod n)` have
/// a solution)?  Works for any modulus `n ≥ 1` by solving the congruence.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::is_quad_residue;
/// assert!(is_quad_residue(2, 7));
/// assert!(!is_quad_residue(3, 7));
/// assert!(!is_quad_residue(2, 15));  // Jacobi symbol is 1, but no root exists
/// assert!(is_quad_residue(4, 15));
/// ```
pub fn is_quad_residue(a: impl Into<BigInt>, n: impl Into<BigInt>) -> bool {
    let a: BigInt = a.into();
    let n: BigInt = n.into();
    if n < BigInt::one() {
        return false;
    }
    if n.is_one() {
        return true;
    }
    // Fast path for odd primes: Euler's criterion.
    if n.is_odd() && isprime_big_internal(&n) {
        let am = a.mod_floor(&n);
        return am.is_zero() || jacobi_big(&am, &n) == 1;
    }
    !sqrt_mod_all(a, n).is_empty()
}

/// Tonelli–Shanks: a square root of `a` modulo an odd prime `p` (or `p = 2`).
fn sqrt_mod_prime(a: &BigInt, p: &BigInt) -> Option<BigInt> {
    let a = a.mod_floor(p);
    if a.is_zero() {
        return Some(BigInt::zero());
    }
    if *p == BigInt::from(2) {
        return Some(a);
    }
    if jacobi_big(&a, p) != 1 {
        return None;
    }
    let one = BigInt::one();
    let two = BigInt::from(2);
    // p ≡ 3 (mod 4): direct formula.
    if p.mod_floor(&BigInt::from(4)) == BigInt::from(3) {
        let r = a.modpow(&((p + &one) / 4), p);
        return Some(r);
    }
    // p − 1 = q · 2^s
    let mut q = p - &one;
    let mut s = 0u32;
    while q.is_even() {
        q /= 2;
        s += 1;
    }
    // Find a non-residue z.
    let mut z = two.clone();
    while jacobi_big(&z, p) != -1 {
        z += &one;
    }
    let mut m = s;
    let mut c = z.modpow(&q, p);
    let mut t = a.modpow(&q, p);
    let mut r = a.modpow(&((&q + &one) / 2), p);
    while !t.is_one() {
        // Find least i with t^(2^i) = 1.
        let mut i = 0u32;
        let mut tt = t.clone();
        while !tt.is_one() {
            tt = (&tt * &tt) % p;
            i += 1;
            if i == m {
                return None;
            }
        }
        let mut b = c.clone();
        for _ in 0..(m - i - 1) {
            b = (&b * &b) % p;
        }
        m = i;
        c = (&b * &b) % p;
        t = (&t * &c) % p;
        r = (&r * &b) % p;
    }
    Some(r)
}

/// All square roots of `a` modulo `p^k` (`p` prime, `k ≥ 1`), sorted.
fn sqrt_mod_prime_power_all(a: &BigInt, p: &BigInt, k: u32) -> Vec<BigInt> {
    let pk = p.pow(k);
    let a = a.mod_floor(&pk);
    // Brute force for tiny moduli keeps the casework simple and exact.
    if pk <= BigInt::from(4096) {
        let m = pk.to_u64().unwrap_or(0);
        let av = a.to_u64().unwrap_or(0);
        return (0..m)
            .filter(|&x| (x * x) % m == av)
            .map(BigInt::from)
            .collect();
    }
    if a.is_zero() {
        // x ≡ 0 mod p^⌈k/2⌉
        let h = k.div_ceil(2);
        let step = p.pow(h);
        let count = p.pow(k - h);
        let Some(count) = count.to_u64() else {
            return vec![];
        };
        let mut out = Vec::with_capacity(count as usize);
        let mut x = BigInt::zero();
        for _ in 0..count {
            out.push(x.clone());
            x += &step;
        }
        return out;
    }
    // Strip p-power from a: a = p^t · b with p ∤ b.
    let mut t = 0u32;
    let mut b = a.clone();
    while (&b % p).is_zero() {
        b /= p;
        t += 1;
    }
    if t % 2 == 1 {
        return vec![];
    }
    if t > 0 {
        // x = p^{t/2} · y with y² ≡ b (mod p^{k−t}), then all lifts.
        let half = t / 2;
        let inner = sqrt_mod_prime_power_all(&b, p, k - t);
        let scale = p.pow(half);
        let period = p.pow(k - half);
        let Some(reps) = p.pow(half).to_u64() else {
            return vec![];
        };
        let mut out = Vec::new();
        for y in inner {
            let base = (&scale * &y).mod_floor(&pk);
            let mut x = base;
            for _ in 0..reps {
                out.push(x.clone());
                x = (&x + &period).mod_floor(&pk);
            }
        }
        out.sort();
        out.dedup();
        return out;
    }
    // p ∤ a.
    if *p == BigInt::from(2) {
        // k ≥ 3 here (2^k > 4096 ⇒ k ≥ 13).
        if a.mod_floor(&BigInt::from(8)) != BigInt::one() {
            return vec![];
        }
        // Lift one root from mod 8 (r = 1) to mod 2^k: if r² ≡ a mod 2^{j+1}
        // keep r, else r + 2^{j−1} works.  The four roots mod 2^k are then
        // ±r and ±r + 2^{k−1}.
        let mut r = BigInt::one();
        for j in 3..k {
            let modulus_next = BigInt::one() << (j + 1) as usize;
            if !((&r * &r - &a).mod_floor(&modulus_next)).is_zero() {
                r += BigInt::one() << (j - 1) as usize;
            }
        }
        let half = BigInt::one() << (k - 1) as usize;
        let mut roots = vec![
            r.mod_floor(&pk),
            (-&r).mod_floor(&pk),
            (&r + &half).mod_floor(&pk),
            (-&r + &half).mod_floor(&pk),
        ];
        roots.sort();
        roots.dedup();
        return roots;
    }
    // Odd p, p ∤ a: Tonelli–Shanks then Hensel lifting.
    let r0 = match sqrt_mod_prime(&a, p) {
        Some(r) => r,
        None => return vec![],
    };
    let mut r = r0;
    let mut modulus = p.clone();
    for _ in 1..k {
        let next = &modulus * p;
        // r ← r − (r² − a) · (2r)^{-1}  (mod next)
        let two_r = (BigInt::from(2) * &r).mod_floor(&next);
        let (_, inv, _) = extended_gcd_big(&two_r, &next);
        let corr = ((&r * &r - &a) * inv).mod_floor(&next);
        r = (&r - corr).mod_floor(&next);
        modulus = next;
    }
    let mut out = vec![r.clone(), (-&r).mod_floor(&pk)];
    out.sort();
    out.dedup();
    out
}

/// All solutions `x ∈ [0, n)` of `x² ≡ a (mod n)`, sorted.
///
/// Works for any modulus `n ≥ 1` by factoring `n`, solving modulo each
/// prime power (Tonelli–Shanks plus Hensel lifting; explicit casework for
/// powers of two and for `a` divisible by `p`), and combining with the
/// Chinese Remainder Theorem.  Returns an empty vector when there is no
/// solution.  Note that when `a ≡ 0` modulo a high prime power the number
/// of roots grows like `p^{k/2}`; all of them are returned.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::sqrt_mod_all;
/// use num_bigint::BigInt;
///
/// let b = |v: i64| BigInt::from(v);
/// assert_eq!(sqrt_mod_all(4, 15), vec![b(2), b(7), b(8), b(13)]);
/// assert_eq!(sqrt_mod_all(2, 7), vec![b(3), b(4)]);
/// assert!(sqrt_mod_all(3, 7).is_empty());
/// assert_eq!(sqrt_mod_all(1, 8), vec![b(1), b(3), b(5), b(7)]);
/// ```
pub fn sqrt_mod_all(a: impl Into<BigInt>, n: impl Into<BigInt>) -> Vec<BigInt> {
    let a: BigInt = a.into();
    let n: BigInt = n.into();
    if n < BigInt::one() {
        return vec![];
    }
    if n.is_one() {
        return vec![BigInt::zero()];
    }
    let mut combined: Vec<(BigInt, BigInt)> = vec![(BigInt::zero(), BigInt::one())]; // (residue, modulus)
    for (p, k) in factorint(n.clone()) {
        let pk = p.pow(k);
        let roots = sqrt_mod_prime_power_all(&a, &p, k);
        if roots.is_empty() {
            return vec![];
        }
        let mut next = Vec::with_capacity(combined.len() * roots.len());
        for (res, m) in &combined {
            for r in &roots {
                if let Some(x) = crt(&[res.clone(), r.clone()], &[m.clone(), pk.clone()]) {
                    next.push((x, m * &pk));
                }
            }
        }
        combined = next;
    }
    let mut out: Vec<BigInt> = combined.into_iter().map(|(x, _)| x).collect();
    out.sort();
    out.dedup();
    out
}

/// A square root of `a` modulo `n`, if one exists: the smallest
/// non-negative solution of `x² ≡ a (mod n)`.
///
/// For prime `n` this is Tonelli–Shanks; for composite `n` the congruence
/// is solved modulo each prime power and combined by CRT (see
/// [`sqrt_mod_all`]).
///
/// # Examples
///
/// ```
/// use symplex::ntheory::sqrt_mod;
/// use num_bigint::BigInt;
///
/// assert_eq!(sqrt_mod(2, 7), Some(BigInt::from(3)));      // 3² = 9 ≡ 2
/// assert_eq!(sqrt_mod(3, 7), None);
/// assert_eq!(sqrt_mod(10, 13), Some(BigInt::from(6)));    // 6² = 36 ≡ 10
/// assert_eq!(sqrt_mod(4, 15), Some(BigInt::from(2)));
/// ```
pub fn sqrt_mod(a: impl Into<BigInt>, n: impl Into<BigInt>) -> Option<BigInt> {
    let a: BigInt = a.into();
    let n: BigInt = n.into();
    if n < BigInt::one() {
        return None;
    }
    if n.is_odd() && isprime_big_internal(&n) {
        let r = sqrt_mod_prime(&a, &n)?;
        let other = (-&r).mod_floor(&n);
        return Some(r.min(other));
    }
    sqrt_mod_all(a, n).into_iter().next()
}

// ═══════════════════════════════════════════════════════════════════════════
// Multiplicative order, primitive roots, discrete logarithms
// ═══════════════════════════════════════════════════════════════════════════

/// Multiplicative order of `a` modulo `n`: the smallest `k ≥ 1` with
/// `aᵏ ≡ 1 (mod n)`.  Returns `None` if `n < 1` or `gcd(a, n) ≠ 1`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::n_order;
/// use num_bigint::BigInt;
///
/// assert_eq!(n_order(2, 7), Some(BigInt::from(3)));   // 2³ = 8 ≡ 1
/// assert_eq!(n_order(3, 7), Some(BigInt::from(6)));   // 3 is a primitive root
/// assert_eq!(n_order(2, 8), None);                    // not coprime
/// ```
pub fn n_order(a: impl Into<BigInt>, n: impl Into<BigInt>) -> Option<BigInt> {
    let a: BigInt = a.into();
    let n: BigInt = n.into();
    if n < BigInt::one() {
        return None;
    }
    if n.is_one() {
        return Some(BigInt::one());
    }
    let a = a.mod_floor(&n);
    if !a.gcd(&n).is_one() {
        return None;
    }
    let phi = totient(n.clone());
    Some(order_from_group_order(&a, &n, &phi))
}

/// Alias for [`n_order`].
pub fn multiplicative_order(a: impl Into<BigInt>, n: impl Into<BigInt>) -> Option<BigInt> {
    n_order(a, n)
}

/// Given that `a^m ≡ 1 (mod n)`, compute the exact order of `a` by
/// stripping prime factors from `m`.
fn order_from_group_order(a: &BigInt, n: &BigInt, m: &BigInt) -> BigInt {
    let mut order = m.clone();
    for (q, _) in factorint(m.clone()) {
        while (&order % &q).is_zero() && a.modpow(&(&order / &q), n).is_one() {
            order /= &q;
        }
    }
    order
}

/// Does the multiplicative group `(ℤ/nℤ)×` have a generator?  True for
/// `n ∈ {1, 2, 4, pᵏ, 2pᵏ}` with `p` an odd prime.
fn has_primitive_root(n: &BigInt) -> bool {
    if *n <= BigInt::from(4) {
        return n.is_positive();
    }
    let mut m = n.clone();
    if m.is_even() {
        m /= 2;
        if m.is_even() {
            return false;
        }
    }
    let f = factorint(m);
    f.len() == 1
}

/// Is `g` a primitive root modulo `n` (a generator of `(ℤ/nℤ)×`)?
///
/// # Examples
///
/// ```
/// use symplex::ntheory::is_primitive_root;
/// assert!(is_primitive_root(3, 7));
/// assert!(!is_primitive_root(2, 7));
/// assert!(is_primitive_root(2, 9));
/// assert!(!is_primitive_root(3, 8));  // (ℤ/8ℤ)× is not cyclic
/// ```
pub fn is_primitive_root(g: impl Into<BigInt>, n: impl Into<BigInt>) -> bool {
    let g: BigInt = g.into();
    let n: BigInt = n.into();
    if n < BigInt::one() || !has_primitive_root(&n) {
        return false;
    }
    if n.is_one() {
        return true;
    }
    let g = g.mod_floor(&n);
    if !g.gcd(&n).is_one() {
        return false;
    }
    let phi = totient(n.clone());
    factorint(phi.clone())
        .iter()
        .all(|(q, _)| !g.modpow(&(&phi / q), &n).is_one())
}

/// Smallest primitive root modulo `n`, if the group `(ℤ/nℤ)×` is cyclic
/// (`n ∈ {1, 2, 4, pᵏ, 2pᵏ}`); `None` otherwise.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::primitive_root;
/// use num_bigint::BigInt;
///
/// assert_eq!(primitive_root(7), Some(BigInt::from(3)));
/// assert_eq!(primitive_root(2), Some(BigInt::from(1)));
/// assert_eq!(primitive_root(8), None);
/// assert_eq!(primitive_root(191), Some(BigInt::from(19)));
/// ```
pub fn primitive_root(n: impl Into<BigInt>) -> Option<BigInt> {
    let n: BigInt = n.into();
    if n < BigInt::one() || !has_primitive_root(&n) {
        return None;
    }
    if n.is_one() {
        return Some(BigInt::zero());
    }
    if n == BigInt::from(2) {
        return Some(BigInt::one());
    }
    let phi = totient(n.clone());
    let phi_factors = factorint(phi.clone());
    let mut g = BigInt::from(2);
    while g < n {
        if g.gcd(&n).is_one()
            && phi_factors
                .iter()
                .all(|(q, _)| !g.modpow(&(&phi / q), &n).is_one())
        {
            return Some(g);
        }
        g += 1;
    }
    None
}

/// Discrete logarithm: the smallest `x ≥ 0` with `aˣ ≡ b (mod n)`.
///
/// Uses Pohlig–Hellman over the factorization of the order of `a`, with
/// baby-step giant-step inside each prime-power subgroup.  Requires
/// `gcd(a, n) = 1`; for non-coprime `a` a brute-force search is used when
/// `n ≤ 10⁶`, otherwise `None` is returned.  Returns `None` when no
/// solution exists.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::discrete_log;
/// use num_bigint::BigInt;
///
/// // 3^x ≡ 13 (mod 17) → x = 4  (3^4 = 81 = 4·17 + 13)
/// assert_eq!(discrete_log(3, 13, 17), Some(BigInt::from(4)));
/// // 2 has order 3 mod 7, so 2^x ≡ 3 is impossible
/// assert_eq!(discrete_log(2, 3, 7), None);
/// // Large prime modulus: round-trip through mod_pow
/// use symplex::ntheory::mod_pow;
/// let p = 1_000_003;
/// let b = mod_pow(5, 12345, p);
/// let x = discrete_log(5, b.clone(), p).unwrap();
/// assert_eq!(mod_pow(5, x, p), b);
/// ```
pub fn discrete_log(
    a: impl Into<BigInt>,
    b: impl Into<BigInt>,
    n: impl Into<BigInt>,
) -> Option<BigInt> {
    let n: BigInt = n.into();
    if n < BigInt::one() {
        return None;
    }
    let a: BigInt = a.into();
    let b: BigInt = b.into();
    let a = a.mod_floor(&n);
    let b = b.mod_floor(&n);
    if n.is_one() {
        return Some(BigInt::zero());
    }
    if b.is_one() {
        return Some(BigInt::zero());
    }
    if !a.gcd(&n).is_one() {
        // Brute force for small moduli.
        if let Some(nu) = n.to_u64()
            && nu <= 1_000_000
        {
            let mut x = BigInt::one();
            let mut cur = a.clone();
            while x <= n {
                if cur == b {
                    return Some(x);
                }
                cur = (&cur * &a) % &n;
                x += 1;
            }
        }
        return None;
    }
    let order = n_order(a.clone(), n.clone())?;
    // b must lie in the subgroup generated by a.
    if !b.modpow(&order, &n).is_one() {
        return None;
    }
    // Pohlig–Hellman.
    let mut residues: Vec<BigInt> = Vec::new();
    let mut moduli: Vec<BigInt> = Vec::new();
    for (q, e) in factorint(order.clone()) {
        let qe = q.pow(e);
        let cof = &order / &qe;
        let a_sub = a.modpow(&cof, &n); // order q^e
        let b_sub = b.modpow(&cof, &n);
        // Solve a_sub^x = b_sub in the q^e-subgroup digit by digit.
        let gamma = a_sub.modpow(&q.pow(e - 1), &n); // element of order q
        let a_inv = mod_inverse(a_sub.clone(), n.clone())?;
        let mut x = BigInt::zero();
        let mut qk = BigInt::one(); // q^k
        for k in 0..e {
            // h = (b_sub · a_sub^{−x})^{q^{e−1−k}}
            let h = (&b_sub * a_inv.modpow(&x, &n)).mod_floor(&n);
            let h = h.modpow(&q.pow(e - 1 - k), &n);
            let d = bsgs(&gamma, &h, &n, &q)?;
            x += &d * &qk;
            qk *= &q;
        }
        residues.push(x);
        moduli.push(qe);
    }
    let x = crt(&residues, &moduli)?;
    // Verify (also normalizes to the smallest solution because crt returns
    // the least non-negative residue mod the order).
    if a.modpow(&x, &n) == b { Some(x) } else { None }
}

/// Baby-step giant-step: smallest `x ∈ [0, order)` with `gˣ ≡ h (mod n)`,
/// where `g` has order `order`.
fn bsgs(g: &BigInt, h: &BigInt, n: &BigInt, order: &BigInt) -> Option<BigInt> {
    let m = order.sqrt() + BigInt::one();
    let m_u = m.to_u64()?;
    if m_u > 50_000_000 {
        return None; // would need too much memory
    }
    let mut table: FxHashMap<BigInt, u64> = FxHashMap::default();
    let mut cur = BigInt::one();
    for j in 0..m_u {
        table.entry(cur.clone()).or_insert(j);
        cur = (&cur * g) % n;
    }
    // giant step factor g^{-m}
    let g_inv = mod_inverse(g.clone(), n.clone())?;
    let factor = g_inv.modpow(&m, n);
    let mut gamma = h.mod_floor(n);
    for i in 0..=m_u {
        if let Some(&j) = table.get(&gamma) {
            let x = BigInt::from(i) * &m + BigInt::from(j);
            if &x < order {
                return Some(x);
            }
        }
        gamma = (&gamma * &factor) % n;
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Digits and continued fractions
// ═══════════════════════════════════════════════════════════════════════════

/// Digits of `|n|` in the given base, most significant first.
/// `digits(0, b)` is `[0]`.
///
/// # Errors
///
/// `InvalidArgument` if `base < 2`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::digits;
/// assert_eq!(digits(1234, 10).unwrap(), vec![1, 2, 3, 4]);
/// assert_eq!(digits(10, 2).unwrap(), vec![1, 0, 1, 0]);
/// assert_eq!(digits(-255, 16).unwrap(), vec![15, 15]);
/// assert!(digits(5, 1).is_err());
/// ```
pub fn digits(n: impl Into<BigInt>, base: u32) -> Result<Vec<u32>, SymplexError> {
    if base < 2 {
        return Err(SymplexError::InvalidArgument {
            operation: "digits",
            reason: format!("base must be ≥ 2, got {base}"),
        });
    }
    let n: BigInt = n.into();
    let mut m = n.abs();
    if m.is_zero() {
        return Ok(vec![0]);
    }
    let b = BigInt::from(base);
    let mut out = Vec::new();
    while !m.is_zero() {
        let (q, r) = m.div_rem(&b);
        out.push(r.to_u32().unwrap_or(0));
        m = q;
    }
    out.reverse();
    Ok(out)
}

/// Is `|n|` a palindrome in the given base?  Bases below 2 return `false`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::is_palindromic;
/// assert!(is_palindromic(12321, 10));
/// assert!(!is_palindromic(12345, 10));
/// assert!(is_palindromic(9, 2));  // 1001
/// ```
pub fn is_palindromic(n: impl Into<BigInt>, base: u32) -> bool {
    match digits(n, base) {
        Ok(d) => d.iter().eq(d.iter().rev()),
        Err(_) => false,
    }
}

/// Simple continued fraction `[a₀; a₁, a₂, …]` of a rational number.
///
/// Terminates (rationals have finite expansions); the last term is `> 1`
/// unless the number is an integer, which makes the representation
/// canonical.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::continued_fraction;
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// let r = Ratio::new(BigInt::from(415), BigInt::from(93));
/// let cf: Vec<i64> = continued_fraction(&r).iter().map(|t| t.try_into().unwrap()).collect();
/// assert_eq!(cf, vec![4, 2, 6, 7]);
/// let neg = Ratio::new(BigInt::from(-7), BigInt::from(3));
/// let cf: Vec<i64> = continued_fraction(&neg).iter().map(|t| t.try_into().unwrap()).collect();
/// assert_eq!(cf, vec![-3, 1, 2]);
/// ```
pub fn continued_fraction(r: &Q) -> Vec<BigInt> {
    let mut out = Vec::new();
    let mut num = r.numer().clone();
    let mut den = r.denom().clone();
    while !den.is_zero() {
        let a = num.div_floor(&den);
        let rem = &num - &a * &den;
        out.push(a);
        num = den;
        den = rem;
    }
    out
}

/// A periodic continued fraction
/// `[pre₀; pre₁, …, (period₀, …, periodₖ) repeating]` — the expansion of a
/// quadratic irrational.
///
/// Produced by [`continued_fraction_periodic`] and consumed by
/// [`continued_fraction_reduce_periodic`] /
/// [`continued_fraction_reduce_periodic_ex`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PeriodicContinuedFraction {
    /// The non-repeating head `[a₀; a₁, …]` (for `√d` this is `[⌊√d⌋]`).
    pub pre_period: Vec<BigInt>,
    /// The repeating block; empty when the value is rational.
    pub period: Vec<BigInt>,
}

/// Periodic continued fraction of `√d` for a non-negative integer `d`:
/// returns `pre_period` and `period` so that
/// `√d = [a₀; (a₁, …, aₖ) repeating]`.
///
/// Perfect squares return `pre_period = [√d]` with an empty `period`.
/// Negative `d` returns `None`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::continued_fraction_periodic;
/// use num_bigint::BigInt;
///
/// let to_i = |v: Vec<BigInt>| -> Vec<i64> { v.iter().map(|t| t.try_into().unwrap()).collect() };
/// let cf = continued_fraction_periodic(2).unwrap();
/// assert_eq!((to_i(cf.pre_period), to_i(cf.period)), (vec![1], vec![2]));          // √2 = [1; 2̄]
/// let cf = continued_fraction_periodic(7).unwrap();
/// assert_eq!((to_i(cf.pre_period), to_i(cf.period)), (vec![2], vec![1, 1, 1, 4])); // √7 = [2; 1,1,1,4]
/// let cf = continued_fraction_periodic(9).unwrap();
/// assert_eq!((to_i(cf.pre_period), to_i(cf.period)), (vec![3], vec![]));
/// ```
pub fn continued_fraction_periodic(d: impl Into<BigInt>) -> Option<PeriodicContinuedFraction> {
    let d: BigInt = d.into();
    if d.is_negative() {
        return None;
    }
    let a0 = d.sqrt();
    if &a0 * &a0 == d {
        return Some(PeriodicContinuedFraction {
            pre_period: vec![a0],
            period: vec![],
        });
    }
    // Standard algorithm: m₀ = 0, d₀ = 1, a₀ = ⌊√d⌋;
    // mₖ₊₁ = dₖ aₖ − mₖ ; dₖ₊₁ = (d − mₖ₊₁²)/dₖ ; aₖ₊₁ = ⌊(a₀ + mₖ₊₁)/dₖ₊₁⌋
    // The period ends when aₖ = 2a₀.
    let mut m = BigInt::zero();
    let mut dd = BigInt::one();
    let mut a = a0.clone();
    let two_a0 = &a0 * 2;
    let mut period = Vec::new();
    loop {
        m = &dd * &a - &m;
        dd = (&d - &m * &m) / &dd;
        a = (&a0 + &m) / &dd;
        period.push(a.clone());
        if a == two_a0 {
            break;
        }
    }
    Some(PeriodicContinuedFraction {
        pre_period: vec![a0],
        period,
    })
}

/// Convergents `hₙ/kₙ` of a continued fraction `[a₀; a₁, a₂, …]`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::continued_fraction_convergents;
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// let cf: Vec<BigInt> = [3, 7, 15, 1].iter().map(|&t| BigInt::from(t)).collect();
/// let conv = continued_fraction_convergents(&cf);
/// let last = &conv[3];
/// assert_eq!(*last.numer(), BigInt::from(355));
/// assert_eq!(*last.denom(), BigInt::from(113));
/// ```
pub fn continued_fraction_convergents(terms: &[BigInt]) -> Vec<Q> {
    let mut out = Vec::with_capacity(terms.len());
    let (mut h_prev, mut h) = (BigInt::zero(), BigInt::one()); // h_{-2}, h_{-1}
    let (mut k_prev, mut k) = (BigInt::one(), BigInt::zero()); // k_{-2}, k_{-1}
    for a in terms {
        let h_next = a * &h + &h_prev;
        let k_next = a * &k + &k_prev;
        h_prev = std::mem::replace(&mut h, h_next);
        k_prev = std::mem::replace(&mut k, k_next);
        if k.is_zero() {
            continue;
        }
        out.push(Ratio::new(h.clone(), k.clone()));
    }
    out
}

/// Greedy (Fibonacci–Sylvester) Egyptian fraction: denominators
/// `d₁ < d₂ < …` of distinct unit fractions summing to `r`.
///
/// Requires `0 < r ≤ 1`; returns `None` otherwise.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::egyptian_fraction;
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// let r = Ratio::new(BigInt::from(4), BigInt::from(13));
/// let d: Vec<i64> = egyptian_fraction(&r).unwrap().iter().map(|t| t.try_into().unwrap()).collect();
/// assert_eq!(d, vec![4, 18, 468]);   // 4/13 = 1/4 + 1/18 + 1/468
/// assert!(egyptian_fraction(&Ratio::from_integer(BigInt::from(2))).is_none());
/// ```
pub fn egyptian_fraction(r: &Q) -> Option<Vec<BigInt>> {
    if !r.is_positive() || *r > Ratio::one() {
        return None;
    }
    let mut out = Vec::new();
    let mut rem = r.clone();
    while !rem.is_zero() {
        // smallest d with 1/d ≤ rem  ⇔  d = ⌈1/rem⌉
        let d = rem.denom().div_ceil(rem.numer());
        out.push(d.clone());
        rem -= Ratio::new(BigInt::one(), d);
    }
    Some(out)
}

// ═══════════════════════════════════════════════════════════════════════════
// Classical sequences
// ═══════════════════════════════════════════════════════════════════════════

/// Fibonacci number `Fₙ` (`F₀ = 0, F₁ = 1`), extended to negative indices
/// by `F₋ₙ = (−1)ⁿ⁺¹ Fₙ`.  Fast doubling, `O(log n)` big-integer
/// multiplications.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::fibonacci;
/// use num_bigint::BigInt;
///
/// assert_eq!(fibonacci(10), BigInt::from(55));
/// assert_eq!(fibonacci(-8), BigInt::from(-21));
/// assert_eq!(fibonacci(100).to_string(), "354224848179261915075");
/// ```
pub fn fibonacci(n: impl Into<BigInt>) -> BigInt {
    let n: BigInt = n.into();
    let neg = n.is_negative();
    let m = n.abs().to_u64().unwrap_or(u64::MAX);
    let (f, _) = fib_pair(m);
    if neg && m.is_multiple_of(2) { -f } else { f }
}

/// `(F_n, F_{n+1})` by fast doubling (iterative over the bits of `n`).
fn fib_pair(n: u64) -> (BigInt, BigInt) {
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

/// Lucas number `Lₙ` (`L₀ = 2, L₁ = 1`), extended to negative indices by
/// `L₋ₙ = (−1)ⁿ Lₙ`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::lucas;
/// use num_bigint::BigInt;
///
/// assert_eq!(lucas(0), BigInt::from(2));
/// assert_eq!(lucas(10), BigInt::from(123));
/// assert_eq!(lucas(-3), BigInt::from(-4));
/// ```
pub fn lucas(n: impl Into<BigInt>) -> BigInt {
    let n: BigInt = n.into();
    let neg = n.is_negative();
    let m = n.abs().to_u64().unwrap_or(u64::MAX);
    // L_n = F_{n-1} + F_{n+1} = 2F_{n+1} − F_n
    let (f, f1) = fib_pair(m);
    let l: BigInt = &f1 * 2 - &f;
    if neg && m % 2 == 1 { -l } else { l }
}

/// Bernoulli number `Bₙ` as an exact rational, with the convention
/// `B₁ = −1/2`.  Returns `None` for negative `n` or `n` too large to fit
/// in `usize`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::bernoulli;
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// let r = |p: i64, q: i64| Ratio::new(BigInt::from(p), BigInt::from(q));
/// assert_eq!(bernoulli(0), Some(r(1, 1)));
/// assert_eq!(bernoulli(1), Some(r(-1, 2)));
/// assert_eq!(bernoulli(2), Some(r(1, 6)));
/// assert_eq!(bernoulli(12), Some(r(-691, 2730)));
/// assert_eq!(bernoulli(3), Some(r(0, 1)));
/// ```
pub fn bernoulli(n: impl Into<BigInt>) -> Option<Q> {
    let n: BigInt = n.into();
    if n.is_negative() {
        return None;
    }
    let n: usize = n.to_usize()?;
    Some(crate::base::bernoulli::bernoulli(n))
}

/// Euler (secant) number `Eₙ`: `E₀ = 1, E₂ = −1, E₄ = 5, E₆ = −61, …`;
/// odd indices are zero.  Returns `None` for negative `n`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::euler_number;
/// use num_bigint::BigInt;
///
/// assert_eq!(euler_number(0), Some(BigInt::from(1)));
/// assert_eq!(euler_number(4), Some(BigInt::from(5)));
/// assert_eq!(euler_number(6), Some(BigInt::from(-61)));
/// assert_eq!(euler_number(10), Some(BigInt::from(-50521)));
/// assert_eq!(euler_number(7), Some(BigInt::from(0)));
/// ```
pub fn euler_number(n: impl Into<BigInt>) -> Option<BigInt> {
    let n: BigInt = n.into();
    if n.is_negative() {
        return None;
    }
    let n = n.to_usize()?;
    if n % 2 == 1 {
        return Some(BigInt::zero());
    }
    // E_{2m} = −Σ_{k<m} C(2m, 2k) E_{2k}
    let m = n / 2;
    let mut e: Vec<BigInt> = Vec::with_capacity(m + 1);
    e.push(BigInt::one());
    for i in 1..=m {
        let mut sum = BigInt::zero();
        for (k, ek) in e.iter().enumerate().take(i) {
            sum += binomial((2 * i) as u64, (2 * k) as u64) * ek;
        }
        e.push(-sum);
    }
    Some(e[m].clone())
}

/// Harmonic number `Hₙ = 1 + 1/2 + … + 1/n` as an exact rational
/// (`H₀ = 0`).  Negative `n` returns `None`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::harmonic;
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// assert_eq!(harmonic(4), Some(Ratio::new(BigInt::from(25), BigInt::from(12))));
/// assert_eq!(harmonic(0), Some(Ratio::from_integer(BigInt::from(0))));
/// ```
pub fn harmonic(n: impl Into<BigInt>) -> Option<Q> {
    let n: BigInt = n.into();
    if n.is_negative() {
        return None;
    }
    let n = n.to_u64()?;
    let mut sum = Ratio::zero();
    for k in 1..=n {
        sum += Ratio::new(BigInt::one(), BigInt::from(k));
    }
    Some(sum)
}

// ═══════════════════════════════════════════════════════════════════════════
// Multi-argument gcd / lcm and denominator clearing
// ═══════════════════════════════════════════════════════════════════════════

/// Greatest common divisor of a list of integers (always `≥ 0`).
///
/// The empty list has gcd `0`, matching `gcd(0, 0) = 0`; a single element
/// gives its absolute value.  Stops early once the running gcd reaches `1`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::gcd_many;
/// use num_bigint::BigInt;
///
/// let v: Vec<BigInt> = [12, 18, 30].iter().map(|&n| BigInt::from(n)).collect();
/// assert_eq!(gcd_many(&v), BigInt::from(6));
/// assert_eq!(gcd_many(&[]), BigInt::from(0));
/// assert_eq!(gcd_many(&[BigInt::from(-8)]), BigInt::from(8));
/// ```
pub fn gcd_many(values: &[BigInt]) -> BigInt {
    let mut g = BigInt::zero();
    for v in values {
        g = num_integer::Integer::gcd(&g, v);
        if g.is_one() {
            break;
        }
    }
    g
}

/// Least common multiple of a list of integers (always `≥ 0`).
///
/// The empty list has lcm `1` (the empty product); any zero entry makes
/// the result `0`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::lcm_many;
/// use num_bigint::BigInt;
///
/// let v: Vec<BigInt> = [4, 6, 10].iter().map(|&n| BigInt::from(n)).collect();
/// assert_eq!(lcm_many(&v), BigInt::from(60));
/// assert_eq!(lcm_many(&[]), BigInt::from(1));
/// assert_eq!(lcm_many(&[BigInt::from(3), BigInt::from(0)]), BigInt::from(0));
/// ```
pub fn lcm_many(values: &[BigInt]) -> BigInt {
    let mut l = BigInt::one();
    for v in values {
        if v.is_zero() {
            return BigInt::zero();
        }
        l = num_integer::Integer::lcm(&l, v);
    }
    l
}

/// [`gcd_many`] for any integer type convertible to [`BigInt`]
/// (`i64`, `u64`, `i128`, `BigInt`, …).
///
/// # Examples
///
/// ```
/// use symplex::ntheory::igcd;
/// use num_bigint::BigInt;
///
/// assert_eq!(igcd(&[12i64, 18, 30]), BigInt::from(6));
/// assert_eq!(igcd(&[-4i64, 6]), BigInt::from(2));
/// assert_eq!(igcd::<i64>(&[]), BigInt::from(0));
/// ```
pub fn igcd<I: Into<BigInt> + Clone>(values: &[I]) -> BigInt {
    let big: Vec<BigInt> = values.iter().cloned().map(Into::into).collect();
    gcd_many(&big)
}

/// [`lcm_many`] for any integer type convertible to [`BigInt`].
///
/// # Examples
///
/// ```
/// use symplex::ntheory::ilcm;
/// use num_bigint::BigInt;
///
/// assert_eq!(ilcm(&[4i64, 6, 10]), BigInt::from(60));
/// assert_eq!(ilcm::<i64>(&[]), BigInt::from(1));
/// ```
pub fn ilcm<I: Into<BigInt> + Clone>(values: &[I]) -> BigInt {
    let big: Vec<BigInt> = values.iter().cloned().map(Into::into).collect();
    lcm_many(&big)
}

/// Least common multiple of the denominators of a list of rationals — the
/// factor that clears all denominators at once.
///
/// Multiplying every entry by the result yields integers.  The empty list
/// gives `1`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::rational_lcm_of_denominators;
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// let q = |n: i64, d: i64| Ratio::new(BigInt::from(n), BigInt::from(d));
/// let v = [q(1, 2), q(2, 3), q(5, 4)];
/// let l = rational_lcm_of_denominators(&v);
/// assert_eq!(l, BigInt::from(12));
/// for r in &v {
///     assert!((r * Ratio::from_integer(l.clone())).is_integer());
/// }
/// ```
pub fn rational_lcm_of_denominators(values: &[Q]) -> BigInt {
    let denoms: Vec<BigInt> = values.iter().map(|r| r.denom().clone()).collect();
    lcm_many(&denoms)
}

// ═══════════════════════════════════════════════════════════════════════════
// Higher power residues and polynomial congruences (0.9.1)
// ═══════════════════════════════════════════════════════════════════════════

/// Every solution `x ∈ [0, m)` of `xⁿ ≡ a (mod m)` (SymPy `nthroot_mod`).
///
/// Returns `None` when the congruence has no solution (and for `n < 1` or
/// `m < 1`); otherwise `Some(roots)` sorted ascending — every root when
/// `all_roots` is `true`, only the smallest one otherwise.
///
/// The modulus is factored.  Modulo each prime `p` the roots are found
/// with Johnston's generalised `q`-th root algorithm: the congruence is
/// first reduced to `x^d ≡ a′` with `d = gcd(n, p − 1)` (a Euclidean
/// reduction of `gcd(xⁿ − a, x^{p−1} − 1)`), then a primitive root and
/// discrete logarithms *inside the `q`-Sylow subgroups for the primes
/// `q | d`* produce one root, and multiplication by the `d`-th roots of
/// unity produces the rest.  `p` may therefore be large as long as the
/// prime factors of `gcd(n, p − 1)` are moderate (baby-step giant-step of
/// size `√q`).  Roots are lifted to `pᵏ` by Hensel's lemma (uniquely when
/// `p ∤ n·x`, exhaustively otherwise) and combined with the Chinese
/// Remainder Theorem.  `n = 2` defers to [`sqrt_mod_all`].
///
/// Root sets with more than `u64::MAX` elements cannot be enumerated and
/// such calls return `None`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::nthroot_mod;
/// use num_bigint::BigInt;
///
/// let b = |v: i64| BigInt::from(v);
/// // SymPy: nthroot_mod(11, 4, 19, True) == [8, 11]
/// assert_eq!(nthroot_mod(11, 4, 19, true), Some(vec![b(8), b(11)]));
/// assert_eq!(nthroot_mod(11, 4, 19, false), Some(vec![b(8)]));
/// // SymPy: nthroot_mod(68, 3, 109, True) == [23, 32, 54]
/// assert_eq!(nthroot_mod(68, 3, 109, true), Some(vec![b(23), b(32), b(54)]));
/// // 2 is not a cube modulo 7
/// assert_eq!(nthroot_mod(2, 3, 7, true), None);
/// // composite modulus: x⁴ ≡ 16 (mod 35)
/// assert_eq!(
///     nthroot_mod(16, 4, 35, true),
///     Some([2, 9, 12, 16, 19, 23, 26, 33].iter().map(|&v| b(v)).collect())
/// );
/// ```
pub fn nthroot_mod(
    a: impl Into<BigInt>,
    n: impl Into<BigInt>,
    m: impl Into<BigInt>,
    all_roots: bool,
) -> Option<Vec<BigInt>> {
    let n: BigInt = n.into();
    let m: BigInt = m.into();
    if n < BigInt::one() || m < BigInt::one() {
        return None;
    }
    let a: BigInt = a.into().mod_floor(&m);
    if m.is_one() {
        return Some(vec![BigInt::zero()]);
    }
    if n.is_one() {
        return Some(vec![a]);
    }
    let roots = if n == BigInt::from(2) {
        sqrt_mod_all(a, m)
    } else {
        let mut per_prime_power = Vec::new();
        for (p, k) in factorint(m.clone()) {
            let roots = nthroot_mod_prime_power(&a, &n, &p, k);
            if roots.is_empty() {
                return None;
            }
            per_prime_power.push((p.pow(k), roots));
        }
        crt_combine_roots(per_prime_power)
    };
    if roots.is_empty() {
        return None;
    }
    if all_roots {
        Some(roots)
    } else {
        roots.into_iter().next().map(|r| vec![r])
    }
}

/// Combine per-prime-power root lists (`(pᵏ, roots mod pᵏ)`) into the
/// sorted list of roots modulo the product, by CRT over every combination.
fn crt_combine_roots(per_prime_power: Vec<(BigInt, Vec<BigInt>)>) -> Vec<BigInt> {
    let mut combined: Vec<(BigInt, BigInt)> = vec![(BigInt::zero(), BigInt::one())];
    for (pk, roots) in per_prime_power {
        let mut next = Vec::with_capacity(combined.len() * roots.len());
        for (res, modulus) in &combined {
            for r in &roots {
                if let Some(x) = crt(&[res.clone(), r.clone()], &[modulus.clone(), pk.clone()]) {
                    next.push((x, modulus * &pk));
                }
            }
        }
        combined = next;
    }
    let mut out: Vec<BigInt> = combined.into_iter().map(|(x, _)| x).collect();
    out.sort();
    out.dedup();
    out
}

/// Does `xⁿ ≡ a (mod pᵏ)` have a solution?  (`p` prime, `k ≥ 1`, any
/// `n ≥ 1`.)  Generalised Euler criterion for odd `p`; the `a ≡ 1
/// (mod 2^{min(ν₂(n)+2, k)})` test for `p = 2`.
fn is_nthpow_residue_prime_power(a: &BigInt, n: &BigInt, p: &BigInt, k: u32) -> bool {
    let pk = p.pow(k);
    let mut a = a.mod_floor(&pk);
    if a.is_zero() {
        return true;
    }
    let mut k = k;
    if (&a % p).is_zero() {
        // a = p^μ · a′ with p ∤ a′: need n | μ, then solve mod p^{k−μ}.
        let mut mu = 0u32;
        while (&a % p).is_zero() {
            a /= p;
            mu += 1;
        }
        if !(BigInt::from(mu) % n).is_zero() {
            return false;
        }
        // μ < k because a ≢ 0 (mod pᵏ).
        k = k.saturating_sub(mu).max(1);
    }
    if *p != BigInt::from(2) {
        let phi = p.pow(k - 1) * (p - BigInt::one());
        let e = &phi / phi.gcd(n);
        return a.modpow(&e, &p.pow(k)).is_one();
    }
    if n.is_odd() {
        return true;
    }
    let v = n.trailing_zeros().unwrap_or(0);
    let c = (v + 2).min(u64::from(k));
    a.mod_floor(&(BigInt::one() << c)).is_one()
}

/// Sorted roots of `xⁿ ≡ a (mod pᵏ)` for prime `p` and `n ≥ 3`.
fn nthroot_mod_prime_power(a: &BigInt, n: &BigInt, p: &BigInt, k: u32) -> Vec<BigInt> {
    let pk = p.pow(k);
    let a = a.mod_floor(&pk);
    if !is_nthpow_residue_prime_power(&a, n, p, k) {
        return vec![];
    }
    let a_mod_p = a.mod_floor(p);
    let base_roots = if a_mod_p.is_zero() {
        vec![BigInt::zero()]
    } else {
        nthroot_mod_prime(&a_mod_p, n, p)
    };
    if k == 1 {
        return base_roots;
    }
    let n_minus_1 = n - BigInt::one();
    hensel_lift_all(
        base_roots,
        p,
        k,
        |x, modulus| (x.modpow(n, modulus) - &a).mod_floor(modulus),
        |x, modulus| (n * x.modpow(&n_minus_1, modulus)).mod_floor(modulus),
    )
}

/// Lift roots of `f` modulo `p` to roots modulo `pᵏ` (Hensel).
///
/// `f(x, m)` must return `f(x) mod m` and `df(x, m)` the derivative
/// `f′(x) mod m`.  When `f′(x) ≢ 0 (mod p)` the lift is unique; otherwise
/// either every one of the `p` candidates `x + v·pˢ` is a root modulo
/// `pˢ⁺¹` or none is.  Returns the sorted, deduplicated roots.
fn hensel_lift_all(
    base_roots: Vec<BigInt>,
    p: &BigInt,
    k: u32,
    f: impl Fn(&BigInt, &BigInt) -> BigInt,
    df: impl Fn(&BigInt, &BigInt) -> BigInt,
) -> Vec<BigInt> {
    let mut stack: Vec<(BigInt, u32)> = base_roots.into_iter().map(|x| (x, 1)).collect();
    let mut out = Vec::new();
    while let Some((x, s)) = stack.pop() {
        if s >= k {
            out.push(x);
            continue;
        }
        let ps = p.pow(s);
        let next = &ps * p;
        // f(x + v·pˢ) ≡ f(x) + v·pˢ·f′(x)  (mod pˢ⁺¹); f(x) ≡ 0 (mod pˢ) by invariant.
        let alpha = df(&x, p);
        let beta = (-(f(&x, &next) / &ps)).mod_floor(p);
        if alpha.is_zero() {
            if beta.is_zero() {
                let Some(count) = p.to_u64() else {
                    continue;
                };
                let mut y = x;
                for _ in 0..count {
                    stack.push((y.clone(), s + 1));
                    y += &ps;
                }
            }
        } else if let Some(inv) = mod_inverse(alpha, p.clone()) {
            let v = (inv * beta).mod_floor(p);
            stack.push((x + v * &ps, s + 1));
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Primes below this bound have their `n`-th roots enumerated directly:
/// `p` machine-word exponentiations beat the primitive-root machinery.
const NTHROOT_BRUTE_FORCE_LIMIT: u64 = 1 << 10;

/// Sorted roots of `xⁿ ≡ a (mod p)` for prime `p`, `p ∤ a`, `n ≥ 1`.
fn nthroot_mod_prime(a: &BigInt, n: &BigInt, p: &BigInt) -> Vec<BigInt> {
    if *p == BigInt::from(2) {
        // a ≡ 1 and xⁿ ≡ 1 (mod 2) ⇔ x ≡ 1.
        return vec![BigInt::one()];
    }
    let pm1 = p - BigInt::one();
    if let (Some(pu), Some(au)) = (p.to_u64(), a.to_u64())
        && pu < NTHROOT_BRUTE_FORCE_LIMIT
    {
        // x^n ≡ x^{n mod (p−1)} for p ∤ x (Fermat); n ≥ 1 so exponent 0 means p − 1.
        let mut e = n.mod_floor(&pm1).to_u64().unwrap_or(0);
        if e == 0 {
            e = pu - 1;
        }
        return (1..pu)
            .filter(|&x| mod_pow_u64(x, e, pu) == au)
            .map(BigInt::from)
            .collect();
    }
    let d = n.gcd(&pm1);
    if !a.modpow(&(&pm1 / &d), p).is_one() {
        return vec![];
    }
    // gcd(xⁿ − a, x^{p−1} − 1) by Euclid on the exponents: the pair
    // (x^{pa} − ca, x^{pb} − cb) becomes (x^{pb} − cb, x^{pa mod pb} − cb^{−q}·ca).
    let (mut pa, mut pb) = (n.clone(), pm1.clone());
    let (mut ca, mut cb) = (a.clone(), BigInt::one());
    if pa < pb {
        std::mem::swap(&mut pa, &mut pb);
        std::mem::swap(&mut ca, &mut cb);
    }
    while !pb.is_zero() {
        let (q, r) = pa.div_rem(&pb);
        let Some(cb_inv) = mod_inverse(cb.clone(), p.clone()) else {
            return vec![];
        };
        let c = (cb_inv.modpow(&q, p) * &ca).mod_floor(p);
        pa = pb;
        pb = r;
        ca = cb;
        cb = c;
    }
    // Now the roots are those of x^{pa} ≡ ca with pa = gcd(n, p − 1).
    if pa.is_one() {
        return vec![ca];
    }
    if pa == BigInt::from(2) {
        return sqrt_mod_all(ca, p.clone());
    }
    nthroot_mod_prime_divisor(&ca, &pa, p)
}

/// Sorted roots of `x^q ≡ s (mod p)` when `q | p − 1` and a root exists
/// (A. M. Johnston, "A generalized qth root algorithm", SODA 1999).
fn nthroot_mod_prime_divisor(s: &BigInt, q: &BigInt, p: &BigInt) -> Vec<BigInt> {
    let Some(g) = primitive_root(p.clone()) else {
        return vec![];
    };
    let pm1 = p - BigInt::one();
    let mut r = s.clone();
    for (qx, ex) in factorint(q.clone()) {
        // f = (p − 1) with every factor qx removed; z ≡ 0 (mod f), z ≡ −1 (mod qx).
        let mut f = &pm1 / qx.pow(ex);
        while (&f % &qx).is_zero() {
            f /= &qx;
        }
        let Some(neg_f_inv) = mod_inverse(-&f, qx.clone()) else {
            return vec![];
        };
        let z = &f * neg_f_inv;
        let x = (&z + BigInt::one()) / &qx;
        // t = log_h(r^f) inside the subgroup generated by h = g^{f·qx}.
        let h = g.modpow(&(&f * &qx), p);
        let Some(mut t) = discrete_log(h, r.modpow(&f, p), p.clone()) else {
            return vec![];
        };
        for _ in 0..ex {
            // (r^x · g^{−z·t})^{qx} = r
            let zt = (&z * &t).mod_floor(&pm1);
            let g_pow = g.modpow(&(&pm1 - &zt), p);
            r = (r.modpow(&x, p) * g_pow).mod_floor(p);
            t /= &qx;
        }
    }
    // All q roots: r · ζ^i with ζ = g^{(p−1)/q} a primitive q-th root of unity.
    let Some(count) = q.to_u64() else {
        return vec![];
    };
    let zeta = g.modpow(&(&pm1 / q), p);
    let mut out = Vec::with_capacity(count as usize);
    let mut cur = r;
    for _ in 0..count {
        out.push(cur.clone());
        cur = (&cur * &zeta).mod_floor(p);
    }
    out.sort();
    out.dedup();
    out
}

/// The quadratic residues modulo `n`: the sorted distinct values of
/// `x² mod n` for `0 ≤ x < n` (SymPy `quadratic_residues`).  Includes `0`.
///
/// Brute force over `x ≤ n/2`; `n` must fit in `u64` (the result has
/// `Θ(n)` entries anyway) and `n < 1` gives an empty vector.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::quadratic_residues;
/// use num_bigint::BigInt;
///
/// let to_i = |v: Vec<BigInt>| -> Vec<i64> { v.iter().map(|t| t.try_into().unwrap()).collect() };
/// assert_eq!(to_i(quadratic_residues(7)), vec![0, 1, 2, 4]);   // SymPy: [0, 1, 2, 4]
/// assert_eq!(to_i(quadratic_residues(8)), vec![0, 1, 4]);      // SymPy: [0, 1, 4]
/// assert_eq!(to_i(quadratic_residues(1)), vec![0]);
/// ```
pub fn quadratic_residues(n: impl Into<BigInt>) -> Vec<BigInt> {
    let n: BigInt = n.into();
    if n < BigInt::one() {
        return vec![];
    }
    let Some(nu) = n.to_u64() else {
        return vec![];
    };
    let mut squares: Vec<u64> = (0..=nu / 2).map(|x| mod_mul_u64(x, x, nu)).collect();
    squares.sort_unstable();
    squares.dedup();
    squares.into_iter().map(BigInt::from).collect()
}

/// Does `xⁿ ≡ a (mod m)` have a solution?  (SymPy `is_nthpow_residue`.)
///
/// `false` for `m < 1` or `n < 0`.  `n = 0` asks whether `a ≡ 1`;
/// everything is a residue modulo `1` (SymPy returns `False` for
/// `n = 0, m = 1`, which this function does not reproduce).  `n = 2` is
/// [`is_quad_residue`]; for `n ≥ 3` the criterion is applied to every
/// prime-power factor of `m` (generalised Euler criterion for odd primes;
/// `a ≡ 1 (mod 2^{min(ν₂(n)+2, k)})` for `2ᵏ`), so no root is ever
/// enumerated and `m` may be large.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::is_nthpow_residue;
///
/// assert!(!is_nthpow_residue(2, 3, 7));   // SymPy: False
/// assert!(is_nthpow_residue(2, 4, 7));    // SymPy: True   (2 ≡ 3⁴ mod 7)
/// assert!(is_nthpow_residue(16, 4, 17));  // SymPy: True
/// assert!(!is_nthpow_residue(2, 2, 15));  // SymPy: False
/// assert!(is_nthpow_residue(0, 2, 15));   // SymPy: True
/// assert!(is_nthpow_residue(17, 4, 32));  // SymPy: True
/// assert!(!is_nthpow_residue(9, 4, 16));  // SymPy: False
/// ```
pub fn is_nthpow_residue(a: impl Into<BigInt>, n: impl Into<BigInt>, m: impl Into<BigInt>) -> bool {
    let n: BigInt = n.into();
    let m: BigInt = m.into();
    if m < BigInt::one() || n.is_negative() {
        return false;
    }
    if m.is_one() {
        return true;
    }
    let a: BigInt = a.into().mod_floor(&m);
    if n.is_zero() {
        return a.is_one();
    }
    if a.is_zero() || n.is_one() {
        return true;
    }
    if n == BigInt::from(2) {
        return is_quad_residue(a, m);
    }
    factorint(m)
        .iter()
        .all(|(p, k)| is_nthpow_residue_prime_power(&a, &n, p, *k))
}

/// Roots in `[0, m)` of an integer polynomial modulo `m` (SymPy
/// `polynomial_congruence`).
///
/// `coeffs` are listed **highest degree first**, like SymPy's
/// `Poly.all_coeffs()` and [`Poly::all_coeffs`](crate::prelude::Poly::all_coeffs):
/// `x² − 1` is `&[1, 0, −1]`.  Returns the sorted roots; an empty vector
/// when there are none or `m < 1`.  Modulo `1` everything is a root and
/// `[0]` is returned.
///
/// Linear and quadratic congruences are solved directly (the latter
/// through [`sqrt_mod_all`] of the discriminant modulo `4am`), the monic
/// binomial `xⁿ − a` through [`nthroot_mod`]; both work for any modulus
/// that can be factored.  Any other polynomial is solved modulo each
/// prime factor `p` of `m` — brute force for `p ≤ 2¹⁶`, otherwise
/// `gcd(f, xᵖ − x)` followed by Cantor–Zassenhaus splitting, which needs
/// `p < 2⁶³` — lifted to the prime power by Hensel's lemma and combined
/// with the Chinese Remainder Theorem.  **Limit:** in that general case a
/// prime factor `p ≥ 2⁶³` of `m` is not supported and yields an empty
/// vector.  If the polynomial vanishes identically modulo some prime
/// `p | m`, all `p` residues are roots and are enumerated.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::polynomial_congruence;
/// use num_bigint::BigInt;
///
/// let c = |v: &[i64]| -> Vec<BigInt> { v.iter().map(|&t| BigInt::from(t)).collect() };
/// let to_i = |v: Vec<BigInt>| -> Vec<i64> { v.iter().map(|t| t.try_into().unwrap()).collect() };
///
/// // x² − 1 ≡ 0 (mod 8)                       SymPy: [1, 3, 5, 7]
/// assert_eq!(to_i(polynomial_congruence(&c(&[1, 0, -1]), 8)), vec![1, 3, 5, 7]);
/// // x⁶ − 2x⁵ − 35 ≡ 0 (mod 6125)             SymPy: [3257]
/// assert_eq!(to_i(polynomial_congruence(&c(&[1, -2, 0, 0, 0, 0, -35]), 6125)), vec![3257]);
/// // 6x⁵ + 10x⁴ + 5x³ + x² + x + 1 (mod 7)     SymPy: [2, 6]
/// assert_eq!(to_i(polynomial_congruence(&c(&[6, 10, 5, 1, 1, 1]), 7)), vec![2, 6]);
/// // x² + 1 has no root modulo 7
/// assert!(polynomial_congruence(&c(&[1, 0, 1]), 7).is_empty());
/// ```
pub fn polynomial_congruence(coeffs: &[BigInt], m: impl Into<BigInt>) -> Vec<BigInt> {
    let m: BigInt = m.into();
    if m < BigInt::one() {
        return vec![];
    }
    if m.is_one() {
        return vec![BigInt::zero()];
    }
    let reduced: Vec<BigInt> = coeffs.iter().map(|c| c.mod_floor(&m)).collect();
    let first_nonzero = reduced
        .iter()
        .position(|c| !c.is_zero())
        .unwrap_or(reduced.len());
    let c = &reduced[first_nonzero..];
    match c {
        [] => all_residues(&m),
        [_] => vec![],
        [a, b] => linear_congruence(a, &-b, &m),
        [a, b, cc] => quadratic_congruence(a, b, cc, &m),
        [lead, middle @ .., last] => {
            if lead.is_one() && middle.iter().all(Zero::is_zero) {
                let degree = BigInt::from(c.len() - 1);
                return nthroot_mod(-last, degree, m, true).unwrap_or_default();
            }
            polynomial_congruence_general(c, &m)
        }
    }
}

/// `[0, m)` as `BigInt`s (empty when `m` does not fit in `u64`).
fn all_residues(m: &BigInt) -> Vec<BigInt> {
    match m.to_u64() {
        Some(count) => (0..count).map(BigInt::from).collect(),
        None => vec![],
    }
}

/// Sorted solutions of `a·x ≡ b (mod m)`, `m ≥ 1`.
fn linear_congruence(a: &BigInt, b: &BigInt, m: &BigInt) -> Vec<BigInt> {
    let a = a.mod_floor(m);
    let b = b.mod_floor(m);
    if a.is_zero() {
        return if b.is_zero() { all_residues(m) } else { vec![] };
    }
    let (g, x, _) = extended_gcd_big(&a, m);
    if !(&b % &g).is_zero() {
        return vec![];
    }
    let step = m / &g;
    let x0 = (x * (&b / &g)).mod_floor(&step);
    let Some(count) = g.to_u64() else {
        return vec![];
    };
    (0..count).map(|t| &x0 + BigInt::from(t) * &step).collect()
}

/// Sorted solutions of `a·x² + b·x + c ≡ 0 (mod m)`, `m ≥ 2`, `a ≢ 0`.
///
/// Multiplying by `4a`: `(2ax + b)² ≡ b² − 4ac (mod 4am)`; every root `i`
/// of that congruence with `i ≡ b (mod 2a)` gives `x = (i − b)/(2a)`.
fn quadratic_congruence(a: &BigInt, b: &BigInt, c: &BigInt, m: &BigInt) -> Vec<BigInt> {
    let two_a = a * 2;
    let disc = b * b - &two_a * 2 * c;
    let modulus = &two_a * 2 * m;
    let mut out: Vec<BigInt> = sqrt_mod_all(disc, modulus)
        .into_iter()
        .filter_map(|i| {
            let (q, r) = (i - b).div_mod_floor(&two_a);
            r.is_zero().then(|| q.mod_floor(m))
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

/// General case of [`polynomial_congruence`]: roots modulo each prime,
/// Hensel lifting, CRT.  `c` is non-empty with a non-zero leading
/// coefficient (highest degree first).
fn polynomial_congruence_general(c: &[BigInt], m: &BigInt) -> Vec<BigInt> {
    let degree = c.len() - 1;
    let derivative: Vec<BigInt> = c[..degree]
        .iter()
        .enumerate()
        .map(|(i, ci)| ci * BigInt::from(degree - i))
        .collect();
    let mut per_prime_power = Vec::new();
    for (p, k) in factorint(m.clone()) {
        let base = poly_roots_mod_prime(c, &p);
        if base.is_empty() {
            return vec![];
        }
        let roots = if k == 1 {
            base
        } else {
            hensel_lift_all(
                base,
                &p,
                k,
                |x, modulus| poly_eval_mod(c, x, modulus),
                |x, modulus| poly_eval_mod(&derivative, x, modulus),
            )
        };
        if roots.is_empty() {
            return vec![];
        }
        per_prime_power.push((p.pow(k), roots));
    }
    crt_combine_roots(per_prime_power)
}

/// Horner evaluation of `c` (highest degree first) at `x` modulo `m`.
fn poly_eval_mod(c: &[BigInt], x: &BigInt, m: &BigInt) -> BigInt {
    let mut acc = BigInt::zero();
    for ci in c {
        acc = (acc * x + ci).mod_floor(m);
    }
    acc
}

/// Threshold below which roots modulo `p` are found by brute force.
const BRUTE_FORCE_ROOT_LIMIT: u64 = 1 << 16;

/// Sorted roots of `c` (highest degree first, non-zero leading
/// coefficient over ℤ) modulo the prime `p`.  Empty for `p ≥ 2⁶³`.
fn poly_roots_mod_prime(c: &[BigInt], p: &BigInt) -> Vec<BigInt> {
    let Some(pu) = p.to_u64() else {
        return vec![];
    };
    if pu >= 1 << 63 {
        return vec![];
    }
    let cu: Vec<u64> = c
        .iter()
        .map(|x| x.mod_floor(p).to_u64().unwrap_or(0))
        .collect();
    if pu <= BRUTE_FORCE_ROOT_LIMIT {
        return (0..pu)
            .filter(|&x| {
                cu.iter()
                    .fold(0u64, |acc, &ci| (mod_mul_u64(acc, x, pu) + ci) % pu)
                    == 0
            })
            .map(BigInt::from)
            .collect();
    }
    // Dense `𝔽ₚ[x]` (the value-level ring in `poly::modpoly`): `gcd(f, xᵖ − x)`
    // followed by Cantor–Zassenhaus splitting, all factors linear.
    let f = PolyIn::over_prime(pu, cu.iter().rev().copied().collect());
    if f.is_zero() {
        // Vanishes identically modulo p: every residue is a root.
        return (0..pu).map(BigInt::from).collect();
    }
    f.roots().into_iter().map(BigInt::from).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Arithmetic functions (0.9.1)
// ═══════════════════════════════════════════════════════════════════════════

/// The multiplicity of `p` in `n`: the largest `k` with `pᵏ | n`
/// (SymPy `multiplicity`).  Works on absolute values; returns `0` when
/// `|p| ≤ 1` or `n = 0` (SymPy raises there — the multiplicity of `0` is
/// infinite and `±1` divides everything).
///
/// # Examples
///
/// ```
/// use symplex::ntheory::multiplicity;
///
/// assert_eq!(multiplicity(2, 40), 3);        // SymPy: 3
/// assert_eq!(multiplicity(3, 81), 4);        // SymPy: 4
/// assert_eq!(multiplicity(6, 72), 2);        // SymPy: 2
/// assert_eq!(multiplicity(5, 7), 0);         // SymPy: 0
/// assert_eq!(multiplicity(10, 1_000_000), 6);
/// assert_eq!(multiplicity(2, 0), 0);         // undefined → 0
/// ```
pub fn multiplicity(p: impl Into<BigInt>, n: impl Into<BigInt>) -> u32 {
    let p: BigInt = p.into().abs();
    let mut n: BigInt = n.into().abs();
    if p <= BigInt::one() || n.is_zero() {
        return 0;
    }
    let mut k = 0u32;
    while (&n % &p).is_zero() {
        n /= &p;
        k += 1;
    }
    k
}

/// `ν(n)`: the number of *distinct* prime factors of `|n|` (SymPy
/// `primenu`).  `0` for `n ∈ {−1, 0, 1}`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::primenu;
///
/// assert_eq!(primenu(1), 0);      // SymPy: 0
/// assert_eq!(primenu(30), 3);     // SymPy: 3  (2·3·5)
/// assert_eq!(primenu(72), 2);     // SymPy: 2  (2³·3²)
/// assert_eq!(primenu(1024), 1);   // SymPy: 1
/// ```
pub fn primenu(n: impl Into<BigInt>) -> u32 {
    let n: BigInt = n.into();
    u32::try_from(factorint(n).len()).unwrap_or(u32::MAX)
}

/// `Ω(n)`: the number of prime factors of `|n|` counted with multiplicity
/// (SymPy `primeomega`).  `0` for `n ∈ {−1, 0, 1}`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::primeomega;
///
/// assert_eq!(primeomega(1), 0);       // SymPy: 0
/// assert_eq!(primeomega(30), 3);      // SymPy: 3
/// assert_eq!(primeomega(72), 5);      // SymPy: 5  (2³·3²)
/// assert_eq!(primeomega(1024), 10);   // SymPy: 10
/// ```
pub fn primeomega(n: impl Into<BigInt>) -> u32 {
    let n: BigInt = n.into();
    factorint(n).iter().map(|(_, e)| *e).sum()
}

/// Balanced product of a list of integers (empty product is `1`).
fn product_tree(mut values: Vec<BigInt>) -> BigInt {
    if values.is_empty() {
        return BigInt::one();
    }
    while values.len() > 1 {
        let mut next = Vec::with_capacity(values.len().div_ceil(2));
        let mut it = values.into_iter();
        while let Some(a) = it.next() {
            match it.next() {
                Some(b) => next.push(a * b),
                None => next.push(a),
            }
        }
        values = next;
    }
    values.into_iter().next().unwrap_or_else(BigInt::one)
}

/// The product of the first `n` primes, `pₙ#` (SymPy `primorial(n)`,
/// i.e. `nth=True`).  `primorial(0) = 1` (SymPy rejects `n < 1`).
///
/// Sieves up to the Rosser–Schoenfeld bound for `n ≤ 10⁷` and walks
/// [`nextprime`] beyond that (correct, but slow — the result has millions
/// of digits anyway).
///
/// # Examples
///
/// ```
/// use symplex::ntheory::primorial;
/// use num_bigint::BigInt;
///
/// assert_eq!(primorial(0), BigInt::from(1));
/// assert_eq!(primorial(1), BigInt::from(2));         // SymPy: 2
/// assert_eq!(primorial(5), BigInt::from(2310));      // SymPy: 2310 = 2·3·5·7·11
/// assert_eq!(primorial(10), BigInt::from(6469693230u64));   // SymPy: 6469693230
/// assert_eq!(primorial(20).to_string(), "557940830126698960967415390");
/// ```
pub fn primorial(n: u64) -> BigInt {
    if n == 0 {
        return BigInt::one();
    }
    if n <= 10_000_000 {
        let bound = if n < 6 {
            12
        } else {
            let nf = n as f64;
            (nf * (nf.ln() + nf.ln().ln())).ceil() as u64 + 10
        };
        let sieve = BitSieve::new(bound);
        let mut primes = Vec::with_capacity(n as usize);
        let mut k = 2u64;
        while k <= bound && (primes.len() as u64) < n {
            if sieve.is_prime(k) {
                primes.push(BigInt::from(k));
            }
            k += if k == 2 { 1 } else { 2 };
        }
        if primes.len() as u64 == n {
            return product_tree(primes);
        }
    }
    let mut primes = Vec::new();
    let mut p = BigInt::from(2);
    for _ in 0..n {
        primes.push(p.clone());
        p = nextprime(p);
    }
    product_tree(primes)
}

/// The product of all primes `≤ n` (SymPy `primorial(n, nth=False)`).
/// `1` for `n < 2`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::primorial_up_to;
/// use num_bigint::BigInt;
///
/// assert_eq!(primorial_up_to(1), BigInt::from(1));       // SymPy: 1
/// assert_eq!(primorial_up_to(5), BigInt::from(30));      // SymPy: 30
/// assert_eq!(primorial_up_to(10), BigInt::from(210));    // SymPy: 210
/// assert_eq!(primorial_up_to(100).to_string(), "2305567963945518424753102147331756070");
/// ```
pub fn primorial_up_to(n: impl Into<BigInt>) -> BigInt {
    let n: BigInt = n.into();
    if n < BigInt::from(2) {
        return BigInt::one();
    }
    product_tree(primerange(2, n + BigInt::one()))
}

/// Is `n` a Carmichael number (SymPy `is_carmichael`)?  Korselt's
/// criterion: `n` is composite, odd, square-free, and `(p − 1) | (n − 1)`
/// for every prime `p | n`.  The smallest is `561 = 3·11·17`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::is_carmichael;
///
/// assert!(is_carmichael(561));     // SymPy: True
/// assert!(is_carmichael(1105));    // SymPy: True
/// assert!(is_carmichael(41041));   // SymPy: True
/// assert!(!is_carmichael(7));      // prime
/// assert!(!is_carmichael(15));     // SymPy: False
/// assert!(!is_carmichael(1));
/// ```
pub fn is_carmichael(n: impl Into<BigInt>) -> bool {
    let n: BigInt = n.into();
    if n < BigInt::from(3) || n.is_even() {
        return false;
    }
    let factors = factorint(n.clone());
    if factors.len() < 2 {
        return false;
    }
    let n_minus_1 = &n - BigInt::one();
    factors
        .iter()
        .all(|(p, e)| *e == 1 && (&n_minus_1 % (p - BigInt::one())).is_zero())
}

/// Are `a` and `b` an amicable pair (SymPy `is_amicable`)?  That is,
/// `a ≠ b`, both positive, and each is the sum of the proper divisors of
/// the other: `σ(a) − a = b` and `σ(b) − b = a`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::is_amicable;
///
/// assert!(is_amicable(220, 284));     // SymPy: True
/// assert!(is_amicable(1184, 1210));   // SymPy: True
/// assert!(is_amicable(2620, 2924));   // SymPy: True
/// assert!(!is_amicable(220, 285));    // SymPy: False
/// assert!(!is_amicable(6, 6));        // perfect, not amicable (SymPy: False)
/// ```
pub fn is_amicable(a: impl Into<BigInt>, b: impl Into<BigInt>) -> bool {
    let a: BigInt = a.into();
    let b: BigInt = b.into();
    if !a.is_positive() || !b.is_positive() || a == b {
        return false;
    }
    let sum = &a + &b;
    divisor_sigma(a, 1) == sum && divisor_sigma(b, 1) == sum
}

/// Row `n` of Pascal's triangle, `[C(n,0), C(n,1), …, C(n,n)]` (SymPy
/// `binomial_coefficients_list`).
///
/// # Examples
///
/// ```
/// use symplex::ntheory::binomial_coefficients_list;
/// use num_bigint::BigInt;
///
/// let to_i = |v: Vec<BigInt>| -> Vec<i64> { v.iter().map(|t| t.try_into().unwrap()).collect() };
/// assert_eq!(to_i(binomial_coefficients_list(0)), vec![1]);
/// assert_eq!(to_i(binomial_coefficients_list(4)), vec![1, 4, 6, 4, 1]);   // SymPy: [1, 4, 6, 4, 1]
/// assert_eq!(to_i(binomial_coefficients_list(6)), vec![1, 6, 15, 20, 15, 6, 1]);
/// ```
pub fn binomial_coefficients_list(n: u32) -> Vec<BigInt> {
    let n = u64::from(n);
    let mut row = Vec::new();
    let mut c = BigInt::one();
    row.push(c.clone());
    for k in 0..n {
        c = c * BigInt::from(n - k) / BigInt::from(k + 1);
        row.push(c.clone());
    }
    row
}

/// The binomial coefficients of `(x + y)ⁿ` keyed by exponent pair (SymPy
/// `binomial_coefficients`, which returns the dictionary
/// `{(k₁, k₂): C(n, k₁)}` with `k₁ + k₂ = n`).  Returned as pairs sorted
/// by `k₁`.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::binomial_coefficients;
/// use num_bigint::BigInt;
///
/// // SymPy: binomial_coefficients(4) == {(0, 4): 1, (1, 3): 4, (2, 2): 6, (3, 1): 4, (4, 0): 1}
/// let c = binomial_coefficients(4);
/// assert_eq!(c.len(), 5);
/// assert_eq!(c[2], ((2, 2), BigInt::from(6)));
/// assert!(c.iter().all(|((k1, k2), _)| k1 + k2 == 4));
/// ```
pub fn binomial_coefficients(n: u32) -> Vec<((u32, u32), BigInt)> {
    binomial_coefficients_list(n)
        .into_iter()
        .enumerate()
        .map(|(k, c)| {
            let k1 = u32::try_from(k).unwrap_or(u32::MAX);
            ((k1, n - k1), c)
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Continued fraction reduction (0.9.1)
// ═══════════════════════════════════════════════════════════════════════════

/// The rational number with the finite simple continued fraction
/// `[a₀; a₁, …, aₙ]` (SymPy `continued_fraction_reduce` on a plain list).
///
/// Evaluated from the tail: `x ← aₙ`, then `x ← aᵢ + 1/x`.  Returns
/// `None` for the empty list and when an intermediate value is `0`
/// (`[1; 0]` would be `1 + 1/0`; SymPy returns `zoo` there).
///
/// # Examples
///
/// ```
/// use symplex::ntheory::continued_fraction_reduce;
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// let cf = |v: &[i64]| -> Vec<BigInt> { v.iter().map(|&t| BigInt::from(t)).collect() };
/// let q = |p: i64, d: i64| Ratio::new(BigInt::from(p), BigInt::from(d));
/// assert_eq!(continued_fraction_reduce(&cf(&[4, 2, 6, 7])), Some(q(415, 93)));   // SymPy: 415/93
/// assert_eq!(continued_fraction_reduce(&cf(&[3, 7, 15, 1])), Some(q(355, 113))); // SymPy: 355/113
/// assert_eq!(continued_fraction_reduce(&cf(&[-3, 1, 2])), Some(q(-7, 3)));       // SymPy: -7/3
/// assert_eq!(continued_fraction_reduce(&cf(&[1, 0])), None);
/// assert_eq!(continued_fraction_reduce(&[]), None);
/// ```
pub fn continued_fraction_reduce(terms: &[BigInt]) -> Option<Q> {
    let (last, init) = terms.split_last()?;
    let mut x = Ratio::from_integer(last.clone());
    for a in init.iter().rev() {
        if x.is_zero() {
            return None;
        }
        x = Ratio::from_integer(a.clone()) + x.recip();
    }
    Some(x)
}

/// The quadratic surd **`(p + √d) / q`** with integer `p`, `q` and `d`.
///
/// `d > 0` is not a perfect square and `q ≠ 0` **may be negative** — that
/// is how a negative radical coefficient is encoded: `(80 − √30)/52` is
/// `(−80 + √30)/(−52)`, i.e. `p = -80, q = -52, d = 30`.  The value is
/// reduced: no integer `g > 1` divides `p` and `q` with `g² | d`.
///
/// Produced by [`continued_fraction_reduce_periodic`];
/// [`continued_fraction_reduce_periodic_ex`] builds the same value as a
/// symbolic `Ex`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct QuadraticSurd {
    /// The integer part of the numerator `p`.
    pub p: BigInt,
    /// The denominator `q ≠ 0`; its sign carries the sign of the radical.
    pub q: BigInt,
    /// The radicand `d > 0`, not a perfect square.
    pub d: BigInt,
}

/// The quadratic irrational with the periodic continued fraction
/// `[pre₀; pre₁, …, (period₀, …, periodₖ) repeating]` (SymPy
/// `continued_fraction_reduce([a₀, …, [b₀, …]])`), returned as the
/// [`QuadraticSurd`] `{ p, q, d }` meaning **`(p + √d) / q`**.
///
/// `d > 0` is not a perfect square, `q ≠ 0` **may be negative** (that is
/// how a negative radical coefficient is encoded: `(80 − √30)/52` is
/// `(−80 + √30)/(−52)`, i.e. `p = -80, q = -52, d = 30`), and the surd is
/// reduced: no integer `g > 1` divides `p` and `q` with `g² | d`.
/// [`continued_fraction_reduce_periodic_ex`] builds the same value as a
/// symbolic `Ex`.
///
/// The purely periodic tail `y = [b₀; …, bₖ, y]` satisfies
/// `k·y² + (k′ − h)·y − h′ = 0` for its last two convergents `h/k`,
/// `h′/k′`; the pre-period is then applied as the Möbius map
/// `x = (P·y + P′)/(Q·y + Q′)` and the denominator rationalised.
///
/// Returns `None` when `period` is empty (use
/// [`continued_fraction_reduce`]) or contains a non-positive term.
///
/// # Examples
///
/// ```
/// use symplex::ntheory::{continued_fraction_reduce_periodic, QuadraticSurd};
/// use num_bigint::BigInt;
///
/// let cf = |v: &[i64]| -> Vec<BigInt> { v.iter().map(|&t| BigInt::from(t)).collect() };
/// let surd = |p: i64, q: i64, d: i64| {
///     Some(QuadraticSurd { p: BigInt::from(p), q: BigInt::from(q), d: BigInt::from(d) })
/// };
/// // √2 = [1; (2)]                       SymPy: sqrt(2)
/// assert_eq!(continued_fraction_reduce_periodic(&cf(&[1]), &cf(&[2])), surd(0, 1, 2));
/// // golden ratio [(1)]                  SymPy: (1 + sqrt(5))/2
/// assert_eq!(continued_fraction_reduce_periodic(&[], &cf(&[1])), surd(1, 2, 5));
/// // √7 = [2; (1, 1, 1, 4)]              SymPy: sqrt(7)
/// assert_eq!(continued_fraction_reduce_periodic(&cf(&[2]), &cf(&[1, 1, 1, 4])), surd(0, 1, 7));
/// // [1; 2, 3, (4, 5)]                   SymPy: (80 - sqrt(30))/52
/// assert_eq!(continued_fraction_reduce_periodic(&cf(&[1, 2, 3]), &cf(&[4, 5])), surd(-80, -52, 30));
/// assert_eq!(continued_fraction_reduce_periodic(&cf(&[1]), &[]), None);
/// ```
pub fn continued_fraction_reduce_periodic(
    pre: &[BigInt],
    period: &[BigInt],
) -> Option<QuadraticSurd> {
    if period.is_empty() || period.iter().any(|b| !b.is_positive()) {
        return None;
    }
    // Last two convergents h/k, h′/k′ of the period (h′/k′ = 1/0 for a
    // one-term period): y = (h·y + h′)/(k·y + k′).
    let (h, h_prev, k, k_prev) = convergent_pair(period);
    // k·y² + (k′ − h)·y − h′ = 0, positive root y = (s + √D)/t.
    let s = &h - &k_prev;
    let disc = &s * &s + BigInt::from(4) * &k * &h_prev;
    let t: BigInt = &k * 2;
    if is_square_big(&disc) || t.is_zero() {
        return None;
    }
    // x = (P·y + P′)/(Q·y + Q′) with y = (s + √D)/t:
    //   numerator   (P·s + P′·t) + P·√D = A + B√D
    //   denominator (Q·s + Q′·t) + Q·√D = C + E√D
    let (pp, pp_prev, qq, qq_prev) = convergent_pair(pre);
    let a = &pp * &s + &pp_prev * &t;
    let b = pp;
    let c = &qq * &s + &qq_prev * &t;
    let e = qq;
    // Rationalise: (A + B√D)(C − E√D) / (C² − E²D).
    let mut alpha = &a * &c - &b * &e * &disc;
    let mut beta = &b * &c - &a * &e;
    let mut gamma = &c * &c - &e * &e * &disc;
    if gamma.is_zero() || beta.is_zero() {
        return None;
    }
    // Fold β into the radical: (α + β√D)/γ = (±α + √(β²D))/(±γ).
    if beta.is_negative() {
        alpha = -alpha;
        gamma = -gamma;
        beta = -beta;
    }
    let mut d = &beta * &beta * &disc;
    // Reduce: strip every prime g with g | α, g | γ, g² | d.
    let g = alpha.gcd(&gamma);
    for (prime, _) in factorint(g) {
        let square = &prime * &prime;
        while (&alpha % &prime).is_zero() && (&gamma % &prime).is_zero() && (&d % &square).is_zero()
        {
            alpha /= &prime;
            gamma /= &prime;
            d /= &square;
        }
    }
    Some(QuadraticSurd {
        p: alpha,
        q: gamma,
        d,
    })
}

/// `(hₙ, hₙ₋₁, kₙ, kₙ₋₁)` — the last two convergent numerators and
/// denominators of `terms`, starting from `h₋₁/k₋₁ = 1/0`,
/// `h₋₂/k₋₂ = 0/1`.  For the empty list this is `(1, 0, 0, 1)`.
fn convergent_pair(terms: &[BigInt]) -> (BigInt, BigInt, BigInt, BigInt) {
    let (mut h_prev, mut h) = (BigInt::zero(), BigInt::one());
    let (mut k_prev, mut k) = (BigInt::one(), BigInt::zero());
    for a in terms {
        let h_next = a * &h + &h_prev;
        let k_next = a * &k + &k_prev;
        h_prev = std::mem::replace(&mut h, h_next);
        k_prev = std::mem::replace(&mut k, k_next);
    }
    (h, h_prev, k, k_prev)
}

/// [`continued_fraction_reduce_periodic`] as a symbolic expression
/// `(p + √d)/q` in `ctx`, which canonicalises it (`√8/2` becomes `√2`).
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::ntheory::continued_fraction_reduce_periodic_ex;
/// use num_bigint::BigInt;
///
/// let ctx = Context::new();
/// let cf = |v: &[i64]| -> Vec<BigInt> { v.iter().map(|&t| BigInt::from(t)).collect() };
/// let sqrt2 = continued_fraction_reduce_periodic_ex(&ctx, &cf(&[1]), &cf(&[2])).unwrap();
/// assert_eq!(sqrt2, ctx.int(2).sqrt());
/// let phi = continued_fraction_reduce_periodic_ex(&ctx, &[], &cf(&[1])).unwrap();
/// assert!((phi.eval_f64().unwrap() - 1.618033988749895).abs() < 1e-12);
/// ```
pub fn continued_fraction_reduce_periodic_ex(
    ctx: &Context,
    pre: &[BigInt],
    period: &[BigInt],
) -> Option<Ex> {
    let QuadraticSurd { p, q, d } = continued_fraction_reduce_periodic(pre, period)?;
    let radical = ctx.from_bigint(d).sqrt();
    Some((ctx.from_bigint(p) + radical) / ctx.from_bigint(q))
}

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

    // ═══════════════════════════════════════════════════════════════════
    // 0.2 additions
    // ═══════════════════════════════════════════════════════════════════

    fn big(s: &str) -> BigInt {
        BigInt::parse_bytes(s.as_bytes(), 10).unwrap()
    }

    fn product_of(factors: &[(BigInt, u32)]) -> BigInt {
        factors.iter().map(|(p, e)| p.pow(*e)).product()
    }

    // ── factorint: rho / ECM ──────────────────────────────────────────

    #[test]
    fn factorint_2_pow_64_plus_1() {
        let n = big("18446744073709551617");
        let f = factorint(n.clone());
        assert_eq!(f, vec![(bi(274177), 1), (big("67280421310721"), 1)]);
        assert_eq!(product_of(&f), n);
    }

    #[test]
    fn factorint_10_pow_18_plus_9_is_prime() {
        // 10^18 + 9 happens to be prime; factorint must recognise that
        // quickly rather than trial-dividing to 10^9.
        let n = 1_000_000_000_000_000_009i64;
        let f = factorint(n);
        assert_eq!(f, vec![(bi(n), 1)]);
        // And neighbours that are composite:
        let f = factorint(n + 2); // 10^18 + 11
        assert_eq!(product_of(&f), bi(n + 2));
        assert!(f.iter().all(|(p, _)| isprime(p.clone())));
    }

    #[test]
    fn factorint_semiprime_of_two_32_bit_primes() {
        let p = 4_294_967_291u64; // largest prime below 2^32
        let q = 4_294_967_279u64;
        let n = BigInt::from(p) * BigInt::from(q);
        let f = factorint(n.clone());
        assert_eq!(f, vec![(BigInt::from(q), 1), (BigInt::from(p), 1)]);
    }

    #[test]
    fn factorint_30_digit_semiprime_two_15_digit_primes() {
        // 1_000_000_000_000_037 × 1_000_000_000_000_091 (both prime)
        let p = big("1000000000000037");
        let q = big("1000000000000091");
        assert!(isprime(p.clone()) && isprime(q.clone()));
        let n = &p * &q;
        let start = std::time::Instant::now();
        let f = factorint(n.clone());
        let elapsed = start.elapsed();
        assert_eq!(f, vec![(p, 1), (q, 1)]);
        eprintln!("30-digit semiprime factored in {elapsed:?}");
    }

    /// ECM on a 39-digit input above the `u128` Montgomery range (BigInt
    /// arithmetic): ~1.5 s in release, ~13 s in debug.  Run with
    /// `cargo test --release --lib factorint_2_pow_128 -- --ignored`.
    #[test]
    #[ignore = "slow in debug builds (~13 s); passes in ~1.5 s with --release"]
    fn factorint_2_pow_128_plus_1() {
        // 2^128 + 1 = 59649589127497217 × 5704689200685129054721
        let n: BigInt = (BigInt::one() << 128usize) + 1;
        let start = std::time::Instant::now();
        let f = factorint(n.clone());
        let elapsed = start.elapsed();
        assert_eq!(
            f,
            vec![
                (big("59649589127497217"), 1),
                (big("5704689200685129054721"), 1)
            ]
        );
        eprintln!("2^128 + 1 factored in {elapsed:?}");
    }

    /// 20-digit factor of a 40-digit semiprime via ECM (B1 = 11 000 range):
    /// ~7 s in release.  Run with
    /// `cargo test --release --lib factorint_40_digit -- --ignored`.
    #[test]
    #[ignore = "slow (~7 s in release); ECM with 20-digit factors"]
    fn factorint_40_digit_with_20_digit_factor() {
        // p = 10^19 + 51 (prime), q = 10^20 + 39 (prime)
        let p = big("10000000000000000051");
        let q = big("100000000000000000039");
        assert!(isprime(p.clone()) && isprime(q.clone()));
        let n = &p * &q;
        let start = std::time::Instant::now();
        let f = factorint(n);
        eprintln!("40-digit semiprime factored in {:?}", start.elapsed());
        assert_eq!(f, vec![(p, 1), (q, 1)]);
    }

    #[test]
    fn factorint_prime_power_beyond_u64() {
        // 4294967291^3
        let p = BigInt::from(4_294_967_291u64);
        let n = p.pow(3);
        assert_eq!(factorint(n), vec![(p, 3)]);
    }

    #[test]
    fn factorint_mixed_big() {
        // 2^70 · 3 · 1000000007 · 274177
        let n: BigInt = (BigInt::one() << 70usize) * 3 * 1_000_000_007i64 * 274177i64;
        let f = factorint(n.clone());
        assert_eq!(product_of(&f), n);
        assert_eq!(f[0], (bi(2), 70));
        assert!(f.iter().all(|(p, _)| isprime(p.clone())));
    }

    #[test]
    fn factorint_u64_range_exhaustive_small_random() {
        // Products of two ~30-bit primes: exercises pollard_brent_u64.
        let ps = [1_000_000_007u64, 998_244_353, 1_000_000_009, 999_999_937];
        for i in 0..ps.len() {
            for j in i..ps.len() {
                let n = ps[i] * ps[j];
                let f = factorint(BigInt::from(n));
                assert_eq!(product_of(&f), BigInt::from(n));
                assert!(f.iter().all(|(p, _)| isprime(p.clone())));
            }
        }
    }

    // ── isprime: BPSW range ───────────────────────────────────────────

    #[test]
    fn isprime_mersenne_127() {
        let m127: BigInt = (BigInt::one() << 127usize) - 1;
        assert!(isprime(m127));
    }

    #[test]
    fn isprime_large_carmichael_like_composites() {
        // 2^127 − 1 is prime, so its square and small multiples are not.
        let m127: BigInt = (BigInt::one() << 127usize) - 1;
        assert!(!isprime(&m127 * &m127));
        assert!(!isprime(&m127 * 3));
        // Strong pseudoprime to base 2: 2047 = 23·89 (below BPSW range but sanity).
        assert!(!isprime(2047));
        // A large composite: (2^89 − 1)(2^107 − 1), both Mersenne primes.
        let a: BigInt = (BigInt::one() << 89usize) - 1;
        let b: BigInt = (BigInt::one() << 107usize) - 1;
        assert!(isprime(a.clone()) && isprime(b.clone()));
        assert!(!isprime(&a * &b));
    }

    #[test]
    fn isprime_large_known_primes() {
        // 10^30 + 57 is prime (first prime after 10^30).
        assert!(isprime(big("1000000000000000000000000000057")));
        assert!(!isprime(big("1000000000000000000000000000056")));
        // Known 40-digit prime: 2^131 − 1 is composite; 2^127−1 tested above.
        // Next prime after 10^40 is 10^40 + 121.
        assert!(isprime(big("10000000000000000000000000000000000000121")));
    }

    #[test]
    fn is_probable_prime_agrees_with_isprime() {
        for n in [2i64, 3, 97, 561, 1105, 104729, 1_000_000_007] {
            assert_eq!(is_probable_prime(n, 10), isprime(n), "{n}");
        }
        let m127: BigInt = (BigInt::one() << 127usize) - 1;
        assert!(is_probable_prime(m127.clone(), 8));
        assert!(!is_probable_prime(&m127 * 7, 8));
    }

    #[test]
    fn strong_lucas_rejects_strong_base2_pseudoprimes() {
        // Strong pseudoprimes base 2 above 2^64 do not fit, so check the
        // Lucas component directly on known spsp(2) values.
        for &n in &[2047u64, 3277, 4033, 4681, 8321, 15841, 29341, 42799, 49141] {
            let nb = BigInt::from(n);
            assert!(strong_fermat_big(&nb, &BigInt::from(2)), "{n} is spsp(2)");
            assert!(!strong_lucas_big(&nb), "{n} must fail strong Lucas");
        }
        for &p in &[101u64, 1_000_003, 2_147_483_647] {
            assert!(strong_lucas_big(&BigInt::from(p)), "{p} is prime");
        }
    }

    // ── prime counting / n-th prime / ranges ──────────────────────────

    #[test]
    fn primepi_known_values() {
        assert_eq!(primepi(1), Some(0));
        assert_eq!(primepi(2), Some(1));
        assert_eq!(primepi(10), Some(4));
        assert_eq!(primepi(1000), Some(168));
        assert_eq!(primepi(10_000), Some(1229));
        assert_eq!(primepi(100_000), Some(9592));
        assert_eq!(primepi(1_000_000), Some(78_498));
        assert_eq!(primepi(10_000_000), Some(664_579));
    }

    #[test]
    fn primepi_matches_sieve_for_small_n() {
        let ps = primes_up_to(2000);
        for n in 0..=2000i64 {
            let expected = ps.iter().filter(|&&p| p <= n).count() as u64;
            assert_eq!(primepi(n), Some(expected), "π({n})");
        }
    }

    #[test]
    fn nth_prime_known_values() {
        assert_eq!(prime(1), Some(bi(2)));
        assert_eq!(prime(2), Some(bi(3)));
        assert_eq!(prime(6), Some(bi(13)));
        assert_eq!(prime(25), Some(bi(97)));
        assert_eq!(prime(1000), Some(bi(7919)));
        assert_eq!(prime(100_000), Some(bi(1_299_709)));
        assert_eq!(prime(0), None);
    }

    #[test]
    fn primerange_segmented_and_iterative() {
        let small: Vec<i64> = primerange(0, 30)
            .iter()
            .map(|p| p.try_into().unwrap())
            .collect();
        assert_eq!(small, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
        // Segment far from zero.
        let seg = primerange(1_000_000, 1_000_100);
        assert_eq!(seg.len(), 6);
        assert_eq!(seg[0], bi(1_000_003));
        // Huge numbers → nextprime iteration.
        let start = big("1000000000000000000000000000000");
        let end = &start + 200;
        let big_ps = primerange(start.clone(), end);
        assert_eq!(big_ps[0], &start + 57);
        assert!(big_ps.iter().all(|p| isprime(p.clone())));
    }

    #[test]
    fn primes_up_to_matches_primepi() {
        assert_eq!(
            primes_up_to(100_000).len() as u64,
            primepi(100_000).unwrap()
        );
    }

    // ── divisor functions ─────────────────────────────────────────────

    #[test]
    fn divisor_sigma_values() {
        assert_eq!(divisor_sigma(1, 0), bi(1));
        assert_eq!(divisor_sigma(12, 0), bi(6));
        assert_eq!(divisor_sigma(12, 1), bi(28));
        assert_eq!(divisor_sigma(12, 2), bi(210));
        assert_eq!(divisor_sigma(0, 1), bi(0));
        // σ₁ matches divisor_sum for a range.
        for n in 1..200i64 {
            assert_eq!(divisor_sigma(n, 1), divisor_sum(n));
            assert_eq!(divisor_sigma(n, 0), bi(divisor_count(n) as i64));
        }
    }

    #[test]
    fn perfect_abundant_deficient_classification() {
        let perfect: Vec<i64> = (1..10_000).filter(|&n| is_perfect(n)).collect();
        assert_eq!(perfect, vec![6, 28, 496, 8128]);
        assert!(is_abundant(12) && is_abundant(945)); // 945 first odd abundant
        assert!(is_deficient(1) && is_deficient(7) && is_deficient(16));
        let abundant_below_50: Vec<i64> = (1..50).filter(|&n| is_abundant(n)).collect();
        assert_eq!(abundant_below_50, vec![12, 18, 20, 24, 30, 36, 40, 42, 48]);
    }

    #[test]
    fn carmichael_lambda_oeis_a002322() {
        let expected = [
            1i64, 1, 2, 2, 4, 2, 6, 2, 6, 4, 10, 2, 12, 6, 4, 4, 16, 6, 18, 4, 6, 10, 22, 2, 20,
            12, 18, 6, 28, 4,
        ];
        for (i, &e) in expected.iter().enumerate() {
            assert_eq!(carmichael_lambda(i as i64 + 1), bi(e), "λ({})", i + 1);
        }
    }

    #[test]
    fn perfect_power_detection() {
        assert_eq!(perfect_power(64), Some((bi(2), 6)));
        assert_eq!(perfect_power(1024), Some((bi(2), 10)));
        assert_eq!(perfect_power(36), Some((bi(6), 2)));
        assert_eq!(perfect_power(216), Some((bi(6), 3)));
        assert_eq!(perfect_power(-8), Some((bi(-2), 3)));
        assert_eq!(perfect_power(-4), None);
        assert_eq!(perfect_power(1), None);
        assert_eq!(perfect_power(0), None);
        assert_eq!(perfect_power(10), None);
        let big_pow = BigInt::from(1_000_000_007u64).pow(5);
        assert_eq!(perfect_power(big_pow), Some((bi(1_000_000_007), 5)));
        assert!(is_perfect_power(3u64.pow(20)));
        assert!(!is_perfect_power(3u64.pow(20) + 1));
    }

    #[test]
    fn mersenne_prime_exponents() {
        let known = [2i64, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127];
        for p in 2..=130i64 {
            assert_eq!(is_mersenne_prime(p), known.contains(&p), "p = {p}");
        }
        assert!(is_mersenne_prime(521));
        assert!(!is_mersenne_prime(-3));
    }

    // ── quadratic residues ────────────────────────────────────────────

    #[test]
    fn jacobi_and_kronecker() {
        assert_eq!(jacobi_symbol(1001, 9907).unwrap(), -1);
        assert_eq!(jacobi_symbol(19, 45).unwrap(), 1);
        assert_eq!(jacobi_symbol(8, 21).unwrap(), -1);
        assert_eq!(jacobi_symbol(5, 21).unwrap(), 1);
        assert_eq!(jacobi_symbol(0, 21).unwrap(), 0);
        assert!(jacobi_symbol(1, 10).is_err());
        assert!(jacobi_symbol(1, -3).is_err());
        // Jacobi agrees with Legendre for primes.
        for p in [3i64, 5, 7, 11, 13, 17, 19, 23] {
            for a in -10..30i64 {
                assert_eq!(
                    jacobi_symbol(a, p).unwrap(),
                    legendre_symbol(a, p),
                    "({a}/{p})"
                );
                assert_eq!(kronecker_symbol(a, p), legendre_symbol(a, p));
            }
        }
        // Kronecker at 2, −1, 0.
        assert_eq!(kronecker_symbol(1, 2), 1);
        assert_eq!(kronecker_symbol(7, 2), 1);
        assert_eq!(kronecker_symbol(3, 2), -1);
        assert_eq!(kronecker_symbol(4, 2), 0);
        assert_eq!(kronecker_symbol(-3, -1), -1);
        assert_eq!(kronecker_symbol(3, -1), 1);
        assert_eq!(kronecker_symbol(1, 0), 1);
        assert_eq!(kronecker_symbol(2, 0), 0);
        // Multiplicativity in the bottom argument: (a/mn) = (a/m)(a/n).
        for a in -7..8i64 {
            assert_eq!(
                kronecker_symbol(a, 12),
                kronecker_symbol(a, 4) * kronecker_symbol(a, 3)
            );
        }
    }

    #[test]
    fn sqrt_mod_primes_brute_force_check() {
        for p in [3i64, 5, 7, 11, 13, 17, 41, 97, 101, 113, 193, 257, 65537] {
            for a in 0..p.min(60) {
                let r = sqrt_mod(a, p);
                let exists = (0..p).any(|x| (x * x) % p == a);
                assert_eq!(r.is_some(), exists, "a={a} p={p}");
                if let Some(r) = r {
                    let r: i64 = r.try_into().unwrap();
                    assert_eq!((r * r) % p, a);
                }
                assert_eq!(is_quad_residue(a, p), exists);
            }
        }
    }

    #[test]
    fn sqrt_mod_all_composite_brute_force_check() {
        for n in [
            1i64, 2, 4, 8, 9, 12, 15, 16, 27, 36, 45, 64, 72, 100, 105, 128, 225, 4096, 8192,
        ] {
            for a in 0..n.min(40) {
                let roots = sqrt_mod_all(a, n);
                let expected: Vec<BigInt> =
                    (0..n).filter(|x| (x * x) % n == a % n).map(bi).collect();
                assert_eq!(roots, expected, "a={a} n={n}");
            }
        }
    }

    #[test]
    fn sqrt_mod_large_prime_and_prime_power() {
        let p = big("1000000000000000000000000000057");
        let a = BigInt::from(123456789u64);
        let a2 = (&a * &a) % &p;
        let r = sqrt_mod(a2.clone(), p.clone()).unwrap();
        assert_eq!((&r * &r) % &p, a2);
        // 7^10 modulus, a = 3^2 · 7^4 → roots exist
        let n = BigInt::from(7).pow(10);
        let a = BigInt::from(9) * BigInt::from(7).pow(4);
        let roots = sqrt_mod_all(a.clone(), n.clone());
        assert!(!roots.is_empty());
        for r in &roots {
            assert_eq!((r * r) % &n, a);
        }
        // 2^20, a = 17 ≡ 1 mod 8 → 4 roots
        let n = BigInt::one() << 20usize;
        let roots = sqrt_mod_all(17, n.clone());
        assert_eq!(roots.len(), 4);
        for r in &roots {
            assert_eq!((r * r) % &n, bi(17));
        }
    }

    // ── orders, primitive roots, discrete logs ────────────────────────

    #[test]
    fn n_order_values() {
        assert_eq!(n_order(2, 7), Some(bi(3)));
        assert_eq!(n_order(3, 7), Some(bi(6)));
        assert_eq!(n_order(1, 7), Some(bi(1)));
        assert_eq!(n_order(2, 8), None);
        assert_eq!(n_order(5, 1), Some(bi(1)));
        assert_eq!(n_order(10, 561), multiplicative_order(10, 561));
        // Order divides λ(n).
        for n in 2..60i64 {
            for a in 1..n {
                if gcd(a, n).is_one() {
                    let ord = n_order(a, n).unwrap();
                    assert!((carmichael_lambda(n) % &ord).is_zero());
                    assert_eq!(mod_pow(a, ord, n), bi(1));
                }
            }
        }
    }

    #[test]
    fn primitive_roots_oeis_a001918() {
        // Smallest primitive root of the n-th prime (n ≥ 2): 2,2,3,2,2,3,2,5,2,3,2,6,3,5,2,2,2,2,7,5,3,2,3,5
        let primes = [
            3i64, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
            89, 97,
        ];
        let expected = [
            2i64, 2, 3, 2, 2, 3, 2, 5, 2, 3, 2, 6, 3, 5, 2, 2, 2, 2, 7, 5, 3, 2, 3, 5,
        ];
        for (p, g) in primes.iter().zip(expected.iter()) {
            assert_eq!(primitive_root(*p), Some(bi(*g)), "p = {p}");
            assert!(is_primitive_root(*g, *p));
        }
        assert_eq!(primitive_root(1), Some(bi(0)));
        assert_eq!(primitive_root(2), Some(bi(1)));
        assert_eq!(primitive_root(4), Some(bi(3)));
        assert_eq!(primitive_root(9), Some(bi(2)));
        assert_eq!(primitive_root(18), Some(bi(5)));
        assert_eq!(primitive_root(8), None);
        assert_eq!(primitive_root(15), None);
        assert_eq!(primitive_root(0), None);
        assert!(!is_primitive_root(2, 15));
    }

    #[test]
    fn discrete_log_brute_force_check() {
        for n in [7i64, 11, 13, 17, 25, 27, 31, 49, 97, 101, 128, 255, 1009] {
            for a in 2..n.min(12) {
                if !gcd(a, n).is_one() {
                    continue;
                }
                for x in 0..20i64 {
                    let b = mod_pow(a, x, n);
                    let got = discrete_log(a, b.clone(), n).expect("solution exists");
                    assert_eq!(mod_pow(a, got.clone(), n), b, "a={a} b={b} n={n}");
                    // Smallest solution: no smaller y works.
                    let got_i: i64 = got.try_into().unwrap();
                    assert!(
                        (0..got_i).all(|y| mod_pow(a, y, n) != b),
                        "a={a} n={n}: {got_i} is not minimal"
                    );
                }
            }
        }
    }

    #[test]
    fn discrete_log_pohlig_hellman_large() {
        // Smooth group order: p = 7 · 2^26 + 1 (an NTT prime).
        let p = 469_762_049i64;
        assert!(isprime(p));
        let g = primitive_root(p).unwrap();
        let x = bi(123_456_789);
        let b = mod_pow(g.clone(), x.clone(), p);
        assert_eq!(discrete_log(g, b, p), Some(x));
        // Non-smooth: p = 1_000_000_007, order = p − 1 = 2 · 500000003
        let p = 1_000_000_007i64;
        let g = primitive_root(p).unwrap();
        let x = bi(987_654_321);
        let b = mod_pow(g.clone(), x.clone(), p);
        assert_eq!(discrete_log(g, b, p), Some(x));
    }

    #[test]
    fn discrete_log_no_solution_and_non_coprime() {
        assert_eq!(discrete_log(2, 3, 7), None); // 2 has order 3
        assert_eq!(discrete_log(4, 2, 8), None); // 4^x ∈ {1, 4, 0}
        assert_eq!(discrete_log(2, 4, 8), Some(bi(2)));
        assert_eq!(discrete_log(3, 1, 7), Some(bi(0)));
    }

    // ── digits, palindromes, continued fractions ──────────────────────

    #[test]
    fn digits_and_palindromes() {
        assert_eq!(digits(0, 10).unwrap(), vec![0]);
        assert_eq!(digits(1234, 10).unwrap(), vec![1, 2, 3, 4]);
        assert_eq!(digits(255, 2).unwrap(), vec![1; 8]);
        assert_eq!(digits(-255, 16).unwrap(), vec![15, 15]);
        assert!(digits(5, 0).is_err());
        assert!(is_palindromic(0, 10));
        assert!(is_palindromic(1221, 10));
        assert!(is_palindromic(585, 2)); // 1001001001
        assert!(!is_palindromic(10, 10));
        assert!(!is_palindromic(10, 1));
    }

    #[test]
    fn continued_fraction_roundtrip_via_convergents() {
        for (p, q) in [
            (415i64, 93i64),
            (-7, 3),
            (1, 1),
            (0, 1),
            (355, 113),
            (-1, 7),
            (100, 3),
        ] {
            let r = Ratio::new(bi(p), bi(q));
            let cf = continued_fraction(&r);
            let conv = continued_fraction_convergents(&cf);
            assert_eq!(*conv.last().unwrap(), r, "{p}/{q}: {cf:?}");
            // Canonical: last term > 1 unless the expansion has length 1.
            if cf.len() > 1 {
                assert!(*cf.last().unwrap() > BigInt::one());
            }
        }
    }

    #[test]
    fn continued_fraction_periodic_known() {
        let cf = |d: i64| {
            let cf = continued_fraction_periodic(d).unwrap();
            let hi: Vec<i64> = cf
                .pre_period
                .iter()
                .map(|t| t.try_into().unwrap())
                .collect();
            let pi: Vec<i64> = cf.period.iter().map(|t| t.try_into().unwrap()).collect();
            (hi, pi)
        };
        assert_eq!(cf(2), (vec![1], vec![2]));
        assert_eq!(cf(3), (vec![1], vec![1, 2]));
        assert_eq!(cf(5), (vec![2], vec![4]));
        assert_eq!(cf(7), (vec![2], vec![1, 1, 1, 4]));
        assert_eq!(cf(13), (vec![3], vec![1, 1, 1, 1, 6]));
        assert_eq!(cf(61), (vec![7], vec![1, 4, 3, 1, 2, 2, 1, 3, 4, 1, 14]));
        assert_eq!(cf(16), (vec![4], vec![]));
        assert_eq!(cf(0), (vec![0], vec![]));
        assert!(continued_fraction_periodic(-2).is_none());
    }

    #[test]
    fn egyptian_fraction_greedy() {
        let to_i =
            |v: Vec<BigInt>| -> Vec<i64> { v.iter().map(|t| t.try_into().unwrap()).collect() };
        assert_eq!(
            to_i(egyptian_fraction(&Ratio::new(bi(4), bi(13))).unwrap()),
            vec![4, 18, 468]
        );
        let e = egyptian_fraction(&Ratio::new(bi(5), bi(121))).unwrap();
        assert_eq!(e.len(), 5);
        assert_eq!(e[0], bi(25));
        assert_eq!(e[1], bi(757));
        assert_eq!(e[2], bi(763309));
        assert_eq!(e[3], bi(873960180913));
        assert_eq!(e[4], big("1527612795642093418846225"));
        assert_eq!(
            to_i(egyptian_fraction(&Ratio::new(bi(1), bi(1))).unwrap()),
            vec![1]
        );
        assert_eq!(
            to_i(egyptian_fraction(&Ratio::new(bi(3), bi(4))).unwrap()),
            vec![2, 4]
        );
        assert!(egyptian_fraction(&Ratio::new(bi(0), bi(1))).is_none());
        assert!(egyptian_fraction(&Ratio::new(bi(-1), bi(2))).is_none());
        // Sum check.
        let r = Ratio::new(bi(7), bi(15));
        let sum: Q = egyptian_fraction(&r)
            .unwrap()
            .iter()
            .map(|d| Ratio::new(BigInt::one(), d.clone()))
            .sum();
        assert_eq!(sum, r);
    }

    // ── sequences ─────────────────────────────────────────────────────

    #[test]
    fn fibonacci_and_lucas_values() {
        let fibs = [0i64, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144];
        for (i, &f) in fibs.iter().enumerate() {
            assert_eq!(fibonacci(i as i64), bi(f), "F({i})");
        }
        assert_eq!(fibonacci(-1), bi(1));
        assert_eq!(fibonacci(-2), bi(-1));
        assert_eq!(fibonacci(-8), bi(-21));
        assert_eq!(fibonacci(-9), bi(34));
        assert_eq!(fibonacci(100), big("354224848179261915075"));
        assert_eq!(fibonacci(300).to_string().len(), 63);
        let lucas_vals = [2i64, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123];
        for (i, &l) in lucas_vals.iter().enumerate() {
            assert_eq!(lucas(i as i64), bi(l), "L({i})");
        }
        assert_eq!(lucas(-1), bi(-1));
        assert_eq!(lucas(-2), bi(3));
        assert_eq!(lucas(-3), bi(-4));
        // Identity: L_n = F_{n−1} + F_{n+1}
        for n in 1..40i64 {
            assert_eq!(lucas(n), fibonacci(n - 1) + fibonacci(n + 1));
        }
    }

    #[test]
    fn bernoulli_euler_harmonic_values() {
        let r = |p: i64, q: i64| Ratio::new(bi(p), bi(q));
        assert_eq!(bernoulli(0), Some(r(1, 1)));
        assert_eq!(bernoulli(1), Some(r(-1, 2)));
        assert_eq!(bernoulli(2), Some(r(1, 6)));
        assert_eq!(bernoulli(4), Some(r(-1, 30)));
        assert_eq!(bernoulli(6), Some(r(1, 42)));
        assert_eq!(bernoulli(8), Some(r(-1, 30)));
        assert_eq!(bernoulli(10), Some(r(5, 66)));
        assert_eq!(bernoulli(12), Some(r(-691, 2730)));
        assert_eq!(bernoulli(20), Some(r(-174611, 330)));
        assert_eq!(bernoulli(-1), None);
        // Euler numbers A000364 (with signs): 1, −1, 5, −61, 1385, −50521, 2702765
        let euler = [1i64, -1, 5, -61, 1385, -50521, 2_702_765, -199_360_981];
        for (i, &e) in euler.iter().enumerate() {
            assert_eq!(euler_number(2 * i as i64), Some(bi(e)), "E({})", 2 * i);
        }
        assert_eq!(euler_number(5), Some(bi(0)));
        assert_eq!(euler_number(-2), None);
        assert_eq!(harmonic(1), Some(r(1, 1)));
        assert_eq!(harmonic(2), Some(r(3, 2)));
        assert_eq!(harmonic(3), Some(r(11, 6)));
        assert_eq!(harmonic(10), Some(r(7381, 2520)));
        assert_eq!(harmonic(-1), None);
    }

    #[test]
    fn gcdex_bezout() {
        for (a, b) in [(240i64, 46i64), (0, 5), (5, 0), (-12, 18), (17, 31), (0, 0)] {
            let g = gcdex(a, b);
            assert_eq!(g.gcd, gcd(a, b));
            assert_eq!(bi(a) * &g.x + bi(b) * &g.y, g.gcd);
        }
    }

    /// `gcdex` delegates to `num_integer`; the Bézout coefficients it
    /// returns must be exactly those of the crate's own `extended_gcd_big`
    /// (same recurrence, same `gcd ≥ 0` normalisation), including the
    /// sign choices for negative and zero inputs.
    #[test]
    fn gcdex_matches_internal_extended_gcd() {
        for a in -40i64..=40 {
            for b in -40i64..=40 {
                let g = gcdex(a, b);
                let (g0, x0, y0) = extended_gcd_big(&bi(a), &bi(b));
                assert_eq!((g.gcd, g.x, g.y), (g0, x0, y0), "gcdex({a}, {b})");
            }
        }
    }

    #[test]
    fn iroot_values() {
        assert_eq!(iroot(1000, 3), Some(bi(10)));
        assert_eq!(iroot(1001, 3), Some(bi(10)));
        assert_eq!(iroot(999, 3), Some(bi(9)));
        assert_eq!(iroot(-8, 3), Some(bi(-2)));
        assert_eq!(iroot(-9, 3), Some(bi(-3)));
        assert_eq!(iroot(-8, 2), None);
        assert_eq!(iroot(8, 0), None);
        assert_eq!(iroot(0, 5), Some(bi(0)));
    }

    #[test]
    fn re_exported_combinatorics_reachable() {
        assert_eq!(bell(5), Some(bi(52)));
        assert_eq!(catalan(5), Some(bi(42)));
        assert_eq!(binomial(10, 3), bi(120));
        assert_eq!(derangements(4), Some(bi(9)));
        assert_eq!(npartitions(5), Some(bi(7)));
        assert_eq!(partitions(4).count(), 5);
        let _: PartitionIter = partitions(3);
    }

    // ── gcd_many / lcm_many / denominators ────────────────────────────────────────

    #[test]
    fn gcd_many_basic_and_negatives() {
        assert_eq!(gcd_many(&[bi(12), bi(18), bi(30)]), bi(6));
        assert_eq!(gcd_many(&[bi(-12), bi(18)]), bi(6));
        assert_eq!(gcd_many(&[bi(-7)]), bi(7));
        assert_eq!(gcd_many(&[bi(0), bi(0)]), bi(0));
        assert_eq!(gcd_many(&[bi(0), bi(5)]), bi(5));
        assert_eq!(gcd_many(&[bi(7), bi(11), bi(13)]), bi(1));
    }

    #[test]
    fn gcd_many_empty_is_zero() {
        assert_eq!(gcd_many(&[]), bi(0));
        assert_eq!(igcd::<i64>(&[]), bi(0));
    }

    #[test]
    fn lcm_many_basic_and_zero() {
        assert_eq!(lcm_many(&[bi(4), bi(6), bi(10)]), bi(60));
        assert_eq!(lcm_many(&[bi(-4), bi(6)]), bi(12));
        assert_eq!(lcm_many(&[bi(3), bi(0)]), bi(0));
        assert_eq!(lcm_many(&[bi(9)]), bi(9));
    }

    #[test]
    fn lcm_many_empty_is_one() {
        assert_eq!(lcm_many(&[]), bi(1));
        assert_eq!(ilcm::<i64>(&[]), bi(1));
    }

    #[test]
    fn igcd_ilcm_accept_i64_slices() {
        assert_eq!(igcd(&[12i64, 18, 30]), bi(6));
        assert_eq!(ilcm(&[2i64, 3, 4]), bi(12));
        assert_eq!(igcd(&[bi(100), bi(75)]), bi(25));
    }

    #[test]
    fn rational_lcm_of_denominators_clears() {
        let q = |n: i64, d: i64| Ratio::new(bi(n), bi(d));
        let v = [q(1, 3), q(1, 7), q(5, 21), q(2, 1)];
        let l = rational_lcm_of_denominators(&v);
        assert_eq!(l, bi(21));
        for r in &v {
            assert!((r * Ratio::from_integer(l.clone())).is_integer());
        }
        assert_eq!(rational_lcm_of_denominators(&[]), bi(1));
        assert_eq!(rational_lcm_of_denominators(&[q(4, 1)]), bi(1));
    }
}
