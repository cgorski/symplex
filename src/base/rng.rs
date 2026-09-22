//! Deterministic pseudo-random generators: seeded, reproducible, not
//! cryptographic.
//!
//! Two tiny generators, each the single spelling of its algorithm in the
//! crate:
//!
//! * [`SplitMix64`] (Steele, Lea & Flood 2014) — the general-purpose
//!   generator behind [`stats::Rng`](crate::stats::Rng), differential
//!   evolution's seed, and the sample points of
//!   [`Ex::probably_equal`](crate::api::expr::Ex::probably_equal).
//! * [`XorShift64Star`] (Vigna 2016) — the generator the randomised
//!   number-theoretic and polynomial algorithms (Miller–Rabin witnesses,
//!   Pollard–Brent, Cantor–Zassenhaus splitting) draw from.
//!
//! Both streams are pinned bit-for-bit by `tests/v20/v20_base.rs`: the
//! algorithms decide which factor a randomised split finds first and which
//! Monte-Carlo sample a test sees, so a change here is an output change.

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::Zero;

/// SplitMix64: a fast, well-distributed 64-bit generator with a 64-bit
/// state (Steele, Lea & Flood, "Fast splittable pseudorandom number
/// generators", OOPSLA 2014).  Every seed gives a full-period stream.
///
/// ```
/// use symplex::SplitMix64;
///
/// let mut a = SplitMix64::new(42);
/// let mut b = SplitMix64::new(42);
/// assert_eq!(a.next_u64(), b.next_u64());          // reproducible
/// assert_eq!(SplitMix64::new(42).next_u64(), 0xBDD7_3226_2FEB_6E95);
/// let u = a.next_f64();
/// assert!((0.0..1.0).contains(&u));
/// assert!(a.below(7) < 7);
/// assert_eq!(SplitMix64::new(0).below(0), 0);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// A generator with the given seed.
    pub const fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` with 53 random bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `0..n` (`0` for `n == 0`).  Plain modulo: the bias is
    /// below `2⁻⁴⁰` for any `n` that fits a realistic population or sample.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform index in `0..n` that is not in `excluded`.
    ///
    /// `excluded` must hold distinct values `< n` and is sorted in place;
    /// the caller guarantees `excluded.len() < n`.
    pub fn below_excluding(&mut self, n: usize, excluded: &mut [usize]) -> usize {
        excluded.sort_unstable();
        let mut r = self.below(n.saturating_sub(excluded.len()));
        for &e in excluded.iter() {
            if r >= e {
                r += 1;
            }
        }
        r
    }

    /// Fisher–Yates shuffle of `items` in place.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

/// xorshift64\*: a 64-bit xorshift state whose output is scrambled by one
/// multiplication (Vigna, "An experimental exploration of Marsaglia's
/// xorshift generators, scrambled", ACM TOMS 42(4), 2016).  The seed is
/// mixed with the golden-ratio constant and a zero seed is replaced by 1,
/// since the all-zero state is a fixed point of xorshift.
///
/// ```
/// use symplex::XorShift64Star;
///
/// assert_eq!(XorShift64Star::new(42).next_u64(), 0x1C28_3E14_F85F_D6CB);
/// // Seeds 0 and 1 coincide: 0 is not a usable xorshift state.
/// assert_eq!(XorShift64Star::new(0).next_u64(), XorShift64Star::new(1).next_u64());
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct XorShift64Star(u64);

impl XorShift64Star {
    /// A generator with the given seed.
    pub const fn new(seed: u64) -> Self {
        let seed = if seed == 0 { 1 } else { seed };
        XorShift64Star(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A big integer uniform in `0..n` up to the bias of reducing 64 more
    /// random bits than `n` has (below `2⁻⁶⁴`).  `n` must be positive.
    ///
    /// # Panics
    ///
    /// If `n` is zero (reduction modulo zero).
    pub fn next_big_below(&mut self, n: &BigInt) -> BigInt {
        let bits = n.bits() as usize + 64;
        let words = bits.div_ceil(64);
        let mut acc = BigInt::zero();
        for _ in 0..words {
            acc = (acc << 64usize) + BigInt::from(self.next_u64());
        }
        acc.mod_floor(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_is_deterministic_and_in_range() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            let u = a.next_f64();
            assert_eq!(u, b.next_f64());
            assert!((0.0..1.0).contains(&u));
            let k = a.below(7);
            assert_eq!(k, b.below(7));
            assert!(k < 7);
        }
        assert_eq!(SplitMix64::new(0).below(0), 0);
    }

    #[test]
    fn below_excluding_never_returns_excluded() {
        let mut rng = SplitMix64::new(7);
        for _ in 0..1000 {
            let i = rng.below(10);
            let r1 = rng.below_excluding(10, &mut [i]);
            assert_ne!(r1, i);
            let r2 = rng.below_excluding(10, &mut [i, r1]);
            assert!(r2 != i && r2 != r1);
            let r3 = rng.below_excluding(10, &mut [i, r1, r2]);
            assert!(r3 != i && r3 != r1 && r3 != r2 && r3 < 10);
        }
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut rng = SplitMix64::new(3);
        let mut v: Vec<usize> = (0..20).collect();
        rng.shuffle(&mut v);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..20).collect::<Vec<_>>());
        assert_ne!(v, sorted, "20 elements should not stay in order");
    }

    #[test]
    fn xorshift_big_below_is_in_range() {
        let mut rng = XorShift64Star::new(5);
        let n = BigInt::from(1u64) << 100usize;
        for _ in 0..50 {
            let x = rng.next_big_below(&n);
            assert!(x >= BigInt::zero() && x < n);
        }
        assert_eq!(rng.next_big_below(&BigInt::from(1)), BigInt::zero());
    }
}
