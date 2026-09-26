//! The even Bernoulli numbers `B₂ₖ` of the asymptotic series (`ln Γ`, `ψ`,
//! `ψ⁽ⁿ⁾` by Stirling's series), from the tangent numbers.
//!
//! The series need `B₂ₖ` up to `k ≈ 0.1·p` at a working precision of `p`
//! bits, and a value that is zero to the precision is pursued to about
//! 3,200 bits (`ZERO_SEARCH_BITS`): some 350 numbers.  The exact
//! rational recurrence of `base::bernoulli` (`B_m = −Σ C(m+1, k)·B_k/(m+1)`,
//! a sum of rationals per number) took 5–8 s to reach them.  The tangent
//! numbers `T₁ = 1, T₂ = 2, T₃ = 16, …` need only integer updates by small
//! factors, `O(n²)` of them for `T₁ … Tₙ` (Brent and Harvey, "Fast
//! computation of Bernoulli, Tangent and Secant numbers", 2011,
//! Algorithm TangentNumbers — the published algorithm, not code), and
//!
//! ```text
//! B₂ₖ = (−1)^(k−1) · 2k · Tₖ / (2^(2k) · (2^(2k) − 1)).
//! ```

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};
use parking_lot::Mutex;

use crate::base::numeric::Q;

/// `B₂, B₄, …, B₂ₙ` computed so far.
static CACHE: Mutex<Vec<Q>> = Mutex::new(Vec::new());

/// The tangent numbers `T₁ … Tₙ` (Brent–Harvey, Algorithm TangentNumbers).
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

/// `B₂ₖ` for `k ≥ 1` (`B₀ = 1` for `k = 0`).
pub(super) fn even(k: usize) -> Q {
    if k == 0 {
        return Q::one();
    }
    let mut cache = CACHE.lock();
    if cache.len() < k {
        // Recomputed from scratch (the algorithm is in place), for at least
        // twice as many numbers as before, so that the work stays `O(n²)`.
        let n = k.max(2 * cache.len()).max(32);
        let t = tangent_numbers(n);
        let mut out = Vec::with_capacity(n);
        for (i, tk) in t.iter().enumerate().skip(1) {
            let two_k = 2 * i;
            let p = BigInt::one() << two_k;
            let den = &p * (&p - BigInt::one());
            let mut num = tk * BigInt::from(two_k);
            if i % 2 == 0 {
                num = -num;
            }
            out.push(Ratio::new(num, den));
        }
        *cache = out;
    }
    cache[k - 1].clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first even Bernoulli numbers, and agreement with the rational
    /// recurrence of `base::bernoulli` up to `B₁₂₀`.
    #[test]
    fn matches_the_rational_recurrence() {
        let q = |n: i64, d: i64| Ratio::new(BigInt::from(n), BigInt::from(d));
        assert_eq!(even(1), q(1, 6));
        assert_eq!(even(2), q(-1, 30));
        assert_eq!(even(3), q(1, 42));
        assert_eq!(even(6), q(691, -2730));
        for k in 1..=60 {
            assert_eq!(
                even(k),
                crate::base::bernoulli::bernoulli(2 * k),
                "B_{}",
                2 * k
            );
        }
    }
}
