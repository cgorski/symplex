//! Sturm sequences for real root counting and isolation.
//!
//! A **Sturm chain** for a square-free polynomial `p` is the sequence
//!
//! ```text
//! P_0 = p,  P_1 = p',  P_{i+1} = -rem(P_{i-1}, P_i).primitive_part()
//! ```
//!
//! terminated when the remainder is zero.  The number of distinct real
//! roots of `p` in an interval `(a, b]` equals `σ(a) − σ(b)`, where
//! `σ(x)` counts the sign variations in the sequence evaluated at `x`.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{Signed, Zero};

use crate::poly::Poly;

/// A Sturm chain built from a polynomial.
#[derive(Debug, Clone)]
pub(crate) struct SturmChain {
    /// The polynomials P_0, P_1, …, P_k forming the chain.
    chain: Vec<Poly>,
}

#[allow(dead_code)] // Used indirectly via Ex::count_real_roots() bridge; will be exposed publicly later
impl SturmChain {
    /// Build a Sturm chain from polynomial `p`.
    ///
    /// The input is first made square-free so that the chain correctly
    /// counts *distinct* real roots.
    pub fn new(p: &Poly) -> Self {
        if p.is_zero() {
            return SturmChain {
                chain: vec![Poly::zero()],
            };
        }

        let p0 = p.square_free_part();
        let p1 = p0.derivative();

        if p1.is_zero() {
            // Constant (degree 0) — no roots.
            return SturmChain { chain: vec![p0] };
        }

        let mut chain = vec![p0.clone(), p1.clone()];
        let mut prev = p0;
        let mut curr = p1;

        loop {
            let rem = prev.rem(&curr);
            if rem.is_zero() {
                break;
            }
            let neg_rem = (-&rem).primitive_part();
            chain.push(neg_rem.clone());
            prev = curr;
            curr = neg_rem;
        }

        SturmChain { chain }
    }

    /// Count sign variations when each polynomial is evaluated at `x`.
    ///
    /// Zeros are skipped (as per the standard Sturm convention).
    pub fn sign_variations_at(&self, x: &Ratio<BigInt>) -> usize {
        let signs: Vec<i8> = self
            .chain
            .iter()
            .map(|p| {
                let v = p.eval(x);
                if v.is_positive() {
                    1
                } else if v.is_negative() {
                    -1
                } else {
                    0
                }
            })
            .collect();
        count_sign_changes(&signs)
    }

    /// Count sign variations at +∞.
    ///
    /// At +∞ the sign of a polynomial equals the sign of its leading
    /// coefficient.
    pub fn sign_variations_at_pos_inf(&self) -> usize {
        let signs: Vec<i8> = self.chain.iter().map(leading_sign).collect();
        count_sign_changes(&signs)
    }

    /// Count sign variations at −∞.
    ///
    /// At −∞ the sign of a polynomial of degree `d` with leading
    /// coefficient `c` is `sign(c) * (-1)^d`.
    pub fn sign_variations_at_neg_inf(&self) -> usize {
        let signs: Vec<i8> = self
            .chain
            .iter()
            .map(|p| {
                let ls = leading_sign(p);
                if ls == 0 {
                    return 0;
                }
                let deg = p.degree().unwrap_or(0);
                if deg % 2 == 0 { ls } else { -ls }
            })
            .collect();
        count_sign_changes(&signs)
    }

    /// Total number of distinct real roots of the polynomial.
    pub fn count_real_roots(&self) -> usize {
        let neg = self.sign_variations_at_neg_inf();
        let pos = self.sign_variations_at_pos_inf();
        neg.saturating_sub(pos)
    }

    /// Quick check: does the polynomial have zero real roots?
    pub fn has_no_real_roots(&self) -> bool {
        self.count_real_roots() == 0
    }

    /// Count real roots in the half-open interval `(a, b]`.
    pub fn count_roots_in(&self, a: &Ratio<BigInt>, b: &Ratio<BigInt>) -> usize {
        let sa = self.sign_variations_at(a);
        let sb = self.sign_variations_at(b);
        sa.saturating_sub(sb)
    }

    /// Isolate real roots into disjoint sub-intervals of `(a, b)` by
    /// bisection.
    ///
    /// Returns a list of `(lo, hi)` intervals, each containing exactly
    /// one root.  `max_depth` bounds the number of bisection levels to
    /// prevent runaway recursion.
    pub fn isolate_roots_in(
        &self,
        a: &Ratio<BigInt>,
        b: &Ratio<BigInt>,
        max_depth: u32,
    ) -> Vec<(Ratio<BigInt>, Ratio<BigInt>)> {
        // Explicit stack (no recursion); intervals are pushed right-first so
        // that the output comes out sorted left to right.
        let two = Ratio::from_integer(BigInt::from(2));
        let mut result = Vec::new();
        let mut stack: Vec<(Ratio<BigInt>, Ratio<BigInt>, u32)> =
            vec![(a.clone(), b.clone(), max_depth)];
        while let Some((lo, hi, depth)) = stack.pop() {
            let n = self.count_roots_in(&lo, &hi);
            if n == 0 {
                continue;
            }
            if n == 1 || depth == 0 {
                result.push((lo, hi));
                continue;
            }
            let mid = (&lo + &hi) / &two;
            stack.push((mid.clone(), hi, depth - 1));
            stack.push((lo, mid, depth - 1));
        }
        result
    }

    /// Isolate **all** distinct real roots of the polynomial into disjoint
    /// rational intervals, each containing exactly one root.
    ///
    /// The search starts from the Cauchy root bound.  Every returned pair
    /// `(lo, hi)` satisfies `lo < hi` and has exactly one root in `(lo, hi]`,
    /// except that a root which is exactly hit by a bisection point is
    /// reported as the degenerate interval `(r, r)`.  Intervals are sorted.
    pub fn isolate_all_real_roots(&self) -> Vec<(Ratio<BigInt>, Ratio<BigInt>)> {
        let Some(p) = self.chain.first() else {
            return vec![];
        };
        if p.degree().unwrap_or(0) == 0 {
            return vec![];
        }
        let bound = cauchy_bound(p) + Ratio::from_integer(BigInt::from(1));
        let neg_bound = -bound.clone();
        let raw = self.isolate_roots_in(&neg_bound, &bound, 256);
        raw.into_iter()
            .map(|(lo, hi)| {
                if p.eval(&hi).is_zero() {
                    (hi.clone(), hi)
                } else {
                    (lo, hi)
                }
            })
            .collect()
    }

    /// Shrink an isolating interval `(lo, hi]` (containing exactly one root)
    /// by bisection until its width is at most `max_width`.
    ///
    /// A root hit exactly by a bisection point is returned as `(r, r)`.
    pub fn refine_interval(
        &self,
        lo: &Ratio<BigInt>,
        hi: &Ratio<BigInt>,
        max_width: &Ratio<BigInt>,
    ) -> (Ratio<BigInt>, Ratio<BigInt>) {
        let two = Ratio::from_integer(BigInt::from(2));
        let mut lo = lo.clone();
        let mut hi = hi.clone();
        // Guard against pathological inputs: at most 512 halvings.
        for _ in 0..512 {
            if &hi - &lo <= *max_width || lo == hi {
                break;
            }
            let mid = (&lo + &hi) / &two;
            if self.chain.first().is_some_and(|p| p.eval(&mid).is_zero()) {
                return (mid.clone(), mid);
            }
            if self.count_roots_in(&lo, &mid) == 1 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        (lo, hi)
    }

    /// Count distinct real roots in the **closed** interval `[a, b]`.
    pub fn count_roots_in_closed(&self, a: &Ratio<BigInt>, b: &Ratio<BigInt>) -> usize {
        if a > b {
            return 0;
        }
        let open_right = self.count_roots_in(a, b);
        let at_a = self.chain.first().is_some_and(|p| p.eval(a).is_zero());
        open_right + usize::from(at_a)
    }

    /// Return the sign of the leading coefficient of the first (original)
    /// polynomial.  Useful for determining constant sign when there are
    /// no real roots.
    pub fn leading_sign_of_original(&self) -> i8 {
        self.chain.first().map_or(0, leading_sign)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Cauchy root bound: every root `z` of `p` satisfies
/// `|z| ≤ 1 + maxᵢ |aᵢ / aₙ|`.
pub(crate) fn cauchy_bound(p: &Poly) -> Ratio<BigInt> {
    let one = Ratio::from_integer(BigInt::from(1));
    let Some(n) = p.degree() else {
        return one;
    };
    if n == 0 {
        return one;
    }
    let lc = p.coeff(n);
    let mut max = Ratio::from_integer(BigInt::from(0));
    for i in 0..n {
        let r = (p.coeff(i) / &lc).abs();
        if r > max {
            max = r;
        }
    }
    max + one
}

/// Sign of the leading coefficient: +1, −1, or 0 (for zero poly).
fn leading_sign(p: &Poly) -> i8 {
    match p.leading_coeff() {
        None => 0,
        Some(c) if c.is_positive() => 1,
        Some(c) if c.is_negative() => -1,
        Some(_) => 0, // zero (shouldn't happen after normalisation)
    }
}

/// Count the number of sign changes in a sequence, ignoring zeros.
fn count_sign_changes(signs: &[i8]) -> usize {
    let nonzero: Vec<i8> = signs.iter().copied().filter(|&s| s != 0).collect();
    nonzero.windows(2).filter(|w| w[0] != w[1]).count()
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_rational::Ratio;

    fn r(n: i64) -> Ratio<BigInt> {
        Ratio::from_integer(BigInt::from(n))
    }

    /// x^2 - 1 = (x-1)(x+1) → 2 real roots
    #[test]
    fn sturm_x2_minus_1() {
        // coeffs: -1 + 0*x + 1*x^2
        let p = Poly::from_coeffs(vec![r(-1), r(0), r(1)]);
        let chain = SturmChain::new(&p);
        assert_eq!(chain.count_real_roots(), 2);
        assert!(!chain.has_no_real_roots());
    }

    /// x^2 + 1 → 0 real roots
    #[test]
    fn sturm_x2_plus_1() {
        let p = Poly::from_coeffs(vec![r(1), r(0), r(1)]);
        let chain = SturmChain::new(&p);
        assert_eq!(chain.count_real_roots(), 0);
        assert!(chain.has_no_real_roots());
    }

    /// x^3 - x = x(x-1)(x+1) → 3 real roots at -1, 0, 1
    #[test]
    fn sturm_x3_minus_x() {
        // coeffs: 0 - x + 0*x^2 + x^3
        let p = Poly::from_coeffs(vec![r(0), r(-1), r(0), r(1)]);
        let chain = SturmChain::new(&p);
        assert_eq!(chain.count_real_roots(), 3);
    }

    /// x → 1 real root at 0
    #[test]
    fn sturm_x() {
        let p = Poly::x();
        let chain = SturmChain::new(&p);
        assert_eq!(chain.count_real_roots(), 1);
    }

    /// count_roots_in for sub-intervals of x^3 - x
    #[test]
    fn count_roots_in_subintervals() {
        let p = Poly::from_coeffs(vec![r(0), r(-1), r(0), r(1)]);
        let chain = SturmChain::new(&p);

        // (-2, 2] should contain all 3 roots: -1, 0, 1
        assert_eq!(chain.count_roots_in(&r(-2), &r(2)), 3);

        // (-2, -1/2] should contain 1 root: -1
        let neg_half = Ratio::new(BigInt::from(-1), BigInt::from(2));
        assert_eq!(chain.count_roots_in(&r(-2), &neg_half), 1);

        // (-1/2, 1/2] should contain 1 root: 0
        let half = Ratio::new(BigInt::from(1), BigInt::from(2));
        assert_eq!(chain.count_roots_in(&neg_half, &half), 1);

        // (1/2, 2] should contain 1 root: 1
        assert_eq!(chain.count_roots_in(&half, &r(2)), 1);

        // (2, 10] should contain 0 roots
        assert_eq!(chain.count_roots_in(&r(2), &r(10)), 0);
    }

    /// isolate_roots_in for x^3 - x in (-10, 10)
    #[test]
    fn isolate_roots_x3_minus_x() {
        let p = Poly::from_coeffs(vec![r(0), r(-1), r(0), r(1)]);
        let chain = SturmChain::new(&p);
        let intervals = chain.isolate_roots_in(&r(-10), &r(10), 50);
        assert_eq!(intervals.len(), 3, "should isolate 3 roots");
        // Each interval should contain exactly one root
        for (lo, hi) in &intervals {
            assert_eq!(chain.count_roots_in(lo, hi), 1);
        }
    }

    /// isolate_roots_in for x^2 + 1 → no roots
    #[test]
    fn isolate_roots_no_real() {
        let p = Poly::from_coeffs(vec![r(1), r(0), r(1)]);
        let chain = SturmChain::new(&p);
        let intervals = chain.isolate_roots_in(&r(-100), &r(100), 50);
        assert!(intervals.is_empty());
    }

    /// Constant polynomial → no roots
    #[test]
    fn sturm_constant() {
        let p = Poly::from_int(5);
        let chain = SturmChain::new(&p);
        assert_eq!(chain.count_real_roots(), 0);
        assert!(chain.has_no_real_roots());
    }

    /// Zero polynomial
    #[test]
    fn sturm_zero() {
        let p = Poly::zero();
        let chain = SturmChain::new(&p);
        assert_eq!(chain.count_real_roots(), 0);
    }

    /// x^2 - 2 → 2 real roots (irrational)
    #[test]
    fn sturm_x2_minus_2() {
        let p = Poly::from_coeffs(vec![r(-2), r(0), r(1)]);
        let chain = SturmChain::new(&p);
        assert_eq!(chain.count_real_roots(), 2);
    }

    /// Repeated roots: (x-1)^2 = x^2 - 2x + 1
    /// square_free_part removes the multiplicity → 1 distinct root
    #[test]
    fn sturm_repeated_root() {
        let p = Poly::from_coeffs(vec![r(1), r(-2), r(1)]); // (x-1)^2
        let chain = SturmChain::new(&p);
        assert_eq!(chain.count_real_roots(), 1);
    }

    /// isolate_all_real_roots on x^3 - x with exact rational roots
    #[test]
    fn isolate_all_x3_minus_x() {
        let p = Poly::from_coeffs(vec![r(0), r(-1), r(0), r(1)]);
        let chain = SturmChain::new(&p);
        let iv = chain.isolate_all_real_roots();
        assert_eq!(iv.len(), 3);
        // Sorted and each contains exactly one root.
        for w in iv.windows(2) {
            assert!(w[0].1 <= w[1].0);
        }
        for (lo, hi) in &iv {
            if lo == hi {
                assert!(p.eval(lo).is_zero());
            } else {
                assert_eq!(chain.count_roots_in(lo, hi), 1);
            }
        }
    }

    /// isolate_all_real_roots on x^2 - 2: two irrational roots
    #[test]
    fn isolate_all_x2_minus_2() {
        let p = Poly::from_coeffs(vec![r(-2), r(0), r(1)]);
        let chain = SturmChain::new(&p);
        let iv = chain.isolate_all_real_roots();
        assert_eq!(iv.len(), 2);
        assert!(iv[0].1 <= r(0) && iv[1].0 >= r(0));
        // Refinement tightens the brackets around ±√2 ≈ ±1.414.
        let width = Ratio::new(BigInt::from(1), BigInt::from(100));
        let (lo, hi) = chain.refine_interval(&iv[1].0, &iv[1].1, &width);
        assert!(&hi - &lo <= width);
        assert!(lo < Ratio::new(BigInt::from(1415), BigInt::from(1000)));
        assert!(hi > Ratio::new(BigInt::from(1414), BigInt::from(1000)));
    }

    /// count_roots_in_closed includes the left endpoint
    #[test]
    fn closed_interval_count() {
        let p = Poly::from_coeffs(vec![r(0), r(-1), r(0), r(1)]); // roots -1, 0, 1
        let chain = SturmChain::new(&p);
        assert_eq!(chain.count_roots_in_closed(&r(-1), &r(1)), 3);
        assert_eq!(chain.count_roots_in(&r(-1), &r(1)), 2);
        assert_eq!(chain.count_roots_in_closed(&r(0), &r(0)), 1);
        assert_eq!(chain.count_roots_in_closed(&r(2), &r(1)), 0);
    }

    /// leading_sign_of_original
    #[test]
    fn leading_sign_positive() {
        let p = Poly::from_coeffs(vec![r(1), r(0), r(1)]); // x^2 + 1
        let chain = SturmChain::new(&p);
        assert_eq!(chain.leading_sign_of_original(), 1);
    }

    #[test]
    fn leading_sign_negative() {
        let p = Poly::from_coeffs(vec![r(-1), r(0), r(-1)]); // -x^2 - 1
        let chain = SturmChain::new(&p);
        assert_eq!(chain.leading_sign_of_original(), -1);
    }
}
