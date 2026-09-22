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
//!
//! Because the count is over the half-open `(a, b]`, every isolating
//! interval produced by bisection here is [`Interval::left_open`] — the
//! root may sit on the upper endpoint but never on the lower one — except
//! that a root hit exactly by a bisection point is reported as the closed
//! singleton [`Interval::point`].

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{Signed, Zero};

use crate::base::interval::{Interval, IntervalKind};
use crate::base::numeric::Q;
use crate::poly::Poly;
use crate::poly::zpoly::{integer_scaled, powers, pseudo_rem_pos, z_primitive};

/// A Sturm chain built from a polynomial.
#[derive(Debug, Clone)]
pub(crate) struct SturmChain {
    /// The polynomials P_0, P_1, …, P_k forming the chain.
    chain: Vec<Poly>,
    /// `chain[i]` multiplied by a positive integer so that every
    /// coefficient is an integer (ascending degree).  Signs are unchanged,
    /// so every sign query is answered from these with integer Horner
    /// evaluation instead of `Ratio<BigInt>` arithmetic (which would
    /// normalise a gcd after every operation).
    int_chain: Vec<Vec<BigInt>>,
}

#[allow(dead_code)] // Used indirectly via Ex::count_real_roots() bridge; will be exposed publicly later
impl SturmChain {
    /// Build a Sturm chain from polynomial `p`.
    ///
    /// The input is first made square-free so that the chain correctly
    /// counts *distinct* real roots.
    ///
    /// The remainders are computed in `ℤ[x]` as primitive pseudo-remainders
    /// (the primitive PRS), which yields exactly the polynomials
    /// `-rem(P_{i-1}, P_i).primitive_part()` of the definition — a
    /// pseudo-remainder is a positive multiple of the remainder when the
    /// leading coefficient is taken in absolute value, and the primitive
    /// part is invariant under positive scaling — without the gcd that
    /// `Ratio<BigInt>` arithmetic performs after every operation.
    pub fn new(p: &Poly) -> Self {
        if p.is_zero() {
            return Self::from_chain(vec![Poly::zero()]);
        }

        let p0 = p.square_free_part();
        let p1 = p0.derivative();

        if p1.is_zero() {
            // Constant (degree 0) — no roots.
            return Self::from_chain(vec![p0]);
        }

        let mut chain = vec![p0.clone(), p1.clone()];
        let mut int_chain = vec![integer_scaled(p0.coeffs()), integer_scaled(p1.coeffs())];

        loop {
            let n = int_chain.len();
            let rem = pseudo_rem_pos(&int_chain[n - 2], &int_chain[n - 1]);
            if rem.is_empty() {
                break;
            }
            let next = z_primitive(&rem.into_iter().map(|c| -c).collect::<Vec<_>>());
            chain.push(Poly::from_coeffs(
                next.iter().cloned().map(Ratio::from_integer).collect(),
            ));
            int_chain.push(next);
        }

        SturmChain { chain, int_chain }
    }

    fn from_chain(chain: Vec<Poly>) -> Self {
        let int_chain = chain.iter().map(|p| integer_scaled(p.coeffs())).collect();
        SturmChain { chain, int_chain }
    }

    /// Largest degree among the chain polynomials.
    fn max_degree(&self) -> usize {
        self.int_chain
            .iter()
            .map(|c| c.len().saturating_sub(1))
            .max()
            .unwrap_or(0)
    }

    /// The sign of every chain polynomial at `x` (`+1`, `-1` or `0`), in
    /// chain order.  `signs[0]` is the sign of the square-free part of the
    /// original polynomial, so `signs[0] == 0` iff `x` is a root.
    fn signs_at(&self, x: &Ratio<BigInt>) -> Vec<i8> {
        let (a, b) = numer_denom(x);
        let b_pows = powers(&b, self.max_degree());
        self.int_chain
            .iter()
            .map(|c| int_sign_at(c, &a, &b, &b_pows))
            .collect()
    }

    /// Is `x` a root of the (square-free part of the) polynomial?
    fn is_root(&self, x: &Ratio<BigInt>) -> bool {
        let Some(p0) = self.int_chain.first() else {
            return false;
        };
        let (a, b) = numer_denom(x);
        let b_pows = powers(&b, p0.len().saturating_sub(1));
        int_sign_at(p0, &a, &b, &b_pows) == 0
    }

    /// Count sign variations when each polynomial is evaluated at `x`.
    ///
    /// Zeros are skipped (as per the standard Sturm convention).
    pub fn sign_variations_at(&self, x: &Ratio<BigInt>) -> usize {
        count_sign_changes(&self.signs_at(x))
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

    /// Isolate the real roots in `(a, b]` into disjoint half-open
    /// sub-intervals `(lo, hi]` by bisection.
    ///
    /// Every returned interval is [`Interval::left_open`] — the count that
    /// drives the bisection is [`count_roots_in`](Self::count_roots_in),
    /// i.e. over `(lo, hi]` — and contains exactly one root, unless
    /// `max_depth` bisection levels were exhausted first, in which case an
    /// interval may still hold several.  The output is sorted left to right.
    pub fn isolate_roots_in(&self, a: &Q, b: &Q, max_depth: u32) -> Vec<Interval<Q>> {
        // Explicit stack (no recursion); intervals are pushed right-first so
        // that the output comes out sorted left to right.  Each entry
        // carries the sign variations at its endpoints, so a bisection
        // point is evaluated once and shared by both halves.
        let two = Ratio::from_integer(BigInt::from(2));
        let mut result = Vec::new();
        let mut stack: Vec<Cell> = vec![Cell {
            lo: a.clone(),
            hi: b.clone(),
            depth: max_depth,
            var_lo: self.sign_variations_at(a),
            var_hi: self.sign_variations_at(b),
        }];
        while let Some(cell) = stack.pop() {
            let n = cell.var_lo.saturating_sub(cell.var_hi);
            if n == 0 {
                continue;
            }
            if n == 1 || cell.depth == 0 {
                result.push(Interval::left_open(cell.lo, cell.hi));
                continue;
            }
            let mid = (&cell.lo + &cell.hi) / &two;
            let var_mid = self.sign_variations_at(&mid);
            stack.push(Cell {
                lo: mid.clone(),
                hi: cell.hi,
                depth: cell.depth - 1,
                var_lo: var_mid,
                var_hi: cell.var_hi,
            });
            stack.push(Cell {
                lo: cell.lo,
                hi: mid,
                depth: cell.depth - 1,
                var_lo: cell.var_lo,
                var_hi: var_mid,
            });
        }
        result
    }

    /// Isolate **all** distinct real roots of the polynomial into disjoint
    /// rational intervals, each containing exactly one root.
    ///
    /// The search starts from the Cauchy root bound.  Each returned
    /// interval is either
    ///
    /// * [`Interval::left_open`] `(lo, hi]` with `lo < hi` and exactly one
    ///   root (the guarantee of [`isolate_roots_in`](Self::isolate_roots_in)),
    ///   whose upper endpoint is *not* a root, or
    /// * the closed singleton [`Interval::point`] `[r, r]` when the one root
    ///   of a `(lo, r]` cell is `r` itself (a bisection point hit it
    ///   exactly).
    ///
    /// Intervals are sorted.
    pub fn isolate_all_real_roots(&self) -> Vec<Interval<Q>> {
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
            .map(|iv| {
                if self.is_root(&iv.upper) {
                    Interval::point(iv.upper)
                } else {
                    iv
                }
            })
            .collect()
    }

    /// Shrink an isolating interval by bisection until its width is at
    /// most `max_width`.
    ///
    /// `iv` must be an interval as produced by
    /// [`isolate_all_real_roots`](Self::isolate_all_real_roots): either
    /// `(lo, hi]` containing exactly one root, or a point `[r, r]`, which
    /// is returned unchanged.  Each bisection keeps the half whose
    /// [`count_roots_in`](Self::count_roots_in) is one, so the result is
    /// again `(lo, hi]` — or the closed singleton [`Interval::point`] `[r, r]`
    /// when a bisection point lands exactly on the root.
    pub fn refine_interval(&self, iv: &Interval<Q>, max_width: &Q) -> Interval<Q> {
        let two = Ratio::from_integer(BigInt::from(2));
        let mut lo = iv.lower.clone();
        let mut hi = iv.upper.clone();
        let mut kind = iv.kind;
        // Sign variations at `lo`, computed on first use and carried across
        // bisections (when the root is in the right half `lo` moves to the
        // midpoint, whose variations were just computed).
        let mut var_lo: Option<usize> = None;
        // Guard against pathological inputs: at most 512 halvings.
        for _ in 0..512 {
            if &hi - &lo <= *max_width || lo == hi {
                break;
            }
            let mid = (&lo + &hi) / &two;
            let signs_mid = self.signs_at(&mid);
            if signs_mid.first() == Some(&0) {
                return Interval::point(mid);
            }
            let var_mid = count_sign_changes(&signs_mid);
            let at_lo = *var_lo.get_or_insert_with(|| self.sign_variations_at(&lo));
            // Exactly one root in `(lo, mid]`?
            if at_lo.saturating_sub(var_mid) == 1 {
                hi = mid;
            } else {
                lo = mid;
                var_lo = Some(var_mid);
            }
            // After a bisection the root is located in `(lo, hi]`.
            kind = IntervalKind::LeftOpen;
        }
        Interval {
            lower: lo,
            upper: hi,
            kind,
        }
    }

    /// Count distinct real roots in the **closed** interval `[a, b]`.
    pub fn count_roots_in_closed(&self, a: &Ratio<BigInt>, b: &Ratio<BigInt>) -> usize {
        if a > b {
            return 0;
        }
        let open_right = self.count_roots_in(a, b);
        open_right + usize::from(self.is_root(a))
    }

    /// Return the sign of the leading coefficient of the first (original)
    /// polynomial.  Useful for determining constant sign when there are
    /// no real roots.
    pub fn leading_sign_of_original(&self) -> i8 {
        self.chain.first().map_or(0, leading_sign)
    }
}

/// A bisection cell of [`SturmChain::isolate_roots_in`] with the sign
/// variations at its endpoints.
struct Cell {
    lo: Q,
    hi: Q,
    depth: u32,
    var_lo: usize,
    var_hi: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// `x = a / b` with `b > 0`.
fn numer_denom(x: &Ratio<BigInt>) -> (BigInt, BigInt) {
    if x.denom().is_negative() {
        (-x.numer(), -x.denom())
    } else {
        (x.numer().clone(), x.denom().clone())
    }
}

/// Sign of the integer polynomial `c` (ascending) at `a / b` with `b > 0`:
/// the sign of `b^n · c(a/b) = Σ cᵢ aⁱ b^{n−i}`, by Horner's rule in `ℤ`.
/// `b_pows[j] = b^j` are precomputed powers shared between the chain
/// polynomials (any missing power is computed on the spot).
fn int_sign_at(c: &[BigInt], a: &BigInt, b: &BigInt, b_pows: &[BigInt]) -> i8 {
    let Some((lead, rest)) = c.split_last() else {
        return 0;
    };
    let n = rest.len();
    let mut v = lead.clone();
    for (i, ci) in rest.iter().enumerate().rev() {
        v *= a;
        if !ci.is_zero() {
            match b_pows.get(n - i) {
                Some(bp) => v += ci * bp,
                None => v += ci * b.pow((n - i) as u32),
            }
        }
    }
    match v.sign() {
        num_bigint::Sign::Plus => 1,
        num_bigint::Sign::Minus => -1,
        num_bigint::Sign::NoSign => 0,
    }
}

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
    use num_traits::ToPrimitive;

    fn r(n: i64) -> Ratio<BigInt> {
        Ratio::from_integer(BigInt::from(n))
    }

    /// `Σ_{j=k}^{n} C(n,j) x^j (1−x)^{n−j} − 1/denom`: the Clopper–Pearson
    /// tail polynomial, whose Cauchy bound is huge (`> 6·10⁷` at `n = 40`)
    /// while all roots lie in `|z| < 1.5`.
    fn binomial_tail(n: usize, k: usize, denom: i64) -> Poly {
        let mut acc = Poly::zero();
        let one_minus_x = Poly::from_coeffs(vec![r(1), r(-1)]);
        let x = Poly::x();
        for j in k..=n {
            let mut c = r(1);
            for i in 0..j {
                c *= Ratio::new(BigInt::from(n - i), BigInt::from(i + 1));
            }
            let mut term = Poly::from_coeffs(vec![c]);
            for _ in 0..j {
                term = &term * &x;
            }
            for _ in 0..(n - j) {
                term = &term * &one_minus_x;
            }
            acc = &acc + &term;
        }
        &acc - &Poly::from_coeffs(vec![Ratio::new(BigInt::from(1), BigInt::from(denom))])
    }

    /// Isolating and refining the roots of the degree-40 tail polynomial
    /// bisects a width-`10⁸` Cauchy interval down to `1/1024` — about 36
    /// halvings per root — and used to take 16 s with rational Horner
    /// evaluation; with integer signs it is milliseconds.
    #[test]
    fn degree_40_tail_isolation_is_fast_and_correct() {
        let f = binomial_tail(40, 12, 40);
        assert_eq!(f.degree(), Some(40));
        let start = std::time::Instant::now();
        let chain = SturmChain::new(&f);
        let ivs = chain.isolate_all_real_roots();
        assert_eq!(ivs.len(), 2, "{ivs:?}");
        let width = Ratio::new(BigInt::from(1), BigInt::from(1024));
        let refined: Vec<Interval<Q>> = ivs
            .iter()
            .map(|iv| chain.refine_interval(iv, &width))
            .collect();
        assert!(start.elapsed().as_secs() < 5, "took {:?}", start.elapsed());
        for (iv, fine) in ivs.iter().zip(&refined) {
            assert!(&fine.upper - &fine.lower <= width);
            assert!(iv.lower <= fine.lower && fine.upper <= iv.upper);
            assert_eq!(chain.count_roots_in(&fine.lower, &fine.upper), 1);
        }
        // The root in (0, 1) is the 2.5% Clopper–Pearson lower bound for
        // 12 successes in 40 trials: 0.16562720439… (scipy.stats.beta.ppf).
        let lo = refined[1].lower.to_f64().unwrap_or(f64::NAN);
        let hi = refined[1].upper.to_f64().unwrap_or(f64::NAN);
        assert!(
            lo <= 0.1656272043932356 && 0.1656272043932356 <= hi,
            "[{lo}, {hi}]"
        );
    }

    /// A deterministic pseudo-random polynomial with small integer
    /// coefficients (some repeated factors when `square` is set).
    fn pseudo_random_poly(seed: u64, degree: usize, square: bool) -> Poly {
        let mut state = seed;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 19) as i64 - 9
        };
        let mut coeffs: Vec<Ratio<BigInt>> = (0..=degree).map(|_| r(next())).collect();
        if coeffs[degree].is_zero() {
            coeffs[degree] = r(1);
        }
        let p = Poly::from_coeffs(coeffs);
        if square {
            let q = Poly::from_coeffs(vec![r(next()), r(next()), r(1)]);
            &(&p * &q) * &q
        } else {
            p
        }
    }

    /// The integer primitive-PRS chain equals the definition computed over
    /// `ℚ`: `P_{i+1} = -rem(P_{i-1}, P_i).primitive_part()`.
    #[test]
    fn integer_chain_matches_rational_definition() {
        for seed in 1..=12u64 {
            let p = pseudo_random_poly(seed, 8 + (seed as usize % 5), seed % 3 == 0);
            let chain = SturmChain::new(&p);
            let p0 = p.square_free_part();
            assert_eq!(chain.chain[0], p0, "seed {seed}: square-free part");
            let mut expected = vec![p0.clone(), p0.derivative()];
            loop {
                let n = expected.len();
                let rem = expected[n - 2].rem(&expected[n - 1]);
                if rem.is_zero() {
                    break;
                }
                expected.push((-&rem).primitive_part());
            }
            assert_eq!(chain.chain, expected, "seed {seed}");
            // The integer copies have the sign of the rational entries.
            for (q, z) in chain.chain.iter().zip(&chain.int_chain) {
                let x = Ratio::new(BigInt::from(-7), BigInt::from(3));
                let zq = Poly::from_coeffs(z.iter().cloned().map(Ratio::from_integer).collect());
                assert_eq!(q.eval(&x).is_positive(), zq.eval(&x).is_positive());
                assert_eq!(q.eval(&x).is_zero(), zq.eval(&x).is_zero());
            }
        }
    }

    /// Sign queries through the integer chain agree with rational
    /// evaluation of the chain polynomials.
    #[test]
    fn integer_sign_evaluation_matches_rational() {
        let p = pseudo_random_poly(5, 11, true);
        let chain = SturmChain::new(&p);
        let xs = [
            r(0),
            r(3),
            r(-2),
            Ratio::new(BigInt::from(5), BigInt::from(7)),
            Ratio::new(BigInt::from(-1234567), BigInt::from(89)),
            Ratio::new(BigInt::from(1), BigInt::from(1) << 40),
        ];
        for x in &xs {
            let expected: Vec<i8> = chain
                .chain
                .iter()
                .map(|q| {
                    let v = q.eval(x);
                    if v.is_positive() {
                        1
                    } else if v.is_negative() {
                        -1
                    } else {
                        0
                    }
                })
                .collect();
            assert_eq!(chain.signs_at(x), expected, "at {x}");
        }
        // A root of `p` is reported as a root.
        let root = Ratio::new(BigInt::from(2), BigInt::from(1));
        let q = &p * &Poly::from_coeffs(vec![r(-2), r(1)]);
        assert!(SturmChain::new(&q).is_root(&root));
        assert!(!SturmChain::new(&q).is_root(&r(1)) || q.eval(&r(1)).is_zero());
    }

    /// The square-free part the chain starts from (`Poly::square_free_part`,
    /// whose gcd runs through `ℤ[x]`) equals `p / gcd(p, p')` with the gcd
    /// computed by Euclid over ℚ.  Regression for the former private
    /// integer-gcd copy of `square_free_part` in this module.
    #[test]
    fn integer_square_free_part_matches_generic() {
        fn euclid_square_free_part(p: &Poly) -> Poly {
            let dp = p.derivative();
            if dp.is_zero() {
                return p.clone();
            }
            p.div(&Poly::gcd_euclid(p, &dp))
        }
        for seed in 1..=10u64 {
            let p = pseudo_random_poly(seed, 6 + seed as usize % 4, true);
            assert_eq!(
                euclid_square_free_part(&p),
                p.square_free_part(),
                "seed {seed}"
            );
            assert_eq!(SturmChain::new(&p).chain[0], p.square_free_part());
            let q = pseudo_random_poly(seed + 100, 9, false);
            assert_eq!(
                euclid_square_free_part(&q),
                q.square_free_part(),
                "seed {seed}"
            );
        }
        // Rational coefficients too.
        let half = Ratio::new(BigInt::from(1), BigInt::from(2));
        let p = Poly::from_coeffs(vec![half.clone(), r(3), half, r(1)]);
        let p2 = &p * &p;
        assert_eq!(euclid_square_free_part(&p2), p2.square_free_part());
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
        // Each interval is `(lo, hi]` and contains exactly one root.
        for iv in &intervals {
            assert_eq!(iv.kind, IntervalKind::LeftOpen);
            assert_eq!(chain.count_roots_in(&iv.lower, &iv.upper), 1);
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
            assert!(w[0].upper <= w[1].lower);
        }
        for i in &iv {
            if i.is_point() {
                assert!(p.eval(&i.lower).is_zero());
            } else {
                assert_eq!(i.kind, IntervalKind::LeftOpen);
                assert_eq!(chain.count_roots_in(&i.lower, &i.upper), 1);
                assert!(!p.eval(&i.upper).is_zero());
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
        assert!(iv[0].upper <= r(0) && iv[1].lower >= r(0));
        // Refinement tightens the brackets around ±√2 ≈ ±1.414 and keeps
        // the `(lo, hi]` kind (√2 is irrational, so no exact hit).
        let width = Ratio::new(BigInt::from(1), BigInt::from(100));
        let refined = chain.refine_interval(&iv[1], &width);
        assert_eq!(refined.kind, IntervalKind::LeftOpen);
        assert!(refined.width() <= width);
        assert!(refined.lower < Ratio::new(BigInt::from(1415), BigInt::from(1000)));
        assert!(refined.upper > Ratio::new(BigInt::from(1414), BigInt::from(1000)));
        // A point stays a point.
        let zero = Interval::point(r(0));
        assert_eq!(chain.refine_interval(&zero, &width), zero);
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
