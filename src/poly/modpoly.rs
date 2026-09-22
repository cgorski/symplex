//! Dense univariate polynomials over a *value-level* ring — one whose
//! parameters (a modulus, a field size) live in a value rather than in a
//! type.
//!
//! [`GenPoly<C>`](super::generic::GenPoly) needs `C::zero()` and
//! `C::one()` with no arguments, so a coefficient type has to carry its
//! ring in the type.  Arithmetic modulo a *runtime* prime `p` cannot do
//! that: `𝔽ₚ[x]` for the primes tried by Berlekamp–Zassenhaus and for the
//! root finding in `ntheory::polynomial_congruence` used to be hand-rolled
//! twice on `Vec<u64>`.  This module is the single spelling of that ring:
//!
//! - [`RingOps`] — the operations of a commutative ring given as methods
//!   on a ring *value* (`&self` carries `p`), with an optional inverse so
//!   that fields and non-fields share one trait;
//! - [`Fp64`] — the prime field `𝔽ₚ` on `u64` residues for `p < 2⁶³`;
//! - [`PolyIn<R>`] — a polynomial with coefficients in `R::El` together
//!   with the ring it lives in: addition, multiplication, Euclidean
//!   division, monic gcd, extended gcd, derivative, square-freeness,
//!   modular exponentiation, and (over [`Fp64`]) `p`-th roots and root
//!   finding by Cantor–Zassenhaus.
//!
//! Coefficients are stored in ascending degree order with no trailing
//! zeros; the zero polynomial has an empty coefficient vector.  Every
//! [`Fp64`] method is `#[inline]`, so the generic loops monomorphise to
//! the same machine code as the hand-rolled `u64` versions they replace.
//!
//! The randomised splitting in [`PolyIn::roots`] draws from the crate's
//! pinned [`XorShift64Star`](crate::base::rng::XorShift64Star): which
//! factor a split finds first depends on that stream, so the iteration
//! order here is an output contract, not an implementation detail.

use std::fmt;

use num_bigint::BigUint;

use crate::base::rng::XorShift64Star as XorShift;

// ═══════════════════════════════════════════════════════════════════════════
// The ring trait
// ═══════════════════════════════════════════════════════════════════════════

/// A commutative ring whose operations are methods on a ring *value*.
///
/// Unlike [`Ring`](super::traits::Ring), `zero`/`one` take `&self`, so the
/// modulus (or any other parameter) may be decided at run time.  `inv`
/// returns `None` for non-units, which lets a field and a non-field ring
/// share the trait; [`PolyIn`]'s division-based algorithms need every
/// non-zero leading coefficient they meet to be a unit.
pub trait RingOps: Clone + fmt::Debug {
    /// Element representation.
    type El: Clone + PartialEq + fmt::Debug;

    /// The additive identity.
    fn zero(&self) -> Self::El;
    /// The multiplicative identity.
    fn one(&self) -> Self::El;
    /// The image of the integer `n`.
    fn embed_u64(&self, n: u64) -> Self::El;
    /// Is `a` the additive identity?
    fn is_zero(&self, a: &Self::El) -> bool;
    /// Is `a` the multiplicative identity?
    fn is_one(&self, a: &Self::El) -> bool;
    /// `a + b`.
    fn add(&self, a: &Self::El, b: &Self::El) -> Self::El;
    /// `a − b`.
    fn sub(&self, a: &Self::El, b: &Self::El) -> Self::El;
    /// `a · b`.
    fn mul(&self, a: &Self::El, b: &Self::El) -> Self::El;
    /// `−a`.
    fn neg(&self, a: &Self::El) -> Self::El;
    /// `a⁻¹`, or `None` when `a` is not a unit.
    fn inv(&self, a: &Self::El) -> Option<Self::El>;
}

// ═══════════════════════════════════════════════════════════════════════════
// 𝔽ₚ on u64
// ═══════════════════════════════════════════════════════════════════════════

/// The prime field `𝔽ₚ` for an odd prime `p < 2⁶³`, elements as `u64`
/// residues in `[0, p)`.
///
/// Products go through `u128` only when `p ≥ 2³²`; below that the `u64`
/// product cannot overflow and a native remainder is used.  Inverses are
/// by Fermat (`a^(p−2)`).  The caller guarantees primality — nothing here
/// checks it — exactly as the hand-rolled predecessors did.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fp64 {
    p: u64,
}

impl Fp64 {
    /// The field with `p` elements.
    ///
    /// `p` must be an odd prime below `2⁶³` (so that `a + b` never
    /// overflows for reduced `a`, `b`).
    #[inline]
    pub const fn new(p: u64) -> Self {
        debug_assert!(p >= 3);
        debug_assert!(p < (1u64 << 63));
        Fp64 { p }
    }

    /// The modulus.
    #[inline]
    pub const fn modulus(self) -> u64 {
        self.p
    }

    /// `x mod p`.
    #[inline]
    pub const fn reduce(self, x: u64) -> u64 {
        x % self.p
    }

    /// `a · b mod p` for reduced `a`, `b`.
    #[inline]
    pub const fn mul_reduced(self, a: u64, b: u64) -> u64 {
        if self.p <= u32::MAX as u64 {
            (a * b) % self.p
        } else {
            ((a as u128 * b as u128) % self.p as u128) as u64
        }
    }

    /// `base^exp mod p` by binary exponentiation.
    pub fn pow(self, base: u64, mut exp: u64) -> u64 {
        let mut result = 1 % self.p;
        let mut base = base % self.p;
        while exp > 0 {
            if exp & 1 == 1 {
                result = self.mul_reduced(result, base);
            }
            base = self.mul_reduced(base, base);
            exp >>= 1;
        }
        result
    }
}

impl RingOps for Fp64 {
    type El = u64;

    #[inline]
    fn zero(&self) -> u64 {
        0
    }

    #[inline]
    fn one(&self) -> u64 {
        1
    }

    #[inline]
    fn embed_u64(&self, n: u64) -> u64 {
        n % self.p
    }

    #[inline]
    fn is_zero(&self, a: &u64) -> bool {
        *a == 0
    }

    #[inline]
    fn is_one(&self, a: &u64) -> bool {
        *a == 1
    }

    #[inline]
    fn add(&self, a: &u64, b: &u64) -> u64 {
        let s = a + b;
        if s >= self.p { s - self.p } else { s }
    }

    #[inline]
    fn sub(&self, a: &u64, b: &u64) -> u64 {
        if a >= b { a - b } else { a + self.p - b }
    }

    #[inline]
    fn mul(&self, a: &u64, b: &u64) -> u64 {
        self.mul_reduced(*a, *b)
    }

    #[inline]
    fn neg(&self, a: &u64) -> u64 {
        if *a == 0 { 0 } else { self.p - a }
    }

    #[inline]
    fn inv(&self, a: &u64) -> Option<u64> {
        let a = a % self.p;
        if a == 0 {
            None
        } else {
            Some(self.pow(a, self.p - 2))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// The polynomial
// ═══════════════════════════════════════════════════════════════════════════

/// Result of [`PolyIn::div_rem`]: `a = quotient · b + remainder` with
/// `deg remainder < deg b`.
#[derive(Clone, Debug, PartialEq)]
pub struct DivRem<R: RingOps> {
    /// The quotient.
    pub quotient: PolyIn<R>,
    /// The remainder, of degree below the divisor's.
    pub remainder: PolyIn<R>,
}

/// Result of [`PolyIn::extended_gcd`]: `u · a + v · b = gcd`, `gcd` monic.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtendedGcd<R: RingOps> {
    /// Bézout cofactor of the first argument.
    pub u: PolyIn<R>,
    /// Bézout cofactor of the second argument.
    pub v: PolyIn<R>,
    /// The monic greatest common divisor.
    pub gcd: PolyIn<R>,
}

/// A dense univariate polynomial over the ring value `ring`.
///
/// Coefficients ascend in degree and the last one is non-zero (the zero
/// polynomial is empty).  Two polynomials compare equal when their
/// coefficient vectors do; operands of a binary operation are assumed to
/// share the ring (the left operand's ring is used).
#[derive(Clone, Debug, PartialEq)]
pub struct PolyIn<R: RingOps> {
    ring: R,
    coeffs: Vec<R::El>,
}

impl<R: RingOps> PolyIn<R> {
    /// The zero polynomial.
    pub fn zero(ring: R) -> Self {
        PolyIn {
            ring,
            coeffs: Vec::new(),
        }
    }

    /// The constant polynomial `1`.
    pub fn one(ring: R) -> Self {
        let one = ring.one();
        PolyIn {
            ring,
            coeffs: vec![one],
        }
    }

    /// The constant polynomial `c`.
    pub fn constant(ring: R, c: R::El) -> Self {
        if ring.is_zero(&c) {
            return Self::zero(ring);
        }
        PolyIn {
            ring,
            coeffs: vec![c],
        }
    }

    /// The variable `x`.
    pub fn x(ring: R) -> Self {
        let (zero, one) = (ring.zero(), ring.one());
        PolyIn {
            ring,
            coeffs: vec![zero, one],
        }
    }

    /// From coefficients in ascending degree order (already reduced into
    /// the ring's canonical representation); trailing zeros are dropped.
    pub fn from_coeffs(ring: R, coeffs: Vec<R::El>) -> Self {
        let mut p = PolyIn { ring, coeffs };
        p.normalize();
        p
    }

    /// The ring the coefficients live in.
    #[inline]
    pub fn ring(&self) -> &R {
        &self.ring
    }

    /// Coefficients in ascending degree order, no trailing zeros.
    #[inline]
    pub fn coeffs(&self) -> &[R::El] {
        &self.coeffs
    }

    /// The coefficient vector (ascending, no trailing zeros).
    pub fn into_coeffs(self) -> Vec<R::El> {
        self.coeffs
    }

    /// Coefficient of `xⁱ` (zero beyond the degree).
    pub fn coeff(&self, i: usize) -> R::El {
        self.coeffs
            .get(i)
            .cloned()
            .unwrap_or_else(|| self.ring.zero())
    }

    /// Is this the zero polynomial?
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Degree, or `None` for the zero polynomial.
    #[inline]
    pub fn degree(&self) -> Option<usize> {
        self.coeffs.len().checked_sub(1)
    }

    /// Leading coefficient, or `None` for the zero polynomial.
    #[inline]
    pub fn leading_coeff(&self) -> Option<&R::El> {
        self.coeffs.last()
    }

    /// Is the leading coefficient `1`?  (`false` for the zero polynomial.)
    pub fn is_monic(&self) -> bool {
        self.coeffs.last().is_some_and(|c| self.ring.is_one(c))
    }

    fn normalize(&mut self) {
        while self.coeffs.last().is_some_and(|c| self.ring.is_zero(c)) {
            self.coeffs.pop();
        }
    }

    fn with(&self, coeffs: Vec<R::El>) -> Self {
        Self::from_coeffs(self.ring.clone(), coeffs)
    }

    /// `self + rhs`.
    #[must_use]
    pub fn add(&self, rhs: &Self) -> Self {
        let n = self.coeffs.len().max(rhs.coeffs.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(match (self.coeffs.get(i), rhs.coeffs.get(i)) {
                (Some(a), Some(b)) => self.ring.add(a, b),
                (Some(a), None) => a.clone(),
                (None, Some(b)) => b.clone(),
                (None, None) => self.ring.zero(),
            });
        }
        self.with(out)
    }

    /// `self − rhs`.
    #[must_use]
    pub fn sub(&self, rhs: &Self) -> Self {
        let n = self.coeffs.len().max(rhs.coeffs.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(match (self.coeffs.get(i), rhs.coeffs.get(i)) {
                (Some(a), Some(b)) => self.ring.sub(a, b),
                (Some(a), None) => a.clone(),
                (None, Some(b)) => self.ring.neg(b),
                (None, None) => self.ring.zero(),
            });
        }
        self.with(out)
    }

    /// `−self`.
    #[must_use]
    pub fn neg(&self) -> Self {
        let out = self.coeffs.iter().map(|c| self.ring.neg(c)).collect();
        self.with(out)
    }

    /// `c · self`.
    #[must_use]
    pub fn scale(&self, c: &R::El) -> Self {
        let out = self.coeffs.iter().map(|a| self.ring.mul(a, c)).collect();
        self.with(out)
    }

    /// `self · rhs` (schoolbook).
    #[must_use]
    pub fn mul(&self, rhs: &Self) -> Self {
        if self.is_zero() || rhs.is_zero() {
            return Self::zero(self.ring.clone());
        }
        let ring = &self.ring;
        let mut out = vec![ring.zero(); self.coeffs.len() + rhs.coeffs.len() - 1];
        for (i, a) in self.coeffs.iter().enumerate() {
            if ring.is_zero(a) {
                continue;
            }
            for (j, b) in rhs.coeffs.iter().enumerate() {
                out[i + j] = ring.add(&out[i + j], &ring.mul(a, b));
            }
        }
        self.with(out)
    }

    /// `self · rhs mod m`.
    #[must_use]
    pub fn mul_mod(&self, rhs: &Self, m: &Self) -> Self {
        self.mul(rhs).rem(m)
    }

    /// Euclidean division `self = quotient · b + remainder` with
    /// `deg remainder < deg b`.
    ///
    /// Every caller passes a `b` whose leading coefficient is a unit (a
    /// factor, a non-zero gcd remainder, a modulus).  Should the zero
    /// polynomial — or a non-unit leading coefficient — ever arrive, the
    /// result is `(0, self)`, the one pair that still satisfies the
    /// identity, rather than a panic.
    #[must_use]
    pub fn div_rem(&self, b: &Self) -> DivRem<R> {
        let ring = &self.ring;
        let trivial = |remainder: Self| DivRem {
            quotient: Self::zero(ring.clone()),
            remainder,
        };
        let Some(db) = b.degree() else {
            return trivial(self.clone());
        };
        let Some(da) = self.degree() else {
            return trivial(Self::zero(ring.clone()));
        };
        if da < db {
            return trivial(self.clone());
        }
        let lc = &b.coeffs[db];
        let monic = ring.is_one(lc);
        let inv_lc = if monic {
            ring.one()
        } else {
            match ring.inv(lc) {
                Some(v) => v,
                None => return trivial(self.clone()),
            }
        };
        let mut rem = self.coeffs.clone();
        let mut quot = vec![ring.zero(); da - db + 1];
        while rem.len() > db {
            let dr = rem.len() - 1;
            let c = if monic {
                rem[dr].clone()
            } else {
                ring.mul(&rem[dr], &inv_lc)
            };
            let shift = dr - db;
            for (j, bj) in b.coeffs.iter().enumerate().take(db) {
                let sub = ring.mul(&c, bj);
                rem[shift + j] = ring.sub(&rem[shift + j], &sub);
            }
            quot[shift] = c;
            // The leading term cancels by construction (`c · lc = rem[dr]`);
            // dropping it unconditionally also keeps the loop finite should a
            // non-unit leading coefficient ever slip through.
            rem.truncate(dr);
            while rem.last().is_some_and(|c| ring.is_zero(c)) {
                rem.pop();
            }
        }
        DivRem {
            quotient: self.with(quot),
            remainder: self.with(rem),
        }
    }

    /// Quotient of the Euclidean division.
    #[must_use]
    pub fn div(&self, b: &Self) -> Self {
        self.div_rem(b).quotient
    }

    /// Remainder of the Euclidean division.
    #[must_use]
    pub fn rem(&self, b: &Self) -> Self {
        self.div_rem(b).remainder
    }

    /// `self / lc(self)`; the zero polynomial is returned unchanged, as is
    /// a polynomial whose leading coefficient is not a unit.
    #[must_use]
    pub fn monic(&self) -> Self {
        let Some(lc) = self.coeffs.last() else {
            return self.clone();
        };
        if self.ring.is_one(lc) {
            return self.clone();
        }
        match self.ring.inv(lc) {
            Some(inv) => self.scale(&inv),
            None => self.clone(),
        }
    }

    /// Monic greatest common divisor by Euclid's algorithm
    /// (`gcd(a, 0) = monic(a)`, `gcd(0, 0) = 0`).
    #[must_use]
    pub fn gcd(&self, other: &Self) -> Self {
        let mut a = self.clone();
        let mut b = other.clone();
        while !b.is_zero() {
            let r = a.rem(&b);
            a = b;
            b = r;
        }
        a.monic()
    }

    /// Extended Euclid: `u · self + v · other = gcd`, `gcd` monic.
    #[must_use]
    pub fn extended_gcd(&self, other: &Self) -> ExtendedGcd<R> {
        let ring = &self.ring;
        let (mut r0, mut r1) = (self.clone(), other.clone());
        let (mut s0, mut s1) = (Self::one(ring.clone()), Self::zero(ring.clone()));
        let (mut t0, mut t1) = (Self::zero(ring.clone()), Self::one(ring.clone()));
        while !r1.is_zero() {
            let DivRem {
                quotient: q,
                remainder: r,
            } = r0.div_rem(&r1);
            let s = s0.sub(&q.mul(&s1));
            let t = t0.sub(&q.mul(&t1));
            r0 = r1;
            r1 = r;
            s0 = s1;
            s1 = s;
            t0 = t1;
            t1 = t;
        }
        if let Some(lc) = r0.coeffs.last()
            && !ring.is_one(lc)
            && let Some(inv) = ring.inv(lc)
        {
            return ExtendedGcd {
                u: s0.scale(&inv),
                v: t0.scale(&inv),
                gcd: r0.scale(&inv),
            };
        }
        ExtendedGcd {
            u: s0,
            v: t0,
            gcd: r0,
        }
    }

    /// Formal derivative.
    #[must_use]
    pub fn derivative(&self) -> Self {
        if self.coeffs.len() <= 1 {
            return Self::zero(self.ring.clone());
        }
        let out = self
            .coeffs
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, c)| self.ring.mul(c, &self.ring.embed_u64(i as u64)))
            .collect();
        self.with(out)
    }

    /// Is `gcd(self, self′)` constant?  A non-zero constant counts as
    /// square-free; a polynomial with vanishing derivative (a `p`-th power
    /// in characteristic `p`) does not.
    pub fn is_squarefree(&self) -> bool {
        let d = self.derivative();
        if d.is_zero() {
            return self.degree() == Some(0);
        }
        self.gcd(&d).degree() == Some(0)
    }

    /// `self^exp mod m`.
    #[must_use]
    pub fn powmod(&self, mut exp: u64, m: &Self) -> Self {
        let mut result = Self::one(self.ring.clone());
        let mut b = self.rem(m);
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul(&b).rem(m);
            }
            exp >>= 1;
            if exp > 0 {
                b = b.mul(&b).rem(m);
            }
        }
        result
    }

    /// `self^exp mod m` with an arbitrary-precision exponent.
    #[must_use]
    pub fn powmod_big(&self, exp: &BigUint, m: &Self) -> Self {
        let mut result = Self::one(self.ring.clone());
        let mut b = self.rem(m);
        let bits = exp.bits();
        for i in 0..bits {
            if exp.bit(i) {
                result = result.mul(&b).rem(m);
            }
            if i + 1 < bits {
                b = b.mul(&b).rem(m);
            }
        }
        result
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 𝔽ₚ[x]-specific operations
// ═══════════════════════════════════════════════════════════════════════════

impl PolyIn<Fp64> {
    /// Build from `u64` residues (already reduced modulo `p`) over `𝔽ₚ`.
    pub fn over_prime(p: u64, coeffs: Vec<u64>) -> Self {
        Self::from_coeffs(Fp64::new(p), coeffs)
    }

    /// The field size `p`.
    #[inline]
    pub fn modulus(&self) -> u64 {
        self.ring.modulus()
    }

    /// `p`-th root of a polynomial whose derivative vanishes, i.e.
    /// `self = b(xᵖ) = b(x)ᵖ`: returns `b`.
    #[must_use]
    pub fn pth_root(&self) -> Self {
        let step = self.ring.modulus() as usize;
        let out = self.coeffs.iter().step_by(step).copied().collect();
        self.with(out)
    }

    /// Sorted roots in `𝔽ₚ` of a non-zero polynomial.
    ///
    /// `g = gcd(self, xᵖ − x)` is the product of `(x − r)` over the
    /// distinct roots; `g` is then split with random
    /// `gcd(g, (x + a)^{(p−1)/2} − 1)` (Cantor–Zassenhaus equal-degree
    /// factorisation, all factors linear).  Splitting draws from the
    /// pinned xorshift64* seeded with `p ^ 0x9E37_79B9_7F4A_7C15`; after
    /// 4096 failed attempts (probability `2⁻⁴⁰⁹⁶`) the roots found so far
    /// are returned.
    pub fn roots(&self) -> Vec<u64> {
        let ring = self.ring;
        let p = ring.modulus();
        let f = self.monic();
        if f.coeffs.len() <= 1 {
            return vec![];
        }
        let x = Self::x(ring);
        let xp_minus_x = x.powmod(p, &f).sub(&x);
        let g = f.gcd(&xp_minus_x);
        let one = Self::one(ring);
        let mut out = Vec::new();
        let mut rng = XorShift::new(p ^ 0x9E37_79B9_7F4A_7C15);
        let mut stack = vec![g];
        let mut attempts = 0u32;
        while let Some(g) = stack.pop() {
            let deg = g.coeffs.len().saturating_sub(1);
            if deg == 0 {
                continue;
            }
            if deg == 1 {
                out.push(ring.neg(&g.coeffs[0]));
                continue;
            }
            // Split g with a random (x + a)^{(p−1)/2} − 1.
            loop {
                attempts += 1;
                if attempts > 4096 {
                    // Probability 2^{−4096}; give up rather than loop forever.
                    return out;
                }
                let a = rng.next_u64() % p;
                let pw = Self::from_coeffs(ring, vec![a, 1])
                    .powmod((p - 1) / 2, &g)
                    .sub(&one);
                let h = g.gcd(&pw);
                let dh = h.coeffs.len().saturating_sub(1);
                if dh > 0 && dh < deg {
                    let q = g.div(&h);
                    stack.push(h);
                    stack.push(q);
                    break;
                }
            }
        }
        out.sort_unstable();
        out
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn fp(p: u64, c: &[u64]) -> PolyIn<Fp64> {
        PolyIn::over_prime(p, c.to_vec())
    }

    #[test]
    fn fp64_scalar_ops() {
        let f = Fp64::new(7);
        assert_eq!(f.add(&5, &4), 2);
        assert_eq!(f.sub(&2, &5), 4);
        assert_eq!(f.mul(&5, &4), 6);
        assert_eq!(f.neg(&0), 0);
        assert_eq!(f.neg(&3), 4);
        assert_eq!(f.inv(&3), Some(5));
        assert_eq!(f.inv(&0), None);
        assert_eq!(f.embed_u64(100), 2);
        assert_eq!(f.pow(3, 6), 1);
        // Above 2³² the u128 path is taken.
        let big = Fp64::new(1_000_000_000_039);
        let a = 999_999_999_999u64;
        assert_eq!(
            big.mul(&a, &a),
            ((a as u128 * a as u128) % 1_000_000_000_039u128) as u64
        );
        assert_eq!(big.mul(&big.inv(&a).unwrap_or(0), &a), 1);
    }

    #[test]
    fn construction_normalizes() {
        assert_eq!(fp(7, &[1, 2, 0, 0]).coeffs(), &[1, 2]);
        assert!(fp(7, &[0, 0]).is_zero());
        assert_eq!(fp(7, &[0, 0]).degree(), None);
        assert_eq!(PolyIn::x(Fp64::new(7)).coeffs(), &[0, 1]);
        assert_eq!(
            PolyIn::constant(Fp64::new(7), 0),
            PolyIn::zero(Fp64::new(7))
        );
        assert!(fp(7, &[3, 1]).is_monic() && !fp(7, &[3, 2]).is_monic());
    }

    #[test]
    fn add_sub_scale_mul() {
        let a = fp(7, &[3, 1, 4]);
        let b = fp(7, &[4, 6, 3]);
        // 7, 7, 7 → the zero polynomial, whose normalised coefficient list is empty.
        assert!(a.add(&b).coeffs().is_empty());
        assert!(a.add(&b).is_zero());
        assert_eq!(a.sub(&b).coeffs(), &[6, 2, 1]);
        assert_eq!(a.neg().coeffs(), &[4, 6, 3]);
        assert_eq!(a.scale(&2).coeffs(), &[6, 2, 1]);
        // (3 + x + 4x²)(4 + 6x + 3x²) mod 7
        assert_eq!(a.mul(&b).coeffs(), &[5, 1, 3, 6, 5]);
        assert!(a.mul(&PolyIn::zero(Fp64::new(7))).is_zero());
    }

    #[test]
    fn div_rem_matches_sympy() {
        // sympy: gf_div([5,1,4,1,3], [1,0,2], 7, ZZ) → q = 5x² + x + 1, r = 6x + 1
        let a = fp(7, &[3, 1, 4, 1, 5]);
        let b = fp(7, &[2, 0, 1]);
        let DivRem {
            quotient: q,
            remainder: r,
        } = a.div_rem(&b);
        assert_eq!(q.coeffs(), &[1, 1, 5]);
        assert_eq!(r.coeffs(), &[1, 6]);
        assert_eq!(q.mul(&b).add(&r), a);
        assert_eq!(a.div(&b), q);
        assert_eq!(a.rem(&b), r);
        // Non-monic divisor.
        let b = fp(7, &[2, 3]);
        let d = a.div_rem(&b);
        assert_eq!(d.quotient.mul(&b).add(&d.remainder), a);
        assert!(d.remainder.degree().unwrap_or(0) < 1);
        // Degenerate divisors.
        let d = a.div_rem(&PolyIn::zero(Fp64::new(7)));
        assert!(d.quotient.is_zero());
        assert_eq!(d.remainder, a);
        let d = fp(7, &[1]).div_rem(&a);
        assert!(d.quotient.is_zero());
        assert_eq!(d.remainder, fp(7, &[1]));
    }

    #[test]
    fn gcd_and_extended_gcd() {
        // sympy: gf_gcdex([1,3,2,1], [1,1,5], 13, ZZ) → ([7,1]... ascending [7,1], [4,4,12], [1])
        let a = fp(13, &[1, 2, 3, 1]);
        let b = fp(13, &[5, 1, 1]);
        let ExtendedGcd { u, v, gcd: g } = a.extended_gcd(&b);
        assert_eq!(g.coeffs(), &[1]);
        assert_eq!(u.coeffs(), &[7, 1]);
        assert_eq!(v.coeffs(), &[4, 4, 12]);
        assert_eq!(u.mul(&a).add(&v.mul(&b)), g);
        // Common factor x + 1 over GF(101):
        // A = (x+1)(x+2)(x²+1), B = (x+1)(x+3)(x+50)
        let a = fp(101, &[2, 3, 3, 3, 1]);
        let b = fp(101, &[49, 1, 54, 1]);
        assert_eq!(a.gcd(&b).coeffs(), &[1, 1]);
        assert_eq!(b.gcd(&a).coeffs(), &[1, 1]);
        let ExtendedGcd { u, v, gcd: g } = a.extended_gcd(&b);
        assert_eq!(g.coeffs(), &[1, 1]);
        assert_eq!(u.coeffs(), &[75, 89]);
        assert_eq!(v.coeffs(), &[32, 20, 12]);
        // gcd with zero is the monic version of the other argument.
        assert_eq!(fp(7, &[2, 4]).gcd(&fp(7, &[])).coeffs(), &[4, 1]);
        assert!(fp(7, &[]).gcd(&fp(7, &[])).is_zero());
    }

    #[test]
    fn derivative_and_squarefree() {
        // sympy: gf_diff([3,2,0,1,0], 5, ZZ) ascending → [1, 0, 1, 2]
        assert_eq!(fp(5, &[0, 1, 0, 2, 3]).derivative().coeffs(), &[1, 0, 1, 2]);
        assert!(fp(5, &[4]).derivative().is_zero());
        assert!(!fp(5, &[1, 2, 1]).is_squarefree()); // (x+1)²
        assert!(fp(5, &[1, 0, 1]).is_squarefree()); // (x+2)(x+3)
        assert!(fp(5, &[3]).is_squarefree());
        assert!(!fp(5, &[1, 0, 0, 0, 0, 1]).is_squarefree()); // x⁵ + 1 = (x+1)⁵
    }

    #[test]
    fn powmod_matches_sympy() {
        // sympy: gf_pow_mod([1,0], 101, [1,0,0,0,3,1], 101, ZZ) ascending → [0, 4, 62, 93]
        let f = fp(101, &[1, 3, 0, 0, 0, 1]);
        let x = PolyIn::x(Fp64::new(101));
        assert_eq!(x.powmod(101, &f).coeffs(), &[0, 4, 62, 93]);
        // sympy: gf_pow_mod([1,2], 50, f, 101, ZZ) ascending → [25, 51, 64, 76, 31]
        assert_eq!(
            fp(101, &[2, 1]).powmod(50, &f).coeffs(),
            &[25, 51, 64, 76, 31]
        );
        assert_eq!(
            fp(101, &[2, 1]).powmod_big(&BigUint::from(50u32), &f),
            fp(101, &[2, 1]).powmod(50, &f)
        );
        assert_eq!(x.powmod(0, &f).coeffs(), &[1]);
        // Large prime: sympy gf_pow_mod([1,0], p, x⁴ − 3x + 5, p) ascending
        let p = 1_000_003u64;
        let f = fp(p, &[5, p - 3, 0, 0, 1]);
        assert_eq!(
            PolyIn::x(Fp64::new(p)).powmod(p, &f).coeffs(),
            &[821_714, 144_019, 352_342, 190_351]
        );
    }

    #[test]
    fn pth_root() {
        // (x + 1)³ = x³ + 1 over GF(3)
        assert_eq!(fp(3, &[1, 0, 0, 1]).pth_root().coeffs(), &[1, 1]);
    }

    #[test]
    fn roots_match_sympy() {
        let p = 1_000_003u64;
        // sympy: polynomial_congruence(x**4 - 3*x + 5, 1000003) == [357940, 847957]
        assert_eq!(fp(p, &[5, p - 3, 0, 0, 1]).roots(), vec![357_940, 847_957]);
        // sympy: polynomial_congruence(x**2 + 1, 1000003) == []
        assert!(fp(p, &[1, 0, 1]).roots().is_empty());
        // sympy: polynomial_congruence(x**6 - 1, 1000003)
        assert_eq!(
            fp(p, &[p - 1, 0, 0, 0, 0, 0, 1]).roots(),
            vec![1, 499_501, 499_502, 500_501, 500_502, 1_000_002]
        );
        // 7000021 = 7 · 1000003 is not prime; 7000009 is (sympy.isprime).
        // sympy: polynomial_congruence(2*x**3 + 5*x + 7, 7000009) == [3330300, 3669710, 7000008]
        let p = 7_000_009u64;
        assert_eq!(
            fp(p, &[7, 5, 0, 2]).roots(),
            vec![3_330_300, 3_669_710, 7_000_008]
        );
        // Constants have no roots.
        assert!(fp(p, &[3]).roots().is_empty());
    }
}
