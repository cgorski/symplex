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
//! - [`RingOps`] — since 0.50 a name for
//!   [`CoeffDomain`](super::domain::CoeffDomain), the operations of a
//!   commutative ring given as methods on a ring *value* (`&self` carries
//!   `p`), with an optional inverse so that fields and non-fields share one
//!   trait;
//! - [`Fp64`] — the prime field `𝔽ₚ` on `u64` residues for `p < 2⁶³`;
//! - [`DensePoly<D>`] (also named [`PolyIn<D>`]) — a polynomial with
//!   coefficients in `D::El` together with the domain it lives in:
//!   addition, multiplication, Euclidean division, monic gcd, extended
//!   gcd, resultant, Yun's square-free decomposition, derivative,
//!   square-freeness, Horner evaluation, modular exponentiation, and (over
//!   [`Fp64`]) `p`-th roots and root finding by Cantor–Zassenhaus.
//!
//! Coefficients are stored in ascending degree order with no trailing
//! zeros; the zero polynomial has an empty coefficient vector.  Every
//! [`Fp64`] method is `#[inline]`, so the generic loops monomorphise to
//! the same machine code as the hand-rolled `u64` versions they replace.
//!
//! The randomised splitting in [`DensePoly::roots`] draws from the crate's
//! pinned [`XorShift64Star`](crate::base::rng::XorShift64Star): which
//! factor a split finds first depends on that stream, so the iteration
//! order here is an output contract, not an implementation detail.

use std::fmt;

use num_bigint::BigUint;

use crate::base::rng::XorShift64Star as XorShift;

// ═══════════════════════════════════════════════════════════════════════════
// The ring trait
// ═══════════════════════════════════════════════════════════════════════════

/// The trait of coefficient domains, under its name before 0.50
/// ([`CoeffDomain`](super::domain::CoeffDomain)); [`DensePoly`]'s
/// division-based algorithms need every non-zero leading coefficient they
/// meet to be a unit.
pub use super::domain::CoeffDomain as RingOps;

// ═══════════════════════════════════════════════════════════════════════════
// 𝔽ₚ on u64
// ═══════════════════════════════════════════════════════════════════════════

/// The prime field `𝔽ₚ` for a prime `p < 2⁶³`, elements as `u64`
/// residues in `[0, p)`.
///
/// Products go through `u128` only when `p ≥ 2³²`; below that the `u64`
/// product cannot overflow and a native remainder is used.  Above it the
/// double-word product is reduced by Barrett's method (Menezes et al.,
/// *Handbook of Applied Cryptography*, Algorithm 14.42, with base 2): two
/// word multiplications by the precomputed `⌊2²ᵏ/p⌋` (`k` the bit length
/// of `p`) and at most two subtractions, instead of a call to the
/// compiler's 128-by-128-bit remainder (`__umodti3`).  Inverses are by the
/// extended Euclidean algorithm on words (before 0.50 by Fermat,
/// `a^(p−2)`: 124 modular products, 46% of the time of Brown's modular
/// gcd).  The caller guarantees primality — nothing here checks it —
/// exactly as the hand-rolled predecessors did; an element that is not a
/// unit (`p` composite) has no inverse.
///
/// `p = 2` is supported: the arithmetic is the same, and
/// [`DensePoly::roots`] tests the two field elements directly, since the
/// Cantor–Zassenhaus splitting it uses for odd `p` (`(x + a)^{(p−1)/2}`)
/// degenerates there.  The factoring routes of the crate never call it
/// with `2` (they need `p ∤ lc(f)` for arbitrary `f`), which is why the
/// earlier documentation said "odd prime".
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fp64 {
    p: u64,
    /// The bit length `k` of `p`.
    bits: u32,
    /// Barrett's constant `⌊2²ᵏ/p⌋`, below `2⁶⁴` for `k ≤ 63`.
    mu: u64,
}

/// Only the modulus: `mu` and `bits` are functions of it.
impl fmt::Debug for Fp64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Fp64").field("p", &self.p).finish()
    }
}

impl Fp64 {
    /// The field with `p` elements.
    ///
    /// `p` must be a prime below `2⁶³` (so that `a + b` never overflows
    /// for reduced `a`, `b`); `2` is allowed.  Only checked in debug builds.
    #[inline]
    pub const fn new(p: u64) -> Self {
        debug_assert!(p >= 2);
        debug_assert!(p < (1u64 << 63));
        let bits = u64::BITS - p.leading_zeros();
        // `2²ᵏ/p < 2ᵏ⁺¹ ≤ 2⁶⁴` since `p ≥ 2ᵏ⁻¹`; for an out-of-contract
        // `p ≥ 2⁶³` the constant is 0 and [`mul_reduced`](Self::mul_reduced)
        // takes its exact fallback.
        let mu = match 1u128.checked_shl(2 * bits) {
            Some(t) if bits <= 63 => (t / p as u128) as u64,
            _ => 0,
        };
        Fp64 { p, bits, mu }
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
            self.barrett(a as u128 * b as u128)
        }
    }

    /// `x mod p` for `p > 2³²` (HAC 14.42): with `k` the bit length of
    /// `p` and `x < 2²ᵏ`, `q = ⌊⌊x/2ᵏ⁻¹⌋·μ / 2ᵏ⁺¹⌋` is at most 2 below
    /// `⌊x/p⌋`, and `x − q·p < 3p`.  A larger `x` (operands that were not
    /// reduced) takes the exact double-word remainder.
    #[inline]
    const fn barrett(self, x: u128) -> u64 {
        let k = self.bits;
        if self.mu == 0 || x >> (2 * k) != 0 {
            return (x % self.p as u128) as u64;
        }
        // `x < 2²ᵏ` so `x / 2ᵏ⁻¹ < 2ᵏ⁺¹ ≤ 2⁶⁴`: no bits are lost.
        let q1 = (x >> (k - 1)) as u64;
        let q = ((q1 as u128 * self.mu as u128) >> (k + 1)) as u64;
        let p = self.p as u128;
        let mut r = x - q as u128 * p;
        // Two conditional subtractions (`r < 3p`), without branches: which
        // of the three cases occurs is data-dependent and unpredictable.
        r -= if r >= p { p } else { 0 };
        r -= if r >= p { p } else { 0 };
        r as u64
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

    /// The extended Euclidean algorithm on `(p, a)`: the cofactors stay
    /// below `p` in absolute value, so `i128` holds them.
    fn inv(&self, a: &u64) -> Option<u64> {
        let a = a % self.p;
        if a == 0 {
            return None;
        }
        let (mut r0, mut r1) = (self.p, a);
        let (mut t0, mut t1) = (0i128, 1i128);
        while r1 != 0 {
            let q = r0 / r1;
            (r0, r1) = (r1, r0 - q * r1);
            (t0, t1) = (t1, t0 - i128::from(q) * t1);
        }
        if r0 != 1 {
            return None;
        }
        u64::try_from(t0.rem_euclid(i128::from(self.p))).ok()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// The polynomial
// ═══════════════════════════════════════════════════════════════════════════

/// Result of [`DensePoly::div_rem`]: `a = quotient · b + remainder` with
/// `deg remainder < deg b`.
#[derive(Clone, Debug, PartialEq)]
pub struct DivRem<R: RingOps> {
    /// The quotient.
    pub quotient: DensePoly<R>,
    /// The remainder, of degree below the divisor's.
    pub remainder: DensePoly<R>,
}

/// Result of [`DensePoly::extended_gcd`]: `u · a + v · b = gcd`, `gcd` monic.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtendedGcd<R: RingOps> {
    /// Bézout cofactor of the first argument.
    pub u: DensePoly<R>,
    /// Bézout cofactor of the second argument.
    pub v: DensePoly<R>,
    /// The monic greatest common divisor.
    pub gcd: DensePoly<R>,
}

/// A dense univariate polynomial over the domain value `ring`
/// ([`CoeffDomain`](super::domain::CoeffDomain)): the one implementation
/// of the dense algorithms for domains given as values (`𝔽ₚ`, `ℚ`, `ℤ`,
/// `ℚ(i)`, …).  Up to 0.49 it was `PolyIn`, still a name for it.
///
/// Coefficients ascend in degree and the last one is non-zero (the zero
/// polynomial is empty).  Two polynomials compare equal when their
/// coefficient vectors do; operands of a binary operation are assumed to
/// share the ring (the left operand's ring is used).
///
/// ```
/// use symplex::factor_zassenhaus::domain::QQ;
/// use symplex::factor_zassenhaus::modpoly::DensePoly;
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// let q = |n: i64| Ratio::from_integer(BigInt::from(n));
/// // (x − 1)²(x + 2) and (x − 1)(x + 3)
/// let f = DensePoly::from_coeffs(QQ, vec![q(2), q(-3), q(0), q(1)]);
/// let g = DensePoly::from_coeffs(QQ, vec![q(-3), q(2), q(1)]);
/// assert_eq!(f.gcd(&g).coeffs(), &[q(-1), q(1)]);
/// assert_eq!(f.resultant(&g), Some(q(0)));
/// let sqf = f.squarefree_factors().unwrap();
/// assert_eq!(sqf.len(), 2);
/// assert_eq!((sqf[1].0.coeffs(), sqf[1].1), (&[q(-1), q(1)][..], 2));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct DensePoly<R: RingOps> {
    ring: R,
    coeffs: Vec<R::El>,
}

/// [`DensePoly`] under its name before 0.50.
pub type PolyIn<R> = DensePoly<R>;

/// `xⁿ` in `ring` by repeated squaring.
fn el_pow<R: RingOps>(ring: &R, x: &R::El, mut n: usize) -> R::El {
    let mut result = ring.one();
    let mut base = x.clone();
    while n > 0 {
        if n & 1 == 1 {
            result = ring.mul(&result, &base);
        }
        n >>= 1;
        if n > 0 {
            base = ring.mul(&base, &base);
        }
    }
    result
}

impl<R: RingOps> DensePoly<R> {
    /// The zero polynomial.
    pub fn zero(ring: R) -> Self {
        DensePoly {
            ring,
            coeffs: Vec::new(),
        }
    }

    /// The constant polynomial `1`.
    pub fn one(ring: R) -> Self {
        let one = ring.one();
        DensePoly {
            ring,
            coeffs: vec![one],
        }
    }

    /// The constant polynomial `c`.
    pub fn constant(ring: R, c: R::El) -> Self {
        if ring.is_zero(&c) {
            return Self::zero(ring);
        }
        DensePoly {
            ring,
            coeffs: vec![c],
        }
    }

    /// The variable `x`.
    pub fn x(ring: R) -> Self {
        let (zero, one) = (ring.zero(), ring.one());
        DensePoly {
            ring,
            coeffs: vec![zero, one],
        }
    }

    /// From coefficients in ascending degree order (already reduced into
    /// the ring's canonical representation); trailing zeros are dropped.
    pub fn from_coeffs(ring: R, coeffs: Vec<R::El>) -> Self {
        let mut p = DensePoly { ring, coeffs };
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

    /// The value at `x` by Horner's rule (`0` for the zero polynomial).
    pub fn eval(&self, x: R::El) -> R::El {
        let ring = &self.ring;
        self.coeffs
            .iter()
            .rev()
            .fold(ring.zero(), |acc, c| ring.add(&ring.mul(&acc, &x), c))
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

    /// `self · rhs` (the domain's [`dense_mul`](RingOps::dense_mul), else
    /// schoolbook).
    #[must_use]
    pub fn mul(&self, rhs: &Self) -> Self {
        if self.is_zero() || rhs.is_zero() {
            return Self::zero(self.ring.clone());
        }
        let ring = &self.ring;
        if let Some(out) = ring.dense_mul(&self.coeffs, &rhs.coeffs) {
            return self.with(out);
        }
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
        self.try_div_rem(b).unwrap_or_else(|| DivRem {
            quotient: Self::zero(self.ring.clone()),
            remainder: self.clone(),
        })
    }

    /// Euclidean division, `None` when `b` is zero or its leading
    /// coefficient is not a unit (and `deg self ≥ deg b`).
    pub fn try_div_rem(&self, b: &Self) -> Option<DivRem<R>> {
        let ring = &self.ring;
        let trivial = |remainder: Self| DivRem {
            quotient: Self::zero(ring.clone()),
            remainder,
        };
        let db = b.degree()?;
        let Some(da) = self.degree() else {
            return Some(trivial(Self::zero(ring.clone())));
        };
        if da < db {
            return Some(trivial(self.clone()));
        }
        let lc = &b.coeffs[db];
        let monic = ring.is_one(lc);
        let inv_lc = if monic { ring.one() } else { ring.inv(lc)? };
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
        Some(DivRem {
            quotient: self.with(quot),
            remainder: self.with(rem),
        })
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
    /// a polynomial whose leading coefficient is not a unit
    /// ([`try_monic`](Self::try_monic) says so).
    #[must_use]
    pub fn monic(&self) -> Self {
        self.try_monic().unwrap_or_else(|| self.clone())
    }

    /// `self / lc(self)`, or `None` when the polynomial is zero or its
    /// leading coefficient is not a unit.
    ///
    /// ```
    /// use num_bigint::BigInt;
    /// use symplex::factor_zassenhaus::domain::ZZ;
    /// use symplex::factor_zassenhaus::modpoly::DensePoly;
    ///
    /// let p = |c: &[i64]| DensePoly::from_coeffs(ZZ, c.iter().map(|&k| BigInt::from(k)).collect());
    /// assert_eq!(p(&[3, -1]).try_monic(), Some(p(&[-3, 1])));
    /// assert_eq!(p(&[1, 2]).try_monic(), None); // 2 is not a unit of ℤ
    /// assert_eq!(p(&[]).try_monic(), None);
    /// ```
    pub fn try_monic(&self) -> Option<Self> {
        let lc = self.coeffs.last()?;
        if self.ring.is_one(lc) {
            return Some(self.clone());
        }
        Some(self.scale(&self.ring.inv(lc)?))
    }

    /// Monic greatest common divisor (`gcd(a, 0) = monic(a)`,
    /// `gcd(0, 0) = 0`): the domain's [`dense_gcd`](RingOps::dense_gcd),
    /// else Euclid's algorithm.
    ///
    /// Over a domain that is not a field Euclid may meet a leading
    /// coefficient that is not a unit; it cannot continue, and the last
    /// remainder reached (a multiple of the gcd, not monic) is returned —
    /// [`try_gcd`](Self::try_gcd) says so.  (Before 0.50 the loop did not
    /// end in that case.)
    #[must_use]
    pub fn gcd(&self, other: &Self) -> Self {
        match self.gcd_steps(other) {
            Ok(g) | Err(g) => g,
        }
    }

    /// [`gcd`](Self::gcd), `None` when a leading coefficient that is not a
    /// unit stopped Euclid's algorithm or the last remainder cannot be made
    /// monic (before 0.50 that remainder was returned as the gcd).
    pub fn try_gcd(&self, other: &Self) -> Option<Self> {
        self.gcd_steps(other).ok()
    }

    /// `Ok(monic gcd)`, or `Err(last remainder)` on a non-unit.
    fn gcd_steps(&self, other: &Self) -> Result<Self, Self> {
        if let Some(g) = self.ring.dense_gcd(&self.coeffs, &other.coeffs) {
            return Ok(self.with(g));
        }
        Self::euclid_gcd(self.clone(), other.clone())
    }

    /// Euclid's algorithm on `a` and `b` (no hook): `Ok(monic gcd)`, or
    /// `Err(last remainder)` on a non-unit.  The element-level
    /// [`GenPoly`](super::generic::GenPoly) runs its gcds here.
    pub(crate) fn euclid_gcd(mut a: Self, mut b: Self) -> Result<Self, Self> {
        while !b.is_zero() {
            let Some(DivRem { remainder: r, .. }) = a.try_div_rem(&b) else {
                return Err(b);
            };
            a = b;
            b = r;
        }
        if a.is_zero() {
            return Ok(a);
        }
        a.try_monic().ok_or(a)
    }

    /// Extended Euclid: `u · self + v · other = gcd`, `gcd` monic (the
    /// domain's [`dense_extended_gcd`](RingOps::dense_extended_gcd) when it
    /// has one).  On a leading coefficient that is not a unit the row
    /// reached is returned (the identity holds; `gcd` is a multiple of the
    /// gcd, not monic).
    #[must_use]
    pub fn extended_gcd(&self, other: &Self) -> ExtendedGcd<R> {
        match self.extended_gcd_steps(other) {
            Ok(e) | Err(e) => e,
        }
    }

    /// [`extended_gcd`](Self::extended_gcd), `None` when a leading
    /// coefficient that is not a unit stopped the algorithm or the gcd
    /// reached cannot be made monic.
    pub fn try_extended_gcd(&self, other: &Self) -> Option<ExtendedGcd<R>> {
        self.extended_gcd_steps(other).ok()
    }

    /// `Ok(the extended gcd)`, or `Err(the row reached)` on a non-unit.
    fn extended_gcd_steps(&self, other: &Self) -> Result<ExtendedGcd<R>, ExtendedGcd<R>> {
        let ring = &self.ring;
        if !other.is_zero()
            && let Some(e) = ring.dense_extended_gcd(&self.coeffs, &other.coeffs)
        {
            return Ok(ExtendedGcd {
                u: self.with(e.x),
                v: self.with(e.y),
                gcd: self.with(e.gcd),
            });
        }
        Self::euclid_extended_gcd(self.clone(), other.clone())
    }

    /// The extended Euclidean algorithm on `a` and `b` (no hook):
    /// `Ok(the extended gcd)`, or `Err(the row reached)` on a non-unit.
    /// The element-level [`GenPoly`](super::generic::GenPoly) runs its
    /// extended gcds here.
    pub(crate) fn euclid_extended_gcd(a: Self, b: Self) -> Result<ExtendedGcd<R>, ExtendedGcd<R>> {
        let ring = a.ring.clone();
        let ring = &ring;
        let (mut r0, mut r1) = (a, b);
        let (mut s0, mut s1) = (Self::one(ring.clone()), Self::zero(ring.clone()));
        let (mut t0, mut t1) = (Self::zero(ring.clone()), Self::one(ring.clone()));
        while !r1.is_zero() {
            let Some(DivRem {
                quotient: q,
                remainder: r,
            }) = r0.try_div_rem(&r1)
            else {
                return Err(ExtendedGcd {
                    u: s1,
                    v: t1,
                    gcd: r1,
                });
            };
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
        {
            let Some(inv) = ring.inv(lc) else {
                return Err(ExtendedGcd {
                    u: s0,
                    v: t0,
                    gcd: r0,
                });
            };
            return Ok(ExtendedGcd {
                u: s0.scale(&inv),
                v: t0.scale(&inv),
                gcd: r0.scale(&inv),
            });
        }
        Ok(ExtendedGcd {
            u: s0,
            v: t0,
            gcd: r0,
        })
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

    /// The resultant `res(self, other)` by the Euclidean algorithm
    /// (`res(a, b) = (−1)^{mn} lc(b)^{m−s} res(b, a mod b)`, `s = deg(a mod
    /// b)`), or the domain's [`dense_resultant`](RingOps::dense_resultant);
    /// `0` when either is zero.  `None` when a leading coefficient that is
    /// not a unit is met.
    pub fn resultant(&self, other: &Self) -> Option<R::El> {
        Self::resultant_of(self.clone(), other.clone())
    }

    /// [`resultant`](Self::resultant) of owned operands.  The element-level
    /// [`GenPoly`](super::generic::GenPoly) computes its resultants here.
    pub(crate) fn resultant_of(mut a: Self, mut b: Self) -> Option<R::El> {
        let ring = a.ring.clone();
        let ring = &ring;
        let odd = |m: usize, n: usize| (m % 2 == 1) && (n % 2 == 1);
        let mut acc = ring.one();
        loop {
            let (Some(m), Some(n)) = (a.degree(), b.degree()) else {
                return Some(ring.zero());
            };
            if m == 0 && n == 0 {
                return Some(acc);
            }
            if m == 0 {
                return Some(ring.mul(&acc, &el_pow(ring, &a.coeffs[0], n)));
            }
            if n == 0 {
                return Some(ring.mul(&acc, &el_pow(ring, &b.coeffs[0], m)));
            }
            if m < n {
                if odd(m, n) {
                    acc = ring.neg(&acc);
                }
                std::mem::swap(&mut a, &mut b);
                continue;
            }
            if let Some(res) = ring.dense_resultant(&a.coeffs, &b.coeffs) {
                return Some(ring.mul(&acc, &res));
            }
            let r = a.try_div_rem(&b)?.remainder;
            let Some(s) = r.degree() else {
                return Some(ring.zero());
            };
            if odd(m, n) {
                acc = ring.neg(&acc);
            }
            acc = ring.mul(&acc, &el_pow(ring, &b.coeffs[n], m - s));
            a = b;
            b = r;
        }
    }

    /// Yun's square-free decomposition: `[(f₁, 1), (f₂, 2), …]` with
    /// `self = lc · ∏ fᵢ^i`, the `fᵢ` monic, square-free, pairwise coprime
    /// and of positive degree; empty for a constant or zero polynomial.
    /// The same steps as `GenPoly::squarefree_factors`.  Needs
    /// characteristic 0 or above the degree; `None` when a leading
    /// coefficient that is not a unit is met.
    pub fn squarefree_factors(&self) -> Option<Vec<(Self, usize)>> {
        let n = self.degree().unwrap_or(0);
        if n == 0 {
            return Some(Vec::new());
        }
        let quo = |a: &Self, b: &Self| a.try_div_rem(b).map(|d| d.quotient);
        let p = self.try_monic()?;
        let p_prime = p.derivative();
        let c = p.try_gcd(&p_prime)?;
        if c.degree().unwrap_or(0) == 0 {
            return Some(vec![(p, 1)]);
        }
        let mut w = quo(&p, &c)?;
        let mut y = quo(&p_prime, &c)?;
        let mut factors = Vec::new();
        let mut i = 1;
        loop {
            if i > n {
                // Only in characteristic p ≤ deg: Yun does not apply.
                return None;
            }
            let z = y.sub(&w.derivative());
            if z.is_zero() {
                if w.degree().unwrap_or(0) > 0 {
                    factors.push((w.monic(), i));
                }
                return Some(factors);
            }
            let g = w.try_gcd(&z)?;
            if g.degree().unwrap_or(0) > 0 {
                factors.push((g.monic(), i));
            }
            w = quo(&w, &g)?;
            y = quo(&z, &g)?;
            i += 1;
        }
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

impl DensePoly<Fp64> {
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

    /// Sorted distinct roots in `𝔽ₚ` of a polynomial (a multiple root is
    /// listed once; the zero polynomial and constants have none).
    ///
    /// `g = gcd(self, xᵖ − x)` is the product of `(x − r)` over the
    /// distinct roots; `g` is then split with random
    /// `gcd(g, (x + a)^{(p−1)/2} − 1)` (Cantor–Zassenhaus equal-degree
    /// factorisation, all factors linear).  Splitting draws from the
    /// pinned xorshift64* seeded with `p ^ 0x9E37_79B9_7F4A_7C15`.
    ///
    /// For `p = 2` the exponent `(p − 1)/2` is `0` and no split can ever
    /// happen, so the two field elements are simply evaluated — `x² + x`
    /// has the roots `{0, 1}`, which the splitting loop used to miss.
    ///
    /// The random split succeeds with probability about `½` per draw; the
    /// loop gives up after 4096 consecutive failures (probability
    /// `2⁻⁴⁰⁹⁶`, a stuck generator rather than bad luck) and returns the
    /// roots found so far, which is then an *incomplete* list — a
    /// `tracing::warn!` says so.  There is no way to signal it in the
    /// return type without changing every caller; in practice it does
    /// not happen.
    pub fn roots(&self) -> Vec<u64> {
        let ring = self.ring;
        let p = ring.modulus();
        let f = self.monic();
        if f.coeffs.len() <= 1 {
            return vec![];
        }
        if p == 2 {
            return [0u64, 1].into_iter().filter(|&r| f.eval(r) == 0).collect();
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
                    tracing::warn!(
                        p,
                        found = out.len(),
                        unsplit_degree = deg,
                        "DensePoly::roots: 4096 failed Cantor–Zassenhaus splits, returning an incomplete root list"
                    );
                    out.sort_unstable();
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

    fn fp(p: u64, c: &[u64]) -> DensePoly<Fp64> {
        DensePoly::over_prime(p, c.to_vec())
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

    /// Barrett's reduction equals the double-word remainder: primes at the
    /// ends of the range it serves (just above `2³²`, the gcd primes below
    /// `2…²`, the largest below `2⁶³`), operands at the edges, random
    /// operands, and unreduced operands (the exact fallback).
    #[test]
    fn barrett_matches_the_double_word_remainder() {
        let mut rng = XorShift::new(0x5eed);
        for p in [
            4_294_967_311u64,
            1_000_000_000_039,
            (1u64 << 62) - 57,
            (1u64 << 62) - 87,
            (1u64 << 63) - 25,
        ] {
            let f = Fp64::new(p);
            let exact = |a: u64, b: u64| ((a as u128 * b as u128) % p as u128) as u64;
            let edge = [0, 1, 2, p / 2, p - 2, p - 1];
            for &a in &edge {
                for &b in &edge {
                    assert_eq!(f.mul_reduced(a, b), exact(a, b), "{p}: {a}·{b}");
                }
            }
            for _ in 0..20_000 {
                let (a, b) = (rng.next_u64() % p, rng.next_u64() % p);
                assert_eq!(f.mul_reduced(a, b), exact(a, b), "{p}: {a}·{b}");
                let (a, b) = (rng.next_u64(), rng.next_u64());
                assert_eq!(f.mul_reduced(a, b), exact(a, b), "{p}: {a}·{b} unreduced");
            }
            let a = rng.next_u64() % (p - 1) + 1;
            assert_eq!(f.mul(&f.inv(&a).unwrap_or(0), &a), 1);
        }
    }

    #[test]
    fn construction_normalizes() {
        assert_eq!(fp(7, &[1, 2, 0, 0]).coeffs(), &[1, 2]);
        assert!(fp(7, &[0, 0]).is_zero());
        assert_eq!(fp(7, &[0, 0]).degree(), None);
        assert_eq!(DensePoly::x(Fp64::new(7)).coeffs(), &[0, 1]);
        assert_eq!(
            DensePoly::constant(Fp64::new(7), 0),
            DensePoly::zero(Fp64::new(7))
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
        assert!(a.mul(&DensePoly::zero(Fp64::new(7))).is_zero());
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
        let d = a.div_rem(&DensePoly::zero(Fp64::new(7)));
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
        let x = DensePoly::x(Fp64::new(101));
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
            DensePoly::x(Fp64::new(p)).powmod(p, &f).coeffs(),
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
