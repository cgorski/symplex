//! Coefficient domains as *values*: one trait, [`CoeffDomain`], for every
//! ring the coefficients of a dense polynomial
//! ([`DensePoly<D>`](super::modpoly::DensePoly)) live in, so that
//! division, gcd, extended gcd, resultant, square-free decomposition,
//! Horner evaluation and modular powers are written once for all of them.
//!
//! The element-level traits of [`traits`](super::traits) (`Ring`, `Field`)
//! need `C::zero()` with no arguments, so a coefficient type has to carry
//! its ring in the type.  A domain whose parameters are decided at run
//! time — a prime `p`, a tower of radicals, a number of parameters — cannot
//! do that, and before 0.50 each such domain re-implemented the polynomial
//! algorithms (`PolyIn` over `𝔽ₚ`, the radical tower's `kp_gcd` and
//! `kp_squarefree`, …) or put its context into every element (`AlgNum`'s
//! minimal polynomial).  Here the domain is a value: `&self` carries the
//! parameters, and a zero-sized domain (`QQ`, `ZZ`, `GaussQQ`,
//! [`FieldOf<C>`]) monomorphises to the same code as the element-level
//! traits.  See CONTRIBUTING.md, "Type rules that keep bug classes out".
//!
//! # Which type
//!
//! | Coefficients | Domain | Element |
//! |---|---|---|
//! | `ℚ` | [`QQ`] | `Ratio<BigInt>` (its `ℤ[x]` fast paths are the domain's hooks) |
//! | `ℤ` (a ring: only `±1` is invertible) | [`ZZ`] | `BigInt` |
//! | `𝔽ₚ`, `p < 2⁶³` decided at run time | [`Fp64`](super::modpoly::Fp64) | `u64` |
//! | `ℚ(i)` | [`GaussQQ`] | [`GaussQ`] |
//! | any element-level [`Field`] (`ℚ(params)`, `ℚ(α)`, …) | [`FieldOf<C>`] | `C` |
//!
//! # Corner cases
//!
//! - **Non-units.** [`CoeffDomain::inv`] returns `None` for zero and for a
//!   non-unit (`2` in `ZZ`), so a field and a ring share the trait;
//!   `DensePoly`'s division-based algorithms need every leading coefficient
//!   they divide by to be a unit and say what they do otherwise.
//! - **One domain per polynomial.**  The operands of a binary operation are
//!   assumed to share the domain (the left operand's is used), as for the
//!   modulus of `𝔽ₚ` before.
//! - **Hooks.** [`CoeffDomain::dense_mul`], [`dense_gcd`](CoeffDomain::dense_gcd),
//!   [`dense_extended_gcd`](CoeffDomain::dense_extended_gcd) and
//!   [`dense_resultant`](CoeffDomain::dense_resultant) let a domain answer
//!   faster than the generic algorithm (`QQ` through the primitive PRS in
//!   `ℤ[x]`); a hook must return what the generic algorithm returns.

use std::fmt;
use std::marker::PhantomData;

use num_bigint::BigInt;
use num_integer::ExtendedGcd;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

use super::traits::{EuclideanDomain, Field, Ring};
use crate::base::numeric::Q;

// ═══════════════════════════════════════════════════════════════════════════
// The trait
// ═══════════════════════════════════════════════════════════════════════════

/// A commutative ring whose operations are methods on a ring *value*.
///
/// Unlike [`Ring`], `zero`/`one` take `&self`, so the modulus (or any other
/// parameter) may be decided at run time.  `inv` returns `None` for zero
/// and for non-units, which lets a field and a non-field ring share the
/// trait.  Up to 0.49 this trait was `modpoly::RingOps` (still a name for
/// it).
///
/// ```
/// use symplex::factor_zassenhaus::domain::{CoeffDomain, QQ, ZZ};
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// let half = Ratio::new(BigInt::from(1), BigInt::from(2));
/// assert_eq!(QQ.inv(&half), Some(Ratio::from_integer(BigInt::from(2))));
/// assert_eq!(ZZ.inv(&BigInt::from(2)), None);
/// assert_eq!(ZZ.inv(&BigInt::from(-1)), Some(BigInt::from(-1)));
/// ```
pub trait CoeffDomain: Clone + fmt::Debug {
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

    /// A faster product of two dense coefficient vectors (ascending, no
    /// trailing zeros, both non-empty), or `None` for the generic one.
    fn dense_mul(&self, a: &[Self::El], b: &[Self::El]) -> Option<Vec<Self::El>> {
        let _ = (a, b);
        None
    }

    /// A faster monic gcd of two dense coefficient vectors, or `None` for
    /// Euclid's algorithm.
    fn dense_gcd(&self, a: &[Self::El], b: &[Self::El]) -> Option<Vec<Self::El>> {
        let _ = (a, b);
        None
    }

    /// A faster extended gcd (`x·a + y·b = gcd`, `gcd` monic, the minimal
    /// cofactors), or `None` for the extended Euclidean algorithm.
    fn dense_extended_gcd(
        &self,
        a: &[Self::El],
        b: &[Self::El],
    ) -> Option<ExtendedGcd<Vec<Self::El>>> {
        let _ = (a, b);
        None
    }

    /// A faster resultant of two non-constant polynomials with
    /// `deg a ≥ deg b`, or `None` for the Euclidean one.
    fn dense_resultant(&self, a: &[Self::El], b: &[Self::El]) -> Option<Self::El> {
        let _ = (a, b);
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ℚ and ℤ
// ═══════════════════════════════════════════════════════════════════════════

/// The rational numbers, elements `Ratio<BigInt>`.
///
/// The arithmetic is that of the element-level [`Ring`]/[`Field`] impls of
/// `Ratio<BigInt>` (no gcd for integer operands, Lehmer gcds), and the
/// hooks route products, gcds, extended gcds and resultants through
/// `ℤ[x]` as `GenPoly<Ratio<BigInt>>` does, so the two give the same
/// polynomials.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct QQ;

impl CoeffDomain for QQ {
    type El = Q;

    #[inline]
    fn zero(&self) -> Q {
        <Q as Zero>::zero()
    }
    #[inline]
    fn one(&self) -> Q {
        <Q as One>::one()
    }
    #[inline]
    fn embed_u64(&self, n: u64) -> Q {
        Q::from_integer(BigInt::from(n))
    }
    #[inline]
    fn is_zero(&self, a: &Q) -> bool {
        Zero::is_zero(a)
    }
    #[inline]
    fn is_one(&self, a: &Q) -> bool {
        One::is_one(a)
    }
    #[inline]
    fn add(&self, a: &Q, b: &Q) -> Q {
        Ring::add(a, b)
    }
    #[inline]
    fn sub(&self, a: &Q, b: &Q) -> Q {
        Ring::sub(a, b)
    }
    #[inline]
    fn mul(&self, a: &Q, b: &Q) -> Q {
        Ring::mul(a, b)
    }
    #[inline]
    fn neg(&self, a: &Q) -> Q {
        Ring::neg(a)
    }
    #[inline]
    fn inv(&self, a: &Q) -> Option<Q> {
        (!Zero::is_zero(a)).then(|| Field::inv(a))
    }
    fn dense_mul(&self, a: &[Q], b: &[Q]) -> Option<Vec<Q>> {
        <Q as Ring>::poly_mul(a, b)
    }
    fn dense_gcd(&self, a: &[Q], b: &[Q]) -> Option<Vec<Q>> {
        <Q as Field>::poly_gcd(a, b)
    }
    fn dense_extended_gcd(&self, a: &[Q], b: &[Q]) -> Option<ExtendedGcd<Vec<Q>>> {
        <Q as Field>::poly_extended_gcd(a, b)
    }
    fn dense_resultant(&self, a: &[Q], b: &[Q]) -> Option<Q> {
        <Q as Field>::poly_resultant(a, b)
    }
}

/// The integers, elements `BigInt`: a ring, whose only units are `±1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ZZ;

impl CoeffDomain for ZZ {
    type El = BigInt;

    #[inline]
    fn zero(&self) -> BigInt {
        <BigInt as Zero>::zero()
    }
    #[inline]
    fn one(&self) -> BigInt {
        <BigInt as One>::one()
    }
    #[inline]
    fn embed_u64(&self, n: u64) -> BigInt {
        BigInt::from(n)
    }
    #[inline]
    fn is_zero(&self, a: &BigInt) -> bool {
        Zero::is_zero(a)
    }
    #[inline]
    fn is_one(&self, a: &BigInt) -> bool {
        One::is_one(a)
    }
    #[inline]
    fn add(&self, a: &BigInt, b: &BigInt) -> BigInt {
        a + b
    }
    #[inline]
    fn sub(&self, a: &BigInt, b: &BigInt) -> BigInt {
        a - b
    }
    #[inline]
    fn mul(&self, a: &BigInt, b: &BigInt) -> BigInt {
        a * b
    }
    #[inline]
    fn neg(&self, a: &BigInt) -> BigInt {
        -a
    }
    #[inline]
    fn inv(&self, a: &BigInt) -> Option<BigInt> {
        One::is_one(&a.abs()).then(|| a.clone())
    }
}

/// The domain of an element-level field `C` ([`Field`]): every operation
/// is `C`'s, every non-zero element is a unit, and the hooks are `C`'s
/// fast paths (`Ring::poly_mul`, `Field::poly_gcd`, …).  It ties the
/// element types of [`GenPoly<C>`](super::generic::GenPoly) (`ℚ(params)`,
/// `ℚ(α)`, …) to [`DensePoly`](super::modpoly::DensePoly).
pub struct FieldOf<C>(PhantomData<fn() -> C>);

impl<C> FieldOf<C> {
    /// The domain value (zero-sized).
    pub const fn new() -> Self {
        FieldOf(PhantomData)
    }
}

impl<C> Default for FieldOf<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C> Clone for FieldOf<C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C> Copy for FieldOf<C> {}

impl<C> fmt::Debug for FieldOf<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FieldOf<{}>", std::any::type_name::<C>())
    }
}

impl<C: Field> CoeffDomain for FieldOf<C> {
    type El = C;

    #[inline]
    fn zero(&self) -> C {
        C::zero()
    }
    #[inline]
    fn one(&self) -> C {
        C::one()
    }
    fn embed_u64(&self, n: u64) -> C {
        // Binary: `n` doublings and additions of `1` (no integer embedding
        // is required of `C`).
        let mut acc = C::zero();
        for i in (0..u64::BITS - n.leading_zeros()).rev() {
            acc = acc.add(&acc);
            if (n >> i) & 1 == 1 {
                acc = acc.add(&C::one());
            }
        }
        acc
    }
    #[inline]
    fn is_zero(&self, a: &C) -> bool {
        a.is_zero()
    }
    #[inline]
    fn is_one(&self, a: &C) -> bool {
        a.is_one()
    }
    #[inline]
    fn add(&self, a: &C, b: &C) -> C {
        a.add(b)
    }
    #[inline]
    fn sub(&self, a: &C, b: &C) -> C {
        a.sub(b)
    }
    #[inline]
    fn mul(&self, a: &C, b: &C) -> C {
        a.mul(b)
    }
    #[inline]
    fn neg(&self, a: &C) -> C {
        a.neg()
    }
    #[inline]
    fn inv(&self, a: &C) -> Option<C> {
        (!a.is_zero()).then(|| Field::inv(a))
    }
    fn dense_mul(&self, a: &[C], b: &[C]) -> Option<Vec<C>> {
        C::poly_mul(a, b)
    }
    fn dense_gcd(&self, a: &[C], b: &[C]) -> Option<Vec<C>> {
        C::poly_gcd(a, b)
    }
    fn dense_extended_gcd(&self, a: &[C], b: &[C]) -> Option<ExtendedGcd<Vec<C>>> {
        C::poly_extended_gcd(a, b)
    }
    fn dense_resultant(&self, a: &[C], b: &[C]) -> Option<C> {
        C::poly_resultant(a, b)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// The Gaussian rationals ℚ(i)
// ═══════════════════════════════════════════════════════════════════════════

/// A Gaussian rational `re + im·i`.
///
/// The one implementation of `ℚ(i)`: before 0.50 the exact values of
/// evalf (`Gauss`), the numeric `RootSum` traces of the integrator
/// (`GaussQ`) and the matrix code's square roots (`GaussianRational`) each
/// had their own.  An element-level [`Field`] (for `GenPoly<GaussQ>`) and
/// the element of the domain [`GaussQQ`].
///
/// ```
/// use symplex::factor_zassenhaus::domain::{CoeffDomain, GaussQ, GaussQQ};
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// let q = |n: i64| Ratio::from_integer(BigInt::from(n));
/// let z = GaussQ { re: q(1), im: q(1) };
/// // (1 + i)⁻¹ = 1/2 − i/2
/// let w = GaussQQ.inv(&z).unwrap();
/// assert_eq!(w.mul(&z), GaussQ::real(q(1)));
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GaussQ {
    /// The real part.
    pub re: Q,
    /// The imaginary part.
    pub im: Q,
}

impl GaussQ {
    /// The real number `re`.
    pub fn real(re: Q) -> GaussQ {
        GaussQ {
            re,
            im: <Q as Zero>::zero(),
        }
    }

    /// The imaginary unit `i`.
    pub fn i() -> GaussQ {
        GaussQ {
            re: <Q as Zero>::zero(),
            im: <Q as One>::one(),
        }
    }

    /// Is this `0`?
    pub fn is_zero(&self) -> bool {
        Zero::is_zero(&self.re) && Zero::is_zero(&self.im)
    }

    /// `self + o`.
    #[must_use]
    pub fn add(&self, o: &GaussQ) -> GaussQ {
        GaussQ {
            re: &self.re + &o.re,
            im: &self.im + &o.im,
        }
    }

    /// `self − o`.
    #[must_use]
    pub fn sub(&self, o: &GaussQ) -> GaussQ {
        GaussQ {
            re: &self.re - &o.re,
            im: &self.im - &o.im,
        }
    }

    /// `self · o` (one rational product for two real operands).
    #[must_use]
    pub fn mul(&self, o: &GaussQ) -> GaussQ {
        if Zero::is_zero(&self.im) && Zero::is_zero(&o.im) {
            return GaussQ::real(&self.re * &o.re);
        }
        GaussQ {
            re: &self.re * &o.re - &self.im * &o.im,
            im: &self.re * &o.im + &self.im * &o.re,
        }
    }

    /// `−self`.
    #[must_use]
    pub fn neg(&self) -> GaussQ {
        GaussQ {
            re: -&self.re,
            im: -&self.im,
        }
    }

    /// `1/self = (re − im·i)/(re² + im²)`, `None` for `0`.
    pub fn recip(&self) -> Option<GaussQ> {
        if self.is_zero() {
            return None;
        }
        if Zero::is_zero(&self.im) {
            return Some(GaussQ::real(self.re.recip()));
        }
        let n = &self.re * &self.re + &self.im * &self.im;
        Some(GaussQ {
            re: &self.re / &n,
            im: -(&self.im / &n),
        })
    }
}

impl Ring for GaussQ {
    fn zero() -> Self {
        GaussQ::real(<Q as Zero>::zero())
    }
    fn one() -> Self {
        GaussQ::real(<Q as One>::one())
    }
    fn is_zero(&self) -> bool {
        GaussQ::is_zero(self)
    }
    fn add(&self, rhs: &Self) -> Self {
        GaussQ::add(self, rhs)
    }
    fn sub(&self, rhs: &Self) -> Self {
        GaussQ::sub(self, rhs)
    }
    fn mul(&self, rhs: &Self) -> Self {
        GaussQ::mul(self, rhs)
    }
    fn neg(&self) -> Self {
        GaussQ::neg(self)
    }
}

impl EuclideanDomain for GaussQ {
    fn div_rem(&self, other: &Self) -> (Self, Self) {
        (Field::div(self, other), <GaussQ as Ring>::zero())
    }
}

impl Field for GaussQ {
    fn div(&self, other: &Self) -> Self {
        GaussQ::mul(self, &Field::inv(other))
    }
    /// `(a + b·i)⁻¹ = (a − b·i)/(a² + b²)`; callers only invert non-zero
    /// elements (leading coefficients, a gcd's constant), and a zero one
    /// gives zero rather than a panic.
    fn inv(&self) -> Self {
        self.recip().unwrap_or_else(<GaussQ as Ring>::zero)
    }
}

/// The Gaussian rationals `ℚ(i)` as a domain, elements [`GaussQ`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GaussQQ;

impl CoeffDomain for GaussQQ {
    type El = GaussQ;

    fn zero(&self) -> GaussQ {
        <GaussQ as Ring>::zero()
    }
    fn one(&self) -> GaussQ {
        <GaussQ as Ring>::one()
    }
    fn embed_u64(&self, n: u64) -> GaussQ {
        GaussQ::real(Ratio::from_integer(BigInt::from(n)))
    }
    fn is_zero(&self, a: &GaussQ) -> bool {
        a.is_zero()
    }
    fn is_one(&self, a: &GaussQ) -> bool {
        Zero::is_zero(&a.im) && One::is_one(&a.re)
    }
    fn add(&self, a: &GaussQ, b: &GaussQ) -> GaussQ {
        a.add(b)
    }
    fn sub(&self, a: &GaussQ, b: &GaussQ) -> GaussQ {
        a.sub(b)
    }
    fn mul(&self, a: &GaussQ, b: &GaussQ) -> GaussQ {
        a.mul(b)
    }
    fn neg(&self, a: &GaussQ) -> GaussQ {
        a.neg()
    }
    fn inv(&self, a: &GaussQ) -> Option<GaussQ> {
        a.recip()
    }
}
