//! Algebraic trait hierarchy for generic polynomial coefficient types.
//!
//! This module defines the layered trait hierarchy that [`Poly<C>`](super::dense::Poly)
//! uses to bound its operations:
//!
//! - [`Ring`] — basic arithmetic (+, -, ×, 0, 1)
//! - [`EuclideanDomain`]: Ring — division with remainder, GCD
//! - [`Field`]: EuclideanDomain — exact division, multiplicative inverse
//!
//! Extension traits for type-specific capabilities:
//!
//! - [`OrderedCoeff`] — ordering (is_positive, is_negative)
//! - [`IntegralCoeff`] — integer embedding (is_integer, from_integer)
//! - [`CoeffDisplay`] — precedence-aware formatting for nested display
//!
//! # Design
//!
//! Bounds are placed at the method level, not the struct level, so `Poly<C>`
//! is maximally flexible:
//!
//! - `impl<C: Ring> Poly<C>` — add, mul, derivative, etc.
//! - `impl<C: Field> Poly<C>` — div_rem, gcd, extended_gcd, etc.
//! - `impl<C: Field + OrderedCoeff> Poly<C>` — sign normalization, factoring
//!
//! # References
//!
//! Pattern follows feanor-math (`RingBase` hierarchy) and Symbolica
//! (`Ring → EuclideanDomain → Field`).

use std::fmt;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

// ═══════════════════════════════════════════════════════════════════════════
// Core algebraic hierarchy
// ═══════════════════════════════════════════════════════════════════════════

/// A commutative ring with identity: supports +, -, ×, 0, 1.
///
/// All elements are `Clone + PartialEq` so that polynomial algorithms
/// can compare and copy coefficients freely.
pub trait Ring: Clone + PartialEq + fmt::Debug + Sized {
    /// The additive identity.
    fn zero() -> Self;

    /// The multiplicative identity.
    fn one() -> Self;

    /// Test for additive identity.
    fn is_zero(&self) -> bool;

    /// Test for multiplicative identity.
    fn is_one(&self) -> bool {
        *self == Self::one()
    }

    /// Addition.
    fn add(&self, rhs: &Self) -> Self;

    /// Subtraction.
    fn sub(&self, rhs: &Self) -> Self;

    /// Multiplication.
    fn mul(&self, rhs: &Self) -> Self;

    /// Additive inverse.
    fn neg(&self) -> Self;

    /// Repeated multiplication: `self^n` for non-negative integer `n`.
    ///
    /// Default implementation uses binary exponentiation.
    fn pow_usize(&self, mut n: usize) -> Self {
        if n == 0 {
            return Self::one();
        }
        let mut base = self.clone();
        let mut result = Self::one();
        while n > 1 {
            if n & 1 == 1 {
                result = result.mul(&base);
            }
            base = base.mul(&base);
            n >>= 1;
        }
        result.mul(&base)
    }
}

/// A Euclidean domain: a ring with division-with-remainder and GCD.
///
/// For fields, `div_rem(a, b) = (a/b, 0)` and `gcd(a, b) = 1` (trivially).
/// For polynomial rings `k[x]` where `k` is a field, these are the standard
/// polynomial Euclidean division and GCD.
pub trait EuclideanDomain: Ring {
    /// Euclidean division: returns `(quotient, remainder)` such that
    /// `self = quotient * other + remainder` and the remainder is
    /// "smaller" than `other` (in the Euclidean sense).
    fn div_rem(&self, other: &Self) -> (Self, Self);

    /// Greatest common divisor.
    ///
    /// Default implementation uses the Euclidean algorithm.
    fn gcd(a: &Self, b: &Self) -> Self {
        let mut a = a.clone();
        let mut b = b.clone();
        while !b.is_zero() {
            let (_, r) = a.div_rem(&b);
            a = b;
            b = r;
        }
        a
    }
}

/// A field: a commutative ring where every nonzero element has a
/// multiplicative inverse.
///
/// Fields are trivially Euclidean domains with `div_rem(a, b) = (a/b, 0)`
/// and `gcd(a, b) = 1`.
pub trait Field: EuclideanDomain {
    /// Exact division: `self / other`.
    ///
    /// # Panics
    ///
    /// May panic if `other` is zero.
    fn div(&self, other: &Self) -> Self;

    /// Multiplicative inverse: `1 / self`.
    ///
    /// # Panics
    ///
    /// May panic if `self` is zero.
    fn inv(&self) -> Self;
}

// ═══════════════════════════════════════════════════════════════════════════
// Extension traits for type-specific capabilities
// ═══════════════════════════════════════════════════════════════════════════

/// Coefficient types that support ordering.
///
/// Only implemented for coefficient types where a total order exists
/// and is mathematically meaningful (e.g., ℚ, ℝ — but not ℂ or
/// finite fields).
pub trait OrderedCoeff: Ring {
    /// Strictly greater than zero.
    fn is_positive(&self) -> bool;

    /// Strictly less than zero.
    fn is_negative(&self) -> bool;

    /// Absolute value.
    fn abs(&self) -> Self {
        if self.is_negative() {
            self.neg()
        } else {
            self.clone()
        }
    }
}

/// Coefficient types that embed the integers.
///
/// Provides conversion between the coefficient type and `BigInt`,
/// allowing construction of polynomials from integer data.
pub trait IntegralCoeff: Ring {
    /// Is this element an integer (denominator = 1 for rationals)?
    fn is_integer(&self) -> bool;

    /// Extract the integer value, if this element is an integer.
    fn to_integer(&self) -> Option<BigInt>;

    /// Embed an integer into this coefficient type.
    fn from_integer(n: BigInt) -> Self;

    /// Convenience: embed a small integer.
    fn from_i64(n: i64) -> Self {
        Self::from_integer(BigInt::from(n))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Precedence-aware display
// ═══════════════════════════════════════════════════════════════════════════

/// Binding strength of the surrounding context, for parenthesization.
///
/// When displaying a coefficient inside a larger expression, the
/// coefficient may need parentheses depending on the surrounding
/// operator.  For example, `3/4` doesn't need parens in a sum
/// (`3/4 + x`) but does in a product (`(3/4)*x` or `(3/4)x`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BindingStrength {
    /// Top-level or inside parentheses — never needs parens.
    Weakest,
    /// Inside a sum: `a + b`.
    Sum,
    /// Inside a product: `a * b`.
    Product,
    /// Inside exponentiation: `a^n`.
    Power,
    /// Atomic — never needs parens (single token).
    Atom,
}

/// Precedence-aware display for coefficient types.
///
/// Used by `Poly<C>`'s `Display` implementation to correctly
/// parenthesize coefficients when they appear inside products
/// or powers.
pub trait CoeffDisplay {
    /// Format this coefficient in the given binding-strength context.
    ///
    /// Implementations should add parentheses when the coefficient's
    /// natural binding strength is weaker than `env`.  For example,
    /// a fraction `3/4` in a `Product` context should display as
    /// `(3/4)` to avoid ambiguity with `3/4*x`.
    fn fmt_coeff(
        &self,
        f: &mut fmt::Formatter<'_>,
        env: BindingStrength,
    ) -> fmt::Result;
}

// ═══════════════════════════════════════════════════════════════════════════
// Implementations for Ratio<BigInt>
// ═══════════════════════════════════════════════════════════════════════════

impl Ring for Ratio<BigInt> {
    #[inline]
    fn zero() -> Self {
        <Ratio<BigInt> as Zero>::zero()
    }

    #[inline]
    fn one() -> Self {
        <Ratio<BigInt> as One>::one()
    }

    #[inline]
    fn is_zero(&self) -> bool {
        Zero::is_zero(self)
    }

    #[inline]
    fn is_one(&self) -> bool {
        One::is_one(self)
    }

    #[inline]
    fn add(&self, rhs: &Self) -> Self {
        self + rhs
    }

    #[inline]
    fn sub(&self, rhs: &Self) -> Self {
        self - rhs
    }

    #[inline]
    fn mul(&self, rhs: &Self) -> Self {
        self * rhs
    }

    #[inline]
    fn neg(&self) -> Self {
        -self
    }
}

impl EuclideanDomain for Ratio<BigInt> {
    #[inline]
    fn div_rem(&self, other: &Self) -> (Self, Self) {
        // Fields have trivial Euclidean division: quotient = a/b, remainder = 0.
        (self / other, <Self as Ring>::zero())
    }

    #[inline]
    fn gcd(a: &Self, b: &Self) -> Self {
        // In a field, gcd(a, b) = 1 whenever at least one is nonzero.
        // gcd(0, 0) = 0 by convention.
        if Ring::is_zero(a) && Ring::is_zero(b) {
            <Self as Ring>::zero()
        } else {
            <Self as Ring>::one()
        }
    }
}

impl Field for Ratio<BigInt> {
    #[inline]
    fn div(&self, other: &Self) -> Self {
        self / other
    }

    #[inline]
    fn inv(&self) -> Self {
        Ratio::new(self.denom().clone(), self.numer().clone())
    }
}

impl OrderedCoeff for Ratio<BigInt> {
    #[inline]
    fn is_positive(&self) -> bool {
        Signed::is_positive(self)
    }

    #[inline]
    fn is_negative(&self) -> bool {
        Signed::is_negative(self)
    }

    #[inline]
    fn abs(&self) -> Self {
        Signed::abs(self)
    }
}

impl IntegralCoeff for Ratio<BigInt> {
    #[inline]
    fn is_integer(&self) -> bool {
        self.denom().is_one()
    }

    fn to_integer(&self) -> Option<BigInt> {
        if self.is_integer() {
            Some(self.numer().clone())
        } else {
            None
        }
    }

    #[inline]
    fn from_integer(n: BigInt) -> Self {
        Ratio::from_integer(n)
    }
}

impl CoeffDisplay for Ratio<BigInt> {
    fn fmt_coeff(
        &self,
        f: &mut fmt::Formatter<'_>,
        env: BindingStrength,
    ) -> fmt::Result {
        if self.denom().is_one() {
            // Integer: no parens needed unless it's negative in a tight context.
            let n = self.numer();
            if n.sign() == num_bigint::Sign::Minus && env >= BindingStrength::Sum {
                write!(f, "({})", n)
            } else {
                write!(f, "{}", n)
            }
        } else {
            // Fraction: needs parens in Product or tighter context.
            if env >= BindingStrength::Product {
                write!(f, "({}/{})", self.numer(), self.denom())
            } else {
                write!(f, "{}/{}", self.numer(), self.denom())
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    type Q = Ratio<BigInt>;

    fn q(n: i64, d: i64) -> Q {
        Ratio::new(BigInt::from(n), BigInt::from(d))
    }

    // Disambiguated helpers — avoid collision between Ring::zero and num_traits::Zero::zero
    fn qzero() -> Q { <Q as Ring>::zero() }
    fn qone() -> Q { <Q as Ring>::one() }

    // ── Ring axiom tests ────────────────────────────────────────────

    #[test]
    fn ring_additive_identity() {
        let a = q(3, 4);
        assert_eq!(Ring::add(&a, &qzero()), a);
        assert_eq!(Ring::add(&qzero(), &a), a);
    }

    #[test]
    fn ring_multiplicative_identity() {
        let a = q(3, 4);
        assert_eq!(Ring::mul(&a, &qone()), a);
        assert_eq!(Ring::mul(&qone(), &a), a);
    }

    #[test]
    fn ring_additive_inverse() {
        let a = q(3, 4);
        let neg_a = Ring::neg(&a);
        assert_eq!(Ring::add(&a, &neg_a), qzero());
    }

    #[test]
    fn ring_commutativity() {
        let a = q(2, 3);
        let b = q(5, 7);
        assert_eq!(Ring::add(&a, &b), Ring::add(&b, &a));
        assert_eq!(Ring::mul(&a, &b), Ring::mul(&b, &a));
    }

    #[test]
    fn ring_associativity() {
        let a = q(1, 2);
        let b = q(1, 3);
        let c = q(1, 5);
        assert_eq!(
            Ring::add(&Ring::add(&a, &b), &c),
            Ring::add(&a, &Ring::add(&b, &c))
        );
        assert_eq!(
            Ring::mul(&Ring::mul(&a, &b), &c),
            Ring::mul(&a, &Ring::mul(&b, &c))
        );
    }

    #[test]
    fn ring_distributivity() {
        let a = q(2, 3);
        let b = q(1, 4);
        let c = q(5, 6);
        // a * (b + c) = a*b + a*c
        let lhs = Ring::mul(&a, &Ring::add(&b, &c));
        let rhs = Ring::add(&Ring::mul(&a, &b), &Ring::mul(&a, &c));
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn ring_is_zero_and_is_one() {
        assert!(Ring::is_zero(&qzero()));
        assert!(!Ring::is_zero(&qone()));
        assert!(Ring::is_one(&qone()));
        assert!(!Ring::is_one(&qzero()));
        assert!(!Ring::is_zero(&q(2, 3)));
        assert!(!Ring::is_one(&q(2, 3)));
    }

    #[test]
    fn ring_pow_usize() {
        let a = q(2, 3);
        assert_eq!(a.pow_usize(0), qone());
        assert_eq!(a.pow_usize(1), q(2, 3));
        assert_eq!(a.pow_usize(2), q(4, 9));
        assert_eq!(a.pow_usize(3), q(8, 27));
    }

    // ── Field tests ─────────────────────────────────────────────────

    #[test]
    fn field_division() {
        let a = q(3, 4);
        let b = q(2, 5);
        // (3/4) / (2/5) = (3/4) * (5/2) = 15/8
        assert_eq!(Field::div(&a, &b), q(15, 8));
    }

    #[test]
    fn field_inverse() {
        let a = q(3, 7);
        let inv = Field::inv(&a);
        assert_eq!(Ring::mul(&a, &inv), qone());
    }

    #[test]
    fn field_div_rem_trivial() {
        let a = q(5, 3);
        let b = q(2, 7);
        let (quot, rem) = EuclideanDomain::div_rem(&a, &b);
        assert_eq!(quot, Field::div(&a, &b));
        assert!(Ring::is_zero(&rem));
    }

    #[test]
    fn field_gcd_is_one() {
        let a = q(3, 4);
        let b = q(5, 6);
        assert_eq!(EuclideanDomain::gcd(&a, &b), qone());
    }

    #[test]
    fn field_gcd_with_zero() {
        let a = qzero();
        let b = q(5, 6);
        assert_eq!(EuclideanDomain::gcd(&a, &b), qone());
        assert!(Ring::is_zero(&EuclideanDomain::gcd(&a, &a)));
    }

    // ── OrderedCoeff tests ──────────────────────────────────────────

    #[test]
    fn ordered_positive() {
        assert!(OrderedCoeff::is_positive(&q(3, 4)));
        assert!(!OrderedCoeff::is_positive(&q(-3, 4)));
        assert!(!OrderedCoeff::is_positive(&qzero()));
    }

    #[test]
    fn ordered_negative() {
        assert!(!OrderedCoeff::is_negative(&q(3, 4)));
        assert!(OrderedCoeff::is_negative(&q(-3, 4)));
        assert!(!OrderedCoeff::is_negative(&qzero()));
    }

    #[test]
    fn ordered_abs() {
        assert_eq!(OrderedCoeff::abs(&q(-3, 4)), q(3, 4));
        assert_eq!(OrderedCoeff::abs(&q(3, 4)), q(3, 4));
        assert_eq!(OrderedCoeff::abs(&qzero()), qzero());
    }

    // ── IntegralCoeff tests ─────────────────────────────────────────

    #[test]
    fn integral_is_integer() {
        assert!(IntegralCoeff::is_integer(&q(5, 1)));
        assert!(!IntegralCoeff::is_integer(&q(5, 3)));
        assert!(IntegralCoeff::is_integer(&qzero()));
    }

    #[test]
    fn integral_to_integer() {
        assert_eq!(
            IntegralCoeff::to_integer(&q(7, 1)),
            Some(BigInt::from(7))
        );
        assert_eq!(IntegralCoeff::to_integer(&q(7, 3)), None);
    }

    #[test]
    fn integral_from_integer() {
        let x = <Q as IntegralCoeff>::from_integer(BigInt::from(42));
        assert_eq!(x, q(42, 1));
    }

    #[test]
    fn integral_from_i64() {
        let x = <Q as IntegralCoeff>::from_i64(-5);
        assert_eq!(x, q(-5, 1));
    }

    // ── CoeffDisplay tests ──────────────────────────────────────────

    fn format_coeff(c: &Q, env: BindingStrength) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        // Use a simple wrapper to call fmt_coeff
        struct Wrapper<'a>(&'a Q, BindingStrength);
        impl fmt::Display for Wrapper<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt_coeff(f, self.1)
            }
        }
        write!(s, "{}", Wrapper(c, env)).unwrap();
        s
    }

    #[test]
    fn display_integer_in_sum() {
        assert_eq!(format_coeff(&q(5, 1), BindingStrength::Sum), "5");
    }

    #[test]
    fn display_integer_in_product() {
        assert_eq!(format_coeff(&q(5, 1), BindingStrength::Product), "5");
    }

    #[test]
    fn display_negative_integer_in_sum() {
        // Negative integers get parens in sum context to avoid ambiguity.
        assert_eq!(format_coeff(&q(-5, 1), BindingStrength::Sum), "(-5)");
    }

    #[test]
    fn display_negative_integer_at_top_level() {
        assert_eq!(format_coeff(&q(-5, 1), BindingStrength::Weakest), "-5");
    }

    #[test]
    fn display_fraction_in_sum() {
        // Fractions don't need parens in a sum: 3/4 + x is unambiguous.
        assert_eq!(format_coeff(&q(3, 4), BindingStrength::Sum), "3/4");
    }

    #[test]
    fn display_fraction_in_product() {
        // Fractions need parens in a product: (3/4)*x, not 3/4*x.
        assert_eq!(format_coeff(&q(3, 4), BindingStrength::Product), "(3/4)");
    }

    #[test]
    fn display_one_in_product() {
        assert_eq!(format_coeff(&q(1, 1), BindingStrength::Product), "1");
    }

    #[test]
    fn display_zero() {
        assert_eq!(format_coeff(&qzero(), BindingStrength::Weakest), "0");
    }

    // ── Binding strength ordering ───────────────────────────────────

    #[test]
    fn binding_strength_ordering() {
        assert!(BindingStrength::Weakest < BindingStrength::Sum);
        assert!(BindingStrength::Sum < BindingStrength::Product);
        assert!(BindingStrength::Product < BindingStrength::Power);
        assert!(BindingStrength::Power < BindingStrength::Atom);
    }
}
