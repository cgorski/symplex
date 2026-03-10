//! Rational functions over ℚ: elements of ℚ(x).
//!
//! [`RationalFn`] represents a rational function `p(x)/q(x)` where `p` and `q`
//! are univariate polynomials with rational coefficients ([`Poly`]).  It is
//! stored in **reduced form**: `gcd(numer, denom) = 1` and `denom` is monic.
//!
//! This type implements [`Ring`], [`EuclideanDomain`], [`Field`], and
//! [`IntegralCoeff`] from the [`traits`](super::traits) module, enabling
//! its use as the coefficient type for [`GenPoly<RationalFn>`](super::generic::GenPoly).
//!
//! # Purpose
//!
//! When the Risch algorithm operates on a tower extension `ℚ(x)(θ)`, the
//! polynomials in `θ` have coefficients in `ℚ(x)`.  `RationalFn` provides
//! exact field arithmetic on these coefficients, and
//! `GenPoly<RationalFn>` gives us polynomial operations (GCD, Hermite
//! reduction, Rothstein-Trager) at the tower level.
//!
//! # Canonicalization
//!
//! After every arithmetic operation, the result is reduced:
//! 1. Cancel `gcd(numer, denom)` from both.
//! 2. Make `denom` monic (leading coefficient = 1).
//! 3. If `denom` is zero, panic (division by zero).

use std::fmt;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};

use super::dense::Poly;
use super::traits::{
    BindingStrength, CoeffDisplay, EuclideanDomain, Field, IntegralCoeff, Ring,
};

// ═══════════════════════════════════════════════════════════════════════════
// The type
// ═══════════════════════════════════════════════════════════════════════════

/// A rational function `numer(x) / denom(x)` in ℚ(x).
///
/// Stored in reduced form with monic denominator.
#[derive(Clone)]
pub struct RationalFn {
    numer: Poly,
    denom: Poly,
}

// ═══════════════════════════════════════════════════════════════════════════
// Construction and accessors
// ═══════════════════════════════════════════════════════════════════════════

impl RationalFn {
    /// Create a new rational function and reduce it to canonical form.
    ///
    /// # Panics
    ///
    /// Panics if `denom` is zero.
    pub fn new(numer: Poly, denom: Poly) -> Self {
        assert!(!denom.is_zero(), "RationalFn: denominator must be nonzero");
        let mut rf = RationalFn { numer, denom };
        rf.reduce();
        rf
    }

    /// Create from a polynomial (denominator = 1).
    pub fn from_poly(p: Poly) -> Self {
        RationalFn {
            numer: p,
            denom: Poly::from_int(1),
        }
    }

    /// Create from an integer.
    pub fn from_int(n: i64) -> Self {
        Self::from_poly(Poly::from_int(n))
    }

    /// Create from a `Ratio<BigInt>`.
    pub fn from_rational(r: Ratio<BigInt>) -> Self {
        Self::from_poly(Poly::constant(r))
    }

    /// The numerator polynomial.
    pub fn numer(&self) -> &Poly {
        &self.numer
    }

    /// The denominator polynomial.
    pub fn denom(&self) -> &Poly {
        &self.denom
    }

    /// Is this a constant rational number (both numer and denom have degree 0)?
    ///
    /// This is the "is constant?" check for the Risch algorithm: an element
    /// of ℚ(x) is a constant iff it lies in ℚ.
    pub fn is_constant_rational(&self) -> bool {
        self.numer.is_constant() && self.denom.is_constant()
    }

    /// Extract the rational number value, if this is a constant.
    ///
    /// Returns `Some(p/q)` if both numer and denom are degree 0,
    /// `None` otherwise.
    pub fn to_rational(&self) -> Option<Ratio<BigInt>> {
        if !self.is_constant_rational() {
            return None;
        }
        let n = self.numer.coeff(0);
        let d = self.denom.coeff(0);
        if Zero::is_zero(&d) {
            return None;
        }
        Some(n / d)
    }

    /// Reduce to canonical form: cancel GCD, make denom monic.
    fn reduce(&mut self) {
        if self.numer.is_zero() {
            self.denom = Poly::from_int(1);
            return;
        }

        // Cancel common factors.
        let g = Poly::gcd(&self.numer, &self.denom);
        if let Some(g_deg) = g.degree() {
            if g_deg > 0 || !One::is_one(g.leading_coeff().unwrap()) {
                self.numer = self.numer.div_rem(&g).0;
                self.denom = self.denom.div_rem(&g).0;
            }
        }

        // Make denom monic.
        if let Some(lc) = self.denom.leading_coeff() {
            if !One::is_one(lc) {
                let lc = lc.clone();
                let inv_lc = Ratio::new(lc.denom().clone(), lc.numer().clone());
                self.numer = self.numer.scale(&inv_lc);
                self.denom = self.denom.scale(&inv_lc);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PartialEq — structural equality after reduction
// ═══════════════════════════════════════════════════════════════════════════

impl PartialEq for RationalFn {
    fn eq(&self, other: &Self) -> bool {
        // Both are in reduced form with monic denom, so structural equality works.
        self.numer == other.numer && self.denom == other.denom
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Ring implementation
// ═══════════════════════════════════════════════════════════════════════════

impl Ring for RationalFn {
    fn zero() -> Self {
        RationalFn {
            numer: Poly::zero(),
            denom: Poly::from_int(1),
        }
    }

    fn one() -> Self {
        RationalFn {
            numer: Poly::from_int(1),
            denom: Poly::from_int(1),
        }
    }

    fn is_zero(&self) -> bool {
        self.numer.is_zero()
    }

    fn is_one(&self) -> bool {
        // After reduction, 1/1 has numer = denom = Poly(1).
        !self.numer.is_zero()
            && self.numer.is_constant()
            && self.denom.is_constant()
            && self.numer.coeff(0) == self.denom.coeff(0)
    }

    fn add(&self, rhs: &Self) -> Self {
        // a/b + c/d = (a*d + c*b) / (b*d)
        let numer = &(&self.numer * &rhs.denom) + &(&rhs.numer * &self.denom);
        let denom = &self.denom * &rhs.denom;
        RationalFn::new(numer, denom)
    }

    fn sub(&self, rhs: &Self) -> Self {
        // a/b - c/d = (a*d - c*b) / (b*d)
        let numer = &(&self.numer * &rhs.denom) - &(&rhs.numer * &self.denom);
        let denom = &self.denom * &rhs.denom;
        RationalFn::new(numer, denom)
    }

    fn mul(&self, rhs: &Self) -> Self {
        // (a/b) * (c/d) = (a*c) / (b*d)
        let numer = &self.numer * &rhs.numer;
        let denom = &self.denom * &rhs.denom;
        RationalFn::new(numer, denom)
    }

    fn neg(&self) -> Self {
        RationalFn {
            numer: -&self.numer,
            denom: self.denom.clone(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// EuclideanDomain implementation (trivial for fields)
// ═══════════════════════════════════════════════════════════════════════════

impl EuclideanDomain for RationalFn {
    fn div_rem(&self, other: &Self) -> (Self, Self) {
        // Fields have trivial Euclidean division: quotient = a/b, remainder = 0.
        (Field::div(self, other), Self::zero())
    }

    fn gcd(a: &Self, b: &Self) -> Self {
        // In a field, gcd(a, b) = 1 whenever at least one is nonzero.
        if Ring::is_zero(a) && Ring::is_zero(b) {
            Self::zero()
        } else {
            Self::one()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Field implementation
// ═══════════════════════════════════════════════════════════════════════════

impl Field for RationalFn {
    fn div(&self, other: &Self) -> Self {
        // (a/b) / (c/d) = (a*d) / (b*c)
        assert!(!other.is_zero(), "RationalFn: division by zero");
        let numer = &self.numer * &other.denom;
        let denom = &self.denom * &other.numer;
        RationalFn::new(numer, denom)
    }

    fn inv(&self) -> Self {
        assert!(!self.is_zero(), "RationalFn: inverse of zero");
        RationalFn::new(self.denom.clone(), self.numer.clone())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// IntegralCoeff implementation (embedding ℤ into ℚ(x))
// ═══════════════════════════════════════════════════════════════════════════

impl IntegralCoeff for RationalFn {
    fn is_integer(&self) -> bool {
        // A rational function is an "integer" (in the embedding sense)
        // if it's a constant integer: numer is a constant integer, denom is 1.
        if !self.is_constant_rational() {
            return false;
        }
        let n = self.numer.coeff(0);
        let d = self.denom.coeff(0);
        if Zero::is_zero(&d) {
            return false;
        }
        let val = n / d;
        val.denom().is_one()
    }

    fn to_integer(&self) -> Option<BigInt> {
        if !self.is_integer() {
            return None;
        }
        let val = self.numer.coeff(0) / self.denom.coeff(0);
        if val.denom().is_one() {
            Some(val.numer().clone())
        } else {
            None
        }
    }

    fn from_integer(n: BigInt) -> Self {
        RationalFn::from_poly(Poly::constant(Ratio::from_integer(n)))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Display and Debug
// ═══════════════════════════════════════════════════════════════════════════

impl fmt::Display for RationalFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denom.is_constant() && One::is_one(&self.denom.coeff(0)) {
            // Denominator is 1: just print the numerator.
            write!(f, "{}", self.numer)
        } else {
            write!(f, "({})/({})", self.numer, self.denom)
        }
    }
}

impl fmt::Debug for RationalFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RationalFn({}/{})", self.numer, self.denom)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CoeffDisplay — precedence-aware formatting for use inside GenPoly
// ═══════════════════════════════════════════════════════════════════════════

impl CoeffDisplay for RationalFn {
    fn fmt_coeff(
        &self,
        f: &mut fmt::Formatter<'_>,
        env: BindingStrength,
    ) -> fmt::Result {
        if self.denom.is_constant() && One::is_one(&self.denom.coeff(0)) {
            // Denominator is 1: display the numerator.
            if self.numer.is_constant() {
                // Constant: delegate to the Ratio<BigInt> CoeffDisplay.
                self.numer.coeff(0).fmt_coeff(f, env)
            } else {
                // Polynomial numerator: needs parens in Product context or tighter.
                let needs_parens = env >= BindingStrength::Product
                    && self.numer.degree().unwrap_or(0) > 0;
                if needs_parens {
                    write!(f, "({})", self.numer)
                } else {
                    write!(f, "{}", self.numer)
                }
            }
        } else {
            // Fraction: always parenthesize in Product context or tighter.
            if env >= BindingStrength::Product {
                write!(f, "(({})/({}))", self.numer, self.denom)
            } else {
                write!(f, "({})/({})", self.numer, self.denom)
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

    fn r(n: i64, d: i64) -> Ratio<BigInt> {
        Ratio::new(BigInt::from(n), BigInt::from(d))
    }

    /// RationalFn from integer.
    fn rf_int(n: i64) -> RationalFn {
        RationalFn::from_int(n)
    }

    /// RationalFn = p(x) / q(x) from integer coefficient slices.
    fn rf(numer: &[i64], denom: &[i64]) -> RationalFn {
        let n = Poly::from_coeffs(numer.iter().map(|&c| r(c, 1)).collect());
        let d = Poly::from_coeffs(denom.iter().map(|&c| r(c, 1)).collect());
        RationalFn::new(n, d)
    }

    /// RationalFn = polynomial from integer coefficients.
    fn rf_poly(cs: &[i64]) -> RationalFn {
        RationalFn::from_poly(Poly::from_coeffs(cs.iter().map(|&c| r(c, 1)).collect()))
    }

    // Disambiguated helpers.
    fn rfzero() -> RationalFn { <RationalFn as Ring>::zero() }
    fn rfone() -> RationalFn { <RationalFn as Ring>::one() }

    // ── Construction ────────────────────────────────────────────────

    #[test]
    fn from_int_basic() {
        let a = rf_int(5);
        assert!(a.is_constant_rational());
        assert_eq!(a.to_rational(), Some(r(5, 1)));
    }

    #[test]
    fn from_poly_basic() {
        // x + 1
        let a = rf_poly(&[1, 1]);
        assert!(!a.is_constant_rational());
        assert_eq!(a.denom(), &Poly::from_int(1));
    }

    #[test]
    fn reduction_cancels_common_factor() {
        // (x² - 1) / (x - 1) should reduce to (x + 1) / 1
        let numer = Poly::from_coeffs(vec![r(-1, 1), r(0, 1), r(1, 1)]); // x² - 1
        let denom = Poly::from_coeffs(vec![r(-1, 1), r(1, 1)]);          // x - 1
        let a = RationalFn::new(numer, denom);
        assert_eq!(a.denom(), &Poly::from_int(1));
        // numer should be x + 1
        assert_eq!(a.numer().degree(), Some(1));
        assert_eq!(a.numer().coeff(0), r(1, 1));
        assert_eq!(a.numer().coeff(1), r(1, 1));
    }

    #[test]
    fn reduction_makes_denom_monic() {
        // 1 / (2x + 4) should reduce to (1/2) / (x + 2)
        let numer = Poly::from_int(1);
        let denom = Poly::from_coeffs(vec![r(4, 1), r(2, 1)]); // 2x + 4
        let a = RationalFn::new(numer, denom);
        // Denom should be monic: x + 2
        assert_eq!(a.denom().leading_coeff(), Some(&r(1, 1)));
    }

    #[test]
    fn zero_numer_gives_zero() {
        let a = RationalFn::new(Poly::zero(), Poly::from_int(7));
        assert!(Ring::is_zero(&a));
        assert_eq!(a.denom(), &Poly::from_int(1));
    }

    // ── Ring axioms ─────────────────────────────────────────────────

    #[test]
    fn ring_additive_identity() {
        let a = rf(&[1, 1], &[1]); // (x + 1)
        assert_eq!(Ring::add(&a, &rfzero()), a);
        assert_eq!(Ring::add(&rfzero(), &a), a);
    }

    #[test]
    fn ring_multiplicative_identity() {
        let a = rf(&[1, 1], &[1]); // (x + 1)
        assert_eq!(Ring::mul(&a, &rfone()), a);
        assert_eq!(Ring::mul(&rfone(), &a), a);
    }

    #[test]
    fn ring_additive_inverse() {
        let a = rf(&[3, 2], &[1, 1]); // (2x + 3) / (x + 1)
        let neg_a = Ring::neg(&a);
        let sum = Ring::add(&a, &neg_a);
        assert!(Ring::is_zero(&sum), "a + (-a) should be 0, got {sum}");
    }

    #[test]
    fn ring_commutativity_add() {
        let a = rf(&[1], &[1, 1]); // 1/(x+1)
        let b = rf(&[1], &[-1, 1]); // 1/(x-1)
        assert_eq!(Ring::add(&a, &b), Ring::add(&b, &a));
    }

    #[test]
    fn ring_commutativity_mul() {
        let a = rf(&[1], &[1, 1]); // 1/(x+1)
        let b = rf(&[1, 1], &[1]); // (x+1)
        assert_eq!(Ring::mul(&a, &b), Ring::mul(&b, &a));
    }

    #[test]
    fn ring_is_zero_and_is_one() {
        assert!(Ring::is_zero(&rfzero()));
        assert!(!Ring::is_zero(&rfone()));
        assert!(Ring::is_one(&rfone()));
        assert!(!Ring::is_one(&rfzero()));
    }

    // ── Arithmetic ──────────────────────────────────────────────────

    #[test]
    fn add_same_denom() {
        // 1/(x+1) + 2/(x+1) = 3/(x+1)
        let a = rf(&[1], &[1, 1]);
        let b = rf(&[2], &[1, 1]);
        let c = Ring::add(&a, &b);
        assert_eq!(c, rf(&[3], &[1, 1]));
    }

    #[test]
    fn add_different_denom() {
        // 1/x + 1/x² = (x + 1)/x²
        let a = rf(&[1], &[0, 1]);       // 1/x
        let b = rf(&[1], &[0, 0, 1]);    // 1/x²
        let c = Ring::add(&a, &b);
        // Should be (x + 1)/x²
        assert_eq!(c.numer().degree(), Some(1));
        assert_eq!(c.denom().degree(), Some(2));
    }

    #[test]
    fn mul_inverse_gives_one() {
        // (x + 1) * 1/(x + 1) = 1
        let a = rf_poly(&[1, 1]);       // x + 1
        let b = rf(&[1], &[1, 1]);      // 1/(x + 1)
        let c = Ring::mul(&a, &b);
        assert!(Ring::is_one(&c), "(x+1) * 1/(x+1) should be 1, got {c}");
    }

    #[test]
    fn mul_fractions() {
        // (x+1)/x * x/(x-1) = (x+1)/(x-1)
        let a = rf(&[1, 1], &[0, 1]);    // (x+1)/x
        let b = rf(&[0, 1], &[-1, 1]);   // x/(x-1)
        let c = Ring::mul(&a, &b);
        assert_eq!(c, rf(&[1, 1], &[-1, 1])); // (x+1)/(x-1)
    }

    #[test]
    fn sub_to_zero() {
        let a = rf(&[1, 2, 3], &[1, 1]);
        let b = Ring::sub(&a, &a);
        assert!(Ring::is_zero(&b));
    }

    // ── Field ───────────────────────────────────────────────────────

    #[test]
    fn field_division() {
        // (1/x) / (1/x²) = (1/x) * (x²/1) = x
        let a = rf(&[1], &[0, 1]);       // 1/x
        let b = rf(&[1], &[0, 0, 1]);    // 1/x²
        let c = Field::div(&a, &b);
        assert_eq!(c, rf_poly(&[0, 1])); // x
    }

    #[test]
    fn field_inverse() {
        let a = rf(&[1, 1], &[0, 1]); // (x+1)/x
        let inv = Field::inv(&a);
        let product = Ring::mul(&a, &inv);
        assert!(Ring::is_one(&product), "a * inv(a) should be 1, got {product}");
    }

    #[test]
    fn field_div_rem_trivial() {
        let a = rf(&[1], &[0, 1]); // 1/x
        let b = rf(&[1], &[1, 1]); // 1/(x+1)
        let (quot, rem) = EuclideanDomain::div_rem(&a, &b);
        assert_eq!(quot, Field::div(&a, &b));
        assert!(Ring::is_zero(&rem));
    }

    // ── is_constant_rational / to_rational ──────────────────────────

    #[test]
    fn constant_rational_detection() {
        assert!(rf_int(3).is_constant_rational());
        assert!(rfzero().is_constant_rational());
        assert!(!rf_poly(&[0, 1]).is_constant_rational()); // x is not constant
        assert!(!rf(&[1], &[0, 1]).is_constant_rational()); // 1/x is not constant
    }

    #[test]
    fn to_rational_extraction() {
        assert_eq!(rf_int(7).to_rational(), Some(r(7, 1)));
        assert_eq!(rfzero().to_rational(), Some(r(0, 1)));
        // 3/4 as a constant RationalFn:
        let three_fourths = RationalFn::from_rational(r(3, 4));
        assert_eq!(three_fourths.to_rational(), Some(r(3, 4)));
        // x is not extractable:
        assert_eq!(rf_poly(&[0, 1]).to_rational(), None);
    }

    // ── IntegralCoeff ───────────────────────────────────────────────

    #[test]
    fn integral_is_integer() {
        assert!(IntegralCoeff::is_integer(&rf_int(5)));
        assert!(IntegralCoeff::is_integer(&rf_int(0)));
        assert!(!IntegralCoeff::is_integer(&RationalFn::from_rational(r(1, 2))));
        assert!(!IntegralCoeff::is_integer(&rf_poly(&[0, 1]))); // x
    }

    #[test]
    fn integral_from_integer() {
        let a = <RationalFn as IntegralCoeff>::from_integer(BigInt::from(42));
        assert_eq!(a, rf_int(42));
    }

    #[test]
    fn integral_to_integer() {
        assert_eq!(IntegralCoeff::to_integer(&rf_int(7)), Some(BigInt::from(7)));
        assert_eq!(IntegralCoeff::to_integer(&rf_poly(&[0, 1])), None);
    }

    // ── Display ─────────────────────────────────────────────────────

    #[test]
    fn display_integer() {
        assert_eq!(format!("{}", rf_int(5)), "5");
    }

    #[test]
    fn display_polynomial() {
        let a = rf_poly(&[1, 1]); // x + 1
        let s = format!("{a}");
        assert!(s.contains("θ"), "should display as polynomial, got: {s}");
    }

    #[test]
    fn display_fraction() {
        let a = rf(&[1], &[1, 1]); // 1/(x+1)
        let s = format!("{a}");
        assert!(s.contains("/"), "should display as fraction, got: {s}");
    }

    // ── CoeffDisplay ────────────────────────────────────────────────

    fn format_rfc(c: &RationalFn, env: BindingStrength) -> String {
        struct W<'a>(&'a RationalFn, BindingStrength);
        impl fmt::Display for W<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt_coeff(f, self.1)
            }
        }
        format!("{}", W(c, env))
    }

    #[test]
    fn coeff_display_integer_in_product() {
        let a = rf_int(5);
        assert_eq!(format_rfc(&a, BindingStrength::Product), "5");
    }

    #[test]
    fn coeff_display_polynomial_in_product() {
        // (x+1) as a coefficient in a product context → needs parens
        let a = rf_poly(&[1, 1]);
        let s = format_rfc(&a, BindingStrength::Product);
        assert!(s.starts_with('('), "polynomial coeff in product should have parens: {s}");
    }

    #[test]
    fn coeff_display_fraction_in_product() {
        // 1/(x+1) as a coefficient in a product context
        let a = rf(&[1], &[1, 1]);
        let s = format_rfc(&a, BindingStrength::Product);
        assert!(s.starts_with('('), "fraction coeff in product should have parens: {s}");
    }

    #[test]
    fn coeff_display_fraction_at_top_level() {
        let a = rf(&[1], &[1, 1]);
        let s = format_rfc(&a, BindingStrength::Weakest);
        assert!(s.contains("/"), "fraction should show division: {s}");
    }

    // ── pow_usize ───────────────────────────────────────────────────

    #[test]
    fn pow_usize_basic() {
        // (1/x)^3 = 1/x³
        let a = rf(&[1], &[0, 1]); // 1/x
        let b = a.pow_usize(3);
        assert_eq!(b.numer().degree(), Some(0)); // numer = 1
        assert_eq!(b.denom().degree(), Some(3)); // denom = x³
    }

    #[test]
    fn pow_usize_zero() {
        let a = rf(&[1, 1], &[0, 1]); // (x+1)/x
        let b = a.pow_usize(0);
        assert!(Ring::is_one(&b));
    }

    // ── Distributivity (comprehensive Ring axiom) ───────────────────

    #[test]
    fn distributivity() {
        // a * (b + c) = a*b + a*c
        let a = rf(&[1], &[1, 1]);     // 1/(x+1)
        let b = rf(&[0, 1], &[1]);      // x
        let c = rf(&[1], &[-1, 1]);     // 1/(x-1)
        let lhs = Ring::mul(&a, &Ring::add(&b, &c));
        let rhs = Ring::add(&Ring::mul(&a, &b), &Ring::mul(&a, &c));
        assert_eq!(lhs, rhs, "distributivity failed:\n  lhs = {lhs}\n  rhs = {rhs}");
    }
}
