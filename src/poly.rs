//! Dense univariate polynomials over ℚ.
//!
//! This module provides [`Poly`], a dense univariate polynomial with
//! rational coefficients (`Ratio<BigInt>` from `num-rational`).
//!
//! # Representation
//!
//! A polynomial is stored as a `Vec<Ratio<BigInt>>` of coefficients in
//! ascending degree order: `coeffs[i]` is the coefficient of `x^i`.
//! The zero polynomial has an empty coefficient vector.  Non-zero
//! polynomials are kept in *normalised* form — the leading coefficient
//! (last element) is always nonzero.
//!
//! # Arithmetic
//!
//! Standard operations are implemented: addition, subtraction,
//! multiplication, Euclidean division (`div_rem`), and GCD.
//!
//! # GCD
//!
//! [`Poly::gcd`] computes the greatest common divisor of two
//! polynomials using the Euclidean algorithm.  The result is
//! normalised to be monic (leading coefficient = 1).

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};
use std::fmt;
use std::ops;

/// A dense univariate polynomial over ℚ.
///
/// Coefficients are stored in ascending degree order:
/// `p(x) = coeffs[0] + coeffs[1]*x + coeffs[2]*x² + …`
///
/// # Invariants
///
/// - The zero polynomial has `coeffs.is_empty()`.
/// - For non-zero polynomials, `coeffs.last()` is always nonzero.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Poly {
    /// Coefficients in ascending degree order.
    coeffs: Vec<Ratio<BigInt>>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Construction
// ═══════════════════════════════════════════════════════════════════════════

impl Poly {
    /// The zero polynomial.
    pub fn zero() -> Self {
        Poly { coeffs: Vec::new() }
    }

    /// A constant polynomial.
    pub fn constant(c: Ratio<BigInt>) -> Self {
        if c.is_zero() {
            Self::zero()
        } else {
            Poly { coeffs: vec![c] }
        }
    }

    /// A constant polynomial from an integer.
    pub fn from_int(n: i64) -> Self {
        Self::constant(Ratio::from_integer(BigInt::from(n)))
    }

    /// The monomial `x` (degree 1, coefficient 1).
    pub fn x() -> Self {
        Poly {
            coeffs: vec![Ratio::zero(), Ratio::one()],
        }
    }

    /// A monomial `c * x^n`.
    pub fn monomial(c: Ratio<BigInt>, degree: usize) -> Self {
        if c.is_zero() {
            return Self::zero();
        }
        let mut coeffs = vec![Ratio::zero(); degree + 1];
        coeffs[degree] = c;
        Poly { coeffs }
    }

    /// Construct from a vector of coefficients (ascending degree order).
    ///
    /// Automatically strips trailing zeros.
    pub fn from_coeffs(coeffs: Vec<Ratio<BigInt>>) -> Self {
        let mut p = Poly { coeffs };
        p.normalise();
        p
    }

    /// Strip trailing zero coefficients.
    fn normalise(&mut self) {
        while self.coeffs.last().is_some_and(|c| c.is_zero()) {
            self.coeffs.pop();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Queries
// ═══════════════════════════════════════════════════════════════════════════

impl Poly {
    /// Returns `true` if this is the zero polynomial.
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// The degree of the polynomial, or `None` for the zero polynomial.
    pub fn degree(&self) -> Option<usize> {
        if self.coeffs.is_empty() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    /// The leading coefficient, or `None` for the zero polynomial.
    pub fn leading_coeff(&self) -> Option<&Ratio<BigInt>> {
        self.coeffs.last()
    }

    /// The coefficient of `x^i`.  Returns zero for degrees beyond
    /// the polynomial's degree.
    pub fn coeff(&self, i: usize) -> Ratio<BigInt> {
        self.coeffs.get(i).cloned().unwrap_or_else(Ratio::zero)
    }

    /// The raw coefficient slice.
    pub fn coeffs(&self) -> &[Ratio<BigInt>] {
        &self.coeffs
    }

    /// Returns `true` if this polynomial has degree ≤ 0.
    pub fn is_constant(&self) -> bool {
        self.coeffs.len() <= 1
    }

    /// Evaluate the polynomial at a rational point using Horner's method.
    pub fn eval(&self, x: &Ratio<BigInt>) -> Ratio<BigInt> {
        if self.coeffs.is_empty() {
            return Ratio::zero();
        }
        let mut result = Ratio::zero();
        for c in self.coeffs.iter().rev() {
            result = result * x + c;
        }
        result
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Arithmetic: Add
// ═══════════════════════════════════════════════════════════════════════════

impl ops::Add for &Poly {
    type Output = Poly;

    fn add(self, rhs: &Poly) -> Poly {
        let len = self.coeffs.len().max(rhs.coeffs.len());
        let mut coeffs = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.coeffs.get(i).cloned().unwrap_or_else(Ratio::zero);
            let b = rhs.coeffs.get(i).cloned().unwrap_or_else(Ratio::zero);
            coeffs.push(a + b);
        }
        Poly::from_coeffs(coeffs)
    }
}

impl ops::Add for Poly {
    type Output = Poly;
    fn add(self, rhs: Poly) -> Poly {
        &self + &rhs
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Arithmetic: Sub
// ═══════════════════════════════════════════════════════════════════════════

impl ops::Sub for &Poly {
    type Output = Poly;

    fn sub(self, rhs: &Poly) -> Poly {
        let len = self.coeffs.len().max(rhs.coeffs.len());
        let mut coeffs = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.coeffs.get(i).cloned().unwrap_or_else(Ratio::zero);
            let b = rhs.coeffs.get(i).cloned().unwrap_or_else(Ratio::zero);
            coeffs.push(a - b);
        }
        Poly::from_coeffs(coeffs)
    }
}

impl ops::Sub for Poly {
    type Output = Poly;
    fn sub(self, rhs: Poly) -> Poly {
        &self - &rhs
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Arithmetic: Neg
// ═══════════════════════════════════════════════════════════════════════════

impl ops::Neg for &Poly {
    type Output = Poly;

    fn neg(self) -> Poly {
        Poly::from_coeffs(self.coeffs.iter().map(|c| -c).collect())
    }
}

impl ops::Neg for Poly {
    type Output = Poly;
    fn neg(self) -> Poly {
        -&self
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Arithmetic: Mul
// ═══════════════════════════════════════════════════════════════════════════

impl ops::Mul for &Poly {
    type Output = Poly;

    fn mul(self, rhs: &Poly) -> Poly {
        if self.is_zero() || rhs.is_zero() {
            return Poly::zero();
        }
        let n = self.coeffs.len();
        let m = rhs.coeffs.len();
        let mut coeffs = vec![Ratio::zero(); n + m - 1];
        for i in 0..n {
            for j in 0..m {
                coeffs[i + j] += &self.coeffs[i] * &rhs.coeffs[j];
            }
        }
        Poly::from_coeffs(coeffs)
    }
}

impl ops::Mul for Poly {
    type Output = Poly;
    fn mul(self, rhs: Poly) -> Poly {
        &self * &rhs
    }
}

/// Scalar multiplication: `c * poly`.
impl Poly {
    /// Multiply every coefficient by a scalar.
    #[must_use]
    pub fn scale(&self, c: &Ratio<BigInt>) -> Poly {
        if c.is_zero() {
            return Poly::zero();
        }
        Poly::from_coeffs(self.coeffs.iter().map(|a| a * c).collect())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Arithmetic: Euclidean division
// ═══════════════════════════════════════════════════════════════════════════

impl Poly {
    /// Euclidean division: `self = quotient * divisor + remainder`.
    ///
    /// Returns `(quotient, remainder)`.
    ///
    /// # Panics
    ///
    /// Panics if `divisor` is the zero polynomial.
    pub fn div_rem(&self, divisor: &Poly) -> (Poly, Poly) {
        assert!(!divisor.is_zero(), "polynomial division by zero");

        if self.is_zero() {
            return (Poly::zero(), Poly::zero());
        }

        let d_deg = divisor.degree().unwrap();
        let mut remainder = self.clone();

        if remainder.degree().is_none_or(|rd| rd < d_deg) {
            return (Poly::zero(), remainder);
        }

        let lc_inv = Ratio::one() / divisor.leading_coeff().unwrap();
        let mut quotient_coeffs = vec![Ratio::zero(); remainder.coeffs.len() - d_deg];

        while let Some(r_deg) = remainder.degree() {
            if r_deg < d_deg {
                break;
            }
            let coeff = remainder.leading_coeff().unwrap() * &lc_inv;
            let shift = r_deg - d_deg;
            quotient_coeffs[shift] = coeff.clone();

            // remainder -= coeff * x^shift * divisor
            for (i, dc) in divisor.coeffs.iter().enumerate() {
                remainder.coeffs[shift + i] -= &coeff * dc;
            }
            remainder.normalise();
        }

        (Poly::from_coeffs(quotient_coeffs), remainder)
    }

    /// Polynomial quotient: `self / divisor` (discarding remainder).
    #[must_use]
    pub fn div(&self, divisor: &Poly) -> Poly {
        self.div_rem(divisor).0
    }

    /// Polynomial remainder: `self % divisor`.
    #[must_use]
    pub fn rem(&self, divisor: &Poly) -> Poly {
        self.div_rem(divisor).1
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GCD
// ═══════════════════════════════════════════════════════════════════════════

/// GCD of two positive rationals: gcd(a/b, c/d) = gcd(a,c) / lcm(b,d).
fn rational_gcd(a: &Ratio<BigInt>, b: &Ratio<BigInt>) -> Ratio<BigInt> {
    use num_integer::Integer;
    let numer_gcd = a.numer().gcd(b.numer());
    let denom_lcm = a.denom().lcm(b.denom());
    Ratio::new(numer_gcd, denom_lcm)
}

impl Poly {
    /// Compute the GCD of two polynomials using the Euclidean algorithm.
    ///
    /// The result is normalised to be **monic** (leading coefficient = 1).
    /// The GCD of two zero polynomials is the zero polynomial.
    pub fn gcd(a: &Poly, b: &Poly) -> Poly {
        if a.is_zero() {
            return b.make_monic();
        }
        if b.is_zero() {
            return a.make_monic();
        }

        let mut r0 = a.clone();
        let mut r1 = b.clone();

        while !r1.is_zero() {
            let r2 = r0.rem(&r1);
            r0 = r1;
            r1 = r2;
        }

        r0.make_monic()
    }

    /// Return a monic version of this polynomial (leading coeff = 1).
    ///
    /// Returns the zero polynomial unchanged.
    #[must_use]
    pub fn make_monic(&self) -> Poly {
        if self.is_zero() {
            return Poly::zero();
        }
        let lc = self.leading_coeff().unwrap().clone();
        let lc_inv = Ratio::one() / lc;
        self.scale(&lc_inv)
    }

    /// Compute the content (GCD of all coefficients) of the polynomial.
    ///
    /// Returns 0 for the zero polynomial.
    pub fn content(&self) -> Ratio<BigInt> {
        if self.is_zero() {
            return Ratio::zero();
        }
        let coeffs: Vec<_> = (0..=self.degree().unwrap_or(0))
            .map(|i| self.coeff(i))
            .filter(|c| !c.is_zero())
            .collect();
        if coeffs.is_empty() {
            return Ratio::one();
        }
        // GCD of rationals: gcd(a/b, c/d) = gcd(a,c) / lcm(b,d)
        let mut result = coeffs[0].clone();
        for c in &coeffs[1..] {
            result = rational_gcd(&result, c);
        }
        if result.is_negative() {
            -result
        } else {
            result
        }
    }

    /// Compute the primitive part: `self / content`.
    #[must_use]
    pub fn primitive_part(&self) -> Self {
        let c = self.content();
        if c.is_one() || c.is_zero() {
            return self.clone();
        }
        self.scale(&(Ratio::one() / c))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Calculus / square-free
// ═══════════════════════════════════════════════════════════════════════════

impl Poly {
    /// Compute the formal derivative of this polynomial.
    /// For p(x) = a_0 + a_1 x + a_2 x^2 + ... + a_n x^n,
    /// p'(x) = a_1 + 2 a_2 x + ... + n a_n x^(n-1).
    #[must_use]
    pub fn derivative(&self) -> Poly {
        if self.coeffs.len() <= 1 {
            return Poly::zero();
        }
        let new_coeffs: Vec<Ratio<BigInt>> = self.coeffs[1..]
            .iter()
            .enumerate()
            .map(|(i, c)| c * Ratio::from_integer(BigInt::from(i as i64 + 1)))
            .collect();
        Poly::from_coeffs(new_coeffs)
    }

    /// Compute the square-free part: p / gcd(p, p').
    /// This removes repeated roots while preserving all distinct roots.
    #[must_use]
    pub fn square_free_part(&self) -> Poly {
        if self.is_zero() {
            return Poly::zero();
        }
        let dp = self.derivative();
        if dp.is_zero() {
            return self.clone(); // constant polynomial
        }
        let g = Poly::gcd(self, &dp);
        self.div_rem(&g).0 // quotient only
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Display
// ═══════════════════════════════════════════════════════════════════════════

impl fmt::Display for Poly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;
        for (i, c) in self.coeffs.iter().enumerate().rev() {
            if c.is_zero() {
                continue;
            }

            let is_neg = c.is_negative();
            let abs_c = c.abs();
            let is_one = abs_c.is_one();

            // Sign / separator.
            if first {
                if is_neg {
                    write!(f, "-")?;
                }
                first = false;
            } else if is_neg {
                write!(f, " - ")?;
            } else {
                write!(f, " + ")?;
            }

            // Coefficient and variable.
            if i == 0 {
                // Constant term: always print the coefficient.
                write!(f, "{abs_c}")?;
            } else if is_one {
                // Coefficient is ±1: omit it.
                if i == 1 {
                    write!(f, "x")?;
                } else {
                    write!(f, "x^{i}")?;
                }
            } else {
                // General case.
                if i == 1 {
                    write!(f, "{abs_c}*x")?;
                } else {
                    write!(f, "{abs_c}*x^{i}")?;
                }
            }
        }

        Ok(())
    }
}

impl fmt::Debug for Poly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Poly({self})")
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

    fn ri(n: i64) -> Ratio<BigInt> {
        Ratio::from_integer(BigInt::from(n))
    }

    // ── Construction ────────────────────────────────────────────────

    #[test]
    fn zero_polynomial() {
        let p = Poly::zero();
        assert!(p.is_zero());
        assert_eq!(p.degree(), None);
        assert_eq!(format!("{p}"), "0");
    }

    #[test]
    fn constant_polynomial() {
        let p = Poly::from_int(5);
        assert!(!p.is_zero());
        assert_eq!(p.degree(), Some(0));
        assert_eq!(format!("{p}"), "5");
    }

    #[test]
    fn monomial_x() {
        let p = Poly::x();
        assert_eq!(p.degree(), Some(1));
        assert_eq!(format!("{p}"), "x");
    }

    #[test]
    fn from_coeffs_strips_trailing_zeros() {
        let p = Poly::from_coeffs(vec![ri(1), ri(2), ri(0), ri(0)]);
        assert_eq!(p.degree(), Some(1));
    }

    #[test]
    fn display_polynomial() {
        // x^2 + 2x + 1
        let p = Poly::from_coeffs(vec![ri(1), ri(2), ri(1)]);
        assert_eq!(format!("{p}"), "x^2 + 2*x + 1");
    }

    #[test]
    fn display_negative_terms() {
        // x^2 - 3x + 2
        let p = Poly::from_coeffs(vec![ri(2), ri(-3), ri(1)]);
        assert_eq!(format!("{p}"), "x^2 - 3*x + 2");
    }

    #[test]
    fn display_rational_coeffs() {
        // (1/2)x + 1/3
        let p = Poly::from_coeffs(vec![r(1, 3), r(1, 2)]);
        assert_eq!(format!("{p}"), "1/2*x + 1/3");
    }

    // ── Evaluation ──────────────────────────────────────────────────

    #[test]
    fn eval_polynomial() {
        // x^2 + 2x + 1 at x=3 → 9 + 6 + 1 = 16
        let p = Poly::from_coeffs(vec![ri(1), ri(2), ri(1)]);
        assert_eq!(p.eval(&ri(3)), ri(16));
    }

    #[test]
    fn eval_zero_polynomial() {
        let p = Poly::zero();
        assert_eq!(p.eval(&ri(42)), ri(0));
    }

    // ── Add ─────────────────────────────────────────────────────────

    #[test]
    fn add_polynomials() {
        // (x + 1) + (x^2 + 2) = x^2 + x + 3
        let a = Poly::from_coeffs(vec![ri(1), ri(1)]);
        let b = Poly::from_coeffs(vec![ri(2), ri(0), ri(1)]);
        let c = &a + &b;
        assert_eq!(format!("{c}"), "x^2 + x + 3");
    }

    #[test]
    fn add_cancellation() {
        // (x + 1) + (-x + 2) = 3
        let a = Poly::from_coeffs(vec![ri(1), ri(1)]);
        let b = Poly::from_coeffs(vec![ri(2), ri(-1)]);
        let c = &a + &b;
        assert_eq!(format!("{c}"), "3");
    }

    #[test]
    fn add_to_zero() {
        let a = Poly::from_coeffs(vec![ri(1), ri(2)]);
        let b = Poly::from_coeffs(vec![ri(-1), ri(-2)]);
        let c = &a + &b;
        assert!(c.is_zero());
    }

    // ── Sub ─────────────────────────────────────────────────────────

    #[test]
    fn sub_polynomials() {
        // (x^2 + x) - (x - 1) = x^2 + 1
        let a = Poly::from_coeffs(vec![ri(0), ri(1), ri(1)]);
        let b = Poly::from_coeffs(vec![ri(-1), ri(1)]);
        let c = &a - &b;
        assert_eq!(format!("{c}"), "x^2 + 1");
    }

    #[test]
    fn sub_self_is_zero() {
        let a = Poly::from_coeffs(vec![ri(3), ri(2), ri(1)]);
        let c = &a - &a;
        assert!(c.is_zero());
    }

    // ── Neg ─────────────────────────────────────────────────────────

    #[test]
    fn neg_polynomial() {
        let a = Poly::from_coeffs(vec![ri(1), ri(-2), ri(3)]);
        let b = -&a;
        assert_eq!(format!("{b}"), "-3*x^2 + 2*x - 1");
    }

    // ── Mul ─────────────────────────────────────────────────────────

    #[test]
    fn mul_polynomials() {
        // (x + 1) * (x - 1) = x^2 - 1
        let a = Poly::from_coeffs(vec![ri(1), ri(1)]);
        let b = Poly::from_coeffs(vec![ri(-1), ri(1)]);
        let c = &a * &b;
        assert_eq!(format!("{c}"), "x^2 - 1");
    }

    #[test]
    fn mul_by_zero() {
        let a = Poly::from_coeffs(vec![ri(1), ri(2), ri(3)]);
        let b = Poly::zero();
        assert!((&a * &b).is_zero());
    }

    #[test]
    fn mul_by_constant() {
        let a = Poly::from_coeffs(vec![ri(1), ri(2)]);
        let b = Poly::from_int(3);
        let c = &a * &b;
        assert_eq!(format!("{c}"), "6*x + 3");
    }

    #[test]
    fn mul_larger() {
        // (x + 1)^2 = x^2 + 2x + 1
        let a = Poly::from_coeffs(vec![ri(1), ri(1)]);
        let c = &a * &a;
        assert_eq!(format!("{c}"), "x^2 + 2*x + 1");
    }

    #[test]
    fn scale_polynomial() {
        let a = Poly::from_coeffs(vec![ri(2), ri(4)]);
        let b = a.scale(&r(1, 2));
        assert_eq!(format!("{b}"), "2*x + 1");
    }

    // ── Div / Rem ───────────────────────────────────────────────────

    #[test]
    fn div_rem_exact() {
        // (x^2 - 1) / (x - 1) = (x + 1), remainder 0
        let dividend = Poly::from_coeffs(vec![ri(-1), ri(0), ri(1)]);
        let divisor = Poly::from_coeffs(vec![ri(-1), ri(1)]);
        let (q, r) = dividend.div_rem(&divisor);
        assert_eq!(format!("{q}"), "x + 1");
        assert!(r.is_zero(), "remainder should be zero, got: {r}");
    }

    #[test]
    fn div_rem_with_remainder() {
        // (x^2 + 1) / (x - 1) = (x + 1), remainder 2
        let dividend = Poly::from_coeffs(vec![ri(1), ri(0), ri(1)]);
        let divisor = Poly::from_coeffs(vec![ri(-1), ri(1)]);
        let (q, r) = dividend.div_rem(&divisor);
        assert_eq!(format!("{q}"), "x + 1");
        assert_eq!(format!("{r}"), "2");
    }

    #[test]
    fn div_rem_lower_degree() {
        // (x + 1) / (x^2 + 1) = 0, remainder (x + 1)
        let dividend = Poly::from_coeffs(vec![ri(1), ri(1)]);
        let divisor = Poly::from_coeffs(vec![ri(1), ri(0), ri(1)]);
        let (q, r) = dividend.div_rem(&divisor);
        assert!(q.is_zero());
        assert_eq!(format!("{r}"), "x + 1");
    }

    #[test]
    fn div_rem_constant_divisor() {
        // (2x + 4) / 2 = (x + 2), remainder 0
        let dividend = Poly::from_coeffs(vec![ri(4), ri(2)]);
        let divisor = Poly::from_int(2);
        let (q, r) = dividend.div_rem(&divisor);
        assert_eq!(format!("{q}"), "x + 2");
        assert!(r.is_zero());
    }

    #[test]
    #[should_panic(expected = "polynomial division by zero")]
    fn div_by_zero_panics() {
        let a = Poly::from_int(1);
        let _ = a.div_rem(&Poly::zero());
    }

    // ── GCD ─────────────────────────────────────────────────────────

    #[test]
    fn gcd_coprime() {
        // gcd(x + 1, x + 2) = 1  (coprime)
        let a = Poly::from_coeffs(vec![ri(1), ri(1)]);
        let b = Poly::from_coeffs(vec![ri(2), ri(1)]);
        let g = Poly::gcd(&a, &b);
        assert_eq!(format!("{g}"), "1");
    }

    #[test]
    fn gcd_common_factor() {
        // gcd(x^2 - 1, x^2 - 2x + 1) = gcd((x-1)(x+1), (x-1)^2) = x - 1
        let a = Poly::from_coeffs(vec![ri(-1), ri(0), ri(1)]); // x^2 - 1
        let b = Poly::from_coeffs(vec![ri(1), ri(-2), ri(1)]); // x^2 - 2x + 1
        let g = Poly::gcd(&a, &b);
        // Should be monic: x - 1 (or equivalent)
        assert_eq!(g.degree(), Some(1), "GCD should be degree 1");
        // Verify by checking that g divides both a and b.
        assert!(a.rem(&g).is_zero(), "g should divide a");
        assert!(b.rem(&g).is_zero(), "g should divide b");
    }

    #[test]
    fn gcd_identical() {
        let a = Poly::from_coeffs(vec![ri(1), ri(2), ri(1)]);
        let g = Poly::gcd(&a, &a);
        // GCD of a with itself is the monic version of a.
        assert_eq!(g.degree(), a.degree());
        assert!(a.rem(&g).is_zero());
    }

    #[test]
    fn gcd_with_zero() {
        let a = Poly::from_coeffs(vec![ri(2), ri(4)]);
        let g = Poly::gcd(&a, &Poly::zero());
        // GCD(a, 0) = monic(a) = x + 1/2 ... actually monic(2x + 2) = x + 1.
        // Wait: 2x + 4. monic → x + 2.
        assert_eq!(g.degree(), Some(1));
        assert_eq!(g.leading_coeff(), Some(&ri(1)));
    }

    #[test]
    fn gcd_both_zero() {
        let g = Poly::gcd(&Poly::zero(), &Poly::zero());
        assert!(g.is_zero());
    }

    #[test]
    fn gcd_quadratic_factors() {
        // a = (x - 1)(x - 2)(x - 3) = x^3 - 6x^2 + 11x - 6
        // b = (x - 1)(x - 2)(x - 4) = x^3 - 7x^2 + 14x - 8
        // gcd = (x - 1)(x - 2) = x^2 - 3x + 2
        let x_minus = |n: i64| Poly::from_coeffs(vec![ri(-n), ri(1)]);
        let a = &(&x_minus(1) * &x_minus(2)) * &x_minus(3);
        let b = &(&x_minus(1) * &x_minus(2)) * &x_minus(4);
        let g = Poly::gcd(&a, &b);
        assert_eq!(g.degree(), Some(2));
        // Verify it divides both.
        assert!(a.rem(&g).is_zero(), "g should divide a");
        assert!(b.rem(&g).is_zero(), "g should divide b");
        // Check it's the right polynomial (x^2 - 3x + 2 monic).
        assert_eq!(g.eval(&ri(1)), ri(0), "g(1) should be 0");
        assert_eq!(g.eval(&ri(2)), ri(0), "g(2) should be 0");
    }

    // ── Monic / Primitive ───────────────────────────────────────────

    #[test]
    fn make_monic() {
        let p = Poly::from_coeffs(vec![ri(2), ri(4)]);
        let m = p.make_monic();
        assert_eq!(m.leading_coeff(), Some(&ri(1)));
        assert_eq!(format!("{m}"), "x + 1/2");
    }

    #[test]
    fn make_monic_already_monic() {
        let p = Poly::from_coeffs(vec![ri(1), ri(1)]);
        let m = p.make_monic();
        assert_eq!(p, m);
    }

    #[test]
    fn make_monic_zero() {
        let p = Poly::zero();
        let m = p.make_monic();
        assert!(m.is_zero());
    }

    // ── Correctness via evaluation ──────────────────────────────────

    #[test]
    fn mul_and_div_roundtrip() {
        // (x + 1) * (x + 2) = x^2 + 3x + 2
        // divided by (x + 1) should give (x + 2)
        let a = Poly::from_coeffs(vec![ri(1), ri(1)]);
        let b = Poly::from_coeffs(vec![ri(2), ri(1)]);
        let product = &a * &b;
        let (q, r) = product.div_rem(&a);
        assert_eq!(q, b);
        assert!(r.is_zero());
    }

    #[test]
    fn div_rem_identity() {
        // For any a, b (b ≠ 0): a = q*b + r
        let a = Poly::from_coeffs(vec![ri(1), ri(-3), ri(0), ri(2), ri(1)]);
        let b = Poly::from_coeffs(vec![ri(1), ri(1)]);
        let (q, r) = a.div_rem(&b);
        let reconstructed = &(&q * &b) + &r;
        assert_eq!(a, reconstructed, "a should equal q*b + r");
    }

    // ── Deep nesting (verify no stack issues) ───────────────────────

    #[test]
    fn multiply_many_linear_factors() {
        // (x-1)(x-2)...(x-10) — degree 10 polynomial
        let mut result = Poly::from_int(1);
        for i in 1..=10 {
            let factor = Poly::from_coeffs(vec![ri(-i), ri(1)]);
            result = &result * &factor;
        }
        assert_eq!(result.degree(), Some(10));
        // All roots should evaluate to 0.
        for i in 1..=10 {
            assert_eq!(result.eval(&ri(i)), ri(0), "should be zero at x={i}");
        }
    }

    #[test]
    fn gcd_of_products() {
        // gcd((x-1)(x-2)(x-3)(x-4), (x-2)(x-4)(x-6)) = (x-2)(x-4)
        let x_minus = |n: i64| Poly::from_coeffs(vec![ri(-n), ri(1)]);
        let a = &(&(&x_minus(1) * &x_minus(2)) * &x_minus(3)) * &x_minus(4);
        let b = &(&x_minus(2) * &x_minus(4)) * &x_minus(6);
        let g = Poly::gcd(&a, &b);
        assert_eq!(g.degree(), Some(2));
        assert_eq!(g.eval(&ri(2)), ri(0));
        assert_eq!(g.eval(&ri(4)), ri(0));
        assert!(g.eval(&ri(1)) != ri(0), "x=1 should not be a root of gcd");
    }

    // ── Content / Primitive part ────────────────────────────────────

    #[test]
    fn content_of_2x_plus_4() {
        // 2x + 4 → content = 2
        let p = Poly::from_coeffs(vec![ri(4), ri(2)]);
        assert_eq!(p.content(), ri(2));
    }

    #[test]
    fn content_of_6x2_4x_2() {
        // 6x² + 4x + 2 → content = 2
        let p = Poly::from_coeffs(vec![ri(2), ri(4), ri(6)]);
        assert_eq!(p.content(), ri(2));
    }

    #[test]
    fn content_of_x_plus_1() {
        // x + 1 → content = 1
        let p = Poly::from_coeffs(vec![ri(1), ri(1)]);
        assert_eq!(p.content(), ri(1));
    }

    #[test]
    fn primitive_part_of_2x_plus_4() {
        // 2x + 4 → primitive part = x + 2
        let p = Poly::from_coeffs(vec![ri(4), ri(2)]);
        let pp = p.primitive_part();
        let expected = Poly::from_coeffs(vec![ri(2), ri(1)]);
        assert_eq!(pp, expected);
    }
}
