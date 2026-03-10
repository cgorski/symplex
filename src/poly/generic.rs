//! Generic dense univariate polynomial over an arbitrary coefficient ring.
//!
//! [`GenPoly<C>`] is parameterized by the coefficient type `C`, which must
//! implement [`Ring`] for basic operations and [`Field`] for division,
//! GCD, and factorization.
//!
//! This complements the existing [`Poly`](super::dense::Poly) type (which
//! is hardcoded to `Ratio<BigInt>` coefficients) by enabling polynomial
//! arithmetic over richer coefficient fields like `RationalFn` (rational
//! functions of x).  The same GCD, Hermite reduction, and Rothstein-Trager
//! algorithms work at any tower level.
//!
//! # Design
//!
//! Bounds are at the method level, not the struct level:
//! - `impl<C: Ring> GenPoly<C>` — add, mul, derivative, etc.
//! - `impl<C: Field> GenPoly<C>` — div_rem, gcd, extended_gcd, etc.
//!
//! # Representation
//!
//! Coefficients are stored in ascending degree order: `coeffs[i]` is the
//! coefficient of θ^i.  The zero polynomial has an empty coefficient vector.
//! Non-zero polynomials are normalized: the leading coefficient is nonzero.

use std::fmt;

use super::traits::{Ring, Field, CoeffDisplay, BindingStrength};

// ═══════════════════════════════════════════════════════════════════════════
// The type
// ═══════════════════════════════════════════════════════════════════════════

/// A dense univariate polynomial with coefficients in `C`.
///
/// `C` must implement at least [`Ring`] for basic arithmetic.
/// [`Field`] is required for division, GCD, and related algorithms.
#[derive(Clone, PartialEq)]
pub struct GenPoly<C> {
    /// Coefficients in ascending degree order.
    /// `coeffs[i]` is the coefficient of θ^i.
    /// Invariant: if non-empty, the last element is nonzero.
    coeffs: Vec<C>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Ring-level operations (require C: Ring)
// ═══════════════════════════════════════════════════════════════════════════

impl<C: Ring> GenPoly<C> {
    /// The zero polynomial.
    pub fn zero() -> Self {
        GenPoly { coeffs: Vec::new() }
    }

    /// The constant polynomial `1`.
    pub fn one() -> Self {
        GenPoly { coeffs: vec![C::one()] }
    }

    /// A constant polynomial.
    pub fn constant(c: C) -> Self {
        if c.is_zero() {
            Self::zero()
        } else {
            GenPoly { coeffs: vec![c] }
        }
    }

    /// The monomial `c · θ^degree`.
    pub fn monomial(c: C, degree: usize) -> Self {
        if c.is_zero() {
            return Self::zero();
        }
        let mut coeffs = Vec::with_capacity(degree + 1);
        for _ in 0..degree {
            coeffs.push(C::zero());
        }
        coeffs.push(c);
        GenPoly { coeffs }
    }

    /// The variable `θ` (= `1 · θ^1`).
    pub fn x() -> Self {
        Self::monomial(C::one(), 1)
    }

    /// Construct from a vector of coefficients (ascending degree order).
    pub fn from_coeffs(coeffs: Vec<C>) -> Self {
        let mut p = GenPoly { coeffs };
        p.normalize();
        p
    }

    /// Is this the zero polynomial?
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Is this a constant (degree 0 or zero)?
    pub fn is_constant(&self) -> bool {
        self.coeffs.len() <= 1
    }

    /// Degree of the polynomial, or `None` for the zero polynomial.
    pub fn degree(&self) -> Option<usize> {
        if self.coeffs.is_empty() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    /// Leading coefficient, or `None` for the zero polynomial.
    pub fn leading_coeff(&self) -> Option<&C> {
        self.coeffs.last()
    }

    /// Coefficient of θ^i (returns zero if i is out of range).
    pub fn coeff(&self, i: usize) -> C {
        self.coeffs.get(i).cloned().unwrap_or_else(C::zero)
    }

    /// All coefficients (ascending degree order).
    pub fn coeffs(&self) -> &[C] {
        &self.coeffs
    }

    /// Remove trailing zero coefficients.
    pub fn normalize(&mut self) {
        while self.coeffs.last().is_some_and(|c| c.is_zero()) {
            self.coeffs.pop();
        }
    }

    /// Polynomial addition.
    pub fn add(&self, rhs: &Self) -> Self {
        let len = self.coeffs.len().max(rhs.coeffs.len());
        let mut coeffs = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.coeff(i);
            let b = rhs.coeff(i);
            coeffs.push(Ring::add(&a, &b));
        }
        let mut p = GenPoly { coeffs };
        p.normalize();
        p
    }

    /// Polynomial subtraction.
    pub fn sub(&self, rhs: &Self) -> Self {
        let len = self.coeffs.len().max(rhs.coeffs.len());
        let mut coeffs = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.coeff(i);
            let b = rhs.coeff(i);
            coeffs.push(Ring::sub(&a, &b));
        }
        let mut p = GenPoly { coeffs };
        p.normalize();
        p
    }

    /// Polynomial negation.
    pub fn neg(&self) -> Self {
        GenPoly {
            coeffs: self.coeffs.iter().map(|c| Ring::neg(c)).collect(),
        }
    }

    /// Polynomial multiplication.
    pub fn mul(&self, rhs: &Self) -> Self {
        if self.is_zero() || rhs.is_zero() {
            return Self::zero();
        }
        let len = self.coeffs.len() + rhs.coeffs.len() - 1;
        let mut coeffs = vec![C::zero(); len];
        for (i, a) in self.coeffs.iter().enumerate() {
            if a.is_zero() {
                continue;
            }
            for (j, b) in rhs.coeffs.iter().enumerate() {
                let product = Ring::mul(a, b);
                coeffs[i + j] = Ring::add(&coeffs[i + j], &product);
            }
        }
        let mut p = GenPoly { coeffs };
        p.normalize();
        p
    }

    /// Multiply by a scalar.
    pub fn scale(&self, c: &C) -> Self {
        if c.is_zero() {
            return Self::zero();
        }
        GenPoly {
            coeffs: self.coeffs.iter().map(|a| Ring::mul(a, c)).collect(),
        }
    }

    /// Formal derivative: d/dθ of the polynomial.
    ///
    /// For `p(θ) = Σ aₖ θᵏ`, returns `Σ k·aₖ θ^(k-1)`.
    ///
    /// The integer multiplier `k` is computed by repeated addition of the
    /// coefficient with itself (since we only have Ring, not a characteristic
    /// map from ℤ).
    pub fn derivative(&self) -> Self {
        if self.coeffs.len() <= 1 {
            return Self::zero();
        }
        let mut coeffs = Vec::with_capacity(self.coeffs.len() - 1);
        for (i, c) in self.coeffs.iter().enumerate().skip(1) {
            // Multiply c by i (the integer k) using repeated addition.
            let mut k_times_c = c.clone();
            for _ in 1..i {
                k_times_c = Ring::add(&k_times_c, c);
            }
            coeffs.push(k_times_c);
        }
        let mut p = GenPoly { coeffs };
        p.normalize();
        p
    }

    /// Evaluate the polynomial at a point using Horner's method.
    pub fn eval(&self, x: &C) -> C {
        if self.is_zero() {
            return C::zero();
        }
        let mut result = self.coeffs.last().unwrap().clone();
        for c in self.coeffs.iter().rev().skip(1) {
            result = Ring::add(&Ring::mul(&result, x), c);
        }
        result
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Operator overloads for ergonomic syntax
// ═══════════════════════════════════════════════════════════════════════════

impl<C: Ring> std::ops::Add for &GenPoly<C> {
    type Output = GenPoly<C>;
    fn add(self, rhs: &GenPoly<C>) -> GenPoly<C> {
        GenPoly::add(self, rhs)
    }
}

impl<C: Ring> std::ops::Sub for &GenPoly<C> {
    type Output = GenPoly<C>;
    fn sub(self, rhs: &GenPoly<C>) -> GenPoly<C> {
        GenPoly::sub(self, rhs)
    }
}

impl<C: Ring> std::ops::Mul for &GenPoly<C> {
    type Output = GenPoly<C>;
    fn mul(self, rhs: &GenPoly<C>) -> GenPoly<C> {
        GenPoly::mul(self, rhs)
    }
}

impl<C: Ring> std::ops::Neg for &GenPoly<C> {
    type Output = GenPoly<C>;
    fn neg(self) -> GenPoly<C> {
        GenPoly::neg(self)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Field-level operations (require C: Field)
// ═══════════════════════════════════════════════════════════════════════════

impl<C: Field> GenPoly<C> {
    /// Euclidean division: `self = quotient * divisor + remainder`
    /// with `deg(remainder) < deg(divisor)`.
    ///
    /// # Panics
    ///
    /// Panics if `divisor` is zero.
    pub fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        assert!(!divisor.is_zero(), "division by zero polynomial");

        let d_deg = match divisor.degree() {
            Some(d) => d,
            None => panic!("division by zero polynomial"),
        };
        let d_lc = divisor.leading_coeff().unwrap();

        let mut rem = self.clone();

        let q_len = if let Some(r_deg) = rem.degree() {
            if r_deg >= d_deg { r_deg - d_deg + 1 } else { 0 }
        } else {
            0
        };
        let mut quot_coeffs = vec![C::zero(); q_len];

        while let Some(r_deg) = rem.degree() {
            if r_deg < d_deg {
                break;
            }
            let r_lc = rem.leading_coeff().unwrap().clone();
            let coeff = Field::div(&r_lc, d_lc);
            let shift = r_deg - d_deg;
            quot_coeffs[shift] = coeff.clone();

            // rem = rem - coeff * θ^shift * divisor
            for (i, dc) in divisor.coeffs.iter().enumerate() {
                let idx = i + shift;
                if idx < rem.coeffs.len() {
                    let sub = Ring::mul(&coeff, dc);
                    rem.coeffs[idx] = Ring::sub(&rem.coeffs[idx], &sub);
                }
            }
            rem.normalize();
        }

        let mut q = GenPoly { coeffs: quot_coeffs };
        q.normalize();
        (q, rem)
    }

    /// Polynomial division (quotient only).
    pub fn div(&self, divisor: &Self) -> Self {
        self.div_rem(divisor).0
    }

    /// Polynomial remainder.
    pub fn rem(&self, divisor: &Self) -> Self {
        self.div_rem(divisor).1
    }

    /// Make the polynomial monic (leading coefficient = 1).
    ///
    /// Returns the zero polynomial unchanged.
    pub fn make_monic(&self) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let lc = self.leading_coeff().unwrap();
        if lc.is_one() {
            return self.clone();
        }
        let inv_lc = Field::inv(lc);
        self.scale(&inv_lc)
    }

    /// Greatest common divisor via the Euclidean algorithm.
    ///
    /// The result is monic (leading coefficient = 1).
    pub fn gcd(a: &Self, b: &Self) -> Self {
        let mut a = a.clone();
        let mut b = b.clone();

        while !b.is_zero() {
            let (_, r) = a.div_rem(&b);
            a = b;
            b = r;
        }

        if a.is_zero() {
            return a;
        }
        a.make_monic()
    }

    /// Extended Euclidean algorithm.
    ///
    /// Returns `(s, t, g)` where `s*a + t*b = g` and `g = gcd(a, b)` (monic).
    pub fn extended_gcd(a: &Self, b: &Self) -> (Self, Self, Self) {
        if b.is_zero() {
            if a.is_zero() {
                return (Self::one(), Self::zero(), Self::zero());
            }
            let lc = a.leading_coeff().unwrap();
            let inv_lc = Field::inv(lc);
            let s = Self::constant(inv_lc.clone());
            return (s, Self::zero(), a.make_monic());
        }

        let (mut r_prev, mut r_curr) = (a.clone(), b.clone());
        let (mut s_prev, mut s_curr) = (Self::one(), Self::zero());
        let (mut t_prev, mut t_curr) = (Self::zero(), Self::one());

        while !r_curr.is_zero() {
            let (q, r_next) = r_prev.div_rem(&r_curr);
            let neg_q = q.neg();
            let s_next = &s_prev + &(&neg_q * &s_curr);
            let t_next = &t_prev + &(&neg_q * &t_curr);

            r_prev = r_curr;
            r_curr = r_next;
            s_prev = s_curr;
            s_curr = s_next;
            t_prev = t_curr;
            t_curr = t_next;
        }

        // Normalize to monic GCD.
        if !r_prev.is_zero() {
            let lc = r_prev.leading_coeff().unwrap();
            let inv_lc = Field::inv(lc);
            r_prev = r_prev.scale(&inv_lc);
            s_prev = s_prev.scale(&inv_lc);
            t_prev = t_prev.scale(&inv_lc);
        }

        (s_prev, t_prev, r_prev)
    }

    /// Square-free part: `p / gcd(p, p')`.
    ///
    /// Removes repeated roots while preserving all distinct roots.
    pub fn square_free_part(&self) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let dp = self.derivative();
        if dp.is_zero() {
            return self.clone();
        }
        let g = Self::gcd(self, &dp);
        self.div(&g)
    }

    /// Square-free factorization via Yun's algorithm.
    ///
    /// Returns `[(f₁, 1), (f₂, 2), …]` where `self = lc · ∏ fᵢ^i`
    /// and each `fᵢ` is square-free and pairwise coprime.
    ///
    /// Constant/zero polynomials return an empty list.
    pub fn squarefree_factors(&self) -> Vec<(Self, usize)> {
        if self.is_zero() || self.is_constant() {
            return vec![];
        }

        let p = self.make_monic();
        let p_prime = p.derivative();
        let c = Self::gcd(&p, &p_prime);

        if c.degree().unwrap_or(0) == 0 {
            // Already square-free.
            return vec![(p, 1)];
        }

        let mut w = p.div(&c);
        let mut y = p_prime.div(&c);

        let mut factors = Vec::new();
        let mut i = 1;

        loop {
            let w_prime = w.derivative();
            let z = y.sub(&w_prime);

            if z.is_zero() {
                if w.degree().unwrap_or(0) > 0 {
                    factors.push((w.make_monic(), i));
                }
                break;
            }

            let g = Self::gcd(&w, &z);
            if g.degree().unwrap_or(0) > 0 {
                factors.push((g.make_monic(), i));
            }

            w = w.div(&g);
            y = z.div(&g);
            i += 1;
        }

        factors
    }

    /// Compute the resultant of two polynomials via the Euclidean algorithm.
    ///
    /// The resultant is zero iff the two polynomials share a common root.
    /// Returns an element of `C`.
    pub fn resultant(a: &Self, b: &Self) -> C {
        if a.is_zero() || b.is_zero() {
            return C::zero();
        }

        let m = match a.degree() {
            Some(d) => d,
            None => return C::zero(),
        };
        let n = match b.degree() {
            Some(d) => d,
            None => return C::zero(),
        };

        if m == 0 && n == 0 {
            return C::one();
        }
        if m == 0 {
            return a.coeff(0).pow_usize(n);
        }
        if n == 0 {
            return b.coeff(0).pow_usize(m);
        }

        // Ensure deg(a) >= deg(b).
        if m < n {
            let sign = if (m * n) % 2 == 0 {
                C::one()
            } else {
                Ring::neg(&C::one())
            };
            return Ring::mul(&sign, &Self::resultant(b, a));
        }

        let r = a.rem(b);
        if r.is_zero() {
            return C::zero();
        }

        let s = r.degree().unwrap_or(0);
        let sign = if (m * n) % 2 == 0 {
            C::one()
        } else {
            Ring::neg(&C::one())
        };
        let lc_b = b.leading_coeff().unwrap().clone();
        let factor = lc_b.pow_usize(m - s);

        Ring::mul(&Ring::mul(&sign, &factor), &Self::resultant(b, &r))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Display
// ═══════════════════════════════════════════════════════════════════════════

impl<C: Ring + CoeffDisplay> fmt::Display for GenPoly<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;
        // Print in descending degree order for readability.
        for i in (0..self.coeffs.len()).rev() {
            let c = &self.coeffs[i];
            if c.is_zero() {
                continue;
            }
            if !first {
                write!(f, " + ")?;
            }
            first = false;

            match i {
                0 => {
                    c.fmt_coeff(f, BindingStrength::Sum)?;
                }
                1 => {
                    if c.is_one() {
                        write!(f, "θ")?;
                    } else {
                        c.fmt_coeff(f, BindingStrength::Product)?;
                        write!(f, "*θ")?;
                    }
                }
                n => {
                    if c.is_one() {
                        write!(f, "θ^{}", n)?;
                    } else {
                        c.fmt_coeff(f, BindingStrength::Product)?;
                        write!(f, "*θ^{}", n)?;
                    }
                }
            }
        }

        if first {
            write!(f, "0")?;
        }
        Ok(())
    }
}

impl<C: Ring + fmt::Debug> fmt::Debug for GenPoly<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GenPoly({:?})", self.coeffs)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Ring/EuclideanDomain impl for GenPoly<C> (polynomials over a field form a ED)
// ═══════════════════════════════════════════════════════════════════════════

impl<C: Ring> Ring for GenPoly<C> {
    fn zero() -> Self { GenPoly::zero() }
    fn one() -> Self { GenPoly::one() }
    fn is_zero(&self) -> bool { self.is_zero() }

    fn add(&self, rhs: &Self) -> Self { GenPoly::add(self, rhs) }
    fn sub(&self, rhs: &Self) -> Self { GenPoly::sub(self, rhs) }
    fn mul(&self, rhs: &Self) -> Self { GenPoly::mul(self, rhs) }
    fn neg(&self) -> Self { GenPoly::neg(self) }
}

impl<C: Field> super::traits::EuclideanDomain for GenPoly<C> {
    fn div_rem(&self, other: &Self) -> (Self, Self) {
        GenPoly::div_rem(self, other)
    }

    fn gcd(a: &Self, b: &Self) -> Self {
        GenPoly::gcd(a, b)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_rational::Ratio;

    type Q = Ratio<BigInt>;
    type P = GenPoly<Q>;

    fn q(n: i64, d: i64) -> Q {
        Ratio::new(BigInt::from(n), BigInt::from(d))
    }

    fn qz() -> Q { <Q as Ring>::zero() }

    /// Build a polynomial from integer coefficients (ascending degree).
    fn p(cs: &[i64]) -> P {
        GenPoly::from_coeffs(cs.iter().map(|&c| q(c, 1)).collect())
    }

    // ── Construction ────────────────────────────────────────────────

    #[test]
    fn zero_polynomial() {
        let z = P::zero();
        assert!(z.is_zero());
        assert_eq!(z.degree(), None);
        assert!(z.is_constant());
    }

    #[test]
    fn one_polynomial() {
        let o = P::one();
        assert!(!o.is_zero());
        assert_eq!(o.degree(), Some(0));
        assert!(o.is_constant());
    }

    #[test]
    fn constant_polynomial() {
        let c = P::constant(q(5, 1));
        assert_eq!(c.degree(), Some(0));
        assert_eq!(c.coeff(0), q(5, 1));
    }

    #[test]
    fn monomial_polynomial() {
        // 3θ²
        let m = P::monomial(q(3, 1), 2);
        assert_eq!(m.degree(), Some(2));
        assert_eq!(m.coeff(0), qz());
        assert_eq!(m.coeff(1), qz());
        assert_eq!(m.coeff(2), q(3, 1));
    }

    #[test]
    fn x_polynomial() {
        let x = P::x();
        assert_eq!(x.degree(), Some(1));
        assert_eq!(x.coeff(0), qz());
        assert_eq!(x.coeff(1), q(1, 1));
    }

    #[test]
    fn from_coeffs_normalizes() {
        // Trailing zeros should be removed.
        let p = GenPoly::from_coeffs(vec![q(1, 1), q(2, 1), q(0, 1), q(0, 1)]);
        assert_eq!(p.degree(), Some(1));
    }

    // ── Arithmetic ──────────────────────────────────────────────────

    #[test]
    fn add_polynomials() {
        let a = p(&[1, 2, 3]); // 3θ² + 2θ + 1
        let b = p(&[4, 5]);    // 5θ + 4
        let c = a.add(&b);
        assert_eq!(c.coeff(0), q(5, 1));
        assert_eq!(c.coeff(1), q(7, 1));
        assert_eq!(c.coeff(2), q(3, 1));
    }

    #[test]
    fn sub_polynomials() {
        let a = p(&[3, 5]);    // 5θ + 3
        let b = p(&[1, 2]);    // 2θ + 1
        let c = a.sub(&b);     // 3θ + 2
        assert_eq!(c.coeff(0), q(2, 1));
        assert_eq!(c.coeff(1), q(3, 1));
    }

    #[test]
    fn sub_to_zero() {
        let a = p(&[1, 2, 3]);
        let b = a.sub(&a);
        assert!(b.is_zero());
    }

    #[test]
    fn neg_polynomial() {
        let a = p(&[1, -2, 3]);
        let b = a.neg();
        assert_eq!(b.coeff(0), q(-1, 1));
        assert_eq!(b.coeff(1), q(2, 1));
        assert_eq!(b.coeff(2), q(-3, 1));
    }

    #[test]
    fn mul_polynomials() {
        // (θ + 1) * (θ - 1) = θ² - 1
        let a = p(&[1, 1]);
        let b = p(&[-1, 1]);
        let c = a.mul(&b);
        assert_eq!(c.coeff(0), q(-1, 1));
        assert_eq!(c.coeff(1), qz());
        assert_eq!(c.coeff(2), q(1, 1));
    }

    #[test]
    fn mul_by_zero() {
        let a = p(&[1, 2, 3]);
        let z = P::zero();
        assert!(a.mul(&z).is_zero());
        assert!(z.mul(&a).is_zero());
    }

    #[test]
    fn scale_polynomial() {
        let a = p(&[2, 4, 6]);
        let b = a.scale(&q(1, 2));
        assert_eq!(b.coeff(0), q(1, 1));
        assert_eq!(b.coeff(1), q(2, 1));
        assert_eq!(b.coeff(2), q(3, 1));
    }

    // ── Derivative ──────────────────────────────────────────────────

    #[test]
    fn derivative_polynomial() {
        // d/dθ (3θ² + 2θ + 1) = 6θ + 2
        let a = p(&[1, 2, 3]);
        let d = a.derivative();
        assert_eq!(d.coeff(0), q(2, 1));
        assert_eq!(d.coeff(1), q(6, 1));
        assert_eq!(d.degree(), Some(1));
    }

    #[test]
    fn derivative_of_constant() {
        let a = p(&[5]);
        assert!(a.derivative().is_zero());
    }

    #[test]
    fn derivative_of_zero() {
        assert!(P::zero().derivative().is_zero());
    }

    #[test]
    fn derivative_of_linear() {
        // d/dθ (3θ + 7) = 3
        let a = p(&[7, 3]);
        let d = a.derivative();
        assert_eq!(d.degree(), Some(0));
        assert_eq!(d.coeff(0), q(3, 1));
    }

    // ── Evaluation ──────────────────────────────────────────────────

    #[test]
    fn eval_at_point() {
        // 3θ² + 2θ + 1 at θ = 2: 3·4 + 2·2 + 1 = 17
        let a = p(&[1, 2, 3]);
        assert_eq!(a.eval(&q(2, 1)), q(17, 1));
    }

    #[test]
    fn eval_zero_at_point() {
        assert_eq!(P::zero().eval(&q(5, 1)), qz());
    }

    // ── Division ────────────────────────────────────────────────────

    #[test]
    fn div_rem_exact() {
        // (θ² - 1) / (θ + 1) = (θ - 1), remainder 0
        let a = p(&[-1, 0, 1]);
        let b = p(&[1, 1]);
        let (q_poly, r) = a.div_rem(&b);
        assert!(r.is_zero());
        assert_eq!(q_poly.coeff(0), q(-1, 1));
        assert_eq!(q_poly.coeff(1), q(1, 1));
    }

    #[test]
    fn div_rem_with_remainder() {
        // (θ² + 1) / (θ + 1) = (θ - 1), remainder 2
        let a = p(&[1, 0, 1]);
        let b = p(&[1, 1]);
        let (q_poly, r) = a.div_rem(&b);
        assert_eq!(q_poly.coeff(0), q(-1, 1));
        assert_eq!(q_poly.coeff(1), q(1, 1));
        assert_eq!(r.coeff(0), q(2, 1));
        assert_eq!(r.degree(), Some(0));
    }

    #[test]
    fn div_rem_lower_degree() {
        // (θ + 1) / (θ² + 1) = 0, remainder (θ + 1)
        let a = p(&[1, 1]);
        let b = p(&[1, 0, 1]);
        let (q_poly, r) = a.div_rem(&b);
        assert!(q_poly.is_zero());
        assert_eq!(r, a);
    }

    // ── GCD ─────────────────────────────────────────────────────────

    #[test]
    fn gcd_coprime() {
        // gcd(θ + 1, θ + 2) = 1
        let a = p(&[1, 1]);
        let b = p(&[2, 1]);
        let g = P::gcd(&a, &b);
        assert_eq!(g.degree(), Some(0));
    }

    #[test]
    fn gcd_common_factor() {
        // gcd(θ² - 1, θ² - 2θ + 1) = θ - 1
        let a = p(&[-1, 0, 1]);         // θ² - 1 = (θ-1)(θ+1)
        let b = p(&[1, -2, 1]);         // θ² - 2θ + 1 = (θ-1)²
        let g = P::gcd(&a, &b);
        assert_eq!(g.degree(), Some(1));
        // Should be monic: θ - 1
        assert_eq!(g.coeff(1), q(1, 1));
        assert_eq!(g.coeff(0), q(-1, 1));
    }

    #[test]
    fn gcd_with_zero() {
        let a = p(&[1, 2, 1]);
        let z = P::zero();
        let g = P::gcd(&a, &z);
        assert_eq!(g.degree(), a.make_monic().degree());
    }

    // ── Extended GCD ────────────────────────────────────────────────

    #[test]
    fn extended_gcd_bezout() {
        // Verify: s*a + t*b = gcd(a, b)
        let a = p(&[-1, 0, 1]);   // θ² - 1
        let b = p(&[1, -2, 1]);   // (θ - 1)²
        let (s, t, g) = P::extended_gcd(&a, &b);

        // Check s*a + t*b = g
        let lhs = s.mul(&a).add(&t.mul(&b));
        assert_eq!(lhs, g, "Bézout identity failed");
    }

    #[test]
    fn extended_gcd_coprime() {
        let a = p(&[1, 1]);  // θ + 1
        let b = p(&[2, 1]);  // θ + 2
        let (s, t, g) = P::extended_gcd(&a, &b);
        assert_eq!(g.degree(), Some(0)); // gcd = 1
        let lhs = s.mul(&a).add(&t.mul(&b));
        assert_eq!(lhs, g);
    }

    // ── Square-free ─────────────────────────────────────────────────

    #[test]
    fn square_free_part_simple() {
        // (θ - 1)²(θ + 1) → square-free part = (θ - 1)(θ + 1) = θ² - 1
        let a = p(&[1, -2, 1]);  // (θ - 1)²
        let b = p(&[1, 1]);      // (θ + 1)
        let prod = a.mul(&b);    // (θ - 1)²(θ + 1)
        let sfp = prod.square_free_part();
        // Should have degree 2 (the two distinct roots)
        assert_eq!(sfp.degree(), Some(2));
    }

    #[test]
    fn squarefree_factors_basic() {
        // (θ + 1)²(θ - 1)
        let f1 = p(&[1, 1]);     // θ + 1
        let f2 = p(&[-1, 1]);    // θ - 1
        let poly = f1.mul(&f1).mul(&f2); // (θ+1)²(θ-1)
        let factors = poly.squarefree_factors();
        assert!(!factors.is_empty());
        // Should have factors with multiplicities 1 and 2
        let max_mult = factors.iter().map(|(_, m)| *m).max().unwrap();
        assert!(max_mult >= 2, "should have multiplicity 2");
    }

    #[test]
    fn squarefree_already_squarefree() {
        // θ² - 1 = (θ-1)(θ+1), already square-free
        let a = p(&[-1, 0, 1]);
        let factors = a.squarefree_factors();
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].1, 1);
    }

    // ── Resultant ───────────────────────────────────────────────────

    #[test]
    fn resultant_coprime() {
        // res(θ + 1, θ + 2) should be nonzero (no common root)
        let a = p(&[1, 1]);
        let b = p(&[2, 1]);
        let r = P::resultant(&a, &b);
        assert!(!r.is_zero(), "resultant of coprime polys should be nonzero");
    }

    #[test]
    fn resultant_common_root() {
        // res(θ - 1, θ² - 1) should be zero (common root at θ = 1)
        let a = p(&[-1, 1]);
        let b = p(&[-1, 0, 1]);
        let r = P::resultant(&a, &b);
        assert!(r.is_zero(), "resultant should be zero for common root");
    }

    #[test]
    fn resultant_quadratics() {
        // res(θ² + 1, θ² - 1) should be nonzero (roots ±i vs ±1)
        let a = p(&[1, 0, 1]);
        let b = p(&[-1, 0, 1]);
        let r = P::resultant(&a, &b);
        assert!(!r.is_zero());
    }

    // ── Operator overloads ──────────────────────────────────────────

    #[test]
    fn operator_add() {
        let a = p(&[1, 2]);
        let b = p(&[3, 4]);
        let c = &a + &b;
        assert_eq!(c.coeff(0), q(4, 1));
        assert_eq!(c.coeff(1), q(6, 1));
    }

    #[test]
    fn operator_sub() {
        let a = p(&[5, 7]);
        let b = p(&[2, 3]);
        let c = &a - &b;
        assert_eq!(c.coeff(0), q(3, 1));
        assert_eq!(c.coeff(1), q(4, 1));
    }

    #[test]
    fn operator_mul() {
        let a = p(&[1, 1]);
        let b = p(&[1, 1]);
        let c = &a * &b;
        assert_eq!(c, p(&[1, 2, 1])); // (θ+1)² = θ²+2θ+1
    }

    #[test]
    fn operator_neg() {
        let a = p(&[1, -2, 3]);
        let b = -&a;
        assert_eq!(b, p(&[-1, 2, -3]));
    }

    // ── Display ─────────────────────────────────────────────────────

    #[test]
    fn display_zero() {
        assert_eq!(format!("{}", P::zero()), "0");
    }

    #[test]
    fn display_constant() {
        assert_eq!(format!("{}", P::constant(q(7, 1))), "7");
    }

    #[test]
    fn display_linear() {
        let a = p(&[3, 1]);
        let s = format!("{a}");
        assert!(s.contains("θ"), "should contain θ: {s}");
        assert!(s.contains("3"), "should contain 3: {s}");
    }

    #[test]
    fn display_quadratic() {
        let a = p(&[1, 0, 1]);
        let s = format!("{a}");
        assert!(s.contains("θ^2"), "should contain θ^2: {s}");
    }

    // ── Ring trait impl ─────────────────────────────────────────────

    #[test]
    fn ring_impl_add() {
        let a = p(&[1, 2]);
        let b = p(&[3, 4]);
        let c = Ring::add(&a, &b);
        assert_eq!(c, p(&[4, 6]));
    }

    #[test]
    fn ring_impl_zero_one() {
        let z: P = Ring::zero();
        let o: P = Ring::one();
        assert!(Ring::is_zero(&z));
        assert!(!Ring::is_zero(&o));
    }

    // ── Fractional coefficients ─────────────────────────────────────

    #[test]
    fn fractional_coefficients() {
        // (1/2)θ + (1/3)
        let a = GenPoly::from_coeffs(vec![q(1, 3), q(1, 2)]);
        assert_eq!(a.eval(&q(2, 1)), q(4, 3)); // 1/2 * 2 + 1/3 = 1 + 1/3 = 4/3
    }

    #[test]
    fn make_monic_fractional() {
        // (2/3)θ² + (1/3)θ → monic: θ² + (1/2)θ
        let a = GenPoly::from_coeffs(vec![q(0, 1), q(1, 3), q(2, 3)]);
        let m = a.make_monic();
        assert_eq!(m.leading_coeff().unwrap(), &q(1, 1));
        assert_eq!(m.coeff(1), q(1, 2));
    }

    // ── Comprehensive FTC check for div_rem ─────────────────────────

    #[test]
    fn div_rem_reconstructs() {
        // For random-ish polynomials, verify a = q*b + r
        let a = p(&[3, -1, 4, 1, 5]);  // degree 4
        let b = p(&[1, 0, 1]);         // θ² + 1
        let (q_poly, r) = a.div_rem(&b);
        let reconstructed = q_poly.mul(&b).add(&r);
        assert_eq!(reconstructed, a, "a should equal q*b + r");
    }

    // ═══════════════════════════════════════════════════════════════
    // Step 5: GenPoly<RationalFn> — polynomials in θ over ℚ(x)
    // ═══════════════════════════════════════════════════════════════

    mod ratfn_tests {
        use super::super::GenPoly;
        use crate::poly::ratfn::RationalFn;
        use crate::poly::dense::Poly;
        use crate::poly::traits::{Ring, Field, EuclideanDomain};
        use num_bigint::BigInt;
        use num_rational::Ratio;

        type RF = RationalFn;
        type RP = GenPoly<RF>;

        fn r(n: i64, d: i64) -> Ratio<BigInt> {
            Ratio::new(BigInt::from(n), BigInt::from(d))
        }

        /// RationalFn from integer.
        fn rf_int(n: i64) -> RF {
            RationalFn::from_int(n)
        }

        /// RationalFn = polynomial from integer coefficients.
        fn rf_poly(cs: &[i64]) -> RF {
            RationalFn::from_poly(Poly::from_coeffs(
                cs.iter().map(|&c| r(c, 1)).collect(),
            ))
        }

        /// RationalFn = p(x)/q(x) from integer coefficient slices.
        fn rf(numer: &[i64], denom: &[i64]) -> RF {
            let n = Poly::from_coeffs(numer.iter().map(|&c| r(c, 1)).collect());
            let d = Poly::from_coeffs(denom.iter().map(|&c| r(c, 1)).collect());
            RationalFn::new(n, d)
        }

        /// Build a GenPoly<RationalFn> from RF coefficients (ascending degree).
        fn rp(cs: &[RF]) -> RP {
            GenPoly::from_coeffs(cs.to_vec())
        }

        #[test]
        fn construct_and_degree() {
            // (1/x)·θ + 1  — degree 1 in θ, coefficient of θ¹ is 1/x
            let p = rp(&[rf_int(1), rf(&[1], &[0, 1])]);
            assert_eq!(p.degree(), Some(1));
            assert_eq!(p.leading_coeff().unwrap(), &rf(&[1], &[0, 1]));
        }

        #[test]
        fn add_polynomials() {
            // (1/x)·θ + 1  +  (1/x)·θ + 2  =  (2/x)·θ + 3
            let a = rp(&[rf_int(1), rf(&[1], &[0, 1])]);
            let b = rp(&[rf_int(2), rf(&[1], &[0, 1])]);
            let c = a.add(&b);
            assert_eq!(c.degree(), Some(1));
            // Constant term: 1 + 2 = 3
            assert_eq!(c.coeff(0), rf_int(3));
        }

        #[test]
        fn mul_polynomials() {
            // (θ + 1)(θ - 1) = θ² - 1  (with integer coefficients in ℚ(x))
            let a = rp(&[rf_int(1), rf_int(1)]);   // θ + 1
            let b = rp(&[rf_int(-1), rf_int(1)]);  // θ - 1
            let c = a.mul(&b);
            assert_eq!(c.degree(), Some(2));
            assert_eq!(c.coeff(0), rf_int(-1));
            assert!(Ring::is_zero(&c.coeff(1)));
            assert_eq!(c.coeff(2), rf_int(1));
        }

        #[test]
        fn mul_with_ratfn_coefficients() {
            // ((1/x)·θ) · (x·θ) = θ²  (coefficients cancel)
            let a = rp(&[<RF as Ring>::zero(), rf(&[1], &[0, 1])]); // (1/x)·θ
            let b = rp(&[<RF as Ring>::zero(), rf_poly(&[0, 1])]);  // x·θ
            let c = a.mul(&b);
            assert_eq!(c.degree(), Some(2));
            // θ² coefficient should be (1/x)·x = 1
            assert!(Ring::is_one(&c.coeff(2)));
        }

        #[test]
        fn div_rem_exact() {
            // (θ² - 1) / (θ + 1) = (θ - 1), remainder 0
            let a = rp(&[rf_int(-1), rf_int(0), rf_int(1)]);  // θ² - 1
            let b = rp(&[rf_int(1), rf_int(1)]);               // θ + 1
            let (q, rem) = a.div_rem(&b);
            assert!(rem.is_zero(), "remainder should be 0, got {rem}");
            assert_eq!(q.coeff(0), rf_int(-1));
            assert_eq!(q.coeff(1), rf_int(1));
        }

        #[test]
        fn div_rem_with_ratfn_coefficients() {
            // ((1/x)·θ + 1) / ((1/x)·θ) = 1, remainder 1
            let a = rp(&[rf_int(1), rf(&[1], &[0, 1])]);       // (1/x)·θ + 1
            let b = rp(&[<RF as Ring>::zero(), rf(&[1], &[0, 1])]); // (1/x)·θ
            let (q, rem) = a.div_rem(&b);
            assert!(Ring::is_one(&q.coeff(0)), "quotient should be 1, got {q}");
            assert_eq!(rem.coeff(0), rf_int(1));
        }

        #[test]
        fn div_rem_reconstructs() {
            // Verify a = q*b + r for a non-trivial case
            let a = rp(&[rf_int(1), rf_int(2), rf_int(3)]);  // 3θ² + 2θ + 1
            let b = rp(&[rf_int(1), rf_int(1)]);              // θ + 1
            let (q, rem) = a.div_rem(&b);
            let reconstructed = q.mul(&b).add(&rem);
            assert_eq!(reconstructed, a, "a should equal q*b + r");
        }

        #[test]
        fn gcd_coprime() {
            // gcd(θ + 1, θ + 2) = 1 (over ℚ(x) coefficients)
            let a = rp(&[rf_int(1), rf_int(1)]);
            let b = rp(&[rf_int(2), rf_int(1)]);
            let g = RP::gcd(&a, &b);
            assert_eq!(g.degree(), Some(0), "coprime polys should have gcd of degree 0");
        }

        #[test]
        fn gcd_common_factor() {
            // gcd(θ² - 1, (θ - 1)²) = θ - 1
            let a = rp(&[rf_int(-1), rf_int(0), rf_int(1)]);  // θ² - 1
            let b = rp(&[rf_int(1), rf_int(-2), rf_int(1)]);  // θ² - 2θ + 1
            let g = RP::gcd(&a, &b);
            assert_eq!(g.degree(), Some(1), "gcd should be degree 1 (θ-1)");
        }

        #[test]
        fn gcd_with_ratfn_coefficients() {
            // gcd((1/x)·(θ² - 1), (1/x)·(θ - 1)) = θ - 1
            // Because (1/x) is a unit in ℚ(x), the GCD ignores it.
            let inv_x = rf(&[1], &[0, 1]); // 1/x
            let a = rp(&[
                Ring::mul(&inv_x, &rf_int(-1)),
                <RF as Ring>::zero(),
                inv_x.clone(),
            ]); // (1/x)·θ² - (1/x)
            let b = rp(&[
                Ring::mul(&inv_x, &rf_int(-1)),
                inv_x.clone(),
            ]); // (1/x)·θ - (1/x)
            let g = RP::gcd(&a, &b);
            // GCD should be degree 1 (monic: θ - 1)
            assert_eq!(g.degree(), Some(1), "gcd should be degree 1, got {:?}", g.degree());
        }

        #[test]
        fn extended_gcd_bezout() {
            // Verify s*a + t*b = gcd
            let a = rp(&[rf_int(-1), rf_int(0), rf_int(1)]);  // θ² - 1
            let b = rp(&[rf_int(1), rf_int(-2), rf_int(1)]);  // (θ - 1)²
            let (s, t, g) = RP::extended_gcd(&a, &b);
            let lhs = s.mul(&a).add(&t.mul(&b));
            assert_eq!(lhs, g, "Bézout identity failed for GenPoly<RationalFn>");
        }

        #[test]
        fn squarefree_factors_over_ratfn() {
            // (θ + 1)²(θ - 1) should factor as [(θ-1, 1), (θ+1, 2)]
            let f1 = rp(&[rf_int(1), rf_int(1)]);    // θ + 1
            let f2 = rp(&[rf_int(-1), rf_int(1)]);   // θ - 1
            let poly = f1.mul(&f1).mul(&f2);          // (θ+1)²(θ-1)
            let factors = poly.squarefree_factors();
            assert!(!factors.is_empty(), "should have factors");
            let max_mult = factors.iter().map(|(_, m)| *m).max().unwrap();
            assert!(max_mult >= 2, "should have multiplicity ≥ 2");
        }

        #[test]
        fn derivative_with_constant_coefficients() {
            // d/dθ (3θ² + 2θ + 1) = 6θ + 2  (coefficients in ℚ, not ℚ(x))
            let a = rp(&[rf_int(1), rf_int(2), rf_int(3)]);
            let d = a.derivative();
            assert_eq!(d.coeff(0), rf_int(2));
            assert_eq!(d.coeff(1), rf_int(6));
        }

        #[test]
        fn derivative_with_ratfn_coefficients() {
            // d/dθ ((1/x)·θ) = 1/x  (constant w.r.t. θ)
            let inv_x = rf(&[1], &[0, 1]);
            let a = rp(&[<RF as Ring>::zero(), inv_x.clone()]);
            let d = a.derivative();
            assert_eq!(d.degree(), Some(0));
            assert_eq!(d.coeff(0), inv_x);
        }

        #[test]
        fn resultant_common_root() {
            // res(θ - 1, θ² - 1) should be zero (common root at θ = 1)
            let a = rp(&[rf_int(-1), rf_int(1)]);
            let b = rp(&[rf_int(-1), rf_int(0), rf_int(1)]);
            let res = RP::resultant(&a, &b);
            assert!(Ring::is_zero(&res), "resultant should be zero for common root");
        }

        #[test]
        fn resultant_coprime() {
            // res(θ + 1, θ + 2) should be nonzero
            let a = rp(&[rf_int(1), rf_int(1)]);
            let b = rp(&[rf_int(2), rf_int(1)]);
            let res = RP::resultant(&a, &b);
            assert!(!Ring::is_zero(&res), "resultant should be nonzero for coprime polys");
        }

        #[test]
        fn eval_at_ratfn_point() {
            // p(θ) = θ + 1 evaluated at θ = 1/x: should give 1/x + 1 = (x+1)/x
            let p = rp(&[rf_int(1), rf_int(1)]);  // θ + 1
            let point = rf(&[1], &[0, 1]);         // 1/x
            let val = p.eval(&point);
            let expected = rf(&[1, 1], &[0, 1]);   // (x+1)/x
            assert_eq!(val, expected, "θ+1 at θ=1/x should be (x+1)/x");
        }

        #[test]
        fn display_genpoly_ratfn() {
            let p = rp(&[rf_int(1), rf(&[1], &[0, 1])]);  // (1/x)·θ + 1
            let s = format!("{p}");
            assert!(s.contains("θ"), "should display θ: {s}");
        }

        #[test]
        fn ring_impl_zero_one() {
            let z: RP = Ring::zero();
            let o: RP = Ring::one();
            assert!(Ring::is_zero(&z));
            assert!(!Ring::is_zero(&o));
        }
    }
}
