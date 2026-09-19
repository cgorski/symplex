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
use std::hash;

use super::traits::{BindingStrength, CoeffDisplay, Field, IntegralCoeff, Ring};

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
    pub(crate) coeffs: Vec<C>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Conditional Eq and Hash
// ═══════════════════════════════════════════════════════════════════════════

impl<C: Eq> Eq for GenPoly<C> {}

impl<C: hash::Hash> hash::Hash for GenPoly<C> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.coeffs.hash(state);
    }
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
        GenPoly {
            coeffs: vec![C::one()],
        }
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
        let Some(lc) = self.coeffs.last() else {
            return C::zero();
        };
        let mut result = lc.clone();
        for c in self.coeffs.iter().rev().skip(1) {
            result = Ring::add(&Ring::mul(&result, x), c);
        }
        result
    }

    /// `self^n` by repeated squaring.
    pub fn pow(&self, mut n: usize) -> Self {
        let mut result = Self::one();
        let mut base = self.clone();
        while n > 0 {
            if n & 1 == 1 {
                result = result.mul(&base);
            }
            n >>= 1;
            if n > 0 {
                base = base.mul(&base);
            }
        }
        result
    }

    /// Functional composition `self(g)`: substitute `g` for the variable.
    ///
    /// Evaluated with Horner's rule in the polynomial ring, so the cost is
    /// `deg(self)` polynomial multiplications by `g`.
    pub fn compose(&self, g: &Self) -> Self {
        let Some(lc) = self.coeffs.last() else {
            return Self::zero();
        };
        let mut result = Self::constant(lc.clone());
        for c in self.coeffs.iter().rev().skip(1) {
            result = result.mul(g).add(&Self::constant(c.clone()));
        }
        result
    }

    /// Taylor shift: returns `self(θ + a)`.
    pub fn taylor_shift(&self, a: &C) -> Self {
        let shifted = Self::from_coeffs(vec![a.clone(), C::one()]);
        self.compose(&shifted)
    }

    /// Reciprocal polynomial `θ^n · self(1/θ)`: the coefficient list reversed.
    ///
    /// Trailing zero coefficients of the input become leading zeros and are
    /// stripped, so the result may have lower degree when `self(0) = 0`.
    pub fn reverse(&self) -> Self {
        let mut coeffs = self.coeffs.clone();
        coeffs.reverse();
        Self::from_coeffs(coeffs)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// IntegralCoeff convenience constructors
// ═══════════════════════════════════════════════════════════════════════════

impl<C: IntegralCoeff> GenPoly<C> {
    /// A constant polynomial from an integer.
    pub fn from_int(n: i64) -> Self {
        Self::constant(C::from_i64(n))
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

// Owned-value overloads: GenPoly op GenPoly

impl<C: Ring> std::ops::Add for GenPoly<C> {
    type Output = GenPoly<C>;
    fn add(self, rhs: GenPoly<C>) -> GenPoly<C> {
        GenPoly::add(&self, &rhs)
    }
}

impl<C: Ring> std::ops::Sub for GenPoly<C> {
    type Output = GenPoly<C>;
    fn sub(self, rhs: GenPoly<C>) -> GenPoly<C> {
        GenPoly::sub(&self, &rhs)
    }
}

impl<C: Ring> std::ops::Mul for GenPoly<C> {
    type Output = GenPoly<C>;
    fn mul(self, rhs: GenPoly<C>) -> GenPoly<C> {
        GenPoly::mul(&self, &rhs)
    }
}

impl<C: Ring> std::ops::Neg for GenPoly<C> {
    type Output = GenPoly<C>;
    fn neg(self) -> GenPoly<C> {
        GenPoly::neg(&self)
    }
}

// Mixed overloads: &GenPoly op GenPoly

impl<C: Ring> std::ops::Add<GenPoly<C>> for &GenPoly<C> {
    type Output = GenPoly<C>;
    fn add(self, rhs: GenPoly<C>) -> GenPoly<C> {
        GenPoly::add(self, &rhs)
    }
}

impl<C: Ring> std::ops::Sub<GenPoly<C>> for &GenPoly<C> {
    type Output = GenPoly<C>;
    fn sub(self, rhs: GenPoly<C>) -> GenPoly<C> {
        GenPoly::sub(self, &rhs)
    }
}

impl<C: Ring> std::ops::Mul<GenPoly<C>> for &GenPoly<C> {
    type Output = GenPoly<C>;
    fn mul(self, rhs: GenPoly<C>) -> GenPoly<C> {
        GenPoly::mul(self, &rhs)
    }
}

// Mixed overloads: GenPoly op &GenPoly

impl<C: Ring> std::ops::Add<&GenPoly<C>> for GenPoly<C> {
    type Output = GenPoly<C>;
    fn add(self, rhs: &GenPoly<C>) -> GenPoly<C> {
        GenPoly::add(&self, rhs)
    }
}

impl<C: Ring> std::ops::Sub<&GenPoly<C>> for GenPoly<C> {
    type Output = GenPoly<C>;
    fn sub(self, rhs: &GenPoly<C>) -> GenPoly<C> {
        GenPoly::sub(&self, rhs)
    }
}

impl<C: Ring> std::ops::Mul<&GenPoly<C>> for GenPoly<C> {
    type Output = GenPoly<C>;
    fn mul(self, rhs: &GenPoly<C>) -> GenPoly<C> {
        GenPoly::mul(&self, rhs)
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

        let d_deg = divisor.coeffs.len() - 1;
        let d_lc = &divisor.coeffs[d_deg];

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
            let Some(r_lc) = rem.leading_coeff() else {
                break;
            };
            let coeff = Field::div(r_lc, d_lc);
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

        let mut q = GenPoly {
            coeffs: quot_coeffs,
        };
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
        let Some(lc) = self.leading_coeff() else {
            return Self::zero();
        };
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

    /// Compute the Euclidean polynomial remainder sequence (PRS).
    ///
    /// Returns a map from degree to the PRS member at that degree.
    /// The PRS is the sequence of remainders produced during the
    /// Euclidean algorithm: `[a, b, rem(a,b), rem(b,rem(a,b)), ...]`.
    ///
    /// This is used by the Lazard-Rioboo-Trager algorithm to extract
    /// the bivariate logarithmic argument `h(t, x)` from the resultant
    /// computation.
    ///
    /// Assumes `deg(a) >= deg(b)`.  If not, the arguments are swapped
    /// internally.
    pub fn euclidean_prs(a: &Self, b: &Self) -> std::collections::BTreeMap<usize, Self> {
        let mut prs = std::collections::BTreeMap::new();

        if a.is_zero() || b.is_zero() {
            tracing::debug!("euclidean_prs: input is zero, returning empty PRS");
            return prs;
        }

        // Ensure deg(curr) >= deg(next).
        let (mut curr, mut next) = if a.degree().unwrap_or(0) >= b.degree().unwrap_or(0) {
            (a.clone(), b.clone())
        } else {
            (b.clone(), a.clone())
        };

        tracing::debug!(
            deg_a = ?curr.degree(),
            deg_b = ?next.degree(),
            "euclidean_prs: starting PRS computation"
        );

        // Record initial members.
        if let Some(d) = curr.degree() {
            prs.insert(d, curr.clone());
        }
        if let Some(d) = next.degree() {
            prs.insert(d, next.clone());
        }

        // Euclidean remainder loop.
        let mut step = 0u32;
        while !next.is_zero() {
            let (_, r) = curr.div_rem(&next);
            if let Some(d) = r.degree() {
                tracing::trace!(step, remainder_degree = d, "euclidean_prs: remainder");
                prs.insert(d, r.clone());
            }
            curr = next;
            next = r;
            step += 1;
        }

        tracing::debug!(
            num_members = prs.len(),
            degrees = ?prs.keys().collect::<Vec<_>>(),
            "euclidean_prs: PRS complete"
        );

        prs
    }

    /// Extended Euclidean algorithm.
    ///
    /// Returns `(s, t, g)` where `s*a + t*b = g` and `g = gcd(a, b)` (monic).
    pub fn extended_gcd(a: &Self, b: &Self) -> (Self, Self, Self) {
        if b.is_zero() {
            let Some(lc) = a.leading_coeff() else {
                return (Self::one(), Self::zero(), Self::zero());
            };
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

        // Normalize to monic GCD (the zero GCD, from a = b = 0, stays as is).
        if let Some(lc) = r_prev.leading_coeff() {
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

    /// Discriminant `disc(f) = (−1)^{n(n−1)/2} · res(f, f′) / lc(f)`.
    ///
    /// Zero iff `f` has a repeated root.  Constants and the zero polynomial
    /// return `None`; linear polynomials return `1`.
    ///
    /// This formula requires the coefficient ring to have characteristic
    /// zero (it is used for `ℚ`); over `GF(p)` with `p | n` the derivative
    /// degenerates and the result is not the algebraic discriminant.
    pub fn discriminant(&self) -> Option<C> {
        let n = self.degree()?;
        if n == 0 {
            return None;
        }
        if n == 1 {
            return Some(C::one());
        }
        let lc = self.leading_coeff()?.clone();
        let res = Self::resultant(self, &self.derivative());
        let sign = if (n * (n - 1) / 2) % 2 == 0 {
            C::one()
        } else {
            Ring::neg(&C::one())
        };
        Some(Field::div(&Ring::mul(&sign, &res), &lc))
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
        // b is non-zero (checked above), so its leading coefficient exists.
        let Some(lc_b) = b.leading_coeff() else {
            return C::zero();
        };
        let factor = lc_b.pow_usize(m - s);

        Ring::mul(&Ring::mul(&sign, &factor), &Self::resultant(b, &r))
    }

    /// Compute the parametric resultant `R(z) = res_θ(f(θ), g(θ) − z·h(θ))`
    /// as a polynomial in `z` with `C`-coefficients.
    ///
    /// This is used by Rothstein-Trager: given a square-free denominator `D(θ)`
    /// and numerator `A(θ)`, compute `R(z) = res_θ(D, A − z·D')` where the
    /// roots of `R(z)` are the residues of `A/D`.
    ///
    /// Uses evaluation-interpolation: evaluate at `z = 0, 1, 2, …, deg(f)`
    /// (embedded via `IntegralCoeff::from_i64`), compute scalar resultants,
    /// then Lagrange-interpolate to recover `R(z)`.
    pub fn resultant_poly(f: &Self, g: &Self, h: &Self) -> Self
    where
        C: Field + super::traits::IntegralCoeff,
    {
        let d = match f.degree() {
            Some(deg) => deg,
            None => return Self::zero(),
        };

        // We need d+1 evaluation points (R(z) has degree ≤ d in z).
        let num_pts = d + 1;
        let mut points: Vec<(i64, C)> = Vec::with_capacity(num_pts);

        for k in 0..num_pts {
            let z_val = <C as super::traits::IntegralCoeff>::from_i64(k as i64);
            // b_at_z(θ) = g(θ) − z_val · h(θ)
            let z_h = h.scale(&z_val);
            let b_at_z = g.sub(&z_h);
            let res_val = Self::resultant(f, &b_at_z);
            points.push((k as i64, res_val));
        }

        lagrange_interpolate_generic(&points)
    }
}

impl<C: Field + IntegralCoeff> GenPoly<C> {
    /// Functional decomposition `f = g₁ ∘ g₂ ∘ … ∘ gₖ` into indecomposable
    /// components, outermost first.
    ///
    /// Uses the Kozen–Landau approach: for each proper divisor `r` of
    /// `deg f` (smallest first) the unique monic candidate `h` of degree `r`
    /// with `h(0) = 0` is read off from the top coefficients of `f`
    /// (`f ≡ h^{n/r}` in the top `r` coefficients); if `f` is a polynomial
    /// in `h`, the quotient polynomial `g` is decomposed further.
    ///
    /// Indecomposable polynomials (including everything of prime degree)
    /// return `vec![self.clone()]`; constants and zero return an empty
    /// vector.  Requires characteristic zero.
    pub fn decompose(&self) -> Vec<Self> {
        let Some(n) = self.degree() else {
            return vec![];
        };
        if n == 0 {
            return vec![];
        }

        // Iterate: peel the innermost (right) component repeatedly.
        let mut components: Vec<Self> = Vec::new();
        let mut current = self.clone();
        loop {
            match decompose_step(&current) {
                Some((g, h)) => {
                    components.push(h);
                    current = g;
                }
                None => {
                    components.push(current);
                    break;
                }
            }
        }
        components.reverse();
        components
    }
}

/// One step of functional decomposition: find `(g, h)` with `f = g(h)`,
/// `1 < deg h < deg f`, `h` monic with `h(0) = 0`, and `deg h` minimal.
fn decompose_step<C: Field + IntegralCoeff>(f: &GenPoly<C>) -> Option<(GenPoly<C>, GenPoly<C>)> {
    let n = f.degree()?;
    if n < 4 {
        // Degree 2 and 3 are prime (or too small): indecomposable.
        return None;
    }
    let f_monic = f.make_monic();

    for r in 2..n {
        if n % r != 0 {
            continue;
        }
        let s = n / r;
        let s_c = <C as IntegralCoeff>::from_i64(s as i64);

        // Build h = θ^r + h_{r-1} θ^{r-1} + … + h_1 θ coefficient by coefficient.
        let mut h_coeffs = vec![C::zero(); r + 1];
        h_coeffs[r] = C::one();
        for k in 1..r {
            let h_partial = GenPoly::from_coeffs(h_coeffs.clone());
            let hs = h_partial.pow(s);
            // Coefficient of θ^{n-k} in h^s is s·h_{r-k} + (known terms);
            // with h_{r-k} = 0 so far, the known part is hs.coeff(n-k).
            let known = hs.coeff(n - k);
            let target = f_monic.coeff(n - k);
            h_coeffs[r - k] = Field::div(&Ring::sub(&target, &known), &s_c);
        }
        let h = GenPoly::from_coeffs(h_coeffs);
        if h.degree() != Some(r) {
            continue;
        }

        // Is f a polynomial in h?  Repeated division: f = q₀ + h(q₁ + h(…)).
        let mut g_coeffs: Vec<C> = Vec::with_capacity(s + 1);
        let mut rest = f.clone();
        let mut ok = true;
        while !rest.is_zero() {
            let (q, rem) = rest.div_rem(&h);
            if rem.degree().unwrap_or(0) > 0 {
                ok = false;
                break;
            }
            g_coeffs.push(rem.coeff(0));
            rest = q;
        }
        if !ok {
            continue;
        }
        let g = GenPoly::from_coeffs(g_coeffs);
        if g.degree() == Some(s) && g.compose(&h) == *f {
            return Some((g, h));
        }
    }
    None
}

/// Lagrange interpolation for `GenPoly<C>` through rational-valued points
/// at integer abscissae.
///
/// Given `(x₀, y₀), …, (xₙ, yₙ)` with integer `xᵢ` and `C`-valued `yᵢ`,
/// returns the unique polynomial of degree ≤ n passing through all points.
fn lagrange_interpolate_generic<C: Field + super::traits::IntegralCoeff>(
    points: &[(i64, C)],
) -> GenPoly<C> {
    let n = points.len();
    if n == 0 {
        return GenPoly::zero();
    }

    let mut result = GenPoly::zero();

    for i in 0..n {
        let (xi, yi) = &points[i];
        if yi.is_zero() {
            continue;
        }

        // L_i(z) = ∏_{j≠i} (z − xⱼ) / (xᵢ − xⱼ)
        let mut basis = GenPoly::one();
        let mut denom_scalar = C::one();

        for (j, (xj, _)) in points.iter().enumerate() {
            if i == j {
                continue;
            }
            // (z - xj)
            let neg_xj = Ring::neg(&<C as super::traits::IntegralCoeff>::from_i64(*xj));
            let linear = GenPoly::from_coeffs(vec![neg_xj, C::one()]);
            basis = basis.mul(&linear);

            // (xi - xj)
            let diff = <C as super::traits::IntegralCoeff>::from_i64(*xi - *xj);
            denom_scalar = Ring::mul(&denom_scalar, &diff);
        }

        // L_i(z) = basis / denom_scalar
        let inv_denom = Field::inv(&denom_scalar);
        let scaled_basis = basis.scale(&Ring::mul(yi, &inv_denom));
        result = result.add(&scaled_basis);
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Display
// ═══════════════════════════════════════════════════════════════════════════

impl<C: Ring + CoeffDisplay> fmt::Display for GenPoly<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        // Helper to render a coefficient into a String via CoeffDisplay.
        struct FmtCoeff<'a, C>(&'a C, BindingStrength);
        impl<C: CoeffDisplay> fmt::Display for FmtCoeff<'_, C> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt_coeff(f, self.1)
            }
        }

        let mut first = true;
        // Print in descending degree order for readability.
        for i in (0..self.coeffs.len()).rev() {
            let c = &self.coeffs[i];
            if c.is_zero() {
                continue;
            }

            // Detect sign by formatting at Weakest level (no forced parens).
            let weak_str = format!("{}", FmtCoeff(c, BindingStrength::Weakest));
            let is_neg = weak_str.starts_with('-');

            // Get the positive version of the coefficient for display.
            let abs_c;
            let pos = if is_neg {
                abs_c = Ring::neg(c);
                &abs_c
            } else {
                c
            };

            // Separator / sign.
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
            match i {
                0 => {
                    // Constant term: always print the (positive) coefficient.
                    pos.fmt_coeff(f, BindingStrength::Sum)?;
                }
                1 => {
                    if pos.is_one() {
                        write!(f, "θ")?;
                    } else {
                        pos.fmt_coeff(f, BindingStrength::Product)?;
                        write!(f, "*θ")?;
                    }
                }
                n => {
                    if pos.is_one() {
                        write!(f, "θ^{n}")?;
                    } else {
                        pos.fmt_coeff(f, BindingStrength::Product)?;
                        write!(f, "*θ^{n}")?;
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
    fn zero() -> Self {
        GenPoly::zero()
    }
    fn one() -> Self {
        GenPoly::one()
    }
    fn is_zero(&self) -> bool {
        self.is_zero()
    }

    fn add(&self, rhs: &Self) -> Self {
        GenPoly::add(self, rhs)
    }
    fn sub(&self, rhs: &Self) -> Self {
        GenPoly::sub(self, rhs)
    }
    fn mul(&self, rhs: &Self) -> Self {
        GenPoly::mul(self, rhs)
    }
    fn neg(&self) -> Self {
        GenPoly::neg(self)
    }
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

    fn qz() -> Q {
        <Q as Ring>::zero()
    }

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
        let b = p(&[4, 5]); // 5θ + 4
        let c = a.add(&b);
        assert_eq!(c.coeff(0), q(5, 1));
        assert_eq!(c.coeff(1), q(7, 1));
        assert_eq!(c.coeff(2), q(3, 1));
    }

    #[test]
    fn sub_polynomials() {
        let a = p(&[3, 5]); // 5θ + 3
        let b = p(&[1, 2]); // 2θ + 1
        let c = a.sub(&b); // 3θ + 2
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
        let a = p(&[-1, 0, 1]); // θ² - 1 = (θ-1)(θ+1)
        let b = p(&[1, -2, 1]); // θ² - 2θ + 1 = (θ-1)²
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
        let a = p(&[-1, 0, 1]); // θ² - 1
        let b = p(&[1, -2, 1]); // (θ - 1)²
        let (s, t, g) = P::extended_gcd(&a, &b);

        // Check s*a + t*b = g
        let lhs = s.mul(&a).add(&t.mul(&b));
        assert_eq!(lhs, g, "Bézout identity failed");
    }

    #[test]
    fn extended_gcd_coprime() {
        let a = p(&[1, 1]); // θ + 1
        let b = p(&[2, 1]); // θ + 2
        let (s, t, g) = P::extended_gcd(&a, &b);
        assert_eq!(g.degree(), Some(0)); // gcd = 1
        let lhs = s.mul(&a).add(&t.mul(&b));
        assert_eq!(lhs, g);
    }

    // ── Square-free ─────────────────────────────────────────────────

    #[test]
    fn square_free_part_simple() {
        // (θ - 1)²(θ + 1) → square-free part = (θ - 1)(θ + 1) = θ² - 1
        let a = p(&[1, -2, 1]); // (θ - 1)²
        let b = p(&[1, 1]); // (θ + 1)
        let prod = a.mul(&b); // (θ - 1)²(θ + 1)
        let sfp = prod.square_free_part();
        // Should have degree 2 (the two distinct roots)
        assert_eq!(sfp.degree(), Some(2));
    }

    #[test]
    fn squarefree_factors_basic() {
        // (θ + 1)²(θ - 1)
        let f1 = p(&[1, 1]); // θ + 1
        let f2 = p(&[-1, 1]); // θ - 1
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
        let a = p(&[3, -1, 4, 1, 5]); // degree 4
        let b = p(&[1, 0, 1]); // θ² + 1
        let (q_poly, r) = a.div_rem(&b);
        let reconstructed = q_poly.mul(&b).add(&r);
        assert_eq!(reconstructed, a, "a should equal q*b + r");
    }

    // ═══════════════════════════════════════════════════════════════
    // Step 5: GenPoly<RationalFn> — polynomials in θ over ℚ(x)
    // ═══════════════════════════════════════════════════════════════

    mod ratfn_tests {
        use super::super::GenPoly;
        use crate::poly::dense::Poly;
        use crate::poly::ratfn::RationalFn;
        use crate::poly::traits::Ring;
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
            RationalFn::from_poly(Poly::from_coeffs(cs.iter().map(|&c| r(c, 1)).collect()))
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
            let a = rp(&[rf_int(1), rf_int(1)]); // θ + 1
            let b = rp(&[rf_int(-1), rf_int(1)]); // θ - 1
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
            let b = rp(&[<RF as Ring>::zero(), rf_poly(&[0, 1])]); // x·θ
            let c = a.mul(&b);
            assert_eq!(c.degree(), Some(2));
            // θ² coefficient should be (1/x)·x = 1
            assert!(Ring::is_one(&c.coeff(2)));
        }

        #[test]
        fn div_rem_exact() {
            // (θ² - 1) / (θ + 1) = (θ - 1), remainder 0
            let a = rp(&[rf_int(-1), rf_int(0), rf_int(1)]); // θ² - 1
            let b = rp(&[rf_int(1), rf_int(1)]); // θ + 1
            let (q, rem) = a.div_rem(&b);
            assert!(rem.is_zero(), "remainder should be 0, got {rem}");
            assert_eq!(q.coeff(0), rf_int(-1));
            assert_eq!(q.coeff(1), rf_int(1));
        }

        #[test]
        fn div_rem_with_ratfn_coefficients() {
            // ((1/x)·θ + 1) / ((1/x)·θ) = 1, remainder 1
            let a = rp(&[rf_int(1), rf(&[1], &[0, 1])]); // (1/x)·θ + 1
            let b = rp(&[<RF as Ring>::zero(), rf(&[1], &[0, 1])]); // (1/x)·θ
            let (q, rem) = a.div_rem(&b);
            assert!(Ring::is_one(&q.coeff(0)), "quotient should be 1, got {q}");
            assert_eq!(rem.coeff(0), rf_int(1));
        }

        #[test]
        fn div_rem_reconstructs() {
            // Verify a = q*b + r for a non-trivial case
            let a = rp(&[rf_int(1), rf_int(2), rf_int(3)]); // 3θ² + 2θ + 1
            let b = rp(&[rf_int(1), rf_int(1)]); // θ + 1
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
            assert_eq!(
                g.degree(),
                Some(0),
                "coprime polys should have gcd of degree 0"
            );
        }

        #[test]
        fn gcd_common_factor() {
            // gcd(θ² - 1, (θ - 1)²) = θ - 1
            let a = rp(&[rf_int(-1), rf_int(0), rf_int(1)]); // θ² - 1
            let b = rp(&[rf_int(1), rf_int(-2), rf_int(1)]); // θ² - 2θ + 1
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
            let b = rp(&[Ring::mul(&inv_x, &rf_int(-1)), inv_x.clone()]); // (1/x)·θ - (1/x)
            let g = RP::gcd(&a, &b);
            // GCD should be degree 1 (monic: θ - 1)
            assert_eq!(
                g.degree(),
                Some(1),
                "gcd should be degree 1, got {:?}",
                g.degree()
            );
        }

        #[test]
        fn extended_gcd_bezout() {
            // Verify s*a + t*b = gcd
            let a = rp(&[rf_int(-1), rf_int(0), rf_int(1)]); // θ² - 1
            let b = rp(&[rf_int(1), rf_int(-2), rf_int(1)]); // (θ - 1)²
            let (s, t, g) = RP::extended_gcd(&a, &b);
            let lhs = s.mul(&a).add(&t.mul(&b));
            assert_eq!(lhs, g, "Bézout identity failed for GenPoly<RationalFn>");
        }

        #[test]
        fn squarefree_factors_over_ratfn() {
            // (θ + 1)²(θ - 1) should factor as [(θ-1, 1), (θ+1, 2)]
            let f1 = rp(&[rf_int(1), rf_int(1)]); // θ + 1
            let f2 = rp(&[rf_int(-1), rf_int(1)]); // θ - 1
            let poly = f1.mul(&f1).mul(&f2); // (θ+1)²(θ-1)
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
            assert!(
                Ring::is_zero(&res),
                "resultant should be zero for common root"
            );
        }

        #[test]
        fn resultant_coprime() {
            // res(θ + 1, θ + 2) should be nonzero
            let a = rp(&[rf_int(1), rf_int(1)]);
            let b = rp(&[rf_int(2), rf_int(1)]);
            let res = RP::resultant(&a, &b);
            assert!(
                !Ring::is_zero(&res),
                "resultant should be nonzero for coprime polys"
            );
        }

        #[test]
        fn eval_at_ratfn_point() {
            // p(θ) = θ + 1 evaluated at θ = 1/x: should give 1/x + 1 = (x+1)/x
            let p = rp(&[rf_int(1), rf_int(1)]); // θ + 1
            let point = rf(&[1], &[0, 1]); // 1/x
            let val = p.eval(&point);
            let expected = rf(&[1, 1], &[0, 1]); // (x+1)/x
            assert_eq!(val, expected, "θ+1 at θ=1/x should be (x+1)/x");
        }

        #[test]
        fn display_genpoly_ratfn() {
            let p = rp(&[rf_int(1), rf(&[1], &[0, 1])]); // (1/x)·θ + 1
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
