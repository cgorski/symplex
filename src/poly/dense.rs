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
use num_integer::Integer;
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

    /// Extended Euclidean algorithm for polynomials.
    ///
    /// Returns `(s, t, g)` such that `s * a + t * b = g` where
    /// `g = gcd(a, b)`, normalised to be monic.
    pub fn extended_gcd(a: &Poly, b: &Poly) -> (Poly, Poly, Poly) {
        if b.is_zero() {
            if a.is_zero() {
                return (Poly::from_int(1), Poly::zero(), Poly::zero());
            }
            let lc_inv = Ratio::one() / a.leading_coeff().unwrap();
            return (Poly::constant(lc_inv), Poly::zero(), a.make_monic());
        }

        let (mut r_prev, mut r_curr) = (a.clone(), b.clone());
        let (mut s_prev, mut s_curr) = (Poly::from_int(1), Poly::zero());
        let (mut t_prev, mut t_curr) = (Poly::zero(), Poly::from_int(1));

        while !r_curr.is_zero() {
            let (q, r_next) = r_prev.div_rem(&r_curr);
            let s_next = &s_prev - &(&q * &s_curr);
            let t_next = &t_prev - &(&q * &t_curr);

            r_prev = r_curr;
            r_curr = r_next;
            s_prev = s_curr;
            s_curr = s_next;
            t_prev = t_curr;
            t_curr = t_next;
        }

        // Normalise GCD to be monic.
        if !r_prev.is_zero() {
            let lc_inv = Ratio::one() / r_prev.leading_coeff().unwrap();
            r_prev = r_prev.scale(&lc_inv);
            s_prev = s_prev.scale(&lc_inv);
            t_prev = t_prev.scale(&lc_inv);
        }

        (s_prev, t_prev, r_prev)
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

    /// Square-free factorisation via Yun's algorithm.
    ///
    /// Returns `[(f₁, 1), (f₂, 2), …]` where `self = content · ∏ fᵢ^i`
    /// and each `fᵢ` is square-free and pairwise coprime.
    /// Constant / zero polynomials return an empty list.
    pub fn squarefree_factors(&self) -> Vec<(Poly, usize)> {
        if self.is_zero() || self.is_constant() {
            return vec![];
        }
        let mut content = self.content();
        let mut prim = self.primitive_part();
        if prim.leading_coeff().is_some_and(|lc| lc.is_negative()) {
            content = -content;
            prim = -&prim;
        }
        let _ = content; // content is separated out
        square_free_decomposition(&prim)
            .into_iter()
            .map(|(f, m)| (f, m as usize))
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Resultant
// ═══════════════════════════════════════════════════════════════════════════

impl Poly {
    /// Compute the resultant of two polynomials via the Euclidean algorithm.
    ///
    /// The resultant is zero iff the two polynomials share a common root
    /// (including at infinity when one has smaller degree than expected).
    ///
    /// Uses the identity:
    ///   `res(f, g) = (-1)^(mn) · lc(g)^(m - deg(r)) · res(g, r)`
    /// where `r = f mod g`, `m = deg(f)`, `n = deg(g)`.
    pub fn resultant(a: &Poly, b: &Poly) -> Ratio<BigInt> {
        // Base cases.
        if a.is_zero() || b.is_zero() {
            return Ratio::zero();
        }

        let m = match a.degree() {
            Some(d) => d,
            None => return Ratio::zero(),
        };
        let n = match b.degree() {
            Some(d) => d,
            None => return Ratio::zero(),
        };

        // If both are constants: res = 1 (they share no root).
        if m == 0 && n == 0 {
            return Ratio::one();
        }

        // res(constant, g) = constant^deg(g).
        if m == 0 {
            return num_traits::pow(a.coeff(0), n);
        }
        if n == 0 {
            return num_traits::pow(b.coeff(0), m);
        }

        // Ensure deg(a) >= deg(b); swap with sign correction.
        if m < n {
            let sign = if (m * n) % 2 == 0 {
                Ratio::one()
            } else {
                -Ratio::<BigInt>::one()
            };
            return sign * Poly::resultant(b, a);
        }

        // Recursive step: r = a mod b.
        let r = a.rem(b);

        if r.is_zero() {
            // gcd has positive degree → resultant is 0.
            return Ratio::zero();
        }

        let s = r.degree().unwrap_or(0);
        let sign = if (m * n) % 2 == 0 {
            Ratio::one()
        } else {
            -Ratio::<BigInt>::one()
        };
        let lc_b = b.leading_coeff().unwrap().clone();
        let factor = num_traits::pow(lc_b, m - s);

        sign * factor * Poly::resultant(b, &r)
    }

    /// Compute `R(t) = res_x(f(x), g(x) − t · h(x))` as a polynomial in `t`
    /// using evaluation–interpolation.
    ///
    /// The polynomial `g(x) − t · h(x)` is linear in the parameter `t`.
    /// We evaluate at `t = 0, 1, 2, …, d` (where `d = deg_x(f)`) to obtain
    /// `d+1` scalar resultants, then Lagrange-interpolate to recover `R(t)`.
    ///
    /// This avoids building bivariate polynomial infrastructure entirely.
    pub fn resultant_poly(f: &Poly, g: &Poly, h: &Poly) -> Poly {
        let d = match f.degree() {
            Some(deg) => deg,
            None => return Poly::zero(),
        };

        // We need d+1 evaluation points (R(t) has degree ≤ d in t).
        let num_pts = d + 1;
        let mut points: Vec<(i64, Ratio<BigInt>)> = Vec::with_capacity(num_pts);

        for k in 0..num_pts {
            let t_val = Ratio::from_integer(BigInt::from(k as i64));
            // b_at_t(x) = g(x) − t_val · h(x)
            let b_at_t = g - &h.scale(&t_val);
            let res_val = Poly::resultant(f, &b_at_t);
            points.push((k as i64, res_val));
        }

        lagrange_interpolate_rational(&points)
    }
}

/// Lagrange interpolation through rational-valued points at integer abscissae.
///
/// Given `(x₀, y₀), …, (xₙ, yₙ)` with integer `xᵢ` and rational `yᵢ`,
/// returns the unique polynomial of degree ≤ n passing through all points.
pub(crate) fn lagrange_interpolate_rational(
    points: &[(i64, Ratio<BigInt>)],
) -> Poly {
    let n = points.len();
    if n == 0 {
        return Poly::zero();
    }

    let mut result = Poly::zero();

    for i in 0..n {
        let (xi, yi) = &points[i];
        if yi.is_zero() {
            continue;
        }

        // L_i(x) = ∏_{j≠i} (x − xⱼ) / (xᵢ − xⱼ)
        let mut basis = Poly::from_int(1);
        let mut denom = BigInt::one();

        for (j, (xj, _)) in points.iter().enumerate() {
            if i == j {
                continue;
            }
            let linear = Poly::from_coeffs(vec![
                Ratio::from_integer(BigInt::from(-*xj)),
                Ratio::one(),
            ]);
            basis = &basis * &linear;
            denom *= BigInt::from(*xi - *xj);
        }

        if denom.is_zero() {
            // Duplicate x-values — shouldn't happen with our construction.
            continue;
        }

        let scale = Ratio::new(yi.numer().clone() * denom.signum(), yi.denom().clone() * denom.abs());
        result = &result + &basis.scale(&scale);
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Factoring over ℤ
// ═══════════════════════════════════════════════════════════════════════════

impl Poly {
    /// Returns `true` if every coefficient is an integer (denominator 1).
    pub fn has_integer_coeffs(&self) -> bool {
        self.coeffs.iter().all(|c| c.denom().is_one())
    }

    /// Factor this polynomial over ℤ.
    ///
    /// Returns `(content, factors)` where:
    /// - `content` is the rational GCD of all coefficients, with sign chosen
    ///   so that each factor has a positive leading coefficient.
    /// - `factors` is a list of `(irreducible_factor, multiplicity)` pairs.
    ///
    /// The original polynomial equals `content * ∏ factor^multiplicity`.
    pub fn factor_over_z(&self) -> (Ratio<BigInt>, Vec<(Poly, u32)>) {
        if self.is_zero() {
            return (Ratio::zero(), vec![]);
        }
        if self.is_constant() {
            return (self.coeff(0), vec![]);
        }

        // 1. Extract content and make primitive.
        let mut content = self.content();
        let mut prim = self.primitive_part();

        // Ensure positive leading coefficient.
        if prim.leading_coeff().is_some_and(|lc| lc.is_negative()) {
            content = -content;
            prim = -&prim;
        }

        let deg = prim.degree().unwrap_or(0);
        if deg <= 1 {
            return (content, vec![(prim, 1)]);
        }

        // 2. Square-free decomposition.
        let sfd = square_free_decomposition(&prim);

        // 3. Factor each square-free component into irreducibles.
        let mut all_factors: Vec<(Poly, u32)> = Vec::new();
        for (sf, mult) in sfd {
            let irreducibles = factor_squarefree(&sf);
            for irr in irreducibles {
                all_factors.push((irr, mult));
            }
        }

        if all_factors.is_empty() {
            all_factors.push((prim, 1));
        }

        (content, all_factors)
    }
}

// ── Square-free decomposition (Yun's algorithm) ────────────────────────

/// Compute the square-free decomposition of a primitive polynomial with
/// positive leading coefficient.
///
/// Returns `[(a₁, 1), (a₂, 2), …]` where `f = ∏ aᵢ^i` (up to a unit)
/// and each `aᵢ` is square-free and pairwise coprime.
fn square_free_decomposition(f: &Poly) -> Vec<(Poly, u32)> {
    let deg = match f.degree() {
        Some(d) if d >= 1 => d,
        _ => return vec![],
    };

    let df = f.derivative();
    if df.is_zero() {
        // Shouldn't happen for degree ≥ 1 in characteristic 0.
        return vec![(ensure_positive_lc(f), 1)];
    }

    let g = Poly::gcd(f, &df);

    // If gcd is trivial (constant), f is already square-free.
    if g.degree().unwrap_or(0) == 0 {
        return vec![(ensure_positive_lc(f), 1)];
    }

    // Yun's iterative decomposition.
    let mut w = f.div(&g); // product of all distinct irreducible factors
    let mut c = g; //          ∏ pᵢ^(eᵢ−1)
    let mut result: Vec<(Poly, u32)> = Vec::new();
    let mut i = 1u32;

    loop {
        if w.degree().unwrap_or(0) == 0 {
            break;
        }

        let y = Poly::gcd(&w, &c); // factors with multiplicity > i
        let z = w.div(&y); //         factors with multiplicity exactly i

        if z.degree().unwrap_or(0) > 0 {
            // Normalize to primitive with positive leading coefficient.
            let z_norm = ensure_positive_lc(&z.primitive_part());
            result.push((z_norm, i));
        }

        w = y;
        if c.degree().unwrap_or(0) > 0 && w.degree().unwrap_or(0) > 0 {
            c = c.div(&w);
        } else {
            c = Poly::from_int(1);
        }
        i += 1;

        // Safety bound.
        if i > deg as u32 + 1 {
            break;
        }
    }

    if result.is_empty() {
        result.push((ensure_positive_lc(f), 1));
    }

    result
}

/// Return a copy of `p` with positive leading coefficient.
fn ensure_positive_lc(p: &Poly) -> Poly {
    match p.leading_coeff() {
        Some(lc) if lc.is_negative() => -p,
        _ => p.clone(),
    }
}

// ── Irreducible factoring of a square-free polynomial ──────────────────

/// Factor a square-free, primitive polynomial into irreducible factors
/// over ℤ using the Rational Root Theorem followed by Kronecker's method.
fn factor_squarefree(f: &Poly) -> Vec<Poly> {
    let deg = match f.degree() {
        Some(d) if d >= 1 => d,
        _ => return vec![f.clone()],
    };

    if deg == 1 {
        return vec![ensure_positive_lc(f)];
    }

    // Step 1: extract all linear factors via the Rational Root Theorem.
    let (mut remaining, mut factors) = extract_rational_roots(f);

    if remaining.degree().unwrap_or(0) == 0 {
        return factors;
    }
    if remaining.degree() == Some(1) {
        factors.push(ensure_positive_lc(&remaining));
        return factors;
    }

    // Step 2: Kronecker's method for degree-2 … degree-⌊n/2⌋.
    let max_trial = (remaining.degree().unwrap_or(0) / 2).min(6);
    for trial_deg in 2..=max_trial {
        loop {
            let rem_deg = remaining.degree().unwrap_or(0);
            if rem_deg < 2 * trial_deg {
                break;
            }
            match kronecker_find_factor(&remaining, trial_deg) {
                Some((fac, quot)) => {
                    // The factor might itself be reducible — recurse.
                    factors.extend(factor_squarefree(&fac));
                    remaining = quot;
                }
                None => break,
            }
        }
        if remaining.degree().unwrap_or(0) < 2 {
            break;
        }
    }

    // Whatever remains is irreducible (or we couldn't split it further).
    if remaining.degree().unwrap_or(0) >= 1 {
        factors.push(ensure_positive_lc(&remaining));
    }

    factors
}

// ── Rational Root Theorem ──────────────────────────────────────────────

/// Extract all linear factors using the Rational Root Theorem.
///
/// Returns `(remaining, linear_factors)`.
fn extract_rational_roots(f: &Poly) -> (Poly, Vec<Poly>) {
    let mut remaining = f.clone();
    let mut factors: Vec<Poly> = Vec::new();

    loop {
        let deg = match remaining.degree() {
            Some(d) if d >= 1 => d,
            _ => break,
        };

        let a0 = remaining.coeff(0);

        // Handle root at x = 0.
        if a0.is_zero() {
            remaining = remaining.div(&Poly::x());
            factors.push(Poly::x());
            continue;
        }

        // Need integer coefficients.
        if !remaining.has_integer_coeffs() {
            break;
        }

        let an = remaining.coeff(deg);
        let a0_abs = a0.numer().abs();
        let an_abs = an.numer().abs();

        let p_divs = positive_divisors(&a0_abs);
        let q_divs = positive_divisors(&an_abs);

        // Safety cap.
        if p_divs.is_empty() || q_divs.is_empty() {
            break;
        }
        if p_divs.len() * q_divs.len() > 500 {
            break;
        }

        let mut found = false;

        'search: for p in &p_divs {
            for q in &q_divs {
                // Only consider coprime (p, q) to avoid redundant/non-primitive factors.
                if p.gcd(q) != BigInt::one() {
                    continue;
                }
                for &sign in &[1i64, -1i64] {
                    let candidate = Ratio::new(
                        p * BigInt::from(sign),
                        q.clone(),
                    );
                    if remaining.eval(&candidate).is_zero() {
                        // Build integer linear factor (q·x − sign·p).
                        let int_factor = Poly::from_coeffs(vec![
                            Ratio::from_integer(-(p * BigInt::from(sign))),
                            Ratio::from_integer(q.clone()),
                        ]);
                        let prim_factor = int_factor.primitive_part();
                        let prim_factor = ensure_positive_lc(&prim_factor);
                        let (quot, rem) = remaining.div_rem(&prim_factor);
                        if rem.is_zero() {
                            remaining = if quot.has_integer_coeffs() {
                                quot
                            } else {
                                // Normalize to integer coefficients.
                                ensure_positive_lc(&quot.primitive_part())
                            };
                            factors.push(prim_factor);
                            found = true;
                            break 'search;
                        }
                    }
                }
            }
        }

        if !found {
            break;
        }
    }

    (remaining, factors)
}

// ── Kronecker's method ─────────────────────────────────────────────────

/// Try to find a non-trivial factor of `f` with degree exactly
/// `trial_deg` using Kronecker's method.
///
/// Returns `Some((factor, quotient))` on success.
fn kronecker_find_factor(f: &Poly, trial_deg: usize) -> Option<(Poly, Poly)> {
    let f_deg = f.degree()?;
    if trial_deg == 0 || trial_deg * 2 > f_deg {
        return None;
    }

    let num_points = trial_deg + 1;

    // Choose small integer evaluation points: 0, 1, −1, 2, −2, …
    let eval_pts = small_integer_points(num_points);

    // Evaluate f at each point (all results are integers).
    let vals: Vec<BigInt> = eval_pts
        .iter()
        .map(|&x| {
            let v = f.eval(&Ratio::from_integer(BigInt::from(x)));
            // Integer-coeff poly at integer arg ⇒ integer.
            v.numer().clone()
        })
        .collect();

    // If any evaluation is zero, rational root extraction should have
    // caught it already.  Skip to avoid infinite divisor enumeration.
    if vals.iter().any(|v| v.is_zero()) {
        return None;
    }

    // Signed divisors for each evaluation value.
    let div_lists: Vec<Vec<BigInt>> = vals.iter().map(signed_divisors).collect();

    // Bail out if any divisor list is empty (value too large) or
    // total combinations exceed a practical limit.
    if div_lists.iter().any(Vec::is_empty) {
        return None;
    }
    let total: usize = div_lists
        .iter()
        .map(|d| d.len())
        .try_fold(1usize, |acc, n| acc.checked_mul(n))
        .unwrap_or(usize::MAX);
    if total > 100_000 {
        return None;
    }

    let mut indices = vec![0usize; num_points];
    let mut count = 0usize;

    loop {
        // Build the current divisor combination.
        let points: Vec<(i64, BigInt)> = (0..num_points)
            .map(|i| (eval_pts[i], div_lists[i][indices[i]].clone()))
            .collect();

        if let Some(candidate) = lagrange_interpolate(&points)
            && candidate.degree() == Some(trial_deg) && candidate.has_integer_coeffs()
        {
            let prim = ensure_positive_lc(&candidate.primitive_part());
            if prim.degree() == Some(trial_deg) {
                let (quot, rem) = f.div_rem(&prim);
                if rem.is_zero() && quot.has_integer_coeffs() {
                    return Some((prim, quot));
                }
            }
        }

        // Advance odometer.
        count += 1;
        if count >= total {
            break;
        }
        let mut carry = true;
        for k in (0..num_points).rev() {
            if carry {
                indices[k] += 1;
                if indices[k] >= div_lists[k].len() {
                    indices[k] = 0;
                } else {
                    carry = false;
                    break;
                }
            }
        }
        if carry {
            break;
        }
    }

    None
}

// ── Helper: Lagrange interpolation ─────────────────────────────────────

/// Lagrange interpolation through `(xᵢ, yᵢ)` integer points.
fn lagrange_interpolate(points: &[(i64, BigInt)]) -> Option<Poly> {
    let n = points.len();
    let mut result = Poly::zero();

    for i in 0..n {
        let (xi, yi) = &points[i];
        if yi.is_zero() {
            continue;
        }

        // L_i(x) = ∏_{j≠i} (x − xⱼ) / (xᵢ − xⱼ)
        let mut basis = Poly::from_int(1);
        let mut denom = BigInt::one();

        for (j, point_j) in points.iter().enumerate() {
            if i == j {
                continue;
            }
            let xj = point_j.0;
            let linear = Poly::from_coeffs(vec![
                Ratio::from_integer(BigInt::from(-xj)),
                Ratio::one(),
            ]);
            basis = &basis * &linear;
            denom *= BigInt::from(*xi - xj);
        }

        if denom.is_zero() {
            return None;
        }

        let scale = Ratio::new(yi.clone(), denom);
        result = &result + &basis.scale(&scale);
    }

    Some(result)
}

// ── Helper: integer point generation ───────────────────────────────────

/// Generate `n` distinct small integer points: 0, 1, −1, 2, −2, …
fn small_integer_points(n: usize) -> Vec<i64> {
    let mut pts = Vec::with_capacity(n);
    pts.push(0);
    let mut k = 1i64;
    while pts.len() < n {
        pts.push(k);
        if pts.len() < n {
            pts.push(-k);
        }
        k += 1;
    }
    pts
}

// ── Helper: integer divisors ───────────────────────────────────────────

/// All positive divisors of `|n|` in ascending order.
///
/// Returns an empty list if `n` is zero or `|n|` exceeds an internal
/// threshold (to keep Kronecker practical).
fn positive_divisors(n: &BigInt) -> Vec<BigInt> {
    if n.is_zero() {
        return vec![];
    }
    let n_abs = n.abs();
    // Bail out for very large values.
    if n_abs > BigInt::from(1_000_000_000i64) {
        return vec![];
    }

    let mut small = Vec::new();
    let mut large = Vec::new();
    let mut i = BigInt::one();
    while &i * &i <= n_abs {
        if (&n_abs % &i).is_zero() {
            let complement = &n_abs / &i;
            if complement != i {
                large.push(complement);
            }
            small.push(i.clone());
        }
        i += 1;
    }

    large.reverse();
    small.extend(large);
    small
}

/// All signed (positive and negative) divisors of `|n|`.
fn signed_divisors(n: &BigInt) -> Vec<BigInt> {
    let pos = positive_divisors(n);
    let mut result = Vec::with_capacity(pos.len() * 2);
    for d in pos {
        result.push(d.clone());
        result.push(-d);
    }
    result
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

    // ── Resultant ───────────────────────────────────────────────────

    #[test]
    fn resultant_coprime() {
        // res(x+1, x+2) = lc(x+1)^1 · (x+2)(root of x+1)
        //               = 1 · ((-1)+2) = 1
        let a = Poly::from_coeffs(vec![ri(1), ri(1)]); // x + 1
        let b = Poly::from_coeffs(vec![ri(2), ri(1)]); // x + 2
        let res = Poly::resultant(&a, &b);
        assert_eq!(res, ri(1));
    }

    #[test]
    fn resultant_common_root() {
        // x^2-1 and x-1 share root x=1 → resultant = 0
        let a = Poly::from_coeffs(vec![ri(-1), ri(0), ri(1)]); // x^2 - 1
        let b = Poly::from_coeffs(vec![ri(-1), ri(1)]);         // x - 1
        assert_eq!(Poly::resultant(&a, &b), ri(0));
    }

    #[test]
    fn resultant_x2_plus_1_x_minus_1() {
        // res(x^2+1, x-1) = (x^2+1) at x=1 = 2
        let a = Poly::from_coeffs(vec![ri(1), ri(0), ri(1)]); // x^2 + 1
        let b = Poly::from_coeffs(vec![ri(-1), ri(1)]);        // x - 1
        let res = Poly::resultant(&a, &b);
        assert_eq!(res, ri(2));
    }

    #[test]
    fn resultant_two_quadratics() {
        // res(x^2+1, x^2-1): roots of x^2-1 are ±1.
        // res = (1+1)((-1)^2+1) = 2*2 = 4  (lc(a)^deg(b) * ∏ a(root_of_b))
        let a = Poly::from_coeffs(vec![ri(1), ri(0), ri(1)]); // x^2+1
        let b = Poly::from_coeffs(vec![ri(-1), ri(0), ri(1)]); // x^2-1
        assert_eq!(Poly::resultant(&a, &b), ri(4));
    }

    #[test]
    fn resultant_with_zero() {
        let a = Poly::from_coeffs(vec![ri(1), ri(1)]); // x+1
        assert_eq!(Poly::resultant(&a, &Poly::zero()), ri(0));
        assert_eq!(Poly::resultant(&Poly::zero(), &a), ri(0));
    }

    #[test]
    fn resultant_2x_plus_3_x_plus_1() {
        // Sylvester: det [[2,3],[1,1]] = 2-3 = -1
        let a = Poly::from_coeffs(vec![ri(3), ri(2)]); // 2x+3
        let b = Poly::from_coeffs(vec![ri(1), ri(1)]); // x+1
        assert_eq!(Poly::resultant(&a, &b), ri(-1));
    }

    #[test]
    fn resultant_constant() {
        let a = Poly::from_int(5);
        let b = Poly::from_coeffs(vec![ri(1), ri(0), ri(1)]); // x^2+1
        // res(5, x^2+1) = 5^2 = 25
        assert_eq!(Poly::resultant(&a, &b), ri(25));
    }

    // ── Resultant polynomial (eval-interpolation) ───────────────────

    #[test]
    fn resultant_poly_linear_param() {
        // f = x^2 + 1, g = x, h = 1  →  res_x(x^2+1, x - t)
        // = (x^2+1) evaluated at x=t = t^2 + 1
        let f = Poly::from_coeffs(vec![ri(1), ri(0), ri(1)]); // x^2+1
        let g = Poly::x();                                      // x
        let h = Poly::from_int(1);                               // 1
        let r = Poly::resultant_poly(&f, &g, &h);
        let expected = Poly::from_coeffs(vec![ri(1), ri(0), ri(1)]); // t^2+1
        assert_eq!(r, expected, "res_x(x^2+1, x-t) should be t^2+1, got {r}");
    }

    #[test]
    fn resultant_poly_x5_plus_1() {
        // f = x^5+1, g = 1, h = 5x^4  →  R(t) = res_x(x^5+1, 1-5t·x^4)
        // By calculation: R(t) = 1 - 3125·t^5
        let mut f_coeffs = vec![ri(0); 6];
        f_coeffs[0] = ri(1);
        f_coeffs[5] = ri(1);
        let f = Poly::from_coeffs(f_coeffs); // x^5+1

        let g = Poly::from_int(1);

        let mut h_coeffs = vec![ri(0); 5];
        h_coeffs[4] = ri(5);
        let h = Poly::from_coeffs(h_coeffs); // 5x^4

        let r = Poly::resultant_poly(&f, &g, &h);
        // R(t) should be 1 - 3125 t^5
        assert_eq!(r.degree(), Some(5), "R(t) should have degree 5, got {:?}", r.degree());
        assert_eq!(r.coeff(0), ri(1), "constant term should be 1");
        assert_eq!(r.coeff(5), ri(-3125), "t^5 coeff should be -3125, got {}", r.coeff(5));
        // Middle terms should be 0
        for k in 1..5 {
            assert_eq!(r.coeff(k), ri(0), "coeff of t^{k} should be 0");
        }
    }

    // ── Squarefree factors ──────────────────────────────────────────

    #[test]
    fn squarefree_factors_square() {
        // (x+1)^2 = x^2+2x+1
        let p = Poly::from_coeffs(vec![ri(1), ri(2), ri(1)]);
        let factors = p.squarefree_factors();
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].1, 2);
        assert_eq!(factors[0].0.degree(), Some(1));
    }

    #[test]
    fn squarefree_factors_distinct() {
        // (x-1)(x-2) = x^2-3x+2 — already squarefree
        let p = Poly::from_coeffs(vec![ri(2), ri(-3), ri(1)]);
        let factors = p.squarefree_factors();
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].1, 1);
    }

    #[test]
    fn squarefree_factors_mixed() {
        // (x-1)^2*(x-2) = x^3-4x^2+5x-2
        let x_minus = |n: i64| Poly::from_coeffs(vec![ri(-n), ri(1)]);
        let p = &(&x_minus(1) * &x_minus(1)) * &x_minus(2);
        let factors = p.squarefree_factors();
        // Should have two factors: (x-1) with mult 2, (x-2) with mult 1
        // or equivalently, something that reconstructs correctly
        let mut product = Poly::from_int(1);
        for (f, m) in &factors {
            for _ in 0..*m {
                product = &product * f;
            }
        }
        let product_monic = product.make_monic();
        let p_monic = p.make_monic();
        assert_eq!(product_monic, p_monic,
            "product of squarefree factors should reconstruct original");
    }

    // ── Lagrange interpolation (rational) ───────────────────────────

    #[test]
    fn lagrange_rational_constant() {
        // f(x) = 5 → interpolate from (0,5), (1,5)
        let pts = vec![(0i64, ri(5)), (1, ri(5))];
        let p = lagrange_interpolate_rational(&pts);
        assert_eq!(p, Poly::from_int(5));
    }

    #[test]
    fn lagrange_rational_linear() {
        // f(x) = 2x + 1 → (0,1), (1,3)
        let pts = vec![(0i64, ri(1)), (1, ri(3))];
        let p = lagrange_interpolate_rational(&pts);
        assert_eq!(p, Poly::from_coeffs(vec![ri(1), ri(2)]));
    }

    #[test]
    fn lagrange_rational_quadratic() {
        // f(x) = x^2 → (0,0), (1,1), (2,4)
        let pts = vec![(0i64, ri(0)), (1, ri(1)), (2, ri(4))];
        let p = lagrange_interpolate_rational(&pts);
        assert_eq!(p, Poly::from_coeffs(vec![ri(0), ri(0), ri(1)]));
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

    // ── Factoring over ℤ ────────────────────────────────────────────────

    #[test]
    fn has_integer_coeffs_true() {
        let p = Poly::from_coeffs(vec![ri(1), ri(2), ri(3)]);
        assert!(p.has_integer_coeffs());
    }

    #[test]
    fn has_integer_coeffs_false() {
        let p = Poly::from_coeffs(vec![r(1, 2), ri(1)]);
        assert!(!p.has_integer_coeffs());
    }

    #[test]
    fn factor_z_x2_minus_1() {
        // x^2 - 1 = (x - 1)(x + 1)
        let p = Poly::from_coeffs(vec![ri(-1), ri(0), ri(1)]);
        let (content, factors) = p.factor_over_z();
        assert_eq!(content, ri(1));
        assert_eq!(factors.len(), 2, "should have 2 factors: {factors:?}");
        // Verify product.
        let mut product = Poly::from_int(1);
        for (f, m) in &factors {
            for _ in 0..*m {
                product = &product * f;
            }
        }
        product = product.scale(&content);
        assert_eq!(product, p, "product of factors should equal original");
    }

    #[test]
    fn factor_z_x2_plus_1_irreducible() {
        // x^2 + 1 is irreducible over ℤ
        let p = Poly::from_coeffs(vec![ri(1), ri(0), ri(1)]);
        let (content, factors) = p.factor_over_z();
        assert_eq!(content, ri(1));
        assert_eq!(factors.len(), 1, "x^2+1 is irreducible: {factors:?}");
        assert_eq!(factors[0].1, 1);
    }

    #[test]
    fn factor_z_x4_minus_1() {
        // x^4 - 1 = (x - 1)(x + 1)(x^2 + 1)
        let p = Poly::from_coeffs(vec![ri(-1), ri(0), ri(0), ri(0), ri(1)]);
        let (content, factors) = p.factor_over_z();
        assert_eq!(content, ri(1));
        assert!(
            factors.len() >= 3,
            "x^4-1 should have >= 3 factors: {factors:?}"
        );
        // Verify product.
        let mut product = Poly::from_int(1);
        for (f, m) in &factors {
            for _ in 0..*m {
                product = &product * f;
            }
        }
        product = product.scale(&content);
        assert_eq!(product, p, "product mismatch");
    }

    #[test]
    fn factor_z_x4_plus_5x2_plus_6() {
        // x^4 + 5x^2 + 6 = (x^2 + 2)(x^2 + 3)
        let p = Poly::from_coeffs(vec![ri(6), ri(0), ri(5), ri(0), ri(1)]);
        let (content, factors) = p.factor_over_z();
        assert_eq!(content, ri(1));
        assert_eq!(
            factors.len(),
            2,
            "should factor into 2 quadratics: {factors:?}"
        );
        // Verify product.
        let mut product = Poly::from_int(1);
        for (f, m) in &factors {
            for _ in 0..*m {
                product = &product * f;
            }
        }
        product = product.scale(&content);
        assert_eq!(product, p, "product mismatch");
    }

    #[test]
    fn factor_z_perfect_square() {
        // x^2 + 2x + 1 = (x + 1)^2
        let p = Poly::from_coeffs(vec![ri(1), ri(2), ri(1)]);
        let (content, factors) = p.factor_over_z();
        assert_eq!(content, ri(1));
        assert_eq!(factors.len(), 1, "perfect square has 1 unique factor");
        assert_eq!(factors[0].1, 2, "multiplicity should be 2");
    }

    #[test]
    fn factor_z_6x2_plus_12x_plus_6() {
        // 6x^2 + 12x + 6 = 6·(x + 1)^2
        let p = Poly::from_coeffs(vec![ri(6), ri(12), ri(6)]);
        let (content, factors) = p.factor_over_z();
        assert_eq!(content, ri(6));
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].1, 2);
        assert_eq!(factors[0].0.degree(), Some(1));
    }

    #[test]
    fn factor_z_cubic_three_roots() {
        // x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
        let p = Poly::from_coeffs(vec![ri(-6), ri(11), ri(-6), ri(1)]);
        let (content, factors) = p.factor_over_z();
        assert_eq!(content, ri(1));
        assert_eq!(factors.len(), 3, "should have 3 linear factors: {factors:?}");
        // Verify product.
        let mut product = Poly::from_int(1);
        for (f, m) in &factors {
            for _ in 0..*m {
                product = &product * f;
            }
        }
        product = product.scale(&content);
        assert_eq!(product, p, "product mismatch");
    }

    #[test]
    fn factor_z_x6_minus_1() {
        // x^6 - 1 = (x-1)(x+1)(x^2-x+1)(x^2+x+1)
        let mut coeffs = vec![ri(0); 7];
        coeffs[0] = ri(-1);
        coeffs[6] = ri(1);
        let p = Poly::from_coeffs(coeffs);
        let (content, factors) = p.factor_over_z();
        assert_eq!(content, ri(1));
        assert!(
            factors.len() >= 4,
            "x^6-1 should have >= 4 factors: {factors:?}"
        );
        // Verify product.
        let mut product = Poly::from_int(1);
        for (f, m) in &factors {
            for _ in 0..*m {
                product = &product * f;
            }
        }
        product = product.scale(&content);
        assert_eq!(product, p, "product mismatch");
    }

    #[test]
    fn factor_z_constant() {
        let p = Poly::from_int(42);
        let (content, factors) = p.factor_over_z();
        assert_eq!(content, ri(42));
        assert!(factors.is_empty());
    }

    #[test]
    fn factor_z_zero() {
        let p = Poly::zero();
        let (content, factors) = p.factor_over_z();
        assert!(content.is_zero());
        assert!(factors.is_empty());
    }

    #[test]
    fn factor_z_linear() {
        let p = Poly::from_coeffs(vec![ri(4), ri(2)]); // 2x + 4
        let (content, factors) = p.factor_over_z();
        assert_eq!(content, ri(2));
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].1, 1);
        // Factor should be (x + 2).
        let f = &factors[0].0;
        assert_eq!(f.eval(&ri(-2)), ri(0));
    }

    #[test]
    fn factor_z_preserves_value_at_points() {
        // x^4 + 5x^2 + 6
        let p = Poly::from_coeffs(vec![ri(6), ri(0), ri(5), ri(0), ri(1)]);
        let (content, factors) = p.factor_over_z();
        for &x in &[-3i64, -2, -1, 0, 1, 2, 3] {
            let xr = ri(x);
            let original = p.eval(&xr);
            let mut factored = content.clone();
            for (f, m) in &factors {
                let val = f.eval(&xr);
                for _ in 0..*m {
                    factored = &factored * &val;
                }
            }
            assert_eq!(
                original, factored,
                "value mismatch at x={x}: orig={original}, factored={factored}"
            );
        }
    }
}
