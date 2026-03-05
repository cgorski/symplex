//! Sparse multivariate polynomials over ℚ.
//!
//! `MultiPoly` represents polynomials in multiple variables with
//! exact rational coefficients. Monomials are stored sparsely
//! as `(exponent_vector, coefficient)` pairs.
//!
//! This module provides:
//! - Polynomial arithmetic (+, -, ×)
//! - Degree computation
//! - Evaluation
//! - Partial derivatives
//! - Variable substitution
//!
//! Gröbner bases and system solving will be added in a future module.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};
use std::collections::BTreeMap;
use std::fmt;
use std::ops;

/// An exponent vector representing a monomial x₀^a · x₁^b · x₂^c · … as [a, b, c, …].
/// The length equals the number of variables.
pub type Exponent = Vec<u32>;

/// A sparse multivariate polynomial over ℚ.
///
/// Internally stored as a map from exponent vectors to coefficients.
/// Zero coefficients are never stored.
///
/// # Invariants
///
/// - Every key in `terms` has length `num_vars`.
/// - No value in `terms` is zero.
/// - The zero polynomial has an empty `terms` map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiPoly {
    /// Number of variables.
    num_vars: usize,
    /// Map from exponent vector to coefficient.
    /// Uses `BTreeMap` for deterministic ordering.
    terms: BTreeMap<Exponent, Ratio<BigInt>>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper: rational from i64
// ═══════════════════════════════════════════════════════════════════════════

fn rat(n: i64) -> Ratio<BigInt> {
    Ratio::from_integer(BigInt::from(n))
}

// ═══════════════════════════════════════════════════════════════════════════
// Monomial ordering — graded reverse lexicographic (grevlex)
// ═══════════════════════════════════════════════════════════════════════════

/// Compare two exponent vectors under graded reverse lexicographic order.
///
/// 1. First compare total degree (higher = greater).
/// 2. If equal, compare variable exponents from the **last** variable
///    towards the first: the monomial with the **smaller** exponent in
///    the rightmost differing position is considered **greater**.
fn cmp_grevlex(a: &[u32], b: &[u32]) -> std::cmp::Ordering {
    let deg_a: u32 = a.iter().sum();
    let deg_b: u32 = b.iter().sum();
    match deg_a.cmp(&deg_b) {
        std::cmp::Ordering::Equal => {
            // Reverse lex: scan from rightmost variable
            for i in (0..a.len()).rev() {
                match a[i].cmp(&b[i]) {
                    std::cmp::Ordering::Equal => continue,
                    // In grevlex, *smaller* exponent in the rightmost
                    // differing position means *greater* monomial.
                    std::cmp::Ordering::Less => return std::cmp::Ordering::Greater,
                    std::cmp::Ordering::Greater => return std::cmp::Ordering::Less,
                }
            }
            std::cmp::Ordering::Equal
        }
        ord => ord,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Construction
// ═══════════════════════════════════════════════════════════════════════════

impl MultiPoly {
    /// Create the zero polynomial in `num_vars` variables.
    pub fn zero(num_vars: usize) -> Self {
        MultiPoly {
            num_vars,
            terms: BTreeMap::new(),
        }
    }

    /// Create a constant polynomial.
    pub fn constant(num_vars: usize, c: Ratio<BigInt>) -> Self {
        let mut p = Self::zero(num_vars);
        if !c.is_zero() {
            p.terms.insert(vec![0; num_vars], c);
        }
        p
    }

    /// Create a polynomial from an integer constant.
    pub fn from_int(num_vars: usize, n: i64) -> Self {
        Self::constant(num_vars, rat(n))
    }

    /// Create a single-variable polynomial: the `var_index`-th variable
    /// (0-indexed).
    ///
    /// # Panics
    ///
    /// Panics if `var_index >= num_vars`.
    pub fn var(num_vars: usize, var_index: usize) -> Self {
        assert!(
            var_index < num_vars,
            "var_index {var_index} out of range for {num_vars} variables"
        );
        let mut exp = vec![0u32; num_vars];
        exp[var_index] = 1;
        let mut terms = BTreeMap::new();
        terms.insert(exp, Ratio::one());
        MultiPoly { num_vars, terms }
    }

    /// Create a monomial: `c · x₀^e₀ · x₁^e₁ · …`
    ///
    /// The number of variables is inferred from the length of `exponents`.
    pub fn monomial(c: Ratio<BigInt>, exponents: Exponent) -> Self {
        let num_vars = exponents.len();
        let mut p = Self::zero(num_vars);
        if !c.is_zero() {
            p.terms.insert(exponents, c);
        }
        p
    }

    // ─── internal helper: insert a term, combining with any existing ───

    fn insert_term(&mut self, exp: Exponent, coeff: Ratio<BigInt>) {
        if coeff.is_zero() {
            return;
        }
        let entry = self.terms.entry(exp).or_insert_with(|| Ratio::from_integer(BigInt::from(0)));
        *entry += coeff;
        // Clone the key before checking so we can remove if needed.
        // BTreeMap doesn't have retain, so we handle it inline.
    }

    /// Remove any terms whose coefficient has become zero.
    fn prune(&mut self) {
        self.terms.retain(|_, c| !c.is_zero());
    }

    /// Assert that both polynomials live in the same variable ring.
    fn assert_compatible(&self, other: &MultiPoly) {
        assert_eq!(
            self.num_vars, other.num_vars,
            "MultiPoly: incompatible variable counts ({} vs {})",
            self.num_vars, other.num_vars
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Queries
// ═══════════════════════════════════════════════════════════════════════════

impl MultiPoly {
    /// Check if this is the zero polynomial.
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Number of variables.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Number of non-zero terms.
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Total degree: maximum sum of exponents over all terms.
    ///
    /// Returns `None` for the zero polynomial (which has no defined degree).
    pub fn total_degree(&self) -> Option<u32> {
        self.terms.keys().map(|e| e.iter().sum()).max()
    }

    /// Degree in a specific variable.
    ///
    /// Returns 0 for the zero polynomial.
    ///
    /// # Panics
    ///
    /// Panics if `var_index >= num_vars`.
    pub fn degree_in(&self, var_index: usize) -> u32 {
        assert!(
            var_index < self.num_vars,
            "var_index {var_index} out of range for {} variables",
            self.num_vars
        );
        self.terms
            .keys()
            .map(|e| e[var_index])
            .max()
            .unwrap_or(0)
    }

    /// Leading term under graded reverse lexicographic order.
    ///
    /// Returns `None` for the zero polynomial.
    pub fn leading_term_grevlex(&self) -> Option<(&Exponent, &Ratio<BigInt>)> {
        self.terms
            .iter()
            .max_by(|(a, _), (b, _)| cmp_grevlex(a, b))
    }

    /// Leading coefficient under grevlex order.
    ///
    /// Returns `None` for the zero polynomial.
    pub fn leading_coeff_grevlex(&self) -> Option<&Ratio<BigInt>> {
        self.leading_term_grevlex().map(|(_, c)| c)
    }

    /// Iterate over all terms as `(exponent_vector, coefficient)` pairs.
    pub fn terms(&self) -> impl Iterator<Item = (&Exponent, &Ratio<BigInt>)> {
        self.terms.iter()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Evaluation
// ═══════════════════════════════════════════════════════════════════════════

impl MultiPoly {
    /// Evaluate the polynomial at a point: substitute each variable with
    /// a rational value.
    ///
    /// # Panics
    ///
    /// Panics if `values.len() != self.num_vars`.
    pub fn eval(&self, values: &[Ratio<BigInt>]) -> Ratio<BigInt> {
        assert_eq!(
            values.len(),
            self.num_vars,
            "eval: expected {} values, got {}",
            self.num_vars,
            values.len()
        );
        let mut result = Ratio::from_integer(BigInt::from(0));
        for (exp, coeff) in &self.terms {
            let mut term_val = coeff.clone();
            for (i, &e) in exp.iter().enumerate() {
                if e > 0 {
                    term_val *= pow_ratio(&values[i], e);
                }
            }
            result += term_val;
        }
        result
    }
}

/// Raise a rational to a non-negative integer power.
fn pow_ratio(base: &Ratio<BigInt>, exp: u32) -> Ratio<BigInt> {
    let mut result = Ratio::one();
    for _ in 0..exp {
        result *= base;
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Calculus
// ═══════════════════════════════════════════════════════════════════════════

impl MultiPoly {
    /// Partial derivative with respect to variable `var_index`.
    ///
    /// # Panics
    ///
    /// Panics if `var_index >= num_vars`.
    pub fn partial_derivative(&self, var_index: usize) -> MultiPoly {
        assert!(
            var_index < self.num_vars,
            "partial_derivative: var_index {var_index} out of range for {} variables",
            self.num_vars
        );
        let mut result = Self::zero(self.num_vars);
        for (exp, coeff) in &self.terms {
            let e_i = exp[var_index];
            if e_i == 0 {
                continue; // derivative kills this term
            }
            let new_coeff = coeff * Ratio::from_integer(BigInt::from(e_i));
            let mut new_exp = exp.clone();
            new_exp[var_index] -= 1;
            result.terms.insert(new_exp, new_coeff);
        }
        result
    }

    /// Substitute a value for one variable, reducing the number of
    /// variables by one.
    ///
    /// The resulting polynomial lives in a ring with `num_vars - 1`
    /// variables. Variable indices above `var_index` are shifted down
    /// by one.
    ///
    /// # Panics
    ///
    /// Panics if `var_index >= num_vars` or `num_vars == 0`.
    pub fn substitute(&self, var_index: usize, value: &Ratio<BigInt>) -> MultiPoly {
        assert!(
            var_index < self.num_vars,
            "substitute: var_index {var_index} out of range for {} variables",
            self.num_vars
        );
        assert!(self.num_vars > 0, "substitute: cannot reduce below 0 variables");
        let new_num_vars = self.num_vars - 1;
        let mut result = Self::zero(new_num_vars);
        for (exp, coeff) in &self.terms {
            let e_i = exp[var_index];
            let val_pow = pow_ratio(value, e_i);
            let new_coeff = coeff * val_pow;
            if new_coeff.is_zero() {
                continue;
            }
            // Build new exponent vector without var_index
            let mut new_exp = Vec::with_capacity(new_num_vars);
            for (j, &ej) in exp.iter().enumerate() {
                if j != var_index {
                    new_exp.push(ej);
                }
            }
            result.insert_term(new_exp, new_coeff);
        }
        result.prune();
        result
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Arithmetic
// ═══════════════════════════════════════════════════════════════════════════

impl MultiPoly {
    /// Add two polynomials.
    ///
    /// # Panics
    ///
    /// Panics if the polynomials have different numbers of variables.
    pub fn add(&self, other: &MultiPoly) -> MultiPoly {
        self.assert_compatible(other);
        let mut result = self.clone();
        for (exp, coeff) in &other.terms {
            result.insert_term(exp.clone(), coeff.clone());
        }
        result.prune();
        result
    }

    /// Subtract two polynomials.
    ///
    /// # Panics
    ///
    /// Panics if the polynomials have different numbers of variables.
    pub fn sub(&self, other: &MultiPoly) -> MultiPoly {
        self.assert_compatible(other);
        let mut result = self.clone();
        for (exp, coeff) in &other.terms {
            result.insert_term(exp.clone(), -coeff.clone());
        }
        result.prune();
        result
    }

    /// Negate the polynomial.
    pub fn neg(&self) -> MultiPoly {
        let terms = self
            .terms
            .iter()
            .map(|(e, c)| (e.clone(), -c.clone()))
            .collect();
        MultiPoly {
            num_vars: self.num_vars,
            terms,
        }
    }

    /// Multiply two polynomials.
    ///
    /// # Panics
    ///
    /// Panics if the polynomials have different numbers of variables.
    pub fn mul(&self, other: &MultiPoly) -> MultiPoly {
        self.assert_compatible(other);
        let mut result = Self::zero(self.num_vars);
        for (exp_a, coeff_a) in &self.terms {
            for (exp_b, coeff_b) in &other.terms {
                let new_coeff = coeff_a * coeff_b;
                let new_exp: Exponent = exp_a
                    .iter()
                    .zip(exp_b.iter())
                    .map(|(&a, &b)| a + b)
                    .collect();
                result.insert_term(new_exp, new_coeff);
            }
        }
        result.prune();
        result
    }

    /// Scale by a rational constant.
    pub fn scale(&self, c: &Ratio<BigInt>) -> MultiPoly {
        if c.is_zero() {
            return Self::zero(self.num_vars);
        }
        let terms = self
            .terms
            .iter()
            .map(|(e, coeff)| (e.clone(), coeff * c))
            .collect();
        MultiPoly {
            num_vars: self.num_vars,
            terms,
        }
    }

    /// Polynomial division with remainder in one variable (treating others
    /// as parameters).
    ///
    /// This is useful for univariate operations within a multivariate context.
    pub fn div_rem_univariate(
        &self,
        _divisor: &MultiPoly,
        _var_index: usize,
    ) -> (MultiPoly, MultiPoly) {
        todo!("univariate division within multivariate ring")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Display
// ═══════════════════════════════════════════════════════════════════════════

impl fmt::Display for MultiPoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        // Collect and sort terms by grevlex (descending).
        let mut sorted_terms: Vec<(&Exponent, &Ratio<BigInt>)> = self.terms.iter().collect();
        sorted_terms.sort_by(|(a, _), (b, _)| cmp_grevlex(b, a)); // descending

        let mut first = true;
        for (exp, coeff) in sorted_terms {
            let is_constant = exp.iter().all(|&e| e == 0);
            let coeff_is_one = *coeff == Ratio::one();
            let coeff_is_neg_one = *coeff == -Ratio::<BigInt>::one();
            let is_negative = coeff < &Ratio::from_integer(BigInt::from(0));

            if first {
                if is_constant {
                    write!(f, "{coeff}")?;
                } else if coeff_is_one {
                    write!(f, "{}", format_monomial_vars(exp))?;
                } else if coeff_is_neg_one {
                    write!(f, "-{}", format_monomial_vars(exp))?;
                } else {
                    write!(f, "{}*{}", coeff, format_monomial_vars(exp))?;
                }
            } else if is_constant {
                if is_negative {
                    write!(f, " - {}", -coeff.clone())?;
                } else {
                    write!(f, " + {coeff}")?;
                }
            } else if coeff_is_one {
                write!(f, " + {}", format_monomial_vars(exp))?;
            } else if coeff_is_neg_one {
                write!(f, " - {}", format_monomial_vars(exp))?;
            } else if is_negative {
                let pos = -coeff.clone();
                write!(f, " - {}*{}", pos, format_monomial_vars(exp))?;
            } else {
                write!(f, " + {}*{}", coeff, format_monomial_vars(exp))?;
            }
            first = false;
        }
        Ok(())
    }
}

/// Format the variable part of a monomial, e.g. `x0^2*x1`.
fn format_monomial_vars(exp: &Exponent) -> String {
    let mut parts = Vec::new();
    for (i, &e) in exp.iter().enumerate() {
        if e == 0 {
            continue;
        } else if e == 1 {
            parts.push(format!("x{i}"));
        } else {
            parts.push(format!("x{i}^{e}"));
        }
    }
    if parts.is_empty() {
        "1".to_string()
    } else {
        parts.join("*")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Operator overloads for &MultiPoly
// ═══════════════════════════════════════════════════════════════════════════

impl ops::Add for &MultiPoly {
    type Output = MultiPoly;
    fn add(self, rhs: &MultiPoly) -> MultiPoly {
        MultiPoly::add(self, rhs)
    }
}

impl ops::Sub for &MultiPoly {
    type Output = MultiPoly;
    fn sub(self, rhs: &MultiPoly) -> MultiPoly {
        MultiPoly::sub(self, rhs)
    }
}

impl ops::Mul for &MultiPoly {
    type Output = MultiPoly;
    fn mul(self, rhs: &MultiPoly) -> MultiPoly {
        MultiPoly::mul(self, rhs)
    }
}

impl ops::Neg for &MultiPoly {
    type Output = MultiPoly;
    fn neg(self) -> MultiPoly {
        MultiPoly::neg(self)
    }
}

// Also implement for owned values for convenience.

impl ops::Add for MultiPoly {
    type Output = MultiPoly;
    fn add(self, rhs: MultiPoly) -> MultiPoly {
        MultiPoly::add(&self, &rhs)
    }
}

impl ops::Sub for MultiPoly {
    type Output = MultiPoly;
    fn sub(self, rhs: MultiPoly) -> MultiPoly {
        MultiPoly::sub(&self, &rhs)
    }
}

impl ops::Mul for MultiPoly {
    type Output = MultiPoly;
    fn mul(self, rhs: MultiPoly) -> MultiPoly {
        MultiPoly::mul(&self, &rhs)
    }
}

impl ops::Neg for MultiPoly {
    type Output = MultiPoly;
    fn neg(self) -> MultiPoly {
        MultiPoly::neg(&self)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests (unit, kept close to the code)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grevlex_ordering() {
        // x^2 > xy in grevlex? Both total degree 2.
        // x^2 = [2,0], xy = [1,1].
        // Rightmost differing position: index 1 → 0 < 1, so [2,0] > [1,1].
        assert_eq!(
            cmp_grevlex(&[2, 0], &[1, 1]),
            std::cmp::Ordering::Greater
        );

        // xy > y^2 in grevlex? Both total degree 2.
        // xy = [1,1], y^2 = [0,2].
        // Rightmost differing: index 1 → 1 < 2, so [1,1] > [0,2].
        assert_eq!(
            cmp_grevlex(&[1, 1], &[0, 2]),
            std::cmp::Ordering::Greater
        );

        // Higher total degree always wins.
        assert_eq!(
            cmp_grevlex(&[3, 0], &[1, 1]),
            std::cmp::Ordering::Greater
        );
    }

    #[test]
    fn zero_is_zero() {
        let z = MultiPoly::zero(3);
        assert!(z.is_zero());
        assert_eq!(z.num_terms(), 0);
        assert_eq!(z.total_degree(), None);
    }

    #[test]
    fn constant_round_trip() {
        let c = MultiPoly::from_int(2, 42);
        assert!(!c.is_zero());
        assert_eq!(c.num_terms(), 1);
        assert_eq!(c.total_degree(), Some(0));
        assert_eq!(c.eval(&[rat(0), rat(0)]), rat(42));
    }
}
