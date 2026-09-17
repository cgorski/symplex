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
//! - Multivariate polynomial division
//! - S-polynomials for Gröbner basis computation
//!
//! The monomial ordering is parameterized via the `MonomialOrd` trait,
//! enabling graded reverse lex (default), lex, and graded lex orderings.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};
use std::collections::BTreeMap;
use std::fmt;
use std::ops;

// ═══════════════════════════════════════════════════════════════════════════
// Monomial orderings
// ═══════════════════════════════════════════════════════════════════════════

/// Trait for monomial orderings. Implemented by zero-sized types.
pub trait MonomialOrd: 'static + Clone + Send + Sync + std::fmt::Debug {
    /// Compare two exponent vectors according to this ordering.
    fn cmp_exponents(a: &[u32], b: &[u32]) -> std::cmp::Ordering;
}

/// Graded reverse lexicographic ordering (default for Gröbner computation).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GrevLex;

/// Pure lexicographic ordering (for elimination/back-substitution).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Lex;

/// Graded lexicographic ordering.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GrLex;

impl MonomialOrd for GrevLex {
    fn cmp_exponents(a: &[u32], b: &[u32]) -> std::cmp::Ordering {
        let deg_a: u32 = a.iter().sum();
        let deg_b: u32 = b.iter().sum();
        deg_a.cmp(&deg_b).then_with(|| {
            // Reverse lex: compare from the LAST variable, REVERSED
            for (ai, bi) in a.iter().rev().zip(b.iter().rev()) {
                match bi.cmp(ai) {
                    // Note: reversed! Higher last component = SMALLER in grevlex
                    std::cmp::Ordering::Equal => continue,
                    other => return other,
                }
            }
            std::cmp::Ordering::Equal
        })
    }
}

impl MonomialOrd for Lex {
    fn cmp_exponents(a: &[u32], b: &[u32]) -> std::cmp::Ordering {
        // Compare from first variable (highest priority)
        for (ai, bi) in a.iter().zip(b.iter()) {
            match ai.cmp(bi) {
                std::cmp::Ordering::Equal => continue,
                other => return other,
            }
        }
        std::cmp::Ordering::Equal
    }
}

impl MonomialOrd for GrLex {
    fn cmp_exponents(a: &[u32], b: &[u32]) -> std::cmp::Ordering {
        let deg_a: u32 = a.iter().sum();
        let deg_b: u32 = b.iter().sum();
        deg_a.cmp(&deg_b).then_with(|| Lex::cmp_exponents(a, b))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MonoKey — exponent vector with ordering
// ═══════════════════════════════════════════════════════════════════════════

/// A monomial exponent vector with ordering determined by type parameter O.
#[derive(Clone, Debug)]
pub struct MonoKey<O: MonomialOrd> {
    /// The exponent vector.
    pub exponents: Vec<u32>,
    _phantom: std::marker::PhantomData<O>,
}

impl<O: MonomialOrd> MonoKey<O> {
    /// Create a new `MonoKey` from an exponent vector.
    pub fn new(exponents: Vec<u32>) -> Self {
        Self {
            exponents,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<O: MonomialOrd> PartialEq for MonoKey<O> {
    fn eq(&self, other: &Self) -> bool {
        self.exponents == other.exponents
    }
}
impl<O: MonomialOrd> Eq for MonoKey<O> {}

impl<O: MonomialOrd> PartialOrd for MonoKey<O> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<O: MonomialOrd> Ord for MonoKey<O> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        O::cmp_exponents(&self.exponents, &other.exponents)
    }
}

impl<O: MonomialOrd> std::hash::Hash for MonoKey<O> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.exponents.hash(state);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Exponent type alias (convenience)
// ═══════════════════════════════════════════════════════════════════════════

/// An exponent vector representing a monomial x₀^a · x₁^b · x₂^c · … as [a, b, c, …].
/// The length equals the number of variables.
pub type Exponent = Vec<u32>;

// ═══════════════════════════════════════════════════════════════════════════
// MultiPoly
// ═══════════════════════════════════════════════════════════════════════════

/// A sparse multivariate polynomial over ℚ.
///
/// Internally stored as a map from exponent vectors to coefficients,
/// ordered by the monomial ordering `O`.
/// Zero coefficients are never stored.
///
/// # Invariants
///
/// - Every key in `terms` has exponent vector of length `num_vars`.
/// - No value in `terms` is zero.
/// - The zero polynomial has an empty `terms` map.
#[derive(Clone, Debug)]
pub struct MultiPoly<O: MonomialOrd = GrevLex> {
    /// Number of variables.
    num_vars: usize,
    /// Map from exponent vector to coefficient.
    /// Uses `BTreeMap` with `MonoKey<O>` for ordering-aware storage.
    terms: BTreeMap<MonoKey<O>, Ratio<BigInt>>,
}

impl<O: MonomialOrd> PartialEq for MultiPoly<O> {
    fn eq(&self, other: &Self) -> bool {
        self.num_vars == other.num_vars && self.terms == other.terms
    }
}
impl<O: MonomialOrd> Eq for MultiPoly<O> {}

// ═══════════════════════════════════════════════════════════════════════════
// Helper: rational from i64
// ═══════════════════════════════════════════════════════════════════════════

fn rat(n: i64) -> Ratio<BigInt> {
    Ratio::from_integer(BigInt::from(n))
}

// ═══════════════════════════════════════════════════════════════════════════
// Monomial helper functions
// ═══════════════════════════════════════════════════════════════════════════

/// Component-wise maximum of two exponent vectors (LCM of monomials).
pub fn monomial_lcm(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| ai.max(bi))
        .collect()
}

/// Check if monomial a divides monomial b (component-wise ≤).
pub fn monomial_divides(a: &[u32], b: &[u32]) -> bool {
    a.iter().zip(b.iter()).all(|(&ai, &bi)| ai <= bi)
}

/// Divide monomial b by a (component-wise subtraction). Returns None if a doesn't divide b.
pub fn monomial_div(a: &[u32], b: &[u32]) -> Option<Vec<u32>> {
    if !monomial_divides(a, b) {
        return None;
    }
    Some(a.iter().zip(b.iter()).map(|(&ai, &bi)| bi - ai).collect())
}

/// Multiply two monomials (component-wise addition).
pub fn monomial_mul(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai + bi).collect()
}

/// Check if two monomials are coprime (no shared variable).
pub fn monomial_coprime(a: &[u32], b: &[u32]) -> bool {
    a.iter().zip(b.iter()).all(|(&ai, &bi)| ai == 0 || bi == 0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Construction
// ═══════════════════════════════════════════════════════════════════════════

impl<O: MonomialOrd> MultiPoly<O> {
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
            p.terms.insert(MonoKey::new(vec![0; num_vars]), c);
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
        terms.insert(MonoKey::new(exp), Ratio::one());
        MultiPoly { num_vars, terms }
    }

    /// Create a monomial: `c · x₀^e₀ · x₁^e₁ · …`
    ///
    /// The number of variables is inferred from the length of `exponents`.
    pub fn monomial(c: Ratio<BigInt>, exponents: Exponent) -> Self {
        let num_vars = exponents.len();
        let mut p = Self::zero(num_vars);
        if !c.is_zero() {
            p.terms.insert(MonoKey::new(exponents), c);
        }
        p
    }

    // ─── internal helper: insert a term, combining with any existing ───

    fn insert_term(&mut self, exp: Vec<u32>, coeff: Ratio<BigInt>) {
        if coeff.is_zero() {
            return;
        }
        let key = MonoKey::new(exp);
        let entry = self
            .terms
            .entry(key)
            .or_insert_with(|| Ratio::from_integer(BigInt::from(0)));
        *entry += coeff;
    }

    /// Look up the coefficient of a given exponent vector.
    #[allow(dead_code)]
    fn get_coeff(&self, exp: &[u32]) -> Option<&Ratio<BigInt>> {
        self.terms.get(&MonoKey::<O>::new(exp.to_vec()))
    }

    /// Remove any terms whose coefficient has become zero.
    fn prune(&mut self) {
        self.terms.retain(|_, c| !c.is_zero());
    }

    /// Assert that both polynomials live in the same variable ring.
    fn assert_compatible(&self, other: &MultiPoly<O>) {
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

impl<O: MonomialOrd> MultiPoly<O> {
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
        self.terms.keys().map(|k| k.exponents.iter().sum()).max()
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
            .map(|k| k.exponents[var_index])
            .max()
            .unwrap_or(0)
    }

    /// Leading term (highest monomial under the ordering O).
    /// O(log n) via BTreeMap::last().
    pub fn leading_term(&self) -> Option<(&[u32], &Ratio<BigInt>)> {
        self.terms
            .last_key_value()
            .map(|(k, v)| (k.exponents.as_slice(), v))
    }

    /// Leading monomial exponent vector.
    pub fn leading_monomial(&self) -> Option<&[u32]> {
        self.terms
            .last_key_value()
            .map(|(k, _)| k.exponents.as_slice())
    }

    /// Leading coefficient.
    pub fn leading_coeff(&self) -> Option<&Ratio<BigInt>> {
        self.terms.last_key_value().map(|(_, v)| v)
    }

    /// Iterate over all terms as `(exponent_slice, coefficient)` pairs.
    pub fn terms(&self) -> impl Iterator<Item = (&[u32], &Ratio<BigInt>)> {
        self.terms.iter().map(|(k, v)| (k.exponents.as_slice(), v))
    }

    /// Convert this polynomial to a different monomial ordering.
    pub fn convert_order<B: MonomialOrd>(&self) -> MultiPoly<B> {
        let mut new_terms = BTreeMap::new();
        for (key, coeff) in &self.terms {
            new_terms.insert(MonoKey::<B>::new(key.exponents.clone()), coeff.clone());
        }
        MultiPoly {
            num_vars: self.num_vars,
            terms: new_terms,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Evaluation
// ═══════════════════════════════════════════════════════════════════════════

impl<O: MonomialOrd> MultiPoly<O> {
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
        for (key, coeff) in &self.terms {
            let mut term_val = coeff.clone();
            for (i, &e) in key.exponents.iter().enumerate() {
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

impl<O: MonomialOrd> MultiPoly<O> {
    /// Partial derivative with respect to variable `var_index`.
    ///
    /// # Panics
    ///
    /// Panics if `var_index >= num_vars`.
    pub fn partial_derivative(&self, var_index: usize) -> MultiPoly<O> {
        assert!(
            var_index < self.num_vars,
            "partial_derivative: var_index {var_index} out of range for {} variables",
            self.num_vars
        );
        let mut result = Self::zero(self.num_vars);
        for (key, coeff) in &self.terms {
            let e_i = key.exponents[var_index];
            if e_i == 0 {
                continue; // derivative kills this term
            }
            let new_coeff = coeff * Ratio::from_integer(BigInt::from(e_i));
            let mut new_exp = key.exponents.clone();
            new_exp[var_index] -= 1;
            result.terms.insert(MonoKey::new(new_exp), new_coeff);
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
    pub fn substitute(&self, var_index: usize, value: &Ratio<BigInt>) -> MultiPoly<O> {
        assert!(
            var_index < self.num_vars,
            "substitute: var_index {var_index} out of range for {} variables",
            self.num_vars
        );
        assert!(
            self.num_vars > 0,
            "substitute: cannot reduce below 0 variables"
        );
        let new_num_vars = self.num_vars - 1;
        let mut result = Self::zero(new_num_vars);
        for (key, coeff) in &self.terms {
            let e_i = key.exponents[var_index];
            let val_pow = pow_ratio(value, e_i);
            let new_coeff = coeff * val_pow;
            if new_coeff.is_zero() {
                continue;
            }
            // Build new exponent vector without var_index
            let mut new_exp = Vec::with_capacity(new_num_vars);
            for (j, &ej) in key.exponents.iter().enumerate() {
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

impl<O: MonomialOrd> MultiPoly<O> {
    /// Add two polynomials.
    ///
    /// # Panics
    ///
    /// Panics if the polynomials have different numbers of variables.
    pub fn add(&self, other: &MultiPoly<O>) -> MultiPoly<O> {
        self.assert_compatible(other);
        let mut result = self.clone();
        for (key, coeff) in &other.terms {
            result.insert_term(key.exponents.clone(), coeff.clone());
        }
        result.prune();
        result
    }

    /// Subtract two polynomials.
    ///
    /// # Panics
    ///
    /// Panics if the polynomials have different numbers of variables.
    pub fn sub(&self, other: &MultiPoly<O>) -> MultiPoly<O> {
        self.assert_compatible(other);
        let mut result = self.clone();
        for (key, coeff) in &other.terms {
            result.insert_term(key.exponents.clone(), -coeff.clone());
        }
        result.prune();
        result
    }

    /// Negate the polynomial.
    pub fn neg(&self) -> MultiPoly<O> {
        let terms = self
            .terms
            .iter()
            .map(|(k, c)| (k.clone(), -c.clone()))
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
    pub fn mul(&self, other: &MultiPoly<O>) -> MultiPoly<O> {
        self.assert_compatible(other);
        let mut result = Self::zero(self.num_vars);
        for (key_a, coeff_a) in &self.terms {
            for (key_b, coeff_b) in &other.terms {
                let new_coeff = coeff_a * coeff_b;
                let new_exp: Vec<u32> = key_a
                    .exponents
                    .iter()
                    .zip(key_b.exponents.iter())
                    .map(|(&a, &b)| a + b)
                    .collect();
                result.insert_term(new_exp, new_coeff);
            }
        }
        result.prune();
        result
    }

    /// Scale by a rational constant.
    pub fn scale(&self, c: &Ratio<BigInt>) -> MultiPoly<O> {
        if c.is_zero() {
            return Self::zero(self.num_vars);
        }
        let terms = self
            .terms
            .iter()
            .map(|(k, coeff)| (k.clone(), coeff * c))
            .collect();
        MultiPoly {
            num_vars: self.num_vars,
            terms,
        }
    }

    /// Multiply by a single monomial: coeff * x^exp
    pub fn mul_monomial(&self, coeff: &Ratio<BigInt>, exp: &[u32]) -> Self {
        if coeff.is_zero() {
            return Self::zero(self.num_vars);
        }
        let mut result = BTreeMap::new();
        for (key, c) in &self.terms {
            let new_exp = monomial_mul(&key.exponents, exp);
            let new_coeff = c * coeff;
            if !new_coeff.is_zero() {
                result.insert(MonoKey::new(new_exp), new_coeff);
            }
        }
        MultiPoly {
            num_vars: self.num_vars,
            terms: result,
        }
    }

    // Removed: div_rem_univariate was a todo!() stub. Use reduce() for multivariate division.
}

// ═══════════════════════════════════════════════════════════════════════════
// Monic and primitive part
// ═══════════════════════════════════════════════════════════════════════════

impl<O: MonomialOrd> MultiPoly<O> {
    /// Make monic: divide all coefficients by the leading coefficient.
    pub fn monic(&self) -> Self {
        if self.is_zero() {
            return self.clone();
        }
        let lc = self.leading_coeff().unwrap().clone();
        self.scale(&(Ratio::one() / lc))
    }

    /// Primitive part over ℚ: clear denominators, then divide by GCD of integer coefficients.
    pub fn primitive_part_q(&self) -> Self {
        if self.is_zero() {
            return self.clone();
        }
        // Find LCM of all denominators
        let mut denom_lcm = BigInt::one();
        for (_, coeff) in self.terms() {
            denom_lcm = num_integer::lcm(denom_lcm, coeff.denom().clone());
        }
        // Multiply through to clear denominators
        let scale_factor = Ratio::from_integer(denom_lcm);
        let integer_poly = self.scale(&scale_factor);
        // Find GCD of all numerators
        let mut content = BigInt::zero();
        for (_, coeff) in integer_poly.terms() {
            content = num_integer::gcd(content, coeff.numer().clone());
        }
        if content.is_zero() || content.is_one() {
            return integer_poly;
        }
        integer_poly.scale(&Ratio::new(BigInt::one(), content))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Multivariate division
// ═══════════════════════════════════════════════════════════════════════════

impl<O: MonomialOrd> MultiPoly<O> {
    /// Reduce this polynomial modulo a set of divisors.
    /// Returns the remainder after multivariate division.
    pub fn reduce(&self, divisors: &[&MultiPoly<O>]) -> MultiPoly<O> {
        if self.is_zero() || divisors.is_empty() {
            return self.clone();
        }

        let mut remainder = MultiPoly::zero(self.num_vars);
        let mut p = self.clone();

        while !p.is_zero() {
            let mut divided = false;
            let (lt_exp, lt_coeff) = p.leading_term().unwrap();
            let lt_exp = lt_exp.to_vec();
            let lt_coeff = lt_coeff.clone();

            for divisor in divisors {
                if divisor.is_zero() {
                    continue;
                }
                let (div_lt_exp, div_lt_coeff) = divisor.leading_term().unwrap();

                if monomial_divides(div_lt_exp, &lt_exp) {
                    // Can divide: subtract (lt/div_lt) * divisor from p
                    let quot_exp = monomial_div(div_lt_exp, &lt_exp).unwrap();
                    let quot_coeff = &lt_coeff / div_lt_coeff;

                    // p -= quot_monomial * divisor
                    let subtrahend = divisor.mul_monomial(&quot_coeff, &quot_exp);
                    p = p.sub(&subtrahend);
                    divided = true;
                    break;
                }
            }

            if !divided {
                // Leading term not divisible by any divisor — move to remainder
                remainder.insert_term(lt_exp.clone(), lt_coeff);
                // Remove leading term from p
                p.terms.remove(&MonoKey::<O>::new(lt_exp));
            }
        }

        remainder
    }

    /// Exact division by a single polynomial.
    ///
    /// Returns `Some(q)` with `self == q · divisor` when `divisor` divides
    /// `self` in ℚ[x₁, …, xₙ], and `None` otherwise (including for a zero
    /// divisor).  Uses the multivariate division algorithm with one divisor,
    /// for which the remainder vanishes iff the division is exact.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::multipoly::MultiPoly;
    ///
    /// let [x, y]: [MultiPoly; 2] = [MultiPoly::var(2, 0), MultiPoly::var(2, 1)];
    /// let f = x.mul(&x).sub(&y.mul(&y));            // x² − y²
    /// let g = x.sub(&y);                            // x − y
    /// assert_eq!(f.div_exact(&g), Some(x.add(&y))); // x + y
    /// assert_eq!(f.div_exact(&x), None);
    /// ```
    pub fn div_exact(&self, divisor: &MultiPoly<O>) -> Option<MultiPoly<O>> {
        self.assert_compatible(divisor);
        let (div_lt_exp, div_lt_coeff) = divisor.leading_term()?;
        let div_lt_exp = div_lt_exp.to_vec();
        let div_lt_coeff = div_lt_coeff.clone();

        let mut quotient = MultiPoly::zero(self.num_vars);
        let mut p = self.clone();
        while let Some((lt_exp, lt_coeff)) = p.leading_term() {
            let quot_exp = monomial_div(&div_lt_exp, lt_exp)?;
            let quot_coeff = lt_coeff / &div_lt_coeff;
            let subtrahend = divisor.mul_monomial(&quot_coeff, &quot_exp);
            quotient.insert_term(quot_exp, quot_coeff);
            p = p.sub(&subtrahend);
        }
        Some(quotient)
    }

    /// Component-wise minimum of all exponent vectors: the largest monomial
    /// dividing every term.  Returns the all-zero vector for the zero
    /// polynomial.
    pub fn monomial_content(&self) -> Vec<u32> {
        let mut min: Option<Vec<u32>> = None;
        for (exp, _) in self.terms() {
            match &mut min {
                None => min = Some(exp.to_vec()),
                Some(m) => {
                    for (mi, &e) in m.iter_mut().zip(exp) {
                        *mi = (*mi).min(e);
                    }
                }
            }
        }
        min.unwrap_or_else(|| vec![0; self.num_vars])
    }

    /// Indices of the variables that actually occur (with positive
    /// exponent) in some term.
    pub fn variables_present(&self) -> Vec<usize> {
        (0..self.num_vars)
            .filter(|&i| self.terms.keys().any(|k| k.exponents[i] > 0))
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// S-polynomial
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the S-polynomial of f and g.
pub fn s_polynomial<O: MonomialOrd>(f: &MultiPoly<O>, g: &MultiPoly<O>) -> MultiPoly<O> {
    assert_eq!(f.num_vars(), g.num_vars());
    if f.is_zero() || g.is_zero() {
        return MultiPoly::zero(f.num_vars());
    }

    let (lm_f, lc_f) = f.leading_term().unwrap();
    let (lm_g, lc_g) = g.leading_term().unwrap();

    let lcm = monomial_lcm(lm_f, lm_g);

    // LCM/LT(f) * f - LCM/LT(g) * g
    let quot_f = monomial_div(lm_f, &lcm).unwrap();
    let quot_g = monomial_div(lm_g, &lcm).unwrap();

    let coeff_f = Ratio::one() / lc_f;
    let coeff_g = Ratio::one() / lc_g;

    let scaled_f = f.mul_monomial(&coeff_f, &quot_f);
    let scaled_g = g.mul_monomial(&coeff_g, &quot_g);

    scaled_f.sub(&scaled_g)
}

// ═══════════════════════════════════════════════════════════════════════════
// Display
// ═══════════════════════════════════════════════════════════════════════════

impl<O: MonomialOrd> fmt::Display for MultiPoly<O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        // Iterate in REVERSE (highest term first) for conventional display
        let mut first = true;
        for (key, coeff) in self.terms.iter().rev() {
            let exp = &key.exponents;
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
fn format_monomial_vars(exp: &[u32]) -> String {
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
// Operator overloads for &MultiPoly<O>
// ═══════════════════════════════════════════════════════════════════════════

impl<O: MonomialOrd> ops::Add for &MultiPoly<O> {
    type Output = MultiPoly<O>;
    fn add(self, rhs: &MultiPoly<O>) -> MultiPoly<O> {
        MultiPoly::add(self, rhs)
    }
}

impl<O: MonomialOrd> ops::Sub for &MultiPoly<O> {
    type Output = MultiPoly<O>;
    fn sub(self, rhs: &MultiPoly<O>) -> MultiPoly<O> {
        MultiPoly::sub(self, rhs)
    }
}

impl<O: MonomialOrd> ops::Mul for &MultiPoly<O> {
    type Output = MultiPoly<O>;
    fn mul(self, rhs: &MultiPoly<O>) -> MultiPoly<O> {
        MultiPoly::mul(self, rhs)
    }
}

impl<O: MonomialOrd> ops::Neg for &MultiPoly<O> {
    type Output = MultiPoly<O>;
    fn neg(self) -> MultiPoly<O> {
        MultiPoly::neg(self)
    }
}

// Also implement for owned values for convenience.

impl<O: MonomialOrd> ops::Add for MultiPoly<O> {
    type Output = MultiPoly<O>;
    fn add(self, rhs: MultiPoly<O>) -> MultiPoly<O> {
        MultiPoly::add(&self, &rhs)
    }
}

impl<O: MonomialOrd> ops::Sub for MultiPoly<O> {
    type Output = MultiPoly<O>;
    fn sub(self, rhs: MultiPoly<O>) -> MultiPoly<O> {
        MultiPoly::sub(&self, &rhs)
    }
}

impl<O: MonomialOrd> ops::Mul for MultiPoly<O> {
    type Output = MultiPoly<O>;
    fn mul(self, rhs: MultiPoly<O>) -> MultiPoly<O> {
        MultiPoly::mul(&self, &rhs)
    }
}

impl<O: MonomialOrd> ops::Neg for MultiPoly<O> {
    type Output = MultiPoly<O>;
    fn neg(self) -> MultiPoly<O> {
        MultiPoly::neg(&self)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Operator overloads: MultiPoly<O> with i64
// ═══════════════════════════════════════════════════════════════════════════

impl<O: MonomialOrd> ops::Add<i64> for &MultiPoly<O> {
    type Output = MultiPoly<O>;
    fn add(self, rhs: i64) -> MultiPoly<O> {
        let c = MultiPoly::from_int(self.num_vars(), rhs);
        MultiPoly::add(self, &c)
    }
}
impl<O: MonomialOrd> ops::Add<i64> for MultiPoly<O> {
    type Output = MultiPoly<O>;
    fn add(self, rhs: i64) -> MultiPoly<O> {
        (&self) + rhs
    }
}

impl<O: MonomialOrd> ops::Sub<i64> for &MultiPoly<O> {
    type Output = MultiPoly<O>;
    fn sub(self, rhs: i64) -> MultiPoly<O> {
        let c = MultiPoly::from_int(self.num_vars(), rhs);
        MultiPoly::sub(self, &c)
    }
}
impl<O: MonomialOrd> ops::Sub<i64> for MultiPoly<O> {
    type Output = MultiPoly<O>;
    fn sub(self, rhs: i64) -> MultiPoly<O> {
        (&self) - rhs
    }
}

impl<O: MonomialOrd> ops::Mul<i64> for &MultiPoly<O> {
    type Output = MultiPoly<O>;
    fn mul(self, rhs: i64) -> MultiPoly<O> {
        let c = Ratio::from_integer(BigInt::from(rhs));
        self.scale(&c)
    }
}
impl<O: MonomialOrd> ops::Mul<i64> for MultiPoly<O> {
    type Output = MultiPoly<O>;
    fn mul(self, rhs: i64) -> MultiPoly<O> {
        (&self) * rhs
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Ring builder
// ═══════════════════════════════════════════════════════════════════════════

/// Create variable polynomials for a ring with the given number of variables.
///
/// Returns a vector where element i is the polynomial xᵢ.
///
/// # Example
/// ```
/// use symplex::multipoly::*;
/// let vars = multipoly_vars::<GrevLex>(2);
/// let x = &vars[0];
/// let y = &vars[1];
/// let circle = x * x + y * y - 1;  // x² + y² - 1
/// ```
pub fn multipoly_vars<O: MonomialOrd>(num_vars: usize) -> Vec<MultiPoly<O>> {
    (0..num_vars).map(|i| MultiPoly::var(num_vars, i)).collect()
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
            GrevLex::cmp_exponents(&[2, 0], &[1, 1]),
            std::cmp::Ordering::Greater
        );

        // xy > y^2 in grevlex? Both total degree 2.
        // xy = [1,1], y^2 = [0,2].
        // Rightmost differing: index 1 → 1 < 2, so [1,1] > [0,2].
        assert_eq!(
            GrevLex::cmp_exponents(&[1, 1], &[0, 2]),
            std::cmp::Ordering::Greater
        );

        // Higher total degree always wins.
        assert_eq!(
            GrevLex::cmp_exponents(&[3, 0], &[1, 1]),
            std::cmp::Ordering::Greater
        );
    }

    #[test]
    fn zero_is_zero() {
        let z: MultiPoly<GrevLex> = MultiPoly::zero(3);
        assert!(z.is_zero());
        assert_eq!(z.num_terms(), 0);
        assert_eq!(z.total_degree(), None);
    }

    #[test]
    fn constant_round_trip() {
        let c: MultiPoly<GrevLex> = MultiPoly::from_int(2, 42);
        assert!(!c.is_zero());
        assert_eq!(c.num_terms(), 1);
        assert_eq!(c.total_degree(), Some(0));
        assert_eq!(c.eval(&[rat(0), rat(0)]), rat(42));
    }
}
