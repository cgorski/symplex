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

use super::zpoly::{self, ZPoly, pow_ratio};
use crate::base::errors::SymplexError;

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

/// A monomial order chosen at run time — the value-level counterpart of
/// the zero-sized order types [`Lex`] and [`GrevLex`], for APIs that take
/// the order as an argument (`Ex::groebner`, `Ex::reduce_modulo`).
///
/// `Lex` orders by the first variable first (elimination / triangular
/// bases); `GrevLex` orders by total degree, then reverse lexicographically
/// (the default for Gröbner computation, usually much faster).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MonomialOrder {
    /// Pure lexicographic order ([`Lex`]).
    Lex,
    /// Graded reverse lexicographic order ([`GrevLex`]).
    GrevLex,
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
///
/// Returns `None` if the exponent vectors have different lengths or an
/// exponent of the product would overflow `u32`.
///
/// # Examples
///
/// ```
/// use symplex::multipoly::monomial_mul;
///
/// assert_eq!(monomial_mul(&[2, 1], &[1, 3]), Some(vec![3, 4]));
/// assert_eq!(monomial_mul(&[u32::MAX], &[1]), None);
/// assert_eq!(monomial_mul(&[1, 2], &[1]), None);
/// ```
pub fn monomial_mul(a: &[u32], b: &[u32]) -> Option<Vec<u32>> {
    if a.len() != b.len() {
        return None;
    }
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| ai.checked_add(bi))
        .collect()
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
    /// Panics if `var_index >= num_vars`; [`try_var`](Self::try_var)
    /// returns `None` instead.
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

    /// The `var_index`-th variable (0-indexed) of the ring in `num_vars`
    /// variables, as [`var`](Self::var); `None` if `var_index >= num_vars`.
    ///
    /// ```
    /// use symplex::multipoly::MultiPoly;
    ///
    /// let y = MultiPoly::<symplex::multipoly::GrevLex>::try_var(2, 1);
    /// assert_eq!(y, Some(MultiPoly::var(2, 1)));
    /// assert_eq!(MultiPoly::<symplex::multipoly::GrevLex>::try_var(2, 2), None);
    /// ```
    pub fn try_var(num_vars: usize, var_index: usize) -> Option<Self> {
        (var_index < num_vars).then(|| Self::var(num_vars, var_index))
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
    ///
    /// Returns `None` when the monomial is absent (its coefficient is zero)
    /// or when `exp` has the wrong length.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::multipoly::MultiPoly;
    /// use num_bigint::BigInt;
    /// use num_rational::Ratio;
    ///
    /// let [x, y]: [MultiPoly; 2] = [MultiPoly::var(2, 0), MultiPoly::var(2, 1)];
    /// let f = x.mul(&y).scale(&Ratio::from_integer(BigInt::from(3))).add(&x);
    /// assert_eq!(f.coeff(&[1, 1]), Some(&Ratio::from_integer(BigInt::from(3))));
    /// assert_eq!(f.coeff(&[0, 1]), None);
    /// ```
    pub fn coeff(&self, exp: &[u32]) -> Option<&Ratio<BigInt>> {
        if exp.len() != self.num_vars {
            return None;
        }
        self.terms.get(&MonoKey::<O>::new(exp.to_vec()))
    }

    /// Build a polynomial from `(exponent_vector, coefficient)` pairs.
    ///
    /// Repeated monomials are summed and zero coefficients are dropped.
    /// Returns `None` if any exponent vector does not have length
    /// `num_vars`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::multipoly::MultiPoly;
    /// use num_bigint::BigInt;
    /// use num_rational::Ratio;
    ///
    /// let r = |n: i64| Ratio::from_integer(BigInt::from(n));
    /// let f: MultiPoly = MultiPoly::from_terms(2, vec![(vec![1, 0], r(2)), (vec![1, 0], r(-2)), (vec![0, 1], r(5))]).unwrap();
    /// assert_eq!(f.num_terms(), 1);
    /// assert_eq!(f.coeff(&[0, 1]), Some(&r(5)));
    /// assert!(MultiPoly::<symplex::multipoly::GrevLex>::from_terms(2, vec![(vec![1], r(1))]).is_none());
    /// ```
    pub fn from_terms(num_vars: usize, terms: Vec<(Vec<u32>, Ratio<BigInt>)>) -> Option<Self> {
        let mut p = Self::zero(num_vars);
        for (exp, c) in terms {
            if exp.len() != num_vars {
                return None;
            }
            p.insert_term(exp, c);
        }
        p.prune();
        Some(p)
    }

    /// [`from_terms`](Self::from_terms) for pairs that are already reduced
    /// and (normally) free of duplicates and zeros: each coefficient is
    /// stored as given instead of being added to a fresh zero.  Duplicate
    /// monomials are still summed and zero coefficients dropped, so the
    /// result is well-formed either way.  `None` if an exponent vector does
    /// not have length `num_vars`.
    pub(crate) fn from_distinct_terms(
        num_vars: usize,
        terms: Vec<(Vec<u32>, Ratio<BigInt>)>,
    ) -> Option<Self> {
        let mut p = Self::zero(num_vars);
        for (exp, c) in terms {
            if exp.len() != num_vars {
                return None;
            }
            if c.is_zero() {
                continue;
            }
            match p.terms.entry(MonoKey::new(exp)) {
                std::collections::btree_map::Entry::Vacant(slot) => {
                    slot.insert(c);
                }
                std::collections::btree_map::Entry::Occupied(mut slot) => {
                    *slot.get_mut() += c;
                }
            }
        }
        p.prune();
        Some(p)
    }

    /// A polynomial from a ready-made term map (zero coefficients are
    /// dropped; every key must have `num_vars` exponents).
    pub(crate) fn from_term_map(
        num_vars: usize,
        terms: BTreeMap<MonoKey<O>, Ratio<BigInt>>,
    ) -> Self {
        let mut p = MultiPoly { num_vars, terms };
        p.prune();
        p
    }

    /// Apply `f` to every coefficient, dropping terms that become zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::multipoly::MultiPoly;
    /// use num_bigint::BigInt;
    /// use num_rational::Ratio;
    ///
    /// let x: MultiPoly = MultiPoly::var(1, 0);
    /// let f = x.scale(&Ratio::from_integer(BigInt::from(6))) + 4;
    /// let halved = f.map_coeffs(|c| c / Ratio::from_integer(BigInt::from(2)));
    /// assert_eq!(halved, x.scale(&Ratio::from_integer(BigInt::from(3))) + 2);
    /// ```
    pub fn map_coeffs(&self, mut f: impl FnMut(&Ratio<BigInt>) -> Ratio<BigInt>) -> Self {
        let mut terms = BTreeMap::new();
        for (k, c) in &self.terms {
            let nc = f(c);
            if !nc.is_zero() {
                terms.insert(k.clone(), nc);
            }
        }
        MultiPoly {
            num_vars: self.num_vars,
            terms,
        }
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
    /// Returns 0 for the zero polynomial, and for `var_index >= num_vars`:
    /// a polynomial in `x₀, …, xₙ₋₁` does not involve any other `xₖ`
    /// (as [`coeff`](Self::coeff) reports an exponent vector of the wrong
    /// length as absent).  Before 0.29 an index out of range panicked.
    ///
    /// ```
    /// use symplex::multipoly::MultiPoly;
    ///
    /// let [x, y]: [MultiPoly; 2] = [MultiPoly::var(2, 0), MultiPoly::var(2, 1)];
    /// let p = x.mul(&x).mul(&y);                   // x²y
    /// assert_eq!((p.degree_in(0), p.degree_in(1), p.degree_in(2)), (2, 1, 0));
    /// ```
    pub fn degree_in(&self, var_index: usize) -> u32 {
        self.terms
            .keys()
            .filter_map(|k| k.exponents.get(var_index).copied())
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

    /// Iterate over all terms as `(exponent_slice, coefficient)` pairs, in
    /// ascending order of `O`; `.rev()` walks them from the leading term
    /// down.
    pub fn terms(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&[u32], &Ratio<BigInt>)> + ExactSizeIterator {
        self.terms.iter().map(|(k, v)| (k.exponents.as_slice(), v))
    }

    /// The value of a constant polynomial (`Some(0)` for the zero
    /// polynomial), or `None` when some variable occurs.
    ///
    /// ```
    /// use symplex::multipoly::MultiPoly;
    /// use symplex::linprog::qi;
    ///
    /// let x: MultiPoly = MultiPoly::var(2, 0);
    /// assert_eq!(MultiPoly::<symplex::multipoly::GrevLex>::zero(2).as_constant(), Some(qi(0)));
    /// assert_eq!(MultiPoly::<symplex::multipoly::GrevLex>::from_int(2, 7).as_constant(), Some(qi(7)));
    /// assert_eq!(x.as_constant(), None);
    /// ```
    pub fn as_constant(&self) -> Option<Ratio<BigInt>> {
        match self.terms.len() {
            0 => Some(Ratio::from_integer(BigInt::from(0))),
            1 => self
                .terms
                .iter()
                .next()
                .filter(|(k, _)| k.exponents.iter().all(|&e| e == 0))
                .map(|(_, c)| c.clone()),
            _ => None,
        }
    }

    /// The polynomial as an affine form `a₀x₀ + … + a_{n−1}x_{n−1} + c`,
    /// or `None` if its total degree exceeds one.  The zero polynomial is
    /// `(0, …, 0; 0)`.
    ///
    /// This is the bridge from a polynomial that is affine in its variables
    /// (a half-space `p ≥ 0`, say) to explicit coefficients.
    ///
    /// ```
    /// use symplex::multipoly::MultiPoly;
    /// use symplex::linprog::qi;
    ///
    /// let [x, y]: [MultiPoly; 2] = [MultiPoly::var(2, 0), MultiPoly::var(2, 1)];
    /// let p = x.scale(&qi(3)).sub(&y).add(&MultiPoly::from_int(2, 5));
    /// assert_eq!(p.affine_form(), Some((vec![qi(3), qi(-1)], qi(5))));
    /// assert_eq!(x.mul(&y).affine_form(), None);
    /// ```
    pub fn affine_form(&self) -> Option<(Vec<Ratio<BigInt>>, Ratio<BigInt>)> {
        let zero = Ratio::from_integer(BigInt::from(0));
        let mut coeffs = vec![zero.clone(); self.num_vars];
        let mut constant = zero;
        for (key, c) in &self.terms {
            let degree: u32 = key.exponents.iter().sum();
            match degree {
                0 => constant = c.clone(),
                1 => {
                    let i = key.exponents.iter().position(|&e| e == 1)?;
                    coeffs[i] = c.clone();
                }
                _ => return None,
            }
        }
        Some((coeffs, constant))
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
    /// Computed over a common denominator with a single reduction at the
    /// end (`zpoly::ZPoly::eval`).
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if
    /// `values.len() != self.num_vars()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::multipoly::MultiPoly;
    /// use symplex::linprog::qi;
    ///
    /// let [x, y]: [MultiPoly; 2] = [MultiPoly::var(2, 0), MultiPoly::var(2, 1)];
    /// let p = x.mul(&y).add(&x);                    // x·y + x
    /// assert_eq!(p.eval(&[qi(2), qi(3)]).unwrap(), qi(8));
    /// assert!(p.eval(&[qi(2)]).is_err());
    /// ```
    pub fn eval(&self, values: &[Ratio<BigInt>]) -> Result<Ratio<BigInt>, SymplexError> {
        if values.len() != self.num_vars {
            return Err(SymplexError::invalid_argument(
                "MultiPoly::eval",
                format!("expected {} values, got {}", self.num_vars, values.len()),
            ));
        }
        Ok(ZPoly::from_multipoly(self).eval(values))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Calculus
// ═══════════════════════════════════════════════════════════════════════════

impl<O: MonomialOrd> MultiPoly<O> {
    /// Partial derivative with respect to variable `var_index`.
    ///
    /// For `var_index >= num_vars` it is the zero polynomial (in the same
    /// `num_vars` variables): the polynomial does not involve that variable
    /// ([`degree_in`](Self::degree_in) is 0).  Before 0.29 an index out of
    /// range panicked.
    ///
    /// ```
    /// use symplex::multipoly::MultiPoly;
    ///
    /// let [x, y]: [MultiPoly; 2] = [MultiPoly::var(2, 0), MultiPoly::var(2, 1)];
    /// let p = x.mul(&x).mul(&y);                   // x²y
    /// assert_eq!(p.partial_derivative(0), x.mul(&y).scale(&symplex::linprog::qi(2)));
    /// assert_eq!(p.partial_derivative(2), MultiPoly::zero(2));
    /// ```
    pub fn partial_derivative(&self, var_index: usize) -> MultiPoly<O> {
        if var_index >= self.num_vars {
            return Self::zero(self.num_vars);
        }
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

    /// Substitute a value for one variable, **keeping** the number of
    /// variables (the variable simply no longer occurs), so that the
    /// remaining variables keep their indices.  This is the operation for
    /// instantiating a parameter: `p(j, x, y)` at `j = 3` is `p(3, x, y)`
    /// as a polynomial in the same three slots.  Use
    /// [`substitute`](Self::substitute) to also drop the variable.
    ///
    /// ```
    /// use symplex::multipoly::MultiPoly;
    /// use symplex::linprog::qi;
    ///
    /// let [j, x]: [MultiPoly; 2] = [MultiPoly::var(2, 0), MultiPoly::var(2, 1)];
    /// let p = j.mul(&x).add(&j.mul(&j));           // j·x + j²
    /// let at3 = p.eval_var(0, &qi(3));
    /// assert_eq!(at3.num_vars(), 2);
    /// assert_eq!(at3.affine_form(), Some((vec![qi(0), qi(3)], qi(9))));
    /// // A variable the ring does not have does not occur: unchanged.
    /// assert_eq!(p.eval_var(2, &qi(3)), p);
    /// ```
    ///
    /// For `var_index >= num_vars` the polynomial is returned unchanged (it
    /// does not involve that variable).  Before 0.29 an index out of range
    /// panicked.
    pub fn eval_var(&self, var_index: usize, value: &Ratio<BigInt>) -> MultiPoly<O> {
        if var_index >= self.num_vars {
            return self.clone();
        }
        let mut result = Self::zero(self.num_vars);
        for (key, coeff) in &self.terms {
            let e_i = key.exponents[var_index];
            let new_coeff = coeff * pow_ratio(value, e_i);
            if new_coeff.is_zero() {
                continue;
            }
            let mut new_exp = key.exponents.clone();
            new_exp[var_index] = 0;
            result.insert_term(new_exp, new_coeff);
        }
        result.prune();
        result
    }

    /// Substitute a value for one variable, reducing the number of
    /// variables by one.
    ///
    /// The resulting polynomial lives in a ring with `num_vars - 1`
    /// variables. Variable indices above `var_index` are shifted down
    /// by one.  See [`eval_var`](Self::eval_var) to keep the indices.
    ///
    /// # Panics
    ///
    /// Panics if `var_index >= num_vars` (in particular if `num_vars == 0`);
    /// [`try_substitute`](Self::try_substitute) returns `None` instead.
    pub fn substitute(&self, var_index: usize, value: &Ratio<BigInt>) -> MultiPoly<O> {
        // `var_index < num_vars` also makes `num_vars - 1` well defined.
        assert!(
            var_index < self.num_vars,
            "substitute: var_index {var_index} out of range for {} variables",
            self.num_vars
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

    /// [`substitute`](Self::substitute): a value for one variable, which
    /// is dropped from the ring; `None` if `var_index >= num_vars`.
    ///
    /// ```
    /// use symplex::multipoly::MultiPoly;
    /// use symplex::linprog::qi;
    ///
    /// let [j, x]: [MultiPoly; 2] = [MultiPoly::var(2, 0), MultiPoly::var(2, 1)];
    /// let p = j.mul(&x).add(&j.mul(&j));           // j·x + j²
    /// let at3 = p.try_substitute(0, &qi(3)).unwrap();
    /// assert_eq!(at3, MultiPoly::var(1, 0).scale(&qi(3)) + 9);
    /// assert_eq!(p.try_substitute(2, &qi(3)), None);
    /// ```
    pub fn try_substitute(&self, var_index: usize, value: &Ratio<BigInt>) -> Option<MultiPoly<O>> {
        (var_index < self.num_vars).then(|| self.substitute(var_index, value))
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
    /// Panics if the polynomials have different numbers of variables (this
    /// is the body of `+`, which cannot return an error); use
    /// [`try_add`](Self::try_add) to observe the mismatch.
    pub fn add(&self, other: &MultiPoly<O>) -> MultiPoly<O> {
        self.assert_compatible(other);
        let mut result = self.clone();
        for (key, coeff) in &other.terms {
            result.insert_term(key.exponents.clone(), coeff.clone());
        }
        result.prune();
        result
    }

    /// Add two polynomials; `None` if they have different numbers of
    /// variables.
    pub fn try_add(&self, other: &MultiPoly<O>) -> Option<MultiPoly<O>> {
        (self.num_vars == other.num_vars).then(|| self.add(other))
    }

    /// Subtract two polynomials.
    ///
    /// # Panics
    ///
    /// Panics if the polynomials have different numbers of variables (this
    /// is the body of `-`, which cannot return an error); use
    /// [`try_sub`](Self::try_sub) to observe the mismatch.
    pub fn sub(&self, other: &MultiPoly<O>) -> MultiPoly<O> {
        self.assert_compatible(other);
        let mut result = self.clone();
        for (key, coeff) in &other.terms {
            result.insert_term(key.exponents.clone(), -coeff.clone());
        }
        result.prune();
        result
    }

    /// Subtract two polynomials; `None` if they have different numbers of
    /// variables.
    pub fn try_sub(&self, other: &MultiPoly<O>) -> Option<MultiPoly<O>> {
        (self.num_vars == other.num_vars).then(|| self.sub(other))
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
    /// The product is accumulated over a common denominator in `ℤ` and
    /// reduced once per term (`zpoly::ZPoly`).
    ///
    /// # Panics
    ///
    /// Panics if the polynomials have different numbers of variables, or
    /// if an exponent of the product overflows `u32` (this is the body of
    /// `*`, which cannot return an error; use [`try_mul`](Self::try_mul) to
    /// observe either).
    pub fn mul(&self, other: &MultiPoly<O>) -> MultiPoly<O> {
        self.assert_compatible(other);
        let prod = self.try_mul(other);
        assert!(prod.is_some(), "MultiPoly::mul: exponent overflow");
        prod.unwrap_or_else(|| Self::zero(self.num_vars))
    }

    /// Multiply two polynomials; `None` if the polynomials have different
    /// numbers of variables or an exponent of the product would overflow
    /// `u32`.
    pub fn try_mul(&self, other: &MultiPoly<O>) -> Option<MultiPoly<O>> {
        if self.num_vars != other.num_vars {
            return None;
        }
        ZPoly::from_multipoly(self)
            .mul(&ZPoly::from_multipoly(other))
            .map(|z| z.into_multipoly(self.num_vars))
    }

    /// `self^n` by repeated squaring (`self^0 = 1`, also for the zero
    /// polynomial).
    ///
    /// # Panics
    ///
    /// Panics if an exponent of the power overflows `u32` (use
    /// [`try_pow`](Self::try_pow) to observe the overflow).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::multipoly::MultiPoly;
    ///
    /// let [x, y]: [MultiPoly; 2] = [MultiPoly::var(2, 0), MultiPoly::var(2, 1)];
    /// let s = x.add(&y);
    /// assert_eq!(s.pow(3), s.mul(&s).mul(&s));
    /// assert_eq!(s.pow(0), MultiPoly::from_int(2, 1));
    /// ```
    pub fn pow(&self, n: u32) -> MultiPoly<O> {
        let p = self.try_pow(n);
        assert!(p.is_some(), "MultiPoly::pow: exponent overflow");
        p.unwrap_or_else(|| Self::zero(self.num_vars))
    }

    /// `self^n` by repeated squaring; `None` if an exponent would overflow
    /// `u32`.
    pub fn try_pow(&self, n: u32) -> Option<MultiPoly<O>> {
        match n {
            0 => return Some(Self::from_int(self.num_vars, 1)),
            1 => return Some(self.clone()),
            _ => {}
        }
        ZPoly::from_multipoly(self)
            .pow(self.num_vars, n)
            .map(|z| z.into_multipoly(self.num_vars))
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

    /// Multiply by a single monomial: `coeff · x^exp`.
    ///
    /// Equivalent to `self.mul(&MultiPoly::monomial(coeff, exp))`, without
    /// the general product.
    ///
    /// # Panics
    ///
    /// As [`mul`](Self::mul): panics if `exp.len() != self.num_vars()` or an
    /// exponent of the product overflows `u32`; use
    /// [`try_mul_monomial`](Self::try_mul_monomial) to observe either.
    /// (Before 0.29 the exponent wrapped around in release builds.)
    pub fn mul_monomial(&self, coeff: &Ratio<BigInt>, exp: &[u32]) -> Self {
        match self.try_mul_monomial(coeff, exp) {
            Some(p) => p,
            // Wrong length or overflow: `mul` reports it exactly as the
            // general product would (its documented panic).
            None => self.mul(&Self::monomial(coeff.clone(), exp.to_vec())),
        }
    }

    /// Multiply by a single monomial: `coeff · x^exp`; `None` if
    /// `exp.len() != self.num_vars()` or an exponent of the product would
    /// overflow `u32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::multipoly::MultiPoly;
    /// use symplex::linprog::qi;
    ///
    /// let [x, y]: [MultiPoly; 2] = [MultiPoly::var(2, 0), MultiPoly::var(2, 1)];
    /// let p = x.add(&y);
    /// assert_eq!(p.try_mul_monomial(&qi(2), &[1, 0]), Some(x.mul(&p).scale(&qi(2))));
    /// assert_eq!(p.try_mul_monomial(&qi(1), &[u32::MAX, 0]), None);
    /// assert_eq!(p.try_mul_monomial(&qi(1), &[1]), None);
    /// ```
    pub fn try_mul_monomial(&self, coeff: &Ratio<BigInt>, exp: &[u32]) -> Option<Self> {
        if exp.len() != self.num_vars {
            return None;
        }
        if coeff.is_zero() {
            return Some(Self::zero(self.num_vars));
        }
        let mut result = BTreeMap::new();
        for (key, c) in &self.terms {
            let new_exp = monomial_mul(&key.exponents, exp)?;
            let new_coeff = c * coeff;
            if !new_coeff.is_zero() {
                result.insert(MonoKey::new(new_exp), new_coeff);
            }
        }
        Some(MultiPoly {
            num_vars: self.num_vars,
            terms: result,
        })
    }

    // Removed: div_rem_univariate was a todo!() stub. Use reduce() for multivariate division.
}

// ═══════════════════════════════════════════════════════════════════════════
// Monic and primitive part
// ═══════════════════════════════════════════════════════════════════════════

impl<O: MonomialOrd> MultiPoly<O> {
    /// Make monic: divide all coefficients by the leading coefficient.
    pub fn monic(&self) -> Self {
        let Some(lc) = self.leading_coeff() else {
            return self.clone();
        };
        self.scale(&(Ratio::one() / lc.clone()))
    }

    /// Primitive part over ℚ: clear denominators, then divide by GCD of integer coefficients.
    pub fn primitive_part_q(&self) -> Self {
        if self.is_zero() {
            return self.clone();
        }
        let (_, integer_poly) = self.clear_denominators();
        let content = integer_poly.integer_content();
        if content.is_zero() || content.is_one() {
            return integer_poly;
        }
        integer_poly.scale(&Ratio::new(BigInt::one(), content))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Integer content, denominators, heuristic GCD
// ═══════════════════════════════════════════════════════════════════════════

/// Maximum number of evaluation points tried by the heuristic GCD before
/// giving up.
const HEUGCD_MAX_TRIES: usize = 6;

/// Symmetric remainder of `c` modulo `m`: the representative of `c mod m`
/// in `(-m/2, m/2]`.
fn symmetric_mod(c: &BigInt, m: &BigInt) -> BigInt {
    use num_integer::Integer;
    let r = c.mod_floor(m);
    if &r + &r > *m { r - m } else { r }
}

impl<O: MonomialOrd> MultiPoly<O> {
    /// GCD of the numerators of all coefficients (non-negative).
    ///
    /// For a polynomial with integer coefficients this is the usual
    /// integer content; denominators are ignored, so call
    /// [`clear_denominators`](Self::clear_denominators) first for a
    /// general rational polynomial.  The zero polynomial has content `0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::multipoly::MultiPoly;
    /// use num_bigint::BigInt;
    ///
    /// let x: MultiPoly = MultiPoly::var(1, 0);
    /// let f = x.clone() * 6 + 9;
    /// assert_eq!(f.integer_content(), BigInt::from(3));
    /// assert_eq!(MultiPoly::<symplex::multipoly::GrevLex>::zero(1).integer_content(), BigInt::from(0));
    /// ```
    pub fn integer_content(&self) -> BigInt {
        zpoly::integer_content(self.terms().map(|(_, c)| c.numer()))
    }

    /// Multiply through by the least common multiple `d` of all coefficient
    /// denominators, returning `(d, d · self)`; the second component has
    /// integer coefficients.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::multipoly::MultiPoly;
    /// use num_bigint::BigInt;
    /// use num_rational::Ratio;
    ///
    /// let x: MultiPoly = MultiPoly::var(1, 0);
    /// let f = x.scale(&Ratio::new(BigInt::from(1), BigInt::from(2))) + 1;   // x/2 + 1
    /// let (d, g) = f.clear_denominators();
    /// assert_eq!(d, BigInt::from(2));
    /// assert_eq!(g, x + 2);
    /// ```
    pub fn clear_denominators(&self) -> (BigInt, Self) {
        let d = zpoly::denominator_lcm(self.terms().map(|(_, c)| c));
        if d.is_one() {
            return (d, self.clone());
        }
        let scaled = self.scale(&Ratio::from_integer(d.clone()));
        (d, scaled)
    }

    /// Largest absolute value of a coefficient numerator (the max-norm for
    /// integer polynomials).  Zero for the zero polynomial.
    fn max_norm(&self) -> BigInt {
        let mut m = BigInt::zero();
        for (_, c) in self.terms() {
            let a = num_traits::Signed::abs(c.numer());
            if a > m {
                m = a;
            }
        }
        m
    }

    /// `true` if the leading coefficient (under `O`) is negative.
    fn leading_is_negative(&self) -> bool {
        self.leading_coeff()
            .is_some_and(num_traits::Signed::is_negative)
    }

    /// Integer-normalised form used by [`gcd`](Self::gcd): denominators
    /// cleared and leading coefficient made positive.
    fn normalized_over_z(&self) -> Self {
        let (_, z) = self.clear_denominators();
        if z.leading_is_negative() { z.neg() } else { z }
    }

    /// Greatest common divisor in ℤ[x₁, …, xₙ] of the inputs after clearing
    /// denominators, computed with the heuristic GCD algorithm (GCDHEU).
    ///
    /// The result has integer coefficients, positive leading coefficient
    /// (under `O`), and integer content equal to the GCD of the inputs'
    /// integer contents.  For inputs with integer coefficients this is the
    /// ordinary GCD over ℤ; over ℚ the GCD is only defined up to a nonzero
    /// rational factor, and this normalisation picks one representative.
    /// `gcd(0, 0) = 0`, `gcd(f, 0)` is the normalised `f`.
    ///
    /// The heuristic evaluates one variable at a large integer, recurses on
    /// the remaining variables (integer GCD in the univariate case), and
    /// reconstructs the candidate by symmetric ξ-adic expansion; the
    /// candidate is verified by exact division of both inputs, so a wrong
    /// answer is never returned.  If every evaluation point fails (which
    /// does not happen for inputs of realistic size), the constant `1` is
    /// returned, meaning "no common factor found".
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::multipoly::MultiPoly;
    ///
    /// let [x, y]: [MultiPoly; 2] = [MultiPoly::var(2, 0), MultiPoly::var(2, 1)];
    /// let s = x.add(&y);                 // x + y
    /// let d = x.sub(&y);                 // x − y
    /// let f = s.mul(&d);                 // x² − y²
    /// let g = s.mul(&s);                 // (x + y)²
    /// assert_eq!(MultiPoly::gcd(&f, &g), s);
    /// assert_eq!(MultiPoly::gcd(&x, &y), MultiPoly::from_int(2, 1));
    /// ```
    pub fn gcd(a: &Self, b: &Self) -> Self {
        if a.num_vars != b.num_vars {
            return Self::from_int(a.num_vars, 1);
        }
        match (a.is_zero(), b.is_zero()) {
            (true, true) => return Self::zero(a.num_vars),
            (true, false) => return b.normalized_over_z(),
            (false, true) => return a.normalized_over_z(),
            (false, false) => {}
        }
        let (_, az) = a.clear_denominators();
        let (_, bz) = b.clear_denominators();
        match Self::heugcd_z(&az, &bz, 0) {
            Some(h) => {
                if h.leading_is_negative() {
                    h.neg()
                } else {
                    h
                }
            }
            None => Self::from_int(a.num_vars, 1),
        }
    }

    /// Least common multiple `a · b / gcd(a, b)` (integer-normalised like
    /// [`gcd`](Self::gcd)); zero if either input is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::multipoly::MultiPoly;
    ///
    /// let [x, y]: [MultiPoly; 2] = [MultiPoly::var(2, 0), MultiPoly::var(2, 1)];
    /// let f = x.mul(&y);                 // xy
    /// let g = y.mul(&y);                 // y²
    /// assert_eq!(MultiPoly::lcm(&f, &g), x.mul(&y).mul(&y));
    /// ```
    pub fn lcm(a: &Self, b: &Self) -> Self {
        if a.is_zero() || b.is_zero() {
            return Self::zero(a.num_vars);
        }
        let g = Self::gcd(a, b);
        let az = a.normalized_over_z();
        let bz = b.normalized_over_z();
        let prod = az.mul(&bz);
        match prod.div_exact(&g) {
            Some(l) => l,
            None => prod,
        }
    }

    /// Heuristic GCD over ℤ for nonzero integer-coefficient inputs with the
    /// same number of variables.  Returns the full GCD (including integer
    /// content), or `None` if every evaluation point failed.
    fn heugcd_z(f: &Self, g: &Self, depth: usize) -> Option<Self> {
        let nv = f.num_vars;
        // Integer content.
        let cf = f.integer_content();
        let cg = g.integer_content();
        if cf.is_zero() || cg.is_zero() {
            return None;
        }
        let c = num_integer::gcd(cf.clone(), cg.clone());
        let inv_cf = Ratio::new(BigInt::one(), cf);
        let inv_cg = Ratio::new(BigInt::one(), cg);
        let f = f.scale(&inv_cf);
        let g = g.scale(&inv_cg);
        let c_rat = Ratio::from_integer(c);

        if nv == 0 {
            // Both are ±1 after content removal.
            return Some(Self::constant(0, c_rat));
        }
        // A primitive constant is ±1: the GCD is the content GCD.
        if f.total_degree() == Some(0) || g.total_degree() == Some(0) {
            return Some(Self::constant(nv, c_rat));
        }
        if f == g || f == g.neg() {
            return Some(f.scale(&c_rat));
        }
        // Cheap exact-division shortcuts.
        if g.div_exact(&f).is_some() {
            return Some(f.scale(&c_rat));
        }
        if f.div_exact(&g).is_some() {
            return Some(g.scale(&c_rat));
        }
        // Guard against pathological recursion depth (one level per variable).
        if depth > nv + 1 {
            return None;
        }

        let var = nv - 1;
        let f_norm = f.max_norm();
        let g_norm = g.max_norm();
        let two = BigInt::from(2);
        let mut xi: BigInt = &two * f_norm.min(g_norm) + BigInt::from(29);

        for _ in 0..HEUGCD_MAX_TRIES {
            if let Some(h) = Self::heugcd_attempt(&f, &g, var, &xi, depth) {
                return Some(h.scale(&c_rat));
            }
            xi = Self::next_xi(&xi);
        }
        None
    }

    /// Next evaluation point: `73794 · ξ · ξ^(1/4) / 27011` (grows like
    /// `ξ^1.25`, the schedule used by the classical implementations).
    fn next_xi(xi: &BigInt) -> BigInt {
        let root4 = xi.sqrt().sqrt().max(BigInt::from(2));
        (BigInt::from(73794) * xi * root4) / BigInt::from(27011)
    }

    /// One evaluation/interpolation round of the heuristic GCD for primitive
    /// inputs `f`, `g` in variable `var` at the point `xi`.  Returns the
    /// verified GCD candidate or `None`.
    fn heugcd_attempt(f: &Self, g: &Self, var: usize, xi: &BigInt, depth: usize) -> Option<Self> {
        let xi_rat = Ratio::from_integer(xi.clone());
        let ff = f.substitute(var, &xi_rat);
        let gg = g.substitute(var, &xi_rat);
        if ff.is_zero() || gg.is_zero() {
            return None;
        }
        let h = Self::heugcd_z(&ff, &gg, depth + 1)?;
        let h = Self::interpolate_xi(&h, xi, var);
        if h.is_zero() {
            return None;
        }
        // Make primitive (the content of the true GCD is 1 here).
        let content = h.integer_content();
        if content.is_zero() {
            return None;
        }
        let h = h.scale(&Ratio::new(BigInt::one(), content));
        if f.div_exact(&h).is_some() && g.div_exact(&h).is_some() {
            Some(h)
        } else {
            None
        }
    }

    /// Reconstruct a polynomial in variable `var` from its value `h` at
    /// `xi` by symmetric ξ-adic expansion: `h = Σᵢ gᵢ · ξⁱ` with the
    /// coefficients of each `gᵢ` in `(-ξ/2, ξ/2]`.
    fn interpolate_xi(h: &Self, xi: &BigInt, var: usize) -> Self {
        let nv = h.num_vars + 1;
        let mut result = Self::zero(nv);
        let mut rest = h.clone();
        let mut i: u32 = 0;
        let xi_rat = Ratio::from_integer(xi.clone());
        while !rest.is_zero() {
            // With exact integer arithmetic every digit step divides the
            // remaining magnitude by ξ, so this terminates; the cap only
            // guards against a non-integer `h` slipping through.
            if i > 4096 || rest.terms().any(|(_, c)| !c.is_integer()) {
                return Self::zero(nv);
            }
            let digit = rest.map_coeffs(|c| Ratio::from_integer(symmetric_mod(c.numer(), xi)));
            for (exp, c) in digit.terms() {
                let mut e = Vec::with_capacity(nv);
                e.extend_from_slice(&exp[..var]);
                e.push(i);
                e.extend_from_slice(&exp[var..]);
                result.insert_term(e, c.clone());
            }
            rest = rest.sub(&digit).map_coeffs(|c| c / &xi_rat);
            i += 1;
        }
        result.prune();
        result
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Multivariate division
// ═══════════════════════════════════════════════════════════════════════════

impl<O: MonomialOrd> MultiPoly<O> {
    /// Reduce this polynomial modulo a set of divisors.
    /// Returns the remainder after multivariate division.
    ///
    /// # Panics
    ///
    /// As [`mul_monomial`](Self::mul_monomial): panics if a divisor has a
    /// different number of variables, or if an exponent of an intermediate
    /// product `(lt/lt(d))·d` overflows `u32` (it wrapped around in release
    /// builds before 0.29).
    pub fn reduce(&self, divisors: &[&MultiPoly<O>]) -> MultiPoly<O> {
        if self.is_zero() || divisors.is_empty() {
            return self.clone();
        }

        let mut remainder = MultiPoly::zero(self.num_vars);
        let mut p = self.clone();

        while let Some((lt_exp, lt_coeff)) = p.leading_term() {
            let mut divided = false;
            let lt_exp = lt_exp.to_vec();
            let lt_coeff = lt_coeff.clone();

            for divisor in divisors {
                // A zero divisor has no leading term and cannot divide anything.
                let Some((div_lt_exp, div_lt_coeff)) = divisor.leading_term() else {
                    continue;
                };

                if let Some(quot_exp) = monomial_div(div_lt_exp, &lt_exp) {
                    // Can divide: subtract (lt/div_lt) * divisor from p
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
    /// divisor, and a divisor with a different number of variables).  Uses
    /// the multivariate division algorithm with one divisor, for which the
    /// remainder vanishes iff the division is exact.  (An exact quotient
    /// never needs an exponent above those of `self`, so an exponent
    /// overflow along the way also means "does not divide".)
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
        if self.num_vars != divisor.num_vars {
            return None;
        }
        let (div_lt_exp, div_lt_coeff) = divisor.leading_term()?;
        let div_lt_exp = div_lt_exp.to_vec();
        let div_lt_coeff = div_lt_coeff.clone();

        let mut quotient = MultiPoly::zero(self.num_vars);
        let mut p = self.clone();
        while let Some((lt_exp, lt_coeff)) = p.leading_term() {
            let quot_exp = monomial_div(&div_lt_exp, lt_exp)?;
            let quot_coeff = lt_coeff / &div_lt_coeff;
            let subtrahend = divisor.try_mul_monomial(&quot_coeff, &quot_exp)?;
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
///
/// # Panics
///
/// Panics if `f` and `g` have different numbers of variables (as
/// [`MultiPoly::sub`] does), or if an exponent of the result exceeds
/// `u32::MAX` (as [`MultiPoly::mul_monomial`] does): with `L` the lcm of
/// the leading monomials `F` of `f` and `G` of `g`, some term of `f` has
/// `eᵢ + Lᵢ − Fᵢ > u32::MAX` in a variable `i`, or likewise for `g`.
/// Before 0.29 the exponent wrapped around in release builds.
/// [`try_s_polynomial`] returns `None` in either case.
pub fn s_polynomial<O: MonomialOrd>(f: &MultiPoly<O>, g: &MultiPoly<O>) -> MultiPoly<O> {
    assert_eq!(
        f.num_vars(),
        g.num_vars(),
        "s_polynomial: incompatible variable counts"
    );
    // S(f, 0) = S(0, g) = 0: a zero operand has no leading term.
    let Some([(quot_f, coeff_f), (quot_g, coeff_g)]) = s_polynomial_cofactors(f, g) else {
        return MultiPoly::zero(f.num_vars());
    };

    let scaled_f = f.mul_monomial(&coeff_f, &quot_f);
    let scaled_g = g.mul_monomial(&coeff_g, &quot_g);

    scaled_f.sub(&scaled_g)
}

/// The S-polynomial of `f` and `g`, as [`s_polynomial`]; `None` if `f`
/// and `g` have different numbers of variables or an exponent of the
/// result would exceed `u32::MAX`.
///
/// ```
/// use symplex::multipoly::{MultiPoly, try_s_polynomial};
///
/// let [x, y]: [MultiPoly; 2] = [MultiPoly::var(2, 0), MultiPoly::var(2, 1)];
/// // S(x² + y, xy) = y·(x² + y) − x·(xy) = y²
/// let s = try_s_polynomial(&x.mul(&x).add(&y), &x.mul(&y));
/// assert_eq!(s, Some(y.mul(&y)));
/// assert_eq!(try_s_polynomial(&x, &MultiPoly::var(3, 0)), None);
/// ```
pub fn try_s_polynomial<O: MonomialOrd>(
    f: &MultiPoly<O>,
    g: &MultiPoly<O>,
) -> Option<MultiPoly<O>> {
    if f.num_vars() != g.num_vars() {
        return None;
    }
    let Some([(quot_f, coeff_f), (quot_g, coeff_g)]) = s_polynomial_cofactors(f, g) else {
        return Some(MultiPoly::zero(f.num_vars()));
    };
    let scaled_f = f.try_mul_monomial(&coeff_f, &quot_f)?;
    let scaled_g = g.try_mul_monomial(&coeff_g, &quot_g)?;
    scaled_f.try_sub(&scaled_g)
}

/// The monomial cofactors `(L/F, 1/lc(f))` and `(L/G, 1/lc(g))` of the
/// S-polynomial of `f` and `g` (`L` the lcm of their leading monomials `F`
/// and `G`); `None` if either is zero (it has no leading term, and the
/// S-polynomial is 0).
fn s_polynomial_cofactors<O: MonomialOrd>(
    f: &MultiPoly<O>,
    g: &MultiPoly<O>,
) -> Option<[(Vec<u32>, Ratio<BigInt>); 2]> {
    let (Some((lm_f, lc_f)), Some((lm_g, lc_g))) = (f.leading_term(), g.leading_term()) else {
        return None;
    };

    let lcm = monomial_lcm(lm_f, lm_g);

    // LCM/LT(f) * f - LCM/LT(g) * g.  Each leading monomial divides the lcm
    // component-wise (lcm = max), so the quotients are plain differences.
    let quot_f: Vec<u32> = lcm.iter().zip(lm_f).map(|(&l, &e)| l - e).collect();
    let quot_g: Vec<u32> = lcm.iter().zip(lm_g).map(|(&l, &e)| l - e).collect();

    let coeff_f = Ratio::one() / lc_f;
    let coeff_g = Ratio::one() / lc_g;

    Some([(quot_f, coeff_f), (quot_g, coeff_g)])
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
        assert_eq!(c.eval(&[rat(0), rat(0)]).unwrap(), rat(42));
    }

    // ── heuristic GCD ──────────────────────────────────────────────────────────────

    fn vars2() -> (MultiPoly<GrevLex>, MultiPoly<GrevLex>) {
        (MultiPoly::var(2, 0), MultiPoly::var(2, 1))
    }

    fn pow(p: &MultiPoly<GrevLex>, n: u32) -> MultiPoly<GrevLex> {
        let mut acc = MultiPoly::from_int(p.num_vars(), 1);
        for _ in 0..n {
            acc = acc.mul(p);
        }
        acc
    }

    #[test]
    fn gcd_coprime_is_one() {
        let (x, y) = vars2();
        let f = x.mul(&x).add(&y); // x² + y
        let g = x.add(&y).add(&MultiPoly::from_int(2, 1)); // x + y + 1
        assert_eq!(MultiPoly::gcd(&f, &g), MultiPoly::from_int(2, 1));
        assert_eq!(MultiPoly::gcd(&x, &y), MultiPoly::from_int(2, 1));
    }

    #[test]
    fn gcd_shared_linear_factor() {
        let (x, y) = vars2();
        let s = x.add(&y);
        let d = x.sub(&y);
        let f = s.mul(&d); // x² − y²
        let g = s.mul(&s); // (x + y)²
        assert_eq!(MultiPoly::gcd(&f, &g), s);
        // Negated input: sign is normalised away.
        assert_eq!(MultiPoly::gcd(&f.neg(), &g), s);
    }

    #[test]
    fn gcd_three_variables() {
        let x: MultiPoly<GrevLex> = MultiPoly::var(3, 0);
        let y: MultiPoly<GrevLex> = MultiPoly::var(3, 1);
        let z: MultiPoly<GrevLex> = MultiPoly::var(3, 2);
        // h = xy + z + 1, f = h·(x − z), g = h·(y² + x)
        let h = x.mul(&y).add(&z).add(&MultiPoly::from_int(3, 1));
        let f = h.mul(&x.sub(&z));
        let g = h.mul(&y.mul(&y).add(&x));
        assert_eq!(MultiPoly::gcd(&f, &g), h);
        assert_eq!(MultiPoly::gcd(&g, &f), h);
    }

    #[test]
    fn gcd_zero_handling() {
        let (x, y) = vars2();
        let f = x.mul(&y).scale(&rat(-4)); // −4xy
        let z: MultiPoly<GrevLex> = MultiPoly::zero(2);
        assert!(MultiPoly::gcd(&z, &z).is_zero());
        // gcd(f, 0) is f with a positive leading coefficient.
        assert_eq!(MultiPoly::gcd(&f, &z), x.mul(&y).scale(&rat(4)));
        assert_eq!(MultiPoly::gcd(&z, &f), x.mul(&y).scale(&rat(4)));
    }

    #[test]
    fn gcd_includes_integer_content() {
        let (x, _y) = vars2();
        let f = x.scale(&rat(6)); // 6x
        let g = x.mul(&x).scale(&rat(4)); // 4x²
        assert_eq!(MultiPoly::gcd(&f, &g), x.scale(&rat(2)));
        // Constants.
        let twelve: MultiPoly<GrevLex> = MultiPoly::from_int(2, 12);
        assert_eq!(
            MultiPoly::gcd(&twelve, &MultiPoly::from_int(2, 18)),
            MultiPoly::from_int(2, 6)
        );
    }

    #[test]
    fn gcd_clears_rational_denominators() {
        let (x, y) = vars2();
        let s = x.add(&y);
        let half = Ratio::new(BigInt::from(1), BigInt::from(2));
        let third = Ratio::new(BigInt::from(1), BigInt::from(3));
        let f = s.mul(&x).scale(&half); // (x + y)x / 2
        let g = s.mul(&y).scale(&third); // (x + y)y / 3
        let h = MultiPoly::gcd(&f, &g);
        assert!(f.div_exact(&h).is_some() && g.div_exact(&h).is_some());
        assert_eq!(h, s);
    }

    #[test]
    fn gcd_large_coefficients() {
        let (x, y) = vars2();
        let big = |s: &str| Ratio::from_integer(s.parse::<BigInt>().unwrap());
        // h = 123456789012345678901234567890·x + 987654321098765432109876543210·y + 1
        let h = x
            .scale(&big("123456789012345678901234567890"))
            .add(&y.scale(&big("987654321098765432109876543210")))
            .add(&MultiPoly::from_int(2, 1));
        let f = h.mul(&x.add(&MultiPoly::from_int(2, 7)));
        let g = h.mul(&y.sub(&x.scale(&big("5555555555555555555"))));
        assert_eq!(MultiPoly::gcd(&f, &g), h);
    }

    #[test]
    fn gcd_first_evaluation_point_fails_then_retry_succeeds() {
        // gcd = (x + 1)⁸ has a coefficient 70, but the inputs have max-norms
        // 28 and 112, so the first evaluation point ξ = 2·28 + 29 = 85 cannot
        // represent 70 as a symmetric digit: the first attempt must fail and
        // the next ξ must recover the answer.
        let x: MultiPoly<GrevLex> = MultiPoly::var(1, 0);
        let one = MultiPoly::from_int(1, 1);
        let h = pow(&x.add(&one), 8);
        let f = h.mul(&x.sub(&one)); // norm 28
        let g = h.mul(&x.mul(&x).add(&one)); // norm 112
        assert_eq!(f.max_norm(), BigInt::from(28));
        assert_eq!(g.max_norm(), BigInt::from(112));
        let xi0 = BigInt::from(85);
        assert!(MultiPoly::heugcd_attempt(&f, &g, 0, &xi0, 0).is_none());
        let xi1 = MultiPoly::<GrevLex>::next_xi(&xi0);
        assert!(xi1 > BigInt::from(140), "next ξ = {xi1}");
        assert_eq!(
            MultiPoly::heugcd_attempt(&f, &g, 0, &xi1, 0),
            Some(h.clone())
        );
        assert_eq!(MultiPoly::gcd(&f, &g), h);
    }

    #[test]
    fn gcd_never_wrong_on_random_products() {
        // Deterministic pseudo-random small polynomials: gcd(h·a, h·b) must
        // be divisible by h and divide both products.
        let (x, y) = vars2();
        let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % 7) as i64 - 3
        };
        let mut rand_poly = || {
            let mut p = MultiPoly::zero(2);
            for ex in 0..3u32 {
                for ey in 0..3u32 {
                    let c = next();
                    if c != 0 {
                        p = p.add(&MultiPoly::monomial(rat(c), vec![ex, ey]));
                    }
                }
            }
            if p.is_zero() { x.add(&y) } else { p }
        };
        for _ in 0..12 {
            let h = rand_poly();
            let a = rand_poly();
            let b = rand_poly();
            let f = h.mul(&a);
            let g = h.mul(&b);
            let d = MultiPoly::gcd(&f, &g);
            assert!(f.div_exact(&d).is_some(), "gcd does not divide f");
            assert!(g.div_exact(&d).is_some(), "gcd does not divide g");
            assert!(d.div_exact(&h).is_some(), "gcd {d} misses factor {h}");
        }
    }

    #[test]
    fn lcm_of_monomials() {
        let (x, y) = vars2();
        let f = x.mul(&y);
        let g = y.mul(&y);
        assert_eq!(MultiPoly::lcm(&f, &g), x.mul(&y).mul(&y));
        assert!(MultiPoly::lcm(&f, &MultiPoly::zero(2)).is_zero());
    }

    #[test]
    fn from_terms_and_map_coeffs() {
        let p: MultiPoly<GrevLex> =
            MultiPoly::from_terms(2, vec![(vec![1, 0], rat(2)), (vec![1, 0], rat(-2))]).unwrap();
        assert!(p.is_zero());
        let q: MultiPoly<GrevLex> = MultiPoly::from_terms(2, vec![(vec![2, 1], rat(3))]).unwrap();
        assert_eq!(q.coeff(&[2, 1]), Some(&rat(3)));
        assert_eq!(q.coeff(&[2]), None);
        let doubled = q.map_coeffs(|c| c * rat(2));
        assert_eq!(doubled.coeff(&[2, 1]), Some(&rat(6)));
        let killed = q.map_coeffs(|_| rat(0));
        assert!(killed.is_zero());
    }

    #[test]
    fn integer_content_and_clear_denominators() {
        let (x, y) = vars2();
        let f = x.scale(&rat(6)).add(&y.scale(&rat(9)));
        assert_eq!(f.integer_content(), BigInt::from(3));
        let g = x
            .scale(&Ratio::new(BigInt::from(1), BigInt::from(2)))
            .add(&y.scale(&Ratio::new(BigInt::from(2), BigInt::from(3))));
        let (d, gz) = g.clear_denominators();
        assert_eq!(d, BigInt::from(6));
        assert_eq!(gz, x.scale(&rat(3)).add(&y.scale(&rat(4))));
    }
}
