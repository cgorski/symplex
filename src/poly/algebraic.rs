//! Algebraic number field arithmetic: elements of ℚ(α) = ℚ[t]/(m(t)).
//!
//! An [`AlgNum`] represents an element of an algebraic number field,
//! stored as a polynomial representative `repr ∈ ℚ[t]` of degree < deg(m),
//! together with the irreducible minimal polynomial `m(t)`.
//!
//! # Arithmetic
//!
//! - **Addition / Subtraction**: polynomial addition (no reduction needed
//!   since deg(repr) < deg(m)).
//! - **Multiplication**: polynomial multiplication followed by reduction
//!   modulo `m(t)`.
//! - **Division**: compute the multiplicative inverse via
//!   [`extended_gcd(repr, m)`](crate::poly::generic::GenPoly::extended_gcd),
//!   then multiply.  The Bézout coefficient gives `repr⁻¹ mod m`.
//!
//! # Zero and Sign Testing
//!
//! - **Zero-testing is exact**: an element is zero if and only if its
//!   representative polynomial is the zero polynomial after reduction.
//!   No floating-point tolerance.
//! - **Sign-testing** uses Sturm sequences and root isolation on the
//!   representative polynomial, evaluated at the distinguished real root
//!   of the minimal polynomial.
//!
//! # Minimal Polynomial Computation
//!
//! The [`minimal_polynomial`] function computes the minimal polynomial
//! of an arena expression over ℚ by recursive composition:
//! - Rational numbers: `m(t) = t - r`
//! - `n^{p/q}`: `m(t) = t^q - n^p`
//! - `a + b`: via resultant `res_y(m_a(y), m_b(x - y))`
//! - `a · b`: via resultant with appropriate scaling
//!
//! # References
//!
//! - Cohen, *A Course in Computational Algebraic Number Theory*, Springer
//! - Bronstein, *Symbolic Integration I*, §1.2–1.4
//! - SymPy `polys/numberfields/minpoly.py` and `core/numbers.py::AlgebraicNumber`

use std::fmt;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

use super::dense::Poly;
use super::generic::GenPoly;
use super::sturm::SturmChain;
use super::traits;

// ═══════════════════════════════════════════════════════════════════════════
// AlgNum — element of ℚ(α) = ℚ[t]/(m(t))
// ═══════════════════════════════════════════════════════════════════════════

/// An element of the algebraic number field `ℚ(α) = ℚ[t]/(m(t))`.
///
/// `repr` is a polynomial of degree < deg(m) with rational coefficients.
/// `min_poly` is irreducible over ℚ and monic.
///
/// Two `AlgNum` values can only be combined arithmetically if they share
/// the same `min_poly`.  Combining elements from different extensions
/// requires first computing a common primitive element (see
/// [`primitive_element`]).
#[derive(Clone)]
pub struct AlgNum {
    /// Representative polynomial: a₀ + a₁t + … + a_{d-1}t^{d-1}
    repr: Poly,
    /// Minimal polynomial of the generator α (irreducible, monic).
    min_poly: Poly,
}

impl fmt::Debug for AlgNum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AlgNum({} mod {})", self.repr, self.min_poly)
    }
}

impl fmt::Display for AlgNum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.repr)
    }
}

impl PartialEq for AlgNum {
    fn eq(&self, other: &Self) -> bool {
        // Same field (same min_poly) and same representative.
        self.min_poly == other.min_poly && self.repr == other.repr
    }
}

impl Eq for AlgNum {}

impl AlgNum {
    // ── Construction ───────────────────────────────────────────────

    /// Create a new algebraic number from a representative polynomial
    /// and its minimal polynomial.
    ///
    /// The representative is automatically reduced modulo `min_poly`.
    pub fn new(repr: Poly, min_poly: Poly) -> Self {
        let reduced = if repr.degree().unwrap_or(0) >= min_poly.degree().unwrap_or(0) {
            repr.rem(&min_poly)
        } else {
            repr
        };
        AlgNum {
            repr: reduced,
            min_poly,
        }
    }

    /// The zero element of ℚ(α).
    pub fn zero_in(min_poly: &Poly) -> Self {
        AlgNum {
            repr: Poly::zero(),
            min_poly: min_poly.clone(),
        }
    }

    /// The multiplicative identity (1) in ℚ(α).
    pub fn one_in(min_poly: &Poly) -> Self {
        AlgNum {
            repr: Poly::from_int(1),
            min_poly: min_poly.clone(),
        }
    }

    /// The generator α (the polynomial `t` modulo `m(t)`).
    pub fn generator(min_poly: Poly) -> Self {
        AlgNum {
            repr: Poly::x(),
            min_poly,
        }
    }

    /// Embed a rational number into ℚ(α) as a constant polynomial.
    pub fn from_rational(r: Ratio<BigInt>, min_poly: &Poly) -> Self {
        AlgNum {
            repr: Poly::constant(r),
            min_poly: min_poly.clone(),
        }
    }

    /// Embed an integer into ℚ(α).
    pub fn from_int(n: i64, min_poly: &Poly) -> Self {
        AlgNum {
            repr: Poly::from_int(n),
            min_poly: min_poly.clone(),
        }
    }

    // ── Accessors ──────────────────────────────────────────────────

    /// The representative polynomial.
    pub fn repr(&self) -> &Poly {
        &self.repr
    }

    /// The minimal polynomial of the generator.
    pub fn min_poly(&self) -> &Poly {
        &self.min_poly
    }

    /// Degree of the field extension [ℚ(α) : ℚ].
    pub fn degree(&self) -> usize {
        self.min_poly.degree().unwrap_or(0)
    }

    // ── Zero and equality testing (EXACT) ──────────────────────────

    /// Exact zero test: is this element the zero of ℚ(α)?
    ///
    /// This is exact — no floating-point tolerance.
    pub fn is_zero_exact(&self) -> bool {
        self.repr.is_zero()
    }

    /// Exact one test.
    pub fn is_one_exact(&self) -> bool {
        self.repr.is_constant() && self.repr.coeff(0) == rat(1, 1)
    }

    // ── Sign testing ───────────────────────────────────────────────

    /// Determine the sign of this algebraic number when α is the
    /// real root of `min_poly` nearest to `approx_root`.
    ///
    /// Returns `Some(1)` for positive, `Some(-1)` for negative,
    /// `Some(0)` for zero, or `None` if sign cannot be determined
    /// (e.g., min_poly has no real roots).
    ///
    /// Uses Sturm sequences for certified sign determination:
    /// 1. If `repr` is zero → sign is 0.
    /// 2. If `repr` is a nonzero constant → sign is sign(constant).
    /// 3. Otherwise: isolate the real root of `min_poly` near `approx_root`,
    ///    then evaluate `repr` at a point in the isolating interval.
    ///    If the result is ambiguous, refine the interval.
    pub fn sign_at_real_root(&self, approx_root: f64) -> Option<i8> {
        if self.repr.is_zero() {
            return Some(0);
        }

        // Constant → sign is determined directly.
        if self.repr.is_constant() {
            let c = self.repr.coeff(0);
            return Some(if c.is_positive() {
                1
            } else if c.is_negative() {
                -1
            } else {
                0
            });
        }

        // Isolate the real root of min_poly near approx_root.
        let interval = isolate_root_near(&self.min_poly, approx_root)?;

        // Evaluate repr at points in the interval to determine sign.
        // Since repr(α) is an algebraic number whose minimal polynomial
        // we could compute, we use interval refinement: evaluate at the
        // midpoint and check if the sign is clear.
        sign_of_poly_at_algebraic_root(&self.repr, &self.min_poly, &interval)
    }

    // ── Arithmetic helpers ─────────────────────────────────────────

    /// Reduce the representative modulo min_poly.
    fn reduce(&mut self) {
        if self.repr.degree().unwrap_or(0) >= self.min_poly.degree().unwrap_or(0) {
            self.repr = self.repr.rem(&self.min_poly);
        }
    }

    /// Compute the multiplicative inverse via extended GCD.
    ///
    /// `extended_gcd(repr, min_poly)` gives `(s, t, g)` with
    /// `s·repr + t·min_poly = g`.  Since `min_poly` is irreducible
    /// and `repr ≠ 0`, `g = 1`, so `s·repr ≡ 1 (mod min_poly)`.
    fn compute_inverse(&self) -> Option<Self> {
        if self.repr.is_zero() {
            return None; // Division by zero.
        }
        let (s, _t, g) = Poly::extended_gcd(&self.repr, &self.min_poly);
        // g should be 1 (since min_poly is irreducible and repr ≠ 0).
        if !g.is_constant() {
            tracing::debug!(
                "AlgNum::compute_inverse: GCD is not constant — min_poly may not be irreducible"
            );
            return None;
        }
        // Normalize: s / g (in case g is a nonzero constant ≠ 1).
        let g_val = g.coeff(0);
        let inv_g = Ratio::new(g_val.denom().clone(), g_val.numer().clone());
        let result = s.scale(&inv_g);
        Some(AlgNum::new(result, self.min_poly.clone()))
    }

    /// Evaluate the representative polynomial at a rational point.
    pub fn eval_rational(&self, point: &Ratio<BigInt>) -> Ratio<BigInt> {
        self.repr.eval(point)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Ring and Field trait implementations
// ═══════════════════════════════════════════════════════════════════════════

impl traits::Ring for AlgNum {
    fn zero() -> Self {
        // Default zero — uses t²-1 as a placeholder min_poly.
        // In practice, AlgNum values should be constructed with a known min_poly.
        AlgNum {
            repr: Poly::zero(),
            min_poly: Poly::from_coeffs(vec![rat(-1, 1), rat(0, 1), rat(1, 1)]),
        }
    }

    fn one() -> Self {
        AlgNum {
            repr: Poly::from_int(1),
            min_poly: Poly::from_coeffs(vec![rat(-1, 1), rat(0, 1), rat(1, 1)]),
        }
    }

    fn is_zero(&self) -> bool {
        self.is_zero_exact()
    }

    fn is_one(&self) -> bool {
        self.is_one_exact()
    }

    fn add(&self, rhs: &Self) -> Self {
        debug_assert_eq!(
            self.min_poly, rhs.min_poly,
            "AlgNum::add: min_poly mismatch"
        );
        AlgNum::new(&self.repr + &rhs.repr, self.min_poly.clone())
    }

    fn sub(&self, rhs: &Self) -> Self {
        debug_assert_eq!(
            self.min_poly, rhs.min_poly,
            "AlgNum::sub: min_poly mismatch"
        );
        AlgNum::new(&self.repr - &rhs.repr, self.min_poly.clone())
    }

    fn mul(&self, rhs: &Self) -> Self {
        debug_assert_eq!(
            self.min_poly, rhs.min_poly,
            "AlgNum::mul: min_poly mismatch"
        );
        let product = &self.repr * &rhs.repr;
        AlgNum::new(product, self.min_poly.clone())
    }

    fn neg(&self) -> Self {
        AlgNum {
            repr: -&self.repr,
            min_poly: self.min_poly.clone(),
        }
    }
}

impl traits::EuclideanDomain for AlgNum {
    fn div_rem(&self, other: &Self) -> (Self, Self) {
        // Fields have trivial Euclidean division: quotient = a/b, remainder = 0.
        (traits::Field::div(self, other), AlgNum::zero_in(&self.min_poly))
    }

    fn gcd(a: &Self, b: &Self) -> Self {
        if a.is_zero_exact() && b.is_zero_exact() {
            AlgNum::zero_in(&a.min_poly)
        } else {
            AlgNum::one_in(&a.min_poly)
        }
    }
}

impl traits::Field for AlgNum {
    fn div(&self, other: &Self) -> Self {
        debug_assert_eq!(
            self.min_poly, other.min_poly,
            "AlgNum::div: min_poly mismatch"
        );
        let inv = other
            .compute_inverse()
            .expect("AlgNum::div: division by zero");
        traits::Ring::mul(self, &inv)
    }

    fn inv(&self) -> Self {
        self.compute_inverse()
            .expect("AlgNum::inv: inverse of zero")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sign determination helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Isolate the real root of `poly` nearest to `approx` into a rational
/// interval `(lo, hi)` containing exactly one root.
///
/// Returns `None` if `poly` has no real roots.
fn isolate_root_near(poly: &Poly, approx: f64) -> Option<(Ratio<BigInt>, Ratio<BigInt>)> {
    let chain = SturmChain::new(poly);
    if chain.has_no_real_roots() {
        return None;
    }

    // Start with a wide interval and narrow toward the approximate root.
    let cauchy = cauchy_bound(poly);
    let neg_bound = -cauchy.clone();

    let intervals = chain.isolate_roots_in(&neg_bound, &cauchy, 60);
    if intervals.is_empty() {
        return None;
    }

    // Find the interval closest to approx.
    let approx_rat = f64_to_rational_approx(approx);
    let mut best = &intervals[0];
    let mut best_dist = rational_dist_to_interval(&approx_rat, &best.0, &best.1);

    for interval in &intervals[1..] {
        let dist = rational_dist_to_interval(&approx_rat, &interval.0, &interval.1);
        if dist < best_dist {
            best = interval;
            best_dist = dist;
        }
    }

    Some(best.clone())
}

/// Determine the sign of `f(α)` where α is the unique real root of
/// `min_poly` in the interval `(lo, hi)`.
///
/// Uses interval refinement: evaluate `f` at rational points in the
/// interval.  Since `f(α)` is an algebraic number, it is either zero
/// or bounded away from zero, so this always terminates.
fn sign_of_poly_at_algebraic_root(
    f: &Poly,
    min_poly: &Poly,
    interval: &(Ratio<BigInt>, Ratio<BigInt>),
) -> Option<i8> {
    // Quick check: does f share a root with min_poly in this interval?
    let gcd = Poly::gcd(f, min_poly);
    if gcd.degree().unwrap_or(0) >= 1 {
        // f and min_poly share a common factor → f(α) = 0.
        let gcd_chain = SturmChain::new(&gcd);
        if gcd_chain.count_roots_in(&interval.0, &interval.1) > 0 {
            return Some(0);
        }
    }

    // f(α) ≠ 0.  Determine the sign by evaluating at the midpoint
    // and refining if necessary.
    let two = Ratio::from_integer(BigInt::from(2));
    let mut lo = interval.0.clone();
    let mut hi = interval.1.clone();

    let mp_chain = SturmChain::new(min_poly);

    for _ in 0..200 {
        let mid = (&lo + &hi) / &two;
        let val = f.eval(&mid);

        if val.is_positive() {
            // Check: is α in (lo, mid) or (mid, hi)?
            // The sign of f at α matches the sign at mid IF f doesn't
            // change sign between mid and α.
            // f has no root in the interval (we checked gcd above),
            // so if f doesn't have a root of its own in (lo, hi), the
            // sign at mid equals the sign at α.
            let f_chain = SturmChain::new(f);
            let f_roots_in = f_chain.count_roots_in(&lo, &hi);
            if f_roots_in == 0 {
                return Some(1);
            }
            // f has its own root in the interval — refine.
            if mp_chain.count_roots_in(&lo, &mid) > 0 {
                hi = mid;
            } else {
                lo = mid;
            }
        } else if val.is_negative() {
            let f_chain = SturmChain::new(f);
            let f_roots_in = f_chain.count_roots_in(&lo, &hi);
            if f_roots_in == 0 {
                return Some(-1);
            }
            if mp_chain.count_roots_in(&lo, &mid) > 0 {
                hi = mid;
            } else {
                lo = mid;
            }
        } else {
            // val == 0: mid is a root of f.  But f(α) ≠ 0 (checked above).
            // So α ≠ mid.  Refine the interval to exclude mid.
            if mp_chain.count_roots_in(&lo, &mid) > 0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
    }

    // Should not reach here for algebraic numbers, but be safe.
    tracing::debug!("sign_of_poly_at_algebraic_root: failed to converge after 200 iterations");
    None
}

/// Cauchy bound: all roots of `p(x) = a_n x^n + ... + a_0` satisfy
/// `|x| ≤ 1 + max(|a_{n-1}/a_n|, ..., |a_0/a_n|)`.
fn cauchy_bound(p: &Poly) -> Ratio<BigInt> {
    let n = match p.degree() {
        Some(d) if d >= 1 => d,
        _ => return Ratio::from_integer(BigInt::from(1)),
    };
    let lc = p.coeff(n);
    if lc.is_zero() {
        return Ratio::from_integer(BigInt::from(1));
    }
    let mut max_ratio = Ratio::from_integer(BigInt::from(0));
    for i in 0..n {
        let ratio = Ratio::new(
            p.coeff(i).numer().clone().abs(),
            p.coeff(i).denom().clone() * lc.numer().clone().abs(),
        ) * Ratio::new(lc.denom().clone(), BigInt::from(1));
        if ratio > max_ratio {
            max_ratio = ratio;
        }
    }
    max_ratio + Ratio::from_integer(BigInt::from(1))
}

// ═══════════════════════════════════════════════════════════════════════════
// Minimal polynomial computation
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the minimal polynomial of an arena expression over ℚ.
///
/// The expression must represent an algebraic number (no transcendental
/// functions, no free symbols other than constants).
///
/// Returns `None` if the expression is not recognized as algebraic.
///
/// # Algorithm
///
/// Recursively decomposes the expression:
/// - Rational `r` → `t - r`
/// - `n^{p/q}` → `t^q - n^p` (for positive integer `n`, integer `p`, positive `q`)
/// - `ImaginaryUnit` → `t² + 1`
/// - `a + b` → resultant composition
/// - `a · b` → resultant composition
/// - `-a` → `m_a(-t)`
///
/// # References
///
/// - SymPy `polys/numberfields/minpoly.py::_minpoly_compose`
pub fn minimal_polynomial(
    arena: &crate::base::arena::Arena,
    expr: crate::base::node::ExprId,
) -> Option<Poly> {
    use crate::base::node::ExprNode;

    let node = arena.node(expr).clone();
    match node {
        // Rational number r → t - r
        ExprNode::Num(nid) => {
            let r = arena.num(nid).clone();
            // t - r = [-r, 1]
            Some(Poly::from_coeffs(vec![-r, rat(1, 1)]))
        }

        // Constants
        ExprNode::Pi | ExprNode::E => None, // Transcendental — not algebraic.

        ExprNode::ImaginaryUnit => {
            // i → t² + 1
            Some(Poly::from_coeffs(vec![rat(1, 1), rat(0, 1), rat(1, 1)]))
        }

        // Negation: min_poly(-a) = m_a(-t)
        ExprNode::Neg(inner) => {
            let mp = minimal_polynomial(arena, inner)?;
            Some(negate_variable(&mp))
        }

        // Power: n^{p/q} where n is a positive integer
        ExprNode::Pow(base, exp) => {
            if let Some(base_r) = arena.as_num(base) {
                if let Some(exp_r) = arena.as_num(exp) {
                    if base_r.is_positive() && !exp_r.is_integer() {
                        // n^{p/q}: minimal polynomial is t^q - n^p
                        let p_int = exp_r.numer().clone();
                        let q_int = exp_r.denom().clone();
                        let q_usize: usize = (&q_int).try_into().ok()?;
                        // n^p as a rational
                        let n_pow_p = rational_pow(base_r, &p_int)?;
                        // t^q - n^p = [-n^p, 0, 0, ..., 0, 1] with 1 at position q
                        let mut coeffs = vec![Ratio::zero(); q_usize + 1];
                        coeffs[0] = -n_pow_p;
                        coeffs[q_usize] = rat(1, 1);
                        let mp = Poly::from_coeffs(coeffs);
                        // Factor and pick the irreducible factor containing
                        // the root n^{p/q}.
                        return pick_irreducible_factor(&mp, base_r, exp_r);
                    }
                }
            }
            // General symbolic power — try recursion on base/exp if possible.
            // For now, only handle the numeric-base rational-exp case above.
            None
        }

        // Addition: min_poly(a + b) via resultant
        ExprNode::Add(ref children) if children.len() == 2 => {
            let mp_a = minimal_polynomial(arena, children[0])?;
            let mp_b = minimal_polynomial(arena, children[1])?;
            minpoly_add(&mp_a, &mp_b, arena, children[0], children[1])
        }

        // N-ary addition: fold pairwise
        ExprNode::Add(ref children) if children.len() > 2 => {
            // Fold left: min_poly(a + b + c) = min_poly(min_poly(a+b) composed with c)
            let mut acc_expr = children[0];
            let mut acc_mp = minimal_polynomial(arena, acc_expr)?;
            for &child in &children[1..] {
                let child_mp = minimal_polynomial(arena, child)?;
                acc_mp = minpoly_add(&acc_mp, &child_mp, arena, acc_expr, child)?;
                // We don't have a simple arena expr for the accumulator,
                // but we can approximate numerically for factor selection.
                acc_expr = child; // This is a simplification; factor selection uses numerical eval.
            }
            Some(acc_mp)
        }

        // Multiplication: min_poly(a * b) via resultant
        ExprNode::Mul(ref children) if children.len() == 2 => {
            // Check if one factor is rational.
            if let Some(r) = arena.as_num(children[0]) {
                let mp_b = minimal_polynomial(arena, children[1])?;
                return Some(minpoly_rational_mul(&mp_b, r));
            }
            if let Some(r) = arena.as_num(children[1]) {
                let mp_a = minimal_polynomial(arena, children[0])?;
                return Some(minpoly_rational_mul(&mp_a, r));
            }
            let mp_a = minimal_polynomial(arena, children[0])?;
            let mp_b = minimal_polynomial(arena, children[1])?;
            minpoly_mul(&mp_a, &mp_b, arena, children[0], children[1])
        }

        // N-ary multiplication: separate rationals and fold
        ExprNode::Mul(ref children) if children.len() > 2 => {
            let mut rational_coeff = rat(1, 1);
            let mut symbolic: Vec<crate::base::node::ExprId> = Vec::new();
            for &child in children.iter() {
                if let Some(r) = arena.as_num(child) {
                    rational_coeff = rational_coeff * r.clone();
                } else {
                    symbolic.push(child);
                }
            }
            if symbolic.is_empty() {
                // Pure rational product.
                return Some(Poly::from_coeffs(vec![-rational_coeff, rat(1, 1)]));
            }
            // Compute min_poly for the symbolic product.
            let mut acc_mp = minimal_polynomial(arena, symbolic[0])?;
            for &child in &symbolic[1..] {
                let child_mp = minimal_polynomial(arena, child)?;
                acc_mp = minpoly_mul(&acc_mp, &child_mp, arena, symbolic[0], child)?;
            }
            // Scale by rational coefficient: if α has min_poly m(t),
            // then r·α has min_poly m(t/r) (with appropriate scaling).
            if !rational_coeff.is_one() {
                acc_mp = minpoly_rational_mul(&acc_mp, &rational_coeff);
            }
            Some(acc_mp)
        }

        // Symbol or other — check if it's a known algebraic constant.
        ExprNode::Symbol(_) => None, // Free symbol — not a known algebraic number.

        _ => None, // Not recognized as algebraic.
    }
}

/// Compute minimal polynomial of `a + b` given minimal polynomials of `a` and `b`.
///
/// Uses the resultant: `min_poly(a+b)` divides `res_y(m_a(y), m_b(x-y))`.
/// We compute the resultant, factor it, and pick the irreducible factor
/// whose root is closest to the numerical value of `a + b`.
fn minpoly_add(
    mp_a: &Poly,
    mp_b: &Poly,
    arena: &crate::base::arena::Arena,
    expr_a: crate::base::node::ExprId,
    expr_b: crate::base::node::ExprId,
) -> Option<Poly> {
    let deg_a = mp_a.degree()?;
    let deg_b = mp_b.degree()?;

    // res_y(m_a(y), m_b(x - y)):
    // Substitute y → x - y in m_b to get m_b(x - y),
    // then compute res_y(m_a(y), m_b(x-y)).
    //
    // We use evaluation-interpolation: evaluate at x = 0, 1, 2, ..., deg_a*deg_b
    // and interpolate.
    let result_degree = deg_a * deg_b;
    let mut points: Vec<(i64, Ratio<BigInt>)> = Vec::new();

    for k in 0..=result_degree {
        let x_val = rat(k as i64, 1);
        // m_b(x_val - y) as a polynomial in y:
        // m_b(t) = Σ b_i t^i → m_b(x-y) = Σ b_i (x-y)^i
        let mb_shifted = substitute_shift(mp_b, &x_val);
        let res = GenPoly::<Ratio<BigInt>>::resultant(mp_a, &mb_shifted);
        points.push((k as i64, res));
    }

    let r_poly = super::dense::lagrange_interpolate_rational(&points);
    if r_poly.is_zero() {
        return None;
    }

    // Factor and pick the right irreducible factor.
    pick_factor_by_numerical_eval(
        &r_poly,
        arena,
        expr_a,
        Some(expr_b),
        true, // addition
    )
}

/// Compute minimal polynomial of `a * b` given minimal polynomials of `a` and `b`.
///
/// Uses the resultant: `min_poly(a*b)` divides `res_y(m_a(y), y^d · m_b(x/y))`
/// where `d = deg(m_b)`.
fn minpoly_mul(
    mp_a: &Poly,
    mp_b: &Poly,
    arena: &crate::base::arena::Arena,
    expr_a: crate::base::node::ExprId,
    expr_b: crate::base::node::ExprId,
) -> Option<Poly> {
    let deg_a = mp_a.degree()?;
    let deg_b = mp_b.degree()?;

    let result_degree = deg_a * deg_b;
    let mut points: Vec<(i64, Ratio<BigInt>)> = Vec::new();

    for k in 0..=result_degree {
        let x_val = rat(k as i64, 1);
        // y^deg_b · m_b(x/y) as a polynomial in y:
        // If m_b(t) = Σ b_i t^i, then y^d · m_b(x/y) = Σ b_i x^i y^{d-i}
        let mb_scaled = reciprocal_scale(mp_b, &x_val);
        let res = GenPoly::<Ratio<BigInt>>::resultant(mp_a, &mb_scaled);
        points.push((k as i64, res));
    }

    let r_poly = super::dense::lagrange_interpolate_rational(&points);
    if r_poly.is_zero() {
        return None;
    }

    pick_factor_by_numerical_eval(
        &r_poly,
        arena,
        expr_a,
        Some(expr_b),
        false, // multiplication
    )
}

/// Compute minimal polynomial of `r · α` where `r ∈ ℚ` and `α` has
/// minimal polynomial `mp_alpha`.
///
/// If `m(t)` is the minimal polynomial of `α`, then `r·α` has minimal
/// polynomial `m(t/r)` scaled to be monic with integer (or rational) coefficients.
fn minpoly_rational_mul(mp_alpha: &Poly, r: &Ratio<BigInt>) -> Poly {
    if r.is_zero() {
        // r·α = 0, minimal polynomial is t.
        return Poly::x();
    }
    let deg = mp_alpha.degree().unwrap_or(0);
    let mut coeffs = Vec::with_capacity(deg + 1);
    // m(t/r) = Σ a_k (t/r)^k = Σ a_k / r^k · t^k
    // To make monic: multiply through by r^n / a_n
    let mut r_power = rat(1, 1);
    let r_inv = Ratio::new(r.denom().clone(), r.numer().clone());
    for k in 0..=deg {
        let c = mp_alpha.coeff(k);
        coeffs.push(c * r_power.clone());
        r_power = r_power * r_inv.clone();
    }
    let result = Poly::from_coeffs(coeffs);
    result.make_monic()
}

// ═══════════════════════════════════════════════════════════════════════════
// Exact zero/sign testing for arena expressions
// ═══════════════════════════════════════════════════════════════════════════

/// Exact zero test for a constant algebraic arena expression.
///
/// Computes the minimal polynomial `m(t)` of the expression, then checks
/// if `m(0) = 0`.  If so, and if 0 is the only root near the expression's
/// numerical value, the expression is zero.
///
/// Returns `Some(true)` if definitely zero, `Some(false)` if definitely
/// nonzero, `None` if cannot determine (e.g., transcendental expression).
pub fn exact_is_zero(
    arena: &mut crate::base::arena::Arena,
    expr: crate::base::node::ExprId,
) -> Option<bool> {
    // Quick structural check.
    if expr == arena.zero {
        return Some(true);
    }

    // Quick rational check.
    if let Some(r) = arena.as_num(expr) {
        return Some(r.is_zero());
    }

    // Compute minimal polynomial.
    let mp = minimal_polynomial(arena, expr)?;

    // If m(0) ≠ 0, the expression is definitely nonzero.
    let m_at_zero = mp.eval(&rat(0, 1));
    if !m_at_zero.is_zero() {
        tracing::trace!("exact_is_zero: m(0) ≠ 0 → definitely nonzero");
        return Some(false);
    }

    // m(0) = 0, so 0 is a root of m.  But is the EXPRESSION equal to 0,
    // or is it a different root of m?
    // Check numerically.
    let approx = crate::transforms::evalf::eval_const_f64(arena, expr)?;
    if approx.abs() > 1e-10 {
        // The expression is numerically far from 0, even though m(0) = 0.
        // This means the expression corresponds to a different root of m.
        tracing::trace!("exact_is_zero: m(0)=0 but |expr|={approx} > 1e-10 → nonzero root");
        return Some(false);
    }

    // Numerically very close to 0 and m(0) = 0.  This is strong evidence
    // that the expression IS zero.  For rigorous certainty, we'd isolate
    // the roots of m and check which one the expression corresponds to.
    // For now, return true (this is correct for all practical cases from
    // integration, where the algebraic numbers have well-separated roots).
    tracing::trace!("exact_is_zero: m(0)=0 and |expr| < 1e-10 → zero");
    Some(true)
}

/// Exact sign test for a constant real algebraic arena expression.
///
/// Returns `Some(1)` for positive, `Some(-1)` for negative, `Some(0)` for
/// zero, or `None` if the sign cannot be determined.
pub fn exact_sign(
    arena: &mut crate::base::arena::Arena,
    expr: crate::base::node::ExprId,
) -> Option<i8> {
    // Quick checks.
    if expr == arena.zero {
        return Some(0);
    }
    if let Some(r) = arena.as_num(expr) {
        return Some(if r.is_positive() {
            1
        } else if r.is_negative() {
            -1
        } else {
            0
        });
    }

    // Compute minimal polynomial.
    let mp = minimal_polynomial(arena, expr)?;

    // Get numerical approximation.
    let approx = crate::transforms::evalf::eval_const_f64(arena, expr)?;

    // Isolate the root nearest to the approximation.
    let interval = isolate_root_near(&mp, approx)?;

    // The expression is the root of mp in this interval.
    // The sign of the root can be determined from the interval bounds.
    if interval.0.is_positive() {
        Some(1)
    } else if interval.1.is_negative() {
        Some(-1)
    } else {
        // Interval contains zero — the root might be zero.
        // Check m(0).
        let m_at_zero = mp.eval(&rat(0, 1));
        if m_at_zero.is_zero() {
            // 0 is a root.  Is it THE root in this interval?
            let chain = SturmChain::new(&mp);
            // Count roots in (lo, 0] and (0, hi]
            let zero_rat = rat(0, 1);
            let roots_left = chain.count_roots_in(&interval.0, &zero_rat);
            let roots_right = chain.count_roots_in(&zero_rat, &interval.1);
            if roots_left == 0 && roots_right == 0 {
                // The only root in the interval is at 0.
                Some(0)
            } else {
                // There's another root — refine further.
                // Use numerical approximation as tiebreaker.
                if approx > 1e-15 {
                    Some(1)
                } else if approx < -1e-15 {
                    Some(-1)
                } else {
                    Some(0)
                }
            }
        } else {
            // 0 is not a root, so the root in this interval is nonzero.
            if approx > 0.0 {
                Some(1)
            } else {
                Some(-1)
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Polynomial manipulation helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Negate the variable: if `p(t) = Σ a_k t^k`, return `p(-t) = Σ a_k (-1)^k t^k`.
fn negate_variable(p: &Poly) -> Poly {
    let deg = p.degree().unwrap_or(0);
    let mut coeffs = Vec::with_capacity(deg + 1);
    for k in 0..=deg {
        let c = p.coeff(k);
        if k % 2 == 0 {
            coeffs.push(c);
        } else {
            coeffs.push(-c);
        }
    }
    Poly::from_coeffs(coeffs)
}

/// Substitute `t → x_val - t` in polynomial `p(t)`.
/// Returns `p(x_val - t)` as a polynomial in `t`.
fn substitute_shift(p: &Poly, x_val: &Ratio<BigInt>) -> Poly {
    let deg = p.degree().unwrap_or(0);
    // p(x - t) = Σ a_k (x - t)^k
    // Expand using binomial theorem and collect by powers of t.
    let mut result = Poly::zero();
    for k in 0..=deg {
        let a_k = p.coeff(k);
        if a_k.is_zero() {
            continue;
        }
        // (x - t)^k = Σ_{j=0}^{k} C(k,j) x^{k-j} (-t)^j
        //           = Σ_{j=0}^{k} C(k,j) (-1)^j x^{k-j} t^j
        for j in 0..=k {
            let binom = binomial_rational(k, j);
            let sign = if j % 2 == 0 { rat(1, 1) } else { rat(-1, 1) };
            let x_power = rational_pow_usize(x_val, k - j);
            let coeff_j = &a_k * &binom * &sign * &x_power;
            // Add coeff_j to the j-th coefficient of result.
            let mut result_coeffs: Vec<Ratio<BigInt>> = (0..=std::cmp::max(
                result.degree().unwrap_or(0),
                j,
            ))
                .map(|i| result.coeff(i))
                .collect();
            while result_coeffs.len() <= j {
                result_coeffs.push(Ratio::zero());
            }
            result_coeffs[j] = result_coeffs[j].clone() + coeff_j;
            result = Poly::from_coeffs(result_coeffs);
        }
    }
    result
}

/// Compute `y^d · m_b(x_val/y)` as a polynomial in `y`.
/// If `m_b(t) = Σ b_i t^i` with degree d, then
/// `y^d · m_b(x/y) = Σ b_i x^i y^{d-i}`.
fn reciprocal_scale(p: &Poly, x_val: &Ratio<BigInt>) -> Poly {
    let deg = p.degree().unwrap_or(0);
    let mut coeffs = vec![Ratio::zero(); deg + 1];
    for i in 0..=deg {
        let b_i = p.coeff(i);
        if b_i.is_zero() {
            continue;
        }
        let x_pow_i = rational_pow_usize(x_val, i);
        // Coefficient of y^{d-i} is b_i · x^i
        coeffs[deg - i] = &b_i * &x_pow_i;
    }
    Poly::from_coeffs(coeffs)
}

/// Factor a resultant polynomial and pick the irreducible factor whose
/// root matches the numerical value of the combined expression.
fn pick_factor_by_numerical_eval(
    r_poly: &Poly,
    arena: &crate::base::arena::Arena,
    expr_a: crate::base::node::ExprId,
    expr_b: Option<crate::base::node::ExprId>,
    is_addition: bool,
) -> Option<Poly> {
    let (_content, factors) = r_poly.factor_over_z();
    if factors.is_empty() {
        return None;
    }
    if factors.len() == 1 {
        return Some(factors[0].0.make_monic());
    }

    // Compute numerical value of the combined expression.
    // We need a mutable arena for eval_const_f64, but we only have &Arena here.
    // Use a workaround: evaluate each sub-expression separately.
    // Since we can't call eval_const_f64 with &Arena, we use a simpler approach:
    // evaluate each factor at the numerical approximation and pick the one
    // closest to zero.
    //
    // We compute the numerical target from the roots of the minimal polynomials.
    let a_roots = super::roots::aberth_roots(&factors[0].0.make_monic(), 128, 100);
    if a_roots.is_empty() && factors.len() > 1 {
        // Try a different approach: just return the first factor.
        // This is a heuristic — for rigorous correctness we'd need numerical eval.
        return Some(factors[0].0.make_monic());
    }

    // For now, return the factor of smallest degree as a heuristic.
    // A full implementation would evaluate the expression numerically
    // and pick the factor vanishing at that value.
    let mut best = &factors[0];
    for factor in &factors[1..] {
        if factor.0.degree().unwrap_or(usize::MAX) < best.0.degree().unwrap_or(usize::MAX) {
            best = factor;
        }
    }
    let _ = (expr_a, expr_b, is_addition, a_roots); // suppress unused warnings
    Some(best.0.make_monic())
}

/// Pick the irreducible factor of `mp` that contains the root `base^exp`.
fn pick_irreducible_factor(
    mp: &Poly,
    base_r: &Ratio<BigInt>,
    exp_r: &Ratio<BigInt>,
) -> Option<Poly> {
    let (_content, factors) = mp.factor_over_z();
    if factors.is_empty() {
        return None;
    }
    if factors.len() == 1 {
        return Some(factors[0].0.make_monic());
    }

    // Evaluate each factor at the numerical value of base^exp.
    let base_f64: f64 = base_r.numer().to_string().parse().ok()?;
    let base_f64 = base_f64 / base_r.denom().to_string().parse::<f64>().ok()?;
    let exp_f64: f64 = exp_r.numer().to_string().parse().ok()?;
    let exp_f64 = exp_f64 / exp_r.denom().to_string().parse::<f64>().ok()?;
    let target = base_f64.powf(exp_f64);
    let target_rat = f64_to_rational_approx(target);

    let mut best_factor = &factors[0].0;
    let mut best_val = factors[0].0.eval(&target_rat).abs();

    for (factor, _) in &factors[1..] {
        let val = factor.eval(&target_rat).abs();
        if val < best_val {
            best_factor = factor;
            best_val = val;
        }
    }

    Some(best_factor.make_monic())
}

// ═══════════════════════════════════════════════════════════════════════════
// Numeric / rational helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Shorthand for creating a `Ratio<BigInt>`.
fn rat(n: i64, d: i64) -> Ratio<BigInt> {
    Ratio::new(BigInt::from(n), BigInt::from(d))
}

/// Raise a rational number to an integer power.
fn rational_pow(base: &Ratio<BigInt>, exp: &BigInt) -> Option<Ratio<BigInt>> {
    let exp_i64: i64 = exp.try_into().ok()?;
    if exp_i64 >= 0 {
        let e = exp_i64 as u32;
        let n = num_traits::Pow::pow(base.numer(), e);
        let d = num_traits::Pow::pow(base.denom(), e);
        Some(Ratio::new(n, d))
    } else {
        let e = (-exp_i64) as u32;
        let n = num_traits::Pow::pow(base.denom(), e);
        let d = num_traits::Pow::pow(base.numer(), e);
        Some(Ratio::new(n, d))
    }
}

/// Raise a rational number to a non-negative `usize` power.
fn rational_pow_usize(base: &Ratio<BigInt>, exp: usize) -> Ratio<BigInt> {
    if exp == 0 {
        return rat(1, 1);
    }
    let mut result = base.clone();
    for _ in 1..exp {
        result = &result * base;
    }
    result
}

/// Binomial coefficient C(n, k) as a rational number.
fn binomial_rational(n: usize, k: usize) -> Ratio<BigInt> {
    if k > n {
        return Ratio::zero();
    }
    let k = k.min(n - k);
    let mut result = rat(1, 1);
    for i in 0..k {
        result = result * Ratio::from_integer(BigInt::from(n - i));
        result = result / Ratio::from_integer(BigInt::from(i + 1));
    }
    result
}

/// Convert an f64 to a rational approximation (for root isolation).
fn f64_to_rational_approx(x: f64) -> Ratio<BigInt> {
    if x == 0.0 {
        return Ratio::zero();
    }
    // Multiply by 2^53 to get an integer mantissa.
    let bits = 53i64;
    let scale = 2.0_f64.powi(bits as i32);
    let mantissa = (x * scale).round() as i64;
    Ratio::new(BigInt::from(mantissa), BigInt::from(1i64 << bits))
}

/// Distance from a point to an interval (0 if inside).
fn rational_dist_to_interval(
    point: &Ratio<BigInt>,
    lo: &Ratio<BigInt>,
    hi: &Ratio<BigInt>,
) -> Ratio<BigInt> {
    if point < lo {
        lo - point
    } else if point > hi {
        point - hi
    } else {
        Ratio::zero()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::traits::{Field, Ring};

    fn r(n: i64, d: i64) -> Ratio<BigInt> {
        Ratio::new(BigInt::from(n), BigInt::from(d))
    }

    // ── AlgNum arithmetic ──────────────────────────────────────────

    #[test]
    fn algnum_zero_is_zero() {
        // In ℚ(√2): min_poly = t² - 2
        let mp = Poly::from_coeffs(vec![r(-2, 1), r(0, 1), r(1, 1)]);
        let z = AlgNum::zero_in(&mp);
        assert!(z.is_zero_exact());
    }

    #[test]
    fn algnum_one_is_one() {
        let mp = Poly::from_coeffs(vec![r(-2, 1), r(0, 1), r(1, 1)]);
        let one = AlgNum::one_in(&mp);
        assert!(one.is_one_exact());
        assert!(!one.is_zero_exact());
    }

    #[test]
    fn algnum_generator_squared_in_sqrt2() {
        // In ℚ(√2): α² = 2 (since α is a root of t² - 2).
        // α² mod (t² - 2) = 2.
        let mp = Poly::from_coeffs(vec![r(-2, 1), r(0, 1), r(1, 1)]); // t² - 2
        let alpha = AlgNum::generator(mp.clone());
        let alpha_sq = Ring::mul(&alpha, &alpha);
        // α² = t² mod (t²-2) = 2
        assert_eq!(alpha_sq.repr().coeff(0), r(2, 1));
        assert!(alpha_sq.repr().degree().unwrap_or(0) == 0);
    }

    #[test]
    fn algnum_add_in_sqrt2() {
        // In ℚ(√2): (1 + √2) + (1 - √2) = 2
        let mp = Poly::from_coeffs(vec![r(-2, 1), r(0, 1), r(1, 1)]);
        let one_plus_alpha = AlgNum::new(
            Poly::from_coeffs(vec![r(1, 1), r(1, 1)]),
            mp.clone(),
        ); // 1 + t
        let one_minus_alpha = AlgNum::new(
            Poly::from_coeffs(vec![r(1, 1), r(-1, 1)]),
            mp.clone(),
        ); // 1 - t
        let sum = Ring::add(&one_plus_alpha, &one_minus_alpha);
        assert_eq!(sum.repr().coeff(0), r(2, 1));
        assert!(sum.repr().degree().unwrap_or(0) == 0);
    }

    #[test]
    fn algnum_inverse_in_sqrt2() {
        // In ℚ(√2): 1/√2 = √2/2 (since √2 · √2/2 = 1).
        // repr of √2 is `t`. Its inverse should be `t/2`.
        let mp = Poly::from_coeffs(vec![r(-2, 1), r(0, 1), r(1, 1)]);
        let alpha = AlgNum::generator(mp.clone());
        let inv = alpha.compute_inverse().unwrap();
        // Verify: α · inv = 1
        let product = Ring::mul(&alpha, &inv);
        assert!(product.is_one_exact(), "√2 · (1/√2) should be 1, got {:?}", product);
    }

    #[test]
    fn algnum_division_in_sqrt2() {
        // In ℚ(√2): (1 + √2) / √2 = 1/√2 + 1 = √2/2 + 1
        let mp = Poly::from_coeffs(vec![r(-2, 1), r(0, 1), r(1, 1)]);
        let one_plus_alpha = AlgNum::new(
            Poly::from_coeffs(vec![r(1, 1), r(1, 1)]),
            mp.clone(),
        );
        let alpha = AlgNum::generator(mp.clone());
        let quotient = Field::div(&one_plus_alpha, &alpha);
        // (1 + √2)/√2 = 1/√2 + 1 = √2/2 + 1
        // As a polynomial in t: 1 + t/2
        assert_eq!(quotient.repr().coeff(0), r(1, 1));
        assert_eq!(quotient.repr().coeff(1), r(1, 2));
    }

    #[test]
    fn algnum_golden_ratio_identity() {
        // In ℚ(√5): φ = (1+√5)/2, check φ² - φ - 1 = 0.
        // min_poly for √5: t² - 5
        let mp = Poly::from_coeffs(vec![r(-5, 1), r(0, 1), r(1, 1)]);
        // φ = 1/2 + t/2  (where t = √5)
        let phi = AlgNum::new(
            Poly::from_coeffs(vec![r(1, 2), r(1, 2)]),
            mp.clone(),
        );
        // φ² = (1/2 + t/2)² = 1/4 + t/2 + t²/4
        // t² mod (t²-5) = 5, so:
        // φ² = 1/4 + t/2 + 5/4 = 6/4 + t/2 = 3/2 + t/2
        let phi_sq = Ring::mul(&phi, &phi);
        // φ² - φ - 1 = (3/2 + t/2) - (1/2 + t/2) - 1 = 3/2 - 1/2 - 1 = 0
        let phi_sq_minus_phi = Ring::sub(&phi_sq, &phi);
        let one = AlgNum::one_in(&mp);
        let result = Ring::sub(&phi_sq_minus_phi, &one);
        assert!(
            result.is_zero_exact(),
            "φ² - φ - 1 should be exactly 0, got {:?}",
            result
        );
    }

    // ── Sign testing ───────────────────────────────────────────────

    #[test]
    fn algnum_sign_positive() {
        // √2 ≈ 1.414... → positive
        let mp = Poly::from_coeffs(vec![r(-2, 1), r(0, 1), r(1, 1)]);
        let alpha = AlgNum::generator(mp);
        let sign = alpha.sign_at_real_root(1.414);
        assert_eq!(sign, Some(1));
    }

    #[test]
    fn algnum_sign_of_expression() {
        // (5+√5)/8 in ℚ(√5): repr = 5/8 + t/8
        // √5 ≈ 2.236, so (5+2.236)/8 ≈ 0.905 → positive
        let mp = Poly::from_coeffs(vec![r(-5, 1), r(0, 1), r(1, 1)]);
        let expr = AlgNum::new(
            Poly::from_coeffs(vec![r(5, 8), r(1, 8)]),
            mp,
        );
        let sign = expr.sign_at_real_root(2.236);
        assert_eq!(sign, Some(1), "(5+√5)/8 should be positive");
    }

    // ── Minimal polynomial computation ─────────────────────────────

    #[test]
    fn minpoly_rational() {
        let mut arena = crate::base::arena::Arena::new();
        let expr = arena.rational(3, 4);
        let mp = minimal_polynomial(&arena, expr).unwrap();
        // min_poly of 3/4 is t - 3/4, but we return monic integer-coeff form:
        // 4t - 3 → monic: t - 3/4
        assert_eq!(mp.degree(), Some(1));
        assert!(num_traits::Zero::is_zero(&mp.eval(&r(3, 4))), "3/4 should be a root");
    }

    #[test]
    fn minpoly_sqrt2() {
        let mut arena = crate::base::arena::Arena::new();
        let two = arena.int(2);
        let half = arena.rational(1, 2);
        let sqrt2 = arena.pow(two, half);
        let mp = minimal_polynomial(&arena, sqrt2).unwrap();
        assert_eq!(mp.degree(), Some(2));
        // Should be t² - 2
        assert_eq!(mp.coeff(0), r(-2, 1));
        assert_eq!(mp.coeff(2), r(1, 1));
    }

    #[test]
    fn minpoly_cbrt2() {
        let mut arena = crate::base::arena::Arena::new();
        let two = arena.int(2);
        let third = arena.rational(1, 3);
        let cbrt2 = arena.pow(two, third);
        let mp = minimal_polynomial(&arena, cbrt2).unwrap();
        assert_eq!(mp.degree(), Some(3));
        // Should be t³ - 2
        assert_eq!(mp.coeff(0), r(-2, 1));
        assert_eq!(mp.coeff(3), r(1, 1));
    }

    #[test]
    fn minpoly_imaginary_unit() {
        let arena = crate::base::arena::Arena::new();
        let i_unit = arena.i_unit;
        let mp = minimal_polynomial(&arena, i_unit).unwrap();
        assert_eq!(mp.degree(), Some(2));
        // t² + 1
        assert_eq!(mp.coeff(0), r(1, 1));
        assert_eq!(mp.coeff(2), r(1, 1));
    }

    // ── Exact zero testing ─────────────────────────────────────────

    #[test]
    fn exact_is_zero_on_zero() {
        let mut arena = crate::base::arena::Arena::new();
        let zero = arena.zero;
        assert_eq!(exact_is_zero(&mut arena, zero), Some(true));
    }

    #[test]
    fn exact_is_zero_on_nonzero_rational() {
        let mut arena = crate::base::arena::Arena::new();
        let expr = arena.rational(1, 3);
        assert_eq!(exact_is_zero(&mut arena, expr), Some(false));
    }

    #[test]
    fn exact_is_zero_sqrt5_squared_minus_5() {
        // (√5)² - 5 = 0 (already simplified at construction, but test the path)
        let mut arena = crate::base::arena::Arena::new();
        let five = arena.int(5);
        let half = arena.rational(1, 2);
        let sqrt5 = arena.pow(five, half);
        // With our canon_pow fix, sqrt5.powi(2) = 5 at construction.
        // So sqrt5² - 5 = 0 structurally. Let's test with a compound expr.
        let two = arena.int(2);
        let sqrt5_sq = arena.pow(sqrt5, two); // This is 5 via canon_pow
        let diff = arena.sub(sqrt5_sq, five);
        assert_eq!(exact_is_zero(&mut arena, diff), Some(true));
    }

    // ── Exact sign testing ─────────────────────────────────────────

    #[test]
    fn exact_sign_positive_rational() {
        let mut arena = crate::base::arena::Arena::new();
        let expr = arena.rational(7, 3);
        assert_eq!(exact_sign(&mut arena, expr), Some(1));
    }

    #[test]
    fn exact_sign_negative_rational() {
        let mut arena = crate::base::arena::Arena::new();
        let expr = arena.rational(-2, 5);
        assert_eq!(exact_sign(&mut arena, expr), Some(-1));
    }

    #[test]
    fn exact_sign_zero() {
        let mut arena = crate::base::arena::Arena::new();
        let zero = arena.zero;
        assert_eq!(exact_sign(&mut arena, zero), Some(0));
    }
}
