//! Rothstein-Trager algorithm for the logarithmic part of rational function
//! integration.
//!
//! Given `∫ A(x)/D(x) dx` where `D` is **square-free** and `deg(A) < deg(D)`,
//! computes the logarithmic part:
//!
//! ```text
//!   ∫ A/D dx  =  Σ cᵢ · ln(vᵢ(x))
//! ```
//!
//! # Algorithm
//!
//! 1. Compute `R(t) = res_x(D, A − t·D')` — a polynomial in `t`.
//! 2. Factor `R(t)` over ℤ: irreducible factors `q` with multiplicities `i`.
//! 3. For each linear factor `(t − c)`, `c ∈ ℚ`:
//!    - `v(x) = gcd(D(x), A(x) − c·D'(x))` — the logarithmic argument.
//!    - Emit `c · ln(v)`.
//! 4. For each irreducible factor `q` of degree > 1 and multiplicity `i`
//!    (Lazard–Rioboo–Trager): the log argument at a root `α` of `q` is
//!    `gcd(D, A − α·D')`, of degree `i` in `x`, and equals `S_i(α, x)` for
//!    the member `S_i` of degree `i` of the remainder sequence of `D` and
//!    `A − t·D'` over `ℚ(t)`.  Emit an `Algebraic` term
//!    `Σ_{q(α)=0} α·ln(S_i(α, x))`, with `S_i` made monic in `x` and its
//!    coefficients reduced modulo `q`.
//!
//! The distinct roots of `R(t)` are exactly the residues of `A/D`, and the
//! splitting field of `R(t)` is the **minimal** algebraic extension needed
//! to express the integral.  A residue of multiplicity `i` belongs to a log
//! argument of degree `i`: before 0.25 every factor got the degree-1 member,
//! which only exists when all multiplicities are 1, so `∫ x³/(x⁸+1) dx`
//! (`R = (64t²+1)⁴`) fell through to a partial-fraction route over the
//! complex roots whose expansion exhausted memory.
//!
//! # References
//!
//! - Rothstein, "A new algorithm for the integration of exponential and
//!   logarithmic functions", 1977
//! - Trager, "Algebraic factoring and rational function integration", 1976
//! - Bronstein, *Symbolic Integration I*, §2.4–2.5
//! - Lazard & Rioboo, "Integration of rational functions: rational computation
//!   of the logarithmic part", 1990
//! - Bronstein, *Symbolic Integration I*, §2.5 (`IntRationalLogPart`); SymPy's
//!   `ratint_logpart` (`sympy/integrals/rationaltools.py`) is the same
//!   algorithm.

use std::collections::BTreeMap;

use num_rational::Ratio;
use num_traits::Zero;

use super::LogTerm;
use super::log_to_real::{poly_to_genpoly_rf, poly_to_genpoly_rf_times_t};
use crate::base::numeric::Q;
use crate::poly::dense::Poly;
use crate::poly::generic::GenPoly;
use crate::poly::ratfn::RationalFn;

// ═══════════════════════════════════════════════════════════════════════════
// Public types
// ═══════════════════════════════════════════════════════════════════════════

/// Result of the Rothstein-Trager algorithm.
#[derive(Clone, Debug)]
pub struct LogPartResult {
    /// The logarithmic terms `Σ cᵢ ln(vᵢ)`.
    pub terms: Vec<LogTerm>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Main algorithm
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the logarithmic part of `∫ A/D dx` where `D` is square-free
/// and `deg(A) < deg(D)`.
///
/// If `A` is zero, returns an empty result (no logarithmic part).
///
/// # Panics
///
/// Panics if `D` is zero.
pub fn logarithmic_part(a: &Poly, d: &Poly) -> LogPartResult {
    assert!(
        !d.is_zero(),
        "logarithmic_part: denominator must be nonzero"
    );

    if a.is_zero() {
        return LogPartResult { terms: vec![] };
    }

    let d_prime = d.derivative();

    // Handle the trivial case: D is linear (degree 1).
    // Then ∫ A/D dx = (A/lc(D)) · ln(D).  Since deg(A) < deg(D) = 1,
    // A is a constant.
    if d.degree() == Some(1) {
        let a_val = a.coeff(0);
        let d_lc = d.coeff(1);
        let coeff = a_val / d_lc;
        if coeff.is_zero() {
            return LogPartResult { terms: vec![] };
        }
        return LogPartResult {
            terms: vec![LogTerm::Rational {
                coeff,
                argument: d.make_monic(),
            }],
        };
    }

    // R(t) = res_x(D, A − t·D')
    //
    // Poly::resultant_poly(f, g, h) computes res_x(f, g − t·h) as a
    // polynomial in t via evaluation-interpolation.
    let r_poly = Poly::resultant_poly(d, a, &d_prime);

    if r_poly.is_zero() {
        // Degenerate case: resultant is identically zero.
        // This means D and A - t·D' share a common factor for all t,
        // which shouldn't happen for a proper fraction with square-free D.
        return LogPartResult { terms: vec![] };
    }

    // Factor R(t) over ℤ.  A factor's multiplicity is the degree in x of
    // the log argument belonging to its roots.
    let (_content, r_factors) = r_poly.factor_over_z();

    // The remainder sequence, down to the lowest degree an algebraic factor
    // needs (the members below it are the expensive ones).
    let lowest = r_factors
        .iter()
        .filter(|(f, _)| f.degree().is_some_and(|d| d >= 2))
        .map(|(_, i)| *i as usize)
        .min();
    let prs = lowest.map(|i| lrt_prs(a, d, &d_prime, i));

    let mut terms: Vec<LogTerm> = Vec::new();

    for (factor, multiplicity) in &r_factors {
        let deg = match factor.degree() {
            Some(d) => d,
            None => continue, // zero polynomial — skip
        };

        if deg == 0 {
            // Constant factor — no roots, skip.
            continue;
        }

        if deg == 1 {
            // Linear factor: (t - c) where c = -factor[0] / factor[1].
            let c = extract_linear_root(factor);

            // v(x) = gcd(D(x), A(x) − c · D'(x))
            let a_minus_c_dprime = a - &d_prime.scale(&c);
            let v = Poly::gcd(d, &a_minus_c_dprime);

            if v.degree().unwrap_or(0) == 0 {
                // GCD is constant — this root doesn't contribute a log term.
                continue;
            }

            if !c.is_zero() {
                terms.push(LogTerm::Rational {
                    coeff: c,
                    argument: v.make_monic(),
                });
            }
        } else {
            // Irreducible factor of degree > 1: roots are algebraic numbers
            // not in ℚ, with log argument S_i(α, x), i = multiplicity.
            let i = *multiplicity as usize;
            let log_arg = if d.degree() == Some(i) {
                Some(poly_to_genpoly_rf(&d.make_monic()))
            } else {
                prs.as_ref()
                    .and_then(|p| p.get(&i))
                    .and_then(|s| reduce_log_arg(s, factor))
            };
            if log_arg.is_none() {
                tracing::debug!(
                    min_poly_degree = deg,
                    multiplicity = i,
                    "logarithmic_part: no usable remainder of degree i"
                );
            }
            terms.push(LogTerm::Algebraic {
                min_poly: factor.clone(),
                log_arg,
            });
        }
    }

    LogPartResult { terms }
}

/// The Euclidean remainder sequence of `D` and `A − t·D'` over `ℚ(t)`,
/// keyed by degree in `x`, stopping once the member of degree `stop_at` is
/// produced.  Each remainder is made primitive over `ℚ[t]`: Euclid over
/// `ℚ(t)` grows the coefficients' degree in `t` at every step (the degree
/// 2 → 1 step of `∫ atan(√x − x³) dx` alone took 4 s), while a primitive
/// member is the subresultant of its degree divided by its content, so it
/// specialises at a root of the resultant to the same polynomial up to a
/// non-zero constant.
fn lrt_prs(
    a: &Poly,
    d: &Poly,
    d_prime: &Poly,
    stop_at: usize,
) -> BTreeMap<usize, GenPoly<RationalFn>> {
    let d_gp = poly_to_genpoly_rf(d);
    let b_gp = &poly_to_genpoly_rf(a) - &poly_to_genpoly_rf_times_t(d_prime);
    GenPoly::<RationalFn>::euclidean_prs_normalized(&d_gp, &b_gp, Some(stop_at), primitive_over_q_t)
}

/// `p ∈ ℚ(t)[x]` scaled by a non-zero element of ℚ(t) so that its
/// coefficients are polynomials in `t` with no common factor: clear the
/// denominators (their lcm in `ℚ[t]`) and divide by the gcd of the numerators.
pub(super) fn primitive_over_q_t(p: GenPoly<RationalFn>) -> GenPoly<RationalFn> {
    let coeffs = &p.coeffs;
    let mut lcm = Poly::from_int(1);
    for c in coeffs {
        let d = c.denom();
        if !d.is_constant() {
            let g = Poly::gcd(&lcm, d);
            lcm = (&lcm * d).div(&g);
        }
    }
    let numers: Vec<Poly> = coeffs
        .iter()
        .map(|c| (c.numer() * &lcm).div(c.denom()))
        .collect();
    let content = numers
        .iter()
        .filter(|n| !n.is_zero())
        .fold(Poly::zero(), |g, n| Poly::gcd(&g, n));
    if content.is_zero() {
        return p;
    }
    GenPoly::from_coeffs(
        numers
            .iter()
            .map(|n| RationalFn::from_poly(n.div(&content)))
            .collect(),
    )
}

/// `S(t, x)` made monic in `x`, each coefficient reduced to a polynomial in
/// `t` of degree below `deg q` — an element of `ℚ[t]/(q)`, the field of
/// the roots of the irreducible `q`.  `None` if the leading coefficient
/// vanishes at the roots of `q` (for a primitive remainder of the right
/// degree it cannot: the specialisation is then a non-zero multiple of
/// the gcd, which has that degree).
fn reduce_log_arg(s: &GenPoly<RationalFn>, q: &Poly) -> Option<GenPoly<RationalFn>> {
    let lc = s.leading_coeff()?.clone();
    let mut coeffs = Vec::with_capacity(s.coeffs.len());
    for c in &s.coeffs {
        // c / lc = (c.numer · lc.denom) / (c.denom · lc.numer)
        let num = (c.numer() * lc.denom()).rem(q);
        let den = (c.denom() * lc.numer()).rem(q);
        let inv = inverse_mod(&den, q)?;
        coeffs.push(RationalFn::from_poly((&num * &inv).rem(q)));
    }
    Some(GenPoly::from_coeffs(coeffs))
}

/// The inverse of `a` modulo `q`, if `gcd(a, q) = 1`.
fn inverse_mod(a: &Poly, q: &Poly) -> Option<Poly> {
    if a.is_zero() {
        return None;
    }
    let eg = Poly::extended_gcd(a, q);
    if eg.gcd.degree() != Some(0) {
        return None;
    }
    // eg.gcd is monic, i.e. 1: eg.x · a ≡ 1 (mod q).
    Some(eg.x.rem(q))
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Extract the rational root of a monic-ish linear polynomial `a·t + b`.
/// Returns `c = -b/a`.
fn extract_linear_root(factor: &Poly) -> Q {
    let a = factor.coeff(1);
    let b = factor.coeff(0);
    if a.is_zero() { Ratio::zero() } else { -b / a }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_traits::Signed;

    fn rat(n: i64, d: i64) -> Q {
        Ratio::new(BigInt::from(n), BigInt::from(d))
    }

    /// Verify a logarithmic part result by differentiating and comparing
    /// to the original integrand A/D.
    ///
    /// For Σ cᵢ ln(vᵢ), the derivative is Σ cᵢ · vᵢ'/vᵢ.
    /// This should equal A/D (as rational functions).
    fn verify_log_part(a: &Poly, d: &Poly, result: &LogPartResult) {
        // Compute Σ cᵢ · vᵢ' / vᵢ  as a single fraction.
        let mut sum_numer = Poly::zero();
        let mut sum_denom = Poly::from_int(1);

        for term in &result.terms {
            match term {
                LogTerm::Rational { coeff, argument } => {
                    let v = argument;
                    let v_prime = v.derivative();
                    // Add c · v'/v to the running sum.
                    // sum = sum_numer/sum_denom + c·v'/(v)
                    //     = (sum_numer · v + c · v' · sum_denom) / (sum_denom · v)
                    let c_poly = Poly::constant(coeff.clone());
                    sum_numer = &(&sum_numer * v) + &(&(&c_poly * &v_prime) * &sum_denom);
                    sum_denom = &sum_denom * v;
                }
                LogTerm::Algebraic { .. } => {
                    // Can't verify algebraic terms numerically at the Poly level.
                    // Skip verification for these.
                    return;
                }
            }
        }

        // Now check: sum_numer/sum_denom == A/D
        // i.e., sum_numer * D == A * sum_denom
        let lhs = &sum_numer * d;
        let rhs = &(a * &sum_denom);
        let diff = &lhs - rhs;

        assert!(
            diff.is_zero(),
            "Rothstein-Trager verification failed:\n  d/dx(Σ cᵢ ln(vᵢ)) ≠ A/D\n  diff = {diff}"
        );
    }

    #[test]
    fn log_part_one_over_x() {
        // ∫ 1/x dx = ln(x)
        let a = Poly::from_int(1);
        let d = Poly::x(); // x

        let result = logarithmic_part(&a, &d);
        assert_eq!(result.terms.len(), 1, "should have one log term");
        match &result.terms[0] {
            LogTerm::Rational { coeff, argument } => {
                assert_eq!(*coeff, rat(1, 1), "coefficient should be 1");
                assert_eq!(argument.degree(), Some(1), "argument should be linear (x)");
            }
            _ => panic!("expected rational log term"),
        }
        verify_log_part(&a, &d, &result);
    }

    #[test]
    fn log_part_one_over_x_minus_1() {
        // ∫ 1/(x-1) dx = ln(x-1)
        let a = Poly::from_int(1);
        let d = Poly::from_coeffs(vec![rat(-1, 1), rat(1, 1)]); // x - 1

        let result = logarithmic_part(&a, &d);
        assert_eq!(result.terms.len(), 1);
        match &result.terms[0] {
            LogTerm::Rational { coeff, argument } => {
                assert_eq!(*coeff, rat(1, 1));
                // argument should be (x - 1) or equivalent monic form
                assert_eq!(argument.degree(), Some(1));
            }
            _ => panic!("expected rational log term"),
        }
        verify_log_part(&a, &d, &result);
    }

    #[test]
    fn log_part_one_over_x_squared_minus_1() {
        // ∫ 1/(x^2 - 1) dx = 1/2 · ln(x - 1) - 1/2 · ln(x + 1)
        // D = x^2 - 1 = (x-1)(x+1), already square-free.
        let a = Poly::from_int(1);
        let d = Poly::from_coeffs(vec![rat(-1, 1), rat(0, 1), rat(1, 1)]); // x^2 - 1

        let result = logarithmic_part(&a, &d);
        // Should have 2 rational log terms with coefficients ±1/2.
        let rational_count = result
            .terms
            .iter()
            .filter(|t| matches!(t, LogTerm::Rational { .. }))
            .count();
        assert!(
            rational_count >= 1,
            "should have rational log terms for 1/(x^2-1), got {} rational terms out of {} total",
            rational_count,
            result.terms.len()
        );
        verify_log_part(&a, &d, &result);
    }

    #[test]
    fn log_part_2x_over_x_squared_plus_1() {
        // ∫ 2x/(x^2 + 1) dx = ln(x^2 + 1)
        // A = 2x, D = x^2 + 1 (square-free)
        let a = Poly::from_coeffs(vec![rat(0, 1), rat(2, 1)]); // 2x
        let d = Poly::from_coeffs(vec![rat(1, 1), rat(0, 1), rat(1, 1)]); // x^2 + 1

        let result = logarithmic_part(&a, &d);
        // Should produce 1 · ln(x^2 + 1)
        assert!(
            !result.terms.is_empty(),
            "should have at least one log term for 2x/(x^2+1)"
        );
        verify_log_part(&a, &d, &result);
    }

    #[test]
    fn log_part_one_over_x_squared_plus_1() {
        // ∫ 1/(x^2 + 1) dx = arctan(x)
        // But in the Risch framework, this is expressed as:
        //   (i/2)·ln(x - i) - (i/2)·ln(x + i)
        // The resultant R(t) = 4t^2 + 1, which is irreducible over ℤ.
        // So we should get an Algebraic term.
        let a = Poly::from_int(1);
        let d = Poly::from_coeffs(vec![rat(1, 1), rat(0, 1), rat(1, 1)]); // x^2 + 1

        let result = logarithmic_part(&a, &d);
        // Should have an algebraic term (roots are ±i/2).
        let has_algebraic = result
            .terms
            .iter()
            .any(|t| matches!(t, LogTerm::Algebraic { .. }));
        assert!(
            has_algebraic || result.terms.is_empty(),
            "1/(x^2+1) should produce algebraic log terms or be handled as arctan"
        );
    }

    #[test]
    fn log_part_zero_numerator() {
        let a = Poly::zero();
        let d = Poly::from_coeffs(vec![rat(-1, 1), rat(0, 1), rat(1, 1)]); // x^2 - 1
        let result = logarithmic_part(&a, &d);
        assert!(
            result.terms.is_empty(),
            "zero numerator should give empty result"
        );
    }

    #[test]
    fn log_part_partial_fraction_style() {
        // ∫ (3x + 5) / ((x + 1)(x + 2)) dx = ∫ (3x+5)/(x^2+3x+2) dx
        // = 2·ln(x+1) + 1·ln(x+2)    [partial fractions: 2/(x+1) + 1/(x+2)]
        let a = Poly::from_coeffs(vec![rat(5, 1), rat(3, 1)]); // 3x + 5
        let d = Poly::from_coeffs(vec![rat(2, 1), rat(3, 1), rat(1, 1)]); // x^2 + 3x + 2

        let result = logarithmic_part(&a, &d);
        verify_log_part(&a, &d, &result);

        // Should have exactly 2 rational log terms.
        let rational_terms: Vec<_> = result
            .terms
            .iter()
            .filter_map(|t| match t {
                LogTerm::Rational { coeff, argument } => Some((coeff.clone(), argument.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            rational_terms.len(),
            2,
            "should have 2 rational log terms, got {}: {:?}",
            rational_terms.len(),
            rational_terms
        );

        // Check coefficients sum to 3 (= leading coeff of A divided by leading coeff of D)
        // Actually, the coefficients should be 2 and 1.
        let mut coeffs: Vec<Q> = rational_terms.iter().map(|(c, _)| c.clone()).collect();
        coeffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(
            coeffs,
            vec![rat(1, 1), rat(2, 1)],
            "coefficients should be [1, 2], got {coeffs:?}"
        );
    }

    #[test]
    fn log_part_one_over_x_cubed_minus_1() {
        // ∫ 1/(x^3 - 1) dx
        // D = x^3 - 1 = (x - 1)(x^2 + x + 1), square-free.
        // The (x-1) factor gives a rational log term.
        // The (x^2+x+1) factor gives algebraic log terms (cube roots of unity).
        let a = Poly::from_int(1);
        let d = Poly::from_coeffs(vec![rat(-1, 1), rat(0, 1), rat(0, 1), rat(1, 1)]); // x^3 - 1

        let result = logarithmic_part(&a, &d);
        // Should have at least one rational term (from the x-1 factor)
        // and possibly algebraic terms.
        assert!(!result.terms.is_empty(), "1/(x^3-1) should have log terms");
        // Only verify if all terms are rational (can't verify algebraic).
        let all_rational = result
            .terms
            .iter()
            .all(|t| matches!(t, LogTerm::Rational { .. }));
        if all_rational {
            verify_log_part(&a, &d, &result);
        }
    }

    #[test]
    fn log_part_x_over_x_squared_minus_1() {
        // ∫ x/(x^2 - 1) dx = 1/2 · ln(x^2 - 1) = 1/2·ln(x-1) + 1/2·ln(x+1)
        let a = Poly::x(); // x
        let d = Poly::from_coeffs(vec![rat(-1, 1), rat(0, 1), rat(1, 1)]); // x^2 - 1

        let result = logarithmic_part(&a, &d);
        verify_log_part(&a, &d, &result);

        // Both log terms should have coefficient 1/2.
        let rational_terms: Vec<_> = result
            .terms
            .iter()
            .filter_map(|t| match t {
                LogTerm::Rational { coeff, .. } => Some(coeff.clone()),
                _ => None,
            })
            .collect();
        for c in &rational_terms {
            assert_eq!(c.abs(), rat(1, 2), "coefficients should be ±1/2, got {c}");
        }
    }

    /// `S(t, x)` divides `D(x)` in `(ℚ[t]/q)[x]`: every root `α` of `q` makes
    /// `S(α, x)` a factor of `D`.
    fn assert_log_arg_divides(d: &Poly, q: &Poly, s: &GenPoly<RationalFn>) {
        let d_gp = poly_to_genpoly_rf(d);
        let (_, rem) = d_gp.div_rem(s);
        for c in &rem.coeffs {
            assert!(c.denom().is_constant(), "remainder coefficient {c:?}");
            assert!(
                c.numer().rem(q).is_zero(),
                "S does not divide D modulo q: remainder coefficient {c:?}"
            );
        }
    }

    fn algebraic_terms(result: &LogPartResult) -> Vec<(Poly, GenPoly<RationalFn>)> {
        result
            .terms
            .iter()
            .filter_map(|t| match t {
                LogTerm::Algebraic { min_poly, log_arg } => {
                    Some((min_poly.clone(), log_arg.clone().expect("log argument")))
                }
                LogTerm::Rational { .. } => None,
            })
            .collect()
    }

    fn poly(coeffs: &[i64]) -> Poly {
        Poly::from_coeffs(coeffs.iter().map(|&c| rat(c, 1)).collect())
    }

    #[test]
    fn log_argument_has_the_degree_of_the_multiplicity() {
        // ∫ x³/(x⁸ + 1): R(t) = (64t² + 1)⁴ (SymPy 1.14:
        // factor_list(resultant(x**8+1, x**3 - t*diff(x**8+1, x), x), t)),
        // so each residue ±i/8 has a log argument of degree 4:
        // gcd(x⁸ + 1, x³ − α·8x⁷) = x⁴ + 8α.
        let a = poly(&[0, 0, 0, 1]);
        let d = poly(&[1, 0, 0, 0, 0, 0, 0, 0, 1]);
        let result = logarithmic_part(&a, &d);
        let alg = algebraic_terms(&result);
        assert_eq!(alg.len(), 1, "{:?}", result.terms);
        let (q, s) = &alg[0];
        assert_eq!(q.primitive_part(), poly(&[1, 0, 64]));
        assert_eq!(s.degree(), Some(4));
        assert_eq!(s.coeff(0).numer(), &poly(&[0, 8]));
        for k in 1..4 {
            assert!(s.coeff(k).numer().is_zero());
        }
        assert_log_arg_divides(&d, q, s);
    }

    #[test]
    fn log_arguments_of_mixed_factors() {
        // ∫ x/(x⁶ + 1): R(t) = (6t − 1)²·(36t² + 6t + 1)² (SymPy 1.14, as
        // above): a rational residue 1/6 with log argument x² + 1, and a
        // quadratic factor with log arguments of degree 2.
        let a = poly(&[0, 1]);
        let d = poly(&[1, 0, 0, 0, 0, 0, 1]);
        let result = logarithmic_part(&a, &d);
        let rational: Vec<_> = result
            .terms
            .iter()
            .filter_map(|t| match t {
                LogTerm::Rational { coeff, argument } => Some((coeff.clone(), argument.clone())),
                LogTerm::Algebraic { .. } => None,
            })
            .collect();
        assert_eq!(rational, vec![(rat(1, 6), poly(&[1, 0, 1]))]);
        let alg = algebraic_terms(&result);
        assert_eq!(alg.len(), 1);
        let (q, s) = &alg[0];
        assert_eq!(q.primitive_part(), poly(&[1, 6, 36]));
        assert_eq!(s.degree(), Some(2));
        assert_log_arg_divides(&d, q, s);
    }

    #[test]
    fn log_argument_of_a_simple_quadratic_factor() {
        // ∫ 1/(x² + x + 1): q = 3t² + 1, S = x + 3t/2 + 1/2 (the example in
        // SymPy's `ratint_logpart` docstring).
        let a = poly(&[1]);
        let d = poly(&[1, 1, 1]);
        let alg = algebraic_terms(&logarithmic_part(&a, &d));
        assert_eq!(alg.len(), 1);
        let (q, s) = &alg[0];
        assert_eq!(q.primitive_part(), poly(&[1, 0, 3]));
        assert_eq!(s.degree(), Some(1));
        assert_eq!(
            s.coeff(0).numer(),
            &Poly::from_coeffs(vec![rat(1, 2), rat(3, 2)])
        );
        assert_log_arg_divides(&d, q, s);
    }

    #[test]
    fn extract_linear_root_basic() {
        // t - 3  →  root = 3
        let p = Poly::from_coeffs(vec![rat(-3, 1), rat(1, 1)]);
        assert_eq!(extract_linear_root(&p), rat(3, 1));
    }

    #[test]
    fn extract_linear_root_scaled() {
        // 2t + 6  →  root = -3
        let p = Poly::from_coeffs(vec![rat(6, 1), rat(2, 1)]);
        assert_eq!(extract_linear_root(&p), rat(-3, 1));
    }

    #[test]
    fn extract_linear_root_fractional() {
        // 3t - 1  →  root = 1/3
        let p = Poly::from_coeffs(vec![rat(-1, 1), rat(3, 1)]);
        assert_eq!(extract_linear_root(&p), rat(1, 3));
    }
}
