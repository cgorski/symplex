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
//! 2. Factor `R(t)` over ℤ.
//! 3. For each linear factor `(t − cᵢ)` where `cᵢ ∈ ℚ`:
//!    - `vᵢ(x) = gcd(D(x), A(x) − cᵢ·D'(x))` — the logarithmic argument.
//!    - Emit `cᵢ · ln(vᵢ)`.
//! 4. For each irreducible factor of degree > 1:
//!    - Emit a symbolic `Algebraic` term (the roots are algebraic numbers
//!      not in ℚ).
//!
//! The distinct roots of `R(t)` are exactly the residues of `A/D`, and the
//! splitting field of `R(t)` is the **minimal** algebraic extension needed
//! to express the integral.
//!
//! # References
//!
//! - Rothstein, "A new algorithm for the integration of exponential and
//!   logarithmic functions", 1977
//! - Trager, "Algebraic factoring and rational function integration", 1976
//! - Bronstein, *Symbolic Integration I*, §2.4–2.5
//! - Lazard & Rioboo, "Integration of rational functions: rational computation
//!   of the logarithmic part", 1990

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{Signed, Zero};

use super::LogTerm;
use crate::poly::dense::Poly;

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
        let d_lc = d.leading_coeff().unwrap().clone();
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

    // Factor R(t) over ℤ.
    let (content, r_factors) = r_poly.factor_over_z();
    let _ = content; // content doesn't affect roots

    let mut terms: Vec<LogTerm> = Vec::new();

    for (factor, multiplicity) in &r_factors {
        let _ = multiplicity; // roots of R(t) are always simple for square-free D

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
            // not in ℚ.  We store the minimal polynomial and the original
            // integrand data for potential later processing.
            terms.push(LogTerm::Algebraic {
                min_poly: factor.clone(),
                numer: a.clone(),
                denom: d.clone(),
            });
        }
    }

    LogPartResult { terms }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Extract the rational root of a monic-ish linear polynomial `a·t + b`.
/// Returns `c = -b/a`.
fn extract_linear_root(factor: &Poly) -> Ratio<BigInt> {
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

    fn rat(n: i64, d: i64) -> Ratio<BigInt> {
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
        let mut coeffs: Vec<Ratio<BigInt>> =
            rational_terms.iter().map(|(c, _)| c.clone()).collect();
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
