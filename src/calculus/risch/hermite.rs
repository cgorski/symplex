//! Hermite reduction — Mack's linear version.
//!
//! Given a proper rational function `A(x)/D(x)` (with `deg(A) < deg(D)`),
//! Hermite reduction extracts the *rational part* of the integral using
//! only polynomial GCD operations — no algebraic number field extensions
//! are needed.
//!
//! The result is:
//!
//! ```text
//!   ∫ A/D dx  =  g_numer/g_denom  +  ∫ h_numer/h_denom dx
//! ```
//!
//! where `h_denom` is **square-free**.  The remaining integral with
//! square-free denominator is then handled by the Rothstein-Trager
//! algorithm ([`super::rothstein_trager`]).
//!
//! # Algorithm
//!
//! Follows Bronstein §2.3 and SymPy's `hermite_reduce` in `risch.py`
//! (BSD-3-Clause; notice in `THIRD-PARTY-NOTICES.md`).
//!
//! 1. Compute `D₋ = gcd(D, D')` — this captures all repeated factors.
//! 2. Compute `D* = D / D₋` — the square-free cofactor.
//! 3. While `deg(D₋) > 0`:
//!    - `D₋₂ = gcd(D₋, D₋')`
//!    - `D₋* = D₋ / D₋₂`
//!    - Solve via extended GCD:  `B · (−D* · D₋' / D₋) + C · D₋* = A`
//!    - Accumulate:  `g += B / D₋`
//!    - Update:  `A ← C − B' · (D* / D₋*)`,  `D₋ ← D₋₂`
//! 4. Return `(g, A / D*)`.
//!
//! # References
//!
//! - Bronstein, *Symbolic Integration I*, §2.3
//! - Mack, "On the computation of the Hermite normal form", 1975
//! - Geddes, Czapor, Labahn, *Algorithms for Computer Algebra*, §11.3

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};

use crate::poly::dense::Poly;

/// Result of Hermite reduction.
#[derive(Clone, Debug)]
pub struct HermiteResult {
    /// Rational part numerator: the `g` in `∫ A/D = g + ∫ h`.
    pub g_numer: Poly,
    /// Rational part denominator.
    pub g_denom: Poly,
    /// Remaining integrand numerator (denominator is square-free).
    pub h_numer: Poly,
    /// Remaining integrand denominator (**guaranteed square-free**).
    pub h_denom: Poly,
}

/// Perform Hermite reduction on the rational function `a / d`.
///
/// The input does not need to be proper — if `deg(a) ≥ deg(d)`, the
/// polynomial part is extracted first via Euclidean division and added
/// to the remainder numerator (since `∫ poly dx` is trivially a
/// polynomial and doesn't need Hermite reduction).
///
/// `None` if `d` is zero.  The integrator never passes one (it rejects a
/// zero denominator first); up to 0.28 this was a runtime `assert!`.
pub fn hermite_reduce(a: &Poly, d: &Poly) -> Option<HermiteResult> {
    (!d.is_zero()).then(|| hermite_reduce_nonzero(a, d))
}

/// [`hermite_reduce`] for a nonzero `d`.
fn hermite_reduce_nonzero(a: &Poly, d: &Poly) -> HermiteResult {
    // Make d monic for consistent GCD computations (lc(d) is the last
    // coefficient; d is nonzero, so it exists).
    let d_lc = d.coeff(d.coeffs().len() - 1);
    let d_monic = d.make_monic();
    // Scale a by 1/lc(d) to compensate.
    let inv_lc = Ratio::one() / &d_lc;
    let a_scaled = a.scale(&inv_lc);

    // Euclidean division: a_scaled = q * d_monic + a_proper
    // where deg(a_proper) < deg(d_monic).
    let (poly_part, a_proper) = a_scaled.div_rem(&d_monic);

    // The polynomial part integrates trivially; we'll add it to the
    // Hermite result later.  For now, work with the proper fraction.
    hermite_reduce_proper(&a_proper, &d_monic, &poly_part)
}

/// Core Hermite reduction for a proper fraction `a / d` where `d` is monic
/// and `deg(a) < deg(d)`.  `poly_part` is carried through to the result.
fn hermite_reduce_proper(a: &Poly, d: &Poly, poly_part: &Poly) -> HermiteResult {
    // D₋ = gcd(D, D')
    let d_prime = d.derivative();
    let mut d_minus = Poly::gcd(d, &d_prime);

    // D* = D / D₋  (the square-free part of D)
    let d_star = d.div(&d_minus);

    // If D₋ is constant, D is already square-free — nothing to reduce.
    if d_minus.degree().unwrap_or(0) == 0 {
        // Incorporate the polynomial part into the remainder.
        // ∫ (poly_part + a/d) dx — the poly part's integral is trivial,
        // but for the Hermite result we report it as part of g (= 0)
        // and h = (poly_part * d + a) / d.
        //
        // Actually, we return them separately: g = integral of poly_part
        // (which the caller handles), and h = a/d.
        //
        // Simplest: return g = 0, h = original proper fraction, and
        // let the caller handle poly_part separately.
        return HermiteResult {
            g_numer: integrate_poly(poly_part),
            g_denom: Poly::from_int(1),
            h_numer: a.clone(),
            h_denom: d.clone(),
        };
    }

    // Accumulator for the rational part g = g_numer / g_denom.
    let mut g_numer = Poly::zero();
    let mut g_denom = Poly::from_int(1);

    let mut a_curr = a.clone();

    // Iteratively reduce: at each step, D₋ has one fewer layer of
    // repeated factors.
    while d_minus.degree().unwrap_or(0) > 0 {
        let d_minus_prime = d_minus.derivative();
        let d_minus_2 = Poly::gcd(&d_minus, &d_minus_prime);
        let d_minus_star = d_minus.div(&d_minus_2);

        // We need to solve:
        //   B · (-D* · D₋' / D₋) + C · D₋* = A
        //
        // Let lhs_coeff = (-D* · D₋') / D₋  (exact division).
        let neg_dstar_dminus_prime = -&(&d_star * &d_minus_prime);
        let lhs_coeff = neg_dstar_dminus_prime.div(&d_minus);

        // Extended GCD: find (s, t, g) such that s·lhs_coeff + t·D₋* = g.
        let num_integer::ExtendedGcd {
            gcd: gcd_val,
            x: s,
            y: t,
        } = Poly::extended_gcd(&lhs_coeff, &d_minus_star);

        // Check that gcd divides a_curr (it should, by the theory).
        let (scale, rem) = a_curr.div_rem(&gcd_val);
        if !rem.is_zero() {
            // This shouldn't happen for a valid rational function.
            // If it does, bail out — return what we have so far.
            break;
        }

        // B = s · scale, C = t · scale
        let b_full = &s * &scale;
        let _c_full = &t * &scale;

        // Reduce B mod D₋* to ensure deg(B) < deg(D₋*).
        let (_, b) = b_full.div_rem(&d_minus_star);

        // Recompute C from: A = B · lhs_coeff + C · D₋*
        // C = (A - B · lhs_coeff) / D₋*
        let b_times_lhs = &b * &lhs_coeff;
        let numerator_for_c = &a_curr - &b_times_lhs;
        let c = numerator_for_c.div(&d_minus_star);

        // Accumulate rational part: g += B / D₋
        // g = (g_numer · D₋ + B · g_denom) / (g_denom · D₋)
        g_numer = &(&g_numer * &d_minus) + &(&b * &g_denom);
        g_denom = &g_denom * &d_minus;

        // Simplify g by cancelling GCD.
        let g_gcd = Poly::gcd(&g_numer, &g_denom);
        if g_gcd.degree().unwrap_or(0) > 0 || !g_gcd.is_constant() {
            g_numer = g_numer.div(&g_gcd);
            g_denom = g_denom.div(&g_gcd);
        }

        // Update integrand: A ← C − B' · (D* / D₋*)
        let b_prime = b.derivative();
        let dstar_over_dmstar = d_star.div(&d_minus_star);
        a_curr = &c - &(&b_prime * &dstar_over_dmstar);

        // Move to next level: D₋ ← D₋₂
        d_minus = d_minus_2;
    }

    // Make g_denom monic for cleanliness.
    if !g_denom.is_zero()
        && let Some(lc) = g_denom.leading_coeff()
        && !lc.is_one()
    {
        let inv = Ratio::one() / lc;
        g_numer = g_numer.scale(&inv);
        g_denom = g_denom.scale(&inv);
    }

    // Add the polynomial part integral to g.
    if !poly_part.is_zero() {
        let poly_integral = integrate_poly(poly_part);
        // g += poly_integral / 1
        // g = (g_numer + poly_integral * g_denom) / g_denom
        g_numer = &g_numer + &(&poly_integral * &g_denom);
    }

    HermiteResult {
        g_numer,
        g_denom,
        h_numer: a_curr,
        h_denom: d_star,
    }
}

/// Integrate a polynomial term-by-term: ∫ Σ aₖ xᵏ dx = Σ aₖ/(k+1) x^(k+1).
fn integrate_poly(p: &Poly) -> Poly {
    if p.is_zero() {
        return Poly::zero();
    }
    let coeffs = p.coeffs();
    let mut result = vec![Ratio::zero()]; // constant term = 0
    for (k, c) in coeffs.iter().enumerate() {
        let k_plus_1 = Ratio::from_integer(BigInt::from((k + 1) as i64));
        result.push(c / &k_plus_1);
    }
    Poly::from_coeffs(result)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::numeric::Q;

    fn rat(n: i64, d: i64) -> Q {
        Ratio::new(BigInt::from(n), BigInt::from(d))
    }

    /// Verify the Hermite reduction by checking the FTC:
    ///   d/dx(g) + h  ==  A/D
    /// where g = g_numer/g_denom and h = h_numer/h_denom.
    fn verify_hermite(a: &Poly, d: &Poly, result: &HermiteResult) {
        // d/dx(g_numer / g_denom)
        //   = (g_numer' * g_denom - g_numer * g_denom') / g_denom^2
        let gn_prime = result.g_numer.derivative();
        let gd_prime = result.g_denom.derivative();
        let dg_numer = &(&gn_prime * &result.g_denom) - &(&result.g_numer * &gd_prime);
        let dg_denom = &result.g_denom * &result.g_denom;

        // d/dx(g) + h = dg_numer/dg_denom + h_numer/h_denom
        //             = (dg_numer * h_denom + h_numer * dg_denom) / (dg_denom * h_denom)
        let sum_numer = &(&dg_numer * &result.h_denom) + &(&result.h_numer * &dg_denom);
        let sum_denom = &dg_denom * &result.h_denom;

        // Compare with A/D: should have sum_numer * D == A * sum_denom
        // (cross-multiply to avoid division).
        let lhs = &sum_numer * d;
        let rhs = &(a * &sum_denom);

        // Reduce both to monic form for comparison.
        // Actually, just check lhs - rhs == 0.
        let diff = &lhs - rhs;
        assert!(
            diff.is_zero(),
            "FTC verification failed:\n  d/dx(g) + h ≠ A/D\n  diff = {diff}"
        );
    }

    #[test]
    fn a_zero_denominator_is_none_not_a_panic() {
        // Up to 0.28 both asserted (a panic in release builds too).
        let a = Poly::from_int(1);
        assert!(hermite_reduce(&a, &Poly::zero()).is_none());
        assert!(super::super::rothstein_trager::logarithmic_part(&a, &Poly::zero()).is_none());
        let rde = super::super::rde::solve_risch_de_rational(&a, &Poly::zero(), &a, &a);
        assert!(matches!(
            rde,
            super::super::rde::RdeResult::NotImplemented(_)
        ));
    }

    #[test]
    fn hermite_reduce_already_squarefree() {
        // ∫ 1/(x^2 - 1) dx — denominator is already square-free.
        // Hermite reduction should be a no-op (g = 0).
        let a = Poly::from_int(1);
        let d = Poly::from_coeffs(vec![rat(-1, 1), rat(0, 1), rat(1, 1)]); // x^2 - 1

        let result = hermite_reduce(&a, &d).unwrap();
        // h should still be 1/(x^2 - 1)
        assert_eq!(result.h_numer.degree(), Some(0));
        assert!(result.h_denom.degree().unwrap_or(0) >= 2);
        // Verify FTC
        verify_hermite(&a, &d, &result);
    }

    #[test]
    fn hermite_reduce_one_over_x_squared() {
        // ∫ 1/x^2 dx = -1/x
        // D = x^2 has repeated root at 0.
        let a = Poly::from_int(1);
        let d = Poly::from_coeffs(vec![rat(0, 1), rat(0, 1), rat(1, 1)]); // x^2

        let result = hermite_reduce(&a, &d).unwrap();
        // g should be -1/x: g_numer = -1, g_denom = x (or equivalent)
        // h should be 0/1 (no logarithmic part for 1/x^2)
        assert!(
            result.h_numer.is_zero(),
            "∫ 1/x^2: log part should be zero, got h = {}/{}",
            result.h_numer,
            result.h_denom
        );
        verify_hermite(&a, &d, &result);
    }

    #[test]
    fn hermite_reduce_one_over_x_plus_1_squared() {
        // ∫ 1/(x+1)^2 dx = -1/(x+1)
        // D = (x+1)^2 = x^2 + 2x + 1
        let a = Poly::from_int(1);
        let d = Poly::from_coeffs(vec![rat(1, 1), rat(2, 1), rat(1, 1)]); // x^2 + 2x + 1

        let result = hermite_reduce(&a, &d).unwrap();
        // h should be zero (no logarithmic part).
        assert!(
            result.h_numer.is_zero(),
            "∫ 1/(x+1)^2: log part should be zero, got h = {}/{}",
            result.h_numer,
            result.h_denom
        );
        verify_hermite(&a, &d, &result);
    }

    #[test]
    fn hermite_reduce_one_over_x_cubed() {
        // ∫ 1/x^3 dx = -1/(2x^2)
        // D = x^3 (triple root at 0).
        let a = Poly::from_int(1);
        let d = Poly::from_coeffs(vec![rat(0, 1), rat(0, 1), rat(0, 1), rat(1, 1)]); // x^3

        let result = hermite_reduce(&a, &d).unwrap();
        assert!(result.h_numer.is_zero(), "∫ 1/x^3: log part should be zero");
        verify_hermite(&a, &d, &result);
    }

    #[test]
    fn hermite_reduce_repeated_quadratic() {
        // ∫ 1/(x^2+1)^2 dx — repeated irreducible quadratic.
        // D = x^4 + 2x^2 + 1 = (x^2 + 1)^2
        let a = Poly::from_int(1);
        let d = Poly::from_coeffs(vec![rat(1, 1), rat(0, 1), rat(2, 1), rat(0, 1), rat(1, 1)]); // x^4 + 2x^2 + 1

        let result = hermite_reduce(&a, &d).unwrap();
        // Hermite should extract a rational part; remainder has denom (x^2+1).
        let h_deg = result.h_denom.degree().unwrap_or(0);
        assert!(
            h_deg <= 2,
            "∫ 1/(x^2+1)^2: remainder denom should be degree ≤ 2 (square-free), got {h_deg}"
        );
        verify_hermite(&a, &d, &result);
    }

    #[test]
    fn hermite_reduce_2x_plus_3_over_x_plus_1_cubed() {
        // ∫ (2x+3)/(x+1)^3 dx
        // D = (x+1)^3 = x^3 + 3x^2 + 3x + 1
        let a = Poly::from_coeffs(vec![rat(3, 1), rat(2, 1)]); // 2x + 3
        let d = Poly::from_coeffs(vec![rat(1, 1), rat(3, 1), rat(3, 1), rat(1, 1)]); // x^3 + 3x^2 + 3x + 1

        let result = hermite_reduce(&a, &d).unwrap();
        verify_hermite(&a, &d, &result);
    }

    #[test]
    fn hermite_reduce_improper_fraction() {
        // ∫ x^3 / (x+1)^2 dx — improper fraction (deg(A) > deg(D))
        // Should extract polynomial part first, then reduce remainder.
        let a = Poly::from_coeffs(vec![rat(0, 1), rat(0, 1), rat(0, 1), rat(1, 1)]); // x^3
        let d = Poly::from_coeffs(vec![rat(1, 1), rat(2, 1), rat(1, 1)]); // (x+1)^2

        let result = hermite_reduce(&a, &d).unwrap();
        verify_hermite(&a, &d, &result);
    }

    #[test]
    fn hermite_reduce_with_mixed_roots() {
        // D = x^2 * (x - 1) = x^3 - x^2  (one simple root, one double root)
        // A = 1
        let a = Poly::from_int(1);
        let d = Poly::from_coeffs(vec![rat(0, 1), rat(0, 1), rat(-1, 1), rat(1, 1)]); // x^3 - x^2

        let result = hermite_reduce(&a, &d).unwrap();
        // Remainder denominator should be square-free part: x*(x-1) = x^2 - x
        let h_deg = result.h_denom.degree().unwrap_or(0);
        assert!(
            h_deg <= 2,
            "remainder denom degree should be ≤ 2, got {h_deg}"
        );
        verify_hermite(&a, &d, &result);
    }

    #[test]
    fn integrate_poly_basic() {
        // ∫ (3x^2 + 2x + 1) dx = x^3 + x^2 + x
        let p = Poly::from_coeffs(vec![rat(1, 1), rat(2, 1), rat(3, 1)]);
        let integral = integrate_poly(&p);
        // Should be 0 + x + x^2 + x^3
        assert_eq!(integral.degree(), Some(3));
        assert_eq!(integral.coeff(0), rat(0, 1));
        assert_eq!(integral.coeff(1), rat(1, 1));
        assert_eq!(integral.coeff(2), rat(1, 1));
        assert_eq!(integral.coeff(3), rat(1, 1));
    }

    #[test]
    fn integrate_poly_constant() {
        // ∫ 5 dx = 5x
        let p = Poly::from_int(5);
        let integral = integrate_poly(&p);
        assert_eq!(integral.degree(), Some(1));
        assert_eq!(integral.coeff(1), rat(5, 1));
    }

    #[test]
    fn integrate_poly_zero() {
        let p = Poly::zero();
        let integral = integrate_poly(&p);
        assert!(integral.is_zero());
    }

    #[test]
    fn hermite_h_denom_is_squarefree() {
        // For any input, the output h_denom should be square-free:
        // gcd(h_denom, h_denom') should have degree 0.
        let test_cases: Vec<(Poly, Poly)> = vec![
            // 1/(x+1)^2
            (
                Poly::from_int(1),
                Poly::from_coeffs(vec![rat(1, 1), rat(2, 1), rat(1, 1)]),
            ),
            // 1/x^3
            (
                Poly::from_int(1),
                Poly::from_coeffs(vec![rat(0, 1), rat(0, 1), rat(0, 1), rat(1, 1)]),
            ),
            // (x+2)/(x^2(x-1)^2) = (x+2)/(x^4 - 2x^3 + x^2)
            (
                Poly::from_coeffs(vec![rat(2, 1), rat(1, 1)]),
                Poly::from_coeffs(vec![rat(0, 1), rat(0, 1), rat(1, 1), rat(-2, 1), rat(1, 1)]),
            ),
        ];

        for (a, d) in &test_cases {
            let result = hermite_reduce(a, d).unwrap();
            if !result.h_numer.is_zero() {
                let h_d_prime = result.h_denom.derivative();
                let g = Poly::gcd(&result.h_denom, &h_d_prime);
                assert!(
                    g.degree().unwrap_or(0) == 0,
                    "h_denom should be square-free for A={a}, D={d}, got gcd(h_denom, h_denom') = {g}"
                );
            }
        }
    }
}
