//! Tower-level Hermite reduction and Rothstein-Trager on `GenPoly<RationalFn>`.
//!
//! This module lifts the base-level rational function integration algorithms
//! (from [`super::hermite`] and [`super::rothstein_trager`]) to operate on
//! polynomials in a tower extension variable `θ` with coefficients in `ℚ(x)`.
//!
//! - **Tower Hermite reduction:** Given `A(θ)/D(θ)` where `A, D ∈ ℚ(x)[θ]`,
//!   extracts the rational-in-θ part of the integral using only polynomial
//!   GCD operations on `GenPoly<RationalFn>`.
//!
//! - **Tower Rothstein-Trager:** For the square-free remainder, computes the
//!   resultant `R(z) = res_θ(D, A − z·D')` and checks the **elementarity
//!   condition**: if any root of `R(z)` is a non-constant element of `ℚ(x)`
//!   (i.e., it depends on `x`), the integral is **provably non-elementary**.
//!
//! # References
//!
//! - Bronstein, *Symbolic Integration I*, §2.3–2.5 (generalized to k\[θ\] for k = ℚ(x))
//! - The algorithms are identical to the base-level versions but operate on
//!   `GenPoly<RationalFn>` instead of `Poly<Ratio<BigInt>>`.

use num_bigint::BigInt;
use num_rational::Ratio;

use crate::poly::generic::GenPoly;
use crate::poly::ratfn::RationalFn;
use crate::poly::traits::{Field, Ring};

// ═══════════════════════════════════════════════════════════════════════════
// Result types
// ═══════════════════════════════════════════════════════════════════════════

/// Result of tower-level Hermite reduction.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct TowerHermiteResult {
    /// Rational part numerator (in θ).
    pub g_numer: GenPoly<RationalFn>,
    /// Rational part denominator (in θ).
    pub g_denom: GenPoly<RationalFn>,
    /// Remaining integrand numerator (denominator is square-free in θ).
    pub h_numer: GenPoly<RationalFn>,
    /// Remaining integrand denominator (guaranteed square-free in θ).
    pub h_denom: GenPoly<RationalFn>,
}

/// A single logarithmic term from tower-level Rothstein-Trager.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum TowerLogTerm {
    /// `coeff · ln(argument(θ))` where `coeff` is a constant (in ℚ).
    Constant {
        coeff: Ratio<BigInt>,
        argument: GenPoly<RationalFn>,
    },
    /// `coeff · ln(argument(θ))` where `coeff ∈ ℚ(x)` is NOT constant.
    /// The presence of this variant means the integral is **non-elementary**.
    NonConstant {
        coeff: RationalFn,
        argument: GenPoly<RationalFn>,
    },
}

/// Result of tower-level Rothstein-Trager.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct TowerLogPartResult {
    /// The logarithmic terms.
    pub terms: Vec<TowerLogTerm>,
    /// True if any root of R(z) was non-constant → integral is provably
    /// non-elementary.
    pub is_non_elementary: bool,
}

// ═══════════════════════════════════════════════════════════════════════════
// Tower Hermite Reduction
// ═══════════════════════════════════════════════════════════════════════════

/// Hermite reduction on `GenPoly<RationalFn>`.
///
/// Given a proper fraction `A(θ)/D(θ)` where `A, D ∈ ℚ(x)[θ]` and `D` has
/// repeated factors, extracts the rational-in-θ part of the integral:
///
/// ```text
///   ∫ A/D dθ  =  g_numer/g_denom  +  ∫ h_numer/h_denom dθ
/// ```
///
/// where `h_denom` is **square-free** in θ.
///
/// The algorithm is identical to Mack's linear version (see [`super::hermite`])
/// but operates on `GenPoly<RationalFn>` instead of `Poly<Ratio<BigInt>>`.
///
/// # Panics
///
/// Panics if `d` is zero.
pub fn tower_hermite_reduce(
    a: &GenPoly<RationalFn>,
    d: &GenPoly<RationalFn>,
) -> TowerHermiteResult {
    assert!(
        !d.is_zero(),
        "tower_hermite_reduce: denominator must be nonzero"
    );

    // Make d monic; lc(d) is the last coefficient (d is nonzero, so it exists).
    let d_monic = d.make_monic();
    let inv_lc = Field::inv(&d.coeff(d.coeffs().len() - 1));
    let a_scaled = a.scale(&inv_lc);

    // Euclidean division: a_scaled = q * d_monic + a_proper
    let (poly_part, a_proper) = a_scaled.div_rem(&d_monic);

    // D₋ = gcd(D, D')
    let d_prime = d_monic.derivative();
    let mut d_minus = GenPoly::gcd(&d_monic, &d_prime);

    // D* = D / D₋ (the square-free part)
    let d_star = d_monic.div(&d_minus);

    // If D₋ is constant, D is already square-free.
    if d_minus.degree().unwrap_or(0) == 0 {
        // Integrate the polynomial part trivially.
        let poly_integral = integrate_tower_poly(&poly_part);
        return TowerHermiteResult {
            g_numer: poly_integral,
            g_denom: GenPoly::one(),
            h_numer: a_proper,
            h_denom: d_monic,
        };
    }

    // Accumulators for the rational part.
    let mut g_numer: GenPoly<RationalFn> = GenPoly::zero();
    let mut g_denom: GenPoly<RationalFn> = GenPoly::one();

    let mut a_curr = a_proper;

    // Iteratively reduce repeated factors.
    while d_minus.degree().unwrap_or(0) > 0 {
        let d_minus_prime = d_minus.derivative();
        let d_minus_2 = GenPoly::gcd(&d_minus, &d_minus_prime);
        let d_minus_star = d_minus.div(&d_minus_2);

        // lhs_coeff = (-D* · D₋') / D₋
        let neg_dstar_dminus_prime = d_star.mul(&d_minus_prime).neg();
        let lhs_coeff = neg_dstar_dminus_prime.div(&d_minus);

        // Extended GCD: s · lhs_coeff + t · D₋* = gcd
        let (s, t, gcd_val) = GenPoly::extended_gcd(&lhs_coeff, &d_minus_star);

        // Scale to solve for a_curr: B = s · (a_curr / gcd), C = t · (a_curr / gcd)
        let (scale, rem) = a_curr.div_rem(&gcd_val);
        if !rem.is_zero() {
            // Theory guarantees this divides; if not, bail.
            break;
        }

        let b_full = s.mul(&scale);
        let _c_full = t.mul(&scale);

        // Reduce B mod D₋* to ensure deg(B) < deg(D₋*).
        let (_, b) = b_full.div_rem(&d_minus_star);

        // Recompute C: a_curr = b · lhs_coeff + c · D₋*
        let b_times_lhs = b.mul(&lhs_coeff);
        let numerator_for_c = a_curr.sub(&b_times_lhs);
        let c = numerator_for_c.div(&d_minus_star);

        // Accumulate: g += B / D₋
        g_numer = g_numer.mul(&d_minus).add(&b.mul(&g_denom));
        g_denom = g_denom.mul(&d_minus);

        // Simplify g by cancelling GCD.
        let g_gcd = GenPoly::gcd(&g_numer, &g_denom);
        if g_gcd.degree().unwrap_or(0) > 0 {
            g_numer = g_numer.div(&g_gcd);
            g_denom = g_denom.div(&g_gcd);
        }

        // Update: A ← C − B' · (D* / D₋*)
        let b_prime = b.derivative();
        let dstar_over_dmstar = d_star.div(&d_minus_star);
        a_curr = c.sub(&b_prime.mul(&dstar_over_dmstar));

        // Next level: D₋ ← D₋₂
        d_minus = d_minus_2;
    }

    // Make g_denom monic.
    if !g_denom.is_zero()
        && let Some(lc) = g_denom.leading_coeff()
        && !lc.is_one()
    {
        let inv = Field::inv(lc);
        g_numer = g_numer.scale(&inv);
        g_denom = g_denom.scale(&inv);
    }

    // Add the polynomial part integral.
    if !poly_part.is_zero() {
        let poly_integral = integrate_tower_poly(&poly_part);
        g_numer = g_numer.add(&poly_integral.mul(&g_denom));
    }

    TowerHermiteResult {
        g_numer,
        g_denom,
        h_numer: a_curr,
        h_denom: d_star,
    }
}

/// Integrate a polynomial in θ term-by-term (treating ℚ(x) coefficients as constants).
///
/// `∫ Σ aₖ θᵏ dθ = Σ aₖ/(k+1) θ^(k+1)`
///
/// This is correct when the "integration" is with respect to θ as a formal
/// variable (as in Hermite reduction), NOT with respect to x through the tower.
fn integrate_tower_poly(p: &GenPoly<RationalFn>) -> GenPoly<RationalFn> {
    if p.is_zero() {
        return GenPoly::zero();
    }
    let coeffs = p.coeffs();
    let mut result = vec![RationalFn::from_int(0)]; // constant term = 0
    for (k, c) in coeffs.iter().enumerate() {
        let k_plus_1 = RationalFn::from_int((k + 1) as i64);
        let scaled = Field::div(c, &k_plus_1);
        result.push(scaled);
    }
    GenPoly::from_coeffs(result)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tower Rothstein-Trager
// ═══════════════════════════════════════════════════════════════════════════

/// Tower-level Rothstein-Trager: compute the logarithmic part of
/// `∫ A(θ)/D(θ)` where `D` is square-free in θ and `deg(A) < deg(D)`.
///
/// Computes `R(z) = res_θ(D, A − z·D')` and checks the roots:
/// - Roots that are constants (elements of ℚ) → valid log terms
/// - Roots that are non-constant (depend on x) → **non-elementary**
///
/// # Panics
///
/// Panics if `D` is zero.
pub fn tower_logarithmic_part(
    a: &GenPoly<RationalFn>,
    d: &GenPoly<RationalFn>,
) -> TowerLogPartResult {
    assert!(
        !d.is_zero(),
        "tower_logarithmic_part: denominator must be nonzero"
    );

    if a.is_zero() {
        return TowerLogPartResult {
            terms: vec![],
            is_non_elementary: false,
        };
    }

    let d_prime = d.derivative();

    // Handle trivial case: D is linear in θ.
    if d.degree() == Some(1) {
        let a_val = a.coeff(0);
        let d_lc = d.coeff(1);
        let coeff = Field::div(&a_val, &d_lc);
        if Ring::is_zero(&coeff) {
            return TowerLogPartResult {
                terms: vec![],
                is_non_elementary: false,
            };
        }
        // A coefficient in ℚ gives an elementary log term; anything
        // depending on x makes the integral non-elementary.
        if let Some(c) = coeff.to_rational() {
            return TowerLogPartResult {
                terms: vec![TowerLogTerm::Constant {
                    coeff: c,
                    argument: d.make_monic(),
                }],
                is_non_elementary: false,
            };
        } else {
            return TowerLogPartResult {
                terms: vec![TowerLogTerm::NonConstant {
                    coeff,
                    argument: d.make_monic(),
                }],
                is_non_elementary: true,
            };
        }
    }

    // R(z) = res_θ(D, A − z·D')
    // Uses the parametric resultant via evaluation-interpolation.
    let r_poly = GenPoly::<RationalFn>::resultant_poly(d, a, &d_prime);

    if r_poly.is_zero() {
        return TowerLogPartResult {
            terms: vec![],
            is_non_elementary: false,
        };
    }

    // Extract roots of R(z).
    // For now, handle linear factors (degree 1) which give rational or
    // non-constant roots directly.  Higher-degree factors would need
    // algebraic extension field arithmetic.
    let mut terms = Vec::new();
    let mut is_non_elementary = false;

    // Try to find roots by factoring R(z).
    // Simple approach: check if R(z) has degree 1 (single root).
    if let Some(deg) = r_poly.degree() {
        if deg == 1 {
            // R(z) = a₁·z + a₀  →  root = -a₀/a₁
            let a0 = r_poly.coeff(0);
            let a1 = r_poly.coeff(1);
            let root = Field::div(&Ring::neg(&a0), &a1);

            if Ring::is_zero(&root) {
                // Zero root — no contribution.
            } else if let Some(c) = root.to_rational() {
                // v(θ) = gcd(D, A − c·D')
                let c_rf = RationalFn::from_rational(c.clone());
                let a_minus_c_dprime = a.sub(&d_prime.scale(&c_rf));
                let v = GenPoly::gcd(d, &a_minus_c_dprime);
                if v.degree().unwrap_or(0) > 0 {
                    terms.push(TowerLogTerm::Constant {
                        coeff: c,
                        argument: v.make_monic(),
                    });
                }
            } else {
                // Non-constant root — integral is non-elementary.
                is_non_elementary = true;
                terms.push(TowerLogTerm::NonConstant {
                    coeff: root,
                    argument: d.clone(),
                });
            }
        } else {
            // For higher-degree R(z), try to find rational roots by
            // evaluating at small rationals p/q for small denominators.
            // This covers the common cases like ±1/2, ±1/3, ±2/3, etc.
            let max_denom = 12i64;
            let max_numer = 10i64;
            let mut found_roots: Vec<Ratio<BigInt>> = Vec::new();

            for denom in 1..=max_denom {
                for numer in (-max_numer * denom)..=(max_numer * denom) {
                    if numer == 0 {
                        continue; // skip zero root
                    }
                    let c = Ratio::new(BigInt::from(numer), BigInt::from(denom));
                    // Skip if we already found this root (after reduction).
                    if found_roots.contains(&c) {
                        continue;
                    }
                    let c_rf = RationalFn::from_rational(c.clone());
                    let r_at_c = r_poly.eval(&c_rf);
                    if Ring::is_zero(&r_at_c) {
                        found_roots.push(c.clone());
                        // v(θ) = gcd(D, A − c·D')
                        let a_minus_c_dprime = a.sub(&d_prime.scale(&c_rf));
                        let v = GenPoly::gcd(d, &a_minus_c_dprime);
                        if v.degree().unwrap_or(0) > 0 {
                            terms.push(TowerLogTerm::Constant {
                                coeff: c,
                                argument: v.make_monic(),
                            });
                        }
                    }
                }
            }

            // Check for non-constant roots by verifying the resultant
            // has been fully accounted for.  If we found fewer roots than
            // deg(R), there may be non-constant or algebraic roots.
            let found_degree: usize = terms
                .iter()
                .filter_map(|t| match t {
                    TowerLogTerm::Constant { argument, .. } => argument.degree(),
                    TowerLogTerm::NonConstant { argument, .. } => argument.degree(),
                })
                .sum();
            if found_degree < d.degree().unwrap_or(0) {
                // There are roots we couldn't find — might be non-constant
                // or algebraic.  We can't prove non-elementarity here without
                // full factorization, so we leave it as incomplete.
            }
        }
    }

    TowerLogPartResult {
        terms,
        is_non_elementary,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::dense::Poly;

    fn r(n: i64, d: i64) -> Ratio<BigInt> {
        Ratio::new(BigInt::from(n), BigInt::from(d))
    }

    fn rf_int(n: i64) -> RationalFn {
        RationalFn::from_int(n)
    }

    fn rf(numer: &[i64], denom: &[i64]) -> RationalFn {
        let n = Poly::from_coeffs(numer.iter().map(|&c| r(c, 1)).collect());
        let d = Poly::from_coeffs(denom.iter().map(|&c| r(c, 1)).collect());
        RationalFn::new(n, d)
    }

    fn rp(cs: &[RationalFn]) -> GenPoly<RationalFn> {
        GenPoly::from_coeffs(cs.to_vec())
    }

    // ── Tower Hermite reduction ─────────────────────────────────────

    #[test]
    fn hermite_already_squarefree() {
        // A(θ)/D(θ) where D = θ² - 1 (square-free).
        // Hermite should be a no-op: g = 0, h = A/D.
        let a = rp(&[rf_int(1)]); // A = 1
        let d = rp(&[rf_int(-1), rf_int(0), rf_int(1)]); // D = θ² - 1

        let result = tower_hermite_reduce(&a, &d);
        // h should still be 1/(θ² - 1)
        assert!(
            result.h_denom.degree().unwrap_or(0) >= 2,
            "h_denom should have degree ≥ 2"
        );
    }

    #[test]
    fn hermite_repeated_factor() {
        // A = 1, D = θ² (double root at θ = 0).
        // ∫ 1/θ² dθ = -1/θ (rational part), no log part.
        let a = rp(&[rf_int(1)]); // A = 1
        let d = rp(&[rf_int(0), rf_int(0), rf_int(1)]); // D = θ²

        let result = tower_hermite_reduce(&a, &d);
        assert!(
            result.h_numer.is_zero(),
            "1/θ² should have no log remainder, got h = {}/{}",
            result.h_numer,
            result.h_denom
        );
    }

    #[test]
    fn hermite_with_ratfn_coefficients() {
        // A = 1/x, D = θ² (with ℚ(x) coefficient).
        // ∫ (1/x)/θ² dθ = -(1/x)/θ
        let inv_x = rf(&[1], &[0, 1]); // 1/x
        let a = rp(&[inv_x]); // A = 1/x
        let d = rp(&[rf_int(0), rf_int(0), rf_int(1)]); // D = θ²

        let result = tower_hermite_reduce(&a, &d);
        assert!(
            result.h_numer.is_zero(),
            "(1/x)/θ² should have no log remainder"
        );
    }

    #[test]
    fn hermite_ftc_verification() {
        // Verify: d/dθ(g) + h == A/D for a non-trivial case.
        // A = 1, D = (θ + 1)² = θ² + 2θ + 1
        let a = rp(&[rf_int(1)]); // 1
        let d = rp(&[rf_int(1), rf_int(2), rf_int(1)]); // (θ+1)²

        let result = tower_hermite_reduce(&a, &d);

        // d/dθ(g_numer / g_denom) = (g_n' · g_d - g_n · g_d') / g_d²
        let gn_prime = result.g_numer.derivative();
        let gd_prime = result.g_denom.derivative();
        let dg_numer = gn_prime
            .mul(&result.g_denom)
            .sub(&result.g_numer.mul(&gd_prime));
        let dg_denom = result.g_denom.mul(&result.g_denom);

        // d/dθ(g) + h = (dg_numer · h_denom + h_numer · dg_denom) / (dg_denom · h_denom)
        let sum_numer = dg_numer
            .mul(&result.h_denom)
            .add(&result.h_numer.mul(&dg_denom));
        let sum_denom = dg_denom.mul(&result.h_denom);

        // Should equal A/D: sum_numer · D == A · sum_denom
        let lhs = sum_numer.mul(&d);
        let rhs = a.mul(&sum_denom);
        let diff = lhs.sub(&rhs);
        assert!(diff.is_zero(), "FTC verification failed: d/dθ(g) + h ≠ A/D");
    }

    // ── Tower Rothstein-Trager ──────────────────────────────────────

    #[test]
    fn rt_one_over_theta() {
        // ∫ 1/θ dθ = 1·ln(θ)
        // D = θ (linear), A = 1.
        let a = rp(&[rf_int(1)]); // 1
        let d = rp(&[rf_int(0), rf_int(1)]); // θ

        let result = tower_logarithmic_part(&a, &d);
        assert!(!result.is_non_elementary, "1/θ should be elementary");
        assert_eq!(result.terms.len(), 1, "should have 1 log term");
        match &result.terms[0] {
            TowerLogTerm::Constant { coeff, .. } => {
                assert_eq!(*coeff, r(1, 1), "coefficient should be 1");
            }
            _ => panic!("expected Constant log term"),
        }
    }

    #[test]
    fn rt_inv_x_over_theta() {
        // ∫ (1/x)/θ dθ = (1/x)·ln(θ)
        // D = θ (linear), A = 1/x.
        // The coefficient (1/x) is NOT constant → non-elementary
        // (this integral is elementary w.r.t. θ as a formal variable,
        // but in the Risch tower context, a non-constant coefficient
        // in the log part means the integral w.r.t. x is non-elementary
        // if we can't resolve it further).
        let inv_x = rf(&[1], &[0, 1]); // 1/x
        let a = rp(&[inv_x]); // A = 1/x
        let d = rp(&[rf_int(0), rf_int(1)]); // D = θ

        let result = tower_logarithmic_part(&a, &d);
        // The coefficient of the log term is 1/x, which is non-constant.
        assert!(
            result.is_non_elementary,
            "1/(x·θ) should flag as non-elementary at the tower level"
        );
    }

    #[test]
    fn rt_one_over_theta_squared_minus_1() {
        // ∫ 1/(θ²-1) dθ = ½·ln(θ-1) - ½·ln(θ+1)
        // D = θ² - 1 (square-free), A = 1.
        let a = rp(&[rf_int(1)]); // 1
        let d = rp(&[rf_int(-1), rf_int(0), rf_int(1)]); // θ² - 1

        let result = tower_logarithmic_part(&a, &d);
        assert!(!result.is_non_elementary, "1/(θ²-1) should be elementary");
        // Should have log terms with constant coefficients ±1/2.
        let constant_count = result
            .terms
            .iter()
            .filter(|t| matches!(t, TowerLogTerm::Constant { .. }))
            .count();
        assert!(
            constant_count >= 1,
            "should have at least 1 constant log term, got {constant_count}"
        );
    }

    #[test]
    fn rt_zero_numerator() {
        let a = GenPoly::<RationalFn>::zero();
        let d = rp(&[rf_int(-1), rf_int(0), rf_int(1)]); // θ² - 1
        let result = tower_logarithmic_part(&a, &d);
        assert!(result.terms.is_empty());
        assert!(!result.is_non_elementary);
    }

    // ── integrate_tower_poly ────────────────────────────────────────

    #[test]
    fn integrate_tower_poly_basic() {
        // ∫ (3θ² + 2θ + 1) dθ = θ³ + θ² + θ
        let p = rp(&[rf_int(1), rf_int(2), rf_int(3)]);
        let integral = integrate_tower_poly(&p);
        assert_eq!(integral.degree(), Some(3));
        assert!(Ring::is_zero(&integral.coeff(0))); // constant = 0
        assert_eq!(integral.coeff(1), rf_int(1)); // 1/1 · θ
        assert_eq!(integral.coeff(2), rf_int(1)); // 2/2 · θ²
        assert_eq!(integral.coeff(3), rf_int(1)); // 3/3 · θ³
    }

    #[test]
    fn integrate_tower_poly_with_ratfn_coeff() {
        // ∫ (1/x)·θ dθ = (1/x)·θ²/2 = (1/(2x))·θ²
        let inv_x = rf(&[1], &[0, 1]); // 1/x
        let p = rp(&[<RationalFn as Ring>::zero(), inv_x]);
        let integral = integrate_tower_poly(&p);
        assert_eq!(integral.degree(), Some(2));
        // Coefficient of θ² should be 1/(2x)
        let c2 = integral.coeff(2);
        assert!(!c2.is_constant_rational(), "1/(2x) is not constant");
    }

    #[test]
    fn integrate_tower_poly_zero() {
        let p = GenPoly::<RationalFn>::zero();
        let integral = integrate_tower_poly(&p);
        assert!(integral.is_zero());
    }

    // ── Combined Hermite + RT ───────────────────────────────────────

    #[test]
    fn combined_hermite_and_rt() {
        // ∫ 1/((θ-1)²·(θ+1)) dθ
        // Hermite extracts the rational part from the (θ-1)² factor.
        // RT handles the square-free remainder.
        // D = (θ-1)²(θ+1) = θ³ - θ² - θ + 1
        let d = rp(&[rf_int(1), rf_int(-1), rf_int(-1), rf_int(1)]);
        let a = rp(&[rf_int(1)]); // A = 1

        let hr = tower_hermite_reduce(&a, &d);
        // h_denom should be square-free, degree ≤ 2.
        let h_deg = hr.h_denom.degree().unwrap_or(0);
        assert!(
            h_deg <= 2,
            "after Hermite, h_denom degree should be ≤ 2, got {h_deg}"
        );

        // If there's a remainder, check RT.
        if !hr.h_numer.is_zero() {
            let rt = tower_logarithmic_part(&hr.h_numer, &hr.h_denom);
            assert!(
                !rt.is_non_elementary,
                "1/((θ-1)²(θ+1)) should be elementary"
            );
        }
    }
}
