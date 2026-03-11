//! The Risch algorithm for symbolic integration.
//!
//! This module implements the Risch algorithm following Bronstein's
//! *Symbolic Integration I: Transcendental Functions*.  The implementation
//! is layered:
//!
//! - **Phase 1** ([`hermite`]): Hermite reduction — extracts the rational
//!   part of a rational function integral using only polynomial GCD operations.
//! - **Phase 2** ([`rothstein_trager`]): Computes the logarithmic part of
//!   a rational function integral via resultants.
//! - **Phase 3** ([`tower`]): Builds a differential extension tower from
//!   an expression containing `exp`/`ln` subexpressions.
//! - **Phase 4** ([`integrate`]): The full recursive Risch integrator for
//!   transcendental elementary functions.
//! - **Phase 5** ([`rde`]): Risch differential equation solver (`y' + fy = g`).
//!
//! All core algorithms operate at the [`Poly`](crate::poly::dense::Poly) /
//! `Ratio<BigInt>` level.  The [`try_risch_rational`] function bridges
//! from the arena world to the polynomial world using the existing
//! [`polybridge`](crate::poly::polybridge) infrastructure.

pub mod hermite;
pub mod integrate;
pub mod rde;
pub mod rothstein_trager;
pub mod tower;
pub mod tower_integrate;

use std::cell::Cell;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::One;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::poly::dense::Poly;

// Recursion guard: prevents try_risch_rational from re-entering itself
// when it calls the heuristic integrator on an algebraic remainder.
//
// Uses a RAII pattern so the guard is reset even if the integration
// panics — the Drop impl runs during unwinding.  This is also safe
// with tokio: our integrate path is entirely synchronous (no .await
// points), so a single call completes without yielding the thread.
thread_local! {
    static RISCH_GUARD: Cell<bool> = const { Cell::new(false) };
}

/// RAII guard that sets `RISCH_GUARD` to `true` on creation and
/// resets it to `false` on drop (including during panic unwinding).
struct RischRecursionGuard;

impl RischRecursionGuard {
    /// Try to enter the Risch rational integration path.
    ///
    /// Returns `Some(guard)` if we're not already inside a Risch call.
    /// Returns `None` if we're already inside (recursion detected).
    /// The guard resets the flag when dropped.
    fn enter() -> Option<Self> {
        RISCH_GUARD.with(|g| {
            if g.get() {
                None // already inside — recursion detected
            } else {
                g.set(true);
                Some(RischRecursionGuard)
            }
        })
    }
}

impl Drop for RischRecursionGuard {
    fn drop(&mut self) {
        RISCH_GUARD.with(|g| g.set(false));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public result types
// ═══════════════════════════════════════════════════════════════════════════

/// A single logarithmic term in the integral.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum LogTerm {
    /// `coeff * ln(argument(x))` where `coeff` is rational.
    Rational {
        coeff: Ratio<BigInt>,
        argument: Poly,
    },
    /// Symbolic sum over roots of a minimal polynomial:
    /// `Σ_{α: min_poly(α)=0} α * ln(gcd(denom, numer - α·denom'))`.
    ///
    /// Used when the resultant has irreducible factors of degree > 1
    /// whose roots are algebraic numbers not in ℚ.
    Algebraic {
        min_poly: Poly,
        /// The original numerator and denominator, for reconstruction.
        numer: Poly,
        denom: Poly,
    },
}

/// Result of the Risch integration.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum RischResult {
    /// Successfully found an elementary antiderivative, expressed as
    /// the sum of a rational function plus logarithmic terms.
    Elementary {
        /// Rational part numerator.
        rational_numer: Poly,
        /// Rational part denominator.
        rational_denom: Poly,
        /// Logarithmic terms `Σ cᵢ ln(vᵢ)`.
        log_terms: Vec<LogTerm>,
    },
    /// Proved that no elementary antiderivative exists.
    NonElementary,
    /// Hit an unimplemented case or internal limitation.
    Failed(String),
}

// ═══════════════════════════════════════════════════════════════════════════
// Arena ↔ Poly bridge
// ═══════════════════════════════════════════════════════════════════════════

/// Try to integrate a rational function `A(x)/D(x)` using Hermite reduction
/// followed by Rothstein-Trager.
///
/// Returns `Some(result_expr_id)` if the expression is a rational function
/// and integration succeeds.  Returns `None` if the expression is not a
/// rational function or if algebraic log terms can't be represented.
pub fn try_risch_rational(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<ExprId> {
    // Recursion guard: if we're already inside try_risch_rational
    // (integrating an algebraic remainder), skip to avoid infinite loop.
    // The RAII guard resets the flag on drop, even during panics.
    let _guard = RischRecursionGuard::enter()?;

    // Decompose expr into numerator / denominator.
    let (numer_id, denom_id) = crate::poly::polybridge::as_numer_denom(arena, expr);

    // Only proceed if there's a non-trivial denominator.
    if denom_id == arena.one() {
        return None;
    }

    // Expand and evaluate both numer and denom before converting to Poly.
    // This handles cases like (x²+1)² which need expansion to x⁴+2x²+1
    // before expr_to_poly can parse them as univariate polynomials.
    let numer_exp = crate::transforms::expand::expand(arena, numer_id);
    let numer_expanded = crate::transforms::eval::eval(arena, numer_exp);
    let denom_exp = crate::transforms::expand::expand(arena, denom_id);
    let denom_expanded = crate::transforms::eval::eval(arena, denom_exp);

    // Convert arena expressions to Poly using existing polybridge.
    let numer_poly = crate::poly::polybridge::expr_to_poly(arena, numer_expanded, var)?;
    let denom_poly = crate::poly::polybridge::expr_to_poly(arena, denom_expanded, var)?;

    // Skip if denominator is constant (not a rational function integration problem).
    if denom_poly.is_constant() {
        return None;
    }

    // Phase 1: Hermite reduction.
    let hr = hermite::hermite_reduce(&numer_poly, &denom_poly);

    // Phase 2: Rothstein-Trager on the square-free remainder.
    let log_result = if hr.h_numer.is_zero() {
        rothstein_trager::LogPartResult { terms: vec![] }
    } else {
        rothstein_trager::logarithmic_part(&hr.h_numer, &hr.h_denom)
    };

    // Convert results back to arena expressions.
    let mut terms: Vec<ExprId> = Vec::new();

    // Rational part: g_numer / g_denom
    if !hr.g_numer.is_zero() {
        let g_num_id = crate::poly::polybridge::poly_to_expr(arena, &hr.g_numer, var);
        let g_den_id = crate::poly::polybridge::poly_to_expr(arena, &hr.g_denom, var);
        if g_den_id == arena.one() {
            terms.push(g_num_id);
        } else {
            terms.push(arena.div(g_num_id, g_den_id));
        }
    }

    // ── Logarithmic terms ──────────────────────────────────────────────
    //
    // Pass 1: Collect rational (c_i, v_i) pairs and detect algebraic terms.
    let mut rational_log_parts: Vec<(Ratio<BigInt>, Poly)> = Vec::new();
    let mut has_algebraic = false;
    let mut n_algebraic = 0usize;

    for term in &log_result.terms {
        match term {
            LogTerm::Rational { coeff, argument } => {
                tracing::debug!(
                    coeff = %coeff,
                    argument_degree = ?argument.degree(),
                    "try_risch_rational: found rational log term"
                );
                rational_log_parts.push((coeff.clone(), argument.clone()));
            }
            LogTerm::Algebraic { min_poly, .. } => {
                tracing::debug!(
                    min_poly_degree = ?min_poly.degree(),
                    "try_risch_rational: found algebraic log term"
                );
                has_algebraic = true;
                n_algebraic += 1;
            }
        }
    }

    tracing::debug!(
        n_rational = rational_log_parts.len(),
        n_algebraic,
        has_algebraic,
        "try_risch_rational: Rothstein-Trager classification"
    );

    // Pass 2: Always emit the rational log terms — these are exact
    // coefficients from the Rothstein-Trager algorithm.
    for (coeff, argument) in &rational_log_parts {
        let arg_id = crate::poly::polybridge::poly_to_expr(arena, argument, var);
        // Wrap in abs() for real-valued integration correctness:
        // ln(|v(x)|) is defined on the full real domain, while
        // ln(v(x)) requires v(x) > 0.
        let abs_arg = arena.abs(arg_id);
        let ln_arg = arena.ln(abs_arg);
        if coeff.is_one() {
            terms.push(ln_arg);
        } else if (-coeff.clone()).is_one() {
            terms.push(arena.neg(ln_arg));
        } else {
            let coeff_id = rational_to_expr(arena, coeff);
            terms.push(arena.mul(&[coeff_id, ln_arg]));
        }
    }

    // Pass 3: If algebraic terms exist, compute the algebraic remainder
    // by subtracting the rational log contributions from the integrand.
    //
    // The key identity: if the rational log terms contribute
    //   Σ c_i · ln(v_i)
    // then their derivative is
    //   Σ c_i · v_i'(x) / v_i(x)
    // and the algebraic part of the integrand is
    //   A/D − Σ c_i · v_i' · (D/v_i) / D  =  A_alg / D
    // where A_alg = A − Σ c_i · v_i' · (D / v_i).
    //
    // The divisibility Π v_i | A_alg is guaranteed by the residue theorem:
    // subtracting the rational poles removes them from the numerator.
    // After GCD cancellation, the reduced A_alg/D_reduced has only the
    // irreducible quadratic (or higher) factors in its denominator.
    if has_algebraic && !hr.h_numer.is_zero() {
        tracing::debug!(
            h_numer_degree = ?hr.h_numer.degree(),
            h_denom_degree = ?hr.h_denom.degree(),
            n_rational_to_subtract = rational_log_parts.len(),
            "try_risch_rational: computing algebraic remainder"
        );

        // Compute A_alg = h_numer − Σ c_i · v_i' · (h_denom / v_i)
        let mut a_alg = hr.h_numer.clone();
        for (coeff, v_i) in &rational_log_parts {
            let v_i_prime = v_i.derivative();
            let cofactor = hr.h_denom.div(v_i); // exact: v_i | h_denom
            debug_assert!(
                {
                    let product = &cofactor * v_i;
                    product == hr.h_denom
                },
                "h_denom / v_i must be exact polynomial division"
            );
            let contribution = (&v_i_prime * &cofactor).scale(coeff);
            tracing::trace!(
                coeff = %coeff,
                v_i_degree = ?v_i.degree(),
                cofactor_degree = ?cofactor.degree(),
                "try_risch_rational: subtracting rational contribution"
            );
            a_alg = &a_alg - &contribution;
        }

        if !a_alg.is_zero() {
            // GCD-cancel: A_alg is divisible by Π v_i (mathematical guarantee).
            let g = Poly::gcd(&a_alg, &hr.h_denom);
            let a_reduced = a_alg.div(&g);
            let d_reduced = hr.h_denom.div(&g);

            tracing::debug!(
                a_alg_degree = ?a_alg.degree(),
                gcd_degree = ?g.degree(),
                a_reduced_degree = ?a_reduced.degree(),
                d_reduced_degree = ?d_reduced.degree(),
                "try_risch_rational: algebraic remainder after GCD cancellation"
            );

            debug_assert!(
                {
                    let (_, rem) = a_alg.div_rem(&g);
                    rem.is_zero()
                },
                "A_alg must be divisible by gcd(A_alg, h_denom)"
            );

            let alg_num_id = crate::poly::polybridge::poly_to_expr(arena, &a_reduced, var);
            let alg_den_id = crate::poly::polybridge::poly_to_expr(arena, &d_reduced, var);
            let algebraic_remainder = arena.div(alg_num_id, alg_den_id);

            // Recursively integrate only the algebraic remainder.
            // The RAII guard (_guard) is still active, so try_risch_rational
            // will return None if called from inside integrate — preventing
            // infinite loops.  The heuristic integrator's other strategies
            // (standard forms, partial fractions, u-sub) will still fire.
            tracing::debug!("try_risch_rational: recursively integrating algebraic remainder");
            let alg_integral =
                crate::transforms::integrate::integrate(arena, algebraic_remainder, var);

            let alg_has_uneval = crate::base::walk::has_unevaluated(arena, alg_integral);
            tracing::debug!(
                has_unevaluated = alg_has_uneval,
                "try_risch_rational: algebraic remainder integration complete"
            );
            terms.push(alg_integral);
        } else {
            tracing::debug!(
                "try_risch_rational: A_alg is zero — rational terms fully account for the integrand"
            );
        }
    }

    if terms.is_empty() {
        Some(arena.zero())
    } else if terms.len() == 1 {
        Some(terms[0])
    } else {
        Some(arena.add(&terms))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper: Ratio<BigInt> → ExprId
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a `Ratio<BigInt>` to an arena expression.
fn rational_to_expr(arena: &mut Arena, r: &Ratio<BigInt>) -> ExprId {
    let nid = arena.intern_num(r.clone());
    arena.intern(ExprNode::Num(nid))
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(arena: &mut Arena, name: &str) -> ExprId {
        arena.symbol(name)
    }

    fn display(arena: &Arena, id: ExprId) -> String {
        arena.display(id).to_string()
    }

    #[test]
    fn rational_to_expr_integer() {
        let mut arena = Arena::new();
        let r = Ratio::from_integer(BigInt::from(42));
        let expr = rational_to_expr(&mut arena, &r);
        assert_eq!(display(&arena, expr), "42");
    }

    #[test]
    fn rational_to_expr_fraction() {
        let mut arena = Arena::new();
        let r = Ratio::new(BigInt::from(3), BigInt::from(4));
        let expr = rational_to_expr(&mut arena, &r);
        assert_eq!(display(&arena, expr), "3/4");
    }

    #[test]
    fn try_risch_rational_on_non_rational_returns_none() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        // sin(x) is not a rational function.
        let expr = arena.sin(x);
        assert!(try_risch_rational(&mut arena, expr, x).is_none());
    }

    #[test]
    fn try_risch_rational_on_polynomial_returns_none() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        // x^2 + 1 has denominator 1 — not a rational function integration target.
        let two = arena.int(2);
        let one = arena.one();
        let x_sq = arena.pow(x, two);
        let expr = arena.add(&[x_sq, one]);
        assert!(try_risch_rational(&mut arena, expr, x).is_none());
    }

    #[test]
    fn try_risch_rational_one_over_x() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let neg1 = arena.int(-1);
        // 1/x = x^(-1)
        let expr = arena.pow(x, neg1);
        let result = try_risch_rational(&mut arena, expr, x);
        // 1/x should integrate — the Hermite reduction is a no-op (denom is
        // already square-free), and Rothstein-Trager gives ln(x).
        assert!(
            result.is_some(),
            "∫ 1/x dx should succeed via Risch rational"
        );
        let s = display(&arena, result.unwrap());
        assert!(
            s.contains("ln") && s.contains("x"),
            "∫ 1/x dx should contain ln(x), got: {s}"
        );
    }

    #[test]
    fn try_risch_rational_one_over_x_sq_plus_1_squared() {
        // 1/(x²+1)² = Pow(Add(x²,1), -2).
        // Hermite reduction extracts the rational part x/(2(x²+1)).
        // The remaining 1/(2(x²+1)) has algebraic log terms (arctan).
        // The recursive integration resolves the arctan remainder via
        // the heuristic integrator's standard-form detector.
        // Final result: x/(2(x²+1)) + (1/2)·arctan(x).
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let two = arena.int(2);
        let x_sq = arena.pow(x, two);
        let one = arena.one();
        let x_sq_plus_1 = arena.add(&[x_sq, one]);
        let neg2 = arena.int(-2);
        let expr = arena.pow(x_sq_plus_1, neg2); // (x²+1)^(-2)

        let result = try_risch_rational(&mut arena, expr, x);
        assert!(result.is_some(), "∫ 1/(x²+1)² dx should succeed");

        let result_expr = result.unwrap();
        let s = display(&arena, result_expr);
        // Should contain the Hermite rational part and arctan — fully evaluated.
        assert!(
            s.contains("x") && s.contains("atan"),
            "result should have rational part + arctan, got: {s}"
        );
        // Should NOT contain unevaluated Integral.
        assert!(
            !s.contains("Integral"),
            "result should be fully evaluated, got: {s}"
        );
    }

    #[test]
    fn try_risch_rational_one_over_x_squared() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let neg2 = arena.int(-2);
        // 1/x^2 = x^(-2)
        let expr = arena.pow(x, neg2);
        let result = try_risch_rational(&mut arena, expr, x);
        // Hermite reduction extracts -1/x, no log part.
        assert!(
            result.is_some(),
            "∫ 1/x^2 dx should succeed via Risch rational"
        );
        let s = display(&arena, result.unwrap());
        // Should be -1/x or equivalent
        assert!(s.contains("x"), "∫ 1/x^2 dx should be -1/x, got: {s}");
    }

    // ── Phase 4 infrastructure verification ────────────────────────

    #[test]
    fn prs_x3_minus_1_has_degree_1_member() {
        // Verify that the Euclidean PRS of D(x) = x³-1 and B(x,t) = 1-3t·x²
        // (computed in GenPoly<RationalFn>) has a degree-1 member h(t,x) = x - 3t.
        use crate::poly::dense::Poly;
        use crate::poly::generic::GenPoly;
        use crate::poly::ratfn::RationalFn;
        use crate::poly::traits::Ring;

        fn r(n: i64, d: i64) -> Ratio<BigInt> {
            Ratio::new(BigInt::from(n), BigInt::from(d))
        }

        // D(x) = x³ - 1
        let d_gp: GenPoly<RationalFn> = GenPoly::from_coeffs(vec![
            RationalFn::from_rational(r(-1, 1)),
            RationalFn::from_rational(r(0, 1)),
            RationalFn::from_rational(r(0, 1)),
            RationalFn::from_rational(r(1, 1)),
        ]);

        // B(x,t) = 1 - 3t·x²
        let neg_3t = RationalFn::from_poly(Poly::from_coeffs(vec![r(0, 1), r(-3, 1)]));
        let b_gp: GenPoly<RationalFn> = GenPoly::from_coeffs(vec![
            RationalFn::from_rational(r(1, 1)),
            RationalFn::from_rational(r(0, 1)),
            neg_3t,
        ]);

        // Compute PRS
        let prs = GenPoly::<RationalFn>::euclidean_prs(&d_gp, &b_gp);

        // Must have a degree-1 member
        assert!(
            prs.contains_key(&1),
            "PRS should contain a degree-1 member, got degrees: {:?}",
            prs.keys().collect::<Vec<_>>()
        );

        // Make it monic
        let h = prs.get(&1).unwrap();
        let h_monic = h.make_monic();

        // h(t,x) should be x - 3t (monic in x)
        // coeff(1) should be RF(1) (the leading coefficient, monic)
        let c1 = h_monic.coeff(1);
        assert!(
            c1.numer().is_constant() && c1.denom().is_constant(),
            "x coefficient should be a constant RationalFn"
        );
        let c1_val = c1.to_rational().expect("should be rational");
        assert_eq!(c1_val, r(1, 1), "x coefficient should be 1");

        // coeff(0) should be RF(-3t) = RationalFn(numer=-3t, denom=1)
        let c0 = h_monic.coeff(0);
        assert!(
            c0.denom().is_constant(),
            "constant term denominator should be 1"
        );
        let c0_numer = c0.numer();
        assert_eq!(c0_numer.degree(), Some(1), "constant term should be linear in t");
        assert_eq!(c0_numer.coeff(0), r(0, 1), "constant of -3t should be 0");
        assert_eq!(c0_numer.coeff(1), r(-3, 1), "slope of -3t should be -3");
    }

    #[test]
    fn solve_quadratic_factor_produces_conjugate_roots() {
        // Verify that solve on q(t) = 9t²+3t+1 produces complex roots
        // that as_real_imag can decompose into (u, v) = (-1/6, ±√3/6).
        let mut arena = Arena::new();
        let t = sym(&mut arena, "t");

        // q(t) = 9t² + 3t + 1
        let nine = arena.int(9);
        let three = arena.int(3);
        let one = arena.one();
        let two = arena.int(2);
        let t_sq = arena.pow(t, two);
        let term_9t2 = arena.mul(&[nine, t_sq]);
        let term_3t = arena.mul(&[three, t]);
        let q_expr = arena.add(&[term_9t2, term_3t, one]);

        let roots = crate::transforms::solve::solve(&mut arena, q_expr, t);
        assert_eq!(roots.len(), 2, "quadratic should have 2 roots, got {}", roots.len());

        // Decompose each root into (Re, Im)
        let mut pos_im_found = false;
        let mut neg_im_found = false;

        for root in &roots {
            let (re, im) = crate::base::complex::as_real_imag(&mut arena, root.value);
            let re = crate::transforms::eval::eval(&mut arena, re);
            let im = crate::transforms::eval::eval(&mut arena, im);

            // Re should be -1/6
            let re_f64 = crate::transforms::evalf::eval_const_f64(&mut arena, re);
            let im_f64 = crate::transforms::evalf::eval_const_f64(&mut arena, im);

            if let (Some(re_v), Some(im_v)) = (re_f64, im_f64) {
                assert!(
                    (re_v - (-1.0 / 6.0)).abs() < 1e-10,
                    "Re should be -1/6, got {re_v}"
                );
                let expected_im = 3.0_f64.sqrt() / 6.0;
                assert!(
                    (im_v.abs() - expected_im).abs() < 1e-10,
                    "|Im| should be √3/6 ≈ {expected_im}, got {}", im_v.abs()
                );
                if im_v > 0.0 {
                    pos_im_found = true;
                } else {
                    neg_im_found = true;
                }
            } else {
                panic!("Could not evaluate root to f64: {}", display(&arena, root.value));
            }
        }

        assert!(pos_im_found, "should have a root with positive imaginary part");
        assert!(neg_im_found, "should have a root with negative imaginary part");
    }

    #[test]
    fn try_risch_rational_one_over_x_cubed_minus_1_numerical() {
        // End-to-end: ∫ 1/(x³-1) dx with numerical verification.
        // This exercises the Phase 1 fix (algebraic remainder subtraction).
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let three = arena.int(3);
        let one = arena.one();
        let x_cubed = arena.pow(x, three);
        let denom = arena.sub(x_cubed, one);
        let neg1 = arena.int(-1);
        let expr = arena.pow(denom, neg1); // (x³-1)^(-1)

        let result = try_risch_rational(&mut arena, expr, x);
        assert!(result.is_some(), "∫ 1/(x³-1) dx should succeed");

        let anti = result.unwrap();
        assert!(
            !crate::base::walk::has_unevaluated(&arena, anti),
            "result should not contain unevaluated integrals: {}",
            display(&arena, anti)
        );

        // Numerical check: F(3) - F(2)
        let val_3 = arena.int(3);
        let val_2 = arena.int(2);
        let f3 = crate::transforms::subs::subs(&mut arena, anti, x, val_3);
        let f3 = crate::transforms::eval::eval(&mut arena, f3);
        let f2 = crate::transforms::subs::subs(&mut arena, anti, x, val_2);
        let f2 = crate::transforms::eval::eval(&mut arena, f2);

        let f3_f64 = crate::transforms::evalf::eval_const_f64(&mut arena, f3);
        let f2_f64 = crate::transforms::evalf::eval_const_f64(&mut arena, f2);

        if let (Some(f3v), Some(f2v)) = (f3_f64, f2_f64) {
            let integral = f3v - f2v;
            assert!(
                (integral - 0.07539).abs() < 0.001,
                "∫₂³ 1/(x³-1) dx ≈ 0.07539, got {integral}"
            );
        }
    }
}
