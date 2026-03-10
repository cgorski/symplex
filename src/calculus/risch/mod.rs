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
pub mod rothstein_trager;
pub mod tower;
pub mod rde;
pub mod integrate;

use std::cell::Cell;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};

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
pub fn try_risch_rational(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Option<ExprId> {
    // Recursion guard: if we're already inside try_risch_rational
    // (integrating an algebraic remainder), skip to avoid infinite loop.
    // The RAII guard resets the flag on drop, even during panics.
    let _guard = match RischRecursionGuard::enter() {
        Some(g) => g,
        None => return None, // already inside — skip to avoid infinite recursion
    };

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

    // Logarithmic terms.
    let mut has_algebraic = false;
    for term in &log_result.terms {
        match term {
            LogTerm::Rational { coeff, argument } => {
                let arg_id = crate::poly::polybridge::poly_to_expr(arena, argument, var);
                let ln_arg = arena.ln(arg_id);
                if coeff.is_one() {
                    terms.push(ln_arg);
                } else if (-coeff.clone()).is_one() {
                    terms.push(arena.neg(ln_arg));
                } else {
                    let coeff_id = rational_to_expr(arena, coeff);
                    terms.push(arena.mul(&[coeff_id, ln_arg]));
                }
            }
            LogTerm::Algebraic { .. } => {
                // We can't represent algebraic log terms in the arena yet.
                // Instead of bailing entirely, we'll emit the rational part
                // we computed (from Hermite) and leave the algebraic remainder
                // as an unevaluated Integral.
                has_algebraic = true;
            }
        }
    }

    // If there are algebraic log terms, try to recursively integrate the
    // square-free remainder using the heuristic integrator.  For example,
    // 1/(x²+1) has algebraic residues (±i/2) but the heuristic integrator
    // knows it's arctan(x) via the standard-form detector.
    if has_algebraic && !hr.h_numer.is_zero() {
        let h_num_id = crate::poly::polybridge::poly_to_expr(arena, &hr.h_numer, var);
        let h_den_id = crate::poly::polybridge::poly_to_expr(arena, &hr.h_denom, var);
        let remainder = arena.div(h_num_id, h_den_id);

        // Try the heuristic integrator on the remainder.
        // The RAII guard (_guard) is still active, so try_risch_rational
        // will return None if called from inside integrate — preventing
        // infinite loops.  The heuristic integrator's other strategies
        // (standard forms, partial fractions, u-sub) will still fire.
        let remainder_integral = crate::transforms::integrate::integrate(arena, remainder, var);

        terms.push(remainder_integral);
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
        assert!(result.is_some(), "∫ 1/x dx should succeed via Risch rational");
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
        assert!(result.is_some(), "∫ 1/x^2 dx should succeed via Risch rational");
        let s = display(&arena, result.unwrap());
        // Should be -1/x or equivalent
        assert!(
            s.contains("x"),
            "∫ 1/x^2 dx should be -1/x, got: {s}"
        );
    }
}
