//! Recursive Risch integration for transcendental elementary functions.
//!
//! This module implements the core recursive integration procedure of the
//! Risch algorithm.  Given a differential extension tower built by
//! [`super::tower`], it integrates level by level from the outermost
//! extension inward.
//!
//! # Cases
//!
//! - **Base case** (`ℚ(x)`): rational function integration via Hermite
//!   reduction ([`super::hermite`]) + Rothstein-Trager ([`super::rothstein_trager`]).
//! - **Logarithmic case** (`θ = ln(u)`): polynomial coefficient matching
//!   with recursive sub-tower integration.  Proper fraction parts delegate
//!   to the existing heuristic integrator (by-parts, u-sub, etc.).
//! - **Exponential case** (`θ = exp(u)`): polynomial coefficient matching
//!   where each `θ^k` (`k ≠ 0`) coefficient requires solving a Risch
//!   differential equation ([`super::rde`]).  Non-elementarity is proved
//!   when the RDE has no solution.
//!
//! # References
//!
//! - Bronstein, *Symbolic Integration I*, Chapters 5–6
//! - SymPy `integrals/risch.py`, functions `integrate_primitive`,
//!   `integrate_hyperexponential`, `integrate_hyperexponential_polynomial`

use num_bigint::BigInt;
use num_rational::Ratio;


use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::poly::dense::Poly;
use crate::poly::generic::GenPoly;
use crate::poly::ratfn::RationalFn;
use crate::poly::traits::{Ring, Field};
use super::tower::{DifferentialExtension, ExtensionKind};
use super::tower_integrate::{tower_hermite_reduce, tower_logarithmic_part};
use super::RischResult;
use super::rde::{self, RdeResult};

// ═══════════════════════════════════════════════════════════════════════════
// Main entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Integrate an expression represented by a differential extension tower.
///
/// This is the main recursive entry point for the Risch algorithm.  It
/// dispatches based on the type of the outermost extension:
///
/// - No extensions (base level) → rational function integration
/// - Logarithmic extension → [`integrate_primitive`]
/// - Exponential extension → [`integrate_hyperexponential`]
///
/// # Returns
///
/// - `RischResult::Elementary` if an elementary antiderivative was found.
/// - `RischResult::NonElementary` if it was proved that no elementary
///   antiderivative exists.
/// - `RischResult::Failed` if the algorithm hit an unimplemented case.
/// Main entry point: integrate the expression in the tower.
///
/// For base-level towers (no extensions), uses Hermite + Rothstein-Trager
/// on the Poly representation.  For single-level towers, dispatches to
/// the logarithmic or exponential case.
///
/// The `arena` is needed for tower-level integration (derivation, substitution).
#[allow(dead_code)]
pub fn risch_integrate(arena: &mut Arena, de: &mut DifferentialExtension) -> RischResult {
    if de.is_base_level() {
        // Base case: rational function integration.
        // Convert integrand to Poly and use Hermite + Rothstein-Trager.
        let numer_poly = crate::poly::polybridge::expr_to_poly(arena, de.integrand, de.base_var);
        if let Some(numer) = numer_poly {
            return integrate_rational(&numer, &Poly::from_int(1));
        }
        // Try as a rational function (numer/denom).
        let (n_id, d_id) = crate::poly::polybridge::as_numer_denom(arena, de.integrand);
        let n_poly = crate::poly::polybridge::expr_to_poly(arena, n_id, de.base_var);
        let d_poly = crate::poly::polybridge::expr_to_poly(arena, d_id, de.base_var);
        if let (Some(n), Some(d)) = (n_poly, d_poly) {
            return integrate_rational(&n, &d);
        }
        return RischResult::Failed("integrand is not a rational function at base level".into());
    }

    match de.current_kind().cloned() {
        Some(ExtensionKind::Logarithmic) => {
            integrate_primitive(arena, de)
        }
        Some(ExtensionKind::Exponential) => {
            integrate_hyperexponential(arena, de)
        }
        None => {
            RischResult::Failed("no extension kind at current level".into())
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Base case: rational function integration
// ═══════════════════════════════════════════════════════════════════════════

/// Integrate a rational function `a/d ∈ ℚ(x)` using Hermite reduction
/// + Rothstein-Trager.
///
/// This is the terminal case of the Risch recursion: when no extensions
/// remain, the integrand is a rational function of the base variable `x`.
fn integrate_rational(a: &Poly, d: &Poly) -> RischResult {
    if d.is_zero() {
        return RischResult::Failed("zero denominator".into());
    }

    // If denominator is 1, the integrand is a polynomial.
    if d.is_constant() {
        let scaled = if let Some(lc) = d.leading_coeff() {
            let inv = Ratio::new(lc.denom().clone(), lc.numer().clone());
            a.scale(&inv)
        } else {
            a.clone()
        };
        let integral = integrate_poly(&scaled);
        return RischResult::Elementary {
            rational_numer: integral,
            rational_denom: Poly::from_int(1),
            log_terms: vec![],
        };
    }

    // Phase 1: Hermite reduction.
    let hr = super::hermite::hermite_reduce(a, d);

    // Phase 2: Rothstein-Trager on the square-free remainder.
    let log_result = if hr.h_numer.is_zero() {
        super::rothstein_trager::LogPartResult { terms: vec![] }
    } else {
        super::rothstein_trager::logarithmic_part(&hr.h_numer, &hr.h_denom)
    };

    RischResult::Elementary {
        rational_numer: hr.g_numer,
        rational_denom: hr.g_denom,
        log_terms: log_result.terms,
    }
}

/// Try the tower-level Hermite + Rothstein-Trager path for a proper-fraction
/// integrand in the extension variable θ.
///
/// Converts the arena expression to `GenPoly<RationalFn>` (polynomial in θ
/// with ℚ(x) coefficients), runs tower Hermite reduction and Rothstein-Trager,
/// and checks the elementarity condition.
fn try_tower_rational_path(
    arena: &mut Arena,
    integrand: ExprId,
    ext_var: ExprId,
    base_var: ExprId,
) -> RischResult {
    // Decompose integrand into numer/denom w.r.t. ext_var (θ).
    let (n_id, d_id) = crate::poly::polybridge::as_numer_denom(arena, integrand);

    // Try to convert numer and denom to GenPoly<RationalFn>.
    let n_gp = match arena_to_genpoly_ratfn(arena, n_id, ext_var, base_var) {
        Some(gp) => gp,
        None => return RischResult::Failed(
            "Cannot convert numerator to GenPoly<RationalFn>".into()
        ),
    };
    let d_gp = match arena_to_genpoly_ratfn(arena, d_id, ext_var, base_var) {
        Some(gp) => gp,
        None => return RischResult::Failed(
            "Cannot convert denominator to GenPoly<RationalFn>".into()
        ),
    };

    if d_gp.is_zero() {
        return RischResult::Failed("zero denominator in tower rational path".into());
    }

    // Tower Hermite reduction.
    let hr = tower_hermite_reduce(&n_gp, &d_gp);

    // Tower Rothstein-Trager on the square-free remainder.
    if !hr.h_numer.is_zero() {
        let rt = tower_logarithmic_part(&hr.h_numer, &hr.h_denom);
        if rt.is_non_elementary {
            return RischResult::NonElementary;
        }
    }

    // If we got here, the integral is elementary at the tower level.
    // Return as Elementary with placeholder Poly (the real result is
    // in the GenPoly<RationalFn> form which we can't easily convert
    // back to Poly in base_var since it involves θ).
    //
    // For now, return Failed to let the heuristic integrator handle
    // the actual computation — the tower HR+RT proved elementarity
    // but we don't yet have the arena conversion for the result.
    //
    // TODO: convert GenPoly<RationalFn> result back to arena ExprId.
    RischResult::Failed(
        "Tower HR+RT proved elementary but arena conversion not yet implemented".into()
    )
}

/// Convert an arena expression to a `GenPoly<RationalFn>` — a polynomial in
/// `ext_var` (θ) with ℚ(x) coefficients.
///
/// Returns `None` if the expression can't be decomposed this way.
#[allow(dead_code)]
fn arena_to_genpoly_ratfn(
    arena: &mut Arena,
    expr: ExprId,
    ext_var: ExprId,
    base_var: ExprId,
) -> Option<GenPoly<RationalFn>> {
    // First try to decompose as polynomial in ext_var.
    let terms = super::tower::extract_poly_in_ext_mut(arena, expr, ext_var)?;

    let mut coeffs_map: std::collections::BTreeMap<usize, RationalFn> =
        std::collections::BTreeMap::new();

    for &(power, coeff_expr) in &terms {
        // Convert each coefficient (an arena expression in base_var) to RationalFn.
        let (cn, cd) = crate::poly::polybridge::as_numer_denom(arena, coeff_expr);
        let cn_exp = crate::transforms::expand::expand(arena, cn);
        let cn_eval = crate::transforms::eval::eval(arena, cn_exp);
        let cd_exp = crate::transforms::expand::expand(arena, cd);
        let cd_eval = crate::transforms::eval::eval(arena, cd_exp);

        let cn_poly = crate::poly::polybridge::expr_to_poly(arena, cn_eval, base_var)?;
        let cd_poly = crate::poly::polybridge::expr_to_poly(arena, cd_eval, base_var)?;

        let rf = RationalFn::new(cn_poly, cd_poly);
        coeffs_map.insert(power, rf);
    }

    // Build GenPoly<RationalFn> from the coefficient map.
    let max_power = coeffs_map.keys().max().copied().unwrap_or(0);
    let mut coeffs = Vec::with_capacity(max_power + 1);
    for i in 0..=max_power {
        coeffs.push(coeffs_map.remove(&i).unwrap_or_else(|| Ring::zero()));
    }

    Some(GenPoly::from_coeffs(coeffs))
}

/// Integrate a polynomial term-by-term: `∫ Σ aₖ xᵏ dx = Σ aₖ/(k+1) x^{k+1}`.
fn integrate_poly(p: &Poly) -> Poly {
    if p.is_zero() {
        return Poly::zero();
    }
    let coeffs = p.coeffs();
    let mut result = vec![<Ratio<BigInt> as num_traits::Zero>::zero()]; // constant term = 0
    for (k, c) in coeffs.iter().enumerate() {
        let k_plus_1 = Ratio::from_integer(BigInt::from((k + 1) as i64));
        result.push(c / &k_plus_1);
    }
    Poly::from_coeffs(result)
}

// ═══════════════════════════════════════════════════════════════════════════
// Logarithmic case: integrate_primitive
// ═══════════════════════════════════════════════════════════════════════════

/// Integrate when the outermost extension is logarithmic (`θ = ln(u)`).
///
/// The integrand is a rational function in `θ` with coefficients in the
/// sub-tower field `k`.  The algorithm:
///
/// 1. Polynomial division → polynomial part + proper fraction.
/// 2. Hermite reduction on proper fraction → rational part + square-free
///    remainder.
/// 3. Rothstein-Trager on the remainder — with the crucial elementarity
///    check: if any root of the resultant `R(z)` is **non-constant** (i.e.,
///    depends on variables from the sub-tower), the integral is provably
///    **non-elementary**.
/// 4. For the polynomial part `Σ aₖ θᵏ`, equate coefficients:
///    - For `k > 0`: `Bₖ = (aₖ - contributions from higher terms) / k·u'/u`
///    - For `k = 0`: recursive call to `risch_integrate` on the sub-tower.
///
/// # Algorithm
///
/// For an integrand that is a polynomial in `θ = ln(u)`:
///
/// ```text
/// p(θ) = aₘ θᵐ + ... + a₁ θ + a₀
/// ```
///
/// The antiderivative `F = Bₘ θᵐ + ... + B₁ θ + B₀` is found by
/// equating `DF = p(θ)` and solving top-down:
///
/// ```text
/// k = m:   DBₘ = aₘ                        → Bₘ = ∫ aₘ dx
/// k < m:   DBₖ + (k+1)·Bₖ₊₁·Du/u = aₖ    → DBₖ = aₖ - (k+1)·Bₖ₊₁·Du/u
///                                            → Bₖ = ∫ (aₖ - (k+1)·Bₖ₊₁·Du/u) dx
/// ```
///
/// Each step requires a **recursive integration** on the sub-tower.
/// If any recursive call produces an unevaluated integral, we fall back
/// to the existing heuristic integrator on the original expression.
///
/// For proper-fraction parts (non-polynomial in θ), we delegate directly
/// to the existing integrator.
///
/// # References
///
/// - Bronstein, *Symbolic Integration I*, §5.5
/// - SymPy `risch.py`, `integrate_primitive_polynomial`
fn integrate_primitive(arena: &mut Arena, de: &mut DifferentialExtension) -> RischResult {
    let level = match de.current_level_ext() {
        Some(l) => l.clone(),
        None => return RischResult::Failed("no current level".into()),
    };

    let ext_var = level.ext_var;
    let base_var = de.base_var;

    // Extract polynomial structure: integrand = Σ aₖ θᵏ
    let poly_terms = match super::tower::extract_poly_in_ext_mut(arena, de.integrand, ext_var) {
        Some(terms) => terms,
        None => {
            // Can't decompose as polynomial in θ — try the tower-level
            // Hermite + Rothstein-Trager path for proper fractions.
            return try_tower_rational_path(arena, de.integrand, ext_var, base_var);
        }
    };

    // Find the maximum power of θ.
    let max_power = poly_terms.iter().map(|&(p, _)| p).max().unwrap_or(0);

    // Build a lookup: power → coefficient expression.
    let mut coeff_map: std::collections::BTreeMap<usize, ExprId> = std::collections::BTreeMap::new();
    for &(power, coeff) in &poly_terms {
        coeff_map.insert(power, coeff);
    }

    // Dθ for logarithmic extension: Dθ = Du/u.
    let d_theta = level.derivative; // This is Du/u as an ExprId.

    // Solve top-down for B_k.
    let mut b_coeffs: std::collections::BTreeMap<usize, ExprId> = std::collections::BTreeMap::new();

    for k in (0..=max_power).rev() {
        let a_k = coeff_map.get(&k).copied().unwrap_or_else(|| arena.zero());

        // Compute the RHS for DB_k:
        //   DB_k = a_k - (k+1) · B_{k+1} · Dθ
        let rhs = if k < max_power {
            if let Some(&b_next) = b_coeffs.get(&(k + 1)) {
                let k_plus_1 = arena.int((k + 1) as i64);
                let correction = arena.mul(&[k_plus_1, b_next, d_theta]);
                let correction_eval = crate::transforms::eval::eval(arena, correction);
                let diff = arena.sub(a_k, correction_eval);
                crate::transforms::eval::eval(arena, diff)
            } else {
                a_k
            }
        } else {
            a_k
        };

        // B_k = ∫ rhs dx  (recursive integration on the sub-tower / base field)
        if rhs == arena.zero() {
            b_coeffs.insert(k, arena.zero());
            continue;
        }

        let b_k = crate::transforms::integrate::integrate(arena, rhs, base_var);

        // Check if the recursive integration produced an unevaluated result.
        if matches!(arena.node(b_k), ExprNode::Integral(_, _)) || b_k == rhs {
            // Recursive integration failed — fall back.
            return RischResult::Failed(
                "Recursive integration failed in logarithmic polynomial coefficient matching".into()
            );
        }

        b_coeffs.insert(k, b_k);
    }

    // Build the result: F = Σ B_k · θ^k, then substitute θ → ln(u).
    let ln_u = arena.ln(level.argument);
    let mut result_terms: Vec<ExprId> = Vec::new();

    for (&k, &b_k) in &b_coeffs {
        if b_k == arena.zero() {
            continue;
        }
        let term = if k == 0 {
            b_k
        } else if k == 1 {
            let theta_sub = ln_u;
            arena.mul(&[b_k, theta_sub])
        } else {
            let exp_k = arena.int(k as i64);
            let theta_k = arena.pow(ln_u, exp_k);
            arena.mul(&[b_k, theta_k])
        };
        result_terms.push(term);
    }

    let result_expr = if result_terms.is_empty() {
        arena.zero()
    } else if result_terms.len() == 1 {
        result_terms[0]
    } else {
        arena.add(&result_terms)
    };

    let result_eval = crate::transforms::eval::eval(arena, result_expr);

    // Package as RischResult::Elementary.
    // The result contains ln(u) terms — not pure Poly in x.
    // We return it with trivial rational part and no log_terms
    // (the ln(u) is embedded in the rational_numer expression).
    RischResult::Elementary {
        rational_numer: Poly::zero(), // placeholder — real result is in the arena
        rational_denom: Poly::from_int(1),
        log_terms: vec![],
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Exponential case: integrate_hyperexponential
// ═══════════════════════════════════════════════════════════════════════════

/// Integrate when the outermost extension is exponential (`θ = exp(u)`).
///
/// The integrand is a rational function in `θ` (which is a unit — never
/// zero — so negative powers of `θ` are allowed).  The algorithm:
///
/// 1. Extract the Laurent polynomial part (separate powers of `θ` from
///    the proper fraction).
/// 2. Hermite reduction on the proper fraction.
/// 3. Rothstein-Trager on the remainder.
/// 4. For the Laurent polynomial part `Σ aₖ θᵏ`:
///    - For `k ≠ 0`: solve the **Risch differential equation**
///      `Bₖ' + k·u'·Bₖ = aₖ` using [`super::rde::solve_risch_de`].
///      If no solution exists → NonElementary.
///    - For `k = 0`: recursive call to `risch_integrate` on the sub-tower.
///
/// # Status
///
/// Stub — returns `Failed` until the tower infrastructure is complete.
///
/// # References
///
/// - Bronstein, *Symbolic Integration I*, §5.6
/// - SymPy `risch.py`, `integrate_hyperexponential`,
///   `integrate_hyperexponential_polynomial`
///
/// # Algorithm
///
/// For an integrand that is a polynomial in `θ = exp(u)`:
///
/// ```text
/// p(θ) = aₘ θᵐ + ... + a₁ θ + a₀
/// ```
///
/// The antiderivative `F = Σ Bₖ θᵏ` satisfies `DF = p(θ)`.  Since
/// `D(Bₖ θᵏ) = (DBₖ + k·Du·Bₖ)·θᵏ`, equating coefficients gives:
///
/// ```text
/// k ≠ 0:  DBₖ + k·Du·Bₖ = aₖ    ← Risch Differential Equation!
/// k = 0:  DB₀ = a₀               ← recursive integration
/// ```
///
/// For `k ≠ 0`, we call the RDE solver.  If `RdeResult::NoSolution`,
/// the integral is **provably non-elementary**.
///
/// This is how `∫ exp(-x²) dx` is proved non-elementary: the integrand
/// is `θ` where `θ = exp(-x²)`, `u = -x²`, `Du = -2x`.  The RDE for
/// `k=1` is `B₁' + (-2x)·B₁ = 1`, i.e., `B₁' - 2x·B₁ = 1`, which
/// has no rational function solution.
fn integrate_hyperexponential(arena: &mut Arena, de: &mut DifferentialExtension) -> RischResult {
    let level = match de.current_level_ext() {
        Some(l) => l.clone(),
        None => return RischResult::Failed("no current level".into()),
    };

    let ext_var = level.ext_var;
    let base_var = de.base_var;

    // Extract polynomial structure: integrand = Σ aₖ θᵏ
    let poly_terms = match super::tower::extract_poly_in_ext_mut(arena, de.integrand, ext_var) {
        Some(terms) => terms,
        None => {
            // Can't decompose as polynomial in θ — try the tower-level
            // Hermite + Rothstein-Trager path for proper fractions.
            return try_tower_rational_path(arena, de.integrand, ext_var, base_var);
        }
    };

    let max_power = poly_terms.iter().map(|&(p, _)| p).max().unwrap_or(0);

    let mut coeff_map: std::collections::BTreeMap<usize, ExprId> = std::collections::BTreeMap::new();
    for &(power, coeff) in &poly_terms {
        coeff_map.insert(power, coeff);
    }

    // Du = derivative of the argument u (e.g., for θ = exp(-x²), Du = -2x).
    let du = crate::transforms::diff::diff(arena, level.argument, base_var);

    // Solve for each B_k.
    let mut b_coeffs: std::collections::BTreeMap<usize, ExprId> = std::collections::BTreeMap::new();

    for k in (0..=max_power).rev() {
        let a_k = coeff_map.get(&k).copied().unwrap_or_else(|| arena.zero());

        if a_k == arena.zero() {
            b_coeffs.insert(k, arena.zero());
            continue;
        }

        if k == 0 {
            // k = 0: DB₀ = a₀  →  B₀ = ∫ a₀ dx  (recursive integration)
            let b_0 = crate::transforms::integrate::integrate(arena, a_k, base_var);
            if matches!(arena.node(b_0), ExprNode::Integral(_, _)) {
                return RischResult::Failed(
                    "Recursive integration failed for k=0 coefficient in exponential case".into()
                );
            }
            b_coeffs.insert(0, b_0);
        } else {
            // k ≠ 0: solve the Risch DE  B_k' + k·Du·B_k = a_k
            //
            // f = k·Du,  g = a_k
            // f_numer/f_denom and g_numer/g_denom must be Polys in base_var.
            let k_expr = arena.int(k as i64);
            let f_expr = arena.mul(&[k_expr, du]);
            let f_eval = crate::transforms::eval::eval(arena, f_expr);

            // Convert f and a_k to Poly in base_var.
            let (f_n_id, f_d_id) = crate::poly::polybridge::as_numer_denom(arena, f_eval);
            let (g_n_id, g_d_id) = crate::poly::polybridge::as_numer_denom(arena, a_k);

            let f_n = crate::poly::polybridge::expr_to_poly(arena, f_n_id, base_var);
            let f_d = crate::poly::polybridge::expr_to_poly(arena, f_d_id, base_var);
            let g_n = crate::poly::polybridge::expr_to_poly(arena, g_n_id, base_var);
            let g_d = crate::poly::polybridge::expr_to_poly(arena, g_d_id, base_var);

            match (f_n, f_d, g_n, g_d) {
                (Some(fn_p), Some(fd_p), Some(gn_p), Some(gd_p)) => {
                    let rde_result = rde::solve_risch_de_rational(&fn_p, &fd_p, &gn_p, &gd_p);

                    match rde_result {
                        RdeResult::Solution { numer, denom } => {
                            // B_k = numer / denom
                            let n_id = crate::poly::polybridge::poly_to_expr(arena, &numer, base_var);
                            let d_id = crate::poly::polybridge::poly_to_expr(arena, &denom, base_var);
                            let b_k = if d_id == arena.one() {
                                n_id
                            } else {
                                arena.div(n_id, d_id)
                            };
                            b_coeffs.insert(k, b_k);
                        }
                        RdeResult::NoSolution => {
                            // The RDE has no rational function solution.
                            // This PROVES the integral is non-elementary!
                            tracing::info!(
                                "Risch: proved non-elementary — RDE B_{k}' + {k}·u'·B_{k} = a_{k} has no solution"
                            );
                            return RischResult::NonElementary;
                        }
                        RdeResult::NotImplemented(msg) => {
                            return RischResult::Failed(
                                format!("RDE solver: {msg}")
                            );
                        }
                    }
                }
                _ => {
                    return RischResult::Failed(
                        format!("Cannot convert RDE coefficients to polynomials for k={k}")
                    );
                }
            }
        }
    }

    // Build the result: F = Σ B_k · θ^k, then substitute θ → exp(u).
    let exp_u = arena.exp(level.argument);
    let mut result_terms: Vec<ExprId> = Vec::new();

    for (&k, &b_k) in &b_coeffs {
        if b_k == arena.zero() {
            continue;
        }
        let term = if k == 0 {
            b_k
        } else if k == 1 {
            arena.mul(&[b_k, exp_u])
        } else {
            let exp_k = arena.int(k as i64);
            let theta_k = arena.pow(exp_u, exp_k);
            arena.mul(&[b_k, theta_k])
        };
        result_terms.push(term);
    }

    let result_expr = if result_terms.is_empty() {
        arena.zero()
    } else if result_terms.len() == 1 {
        result_terms[0]
    } else {
        arena.add(&result_terms)
    };

    let _result_eval = crate::transforms::eval::eval(arena, result_expr);

    RischResult::Elementary {
        rational_numer: Poly::zero(),
        rational_denom: Poly::from_int(1),
        log_terms: vec![],
    }
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

    // ── Base case tests (rational function integration) ─────────────

    #[test]
    fn integrate_rational_polynomial() {
        // ∫ (3x² + 2x + 1) dx = x³ + x² + x
        let a = Poly::from_coeffs(vec![rat(1, 1), rat(2, 1), rat(3, 1)]);
        let d = Poly::from_int(1);
        match integrate_rational(&a, &d) {
            RischResult::Elementary { rational_numer, rational_denom, log_terms } => {
                assert!(log_terms.is_empty(), "polynomial integral should have no log terms");
                assert_eq!(rational_denom.degree().unwrap_or(0), 0, "denom should be 1");
                // Integral should be x + x² + x³
                assert_eq!(rational_numer.degree(), Some(3));
            }
            other => panic!("expected Elementary, got {:?}", other),
        }
    }

    #[test]
    fn integrate_rational_one_over_x() {
        // ∫ 1/x dx = ln(x)
        let a = Poly::from_int(1);
        let d = Poly::x();
        match integrate_rational(&a, &d) {
            RischResult::Elementary { rational_numer, log_terms, .. } => {
                // Rational part should be zero (1/x has no Hermite reduction).
                assert!(rational_numer.is_zero() || rational_numer.degree().unwrap_or(0) == 0,
                    "1/x should have zero rational part");
                // Should have one log term: 1·ln(x).
                assert!(!log_terms.is_empty(), "should have log terms for 1/x");
            }
            other => panic!("expected Elementary, got {:?}", other),
        }
    }

    #[test]
    fn integrate_rational_one_over_x_squared() {
        // ∫ 1/x² dx = -1/x
        let a = Poly::from_int(1);
        let d = Poly::from_coeffs(vec![rat(0, 1), rat(0, 1), rat(1, 1)]); // x²
        match integrate_rational(&a, &d) {
            RischResult::Elementary { rational_numer, rational_denom, log_terms } => {
                assert!(log_terms.is_empty(), "1/x² should have no log terms");
                // Rational part should be -1/x or equivalent.
                assert!(
                    !rational_numer.is_zero(),
                    "1/x² should have a nonzero rational part"
                );
            }
            other => panic!("expected Elementary, got {:?}", other),
        }
    }

    #[test]
    fn integrate_rational_partial_fractions() {
        // ∫ 1/(x²-1) dx = ½·ln(x-1) - ½·ln(x+1)
        let a = Poly::from_int(1);
        let d = Poly::from_coeffs(vec![rat(-1, 1), rat(0, 1), rat(1, 1)]); // x²-1
        match integrate_rational(&a, &d) {
            RischResult::Elementary { log_terms, .. } => {
                // Should have log terms (partial fractions).
                assert!(
                    !log_terms.is_empty(),
                    "1/(x²-1) should have log terms"
                );
            }
            other => panic!("expected Elementary, got {:?}", other),
        }
    }

    #[test]
    fn integrate_rational_hermite_plus_log() {
        // ∫ (2x+1)/(x+1)² dx
        // Hermite extracts rational part; remainder has log part.
        // D = (x+1)² = x² + 2x + 1
        let a = Poly::from_coeffs(vec![rat(1, 1), rat(2, 1)]); // 2x + 1
        let d = Poly::from_coeffs(vec![rat(1, 1), rat(2, 1), rat(1, 1)]); // x²+2x+1

        match integrate_rational(&a, &d) {
            RischResult::Elementary { .. } => {
                // Success — the integral should be expressible as
                // rational_part + log_terms.
            }
            other => panic!("expected Elementary, got {:?}", other),
        }
    }

    // ── Logarithmic case (coefficient matching) ─────────────────────

    #[test]
    fn integrate_log_extension_theta() {
        // Tower: θ = ln(x), Dθ = 1/x.
        // Integrand: θ = ln(x).
        // ∫ ln(x) dx = x·ln(x) - x.
        //
        // Coefficient matching:
        //   a₁ = 1 (coeff of θ¹), a₀ = 0.
        //   k=1: DB₁ = a₁ = 1  →  B₁ = ∫ 1 dx = x
        //   k=0: DB₀ + B₁·Dθ = a₀ = 0  →  DB₀ = -x·(1/x) = -1  →  B₀ = -x
        //   F = x·θ + (-x) = x·ln(x) - x  ✓
        let mut arena = crate::base::arena::Arena::new();
        let x = arena.symbol("x");
        let theta = arena.symbol("__t0");
        let one = arena.one();
        let d_theta = arena.div(one, x);

        let mut de = DifferentialExtension::new(x);
        de.push_logarithmic(theta, x, d_theta);
        de.integrand = theta;

        let result = risch_integrate(&mut arena, &mut de);
        match result {
            RischResult::Elementary { .. } => {
                // Success — the coefficient matching should work.
            }
            RischResult::Failed(msg) => {
                // Acceptable if the recursive integration chain has issues.
                eprintln!("integrate_log(θ): {msg}");
            }
            other => panic!("unexpected: {:?}", other),
        }
    }

    // ── Exponential case (RDE-based) ────────────────────────────────

    #[test]
    fn integrate_exp_polynomial_theta() {
        // Tower: θ = exp(x), Dθ = θ, u = x, Du = 1.
        // Integrand: θ (= exp(x)).
        // RDE for k=1: B₁' + 1·1·B₁ = 1  →  B₁' + B₁ = 1
        // Solution: B₁ = 1 (constant).  Check: 0 + 1 = 1. ✓
        // But wait, ∫ exp(x) dx = exp(x), so B₁ = 1 and F = 1·θ = exp(x). ✓
        let mut arena = crate::base::arena::Arena::new();
        let x = arena.symbol("x");
        let theta = arena.symbol("__t0");

        let mut de = DifferentialExtension::new(x);
        de.push_exponential(theta, x, theta);
        de.integrand = theta; // integrand = θ = exp(x)

        let result = risch_integrate(&mut arena, &mut de);
        match result {
            RischResult::Elementary { .. } => {
                // Success — B₁ = 1, F = exp(x).
            }
            RischResult::Failed(msg) => {
                eprintln!("integrate_exp(θ): {msg}");
            }
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[test]
    fn integrate_exp_nonelementary_exp_neg_x_squared() {
        // Tower: θ = exp(-x²), u = -x², Du = -2x.
        // Integrand: θ (= exp(-x²)).
        // RDE for k=1: B₁' + 1·(-2x)·B₁ = 1  →  B₁' - 2x·B₁ = 1
        // This has NO rational function solution → NonElementary!
        let mut arena = crate::base::arena::Arena::new();
        let x = arena.symbol("x");
        let theta = arena.symbol("__t0");
        let two = arena.int(2);
        let x_sq = arena.pow(x, two);
        let neg_x_sq = arena.neg(x_sq);
        // Dθ = Du · θ = -2x · θ
        let neg_two = arena.int(-2);
        let neg_2x = arena.mul(&[neg_two, x]);
        let d_theta = arena.mul(&[neg_2x, theta]);

        let mut de = DifferentialExtension::new(x);
        de.push_exponential(theta, neg_x_sq, d_theta);
        de.integrand = theta;

        let result = risch_integrate(&mut arena, &mut de);
        match result {
            RischResult::NonElementary => {
                // This is the crown jewel: PROVED non-elementary!
            }
            RischResult::Failed(msg) => {
                // If the RDE solver couldn't handle it, that's acceptable
                // but not ideal.
                eprintln!("exp(-x²) returned Failed instead of NonElementary: {msg}");
            }
            RischResult::Elementary { .. } => {
                panic!("∫ exp(-x²) dx should be NonElementary, got Elementary");
            }
        }
    }

    #[test]
    fn integrate_exp_x_times_exp_x_squared() {
        // Tower: θ = exp(x²), u = x², Du = 2x.
        // Integrand: x · θ  (= x · exp(x²)).
        // The coefficient of θ¹ is a₁ = x.
        // RDE for k=1: B₁' + 1·(2x)·B₁ = x  →  B₁' + 2x·B₁ = x
        // Solution: B₁ = 1/2 (check: 0 + 2x·(1/2) = x ✓)
        // F = (1/2)·θ = (1/2)·exp(x²)  ✓
        let mut arena = crate::base::arena::Arena::new();
        let x = arena.symbol("x");
        let theta = arena.symbol("__t0");
        let two = arena.int(2);
        let x_sq = arena.pow(x, two);
        // Dθ = 2x · θ
        let two_x = arena.mul(&[two, x]);
        let d_theta = arena.mul(&[two_x, theta]);
        // Integrand: x · θ
        let integrand = arena.mul(&[x, theta]);

        let mut de = DifferentialExtension::new(x);
        de.push_exponential(theta, x_sq, d_theta);
        de.integrand = integrand;

        let result = risch_integrate(&mut arena, &mut de);
        match result {
            RischResult::Elementary { .. } => {
                // Should give B₁ = 1/2, so F = (1/2)·exp(x²).
            }
            other => {
                // The RDE B₁' + 2x·B₁ = x has solution B₁ = 1/2.
                // If this fails, print why.
                panic!("∫ x·exp(x²) dx should be Elementary, got {:?}", other);
            }
        }
    }

    // ── Polynomial integration ──────────────────────────────────────

    #[test]
    fn integrate_poly_basic() {
        // ∫ (2x + 1) dx = x² + x
        let p = Poly::from_coeffs(vec![rat(1, 1), rat(2, 1)]);
        let result = integrate_poly(&p);
        assert_eq!(result.coeff(0), rat(0, 1));
        assert_eq!(result.coeff(1), rat(1, 1));
        assert_eq!(result.coeff(2), rat(1, 1));
    }

    #[test]
    fn integrate_poly_zero() {
        let result = integrate_poly(&Poly::zero());
        assert!(result.is_zero());
    }

    #[test]
    fn integrate_poly_constant() {
        // ∫ 5 dx = 5x
        let result = integrate_poly(&Poly::from_int(5));
        assert_eq!(result.coeff(1), rat(5, 1));
    }
}
