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
//! All core algorithms operate at the [`Poly`] /
//! `Ratio<BigInt>` level.  The [`try_risch_rational`] function bridges
//! from the arena world to the polynomial world using the existing
//! [`polybridge`](crate::poly::polybridge) infrastructure.

pub mod hermite;
pub mod integrate;
pub mod log_to_real;
pub mod rde;
pub mod rothstein_trager;
pub mod tower;
pub mod tower_integrate;

use std::cell::Cell;

use num_traits::{One, Signed};

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::base::stage::stage;
use crate::poly::dense::Poly;
use crate::poly::generic::GenPoly;
use crate::poly::ratfn::RationalFn;

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

/// Is a `try_risch_rational` call in progress on this thread (so that a
/// nested one returns `None`)?  Part of the integrator's memo key.
pub(crate) fn rational_guard_active() -> bool {
    RISCH_GUARD.with(Cell::get)
}

// ═══════════════════════════════════════════════════════════════════════════
// Public result types
// ═══════════════════════════════════════════════════════════════════════════

/// A single logarithmic term in the integral.
#[derive(Clone, Debug)]
pub enum LogTerm {
    /// `coeff * ln(argument(x))` where `coeff` is rational.
    Rational { coeff: Q, argument: Poly },
    /// Symbolic sum over roots of a minimal polynomial:
    /// `Σ_{α: min_poly(α)=0} α * ln(gcd(denom, numer - α·denom'))`.
    ///
    /// Used when the resultant has irreducible factors of degree > 1
    /// whose roots are algebraic numbers not in ℚ.  `log_arg` is
    /// `S(t, x)` with `S(α, x) = gcd(denom, numer − α·denom')`: monic in
    /// `x`, of degree the multiplicity of `min_poly` in the resultant, its
    /// coefficients polynomials in `t` reduced modulo `min_poly`; `None`
    /// if the remainder sequence had no member of that degree.
    Algebraic {
        min_poly: Poly,
        log_arg: Option<GenPoly<RationalFn>>,
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
        /// Tower-level result as an arena expression (when the result
        /// involves transcendental extensions like exp/ln and can't be
        /// represented as a pure `Poly` in the base variable).
        arena_expr: Option<crate::base::node::ExprId>,
    },
    /// Proved that no elementary antiderivative exists.
    NonElementary,
    /// Hit an unimplemented case or internal limitation.
    Failed(String),
}

/// Result of attempting the Risch tower integration from the production
/// `integrate()` dispatcher.
pub(crate) enum TowerResult {
    /// Found an elementary antiderivative (as an arena expression).
    Elementary(crate::base::node::ExprId),
    /// Proved that no elementary antiderivative exists.
    NonElementary,
    /// Tower couldn't handle this expression — fall through to heuristics.
    NotApplicable,
}

/// Try to integrate an expression using the Risch algorithm's differential
/// extension tower.
///
/// This handles integrands containing `exp(...)` and `ln(...)` subexpressions
/// that the rule-based integrator couldn't handle.  Returns
/// `TowerResult::Elementary(id)` with the antiderivative,
/// `TowerResult::NonElementary` if no elementary antiderivative exists, or
/// `TowerResult::NotApplicable` if the expression can't be handled by the tower.
pub(crate) fn try_risch_tower(
    arena: &mut crate::base::arena::Arena,
    expr: crate::base::node::ExprId,
    var: crate::base::node::ExprId,
) -> TowerResult {
    // Build the differential extension tower.
    let mut de = match tower::build_tower(arena, expr, var) {
        Ok(de) => de,
        Err(_reason) => {
            tracing::debug!(reason = _reason.as_str(), "Risch tower: build_tower failed");
            return TowerResult::NotApplicable;
        }
    };

    // Base-level towers (no exp/ln) are handled by try_risch_rational
    // inside integrate_node — skip to avoid redundant work.
    if de.is_base_level() {
        return TowerResult::NotApplicable;
    }

    // Run the Risch integrator on the tower.
    let result = integrate::risch_integrate(arena, &mut de);

    match result {
        RischResult::Elementary {
            arena_expr: Some(id),
            rational_numer: _rn,
            rational_denom: _rd,
            log_terms: _lt,
        } => {
            tracing::debug!(
                tower_depth = de.depth(),
                "Risch tower: elementary antiderivative found"
            );
            TowerResult::Elementary(id)
        }
        RischResult::Elementary {
            arena_expr: None,
            rational_numer: _rn,
            rational_denom: _rd,
            log_terms: _lt,
        } => {
            // Base-level Poly result (shouldn't happen for tower-level,
            // but if it does, let heurisch handle it).
            TowerResult::NotApplicable
        }
        RischResult::NonElementary => TowerResult::NonElementary,
        RischResult::Failed(_msg) => {
            tracing::debug!(
                reason = _msg.as_str(),
                "Risch tower integration failed, falling through"
            );
            TowerResult::NotApplicable
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Arena ↔ Poly bridge
// ═══════════════════════════════════════════════════════════════════════════

/// The largest exponent `try_risch_rational` accepts on a power: larger
/// ones would make `expr_to_poly` build a dense polynomial of that degree
/// (`1/(x^1000000000 + 1)` asked for a billion coefficients).
const MAX_RATIONAL_EXPONENT: i64 = 1000;

/// Is `e` an element of `ℚ(x)` as written: built from rational numbers and
/// `var` by sums, products and integer powers (of magnitude at most
/// [`MAX_RATIONAL_EXPONENT`])?  Anything else — `i`, `π`, `√2`, a
/// function, another symbol — is refused before any expansion: the
/// integrator over `ℚ` cannot use it, and expanding it can be ruinous
/// (0.24: the partial fractions of `x³/(x⁸+1)` over its complex roots,
/// put over a common denominator, expanded towards 10⁸ terms).
fn is_rational_function_over_q(arena: &Arena, e: ExprId, var: ExprId) -> bool {
    crate::base::walk::post_order_ids(arena, e)
        .into_iter()
        .all(|id| match arena.node(id) {
            ExprNode::Num(_) | ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => true,
            ExprNode::Pow(_, exp) => arena.as_num(*exp).is_some_and(|r| {
                r.is_integer()
                    && i64::try_from(r.to_integer()).is_ok_and(|n| n.abs() <= MAX_RATIONAL_EXPONENT)
            }),
            _ => id == var,
        })
}

/// Does `e` contain `g^(−n)` (a negative integer power) of something that
/// depends on `var`?
fn has_negative_power_of(arena: &Arena, e: ExprId, var: ExprId) -> bool {
    crate::base::walk::post_order_ids(arena, e)
        .into_iter()
        .any(|id| match arena.node(id) {
            ExprNode::Pow(base, exp) => {
                arena
                    .as_num(*exp)
                    .is_some_and(|r| r.is_integer() && r.is_negative())
                    && crate::base::walk::contains(arena, *base, var)
            }
            _ => false,
        })
}

/// Try to integrate a rational function `A(x)/D(x) ∈ ℚ(x)` using Hermite
/// reduction followed by Rothstein–Trager / Lazard–Rioboo–Trager.
///
/// Returns `Some(result_expr_id)` if the expression is a rational function
/// over `ℚ` and integration succeeds.  Returns `None` if it is not, or if
/// the algebraic log terms cannot be represented.
pub fn try_risch_rational(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<ExprId> {
    if !is_rational_function_over_q(arena, expr, var) {
        return None;
    }
    // Recursion guard: if we're already inside try_risch_rational
    // (integrating an algebraic remainder), skip to avoid infinite loop.
    // The RAII guard resets the flag on drop, even during panics.
    let _guard = RischRecursionGuard::enter()?;

    // Decompose expr into numerator / denominator.  `as_numer_denom` only
    // lifts negative powers that are *factors*; a reciprocal inside a sum
    // (`x²·(x + 1/x)`) leaves the denominator at 1 and the rational function
    // unrecognised (0.22: `∫ x²(x + 1/x) dx` stayed unevaluated).  Combine
    // over a common denominator first in that case.
    let (mut numer_id, mut denom_id) = crate::poly::polybridge::as_numer_denom(arena, expr);
    if denom_id == arena.one() && has_negative_power_of(arena, expr, var) {
        let combined = crate::poly::polybridge::together_deep(arena, expr);
        (numer_id, denom_id) = crate::poly::polybridge::as_numer_denom(arena, combined);
        if denom_id == arena.one() {
            // The reciprocals cancelled (`x²(x + 1/x) = x³ + x`): a
            // polynomial, integrated termwise.
            let n = crate::transforms::expand::expand(arena, numer_id);
            let n = crate::transforms::eval::eval(arena, n);
            let p = crate::poly::polybridge::expr_to_poly(arena, n, var)?;
            let hr = hermite::hermite_reduce(&p, &Poly::from_int(1));
            return Some(crate::poly::polybridge::poly_to_expr(
                arena,
                &hr.g_numer,
                var,
            ));
        }
    }

    // Only proceed if there's a non-trivial denominator.
    if denom_id == arena.one() {
        return None;
    }

    // Expand and evaluate both numer and denom before converting to Poly.
    // This handles cases like (x²+1)² which need expansion to x⁴+2x²+1
    // before expr_to_poly can parse them as univariate polynomials.
    let numer_exp = stage!(
        arena,
        "expand_numer",
        numer_id,
        crate::transforms::expand::expand(arena, numer_id)
    );
    let numer_expanded = crate::transforms::eval::eval(arena, numer_exp);
    let denom_exp = stage!(
        arena,
        "expand_denom",
        denom_id,
        crate::transforms::expand::expand(arena, denom_id)
    );
    let denom_expanded = crate::transforms::eval::eval(arena, denom_exp);

    // Convert arena expressions to Poly using existing polybridge.
    let numer_poly = crate::poly::polybridge::expr_to_poly(arena, numer_expanded, var)?;
    let denom_poly = crate::poly::polybridge::expr_to_poly(arena, denom_expanded, var)?;

    // Skip if denominator is constant (not a rational function integration problem).
    if denom_poly.is_constant() {
        return None;
    }

    integrate_rational_function(arena, &numer_poly, &denom_poly, var, expr)
}

/// `k ≥ 2` if `A/D = x^(k−1)·F(x^k)`: every exponent of `D` is a multiple of
/// `k` and every exponent of `A` is `≡ k − 1 (mod k)`.  The largest such `k`.
fn power_substitution_degree(a: &Poly, d: &Poly) -> Option<usize> {
    let mut g = 0usize;
    for (e, c) in d.coeffs().iter().enumerate() {
        if !num_traits::Zero::is_zero(c) {
            g = num_integer::gcd(g, e);
        }
    }
    for (e, c) in a.coeffs().iter().enumerate() {
        if !num_traits::Zero::is_zero(c) {
            g = num_integer::gcd(g, e + 1);
        }
    }
    (g >= 2).then_some(g)
}

/// The polynomial `Σ c_{k·j + shift}·u^j` of the coefficients of `p` at the
/// exponents `≡ shift (mod k)`.
fn compress_exponents(p: &Poly, k: usize, shift: usize) -> Poly {
    Poly::from_coeffs(p.coeffs().iter().skip(shift).step_by(k).cloned().collect())
}

/// `∫ A/D dx` for `A, D ∈ ℚ[x]`, `D` not constant.  `label` only names the
/// integrand in stage traces.
fn integrate_rational_function(
    arena: &mut Arena,
    numer_poly: &Poly,
    denom_poly: &Poly,
    var: ExprId,
    label: ExprId,
) -> Option<ExprId> {
    let expr = label;

    // ── x^(k−1)·F(x^k): u = x^k gives ∫ F(u) du / k, of lower degree ──
    // `∫ x³/(x⁸ + 1) dx = ¼ ∫ du/(u² + 1) = atan(x⁴)/4` instead of eight
    // quadratic factors over ℚ(√(2 ± √2)) (SymPy's answer comes out the
    // same way, through the degree-4 log argument).
    if let Some(k) = power_substitution_degree(numer_poly, denom_poly) {
        let k_q = Q::from_integer(k.into());
        let a_u = compress_exponents(numer_poly, k, k - 1).scale(&k_q.recip());
        let d_u = compress_exponents(denom_poly, k, 0);
        let u = arena.symbol("__rr_u");
        let f_u = integrate_rational_function(arena, &a_u, &d_u, u, label)?;
        let k_id = arena.int(i64::try_from(k).ok()?);
        let x_k = arena.pow(var, k_id);
        let f_x = crate::transforms::subs::subs(arena, f_u, u, x_k);
        return Some(crate::transforms::eval::eval(arena, f_x));
    }

    // Phase 1: Hermite reduction.
    let hr = stage!(
        arena,
        "hermite",
        expr,
        hermite::hermite_reduce(numer_poly, denom_poly)
    );

    // Phase 2: Rothstein-Trager on the square-free remainder.
    let log_result = if hr.h_numer.is_zero() {
        rothstein_trager::LogPartResult { terms: vec![] }
    } else {
        stage!(
            arena,
            "rothstein_trager",
            expr,
            rothstein_trager::logarithmic_part(&hr.h_numer, &hr.h_denom)
        )
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
    let mut rational_log_parts: Vec<(Q, Poly)> = Vec::new();
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
            let coeff_id = arena.num_ratio(coeff.clone());
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
        // ── Algebraic factors: exact real forms (Lazard–Rioboo–Trager) ──
        //
        // Every irreducible factor q of the resultant of degree > 1 comes
        // with its log argument S(t, x) (`rothstein_trager::logarithmic_part`),
        // and contributes Σ_{q(α)=0} α·ln(S(α, x)).  A quadratic q is
        // converted exactly over ℚ[x] (`quadratic_log_to_real`); a q of
        // higher degree with S linear in x through its radical roots
        // (`log_to_real`).
        type Factor<'a> = (
            &'a Poly,
            Option<&'a GenPoly<RationalFn>>,
            Option<Vec<ExprId>>,
        );
        let mut per_factor: Vec<Factor<'_>> = Vec::new();
        for term in &log_result.terms {
            let LogTerm::Algebraic { min_poly, log_arg } = term else {
                continue;
            };
            let log_arg = log_arg.as_ref();
            let real_terms = match log_arg {
                Some(s) if min_poly.degree() == Some(2) => stage!(
                    arena,
                    "quadratic_log_to_real",
                    expr,
                    log_to_real::quadratic_log_to_real(arena, var, min_poly, s)
                ),
                Some(s) if s.degree() == Some(1) => stage!(
                    arena,
                    "log_to_real",
                    expr,
                    log_to_real::log_to_real(arena, var, min_poly, s)
                ),
                _ => None,
            };
            if real_terms.is_none() {
                tracing::debug!(
                    min_poly_degree = ?min_poly.degree(),
                    log_arg_degree = ?log_arg.and_then(GenPoly::degree),
                    "try_risch_rational: no real form for algebraic factor"
                );
            }
            per_factor.push((min_poly, log_arg, real_terms));
        }

        if per_factor.iter().all(|(_, _, t)| t.is_some()) {
            for (_, _, real_terms) in per_factor {
                terms.extend(real_terms.into_iter().flatten());
            }
        } else {
            // ── Fallback: algebraic remainder path, then RootSum ───
            //
            // Subtract the rational log contributions and integrate the
            // remaining algebraic part with the heuristic integrator (its
            // partial fractions over the roots may find a closed form, as
            // for `1/(x⁸ + 1)`).  If that stays unevaluated, the factors
            // without a real form become `RootSum`s — the exact answer in
            // implicit form — next to the real forms of the others.
            //
            // A_alg = h_numer − Σ c_i · v_i' · (h_denom / v_i): by the
            // residue theorem Π v_i divides A_alg, and after cancelling the
            // gcd only the algebraic factors remain in the denominator.
            tracing::debug!(
                h_numer_degree = ?hr.h_numer.degree(),
                h_denom_degree = ?hr.h_denom.degree(),
                n_rational_to_subtract = rational_log_parts.len(),
                "try_risch_rational: trying the algebraic remainder path"
            );

            let mut a_alg = hr.h_numer.clone();
            for (coeff, v_i) in &rational_log_parts {
                let v_i_prime = v_i.derivative();
                let cofactor = hr.h_denom.div(v_i);
                debug_assert!(
                    {
                        let product = &cofactor * v_i;
                        product == hr.h_denom
                    },
                    "h_denom / v_i must be exact polynomial division"
                );
                let contribution = (&v_i_prime * &cofactor).scale(coeff);
                a_alg = &a_alg - &contribution;
            }

            if a_alg.is_zero() {
                tracing::debug!(
                    "try_risch_rational: A_alg is zero — rational terms fully account for the integrand"
                );
            } else {
                let g = Poly::gcd(&a_alg, &hr.h_denom);
                let a_reduced = a_alg.div(&g);
                let d_reduced = hr.h_denom.div(&g);

                let alg_num_id = crate::poly::polybridge::poly_to_expr(arena, &a_reduced, var);
                let alg_den_id = crate::poly::polybridge::poly_to_expr(arena, &d_reduced, var);
                let algebraic_remainder = arena.div(alg_num_id, alg_den_id);

                let alg_integral =
                    crate::transforms::integrate::integrate(arena, algebraic_remainder, var);

                if !crate::base::walk::has_unevaluated(arena, alg_integral) {
                    terms.push(alg_integral);
                } else if per_factor.iter().all(|(_, s, _)| s.is_some()) {
                    for (min_poly, log_arg, real_terms) in per_factor {
                        match (real_terms, log_arg) {
                            (Some(real_terms), _) => terms.extend(real_terms),
                            (None, Some(s)) => {
                                tracing::debug!(
                                    min_poly_degree = ?min_poly.degree(),
                                    "try_risch_rational: emitting RootSum for algebraic factor"
                                );
                                terms.push(root_sum_term(arena, var, min_poly, s));
                            }
                            (None, None) => {}
                        }
                    }
                } else {
                    tracing::debug!(
                        "try_risch_rational: no log argument for RootSum, keeping the unevaluated algebraic integral"
                    );
                    terms.push(alg_integral);
                }
            }
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

/// `RootSum(q, t ↦ t·ln(S(t, x)))`, i.e. `Σ_{q(α)=0} α·ln(S(α, x))`: the
/// logarithmic part of an algebraic factor in implicit form.
fn root_sum_term(arena: &mut Arena, var: ExprId, q: &Poly, s: &GenPoly<RationalFn>) -> ExprId {
    let t = arena.symbol("__rs_t");
    let mut s_terms: Vec<ExprId> = Vec::new();
    for (k, c) in s.coeffs.iter().enumerate() {
        if c.numer().is_zero() {
            continue;
        }
        let c_expr = crate::poly::polybridge::ratfn_to_expr(arena, c, t);
        let x_k = match k {
            0 => arena.one(),
            1 => var,
            _ => {
                let k_id = arena.int(i64::try_from(k).unwrap_or(i64::MAX));
                arena.pow(var, k_id)
            }
        };
        s_terms.push(arena.mul(&[c_expr, x_k]));
    }
    let s_expr = arena.add(&s_terms);
    let ln_s = arena.ln(s_expr);
    let body = arena.mul(&[t, ln_s]);
    let q_expr = crate::poly::polybridge::poly_to_expr(arena, q, t);
    arena.intern(ExprNode::RootSum(q_expr, body, t))
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper: Ratio<BigInt> → ExprId
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_rational::Ratio;

    fn sym(arena: &mut Arena, name: &str) -> ExprId {
        arena.symbol(name)
    }

    fn display(arena: &Arena, id: ExprId) -> String {
        arena.display(id).to_string()
    }

    #[test]
    fn from_ratio_integer() {
        let mut arena = Arena::new();
        let expr = arena.num_ratio(Ratio::from_integer(BigInt::from(42)));
        assert_eq!(display(&arena, expr), "42");
    }

    #[test]
    fn from_ratio_fraction() {
        let mut arena = Arena::new();
        let expr = arena.num_ratio(Ratio::new(BigInt::from(3), BigInt::from(4)));
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

        fn r(n: i64, d: i64) -> Q {
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
        assert_eq!(
            c0_numer.degree(),
            Some(1),
            "constant term should be linear in t"
        );
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
        assert_eq!(
            roots.len(),
            2,
            "quadratic should have 2 roots, got {}",
            roots.len()
        );

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
                    "|Im| should be √3/6 ≈ {expected_im}, got {}",
                    im_v.abs()
                );
                if im_v > 0.0 {
                    pos_im_found = true;
                } else {
                    neg_im_found = true;
                }
            } else {
                panic!(
                    "Could not evaluate root to f64: {}",
                    display(&arena, root.value)
                );
            }
        }

        assert!(
            pos_im_found,
            "should have a root with positive imaginary part"
        );
        assert!(
            neg_im_found,
            "should have a root with negative imaginary part"
        );
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
