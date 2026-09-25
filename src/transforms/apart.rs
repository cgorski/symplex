//! Partial fraction decomposition.
//!
//! Implements [`apart`], which decomposes a rational expression into a
//! sum of simpler fractions.
//!
//! # Algorithm
//!
//! 1. Decompose expression into numerator/denominator.
//! 2. Convert to polynomials, perform long division if needed.
//! 3. Factor the denominator over ℤ using `Poly::factor_over_z`.
//! 4. Use the extended Euclidean algorithm to separate fractions for
//!    coprime factors, and polynomial division for repeated factors.
//! 5. Assemble as `Σ R_ij / f_i^j` + polynomial remainder.
//!
//! For single irreducible factors of degree > 2, falls back to a
//! root-based approach that groups complex conjugate pairs into real
//! quadratic partial-fraction terms.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::poly::Poly;
use crate::poly::polybridge;
use crate::transforms::solve::Solution;

/// Decompose `expr` into partial fractions with respect to `var`.
///
/// Returns the expression unchanged if it's not a rational function
/// in `var` or if the denominator cannot be decomposed further.
pub(crate) fn apart(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    tracing::debug!("apart: decomposing expression");

    // Step 1: Decompose into numerator / denominator.
    let (numer, denom) = polybridge::as_numer_denom(arena, expr);

    // If denominator is 1, nothing to decompose.
    if denom == arena.one {
        return expr;
    }

    // Step 2: Convert both to polynomials.
    let numer_poly = match polybridge::expr_to_poly(arena, numer, var) {
        Some(p) => p,
        None => return expr,
    };
    let denom_poly = match polybridge::expr_to_poly(arena, denom, var) {
        Some(p) => p,
        None => return expr,
    };

    // The denominator must have degree >= 1.
    let denom_degree = match denom_poly.degree() {
        Some(d) if d >= 1 => d,
        _ => return expr,
    };

    // Step 3: If numerator degree >= denominator degree, do polynomial division.
    let (quotient, remainder) = if numer_poly.degree().unwrap_or(0) >= denom_degree {
        numer_poly.div_rem(&denom_poly)
    } else {
        (Poly::zero(), numer_poly.clone())
    };

    if remainder.is_zero() {
        // Exact division — return the polynomial quotient only.
        if quotient.is_zero() {
            return arena.zero;
        }
        return polybridge::poly_to_expr(arena, &quotient, var);
    }

    // Step 4: Factor the denominator over ℤ.
    let (content, factors) = denom_poly.factor_over_z();
    tracing::debug!("apart: found {} factors", factors.len());

    // If factoring produced nothing useful, try the root-based fallback.
    if factors.is_empty() {
        return assemble_with_quotient(arena, var, expr, &quotient, &remainder, &denom_poly);
    }

    // Single irreducible factor with multiplicity 1: no further
    // decomposition of the remainder is possible over ℚ.
    if factors.len() == 1 && factors[0].1 == 1 {
        // Even though we can't decompose the fraction part, there may
        // be a polynomial quotient from long division that we should
        // split off.  If so, return  quotient + remainder/denom.
        if !quotient.is_zero() {
            let rem_expr = polybridge::poly_to_expr(arena, &remainder, var);
            let den_expr = polybridge::poly_to_expr(arena, &denom_poly, var);
            let frac = arena.div(rem_expr, den_expr);
            let quot_expr = polybridge::poly_to_expr(arena, &quotient, var);
            return arena.add(&[quot_expr, frac]);
        }
        // Try RT refinement to split the irreducible factor into smaller pieces.
        if factors[0].0.degree().unwrap_or(0) > 2 {
            if let Some(rt_factors) = try_rt_refine(&remainder, &denom_poly) {
                let content_inv = if content.is_one() {
                    Ratio::one()
                } else {
                    Ratio::one() / &content
                };
                let scaled = remainder.scale(&content_inv);
                let rt_terms = decompose_poly_fraction(&scaled, &rt_factors);
                if !rt_terms.is_empty() {
                    let mut result_terms: Vec<ExprId> = Vec::new();
                    for (numer_p, factor_p, power) in &rt_terms {
                        if numer_p.is_zero() {
                            continue;
                        }
                        let numer_expr = polybridge::poly_to_expr(arena, numer_p, var);
                        let factor_expr = polybridge::poly_to_expr(arena, factor_p, var);
                        let denom_expr = if *power == 1 {
                            factor_expr
                        } else {
                            let exp = arena.int(*power as i64);
                            arena.pow(factor_expr, exp)
                        };
                        let term = arena.div(numer_expr, denom_expr);
                        result_terms.push(term);
                    }
                    if !quotient.is_zero() {
                        result_terms.push(polybridge::poly_to_expr(arena, &quotient, var));
                    }
                    if !result_terms.is_empty() {
                        return if result_terms.len() == 1 {
                            result_terms[0]
                        } else {
                            arena.add(&result_terms)
                        };
                    }
                }
            }
            // Fall back to the root-based approach.
            let fb = try_root_based_apart(arena, var, &quotient, &remainder, &denom_poly);
            if fb != expr {
                return fb;
            }
        }
        return expr;
    }

    // Step 5: Decompose the remainder over the factored denominator.
    //
    // denom_poly = content * Π f_i^e_i
    // remainder / denom_poly = (remainder / content) / (Π f_i^e_i)
    let content_inv = if content.is_one() {
        Ratio::one()
    } else {
        Ratio::one() / &content
    };
    let scaled_remainder = remainder.scale(&content_inv);

    let initial_terms = decompose_poly_fraction(&scaled_remainder, &factors);

    if initial_terms.is_empty() {
        return expr;
    }

    // Step 5b: Try RT refinement on any term whose denominator is still
    //          a single irreducible factor of degree > 2.
    let terms: Vec<(Poly, Poly, u32)> = initial_terms
        .into_iter()
        .flat_map(|(n, f, p)| {
            if p == 1
                && f.degree().unwrap_or(0) > 2
                && let Some(sub_factors) = try_rt_refine(&n, &f)
            {
                let sub_terms = decompose_poly_fraction(&n, &sub_factors);
                if !sub_terms.is_empty() {
                    return sub_terms;
                }
            }
            vec![(n, f, p)]
        })
        .collect();

    if terms.is_empty() {
        return expr;
    }

    // Step 6: Convert each (numer, factor, power) triple back to an expression.
    let mut result_terms: Vec<ExprId> = Vec::new();

    for (numer_p, factor_p, power) in &terms {
        if numer_p.is_zero() {
            continue;
        }
        let numer_expr = polybridge::poly_to_expr(arena, numer_p, var);
        let factor_expr = polybridge::poly_to_expr(arena, factor_p, var);
        let denom_expr = if *power == 1 {
            factor_expr
        } else {
            let exp = arena.int(*power as i64);
            arena.pow(factor_expr, exp)
        };
        let term = arena.div(numer_expr, denom_expr);
        result_terms.push(term);
    }

    // Add the polynomial quotient part if any.
    if !quotient.is_zero() {
        result_terms.push(polybridge::poly_to_expr(arena, &quotient, var));
    }

    if result_terms.is_empty() {
        return expr;
    }

    if result_terms.len() == 1 {
        result_terms[0]
    } else {
        arena.add(&result_terms)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Polynomial partial-fraction decomposition
// ═══════════════════════════════════════════════════════════════════════════

/// Decompose `numer / Π f_i^{e_i}` into a list of
/// `(R, f_i, j)` triples representing `R / f_i^j`.
///
/// All `f_i` must be pairwise coprime (guaranteed by `factor_over_z`).
fn decompose_poly_fraction(numer: &Poly, factors: &[(Poly, u32)]) -> Vec<(Poly, Poly, u32)> {
    if factors.is_empty() || numer.is_zero() {
        return vec![];
    }

    if factors.len() == 1 {
        let (f, e) = &factors[0];
        return decompose_single_factor(numer, f, *e);
    }

    // Split: first factor vs. the rest.
    let (f1, e1) = &factors[0];
    let rest = &factors[1..];

    let d1 = poly_pow(f1, *e1);
    let d2 = rest
        .iter()
        .fold(Poly::from_int(1), |acc, (f, e)| &acc * &poly_pow(f, *e));

    // Extended GCD: s*d1 + t*d2 = 1 (since d1 and d2 are coprime).
    let num_integer::ExtendedGcd { x: s, y: t, .. } = Poly::extended_gcd(&d1, &d2);

    // numer / (d1*d2) = (numer*t)/d1 + (numer*s)/d2
    // Reduce modulo the respective denominators.
    let nt = &(numer * &t);
    let ns = &(numer * &s);
    let r1 = nt.rem(&d1);
    let r2 = ns.rem(&d2);

    let mut result = decompose_single_factor(&r1, f1, *e1);
    result.extend(decompose_poly_fraction(&r2, rest));
    result
}

/// Decompose `numer / factor^power` into terms `R_k / factor^k`
/// for `k = 1, …, power` via repeated polynomial division.
fn decompose_single_factor(numer: &Poly, factor: &Poly, power: u32) -> Vec<(Poly, Poly, u32)> {
    if power == 0 || numer.is_zero() {
        return vec![];
    }

    let mut result = Vec::new();
    let mut remaining = numer.clone();

    for k in (1..=power).rev() {
        if remaining.is_zero() {
            break;
        }
        let (q, r) = remaining.div_rem(factor);
        if !r.is_zero() {
            result.push((r, factor.clone(), k));
        }
        remaining = q;
    }

    result
}

/// Raise a polynomial to a non-negative integer power.
fn poly_pow(p: &Poly, n: u32) -> Poly {
    if n == 0 {
        return Poly::from_int(1);
    }
    let mut result = p.clone();
    for _ in 1..n {
        result = &result * p;
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Root-based fallback for irreducible factors of degree > 2
// ═══════════════════════════════════════════════════════════════════════════

/// When `factor_over_z` can't decompose the denominator further
/// (single irreducible factor of high degree), try using `solve()` to
/// find roots and group complex conjugate pairs into real quadratic
/// partial-fraction terms.
fn try_root_based_apart(
    arena: &mut Arena,
    var: ExprId,
    quotient: &Poly,
    remainder: &Poly,
    denom_poly: &Poly,
) -> ExprId {
    let denom_expr = polybridge::poly_to_expr(arena, denom_poly, var);
    let remainder_expr = polybridge::poly_to_expr(arena, remainder, var);
    let orig_frac = arena.div(remainder_expr, denom_expr);

    let solutions = crate::transforms::solve::solve(arena, denom_expr, var);
    if solutions.is_empty() {
        return assemble_with_quotient_expr(arena, var, orig_frac, quotient);
    }

    let denom_deriv = denom_poly.derivative();

    // ── Numeric log-to-real: try clean decomposition via f64 eval ──
    if let Some(numeric_terms) =
        try_log_to_real_numeric(arena, var, &solutions, remainder, denom_poly, &denom_deriv)
    {
        tracing::debug!(
            "apart: numeric log_to_real succeeded with {} terms",
            numeric_terms.len()
        );
        let mut partial_terms = numeric_terms;
        if !quotient.is_zero() {
            partial_terms.push(polybridge::poly_to_expr(arena, quotient, var));
        }
        return match partial_terms.len() {
            0 => assemble_with_quotient_expr(arena, var, orig_frac, quotient),
            1 => partial_terms[0],
            _ => arena.add(&partial_terms),
        };
    }

    let mut partial_terms: Vec<ExprId> = Vec::new();
    let mut used: Vec<bool> = vec![false; solutions.len()];

    // ── Pass 1: Rational (Num) roots ───────────────────────────────────
    for (idx, sol) in solutions.iter().enumerate() {
        if used[idx] {
            continue;
        }
        if let ExprNode::Num(nid) = arena.node(sol.value) {
            let root_val = arena.num(*nid).clone();
            let numer_at_root = remainder.eval(&root_val);
            let denom_deriv_at_root = denom_deriv.eval(&root_val);
            if denom_deriv_at_root.is_zero() {
                continue;
            }
            let residue = numer_at_root / denom_deriv_at_root;
            if !residue.is_zero() {
                let residue_id = arena.num_ratio(residue.clone());
                let factor = arena.sub(var, sol.value);
                let term = arena.div(residue_id, factor);
                partial_terms.push(term);
            }
            used[idx] = true;
        }
    }

    // ── Pass 2: Irrational real roots and complex conjugate pairs ─────
    //
    // A root is classified by its value, not by its spelling: whether it
    // contains an explicit `i` says nothing (`√(1/2 − √5/2)` is imaginary,
    // a casus-irreducibilis root of a cubic is real), which up to 0.28.0
    // is how it was decided.  A non-real root is paired with its conjugate,
    // found by certified value ([`conjugate_by_value`]) or, failing that,
    // as a root whose sum and product with it are known real
    // ([`are_conjugates`]).  Any other root — real, or undecided — gets
    // its own residue term, which is exact over ℂ whatever its realness.
    let mut reals = crate::base::assumptions::AssumptionCache::new();
    let realness: Vec<Option<bool>> = solutions
        .iter()
        .map(|s| {
            crate::transforms::realness::constant_realness(
                arena,
                s.value,
                REALNESS_DIGITS,
                &mut reals,
            )
        })
        .collect();
    let values: Vec<Option<crate::base::bigcomplex::Complex>> = solutions
        .iter()
        .map(|s| crate::transforms::evalf::evalf_complex(arena, s.value, REALNESS_DIGITS).ok())
        .collect();
    for idx in 0..solutions.len() {
        if used[idx] {
            continue;
        }
        used[idx] = true;
        let root = solutions[idx].value;
        let partner = if realness[idx] == Some(false) {
            conjugate_by_value(&values, idx)
                .filter(|&j| !used[j])
                .or_else(|| {
                    ((idx + 1)..solutions.len()).find(|&j| {
                        !used[j]
                            && realness[j] == Some(false)
                            && are_conjugates(arena, root, solutions[j].value, &mut reals)
                    })
                })
        } else {
            None
        };
        if let Some(j) = partner {
            used[j] = true;
            let conj = solutions[j].value;
            if let Some(term) = build_conjugate_pair_term(
                arena,
                var,
                root,
                conj,
                remainder,
                denom_poly,
                &denom_deriv,
            ) {
                partial_terms.push(term);
                continue;
            }
            // No real form: the two residue terms are exact all the same.
            if let Some(term) = try_symbolic_residue_term(arena, var, conj, remainder, &denom_deriv)
            {
                partial_terms.push(term);
            }
        }
        if let Some(term) = try_symbolic_residue_term(arena, var, root, remainder, &denom_deriv) {
            partial_terms.push(term);
        }
    }

    if partial_terms.is_empty() {
        return assemble_with_quotient_expr(arena, var, orig_frac, quotient);
    }

    if !quotient.is_zero() {
        partial_terms.push(polybridge::poly_to_expr(arena, quotient, var));
    }

    if partial_terms.len() == 1 {
        partial_terms[0]
    } else {
        arena.add(&partial_terms)
    }
}

/// Digits to which [`crate::transforms::realness::constant_realness`] decides
/// whether a root is real (as the integrator's self-check does).
const REALNESS_DIGITS: u32 = 30;

/// Relative distance within which a root's certified value (to
/// [`REALNESS_DIGITS`] digits) is taken to be the conjugate of another's.
const CONJUGATE_TOL: f64 = 1e-20;

/// The index of the root that is the conjugate of root `idx`, from the
/// roots' certified values: `D` is real, so `conj(r)` is one of the roots,
/// the one whose value agrees with `conj(r)` to [`CONJUGATE_TOL`].  `None`
/// when a value is missing or the nearest root is not unique at that
/// tolerance (roots closer together than the digits resolve).
fn conjugate_by_value(
    values: &[Option<crate::base::bigcomplex::Complex>],
    idx: usize,
) -> Option<usize> {
    let z = values[idx].as_ref()?;
    let conj = (z.0.clone(), crate::base::bigcomplex::c_neg(z).1);
    let scale = crate::transforms::evalf::abs_to_f64(z)?.max(1.0);
    let mut found = None;
    for (k, v) in values.iter().enumerate() {
        if k == idx {
            continue;
        }
        let d = crate::transforms::evalf::distance_to_f64(&conj, v.as_ref()?)?;
        if d <= CONJUGATE_TOL * scale {
            if found.is_some() {
                return None;
            }
            found = Some(k);
        }
    }
    found
}

/// Are the non-real roots `r` and `s` conjugate?  They are exactly when
/// `r + s` and `r·s` are real: `r` and `s` are then the roots of
/// `t² − (r + s)·t + r·s`, a real quadratic, whose non-real roots are
/// conjugate.  Realness is established by
/// [`crate::transforms::realness::constant_realness`]; undecided is not.
fn are_conjugates(
    arena: &mut Arena,
    r: ExprId,
    s: ExprId,
    reals: &mut crate::base::assumptions::AssumptionCache,
) -> bool {
    let sum = arena.add(&[r, s]);
    let sum = crate::transforms::eval::eval(arena, sum);
    if crate::transforms::realness::constant_realness(arena, sum, REALNESS_DIGITS, reals)
        != Some(true)
    {
        return false;
    }
    let product = arena.mul(&[r, s]);
    let product = crate::transforms::eval::eval(arena, product);
    crate::transforms::realness::constant_realness(arena, product, REALNESS_DIGITS, reals)
        == Some(true)
}

/// Build a term `residue / (x - root)` where both residue and root
/// may be symbolic (irrational) expressions.  Returns `None` if
/// symbolic evaluation fails to simplify.
fn try_symbolic_residue_term(
    arena: &mut Arena,
    var: ExprId,
    root: ExprId,
    remainder: &Poly,
    denom_deriv: &Poly,
) -> Option<ExprId> {
    let numer_expr = polybridge::poly_to_expr(arena, remainder, var);
    let denom_deriv_expr = polybridge::poly_to_expr(arena, denom_deriv, var);

    let numer_at_root = crate::transforms::subs::subs(arena, numer_expr, var, root);
    let numer_at_root = crate::transforms::eval::eval(arena, numer_at_root);

    let denom_at_root = crate::transforms::subs::subs(arena, denom_deriv_expr, var, root);
    let denom_at_root = crate::transforms::eval::eval(arena, denom_at_root);

    if denom_at_root == arena.zero {
        return None;
    }

    let residue = arena.div(numer_at_root, denom_at_root);
    let residue = crate::transforms::eval::eval(arena, residue);

    if residue == arena.zero {
        return None;
    }

    let factor = arena.sub(var, root);
    Some(arena.div(residue, factor))
}

/// The real term `(A·x + B)/Q` of the conjugate roots `r`, `s` of `D`,
/// `Q = x² + p·x + q` with `p = −(r + s)`, `q = r·s` (real), computed
/// without the residues: with `D = Q·S`, `(A·x + B)·S ≡ N (mod Q)`.
/// Reducing `S ≡ s₁x + s₀` and `N ≡ n₁x + n₀` and using `x² ≡ −p·x − q`,
/// that is the linear system
///
/// ```text
/// (s₀ − p·s₁)·A + s₁·B = n₁
///      −q·s₁·A + s₀·B = n₀
/// ```
///
/// solved by Cramer's rule in exact expression arithmetic and checked
/// with [`numerator_verifies_mod_quadratic`].  `None` if `Q` does not
/// divide `D` exactly, the system is singular, or the check fails.
fn build_pair_term_from_both_roots(
    arena: &mut Arena,
    var: ExprId,
    r: ExprId,
    s: ExprId,
    remainder: &Poly,
    denom_poly: &Poly,
) -> Option<ExprId> {
    let sum = arena.add(&[r, s]);
    let neg_sum = arena.neg(sum);
    let p = alg_normalize(arena, neg_sum);
    let prod = arena.mul(&[r, s]);
    let q = alg_normalize(arena, prod);
    let (cofactor, rem_lin, rem_const) = div_by_monic_quadratic_alg(arena, denom_poly, p, q)?;
    if !alg_is_zero(arena, rem_lin) || !alg_is_zero(arena, rem_const) {
        return None;
    }
    let (s1, s0) = rem_by_monic_quadratic_alg(arena, cofactor.clone(), p, q);
    let deg_n = remainder.degree()?;
    let numer: Vec<ExprId> = (0..=deg_n)
        .rev()
        .map(|k| arena.num_ratio(remainder.coeff(k)))
        .collect();
    let (n1, n0) = rem_by_monic_quadratic_alg(arena, numer, p, q);
    // m = s₀ − p·s₁;  det = m·s₀ + q·s₁².
    let p_s1 = arena.mul(&[p, s1]);
    let m = arena.sub(s0, p_s1);
    let m = alg_normalize(arena, m);
    let m_s0 = arena.mul(&[m, s0]);
    let q_s1_s1 = arena.mul(&[q, s1, s1]);
    let det = arena.add(&[m_s0, q_s1_s1]);
    let det = alg_normalize(arena, det);
    if alg_is_zero(arena, det) {
        return None;
    }
    // A = (n₁·s₀ − n₀·s₁)/det;  B = (m·n₀ + q·s₁·n₁)/det.
    let n1_s0 = arena.mul(&[n1, s0]);
    let n0_s1 = arena.mul(&[n0, s1]);
    let a_num = arena.sub(n1_s0, n0_s1);
    let m_n0 = arena.mul(&[m, n0]);
    let q_s1_n1 = arena.mul(&[q, s1, n1]);
    let b_num = arena.add(&[m_n0, q_s1_n1]);
    let a = arena.div(a_num, det);
    let a = clean_coefficient(arena, a);
    let b = arena.div(b_num, det);
    let b = clean_coefficient(arena, b);
    if !numerator_verifies_mod_quadratic(arena, remainder, &cofactor, a, b, p, q) {
        return None;
    }
    let two = arena.int(2);
    let x_sq = arena.pow(var, two);
    let p_x = arena.mul(&[p, var]);
    let quad = arena.add(&[x_sq, p_x, q]);
    let a_x = arena.mul(&[a, var]);
    let lin = arena.add(&[a_x, b]);
    let lin = crate::transforms::eval::eval(arena, lin);
    Some(arena.div(lin, quad))
}

/// An algebraic-number quotient in a canonical, denominator-free form
/// where one is available: rationalised, then [`alg_normalize`]d.
fn clean_coefficient(arena: &mut Arena, e: ExprId) -> ExprId {
    let e = alg_normalize(arena, e);
    let r = arena.rationalize_denom_expr(e);
    alg_normalize(arena, r)
}

/// The remainder `r₁·x + r₀` of the polynomial with descending expression
/// coefficients `coeffs` modulo `x² + p·x + q`, by synthetic division.
fn rem_by_monic_quadratic_alg(
    arena: &mut Arena,
    mut coeffs: Vec<ExprId>,
    p: ExprId,
    q: ExprId,
) -> (ExprId, ExprId) {
    match coeffs.len() {
        0 => return (arena.zero, arena.zero),
        1 => return (arena.zero, coeffs[0]),
        _ => {}
    }
    let n = coeffs.len();
    for i in 0..(n - 2) {
        let lead = coeffs[i];
        if arena.is_zero_structural(lead) {
            continue;
        }
        let lp = arena.mul(&[lead, p]);
        let next = arena.sub(coeffs[i + 1], lp);
        coeffs[i + 1] = alg_normalize(arena, next);
        let lq = arena.mul(&[lead, q]);
        let next2 = arena.sub(coeffs[i + 2], lq);
        coeffs[i + 2] = alg_normalize(arena, next2);
    }
    (coeffs[n - 2], coeffs[n - 1])
}

/// Given a complex root `r` and its conjugate `conj`, build the real
/// partial-fraction term `(2a·x − 2aα − 2bβ) / (x² − 2αx + α²+β²)`
/// where `α + iβ = r` and `a + ib = residue(r)`.
///
/// When `r` does not split into explicit real and imaginary parts (`√`
/// of a negative constant, say, which `as_real_imag` keeps as `re(…)`,
/// `im(…)`), the term is built from the real quadratic `(x − r)(x − r̄)`
/// instead, by [`build_pair_term_from_both_roots`], where that verifies;
/// otherwise the `re(…)`/`im(…)` form, exact all the same, is kept.
#[allow(clippy::too_many_arguments)]
fn build_conjugate_pair_term(
    arena: &mut Arena,
    var: ExprId,
    root: ExprId,
    conj: ExprId,
    remainder: &Poly,
    denom_poly: &Poly,
    denom_deriv: &Poly,
) -> Option<ExprId> {
    let (alpha, beta) = crate::base::complex::as_real_imag(arena, root);
    let opaque = |arena: &Arena, e: ExprId| {
        crate::base::walk::post_order_ids(arena, e)
            .into_iter()
            .any(|id| matches!(arena.node(id), ExprNode::Re(_) | ExprNode::Im(_)))
    };
    if (opaque(arena, alpha) || opaque(arena, beta))
        && let Some(term) =
            build_pair_term_from_both_roots(arena, var, root, conj, remainder, denom_poly)
    {
        return Some(term);
    }
    // Evaluate the residue at the complex root symbolically.
    let numer_expr = polybridge::poly_to_expr(arena, remainder, var);
    let denom_deriv_expr = polybridge::poly_to_expr(arena, denom_deriv, var);

    let numer_at_root = crate::transforms::subs::subs(arena, numer_expr, var, root);
    let numer_at_root = crate::transforms::eval::eval(arena, numer_at_root);

    let denom_at_root = crate::transforms::subs::subs(arena, denom_deriv_expr, var, root);
    let denom_at_root = crate::transforms::eval::eval(arena, denom_at_root);

    if denom_at_root == arena.zero {
        return None;
    }

    let residue = arena.div(numer_at_root, denom_at_root);
    let residue = crate::transforms::eval::eval(arena, residue);

    // Split root and residue into real / imaginary parts.
    let alpha = crate::transforms::eval::eval(arena, alpha);
    let beta = crate::transforms::eval::eval(arena, beta);

    let (a, b) = crate::base::complex::as_real_imag(arena, residue);
    let a = crate::transforms::eval::eval(arena, a);
    let b = crate::transforms::eval::eval(arena, b);

    // Quadratic denominator: x² − 2α·x + (α² + β²)
    let two = arena.int(2);
    let two_alpha = arena.mul(&[two, alpha]);
    let alpha_sq = arena.mul(&[alpha, alpha]);
    let beta_sq = arena.mul(&[beta, beta]);
    let alpha_sq_plus_beta_sq = arena.add(&[alpha_sq, beta_sq]);
    let alpha_sq_plus_beta_sq = crate::transforms::eval::eval(arena, alpha_sq_plus_beta_sq);

    let x_sq = arena.pow(var, two);
    let tmp = arena.mul(&[two_alpha, var]);
    let neg_two_alpha_x = arena.neg(tmp);
    let quad_denom = arena.add(&[x_sq, neg_two_alpha_x, alpha_sq_plus_beta_sq]);
    let quad_denom = crate::transforms::eval::eval(arena, quad_denom);

    // Linear numerator: 2a·x + (−2aα − 2bβ)
    let two_a = arena.mul(&[two, a]);
    let two_a_alpha = arena.mul(&[two, a, alpha]);
    let two_b_beta = arena.mul(&[two, b, beta]);
    let tmp2 = arena.add(&[two_a_alpha, two_b_beta]);
    let const_part = arena.neg(tmp2);
    let const_part = crate::transforms::eval::eval(arena, const_part);

    let two_a_x = arena.mul(&[two_a, var]);
    let lin_numer = arena.add(&[two_a_x, const_part]);
    let lin_numer = crate::transforms::eval::eval(arena, lin_numer);

    Some(arena.div(lin_numer, quad_denom))
}

// ═══════════════════════════════════════════════════════════════════════════
// Numeric log-to-real conversion
// ═══════════════════════════════════════════════════════════════════════════

/// Try to decompose a rational function into clean real partial fractions
/// using numeric evaluation of roots and `nsimplify` for exact coefficient
/// recovery.
///
/// Every `nsimplify` candidate is a *guess* from a 16-digit float and is
/// accepted only after an **exact** check against the polynomials: a real
/// root `r` must satisfy `D(r) = 0`, a quadratic `x² + p·x + q` must divide
/// `D` exactly, and each numerator must satisfy the partial-fraction
/// identity modulo its factor (see [`alg_is_zero`]).  Without this, an
/// irrational pole is silently returned as the 12-digit rational it was
/// rounded to (`1/(x³ − 2)` → poles at `251984209979/200000000000`).
///
/// Returns `Some(terms)` if every root can be evaluated numerically and
/// every coefficient can be recovered as a clean closed-form expression
/// that verifies exactly.  Returns `None` on failure — the caller falls
/// back to the symbolic approach.
fn try_log_to_real_numeric(
    arena: &mut Arena,
    var: ExprId,
    solutions: &[Solution],
    remainder: &Poly,
    denom_poly: &Poly,
    denom_deriv: &Poly,
) -> Option<Vec<ExprId>> {
    let n_roots = solutions.len();
    tracing::debug!(
        "log_to_real: attempting numeric conversion for {} roots",
        n_roots
    );

    // Evaluate all roots to complex f64.
    let mut complex_vals: Vec<(f64, f64)> = Vec::with_capacity(n_roots);
    for sol in solutions {
        complex_vals.push(eval_complex_f64(arena, sol.value)?);
    }

    let tol = 1e-10;
    let mut terms: Vec<ExprId> = Vec::new();
    let mut used = vec![false; n_roots];
    let mut n_pairs = 0usize;

    // Pass A: Rational (exact Num) roots — handle exactly as before.
    for (idx, sol) in solutions.iter().enumerate() {
        if let ExprNode::Num(nid) = arena.node(sol.value) {
            let root_val = arena.num(*nid).clone();
            let numer_at = remainder.eval(&root_val);
            let denom_at = denom_deriv.eval(&root_val);
            if denom_at.is_zero() {
                continue;
            }
            let residue = numer_at / denom_at;
            if !residue.is_zero() {
                let res_expr = arena.num_ratio(residue.clone());
                let factor = arena.sub(var, sol.value);
                terms.push(arena.div(res_expr, factor));
            }
            used[idx] = true;
        }
    }

    // Pass B: Group remaining roots into conjugate pairs (or real irrationals).
    for i in 0..n_roots {
        if used[i] {
            continue;
        }
        let (re_i, im_i) = complex_vals[i];

        // Irrational real root.
        if im_i.abs() < tol {
            let root_exact = nsimplify_extended(arena, re_i, 1e-9);
            if !is_clean_expr(arena, root_exact) {
                tracing::trace!("log_to_real: irrational root nsimplify not clean, bailing");
                return None;
            }
            // Exact check: the candidate must be a root of the denominator.
            let d_at_root = poly_eval_alg(arena, denom_poly, root_exact);
            if !alg_is_zero(arena, d_at_root) {
                tracing::trace!("log_to_real: nsimplified root is not an exact root, bailing");
                return None;
            }
            let (n_re, _) = eval_poly_complex_f64(remainder, re_i, 0.0);
            let (d_re, _) = eval_poly_complex_f64(denom_deriv, re_i, 0.0);
            if d_re.abs() < 1e-15 {
                return None;
            }
            let res_f64 = n_re / d_re;
            if res_f64.abs() > tol {
                let res_exact = nsimplify_extended(arena, res_f64, 1e-9);
                if !is_clean_expr(arena, res_exact) {
                    return None;
                }
                // Exact check: res · D'(r) = N(r).
                let dd_at_root = poly_eval_alg(arena, denom_deriv, root_exact);
                let n_at_root = poly_eval_alg(arena, remainder, root_exact);
                let lhs = arena.mul(&[res_exact, dd_at_root]);
                let diff = arena.sub(lhs, n_at_root);
                if !alg_is_zero(arena, diff) {
                    tracing::trace!("log_to_real: nsimplified residue does not verify, bailing");
                    return None;
                }
                let factor = arena.sub(var, root_exact);
                terms.push(arena.div(res_exact, factor));
            }
            used[i] = true;
            continue;
        }

        // Complex root — find its conjugate partner.
        let mut conj_idx = None;
        for j in (i + 1)..n_roots {
            if used[j] {
                continue;
            }
            let (re_j, im_j) = complex_vals[j];
            if (re_i - re_j).abs() < tol && (im_i + im_j).abs() < tol {
                conj_idx = Some(j);
                break;
            }
        }
        let j = conj_idx?; // no conjugate found → bail
        used[i] = true;
        used[j] = true;
        n_pairs += 1;

        // Use the root with positive imaginary part.
        let (alpha, beta) = if im_i > 0.0 {
            (re_i, im_i)
        } else {
            (complex_vals[j].0, complex_vals[j].1.abs())
        };

        // Quadratic denominator: x² + p·x + q  with p = −2α, q = α²+β².
        let p_f64 = -2.0 * alpha;
        let q_f64 = alpha * alpha + beta * beta;
        let p_exact = nsimplify_extended(arena, p_f64, 1e-9);
        let q_exact = nsimplify_extended(arena, q_f64, 1e-9);

        if !is_clean_expr(arena, p_exact) || !is_clean_expr(arena, q_exact) {
            tracing::trace!("log_to_real: quadratic coeff nsimplify not clean, bailing");
            return None;
        }

        // Exact check: x² + p·x + q must divide the denominator.  The
        // quotient `cofactor` is needed again to verify the numerator.
        let (cofactor, rem_lin, rem_const) =
            div_by_monic_quadratic_alg(arena, denom_poly, p_exact, q_exact)?;
        if !alg_is_zero(arena, rem_lin) || !alg_is_zero(arena, rem_const) {
            tracing::trace!(
                "log_to_real: nsimplified quadratic does not divide D exactly, bailing"
            );
            return None;
        }

        tracing::debug!(
            "log_to_real: recovered exact quadratic x² + {}·x + {}",
            arena.display(p_exact),
            arena.display(q_exact)
        );

        let two = arena.int(2);
        let x_sq = arena.pow(var, two);
        let p_x = arena.mul(&[p_exact, var]);
        let quad_denom = arena.add(&[x_sq, p_x, q_exact]);
        let quad_denom = crate::transforms::eval::eval(arena, quad_denom);

        // Compute residue at α + βi via polynomial evaluation.
        let (res_re, res_im) = {
            let (n_re, n_im) = eval_poly_complex_f64(remainder, alpha, beta);
            let (d_re, d_im) = eval_poly_complex_f64(denom_deriv, alpha, beta);
            let mag_sq = d_re * d_re + d_im * d_im;
            if mag_sq < 1e-20 {
                return None;
            }
            (
                (n_re * d_re + n_im * d_im) / mag_sq,
                (n_im * d_re - n_re * d_im) / mag_sq,
            )
        };

        // Linear numerator: 2a·x + (−2aα − 2bβ).
        let a_coeff_f64 = 2.0 * res_re;
        let b_const_f64 = -2.0 * res_re * alpha - 2.0 * res_im * beta;
        let a_exact = nsimplify_extended(arena, a_coeff_f64, 1e-9);
        let b_exact = nsimplify_extended(arena, b_const_f64, 1e-9);

        if !is_clean_expr(arena, a_exact) || !is_clean_expr(arena, b_exact) {
            tracing::trace!("log_to_real: numerator coeff nsimplify not clean, bailing");
            return None;
        }

        // Exact check of the partial-fraction identity for this factor:
        // with D = Q·S, the term (A·x + B)/Q is right iff
        // (A·x + B)·S ≡ N (mod Q).
        if !numerator_verifies_mod_quadratic(
            arena, remainder, &cofactor, a_exact, b_exact, p_exact, q_exact,
        ) {
            tracing::trace!("log_to_real: nsimplified numerator does not verify, bailing");
            return None;
        }

        let a_x = arena.mul(&[a_exact, var]);
        let lin_numer = arena.add(&[a_x, b_exact]);
        let lin_numer = crate::transforms::eval::eval(arena, lin_numer);

        terms.push(arena.div(lin_numer, quad_denom));
    }

    // All roots must have been used.
    if used.iter().any(|&u| !u) {
        return None;
    }

    tracing::debug!(
        "log_to_real: {} roots, {} conjugate pairs",
        n_roots,
        n_pairs
    );
    Some(terms)
}

/// Evaluate a symbolic expression to a complex `(re, im)` pair at `f64`
/// precision (16 digits, rounded straight from the arbitrary-precision
/// value).
fn eval_complex_f64(arena: &Arena, expr: ExprId) -> Option<(f64, f64)> {
    let z = crate::transforms::evalf::evalf_complex64(arena, expr).ok()?;
    Some((z.re, z.im))
}

/// Evaluate a [`Poly`] at a complex point `z = (re, im)` using Horner's
/// method, returning `(re, im)` of the result.
fn eval_poly_complex_f64(poly: &Poly, z_re: f64, z_im: f64) -> (f64, f64) {
    let deg = match poly.degree() {
        Some(d) => d,
        None => return (0.0, 0.0),
    };
    let coeff_f64 =
        |k: usize| crate::base::numeric::ratio_to_f64(&poly.coeff(k)).unwrap_or(f64::NAN);
    let mut acc_re = coeff_f64(deg);
    let mut acc_im = 0.0;
    for k in (0..deg).rev() {
        let t_re = acc_re * z_re - acc_im * z_im;
        let t_im = acc_re * z_im + acc_im * z_re;
        acc_re = t_re + coeff_f64(k);
        acc_im = t_im;
    }
    (acc_re, acc_im)
}

/// Convert an `f64` to a high-precision rational expression suitable for
/// `nsimplify`.
fn f64_to_rational_expr(arena: &mut Arena, val: f64) -> ExprId {
    if val.abs() < 1e-15 {
        return arena.zero;
    }
    let scale = 1_000_000_000_000i64; // 10^12
    let numer = (val * scale as f64).round() as i64;
    arena.rational(numer, scale)
}

/// Extended `nsimplify`: tries the standard strategies first, then
/// `(a + b·√n) / c` patterns for small integers.
fn nsimplify_extended(arena: &mut Arena, val: f64, tol: f64) -> ExprId {
    if val.abs() < tol {
        return arena.zero;
    }
    // Standard nsimplify.
    let approx = f64_to_rational_expr(arena, val);
    let standard = crate::simplify::nsimplify::nsimplify(arena, approx, tol);
    if standard != approx {
        return standard;
    }
    // Extended: (a + b√n) / c.
    if let Some(expr) = find_rational_plus_sqrt(arena, val, tol) {
        return expr;
    }
    standard
}

/// Try to express `val` as `(a + b·√n) / c` for small integer `a, b, c, n`.
fn find_rational_plus_sqrt(arena: &mut Arena, val: f64, tol: f64) -> Option<ExprId> {
    for n in 2..=20i64 {
        let sqrt_n = (n as f64).sqrt();
        let sqrt_n_int = sqrt_n.round() as i64;
        if sqrt_n_int * sqrt_n_int == n {
            continue; // perfect square
        }
        for c in 1..=12i64 {
            let vc = val * c as f64;
            for b in -8..=8i64 {
                if b == 0 {
                    continue;
                }
                let a_f64 = vc - b as f64 * sqrt_n;
                let a = a_f64.round() as i64;
                if a.unsigned_abs() > 100 {
                    continue;
                }
                let reconstructed = (a as f64 + b as f64 * sqrt_n) / c as f64;
                if (reconstructed - val).abs() < tol {
                    return Some(build_rational_plus_sqrt_expr(arena, a, b, n, c));
                }
            }
        }
    }
    None
}

/// Build the symbolic expression `(a + b·√n) / c`, simplified.
fn build_rational_plus_sqrt_expr(arena: &mut Arena, a: i64, b: i64, n: i64, c: i64) -> ExprId {
    let half = arena.rational(1, 2);
    let n_expr = arena.int(n);
    let sqrt_n = arena.pow(n_expr, half);

    let b_sqrt = if b == 1 {
        sqrt_n
    } else if b == -1 {
        arena.neg(sqrt_n)
    } else {
        let b_expr = arena.int(b);
        arena.mul(&[b_expr, sqrt_n])
    };

    let numerator = if a == 0 {
        b_sqrt
    } else {
        let a_expr = arena.int(a);
        arena.add(&[a_expr, b_sqrt])
    };

    if c == 1 {
        numerator
    } else {
        let c_expr = arena.int(c);
        arena.div(numerator, c_expr)
    }
}

/// Quick check that an nsimplified expression is "clean" (small, no
/// nested radicals or huge fractions).
fn is_clean_expr(arena: &Arena, expr: ExprId) -> bool {
    let s = arena.display(expr).to_string();
    s.len() < 50 && !s.contains("cbrt")
}

// ── Exact verification of nsimplified candidates ─────────────────────────────
//
// `nsimplify` turns a 16-digit float into a candidate such as `(1 + √5)/2`
// — or, when nothing matches, into the 12-digit rational it was rounded
// to.  The helpers below do exact arithmetic on such candidates (rational
// numbers and radicals: `expand` distributes, `eval` folds `√5·√5 → 5` and
// collects like terms) so that a wrong guess is rejected instead of being
// emitted as an "exact" pole.

/// Canonicalise an algebraic-number expression: expand, then evaluate.
fn alg_normalize(arena: &mut Arena, e: ExprId) -> ExprId {
    let e = crate::transforms::expand::expand(arena, e);
    crate::transforms::eval::eval(arena, e)
}

/// Is `e` exactly zero after [`alg_normalize`]?
fn alg_is_zero(arena: &mut Arena, e: ExprId) -> bool {
    let e = alg_normalize(arena, e);
    arena.is_zero_structural(e) || arena.as_num(e).is_some_and(|r| r.is_zero())
}

/// `poly(r)` by Horner's rule with exact expression arithmetic.
fn poly_eval_alg(arena: &mut Arena, poly: &Poly, r: ExprId) -> ExprId {
    let Some(deg) = poly.degree() else {
        return arena.zero;
    };
    let mut acc = arena.num_ratio(poly.coeff(deg));
    for k in (0..deg).rev() {
        let prod = arena.mul(&[acc, r]);
        let ck = arena.num_ratio(poly.coeff(k));
        let sum = arena.add(&[prod, ck]);
        acc = alg_normalize(arena, sum);
    }
    acc
}

/// Divide `poly` by the monic quadratic `x² + p·x + q` (coefficients are
/// expressions) with exact expression arithmetic.
///
/// Returns `(quotient, r₁, r₀)` with the quotient's coefficients in
/// descending degree and the remainder `r₁·x + r₀`.  `None` if
/// `deg poly < 2`.
fn div_by_monic_quadratic_alg(
    arena: &mut Arena,
    poly: &Poly,
    p: ExprId,
    q: ExprId,
) -> Option<(Vec<ExprId>, ExprId, ExprId)> {
    let deg = poly.degree()?;
    if deg < 2 {
        return None;
    }
    // Synthetic division on descending coefficients.
    let mut work: Vec<ExprId> = (0..=deg)
        .rev()
        .map(|k| arena.num_ratio(poly.coeff(k)))
        .collect();
    for i in 0..=(deg - 2) {
        let lead = work[i];
        if arena.is_zero_structural(lead) {
            continue;
        }
        let lp = arena.mul(&[lead, p]);
        let next = arena.sub(work[i + 1], lp);
        work[i + 1] = alg_normalize(arena, next);
        let lq = arena.mul(&[lead, q]);
        let next2 = arena.sub(work[i + 2], lq);
        work[i + 2] = alg_normalize(arena, next2);
    }
    let rem_const = work[deg];
    let rem_lin = work[deg - 1];
    work.truncate(deg - 1);
    Some((work, rem_lin, rem_const))
}

/// Exact check of the partial-fraction identity for the factor
/// `Q = x² + p·x + q` of `D = Q·S`: the term `(a·x + b)/Q` is correct iff
/// `(a·x + b)·S ≡ N (mod Q)`.  `cofactor` is `S` in descending degree.
fn numerator_verifies_mod_quadratic(
    arena: &mut Arena,
    numer: &Poly,
    cofactor: &[ExprId],
    a: ExprId,
    b: ExprId,
    p: ExprId,
    q: ExprId,
) -> bool {
    // T = (a·x + b)·S − N, descending coefficients, degree deg S + 1.
    let deg_s = cofactor.len().saturating_sub(1);
    let deg_t = deg_s + 1;
    let mut t: Vec<ExprId> = vec![arena.zero; deg_t + 1];
    for (i, &s) in cofactor.iter().enumerate() {
        // s is the coefficient of x^(deg_s - i); a·x·s lands one slot
        // higher (index i), b·s at index i + 1.
        let a_s = arena.mul(&[a, s]);
        let sum = arena.add(&[t[i], a_s]);
        t[i] = alg_normalize(arena, sum);
        let b_s = arena.mul(&[b, s]);
        let sum = arena.add(&[t[i + 1], b_s]);
        t[i + 1] = alg_normalize(arena, sum);
    }
    if let Some(deg_n) = numer.degree() {
        if deg_n > deg_t {
            return false; // N should already be reduced below deg D
        }
        for k in 0..=deg_n {
            let idx = deg_t - k;
            let nk = arena.num_ratio(numer.coeff(k));
            let diff = arena.sub(t[idx], nk);
            t[idx] = alg_normalize(arena, diff);
        }
    }
    // (An empty numerator is the zero polynomial; nothing to subtract.)
    // Reduce T modulo Q by synthetic division and require a zero remainder.
    if deg_t < 2 {
        return t.iter().all(|&c| alg_is_zero(arena, c));
    }
    for i in 0..=(deg_t - 2) {
        let lead = t[i];
        if arena.is_zero_structural(lead) {
            continue;
        }
        let lp = arena.mul(&[lead, p]);
        let next = arena.sub(t[i + 1], lp);
        t[i + 1] = alg_normalize(arena, next);
        let lq = arena.mul(&[lead, q]);
        let next2 = arena.sub(t[i + 2], lq);
        t[i + 2] = alg_normalize(arena, next2);
    }
    alg_is_zero(arena, t[deg_t - 1]) && alg_is_zero(arena, t[deg_t])
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Return `quotient_poly_expr + frac_expr`, skipping zero quotients.
fn assemble_with_quotient_expr(
    arena: &mut Arena,
    var: ExprId,
    frac_expr: ExprId,
    quotient: &Poly,
) -> ExprId {
    if quotient.is_zero() {
        return frac_expr;
    }
    let quot_expr = polybridge::poly_to_expr(arena, quotient, var);
    arena.add(&[frac_expr, quot_expr])
}

/// Build the fallback expression `remainder/denom + quotient`.
fn assemble_with_quotient(
    arena: &mut Arena,
    var: ExprId,
    orig_expr: ExprId,
    quotient: &Poly,
    remainder: &Poly,
    denom_poly: &Poly,
) -> ExprId {
    let _ = (remainder, denom_poly); // suppress unused warnings in the simple path
    if quotient.is_zero() {
        return orig_expr;
    }
    let quot_expr = polybridge::poly_to_expr(arena, quotient, var);
    // The original expression already contains the remainder/denom;
    // we only need to add the quotient if it was separated by long division.
    // But `orig_expr` was the pre-division expression, so rebuild.
    let rem_expr = polybridge::poly_to_expr(arena, remainder, var);
    let den_expr = polybridge::poly_to_expr(arena, denom_poly, var);
    let frac = arena.div(rem_expr, den_expr);
    arena.add(&[frac, quot_expr])
}

// ═══════════════════════════════════════════════════════════════════════════
// Rothstein-Trager factor refinement
// ═══════════════════════════════════════════════════════════════════════════

/// Attempt to split a polynomial `denom` into finer factors using the
/// Rothstein-Trager resultant technique.
///
/// Given `N / D` with `D` squarefree of degree > 2, computes
/// `R(t) = res_x(D, N − t·D')`, factors `R(t)` over ℤ, and for each
/// factor `q_i(t)` computes the corresponding factor of `D` via
/// `gcd(D, res_t(N − t·D', q_i(t)))`.
///
/// Returns `Some(factors)` if a non-trivial factorisation is found,
/// `None` otherwise.
fn try_rt_refine(numer: &Poly, denom: &Poly) -> Option<Vec<(Poly, u32)>> {
    let d_deg = denom.degree()?;
    if d_deg <= 2 {
        return None;
    }

    let d_prime = denom.derivative();
    if d_prime.is_zero() {
        return None;
    }

    // R(t) = res_x(D, N − t·D')
    let r_poly = Poly::resultant_poly(denom, numer, &d_prime);
    if r_poly.is_zero() {
        return None;
    }

    // Factor R(t) over ℤ.
    let (_content, r_factors) = r_poly.factor_over_z();
    if r_factors.len() <= 1 {
        // R(t) is irreducible — no finer factorisation from this route.
        return None;
    }

    // For each factor q_i(t) of R(t), compute the corresponding factor of D.
    let mut d_subfactors: Vec<Poly> = Vec::new();
    let mut remaining = denom.make_monic();

    for (q_i, _mult) in &r_factors {
        let q_deg = q_i.degree().unwrap_or(0);
        if q_deg == 0 {
            continue;
        }
        if remaining.degree().unwrap_or(0) == 0 {
            break;
        }

        // H(x) = res_t(N(x) − t·D'(x),  q_i(t))
        let h = compute_res_t_poly(numer, &d_prime, q_i);
        if h.is_zero() {
            continue;
        }

        let g = Poly::gcd(&remaining, &h);
        let g_deg = g.degree().unwrap_or(0);
        if g_deg > 0 && g_deg < remaining.degree().unwrap_or(0) {
            d_subfactors.push(g.make_monic());
            remaining = remaining.div(&g).make_monic();
        }
    }

    // Include any leftover factor.
    if remaining.degree().unwrap_or(0) > 0 {
        d_subfactors.push(remaining.make_monic());
    }

    if d_subfactors.len() <= 1 {
        return None;
    }

    Some(d_subfactors.into_iter().map(|f| (f, 1u32)).collect())
}

/// Compute `res_t(g(x) − t·h(x),  q(t))` as a polynomial in `x`.
///
/// For `P(t) = −h·t + g`  (degree 1 in t) and `Q(t) = q(t)`  (degree d):
///
/// ```text
///   res_t(P, Q) = (−1)^d · Σ_{k=0}^{d} c_k · g^k · h^{d−k}
/// ```
///
/// where `c_k` is the coefficient of `t^k` in `q(t)`.
fn compute_res_t_poly(g: &Poly, h: &Poly, q: &Poly) -> Poly {
    let d = q.degree().unwrap_or(0);
    if d == 0 {
        return Poly::constant(q.coeff(0));
    }

    let mut result = Poly::zero();
    for k in 0..=d {
        let c_k = q.coeff(k);
        if c_k.is_zero() {
            continue;
        }
        let g_power = poly_pow(g, k as u32);
        let h_power = poly_pow(h, (d - k) as u32);
        let term = (&g_power * &h_power).scale(&c_k);
        result = &result + &term;
    }

    if !d.is_multiple_of(2) {
        result = -&result;
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Hermite reduction (standalone utility)
// ═══════════════════════════════════════════════════════════════════════════

/// Hermite reduction for `numer / factor^power` where `factor` is
/// squarefree.
///
/// Decomposes the integral:
///
/// ```text
///   ∫ numer / factor^power  dx  =  A / factor^{power−1}  +  ∫ B / factor  dx
/// ```
///
/// Returns `Some((A, factor^{power-1}, B, factor))` on success,
/// `None` if `power ≤ 1` or if `factor` is not squarefree.
///
/// The caller is responsible for integrating `B / factor` (the
/// logarithmic part with squarefree denominator).
#[allow(dead_code)]
pub(crate) fn hermite_reduce_factor(
    numer: &Poly,
    factor: &Poly,
    power: u32,
) -> Option<(Poly, Poly, Poly, Poly)> {
    if power <= 1 || factor.is_zero() || numer.is_zero() {
        return None;
    }

    let f = factor;
    let f_prime = f.derivative();

    // Extended GCD:  s·f + t·f' = 1  (works because f is squarefree).
    let num_integer::ExtendedGcd { gcd: g, y: t, .. } = Poly::extended_gcd(f, &f_prime);
    if g.degree().unwrap_or(0) != 0 {
        return None; // f is not squarefree
    }

    // Iterate, reducing multiplicity by 1 each step.
    //   numer / f^k  =  d/dx( C / ((1−k)·f^{k−1}) )  +  D / f^{k−1}
    //
    // where  C ≡ numer·t  (mod f),
    //        E = (numer − C·f') / f,
    //        D = E + C' / (k − 1).
    //
    // The antiderivative accumulates  Σ C_j / ((1−j)·f^{j−1}).
    let mut current_numer = numer.clone();
    let mut current_power = power;
    let mut rational_numer_acc = Poly::zero();

    while current_power > 1 {
        let k = current_power;

        let c = (&current_numer * &t).rem(f);

        let diff = &current_numer - &(&c * &f_prime);
        let (e, rem) = diff.div_rem(f);
        if !rem.is_zero() {
            return None; // unexpected; bail
        }

        let c_prime = c.derivative();
        let km1 = Ratio::from_integer(BigInt::from((k - 1) as i64));
        let d = &e + &c_prime.scale(&(Ratio::one() / &km1));

        // Accumulate C / ((1−k) · f^{k−1}) expressed over the common
        // denominator f^{power−1}:
        //   C / ((1−k) · f^{k−1})  =  C · f^{power−k} / ((1−k) · f^{power−1})
        let one_minus_k = Ratio::from_integer(BigInt::from(1 - k as i64));
        let f_extra = poly_pow(f, power - k);
        let contrib = (&c * &f_extra).scale(&(Ratio::one() / &one_minus_k));
        rational_numer_acc = &rational_numer_acc + &contrib;

        current_numer = d;
        current_power -= 1;
    }

    let rational_denom = poly_pow(f, power - 1);
    Some((rational_numer_acc, rational_denom, current_numer, f.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::arena::Arena;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }
    fn display(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    #[test]
    fn apart_simple_fraction() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // 1/(x^2 - 1) = 1/((x-1)(x+1)) = 1/2 * (1/(x-1) - 1/(x+1))
        let one = a.one;
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let x2_minus_1 = a.sub(x2, one);
        let expr = a.div(one, x2_minus_1);
        let result = apart(&mut a, expr, x);
        let s = display(&a, result);
        // Should contain terms with (x-1) and (x+1) denominators
        assert_ne!(s, display(&a, expr), "should be decomposed: {s}");
    }

    #[test]
    fn apart_already_simple() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x + 1 — no denominator, returns unchanged
        let one = a.one;
        let expr = a.add(&[x, one]);
        let result = apart(&mut a, expr, x);
        assert_eq!(display(&a, result), "x + 1");
    }

    #[test]
    fn quadratic_surd_factor_verifies_exactly() {
        // x⁴ + 1 = (x² + √2·x + 1)(x² − √2·x + 1): the surd quadratic divides
        // exactly, and the partial-fraction numerator (√2/4)·x + 1/2 for the
        // `+√2` factor verifies; a wrong numerator does not.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let four = a.int(4);
        let x4 = a.pow(x, four);
        let one = a.one;
        let d_expr = a.add(&[x4, one]);
        let d = polybridge::expr_to_poly(&a, d_expr, x).unwrap();
        let two = a.int(2);
        let sqrt2 = a.sqrt(two);
        let (cofactor, r1, r0) = div_by_monic_quadratic_alg(&mut a, &d, sqrt2, one).unwrap();
        assert!(alg_is_zero(&mut a, r1) && alg_is_zero(&mut a, r0));
        assert_eq!(cofactor.len(), 3);
        let n = polybridge::expr_to_poly(&a, one, x).unwrap();
        let quarter = a.rational(1, 4);
        let a_coeff = a.mul(&[quarter, sqrt2]);
        let b_coeff = a.rational(1, 2);
        assert!(numerator_verifies_mod_quadratic(
            &mut a, &n, &cofactor, a_coeff, b_coeff, sqrt2, one
        ));
        let wrong_b = a.rational(1, 3);
        assert!(!numerator_verifies_mod_quadratic(
            &mut a, &n, &cofactor, a_coeff, wrong_b, sqrt2, one
        ));
    }

    #[test]
    fn float_rounded_rational_is_not_an_exact_root() {
        // The cube root of 2 rounded to 12 digits is not a root of x³ − 2.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let x3 = a.pow(x, three);
        let two = a.int(2);
        let d_expr = a.sub(x3, two);
        let d = polybridge::expr_to_poly(&a, d_expr, x).unwrap();
        let rounded = a.rational(251_984_209_979, 200_000_000_000);
        let at = poly_eval_alg(&mut a, &d, rounded);
        assert!(!alg_is_zero(&mut a, at));
        let cbrt2 = a.cbrt(two);
        let at = poly_eval_alg(&mut a, &d, cbrt2);
        assert!(
            alg_is_zero(&mut a, at),
            "cbrt(2)³ − 2 should normalise to 0"
        );
    }

    #[test]
    fn apart_non_polynomial_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.sin(x);
        let result = apart(&mut a, expr, x);
        assert_eq!(display(&a, result), "sin(x)");
    }

    #[test]
    fn apart_with_quotient() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 / (x - 1) has degree(numer) >= degree(denom)
        // = x + 1 + 1/(x-1)
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let one = a.one;
        let denom = a.sub(x, one);
        let expr = a.div(x2, denom);
        let result = apart(&mut a, expr, x);
        let s = display(&a, result);
        assert_ne!(s, display(&a, expr), "should be decomposed: {s}");
    }

    #[test]
    fn apart_x_cubed_minus_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // 1/(x^3 - 1) should decompose into linear + quadratic terms
        // since x^3-1 = (x-1)(x^2+x+1)
        let one = a.one;
        let three = a.int(3);
        let x3 = a.pow(x, three);
        let denom = a.sub(x3, one);
        let expr = a.div(one, denom);
        let result = apart(&mut a, expr, x);
        let s = display(&a, result);
        // The decomposition should differ from the original
        assert_ne!(s, display(&a, expr), "should be decomposed: {s}");
    }

    #[test]
    fn apart_x4_plus_x2_plus_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x / (x^4 + x^2 + 1) should decompose since
        // x^4+x^2+1 = (x^2+x+1)(x^2-x+1)
        let one = a.one;
        let two = a.int(2);
        let four = a.int(4);
        let x2 = a.pow(x, two);
        let x4 = a.pow(x, four);
        let denom = a.add(&[x4, x2, one]);
        let expr = a.div(x, denom);
        let result = apart(&mut a, expr, x);
        let s = display(&a, result);
        assert_ne!(s, display(&a, expr), "should decompose: {s}");
    }

    /// Verify that apart produces a numerically correct decomposition
    /// for 1/(x²-1) at a few rational points.
    #[test]
    fn apart_x2_minus_1_numerical() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let denom = a.sub(x2, one);
        let expr = a.div(one, denom);
        let result = apart(&mut a, expr, x);

        // Evaluate at x = 2: original = 1/3, decomposed should also = 1/3.
        let val2 = a.int(2);
        let orig_at_2 = crate::transforms::subs::subs(&mut a, expr, x, val2);
        let orig_at_2 = crate::transforms::eval::eval(&mut a, orig_at_2);
        let decomp_at_2 = crate::transforms::subs::subs(&mut a, result, x, val2);
        let decomp_at_2 = crate::transforms::eval::eval(&mut a, decomp_at_2);
        assert_eq!(
            display(&a, orig_at_2),
            display(&a, decomp_at_2),
            "apart output must match original at x=2"
        );

        // Evaluate at x = 3.
        let val3 = a.int(3);
        let orig_at_3 = crate::transforms::subs::subs(&mut a, expr, x, val3);
        let orig_at_3 = crate::transforms::eval::eval(&mut a, orig_at_3);
        let decomp_at_3 = crate::transforms::subs::subs(&mut a, result, x, val3);
        let decomp_at_3 = crate::transforms::eval::eval(&mut a, decomp_at_3);
        assert_eq!(
            display(&a, orig_at_3),
            display(&a, decomp_at_3),
            "apart output must match original at x=3"
        );
    }

    // ── Rothstein-Trager infrastructure ─────────────────────────────

    /// Helper: build Ratio<BigInt> from i64.
    fn ri(n: i64) -> Ratio<BigInt> {
        Ratio::from_integer(BigInt::from(n))
    }

    #[test]
    fn rt_refine_x3_minus_x2_plus_x_minus_1() {
        // D = x³ − x² + x − 1 = (x−1)(x²+1).
        // R(t) factors over ℤ → RT finds the (x−1) and (x²+1) sub-factors.
        let d = Poly::from_coeffs(vec![ri(-1), ri(1), ri(-1), ri(1)]);
        let n = Poly::from_int(1);
        let result = try_rt_refine(&n, &d);
        assert!(result.is_some(), "RT should split x³−x²+x−1");
        let factors = result.unwrap();
        assert_eq!(factors.len(), 2, "should have 2 sub-factors: {factors:?}");
        // Verify product.
        let mut product = Poly::from_int(1);
        for (f, m) in &factors {
            for _ in 0..*m {
                product = &product * f;
            }
        }
        assert_eq!(
            product.make_monic(),
            d.make_monic(),
            "product of RT sub-factors must reconstruct D"
        );
    }

    #[test]
    fn compute_res_t_poly_linear_factor() {
        // g = 1, h = 3x² − 2x + 1 (= D' of x³−x²+x−1), q = 2t − 1.
        // res_t(1 − t·h, 2t − 1) = (−1)^1 · (c₀·h + c₁·1)
        //   where c₀ = −1, c₁ = 2  →  −(−h + 2) = h − 2 = 3x² − 2x − 1.
        let g = Poly::from_int(1);
        let h = Poly::from_coeffs(vec![ri(1), ri(-2), ri(3)]);
        let q = Poly::from_coeffs(vec![ri(-1), ri(2)]); // 2t − 1
        let res = compute_res_t_poly(&g, &h, &q);
        let expected = Poly::from_coeffs(vec![ri(-1), ri(-2), ri(3)]); // 3x²−2x−1
        assert_eq!(res, expected, "res_t should be 3x²−2x−1, got {res}");
    }

    #[test]
    fn hermite_reduce_1_over_x2_plus_1_squared() {
        // ∫ 1/(x²+1)² dx = x/(2(x²+1)) + ∫ 1/(2(x²+1)) dx
        let numer = Poly::from_int(1);
        let factor = Poly::from_coeffs(vec![ri(1), ri(0), ri(1)]); // x²+1
        let result = hermite_reduce_factor(&numer, &factor, 2);
        assert!(result.is_some(), "Hermite should reduce 1/(x²+1)²");
        let (a_num, a_den, b_num, b_den) = result.unwrap();
        // Rational part: A/D* = (x/2)/(x²+1) → A = x/2.
        // Check that 2·A = x.
        let two_a = a_num.scale(&ri(2));
        assert_eq!(
            two_a,
            Poly::x(),
            "rational numer should be x/2, got {a_num}"
        );
        // Rational denom = x²+1
        assert_eq!(a_den, factor, "rational denom should be x²+1");
        // Logarithmic part: B/Ds = (1/2)/(x²+1) → B = 1/2.
        let two_b = b_num.scale(&ri(2));
        assert_eq!(
            two_b,
            Poly::from_int(1),
            "log numer should be 1/2, got {b_num}"
        );
        assert_eq!(b_den, factor, "log denom should be x²+1");
    }

    #[test]
    fn hermite_reduce_trivial_for_squarefree() {
        // When power = 1, Hermite reduction should return None.
        let numer = Poly::from_int(1);
        let factor = Poly::from_coeffs(vec![ri(1), ri(0), ri(1)]);
        assert!(hermite_reduce_factor(&numer, &factor, 1).is_none());
    }

    #[test]
    fn hermite_reduce_1_over_x_minus_1_cubed() {
        // ∫ 1/(x−1)³ dx = −1/(2(x−1)²) + ∫ 0/(x−1) dx
        // So rational part = −1/(2(x−1)²), log part = 0/(x−1).
        let numer = Poly::from_int(1);
        let factor = Poly::from_coeffs(vec![ri(-1), ri(1)]); // x − 1
        let result = hermite_reduce_factor(&numer, &factor, 3);
        assert!(result.is_some(), "Hermite should reduce 1/(x−1)³");
        let (_a_num, a_den, b_num, _b_den) = result.unwrap();
        // Rational denom should be (x−1)²
        assert_eq!(a_den.degree(), Some(2));
        // Logarithmic numer should be zero
        assert!(
            b_num.is_zero(),
            "log numer should be 0 for 1/(x−1)³, got {b_num}"
        );
    }

    #[test]
    fn apart_quartic_decomposition_numerical() {
        // Verify that apart(1/(x⁵+1)) decomposes correctly and the
        // quartic piece is numerically accurate.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let five = a.int(5);
        let x5 = a.pow(x, five);
        let denom = a.add(&[x5, one]);
        let expr = a.div(one, denom);
        let result = apart(&mut a, expr, x);

        // Decomposed form should differ from original.
        assert_ne!(
            display(&a, result),
            display(&a, expr),
            "apart should decompose 1/(x⁵+1)"
        );

        // Numerical check at x = 2, 3, 4.
        for &val in &[2i64, 3, 4] {
            let v = a.int(val);
            let orig = crate::transforms::subs::subs(&mut a, expr, x, v);
            let orig = crate::transforms::eval::eval(&mut a, orig);
            let dec = crate::transforms::subs::subs(&mut a, result, x, v);
            let dec = crate::transforms::eval::eval(&mut a, dec);
            assert_eq!(
                display(&a, orig),
                display(&a, dec),
                "apart(1/(x⁵+1)) must match at x={val}"
            );
        }
    }
}
