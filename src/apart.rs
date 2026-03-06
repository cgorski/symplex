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

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode};
use crate::poly::Poly;
use crate::polybridge;

/// Decompose `expr` into partial fractions with respect to `var`.
///
/// Returns the expression unchanged if it's not a rational function
/// in `var` or if the denominator cannot be decomposed further.
pub(crate) fn apart(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
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
        // Try the root-based fallback for higher-degree factors.
        if factors[0].0.degree().unwrap_or(0) > 2 {
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

    let terms = decompose_poly_fraction(&scaled_remainder, &factors);

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
fn decompose_poly_fraction(
    numer: &Poly,
    factors: &[(Poly, u32)],
) -> Vec<(Poly, Poly, u32)> {
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
    let d2 = rest.iter().fold(Poly::from_int(1), |acc, (f, e)| {
        &acc * &poly_pow(f, *e)
    });

    // Extended GCD: s*d1 + t*d2 = 1 (since d1 and d2 are coprime).
    let (s, t, _g) = Poly::extended_gcd(&d1, &d2);

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
fn decompose_single_factor(
    numer: &Poly,
    factor: &Poly,
    power: u32,
) -> Vec<(Poly, Poly, u32)> {
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

    let solutions = crate::solve::solve(arena, denom_expr, var);
    if solutions.is_empty() {
        return assemble_with_quotient_expr(arena, var, orig_frac, quotient);
    }

    let denom_deriv = denom_poly.derivative();
    let i_unit = arena.i_unit;

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
                let residue_id = rational_to_expr(arena, &residue);
                let factor = arena.sub(var, sol.value);
                let term = arena.div(residue_id, factor);
                partial_terms.push(term);
            }
            used[idx] = true;
        }
    }

    // ── Pass 2: Complex conjugate pairs ────────────────────────────────
    for idx in 0..solutions.len() {
        if used[idx] {
            continue;
        }
        let root = solutions[idx].value;
        let is_complex = crate::walk::contains(arena, root, i_unit);
        if !is_complex {
            // Irrational real root — compute residue symbolically.
            if let Some(term) = try_symbolic_residue_term(arena, var, root, remainder, &denom_deriv)
            {
                partial_terms.push(term);
            }
            used[idx] = true;
            continue;
        }

        // Find the conjugate partner.
        let neg_i = arena.neg(i_unit);
        let conj_root = crate::subs::subs(arena, root, i_unit, neg_i);
        let conj_root = crate::eval::eval(arena, conj_root);

        let mut conj_idx = None;
        for j in (idx + 1)..solutions.len() {
            if used[j] {
                continue;
            }
            if solutions[j].value == conj_root {
                conj_idx = Some(j);
                break;
            }
        }

        if let Some(j) = conj_idx {
            used[idx] = true;
            used[j] = true;

            // Compute quadratic factor and (Ax+B) numerator from the
            // conjugate pair.
            if let Some(term) = build_conjugate_pair_term(
                arena,
                var,
                root,
                remainder,
                &denom_deriv,
            ) {
                partial_terms.push(term);
            }
        } else {
            // No conjugate found — try a plain symbolic residue.
            if let Some(term) =
                try_symbolic_residue_term(arena, var, root, remainder, &denom_deriv)
            {
                partial_terms.push(term);
            }
            used[idx] = true;
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

    let numer_at_root = crate::subs::subs(arena, numer_expr, var, root);
    let numer_at_root = crate::eval::eval(arena, numer_at_root);

    let denom_at_root = crate::subs::subs(arena, denom_deriv_expr, var, root);
    let denom_at_root = crate::eval::eval(arena, denom_at_root);

    if denom_at_root == arena.zero {
        return None;
    }

    let residue = arena.div(numer_at_root, denom_at_root);
    let residue = crate::eval::eval(arena, residue);

    if residue == arena.zero {
        return None;
    }

    let factor = arena.sub(var, root);
    Some(arena.div(residue, factor))
}

/// Given a complex root `r` (with conjugate `r̄`), build the real
/// partial-fraction term `(2a·x − 2aα − 2bβ) / (x² − 2αx + α²+β²)`
/// where `α + iβ = r` and `a + ib = residue(r)`.
fn build_conjugate_pair_term(
    arena: &mut Arena,
    var: ExprId,
    root: ExprId,
    remainder: &Poly,
    denom_deriv: &Poly,
) -> Option<ExprId> {
    // Evaluate the residue at the complex root symbolically.
    let numer_expr = polybridge::poly_to_expr(arena, remainder, var);
    let denom_deriv_expr = polybridge::poly_to_expr(arena, denom_deriv, var);

    let numer_at_root = crate::subs::subs(arena, numer_expr, var, root);
    let numer_at_root = crate::eval::eval(arena, numer_at_root);

    let denom_at_root = crate::subs::subs(arena, denom_deriv_expr, var, root);
    let denom_at_root = crate::eval::eval(arena, denom_at_root);

    if denom_at_root == arena.zero {
        return None;
    }

    let residue = arena.div(numer_at_root, denom_at_root);
    let residue = crate::eval::eval(arena, residue);

    // Split root and residue into real / imaginary parts.
    let (alpha, beta) = crate::complex::as_real_imag(arena, root);
    let alpha = crate::eval::eval(arena, alpha);
    let beta = crate::eval::eval(arena, beta);

    let (a, b) = crate::complex::as_real_imag(arena, residue);
    let a = crate::eval::eval(arena, a);
    let b = crate::eval::eval(arena, b);

    // Quadratic denominator: x² − 2α·x + (α² + β²)
    let two = arena.int(2);
    let two_alpha = arena.mul(&[two, alpha]);
    let alpha_sq = arena.mul(&[alpha, alpha]);
    let beta_sq = arena.mul(&[beta, beta]);
    let alpha_sq_plus_beta_sq = arena.add(&[alpha_sq, beta_sq]);
    let alpha_sq_plus_beta_sq = crate::eval::eval(arena, alpha_sq_plus_beta_sq);

    let x_sq = arena.pow(var, two);
    let tmp = arena.mul(&[two_alpha, var]);
    let neg_two_alpha_x = arena.neg(tmp);
    let quad_denom = arena.add(&[x_sq, neg_two_alpha_x, alpha_sq_plus_beta_sq]);
    let quad_denom = crate::eval::eval(arena, quad_denom);

    // Linear numerator: 2a·x + (−2aα − 2bβ)
    let two_a = arena.mul(&[two, a]);
    let two_a_alpha = arena.mul(&[two, a, alpha]);
    let two_b_beta = arena.mul(&[two, b, beta]);
    let tmp2 = arena.add(&[two_a_alpha, two_b_beta]);
    let const_part = arena.neg(tmp2);
    let const_part = crate::eval::eval(arena, const_part);

    let two_a_x = arena.mul(&[two_a, var]);
    let lin_numer = arena.add(&[two_a_x, const_part]);
    let lin_numer = crate::eval::eval(arena, lin_numer);

    Some(arena.div(lin_numer, quad_denom))
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn rational_to_expr(arena: &mut Arena, r: &Ratio<BigInt>) -> ExprId {
    let nid = arena.intern_num(r.clone());
    arena.intern(ExprNode::Num(nid))
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;

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
        let orig_at_2 = crate::subs::subs(&mut a, expr, x, val2);
        let orig_at_2 = crate::eval::eval(&mut a, orig_at_2);
        let decomp_at_2 = crate::subs::subs(&mut a, result, x, val2);
        let decomp_at_2 = crate::eval::eval(&mut a, decomp_at_2);
        assert_eq!(
            display(&a, orig_at_2),
            display(&a, decomp_at_2),
            "apart output must match original at x=2"
        );

        // Evaluate at x = 3.
        let val3 = a.int(3);
        let orig_at_3 = crate::subs::subs(&mut a, expr, x, val3);
        let orig_at_3 = crate::eval::eval(&mut a, orig_at_3);
        let decomp_at_3 = crate::subs::subs(&mut a, result, x, val3);
        let decomp_at_3 = crate::eval::eval(&mut a, decomp_at_3);
        assert_eq!(
            display(&a, orig_at_3),
            display(&a, decomp_at_3),
            "apart output must match original at x=3"
        );
    }
}
