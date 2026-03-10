//! Polynomial factoring via rational root finding + Kronecker's method.
//!
//! This module implements [`factor`], which attempts to factor a
//! polynomial expression over ℤ into a product of irreducible factors
//! with their multiplicities.
//!
//! # Algorithm
//!
//! 1. Convert the expression to a [`Poly`] in the given variable.
//! 2. Call [`Poly::factor_over_z`] which performs:
//!    a. Content extraction (GCD of all coefficients).
//!    b. Square-free decomposition (Yun's algorithm).
//!    c. Rational root extraction (Rational Root Theorem).
//!    d. Higher-degree factor finding (Kronecker's method).
//! 3. Convert the factors back to expressions and build the product.
//! 4. If the poly-level factoring finds nothing new, fall back to
//!    the expression-level solver for rational roots.
//!
//! # Improvements over the previous version
//!
//! - Finds non-linear factors (e.g. x²+1 in x⁴−1).
//! - Handles repeated roots via square-free decomposition.
//! - Extracts content for any polynomial, including linear ones.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::poly::Poly;
use crate::poly::polybridge;

/// Factor a polynomial expression into a product of irreducible factors.
///
/// Returns the expression unchanged if it is not polynomial in `var`
/// or if no factorisation can be found.
pub(crate) fn factor(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    // Step 1: Convert to polynomial.
    let poly = match polybridge::expr_to_poly(arena, expr, var) {
        Some(p) => p,
        None => return expr,
    };

    // Trivial cases.
    if poly.is_zero() || poly.is_constant() {
        return expr;
    }

    let degree = match poly.degree() {
        Some(d) => d,
        None => return expr,
    };

    // Step 2: Try enhanced polynomial factoring (SFD + rational roots + Kronecker).
    let (content, factors) = poly.factor_over_z();

    let nontrivial = factors.len() > 1 || factors.iter().any(|(_, m)| *m > 1) || !content.is_one();

    if nontrivial {
        return build_factored_expr(arena, var, &content, &factors);
    }

    // Step 3: No non-trivial poly-level factoring found.
    if degree <= 1 {
        return expr; // Linear or constant — already factored.
    }

    // Step 4: Fallback — use the expression-level solver for rational roots.
    // This catches edge cases that the poly-level approach may miss.
    factor_via_solver(arena, expr, var, &poly)
}

/// Build a symbolic expression from the factored polynomial form.
///
/// Constructs `content * ∏ factor^mult`.
fn build_factored_expr(
    arena: &mut Arena,
    var: ExprId,
    content: &Ratio<BigInt>,
    factors: &[(Poly, u32)],
) -> ExprId {
    let mut parts: Vec<ExprId> = Vec::new();

    // Add content if it isn't 1.
    if !content.is_one() {
        let nid = arena.intern_num(content.clone());
        let cid = arena.intern(ExprNode::Num(nid));
        parts.push(cid);
    }

    // Add each factor (possibly raised to a power).
    for (factor, mult) in factors {
        let fexpr = polybridge::poly_to_expr(arena, factor, var);
        if *mult == 1 {
            parts.push(fexpr);
        } else {
            let exp = arena.int(*mult as i64);
            parts.push(arena.pow(fexpr, exp));
        }
    }

    match parts.len() {
        0 => arena.one,
        1 => parts[0],
        _ => arena.mul(&parts),
    }
}

/// Fallback factoring using the expression-level equation solver.
///
/// This is the original algorithm: find rational roots via `solve`,
/// divide out the corresponding linear factors, and return the product.
fn factor_via_solver(arena: &mut Arena, expr: ExprId, var: ExprId, poly: &Poly) -> ExprId {
    // Extract content (GCD of all coefficients).
    let content = poly_content(poly);
    let primitive = if content != Ratio::one() {
        poly.scale(&(Ratio::one() / &content))
    } else {
        poly.clone()
    };

    // Find rational roots of the primitive polynomial via the solver.
    let prim_expr = polybridge::poly_to_expr(arena, &primitive, var);
    let solutions = crate::transforms::solve::solve(arena, prim_expr, var);
    if solutions.is_empty() {
        return expr; // No rational roots found.
    }

    // For each root, divide out (x − root) repeatedly (handles multiplicity).
    let mut remaining = primitive;
    let mut factors: Vec<ExprId> = Vec::new();

    for sol in &solutions {
        // Extract the rational value of the root.
        let root_rational = match arena.node(sol.value) {
            ExprNode::Num(nid) => arena.num(*nid).clone(),
            _ => continue, // Non-rational root, skip.
        };

        // Build the linear factor (x − r) as a Poly.
        let linear = Poly::from_coeffs(vec![-root_rational.clone(), Ratio::one()]);

        // Divide out this factor as many times as possible (multiplicity).
        loop {
            let (quotient, rem) = remaining.div_rem(&linear);
            if rem.is_zero() {
                remaining = quotient;
                let root_id = sol.value;
                let factor_expr = arena.sub(var, root_id);
                factors.push(factor_expr);
            } else {
                break;
            }
        }
    }

    if factors.is_empty() {
        return expr; // No roots produced clean division.
    }

    // Build the result as content × product_of_factors × remainder.
    let remainder_expr = polybridge::poly_to_expr(arena, &remaining, var);
    let remainder_is_one = arena.is_one_structural(remainder_expr);

    let mut all_parts: Vec<ExprId> = Vec::new();

    // Add content if ≠ 1.
    if content != Ratio::one() {
        let content_nid = arena.intern_num(content);
        let content_id = arena.intern(ExprNode::Num(content_nid));
        all_parts.push(content_id);
    }

    all_parts.extend_from_slice(&factors);

    if !remainder_is_one {
        all_parts.push(remainder_expr);
    }

    if all_parts.len() == 1 {
        all_parts[0]
    } else {
        arena.mul(&all_parts)
    }
}

/// Compute the content (GCD of all coefficients) of a polynomial.
fn poly_content(poly: &Poly) -> Ratio<BigInt> {
    let coeffs = poly.coeffs();
    if coeffs.is_empty() {
        return Ratio::one();
    }

    // Start with the first nonzero coefficient.
    let mut result: Option<Ratio<BigInt>> = None;
    for c in coeffs {
        if c.is_zero() {
            continue;
        }
        match &result {
            None => result = Some(c.abs()),
            Some(g) => {
                let a = g.clone();
                let b = c.abs();
                result = Some(rational_gcd(&a, &b));
            }
        }
    }

    result.unwrap_or_else(Ratio::one)
}

/// GCD of two positive rationals: gcd(a/b, c/d) = gcd(a,c) / lcm(b,d).
fn rational_gcd(a: &Ratio<BigInt>, b: &Ratio<BigInt>) -> Ratio<BigInt> {
    use num_integer::Integer;
    let num_gcd = a.numer().gcd(b.numer());
    let den_lcm = a.denom().lcm(b.denom());
    Ratio::new(num_gcd, den_lcm)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

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
    fn factor_x_squared_minus_one() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 - 1 = (x - 1)(x + 1)
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let one = a.one;
        let expr = a.sub(x2, one);
        let result = factor(&mut a, expr, x);
        let s = display(&a, result);
        // Should be a product involving (x - 1) and (x + 1) or equivalent
        // The canonical form might be (x + 1)*(-1 + x) or similar
        assert!(!s.contains("x^2"), "should be factored (no x^2): {s}");
    }

    #[test]
    fn factor_quadratic_two_roots() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 - 5x + 6 = (x - 2)(x - 3)
        let two = a.int(2);
        let five = a.int(5);
        let six = a.int(6);
        let x2 = a.pow(x, two);
        let five_x = a.mul(&[five, x]);
        let minus_5x = a.neg(five_x);
        let expr = a.add(&[x2, minus_5x, six]);
        let result = factor(&mut a, expr, x);
        let s = display(&a, result);
        assert!(!s.contains("x^2"), "should be factored (no x^2): {s}");
    }

    #[test]
    fn factor_already_linear() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x + 1 — already linear, should stay unchanged
        let one = a.one;
        let expr = a.add(&[x, one]);
        let result = factor(&mut a, expr, x);
        assert_eq!(display(&a, result), "x + 1");
    }

    #[test]
    fn factor_no_rational_roots() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 + 1 — no real rational roots
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let one = a.one;
        let expr = a.add(&[x2, one]);
        let result = factor(&mut a, expr, x);
        // Should remain unfactored
        assert_eq!(display(&a, result), "x^2 + 1");
    }

    #[test]
    fn factor_cubic_all_rational() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
        let three = a.int(3);
        let x3 = a.pow(x, three);
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let six_a = a.int(6);
        let six_x2 = a.mul(&[six_a, x2]);
        let eleven = a.int(11);
        let eleven_x = a.mul(&[eleven, x]);
        let six = a.int(6);
        let neg_six_x2 = a.neg(six_x2);
        let neg_six = a.neg(six);
        let expr = a.add(&[x3, neg_six_x2, eleven_x, neg_six]);
        let result = factor(&mut a, expr, x);
        let s = display(&a, result);
        assert!(
            !s.contains("x^3") && !s.contains("x^2"),
            "should be fully factored: {s}"
        );
    }

    #[test]
    fn factor_non_polynomial_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.sin(x);
        let result = factor(&mut a, expr, x);
        assert_eq!(display(&a, result), "sin(x)");
    }

    #[test]
    fn factor_constant_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let result = factor(&mut a, five, x);
        assert_eq!(display(&a, result), "5");
    }
}
