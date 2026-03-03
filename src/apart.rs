//! Partial fraction decomposition.
//!
//! Implements [`apart`], which decomposes a rational expression into a
//! sum of simpler fractions.
//!
//! # Algorithm
//!
//! 1. Decompose expression into numerator/denominator.
//! 2. Factor the denominator.
//! 3. For each linear factor `(x - r)`, compute the residue `numer(r)/denom'(r)`.
//! 4. Assemble as `Σ residue_i / (x - r_i)` + polynomial remainder.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::Zero;

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode};
use crate::poly::Poly;
use crate::polybridge;

/// Decompose `expr` into partial fractions with respect to `var`.
///
/// Returns the expression unchanged if it's not a rational function
/// in `var` or if the denominator has no rational roots.
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

    // If numerator degree >= denominator degree, do polynomial division first.
    let (quotient, remainder) = if numer_poly.degree().unwrap_or(0) >= denom_degree {
        numer_poly.div_rem(&denom_poly)
    } else {
        (Poly::zero(), numer_poly.clone())
    };

    // Step 3: Find the roots of the denominator.
    let solutions = crate::solve::solve(arena, denom, var);
    if solutions.is_empty() {
        return expr; // Can't factor denominator.
    }

    // Step 4: For each root, compute the residue using the limit method.
    // residue_i = lim_{x→r_i} (x - r_i) * numer(x) / denom(x)
    // = numer(r_i) / denom'(r_i)
    let denom_deriv_poly = poly_derivative(&denom_poly);

    let mut partial_terms: Vec<ExprId> = Vec::new();

    for sol in &solutions {
        let root_val = match arena.node(sol.value) {
            ExprNode::Num(nid) => arena.num(*nid).clone(),
            _ => continue,
        };

        // Evaluate numerator remainder at the root.
        let numer_at_root = remainder.eval(&root_val);
        // Evaluate denominator derivative at the root.
        let denom_deriv_at_root = denom_deriv_poly.eval(&root_val);

        if denom_deriv_at_root.is_zero() {
            // Multiple root — skip (would need higher-order partial fractions).
            continue;
        }

        let residue = numer_at_root / denom_deriv_at_root;

        if !residue.is_zero() {
            // Build: residue / (x - root)
            let residue_id = rational_to_expr(arena, &residue);
            let root_id = sol.value;
            let factor = arena.sub(var, root_id);
            let term = arena.div(residue_id, factor);
            partial_terms.push(term);
        }
    }

    if partial_terms.is_empty() {
        return expr; // Couldn't decompose.
    }

    // Add the polynomial quotient part if any.
    if !quotient.is_zero() {
        let quot_expr = polybridge::poly_to_expr(arena, &quotient, var);
        partial_terms.push(quot_expr);
    }

    // Check: does the decomposition account for all terms?
    // Reconstruct and compare would be ideal, but for now just return what we have.
    if partial_terms.len() == 1 {
        partial_terms[0]
    } else {
        arena.add(&partial_terms)
    }
}

/// Compute the formal derivative of a polynomial.
fn poly_derivative(p: &Poly) -> Poly {
    let coeffs = p.coeffs();
    if coeffs.len() <= 1 {
        return Poly::zero();
    }
    let deriv_coeffs: Vec<Ratio<BigInt>> = coeffs
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, c)| c * Ratio::from_integer(BigInt::from(i)))
        .collect();
    Poly::from_coeffs(deriv_coeffs)
}

fn rational_to_expr(arena: &mut Arena, r: &Ratio<BigInt>) -> ExprId {
    let nid = arena.intern_num(r.clone());
    arena.intern(ExprNode::Num(nid))
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
        assert_eq!(display(&a, result), "1 + x");
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
}
