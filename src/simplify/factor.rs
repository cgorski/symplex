//! Polynomial factoring over ℤ for symbolic expressions.
//!
//! This module implements [`factor`], [`factor_auto`] and [`factor_list`],
//! which factor a polynomial expression over ℤ into a product of irreducible
//! factors with their multiplicities.
//!
//! # Algorithm
//!
//! 1. Convert the expression to a univariate [`Poly`] in the given variable.
//!    If that succeeds, call [`Poly::factor_over_z`] (content extraction,
//!    Yun's square-free decomposition, Berlekamp–Zassenhaus — see
//!    [`crate::poly::factor_zassenhaus`]).
//! 2. Otherwise, if the expression is a polynomial in a handful of symbols,
//!    convert it to a [`MultiPoly`](crate::poly::multipoly::MultiPoly) and
//!    factor by Kronecker substitution
//!    ([`factor_multivariate`]).
//! 3. Convert the factors back to expressions and build the product.
//!
//! Every factorization is verified by multiplying back before it is
//! returned; expressions that are not polynomial are returned unchanged.

use num_traits::One;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::base::walk;
use crate::poly::Poly;
use crate::poly::factor_zassenhaus::factor_multivariate;
use crate::poly::polybridge;

/// Maximum number of distinct symbols handled by the multivariate path.
const MAX_MULTIVARIATE_SYMBOLS: usize = 4;

/// Rational content plus `(factor expression, multiplicity)` pairs.
type FactorParts = (Q, Vec<(ExprId, u32)>);

/// Factor a polynomial expression in `var` into a product of irreducible
/// factors over ℤ.
///
/// Returns the expression unchanged if it is not polynomial in `var`, if it
/// is constant in `var`, or if it is already irreducible with unit content.
pub(crate) fn factor(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    match factor_parts(arena, expr, Some(var)) {
        Some((content, factors)) if is_nontrivial(&content, &factors) => {
            build_factored_expr(arena, &content, &factors)
        }
        _ => expr,
    }
}

/// Factor a polynomial expression with the variable(s) inferred from its
/// free symbols.
///
/// With exactly one free symbol this is [`factor`] in that symbol; with
/// two to four symbols the multivariate path is used.  Expressions with
/// no free symbols, too many symbols, or non-polynomial structure are
/// returned unchanged.
pub(crate) fn factor_auto(arena: &mut Arena, expr: ExprId) -> ExprId {
    match factor_parts(arena, expr, None) {
        Some((content, factors)) if is_nontrivial(&content, &factors) => {
            build_factored_expr(arena, &content, &factors)
        }
        _ => expr,
    }
}

/// Factor and return the pieces: `(content, [(factor, multiplicity), …])`
/// such that `expr = content · ∏ factorᵢ^multᵢ`.
///
/// If `var` is `None` the variables are inferred as in [`factor_auto`].
/// For inputs that cannot be factored (non-polynomial, constant, …) the
/// result is `(1, [(expr, 1)])`.
pub(crate) fn factor_list(
    arena: &mut Arena,
    expr: ExprId,
    var: Option<ExprId>,
) -> (ExprId, Vec<(ExprId, u32)>) {
    match factor_parts(arena, expr, var) {
        Some((content, factors)) => {
            let content_id = {
                let nid = arena.intern_num(content);
                arena.intern(ExprNode::Num(nid))
            };
            (content_id, factors)
        }
        None => (arena.one, vec![(expr, 1)]),
    }
}

/// Shared driver: returns the rational content and the irreducible factors
/// (as expressions) or `None` if the expression is not a non-constant
/// polynomial in the chosen variable(s).
fn factor_parts(arena: &mut Arena, expr: ExprId, var: Option<ExprId>) -> Option<FactorParts> {
    let mut syms = walk::free_symbols(arena, expr);
    syms.sort_by_key(|id| id.0);
    syms.dedup();

    match var {
        Some(v) => {
            if let Some(poly) = polybridge::expr_to_poly(arena, expr, v) {
                return factor_univariate(arena, &poly, v);
            }
            if !syms.contains(&v) {
                return None;
            }
            factor_multi(arena, expr, &syms)
        }
        None => match syms.len() {
            0 => None,
            1 => {
                let v = syms[0];
                let poly = polybridge::expr_to_poly(arena, expr, v)?;
                factor_univariate(arena, &poly, v)
            }
            _ => factor_multi(arena, expr, &syms),
        },
    }
}

fn factor_univariate(arena: &mut Arena, poly: &Poly, var: ExprId) -> Option<FactorParts> {
    if poly.is_zero() || poly.is_constant() {
        return None;
    }
    let (content, factors) = poly.factor_over_z();
    let factor_ids = factors
        .iter()
        .map(|(f, m)| (polybridge::poly_to_expr(arena, f, var), *m))
        .collect();
    Some((content, factor_ids))
}

fn factor_multi(arena: &mut Arena, expr: ExprId, syms: &[ExprId]) -> Option<FactorParts> {
    if syms.len() < 2 || syms.len() > MAX_MULTIVARIATE_SYMBOLS {
        return None;
    }
    let mp = polybridge::expr_to_multipoly(arena, expr, syms)?;
    if mp.is_zero() || mp.total_degree().unwrap_or(0) == 0 {
        return None;
    }
    let (content, factors) = factor_multivariate(&mp)?;
    let factor_ids = factors
        .iter()
        .map(|(f, m)| (polybridge::multipoly_to_expr(arena, f, syms), *m))
        .collect();
    Some((content, factor_ids))
}

/// A factorization is worth reporting if it has more than one factor, a
/// repeated factor, or a non-unit content.
fn is_nontrivial(content: &Q, factors: &[(ExprId, u32)]) -> bool {
    factors.len() > 1 || factors.iter().any(|(_, m)| *m > 1) || !content.is_one()
}

/// Build `content · ∏ factor^mult` as an expression.
fn build_factored_expr(arena: &mut Arena, content: &Q, factors: &[(ExprId, u32)]) -> ExprId {
    let mut parts: Vec<ExprId> = Vec::with_capacity(factors.len() + 1);

    if !content.is_one() {
        let nid = arena.intern_num(content.clone());
        parts.push(arena.intern(ExprNode::Num(nid)));
    }

    for &(fexpr, mult) in factors {
        if mult == 1 {
            parts.push(fexpr);
        } else {
            let exp = arena.int(mult as i64);
            parts.push(arena.pow(fexpr, exp));
        }
    }

    match parts.len() {
        0 => arena.one,
        1 => parts[0],
        _ => arena.mul(&parts),
    }
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

    fn powi(a: &mut Arena, base: ExprId, n: i64) -> ExprId {
        let e = a.int(n);
        a.pow(base, e)
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

    #[test]
    fn factor_x12_minus_1_fully() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let x12 = powi(&mut a, x, 12);
        let one = a.one;
        let expr = a.sub(x12, one);
        let (content, factors) = factor_list(&mut a, expr, Some(x));
        assert_eq!(display(&a, content), "1");
        assert_eq!(factors.len(), 6, "τ(12) = 6 cyclotomic factors");
        let factored = factor(&mut a, expr, x);
        let s = display(&a, factored);
        assert!(s.contains("x^4 - x^2 + 1"), "should contain Φ12: {s}");
    }

    #[test]
    fn factor_multivariate_difference_of_squares() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let x2 = powi(&mut a, x, 2);
        let y2 = powi(&mut a, y, 2);
        let expr = a.sub(x2, y2);
        let result = factor(&mut a, expr, x);
        let s = display(&a, result);
        assert!(
            s.contains("x - y") && s.contains("x + y"),
            "x^2 - y^2 should factor: {s}"
        );
    }

    #[test]
    fn factor_auto_single_symbol() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let x2 = powi(&mut a, x, 2);
        let one = a.one;
        let expr = a.sub(x2, one);
        let result = factor_auto(&mut a, expr);
        assert!(!display(&a, result).contains("x^2"));
    }

    #[test]
    fn factor_auto_multivariate_with_content() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        // x^2 y - y = y (x - 1)(x + 1)
        let x2 = powi(&mut a, x, 2);
        let x2y = a.mul(&[x2, y]);
        let expr = a.sub(x2y, y);
        let (content, factors) = factor_list(&mut a, expr, None);
        assert_eq!(display(&a, content), "1");
        assert_eq!(factors.len(), 3);
        let names: Vec<String> = factors.iter().map(|(f, _)| display(&a, *f)).collect();
        assert!(names.iter().any(|n| n == "y"), "{names:?}");
    }

    #[test]
    fn factor_list_non_polynomial() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.sin(x);
        let (content, factors) = factor_list(&mut a, expr, Some(x));
        assert_eq!(content, a.one);
        assert_eq!(factors, vec![(expr, 1)]);
    }
}
