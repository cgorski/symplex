//! Bridge between symbolic expressions ([`ExprId`]) and polynomials ([`Poly`]).
//!
//! This module provides:
//!
//! - [`expr_to_poly`]: convert a symbolic expression to a [`Poly`] in a
//!   given variable (returns `None` if the expression is not polynomial
//!   in that variable).
//! - [`poly_to_expr`]: convert a [`Poly`] back to a symbolic expression.
//! - [`cancel`]: cancel common polynomial factors in a rational
//!   expression (numerator / denominator).
//!
//! # Design
//!
//! Expression → Poly conversion walks the expression tree iteratively
//! (using [`walk::post_order_ids`]) and builds a `Poly` bottom-up.
//! At each node it checks whether the sub-expression is polynomial in
//! the given variable; if not, the conversion fails with `None`.
//!
//! Poly → expression conversion builds the expression from the
//! coefficient list using Horner's method for efficiency.
//!
//! `cancel` decomposes an expression into numerator and denominator
//! (by collecting negative-exponent factors), converts both to `Poly`,
//! divides out their GCD, and rebuilds the expression.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode};
use crate::poly::Poly;
use crate::walk;

// ═══════════════════════════════════════════════════════════════════════════
// Expression → Poly
// ═══════════════════════════════════════════════════════════════════════════

/// Try to convert an expression into a univariate polynomial in `var`.
///
/// Returns `None` if the expression contains terms that are not
/// polynomial in `var` (e.g., `sin(x)`, `x^(1/2)`, `x^y`).
///
/// Constant sub-expressions (not containing `var`) are treated as
/// degree-0 coefficients.
pub(crate) fn expr_to_poly(arena: &Arena, expr: ExprId, var: ExprId) -> Option<Poly> {
    // Fast path: if the expression doesn't contain var at all, it's
    // a constant polynomial.
    if !contains_id(arena, expr, var) {
        let coeff = expr_to_rational(arena, expr)?;
        return Some(Poly::constant(coeff));
    }

    // The variable itself.
    if expr == var {
        return Some(Poly::x());
    }

    // Walk the tree bottom-up.
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, Poly> = FxHashMap::default();

    for &id in &post_order {
        let poly = convert_node(arena, id, var, &cache)?;
        cache.insert(id, poly);
    }

    cache.remove(&expr)
}

/// Convert a single node to a Poly, given its children's Poly values.
fn convert_node(
    arena: &Arena,
    id: ExprId,
    var: ExprId,
    cache: &FxHashMap<ExprId, Poly>,
) -> Option<Poly> {
    // The variable itself.
    if id == var {
        return Some(Poly::x());
    }

    let node = arena.node(id);

    match node {
        // Numeric literal → constant polynomial.
        ExprNode::Num(nid) => {
            let r = arena.num(*nid).clone();
            Some(Poly::constant(r))
        }

        // Symbol that is NOT the variable → try as rational constant.
        ExprNode::Symbol(_) => {
            if !contains_id(arena, id, var) {
                // It's a different symbol — not polynomial unless we
                // treat it as a parameter.  For univariate polynomials,
                // other symbols make conversion fail.
                // Exception: if it doesn't depend on var, treat as a
                // "constant" but we can't represent it as Ratio<BigInt>.
                None
            } else {
                // This IS the variable (caught above), shouldn't reach here.
                Some(Poly::x())
            }
        }

        // Constants.
        ExprNode::Pi | ExprNode::E | ExprNode::ImaginaryUnit => None,
        ExprNode::Infinity | ExprNode::NegInfinity | ExprNode::ComplexInfinity | ExprNode::NaN => {
            None
        }

        // Add: sum of child polynomials.
        ExprNode::Add(children) => {
            let mut result = Poly::zero();
            for &child in children.iter() {
                let child_poly = cache.get(&child)?;
                result = &result + child_poly;
            }
            Some(result)
        }

        // Mul: product of child polynomials.
        ExprNode::Mul(children) => {
            let mut result = Poly::from_int(1);
            for &child in children.iter() {
                let child_poly = cache.get(&child)?;
                result = &result * child_poly;
            }
            Some(result)
        }

        // Pow: base^exp where exp must be a non-negative integer.
        ExprNode::Pow(base, exp) => {
            let base_poly = cache.get(base)?;

            // The exponent must be a non-negative integer constant
            // (not depending on var).
            if contains_id(arena, *exp, var) {
                return None; // x^x is not polynomial
            }

            let exp_val = expr_to_rational(arena, *exp)?;
            if !exp_val.is_integer() {
                return None; // x^(1/2) is not polynomial
            }

            let n: i64 = exp_val.to_integer().try_into().ok()?;
            if n < 0 {
                return None; // x^(-1) is not polynomial
            }

            let mut result = Poly::from_int(1);
            for _ in 0..n {
                result = &result * base_poly;
            }
            Some(result)
        }

        // Neg: negate the child polynomial.
        ExprNode::Neg(inner) => {
            let inner_poly = cache.get(inner)?;
            Some(-inner_poly)
        }

        // Functions containing var → not polynomial.
        ExprNode::Sin(_)
        | ExprNode::Cos(_)
        | ExprNode::Tan(_)
        | ExprNode::Exp(_)
        | ExprNode::Ln(_)
        | ExprNode::Sqrt(_)
        | ExprNode::Abs(_) => {
            // If the function argument doesn't contain var, the whole
            // thing is a constant — but we can't represent transcendentals
            // as Ratio<BigInt>, so we fail.
            None
        }

        ExprNode::Apply(_, _) | ExprNode::Derivative(_, _) | ExprNode::Integral(_, _) => None,
    }
}

/// Try to extract a rational number from an expression that is a
/// pure numeric constant (no symbols, no transcendentals).
fn expr_to_rational(arena: &Arena, id: ExprId) -> Option<Ratio<BigInt>> {
    match arena.node(id) {
        ExprNode::Num(nid) => Some(arena.num(*nid).clone()),
        // For Add/Mul/Pow/Neg of pure numbers, the canonical form
        // should already be a single Num node.  But just in case:
        _ => None,
    }
}

/// Check if `needle` appears anywhere in the expression tree rooted
/// at `haystack`.  Uses an explicit stack (no recursion).
fn contains_id(arena: &Arena, haystack: ExprId, needle: ExprId) -> bool {
    if haystack == needle {
        return true;
    }
    let mut visited: FxHashMap<ExprId, ()> = FxHashMap::default();
    let mut stack: Vec<ExprId> = vec![haystack];
    while let Some(id) = stack.pop() {
        if id == needle {
            return true;
        }
        if visited.contains_key(&id) {
            continue;
        }
        visited.insert(id, ());
        let children = arena.node(id).children();
        stack.extend_from_slice(&children);
    }
    false
}

// ═══════════════════════════════════════════════════════════════════════════
// Poly → Expression
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a polynomial back to a symbolic expression in `var`.
///
/// Uses Horner-style construction for efficiency:
/// `a₀ + a₁x + a₂x² = a₀ + x*(a₁ + x*a₂)`
///
/// However, for clarity and canonical form, we build
/// `a₀ + a₁*x + a₂*x^2 + …` and let the arena's canonicalization
/// handle the rest.
pub(crate) fn poly_to_expr(arena: &mut Arena, poly: &Poly, var: ExprId) -> ExprId {
    if poly.is_zero() {
        return arena.zero;
    }

    let coeffs = poly.coeffs();
    let mut terms: SmallVec<[ExprId; 8]> = SmallVec::new();

    for (i, c) in coeffs.iter().enumerate() {
        if c.is_zero() {
            continue;
        }

        let coeff_id = {
            let nid = arena.intern_num(c.clone());
            arena.intern(ExprNode::Num(nid))
        };

        if i == 0 {
            // Constant term.
            terms.push(coeff_id);
        } else {
            // c * var^i
            let power = if i == 1 {
                var
            } else {
                let exp = arena.int(i as i64);
                arena.pow(var, exp)
            };

            if c.is_one() {
                terms.push(power);
            } else {
                terms.push(arena.mul(&[coeff_id, power]));
            }
        }
    }

    match terms.len() {
        0 => arena.zero,
        1 => terms[0],
        _ => arena.add(&terms),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Numerator / Denominator decomposition
// ═══════════════════════════════════════════════════════════════════════════

/// Decompose an expression into (numerator, denominator) where both
/// are free of negative-exponent factors.
///
/// For example:
/// - `x` → `(x, 1)`
/// - `1/x` = `x^(-1)` → `(1, x)`
/// - `(x+1)/(x-1)` = `(x+1)*(x-1)^(-1)` → `(x+1, x-1)`
/// - `x^2 * y^(-3)` → `(x^2, y^3)`
pub(crate) fn as_numer_denom(arena: &mut Arena, expr: ExprId) -> (ExprId, ExprId) {
    let node = arena.node(expr).clone();

    match node {
        // Pow(base, exp) where exp is a negative integer → denominator factor.
        ExprNode::Pow(base, exp) => {
            if let Some(r) = arena.as_num(exp) {
                let r = r.clone();
                if r.is_negative() {
                    // base^(-n) → numerator=1, denominator=base^n
                    let pos_exp = {
                        let neg_r = -r;
                        let nid = arena.intern_num(neg_r);
                        arena.intern(ExprNode::Num(nid))
                    };
                    let denom = arena.pow(base, pos_exp);
                    return (arena.one, denom);
                }
            }
            (expr, arena.one)
        }

        // Mul(factors): separate into numer_factors and denom_factors.
        ExprNode::Mul(ref children) => {
            let children = children.clone();
            let mut numer_factors: SmallVec<[ExprId; 6]> = SmallVec::new();
            let mut denom_factors: SmallVec<[ExprId; 6]> = SmallVec::new();

            for &child in &children {
                let (n, d) = as_numer_denom(arena, child);
                if n != arena.one {
                    numer_factors.push(n);
                }
                if d != arena.one {
                    denom_factors.push(d);
                }
            }

            let numer = if numer_factors.is_empty() {
                arena.one
            } else if numer_factors.len() == 1 {
                numer_factors[0]
            } else {
                arena.mul(&numer_factors)
            };

            let denom = if denom_factors.is_empty() {
                arena.one
            } else if denom_factors.len() == 1 {
                denom_factors[0]
            } else {
                arena.mul(&denom_factors)
            };

            (numer, denom)
        }

        // Everything else: numerator is the expression, denominator is 1.
        _ => (expr, arena.one),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// cancel()
// ═══════════════════════════════════════════════════════════════════════════

/// Cancel common polynomial factors in a rational expression.
///
/// Decomposes the expression into numerator / denominator, converts
/// both to polynomials in `var`, divides out the GCD, and rebuilds
/// the expression.
///
/// Returns the original expression unchanged if:
/// - The expression is not a rational function in `var`
/// - The numerator and denominator have no common factor
///
/// # Example
///
/// `(x² - 1) / (x - 1)` → `x + 1`
pub(crate) fn cancel(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    let (numer, denom) = as_numer_denom(arena, expr);

    // If denominator is 1, nothing to cancel.
    if denom == arena.one {
        return expr;
    }

    // Try to convert both to polynomials.
    let numer_poly = match expr_to_poly(arena, numer, var) {
        Some(p) => p,
        None => return expr,
    };
    let denom_poly = match expr_to_poly(arena, denom, var) {
        Some(p) => p,
        None => return expr,
    };

    // Compute GCD.
    let gcd = Poly::gcd(&numer_poly, &denom_poly);

    // If GCD is constant (degree 0 or less), nothing to cancel.
    if gcd.is_constant() {
        return expr;
    }

    // Divide both by the GCD.
    let new_numer = numer_poly.div(&gcd);
    let new_denom = denom_poly.div(&gcd);

    // Convert back to expressions.
    let new_numer_expr = poly_to_expr(arena, &new_numer, var);

    if new_denom.is_zero() {
        // Shouldn't happen (GCD divides the denominator), but guard.
        return expr;
    }

    if new_denom.degree() == Some(0) && new_denom.coeff(0).is_one() {
        // Denominator cancelled to 1.
        return new_numer_expr;
    }

    // Rebuild as numer * denom^(-1).
    let new_denom_expr = poly_to_expr(arena, &new_denom, var);
    let neg_one = arena.neg_one;
    let denom_inv = arena.pow(new_denom_expr, neg_one);
    arena.mul(&[new_numer_expr, denom_inv])
}

/// Group an expression by powers of `var`.
///
/// Converts the expression to a univariate polynomial in `var`,
/// then rebuilds it term-by-term. This naturally groups coefficients
/// by power of `var`.
///
/// Returns the expression unchanged if it is not polynomial in `var`.
pub(crate) fn collect(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    let poly = match expr_to_poly(arena, expr, var) {
        Some(p) => p,
        None => return expr,
    };
    poly_to_expr(arena, &poly, var)
}

/// Combine fractions over a common denominator.
///
/// For an Add node, decomposes each term into numerator/denominator
/// via `as_numer_denom`, computes a common denominator (product of
/// all unique denominators, simplified via polynomial GCD), scales
/// each numerator, and rebuilds as `sum_of_numerators / common_denom`.
///
/// Returns the expression unchanged if it is not an Add, or if all
/// terms already have denominator 1.
pub(crate) fn together(arena: &mut Arena, expr: ExprId) -> ExprId {
    let node = arena.node(expr).clone();

    let children = match node {
        ExprNode::Add(ref ch) => ch.clone(),
        _ => return expr,
    };

    // Decompose each term into (numerator, denominator).
    let mut parts: Vec<(ExprId, ExprId)> = Vec::with_capacity(children.len());
    let mut all_denom_one = true;
    for &child in &children {
        let (n, d) = as_numer_denom(arena, child);
        if d != arena.one {
            all_denom_one = false;
        }
        parts.push((n, d));
    }

    // If every term has denominator 1, nothing to do.
    if all_denom_one {
        return expr;
    }

    // Compute common denominator as the product of all distinct denominators.
    // We deduplicate by ExprId to avoid multiplying the same denom twice.
    let mut unique_denoms: Vec<ExprId> = Vec::new();
    for &(_, d) in &parts {
        if d != arena.one && !unique_denoms.contains(&d) {
            unique_denoms.push(d);
        }
    }

    let common_denom = if unique_denoms.len() == 1 {
        unique_denoms[0]
    } else {
        arena.mul(&unique_denoms)
    };

    // Scale each numerator: numer_i * (common_denom / denom_i).
    let mut scaled_numers: SmallVec<[ExprId; 6]> = SmallVec::new();
    for &(n, d) in &parts {
        if d == arena.one {
            // numer * common_denom
            let scaled = arena.mul(&[n, common_denom]);
            scaled_numers.push(scaled);
        } else {
            // numer * (common_denom / denom) = numer * product_of_other_denoms
            let mut other_denoms: Vec<ExprId> = Vec::new();
            for &ud in &unique_denoms {
                if ud != d {
                    other_denoms.push(ud);
                }
            }
            if other_denoms.is_empty() {
                // denom IS the common denom, so scale factor is 1
                scaled_numers.push(n);
            } else {
                let scale = if other_denoms.len() == 1 {
                    other_denoms[0]
                } else {
                    arena.mul(&other_denoms)
                };
                let scaled = arena.mul(&[n, scale]);
                scaled_numers.push(scaled);
            }
        }
    }

    // Sum the scaled numerators.
    let numer_sum = arena.add(&scaled_numers);

    // Build result: numer_sum / common_denom.
    arena.div(numer_sum, common_denom)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

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

    // ── expr_to_poly ────────────────────────────────────────────────

    #[test]
    fn expr_to_poly_constant() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let p = expr_to_poly(&a, five, x).unwrap();
        assert_eq!(format!("{p}"), "5");
    }

    #[test]
    fn expr_to_poly_variable() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let p = expr_to_poly(&a, x, x).unwrap();
        assert_eq!(format!("{p}"), "x");
    }

    #[test]
    fn expr_to_poly_linear() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let three = a.int(3);
        // 2*x + 3
        let two_x = a.mul(&[two, x]);
        let expr = a.add(&[two_x, three]);
        let p = expr_to_poly(&a, expr, x).unwrap();
        assert_eq!(p.degree(), Some(1));
        assert_eq!(format!("{p}"), "2*x + 3");
    }

    #[test]
    fn expr_to_poly_quadratic() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        // x^2 + 2*x + 1
        let x_sq = a.pow(x, two);
        let two_x = a.mul(&[two, x]);
        let one = a.one;
        let expr = a.add(&[x_sq, two_x, one]);
        let p = expr_to_poly(&a, expr, x).unwrap();
        assert_eq!(p.degree(), Some(2));
        // Evaluate at x=3: should be 16.
        let val = p.eval(&Ratio::from_integer(BigInt::from(3)));
        assert_eq!(val, Ratio::from_integer(BigInt::from(16)));
    }

    #[test]
    fn expr_to_poly_fails_for_sin() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.sin(x);
        assert!(expr_to_poly(&a, expr, x).is_none());
    }

    #[test]
    fn expr_to_poly_fails_for_fractional_power() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let half = a.rational(1, 2);
        let expr = a.pow(x, half);
        assert!(expr_to_poly(&a, expr, x).is_none());
    }

    #[test]
    fn expr_to_poly_fails_for_negative_power() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let neg_one = a.int(-1);
        let expr = a.pow(x, neg_one);
        assert!(expr_to_poly(&a, expr, x).is_none());
    }

    #[test]
    fn expr_to_poly_product() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        // (x + 1) * (x - 1) — expanded by canon_mul's Number*Add distribution,
        // but the overall product might stay as Mul if factors aren't numeric.
        // Let's build x^2 - 1 directly.
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let expr = a.sub(x_sq, one);
        let p = expr_to_poly(&a, expr, x).unwrap();
        assert_eq!(p.degree(), Some(2));
        // Check roots.
        let val_1 = p.eval(&Ratio::from_integer(BigInt::from(1)));
        let val_neg1 = p.eval(&Ratio::from_integer(BigInt::from(-1)));
        assert!(val_1.is_zero(), "x²-1 at x=1 should be 0");
        assert!(val_neg1.is_zero(), "x²-1 at x=-1 should be 0");
    }

    // ── poly_to_expr ────────────────────────────────────────────────

    #[test]
    fn poly_to_expr_constant() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let p = Poly::from_int(7);
        let expr = poly_to_expr(&mut a, &p, x);
        assert_eq!(display(&a, expr), "7");
    }

    #[test]
    fn poly_to_expr_linear() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let p = Poly::from_coeffs(vec![
            Ratio::from_integer(BigInt::from(3)),
            Ratio::from_integer(BigInt::from(2)),
        ]);
        let expr = poly_to_expr(&mut a, &p, x);
        assert_eq!(display(&a, expr), "3 + 2*x");
    }

    #[test]
    fn poly_to_expr_zero() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let p = Poly::zero();
        let expr = poly_to_expr(&mut a, &p, x);
        assert_eq!(expr, a.zero);
    }

    #[test]
    fn poly_to_expr_quadratic() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let p = Poly::from_coeffs(vec![
            Ratio::from_integer(BigInt::from(1)),
            Ratio::from_integer(BigInt::from(2)),
            Ratio::from_integer(BigInt::from(1)),
        ]);
        let expr = poly_to_expr(&mut a, &p, x);
        assert_eq!(display(&a, expr), "1 + x^2 + 2*x");
    }

    #[test]
    fn roundtrip_expr_poly_expr() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let three = a.int(3);
        // x^2 + 3*x + 2
        let x_sq = a.pow(x, two);
        let three_x = a.mul(&[three, x]);
        let orig = a.add(&[x_sq, three_x, two]);

        let poly = expr_to_poly(&a, orig, x).unwrap();
        let rebuilt = poly_to_expr(&mut a, &poly, x);

        // Should produce the same canonical expression.
        assert_eq!(orig, rebuilt, "roundtrip should preserve expression");
    }

    // ── as_numer_denom ──────────────────────────────────────────────

    #[test]
    fn numer_denom_plain_symbol() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let (n, d) = as_numer_denom(&mut a, x);
        assert_eq!(n, x);
        assert_eq!(d, a.one);
    }

    #[test]
    fn numer_denom_inverse() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let neg_one = a.int(-1);
        let expr = a.pow(x, neg_one); // x^(-1)
        let (n, d) = as_numer_denom(&mut a, expr);
        assert_eq!(n, a.one, "numerator of 1/x should be 1");
        assert_eq!(d, x, "denominator of 1/x should be x");
    }

    #[test]
    fn numer_denom_fraction() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        // x / y = x * y^(-1)
        let expr = a.div(x, y);
        let (n, d) = as_numer_denom(&mut a, expr);
        assert_eq!(display(&a, n), "x");
        assert_eq!(display(&a, d), "y");
    }

    // ── cancel ──────────────────────────────────────────────────────

    #[test]
    fn cancel_x_squared_minus_1_over_x_minus_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let one = a.one;
        // (x^2 - 1) / (x - 1)
        let x_sq = a.pow(x, two);
        let numer = a.sub(x_sq, one);
        let denom = a.sub(x, one);
        let expr = a.div(numer, denom);

        let result = cancel(&mut a, expr, x);
        let s = display(&a, result);
        // Should simplify to x + 1.
        assert_eq!(s, "1 + x", "cancel should give x + 1, got: {s}");
    }

    #[test]
    fn cancel_no_common_factor() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let one = a.one;
        // (x^2 + 1) / (x + 1) — no common factor.
        let x_sq = a.pow(x, two);
        let numer = a.add(&[x_sq, one]);
        let denom = a.add(&[x, one]);
        let expr = a.div(numer, denom);

        let result = cancel(&mut a, expr, x);
        // Should be unchanged.
        assert_eq!(result, expr, "no common factor → unchanged");
    }

    #[test]
    fn cancel_already_simple() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x + 1 — no denominator.
        let expr = a.add(&[x, a.one]);
        let result = cancel(&mut a, expr, x);
        assert_eq!(result, expr, "no denominator → unchanged");
    }

    #[test]
    fn cancel_quadratic_common_factor() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let three = a.int(3);
        let six = a.int(6);

        // numer = x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
        // denom = x^2 - 3x + 2 = (x-1)(x-2)
        // cancel → x - 3
        let x2 = a.pow(x, two);
        let x3 = a.pow(x, three);
        let six_x2 = a.mul(&[six, x2]);
        let eleven = a.int(11);
        let eleven_x = a.mul(&[eleven, x]);
        let neg_six_x2 = a.neg(six_x2);
        let neg_six = a.neg(six);
        let numer = a.add(&[x3, neg_six_x2, eleven_x, neg_six]);

        let three_x = a.mul(&[three, x]);
        let neg_three_x = a.neg(three_x);
        let denom = a.add(&[x2, neg_three_x, two]);

        let expr = a.div(numer, denom);
        let result = cancel(&mut a, expr, x);
        let s = display(&a, result);

        // The result should be x - 3 = -3 + x.
        assert!(
            s.contains('x') && s.contains('3'),
            "cancel should give x - 3, got: {s}"
        );
        // Verify numerically: at x=10, (10-1)(10-2)(10-3)/((10-1)(10-2)) = 7.
        let ten = a.int(10);
        let val = crate::subs::subs(&mut a, result, x, ten);
        assert_eq!(display(&a, val), "7", "at x=10, x-3 = 7");
    }

    #[test]
    fn cancel_with_coefficients() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        // (2*x^2 - 2) / (2*x - 2) = (2*(x^2 - 1)) / (2*(x - 1))
        //   = (2*(x-1)*(x+1)) / (2*(x-1)) = x + 1
        let x_sq = a.pow(x, two);
        let two_x_sq = a.mul(&[two, x_sq]);
        let numer = a.sub(two_x_sq, two);
        let two_x = a.mul(&[two, x]);
        let denom = a.sub(two_x, two);
        let expr = a.div(numer, denom);

        let result = cancel(&mut a, expr, x);
        let s = display(&a, result);
        assert_eq!(s, "1 + x", "cancel should give x + 1, got: {s}");
    }

    #[test]
    fn cancel_non_polynomial_returns_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // sin(x) / x — not polynomial in numerator.
        let sin_x = a.sin(x);
        let expr = a.div(sin_x, x);
        let result = cancel(&mut a, expr, x);
        assert_eq!(result, expr, "non-polynomial should be unchanged");
    }
}
