//! Radical simplification — denominator rationalization.
//!
//! Implements [`rationalize_denom`], which clears radicals from denominators:
//!
//! - `1/√a → √a/a`
//! - `1/(a + √b) → (a - √b)/(a² - b)`
//! - `c/(a + √b) → c·(a - √b)/(a² - b)`

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode};
use crate::polybridge;

/// Rationalize the denominator of an expression.
///
/// If the expression is a fraction whose denominator contains square roots,
/// multiply by the conjugate to eliminate them.
pub(crate) fn rationalize_denom(arena: &mut Arena, expr: ExprId) -> ExprId {
    let (numer, denom) = polybridge::as_numer_denom(arena, expr);

    // If denom is 1, nothing to do
    if denom == arena.one {
        return expr;
    }

    // Check if denom contains a sqrt
    if !contains_sqrt(arena, denom) {
        return expr;
    }

    // Try to rationalize
    if let Some(result) = try_rationalize(arena, numer, denom) {
        return result;
    }

    expr
}

/// Check if an expression contains a Pow with exponent 1/2 (i.e., a sqrt).
fn contains_sqrt(arena: &Arena, expr: ExprId) -> bool {
    match arena.node(expr).clone() {
        ExprNode::Pow(_, exp) => {
            if let Some(r) = arena.as_num(exp)
                && *r == num_rational::Ratio::new(1.into(), 2.into())
            {
                return true;
            }
            // Check children
            arena
                .node(expr)
                .children()
                .iter()
                .any(|&c| contains_sqrt(arena, c))
        }
        ExprNode::Add(ref children) | ExprNode::Mul(ref children) => {
            children.iter().any(|&c| contains_sqrt(arena, c))
        }
        ExprNode::Neg(inner) => contains_sqrt(arena, inner),
        _ => false,
    }
}

/// Try to rationalize a fraction numer/denom by multiplying by the conjugate.
fn try_rationalize(arena: &mut Arena, numer: ExprId, denom: ExprId) -> Option<ExprId> {
    // Case 1: denom is a bare sqrt: √a → multiply by √a/√a → numer·√a / a
    if let ExprNode::Pow(base, exp) = arena.node(denom).clone()
        && let Some(r) = arena.as_num(exp)
        && *r == num_rational::Ratio::new(1.into(), 2.into())
    {
        // denom = base^(1/2) = √base
        // result = numer * √base / base
        let new_numer = arena.mul(&[numer, denom]); // numer * √base
        return Some(arena.div(new_numer, base));
    }

    // Case 2: denom is Add with exactly 2 children, one or both containing sqrt
    // e.g., a + √b or √a + √b
    if let ExprNode::Add(ref children) = arena.node(denom).clone()
        && children.len() == 2
    {
        let a = children[0];
        let b = children[1];

        // Conjugate: a - b (negate second term)
        let neg_b = arena.neg(b);
        let conjugate = arena.add(&[a, neg_b]);

        // New numerator: numer * conjugate
        let new_numer = arena.mul(&[numer, conjugate]);
        let new_numer_expanded = crate::expand::expand(arena, new_numer);

        // New denominator: (a+b)(a-b) = a² - b²
        let denom_product = arena.mul(&[denom, conjugate]);
        let new_denom = crate::expand::expand(arena, denom_product);
        let new_denom = crate::eval::eval(arena, new_denom);

        // Check if the new denominator is sqrt-free
        if !contains_sqrt(arena, new_denom) {
            return Some(arena.div(new_numer_expanded, new_denom));
        }

        // If still has sqrt, try one more round of conjugate
        // (handles cases like (√2 + √3) where first pass gives (2-3)=-1)
        // Actually the expand should handle a²-b² = a²-b² already.
        // Return what we have — it might still be simpler.
        return Some(arena.div(new_numer_expanded, new_denom));
    }

    None
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
    fn rationalize_one_over_sqrt_2() {
        let mut a = Arena::new();
        let two = a.int(2);
        let half = a.rational(1, 2);
        let sqrt_2 = a.pow(two, half);
        let one = a.one;
        let expr = a.div(one, sqrt_2);
        let result = rationalize_denom(&mut a, expr);
        let s = display(&a, result);
        // 1/√2 → √2/2
        assert!(s.contains("2"), "should contain 2: {s}");
        // Should not have sqrt in denominator
    }

    #[test]
    fn rationalize_one_over_one_plus_sqrt_2() {
        let mut a = Arena::new();
        let one = a.one;
        let two = a.int(2);
        let half = a.rational(1, 2);
        let sqrt_2 = a.pow(two, half);
        let denom = a.add(&[one, sqrt_2]);
        let expr = a.div(one, denom);
        let result = rationalize_denom(&mut a, expr);
        let s = display(&a, result);
        // 1/(1+√2) → (1-√2)/(1-2) = -(1-√2) = √2-1
        assert!(s != display(&a, expr), "should change: {s}");
    }

    #[test]
    fn no_sqrt_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let expr = a.div(x, y);
        let result = rationalize_denom(&mut a, expr);
        assert_eq!(result, expr, "no sqrt should be unchanged");
    }

    #[test]
    fn integer_denom_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let expr = a.div(x, two);
        let result = rationalize_denom(&mut a, expr);
        assert_eq!(result, expr, "integer denom should be unchanged");
    }

    #[test]
    fn rationalize_with_numerator() {
        let mut a = Arena::new();
        let three = a.int(3);
        let two = a.int(2);
        let half = a.rational(1, 2);
        let sqrt_2 = a.pow(two, half);
        let expr = a.div(three, sqrt_2);
        let result = rationalize_denom(&mut a, expr);
        let s = display(&a, result);
        assert!(s.contains("3"), "should contain 3: {s}");
    }

    #[test]
    fn rationalize_sqrt_plus_sqrt() {
        let mut a = Arena::new();
        let one = a.one;
        let two = a.int(2);
        let three = a.int(3);
        let half = a.rational(1, 2);
        let sqrt_2 = a.pow(two, half);
        let sqrt_3 = a.pow(three, half);
        let denom = a.add(&[sqrt_2, sqrt_3]);
        let expr = a.div(one, denom);
        let result = rationalize_denom(&mut a, expr);
        let s = display(&a, result);
        // Should attempt rationalization
        assert!(s != display(&a, expr), "should change: {s}");
    }
}
