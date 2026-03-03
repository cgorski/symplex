//! Taylor series expansion.
//!
//! This module implements [`series`], which computes the Taylor/Maclaurin
//! series of an expression around a point.
//!
//! # Algorithm
//!
//! The Taylor series of `f(x)` around `x = a` to order `n` is:
//!
//! ```text
//! f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + ... + f^(n-1)(a)(x-a)^(n-1)/(n-1)!
//! ```
//!
//! This is computed by repeated differentiation (via `diff`) and
//! substitution (via `subs`), both of which are already implemented.
//!
//! # Design
//!
//! The result is returned as a plain expression (polynomial in `(x - a)`).
//! No `O(x^n)` term is appended — the truncation order is implicit in
//! the `order` parameter.

use num_bigint::BigInt;
use num_rational::Ratio;

use crate::arena::Arena;
use crate::node::ExprId;

/// Compute the Taylor series of `expr` in `var` around `point` to the
/// given `order` (number of terms).
///
/// Returns the truncated polynomial:
/// `f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + ...`
///
/// If `point` is zero, this is a Maclaurin series and the result is a
/// polynomial in `var`.
pub(crate) fn series(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    order: u32,
) -> ExprId {
    if order == 0 {
        return arena.zero;
    }

    let mut terms: Vec<ExprId> = Vec::with_capacity(order as usize);
    let mut current_deriv = expr;
    let mut factorial: Ratio<BigInt> = Ratio::from_integer(BigInt::from(1));

    for k in 0..order {
        // Evaluate the k-th derivative at the point.
        let value_at_point = {
            let subst = crate::subs::subs(arena, current_deriv, var, point);
            // Evaluate special values (e.g., cos(0) → 1, sin(0) → 0).
            crate::eval::eval(arena, subst)
        };

        // Check for poles: if the value at the point is infinite or NaN,
        // we cannot form a Taylor series. Return the original expression.
        if value_at_point == arena.infinity
            || value_at_point == arena.neg_infinity
            || value_at_point == arena.nan
            || value_at_point == arena.complex_infinity
        {
            return expr;
        }

        // Compute the term: value_at_point * (x - a)^k / k!
        if !arena.is_zero_structural(value_at_point) {
            let term = if k == 0 {
                value_at_point
            } else {
                // Build (x - a)^k
                let x_minus_a = if arena.is_zero_structural(point) {
                    var
                } else {
                    arena.sub(var, point)
                };

                let power = if k == 1 {
                    x_minus_a
                } else {
                    let k_id = arena.int(k as i64);
                    arena.pow(x_minus_a, k_id)
                };

                // Build coefficient: 1/k!
                let coeff_nid =
                    arena.intern_num(Ratio::from_integer(BigInt::from(1)) / factorial.clone());
                let coeff = arena.intern(crate::node::ExprNode::Num(coeff_nid));

                // term = value_at_point * coeff * power
                arena.mul(&[value_at_point, coeff, power])
            };
            terms.push(term);
        }

        // Compute next derivative for the next iteration.
        if k + 1 < order {
            current_deriv = crate::diff::diff(arena, current_deriv, var);
            factorial *= Ratio::from_integer(BigInt::from(k as i64 + 1));
        }
    }

    if terms.is_empty() {
        arena.zero
    } else if terms.len() == 1 {
        terms[0]
    } else {
        arena.add(&terms)
    }
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

    #[test]
    fn series_constant() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let zero = a.zero;
        let result = series(&mut a, five, x, zero, 3);
        assert_eq!(display(&a, result), "5");
    }

    #[test]
    fn series_x_around_zero() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        // series(x, x, 0, 3) = 0 + 1*x + 0 = x
        let result = series(&mut a, x, x, zero, 3);
        assert_eq!(display(&a, result), "x");
    }

    #[test]
    fn series_x_squared_around_zero() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let zero = a.zero;
        // series(x^2, x, 0, 3) = 0 + 0*x + 2/2! * x^2 = x^2
        let result = series(&mut a, x2, x, zero, 3);
        assert_eq!(display(&a, result), "x^2");
    }

    #[test]
    fn series_exp_x_around_zero_order_4() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.exp_fn(x);
        let zero = a.zero;
        // series(exp(x), x, 0, 4) = 1 + x + x^2/2 + x^3/6
        let result = series(&mut a, expr, x, zero, 4);
        let expanded = crate::expand::expand(&mut a, result);
        let evaled = crate::eval::eval(&mut a, expanded);
        let s = display(&a, evaled);
        assert!(s.contains("1"), "should have constant term 1: {s}");
        assert!(s.contains("x"), "should have x term: {s}");
        assert!(s.contains("x^2"), "should have x^2 term: {s}");
        assert!(s.contains("x^3"), "should have x^3 term: {s}");
    }

    #[test]
    fn series_sin_x_around_zero_order_4() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.sin(x);
        let zero = a.zero;
        // series(sin(x), x, 0, 4) = x - x^3/6
        let result = series(&mut a, expr, x, zero, 4);
        let expanded = crate::expand::expand(&mut a, result);
        let evaled = crate::eval::eval(&mut a, expanded);
        let s = display(&a, evaled);
        assert!(s.contains("x"), "should have x term: {s}");
        assert!(s.contains("x^3"), "should have x^3 term: {s}");
    }

    #[test]
    fn series_order_zero_is_zero() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        let result = series(&mut a, x, x, zero, 0);
        assert_eq!(result, a.zero);
    }

    #[test]
    fn series_polynomial_is_exact() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let x3 = a.pow(x, three);
        let zero = a.zero;
        // series(x^3, x, 0, 5) should give exactly x^3
        // (higher-order terms are zero)
        let result = series(&mut a, x3, x, zero, 5);
        let s = display(&a, result);
        assert!(s.contains("x^3"), "should recover x^3: {s}");
    }
}
