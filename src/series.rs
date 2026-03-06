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
use crate::node::{ExprId, ExprNode};

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
) -> Result<ExprId, crate::errors::SymplexError> {
    if order == 0 {
        return Ok(arena.zero);
    }

    // Fast path: known Maclaurin series coefficients
    if point == arena.zero
        && let Some(result) = try_known_maclaurin(arena, expr, var, order as usize)
    {
        return Ok(result);
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
            return Err(crate::errors::SymplexError::ComputationFailed {
                operation: "series",
                reason: "pole detected at expansion point".into(),
            });
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
        Ok(arena.zero)
    } else if terms.len() == 1 {
        Ok(terms[0])
    } else {
        Ok(arena.add(&terms))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Known-coefficient fast paths
// ═══════════════════════════════════════════════════════════════════════════

/// Try to build a Maclaurin series using known coefficients.
/// Returns None if the expression isn't a recognized elementary function.
fn try_known_maclaurin(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    order: usize,
) -> Option<ExprId> {
    match arena.node(expr).clone() {
        // sin(x): coefficients (-1)^k / (2k+1)! for odd terms
        ExprNode::Sin(inner) if inner == var => {
            let mut terms = Vec::new();
            for k in 0..order {
                let n = 2 * k + 1;
                if n >= order {
                    break;
                }
                let sign: i64 = if k % 2 == 0 { 1 } else { -1 };
                let factorial = factorial_value(n as u64);
                let coeff = Ratio::new(BigInt::from(sign), factorial);
                let coeff_id = {
                    let nid = arena.intern_num(coeff);
                    arena.intern(ExprNode::Num(nid))
                };
                let power = arena.int(n as i64);
                let var_pow = arena.pow(var, power);
                let term = arena.mul(&[coeff_id, var_pow]);
                terms.push(term);
            }
            if terms.is_empty() {
                return Some(arena.zero);
            }
            Some(arena.add(&terms))
        }
        // cos(x): coefficients (-1)^k / (2k)! for even terms
        ExprNode::Cos(inner) if inner == var => {
            let mut terms = Vec::new();
            for k in 0..order {
                let n = 2 * k;
                if n >= order {
                    break;
                }
                let sign: i64 = if k % 2 == 0 { 1 } else { -1 };
                let factorial = factorial_value(n as u64);
                let coeff = Ratio::new(BigInt::from(sign), factorial);
                let coeff_id = {
                    let nid = arena.intern_num(coeff);
                    arena.intern(ExprNode::Num(nid))
                };
                if n == 0 {
                    terms.push(coeff_id);
                } else {
                    let power = arena.int(n as i64);
                    let var_pow = arena.pow(var, power);
                    let term = arena.mul(&[coeff_id, var_pow]);
                    terms.push(term);
                }
            }
            if terms.is_empty() {
                return Some(arena.zero);
            }
            Some(arena.add(&terms))
        }
        // exp(x): coefficients 1/k!
        ExprNode::Exp(inner) if inner == var => {
            let mut terms = Vec::new();
            for k in 0..order {
                let factorial = factorial_value(k as u64);
                let coeff = Ratio::new(BigInt::from(1), factorial);
                let coeff_id = {
                    let nid = arena.intern_num(coeff);
                    arena.intern(ExprNode::Num(nid))
                };
                if k == 0 {
                    terms.push(coeff_id);
                } else {
                    let power = arena.int(k as i64);
                    let var_pow = arena.pow(var, power);
                    let term = arena.mul(&[coeff_id, var_pow]);
                    terms.push(term);
                }
            }
            Some(arena.add(&terms))
        }
        _ => None,
    }
}

/// Compute n! as BigInt.
fn factorial_value(n: u64) -> BigInt {
    let mut result = BigInt::from(1);
    for i in 2..=n {
        result *= BigInt::from(i);
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Laurent series
// ═══════════════════════════════════════════════════════════════════════════

/// Compute a Laurent series expansion of `expr` in `var` around `point`
/// to the given `order` (number of terms in the regular part).
///
/// A Laurent series extends a Taylor series to allow negative powers of
/// `(x - a)`, i.e. poles. The result includes terms from `(x-a)^{-m}`
/// up to `(x-a)^{order-1}` where `m` is the detected pole order.
///
/// # Algorithm
///
/// 1. First try a regular Taylor series — if it succeeds, return it
///    (no pole, Laurent = Taylor).
/// 2. Otherwise, multiply `expr` by `(x - a)^k` for `k = 1, 2, …, 5`
///    until the Taylor series of the modified expression succeeds.
/// 3. Divide the resulting Taylor series back by `(x - a)^k` to recover
///    the Laurent series with negative-power terms.
///
/// Returns `Ok(series)` on success, `Err` if the expansion cannot be
/// computed (e.g. essential singularity, or pole order > 5).
pub(crate) fn laurent_series(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    order: u32,
) -> Result<ExprId, crate::errors::SymplexError> {
    // First try regular Taylor — if it works, there is no pole.
    if let Ok(ts) = series(arena, expr, var, point, order) {
        return Ok(ts);
    }

    // Build (var - point) once; reuse for each attempt.
    let x_minus_a = if arena.is_zero_structural(point) {
        var
    } else {
        arena.sub(var, point)
    };

    // Try multiplying by (x - a)^k for k = 1..=5 until the pole is
    // cancelled and a Taylor series succeeds.
    for k in 1u32..=5 {
        let k_id = arena.int(k as i64);
        let multiplier = arena.pow(x_minus_a, k_id);
        let modified = arena.mul(&[expr, multiplier]);

        // We request order + k terms so that after dividing back by
        // (x-a)^k we still have `order` terms in the regular part.
        if let Ok(ts) = series(arena, modified, var, point, order + k) {
            // Divide back by (x - a)^k to restore the negative powers.
            let neg_k = arena.int(-(k as i64));
            let divisor = arena.pow(x_minus_a, neg_k);
            let result = arena.mul(&[ts, divisor]);

            // Expand so that the product distributes across the sum,
            // giving explicit negative-power terms.
            let result = crate::expand::expand(arena, result);
            let result = crate::eval::eval(arena, result);
            return Ok(result);
        }
    }

    Err(crate::errors::SymplexError::ComputationFailed {
        operation: "laurent_series",
        reason: "could not determine pole order (tried up to order 5)".into(),
    })
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
        let result = series(&mut a, five, x, zero, 3).unwrap();
        assert_eq!(display(&a, result), "5");
    }

    #[test]
    fn series_x_around_zero() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        // series(x, x, 0, 3) = 0 + 1*x + 0 = x
        let result = series(&mut a, x, x, zero, 3).unwrap();
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
        let result = series(&mut a, x2, x, zero, 3).unwrap();
        assert_eq!(display(&a, result), "x^2");
    }

    #[test]
    fn series_exp_x_around_zero_order_4() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.exp(x);
        let zero = a.zero;
        // series(exp(x), x, 0, 4) = 1 + x + x^2/2 + x^3/6
        let result = series(&mut a, expr, x, zero, 4).unwrap();
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
        let result = series(&mut a, expr, x, zero, 4).unwrap();
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
        let result = series(&mut a, x, x, zero, 0).unwrap();
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
        let result = series(&mut a, x3, x, zero, 5).unwrap();
        let s = display(&a, result);
        assert!(s.contains("x^3"), "should recover x^3: {s}");
    }

    #[test]
    fn series_sin_fast_path() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let zero = a.zero;
        // sin(x) Maclaurin order 6: x - x³/6 + x⁵/120
        let result = series(&mut a, sin_x, x, zero, 6).unwrap();
        let s = a.display(result).to_string();
        assert!(s.contains("x"), "should contain x: {s}");
    }

    #[test]
    fn series_cos_fast_path() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let cos_x = a.cos(x);
        let zero = a.zero;
        let result = series(&mut a, cos_x, x, zero, 5).unwrap();
        let s = a.display(result).to_string();
        assert!(s.contains("1"), "cos series starts with 1: {s}");
    }

    #[test]
    fn series_exp_fast_path() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let exp_x = a.exp(x);
        let zero = a.zero;
        let result = series(&mut a, exp_x, x, zero, 5).unwrap();
        let s = a.display(result).to_string();
        assert!(s.contains("1") && s.contains("x"), "exp series: {s}");
    }
}
