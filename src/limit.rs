//! Symbolic limit computation.
//!
//! Implements [`limit`], which computes the limit of an expression
//! as a variable approaches a point.
//!
//! # Algorithm
//!
//! 1. **Direct substitution:** substitute the point and evaluate.
//!    If the result is a finite number, return it.
//! 2. **L'Hôpital's rule:** if the result is 0/0 or ∞/∞,
//!    differentiate numerator and denominator and retry.
//! 3. **Series fallback:** expand as Taylor series around the point,
//!    take the constant term.
//! 4. **Fallback:** return the expression unchanged.

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode};

/// Maximum L'Hôpital iterations to prevent infinite loops.
const MAX_LHOPITAL: usize = 5;

/// Compute the limit of `expr` as `var` approaches `point`.
pub(crate) fn limit(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Result<ExprId, crate::errors::SymplexError> {
    // Step 1+2 combined: decompose into numerator/denominator first.
    //
    // We must check for indeterminate forms *before* doing a naive
    // direct substitution on the whole expression, because the arena's
    // canonicalization of `Mul([0, Pow(0,-1)])` eagerly collapses
    // `0 * anything` to `0` without noticing that the "anything" is
    // actually `0^(-1)` (i.e. infinity).  By evaluating numerator and
    // denominator *separately* at the point we sidestep this issue.
    let (numer, denom) = crate::polybridge::as_numer_denom(arena, expr);

    if denom != arena.one {
        // We have a genuine fraction — evaluate num and denom separately.
        let n_subst = crate::subs::subs(arena, numer, var, point);
        let n_val = crate::eval::eval(arena, n_subst);
        let d_subst = crate::subs::subs(arena, denom, var, point);
        let d_val = crate::eval::eval(arena, d_subst);

        // If both are finite and denom is nonzero, compute directly.
        if is_finite_number(arena, n_val)
            && is_finite_number(arena, d_val)
            && !arena.is_zero_structural(d_val)
        {
            let ratio = arena.div(n_val, d_val);
            let result = crate::eval::eval(arena, ratio);
            if is_finite_number(arena, result) {
                return Ok(result);
            }
        }

        // Try L'Hôpital if we have an indeterminate form.
        if let Some(r) = try_lhopital(arena, numer, denom, var, point, 0) {
            return Ok(r);
        }
    } else {
        // No denominator — try plain direct substitution.
        let subst = crate::subs::subs(arena, expr, var, point);
        let evaled = crate::eval::eval(arena, subst);
        if is_finite_number(arena, evaled) {
            return Ok(evaled);
        }
    }

    // Step 3: Series fallback.
    // Expand as Taylor series around the point, order 1 gives the limit.
    if let Ok(series_result) = crate::series::series(arena, expr, var, point, 1) {
        let series_evaled = crate::eval::eval(arena, series_result);
        if is_finite_number(arena, series_evaled) && series_evaled != expr {
            return Ok(series_evaled);
        }
    }

    // Step 4: Couldn't compute — return an error.
    Err(crate::errors::SymplexError::ComputationFailed {
        operation: "limit",
        reason:
            "could not determine the limit via substitution, L'Hôpital's rule, or series expansion"
                .into(),
    })
}

/// Try L'Hôpital's rule: lim f/g = lim f'/g' when f(a)=g(a)=0 or both →∞.
fn try_lhopital(
    arena: &mut Arena,
    numer: ExprId,
    denom: ExprId,
    var: ExprId,
    point: ExprId,
    depth: usize,
) -> Option<ExprId> {
    if depth >= MAX_LHOPITAL {
        return None;
    }

    // Evaluate numerator and denominator at the point *separately*
    // to avoid premature cancellation in Mul canonicalization.
    let n_subst = crate::subs::subs(arena, numer, var, point);
    let n_at_point = crate::eval::eval(arena, n_subst);
    let d_subst = crate::subs::subs(arena, denom, var, point);
    let d_at_point = crate::eval::eval(arena, d_subst);

    let n_is_zero = arena.is_zero_structural(n_at_point);
    let d_is_zero = arena.is_zero_structural(d_at_point);

    let n_is_inf = n_at_point == arena.infinity || n_at_point == arena.neg_infinity;
    let d_is_inf = d_at_point == arena.infinity || d_at_point == arena.neg_infinity;

    // Only apply L'Hôpital to genuine indeterminate forms: 0/0 or ∞/∞.
    if !((n_is_zero && d_is_zero) || (n_is_inf && d_is_inf)) {
        return None;
    }

    let n_prime = crate::diff::diff(arena, numer, var);
    let d_prime = crate::diff::diff(arena, denom, var);

    // Try direct substitution of the differentiated ratio.
    let ratio = arena.div(n_prime, d_prime);
    let subst = crate::subs::subs(arena, ratio, var, point);
    let evaled = crate::eval::eval(arena, subst);

    if is_finite_number(arena, evaled) {
        return Some(evaled);
    }

    // Also try evaluating the differentiated num/denom separately,
    // in case the combined ratio has the same canonicalization issue.
    let np_subst = crate::subs::subs(arena, n_prime, var, point);
    let np_val = crate::eval::eval(arena, np_subst);
    let dp_subst = crate::subs::subs(arena, d_prime, var, point);
    let dp_val = crate::eval::eval(arena, dp_subst);

    if is_finite_number(arena, np_val)
        && is_finite_number(arena, dp_val)
        && !arena.is_zero_structural(dp_val)
    {
        let result_ratio = arena.div(np_val, dp_val);
        let result = crate::eval::eval(arena, result_ratio);
        if is_finite_number(arena, result) {
            return Some(result);
        }
    }

    // Recurse with differentiated numerator/denominator.
    try_lhopital(arena, n_prime, d_prime, var, point, depth + 1)
}

/// Check if an expression is a finite number (not infinity, not NaN, not symbolic).
fn is_finite_number(arena: &Arena, id: ExprId) -> bool {
    if id == arena.infinity
        || id == arena.neg_infinity
        || id == arena.nan
        || id == arena.complex_infinity
    {
        return false;
    }
    matches!(arena.node(id), ExprNode::Num(_))
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
    fn limit_polynomial_direct_sub() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        // lim_{x→2} x^2 = 4
        let x2 = a.pow(x, two);
        let result = limit(&mut a, x2, x, two).unwrap();
        assert_eq!(display(&a, result), "4");
    }

    #[test]
    fn limit_sin_x_over_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        // lim_{x→0} sin(x)/x = 1 (L'Hôpital: cos(x)/1 at x=0 = 1)
        let sin_x = a.sin(x);
        let expr = a.div(sin_x, x);
        let result = limit(&mut a, expr, x, zero).unwrap();
        assert_eq!(display(&a, result), "1");
    }

    #[test]
    fn limit_constant() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let zero = a.zero;
        let result = limit(&mut a, five, x, zero).unwrap();
        assert_eq!(display(&a, result), "5");
    }

    #[test]
    fn limit_x_squared_minus_one_over_x_minus_one() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        // lim_{x→1} (x^2-1)/(x-1) = lim_{x→1} 2x/1 = 2
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let numer = a.sub(x2, one);
        let denom = a.sub(x, one);
        let expr = a.div(numer, denom);
        let result = limit(&mut a, expr, x, one).unwrap();
        assert_eq!(display(&a, result), "2");
    }

    #[test]
    fn limit_direct_sub_works() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        // lim_{x→3} (x+1) = 4
        let expr = a.add(&[x, a.one]);
        let result = limit(&mut a, expr, x, three).unwrap();
        assert_eq!(display(&a, result), "4");
    }
}
