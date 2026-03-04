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
use tracing;

/// Maximum L'Hôpital iterations to prevent infinite loops.
const MAX_LHOPITAL: usize = 5;

/// Compute the limit of `expr` as `var` approaches `point`.
pub(crate) fn limit(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Result<ExprId, crate::errors::SymplexError> {
    // Check for limit at infinity
    if point == arena.infinity {
        return limit_at_infinity(arena, expr, var, true);
    }
    if point == arena.neg_infinity {
        return limit_at_infinity(arena, expr, var, false);
    }

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

        let n_val_is_zero = arena.is_zero_structural(n_val);
        let d_val_is_zero = arena.is_zero_structural(d_val);
        tracing::debug!(
            numer_at_point = ?n_val_is_zero,
            denom_at_point = ?d_val_is_zero,
            "limit: evaluated numer/denom at point"
        );

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

    tracing::debug!(
        depth = depth,
        numer_zero = n_is_zero,
        denom_zero = d_is_zero,
        numer_inf = n_is_inf,
        denom_inf = d_is_inf,
        "L'Hôpital: indeterminate form detected"
    );

    let n_prime = crate::diff::diff(arena, numer, var);
    let d_prime = crate::diff::diff(arena, denom, var);

    // After computing derivatives:
    tracing::debug!("L'Hôpital: differentiated, trying substitution");

    // Evaluate the differentiated num/denom SEPARATELY first so we can
    // detect another 0/0 indeterminate form.  The combined-ratio path
    // (`arena.div` then `eval`) can mis-canonicalize 0/0 → 0 because
    // the arena sees `Mul([0, Pow(0,-1)])` and eagerly folds `0*_ → 0`.
    let np_subst = crate::subs::subs(arena, n_prime, var, point);
    let np_val = crate::eval::eval(arena, np_subst);
    let dp_subst = crate::subs::subs(arena, d_prime, var, point);
    let dp_val = crate::eval::eval(arena, dp_subst);

    let np_zero = arena.is_zero_structural(np_val);
    let dp_zero = arena.is_zero_structural(dp_val);

    tracing::debug!(
        np_finite = is_finite_number(arena, np_val),
        dp_finite = is_finite_number(arena, dp_val),
        np_zero = np_zero,
        dp_zero = dp_zero,
        "L'Hôpital: separate derivative evaluation"
    );

    // If both derivatives are still 0 at the point, it's another 0/0 —
    // recurse immediately instead of trusting the canonicalized ratio.
    if np_zero && dp_zero {
        return try_lhopital(arena, n_prime, d_prime, var, point, depth + 1);
    }

    // If denom is nonzero and both are finite, compute the ratio directly.
    if is_finite_number(arena, np_val) && is_finite_number(arena, dp_val) && !dp_zero {
        let result_ratio = arena.div(np_val, dp_val);
        let result = crate::eval::eval(arena, result_ratio);
        if is_finite_number(arena, result) {
            return Some(result);
        }
    }

    // Try the combined ratio substitution as a fallback (works when the
    // arena doesn't mis-canonicalize).
    let ratio = arena.div(n_prime, d_prime);
    let subst = crate::subs::subs(arena, ratio, var, point);
    let evaled = crate::eval::eval(arena, subst);

    if is_finite_number(arena, evaled) {
        return Some(evaled);
    }

    // Recurse with differentiated numerator/denominator.
    try_lhopital(arena, n_prime, d_prime, var, point, depth + 1)
}

/// Compute lim(x→∞) expr or lim(x→-∞) expr.
///
/// Strategies:
/// 1. For rational functions P(x)/Q(x): compare leading degrees
/// 2. For exp/ln compositions: direct evaluation of dominant terms
/// 3. Substitution x = 1/t, compute lim(t→0+)
pub(crate) fn limit_at_infinity(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    positive: bool, // true for +∞, false for -∞
) -> Result<ExprId, crate::errors::SymplexError> {
    // Strategy 1: Try substitution x = 1/t and take limit as t→0
    // This converts lim(x→∞) to lim(t→0+)
    tracing::debug!(
        strategy = "substitution x=1/t",
        "limit_at_infinity: trying reciprocal substitution"
    );
    let t = arena.symbol("__limit_t");
    let one = arena.one;
    let t_inv = arena.div(one, t); // 1/t

    // For -∞: substitute x = -1/t
    let sub_expr = if positive {
        arena.subs_structural(expr, var, t_inv)
    } else {
        let neg_t_inv = arena.neg(t_inv);
        arena.subs_structural(expr, var, neg_t_inv)
    };

    // Simplify the substituted expression
    let simplified = crate::eval::eval(arena, sub_expr);
    let expanded = crate::expand::expand(arena, simplified);
    let evaled = crate::eval::eval(arena, expanded);

    // Now take lim(t→0)
    // First try direct substitution
    let at_zero = arena.subs_structural(evaled, t, arena.zero);
    let at_zero_eval = crate::eval::eval(arena, at_zero);

    // Check if result is finite
    if is_finite_result(arena, at_zero_eval) {
        return Ok(at_zero_eval);
    }

    // Strategy 2: For rational functions, analyze degree
    // Try as_numer_denom, get degrees of numerator and denominator in t
    let (numer, denom) = crate::polybridge::as_numer_denom(arena, evaled);
    if denom != arena.one {
        // We have a fraction in t — try to determine the limit
        let n_deg = crate::polybridge::poly_degree(arena, numer, t);
        let d_deg = crate::polybridge::poly_degree(arena, denom, t);

        if let (Some(nd), Some(dd)) = (n_deg, d_deg) {
            tracing::debug!(
                numer_degree = nd,
                denom_degree = dd,
                "limit_at_infinity: degree comparison"
            );
            if nd < dd {
                // Numerator degree < denominator degree → limit is 0
                // (in terms of t→0, this means the numerator vanishes faster)
                // If numer has lower degree, numer→0 faster, limit = 0
                return Ok(arena.zero);
            }
            if nd == dd {
                // Same degree: limit is ratio of leading coefficients
                // Cancel common factors and substitute t=0
                let cancelled = crate::polybridge::cancel(arena, evaled, t);
                let result = arena.subs_structural(cancelled, t, arena.zero);
                let result = crate::eval::eval(arena, result);
                if is_finite_result(arena, result) {
                    return Ok(result);
                }
            }
            // nd > dd: limit is ±∞ (in the original variable)
            if nd > dd {
                return Ok(arena.infinity);
            }
        }
    }

    // Strategy 3: Try L'Hôpital on the substituted form
    // Use the existing limit machinery on lim(t→0)
    let zero = arena.zero;
    match limit(arena, evaled, t, zero) {
        Ok(result) => Ok(result),
        Err(_) => Err(crate::errors::SymplexError::ComputationFailed {
            operation: "limit",
            reason: "could not compute limit at infinity".into(),
        }),
    }
}

/// Check if a result is a finite number (not infinity, NaN, etc.)
fn is_finite_result(arena: &Arena, id: ExprId) -> bool {
    matches!(
        arena.node(id),
        ExprNode::Num(_) | ExprNode::Pi | ExprNode::E
    )
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
