//! Thin shim between `eval()` and the symbolic summation engine.
//!
//! All strategies (Faulhaber with Bernoulli numbers, partial-fraction
//! telescoping, geometric / arithmetico-geometric series, binomial
//! identities, Gosper's algorithm, p-series and power-series recognition,
//! …) live in [`crate::calculus::summation`].  This module only adapts the
//! engine's [`SumOutcome`](crate::calculus::summation::SumOutcome) to the
//! `Option<ExprId>` contract that `eval` expects.

use crate::base::arena::Arena;
use crate::base::node::ExprId;
use crate::calculus::summation::{self, SumOutcome};

/// Attempt closed-form evaluation of `Σ_{var=lower}^{upper} body`.
///
/// Returns `Some(result)` if a closed form was found (or the sum was proven
/// to diverge to `±∞`, in which case the infinity is returned), `None`
/// otherwise.
pub(crate) fn eval_sum_symbolic(
    arena: &mut Arena,
    body: ExprId,
    var: ExprId,
    lower: ExprId,
    upper: ExprId,
) -> Option<ExprId> {
    match summation::summation(arena, body, var, lower, upper) {
        SumOutcome::Closed(id) => Some(id),
        SumOutcome::Divergent(Some(inf)) => Some(inf),
        SumOutcome::Divergent(None) | SumOutcome::Unevaluated => None,
    }
}

// Not yet called from `eval` (that file is owned elsewhere); see the doc below.
#[allow(dead_code)]
/// Attempt closed-form evaluation of `Π_{var=lower}^{upper} body`.
///
/// Same contract as [`eval_sum_symbolic`]; called from the `Product_` arm of
/// `transforms::eval`.
pub(crate) fn eval_product_symbolic(
    arena: &mut Arena,
    body: ExprId,
    var: ExprId,
    lower: ExprId,
    upper: ExprId,
) -> Option<ExprId> {
    match summation::product(arena, body, var, lower, upper) {
        SumOutcome::Closed(id) => Some(id),
        SumOutcome::Divergent(Some(inf)) => Some(inf),
        SumOutcome::Divergent(None) | SumOutcome::Unevaluated => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build arena, create sum, try closed form, then evaluate numerically.
    fn eval_sum(power: usize, lo: i64, hi: i64) -> String {
        let mut arena = Arena::new();
        let var = arena.symbol("k");
        let lower = arena.int(lo);
        let upper = arena.int(hi);

        let body = if power == 1 {
            var
        } else {
            let exp = arena.int(power as i64);
            arena.pow(var, exp)
        };

        let result = eval_sum_symbolic(&mut arena, body, var, lower, upper);
        assert!(
            result.is_some(),
            "Expected closed form for k^{power} from {lo} to {hi}"
        );
        let expr = result.unwrap();
        // Evaluate the closed form
        let evaled = crate::transforms::eval::eval(&mut arena, expr);
        arena.display(evaled).to_string()
    }

    #[test]
    fn faulhaber_k_1_to_10() {
        // Σk from 1 to 10 = 55
        assert_eq!(eval_sum(1, 1, 10), "55");
    }

    #[test]
    fn faulhaber_k_sq_1_to_10() {
        // Σk² from 1 to 10 = 385
        assert_eq!(eval_sum(2, 1, 10), "385");
    }

    #[test]
    fn faulhaber_k_cube_1_to_10() {
        // Σk³ from 1 to 10 = 3025
        assert_eq!(eval_sum(3, 1, 10), "3025");
    }

    #[test]
    fn faulhaber_k_4_1_to_10() {
        // Σk⁴ from 1 to 10 = 25333
        assert_eq!(eval_sum(4, 1, 10), "25333");
    }

    #[test]
    fn faulhaber_k_7_big_range_uses_closed_form() {
        // Σ_{k=1}^{5000} k^7 — beyond the enumeration limit, so Faulhaber must apply.
        let mut arena = Arena::new();
        let var = arena.symbol("k");
        let lower = arena.int(1);
        let upper = arena.int(5000);
        let seven = arena.int(7);
        let body = arena.pow(var, seven);
        let result = eval_sum_symbolic(&mut arena, body, var, lower, upper).unwrap();
        let evaled = crate::transforms::eval::eval(&mut arena, result);
        // Known: Σ_{k=1}^{n} k^7 with n = 5000
        let s = arena.display(evaled).to_string();
        assert_eq!(s, "48867196614583151041668750000");
    }

    #[test]
    fn constant_body() {
        let mut arena = Arena::new();
        let var = arena.symbol("k");
        let lower = arena.int(1);
        let upper = arena.int(10);
        let body = arena.int(5);

        let result = eval_sum_symbolic(&mut arena, body, var, lower, upper);
        assert!(result.is_some());
        let evaled = crate::transforms::eval::eval(&mut arena, result.unwrap());
        assert_eq!(arena.display(evaled).to_string(), "50");
    }

    #[test]
    fn geometric_series() {
        let mut arena = Arena::new();
        let var = arena.symbol("k");
        let lower = arena.int(0);
        let upper = arena.int(9);
        let two = arena.int(2);
        let body = arena.pow(two, var); // 2^k

        let result = eval_sum_symbolic(&mut arena, body, var, lower, upper);
        assert!(result.is_some());
        let evaled = crate::transforms::eval::eval(&mut arena, result.unwrap());
        assert_eq!(arena.display(evaled).to_string(), "1023");
    }

    #[test]
    fn product_shim_factorial() {
        let mut arena = Arena::new();
        let var = arena.symbol("k");
        let n = arena.symbol("n");
        let lower = arena.int(1);
        let result = eval_product_symbolic(&mut arena, var, var, lower, n).unwrap();
        assert_eq!(arena.display(result).to_string(), "n!");
    }
}
