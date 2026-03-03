//! Simplification engine — try multiple strategies, keep shortest.
//!
//! This module provides [`count_ops`] for measuring expression complexity,
//! and [`smart_simplify`] which tries multiple simplification strategies
//! and returns the result with the lowest operation count.

use crate::arena::Arena;
use crate::node::ExprId;
use crate::walk;

/// Count the number of operations (nodes) in an expression.
///
/// Each non-atom node counts as 1. Atoms (numbers, symbols, constants)
/// count as 0. This gives a rough measure of expression complexity.
pub(crate) fn count_ops(arena: &Arena, expr: ExprId) -> usize {
    let post_order = walk::post_order_ids(arena, expr);
    let mut count = 0;
    for &id in &post_order {
        if !arena.node(id).is_atom() {
            count += 1;
        }
    }
    count
}

/// Try multiple simplification strategies and return the "simplest" result.
///
/// Strategies tried:
/// 1. eval only
/// 2. eval → simplify (pattern rules)
/// 3. eval → expand → simplify
/// 4. eval → cancel (with dummy var) → simplify
/// 5. eval → expand_trig → simplify
/// 6. eval → logcombine → simplify
/// 7. eval → factor_terms → simplify
///
/// The result with the lowest `count_ops` is returned.
/// If the best result is more than 1.7× the complexity of the original,
/// the original is returned (to prevent "simplification" that makes things worse).
pub(crate) fn smart_simplify(arena: &mut Arena, expr: ExprId) -> ExprId {
    let original_ops = count_ops(arena, expr);

    // Strategy 1: eval only
    let s1 = crate::eval::eval(arena, expr);

    // Strategy 2: eval → simplify
    let rules = crate::pattern::basic_rules(arena);
    let s2_eval = crate::eval::eval(arena, expr);
    let (s2, _) = crate::pattern::apply_rules(arena, s2_eval, &rules);

    // Strategy 3: eval → expand → simplify
    let s3_eval = crate::eval::eval(arena, expr);
    let s3_expand = crate::expand::expand(arena, s3_eval);
    let s3_eval2 = crate::eval::eval(arena, s3_expand);
    let (s3, _) = crate::pattern::apply_rules(arena, s3_eval2, &rules);

    // Strategy 4: eval → factor_terms → simplify
    let s4_eval = crate::eval::eval(arena, expr);
    let s4_factor = crate::factor_terms::factor_terms(arena, s4_eval);
    let (s4, _) = crate::pattern::apply_rules(arena, s4_factor, &rules);

    // Strategy 5: eval → expand_trig → simplify
    let s5_eval = crate::eval::eval(arena, expr);
    let s5_trig = crate::trig_expand::expand_trig(arena, s5_eval);
    let s5_eval2 = crate::eval::eval(arena, s5_trig);
    let (s5, _) = crate::pattern::apply_rules(arena, s5_eval2, &rules);

    // Strategy 6: eval → logcombine → simplify
    let s6_eval = crate::eval::eval(arena, expr);
    let s6_log = crate::log_combine::log_combine(arena, s6_eval);
    let (s6, _) = crate::pattern::apply_rules(arena, s6_log, &rules);

    // Collect candidates
    let candidates = [expr, s1, s2, s3, s4, s5, s6];

    // Pick the one with lowest ops
    let best = candidates
        .iter()
        .copied()
        .min_by_key(|&e| count_ops(arena, e))
        .unwrap_or(expr);

    // Guard: don't return something much more complex than original
    let best_ops = count_ops(arena, best);
    if original_ops > 0 && best_ops as f64 > 1.7 * original_ops as f64 {
        return expr;
    }

    best
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
    fn count_ops_atom() {
        let a = Arena::new();
        assert_eq!(count_ops(&a, a.zero), 0);
        assert_eq!(count_ops(&a, a.one), 0);
        assert_eq!(count_ops(&a, a.pi), 0);
    }

    #[test]
    fn count_ops_simple() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let expr = a.add(&[x, two]);
        assert_eq!(count_ops(&a, expr), 1); // one Add
    }

    #[test]
    fn count_ops_nested() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let two = a.int(2);
        let expr = a.pow(sin_x, two);
        assert_eq!(count_ops(&a, expr), 2); // Sin + Pow
    }

    #[test]
    fn smart_simplify_trig_identity() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let cos_x = a.cos(x);
        let two = a.int(2);
        let sin2 = a.pow(sin_x, two);
        let cos2 = a.pow(cos_x, two);
        let expr = a.add(&[sin2, cos2]);
        let result = smart_simplify(&mut a, expr);
        assert_eq!(display(&a, result), "1");
    }

    #[test]
    fn smart_simplify_exp_ln() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let ln_x = a.ln(x);
        let expr = a.exp(ln_x);
        let result = smart_simplify(&mut a, expr);
        assert_eq!(display(&a, result), "x");
    }

    #[test]
    fn smart_simplify_doesnt_bloat() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x is already simple — shouldn't get more complex
        let result = smart_simplify(&mut a, x);
        assert_eq!(result, x);
    }

    #[test]
    fn smart_simplify_eval_catches_sin_zero() {
        let mut a = Arena::new();
        let expr = a.sin(a.zero);
        let result = smart_simplify(&mut a, expr);
        assert_eq!(display(&a, result), "0");
    }

    #[test]
    fn count_ops_complex_expression() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let xy = a.mul(&[x, y]);
        let expr = a.add(&[x2, xy, y]);
        // Pow + Mul + Add = 3
        assert_eq!(count_ops(&a, expr), 3);
    }
}
