//! Simplification engine — try multiple strategies, keep shortest.
//!
//! This module provides [`count_ops`] for measuring expression complexity,
//! [`smart_simplify`] which tries multiple simplification strategies
//! and returns the result with the lowest operation count, and
//! [`full_simplify`] / [`full_simplify_trace`] which iterate
//! eval → expand → simplify → **cancel** until convergence.

use crate::arena::Arena;
use crate::node::ExprId;
use crate::walk;
use num_traits::One;
use tracing;

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
/// 4. eval → factor_terms → simplify
/// 5. eval → expand_trig → simplify
/// 6. eval → logcombine → simplify
/// 7. eval → cancel with free symbols → simplify
///
/// The result with the lowest `count_ops` is returned.
/// If the best result is more than 1.7× the complexity of the original,
/// the original is returned (to prevent "simplification" that makes things worse).
pub(crate) fn smart_simplify(arena: &mut Arena, expr: ExprId) -> ExprId {
    let original_ops = count_ops(arena, expr);

    // Strategy 1: eval only
    let s1 = crate::eval::eval(arena, expr);
    tracing::trace!(
        strategy = "eval",
        ops = count_ops(arena, s1),
        "strategy evaluated"
    );

    // Strategy 2: eval → simplify
    let rules = crate::pattern::basic_rules(arena);
    let s2_eval = crate::eval::eval(arena, expr);
    let (s2, _) = crate::pattern::apply_rules(arena, s2_eval, &rules);
    tracing::trace!(
        strategy = "eval+rules",
        ops = count_ops(arena, s2),
        "strategy evaluated"
    );

    // Strategy 3: eval → expand → simplify
    let s3_eval = crate::eval::eval(arena, expr);
    let s3_expand = crate::expand::expand(arena, s3_eval);
    let s3_eval2 = crate::eval::eval(arena, s3_expand);
    let (s3, _) = crate::pattern::apply_rules(arena, s3_eval2, &rules);
    tracing::trace!(
        strategy = "eval+expand+rules",
        ops = count_ops(arena, s3),
        "strategy evaluated"
    );

    // Strategy 4: eval → factor_terms → simplify inner → multiply GCD back
    let s4_eval = crate::eval::eval(arena, expr);
    let (gcd, s4_inner) = crate::factor_terms::factor_terms_pair(arena, s4_eval);
    let (s4_simplified, _) = crate::pattern::apply_rules(arena, s4_inner, &rules);
    let s4 = if gcd.is_one() {
        s4_simplified
    } else {
        let nid = arena.intern_num(gcd);
        let gcd_id = arena.intern(crate::node::ExprNode::Num(nid));
        arena.mul(&[gcd_id, s4_simplified])
    };
    tracing::trace!(
        strategy = "eval+factor+rules",
        ops = count_ops(arena, s4),
        "strategy evaluated"
    );

    // Strategy 5: eval → expand_trig → simplify
    let s5_eval = crate::eval::eval(arena, expr);
    let s5_trig = crate::trig_expand::expand_trig(arena, s5_eval);
    let s5_eval2 = crate::eval::eval(arena, s5_trig);
    let (s5, _) = crate::pattern::apply_rules(arena, s5_eval2, &rules);
    tracing::trace!(
        strategy = "eval+trig_expand+rules",
        ops = count_ops(arena, s5),
        "strategy evaluated"
    );

    // Strategy 6: eval → logcombine → simplify
    let s6_eval = crate::eval::eval(arena, expr);
    let s6_log = crate::log_combine::log_combine(arena, s6_eval);
    let (s6, _) = crate::pattern::apply_rules(arena, s6_log, &rules);
    tracing::trace!(
        strategy = "eval+logcombine+rules",
        ops = count_ops(arena, s6),
        "strategy evaluated"
    );

    // Strategy 7: eval → cancel with free symbols → simplify
    // For rational expressions like (x²-1)/(x-1) → x+1
    let s7 = {
        let evaled = crate::eval::eval(arena, expr);
        let free = crate::walk::free_symbols(arena, evaled);
        let mut best = evaled;
        let mut best_ops = count_ops(arena, evaled);
        for &sym in &free {
            let cancelled = crate::polybridge::cancel(arena, evaled, sym);
            let cancelled_eval = crate::eval::eval(arena, cancelled);
            let (cancelled_simp, _) = crate::pattern::apply_rules(arena, cancelled_eval, &rules);
            let ops = count_ops(arena, cancelled_simp);
            if ops < best_ops {
                best = cancelled_simp;
                best_ops = ops;
            }
        }
        best
    };
    tracing::trace!(
        strategy = "eval+cancel+rules",
        ops = count_ops(arena, s7),
        "strategy evaluated"
    );

    // Collect candidates
    let candidates = [expr, s1, s2, s3, s4, s5, s6, s7];
    let strategy_names = [
        "original",
        "eval",
        "eval+rules",
        "eval+expand+rules",
        "eval+factor+rules",
        "eval+trig_expand+rules",
        "eval+logcombine+rules",
        "eval+cancel+rules",
    ];

    // Pick the one with lowest ops
    let best_idx = candidates
        .iter()
        .enumerate()
        .min_by_key(|&(_, &e)| count_ops(arena, e))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let best = candidates[best_idx];
    let strategy_name = strategy_names[best_idx];

    // Guard: don't return something much more complex than original
    let best_ops = count_ops(arena, best);

    tracing::debug!(
        winning_strategy = strategy_name,
        ops_original = original_ops,
        ops_result = best_ops,
        "smart_simplify selected strategy"
    );

    if original_ops > 0 && best_ops as f64 > 1.7 * original_ops as f64 {
        return expr;
    }

    best
}

/// Fully simplify an expression by iterating eval → expand → simplify → cancel.
///
/// This is the engine-level counterpart of `Expr::full_simplify`.  Unlike
/// the original `full_simplify_trace` loop in `expr.rs`, this version
/// includes a polynomial-cancellation step (via [`crate::polybridge::cancel`])
/// after every simplify pass, so rational expressions like `(x²-4)/(x-2)`
/// are reduced to `x+2`.
pub(crate) fn full_simplify(arena: &mut Arena, expr: ExprId) -> ExprId {
    full_simplify_trace(arena, expr).0
}

/// Like [`full_simplify`] but also returns the accumulated rewrite-rule
/// trace (one [`crate::pattern::Step`] per rule firing).
pub(crate) fn full_simplify_trace(
    arena: &mut Arena,
    expr: ExprId,
) -> (ExprId, Vec<crate::pattern::Step>) {
    const MAX_ITERATIONS: usize = 10;
    let rules = crate::pattern::basic_rules(arena);
    let mut current = expr;
    let mut all_steps: Vec<crate::pattern::Step> = Vec::new();

    for i in 0..MAX_ITERATIONS {
        let evaled = crate::eval::eval(arena, current);

        // ── try cancel BEFORE expand (preserves rational structure) ──
        let free = crate::walk::free_symbols(arena, evaled);
        let mut cancelled = evaled;
        for &sym in &free {
            cancelled = crate::polybridge::cancel(arena, cancelled, sym);
        }
        let cancelled_eval = crate::eval::eval(arena, cancelled);
        let (cancelled_simp, cancel_steps) =
            crate::pattern::apply_rules(arena, cancelled_eval, &rules);
        all_steps.extend(cancel_steps);

        // ── also try the classic path: expand → simplify ──
        let expanded = crate::expand::expand(arena, evaled);
        let expanded_eval = crate::eval::eval(arena, expanded);
        let (expanded_simp, expand_steps) =
            crate::pattern::apply_rules(arena, expanded_eval, &rules);
        all_steps.extend(expand_steps);

        // Pick whichever result has fewer operations.
        let best = if count_ops(arena, cancelled_simp) <= count_ops(arena, expanded_simp) {
            cancelled_simp
        } else {
            expanded_simp
        };

        tracing::debug!(
            iteration = i,
            changed = (best != current),
            "full_simplify iteration"
        );

        if best == current {
            return (best, all_steps);
        }
        current = best;
    }

    (current, all_steps)
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

    #[test]
    fn smart_simplify_cancel_rational() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let one = a.one;
        // (x²-1)/(x-1) should simplify to x+1
        let x2 = a.pow(x, two);
        let numer = a.sub(x2, one);
        let denom = a.sub(x, one);
        let expr = a.div(numer, denom);
        let result = smart_simplify(&mut a, expr);
        let s = a.display(result).to_string();
        assert!(
            s.contains("1") && s.contains("x") && !s.contains("/"),
            "(x²-1)/(x-1) should simplify to x+1, got: {s}"
        );
    }

    /// Regression: Strategy 4 calls `factor_terms_pair` but discards the GCD,
    /// entering only the GCD-stripped inner expression into the candidate pool.
    /// Because `count_ops` is lower for the stripped version, it wins — producing
    /// a result that is NOT mathematically equivalent to the input.
    ///
    /// For `6*x + 12`:
    ///   factor_terms_pair → (gcd=6, inner=x+2)
    ///   Strategy 4 keeps only `x+2`, which has 1 op vs 3 ops for `6*x+12`.
    ///   So smart_simplify returns `x+2` instead of `6*x+12` (or `6*(x+2)`).
    #[test]
    fn smart_simplify_strategy4_discards_gcd() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let six = a.int(6);
        let twelve = a.int(12);
        let six_x = a.mul(&[six, x]);
        let expr = a.add(&[six_x, twelve]); // 6*x + 12

        // Confirm what factor_terms_pair returns
        let eval_expr = crate::eval::eval(&mut a, expr);
        let (gcd, inner) = crate::factor_terms::factor_terms_pair(&mut a, eval_expr);
        let inner_s = display(&a, inner);
        assert_eq!(
            gcd,
            num_rational::Ratio::from_integer(num_bigint::BigInt::from(6)),
            "GCD should be 6"
        );
        assert!(
            inner_s.contains("x") && inner_s.contains("2"),
            "inner should be x+2, got: {inner_s}"
        );

        // Now run smart_simplify — the bug causes it to return x+2
        let result = smart_simplify(&mut a, expr);
        let result_s = display(&a, result);

        // The simplified form must still contain the factor 6 (or be
        // equivalent, e.g. "6*(x + 2)" or "6*x + 12").  If it equals
        // just "x + 2" then Strategy 4's GCD was silently dropped.
        let result_ops = count_ops(&a, result);
        let inner_ops = count_ops(&a, inner);
        assert!(
            result_s.contains("6") || result_s.contains("12"),
            "BUG: smart_simplify returned '{}' for 6*x+12 — \
             the GCD factor 6 was discarded by Strategy 4 \
             (result has {} ops vs inner's {} ops)",
            result_s,
            result_ops,
            inner_ops
        );
    }

    #[test]
    fn smart_simplify_cancels_x2_minus_4_over_x_minus_2() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let four = a.int(4);
        let x_sq = a.pow(x, two);
        let numer = a.sub(x_sq, four); // x² - 4
        let denom = a.sub(x, two); // x - 2
        let expr = a.div(numer, denom); // (x²-4)/(x-2)
        let result = smart_simplify(&mut a, expr);
        let expected = a.add(&[x, two]); // x + 2
        assert_eq!(
            result,
            expected,
            "(x²-4)/(x-2) should smart_simplify to x+2, got: {}",
            display(&a, result)
        );
    }

    #[test]
    fn full_simplify_cancels_x2_minus_4_over_x_minus_2() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let four = a.int(4);
        let x_sq = a.pow(x, two);
        let numer = a.sub(x_sq, four); // x² - 4
        let denom = a.sub(x, two); // x - 2
        let expr = a.div(numer, denom); // (x²-4)/(x-2)
        let result = full_simplify(&mut a, expr);
        let expected = a.add(&[x, two]); // x + 2
        assert_eq!(
            result,
            expected,
            "(x²-4)/(x-2) should full_simplify to x+2, got: {}",
            display(&a, result)
        );
    }

    #[test]
    fn full_simplify_x2_minus_1_over_x_minus_1() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let numer = a.sub(x2, a.one); // x² - 1
        let denom = a.sub(x, a.one); // x - 1
        let expr = a.div(numer, denom);
        let result = full_simplify(&mut a, expr);
        let s = a.display(result).to_string();
        assert!(
            s.contains("1") && s.contains("x") && !s.contains("/"),
            "(x²-1)/(x-1) should full_simplify to x+1, got: {s}"
        );
    }

    #[test]
    fn full_simplify_already_simple() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let expr = a.add(&[x, two]); // x + 2 — nothing to cancel
        let result = full_simplify(&mut a, expr);
        assert_eq!(result, expr, "x+2 should stay x+2 through full_simplify");
    }

    #[test]
    fn smart_simplify_preserves_gcd_factor() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let six = a.int(6);
        let twelve = a.int(12);
        let six_x = a.mul(&[six, x]);
        let expr = a.add(&[six_x, twelve]); // 6x + 12
        let result = smart_simplify(&mut a, expr);
        // The result must be mathematically equivalent to 6x + 12
        // It must NOT be x + 2 (which drops the factor of 6)
        let two = a.int(2);
        let x_plus_2 = a.add(&[x, two]);
        assert_ne!(result, x_plus_2, "smart_simplify must not drop GCD factor");
    }
}
