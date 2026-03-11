//! Simplification engine — try multiple strategies, keep shortest.
//!
//! This module provides [`count_ops`] for measuring expression complexity,
//! [`smart_simplify`] which tries multiple simplification strategies
//! and returns the result with the lowest operation count, and
//! [`full_simplify`] / [`full_simplify_trace`] which iterate
//! eval → expand → simplify → **cancel** until convergence.

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;
use num_bigint::BigInt;
use num_traits::Signed;

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

// ── Type-based strategy gating ─────────────────────────────────────────

/// Flags indicating which node types appear in an expression.
/// Used to skip irrelevant simplification strategies.
struct ExprFlags {
    has_trig: bool,
    has_hyp: bool,
    has_exp_ln: bool,
    has_neg_pow: bool,
    has_add: bool,
    has_apply: bool,
    is_atom: bool,
    node_count: usize,
    /// Expression contains `Abs` nodes — gates refine strategy.
    has_abs: bool,
    /// Expression contains `Sign` nodes — gates refine strategy.
    has_sign: bool,
    /// Expression contains `Floor` or `Ceiling` nodes — gates refine strategy.
    has_floor_ceil: bool,
    /// Expression contains `Pow` nodes (general, not just negative exponent).
    has_pow: bool,
    /// Expression contains `Factorial` or `Binomial` nodes — gates combsimp strategy.
    has_factorial: bool,
    /// Expression contains `Gamma` or `LogGamma` nodes — gates combsimp strategy.
    has_gamma: bool,
    /// Expression contains `Pow` with exponent 1/2 (square root) — gates radsimp strategy.
    has_sqrt: bool,
}

/// Compute expression flags in a single tree walk.
fn compute_flags(arena: &Arena, expr: ExprId) -> ExprFlags {
    let post_order = crate::base::walk::post_order_ids(arena, expr);
    let mut flags = ExprFlags {
        has_trig: false,
        has_hyp: false,
        has_exp_ln: false,
        has_neg_pow: false,
        has_add: false,
        has_apply: false,
        is_atom: post_order.len() == 1 && arena.node(expr).is_atom(),
        node_count: post_order.len(),
        has_abs: false,
        has_sign: false,
        has_floor_ceil: false,
        has_pow: false,
        has_factorial: false,
        has_gamma: false,
        has_sqrt: false,
    };

    for &id in &post_order {
        match arena.node(id) {
            ExprNode::Sin(_)
            | ExprNode::Cos(_)
            | ExprNode::Tan(_)
            | ExprNode::Asin(_)
            | ExprNode::Acos(_)
            | ExprNode::Atan(_) => {
                flags.has_trig = true;
            }
            ExprNode::Sinh(_)
            | ExprNode::Cosh(_)
            | ExprNode::Tanh(_)
            | ExprNode::Asinh(_)
            | ExprNode::Acosh(_)
            | ExprNode::Atanh(_) => {
                flags.has_hyp = true;
            }
            ExprNode::Exp(_) | ExprNode::Ln(_) => {
                flags.has_exp_ln = true;
            }
            ExprNode::Pow(_, exp) => {
                flags.has_pow = true;
                if let Some(r) = arena.as_num(*exp) {
                    if r.is_negative() {
                        flags.has_neg_pow = true;
                    }
                    // Check for sqrt: exponent == 1/2
                    if *r.numer() == BigInt::from(1) && *r.denom() == BigInt::from(2) {
                        flags.has_sqrt = true;
                    }
                }
            }
            ExprNode::Factorial(_) | ExprNode::Binomial(_, _) => {
                flags.has_factorial = true;
            }
            ExprNode::Gamma(_) | ExprNode::LogGamma(_) => {
                flags.has_gamma = true;
            }
            ExprNode::Add(_) => flags.has_add = true,
            ExprNode::Apply(_, _) => flags.has_apply = true,
            ExprNode::Abs(_) => flags.has_abs = true,
            ExprNode::Sign(_) => flags.has_sign = true,
            ExprNode::Floor(_) | ExprNode::Ceiling(_) => flags.has_floor_ceil = true,
            _ => {}
        }
    }

    flags
}

/// Helper: update `best` / `best_ops` if `candidate` has a lower op-count.
fn update_best(arena: &Arena, best: &mut ExprId, best_ops: &mut usize, candidate: ExprId) {
    let ops = count_ops(arena, candidate);
    if ops < *best_ops {
        *best = candidate;
        *best_ops = ops;
    }
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
/// 8. eval → refine (assumption-aware, e.g. abs(x)→x when x≥0)
///
/// The result with the lowest `count_ops` is returned.
/// If the best result is more than 1.7× the complexity of the original,
/// the original is returned (to prevent "simplification" that makes things worse).
pub(crate) fn smart_simplify(arena: &mut Arena, expr: ExprId) -> ExprId {
    let flags = compute_flags(arena, expr);

    // Early exit for atoms — nothing to simplify
    if flags.is_atom {
        tracing::debug!("smart_simplify: early exit for atom expression");
        return expr;
    }

    let original_ops = count_ops(arena, expr);
    let mut best = expr;
    let mut best_ops = original_ops;

    // Strategy 1: eval only (always try — cheap)
    let s1 = crate::transforms::eval::eval(arena, expr);
    tracing::trace!(
        strategy = "eval",
        ops = count_ops(arena, s1),
        "strategy evaluated"
    );
    update_best(arena, &mut best, &mut best_ops, s1);

    // Strategy 2: eval → pattern rules (only if trig/exp/hyp present)
    if flags.has_trig || flags.has_exp_ln || flags.has_hyp {
        let rules = crate::transforms::pattern::basic_rules(arena);
        let s2_eval = crate::transforms::eval::eval(arena, expr);
        let (s2, _) = crate::transforms::pattern::apply_rules(arena, s2_eval, &rules);
        tracing::trace!(
            strategy = "eval+rules",
            ops = count_ops(arena, s2),
            "strategy evaluated"
        );
        update_best(arena, &mut best, &mut best_ops, s2);
    } else {
        tracing::debug!("smart_simplify: skipping pattern rules (no trig/exp/hyp nodes)");
    }

    // Strategy 3: eval → expand → simplify (always try — helps with polynomial algebra)
    {
        let rules = crate::transforms::pattern::basic_rules(arena);
        let s3_eval = crate::transforms::eval::eval(arena, expr);
        let s3_expand = crate::transforms::expand::expand(arena, s3_eval);
        let s3_eval2 = crate::transforms::eval::eval(arena, s3_expand);
        let (s3, _) = crate::transforms::pattern::apply_rules(arena, s3_eval2, &rules);
        tracing::trace!(
            strategy = "eval+expand+rules",
            ops = count_ops(arena, s3),
            "strategy evaluated"
        );
        update_best(arena, &mut best, &mut best_ops, s3);
    }

    // Strategy 4: eval → factor_terms (symbolic) → simplify (only if has Add)
    if flags.has_add {
        let rules = crate::transforms::pattern::basic_rules(arena);
        let s4_eval = crate::transforms::eval::eval(arena, expr);
        let (gcd_id, s4_inner) =
            crate::simplify::factor_terms::symbolic_factor_terms_pair(arena, s4_eval);
        let (s4_simplified, _) = crate::transforms::pattern::apply_rules(arena, s4_inner, &rules);
        let s4 = if gcd_id == arena.one {
            s4_simplified
        } else {
            arena.mul(&[gcd_id, s4_simplified])
        };
        tracing::trace!(
            strategy = "eval+factor+rules",
            ops = count_ops(arena, s4),
            "strategy evaluated"
        );
        update_best(arena, &mut best, &mut best_ops, s4);
    } else {
        tracing::debug!("smart_simplify: skipping factor_terms (no Add nodes)");
    }

    // Strategy 5: eval → expand_trig → simplify (only if has trig)
    if flags.has_trig {
        let rules = crate::transforms::pattern::basic_rules(arena);
        let s5_eval = crate::transforms::eval::eval(arena, expr);
        let s5_trig = crate::simplify::trig_expand::expand_trig(arena, s5_eval);
        let s5_eval2 = crate::transforms::eval::eval(arena, s5_trig);
        let (s5, _) = crate::transforms::pattern::apply_rules(arena, s5_eval2, &rules);
        tracing::trace!(
            strategy = "eval+trig_expand+rules",
            ops = count_ops(arena, s5),
            "strategy evaluated"
        );
        update_best(arena, &mut best, &mut best_ops, s5);
    } else {
        tracing::debug!("smart_simplify: skipping trig_expand (no trig nodes)");
    }

    // Strategy 6: eval → logcombine → simplify (only if has Ln nodes)
    if flags.has_exp_ln {
        let rules = crate::transforms::pattern::basic_rules(arena);
        let s6_eval = crate::transforms::eval::eval(arena, expr);
        let s6_log = crate::simplify::log_combine::log_combine(arena, s6_eval);
        let (s6, _) = crate::transforms::pattern::apply_rules(arena, s6_log, &rules);
        tracing::trace!(
            strategy = "eval+logcombine+rules",
            ops = count_ops(arena, s6),
            "strategy evaluated"
        );
        update_best(arena, &mut best, &mut best_ops, s6);
    } else {
        tracing::debug!("smart_simplify: skipping logcombine (no exp/ln nodes)");
    }

    // Strategy 7: eval → cancel with free symbols → simplify (only if has neg powers / fractions)
    if flags.has_neg_pow {
        let rules = crate::transforms::pattern::basic_rules(arena);
        let evaled = crate::transforms::eval::eval(arena, expr);
        let free = crate::base::walk::free_symbols(arena, evaled);
        let mut cancel_best = evaled;
        let mut cancel_best_ops = count_ops(arena, evaled);
        for &sym in &free {
            let cancelled = crate::poly::polybridge::cancel(arena, evaled, sym);
            let cancelled_eval = crate::transforms::eval::eval(arena, cancelled);
            let (cancelled_simp, _) =
                crate::transforms::pattern::apply_rules(arena, cancelled_eval, &rules);
            let ops = count_ops(arena, cancelled_simp);
            if ops < cancel_best_ops {
                cancel_best = cancelled_simp;
                cancel_best_ops = ops;
            }
        }
        tracing::trace!(
            strategy = "eval+cancel+rules",
            ops = cancel_best_ops,
            "strategy evaluated"
        );
        update_best(arena, &mut best, &mut best_ops, cancel_best);
    } else {
        tracing::debug!("smart_simplify: skipping cancel (no negative-power nodes)");
    }

    // Strategy 8: eval → refine (assumption-aware) (only if refinable nodes present)
    if flags.has_abs || flags.has_sign || flags.has_floor_ceil || flags.has_pow {
        let s8_eval = crate::transforms::eval::eval(arena, expr);
        // refine needs &mut AssumptionCache — but smart_simplify only has &mut Arena.
        // We build a temporary cache for this pass.  When called through the public
        // Ex::simplify() path the real cache is used; here we create a fresh one
        // that still reads symbol-level assumptions from the arena.
        let mut temp_assumptions = crate::base::assumptions::AssumptionCache::new();
        let s8 = crate::simplify::refine::refine_full(arena, &mut temp_assumptions, s8_eval);
        tracing::trace!(
            strategy = "eval+refine",
            ops = count_ops(arena, s8),
            "strategy evaluated"
        );
        update_best(arena, &mut best, &mut best_ops, s8);
    } else {
        tracing::debug!("smart_simplify: skipping refine (no abs/sign/floor/ceil/pow nodes)");
    }

    // Strategy 9: eval → powsimp → eval (only if has Pow nodes)
    if flags.has_pow {
        let s9_eval = crate::transforms::eval::eval(arena, expr);
        let s9_pow = crate::simplify::powsimp::powsimp(arena, s9_eval);
        let s9 = crate::transforms::eval::eval(arena, s9_pow);
        tracing::trace!(
            strategy = "eval+powsimp",
            ops = count_ops(arena, s9),
            "strategy evaluated"
        );
        update_best(arena, &mut best, &mut best_ops, s9);
    } else {
        tracing::debug!("smart_simplify: skipping powsimp (no Pow nodes)");
    }

    // Strategy 10: eval → combsimp → eval (only if has factorial/gamma)
    if flags.has_factorial || flags.has_gamma {
        let s10_eval = crate::transforms::eval::eval(arena, expr);
        let s10_comb = crate::simplify::combsimp::combsimp(arena, s10_eval);
        let s10 = crate::transforms::eval::eval(arena, s10_comb);
        tracing::trace!(
            strategy = "eval+combsimp",
            ops = count_ops(arena, s10),
            "strategy evaluated"
        );
        update_best(arena, &mut best, &mut best_ops, s10);
    } else {
        tracing::debug!("smart_simplify: skipping combsimp (no factorial/gamma nodes)");
    }

    // Strategy 11: eval → radsimp (rationalize denominator) (only if has sqrt in denominator)
    if flags.has_sqrt && flags.has_neg_pow {
        let s11_eval = crate::transforms::eval::eval(arena, expr);
        let s11_rad = crate::simplify::radsimp::rationalize_denom(arena, s11_eval);
        let s11 = crate::transforms::eval::eval(arena, s11_rad);
        tracing::trace!(
            strategy = "eval+radsimp",
            ops = count_ops(arena, s11),
            "strategy evaluated"
        );
        update_best(arena, &mut best, &mut best_ops, s11);
    } else {
        tracing::debug!("smart_simplify: skipping radsimp (no sqrt+neg_pow nodes)");
    }

    // Strategy 12: eval → powdenest → powsimp_base → eval (only if has Pow)
    //
    // Full radical simplification chain:
    //   powdenest:    (a·b)^n → a^n · b^n  (e.g., (½·√5)² → ¼·5 = 5/4)
    //   powsimp_base: a^e · b^e → (a·b)^e  (e.g., √2·√3 → √6)
    //
    // Both are applied in sequence because they are complementary:
    // powdenest breaks apart compound bases, powsimp_base recombines
    // separate bases.  The eval passes before and after handle
    // canonicalization of the results (e.g., 5/4 - 5/4 → 0).
    if flags.has_pow {
        let s12_eval = crate::transforms::eval::eval(arena, expr);
        let s12_denest = crate::simplify::powsimp::powdenest(arena, s12_eval);
        let s12_base = crate::simplify::powsimp::powsimp_base(arena, s12_denest);
        let s12 = crate::transforms::eval::eval(arena, s12_base);
        tracing::trace!(
            strategy = "eval+powdenest+powsimp_base+eval",
            ops = count_ops(arena, s12),
            changed = (s12 != s12_eval),
            "strategy evaluated"
        );
        update_best(arena, &mut best, &mut best_ops, s12);

        // Also try the individual strategies alone (one may help where
        // the other doesn't, and the combined chain may increase op count).
        let s12b_denest = crate::simplify::powsimp::powdenest(arena, s12_eval);
        let s12b = crate::transforms::eval::eval(arena, s12b_denest);
        tracing::trace!(
            strategy = "eval+powdenest",
            ops = count_ops(arena, s12b),
            changed = (s12b != s12_eval),
            "strategy evaluated"
        );
        update_best(arena, &mut best, &mut best_ops, s12b);

        let s12c_base = crate::simplify::powsimp::powsimp_base(arena, s12_eval);
        let s12c = crate::transforms::eval::eval(arena, s12c_base);
        tracing::trace!(
            strategy = "eval+powsimp_base",
            ops = count_ops(arena, s12c),
            changed = (s12c != s12_eval),
            "strategy evaluated"
        );
        update_best(arena, &mut best, &mut best_ops, s12c);
    } else {
        tracing::debug!("smart_simplify: skipping powdenest/powsimp_base (no Pow nodes)");
    }

    tracing::debug!(
        ops_original = original_ops,
        ops_result = best_ops,
        node_count = flags.node_count,
        has_trig = flags.has_trig,
        has_hyp = flags.has_hyp,
        has_exp_ln = flags.has_exp_ln,
        has_neg_pow = flags.has_neg_pow,
        has_add = flags.has_add,
        has_apply = flags.has_apply,
        has_factorial = flags.has_factorial,
        has_gamma = flags.has_gamma,
        has_sqrt = flags.has_sqrt,
        "smart_simplify selected best"
    );

    // Guard: don't return something much more complex than original
    if original_ops > 0 && best_ops as f64 > 1.7 * original_ops as f64 {
        return expr;
    }

    best
}

/// Fully simplify an expression by iterating eval → expand → simplify → cancel.
///
/// This is the engine-level counterpart of `Expr::full_simplify`.  Unlike
/// the original `full_simplify_trace` loop in `expr.rs`, this version
/// includes a polynomial-cancellation step (via [`crate::poly::polybridge::cancel`])
/// after every simplify pass, so rational expressions like `(x²-4)/(x-2)`
/// are reduced to `x+2`.
#[allow(dead_code)]
pub(crate) fn full_simplify(arena: &mut Arena, expr: ExprId) -> ExprId {
    full_simplify_trace(arena, expr).0
}

/// Like [`full_simplify`] but also returns the accumulated rewrite-rule
/// trace (one [`crate::transforms::pattern::Step`] per rule firing).
pub(crate) fn full_simplify_trace(
    arena: &mut Arena,
    expr: ExprId,
) -> (ExprId, Vec<crate::transforms::pattern::Step>) {
    const MAX_ITERATIONS: usize = 10;
    let rules = crate::transforms::pattern::basic_rules(arena);
    let mut current = expr;
    let mut all_steps: Vec<crate::transforms::pattern::Step> = Vec::new();

    for i in 0..MAX_ITERATIONS {
        let evaled = crate::transforms::eval::eval(arena, current);

        // ── try cancel BEFORE expand (preserves rational structure) ──
        let free = crate::base::walk::free_symbols(arena, evaled);
        let mut cancelled = evaled;
        for &sym in &free {
            cancelled = crate::poly::polybridge::cancel(arena, cancelled, sym);
        }
        let cancelled_eval = crate::transforms::eval::eval(arena, cancelled);
        let (cancelled_simp, cancel_steps) =
            crate::transforms::pattern::apply_rules(arena, cancelled_eval, &rules);
        all_steps.extend(cancel_steps);

        // ── also try the classic path: expand → simplify ──
        let expanded = crate::transforms::expand::expand(arena, evaled);
        let expanded_eval = crate::transforms::eval::eval(arena, expanded);
        let (expanded_simp, expand_steps) =
            crate::transforms::pattern::apply_rules(arena, expanded_eval, &rules);
        all_steps.extend(expand_steps);

        // ── also try radical simplification: powdenest → powsimp_base → eval ──
        let radical_denest = crate::simplify::powsimp::powdenest(arena, evaled);
        let radical_base = crate::simplify::powsimp::powsimp_base(arena, radical_denest);
        let radical_simp = crate::transforms::eval::eval(arena, radical_base);

        // Pick whichever result has fewest operations.
        let mut best = cancelled_simp;
        let mut best_ops = count_ops(arena, best);
        let expanded_ops = count_ops(arena, expanded_simp);
        if expanded_ops < best_ops {
            best = expanded_simp;
            best_ops = expanded_ops;
        }
        let radical_ops = count_ops(arena, radical_simp);
        if radical_ops < best_ops {
            tracing::trace!(
                iteration = i,
                radical_ops,
                prev_best_ops = best_ops,
                "full_simplify: radical simplification produced better result"
            );
            best = radical_simp;
        }

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
    use crate::base::arena::Arena;

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
        let eval_expr = crate::transforms::eval::eval(&mut a, expr);
        let (gcd, inner) = crate::simplify::factor_terms::factor_terms_pair(&mut a, eval_expr);
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

    // ── Tests for compute_flags ────────────────────────────────────────

    #[test]
    fn compute_flags_atom() {
        let a = Arena::new();
        let flags = compute_flags(&a, a.zero);
        assert!(flags.is_atom);
        assert!(!flags.has_trig);
        assert!(!flags.has_hyp);
        assert!(!flags.has_exp_ln);
        assert!(!flags.has_neg_pow);
        assert!(!flags.has_add);
        assert!(!flags.has_apply);
        assert_eq!(flags.node_count, 1);
    }

    #[test]
    fn compute_flags_trig() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let flags = compute_flags(&a, sin_x);
        assert!(flags.has_trig);
        assert!(!flags.has_hyp);
        assert!(!flags.has_exp_ln);
        assert!(!flags.is_atom);
    }

    #[test]
    fn compute_flags_exp_ln() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let ln_x = a.ln(x);
        let exp_ln_x = a.exp(ln_x);
        let flags = compute_flags(&a, exp_ln_x);
        assert!(flags.has_exp_ln);
        assert!(!flags.has_trig);
    }

    #[test]
    fn compute_flags_neg_pow() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let neg_one = a.int(-1);
        let recip = a.pow(x, neg_one);
        let flags = compute_flags(&a, recip);
        assert!(flags.has_neg_pow);
        assert!(!flags.has_trig);
    }

    #[test]
    fn compute_flags_add() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let sum = a.add(&[x, y]);
        let flags = compute_flags(&a, sum);
        assert!(flags.has_add);
        assert!(!flags.has_trig);
    }

    #[test]
    fn compute_flags_hyp() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sinh_x = a.sinh(x);
        let flags = compute_flags(&a, sinh_x);
        assert!(flags.has_hyp);
        assert!(!flags.has_trig);
    }
}
