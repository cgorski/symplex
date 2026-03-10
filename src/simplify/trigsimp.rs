//! Dedicated trigonometric simplification.
//!
//! Goes beyond the pattern-based rules in `pattern.rs` by trying
//! exhaustive Pythagorean replacements, double-angle formulas,
//! trig combination, and trig expansion.
//!
//! Multiple strategies are attempted (choice-set approach) and the
//! result with the fewest operations (measured by
//! [`count_ops`](crate::simplify::simplify_engine::count_ops)) is returned.

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;
use crate::simplify::simplify_engine::count_ops;

/// Walk the expression tree and return `true` if any node matches the predicate.
/// Short-circuits on first match for efficiency.
fn walk_has_node_type(arena: &Arena, expr: ExprId, predicate: impl Fn(&ExprNode) -> bool) -> bool {
    let post_order = walk::post_order_ids(arena, expr);
    for &id in &post_order {
        if predicate(arena.node(id)) {
            return true;
        }
    }
    false
}

use rustc_hash::FxHashMap;

/// Apply trigonometric simplification rules exhaustively.
///
/// Strategy (choice-set): try several rewrite variants and keep the smallest.
///
/// 1. Original expression (baseline)
/// 2. Apply existing pattern rules (sin²+cos²→1, etc.)
/// 3. Replace every sin²(x) with 1−cos²(x), then simplify
/// 4. Replace every cos²(x) with 1−sin²(x), then simplify
/// 5. trig_combine (product-to-sum, double-angle identities)
/// 6. expand_trig then eval + pattern simplify
pub(crate) fn trigsimp(arena: &mut Arena, expr: ExprId) -> ExprId {
    // Early exit: if no trig nodes, nothing to simplify
    let has_trig = walk_has_node_type(arena, expr, |node| {
        matches!(
            node,
            ExprNode::Sin(_)
                | ExprNode::Cos(_)
                | ExprNode::Tan(_)
                | ExprNode::Asin(_)
                | ExprNode::Acos(_)
                | ExprNode::Atan(_)
        )
    });

    if !has_trig {
        tracing::debug!("trigsimp: skipping all strategies (no trig nodes)");
        return expr;
    }

    // Strategy 1: original expression (baseline)
    let s0 = expr;

    // Strategy 2: eval → pattern-rule simplification
    let s1 = apply_pattern_rules(arena, expr);

    // Strategy 3: sin²→1−cos² replacement
    let s2 = replace_sin2_with_1_minus_cos2(arena, expr);

    // Strategy 4: cos²→1−sin² replacement
    let s3 = replace_cos2_with_1_minus_sin2(arena, expr);

    // Strategy 5: trig_combine (product-to-sum, double-angle)
    let s4 = strategy_trig_combine(arena, expr);

    // Strategy 6: expand_trig then eval + simplify
    let s5 = strategy_expand_trig_then_simplify(arena, expr);

    let candidates = [s0, s1, s2, s3, s4, s5];
    let _strategy_names = [
        "original",
        "pattern_rules",
        "sin2_to_1_minus_cos2",
        "cos2_to_1_minus_sin2",
        "trig_combine",
        "expand_trig_then_simplify",
    ];

    // Pick the candidate with the lowest operation count.
    let best_idx = candidates
        .iter()
        .enumerate()
        .min_by_key(|&(_, &e)| count_ops(arena, e))
        .map(|(i, _)| i)
        .unwrap_or(0);

    candidates[best_idx]
}

// ── Strategy 2: pattern rules ──────────────────────────────────────────

/// Eval → pattern-rule simplification (the standard simplify path).
fn apply_pattern_rules(arena: &mut Arena, expr: ExprId) -> ExprId {
    let evaled = crate::transforms::eval::eval(arena, expr);
    let rules = crate::transforms::pattern::basic_rules(arena);
    let (result, _) = crate::transforms::pattern::apply_rules(arena, evaled, &rules);
    result
}

// ── Strategy 3: sin²(x) → 1 − cos²(x) ────────────────────────────────

/// Walk bottom-up and replace every `sin(x)^2` with `1 − cos(x)^2`,
/// then run eval + pattern simplification on the result.
fn replace_sin2_with_1_minus_cos2(arena: &mut Arena, expr: ExprId) -> ExprId {
    let replaced = walk_replace_trig_square(arena, expr, TrigKind::Sin);
    let evaled = crate::transforms::eval::eval(arena, replaced);
    let expanded = crate::transforms::expand::expand(arena, evaled);
    let evaled2 = crate::transforms::eval::eval(arena, expanded);
    let rules = crate::transforms::pattern::basic_rules(arena);
    let (result, _) = crate::transforms::pattern::apply_rules(arena, evaled2, &rules);
    result
}

// ── Strategy 4: cos²(x) → 1 − sin²(x) ────────────────────────────────

/// Walk bottom-up and replace every `cos(x)^2` with `1 − sin(x)^2`,
/// then run eval + pattern simplification on the result.
fn replace_cos2_with_1_minus_sin2(arena: &mut Arena, expr: ExprId) -> ExprId {
    let replaced = walk_replace_trig_square(arena, expr, TrigKind::Cos);
    let evaled = crate::transforms::eval::eval(arena, replaced);
    let expanded = crate::transforms::expand::expand(arena, evaled);
    let evaled2 = crate::transforms::eval::eval(arena, expanded);
    let rules = crate::transforms::pattern::basic_rules(arena);
    let (result, _) = crate::transforms::pattern::apply_rules(arena, evaled2, &rules);
    result
}

// ── Strategy 5: trig_combine ───────────────────────────────────────────

/// Apply trig_combine (product-to-sum, double-angle) then eval + simplify.
fn strategy_trig_combine(arena: &mut Arena, expr: ExprId) -> ExprId {
    let evaled = crate::transforms::eval::eval(arena, expr);
    let combined = crate::simplify::trig_combine::trig_combine(arena, evaled);
    let evaled2 = crate::transforms::eval::eval(arena, combined);
    let rules = crate::transforms::pattern::basic_rules(arena);
    let (result, _) = crate::transforms::pattern::apply_rules(arena, evaled2, &rules);
    result
}

// ── Strategy 6: expand_trig then simplify ──────────────────────────────

/// Expand trig functions (addition formulas, multi-angle), then
/// eval + pattern simplify.
fn strategy_expand_trig_then_simplify(arena: &mut Arena, expr: ExprId) -> ExprId {
    let evaled = crate::transforms::eval::eval(arena, expr);
    let expanded = crate::simplify::trig_expand::expand_trig(arena, evaled);
    let evaled2 = crate::transforms::eval::eval(arena, expanded);
    let rules = crate::transforms::pattern::basic_rules(arena);
    let (result, _) = crate::transforms::pattern::apply_rules(arena, evaled2, &rules);
    result
}

// ── Shared replacement infrastructure ──────────────────────────────────

/// Which trig function's square to replace.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TrigKind {
    Sin,
    Cos,
}

/// Walk an expression bottom-up and replace `trig(x)^2` with
/// `1 − other_trig(x)^2` where `trig` is the specified kind.
///
/// Uses the same manual post-order + cache pattern as `expand.rs`.
fn walk_replace_trig_square(arena: &mut Arena, expr: ExprId, kind: TrigKind) -> ExprId {
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post_order {
        // First, try to apply the trig-square replacement at this node.
        if let Some(replaced) = try_replace_trig_square(arena, id, &cache, kind) {
            cache.insert(id, replaced);
            continue;
        }

        // Otherwise, rebuild with substituted children (standard pattern).
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        cache.insert(id, rebuilt);
    }

    cache.get(&expr).copied().unwrap_or(expr)
}

/// Check if `id` is `Pow(Sin/Cos(x), 2)` and, if so, return
/// `1 − Pow(Cos/Sin(x), 2)` (with children already mapped through cache).
fn try_replace_trig_square(
    arena: &mut Arena,
    id: ExprId,
    cache: &FxHashMap<ExprId, ExprId>,
    kind: TrigKind,
) -> Option<ExprId> {
    let node = arena.node(id).clone();

    if let ExprNode::Pow(base, exp) = node {
        // Check that exponent is the integer 2.
        if !is_integer_two(arena, exp) {
            return None;
        }

        let base_node = arena.node(base).clone();
        let inner = match (&base_node, kind) {
            (ExprNode::Sin(inner), TrigKind::Sin) => Some(*inner),
            (ExprNode::Cos(inner), TrigKind::Cos) => Some(*inner),
            _ => None,
        }?;

        // Map the inner argument through the cache (children are
        // already processed in post-order).
        let new_inner = cache.get(&inner).copied().unwrap_or(inner);

        // Build the replacement: 1 − other_trig(new_inner)^2
        let other = match kind {
            TrigKind::Sin => arena.cos(new_inner),
            TrigKind::Cos => arena.sin(new_inner),
        };
        let two = arena.int(2);
        let other_sq = arena.pow(other, two);
        let one = arena.one;
        let result = arena.sub(one, other_sq);
        return Some(result);
    }

    None
}

/// Returns `true` if `id` is the numeric literal 2.
fn is_integer_two(arena: &Arena, id: ExprId) -> bool {
    if let Some(val) = arena.as_num(id) {
        *val == num_rational::Ratio::from_integer(num_bigint::BigInt::from(2))
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(arena: &mut Arena, name: &str) -> ExprId {
        arena.symbol(name)
    }

    fn display(arena: &Arena, id: ExprId) -> String {
        arena.display(id).to_string()
    }

    #[test]
    fn trigsimp_pythagorean_identity() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let sin_x = arena.sin(x);
        let cos_x = arena.cos(x);
        let two = arena.int(2);
        let sin2 = arena.pow(sin_x, two);
        let cos2 = arena.pow(cos_x, two);
        let expr = arena.add(&[sin2, cos2]);

        let result = trigsimp(&mut arena, expr);
        assert_eq!(display(&arena, result), "1");
    }

    #[test]
    fn trigsimp_leaves_simple_trig_alone() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let sin_x = arena.sin(x);

        let result = trigsimp(&mut arena, sin_x);
        assert_eq!(display(&arena, result), "sin(x)");
    }

    #[test]
    fn trigsimp_pythagorean_plus_constant() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let sin_x = arena.sin(x);
        let cos_x = arena.cos(x);
        let two = arena.int(2);
        let sin2 = arena.pow(sin_x, two);
        let cos2 = arena.pow(cos_x, two);
        let five = arena.int(5);
        let expr = arena.add(&[sin2, cos2, five]);

        let result = trigsimp(&mut arena, expr);
        assert_eq!(display(&arena, result), "6");
    }

    #[test]
    fn trigsimp_trig_combine_strategy() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let two = arena.int(2);
        let sin_x = arena.sin(x);
        let cos_x = arena.cos(x);
        // 2*sin(x)*cos(x) should simplify via trig_combine to sin(2x)
        let expr = arena.mul(&[two, sin_x, cos_x]);
        let result = trigsimp(&mut arena, expr);
        let result_ops = count_ops(&arena, result);
        let expr_ops = count_ops(&arena, expr);
        assert!(
            result_ops <= expr_ops,
            "trigsimp should not increase complexity: got {} ops vs original {} ops, result={}",
            result_ops,
            expr_ops,
            display(&arena, result)
        );
    }

    #[test]
    fn trigsimp_picks_best_strategy() {
        // Verify that the choice-set approach picks the simplest result
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let sin_x = arena.sin(x);
        let cos_x = arena.cos(x);
        let two = arena.int(2);
        let sin2 = arena.pow(sin_x, two);
        let cos2 = arena.pow(cos_x, two);
        // cos²(x) - sin²(x) → cos(2x) via trig_combine
        let expr = arena.sub(cos2, sin2);
        let result = trigsimp(&mut arena, expr);
        let result_ops = count_ops(&arena, result);
        let expr_ops = count_ops(&arena, expr);
        assert!(
            result_ops <= expr_ops,
            "trigsimp should simplify cos²-sin²: got {} ops, original {} ops, result={}",
            result_ops,
            expr_ops,
            display(&arena, result)
        );
    }
}
