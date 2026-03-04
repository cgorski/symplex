//! Dedicated trigonometric simplification.
//!
//! Goes beyond the pattern-based rules in `pattern.rs` by trying
//! exhaustive Pythagorean replacements and double-angle formulas.
//! Multiple strategies are attempted and the result with the fewest
//! operations (measured by [`count_ops`](crate::simplify_engine::count_ops))
//! is returned.

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode};
use crate::simplify_engine::count_ops;
use crate::walk;

use rustc_hash::FxHashMap;

/// Apply trigonometric simplification rules exhaustively.
///
/// Strategy: try several rewrite variants and keep the smallest.
///
/// 1. Apply existing pattern rules (sin²+cos²→1, etc.)
/// 2. Replace every sin²(x) with 1−cos²(x), then simplify
/// 3. Replace every cos²(x) with 1−sin²(x), then simplify
pub(crate) fn trigsimp(arena: &mut Arena, expr: ExprId) -> ExprId {
    let strategies = [
        apply_pattern_rules(arena, expr),
        replace_sin2_with_1_minus_cos2(arena, expr),
        replace_cos2_with_1_minus_sin2(arena, expr),
    ];

    strategies
        .into_iter()
        .min_by_key(|&id| count_ops(arena, id))
        .unwrap_or(expr)
}

// ── Strategy 1: pattern rules ──────────────────────────────────────────

/// Eval → pattern-rule simplification (the standard simplify path).
fn apply_pattern_rules(arena: &mut Arena, expr: ExprId) -> ExprId {
    let evaled = crate::eval::eval(arena, expr);
    let rules = crate::pattern::basic_rules(arena);
    let (result, _) = crate::pattern::apply_rules(arena, evaled, &rules);
    result
}

// ── Strategy 2: sin²(x) → 1 − cos²(x) ────────────────────────────────

/// Walk bottom-up and replace every `sin(x)^2` with `1 − cos(x)^2`,
/// then run eval + pattern simplification on the result.
fn replace_sin2_with_1_minus_cos2(arena: &mut Arena, expr: ExprId) -> ExprId {
    let replaced = walk_replace_trig_square(arena, expr, TrigKind::Sin);
    let evaled = crate::eval::eval(arena, replaced);
    let expanded = crate::expand::expand(arena, evaled);
    let evaled2 = crate::eval::eval(arena, expanded);
    let rules = crate::pattern::basic_rules(arena);
    let (result, _) = crate::pattern::apply_rules(arena, evaled2, &rules);
    result
}

// ── Strategy 3: cos²(x) → 1 − sin²(x) ────────────────────────────────

/// Walk bottom-up and replace every `cos(x)^2` with `1 − sin(x)^2`,
/// then run eval + pattern simplification on the result.
fn replace_cos2_with_1_minus_sin2(arena: &mut Arena, expr: ExprId) -> ExprId {
    let replaced = walk_replace_trig_square(arena, expr, TrigKind::Cos);
    let evaled = crate::eval::eval(arena, replaced);
    let expanded = crate::expand::expand(arena, evaled);
    let evaled2 = crate::eval::eval(arena, expanded);
    let rules = crate::pattern::basic_rules(arena);
    let (result, _) = crate::pattern::apply_rules(arena, evaled2, &rules);
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
        let rebuilt = crate::walk::rebuild_with_cache(arena, id, &cache);
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
}
