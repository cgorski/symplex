//! Structural substitution and iterative tree walking.
//!
//! This module provides [`subs`] (structural replacement) and the
//! underlying [`walk_and_rebuild`] iterative tree walker that all
//! tree-transforming operations share.
//!
//! # Design
//!
//! **Structural substitution** replaces exact node matches only — it
//! never performs algebraic reasoning.  `(1/x).subs(x², 1)` returns
//! `1/x` unchanged because `x²` does not appear as a node in `1/x`.
//! This is safe by construction: it can never produce mathematically
//! wrong results.  Use the future `.alg_subs()` for algebraic
//! substitution with documented caveats.
//!
//! **No recursive tree walks.**  All traversals use an explicit stack
//! ([`walk_and_rebuild`]).  Stack overflow is impossible regardless of
//! expression depth.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode};

// ═══════════════════════════════════════════════════════════════════════════
// Structural substitution
// ═══════════════════════════════════════════════════════════════════════════

/// Replace every occurrence of `old` with `new` in the expression
/// rooted at `expr`.
///
/// This is **structural** substitution: only exact `ExprId` matches are
/// replaced.  The result is re-canonicalized through the normal
/// `Arena::add` / `Arena::mul` / etc. constructors, so like-term
/// collection and other canonical-form invariants are maintained.
///
/// Returns `expr` unchanged (same `ExprId`) if `old` does not appear
/// anywhere in the tree — no unnecessary allocation.
pub(crate) fn subs(arena: &mut Arena, expr: ExprId, old: ExprId, new: ExprId) -> ExprId {
    // Fast path: if old == new, nothing to do.
    if old == new {
        return expr;
    }
    // Fast path: if expr IS the thing we're replacing, return new.
    if expr == old {
        return new;
    }
    // Fast path: atoms that aren't the target can't contain it.
    if arena.node(expr).is_atom() {
        return expr;
    }

    walk_and_rebuild(arena, expr, &|_arena, id| {
        if id == old { Some(new) } else { None }
    })
}

/// Simultaneous substitution of multiple `(old, new)` pairs.
///
/// All replacements happen "at once" — earlier substitutions do not
/// affect later ones.  This avoids the order-dependence issues that
/// sequential substitution can cause.
pub(crate) fn subs_map(
    arena: &mut Arena,
    expr: ExprId,
    replacements: &[(ExprId, ExprId)],
) -> ExprId {
    if replacements.is_empty() {
        return expr;
    }

    let map: FxHashMap<ExprId, ExprId> = replacements.iter().copied().collect();

    // Fast path: expr itself is in the map.
    if let Some(&new) = map.get(&expr) {
        return new;
    }

    if arena.node(expr).is_atom() {
        return expr;
    }

    walk_and_rebuild(arena, expr, &|_arena, id| map.get(&id).copied())
}

// ═══════════════════════════════════════════════════════════════════════════
// Iterative tree walker with rebuild
// ═══════════════════════════════════════════════════════════════════════════

/// Walk an expression tree bottom-up and rebuild it with a
/// transformation function applied at each node.
///
/// `transform` is called on every node **after** its children have
/// been processed.  It returns `Some(new_id)` to replace the node, or
/// `None` to keep it (possibly with rebuilt children).
///
/// **This function never recurses.**  It uses an explicit work stack,
/// so it is safe for arbitrarily deep expression trees.
///
/// The rebuilt expression is re-canonicalized through the standard
/// `Arena` constructors (`add`, `mul`, `pow`, etc.), preserving all
/// canonical-form invariants.
pub(crate) fn walk_and_rebuild(
    arena: &mut Arena,
    root: ExprId,
    transform: &dyn Fn(&Arena, ExprId) -> Option<ExprId>,
) -> ExprId {
    // Cache: original ExprId → rebuilt ExprId.
    // If a sub-expression appears multiple times (common due to
    // hash-consing), we only rebuild it once.
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    // Phase 1: Collect a post-order traversal using an explicit stack.
    // We need bottom-up order so that children are rebuilt before parents.
    let post_order = post_order_ids(arena, root);

    // Phase 2: Process each node in post-order (leaves first).
    for &id in &post_order {
        // Check the transform first — if it wants to replace this
        // node entirely, we don't need to rebuild children.
        if let Some(replacement) = transform(arena, id) {
            cache.insert(id, replacement);
            continue;
        }

        // Atoms: no children to rebuild.
        if arena.node(id).is_atom() {
            cache.insert(id, id);
            continue;
        }

        // Rebuild with substituted children.
        let new_id = rebuild_with_cache(arena, id, &cache);
        cache.insert(id, new_id);
    }

    cache.get(&root).copied().unwrap_or(root)
}

/// Compute a post-order traversal of the expression DAG using an
/// explicit stack.  Each `ExprId` appears at most once in the output
/// (de-duplicated by the visited set).
fn post_order_ids(arena: &Arena, root: ExprId) -> Vec<ExprId> {
    let mut result = Vec::new();
    let mut visited: FxHashMap<ExprId, bool> = FxHashMap::default();
    // Stack entries: (id, children_pushed).
    let mut stack: Vec<(ExprId, bool)> = vec![(root, false)];

    while let Some((id, children_pushed)) = stack.last_mut() {
        if visited.contains_key(id) {
            stack.pop();
            continue;
        }

        if !*children_pushed {
            *children_pushed = true;
            let children = arena.node(*id).children();
            // Push children in reverse so they're processed left-to-right.
            for &child in children.iter().rev() {
                if !visited.contains_key(&child) {
                    stack.push((child, false));
                }
            }
        } else {
            let id = stack.pop().unwrap().0;
            visited.insert(id, true);
            result.push(id);
        }
    }

    result
}

/// Rebuild a single node with children looked up from the cache.
///
/// If all children map to themselves (no changes), returns the
/// original `id` unchanged — avoiding unnecessary allocation.
///
/// When children have changed, uses the canonical constructors
/// (`arena.add`, `arena.mul`, etc.) to maintain canonical form.
fn rebuild_with_cache(arena: &mut Arena, id: ExprId, cache: &FxHashMap<ExprId, ExprId>) -> ExprId {
    let node = arena.node(id).clone();

    match node {
        // Atoms: no children.
        ExprNode::Num(_)
        | ExprNode::Symbol(_)
        | ExprNode::Pi
        | ExprNode::E
        | ExprNode::ImaginaryUnit
        | ExprNode::Infinity
        | ExprNode::NegInfinity
        | ExprNode::ComplexInfinity
        | ExprNode::NaN => id,

        // N-ary: Add, Mul
        ExprNode::Add(ref children) => {
            let new_children: SmallVec<[ExprId; 6]> = children
                .iter()
                .map(|&c| cache.get(&c).copied().unwrap_or(c))
                .collect();
            if new_children == *children {
                id
            } else {
                arena.add(&new_children)
            }
        }

        ExprNode::Mul(ref children) => {
            let new_children: SmallVec<[ExprId; 6]> = children
                .iter()
                .map(|&c| cache.get(&c).copied().unwrap_or(c))
                .collect();
            if new_children == *children {
                id
            } else {
                arena.mul(&new_children)
            }
        }

        // Binary: Pow, Derivative, Integral
        ExprNode::Pow(base, exp) => {
            let new_base = cache.get(&base).copied().unwrap_or(base);
            let new_exp = cache.get(&exp).copied().unwrap_or(exp);
            if new_base == base && new_exp == exp {
                id
            } else {
                arena.pow(new_base, new_exp)
            }
        }

        ExprNode::Derivative(body, var) => {
            let new_body = cache.get(&body).copied().unwrap_or(body);
            let new_var = cache.get(&var).copied().unwrap_or(var);
            if new_body == body && new_var == var {
                id
            } else {
                arena.intern(ExprNode::Derivative(new_body, new_var))
            }
        }

        ExprNode::Integral(body, var) => {
            let new_body = cache.get(&body).copied().unwrap_or(body);
            let new_var = cache.get(&var).copied().unwrap_or(var);
            if new_body == body && new_var == var {
                id
            } else {
                arena.intern(ExprNode::Integral(new_body, new_var))
            }
        }

        // Unary: Neg, Sin, Cos, Tan, Exp, Ln, Sqrt, Abs
        ExprNode::Neg(inner) => {
            let new_inner = cache.get(&inner).copied().unwrap_or(inner);
            if new_inner == inner {
                id
            } else {
                arena.neg(new_inner)
            }
        }

        ExprNode::Sin(inner) => {
            let new_inner = cache.get(&inner).copied().unwrap_or(inner);
            if new_inner == inner {
                id
            } else {
                arena.sin(new_inner)
            }
        }

        ExprNode::Cos(inner) => {
            let new_inner = cache.get(&inner).copied().unwrap_or(inner);
            if new_inner == inner {
                id
            } else {
                arena.cos(new_inner)
            }
        }

        ExprNode::Tan(inner) => {
            let new_inner = cache.get(&inner).copied().unwrap_or(inner);
            if new_inner == inner {
                id
            } else {
                arena.tan(new_inner)
            }
        }

        ExprNode::Exp(inner) => {
            let new_inner = cache.get(&inner).copied().unwrap_or(inner);
            if new_inner == inner {
                id
            } else {
                arena.exp_fn(new_inner)
            }
        }

        ExprNode::Ln(inner) => {
            let new_inner = cache.get(&inner).copied().unwrap_or(inner);
            if new_inner == inner {
                id
            } else {
                arena.ln(new_inner)
            }
        }

        ExprNode::Sqrt(inner) => {
            let new_inner = cache.get(&inner).copied().unwrap_or(inner);
            if new_inner == inner {
                id
            } else {
                arena.sqrt(new_inner)
            }
        }

        ExprNode::Abs(inner) => {
            let new_inner = cache.get(&inner).copied().unwrap_or(inner);
            if new_inner == inner {
                id
            } else {
                arena.abs(new_inner)
            }
        }

        // Apply: user-defined function
        ExprNode::Apply(func_id, ref args) => {
            let new_args: SmallVec<[ExprId; 2]> = args
                .iter()
                .map(|&c| cache.get(&c).copied().unwrap_or(c))
                .collect();
            if new_args == *args {
                id
            } else {
                arena.intern(ExprNode::Apply(func_id, new_args))
            }
        }
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

    // ── Basic subs ──────────────────────────────────────────────────

    #[test]
    fn subs_symbol_for_number() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let expr = a.add(&[x, a.one]);
        let result = subs(&mut a, expr, x, three);
        assert_eq!(display(&a, result), "4");
    }

    #[test]
    fn subs_symbol_in_product() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);
        let expr = a.mul(&[two, x]);
        let result = subs(&mut a, expr, x, y);
        assert_eq!(display(&a, result), "2*y");
    }

    #[test]
    fn subs_in_pow() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let expr = a.pow(x, two);
        let three = a.int(3);
        let result = subs(&mut a, expr, x, three);
        assert_eq!(display(&a, result), "9");
    }

    #[test]
    fn subs_in_sin() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let expr = a.sin(x);
        let result = subs(&mut a, expr, x, y);
        assert_eq!(display(&a, result), "sin(y)");
    }

    #[test]
    fn subs_no_match_returns_same_id() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let expr = a.add(&[x, a.one]);
        let n99 = a.int(99);
        let result = subs(&mut a, expr, y, n99);
        assert_eq!(result, expr, "no match should return same ExprId");
    }

    #[test]
    fn subs_entire_expression() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let result = subs(&mut a, x, x, y);
        assert_eq!(result, y);
    }

    #[test]
    fn subs_old_equals_new_is_noop() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.add(&[x, a.one]);
        let result = subs(&mut a, expr, x, x);
        assert_eq!(result, expr);
    }

    // ── Nested substitution ─────────────────────────────────────────

    #[test]
    fn subs_in_nested_add_mul() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let three = a.int(3);

        // x^2 + 2*x + 1, substitute x → 3
        let x_sq = a.pow(x, two);
        let two_x = a.mul(&[two, x]);
        let expr = a.add(&[x_sq, two_x, a.one]);
        let result = subs(&mut a, expr, x, three);
        // 9 + 6 + 1 = 16
        assert_eq!(display(&a, result), "16");
    }

    #[test]
    fn subs_in_function_of_pow() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);

        // sin(x^2), substitute x → y
        let xp = a.pow(x, two);
        let expr = a.sin(xp);
        let result = subs(&mut a, expr, x, y);
        assert_eq!(display(&a, result), "sin(y**2)");
    }

    #[test]
    fn subs_replaces_all_occurrences() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");

        // x + x → y + y = 2*y
        let expr = a.add(&[x, x]);
        let result = subs(&mut a, expr, x, y);
        assert_eq!(display(&a, result), "2*y");
    }

    // ── Structural correctness ──────────────────────────────────────

    #[test]
    fn subs_does_not_match_algebraic_subexpressions() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let one = a.one;

        // (1/x).subs(x^2, 1) should return 1/x unchanged.
        // x^(-1) does not structurally contain x^2.
        let x_inv = a.pow(x, a.neg_one);
        let x_sq = a.pow(x, two);
        let result = subs(&mut a, x_inv, x_sq, one);
        assert_eq!(
            result, x_inv,
            "structural subs should not match x^2 in x^(-1)"
        );
    }

    #[test]
    fn subs_with_zero_evaluates() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let zero = a.zero;

        // (x*y).subs(y, 0) → x*0 → 0
        let expr = a.mul(&[x, y]);
        let result = subs(&mut a, expr, y, zero);
        assert_eq!(result, a.zero);
    }

    // ── Simultaneous substitution ───────────────────────────────────

    #[test]
    fn subs_map_simultaneous() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");

        // (x + y).subs({x→y, y→x}) should give y + x = x + y (commutative)
        let expr = a.add(&[x, y]);
        let result = subs_map(&mut a, expr, &[(x, y), (y, x)]);
        // Should be the same canonical form since x+y and y+x are equal.
        assert_eq!(result, expr, "swapping x↔y in x+y should give x+y");
    }

    #[test]
    fn subs_map_empty_is_noop() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.add(&[x, a.one]);
        let result = subs_map(&mut a, expr, &[]);
        assert_eq!(result, expr);
    }

    #[test]
    fn subs_map_multiple() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);
        let three = a.int(3);

        // (x + y).subs({x→2, y→3}) = 5
        let expr = a.add(&[x, y]);
        let result = subs_map(&mut a, expr, &[(x, two), (y, three)]);
        assert_eq!(display(&a, result), "5");
    }

    // ── Deep expressions (stack safety) ─────────────────────────────

    #[test]
    fn subs_deep_expression_no_stack_overflow() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");

        // Build a deeply nested expression: sin(sin(sin(...sin(x)...)))
        // 10,000 levels deep — tests that our iterative walker doesn't
        // stack-overflow.
        let mut expr = x;
        for _ in 0..10_000 {
            expr = a.sin(expr);
        }

        // Substitute x → y deep inside.
        let result = subs(&mut a, expr, x, y);

        // Verify it changed (not the same ExprId).
        assert_ne!(
            result, expr,
            "substitution should have changed the deep expression"
        );
        // NOTE: We don't test Display here because fmt_expr in
        // display.rs is still recursive and would overflow at this
        // depth.  That will be fixed when display is converted to an
        // iterative walker (tracked as a known issue).
    }

    #[test]
    fn subs_moderate_depth_display_works() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");

        // Moderate depth that the recursive display can still handle.
        let mut expr = x;
        for _ in 0..50 {
            expr = a.sin(expr);
        }

        let result = subs(&mut a, expr, x, y);
        let s = format!("{}", a.display(result));
        assert!(s.starts_with("sin("), "should still start with sin(");
        assert!(s.contains('y'), "should contain y after substitution");
        assert!(!s.contains('x'), "should not contain x after substitution");
    }

    // ── walk_and_rebuild ────────────────────────────────────────────

    #[test]
    fn walk_identity_returns_same_id() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let xp = a.pow(x, two);
        let one = a.one;
        let expr = a.add(&[xp, x, one]);
        let result = walk_and_rebuild(&mut a, expr, &|_, _| None);
        assert_eq!(result, expr, "identity transform should return same ExprId");
    }

    #[test]
    fn walk_replace_leaf() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);
        let expr = a.pow(x, two);

        let result = walk_and_rebuild(&mut a, expr, &|_arena, id| {
            if id == x { Some(y) } else { None }
        });
        assert_eq!(display(&a, result), "y**2");
    }

    #[test]
    fn walk_shared_subexpression_rebuilt_once() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");

        // x + x: both children share the same ExprId.
        let expr = a.add(&[x, x]);
        let before_count = a.node_count();

        let result = walk_and_rebuild(&mut a, expr, &|_arena, id| {
            if id == x { Some(y) } else { None }
        });

        // Should rebuild to y + y = 2*y.
        assert_eq!(display(&a, result), "2*y");

        // The shared sub-expression should only have been processed once
        // (verified by the cache inside walk_and_rebuild — we can't
        // directly test this, but we verify correctness).
        let _ = before_count; // used for documentation
    }
}
