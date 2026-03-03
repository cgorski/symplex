//! Iterative tree traversal infrastructure.
//!
//! This module provides the shared building blocks for all tree-walking
//! operations in symplex: substitution, differentiation, pattern
//! matching, simplification, and any future tree transformations.
//!
//! **No recursion.**  All traversals use explicit stacks, guaranteeing
//! stack safety for arbitrarily deep expression trees (Principle 5).
//!
//! # Core functions
//!
//! - [`post_order_ids`] — compute a de-duplicated post-order traversal
//!   of the expression DAG.
//! - [`walk_and_rebuild`] — walk bottom-up, applying a transformation at
//!   each node, and rebuild the tree with canonical constructors.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::arena::Arena;
use crate::node::SymbolId;
use crate::node::{ExprId, ExprNode};

// ═══════════════════════════════════════════════════════════════════════════
// Post-order traversal
// ═══════════════════════════════════════════════════════════════════════════

/// Compute a post-order traversal of the expression DAG rooted at `root`.
///
/// Each [`ExprId`] appears **at most once** in the output (de-duplicated
/// via a visited set).  Leaves appear before their parents, so any
/// bottom-up computation can process nodes in the returned order and
/// be guaranteed that all children are already processed.
///
/// Uses an explicit stack — never recurses.
pub(crate) fn post_order_ids(arena: &Arena, root: ExprId) -> Vec<ExprId> {
    let mut result = Vec::new();
    let mut visited: FxHashMap<ExprId, ()> = FxHashMap::default();
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
            visited.insert(id, ());
            result.push(id);
        }
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Walk-and-rebuild
// ═══════════════════════════════════════════════════════════════════════════

/// Walk an expression tree bottom-up and rebuild it with a
/// transformation function applied at each node.
///
/// `transform` is called on every node **after** its children have
/// been processed.  It returns `Some(new_id)` to replace the node, or
/// `None` to keep it (possibly with rebuilt children).
///
/// **This function never recurses.**  It uses [`post_order_ids`] for
/// traversal and an explicit cache for rebuilt results.
///
/// The rebuilt expression is re-canonicalized through the standard
/// `Arena` constructors (`add`, `mul`, `pow`, etc.), preserving all
/// canonical-form invariants.
///
/// If no transformation fires and no children change, the original
/// `root` [`ExprId`] is returned unchanged — no unnecessary allocation.
pub(crate) fn walk_and_rebuild(
    arena: &mut Arena,
    root: ExprId,
    transform: &dyn Fn(&Arena, ExprId) -> Option<ExprId>,
) -> ExprId {
    // Cache: original ExprId → rebuilt ExprId.
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    // Phase 1: Collect post-order traversal.
    let post_order = post_order_ids(arena, root);

    // Phase 2: Process each node in post-order (leaves first).
    for &id in &post_order {
        // Check the transform first.
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

// ═══════════════════════════════════════════════════════════════════════════
// Rebuild helper
// ═══════════════════════════════════════════════════════════════════════════

/// Rebuild a single node with children looked up from the cache.
///
/// If all children map to themselves (no changes), returns the
/// original `id` unchanged — avoiding unnecessary allocation.
///
/// When children have changed, uses the canonical constructors
/// (`arena.add`, `arena.mul`, etc.) to maintain canonical form.
pub(crate) fn rebuild_with_cache(
    arena: &mut Arena,
    id: ExprId,
    cache: &FxHashMap<ExprId, ExprId>,
) -> ExprId {
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
        ExprNode::Neg(inner) => rebuild_unary(arena, id, inner, cache, Arena::neg),
        ExprNode::Sin(inner) => rebuild_unary(arena, id, inner, cache, Arena::sin),
        ExprNode::Cos(inner) => rebuild_unary(arena, id, inner, cache, Arena::cos),
        ExprNode::Tan(inner) => rebuild_unary(arena, id, inner, cache, Arena::tan),
        ExprNode::Exp(inner) => rebuild_unary(arena, id, inner, cache, Arena::exp),
        ExprNode::Ln(inner) => rebuild_unary(arena, id, inner, cache, Arena::ln),
        ExprNode::Sqrt(inner) => rebuild_unary(arena, id, inner, cache, Arena::sqrt),
        ExprNode::Abs(inner) => rebuild_unary(arena, id, inner, cache, Arena::abs),

        // Unary: Asin, Acos, Atan, Sinh, Cosh, Tanh
        ExprNode::Asin(inner) => {
            let new_inner = cache.get(&inner).copied().unwrap_or(inner);
            if new_inner == inner {
                id
            } else {
                arena.intern(ExprNode::Asin(new_inner))
            }
        }
        ExprNode::Acos(inner) => {
            let new_inner = cache.get(&inner).copied().unwrap_or(inner);
            if new_inner == inner {
                id
            } else {
                arena.intern(ExprNode::Acos(new_inner))
            }
        }
        ExprNode::Atan(inner) => {
            let new_inner = cache.get(&inner).copied().unwrap_or(inner);
            if new_inner == inner {
                id
            } else {
                arena.intern(ExprNode::Atan(new_inner))
            }
        }
        ExprNode::Sinh(inner) => {
            let new_inner = cache.get(&inner).copied().unwrap_or(inner);
            if new_inner == inner {
                id
            } else {
                arena.intern(ExprNode::Sinh(new_inner))
            }
        }
        ExprNode::Cosh(inner) => {
            let new_inner = cache.get(&inner).copied().unwrap_or(inner);
            if new_inner == inner {
                id
            } else {
                arena.intern(ExprNode::Cosh(new_inner))
            }
        }
        ExprNode::Tanh(inner) => {
            let new_inner = cache.get(&inner).copied().unwrap_or(inner);
            if new_inner == inner {
                id
            } else {
                arena.intern(ExprNode::Tanh(new_inner))
            }
        }
        ExprNode::Asinh(inner) => {
            let new_inner = cache.get(&inner).copied().unwrap_or(inner);
            if new_inner == inner {
                id
            } else {
                arena.intern(ExprNode::Asinh(new_inner))
            }
        }
        ExprNode::Acosh(inner) => {
            let new_inner = cache.get(&inner).copied().unwrap_or(inner);
            if new_inner == inner {
                id
            } else {
                arena.intern(ExprNode::Acosh(new_inner))
            }
        }
        ExprNode::Atanh(inner) => {
            let new_inner = cache.get(&inner).copied().unwrap_or(inner);
            if new_inner == inner {
                id
            } else {
                arena.intern(ExprNode::Atanh(new_inner))
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

/// Helper for rebuilding unary nodes — avoids repeating the same
/// pattern 8 times.
#[inline]
fn rebuild_unary(
    arena: &mut Arena,
    id: ExprId,
    inner: ExprId,
    cache: &FxHashMap<ExprId, ExprId>,
    constructor: fn(&mut Arena, ExprId) -> ExprId,
) -> ExprId {
    let new_inner = cache.get(&inner).copied().unwrap_or(inner);
    if new_inner == inner {
        id
    } else {
        constructor(arena, new_inner)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Utility: check if an expression contains a given sub-expression
// ═══════════════════════════════════════════════════════════════════════════

/// Returns `true` if `needle` appears anywhere in the expression tree
/// rooted at `haystack`.
///
/// Uses an explicit stack — never recurses.
// Used for structural queries.
pub(crate) fn contains(arena: &Arena, haystack: ExprId, needle: ExprId) -> bool {
    if haystack == needle {
        return true;
    }

    let mut visited: FxHashMap<ExprId, ()> = FxHashMap::default();
    let mut stack: Vec<ExprId> = vec![haystack];

    while let Some(id) = stack.pop() {
        if id == needle {
            return true;
        }
        if visited.contains_key(&id) {
            continue;
        }
        visited.insert(id, ());

        let children = arena.node(id).children();
        stack.extend_from_slice(&children);
    }

    false
}

/// Collect the [`ExprId`] of every free symbol that appears in the
/// expression tree.  Each symbol appears at most once.
///
/// Uses an explicit stack — never recurses.
// Used for structural queries.
pub(crate) fn free_symbols(arena: &Arena, root: ExprId) -> Vec<ExprId> {
    let mut result = Vec::new();
    let mut visited: FxHashMap<ExprId, ()> = FxHashMap::default();
    let mut seen_syms: FxHashMap<SymbolId, ()> = FxHashMap::default();
    let mut stack: Vec<ExprId> = vec![root];

    while let Some(id) = stack.pop() {
        if visited.contains_key(&id) {
            continue;
        }
        visited.insert(id, ());

        if let ExprNode::Symbol(sid) = arena.node(id)
            && !seen_syms.contains_key(sid)
        {
            seen_syms.insert(*sid, ());
            result.push(id);
        }

        let children = arena.node(id).children();
        stack.extend_from_slice(&children);
    }

    result
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

    // ── post_order_ids ──────────────────────────────────────────────

    #[test]
    fn post_order_atom() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let order = post_order_ids(&a, x);
        assert_eq!(order, vec![x]);
    }

    #[test]
    fn post_order_add() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let sum = a.add(&[x, y]);
        let order = post_order_ids(&a, sum);
        // Children before parent. x and y come before sum.
        assert!(
            order.iter().position(|&id| id == x).unwrap()
                < order.iter().position(|&id| id == sum).unwrap()
        );
        assert!(
            order.iter().position(|&id| id == y).unwrap()
                < order.iter().position(|&id| id == sum).unwrap()
        );
    }

    #[test]
    fn post_order_deduplicates() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x + x canonicalizes to 2*x which has x as a child.
        // x should appear only once in the traversal.
        let expr = a.add(&[x, x]);
        let order = post_order_ids(&a, expr);
        let x_count = order.iter().filter(|&&id| id == x).count();
        assert_eq!(x_count, 1, "x should appear exactly once");
    }

    // ── walk_and_rebuild ────────────────────────────────────────────

    #[test]
    fn walk_identity_is_noop() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let xp = a.pow(x, two);
        let one = a.one;
        let expr = a.add(&[xp, x, one]);
        let result = walk_and_rebuild(&mut a, expr, &|_, _| None);
        assert_eq!(result, expr, "identity walk should return same ExprId");
    }

    #[test]
    fn walk_replace_leaf() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);
        let expr = a.pow(x, two);
        let result = walk_and_rebuild(&mut a, expr, &|_, id| {
            if id == x { Some(y) } else { None }
        });
        assert_eq!(display(&a, result), "y^2");
    }

    #[test]
    fn walk_deep_no_overflow() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        // 1000-deep sin chain.
        let mut expr = x;
        for _ in 0..1000 {
            expr = a.sin(expr);
        }
        let result = walk_and_rebuild(&mut a, expr, &|_, id| {
            if id == x { Some(y) } else { None }
        });
        assert_ne!(result, expr);
    }

    // ── contains ────────────────────────────────────────────────────

    #[test]
    fn contains_self() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        assert!(contains(&a, x, x));
    }

    #[test]
    fn contains_child() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let sum = a.add(&[x, y]);
        assert!(contains(&a, sum, x));
        assert!(contains(&a, sum, y));
    }

    #[test]
    fn does_not_contain() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let z = sym(&mut a, "z");
        let sum = a.add(&[x, y]);
        assert!(!contains(&a, sum, z));
    }

    #[test]
    fn contains_deep() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let xp = a.pow(x, two);
        let expr = a.sin(xp);
        assert!(contains(&a, expr, x));
        assert!(contains(&a, expr, xp));
    }

    // ── free_symbols ────────────────────────────────────────────────

    #[test]
    fn free_symbols_atom() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let syms = free_symbols(&a, x);
        assert_eq!(syms.len(), 1);
    }

    #[test]
    fn free_symbols_expression() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let expr = a.add(&[x, y]);
        let syms = free_symbols(&a, expr);
        assert_eq!(syms.len(), 2);
    }

    #[test]
    fn free_symbols_no_duplicates() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x + x = 2*x, which contains x once.
        let expr = a.add(&[x, x]);
        let syms = free_symbols(&a, expr);
        assert_eq!(syms.len(), 1, "x should appear once even in 2*x");
    }

    #[test]
    fn free_symbols_constant_has_none() {
        let a = Arena::new();
        let syms = free_symbols(&a, a.pi);
        assert!(syms.is_empty(), "pi has no free symbols");
    }
}
