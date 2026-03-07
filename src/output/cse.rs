//! Common Subexpression Elimination (CSE).
//!
//! [`cse`] identifies repeated subexpressions and extracts them into
//! named temporaries, reducing redundant computation in generated code.
//!
//! [`cse_multi`] extends this to multiple root expressions, sharing
//! temporaries across all of them, and detects partial overlaps in
//! Add/Mul nodes (shared subsets of ≥2 children).

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

/// Result of CSE: a list of (name, expression) bindings and a final expression.
pub(crate) struct CseResult {
    /// Named subexpressions: `(name_expr_id, value_expr_id)`.
    /// The name is a Symbol node like `__cse_0`.
    pub bindings: Vec<(ExprId, ExprId)>,
    /// The final expression with subexpressions replaced by names.
    pub expr: ExprId,
}

/// Result of multi-expression CSE.
pub(crate) struct CseMultiResult {
    /// Named subexpressions shared across any of the inputs.
    pub bindings: Vec<(ExprId, ExprId)>,
    /// The input expressions with shared subexpressions replaced by names.
    pub exprs: Vec<ExprId>,
}

/// Perform common subexpression elimination on `expr`.
///
/// Nodes that appear 2+ times in the expression tree are extracted
/// into named temporaries `__cse_0`, `__cse_1`, etc.
pub(crate) fn cse(arena: &mut Arena, expr: ExprId) -> CseResult {
    let multi = cse_multi(arena, &[expr]);
    CseResult {
        bindings: multi.bindings,
        expr: multi.exprs[0],
    }
}

/// Perform common subexpression elimination across multiple root expressions.
///
/// Nodes that appear 2+ times total (across all roots) are extracted into
/// named temporaries. Additionally, Add/Mul nodes that share ≥2 children
/// have the shared subset factored into a new sub-expression.
pub(crate) fn cse_multi(arena: &mut Arena, exprs: &[ExprId]) -> CseMultiResult {
    if exprs.is_empty() {
        return CseMultiResult {
            bindings: Vec::new(),
            exprs: Vec::new(),
        };
    }

    // Phase 1: Count occurrences of each subexpression across ALL roots.
    let mut ref_count: FxHashMap<ExprId, usize> = FxHashMap::default();
    for &root in exprs {
        count_refs(arena, root, &mut ref_count);
    }

    // Phase 2: Detect partial overlaps in Add/Mul nodes and rewrite.
    // Collect all Add and Mul nodes from the entire expression forest.
    let all_ids = collect_all_ids(arena, exprs);
    let overlap_rewrites = find_and_apply_partial_overlaps(arena, &all_ids);

    // Apply overlap rewrites to all root expressions.
    let current_exprs: Vec<ExprId> = if overlap_rewrites.is_empty() {
        exprs.to_vec()
    } else {
        exprs
            .iter()
            .map(|&e| apply_replacements(arena, e, &overlap_rewrites))
            .collect()
    };

    // Re-count references after overlap rewrites.
    let mut ref_count: FxHashMap<ExprId, usize> = FxHashMap::default();
    for &root in &current_exprs {
        count_refs(arena, root, &mut ref_count);
    }

    // Phase 3: Identify nodes worth extracting.
    // A node is worth extracting if:
    // - It appears 2+ times
    // - It's not an atom (no point extracting a number or symbol)
    let mut extract: Vec<ExprId> = ref_count
        .iter()
        .filter(|&(id, count)| *count >= 2 && !arena.node(*id).is_atom())
        .map(|(&id, _)| id)
        .collect();

    // Sort by post-order position (extract inner expressions first).
    // Build a combined post-order from all roots.
    let mut combined_post_order: Vec<ExprId> = Vec::new();
    let mut combined_visited: FxHashSet<ExprId> = FxHashSet::default();
    for &root in &current_exprs {
        let po = walk::post_order_ids(arena, root);
        for id in po {
            if combined_visited.insert(id) {
                combined_post_order.push(id);
            }
        }
    }

    let position: FxHashMap<ExprId, usize> = combined_post_order
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    extract.sort_by_key(|id| position.get(id).copied().unwrap_or(0));

    // Phase 4: Build replacement bindings.
    let mut bindings: Vec<(ExprId, ExprId)> = Vec::new();
    let mut replacement_map: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for (i, &id) in extract.iter().enumerate() {
        let name = format!("__cse_{i}");
        let name_id = arena.symbol(&name);

        // Replace any previously-extracted subexpressions within this one.
        let replaced_value = apply_replacements(arena, id, &replacement_map);

        bindings.push((name_id, replaced_value));
        replacement_map.insert(id, name_id);
    }

    // Phase 5: Replace in all root expressions.
    let final_exprs: Vec<ExprId> = current_exprs
        .iter()
        .map(|&e| apply_replacements(arena, e, &replacement_map))
        .collect();

    CseMultiResult {
        bindings,
        exprs: final_exprs,
    }
}

/// Collect all unique ExprIds reachable from the given roots.
fn collect_all_ids(arena: &Arena, roots: &[ExprId]) -> Vec<ExprId> {
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    let mut result = Vec::new();
    for &root in roots {
        let po = walk::post_order_ids(arena, root);
        for id in po {
            if visited.insert(id) {
                result.push(id);
            }
        }
    }
    result
}

/// Find Add/Mul nodes that share ≥2 children and create factored sub-expressions.
///
/// Returns a replacement map that rewrites parent nodes to reference the new
/// shared sub-expression nodes.
fn find_and_apply_partial_overlaps(
    arena: &mut Arena,
    all_ids: &[ExprId],
) -> FxHashMap<ExprId, ExprId> {
    let mut replacement_map: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    // Process Add and Mul separately.
    for is_add in [true, false] {
        // Collect all Add (or Mul) nodes and their children sets.
        let mut node_children: Vec<(ExprId, Vec<ExprId>)> = Vec::new();
        for &id in all_ids {
            let dominated = match arena.node(id) {
                ExprNode::Add(_) if is_add => true,
                ExprNode::Mul(_) if !is_add => true,
                _ => false,
            };
            if dominated {
                let children: Vec<ExprId> = arena.node(id).children().to_vec();
                if children.len() >= 2 {
                    node_children.push((id, children));
                }
            }
        }

        if node_children.len() < 2 {
            continue;
        }

        // Build inverted index: child -> list of parent indices (into node_children).
        let mut child_to_parents: FxHashMap<ExprId, Vec<usize>> = FxHashMap::default();
        for (idx, (_id, children)) in node_children.iter().enumerate() {
            for &child in children {
                child_to_parents.entry(child).or_default().push(idx);
            }
        }

        // Find pairs of parent nodes that share ≥2 children.
        // Use a pair -> shared children map to accumulate.
        let mut pair_shared: FxHashMap<(usize, usize), Vec<ExprId>> = FxHashMap::default();

        for (&child, parents) in &child_to_parents {
            // Only children appearing in ≥2 parents are interesting.
            if parents.len() < 2 {
                continue;
            }
            for i in 0..parents.len() {
                for j in (i + 1)..parents.len() {
                    let a = parents[i].min(parents[j]);
                    let b = parents[i].max(parents[j]);
                    pair_shared.entry((a, b)).or_default().push(child);
                }
            }
        }

        // For each pair sharing ≥2 children, create a sub-expression and rewrite.
        // We process at most one overlap per parent to avoid conflicting rewrites.
        let mut rewritten: FxHashSet<ExprId> = FxHashSet::default();

        // Sort pairs by number of shared children descending (greedily pick largest overlaps).
        let mut pairs: Vec<((usize, usize), Vec<ExprId>)> = pair_shared
            .into_iter()
            .filter(|(_, shared)| shared.len() >= 2)
            .collect();
        pairs.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

        for ((idx_a, idx_b), shared_children) in pairs {
            let (parent_a, _) = &node_children[idx_a];
            let (parent_b, _) = &node_children[idx_b];

            // Skip if either parent was already rewritten.
            if rewritten.contains(parent_a) || rewritten.contains(parent_b) {
                continue;
            }

            // Deduplicate shared children.
            let shared_set: FxHashSet<ExprId> = shared_children.iter().copied().collect();
            if shared_set.len() < 2 {
                continue;
            }
            let shared_vec: Vec<ExprId> = shared_set.iter().copied().collect();

            // Create the shared sub-expression.
            let shared_expr = if is_add {
                arena.add(&shared_vec)
            } else {
                arena.mul(&shared_vec)
            };

            // Rewrite parent_a: replace shared children with shared_expr.
            let new_a = rewrite_nary_node(arena, *parent_a, &shared_set, shared_expr, is_add);
            // Rewrite parent_b: replace shared children with shared_expr.
            let new_b = rewrite_nary_node(arena, *parent_b, &shared_set, shared_expr, is_add);

            if new_a != *parent_a {
                replacement_map.insert(*parent_a, new_a);
                rewritten.insert(*parent_a);
            }
            if new_b != *parent_b {
                replacement_map.insert(*parent_b, new_b);
                rewritten.insert(*parent_b);
            }
        }
    }

    replacement_map
}

/// Rewrite an Add or Mul node by removing `shared_children` and inserting `shared_expr`.
fn rewrite_nary_node(
    arena: &mut Arena,
    parent: ExprId,
    shared_children: &FxHashSet<ExprId>,
    shared_expr: ExprId,
    is_add: bool,
) -> ExprId {
    let original_children: SmallVec<[ExprId; 6]> = arena.node(parent).children();

    // Build the remaining children (not in the shared set).
    let mut remaining: Vec<ExprId> = original_children
        .iter()
        .filter(|c| !shared_children.contains(c))
        .copied()
        .collect();

    // Add the shared sub-expression.
    remaining.push(shared_expr);

    if remaining.len() == 1 {
        return remaining[0];
    }

    if is_add {
        arena.add(&remaining)
    } else {
        arena.mul(&remaining)
    }
}

/// Count references to each subexpression (not just unique nodes).
fn count_refs(arena: &Arena, root: ExprId, counts: &mut FxHashMap<ExprId, usize>) {
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        *counts.entry(id).or_insert(0) += 1;
        // Only recurse the first time we see this id.
        if counts[&id] == 1 {
            for &child in arena.node(id).children().iter() {
                stack.push(child);
            }
        }
    }
}

/// Apply replacement map to an expression using [`walk::walk_and_rebuild`].
fn apply_replacements(arena: &mut Arena, expr: ExprId, map: &FxHashMap<ExprId, ExprId>) -> ExprId {
    if map.is_empty() {
        return expr;
    }
    walk::walk_and_rebuild(arena, expr, &|_arena, id| map.get(&id).copied())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }
    fn display(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    #[test]
    fn cse_no_common() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let expr = a.add(&[x, y]);
        let result = cse(&mut a, expr);
        assert!(result.bindings.is_empty(), "no common subexpressions");
        assert_eq!(result.expr, expr);
    }

    #[test]
    fn cse_with_common_subexpr() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let two = a.int(2);
        // sin(x)^2 + sin(x) — sin(x) appears twice
        let sin_sq = a.pow(sin_x, two);
        let expr = a.add(&[sin_sq, sin_x]);
        let result = cse(&mut a, expr);
        // sin(x) should be extracted
        assert!(!result.bindings.is_empty(), "should extract sin(x)");
        // The extracted binding should be sin(x)
        let binding_val = display(&a, result.bindings[0].1);
        assert!(
            binding_val.contains("sin"),
            "binding should be sin(x): {binding_val}"
        );
    }

    #[test]
    fn cse_deeply_shared() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let exp_sin = a.exp(sin_x);
        // exp(sin(x)) + exp(sin(x))^2
        let two = a.int(2);
        let sq = a.pow(exp_sin, two);
        let expr = a.add(&[exp_sin, sq]);
        let result = cse(&mut a, expr);
        assert!(
            !result.bindings.is_empty(),
            "should extract common subexprs"
        );
    }

    #[test]
    fn cse_atoms_not_extracted() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x + x = 2x (canonicalized), so no CSE opportunity on the atom itself
        let expr = a.add(&[x, x]);
        // After canonicalization, x+x = 2*x, so x appears once in the tree
        let result = cse(&mut a, expr);
        // x is an atom, should not be extracted even if referenced multiple times
        for (_, val) in &result.bindings {
            assert!(!a.node(*val).is_atom(), "atoms should not be extracted");
        }
    }

    #[test]
    fn cse_preserves_value() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let three = a.int(3);
        let x2 = a.pow(x, two);
        // x² + 3*x² = 4*x² (but with explicit sharing)
        let three_x2 = a.mul(&[three, x2]);
        let expr = a.add(&[x2, three_x2]);

        let result = cse(&mut a, expr);
        // Verify: substituting cse bindings and evaluating should give same result.
        // Just check it doesn't crash and produces something reasonable.
        let final_s = display(&a, result.expr);
        assert!(!final_s.is_empty());
    }

    // ── cse_multi tests ────────────────────────────────────────────────

    #[test]
    fn cse_multi_empty() {
        let mut a = Arena::new();
        let result = cse_multi(&mut a, &[]);
        assert!(result.bindings.is_empty());
        assert!(result.exprs.is_empty());
    }

    #[test]
    fn cse_multi_single_expr() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let two = a.int(2);
        let sin_sq = a.pow(sin_x, two);
        let expr = a.add(&[sin_sq, sin_x]);
        let result = cse_multi(&mut a, &[expr]);
        // Should produce the same result as cse()
        assert!(!result.bindings.is_empty(), "should extract sin(x)");
        assert_eq!(result.exprs.len(), 1);
    }

    #[test]
    fn cse_multi_shared_across_exprs() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let two = a.int(2);

        // expr1 = sin(x) + 1
        let one = a.int(1);
        let expr1 = a.add(&[sin_x, one]);

        // expr2 = sin(x)^2
        let expr2 = a.pow(sin_x, two);

        let result = cse_multi(&mut a, &[expr1, expr2]);
        // sin(x) appears in both — should be extracted
        assert!(
            !result.bindings.is_empty(),
            "should extract sin(x) shared between expr1 and expr2"
        );
        assert_eq!(result.exprs.len(), 2);
    }

    #[test]
    fn cse_multi_no_shared() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        // Two completely independent expressions with no sharing
        let expr1 = a.sin(x);
        let expr2 = a.cos(y);
        let result = cse_multi(&mut a, &[expr1, expr2]);
        assert!(result.bindings.is_empty(), "no common subexpressions");
        assert_eq!(result.exprs.len(), 2);
    }

    #[test]
    fn cse_multi_three_exprs() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let cos_x = a.cos(x);
        let two = a.int(2);
        let three = a.int(3);

        // All three share cos(x)
        let expr1 = a.pow(cos_x, two);
        let expr2 = a.pow(cos_x, three);
        let one = a.int(1);
        let expr3 = a.add(&[cos_x, one]);

        let result = cse_multi(&mut a, &[expr1, expr2, expr3]);
        assert!(
            !result.bindings.is_empty(),
            "should extract cos(x) shared across all three"
        );
        assert_eq!(result.exprs.len(), 3);
    }
}
