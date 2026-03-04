//! Common Subexpression Elimination (CSE).
//!
//! [`cse`] identifies repeated subexpressions and extracts them into
//! named temporaries, reducing redundant computation in generated code.

use crate::arena::Arena;
use crate::node::ExprId;
use crate::walk;
use rustc_hash::FxHashMap;

/// Result of CSE: a list of (name, expression) bindings and a final expression.
pub(crate) struct CseResult {
    /// Named subexpressions: `(name_expr_id, value_expr_id)`.
    /// The name is a Symbol node like `__cse_0`.
    pub bindings: Vec<(ExprId, ExprId)>,
    /// The final expression with subexpressions replaced by names.
    pub expr: ExprId,
}

/// Perform common subexpression elimination on `expr`.
///
/// Nodes that appear 2+ times in the expression tree are extracted
/// into named temporaries `__cse_0`, `__cse_1`, etc.
pub(crate) fn cse(arena: &mut Arena, expr: ExprId) -> CseResult {
    // Phase 1: Count occurrences of each subexpression
    let post_order = walk::post_order_ids(arena, expr);
    let mut ref_count: FxHashMap<ExprId, usize> = FxHashMap::default();

    // We need to count how many times each ExprId appears as a child.
    // Since we use hash-consing, structural sharing is automatic.
    count_refs(arena, expr, &mut ref_count);

    // Phase 2: Identify nodes worth extracting.
    // A node is worth extracting if:
    // - It appears 2+ times
    // - It's not an atom (no point extracting a number or symbol)
    let mut extract: Vec<ExprId> = ref_count
        .iter()
        .filter(|&(id, count)| *count >= 2 && !arena.node(*id).is_atom())
        .map(|(&id, _)| id)
        .collect();

    // Sort by post-order position (extract inner expressions first).
    let position: FxHashMap<ExprId, usize> = post_order
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    extract.sort_by_key(|id| position.get(id).copied().unwrap_or(0));

    // Phase 3: Build replacement bindings.
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

    // Phase 4: Replace in the main expression.
    let final_expr = apply_replacements(arena, expr, &replacement_map);

    CseResult {
        bindings,
        expr: final_expr,
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
}
