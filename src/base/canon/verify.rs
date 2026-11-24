//! Verify canonical form (part of [`canon`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Verify canonical form
// ═══════════════════════════════════════════════════════════════════════════

/// Verify that an expression is in canonical form.
///
/// Returns a list of violations found. An empty list means the
/// expression is properly canonical.
///
/// Every node reachable from `id` is checked once (the walk is over the
/// hash-consed DAG with a visited set, using an explicit stack), so the
/// cost is proportional to the DAG, not the unfolded tree.
///
/// This is intended for use in `debug_assert!` and property-based tests.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn verify_canonical(arena: &mut Arena, id: ExprId) -> Vec<String> {
    let mut errors = Vec::new();
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack: Vec<ExprId> = vec![id];
    while let Some(cur) = stack.pop() {
        if !visited.insert(cur) {
            continue;
        }
        verify_node(arena, cur, &mut errors);
        arena.node(cur).for_each_child(|c| stack.push(c));
    }
    errors
}

/// Check the canonical-form invariants of the node `id` itself (not its
/// descendants).  Constructors call this in debug builds on the node they
/// just built: the children were verified when *they* were built.
pub(crate) fn verify_canonical_shallow(arena: &mut Arena, id: ExprId) -> Vec<String> {
    let mut errors = Vec::new();
    verify_node(arena, id, &mut errors);
    errors
}

/// The per-node invariants behind [`verify_canonical`]; does not recurse.
fn verify_node(arena: &mut Arena, id: ExprId, errors: &mut Vec<String>) {
    match arena.node(id).clone() {
        ExprNode::Add(ref children) => {
            // 1. Must have >= 2 children
            if children.len() < 2 {
                errors.push(format!("Add with {} children (need >= 2)", children.len()));
            }
            // 2. No child should be zero
            for (i, &child) in children.iter().enumerate() {
                if child == arena.zero {
                    errors.push(format!("Add child {i} is zero"));
                }
            }
            // 3. No child should be an Add (must be flattened)
            for (i, &child) in children.iter().enumerate() {
                if matches!(arena.node(child), ExprNode::Add(_)) {
                    errors.push(format!("Add child {i} is nested Add (not flattened)"));
                }
            }
            // 4. Children must be sorted by SortKey
            for i in 1..children.len() {
                let key_prev = arena.sort_key(children[i - 1]);
                let key_curr = arena.sort_key(children[i]);
                if key_prev > key_curr {
                    errors.push(format!(
                        "Add children {}/{} not sorted: {:?} > {:?}",
                        i - 1,
                        i,
                        key_prev,
                        key_curr
                    ));
                }
            }
            // 5. No two children should have the same term key
            //    (like terms should be merged)
            {
                let mut seen_keys = FxHashSet::default();
                for (idx, &child) in children.iter().enumerate() {
                    let (_, key) = arena.as_coeff_term(child);
                    if !seen_keys.insert(key) {
                        errors.push(format!(
                            "Add child {idx} has duplicate term key (like terms not merged)"
                        ));
                    }
                }
            }
        }
        ExprNode::Mul(ref children) => {
            // 1. Must have >= 2 children
            if children.len() < 2 {
                errors.push(format!("Mul with {} children (need >= 2)", children.len()));
            }
            // 2. No child should be one
            for (i, &child) in children.iter().enumerate() {
                if child == arena.one {
                    errors.push(format!("Mul child {i} is one"));
                }
            }
            // 3. No child should be a Mul (must be flattened)
            for (i, &child) in children.iter().enumerate() {
                if matches!(arena.node(child), ExprNode::Mul(_)) {
                    errors.push(format!("Mul child {i} is nested Mul (not flattened)"));
                }
            }
            // 4. Children must be sorted by SortKey
            for i in 1..children.len() {
                let key_prev = arena.sort_key(children[i - 1]);
                let key_curr = arena.sort_key(children[i]);
                if key_prev > key_curr {
                    errors.push(format!("Mul children {}/{} not sorted", i - 1, i));
                }
            }
            // 5. At most one Num child
            let num_count = children
                .iter()
                .filter(|&&c| matches!(arena.node(c), ExprNode::Num(_)))
                .count();
            if num_count > 1 {
                errors.push(format!(
                    "Mul has {num_count} Num children (should be at most 1)"
                ));
            }
        }
        ExprNode::Neg(inner) => {
            // 1. No double negation
            if matches!(arena.node(inner), ExprNode::Neg(_)) {
                errors.push("Double negation Neg(Neg(...))".to_string());
            }
            // 2. Inner should not be zero
            if inner == arena.zero {
                errors.push("Neg(0) should be 0".to_string());
            }
            // 3. Inner should not be a Num (should be folded into negative Num)
            if matches!(arena.node(inner), ExprNode::Num(_)) {
                errors.push("Neg(Num) should be folded into negative Num".to_string());
            }
        }
        ExprNode::Pow(base, exp) => {
            // 1. exp should not be 0 (should be 1)
            if exp == arena.zero {
                errors.push("Pow(x, 0) should be 1".to_string());
            }
            // 2. exp should not be 1 (should be base)
            if exp == arena.one {
                errors.push("Pow(x, 1) should be x".to_string());
            }
            // 3. base should not be 1 (should be 1)
            if base == arena.one {
                errors.push("Pow(1, x) should be 1".to_string());
            }
        }
        ExprNode::And(ref children) | ExprNode::Or(ref children) => {
            let label = if matches!(arena.node(id), ExprNode::And(_)) {
                "And"
            } else {
                "Or"
            };
            // 1. Must have >= 2 children
            if children.len() < 2 {
                errors.push(format!(
                    "{} with {} children (need >= 2)",
                    label,
                    children.len()
                ));
            }
            // 2. Children must be sorted by SortKey
            for i in 1..children.len() {
                let key_prev = arena.sort_key(children[i - 1]);
                let key_curr = arena.sort_key(children[i]);
                if key_prev > key_curr {
                    errors.push(format!(
                        "{} children {}/{} not sorted: {:?} > {:?}",
                        label,
                        i - 1,
                        i,
                        key_prev,
                        key_curr
                    ));
                }
            }
        }
        // Atoms and functions carry no ordering invariants of their own.
        _ => {}
    }
}
