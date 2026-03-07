//! Separate an expression into factors depending on disjoint variable sets.
//!
//! Given `f(x)·g(y)·h(x,y)`, returns the factored form where
//! `f` depends only on `x`, `g` depends only on `y`, and `h`
//! depends on both (left as-is).

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};

/// Separate variables in a Mul expression.
///
/// Given an expression and a list of variable groups, partition
/// the factors by which group they belong to. Factors that depend
/// on multiple groups are collected into a "mixed" group.
///
/// Returns a `Vec<(Vec<ExprId>, ExprId)>` where each entry is
/// `(dependent_vars, product_of_factors)`. The `dependent_vars`
/// list contains only those vars from `vars` that the factor group
/// depends on. An empty `dependent_vars` means "constant" (depends
/// on none of the given vars).
pub(crate) fn separatevars(
    arena: &mut Arena,
    expr: ExprId,
    vars: &[ExprId],
) -> Vec<(Vec<ExprId>, ExprId)> {
    // Get the top-level multiplicative factors.
    let children = match arena.node(expr).clone() {
        ExprNode::Mul(c) => c.to_vec(),
        _ => {
            // Not a Mul — classify the whole expression as one group.
            let dep = dependent_vars(arena, expr, vars);
            return vec![(dep, expr)];
        }
    };

    // Group factors by which vars they depend on.
    let mut groups: Vec<(Vec<ExprId>, Vec<ExprId>)> = Vec::new();

    for &factor in &children {
        let dep = dependent_vars(arena, factor, vars);

        // Find existing group with same dependency set.
        let mut found = false;
        for (gvars, gfactors) in &mut groups {
            if *gvars == dep {
                gfactors.push(factor);
                found = true;
                break;
            }
        }
        if !found {
            groups.push((dep, vec![factor]));
        }
    }

    // Combine each group's factors into a single product.
    groups
        .into_iter()
        .map(|(dep, factors)| {
            let product = if factors.len() == 1 {
                factors[0]
            } else {
                arena.mul(&factors)
            };
            (dep, product)
        })
        .collect()
}

/// Determine which of `vars` appear as free symbols in `expr`.
fn dependent_vars(arena: &Arena, expr: ExprId, vars: &[ExprId]) -> Vec<ExprId> {
    let factor_syms = crate::base::walk::free_symbols(arena, expr);
    let mut dep = Vec::new();
    for &var in vars {
        if factor_syms.contains(&var) {
            dep.push(var);
        }
    }
    dep
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::arena::Arena;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }

    #[test]
    fn separate_simple_product() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        // expr = x * y
        let expr = a.mul(&[x, y]);
        let result = separatevars(&mut a, expr, &[x, y]);
        // Should separate into x-group and y-group
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn non_mul_returns_single_group() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let result = separatevars(&mut a, x, &[x]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, vec![x]);
        assert_eq!(result[0].1, x);
    }

    #[test]
    fn constant_factor_has_empty_deps() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        // expr = 2 * x
        let expr = a.mul(&[two, x]);
        let result = separatevars(&mut a, expr, &[x]);
        // Should have a constant group (empty deps) and an x group
        let const_groups: Vec<_> = result.iter().filter(|(d, _)| d.is_empty()).collect();
        let x_groups: Vec<_> = result.iter().filter(|(d, _)| d == &vec![x]).collect();
        assert_eq!(const_groups.len(), 1);
        assert_eq!(x_groups.len(), 1);
    }

    #[test]
    fn mixed_factor_stays_together() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        // expr = (x + y)  — this is Add, not Mul, so it's one group
        let xy_sum = a.add(&[x, y]);
        let result = separatevars(&mut a, xy_sum, &[x, y]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0.len(), 2); // depends on both x and y
    }
}
