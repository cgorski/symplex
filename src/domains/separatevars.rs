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

/// Additive separation: partition the terms of a sum by the variables
/// they depend on.
///
/// `f(x) + g(y) + h(x, y) + 3` becomes the groups `([x], f(x))`,
/// `([y], g(y))`, `([x, y], h(x, y))`, `([], 3)`.  Non-`Add` expressions
/// form a single group.  Terms with the same dependency set are summed.
pub(crate) fn separatevars_additive(
    arena: &mut Arena,
    expr: ExprId,
    vars: &[ExprId],
) -> Vec<(Vec<ExprId>, ExprId)> {
    let children = match arena.node(expr).clone() {
        ExprNode::Add(c) => c.to_vec(),
        _ => {
            let dep = dependent_vars(arena, expr, vars);
            return vec![(dep, expr)];
        }
    };

    let mut groups: Vec<(Vec<ExprId>, Vec<ExprId>)> = Vec::new();
    for &term in &children {
        let dep = dependent_vars(arena, term, vars);
        match groups.iter_mut().find(|(g, _)| *g == dep) {
            Some((_, terms)) => terms.push(term),
            None => groups.push((dep, vec![term])),
        }
    }

    groups
        .into_iter()
        .map(|(dep, terms)| {
            let sum = if terms.len() == 1 {
                terms[0]
            } else {
                arena.add(&terms)
            };
            (dep, sum)
        })
        .collect()
}

/// Multiplicative separation into one factor per variable.
///
/// Returns `Some(factors)` with `factors[i]` depending on `vars[i]` only
/// (or `1`) such that the product of all factors equals `expr`, or `None`
/// when some factor depends on two or more of the variables.  Any factor
/// free of every variable is multiplied into the *first* variable's
/// factor so that the product is exact.
///
/// Before giving up, a sum is first written as `content · (…)` via
/// [`symbolic_factor_terms_pair`](crate::simplify::factor_terms::symbolic_factor_terms_pair)
/// so that `x·y + x·z = x·(y + z)` separates when `y + z` involves a
/// single variable.
pub(crate) fn separatevars_dict(
    arena: &mut Arena,
    expr: ExprId,
    vars: &[ExprId],
) -> Option<Vec<ExprId>> {
    if vars.is_empty() {
        return None;
    }
    // Try the expression as-is, then its factored form.
    if let Some(r) = try_separate_dict(arena, expr, vars) {
        return Some(r);
    }
    if matches!(arena.node(expr), ExprNode::Add(_)) {
        let (content, inner) =
            crate::simplify::factor_terms::symbolic_factor_terms_pair(arena, expr);
        if content != arena.one {
            let factored = arena.mul(&[content, inner]);
            if factored != expr {
                return try_separate_dict(arena, factored, vars);
            }
        }
    }
    None
}

fn try_separate_dict(arena: &mut Arena, expr: ExprId, vars: &[ExprId]) -> Option<Vec<ExprId>> {
    let groups = separatevars(arena, expr, vars);
    let mut factors: Vec<ExprId> = vec![arena.one; vars.len()];
    let mut coeff: Vec<ExprId> = Vec::new();
    for (dep, product) in groups {
        match dep.len() {
            0 => coeff.push(product),
            1 => {
                let i = vars.iter().position(|&v| v == dep[0])?;
                factors[i] = arena.mul(&[factors[i], product]);
            }
            _ => return None,
        }
    }
    if !coeff.is_empty() {
        coeff.push(factors[0]);
        factors[0] = arena.mul(&coeff);
    }
    Some(factors)
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

    // ── additive / dict separation ─────────────────────────────────

    #[test]
    fn additive_groups_by_dependency() {
        let mut a = Arena::new();
        let (x, y) = (sym(&mut a, "x"), sym(&mut a, "y"));
        let sx = a.sin(x);
        let two = a.int(2);
        let y2 = a.pow(y, two);
        let xy = a.mul(&[x, y]);
        let three = a.int(3);
        let e = a.add(&[sx, y2, xy, three]);
        let groups = separatevars_additive(&mut a, e, &[x, y]);
        assert_eq!(groups.len(), 4);
        assert!(groups.iter().any(|(d, g)| d.is_empty() && *g == three));
        assert!(groups.iter().any(|(d, g)| d.len() == 2 && *g == xy));
    }

    #[test]
    fn dict_separable_product_and_failure() {
        let mut a = Arena::new();
        let (x, y) = (sym(&mut a, "x"), sym(&mut a, "y"));
        let sx = a.sin(x);
        let two = a.int(2);
        let e = a.mul(&[two, sx, y]);
        let f = separatevars_dict(&mut a, e, &[x, y]).expect("separable");
        assert_eq!(a.display(f[0]).to_string(), "2*sin(x)");
        assert_eq!(f[1], y);
        let sum = a.add(&[x, y]);
        assert!(separatevars_dict(&mut a, sum, &[x, y]).is_none());
        assert!(separatevars_dict(&mut a, e, &[]).is_none());
    }

    #[test]
    fn dict_factors_sums_before_separating() {
        let mut a = Arena::new();
        let (x, y) = (sym(&mut a, "x"), sym(&mut a, "y"));
        let xy = a.mul(&[x, y]);
        let two = a.int(2);
        let y2 = a.pow(y, two);
        let xy2 = a.mul(&[x, y2]);
        let e = a.add(&[xy, xy2]); // x*y + x*y^2
        let f = separatevars_dict(&mut a, e, &[x, y]).expect("separable after factoring");
        assert_eq!(f[0], x);
        assert_eq!(a.display(f[1]).to_string(), "y*(y + 1)");
    }
}
