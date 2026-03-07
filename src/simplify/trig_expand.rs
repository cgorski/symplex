//! Trigonometric expansion.
//!
//! Implements [`expand_trig`], which applies addition formulas to
//! trigonometric functions with composite arguments:
//!
//! - `sin(a + b)` → `sin(a)·cos(b) + cos(a)·sin(b)`
//! - `cos(a + b)` → `cos(a)·cos(b) - sin(a)·sin(b)`
//! - `sin(n·x)` → expansion via repeated addition formula
//! - `cos(n·x)` → expansion via repeated addition formula
//!
//! This is a specialized expansion method, separate from `simplify`
//! because it makes expressions larger (not simpler).

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;

/// Expand trigonometric functions with composite arguments.
///
/// Walks the expression bottom-up and applies addition formulas
/// to `sin` and `cos` nodes whose arguments are `Add` nodes.
pub(crate) fn expand_trig(arena: &mut Arena, expr: ExprId) -> ExprId {
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache = rustc_hash::FxHashMap::default();

    for &id in &post_order {
        let node = arena.node(id).clone();
        let expanded = match node {
            ExprNode::Sin(inner) => {
                let inner = cache.get(&inner).copied().unwrap_or(inner);
                expand_sin_mul(arena, inner)
                    .or_else(|| expand_sin_add(arena, inner))
                    .unwrap_or_else(|| arena.sin(inner))
            }
            ExprNode::Cos(inner) => {
                let inner = cache.get(&inner).copied().unwrap_or(inner);
                expand_cos_mul(arena, inner)
                    .or_else(|| expand_cos_add(arena, inner))
                    .unwrap_or_else(|| arena.cos(inner))
            }
            // Rebuild Add/Mul/Pow/Neg with expanded children.
            ExprNode::Add(ref children) => {
                let new: smallvec::SmallVec<[ExprId; 6]> = children
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();
                if new == *children {
                    id
                } else {
                    arena.add(&new)
                }
            }
            ExprNode::Mul(ref children) => {
                let new: smallvec::SmallVec<[ExprId; 6]> = children
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();
                if new == *children {
                    id
                } else {
                    arena.mul(&new)
                }
            }
            ExprNode::Pow(base, exp) => {
                let nb = cache.get(&base).copied().unwrap_or(base);
                let ne = cache.get(&exp).copied().unwrap_or(exp);
                if nb == base && ne == exp {
                    id
                } else {
                    arena.pow(nb, ne)
                }
            }
            ExprNode::Neg(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                if ni == inner { id } else { arena.neg(ni) }
            }
            _ => {
                // Rebuild any other node with cached children
                crate::base::walk::rebuild_with_cache(arena, id, &cache)
            }
        };

        cache.insert(id, expanded);
    }

    cache.get(&expr).copied().unwrap_or(expr)
}

/// Expand `sin(inner)` when `inner` is an Add: `sin(a + b) → sin(a)cos(b) + cos(a)sin(b)`.
fn expand_sin_add(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if let ExprNode::Add(ref children) = arena.node(inner).clone() {
        if children.len() < 2 {
            return None;
        }
        // Split into first term and rest.
        let a = children[0];
        let rest: smallvec::SmallVec<[ExprId; 6]> = children[1..].iter().copied().collect();
        let b = if rest.len() == 1 {
            rest[0]
        } else {
            arena.add(&rest)
        };

        // sin(a+b) = sin(a)*cos(b) + cos(a)*sin(b)
        let sin_a = arena.sin(a);
        let cos_b = arena.cos(b);
        let cos_a = arena.cos(a);
        let sin_b = arena.sin(b);
        let term1 = arena.mul(&[sin_a, cos_b]);
        let term2 = arena.mul(&[cos_a, sin_b]);
        Some(arena.add(&[term1, term2]))
    } else {
        None
    }
}

/// Expand `cos(inner)` when `inner` is an Add: `cos(a + b) → cos(a)cos(b) - sin(a)sin(b)`.
fn expand_cos_add(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if let ExprNode::Add(ref children) = arena.node(inner).clone() {
        if children.len() < 2 {
            return None;
        }
        let a = children[0];
        let rest: smallvec::SmallVec<[ExprId; 6]> = children[1..].iter().copied().collect();
        let b = if rest.len() == 1 {
            rest[0]
        } else {
            arena.add(&rest)
        };

        // cos(a+b) = cos(a)*cos(b) - sin(a)*sin(b)
        let cos_a = arena.cos(a);
        let cos_b = arena.cos(b);
        let sin_a = arena.sin(a);
        let sin_b = arena.sin(b);
        let term1 = arena.mul(&[cos_a, cos_b]);
        let term2 = arena.mul(&[sin_a, sin_b]);
        Some(arena.sub(term1, term2))
    } else {
        None
    }
}

/// Expand `sin(inner)` when `inner` is `Mul(n, arg)` with integer n ≥ 2.
///
/// Rewrites `sin(n·x)` as `sin(x)·cos((n-1)·x) + cos(x)·sin((n-1)·x)` and
/// recursively expands the result.
fn expand_sin_mul(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let children = match arena.node(inner).clone() {
        ExprNode::Mul(children) => children,
        _ => return None,
    };
    if children.len() != 2 {
        return None;
    }
    let n_ratio = arena.as_num(children[0])?.clone();
    if !n_ratio.is_integer() {
        return None;
    }
    let n_int: i64 = n_ratio.to_integer().try_into().ok()?;
    if !(2..=20).contains(&n_int) {
        return None;
    }
    let arg = children[1];
    // sin(n*x) = sin(x)*cos((n-1)*x) + cos(x)*sin((n-1)*x)
    let n_minus_1 = arena.int(n_int - 1);
    let rest = arena.mul(&[n_minus_1, arg]);
    let sin_a = arena.sin(arg);
    let cos_b = arena.cos(rest);
    let cos_a = arena.cos(arg);
    let sin_b = arena.sin(rest);
    let term1 = arena.mul(&[sin_a, cos_b]);
    let term2 = arena.mul(&[cos_a, sin_b]);
    let expanded = arena.add(&[term1, term2]);
    // Recursively expand remaining trig(k*x) terms
    Some(expand_trig(arena, expanded))
}

/// Expand `cos(inner)` when `inner` is `Mul(n, arg)` with integer n ≥ 2.
///
/// Rewrites `cos(n·x)` as `cos(x)·cos((n-1)·x) - sin(x)·sin((n-1)·x)` and
/// recursively expands the result.
fn expand_cos_mul(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let children = match arena.node(inner).clone() {
        ExprNode::Mul(children) => children,
        _ => return None,
    };
    if children.len() != 2 {
        return None;
    }
    let n_ratio = arena.as_num(children[0])?.clone();
    if !n_ratio.is_integer() {
        return None;
    }
    let n_int: i64 = n_ratio.to_integer().try_into().ok()?;
    if !(2..=20).contains(&n_int) {
        return None;
    }
    let arg = children[1];
    // cos(n*x) = cos(x)*cos((n-1)*x) - sin(x)*sin((n-1)*x)
    let n_minus_1 = arena.int(n_int - 1);
    let rest = arena.mul(&[n_minus_1, arg]);
    let cos_a = arena.cos(arg);
    let cos_b = arena.cos(rest);
    let sin_a = arena.sin(arg);
    let sin_b = arena.sin(rest);
    let term1 = arena.mul(&[cos_a, cos_b]);
    let term2 = arena.mul(&[sin_a, sin_b]);
    let expanded = arena.sub(term1, term2);
    // Recursively expand remaining trig(k*x) terms
    Some(expand_trig(arena, expanded))
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
    fn expand_sin_a_plus_b() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let sum = a.add(&[x, y]);
        let expr = a.sin(sum);
        let result = expand_trig(&mut a, expr);
        let s = display(&a, result);
        // sin(x+y) = sin(x)*cos(y) + cos(x)*sin(y)
        assert!(s.contains("sin(x)"), "should contain sin(x): {s}");
        assert!(s.contains("cos(y)"), "should contain cos(y): {s}");
        assert!(s.contains("cos(x)"), "should contain cos(x): {s}");
        assert!(s.contains("sin(y)"), "should contain sin(y): {s}");
    }

    #[test]
    fn expand_cos_a_plus_b() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let sum = a.add(&[x, y]);
        let expr = a.cos(sum);
        let result = expand_trig(&mut a, expr);
        let s = display(&a, result);
        // cos(x+y) = cos(x)*cos(y) - sin(x)*sin(y)
        assert!(s.contains("cos(x)"), "should contain cos(x): {s}");
        assert!(s.contains("cos(y)"), "should contain cos(y): {s}");
        assert!(s.contains("sin(x)"), "should contain sin(x): {s}");
        assert!(s.contains("sin(y)"), "should contain sin(y): {s}");
    }

    #[test]
    fn expand_sin_bare_x_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.sin(x);
        let result = expand_trig(&mut a, expr);
        assert_eq!(display(&a, result), "sin(x)");
    }

    #[test]
    fn expand_non_trig_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.exp(x);
        let result = expand_trig(&mut a, expr);
        assert_eq!(display(&a, result), "exp(x)");
    }

    #[test]
    fn expand_trig_inside_exp() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let sum = a.add(&[x, y]);
        let sin_sum = a.sin(sum);
        let expr = a.exp(sin_sum);
        let result = crate::simplify::trig_expand::expand_trig(&mut a, expr);
        let s = a.display(result).to_string();
        // sin(x+y) should be expanded even inside exp()
        assert!(
            !s.contains("sin(x + y)"),
            "trig expansion should work inside exp(): {s}"
        );
    }

    #[test]
    fn expand_trig_sin_2x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let mul_2x = a.mul(&[two, x]);
        let sin_2x = a.sin(mul_2x);
        let result = expand_trig(&mut a, sin_2x);
        // sin(2x) = 2*sin(x)*cos(x)
        let s = display(&a, result);
        assert!(
            s.contains("sin") && s.contains("cos"),
            "sin(2x) should expand to 2*sin(x)*cos(x), got: {s}"
        );
    }

    #[test]
    fn expand_trig_cos_2x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let mul_2x = a.mul(&[two, x]);
        let cos_2x = a.cos(mul_2x);
        let result = expand_trig(&mut a, cos_2x);
        let s = display(&a, result);
        assert!(
            s.contains("sin") || s.contains("cos"),
            "cos(2x) should expand, got: {s}"
        );
    }

    #[test]
    fn expand_trig_sin_3x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let mul_3x = a.mul(&[three, x]);
        let sin_3x = a.sin(mul_3x);
        let result = expand_trig(&mut a, sin_3x);
        let s = display(&a, result);
        // Should be fully expanded with no sin(2*x) or cos(2*x) remaining
        assert!(
            !s.contains("2*x") && !s.contains("3*x"),
            "sin(3x) should be fully expanded into sin(x) and cos(x) only, got: {s}"
        );
    }

    #[test]
    fn expand_trig_sin_bare_mul_not_integer() {
        // sin(pi*x) should not be expanded (pi is not an integer coefficient)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let pi = a.symbol("pi");
        let mul_pi_x = a.mul(&[pi, x]);
        let expr = a.sin(mul_pi_x);
        let result = expand_trig(&mut a, expr);
        let s = display(&a, result);
        // Should remain unexpanded
        assert!(
            !s.contains("cos"),
            "sin(pi*x) should not be expanded, got: {s}"
        );
    }
}
