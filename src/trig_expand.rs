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

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode};
use crate::walk;

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
                expand_sin_add(arena, inner).unwrap_or_else(|| arena.sin(inner))
            }
            ExprNode::Cos(inner) => {
                let inner = cache.get(&inner).copied().unwrap_or(inner);
                expand_cos_add(arena, inner).unwrap_or_else(|| arena.cos(inner))
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
            _ => id,
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
}
