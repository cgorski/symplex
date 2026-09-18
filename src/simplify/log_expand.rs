//! Logarithm expansion.
//!
//! Implements [`expand_log`], which applies logarithm properties:
//!
//! - `ln(a * b)` → `ln(a) + ln(b)`
//! - `ln(a / b)` → `ln(a) - ln(b)` (i.e., `ln(a * b^(-1))`)
//! - `ln(a^n)` → `n * ln(a)`
//!
//! These rules are always valid for positive real arguments.
//! [`expand_log`] applies them unconditionally (`force = true`);
//! [`expand_log_with`] with `force = false` only expands when every
//! factor is known positive (and, for `ln(a^n)`, `n` is known real)
//! through the assumption system.

use crate::base::arena::Arena;
use crate::base::assumptions::{AssumptionCache, Props};
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;
use num_traits::Signed;
use rustc_hash::FxHashMap;

/// Expand logarithmic expressions unconditionally.
///
/// Walks the expression bottom-up and applies log expansion rules
/// to `Ln` nodes whose arguments are products, quotients, or powers.
/// Equivalent to [`expand_log_with`] with `force = true`.
pub(crate) fn expand_log(arena: &mut Arena, expr: ExprId) -> ExprId {
    expand_log_with(arena, expr, true)
}

/// Expand logarithmic expressions, honouring the positivity guard unless
/// `force` is set.
///
/// # Branch reasoning
///
/// For complex arguments `ln(a·b) = ln a + ln b + 2πi·k` where `k ∈ {−1, 0, 1}`
/// depends on the arguments' phases, and `ln(a^n) = n·ln a` fails as
/// soon as `n·arg(a)` leaves `(−π, π]`.  Both identities are exact when
/// the arguments are positive reals, which is what the guard checks.
pub(crate) fn expand_log_with(arena: &mut Arena, expr: ExprId, force: bool) -> ExprId {
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    let mut assumptions = AssumptionCache::new();

    for &id in &post_order {
        let node = arena.node(id).clone();
        let expanded = match node {
            ExprNode::Ln(inner) => {
                let inner = cache.get(&inner).copied().unwrap_or(inner);
                expand_ln_node_guarded(arena, &mut assumptions, inner, force)
            }
            // Rebuild other nodes with expanded children.
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

/// Expand a single `ln(inner)` node, checking the positivity guard first
/// unless `force` is set.  Returns `ln(inner)` unchanged when the guard
/// rejects.
pub(crate) fn expand_ln_node_guarded(
    arena: &mut Arena,
    assumptions: &mut AssumptionCache,
    inner: ExprId,
    force: bool,
) -> ExprId {
    if !force {
        let ok = match arena.node(inner).clone() {
            // Every factor must be a positive real (then so is its
            // reciprocal, which the a^(-n) branch below relies on).
            ExprNode::Mul(ref children) => children
                .iter()
                .all(|&c| assumptions.query(arena, c, Props::POSITIVE) == Some(true)),
            ExprNode::Pow(base, exp) => {
                assumptions.query(arena, base, Props::POSITIVE) == Some(true)
                    && assumptions.query(arena, exp, Props::REAL) == Some(true)
            }
            _ => true,
        };
        if !ok {
            return arena.ln(inner);
        }
    }
    expand_ln_node(arena, inner)
}

/// Expand a single `ln(inner)` node.
fn expand_ln_node(arena: &mut Arena, inner: ExprId) -> ExprId {
    match arena.node(inner).clone() {
        // ln(a * b * ...) → ln(a) + ln(b) + ...
        ExprNode::Mul(ref children) => {
            let terms: Vec<ExprId> = children
                .iter()
                .map(|&child| {
                    // Check for negative exponent: a^(-n) contributes -ln(a^n)
                    if let ExprNode::Pow(base, exp) = arena.node(child).clone()
                        && let Some(r) = arena.as_num(exp)
                        && r.is_negative()
                    {
                        let pos_exp = {
                            let pos = -r.clone();
                            let nid = arena.intern_num(pos);
                            arena.intern(ExprNode::Num(nid))
                        };
                        let pos_pow = arena.pow(base, pos_exp);
                        let ln_pos = arena.ln(pos_pow);
                        return arena.neg(ln_pos);
                    }
                    arena.ln(child)
                })
                .collect();
            if terms.len() == 1 {
                terms[0]
            } else {
                arena.add(&terms)
            }
        }

        // ln(a^n) → n * ln(a)
        ExprNode::Pow(base, exp) => {
            let ln_base = arena.ln(base);
            arena.mul(&[exp, ln_base])
        }

        // Nothing to expand.
        _ => arena.ln(inner),
    }
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
    fn expand_ln_product() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let product = a.mul(&[x, y]);
        let expr = a.ln(product);
        let result = expand_log(&mut a, expr);
        let s = display(&a, result);
        // ln(x*y) → ln(x) + ln(y)
        assert!(s.contains("ln(x)") && s.contains("ln(y)"), "got: {s}");
    }

    #[test]
    fn expand_ln_power() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let expr = a.ln(x2);
        let result = expand_log(&mut a, expr);
        let s = display(&a, result);
        // ln(x^2) → 2*ln(x)
        assert!(s.contains("ln(x)") && s.contains("2"), "got: {s}");
    }

    #[test]
    fn expand_ln_quotient() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let quotient = a.div(x, y);
        let expr = a.ln(quotient);
        let result = expand_log(&mut a, expr);
        let s = display(&a, result);
        // ln(x/y) = ln(x * y^(-1)) → ln(x) - ln(y)
        assert!(s.contains("ln(x)") && s.contains("ln(y)"), "got: {s}");
    }

    #[test]
    fn expand_ln_bare_symbol_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.ln(x);
        let result = expand_log(&mut a, expr);
        assert_eq!(display(&a, result), "ln(x)");
    }

    #[test]
    fn expand_log_inside_sin() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let product = a.mul(&[x, y]);
        let ln_product = a.ln(product);
        let expr = a.sin(ln_product);
        let result = expand_log(&mut a, expr);
        let s = display(&a, result);
        // ln(x*y) should be expanded even inside sin()
        assert!(
            !s.contains("ln(x*y)"),
            "log expansion should work inside sin(): {s}"
        );
    }

    #[test]
    fn expand_ln_nested() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);
        // ln(x^2 * y) → first pass: ln(x^2) + ln(y) → second pass: 2*ln(x) + ln(y)
        let x2 = a.pow(x, two);
        let product = a.mul(&[x2, y]);
        let expr = a.ln(product);
        // Apply twice for full expansion (product then power).
        let pass1 = expand_log(&mut a, expr);
        let pass2 = expand_log(&mut a, pass1);
        let s = display(&a, pass2);
        assert!(s.contains("ln(x)") && s.contains("ln(y)"), "got: {s}");
    }
}
