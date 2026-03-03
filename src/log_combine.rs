//! Logarithm combination (inverse of log expansion).
//!
//! Implements [`log_combine`], which applies logarithm combination rules:
//!
//! - `ln(a) + ln(b)` → `ln(a * b)`
//! - `n * ln(a)` → `ln(a^n)`
//!
//! This is the inverse of [`expand_log`](crate::log_expand::expand_log).

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode};
use crate::walk;
use rustc_hash::FxHashMap;

/// Combine logarithmic expressions.
///
/// Walks the expression bottom-up and applies log combination rules:
///
/// - Sums containing multiple `ln` terms are combined into a single
///   logarithm of a product.
/// - Products of the form `n · ln(a)` are rewritten as `ln(a^n)`.
pub(crate) fn log_combine(arena: &mut Arena, expr: ExprId) -> ExprId {
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post_order {
        let node = arena.node(id).clone();
        let combined = match node {
            // ── n-ary product ──────────────────────────────────────────
            // If exactly one factor is `Ln(arg)` and there is at least one
            // other factor, rewrite `coeff · ln(arg)` → `ln(arg ^ coeff)`.
            ExprNode::Mul(ref children) => {
                let new: smallvec::SmallVec<[ExprId; 6]> = children
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();

                combine_mul_ln(arena, id, children, &new)
            }

            // ── n-ary sum ──────────────────────────────────────────────
            // Collect all `Ln(…)` children, combine into a single log of
            // a product, and keep non-log children untouched.
            ExprNode::Add(ref children) => {
                let new: smallvec::SmallVec<[ExprId; 6]> = children
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();

                combine_add_ln(arena, id, children, &new)
            }

            // ── rebuild other nodes with cached children ───────────────
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
            ExprNode::Ln(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                if ni == inner { id } else { arena.ln(ni) }
            }
            _ => id,
        };

        cache.insert(id, combined);
    }

    cache.get(&expr).copied().unwrap_or(expr)
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// If a `Mul` node has exactly one `Ln` factor among its children,
/// rewrite `coeff · ln(arg)` as `ln(arg ^ coeff)`.
///
/// Otherwise, rebuild the node with cached children (or return the
/// original id when nothing changed).
fn combine_mul_ln(arena: &mut Arena, id: ExprId, original: &[ExprId], new: &[ExprId]) -> ExprId {
    // Scan for Ln factors.
    let mut ln_index: Option<usize> = None;
    let mut ln_count: usize = 0;

    for (i, &child) in new.iter().enumerate() {
        if matches!(arena.node(child), ExprNode::Ln(_)) {
            ln_count += 1;
            if ln_count == 1 {
                ln_index = Some(i);
            }
        }
    }

    if ln_count == 1 && new.len() >= 2 {
        let idx = ln_index.unwrap();
        let ln_arg = match *arena.node(new[idx]) {
            ExprNode::Ln(inner) => inner,
            _ => unreachable!(),
        };

        // Build the coefficient from all non-Ln factors.
        let coeff_factors: smallvec::SmallVec<[ExprId; 6]> = new
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != idx)
            .map(|(_, &c)| c)
            .collect();

        let coeff = if coeff_factors.len() == 1 {
            coeff_factors[0]
        } else {
            arena.mul(&coeff_factors)
        };

        let powered = arena.pow(ln_arg, coeff);
        arena.ln(powered)
    } else if new == original {
        id
    } else {
        arena.mul(new)
    }
}

/// Scan the children of an `Add` node for `Ln(…)` terms and combine them
/// into a single `Ln(product)`.
///
/// Non-log children are kept as-is. If fewer than two `Ln` terms are
/// found, the node is returned unchanged (or rebuilt if children were
/// modified by the cache).
fn combine_add_ln(arena: &mut Arena, id: ExprId, original: &[ExprId], new: &[ExprId]) -> ExprId {
    // Partition children into ln-inner-arguments and everything else.
    let mut ln_inner_args: smallvec::SmallVec<[ExprId; 6]> = smallvec::SmallVec::new();
    let mut others: smallvec::SmallVec<[ExprId; 6]> = smallvec::SmallVec::new();

    for &child in new {
        if let ExprNode::Ln(inner) = *arena.node(child) {
            ln_inner_args.push(inner);
        } else {
            others.push(child);
        }
    }

    if ln_inner_args.len() >= 2 {
        // ln(a) + ln(b) + … → ln(a·b·…)
        let product = arena.mul(&ln_inner_args);
        let combined_ln = arena.ln(product);

        if others.is_empty() {
            combined_ln
        } else {
            others.push(combined_ln);
            arena.add(&others)
        }
    } else if new == original {
        id
    } else {
        arena.add(new)
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
    fn combine_two_logs() {
        // ln(x) + ln(y) → ln(x*y)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let lnx = a.ln(x);
        let lny = a.ln(y);
        let expr = a.add(&[lnx, lny]);
        let result = log_combine(&mut a, expr);
        let s = display(&a, result);
        assert!(s.contains("ln("), "expected ln(...), got: {s}");
        assert!(!s.contains('+'), "should not contain +, got: {s}");
        // The argument of the combined ln should be a product of x and y.
        assert!(s.contains('x') && s.contains('y'), "got: {s}");
    }

    #[test]
    fn combine_three_logs() {
        // ln(x) + ln(y) + ln(z) → ln(x*y*z)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let z = sym(&mut a, "z");
        let lnx = a.ln(x);
        let lny = a.ln(y);
        let lnz = a.ln(z);
        let expr = a.add(&[lnx, lny, lnz]);
        let result = log_combine(&mut a, expr);
        let s = display(&a, result);
        assert!(s.contains("ln("), "expected ln(...), got: {s}");
        assert!(!s.contains('+'), "should not contain +, got: {s}");
        assert!(
            s.contains('x') && s.contains('y') && s.contains('z'),
            "got: {s}"
        );
    }

    #[test]
    fn combine_coeff_log() {
        // 2*ln(x) → ln(x^2)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let lnx = a.ln(x);
        let expr = a.mul(&[two, lnx]);

        // The Mul-level rule fires, giving ln(x^2).
        let result = log_combine(&mut a, expr);
        let s = display(&a, result);
        assert!(s.contains("ln("), "expected ln(...), got: {s}");
        assert!(s.contains('x') && s.contains('2'), "got: {s}");
    }

    #[test]
    fn no_combine_single_log() {
        // ln(x) + 1 stays as is (only one log term).
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let lnx = a.ln(x);
        let one = a.int(1);
        let expr = a.add(&[lnx, one]);
        let result = log_combine(&mut a, expr);
        let s = display(&a, result);
        assert!(s.contains("ln(x)"), "ln(x) should remain, got: {s}");
        assert!(
            s.contains('+') || s.contains('1'),
            "1 should remain, got: {s}"
        );
    }

    #[test]
    fn combine_nested() {
        // ln(x) + ln(y) + z → ln(x*y) + z
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let z = sym(&mut a, "z");
        let lnx = a.ln(x);
        let lny = a.ln(y);
        let expr = a.add(&[lnx, lny, z]);
        let result = log_combine(&mut a, expr);
        let s = display(&a, result);
        // Should have exactly one ln(...) wrapping a product, plus z.
        assert!(s.contains("ln("), "expected ln(...), got: {s}");
        assert!(s.contains('z'), "z should remain, got: {s}");
        // The two logs should be combined, so there shouldn't be two ln(
        let ln_count = s.matches("ln(").count();
        assert_eq!(ln_count, 1, "expected 1 ln term, got {ln_count} in: {s}");
    }

    #[test]
    fn bare_symbol_unchanged() {
        // x + y → unchanged
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let expr = a.add(&[x, y]);
        let before = display(&a, expr);
        let result = log_combine(&mut a, expr);
        let after = display(&a, result);
        assert_eq!(before, after, "x + y should be unchanged");
    }
}
