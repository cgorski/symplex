//! Power simplification.
//!
//! Combines `x^a * x^b → x^(a+b)` for symbolic exponents (beyond
//! the numeric-only merging done during canonicalization).
//!
//! Also recognises `exp(a) * exp(b) → exp(a+b)` by decomposing
//! `exp(a)` as `E^a` before grouping by base.

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// Simplify powers: combine like bases in products, walking the full tree.
///
/// At every `Mul` node the factors are decomposed via [`as_base_exp`]
/// (extended to treat `exp(a)` as `E^a`), grouped by base, and
/// exponents summed.  The result is rebuilt through the canonical
/// constructors so all invariants are preserved.
pub(crate) fn powsimp(arena: &mut Arena, expr: ExprId) -> ExprId {
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post_order {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);

        let result = match arena.node(rebuilt).clone() {
            ExprNode::Mul(ref children) => powsimp_mul(arena, rebuilt, children),
            _ => rebuilt,
        };

        cache.insert(id, result);
    }

    cache.get(&expr).copied().unwrap_or(expr)
}

/// Decompose `id` into `(base, exponent)`, extending the standard
/// [`Arena::as_base_exp`] to also recognise `exp(a)` as `E^a`.
fn extended_base_exp(arena: &Arena, id: ExprId) -> (ExprId, ExprId) {
    match arena.node(id) {
        ExprNode::Pow(base, exp) => (*base, *exp),
        ExprNode::Exp(inner) => (arena.e_const, *inner),
        _ => (id, arena.one),
    }
}

/// Try to combine like bases inside a single `Mul` node.
///
/// Groups factors by base (using [`extended_base_exp`]), sums their
/// exponents, and rebuilds the product.  If nothing changed the
/// original node is returned as-is.
fn powsimp_mul(arena: &mut Arena, original: ExprId, children: &[ExprId]) -> ExprId {
    // Map: base → list of exponents to sum.
    // We use a Vec to preserve insertion order so the output is
    // deterministic (final sort is handled by canon_mul anyway).
    let mut bases: Vec<(ExprId, SmallVec<[ExprId; 4]>)> = Vec::new();
    let mut base_index: FxHashMap<ExprId, usize> = FxHashMap::default();

    for &child in children {
        let (base, exp) = extended_base_exp(arena, child);

        if let Some(&idx) = base_index.get(&base) {
            bases[idx].1.push(exp);
        } else {
            base_index.insert(base, bases.len());
            let mut exps = SmallVec::new();
            exps.push(exp);
            bases.push((base, exps));
        }
    }

    // Quick exit: if every base appeared exactly once, nothing to merge.
    let any_merged = bases.iter().any(|(_, exps)| exps.len() > 1);
    if !any_merged {
        return original;
    }

    // Rebuild factors with combined exponents.
    let mut factors: SmallVec<[ExprId; 6]> = SmallVec::new();
    for (base, exponents) in &bases {
        let combined_exp = if exponents.len() == 1 {
            exponents[0]
        } else {
            arena.add(exponents)
        };
        let factor = arena.pow(*base, combined_exp);
        factors.push(factor);
    }

    arena.mul(&factors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(arena: &mut Arena, name: &str) -> ExprId {
        arena.symbol(name)
    }

    fn display(arena: &Arena, id: ExprId) -> String {
        arena.display(id).to_string()
    }

    #[test]
    fn powsimp_combines_symbolic_exponents() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let a = sym(&mut arena, "a");
        let b = sym(&mut arena, "b");
        let xa = arena.pow(x, a);
        let xb = arena.pow(x, b);
        let expr = arena.mul(&[xa, xb]);

        let result = powsimp(&mut arena, expr);
        let s = display(&arena, result);
        // Should contain a+b in the exponent
        assert!(
            s.contains("a + b") || s.contains("b + a"),
            "powsimp should combine exponents: {s}"
        );
    }

    #[test]
    fn powsimp_combines_exp_calls() {
        let mut arena = Arena::new();
        let a = sym(&mut arena, "a");
        let b = sym(&mut arena, "b");
        let exp_a = arena.exp(a);
        let exp_b = arena.exp(b);
        let expr = arena.mul(&[exp_a, exp_b]);

        let result = powsimp(&mut arena, expr);
        let s = display(&arena, result);
        // exp(a)*exp(b) → E^(a+b) or exp(a+b)
        assert!(
            s.contains("a + b") || s.contains("b + a"),
            "powsimp should combine exp calls: {s}"
        );
    }

    #[test]
    fn powsimp_noop_when_bases_differ() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let y = sym(&mut arena, "y");
        let a = sym(&mut arena, "a");
        let b = sym(&mut arena, "b");
        let xa = arena.pow(x, a);
        let yb = arena.pow(y, b);
        let expr = arena.mul(&[xa, yb]);

        let result = powsimp(&mut arena, expr);
        // Nothing to combine — result should be structurally identical.
        assert_eq!(result, expr);
    }

    #[test]
    fn powsimp_atom_unchanged() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let result = powsimp(&mut arena, x);
        assert_eq!(result, x);
    }
}
