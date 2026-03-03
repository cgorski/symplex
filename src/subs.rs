//! Structural substitution.
//!
//! This module provides [`subs`] (structural replacement) and
//! [`subs_map`] (simultaneous multi-replacement).
//!
//! # Design
//!
//! **Structural substitution** replaces exact node matches only — it
//! never performs algebraic reasoning.  `(1/x).subs(x², 1)` returns
//! `1/x` unchanged because `x²` does not appear as a node in `1/x`.
//! This is safe by construction: it can never produce mathematically
//! wrong results.  Use the future `.alg_subs()` for algebraic
//! substitution with documented caveats.
//!
//! **No recursive tree walks.**  All traversals use an explicit stack
//! ([`crate::walk::walk_and_rebuild`]).  Stack overflow is impossible
//! regardless of expression depth.

use rustc_hash::FxHashMap;

use crate::arena::Arena;
use crate::node::ExprId;

// ═══════════════════════════════════════════════════════════════════════════
// Structural substitution
// ═══════════════════════════════════════════════════════════════════════════

/// Replace every occurrence of `old` with `new` in the expression
/// rooted at `expr`.
///
/// This is **structural** substitution: only exact `ExprId` matches are
/// replaced.  The result is re-canonicalized through the normal
/// `Arena::add` / `Arena::mul` / etc. constructors, so like-term
/// collection and other canonical-form invariants are maintained.
///
/// Returns `expr` unchanged (same `ExprId`) if `old` does not appear
/// anywhere in the tree — no unnecessary allocation.
pub(crate) fn subs(arena: &mut Arena, expr: ExprId, old: ExprId, new: ExprId) -> ExprId {
    // Fast path: if old == new, nothing to do.
    if old == new {
        return expr;
    }
    // Fast path: if expr IS the thing we're replacing, return new.
    if expr == old {
        return new;
    }
    // Fast path: atoms that aren't the target can't contain it.
    if arena.node(expr).is_atom() {
        return expr;
    }

    crate::walk::walk_and_rebuild(arena, expr, &|_arena, id| {
        if id == old { Some(new) } else { None }
    })
}

/// Simultaneous substitution of multiple `(old, new)` pairs.
///
/// All replacements happen "at once" — earlier substitutions do not
/// affect later ones.  This avoids the order-dependence issues that
/// sequential substitution can cause.
pub(crate) fn subs_map(
    arena: &mut Arena,
    expr: ExprId,
    replacements: &[(ExprId, ExprId)],
) -> ExprId {
    if replacements.is_empty() {
        return expr;
    }

    let map: FxHashMap<ExprId, ExprId> = replacements.iter().copied().collect();

    // Fast path: expr itself is in the map.
    if let Some(&new) = map.get(&expr) {
        return new;
    }

    if arena.node(expr).is_atom() {
        return expr;
    }

    crate::walk::walk_and_rebuild(arena, expr, &|_arena, id| map.get(&id).copied())
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

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

    // ── Basic subs ──────────────────────────────────────────────────

    #[test]
    fn subs_symbol_for_number() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let expr = a.add(&[x, a.one]);
        let result = subs(&mut a, expr, x, three);
        assert_eq!(display(&a, result), "4");
    }

    #[test]
    fn subs_symbol_in_product() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);
        let expr = a.mul(&[two, x]);
        let result = subs(&mut a, expr, x, y);
        assert_eq!(display(&a, result), "2*y");
    }

    #[test]
    fn subs_in_pow() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let expr = a.pow(x, two);
        let three = a.int(3);
        let result = subs(&mut a, expr, x, three);
        assert_eq!(display(&a, result), "9");
    }

    #[test]
    fn subs_in_sin() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let expr = a.sin(x);
        let result = subs(&mut a, expr, x, y);
        assert_eq!(display(&a, result), "sin(y)");
    }

    #[test]
    fn subs_no_match_returns_same_id() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let expr = a.add(&[x, a.one]);
        let n99 = a.int(99);
        let result = subs(&mut a, expr, y, n99);
        assert_eq!(result, expr, "no match should return same ExprId");
    }

    #[test]
    fn subs_entire_expression() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let result = subs(&mut a, x, x, y);
        assert_eq!(result, y);
    }

    #[test]
    fn subs_old_equals_new_is_noop() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.add(&[x, a.one]);
        let result = subs(&mut a, expr, x, x);
        assert_eq!(result, expr);
    }

    // ── Nested substitution ─────────────────────────────────────────

    #[test]
    fn subs_in_nested_add_mul() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let three = a.int(3);

        // x^2 + 2*x + 1, substitute x → 3
        let x_sq = a.pow(x, two);
        let two_x = a.mul(&[two, x]);
        let expr = a.add(&[x_sq, two_x, a.one]);
        let result = subs(&mut a, expr, x, three);
        // 9 + 6 + 1 = 16
        assert_eq!(display(&a, result), "16");
    }

    #[test]
    fn subs_in_function_of_pow() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);

        // sin(x^2), substitute x → y
        let xp = a.pow(x, two);
        let expr = a.sin(xp);
        let result = subs(&mut a, expr, x, y);
        assert_eq!(display(&a, result), "sin(y^2)");
    }

    #[test]
    fn subs_replaces_all_occurrences() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");

        // x + x → y + y = 2*y
        let expr = a.add(&[x, x]);
        let result = subs(&mut a, expr, x, y);
        assert_eq!(display(&a, result), "2*y");
    }

    // ── Structural correctness ──────────────────────────────────────

    #[test]
    fn subs_does_not_match_algebraic_subexpressions() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let one = a.one;

        // (1/x).subs(x^2, 1) should return 1/x unchanged.
        // x^(-1) does not structurally contain x^2.
        let x_inv = a.pow(x, a.neg_one);
        let x_sq = a.pow(x, two);
        let result = subs(&mut a, x_inv, x_sq, one);
        assert_eq!(
            result, x_inv,
            "structural subs should not match x^2 in x^(-1)"
        );
    }

    #[test]
    fn subs_with_zero_evaluates() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let zero = a.zero;

        // (x*y).subs(y, 0) → x*0 → 0
        let expr = a.mul(&[x, y]);
        let result = subs(&mut a, expr, y, zero);
        assert_eq!(result, a.zero);
    }

    // ── Simultaneous substitution ───────────────────────────────────

    #[test]
    fn subs_map_simultaneous() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");

        // (x + y).subs({x→y, y→x}) should give y + x = x + y (commutative)
        let expr = a.add(&[x, y]);
        let result = subs_map(&mut a, expr, &[(x, y), (y, x)]);
        // Should be the same canonical form since x+y and y+x are equal.
        assert_eq!(result, expr, "swapping x↔y in x+y should give x+y");
    }

    #[test]
    fn subs_map_empty_is_noop() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.add(&[x, a.one]);
        let result = subs_map(&mut a, expr, &[]);
        assert_eq!(result, expr);
    }

    #[test]
    fn subs_map_multiple() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);
        let three = a.int(3);

        // (x + y).subs({x→2, y→3}) = 5
        let expr = a.add(&[x, y]);
        let result = subs_map(&mut a, expr, &[(x, two), (y, three)]);
        assert_eq!(display(&a, result), "5");
    }

    // ── Deep expressions (stack safety) ─────────────────────────────

    #[test]
    fn subs_deep_expression_no_stack_overflow() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");

        // Build a deeply nested expression: sin(sin(sin(...sin(x)...)))
        // 10,000 levels deep — tests that our iterative walker doesn't
        // stack-overflow.
        let mut expr = x;
        for _ in 0..10_000 {
            expr = a.sin(expr);
        }

        // Substitute x → y deep inside.
        let result = subs(&mut a, expr, x, y);

        // Verify it changed (not the same ExprId).
        assert_ne!(
            result, expr,
            "substitution should have changed the deep expression"
        );
        // NOTE: We don't test Display here because fmt_expr in
        // display.rs is still recursive and would overflow at this
        // depth.  That will be fixed when display is converted to an
        // iterative walker (tracked as a known issue).
    }

    #[test]
    fn subs_moderate_depth_display_works() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");

        // Moderate depth that the recursive display can still handle.
        let mut expr = x;
        for _ in 0..50 {
            expr = a.sin(expr);
        }

        let result = subs(&mut a, expr, x, y);
        let s = format!("{}", a.display(result));
        assert!(s.starts_with("sin("), "should still start with sin(");
        assert!(s.contains('y'), "should contain y after substitution");
        assert!(!s.contains('x'), "should not contain x after substitution");
    }
}
