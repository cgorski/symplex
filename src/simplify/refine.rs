//! Assumption-aware expression refinement.
//!
//! [`refine`] simplifies an expression using known mathematical properties
//! of its subexpressions.  Unlike [`simplify`](super::simplify_engine),
//! which performs structural rewriting, `refine` uses the assumption system
//! to apply rewrites that are only valid under certain conditions.
//!
//! # Examples
//!
//! - `abs(x)` → `x` when `x` is known nonnegative
//! - `sqrt(x²)` → `x` when `x` is known positive
//! - `sign(x)` → `1` when `x` is known positive
//! - `floor(x)` → `x` when `x` is known integer
//!
//! # Architecture
//!
//! The implementation uses a **two-phase approach** to work within the
//! constraints of [`walk_and_rebuild`], whose transform closure receives
//! only `&Arena` (immutable):
//!
//! 1. **Pre-compute**: walk the expression in post-order, querying the
//!    [`AssumptionCache`] for each subexpression and storing results in a
//!    `FxHashMap<ExprId, CachedProps>`.
//! 2. **Rewrite**: call `walk_and_rebuild` with a closure that reads from
//!    the pre-populated cache (immutable lookups only) and returns
//!    replacement nodes for refinable subexpressions.
//!
//! Rewrites that need to **create new nodes** (e.g., `abs(x) → -x` needs
//! `arena.neg()`) are handled in a separate mutable pass.

use num_bigint::BigInt;
use num_rational::Ratio;

use rustc_hash::FxHashMap;

use crate::base::arena::Arena;
use crate::base::assumptions::{AssumptionCache, Props};
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;

// ═══════════════════════════════════════════════════════════════════════════
// Cached property snapshot
// ═══════════════════════════════════════════════════════════════════════════

/// Pre-computed property snapshot for a single subexpression.
#[derive(Clone, Debug, Default)]
struct CachedProps {
    is_positive: Option<bool>,
    is_nonneg: Option<bool>,
    is_integer: Option<bool>,
    is_zero: Option<bool>,
    is_even: Option<bool>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Public entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Refine an expression using assumptions.
///
/// Performs assumption-aware simplification by querying the
/// [`AssumptionCache`] for mathematical properties of each
/// subexpression and applying rewrites that are valid under those
/// assumptions.
///
/// This is the full pipeline: immutable rewrites (identity replacements)
/// followed by mutable rewrites (rewrites that create new nodes).
pub(crate) fn refine_full(
    arena: &mut Arena,
    assumptions: &mut AssumptionCache,
    root: ExprId,
) -> ExprId {
    tracing::info!("refine: entry");
    // Phase A: immutable rewrites via walk_and_rebuild.
    let after_immut = refine_immutable(arena, assumptions, root);
    // Phase B: mutable rewrites for cases that need new node creation.
    let result = refine_mutable(arena, assumptions, after_immut);
    if result != root {
        tracing::debug!("refine: expression was modified");
    } else {
        tracing::debug!("refine: expression unchanged");
    }
    result
}

/// Phase A: rewrites that only return existing ExprIds (no new nodes).
///
/// Uses `walk_and_rebuild` which takes `&Arena` in the closure.
fn refine_immutable(arena: &mut Arena, assumptions: &mut AssumptionCache, root: ExprId) -> ExprId {
    // ── Phase 1: pre-compute assumptions for every subexpression ────
    let post_order = walk::post_order_ids(arena, root);
    let mut prop_cache: FxHashMap<ExprId, CachedProps> = FxHashMap::default();

    for &id in &post_order {
        let props = CachedProps {
            is_positive: assumptions.query(arena, id, Props::POSITIVE),
            is_nonneg: assumptions.query(arena, id, Props::NONNEGATIVE),
            is_integer: assumptions.query(arena, id, Props::INTEGER),
            is_zero: assumptions.query(arena, id, Props::ZERO),
            is_even: assumptions.query(arena, id, Props::EVEN),
        };
        prop_cache.insert(id, props);
    }

    // ── Phase 2: walk and rewrite ──────────────────────────────────
    walk::walk_and_rebuild(arena, root, &|arena, id| {
        refine_node_immut(arena, id, &prop_cache)
    })
}

/// Attempt an immutable rewrite for a single node.
///
/// Returns `Some(replacement_id)` if a rewrite fires, `None` otherwise.
/// This function must NOT create new arena nodes — it can only return
/// existing `ExprId`s or children of the current node.
fn refine_node_immut(
    arena: &Arena,
    id: ExprId,
    prop_cache: &FxHashMap<ExprId, CachedProps>,
) -> Option<ExprId> {
    match arena.node(id).clone() {
        // ── abs(x) where x ≥ 0  →  x ──────────────────────────────
        ExprNode::Abs(inner) => {
            let props = prop_cache.get(&inner)?;
            if props.is_nonneg == Some(true) {
                tracing::debug!("refine: abs(x) -> x (nonnegative)");
                return Some(inner);
            }
            // abs(x) where x < 0 needs arena.neg() — handled in mutable pass.
            None
        }

        // ── sign(x) where x > 0  →  1 ─────────────────────────────
        ExprNode::Sign(inner) => {
            let props = prop_cache.get(&inner)?;
            if props.is_positive == Some(true) {
                tracing::debug!("refine: sign(x) -> 1 (positive)");
                return Some(arena.one);
            }
            if props.is_zero == Some(true) {
                tracing::debug!("refine: sign(x) -> 0 (zero)");
                return Some(arena.zero);
            }
            // sign(x) where x < 0 needs arena.neg_one — handled in mutable pass.
            None
        }

        // ── floor(x) where x ∈ ℤ  →  x ────────────────────────────
        ExprNode::Floor(inner) => {
            let props = prop_cache.get(&inner)?;
            if props.is_integer == Some(true) {
                tracing::debug!("refine: floor(x) -> x (integer)");
                return Some(inner);
            }
            None
        }

        // ── ceiling(x) where x ∈ ℤ  →  x ──────────────────────────
        ExprNode::Ceiling(inner) => {
            let props = prop_cache.get(&inner)?;
            if props.is_integer == Some(true) {
                tracing::debug!("refine: ceiling(x) -> x (integer)");
                return Some(inner);
            }
            None
        }

        // ── sqrt(x²) patterns ──────────────────────────────────────
        // Pow(Pow(x, 2), 1/2) where x > 0  →  x
        ExprNode::Pow(base, exp) => refine_pow_immut(arena, base, exp, prop_cache),

        _ => None,
    }
}

/// Refine Pow nodes (immutable pass).
///
/// Handles:
/// - `sqrt(x²)` where x > 0 → x
/// - `(-1)^(2k)` where k is integer → 1  (via even exponent)
fn refine_pow_immut(
    arena: &Arena,
    base: ExprId,
    exp: ExprId,
    prop_cache: &FxHashMap<ExprId, CachedProps>,
) -> Option<ExprId> {
    let half = Ratio::new(BigInt::from(1), BigInt::from(2));
    let two = Ratio::from(BigInt::from(2));

    // ── Pattern: Pow(Pow(inner, 2), 1/2) = sqrt(x²) ───────────────
    // Only applies when exponent is literally 1/2.
    if let Some(exp_num) = arena.as_num(exp) {
        if *exp_num == half {
            if let ExprNode::Pow(inner_base, inner_exp) = arena.node(base).clone() {
                if let Some(inner_exp_num) = arena.as_num(inner_exp) {
                    if *inner_exp_num == two {
                        // sqrt(x²) where x > 0 → x
                        if let Some(inner_props) = prop_cache.get(&inner_base) {
                            if inner_props.is_positive == Some(true) {
                                tracing::debug!("refine: sqrt(x²) -> x (positive)");
                                return Some(inner_base);
                            }
                        }
                        // sqrt(x²) where x ∈ ℝ → abs(x) — needs arena.abs(),
                        // handled in mutable pass.
                    }
                }
            }
        }
    }

    // ── Pattern: base^exp where exp is known even → simplify ───────
    // Special case: (-1)^(even) → 1
    if let Some(exp_props) = prop_cache.get(&exp) {
        if exp_props.is_even == Some(true) {
            if let Some(base_num) = arena.as_num(base) {
                if *base_num == Ratio::from(BigInt::from(-1)) {
                    tracing::debug!("refine: (-1)^(even) -> 1");
                    return Some(arena.one);
                }
            }
        }
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase B: mutable rewrites (need to create new nodes)
// ═══════════════════════════════════════════════════════════════════════════

/// Phase B: rewrites that need `&mut Arena` to create new nodes.
///
/// Runs a post-order traversal and applies mutable rewrites directly.
fn refine_mutable(arena: &mut Arena, assumptions: &mut AssumptionCache, root: ExprId) -> ExprId {
    let post_order = walk::post_order_ids(arena, root);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post_order {
        let node = arena.node(id).clone();
        let replacement = match node {
            // ── abs(x) where x < 0  →  -x ─────────────────────────
            ExprNode::Abs(inner) => {
                let inner = cache.get(&inner).copied().unwrap_or(inner);
                let is_neg = assumptions.query(arena, inner, Props::NEGATIVE);
                if is_neg == Some(true) {
                    tracing::debug!("refine: abs(x) -> -x (negative)");
                    Some(arena.neg(inner))
                } else {
                    // If children changed, rebuild; otherwise keep.
                    if cache.contains_key(&inner) {
                        Some(arena.abs(inner))
                    } else {
                        None
                    }
                }
            }

            // ── sign(x) where x < 0  →  -1 ────────────────────────
            ExprNode::Sign(inner) => {
                let inner = cache.get(&inner).copied().unwrap_or(inner);
                let is_neg = assumptions.query(arena, inner, Props::NEGATIVE);
                if is_neg == Some(true) {
                    tracing::debug!("refine: sign(x) -> -1 (negative)");
                    Some(arena.neg_one)
                } else if cache.contains_key(&inner) {
                    Some(arena.sign(inner))
                } else {
                    None
                }
            }

            // ── sqrt(x²) where x ∈ ℝ  →  abs(x) ──────────────────
            ExprNode::Pow(base, exp) => {
                let base = cache.get(&base).copied().unwrap_or(base);
                let exp = cache.get(&exp).copied().unwrap_or(exp);

                let half = Ratio::new(BigInt::from(1), BigInt::from(2));
                let two = Ratio::from(BigInt::from(2));

                if let Some(exp_num) = arena.as_num(exp) {
                    if *exp_num == half {
                        if let ExprNode::Pow(inner_base, inner_exp) = arena.node(base).clone() {
                            if let Some(ie_num) = arena.as_num(inner_exp) {
                                if *ie_num == two {
                                    let is_real = assumptions.query(arena, inner_base, Props::REAL);
                                    let is_positive =
                                        assumptions.query(arena, inner_base, Props::POSITIVE);

                                    // Positive was already handled in immutable pass.
                                    // Here we handle the real-but-not-positive case.
                                    if is_real == Some(true) && is_positive != Some(true) {
                                        tracing::debug!("refine: sqrt(x²) -> abs(x) (real)");
                                        return arena.abs(inner_base);
                                    }
                                }
                            }
                        }
                    }
                }

                // Rebuild if children changed.
                if cache.contains_key(&base) || cache.contains_key(&exp) {
                    Some(arena.pow(base, exp))
                } else {
                    None
                }
            }

            _ => None,
        };

        if let Some(new_id) = replacement {
            cache.insert(id, new_id);
        }
    }

    cache.get(&root).copied().unwrap_or(root)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::assumptions::Assumption;

    /// Shared context for all refine tests so that expressions from
    /// different helpers can be combined without cross-context panics.
    fn tctx() -> crate::api::context::Context {
        crate::api::context::Context::new()
    }

    /// Helper: create a variable with assumptions, returning (context, var_ex).
    fn var_with(
        ctx: &crate::api::context::Context,
        name: &str,
        assumption: Assumption,
    ) -> crate::api::expr::Ex {
        let v = ctx.symbol(name);
        v.assume(assumption)
    }

    /// Helper: run refine on an expression and return the result.
    fn do_refine(expr: &crate::api::expr::Ex) -> crate::api::expr::Ex {
        let mut inner = expr.inner.write();
        // Split borrow: destructure ContextInner so we can pass &mut arena
        // and &mut assumptions (via MutexGuard) simultaneously without
        // conflicting borrows through the same parent struct.
        let crate::api::context::ContextInner {
            ref mut arena,
            ref assumptions,
            ..
        } = *inner;
        let mut assumptions_guard = assumptions.lock();
        let result_id = refine_full(arena, &mut assumptions_guard, expr.id());
        drop(assumptions_guard);
        drop(inner);
        expr.wrap(result_id)
    }

    // ── abs ────────────────────────────────────────────────────────

    #[test]
    fn refine_abs_positive() {
        let ctx = crate::api::context::Context::new();
        let x = var_with(&ctx, "x_ref_apos", Assumption::Positive);
        let expr = x.abs();
        let result = do_refine(&expr);
        assert_eq!(
            result.id(),
            x.id(),
            "abs(x) should refine to x when positive"
        );
    }

    #[test]
    fn refine_abs_nonneg() {
        let ctx = tctx();
        let x = var_with(&ctx, "x_ref_ann", Assumption::NonNegative);
        let expr = x.abs();
        let result = do_refine(&expr);
        assert_eq!(result.id(), x.id(), "abs(x) should refine to x when nonneg");
    }

    #[test]
    fn refine_abs_negative() {
        let ctx = tctx();
        let x = var_with(&ctx, "x_ref_aneg", Assumption::Negative);
        let expr = x.abs();
        let result = do_refine(&expr);
        // abs(x) when x < 0 → -x
        let neg_x = -&x;
        assert_eq!(
            result.id(),
            neg_x.id(),
            "abs(x) should refine to -x when negative"
        );
    }

    #[test]
    fn refine_abs_unknown() {
        let ctx = tctx();
        let x = ctx.symbol("x_ref_aunk");
        let expr = x.abs();
        let result = do_refine(&expr);
        assert_eq!(
            result.id(),
            expr.id(),
            "abs(x) should be unchanged with no assumptions"
        );
    }

    // ── sign ───────────────────────────────────────────────────────

    #[test]
    fn refine_sign_positive() {
        let ctx = tctx();
        let x = var_with(&ctx, "x_ref_spos", Assumption::Positive);
        let expr = x.sign();
        let result = do_refine(&expr);
        let one = ctx.int(1);
        assert_eq!(result.id(), one.id(), "sign(x) -> 1 when positive");
    }

    #[test]
    fn refine_sign_negative() {
        let ctx = tctx();
        let x = var_with(&ctx, "x_ref_sneg", Assumption::Negative);
        let expr = x.sign();
        let result = do_refine(&expr);
        let neg_one = ctx.int(-1);
        assert_eq!(result.id(), neg_one.id(), "sign(x) -> -1 when negative");
    }

    #[test]
    fn refine_sign_zero() {
        let ctx = tctx();
        let x = var_with(&ctx, "x_ref_szero", Assumption::Zero);
        let expr = x.sign();
        let result = do_refine(&expr);
        let zero = ctx.int(0);
        assert_eq!(result.id(), zero.id(), "sign(x) -> 0 when zero");
    }

    #[test]
    fn refine_sign_unknown() {
        let ctx = tctx();
        let x = ctx.symbol("x_ref_sunk");
        let expr = x.sign();
        let result = do_refine(&expr);
        assert_eq!(
            result.id(),
            expr.id(),
            "sign(x) should be unchanged with no assumptions"
        );
    }

    // ── floor / ceiling ────────────────────────────────────────────

    #[test]
    fn refine_floor_integer() {
        let ctx = tctx();
        let x = var_with(&ctx, "x_ref_flint", Assumption::Integer);
        let expr = x.floor();
        let result = do_refine(&expr);
        assert_eq!(result.id(), x.id(), "floor(x) -> x when integer");
    }

    #[test]
    fn refine_floor_unknown() {
        let ctx = tctx();
        let x = ctx.symbol("x_ref_flunk");
        let expr = x.floor();
        let result = do_refine(&expr);
        assert_eq!(
            result.id(),
            expr.id(),
            "floor(x) unchanged with no assumptions"
        );
    }

    #[test]
    fn refine_ceiling_integer() {
        let ctx = crate::api::context::Context::new();
        let x = var_with(&ctx, "x_ref_ceint", Assumption::Integer);
        let expr = x.ceiling();
        let result = do_refine(&expr);
        assert_eq!(result.id(), x.id(), "ceiling(x) -> x when integer");
    }

    // ── sqrt(x²) ──────────────────────────────────────────────────

    #[test]
    fn refine_sqrt_x_squared_positive() {
        let ctx = tctx();
        let x = var_with(&ctx, "x_ref_sqpos", Assumption::Positive);
        let x2 = x.powi(2);
        let expr = x2.sqrt();
        let result = do_refine(&expr);
        assert_eq!(result.id(), x.id(), "sqrt(x²) -> x when positive");
    }

    #[test]
    fn refine_sqrt_x_squared_real() {
        let ctx = tctx();
        let x = var_with(&ctx, "x_ref_sqreal", Assumption::Real);
        let x2 = x.powi(2);
        let expr = x2.sqrt();
        let result = do_refine(&expr);
        let abs_x = x.abs();
        assert_eq!(result.id(), abs_x.id(), "sqrt(x²) -> abs(x) when real");
    }

    #[test]
    fn refine_sqrt_x_squared_unknown() {
        let ctx = tctx();
        let x = ctx.symbol("x_ref_squnk");
        let x2 = x.powi(2);
        let expr = x2.sqrt();
        let result = do_refine(&expr);
        assert_eq!(
            result.id(),
            expr.id(),
            "sqrt(x²) unchanged with no assumptions"
        );
    }

    // ── (-1)^(even) ────────────────────────────────────────────────

    #[test]
    fn refine_neg_one_to_even_power() {
        let ctx = tctx();
        let n = var_with(&ctx, "n_ref_even", Assumption::Even);
        let neg_one = ctx.int(-1);
        let expr = neg_one.pow(&n);
        let result = do_refine(&expr);
        let one = ctx.int(1);
        assert_eq!(result.id(), one.id(), "(-1)^(even n) -> 1");
    }

    // ── compound expression ────────────────────────────────────────

    #[test]
    fn refine_abs_sum_of_positives() {
        let ctx = tctx();
        // abs(x + y) where both x, y are positive should refine to x + y
        // because the assumption cache infers x + y is positive.
        let x = var_with(&ctx, "x_ref_cmpd1", Assumption::Positive);
        let y = var_with(&ctx, "y_ref_cmpd1", Assumption::Positive);
        let sum = &x + &y;
        let expr = sum.abs();
        let result = do_refine(&expr);
        assert_eq!(result.id(), sum.id(), "abs(x+y) -> x+y when both positive");
    }

    // ── no-op (expression with nothing to refine) ──────────────────

    #[test]
    fn refine_no_change() {
        let ctx = tctx();
        let x = ctx.symbol("x_ref_noop");
        let expr = &x + &ctx.int(1);
        let result = do_refine(&expr);
        assert_eq!(
            result.id(),
            expr.id(),
            "x + 1 should be unchanged by refine"
        );
    }

    // ── known numeric constants ────────────────────────────────────

    #[test]
    fn refine_sign_of_literal() {
        let ctx = tctx();
        // sign(5) — 5 is known positive in the assumption system.
        let five = ctx.int(5);
        let expr = five.sign();
        let result = do_refine(&expr);
        let one = ctx.int(1);
        assert_eq!(result.id(), one.id(), "sign(5) -> 1");
    }

    #[test]
    fn refine_floor_of_integer_literal() {
        let ctx = tctx();
        // floor(3) — 3 is known integer.
        let three = ctx.int(3);
        let expr = three.floor();
        let result = do_refine(&expr);
        assert_eq!(result.id(), three.id(), "floor(3) -> 3");
    }
}
