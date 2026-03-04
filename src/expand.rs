//! Algebraic expansion.
//!
//! This module implements [`expand`], which distributes products over
//! sums and expands integer powers of sums.
//!
//! # What `expand` does
//!
//! - `a * (b + c)` → `a*b + a*c`
//! - `(a + b) * (c + d)` → `a*c + a*d + b*c + b*d`
//! - `(a + b)^n` for non-negative integer `n` → multinomial expansion
//! - Recursively expands nested products/powers of sums
//!
//! # What `expand` does NOT do
//!
//! - Does not factor, collect, or simplify
//! - Does not evaluate functions (`sin`, `cos`, etc.)
//! - Does not cancel common factors in fractions
//!
//! # Design
//!
//! Expansion is performed bottom-up using an explicit post-order
//! traversal (no recursion).  Each node is expanded after its children,
//! so by the time we reach a `Mul` or `Pow`, the children are already
//! in expanded form.

use smallvec::SmallVec;

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode};
use crate::walk;

/// Fully expand an expression: distribute products over sums and
/// expand integer powers of sums.
///
/// The result is a sum of products — no unexpanded `Mul(…, Add(…))`
/// or `Pow(Add(…), positive_int)` nodes remain.
pub(crate) fn expand(arena: &mut Arena, expr: ExprId) -> ExprId {
    // Bottom-up: expand children first, then handle the current node.
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache = rustc_hash::FxHashMap::<ExprId, ExprId>::default();

    for &id in &post_order {
        let node = arena.node(id).clone();
        let expanded = match node {
            // Add: expand each child, then re-add.
            ExprNode::Add(ref children) => {
                let new_children: SmallVec<[ExprId; 6]> = children
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();
                if new_children == *children {
                    id
                } else {
                    arena.add(&new_children)
                }
            }

            // Mul: expand each child, then distribute.
            ExprNode::Mul(ref children) => {
                let new_children: SmallVec<[ExprId; 6]> = children
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();
                expand_mul(arena, &new_children)
            }

            // Pow: if base is Add and exp is a non-negative integer,
            // expand via repeated multiplication.
            ExprNode::Pow(base, exp) => {
                let new_base = cache.get(&base).copied().unwrap_or(base);
                let new_exp = cache.get(&exp).copied().unwrap_or(exp);
                expand_pow(arena, new_base, new_exp)
            }

            // Neg: expand the inner, then negate.
            ExprNode::Neg(inner) => {
                let new_inner = cache.get(&inner).copied().unwrap_or(inner);
                if new_inner == inner {
                    id
                } else {
                    arena.neg(new_inner)
                }
            }

            // Unary functions: just rebuild with expanded child.
            ExprNode::Sin(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::sin),
            ExprNode::Cos(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::cos),
            ExprNode::Tan(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::tan),
            ExprNode::Exp(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::exp),
            ExprNode::Ln(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::ln),
            ExprNode::Abs(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::abs),
            ExprNode::Asin(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::asin),
            ExprNode::Acos(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::acos),
            ExprNode::Atan(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::atan),
            ExprNode::Atan2(y, x) => {
                let ny = cache.get(&y).copied().unwrap_or(y);
                let nx = cache.get(&x).copied().unwrap_or(x);
                if ny == y && nx == x {
                    id
                } else {
                    arena.atan2(ny, nx)
                }
            }
            ExprNode::Sinh(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::sinh),
            ExprNode::Cosh(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::cosh),
            ExprNode::Tanh(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::tanh),
            ExprNode::Asinh(inner) => {
                rebuild_unary_expanded(arena, id, inner, &cache, Arena::asinh)
            }
            ExprNode::Acosh(inner) => {
                rebuild_unary_expanded(arena, id, inner, &cache, Arena::acosh)
            }
            ExprNode::Atanh(inner) => {
                rebuild_unary_expanded(arena, id, inner, &cache, Arena::atanh)
            }
            ExprNode::Sign(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::sign),
            ExprNode::Not(inner) => rebuild_unary_expanded(arena, id, inner, &cache, Arena::not),

            // Boolean atoms: unchanged.
            ExprNode::BoolTrue | ExprNode::BoolFalse => id,

            // Relational operators: rebuild binary with expanded children.
            ExprNode::Gt(a, b) => {
                let na = cache.get(&a).copied().unwrap_or(a);
                let nb = cache.get(&b).copied().unwrap_or(b);
                if na == a && nb == b {
                    id
                } else {
                    arena.gt(na, nb)
                }
            }
            ExprNode::Ge(a, b) => {
                let na = cache.get(&a).copied().unwrap_or(a);
                let nb = cache.get(&b).copied().unwrap_or(b);
                if na == a && nb == b {
                    id
                } else {
                    arena.ge(na, nb)
                }
            }
            ExprNode::Eq_(a, b) => {
                let na = cache.get(&a).copied().unwrap_or(a);
                let nb = cache.get(&b).copied().unwrap_or(b);
                if na == a && nb == b {
                    id
                } else {
                    arena.eq_(na, nb)
                }
            }
            ExprNode::Ne(a, b) => {
                let na = cache.get(&a).copied().unwrap_or(a);
                let nb = cache.get(&b).copied().unwrap_or(b);
                if na == a && nb == b {
                    id
                } else {
                    arena.ne_(na, nb)
                }
            }

            // N-ary logical / piecewise: rebuild with expanded children.
            ExprNode::And(ref children) => {
                let new: SmallVec<[ExprId; 6]> = children
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();
                if new == *children {
                    id
                } else {
                    arena.and(&new)
                }
            }
            ExprNode::Or(ref children) => {
                let new: SmallVec<[ExprId; 6]> = children
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();
                if new == *children { id } else { arena.or(&new) }
            }
            ExprNode::Piecewise(ref pairs) => {
                let new: SmallVec<[(ExprId, ExprId); 3]> = pairs
                    .iter()
                    .map(|&(val, cond)| {
                        let nv = cache.get(&val).copied().unwrap_or(val);
                        let nc = cache.get(&cond).copied().unwrap_or(cond);
                        (nv, nc)
                    })
                    .collect();
                if new == *pairs {
                    id
                } else {
                    arena.intern(ExprNode::Piecewise(new))
                }
            }

            ExprNode::Derivative(body, var) => {
                let new_body = *cache.get(&body).unwrap_or(&body);
                let new_var = *cache.get(&var).unwrap_or(&var);
                if new_body == body && new_var == var {
                    id
                } else {
                    arena.intern(ExprNode::Derivative(new_body, new_var))
                }
            }
            ExprNode::Integral(body, var) => {
                let new_body = *cache.get(&body).unwrap_or(&body);
                let new_var = *cache.get(&var).unwrap_or(&var);
                if new_body == body && new_var == var {
                    id
                } else {
                    arena.intern(ExprNode::Integral(new_body, new_var))
                }
            }
            ExprNode::Apply(func_id, ref args) => {
                let new_args: smallvec::SmallVec<[crate::node::ExprId; 2]> =
                    args.iter().map(|&a| *cache.get(&a).unwrap_or(&a)).collect();
                if new_args == *args {
                    id
                } else {
                    arena.intern(ExprNode::Apply(func_id, new_args))
                }
            }
            // Keep the wildcard for truly inert nodes (atoms handled earlier)
            _ => id,
        };

        cache.insert(id, expanded);
    }

    cache.get(&expr).copied().unwrap_or(expr)
}

/// Helper for rebuilding unary nodes with expanded children.
#[inline]
fn rebuild_unary_expanded(
    arena: &mut Arena,
    id: ExprId,
    inner: ExprId,
    cache: &rustc_hash::FxHashMap<ExprId, ExprId>,
    ctor: fn(&mut Arena, ExprId) -> ExprId,
) -> ExprId {
    let new_inner = cache.get(&inner).copied().unwrap_or(inner);
    if new_inner == inner {
        id
    } else {
        ctor(arena, new_inner)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Mul expansion (distribution)
// ═══════════════════════════════════════════════════════════════════════════

/// Expand a product by distributing over any Add factors.
///
/// Given factors `[f₁, f₂, …, fₙ]`, if any `fᵢ` is an `Add`, we
/// distribute.  The distribution is done incrementally: start with a
/// running "partial product" (list of terms), and for each factor
/// either multiply it into every term (if the factor is not an Add)
/// or cross-multiply with all Add children.
///
/// Example: `(a + b) * (c + d)` → partial starts as `[a, b]`, then
/// crossed with `[c, d]` → `[a*c, a*d, b*c, b*d]`.
fn expand_mul(arena: &mut Arena, factors: &[ExprId]) -> ExprId {
    if factors.is_empty() {
        return arena.one;
    }
    if factors.len() == 1 {
        return factors[0];
    }

    // Separate numeric coefficient from symbolic factors.
    // (The canonical Mul may have a leading Num.)
    let mut coeff_factors: SmallVec<[ExprId; 4]> = SmallVec::new();
    let mut symbolic_factors: SmallVec<[ExprId; 6]> = SmallVec::new();

    for &f in factors {
        if let ExprNode::Num(_) = arena.node(f) {
            coeff_factors.push(f);
        } else {
            symbolic_factors.push(f);
        }
    }

    // Check if any symbolic factor is an Add.
    let has_add = symbolic_factors
        .iter()
        .any(|&f| matches!(arena.node(f), ExprNode::Add(_)));

    if !has_add {
        // No Add factors — nothing to distribute.  Just rebuild.
        let mut all: SmallVec<[ExprId; 6]> = SmallVec::new();
        all.extend_from_slice(&coeff_factors);
        all.extend_from_slice(&symbolic_factors);
        return arena.mul(&all);
    }

    // Incremental distribution.
    // `terms` holds the running list of partially-multiplied summands.
    let mut terms: Vec<SmallVec<[ExprId; 4]>> = vec![coeff_factors.clone()];

    for &factor in &symbolic_factors {
        let factor_node = arena.node(factor).clone();
        if let ExprNode::Add(add_children) = factor_node {
            // Cross-multiply: for each existing term, for each Add child,
            // produce a new term = existing_factors ++ [child].
            let mut new_terms: Vec<SmallVec<[ExprId; 4]>> = Vec::new();
            for existing in &terms {
                for &child in &add_children {
                    let mut combined = existing.clone();
                    combined.push(child);
                    new_terms.push(combined);
                }
            }
            terms = new_terms;
        } else {
            // Non-Add factor: append to every existing term.
            for term in &mut terms {
                term.push(factor);
            }
        }
    }

    // Build the final sum of products.
    let sum_terms: SmallVec<[ExprId; 6]> = terms
        .into_iter()
        .map(|factors| arena.mul(&factors))
        .collect();

    if sum_terms.len() == 1 {
        sum_terms[0]
    } else {
        arena.add(&sum_terms)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pow expansion
// ═══════════════════════════════════════════════════════════════════════════

/// Expand `base^exp` when base is an Add and exp is a non-negative
/// integer.  Otherwise returns the Pow unchanged.
fn expand_pow(arena: &mut Arena, base: ExprId, exp: ExprId) -> ExprId {
    // Only expand when exp is a positive integer.
    let exp_val = match arena.as_num(exp) {
        Some(r) if r.is_integer() => {
            let n: i64 = match r.to_integer().try_into() {
                Ok(n) => n,
                Err(_) => return arena.pow(base, exp),
            };
            n
        }
        _ => return arena.pow(base, exp),
    };

    // Only expand positive integer powers of sums.
    if exp_val <= 0 {
        return arena.pow(base, exp);
    }

    // Only expand if base is an Add.
    if !matches!(arena.node(base), ExprNode::Add(_)) {
        return arena.pow(base, exp);
    }

    // Guard: don't expand huge powers (could create exponential terms).
    let max_expand = arena.config.max_pow_exponent.min(20);
    if (exp_val as usize) > max_expand {
        return arena.pow(base, exp);
    }

    // Expand via repeated multiplication:
    // (a + b)^3 = (a + b) * (a + b) * (a + b)
    // We build the factors list and call expand_mul which distributes.
    let factors: Vec<ExprId> = (0..exp_val as usize).map(|_| base).collect();
    expand_mul(arena, &factors)
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

    // ── Basic distribution ──────────────────────────────────────────

    #[test]
    fn expand_no_add_is_noop() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let expr = a.mul(&[x, y]);
        let result = expand(&mut a, expr);
        assert_eq!(result, expr, "x*y should not change");
    }

    #[test]
    fn expand_atom_is_noop() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let result = expand(&mut a, x);
        assert_eq!(result, x);
    }

    #[test]
    fn expand_number_times_add() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);
        // 2*(x + y) — this is already distributed by canon_mul.
        // But let's build it via raw and expand.
        let sum = a.add(&[x, y]);
        let expr = a.mul(&[two, sum]);
        // Already distributed by Number*Add rule: 2*x + 2*y
        let s = display(&a, expr);
        assert!(
            s.contains('+'),
            "2*(x+y) should already be distributed, got: {s}"
        );
    }

    #[test]
    fn expand_symbol_times_add() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let z = sym(&mut a, "z");
        // z * (x + y) — NOT distributed by canon_mul (symbolic, not numeric)
        let sum = a.add(&[x, y]);
        let expr = a.mul(&[z, sum]);
        assert_eq!(display(&a, expr), "z*(x + y)");

        let result = expand(&mut a, expr);
        assert_eq!(display(&a, result), "x*z + y*z");
    }

    #[test]
    fn expand_add_times_add() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let u = sym(&mut a, "u");
        let v = sym(&mut a, "v");
        // (x + y) * (u + v) → x*u + x*v + y*u + y*v
        let sum1 = a.add(&[x, y]);
        let sum2 = a.add(&[u, v]);
        let expr = a.mul(&[sum1, sum2]);
        let result = expand(&mut a, expr);
        let s = display(&a, result);
        // Should have 4 terms.
        assert!(
            s.contains("x*u") || s.contains("u*x"),
            "should contain x*u term, got: {s}"
        );
    }

    // ── Power expansion ─────────────────────────────────────────────

    #[test]
    fn expand_x_plus_1_squared() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let sum = a.add(&[x, one]);
        let two = a.int(2);
        let expr = a.pow(sum, two);
        assert_eq!(display(&a, expr), "(x + 1)^2");

        let result = expand(&mut a, expr);
        assert_eq!(display(&a, result), "x^2 + 2*x + 1");
    }

    #[test]
    fn expand_x_plus_1_cubed() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let sum = a.add(&[x, one]);
        let three = a.int(3);
        let expr = a.pow(sum, three);

        let result = expand(&mut a, expr);
        assert_eq!(display(&a, result), "x^3 + 3*x^2 + 3*x + 1");
    }

    #[test]
    fn expand_x_plus_y_squared() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let sum = a.add(&[x, y]);
        let two = a.int(2);
        let expr = a.pow(sum, two);

        let result = expand(&mut a, expr);
        let s = display(&a, result);
        // (x + y)^2 = x^2 + 2*x*y + y^2
        assert!(s.contains("x^2"), "should contain x^2, got: {s}");
        assert!(s.contains("y^2"), "should contain y^2, got: {s}");
        assert!(
            s.contains("2*x*y") || s.contains("2*y*x"),
            "should contain 2*x*y, got: {s}"
        );
    }

    #[test]
    fn expand_pow_zero_is_one() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sum = a.add(&[x, a.one]);
        let zero = a.zero;
        let expr = a.pow(sum, zero);
        // (x+1)^0 = 1 (canonical)
        assert_eq!(display(&a, expr), "1");
        let result = expand(&mut a, expr);
        assert_eq!(display(&a, result), "1");
    }

    #[test]
    fn expand_pow_one_is_identity() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sum = a.add(&[x, a.one]);
        let one = a.one;
        let expr = a.pow(sum, one);
        // (x+1)^1 = x+1 (canonical)
        let result = expand(&mut a, expr);
        assert_eq!(display(&a, result), "x + 1");
    }

    #[test]
    fn expand_pow_negative_not_expanded() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sum = a.add(&[x, a.one]);
        let neg_two = a.int(-2);
        let expr = a.pow(sum, neg_two);
        // (x+1)^(-2) should NOT be expanded.
        let result = expand(&mut a, expr);
        assert_eq!(display(&a, result), "(x + 1)^(-2)");
    }

    #[test]
    fn expand_pow_non_add_base_not_expanded() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let expr = a.pow(x, three);
        // x^3 — base is not an Add, no expansion.
        let result = expand(&mut a, expr);
        assert_eq!(result, expr);
    }

    // ── Nested expansion ────────────────────────────────────────────

    #[test]
    fn expand_nested_mul_of_add() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let z = sym(&mut a, "z");
        // x * (y + z) * (x + 1)
        let sum1 = a.add(&[y, z]);
        let sum2 = a.add(&[x, a.one]);
        let expr = a.mul(&[x, sum1, sum2]);
        let result = expand(&mut a, expr);
        let s = display(&a, result);
        // Should be fully distributed: x*y*x + x*y + x*z*x + x*z
        // = x^2*y + x*y + x^2*z + x*z
        assert!(!s.contains('('), "should be fully expanded, got: {s}");
    }

    #[test]
    fn expand_product_of_expanded_power() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);
        // y * (x + 1)^2
        let sum = a.add(&[x, a.one]);
        let pow = a.pow(sum, two);
        let expr = a.mul(&[y, pow]);
        let result = expand(&mut a, expr);
        let s = display(&a, result);
        // y * (x^2 + 2x + 1) = x^2*y + 2*x*y + y
        assert!(!s.contains("^2)"), "power should be expanded, got: {s}");
        assert!(s.contains('y'), "should contain y, got: {s}");
    }

    // ── No-op cases ─────────────────────────────────────────────────

    #[test]
    fn expand_already_expanded_is_noop() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);
        // x^2 + 2*x*y + y^2 — already expanded.
        let x2 = a.pow(x, two);
        let y2 = a.pow(y, two);
        let two_xy = a.mul(&[two, x, y]);
        let expr = a.add(&[x2, two_xy, y2]);
        let result = expand(&mut a, expr);
        assert_eq!(result, expr, "already expanded should be unchanged");
    }

    #[test]
    fn expand_sum_of_symbols() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let expr = a.add(&[x, y]);
        let result = expand(&mut a, expr);
        assert_eq!(result, expr, "x + y should be unchanged");
    }

    // ── Idempotence ─────────────────────────────────────────────────

    #[test]
    fn expand_is_idempotent() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sum = a.add(&[x, a.one]);
        let two = a.int(2);
        let expr = a.pow(sum, two);

        let first = expand(&mut a, expr);
        let second = expand(&mut a, first);
        assert_eq!(first, second, "expand should be idempotent");
    }

    // ── Correctness via substitution ────────────────────────────────

    #[test]
    fn expand_x_plus_1_squared_evaluates_correctly() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sum = a.add(&[x, a.one]);
        let two = a.int(2);
        let expr = a.pow(sum, two);

        let expanded = expand(&mut a, expr);
        // Evaluate both at x=5: (5+1)^2 = 36
        let five = a.int(5);
        let orig_val = crate::subs::subs(&mut a, expr, x, five);
        let exp_val = crate::subs::subs(&mut a, expanded, x, five);
        assert_eq!(
            orig_val, exp_val,
            "expanded form should evaluate to same value"
        );
        assert_eq!(display(&a, orig_val), "36");
    }

    #[test]
    fn expand_x_plus_y_cubed_evaluates_correctly() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let sum = a.add(&[x, y]);
        let three = a.int(3);
        let expr = a.pow(sum, three);

        let expanded = expand(&mut a, expr);
        // Evaluate at x=2, y=3: (2+3)^3 = 125
        let two = a.int(2);
        let three_val = a.int(3);
        let orig_val = crate::subs::subs(&mut a, expr, x, two);
        let orig_val = crate::subs::subs(&mut a, orig_val, y, three_val);
        let exp_val = crate::subs::subs(&mut a, expanded, x, two);
        let exp_val = crate::subs::subs(&mut a, exp_val, y, three_val);
        assert_eq!(
            orig_val, exp_val,
            "expanded form should evaluate to same value"
        );
        assert_eq!(display(&a, orig_val), "125");
    }

    // ── Deep nesting (stack safety) ─────────────────────────────────

    #[test]
    fn expand_deep_no_overflow() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // Build deeply nested: sin(sin(sin(...(x+1)^2...)))
        let sum = a.add(&[x, a.one]);
        let two = a.int(2);
        let mut expr = a.pow(sum, two);
        for _ in 0..50 {
            expr = a.sin(expr);
        }
        // Should not overflow — uses iterative walker.
        let _result = expand(&mut a, expr);
    }

    #[test]
    fn expand_inside_sinh() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let sum = a.add(&[x, one]);
        let two = a.int(2);
        let sq = a.pow(sum, two);
        let expr = a.sinh(sq);
        // sinh((x+1)^2) → expand inner → sinh(1 + x^2 + 2*x)
        let result = expand(&mut a, expr);
        let s = display(&a, result);
        assert!(
            s.starts_with("sinh("),
            "should still be sinh(...), got: {s}"
        );
        assert!(
            s.contains("x^2"),
            "inner should be expanded to contain x^2, got: {s}"
        );
        assert!(
            !s.contains("(1 + x)^2"),
            "inner should no longer contain (1 + x)^2, got: {s}"
        );
    }

    #[test]
    fn expand_inside_derivative() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let one = arena.int(1);
        let sum = arena.add(&[x, one]); // x + 1
        let two = arena.int(2);
        let sq = arena.pow(sum, two); // (x+1)^2
        let deriv = arena.intern(crate::node::ExprNode::Derivative(sq, x));
        let expanded = expand(&mut arena, deriv);
        // The body should be expanded: x^2 + 2x + 1
        if let crate::node::ExprNode::Derivative(body, _) = arena.node(expanded) {
            // body should NOT be (x+1)^2 anymore
            assert_ne!(*body, sq, "body should be expanded inside Derivative");
        } else {
            panic!("result should still be a Derivative");
        }
    }
}
