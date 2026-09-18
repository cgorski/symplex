//! Power simplification.
//!
//! Three complementary strategies (matching SymPy's `powsimp` modes):
//!
//! - **[`powsimp`]** (`combine='exp'`): Combines like bases by summing
//!   exponents: `x^a * x^b → x^(a+b)`.  Also recognises
//!   `exp(a) * exp(b) → exp(a+b)`.
//!
//! - **[`powsimp_base`]** (`combine='base'`): Combines like exponents by
//!   multiplying bases: `a^e * b^e → (a*b)^e` when both `a` and `b` are
//!   nonnegative rationals.  Handles `√2·√3 → √6`.
//!
//! - **[`powdenest`]**: Distributes integer powers over products:
//!   `(a·b)^n → a^n · b^n` when safe (all factors nonneg or exponent is
//!   integer).  Handles `(½·√5)² → ¼·5 = 5/4`.

use crate::base::arena::Arena;
use crate::base::assumptions::{AssumptionCache, Props};
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::Signed;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// Simplify powers: combine like bases in products, walking the full tree.
///
/// At every `Mul` node the factors are decomposed via [`as_base_exp`](crate::base::arena::Arena::as_base_exp)
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

// ═══════════════════════════════════════════════════════════════════════════
// powdenest — distribute powers over products
// ═══════════════════════════════════════════════════════════════════════════

/// Distribute powers over products: `(a·b·c)^n → a^n · b^n · c^n`.
///
/// This is safe (no branch-cut issues) when:
/// - The exponent is an integer, OR
/// - All factors of the base are nonnegative rationals.
///
/// The result passes through [`canon_pow`](crate::base::canon::canon_pow),
/// so e.g. `(½·√5)^2` becomes `(½)^2 · (√5)^2 = ¼ · 5 = 5/4`.
///
/// Walks the full expression tree bottom-up.
pub(crate) fn powdenest(arena: &mut Arena, expr: ExprId) -> ExprId {
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post_order {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);

        let result = match arena.node(rebuilt).clone() {
            ExprNode::Pow(base, exp) => powdenest_pow(arena, rebuilt, base, exp),
            _ => rebuilt,
        };

        cache.insert(id, result);
    }

    cache.get(&expr).copied().unwrap_or(expr)
}

/// Full power denesting, assumption-aware.
///
/// In addition to the product distribution of [`powdenest`], this handles
/// nested powers and square roots of squares:
///
/// | Rewrite                    | Condition (any of)                                   |
/// |----------------------------|------------------------------------------------------|
/// | `(a·b)^e → a^e·b^e`        | `e ∈ ℤ`; all factors but at most one known non-negative; `force` |
/// | `(x^a)^b → x^(a·b)`        | `b ∈ ℤ`; `x > 0` and `a` real; `force`               |
/// | `√(x²) → x`                | `x ≥ 0`; `force`                                     |
/// | `√(x²) → ∣x∣`              | `x` real (not known non-real)                        |
///
/// # Branch reasoning
///
/// `(x^a)^b = exp(b·Log(exp(a·Log x)))` equals `x^(ab) = exp(ab·Log x)`
/// exactly when `Log(exp(a·Log x)) = a·Log x`, i.e. `Im(a·Log x) ∈ (−π, π]`
/// — guaranteed for `x > 0` and real `a` — or when `b` is an integer
/// (integer powers of `exp(z)` are `exp(bz)` regardless of branch).
/// `√(x²) = |x|` needs `x` real: for `x = i`, `√(i²) = i ≠ |i| = 1`.
///
/// With `force = true` every symbol is treated as positive (like SymPy's
/// `powdenest(force=True)`), so all rewrites fire unconditionally.
pub(crate) fn powdenest_with(arena: &mut Arena, expr: ExprId, force: bool) -> ExprId {
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    let mut assumptions = AssumptionCache::new();

    for &id in &post_order {
        let rebuilt = walk::rebuild_with_cache(arena, id, &cache);

        let result = match arena.node(rebuilt).clone() {
            ExprNode::Pow(base, exp) => {
                let distributed = if force {
                    powdenest_pow_forced(arena, rebuilt, base, exp)
                } else {
                    powdenest_pow(arena, rebuilt, base, exp)
                };
                if distributed != rebuilt {
                    distributed
                } else if let ExprNode::Mul(children) = arena.node(base).clone() {
                    // Assumption-aware factor check: all factors but at most one
                    // known non-negative (see `expand::at_most_one_non_nonneg`).
                    if crate::transforms::expand::at_most_one_non_nonneg(
                        arena,
                        &mut assumptions,
                        &children,
                    ) {
                        powdenest_pow_forced(arena, rebuilt, base, exp)
                    } else {
                        rebuilt
                    }
                } else {
                    denest_pow_pow(arena, &mut assumptions, rebuilt, base, exp, force)
                }
            }
            _ => rebuilt,
        };

        cache.insert(id, result);
    }

    cache.get(&expr).copied().unwrap_or(expr)
}

/// `(x^a)^b` / `√(x²)` handling for [`powdenest_with`].
fn denest_pow_pow(
    arena: &mut Arena,
    assumptions: &mut AssumptionCache,
    original: ExprId,
    base: ExprId,
    exp: ExprId,
    force: bool,
) -> ExprId {
    let (inner_base, inner_exp) = match arena.node(base) {
        ExprNode::Pow(b, e) => (*b, *e),
        _ => return original,
    };

    let half = Ratio::new(BigInt::from(1), BigInt::from(2));
    let two = Ratio::from_integer(BigInt::from(2));
    let is_sqrt_of_square =
        arena.as_num(exp) == Some(&half) && arena.as_num(inner_exp) == Some(&two);

    if is_sqrt_of_square {
        if force || assumptions.query(arena, inner_base, Props::NONNEGATIVE) == Some(true) {
            return inner_base;
        }
        if assumptions.query(arena, inner_base, Props::REAL) != Some(false) {
            return arena.abs(inner_base);
        }
        return original;
    }

    let outer_integer = arena.as_num(exp).is_some_and(|r| r.is_integer())
        || assumptions.query(arena, exp, Props::INTEGER) == Some(true);
    let base_positive_real_exp = assumptions.query(arena, inner_base, Props::POSITIVE)
        == Some(true)
        && assumptions.query(arena, inner_exp, Props::REAL) == Some(true);

    if force || outer_integer || base_positive_real_exp {
        let product = arena.mul(&[inner_exp, exp]);
        return arena.pow(inner_base, product);
    }
    original
}

/// Unconditional `(a·b)^e → a^e·b^e` (used by `force`).
fn powdenest_pow_forced(arena: &mut Arena, original: ExprId, base: ExprId, exp: ExprId) -> ExprId {
    let children = match arena.node(base).clone() {
        ExprNode::Mul(c) => c,
        _ => return original,
    };
    let distributed: SmallVec<[ExprId; 6]> = children
        .iter()
        .map(|&factor| arena.pow(factor, exp))
        .collect();
    arena.mul(&distributed)
}

/// Try to distribute a single `Pow(base, exp)` when base is a `Mul`.
fn powdenest_pow(arena: &mut Arena, original: ExprId, base: ExprId, exp: ExprId) -> ExprId {
    let children = match arena.node(base).clone() {
        ExprNode::Mul(c) => c,
        _ => return original,
    };

    // Check safety: exponent must be integer, OR all factors nonneg.
    let exp_is_integer = arena.as_num(exp).is_some_and(|r| r.is_integer());
    let all_factors_nonneg = children.iter().all(|&c| {
        // A factor is nonneg if it's a positive rational number or a Pow
        // with a positive rational base (e.g., √5 = Pow(5, 1/2)).
        if let Some(r) = arena.as_num(c) {
            return !r.is_negative();
        }
        if let ExprNode::Pow(inner_base, _) = arena.node(c)
            && let Some(r) = arena.as_num(*inner_base)
        {
            return r.is_positive();
        }
        false
    });

    if !exp_is_integer && !all_factors_nonneg {
        tracing::trace!("powdenest: skipping — exponent not integer and factors not all nonneg");
        return original;
    }

    tracing::debug!(
        n_factors = children.len(),
        exp_is_integer,
        all_factors_nonneg,
        "powdenest: distributing power over product"
    );

    // Distribute: (a·b·c)^n → a^n · b^n · c^n
    // Each Pow(factor, exp) goes through canon_pow, which may simplify
    // further (e.g., Pow(Pow(5, 1/2), 2) → Pow(5, 1) → 5).
    let distributed: SmallVec<[ExprId; 6]> = children
        .iter()
        .map(|&factor| arena.pow(factor, exp))
        .collect();

    arena.mul(&distributed)
}

// ═══════════════════════════════════════════════════════════════════════════
// powsimp_base — combine like exponents by multiplying bases
// ═══════════════════════════════════════════════════════════════════════════

/// Combine factors with the same exponent: `a^e * b^e → (a*b)^e`.
///
/// Only combines when both bases are nonnegative rational numbers (to
/// avoid branch-cut issues with negative or complex bases).  This handles
/// the common cases:
/// - `√2 · √3 → √6` (both bases positive integers, exponent `1/2`)
/// - `2^{1/3} · 4^{1/3} → 8^{1/3} → 2` (positive integers, exponent `1/3`)
///
/// Walks the full expression tree bottom-up.
pub(crate) fn powsimp_base(arena: &mut Arena, expr: ExprId) -> ExprId {
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post_order {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);

        let result = match arena.node(rebuilt).clone() {
            ExprNode::Mul(ref children) => powsimp_base_mul(arena, rebuilt, children),
            _ => rebuilt,
        };

        cache.insert(id, result);
    }

    cache.get(&expr).copied().unwrap_or(expr)
}

/// Try to combine like exponents inside a single `Mul` node.
fn powsimp_base_mul(arena: &mut Arena, original: ExprId, children: &[ExprId]) -> ExprId {
    // Group factors by exponent.  For each factor, decompose as base^exp.
    // Only group factors whose bases are nonneg rationals (or positive
    // rational powers like √5 = Pow(5, 1/2)).
    let mut exp_groups: Vec<(ExprId, SmallVec<[ExprId; 4]>)> = Vec::new();
    let mut exp_index: FxHashMap<ExprId, usize> = FxHashMap::default();
    let mut ungrouped: SmallVec<[ExprId; 6]> = SmallVec::new();

    for &child in children {
        let (base, exp) = extended_base_exp(arena, child);

        // Only group if the base is a nonneg rational number.
        let base_is_nonneg_rational = arena.as_num(base).is_some_and(|r| !r.is_negative());

        if !base_is_nonneg_rational {
            ungrouped.push(child);
            continue;
        }

        // Group by exponent ExprId.
        if let Some(&idx) = exp_index.get(&exp) {
            exp_groups[idx].1.push(base);
        } else {
            exp_index.insert(exp, exp_groups.len());
            let mut bases = SmallVec::new();
            bases.push(base);
            exp_groups.push((exp, bases));
        }
    }

    // Check if any group has > 1 base to combine.
    let any_combined = exp_groups.iter().any(|(_, bases)| bases.len() > 1);
    if !any_combined {
        return original;
    }

    tracing::debug!(
        n_groups = exp_groups.len(),
        n_ungrouped = ungrouped.len(),
        "powsimp_base: combining like-exponent bases"
    );

    // Build combined factors.
    let mut factors: SmallVec<[ExprId; 6]> = SmallVec::new();

    for (exp, bases) in &exp_groups {
        if bases.len() == 1 {
            // Single base — reconstruct the original factor.
            factors.push(arena.pow(bases[0], *exp));
        } else {
            // Multiple bases with the same exponent: multiply them.
            let combined_base = arena.mul(bases);
            tracing::trace!(
                n_bases = bases.len(),
                "powsimp_base: combined bases under common exponent"
            );
            factors.push(arena.pow(combined_base, *exp));
        }
    }

    // Re-include ungrouped factors.
    factors.extend_from_slice(&ungrouped);

    if factors.len() == 1 {
        factors[0]
    } else {
        arena.mul(&factors)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

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
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let result = powsimp(&mut a, x);
        assert_eq!(result, x);
    }

    // ── powdenest tests ────────────────────────────────────────────

    #[test]
    fn powdenest_half_sqrt5_squared() {
        // (1/2 · √5)^2 → (1/2)^2 · (√5)^2 = 1/4 · 5 = 5/4
        let mut a = Arena::new();
        let half = a.rational(1, 2);
        let five = a.int(5);
        let half_exp = a.rational(1, 2);
        let sqrt5 = a.pow(five, half_exp);
        let product = a.mul(&[half, sqrt5]);
        let two = a.int(2);
        let expr = a.pow(product, two);

        let result = powdenest(&mut a, expr);
        let s = display(&a, result);
        assert_eq!(s, "5/4", "powdenest((1/2·√5)²) should be 5/4, got: {s}");
    }

    #[test]
    fn powdenest_noop_for_symbolic_base() {
        // (x · y)^(1/2) should NOT distribute (x, y might be negative)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let product = a.mul(&[x, y]);
        let half = a.rational(1, 2);
        let expr = a.pow(product, half);

        let result = powdenest(&mut a, expr);
        assert_eq!(result, expr, "should not distribute for symbolic bases");
    }

    #[test]
    fn powdenest_integer_exponent_distributes() {
        // (x · 2)^3 → x^3 · 8 (integer exponent is always safe)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two_num = a.int(2);
        let product = a.mul(&[x, two_num]);
        let three = a.int(3);
        let expr = a.pow(product, three);

        let result = powdenest(&mut a, expr);
        let s = display(&a, result);
        assert!(
            s.contains("8"),
            "powdenest((x·2)³) should contain 8, got: {s}"
        );
    }

    // ── powsimp_base tests ─────────────────────────────────────────

    #[test]
    fn powsimp_base_sqrt2_times_sqrt3() {
        // √2 · √3 → √6
        let mut a = Arena::new();
        let two = a.int(2);
        let three = a.int(3);
        let half = a.rational(1, 2);
        let sqrt2 = a.pow(two, half);
        let sqrt3 = a.pow(three, half);
        let expr = a.mul(&[sqrt2, sqrt3]);

        let result = powsimp_base(&mut a, expr);
        let s = display(&a, result);
        assert_eq!(s, "sqrt(6)", "√2·√3 should simplify to √6, got: {s}");
    }

    #[test]
    fn powsimp_base_cbrt2_times_cbrt4() {
        // 2^(1/3) · 4^(1/3) → 8^(1/3) → 2
        let mut a = Arena::new();
        let two = a.int(2);
        let four = a.int(4);
        let third = a.rational(1, 3);
        let cbrt2 = a.pow(two, third);
        let cbrt4 = a.pow(four, third);
        let expr = a.mul(&[cbrt2, cbrt4]);

        let result = powsimp_base(&mut a, expr);
        let s = display(&a, result);
        assert_eq!(s, "2", "2^(1/3)·4^(1/3) should simplify to 2, got: {s}");
    }

    #[test]
    fn powsimp_base_noop_for_different_exponents() {
        // √2 · ∛3 should NOT combine (different exponents)
        let mut a = Arena::new();
        let two = a.int(2);
        let three = a.int(3);
        let half = a.rational(1, 2);
        let third = a.rational(1, 3);
        let sqrt2 = a.pow(two, half);
        let cbrt3 = a.pow(three, third);
        let expr = a.mul(&[sqrt2, cbrt3]);

        let result = powsimp_base(&mut a, expr);
        assert_eq!(result, expr, "different exponents should not combine");
    }

    #[test]
    fn powsimp_base_noop_for_symbolic_bases() {
        // x^(1/2) · y^(1/2) should NOT combine (x, y not nonneg rational)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let half = a.rational(1, 2);
        let sx = a.pow(x, half);
        let sy = a.pow(y, half);
        let expr = a.mul(&[sx, sy]);

        let result = powsimp_base(&mut a, expr);
        assert_eq!(result, expr, "symbolic bases should not combine");
    }

    // ── powdenest_with ─────────────────────────────────────────────

    fn set_assumption(a: &mut Arena, sym: ExprId, prop: crate::base::assumptions::Props) {
        if let ExprNode::Symbol(sid) = *a.node(sym) {
            let mut asm = crate::base::assumptions::Assumptions::default();
            asm.assert_true(prop);
            asm.forward_chain();
            a.set_symbol_assumptions(sid, asm);
        }
    }

    #[test]
    fn powdenest_with_nested_symbolic_powers() {
        let mut a = Arena::new();
        let (x, p, q) = (a.symbol("x"), a.symbol("p"), a.symbol("q"));
        let inner = a.pow(x, p);
        let e = a.pow(inner, q);
        assert_eq!(
            powdenest_with(&mut a, e, false),
            e,
            "no assumptions: unchanged"
        );
        let forced = powdenest_with(&mut a, e, true);
        assert_eq!(a.display(forced).to_string(), "x^(p*q)");
        // Integer outer exponent is always valid.
        let three = a.int(3);
        let cubed = a.pow(inner, three);
        let r = powdenest_with(&mut a, cubed, false);
        assert_eq!(a.display(r).to_string(), "x^(3*p)");
    }

    #[test]
    fn powdenest_with_positive_base_real_exponent() {
        let mut a = Arena::new();
        let (x, p, q) = (a.symbol("x"), a.symbol("p"), a.symbol("q"));
        set_assumption(&mut a, x, Props::POSITIVE);
        set_assumption(&mut a, p, Props::REAL);
        let inner = a.pow(x, p);
        let e = a.pow(inner, q);
        let r = powdenest_with(&mut a, e, false);
        assert_eq!(a.display(r).to_string(), "x^(p*q)");
    }

    #[test]
    fn powdenest_with_sqrt_of_square() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let half = a.rational(1, 2);
        let sq = a.pow(x, two);
        let e = a.pow(sq, half);
        let r = powdenest_with(&mut a, e, false);
        assert_eq!(a.display(r).to_string(), "abs(x)");
        let f = powdenest_with(&mut a, e, true);
        assert_eq!(f, x);
        let n = a.symbol("n");
        set_assumption(&mut a, n, Props::NONNEGATIVE);
        let nsq = a.pow(n, two);
        let en = a.pow(nsq, half);
        assert_eq!(powdenest_with(&mut a, en, false), n);
    }
}
