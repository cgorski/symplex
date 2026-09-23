//! Logarithm expansion.
//!
//! Implements [`expand_log`], which applies logarithm properties:
//!
//! - `ln(a * b)` → `ln(a) + ln(b)`
//! - `ln(a / b)` → `ln(a) - ln(b)` (i.e., `ln(a * b^(-1))`)
//! - `ln(a^n)` → `n * ln(a)`
//!
//! These rules hold for positive real arguments, and not in general:
//! `ln((−1)·(−1)) = 0 ≠ 2πi = ln(−1) + ln(−1)`.  [`expand_log`] (the
//! default, SymPy's `expand_log`) applies them only where the assumption
//! system shows they hold; [`expand_log_with`] with `force = true` applies
//! them unconditionally (SymPy's `force=True`).

use crate::base::arena::Arena;
use crate::base::assumptions::{AssumptionCache, Props};
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;
use rustc_hash::FxHashMap;

/// Expand logarithms where the identities hold.
///
/// Walks the expression bottom-up and expands `Ln` nodes whose arguments
/// are products, quotients, or powers, as far as the assumptions allow.
/// Equivalent to [`expand_log_with`] with `force = false`.
pub(crate) fn expand_log(arena: &mut Arena, expr: ExprId) -> ExprId {
    expand_log_with(arena, expr, false)
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

/// Expand a single `ln(inner)` node: unconditionally with `force`,
/// otherwise only as far as the identities hold (SymPy's
/// `log._eval_expand_log`):
///
/// * a product splits off every factor known positive, and a factor known
///   negative as `ln(−f)` with its sign left in the remaining product:
///   `ln(2·x) = ln 2 + ln x`, `ln(−2·x) = ln 2 + ln(−x)`, `ln(p·x) = ln p
///   + ln x` for `p > 0` — exact, since `arg(p·z) = arg z` for `p > 0`;
/// * `ln(b^e) = e·ln b` for real `e` when `b > 0`, or when `−1 < e ≤ 1`
///   (then `e·arg b ∈ (−π, π]`): `ln(√x) = ½·ln x` for every `x`, while
///   `ln(x²)` and `ln(1/x)` stay.
///
/// With `force` every factor counts as positive and every exponent as
/// admissible (SymPy's `force=True`).  Nested arguments expand all the way
/// (`ln(x²·y)` → `2·ln x + ln y` for positive `x`, `y`), through an explicit
/// work list of `(coefficient, argument)` pairs rather than recursion.
pub(crate) fn expand_ln_node_guarded(
    arena: &mut Arena,
    assumptions: &mut AssumptionCache,
    inner: ExprId,
    force: bool,
) -> ExprId {
    let mut work: Vec<(ExprId, ExprId)> = vec![(arena.one, inner)];
    let mut terms: smallvec::SmallVec<[ExprId; 6]> = smallvec::SmallVec::new();
    while let Some((coeff, arg)) = work.pop() {
        match arena.node(arg).clone() {
            ExprNode::Mul(ref children) => {
                let mut split = false;
                let mut rest: smallvec::SmallVec<[ExprId; 6]> = smallvec::SmallVec::new();
                for &c in children {
                    if force || assumptions.query(arena, c, Props::POSITIVE) == Some(true) {
                        work.push((coeff, c));
                        split = true;
                    } else if c != arena.neg_one
                        && assumptions.query(arena, c, Props::NEGATIVE) == Some(true)
                    {
                        let neg_c = arena.neg(c);
                        work.push((coeff, neg_c));
                        rest.push(arena.neg_one);
                        split = true;
                    } else {
                        rest.push(c);
                    }
                }
                if !split {
                    let ln_arg = arena.ln(arg);
                    terms.push(arena.mul(&[coeff, ln_arg]));
                } else if !rest.is_empty() {
                    let remaining = arena.mul(&rest);
                    if remaining == arena.neg_one {
                        // ln(−1) = iπ (`ln(−3p) = ln 3 + ln p + iπ` for p > 0).
                        let i_pi = arena.mul(&[arena.i_unit, arena.pi]);
                        terms.push(arena.mul(&[coeff, i_pi]));
                    } else if remaining != arena.one {
                        work.push((coeff, remaining));
                    }
                }
            }
            ExprNode::Pow(base, exp) if pow_log_splits(arena, assumptions, base, exp, force) => {
                let c = arena.mul(&[coeff, exp]);
                work.push((c, base));
            }
            // ln(e^w) = w for real w (Im w ∈ (−π, π] suffices; real is what
            // the assumptions can show).
            ExprNode::Exp(w) if force || assumptions.query(arena, w, Props::REAL) == Some(true) => {
                terms.push(arena.mul(&[coeff, w]));
            }
            _ => {
                let ln_arg = arena.ln(arg);
                terms.push(arena.mul(&[coeff, ln_arg]));
            }
        }
    }
    match terms.len() {
        0 => arena.zero,
        1 => terms[0],
        _ => arena.add(&terms),
    }
}

/// `ln(b^e) = e·ln b`?  For real `e` when `b > 0`, or when `−1 < e ≤ 1`
/// (then `e·arg b ∈ (−π, π]`, so the principal logarithms agree); always
/// with `force`.
fn pow_log_splits(
    arena: &Arena,
    assumptions: &mut AssumptionCache,
    base: ExprId,
    exp: ExprId,
    force: bool,
) -> bool {
    if force {
        return true;
    }
    if assumptions.query(arena, exp, Props::REAL) != Some(true) {
        return false;
    }
    let in_principal_range = arena
        .as_num(exp)
        .is_some_and(|e| *e > crate::base::numeric::qi(-1) && *e <= crate::base::numeric::qi(1));
    in_principal_range || assumptions.query(arena, base, Props::POSITIVE) == Some(true)
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

    fn positive(a: &mut Arena, name: &str) -> ExprId {
        let s = a.symbol(name);
        if let ExprNode::Symbol(sid) = *a.node(s) {
            let mut asm = crate::base::assumptions::Assumptions::default();
            asm.assert_true(Props::POSITIVE);
            asm.forward_chain();
            a.set_symbol_assumptions(sid, asm);
        }
        s
    }

    fn forced(a: &mut Arena, e: ExprId) -> String {
        let r = expand_log_with(a, e, true);
        display(a, r)
    }

    fn guarded(a: &mut Arena, e: ExprId) -> String {
        let r = expand_log(a, e);
        display(a, r)
    }

    #[test]
    fn forced_expand_ln_product() {
        let mut a = Arena::new();
        let (x, y) = (sym(&mut a, "x"), sym(&mut a, "y"));
        let product = a.mul(&[x, y]);
        let expr = a.ln(product);
        assert_eq!(forced(&mut a, expr), "ln(x) + ln(y)");
    }

    #[test]
    fn forced_expand_ln_power_and_quotient() {
        let mut a = Arena::new();
        let (x, y) = (sym(&mut a, "x"), sym(&mut a, "y"));
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let e = a.ln(x2);
        assert_eq!(forced(&mut a, e), "2*ln(x)");
        let q = a.div(x, y);
        let e = a.ln(q);
        assert_eq!(forced(&mut a, e), "-ln(y) + ln(x)");
    }

    #[test]
    fn expand_ln_bare_symbol_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.ln(x);
        assert_eq!(guarded(&mut a, expr), "ln(x)");
        assert_eq!(forced(&mut a, expr), "ln(x)");
    }

    #[test]
    fn expand_log_inside_sin() {
        let mut a = Arena::new();
        let (x, y) = (positive(&mut a, "x"), positive(&mut a, "y"));
        let product = a.mul(&[x, y]);
        let ln_product = a.ln(product);
        let expr = a.sin(ln_product);
        assert_eq!(guarded(&mut a, expr), "sin(ln(x) + ln(y))");
    }

    #[test]
    fn expand_ln_nested_in_one_pass() {
        let mut a = Arena::new();
        let (x, y) = (positive(&mut a, "x"), sym(&mut a, "y"));
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let product = a.mul(&[x2, y]);
        let expr = a.ln(product);
        // SymPy: expand_log(log(p**2*y)) = 2*log(p) + log(y)
        assert_eq!(guarded(&mut a, expr), "2*ln(x) + ln(y)");
    }

    // ── the default: only what holds for every complex value ─────────────

    #[test]
    fn guarded_expand_splits_off_positive_factors() {
        // SymPy: expand_log(log(2*x)) = log(x) + log(2)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let e = a.mul(&[two, x]);
        let e = a.ln(e);
        assert_eq!(guarded(&mut a, e), "ln(2) + ln(x)");
    }

    #[test]
    fn guarded_expand_moves_the_sign_of_a_negative_factor() {
        // SymPy: expand_log(log(-2*x)) = log(-x) + log(2),
        //        expand_log(log(-3*p)) = log(p) + log(3) + I*pi  (p > 0)
        let mut a = Arena::new();
        let (x, p) = (sym(&mut a, "x"), positive(&mut a, "p"));
        let (m2, m3) = (a.int(-2), a.int(-3));
        let e = a.mul(&[m2, x]);
        let e = a.ln(e);
        assert_eq!(guarded(&mut a, e), "ln(2) + ln(-x)");
        let e = a.mul(&[m3, p]);
        let e = a.ln(e);
        assert_eq!(guarded(&mut a, e), "pi*I + ln(3) + ln(p)");
    }

    #[test]
    fn guarded_expand_keeps_powers_outside_the_principal_range() {
        // SymPy: expand_log(log(x**2)) = log(x**2), expand_log(log(1/x)) = log(1/x),
        //        expand_log(log(sqrt(x))) = log(x)/2, expand_log(log(x**(1/3))) = log(x)/3
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let (two, neg_one) = (a.int(2), a.int(-1));
        for exp in [two, neg_one] {
            let p = a.pow(x, exp);
            let e = a.ln(p);
            assert_eq!(expand_log(&mut a, e), e);
        }
        let half = a.rational(1, 2);
        let e = a.pow(x, half);
        let e = a.ln(e);
        assert_eq!(guarded(&mut a, e), "1/2*ln(x)");
        let third = a.rational(1, 3);
        let e = a.pow(x, third);
        let e = a.ln(e);
        assert_eq!(guarded(&mut a, e), "1/3*ln(x)");
    }

    #[test]
    fn guarded_expand_of_exp_needs_a_real_exponent() {
        // SymPy: expand_log(log(exp(x))) = log(exp(x)); force=True gives x.
        let mut a = Arena::new();
        let (x, p) = (sym(&mut a, "x"), positive(&mut a, "p"));
        let ex = a.exp(x);
        let e = a.ln(ex);
        assert_eq!(expand_log(&mut a, e), e);
        assert_eq!(forced(&mut a, e), "x");
        let ep = a.exp(p);
        let e = a.ln(ep);
        assert_eq!(guarded(&mut a, e), "p");
    }

    // ── guarded expansion ──────────────────────────────────────────

    #[test]
    fn guarded_expand_requires_positive_factors() {
        let mut a = Arena::new();
        let (x, y) = (sym(&mut a, "x"), sym(&mut a, "y"));
        let xy = a.mul(&[x, y]);
        let e = a.ln(xy);
        assert_eq!(expand_log_with(&mut a, e, false), e);
        let forced = expand_log_with(&mut a, e, true);
        assert_eq!(display(&a, forced), "ln(x) + ln(y)");
        // Positive symbols pass the guard.
        for s in [x, y] {
            if let ExprNode::Symbol(sid) = *a.node(s) {
                let mut asm = crate::base::assumptions::Assumptions::default();
                asm.assert_true(crate::base::assumptions::Props::POSITIVE);
                asm.forward_chain();
                a.set_symbol_assumptions(sid, asm);
            }
        }
        let guarded = expand_log_with(&mut a, e, false);
        assert_eq!(display(&a, guarded), "ln(x) + ln(y)");
    }

    #[test]
    fn forced_expand_ln_of_product_with_power_factor() {
        let mut a = Arena::new();
        let (x, y) = (sym(&mut a, "x"), sym(&mut a, "y"));
        let two = a.int(2);
        let y2 = a.pow(y, two);
        let xy2 = a.mul(&[x, y2]);
        let e = a.ln(xy2);
        assert_eq!(forced(&mut a, e), "2*ln(y) + ln(x)");
    }
}
