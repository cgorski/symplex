//! Logarithm combination (inverse of log expansion).
//!
//! Implements [`log_combine`], which applies logarithm combination rules:
//!
//! - `ln(a) + ln(b)` → `ln(a * b)`
//! - `n * ln(a)` → `ln(a^n)`
//!
//! This is the inverse of [`expand_log`](crate::simplify::log_expand::expand_log).
//!
//! Neither holds for every complex value (`ln(−1) + ln(−1) = 2πi ≠
//! ln 1`), so [`log_combine`] (the default) applies them only where they
//! do — see [`log_combine_with`] — and `force = true` applies them
//! unconditionally (SymPy's `logcombine(force=True)`).

use crate::base::arena::Arena;
use crate::base::assumptions::{AssumptionCache, Props};
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;
use rustc_hash::FxHashMap;

/// Combine logarithms where the identities hold.
///
/// Walks the expression bottom-up; sums of `ln` terms become the logarithm
/// of a product and `c·ln a` becomes `ln(a^c)`, as far as
/// [`log_combine_with`] with `force = false` allows.
pub(crate) fn log_combine(arena: &mut Arena, expr: ExprId) -> ExprId {
    log_combine_with(arena, expr, false)
}

/// Combine logarithms: exactly where the identities hold, or everywhere
/// with `force` (SymPy's `logcombine(force=True)`).
///
/// # Branch reasoning
///
/// `ln a + ln b = ln(a·b)` holds exactly when `arg a + arg b ∈ (−π, π]`,
/// and `c·ln a = ln(a^c)` when `c·arg a ∈ (−π, π]`.  Without `force`:
///
/// * the logarithms of known-positive arguments combine with each other
///   *and with at most one other logarithm*: `arg p = 0`, so `ln p + ln z =
///   ln(p·z)` for every complex `z` (`ln 2 + ln x → ln(2x)`).  SymPy's
///   `logcombine` combines only when every argument is positive; this is
///   the same identity used to its full extent.
/// * `c·ln a → ln(a^c)` when `a > 0` and `c` is real, or `c` is a rational
///   in `(−1, 1]` (then `c·arg a ∈ (−π, π]` for every `a`: `½·ln x =
///   ln √x`).  `2·ln x` and `−ln x` stay: `−ln(−1) = −iπ ≠ ln(−1) = iπ`.
pub(crate) fn log_combine_with(arena: &mut Arena, expr: ExprId, force: bool) -> ExprId {
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    let mut assumptions = AssumptionCache::new();

    for &id in &post_order {
        let node = arena.node(id).clone();
        let combined = match node {
            // ── n-ary product ──────────────────────────────────────
            // If exactly one factor is `Ln(arg)` and there is at least one
            // other factor, rewrite `coeff · ln(arg)` → `ln(arg ^ coeff)`.
            ExprNode::Mul(ref children) => {
                let new: smallvec::SmallVec<[ExprId; 6]> = children
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();

                combine_mul_ln(arena, &mut assumptions, id, children, &new, force)
            }

            // ── n-ary sum ──────────────────────────────────────────
            // Collect all `Ln(…)` children, combine into a single log of
            // a product, and keep non-log children untouched.
            ExprNode::Add(ref children) => {
                let new: smallvec::SmallVec<[ExprId; 6]> = children
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();

                combine_add_ln(arena, &mut assumptions, id, children, &new, force)
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
            _ => {
                // Rebuild any other node with cached children
                crate::base::walk::rebuild_with_cache(arena, id, &cache)
            }
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
fn combine_mul_ln(
    arena: &mut Arena,
    assumptions: &mut AssumptionCache,
    id: ExprId,
    original: &[ExprId],
    new: &[ExprId],
    force: bool,
) -> ExprId {
    // Scan for Ln factors, remembering the first as (index, argument).
    let mut ln_factor: Option<(usize, ExprId)> = None;
    let mut ln_count: usize = 0;

    for (i, &child) in new.iter().enumerate() {
        if let ExprNode::Ln(inner) = *arena.node(child) {
            ln_count += 1;
            if ln_count == 1 {
                ln_factor = Some((i, inner));
            }
        }
    }

    let (idx, ln_arg) = match ln_factor {
        Some(f) if ln_count == 1 && new.len() >= 2 => f,
        _ => return if new == original { id } else { arena.mul(new) },
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

    // Guard: c·ln(a) = ln(a^c) needs c·arg a ∈ (−π, π]: a > 0 with c real,
    // or c a rational in (−1, 1].
    let coeff_in_principal_range = arena
        .as_num(coeff)
        .is_some_and(|c| *c > crate::base::numeric::qi(-1) && *c <= crate::base::numeric::qi(1));
    let guard_ok = force
        || coeff_in_principal_range
        || (assumptions.query(arena, ln_arg, Props::POSITIVE) == Some(true)
            && assumptions.query(arena, coeff, Props::REAL) == Some(true));
    if !guard_ok {
        return if new == original { id } else { arena.mul(new) };
    }

    let powered = arena.pow(ln_arg, coeff);
    arena.ln(powered)
}

/// Scan the children of an `Add` node for `Ln(…)` terms and combine them
/// into a single `Ln(product)`.
///
/// Non-log children are kept as-is. If fewer than two `Ln` terms are
/// found, the node is returned unchanged (or rebuilt if children were
/// modified by the cache).
fn combine_add_ln(
    arena: &mut Arena,
    assumptions: &mut AssumptionCache,
    id: ExprId,
    original: &[ExprId],
    new: &[ExprId],
    force: bool,
) -> ExprId {
    // Partition the children: logarithms of known-positive arguments,
    // other logarithms, everything else.  The positive ones combine with
    // each other and with at most one other logarithm (see
    // `log_combine_with`); with `force` every logarithm combines.
    let mut ln_inner_args: smallvec::SmallVec<[ExprId; 6]> = smallvec::SmallVec::new();
    let mut unknown: smallvec::SmallVec<[(ExprId, ExprId); 6]> = smallvec::SmallVec::new();
    let mut others: smallvec::SmallVec<[ExprId; 6]> = smallvec::SmallVec::new();

    for &child in new {
        if let ExprNode::Ln(inner) = *arena.node(child) {
            if force || assumptions.query(arena, inner, Props::POSITIVE) == Some(true) {
                ln_inner_args.push(inner);
            } else {
                unknown.push((child, inner));
            }
        } else {
            others.push(child);
        }
    }
    if unknown.len() == 1 {
        ln_inner_args.push(unknown[0].1);
    } else {
        others.extend(unknown.iter().map(|&(child, _)| child));
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
        let r = log_combine_with(a, e, true);
        display(a, r)
    }

    fn guarded(a: &mut Arena, e: ExprId) -> String {
        let r = log_combine(a, e);
        display(a, r)
    }

    #[test]
    fn forced_combine_two_and_three_logs() {
        let mut a = Arena::new();
        let (x, y, z) = (sym(&mut a, "x"), sym(&mut a, "y"), sym(&mut a, "z"));
        let (lnx, lny, lnz) = (a.ln(x), a.ln(y), a.ln(z));
        let two = a.add(&[lnx, lny]);
        assert_eq!(forced(&mut a, two), "ln(x*y)");
        let three = a.add(&[lnx, lny, lnz]);
        assert_eq!(forced(&mut a, three), "ln(x*y*z)");
    }

    #[test]
    fn forced_combine_coeff_log() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let lnx = a.ln(x);
        let expr = a.mul(&[two, lnx]);
        assert_eq!(forced(&mut a, expr), "ln(x^2)");
    }

    #[test]
    fn no_combine_single_log() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let lnx = a.ln(x);
        let one = a.int(1);
        let expr = a.add(&[lnx, one]);
        assert_eq!(log_combine(&mut a, expr), expr);
        assert_eq!(log_combine_with(&mut a, expr, true), expr);
    }

    #[test]
    fn forced_combine_keeps_other_terms() {
        let mut a = Arena::new();
        let (x, y, z) = (sym(&mut a, "x"), sym(&mut a, "y"), sym(&mut a, "z"));
        let (lnx, lny) = (a.ln(x), a.ln(y));
        let expr = a.add(&[lnx, lny, z]);
        assert_eq!(forced(&mut a, expr), "z + ln(x*y)");
    }

    #[test]
    fn combine_logs_inside_exp() {
        let mut a = Arena::new();
        let (x, y) = (positive(&mut a, "x"), positive(&mut a, "y"));
        let (ln_x, ln_y) = (a.ln(x), a.ln(y));
        let sum = a.add(&[ln_x, ln_y]);
        let expr = a.exp(sum);
        assert_eq!(guarded(&mut a, expr), "exp(ln(x*y))");
    }

    #[test]
    fn bare_symbol_unchanged() {
        let mut a = Arena::new();
        let (x, y) = (sym(&mut a, "x"), sym(&mut a, "y"));
        let expr = a.add(&[x, y]);
        assert_eq!(log_combine(&mut a, expr), expr);
    }

    // ── the default: only what holds for every complex value ─────────────

    #[test]
    fn guarded_combine_leaves_two_unknown_logs() {
        // SymPy: logcombine(log(x) + log(y)) = log(x) + log(y);
        // ln(-1) + ln(-1) = 2*pi*I but ln(1) = 0.
        let mut a = Arena::new();
        let (x, y) = (sym(&mut a, "x"), sym(&mut a, "y"));
        let (lnx, lny) = (a.ln(x), a.ln(y));
        let e = a.add(&[lnx, lny]);
        assert_eq!(log_combine(&mut a, e), e);
    }

    #[test]
    fn guarded_combine_joins_positive_logs_with_one_other() {
        // arg(p) = 0 for p > 0, so ln p + ln z = ln(p*z) for every complex z.
        // SymPy: logcombine(log(2) + log(x)) = log(2*x),
        //        logcombine(log(p) + log(x)) = log(p*x);
        //        logcombine(log(p) + log(x) + log(y)) leaves all three.
        let mut a = Arena::new();
        let (x, y, p) = (sym(&mut a, "x"), sym(&mut a, "y"), positive(&mut a, "p"));
        let two = a.int(2);
        let (ln2, lnx, lny, lnp) = (a.ln(two), a.ln(x), a.ln(y), a.ln(p));
        let e = a.add(&[ln2, lnx]);
        assert_eq!(guarded(&mut a, e), "ln(2*x)");
        let e = a.add(&[lnp, lnx]);
        assert_eq!(guarded(&mut a, e), "ln(p*x)");
        let e = a.add(&[ln2, lnp, lnx, lny]);
        assert_eq!(guarded(&mut a, e), "ln(x) + ln(y) + ln(2*p)");
    }

    #[test]
    fn guarded_combine_coefficient_needs_positive_argument_or_principal_range() {
        // SymPy: logcombine(2*log(p)) = log(p**2); logcombine(2*log(x)) and
        // logcombine(-log(x)) leave them; logcombine(log(x)/2) = log(x)/2
        // (though ln(sqrt(x)) = ln(x)/2 for every x: -pi < arg(x)/2 <= pi/2).
        let mut a = Arena::new();
        let (x, p) = (sym(&mut a, "x"), positive(&mut a, "p"));
        let (two, neg_one, half) = (a.int(2), a.int(-1), a.rational(1, 2));
        let (lnx, lnp) = (a.ln(x), a.ln(p));
        let e = a.mul(&[two, lnp]);
        assert_eq!(guarded(&mut a, e), "ln(p^2)");
        for c in [two, neg_one] {
            let e = a.mul(&[c, lnx]);
            assert_eq!(log_combine(&mut a, e), e);
        }
        let e = a.mul(&[half, lnx]);
        assert_eq!(guarded(&mut a, e), "ln(sqrt(x))");
    }
}
