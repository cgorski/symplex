//! Expression rewriting protocol.
//!
//! Converts between equivalent representations:
//! - `sin(x) → (exp(ix) - exp(-ix)) / (2i)`
//! - `cos(x) → (exp(ix) + exp(-ix)) / 2`
//! - `tan(x) → -i·(exp(ix) - exp(-ix)) / (exp(ix) + exp(-ix))`
//! - `exp(ix) → cos(x) + i·sin(x)` (Euler's formula)
//!
//! Uses the same manual post-order + cache pattern as `expand.rs`
//! because node construction requires `&mut Arena`.

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode};
use crate::walk;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// Target representation for rewriting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum RewriteTarget {
    /// Rewrite trig functions in terms of complex exponentials.
    Exp,
    /// Rewrite exponentials in terms of trig (Euler's formula).
    Trig,
}

/// Rewrite `expr` towards the given `target` representation.
#[allow(dead_code)]
pub(crate) fn rewrite(arena: &mut Arena, expr: ExprId, target: RewriteTarget) -> ExprId {
    match target {
        RewriteTarget::Exp => rewrite_as_exp(arena, expr),
        RewriteTarget::Trig => rewrite_as_trig(arena, expr),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Trig → Exp
// ═══════════════════════════════════════════════════════════════════════════

/// Rewrite trigonometric functions as complex exponentials.
///
/// - `sin(x) → (exp(ix) − exp(−ix)) / (2i)`
/// - `cos(x) → (exp(ix) + exp(−ix)) / 2`
/// - `tan(x) → sin(x)/cos(x)` converted (i.e. both parts rewritten)
pub(crate) fn rewrite_as_exp(arena: &mut Arena, expr: ExprId) -> ExprId {
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post_order {
        let rebuilt = crate::walk::rebuild_with_cache(arena, id, &cache);

        let result = match arena.node(rebuilt).clone() {
            ExprNode::Sin(inner) => {
                // sin(x) = (exp(ix) - exp(-ix)) / (2i)
                let ix = make_i_times(arena, inner);
                let neg_ix = arena.neg(ix);
                let exp_ix = arena.exp(ix);
                let exp_neg_ix = arena.exp(neg_ix);
                let diff = arena.sub(exp_ix, exp_neg_ix);
                let i_unit = arena.i_unit;
                let two = arena.int(2);
                let two_i = arena.mul(&[two, i_unit]);
                arena.div(diff, two_i)
            }
            ExprNode::Cos(inner) => {
                // cos(x) = (exp(ix) + exp(-ix)) / 2
                let ix = make_i_times(arena, inner);
                let neg_ix = arena.neg(ix);
                let exp_ix = arena.exp(ix);
                let exp_neg_ix = arena.exp(neg_ix);
                let sum = arena.add(&[exp_ix, exp_neg_ix]);
                let two = arena.int(2);
                arena.div(sum, two)
            }
            ExprNode::Tan(inner) => {
                // tan(x) = -i*(exp(ix) - exp(-ix)) / (exp(ix) + exp(-ix))
                let ix = make_i_times(arena, inner);
                let neg_ix = arena.neg(ix);
                let exp_ix = arena.exp(ix);
                let exp_neg_ix = arena.exp(neg_ix);
                let diff = arena.sub(exp_ix, exp_neg_ix);
                let sum = arena.add(&[exp_ix, exp_neg_ix]);
                let i_unit = arena.i_unit;
                let neg_i = arena.neg(i_unit);
                let num = arena.mul(&[neg_i, diff]);
                arena.div(num, sum)
            }
            _ => rebuilt,
        };

        cache.insert(id, result);
    }

    cache.get(&expr).copied().unwrap_or(expr)
}

/// Build `i * x`, going through canonical `mul` so the result is sorted.
fn make_i_times(arena: &mut Arena, x: ExprId) -> ExprId {
    let i_unit = arena.i_unit;
    arena.mul(&[i_unit, x])
}

// ═══════════════════════════════════════════════════════════════════════════
// Exp → Trig  (Euler's formula)
// ═══════════════════════════════════════════════════════════════════════════

/// Rewrite complex exponentials as trigonometric functions.
///
/// Detects `exp(i·θ)` and replaces it with `cos(θ) + i·sin(θ)`.
/// Also handles `exp(a + i·θ) → exp(a)·(cos(θ) + i·sin(θ))`.
pub(crate) fn rewrite_as_trig(arena: &mut Arena, expr: ExprId) -> ExprId {
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post_order {
        let rebuilt = crate::walk::rebuild_with_cache(arena, id, &cache);

        let result = match arena.node(rebuilt).clone() {
            ExprNode::Exp(inner) => try_exp_to_trig(arena, rebuilt, inner).unwrap_or(rebuilt),
            // Also handle E^(i*x) which is how exp can appear after powsimp
            ExprNode::Pow(base, exp) if base == arena.e_const => {
                try_exp_to_trig(arena, rebuilt, exp).unwrap_or(rebuilt)
            }
            _ => rebuilt,
        };

        cache.insert(id, result);
    }

    cache.get(&expr).copied().unwrap_or(expr)
}

/// Try to convert `exp(arg)` into trig form.
///
/// Returns `Some(replacement)` if `arg` contains the imaginary unit,
/// `None` otherwise.
fn try_exp_to_trig(arena: &mut Arena, original: ExprId, arg: ExprId) -> Option<ExprId> {
    // Case 1: arg = i*θ  (Mul containing ImaginaryUnit)
    if let Some(theta) = extract_i_coefficient(arena, arg) {
        let cos_t = arena.cos(theta);
        let sin_t = arena.sin(theta);
        let i_unit = arena.i_unit;
        let i_sin = arena.mul(&[i_unit, sin_t]);
        return Some(arena.add(&[cos_t, i_sin]));
    }

    // Case 2: arg = Add([..real_terms.., i*θ])
    // Split into real part + imaginary part.
    if let ExprNode::Add(ref terms) = arena.node(arg).clone() {
        let mut real_terms: SmallVec<[ExprId; 4]> = SmallVec::new();
        let mut imag_angles: SmallVec<[ExprId; 4]> = SmallVec::new();

        for &term in terms {
            if let Some(theta) = extract_i_coefficient(arena, term) {
                imag_angles.push(theta);
            } else {
                real_terms.push(term);
            }
        }

        if !imag_angles.is_empty() {
            let theta = if imag_angles.len() == 1 {
                imag_angles[0]
            } else {
                arena.add(&imag_angles)
            };

            let cos_t = arena.cos(theta);
            let sin_t = arena.sin(theta);
            let i_unit = arena.i_unit;
            let i_sin = arena.mul(&[i_unit, sin_t]);
            let euler = arena.add(&[cos_t, i_sin]);

            if real_terms.is_empty() {
                return Some(euler);
            }

            let real_part = if real_terms.len() == 1 {
                real_terms[0]
            } else {
                arena.add(&real_terms)
            };
            let exp_real = arena.exp(real_part);
            return Some(arena.mul(&[exp_real, euler]));
        }
    }

    // Case 3: arg = ImaginaryUnit itself (exp(i) → cos(1) + i*sin(1))
    if let ExprNode::ImaginaryUnit = arena.node(arg) {
        let one = arena.one;
        let cos_1 = arena.cos(one);
        let sin_1 = arena.sin(one);
        let i_unit = arena.i_unit;
        let i_sin = arena.mul(&[i_unit, sin_1]);
        return Some(arena.add(&[cos_1, i_sin]));
    }

    // Couldn't detect imaginary component — leave unchanged.
    let _ = original;
    None
}

/// If `expr` is `i * θ` (ImaginaryUnit times something), return `θ`.
///
/// Handles:
/// - `ImaginaryUnit` alone → θ = 1
/// - `Mul([ImaginaryUnit, θ])` → θ
/// - `Mul([coeff, ImaginaryUnit, rest...])` → θ = coeff * rest...
/// - `Neg(ImaginaryUnit)` → θ = -1
/// - `Neg(Mul([ImaginaryUnit, ...]))` → θ = negated rest
fn extract_i_coefficient(arena: &mut Arena, expr: ExprId) -> Option<ExprId> {
    match arena.node(expr).clone() {
        ExprNode::ImaginaryUnit => Some(arena.one),

        ExprNode::Neg(inner) => {
            let theta = extract_i_coefficient(arena, inner)?;
            Some(arena.neg(theta))
        }

        ExprNode::Mul(ref children) => {
            let i_pos = children
                .iter()
                .position(|&c| matches!(arena.node(c), ExprNode::ImaginaryUnit))?;

            let rest: SmallVec<[ExprId; 4]> = children
                .iter()
                .enumerate()
                .filter(|&(idx, _)| idx != i_pos)
                .map(|(_, &c)| c)
                .collect();

            if rest.is_empty() {
                Some(arena.one)
            } else if rest.len() == 1 {
                Some(rest[0])
            } else {
                Some(arena.mul(&rest))
            }
        }

        _ => None,
    }
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
    fn rewrite_sin_to_exp() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let sin_x = arena.sin(x);

        let result = rewrite_as_exp(&mut arena, sin_x);
        let s = display(&arena, result);
        // Should contain exp and I (or i)
        assert!(
            s.contains("exp") || s.contains("E"),
            "rewrite should produce exponentials: {s}"
        );
    }

    #[test]
    fn rewrite_cos_to_exp() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let cos_x = arena.cos(x);

        let result = rewrite_as_exp(&mut arena, cos_x);
        let s = display(&arena, result);
        assert!(
            s.contains("exp") || s.contains("E"),
            "rewrite should produce exponentials: {s}"
        );
    }

    #[test]
    fn rewrite_exp_ix_to_trig() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let i_unit = arena.i_unit;
        let ix = arena.mul(&[i_unit, x]);
        let exp_ix = arena.exp(ix);

        let result = rewrite_as_trig(&mut arena, exp_ix);
        let s = display(&arena, result);
        // Should contain cos and sin
        assert!(
            s.contains("cos") && s.contains("sin"),
            "rewrite should produce trig: {s}"
        );
    }

    #[test]
    fn rewrite_atom_unchanged() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");

        let result_exp = rewrite_as_exp(&mut arena, x);
        assert_eq!(result_exp, x);

        let result_trig = rewrite_as_trig(&mut arena, x);
        assert_eq!(result_trig, x);
    }
}
