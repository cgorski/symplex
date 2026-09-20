//! Shared display-ordering and negation-detection helpers used by both
//! `display.rs` (plain-text) and `latex.rs` (LaTeX) renderers.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::Signed;
use smallvec::SmallVec;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};

// ═══════════════════════════════════════════════════════════════════════════
// Display ordering helpers for Add children
// ═══════════════════════════════════════════════════════════════════════════

/// Classification of a term for display ordering within Add.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum DisplayCategory {
    /// Polynomial term — sorted by degree descending, then variable name.
    Polynomial,
    /// Function application (sin, cos, exp, etc.) — after polynomials.
    Function,
    /// Constant (number, pi, e, i) — displayed last.
    Constant,
}

/// Compute a display sort key for an Add child.
/// Returns (category, negative_degree, var_sort_key_bytes, canonical_sort_key_bytes).
/// Sorted ascending: higher-degree polynomials first, then functions, then constants.
pub(crate) fn display_sort_key(
    arena: &Arena,
    id: ExprId,
) -> (DisplayCategory, i64, SmallVec<[u8; 24]>, SmallVec<[u8; 24]>) {
    let cat = display_category(arena, id);
    let degree = estimate_display_degree(arena, id);
    let var_key = dominant_var_key(arena, id);
    let canon_key = SmallVec::from_slice(arena.sort_key(id).as_bytes());
    (cat, -(degree as i64), var_key, canon_key)
}

pub(crate) fn display_category(arena: &Arena, id: ExprId) -> DisplayCategory {
    match arena.node(id) {
        ExprNode::Num(_)
        | ExprNode::Pi
        | ExprNode::E
        | ExprNode::ImaginaryUnit
        | ExprNode::EulerGamma
        | ExprNode::Catalan
        | ExprNode::GoldenRatio
        | ExprNode::PhysicalConstant(_, _)
        | ExprNode::BoolTrue
        | ExprNode::BoolFalse => DisplayCategory::Constant,
        ExprNode::Infinity | ExprNode::NegInfinity | ExprNode::ComplexInfinity | ExprNode::NaN => {
            DisplayCategory::Constant
        }
        ExprNode::Symbol(_) => DisplayCategory::Polynomial,
        ExprNode::Pow(base, exp) => {
            // x^n with integer n → polynomial; x^(1/2) etc. → function-like
            if matches!(arena.node(*base), ExprNode::Symbol(_))
                && let Some(r) = arena.as_num(*exp)
                && r.is_integer()
            {
                return DisplayCategory::Polynomial;
            }
            DisplayCategory::Function
        }
        ExprNode::Mul(children) => {
            // If any child is polynomial (Symbol or Pow(Symbol, int)), this is polynomial
            for &child in children.iter() {
                let child_cat = display_category(arena, child);
                if child_cat == DisplayCategory::Polynomial {
                    return DisplayCategory::Polynomial;
                }
            }
            // Pure numeric Mul → constant
            if children
                .iter()
                .all(|&c| matches!(arena.node(c), ExprNode::Num(_)))
            {
                DisplayCategory::Constant
            } else {
                DisplayCategory::Function
            }
        }
        ExprNode::Neg(inner) => display_category(arena, *inner),
        // All function nodes → Function category
        _ => DisplayCategory::Function,
    }
}

/// Estimate the polynomial degree of a term for display ordering.
/// Higher degree terms should display first within the Polynomial category.
pub(crate) fn estimate_display_degree(arena: &Arena, id: ExprId) -> u32 {
    match arena.node(id) {
        ExprNode::Num(_)
        | ExprNode::Pi
        | ExprNode::E
        | ExprNode::ImaginaryUnit
        | ExprNode::EulerGamma
        | ExprNode::Catalan
        | ExprNode::GoldenRatio
        | ExprNode::PhysicalConstant(_, _)
        | ExprNode::BoolTrue
        | ExprNode::BoolFalse
        | ExprNode::Infinity
        | ExprNode::NegInfinity
        | ExprNode::ComplexInfinity
        | ExprNode::NaN => 0,
        ExprNode::Symbol(_) => 1,
        ExprNode::Pow(_base, exp) => {
            if let Some(r) = arena.as_num(*exp)
                && r.is_integer()
                && !r.is_negative()
            {
                return r.to_integer().try_into().unwrap_or(1);
            }
            1
        }
        ExprNode::Mul(children) => {
            // Degree is sum of degrees of variable factors
            // e.g., x*y → degree 2, 3*x^2 → degree 2, 2*x → degree 1
            let mut total_degree = 0u32;
            for &child in children.iter() {
                let d = estimate_display_degree(arena, child);
                if d > 0 {
                    total_degree += d;
                }
            }
            total_degree
        }
        ExprNode::Neg(inner) => estimate_display_degree(arena, *inner),
        _ => 0,
    }
}

/// Get the sort key of the dominant variable in a term, for secondary ordering.
/// For `3*x^2`, this returns the sort key of `x`.
/// For `x*y`, this returns the sort key of `x` (first variable alphabetically).
pub(crate) fn dominant_var_key(arena: &Arena, id: ExprId) -> SmallVec<[u8; 24]> {
    match arena.node(id) {
        ExprNode::Symbol(_) => SmallVec::from_slice(arena.sort_key(id).as_bytes()),
        ExprNode::Pow(base, _) => {
            if matches!(arena.node(*base), ExprNode::Symbol(_)) {
                SmallVec::from_slice(arena.sort_key(*base).as_bytes())
            } else {
                SmallVec::new()
            }
        }
        ExprNode::Mul(children) => {
            // Find the first symbol/power-of-symbol child
            for &child in children.iter() {
                let key = dominant_var_key(arena, child);
                if !key.is_empty() {
                    return key;
                }
            }
            SmallVec::new()
        }
        ExprNode::Neg(inner) => dominant_var_key(arena, *inner),
        _ => SmallVec::new(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Negation-detection helpers for Add rendering
// ═══════════════════════════════════════════════════════════════════════════

/// Check if `id` is a Mul node whose first factor is the number −1.
pub(crate) fn is_neg_one_mul(arena: &Arena, id: ExprId) -> bool {
    if let ExprNode::Mul(children) = arena.node(id)
        && let Some(&first) = children.first()
        && let ExprNode::Num(nid) = arena.node(first)
    {
        let r = arena.num(*nid);
        return *r == Ratio::from(BigInt::from(-1));
    }
    false
}

/// Check if `id` is a Mul node whose first factor is a negative number (not just -1).
/// Returns true if the leading coefficient is negative and not equal to -1.
pub(crate) fn is_neg_coeff_mul(arena: &Arena, id: ExprId) -> bool {
    if let ExprNode::Mul(children) = arena.node(id)
        && children.len() >= 2
        && let Some(&first) = children.first()
        && let ExprNode::Num(nid) = arena.node(first)
    {
        let r = arena.num(*nid);
        return r.is_negative() && *r != Ratio::from(BigInt::from(-1));
    }
    false
}

// ═══════════════════════════════════════════════════════════════════════════
// Mul rendering helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Check if `id` is Pow(base, negative_exponent).
/// Returns (base_id, positive_exponent_string) if so.
pub(crate) fn extract_negative_power(arena: &Arena, id: ExprId) -> Option<(ExprId, String)> {
    if let ExprNode::Pow(base, exp) = arena.node(id)
        && let ExprNode::Num(nid) = arena.node(*exp)
    {
        let r = arena.num(*nid);
        if r.is_negative() {
            let pos_r = -r.clone();
            if pos_r.is_integer() {
                let n = pos_r.to_integer();
                return Some((*base, format!("{}", n)));
            } else {
                return Some((
                    *base,
                    format!("\\frac{{{}}}{{{}}}", pos_r.numer(), pos_r.denom()),
                ));
            }
        }
    }
    None
}

/// The variant name of a node for error messages (`"Sin"`, `"Integral"`,
/// …), taken from its `Debug` rendering.
pub(crate) fn describe(node: &crate::base::node::ExprNode) -> String {
    let dbg = format!("{node:?}");
    dbg.split(['(', ' ']).next().unwrap_or("node").to_string()
}
