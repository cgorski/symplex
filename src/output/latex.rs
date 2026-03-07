//! LaTeX rendering for symbolic expressions.
//!
//! Provides `to_latex()` methods on `Expr`, `Matrix`, and `Quaternion`.
//! Uses direct `ExprNode` matching (no Display string parsing) for
//! robustness against formatting changes.
//!
//! # Architecture
//!
//! Like `display.rs`, this module uses an **explicit work-stack** rather
//! than recursive function calls, guaranteeing stack safety for
//! arbitrarily deep expression trees (Principle 5).
//!
//! For nodes that require global knowledge of all children before
//! rendering (Add sorting, Mul fraction detection), the children are
//! inspected eagerly and the result pushed as an `Owned` string.

use std::fmt;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::Signed;
use smallvec::SmallVec;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Greek letter table
// ═══════════════════════════════════════════════════════════════════════════

const GREEK_LETTERS: &[&str] = &[
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
    "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
    // Uppercase variants
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi", "Rho",
    "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega",
    // Common variants
    "varepsilon", "varphi", "vartheta", "varrho", "varsigma",
];

/// Convert a symbol name to LaTeX, handling Greek letters and subscripts.
fn symbol_to_latex(name: &str) -> String {
    // Check for subscript patterns like "alpha_1" — handle the base name
    if let Some(idx) = name.find('_') {
        let base = &name[..idx];
        let sub = &name[idx + 1..];
        let latex_base = greek_base(base).unwrap_or_else(|| base.to_string());
        return format!("{}_{{{}}}", latex_base, sub);
    }

    if let Some(g) = greek_base(name) {
        return g;
    }

    name.to_string()
}

/// If `name` is a Greek letter, return `\name`; otherwise None.
fn greek_base(name: &str) -> Option<String> {
    for &g in GREEK_LETTERS {
        if name == g {
            return Some(format!("\\{}", g));
        }
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Work-stack items
// ═══════════════════════════════════════════════════════════════════════════

/// An item on the LaTeX work-stack.
enum LatexItem {
    /// A literal string to write verbatim.
    Lit(&'static str),
    /// A dynamically-built string to write.
    Owned(String),
    /// An expression that needs to be expanded.
    Expr(ExprId),
}

// ═══════════════════════════════════════════════════════════════════════════
// Core formatting function (iterative)
// ═══════════════════════════════════════════════════════════════════════════

/// Format an expression rooted at `id` as LaTeX into `f`.
///
/// **This function uses an explicit stack — it never recurses.**
pub(crate) fn fmt_latex(
    arena: &Arena,
    f: &mut fmt::Formatter<'_>,
    id: ExprId,
) -> fmt::Result {
    let mut stack: Vec<LatexItem> = Vec::with_capacity(32);
    stack.push(LatexItem::Expr(id));

    while let Some(item) = stack.pop() {
        match item {
            LatexItem::Lit(s) => write!(f, "{s}")?,
            LatexItem::Owned(s) => write!(f, "{s}")?,
            LatexItem::Expr(eid) => expand_latex(arena, eid, &mut stack),
        }
    }

    Ok(())
}

/// Helper: render an ExprId to a String using `fmt_latex`.
fn latex_to_string(arena: &Arena, id: ExprId) -> String {
    let w = LatexWriter { arena, id };
    format!("{}", w)
}

/// Display wrapper for converting an ExprId to a LaTeX string.
struct LatexWriter<'a> {
    arena: &'a Arena,
    id: ExprId,
}

impl fmt::Display for LatexWriter<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_latex(self.arena, f, self.id)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Display ordering helpers for Add children (mirrors display.rs)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum DisplayCategory {
    Polynomial,
    Function,
    Constant,
}

fn display_category(arena: &Arena, id: ExprId) -> DisplayCategory {
    match arena.node(id) {
        ExprNode::Num(_)
        | ExprNode::Pi
        | ExprNode::E
        | ExprNode::ImaginaryUnit
        | ExprNode::BoolTrue
        | ExprNode::BoolFalse => DisplayCategory::Constant,
        ExprNode::Infinity | ExprNode::NegInfinity | ExprNode::ComplexInfinity | ExprNode::NaN => {
            DisplayCategory::Constant
        }
        ExprNode::Symbol(_) => DisplayCategory::Polynomial,
        ExprNode::Pow(base, exp) => {
            if matches!(arena.node(*base), ExprNode::Symbol(_))
                && let Some(r) = arena.as_num(*exp)
                && r.is_integer()
            {
                return DisplayCategory::Polynomial;
            }
            DisplayCategory::Function
        }
        ExprNode::Mul(children) => {
            for &child in children.iter() {
                if display_category(arena, child) == DisplayCategory::Polynomial {
                    return DisplayCategory::Polynomial;
                }
            }
            if children.iter().all(|&c| matches!(arena.node(c), ExprNode::Num(_))) {
                DisplayCategory::Constant
            } else {
                DisplayCategory::Function
            }
        }
        ExprNode::Neg(inner) => display_category(arena, *inner),
        _ => DisplayCategory::Function,
    }
}

fn estimate_display_degree(arena: &Arena, id: ExprId) -> u32 {
    match arena.node(id) {
        ExprNode::Num(_)
        | ExprNode::Pi
        | ExprNode::E
        | ExprNode::ImaginaryUnit
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

fn dominant_var_key(arena: &Arena, id: ExprId) -> SmallVec<[u8; 24]> {
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

fn display_sort_key(
    arena: &Arena,
    id: ExprId,
) -> (DisplayCategory, i64, SmallVec<[u8; 24]>, SmallVec<[u8; 24]>) {
    let cat = display_category(arena, id);
    let degree = estimate_display_degree(arena, id);
    let var_key = dominant_var_key(arena, id);
    let canon_key = SmallVec::from_slice(arena.sort_key(id).as_bytes());
    (cat, -(degree as i64), var_key, canon_key)
}

// ═══════════════════════════════════════════════════════════════════════════
// Mul rendering helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Check if `id` is Pow(base, negative_exponent).
/// Returns (base_id, positive_exponent_id) if so.
fn extract_negative_power(arena: &Arena, id: ExprId) -> Option<(ExprId, String)> {
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
                return Some((*base, format!(
                    "\\frac{{{}}}{{{}}}",
                    pos_r.numer(),
                    pos_r.denom()
                )));
            }
        }
    }
    None
}

/// Heuristic: does this rendered string look like a plain number?
fn looks_like_number(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let s = s.strip_prefix('-').unwrap_or(s);
    s.chars().all(|c| c.is_ascii_digit())
}

/// Join multiplication factors with implicit multiplication (space) or \cdot.
fn join_mul_factors(factors: &[String]) -> String {
    if factors.len() == 1 {
        return factors[0].clone();
    }

    let mut result = String::new();
    for (i, f) in factors.iter().enumerate() {
        if i > 0 {
            let prev = &factors[i - 1];
            if looks_like_number(prev) && looks_like_number(f) {
                // Two adjacent numbers need an explicit multiplication sign.
                result.push_str(r" \cdot ");
            } else if looks_like_number(prev) {
                // Numeric coefficient followed by a non-numeric factor:
                // use implicit multiplication (no separator), e.g. "3x^{2}".
            } else {
                result.push(' ');
            }
        }
        result.push_str(f);
    }
    result
}

/// Render a single Mul factor, wrapping Add children in \left(...\right).
fn render_mul_factor(arena: &Arena, id: ExprId) -> String {
    let s = latex_to_string(arena, id);
    if matches!(arena.node(id), ExprNode::Add(_)) {
        format!("\\left({}\\right)", s)
    } else {
        s
    }
}

/// Render a Pow base, wrapping compound expressions in \left(...\right).
fn render_pow_base(arena: &Arena, base: ExprId) -> String {
    let s = latex_to_string(arena, base);
    match arena.node(base) {
        ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => {
            format!("\\left({}\\right)", s)
        }
        _ => s,
    }
}

/// Render a Mul node to a complete LaTeX string.
///
/// Handles: leading -1/1, fraction rendering (negative powers → denominator),
/// implicit multiplication.
fn render_mul(arena: &Arena, children: &[ExprId]) -> String {
    if children.is_empty() {
        return "1".to_string();
    }

    // Check for leading -1 or 1
    let first_node = arena.node(children[0]);
    let is_neg_one = if let ExprNode::Num(nid) = first_node {
        *arena.num(*nid) == Ratio::from(BigInt::from(-1))
    } else {
        false
    };
    let is_pos_one = if let ExprNode::Num(nid) = first_node {
        *arena.num(*nid) == Ratio::from(BigInt::from(1))
    } else {
        false
    };

    let (skip_first, prefix) = if children.len() > 1 && is_neg_one {
        (true, "-")
    } else if children.len() > 1 && is_pos_one {
        (true, "")
    } else {
        (false, "")
    };

    let start_idx = if skip_first { 1 } else { 0 };
    let factors = &children[start_idx..];

    // Separate numerator and denominator factors
    let mut numer_factors: Vec<String> = Vec::new();
    let mut denom_factors: Vec<String> = Vec::new();

    for &factor in factors {
        if let Some((base_id, pos_exp_str)) = extract_negative_power(arena, factor) {
            let base_latex = render_pow_base(arena, base_id);
            if pos_exp_str == "1" {
                denom_factors.push(base_latex);
            } else {
                denom_factors.push(format!("{}^{{{}}}", base_latex, pos_exp_str));
            }
        } else {
            numer_factors.push(render_mul_factor(arena, factor));
        }
    }

    if denom_factors.is_empty() {
        // Pure product
        let body = if numer_factors.is_empty() {
            "1".to_string()
        } else {
            join_mul_factors(&numer_factors)
        };
        format!("{}{}", prefix, body)
    } else {
        // Fraction
        let numer_str = if numer_factors.is_empty() {
            "1".to_string()
        } else {
            join_mul_factors(&numer_factors)
        };
        let denom_str = join_mul_factors(&denom_factors);
        if prefix == "-" {
            format!("-\\frac{{{}}}{{{}}}", numer_str, denom_str)
        } else {
            format!("\\frac{{{}}}{{{}}}", numer_str, denom_str)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Add rendering helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Check if `id` is a Mul node whose first factor is the number −1.
fn is_neg_one_mul(arena: &Arena, id: ExprId) -> bool {
    if let ExprNode::Mul(children) = arena.node(id)
        && let Some(&first) = children.first()
        && let ExprNode::Num(nid) = arena.node(first)
    {
        let r = arena.num(*nid);
        return *r == Ratio::from(BigInt::from(-1));
    }
    false
}

/// Check if `id` is a Mul node whose first factor is a negative number (not -1).
fn is_neg_coeff_mul(arena: &Arena, id: ExprId) -> bool {
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

/// Render a Mul with leading -1 without the leading factor.
/// Returns the LaTeX for the remaining factors.
fn render_mul_without_neg_one(arena: &Arena, id: ExprId) -> String {
    if let ExprNode::Mul(children) = arena.node(id) {
        let rest = &children[1..];
        if rest.len() == 1 {
            return latex_to_string(arena, rest[0]);
        }
        // Render as a Mul of the remaining factors
        render_mul(arena, rest)
    } else {
        latex_to_string(arena, id)
    }
}

/// Render a Mul with negative leading coefficient as subtraction content.
/// Returns the LaTeX string with the coefficient negated.
fn render_neg_coeff_mul_as_subtraction(arena: &Arena, id: ExprId) -> String {
    if let ExprNode::Mul(children) = arena.node(id)
        && let Some(&first) = children.first()
        && let ExprNode::Num(nid) = arena.node(first)
    {
        let r = arena.num(*nid);
        let pos_r = -r.clone();

        // Build a new children list with the positive coefficient
        // Replace the first child in the rendering
        let coeff_str = render_num_value(&pos_r);

        let rest_factors: Vec<String> = children[1..]
            .iter()
            .map(|&c| render_mul_factor(arena, c))
            .collect();

        if pos_r == Ratio::from(BigInt::from(1)) {
            // Coefficient is 1 after negation → just render rest
            if rest_factors.is_empty() {
                return "1".to_string();
            }
            return join_mul_factors(&rest_factors);
        }

        let mut all_factors = vec![coeff_str];
        all_factors.extend(rest_factors);

        // Check for fraction rendering (negative powers in rest)
        // For simplicity, delegate to render_mul with modified children
        // Actually, let's just join them
        return join_mul_factors(&all_factors);
    }
    latex_to_string(arena, id)
}

// ═══════════════════════════════════════════════════════════════════════════
// Number rendering helper
// ═══════════════════════════════════════════════════════════════════════════

/// Render a rational number to LaTeX.
fn render_num_value(r: &Ratio<BigInt>) -> String {
    if r.is_integer() {
        format!("{}", r.numer())
    } else if r.numer().is_negative() {
        format!("-\\frac{{{}}}{{{}}}", -r.numer(), r.denom())
    } else {
        format!("\\frac{{{}}}{{{}}}", r.numer(), r.denom())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Node expansion
// ═══════════════════════════════════════════════════════════════════════════

/// Push a LaTeX function call `\name\left(arg\right)` onto the stack.
/// If `id` is a trig/hyperbolic function node, return (`\funcname`, arg).
fn trig_func_parts(arena: &Arena, id: ExprId) -> Option<(&'static str, ExprId)> {
    match arena.node(id) {
        ExprNode::Sin(x) => Some((r"\sin", *x)),
        ExprNode::Cos(x) => Some((r"\cos", *x)),
        ExprNode::Tan(x) => Some((r"\tan", *x)),
        ExprNode::Sinh(x) => Some((r"\sinh", *x)),
        ExprNode::Cosh(x) => Some((r"\cosh", *x)),
        ExprNode::Tanh(x) => Some((r"\tanh", *x)),
        ExprNode::Asin(x) => Some((r"\arcsin", *x)),
        ExprNode::Acos(x) => Some((r"\arccos", *x)),
        ExprNode::Atan(x) => Some((r"\arctan", *x)),
        ExprNode::Asinh(x) => Some((r"\operatorname{asinh}", *x)),
        ExprNode::Acosh(x) => Some((r"\operatorname{acosh}", *x)),
        ExprNode::Atanh(x) => Some((r"\operatorname{atanh}", *x)),
        _ => None,
    }
}

fn push_latex_func(name: &'static str, arg: ExprId, stack: &mut Vec<LatexItem>) {
    stack.push(LatexItem::Lit(r"\right)"));
    stack.push(LatexItem::Expr(arg));
    stack.push(LatexItem::Owned(format!("{}\\left(", name)));
}

/// Expand a single expression node into LaTeX work items on the stack.
///
/// Items are pushed in **reverse** display order so that popping
/// yields left-to-right output.
fn expand_latex(arena: &Arena, id: ExprId, stack: &mut Vec<LatexItem>) {
    let node = arena.node(id).clone();

    match node {
        // ── Atoms ──────────────────────────────────────────────────
        ExprNode::Num(nid) => {
            let r = arena.num(nid);
            stack.push(LatexItem::Owned(render_num_value(r)));
        }

        ExprNode::Symbol(sid) => {
            let name = arena.symbol_name(sid);
            stack.push(LatexItem::Owned(symbol_to_latex(name)));
        }

        ExprNode::Pi => stack.push(LatexItem::Lit(r"\pi")),
        ExprNode::E => stack.push(LatexItem::Lit("e")),
        ExprNode::ImaginaryUnit => stack.push(LatexItem::Lit("i")),
        ExprNode::PhysicalConstant(name_id, _) => {
            stack.push(LatexItem::Owned(symbol_to_latex(arena.symbol_name(name_id))));
        }
        ExprNode::Infinity => stack.push(LatexItem::Lit(r"\infty")),
        ExprNode::NegInfinity => stack.push(LatexItem::Lit(r"-\infty")),
        ExprNode::ComplexInfinity => stack.push(LatexItem::Lit(r"\tilde{\infty}")),
        ExprNode::NaN => stack.push(LatexItem::Lit(r"\text{NaN}")),

        // ── Add ────────────────────────────────────────────────────
        //
        // Rendered eagerly because we need to sort children and detect
        // subtraction patterns across the entire child list.
        ExprNode::Add(ref children) => {
            let children = children.clone();
            if children.is_empty() {
                stack.push(LatexItem::Lit("0"));
                return;
            }

            // Sort children by display key
            let mut display_order: SmallVec<[ExprId; 6]> = children;
            display_order.sort_by_key(|a| display_sort_key(arena, *a));

            let mut result = String::new();

            for (i, &child) in display_order.iter().enumerate() {
                let child_node = arena.node(child);

                if let ExprNode::Neg(inner) = child_node {
                    // Neg(x) → " - x"
                    let inner = *inner;
                    let inner_latex = latex_to_string(arena, inner);
                    if i == 0 {
                        // Wrap Add children of inner in parens
                        if matches!(arena.node(inner), ExprNode::Add(_)) {
                            result.push_str(&format!("-\\left({}\\right)", inner_latex));
                        } else {
                            result.push_str(&format!("-{}", inner_latex));
                        }
                    } else if matches!(arena.node(inner), ExprNode::Add(_)) {
                        result.push_str(&format!(" - \\left({}\\right)", inner_latex));
                    } else {
                        result.push_str(&format!(" - {}", inner_latex));
                    }
                } else if is_neg_one_mul(arena, child) {
                    // Mul([-1, rest...]) → " - rest"
                    let rest_latex = render_mul_without_neg_one(arena, child);
                    if i == 0 {
                        result.push_str(&format!("-{}", rest_latex));
                    } else {
                        result.push_str(&format!(" - {}", rest_latex));
                    }
                } else if is_neg_coeff_mul(arena, child) {
                    // Mul([-n, rest...]) → " - n*rest"
                    let sub_latex = render_neg_coeff_mul_as_subtraction(arena, child);
                    if i == 0 {
                        result.push_str(&format!("-{}", sub_latex));
                    } else {
                        result.push_str(&format!(" - {}", sub_latex));
                    }
                } else if i == 0 {
                    // Check if the child is a negative number
                    if let ExprNode::Num(nid) = arena.node(child) {
                        let r = arena.num(*nid);
                        if r.is_negative() && !r.is_integer() {
                            // Negative fraction at leading position
                            result.push_str(&render_num_value(r));
                        } else {
                            result.push_str(&latex_to_string(arena, child));
                        }
                    } else {
                        result.push_str(&latex_to_string(arena, child));
                    }
                } else {
                    // Check if child is a negative number
                    if let ExprNode::Num(nid) = arena.node(child) {
                        let r = arena.num(*nid);
                        if r.is_negative() {
                            let pos_r = -r.clone();
                            let pos_str = render_num_value(&pos_r);
                            result.push_str(&format!(" - {}", pos_str));
                        } else {
                            result.push_str(&format!(" + {}", latex_to_string(arena, child)));
                        }
                    } else {
                        result.push_str(&format!(" + {}", latex_to_string(arena, child)));
                    }
                }
            }

            stack.push(LatexItem::Owned(result));
        }

        // ── Mul ────────────────────────────────────────────────────
        //
        // Rendered eagerly because fraction detection requires scanning
        // all children.
        ExprNode::Mul(ref children) => {
            let children_vec: Vec<ExprId> = children.iter().copied().collect();
            stack.push(LatexItem::Owned(render_mul(arena, &children_vec)));
        }

        // ── Pow ────────────────────────────────────────────────────
        ExprNode::Pow(base, exp) => {
            // Special case: exp = 1/2 → \sqrt{base}
            if let ExprNode::Num(nid) = arena.node(exp) {
                let r = arena.num(*nid);
                if *r == Ratio::new(BigInt::from(1), BigInt::from(2)) {
                    stack.push(LatexItem::Lit("}"));
                    stack.push(LatexItem::Expr(base));
                    stack.push(LatexItem::Lit(r"\sqrt{"));
                    return;
                }
                // exp = 1/3 → \sqrt[3]{base}
                if *r == Ratio::new(BigInt::from(1), BigInt::from(3)) {
                    stack.push(LatexItem::Lit("}"));
                    stack.push(LatexItem::Expr(base));
                    stack.push(LatexItem::Lit(r"\sqrt[3]{"));
                    return;
                }
                // exp = 1/n → \sqrt[n]{base}
                if !r.is_integer()
                    && !r.is_negative()
                    && *r.numer() == BigInt::from(1)
                {
                    let n = r.denom();
                    stack.push(LatexItem::Lit("}"));
                    stack.push(LatexItem::Expr(base));
                    stack.push(LatexItem::Owned(format!("\\sqrt[{}]{{", n)));
                    return;
                }
                // exp = -1 → \frac{1}{base}
                if *r == Ratio::from(BigInt::from(-1)) {
                    let base_latex = render_pow_base(arena, base);
                    stack.push(LatexItem::Owned(format!(
                        "\\frac{{1}}{{{}}}",
                        base_latex
                    )));
                    return;
                }
                // exp = -n (negative integer, not -1) → \frac{1}{base^{n}}
                if r.is_integer() && r.is_negative() {
                    let pos_n = -r.numer();
                    let base_latex = render_pow_base(arena, base);
                    stack.push(LatexItem::Owned(format!(
                        "\\frac{{1}}{{{}^{{{}}}}}",
                        base_latex, pos_n
                    )));
                    return;
                }
            }

            // Trig function power: sin(x)^n → \sin^{n}\left(x\right)
            if let Some((func_name, arg)) = trig_func_parts(arena, base) {
                stack.push(LatexItem::Lit(r"\right)"));
                stack.push(LatexItem::Expr(arg));
                stack.push(LatexItem::Lit(r"}\left("));
                stack.push(LatexItem::Expr(exp));
                stack.push(LatexItem::Owned(format!("{}^{{", func_name)));
                return;
            }

            // General case: base^{exp}
            let base_latex = render_pow_base(arena, base);
            stack.push(LatexItem::Lit("}"));
            stack.push(LatexItem::Expr(exp));
            stack.push(LatexItem::Owned(format!("{}^{{", base_latex)));
        }

        // ── Neg ────────────────────────────────────────────────────
        ExprNode::Neg(inner) => {
            if matches!(arena.node(inner), ExprNode::Add(_)) {
                stack.push(LatexItem::Lit(r"\right)"));
                stack.push(LatexItem::Expr(inner));
                stack.push(LatexItem::Lit(r"-\left("));
            } else {
                stack.push(LatexItem::Expr(inner));
                stack.push(LatexItem::Lit("-"));
            }
        }

        // ── Trig functions ─────────────────────────────────────────
        ExprNode::Sin(x) => push_latex_func(r"\sin", x, stack),
        ExprNode::Cos(x) => push_latex_func(r"\cos", x, stack),
        ExprNode::Tan(x) => push_latex_func(r"\tan", x, stack),
        ExprNode::Asin(x) => push_latex_func(r"\arcsin", x, stack),
        ExprNode::Acos(x) => push_latex_func(r"\arccos", x, stack),
        ExprNode::Atan(x) => push_latex_func(r"\arctan", x, stack),
        ExprNode::Sinh(x) => push_latex_func(r"\sinh", x, stack),
        ExprNode::Cosh(x) => push_latex_func(r"\cosh", x, stack),
        ExprNode::Tanh(x) => push_latex_func(r"\tanh", x, stack),
        ExprNode::Asinh(x) => push_latex_func(r"\operatorname{asinh}", x, stack),
        ExprNode::Acosh(x) => push_latex_func(r"\operatorname{acosh}", x, stack),
        ExprNode::Atanh(x) => push_latex_func(r"\operatorname{atanh}", x, stack),

        // ── Transcendental ─────────────────────────────────────────
        ExprNode::Exp(x) => push_latex_func(r"\exp", x, stack),
        ExprNode::Ln(x) => push_latex_func(r"\ln", x, stack),

        // ── Special functions ──────────────────────────────────────
        ExprNode::Sign(x) => push_latex_func(r"\operatorname{sgn}", x, stack),
        ExprNode::Heaviside(x) => push_latex_func(r"\operatorname{H}", x, stack),
        ExprNode::DiracDelta(x) => push_latex_func(r"\delta", x, stack),
        ExprNode::Gamma(x) => push_latex_func(r"\Gamma", x, stack),
        ExprNode::LogGamma(x) => push_latex_func(r"\ln \Gamma", x, stack),
        ExprNode::Digamma(x) => push_latex_func(r"\psi", x, stack),
        ExprNode::Erf(x) => push_latex_func(r"\operatorname{erf}", x, stack),
        ExprNode::Erfc(x) => push_latex_func(r"\operatorname{erfc}", x, stack),
        ExprNode::LambertW(x) => push_latex_func(r"\operatorname{W}", x, stack),

        // ── Abs: \left|x\right| ───────────────────────────────────
        ExprNode::Abs(x) => {
            stack.push(LatexItem::Lit(r"\right|"));
            stack.push(LatexItem::Expr(x));
            stack.push(LatexItem::Lit(r"\left|"));
        }

        // ── Floor / Ceiling ────────────────────────────────────────
        ExprNode::Floor(x) => {
            stack.push(LatexItem::Lit(r"\rfloor"));
            stack.push(LatexItem::Expr(x));
            stack.push(LatexItem::Lit(r"\lfloor "));
        }
        ExprNode::Ceiling(x) => {
            stack.push(LatexItem::Lit(r"\rceil"));
            stack.push(LatexItem::Expr(x));
            stack.push(LatexItem::Lit(r"\lceil "));
        }

        // ── Factorial: n! ──────────────────────────────────────────
        ExprNode::Factorial(x) => {
            if arena.node(x).is_atom() {
                stack.push(LatexItem::Lit("!"));
                stack.push(LatexItem::Expr(x));
            } else {
                stack.push(LatexItem::Lit(r"\right)!"));
                stack.push(LatexItem::Expr(x));
                stack.push(LatexItem::Lit(r"\left("));
            }
        }

        // ── Binomial: \binom{n}{k} ────────────────────────────────
        ExprNode::Binomial(n, k) => {
            stack.push(LatexItem::Lit("}"));
            stack.push(LatexItem::Expr(k));
            stack.push(LatexItem::Lit("}{"));
            stack.push(LatexItem::Expr(n));
            stack.push(LatexItem::Lit(r"\binom{"));
        }

        // ── Beta: \mathrm{B}(a, b) ────────────────────────────────
        ExprNode::Beta(a, b) => {
            stack.push(LatexItem::Lit(r"\right)"));
            stack.push(LatexItem::Expr(b));
            stack.push(LatexItem::Lit(", "));
            stack.push(LatexItem::Expr(a));
            stack.push(LatexItem::Lit(r"\mathrm{B}\left("));
        }

        // ── Atan2 ──────────────────────────────────────────────────
        ExprNode::Atan2(y, x) => {
            stack.push(LatexItem::Lit(r"\right)"));
            stack.push(LatexItem::Expr(x));
            stack.push(LatexItem::Lit(", "));
            stack.push(LatexItem::Expr(y));
            stack.push(LatexItem::Lit(r"\operatorname{atan2}\left("));
        }

        // ── Min / Max ──────────────────────────────────────────────
        ExprNode::Min(ref args) => {
            let args = args.clone();
            stack.push(LatexItem::Lit(r"\right)"));
            for (i, &arg) in args.iter().enumerate().rev() {
                stack.push(LatexItem::Expr(arg));
                if i > 0 {
                    stack.push(LatexItem::Lit(", "));
                }
            }
            stack.push(LatexItem::Lit(r"\min\left("));
        }
        ExprNode::Max(ref args) => {
            let args = args.clone();
            stack.push(LatexItem::Lit(r"\right)"));
            for (i, &arg) in args.iter().enumerate().rev() {
                stack.push(LatexItem::Expr(arg));
                if i > 0 {
                    stack.push(LatexItem::Lit(", "));
                }
            }
            stack.push(LatexItem::Lit(r"\max\left("));
        }

        // ── Sum / Product ──────────────────────────────────────────
        ExprNode::Sum(body, var, lo, hi) => {
            stack.push(LatexItem::Expr(body));
            stack.push(LatexItem::Lit("} "));
            stack.push(LatexItem::Expr(hi));
            stack.push(LatexItem::Lit("^{"));
            stack.push(LatexItem::Expr(lo));
            stack.push(LatexItem::Lit("="));
            stack.push(LatexItem::Expr(var));
            stack.push(LatexItem::Lit(r"\sum_{"));
        }
        ExprNode::Product_(body, var, lo, hi) => {
            stack.push(LatexItem::Expr(body));
            stack.push(LatexItem::Lit("} "));
            stack.push(LatexItem::Expr(hi));
            stack.push(LatexItem::Lit("^{"));
            stack.push(LatexItem::Expr(lo));
            stack.push(LatexItem::Lit("="));
            stack.push(LatexItem::Expr(var));
            stack.push(LatexItem::Lit(r"\prod_{"));
        }

        // ── Derivative: \frac{d}{dvar} body ───────────────────────
        ExprNode::Derivative(body, var) => {
            stack.push(LatexItem::Expr(body));
            stack.push(LatexItem::Lit("} "));
            stack.push(LatexItem::Expr(var));
            stack.push(LatexItem::Lit(r"\frac{d}{d"));
        }

        // ── Integral: \int body \, dvar ───────────────────────────
        ExprNode::Integral(body, var) => {
            stack.push(LatexItem::Expr(var));
            stack.push(LatexItem::Lit(r"\, d"));
            stack.push(LatexItem::Expr(body));
            stack.push(LatexItem::Lit(r"\int "));
        }

        // ── Apply (user-defined function) ──────────────────────────
        ExprNode::Apply(sym_id, ref args) => {
            let name = arena.symbol_name(sym_id).to_owned();
            let args = args.clone();
            stack.push(LatexItem::Lit(r"\right)"));
            for (i, &arg) in args.iter().enumerate().rev() {
                stack.push(LatexItem::Expr(arg));
                if i > 0 {
                    stack.push(LatexItem::Lit(", "));
                }
            }
            stack.push(LatexItem::Owned(format!("{}\\left(", name)));
        }

        // ── Piecewise ──────────────────────────────────────────────
        ExprNode::Piecewise(ref pieces) => {
            let pieces = pieces.clone();
            let mut result = String::from("\\begin{cases}\n");
            for (i, &(val, cond)) in pieces.iter().enumerate() {
                let val_latex = latex_to_string(arena, val);
                let cond_latex = latex_to_string(arena, cond);
                result.push_str(&format!("  {} & \\text{{if }} {}", val_latex, cond_latex));
                if i + 1 < pieces.len() {
                    result.push_str(" \\\\\n");
                } else {
                    result.push('\n');
                }
            }
            result.push_str("\\end{cases}");
            stack.push(LatexItem::Owned(result));
        }

        // ── Boolean atoms ──────────────────────────────────────────
        ExprNode::BoolTrue => stack.push(LatexItem::Lit(r"\text{True}")),
        ExprNode::BoolFalse => stack.push(LatexItem::Lit(r"\text{False}")),

        // ── Relational operators ───────────────────────────────────
        ExprNode::Gt(a, b) => {
            stack.push(LatexItem::Expr(b));
            stack.push(LatexItem::Lit(" > "));
            stack.push(LatexItem::Expr(a));
        }
        ExprNode::Ge(a, b) => {
            stack.push(LatexItem::Expr(b));
            stack.push(LatexItem::Lit(r" \geq "));
            stack.push(LatexItem::Expr(a));
        }
        ExprNode::Eq_(a, b) => {
            stack.push(LatexItem::Expr(b));
            stack.push(LatexItem::Lit(" = "));
            stack.push(LatexItem::Expr(a));
        }
        ExprNode::Ne(a, b) => {
            stack.push(LatexItem::Expr(b));
            stack.push(LatexItem::Lit(r" \neq "));
            stack.push(LatexItem::Expr(a));
        }

        // ── Logical connectives ────────────────────────────────────
        ExprNode::And(ref children) => {
            let children = children.clone();
            for (i, &child) in children.iter().enumerate().rev() {
                stack.push(LatexItem::Expr(child));
                if i > 0 {
                    stack.push(LatexItem::Lit(r" \land "));
                }
            }
        }
        ExprNode::Or(ref children) => {
            let children = children.clone();
            for (i, &child) in children.iter().enumerate().rev() {
                stack.push(LatexItem::Expr(child));
                if i > 0 {
                    stack.push(LatexItem::Lit(r" \lor "));
                }
            }
        }
        ExprNode::Not(x) => {
            stack.push(LatexItem::Expr(x));
            stack.push(LatexItem::Lit(r"\lnot "));
        }

        // ── Set atoms ──────────────────────────────────────────────
        ExprNode::EmptySet => stack.push(LatexItem::Lit(r"\emptyset")),
        ExprNode::UniversalSet => stack.push(LatexItem::Lit(r"\mathbb{R}")),

        // ── Set constructors — fallback to Display ─────────────────
        ExprNode::Interval(a, b, flags) => {
            let left = if flags & crate::base::node::INTERVAL_LEFT_OPEN != 0 {
                "("
            } else {
                "["
            };
            let right = if flags & crate::base::node::INTERVAL_RIGHT_OPEN != 0 {
                ")"
            } else {
                "]"
            };
            let a_latex = latex_to_string(arena, a);
            let b_latex = latex_to_string(arena, b);
            stack.push(LatexItem::Owned(format!(
                "{}{}, {}{}",
                left, a_latex, b_latex, right
            )));
        }

        ExprNode::FiniteSet(ref elems) => {
            let elems = elems.clone();
            stack.push(LatexItem::Lit("\\}"));
            for (i, &elem) in elems.iter().enumerate().rev() {
                stack.push(LatexItem::Expr(elem));
                if i > 0 {
                    stack.push(LatexItem::Lit(", "));
                }
            }
            stack.push(LatexItem::Lit("\\{"));
        }

        ExprNode::SetUnion(ref sets) => {
            let sets = sets.clone();
            for (i, &set) in sets.iter().enumerate().rev() {
                stack.push(LatexItem::Expr(set));
                if i > 0 {
                    stack.push(LatexItem::Lit(r" \cup "));
                }
            }
        }

        ExprNode::SetIntersection(ref sets) => {
            let sets = sets.clone();
            for (i, &set) in sets.iter().enumerate().rev() {
                stack.push(LatexItem::Expr(set));
                if i > 0 {
                    stack.push(LatexItem::Lit(r" \cap "));
                }
            }
        }

        ExprNode::SetComplement(a, b) => {
            stack.push(LatexItem::Expr(b));
            stack.push(LatexItem::Lit(r" \setminus "));
            stack.push(LatexItem::Expr(a));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public API on Expr<S>
// ═══════════════════════════════════════════════════════════════════════════

impl<S: Sort> Expr<S> {
    /// Render this expression as a LaTeX math string (no delimiters).
    ///
    /// Uses direct `ExprNode` matching for robustness.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let x = symplex::var("x");
    /// assert_eq!(x.powi(2).to_latex(), r"x^{2}");
    /// assert_eq!(x.sin().to_latex(), r"\sin\left(x\right)");
    /// ```
    pub fn to_latex(&self) -> String {
        let inner = self.inner.read();
        let w = LatexWriter {
            arena: &inner.arena,
            id: self.id,
        };
        format!("{}", w)
    }

    /// Render as an inline LaTeX expression with `$` delimiters.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let x = symplex::var("x");
    /// assert_eq!(x.to_latex_inline(), "$x$");
    /// ```
    pub fn to_latex_inline(&self) -> String {
        format!("${}$", self.to_latex())
    }

    /// Render as a display LaTeX expression with `$$` delimiters.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let x = symplex::var("x");
    /// assert_eq!(x.to_latex_display(), "$$x$$");
    /// ```
    pub fn to_latex_display(&self) -> String {
        format!("$${}$$", self.to_latex())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    // ── Number tests ───────────────────────────────────────────────

    #[test]
    fn latex_integer() {
        assert_eq!(symplex::int(42).to_latex(), "42");
    }

    #[test]
    fn latex_zero() {
        assert_eq!(symplex::int(0).to_latex(), "0");
    }

    #[test]
    fn latex_negative_integer() {
        let neg = symplex::int(-3);
        let latex = neg.to_latex();
        assert!(latex == "-3", "got: {latex}");
    }

    #[test]
    fn latex_fraction() {
        let half = symplex::rational(1, 2);
        assert_eq!(half.to_latex(), r"\frac{1}{2}");
    }

    #[test]
    fn latex_negative_fraction() {
        let neg_frac = symplex::rational(-5, 7);
        let latex = neg_frac.to_latex();
        assert!(
            latex == r"-\frac{5}{7}" || latex == r"\frac{-5}{7}",
            "got: {latex}"
        );
    }

    // ── Symbol tests ───────────────────────────────────────────────

    #[test]
    fn latex_symbol() {
        assert_eq!(symplex::var("x").to_latex(), "x");
    }

    #[test]
    fn latex_symbol_multichar() {
        assert_eq!(symplex::var("foo").to_latex(), "foo");
    }

    #[test]
    fn latex_greek_theta() {
        assert_eq!(symplex::var("theta").to_latex(), r"\theta");
    }

    #[test]
    fn latex_greek_alpha() {
        assert_eq!(symplex::var("alpha").to_latex(), r"\alpha");
    }

    #[test]
    fn latex_greek_omega() {
        assert_eq!(symplex::var("omega").to_latex(), r"\omega");
    }

    #[test]
    fn latex_greek_lambda() {
        assert_eq!(symplex::var("lambda").to_latex(), r"\lambda");
    }

    #[test]
    fn latex_symbol_subscript() {
        let x1 = symplex::var("x_1");
        assert_eq!(x1.to_latex(), "x_{1}");
    }

    // ── Constant tests ─────────────────────────────────────────────

    #[test]
    fn latex_pi() {
        assert_eq!(symplex::pi().to_latex(), r"\pi");
    }

    #[test]
    fn latex_e() {
        assert_eq!(symplex::e().to_latex(), "e");
    }

    #[test]
    fn latex_imaginary() {
        assert_eq!(symplex::i_unit().to_latex(), "i");
    }

    #[test]
    fn latex_infinity() {
        assert_eq!(symplex::infinity().to_latex(), r"\infty");
    }

    #[test]
    fn latex_neg_infinity() {
        assert_eq!(symplex::neg_infinity().to_latex(), r"-\infty");
    }

    // ── Pow tests ──────────────────────────────────────────────────

    #[test]
    fn latex_power() {
        let x = symplex::var("x");
        let expr = x.powi(2);
        assert_eq!(expr.to_latex(), r"x^{2}");
    }

    #[test]
    fn latex_power_cube() {
        let x = symplex::var("x");
        let expr = x.powi(3);
        assert_eq!(expr.to_latex(), r"x^{3}");
    }

    #[test]
    fn latex_sqrt() {
        let x = symplex::var("x");
        let expr = x.sqrt();
        assert_eq!(expr.to_latex(), r"\sqrt{x}");
    }

    #[test]
    fn latex_cbrt() {
        let x = symplex::var("x");
        let expr = x.cbrt();
        assert_eq!(expr.to_latex(), r"\sqrt[3]{x}");
    }

    #[test]
    fn latex_inverse() {
        let x = symplex::var("x");
        let expr = x.powi(-1);
        let latex = expr.to_latex();
        assert_eq!(latex, r"\frac{1}{x}");
    }

    // ── Neg tests ──────────────────────────────────────────────────

    #[test]
    fn latex_neg_symbol() {
        let x = symplex::var("x");
        let expr = -&x;
        assert_eq!(expr.to_latex(), "-x");
    }

    // ── Add tests ──────────────────────────────────────────────────

    #[test]
    fn latex_add_simple() {
        let x = symplex::var("x");
        let expr = &x + 1;
        let latex = expr.to_latex();
        assert!(
            latex == "x + 1" || latex == "1 + x",
            "got: {latex}"
        );
    }

    #[test]
    fn latex_add_power_and_const() {
        let x = symplex::var("x");
        let expr = x.powi(2) + 1;
        let latex = expr.to_latex();
        assert!(
            latex == r"x^{2} + 1" || latex == r"1 + x^{2}",
            "got: {latex}"
        );
    }

    // ── Mul tests ──────────────────────────────────────────────────

    #[test]
    fn latex_mul_coefficient() {
        let x = symplex::var("x");
        let expr = &x * 2;
        let latex = expr.to_latex();
        assert!(
            latex == "2 x" || latex == "x \\cdot 2" || latex == "2x",
            "got: {latex}"
        );
    }

    // ── Trig function tests ────────────────────────────────────────

    #[test]
    fn latex_sin() {
        let x = symplex::var("x");
        let expr = x.sin();
        assert_eq!(expr.to_latex(), r"\sin\left(x\right)");
    }

    #[test]
    fn latex_cos() {
        let x = symplex::var("x");
        let expr = x.cos();
        assert_eq!(expr.to_latex(), r"\cos\left(x\right)");
    }

    #[test]
    fn latex_tan() {
        let x = symplex::var("x");
        let expr = x.tan();
        assert_eq!(expr.to_latex(), r"\tan\left(x\right)");
    }

    #[test]
    fn latex_sinh() {
        let x = symplex::var("x");
        let expr = x.sinh();
        assert_eq!(expr.to_latex(), r"\sinh\left(x\right)");
    }

    #[test]
    fn latex_cosh() {
        let x = symplex::var("x");
        let expr = x.cosh();
        assert_eq!(expr.to_latex(), r"\cosh\left(x\right)");
    }

    // ── Exp/Ln tests ───────────────────────────────────────────────

    #[test]
    fn latex_exp() {
        let x = symplex::var("x");
        let expr = x.exp();
        assert_eq!(expr.to_latex(), r"\exp\left(x\right)");
    }

    #[test]
    fn latex_ln() {
        let x = symplex::var("x");
        let expr = x.ln();
        assert_eq!(expr.to_latex(), r"\ln\left(x\right)");
    }

    // ── Abs test ───────────────────────────────────────────────────

    #[test]
    fn latex_abs() {
        let x = symplex::var("x");
        let expr = x.abs();
        assert_eq!(expr.to_latex(), r"\left|x\right|");
    }

    // ── Inverse trig ───────────────────────────────────────────────

    #[test]
    fn latex_asin() {
        let x = symplex::var("x");
        let expr = x.asin();
        assert_eq!(expr.to_latex(), r"\arcsin\left(x\right)");
    }

    #[test]
    fn latex_acos() {
        let x = symplex::var("x");
        let expr = x.acos();
        assert_eq!(expr.to_latex(), r"\arccos\left(x\right)");
    }

    #[test]
    fn latex_atan() {
        let x = symplex::var("x");
        let expr = x.atan();
        assert_eq!(expr.to_latex(), r"\arctan\left(x\right)");
    }

    // ── Derivative test ────────────────────────────────────────────

    #[test]
    fn latex_derivative() {
        let x = symplex::var("x");
        let expr = x.powi(2).formal_diff(&x);
        let latex = expr.to_latex();
        assert_eq!(latex, r"\frac{d}{dx} x^{2}");
    }

    // ── Integral test ──────────────────────────────────────────────

    #[test]
    fn latex_integral() {
        let x = symplex::var("x");
        let expr = x.sin().sin();
        let integral = expr.integrate(&x);
        let latex = integral.to_latex();
        if integral.expr_type() == crate::api::expr::ExprType::Integral {
            assert!(latex.starts_with(r"\int"), "got: {latex}");
            assert!(latex.ends_with(r"\, dx"), "got: {latex}");
        }
        assert!(!latex.is_empty());
    }

    // ── Compound expressions ───────────────────────────────────────

    #[test]
    fn latex_sin_squared() {
        let x = symplex::var("x");
        let expr = x.sin().powi(2);
        let latex = expr.to_latex();
        assert_eq!(latex, r"\sin^{2}\left(x\right)");
    }

    #[test]
    fn latex_nested_function() {
        let x = symplex::var("x");
        let expr = x.sin().exp();
        let latex = expr.to_latex();
        assert_eq!(latex, r"\exp\left(\sin\left(x\right)\right)");
    }

    // ── Delimiter tests ────────────────────────────────────────────

    #[test]
    fn latex_inline_delimiters() {
        let x = symplex::var("x");
        assert_eq!(x.to_latex_inline(), "$x$");
    }

    #[test]
    fn latex_display_delimiters() {
        let x = symplex::var("x");
        assert_eq!(x.to_latex_display(), "$$x$$");
    }

    // ── Boolean / Relational ───────────────────────────────────────

    #[test]
    fn latex_relational_gt() {
        let x = symplex::var("x");
        let y = symplex::var("y");
        let expr = x.gt(&y);
        assert_eq!(expr.to_latex(), "x > y");
    }

    #[test]
    fn latex_relational_eq() {
        let x = symplex::var("x");
        let expr = x.eq_expr(&symplex::int(0));
        assert_eq!(expr.to_latex(), "x = 0");
    }

    // ── Deep expression (stack safety) ─────────────────────────────

    #[test]
    fn latex_deep_expression_no_stack_overflow() {
        let x = symplex::var("x");
        let mut expr = x.clone();
        for _ in 0..1000 {
            expr = expr.sin();
        }
        let latex = expr.to_latex();
        assert!(latex.starts_with(r"\sin\left("));
        assert!(latex.ends_with(r"\right)"));
    }

    // ── Add with subtraction ───────────────────────────────────────

    #[test]
    fn latex_add_with_neg_term() {
        let x = symplex::var("x");
        let y = symplex::var("y");
        let expr = &x - &y;
        let latex = expr.to_latex();
        assert!(
            latex == "x - y" || latex == "-y + x",
            "got: {latex}"
        );
    }

    // ── Binomial ───────────────────────────────────────────────────

    #[test]
    fn latex_binomial() {
        let ctx = Context::new();
        let n = ctx.symbol("n");
        let k = ctx.symbol("k");
        let expr = n.binomial(&k);
        let latex = expr.to_latex();
        assert_eq!(latex, r"\binom{n}{k}");
    }

    // ── Factorial ──────────────────────────────────────────────────

    #[test]
    fn latex_factorial() {
        let ctx = Context::new();
        let n = ctx.symbol("n");
        let expr = n.factorial();
        let latex = expr.to_latex();
        assert_eq!(latex, "n!");
    }

    // ── Mul coefficient spacing ────────────────────────────────────

    #[test]
    fn latex_coefficient_no_space() {
        let x = symplex::var("x");
        let expr = &x * 3; // 3*x
        let latex = expr.to_latex();
        // Should be "3x" — no "\cdot" for integer coefficient times variable
        assert!(
            !latex.contains(r"\cdot"),
            "coefficient times variable shouldn't use cdot: {latex}"
        );
    }
}
