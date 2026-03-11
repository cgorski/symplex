//! 2D Unicode pretty printer for symbolic expressions.
//!
//! Renders expressions as multi-line text art with proper baseline alignment,
//! stacked fractions, superscripts, and height-matched parentheses.
//!
//! # Architecture
//!
//! The printer has two layers:
//!
//! 1. **[`MathBox`]** — a rectangular 2D text box with a baseline row.
//!    Composition operations (`hcat`, `frac`, `superscript`, `parens`)
//!    produce new boxes from existing ones, preserving baseline metadata
//!    through every transformation.
//!
//! 2. **[`pretty_print`]** — walks `ExprNode` variants bottom-up and
//!    builds `MathBox` compositions.
//!
//! # Dual-mode rendering
//!
//! The printer supports both Unicode and ASCII modes. Unicode mode uses
//! box-drawing characters for fraction bars, bracket pieces for tall
//! parentheses, and the middle-dot for multiplication. ASCII mode uses
//! `-`, `(`, `|`, `)`, and `*` respectively.
//!
//! # Design decisions (from expert typography consultation)
//!
//! - Fraction bar = baseline (equivalent to TeX's math axis for monospace)
//! - Superscripts placed ABOVE the entire base box, not aligned with numerator
//! - Graduated fraction bar weights: heavy `━` → normal `─` → dashed `╌`
//! - No combining overline for sqrt — uses flat `sqrt(...)` notation
//! - Width-scoped fraction bars span only their own content
//! - Nesting depth > 2 falls back to slashed fractions `a/b`

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed};

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};

// ═══════════════════════════════════════════════════════════════════════════
// Render mode
// ═══════════════════════════════════════════════════════════════════════════

/// Controls whether Unicode or ASCII characters are used for rendering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum RenderMode {
    /// Use Unicode box-drawing, bracket pieces, middle-dot, etc.
    #[default]
    Unicode,
    /// Use only ASCII: `-`, `(`, `|`, `)`, `*`, `^`.
    Ascii,
}

// ═══════════════════════════════════════════════════════════════════════════
// MathBox — the core 2D layout type
// ═══════════════════════════════════════════════════════════════════════════

/// A rectangular 2D text box with a baseline for alignment.
///
/// The `baseline` row is the "math axis" — the row that aligns with
/// surrounding operators and text when boxes are composed horizontally.
/// For fractions, this is the fraction bar row. For plain text, it's
/// row 0.
#[derive(Clone, Debug)]
pub(crate) struct MathBox {
    /// Lines of equal width (padded with spaces).
    lines: Vec<String>,
    /// The row index of the alignment line (0-based).
    baseline: usize,
    /// Nesting depth for fraction bar style selection.
    nesting_depth: u8,
}

impl MathBox {
    // ── Constructors ───────────────────────────────────────────────

    /// Create a single-line box from a string.
    pub fn text(s: &str) -> Self {
        MathBox {
            lines: vec![s.to_string()],
            baseline: 0,
            nesting_depth: 0,
        }
    }

    /// Create an empty box (zero width, zero height).
    pub fn empty() -> Self {
        MathBox {
            lines: vec![String::new()],
            baseline: 0,
            nesting_depth: 0,
        }
    }

    // ── Queries ────────────────────────────────────────────────────

    /// Width in characters.
    pub fn width(&self) -> usize {
        self.lines.first().map_or(0, |l| l.chars().count())
    }

    /// Height in rows.
    pub fn height(&self) -> usize {
        self.lines.len()
    }

    /// Rows above the baseline (ascent).
    pub fn ascent(&self) -> usize {
        self.baseline
    }

    /// Rows below the baseline (descent), not counting the baseline row.
    pub fn descent(&self) -> usize {
        if self.height() == 0 {
            0
        } else {
            self.height() - self.baseline - 1
        }
    }

    // ── Rendering ──────────────────────────────────────────────────

    /// Join lines with newline to produce the final string.
    pub fn render(&self) -> String {
        self.lines.join("\n")
    }

    // ── Composition operations ─────────────────────────────────────

    /// Horizontal concatenation: place boxes side-by-side, aligned on baselines.
    ///
    /// Shorter boxes are padded with blank rows above and below as needed
    /// so that all baselines line up on the same row.
    pub fn hcat(boxes: &[MathBox]) -> MathBox {
        if boxes.is_empty() {
            return MathBox::empty();
        }
        if boxes.len() == 1 {
            return boxes[0].clone();
        }

        // The combined baseline is the maximum ascent across all boxes.
        let new_baseline = boxes.iter().map(|b| b.ascent()).max().unwrap_or(0);
        // Total height = max ascent + max descent + 1 (for the baseline row).
        let max_descent = boxes.iter().map(|b| b.descent()).max().unwrap_or(0);
        let new_height = new_baseline + max_descent + 1;
        // Max nesting depth.
        let new_depth = boxes.iter().map(|b| b.nesting_depth).max().unwrap_or(0);

        // Pad each box to new_height, aligning on baselines.
        let padded: Vec<Vec<String>> = boxes
            .iter()
            .map(|b| {
                let w = b.width();
                let blank = " ".repeat(w);
                let pad_above = new_baseline - b.ascent();
                let pad_below = new_height - pad_above - b.height();
                let mut rows = Vec::with_capacity(new_height);
                for _ in 0..pad_above {
                    rows.push(blank.clone());
                }
                for line in &b.lines {
                    rows.push(line.clone());
                }
                for _ in 0..pad_below {
                    rows.push(blank.clone());
                }
                rows
            })
            .collect();

        // Zip rows across all padded boxes.
        let lines: Vec<String> = (0..new_height)
            .map(|row| {
                padded
                    .iter()
                    .map(|p| p[row].as_str())
                    .collect::<Vec<_>>()
                    .join("")
            })
            .collect();

        MathBox {
            lines,
            baseline: new_baseline,
            nesting_depth: new_depth,
        }
    }

    /// Stacked fraction: numerator over denominator with a horizontal bar.
    ///
    /// The bar character is selected based on `nesting_depth`:
    /// - 0: heavy `━` (U+2501) / `=` in ASCII
    /// - 1: normal `─` (U+2500) / `-` in ASCII
    /// - 2+: dashed `╌` (U+254C) / `-` in ASCII
    pub fn frac(numer: MathBox, denom: MathBox, nesting_depth: u8, mode: RenderMode) -> MathBox {
        let content_width = numer.width().max(denom.width());
        // Bar extends 1 char padding on each side.
        let bar_width = content_width + 2;

        let bar_char = match mode {
            RenderMode::Unicode => match nesting_depth {
                0 => '━',
                1 => '─',
                _ => '╌',
            },
            RenderMode::Ascii => '-',
        };
        let bar: String = std::iter::repeat_n(bar_char, bar_width).collect();

        // Center numerator and denominator within bar_width.
        let numer_centered = center_box(&numer, bar_width);
        let denom_centered = center_box(&denom, bar_width);

        let mut lines = Vec::new();
        for l in &numer_centered.lines {
            lines.push(l.clone());
        }
        lines.push(bar);
        for l in &denom_centered.lines {
            lines.push(l.clone());
        }

        let baseline = numer_centered.height(); // The bar row.

        MathBox {
            lines,
            baseline,
            nesting_depth: nesting_depth + 1,
        }
    }

    /// Superscript: place `exp` above-right of `base`.
    ///
    /// The exponent sits above the top row of the base, shifted right by
    /// the base's width. The resulting baseline is the base's baseline,
    /// shifted down by the exponent's height.
    pub fn superscript(base: MathBox, exp: MathBox) -> MathBox {
        let exp_w = exp.width();
        let base_w = base.width();
        let total_w = base_w + exp_w;
        let total_h = exp.height() + base.height();

        let mut lines = Vec::with_capacity(total_h);

        // Exponent rows: padded on the left with base_w spaces.
        let left_pad: String = " ".repeat(base_w);
        for l in &exp.lines {
            let mut row = left_pad.clone();
            row.push_str(l);
            // Pad right to total_w.
            while row.chars().count() < total_w {
                row.push(' ');
            }
            lines.push(row);
        }

        // Base rows: padded on the right with exp_w spaces.
        for l in &base.lines {
            let mut row = l.clone();
            let right_pad: String = " ".repeat(exp_w);
            row.push_str(&right_pad);
            while row.chars().count() < total_w {
                row.push(' ');
            }
            lines.push(row);
        }

        // Baseline is the base's baseline, offset by the exponent height.
        let baseline = exp.height() + base.baseline;

        MathBox {
            lines,
            baseline,
            nesting_depth: base.nesting_depth.max(exp.nesting_depth),
        }
    }

    /// Wrap in height-matched parentheses.
    pub fn parens(inner: MathBox, mode: RenderMode) -> MathBox {
        let h = inner.height();
        if h <= 1 {
            // Single-line: use plain parens.
            let s = format!("({})", inner.lines.first().map_or("", |l| l.as_str()));
            return MathBox {
                lines: vec![s],
                baseline: inner.baseline,
                nesting_depth: inner.nesting_depth,
            };
        }

        let (top_l, mid_l, bot_l, top_r, mid_r, bot_r) = match mode {
            RenderMode::Unicode => ('⎛', '⎜', '⎝', '⎞', '⎟', '⎠'),
            RenderMode::Ascii => ('/', '|', '\\', '\\', '|', '/'),
        };

        let mut lines = Vec::with_capacity(h);
        for i in 0..h {
            let lc = if i == 0 {
                top_l
            } else if i == h - 1 {
                bot_l
            } else {
                mid_l
            };
            let rc = if i == 0 {
                top_r
            } else if i == h - 1 {
                bot_r
            } else {
                mid_r
            };
            lines.push(format!("{}{}{}", lc, &inner.lines[i], rc));
        }

        MathBox {
            lines,
            baseline: inner.baseline,
            nesting_depth: inner.nesting_depth,
        }
    }

    /// Absolute value bars: `|...|` with height-matched bars.
    pub fn abs_bars(inner: MathBox, mode: RenderMode) -> MathBox {
        let bar = match mode {
            RenderMode::Unicode => '│',
            RenderMode::Ascii => '|',
        };
        let lines: Vec<String> = inner
            .lines
            .iter()
            .map(|l| format!("{}{}{}", bar, l, bar))
            .collect();

        MathBox {
            lines,
            baseline: inner.baseline,
            nesting_depth: inner.nesting_depth,
        }
    }
}

/// Center a box horizontally within a given width, padding with spaces.
fn center_box(b: &MathBox, target_width: usize) -> MathBox {
    let w = b.width();
    if w >= target_width {
        return b.clone();
    }
    let left_pad = (target_width - w) / 2;
    let right_pad = target_width - w - left_pad;
    let left: String = " ".repeat(left_pad);
    let right: String = " ".repeat(right_pad);
    let lines: Vec<String> = b
        .lines
        .iter()
        .map(|l| format!("{}{}{}", left, l, right))
        .collect();
    MathBox {
        lines,
        baseline: b.baseline,
        nesting_depth: b.nesting_depth,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pretty printer — ExprNode → MathBox
// ═══════════════════════════════════════════════════════════════════════════

/// Render an expression as a 2D `MathBox`.
pub(crate) fn pretty_print(arena: &Arena, id: ExprId, mode: RenderMode) -> MathBox {
    pretty_node(arena, id, mode, 0)
}

/// Recursive (bounded by expression depth) renderer.
///
/// `depth` is the fraction nesting depth for bar-weight selection.
fn pretty_node(arena: &Arena, id: ExprId, mode: RenderMode, depth: u8) -> MathBox {
    let node = arena.node(id).clone();
    match node {
        // ── Atoms ──────────────────────────────────────────────────
        ExprNode::Num(nid) => pretty_num(arena, nid, mode, depth),
        ExprNode::Symbol(sid) => MathBox::text(arena.symbol_name(sid)),
        ExprNode::Pi => MathBox::text(if mode == RenderMode::Unicode {
            "π"
        } else {
            "pi"
        }),
        ExprNode::E => MathBox::text("e"),
        ExprNode::ImaginaryUnit => MathBox::text("i"),
        ExprNode::Infinity => MathBox::text("∞"),
        ExprNode::NegInfinity => MathBox::text("-∞"),
        ExprNode::ComplexInfinity => MathBox::text("zoo"),
        ExprNode::NaN => MathBox::text("NaN"),
        ExprNode::BoolTrue => MathBox::text("True"),
        ExprNode::BoolFalse => MathBox::text("False"),

        // ── Add ────────────────────────────────────────────────────
        ExprNode::Add(ref children) => pretty_add(arena, children, mode, depth),

        // ── Mul ────────────────────────────────────────────────────
        ExprNode::Mul(ref children) => pretty_mul(arena, children, mode, depth),

        // ── Neg ────────────────────────────────────────────────────
        ExprNode::Neg(inner) => {
            let inner_box = pretty_node(arena, inner, mode, depth);
            let minus = MathBox::text("-");
            let needs_parens = matches!(arena.node(inner), ExprNode::Add(_));
            if needs_parens {
                MathBox::hcat(&[minus, MathBox::parens(inner_box, mode)])
            } else {
                MathBox::hcat(&[minus, inner_box])
            }
        }

        // ── Pow ────────────────────────────────────────────────────
        ExprNode::Pow(base, exp) => pretty_pow(arena, base, exp, mode, depth),

        // ── Abs ────────────────────────────────────────────────────
        ExprNode::Abs(inner) => {
            let inner_box = pretty_node(arena, inner, mode, depth);
            MathBox::abs_bars(inner_box, mode)
        }

        // ── Unary functions ────────────────────────────────────────
        ExprNode::Sin(inner) => pretty_func("sin", arena, inner, mode, depth),
        ExprNode::Cos(inner) => pretty_func("cos", arena, inner, mode, depth),
        ExprNode::Tan(inner) => pretty_func("tan", arena, inner, mode, depth),
        ExprNode::Asin(inner) => pretty_func("asin", arena, inner, mode, depth),
        ExprNode::Acos(inner) => pretty_func("acos", arena, inner, mode, depth),
        ExprNode::Atan(inner) => pretty_func("atan", arena, inner, mode, depth),
        ExprNode::Sinh(inner) => pretty_func("sinh", arena, inner, mode, depth),
        ExprNode::Cosh(inner) => pretty_func("cosh", arena, inner, mode, depth),
        ExprNode::Tanh(inner) => pretty_func("tanh", arena, inner, mode, depth),
        ExprNode::Asinh(inner) => pretty_func("asinh", arena, inner, mode, depth),
        ExprNode::Acosh(inner) => pretty_func("acosh", arena, inner, mode, depth),
        ExprNode::Atanh(inner) => pretty_func("atanh", arena, inner, mode, depth),
        ExprNode::Exp(inner) => pretty_func("exp", arena, inner, mode, depth),
        ExprNode::Ln(inner) => pretty_func("ln", arena, inner, mode, depth),
        ExprNode::Gamma(inner) => {
            let name = if mode == RenderMode::Unicode {
                "Γ"
            } else {
                "Gamma"
            };
            pretty_func(name, arena, inner, mode, depth)
        }
        ExprNode::Erf(inner) => pretty_func("erf", arena, inner, mode, depth),
        ExprNode::Erfc(inner) => pretty_func("erfc", arena, inner, mode, depth),
        ExprNode::LambertW(inner) => pretty_func("W", arena, inner, mode, depth),
        ExprNode::LogGamma(inner) => pretty_func("lgamma", arena, inner, mode, depth),
        ExprNode::Digamma(inner) => {
            let name = if mode == RenderMode::Unicode {
                "ψ"
            } else {
                "psi"
            };
            pretty_func(name, arena, inner, mode, depth)
        }
        ExprNode::Floor(inner) => {
            let inner_box = pretty_node(arena, inner, mode, depth);
            let (l, r) = if mode == RenderMode::Unicode {
                ("⌊", "⌋")
            } else {
                ("floor(", ")")
            };
            MathBox::hcat(&[MathBox::text(l), inner_box, MathBox::text(r)])
        }
        ExprNode::Ceiling(inner) => {
            let inner_box = pretty_node(arena, inner, mode, depth);
            let (l, r) = if mode == RenderMode::Unicode {
                ("⌈", "⌉")
            } else {
                ("ceil(", ")")
            };
            MathBox::hcat(&[MathBox::text(l), inner_box, MathBox::text(r)])
        }
        ExprNode::Sign(inner) => pretty_func("sgn", arena, inner, mode, depth),
        ExprNode::Factorial(inner) => {
            let inner_box = pretty_node(arena, inner, mode, depth);
            let needs_parens = !matches!(arena.node(inner), ExprNode::Num(_) | ExprNode::Symbol(_));
            if needs_parens {
                MathBox::hcat(&[MathBox::parens(inner_box, mode), MathBox::text("!")])
            } else {
                MathBox::hcat(&[inner_box, MathBox::text("!")])
            }
        }

        // ── Fallback: use flat Display formatting ──────────────────
        _ => {
            let s = format!("{}", DisplayExpr(arena, id));
            MathBox::text(&s)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Number rendering
// ═══════════════════════════════════════════════════════════════════════════

fn pretty_num(
    arena: &Arena,
    nid: crate::base::node::NumId,
    mode: RenderMode,
    depth: u8,
) -> MathBox {
    let r = arena.num(nid);
    if r.is_integer() {
        MathBox::text(&r.numer().to_string())
    } else if depth < 3 {
        // Render as stacked fraction.
        let numer_box = MathBox::text(&r.numer().to_string());
        let denom_box = MathBox::text(&r.denom().to_string());
        MathBox::frac(numer_box, denom_box, depth, mode)
    } else {
        // Deep nesting: use slashed fraction.
        MathBox::text(&format!("{}/{}", r.numer(), r.denom()))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Add rendering
// ═══════════════════════════════════════════════════════════════════════════

fn pretty_add(arena: &Arena, children: &[ExprId], mode: RenderMode, depth: u8) -> MathBox {
    if children.is_empty() {
        return MathBox::text("0");
    }

    let mut parts: Vec<MathBox> = Vec::new();

    for (i, &child) in children.iter().enumerate() {
        if i > 0 {
            // Check if this term is negative.
            if let ExprNode::Neg(inner) = arena.node(child).clone() {
                parts.push(MathBox::text(" - "));
                let inner_box = pretty_node(arena, inner, mode, depth);
                parts.push(inner_box);
                continue;
            }
            if is_neg_coeff(arena, child) {
                parts.push(MathBox::text(" - "));
                let pos_box = pretty_negated_mul(arena, child, mode, depth);
                parts.push(pos_box);
                continue;
            }
            if let ExprNode::Num(nid) = arena.node(child) {
                let r = arena.num(*nid);
                if r.is_negative() {
                    let pos = -r;
                    if pos.is_integer() {
                        parts.push(MathBox::text(" - "));
                        parts.push(MathBox::text(&pos.numer().to_string()));
                        continue;
                    }
                }
            }
            parts.push(MathBox::text(" + "));
        }

        let child_box = pretty_node(arena, child, mode, depth);
        parts.push(child_box);
    }

    MathBox::hcat(&parts)
}

/// Check if a Mul node has a negative leading coefficient.
fn is_neg_coeff(arena: &Arena, id: ExprId) -> bool {
    if let ExprNode::Mul(ref children) = arena.node(id).clone()
        && let Some(&first) = children.first()
        && let ExprNode::Num(nid) = arena.node(first)
    {
        return arena.num(*nid).is_negative();
    }
    false
}

/// Render a Mul node with its leading negative coefficient negated.
fn pretty_negated_mul(arena: &Arena, id: ExprId, mode: RenderMode, depth: u8) -> MathBox {
    if let ExprNode::Mul(ref children) = arena.node(id).clone()
        && children.len() >= 2
        && let ExprNode::Num(nid) = arena.node(children[0])
    {
        let r = arena.num(*nid);
        let pos = -r;
        let mut parts: Vec<MathBox> = Vec::new();
        if !pos.is_one() {
            parts.push(pretty_ratio(&pos, mode, depth));
            parts.push(mul_dot(mode));
        }
        for (i, &factor) in children.iter().enumerate().skip(1) {
            if i > 1 {
                parts.push(mul_dot(mode));
            }
            parts.push(pretty_node(arena, factor, mode, depth));
        }
        return MathBox::hcat(&parts);
    }
    // Fallback.
    pretty_node(arena, id, mode, depth)
}

fn pretty_ratio(r: &Ratio<BigInt>, mode: RenderMode, depth: u8) -> MathBox {
    if r.is_integer() {
        MathBox::text(&r.numer().to_string())
    } else if depth < 3 {
        let n = MathBox::text(&r.numer().to_string());
        let d = MathBox::text(&r.denom().to_string());
        MathBox::frac(n, d, depth, mode)
    } else {
        MathBox::text(&format!("{}/{}", r.numer(), r.denom()))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Mul rendering
// ═══════════════════════════════════════════════════════════════════════════

fn pretty_mul(arena: &Arena, children: &[ExprId], mode: RenderMode, depth: u8) -> MathBox {
    if children.is_empty() {
        return MathBox::text("1");
    }

    // Separate numerator and denominator factors.
    let mut numer_factors: Vec<ExprId> = Vec::new();
    let mut denom_factors: Vec<ExprId> = Vec::new();

    for &child in children {
        if let ExprNode::Pow(base, exp) = arena.node(child).clone()
            && let Some(r) = arena.as_num(exp)
            && r.is_negative()
        {
            // This is base^(-|exp|) → goes in denominator as base^|exp|.
            let pos_exp_r = -r;
            if pos_exp_r.is_one() {
                denom_factors.push(base);
            } else {
                // We can't easily reconstruct base^pos_exp here without arena,
                // so just track the original and handle in rendering.
                denom_factors.push(child);
            }
            continue;
        }
        numer_factors.push(child);
    }

    // If there are denominator factors, render as a fraction.
    if !denom_factors.is_empty() {
        let numer_box = if numer_factors.is_empty() {
            MathBox::text("1")
        } else {
            render_product(arena, &numer_factors, mode, depth)
        };
        let denom_box = render_denom_product(arena, &denom_factors, mode, depth);
        return MathBox::frac(numer_box, denom_box, depth, mode);
    }

    // No denominator — pure product.
    render_product(arena, &numer_factors, mode, depth)
}

/// Render a product of factors with multiplication dots.
fn render_product(arena: &Arena, factors: &[ExprId], mode: RenderMode, depth: u8) -> MathBox {
    if factors.is_empty() {
        return MathBox::text("1");
    }

    let mut parts: Vec<MathBox> = Vec::new();
    let mut skip_dot_before_next = false;

    for (i, &factor) in factors.iter().enumerate() {
        // Leading numeric coefficient: don't put dot between number and symbol.
        if i == 0
            && let ExprNode::Num(nid) = arena.node(factor)
        {
            let r = arena.num(*nid);
            if r.is_one() && factors.len() > 1 {
                // Skip coefficient of 1.
                continue;
            }
            if (*r == Ratio::from(BigInt::from(-1))) && factors.len() > 1 {
                // Coefficient of -1: just show minus.
                parts.push(MathBox::text("-"));
                skip_dot_before_next = true;
                continue;
            }
            parts.push(pretty_ratio(r, mode, depth));
            skip_dot_before_next = false;
            continue;
        }

        if i > 0 && !skip_dot_before_next {
            parts.push(mul_dot(mode));
        }
        skip_dot_before_next = false;

        let factor_box = pretty_node(arena, factor, mode, depth);
        let needs_parens = matches!(arena.node(factor), ExprNode::Add(_));
        if needs_parens {
            parts.push(MathBox::parens(factor_box, mode));
        } else {
            parts.push(factor_box);
        }
    }

    MathBox::hcat(&parts)
}

/// Render denominator factors (strip negative exponents).
fn render_denom_product(arena: &Arena, factors: &[ExprId], mode: RenderMode, depth: u8) -> MathBox {
    if factors.len() == 1 {
        let factor = factors[0];
        // If it was base^(-n), render just base^n or base.
        if let ExprNode::Pow(base, exp) = arena.node(factor).clone()
            && let Some(r) = arena.as_num(exp)
        {
            let pos = -r;
            if pos.is_one() {
                return pretty_node(arena, base, mode, depth);
            }
            // base^pos_exp — render as base with superscript.
            let base_box = pretty_node(arena, base, mode, depth);
            let exp_box = pretty_ratio(&pos, mode, depth.saturating_add(1));
            return MathBox::superscript(base_box, exp_box);
        }
        return pretty_node(arena, factor, mode, depth);
    }

    // Multiple denominator factors: render as product.
    let mut parts: Vec<MathBox> = Vec::new();
    for (i, &factor) in factors.iter().enumerate() {
        if i > 0 {
            parts.push(mul_dot(mode));
        }
        // Strip negative exponent for display.
        if let ExprNode::Pow(base, exp) = arena.node(factor).clone()
            && let Some(r) = arena.as_num(exp)
        {
            let pos = -r;
            if pos.is_one() {
                parts.push(pretty_node(arena, base, mode, depth));
                continue;
            }
        }
        parts.push(pretty_node(arena, factor, mode, depth));
    }
    MathBox::hcat(&parts)
}

fn mul_dot(mode: RenderMode) -> MathBox {
    match mode {
        RenderMode::Unicode => MathBox::text("·"),
        RenderMode::Ascii => MathBox::text("*"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pow rendering
// ═══════════════════════════════════════════════════════════════════════════

fn pretty_pow(arena: &Arena, base: ExprId, exp: ExprId, mode: RenderMode, depth: u8) -> MathBox {
    // Special case: exp = 1/2 → sqrt
    if let Some(r) = arena.as_num(exp) {
        if *r == Ratio::new(BigInt::from(1), BigInt::from(2)) {
            let inner_box = pretty_node(arena, base, mode, depth);
            let sqrt_sym = if mode == RenderMode::Unicode {
                "√"
            } else {
                "sqrt"
            };
            return MathBox::hcat(&[
                MathBox::text(sqrt_sym),
                MathBox::text("("),
                inner_box,
                MathBox::text(")"),
            ]);
        }
        // exp = -1 → handled by Mul as denominator
        if *r == Ratio::from(BigInt::from(-1)) {
            let base_box = pretty_node(arena, base, mode, depth);
            let numer = MathBox::text("1");
            return MathBox::frac(numer, base_box, depth, mode);
        }
    }

    let base_box = pretty_node(arena, base, mode, depth);
    let exp_box = pretty_node(arena, exp, mode, depth.saturating_add(1));

    // If exponent is a single-line small integer, try Unicode superscript.
    if mode == RenderMode::Unicode
        && exp_box.height() == 1
        && let Some(r) = arena.as_num(exp)
        && r.is_integer()
    {
        let n = r.numer();
        // Only use Unicode superscripts for small non-negative integers
        // (avoid the EAW Ambiguous chars ¹²³⁴ for now — use 2D layout).
        if let Some(sup) = to_unicode_superscript_safe(n) {
            let needs_parens = matches!(
                arena.node(base),
                ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_)
            );
            let base_box = if needs_parens {
                MathBox::parens(base_box, mode)
            } else {
                base_box
            };
            return MathBox::hcat(&[base_box, MathBox::text(&sup)]);
        }
    }

    // General case: 2D superscript layout.
    let needs_parens = matches!(
        arena.node(base),
        ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Pow(_, _) | ExprNode::Neg(_)
    );
    let base_box = if needs_parens {
        MathBox::parens(base_box, mode)
    } else {
        base_box
    };

    MathBox::superscript(base_box, exp_box)
}

/// Convert a BigInt to Unicode superscript string, but ONLY use the
/// EAW Neutral codepoints (⁰, ⁵, ⁶, ⁷, ⁸, ⁹). For digits 1-4
/// (which are EAW Ambiguous and break in CJK terminals), return None
/// to force the 2D superscript layout.
fn to_unicode_superscript_safe(n: &BigInt) -> Option<String> {
    let s = n.to_string();
    let mut result = String::new();

    for ch in s.chars() {
        match ch {
            '-' => result.push('⁻'),
            '0' => result.push('⁰'),
            // '1' => '¹'  — EAW Ambiguous, skip
            // '2' => '²'  — EAW Ambiguous, skip
            // '3' => '³'  — EAW Ambiguous, skip
            // '4' => '⁴'  — EAW Ambiguous, skip
            '5' => result.push('⁵'),
            '6' => result.push('⁶'),
            '7' => result.push('⁷'),
            '8' => result.push('⁸'),
            '9' => result.push('⁹'),
            _ => return None, // Contains 1-4 or non-digit — use 2D layout
        }
    }

    Some(result)
}

// ═══════════════════════════════════════════════════════════════════════════
// Function rendering
// ═══════════════════════════════════════════════════════════════════════════

fn pretty_func(name: &str, arena: &Arena, inner: ExprId, mode: RenderMode, depth: u8) -> MathBox {
    let arg_box = pretty_node(arena, inner, mode, depth);
    if arg_box.height() <= 1 {
        // Single-line argument: func(arg)
        MathBox::hcat(&[
            MathBox::text(name),
            MathBox::text("("),
            arg_box,
            MathBox::text(")"),
        ])
    } else {
        // Multi-line argument: func followed by height-matched parens
        let paren_box = MathBox::parens(arg_box, mode);
        MathBox::hcat(&[MathBox::text(name), paren_box])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DisplayExpr helper — used for fallback rendering
// ═══════════════════════════════════════════════════════════════════════════

// We need a way to get the flat display string for fallback cases.
// This is done via the Display impl which is already on ExprId via
// the display module. We use a newtype wrapper.

/// Newtype for Display-formatting an ExprId given an arena reference.
pub(crate) struct DisplayExpr<'a>(pub &'a Arena, pub ExprId);

impl<'a> std::fmt::Display for DisplayExpr<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::output::display::fmt_expr(self.0, f, self.1, 0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::arena::Arena;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }

    fn pp(arena: &Arena, id: ExprId) -> String {
        pretty_print(arena, id, RenderMode::Unicode).render()
    }

    fn pp_ascii(arena: &Arena, id: ExprId) -> String {
        pretty_print(arena, id, RenderMode::Ascii).render()
    }

    // ── Atoms ──────────────────────────────────────────────────────

    #[test]
    fn pretty_integer() {
        let mut a = Arena::new();
        let n = a.int(42);
        assert_eq!(pp(&a, n), "42");
    }

    #[test]
    fn pretty_symbol() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        assert_eq!(pp(&a, x), "x");
    }

    #[test]
    fn pretty_pi() {
        let a = Arena::new();
        assert_eq!(pp(&a, a.pi), "π");
        assert_eq!(pp_ascii(&a, a.pi), "pi");
    }

    // ── Fractions ──────────────────────────────────────────────────

    #[test]
    fn pretty_fraction_half() {
        let mut a = Arena::new();
        let half = a.rational(1, 2);
        let s = pp(&a, half);
        // Should be a stacked fraction:
        //  1
        // ━━━
        //  2
        assert!(
            s.contains('━'),
            "should use heavy bar for outer fraction: {s}"
        );
        assert!(s.lines().count() == 3, "fraction should be 3 lines: {s}");
    }

    #[test]
    fn pretty_fraction_ascii() {
        let mut a = Arena::new();
        let half = a.rational(1, 2);
        let s = pp_ascii(&a, half);
        assert!(s.contains('-'), "ASCII fraction should use dashes: {s}");
        assert!(s.lines().count() == 3, "fraction should be 3 lines: {s}");
    }

    // ── Powers ─────────────────────────────────────────────────────

    #[test]
    fn pretty_power_safe_superscript() {
        // x^5 should use Unicode superscript ⁵ (EAW Neutral, safe)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let x5 = a.pow(x, five);
        let s = pp(&a, x5);
        assert!(s.contains('⁵'), "x^5 should use superscript ⁵: {s}");
    }

    #[test]
    fn pretty_power_unsafe_uses_2d() {
        // x^2 should NOT use Unicode superscript ² (EAW Ambiguous)
        // Instead should use 2D layout
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let s = pp(&a, x2);
        assert!(
            s.lines().count() == 2,
            "x^2 should use 2D superscript (2 lines), got: {s}"
        );
    }

    #[test]
    fn pretty_sqrt() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let half = a.rational(1, 2);
        let sqrt_x = a.pow(x, half);
        let s = pp(&a, sqrt_x);
        assert!(s.contains('√'), "sqrt should contain √: {s}");
    }

    // ── Addition ───────────────────────────────────────────────────

    #[test]
    fn pretty_simple_sum() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let sum = a.add(&[x, y]);
        let s = pp(&a, sum);
        assert!(s.contains('+'), "sum should contain +: {s}");
    }

    #[test]
    fn pretty_sum_with_fraction() {
        // x + 1/2 — the x should align with the fraction bar
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let half = a.rational(1, 2);
        let sum = a.add(&[x, half]);
        let s = pp(&a, sum);
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 3, "sum with fraction should be 3 lines: {s}");
        // The middle line (baseline) should contain both x and the fraction bar.
        assert!(
            lines[1].contains('x') || lines[1].contains('+'),
            "baseline should have x or +: {s}"
        );
    }

    // ── Multiplication ─────────────────────────────────────────────

    #[test]
    fn pretty_product() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let prod = a.mul(&[x, y]);
        let s = pp(&a, prod);
        assert!(
            s.contains('·') || s.contains('*'),
            "product should contain multiplication: {s}"
        );
    }

    // ── Functions ──────────────────────────────────────────────────

    #[test]
    fn pretty_sin() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        assert_eq!(pp(&a, sin_x), "sin(x)");
    }

    #[test]
    fn pretty_gamma() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let gx = a.gamma(x);
        assert_eq!(pp(&a, gx), "Γ(x)");
        assert_eq!(pp_ascii(&a, gx), "Gamma(x)");
    }

    #[test]
    fn pretty_lambertw() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let wx = a.lambertw(x);
        assert_eq!(pp(&a, wx), "W(x)");
    }

    // ── Abs ────────────────────────────────────────────────────────

    #[test]
    fn pretty_abs() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let abs_x = a.abs(x);
        let s = pp(&a, abs_x);
        assert!(s.contains('│'), "abs should contain │: {s}");
    }

    // ── Neg ────────────────────────────────────────────────────────

    #[test]
    fn pretty_neg() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let neg_x = a.neg(x);
        let s = pp(&a, neg_x);
        assert!(s.starts_with('-'), "neg should start with -: {s}");
    }

    // ── Floor / Ceiling ────────────────────────────────────────────

    #[test]
    fn pretty_floor() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let fx = a.floor(x);
        let s = pp(&a, fx);
        assert!(s.contains('⌊') && s.contains('⌋'), "floor: {s}");
    }

    #[test]
    fn pretty_ceiling() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let cx = a.ceiling(x);
        let s = pp(&a, cx);
        assert!(s.contains('⌈') && s.contains('⌉'), "ceiling: {s}");
    }

    // ── Compound expressions ───────────────────────────────────────

    #[test]
    fn pretty_fraction_expression() {
        // (x^2 + 1) / (x - 1) — should render as stacked fraction
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let one = a.one;
        let numer = a.add(&[x2, one]);
        let x_minus_1 = a.sub(x, one);
        let neg_one = a.neg_one;
        let x_m1_pow = a.pow(x_minus_1, neg_one);
        let expr = a.mul(&[numer, x_m1_pow]);
        let s = pp(&a, expr);
        // Should be multi-line with a fraction bar.
        assert!(
            s.lines().count() >= 3,
            "fraction expression should be multi-line: {s}"
        );
    }

    #[test]
    fn pretty_factorial() {
        let mut a = Arena::new();
        let n = sym(&mut a, "n");
        let nf = a.factorial(n);
        assert_eq!(pp(&a, nf), "n!");
    }
}
