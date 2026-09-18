//! Lean 4 / Mathlib rendering of expressions (`Ex::to_lean`).
//!
//! The output is a term of type `ℝ` (or `Prop` for a [`BoolEx`](crate::api::expr::BoolEx)) in
//! Mathlib's surface syntax and spacing conventions, ready to paste into a
//! `theorem` statement or a `have`: binary operators are surrounded by
//! single spaces (`2 * J + 1`), powers use `^` with a natural-number
//! exponent (`J ^ 2`), rational literals are ascribed (`(3 / 31 : ℝ)`) so
//! that they never elaborate as natural-number division, negative integer
//! powers become `⁻¹` / `1 / …`, and function applications use the
//! `Real.` namespace (`Real.sin (2 * x)`, `Real.sqrt x`, `Real.exp 1`,
//! `Real.pi`).
//!
//! Rendering is total for the rational-function / elementary-function
//! fragment; nodes without a standard Mathlib spelling (Bessel functions,
//! unevaluated integrals, sets, …) yield
//! [`SymplexError::NotImplemented`] rather than a guess.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};
use rustc_hash::FxHashMap;

use crate::api::expr::{Expr, Sort};
use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;
use crate::output::common::display_sort_key;

/// Options for [`Ex::to_lean`](crate::api::expr::Ex::to_lean).
#[derive(Clone, Debug)]
pub struct LeanOpts {
    /// The carrier type used in numeral ascriptions and casts (default `ℝ`).
    pub real_type: String,
    /// Ascribe every integer literal (`(2 : ℝ) * J`) instead of only
    /// rationals and bare top-level numbers.  Useful when the surrounding
    /// context cannot infer the type.
    pub ascribe_integers: bool,
}

impl Default for LeanOpts {
    fn default() -> Self {
        LeanOpts {
            real_type: "ℝ".to_string(),
            ascribe_integers: false,
        }
    }
}

// ── Precedence levels (Lean 4 core) ──────────────────────────────────────

/// `+`, `-` (infixl:65)
const PREC_ADD: u8 = 65;
/// `*`, `/` (infixl:70)
const PREC_MUL: u8 = 70;
/// `^` (infixr:75) and prefix `-` (max prec 75 for the argument)
const PREC_POW: u8 = 75;
/// Function application and atoms.
const PREC_MAX: u8 = 100;
/// Relations `<`, `≤`, `=`, `≠` (50).
const PREC_REL: u8 = 50;
/// `∧` (35), `∨` (30), `¬` (max).
const PREC_AND: u8 = 35;
const PREC_OR: u8 = 30;

/// A rendered sub-term with the precedence of its outermost operator.
#[derive(Clone)]
struct Rendered {
    text: String,
    prec: u8,
}

impl Rendered {
    fn atom(text: impl Into<String>) -> Self {
        Rendered {
            text: text.into(),
            prec: PREC_MAX,
        }
    }

    /// The text, parenthesised if this term binds looser than `min_prec`.
    fn at(&self, min_prec: u8) -> String {
        if self.prec < min_prec {
            format!("({})", self.text)
        } else {
            self.text.clone()
        }
    }
}

fn unsupported(what: impl std::fmt::Display) -> SymplexError {
    SymplexError::NotImplemented(format!("to_lean: no Mathlib rendering for {what}"))
}

/// A Lean identifier: plain when it is one already, otherwise `«…»`-quoted.
pub(crate) fn lean_ident(name: &str) -> String {
    let mut chars = name.chars();
    let ok_start = chars
        .next()
        .is_some_and(|c| c.is_alphabetic() || c == '_' || is_greek(c));
    let ok_rest = name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '\'' || is_greek(c) || is_subscript(c));
    if ok_start && ok_rest && !is_lean_keyword(name) {
        name.to_string()
    } else {
        format!("«{name}»")
    }
}

fn is_greek(c: char) -> bool {
    ('α'..='ω').contains(&c) || ('Α'..='Ω').contains(&c)
}

fn is_subscript(c: char) -> bool {
    ('₀'..='₉').contains(&c)
}

fn is_lean_keyword(s: &str) -> bool {
    matches!(
        s,
        "fun"
            | "let"
            | "have"
            | "show"
            | "from"
            | "if"
            | "then"
            | "else"
            | "match"
            | "with"
            | "do"
            | "at"
            | "by"
            | "in"
            | "def"
            | "theorem"
            | "lemma"
            | "example"
            | "open"
            | "end"
            | "section"
            | "namespace"
            | "variable"
            | "instance"
            | "class"
            | "structure"
            | "inductive"
            | "where"
            | "import"
            | "set_option"
            | "sorry"
            | "calc"
            | "exact"
            | "intro"
            | "apply"
            | "cases"
            | "rcases"
            | "obtain"
            | "simp"
            | "ring"
            | "linarith"
            | "nlinarith"
            | "positivity"
            | "norm_num"
            | "field_simp"
    )
}

/// Integer literal as an operand: negative numbers are parenthesised.
fn int_literal(n: &BigInt, opts: &LeanOpts, ascribe: bool) -> Rendered {
    if ascribe || opts.ascribe_integers {
        Rendered::atom(format!("({n} : {})", opts.real_type))
    } else if n.is_negative() {
        Rendered {
            text: n.to_string(),
            prec: PREC_POW,
        }
    } else {
        Rendered::atom(n.to_string())
    }
}

/// Rational literal `(p / q : ℝ)` (always ascribed: bare `p / q` would be
/// natural-number division in Lean).
fn ratio_literal(r: &Ratio<BigInt>, opts: &LeanOpts, ascribe: bool) -> Rendered {
    if r.is_integer() {
        int_literal(r.numer(), opts, ascribe)
    } else {
        Rendered::atom(format!(
            "({} / {} : {})",
            r.numer(),
            r.denom(),
            opts.real_type
        ))
    }
}

/// `f x` or `f (x + 1)`.
fn app(name: &str, args: &[&Rendered]) -> Rendered {
    let mut text = name.to_string();
    for a in args {
        text.push(' ');
        text.push_str(&a.at(PREC_MAX));
    }
    Rendered {
        text,
        prec: PREC_MAX - 1,
    }
}

/// Render `expr` (iterative post-order walk; children before parents).
pub(crate) fn render(arena: &Arena, expr: ExprId, opts: &LeanOpts) -> Result<String, SymplexError> {
    let order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, Rendered> = FxHashMap::default();

    for &id in &order {
        let node = arena.node(id);
        let child = |c: &ExprId| -> Result<Rendered, SymplexError> {
            cache
                .get(c)
                .cloned()
                .ok_or_else(|| unsupported("an unrendered sub-expression"))
        };
        let rendered = match node {
            // ── Atoms ──────────────────────────────────────────────────
            ExprNode::Num(nid) => ratio_literal(arena.num(*nid), opts, id == expr),
            ExprNode::Symbol(sid) => Rendered::atom(lean_ident(arena.symbol_name(*sid))),
            ExprNode::Pi => Rendered::atom("Real.pi"),
            ExprNode::E => Rendered {
                text: "Real.exp 1".into(),
                prec: PREC_MAX - 1,
            },
            ExprNode::EulerGamma => Rendered::atom("Real.eulerMascheroniConstant"),
            ExprNode::GoldenRatio => Rendered::atom("goldenRatio"),
            ExprNode::ImaginaryUnit => Rendered::atom("Complex.I"),
            ExprNode::BoolTrue => Rendered::atom("True"),
            ExprNode::BoolFalse => Rendered::atom("False"),

            // ── Arithmetic ─────────────────────────────────────────────
            ExprNode::Add(children) => {
                // Same term order as `Display` (powers descending, then
                // functions, then constants) so `x + 1`, not `1 + x`.
                let mut ordered: Vec<ExprId> = children.to_vec();
                ordered.sort_by_key(|a| display_sort_key(arena, *a));
                let mut text = String::new();
                for (i, c) in ordered.iter().enumerate() {
                    let (negative, body) = negated_view(arena, *c, opts, &cache)?;
                    if i == 0 {
                        if negative {
                            text.push('-');
                            text.push_str(&body.at(PREC_POW));
                        } else {
                            text.push_str(&body.at(PREC_ADD));
                        }
                    } else {
                        text.push_str(if negative { " - " } else { " + " });
                        // Right operands of a left-assoc `+`/`-` need parens
                        // when they are sums themselves.
                        text.push_str(&body.at(PREC_ADD + 1));
                    }
                }
                Rendered {
                    text,
                    prec: PREC_ADD,
                }
            }
            ExprNode::Mul(children) => render_mul(arena, children, opts, &cache)?,
            ExprNode::Neg(inner) => {
                let r = child(inner)?;
                Rendered {
                    text: format!("-{}", r.at(PREC_POW)),
                    prec: PREC_POW,
                }
            }
            ExprNode::Pow(base, exp) => render_pow(arena, *base, *exp, opts, &cache)?,

            // ── Elementary functions ───────────────────────────────────
            ExprNode::Sin(a) => app("Real.sin", &[&child(a)?]),
            ExprNode::Cos(a) => app("Real.cos", &[&child(a)?]),
            ExprNode::Tan(a) => app("Real.tan", &[&child(a)?]),
            ExprNode::Exp(a) => app("Real.exp", &[&child(a)?]),
            ExprNode::Ln(a) => app("Real.log", &[&child(a)?]),
            ExprNode::Asin(a) => app("Real.arcsin", &[&child(a)?]),
            ExprNode::Acos(a) => app("Real.arccos", &[&child(a)?]),
            ExprNode::Atan(a) => app("Real.arctan", &[&child(a)?]),
            ExprNode::Sinh(a) => app("Real.sinh", &[&child(a)?]),
            ExprNode::Cosh(a) => app("Real.cosh", &[&child(a)?]),
            ExprNode::Tanh(a) => app("Real.tanh", &[&child(a)?]),
            ExprNode::Asinh(a) => app("Real.arsinh", &[&child(a)?]),
            ExprNode::Gamma(a) => app("Real.Gamma", &[&child(a)?]),
            ExprNode::Abs(a) => Rendered::atom(format!("|{}|", child(a)?.text)),
            ExprNode::Floor(a) => {
                Rendered::atom(format!("(⌊{}⌋ : {})", child(a)?.text, opts.real_type))
            }
            ExprNode::Ceiling(a) => {
                Rendered::atom(format!("(⌈{}⌉ : {})", child(a)?.text, opts.real_type))
            }
            ExprNode::Min(children) => fold_binary("min", children, &cache)?,
            ExprNode::Max(children) => fold_binary("max", children, &cache)?,

            // ── Relations and logic (Prop) ─────────────────────────────
            ExprNode::Gt(a, b) => relation(&child(b)?, "<", &child(a)?),
            ExprNode::Ge(a, b) => relation(&child(b)?, "≤", &child(a)?),
            ExprNode::Eq_(a, b) => relation(&child(a)?, "=", &child(b)?),
            ExprNode::Ne(a, b) => relation(&child(a)?, "≠", &child(b)?),
            ExprNode::And(children) => connective(" ∧ ", PREC_AND, children, &cache)?,
            ExprNode::Or(children) => connective(" ∨ ", PREC_OR, children, &cache)?,
            ExprNode::Not(a) => Rendered {
                text: format!("¬{}", child(a)?.at(PREC_MAX)),
                prec: PREC_MAX - 1,
            },
            ExprNode::Piecewise(pairs) => render_piecewise(pairs, &cache)?,

            other => return Err(unsupported(format!("`{}`", describe(other)))),
        };
        cache.insert(id, rendered);
    }

    cache
        .get(&expr)
        .map(|r| r.text.clone())
        .ok_or_else(|| unsupported("the expression"))
}

/// Short description of a node kind for error messages.
fn describe(node: &ExprNode) -> String {
    let dbg = format!("{node:?}");
    dbg.split(['(', ' ']).next().unwrap_or("node").to_string()
}

/// `(true, |t|)` if the summand `t` is a negative literal or a product with
/// a negative coefficient (so the sum can print `a - b`), else `(false, t)`.
fn negated_view(
    arena: &Arena,
    id: ExprId,
    opts: &LeanOpts,
    cache: &FxHashMap<ExprId, Rendered>,
) -> Result<(bool, Rendered), SymplexError> {
    match arena.node(id) {
        ExprNode::Num(nid) if arena.num(*nid).is_negative() => {
            let r = -arena.num(*nid).clone();
            Ok((true, ratio_literal(&r, opts, false)))
        }
        ExprNode::Neg(inner) => Ok((
            true,
            cache
                .get(inner)
                .cloned()
                .ok_or_else(|| unsupported("a sub-expression"))?,
        )),
        ExprNode::Mul(children) => {
            if let Some(first) = children.first()
                && let ExprNode::Num(nid) = arena.node(*first)
                && arena.num(*nid).is_negative()
            {
                let positive = -arena.num(*nid).clone();
                let body =
                    render_mul_with_coeff(arena, Some(&positive), &children[1..], opts, cache)?;
                return Ok((true, body));
            }
            Ok((false, render_mul(arena, children, opts, cache)?))
        }
        _ => Ok((
            false,
            cache
                .get(&id)
                .cloned()
                .ok_or_else(|| unsupported("a sub-expression"))?,
        )),
    }
}

fn render_mul(
    arena: &Arena,
    children: &[ExprId],
    opts: &LeanOpts,
    cache: &FxHashMap<ExprId, Rendered>,
) -> Result<Rendered, SymplexError> {
    if let Some(first) = children.first()
        && let ExprNode::Num(nid) = arena.node(*first)
    {
        let coeff = arena.num(*nid).clone();
        if coeff.is_negative() {
            let body = render_mul_with_coeff(arena, Some(&(-coeff)), &children[1..], opts, cache)?;
            return Ok(Rendered {
                text: format!("-{}", body.at(PREC_POW)),
                prec: PREC_POW,
            });
        }
        return render_mul_with_coeff(arena, Some(&coeff), &children[1..], opts, cache);
    }
    render_mul_with_coeff(arena, None, children, opts, cache)
}

/// `coeff * factors` as `numerator / denominator`, where a rational
/// coefficient `p/q` contributes `p` above and `q` below and `x⁻ⁿ`
/// factors go below.
fn render_mul_with_coeff(
    arena: &Arena,
    coeff: Option<&Ratio<BigInt>>,
    factors: &[ExprId],
    opts: &LeanOpts,
    cache: &FxHashMap<ExprId, Rendered>,
) -> Result<Rendered, SymplexError> {
    let mut numer: Vec<Rendered> = Vec::new();
    let mut denom: Vec<String> = Vec::new();
    if let Some(c) = coeff {
        if !c.numer().is_one() {
            numer.push(int_literal(c.numer(), opts, false));
        }
        if !c.denom().is_one() {
            denom.push(c.denom().to_string());
        }
    }
    for &f in factors {
        if let ExprNode::Pow(base, exp) = arena.node(f)
            && let ExprNode::Num(nid) = arena.node(*exp)
            && arena.num(*nid).is_integer()
            && arena.num(*nid).is_negative()
        {
            let b = cache
                .get(base)
                .ok_or_else(|| unsupported("a sub-expression"))?;
            let n = -arena.num(*nid).to_integer();
            if n.is_one() {
                denom.push(b.at(PREC_MUL + 1));
            } else {
                denom.push(format!("{} ^ {n}", b.at(PREC_POW + 1)));
            }
            continue;
        }
        let r = cache
            .get(&f)
            .ok_or_else(|| unsupported("a sub-expression"))?;
        numer.push(r.clone());
    }
    let numer_rendered = match numer.len() {
        0 => Rendered::atom("1"),
        1 => numer.remove(0),
        _ => Rendered {
            text: numer
                .iter()
                .map(|r| r.at(PREC_MUL))
                .collect::<Vec<_>>()
                .join(" * "),
            prec: PREC_MUL,
        },
    };
    if denom.is_empty() {
        return Ok(numer_rendered);
    }
    let denom_text = if denom.len() == 1 {
        denom.remove(0)
    } else {
        format!("({})", denom.join(" * "))
    };
    Ok(Rendered {
        text: format!("{} / {denom_text}", numer_rendered.at(PREC_MUL)),
        prec: PREC_MUL,
    })
}

fn render_pow(
    arena: &Arena,
    base: ExprId,
    exp: ExprId,
    opts: &LeanOpts,
    cache: &FxHashMap<ExprId, Rendered>,
) -> Result<Rendered, SymplexError> {
    let b = cache
        .get(&base)
        .ok_or_else(|| unsupported("a sub-expression"))?;
    if let ExprNode::Num(nid) = arena.node(exp) {
        let k = arena.num(*nid).clone();
        if k.is_integer() {
            let n = k.to_integer();
            if n.is_zero() {
                return Ok(Rendered::atom("1"));
            }
            if n.is_negative() {
                let m = -n;
                let text = if m.is_one() {
                    format!("{}⁻¹", b.at(PREC_MAX))
                } else {
                    format!("({} ^ {m})⁻¹", b.at(PREC_POW + 1))
                };
                return Ok(Rendered::atom(text));
            }
            return Ok(Rendered {
                text: format!("{} ^ {n}", b.at(PREC_POW + 1)),
                prec: PREC_POW,
            });
        }
        // Rational exponent: `Real.sqrt` for 1/2, otherwise a real power.
        if k == Ratio::new(BigInt::one(), BigInt::from(2)) {
            return Ok(app("Real.sqrt", &[b]));
        }
        let e = ratio_literal(&k, opts, false);
        return Ok(Rendered {
            text: format!("{} ^ {}", b.at(PREC_POW + 1), e.at(PREC_MAX)),
            prec: PREC_POW,
        });
    }
    // Symbolic exponent: real power `x ^ y` (Real.rpow).
    let e = cache
        .get(&exp)
        .ok_or_else(|| unsupported("a sub-expression"))?;
    Ok(Rendered {
        text: format!("{} ^ {}", b.at(PREC_POW + 1), e.at(PREC_POW)),
        prec: PREC_POW,
    })
}

/// `min a (min b c)` for n-ary min/max.
fn fold_binary(
    name: &str,
    children: &[ExprId],
    cache: &FxHashMap<ExprId, Rendered>,
) -> Result<Rendered, SymplexError> {
    let rendered: Vec<Rendered> = children
        .iter()
        .map(|c| {
            cache
                .get(c)
                .cloned()
                .ok_or_else(|| unsupported("a sub-expression"))
        })
        .collect::<Result<_, _>>()?;
    let Some((last, init)) = rendered.split_last() else {
        return Err(unsupported(format!("an empty `{name}`")));
    };
    let mut acc = last.clone();
    for r in init.iter().rev() {
        acc = app(name, &[r, &acc]);
    }
    Ok(acc)
}

fn relation(lhs: &Rendered, op: &str, rhs: &Rendered) -> Rendered {
    Rendered {
        text: format!("{} {op} {}", lhs.at(PREC_REL + 1), rhs.at(PREC_REL + 1)),
        prec: PREC_REL,
    }
}

fn connective(
    sep: &str,
    prec: u8,
    children: &[ExprId],
    cache: &FxHashMap<ExprId, Rendered>,
) -> Result<Rendered, SymplexError> {
    let parts: Vec<String> = children
        .iter()
        .map(|c| {
            cache
                .get(c)
                .map(|r| r.at(prec + 1))
                .ok_or_else(|| unsupported("a sub-expression"))
        })
        .collect::<Result<_, _>>()?;
    Ok(Rendered {
        text: parts.join(sep),
        prec,
    })
}

/// `if c₁ then v₁ else if c₂ then v₂ else v₃`; a final `True` condition
/// is the `else` branch.
fn render_piecewise(
    pairs: &[(ExprId, ExprId)],
    cache: &FxHashMap<ExprId, Rendered>,
) -> Result<Rendered, SymplexError> {
    let get = |id: &ExprId| {
        cache
            .get(id)
            .cloned()
            .ok_or_else(|| unsupported("a sub-expression"))
    };
    let mut text = String::new();
    let n = pairs.len();
    for (i, (value, cond)) in pairs.iter().enumerate() {
        let v = get(value)?;
        let c = get(cond)?;
        if i + 1 == n && c.text == "True" {
            text.push_str(&v.text);
            return Ok(Rendered { text, prec: 0 });
        }
        text.push_str(&format!("if {} then {} else ", c.text, v.text));
    }
    Err(unsupported(
        "a `Piecewise` without a catch-all final branch (Lean's `if` needs an `else`)",
    ))
}

// ═══════════════════════════════════════════════════════════════════════════
// Line wrapping
// ═══════════════════════════════════════════════════════════════════════════

/// Mathlib's maximum line length (`linter.style.longLine`).
pub const MATHLIB_LINE_WIDTH: usize = 100;

/// Re-flow Lean source so that no line exceeds `width` columns, breaking
/// only at spaces, preferring a space right after a comma (so a
/// `nlinarith [a, b, c]` hint list breaks between hints) and otherwise the
/// shallowest bracket depth (so a `theorem` signature breaks between
/// binders).  Continuation lines are indented four
/// columns past a `theorem`/`lemma`/`example` line and two past anything
/// else, which is more than the enclosing block in every case Lean's
/// indentation-sensitive parser cares about.  Lines that cannot be split
/// (a single token longer than `width`) are left as they are.
///
/// The certificate emitters ([`Certificate::to_lean`](crate::certificates::Certificate::to_lean)
/// and friends) apply this at [`MATHLIB_LINE_WIDTH`]; [`Ex::to_lean`](crate::api::expr::Ex::to_lean)
/// returns a single line so it can be embedded anywhere, and callers wrap
/// the finished statement with this function.
///
/// ```
/// use symplex::lean::wrap_lean;
///
/// let long = "theorem t (x y : ℝ) (h_x_lo : (0 : ℝ) ≤ x) (h_x_hi : x ≤ (1 : ℝ)) (h_y_lo : (0 : ℝ) ≤ y) :\n    0 ≤ x := by\n  linarith\n";
/// let wrapped = wrap_lean(long, 60);
/// assert!(wrapped.lines().all(|l| l.chars().count() <= 60));
/// assert!(wrapped.starts_with("theorem t (x y : ℝ) (h_x_lo : (0 : ℝ) ≤ x)\n    (h_x_hi : x ≤ (1 : ℝ))"), "{wrapped}");
/// ```
pub fn wrap_lean(text: &str, width: usize) -> String {
    let mut out = String::with_capacity(text.len() + 64);
    for line in text.split_inclusive('\n') {
        let (body, newline) = match line.strip_suffix('\n') {
            Some(b) => (b, "\n"),
            None => (line, ""),
        };
        wrap_line(body, width, &mut out);
        out.push_str(newline);
    }
    out
}

fn wrap_line(line: &str, width: usize, out: &mut String) {
    let indent = line.chars().take_while(|c| *c == ' ').count();
    let trimmed = &line[indent..];
    let is_decl = trimmed.starts_with("theorem ")
        || trimmed.starts_with("lemma ")
        || trimmed.starts_with("example ");
    let cont_indent = indent + if is_decl { 4 } else { 2 };
    let mut current: Vec<char> = line.chars().collect();
    let mut current_indent = indent;
    loop {
        if current.len() <= width {
            out.extend(current.iter());
            return;
        }
        // Candidate break positions: spaces (not in the indent), scored by
        // bracket depth, then "after a comma", then latest.
        let mut depth: i32 = 0;
        let mut best: Option<(u8, i32, usize)> = None;
        let mut in_guillemets = false;
        for (i, &c) in current.iter().enumerate() {
            match c {
                '«' => in_guillemets = true,
                '»' => in_guillemets = false,
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                ' ' if i > current_indent && !in_guillemets => {
                    if i > width {
                        break;
                    }
                    // Keep `nlinarith [` / `linarith [` together.
                    if current.get(i + 1) == Some(&'[') {
                        continue;
                    }
                    // Prefer breaking right after a comma (hint lists), then
                    // at the shallowest bracket depth (between binders), then
                    // as late as possible.
                    let after_comma = i > 0 && current[i - 1] == ',';
                    let key = (u8::from(!after_comma), depth, usize::MAX - i);
                    if best.is_none_or(|b| key < b) {
                        best = Some(key);
                    }
                }
                _ => {}
            }
        }
        let Some((_, _, inv_pos)) = best else {
            // No usable break point: emit as is.
            out.extend(current.iter());
            return;
        };
        let pos = usize::MAX - inv_pos;
        let head: String = current[..pos].iter().collect();
        out.push_str(head.trim_end());
        out.push('\n');
        let rest: String = current[pos + 1..].iter().collect();
        let mut next: Vec<char> = " ".repeat(cont_indent).chars().collect();
        next.extend(rest.trim_start().chars());
        current = next;
        current_indent = cont_indent;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

impl<S: Sort> Expr<S> {
    /// Render this expression as a Lean 4 / Mathlib term (type `ℝ`, or
    /// `Prop` for a [`BoolEx`](crate::api::expr::BoolEx)) with Mathlib's
    /// spacing conventions.
    ///
    /// See the [`lean`](crate::lean) module docs for the exact conventions.
    ///
    /// # Errors
    ///
    /// [`SymplexError::NotImplemented`] for nodes without a standard Mathlib
    /// spelling (special functions beyond the elementary ones, unevaluated
    /// integrals/limits, sets, …).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (j, x) = (ctx.symbol("j"), ctx.symbol("x"));
    /// assert_eq!((2 * &j + 1).to_lean().unwrap(), "2 * j + 1");
    /// assert_eq!((&j.powi(2) - &j / 2).to_lean().unwrap(), "j ^ 2 - j / 2");
    /// assert_eq!(((&j - 1) / (2 * &j)).to_lean().unwrap(), "(j - 1) / (2 * j)");
    /// assert_eq!(ctx.rational(3, 31).to_lean().unwrap(), "(3 / 31 : ℝ)");
    /// assert_eq!((&x.sin().powi(2) + &x.exp()).to_lean().unwrap(), "Real.sin x ^ 2 + Real.exp x");
    /// assert_eq!((&x * 2).sqrt().to_lean().unwrap(), "Real.sqrt (2 * x)");
    /// assert_eq!(x.gt(&ctx.int(0)).to_lean().unwrap(), "0 < x");
    /// assert!(x.bessel_j(&ctx.int(0)).to_lean().is_err());
    /// ```
    pub fn to_lean(&self) -> Result<String, SymplexError> {
        self.to_lean_with(&LeanOpts::default())
    }

    /// [`to_lean`](Self::to_lean) with explicit [`LeanOpts`].
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::lean::LeanOpts;
    ///
    /// let ctx = Context::new();
    /// let j = ctx.symbol("j");
    /// let opts = LeanOpts { ascribe_integers: true, ..LeanOpts::default() };
    /// assert_eq!((2 * &j + 1).to_lean_with(&opts).unwrap(), "(2 : ℝ) * j + (1 : ℝ)");
    /// ```
    pub fn to_lean_with(&self, opts: &LeanOpts) -> Result<String, SymplexError> {
        let inner = self.inner.read();
        render(&inner.arena, self.raw_id(), opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    fn lean(e: &Ex) -> String {
        e.to_lean().unwrap()
    }

    #[test]
    fn literals() {
        let ctx = Context::new();
        assert_eq!(lean(&ctx.int(3)), "(3 : ℝ)");
        assert_eq!(lean(&ctx.int(-3)), "(-3 : ℝ)");
        assert_eq!(lean(&ctx.rational(-3, 4)), "(-3 / 4 : ℝ)");
        assert_eq!(lean(&ctx.pi()), "Real.pi");
        assert_eq!(lean(&ctx.e()), "Real.exp 1");
    }

    #[test]
    fn sums_products_powers() {
        let ctx = Context::new();
        let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
        assert_eq!(lean(&(&x + &y)), "x + y");
        assert_eq!(lean(&(&x - &y)), "x - y");
        assert_eq!(lean(&(-&x + 1)), "-x + 1");
        assert_eq!(lean(&(&x * &y * 3)), "3 * x * y");
        assert_eq!(lean(&(-&x * &y)), "-(x * y)");
        assert_eq!(lean(&(&x * (&y + 1))), "x * (y + 1)");
        assert_eq!(lean(&(&x + 1).powi(3)), "(x + 1) ^ 3");
        assert_eq!(lean(&x.powi(-1)), "x⁻¹");
        assert_eq!(lean(&x.powi(-2)), "(x ^ 2)⁻¹");
        assert_eq!(lean(&(&x / &y)), "x / y");
        assert_eq!(lean(&(&x / (&y * 2))), "x / (2 * y)");
        assert_eq!(lean(&(ctx.rational(3, 2) * &x * &y)), "3 * x * y / 2");
        assert_eq!(lean(&(&x - ctx.rational(1, 2) * &y)), "x - y / 2");
        assert_eq!(lean(&x.pow(&y)), "x ^ y");
        assert_eq!(lean(&x.pow(&ctx.rational(1, 3))), "x ^ (1 / 3 : ℝ)");
        assert_eq!(lean(&(2 * &x).powi(2)), "4 * x ^ 2"); // canonicalised
        assert_eq!(lean(&(&x + &y).powi(2)), "(x + y) ^ 2");
        assert_eq!(lean(&x.powi(2).powi(3)), "x ^ 6");
    }

    #[test]
    fn functions_and_props() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        assert_eq!(lean(&x.sin()), "Real.sin x");
        assert_eq!(lean(&(&x + 1).ln()), "Real.log (x + 1)");
        assert_eq!(lean(&x.abs()), "|x|");
        assert_eq!(lean(&x.floor()), "(⌊x⌋ : ℝ)");
        assert_eq!(lean(&x.min_with(&ctx.int(1))), "min x 1");
        assert_eq!(x.ge(&ctx.int(1)).to_lean().unwrap(), "1 ≤ x");
        assert_eq!(
            x.gt(&ctx.int(0)).and(&x.lt(&ctx.int(1))).to_lean().unwrap(),
            "0 < x ∧ x < 1"
        );
        assert_eq!(x.eq_expr(&ctx.int(2)).not().to_lean().unwrap(), "¬(x = 2)");
    }

    #[test]
    fn identifiers_are_quoted_when_needed() {
        assert_eq!(lean_ident("j"), "j");
        assert_eq!(lean_ident("x_1"), "x_1");
        assert_eq!(lean_ident("θ"), "θ");
        assert_eq!(lean_ident("a'"), "a'");
        assert_eq!(lean_ident("fun"), "«fun»");
        assert_eq!(lean_ident("x-1"), "«x-1»");
    }

    #[test]
    fn unsupported_nodes_are_errors_not_guesses() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        assert!(matches!(
            x.gamma().digamma().to_lean(),
            Err(SymplexError::NotImplemented(_))
        ));
        // An unevaluated formal derivative has no Mathlib term.
        let y = ctx.symbol("y");
        assert!(matches!(
            y.formal_diff(&x).to_lean(),
            Err(SymplexError::NotImplemented(_))
        ));
    }
}
