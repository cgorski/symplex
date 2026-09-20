//! Presentation MathML rendering of expressions (`Ex::to_mathml`).  (0.9)
//!
//! [`Ex::to_mathml`](crate::api::expr::Ex::to_mathml) renders an expression
//! (or a [`BoolEx`](crate::api::expr::BoolEx)) as a
//! `<math xmlns="http://www.w3.org/1998/Math/MathML">…</math>` element in
//! Presentation MathML — the markup browsers and MathJax render directly
//! (SymPy: `mathml(expr, printer='presentation')`).
//!
//! The layout decisions mirror [`to_latex`](crate::api::expr::Ex::to_latex):
//! sums are printed in display order with subtraction for negated terms,
//! products with negative powers become `<mfrac>`, `x^(1/2)` is `<msqrt>`,
//! `x^(1/n)` is `<mroot>`, `x^(-n)` is a fraction, `sin(x)^2` is
//! `<msup><mi>sin</mi><mn>2</mn></msup>`, and a sum inside a product or a
//! compound base of a power is parenthesised with explicit
//! `<mo>(</mo>`/`<mo>)</mo>` (no `<mfenced>`).  Function application uses
//! `<mi>sin</mi><mo>&#x2061;</mo>` (the invisible `ApplyFunction`
//! operator) and implicit multiplication `<mo>&#x2062;</mo>`
//! (`InvisibleTimes`); Greek symbol names become numeric character
//! references (`alpha` → `<mi>&#x3B1;</mi>`), `x_1` becomes `<msub>`.
//!
//! Every character reference is numeric, so the output is well-formed XML
//! without a DTD.  Nodes with no standard presentation (`Series`,
//! `DSolve`, `RootOf`, `RootSum`) yield
//! [`SymplexError::NotImplemented`] rather than a guess.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed};
use rustc_hash::FxHashMap;

use crate::api::expr::{Expr, Sort};
use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode, INTERVAL_LEFT_OPEN, INTERVAL_RIGHT_OPEN};
use crate::base::walk;
use crate::output::common::{display_sort_key, is_neg_coeff_mul, is_neg_one_mul};

/// The MathML namespace.
pub const MATHML_NS: &str = "http://www.w3.org/1998/Math/MathML";

// ═══════════════════════════════════════════════════════════════════════════
// Character data
// ═══════════════════════════════════════════════════════════════════════════

/// `ApplyFunction` (U+2061).
const APPLY: &str = "<mo>&#x2061;</mo>";
/// `InvisibleTimes` (U+2062).
const TIMES: &str = "<mo>&#x2062;</mo>";
/// A visible `⋅` between two adjacent numbers.
const CDOT: &str = "<mo>&#x22C5;</mo>";

/// Greek letter names (as used in symbol names and by `to_latex`) with
/// their Unicode code points.
const GREEK: &[(&str, u32)] = &[
    ("alpha", 0x3B1),
    ("beta", 0x3B2),
    ("gamma", 0x3B3),
    ("delta", 0x3B4),
    ("epsilon", 0x3F5),
    ("varepsilon", 0x3B5),
    ("zeta", 0x3B6),
    ("eta", 0x3B7),
    ("theta", 0x3B8),
    ("vartheta", 0x3D1),
    ("iota", 0x3B9),
    ("kappa", 0x3BA),
    ("lambda", 0x3BB),
    ("mu", 0x3BC),
    ("nu", 0x3BD),
    ("xi", 0x3BE),
    ("omicron", 0x3BF),
    ("pi", 0x3C0),
    ("rho", 0x3C1),
    ("varrho", 0x3F1),
    ("sigma", 0x3C3),
    ("varsigma", 0x3C2),
    ("tau", 0x3C4),
    ("upsilon", 0x3C5),
    ("phi", 0x3D5),
    ("varphi", 0x3C6),
    ("chi", 0x3C7),
    ("psi", 0x3C8),
    ("omega", 0x3C9),
    ("Alpha", 0x391),
    ("Beta", 0x392),
    ("Gamma", 0x393),
    ("Delta", 0x394),
    ("Epsilon", 0x395),
    ("Zeta", 0x396),
    ("Eta", 0x397),
    ("Theta", 0x398),
    ("Iota", 0x399),
    ("Kappa", 0x39A),
    ("Lambda", 0x39B),
    ("Mu", 0x39C),
    ("Nu", 0x39D),
    ("Xi", 0x39E),
    ("Omicron", 0x39F),
    ("Pi", 0x3A0),
    ("Rho", 0x3A1),
    ("Sigma", 0x3A3),
    ("Tau", 0x3A4),
    ("Upsilon", 0x3A5),
    ("Phi", 0x3A6),
    ("Chi", 0x3A7),
    ("Psi", 0x3A8),
    ("Omega", 0x3A9),
];

/// Escape the three characters that are special in XML character data.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(c),
        }
    }
    out
}

/// `<mi>` around trusted markup (a library name or a character reference).
/// User-supplied names go through [`ident`], which escapes them.
fn mi(text: &str) -> String {
    format!("<mi>{text}</mi>")
}

fn mn(text: &str) -> String {
    format!("<mn>{text}</mn>")
}

fn mo(text: &str) -> String {
    format!("<mo>{text}</mo>")
}

fn mrow(inner: &str) -> String {
    format!("<mrow>{inner}</mrow>")
}

/// `<mrow><mo>(</mo>…<mo>)</mo></mrow>`.
fn paren(inner: &str) -> String {
    format!("<mrow><mo>(</mo>{inner}<mo>)</mo></mrow>")
}

/// A single (user-supplied) identifier: a Greek letter becomes its
/// character reference, anything else is escaped.
fn ident(name: &str) -> String {
    match GREEK.iter().find(|(g, _)| *g == name) {
        Some((_, cp)) => format!("<mi>&#x{cp:X};</mi>"),
        None => mi(&escape(name)),
    }
}

/// Symbol names: `alpha` → `<mi>&#x3B1;</mi>`, `x_1` → `<msub>…</msub>`.
fn symbol(name: &str) -> String {
    if let Some(idx) = name.find('_') {
        let base = &name[..idx];
        let sub = &name[idx + 1..];
        if !base.is_empty() && !sub.is_empty() {
            let sub_xml = if sub.chars().all(|c| c.is_ascii_digit()) {
                mn(sub)
            } else {
                ident(sub)
            };
            return format!("<msub>{}{}</msub>", ident(base), sub_xml);
        }
    }
    ident(name)
}

/// A rational literal.
fn number(r: &Ratio<BigInt>) -> String {
    let magnitude = if r.is_integer() {
        mn(&r.numer().abs().to_string())
    } else {
        format!(
            "<mfrac>{}{}</mfrac>",
            mn(&r.numer().abs().to_string()),
            mn(&r.denom().to_string())
        )
    };
    if r.is_negative() {
        mrow(&format!("{}{magnitude}", mo("-")))
    } else {
        magnitude
    }
}

/// `f⁡(a, b, …)` with the invisible apply operator.
fn apply(head: &str, args: &[&str]) -> String {
    let mut inner = String::new();
    for (i, a) in args.iter().enumerate() {
        if i > 0 {
            inner.push_str(&mo(","));
        }
        inner.push_str(a);
    }
    mrow(&format!("{head}{APPLY}{}", paren(&inner)))
}

// ═══════════════════════════════════════════════════════════════════════════
// Renderer
// ═══════════════════════════════════════════════════════════════════════════

fn unsupported(what: &str) -> SymplexError {
    SymplexError::NotImplemented(format!("to_mathml: no presentation for {what}"))
}

/// Rendered children, looked up by id.
///
/// An entry lives only until its last reader has been rendered (see
/// [`cache_reads`]): keeping every node's markup alive would make the live
/// text O(depth × size) — a chain of 5 000 nested `sin`s held a gigabyte.
type Cache = FxHashMap<ExprId, String>;

fn cached(cache: &Cache, id: ExprId) -> Result<String, SymplexError> {
    cache
        .get(&id)
        .cloned()
        .ok_or_else(|| unsupported("an unrendered sub-expression"))
}

/// The cache entries the renderer of `id` reads, with multiplicity.
///
/// Mostly the direct children, but a few layouts reach *through* a child
/// (and then do not read the child itself): a summand `-t` or `-c·t`
/// contributes `t`'s factors ([`signed_term`]), a factor `b^(-n)` its base
/// `b` ([`render_mul_parts`]), and `sin(x)^2` the trig argument `x`
/// ([`render_pow`]).  Must stay in step with those functions: an entry
/// dropped too early surfaces as an "unrendered sub-expression" error, one
/// kept too long only costs memory — so over-approximate when in doubt.
fn cache_reads(arena: &Arena, id: ExprId, out: &mut Vec<ExprId>) {
    // The reads of `render_mul_parts` over `factors`.
    let mul_parts = |factors: &[ExprId], out: &mut Vec<ExprId>| {
        for &f in factors {
            match negative_power(arena, f) {
                Some((base, _)) => out.push(base),
                None => out.push(f),
            }
        }
    };
    match arena.node(id) {
        ExprNode::Add(children) => {
            for &c in children.iter() {
                match arena.node(c) {
                    ExprNode::Num(_) => out.push(c),
                    ExprNode::Neg(inner) => out.push(*inner),
                    ExprNode::Mul(ch)
                        if (is_neg_one_mul(arena, c) || is_neg_coeff_mul(arena, c))
                            && arena.as_num(ch[0]).is_some() =>
                    {
                        mul_parts(&ch[1..], out);
                    }
                    _ => out.push(c),
                }
            }
        }
        ExprNode::Mul(children) => {
            if let Some(&first) = children.first()
                && arena.as_num(first).is_some()
                && children.len() > 1
            {
                mul_parts(&children[1..], out);
            } else {
                mul_parts(children, out);
            }
        }
        ExprNode::Pow(base, exp) => {
            let root_or_reciprocal = arena.as_num(*exp).is_some_and(|r| {
                (!r.is_integer() && !r.is_negative() && r.numer().is_one())
                    || (r.is_integer() && r.is_negative())
            });
            if root_or_reciprocal {
                out.push(*base);
            } else {
                out.push(*exp);
                match trig_head(arena.node(*base)) {
                    Some((_, arg)) => out.push(arg),
                    None => out.push(*base),
                }
            }
        }
        // `LaplaceTransform`/`InverseLaplaceTransform` read only the body,
        // `PhysicalConstant` nothing; listing every child is a harmless
        // over-approximation.
        other => other.for_each_child(|c| out.push(c)),
    }
}

/// Is `id` a non-negative integer literal (a plain `<mn>`)?
fn is_plain_number(arena: &Arena, id: ExprId) -> bool {
    arena
        .as_num(id)
        .is_some_and(|r| r.is_integer() && !r.is_negative())
}

/// A factor of a product: sums are parenthesised (as in `to_latex`).
fn mul_factor(arena: &Arena, id: ExprId, cache: &Cache) -> Result<String, SymplexError> {
    let xml = cached(cache, id)?;
    Ok(if is_bool_or_add(arena, id) {
        paren(&xml)
    } else {
        xml
    })
}

fn is_bool_or_add(arena: &Arena, id: ExprId) -> bool {
    matches!(
        arena.node(id),
        ExprNode::Add(_)
            | ExprNode::Gt(_, _)
            | ExprNode::Ge(_, _)
            | ExprNode::Eq_(_, _)
            | ExprNode::Ne(_, _)
            | ExprNode::And(_)
            | ExprNode::Or(_)
            | ExprNode::Not(_)
    )
}

/// The base of a power: compound bases and negative or fractional numbers
/// are parenthesised (same rule as `to_latex`).
fn pow_base(arena: &Arena, base: ExprId, cache: &Cache) -> Result<String, SymplexError> {
    let xml = cached(cache, base)?;
    let wrap = match arena.node(base) {
        ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) | ExprNode::Pow(_, _) => true,
        ExprNode::Num(nid) => {
            let r = arena.num(*nid);
            r.is_negative() || !r.is_integer()
        }
        _ => is_bool_or_add(arena, base),
    };
    Ok(if wrap { paren(&xml) } else { xml })
}

/// Join factors with `InvisibleTimes`, or `⋅` between two plain numbers.
fn join_factors(factors: &[(String, bool)]) -> String {
    if factors.len() == 1 {
        return factors[0].0.clone();
    }
    let mut out = String::new();
    for (i, (xml, numeric)) in factors.iter().enumerate() {
        if i > 0 {
            out.push_str(if factors[i - 1].1 && *numeric {
                CDOT
            } else {
                TIMES
            });
        }
        out.push_str(xml);
    }
    mrow(&out)
}

/// `Pow(base, -n)` → `(base, n)` for a negative rational exponent.
fn negative_power(arena: &Arena, id: ExprId) -> Option<(ExprId, Ratio<BigInt>)> {
    if let ExprNode::Pow(base, exp) = arena.node(id)
        && let Some(r) = arena.as_num(*exp)
        && r.is_negative()
    {
        return Some((*base, -r.clone()));
    }
    None
}

/// The exponent of a denominator factor: `<mn>n</mn>` or a fraction.
fn exponent_xml(r: &Ratio<BigInt>) -> String {
    number(r)
}

/// Render `coeff · factors` (a `Mul` node's children, possibly with the
/// leading coefficient replaced).  `coeff == None` means "no leading
/// number"; `Some(1)` is dropped.
fn render_mul_parts(
    arena: &Arena,
    coeff: Option<&Ratio<BigInt>>,
    factors: &[ExprId],
    cache: &Cache,
) -> Result<String, SymplexError> {
    let mut numer: Vec<(String, bool)> = Vec::new();
    let mut denom: Vec<(String, bool)> = Vec::new();
    if let Some(c) = coeff {
        if c.is_integer() {
            if !c.numer().is_one() {
                numer.push((mn(&c.numer().to_string()), true));
            }
        } else {
            if !c.numer().is_one() {
                numer.push((mn(&c.numer().to_string()), true));
            }
            denom.push((mn(&c.denom().to_string()), true));
        }
    }
    for &f in factors {
        if let Some((base, pos_exp)) = negative_power(arena, f) {
            let base_xml = pow_base(arena, base, cache)?;
            if pos_exp.is_one() {
                denom.push((base_xml, is_plain_number(arena, base)));
            } else {
                denom.push((
                    format!("<msup>{base_xml}{}</msup>", exponent_xml(&pos_exp)),
                    false,
                ));
            }
        } else {
            numer.push((mul_factor(arena, f, cache)?, is_plain_number(arena, f)));
        }
    }
    let numer_xml = if numer.is_empty() {
        mn("1")
    } else {
        join_factors(&numer)
    };
    Ok(if denom.is_empty() {
        numer_xml
    } else {
        format!("<mfrac>{numer_xml}{}</mfrac>", join_factors(&denom))
    })
}

/// A `Mul` node: a leading `-1` becomes a prefix minus, a leading rational
/// splits into numerator and denominator.
fn render_mul(arena: &Arena, children: &[ExprId], cache: &Cache) -> Result<String, SymplexError> {
    if children.is_empty() {
        return Ok(mn("1"));
    }
    if let Some(&first) = children.first()
        && let Some(c) = arena.as_num(first)
        && children.len() > 1
    {
        if c.is_negative() {
            let body = render_mul_parts(arena, Some(&(-c.clone())), &children[1..], cache)?;
            return Ok(mrow(&format!("{}{body}", mo("-"))));
        }
        return render_mul_parts(arena, Some(c), &children[1..], cache);
    }
    render_mul_parts(arena, None, children, cache)
}

/// `(negative, |term|)` for a summand: negative literals, `Neg(x)` and
/// products with a negative coefficient print as subtractions.
fn signed_term(arena: &Arena, id: ExprId, cache: &Cache) -> Result<(bool, String), SymplexError> {
    match arena.node(id) {
        ExprNode::Num(nid) if arena.num(*nid).is_negative() => {
            Ok((true, number(&-arena.num(*nid).clone())))
        }
        ExprNode::Neg(inner) => {
            let xml = cached(cache, *inner)?;
            Ok((
                true,
                if matches!(arena.node(*inner), ExprNode::Add(_)) {
                    paren(&xml)
                } else {
                    xml
                },
            ))
        }
        ExprNode::Mul(children) if is_neg_one_mul(arena, id) || is_neg_coeff_mul(arena, id) => {
            let Some(c) = arena.as_num(children[0]) else {
                return Ok((false, cached(cache, id)?));
            };
            let body = render_mul_parts(arena, Some(&(-c.clone())), &children[1..], cache)?;
            Ok((true, body))
        }
        _ => Ok((false, cached(cache, id)?)),
    }
}

/// `sin`, `cos`, … as `<mi>` heads for the trig-power form.
fn trig_head(node: &ExprNode) -> Option<(&'static str, ExprId)> {
    Some(match node {
        ExprNode::Sin(x) => ("sin", *x),
        ExprNode::Cos(x) => ("cos", *x),
        ExprNode::Tan(x) => ("tan", *x),
        ExprNode::Sinh(x) => ("sinh", *x),
        ExprNode::Cosh(x) => ("cosh", *x),
        ExprNode::Tanh(x) => ("tanh", *x),
        ExprNode::Asin(x) => ("arcsin", *x),
        ExprNode::Acos(x) => ("arccos", *x),
        ExprNode::Atan(x) => ("arctan", *x),
        _ => return None,
    })
}

fn render_pow(
    arena: &Arena,
    base: ExprId,
    exp: ExprId,
    cache: &Cache,
) -> Result<String, SymplexError> {
    if let Some(r) = arena.as_num(exp) {
        let one = BigInt::one();
        if !r.is_integer() && !r.is_negative() && *r.numer() == one {
            let base_xml = cached(cache, base)?;
            if *r.denom() == BigInt::from(2) {
                return Ok(format!("<msqrt>{base_xml}</msqrt>"));
            }
            return Ok(format!(
                "<mroot>{base_xml}{}</mroot>",
                mn(&r.denom().to_string())
            ));
        }
        if r.is_integer() && r.is_negative() {
            let base_xml = pow_base(arena, base, cache)?;
            let n = -r.numer();
            let denom = if n.is_one() {
                base_xml
            } else {
                format!("<msup>{base_xml}{}</msup>", mn(&n.to_string()))
            };
            return Ok(format!("<mfrac>{}{denom}</mfrac>", mn("1")));
        }
    }
    let exp_xml = cached(cache, exp)?;
    if let Some((name, arg)) = trig_head(arena.node(base)) {
        let head = format!("<msup>{}{exp_xml}</msup>", mi(name));
        return Ok(apply(&head, &[&cached(cache, arg)?]));
    }
    Ok(format!(
        "<msup>{}{exp_xml}</msup>",
        pow_base(arena, base, cache)?
    ))
}

/// `a op b op c` with the operands of mixed connectives parenthesised.
fn connective(
    arena: &Arena,
    op: &str,
    children: &[ExprId],
    cache: &Cache,
) -> Result<String, SymplexError> {
    let mut out = String::new();
    for (i, &c) in children.iter().enumerate() {
        if i > 0 {
            out.push_str(&format!("<mo>{op}</mo>"));
        }
        let xml = cached(cache, c)?;
        if matches!(arena.node(c), ExprNode::And(_) | ExprNode::Or(_)) {
            out.push_str(&paren(&xml));
        } else {
            out.push_str(&xml);
        }
    }
    Ok(mrow(&out))
}

fn relation(lhs: &str, op: &str, rhs: &str) -> String {
    mrow(&format!("{lhs}<mo>{op}</mo>{rhs}"))
}

/// `∑`/`∏` with `<munderover>`.
fn big_op(op: &str, var: &str, lo: &str, hi: &str, body: &str) -> String {
    mrow(&format!(
        "<munderover><mo>{op}</mo>{}{hi}</munderover>{body}",
        mrow(&format!("{var}<mo>=</mo>{lo}"))
    ))
}

/// Describe a node kind for error messages.
fn describe(node: &ExprNode) -> String {
    let dbg = format!("{node:?}");
    dbg.split(['(', ' ']).next().unwrap_or("node").to_string()
}

/// Render `expr` as MathML markup (no `<math>` wrapper).  Iterative
/// post-order: every child is rendered before its parent, and a child's
/// markup is dropped from the cache once its last reader has consumed it
/// (the arena is a hash-consed DAG, so a node may have several readers).
pub(crate) fn render(arena: &Arena, expr: ExprId) -> Result<String, SymplexError> {
    let order = walk::post_order_ids(arena, expr);
    let mut cache: Cache = FxHashMap::default();

    // Pending reads per node; the root is read once more by the final lookup.
    let mut pending: FxHashMap<ExprId, usize> = FxHashMap::default();
    let mut reads: Vec<ExprId> = Vec::new();
    for &id in &order {
        cache_reads(arena, id, &mut reads);
        for &r in &reads {
            *pending.entry(r).or_insert(0) += 1;
        }
        reads.clear();
    }
    *pending.entry(expr).or_insert(0) += 1;

    for &id in &order {
        let node = arena.node(id);
        let child = |c: &ExprId| cached(&cache, *c);
        let func = |name: &str, c: &ExprId| -> Result<String, SymplexError> {
            Ok(apply(&mi(name), &[&child(c)?]))
        };
        let xml = match node {
            // ── Atoms ──────────────────────────────────────────────────
            ExprNode::Num(nid) => number(arena.num(*nid)),
            ExprNode::Symbol(sid) => symbol(arena.symbol_name(*sid)),
            ExprNode::PhysicalConstant(name_id, _) => symbol(arena.symbol_name(*name_id)),
            ExprNode::Pi => ident("pi"),
            // `ⅇ` / `ⅈ` (SymPy: `&ExponentialE;` / `&ImaginaryI;`), numeric.
            ExprNode::E => mi("&#x2147;"),
            ExprNode::ImaginaryUnit => mi("&#x2148;"),
            ExprNode::EulerGamma => ident("gamma"),
            ExprNode::Catalan => mi("G"),
            ExprNode::GoldenRatio => ident("phi"),
            ExprNode::Infinity => mi("&#x221E;"),
            ExprNode::NegInfinity => mrow(&format!("{}{}", mo("-"), mi("&#x221E;"))),
            ExprNode::ComplexInfinity => format!("<mover>{}<mo>~</mo></mover>", mi("&#x221E;")),
            ExprNode::NaN => "<mtext>NaN</mtext>".to_string(),
            ExprNode::BoolTrue => "<mtext>True</mtext>".to_string(),
            ExprNode::BoolFalse => "<mtext>False</mtext>".to_string(),
            ExprNode::EmptySet => mi("&#x2205;"),
            ExprNode::UniversalSet => "<mi mathvariant=\"double-struck\">R</mi>".to_string(),

            // ── Arithmetic ─────────────────────────────────────────────
            ExprNode::Add(children) => {
                if children.is_empty() {
                    mn("0")
                } else {
                    let mut ordered: Vec<ExprId> = children.to_vec();
                    ordered.sort_by_key(|a| display_sort_key(arena, *a));
                    let mut out = String::new();
                    for (i, &c) in ordered.iter().enumerate() {
                        let (negative, body) = signed_term(arena, c, &cache)?;
                        if i > 0 {
                            out.push_str(&mo(if negative { "-" } else { "+" }));
                        } else if negative {
                            out.push_str(&mo("-"));
                        }
                        out.push_str(&body);
                    }
                    mrow(&out)
                }
            }
            ExprNode::Mul(children) => render_mul(arena, children, &cache)?,
            ExprNode::Neg(inner) => {
                let xml = child(inner)?;
                let body = if matches!(arena.node(*inner), ExprNode::Add(_)) {
                    paren(&xml)
                } else {
                    xml
                };
                mrow(&format!("{}{body}", mo("-")))
            }
            ExprNode::Pow(base, exp) => render_pow(arena, *base, *exp, &cache)?,

            // ── Elementary functions ───────────────────────────────────
            ExprNode::Sin(a) => func("sin", a)?,
            ExprNode::Cos(a) => func("cos", a)?,
            ExprNode::Tan(a) => func("tan", a)?,
            ExprNode::Asin(a) => func("arcsin", a)?,
            ExprNode::Acos(a) => func("arccos", a)?,
            ExprNode::Atan(a) => func("arctan", a)?,
            ExprNode::Sinh(a) => func("sinh", a)?,
            ExprNode::Cosh(a) => func("cosh", a)?,
            ExprNode::Tanh(a) => func("tanh", a)?,
            ExprNode::Asinh(a) => func("asinh", a)?,
            ExprNode::Acosh(a) => func("acosh", a)?,
            ExprNode::Atanh(a) => func("atanh", a)?,
            ExprNode::Exp(a) => func("exp", a)?,
            ExprNode::Ln(a) => func("ln", a)?,
            ExprNode::Sign(a) => func("sgn", a)?,
            ExprNode::Heaviside(a) => func("H", a)?,
            ExprNode::DiracDelta(a) => apply(&ident("delta"), &[&child(a)?]),
            ExprNode::Gamma(a) => apply(&ident("Gamma"), &[&child(a)?]),
            ExprNode::LogGamma(a) => apply(
                &mrow(&format!("{}{}", mi("ln"), ident("Gamma"))),
                &[&child(a)?],
            ),
            ExprNode::Digamma(a) => apply(&ident("psi"), &[&child(a)?]),
            ExprNode::Erf(a) => func("erf", a)?,
            ExprNode::Erfc(a) => func("erfc", a)?,
            ExprNode::LambertW(a) => func("W", a)?,
            ExprNode::Re(a) => func("&#x211C;", a)?,
            ExprNode::Im(a) => func("&#x2111;", a)?,
            ExprNode::Conjugate(a) => format!("<mover>{}<mo>&#xAF;</mo></mover>", child(a)?),
            ExprNode::Arg(a) => func("arg", a)?,
            ExprNode::Si(a) => func("Si", a)?,
            ExprNode::Ci(a) => func("Ci", a)?,
            ExprNode::Ei(a) => func("Ei", a)?,
            ExprNode::Li(a) => func("li", a)?,
            ExprNode::Zeta(a) => apply(&ident("zeta"), &[&child(a)?]),
            ExprNode::Polygamma(n, a) => {
                let head = format!("<msup>{}{}</msup>", ident("psi"), paren(&child(n)?));
                apply(&head, &[&child(a)?])
            }
            ExprNode::KroneckerDelta(i, j) => format!(
                "<msub>{}{}</msub>",
                ident("delta"),
                mrow(&format!("{}{}", child(i)?, child(j)?))
            ),
            ExprNode::Abs(a) => mrow(&format!("{}{}{}", mo("|"), child(a)?, mo("|"))),
            ExprNode::Floor(a) => mrow(&format!("<mo>&#x230A;</mo>{}<mo>&#x230B;</mo>", child(a)?)),
            ExprNode::Ceiling(a) => {
                mrow(&format!("<mo>&#x2308;</mo>{}<mo>&#x2309;</mo>", child(a)?))
            }
            ExprNode::Factorial(a) => {
                let xml = child(a)?;
                let body = if arena.node(*a).is_atom() {
                    xml
                } else {
                    paren(&xml)
                };
                mrow(&format!("{body}{}", mo("!")))
            }
            ExprNode::Binomial(n, k) => mrow(&format!(
                "<mo>(</mo><mfrac linethickness=\"0\">{}{}</mfrac><mo>)</mo>",
                child(n)?,
                child(k)?
            )),
            ExprNode::Beta(a, b) => apply(&mi("B"), &[&child(a)?, &child(b)?]),
            ExprNode::Atan2(y, x) => apply(&mi("atan2"), &[&child(y)?, &child(x)?]),
            ExprNode::Min(args) => {
                let rendered: Vec<String> = args.iter().map(child).collect::<Result<_, _>>()?;
                let refs: Vec<&str> = rendered.iter().map(String::as_str).collect();
                apply(&mi("min"), &refs)
            }
            ExprNode::Max(args) => {
                let rendered: Vec<String> = args.iter().map(child).collect::<Result<_, _>>()?;
                let refs: Vec<&str> = rendered.iter().map(String::as_str).collect();
                apply(&mi("max"), &refs)
            }
            ExprNode::Apply(sid, args) => {
                let rendered: Vec<String> = args.iter().map(child).collect::<Result<_, _>>()?;
                render_apply(arena.symbol_name(*sid), &rendered)
            }

            // ── Calculus ───────────────────────────────────────────────
            ExprNode::Derivative(body, var) => mrow(&format!(
                "<mfrac>{}{}</mfrac>{}",
                mi("d"),
                mrow(&format!("{}{}", mi("d"), child(var)?)),
                child(body)?
            )),
            ExprNode::Integral(body, var) => mrow(&format!(
                "<mo>&#x222B;</mo>{}{}{}",
                child(body)?,
                mi("d"),
                child(var)?
            )),
            ExprNode::DefiniteIntegral(body, var, lo, hi) => mrow(&format!(
                "<msubsup><mo>&#x222B;</mo>{}{}</msubsup>{}{}{}",
                child(lo)?,
                child(hi)?,
                child(body)?,
                mi("d"),
                child(var)?
            )),
            ExprNode::Sum(body, var, lo, hi) => big_op(
                "&#x2211;",
                &child(var)?,
                &child(lo)?,
                &child(hi)?,
                &child(body)?,
            ),
            ExprNode::Product_(body, var, lo, hi) => big_op(
                "&#x220F;",
                &child(var)?,
                &child(lo)?,
                &child(hi)?,
                &child(body)?,
            ),
            ExprNode::Limit(body, var, point) => mrow(&format!(
                "<munder><mo>lim</mo>{}</munder>{}",
                mrow(&format!(
                    "{}<mo>&#x2192;</mo>{}",
                    child(var)?,
                    child(point)?
                )),
                child(body)?
            )),
            ExprNode::LaplaceTransform(body, _, _) => mrow(&format!(
                "<mi mathvariant=\"script\">L</mi>{APPLY}<mrow><mo>{{</mo>{}<mo>}}</mo></mrow>",
                child(body)?
            )),
            ExprNode::InverseLaplaceTransform(body, _, _) => mrow(&format!(
                "<msup><mi mathvariant=\"script\">L</mi>{}</msup>{APPLY}<mrow><mo>{{</mo>{}<mo>}}</mo></mrow>",
                mrow(&format!("{}{}", mo("-"), mn("1"))),
                child(body)?
            )),
            ExprNode::Residue(body, var, point) => mrow(&format!(
                "<munder><mi>Res</mi>{}</munder>{}",
                mrow(&format!("{}<mo>=</mo>{}", child(var)?, child(point)?)),
                child(body)?
            )),

            // ── Relations and logic ────────────────────────────────────
            ExprNode::Gt(a, b) => relation(&child(a)?, "&gt;", &child(b)?),
            ExprNode::Ge(a, b) => relation(&child(a)?, "&#x2265;", &child(b)?),
            ExprNode::Eq_(a, b) => relation(&child(a)?, "=", &child(b)?),
            ExprNode::Ne(a, b) => relation(&child(a)?, "&#x2260;", &child(b)?),
            ExprNode::And(children) => connective(arena, "&#x2227;", children, &cache)?,
            ExprNode::Or(children) => connective(arena, "&#x2228;", children, &cache)?,
            ExprNode::Not(a) => {
                let xml = child(a)?;
                let body = if matches!(arena.node(*a), ExprNode::And(_) | ExprNode::Or(_)) {
                    paren(&xml)
                } else {
                    xml
                };
                mrow(&format!("<mo>&#xAC;</mo>{body}"))
            }
            ExprNode::Piecewise(pieces) => {
                let mut rows = String::new();
                for &(val, cond) in pieces.iter() {
                    // A `True` guard is the default branch: "otherwise".
                    let guard = if matches!(arena.node(cond), ExprNode::BoolTrue) {
                        "<mtext>otherwise</mtext>".to_string()
                    } else {
                        format!("<mtext>if&#xA0;</mtext>{}", child(&cond)?)
                    };
                    rows.push_str(&format!(
                        "<mtr><mtd>{}</mtd><mtd>{guard}</mtd></mtr>",
                        child(&val)?
                    ));
                }
                mrow(&format!(
                    "<mo>{{</mo><mtable columnalign=\"left\">{rows}</mtable>"
                ))
            }

            // ── Sets ───────────────────────────────────────────────────
            ExprNode::Interval(a, b, flags) => {
                let left = if flags & INTERVAL_LEFT_OPEN != 0 {
                    "("
                } else {
                    "["
                };
                let right = if flags & INTERVAL_RIGHT_OPEN != 0 {
                    ")"
                } else {
                    "]"
                };
                mrow(&format!(
                    "{}{}{}{}{}",
                    mo(left),
                    child(a)?,
                    mo(","),
                    child(b)?,
                    mo(right)
                ))
            }
            ExprNode::FiniteSet(elems) => {
                let mut inner = String::new();
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        inner.push_str(&mo(","));
                    }
                    inner.push_str(&child(e)?);
                }
                mrow(&format!("<mo>{{</mo>{inner}<mo>}}</mo>"))
            }
            ExprNode::SetUnion(sets) => connective(arena, "&#x222A;", sets, &cache)?,
            ExprNode::SetIntersection(sets) => connective(arena, "&#x2229;", sets, &cache)?,
            ExprNode::SetComplement(a, b) => relation(&child(a)?, "&#x2216;", &child(b)?),
            ExprNode::ConditionSet(var, cond) => mrow(&format!(
                "<mo>{{</mo>{}<mo>|</mo>{}<mo>}}</mo>",
                child(var)?,
                child(cond)?
            )),

            // No standard presentation: internal CAS forms.
            ExprNode::Series(..)
            | ExprNode::DSolve(..)
            | ExprNode::RootOf(..)
            | ExprNode::RootSum(..) => {
                return Err(unsupported(&format!("`{}`", describe(node))));
            }
        };
        // Release the entries this node has just consumed.
        cache_reads(arena, id, &mut reads);
        for r in reads.drain(..) {
            if let Some(n) = pending.get_mut(&r) {
                *n = n.saturating_sub(1);
                if *n == 0 {
                    cache.remove(&r);
                }
            }
        }
        if pending.get(&id).is_some_and(|&n| n > 0) {
            cache.insert(id, xml);
        }
    }

    cached(&cache, expr)
}

/// A library or user `Apply` node: Bessel functions as `J_n(x)`, everything
/// else as `name(args)`.
fn render_apply(name: &str, args: &[String]) -> String {
    use crate::base::arena::{FN_BESSELI, FN_BESSELJ, FN_BESSELK, FN_BESSELY};
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let bessel = match name {
        n if n == FN_BESSELJ => Some("J"),
        n if n == FN_BESSELY => Some("Y"),
        n if n == FN_BESSELI => Some("I"),
        n if n == FN_BESSELK => Some("K"),
        _ => None,
    };
    if let (Some(letter), [order, arg]) = (bessel, refs.as_slice()) {
        let head = format!("<msub>{}{order}</msub>", mi(letter));
        return apply(&head, &[arg]);
    }
    apply(&mi(&escape(name)), &refs)
}

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

impl<S: Sort> Expr<S> {
    /// Render this expression as Presentation MathML, wrapped in a
    /// `<math xmlns="http://www.w3.org/1998/Math/MathML">` element
    /// (SymPy: `mathml(expr, printer='presentation')`).
    ///
    /// Layout follows [`to_latex`](Self::to_latex): `<mfrac>` for
    /// quotients, `<msqrt>`/`<mroot>` for roots, `<msup>` for powers,
    /// `<mi>sin</mi><mo>&#x2061;</mo>` for function application, Greek
    /// symbol names as character references, explicit `<mo>(</mo>` for
    /// parentheses.  Relations and connectives
    /// ([`BoolEx`](crate::api::expr::BoolEx)) render with `&gt;`, `&#x2265;`,
    /// `&#x2227;`, `&#x2228;`, `&#xAC;`.  See the [`mathml`](crate::mathml)
    /// module docs.
    ///
    /// # Errors
    ///
    /// [`SymplexError::NotImplemented`] for nodes without a standard
    /// presentation (`Series`, `DSolve`, `RootOf`, `RootSum`).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!(
    ///     (&x.powi(2) + 1).to_mathml().unwrap(),
    ///     "<math xmlns=\"http://www.w3.org/1998/Math/MathML\">\
    ///      <mrow><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><mn>1</mn></mrow></math>"
    /// );
    /// assert_eq!(
    ///     (&x / 2).to_mathml().unwrap(),
    ///     "<math xmlns=\"http://www.w3.org/1998/Math/MathML\">\
    ///      <mfrac><mi>x</mi><mn>2</mn></mfrac></math>"
    /// );
    /// assert_eq!(
    ///     x.sin().to_mathml().unwrap(),
    ///     "<math xmlns=\"http://www.w3.org/1998/Math/MathML\">\
    ///      <mrow><mi>sin</mi><mo>&#x2061;</mo><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow></math>"
    /// );
    /// assert_eq!(
    ///     x.gt(&ctx.int(0)).to_mathml().unwrap(),
    ///     "<math xmlns=\"http://www.w3.org/1998/Math/MathML\">\
    ///      <mrow><mi>x</mi><mo>&gt;</mo><mn>0</mn></mrow></math>"
    /// );
    /// ```
    pub fn to_mathml(&self) -> Result<String, SymplexError> {
        let inner = self.inner.read();
        let body = render(&inner.arena, self.raw_id())?;
        Ok(format!("<math xmlns=\"{MATHML_NS}\">{body}</math>"))
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    fn body(e: &str) -> String {
        let ctx = Context::new();
        let ex = ctx.parse(e).unwrap();
        let xml = ex.to_mathml().unwrap();
        let start = "<math xmlns=\"http://www.w3.org/1998/Math/MathML\">";
        assert!(xml.starts_with(start) && xml.ends_with("</math>"), "{xml}");
        xml[start.len()..xml.len() - "</math>".len()].to_string()
    }

    /// Every `<tag>` has a matching `</tag>` in the right order.
    fn assert_balanced(xml: &str) {
        let mut stack: Vec<String> = Vec::new();
        let mut rest = xml;
        while let Some(open) = rest.find('<') {
            let Some(close) = rest[open..].find('>') else {
                panic!("unterminated tag in {xml}");
            };
            let tag = &rest[open + 1..open + close];
            if let Some(name) = tag.strip_prefix('/') {
                assert_eq!(stack.pop().as_deref(), Some(name), "mismatch in {xml}");
            } else if !tag.ends_with('/') {
                let name = tag.split(' ').next().unwrap_or(tag);
                stack.push(name.to_string());
            }
            rest = &rest[open + close + 1..];
        }
        assert!(stack.is_empty(), "unclosed {stack:?} in {xml}");
    }

    #[test]
    fn atoms() {
        assert_eq!(body("42"), "<mn>42</mn>");
        assert_eq!(body("-3"), "<mrow><mo>-</mo><mn>3</mn></mrow>");
        assert_eq!(body("1/2"), "<mfrac><mn>1</mn><mn>2</mn></mfrac>");
        assert_eq!(
            body("-1/2"),
            "<mrow><mo>-</mo><mfrac><mn>1</mn><mn>2</mn></mfrac></mrow>"
        );
        assert_eq!(body("alpha"), "<mi>&#x3B1;</mi>");
        assert_eq!(body("x_1"), "<msub><mi>x</mi><mn>1</mn></msub>");
        assert_eq!(
            body("theta_max"),
            "<msub><mi>&#x3B8;</mi><mi>max</mi></msub>"
        );
        assert_eq!(body("pi"), "<mi>&#x3C0;</mi>");
        assert_eq!(body("inf"), "<mi>&#x221E;</mi>");
    }

    #[test]
    fn products_and_fractions() {
        assert_eq!(
            body("2*x*y"),
            "<mrow><mn>2</mn><mo>&#x2062;</mo><mi>x</mi><mo>&#x2062;</mo><mi>y</mi></mrow>"
        );
        assert_eq!(body("x/y"), "<mfrac><mi>x</mi><mi>y</mi></mfrac>");
        assert_eq!(body("1/x"), "<mfrac><mn>1</mn><mi>x</mi></mfrac>");
        assert_eq!(
            body("1/x^2"),
            "<mfrac><mn>1</mn><msup><mi>x</mi><mn>2</mn></msup></mfrac>"
        );
        assert_eq!(body("-x"), "<mrow><mo>-</mo><mi>x</mi></mrow>");
        assert_eq!(
            body("-x/2"),
            "<mrow><mo>-</mo><mfrac><mi>x</mi><mn>2</mn></mfrac></mrow>"
        );
        assert_eq!(
            body("y*(x+1)"),
            "<mrow><mi>y</mi><mo>&#x2062;</mo><mrow><mo>(</mo><mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow><mo>)</mo></mrow></mrow>"
        );
        assert_eq!(
            body("2*3^x"),
            "<mrow><mn>2</mn><mo>&#x2062;</mo><msup><mn>3</mn><mi>x</mi></msup></mrow>"
        );
    }

    #[test]
    fn symbol_names_are_escaped() {
        let ctx = Context::new();
        let odd = ctx.symbol("a<b&c");
        assert!(odd.to_mathml().unwrap().contains("<mi>a&lt;b&amp;c</mi>"));
    }

    #[test]
    fn sums_and_subtraction() {
        assert_eq!(body("x - y"), "<mrow><mi>x</mi><mo>-</mo><mi>y</mi></mrow>");
        assert_eq!(
            body("x^2 - 2*x + 1"),
            "<mrow><msup><mi>x</mi><mn>2</mn></msup><mo>-</mo><mrow><mn>2</mn><mo>&#x2062;</mo><mi>x</mi></mrow><mo>+</mo><mn>1</mn></mrow>"
        );
        assert_eq!(
            body("1 - x/2"),
            "<mrow><mo>-</mo><mfrac><mi>x</mi><mn>2</mn></mfrac><mo>+</mo><mn>1</mn></mrow>"
        );
    }

    #[test]
    fn powers_and_roots() {
        assert_eq!(body("sqrt(x)"), "<msqrt><mi>x</mi></msqrt>");
        assert_eq!(body("cbrt(x)"), "<mroot><mi>x</mi><mn>3</mn></mroot>");
        assert_eq!(
            body("(x+1)^2"),
            "<msup><mrow><mo>(</mo><mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow><mo>)</mo></mrow><mn>2</mn></msup>"
        );
        assert_eq!(body("2^x"), "<msup><mn>2</mn><mi>x</mi></msup>");
        assert_eq!(
            body("sin(x)^2"),
            "<mrow><msup><mi>sin</mi><mn>2</mn></msup><mo>&#x2061;</mo><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>"
        );
        assert_eq!(
            body("(-2)^x"),
            "<msup><mrow><mo>(</mo><mrow><mo>-</mo><mn>2</mn></mrow><mo>)</mo></mrow><mi>x</mi></msup>"
        );
    }

    #[test]
    fn functions_and_special_nodes() {
        assert_eq!(
            body("exp(x)"),
            "<mrow><mi>exp</mi><mo>&#x2061;</mo><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>"
        );
        assert_eq!(
            body("abs(x)"),
            "<mrow><mo>|</mo><mi>x</mi><mo>|</mo></mrow>"
        );
        assert_eq!(
            body("floor(x)"),
            "<mrow><mo>&#x230A;</mo><mi>x</mi><mo>&#x230B;</mo></mrow>"
        );
        assert_eq!(body("x!"), "<mrow><mi>x</mi><mo>!</mo></mrow>");
        assert_eq!(
            body("gamma(x)"),
            "<mrow><mi>&#x393;</mi><mo>&#x2061;</mo><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>"
        );
        assert_eq!(
            body("besselj(0, x)"),
            "<mrow><msub><mi>J</mi><mn>0</mn></msub><mo>&#x2061;</mo><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>"
        );
        assert_eq!(
            body("Integral(x^2, x)"),
            "<mrow><mo>&#x222B;</mo><msup><mi>x</mi><mn>2</mn></msup><mi>d</mi><mi>x</mi></mrow>"
        );
        assert_eq!(
            body("Sum(k, k=1..n)"),
            "<mrow><munderover><mo>&#x2211;</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><mi>k</mi></mrow>"
        );
    }

    #[test]
    fn relations_and_logic() {
        let ctx = Context::new();
        let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
        let zero = ctx.int(0);
        let p = x.gt(&zero).and(&y.le(&zero));
        let xml = p.to_mathml().unwrap();
        assert!(
            xml.contains("<mrow><mrow><mi>x</mi><mo>&gt;</mo><mn>0</mn></mrow><mo>&#x2227;</mo><mrow><mn>0</mn><mo>&#x2265;</mo><mi>y</mi></mrow></mrow>"),
            "{xml}"
        );
        let q = x.gt(&zero).or(&y.gt(&zero)).not();
        assert!(
            q.to_mathml()
                .unwrap()
                .contains("<mo>&#xAC;</mo><mrow><mo>(</mo>"),
            "{}",
            q.to_mathml().unwrap()
        );
        let pw = Ex::piecewise(&[(&x, &x.gt(&zero)), (&(-&x), &x.le(&zero))]);
        let xml = pw.to_mathml().unwrap();
        assert!(
            xml.contains("<mtable columnalign=\"left\"><mtr><mtd><mi>x</mi></mtd>"),
            "{xml}"
        );
        assert_balanced(&xml);
    }

    #[test]
    fn well_formed_and_errors() {
        for e in [
            "x^2 + 1",
            "sin(x)/2 + sqrt(x)",
            "(x+1)/(x-1)",
            "-(x+1)^3",
            "exp(-x^2/2)/sqrt(2*pi)",
            "Limit(sin(x)/x, x, 0)",
            "Integral(exp(-x), x, 0, inf)",
            "besselk(2, x) + erf(x)*polygamma(1, x)",
            "min(x, y, 1) + C(n, k) + B(a, b)",
        ] {
            assert_balanced(&body(e));
        }
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let r = ctx.parse("RootOf(x^5 - x - 1, 0)").unwrap();
        assert!(matches!(
            r.to_mathml(),
            Err(SymplexError::NotImplemented(_))
        ));
        assert!(x.to_mathml().unwrap().contains("<mi>x</mi>"));
    }

    #[test]
    fn deep_expression_no_stack_overflow() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let mut e = x.clone();
        for _ in 0..5000 {
            e = e.sin();
        }
        let xml = e.to_mathml().unwrap();
        assert_eq!(xml.matches("<mi>sin</mi>").count(), 5000);
    }
}
