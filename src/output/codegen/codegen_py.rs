//! Python, NumPy and Julia code generation from symbolic expressions (0.9.1).
//!
//! One table-driven printer serves three targets (SymPy: `pycode`,
//! `NumPyPrinter().doprint`, `julia_code`):
//!
//! | Target | Functions | Constants | Power | Relations / logic |
//! |--------|-----------|-----------|-------|-------------------|
//! | Python | `math.sin`, `math.exp`, `math.sqrt`, `math.gamma`, `math.erf`, … ; `abs`, `min`, `max` | `math.pi`, `math.e`, `math.inf`, `math.nan` | `**` | `>`, `and`, `or`, `not`; `(v if c else …)` |
//! | NumPy  | `numpy.sin`, `numpy.sqrt`, `numpy.floor`, `numpy.sign`, `numpy.minimum`, … | `numpy.pi`, `numpy.e`, `numpy.inf`, `numpy.nan` | `**` | `numpy.greater`, `numpy.logical_and`, `numpy.select` |
//! | Julia  | `sin`, `exp`, `sqrt`, `cbrt`, `abs`, `floor`, `sign`, `min`, `max`, `atan(y, x)` | `pi`, `ℯ`, `Inf`, `NaN` | `^` | `>`, `&&`, `\|\|`, `!`; `(c ? v : …)` |
//!
//! Numbers are exact: integers print as integers, rationals as `(p/q)`
//! (a float division in all three languages).  Sums are printed in display
//! order with `-` for negated terms; `x^(-1)` factors become divisions.
//! Rational powers with an odd denominator are *real* roots, as in the
//! Rust and C back ends and `compile()`: `x^(1/3)` is `math.copysign(abs(x)**(1/3), x)`
//! / `numpy.cbrt(x)` / `cbrt(x)`, and `x^(p/q)` is `copysign(abs(x)**(p/q), x)`
//! for odd `p` or `abs(x)**(p/q)` for even `p` (a bare `x**(1/3)` is complex
//! for negative `x` in Python).  `x!` is `math.gamma(x + 1)` in Python
//! (`math.factorial` rejects non-integers).
//! Nothing is emitted for a node the target cannot express (Bessel
//! functions, `digamma`, `LambertW`, `zeta`, unevaluated integrals, sets,
//! `I`) — those return [`SymplexError::NotImplemented`] instead of a guess.
//! NumPy has no `gamma`/`erf`/`factorial` (SciPy would be needed) and base
//! Julia has no `gamma` (SpecialFunctions.jl); they are refused too.
//!
//! The `*_fn` variants wrap the expression in a function definition with
//! common subexpressions hoisted into `t0`, `t1`, … temporaries; symbols
//! that are not parameters are [`SymplexError::FreeSymbol`] errors.  The
//! expression-only variants (`to_python` & co.) print every symbol by name.
//!
//! The printer is an iterative post-order walk (no recursion).

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed};
use rustc_hash::FxHashMap;

use crate::api::expr::{Expr, Sort};
use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;
use crate::output::common::display_sort_key;

// ═══════════════════════════════════════════════════════════════════════════
// Targets
// ═══════════════════════════════════════════════════════════════════════════

/// The language and library dialect to print.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Target {
    /// Python 3 with the `math` module.
    Python,
    /// Python 3 with `numpy` (vectorised).
    NumPy,
    /// Julia (base library only).
    Julia,
}

impl Target {
    fn name(self) -> &'static str {
        match self {
            Target::Python => "to_python",
            Target::NumPy => "to_numpy",
            Target::Julia => "to_julia",
        }
    }

    fn pow_op(self) -> &'static str {
        match self {
            Target::Python | Target::NumPy => "**",
            Target::Julia => "^",
        }
    }

    fn and_op(self) -> &'static str {
        match self {
            Target::Python => " and ",
            Target::NumPy => "",
            Target::Julia => " && ",
        }
    }

    fn or_op(self) -> &'static str {
        match self {
            Target::Python => " or ",
            Target::NumPy => "",
            Target::Julia => " || ",
        }
    }

    /// The name of a one-argument elementary function, or `None` when the
    /// target has no spelling for it.
    fn unary(self, node: &ExprNode) -> Option<&'static str> {
        use ExprNode as N;
        let py = |m: &'static str, np: &'static str, jl: &'static str| match self {
            Target::Python => m,
            Target::NumPy => np,
            Target::Julia => jl,
        };
        Some(match node {
            N::Sin(_) => py("math.sin", "numpy.sin", "sin"),
            N::Cos(_) => py("math.cos", "numpy.cos", "cos"),
            N::Tan(_) => py("math.tan", "numpy.tan", "tan"),
            N::Asin(_) => py("math.asin", "numpy.arcsin", "asin"),
            N::Acos(_) => py("math.acos", "numpy.arccos", "acos"),
            N::Atan(_) => py("math.atan", "numpy.arctan", "atan"),
            N::Sinh(_) => py("math.sinh", "numpy.sinh", "sinh"),
            N::Cosh(_) => py("math.cosh", "numpy.cosh", "cosh"),
            N::Tanh(_) => py("math.tanh", "numpy.tanh", "tanh"),
            N::Asinh(_) => py("math.asinh", "numpy.arcsinh", "asinh"),
            N::Acosh(_) => py("math.acosh", "numpy.arccosh", "acosh"),
            N::Atanh(_) => py("math.atanh", "numpy.arctanh", "atanh"),
            N::Exp(_) => py("math.exp", "numpy.exp", "exp"),
            N::Ln(_) => py("math.log", "numpy.log", "log"),
            N::Abs(_) => py("abs", "abs", "abs"),
            N::Floor(_) => py("math.floor", "numpy.floor", "floor"),
            N::Ceiling(_) => py("math.ceil", "numpy.ceil", "ceil"),
            N::Sign(_) => match self {
                Target::Python => return None,
                Target::NumPy => "numpy.sign",
                Target::Julia => "sign",
            },
            N::Gamma(_) if self == Target::Python => "math.gamma",
            N::LogGamma(_) if self == Target::Python => "math.lgamma",
            N::Erf(_) if self == Target::Python => "math.erf",
            N::Erfc(_) if self == Target::Python => "math.erfc",
            // `Factorial` is rendered as `math.gamma(x + 1)` (real argument).
            _ => return None,
        })
    }

    fn constant(self, node: &ExprNode) -> Option<&'static str> {
        let py = |m: &'static str, np: &'static str, jl: &'static str| match self {
            Target::Python => m,
            Target::NumPy => np,
            Target::Julia => jl,
        };
        Some(match node {
            ExprNode::Pi => py("math.pi", "numpy.pi", "pi"),
            ExprNode::E => py("math.e", "numpy.e", "ℯ"),
            ExprNode::Infinity => py("math.inf", "numpy.inf", "Inf"),
            ExprNode::NegInfinity => py("(-math.inf)", "(-numpy.inf)", "(-Inf)"),
            ExprNode::NaN => py("math.nan", "numpy.nan", "NaN"),
            ExprNode::EulerGamma => "0.5772156649015329",
            ExprNode::Catalan => "0.915965594177219",
            ExprNode::GoldenRatio => "1.618033988749895",
            _ => return None,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Precedence
// ═══════════════════════════════════════════════════════════════════════════

// Shared by all three targets: Python's `or < and < not < comparison <
// + - < * / < unary - < **` and Julia's `|| < && < comparison < + - <
// * / < unary - < ^` agree on the relative order of everything we emit.
const PREC_OR: u8 = 1;
const PREC_AND: u8 = 2;
const PREC_NOT: u8 = 3;
const PREC_REL: u8 = 4;
const PREC_ADD: u8 = 5;
const PREC_MUL: u8 = 6;
const PREC_NEG: u8 = 7;
const PREC_POW: u8 = 8;
const PREC_ATOM: u8 = 9;

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
            prec: PREC_ATOM,
        }
    }

    fn new(text: String, prec: u8) -> Self {
        Rendered { text, prec }
    }

    /// The text, parenthesised if it binds looser than `min_prec`.
    fn at(&self, min_prec: u8) -> String {
        if self.prec < min_prec {
            format!("({})", self.text)
        } else {
            self.text.clone()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Emitter
// ═══════════════════════════════════════════════════════════════════════════

type Cache = FxHashMap<ExprId, Rendered>;

struct Emitter<'a> {
    arena: &'a Arena,
    target: Target,
    /// `Some(params)`: only these symbols may appear (function mode).
    params: Option<&'a [&'a str]>,
    /// CSE binding symbols → temporary index.
    cse_slots: &'a FxHashMap<ExprId, usize>,
}

impl Emitter<'_> {
    fn unsupported(&self, what: &str) -> SymplexError {
        SymplexError::NotImplemented(format!(
            "{}: cannot generate code for `{what}`",
            self.target.name()
        ))
    }

    fn cached(&self, cache: &Cache, id: ExprId) -> Result<Rendered, SymplexError> {
        cache
            .get(&id)
            .cloned()
            .ok_or_else(|| self.unsupported("an unrendered sub-expression"))
    }

    /// An exact literal: integers bare, rationals as `(p/q)`.
    fn number(&self, r: &Ratio<BigInt>) -> Rendered {
        if r.is_integer() {
            if r.is_negative() {
                Rendered::new(r.numer().to_string(), PREC_NEG)
            } else {
                Rendered::atom(r.numer().to_string())
            }
        } else {
            Rendered::atom(format!("({}/{})", r.numer(), r.denom()))
        }
    }

    fn call(&self, name: &str, args: &[&Rendered]) -> Rendered {
        let list: Vec<String> = args.iter().map(|a| a.at(0)).collect();
        Rendered::atom(format!("{name}({})", list.join(", ")))
    }

    /// `coeff * factors / denominators`; `coeff` is already non-negative.
    fn product(
        &self,
        coeff: Option<&Ratio<BigInt>>,
        factors: &[ExprId],
        cache: &Cache,
    ) -> Result<Rendered, SymplexError> {
        let mut numer: Vec<String> = Vec::new();
        let mut denom: Vec<String> = Vec::new();
        // The lone factor when the coefficient contributes nothing: it is
        // returned as rendered, keeping its own precedence.
        let mut single: Option<Rendered> = None;
        if let Some(c) = coeff {
            if !c.numer().is_one() {
                numer.push(c.numer().to_string());
            }
            if !c.denom().is_one() {
                denom.push(c.denom().to_string());
            }
        }
        for &f in factors {
            if let ExprNode::Pow(base, exp) = self.arena.node(f)
                && let Some(r) = self.arena.as_num(*exp)
                && r.is_negative()
            {
                let base_r = self.cached(cache, *base)?;
                let pos = -r.clone();
                if pos.is_one() {
                    denom.push(base_r.at(PREC_MUL + 1));
                } else {
                    denom.push(self.pow_rational(&base_r, &pos).at(PREC_MUL + 1));
                }
                continue;
            }
            let r = self.cached(cache, f)?;
            numer.push(r.at(PREC_MUL + 1));
            single = if numer.len() == 1 { Some(r) } else { None };
        }
        let numer_text = if numer.is_empty() {
            "1".to_string()
        } else {
            numer.join("*")
        };
        if denom.is_empty() {
            return Ok(match single {
                Some(r) => r,
                None if numer.len() <= 1 => Rendered::atom(numer_text),
                None => Rendered::new(numer_text, PREC_MUL),
            });
        }
        let denom_text = if denom.len() == 1 {
            denom.remove(0)
        } else {
            format!("({})", denom.join("*"))
        };
        Ok(Rendered::new(
            format!("{numer_text}/{denom_text}"),
            PREC_MUL,
        ))
    }

    /// `base ** exp` with the target's operator; the base is parenthesised
    /// unless it is an atom, the exponent unless it is at least a power.
    fn pow_text(&self, base: &Rendered, exp: &Rendered) -> String {
        format!(
            "{}{}{}",
            base.at(PREC_ATOM),
            self.target.pow_op(),
            exp.at(PREC_POW)
        )
    }

    /// `b ^ r` for an exact rational exponent.
    ///
    /// `1/2` is `sqrt`; an odd denominator `q` is a *real* root, as in
    /// `compile()` and the Rust/C back ends: `sign(b)·|b|^(p/q)` for odd
    /// `p`, `|b|^(p/q)` for even `p` (`numpy.cbrt`/`cbrt` for `1/3`).  A
    /// bare `b**(1/3)` would be complex for negative `b` in Python.
    fn pow_rational(&self, b: &Rendered, r: &Ratio<BigInt>) -> Rendered {
        let one = BigInt::one();
        let two = BigInt::from(2);
        if *r.numer() == one && *r.denom() == two {
            let sqrt = match self.target {
                Target::Python => "math.sqrt",
                Target::NumPy => "numpy.sqrt",
                Target::Julia => "sqrt",
            };
            return self.call(sqrt, &[b]);
        }
        if *r.numer() == one
            && *r.denom() == BigInt::from(3)
            && let Some(cbrt) = match self.target {
                Target::Python => None,
                Target::NumPy => Some("numpy.cbrt"),
                Target::Julia => Some("cbrt"),
            }
        {
            return self.call(cbrt, &[b]);
        }
        if r.is_integer() {
            if *r.numer() == -one {
                return Rendered::new(format!("1/{}", b.at(PREC_MUL + 1)), PREC_MUL);
            }
            return Rendered::new(self.pow_text(b, &self.number(r)), PREC_POW);
        }
        if (r.denom() % &two) != BigInt::from(0) {
            let (abs, copysign) = match self.target {
                Target::Python => ("abs", "math.copysign"),
                Target::NumPy => ("numpy.abs", "numpy.copysign"),
                Target::Julia => ("abs", "copysign"),
            };
            let mag = Rendered::new(
                self.pow_text(&self.call(abs, &[b]), &self.number(r)),
                PREC_POW,
            );
            let odd_numer = (r.numer() % &two) != BigInt::from(0);
            return if odd_numer {
                self.call(copysign, &[&mag, b])
            } else {
                mag
            };
        }
        Rendered::new(self.pow_text(b, &self.number(r)), PREC_POW)
    }

    fn render_pow(
        &self,
        base: ExprId,
        exp: ExprId,
        cache: &Cache,
    ) -> Result<Rendered, SymplexError> {
        let b = self.cached(cache, base)?;
        if let Some(r) = self.arena.as_num(exp) {
            return Ok(self.pow_rational(&b, r));
        }
        let e = self.cached(cache, exp)?;
        Ok(Rendered::new(self.pow_text(&b, &e), PREC_POW))
    }

    /// `(negative, |term|)` for a summand.
    fn signed_term(&self, id: ExprId, cache: &Cache) -> Result<(bool, Rendered), SymplexError> {
        match self.arena.node(id) {
            ExprNode::Num(nid) if self.arena.num(*nid).is_negative() => {
                Ok((true, self.number(&-self.arena.num(*nid).clone())))
            }
            ExprNode::Neg(inner) => Ok((true, self.cached(cache, *inner)?)),
            ExprNode::Mul(children) => {
                if let Some(&first) = children.first()
                    && let Some(c) = self.arena.as_num(first)
                    && c.is_negative()
                    && children.len() > 1
                {
                    let body = self.product(Some(&(-c.clone())), &children[1..], cache)?;
                    return Ok((true, body));
                }
                Ok((false, self.cached(cache, id)?))
            }
            _ => Ok((false, self.cached(cache, id)?)),
        }
    }

    fn render_add(&self, children: &[ExprId], cache: &Cache) -> Result<Rendered, SymplexError> {
        if children.is_empty() {
            return Ok(Rendered::atom("0"));
        }
        let mut ordered: Vec<ExprId> = children.to_vec();
        ordered.sort_by_key(|a| display_sort_key(self.arena, *a));
        let mut text = String::new();
        for (i, &c) in ordered.iter().enumerate() {
            let (negative, body) = self.signed_term(c, cache)?;
            if i == 0 {
                if negative {
                    // `-a*b` and `-a/b` need no parentheses: unary minus
                    // distributes over a product.
                    text.push('-');
                    text.push_str(&body.at(PREC_MUL));
                } else {
                    text.push_str(&body.at(PREC_ADD));
                }
            } else {
                text.push_str(if negative { " - " } else { " + " });
                text.push_str(&body.at(PREC_ADD + 1));
            }
        }
        let prec = if ordered.len() == 1 {
            let (negative, body) = self.signed_term(ordered[0], cache)?;
            if negative { PREC_NEG } else { body.prec }
        } else {
            PREC_ADD
        };
        Ok(Rendered::new(text, prec))
    }

    fn render_mul(&self, children: &[ExprId], cache: &Cache) -> Result<Rendered, SymplexError> {
        if children.is_empty() {
            return Ok(Rendered::atom("1"));
        }
        if let Some(&first) = children.first()
            && let Some(c) = self.arena.as_num(first)
            && children.len() > 1
        {
            if c.is_negative() {
                let body = self.product(Some(&(-c.clone())), &children[1..], cache)?;
                return Ok(Rendered::new(format!("-{}", body.at(PREC_MUL)), PREC_NEG));
            }
            return self.product(Some(c), &children[1..], cache);
        }
        self.product(None, children, cache)
    }

    fn relation(&self, op: &str, lhs: &Rendered, rhs: &Rendered) -> Rendered {
        Rendered::new(
            format!("{} {op} {}", lhs.at(PREC_ADD), rhs.at(PREC_ADD)),
            PREC_REL,
        )
    }

    /// NumPy's element-wise comparison functions.
    fn np_relation(&self, func: &str, lhs: &Rendered, rhs: &Rendered) -> Rendered {
        self.call(&format!("numpy.{func}"), &[lhs, rhs])
    }

    /// `a and b and c` / `numpy.logical_and(numpy.logical_and(a, b), c)`.
    fn connective(
        &self,
        op: &str,
        np_func: &str,
        prec: u8,
        children: &[ExprId],
        cache: &Cache,
    ) -> Result<Rendered, SymplexError> {
        let parts: Vec<Rendered> = children
            .iter()
            .map(|c| self.cached(cache, *c))
            .collect::<Result<_, _>>()?;
        if self.target == Target::NumPy {
            let mut acc = parts
                .first()
                .cloned()
                .unwrap_or_else(|| Rendered::atom("True"));
            for p in &parts[1.min(parts.len())..] {
                acc = self.call(&format!("numpy.{np_func}"), &[&acc, p]);
            }
            return Ok(acc);
        }
        let text: Vec<String> = parts.iter().map(|p| p.at(prec + 1)).collect();
        Ok(Rendered::new(text.join(op), prec))
    }

    fn render_piecewise(
        &self,
        pieces: &[(ExprId, ExprId)],
        cache: &Cache,
    ) -> Result<Rendered, SymplexError> {
        let nan = match self.target {
            Target::Python => "math.nan",
            Target::NumPy => "numpy.nan",
            Target::Julia => "NaN",
        };
        if self.target == Target::NumPy {
            let mut conds = Vec::new();
            let mut vals = Vec::new();
            for &(v, c) in pieces {
                let c_r = if c == self.arena.bool_true() {
                    Rendered::atom("True")
                } else {
                    self.cached(cache, c)?
                };
                conds.push(c_r.at(0));
                vals.push(self.cached(cache, v)?.at(0));
            }
            return Ok(Rendered::atom(format!(
                "numpy.select([{}], [{}], default={nan})",
                conds.join(", "),
                vals.join(", ")
            )));
        }
        // Right-nested conditional expression; a trailing `True` branch is
        // the unconditional default.  Inner conditionals are parenthesised
        // for readability (both languages nest to the right anyway).
        let mut acc = nan.to_string();
        let mut acc_is_cond = false;
        for (i, &(v, c)) in pieces.iter().enumerate().rev() {
            let v_r = self.cached(cache, v)?;
            if i + 1 == pieces.len() && c == self.arena.bool_true() {
                acc = v_r.at(PREC_OR);
                continue;
            }
            let c_r = self.cached(cache, c)?;
            let rest = if acc_is_cond { format!("({acc})") } else { acc };
            acc = match self.target {
                Target::Julia => format!("{} ? {} : {rest}", c_r.at(PREC_OR), v_r.at(PREC_OR)),
                _ => format!("{} if {} else {rest}", v_r.at(PREC_OR), c_r.at(PREC_OR)),
            };
            acc_is_cond = true;
        }
        Ok(Rendered::atom(format!("({acc})")))
    }

    /// Render `root`; every symbol either a parameter, a CSE temporary, or
    /// (expression mode) itself.
    fn render(&self, root: ExprId) -> Result<Rendered, SymplexError> {
        let arena = self.arena;
        let order = walk::post_order_ids(arena, root);
        let mut cache: Cache = FxHashMap::default();
        for &id in &order {
            let node = arena.node(id);
            let child = |c: &ExprId| self.cached(&cache, *c);
            let rendered = match node {
                ExprNode::Num(nid) => self.number(arena.num(*nid)),
                ExprNode::Symbol(sid) => {
                    if let Some(&slot) = self.cse_slots.get(&id) {
                        Rendered::atom(format!("t{slot}"))
                    } else {
                        let name = arena.symbol_name(*sid);
                        if let Some(params) = self.params
                            && !params.contains(&name)
                        {
                            return Err(SymplexError::FreeSymbol {
                                name: name.to_string(),
                            });
                        }
                        Rendered::atom(name)
                    }
                }
                ExprNode::PhysicalConstant(_, value) => child(value)?,
                ExprNode::Pi
                | ExprNode::E
                | ExprNode::Infinity
                | ExprNode::NegInfinity
                | ExprNode::NaN
                | ExprNode::EulerGamma
                | ExprNode::Catalan
                | ExprNode::GoldenRatio => {
                    let text = self
                        .target
                        .constant(node)
                        .ok_or_else(|| self.unsupported("a named constant"))?;
                    Rendered::atom(text)
                }
                ExprNode::BoolTrue => Rendered::atom(match self.target {
                    Target::Julia => "true",
                    _ => "True",
                }),
                ExprNode::BoolFalse => Rendered::atom(match self.target {
                    Target::Julia => "false",
                    _ => "False",
                }),

                ExprNode::Add(children) => self.render_add(children, &cache)?,
                ExprNode::Mul(children) => self.render_mul(children, &cache)?,
                ExprNode::Neg(inner) => {
                    let r = child(inner)?;
                    Rendered::new(format!("-{}", r.at(PREC_MUL)), PREC_NEG)
                }
                ExprNode::Pow(base, exp) => self.render_pow(*base, *exp, &cache)?,

                ExprNode::Sign(a) if self.target == Target::Python => {
                    let r = child(a)?;
                    Rendered::atom(format!(
                        "(0.0 if {} == 0 else math.copysign(1, {}))",
                        r.at(PREC_ADD),
                        r.at(0)
                    ))
                }
                ExprNode::Heaviside(a) => {
                    let r = child(a)?;
                    match self.target {
                        Target::Python => Rendered::atom(format!(
                            "(0.0 if {a} < 0 else (0.5 if {a} == 0 else 1.0))",
                            a = r.at(PREC_ADD)
                        )),
                        Target::NumPy => {
                            self.call("numpy.heaviside", &[&r, &Rendered::atom("0.5")])
                        }
                        Target::Julia => Rendered::atom(format!(
                            "({a} < 0 ? 0.0 : ({a} == 0 ? 0.5 : 1.0))",
                            a = r.at(PREC_ADD)
                        )),
                    }
                }
                ExprNode::Atan2(y, x) => {
                    let name = match self.target {
                        Target::Python => "math.atan2",
                        Target::NumPy => "numpy.arctan2",
                        Target::Julia => "atan",
                    };
                    self.call(name, &[&child(y)?, &child(x)?])
                }
                // `math.factorial` is integer-only (TypeError on floats since
                // Python 3.10); the Rust/C back ends use Γ(x + 1) on reals too.
                // NumPy and base Julia have no gamma: refused below.
                ExprNode::Factorial(a) if self.target == Target::Python => {
                    let r = child(a)?;
                    Rendered::atom(format!("math.gamma({} + 1)", r.at(PREC_ADD)))
                }
                ExprNode::Min(args) | ExprNode::Max(args) => {
                    let is_min = matches!(node, ExprNode::Min(_));
                    let parts: Vec<Rendered> = args.iter().map(child).collect::<Result<_, _>>()?;
                    if parts.is_empty() {
                        // min ∅ = +∞, max ∅ = −∞ (as in the Rust and C back ends),
                        // spelled like the `Infinity`/`NegInfinity` constants.
                        let empty = if is_min {
                            ExprNode::Infinity
                        } else {
                            ExprNode::NegInfinity
                        };
                        let text = self
                            .target
                            .constant(&empty)
                            .ok_or_else(|| self.unsupported("an empty min/max"))?;
                        Rendered::atom(text)
                    } else if self.target == Target::NumPy {
                        let f = if is_min {
                            "numpy.minimum"
                        } else {
                            "numpy.maximum"
                        };
                        let mut acc = parts[0].clone();
                        for p in &parts[1..] {
                            acc = self.call(f, &[&acc, p]);
                        }
                        acc
                    } else {
                        let refs: Vec<&Rendered> = parts.iter().collect();
                        self.call(if is_min { "min" } else { "max" }, &refs)
                    }
                }
                ExprNode::Piecewise(pieces) => self.render_piecewise(pieces, &cache)?,

                ExprNode::Gt(a, b) => match self.target {
                    Target::NumPy => self.np_relation("greater", &child(a)?, &child(b)?),
                    _ => self.relation(">", &child(a)?, &child(b)?),
                },
                ExprNode::Ge(a, b) => match self.target {
                    Target::NumPy => self.np_relation("greater_equal", &child(a)?, &child(b)?),
                    _ => self.relation(">=", &child(a)?, &child(b)?),
                },
                ExprNode::Eq_(a, b) => match self.target {
                    Target::NumPy => self.np_relation("equal", &child(a)?, &child(b)?),
                    _ => self.relation("==", &child(a)?, &child(b)?),
                },
                ExprNode::Ne(a, b) => match self.target {
                    Target::NumPy => self.np_relation("not_equal", &child(a)?, &child(b)?),
                    _ => self.relation("!=", &child(a)?, &child(b)?),
                },
                ExprNode::And(children) => self.connective(
                    self.target.and_op(),
                    "logical_and",
                    PREC_AND,
                    children,
                    &cache,
                )?,
                ExprNode::Or(children) => {
                    self.connective(self.target.or_op(), "logical_or", PREC_OR, children, &cache)?
                }
                ExprNode::Not(a) => {
                    let r = child(a)?;
                    match self.target {
                        Target::Python => {
                            Rendered::new(format!("not {}", r.at(PREC_REL + 1)), PREC_NOT)
                        }
                        Target::NumPy => self.call("numpy.logical_not", &[&r]),
                        Target::Julia => Rendered::new(format!("!{}", r.at(PREC_ATOM)), PREC_NEG),
                    }
                }

                // Distributional: pointwise value is 0 (as in the C backend).
                ExprNode::DiracDelta(_) => Rendered::atom("0.0"),

                // Every remaining one-argument function goes through the
                // name table; anything without a spelling is refused.
                ExprNode::Sin(a)
                | ExprNode::Cos(a)
                | ExprNode::Tan(a)
                | ExprNode::Asin(a)
                | ExprNode::Acos(a)
                | ExprNode::Atan(a)
                | ExprNode::Sinh(a)
                | ExprNode::Cosh(a)
                | ExprNode::Tanh(a)
                | ExprNode::Asinh(a)
                | ExprNode::Acosh(a)
                | ExprNode::Atanh(a)
                | ExprNode::Exp(a)
                | ExprNode::Ln(a)
                | ExprNode::Abs(a)
                | ExprNode::Floor(a)
                | ExprNode::Ceiling(a)
                | ExprNode::Sign(a)
                | ExprNode::Gamma(a)
                | ExprNode::LogGamma(a)
                | ExprNode::Erf(a)
                | ExprNode::Erfc(a)
                | ExprNode::Factorial(a) => {
                    let name = self
                        .target
                        .unary(node)
                        .ok_or_else(|| self.unsupported(&describe(node)))?;
                    self.call(name, &[&child(a)?])
                }
                ExprNode::Apply(sid, _) => {
                    return Err(
                        self.unsupported(&format!("the function `{}`", arena.symbol_name(*sid)))
                    );
                }
                other => return Err(self.unsupported(&describe(other))),
            };
            cache.insert(id, rendered);
        }
        self.cached(&cache, root)
    }
}

/// Short description of a node kind for error messages.
fn describe(node: &ExprNode) -> String {
    let dbg = format!("{node:?}");
    dbg.split(['(', ' ']).next().unwrap_or("node").to_string()
}

// ═══════════════════════════════════════════════════════════════════════════
// Entry points
// ═══════════════════════════════════════════════════════════════════════════

/// The expression alone, every symbol printed by name.
pub(crate) fn to_expr_code(
    arena: &Arena,
    expr: ExprId,
    target: Target,
) -> Result<String, SymplexError> {
    let slots = FxHashMap::default();
    let em = Emitter {
        arena,
        target,
        params: None,
        cse_slots: &slots,
    };
    Ok(em.render(expr)?.text)
}

/// A function definition with CSE temporaries.
pub(crate) fn to_fn_code(
    arena: &mut Arena,
    expr: ExprId,
    name: &str,
    args: &[&str],
    target: Target,
) -> Result<String, SymplexError> {
    let cse = crate::output::cse::cse(arena, expr);
    let mut cse_slots: FxHashMap<ExprId, usize> = FxHashMap::default();
    for (i, (name_id, _)) in cse.bindings.iter().enumerate() {
        cse_slots.insert(*name_id, i);
    }
    let em = Emitter {
        arena,
        target,
        params: Some(args),
        cse_slots: &cse_slots,
    };
    let mut lines: Vec<String> = Vec::new();
    for (i, (_, value)) in cse.bindings.iter().enumerate() {
        lines.push(format!("    t{i} = {}", em.render(*value)?.text));
    }
    let result = em.render(cse.expr)?.text;
    let params = args.join(", ");
    let mut out = String::new();
    match target {
        Target::Python | Target::NumPy => {
            out.push_str(&format!("def {name}({params}):\n"));
            for l in &lines {
                out.push_str(l);
                out.push('\n');
            }
            out.push_str(&format!("    return {result}\n"));
        }
        Target::Julia => {
            out.push_str(&format!("function {name}({params})\n"));
            for l in &lines {
                out.push_str(l);
                out.push('\n');
            }
            out.push_str(&format!("    return {result}\nend\n"));
        }
    }
    Ok(out)
}

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

impl<S: Sort> Expr<S> {
    /// A Python 3 expression using the `math` module (SymPy: `pycode`).
    ///
    /// Integer powers print as `x**2`, rationals as `(1/2)`, `x^(1/2)` as
    /// `math.sqrt(x)`, constants as `math.pi`/`math.e`; relations and
    /// connectives ([`BoolEx`](crate::api::expr::BoolEx)) as `x > 0`,
    /// `and`, `or`, `not`; piecewise as `(v if c else …)`.  Symbols print by
    /// name; the caller needs `import math`.  See
    /// [`codegen`](crate::codegen) for the function table.
    ///
    /// # Errors
    ///
    /// [`SymplexError::NotImplemented`] for a node Python's `math` module
    /// cannot express (Bessel functions, `digamma`, `LambertW`, `zeta`,
    /// unevaluated integrals, sets, `I`) — never a silently wrong formula.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// assert_eq!((&x.sin().powi(2) + &x.exp()).to_python().unwrap(), "math.sin(x)**2 + math.exp(x)");
    /// assert_eq!((&x.powi(2) + 1).to_python().unwrap(), "x**2 + 1");
    /// assert_eq!((&x / 2).to_python().unwrap(), "x/2");
    /// assert_eq!((&x * 2 / 3).to_python().unwrap(), "2*x/3");
    /// assert_eq!((&x + 1).sqrt().to_python().unwrap(), "math.sqrt(x + 1)");
    /// assert_eq!((-&x * &y).to_python().unwrap(), "-x*y");
    /// assert_eq!((&x - &y).powi(3).to_python().unwrap(), "(x - y)**3");
    /// assert_eq!(x.gt(&ctx.int(0)).and(&x.lt(&ctx.int(1))).to_python().unwrap(), "x > 0 and 1 > x");
    /// assert!(x.bessel_j(&ctx.int(0)).to_python().is_err());
    /// ```
    pub fn to_python(&self) -> Result<String, SymplexError> {
        let inner = self.inner.read();
        to_expr_code(&inner.arena, self.raw_id(), Target::Python)
    }

    /// A vectorised Python expression using `numpy.` (SymPy:
    /// `NumPyPrinter().doprint`).
    ///
    /// Same layout as [`to_python`](Self::to_python) with `numpy.sin`,
    /// `numpy.sqrt`, `numpy.pi`, `numpy.greater`, `numpy.logical_and`,
    /// `numpy.select` for piecewise, `numpy.minimum`/`numpy.maximum`.  The
    /// caller needs `import numpy`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::NotImplemented`] for nodes NumPy itself lacks
    /// (`gamma`, `erf`, `factorial` need SciPy) or that have no numerical
    /// meaning.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!(x.sin().to_numpy().unwrap(), "numpy.sin(x)");
    /// assert_eq!((&x.sin().powi(2) + &x.exp()).to_numpy().unwrap(), "numpy.sin(x)**2 + numpy.exp(x)");
    /// assert_eq!((&x * ctx.pi()).sqrt().to_numpy().unwrap(), "numpy.sqrt(x*numpy.pi)");
    /// assert_eq!(x.gt(&ctx.int(0)).to_numpy().unwrap(), "numpy.greater(x, 0)");
    /// assert!(x.gamma().to_numpy().is_err());
    /// ```
    pub fn to_numpy(&self) -> Result<String, SymplexError> {
        let inner = self.inner.read();
        to_expr_code(&inner.arena, self.raw_id(), Target::NumPy)
    }

    /// A Julia expression (SymPy: `julia_code`).
    ///
    /// `sin`, `exp`, `sqrt`, `cbrt`, `abs`, `^` for powers, `pi`, `ℯ`,
    /// `Inf`, `NaN`, `&&`/`||`/`!`, `(c ? v : …)` for piecewise.  Base
    /// Julia only: `gamma`/`erf` (SpecialFunctions.jl) are refused.
    ///
    /// # Errors
    ///
    /// [`SymplexError::NotImplemented`] for nodes base Julia cannot express.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!((&x.sin().powi(2) + &x.exp()).to_julia().unwrap(), "sin(x)^2 + exp(x)");
    /// assert_eq!((&x.powi(2) + ctx.pi()).to_julia().unwrap(), "x^2 + pi");
    /// assert_eq!((ctx.pi() * &x / 2).to_julia().unwrap(), "x*pi/2");
    /// assert_eq!((ctx.e() * &x).to_julia().unwrap(), "x*ℯ");
    /// ```
    pub fn to_julia(&self) -> Result<String, SymplexError> {
        let inner = self.inner.read();
        to_expr_code(&inner.arena, self.raw_id(), Target::Julia)
    }

    /// A Python function `def name(args): …` with common subexpressions
    /// hoisted into `t0`, `t1`, … (SymPy: `pycode` + `cse`).
    ///
    /// # Errors
    ///
    /// [`SymplexError::FreeSymbol`] for a symbol not in `args`;
    /// [`SymplexError::NotImplemented`] as for [`to_python`](Self::to_python).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let f = &x.sin().powi(2) + &x.sin() * &y;
    /// assert_eq!(
    ///     f.to_python_fn("f", &["x", "y"]).unwrap(),
    ///     "def f(x, y):\n    t0 = math.sin(x)\n    return t0**2 + t0*y\n"
    /// );
    /// assert!(matches!(f.to_python_fn("f", &["x"]), Err(SymplexError::FreeSymbol { .. })));
    /// ```
    pub fn to_python_fn(&self, name: &str, args: &[&str]) -> Result<String, SymplexError> {
        let mut guard = self.inner.write();
        to_fn_code(&mut guard.arena, self.raw_id(), name, args, Target::Python)
    }

    /// A NumPy function `def name(args): …` with CSE temporaries; see
    /// [`to_numpy`](Self::to_numpy).
    ///
    /// # Errors
    ///
    /// [`SymplexError::FreeSymbol`] for a symbol not in `args`;
    /// [`SymplexError::NotImplemented`] as for [`to_numpy`](Self::to_numpy).
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!(
    ///     x.exp().to_numpy_fn("g", &["x"]).unwrap(),
    ///     "def g(x):\n    return numpy.exp(x)\n"
    /// );
    /// ```
    pub fn to_numpy_fn(&self, name: &str, args: &[&str]) -> Result<String, SymplexError> {
        let mut guard = self.inner.write();
        to_fn_code(&mut guard.arena, self.raw_id(), name, args, Target::NumPy)
    }

    /// A Julia function `function name(args) … end` with CSE temporaries;
    /// see [`to_julia`](Self::to_julia).
    ///
    /// # Errors
    ///
    /// [`SymplexError::FreeSymbol`] for a symbol not in `args`;
    /// [`SymplexError::NotImplemented`] as for [`to_julia`](Self::to_julia).
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!(
    ///     (&x.powi(2) + 1).to_julia_fn("h", &["x"]).unwrap(),
    ///     "function h(x)\n    return x^2 + 1\nend\n"
    /// );
    /// ```
    pub fn to_julia_fn(&self, name: &str, args: &[&str]) -> Result<String, SymplexError> {
        let mut guard = self.inner.write();
        to_fn_code(&mut guard.arena, self.raw_id(), name, args, Target::Julia)
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    fn py(s: &str) -> String {
        let ctx = Context::new();
        ctx.parse(s).unwrap().to_python().unwrap()
    }

    fn np(s: &str) -> String {
        let ctx = Context::new();
        ctx.parse(s).unwrap().to_numpy().unwrap()
    }

    fn jl(s: &str) -> String {
        let ctx = Context::new();
        ctx.parse(s).unwrap().to_julia().unwrap()
    }

    #[test]
    fn python_arithmetic() {
        assert_eq!(py("x^2 + 1"), "x**2 + 1");
        assert_eq!(py("x/2"), "x/2");
        assert_eq!(py("1/x"), "1/x");
        assert_eq!(py("2/3"), "(2/3)");
        assert_eq!(py("x/y"), "x/y");
        assert_eq!(py("x/(y*z)"), "x/(y*z)");
        assert_eq!(py("x/(y+1)"), "x/(y + 1)");
        assert_eq!(py("-x"), "-x");
        assert_eq!(py("-x^2"), "-x**2");
        assert_eq!(py("(-x)^y"), "(-x)**y");
        assert_eq!(py("x - y"), "x - y");
        assert_eq!(py("1 - x/2"), "-x/2 + 1");
        assert_eq!(py("x^(-2)"), "x**(-2)");
        assert_eq!(py("x^(3/2)"), "x**(3/2)");
        assert_eq!(py("x^y"), "x**y");
        assert_eq!(py("(x+1)^2"), "(x + 1)**2");
        assert_eq!(py("2^x"), "2**x");
        assert_eq!(py("x^y^z"), "x**y**z");
        assert_eq!(py("(x^y)^z"), "(x**y)**z");
        assert_eq!(py("x^(y+1)"), "x**(y + 1)");
        assert_eq!(py("e^x"), "math.exp(x)");
        assert_eq!(py("pi*e"), "math.pi*math.e");
        assert_eq!(py("inf"), "math.inf");
    }

    #[test]
    fn python_functions() {
        assert_eq!(py("sin(x)^2 + exp(x)"), "math.sin(x)**2 + math.exp(x)");
        assert_eq!(py("sqrt(x)"), "math.sqrt(x)");
        // 0.11.1: real cube root (`x**(1/3)` is complex for negative `x`).
        assert_eq!(py("cbrt(x)"), "math.copysign(abs(x)**(1/3), x)");
        assert_eq!(py("abs(x)"), "abs(x)");
        assert_eq!(py("floor(x) + ceil(y)"), "math.floor(x) + math.ceil(y)");
        assert_eq!(py("gamma(x) + erf(x)"), "math.gamma(x) + math.erf(x)");
        assert_eq!(py("loggamma(x)"), "math.lgamma(x)");
        assert_eq!(py("atan2(y, x)"), "math.atan2(y, x)");
        assert_eq!(py("min(x, y)"), "min(x, y)");
        // 0.11.1: `math.factorial` is integer-only; Γ(x + 1) matches Rust/C.
        assert_eq!(py("x!"), "math.gamma(x + 1)");
        assert_eq!(py("(x - y)!"), "math.gamma(x - y + 1)");
        assert_eq!(py("ln(x)"), "math.log(x)");
        assert_eq!(py("sign(x)"), "(0.0 if x == 0 else math.copysign(1, x))");
    }

    #[test]
    fn python_logic_and_piecewise() {
        let ctx = Context::new();
        let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
        let zero = ctx.int(0);
        assert_eq!(x.gt(&zero).to_python().unwrap(), "x > 0");
        assert_eq!(
            x.gt(&zero).and(&y.ge(&zero)).to_python().unwrap(),
            "x > 0 and y >= 0"
        );
        assert_eq!(
            x.gt(&zero)
                .and(&y.ge(&zero))
                .or(&x.eq_expr(&y))
                .to_python()
                .unwrap(),
            "x > 0 and y >= 0 or x == y"
        );
        assert_eq!(
            x.gt(&zero)
                .or(&y.ge(&zero))
                .and(&x.ne_expr(&y))
                .to_python()
                .unwrap(),
            "(x > 0 or y >= 0) and x != y"
        );
        assert_eq!(x.gt(&zero).not().to_python().unwrap(), "not (x > 0)");
        let pw = Ex::piecewise(&[(&x, &x.gt(&zero)), (&(-&x), &x.le(&zero))]);
        assert_eq!(
            pw.to_python().unwrap(),
            "(x if x > 0 else (-x if 0 >= x else math.nan))"
        );
        let pw2 = Ex::piecewise(&[(&x.powi(2), &x.lt(&zero)), (&x, &x.ge(&zero))]);
        assert_eq!(
            pw2.to_python().unwrap(),
            "(x**2 if 0 > x else (x if x >= 0 else math.nan))"
        );
    }

    #[test]
    fn numpy_and_julia() {
        assert_eq!(np("sin(x)"), "numpy.sin(x)");
        assert_eq!(np("x^2 + 1"), "x**2 + 1");
        assert_eq!(np("pi"), "numpy.pi");
        assert_eq!(np("cbrt(x)"), "numpy.cbrt(x)");
        assert_eq!(np("min(x, y, z)"), "numpy.minimum(numpy.minimum(x, y), z)");
        assert_eq!(np("sign(x)"), "numpy.sign(x)");
        assert_eq!(np("asin(x)"), "numpy.arcsin(x)");
        let ctx = Context::new();
        let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
        let zero = ctx.int(0);
        assert_eq!(
            x.gt(&zero).and(&y.ge(&zero)).to_numpy().unwrap(),
            "numpy.logical_and(numpy.greater(x, 0), numpy.greater_equal(y, 0))"
        );
        let pw = Ex::piecewise(&[(&x, &x.gt(&zero)), (&(-&x), &x.le(&zero))]);
        assert_eq!(
            pw.to_numpy().unwrap(),
            "numpy.select([numpy.greater(x, 0), numpy.greater_equal(0, x)], [x, -x], default=numpy.nan)"
        );
        assert!(ctx.parse("gamma(x)").unwrap().to_numpy().is_err());

        assert_eq!(jl("sin(x)^2 + exp(x)"), "sin(x)^2 + exp(x)");
        assert_eq!(jl("x^2 + pi"), "x^2 + pi");
        assert_eq!(jl("x/2"), "x/2");
        assert_eq!(jl("2/3*x"), "2*x/3");
        assert_eq!(jl("atan2(y, x)"), "atan(y, x)");
        assert_eq!(
            x.gt(&zero).and(&y.ge(&zero)).to_julia().unwrap(),
            "x > 0 && y >= 0"
        );
        assert_eq!(x.gt(&zero).not().to_julia().unwrap(), "!(x > 0)");
        assert_eq!(pw.to_julia().unwrap(), "(x > 0 ? x : (0 >= x ? -x : NaN))");
        assert!(ctx.parse("gamma(x)").unwrap().to_julia().is_err());
    }

    #[test]
    fn real_roots_match_compile_semantics() {
        // Odd denominators are real roots: sign(x)·|x|^(p/q) for odd p,
        // |x|^(p/q) for even p — the same rule as `compile()`, Rust and C.
        assert_eq!(py("x^(3/5)"), "math.copysign(abs(x)**(3/5), x)");
        assert_eq!(py("x^(2/5)"), "abs(x)**(2/5)");
        assert_eq!(py("x^(-1/3)"), "math.copysign(abs(x)**(-1/3), x)");
        assert_eq!(py("y*x^(-1/3)"), "y/math.copysign(abs(x)**(1/3), x)");
        assert_eq!(py("y/sqrt(x)"), "y/math.sqrt(x)");
        // Even denominators are left alone (complex for negative bases anyway).
        assert_eq!(py("x^(3/2)"), "x**(3/2)");
        assert_eq!(py("x^(1/4)"), "x**(1/4)");
        assert_eq!(np("cbrt(x)"), "numpy.cbrt(x)");
        assert_eq!(np("x^(3/5)"), "numpy.copysign(numpy.abs(x)**(3/5), x)");
        assert_eq!(np("x^(2/5)"), "numpy.abs(x)**(2/5)");
        assert_eq!(jl("cbrt(x)"), "cbrt(x)");
        assert_eq!(jl("x^(3/5)"), "copysign(abs(x)^(3/5), x)");
        assert_eq!(jl("x^(2/5)"), "abs(x)^(2/5)");
    }

    #[test]
    fn empty_min_max_and_single_factor_products() {
        use crate::base::arena::Arena;
        use crate::base::node::ExprNode;
        use crate::output::codegen::codegen_py::{Target, to_expr_code};
        // `min_of`/`max_of` never build an empty node, so go through the arena.
        let mut a = Arena::new();
        let empty_min = a.intern(ExprNode::Min(smallvec::smallvec![]));
        let empty_max = a.intern(ExprNode::Max(smallvec::smallvec![]));
        assert_eq!(
            to_expr_code(&a, empty_min, Target::Python).unwrap(),
            "math.inf"
        );
        assert_eq!(
            to_expr_code(&a, empty_max, Target::Python).unwrap(),
            "(-math.inf)"
        );
        assert_eq!(
            to_expr_code(&a, empty_min, Target::NumPy).unwrap(),
            "numpy.inf"
        );
        assert_eq!(
            to_expr_code(&a, empty_max, Target::Julia).unwrap(),
            "(-Inf)"
        );
        // A one-factor product keeps the factor's precedence: `Mul([x^2])`
        // as a power base must be `(x**2)**y`, not `x**2**y`.
        let x = a.symbol("x");
        let y = a.symbol("y");
        let two = a.int(2);
        let x2 = a.intern(ExprNode::Pow(x, two));
        let one_factor = a.intern(ExprNode::Mul(smallvec::smallvec![x2]));
        let p = a.intern(ExprNode::Pow(one_factor, y));
        assert_eq!(to_expr_code(&a, p, Target::Python).unwrap(), "(x**2)**y");
    }

    #[test]
    fn functions_with_cse_and_errors() {
        let ctx = Context::new();
        let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
        let f = &x.sin().powi(2) + &x.sin() * &y;
        assert_eq!(
            f.to_python_fn("f", &["x", "y"]).unwrap(),
            "def f(x, y):\n    t0 = math.sin(x)\n    return t0**2 + t0*y\n"
        );
        assert_eq!(
            f.to_julia_fn("f", &["x", "y"]).unwrap(),
            "function f(x, y)\n    t0 = sin(x)\n    return t0^2 + t0*y\nend\n"
        );
        assert!(matches!(
            f.to_python_fn("f", &["x"]),
            Err(SymplexError::FreeSymbol { .. })
        ));
        assert!(matches!(
            x.bessel_j(&ctx.int(0)).to_python(),
            Err(SymplexError::NotImplemented(_))
        ));
        assert!(matches!(
            ctx.parse("Integral(x, x)").unwrap().to_python(),
            Err(SymplexError::NotImplemented(_))
        ));
        assert!(matches!(
            ctx.parse("I").unwrap().to_python(),
            Err(SymplexError::NotImplemented(_))
        ));
    }
}
