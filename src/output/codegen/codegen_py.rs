//! Python, NumPy and Julia code generation from symbolic expressions (0.9.1).
//!
//! One table-driven printer serves three targets (SymPy: `pycode`,
//! `NumPyPrinter().doprint`, `julia_code`):
//!
//! | Target | Functions | Constants | Power | Relations / logic |
//! |--------|-----------|-----------|-------|-------------------|
//! | Python | `math.sin`, `math.exp`, `math.sqrt`, `math.gamma`, `math.erf`, … ; `abs`, `min`, `max` | `math.pi`, `math.e`, `math.inf`, `math.nan` | `**` | `>`, `and`, `or`, `not`; `(v if c else …)` |
//! | NumPy  | `numpy.sin`, `numpy.sqrt`, `numpy.floor`, `numpy.sign`, `numpy.minimum`, … | `numpy.pi`, `numpy.e`, `numpy.inf`, `numpy.nan` | `**` | `numpy.greater`, `numpy.logical_and`, `numpy.select` |
//! | Julia  | `sin`, `exp`, `sqrt`, `abs`, `floor`, `sign`, `min`, `max`, `atan(y, x)` | `pi`, `ℯ`, `Inf`, `NaN` | `^` | `>`, `&&`, `\|\|`, `!`; `(c ? v : …)` |
//!
//! Numbers are exact: integers print as integers, rationals as `(p/q)`
//! (a float division in all three languages).  Sums are printed in display
//! order with `-` for negated terms; `x^(-1)` factors become divisions.
//! The emitted code computes the value `evalf` gives, or NaN where that
//! value is not real, as `compile()` and the Rust and C back ends do.  A
//! non-integer rational power of a negative base is not real — an odd
//! denominator included (the principal `(-8)^(1/3)` is `1 + √3·i`): NaN —
//! `(lambda b: b**(3/2) if b >= 0 else math.nan)(x)` /
//! `numpy.power(x, (3/2))` / `(b -> b >= 0 ? b^(3/2) : NaN)(x)` (Python's
//! bare `(-1.0)**(3/2)` is a complex number, Julia's a `DomainError`).
//! Before 0.30 an odd denominator was the real root; that is `real_root`,
//! `sign(x)·|x|^(1/q)`.  `loggamma(x)` is SymPy's (not real for a negative
//! non-integer `x`): `(lambda b: math.lgamma(b) if b > 0 else math.inf if
//! b % 1 == 0 else math.nan)(x)`; `ln(abs(gamma(x)))` is `math.lgamma(x)`.
//! `min`/`max` propagate NaN like `compile()`: Python's builtins keep the
//! first operand of a comparison with NaN (`min(2, nan)` is `2`), so they
//! get the key `(v == v, v)` / `(v != v, v)` that ranks NaN first.
//! `x!` is `math.gamma(x + 1)` in Python
//! (`math.factorial` rejects non-integers).  Constants whose formula would
//! not evaluate to their real value (`abs(atanh(9))`) are printed as
//! literals, and non-real constants (`atanh(9)`) refused
//! (`fold_constants_for_emission` in `output/codegen.rs`).
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
use num_traits::{One, Signed, ToPrimitive};
use rustc_hash::FxHashMap;

use crate::api::expr::{Expr, Sort};
use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::libfn::LibFn;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
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
    /// Folded constants: symbol → value.
    folds: &'a FxHashMap<ExprId, f64>,
}

/// An `f64` literal in Python / Julia syntax (shortest round trip, with a
/// decimal point or exponent).
fn float_literal(v: f64) -> String {
    let s = format!("{v:?}");
    if s.contains('.') || s.contains('e') || s.contains("inf") || s.contains("NaN") {
        s
    } else {
        format!("{s}.0")
    }
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

    /// A literal standing for `r` in a float context: its correctly rounded
    /// `f64`, the target's infinity beyond the range.
    fn float_value(&self, r: &Q) -> Rendered {
        let v = super::rational_to_f64(r);
        let text = if v.is_infinite() {
            let inf = self
                .target
                .constant(&ExprNode::Infinity)
                .unwrap_or("math.inf");
            if v < 0.0 {
                format!("(-{inf})")
            } else {
                inf.to_string()
            }
        } else {
            float_literal(v)
        };
        if v < 0.0 {
            Rendered::new(text, PREC_NEG)
        } else {
            Rendered::atom(text)
        }
    }

    /// Whether the exact form of `r` cannot be used: an integer part past
    /// the `f64` range in Python and Julia (`x + 10**400` raises
    /// `OverflowError`), past 2⁵³ in NumPy (`numpy.sqrt(10**20 + 1)`, an
    /// integer beyond `int64`, raises `TypeError`); a quotient `(p/q)`
    /// that overflows.
    fn needs_float(&self, r: &Q) -> bool {
        let limit = match self.target {
            Target::NumPy => 53,
            _ => 1023,
        };
        if r.is_integer() {
            return r.numer().magnitude().bits() > limit;
        }
        super::rational_to_f64(r).is_infinite()
    }

    /// `b^e` (an exact rational) for a rational base and a negative integer
    /// exponent whose power `b^(−e)` is an integer the target cannot
    /// convert (see [`needs_float`](Self::needs_float)); `None` otherwise.
    /// A power beyond 2¹⁰⁰⁰⁰ bits is the value 0 or ∞ it rounds to.
    fn huge_constant_power(&self, base: ExprId, e: &Q) -> Option<Q> {
        let b = self.arena.as_num(base)?;
        if !e.is_integer()
            || b.is_negative()
            || !b.is_integer()
            || b.is_one()
            || b.numer().bits() == 0
        {
            return None;
        }
        let n = (-e.numer().clone()).to_u32().filter(|n| *n > 0)?;
        let bits = u64::from(n).saturating_mul(b.numer().bits());
        let limit = match self.target {
            Target::NumPy => 53,
            _ => 1023,
        };
        if bits <= limit + 1 {
            return None;
        }
        if bits > 10_000 {
            return Some(Q::from_integer(BigInt::from(0)));
        }
        let p = num_traits::pow::pow(b.numer().clone(), n as usize);
        Some(Q::new(BigInt::one(), p))
    }

    /// An exact literal: integers bare, rationals as `(p/q)`.
    fn number(&self, r: &Q) -> Rendered {
        if self.needs_float(r) {
            return self.float_value(r);
        }
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
        coeff: Option<&Q>,
        factors: &[ExprId],
        cache: &Cache,
    ) -> Result<Rendered, SymplexError> {
        let mut numer: Vec<String> = Vec::new();
        let mut denom: Vec<String> = Vec::new();
        // The lone factor when the coefficient contributes nothing: it is
        // returned as rendered, keeping its own precedence.
        let mut single: Option<Rendered> = None;
        if let Some(c) = coeff {
            let limit = match self.target {
                Target::NumPy => 53,
                _ => 1023,
            };
            if c.numer().magnitude().bits() > limit || c.denom().magnitude().bits() > limit {
                // `x/2**1074` raised OverflowError (the integer is not a
                // float): the coefficient's value instead.
                numer.push(self.float_value(c).at(PREC_MUL + 1));
            } else {
                if !c.numer().is_one() {
                    numer.push(c.numer().to_string());
                }
                if !c.denom().is_one() {
                    denom.push(c.denom().to_string());
                }
            }
        }
        for &f in factors {
            if let ExprNode::Pow(base, exp) = self.arena.node(f)
                && let Some(r) = self.arena.as_num(*exp)
                && r.is_negative()
            {
                if let Some(v) = self.huge_constant_power(*base, r) {
                    // `x/2**1074` raised OverflowError: the denominator is
                    // an integer past the float range.  Its value instead.
                    numer.push(self.float_value(&v).at(PREC_MUL + 1));
                    single = None;
                    continue;
                }
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
    /// `1/2` is `sqrt`; any other non-integer exponent is NaN for a
    /// negative (or NaN) base, whose principal power is not real — an odd
    /// denominator too, as in `compile()` and the Rust/C back ends (before
    /// 0.30 it was the real root, `copysign(abs(b)**(p/q), b)`; a bare
    /// `b**(1/3)` would be complex for negative `b` in Python).
    fn pow_rational(&self, b: &Rendered, r: &Q) -> Rendered {
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
        if r.is_integer() {
            if *r.numer() == -one {
                return Rendered::new(format!("1/{}", b.at(PREC_MUL + 1)), PREC_MUL);
            }
            if let Some(e) = super::odd_integer_beyond_f64(r) {
                // The exponent converts to an even float: keep the sign of
                // the odd power (`(-1.0)**(2**53 + 1)` was 1.0).
                let e = float_literal(e);
                return Rendered::atom(match self.target {
                    Target::Python => {
                        format!("(lambda b: math.copysign(abs(b)**{e}, b))({})", b.at(0))
                    }
                    Target::NumPy => format!(
                        "(lambda b: numpy.copysign(numpy.abs(b)**{e}, b))({})",
                        b.at(0)
                    ),
                    Target::Julia => format!("(b -> copysign(abs(b)^{e}, b))({})", b.at(0)),
                });
            }
            return Rendered::new(self.pow_text(b, &self.number(r)), PREC_POW);
        }
        // NaN for a negative (or NaN) base, the base evaluated once.
        let param = Rendered::atom("b");
        let body = self.pow_text(&param, &self.number(r));
        match self.target {
            Target::Python => Rendered::atom(format!(
                "(lambda b: {body} if b >= 0 else math.nan)({})",
                b.at(0)
            )),
            Target::NumPy => self.call("numpy.power", &[b, &self.number(r)]),
            Target::Julia => Rendered::atom(format!("(b -> b >= 0 ? {body} : NaN)({})", b.at(0))),
        }
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
        if self.target == Target::Python && !self.surely_nonnegative(base) {
            // Python's `(-2.0)**0.5` is a complex number; the power of a
            // negative base is real only for an integer exponent.
            return Ok(Rendered::atom(format!(
                "(lambda b, e: b**e if b >= 0 or e % 1 == 0 else math.nan)({}, {})",
                b.at(0),
                e.at(0)
            )));
        }
        Ok(Rendered::new(self.pow_text(&b, &e), PREC_POW))
    }

    /// `id` is non-negative by its form (a non-negative constant, `|u|`,
    /// `e^u`, `√u`, an even power, powers and products of such); NaN aside.
    fn surely_nonnegative(&self, id: ExprId) -> bool {
        let mut stack = vec![id];
        while let Some(cur) = stack.pop() {
            let ok = match self.arena.node(cur) {
                ExprNode::Num(n) => !self.arena.num(*n).is_negative(),
                ExprNode::Pi
                | ExprNode::E
                | ExprNode::GoldenRatio
                | ExprNode::Catalan
                | ExprNode::Infinity
                | ExprNode::Abs(_)
                | ExprNode::Exp(_)
                | ExprNode::Cosh(_) => true,
                ExprNode::Symbol(_) => self.folds.get(&cur).is_some_and(|v| *v >= 0.0),
                ExprNode::Pow(b, e) => {
                    // An even power or a square root; any power of a
                    // non-negative base.
                    let even_or_half = self.arena.as_num(*e).is_some_and(|r| {
                        (r.is_integer() && num_integer::Integer::is_even(r.numer()))
                            || (*r.numer() == BigInt::one() && *r.denom() == BigInt::from(2))
                    });
                    if !even_or_half {
                        stack.push(*b);
                    }
                    true
                }
                ExprNode::Mul(ch) => {
                    stack.extend(ch.iter().copied());
                    true
                }
                _ => false,
            };
            if !ok {
                return false;
            }
        }
        true
    }

    /// Whether the rendering of `id` repeats the rendering of a
    /// subexpression (`sign`, `Heaviside` and `im` test their argument
    /// several times): a node that repeats its own operand binds it with a
    /// `lambda` when this holds, so nesting stays linear (nested `sign`
    /// grew as 4ⁿ, 1.6 MB at depth 8).
    fn repeats_operand(&self, id: ExprId) -> bool {
        let mut stack = vec![id];
        let mut seen = rustc_hash::FxHashSet::default();
        while let Some(cur) = stack.pop() {
            if !seen.insert(cur) {
                continue;
            }
            let node = self.arena.node(cur);
            let dup = match node {
                ExprNode::Sign(_) | ExprNode::Im(_) => self.target == Target::Python,
                ExprNode::Heaviside(_) => self.target != Target::NumPy,
                _ => false,
            };
            if dup {
                return true;
            }
            node.for_each_child(|c| stack.push(c));
        }
        false
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

    /// The target's name for `expm1` / `log1p`.
    fn m1_name(&self, log: bool) -> &'static str {
        match (self.target, log) {
            (Target::Python, false) => "math.expm1",
            (Target::Python, true) => "math.log1p",
            (Target::NumPy, false) => "numpy.expm1",
            (Target::NumPy, true) => "numpy.log1p",
            (Target::Julia, false) => "expm1",
            (Target::Julia, true) => "log1p",
        }
    }

    fn is_exact_int(&self, id: ExprId, v: i64) -> bool {
        self.arena
            .as_num(id)
            .is_some_and(|r| r.is_integer() && *r.numer() == BigInt::from(v))
    }

    fn render_add(&self, children: &[ExprId], cache: &Cache) -> Result<Rendered, SymplexError> {
        if children.is_empty() {
            return Ok(Rendered::atom("0"));
        }
        let mut ordered: Vec<ExprId> = children.to_vec();
        ordered.sort_by_key(|a| display_sort_key(self.arena, *a));
        let mut text = String::new();
        // e^u − 1 [+ rest] is `expm1(u)`, as in the Rust and C back ends and
        // `compile()` (`math.exp(1e-20) - 1` is 0, its value 1e-20).
        let exp_at = ordered
            .iter()
            .position(|&c| matches!(self.arena.node(c), ExprNode::Exp(_)));
        let one_at = ordered.iter().position(|&c| self.is_exact_int(c, -1));
        if let (Some(ei), Some(oi)) = (exp_at, one_at)
            && let ExprNode::Exp(u) = self.arena.node(ordered[ei])
        {
            let m1 = self.call(self.m1_name(false), &[&self.cached(cache, *u)?]);
            text.push_str(&m1.text);
            for (i, &c) in ordered.iter().enumerate() {
                if i == ei || i == oi {
                    continue;
                }
                let (negative, body) = self.signed_term(c, cache)?;
                text.push_str(if negative { " - " } else { " + " });
                text.push_str(&body.at(PREC_ADD + 1));
            }
            let prec = if ordered.len() == 2 {
                PREC_ATOM
            } else {
                PREC_ADD
            };
            return Ok(Rendered::new(text, prec));
        }
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
                    } else if let Some(&value) = self.folds.get(&id) {
                        let text = float_literal(value);
                        if value < 0.0 {
                            Rendered::new(text, PREC_NEG)
                        } else {
                            Rendered::atom(text)
                        }
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

                // A NaN argument gives NaN (`copysign(1, nan)` is ±1 and the
                // Heaviside chains ended in 1.0), as in `compile()`.
                ExprNode::Sign(a) if self.target == Target::Python => {
                    let r = child(a)?;
                    if self.repeats_operand(*a) {
                        Rendered::atom(format!(
                            "(lambda a: 0.0 if a == 0 else math.copysign(1, a) if a == a else math.nan)({})",
                            r.at(0)
                        ))
                    } else {
                        Rendered::atom(format!(
                            "(0.0 if {a} == 0 else math.copysign(1, {b}) if {a} == {a} else math.nan)",
                            a = r.at(PREC_ADD),
                            b = r.at(0)
                        ))
                    }
                }
                ExprNode::Heaviside(a) => {
                    let r = child(a)?;
                    let bind = self.repeats_operand(*a);
                    match self.target {
                        Target::Python if bind => Rendered::atom(format!(
                            "(lambda a: 0.0 if a < 0 else (0.5 if a == 0 else (1.0 if a > 0 else math.nan)))({})",
                            r.at(0)
                        )),
                        Target::Python => Rendered::atom(format!(
                            "(0.0 if {a} < 0 else (0.5 if {a} == 0 else (1.0 if {a} > 0 else math.nan)))",
                            a = r.at(PREC_ADD)
                        )),
                        Target::NumPy => {
                            self.call("numpy.heaviside", &[&r, &Rendered::atom("0.5")])
                        }
                        Target::Julia if bind => Rendered::atom(format!(
                            "(a -> a < 0 ? 0.0 : (a == 0 ? 0.5 : (a > 0 ? 1.0 : NaN)))({})",
                            r.at(0)
                        )),
                        Target::Julia => Rendered::atom(format!(
                            "({a} < 0 ? 0.0 : ({a} == 0 ? 0.5 : ({a} > 0 ? 1.0 : NaN)))",
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
                // SymPy's `loggamma`: `math.lgamma` (which is ln|Γ|) for
                // b > 0, +∞ at the poles (`loggamma(-3) = oo`; `math.lgamma`
                // raises there), NaN elsewhere left of 0, where the value
                // ln|Γ(b)| − iπ⌈−b⌉ is not real.  NumPy and base Julia have no
                // log-gamma: refused below.
                ExprNode::LogGamma(a) if self.target == Target::Python => {
                    let r = child(a)?;
                    Rendered::atom(format!(
                        "(lambda b: math.lgamma(b) if b > 0 else math.inf if b % 1 == 0 else math.nan)({})",
                        r.at(0)
                    ))
                }
                // Every value is real (NaN standing for one that is not): `re`
                // is the value, `im` is 0, as in `compile()` (`real_root` of a
                // symbol not known to be real is a `Piecewise` on `im(x) = 0`).
                ExprNode::Re(a) => child(a)?,
                ExprNode::Im(a) => {
                    let r = child(a)?;
                    match self.target {
                        Target::Python if self.repeats_operand(*a) => Rendered::atom(format!(
                            "(lambda a: 0.0 if a == a else math.nan)({})",
                            r.at(0)
                        )),
                        Target::Python => Rendered::atom(format!(
                            "(0.0 if {a} == {a} else math.nan)",
                            a = r.at(PREC_ADD)
                        )),
                        Target::NumPy => Rendered::atom(format!(
                            "numpy.where(numpy.isnan({}), numpy.nan, 0.0)",
                            r.at(0)
                        )),
                        Target::Julia => {
                            Rendered::atom(format!("(isnan({}) ? NaN : 0.0)", r.at(0)))
                        }
                    }
                }
                // ln(1 + u) is `log1p(u)` (see `render_add`).
                ExprNode::Ln(a)
                    if matches!(self.arena.node(*a), ExprNode::Add(ch)
                        if ch.len() == 2 && ch.iter().any(|&c| self.is_exact_int(c, 1))) =>
                {
                    let ExprNode::Add(ch) = self.arena.node(*a) else {
                        return Err(self.unsupported("log1p"));
                    };
                    let u = if self.is_exact_int(ch[0], 1) {
                        ch[1]
                    } else {
                        ch[0]
                    };
                    self.call(self.m1_name(true), &[&child(&u)?])
                }
                // ln|Γ(u)| is `math.lgamma(u)`, which does not overflow.
                ExprNode::Ln(a)
                    if self.target == Target::Python
                        && crate::output::codegen::abs_gamma_arg(arena, *a).is_some() =>
                {
                    let u = crate::output::codegen::abs_gamma_arg(arena, *a).unwrap_or(*a);
                    self.call("math.lgamma", &[&child(&u)?])
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
                    } else if self.target == Target::Python && parts.len() > 1 {
                        let list: Vec<String> = parts.iter().map(|p| p.at(0)).collect();
                        let (f, key) = if is_min {
                            ("min", "(v == v, v)")
                        } else {
                            ("max", "(v != v, v)")
                        };
                        Rendered::atom(format!("{f}({}, key=lambda v: {key})", list.join(", ")))
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
                // Library special functions are outside `math`/`numpy`/base
                // Julia; the error names the SciPy routine that computes the
                // function, when there is one.
                ExprNode::Apply(sid, _) => {
                    let name = arena.symbol_name(*sid);
                    let spelling = arena.lib_fn(*sid).and_then(scipy_spelling);
                    return Err(match spelling {
                        Some(scipy) => {
                            self.unsupported(&format!("the function `{name}` (SciPy: `{scipy}`)"))
                        }
                        None => self.unsupported(&format!("the function `{name}`")),
                    });
                }
                other => return Err(self.unsupported(&describe(other))),
            };
            cache.insert(id, rendered);
        }
        self.cached(&cache, root)
    }
}

/// Short description of a node kind for error messages.
use crate::output::common::describe;

/// The `scipy.special` routine with the value of a library function, written
/// as a call template over the function's own argument names (SciPy 1.13
/// documentation).  `None` when SciPy has no routine (the integer sequences,
/// `polylog`, `dirichlet_eta`, `elliptic_pi`) or only a differently shaped
/// one (`bernoulli`/`euler` return the whole sequence; `lambertw` is
/// complex-valued).
///
/// The Python back ends do not emit these — `to_python` is `math` only and
/// `to_numpy` is `numpy` only, and both are pinned to refuse every `Apply`
/// node — but the refusal names the routine.  Exhaustive over [`LibFn`].
pub(crate) fn scipy_spelling(f: LibFn) -> Option<&'static str> {
    Some(match f {
        LibFn::Factorial2 => "scipy.special.factorial2(n)",
        LibFn::RisingFactorial => "scipy.special.poch(x, n)",
        // x^(n) = Γ(x+1)/Γ(x−n+1) = (x−n+1)_n.
        LibFn::FallingFactorial => "scipy.special.poch(x - n + 1, n)",
        LibFn::Stirling2 => "scipy.special.stirling2(n, k)",
        LibFn::BesselJ => "scipy.special.jv(order, x)",
        LibFn::BesselY => "scipy.special.yv(order, x)",
        LibFn::BesselI => "scipy.special.iv(order, x)",
        LibFn::BesselK => "scipy.special.kv(order, x)",
        LibFn::Legendre => "scipy.special.eval_legendre(n, x)",
        LibFn::ChebyshevT => "scipy.special.eval_chebyt(n, x)",
        LibFn::ChebyshevU => "scipy.special.eval_chebyu(n, x)",
        LibFn::Hermite => "scipy.special.eval_hermite(n, x)",
        LibFn::Laguerre => "scipy.special.eval_laguerre(n, x)",
        LibFn::Erfi => "scipy.special.erfi(x)",
        LibFn::ErfInv => "scipy.special.erfinv(y)",
        LibFn::ErfcInv => "scipy.special.erfcinv(y)",
        LibFn::ExpInt => "scipy.special.expn(n, x)",
        // `shichi` returns the pair (Shi, Chi).
        LibFn::Shi => "scipy.special.shichi(x)[0]",
        LibFn::Chi => "scipy.special.shichi(x)[1]",
        // `fresnel` returns the pair (S, C).
        LibFn::FresnelS => "scipy.special.fresnel(x)[0]",
        LibFn::FresnelC => "scipy.special.fresnel(x)[1]",
        // `gammainc`/`gammaincc` are the regularised P and Q.
        LibFn::LowerGamma => "scipy.special.gamma(s) * scipy.special.gammainc(s, x)",
        LibFn::UpperGamma => "scipy.special.gamma(s) * scipy.special.gammaincc(s, x)",
        // `airy` returns (Ai, Ai', Bi, Bi').
        LibFn::AiryAi => "scipy.special.airy(x)[0]",
        LibFn::AiryAiPrime => "scipy.special.airy(x)[1]",
        LibFn::AiryBi => "scipy.special.airy(x)[2]",
        LibFn::AiryBiPrime => "scipy.special.airy(x)[3]",
        LibFn::EllipticK => "scipy.special.ellipk(m)",
        LibFn::EllipticE => "scipy.special.ellipe(m)",
        LibFn::EllipticF => "scipy.special.ellipkinc(phi, m)",
        LibFn::Gegenbauer => "scipy.special.eval_gegenbauer(n, alpha, x)",
        LibFn::Jacobi => "scipy.special.eval_jacobi(n, alpha, beta, x)",
        // Both use the Condon–Shortley phase; SciPy takes the order first.
        LibFn::AssocLegendre => "scipy.special.lpmv(m, n, x)",
        LibFn::AssocLaguerre => "scipy.special.eval_genlaguerre(n, alpha, x)",
        // SciPy's `betainc` is the regularised I_x(a, b) of one bound.
        LibFn::BetaInc => {
            "scipy.special.beta(a, b) * (scipy.special.betainc(a, b, x2) - scipy.special.betainc(a, b, x1))"
        }
        LibFn::BetaIncRegularized => {
            "scipy.special.betainc(a, b, x2) - scipy.special.betainc(a, b, x1)"
        }
        LibFn::Subfactorial
        | LibFn::Fibonacci
        | LibFn::Lucas
        | LibFn::Bernoulli
        | LibFn::Harmonic
        | LibFn::Catalan
        | LibFn::Bell
        | LibFn::EulerNumber
        | LibFn::Stirling1
        | LibFn::PartitionCount
        | LibFn::LambertW
        | LibFn::PolyLog
        | LibFn::DirichletEta
        | LibFn::EllipticPi => return None,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Entry points
// ═══════════════════════════════════════════════════════════════════════════

/// The expression alone, every symbol printed by name.
pub(crate) fn to_expr_code(
    arena: &mut Arena,
    expr: ExprId,
    target: Target,
) -> Result<String, SymplexError> {
    let folds = super::fold_constants_for_emission(arena, expr, target.name())?;
    let slots = FxHashMap::default();
    let em = Emitter {
        arena,
        target,
        params: None,
        cse_slots: &slots,
        folds: &folds.values,
    };
    Ok(em.render(folds.expr)?.text)
}

/// A function definition with CSE temporaries.
pub(crate) fn to_fn_code(
    arena: &mut Arena,
    expr: ExprId,
    name: &str,
    args: &[&str],
    target: Target,
) -> Result<String, SymplexError> {
    let fn_name = match target {
        Target::Python => "to_python_fn",
        Target::NumPy => "to_numpy_fn",
        Target::Julia => "to_julia_fn",
    };
    let folds = super::fold_constants_for_emission(arena, expr, fn_name)?;
    // Python and Julia raise on a domain error (`math.sqrt(-1)`): a
    // temporary is only hoisted out of a subexpression evaluated on every
    // path, not out of a piecewise branch (`t0 = math.sqrt(x)` before
    // `(t0*y + t0 if x > 0 else y)` raised for x < 0).  NumPy evaluates
    // every branch anyway.
    let cse = match target {
        Target::NumPy => crate::output::cse::cse(arena, folds.expr),
        Target::Python | Target::Julia => crate::output::cse::cse_unconditional(arena, folds.expr),
    };
    let mut cse_slots: FxHashMap<ExprId, usize> = FxHashMap::default();
    for (i, (name_id, _)) in cse.bindings.iter().enumerate() {
        cse_slots.insert(*name_id, i);
    }
    let em = Emitter {
        arena,
        target,
        params: Some(args),
        cse_slots: &cse_slots,
        folds: &folds.values,
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
        let mut inner = self.inner.write();
        to_expr_code(&mut inner.arena, self.raw_id(), Target::Python)
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
        let mut inner = self.inner.write();
        to_expr_code(&mut inner.arena, self.raw_id(), Target::NumPy)
    }

    /// A Julia expression (SymPy: `julia_code`).
    ///
    /// `sin`, `exp`, `sqrt`, `abs`, `^` for powers (NaN for a negative base
    /// and a non-integer exponent, where Julia's `^` throws), `pi`, `ℯ`,
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
        let mut inner = self.inner.write();
        to_expr_code(&mut inner.arena, self.raw_id(), Target::Julia)
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
        // A base that may be negative: Python's `(-2.0)**0.5` is complex,
        // so a non-integer exponent gives NaN (it was `(-x)**y`).
        assert_eq!(
            py("(-x)^y"),
            "(lambda b, e: b**e if b >= 0 or e % 1 == 0 else math.nan)(-x, y)"
        );
        assert_eq!(py("(x^2)^y"), "(x**2)**y");
        assert_eq!(py("x - y"), "x - y");
        assert_eq!(py("1 - x/2"), "-x/2 + 1");
        assert_eq!(py("x^(-2)"), "x**(-2)");
        // An even denominator: NaN for a negative base (was `x**(3/2)`,
        // complex in Python), as in the other f64 back ends.
        assert_eq!(
            py("x^(3/2)"),
            "(lambda b: b**(3/2) if b >= 0 else math.nan)(x)"
        );
        // A general power is NaN for a negative base and a non-integer
        // exponent (Python's `(-2.0)**0.5` is complex; it was `x**y`); a
        // base of known sign keeps the operator.
        assert_eq!(
            py("x^y"),
            "(lambda b, e: b**e if b >= 0 or e % 1 == 0 else math.nan)(x, y)"
        );
        assert_eq!(py("(x+1)^2"), "(x + 1)**2");
        assert_eq!(py("2^x"), "2**x");
        assert_eq!(py("abs(x)^abs(y)^z"), "abs(x)**abs(y)**z");
        assert_eq!(py("(abs(x)^y)^z"), "(abs(x)**y)**z");
        assert_eq!(py("abs(x)^(y+1)"), "abs(x)**(y + 1)");
        assert_eq!(py("e^x"), "math.exp(x)");
        assert_eq!(py("pi*e"), "math.pi*math.e");
        assert_eq!(py("inf"), "math.inf");
    }

    #[test]
    fn python_functions() {
        assert_eq!(py("sin(x)^2 + exp(x)"), "math.sin(x)**2 + math.exp(x)");
        assert_eq!(py("sqrt(x)"), "math.sqrt(x)");
        // 0.30: the principal cube root, NaN for negative `x` (0.11.1 to
        // 0.29 emitted the real root, `math.copysign(abs(x)**(1/3), x)`).
        assert_eq!(
            py("cbrt(x)"),
            "(lambda b: b**(1/3) if b >= 0 else math.nan)(x)"
        );
        assert_eq!(py("abs(x)"), "abs(x)");
        assert_eq!(py("floor(x) + ceil(y)"), "math.floor(x) + math.ceil(y)");
        assert_eq!(py("gamma(x) + erf(x)"), "math.gamma(x) + math.erf(x)");
        // 0.30: SymPy's `loggamma` is not real left of 0 (it was
        // `math.lgamma(x)`, ln|Γ|); `ln(abs(gamma(x)))` is `math.lgamma`.
        assert_eq!(
            py("loggamma(x)"),
            "(lambda b: math.lgamma(b) if b > 0 else math.inf if b % 1 == 0 else math.nan)(x)"
        );
        assert_eq!(py("ln(abs(gamma(x)))"), "math.lgamma(x)");
        assert_eq!(py("atan2(y, x)"), "math.atan2(y, x)");
        // NaN-propagating like `compile()` (`min(2, nan)` is 2 in Python).
        assert_eq!(py("min(x, y)"), "min(x, y, key=lambda v: (v == v, v))");
        assert_eq!(py("max(x, y)"), "max(x, y, key=lambda v: (v != v, v))");
        // 0.11.1: `math.factorial` is integer-only; Γ(x + 1) matches Rust/C.
        assert_eq!(py("x!"), "math.gamma(x + 1)");
        assert_eq!(py("(x - y)!"), "math.gamma(x - y + 1)");
        assert_eq!(py("ln(x)"), "math.log(x)");
        // 0.30: NaN stays NaN (`math.copysign(1, nan)` is ±1).
        assert_eq!(
            py("sign(x)"),
            "(0.0 if x == 0 else math.copysign(1, x) if x == x else math.nan)"
        );
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
        assert_eq!(np("cbrt(x)"), "numpy.power(x, (1/3))");
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
    fn rational_powers_match_compile_semantics() {
        // 0.30: every non-integer rational power of a negative base is NaN,
        // odd denominators included (the principal value is not real), the
        // rule of `compile()`, Rust and C.  0.11.1 to 0.29 emitted the real
        // root: `math.copysign(abs(x)**(3/5), x)`, `abs(x)**(2/5)`; that is
        // `real_root`.
        let nan_unless_nonneg =
            |p: &str| format!("(lambda b: b**({p}) if b >= 0 else math.nan)(x)");
        assert_eq!(py("x^(3/5)"), nan_unless_nonneg("3/5"));
        assert_eq!(py("x^(2/5)"), nan_unless_nonneg("2/5"));
        assert_eq!(py("x^(-1/3)"), nan_unless_nonneg("-1/3"));
        assert_eq!(py("y*x^(-1/3)"), format!("y/{}", nan_unless_nonneg("1/3")));
        assert_eq!(py("y/sqrt(x)"), "y/math.sqrt(x)");
        // Even denominators: NaN for negative bases, as `compile()` (they
        // were left bare, complex in Python for a negative base).
        assert_eq!(
            py("x^(3/2)"),
            "(lambda b: b**(3/2) if b >= 0 else math.nan)(x)"
        );
        assert_eq!(
            py("x^(1/4)"),
            "(lambda b: b**(1/4) if b >= 0 else math.nan)(x)"
        );
        assert_eq!(np("x^(1/4)"), "numpy.power(x, (1/4))");
        assert_eq!(np("cbrt(x)"), "numpy.power(x, (1/3))");
        assert_eq!(np("x^(3/5)"), "numpy.power(x, (3/5))");
        assert_eq!(np("x^(2/5)"), "numpy.power(x, (2/5))");
        assert_eq!(jl("cbrt(x)"), "(b -> b >= 0 ? b^(1/3) : NaN)(x)");
        assert_eq!(jl("x^(3/5)"), "(b -> b >= 0 ? b^(3/5) : NaN)(x)");
        assert_eq!(jl("x^(2/5)"), "(b -> b >= 0 ? b^(2/5) : NaN)(x)");
        // The real root is `real_root`: sign(x)·|x|^(1/q), −2 at −8.
        let ctx = Context::new();
        let r = ctx
            .symbol_with("r", &[crate::base::assumptions::Assumption::Real])
            .unwrap();
        let code = r.real_root(3).unwrap().to_python().unwrap();
        assert!(code.contains("math.copysign(1, r)"), "{code}");
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
            to_expr_code(&mut a, empty_min, Target::Python).unwrap(),
            "math.inf"
        );
        assert_eq!(
            to_expr_code(&mut a, empty_max, Target::Python).unwrap(),
            "(-math.inf)"
        );
        assert_eq!(
            to_expr_code(&mut a, empty_min, Target::NumPy).unwrap(),
            "numpy.inf"
        );
        assert_eq!(
            to_expr_code(&mut a, empty_max, Target::Julia).unwrap(),
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
        assert_eq!(
            to_expr_code(&mut a, p, Target::Python).unwrap(),
            "(x**2)**y"
        );
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
