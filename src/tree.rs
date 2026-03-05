//! Serializable expression tree for interchange.
//!
//! [`ExprTree`] is a standalone, self-contained representation of a
//! symbolic expression that can be serialized to JSON (or any serde
//! format) and deserialized back. It is the primary machine-readable
//! output format for symplex.
//!
//! # Round-trip
//!
//! ```
//! use symplex::prelude::*;
//!
//! let ctx = Context::new();
//! let x = ctx.symbol("x");
//! let expr = &x.powi(2) + &x + 1;
//!
//! // Serialize to tree
//! let tree = expr.to_tree();
//!
//! // Serialize to JSON
//! let json = serde_json::to_string(&tree).unwrap();
//!
//! // Deserialize back
//! let tree2: symplex::tree::ExprTree = serde_json::from_str(&json).unwrap();
//!
//! // Convert back to Ex in the same context
//! let expr2 = ctx.from_tree(&tree2);
//! assert_eq!(format!("{expr}"), format!("{expr2}"));
//! ```

use num_bigint::BigInt;
use num_rational::Ratio;
use serde::{Deserialize, Serialize};

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode};

/// A serializable expression tree.
///
/// This is a standalone tree (not arena-indexed) that can be serialized
/// to JSON or any serde-supported format. Use [`Ex::to_tree()`] to
/// convert from an expression handle, and [`Context::from_tree()`] to
/// convert back.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ExprTree {
    /// A rational number with arbitrary-precision numerator and denominator.
    Num { numer: String, denom: String },
    /// A symbolic variable.
    Symbol { name: String },
    /// The constant π.
    Pi,
    /// Euler's number e.
    E,
    /// The imaginary unit i.
    ImaginaryUnit,
    /// Positive infinity.
    Infinity,
    /// Negative infinity.
    NegInfinity,
    /// Complex infinity (undirected).
    ComplexInfinity,
    /// Not-a-number.
    NaN,
    /// A sum of terms.
    Add { terms: Vec<ExprTree> },
    /// A product of factors.
    Mul { factors: Vec<ExprTree> },
    /// Exponentiation: base^exp.
    Pow {
        base: Box<ExprTree>,
        exp: Box<ExprTree>,
    },
    /// Negation: -inner.
    Neg { inner: Box<ExprTree> },
    /// Sine.
    Sin { arg: Box<ExprTree> },
    /// Cosine.
    Cos { arg: Box<ExprTree> },
    /// Tangent.
    Tan { arg: Box<ExprTree> },
    /// Natural exponential.
    Exp { arg: Box<ExprTree> },
    /// Natural logarithm.
    Ln { arg: Box<ExprTree> },
    /// Square root.
    Sqrt { arg: Box<ExprTree> },
    /// Absolute value.
    Abs { arg: Box<ExprTree> },
    /// Inverse sine.
    Asin { arg: Box<ExprTree> },
    /// Inverse cosine.
    Acos { arg: Box<ExprTree> },
    /// Inverse tangent.
    Atan { arg: Box<ExprTree> },
    /// Two-argument arctangent: atan2(y, x).
    Atan2 { y: Box<ExprTree>, x: Box<ExprTree> },
    /// Hyperbolic sine.
    Sinh { arg: Box<ExprTree> },
    /// Hyperbolic cosine.
    Cosh { arg: Box<ExprTree> },
    /// Hyperbolic tangent.
    Tanh { arg: Box<ExprTree> },
    /// Inverse hyperbolic sine.
    Asinh { arg: Box<ExprTree> },
    /// Inverse hyperbolic cosine.
    Acosh { arg: Box<ExprTree> },
    /// Inverse hyperbolic tangent.
    Atanh { arg: Box<ExprTree> },
    /// Sign function: 1 if positive, -1 if negative, 0 if zero.
    Sign { arg: Box<ExprTree> },
    /// Heaviside step function: H(x) = 0 for x<0, 1/2 for x=0, 1 for x>0.
    Heaviside { arg: Box<ExprTree> },
    /// Dirac delta distribution: δ(x) = 0 for x≠0, symbolic at x=0.
    DiracDelta { arg: Box<ExprTree> },
    /// Gamma function: Γ(x).
    Gamma { arg: Box<ExprTree> },
    /// Log-gamma function: ln(Γ(x)).
    LogGamma { arg: Box<ExprTree> },
    /// Digamma function: ψ(x) = Γ'(x)/Γ(x).
    Digamma { arg: Box<ExprTree> },
    /// Error function: erf(x).
    Erf { arg: Box<ExprTree> },
    /// Complementary error function: erfc(x) = 1 - erf(x).
    Erfc { arg: Box<ExprTree> },
    /// Beta function: B(a, b) = Γ(a)Γ(b)/Γ(a+b).
    Beta { a: Box<ExprTree>, b: Box<ExprTree> },
    /// Floor function: greatest integer <= x.
    Floor { arg: Box<ExprTree> },
    /// Ceiling function: least integer >= x.
    Ceiling { arg: Box<ExprTree> },
    /// N-ary minimum.
    Min { args: Vec<ExprTree> },
    /// N-ary maximum.
    Max { args: Vec<ExprTree> },
    /// Boolean true.
    BoolTrue,
    /// Boolean false.
    BoolFalse,
    /// Greater than: lhs > rhs.
    Gt {
        lhs: Box<ExprTree>,
        rhs: Box<ExprTree>,
    },
    /// Greater than or equal: lhs >= rhs.
    Ge {
        lhs: Box<ExprTree>,
        rhs: Box<ExprTree>,
    },
    /// Mathematical equality test: lhs == rhs.
    Eq_ {
        lhs: Box<ExprTree>,
        rhs: Box<ExprTree>,
    },
    /// Not equal: lhs != rhs.
    Ne {
        lhs: Box<ExprTree>,
        rhs: Box<ExprTree>,
    },
    /// Logical conjunction (n-ary).
    And { args: Vec<ExprTree> },
    /// Logical disjunction (n-ary).
    Or { args: Vec<ExprTree> },
    /// Logical negation.
    Not { arg: Box<ExprTree> },
    /// Piecewise function: list of (value, condition) pairs.
    Piecewise { pieces: Vec<(ExprTree, ExprTree)> },
    /// Application of a named function.
    Apply { name: String, args: Vec<ExprTree> },
    /// Formal derivative.
    Derivative {
        body: Box<ExprTree>,
        var: Box<ExprTree>,
    },
    /// Formal integral.
    Integral {
        body: Box<ExprTree>,
        var: Box<ExprTree>,
    },
    /// Symbolic summation: Sum(body, var, lower, upper).
    Sum {
        body: Box<ExprTree>,
        var: Box<ExprTree>,
        lower: Box<ExprTree>,
        upper: Box<ExprTree>,
    },
    /// Symbolic product: Product(body, var, lower, upper).
    Product_ {
        body: Box<ExprTree>,
        var: Box<ExprTree>,
        lower: Box<ExprTree>,
        upper: Box<ExprTree>,
    },
    /// The empty set ∅.
    EmptySet,
    /// The universal set.
    UniversalSet,
    /// A closed/open interval with flags encoding open/closed.
    /// Bits: 0x01 = left_open, 0x02 = right_open.
    Interval {
        start: Box<ExprTree>,
        end: Box<ExprTree>,
        flags: u8,
    },
    /// A finite set of elements {a, b, c, ...}.
    FiniteSet { elements: Vec<ExprTree> },
    /// Union of sets: A ∪ B ∪ C ∪ ...
    SetUnion { sets: Vec<ExprTree> },
    /// Intersection of sets: A ∩ B ∩ C ∩ ...
    SetIntersection { sets: Vec<ExprTree> },
    /// Set complement (relative): A \ B.
    SetComplement {
        set: Box<ExprTree>,
        universe: Box<ExprTree>,
    },
}

// ═══════════════════════════════════════════════════════════════════════════
// ExprId → ExprTree (serialization direction)
// ═══════════════════════════════════════════════════════════════════════════

/// Convert an arena expression to a standalone [`ExprTree`].
///
/// Uses exhaustive matching on [`ExprNode`] — adding a new variant
/// without handling it here is a compile error.
pub(crate) fn expr_to_tree(arena: &Arena, id: ExprId) -> ExprTree {
    match arena.node(id).clone() {
        ExprNode::Num(nid) => {
            let r = arena.num(nid);
            ExprTree::Num {
                numer: r.numer().to_string(),
                denom: r.denom().to_string(),
            }
        }
        ExprNode::Symbol(sid) => ExprTree::Symbol {
            name: arena.symbol_name(sid).to_owned(),
        },
        ExprNode::Pi => ExprTree::Pi,
        ExprNode::E => ExprTree::E,
        ExprNode::ImaginaryUnit => ExprTree::ImaginaryUnit,
        ExprNode::Infinity => ExprTree::Infinity,
        ExprNode::NegInfinity => ExprTree::NegInfinity,
        ExprNode::ComplexInfinity => ExprTree::ComplexInfinity,
        ExprNode::NaN => ExprTree::NaN,
        ExprNode::Add(children) => ExprTree::Add {
            terms: children.iter().map(|&c| expr_to_tree(arena, c)).collect(),
        },
        ExprNode::Mul(children) => ExprTree::Mul {
            factors: children.iter().map(|&c| expr_to_tree(arena, c)).collect(),
        },
        ExprNode::Pow(base, exp) => ExprTree::Pow {
            base: Box::new(expr_to_tree(arena, base)),
            exp: Box::new(expr_to_tree(arena, exp)),
        },
        ExprNode::Neg(inner) => ExprTree::Neg {
            inner: Box::new(expr_to_tree(arena, inner)),
        },
        ExprNode::Sin(x) => ExprTree::Sin {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Cos(x) => ExprTree::Cos {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Tan(x) => ExprTree::Tan {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Exp(x) => ExprTree::Exp {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Ln(x) => ExprTree::Ln {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Abs(x) => ExprTree::Abs {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Asin(x) => ExprTree::Asin {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Acos(x) => ExprTree::Acos {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Atan(x) => ExprTree::Atan {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Atan2(y, x) => ExprTree::Atan2 {
            y: Box::new(expr_to_tree(arena, y)),
            x: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Sinh(x) => ExprTree::Sinh {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Cosh(x) => ExprTree::Cosh {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Tanh(x) => ExprTree::Tanh {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Asinh(x) => ExprTree::Asinh {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Acosh(x) => ExprTree::Acosh {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Atanh(x) => ExprTree::Atanh {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Sign(x) => ExprTree::Sign {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Heaviside(x) => ExprTree::Heaviside {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::DiracDelta(x) => ExprTree::DiracDelta {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Gamma(x) => ExprTree::Gamma {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::LogGamma(x) => ExprTree::LogGamma {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Digamma(x) => ExprTree::Digamma {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Erf(x) => ExprTree::Erf {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Erfc(x) => ExprTree::Erfc {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Beta(a, b) => ExprTree::Beta {
            a: Box::new(expr_to_tree(arena, a)),
            b: Box::new(expr_to_tree(arena, b)),
        },
        ExprNode::Floor(x) => ExprTree::Floor {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Ceiling(x) => ExprTree::Ceiling {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Min(children) => ExprTree::Min {
            args: children.iter().map(|&c| expr_to_tree(arena, c)).collect(),
        },
        ExprNode::Max(children) => ExprTree::Max {
            args: children.iter().map(|&c| expr_to_tree(arena, c)).collect(),
        },
        ExprNode::Apply(sid, args) => ExprTree::Apply {
            name: arena.symbol_name(sid).to_owned(),
            args: args.iter().map(|&a| expr_to_tree(arena, a)).collect(),
        },
        ExprNode::Derivative(body, var) => ExprTree::Derivative {
            body: Box::new(expr_to_tree(arena, body)),
            var: Box::new(expr_to_tree(arena, var)),
        },
        ExprNode::Integral(body, var) => ExprTree::Integral {
            body: Box::new(expr_to_tree(arena, body)),
            var: Box::new(expr_to_tree(arena, var)),
        },
        ExprNode::Sum(body, var, lo, hi) => ExprTree::Sum {
            body: Box::new(expr_to_tree(arena, body)),
            var: Box::new(expr_to_tree(arena, var)),
            lower: Box::new(expr_to_tree(arena, lo)),
            upper: Box::new(expr_to_tree(arena, hi)),
        },
        ExprNode::Product_(body, var, lo, hi) => ExprTree::Product_ {
            body: Box::new(expr_to_tree(arena, body)),
            var: Box::new(expr_to_tree(arena, var)),
            lower: Box::new(expr_to_tree(arena, lo)),
            upper: Box::new(expr_to_tree(arena, hi)),
        },
        ExprNode::Factorial(x) => ExprTree::Apply {
            name: "factorial".to_owned(),
            args: vec![expr_to_tree(arena, x)],
        },
        ExprNode::Binomial(n, k) => ExprTree::Apply {
            name: "binomial".to_owned(),
            args: vec![expr_to_tree(arena, n), expr_to_tree(arena, k)],
        },
        ExprNode::BoolTrue => ExprTree::BoolTrue,
        ExprNode::BoolFalse => ExprTree::BoolFalse,
        ExprNode::Gt(a, b) => ExprTree::Gt {
            lhs: Box::new(expr_to_tree(arena, a)),
            rhs: Box::new(expr_to_tree(arena, b)),
        },
        ExprNode::Ge(a, b) => ExprTree::Ge {
            lhs: Box::new(expr_to_tree(arena, a)),
            rhs: Box::new(expr_to_tree(arena, b)),
        },
        ExprNode::Eq_(a, b) => ExprTree::Eq_ {
            lhs: Box::new(expr_to_tree(arena, a)),
            rhs: Box::new(expr_to_tree(arena, b)),
        },
        ExprNode::Ne(a, b) => ExprTree::Ne {
            lhs: Box::new(expr_to_tree(arena, a)),
            rhs: Box::new(expr_to_tree(arena, b)),
        },
        ExprNode::And(children) => ExprTree::And {
            args: children.iter().map(|&c| expr_to_tree(arena, c)).collect(),
        },
        ExprNode::Or(children) => ExprTree::Or {
            args: children.iter().map(|&c| expr_to_tree(arena, c)).collect(),
        },
        ExprNode::Not(x) => ExprTree::Not {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Piecewise(children) => ExprTree::Piecewise {
            pieces: children
                .iter()
                .map(|&(val, cond)| (expr_to_tree(arena, val), expr_to_tree(arena, cond)))
                .collect(),
        },
        ExprNode::EmptySet => ExprTree::EmptySet,
        ExprNode::UniversalSet => ExprTree::UniversalSet,
        ExprNode::Interval(start, end, flags) => ExprTree::Interval {
            start: Box::new(expr_to_tree(arena, start)),
            end: Box::new(expr_to_tree(arena, end)),
            flags,
        },
        ExprNode::FiniteSet(elems) => ExprTree::FiniteSet {
            elements: elems.iter().map(|&e| expr_to_tree(arena, e)).collect(),
        },
        ExprNode::SetUnion(sets) => ExprTree::SetUnion {
            sets: sets.iter().map(|&s| expr_to_tree(arena, s)).collect(),
        },
        ExprNode::SetIntersection(sets) => ExprTree::SetIntersection {
            sets: sets.iter().map(|&s| expr_to_tree(arena, s)).collect(),
        },
        ExprNode::SetComplement(a, b) => ExprTree::SetComplement {
            set: Box::new(expr_to_tree(arena, a)),
            universe: Box::new(expr_to_tree(arena, b)),
        },
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ExprTree → ExprId (deserialization direction)
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a standalone [`ExprTree`] back into an arena expression.
///
/// The resulting expression is fully canonicalized (it goes through
/// the arena's canonical constructors).
pub(crate) fn tree_to_expr(arena: &mut Arena, tree: &ExprTree) -> ExprId {
    match tree {
        ExprTree::Num { numer, denom } => {
            let n: BigInt = numer.parse().unwrap_or_default();
            let d: BigInt = denom.parse().unwrap_or_else(|_| BigInt::from(1));
            let r = Ratio::new(n, d);
            let nid = arena.intern_num(r);
            arena.intern(ExprNode::Num(nid))
        }
        ExprTree::Symbol { name } => arena.symbol(name),
        ExprTree::Pi => arena.pi,
        ExprTree::E => arena.e_const,
        ExprTree::ImaginaryUnit => arena.i_unit,
        ExprTree::Infinity => arena.infinity,
        ExprTree::NegInfinity => arena.neg_infinity,
        ExprTree::ComplexInfinity => arena.complex_infinity,
        ExprTree::NaN => arena.nan,
        ExprTree::Add { terms } => {
            let ids: Vec<ExprId> = terms.iter().map(|t| tree_to_expr(arena, t)).collect();
            arena.add(&ids)
        }
        ExprTree::Mul { factors } => {
            let ids: Vec<ExprId> = factors.iter().map(|f| tree_to_expr(arena, f)).collect();
            arena.mul(&ids)
        }
        ExprTree::Pow { base, exp } => {
            let b = tree_to_expr(arena, base);
            let e = tree_to_expr(arena, exp);
            arena.pow(b, e)
        }
        ExprTree::Neg { inner } => {
            let x = tree_to_expr(arena, inner);
            arena.neg(x)
        }
        ExprTree::Sin { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.sin(x)
        }
        ExprTree::Cos { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.cos(x)
        }
        ExprTree::Tan { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.tan(x)
        }
        ExprTree::Exp { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.exp(x)
        }
        ExprTree::Ln { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.ln(x)
        }
        ExprTree::Sqrt { arg } => {
            // Legacy compatibility: convert to Pow(arg, 1/2)
            let x = tree_to_expr(arena, arg);
            arena.sqrt(x) // which now produces Pow(x, 1/2)
        }
        ExprTree::Abs { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.abs(x)
        }
        ExprTree::Asin { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.asin(x)
        }
        ExprTree::Acos { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.acos(x)
        }
        ExprTree::Atan { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.atan(x)
        }
        ExprTree::Atan2 { y, x } => {
            let yid = tree_to_expr(arena, y);
            let xid = tree_to_expr(arena, x);
            arena.atan2(yid, xid)
        }
        ExprTree::Sinh { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.sinh(x)
        }
        ExprTree::Cosh { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.cosh(x)
        }
        ExprTree::Tanh { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.tanh(x)
        }
        ExprTree::Asinh { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.asinh(x)
        }
        ExprTree::Acosh { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.acosh(x)
        }
        ExprTree::Atanh { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.atanh(x)
        }
        ExprTree::Sign { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.sign(x)
        }
        ExprTree::Heaviside { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.heaviside(x)
        }
        ExprTree::DiracDelta { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.dirac_delta(x)
        }
        ExprTree::Gamma { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.gamma(x)
        }
        ExprTree::LogGamma { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.log_gamma(x)
        }
        ExprTree::Digamma { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.digamma(x)
        }
        ExprTree::Erf { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.erf(x)
        }
        ExprTree::Erfc { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.erfc(x)
        }
        ExprTree::Beta { a, b } => {
            let aid = tree_to_expr(arena, a);
            let bid = tree_to_expr(arena, b);
            arena.beta(aid, bid)
        }
        ExprTree::Floor { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.floor(x)
        }
        ExprTree::Ceiling { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.ceiling(x)
        }
        ExprTree::Min { args } => {
            let ids: smallvec::SmallVec<[ExprId; 4]> =
                args.iter().map(|a| tree_to_expr(arena, a)).collect();
            arena.intern(ExprNode::Min(ids))
        }
        ExprTree::Max { args } => {
            let ids: smallvec::SmallVec<[ExprId; 4]> =
                args.iter().map(|a| tree_to_expr(arena, a)).collect();
            arena.intern(ExprNode::Max(ids))
        }
        ExprTree::Apply { name, args } => {
            let sym_id = arena.symbols.intern(name);
            let arg_ids: Vec<ExprId> = args.iter().map(|a| tree_to_expr(arena, a)).collect();
            let sv: smallvec::SmallVec<[ExprId; 2]> = arg_ids.into_iter().collect();
            arena.intern(ExprNode::Apply(sym_id, sv))
        }
        ExprTree::Derivative { body, var } => {
            let b = tree_to_expr(arena, body);
            let v = tree_to_expr(arena, var);
            arena.intern(ExprNode::Derivative(b, v))
        }
        ExprTree::Integral { body, var } => {
            let b = tree_to_expr(arena, body);
            let v = tree_to_expr(arena, var);
            arena.intern(ExprNode::Integral(b, v))
        }
        ExprTree::Sum {
            body,
            var,
            lower,
            upper,
        } => {
            let b = tree_to_expr(arena, body);
            let v = tree_to_expr(arena, var);
            let lo = tree_to_expr(arena, lower);
            let hi = tree_to_expr(arena, upper);
            arena.intern(ExprNode::Sum(b, v, lo, hi))
        }
        ExprTree::Product_ {
            body,
            var,
            lower,
            upper,
        } => {
            let b = tree_to_expr(arena, body);
            let v = tree_to_expr(arena, var);
            let lo = tree_to_expr(arena, lower);
            let hi = tree_to_expr(arena, upper);
            arena.intern(ExprNode::Product_(b, v, lo, hi))
        }
        ExprTree::BoolTrue => arena.bool_true,
        ExprTree::BoolFalse => arena.bool_false,
        ExprTree::Gt { lhs, rhs } => {
            let l = tree_to_expr(arena, lhs);
            let r = tree_to_expr(arena, rhs);
            arena.gt(l, r)
        }
        ExprTree::Ge { lhs, rhs } => {
            let l = tree_to_expr(arena, lhs);
            let r = tree_to_expr(arena, rhs);
            arena.ge(l, r)
        }
        ExprTree::Eq_ { lhs, rhs } => {
            let l = tree_to_expr(arena, lhs);
            let r = tree_to_expr(arena, rhs);
            arena.eq_(l, r)
        }
        ExprTree::Ne { lhs, rhs } => {
            let l = tree_to_expr(arena, lhs);
            let r = tree_to_expr(arena, rhs);
            arena.ne_(l, r)
        }
        ExprTree::And { args } => {
            let ids: Vec<ExprId> = args.iter().map(|a| tree_to_expr(arena, a)).collect();
            arena.and(&ids)
        }
        ExprTree::Or { args } => {
            let ids: Vec<ExprId> = args.iter().map(|a| tree_to_expr(arena, a)).collect();
            arena.or(&ids)
        }
        ExprTree::Not { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.not(x)
        }
        ExprTree::Piecewise { pieces } => {
            let pairs: smallvec::SmallVec<[(ExprId, ExprId); 3]> = pieces
                .iter()
                .map(|(val, cond)| (tree_to_expr(arena, val), tree_to_expr(arena, cond)))
                .collect();
            arena.intern(ExprNode::Piecewise(pairs))
        }
        ExprTree::EmptySet => arena.intern(ExprNode::EmptySet),
        ExprTree::UniversalSet => arena.intern(ExprNode::UniversalSet),
        ExprTree::Interval { start, end, flags } => {
            let s = tree_to_expr(arena, start);
            let e = tree_to_expr(arena, end);
            arena.intern(ExprNode::Interval(s, e, *flags))
        }
        ExprTree::FiniteSet { elements } => {
            let ids: smallvec::SmallVec<[ExprId; 4]> =
                elements.iter().map(|e| tree_to_expr(arena, e)).collect();
            arena.intern(ExprNode::FiniteSet(ids))
        }
        ExprTree::SetUnion { sets } => {
            let ids: smallvec::SmallVec<[ExprId; 4]> =
                sets.iter().map(|s| tree_to_expr(arena, s)).collect();
            arena.intern(ExprNode::SetUnion(ids))
        }
        ExprTree::SetIntersection { sets } => {
            let ids: smallvec::SmallVec<[ExprId; 4]> =
                sets.iter().map(|s| tree_to_expr(arena, s)).collect();
            arena.intern(ExprNode::SetIntersection(ids))
        }
        ExprTree::SetComplement { set, universe } => {
            let s = tree_to_expr(arena, set);
            let u = tree_to_expr(arena, universe);
            arena.intern(ExprNode::SetComplement(s, u))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;

    fn display(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    #[test]
    fn roundtrip_integer() {
        let mut a = Arena::new();
        let expr = a.int(42);
        let tree = expr_to_tree(&a, expr);
        let back = tree_to_expr(&mut a, &tree);
        assert_eq!(display(&a, back), "42");
    }

    #[test]
    fn roundtrip_rational() {
        let mut a = Arena::new();
        let expr = a.rational(3, 7);
        let tree = expr_to_tree(&a, expr);
        let back = tree_to_expr(&mut a, &tree);
        assert_eq!(display(&a, back), "3/7");
    }

    #[test]
    fn roundtrip_symbol() {
        let mut a = Arena::new();
        let expr = a.symbol("x");
        let tree = expr_to_tree(&a, expr);
        let back = tree_to_expr(&mut a, &tree);
        assert_eq!(display(&a, back), "x");
    }

    #[test]
    fn roundtrip_polynomial() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let two_id = a.int(2);
        let two_x = a.mul(&[two_id, x]);
        let one = a.one;
        let expr = a.add(&[x2, two_x, one]);
        let tree = expr_to_tree(&a, expr);
        let back = tree_to_expr(&mut a, &tree);
        assert_eq!(display(&a, expr), display(&a, back));
    }

    #[test]
    fn roundtrip_sin() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let expr = a.sin(x);
        let tree = expr_to_tree(&a, expr);
        let back = tree_to_expr(&mut a, &tree);
        assert_eq!(display(&a, back), "sin(x)");
    }

    #[test]
    fn roundtrip_pi() {
        let a = Arena::new();
        let tree = expr_to_tree(&a, a.pi);
        let mut a2 = Arena::new();
        let back = tree_to_expr(&mut a2, &tree);
        assert_eq!(display(&a2, back), "pi");
    }

    #[test]
    fn json_roundtrip() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let expr = a.pow(x, two);
        let tree = expr_to_tree(&a, expr);
        let json = serde_json::to_string(&tree).unwrap();
        let tree2: ExprTree = serde_json::from_str(&json).unwrap();
        assert_eq!(tree, tree2);
        let back = tree_to_expr(&mut a, &tree2);
        assert_eq!(display(&a, back), "x^2");
    }

    #[test]
    fn json_deserialize_from_scratch() {
        // Simulate receiving JSON from an external source
        let json = r#"{"type":"Add","terms":[{"type":"Num","numer":"1","denom":"1"},{"type":"Symbol","name":"x"}]}"#;
        let tree: ExprTree = serde_json::from_str(json).unwrap();
        let mut a = Arena::new();
        let expr = tree_to_expr(&mut a, &tree);
        assert_eq!(display(&a, expr), "x + 1");
    }

    #[test]
    fn roundtrip_all_functions() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let funcs = [
            a.sin(x),
            a.cos(x),
            a.tan(x),
            a.exp(x),
            a.ln(x),
            a.sqrt(x),
            a.abs(x),
            a.asin(x),
            a.acos(x),
            a.atan(x),
            a.sinh(x),
            a.cosh(x),
            a.tanh(x),
        ];
        for &expr in &funcs {
            let tree = expr_to_tree(&a, expr);
            let back = tree_to_expr(&mut a, &tree);
            assert_eq!(
                display(&a, expr),
                display(&a, back),
                "roundtrip failed for {}",
                display(&a, expr)
            );
        }
    }

    #[test]
    fn roundtrip_derivative() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let body = a.sin(x);
        let expr = a.intern(ExprNode::Derivative(body, x));
        let tree = expr_to_tree(&a, expr);
        let back = tree_to_expr(&mut a, &tree);
        assert_eq!(display(&a, back), "Derivative(sin(x), x)");
    }

    #[test]
    fn roundtrip_integral() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let body = a.sin(x);
        let expr = a.intern(ExprNode::Integral(body, x));
        let tree = expr_to_tree(&a, expr);
        let back = tree_to_expr(&mut a, &tree);
        assert_eq!(display(&a, back), "Integral(sin(x), x)");
    }
}
