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

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};

/// A serializable expression tree.
///
/// This is a standalone tree (not arena-indexed) that can be serialized
/// to JSON or any serde-supported format. Use [`Ex::to_tree()`](crate::api::expr::Ex::to_tree) to
/// convert from an expression handle, and [`Context::from_tree()`](crate::api::context::Context::from_tree) to
/// convert back.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ExprTree {
    /// A rational number with arbitrary-precision numerator and denominator.
    Num {
        /// Numerator as a decimal string.
        numer: String,
        /// Denominator as a decimal string.
        denom: String,
    },
    /// A symbolic variable.
    Symbol {
        /// The variable name.
        name: String,
    },
    /// The constant π.
    Pi,
    /// Euler's number e.
    E,
    /// The imaginary unit i.
    ImaginaryUnit,
    /// The Euler–Mascheroni constant γ.
    EulerGamma,
    /// Catalan's constant G.
    Catalan,
    /// The golden ratio φ.
    GoldenRatio,
    /// A named physical constant with a known exact value.
    PhysicalConstant {
        /// The display name (e.g., "c", "h", "k_B").
        name: String,
        /// The exact value of the constant.
        value: Box<ExprTree>,
    },
    /// Positive infinity.
    Infinity,
    /// Negative infinity.
    NegInfinity,
    /// Complex infinity (undirected).
    ComplexInfinity,
    /// Not-a-number.
    NaN,
    /// A sum of terms.
    Add {
        /// The summands.
        terms: Vec<ExprTree>,
    },
    /// A product of factors.
    Mul {
        /// The multiplicands.
        factors: Vec<ExprTree>,
    },
    /// Exponentiation: base^exp.
    Pow {
        /// The base expression.
        base: Box<ExprTree>,
        /// The exponent expression.
        exp: Box<ExprTree>,
    },
    /// Negation: -inner.
    Neg {
        /// The negated expression.
        inner: Box<ExprTree>,
    },
    /// Sine.
    Sin {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Cosine.
    Cos {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Tangent.
    Tan {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Natural exponential.
    Exp {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Natural logarithm.
    Ln {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Square root.
    Sqrt {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Absolute value.
    Abs {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Inverse sine.
    Asin {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Inverse cosine.
    Acos {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Inverse tangent.
    Atan {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Two-argument arctangent: atan2(y, x).
    Atan2 {
        /// The y coordinate.
        y: Box<ExprTree>,
        /// The x coordinate.
        x: Box<ExprTree>,
    },
    /// Hyperbolic sine.
    Sinh {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Hyperbolic cosine.
    Cosh {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Hyperbolic tangent.
    Tanh {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Inverse hyperbolic sine.
    Asinh {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Inverse hyperbolic cosine.
    Acosh {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Inverse hyperbolic tangent.
    Atanh {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Sign function: 1 if positive, -1 if negative, 0 if zero.
    Sign {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Heaviside step function: H(x) = 0 for x<0, 1/2 for x=0, 1 for x>0.
    Heaviside {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Dirac delta distribution: δ(x) = 0 for x≠0, symbolic at x=0.
    DiracDelta {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Gamma function: Γ(x).
    Gamma {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Log-gamma function: ln(Γ(x)).
    LogGamma {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Digamma function: ψ(x) = Γ'(x)/Γ(x).
    Digamma {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Error function: erf(x).
    Erf {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Complementary error function: erfc(x) = 1 - erf(x).
    Erfc {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Lambert W function (principal branch): W(x)·exp(W(x)) = x.  The
    /// other branches `W_k(x)` are `Apply { name: "lambertw", args: [x, k] }`.
    LambertW {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Beta function: B(a, b) = Γ(a)Γ(b)/Γ(a+b).
    Beta {
        /// First parameter.
        a: Box<ExprTree>,
        /// Second parameter.
        b: Box<ExprTree>,
    },
    /// Real part: re(z).
    Re {
        /// The complex argument.
        arg: Box<ExprTree>,
    },
    /// Imaginary part: im(z).
    Im {
        /// The complex argument.
        arg: Box<ExprTree>,
    },
    /// Complex conjugate.
    Conjugate {
        /// The complex argument.
        arg: Box<ExprTree>,
    },
    /// Principal complex argument arg(z) ∈ (−π, π].
    Arg {
        /// The complex argument.
        arg: Box<ExprTree>,
    },
    /// Sine integral Si(x).
    Si {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Cosine integral Ci(x).
    Ci {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Exponential integral Ei(x).
    Ei {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Logarithmic integral li(x).
    Li {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Riemann zeta function ζ(s).
    Zeta {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Polygamma function ψ⁽ⁿ⁾(x).
    Polygamma {
        /// The derivative order n.
        n: Box<ExprTree>,
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Kronecker delta δᵢⱼ.
    KroneckerDelta {
        /// First index.
        i: Box<ExprTree>,
        /// Second index.
        j: Box<ExprTree>,
    },
    /// Floor function: greatest integer <= x.
    Floor {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// Ceiling function: least integer >= x.
    Ceiling {
        /// The function argument.
        arg: Box<ExprTree>,
    },
    /// N-ary minimum.
    Min {
        /// The candidate expressions.
        args: Vec<ExprTree>,
    },
    /// N-ary maximum.
    Max {
        /// The candidate expressions.
        args: Vec<ExprTree>,
    },
    /// Boolean true.
    BoolTrue,
    /// Boolean false.
    BoolFalse,
    /// Greater than: lhs > rhs.
    Gt {
        /// Left operand.
        lhs: Box<ExprTree>,
        /// Right operand.
        rhs: Box<ExprTree>,
    },
    /// Greater than or equal: lhs >= rhs.
    Ge {
        /// Left operand.
        lhs: Box<ExprTree>,
        /// Right operand.
        rhs: Box<ExprTree>,
    },
    /// Mathematical equality test: lhs == rhs.
    Eq_ {
        /// Left operand.
        lhs: Box<ExprTree>,
        /// Right operand.
        rhs: Box<ExprTree>,
    },
    /// Not equal: lhs != rhs.
    Ne {
        /// Left operand.
        lhs: Box<ExprTree>,
        /// Right operand.
        rhs: Box<ExprTree>,
    },
    /// Logical conjunction (n-ary).
    And {
        /// The conjuncts.
        args: Vec<ExprTree>,
    },
    /// Logical disjunction (n-ary).
    Or {
        /// The disjuncts.
        args: Vec<ExprTree>,
    },
    /// Logical negation.
    Not {
        /// The negated expression.
        arg: Box<ExprTree>,
    },
    /// Piecewise function: list of (value, condition) pairs.
    Piecewise {
        /// List of (value, condition) pairs.
        pieces: Vec<(ExprTree, ExprTree)>,
    },
    /// Application of a named function.
    Apply {
        /// The function name.
        name: String,
        /// The function arguments.
        args: Vec<ExprTree>,
    },
    /// Formal derivative.
    Derivative {
        /// The expression being differentiated.
        body: Box<ExprTree>,
        /// The variable of differentiation.
        var: Box<ExprTree>,
    },
    /// Formal integral.
    Integral {
        /// The integrand.
        body: Box<ExprTree>,
        /// The variable of integration.
        var: Box<ExprTree>,
    },
    /// Formal definite integral: `∫_lower^upper body dvar`.
    DefiniteIntegral {
        /// The integrand.
        body: Box<ExprTree>,
        /// The variable of integration (bound inside `body`).
        var: Box<ExprTree>,
        /// Lower bound of integration.
        lower: Box<ExprTree>,
        /// Upper bound of integration.
        upper: Box<ExprTree>,
    },
    /// Symbolic summation: Sum(body, var, lower, upper).
    Sum {
        /// The expression being summed.
        body: Box<ExprTree>,
        /// The index variable.
        var: Box<ExprTree>,
        /// Lower bound of summation.
        lower: Box<ExprTree>,
        /// Upper bound of summation.
        upper: Box<ExprTree>,
    },
    /// Symbolic product: Product(body, var, lower, upper).
    Product_ {
        /// The expression being multiplied.
        body: Box<ExprTree>,
        /// The index variable.
        var: Box<ExprTree>,
        /// Lower bound of the product.
        lower: Box<ExprTree>,
        /// Upper bound of the product.
        upper: Box<ExprTree>,
    },
    /// The empty set ∅.
    EmptySet,
    /// The universal set.
    UniversalSet,
    /// A closed/open interval with flags encoding open/closed.
    /// Bits: 0x01 = left_open, 0x02 = right_open.
    Interval {
        /// Left endpoint.
        start: Box<ExprTree>,
        /// Right endpoint.
        end: Box<ExprTree>,
        /// Bitfield: 0x01 = left open, 0x02 = right open.
        flags: u8,
    },
    /// A finite set of elements {a, b, c, ...}.
    FiniteSet {
        /// The set elements.
        elements: Vec<ExprTree>,
    },
    /// Union of sets: A ∪ B ∪ C ∪ ...
    SetUnion {
        /// The sets being united.
        sets: Vec<ExprTree>,
    },
    /// Intersection of sets: A ∩ B ∩ C ∩ ...
    SetIntersection {
        /// The sets being intersected.
        sets: Vec<ExprTree>,
    },
    /// Set complement (relative): A \ B.
    SetComplement {
        /// The set to complement.
        set: Box<ExprTree>,
        /// The universe set to complement within.
        universe: Box<ExprTree>,
    },
    /// Limit: lim_{var -> point} body.
    Limit {
        /// The expression to take the limit of.
        body: Box<ExprTree>,
        /// The variable approaching the limit point.
        var: Box<ExprTree>,
        /// The point being approached.
        point: Box<ExprTree>,
    },
    /// Series expansion of body around point in var up to order.
    Series {
        /// The expression to expand.
        body: Box<ExprTree>,
        /// The expansion variable.
        var: Box<ExprTree>,
        /// The expansion point.
        point: Box<ExprTree>,
        /// The truncation order.
        order: Box<ExprTree>,
    },
    /// Laplace transform: L{body}(t -> s).
    LaplaceTransform {
        /// The time-domain expression.
        body: Box<ExprTree>,
        /// The time variable.
        t: Box<ExprTree>,
        /// The frequency variable.
        s: Box<ExprTree>,
    },
    /// Inverse Laplace transform: L^{-1}{body}(s -> t).
    InverseLaplaceTransform {
        /// The frequency-domain expression.
        body: Box<ExprTree>,
        /// The frequency variable.
        s: Box<ExprTree>,
        /// The time variable.
        t: Box<ExprTree>,
    },
    /// Residue of body at var = point.
    Residue {
        /// The expression to compute the residue of.
        body: Box<ExprTree>,
        /// The variable.
        var: Box<ExprTree>,
        /// The pole location.
        point: Box<ExprTree>,
    },
    /// Root of a polynomial: RootOf(poly, index), or RootOf(poly, var,
    /// index) when `poly` has symbols other than `var`.
    RootOf {
        /// The polynomial expression.
        poly: Box<ExprTree>,
        /// The polynomial's variable, bound in `poly`.  `None` (and absent
        /// from JSON) when it is the only symbol of `poly`.  A tree without
        /// it whose polynomial has several symbols (written before 0.29)
        /// takes the first symbol met in `poly`, as evaluation did then.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        var: Option<Box<ExprTree>>,
        /// The root index (0-based).
        index: Box<ExprTree>,
    },
    /// Differential equation solver: DSolve(expr, func, var).
    DSolve {
        /// The ODE expression (equal to zero).
        expr: Box<ExprTree>,
        /// The unknown function.
        func: Box<ExprTree>,
        /// The independent variable.
        var: Box<ExprTree>,
    },
    /// Sum over roots of a polynomial: RootSum(poly, body, sumvar).
    RootSum {
        /// The polynomial whose roots are summed over.
        poly: Box<ExprTree>,
        /// The body expression evaluated at each root.
        body: Box<ExprTree>,
        /// The bound summation variable.
        sumvar: Box<ExprTree>,
    },
    /// Condition set: {var | condition}.
    ConditionSet {
        /// The set variable.
        var: Box<ExprTree>,
        /// The membership condition.
        condition: Box<ExprTree>,
    },
    /// Unevaluated substitution `body|_{var = point}`: Subs(body, var,
    /// point), e.g. `f′(0)` for an undefined `f`.
    Subs {
        /// The expression, a function of `var`.
        body: Box<ExprTree>,
        /// The bound variable.
        var: Box<ExprTree>,
        /// The point `var` is set to.
        point: Box<ExprTree>,
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
        ExprNode::EulerGamma => ExprTree::EulerGamma,
        ExprNode::Catalan => ExprTree::Catalan,
        ExprNode::GoldenRatio => ExprTree::GoldenRatio,
        ExprNode::PhysicalConstant(name_id, value_id) => ExprTree::PhysicalConstant {
            name: arena.symbol_name(name_id).to_owned(),
            value: Box::new(expr_to_tree(arena, value_id)),
        },
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
        ExprNode::LambertW(x) => ExprTree::LambertW {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Beta(a, b) => ExprTree::Beta {
            a: Box::new(expr_to_tree(arena, a)),
            b: Box::new(expr_to_tree(arena, b)),
        },
        ExprNode::Re(x) => ExprTree::Re {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Im(x) => ExprTree::Im {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Conjugate(x) => ExprTree::Conjugate {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Arg(x) => ExprTree::Arg {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Si(x) => ExprTree::Si {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Ci(x) => ExprTree::Ci {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Ei(x) => ExprTree::Ei {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Li(x) => ExprTree::Li {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Zeta(x) => ExprTree::Zeta {
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::Polygamma(n, x) => ExprTree::Polygamma {
            n: Box::new(expr_to_tree(arena, n)),
            arg: Box::new(expr_to_tree(arena, x)),
        },
        ExprNode::KroneckerDelta(i, j) => ExprTree::KroneckerDelta {
            i: Box::new(expr_to_tree(arena, i)),
            j: Box::new(expr_to_tree(arena, j)),
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
        ExprNode::DefiniteIntegral(body, var, lo, hi) => ExprTree::DefiniteIntegral {
            body: Box::new(expr_to_tree(arena, body)),
            var: Box::new(expr_to_tree(arena, var)),
            lower: Box::new(expr_to_tree(arena, lo)),
            upper: Box::new(expr_to_tree(arena, hi)),
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
        ExprNode::Limit(body, var, point) => ExprTree::Limit {
            body: Box::new(expr_to_tree(arena, body)),
            var: Box::new(expr_to_tree(arena, var)),
            point: Box::new(expr_to_tree(arena, point)),
        },
        ExprNode::Series(body, var, point, order) => ExprTree::Series {
            body: Box::new(expr_to_tree(arena, body)),
            var: Box::new(expr_to_tree(arena, var)),
            point: Box::new(expr_to_tree(arena, point)),
            order: Box::new(expr_to_tree(arena, order)),
        },
        ExprNode::LaplaceTransform(body, t, s) => ExprTree::LaplaceTransform {
            body: Box::new(expr_to_tree(arena, body)),
            t: Box::new(expr_to_tree(arena, t)),
            s: Box::new(expr_to_tree(arena, s)),
        },
        ExprNode::InverseLaplaceTransform(body, s, t) => ExprTree::InverseLaplaceTransform {
            body: Box::new(expr_to_tree(arena, body)),
            s: Box::new(expr_to_tree(arena, s)),
            t: Box::new(expr_to_tree(arena, t)),
        },
        ExprNode::Residue(body, var, point) => ExprTree::Residue {
            body: Box::new(expr_to_tree(arena, body)),
            var: Box::new(expr_to_tree(arena, var)),
            point: Box::new(expr_to_tree(arena, point)),
        },
        ExprNode::RootOf(poly, var, index) => ExprTree::RootOf {
            poly: Box::new(expr_to_tree(arena, poly)),
            var: (crate::base::walk::root_of_implied_var(arena, poly) != Some(var))
                .then(|| Box::new(expr_to_tree(arena, var))),
            index: Box::new(expr_to_tree(arena, index)),
        },
        ExprNode::DSolve(expr, func, var) => ExprTree::DSolve {
            expr: Box::new(expr_to_tree(arena, expr)),
            func: Box::new(expr_to_tree(arena, func)),
            var: Box::new(expr_to_tree(arena, var)),
        },
        ExprNode::RootSum(poly, body, sumvar) => ExprTree::RootSum {
            poly: Box::new(expr_to_tree(arena, poly)),
            body: Box::new(expr_to_tree(arena, body)),
            sumvar: Box::new(expr_to_tree(arena, sumvar)),
        },
        ExprNode::ConditionSet(var, condition) => ExprTree::ConditionSet {
            var: Box::new(expr_to_tree(arena, var)),
            condition: Box::new(expr_to_tree(arena, condition)),
        },
        ExprNode::Subs(body, var, point) => ExprTree::Subs {
            body: Box::new(expr_to_tree(arena, body)),
            var: Box::new(expr_to_tree(arena, var)),
            point: Box::new(expr_to_tree(arena, point)),
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
        ExprTree::EulerGamma => arena.euler_gamma,
        ExprTree::Catalan => arena.catalan,
        ExprTree::GoldenRatio => arena.golden_ratio,
        ExprTree::PhysicalConstant { name, value } => {
            let val_id = tree_to_expr(arena, value);
            arena.physical_constant(name, val_id)
        }
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
        ExprTree::LambertW { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.lambertw(x)
        }
        ExprTree::Beta { a, b } => {
            let aid = tree_to_expr(arena, a);
            let bid = tree_to_expr(arena, b);
            arena.beta(aid, bid)
        }
        ExprTree::Re { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.re(x)
        }
        ExprTree::Im { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.im(x)
        }
        ExprTree::Conjugate { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.conjugate(x)
        }
        ExprTree::Arg { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.arg(x)
        }
        ExprTree::Si { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.si(x)
        }
        ExprTree::Ci { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.ci(x)
        }
        ExprTree::Ei { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.ei(x)
        }
        ExprTree::Li { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.li(x)
        }
        ExprTree::Zeta { arg } => {
            let x = tree_to_expr(arena, arg);
            arena.zeta(x)
        }
        ExprTree::Polygamma { n, arg } => {
            let nid = tree_to_expr(arena, n);
            let x = tree_to_expr(arena, arg);
            arena.polygamma(nid, x)
        }
        ExprTree::KroneckerDelta { i, j } => {
            let iid = tree_to_expr(arena, i);
            let jid = tree_to_expr(arena, j);
            arena.kronecker_delta(iid, jid)
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
        // `to_tree` writes the built-in `Factorial` / `Binomial` nodes as
        // `Apply` (as the parser reads them); map them back so the round
        // trip is the identity.
        ExprTree::Apply { name, args } if name == "factorial" && args.len() == 1 => {
            let x = tree_to_expr(arena, &args[0]);
            arena.factorial(x)
        }
        ExprTree::Apply { name, args } if name == "binomial" && args.len() == 2 => {
            let n = tree_to_expr(arena, &args[0]);
            let k = tree_to_expr(arena, &args[1]);
            arena.binomial(n, k)
        }
        // The branch `k` of Lambert W (`W(x, 0)` is the principal-branch node).
        ExprTree::Apply { name, args } if name == "lambertw" && args.len() == 2 => {
            let x = tree_to_expr(arena, &args[0]);
            let k = tree_to_expr(arena, &args[1]);
            arena.lambertw_branch(x, k)
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
        ExprTree::DefiniteIntegral {
            body,
            var,
            lower,
            upper,
        } => {
            let b = tree_to_expr(arena, body);
            let v = tree_to_expr(arena, var);
            let lo = tree_to_expr(arena, lower);
            let hi = tree_to_expr(arena, upper);
            arena.definite_integral(b, v, lo, hi)
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
        ExprTree::Limit { body, var, point } => {
            let b = tree_to_expr(arena, body);
            let v = tree_to_expr(arena, var);
            let p = tree_to_expr(arena, point);
            arena.intern(ExprNode::Limit(b, v, p))
        }
        ExprTree::Series {
            body,
            var,
            point,
            order,
        } => {
            let b = tree_to_expr(arena, body);
            let v = tree_to_expr(arena, var);
            let p = tree_to_expr(arena, point);
            let o = tree_to_expr(arena, order);
            arena.intern(ExprNode::Series(b, v, p, o))
        }
        ExprTree::LaplaceTransform { body, t, s } => {
            let b = tree_to_expr(arena, body);
            let ti = tree_to_expr(arena, t);
            let si = tree_to_expr(arena, s);
            arena.intern(ExprNode::LaplaceTransform(b, ti, si))
        }
        ExprTree::InverseLaplaceTransform { body, s, t } => {
            let b = tree_to_expr(arena, body);
            let si = tree_to_expr(arena, s);
            let ti = tree_to_expr(arena, t);
            arena.intern(ExprNode::InverseLaplaceTransform(b, si, ti))
        }
        ExprTree::Residue { body, var, point } => {
            let b = tree_to_expr(arena, body);
            let v = tree_to_expr(arena, var);
            let p = tree_to_expr(arena, point);
            arena.intern(ExprNode::Residue(b, v, p))
        }
        ExprTree::RootOf { poly, var, index } => {
            let p = tree_to_expr(arena, poly);
            let v = match var {
                Some(v) => tree_to_expr(arena, v),
                None => root_of_legacy_var(arena, p),
            };
            let i = tree_to_expr(arena, index);
            arena.intern(ExprNode::RootOf(p, v, i))
        }
        ExprTree::DSolve { expr, func, var } => {
            let e = tree_to_expr(arena, expr);
            let f = tree_to_expr(arena, func);
            let v = tree_to_expr(arena, var);
            arena.intern(ExprNode::DSolve(e, f, v))
        }
        ExprTree::RootSum { poly, body, sumvar } => {
            let p = tree_to_expr(arena, poly);
            let b = tree_to_expr(arena, body);
            let s = tree_to_expr(arena, sumvar);
            arena.intern(ExprNode::RootSum(p, b, s))
        }
        ExprTree::ConditionSet { var, condition } => {
            let v = tree_to_expr(arena, var);
            let c = tree_to_expr(arena, condition);
            arena.intern(ExprNode::ConditionSet(v, c))
        }
        ExprTree::Subs { body, var, point } => {
            let b = tree_to_expr(arena, body);
            let v = tree_to_expr(arena, var);
            let p = tree_to_expr(arena, point);
            crate::transforms::subs::subs(arena, b, v, p)
        }
    }
}

/// The variable of a `RootOf` tree that does not name one: the only
/// symbol of `poly`; for a tree written before 0.29 whose polynomial has
/// several, the first met (the one evaluation used then); for a constant
/// polynomial, which has no roots, a symbol `x` (bound, so it cannot clash
/// with an `x` outside).
fn root_of_legacy_var(arena: &mut Arena, poly: ExprId) -> ExprId {
    match crate::base::walk::free_symbols(arena, poly).first() {
        Some(&v) => v,
        None => arena.symbol("x"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// srepr and DOT (0.9.1)
// ═══════════════════════════════════════════════════════════════════════════

impl ExprTree {
    /// The constructor head and the ordered children of this node.
    ///
    /// `None` children marks an atom (printed as the bare head); `Some`
    /// marks a compound node (printed as `head(child, …)`, even with zero
    /// children).  Heads follow SymPy's `srepr` where SymPy has the node
    /// (`Integer`, `Rational`, `Symbol('x')`, `Add`, `Mul`, `Pow`, `sin`,
    /// `log`, `StrictGreaterThan`, `Interval`, …) and the symplex name
    /// otherwise (`Neg`, `DefiniteIntegral`, `Series`, `RootSum`).
    fn head_and_children(&self) -> (String, Option<Vec<&ExprTree>>) {
        type Parts<'t> = (String, Option<Vec<&'t ExprTree>>);
        fn one<'t>(head: &str, a: &'t ExprTree) -> Parts<'t> {
            (head.to_string(), Some(vec![a]))
        }
        fn two<'t>(head: &str, a: &'t ExprTree, b: &'t ExprTree) -> Parts<'t> {
            (head.to_string(), Some(vec![a, b]))
        }
        fn many<'t>(head: &str, items: &'t [ExprTree]) -> Parts<'t> {
            (head.to_string(), Some(items.iter().collect()))
        }
        fn atom<'t>(head: &str) -> Parts<'t> {
            (head.to_string(), None)
        }
        match self {
            ExprTree::Num { numer, denom } => {
                if denom == "1" {
                    atom(&format!("Integer({numer})"))
                } else {
                    atom(&format!("Rational({numer}, {denom})"))
                }
            }
            ExprTree::Symbol { name } => atom(&format!("Symbol('{name}')")),
            ExprTree::Pi => atom("pi"),
            ExprTree::E => atom("E"),
            ExprTree::ImaginaryUnit => atom("I"),
            ExprTree::EulerGamma => atom("EulerGamma"),
            ExprTree::Catalan => atom("Catalan"),
            ExprTree::GoldenRatio => atom("GoldenRatio"),
            ExprTree::PhysicalConstant { name, value } => {
                one(&format!("PhysicalConstant('{name}')"), value)
            }
            ExprTree::Infinity => atom("oo"),
            ExprTree::NegInfinity => atom("-oo"),
            ExprTree::ComplexInfinity => atom("zoo"),
            ExprTree::NaN => atom("nan"),
            ExprTree::Add { terms } => many("Add", terms),
            ExprTree::Mul { factors } => many("Mul", factors),
            ExprTree::Pow { base, exp } => two("Pow", base, exp),
            ExprTree::Neg { inner } => one("Neg", inner),
            ExprTree::Sin { arg } => one("sin", arg),
            ExprTree::Cos { arg } => one("cos", arg),
            ExprTree::Tan { arg } => one("tan", arg),
            ExprTree::Exp { arg } => one("exp", arg),
            ExprTree::Ln { arg } => one("log", arg),
            ExprTree::Sqrt { arg } => one("sqrt", arg),
            ExprTree::Abs { arg } => one("Abs", arg),
            ExprTree::Asin { arg } => one("asin", arg),
            ExprTree::Acos { arg } => one("acos", arg),
            ExprTree::Atan { arg } => one("atan", arg),
            ExprTree::Atan2 { y, x } => two("atan2", y, x),
            ExprTree::Sinh { arg } => one("sinh", arg),
            ExprTree::Cosh { arg } => one("cosh", arg),
            ExprTree::Tanh { arg } => one("tanh", arg),
            ExprTree::Asinh { arg } => one("asinh", arg),
            ExprTree::Acosh { arg } => one("acosh", arg),
            ExprTree::Atanh { arg } => one("atanh", arg),
            ExprTree::Sign { arg } => one("sign", arg),
            ExprTree::Heaviside { arg } => one("Heaviside", arg),
            ExprTree::DiracDelta { arg } => one("DiracDelta", arg),
            ExprTree::Gamma { arg } => one("gamma", arg),
            ExprTree::LogGamma { arg } => one("loggamma", arg),
            ExprTree::Digamma { arg } => one("digamma", arg),
            ExprTree::Erf { arg } => one("erf", arg),
            ExprTree::Erfc { arg } => one("erfc", arg),
            ExprTree::LambertW { arg } => one("LambertW", arg),
            ExprTree::Beta { a, b } => two("beta", a, b),
            ExprTree::Re { arg } => one("re", arg),
            ExprTree::Im { arg } => one("im", arg),
            ExprTree::Conjugate { arg } => one("conjugate", arg),
            ExprTree::Arg { arg } => one("arg", arg),
            ExprTree::Si { arg } => one("Si", arg),
            ExprTree::Ci { arg } => one("Ci", arg),
            ExprTree::Ei { arg } => one("Ei", arg),
            ExprTree::Li { arg } => one("li", arg),
            ExprTree::Zeta { arg } => one("zeta", arg),
            ExprTree::Polygamma { n, arg } => two("polygamma", n, arg),
            ExprTree::KroneckerDelta { i, j } => two("KroneckerDelta", i, j),
            ExprTree::Floor { arg } => one("floor", arg),
            ExprTree::Ceiling { arg } => one("ceiling", arg),
            ExprTree::Min { args } => many("Min", args),
            ExprTree::Max { args } => many("Max", args),
            ExprTree::BoolTrue => atom("true"),
            ExprTree::BoolFalse => atom("false"),
            ExprTree::Gt { lhs, rhs } => two("StrictGreaterThan", lhs, rhs),
            ExprTree::Ge { lhs, rhs } => two("GreaterThan", lhs, rhs),
            ExprTree::Eq_ { lhs, rhs } => two("Equality", lhs, rhs),
            ExprTree::Ne { lhs, rhs } => two("Unequality", lhs, rhs),
            ExprTree::And { args } => many("And", args),
            ExprTree::Or { args } => many("Or", args),
            ExprTree::Not { arg } => one("Not", arg),
            ExprTree::Piecewise { pieces } => (
                "Piecewise".to_string(),
                Some(pieces.iter().flat_map(|(v, c)| [v, c]).collect()),
            ),
            // SymPy's head for the branch `k` of Lambert W, `LambertW(x, k)`.
            ExprTree::Apply { name, args } if name == "lambertw" => many("LambertW", args),
            ExprTree::Apply { name, args } => many(name, args),
            ExprTree::Derivative { body, var } => two("Derivative", body, var),
            ExprTree::Integral { body, var } => two("Integral", body, var),
            ExprTree::DefiniteIntegral {
                body,
                var,
                lower,
                upper,
            } => (
                "DefiniteIntegral".to_string(),
                Some(vec![body, var, lower, upper]),
            ),
            ExprTree::Sum {
                body,
                var,
                lower,
                upper,
            } => ("Sum".to_string(), Some(vec![body, var, lower, upper])),
            ExprTree::Product_ {
                body,
                var,
                lower,
                upper,
            } => ("Product".to_string(), Some(vec![body, var, lower, upper])),
            ExprTree::EmptySet => atom("EmptySet"),
            ExprTree::UniversalSet => atom("UniversalSet"),
            ExprTree::Interval { start, end, .. } => two("Interval", start, end),
            ExprTree::FiniteSet { elements } => many("FiniteSet", elements),
            ExprTree::SetUnion { sets } => many("Union", sets),
            ExprTree::SetIntersection { sets } => many("Intersection", sets),
            ExprTree::SetComplement { set, universe } => two("Complement", set, universe),
            ExprTree::Limit { body, var, point } => {
                ("Limit".to_string(), Some(vec![body, var, point]))
            }
            ExprTree::Series {
                body,
                var,
                point,
                order,
            } => ("Series".to_string(), Some(vec![body, var, point, order])),
            ExprTree::LaplaceTransform { body, t, s } => {
                ("LaplaceTransform".to_string(), Some(vec![body, t, s]))
            }
            ExprTree::InverseLaplaceTransform { body, s, t } => (
                "InverseLaplaceTransform".to_string(),
                Some(vec![body, s, t]),
            ),
            ExprTree::Residue { body, var, point } => {
                ("Residue".to_string(), Some(vec![body, var, point]))
            }
            ExprTree::RootOf {
                poly,
                var: None,
                index,
            } => two("RootOf", poly, index),
            ExprTree::RootOf {
                poly,
                var: Some(var),
                index,
            } => ("RootOf".to_string(), Some(vec![poly, var, index])),
            ExprTree::DSolve { expr, func, var } => {
                ("DSolve".to_string(), Some(vec![expr, func, var]))
            }
            ExprTree::RootSum { poly, body, sumvar } => {
                ("RootSum".to_string(), Some(vec![poly, body, sumvar]))
            }
            ExprTree::ConditionSet { var, condition } => two("ConditionSet", var, condition),
            ExprTree::Subs { body, var, point } => {
                ("Subs".to_string(), Some(vec![body, var, point]))
            }
        }
    }

    /// Extra literal arguments printed after the children in `to_srepr`
    /// (the open/closed flags of an `Interval`).
    fn srepr_trailing(&self) -> Option<String> {
        match self {
            ExprTree::Interval { flags, .. } => {
                Some(format!(", {}, {}", flags & 0x01 != 0, flags & 0x02 != 0))
            }
            _ => None,
        }
    }

    /// SymPy-`srepr`-style constructor form of this tree — unambiguous and
    /// total (every variant prints).  See [`Ex::to_srepr`](crate::api::expr::Ex::to_srepr).
    ///
    /// ```
    /// use symplex::tree::ExprTree;
    ///
    /// let t = ExprTree::Pow {
    ///     base: Box::new(ExprTree::Symbol { name: "x".into() }),
    ///     exp: Box::new(ExprTree::Num { numer: "1".into(), denom: "2".into() }),
    /// };
    /// assert_eq!(t.to_srepr(), "Pow(Symbol('x'), Rational(1, 2))");
    /// ```
    #[must_use]
    pub fn to_srepr(&self) -> String {
        enum Item<'t> {
            Text(String),
            Node(&'t ExprTree),
        }
        let mut out = String::new();
        let mut stack: Vec<Item<'_>> = vec![Item::Node(self)];
        while let Some(item) = stack.pop() {
            match item {
                Item::Text(s) => out.push_str(&s),
                Item::Node(node) => {
                    let (head, children) = node.head_and_children();
                    out.push_str(&head);
                    let Some(children) = children else { continue };
                    out.push('(');
                    // Pushed in reverse so that popping yields left-to-right.
                    let mut close = String::new();
                    if let Some(trailing) = node.srepr_trailing() {
                        close.push_str(&trailing);
                    }
                    close.push(')');
                    stack.push(Item::Text(close));
                    let pairwise = matches!(node, ExprTree::Piecewise { .. });
                    for (i, child) in children.iter().enumerate().rev() {
                        if pairwise {
                            // `Piecewise((v1, c1), (v2, c2))`
                            if i % 2 == 1 {
                                stack.push(Item::Text(")".into()));
                                stack.push(Item::Node(child));
                                stack.push(Item::Text(", ".into()));
                            } else {
                                stack.push(Item::Node(child));
                                stack.push(Item::Text(if i == 0 { "(" } else { ", (" }.into()));
                            }
                        } else {
                            stack.push(Item::Node(child));
                            if i > 0 {
                                stack.push(Item::Text(", ".into()));
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// Graphviz `digraph` of this tree (SymPy: `dotprint`).  See
    /// [`Ex::to_dot`](crate::api::expr::Ex::to_dot).
    ///
    /// Node ids are assigned in pre-order (`n0` is the root, children
    /// left to right), so the output is deterministic; labels are the
    /// node kind with its value for atoms (`Symbol('x')`, `Integer(2)`).
    #[must_use]
    pub fn to_dot(&self) -> String {
        let mut nodes: Vec<String> = Vec::new();
        let mut edges: Vec<String> = Vec::new();
        // (node, parent id); pre-order with children pushed in reverse.
        let mut stack: Vec<(&ExprTree, Option<usize>)> = vec![(self, None)];
        while let Some((node, parent)) = stack.pop() {
            let id = nodes.len();
            let (head, children) = node.head_and_children();
            let label = match node {
                ExprTree::Interval { flags, .. } => format!(
                    "Interval('{}{}')",
                    if flags & 0x01 != 0 { '(' } else { '[' },
                    if flags & 0x02 != 0 { ')' } else { ']' }
                ),
                _ => head,
            };
            nodes.push(format!(
                "    n{id} [label=\"{}\"];",
                label.replace('\\', "\\\\").replace('"', "\\\"")
            ));
            if let Some(p) = parent {
                edges.push(format!("    n{p} -> n{id};"));
            }
            if let Some(children) = children {
                for child in children.into_iter().rev() {
                    stack.push((child, Some(id)));
                }
            }
        }
        let mut out = String::from("digraph {\n    ordering=out;\n    rankdir=TD;\n");
        for n in &nodes {
            out.push_str(n);
            out.push('\n');
        }
        for e in &edges {
            out.push_str(e);
            out.push('\n');
        }
        out.push_str("}\n");
        out
    }
}

impl<S: crate::api::expr::Sort> crate::api::expr::Expr<S> {
    /// SymPy-`srepr`-style constructor form: an unambiguous, parseable-by-eye
    /// rendering of the exact tree, derived from [`to_tree`](Self::to_tree)
    /// so it is total (SymPy: `srepr(expr)`).
    ///
    /// Atoms print as `Integer(2)`, `Rational(1, 2)`, `Symbol('x')`, `pi`,
    /// `E`, `I`, `oo`; compound nodes as `Head(child, …)` with SymPy's
    /// heads where they exist (`Add`, `Mul`, `Pow`, `sin`, `log`, `Abs`,
    /// `StrictGreaterThan`, `Interval(a, b, false, true)`, …) and symplex's
    /// otherwise (`Neg`, `DefiniteIntegral(f, x, a, b)`).  Library and
    /// user functions print as `name(args)`.  Children appear in the
    /// arena's canonical order (numbers first in a sum), not display order:
    /// this is the exact tree, as `to_tree`/`to_json` see it.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!((2 * &x + 1).to_srepr(), "Add(Integer(1), Mul(Integer(2), Symbol('x')))");
    /// assert_eq!((&x / 2).to_srepr(), "Mul(Rational(1, 2), Symbol('x'))");
    /// assert_eq!(x.sin().powi(2).to_srepr(), "Pow(sin(Symbol('x')), Integer(2))");
    /// assert_eq!(x.gt(&ctx.int(0)).to_srepr(), "StrictGreaterThan(Symbol('x'), Integer(0))");
    /// ```
    #[must_use = "returns the rendered string; does not modify in place"]
    pub fn to_srepr(&self) -> String {
        self.to_tree().to_srepr()
    }

    /// Graphviz DOT source for the expression tree (SymPy: `dotprint`).
    ///
    /// One node per tree position (labelled with the node kind and, for
    /// atoms, the value), one edge per child, ids `n0`, `n1`, … assigned
    /// in pre-order so the output is deterministic.  Render with
    /// `dot -Tsvg`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!(
    ///     (2 * &x + 1).to_dot(),
    ///     "digraph {\n\
    ///     \x20   ordering=out;\n\
    ///     \x20   rankdir=TD;\n\
    ///     \x20   n0 [label=\"Add\"];\n\
    ///     \x20   n1 [label=\"Integer(1)\"];\n\
    ///     \x20   n2 [label=\"Mul\"];\n\
    ///     \x20   n3 [label=\"Integer(2)\"];\n\
    ///     \x20   n4 [label=\"Symbol('x')\"];\n\
    ///     \x20   n0 -> n1;\n\
    ///     \x20   n0 -> n2;\n\
    ///     \x20   n2 -> n3;\n\
    ///     \x20   n2 -> n4;\n\
    ///     }\n"
    /// );
    /// ```
    #[must_use = "returns the rendered string; does not modify in place"]
    pub fn to_dot(&self) -> String {
        self.to_tree().to_dot()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::arena::Arena;

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

    #[test]
    fn roundtrip_definite_integral() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let body = a.pow(x, x);
        let one = a.one;
        let expr = a.definite_integral(body, x, a.zero, one);
        let tree = expr_to_tree(&a, expr);
        let json = serde_json::to_string(&tree).unwrap();
        let tree2: ExprTree = serde_json::from_str(&json).unwrap();
        assert_eq!(tree2, tree);
        assert_eq!(tree_to_expr(&mut a, &tree2), expr);
        assert_eq!(display(&a, expr), "Integral(x^x, x, 0, 1)");
    }

    // ── 0.2 nodes ──────────────────────────────────────────────────────────

    #[test]
    fn roundtrip_named_constants() {
        let mut a = Arena::new();
        for id in [a.euler_gamma, a.catalan, a.golden_ratio] {
            let tree = expr_to_tree(&a, id);
            let json = serde_json::to_string(&tree).unwrap();
            let tree2: ExprTree = serde_json::from_str(&json).unwrap();
            assert_eq!(tree2, tree);
            assert_eq!(tree_to_expr(&mut a, &tree2), id);
        }
        assert_eq!(expr_to_tree(&a, a.euler_gamma), ExprTree::EulerGamma);
        assert_eq!(expr_to_tree(&a, a.catalan), ExprTree::Catalan);
        assert_eq!(expr_to_tree(&a, a.golden_ratio), ExprTree::GoldenRatio);
    }

    #[test]
    fn roundtrip_complex_and_special_nodes() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let n = a.symbol("n");
        let nodes = [
            a.intern(ExprNode::Re(x)),
            a.intern(ExprNode::Im(x)),
            a.intern(ExprNode::Conjugate(x)),
            a.intern(ExprNode::Arg(x)),
            a.intern(ExprNode::Si(x)),
            a.intern(ExprNode::Ci(x)),
            a.intern(ExprNode::Ei(x)),
            a.intern(ExprNode::Li(x)),
            a.intern(ExprNode::Zeta(x)),
            a.intern(ExprNode::Polygamma(n, x)),
            a.intern(ExprNode::KroneckerDelta(n, x)),
        ];
        for id in nodes {
            let tree = expr_to_tree(&a, id);
            let json = serde_json::to_string(&tree).unwrap();
            let tree2: ExprTree = serde_json::from_str(&json).unwrap();
            let back = tree_to_expr(&mut a, &tree2);
            assert_eq!(back, id, "round trip of {}", display(&a, id));
        }
    }

    #[test]
    fn tree_to_expr_uses_canonical_constructors() {
        // Deserialising `Zeta(2)` folds to π²/6, like the constructor does.
        let mut a = Arena::new();
        let two = a.int(2);
        let tree = ExprTree::Zeta {
            arg: Box::new(expr_to_tree(&a, two)),
        };
        let id = tree_to_expr(&mut a, &tree);
        assert_eq!(display(&a, id), "1/6*pi^2");
    }
}
