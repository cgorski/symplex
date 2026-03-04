//! Core node types for the symplex symbolic expression graph.
//!
//! Every expression in symplex is represented as a node in a directed acyclic
//! graph (DAG).  Nodes are identified by [`ExprId`] handles that index into an
//! arena held by the expression context.  The actual payload of each node is
//! described by the [`ExprNode`] enum.
//!
//! Numeric literals and symbolic names are stored in separate side‐tables and
//! referenced via [`NumId`] and [`SymbolId`] respectively, keeping the core
//! node type small and cheap to clone.

use smallvec::{SmallVec, smallvec};
use std::fmt;

// ---------------------------------------------------------------------------
// Id newtypes
// ---------------------------------------------------------------------------

/// Opaque handle that identifies an expression node inside an expression
/// context.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExprId(pub u32);

impl fmt::Debug for ExprId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "e{}", self.0)
    }
}

/// Index into the numeric literal side‐table of an expression context.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NumId(pub u32);

impl fmt::Debug for NumId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n{}", self.0)
    }
}

/// Index into the symbol name table of an expression context.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SymbolId(pub u32);

impl fmt::Debug for SymbolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "s{}", self.0)
    }
}

/// Identifies an expression context (arena).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CtxId(pub u32);

impl fmt::Debug for CtxId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ctx{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// ExprNode
// ---------------------------------------------------------------------------

/// The payload of a single node in the expression DAG.
///
/// Variants fall into three broad categories:
///
/// * **Atoms** – leaves that carry no child expressions (`Num`, `Symbol`,
///   mathematical constants, and special values).
/// * **Operators** – internal nodes with one or more child [`ExprId`]s.
/// * **Calculus forms** – `Derivative` and `Integral`, which record both the
///   body expression and the variable of differentiation / integration.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum ExprNode {
    // -- atoms ---------------------------------------------------------------
    /// A numeric literal.  The actual value lives in the number side‐table
    /// and is looked up via the contained [`NumId`].
    Num(NumId),

    /// A symbolic name (variable, constant, or function name).  The actual
    /// string is stored in the symbol table and looked up via [`SymbolId`].
    Symbol(SymbolId),

    /// The mathematical constant π ≈ 3.14159…
    Pi,

    /// Euler's number *e* ≈ 2.71828…
    E,

    /// The imaginary unit *i*, satisfying *i*² = −1.
    ImaginaryUnit,

    /// Positive infinity (+∞).
    Infinity,

    /// Negative infinity (−∞).
    NegInfinity,

    /// Complex infinity (∞̃) — magnitude is infinite, direction is undefined.
    ComplexInfinity,

    /// Not‐a‐number, representing an indeterminate or undefined result.
    NaN,

    // -- n‐ary operators -----------------------------------------------------
    /// An n‐ary sum: `a + b + c + …`
    Add(SmallVec<[ExprId; 6]>),

    /// An n‐ary product: `a · b · c · …`
    Mul(SmallVec<[ExprId; 6]>),

    // -- binary operators ----------------------------------------------------
    /// Exponentiation: `base ^ exponent`.
    Pow(ExprId, ExprId),

    // -- unary operators -----------------------------------------------------
    /// Unary arithmetic negation: `−x`.
    Neg(ExprId),

    /// Sine function: `sin(x)`.
    Sin(ExprId),

    /// Cosine function: `cos(x)`.
    Cos(ExprId),

    /// Tangent function: `tan(x)`.
    Tan(ExprId),

    /// Natural exponential function: `eˣ`.
    Exp(ExprId),

    /// Natural logarithm: `ln(x)`.
    Ln(ExprId),

    /// Absolute value (or complex modulus): `|x|`.
    Abs(ExprId),

    /// Inverse sine: `asin(x)` (arcsin).
    Asin(ExprId),

    /// Inverse cosine: `acos(x)` (arccos).
    Acos(ExprId),

    /// Inverse tangent: `atan(x)` (arctan).
    Atan(ExprId),

    /// Hyperbolic sine: `sinh(x)`.
    Sinh(ExprId),

    /// Hyperbolic cosine: `cosh(x)`.
    Cosh(ExprId),

    /// Hyperbolic tangent: `tanh(x)`.
    Tanh(ExprId),

    /// Inverse hyperbolic sine: `asinh(x)`.
    Asinh(ExprId),

    /// Inverse hyperbolic cosine: `acosh(x)`.
    Acosh(ExprId),

    /// Inverse hyperbolic tangent: `atanh(x)`.
    Atanh(ExprId),

    // -- combinatorial -------------------------------------------------------
    /// Factorial: `n!`
    Factorial(ExprId),

    /// Binomial coefficient: `C(n, k)` = n! / (k! * (n-k)!)
    Binomial(ExprId, ExprId),

    // -- composite forms -----------------------------------------------------
    /// Application of a user‐defined or library function identified by
    /// [`SymbolId`] to a list of argument expressions.
    Apply(SymbolId, SmallVec<[ExprId; 2]>),

    /// Formal derivative: d/d(var) of an expression.
    ///
    /// `Derivative(body, var)` represents ∂/∂`var` `body`.
    Derivative(ExprId, ExprId),

    /// Formal indefinite integral: ∫ expr d(var).
    ///
    /// `Integral(body, var)` represents ∫ `body` d`var`.
    Integral(ExprId, ExprId),
}

impl ExprNode {
    /// Returns all child [`ExprId`]s contained in this node.
    ///
    /// For atoms (numbers, symbols, constants, and special values) the
    /// returned collection is empty.  For compound nodes the children are
    /// returned in the order they appear in the variant.
    pub fn children(&self) -> SmallVec<[ExprId; 6]> {
        match self {
            // atoms — no children
            ExprNode::Num(_)
            | ExprNode::Symbol(_)
            | ExprNode::Pi
            | ExprNode::E
            | ExprNode::ImaginaryUnit
            | ExprNode::Infinity
            | ExprNode::NegInfinity
            | ExprNode::ComplexInfinity
            | ExprNode::NaN => smallvec![],

            // n‐ary
            ExprNode::Add(ids) | ExprNode::Mul(ids) => ids.clone(),

            // binary
            ExprNode::Pow(a, b)
            | ExprNode::Binomial(a, b)
            | ExprNode::Derivative(a, b)
            | ExprNode::Integral(a, b) => {
                smallvec![*a, *b]
            }

            // unary
            ExprNode::Neg(x)
            | ExprNode::Sin(x)
            | ExprNode::Cos(x)
            | ExprNode::Tan(x)
            | ExprNode::Exp(x)
            | ExprNode::Ln(x)
            | ExprNode::Abs(x)
            | ExprNode::Asin(x)
            | ExprNode::Acos(x)
            | ExprNode::Atan(x)
            | ExprNode::Sinh(x)
            | ExprNode::Cosh(x)
            | ExprNode::Tanh(x)
            | ExprNode::Asinh(x)
            | ExprNode::Acosh(x)
            | ExprNode::Atanh(x)
            | ExprNode::Factorial(x) => smallvec![*x],

            // function application
            ExprNode::Apply(_, args) => {
                // Promote the SmallVec<[ExprId; 2]> into SmallVec<[ExprId; 6]>
                args.iter().copied().collect()
            }
        }
    }

    /// Returns `true` if this node is an atom (a leaf with no child
    /// expressions).
    ///
    /// Atoms are: [`Num`](ExprNode::Num), [`Symbol`](ExprNode::Symbol),
    /// [`Pi`](ExprNode::Pi), [`E`](ExprNode::E),
    /// [`ImaginaryUnit`](ExprNode::ImaginaryUnit),
    /// [`Infinity`](ExprNode::Infinity),
    /// [`NegInfinity`](ExprNode::NegInfinity),
    /// [`ComplexInfinity`](ExprNode::ComplexInfinity), and
    /// [`NaN`](ExprNode::NaN).
    pub fn is_atom(&self) -> bool {
        matches!(
            self,
            ExprNode::Num(_)
                | ExprNode::Symbol(_)
                | ExprNode::Pi
                | ExprNode::E
                | ExprNode::ImaginaryUnit
                | ExprNode::Infinity
                | ExprNode::NegInfinity
                | ExprNode::ComplexInfinity
                | ExprNode::NaN
        )
    }
}

impl fmt::Debug for ExprNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExprNode::Num(id) => write!(f, "Num({id:?})"),
            ExprNode::Symbol(id) => write!(f, "Symbol({id:?})"),
            ExprNode::Pi => write!(f, "Pi"),
            ExprNode::E => write!(f, "E"),
            ExprNode::ImaginaryUnit => write!(f, "ImaginaryUnit"),
            ExprNode::Infinity => write!(f, "Infinity"),
            ExprNode::NegInfinity => write!(f, "NegInfinity"),
            ExprNode::ComplexInfinity => write!(f, "ComplexInfinity"),
            ExprNode::NaN => write!(f, "NaN"),
            ExprNode::Add(ids) => f.debug_tuple("Add").field(ids).finish(),
            ExprNode::Mul(ids) => f.debug_tuple("Mul").field(ids).finish(),
            ExprNode::Pow(base, exp) => f.debug_tuple("Pow").field(base).field(exp).finish(),
            ExprNode::Neg(x) => f.debug_tuple("Neg").field(x).finish(),
            ExprNode::Sin(x) => f.debug_tuple("Sin").field(x).finish(),
            ExprNode::Cos(x) => f.debug_tuple("Cos").field(x).finish(),
            ExprNode::Tan(x) => f.debug_tuple("Tan").field(x).finish(),
            ExprNode::Exp(x) => f.debug_tuple("Exp").field(x).finish(),
            ExprNode::Ln(x) => f.debug_tuple("Ln").field(x).finish(),
            ExprNode::Abs(x) => f.debug_tuple("Abs").field(x).finish(),
            ExprNode::Asin(x) => f.debug_tuple("Asin").field(x).finish(),
            ExprNode::Acos(x) => f.debug_tuple("Acos").field(x).finish(),
            ExprNode::Atan(x) => f.debug_tuple("Atan").field(x).finish(),
            ExprNode::Sinh(x) => f.debug_tuple("Sinh").field(x).finish(),
            ExprNode::Cosh(x) => f.debug_tuple("Cosh").field(x).finish(),
            ExprNode::Tanh(x) => f.debug_tuple("Tanh").field(x).finish(),
            ExprNode::Asinh(x) => f.debug_tuple("Asinh").field(x).finish(),
            ExprNode::Acosh(x) => f.debug_tuple("Acosh").field(x).finish(),
            ExprNode::Atanh(x) => f.debug_tuple("Atanh").field(x).finish(),
            ExprNode::Factorial(id) => write!(f, "Factorial({id:?})"),
            ExprNode::Binomial(n, k) => write!(f, "Binomial({n:?}, {k:?})"),
            ExprNode::Apply(sym, args) => f.debug_tuple("Apply").field(sym).field(args).finish(),
            ExprNode::Derivative(body, var) => {
                f.debug_tuple("Derivative").field(body).field(var).finish()
            }
            ExprNode::Integral(body, var) => {
                f.debug_tuple("Integral").field(body).field(var).finish()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_expr_id() {
        let id = ExprId(42);
        assert_eq!(format!("{id:?}"), "e42");
    }

    #[test]
    fn debug_num_id() {
        let id = NumId(7);
        assert_eq!(format!("{id:?}"), "n7");
    }

    #[test]
    fn debug_symbol_id() {
        let id = SymbolId(3);
        assert_eq!(format!("{id:?}"), "s3");
    }

    #[test]
    fn debug_ctx_id() {
        let id = CtxId(0);
        assert_eq!(format!("{id:?}"), "ctx0");
    }

    #[test]
    fn atom_has_no_children() {
        let node = ExprNode::Pi;
        assert!(node.is_atom());
        assert!(node.children().is_empty());
    }

    #[test]
    fn num_is_atom() {
        let node = ExprNode::Num(NumId(0));
        assert!(node.is_atom());
        assert!(node.children().is_empty());
    }

    #[test]
    fn symbol_is_atom() {
        let node = ExprNode::Symbol(SymbolId(1));
        assert!(node.is_atom());
    }

    #[test]
    fn special_values_are_atoms() {
        for node in [
            ExprNode::Infinity,
            ExprNode::NegInfinity,
            ExprNode::ComplexInfinity,
            ExprNode::NaN,
            ExprNode::ImaginaryUnit,
            ExprNode::E,
        ] {
            assert!(node.is_atom(), "{node:?} should be an atom");
            assert!(node.children().is_empty());
        }
    }

    #[test]
    fn add_children() {
        let ids: SmallVec<[ExprId; 6]> = smallvec![ExprId(1), ExprId(2), ExprId(3)];
        let node = ExprNode::Add(ids.clone());
        assert!(!node.is_atom());
        assert_eq!(node.children(), ids);
    }

    #[test]
    fn mul_children() {
        let ids: SmallVec<[ExprId; 6]> = smallvec![ExprId(4), ExprId(5)];
        let node = ExprNode::Mul(ids.clone());
        assert!(!node.is_atom());
        assert_eq!(node.children(), ids);
    }

    #[test]
    fn pow_children() {
        let node = ExprNode::Pow(ExprId(10), ExprId(20));
        assert!(!node.is_atom());
        let kids = node.children();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0], ExprId(10));
        assert_eq!(kids[1], ExprId(20));
    }

    #[test]
    fn unary_children() {
        for node in [
            ExprNode::Neg(ExprId(1)),
            ExprNode::Sin(ExprId(1)),
            ExprNode::Cos(ExprId(1)),
            ExprNode::Tan(ExprId(1)),
            ExprNode::Exp(ExprId(1)),
            ExprNode::Ln(ExprId(1)),
            ExprNode::Abs(ExprId(1)),
        ] {
            assert!(!node.is_atom());
            let kids = node.children();
            assert_eq!(kids.len(), 1, "{node:?} should have exactly 1 child");
            assert_eq!(kids[0], ExprId(1));
        }
    }

    #[test]
    fn apply_children() {
        let args: SmallVec<[ExprId; 2]> = smallvec![ExprId(5), ExprId(6)];
        let node = ExprNode::Apply(SymbolId(0), args);
        assert!(!node.is_atom());
        let kids = node.children();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0], ExprId(5));
        assert_eq!(kids[1], ExprId(6));
    }

    #[test]
    fn derivative_children() {
        let node = ExprNode::Derivative(ExprId(3), ExprId(7));
        assert!(!node.is_atom());
        let kids = node.children();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0], ExprId(3));
        assert_eq!(kids[1], ExprId(7));
    }

    #[test]
    fn integral_children() {
        let node = ExprNode::Integral(ExprId(8), ExprId(9));
        assert!(!node.is_atom());
        let kids = node.children();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0], ExprId(8));
        assert_eq!(kids[1], ExprId(9));
    }
}
