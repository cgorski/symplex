//! Canonical ordering support for expression nodes.
//!
//! [`SortKey`] is a compact, lexicographically comparable byte sequence that
//! defines a total order over expression nodes.  The ordering is designed so
//! that structurally simpler terms precede complex ones:
//!
//! 1. **Numbers** (lowest rank) — ordered by their serialised rational value.
//! 2. **Symbols** — ordered alphabetically by name.
//! 3. **Pow**, **Mul**, **Add** — ordered by the sort keys of their children.
//! 4. **Functions** (Sin, Cos, …, Apply) — distinguished by a sub-rank byte,
//!    then ordered by the sort key of their argument(s).
//! 5. **Derivative**, **Integral** — ordered by body then variable.
//! 6. **Constants** (Pi, E, ImaginaryUnit) — each gets a unique sub-rank.
//! 7. **Specials** (Infinity, NegInfinity, ComplexInfinity, NaN, Neg) — highest
//!    rank, each with a unique sub-rank.
//!
//! The [`compute_sort_key`] function builds a [`SortKey`] for a given
//! [`ExprNode`] by recursively combining the keys of its children via
//! caller-provided closures.  This keeps the module decoupled from any
//! particular arena or storage backend.

use std::fmt;

use smallvec::SmallVec;

use crate::node::{ExprId, ExprNode, NumId, SymbolId};

// ---------------------------------------------------------------------------
// Class rank constants
// ---------------------------------------------------------------------------

/// Rank byte for numeric literals — sorts first.
const RANK_NUM: u8 = 0;

/// Rank byte for symbolic names — sorts after numbers.
const RANK_SYMBOL: u8 = 10;

/// Rank byte for exponentiation nodes.
const RANK_POW: u8 = 20;

/// Rank byte for multiplication (product) nodes.
const RANK_MUL: u8 = 30;

/// Rank byte for addition (sum) nodes.
const RANK_ADD: u8 = 40;

/// Rank byte for built-in and user-defined function applications
/// (Sin, Cos, Tan, Exp, Ln, Sqrt, Abs, Apply).
const RANK_FUNCTION: u8 = 50;

/// Rank byte for formal derivative nodes.
const RANK_DERIVATIVE: u8 = 60;

/// Rank byte for formal integral nodes.
const RANK_INTEGRAL: u8 = 70;

/// Rank byte for mathematical constants (Pi, E, ImaginaryUnit).
const RANK_CONSTANT: u8 = 80;

/// Rank byte for special sentinel values
/// (Infinity, NegInfinity, ComplexInfinity, NaN, Neg).
const RANK_SPECIAL: u8 = 90;

// ---------------------------------------------------------------------------
// Function discriminant bytes (used within the RANK_FUNCTION class)
// ---------------------------------------------------------------------------

const FN_SIN: u8 = 0;
const FN_COS: u8 = 1;
const FN_TAN: u8 = 2;
const FN_EXP: u8 = 3;
const FN_LN: u8 = 4;
const FN_ABS: u8 = 6;
const FN_ASIN: u8 = 8;
const FN_ACOS: u8 = 9;
const FN_ATAN: u8 = 10;
const FN_SINH: u8 = 11;
const FN_COSH: u8 = 12;
const FN_TANH: u8 = 13;
const FN_ASINH: u8 = 14;
const FN_ACOSH: u8 = 15;
const FN_ATANH: u8 = 16;
const FN_APPLY: u8 = 17;
const FN_SIGN: u8 = 18;

// ---------------------------------------------------------------------------
// Constant sub-rank bytes (used within the RANK_CONSTANT class)
// ---------------------------------------------------------------------------

const CONST_PI: u8 = 0;
const CONST_E: u8 = 1;
const CONST_IMAGINARY_UNIT: u8 = 2;

// ---------------------------------------------------------------------------
// Special sub-rank bytes (used within the RANK_SPECIAL class)
// ---------------------------------------------------------------------------

const SPECIAL_INFINITY: u8 = 0;
const SPECIAL_NEG_INFINITY: u8 = 1;
const SPECIAL_COMPLEX_INFINITY: u8 = 2;
const SPECIAL_NAN: u8 = 3;
const SPECIAL_NEG: u8 = 4;

// ---------------------------------------------------------------------------
// SortKey
// ---------------------------------------------------------------------------

/// A compact byte sequence whose lexicographic order defines the canonical
/// ordering of expression nodes.
///
/// The inner [`SmallVec`] is stack-allocated for keys up to 24 bytes, which
/// covers the vast majority of leaf and simple composite nodes without a heap
/// allocation.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SortKey(SmallVec<[u8; 24]>);

impl SortKey {
    /// Creates an empty sort key.
    #[inline]
    fn new() -> Self {
        SortKey(SmallVec::new())
    }

    /// Pushes a single byte onto the key.
    #[inline]
    fn push(&mut self, byte: u8) {
        self.0.push(byte);
    }

    /// Appends a byte slice onto the key.
    #[inline]
    fn extend(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }

    /// Returns the key as a byte slice.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl PartialOrd for SortKey {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SortKey {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.as_slice().cmp(other.0.as_slice())
    }
}

impl fmt::Debug for SortKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SortKey({:?})", self.0.as_slice())
    }
}

// ---------------------------------------------------------------------------
// compute_sort_key
// ---------------------------------------------------------------------------

/// Computes the [`SortKey`] for a single [`ExprNode`].
///
/// The three closures abstract over whatever arena or context owns the actual
/// data:
///
/// * `get_key` — returns the (already computed) sort key for a child
///   expression identified by [`ExprId`].
/// * `get_num_bytes` — serialises the rational value behind a [`NumId`] into a
///   byte sequence suitable for lexicographic comparison.
/// * `get_sym_name` — returns the interned name of a [`SymbolId`].
///
/// # Ordering summary
///
/// | Rank | Node kind(s)                                       |
/// |------|----------------------------------------------------|
/// |  0   | `Num`                                              |
/// | 10   | `Symbol`                                           |
/// | 20   | `Pow`                                              |
/// | 30   | `Mul`                                              |
/// | 40   | `Add`                                              |
/// | 50   | `Sin`, `Cos`, `Tan`, `Exp`, `Ln`, `Abs`, `Asin`, `Acos`, `Atan`, `Sinh`, `Cosh`, `Tanh`, `Asinh`, `Acosh`, `Atanh`, `Apply` |
/// | 60   | `Derivative`                                       |
/// | 70   | `Integral`                                         |
/// | 80   | `Pi`, `E`, `ImaginaryUnit`                         |
/// | 90   | `Infinity`, `NegInfinity`, `ComplexInfinity`, `NaN`, `Neg` |
pub fn compute_sort_key(
    node: &ExprNode,
    get_key: impl Fn(ExprId) -> SortKey,
    get_num_bytes: impl Fn(NumId) -> Vec<u8>,
    get_sym_name: impl Fn(SymbolId) -> String,
) -> SortKey {
    let mut key = SortKey::new();

    match node {
        // -- atoms -----------------------------------------------------------
        ExprNode::Num(id) => {
            key.push(RANK_NUM);
            key.extend(&get_num_bytes(*id));
        }

        ExprNode::Symbol(id) => {
            key.push(RANK_SYMBOL);
            key.extend(get_sym_name(*id).as_bytes());
        }

        // -- n-ary operators -------------------------------------------------
        ExprNode::Add(children) => {
            key.push(RANK_ADD);
            for &child in children {
                key.extend(get_key(child).as_bytes());
            }
        }

        ExprNode::Mul(children) => {
            key.push(RANK_MUL);
            for &child in children {
                key.extend(get_key(child).as_bytes());
            }
        }

        // -- binary operators ------------------------------------------------
        ExprNode::Pow(base, exp) => {
            key.push(RANK_POW);
            key.extend(get_key(*base).as_bytes());
            key.extend(get_key(*exp).as_bytes());
        }

        ExprNode::Derivative(body, var) => {
            key.push(RANK_DERIVATIVE);
            key.extend(get_key(*body).as_bytes());
            key.extend(get_key(*var).as_bytes());
        }

        ExprNode::Integral(body, var) => {
            key.push(RANK_INTEGRAL);
            key.extend(get_key(*body).as_bytes());
            key.extend(get_key(*var).as_bytes());
        }

        // -- unary functions -------------------------------------------------
        ExprNode::Sin(x) => {
            key.push(RANK_FUNCTION);
            key.push(FN_SIN);
            key.extend(get_key(*x).as_bytes());
        }

        ExprNode::Cos(x) => {
            key.push(RANK_FUNCTION);
            key.push(FN_COS);
            key.extend(get_key(*x).as_bytes());
        }

        ExprNode::Tan(x) => {
            key.push(RANK_FUNCTION);
            key.push(FN_TAN);
            key.extend(get_key(*x).as_bytes());
        }

        ExprNode::Exp(x) => {
            key.push(RANK_FUNCTION);
            key.push(FN_EXP);
            key.extend(get_key(*x).as_bytes());
        }

        ExprNode::Ln(x) => {
            key.push(RANK_FUNCTION);
            key.push(FN_LN);
            key.extend(get_key(*x).as_bytes());
        }

        ExprNode::Abs(x) => {
            key.push(RANK_FUNCTION);
            key.push(FN_ABS);
            key.extend(get_key(*x).as_bytes());
        }

        ExprNode::Asin(x) => {
            key.push(RANK_FUNCTION);
            key.push(FN_ASIN);
            key.extend(get_key(*x).as_bytes());
        }

        ExprNode::Acos(x) => {
            key.push(RANK_FUNCTION);
            key.push(FN_ACOS);
            key.extend(get_key(*x).as_bytes());
        }

        ExprNode::Atan(x) => {
            key.push(RANK_FUNCTION);
            key.push(FN_ATAN);
            key.extend(get_key(*x).as_bytes());
        }

        ExprNode::Sinh(x) => {
            key.push(RANK_FUNCTION);
            key.push(FN_SINH);
            key.extend(get_key(*x).as_bytes());
        }

        ExprNode::Cosh(x) => {
            key.push(RANK_FUNCTION);
            key.push(FN_COSH);
            key.extend(get_key(*x).as_bytes());
        }

        ExprNode::Tanh(x) => {
            key.push(RANK_FUNCTION);
            key.push(FN_TANH);
            key.extend(get_key(*x).as_bytes());
        }

        ExprNode::Asinh(x) => {
            key.push(RANK_FUNCTION);
            key.push(FN_ASINH);
            key.extend(get_key(*x).as_bytes());
        }

        ExprNode::Acosh(x) => {
            key.push(RANK_FUNCTION);
            key.push(FN_ACOSH);
            key.extend(get_key(*x).as_bytes());
        }

        ExprNode::Atanh(x) => {
            key.push(RANK_FUNCTION);
            key.push(FN_ATANH);
            key.extend(get_key(*x).as_bytes());
        }

        ExprNode::Sign(x) => {
            key.push(RANK_FUNCTION);
            key.push(FN_SIGN);
            key.extend(get_key(*x).as_bytes());
        }

        ExprNode::Apply(sym, args) => {
            key.push(RANK_FUNCTION);
            key.push(FN_APPLY);
            // Embed the function name so that distinct named functions sort
            // alphabetically among themselves.
            key.extend(get_sym_name(*sym).as_bytes());
            // Null byte separator to avoid ambiguity between name and args.
            key.push(0x00);
            for &arg in args {
                key.extend(get_key(arg).as_bytes());
            }
        }

        // -- constants -------------------------------------------------------
        ExprNode::Pi => {
            key.push(RANK_CONSTANT);
            key.push(CONST_PI);
        }

        ExprNode::E => {
            key.push(RANK_CONSTANT);
            key.push(CONST_E);
        }

        ExprNode::ImaginaryUnit => {
            key.push(RANK_CONSTANT);
            key.push(CONST_IMAGINARY_UNIT);
        }

        // -- special values --------------------------------------------------
        ExprNode::Infinity => {
            key.push(RANK_SPECIAL);
            key.push(SPECIAL_INFINITY);
        }

        ExprNode::NegInfinity => {
            key.push(RANK_SPECIAL);
            key.push(SPECIAL_NEG_INFINITY);
        }

        ExprNode::ComplexInfinity => {
            key.push(RANK_SPECIAL);
            key.push(SPECIAL_COMPLEX_INFINITY);
        }

        ExprNode::NaN => {
            key.push(RANK_SPECIAL);
            key.push(SPECIAL_NAN);
        }

        ExprNode::Neg(x) => {
            key.push(RANK_SPECIAL);
            key.push(SPECIAL_NEG);
            key.extend(get_key(*x).as_bytes());
        }

        // -- combinatorial ---------------------------------------------------
        ExprNode::Factorial(x) => {
            key.push(RANK_FUNCTION);
            key.push(FN_ABS + 1); // slot after last built-in function
            key.extend(get_key(*x).as_bytes());
        }

        ExprNode::Binomial(n, k) => {
            key.push(RANK_FUNCTION);
            key.push(FN_ABS + 2);
            key.extend(get_key(*n).as_bytes());
            key.extend(get_key(*k).as_bytes());
        }
    }

    key
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    /// Helper: builds a sort key for an atom with dummy closures.
    fn atom_key(node: &ExprNode) -> SortKey {
        compute_sort_key(
            node,
            |_| unreachable!("atom should not query children"),
            |_| vec![0x42],
            |id| format!("s{}", id.0),
        )
    }

    #[test]
    fn numbers_sort_before_symbols() {
        let num_key = atom_key(&ExprNode::Num(NumId(0)));
        let sym_key = atom_key(&ExprNode::Symbol(SymbolId(0)));
        assert!(num_key < sym_key, "Num should sort before Symbol");
    }

    #[test]
    fn symbols_sort_alphabetically() {
        let key_a = compute_sort_key(
            &ExprNode::Symbol(SymbolId(0)),
            |_| unreachable!(),
            |_| unreachable!(),
            |_| "alpha".to_string(),
        );
        let key_b = compute_sort_key(
            &ExprNode::Symbol(SymbolId(1)),
            |_| unreachable!(),
            |_| unreachable!(),
            |_| "beta".to_string(),
        );
        assert!(key_a < key_b);
    }

    #[test]
    fn constants_sort_after_composites() {
        let add_key = compute_sort_key(
            &ExprNode::Add(smallvec![ExprId(0)]),
            |_| SortKey(SmallVec::from_slice(&[RANK_NUM, 0x01])),
            |_| unreachable!(),
            |_| unreachable!(),
        );
        let pi_key = atom_key(&ExprNode::Pi);
        assert!(add_key < pi_key, "Add should sort before Pi");
    }

    #[test]
    fn constant_sub_ranks_are_distinct() {
        let pi = atom_key(&ExprNode::Pi);
        let e = atom_key(&ExprNode::E);
        let i = atom_key(&ExprNode::ImaginaryUnit);
        assert!(pi < e);
        assert!(e < i);
    }

    #[test]
    fn special_values_sort_last() {
        let pi_key = atom_key(&ExprNode::Pi);
        let inf_key = atom_key(&ExprNode::Infinity);
        assert!(pi_key < inf_key, "Constants should sort before specials");
    }

    #[test]
    fn function_discriminants_differ() {
        let sin_key = compute_sort_key(
            &ExprNode::Sin(ExprId(0)),
            |_| SortKey(SmallVec::from_slice(&[RANK_NUM, 0x01])),
            |_| unreachable!(),
            |_| unreachable!(),
        );
        let cos_key = compute_sort_key(
            &ExprNode::Cos(ExprId(0)),
            |_| SortKey(SmallVec::from_slice(&[RANK_NUM, 0x01])),
            |_| unreachable!(),
            |_| unreachable!(),
        );
        assert_ne!(sin_key, cos_key);
        assert!(
            sin_key < cos_key,
            "Sin (discriminant 0) < Cos (discriminant 1)"
        );
    }

    #[test]
    fn sort_key_ord_is_consistent() {
        let a = SortKey(SmallVec::from_slice(&[10, 20]));
        let b = SortKey(SmallVec::from_slice(&[10, 30]));
        let c = SortKey(SmallVec::from_slice(&[20]));
        assert!(a < b);
        assert!(b < c);
        assert!(a < c);
    }

    #[test]
    fn neg_is_special_rank() {
        let neg_key = compute_sort_key(
            &ExprNode::Neg(ExprId(0)),
            |_| SortKey(SmallVec::from_slice(&[RANK_NUM, 0x01])),
            |_| unreachable!(),
            |_| unreachable!(),
        );
        assert_eq!(neg_key.as_bytes()[0], RANK_SPECIAL);
        assert_eq!(neg_key.as_bytes()[1], SPECIAL_NEG);
    }
}
