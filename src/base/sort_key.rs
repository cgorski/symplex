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
//! 5. **Derivative**, **Integral**, **DefiniteIntegral** — ordered by body,
//!    then variable (then bounds).
//! 6. **Constants** (Pi, E, ImaginaryUnit) — each gets a unique sub-rank.
//! 7. **Specials** (Infinity, NegInfinity, ComplexInfinity, NaN, Neg) — highest
//!    rank, each with a unique sub-rank.
//!
//! The [`compute_sort_key`] function assembles the key of a given
//! [`ExprNode`] from the (already computed) keys of its children, which
//! caller-provided closures lend out, and [`push_bounded_key`] stores it.
//! This keeps the module decoupled from any particular arena or storage
//! backend.
//!
//! # Bounded size
//!
//! A key is the rank byte(s) of the node followed by the keys of its
//! children.  Left unbounded, that concatenation grows with the size of the
//! *unfolded tree*, not the hash-consed DAG: for `e ← sin(e) + cos(e)` the
//! key doubles at every step and construction becomes exponential.  Keys are
//! therefore capped at [`MAX_KEY_BYTES`]; a key that would be longer is cut
//! to that prefix and suffixed with a 64-bit structural digest of the whole
//! (pre-truncation) byte string.  Because a child's stored key already
//! carries its own digest when it was truncated, the digest depends only on
//! the expression's structure — never on arena allocation order — so the
//! resulting order is a deterministic total preorder that agrees with the
//! unbounded lexicographic order whenever two keys differ within the first
//! [`MAX_KEY_BYTES`] bytes.

use std::hash::Hasher;

use crate::base::node::{ExprId, ExprNode, NumId, SymbolId};

// ---------------------------------------------------------------------------
// Class rank constants
// ---------------------------------------------------------------------------

/// Rank byte for numeric literals — sorts first.
const RANK_NUM: u8 = 0;

/// Rank byte for symbolic names — sorts after numbers.
const RANK_SYMBOL: u8 = 20;

/// Rank byte for exponentiation nodes.
const RANK_POW: u8 = 40;

/// Rank byte for multiplication (product) nodes.
const RANK_MUL: u8 = 60;

/// Rank byte for addition (sum) nodes.
const RANK_ADD: u8 = 80;

/// Rank byte for built-in and user-defined function applications
/// (Sin, Cos, Tan, Exp, Ln, Sqrt, Abs, Apply).
const RANK_FUNCTION: u8 = 100;

/// Rank byte for relational operators (Gt, Ge, Eq_, Ne).
const RANK_RELATIONAL: u8 = 110;

/// Rank byte for logical conjunction (And).
const RANK_AND: u8 = 112;

/// Rank byte for logical disjunction (Or).
const RANK_OR: u8 = 114;

/// Rank byte for logical negation (Not).
const RANK_NOT: u8 = 116;

/// Rank byte for piecewise functions.
const RANK_PIECEWISE: u8 = 118;

/// Rank byte for formal derivative nodes.
const RANK_DERIVATIVE: u8 = 120;

/// Rank byte for n-ary minimum.
const RANK_MIN: u8 = 130;

/// Rank byte for n-ary maximum.
const RANK_MAX: u8 = 132;

/// Rank byte for formal integral nodes.
const RANK_INTEGRAL: u8 = 140;

/// Rank byte for formal definite integral nodes.
const RANK_DEFINITE_INTEGRAL: u8 = 142;

/// Rank byte for symbolic summation.
const RANK_SUM: u8 = 150;

/// Rank byte for symbolic product.
const RANK_PRODUCT: u8 = 152;

/// Rank byte for formal limit nodes.
const RANK_LIMIT: u8 = 154;

/// Rank byte for formal series expansion nodes.
const RANK_SERIES: u8 = 155;

/// Rank byte for formal Laplace transform nodes.
const RANK_LAPLACE_TRANSFORM: u8 = 156;

/// Rank byte for formal inverse Laplace transform nodes.
const RANK_INV_LAPLACE_TRANSFORM: u8 = 157;

/// Rank byte for formal residue nodes.
const RANK_RESIDUE: u8 = 158;

/// Rank byte for RootOf nodes.
const RANK_ROOTOF: u8 = 160;

/// Rank byte for formal DSolve nodes.
const RANK_DSOLVE: u8 = 162;

/// Rank byte for RootSum nodes.
const RANK_ROOTSUM: u8 = 163;

/// Rank byte for ConditionSet nodes.
const RANK_CONDITION_SET: u8 = 164;

/// Rank byte for Subs nodes (evaluation at a point).
const RANK_SUBS: u8 = 165;

/// Rank byte for mathematical constants (Pi, E, ImaginaryUnit).
const RANK_CONSTANT: u8 = 170;

/// Rank byte for set-valued nodes (EmptySet, UniversalSet, Interval, FiniteSet,
/// SetUnion, SetIntersection, SetComplement, ImageSet).
const RANK_SET: u8 = 190;

/// Rank byte for special sentinel values
/// (Infinity, NegInfinity, ComplexInfinity, NaN, Neg).
const RANK_SPECIAL: u8 = 210;

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
const FN_ATAN2: u8 = 19;
const FN_FLOOR: u8 = 20;
const FN_CEILING: u8 = 21;
const FN_GAMMA: u8 = 22;
const FN_LOG_GAMMA: u8 = 23;
const FN_DIGAMMA: u8 = 24;
const FN_ERF: u8 = 25;
const FN_ERFC: u8 = 26;
const FN_BETA: u8 = 27;
const FN_HEAVISIDE: u8 = 28;
const FN_DIRAC_DELTA: u8 = 29;
const FN_LAMBERT_W: u8 = 30;
const FN_RE: u8 = 31;
const FN_IM: u8 = 32;
const FN_CONJUGATE: u8 = 33;
const FN_ARG: u8 = 34;
const FN_SI: u8 = 35;
const FN_CI: u8 = 36;
const FN_EI: u8 = 37;
const FN_LI: u8 = 38;
const FN_ZETA: u8 = 39;
const FN_POLYGAMMA: u8 = 40;
const FN_KRONECKER_DELTA: u8 = 41;

// ---------------------------------------------------------------------------
// Constant sub-rank bytes (used within the RANK_CONSTANT class)
// ---------------------------------------------------------------------------

const CONST_PI: u8 = 0;
const CONST_E: u8 = 1;
const CONST_IMAGINARY_UNIT: u8 = 2;
const CONST_PHYSICAL: u8 = 3;
const CONST_EULER_GAMMA: u8 = 4;
const CONST_CATALAN: u8 = 5;
const CONST_GOLDEN_RATIO: u8 = 6;
const CONST_BOOL_TRUE: u8 = 10;
const CONST_BOOL_FALSE: u8 = 11;

// ---------------------------------------------------------------------------
// Special sub-rank bytes (used within the RANK_SPECIAL class)
// ---------------------------------------------------------------------------

const SPECIAL_INFINITY: u8 = 0;
const SPECIAL_NEG_INFINITY: u8 = 1;
const SPECIAL_COMPLEX_INFINITY: u8 = 2;
const SPECIAL_NAN: u8 = 3;
const SPECIAL_NEG: u8 = 4;

// ---------------------------------------------------------------------------
// Set sub-rank bytes (used within the RANK_SET class)
// ---------------------------------------------------------------------------

const SET_EMPTY: u8 = 0;
const SET_UNIVERSAL: u8 = 1;
const SET_INTERVAL: u8 = 2;
const SET_FINITE_SET: u8 = 3;
const SET_UNION: u8 = 4;
const SET_INTERSECTION: u8 = 5;
const SET_COMPLEMENT: u8 = 6;
const SET_IMAGE: u8 = 7;

// ---------------------------------------------------------------------------
// SortKey
// ---------------------------------------------------------------------------

/// Maximum number of structural bytes kept in a [`SortKey`].
///
/// Keys longer than this are truncated and suffixed with an 8-byte digest
/// (see the module docs), so no stored key exceeds `MAX_KEY_BYTES + 8`
/// bytes.
pub const MAX_KEY_BYTES: usize = 256;

/// A sort key: a byte string whose lexicographic order defines the canonical
/// ordering of expression nodes.  No key is longer than
/// [`MAX_KEY_BYTES`]` + 8`.
///
/// The arena stores the keys of all its nodes one after the other in one
/// byte pool and lends them out as slices (`Arena::sort_key`); up to 0.49
/// each node owned a key object, and half of them (the ones longer than 24
/// bytes) a heap allocation.
pub type SortKey = [u8];

// ---------------------------------------------------------------------------
// compute_sort_key
// ---------------------------------------------------------------------------

/// Appends the full (unbounded) [`SortKey`] of a single [`ExprNode`] to
/// `out`; [`push_bounded_key`] then stores it, cut to the size bound.
///
/// The three closures abstract over whatever arena or context owns the actual
/// data:
///
/// * `get_key` — lends the (already computed, bounded) sort key of a child
///   expression identified by [`ExprId`].
/// * `write_num` — appends the rational value behind a [`NumId`] in decimal
///   (`-3`, `7/2`).
/// * `get_sym_name` — lends the interned name of a [`SymbolId`].
///
/// # Ordering summary
///
/// | Rank | Node kind(s)                                       |
/// |------|----------------------------------------------------|
/// |   0  | `Num`                                              |
/// |  20  | `Symbol`                                           |
/// |  40  | `Pow`                                              |
/// |  60  | `Mul`                                              |
/// |  80  | `Add`                                              |
/// | 100  | `Sin`, `Cos`, `Tan`, `Exp`, `Ln`, `Abs`, `Asin`, `Acos`, `Atan`, `Sinh`, `Cosh`, `Tanh`, `Asinh`, `Acosh`, `Atanh`, `Apply`, `Lib`, special functions, `Re`, `Im`, `Conjugate`, `Arg` |
/// | 110  | `Gt`, `Ge`, `Eq_`, `Ne` (relational)               |
/// | 112  | `And`                                              |
/// | 114  | `Or`                                               |
/// | 116  | `Not`                                              |
/// | 118  | `Piecewise`                                        |
/// | 120  | `Derivative`                                       |
/// | 130  | `Min`                                              |
/// | 132  | `Max`                                              |
/// | 140  | `Integral`                                         |
/// | 142  | `DefiniteIntegral`                                 |
/// | 150  | `Sum`                                              |
/// | 152  | `Product_`                                         |
/// | 170  | `Pi`, `E`, `ImaginaryUnit`, `EulerGamma`, `Catalan`, `GoldenRatio` |
/// | 190  | `EmptySet`, `UniversalSet`, `Interval`, `FiniteSet`, `SetUnion`, `SetIntersection`, `SetComplement` |
/// | 210  | `Infinity`, `NegInfinity`, `ComplexInfinity`, `NaN`, `Neg` |
pub fn compute_sort_key<'k>(
    node: &ExprNode,
    out: &mut Vec<u8>,
    get_key: impl Fn(ExprId) -> &'k SortKey,
    write_num: impl Fn(NumId, &mut Vec<u8>),
    get_sym_name: impl Fn(SymbolId) -> &'k str,
) {
    match node {
        // -- atoms -----------------------------------------------------------
        ExprNode::Num(id) => {
            out.push(RANK_NUM);
            write_num(*id, out);
        }

        ExprNode::Symbol(id) => {
            out.push(RANK_SYMBOL);
            out.extend_from_slice(get_sym_name(*id).as_bytes());
        }

        // -- n-ary operators -------------------------------------------------
        ExprNode::Add(children) => {
            out.push(RANK_ADD);
            for &child in children {
                out.extend_from_slice(get_key(child));
            }
        }

        ExprNode::Mul(children) => {
            out.push(RANK_MUL);
            for &child in children {
                out.extend_from_slice(get_key(child));
            }
        }

        // -- binary operators ------------------------------------------------
        ExprNode::Pow(base, exp) => {
            out.push(RANK_POW);
            out.extend_from_slice(get_key(*base));
            out.extend_from_slice(get_key(*exp));
        }

        ExprNode::Derivative(body, var) => {
            out.push(RANK_DERIVATIVE);
            out.extend_from_slice(get_key(*body));
            out.extend_from_slice(get_key(*var));
        }

        ExprNode::Integral(body, var) => {
            out.push(RANK_INTEGRAL);
            out.extend_from_slice(get_key(*body));
            out.extend_from_slice(get_key(*var));
        }

        ExprNode::DefiniteIntegral(body, var, lo, hi) => {
            out.push(RANK_DEFINITE_INTEGRAL);
            out.extend_from_slice(get_key(*body));
            out.extend_from_slice(get_key(*var));
            out.extend_from_slice(get_key(*lo));
            out.extend_from_slice(get_key(*hi));
        }

        // -- unary functions -------------------------------------------------
        ExprNode::Sin(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_SIN);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Cos(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_COS);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Tan(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_TAN);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Exp(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_EXP);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Ln(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_LN);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Abs(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_ABS);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Asin(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_ASIN);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Acos(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_ACOS);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Atan(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_ATAN);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Atan2(y, x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_ATAN2);
            out.extend_from_slice(get_key(*y));
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Sinh(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_SINH);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Cosh(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_COSH);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Tanh(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_TANH);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Asinh(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_ASINH);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Acosh(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_ACOSH);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Atanh(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_ATANH);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Sign(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_SIGN);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Heaviside(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_HEAVISIDE);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::DiracDelta(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_DIRAC_DELTA);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Gamma(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_GAMMA);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::LogGamma(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_LOG_GAMMA);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Digamma(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_DIGAMMA);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Erf(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_ERF);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Erfc(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_ERFC);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::LambertW(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_LAMBERT_W);
            out.extend_from_slice(get_key(*x));
        }

        // -- complex analysis ------------------------------------------------
        ExprNode::Re(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_RE);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Im(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_IM);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Conjugate(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_CONJUGATE);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Arg(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_ARG);
            out.extend_from_slice(get_key(*x));
        }

        // -- special functions (0.2) -----------------------------------------
        ExprNode::Si(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_SI);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Ci(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_CI);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Ei(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_EI);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Li(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_LI);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Zeta(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_ZETA);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Polygamma(n, x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_POLYGAMMA);
            out.extend_from_slice(get_key(*n));
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::KroneckerDelta(i, j) => {
            out.push(RANK_FUNCTION);
            out.push(FN_KRONECKER_DELTA);
            out.extend_from_slice(get_key(*i));
            out.extend_from_slice(get_key(*j));
        }

        ExprNode::Beta(a, b) => {
            out.push(RANK_FUNCTION);
            out.push(FN_BETA);
            out.extend_from_slice(get_key(*a));
            out.extend_from_slice(get_key(*b));
        }

        ExprNode::Floor(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_FLOOR);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Ceiling(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_CEILING);
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Min(children) => {
            out.push(RANK_MIN);
            for &child in children {
                out.extend_from_slice(get_key(child));
            }
        }

        ExprNode::Max(children) => {
            out.push(RANK_MAX);
            for &child in children {
                out.extend_from_slice(get_key(child));
            }
        }

        ExprNode::Sum(body, var, lo, hi) => {
            out.push(RANK_SUM);
            out.extend_from_slice(get_key(*body));
            out.extend_from_slice(get_key(*var));
            out.extend_from_slice(get_key(*lo));
            out.extend_from_slice(get_key(*hi));
        }

        ExprNode::Product_(body, var, lo, hi) => {
            out.push(RANK_PRODUCT);
            out.extend_from_slice(get_key(*body));
            out.extend_from_slice(get_key(*var));
            out.extend_from_slice(get_key(*lo));
            out.extend_from_slice(get_key(*hi));
        }

        ExprNode::Apply(sym, args) => {
            out.push(RANK_FUNCTION);
            out.push(FN_APPLY);
            // Embed the function name so that distinct named functions sort
            // alphabetically among themselves.
            out.extend_from_slice(get_sym_name(*sym).as_bytes());
            // Null byte separator to avoid ambiguity between name and args.
            out.push(0x00);
            for &arg in args {
                out.extend_from_slice(get_key(arg));
            }
        }

        // Byte-identical to the `Apply` key the library functions had until
        // 0.49 (they sort by name among themselves and with user functions).
        ExprNode::Lib(lib, args) => {
            out.push(RANK_FUNCTION);
            out.push(FN_APPLY);
            out.extend_from_slice(lib.name().as_bytes());
            out.push(0x00);
            for &arg in args {
                out.extend_from_slice(get_key(arg));
            }
        }

        // Byte-identical to the `Apply` key the internal functions had until
        // 0.49.
        ExprNode::Internal(f, args) => {
            out.push(RANK_FUNCTION);
            out.push(FN_APPLY);
            out.extend_from_slice(f.name().as_bytes());
            out.push(0x00);
            for &arg in args {
                out.extend_from_slice(get_key(arg));
            }
        }

        // -- constants -------------------------------------------------------
        ExprNode::Pi => {
            out.push(RANK_CONSTANT);
            out.push(CONST_PI);
        }

        ExprNode::E => {
            out.push(RANK_CONSTANT);
            out.push(CONST_E);
        }

        ExprNode::ImaginaryUnit => {
            out.push(RANK_CONSTANT);
            out.push(CONST_IMAGINARY_UNIT);
        }

        ExprNode::EulerGamma => {
            out.push(RANK_CONSTANT);
            out.push(CONST_EULER_GAMMA);
        }

        ExprNode::Catalan => {
            out.push(RANK_CONSTANT);
            out.push(CONST_CATALAN);
        }

        ExprNode::GoldenRatio => {
            out.push(RANK_CONSTANT);
            out.push(CONST_GOLDEN_RATIO);
        }

        ExprNode::PhysicalConstant(name_id, _) => {
            out.push(RANK_CONSTANT);
            out.push(CONST_PHYSICAL);
            // Include the name for deterministic ordering among physical constants
            out.extend_from_slice(get_sym_name(*name_id).as_bytes());
        }

        // -- special values --------------------------------------------------
        ExprNode::Infinity => {
            out.push(RANK_SPECIAL);
            out.push(SPECIAL_INFINITY);
        }

        ExprNode::NegInfinity => {
            out.push(RANK_SPECIAL);
            out.push(SPECIAL_NEG_INFINITY);
        }

        ExprNode::ComplexInfinity => {
            out.push(RANK_SPECIAL);
            out.push(SPECIAL_COMPLEX_INFINITY);
        }

        ExprNode::NaN => {
            out.push(RANK_SPECIAL);
            out.push(SPECIAL_NAN);
        }

        ExprNode::Neg(x) => {
            out.push(RANK_SPECIAL);
            out.push(SPECIAL_NEG);
            out.extend_from_slice(get_key(*x));
        }

        // -- combinatorial ---------------------------------------------------
        ExprNode::Factorial(x) => {
            out.push(RANK_FUNCTION);
            out.push(FN_ABS + 1); // slot after last built-in function
            out.extend_from_slice(get_key(*x));
        }

        ExprNode::Binomial(n, k) => {
            out.push(RANK_FUNCTION);
            out.push(FN_ABS + 2);
            out.extend_from_slice(get_key(*n));
            out.extend_from_slice(get_key(*k));
        }

        // -- boolean atoms ---------------------------------------------------
        ExprNode::BoolTrue => {
            out.push(RANK_CONSTANT);
            out.push(CONST_BOOL_TRUE);
        }

        ExprNode::BoolFalse => {
            out.push(RANK_CONSTANT);
            out.push(CONST_BOOL_FALSE);
        }

        // -- relational operators --------------------------------------------
        ExprNode::Gt(lhs, rhs) => {
            out.push(RANK_RELATIONAL);
            out.push(0); // Gt discriminant
            out.extend_from_slice(get_key(*lhs));
            out.extend_from_slice(get_key(*rhs));
        }

        ExprNode::Ge(lhs, rhs) => {
            out.push(RANK_RELATIONAL);
            out.push(1);
            out.extend_from_slice(get_key(*lhs));
            out.extend_from_slice(get_key(*rhs));
        }

        ExprNode::Eq_(lhs, rhs) => {
            out.push(RANK_RELATIONAL);
            out.push(2);
            out.extend_from_slice(get_key(*lhs));
            out.extend_from_slice(get_key(*rhs));
        }

        ExprNode::Ne(lhs, rhs) => {
            out.push(RANK_RELATIONAL);
            out.push(3);
            out.extend_from_slice(get_key(*lhs));
            out.extend_from_slice(get_key(*rhs));
        }

        // -- logical connectives ---------------------------------------------
        ExprNode::And(children) => {
            out.push(RANK_AND);
            for &c in children {
                out.extend_from_slice(get_key(c));
            }
        }

        ExprNode::Or(children) => {
            out.push(RANK_OR);
            for &c in children {
                out.extend_from_slice(get_key(c));
            }
        }

        ExprNode::Not(inner) => {
            out.push(RANK_NOT);
            out.extend_from_slice(get_key(*inner));
        }

        // -- piecewise -------------------------------------------------------
        ExprNode::Piecewise(children) => {
            out.push(RANK_PIECEWISE);
            for &(val, cond) in children {
                out.extend_from_slice(get_key(val));
                out.extend_from_slice(get_key(cond));
            }
        }

        // -- set atoms -------------------------------------------------------
        ExprNode::EmptySet => {
            out.push(RANK_SET);
            out.push(SET_EMPTY);
        }

        ExprNode::UniversalSet => {
            out.push(RANK_SET);
            out.push(SET_UNIVERSAL);
        }

        // -- set constructors ------------------------------------------------
        ExprNode::Interval(start, end, kind) => {
            out.push(RANK_SET);
            out.push(SET_INTERVAL);
            out.push(kind.flag_bits());
            out.extend_from_slice(get_key(*start));
            out.extend_from_slice(get_key(*end));
        }

        ExprNode::FiniteSet(elems) => {
            out.push(RANK_SET);
            out.push(SET_FINITE_SET);
            for &elem in elems {
                out.extend_from_slice(get_key(elem));
            }
        }

        ExprNode::SetUnion(sets) => {
            out.push(RANK_SET);
            out.push(SET_UNION);
            for &s in sets {
                out.extend_from_slice(get_key(s));
            }
        }

        ExprNode::SetIntersection(sets) => {
            out.push(RANK_SET);
            out.push(SET_INTERSECTION);
            for &s in sets {
                out.extend_from_slice(get_key(s));
            }
        }

        ExprNode::SetComplement(a, b) => {
            out.push(RANK_SET);
            out.push(SET_COMPLEMENT);
            out.extend_from_slice(get_key(*a));
            out.extend_from_slice(get_key(*b));
        }

        // The bound variable last: it only breaks ties.
        ExprNode::ImageSet(var, body) => {
            out.push(RANK_SET);
            out.push(SET_IMAGE);
            out.extend_from_slice(get_key(*body));
            out.extend_from_slice(get_key(*var));
        }

        // -- formal analysis nodes -------------------------------------------
        ExprNode::Limit(body, var, point) => {
            out.push(RANK_LIMIT);
            out.extend_from_slice(get_key(*body));
            out.extend_from_slice(get_key(*var));
            out.extend_from_slice(get_key(*point));
        }

        ExprNode::Series(body, var, point, order) => {
            out.push(RANK_SERIES);
            out.extend_from_slice(get_key(*body));
            out.extend_from_slice(get_key(*var));
            out.extend_from_slice(get_key(*point));
            out.extend_from_slice(get_key(*order));
        }

        ExprNode::LaplaceTransform(body, t, s) => {
            out.push(RANK_LAPLACE_TRANSFORM);
            out.extend_from_slice(get_key(*body));
            out.extend_from_slice(get_key(*t));
            out.extend_from_slice(get_key(*s));
        }

        ExprNode::InverseLaplaceTransform(body, s, t) => {
            out.push(RANK_INV_LAPLACE_TRANSFORM);
            out.extend_from_slice(get_key(*body));
            out.extend_from_slice(get_key(*s));
            out.extend_from_slice(get_key(*t));
        }

        ExprNode::Residue(body, var, point) => {
            out.push(RANK_RESIDUE);
            out.extend_from_slice(get_key(*body));
            out.extend_from_slice(get_key(*var));
            out.extend_from_slice(get_key(*point));
        }

        // The variable last: it is determined by the polynomial unless that
        // has parameters, so it only breaks ties.
        ExprNode::RootOf(poly, var, index) => {
            out.push(RANK_ROOTOF);
            out.extend_from_slice(get_key(*poly));
            out.extend_from_slice(get_key(*index));
            out.extend_from_slice(get_key(*var));
        }

        ExprNode::DSolve(expr, func, var) => {
            out.push(RANK_DSOLVE);
            out.extend_from_slice(get_key(*expr));
            out.extend_from_slice(get_key(*func));
            out.extend_from_slice(get_key(*var));
        }

        ExprNode::RootSum(poly, body, sumvar) => {
            out.push(RANK_ROOTSUM);
            out.extend_from_slice(get_key(*poly));
            out.extend_from_slice(get_key(*body));
            out.extend_from_slice(get_key(*sumvar));
        }

        ExprNode::ConditionSet(var, cond) => {
            out.push(RANK_CONDITION_SET);
            out.extend_from_slice(get_key(*var));
            out.extend_from_slice(get_key(*cond));
        }

        ExprNode::Subs(body, var, point) => {
            out.push(RANK_SUBS);
            out.extend_from_slice(get_key(*body));
            out.extend_from_slice(get_key(*var));
            out.extend_from_slice(get_key(*point));
        }
    }
}

/// Appends the stored form of the key `full` (as [`compute_sort_key`]
/// assembled it) to `pool`: `full` itself when it has at most
/// [`MAX_KEY_BYTES`] bytes, else its first `MAX_KEY_BYTES` bytes followed by
/// the big-endian 64-bit `FxHasher` digest of all of `full`, so that distinct
/// long keys still (almost surely) compare unequal and the order stays
/// deterministic.
///
/// The digest is that of the whole pre-truncation byte string, hashed in one
/// `write` exactly as up to 0.48: it decides the order of long keys, so it
/// must stay bit-exact (and `rustc-hash` digests depend on the pointer width:
/// on wasm32 long keys order differently than on 64-bit targets, as before).
///
/// A non-truncated key of exactly `MAX_KEY_BYTES` bytes is a proper prefix of
/// any truncated key sharing those bytes and thus sorts first — consistent
/// with the unbounded lexicographic order.
pub(crate) fn push_bounded_key(full: &[u8], pool: &mut Vec<u8>) {
    match full.get(..MAX_KEY_BYTES) {
        Some(prefix) if full.len() > MAX_KEY_BYTES => {
            let mut h = rustc_hash::FxHasher::default();
            h.write(full);
            pool.extend_from_slice(prefix);
            pool.extend_from_slice(&h.finish().to_be_bytes());
        }
        _ => pool.extend_from_slice(full),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    /// The stored key of `node` whose children all have the key `child`,
    /// whose numbers write `0x42` and whose symbol `s<i>` is named `name`.
    fn key_with(node: &ExprNode, child: &[u8], name: &str) -> Vec<u8> {
        let mut full = Vec::new();
        compute_sort_key(
            node,
            &mut full,
            |_| child,
            |_, out| out.push(0x42),
            |_| name,
        );
        let mut pool = Vec::new();
        push_bounded_key(&full, &mut pool);
        pool
    }

    /// The key of an atom (no child is ever asked for).
    fn atom_key(node: &ExprNode) -> Vec<u8> {
        key_with(node, &[], "s")
    }

    fn is_truncated(key: &[u8]) -> bool {
        key.len() > MAX_KEY_BYTES
    }

    #[test]
    fn numbers_sort_before_symbols() {
        let num_key = atom_key(&ExprNode::Num(NumId(0)));
        let sym_key = atom_key(&ExprNode::Symbol(SymbolId(0)));
        assert!(num_key < sym_key, "Num should sort before Symbol");
    }

    #[test]
    fn symbols_sort_alphabetically() {
        let key_a = key_with(&ExprNode::Symbol(SymbolId(0)), &[], "alpha");
        let key_b = key_with(&ExprNode::Symbol(SymbolId(1)), &[], "beta");
        assert!(key_a < key_b);
    }

    #[test]
    fn constants_sort_after_composites() {
        let add_key = key_with(&ExprNode::Add(smallvec![ExprId(0)]), &[RANK_NUM, 0x01], "");
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
        let sin_key = key_with(&ExprNode::Sin(ExprId(0)), &[RANK_NUM, 0x01], "");
        let cos_key = key_with(&ExprNode::Cos(ExprId(0)), &[RANK_NUM, 0x01], "");
        assert_ne!(sin_key, cos_key);
        assert!(
            sin_key < cos_key,
            "Sin (discriminant 0) < Cos (discriminant 1)"
        );
    }

    #[test]
    fn neg_is_special_rank() {
        let neg_key = key_with(&ExprNode::Neg(ExprId(0)), &[RANK_NUM, 0x01], "");
        assert_eq!(neg_key[0], RANK_SPECIAL);
        assert_eq!(neg_key[1], SPECIAL_NEG);
    }

    #[test]
    fn named_constants_have_distinct_constant_subranks() {
        let keys = [
            atom_key(&ExprNode::Pi),
            atom_key(&ExprNode::E),
            atom_key(&ExprNode::ImaginaryUnit),
            atom_key(&ExprNode::EulerGamma),
            atom_key(&ExprNode::Catalan),
            atom_key(&ExprNode::GoldenRatio),
        ];
        for k in &keys {
            assert_eq!(k[0], RANK_CONSTANT);
        }
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert!(keys[i] < keys[j], "constant sub-ranks must be ordered");
            }
        }
    }

    #[test]
    fn long_keys_are_bounded_and_deterministic() {
        // A child key that is already at the bound.
        let big = vec![7u8; MAX_KEY_BYTES];
        let add = ExprNode::Add(smallvec![ExprId(0), ExprId(1)]);
        let k1 = key_with(&add, &big, "");
        assert!(is_truncated(&k1));
        assert_eq!(k1.len(), MAX_KEY_BYTES + 8);
        // Same structure → identical key (digest depends only on bytes).
        let k2 = key_with(&add, &big, "");
        assert_eq!(k1, k2);
        // Different structure with the same prefix → different digest.
        let add3 = ExprNode::Add(smallvec![ExprId(0), ExprId(1), ExprId(2)]);
        let k3 = key_with(&add3, &big, "");
        assert!(is_truncated(&k3));
        assert_ne!(k1, k3);
        assert_eq!(k1[..MAX_KEY_BYTES], k3[..MAX_KEY_BYTES]);
        // A non-truncated key sharing the prefix sorts before a truncated one.
        let exact = &k1[..MAX_KEY_BYTES];
        assert!(!is_truncated(exact));
        assert!(exact < &k1[..]);
        // Short keys are untouched.
        let small = atom_key(&ExprNode::Pi);
        assert!(!is_truncated(&small));
        assert_eq!(small.len(), 2);
    }

    /// The digest of a truncated key is the `FxHasher` digest of the whole
    /// pre-truncation byte string in one `write`, big-endian, as in 0.48
    /// (`SortKey::bound`): the order of long keys depends on it.
    #[test]
    fn truncated_key_digest_is_the_hash_of_the_full_bytes() {
        let full: Vec<u8> = (0..600u32).map(|i| (i % 251) as u8).collect();
        let mut pool = vec![9u8; 3];
        push_bounded_key(&full, &mut pool);
        let mut h = rustc_hash::FxHasher::default();
        h.write(&full);
        let mut expected = vec![9u8; 3];
        expected.extend_from_slice(&full[..MAX_KEY_BYTES]);
        expected.extend_from_slice(&h.finish().to_be_bytes());
        assert_eq!(pool, expected);
        // At the bound exactly: stored as is.
        let mut pool = Vec::new();
        push_bounded_key(&full[..MAX_KEY_BYTES], &mut pool);
        assert_eq!(pool, &full[..MAX_KEY_BYTES]);
    }

    #[test]
    fn new_function_discriminants_are_distinct() {
        let child = [RANK_NUM, 0x01];
        let nodes = [
            ExprNode::Re(ExprId(0)),
            ExprNode::Im(ExprId(0)),
            ExprNode::Conjugate(ExprId(0)),
            ExprNode::Arg(ExprId(0)),
            ExprNode::Si(ExprId(0)),
            ExprNode::Ci(ExprId(0)),
            ExprNode::Ei(ExprId(0)),
            ExprNode::Li(ExprId(0)),
            ExprNode::Zeta(ExprId(0)),
            ExprNode::Polygamma(ExprId(0), ExprId(0)),
            ExprNode::KroneckerDelta(ExprId(0), ExprId(0)),
            ExprNode::LambertW(ExprId(0)),
        ];
        let keys: Vec<Vec<u8>> = nodes.iter().map(|n| key_with(n, &child, "")).collect();
        for k in &keys {
            assert_eq!(k[0], RANK_FUNCTION);
        }
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(keys[i], keys[j], "{:?} vs {:?}", nodes[i], nodes[j]);
            }
        }
    }
}
