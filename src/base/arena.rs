//! Central arena for the symplex expression DAG.
//!
//! The [`Arena`] is the owner of every expression node, numeric literal, and
//! symbol in a single CAS session.  It performs **hash-consing** (structural
//! deduplication) so that structurally identical sub-expressions always share
//! the same [`ExprId`], enabling O(1) equality checks and compact memory use.
//!
//! # Quick start
//!
//! ```text
//! let mut a = Arena::new();
//! let x  = a.symbol("x");
//! let two = a.int(2);
//! let x2  = a.pow(x, two);   // x²
//! let sum = a.add(&[x2, x]); // x² + x
//! ```

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::base::budget::BudgetKind;
use crate::base::config::EvalConfig;
use crate::base::interval::IntervalKind;
use crate::base::libfn::LibFn;
use crate::base::node::{ExprId, ExprNode, NumId, SymbolId};
use crate::base::numeric::Q;
use crate::base::sort_key::{SortKey, compute_sort_key, push_bounded_key};
use crate::base::symbol::SymbolTable;
use crate::base::truth::Truth;

/// The signs of the real and imaginary parts of a constant
/// ([`Arena::part_signs`]); `None` for a part whose sign is not settled (or
/// that is 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PartSigns {
    /// The sign of the real part.
    pub(crate) re: Option<std::cmp::Ordering>,
    /// The sign of the imaginary part.
    pub(crate) im: Option<std::cmp::Ordering>,
}

/// The question an [`Arena::constant_facts`] entry answers about a
/// symbol-free constant `x`: whose value (`x`, `re x` or `im x`) and which
/// decider.  Each decider owns its keys, so none can read another's answer
/// as its own.  Up to 0.48 the keys were hand-picked `u8` codes, `canon`'s
/// sign codes 2–7 overlapped `eval`'s class codes 2–9, and once `(ln 2 −
/// 1)·∞` had memoised the sign of `ln 2 − 1`, `(ln 2 − 1).is_positive()`
/// read that entry as the class "positive" (`Some(true)`; the value is
/// −0.307).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum FactKey {
    /// Is `x` real: `eval`'s `known_real` (`REAL` when proved, `UNKNOWN`
    /// when not).
    Real,
    /// Is `x` a negative real: `eval`'s `constant_is_negative` (`NEGATIVE`
    /// when proved, `UNKNOWN` when not).
    Negative,
    /// The signs `eval`'s `constant_class` decides for `x`: the facts the
    /// assumption queries of constants add.
    Class,
    /// The sign of `x` as a factor of a directed infinity: `canon`'s
    /// `known_sign_memo`.
    Sign,
    /// The sign of `re(x)`, likewise.
    RePart,
    /// The sign of `im(x)`, likewise.
    ImPart,
    /// Is `x` zero: `calculus_util`'s `hidden_zero`.  `ZERO` when proved 0,
    /// `NONZERO` when proved not 0, `UNKNOWN` when `x` is 0 to the
    /// precision reached but not proved (a test with no evidence at all is
    /// not memoised).
    HiddenZero,
    /// Is `x` nonzero by a certified value: the assumption rules for a
    /// power `x^e` of a constant (`assumptions`' `certified_nonzero`;
    /// `NONZERO` when proved, `UNKNOWN` when not).
    NonzeroValue,
    /// Is `x` (no `±∞`, `zoo`, `nan` in it) finite: `value_class`'s
    /// `ValueClass::of` (`UNKNOWN`, any finite sign, when proved by the
    /// assumptions or a certified value; `EMPTY` when not proved).
    Finite,
}

/// The `sym_bits` entry of a symbol name without a `Symbol` node.
const NO_SYMBOL_NODE: u8 = u8::MAX;

/// The bit of the [`NodeFacts`] free-symbol mask shared by the symbols
/// whose `Symbol` node was created after the first 63.
pub(crate) const SHARED_SYMBOL_BIT: u64 = 1 << 63;

/// Structural facts of a node, computed once when the node is interned — in
/// O(arity), from the facts of its children — and read in O(1): the tree
/// queries that used to walk the DAG on every call ([`walk::has_unevaluated`],
/// [`walk::has_free_symbol`], [`walk::free_symbols`],
/// [`Arena::tree_size_capped`]) answer from them.
///
/// The free-symbol mask follows [`walk::free_symbols`]' binder rules exactly
/// (the table of [`walk::binder`]): bit `k < 63` is set exactly when the
/// symbol whose `Symbol` node was the `k`-th created in this arena is free in
/// the node, so a query about such a symbol is decided by one bit.  The
/// symbols created later share bit 63, which is set when one of them is free
/// and may also be set when none is (a binder of such a symbol cannot clear
/// it: [`NodeFacts::SHARED_BIT_INEXACT`]); a query that meets bit 63 walks.
/// Which symbol gets which bit depends on the arena's history, but every
/// answer is exact, so no output does.
///
/// [`walk::has_unevaluated`]: crate::base::walk::has_unevaluated
/// [`walk::has_free_symbol`]: crate::base::walk::has_free_symbol
/// [`walk::free_symbols`]: crate::base::walk::free_symbols
/// [`walk::binder`]: crate::base::walk::binder
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NodeFacts {
    /// The number of nodes of the unfolded tree (a shared child counted at
    /// every occurrence), saturating at `u32::MAX`.
    pub(crate) tree_size: u32,
    /// 1 for an atom, else 1 + the greatest height of a child; saturating.
    pub(crate) height: u16,
    /// The `HAS_*` flags below.
    flags: u16,
    /// The free symbols, one bit each (see the type's documentation).
    sym_mask: u64,
}

impl NodeFacts {
    /// The tree contains an unevaluated formal node (the variants
    /// [`walk::has_unevaluated`](crate::base::walk::has_unevaluated) lists).
    const HAS_UNEVALUATED: u16 = 1 << 0;
    /// The tree contains a `Symbol` node, free or bound.
    const HAS_SYMBOL: u16 = 1 << 1;
    /// The tree contains a binder whose variable is a symbol.
    const HAS_BINDER: u16 = 1 << 2;
    /// Bit 63 of the mask may be set although no symbol owning it is free.
    const SHARED_BIT_INEXACT: u16 = 1 << 3;

    /// The facts of an atom other than a symbol.
    const ATOM: NodeFacts = NodeFacts {
        tree_size: 1,
        height: 1,
        flags: 0,
        sym_mask: 0,
    };

    /// Does the tree contain an unevaluated formal node?
    #[inline]
    pub(crate) fn has_unevaluated(self) -> bool {
        self.flags & Self::HAS_UNEVALUATED != 0
    }

    /// Does the tree contain a `Symbol` node (free or bound)?
    #[inline]
    pub(crate) fn has_symbol(self) -> bool {
        self.flags & Self::HAS_SYMBOL != 0
    }

    /// Does the tree contain a binder of a symbol?  Without one, a symbol
    /// occurs in the tree exactly when it is free in it.
    #[inline]
    pub(crate) fn has_binder(self) -> bool {
        self.flags & Self::HAS_BINDER != 0
    }

    /// Could a tree with these facts contain one with the facts `inner`?
    /// `false` when it is too small, too low, or lacks a structure `inner`
    /// has (a `Symbol` node, a binder, an unevaluated node).
    #[inline]
    pub(crate) fn may_contain(self, inner: NodeFacts) -> bool {
        let structural = Self::HAS_UNEVALUATED | Self::HAS_SYMBOL | Self::HAS_BINDER;
        self.tree_size >= inner.tree_size
            && self.height >= inner.height
            && self.flags & inner.flags & structural == inner.flags & structural
    }
}

/// Is `node` one of the unevaluated formal nodes
/// [`walk::has_unevaluated`](crate::base::walk::has_unevaluated) looks for?
fn is_unevaluated_node(node: &ExprNode) -> bool {
    matches!(
        node,
        ExprNode::Integral(..)
            | ExprNode::DefiniteIntegral(..)
            | ExprNode::Derivative(..)
            | ExprNode::Limit(..)
            | ExprNode::Series(..)
            | ExprNode::LaplaceTransform(..)
            | ExprNode::InverseLaplaceTransform(..)
            | ExprNode::Residue(..)
            | ExprNode::DSolve(..)
            | ExprNode::ConditionSet(..)
            | ExprNode::Subs(..)
            | ExprNode::Sum(..)
            | ExprNode::Product_(..)
    )
}

// ---------------------------------------------------------------------------
// Arena
// ---------------------------------------------------------------------------

/// The central data structure of the symplex CAS.
///
/// An `Arena` owns all expression nodes, numeric literals, and symbol names.
/// It guarantees that structurally equal expressions map to the same
/// [`ExprId`] via hash-consing, which makes equality checks trivial pointer
/// comparisons.
///
/// A set of commonly used constants (0, 1, −1, π, e, i, γ, G, φ, ∞, −∞,
/// NaN) are pre-interned at construction time and exposed through accessor
/// methods for convenience.
pub struct Arena {
    // -- storage -------------------------------------------------------------
    /// Expression nodes indexed by [`ExprId`].
    nodes: Vec<ExprNode>,

    /// Numeric literal side-table indexed by [`NumId`].
    numbers: Vec<Q>,

    /// The sort keys of all nodes, one after the other: the key of
    /// `nodes[i]` is `key_bytes[key_starts[i]..key_starts[i + 1]]`.  One
    /// pool instead of a key object per node: no allocation per key, 8
    /// bytes per node besides the key's own bytes.
    key_bytes: Vec<u8>,

    /// Where each node's key starts in `key_bytes`, plus the end of the
    /// last one (`key_starts.len() == nodes.len() + 1`).
    key_starts: Vec<usize>,

    /// Reused buffer in which a new node's key is assembled before it is
    /// bounded and appended to `key_bytes`.
    key_scratch: Vec<u8>,

    /// [`NodeFacts`] parallel to `nodes`.
    facts: Vec<NodeFacts>,

    /// The free-symbol bit index of each symbol ([`NodeFacts`]), indexed by
    /// [`SymbolId`]: `k` for the symbol whose `Symbol` node was the `k`-th
    /// created (`k < 63`), 63 for the later ones, [`NO_SYMBOL_NODE`] for a
    /// name without a `Symbol` node (a function name, or none yet).
    sym_bits: Vec<u8>,

    /// The `Symbol` nodes owning the exact bits 0–62, in bit order.
    bit_symbols: Vec<ExprId>,

    /// Hash-consing deduplication map.
    ///
    /// Maps a `u64` hash of an [`ExprNode`] to the set of [`ExprId`]s that
    /// produced that hash.  On lookup the candidates are compared by value to
    /// handle hash collisions.
    dedup: FxHashMap<u64, SmallVec<[ExprId; 2]>>,

    /// Numeric literal deduplication map.
    ///
    /// Maps a `u64` hash of a [`Ratio<BigInt>`] to the set of [`NumId`]s that
    /// produced that hash. On lookup the candidates are compared by value to
    /// handle hash collisions.
    num_dedup: FxHashMap<u64, SmallVec<[NumId; 2]>>,

    /// Interned symbol names.
    pub(crate) symbols: SymbolTable,

    /// Evaluation-time configuration (guards against runaway computation).
    pub(crate) config: EvalConfig,

    /// Answers `eval`, `canon` and the zero test of the limit and series
    /// engines proved for symbol-free constants (realness, sign, zero),
    /// keyed by the expression and the question
    /// ([`FactKey`]).  The value is the set of signs the decider proved
    /// possible for the value the key names (`SignSet::UNKNOWN` when it
    /// proved nothing).
    ///
    /// A proof takes a certified evaluation, a deep zero search of the
    /// imaginary part and a minimal polynomial, and every `eval` pass over
    /// a radical by Cardano's or Ferrari's formula asks again: in 0.44
    /// before this memo `∫ 1/(x⁵ + x + 2) dx` took 5.8 s instead of 0.5 s.
    /// The answer depends only on the node and on [`EvalConfig`]
    /// (assumptions do not reach a symbol-free constant), so the memo is
    /// cleared when either can change ([`Self::config_mut`],
    /// [`Self::set_symbol_assumptions`]); a warm entry is what a cold
    /// proof returns.  Behind a lock, read and written through
    /// [`Self::constant_fact`] and [`Self::remember_constant_fact`], so that
    /// a decider holding only `&Arena` memoises too: the assumption rules,
    /// which evaluated the base of `x^(−1)` again in every fresh
    /// `AssumptionCache` (0.51 development: +48% on the `fuzz_roundtrip`
    /// corpus).
    // `AssertUnwindSafe`: the memo only ever holds completed proofs (an
    // entry is inserted whole, under the lock), so a panic elsewhere cannot
    // leave it inconsistent; without the wrapper `parking_lot::Mutex` would
    // make `Arena`, hence the public `ExprView`, lose `RefUnwindSafe`.
    constant_facts: std::panic::AssertUnwindSafe<
        parking_lot::Mutex<FxHashMap<(ExprId, FactKey), crate::base::truth::SignSet>>,
    >,

    // -- pre-interned constants (ExprId) ------------------------------------
    /// The integer zero (0).
    pub(crate) zero: ExprId,

    /// The integer one (1).
    pub(crate) one: ExprId,

    /// The integer negative one (−1).
    pub(crate) neg_one: ExprId,

    /// The mathematical constant π.
    pub(crate) pi: ExprId,

    /// Euler's number *e*.
    pub(crate) e_const: ExprId,

    /// The imaginary unit *i*.
    pub(crate) i_unit: ExprId,

    /// The Euler–Mascheroni constant γ.
    pub(crate) euler_gamma: ExprId,

    /// Catalan's constant G.
    pub(crate) catalan: ExprId,

    /// The golden ratio φ.
    pub(crate) golden_ratio: ExprId,

    /// Positive infinity (+∞).
    pub(crate) infinity: ExprId,

    /// Negative infinity (−∞).
    pub(crate) neg_infinity: ExprId,

    /// Not-a-number (undefined / indeterminate).
    pub(crate) nan: ExprId,

    /// Complex infinity (z∞ — undirected infinity in the complex plane).
    pub(crate) complex_infinity: ExprId,

    /// The empty set ∅.
    pub(crate) empty_set: ExprId,

    /// The universal set.
    pub(crate) universal_set: ExprId,

    /// Boolean true.
    pub(crate) bool_true: ExprId,

    /// Boolean false.
    pub(crate) bool_false: ExprId,

    // -- pre-interned constants (NumId) -------------------------------------
    /// [`NumId`] for the rational value 0.
    pub(crate) zero_num: NumId,

    /// [`NumId`] for the rational value 1.
    pub(crate) one_num: NumId,

    /// [`NumId`] for the rational value −1.
    pub(crate) neg_one_num: NumId,

    /// Node count at the last `compact()` call. Used by `should_compact()` heuristic.
    pub(crate) last_compact_size: usize,

    // -- work accounting and per-arena state ---------------------------------
    /// [`Self::intern`] calls so far, whether the node was new or found:
    /// the deterministic measure of the
    /// [`WorkScope`](crate::base::budget::WorkScope) budgets (the growth of
    /// `nodes` depends on what the arena already holds).
    interns: u64,

    /// The open work scopes, innermost last: each kind with the intern
    /// count at which it is spent.
    work_scopes: Vec<(BudgetKind, u64)>,

    /// A number no other arena of this process has.  Nodes are only ever
    /// appended, so `(serial, id)` names one node for good (the
    /// quadrature's memo of failures; an address can be reused).
    serial: u64,

    /// The number of the next wild symbol `Arena::wild` creates (`_w0`,
    /// `_w1`, … in every arena).
    pub(crate) next_wild: u32,
}

/// The serial of the next arena created ([`Arena::serial`]).
static NEXT_ARENA_SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl Arena {
    /// Creates a new arena with default [`EvalConfig`] and pre-interned
    /// constants.
    pub fn new() -> Self {
        Self::with_config(EvalConfig::default())
    }

    /// Creates a new arena with a custom [`EvalConfig`] and pre-interned
    /// constants.
    pub fn with_config(config: EvalConfig) -> Self {
        let mut arena = Self {
            nodes: Vec::new(),
            numbers: Vec::new(),
            key_bytes: Vec::new(),
            key_starts: vec![0],
            key_scratch: Vec::new(),
            facts: Vec::new(),
            sym_bits: Vec::new(),
            bit_symbols: Vec::new(),
            dedup: FxHashMap::default(),
            num_dedup: FxHashMap::default(),
            symbols: SymbolTable::new(),
            config,
            constant_facts: std::panic::AssertUnwindSafe(parking_lot::Mutex::new(
                FxHashMap::default(),
            )),
            // Temporary placeholders — will be overwritten immediately below.
            zero: ExprId(0),
            one: ExprId(0),
            neg_one: ExprId(0),
            pi: ExprId(0),
            e_const: ExprId(0),
            i_unit: ExprId(0),
            euler_gamma: ExprId(0),
            catalan: ExprId(0),
            golden_ratio: ExprId(0),
            infinity: ExprId(0),
            neg_infinity: ExprId(0),
            nan: ExprId(0),
            complex_infinity: ExprId(0),
            empty_set: ExprId(0),
            universal_set: ExprId(0),
            bool_true: ExprId(0),
            bool_false: ExprId(0),
            zero_num: NumId(0),
            one_num: NumId(0),
            neg_one_num: NumId(0),
            last_compact_size: 0,
            interns: 0,
            work_scopes: Vec::new(),
            serial: NEXT_ARENA_SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            next_wild: 0,
        };

        // 1. Intern the three fundamental numeric values.
        arena.zero_num = arena.intern_num(Ratio::zero());
        arena.one_num = arena.intern_num(Ratio::one());
        arena.neg_one_num = arena.intern_num(-Ratio::<BigInt>::one());

        // 2. Intern Num nodes for each.
        arena.zero = arena.intern(ExprNode::Num(arena.zero_num));
        arena.one = arena.intern(ExprNode::Num(arena.one_num));
        arena.neg_one = arena.intern(ExprNode::Num(arena.neg_one_num));

        // 3. Intern the mathematical constants and special values.
        arena.pi = arena.intern(ExprNode::Pi);
        arena.e_const = arena.intern(ExprNode::E);
        arena.i_unit = arena.intern(ExprNode::ImaginaryUnit);
        arena.euler_gamma = arena.intern(ExprNode::EulerGamma);
        arena.catalan = arena.intern(ExprNode::Catalan);
        arena.golden_ratio = arena.intern(ExprNode::GoldenRatio);
        arena.infinity = arena.intern(ExprNode::Infinity);
        arena.neg_infinity = arena.intern(ExprNode::NegInfinity);
        arena.nan = arena.intern(ExprNode::NaN);
        arena.complex_infinity = arena.intern(ExprNode::ComplexInfinity);

        arena.bool_true = arena.intern(ExprNode::BoolTrue);
        arena.bool_false = arena.intern(ExprNode::BoolFalse);

        arena.empty_set = arena.intern(ExprNode::EmptySet);
        arena.universal_set = arena.intern(ExprNode::UniversalSet);

        arena
    }
}

// ---------------------------------------------------------------------------
// Public accessors for pre-interned constants
// ---------------------------------------------------------------------------

impl Arena {
    /// Returns the pre-interned [`ExprId`] for integer zero (0).
    #[inline]
    pub fn zero(&self) -> ExprId {
        self.zero
    }

    /// Returns the pre-interned [`ExprId`] for integer one (1).
    #[inline]
    pub fn one(&self) -> ExprId {
        self.one
    }

    /// Returns the pre-interned [`ExprId`] for integer negative one (−1).
    #[inline]
    pub fn neg_one(&self) -> ExprId {
        self.neg_one
    }

    /// Returns the pre-interned [`ExprId`] for the constant π.
    #[inline]
    pub fn pi(&self) -> ExprId {
        self.pi
    }

    /// Returns the pre-interned [`ExprId`] for Euler's number *e*.
    #[inline]
    pub fn e_const(&self) -> ExprId {
        self.e_const
    }

    /// Returns the pre-interned [`ExprId`] for the imaginary unit *i*.
    #[inline]
    pub fn i_unit(&self) -> ExprId {
        self.i_unit
    }

    /// Returns the pre-interned [`ExprId`] for the Euler–Mascheroni constant γ.
    #[inline]
    pub fn euler_gamma(&self) -> ExprId {
        self.euler_gamma
    }

    /// Returns the pre-interned [`ExprId`] for Catalan's constant G.
    #[inline]
    pub fn catalan(&self) -> ExprId {
        self.catalan
    }

    /// Returns the pre-interned [`ExprId`] for the golden ratio φ.
    #[inline]
    pub fn golden_ratio(&self) -> ExprId {
        self.golden_ratio
    }

    /// Returns the pre-interned [`ExprId`] for positive infinity (+∞).
    #[inline]
    pub fn infinity(&self) -> ExprId {
        self.infinity
    }

    /// Returns the pre-interned [`ExprId`] for negative infinity (−∞).
    #[inline]
    pub fn neg_infinity(&self) -> ExprId {
        self.neg_infinity
    }

    /// Returns the pre-interned [`ExprId`] for NaN.
    #[inline]
    pub fn nan(&self) -> ExprId {
        self.nan
    }

    /// Returns the pre-interned [`ExprId`] for complex infinity (z∞).
    #[inline]
    pub fn complex_infinity(&self) -> ExprId {
        self.complex_infinity
    }

    /// Returns the pre-interned [`ExprId`] for boolean true.
    #[inline]
    pub fn bool_true(&self) -> ExprId {
        self.bool_true
    }

    /// Returns the pre-interned [`ExprId`] for boolean false.
    #[inline]
    pub fn bool_false(&self) -> ExprId {
        self.bool_false
    }

    /// Returns the pre-interned [`NumId`] for the rational value 0.
    #[inline]
    pub fn zero_num(&self) -> NumId {
        self.zero_num
    }

    /// Returns the pre-interned [`NumId`] for the rational value 1.
    #[inline]
    pub fn one_num(&self) -> NumId {
        self.one_num
    }

    /// Returns the pre-interned [`NumId`] for the rational value −1.
    #[inline]
    pub fn neg_one_num(&self) -> NumId {
        self.neg_one_num
    }

    /// Returns a reference to the evaluation configuration.
    #[inline]
    pub fn config(&self) -> &EvalConfig {
        &self.config
    }

    /// Returns a mutable reference to the evaluation configuration.
    #[inline]
    pub fn config_mut(&mut self) -> &mut EvalConfig {
        self.constant_facts.get_mut().clear();
        &mut self.config
    }

    /// The signs memoised for the symbol-free constant `x` under `key`
    /// ([`Self::constant_facts`]), if any.
    pub(crate) fn constant_fact(
        &self,
        x: ExprId,
        key: FactKey,
    ) -> Option<crate::base::truth::SignSet> {
        self.constant_facts.lock().get(&(x, key)).copied()
    }

    /// Memoise `signs` for the symbol-free constant `x` under `key`.
    pub(crate) fn remember_constant_fact(
        &self,
        x: ExprId,
        key: FactKey,
        signs: crate::base::truth::SignSet,
    ) {
        self.constant_facts.lock().insert((x, key), signs);
    }
}

// ---------------------------------------------------------------------------
// Core interning machinery
// ---------------------------------------------------------------------------

impl Arena {
    /// Computes a `u64` hash for an [`ExprNode`] using the `FxHasher`.
    fn hash_node(node: &ExprNode) -> u64 {
        let mut hasher = rustc_hash::FxHasher::default();
        node.hash(&mut hasher);
        hasher.finish()
    }

    /// Computes a `u64` hash for a [`Ratio<BigInt>`]: equal values hash
    /// alike whatever their representation (reduced or not), as `Ratio`'s
    /// own `Hash` guarantees, in time linear in the digits.
    ///
    /// The hash is the value modulo the prime `P = 2⁶¹ − 1`,
    /// `numer·denom⁻¹ mod P`, tagged by which of two primes it was taken
    /// modulo.  A factor `P` common to numerator and denominator (an
    /// unreduced representation) is divided out first, so the choice depends
    /// on the value only; a value whose reduced denominator is a multiple of
    /// `P` is hashed modulo `2⁶¹ − 1`'s neighbour prime `2⁶² − 57`, and one
    /// whose reduced denominator is a multiple of both falls back to the
    /// continued fraction.  Until 0.35 the hash was always the sequence of
    /// partial quotients of numer/denom (`Ratio`'s scheme, in a loop):
    /// Euclid's algorithm on the digits, quadratic — the 1,000-digit
    /// fractions of an extended gcd's cofactors took seconds to intern.  The
    /// map keyed by this hash is only looked up, never iterated, so the
    /// interned ids (and every output) are unchanged.
    fn hash_num(value: &Q) -> u64 {
        for (tag, p) in [(1u64, (1u64 << 61) - 1), (2, (1u64 << 62) - 57)] {
            match Self::residue_of_value(value.numer(), value.denom(), p) {
                Some(Some(r)) => return r.wrapping_mul(4).wrapping_add(tag),
                // The reduced denominator is a multiple of `p`.
                Some(None) => continue,
                None => break,
            }
        }
        // Same sequence of partial quotients as `Ratio`'s `Hash`, in a loop
        // (`Ratio`'s recursion takes one stack frame per partial quotient).
        use num_integer::Integer;
        let mut hasher = rustc_hash::FxHasher::default();
        let (mut numer, mut denom) = (value.numer().clone(), value.denom().clone());
        loop {
            if denom.is_zero() {
                denom.hash(&mut hasher);
                break;
            }
            let (quot, rem) = numer.div_mod_floor(&denom);
            quot.hash(&mut hasher);
            numer = denom;
            denom = rem;
        }
        hasher.finish() & !3
    }

    /// `numer/denom mod p` for the prime `p < 2⁶²`: `Some(Some(r))`,
    /// `Some(None)` when `p` divides the reduced denominator, `None` for a
    /// zero denominator (not a rational).
    fn residue_of_value(numer: &BigInt, denom: &BigInt, p: u64) -> Option<Option<u64>> {
        use num_traits::{Signed, ToPrimitive};
        let res = |x: &BigInt| -> u64 {
            let m = x.magnitude();
            let r = match m.to_u64() {
                Some(v) => v % p,
                None => (m % p).to_u64().unwrap_or(0),
            };
            if r != 0 && x.is_negative() { p - r } else { r }
        };
        let mul = |a: u64, b: u64| ((u128::from(a) * u128::from(b)) % u128::from(p)) as u64;
        let (mut n, mut d) = (res(numer), res(denom));
        if denom.is_zero() {
            return None;
        }
        let mut owned: Option<(BigInt, BigInt)> = None;
        while d == 0 {
            if n != 0 {
                return Some(None);
            }
            // `p` divides both (an unreduced representation): strip it and
            // look again; each round divides the non-zero denominator by p.
            let pb = BigInt::from(p);
            let (a, b) = owned
                .take()
                .unwrap_or_else(|| (numer.clone(), denom.clone()));
            let (a, b) = (a / &pb, b / &pb);
            (n, d) = (res(&a), res(&b));
            owned = Some((a, b));
        }
        if d == 1 {
            return Some(Some(n));
        }
        // d⁻¹ mod p by the extended Euclidean algorithm (p is prime, d ≠ 0).
        // Every cofactor is below p < 2⁶² in magnitude: machine words.
        let (mut r0, mut r1) = (p, d);
        let (mut t0, mut t1) = (0i64, 1i64);
        while r1 != 0 {
            let q = r0 / r1;
            (r0, r1) = (r1, r0 - q * r1);
            (t0, t1) = (t1, t0.wrapping_sub((q as i64).wrapping_mul(t1)));
        }
        let inv = t0.rem_euclid(p as i64) as u64;
        Some(Some(mul(n, inv)))
    }

    /// Interns an [`ExprNode`], returning its canonical [`ExprId`].
    ///
    /// If a structurally identical node has already been interned, the
    /// existing [`ExprId`] is returned.  Otherwise the node is appended to
    /// the arena, its sort key and [`NodeFacts`] are computed, and the new
    /// [`ExprId`] is recorded in the dedup map.
    ///
    /// A function application at an exact special argument whose value is
    /// rational, infinite or `nan` is never stored: its value is returned
    /// instead (`sin(0)` → `0`, `ln(0)` → `zoo`, `Γ(−2)` → `zoo`; see
    /// [`canon_function`](crate::base::canon::canon_function)).  Every path
    /// that builds nodes — the constructors, substitution, parsing, the
    /// algorithms that intern directly — passes through here, so canonical
    /// arithmetic never meets such an atom (`sin(0)/sin(0)` is `nan`, not 1).
    pub(crate) fn intern(&mut self, node: ExprNode) -> ExprId {
        self.interns += 1;
        if let Some(value) = crate::base::canon::canon_function(self, &node) {
            return value;
        }
        let hash = Self::hash_node(&node);

        // Check the dedup map for an existing equivalent node.
        if let Some(candidates) = self.dedup.get(&hash) {
            for &candidate in candidates {
                if self.nodes[candidate.0 as usize] == node {
                    return candidate;
                }
            }
        }

        // Not found — allocate a new slot.
        let id = ExprId(
            u32::try_from(self.nodes.len())
                .expect("arena overflow: more than 4 billion expression nodes"),
        );
        let facts = self.facts_of_new(id, &node);
        self.push_sort_key(&node);
        self.nodes.push(node);
        self.facts.push(facts);

        self.dedup.entry(hash).or_default().push(id);

        id
    }

    /// Builds `node` with its variant's canonical constructor, or interns
    /// it when the variant has none: the way to turn a node value into an
    /// [`ExprId`] outside `base`.
    ///
    /// [`intern`](Self::intern) stores a node as it is, so a raw
    /// `intern(ExprNode::FiniteSet([1, 1]))` makes a `{1, 1}` that is not
    /// `==` to the `{1}` every constructor builds.  `make` routes `Add`,
    /// `Mul`, `Pow`, `Neg`, the relationals, `And`/`Or`/`Not`, `Piecewise`,
    /// the set constructors (`Interval`, `FiniteSet`, `SetUnion`,
    /// `SetIntersection`, `SetComplement`), `DefiniteIntegral`, `Subs` and
    /// every function with a folding constructor (`abs`, `sign`, `re`, `si`,
    /// `zeta`, `polygamma`, `kronecker_delta`, …) to that constructor, and
    /// interns the rest (atoms, the formal nodes, the functions whose
    /// constructor is a plain intern).  It is what
    /// [`walk::rebuild_with`](crate::base::walk::rebuild_with) builds a node
    /// with new children with ([`ExprNode::map_children`]), and the
    /// `test_canonical_construction` ratchet keeps raw interns of the
    /// canonical variants out of the rest of the crate.
    pub(crate) fn make(&mut self, node: ExprNode) -> ExprId {
        use crate::base::walk::subs_node;
        match node {
            ExprNode::Add(c) => self.add(&c),
            ExprNode::Mul(c) => self.mul(&c),
            ExprNode::Pow(b, e) => self.pow(b, e),
            ExprNode::Neg(x) => self.neg(x),
            ExprNode::Sin(x) => self.sin(x),
            ExprNode::Cos(x) => self.cos(x),
            ExprNode::Tan(x) => self.tan(x),
            ExprNode::Exp(x) => self.exp(x),
            ExprNode::Ln(x) => self.ln(x),
            ExprNode::Abs(x) => self.abs(x),
            ExprNode::Atan2(y, x) => self.atan2(y, x),
            ExprNode::Sign(x) => self.sign(x),
            ExprNode::Re(x) => self.re(x),
            ExprNode::Im(x) => self.im(x),
            ExprNode::Conjugate(x) => self.conjugate(x),
            ExprNode::Arg(x) => self.arg(x),
            ExprNode::Si(x) => self.si(x),
            ExprNode::Ci(x) => self.ci(x),
            ExprNode::Ei(x) => self.ei(x),
            ExprNode::Li(x) => self.li(x),
            ExprNode::Zeta(x) => self.zeta(x),
            ExprNode::Polygamma(n, x) => self.polygamma(n, x),
            ExprNode::KroneckerDelta(i, j) => self.kronecker_delta(i, j),
            ExprNode::Beta(a, b) => self.beta(a, b),
            ExprNode::Factorial(x) => self.factorial(x),
            ExprNode::Binomial(n, k) => self.binomial(n, k),
            ExprNode::Gt(a, b) => self.gt(a, b),
            ExprNode::Ge(a, b) => self.ge(a, b),
            ExprNode::Eq_(a, b) => self.eq_(a, b),
            ExprNode::Ne(a, b) => self.ne_(a, b),
            ExprNode::Not(x) => self.not(x),
            ExprNode::And(c) => self.and(&c),
            ExprNode::Or(c) => self.or(&c),
            ExprNode::Piecewise(pairs) => self.piecewise(&pairs),
            ExprNode::Interval(a, b, kind) => self.interval(a, b, kind),
            ExprNode::FiniteSet(c) => self.finite_set(&c),
            ExprNode::SetUnion(c) => self.set_union(&c),
            ExprNode::SetIntersection(c) => self.set_intersection(&c),
            ExprNode::SetComplement(a, b) => self.set_complement(a, b),
            ExprNode::DefiniteIntegral(f, v, lo, hi) => self.definite_integral(f, v, lo, hi),
            ExprNode::Subs(f, v, p) => subs_node(self, f, v, p),
            ExprNode::Lib(f, args) => self.lib_apply(f, &args),
            other @ (ExprNode::Num(_)
            | ExprNode::Symbol(_)
            | ExprNode::Pi
            | ExprNode::E
            | ExprNode::ImaginaryUnit
            | ExprNode::EulerGamma
            | ExprNode::Catalan
            | ExprNode::GoldenRatio
            | ExprNode::PhysicalConstant(..)
            | ExprNode::Infinity
            | ExprNode::NegInfinity
            | ExprNode::ComplexInfinity
            | ExprNode::NaN
            | ExprNode::Min(_)
            | ExprNode::Max(_)
            | ExprNode::Floor(_)
            | ExprNode::Ceiling(_)
            | ExprNode::Asin(_)
            | ExprNode::Acos(_)
            | ExprNode::Atan(_)
            | ExprNode::Sinh(_)
            | ExprNode::Cosh(_)
            | ExprNode::Tanh(_)
            | ExprNode::Asinh(_)
            | ExprNode::Acosh(_)
            | ExprNode::Atanh(_)
            | ExprNode::Heaviside(_)
            | ExprNode::DiracDelta(_)
            | ExprNode::Gamma(_)
            | ExprNode::LogGamma(_)
            | ExprNode::Digamma(_)
            | ExprNode::Erf(_)
            | ExprNode::Erfc(_)
            | ExprNode::LambertW(_)
            | ExprNode::BoolTrue
            | ExprNode::BoolFalse
            | ExprNode::Apply(..)
            | ExprNode::Derivative(..)
            | ExprNode::Integral(..)
            | ExprNode::Sum(..)
            | ExprNode::Product_(..)
            | ExprNode::Limit(..)
            | ExprNode::Series(..)
            | ExprNode::LaplaceTransform(..)
            | ExprNode::InverseLaplaceTransform(..)
            | ExprNode::Residue(..)
            | ExprNode::RootOf(..)
            | ExprNode::RootSum(..)
            | ExprNode::DSolve(..)
            | ExprNode::ConditionSet(..)
            | ExprNode::EmptySet
            | ExprNode::UniversalSet
            | ExprNode::ImageSet(..)
            | ExprNode::Internal(..)) => self.intern(other),
        }
    }

    /// Interns a rational number, returning its [`NumId`].
    ///
    /// If the same value is already present the existing [`NumId`] is
    /// returned.  A hash-map lookup is used for fast deduplication.
    pub(crate) fn intern_num(&mut self, value: Q) -> NumId {
        let hash = Self::hash_num(&value);

        // Check the num_dedup map for an existing match.
        if let Some(candidates) = self.num_dedup.get(&hash) {
            for &candidate in candidates {
                if self.numbers[candidate.0 as usize] == value {
                    return candidate;
                }
            }
        }

        // Not found — allocate a new slot.
        let id = NumId(
            u32::try_from(self.numbers.len())
                .expect("arena overflow: more than 4 billion numeric literals"),
        );
        self.numbers.push(value);
        self.num_dedup.entry(hash).or_default().push(id);
        id
    }

    /// Appends the sort key of `node` (about to be pushed) to the key pool:
    /// assembled by
    /// [`sort_key::compute_sort_key`](crate::base::sort_key::compute_sort_key)
    /// in the reused scratch buffer from the children's keys in the pool,
    /// then bounded by [`push_bounded_key`].
    fn push_sort_key(&mut self, node: &ExprNode) {
        let mut scratch = std::mem::take(&mut self.key_scratch);
        scratch.clear();
        compute_sort_key(
            node,
            &mut scratch,
            |child_id| self.sort_key(child_id),
            |num_id, out| write_decimal(&self.numbers[num_id.0 as usize], out),
            |sym_id| self.symbols.name(sym_id),
        );
        push_bounded_key(&scratch, &mut self.key_bytes);
        self.key_starts.push(self.key_bytes.len());
        self.key_scratch = scratch;
    }

    /// The [`NodeFacts`] of `node`, about to be stored as `id`, from the
    /// facts of its children (all interned before it): O(arity).  A
    /// `Symbol` node gets the next free-symbol bit.
    fn facts_of_new(&mut self, id: ExprId, node: &ExprNode) -> NodeFacts {
        if let ExprNode::Symbol(s) = *node {
            return NodeFacts {
                tree_size: 1,
                height: 1,
                flags: NodeFacts::HAS_SYMBOL,
                sym_mask: self.assign_symbol_bit(s, id),
            };
        }
        if node.is_atom() {
            return NodeFacts::ATOM;
        }
        let facts = &self.facts;
        let child = |c: ExprId| facts[c.0 as usize];
        let (mut tree_size, mut height, mut flags, mut sym_mask) = (1u32, 0u16, 0u16, 0u64);
        node.for_each_child(|c| {
            let f = child(c);
            tree_size = tree_size.saturating_add(f.tree_size);
            height = height.max(f.height);
            flags |= f.flags;
            sym_mask |= f.sym_mask;
        });
        if is_unevaluated_node(node) {
            flags |= NodeFacts::HAS_UNEVALUATED;
        }
        // A binder of a symbol (`walk::binder`): its variable is free in
        // the outer operands only, and the variable operand itself is not
        // an occurrence.
        if let Some(b) = node.binder_operands()
            && let ExprNode::Symbol(var) = self.nodes[b.var.0 as usize]
        {
            flags |= NodeFacts::HAS_BINDER;
            let outer = b.outer.iter().fold(0, |m, &c| m | child(c).sym_mask);
            let mut scoped = b.scoped.iter().fold(0, |m, &c| m | child(c).sym_mask);
            let var_bit = self.symbol_bit(var).unwrap_or(SHARED_SYMBOL_BIT);
            if var_bit != SHARED_SYMBOL_BIT {
                scoped &= !var_bit;
            } else if scoped & SHARED_SYMBOL_BIT != 0 {
                flags |= NodeFacts::SHARED_BIT_INEXACT;
            }
            sym_mask = outer | scoped;
        }
        NodeFacts {
            tree_size,
            height: height.saturating_add(1),
            flags,
            sym_mask,
        }
    }

    /// Gives the new `Symbol(s)` node `id` the next free-symbol bit and
    /// returns that bit.
    fn assign_symbol_bit(&mut self, s: SymbolId, id: ExprId) -> u64 {
        let k = self.bit_symbols.len();
        let index = if k < 63 {
            self.bit_symbols.push(id);
            k as u8
        } else {
            63
        };
        let i = s.0 as usize;
        if self.sym_bits.len() <= i {
            self.sym_bits.resize(i + 1, NO_SYMBOL_NODE);
        }
        self.sym_bits[i] = index;
        1u64 << index
    }
}

/// Writes `q` in decimal as `Display` does (`-3`, `7/2`): the bytes of a
/// `Num` node's sort key.  Machine-word values (most of them) skip the
/// big-integer conversion and the temporary strings.
fn write_decimal(q: &Q, out: &mut Vec<u8>) {
    use num_traits::ToPrimitive;
    use std::io::Write as _;
    // Writing into a `Vec` cannot fail.
    let _ = match (q.numer().to_i64(), q.denom().to_i64()) {
        (Some(n), Some(1)) => write!(out, "{n}"),
        (Some(n), Some(d)) => write!(out, "{n}/{d}"),
        _ => write!(out, "{q}"),
    };
}

// ---------------------------------------------------------------------------
// Orders on nodes
// ---------------------------------------------------------------------------

/// An [`ExprId`] ordered by its allocation number.
///
/// Ids themselves have no order: the number is allocation order, which
/// depends on everything the context built before.  This wrapper names
/// that order for the places that need *some* fixed order — dedup, a
/// sorted list used as a set or a hash key — and whose order never reaches
/// output.  It must not decide which term is printed first, which
/// candidate is chosen or which branch is tried: that is
/// [`Arena::canonical_cmp`]'s job.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ArenaOrder(pub(crate) ExprId);

impl Ord for ArenaOrder {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.0.cmp(&other.0.0)
    }
}

impl PartialOrd for ArenaOrder {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// The variant name [`Arena::canonical_cmp`] orders distinct heads with
/// equal sort keys by: a library or internal function's is `Apply`, as
/// until 0.49.
fn head_variant_name(n: &ExprNode) -> &'static str {
    match n {
        ExprNode::Lib(..) | ExprNode::Internal(..) => "Apply",
        other => other.variant_name(),
    }
}

impl Arena {
    /// The canonical order of two nodes: the order of their sort keys,
    /// with ties broken structurally.
    ///
    /// Sort keys are only a preorder (a key longer than
    /// [`MAX_KEY_BYTES`](crate::base::sort_key::MAX_KEY_BYTES) is cut and
    /// digested, and the keys of differently shaped nodes may concatenate
    /// to the same bytes), so equal keys are compared by variant name, then
    /// payload (number value, symbol name, interval flags), then arity, and
    /// then child by child in the same way.  The result depends only on the
    /// two expressions, never on the order in which nodes were allocated:
    /// it is the same in every context, and `Equal` exactly when `a == b`
    /// (the arena is hash-consed, so distinct ids differ structurally).
    ///
    /// Explicit stack, no recursion; since distinct ids differ, only the
    /// first differing child pair is ever descended into.
    pub(crate) fn canonical_cmp(&self, a: ExprId, b: ExprId) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        let mut stack: SmallVec<[(ExprId, ExprId); 8]> = SmallVec::new();
        stack.push((a, b));
        while let Some((p, q)) = stack.pop() {
            if p == q {
                continue;
            }
            let ord = self
                .sort_key(p)
                .cmp(self.sort_key(q))
                .then_with(|| self.node_head_cmp(p, q));
            if ord != Ordering::Equal {
                return ord;
            }
            let (cp, cq) = (self.node(p).children(), self.node(q).children());
            if cp.len() != cq.len() {
                return cp.len().cmp(&cq.len());
            }
            stack.extend(cp.iter().copied().zip(cq.iter().copied()).rev());
        }
        Ordering::Equal
    }

    /// [`canonical_cmp`](Self::canonical_cmp) of two nodes whose sort keys
    /// are equal, before their children: variant, then payload.
    fn node_head_cmp(&self, p: ExprId, q: ExprId) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        let (np, nq) = (self.node(p), self.node(q));
        match (np, nq) {
            (ExprNode::Num(m), ExprNode::Num(n)) => self.num(*m).cmp(self.num(*n)),
            (ExprNode::Symbol(s), ExprNode::Symbol(t))
            | (ExprNode::PhysicalConstant(s, _), ExprNode::PhysicalConstant(t, _))
            | (ExprNode::Apply(s, _), ExprNode::Apply(t, _)) => {
                self.symbol_name(*s).cmp(self.symbol_name(*t))
            }
            (ExprNode::Interval(_, _, f), ExprNode::Interval(_, _, g)) => {
                f.flag_bits().cmp(&g.flag_bits())
            }
            // A library function compares as the `Apply` it was until 0.49:
            // by name (a user function never has a library name; if it
            // had, the user function would come first).
            (ExprNode::Lib(f, _), ExprNode::Lib(g, _)) => f.name().cmp(g.name()),
            (ExprNode::Lib(f, _), ExprNode::Apply(t, _)) => {
                f.name().cmp(self.symbol_name(*t)).then(Ordering::Greater)
            }
            (ExprNode::Apply(s, _), ExprNode::Lib(g, _)) => {
                self.symbol_name(*s).cmp(g.name()).then(Ordering::Less)
            }
            // An internal function likewise, after a library function of
            // the same name (there is none).
            (ExprNode::Internal(f, _), ExprNode::Internal(g, _)) => f.name().cmp(g.name()),
            (ExprNode::Internal(f, _), ExprNode::Apply(t, _)) => {
                f.name().cmp(self.symbol_name(*t)).then(Ordering::Greater)
            }
            (ExprNode::Apply(s, _), ExprNode::Internal(g, _)) => {
                self.symbol_name(*s).cmp(g.name()).then(Ordering::Less)
            }
            (ExprNode::Internal(f, _), ExprNode::Lib(g, _)) => {
                f.name().cmp(g.name()).then(Ordering::Greater)
            }
            (ExprNode::Lib(f, _), ExprNode::Internal(g, _)) => {
                f.name().cmp(g.name()).then(Ordering::Less)
            }
            _ if std::mem::discriminant(np) == std::mem::discriminant(nq) => Ordering::Equal,
            // The variant's name (`Add`): reached only by distinct nodes
            // with equal sort keys.
            _ => head_variant_name(np).cmp(head_variant_name(nq)),
        }
    }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

impl Arena {
    /// Returns a reference to the [`ExprNode`] identified by `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` was not produced by this arena.
    pub fn node(&self, id: ExprId) -> &ExprNode {
        &self.nodes[id.0 as usize]
    }

    /// Returns the [`SortKey`] of the node identified by `id`: its bytes in
    /// the arena's key pool.
    ///
    /// # Panics
    ///
    /// Panics if `id` was not produced by this arena.
    #[inline]
    pub fn sort_key(&self, id: ExprId) -> &SortKey {
        let i = id.0 as usize;
        &self.key_bytes[self.key_starts[i]..self.key_starts[i + 1]]
    }

    /// The [`NodeFacts`] of the node `id`.
    #[inline]
    pub(crate) fn facts(&self, id: ExprId) -> NodeFacts {
        self.facts[id.0 as usize]
    }

    /// The free-symbol bit of the symbol `s` ([`NodeFacts`]); `None` when
    /// `s` has no `Symbol` node in this arena, so is free in nothing.
    #[inline]
    pub(crate) fn symbol_bit(&self, s: SymbolId) -> Option<u64> {
        match self.sym_bits.get(s.0 as usize) {
            Some(&k) if k != NO_SYMBOL_NODE => Some(1u64 << k),
            _ => None,
        }
    }

    /// Is `s` free in the tree rooted at `id`, as far as the facts decide:
    /// `False` when the symbol has no node or its bit is clear, `True` when
    /// its bit is set and its own, `Unknown` when it shares bit 63 (a walk
    /// must tell).
    #[inline]
    pub(crate) fn free_symbol_fact(&self, id: ExprId, s: SymbolId) -> Truth {
        let Some(bit) = self.symbol_bit(s) else {
            return Truth::False;
        };
        if self.facts(id).sym_mask & bit == 0 {
            Truth::False
        } else if bit != SHARED_SYMBOL_BIT {
            Truth::True
        } else {
            Truth::Unknown
        }
    }

    /// The free-symbol mask of `id` ([`NodeFacts`]): bits 0–62 exact, bit
    /// 63 ([`SHARED_SYMBOL_BIT`]) a superset.  A pruning walk skips a
    /// subtree whose mask has none of the bits it still looks for.
    #[inline]
    pub(crate) fn free_symbol_mask(&self, id: ExprId) -> u64 {
        self.facts(id).sym_mask
    }

    /// The `Symbol` node owning the exact free-symbol bit `k < 63`.
    #[inline]
    pub(crate) fn bit_symbol(&self, k: u32) -> Option<ExprId> {
        self.bit_symbols.get(k as usize).copied()
    }

    /// Is the tree rooted at `id` free of free symbols (a constant)?  O(1)
    /// unless a symbol sharing bit 63 is bound in it.
    pub(crate) fn is_symbol_free(&self, id: ExprId) -> bool {
        let f = self.facts(id);
        if f.sym_mask == 0 {
            true
        } else if f.sym_mask != SHARED_SYMBOL_BIT || f.flags & NodeFacts::SHARED_BIT_INEXACT == 0 {
            false
        } else {
            crate::base::walk::free_symbols(self, id).is_empty()
        }
    }

    /// The number of nodes of the unfolded tree rooted at `id` (shared
    /// subtrees counted at every occurrence, not the DAG's node count),
    /// capped: `min(size, cap)`.  O(1) from the [`NodeFacts`]; a count that
    /// saturated the facts' `u32` is redone when `cap` is larger.
    pub(crate) fn tree_size_capped(&self, id: ExprId, cap: usize) -> usize {
        let size = self.facts(id).tree_size;
        if size < u32::MAX || cap <= u32::MAX as usize {
            return (size as usize).min(cap);
        }
        let mut sizes: FxHashMap<ExprId, usize> = FxHashMap::default();
        for n in crate::base::walk::post_order_ids(self, id) {
            let mut total = 1usize;
            self.node(n).for_each_child(|c| {
                total = total.saturating_add(sizes.get(&c).copied().unwrap_or(1));
            });
            sizes.insert(n, total.min(cap));
        }
        sizes.get(&id).copied().unwrap_or(1)
    }

    /// Returns a reference to the rational number identified by `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` was not produced by this arena.
    pub fn num(&self, id: NumId) -> &Q {
        &self.numbers[id.0 as usize]
    }

    /// Returns the total number of interned expression nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// The number no other arena of this process has (see the field).
    pub(crate) fn serial(&self) -> u64 {
        self.serial
    }

    /// Open a work scope of `kind` allowing `cap` more interns, within
    /// what the innermost open scope of the same kind has left (used by
    /// [`WorkScope`](crate::base::budget::WorkScope), which closes it).
    pub(crate) fn open_work_scope(&mut self, kind: BudgetKind, cap: u64) {
        let mut end = self.interns.saturating_add(cap);
        if let Some(&(_, outer)) = self.work_scopes.iter().rev().find(|(k, _)| *k == kind) {
            end = end.min(outer);
        }
        self.work_scopes.push((kind, end));
    }

    /// Close the innermost work scope.
    pub(crate) fn close_work_scope(&mut self) {
        self.work_scopes.pop();
    }

    /// Has the innermost open work scope of `kind` been spent?  `false`
    /// when none is open.
    pub(crate) fn work_exhausted(&self, kind: BudgetKind) -> bool {
        self.work_scopes
            .iter()
            .rev()
            .find(|(k, _)| *k == kind)
            .is_some_and(|&(_, end)| self.interns > end)
    }

    /// Returns the interned name of a symbol.
    ///
    /// # Panics
    ///
    /// Panics if `id` was not produced by this arena's symbol table.
    pub fn symbol_name(&self, id: SymbolId) -> &str {
        self.symbols.name(id)
    }

    /// Interns the library-function node `f(args)` ([`ExprNode::Lib`])
    /// without folding.
    ///
    /// The callers pass an argument count `f` accepts (a literal call
    /// shape, or arguments already checked): input from outside the crate
    /// goes through [`Arena::func`](crate::base::func), which returns an
    /// error for a wrong count.  No name is interned: the node is
    /// identified by `f`, not by a symbol.
    pub(crate) fn lib_apply(&mut self, f: LibFn, args: &[ExprId]) -> ExprId {
        debug_assert!(
            f.arity().accepts(args.len()),
            "{f} called with {} argument(s)",
            args.len()
        );
        self.intern(ExprNode::Lib(f, args.iter().copied().collect()))
    }

    /// Get the stored assumptions for a symbol.
    pub(crate) fn symbol_assumptions(&self, id: SymbolId) -> crate::base::assumptions::Assumptions {
        self.symbols.get_assumptions(id)
    }

    /// Set assumptions for a symbol.
    ///
    /// The set is normalised with
    /// [`Assumptions::normalize_declared`](crate::base::assumptions::Assumptions::normalize_declared)
    /// (a symbol declared with a sign is finite, hence real, unless its
    /// finiteness was declared explicitly).
    ///
    /// A self-contradictory set (e.g. `positive` together with `negative`)
    /// is not stored, and the symbol keeps its previous assumptions.  The
    /// public entry points reject one with
    /// `SymplexError::ContradictoryAssumptions` (via
    /// `Assumptions::declare`) before calling this, and the crate's own
    /// callers pass consistent sets, so meeting one here is an internal
    /// bug (a `debug_assert!`).
    pub(crate) fn set_symbol_assumptions(
        &mut self,
        id: SymbolId,
        a: crate::base::assumptions::Assumptions,
    ) {
        let mut a = a;
        a.normalize_declared();
        let consistent = !a.is_contradictory();
        debug_assert!(
            consistent,
            "contradictory assumptions declared on symbol `{}`: {a}",
            self.symbols.name(id)
        );
        if consistent {
            self.constant_facts.get_mut().clear();
            self.symbols.set_assumptions(id, a);
        }
    }

    /// Returns the children of the node identified by `id`.
    ///
    /// For atom nodes the returned collection is empty.
    pub fn children(&self, id: ExprId) -> SmallVec<[ExprId; 6]> {
        self.nodes[id.0 as usize].children()
    }
}

// ---------------------------------------------------------------------------
// Decomposition helpers (used by canonicalization)
// ---------------------------------------------------------------------------

impl Arena {
    /// Decompose an expression into `(coefficient, symbolic_term)`.
    ///
    /// - `Num(n)` → `(n, self.one)`
    /// - `Mul([Num(n), rest...])` → `(n, Mul(rest))` or `(n, rest[0])` if single
    /// - `Neg(x)` → `(-1, x)`
    /// - anything else → `(1, itself)`
    pub(crate) fn as_coeff_term(&mut self, id: ExprId) -> (Q, ExprId) {
        match self.node(id).clone() {
            ExprNode::Num(nid) => (self.num(nid).clone(), self.one),
            ExprNode::Neg(inner) => {
                let neg_one = -Ratio::<BigInt>::one();
                (neg_one, inner)
            }
            ExprNode::Mul(args) if !args.is_empty() => {
                if let ExprNode::Num(nid) = self.node(args[0]) {
                    let coeff = self.num(*nid).clone();
                    let rest = &args[1..];
                    let term = match rest.len() {
                        0 => self.one,
                        1 => rest[0],
                        _ => {
                            let sv: SmallVec<[ExprId; 6]> = rest.iter().copied().collect();
                            self.intern(ExprNode::Mul(sv))
                        }
                    };
                    (coeff, term)
                } else {
                    (Ratio::one(), id)
                }
            }
            _ => (Ratio::one(), id),
        }
    }

    /// Decompose an expression into `(base, exponent)`.
    ///
    /// - `Pow(base, exp)` → `(base, exp)`
    /// - anything else → `(itself, self.one)`
    pub(crate) fn as_base_exp(&self, id: ExprId) -> (ExprId, ExprId) {
        match self.node(id) {
            ExprNode::Pow(base, exp) => (*base, *exp),
            _ => (id, self.one),
        }
    }

    /// Construct `coefficient * term`, simplifying trivial cases.
    ///
    /// - coeff == 0 → self.zero
    /// - coeff == 1 → term
    /// - otherwise → canonical Mul([Num(coeff), term])
    ///
    /// Uses [`canon_mul`](crate::base::canon::canon_mul) to ensure the result
    /// is properly flattened (no nested Mul nodes).
    pub(crate) fn make_coeff_term(&mut self, coeff: Q, term: ExprId) -> ExprId {
        if coeff.is_zero() {
            return self.zero;
        }
        if coeff == Ratio::one() {
            return term;
        }
        let coeff_id = {
            let nid = self.intern_num(coeff);
            self.intern(ExprNode::Num(nid))
        };
        if term == self.one {
            return coeff_id;
        }
        // Use canon_mul to ensure flattening (e.g., coeff * Mul([a, b]) → Mul([coeff, a, b]))
        crate::base::canon::canon_mul(self, &[coeff_id, term])
    }

    /// Check if an expression is a numeric literal and return its value.
    pub(crate) fn as_num(&self, id: ExprId) -> Option<&Q> {
        match self.node(id) {
            ExprNode::Num(nid) => Some(self.num(*nid)),
            _ => None,
        }
    }

    /// Structural zero check — O(1), just compares ExprId.
    pub fn is_zero_structural(&self, id: ExprId) -> bool {
        id == self.zero
    }

    /// Structural one check — O(1).
    pub fn is_one_structural(&self, id: ExprId) -> bool {
        id == self.one
    }
}

// ---------------------------------------------------------------------------
// Convenience construction helpers
// ---------------------------------------------------------------------------

impl Arena {
    /// Creates (or retrieves) an integer expression from an `i64`.
    pub fn int(&mut self, n: i64) -> ExprId {
        let num_id = self.intern_num(Ratio::from_integer(BigInt::from(n)));
        self.intern(ExprNode::Num(num_id))
    }

    /// Creates (or retrieves) an integer expression from a [`BigInt`].
    pub fn big_int(&mut self, n: BigInt) -> ExprId {
        let num_id = self.intern_num(Ratio::from_integer(n));
        self.intern(ExprNode::Num(num_id))
    }

    /// Creates (or retrieves) the numeric expression for an exact rational.
    /// (The one place the solvers, integrators and polynomial bridges turn
    /// a `Ratio<BigInt>` back into a node.)
    pub fn num_ratio(&mut self, r: Q) -> ExprId {
        let num_id = self.intern_num(r);
        self.intern(ExprNode::Num(num_id))
    }

    /// Creates (or retrieves) a rational expression `p/q`.
    ///
    /// The ratio is automatically reduced to lowest terms by `num_rational`.
    /// A zero denominator does not panic: `p/0` with `p ≠ 0` is `zoo`
    /// (complex infinity) and `0/0` is `nan`, mirroring `1/0` built through
    /// division.
    pub fn rational(&mut self, p: i64, q: i64) -> ExprId {
        if q == 0 {
            return if p == 0 {
                self.nan
            } else {
                self.complex_infinity
            };
        }
        let num_id = self.intern_num(Ratio::new(BigInt::from(p), BigInt::from(q)));
        self.intern(ExprNode::Num(num_id))
    }

    /// Creates (or retrieves) a symbolic variable or constant by name.
    pub fn symbol(&mut self, name: &str) -> ExprId {
        let sym_id = self.symbols.intern(name);
        self.intern(ExprNode::Symbol(sym_id))
    }

    /// An internal dummy symbol declared positive (a limit variable
    /// `x → +∞`, a `w → 0⁺`).  Symbols are interned by name, so `name` must
    /// be reserved for positive dummies: every call re-declares it positive.
    pub(crate) fn positive_symbol(&mut self, name: &str) -> ExprId {
        let sym_id = self.symbols.intern(name);
        let mut a = crate::base::assumptions::Assumptions::default();
        a.assert_true(crate::base::assumptions::Props::POSITIVE);
        a.forward_chain();
        self.set_symbol_assumptions(sym_id, a);
        self.intern(ExprNode::Symbol(sym_id))
    }

    /// Creates an `Add` node with full canonicalization.
    ///
    /// Flattens nested Adds, combines like terms, sorts by canonical
    /// order, drops zero-coefficient terms, propagates NaN.
    pub fn add(&mut self, args: &[ExprId]) -> ExprId {
        // Canonical implementation in canon.rs
        crate::base::canon::canon_add(self, args)
    }

    /// Creates a `Mul` node with full canonicalization.
    pub fn mul(&mut self, args: &[ExprId]) -> ExprId {
        crate::base::canon::canon_mul(self, args)
    }

    /// Creates a `Pow` node with canonicalization.
    pub fn pow(&mut self, base: ExprId, exp: ExprId) -> ExprId {
        crate::base::canon::canon_pow(self, base, exp)
    }

    /// Creates a `Neg` node with canonicalization.
    pub fn neg(&mut self, expr: ExprId) -> ExprId {
        crate::base::canon::canon_neg(self, expr)
    }

    /// Creates a division expression `a / b` as `a * b^(-1)`.
    pub fn div(&mut self, a: ExprId, b: ExprId) -> ExprId {
        let neg_one = self.neg_one;
        let b_inv = self.pow(b, neg_one);
        self.mul(&[a, b_inv])
    }

    /// Structural substitution: replace `old` with `new` in `expr`.
    ///
    /// Delegates to [`subs::subs`](crate::transforms::subs::subs).
    pub fn subs_structural(&mut self, expr: ExprId, old: ExprId, new: ExprId) -> ExprId {
        crate::transforms::subs::subs(self, expr, old, new)
    }

    /// Simultaneous structural substitution of multiple pairs.
    ///
    /// Delegates to [`subs::subs_map`](crate::transforms::subs::subs_map).
    pub fn subs_map_structural(
        &mut self,
        expr: ExprId,
        replacements: &[(ExprId, ExprId)],
    ) -> ExprId {
        crate::transforms::subs::subs_map(self, expr, replacements)
    }

    /// Differentiate `expr` with respect to `var`.
    ///
    /// Delegates to [`diff::diff`](crate::transforms::diff::diff).
    pub fn diff_wrt(&mut self, expr: ExprId, var: ExprId) -> ExprId {
        crate::transforms::diff::diff(self, expr, var)
    }

    /// Create a formal (unevaluated) derivative node: d/d(var) expr.
    ///
    /// Unlike `diff_wrt` which evaluates the derivative using the chain rule,
    /// this creates a symbolic `Derivative(expr, var)` node that stays
    /// unevaluated. Used for ODE construction where the derivative is a
    /// structural placeholder, not a computed value.
    pub fn formal_diff(&mut self, expr: ExprId, var: ExprId) -> ExprId {
        self.intern(ExprNode::Derivative(expr, var))
    }

    /// Algebraic expansion: distribute products over sums, expand
    /// integer powers of sums.
    ///
    /// Delegates to [`expand::expand`](crate::transforms::expand::expand).
    pub fn expand_expr(&mut self, expr: ExprId) -> ExprId {
        crate::transforms::expand::expand(self, expr)
    }

    /// Exact evaluation of known special values (sin(0)→0, cos(π)→-1, etc.).
    ///
    /// Delegates to [`eval::eval`](crate::transforms::eval::eval).
    pub fn eval_expr(&mut self, expr: ExprId) -> ExprId {
        crate::transforms::eval::eval(self, expr)
    }

    /// Solve `expr = 0` for `var`.
    ///
    /// Returns a vector of solutions (none when none is found). Delegates to
    /// [`known_real_roots`](crate::transforms::solve::known_real_roots).
    #[deprecated(
        since = "0.50.0",
        note = "an empty list means both \"no root\" and \"not solved\"; use `Arena::solve_roots`"
    )]
    pub fn solve_for(
        &mut self,
        expr: ExprId,
        var: ExprId,
    ) -> Vec<crate::transforms::solve::Solution> {
        crate::transforms::solve::known_real_roots(self, expr, var)
            .into_iter()
            .map(|value| crate::transforms::solve::Solution { value })
            .collect()
    }

    /// The principal roots of `expr = 0` in `var` taken real (as
    /// [`solve_for`](Self::solve_for)), or why there is no list:
    /// `NoSolution` when there provably is none (`eˣ = −1`),
    /// `InfiniteSolutions` for an identity, `ComputationFailed` when the
    /// roots are not found or not all of them.  `solve_for` returns an empty
    /// list in all three cases.
    ///
    /// # Errors
    ///
    /// As above.
    pub fn solve_roots(
        &mut self,
        expr: ExprId,
        var: ExprId,
    ) -> Result<Vec<ExprId>, crate::base::errors::SymplexError> {
        use crate::base::errors::SymplexError;
        use crate::transforms::solve::{Refusal, roots_or_refusal, solve_classified_real};
        let outcome = solve_classified_real(self, expr, var);
        match roots_or_refusal(self, outcome) {
            Ok(roots) if !roots.is_empty() => Ok(roots),
            Ok(_) => Err(SymplexError::ComputationFailed {
                operation: "solve",
                reason: "no root found".into(),
            }),
            Err(Refusal::Identity) => Err(SymplexError::InfiniteSolutions {
                operation: "solve",
                reason: "the equation is an identity".into(),
            }),
            Err(Refusal::NoSolution(reason)) => Err(SymplexError::NoSolution {
                operation: "solve",
                reason,
            }),
            Err(Refusal::Failed(reason)) => Err(SymplexError::ComputationFailed {
                operation: "solve",
                reason,
            }),
        }
    }

    /// Cancel common polynomial factors in a rational expression.
    ///
    /// `var` is the symbol to treat as the polynomial variable.
    /// Delegates to [`polybridge::cancel`](crate::poly::polybridge::cancel).
    pub fn cancel_expr(&mut self, expr: ExprId, var: ExprId) -> ExprId {
        crate::poly::polybridge::cancel(self, expr, var)
    }

    /// Group an expression by powers of `var`.
    ///
    /// Converts to a univariate polynomial in `var` and rebuilds,
    /// naturally grouping coefficients by power.
    /// Returns unchanged if not polynomial in `var`.
    /// Delegates to [`polybridge::collect`](crate::poly::polybridge::collect).
    pub fn collect_expr(&mut self, expr: ExprId, var: ExprId) -> ExprId {
        crate::poly::polybridge::collect(self, expr, var)
    }

    /// Combine all fractions in `expr` — at every depth — into one quotient.
    ///
    /// Delegates to [`polybridge::together_deep`](crate::poly::polybridge::together_deep).
    pub fn together_expr(&mut self, expr: ExprId) -> ExprId {
        crate::poly::polybridge::together_deep(self, expr)
    }

    /// Compute the indefinite integral of `expr` with respect to `var`.
    ///
    /// Returns an unevaluated `Integral` node for integrands that
    /// don't match any known rule.
    /// Delegates to [`integrate::integrate`](crate::transforms::integrate::integrate).
    pub fn integrate_expr(&mut self, expr: ExprId, var: ExprId) -> ExprId {
        crate::transforms::integrate::integrate(self, expr, var)
    }

    /// The series of `expr` in `var` around `point` with every term of
    /// exponent `< order`: the real two-sided asymptotic expansion (the
    /// contract in [`series::series`](crate::calculus::series::series),
    /// which handles poles itself).
    ///
    /// Returns `Err` if no such expansion exists or it cannot be
    /// established.  The fallback to
    /// [`laurent_series`](crate::calculus::series::laurent_series) only
    /// succeeds for a function meromorphic at `point`, whose Laurent series
    /// is that expansion too.
    pub fn series_expr(
        &mut self,
        expr: ExprId,
        var: ExprId,
        point: ExprId,
        order: u32,
    ) -> Result<ExprId, crate::base::errors::SymplexError> {
        // Try Taylor series first.
        match crate::calculus::series::series(self, expr, var, point, order) {
            Ok(result) => Ok(result),
            Err(_) => {
                // A meromorphic function's Laurent series is its real expansion.
                crate::calculus::series::laurent_series(self, expr, var, point, order)
            }
        }
    }

    /// Factor a polynomial expression into a product of linear factors.
    ///
    /// Returns the expression unchanged if no rational roots are found.
    /// Delegates to [`factor::factor`](crate::simplify::factor::factor).
    pub fn factor_expr(&mut self, expr: ExprId, var: ExprId) -> ExprId {
        crate::simplify::factor::factor(self, expr, var)
    }

    /// Return the degree of `expr` as a polynomial in `var`.
    ///
    /// Returns `None` if not polynomial or if the zero polynomial.
    /// Delegates to [`polybridge::poly_degree`](crate::poly::polybridge::poly_degree).
    pub fn degree_of(&self, expr: ExprId, var: ExprId) -> Option<usize> {
        crate::poly::polybridge::poly_degree(self, expr, var)
    }

    /// Return the coefficients of `expr` as a polynomial in `var`,
    /// in ascending degree order.
    ///
    /// Returns `None` if not polynomial in `var`.
    /// Delegates to [`polybridge::poly_coefficients`](crate::poly::polybridge::poly_coefficients).
    pub fn coefficients_of(&mut self, expr: ExprId, var: ExprId) -> Option<Vec<ExprId>> {
        crate::poly::polybridge::poly_coefficients(self, expr, var)
    }

    /// Compute the limit of `expr` as `var` approaches `point`.
    ///
    /// Returns `Err` if the limit cannot be determined.
    /// Delegates to [`limit::limit`](crate::calculus::limit::limit).
    pub fn limit_expr(
        &mut self,
        expr: ExprId,
        var: ExprId,
        point: ExprId,
    ) -> Result<ExprId, crate::base::errors::SymplexError> {
        crate::calculus::limit::limit(self, expr, var, point)
    }

    /// Decompose `expr` into (numerator, denominator), deeply: rational
    /// literals split into integers, sums are combined over a common
    /// denominator, products and integer powers distribute.
    ///
    /// Delegates to [`polybridge::fraction_parts`](crate::poly::polybridge::fraction_parts).
    pub fn as_numer_denom_expr(&mut self, expr: ExprId) -> (ExprId, ExprId) {
        crate::poly::polybridge::fraction_parts(self, expr)
    }

    /// `false` when `expr` certainly does not vanish identically as a
    /// rational function of its generators (one residue at a pseudo-random
    /// point, no expansion).
    ///
    /// Delegates to [`polybridge::may_vanish_identically`](crate::poly::polybridge::may_vanish_identically).
    pub(crate) fn may_vanish_identically(&self, expr: ExprId) -> bool {
        crate::poly::polybridge::may_vanish_identically(self, expr)
    }

    /// `false` when no denominator of `expr` (outside every generator)
    /// vanishes identically (one residue pass, no expansion).
    ///
    /// Delegates to [`polybridge::may_have_vanishing_denominator`](crate::poly::polybridge::may_have_vanishing_denominator).
    pub(crate) fn may_have_vanishing_denominator(&self, expr: ExprId) -> bool {
        crate::poly::polybridge::may_have_vanishing_denominator(self, expr)
    }

    /// Is `expr` structurally 0, or 0 once multiplied out, after the
    /// residue test said it may vanish identically?
    ///
    /// Delegates to [`polybridge::vanishes_identically`](crate::poly::polybridge::vanishes_identically).
    pub(crate) fn vanishes_identically(&mut self, expr: ExprId) -> bool {
        crate::poly::polybridge::vanishes_identically(self, expr)
    }

    /// Does `expr` vanish identically by an identity `simplify` can prove
    /// (zero at certified sample points, then exactly 0)?
    ///
    /// Delegates to [`simplify_engine::vanishes_by_identity`](crate::simplify::simplify_engine::vanishes_by_identity).
    pub(crate) fn vanishes_by_identity(&mut self, expr: ExprId) -> bool {
        crate::simplify::simplify_engine::vanishes_by_identity(self, expr)
    }

    /// Partial fraction decomposition of `expr` with respect to `var`.
    ///
    /// Delegates to [`apart::apart`](crate::transforms::apart::apart).
    pub fn apart_expr(&mut self, expr: ExprId, var: ExprId) -> ExprId {
        crate::transforms::apart::apart(self, expr, var)
    }

    /// Expand trigonometric functions with composite arguments.
    ///
    /// `sin(a+b) → sin(a)cos(b) + cos(a)sin(b)`, etc.
    /// Delegates to [`trig_expand::expand_trig`](crate::simplify::trig_expand::expand_trig).
    pub fn expand_trig_expr(&mut self, expr: ExprId) -> ExprId {
        crate::simplify::trig_expand::expand_trig(self, expr)
    }

    /// Expand logarithmic expressions.
    ///
    /// `ln(a*b) → ln(a)+ln(b)`, `ln(a^n) → n*ln(a)`, etc., where they hold.
    /// Delegates to [`log_expand::expand_log`](crate::simplify::log_expand::expand_log).
    pub fn expand_log_expr(&mut self, expr: ExprId) -> ExprId {
        crate::simplify::log_expand::expand_log(self, expr)
    }

    /// Combine logarithmic terms where it is exact: `ln(a)+ln(b) → ln(a*b)`,
    /// `n*ln(a) → ln(a^n)`.
    /// Delegates to [`log_combine::log_combine`](crate::simplify::log_combine::log_combine).
    pub fn log_combine_expr(&mut self, expr: ExprId) -> ExprId {
        crate::simplify::log_combine::log_combine(self, expr)
    }

    /// Apply trig product-to-sum and double-angle identities.
    /// `sin(a)*cos(b) → ½[sin(a+b)+sin(a-b)]`, etc.
    /// Delegates to [`trig_combine::trig_combine`](crate::simplify::trig_combine::trig_combine).
    pub fn trig_combine_expr(&mut self, expr: ExprId) -> ExprId {
        crate::simplify::trig_combine::trig_combine(self, expr)
    }

    /// Try multiple simplification strategies and return the simplest result.
    /// Delegates to [`simplify_engine::smart_simplify`](crate::simplify::simplify_engine::smart_simplify).
    pub fn smart_simplify_expr(&mut self, expr: ExprId) -> ExprId {
        crate::simplify::simplify_engine::smart_simplify(self, expr)
    }

    /// Count the number of operations (non-atom nodes) in an expression.
    /// Delegates to [`simplify_engine::count_ops`](crate::simplify::simplify_engine::count_ops).
    pub fn count_ops(&self, expr: ExprId) -> usize {
        crate::simplify::simplify_engine::count_ops(self, expr)
    }

    /// Dedicated trigonometric simplification.
    ///
    /// Tries multiple Pythagorean replacement strategies and picks the
    /// result with the fewest operations.
    /// Delegates to [`trigsimp::trigsimp`](crate::simplify::trigsimp::trigsimp).
    pub fn trigsimp_expr(&mut self, expr: ExprId) -> ExprId {
        crate::simplify::trigsimp::trigsimp(self, expr)
    }

    /// Simplify combinatorial expressions (factorials, binomials).
    ///
    /// Cancels common factorial terms in products, e.g.
    /// `n! / (n-1)!` → `n`.
    /// Delegates to [`combsimp::combsimp`](crate::simplify::combsimp::combsimp).
    pub fn combsimp_expr(&mut self, expr: ExprId) -> ExprId {
        crate::simplify::combsimp::combsimp(self, expr)
    }

    /// Find a simple closed-form for a numerical expression.
    ///
    /// Tries rational approximations, π-multiples, and square roots
    /// within the given `tolerance`.
    /// Delegates to [`nsimplify::nsimplify`](crate::simplify::nsimplify::nsimplify).
    pub fn nsimplify_expr(&mut self, expr: ExprId, tol: f64) -> ExprId {
        crate::simplify::nsimplify::nsimplify(self, expr, tol)
    }

    /// Combine like bases in products with symbolic exponents.
    ///
    /// `x^a * x^b → x^(a+b)` even when `a` and `b` are not numeric.
    /// Delegates to [`powsimp::powsimp`](crate::simplify::powsimp::powsimp).
    pub fn powsimp_expr(&mut self, expr: ExprId) -> ExprId {
        crate::simplify::powsimp::powsimp(self, expr)
    }

    /// Rewrite trig functions as complex exponentials.
    ///
    /// `sin(x) → (exp(ix) − exp(−ix)) / (2i)`, etc.
    /// Delegates to [`rewrite::rewrite_as_exp`](crate::simplify::rewrite::rewrite_as_exp).
    pub fn rewrite_as_exp_expr(&mut self, expr: ExprId) -> ExprId {
        crate::simplify::rewrite::rewrite_as_exp(self, expr)
    }

    /// Rewrite complex exponentials as trig functions (Euler's formula).
    ///
    /// `exp(ix) → cos(x) + i·sin(x)`, etc.
    /// Delegates to [`rewrite::rewrite_as_trig`](crate::simplify::rewrite::rewrite_as_trig).
    pub fn rewrite_as_trig_expr(&mut self, expr: ExprId) -> ExprId {
        crate::simplify::rewrite::rewrite_as_trig(self, expr)
    }

    /// Factor out the GCD of numeric coefficients from a sum.
    /// `2x + 2y → 2(x + y)`.
    /// Delegates to [`factor_terms::factor_terms`](crate::simplify::factor_terms::factor_terms).
    pub fn factor_terms_expr(&mut self, expr: ExprId) -> ExprId {
        crate::simplify::factor_terms::factor_terms(self, expr)
    }

    /// Factor out the GCD of numeric coefficients, returning `(gcd, inner)` as expression IDs.
    /// Delegates to [`factor_terms::symbolic_factor_terms_pair`](crate::simplify::factor_terms::symbolic_factor_terms_pair).
    pub fn factor_terms_pair_expr(&mut self, expr: ExprId) -> (ExprId, ExprId) {
        crate::simplify::factor_terms::symbolic_factor_terms_pair(self, expr)
    }

    /// Rationalize the denominator of a fraction containing square roots.
    /// `1/√2 → √2/2`, `1/(1+√2) → √2-1`.
    /// Delegates to [`radsimp::rationalize_denom`](crate::simplify::radsimp::rationalize_denom).
    pub fn rationalize_denom_expr(&mut self, expr: ExprId) -> ExprId {
        crate::simplify::radsimp::rationalize_denom(self, expr)
    }

    /// Separate variables in a multiplicative expression.
    ///
    /// Given a list of variables, partitions the top-level factors
    /// by which variables they depend on. Returns a vec of
    /// `(dependent_vars, product_of_factors)` pairs.
    ///
    /// Delegates to [`separatevars::separatevars`](crate::domains::separatevars::separatevars).
    pub fn separatevars_expr(
        &mut self,
        expr: ExprId,
        vars: &[ExprId],
    ) -> Vec<(Vec<ExprId>, ExprId)> {
        crate::domains::separatevars::separatevars(self, expr, vars)
    }

    /// Decompose an expression into real and imaginary parts.
    /// Returns `(re, im)` such that `expr = re + im * i`.
    /// Delegates to [`complex::as_real_imag`](crate::base::complex::as_real_imag).
    pub fn as_real_imag_expr(&mut self, expr: ExprId) -> (ExprId, ExprId) {
        crate::base::complex::as_real_imag(self, expr)
    }

    /// Compute the polynomial GCD of `a` and `b` with respect to `var`.
    ///
    /// Returns `None` if either expression is not polynomial in `var`.
    pub fn poly_gcd_expr(&mut self, a: ExprId, b: ExprId, var: ExprId) -> Option<ExprId> {
        let pa = crate::poly::polybridge::expr_to_poly(self, a, var)?;
        let pb = crate::poly::polybridge::expr_to_poly(self, b, var)?;
        let g = crate::poly::Poly::gcd(&pa, &pb);
        Some(crate::poly::polybridge::poly_to_expr(self, &g, var))
    }

    /// Compute the polynomial LCM of `a` and `b` with respect to `var`.
    ///
    /// Returns `None` if either expression is not polynomial in `var`.
    pub fn poly_lcm_expr(&mut self, a: ExprId, b: ExprId, var: ExprId) -> Option<ExprId> {
        let pa = crate::poly::polybridge::expr_to_poly(self, a, var)?;
        let pb = crate::poly::polybridge::expr_to_poly(self, b, var)?;
        let g = crate::poly::Poly::gcd(&pa, &pb);
        if g.is_zero() {
            return None;
        }
        // lcm(a,b) = a * b / gcd(a,b)
        let product = &pa * &pb;
        let (lcm, _rem) = product.div_rem(&g);
        Some(crate::poly::polybridge::poly_to_expr(self, &lcm, var))
    }

    /// Creates a subtraction expression `a - b` as `a + neg(b)`.
    pub fn sub(&mut self, a: ExprId, b: ExprId) -> ExprId {
        let neg_b = self.neg(b);
        self.add(&[a, neg_b])
    }

    /// Creates a `Sin` node.
    pub fn sin(&mut self, expr: ExprId) -> ExprId {
        self.intern(ExprNode::Sin(expr))
    }

    /// Creates a `Cos` node.
    pub fn cos(&mut self, expr: ExprId) -> ExprId {
        self.intern(ExprNode::Cos(expr))
    }

    /// Creates a `Tan` node.
    pub fn tan(&mut self, expr: ExprId) -> ExprId {
        self.intern(ExprNode::Tan(expr))
    }

    /// Creates an `Exp` (natural exponential) node.
    ///
    /// Named `exp` — computes e^x.
    pub fn exp(&mut self, expr: ExprId) -> ExprId {
        self.intern(ExprNode::Exp(expr))
    }

    /// Creates an `Ln` (natural logarithm) node.
    pub fn ln(&mut self, expr: ExprId) -> ExprId {
        self.intern(ExprNode::Ln(expr))
    }

    /// Creates a principal square root node: `√expr` = `expr^(1/2)`.
    pub fn sqrt(&mut self, expr: ExprId) -> ExprId {
        let half = self.rational(1, 2);
        self.pow(expr, half)
    }

    /// Creates a cube root node: `∛expr` = `expr^(1/3)`.
    pub fn cbrt(&mut self, expr: ExprId) -> ExprId {
        let third = self.rational(1, 3);
        self.pow(expr, third)
    }

    /// Creates an nth-root node: `expr^(1/n)`.
    pub fn nthroot(&mut self, expr: ExprId, n: i64) -> ExprId {
        let frac = self.rational(1, n);
        self.pow(expr, frac)
    }

    /// Creates a canonical `Abs` (absolute value / complex modulus) node.
    ///
    /// Canonicalization rules:
    /// - `abs(n)` for numeric `n` → `|n|`
    /// - `abs(abs(x))` → `abs(x)` (idempotent, since `|x| ≥ 0`)
    /// - `abs(c * x)` → `|c| * abs(x)` (factor out numeric coefficient)
    /// - `abs(NaN)` → `NaN`, `abs(±∞)` → `∞`
    pub fn abs(&mut self, expr: ExprId) -> ExprId {
        match self.node(expr).clone() {
            ExprNode::NaN => return self.nan,
            ExprNode::Infinity | ExprNode::NegInfinity | ExprNode::ComplexInfinity => {
                return self.infinity;
            }
            ExprNode::Abs(_) => return expr, // abs(abs(x)) = abs(x)
            ExprNode::Num(nid) => {
                let r = self.num(nid).clone();
                if r < Ratio::zero() {
                    let pos = -r;
                    let nid = self.intern_num(pos);
                    return self.intern(ExprNode::Num(nid));
                }
                return expr; // already non-negative
            }
            _ => {}
        }
        // Factor out numeric coefficient: abs(c*x) = |c|*abs(x)
        let (coeff, term) = self.as_coeff_term(expr);
        if coeff == Ratio::one() {
            self.intern(ExprNode::Abs(expr))
        } else {
            let abs_coeff = if coeff < Ratio::zero() { -coeff } else { coeff };
            let abs_term = self.abs(term); // recursive: canonicalises inner
            self.make_coeff_term(abs_coeff, abs_term)
        }
    }

    /// Creates a canonical `Sign` (sign function) node:
    /// `1` if `x > 0`, `-1` if `x < 0`, `0` if `x = 0`.
    ///
    /// Canonicalization rules:
    /// - `sign(n)` for numeric `n` → `1`, `-1`, or `0`
    /// - `sign(c * x)` for positive numeric `c` → `sign(x)`
    /// - `sign(c * x)` for negative numeric `c` → `-sign(x)`
    /// - `sign(NaN)` → `NaN`, `sign(∞)` → `1`, `sign(-∞)` → `-1`
    pub fn sign(&mut self, expr: ExprId) -> ExprId {
        match self.node(expr).clone() {
            ExprNode::NaN => return self.nan,
            ExprNode::Infinity => return self.one,
            ExprNode::NegInfinity => return self.neg_one,
            ExprNode::Num(nid) => {
                let r = self.num(nid).clone();
                if r > Ratio::zero() {
                    return self.one;
                } else if r < Ratio::zero() {
                    return self.neg_one;
                } else {
                    return self.zero;
                }
            }
            _ => {}
        }
        // Factor out numeric coefficient: sign(c*x) = sign(c)*sign(x)
        let (coeff, term) = self.as_coeff_term(expr);
        if coeff < Ratio::zero() {
            // sign(-c * x) = -sign(x) for c > 0
            let sign_term = self.sign(term); // recursive: canonicalises inner
            self.neg(sign_term)
        } else if coeff == Ratio::one() {
            self.intern(ExprNode::Sign(expr))
        } else {
            // sign(c * x) = sign(x) for c > 0, c ≠ 1
            self.sign(term) // recursive: canonicalises inner
        }
    }

    /// Creates a `Floor` node: ⌊x⌋ (greatest integer ≤ x).
    pub fn floor(&mut self, expr: ExprId) -> ExprId {
        self.intern(ExprNode::Floor(expr))
    }

    /// Creates a `Ceiling` node: ⌈x⌉ (least integer ≥ x).
    pub fn ceiling(&mut self, expr: ExprId) -> ExprId {
        self.intern(ExprNode::Ceiling(expr))
    }

    /// Creates an `Asin` (inverse sine / arcsin) node.
    pub fn asin(&mut self, expr: ExprId) -> ExprId {
        self.intern(ExprNode::Asin(expr))
    }

    /// Creates an `Acos` (inverse cosine / arccos) node.
    pub fn acos(&mut self, expr: ExprId) -> ExprId {
        self.intern(ExprNode::Acos(expr))
    }

    /// Creates an `Atan` (inverse tangent / arctan) node.
    pub fn atan(&mut self, expr: ExprId) -> ExprId {
        self.intern(ExprNode::Atan(expr))
    }

    /// Creates an `Atan2` (two-argument arctangent) node: `atan2(y, x)`.
    ///
    /// Returns the angle in (-π, π] between the positive x-axis and
    /// the point (x, y). Correctly handles all four quadrants.
    pub fn atan2(&mut self, y: ExprId, x: ExprId) -> ExprId {
        self.intern(ExprNode::Atan2(y, x))
    }

    /// Creates a `Sinh` (hyperbolic sine) node.
    pub fn sinh(&mut self, expr: ExprId) -> ExprId {
        self.intern(ExprNode::Sinh(expr))
    }

    /// Creates a `Cosh` (hyperbolic cosine) node.
    pub fn cosh(&mut self, expr: ExprId) -> ExprId {
        self.intern(ExprNode::Cosh(expr))
    }

    /// Creates a `Tanh` (hyperbolic tangent) node.
    pub fn tanh(&mut self, expr: ExprId) -> ExprId {
        self.intern(ExprNode::Tanh(expr))
    }

    /// Creates an `Asinh` (inverse hyperbolic sine) node.
    pub fn asinh(&mut self, expr: ExprId) -> ExprId {
        self.intern(ExprNode::Asinh(expr))
    }

    /// Creates an `Acosh` (inverse hyperbolic cosine) node.
    pub fn acosh(&mut self, expr: ExprId) -> ExprId {
        self.intern(ExprNode::Acosh(expr))
    }

    /// Creates an `Atanh` (inverse hyperbolic tangent) node.
    pub fn atanh(&mut self, expr: ExprId) -> ExprId {
        self.intern(ExprNode::Atanh(expr))
    }

    /// Creates a `Gamma` node: Γ(x).
    pub fn gamma(&mut self, arg: ExprId) -> ExprId {
        self.intern(ExprNode::Gamma(arg))
    }

    /// Creates a `LogGamma` node: ln(Γ(x)).
    pub fn log_gamma(&mut self, arg: ExprId) -> ExprId {
        self.intern(ExprNode::LogGamma(arg))
    }

    /// Creates a `Digamma` node: ψ(x) = Γ'(x)/Γ(x).
    pub fn digamma(&mut self, arg: ExprId) -> ExprId {
        self.intern(ExprNode::Digamma(arg))
    }

    /// Creates an `Erf` (error function) node.
    pub fn erf(&mut self, arg: ExprId) -> ExprId {
        self.intern(ExprNode::Erf(arg))
    }

    /// Creates an `Erfc` (complementary error function) node.
    pub fn erfc(&mut self, arg: ExprId) -> ExprId {
        self.intern(ExprNode::Erfc(arg))
    }

    /// Creates a `Beta` node: B(a, b) = Γ(a)Γ(b)/Γ(a+b).
    pub fn beta(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(ExprNode::Beta(a, b))
    }

    // ── Complex analysis ─────────────────────────────────────────────

    /// Real part `re(z)`.
    ///
    /// Evaluates at construction whenever the real part is determinable
    /// (numbers, constants, symbols assumed real, sums, real scalings,
    /// fully decomposable products/powers, `exp`, `sin`, `cos`, …).
    /// Otherwise returns an [`ExprNode::Re`] node.
    /// See [`complex::re`](crate::base::complex::re).
    pub fn re(&mut self, z: ExprId) -> ExprId {
        crate::base::complex::re(self, z)
    }

    /// Imaginary part `im(z)` (real-valued; `z = re(z) + i·im(z)`).
    /// See [`complex::im`](crate::base::complex::im).
    pub fn im(&mut self, z: ExprId) -> ExprId {
        crate::base::complex::im(self, z)
    }

    /// Complex conjugate `conjugate(z)`.
    /// See [`complex::conjugate`](crate::base::complex::conjugate).
    pub fn conjugate(&mut self, z: ExprId) -> ExprId {
        crate::base::complex::conjugate(self, z)
    }

    /// Principal complex argument `arg(z) ∈ (−π, π]`.
    /// See [`complex::arg`](crate::base::complex::arg).
    pub fn arg(&mut self, z: ExprId) -> ExprId {
        crate::base::complex::arg(self, z)
    }

    // ── Special functions (0.2) ───────────────────────────────────────

    /// Sine integral `Si(x)`, folding exact values (`Si(0) = 0`,
    /// `Si(∞) = π/2`, oddness `Si(−x) = −Si(x)`).
    pub fn si(&mut self, x: ExprId) -> ExprId {
        crate::transforms::eval::eval_si(self, x).unwrap_or_else(|| self.intern(ExprNode::Si(x)))
    }

    /// Cosine integral `Ci(x)`, folding exact values (`Ci(∞) = 0`).
    pub fn ci(&mut self, x: ExprId) -> ExprId {
        crate::transforms::eval::eval_ci(self, x).unwrap_or_else(|| self.intern(ExprNode::Ci(x)))
    }

    /// Exponential integral `Ei(x)`, folding exact values (`Ei(−∞) = 0`,
    /// `Ei(∞) = ∞`, `Ei(0) = −∞`).
    pub fn ei(&mut self, x: ExprId) -> ExprId {
        crate::transforms::eval::eval_ei(self, x).unwrap_or_else(|| self.intern(ExprNode::Ei(x)))
    }

    /// Logarithmic integral `li(x)`, folding exact values (`li(0) = 0`,
    /// `li(1) = −∞`, `li(∞) = ∞`, `li(e^y) = Ei(y)`).
    pub fn li(&mut self, x: ExprId) -> ExprId {
        crate::transforms::eval::eval_li(self, x).unwrap_or_else(|| self.intern(ExprNode::Li(x)))
    }

    /// Riemann zeta function `ζ(s)`, folding exact values:
    /// `ζ(1) = z∞`, `ζ(0) = −1/2`, `ζ(−n) = −Bₙ₊₁/(n+1)`,
    /// `ζ(2k) = (−1)^{k+1} B₂ₖ (2π)^{2k} / (2 (2k)!)`, `ζ(∞) = 1`.
    pub fn zeta(&mut self, s: ExprId) -> ExprId {
        crate::transforms::eval::eval_zeta(self, s)
            .unwrap_or_else(|| self.intern(ExprNode::Zeta(s)))
    }

    /// Polygamma function `ψ⁽ⁿ⁾(x)`.
    ///
    /// `polygamma(0, x)` is canonicalised to [`Digamma`](ExprNode::Digamma);
    /// `ψ⁽ⁿ⁾(1)` and `ψ⁽ⁿ⁾(1/2)` fold to multiples of `ζ(n+1)`; and
    /// `ψ⁽ⁿ⁾(m)` / `ψ⁽ⁿ⁾(m + 1/2)` for small positive integers `m` are shifted
    /// back to those base points via the recurrence
    /// `ψ⁽ⁿ⁾(x+1) = ψ⁽ⁿ⁾(x) + (−1)ⁿ n!/xⁿ⁺¹`.
    pub fn polygamma(&mut self, n: ExprId, x: ExprId) -> ExprId {
        crate::transforms::eval::eval_polygamma(self, n, x)
            .unwrap_or_else(|| self.intern(ExprNode::Polygamma(n, x)))
    }

    /// Kronecker delta `δᵢⱼ`.
    ///
    /// Identical arguments give `1`; two distinct numbers give `0`;
    /// otherwise the node is stored with its arguments in canonical order.
    pub fn kronecker_delta(&mut self, i: ExprId, j: ExprId) -> ExprId {
        crate::transforms::eval::eval_kronecker_delta(self, i, j).unwrap_or_else(|| {
            let (a, b) = if self.sort_key(i) <= self.sort_key(j) {
                (i, j)
            } else {
                (j, i)
            };
            self.intern(ExprNode::KroneckerDelta(a, b))
        })
    }

    /// Creates a `Factorial` node: `n!`
    pub fn factorial(&mut self, expr: ExprId) -> ExprId {
        self.intern(ExprNode::Factorial(expr))
    }

    /// Creates a `Binomial` node: `C(n, k)` = n! / (k! * (n-k)!)
    pub fn binomial(&mut self, n: ExprId, k: ExprId) -> ExprId {
        self.intern(ExprNode::Binomial(n, k))
    }

    /// Creates a `Gt` (greater than) node: `lhs > rhs`.
    pub fn gt(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        self.intern(ExprNode::Gt(lhs, rhs))
    }

    /// Creates a `Ge` (greater than or equal) node: `lhs >= rhs`.
    pub fn ge(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        self.intern(ExprNode::Ge(lhs, rhs))
    }

    /// Creates an `Eq_` (mathematical equality test) node: `lhs == rhs`.
    pub fn eq_(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        self.intern(ExprNode::Eq_(lhs, rhs))
    }

    /// Creates a `Ne` (not equal) node: `lhs != rhs`.
    pub fn ne_(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        self.intern(ExprNode::Ne(lhs, rhs))
    }

    /// Creates an `And` (logical conjunction) node.
    ///
    /// Empty args → `BoolTrue`; single arg → that arg; otherwise n-ary `And`.
    pub fn and(&mut self, args: &[ExprId]) -> ExprId {
        if args.is_empty() {
            return self.bool_true;
        }
        if args.len() == 1 {
            return args[0];
        }
        self.intern(ExprNode::And(SmallVec::from_slice(args)))
    }

    /// Creates an `Or` (logical disjunction) node.
    ///
    /// Empty args → `BoolFalse`; single arg → that arg; otherwise n-ary `Or`.
    pub fn or(&mut self, args: &[ExprId]) -> ExprId {
        if args.is_empty() {
            return self.bool_false;
        }
        if args.len() == 1 {
            return args[0];
        }
        self.intern(ExprNode::Or(SmallVec::from_slice(args)))
    }

    /// Creates a `Not` (logical negation) node.
    ///
    /// Double negation eliminated: `Not(Not(x)) → x`.
    /// Constants folded: `Not(true) → false`, `Not(false) → true`.
    pub fn not(&mut self, expr: ExprId) -> ExprId {
        // Double negation: Not(Not(x)) → x
        if let ExprNode::Not(inner) = self.node(expr) {
            return *inner;
        }
        // Not(BoolTrue) → BoolFalse
        if expr == self.bool_true {
            return self.bool_false;
        }
        if expr == self.bool_false {
            return self.bool_true;
        }
        self.intern(ExprNode::Not(expr))
    }

    /// Creates a `Piecewise` node from `(value, condition)` pairs.
    pub fn piecewise(&mut self, pairs: &[(ExprId, ExprId)]) -> ExprId {
        let collected: SmallVec<[(ExprId, ExprId); 3]> = pairs.iter().copied().collect();
        self.intern(ExprNode::Piecewise(collected))
    }

    // ── Formal calculus nodes ───────────────────────────────────────────────

    /// Creates a formal `DefiniteIntegral` node: `∫_lo^hi body dvar`.
    ///
    /// This constructor never attempts integration — that is the job of
    /// [`definite::integrate_definite`](crate::calculus::definite::integrate_definite).
    /// It applies only the cheap, always-valid folds:
    ///
    /// - `lo == hi` (structurally) → `0`
    /// - `body == 0` → `0`
    /// - `body` free of `var` with finite bounds → `body · (hi − lo)`
    /// - numeric bounds with `lo > hi` → `−∫_hi^lo body dvar`
    ///
    /// A constant integrand over an infinite interval is left as a node:
    /// whether it diverges depends on the sign of the constant, which is
    /// decided by the definite integrator.
    pub fn definite_integral(
        &mut self,
        body: ExprId,
        var: ExprId,
        lo: ExprId,
        hi: ExprId,
    ) -> ExprId {
        if lo == hi || self.is_zero_structural(body) {
            return self.zero;
        }
        let infinite = |a: &Arena, id: ExprId| {
            matches!(
                a.node(id),
                ExprNode::Infinity | ExprNode::NegInfinity | ExprNode::ComplexInfinity
            )
        };
        if matches!(self.node(var), ExprNode::Symbol(_))
            && !infinite(self, lo)
            && !infinite(self, hi)
            && !crate::base::walk::contains(self, body, var)
        {
            let width = self.sub(hi, lo);
            return self.mul(&[body, width]);
        }
        if let (Some(rl), Some(rh)) = (self.as_num(lo), self.as_num(hi))
            && rl > rh
        {
            let flipped = self.intern(ExprNode::DefiniteIntegral(body, var, hi, lo));
            return self.neg(flipped);
        }
        self.intern(ExprNode::DefiniteIntegral(body, var, lo, hi))
    }

    // ── Combinatorial functions (library) ──────────────────────────

    /// Creates a `factorial2` (double factorial) node: `n!!`.
    pub fn factorial2(&mut self, n: ExprId) -> ExprId {
        self.lib_apply(LibFn::Factorial2, &[n])
    }

    /// Creates a `subfactorial` (derangement count) node: `!n`.
    pub fn subfactorial(&mut self, n: ExprId) -> ExprId {
        self.lib_apply(LibFn::Subfactorial, &[n])
    }

    /// Creates a `rising_factorial` (Pochhammer symbol) node: `(x)_n`.
    pub fn rising_factorial(&mut self, x: ExprId, n: ExprId) -> ExprId {
        self.lib_apply(LibFn::RisingFactorial, &[x, n])
    }

    /// Creates a `falling_factorial` node: `x^(n)`.
    pub fn falling_factorial(&mut self, x: ExprId, n: ExprId) -> ExprId {
        self.lib_apply(LibFn::FallingFactorial, &[x, n])
    }

    /// Creates a `fibonacci` node: `F(n)`.
    pub fn fibonacci(&mut self, n: ExprId) -> ExprId {
        self.lib_apply(LibFn::Fibonacci, &[n])
    }

    /// Creates a `lucas` node: `L(n)`.
    pub fn lucas(&mut self, n: ExprId) -> ExprId {
        self.lib_apply(LibFn::Lucas, &[n])
    }

    /// Creates a `bernoulli` (Bernoulli number) node: `B(n)`.
    pub fn bernoulli_number(&mut self, n: ExprId) -> ExprId {
        self.lib_apply(LibFn::Bernoulli, &[n])
    }

    /// Creates a `harmonic` (harmonic number) node: `H(n)`.
    pub fn harmonic(&mut self, n: ExprId) -> ExprId {
        self.lib_apply(LibFn::Harmonic, &[n])
    }

    /// Creates a `catalan` (Catalan number) node: `C(n)`.
    pub fn catalan_number(&mut self, n: ExprId) -> ExprId {
        self.lib_apply(LibFn::Catalan, &[n])
    }

    /// Creates a `bell` (Bell number) node: `B(n)`.
    pub fn bell(&mut self, n: ExprId) -> ExprId {
        self.lib_apply(LibFn::Bell, &[n])
    }

    /// Creates an `euler_number` node: `E(n)`.
    pub fn euler_number(&mut self, n: ExprId) -> ExprId {
        self.lib_apply(LibFn::EulerNumber, &[n])
    }

    // ── Combinatorial functions — Phase 1 (library) ────────────────

    /// Creates a `stirling1` (signed Stirling number of the first kind) node: `s(n, k)`.
    pub fn stirling1(&mut self, n: ExprId, k: ExprId) -> ExprId {
        self.lib_apply(LibFn::Stirling1, &[n, k])
    }

    /// Creates a `stirling2` (Stirling number of the second kind) node: `S(n, k)`.
    pub fn stirling2(&mut self, n: ExprId, k: ExprId) -> ExprId {
        self.lib_apply(LibFn::Stirling2, &[n, k])
    }

    /// Creates a `partition_count` node: `p(n)`.
    pub fn partition_count(&mut self, n: ExprId) -> ExprId {
        self.lib_apply(LibFn::PartitionCount, &[n])
    }

    // ── Special / distribution functions ───────────────────────────

    /// Creates a `heaviside` (Heaviside step function) node: 0 for x<0, 1/2 for x=0, 1 for x>0.
    pub fn heaviside(&mut self, arg: ExprId) -> ExprId {
        self.intern(ExprNode::Heaviside(arg))
    }

    /// Creates a `dirac_delta` (Dirac delta distribution) node: 0 for x≠0, symbolic at x=0.
    pub fn dirac_delta(&mut self, arg: ExprId) -> ExprId {
        self.intern(ExprNode::DiracDelta(arg))
    }

    /// Creates a `lambertw` (Lambert W function, principal branch) node: W(x)·exp(W(x)) = x.
    pub fn lambertw(&mut self, arg: ExprId) -> ExprId {
        self.intern(ExprNode::LambertW(arg))
    }

    /// Creates the branch `W_k(x)` of the Lambert W function (SymPy's
    /// `LambertW(x, k)`): `W_k(x)·exp(W_k(x)) = x`, `k` an integer.  The
    /// principal branch `k = 0` is the dedicated `ExprNode::LambertW(x)`
    /// node (so `lambertw_branch(x, 0) == lambertw(x)`); every other `k` is
    /// the library call `lambertw(x, k)`.
    pub fn lambertw_branch(&mut self, arg: ExprId, k: ExprId) -> ExprId {
        if self.as_num(k).is_some_and(|q| q.is_zero()) {
            return self.lambertw(arg);
        }
        self.lib_apply(LibFn::LambertW, &[arg, k])
    }

    // ── Bessel functions (library) ─────────────────────────────────

    /// Creates a `besselj` (Bessel function of the first kind) node: J_ν(x).
    pub fn besselj(&mut self, order: ExprId, arg: ExprId) -> ExprId {
        self.lib_apply(LibFn::BesselJ, &[order, arg])
    }

    /// Creates a `bessely` (Bessel function of the second kind) node: Y_ν(x).
    pub fn bessely(&mut self, order: ExprId, arg: ExprId) -> ExprId {
        self.lib_apply(LibFn::BesselY, &[order, arg])
    }

    /// Creates a `besseli` (modified Bessel function of the first kind) node: I_ν(x).
    pub fn besseli(&mut self, order: ExprId, arg: ExprId) -> ExprId {
        self.lib_apply(LibFn::BesselI, &[order, arg])
    }

    /// Creates a `besselk` (modified Bessel function of the second kind) node: K_ν(x).
    pub fn besselk(&mut self, order: ExprId, arg: ExprId) -> ExprId {
        self.lib_apply(LibFn::BesselK, &[order, arg])
    }

    // ── Orthogonal polynomials (library) ───────────────────────────

    /// Creates a `legendre` (Legendre polynomial) node: P_n(x).
    pub fn legendre(&mut self, n: ExprId, x: ExprId) -> ExprId {
        self.lib_apply(LibFn::Legendre, &[n, x])
    }

    /// Creates a `chebyshev_t` (Chebyshev polynomial of the first kind) node: T_n(x).
    pub fn chebyshev_t(&mut self, n: ExprId, x: ExprId) -> ExprId {
        self.lib_apply(LibFn::ChebyshevT, &[n, x])
    }

    /// Creates a `chebyshev_u` (Chebyshev polynomial of the second kind) node: U_n(x).
    pub fn chebyshev_u(&mut self, n: ExprId, x: ExprId) -> ExprId {
        self.lib_apply(LibFn::ChebyshevU, &[n, x])
    }

    /// Creates a `hermite` (physicist's Hermite polynomial) node: H_n(x).
    pub fn hermite(&mut self, n: ExprId, x: ExprId) -> ExprId {
        self.lib_apply(LibFn::Hermite, &[n, x])
    }

    /// Creates a `laguerre` (Laguerre polynomial) node: L_n(x).
    pub fn laguerre(&mut self, n: ExprId, x: ExprId) -> ExprId {
        self.lib_apply(LibFn::Laguerre, &[n, x])
    }

    /// Create a named physical constant with a known exact value.
    /// The constant displays as `name` but evaluates to `value`.
    pub fn physical_constant(&mut self, name: &str, value: ExprId) -> ExprId {
        let name_id = self.symbols.intern(name);
        self.intern(ExprNode::PhysicalConstant(name_id, value))
    }

    // ── Series extensions (residue, Fourier) ───────────────────────

    /// Compute the residue of `expr` at `var = point`.
    ///
    /// Delegates to [`residue::residue`](crate::calculus::residue::residue).
    pub fn residue_expr(
        &mut self,
        expr: ExprId,
        var: ExprId,
        point: ExprId,
    ) -> Result<ExprId, crate::base::errors::SymplexError> {
        crate::calculus::residue::residue(self, expr, var, point)
    }

    /// Compute the Fourier series of `expr` over [-π, π] with `n_terms` harmonics.
    ///
    /// Delegates to [`fourier::fourier_series`](crate::calculus::fourier::fourier_series).
    pub fn fourier_series_expr(&mut self, expr: ExprId, var: ExprId, n_terms: u32) -> ExprId {
        crate::calculus::fourier::fourier_series(self, expr, var, n_terms)
    }

    /// Compute the Laplace transform of `expr` with respect to time variable `t`,
    /// producing a function of frequency variable `s`.
    ///
    /// Delegates to [`laplace::laplace_transform`](crate::calculus::laplace::laplace_transform).
    pub fn laplace_transform_expr(
        &mut self,
        expr: ExprId,
        t: ExprId,
        s: ExprId,
    ) -> Result<ExprId, crate::base::errors::SymplexError> {
        crate::calculus::laplace::laplace_transform(self, expr, t, s)
    }

    /// Compute the inverse Laplace transform of `expr` (function of `s`)
    /// back to a function of time variable `t`.
    ///
    /// Delegates to [`laplace::inverse_laplace_transform`](crate::calculus::laplace::inverse_laplace_transform).
    pub fn inverse_laplace_transform_expr(
        &mut self,
        expr: ExprId,
        s: ExprId,
        t: ExprId,
    ) -> Result<ExprId, crate::base::errors::SymplexError> {
        crate::calculus::laplace::inverse_laplace_transform(self, expr, s, t)
    }

    /// Compute the Fourier transform of `expr` with respect to time variable `t`,
    /// producing a function of frequency variable `omega`.
    ///
    /// Delegates to [`fourier_transform::fourier_transform`](crate::calculus::fourier_transform::fourier_transform).
    pub fn fourier_transform_expr(
        &mut self,
        expr: ExprId,
        t: ExprId,
        omega: ExprId,
    ) -> Result<ExprId, crate::base::errors::SymplexError> {
        crate::calculus::fourier_transform::fourier_transform(self, expr, t, omega)
    }

    /// Compute the inverse Fourier transform of `expr` (function of `omega`)
    /// back to a function of time variable `t`.
    ///
    /// Delegates to [`fourier_transform::inverse_fourier_transform`](crate::calculus::fourier_transform::inverse_fourier_transform).
    pub fn inverse_fourier_transform_expr(
        &mut self,
        expr: ExprId,
        omega: ExprId,
        t: ExprId,
    ) -> Result<ExprId, crate::base::errors::SymplexError> {
        crate::calculus::fourier_transform::inverse_fourier_transform(self, expr, omega, t)
    }

    /// Compute the Laurent series of `expr` in `var` around `point` to the
    /// given `order`.
    ///
    /// Returns `Ok(series)` if the Laurent series could be computed,
    /// or `Err` if it could not.
    ///
    /// Delegates to [`series::laurent_series`](crate::calculus::series::laurent_series).
    pub fn laurent_series_expr(
        &mut self,
        expr: ExprId,
        var: ExprId,
        point: ExprId,
        order: u32,
    ) -> Result<ExprId, crate::base::errors::SymplexError> {
        crate::calculus::series::laurent_series(self, expr, var, point, order)
    }

    /// The sign of the real, symbol-free `expr` from a certified 16-digit
    /// value (its error ball excludes 0); `None` for anything else.  Used by
    /// canonical multiplication to orient `±∞` (`ln(2)·∞ = ∞`).
    ///
    /// `expr` must be real by the assumptions: a certified value whose
    /// imaginary part shows as 0 proves nothing, since an imaginary part
    /// below the relative precision of the whole value is not shown.  Up to
    /// 0.44 `−1 − i·(e^{10⁻³⁰} − 1 − 10⁻³⁰)` (imaginary part `−5·10⁻⁶¹`)
    /// had sign `−`, though no caller reached it with a non-real value
    /// (they checked realness first).  Callers with another realness proof
    /// use `eval::certified_real_sign`.
    pub(crate) fn sign_of_real_constant(&self, expr: ExprId) -> Option<std::cmp::Ordering> {
        use crate::base::assumptions::{AssumptionCache, Props};
        if !self.is_symbol_free(expr)
            || AssumptionCache::new().query(self, expr, Props::REAL) != Some(true)
        {
            return None;
        }
        crate::transforms::eval::certified_real_sign(self, expr)
    }

    /// The signs of the real and imaginary parts of the symbol-free `expr`
    /// from one certified 16-digit value (see
    /// `eval::certified_part_signs`): canonical multiplication orients
    /// `re(g)·∞` and `im(g)·∞` from one evaluation of `g`.  `None` when
    /// the value does not settle.
    pub(crate) fn part_signs(&self, expr: ExprId) -> Option<PartSigns> {
        crate::transforms::eval::certified_part_signs(self, expr)
    }

    /// Evaluate `expr` numerically to `digits` decimal digits of precision.
    ///
    /// Returns the string representation of the result, or an error if the
    /// expression contains free symbols.
    pub fn evalf_expr(
        &self,
        expr: ExprId,
        digits: u32,
    ) -> Result<String, crate::base::errors::SymplexError> {
        crate::transforms::evalf::evalf(self, expr, digits)
    }

    /// Generate a Rust function as a string from an expression.
    ///
    /// The generated function takes `f64` arguments and returns `f64`.
    /// Uses CSE (common subexpression elimination) for efficient code.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn to_rust_fn(
        &mut self,
        expr: ExprId,
        name: &str,
        args: &[&str],
    ) -> Result<String, crate::base::errors::SymplexError> {
        crate::output::codegen::to_rust_fn(self, expr, name, args)
    }

    // ── Set constructors ───────────────────────────────────────────

    /// Build a canonical interval between `start` and `end`, with the
    /// endpoints `kind` includes.
    ///
    /// Delegates to [`canon::canon_interval`](crate::base::canon::canon_interval) for degenerate-case handling.
    pub(crate) fn interval(&mut self, start: ExprId, end: ExprId, kind: IntervalKind) -> ExprId {
        crate::base::canon::canon_interval(self, start, end, kind)
    }

    /// Build a canonical finite set `{elems[0], elems[1], …}`.
    ///
    /// Sorts, deduplicates, and handles the empty case.
    pub(crate) fn finite_set(&mut self, elems: &[ExprId]) -> ExprId {
        crate::base::canon::canon_finite_set(self, elems)
    }

    /// Build a canonical set union `A ∪ B ∪ …`.
    ///
    /// Flattens nested unions, removes EmptySet, short-circuits on UniversalSet.
    pub(crate) fn set_union(&mut self, sets: &[ExprId]) -> ExprId {
        crate::base::canon::canon_set_union(self, sets)
    }

    /// Build a canonical set intersection `A ∩ B ∩ …`.
    ///
    /// Flattens nested intersections, removes UniversalSet, short-circuits on EmptySet.
    pub(crate) fn set_intersection(&mut self, sets: &[ExprId]) -> ExprId {
        crate::base::canon::canon_set_intersection(self, sets)
    }

    /// Build a set complement (relative): `set \ universe`.
    pub(crate) fn set_complement(&mut self, set: ExprId, universe: ExprId) -> ExprId {
        self.intern(ExprNode::SetComplement(set, universe))
    }

    // ── Inequality solving ─────────────────────────────────────────

    /// Solve `expr rel 0` for `var`, returning the solution as a set `ExprId`.
    ///
    /// Delegates to [`inequalities::solve_inequality`](crate::transforms::inequalities::solve_inequality).
    pub(crate) fn solve_inequality_expr(
        &mut self,
        expr: ExprId,
        var: ExprId,
        rel: crate::transforms::inequalities::Relation,
    ) -> Result<ExprId, crate::base::errors::SymplexError> {
        crate::transforms::inequalities::solve_inequality(self, expr, var, rel)
    }

    /// Solve `expr = 0`, returning solutions as a `FiniteSet` `ExprId`.
    ///
    /// Delegates to [`inequalities::solveset`](crate::transforms::inequalities::solveset).
    pub(crate) fn solveset_expr(&mut self, expr: ExprId, var: ExprId) -> ExprId {
        crate::transforms::inequalities::solveset(self, expr, var)
    }

    // ── Closed-form sum evaluation ─────────────────────────────────

    /// Attempt closed-form evaluation of `Σ_{var=lower}^{upper} body`.
    ///
    /// Delegates to [`sum_eval::eval_sum_symbolic`](crate::calculus::sum_eval::eval_sum_symbolic). Returns `Some(result)`
    /// if a closed form was found, `None` otherwise.
    pub(crate) fn eval_sum_symbolic_expr(
        &mut self,
        body: ExprId,
        var: ExprId,
        lower: ExprId,
        upper: ExprId,
    ) -> Option<ExprId> {
        crate::calculus::sum_eval::eval_sum_symbolic(self, body, var, lower, upper)
    }

    // ── Convergence testing ────────────────────────────────────────

    /// Test whether the infinite series `Σ_{k=1}^{∞} body(var)` converges.
    ///
    /// Returns `Some(true)` if convergent, `Some(false)` if divergent,
    /// `None` if the test is inconclusive.
    ///
    /// Delegates to [`convergence::is_convergent`](crate::calculus::convergence::is_convergent).
    pub(crate) fn is_convergent_expr(&mut self, body: ExprId, var: ExprId) -> Option<bool> {
        crate::calculus::convergence::is_convergent(self, body, var)
    }
}

// ---------------------------------------------------------------------------
// Default
// ---------------------------------------------------------------------------

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Debug
// ---------------------------------------------------------------------------

impl fmt::Debug for Arena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Arena")
            .field("node_count", &self.nodes.len())
            .field("num_count", &self.numbers.len())
            .field("symbol_count", &self.symbols.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    #[test]
    fn pre_interned_constants_are_distinct() {
        let a = Arena::new();
        let ids = [
            a.zero,
            a.one,
            a.neg_one,
            a.pi,
            a.e_const,
            a.i_unit,
            a.euler_gamma,
            a.catalan,
            a.golden_ratio,
            a.infinity,
            a.neg_infinity,
            a.nan,
        ];
        for (i, &id_a) in ids.iter().enumerate() {
            for (j, &id_b) in ids.iter().enumerate() {
                if i != j {
                    assert_ne!(id_a, id_b, "constants at index {i} and {j} should differ");
                }
            }
        }
    }

    #[test]
    fn intern_deduplicates_atoms() {
        let mut a = Arena::new();
        let x1 = a.symbol("x");
        let x2 = a.symbol("x");
        assert_eq!(x1, x2);
    }

    #[test]
    fn named_constants_are_pre_interned() {
        let mut a = Arena::new();
        assert_eq!(a.intern(ExprNode::EulerGamma), a.euler_gamma());
        assert_eq!(a.intern(ExprNode::Catalan), a.catalan());
        assert_eq!(a.intern(ExprNode::GoldenRatio), a.golden_ratio());
        assert!(matches!(a.node(a.euler_gamma()), ExprNode::EulerGamma));
        assert!(matches!(a.node(a.catalan()), ExprNode::Catalan));
        assert!(matches!(a.node(a.golden_ratio()), ExprNode::GoldenRatio));
    }

    #[test]
    fn intern_deduplicates_compound_nodes() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let children: SmallVec<[ExprId; 6]> = smallvec![x, y];
        let add1 = a.intern(ExprNode::Add(children.clone()));
        let add2 = a.intern(ExprNode::Add(children));
        assert_eq!(add1, add2);
    }

    #[test]
    fn different_nodes_get_different_ids() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        assert_ne!(x, y);
    }

    #[test]
    fn int_creates_correct_value() {
        let mut a = Arena::new();
        let five = a.int(5);
        match a.node(five) {
            ExprNode::Num(num_id) => {
                assert_eq!(*a.num(*num_id), Ratio::from_integer(BigInt::from(5)));
            }
            other => panic!("expected Num, got {other:?}"),
        }
    }

    #[test]
    fn int_zero_returns_pre_interned() {
        let mut a = Arena::new();
        let z = a.int(0);
        assert_eq!(z, a.zero);
    }

    #[test]
    fn int_one_returns_pre_interned() {
        let mut a = Arena::new();
        let o = a.int(1);
        assert_eq!(o, a.one);
    }

    #[test]
    fn int_neg_one_returns_pre_interned() {
        let mut a = Arena::new();
        let m1 = a.int(-1);
        assert_eq!(m1, a.neg_one);
    }

    #[test]
    fn rational_reduces() {
        let mut a = Arena::new();
        let half_a = a.rational(1, 2);
        let half_b = a.rational(2, 4);
        assert_eq!(half_a, half_b, "2/4 should reduce to 1/2");
    }

    #[test]
    fn big_int_works() {
        let mut a = Arena::new();
        let big = a.big_int(BigInt::from(999_999_999_999i64));
        match a.node(big) {
            ExprNode::Num(num_id) => {
                let val = a.num(*num_id);
                assert_eq!(*val, Ratio::from_integer(BigInt::from(999_999_999_999i64)));
            }
            other => panic!("expected Num, got {other:?}"),
        }
    }

    #[test]
    fn pow_creates_correct_node() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let p = a.pow(x, two);
        assert_eq!(*a.node(p), ExprNode::Pow(x, two));
    }

    #[test]
    fn neg_creates_correct_node() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let nx = a.neg(x);
        // canon_neg normalizes Neg(x) to Mul(-1, x).
        if let ExprNode::Mul(args) = a.node(nx) {
            assert_eq!(args.len(), 2);
            assert_eq!(args[0], a.neg_one, "first factor should be -1");
            assert_eq!(args[1], x, "second factor should be x");
        } else {
            panic!("expected Mul node, got {:?}", a.node(nx));
        }
    }

    #[test]
    fn trig_and_transcendental_functions() {
        let mut a = Arena::new();
        let x = a.symbol("x");

        let s = a.sin(x);
        assert_eq!(*a.node(s), ExprNode::Sin(x));

        let c = a.cos(x);
        assert_eq!(*a.node(c), ExprNode::Cos(x));

        let t = a.tan(x);
        assert_eq!(*a.node(t), ExprNode::Tan(x));

        let e = a.exp(x);
        assert_eq!(*a.node(e), ExprNode::Exp(x));

        let l = a.ln(x);
        assert_eq!(*a.node(l), ExprNode::Ln(x));

        let sq = a.sqrt(x);
        // sqrt now produces Pow(x, 1/2)
        let half = a.rational(1, 2);
        assert_eq!(*a.node(sq), ExprNode::Pow(x, half));

        let ab = a.abs(x);
        assert_eq!(*a.node(ab), ExprNode::Abs(x));
    }

    #[test]
    fn node_count_grows() {
        let mut a = Arena::new();
        let before = a.node_count();
        let _x = a.symbol("x");
        assert_eq!(a.node_count(), before + 1);
        // Interning again should NOT grow the count.
        let _x2 = a.symbol("x");
        assert_eq!(a.node_count(), before + 1);
    }

    #[test]
    fn symbol_name_roundtrip() {
        let mut a = Arena::new();
        let x = a.symbol("alpha");
        match a.node(x) {
            ExprNode::Symbol(sym_id) => {
                assert_eq!(a.symbol_name(*sym_id), "alpha");
            }
            other => panic!("expected Symbol, got {other:?}"),
        }
    }

    #[test]
    fn children_delegates_correctly() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let sum = a.intern(ExprNode::Add(smallvec![x, y]));
        let kids = a.children(sum);
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0], x);
        assert_eq!(kids[1], y);
    }

    #[test]
    fn sort_keys_are_parallel_to_nodes() {
        let mut a = Arena::new();
        let _x = a.symbol("x");
        let _y = a.symbol("y");
        assert_eq!(a.nodes.len() + 1, a.key_starts.len());
        assert_eq!(a.key_starts.last().copied(), Some(a.key_bytes.len()));
        assert_eq!(a.nodes.len(), a.facts.len());
    }

    /// A number's key bytes are its `Display` form, as in 0.48
    /// (`Ratio::to_string`), including at the `i64` boundaries where the
    /// machine-word fast path hands over to the big integers, and for an
    /// unreduced representation.
    #[test]
    fn decimal_key_bytes_are_display_bytes() {
        let big = |s: &str| s.parse::<BigInt>().unwrap();
        let mut values = Vec::new();
        for n in [
            "0",
            "1",
            "-1",
            "9223372036854775807",
            "9223372036854775808",
            "-9223372036854775808",
            "-9223372036854775809",
            "123456789012345678901234567890",
        ] {
            for d in ["1", "2", "9223372036854775807", "9223372036854775808"] {
                values.push(Ratio::new(big(n), big(d)));
            }
        }
        values.push(Ratio::new_raw(big("3"), big("-4")));
        values.push(Ratio::new_raw(big("-6"), big("-1")));
        values.push(Ratio::new_raw(big("5"), big("1")));
        for q in values {
            let mut out = vec![7u8];
            write_decimal(&q, &mut out);
            assert_eq!(&out[1..], q.to_string().as_bytes(), "{q}");
        }
    }

    #[test]
    fn sort_key_ordering_numbers_before_symbols() {
        let mut a = Arena::new();
        let five = a.int(5);
        let x = a.symbol("x");
        assert!(
            a.sort_key(five) < a.sort_key(x),
            "numbers should sort before symbols"
        );
    }

    #[test]
    fn intern_num_deduplicates() {
        let mut a = Arena::new();
        let n1 = a.intern_num(Ratio::from_integer(BigInt::from(42)));
        let n2 = a.intern_num(Ratio::from_integer(BigInt::from(42)));
        assert_eq!(n1, n2);
    }

    /// Equal values hash alike in any representation, also when a hashing
    /// prime divides the denominator, and huge fractions hash in linear
    /// time (the continued-fraction hash of 0.35 was quadratic).
    #[test]
    fn hash_num_is_representation_invariant() {
        let p1 = BigInt::from((1u64 << 61) - 1);
        let p2 = BigInt::from((1u64 << 62) - 57);
        let big: BigInt = BigInt::from(3).pow(4000u32) + 1;
        let raw = |n: BigInt, d: BigInt| Ratio::new_raw(n, d);
        let cases: Vec<(BigInt, BigInt)> = vec![
            (BigInt::from(0), BigInt::from(1)),
            (BigInt::from(-7), BigInt::from(3)),
            (BigInt::from(5), p1.clone()),
            (BigInt::from(-5), &p1 * &p2),
            (big.clone(), BigInt::from(7).pow(3000u32)),
            (-big.clone(), &p1 * BigInt::from(11)),
        ];
        for (n, d) in cases {
            let reduced = Ratio::new(n.clone(), d.clone());
            let h = Arena::hash_num(&reduced);
            for k in [BigInt::from(2), p1.clone(), &p1 * &p1, &p1 * &p2 * 3] {
                let unreduced = raw(&n * &k, &d * &k);
                assert!(Arena::hash_num(&unreduced) == h, "× {k}");
                // (`Ratio`'s `==` on an unreduced huge value recurses along
                // the continued fraction: small values only.)
                if n.bits() < 256 {
                    let mut a = Arena::new();
                    let i = a.intern_num(reduced.clone());
                    assert_eq!(a.intern_num(unreduced), i);
                }
            }
        }
        assert_ne!(
            Arena::hash_num(&Ratio::new(BigInt::from(1), BigInt::from(3))),
            Arena::hash_num(&Ratio::new(BigInt::from(2), BigInt::from(3)))
        );
        // Two coprime 6,000-digit terms: milliseconds.
        let start = std::time::Instant::now();
        let q = Ratio::new(
            BigInt::from(7).pow(7000u32) + 2,
            BigInt::from(5).pow(8500u32) + 3,
        );
        let _ = Arena::hash_num(&q);
        assert!(start.elapsed().as_secs() < 2, "took {:?}", start.elapsed());
    }

    #[test]
    fn with_config_uses_custom_config() {
        let cfg = EvalConfig {
            max_pow_exponent: 42,
            ..EvalConfig::default()
        };
        let a = Arena::with_config(cfg);
        assert_eq!(a.config.max_pow_exponent, 42);
    }

    #[test]
    fn default_trait_works() {
        let a = Arena::default();
        // Should have pre-interned at least the 9 constants.
        assert!(a.node_count() >= 9);
    }

    #[test]
    fn add_zero_args_returns_zero() {
        let mut a = Arena::new();
        let result = a.add(&[]);
        assert_eq!(result, a.zero);
    }

    #[test]
    fn add_one_arg_returns_arg() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = a.add(&[x]);
        assert_eq!(result, x);
    }

    #[test]
    fn mul_zero_args_returns_one() {
        let mut a = Arena::new();
        let result = a.mul(&[]);
        assert_eq!(result, a.one);
    }

    #[test]
    fn mul_one_arg_returns_arg() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = a.mul(&[x]);
        assert_eq!(result, x);
    }
}
