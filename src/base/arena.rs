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

use crate::base::config::EvalConfig;
use crate::base::libfn::LibFn;
use crate::base::node::{ExprId, ExprNode, NumId, SymbolId};
use crate::base::numeric::Q;
use crate::base::sort_key::{SortKey, compute_sort_key};
use crate::base::symbol::SymbolTable;

// ── Library function names ─────────────────────────────────────────────
// The registry is `base::libfn::LibFn`; consumers match on the enum and
// build nodes with `Arena::lib_apply`.  These aliases remain for the two
// callers outside the registry's consumers (`transforms::integrate`,
// `calculus::laplace`), which compare or intern the name text directly.
pub(crate) const FN_BESSELJ: &str = LibFn::BesselJ.name();
pub(crate) const FN_ERFI: &str = LibFn::Erfi.name();
pub(crate) const FN_SHI: &str = LibFn::Shi.name();
pub(crate) const FN_CHI: &str = LibFn::Chi.name();

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

    /// Sort keys parallel to `nodes` — `sort_keys[i]` is the key for
    /// `nodes[i]`.
    sort_keys: Vec<SortKey>,

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
}

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
            sort_keys: Vec::new(),
            dedup: FxHashMap::default(),
            num_dedup: FxHashMap::default(),
            symbols: SymbolTable::new(),
            config,
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
        &mut self.config
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

    /// Computes a `u64` hash for a [`Ratio<BigInt>`] using the `FxHasher`.
    fn hash_num(value: &Q) -> u64 {
        // `Ratio`'s own `Hash` walks the continued fraction of numer/denom
        // *recursively* — one stack frame per partial quotient — so that
        // unreduced equal values hash alike.  A rational with a very long
        // expansion (a ratio of consecutive huge Fibonacci numbers has one
        // partial quotient per term) would need that many frames.  Same
        // sequence of partial quotients, in a loop.
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
        hasher.finish()
    }

    /// Interns an [`ExprNode`], returning its canonical [`ExprId`].
    ///
    /// If a structurally identical node has already been interned, the
    /// existing [`ExprId`] is returned.  Otherwise the node is appended to
    /// the arena, a sort key is computed, and the new [`ExprId`] is recorded
    /// in the dedup map.
    pub(crate) fn intern(&mut self, node: ExprNode) -> ExprId {
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
        let sort_key = self.compute_sort_key_for(&node);
        self.nodes.push(node);
        self.sort_keys.push(sort_key);

        self.dedup.entry(hash).or_default().push(id);

        id
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

    /// Computes the [`SortKey`] for a node by delegating to
    /// [`sort_key::compute_sort_key`](crate::base::sort_key::compute_sort_key) with closures that resolve through
    /// `self`.
    fn compute_sort_key_for(&self, node: &ExprNode) -> SortKey {
        compute_sort_key(
            node,
            |child_id| self.sort_keys[child_id.0 as usize].clone(),
            |num_id| {
                // Serialize the rational value as a string and return its
                // bytes.  This is simple and gives a consistent (if not
                // optimally compact) ordering.
                self.numbers[num_id.0 as usize].to_string().into_bytes()
            },
            |sym_id| self.symbols.name(sym_id).to_owned(),
        )
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

    /// Returns a reference to the [`SortKey`] for the node identified by
    /// `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` was not produced by this arena.
    pub fn sort_key(&self, id: ExprId) -> &SortKey {
        &self.sort_keys[id.0 as usize]
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

    /// Returns the interned name of a symbol.
    ///
    /// # Panics
    ///
    /// Panics if `id` was not produced by this arena's symbol table.
    pub fn symbol_name(&self, id: SymbolId) -> &str {
        self.symbols.name(id)
    }

    /// The library function an `Apply` head names, if it is one.
    ///
    /// Resolved from the interned text, so the registry never pre-interns a
    /// symbol and `SymbolId` numbering is exactly what the user's own
    /// interning order produces.
    pub(crate) fn lib_fn(&self, sym: SymbolId) -> Option<LibFn> {
        LibFn::from_name(self.symbols.name(sym))
    }

    /// Interns `f(args)` as a library `Apply` node without folding.
    pub(crate) fn lib_apply(&mut self, f: LibFn, args: &[ExprId]) -> ExprId {
        let sid = self.symbols.intern(f.name());
        self.intern(ExprNode::Apply(sid, args.iter().copied().collect()))
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
    /// Returns a vector of solutions. Delegates to [`solve::solve`](crate::transforms::solve::solve).
    pub fn solve_for(
        &mut self,
        expr: ExprId,
        var: ExprId,
    ) -> Vec<crate::transforms::solve::Solution> {
        crate::transforms::solve::solve(self, expr, var)
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

    /// Compute the Taylor series of `expr` in `var` around `point`
    /// to the given `order` (number of terms).
    ///
    /// Returns `Err` if the series cannot be computed (e.g., pole at the
    /// expansion point).
    /// Delegates to [`series::series`](crate::calculus::series::series).
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
                // Taylor failed (likely a pole). Try Laurent series as fallback.
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

    // ── Combinatorial functions (Apply-based) ──────────────────────

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

    // ── Combinatorial functions — Phase 1 (Apply-based) ────────────

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

    // ── Special / distribution functions (Apply-based) ─────────────

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

    // ── Bessel functions (Apply-based) ─────────────────────────────

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

    // ── Orthogonal polynomials (Apply-based) ───────────────────────

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

    /// Build a canonical interval `[start, end]` (or open variants based on `flags`).
    ///
    /// Delegates to [`canon::canon_interval`](crate::base::canon::canon_interval) for degenerate-case handling.
    pub(crate) fn interval(&mut self, start: ExprId, end: ExprId, flags: u8) -> ExprId {
        crate::base::canon::canon_interval(self, start, end, flags)
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
        assert_eq!(a.nodes.len(), a.sort_keys.len());
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
