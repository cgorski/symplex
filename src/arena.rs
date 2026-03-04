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

use crate::config::EvalConfig;
use crate::node::{ExprId, ExprNode, NumId, SymbolId};
use crate::sort_key::{SortKey, compute_sort_key};
use crate::symbol::SymbolTable;

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
/// A set of commonly used constants (0, 1, −1, π, e, i, ∞, −∞, NaN) are
/// pre-interned at construction time and exposed as public fields for
/// convenience.
pub struct Arena {
    // -- storage -------------------------------------------------------------
    /// Expression nodes indexed by [`ExprId`].
    nodes: Vec<ExprNode>,

    /// Numeric literal side-table indexed by [`NumId`].
    numbers: Vec<Ratio<BigInt>>,

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
    pub config: EvalConfig,

    // -- pre-interned constants (ExprId) ------------------------------------
    /// The integer zero (0).
    pub zero: ExprId,

    /// The integer one (1).
    pub one: ExprId,

    /// The integer negative one (−1).
    pub neg_one: ExprId,

    /// The mathematical constant π.
    pub pi: ExprId,

    /// Euler's number *e*.
    pub e_const: ExprId,

    /// The imaginary unit *i*.
    pub i_unit: ExprId,

    /// Positive infinity (+∞).
    pub infinity: ExprId,

    /// Negative infinity (−∞).
    pub neg_infinity: ExprId,

    /// Not-a-number (undefined / indeterminate).
    pub nan: ExprId,

    /// Complex infinity (z∞ — undirected infinity in the complex plane).
    pub complex_infinity: ExprId,

    // -- pre-interned constants (NumId) -------------------------------------
    /// [`NumId`] for the rational value 0.
    pub zero_num: NumId,

    /// [`NumId`] for the rational value 1.
    pub one_num: NumId,

    /// [`NumId`] for the rational value −1.
    pub neg_one_num: NumId,
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
            infinity: ExprId(0),
            neg_infinity: ExprId(0),
            nan: ExprId(0),
            complex_infinity: ExprId(0),
            zero_num: NumId(0),
            one_num: NumId(0),
            neg_one_num: NumId(0),
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
        arena.infinity = arena.intern(ExprNode::Infinity);
        arena.neg_infinity = arena.intern(ExprNode::NegInfinity);
        arena.nan = arena.intern(ExprNode::NaN);
        arena.complex_infinity = arena.intern(ExprNode::ComplexInfinity);

        arena
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
    fn hash_num(value: &Ratio<BigInt>) -> u64 {
        let mut hasher = rustc_hash::FxHasher::default();
        value.hash(&mut hasher);
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
        let id = ExprId(self.nodes.len() as u32);
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
    pub(crate) fn intern_num(&mut self, value: Ratio<BigInt>) -> NumId {
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
        let id = NumId(self.numbers.len() as u32);
        self.numbers.push(value);
        self.num_dedup.entry(hash).or_default().push(id);
        id
    }

    /// Computes the [`SortKey`] for a node by delegating to
    /// [`sort_key::compute_sort_key`] with closures that resolve through
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
    pub fn num(&self, id: NumId) -> &Ratio<BigInt> {
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

    /// Get the stored assumptions for a symbol.
    pub(crate) fn symbol_assumptions(&self, id: SymbolId) -> crate::assumptions::Assumptions {
        self.symbols.get_assumptions(id)
    }

    /// Set assumptions for a symbol.
    pub(crate) fn set_symbol_assumptions(
        &mut self,
        id: SymbolId,
        a: crate::assumptions::Assumptions,
    ) {
        self.symbols.set_assumptions(id, a);
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
    pub(crate) fn as_coeff_term(&mut self, id: ExprId) -> (Ratio<BigInt>, ExprId) {
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
    /// Uses [`canon_mul`](crate::canon::canon_mul) to ensure the result
    /// is properly flattened (no nested Mul nodes).
    pub(crate) fn make_coeff_term(&mut self, coeff: Ratio<BigInt>, term: ExprId) -> ExprId {
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
        crate::canon::canon_mul(self, &[coeff_id, term])
    }

    /// Check if an expression is a numeric literal and return its value.
    pub(crate) fn as_num(&self, id: ExprId) -> Option<&Ratio<BigInt>> {
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

    /// Creates (or retrieves) a rational expression `p/q`.
    ///
    /// The ratio is automatically reduced to lowest terms by `num_rational`.
    pub fn rational(&mut self, p: i64, q: i64) -> ExprId {
        let num_id = self.intern_num(Ratio::new(BigInt::from(p), BigInt::from(q)));
        self.intern(ExprNode::Num(num_id))
    }

    /// Creates (or retrieves) a symbolic variable or constant by name.
    pub fn symbol(&mut self, name: &str) -> ExprId {
        let sym_id = self.symbols.intern(name);
        self.intern(ExprNode::Symbol(sym_id))
    }

    /// Creates an `Add` node with full canonicalization.
    ///
    /// Flattens nested Adds, combines like terms, sorts by canonical
    /// order, drops zero-coefficient terms, propagates NaN.
    pub fn add(&mut self, args: &[ExprId]) -> ExprId {
        // Canonical implementation in canon.rs
        crate::canon::canon_add(self, args)
    }

    /// Creates a `Mul` node with full canonicalization.
    pub fn mul(&mut self, args: &[ExprId]) -> ExprId {
        crate::canon::canon_mul(self, args)
    }

    /// Creates a `Pow` node with canonicalization.
    pub fn pow(&mut self, base: ExprId, exp: ExprId) -> ExprId {
        crate::canon::canon_pow(self, base, exp)
    }

    /// Creates a `Neg` node with canonicalization.
    pub fn neg(&mut self, expr: ExprId) -> ExprId {
        crate::canon::canon_neg(self, expr)
    }

    /// Creates a division expression `a / b` as `a * b^(-1)`.
    pub fn div(&mut self, a: ExprId, b: ExprId) -> ExprId {
        let neg_one = self.neg_one;
        let b_inv = self.pow(b, neg_one);
        self.mul(&[a, b_inv])
    }

    /// Structural substitution: replace `old` with `new` in `expr`.
    ///
    /// Delegates to [`subs::subs`].
    pub fn subs_structural(&mut self, expr: ExprId, old: ExprId, new: ExprId) -> ExprId {
        crate::subs::subs(self, expr, old, new)
    }

    /// Simultaneous structural substitution of multiple pairs.
    ///
    /// Delegates to [`subs::subs_map`].
    pub fn subs_map_structural(
        &mut self,
        expr: ExprId,
        replacements: &[(ExprId, ExprId)],
    ) -> ExprId {
        crate::subs::subs_map(self, expr, replacements)
    }

    /// Differentiate `expr` with respect to `var`.
    ///
    /// Delegates to [`diff::diff`].
    pub fn diff_wrt(&mut self, expr: ExprId, var: ExprId) -> ExprId {
        crate::diff::diff(self, expr, var)
    }

    /// Algebraic expansion: distribute products over sums, expand
    /// integer powers of sums.
    ///
    /// Delegates to [`expand::expand`].
    pub fn expand_expr(&mut self, expr: ExprId) -> ExprId {
        crate::expand::expand(self, expr)
    }

    /// Exact evaluation of known special values (sin(0)→0, cos(π)→-1, etc.).
    ///
    /// Delegates to [`eval::eval`].
    pub fn eval_expr(&mut self, expr: ExprId) -> ExprId {
        crate::eval::eval(self, expr)
    }

    /// Solve `expr = 0` for `var`.
    ///
    /// Returns a vector of solutions. Delegates to [`solve::solve`].
    pub fn solve_for(&mut self, expr: ExprId, var: ExprId) -> Vec<crate::solve::Solution> {
        crate::solve::solve(self, expr, var)
    }

    /// Cancel common polynomial factors in a rational expression.
    ///
    /// `var` is the symbol to treat as the polynomial variable.
    /// Delegates to [`polybridge::cancel`].
    pub fn cancel_expr(&mut self, expr: ExprId, var: ExprId) -> ExprId {
        crate::polybridge::cancel(self, expr, var)
    }

    /// Group an expression by powers of `var`.
    ///
    /// Converts to a univariate polynomial in `var` and rebuilds,
    /// naturally grouping coefficients by power.
    /// Returns unchanged if not polynomial in `var`.
    /// Delegates to [`polybridge::collect`].
    pub fn collect_expr(&mut self, expr: ExprId, var: ExprId) -> ExprId {
        crate::polybridge::collect(self, expr, var)
    }

    /// Combine fractions in an Add over a common denominator.
    ///
    /// Delegates to [`polybridge::together`].
    pub fn together_expr(&mut self, expr: ExprId) -> ExprId {
        crate::polybridge::together(self, expr)
    }

    /// Compute the indefinite integral of `expr` with respect to `var`.
    ///
    /// Returns an unevaluated `Integral` node for integrands that
    /// don't match any known rule.
    /// Delegates to [`integrate::integrate`].
    pub fn integrate_expr(&mut self, expr: ExprId, var: ExprId) -> ExprId {
        crate::integrate::integrate(self, expr, var)
    }

    /// Compute the Taylor series of `expr` in `var` around `point`
    /// to the given `order` (number of terms).
    ///
    /// Returns `Err` if the series cannot be computed (e.g., pole at the
    /// expansion point).
    /// Delegates to [`series::series`].
    pub fn series_expr(
        &mut self,
        expr: ExprId,
        var: ExprId,
        point: ExprId,
        order: u32,
    ) -> Result<ExprId, crate::errors::SymplexError> {
        crate::series::series(self, expr, var, point, order)
    }

    /// Factor a polynomial expression into a product of linear factors.
    ///
    /// Returns the expression unchanged if no rational roots are found.
    /// Delegates to [`factor::factor`].
    pub fn factor_expr(&mut self, expr: ExprId, var: ExprId) -> ExprId {
        crate::factor::factor(self, expr, var)
    }

    /// Return the degree of `expr` as a polynomial in `var`.
    ///
    /// Returns `None` if not polynomial or if the zero polynomial.
    /// Delegates to [`polybridge::poly_degree`].
    pub fn degree_of(&self, expr: ExprId, var: ExprId) -> Option<usize> {
        crate::polybridge::poly_degree(self, expr, var)
    }

    /// Return the coefficients of `expr` as a polynomial in `var`,
    /// in ascending degree order.
    ///
    /// Returns `None` if not polynomial in `var`.
    /// Delegates to [`polybridge::poly_coefficients`].
    pub fn coefficients_of(&mut self, expr: ExprId, var: ExprId) -> Option<Vec<ExprId>> {
        crate::polybridge::poly_coefficients(self, expr, var)
    }

    /// Compute the limit of `expr` as `var` approaches `point`.
    ///
    /// Returns `Err` if the limit cannot be determined.
    /// Delegates to [`limit::limit`].
    pub fn limit_expr(
        &mut self,
        expr: ExprId,
        var: ExprId,
        point: ExprId,
    ) -> Result<ExprId, crate::errors::SymplexError> {
        crate::limit::limit(self, expr, var, point)
    }

    /// Decompose `expr` into (numerator, denominator).
    ///
    /// Delegates to [`polybridge::as_numer_denom`].
    pub fn as_numer_denom_expr(&mut self, expr: ExprId) -> (ExprId, ExprId) {
        crate::polybridge::as_numer_denom(self, expr)
    }

    /// Partial fraction decomposition of `expr` with respect to `var`.
    ///
    /// Delegates to [`apart::apart`].
    pub fn apart_expr(&mut self, expr: ExprId, var: ExprId) -> ExprId {
        crate::apart::apart(self, expr, var)
    }

    /// Expand trigonometric functions with composite arguments.
    ///
    /// `sin(a+b) → sin(a)cos(b) + cos(a)sin(b)`, etc.
    /// Delegates to [`trig_expand::expand_trig`].
    pub fn expand_trig_expr(&mut self, expr: ExprId) -> ExprId {
        crate::trig_expand::expand_trig(self, expr)
    }

    /// Expand logarithmic expressions.
    ///
    /// `ln(a*b) → ln(a)+ln(b)`, `ln(a^n) → n*ln(a)`, etc.
    /// Delegates to [`log_expand::expand_log`].
    pub fn expand_log_expr(&mut self, expr: ExprId) -> ExprId {
        crate::log_expand::expand_log(self, expr)
    }

    /// Combine logarithmic terms: `ln(a)+ln(b) → ln(a*b)`, `n*ln(a) → ln(a^n)`.
    /// Delegates to [`log_combine::log_combine`].
    pub fn log_combine_expr(&mut self, expr: ExprId) -> ExprId {
        crate::log_combine::log_combine(self, expr)
    }

    /// Apply trig product-to-sum and double-angle identities.
    /// `sin(a)*cos(b) → ½[sin(a+b)+sin(a-b)]`, etc.
    /// Delegates to [`trig_combine::trig_combine`].
    pub fn trig_combine_expr(&mut self, expr: ExprId) -> ExprId {
        crate::trig_combine::trig_combine(self, expr)
    }

    /// Try multiple simplification strategies and return the simplest result.
    /// Delegates to [`simplify_engine::smart_simplify`].
    pub fn smart_simplify_expr(&mut self, expr: ExprId) -> ExprId {
        crate::simplify_engine::smart_simplify(self, expr)
    }

    /// Count the number of operations (non-atom nodes) in an expression.
    /// Delegates to [`simplify_engine::count_ops`].
    pub fn count_ops(&self, expr: ExprId) -> usize {
        crate::simplify_engine::count_ops(self, expr)
    }

    /// Factor out the GCD of numeric coefficients from a sum.
    /// `2x + 2y → 2(x + y)`.
    /// Delegates to [`factor_terms::factor_terms`].
    pub fn factor_terms_expr(&mut self, expr: ExprId) -> ExprId {
        crate::factor_terms::factor_terms(self, expr)
    }

    /// Factor out the GCD of numeric coefficients, returning `(gcd, inner)` as expression IDs.
    /// Delegates to [`factor_terms::factor_terms_pair`].
    pub fn factor_terms_pair_expr(&mut self, expr: ExprId) -> (ExprId, ExprId) {
        let (gcd, inner) = crate::factor_terms::factor_terms_pair(self, expr);
        let gcd_id = {
            let nid = self.intern_num(gcd);
            self.intern(ExprNode::Num(nid))
        };
        (gcd_id, inner)
    }

    /// Rationalize the denominator of a fraction containing square roots.
    /// `1/√2 → √2/2`, `1/(1+√2) → √2-1`.
    /// Delegates to [`radsimp::rationalize_denom`].
    pub fn rationalize_denom_expr(&mut self, expr: ExprId) -> ExprId {
        crate::radsimp::rationalize_denom(self, expr)
    }

    /// Decompose an expression into real and imaginary parts.
    /// Returns `(re, im)` such that `expr = re + im * i`.
    /// Delegates to [`complex::as_real_imag`].
    pub fn as_real_imag_expr(&mut self, expr: ExprId) -> (ExprId, ExprId) {
        crate::complex::as_real_imag(self, expr)
    }

    /// Compute the polynomial GCD of `a` and `b` with respect to `var`.
    ///
    /// Returns `None` if either expression is not polynomial in `var`.
    pub fn poly_gcd_expr(&mut self, a: ExprId, b: ExprId, var: ExprId) -> Option<ExprId> {
        let pa = crate::polybridge::expr_to_poly(self, a, var)?;
        let pb = crate::polybridge::expr_to_poly(self, b, var)?;
        let g = crate::poly::Poly::gcd(&pa, &pb);
        Some(crate::polybridge::poly_to_expr(self, &g, var))
    }

    /// Compute the polynomial LCM of `a` and `b` with respect to `var`.
    ///
    /// Returns `None` if either expression is not polynomial in `var`.
    pub fn poly_lcm_expr(&mut self, a: ExprId, b: ExprId, var: ExprId) -> Option<ExprId> {
        let pa = crate::polybridge::expr_to_poly(self, a, var)?;
        let pb = crate::polybridge::expr_to_poly(self, b, var)?;
        let g = crate::poly::Poly::gcd(&pa, &pb);
        if g.is_zero() {
            return None;
        }
        // lcm(a,b) = a * b / gcd(a,b)
        let product = &pa * &pb;
        let (lcm, _rem) = product.div_rem(&g);
        Some(crate::polybridge::poly_to_expr(self, &lcm, var))
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

    /// Creates an `Abs` (absolute value / complex modulus) node.
    pub fn abs(&mut self, expr: ExprId) -> ExprId {
        self.intern(ExprNode::Abs(expr))
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

    /// Creates a `Factorial` node: `n!`
    pub fn factorial(&mut self, expr: ExprId) -> ExprId {
        self.intern(ExprNode::Factorial(expr))
    }

    /// Creates a `Binomial` node: `C(n, k)` = n! / (k! * (n-k)!)
    pub fn binomial(&mut self, n: ExprId, k: ExprId) -> ExprId {
        self.intern(ExprNode::Binomial(n, k))
    }

    /// Evaluate `expr` numerically to `digits` decimal digits of precision.
    ///
    /// Returns the string representation of the result, or an error if the
    /// expression contains free symbols.
    pub fn evalf_expr(
        &self,
        expr: ExprId,
        digits: u32,
    ) -> Result<String, crate::errors::SymplexError> {
        crate::evalf::evalf(self, expr, digits)
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
