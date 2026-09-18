//! User-facing expression handle.
//!
//! [`Ex`] is a lightweight handle to a symbolic expression. It holds a
//! reference-counted pointer to the shared `ContextInner` (which
//! contains both the arena and the assumption cache) plus an expression ID.
//! Clone is cheap (~5ns, just an Arc clone + u32 copy).
//!
//! `Ex` is **not** `Copy` because it contains an `Arc`. This is a
//! deliberate trade-off: storing the context pointer means every method
//! (`pow`, `sin`, `is_positive`, operators, `Display`) works without
//! requiring any special scoping.
//!
//! To reduce clone noise, all binary operators are implemented for every
//! combination of `Ex` and `&Ex`, and for `i64` on both sides.
//!
//! # Examples
//!
//! ```
//! use symplex::prelude::*;
//!
//! let ctx = Context::new();
//! let x = ctx.symbol("x");
//! let expr = &x * &x + &x * 2 + 1;
//! assert_eq!(format!("{expr}"), "x^2 + 2*x + 1");
//! ```
//!
//! Methods are chainable:
//!
//! ```
//! use symplex::prelude::*;
//!
//! let ctx = Context::new();
//! let x = ctx.symbol("x");
//! let expr = x.powi(2).sin();
//! assert_eq!(format!("{expr}"), "sin(x^2)");
//! ```
//!
//! ```
//! use symplex::prelude::*;
//!
//! let ctx = Context::new();
//! let x = ctx.symbol("x");
//! let expr = x.sin();
//! assert_eq!(format!("{expr}"), "sin(x)");
//! ```

use std::marker::PhantomData;
use std::sync::Arc;

use parking_lot::RwLock;
use tracing::debug_span;

use crate::api::context::ContextInner;
use crate::base::errors::SymplexError;
use crate::base::node::{CtxId, ExprId};

/// Options controlling [`Expr::simplify_with`].
pub use crate::simplify::simplify_engine::SimplifyOpts;

// ═══════════════════════════════════════════════════════════════════════════
// Sort system — compile-time expression typing
// ═══════════════════════════════════════════════════════════════════════════

/// Marker trait for expression sorts.
///
/// Sorts distinguish numeric expressions (which support arithmetic,
/// calculus, etc.) from boolean expressions (comparisons, logic)
/// at compile time. The sort is carried as a phantom type parameter
/// on [`Expr`] and has zero runtime cost.
pub trait Sort: 'static + Clone + Send + Sync {}

/// Numeric sort: real, complex, and integer values.
/// Supports arithmetic, calculus, and algebraic operations.
#[derive(Clone)]
pub struct Numeric;

/// Boolean sort: true/false values from comparisons and logic.
/// Supports `and`, `or`, `not` operations.
#[derive(Clone)]
pub struct Boolean;

/// Set-valued sort: intervals, finite sets, unions.
/// Supports set operations (union, intersection, complement).
#[derive(Clone)]
pub struct SetValued;

impl Sort for Numeric {}
impl Sort for Boolean {}
impl Sort for SetValued {}

/// Structural classification of an expression node.
///
/// Returned by [`Expr::expr_type()`]. This collapses the internal
/// `ExprNode` enum (over 70 variants) into a user-friendly classification.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::expr::ExprType;
///
/// let ctx = Context::new();
/// let x = ctx.symbol("x");
/// assert_eq!(x.expr_type(), ExprType::Symbol);
/// assert_eq!((&x + 1).expr_type(), ExprType::Add);
/// assert_eq!(x.sin().expr_type(), ExprType::Function);
/// assert_eq!(ctx.int(42).expr_type(), ExprType::Number);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExprType {
    /// A numeric literal (integer or rational).
    Number,
    /// A symbolic variable.
    Symbol,
    /// A mathematical constant (π, e, i, ∞, etc.).
    Constant,
    /// An n-ary sum.
    Add,
    /// An n-ary product.
    Mul,
    /// Exponentiation.
    Pow,
    /// Unary negation.
    Neg,
    /// A mathematical function (sin, cos, ln, exp, abs, etc.).
    Function,
    /// Application of a user-defined function.
    Apply,
    /// A formal derivative.
    Derivative,
    /// A formal integral.
    Integral,
    /// A set expression (interval, finite set, union, intersection, complement).
    Set,
    /// A formal/unevaluated computation (DefiniteIntegral, Limit, Series,
    /// LaplaceTransform, etc.)
    ///
    /// Every expression of this type also reports
    /// [`has_unevaluated`](Expr::has_unevaluated).  `RootOf`/`RootSum` are
    /// *not* unevaluated: they are exact algebraic values and classify as
    /// [`Constant`](Self::Constant) (numeric polynomial) or
    /// [`Function`](Self::Function) (parametric polynomial).
    Unevaluated,
}

/// A symbolic expression handle, parameterized by sort.
///
/// `Expr<Numeric>` (aliased as [`Ex`]) represents numeric expressions.
/// `Expr<Boolean>` (aliased as [`BoolEx`]) represents boolean expressions.
///
/// The sort parameter is a phantom type — zero runtime cost.
/// It prevents invalid operations at compile time:
/// - `sin(bool_expr)` won't compile (sin is only on `Expr<Numeric>`)
/// - `bool_expr + 1` won't compile (Add is only on `Expr<Numeric>`)
/// - `numeric.and(other)` won't compile (and is only on `Expr<Boolean>`)
#[derive(Clone)]
pub struct Expr<S: Sort> {
    pub(crate) ctx_id: CtxId,
    pub(crate) inner: Arc<RwLock<ContextInner>>,
    id: ExprId,
    pub(crate) _sort: PhantomData<S>,
}

/// A numeric expression — the primary type for symbolic math.
pub type Ex = Expr<Numeric>;

impl AsRef<Ex> for Ex {
    #[inline]
    fn as_ref(&self) -> &Ex {
        self
    }
}

/// A boolean expression — comparisons and logical operations.
pub type BoolEx = Expr<Boolean>;

/// A set-valued expression — intervals, finite sets, unions.
pub type SetEx = Expr<SetValued>;

// ═══════════════════════════════════════════════════════════════════════════
// impl<S: Sort> Expr<S> — wrap helpers
// ═══════════════════════════════════════════════════════════════════════════

impl<S: Sort> Expr<S> {
    /// Construct an `Expr` from raw parts.
    ///
    /// The caller must guarantee that `id` is a valid [`ExprId`] in the
    /// arena behind `inner`.  This is the **only** constructor — all other
    /// modules must go through it because the `id` field is private.
    #[inline]
    pub(crate) fn from_raw_parts(
        ctx_id: CtxId,
        inner: Arc<RwLock<ContextInner>>,
        id: ExprId,
    ) -> Self {
        Expr {
            ctx_id,
            inner,
            id,
            _sort: PhantomData,
        }
    }

    /// Helper — build a new Expr of the SAME sort from the same context.
    #[inline]
    pub(crate) fn wrap(&self, id: ExprId) -> Expr<S> {
        Expr::from_raw_parts(self.ctx_id, Arc::clone(&self.inner), id)
    }

    /// Helper — build a new Expr of a DIFFERENT sort from the same context.
    #[inline]
    pub(crate) fn wrap_as<T: Sort>(&self, id: ExprId) -> Expr<T> {
        Expr::from_raw_parts(self.ctx_id, Arc::clone(&self.inner), id)
    }

    /// Return this expression's arena index.
    ///
    /// This is always safe — you are accessing your own data within
    /// your own context's arena.
    #[inline]
    pub(crate) fn raw_id(&self) -> ExprId {
        self.id
    }

    /// Return another expression's arena index after verifying it belongs
    /// to the same [`Context`](crate::api::context::Context) as `self`.
    ///
    /// # Panics
    ///
    /// Panics with a descriptive message if `self` and `other` belong to
    /// different contexts.  This prevents silent data corruption from
    /// cross-context [`ExprId`] misuse.
    #[inline]
    pub(crate) fn checked_id<T: Sort>(&self, other: &Expr<T>) -> ExprId {
        if self.ctx_id != other.ctx_id {
            panic!(
                "symplex: cannot combine expressions from different contexts \
                 (context {} and context {}). All expressions in an operation \
                 must originate from the same Context.",
                self.ctx_id.0, other.ctx_id.0
            );
        }
        other.id
    }

    /// Returns a [`Context`](crate::api::context::Context) handle that
    /// shares this expression's arena and assumption cache.
    ///
    /// Useful when you need to create new expressions (constants, rationals)
    /// guaranteed to live in the same context as an existing expression.
    #[must_use]
    pub fn context(&self) -> crate::api::context::Context {
        crate::api::context::Context {
            id: self.ctx_id,
            inner: Arc::clone(&self.inner),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// impl<S: Sort> Expr<S> — Common (sort-preserving) methods
// ═══════════════════════════════════════════════════════════════════════════

impl<S: Sort> Expr<S> {
    /// Returns the raw `ExprId` inside this handle.
    ///
    /// **Note:** This is an opaque arena-local index.  It is only
    /// meaningful within the [`Context`](crate::api::context::Context)
    /// that created this expression.  Comparing `ExprId` values across
    /// contexts is undefined.
    #[inline]
    pub fn id(&self) -> ExprId {
        self.id
    }

    /// Returns the `CtxId` that this expression belongs to.
    #[inline]
    pub fn ctx_id(&self) -> CtxId {
        self.ctx_id
    }

    // ── Structural predicates ──────────────────────────────────────

    /// Returns `true` if this expression is structurally zero (O(1)).
    #[must_use]
    pub fn is_zero_structural(&self) -> bool {
        self.inner.read().arena.is_zero_structural(self.id)
    }

    /// Returns `true` if this expression is structurally one (O(1)).
    #[must_use]
    pub fn is_one_structural(&self) -> bool {
        self.inner.read().arena.is_one_structural(self.id)
    }

    // ── Structural introspection ───────────────────────────────────

    /// Returns the set of free symbols in this expression.
    ///
    /// Each symbol appears at most once. The order is deterministic
    /// but unspecified.
    ///
    /// Symbols are always numeric, so this returns `Vec<Ex>` regardless
    /// of the sort of `self`.
    #[must_use]
    pub fn free_symbols(&self) -> Vec<Ex> {
        let inner = self.inner.read();
        let expr_ids = crate::base::walk::free_symbols(&inner.arena, self.id);
        drop(inner);
        expr_ids
            .into_iter()
            .map(|eid| self.wrap_as::<Numeric>(eid))
            .collect()
    }

    /// Returns `true` if this expression contains any unevaluated formal
    /// nodes such as `Integral(...)`, `Derivative(...)`, `Limit(...)`, etc.
    ///
    /// Useful for checking whether a symbolic computation fully evaluated
    /// or left formal/unevaluated placeholders.
    #[must_use]
    pub fn has_unevaluated(&self) -> bool {
        let inner = self.inner.read();
        crate::base::walk::has_unevaluated(&inner.arena, self.raw_id())
    }

    /// Count the number of operations (non-atom nodes) in this expression.
    ///
    /// Atoms (numbers, symbols, constants) count as 0.
    /// Each operator or function application counts as 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!(x.count_ops(), 0);           // atom
    /// assert_eq!((&x + 1).count_ops(), 1);    // one Add
    /// assert_eq!(x.sin().powi(2).count_ops(), 2); // Sin + Pow
    /// ```
    #[must_use]
    pub fn count_ops(&self) -> usize {
        let inner = self.inner.read();
        inner.arena.count_ops(self.id)
    }

    /// Returns the number of top-level terms in this expression.
    ///
    /// For an `Add` node, returns the number of summands.
    /// For anything else, returns 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!((&x + 1).term_count(), 2);
    /// assert_eq!(x.powi(2).term_count(), 1);
    /// ```
    #[must_use]
    pub fn term_count(&self) -> usize {
        let inner = self.inner.read();
        match inner.arena.node(self.id) {
            crate::base::node::ExprNode::Add(children) => children.len(),
            _ => 1,
        }
    }

    /// Returns the direct children (arguments) of this expression.
    ///
    /// - For `Add`: returns the summands.
    /// - For `Mul`: returns the factors.
    /// - For `Pow`: returns `[base, exponent]`.
    /// - For `Neg`: returns `[inner]`.
    /// - For functions (sin, cos, etc.): returns `[argument]`.
    /// - For atoms (numbers, symbols, constants): returns `[]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = &x + 1;
    /// let children = expr.args();
    /// assert_eq!(children.len(), 2);
    /// ```
    #[must_use]
    pub fn args(&self) -> Vec<Expr<S>> {
        let inner = self.inner.read();
        let child_ids = inner.arena.node(self.id).children();
        child_ids.iter().map(|&id| self.wrap(id)).collect()
    }

    /// Returns the structural type of this expression.
    ///
    /// Collapses the internal `ExprNode` enum (over 70 variants) into a
    /// user-friendly [`ExprType`] classification.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::expr::ExprType;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!(x.expr_type(), ExprType::Symbol);
    /// assert_eq!(x.sin().expr_type(), ExprType::Function);
    /// assert_eq!((&x + 1).expr_type(), ExprType::Add);
    /// ```
    #[must_use]
    pub fn expr_type(&self) -> ExprType {
        let inner = self.inner.read();
        match inner.arena.node(self.id) {
            crate::base::node::ExprNode::Num(_) => ExprType::Number,
            crate::base::node::ExprNode::Symbol(_) => ExprType::Symbol,
            crate::base::node::ExprNode::Pi
            | crate::base::node::ExprNode::E
            | crate::base::node::ExprNode::ImaginaryUnit
            | crate::base::node::ExprNode::EulerGamma
            | crate::base::node::ExprNode::Catalan
            | crate::base::node::ExprNode::GoldenRatio
            | crate::base::node::ExprNode::PhysicalConstant(_, _)
            | crate::base::node::ExprNode::Infinity
            | crate::base::node::ExprNode::NegInfinity
            | crate::base::node::ExprNode::ComplexInfinity
            | crate::base::node::ExprNode::NaN => ExprType::Constant,
            crate::base::node::ExprNode::Add(_) => ExprType::Add,
            crate::base::node::ExprNode::Mul(_) => ExprType::Mul,
            crate::base::node::ExprNode::Pow(_, _) => ExprType::Pow,
            crate::base::node::ExprNode::Neg(_) => ExprType::Neg,
            crate::base::node::ExprNode::Sin(_)
            | crate::base::node::ExprNode::Cos(_)
            | crate::base::node::ExprNode::Tan(_)
            | crate::base::node::ExprNode::Exp(_)
            | crate::base::node::ExprNode::Ln(_)
            | crate::base::node::ExprNode::Abs(_)
            | crate::base::node::ExprNode::Asin(_)
            | crate::base::node::ExprNode::Acos(_)
            | crate::base::node::ExprNode::Atan(_)
            | crate::base::node::ExprNode::Atan2(_, _)
            | crate::base::node::ExprNode::Sinh(_)
            | crate::base::node::ExprNode::Cosh(_)
            | crate::base::node::ExprNode::Tanh(_)
            | crate::base::node::ExprNode::Asinh(_)
            | crate::base::node::ExprNode::Acosh(_)
            | crate::base::node::ExprNode::Atanh(_)
            | crate::base::node::ExprNode::Sign(_)
            | crate::base::node::ExprNode::Floor(_)
            | crate::base::node::ExprNode::Ceiling(_)
            | crate::base::node::ExprNode::Min(_)
            | crate::base::node::ExprNode::Max(_)
            | crate::base::node::ExprNode::Sum(_, _, _, _)
            | crate::base::node::ExprNode::Product_(_, _, _, _) => ExprType::Function,
            crate::base::node::ExprNode::Apply(_, _) => ExprType::Apply,
            crate::base::node::ExprNode::Derivative(_, _) => ExprType::Derivative,
            crate::base::node::ExprNode::Integral(_, _) => ExprType::Integral,
            crate::base::node::ExprNode::Factorial(_)
            | crate::base::node::ExprNode::Binomial(_, _)
            | crate::base::node::ExprNode::Gamma(_)
            | crate::base::node::ExprNode::LogGamma(_)
            | crate::base::node::ExprNode::Digamma(_)
            | crate::base::node::ExprNode::Erf(_)
            | crate::base::node::ExprNode::Erfc(_)
            | crate::base::node::ExprNode::LambertW(_)
            | crate::base::node::ExprNode::Beta(_, _)
            | crate::base::node::ExprNode::Re(_)
            | crate::base::node::ExprNode::Im(_)
            | crate::base::node::ExprNode::Conjugate(_)
            | crate::base::node::ExprNode::Arg(_)
            | crate::base::node::ExprNode::Si(_)
            | crate::base::node::ExprNode::Ci(_)
            | crate::base::node::ExprNode::Ei(_)
            | crate::base::node::ExprNode::Li(_)
            | crate::base::node::ExprNode::Zeta(_)
            | crate::base::node::ExprNode::Polygamma(_, _)
            | crate::base::node::ExprNode::KroneckerDelta(_, _) => ExprType::Function,
            crate::base::node::ExprNode::BoolTrue | crate::base::node::ExprNode::BoolFalse => {
                ExprType::Constant
            }
            crate::base::node::ExprNode::Gt(_, _)
            | crate::base::node::ExprNode::Ge(_, _)
            | crate::base::node::ExprNode::Eq_(_, _)
            | crate::base::node::ExprNode::Ne(_, _)
            | crate::base::node::ExprNode::And(_)
            | crate::base::node::ExprNode::Or(_)
            | crate::base::node::ExprNode::Not(_)
            | crate::base::node::ExprNode::Piecewise(_)
            | crate::base::node::ExprNode::Heaviside(_)
            | crate::base::node::ExprNode::DiracDelta(_) => ExprType::Function,
            crate::base::node::ExprNode::EmptySet
            | crate::base::node::ExprNode::UniversalSet
            | crate::base::node::ExprNode::Interval(_, _, _)
            | crate::base::node::ExprNode::FiniteSet(_)
            | crate::base::node::ExprNode::SetUnion(_)
            | crate::base::node::ExprNode::SetIntersection(_)
            | crate::base::node::ExprNode::SetComplement(_, _) => ExprType::Set,
            // `RootOf`/`RootSum` are complete algebraic values, not pending
            // computations (consistent with `has_unevaluated`, which does not
            // report them): a constant when the polynomial has numeric
            // coefficients, otherwise a function of its parameters.
            crate::base::node::ExprNode::RootOf(..) | crate::base::node::ExprNode::RootSum(..) => {
                if crate::base::walk::free_symbols(&inner.arena, self.id).is_empty() {
                    ExprType::Constant
                } else {
                    ExprType::Function
                }
            }
            crate::base::node::ExprNode::DefiniteIntegral(..)
            | crate::base::node::ExprNode::Limit(..)
            | crate::base::node::ExprNode::Series(..)
            | crate::base::node::ExprNode::LaplaceTransform(..)
            | crate::base::node::ExprNode::InverseLaplaceTransform(..)
            | crate::base::node::ExprNode::Residue(..)
            | crate::base::node::ExprNode::DSolve(..)
            | crate::base::node::ExprNode::ConditionSet(..) => ExprType::Unevaluated,
        }
    }

    // ── Substitution ───────────────────────────────────────────────

    /// Structural substitution: replace every occurrence of `old` with `new`.
    ///
    /// This is **structural** — only exact node matches are replaced.
    /// `(1/x).subs(x², 1)` returns `1/x` unchanged because `x²` does
    /// not appear as a node in `x⁻¹`.
    ///
    /// The result is re-canonicalized, so like-term collection and
    /// other invariants are maintained.
    ///
    /// Returns `self` unchanged (same `Expr`) if `old` does not appear.
    #[must_use = "returns a new expression with substitutions applied"]
    pub fn subs(&self, old: &Ex, new: &Ex) -> Expr<S> {
        let old_id = self.checked_id(old);
        let new_id = self.checked_id(new);
        let id = self
            .inner
            .write()
            .arena
            .subs_structural(self.id, old_id, new_id);
        self.wrap(id)
    }

    /// Substitute a symbol with an integer value.
    ///
    /// Convenience shorthand for `self.subs(old, &ctx.int(n))` that
    /// avoids needing to construct the integer expression manually.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = x.powi(2);
    /// let result = expr.subs_i64(&x, 3);
    /// assert_eq!(format!("{result}"), "9");
    /// ```
    #[must_use = "returns a new expression with substitutions applied"]
    pub fn subs_i64(&self, old: &Ex, new: i64) -> Expr<S> {
        let old_id = self.checked_id(old);
        let mut inner = self.inner.write();
        let new_id = inner.arena.int(new);
        let id = inner.arena.subs_structural(self.id, old_id, new_id);
        drop(inner);
        self.wrap(id)
    }

    /// Simultaneous substitution of multiple `(old, new)` pairs.
    ///
    /// All replacements happen "at once" — earlier substitutions do
    /// not affect later ones.
    #[must_use = "returns a new expression with substitutions applied"]
    pub fn subs_map(&self, replacements: &[(&Ex, &Ex)]) -> Expr<S> {
        let pairs: smallvec::SmallVec<[(crate::base::node::ExprId, crate::base::node::ExprId); 4]> =
            replacements
                .iter()
                .map(|(o, n)| (self.checked_id(o), self.checked_id(n)))
                .collect();
        let id = self
            .inner
            .write()
            .arena
            .subs_map_structural(self.id, &pairs);
        self.wrap(id)
    }

    // ── Transformations ────────────────────────────────────────────

    /// Algebraic expansion (distribute products over sums, expand
    /// integer powers of sums).
    ///
    /// - `a * (b + c)` → `a*b + a*c`
    /// - `(a + b)^n` → multinomial expansion
    ///
    /// Does NOT evaluate functions, factor, or simplify.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = (&x + 1).powi(2);
    /// assert_eq!(format!("{}", expr.expand()), "x^2 + 2*x + 1");
    /// ```
    #[must_use = "returns the expanded form; does not modify in place"]
    pub fn expand(&self) -> Expr<S> {
        let _span = debug_span!("expand", expr = ?self.id).entered();
        let id = self.inner.write().arena.expand_expr(self.id);
        self.wrap(id)
    }

    /// Like [`simplify`](Ex::simplify), but with configurable options.
    ///
    /// Use [`SimplifyOpts`] to control the fixpoint iteration count.
    /// This always runs the numeric simplification engine, whatever the
    /// sort of `self`; for [`BoolEx`] and [`SetEx`] prefer their own
    /// `simplify()`, which additionally applies boolean / set algebra.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = &x.sin().powi(2) + &x.cos().powi(2);
    ///
    /// // Single-pass simplification (no fixpoint iteration)
    /// let result = expr.simplify_with(&SimplifyOpts::single_pass());
    /// assert_eq!(format!("{}", result), "1");
    /// ```
    #[must_use = "returns the simplified form; does not modify in place"]
    pub fn simplify_with(&self, opts: &SimplifyOpts) -> Expr<S> {
        let _span = debug_span!("simplify_with", expr = ?self.id).entered();
        let result = {
            let mut inner = self.inner.write();
            crate::simplify::simplify_engine::unified_simplify(&mut inner.arena, self.id, opts)
        };
        self.wrap(result.expr)
    }

    // ── Serialization ──────────────────────────────────────────────

    /// Convert this expression to a standalone serializable [`ExprTree`](crate::output::tree::ExprTree).
    ///
    /// The tree can be serialized to JSON (or any serde format) and
    /// deserialized back via [`Context::from_tree()`](crate::api::context::Context::from_tree).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let tree = x.powi(2).to_tree();
    /// let json = serde_json::to_string(&tree).unwrap();
    /// assert!(json.contains("Pow"));
    /// ```
    #[must_use = "returns a serializable tree; does not modify in place"]
    pub fn to_tree(&self) -> crate::output::tree::ExprTree {
        let inner = self.inner.read();
        crate::output::tree::expr_to_tree(&inner.arena, self.id)
    }

    /// Serialize this expression to a JSON string.
    ///
    /// This is a convenience shorthand for
    /// `serde_json::to_string(&expr.to_tree())`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let json = x.powi(2).to_json().unwrap();
    /// assert!(json.contains("\"type\":\"Pow\""));
    /// ```
    pub fn to_json(&self) -> Result<String, SymplexError> {
        serde_json::to_string(&self.to_tree()).map_err(|e| SymplexError::ComputationFailed {
            operation: "to_json",
            reason: e.to_string(),
        })
    }

    /// Serialize this expression to a pretty-printed JSON string.
    pub fn to_json_pretty(&self) -> Result<String, SymplexError> {
        serde_json::to_string_pretty(&self.to_tree()).map_err(|e| SymplexError::ComputationFailed {
            operation: "to_json_pretty",
            reason: e.to_string(),
        })
    }

    /// Apply a transformation repeatedly until the expression stops changing,
    /// or `max_iterations` is reached.
    ///
    /// Returns the final expression and the number of iterations performed.
    /// Useful for building custom simplification pipelines.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = (&x + 1).powi(2);
    /// let (result, iters) = expr.apply_until_stable(10, |e| e.expand());
    /// assert_eq!(format!("{result}"), "x^2 + 2*x + 1");
    /// assert_eq!(iters, 1); // stabilized after 1 iteration
    /// ```
    #[must_use = "returns the stabilized expression and iteration count"]
    pub fn apply_until_stable<F>(&self, max_iterations: usize, f: F) -> (Expr<S>, usize)
    where
        F: Fn(&Expr<S>) -> Expr<S>,
    {
        let mut current = self.clone();
        for i in 0..max_iterations {
            let next = f(&current);
            if next.raw_id() == current.raw_id() && next.ctx_id == current.ctx_id {
                return (current, i);
            }
            current = next;
        }
        (current, max_iterations)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// impl Expr<Numeric> — evaluation / simplification / structural search
//
// These used to live on the generic `impl<S: Sort>` block.  They are now
// per-sort so that `BoolEx` and `SetEx` can provide their own `eval`,
// `simplify` and `contains` with boolean / set semantics (see
// `expr_sets_ext.rs`).
// ═══════════════════════════════════════════════════════════════════════════

impl Expr<Numeric> {
    /// Returns `true` if `needle` appears as a sub-expression of `self`.
    ///
    /// This is a structural check — it walks the expression DAG and
    /// returns `true` if any node has the same `ExprId` as `needle`.
    ///
    /// For set membership use [`SetEx::contains`] / [`Ex::is_in`].
    #[must_use]
    pub fn contains(&self, needle: &Ex) -> bool {
        let needle_id = self.checked_id(needle);
        let inner = self.inner.read();
        crate::base::walk::contains(&inner.arena, self.id, needle_id)
    }

    /// Exact evaluation of known special values.
    ///
    /// Replaces function applications with their exact values when the
    /// arguments are known constants:
    ///
    /// - `sin(0)` → `0`, `sin(π)` → `0`, `sin(π/2)` → `1`
    /// - `cos(0)` → `1`, `cos(π)` → `-1`
    /// - `exp(0)` → `1`, `ln(1)` → `0`
    /// - `sqrt(4)` → `2`, `abs(-3)` → `3`
    ///
    /// Only evaluates when the result is a simpler atom.
    /// Does NOT evaluate `cos(π/4)` → `√2/2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let expr = ctx.pi().cos();
    /// assert_eq!(format!("{}", expr.eval()), "-1");
    /// ```
    #[must_use = "returns the evaluated form; does not modify in place"]
    pub fn eval(&self) -> Ex {
        let _span = debug_span!("eval", expr = ?self.id).entered();
        let id = self.inner.write().arena.eval_expr(self.id);
        self.wrap(id)
    }

    /// Simplification (identity application, trig identities, etc.).
    ///
    /// Simplify the expression using all available strategies.
    ///
    /// Tries 12+ strategies (eval, expand, factor, trig, log, cancel,
    /// power, radical, assumption-aware refinement, …), picks the
    /// simplest result, then iterates to a fixpoint (up to 10 passes)
    /// until the expression stops getting simpler.
    ///
    /// This is the "just make this simpler" function.  For finer control,
    /// use [`simplify_with`](Self::simplify_with) or the domain-specific
    /// methods ([`simplify_trig`](Ex::simplify_trig),
    /// [`simplify_powers`](Ex::simplify_powers), etc.).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    ///
    /// // Trig identity
    /// let expr = &x.sin().powi(2) + &x.cos().powi(2);
    /// assert_eq!(format!("{}", expr.simplify()), "1");
    ///
    /// // Polynomial cancellation
    /// let expr = &(&x + 1).powi(2) - &x.powi(2) - &x * 2;
    /// assert_eq!(format!("{}", expr.simplify()), "1");
    /// ```
    #[must_use = "returns the simplified form; does not modify in place"]
    pub fn simplify(&self) -> Ex {
        let _span = debug_span!("simplify", expr = ?self.id).entered();
        let result = {
            let mut inner = self.inner.write();
            crate::simplify::simplify_engine::unified_simplify(
                &mut inner.arena,
                self.id,
                &crate::simplify::simplify_engine::SimplifyOpts::default(),
            )
        };
        self.wrap(result.expr)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// impl Expr<Boolean> — Boolean-specific methods
//
// Boolean algebra (`simplify`, `eval`, CNF/DNF, satisfiability, …) lives
// in `expr_sets_ext.rs`; the connectives and escape hatches are here.
// ═══════════════════════════════════════════════════════════════════════════

impl Expr<Boolean> {
    /// Returns `true` if `needle` appears as a sub-expression of `self`.
    ///
    /// This is a structural check — it walks the expression DAG and
    /// returns `true` if any node has the same `ExprId` as `needle`.
    #[must_use]
    pub fn contains(&self, needle: &Ex) -> bool {
        let needle_id = self.checked_id(needle);
        let inner = self.inner.read();
        crate::base::walk::contains(&inner.arena, self.id, needle_id)
    }

    /// Logical conjunction: `self & other`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn and(&self, other: &BoolEx) -> BoolEx {
        let other_id = self.checked_id(other);
        let id = self.inner.write().arena.and(&[self.id, other_id]);
        self.wrap(id)
    }

    /// Logical disjunction: `self | other`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn or(&self, other: &BoolEx) -> BoolEx {
        let other_id = self.checked_id(other);
        let id = self.inner.write().arena.or(&[self.id, other_id]);
        self.wrap(id)
    }

    /// Logical negation: `!self`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn not(&self) -> BoolEx {
        let id = self.inner.write().arena.not(self.id);
        self.wrap(id)
    }

    /// Exclusive or: `self ⊕ other = (self ∧ ¬other) ∨ (¬self ∧ other)`.
    #[must_use]
    pub fn xor(&self, other: &BoolEx) -> BoolEx {
        self.and(&other.not()).or(&self.not().and(other))
    }

    /// Logical implication: `self → other = ¬self ∨ other`.
    #[must_use]
    pub fn implies(&self, other: &BoolEx) -> BoolEx {
        self.not().or(other)
    }

    /// Logical biconditional: `self ↔ other = (self → other) ∧ (other → self)`.
    #[must_use]
    pub fn equivalent(&self, other: &BoolEx) -> BoolEx {
        self.implies(other).and(&other.implies(self))
    }

    /// NAND gate: `¬(self ∧ other)`.
    #[must_use]
    pub fn nand(&self, other: &BoolEx) -> BoolEx {
        self.and(other).not()
    }

    /// NOR gate: `¬(self ∨ other)`.
    #[must_use]
    pub fn nor(&self, other: &BoolEx) -> BoolEx {
        self.or(other).not()
    }

    /// If-then-else: `if self then a else b` = `(self ∧ a) ∨ (¬self ∧ b)`.
    #[must_use]
    pub fn ite(&self, then_: &BoolEx, else_: &BoolEx) -> BoolEx {
        let _ = self.checked_id(then_);
        let _ = self.checked_id(else_);
        self.and(then_).or(&self.not().and(else_))
    }

    /// Convert to untyped numeric expression (escape hatch).
    pub fn into_ex(self) -> Ex {
        Ex {
            ctx_id: self.ctx_id,
            inner: self.inner,
            id: self.id,
            _sort: PhantomData,
        }
    }

    /// Borrow as untyped numeric expression.
    pub fn as_ex(&self) -> Ex {
        Ex {
            ctx_id: self.ctx_id,
            inner: Arc::clone(&self.inner),
            id: self.id,
            _sort: PhantomData,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// impl Expr<SetValued> — Set-specific methods
//
// The three constructors below are *lazy* (structural, canonicalised
// only — principle 1: construction is cheap, evaluation is explicit).
// Set algebra proper (`simplify`, `difference`, `contains`, `is_subset`,
// `inf`/`sup`/`measure`, topology, …) lives in `expr_sets_ext.rs`.
// ═══════════════════════════════════════════════════════════════════════════

impl Expr<SetValued> {
    /// Union: `self ∪ other` (structural; call [`simplify`](SetEx::simplify)
    /// to merge overlapping intervals).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    /// let b = ctx.interval(&ctx.int(2), &ctx.int(3), false, false);
    /// let u = a.union(&b);
    /// let s = format!("{u}");
    /// assert!(!s.is_empty(), "union display: {s}");
    /// ```
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn union(&self, other: &SetEx) -> SetEx {
        let other_id = self.checked_id(other);
        let id = self.inner.write().arena.set_union(&[self.id, other_id]);
        self.wrap(id)
    }

    /// Intersection: `self ∩ other` (structural; call
    /// [`simplify`](SetEx::simplify) to evaluate).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    /// let e = ctx.empty_set();
    /// let result = a.intersection(&e);
    /// assert_eq!(format!("{result}"), "EmptySet");
    /// ```
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn intersection(&self, other: &SetEx) -> SetEx {
        let id = self
            .inner
            .write()
            .arena
            .set_intersection(&[self.id, self.checked_id(other)]);
        self.wrap(id)
    }

    /// Relative complement: `self \ other` (structural).
    ///
    /// This is the lazy constructor; [`difference`](SetEx::difference)
    /// returns the evaluated normal form and
    /// [`absolute_complement`](SetEx::absolute_complement) computes `ℝ \ self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = ctx.interval(&ctx.int(0), &ctx.int(3), false, false);
    /// let b = ctx.interval(&ctx.int(1), &ctx.int(2), true, true);
    /// let lazy = a.complement(&b);
    /// assert_eq!(format!("{lazy}"), "[0, 3] \\ (1, 2)");
    /// assert_eq!(format!("{}", lazy.simplify()), "[0, 1] ∪ [2, 3]");
    /// ```
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn complement(&self, other: &SetEx) -> SetEx {
        let other_id = self.checked_id(other);
        let id = self.inner.write().arena.set_complement(self.id, other_id);
        self.wrap(id)
    }

    /// Convert to untyped numeric expression (escape hatch).
    ///
    /// This allows set-valued expressions to be embedded in contexts
    /// that expect `Ex`. The underlying arena node is unchanged.
    pub fn into_ex(self) -> Ex {
        Ex {
            ctx_id: self.ctx_id,
            inner: self.inner,
            id: self.id,
            _sort: PhantomData,
        }
    }

    /// Borrow as untyped numeric expression.
    pub fn as_ex(&self) -> Ex {
        Ex {
            ctx_id: self.ctx_id,
            inner: Arc::clone(&self.inner),
            id: self.id,
            _sort: PhantomData,
        }
    }
}
