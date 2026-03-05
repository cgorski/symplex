//! User-facing expression handle.
//!
//! [`Ex`] is a lightweight handle to a symbolic expression. It holds a
//! reference-counted pointer to the shared [`ContextInner`] (which
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

use crate::context::ContextInner;
use crate::node::{CtxId, ExprId};

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
/// `ExprNode` enum (30 variants) into a user-friendly classification.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::expr::ExprType;
///
/// let x = symplex::var("x");
/// assert_eq!(x.expr_type(), ExprType::Symbol);
/// assert_eq!((&x + 1).expr_type(), ExprType::Add);
/// assert_eq!(x.sin().expr_type(), ExprType::Function);
/// assert_eq!(symplex::int(42).expr_type(), ExprType::Number);
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
    pub(crate) id: ExprId,
    pub(crate) _sort: PhantomData<S>,
}

/// A numeric expression — the primary type for symbolic math.
pub type Ex = Expr<Numeric>;

/// A boolean expression — comparisons and logical operations.
pub type BoolEx = Expr<Boolean>;

/// A set-valued expression — intervals, finite sets, unions.
pub type SetEx = Expr<SetValued>;

// ═══════════════════════════════════════════════════════════════════════════
// impl<S: Sort> Expr<S> — wrap helpers
// ═══════════════════════════════════════════════════════════════════════════

impl<S: Sort> Expr<S> {
    /// Helper — build a new Expr of the SAME sort from the same context.
    #[inline]
    pub(crate) fn wrap(&self, id: ExprId) -> Expr<S> {
        Expr {
            ctx_id: self.ctx_id,
            inner: Arc::clone(&self.inner),
            id,
            _sort: PhantomData,
        }
    }

    /// Helper — build a new Expr of a DIFFERENT sort from the same context.
    #[inline]
    pub(crate) fn wrap_as<T: Sort>(&self, id: ExprId) -> Expr<T> {
        Expr {
            ctx_id: self.ctx_id,
            inner: Arc::clone(&self.inner),
            id,
            _sort: PhantomData,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// impl<S: Sort> Expr<S> — Common (sort-preserving) methods
// ═══════════════════════════════════════════════════════════════════════════

impl<S: Sort> Expr<S> {
    /// Returns the raw [`ExprId`] inside this handle.
    #[inline]
    pub fn id(&self) -> ExprId {
        self.id
    }

    /// Returns the [`CtxId`] that this expression belongs to.
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
        let expr_ids = crate::walk::free_symbols(&inner.arena, self.id);
        drop(inner);
        expr_ids
            .into_iter()
            .map(|eid| self.wrap_as::<Numeric>(eid))
            .collect()
    }

    /// Returns `true` if `needle` appears as a sub-expression of `self`.
    ///
    /// This is a structural check — it walks the expression DAG and
    /// returns `true` if any node has the same [`ExprId`] as `needle`.
    #[must_use]
    pub fn contains(&self, needle: &Ex) -> bool {
        let inner = self.inner.read();
        crate::walk::contains(&inner.arena, self.id, needle.id)
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
    /// let x = symplex::var("x");
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
    /// let x = symplex::var("x");
    /// assert_eq!((&x + 1).term_count(), 2);
    /// assert_eq!(x.powi(2).term_count(), 1);
    /// ```
    #[must_use]
    pub fn term_count(&self) -> usize {
        let inner = self.inner.read();
        match inner.arena.node(self.id) {
            crate::node::ExprNode::Add(children) => children.len(),
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
    /// Collapses the internal 30-variant `ExprNode` enum into a
    /// user-friendly [`ExprType`] classification.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::expr::ExprType;
    ///
    /// let x = symplex::var("x");
    /// assert_eq!(x.expr_type(), ExprType::Symbol);
    /// assert_eq!(x.sin().expr_type(), ExprType::Function);
    /// assert_eq!((&x + 1).expr_type(), ExprType::Add);
    /// ```
    #[must_use]
    pub fn expr_type(&self) -> ExprType {
        let inner = self.inner.read();
        match inner.arena.node(self.id) {
            crate::node::ExprNode::Num(_) => ExprType::Number,
            crate::node::ExprNode::Symbol(_) => ExprType::Symbol,
            crate::node::ExprNode::Pi
            | crate::node::ExprNode::E
            | crate::node::ExprNode::ImaginaryUnit
            | crate::node::ExprNode::Infinity
            | crate::node::ExprNode::NegInfinity
            | crate::node::ExprNode::ComplexInfinity
            | crate::node::ExprNode::NaN => ExprType::Constant,
            crate::node::ExprNode::Add(_) => ExprType::Add,
            crate::node::ExprNode::Mul(_) => ExprType::Mul,
            crate::node::ExprNode::Pow(_, _) => ExprType::Pow,
            crate::node::ExprNode::Neg(_) => ExprType::Neg,
            crate::node::ExprNode::Sin(_)
            | crate::node::ExprNode::Cos(_)
            | crate::node::ExprNode::Tan(_)
            | crate::node::ExprNode::Exp(_)
            | crate::node::ExprNode::Ln(_)
            | crate::node::ExprNode::Abs(_)
            | crate::node::ExprNode::Asin(_)
            | crate::node::ExprNode::Acos(_)
            | crate::node::ExprNode::Atan(_)
            | crate::node::ExprNode::Atan2(_, _)
            | crate::node::ExprNode::Sinh(_)
            | crate::node::ExprNode::Cosh(_)
            | crate::node::ExprNode::Tanh(_)
            | crate::node::ExprNode::Asinh(_)
            | crate::node::ExprNode::Acosh(_)
            | crate::node::ExprNode::Atanh(_)
            | crate::node::ExprNode::Sign(_)
            | crate::node::ExprNode::Floor(_)
            | crate::node::ExprNode::Ceiling(_)
            | crate::node::ExprNode::Min(_)
            | crate::node::ExprNode::Max(_)
            | crate::node::ExprNode::Sum(_, _, _, _)
            | crate::node::ExprNode::Product_(_, _, _, _) => ExprType::Function,
            crate::node::ExprNode::Apply(_, _) => ExprType::Apply,
            crate::node::ExprNode::Derivative(_, _) => ExprType::Derivative,
            crate::node::ExprNode::Integral(_, _) => ExprType::Integral,
            crate::node::ExprNode::Factorial(_)
            | crate::node::ExprNode::Binomial(_, _)
            | crate::node::ExprNode::Gamma(_)
            | crate::node::ExprNode::LogGamma(_)
            | crate::node::ExprNode::Digamma(_)
            | crate::node::ExprNode::Erf(_)
            | crate::node::ExprNode::Erfc(_)
            | crate::node::ExprNode::Beta(_, _) => ExprType::Function,
            crate::node::ExprNode::BoolTrue | crate::node::ExprNode::BoolFalse => {
                ExprType::Constant
            }
            crate::node::ExprNode::Gt(_, _)
            | crate::node::ExprNode::Ge(_, _)
            | crate::node::ExprNode::Eq_(_, _)
            | crate::node::ExprNode::Ne(_, _)
            | crate::node::ExprNode::And(_)
            | crate::node::ExprNode::Or(_)
            | crate::node::ExprNode::Not(_)
            | crate::node::ExprNode::Piecewise(_)
            | crate::node::ExprNode::Heaviside(_)
            | crate::node::ExprNode::DiracDelta(_) => ExprType::Function,
            crate::node::ExprNode::EmptySet
            | crate::node::ExprNode::UniversalSet
            | crate::node::ExprNode::Interval(_, _, _)
            | crate::node::ExprNode::FiniteSet(_)
            | crate::node::ExprNode::SetUnion(_)
            | crate::node::ExprNode::SetIntersection(_)
            | crate::node::ExprNode::SetComplement(_, _) => ExprType::Set,
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
        let id = self
            .inner
            .write()
            .arena
            .subs_structural(self.id, old.id, new.id);
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
    /// let x = symplex::var("x");
    /// let expr = x.powi(2);
    /// let result = expr.subs_i64(&x, 3);
    /// assert_eq!(format!("{result}"), "9");
    /// ```
    #[must_use = "returns a new expression with substitutions applied"]
    pub fn subs_i64(&self, old: &Ex, new: i64) -> Expr<S> {
        let mut inner = self.inner.write();
        let new_id = inner.arena.int(new);
        let id = inner.arena.subs_structural(self.id, old.id, new_id);
        drop(inner);
        self.wrap(id)
    }

    /// Simultaneous substitution of multiple `(old, new)` pairs.
    ///
    /// All replacements happen "at once" — earlier substitutions do
    /// not affect later ones.
    #[must_use = "returns a new expression with substitutions applied"]
    pub fn subs_map(&self, replacements: &[(&Ex, &Ex)]) -> Expr<S> {
        let pairs: smallvec::SmallVec<[(crate::node::ExprId, crate::node::ExprId); 4]> =
            replacements.iter().map(|(o, n)| (o.id, n.id)).collect();
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
    pub fn eval(&self) -> Expr<S> {
        let _span = debug_span!("eval", expr = ?self.id).entered();
        let id = self.inner.write().arena.eval_expr(self.id);
        self.wrap(id)
    }

    /// Simplification (identity application, trig identities, etc.).
    ///
    /// Applies built-in rewrite rules (e.g., `sin²(x) + cos²(x) → 1`)
    /// in a single bottom-up pass.  For fixpoint simplification, call
    /// repeatedly until the result stops changing.
    ///
    /// ⚠️ **This function is heuristic.**  For deterministic
    /// transformations, use `.expand()`, `.eval()`, or `.diff()` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = &x.sin().powi(2) + &x.cos().powi(2);
    /// assert_eq!(format!("{}", expr.simplify()), "1");
    /// ```
    #[must_use = "returns the simplified form; does not modify in place"]
    pub fn simplify(&self) -> Expr<S> {
        let _span = debug_span!("simplify", expr = ?self.id).entered();
        let (result, _steps) = self.simplify_trace();
        result
    }

    /// Like [`simplify`](Expr::simplify), but also returns a trace of
    /// which rules fired and what they changed.
    ///
    /// Each [`Step`](crate::pattern::Step) records the rule name, the
    /// sub-expression before, and the sub-expression after.
    #[must_use = "returns the simplified form and trace"]
    pub fn simplify_trace(&self) -> (Expr<S>, Vec<crate::pattern::Step>) {
        let mut inner = self.inner.write();
        // Use cached rules if available, otherwise build and cache them.
        if inner.cached_rules.is_none() {
            inner.cached_rules = Some(crate::pattern::basic_rules(&mut inner.arena));
        }
        // Clone the rules out to release the immutable borrow on `inner`
        // before we pass `&mut inner.arena` to `apply_rules`.
        let rules = inner
            .cached_rules
            .as_ref()
            .expect("cached_rules should be populated by is_none() check above")
            .clone();
        let (result_id, steps) = crate::pattern::apply_rules(&mut inner.arena, self.id, &rules);
        drop(inner);
        (self.wrap(result_id), steps)
    }

    /// Full simplification: repeatedly applies [`eval`](Expr::eval),
    /// [`expand`](Expr::expand), and [`simplify`](Expr::simplify) until
    /// the expression stops changing (fixpoint), or a maximum of 10
    /// iterations is reached.
    ///
    /// This is the "just make this simpler" button — it composes all
    /// available simplification passes into a single call.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let x = symplex::var("x");
    /// // (x+1)^2 - x^2 - 2*x simplifies to 1 after expand + canonicalization
    /// let expr = &(&x + 1).powi(2) - &x.powi(2) - &x * 2;
    /// assert_eq!(format!("{}", expr.full_simplify()), "1");
    /// ```
    #[must_use = "returns the fully simplified form; does not modify in place"]
    pub fn full_simplify(&self) -> Expr<S> {
        let (result, _steps) = self.full_simplify_trace();
        result
    }

    /// Like [`full_simplify`](Expr::full_simplify), but also returns a
    /// trace of all steps across all iterations.
    #[must_use = "returns the fully simplified form and accumulated trace"]
    pub fn full_simplify_trace(&self) -> (Expr<S>, Vec<crate::pattern::Step>) {
        let _span = tracing::debug_span!("full_simplify_trace", expr = ?self.id).entered();
        let (result_id, steps) = {
            let mut guard = self.inner.write();
            crate::simplify_engine::full_simplify_trace(&mut guard.arena, self.id)
        };
        (self.wrap(result_id), steps)
    }

    /// Try multiple simplification strategies and return the simplest result.
    ///
    /// Unlike [`simplify`](Self::simplify) which applies a single pass of
    /// rewrite rules, this tries eval, expand, factor_terms, trig_expand,
    /// logcombine and more, then picks whichever result has the fewest
    /// operations (nodes).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = &x.sin().powi(2) + &x.cos().powi(2);
    /// assert_eq!(format!("{}", expr.smart_simplify()), "1");
    /// ```
    #[must_use = "returns the simplified form; does not modify in place"]
    pub fn smart_simplify(&self) -> Expr<S> {
        let id = self.inner.write().arena.smart_simplify_expr(self.id);
        self.wrap(id)
    }

    // ── Serialization ──────────────────────────────────────────────

    /// Convert this expression to a standalone serializable [`ExprTree`](crate::tree::ExprTree).
    ///
    /// The tree can be serialized to JSON (or any serde format) and
    /// deserialized back via [`Context::from_tree()`](crate::context::Context::from_tree).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let x = symplex::var("x");
    /// let tree = x.powi(2).to_tree();
    /// let json = serde_json::to_string(&tree).unwrap();
    /// assert!(json.contains("Pow"));
    /// ```
    #[must_use = "returns a serializable tree; does not modify in place"]
    pub fn to_tree(&self) -> crate::tree::ExprTree {
        let inner = self.inner.read();
        crate::tree::expr_to_tree(&inner.arena, self.id)
    }

    /// Serialize this expression to a JSON string.
    ///
    /// This is a convenience shorthand for
    /// `serde_json::to_string(&expr.to_tree()).unwrap()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let x = symplex::var("x");
    /// let json = x.powi(2).to_json();
    /// assert!(json.contains("\"type\":\"Pow\""));
    /// ```
    #[must_use = "returns a JSON string; does not modify in place"]
    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.to_tree()).expect("ExprTree serialization should not fail")
    }

    /// Serialize this expression to a pretty-printed JSON string.
    #[must_use = "returns a JSON string; does not modify in place"]
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(&self.to_tree())
            .expect("ExprTree serialization should not fail")
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
    /// let x = symplex::var("x");
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
            if next.id == current.id && next.ctx_id == current.ctx_id {
                return (current, i);
            }
            current = next;
        }
        (current, max_iterations)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// impl Expr<Boolean> — Boolean-specific methods
// ═══════════════════════════════════════════════════════════════════════════

impl Expr<Boolean> {
    /// Logical conjunction: `self & other`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn and(&self, other: &BoolEx) -> BoolEx {
        let id = self.inner.write().arena.and(&[self.id, other.id]);
        self.wrap(id)
    }

    /// Logical disjunction: `self | other`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn or(&self, other: &BoolEx) -> BoolEx {
        let id = self.inner.write().arena.or(&[self.id, other.id]);
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
// ═══════════════════════════════════════════════════════════════════════════

impl Expr<SetValued> {
    /// Union: `self ∪ other`.
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
        let id = self.inner.write().arena.set_union(&[self.id, other.id]);
        self.wrap(id)
    }

    /// Intersection: `self ∩ other`.
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
            .set_intersection(&[self.id, other.id]);
        self.wrap(id)
    }

    /// Relative complement: `self \ other`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn complement(&self, other: &SetEx) -> SetEx {
        let id = self.inner.write().arena.set_complement(self.id, other.id);
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
