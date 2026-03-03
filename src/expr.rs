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
//! assert_eq!(format!("{expr}"), "1 + x^2 + 2*x");
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

use std::fmt;
use std::hash;
use std::ops;
use std::sync::Arc;

use parking_lot::RwLock;
use tracing::debug_span;

use crate::assumptions::Assumption;
use crate::assumptions::Props;
use crate::context::ContextInner;
use crate::display::fmt_expr;
use crate::errors::SymplexError;
use crate::node::{CtxId, ExprId};

/// A symbolic expression handle.
///
/// `Ex` wraps an [`ExprId`] together with a reference to the
/// [`ContextInner`] that owns the arena and assumption cache.
/// It is cheaply cloneable and implements standard arithmetic operators.
///
/// # Locking
///
/// - **Operators / construction methods** acquire the outer `RwLock` in
///   write mode (exclusive access to the arena).
/// - **Display / structural predicates** acquire the outer `RwLock` in
///   read mode (shared access).
/// - **Assumption queries** acquire the outer `RwLock` in read mode,
///   then the inner `Mutex<AssumptionCache>` for cache mutation.
#[derive(Clone)]
pub struct Ex {
    pub(crate) ctx_id: CtxId,
    pub(crate) inner: Arc<RwLock<ContextInner>>,
    pub(crate) id: ExprId,
}

impl Ex {
    /// Helper — build a new `Ex` from the same context with a different id.
    #[inline]
    fn wrap(&self, id: ExprId) -> Ex {
        Ex {
            ctx_id: self.ctx_id,
            inner: Arc::clone(&self.inner),
            id,
        }
    }

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

    // ── Math functions ─────────────────────────────────────────────

    /// Raise to a symbolic power: `self ^ exp`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn pow(&self, exp: &Ex) -> Ex {
        let id = self.inner.write().arena.pow(self.id, exp.id);
        self.wrap(id)
    }

    /// Raise to an integer power: `self ^ n`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn powi(&self, n: i64) -> Ex {
        let mut inner = self.inner.write();
        let exp = inner.arena.int(n);
        let id = inner.arena.pow(self.id, exp);
        drop(inner);
        self.wrap(id)
    }

    /// Sine: `sin(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn sin(&self) -> Ex {
        let id = self.inner.write().arena.sin(self.id);
        self.wrap(id)
    }

    /// Cosine: `cos(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn cos(&self) -> Ex {
        let id = self.inner.write().arena.cos(self.id);
        self.wrap(id)
    }

    /// Tangent: `tan(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn tan(&self) -> Ex {
        let id = self.inner.write().arena.tan(self.id);
        self.wrap(id)
    }

    /// Natural exponential: `e^self`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn exp_fn(&self) -> Ex {
        let id = self.inner.write().arena.exp_fn(self.id);
        self.wrap(id)
    }

    /// Natural logarithm: `ln(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn ln(&self) -> Ex {
        let id = self.inner.write().arena.ln(self.id);
        self.wrap(id)
    }

    /// Principal square root: `√self`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn sqrt(&self) -> Ex {
        let id = self.inner.write().arena.sqrt(self.id);
        self.wrap(id)
    }

    /// Absolute value (or complex modulus): `|self|`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn abs(&self) -> Ex {
        let id = self.inner.write().arena.abs(self.id);
        self.wrap(id)
    }

    /// Inverse sine (arcsin): `asin(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn asin(&self) -> Ex {
        let id = self.inner.write().arena.asin(self.id);
        self.wrap(id)
    }

    /// Inverse cosine (arccos): `acos(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn acos(&self) -> Ex {
        let id = self.inner.write().arena.acos(self.id);
        self.wrap(id)
    }

    /// Inverse tangent (arctan): `atan(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn atan(&self) -> Ex {
        let id = self.inner.write().arena.atan(self.id);
        self.wrap(id)
    }

    /// Hyperbolic sine: `sinh(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn sinh(&self) -> Ex {
        let id = self.inner.write().arena.sinh(self.id);
        self.wrap(id)
    }

    /// Hyperbolic cosine: `cosh(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn cosh(&self) -> Ex {
        let id = self.inner.write().arena.cosh(self.id);
        self.wrap(id)
    }

    /// Hyperbolic tangent: `tanh(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn tanh(&self) -> Ex {
        let id = self.inner.write().arena.tanh(self.id);
        self.wrap(id)
    }

    // ── Structural predicates ──────────────────────────────────────

    /// Returns `true` if this expression is structurally zero (O(1)).
    pub fn is_zero_structural(&self) -> bool {
        self.inner.read().arena.is_zero_structural(self.id)
    }

    /// Returns `true` if this expression is structurally one (O(1)).
    pub fn is_one_structural(&self) -> bool {
        self.inner.read().arena.is_one_structural(self.id)
    }

    // ── Assumption queries ─────────────────────────────────────────

    /// Query whether this expression is positive.
    ///
    /// Returns `Some(true)` if provably positive, `Some(false)` if
    /// provably not positive, or `None` if unknown.
    pub fn is_positive(&self) -> Option<bool> {
        let inner = self.inner.read();
        inner
            .assumptions
            .lock()
            .query(&inner.arena, self.id, Props::POSITIVE)
    }

    /// Query whether this expression is zero.
    ///
    /// Uses layered detection:
    /// 1. Structural identity with 0 (O(1))
    /// 2. Assumption system query
    ///
    /// Returns `Some(true)` if provably zero, `Some(false)` if provably
    /// nonzero, or `None` if unknown.
    pub fn is_zero(&self) -> Option<bool> {
        let inner = self.inner.read();
        // Layer 1: structural.
        if inner.arena.is_zero_structural(self.id) {
            return Some(true);
        }
        // Layer 2: assumption system.
        inner
            .assumptions
            .lock()
            .query(&inner.arena, self.id, Props::ZERO)
    }

    /// Query any mathematical property via [`Props`].
    ///
    /// Returns `Some(true)` if provably true, `Some(false)` if provably
    /// false, or `None` if unknown.
    pub fn query(&self, prop: Props) -> Option<bool> {
        let inner = self.inner.read();
        inner.assumptions.lock().query(&inner.arena, self.id, prop)
    }

    /// Query whether this expression is negative.
    ///
    /// Returns `Some(true)` if provably negative, `Some(false)` if
    /// provably not negative, or `None` if unknown.
    pub fn is_negative(&self) -> Option<bool> {
        self.query(Props::NEGATIVE)
    }

    /// Query whether this expression is real.
    ///
    /// Returns `Some(true)` if provably real, `Some(false)` if
    /// provably not real, or `None` if unknown.
    pub fn is_real(&self) -> Option<bool> {
        self.query(Props::REAL)
    }

    /// Query whether this expression is an integer.
    ///
    /// Returns `Some(true)` if provably an integer, `Some(false)` if
    /// provably not an integer, or `None` if unknown.
    pub fn is_integer(&self) -> Option<bool> {
        self.query(Props::INTEGER)
    }

    /// Query whether this expression is nonzero.
    ///
    /// Returns `Some(true)` if provably nonzero, `Some(false)` if
    /// provably zero, or `None` if unknown.
    pub fn is_nonzero(&self) -> Option<bool> {
        self.query(Props::NONZERO)
    }

    /// Query whether this expression is finite.
    ///
    /// Returns `Some(true)` if provably finite, `Some(false)` if
    /// provably not finite, or `None` if unknown.
    pub fn is_finite(&self) -> Option<bool> {
        self.query(Props::FINITE)
    }

    // ── Assumption mutation ────────────────────────────────────────

    /// Set a mathematical assumption on this expression (must be a symbol).
    ///
    /// Returns `self` for fluent chaining. If this expression is not a
    /// symbol, the assumption is silently ignored.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let t = symplex::var("t")
    ///     .assume(Assumption::Positive)
    ///     .assume(Assumption::Real);
    /// assert_eq!(t.is_positive(), Some(true));
    /// assert_eq!(t.is_real(), Some(true));
    /// ```
    pub fn assume(self, assumption: Assumption) -> Ex {
        use crate::node::ExprNode;
        let mut inner = self.inner.write();
        if let ExprNode::Symbol(sid) = inner.arena.node(self.id) {
            let sid = *sid;
            let (prop, value) = assumption.to_prop_value();
            let mut a = inner.arena.symbol_assumptions(sid).clone();
            if value {
                a.assert_true(prop);
            } else {
                a.assert_false(prop);
            }
            inner.arena.set_symbol_assumptions(sid, a);
            inner.assumptions.lock().set_symbol_assumptions(self.id, a);
        }
        drop(inner);
        self
    }

    // ── Structural introspection ───────────────────────────────────

    /// Returns the set of free symbols in this expression.
    ///
    /// Each symbol appears at most once. The order is deterministic
    /// but unspecified.
    pub fn free_symbols(&self) -> Vec<Ex> {
        let inner = self.inner.read();
        let expr_ids = crate::walk::free_symbols(&inner.arena, self.id);
        drop(inner);
        expr_ids.into_iter().map(|eid| self.wrap(eid)).collect()
    }

    /// Returns `true` if `needle` appears as a sub-expression of `self`.
    ///
    /// This is a structural check — it walks the expression DAG and
    /// returns `true` if any node has the same [`ExprId`] as `needle`.
    pub fn contains(&self, needle: &Ex) -> bool {
        let inner = self.inner.read();
        crate::walk::contains(&inner.arena, self.id, needle.id)
    }

    // ── Transformations ────────────────────────────────────────────

    /// Symbolic differentiation with respect to `var`.
    ///
    /// Computes the derivative using standard rules (linearity, product
    /// rule, chain rule, power rule) for all supported node types.
    ///
    /// `var` should be a symbol expression (created via `ctx.symbol()`).
    /// If `var` does not appear in the expression, the result is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = x.powi(3);
    /// let deriv = expr.diff(&x);
    /// assert_eq!(format!("{deriv}"), "3*x^2");
    /// ```
    #[must_use = "returns the derivative as a new expression"]
    pub fn diff(&self, var: &Ex) -> Ex {
        let _span = debug_span!("diff", expr = ?self.id, var = ?var.id).entered();
        let id = self.inner.write().arena.diff_wrt(self.id, var.id);
        self.wrap(id)
    }

    /// Structural substitution: replace every occurrence of `old` with `new`.
    ///
    /// This is **structural** — only exact node matches are replaced.
    /// `(1/x).subs(x², 1)` returns `1/x` unchanged because `x²` does
    /// not appear as a node in `x⁻¹`.
    ///
    /// The result is re-canonicalized, so like-term collection and
    /// other invariants are maintained.
    ///
    /// Returns `self` unchanged (same `Ex`) if `old` does not appear.
    #[must_use = "returns a new expression with substitutions applied"]
    pub fn subs(&self, old: &Ex, new: &Ex) -> Ex {
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
    pub fn subs_i64(&self, old: &Ex, new: i64) -> Ex {
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
    pub fn subs_map(&self, replacements: &[(&Ex, &Ex)]) -> Ex {
        let pairs: smallvec::SmallVec<[(crate::node::ExprId, crate::node::ExprId); 4]> =
            replacements.iter().map(|(o, n)| (o.id, n.id)).collect();
        let id = self
            .inner
            .write()
            .arena
            .subs_map_structural(self.id, &pairs);
        self.wrap(id)
    }

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
    /// assert_eq!(format!("{}", expr.expand()), "1 + x^2 + 2*x");
    /// ```
    #[must_use = "returns the expanded form; does not modify in place"]
    pub fn expand(&self) -> Ex {
        let _span = debug_span!("expand", expr = ?self.id).entered();
        let id = self.inner.write().arena.expand_expr(self.id);
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
    pub fn simplify(&self) -> Ex {
        let _span = debug_span!("simplify", expr = ?self.id).entered();
        let (result, _steps) = self.simplify_trace();
        result
    }

    /// Like [`simplify`](Ex::simplify), but also returns a trace of
    /// which rules fired and what they changed.
    ///
    /// Each [`Step`](crate::pattern::Step) records the rule name, the
    /// sub-expression before, and the sub-expression after.
    #[must_use = "returns the simplified form and trace"]
    pub fn simplify_trace(&self) -> (Ex, Vec<crate::pattern::Step>) {
        let mut inner = self.inner.write();
        // Use cached rules if available, otherwise build and cache them.
        if inner.cached_rules.is_none() {
            inner.cached_rules = Some(crate::pattern::basic_rules(&mut inner.arena));
        }
        // Clone the rules out to release the immutable borrow on `inner`
        // before we pass `&mut inner.arena` to `apply_rules`.
        let rules = inner.cached_rules.as_ref().unwrap().clone();
        let (result_id, steps) = crate::pattern::apply_rules(&mut inner.arena, self.id, &rules);
        drop(inner);
        (self.wrap(result_id), steps)
    }

    /// Full simplification: repeatedly applies [`eval`](Ex::eval),
    /// [`expand`](Ex::expand), and [`simplify`](Ex::simplify) until
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
    pub fn full_simplify(&self) -> Ex {
        let _span = debug_span!("full_simplify", expr = ?self.id).entered();
        let (result, _steps) = self.full_simplify_trace();
        result
    }

    /// Like [`full_simplify`](Ex::full_simplify), but also returns a
    /// trace of all steps across all iterations.
    #[must_use = "returns the fully simplified form and accumulated trace"]
    pub fn full_simplify_trace(&self) -> (Ex, Vec<crate::pattern::Step>) {
        const MAX_ITERATIONS: usize = 10;
        let mut current = self.clone();
        let mut all_steps: Vec<crate::pattern::Step> = Vec::new();

        for _ in 0..MAX_ITERATIONS {
            let evaled = current.eval();
            let expanded = evaled.expand();
            let (simplified, steps) = expanded.simplify_trace();
            all_steps.extend(steps);

            if simplified.id == current.id {
                return (simplified, all_steps);
            }
            current = simplified;
        }

        (current, all_steps)
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

    /// Solve `self = 0` for the given variable.
    ///
    /// Returns a vector of values of `var` that make this expression
    /// zero.  Supports linear, quadratic, and higher-degree polynomial
    /// equations (via rational root finding).
    ///
    /// Returns an empty vector if:
    /// - The expression is not polynomial in `var`.
    /// - No closed-form solutions can be found.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // Solve x² - 5x + 6 = 0
    /// let expr = &x.powi(2) - &x * 5 + 6;
    /// let solutions = expr.solve(&x);
    /// assert_eq!(solutions.len(), 2);
    /// ```
    pub fn solve(&self, var: &Ex) -> Vec<Ex> {
        let _span = debug_span!("solve", expr = ?self.id, var = ?var.id).entered();
        let solutions = self.inner.write().arena.solve_for(self.id, var.id);
        solutions
            .into_iter()
            .map(|sol| self.wrap(sol.value))
            .collect()
    }

    /// Cancel common polynomial factors in a rational expression.
    ///
    /// Decomposes the expression into numerator and denominator,
    /// converts both to univariate polynomials in `var`, divides out
    /// their GCD, and rebuilds the expression.
    ///
    /// Returns the expression unchanged if it is not a rational
    /// function in `var` or if there is no common factor.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // (x² - 1) / (x - 1) → x + 1
    /// let expr = (&x.powi(2) - 1) / (&x - 1);
    /// let cancelled = expr.cancel(&x);
    /// assert_eq!(format!("{cancelled}"), "1 + x");
    /// ```
    #[must_use = "returns the cancelled form; does not modify in place"]
    pub fn cancel(&self, var: &Ex) -> Ex {
        let _span = debug_span!("cancel", expr = ?self.id, var = ?var.id).entered();
        let id = self.inner.write().arena.cancel_expr(self.id, var.id);
        self.wrap(id)
    }

    /// Group an expression by powers of `var`.
    ///
    /// Converts the expression to a univariate polynomial in `var`
    /// and rebuilds it, naturally grouping coefficients by power.
    ///
    /// Returns the expression unchanged if it is not polynomial in `var`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let expr = &x * &y + &x.powi(2) + &y;
    /// let collected = expr.collect(&x);
    /// // Terms are grouped by powers of x.
    /// assert_eq!(format!("{collected}"), "y + x^2 + x*y");
    /// ```
    #[must_use = "returns the collected form; does not modify in place"]
    pub fn collect(&self, var: &Ex) -> Ex {
        let id = self.inner.write().arena.collect_expr(self.id, var.id);
        self.wrap(id)
    }

    /// Combine fractions over a common denominator.
    ///
    /// For a sum of terms, decomposes each into numerator/denominator,
    /// computes a common denominator, scales each numerator, and
    /// rebuilds as a single fraction.
    ///
    /// Returns the expression unchanged if it is not a sum or if all
    /// terms already have denominator 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let expr = &x.powi(-1) + &y.powi(-1);
    /// let combined = expr.together();
    /// // 1/x + 1/y → (x + y) / (x*y)
    /// let s = format!("{combined}");
    /// assert!(s.contains("x*y"), "should have common denom x*y: {s}");
    /// ```
    #[must_use = "returns the combined form; does not modify in place"]
    pub fn together(&self) -> Ex {
        let id = self.inner.write().arena.together_expr(self.id);
        self.wrap(id)
    }

    /// Compute the indefinite integral with respect to `var`.
    ///
    /// Supports power rule, trigonometric, exponential, linearity,
    /// and constant factor extraction. For integrands that don't match
    /// any known rule, returns an unevaluated `Integral(body, var)` node.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = x.powi(2);
    /// let anti = expr.integrate(&x);
    /// assert_eq!(format!("{anti}"), "1/3*x^3");
    /// ```
    #[must_use = "returns the antiderivative; does not modify in place"]
    pub fn integrate(&self, var: &Ex) -> Ex {
        let _span = debug_span!("integrate", expr = ?self.id, var = ?var.id).entered();
        let id = self.inner.write().arena.integrate_expr(self.id, var.id);
        self.wrap(id)
    }

    /// Compute a definite integral: `∫_lower^upper self dx`.
    ///
    /// Computes the antiderivative via [`integrate`](Ex::integrate),
    /// then evaluates `F(upper) - F(lower)`. If the antiderivative
    /// is unevaluated (returned an `Integral` node), the result will
    /// contain unevaluated terms.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // ∫₀¹ x² dx = 1/3
    /// let result = x.powi(2).definite_integral(&x, &ctx.int(0), &ctx.int(1));
    /// assert_eq!(format!("{result}"), "1/3");
    /// ```
    #[must_use = "returns the definite integral value"]
    pub fn definite_integral(&self, var: &Ex, lower: &Ex, upper: &Ex) -> Ex {
        let anti = self.integrate(var);
        let f_upper = anti.subs(var, upper);
        let f_lower = anti.subs(var, lower);
        &f_upper - &f_lower
    }

    /// Return the degree of this expression as a polynomial in `var`.
    ///
    /// Returns `Some(n)` if the expression is a polynomial of degree `n`
    /// in `var`, or `None` if it is not polynomial (e.g., contains `sin(x)`)
    /// or is the zero polynomial.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let x = symplex::var("x");
    /// assert_eq!((&x.powi(3) + &x + 1).degree(&x), Some(3));
    /// assert_eq!(x.sin().degree(&x), None);
    /// ```
    pub fn degree(&self, var: &Ex) -> Option<usize> {
        let inner = self.inner.read();
        inner.arena.degree_of(self.id, var.id)
    }

    /// Return the coefficients of this expression as a polynomial in `var`,
    /// in ascending degree order: `[a_0, a_1, a_2, ...]`.
    ///
    /// Returns `None` if the expression is not polynomial in `var`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // x^2 + 3*x + 5 → coefficients [5, 3, 1]
    /// let expr = &x.powi(2) + &x * 3 + 5;
    /// let cs = expr.coeffs(&x).unwrap();
    /// let strs: Vec<String> = cs.iter().map(|c| format!("{c}")).collect();
    /// assert_eq!(strs, vec!["5", "3", "1"]);
    /// ```
    pub fn coeffs(&self, var: &Ex) -> Option<Vec<Ex>> {
        let mut inner = self.inner.write();
        let ids = inner.arena.coefficients_of(self.id, var.id)?;
        drop(inner);
        Some(ids.into_iter().map(|id| self.wrap(id)).collect())
    }

    /// Compute the Taylor series around `point` to the given `order`.
    ///
    /// Returns the truncated polynomial with `order` terms:
    /// `f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + ...`
    ///
    /// If `point` is zero, this is a Maclaurin series.
    /// Returns the original expression unchanged if expansion around
    /// the point is not possible (e.g., pole at the expansion point).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let zero = ctx.int(0);
    /// let expr = x.exp_fn();
    /// let s = expr.series(&x, &zero, 4);
    /// let expanded = s.expand().eval();
    /// let result = format!("{expanded}");
    /// assert!(result.contains("x"), "should have x term: {result}");
    /// ```
    #[must_use = "returns the series expansion; does not modify in place"]
    pub fn series(&self, var: &Ex, point: &Ex, order: u32) -> Ex {
        let _span = debug_span!("series", expr = ?self.id, order = order).entered();
        let id = self
            .inner
            .write()
            .arena
            .series_expr(self.id, var.id, point.id, order);
        self.wrap(id)
    }

    /// Compute the Maclaurin series (Taylor series around 0) to the
    /// given `order`.
    ///
    /// This is a convenience shorthand for `self.series(var, &zero, order)`
    /// that avoids needing to construct a zero expression manually.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let x = symplex::var("x");
    /// let s = x.sin().maclaurin(&x, 4);
    /// let result = s.expand().eval();
    /// let text = format!("{result}");
    /// assert!(text.contains("x"), "should have x term: {text}");
    /// ```
    #[must_use = "returns the series expansion; does not modify in place"]
    pub fn maclaurin(&self, var: &Ex, order: u32) -> Ex {
        let mut inner = self.inner.write();
        let zero = inner.arena.zero;
        let id = inner.arena.series_expr(self.id, var.id, zero, order);
        drop(inner);
        self.wrap(id)
    }

    /// Factor a polynomial expression into a product of linear factors.
    ///
    /// Finds rational roots via the equation solver, extracts content
    /// (GCD of coefficients), and handles root multiplicities.
    ///
    /// Returns the expression unchanged if it is not polynomial in `var`
    /// or if no rational roots can be found.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = &x.powi(2) - 1;
    /// let factored = expr.factor(&x);
    /// let s = format!("{factored}");
    /// // Should be factored into (x-1)(x+1) form
    /// assert!(!s.contains("x^2"), "should be factored: {s}");
    /// ```
    #[must_use = "returns the factored form; does not modify in place"]
    pub fn factor(&self, var: &Ex) -> Ex {
        let _span = debug_span!("factor", expr = ?self.id, var = ?var.id).entered();
        let id = self.inner.write().arena.factor_expr(self.id, var.id);
        self.wrap(id)
    }

    /// Compute the limit of this expression as `var` approaches `point`.
    ///
    /// Uses direct substitution, L'Hôpital's rule (for 0/0 and ∞/∞),
    /// and series expansion as fallbacks. Returns the expression
    /// unchanged if the limit cannot be determined.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // lim_{x→0} sin(x)/x = 1
    /// let expr = &x.sin() / &x;
    /// let result = expr.limit(&x, &ctx.int(0));
    /// assert_eq!(format!("{result}"), "1");
    /// ```
    #[must_use = "returns the limit value; does not modify in place"]
    pub fn limit(&self, var: &Ex, point: &Ex) -> Ex {
        let _span = debug_span!("limit", expr = ?self.id, var = ?var.id).entered();
        let id = self
            .inner
            .write()
            .arena
            .limit_expr(self.id, var.id, point.id);
        self.wrap(id)
    }

    /// Decompose this expression into (numerator, denominator).
    ///
    /// For `a / b` (expressed as `a * b^(-1)`), returns `(a, b)`.
    /// For expressions without a denominator, returns `(self, 1)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let expr = &x / &y;
    /// let (n, d) = expr.as_numer_denom();
    /// assert_eq!(format!("{n}"), "x");
    /// assert_eq!(format!("{d}"), "y");
    /// ```
    pub fn as_numer_denom(&self) -> (Ex, Ex) {
        let mut inner = self.inner.write();
        let (n, d) = inner.arena.as_numer_denom_expr(self.id);
        drop(inner);
        (self.wrap(n), self.wrap(d))
    }

    /// Returns `true` if this expression is a polynomial in `var`.
    ///
    /// Equivalent to `self.degree(var).is_some()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let x = symplex::var("x");
    /// assert!((&x.powi(2) + 1).is_polynomial(&x));
    /// assert!(!x.sin().is_polynomial(&x));
    /// ```
    pub fn is_polynomial(&self, var: &Ex) -> bool {
        self.degree(var).is_some()
    }

    /// Extract the coefficient of `var^n` in this expression.
    ///
    /// Returns `None` if the expression is not polynomial in `var`.
    /// Returns the zero expression if the coefficient is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = &x.powi(2) * 3 + &x * 5 + 7;
    /// assert_eq!(format!("{}", expr.coeff(&x, 2).unwrap()), "3");
    /// assert_eq!(format!("{}", expr.coeff(&x, 1).unwrap()), "5");
    /// assert_eq!(format!("{}", expr.coeff(&x, 0).unwrap()), "7");
    /// ```
    pub fn coeff(&self, var: &Ex, power: usize) -> Option<Ex> {
        let cs = self.coeffs(var)?;
        if power < cs.len() {
            Some(cs[power].clone())
        } else {
            // Coefficient is zero for powers above the degree.
            let inner = self.inner.read();
            let zero = inner.arena.zero;
            drop(inner);
            Some(self.wrap(zero))
        }
    }

    /// Numeric floating-point evaluation to the given number of decimal
    /// digits.
    ///
    /// Returns the decimal string representation of the evaluated expression.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::FreeSymbol`] if the expression contains
    /// unbound symbols (e.g., `x.evalf(10)` without substituting a value).
    ///
    /// Returns [`SymplexError::Unevaluable`] if the expression contains
    /// nodes that cannot be evaluated to a finite number (infinity, NaN,
    /// imaginary unit, unevaluated derivatives/integrals, user functions).
    ///
    /// Returns [`SymplexError::PrecisionExhausted`] if the requested
    /// precision exceeds `EvalConfig::max_evalf_precision`, or if
    /// intermediate computation produces NaN.
    #[must_use = "returns the numerical value as a string"]
    pub fn evalf(&self, digits: u32) -> Result<String, SymplexError> {
        let _span = debug_span!("evalf", expr = ?self.id, digits = digits).entered();
        let guard = self.inner.read();
        crate::evalf::evalf(&guard.arena, self.id, digits)
    }

    /// Convenience: evaluate to an `f64`.
    ///
    /// Calls [`evalf`](Ex::evalf) with 16 digits of precision and
    /// parses the result to `f64`. This avoids the common pattern of
    /// `.evalf(15).unwrap().parse::<f64>().unwrap()`.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`evalf`](Ex::evalf), plus a
    /// [`SymplexError::NotImplemented`] if the decimal string cannot
    /// be parsed to `f64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let x = symplex::var("x");
    /// let val = x.powi(2).subs_i64(&x, 3).evalf_f64().unwrap();
    /// assert!((val - 9.0).abs() < 1e-10);
    /// ```
    pub fn evalf_f64(&self) -> Result<f64, SymplexError> {
        let s = self.evalf(16)?;
        s.parse::<f64>().map_err(|e| {
            SymplexError::NotImplemented(format!("could not parse '{}' as f64: {}", s, e))
        })
    }

    // ── Collection reduction ───────────────────────────────────────

    /// Sum a collection of expressions.
    ///
    /// All expressions must belong to the same context. Returns zero
    /// if the iterator is empty (using the context of `ctx`).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let terms: Vec<Ex> = (1..=4).map(|n| ctx.int(n)).collect();
    /// let total = Ex::sum_of(&ctx, terms);
    /// assert_eq!(format!("{total}"), "10");
    /// ```
    pub fn sum_of(ctx: &crate::context::Context, exprs: impl IntoIterator<Item = Ex>) -> Ex {
        let items: Vec<Ex> = exprs.into_iter().collect();
        if items.is_empty() {
            return ctx.int(0);
        }
        let mut inner = items[0].inner.write();
        let ids: smallvec::SmallVec<[crate::node::ExprId; 8]> =
            items.iter().map(|e| e.id).collect();
        let id = inner.arena.add(&ids);
        drop(inner);
        items[0].wrap(id)
    }

    /// Multiply a collection of expressions.
    ///
    /// All expressions must belong to the same context. Returns one
    /// if the iterator is empty (using the context of `ctx`).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let factors: Vec<Ex> = (1..=4).map(|n| ctx.int(n)).collect();
    /// let total = Ex::product_of(&ctx, factors);
    /// assert_eq!(format!("{total}"), "24");
    /// ```
    pub fn product_of(ctx: &crate::context::Context, exprs: impl IntoIterator<Item = Ex>) -> Ex {
        let items: Vec<Ex> = exprs.into_iter().collect();
        if items.is_empty() {
            return ctx.int(1);
        }
        let mut inner = items[0].inner.write();
        let ids: smallvec::SmallVec<[crate::node::ExprId; 8]> =
            items.iter().map(|e| e.id).collect();
        let id = inner.arena.mul(&ids);
        drop(inner);
        items[0].wrap(id)
    }

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
    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.to_tree()).expect("ExprTree serialization should not fail")
    }

    /// Serialize this expression to a pretty-printed JSON string.
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
    /// assert_eq!(format!("{result}"), "1 + x^2 + 2*x");
    /// assert_eq!(iters, 1); // stabilized after 1 iteration
    /// ```
    pub fn apply_until_stable<F>(&self, max_iterations: usize, f: F) -> (Ex, usize)
    where
        F: Fn(&Ex) -> Ex,
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
    pub fn term_count(&self) -> usize {
        let inner = self.inner.read();
        match inner.arena.node(self.id) {
            crate::node::ExprNode::Add(children) => children.len(),
            _ => 1,
        }
    }

    /// Mathematical equality: attempts to determine if `self - other == 0`.
    ///
    /// Uses layered detection:
    /// 1. Structural identity (same `ExprId` — O(1))
    /// 2. Compute `self - other` and check if canonically zero
    /// 3. Expand `self - other` and check again
    ///
    /// Returns `Some(true)` if provably equal, `Some(false)` if provably
    /// not equal, or `None` if unknown.
    pub fn equals(&self, other: &Ex) -> Option<bool> {
        // Layer 1: structural identity (same arena node).
        if self.id == other.id && self.ctx_id == other.ctx_id {
            return Some(true);
        }

        // Layer 2: compute self - other and check if zero.
        let diff = self - other;
        if diff.is_zero_structural() {
            return Some(true);
        }

        // Layer 3: expand the difference and check again.
        let expanded = diff.expand();
        if expanded.is_zero_structural() {
            return Some(true);
        }

        // Could not determine equality.
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PartialEq / Eq / Hash
// ═══════════════════════════════════════════════════════════════════════════

impl PartialEq for Ex {
    fn eq(&self, other: &Self) -> bool {
        self.ctx_id == other.ctx_id && self.id == other.id
    }
}

impl Eq for Ex {}

impl hash::Hash for Ex {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.ctx_id.hash(state);
        self.id.hash(state);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Display / Debug
// ═══════════════════════════════════════════════════════════════════════════

impl fmt::Display for Ex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.inner.read();
        fmt_expr(&inner.arena, f, self.id, 0)
    }
}

impl fmt::Debug for Ex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ex({:?}, {:?})", self.ctx_id, self.id)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Operator implementations
// ═══════════════════════════════════════════════════════════════════════════

// ── Macro for n-ary binary operators (Add, Mul) ────────────────────────
//
// These go through `arena.add(&[lhs, rhs])` / `arena.mul(&[lhs, rhs])`.

macro_rules! impl_nary_binop {
    ($trait:ident, $method:ident, $arena_method:ident) => {
        impl ops::$trait<Ex> for Ex {
            type Output = Ex;
            fn $method(self, rhs: Ex) -> Ex {
                let id = self.inner.write().arena.$arena_method(&[self.id, rhs.id]);
                self.wrap(id)
            }
        }
        impl ops::$trait<&Ex> for Ex {
            type Output = Ex;
            fn $method(self, rhs: &Ex) -> Ex {
                let id = self.inner.write().arena.$arena_method(&[self.id, rhs.id]);
                self.wrap(id)
            }
        }
        impl ops::$trait<Ex> for &Ex {
            type Output = Ex;
            fn $method(self, rhs: Ex) -> Ex {
                let id = self.inner.write().arena.$arena_method(&[self.id, rhs.id]);
                self.wrap(id)
            }
        }
        impl ops::$trait<&Ex> for &Ex {
            type Output = Ex;
            fn $method(self, rhs: &Ex) -> Ex {
                let id = self.inner.write().arena.$arena_method(&[self.id, rhs.id]);
                self.wrap(id)
            }
        }
    };
}

impl_nary_binop!(Add, add, add);
impl_nary_binop!(Mul, mul, mul);

// ── Macro for binary operators (Sub, Div) ──────────────────────────────
//
// These go through `arena.sub(lhs, rhs)` / `arena.div(lhs, rhs)`.

macro_rules! impl_binary_binop {
    ($trait:ident, $method:ident, $arena_method:ident) => {
        impl ops::$trait<Ex> for Ex {
            type Output = Ex;
            fn $method(self, rhs: Ex) -> Ex {
                let id = self.inner.write().arena.$arena_method(self.id, rhs.id);
                self.wrap(id)
            }
        }
        impl ops::$trait<&Ex> for Ex {
            type Output = Ex;
            fn $method(self, rhs: &Ex) -> Ex {
                let id = self.inner.write().arena.$arena_method(self.id, rhs.id);
                self.wrap(id)
            }
        }
        impl ops::$trait<Ex> for &Ex {
            type Output = Ex;
            fn $method(self, rhs: Ex) -> Ex {
                let id = self.inner.write().arena.$arena_method(self.id, rhs.id);
                self.wrap(id)
            }
        }
        impl ops::$trait<&Ex> for &Ex {
            type Output = Ex;
            fn $method(self, rhs: &Ex) -> Ex {
                let id = self.inner.write().arena.$arena_method(self.id, rhs.id);
                self.wrap(id)
            }
        }
    };
}

impl_binary_binop!(Sub, sub, sub);
impl_binary_binop!(Div, div, div);

// ── Neg ────────────────────────────────────────────────────────────────

impl ops::Neg for Ex {
    type Output = Ex;
    fn neg(self) -> Ex {
        let id = self.inner.write().arena.neg(self.id);
        self.wrap(id)
    }
}

impl ops::Neg for &Ex {
    type Output = Ex;
    fn neg(self) -> Ex {
        let id = self.inner.write().arena.neg(self.id);
        self.wrap(id)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// i64 operator implementations
// ═══════════════════════════════════════════════════════════════════════════

// ── Macro for n-ary i64 operators (Add, Mul) ───────────────────────────

macro_rules! impl_nary_binop_i64 {
    ($trait:ident, $method:ident, $arena_method:ident) => {
        impl ops::$trait<i64> for Ex {
            type Output = Ex;
            fn $method(self, rhs: i64) -> Ex {
                let mut inner = self.inner.write();
                let rhs_id = inner.arena.int(rhs);
                let id = inner.arena.$arena_method(&[self.id, rhs_id]);
                drop(inner);
                self.wrap(id)
            }
        }
        impl ops::$trait<i64> for &Ex {
            type Output = Ex;
            fn $method(self, rhs: i64) -> Ex {
                let mut inner = self.inner.write();
                let rhs_id = inner.arena.int(rhs);
                let id = inner.arena.$arena_method(&[self.id, rhs_id]);
                drop(inner);
                self.wrap(id)
            }
        }
        impl ops::$trait<Ex> for i64 {
            type Output = Ex;
            fn $method(self, rhs: Ex) -> Ex {
                let mut inner = rhs.inner.write();
                let lhs_id = inner.arena.int(self);
                let id = inner.arena.$arena_method(&[lhs_id, rhs.id]);
                drop(inner);
                rhs.wrap(id)
            }
        }
        impl ops::$trait<&Ex> for i64 {
            type Output = Ex;
            fn $method(self, rhs: &Ex) -> Ex {
                let mut inner = rhs.inner.write();
                let lhs_id = inner.arena.int(self);
                let id = inner.arena.$arena_method(&[lhs_id, rhs.id]);
                drop(inner);
                rhs.wrap(id)
            }
        }
    };
}

impl_nary_binop_i64!(Add, add, add);
impl_nary_binop_i64!(Mul, mul, mul);

// ── Macro for binary i64 operators (Sub, Div) ──────────────────────────
//
// These go through `arena.sub(lhs, rhs)` / `arena.div(lhs, rhs)`.

macro_rules! impl_binary_binop_i64 {
    ($trait:ident, $method:ident, $arena_method:ident) => {
        impl ops::$trait<i64> for Ex {
            type Output = Ex;
            fn $method(self, rhs: i64) -> Ex {
                let mut inner = self.inner.write();
                let rhs_id = inner.arena.int(rhs);
                let id = inner.arena.$arena_method(self.id, rhs_id);
                drop(inner);
                self.wrap(id)
            }
        }
        impl ops::$trait<i64> for &Ex {
            type Output = Ex;
            fn $method(self, rhs: i64) -> Ex {
                let mut inner = self.inner.write();
                let rhs_id = inner.arena.int(rhs);
                let id = inner.arena.$arena_method(self.id, rhs_id);
                drop(inner);
                self.wrap(id)
            }
        }
        impl ops::$trait<Ex> for i64 {
            type Output = Ex;
            fn $method(self, rhs: Ex) -> Ex {
                let mut inner = rhs.inner.write();
                let lhs_id = inner.arena.int(self);
                let id = inner.arena.$arena_method(lhs_id, rhs.id);
                drop(inner);
                rhs.wrap(id)
            }
        }
        impl ops::$trait<&Ex> for i64 {
            type Output = Ex;
            fn $method(self, rhs: &Ex) -> Ex {
                let mut inner = rhs.inner.write();
                let lhs_id = inner.arena.int(self);
                let id = inner.arena.$arena_method(lhs_id, rhs.id);
                drop(inner);
                rhs.wrap(id)
            }
        }
    };
}

impl_binary_binop_i64!(Sub, sub, sub);
impl_binary_binop_i64!(Div, div, div);
