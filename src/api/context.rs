//! The [`Context`] is the user-facing entry point for symplex.
//!
//! It owns an expression arena and an assumption cache, wrapped in a
//! single [`RwLock`] for thread-safe access. Expressions ([`Ex`](crate::api::expr::Ex))
//! store a reference-counted handle to this inner state, so operators
//! and queries work without any special scoping.
//!
//! # Locking architecture
//!
//! ```text
//! Arc<RwLock<ContextInner>>
//!   ├── arena: Arena                       (protected by outer RwLock)
//!   └── assumptions: Mutex<AssumptionCache> (interior lock for cache)
//! ```
//!
//! - **Display / structural checks:** acquire outer `read()` — concurrent.
//! - **Expression construction:** acquire outer `write()` — exclusive.
//! - **Assumption queries:** acquire outer `read()` + inner `assumptions.lock()`
//!   — concurrent arena reads, serialized cache writes.
//!
//! Lock ordering is inherent in the nesting: outer first, inner second.
//! Deadlock is structurally impossible.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use parking_lot::{Mutex, RwLock};

use crate::base::arena::Arena;
use crate::base::assumptions::{Assumption, AssumptionCache, Props};
use crate::base::config::EvalConfig;
use crate::base::node::{CtxId, ExprId};

/// Global counter for unique context IDs.
static NEXT_CTX_ID: AtomicU32 = AtomicU32::new(0);

/// The shared inner state of a [`Context`].
///
/// Protected by a single [`RwLock`].  The [`AssumptionCache`] has its own
/// interior [`Mutex`] so that assumption queries (which mutate the cache)
/// can proceed under a shared read lock on the arena.
pub(crate) struct ContextInner {
    /// The expression arena — nodes, numbers, sort keys, symbols.
    pub(crate) arena: Arena,
    /// Cached assumption computations.  Interior `Mutex` because
    /// queries need `&mut` access for caching, but only need `&Arena`
    /// (shared read) for the actual computation.
    pub(crate) assumptions: Mutex<AssumptionCache>,
}

/// The user-facing entry point for symplex.
///
/// A `Context` owns an expression arena and an assumption cache.
/// It is cheaply cloneable — clones share the same underlying state.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
///
/// let ctx = Context::new();
/// let x = ctx.symbol("x");
/// let expr = &x + 1;
/// assert_eq!(format!("{expr}"), "x + 1");
/// ```
#[derive(Clone)]
pub struct Context {
    pub(crate) id: CtxId,
    pub(crate) inner: Arc<RwLock<ContextInner>>,
}

impl Context {
    /// Creates a new context with default configuration.
    pub fn new() -> Self {
        Self::with_config(EvalConfig::default())
    }

    /// Creates a new context with custom evaluation configuration.
    pub fn with_config(config: EvalConfig) -> Self {
        let id = CtxId(NEXT_CTX_ID.fetch_add(1, Ordering::Relaxed));
        Context {
            id,
            inner: Arc::new(RwLock::new(ContextInner {
                arena: Arena::with_config(config),
                assumptions: Mutex::new(AssumptionCache::new()),
            })),
        }
    }

    /// Helper — wrap an [`ExprId`] into a user-facing [`Ex`] handle.
    #[inline]
    fn make_ex(&self, id: ExprId) -> crate::api::expr::Ex {
        crate::api::expr::Ex::from_raw_parts(self.id, Arc::clone(&self.inner), id)
    }

    /// Helper — wrap an [`ExprId`] into a user-facing [`SetEx`] handle.
    #[inline]
    fn make_set_ex(&self, id: ExprId) -> crate::api::expr::SetEx {
        crate::api::expr::SetEx::from_raw_parts(self.id, Arc::clone(&self.inner), id)
    }

    // ── Atom construction ──────────────────────────────────────────────

    /// Creates a symbolic variable.
    ///
    /// # Panics
    ///
    /// Panics if `name` is empty.
    pub fn symbol(&self, name: &str) -> crate::api::expr::Ex {
        assert!(!name.is_empty(), "symbol name cannot be empty");
        let id = self.inner.write().arena.symbol(name);
        self.make_ex(id)
    }

    /// Create a symbolic variable (alias for [`symbol`](Context::symbol)).
    pub fn var(&self, name: &str) -> crate::api::expr::Ex {
        self.symbol(name)
    }

    /// Parse a mathematical expression string in this context.
    ///
    /// All symbols created during parsing belong to this context,
    /// so the result can be freely combined with other expressions
    /// from the same context.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let expr = ctx.parse("x^2 + 1").unwrap();
    /// assert!(format!("{expr}").contains("x"));
    /// ```
    pub fn parse(
        &self,
        input: &str,
    ) -> Result<crate::api::expr::Ex, crate::base::errors::SymplexError> {
        crate::output::parse::parse(self, input).map_err(|e| {
            crate::base::errors::SymplexError::ComputationFailed {
                operation: "parse",
                reason: e.to_string(),
            }
        })
    }

    /// Create a symbol with mathematical assumptions.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let t = ctx.symbol_with("t", &[Assumption::Positive, Assumption::Real]);
    /// assert_eq!(ctx.query(&t, Props::POSITIVE), Some(true));
    /// assert_eq!(ctx.query(&t, Props::REAL), Some(true));
    /// // Inferred by forward-chaining:
    /// assert_eq!(ctx.query(&t, Props::COMPLEX), Some(true));
    /// ```
    pub fn symbol_with(&self, name: &str, assumptions: &[Assumption]) -> crate::api::expr::Ex {
        assert!(!name.is_empty(), "symbol name cannot be empty");

        let mut inner = self.inner.write();
        let sym_id = inner.arena.symbols.intern(name);
        let expr_id = inner
            .arena
            .intern(crate::base::node::ExprNode::Symbol(sym_id));

        // Build assumption set from the provided assumptions.
        let mut a = crate::base::assumptions::Assumptions::default();
        for assumption in assumptions {
            let (prop, value) = assumption.to_prop_value();
            if value {
                a.assert_true(prop);
            } else {
                a.assert_false(prop);
            }
        }

        // Store on the symbol table.
        inner.arena.set_symbol_assumptions(sym_id, a);

        // Also cache on the expression in the assumption cache.
        inner.assumptions.lock().set_symbol_assumptions(expr_id, a);

        drop(inner);
        self.make_ex(expr_id)
    }

    /// Query a mathematical property of an expression.
    ///
    /// Returns `Some(true)` if the property is provably true,
    /// `Some(false)` if provably false, or `None` if unknown.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let one = ctx.int(1);
    /// assert_eq!(ctx.query(&one, Props::POSITIVE), Some(true));
    /// assert_eq!(ctx.query(&one, Props::INTEGER), Some(true));
    /// ```
    pub fn query(&self, ex: &crate::api::expr::Ex, prop: Props) -> Option<bool> {
        let inner = self.inner.read();
        inner
            .assumptions
            .lock()
            .query(&inner.arena, ex.raw_id(), prop)
    }

    /// Creates an integer expression.
    pub fn int(&self, n: i64) -> crate::api::expr::Ex {
        let id = self.inner.write().arena.int(n);
        self.make_ex(id)
    }

    /// The additive identity (0) in this context.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let z = ctx.zero();
    /// assert_eq!(format!("{z}"), "0");
    /// assert!(z.is_zero_structural());
    /// ```
    #[must_use]
    pub fn zero(&self) -> crate::api::expr::Ex {
        self.int(0)
    }

    /// The multiplicative identity (1) in this context.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let o = ctx.one();
    /// assert_eq!(format!("{o}"), "1");
    /// assert!(o.is_one_structural());
    /// ```
    #[must_use]
    pub fn one(&self) -> crate::api::expr::Ex {
        self.int(1)
    }

    /// Creates a rational expression `p/q`.
    pub fn rational(&self, p: i64, q: i64) -> crate::api::expr::Ex {
        let id = self.inner.write().arena.rational(p, q);
        self.make_ex(id)
    }

    // ── Constants ──────────────────────────────────────────────────────

    /// The constant π.
    pub fn pi(&self) -> crate::api::expr::Ex {
        let id = self.inner.read().arena.pi;
        self.make_ex(id)
    }

    /// Euler's number *e*.
    pub fn e(&self) -> crate::api::expr::Ex {
        let id = self.inner.read().arena.e_const;
        self.make_ex(id)
    }

    /// The imaginary unit *i*.
    pub fn i_unit(&self) -> crate::api::expr::Ex {
        let id = self.inner.read().arena.i_unit;
        self.make_ex(id)
    }

    /// Create a named physical constant with a known exact value.
    ///
    /// The constant displays as `name` but evaluates numerically to `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let c = ctx.physical_constant("c", ctx.int(299_792_458));
    /// assert_eq!(format!("{c}"), "c");
    /// ```
    pub fn physical_constant(
        &self,
        name: &str,
        value: crate::api::expr::Ex,
    ) -> crate::api::expr::Ex {
        let val_id = value.raw_id();
        let id = self.inner.write().arena.physical_constant(name, val_id);
        self.make_ex(id)
    }

    /// Positive infinity.
    pub fn infinity(&self) -> crate::api::expr::Ex {
        let id = self.inner.read().arena.infinity;
        self.make_ex(id)
    }

    /// Negative infinity (-∞).
    pub fn neg_infinity(&self) -> crate::api::expr::Ex {
        let inner = self.inner.read();
        let id = inner.arena.neg_infinity;
        drop(inner);
        self.make_ex(id)
    }

    /// Not-a-number.
    pub fn nan(&self) -> crate::api::expr::Ex {
        let id = self.inner.read().arena.nan;
        self.make_ex(id)
    }

    // ── Set construction ───────────────────────────────────────────────

    /// The empty set ∅.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let e = ctx.empty_set();
    /// assert_eq!(format!("{e}"), "EmptySet");
    /// ```
    pub fn empty_set(&self) -> crate::api::expr::SetEx {
        let id = self.inner.read().arena.empty_set;
        self.make_set_ex(id)
    }

    /// The universal set (all values).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let u = ctx.universal_set();
    /// assert_eq!(format!("{u}"), "UniversalSet");
    /// ```
    pub fn universal_set(&self) -> crate::api::expr::SetEx {
        let id = self.inner.read().arena.universal_set;
        self.make_set_ex(id)
    }

    /// The real number line: `(-∞, ∞)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let r = ctx.reals();
    /// let s = format!("{r}");
    /// assert!(s.contains("-oo") && s.contains("oo"), "reals: {s}");
    /// ```
    pub fn reals(&self) -> crate::api::expr::SetEx {
        let mut inner = self.inner.write();
        let neg_inf = inner.arena.neg_infinity;
        let inf = inner.arena.infinity;
        let id = inner
            .arena
            .interval(neg_inf, inf, crate::base::node::INTERVAL_BOTH_OPEN);
        drop(inner);
        self.make_set_ex(id)
    }

    /// Create an interval with explicit open/closed flags.
    ///
    /// `left_open = true` means the left endpoint is excluded (open bracket).
    /// `right_open = true` means the right endpoint is excluded (open bracket).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// // Closed interval [0, 1]
    /// let i = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    /// let s = format!("{i}");
    /// assert!(s.contains("[") && s.contains("]"), "closed interval: {s}");
    ///
    /// // Open interval (0, 1)
    /// let i = ctx.interval(&ctx.int(0), &ctx.int(1), true, true);
    /// let s = format!("{i}");
    /// assert!(s.contains("(") && s.contains(")"), "open interval: {s}");
    /// ```
    pub fn interval(
        &self,
        start: &crate::api::expr::Ex,
        end: &crate::api::expr::Ex,
        left_open: bool,
        right_open: bool,
    ) -> crate::api::expr::SetEx {
        let mut flags: u8 = 0;
        if left_open {
            flags |= crate::base::node::INTERVAL_LEFT_OPEN;
        }
        if right_open {
            flags |= crate::base::node::INTERVAL_RIGHT_OPEN;
        }
        let id = self
            .inner
            .write()
            .arena
            .interval(start.raw_id(), end.raw_id(), flags);
        self.make_set_ex(id)
    }

    /// Create a finite set `{elements[0], elements[1], …}`.
    ///
    /// Elements are sorted and deduplicated.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let s = ctx.finite_set(&[ctx.int(3), ctx.int(1), ctx.int(2)]);
    /// let display = format!("{s}");
    /// assert!(display.contains("{") && display.contains("}"), "finite set: {display}");
    /// ```
    pub fn finite_set(&self, elements: &[crate::api::expr::Ex]) -> crate::api::expr::SetEx {
        let ids: Vec<ExprId> = elements.iter().map(|e| e.raw_id()).collect();
        let id = self.inner.write().arena.finite_set(&ids);
        self.make_set_ex(id)
    }

    // ── Arena access ───────────────────────────────────────────────────

    /// Provides mutable access to the expression arena.
    ///
    /// The closure receives `&mut Arena` and can call any arena method
    /// (e.g., `arena.add()`, `arena.sin()`, `arena.symbol()`).
    ///
    /// This is primarily used by the [`rule!`](crate::rule) macro to
    /// build pattern expressions directly in the arena.  Most users
    /// should prefer the higher-level `Ex` methods instead.
    ///
    /// # Panics
    ///
    /// **Deadlock warning:** The write lock on the context is held for the
    /// entire duration of the closure. If the closure captures and uses a
    /// `Context` or `Ex` handle from the **same** context (calling methods
    /// like `.sin()`, `.expand()`, `format!()`, or any operation that
    /// acquires the lock), the thread will deadlock.
    ///
    /// **Safe:** Only call `Arena` methods inside the closure.
    ///
    /// **Unsafe (deadlocks):** Do NOT call `Context` methods, `Ex` methods,
    /// or `format!("{}", some_ex)` inside the closure.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let result = ctx.with_arena_mut(|arena| {
    ///     let x = arena.symbol("x");
    ///     let two = arena.int(2);
    ///     arena.pow(x, two)
    /// });
    /// ```
    pub fn with_arena_mut<R>(&self, f: impl FnOnce(&mut crate::base::arena::Arena) -> R) -> R {
        let mut guard = self.inner.write();
        f(&mut guard.arena)
    }

    // ── Deserialization ────────────────────────────────────────────────

    /// Convert a serialized [`ExprTree`](crate::output::tree::ExprTree) back into
    /// an expression handle in this context.
    ///
    /// The resulting expression is fully canonicalized.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = &x.powi(2) + 1;
    /// let tree = expr.to_tree();
    /// let back = ctx.from_tree(&tree);
    /// assert_eq!(format!("{back}"), format!("{expr}"));
    /// ```
    pub fn from_tree(&self, tree: &crate::output::tree::ExprTree) -> crate::api::expr::Ex {
        let mut inner = self.inner.write();
        let id = crate::output::tree::tree_to_expr(&mut inner.arena, tree);
        drop(inner);
        self.make_ex(id)
    }

    /// Parse a JSON string into an expression in this context.
    ///
    /// This is a convenience shorthand for deserializing an
    /// [`ExprTree`](crate::output::tree::ExprTree) from JSON and converting it.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the JSON is malformed or doesn't represent a
    /// valid `ExprTree`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let json = r#"{"type":"Symbol","name":"x"}"#;
    /// let expr = ctx.from_json(json).unwrap();
    /// assert_eq!(format!("{expr}"), "x");
    /// ```
    pub fn from_json(&self, json: &str) -> Result<crate::api::expr::Ex, serde_json::Error> {
        let tree: crate::output::tree::ExprTree = serde_json::from_str(json)?;
        Ok(self.from_tree(&tree))
    }

    // ── Linear system solving ──────────────────────────────────────────

    /// Solve a system of linear equations.
    ///
    /// Each equation in `equations` is an expression that equals zero.
    /// `variables` are the symbols to solve for.
    ///
    /// Returns `Some(vec![(var1, val1), (var2, val2), ...])` if a unique
    /// solution exists, or `None` if the system is underdetermined,
    /// overdetermined, or inconsistent.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// // x + y = 3, x - y = 1  →  x = 2, y = 1
    /// let eq1 = &x + &y - 3;
    /// let eq2 = &x - &y - 1;
    /// let solution = ctx.solve_system(&[eq1, eq2], &[x, y]).unwrap();
    /// assert_eq!(solution.len(), 2);
    /// assert_eq!(format!("{}", solution[0].1), "2");
    /// assert_eq!(format!("{}", solution[1].1), "1");
    /// ```
    pub fn solve_system(
        &self,
        equations: &[crate::api::expr::Ex],
        variables: &[crate::api::expr::Ex],
    ) -> Option<Vec<(crate::api::expr::Ex, crate::api::expr::Ex)>> {
        let eq_ids: Vec<crate::base::node::ExprId> = equations.iter().map(|e| e.raw_id()).collect();
        let var_ids: Vec<crate::base::node::ExprId> =
            variables.iter().map(|v| v.raw_id()).collect();

        let mut inner = self.inner.write();
        let result =
            crate::domains::linalg::solve_linear_system(&mut inner.arena, &eq_ids, &var_ids)?;
        drop(inner);

        Some(
            result
                .pairs
                .into_iter()
                .map(|(var_id, val_id)| (self.make_ex(var_id), self.make_ex(val_id)))
                .collect(),
        )
    }

    // ── Arena compaction (generational GC) ─────────────────────────────

    /// Create a new, compacted context containing only the expression
    /// trees reachable from `roots`.
    ///
    /// Returns the new context and the corresponding root expressions
    /// (in the same order as `roots`).  The old context remains valid —
    /// existing expression handles continue to work.
    ///
    /// # Use Case
    ///
    /// After a heavy computation that creates thousands of intermediate
    /// nodes, compact the handful of results into a fresh arena:
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // ... heavy computation creating many intermediates ...
    /// let result = &x + 1;
    /// let (new_ctx, new_exprs) = ctx.compact(&[result]);
    /// // old ctx can be dropped — only the reachable nodes survive
    /// assert_eq!(format!("{}", new_exprs[0]), "x + 1");
    /// ```
    pub fn compact(&self, roots: &[crate::api::expr::Ex]) -> (Context, Vec<crate::api::expr::Ex>) {
        let new_ctx = Context::new();

        if roots.is_empty() {
            tracing::debug!("compact: 0 roots — returning empty context");
            return (new_ctx, Vec::new());
        }

        let src_inner = self.inner.read();
        let mut dst_inner = new_ctx.inner.write();

        let mut map = rustc_hash::FxHashMap::default();
        let mut new_roots = Vec::with_capacity(roots.len());

        for root in roots {
            let new_id = crate::base::compact::transfer_subtree(
                &src_inner.arena,
                &mut dst_inner.arena,
                root.raw_id(),
                &mut map,
            );
            new_roots.push(crate::api::expr::Ex::from_raw_parts(
                new_ctx.id,
                Arc::clone(&new_ctx.inner),
                new_id,
            ));
        }

        tracing::debug!(
            src_nodes = src_inner.arena.node_count(),
            dst_nodes = dst_inner.arena.node_count(),
            roots = roots.len(),
            transferred = map.len(),
            "compact: arena compaction complete",
        );

        drop(dst_inner);
        drop(src_inner);

        (new_ctx, new_roots)
    }

    // ── Arena info ─────────────────────────────────────────────────────

    /// Number of interned expression nodes.
    pub fn node_count(&self) -> usize {
        self.inner.read().arena.node_count()
    }

    /// Compute the fraction of arena nodes reachable from the given root expressions.
    ///
    /// Returns a value between 0.0 and 1.0. A value of 0.3 means 70% of
    /// arena nodes are unreachable (dead) and would be freed by `compact()`.
    pub fn liveness_ratio(&self, roots: &[crate::api::expr::Ex]) -> f64 {
        let root_ids: Vec<crate::base::node::ExprId> = roots.iter().map(|r| r.raw_id()).collect();
        let inner = self.inner.read();
        crate::base::compact::liveness_ratio(&inner.arena, &root_ids)
    }

    /// Heuristic: should the arena be compacted?
    ///
    /// Returns `true` when the arena has grown large (>100K nodes),
    /// has doubled since the last compact, and less than 50% of nodes
    /// are reachable from the given roots.
    pub fn should_compact(&self, roots: &[crate::api::expr::Ex]) -> bool {
        let root_ids: Vec<crate::base::node::ExprId> = roots.iter().map(|r| r.raw_id()).collect();
        let inner = self.inner.read();
        crate::base::compact::should_compact(&inner.arena, &root_ids)
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("id", &self.id)
            .field("node_count", &self.inner.read().arena.node_count())
            .finish()
    }
}
