//! The [`Context`] is the user-facing entry point for symplex.
//!
//! It owns an expression arena and an assumption cache, wrapped in a
//! single [`RwLock`] for thread-safe access. Expressions ([`Ex`](crate::expr::Ex))
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

use crate::arena::Arena;
use crate::assumptions::{Assumption, AssumptionCache, Props};
use crate::config::EvalConfig;
use crate::node::{CtxId, ExprId};

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
    /// Cached simplification rules. Lazily initialized on first
    /// `simplify()` call to avoid rebuilding rules every time.
    pub(crate) cached_rules: Option<Vec<crate::pattern::Rule>>,
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
                cached_rules: None,
            })),
        }
    }

    /// Helper — wrap an [`ExprId`] into a user-facing [`Ex`] handle.
    #[inline]
    fn make_ex(&self, id: ExprId) -> crate::expr::Ex {
        crate::expr::Ex {
            ctx_id: self.id,
            inner: Arc::clone(&self.inner),
            id,
            _sort: std::marker::PhantomData,
        }
    }

    // ── Atom construction ──────────────────────────────────────────────

    /// Creates a symbolic variable.
    ///
    /// # Panics
    ///
    /// Panics if `name` is empty.
    pub fn symbol(&self, name: &str) -> crate::expr::Ex {
        assert!(!name.is_empty(), "symbol name cannot be empty");
        let id = self.inner.write().arena.symbol(name);
        self.make_ex(id)
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
    pub fn symbol_with(&self, name: &str, assumptions: &[Assumption]) -> crate::expr::Ex {
        assert!(!name.is_empty(), "symbol name cannot be empty");

        let mut inner = self.inner.write();
        let sym_id = inner.arena.symbols.intern(name);
        let expr_id = inner.arena.intern(crate::node::ExprNode::Symbol(sym_id));

        // Build assumption set from the provided assumptions.
        let mut a = crate::assumptions::Assumptions::default();
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
    pub fn query(&self, ex: &crate::expr::Ex, prop: Props) -> Option<bool> {
        let inner = self.inner.read();
        inner.assumptions.lock().query(&inner.arena, ex.id, prop)
    }

    /// Creates an integer expression.
    pub fn int(&self, n: i64) -> crate::expr::Ex {
        let id = self.inner.write().arena.int(n);
        self.make_ex(id)
    }

    /// Creates a rational expression `p/q`.
    pub fn rational(&self, p: i64, q: i64) -> crate::expr::Ex {
        let id = self.inner.write().arena.rational(p, q);
        self.make_ex(id)
    }

    // ── Constants ──────────────────────────────────────────────────────

    /// The constant π.
    pub fn pi(&self) -> crate::expr::Ex {
        let id = self.inner.read().arena.pi;
        self.make_ex(id)
    }

    /// Euler's number *e*.
    pub fn e(&self) -> crate::expr::Ex {
        let id = self.inner.read().arena.e_const;
        self.make_ex(id)
    }

    /// The imaginary unit *i*.
    pub fn i_unit(&self) -> crate::expr::Ex {
        let id = self.inner.read().arena.i_unit;
        self.make_ex(id)
    }

    /// Positive infinity.
    pub fn infinity(&self) -> crate::expr::Ex {
        let id = self.inner.read().arena.infinity;
        self.make_ex(id)
    }

    /// Negative infinity (-∞).
    pub fn neg_infinity(&self) -> crate::expr::Ex {
        let inner = self.inner.read();
        crate::expr::Ex {
            ctx_id: self.id,
            inner: Arc::clone(&self.inner),
            id: inner.arena.neg_infinity,
            _sort: std::marker::PhantomData,
        }
    }

    /// Not-a-number.
    pub fn nan(&self) -> crate::expr::Ex {
        let id = self.inner.read().arena.nan;
        self.make_ex(id)
    }

    // ── Arena access ───────────────────────────────────────────────────

    /// Run a closure with mutable access to the underlying arena.
    ///
    /// This is primarily used by the [`rule!`](crate::rule) macro to
    /// build pattern expressions directly in the arena.  Most users
    /// should prefer the higher-level `Ex` methods instead.
    ///
    /// The closure receives `&mut Arena` and can call any arena method.
    /// The write lock is held for the duration of the closure.
    pub fn with_arena_mut<R>(&self, f: impl FnOnce(&mut crate::arena::Arena) -> R) -> R {
        let mut guard = self.inner.write();
        f(&mut guard.arena)
    }

    // ── Deserialization ────────────────────────────────────────────────

    /// Convert a serialized [`ExprTree`](crate::tree::ExprTree) back into
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
    pub fn from_tree(&self, tree: &crate::tree::ExprTree) -> crate::expr::Ex {
        let mut inner = self.inner.write();
        let id = crate::tree::tree_to_expr(&mut inner.arena, tree);
        drop(inner);
        self.make_ex(id)
    }

    /// Parse a JSON string into an expression in this context.
    ///
    /// This is a convenience shorthand for deserializing an
    /// [`ExprTree`](crate::tree::ExprTree) from JSON and converting it.
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
    pub fn from_json(&self, json: &str) -> Result<crate::expr::Ex, serde_json::Error> {
        let tree: crate::tree::ExprTree = serde_json::from_str(json)?;
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
        equations: &[crate::expr::Ex],
        variables: &[crate::expr::Ex],
    ) -> Option<Vec<(crate::expr::Ex, crate::expr::Ex)>> {
        let eq_ids: Vec<crate::node::ExprId> = equations.iter().map(|e| e.id).collect();
        let var_ids: Vec<crate::node::ExprId> = variables.iter().map(|v| v.id).collect();

        let mut inner = self.inner.write();
        let result = crate::linalg::solve_linear_system(&mut inner.arena, &eq_ids, &var_ids)?;
        drop(inner);

        Some(
            result
                .pairs
                .into_iter()
                .map(|(var_id, val_id)| (self.make_ex(var_id), self.make_ex(val_id)))
                .collect(),
        )
    }

    // ── Arena info ─────────────────────────────────────────────────────

    /// Number of interned expression nodes.
    pub fn node_count(&self) -> usize {
        self.inner.read().arena.node_count()
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
