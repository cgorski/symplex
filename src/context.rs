//! The [`Context`] is the user-facing entry point for symplex.
//!
//! It owns an expression arena and provides methods for creating
//! symbolic expressions. Expressions ([`Ex`](crate::expr::Ex)) store
//! a reference-counted handle to the arena, so operators work without
//! any special scoping.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use parking_lot::RwLock;

use crate::arena::Arena;
use crate::assumptions::{Assumption, AssumptionCache, Assumptions, Props};
use crate::config::EvalConfig;
use crate::node::CtxId;

/// Global counter for unique context IDs.
static NEXT_CTX_ID: AtomicU32 = AtomicU32::new(0);

/// The user-facing entry point for symplex.
///
/// A `Context` owns an expression arena and provides methods to create
/// symbolic expressions. It is cheaply cloneable — clones share the
/// same underlying arena.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
///
/// let ctx = Context::new();
/// let x = ctx.symbol("x");
/// let expr = &x + 1;
/// assert_eq!(format!("{expr}"), "1 + x");
/// ```
#[derive(Clone)]
pub struct Context {
    pub(crate) id: CtxId,
    pub(crate) arena: Arc<RwLock<Arena>>,
    pub(crate) assumptions: Arc<RwLock<AssumptionCache>>,
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
            arena: Arc::new(RwLock::new(Arena::with_config(config))),
            assumptions: Arc::new(RwLock::new(AssumptionCache::new())),
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
        let id = self.arena.write().symbol(name);
        crate::expr::Ex {
            ctx_id: self.id,
            arena: self.arena.clone(),
            id,
        }
    }

    /// Create a symbol with mathematical assumptions.
    pub fn symbol_with(&self, name: &str, assumptions: &[Assumption]) -> crate::expr::Ex {
        assert!(!name.is_empty(), "symbol name cannot be empty");
        let mut arena = self.arena.write();
        let sym_id = arena.symbols.intern(name);
        let expr_id = arena.intern(crate::node::ExprNode::Symbol(sym_id));

        // Build assumption set from the provided assumptions.
        let mut a = Assumptions {
            known_true: Props::empty(),
            known_false: Props::empty(),
        };
        for assumption in assumptions {
            let (prop, value) = assumption.to_prop_value();
            if value {
                a.assert_true(prop);
            } else {
                a.assert_false(prop);
            }
        }

        // Store on the symbol table.
        arena.set_symbol_assumptions(sym_id, a);

        // Also cache on the expression.
        drop(arena);
        self.assumptions.write().set_symbol_assumptions(expr_id, a);

        crate::expr::Ex {
            ctx_id: self.id,
            arena: self.arena.clone(),
            id: expr_id,
        }
    }

    /// Query a mathematical property of an expression.
    pub fn query(&self, ex: &crate::expr::Ex, prop: Props) -> Option<bool> {
        let arena = self.arena.read();
        self.assumptions.write().query(&arena, ex.id, prop)
    }

    /// Creates an integer expression.
    pub fn int(&self, n: i64) -> crate::expr::Ex {
        let id = self.arena.write().int(n);
        crate::expr::Ex {
            ctx_id: self.id,
            arena: self.arena.clone(),
            id,
        }
    }

    /// Creates a rational expression `p/q`.
    pub fn rational(&self, p: i64, q: i64) -> crate::expr::Ex {
        let id = self.arena.write().rational(p, q);
        crate::expr::Ex {
            ctx_id: self.id,
            arena: self.arena.clone(),
            id,
        }
    }

    // ── Constants ──────────────────────────────────────────────────────

    /// The constant π.
    pub fn pi(&self) -> crate::expr::Ex {
        let id = self.arena.read().pi;
        crate::expr::Ex {
            ctx_id: self.id,
            arena: self.arena.clone(),
            id,
        }
    }

    /// Euler's number *e*.
    pub fn e(&self) -> crate::expr::Ex {
        let id = self.arena.read().e_const;
        crate::expr::Ex {
            ctx_id: self.id,
            arena: self.arena.clone(),
            id,
        }
    }

    /// The imaginary unit *i*.
    pub fn i_unit(&self) -> crate::expr::Ex {
        let id = self.arena.read().i_unit;
        crate::expr::Ex {
            ctx_id: self.id,
            arena: self.arena.clone(),
            id,
        }
    }

    /// Positive infinity.
    pub fn infinity(&self) -> crate::expr::Ex {
        let id = self.arena.read().infinity;
        crate::expr::Ex {
            ctx_id: self.id,
            arena: self.arena.clone(),
            id,
        }
    }

    /// Not-a-number.
    pub fn nan(&self) -> crate::expr::Ex {
        let id = self.arena.read().nan;
        crate::expr::Ex {
            ctx_id: self.id,
            arena: self.arena.clone(),
            id,
        }
    }

    // ── Display ────────────────────────────────────────────────────────

    /// Format an expression as a string.
    pub fn display(&self, ex: &crate::expr::Ex) -> String {
        let arena = self.arena.read();
        arena.display(ex.id).to_string()
    }

    // ── Arena info ─────────────────────────────────────────────────────

    /// Number of interned expression nodes.
    pub fn node_count(&self) -> usize {
        self.arena.read().node_count()
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
            .field("node_count", &self.arena.read().node_count())
            .finish()
    }
}
