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
//! assert_eq!(format!("{expr}"), "1 + x**2 + 2*x");
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
//! assert_eq!(format!("{expr}"), "sin(x**2)");
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
    pub fn pow(&self, exp: &Ex) -> Ex {
        let id = self.inner.write().arena.pow(self.id, exp.id);
        self.wrap(id)
    }

    /// Raise to an integer power: `self ^ n`.
    pub fn powi(&self, n: i64) -> Ex {
        let mut inner = self.inner.write();
        let exp = inner.arena.int(n);
        let id = inner.arena.pow(self.id, exp);
        drop(inner);
        self.wrap(id)
    }

    /// Sine: `sin(self)`.
    pub fn sin(&self) -> Ex {
        let id = self.inner.write().arena.sin(self.id);
        self.wrap(id)
    }

    /// Cosine: `cos(self)`.
    pub fn cos(&self) -> Ex {
        let id = self.inner.write().arena.cos(self.id);
        self.wrap(id)
    }

    /// Tangent: `tan(self)`.
    pub fn tan(&self) -> Ex {
        let id = self.inner.write().arena.tan(self.id);
        self.wrap(id)
    }

    /// Natural exponential: `e^self`.
    pub fn exp_fn(&self) -> Ex {
        let id = self.inner.write().arena.exp_fn(self.id);
        self.wrap(id)
    }

    /// Natural logarithm: `ln(self)`.
    pub fn ln(&self) -> Ex {
        let id = self.inner.write().arena.ln(self.id);
        self.wrap(id)
    }

    /// Principal square root: `√self`.
    pub fn sqrt(&self) -> Ex {
        let id = self.inner.write().arena.sqrt(self.id);
        self.wrap(id)
    }

    /// Absolute value (or complex modulus): `|self|`.
    pub fn abs(&self) -> Ex {
        let id = self.inner.write().arena.abs(self.id);
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

    // ── Future stubs ───────────────────────────────────────────────
    // These will be filled in by later stages.

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
    /// assert_eq!(format!("{deriv}"), "3*x**2");
    /// ```
    pub fn diff(&self, var: &Ex) -> Ex {
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
    pub fn subs(&self, old: &Ex, new: &Ex) -> Ex {
        let id = self
            .inner
            .write()
            .arena
            .subs_structural(self.id, old.id, new.id);
        self.wrap(id)
    }

    /// Simultaneous substitution of multiple `(old, new)` pairs.
    ///
    /// All replacements happen "at once" — earlier substitutions do
    /// not affect later ones.
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
    /// assert_eq!(format!("{}", expr.expand()), "1 + x**2 + 2*x");
    /// ```
    pub fn expand(&self) -> Ex {
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
    pub fn simplify(&self) -> Ex {
        let (result, _steps) = self.simplify_trace();
        result
    }

    /// Like [`simplify`](Ex::simplify), but also returns a trace of
    /// which rules fired and what they changed.
    ///
    /// Each [`Step`](crate::pattern::Step) records the rule name, the
    /// sub-expression before, and the sub-expression after.
    pub fn simplify_trace(&self) -> (Ex, Vec<crate::pattern::Step>) {
        let mut inner = self.inner.write();
        let rules = crate::pattern::basic_rules(&mut inner.arena);
        let (result_id, steps) = crate::pattern::apply_rules(&mut inner.arena, self.id, &rules);
        drop(inner);
        (self.wrap(result_id), steps)
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
    pub fn eval(&self) -> Ex {
        let id = self.inner.write().arena.eval_expr(self.id);
        self.wrap(id)
    }

    /// Numeric floating-point evaluation to the given number of decimal
    /// digits.
    ///
    /// Returns `Err(SymplexError::NotImplemented)` until Stage 8.
    pub fn evalf(&self, _digits: u32) -> Result<Ex, SymplexError> {
        Err(SymplexError::NotImplemented(
            "evalf: numeric evaluation not yet implemented".into(),
        ))
    }

    /// Mathematical equality: attempts to determine if `self - other == 0`.
    ///
    /// Returns `Some(true)` if provably equal, `Some(false)` if provably
    /// not equal, or `None` if unknown.
    pub fn equals(&self, other: &Ex) -> Option<bool> {
        // Structural equality is a fast path.
        if self.id == other.id && self.ctx_id == other.ctx_id {
            return Some(true);
        }
        // Full mathematical equality requires the subtraction to be zero.
        // For now, we can only check structurally.
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

// ── i64 Sub ────────────────────────────────────────────────────────────

impl ops::Sub<i64> for Ex {
    type Output = Ex;
    fn sub(self, rhs: i64) -> Ex {
        let mut inner = self.inner.write();
        let rhs_id = inner.arena.int(rhs);
        let id = inner.arena.sub(self.id, rhs_id);
        drop(inner);
        self.wrap(id)
    }
}

impl ops::Sub<i64> for &Ex {
    type Output = Ex;
    fn sub(self, rhs: i64) -> Ex {
        let mut inner = self.inner.write();
        let rhs_id = inner.arena.int(rhs);
        let id = inner.arena.sub(self.id, rhs_id);
        drop(inner);
        self.wrap(id)
    }
}

impl ops::Sub<Ex> for i64 {
    type Output = Ex;
    fn sub(self, rhs: Ex) -> Ex {
        let mut inner = rhs.inner.write();
        let lhs_id = inner.arena.int(self);
        let id = inner.arena.sub(lhs_id, rhs.id);
        drop(inner);
        rhs.wrap(id)
    }
}

impl ops::Sub<&Ex> for i64 {
    type Output = Ex;
    fn sub(self, rhs: &Ex) -> Ex {
        let mut inner = rhs.inner.write();
        let lhs_id = inner.arena.int(self);
        let id = inner.arena.sub(lhs_id, rhs.id);
        drop(inner);
        rhs.wrap(id)
    }
}

// ── i64 Div ────────────────────────────────────────────────────────────

impl ops::Div<i64> for Ex {
    type Output = Ex;
    fn div(self, rhs: i64) -> Ex {
        let mut inner = self.inner.write();
        let rhs_id = inner.arena.int(rhs);
        let id = inner.arena.div(self.id, rhs_id);
        drop(inner);
        self.wrap(id)
    }
}

impl ops::Div<i64> for &Ex {
    type Output = Ex;
    fn div(self, rhs: i64) -> Ex {
        let mut inner = self.inner.write();
        let rhs_id = inner.arena.int(rhs);
        let id = inner.arena.div(self.id, rhs_id);
        drop(inner);
        self.wrap(id)
    }
}
