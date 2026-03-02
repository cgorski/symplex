//! User-facing expression handle.
//!
//! [`Ex`] is a lightweight handle to a symbolic expression. It holds a
//! reference-counted pointer to the arena and an expression ID. Clone
//! is cheap (~5 ns, just an `Arc` clone + `u32` copy).
//!
//! `Ex` is **not** `Copy` because it contains an `Arc`. This is a
//! deliberate trade-off: storing the arena pointer means every method
//! (`pow`, `sin`, operators, `Display`) works without requiring the
//! a reference-counted handle to the arena, so operators work anywhere.
//!
//! To reduce clone noise, all binary operators are implemented for every
//! combination of `Ex` and `&Ex`, and for `i64` on both sides.

use std::fmt;
use std::hash;
use std::ops;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::arena::Arena;
use crate::display::fmt_expr;
use crate::node::{CtxId, ExprId};

/// A symbolic expression handle.
///
/// `Ex` wraps an [`ExprId`] together with a shared reference to the
/// [`Arena`] that owns it. It is cheaply cloneable and implements the
/// standard arithmetic operators (`+`, `-`, `*`, `/`, unary `-`).
///
/// # Operators
///
/// All binary operators accept any combination of owned and borrowed
/// `Ex` values, plus `i64` on either side:
///
/// ```
/// use symplex::context::Context;
///
/// let ctx = Context::new();
/// let x = ctx.symbol("x");
/// let expr = &x * &x + &x * 2 + 1;
/// assert_eq!(format!("{expr}"), "1 + x**2 + 2*x");
/// ```
///
/// # Math Functions
///
/// Transcendental and algebraic functions are available as methods:
///
/// ```
/// use symplex::context::Context;
///
/// let ctx = Context::new();
/// let x = ctx.symbol("x");
/// let expr = x.sin();
/// assert_eq!(format!("{expr}"), "sin(x)");
/// ```
#[derive(Clone)]
pub struct Ex {
    pub(crate) ctx_id: CtxId,
    pub(crate) arena: Arc<RwLock<Arena>>,
    pub(crate) id: ExprId,
}

impl Ex {
    /// Helper — build a new `Ex` from the same context with a different id.
    #[inline]
    fn wrap(&self, id: ExprId) -> Ex {
        Ex {
            ctx_id: self.ctx_id,
            arena: Arc::clone(&self.arena),
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
        let id = self.arena.write().pow(self.id, exp.id);
        self.wrap(id)
    }

    /// Raise to an integer power: `self ^ n`.
    pub fn powi(&self, n: i64) -> Ex {
        let mut arena = self.arena.write();
        let exp = arena.int(n);
        let id = arena.pow(self.id, exp);
        drop(arena);
        self.wrap(id)
    }

    /// Sine: `sin(self)`.
    pub fn sin(&self) -> Ex {
        let id = self.arena.write().sin(self.id);
        self.wrap(id)
    }

    /// Cosine: `cos(self)`.
    pub fn cos(&self) -> Ex {
        let id = self.arena.write().cos(self.id);
        self.wrap(id)
    }

    /// Tangent: `tan(self)`.
    pub fn tan(&self) -> Ex {
        let id = self.arena.write().tan(self.id);
        self.wrap(id)
    }

    /// Natural exponential: `e^self`.
    pub fn exp_fn(&self) -> Ex {
        let id = self.arena.write().exp_fn(self.id);
        self.wrap(id)
    }

    /// Natural logarithm: `ln(self)`.
    pub fn ln(&self) -> Ex {
        let id = self.arena.write().ln(self.id);
        self.wrap(id)
    }

    /// Principal square root: `√self`.
    pub fn sqrt(&self) -> Ex {
        let id = self.arena.write().sqrt(self.id);
        self.wrap(id)
    }

    /// Absolute value (or complex modulus): `|self|`.
    pub fn abs(&self) -> Ex {
        let id = self.arena.write().abs(self.id);
        self.wrap(id)
    }

    // ── Structural predicates ──────────────────────────────────────

    /// Returns `true` if this expression is structurally zero (O(1)).
    pub fn is_zero_structural(&self) -> bool {
        self.arena.read().is_zero_structural(self.id)
    }

    /// Returns `true` if this expression is structurally one (O(1)).
    pub fn is_one_structural(&self) -> bool {
        self.arena.read().is_one_structural(self.id)
    }

    // ── Future stubs ───────────────────────────────────────────────
    // These will be filled in by later stages.

    /// Symbolic differentiation with respect to `var`.
    pub fn diff(&self, _var: &Ex) -> Ex {
        todo!("diff: symbolic differentiation not yet implemented")
    }

    /// Substitute `var` with `replacement` in this expression.
    pub fn subs(&self, _var: &Ex, _replacement: &Ex) -> Ex {
        todo!("subs: symbolic substitution not yet implemented")
    }

    /// Algebraic expansion (distribute products over sums, etc.).
    pub fn expand(&self) -> Ex {
        todo!("expand: algebraic expansion not yet implemented")
    }

    /// Simplification (identity application, trig identities, etc.).
    pub fn simplify(&self) -> Ex {
        todo!("simplify: not yet implemented")
    }

    /// Exact symbolic evaluation (rational arithmetic, known identities).
    pub fn eval(&self) -> Ex {
        todo!("eval: exact evaluation not yet implemented")
    }

    /// Numeric floating-point evaluation.
    pub fn evalf(&self) -> f64 {
        todo!("evalf: numeric evaluation not yet implemented")
    }

    /// Structural equality test (same tree shape, NOT mathematical).
    ///
    /// For mathematical equality checking, use [`equals`](Ex::equals).
    pub fn equals(&self, _other: &Ex) -> bool {
        todo!("equals: mathematical equality not yet implemented")
    }

    /// Returns `true` if this expression is provably positive.
    pub fn is_positive(&self) -> bool {
        todo!("is_positive: assumption system not yet implemented")
    }

    /// Returns `true` if this expression is provably zero
    /// (beyond structural check).
    pub fn is_zero(&self) -> bool {
        todo!("is_zero: deep zero-check not yet implemented")
    }
}

// ── PartialEq / Eq ─────────────────────────────────────────────────────
//
// Two `Ex` values are equal iff they come from the same context and
// refer to the same interned node.  Because the arena deduplicates,
// this is equivalent to structural equality *within* a single context.

impl PartialEq for Ex {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.ctx_id == other.ctx_id && self.id == other.id
    }
}

impl Eq for Ex {}

// ── Hash ────────────────────────────────────────────────────────────────
//
// We hash both `ctx_id` and `id` so that expressions from different
// contexts don't spuriously collide.

impl hash::Hash for Ex {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.ctx_id.hash(state);
        self.id.hash(state);
    }
}

// ── Display ─────────────────────────────────────────────────────────────
//
// We acquire a read lock, then use the arena's display machinery.
// The lock is held for the duration of the `fmt` call.

impl fmt::Display for Ex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let arena = self.arena.read();
        fmt_expr(&arena, f, self.id, 0)
    }
}

impl fmt::Debug for Ex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ex({:?}, {:?})", self.ctx_id, self.id)
    }
}

// ════════════════════════════════════════════════════════════════════════
// Operator implementations
// ════════════════════════════════════════════════════════════════════════
//
// For each binary operator we provide 4 impls (Ex⊕Ex, Ex⊕&Ex,
// &Ex⊕Ex, &Ex⊕&Ex) plus 4 i64 variants (Ex⊕i64, &Ex⊕i64,
// i64⊕Ex, i64⊕&Ex).  A macro handles the repetitive cases.

// ---------------------------------------------------------------------------
// Helper macro for n-ary operators (Add, Mul) that go through
// `arena.$method(&[lhs, rhs])`.
// ---------------------------------------------------------------------------

macro_rules! impl_nary_binop {
    ($Trait:ident, $method:ident, $arena_method:ident) => {
        // Ex op Ex
        impl ops::$Trait<Ex> for Ex {
            type Output = Ex;
            #[inline]
            fn $method(self, rhs: Ex) -> Ex {
                let id = self.arena.write().$arena_method(&[self.id, rhs.id]);
                self.wrap(id)
            }
        }

        // Ex op &Ex
        impl ops::$Trait<&Ex> for Ex {
            type Output = Ex;
            #[inline]
            fn $method(self, rhs: &Ex) -> Ex {
                let id = self.arena.write().$arena_method(&[self.id, rhs.id]);
                self.wrap(id)
            }
        }

        // &Ex op Ex
        impl ops::$Trait<Ex> for &Ex {
            type Output = Ex;
            #[inline]
            fn $method(self, rhs: Ex) -> Ex {
                let id = self.arena.write().$arena_method(&[self.id, rhs.id]);
                self.wrap(id)
            }
        }

        // &Ex op &Ex
        impl ops::$Trait<&Ex> for &Ex {
            type Output = Ex;
            #[inline]
            fn $method(self, rhs: &Ex) -> Ex {
                let id = self.arena.write().$arena_method(&[self.id, rhs.id]);
                self.wrap(id)
            }
        }
    };
}

impl_nary_binop!(Add, add, add);
impl_nary_binop!(Mul, mul, mul);

// ---------------------------------------------------------------------------
// Helper macro for binary operators (Sub, Div) that go through
// `arena.$method(lhs, rhs)`.
// ---------------------------------------------------------------------------

macro_rules! impl_binary_binop {
    ($Trait:ident, $method:ident, $arena_method:ident) => {
        impl ops::$Trait<Ex> for Ex {
            type Output = Ex;
            #[inline]
            fn $method(self, rhs: Ex) -> Ex {
                let id = self.arena.write().$arena_method(self.id, rhs.id);
                self.wrap(id)
            }
        }

        impl ops::$Trait<&Ex> for Ex {
            type Output = Ex;
            #[inline]
            fn $method(self, rhs: &Ex) -> Ex {
                let id = self.arena.write().$arena_method(self.id, rhs.id);
                self.wrap(id)
            }
        }

        impl ops::$Trait<Ex> for &Ex {
            type Output = Ex;
            #[inline]
            fn $method(self, rhs: Ex) -> Ex {
                let id = self.arena.write().$arena_method(self.id, rhs.id);
                self.wrap(id)
            }
        }

        impl ops::$Trait<&Ex> for &Ex {
            type Output = Ex;
            #[inline]
            fn $method(self, rhs: &Ex) -> Ex {
                let id = self.arena.write().$arena_method(self.id, rhs.id);
                self.wrap(id)
            }
        }
    };
}

impl_binary_binop!(Sub, sub, sub);
impl_binary_binop!(Div, div, div);

// ---------------------------------------------------------------------------
// Neg (unary)
// ---------------------------------------------------------------------------

impl ops::Neg for Ex {
    type Output = Ex;
    #[inline]
    fn neg(self) -> Ex {
        let id = self.arena.write().neg(self.id);
        self.wrap(id)
    }
}

impl ops::Neg for &Ex {
    type Output = Ex;
    #[inline]
    fn neg(self) -> Ex {
        let id = self.arena.write().neg(self.id);
        self.wrap(id)
    }
}

// ════════════════════════════════════════════════════════════════════════
// i64 operator implementations
// ════════════════════════════════════════════════════════════════════════

// ---------------------------------------------------------------------------
// Macro for n-ary i64 operators (Add, Mul)
// ---------------------------------------------------------------------------

macro_rules! impl_nary_binop_i64 {
    ($Trait:ident, $method:ident, $arena_method:ident) => {
        // Ex op i64
        impl ops::$Trait<i64> for Ex {
            type Output = Ex;
            #[inline]
            fn $method(self, rhs: i64) -> Ex {
                let mut arena = self.arena.write();
                let rhs_id = arena.int(rhs);
                let id = arena.$arena_method(&[self.id, rhs_id]);
                drop(arena);
                self.wrap(id)
            }
        }

        // &Ex op i64
        impl ops::$Trait<i64> for &Ex {
            type Output = Ex;
            #[inline]
            fn $method(self, rhs: i64) -> Ex {
                let mut arena = self.arena.write();
                let rhs_id = arena.int(rhs);
                let id = arena.$arena_method(&[self.id, rhs_id]);
                drop(arena);
                self.wrap(id)
            }
        }

        // i64 op Ex
        impl ops::$Trait<Ex> for i64 {
            type Output = Ex;
            #[inline]
            fn $method(self, rhs: Ex) -> Ex {
                let mut arena = rhs.arena.write();
                let lhs_id = arena.int(self);
                let id = arena.$arena_method(&[lhs_id, rhs.id]);
                drop(arena);
                rhs.wrap(id)
            }
        }

        // i64 op &Ex
        impl ops::$Trait<&Ex> for i64 {
            type Output = Ex;
            #[inline]
            fn $method(self, rhs: &Ex) -> Ex {
                let mut arena = rhs.arena.write();
                let lhs_id = arena.int(self);
                let id = arena.$arena_method(&[lhs_id, rhs.id]);
                drop(arena);
                rhs.wrap(id)
            }
        }
    };
}

impl_nary_binop_i64!(Add, add, add);
impl_nary_binop_i64!(Mul, mul, mul);

// ---------------------------------------------------------------------------
// Macro for binary i64 operators (Sub, Div)
// ---------------------------------------------------------------------------

macro_rules! impl_binary_binop_i64 {
    ($Trait:ident, $method:ident, $arena_method:ident) => {
        // Ex op i64
        impl ops::$Trait<i64> for Ex {
            type Output = Ex;
            #[inline]
            fn $method(self, rhs: i64) -> Ex {
                let mut arena = self.arena.write();
                let rhs_id = arena.int(rhs);
                let id = arena.$arena_method(self.id, rhs_id);
                drop(arena);
                self.wrap(id)
            }
        }

        // &Ex op i64
        impl ops::$Trait<i64> for &Ex {
            type Output = Ex;
            #[inline]
            fn $method(self, rhs: i64) -> Ex {
                let mut arena = self.arena.write();
                let rhs_id = arena.int(rhs);
                let id = arena.$arena_method(self.id, rhs_id);
                drop(arena);
                self.wrap(id)
            }
        }

        // i64 op Ex
        impl ops::$Trait<Ex> for i64 {
            type Output = Ex;
            #[inline]
            fn $method(self, rhs: Ex) -> Ex {
                let mut arena = rhs.arena.write();
                let lhs_id = arena.int(self);
                let id = arena.$arena_method(lhs_id, rhs.id);
                drop(arena);
                rhs.wrap(id)
            }
        }

        // i64 op &Ex
        impl ops::$Trait<&Ex> for i64 {
            type Output = Ex;
            #[inline]
            fn $method(self, rhs: &Ex) -> Ex {
                let mut arena = rhs.arena.write();
                let lhs_id = arena.int(self);
                let id = arena.$arena_method(lhs_id, rhs.id);
                drop(arena);
                rhs.wrap(id)
            }
        }
    };
}

impl_binary_binop_i64!(Sub, sub, sub);
impl_binary_binop_i64!(Div, div, div);

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use crate::context::Context;

    // ── Construction & identity ─────────────────────────────────────

    #[test]
    fn clone_preserves_identity() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let x2 = x.clone();
        assert_eq!(x, x2);
    }

    #[test]
    fn different_symbols_not_equal() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let y = ctx.symbol("y");
        assert_ne!(x, y);
    }

    #[test]
    fn different_contexts_not_equal() {
        let c1 = Context::new();
        let c2 = Context::new();
        let x1 = c1.symbol("x");
        let x2 = c2.symbol("x");
        // Same name, different context → not equal.
        assert_ne!(x1, x2);
    }

    // ── Debug / Display ─────────────────────────────────────────────

    #[test]
    fn debug_format() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let dbg = format!("{x:?}");
        assert!(dbg.starts_with("Ex("), "got: {dbg}");
    }

    #[test]
    fn display_symbol() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        assert_eq!(format!("{x}"), "x");
    }

    #[test]
    fn display_integer() {
        let ctx = Context::new();
        let n = ctx.int(42);
        assert_eq!(format!("{n}"), "42");
    }

    // ── Arithmetic operators (Ex ⊕ Ex) ──────────────────────────────

    #[test]
    fn add_two_symbols() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let y = ctx.symbol("y");
        let sum = &x + &y;
        let s = format!("{sum}");
        assert!(
            s.contains('x') && s.contains('y') && s.contains('+'),
            "got: {s}"
        );
    }

    #[test]
    fn sub_two_symbols() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let y = ctx.symbol("y");
        let diff = &x - &y;
        let s = format!("{diff}");
        // x - y is canonicalized as x + (-y) = x + (-1)*y
        assert!(s.contains('x') && s.contains('y'), "got: {s}");
    }

    #[test]
    fn mul_two_symbols() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let y = ctx.symbol("y");
        let prod = &x * &y;
        let s = format!("{prod}");
        assert!(
            s.contains('x') && s.contains('y') && s.contains('*'),
            "got: {s}"
        );
    }

    #[test]
    fn div_two_symbols() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let y = ctx.symbol("y");
        let quot = &x / &y;
        let s = format!("{quot}");
        // x / y is canonicalized as x * y^(-1)
        assert!(s.contains('x') && s.contains('y'), "got: {s}");
    }

    #[test]
    fn neg_symbol() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let nx = -&x;
        let s = format!("{nx}");
        assert!(s.contains('-') && s.contains('x'), "got: {s}");
    }

    // ── Arithmetic with i64 ─────────────────────────────────────────

    #[test]
    fn add_i64_rhs() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = &x + 1;
        let s = format!("{expr}");
        assert!(s.contains('1') && s.contains('x'), "got: {s}");
    }

    #[test]
    fn add_i64_lhs() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = 1 + &x;
        let s = format!("{expr}");
        assert!(s.contains('1') && s.contains('x'), "got: {s}");
    }

    #[test]
    fn mul_i64_rhs() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = &x * 3;
        let s = format!("{expr}");
        assert!(s.contains('3') && s.contains('x'), "got: {s}");
    }

    #[test]
    fn mul_i64_lhs() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = 3 * &x;
        let s = format!("{expr}");
        assert!(s.contains('3') && s.contains('x'), "got: {s}");
    }

    #[test]
    fn sub_i64_rhs() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = &x - 5;
        let s = format!("{expr}");
        assert!(s.contains('x'), "got: {s}");
    }

    #[test]
    fn sub_i64_lhs() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = 5 - &x;
        let s = format!("{expr}");
        assert!(s.contains('x') && s.contains('5'), "got: {s}");
    }

    #[test]
    fn div_i64_rhs() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = &x / 2;
        let s = format!("{expr}");
        assert!(s.contains('x'), "got: {s}");
    }

    #[test]
    fn div_i64_lhs() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = 2 / &x;
        let s = format!("{expr}");
        assert!(s.contains('x') && s.contains('2'), "got: {s}");
    }

    // ── Owned-value operator variants ───────────────────────────────

    #[test]
    fn add_owned_owned() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let y = ctx.symbol("y");
        let _sum = x + y; // consumes both
    }

    #[test]
    fn add_owned_ref() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let y = ctx.symbol("y");
        let _sum = x + &y; // consumes x, borrows y
    }

    #[test]
    fn add_ref_owned() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let y = ctx.symbol("y");
        let _sum = &x + y; // borrows x, consumes y
    }

    #[test]
    fn add_i64_owned() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let _expr = x + 1; // consumes x
    }

    #[test]
    fn i64_add_owned() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let _expr = 1 + x; // consumes x
    }

    // ── Math functions ──────────────────────────────────────────────

    #[test]
    fn pow_symbolic() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let two = ctx.int(2);
        let expr = x.pow(&two);
        let s = format!("{expr}");
        assert!(s.contains("x**2"), "got: {s}");
    }

    #[test]
    fn powi() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = x.powi(3);
        let s = format!("{expr}");
        assert!(s.contains("x**3"), "got: {s}");
    }

    #[test]
    fn sin_display() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = x.sin();
        assert_eq!(format!("{expr}"), "sin(x)");
    }

    #[test]
    fn cos_display() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = x.cos();
        assert_eq!(format!("{expr}"), "cos(x)");
    }

    #[test]
    fn tan_display() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = x.tan();
        assert_eq!(format!("{expr}"), "tan(x)");
    }

    #[test]
    fn exp_fn_display() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = x.exp_fn();
        assert_eq!(format!("{expr}"), "exp(x)");
    }

    #[test]
    fn ln_display() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = x.ln();
        assert_eq!(format!("{expr}"), "ln(x)");
    }

    #[test]
    fn sqrt_display() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = x.sqrt();
        assert_eq!(format!("{expr}"), "sqrt(x)");
    }

    #[test]
    fn abs_display() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = x.abs();
        assert_eq!(format!("{expr}"), "abs(x)");
    }

    // ── Structural predicates ───────────────────────────────────────

    #[test]
    fn zero_is_zero() {
        let ctx = Context::new();
        let z = ctx.int(0);
        assert!(z.is_zero_structural());
        assert!(!z.is_one_structural());
    }

    #[test]
    fn one_is_one() {
        let ctx = Context::new();
        let o = ctx.int(1);
        assert!(o.is_one_structural());
        assert!(!o.is_zero_structural());
    }

    #[test]
    fn symbol_is_not_zero() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        assert!(!x.is_zero_structural());
        assert!(!x.is_one_structural());
    }

    // ── Hash ────────────────────────────────────────────────────────

    #[test]
    fn hash_consistent_with_eq() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let ctx = Context::new();
        let x1 = ctx.symbol("x");
        let x2 = ctx.symbol("x"); // same interned id

        assert_eq!(x1, x2);

        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        x1.hash(&mut h1);
        x2.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    // ── Compound expression ─────────────────────────────────────────

    #[test]
    fn compound_expression_x_sq_plus_2x_plus_1() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = &x * &x + &x * 2 + 1;
        let s = format!("{expr}");
        // Should contain all the pieces of x^2 + 2x + 1
        assert!(s.contains("x**2"), "got: {s}");
        assert!(s.contains("2*x"), "got: {s}");
        assert!(s.contains('1'), "got: {s}");
    }
}
