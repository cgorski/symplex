//! Operator overloads and trait implementations for `Expr`.
//!
//! Contains: +, -, *, /, ^ operators for Ex/&Ex/i64 combinations,
//! Neg, Sum, Product, From, PartialEq, Eq, Hash, Display, Debug, FromStr.

use std::fmt;
use std::hash;
use std::ops;
use std::sync::Arc;

use crate::output::display::fmt_expr;

use crate::api::context::Context;
use crate::api::expr::{Ex, Expr, Sort};
use crate::base::node::ExprId;

// ═══════════════════════════════════════════════════════════════════════════
// PartialEq / Eq / Hash
// ═══════════════════════════════════════════════════════════════════════════

/// Structural identity comparison.
///
/// Two expressions are `==` if and only if they share the same arena node
/// (same `ExprId` in the same `Context`). Because the arena uses hash-consing,
/// canonically equivalent expressions (e.g., `x + 1` and `1 + x`) do share
/// the same node and will compare equal.
///
/// **Important:** This is *not* mathematical equality. Expressions that are
/// mathematically equal but structurally different (e.g., `(x+1)^2` and
/// `x^2 + 2*x + 1`) will compare as *not equal* because they have different
/// canonical forms. Use [`Expr::equals()`] for mathematical equality testing,
/// or [`Expr::expand()`] to normalize before comparing.
impl<S: Sort> PartialEq for Expr<S> {
    fn eq(&self, other: &Self) -> bool {
        self.ctx_id == other.ctx_id && self.raw_id() == other.raw_id()
    }
}

impl<S: Sort> Eq for Expr<S> {}

impl<S: Sort> hash::Hash for Expr<S> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.ctx_id.hash(state);
        self.raw_id().hash(state);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Display / Debug
// ═══════════════════════════════════════════════════════════════════════════

impl<S: Sort> fmt::Display for Expr<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.inner.read();
        fmt_expr(&inner.arena, f, self.raw_id(), 0)
    }
}

impl<S: Sort> fmt::Debug for Expr<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ex({:?}, {:?})", self.ctx_id, self.raw_id())
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
                let rhs_id = self.checked_id(&rhs);
                let id = self.inner.write().arena.$arena_method(&[self.raw_id(), rhs_id]);
                self.wrap(id)
            }
        }
        impl ops::$trait<&Ex> for Ex {
            type Output = Ex;
            fn $method(self, rhs: &Ex) -> Ex {
                let rhs_id = self.checked_id(rhs);
                let id = self.inner.write().arena.$arena_method(&[self.raw_id(), rhs_id]);
                self.wrap(id)
            }
        }
        impl ops::$trait<Ex> for &Ex {
            type Output = Ex;
            fn $method(self, rhs: Ex) -> Ex {
                let rhs_id = self.checked_id(&rhs);
                let id = self.inner.write().arena.$arena_method(&[self.raw_id(), rhs_id]);
                self.wrap(id)
            }
        }
        impl ops::$trait<&Ex> for &Ex {
            type Output = Ex;
            fn $method(self, rhs: &Ex) -> Ex {
                let rhs_id = self.checked_id(rhs);
                let id = self.inner.write().arena.$arena_method(&[self.raw_id(), rhs_id]);
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
                let rhs_id = self.checked_id(&rhs);
                let id = self.inner.write().arena.$arena_method(self.raw_id(), rhs_id);
                self.wrap(id)
            }
        }
        impl ops::$trait<&Ex> for Ex {
            type Output = Ex;
            fn $method(self, rhs: &Ex) -> Ex {
                let rhs_id = self.checked_id(rhs);
                let id = self.inner.write().arena.$arena_method(self.raw_id(), rhs_id);
                self.wrap(id)
            }
        }
        impl ops::$trait<Ex> for &Ex {
            type Output = Ex;
            fn $method(self, rhs: Ex) -> Ex {
                let rhs_id = self.checked_id(&rhs);
                let id = self.inner.write().arena.$arena_method(self.raw_id(), rhs_id);
                self.wrap(id)
            }
        }
        impl ops::$trait<&Ex> for &Ex {
            type Output = Ex;
            fn $method(self, rhs: &Ex) -> Ex {
                let rhs_id = self.checked_id(rhs);
                let id = self.inner.write().arena.$arena_method(self.raw_id(), rhs_id);
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
        let id = self.inner.write().arena.neg(self.raw_id());
        self.wrap(id)
    }
}

impl ops::Neg for &Ex {
    type Output = Ex;
    fn neg(self) -> Ex {
        let id = self.inner.write().arena.neg(self.raw_id());
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
                let id = inner.arena.$arena_method(&[self.raw_id(), rhs_id]);
                drop(inner);
                self.wrap(id)
            }
        }
        impl ops::$trait<i64> for &Ex {
            type Output = Ex;
            fn $method(self, rhs: i64) -> Ex {
                let mut inner = self.inner.write();
                let rhs_id = inner.arena.int(rhs);
                let id = inner.arena.$arena_method(&[self.raw_id(), rhs_id]);
                drop(inner);
                self.wrap(id)
            }
        }
        impl ops::$trait<Ex> for i64 {
            type Output = Ex;
            fn $method(self, rhs: Ex) -> Ex {
                let mut inner = rhs.inner.write();
                let lhs_id = inner.arena.int(self);
                let id = inner.arena.$arena_method(&[lhs_id, rhs.raw_id()]);
                drop(inner);
                rhs.wrap(id)
            }
        }
        impl ops::$trait<&Ex> for i64 {
            type Output = Ex;
            fn $method(self, rhs: &Ex) -> Ex {
                let mut inner = rhs.inner.write();
                let lhs_id = inner.arena.int(self);
                let id = inner.arena.$arena_method(&[lhs_id, rhs.raw_id()]);
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
                let id = inner.arena.$arena_method(self.raw_id(), rhs_id);
                drop(inner);
                self.wrap(id)
            }
        }
        impl ops::$trait<i64> for &Ex {
            type Output = Ex;
            fn $method(self, rhs: i64) -> Ex {
                let mut inner = self.inner.write();
                let rhs_id = inner.arena.int(rhs);
                let id = inner.arena.$arena_method(self.raw_id(), rhs_id);
                drop(inner);
                self.wrap(id)
            }
        }
        impl ops::$trait<Ex> for i64 {
            type Output = Ex;
            fn $method(self, rhs: Ex) -> Ex {
                let mut inner = rhs.inner.write();
                let lhs_id = inner.arena.int(self);
                let id = inner.arena.$arena_method(lhs_id, rhs.raw_id());
                drop(inner);
                rhs.wrap(id)
            }
        }
        impl ops::$trait<&Ex> for i64 {
            type Output = Ex;
            fn $method(self, rhs: &Ex) -> Ex {
                let mut inner = rhs.inner.write();
                let lhs_id = inner.arena.int(self);
                let id = inner.arena.$arena_method(lhs_id, rhs.raw_id());
                drop(inner);
                rhs.wrap(id)
            }
        }
    };
}

impl_binary_binop_i64!(Sub, sub, sub);
impl_binary_binop_i64!(Div, div, div);

// ═══════════════════════════════════════════════════════════════════════════
// From<T> conversions — use a shared default context so that all
// values produced via `From` are in the same context and can be combined.
// ═══════════════════════════════════════════════════════════════════════════

/// Returns a lazily-initialised shared [`Context`] used by every
/// `From<integer>` conversion so the resulting expressions are
/// cross-compatible.
fn from_integer_ctx() -> &'static Context {
    static CTX: std::sync::OnceLock<Context> = std::sync::OnceLock::new();
    CTX.get_or_init(Context::new)
}

macro_rules! impl_from_integer {
    ($($t:ty),+) => {
        $(
            impl From<$t> for Ex {
                fn from(n: $t) -> Self {
                    from_integer_ctx().int(n as i64)
                }
            }
        )+
    };
}

impl_from_integer!(i8, i16, i32, i64, u8, u16, u32, isize);

impl From<u64> for Ex {
    fn from(n: u64) -> Self {
        let ctx = from_integer_ctx();
        if n <= i64::MAX as u64 {
            ctx.int(n as i64)
        } else {
            let id = {
                let mut inner = ctx.inner.write();
                inner.arena.big_int(num_bigint::BigInt::from(n))
            };
            Ex::from_raw_parts(ctx.id, Arc::clone(&ctx.inner), id)
        }
    }
}

impl From<usize> for Ex {
    fn from(n: usize) -> Self {
        let ctx = from_integer_ctx();
        if n <= i64::MAX as usize {
            ctx.int(n as i64)
        } else {
            let id = {
                let mut inner = ctx.inner.write();
                inner.arena.big_int(num_bigint::BigInt::from(n))
            };
            Ex::from_raw_parts(ctx.id, Arc::clone(&ctx.inner), id)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sum and Product trait implementations
// ═══════════════════════════════════════════════════════════════════════════

impl std::iter::Sum for Ex {
    fn sum<I: Iterator<Item = Ex>>(iter: I) -> Self {
        let items: Vec<Ex> = iter.collect();
        if items.is_empty() {
            return Context::new().zero();
        }
        let ctx_id = items[0].ctx_id;
        let inner = Arc::clone(&items[0].inner);
        // Check all items are from the same context
        for item in &items[1..] {
            if item.ctx_id != ctx_id {
                panic!(
                    "symplex: cannot combine expressions from different contexts \
                     (context {} and context {})",
                    ctx_id.0, item.ctx_id.0
                );
            }
        }
        let id = {
            let mut guard = inner.write();
            let ids: Vec<ExprId> = items.iter().map(|e| e.raw_id()).collect();
            guard.arena.add(&ids)
        };
        Ex::from_raw_parts(ctx_id, inner, id)
    }
}

impl<'a> std::iter::Sum<&'a Ex> for Ex {
    fn sum<I: Iterator<Item = &'a Ex>>(iter: I) -> Self {
        let items: Vec<&Ex> = iter.collect();
        if items.is_empty() {
            return Context::new().zero();
        }
        let ctx_id = items[0].ctx_id;
        let inner = Arc::clone(&items[0].inner);
        // Check all items are from the same context
        for item in &items[1..] {
            if item.ctx_id != ctx_id {
                panic!(
                    "symplex: cannot combine expressions from different contexts \
                     (context {} and context {})",
                    ctx_id.0, item.ctx_id.0
                );
            }
        }
        let id = {
            let mut guard = inner.write();
            let ids: Vec<ExprId> = items.iter().map(|e| e.raw_id()).collect();
            guard.arena.add(&ids)
        };
        Ex::from_raw_parts(ctx_id, inner, id)
    }
}

impl std::iter::Product for Ex {
    fn product<I: Iterator<Item = Ex>>(iter: I) -> Self {
        let items: Vec<Ex> = iter.collect();
        if items.is_empty() {
            return Context::new().one();
        }
        let ctx_id = items[0].ctx_id;
        let inner = Arc::clone(&items[0].inner);
        // Check all items are from the same context
        for item in &items[1..] {
            if item.ctx_id != ctx_id {
                panic!(
                    "symplex: cannot combine expressions from different contexts \
                     (context {} and context {})",
                    ctx_id.0, item.ctx_id.0
                );
            }
        }
        let id = {
            let mut guard = inner.write();
            let ids: Vec<ExprId> = items.iter().map(|e| e.raw_id()).collect();
            guard.arena.mul(&ids)
        };
        Ex::from_raw_parts(ctx_id, inner, id)
    }
}

impl<'a> std::iter::Product<&'a Ex> for Ex {
    fn product<I: Iterator<Item = &'a Ex>>(iter: I) -> Self {
        let items: Vec<&Ex> = iter.collect();
        if items.is_empty() {
            return Context::new().one();
        }
        let ctx_id = items[0].ctx_id;
        let inner = Arc::clone(&items[0].inner);
        // Check all items are from the same context
        for item in &items[1..] {
            if item.ctx_id != ctx_id {
                panic!(
                    "symplex: cannot combine expressions from different contexts \
                     (context {} and context {})",
                    ctx_id.0, item.ctx_id.0
                );
            }
        }
        let id = {
            let mut guard = inner.write();
            let ids: Vec<ExprId> = items.iter().map(|e| e.raw_id()).collect();
            guard.arena.mul(&ids)
        };
        Ex::from_raw_parts(ctx_id, inner, id)
    }
}
