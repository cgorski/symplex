//! Operator overloads, standard-library trait implementations, and
//! numeric-ingestion helpers for [`Expr`] / [`Context`].
//!
//! # What is implemented here
//!
//! | Trait / item | Notes |
//! |---|---|
//! | `PartialEq`, `Eq`, `Hash` | **Structural** identity (same arena node). Not mathematical equality — see [`Ex::equals`]. |
//! | `Display` | Infix pretty form (`x^2 + 1`). |
//! | `Debug` | `Ex(<display>)` (or `BoolEx(…)` / `SetEx(…)`). |
//! | `+ - * /` | `Ex ⊕ Ex`, with every combination of owned / borrowed operands. |
//! | `+ - * /` with scalars | Right-hand side: any [`Scalar`] (`i32`, `i64`, `i128`, `u32`, `u64`, `f64`, `BigInt`, `Ratio<BigInt>`). Left-hand side: `i64`, `f64`, `BigInt`, `Ratio<BigInt>`. Floats are converted *exactly* (see below). |
//! | `+= -= *= /=` | `Ex` with `Ex`, `&Ex`, and any [`Scalar`]. |
//! | unary `-` | On `Ex` and `&Ex`. |
//! | `Sum` / `Product` | For `Ex` (panics on an empty iterator — there is no context to build `0`/`1` in) and for `Option<Ex>` (`None` on empty). Prefer [`Context::sum`] / [`Context::product`]. |
//! | [`ToEx`] | Trait for "a Rust value that can be turned into an exact symbolic constant in a given context". |
//!
//! # Floats are exact
//!
//! `&x * 0.1` does **not** produce `x/10`.  Every finite `f64` is a dyadic
//! rational, and symplex converts it exactly:
//! `0.1_f64 == 3602879701896397 / 36028797018963968`.  This is deliberate —
//! the library never rounds behind your back.  If you meant the decimal
//! `1/10`, write `ctx.rational(1, 10)`, `ctx.decimal_str("0.1")`, or
//! `ctx.from_f64_nice(0.1)`.  `NaN` maps to the symbolic `nan` node and
//! `±∞` to `oo` / `-oo`.
//!
//! # What is deliberately *not* implemented
//!
//! * **`^` (`BitXor`) as exponentiation.**  Rust gives `^` a *lower*
//!   precedence than `*` and `+`, so `x ^ 2 * 3 + 1` would silently mean
//!   `x ^ (2 * 3 + 1) = x^7`.  A power operator that reads correctly in
//!   Rust source is impossible; use [`Ex::powi`](crate::expr::Ex::powi),
//!   [`Ex::pow`](crate::expr::Ex::pow), or the [`expr!`](crate::expr) macro
//!   (which re-parses `^` with mathematical precedence).
//! * **`From<i64>` / `FromStr` for `Ex`.**  An expression must live in a
//!   [`Context`]; there is no global context to build it in.  Use
//!   `ctx.int(3)`, `ctx.parse("x^2")`, or the [`ToEx`] trait.
//! * **`PartialOrd`.**  Ordering symbolic expressions is a three-valued
//!   question — use [`Ex::compare_numeric`], [`Ex::is_less_than`], or
//!   [`Ex::is_greater_than`].

use std::any::TypeId;
use std::cmp::Ordering;
use std::fmt;
use std::hash;
use std::ops;
use std::sync::Arc;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{ToPrimitive, Zero};
use smallvec::SmallVec;

use crate::api::context::Context;
use crate::api::expr::{Boolean, Ex, Expr, Numeric, SetValued, Sort};
use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric;
use crate::output::display::fmt_expr;

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

/// Name of the handle type for a given sort, used by `Debug`.
fn sort_type_name<S: Sort>() -> &'static str {
    let t = TypeId::of::<S>();
    if t == TypeId::of::<Numeric>() {
        "Ex"
    } else if t == TypeId::of::<Boolean>() {
        "BoolEx"
    } else if t == TypeId::of::<SetValued>() {
        "SetEx"
    } else {
        "Expr"
    }
}

/// `Debug` prints the handle type and the pretty form: `Ex(x^2 + 1)`.
///
/// ```
/// use symplex::prelude::*;
///
/// let ctx = Context::new();
/// let x = ctx.symbol("x");
/// assert_eq!(format!("{:?}", &x + 1), "Ex(x + 1)");
/// ```
impl<S: Sort> fmt::Debug for Expr<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(sort_type_name::<S>())?;
        f.write_str("(")?;
        fmt::Display::fmt(self, f)?;
        f.write_str(")")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Scalar — Rust numbers usable as operator operands
// ═══════════════════════════════════════════════════════════════════════════

mod private {
    /// Seals [`super::Scalar`].
    pub trait Sealed {}
}

/// Rust numeric types accepted as the right-hand operand of `+ - * /`
/// (and `+= -= *= /=`) on [`Ex`]: `i32`, `i64`, `i128`, `u32`, `u64`,
/// `f64`, [`BigInt`], [`Ratio<BigInt>`].
///
/// The operator impls are generic over this trait (`impl<T: Scalar> Add<T>
/// for Ex`) rather than repeated per type so that an unsuffixed literal
/// (`&x + 1`, `&x * 0.5`) still resolves to a unique impl and type
/// inference keeps working.  This trait is sealed; every implementor also
/// implements [`ToEx`].
///
/// On the **left** of an operator only `i64`, `f64`, `BigInt` and
/// `Ratio<BigInt>` are accepted (`2 * &x`, `0.5 * &x`); a wider set there
/// would make `2 * &x` ambiguous.  Write `&x * n` for other integer types.
pub trait Scalar: private::Sealed + Clone {
    /// Intern this value as an arena node (caller holds the write lock).
    #[doc(hidden)]
    fn scalar_to_id(self, arena: &mut Arena) -> ExprId;
}

macro_rules! impl_scalar {
    ($t:ty, |$v:ident, $arena:ident| $body:expr) => {
        impl private::Sealed for $t {}
        impl Scalar for $t {
            #[inline]
            fn scalar_to_id(self, $arena: &mut Arena) -> ExprId {
                let $v = self;
                $body
            }
        }
    };
}

impl_scalar!(i64, |v, arena| arena.int(v));
impl_scalar!(i32, |v, arena| arena.int(i64::from(v)));
impl_scalar!(u32, |v, arena| arena.int(i64::from(v)));
impl_scalar!(u64, |v, arena| match i64::try_from(v) {
    Ok(n) => arena.int(n),
    Err(_) => arena.big_int(BigInt::from(v)),
});
impl_scalar!(i128, |v, arena| match i64::try_from(v) {
    Ok(n) => arena.int(n),
    Err(_) => arena.big_int(BigInt::from(v)),
});
impl_scalar!(f64, |v, arena| f64_to_id(arena, v));
impl_scalar!(BigInt, |v, arena| arena.big_int(v));
impl_scalar!(Ratio<BigInt>, |v, arena| ratio_to_id(arena, v));

/// Intern a rational, handling the (pathological) zero denominator that
/// `Ratio::new_raw` can produce without panicking.
fn ratio_to_id(arena: &mut Arena, r: Ratio<BigInt>) -> ExprId {
    if r.denom().is_zero() {
        return if r.numer().is_zero() {
            arena.nan()
        } else {
            arena.complex_infinity()
        };
    }
    // `Ratio::new` reduces and normalises the sign, so raw (unreduced)
    // inputs intern to the same node as their canonical form.
    let r = Ratio::new(r.numer().clone(), r.denom().clone());
    let nid = arena.intern_num(r);
    arena.intern(ExprNode::Num(nid))
}

/// Exact dyadic conversion; `NaN` → `nan` node, `±∞` → `oo` / `-oo`.
fn f64_to_id(arena: &mut Arena, v: f64) -> ExprId {
    match numeric::f64_to_ratio_exact(v) {
        Some(r) => {
            let nid = arena.intern_num(r);
            arena.intern(ExprNode::Num(nid))
        }
        None if v.is_nan() => arena.nan(),
        None if v > 0.0 => arena.infinity(),
        None => arena.neg_infinity(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ToEx — Rust values that become exact symbolic constants
// ═══════════════════════════════════════════════════════════════════════════

/// A Rust value that can be turned into an exact expression in a [`Context`].
///
/// Implemented for every [`Scalar`] (`i32`, `i64`, `i128`, `u32`, `u64`,
/// `f64` with exact dyadic conversion — see the [module docs](self) —
/// [`BigInt`], [`Ratio<BigInt>`]) and for `Ex` / `&Ex` themselves
/// (identity, with a cross-context check).  This is the bound used by APIs
/// that accept "a number or an expression", such as
/// [`Ex::eval_f64_with`](crate::expr::Ex::eval_f64_with) and the arithmetic
/// operators on [`Equation`](crate::eq::Equation).  Unlike `Scalar`, it is
/// open: implement it for your own numeric types to use them with those
/// APIs.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::eq::ToEx;
///
/// let ctx = Context::new();
/// assert_eq!(format!("{}", 3_i64.to_ex(&ctx)), "3");
/// assert_eq!(format!("{}", 0.5_f64.to_ex(&ctx)), "1/2");
/// assert_eq!(format!("{}", f64::NAN.to_ex(&ctx)), "nan");
/// let x = ctx.symbol("x");
/// assert_eq!(x.to_ex(&ctx), x);
/// ```
pub trait ToEx {
    /// Build the exact expression for this value in `ctx`.
    ///
    /// # Panics
    ///
    /// The `Ex` / `&Ex` implementations panic if the expression belongs to
    /// a different context than `ctx` (the standard cross-context guard).
    fn to_ex(&self, ctx: &Context) -> Ex;
}

impl<T: Scalar> ToEx for T {
    #[inline]
    fn to_ex(&self, ctx: &Context) -> Ex {
        let id = self.clone().scalar_to_id(&mut ctx.inner.write().arena);
        ctx_wrap(ctx, id)
    }
}

impl ToEx for Ex {
    #[inline]
    fn to_ex(&self, ctx: &Context) -> Ex {
        let _ = ctx.own_id(self);
        self.clone()
    }
}

impl ToEx for &Ex {
    #[inline]
    fn to_ex(&self, ctx: &Context) -> Ex {
        let _ = ctx.own_id(self);
        (*self).clone()
    }
}

/// Wrap an `ExprId` that lives in `ctx`'s arena.
#[inline]
fn ctx_wrap(ctx: &Context, id: ExprId) -> Ex {
    Ex::from_raw_parts(ctx.id, Arc::clone(&ctx.inner), id)
}

// ═══════════════════════════════════════════════════════════════════════════
// Operator implementations: Ex ⊕ Ex
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
                let id = self
                    .inner
                    .write()
                    .arena
                    .$arena_method(&[self.raw_id(), rhs_id]);
                self.wrap(id)
            }
        }
        impl ops::$trait<&Ex> for Ex {
            type Output = Ex;
            fn $method(self, rhs: &Ex) -> Ex {
                let rhs_id = self.checked_id(rhs);
                let id = self
                    .inner
                    .write()
                    .arena
                    .$arena_method(&[self.raw_id(), rhs_id]);
                self.wrap(id)
            }
        }
        impl ops::$trait<Ex> for &Ex {
            type Output = Ex;
            fn $method(self, rhs: Ex) -> Ex {
                let rhs_id = self.checked_id(&rhs);
                let id = self
                    .inner
                    .write()
                    .arena
                    .$arena_method(&[self.raw_id(), rhs_id]);
                self.wrap(id)
            }
        }
        impl ops::$trait<&Ex> for &Ex {
            type Output = Ex;
            fn $method(self, rhs: &Ex) -> Ex {
                let rhs_id = self.checked_id(rhs);
                let id = self
                    .inner
                    .write()
                    .arena
                    .$arena_method(&[self.raw_id(), rhs_id]);
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
                let id = self
                    .inner
                    .write()
                    .arena
                    .$arena_method(self.raw_id(), rhs_id);
                self.wrap(id)
            }
        }
        impl ops::$trait<&Ex> for Ex {
            type Output = Ex;
            fn $method(self, rhs: &Ex) -> Ex {
                let rhs_id = self.checked_id(rhs);
                let id = self
                    .inner
                    .write()
                    .arena
                    .$arena_method(self.raw_id(), rhs_id);
                self.wrap(id)
            }
        }
        impl ops::$trait<Ex> for &Ex {
            type Output = Ex;
            fn $method(self, rhs: Ex) -> Ex {
                let rhs_id = self.checked_id(&rhs);
                let id = self
                    .inner
                    .write()
                    .arena
                    .$arena_method(self.raw_id(), rhs_id);
                self.wrap(id)
            }
        }
        impl ops::$trait<&Ex> for &Ex {
            type Output = Ex;
            fn $method(self, rhs: &Ex) -> Ex {
                let rhs_id = self.checked_id(rhs);
                let id = self
                    .inner
                    .write()
                    .arena
                    .$arena_method(self.raw_id(), rhs_id);
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
// Operator implementations: Ex ⊕ scalar (generic over `Scalar`)
// ═══════════════════════════════════════════════════════════════════════════
//
// The scalar is interned under the same write lock as the operation, so
// each operator is a single lock acquisition.  `f64` operands are
// converted *exactly* (see module docs).  A single generic impl per
// operator (rather than one impl per scalar type) is what keeps
// `let e = &x + 1;` inferable: the compiler sees exactly one candidate
// impl for `&Ex + {integer}` and can resolve `Output` before the literal
// falls back to `i32`.

macro_rules! impl_scalar_rhs_nary {
    ($trait:ident, $method:ident, $arena_method:ident) => {
        impl<T: Scalar> ops::$trait<T> for Ex {
            type Output = Ex;
            fn $method(self, rhs: T) -> Ex {
                let mut inner = self.inner.write();
                let rhs_id = rhs.scalar_to_id(&mut inner.arena);
                let id = inner.arena.$arena_method(&[self.raw_id(), rhs_id]);
                drop(inner);
                self.wrap(id)
            }
        }
        impl<T: Scalar> ops::$trait<T> for &Ex {
            type Output = Ex;
            fn $method(self, rhs: T) -> Ex {
                let mut inner = self.inner.write();
                let rhs_id = rhs.scalar_to_id(&mut inner.arena);
                let id = inner.arena.$arena_method(&[self.raw_id(), rhs_id]);
                drop(inner);
                self.wrap(id)
            }
        }
    };
}

macro_rules! impl_scalar_rhs_binary {
    ($trait:ident, $method:ident, $arena_method:ident) => {
        impl<T: Scalar> ops::$trait<T> for Ex {
            type Output = Ex;
            fn $method(self, rhs: T) -> Ex {
                let mut inner = self.inner.write();
                let rhs_id = rhs.scalar_to_id(&mut inner.arena);
                let id = inner.arena.$arena_method(self.raw_id(), rhs_id);
                drop(inner);
                self.wrap(id)
            }
        }
        impl<T: Scalar> ops::$trait<T> for &Ex {
            type Output = Ex;
            fn $method(self, rhs: T) -> Ex {
                let mut inner = self.inner.write();
                let rhs_id = rhs.scalar_to_id(&mut inner.arena);
                let id = inner.arena.$arena_method(self.raw_id(), rhs_id);
                drop(inner);
                self.wrap(id)
            }
        }
    };
}

impl_scalar_rhs_nary!(Add, add, add);
impl_scalar_rhs_nary!(Mul, mul, mul);
impl_scalar_rhs_binary!(Sub, sub, sub);
impl_scalar_rhs_binary!(Div, div, div);

// ── Compound assignment ────────────────────────────────────────────────

impl<T: Scalar> ops::AddAssign<T> for Ex {
    fn add_assign(&mut self, rhs: T) {
        *self = &*self + rhs;
    }
}
impl<T: Scalar> ops::SubAssign<T> for Ex {
    fn sub_assign(&mut self, rhs: T) {
        *self = &*self - rhs;
    }
}
impl<T: Scalar> ops::MulAssign<T> for Ex {
    fn mul_assign(&mut self, rhs: T) {
        *self = &*self * rhs;
    }
}
impl<T: Scalar> ops::DivAssign<T> for Ex {
    fn div_assign(&mut self, rhs: T) {
        *self = &*self / rhs;
    }
}

impl ops::AddAssign<Ex> for Ex {
    fn add_assign(&mut self, rhs: Ex) {
        *self = &*self + &rhs;
    }
}
impl ops::AddAssign<&Ex> for Ex {
    fn add_assign(&mut self, rhs: &Ex) {
        *self = &*self + rhs;
    }
}
impl ops::SubAssign<Ex> for Ex {
    fn sub_assign(&mut self, rhs: Ex) {
        *self = &*self - &rhs;
    }
}
impl ops::SubAssign<&Ex> for Ex {
    fn sub_assign(&mut self, rhs: &Ex) {
        *self = &*self - rhs;
    }
}
impl ops::MulAssign<Ex> for Ex {
    fn mul_assign(&mut self, rhs: Ex) {
        *self = &*self * &rhs;
    }
}
impl ops::MulAssign<&Ex> for Ex {
    fn mul_assign(&mut self, rhs: &Ex) {
        *self = &*self * rhs;
    }
}
impl ops::DivAssign<Ex> for Ex {
    fn div_assign(&mut self, rhs: Ex) {
        *self = &*self / &rhs;
    }
}
impl ops::DivAssign<&Ex> for Ex {
    fn div_assign(&mut self, rhs: &Ex) {
        *self = &*self / rhs;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Operator implementations: scalar ⊕ Ex (explicit types only)
// ═══════════════════════════════════════════════════════════════════════════
//
// A left-hand literal (`2 * &x`) is resolved by method lookup on the
// *literal's* type, so every additional integer type here would make
// `2 * &x` ambiguous.  We therefore accept exactly one integer type
// (`i64`) and one float type (`f64`) plus the arbitrary-precision types,
// which are never written as literals.

macro_rules! impl_scalar_lhs_ops {
    ($t:ty) => {
        impl_scalar_lhs_ops!(@nary $t, Add, add, add);
        impl_scalar_lhs_ops!(@nary $t, Mul, mul, mul);
        impl_scalar_lhs_ops!(@binary $t, Sub, sub, sub);
        impl_scalar_lhs_ops!(@binary $t, Div, div, div);
    };
    (@nary $t:ty, $trait:ident, $method:ident, $arena_method:ident) => {
        impl ops::$trait<Ex> for $t {
            type Output = Ex;
            fn $method(self, rhs: Ex) -> Ex {
                let mut inner = rhs.inner.write();
                let lhs_id = self.scalar_to_id(&mut inner.arena);
                let id = inner.arena.$arena_method(&[lhs_id, rhs.raw_id()]);
                drop(inner);
                rhs.wrap(id)
            }
        }
        impl ops::$trait<&Ex> for $t {
            type Output = Ex;
            fn $method(self, rhs: &Ex) -> Ex {
                let mut inner = rhs.inner.write();
                let lhs_id = self.scalar_to_id(&mut inner.arena);
                let id = inner.arena.$arena_method(&[lhs_id, rhs.raw_id()]);
                drop(inner);
                rhs.wrap(id)
            }
        }
    };
    (@binary $t:ty, $trait:ident, $method:ident, $arena_method:ident) => {
        impl ops::$trait<Ex> for $t {
            type Output = Ex;
            fn $method(self, rhs: Ex) -> Ex {
                let mut inner = rhs.inner.write();
                let lhs_id = self.scalar_to_id(&mut inner.arena);
                let id = inner.arena.$arena_method(lhs_id, rhs.raw_id());
                drop(inner);
                rhs.wrap(id)
            }
        }
        impl ops::$trait<&Ex> for $t {
            type Output = Ex;
            fn $method(self, rhs: &Ex) -> Ex {
                let mut inner = rhs.inner.write();
                let lhs_id = self.scalar_to_id(&mut inner.arena);
                let id = inner.arena.$arena_method(lhs_id, rhs.raw_id());
                drop(inner);
                rhs.wrap(id)
            }
        }
    };
}

impl_scalar_lhs_ops!(i64);
impl_scalar_lhs_ops!(f64);
impl_scalar_lhs_ops!(BigInt);
impl_scalar_lhs_ops!(Ratio<BigInt>);

// ═══════════════════════════════════════════════════════════════════════════
// Sum and Product trait implementations
// ═══════════════════════════════════════════════════════════════════════════

/// Combine a non-empty list of same-context expressions with one n-ary
/// arena operation.  Panics (with the standard cross-context message) if
/// the expressions come from different contexts.
fn combine_nonempty(items: &[&Ex], op: fn(&mut Arena, &[ExprId]) -> ExprId) -> Ex {
    let first = items[0];
    let ids: Vec<ExprId> = items.iter().map(|e| first.checked_id(e)).collect();
    let id = op(&mut first.inner.write().arena, &ids);
    first.wrap(id)
}

const EMPTY_SUM_MSG: &str = "symplex: cannot `sum()` an empty iterator of `Ex` — there is no \
     context to build `0` in. Use `Context::sum(iter)` (returns `0` on empty) or collect into \
     `Option<Ex>` (returns `None` on empty).";

const EMPTY_PRODUCT_MSG: &str = "symplex: cannot `product()` an empty iterator of `Ex` — there \
     is no context to build `1` in. Use `Context::product(iter)` (returns `1` on empty) or \
     collect into `Option<Ex>` (returns `None` on empty).";

/// Sum an iterator of expressions.
///
/// # Panics
///
/// Panics on an **empty** iterator: there is no context in which to build
/// `0`.  Use [`Context::sum`] (yields `0` on empty) or collect into
/// `Option<Ex>` (yields `None` on empty) when the iterator may be empty.
/// Also panics if the expressions come from different contexts.
///
/// ```
/// use symplex::prelude::*;
///
/// let ctx = Context::new();
/// let total: Ex = (1..=4).map(|n| ctx.int(n)).sum();
/// assert_eq!(format!("{total}"), "10");
///
/// let none: Option<Ex> = std::iter::empty::<Ex>().sum();
/// assert!(none.is_none());
/// ```
impl std::iter::Sum for Ex {
    fn sum<I: Iterator<Item = Ex>>(iter: I) -> Self {
        let items: Vec<Ex> = iter.collect();
        assert!(!items.is_empty(), "{EMPTY_SUM_MSG}");
        let refs: Vec<&Ex> = items.iter().collect();
        combine_nonempty(&refs, Arena::add)
    }
}

impl<'a> std::iter::Sum<&'a Ex> for Ex {
    fn sum<I: Iterator<Item = &'a Ex>>(iter: I) -> Self {
        let items: Vec<&Ex> = iter.collect();
        assert!(!items.is_empty(), "{EMPTY_SUM_MSG}");
        combine_nonempty(&items, Arena::add)
    }
}

/// Multiply an iterator of expressions.
///
/// # Panics
///
/// Panics on an **empty** iterator (no context to build `1` in) — see
/// [`Context::product`] and the `Option<Ex>` implementation — and on
/// mixed-context input.
impl std::iter::Product for Ex {
    fn product<I: Iterator<Item = Ex>>(iter: I) -> Self {
        let items: Vec<Ex> = iter.collect();
        assert!(!items.is_empty(), "{EMPTY_PRODUCT_MSG}");
        let refs: Vec<&Ex> = items.iter().collect();
        combine_nonempty(&refs, Arena::mul)
    }
}

impl<'a> std::iter::Product<&'a Ex> for Ex {
    fn product<I: Iterator<Item = &'a Ex>>(iter: I) -> Self {
        let items: Vec<&Ex> = iter.collect();
        assert!(!items.is_empty(), "{EMPTY_PRODUCT_MSG}");
        combine_nonempty(&items, Arena::mul)
    }
}

/// `None` on an empty iterator, otherwise `Some(sum)`.
///
/// ```
/// use symplex::prelude::*;
///
/// let ctx = Context::new();
/// let xs = ctx.symbols_indexed("x", 3);
/// let s: Option<Ex> = xs.iter().sum();
/// assert_eq!(format!("{}", s.unwrap()), "x0 + x1 + x2");
/// let e: Option<Ex> = Vec::<Ex>::new().into_iter().sum();
/// assert!(e.is_none());
/// ```
impl std::iter::Sum<Ex> for Option<Ex> {
    fn sum<I: Iterator<Item = Ex>>(iter: I) -> Self {
        let items: Vec<Ex> = iter.collect();
        if items.is_empty() {
            return None;
        }
        let refs: Vec<&Ex> = items.iter().collect();
        Some(combine_nonempty(&refs, Arena::add))
    }
}

impl<'a> std::iter::Sum<&'a Ex> for Option<Ex> {
    fn sum<I: Iterator<Item = &'a Ex>>(iter: I) -> Self {
        let items: Vec<&Ex> = iter.collect();
        if items.is_empty() {
            return None;
        }
        Some(combine_nonempty(&items, Arena::add))
    }
}

/// `None` on an empty iterator, otherwise `Some(product)`.
impl std::iter::Product<Ex> for Option<Ex> {
    fn product<I: Iterator<Item = Ex>>(iter: I) -> Self {
        let items: Vec<Ex> = iter.collect();
        if items.is_empty() {
            return None;
        }
        let refs: Vec<&Ex> = items.iter().collect();
        Some(combine_nonempty(&refs, Arena::mul))
    }
}

impl<'a> std::iter::Product<&'a Ex> for Option<Ex> {
    fn product<I: Iterator<Item = &'a Ex>>(iter: I) -> Self {
        let items: Vec<&Ex> = iter.collect();
        if items.is_empty() {
            return None;
        }
        Some(combine_nonempty(&items, Arena::mul))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Context — numeric ingestion and bulk construction
// ═══════════════════════════════════════════════════════════════════════════

impl Context {
    /// Crate-internal: the `ExprId` of `e` after verifying it belongs to
    /// this context.  Panics with the standard cross-context message
    /// otherwise (mirror of `Expr::checked_id`).
    #[inline]
    pub(crate) fn own_id(&self, e: &Ex) -> ExprId {
        if e.ctx_id != self.id {
            panic!(
                "symplex: cannot combine expressions from different contexts \
                 (context {} and context {}). All expressions in an operation \
                 must originate from the same Context.",
                self.id.0, e.ctx_id.0
            );
        }
        e.raw_id()
    }

    // ── Floats ─────────────────────────────────────────────────────────

    /// Convert an `f64` to the **exact** rational it represents.
    ///
    /// Every finite `f64` is a dyadic rational `m · 2ᵏ`, and this method
    /// preserves it bit-for-bit.  That means decimal literals are *not*
    /// what they look like:
    ///
    /// * `0.5` → `1/2` (exact in binary — fine)
    /// * `0.1` → `3602879701896397/36028797018963968` (**not** `1/10`)
    /// * `0.3` → `5404319552844595/18014398509481984`
    ///
    /// This is the honest conversion: the library never rounds behind your
    /// back.  When you want the "nice" rational a human meant, use
    /// [`from_f64_approx`](Self::from_f64_approx) /
    /// [`from_f64_nice`](Self::from_f64_nice), or write the decimal as a
    /// string with [`decimal_str`](Self::decimal_str).
    ///
    /// `+∞` / `−∞` become the symbolic `oo` / `-oo` nodes.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for `NaN` — it has no numeric
    /// value.  (Use [`Context::nan`] if you want the symbolic `nan` node.)
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(format!("{}", ctx.from_f64(0.5).unwrap()), "1/2");
    /// assert_eq!(format!("{}", ctx.from_f64(-3.0).unwrap()), "-3");
    /// assert_eq!(
    ///     format!("{}", ctx.from_f64(0.1).unwrap()),
    ///     "3602879701896397/36028797018963968"
    /// );
    /// assert_eq!(format!("{}", ctx.from_f64(f64::INFINITY).unwrap()), "oo");
    /// assert!(ctx.from_f64(f64::NAN).is_err());
    /// ```
    pub fn from_f64(&self, v: f64) -> Result<Ex, SymplexError> {
        if v.is_nan() {
            return Err(SymplexError::InvalidArgument {
                operation: "from_f64",
                reason: "NaN has no exact rational value (use Context::nan() for the \
                         symbolic nan node)"
                    .to_string(),
            });
        }
        let id = f64_to_id(&mut self.inner.write().arena, v);
        Ok(ctx_wrap(self, id))
    }

    /// Convert an `f64` to the closest rational with denominator at most
    /// `max_denominator` — the "human" reading of a float.
    ///
    /// Uses continued-fraction convergents (see
    /// [`numeric::f64_to_ratio_approx`]), so `0.1 → 1/10`,
    /// `0.3333333333333333 → 1/3`, and `3.14159 → 355/113` for
    /// `max_denominator = 1000`.  `±∞` become `oo` / `-oo`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for `NaN` or `max_denominator == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(format!("{}", ctx.from_f64_approx(0.1, 1_000_000).unwrap()), "1/10");
    /// assert_eq!(format!("{}", ctx.from_f64_approx(1.0 / 3.0, 1_000_000).unwrap()), "1/3");
    /// assert_eq!(format!("{}", ctx.from_f64_approx(std::f64::consts::PI, 1000).unwrap()), "355/113");
    /// assert!(ctx.from_f64_approx(1.5, 0).is_err());
    /// ```
    pub fn from_f64_approx(&self, v: f64, max_denominator: u64) -> Result<Ex, SymplexError> {
        if v.is_nan() {
            return Err(SymplexError::InvalidArgument {
                operation: "from_f64_approx",
                reason: "NaN has no rational approximation".to_string(),
            });
        }
        if max_denominator == 0 {
            return Err(SymplexError::InvalidArgument {
                operation: "from_f64_approx",
                reason: "max_denominator must be at least 1".to_string(),
            });
        }
        if v.is_infinite() {
            return Ok(if v > 0.0 {
                self.infinity()
            } else {
                self.neg_infinity()
            });
        }
        // Finite and max_denominator ≥ 1 ⇒ always Some.
        let r = numeric::f64_to_ratio_approx(v, max_denominator).ok_or_else(|| {
            SymplexError::ComputationFailed {
                operation: "from_f64_approx",
                reason: format!("no rational approximation for {v}"),
            }
        })?;
        Ok(self.from_ratio(r))
    }

    /// The "nice" rational for a float: [`from_f64_approx`](Self::from_f64_approx)
    /// with `max_denominator = 1_000_000_000`.
    ///
    /// This recovers every decimal with up to nine fractional digits
    /// exactly (`0.1 → 1/10`, `2.375 → 19/8`, `0.123456789 →
    /// 123456789/1000000000`) while still turning `1.0/3.0` into `1/3`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for `NaN`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(format!("{}", ctx.from_f64_nice(0.1).unwrap()), "1/10");
    /// assert_eq!(format!("{}", ctx.from_f64_nice(0.1 + 0.2).unwrap()), "3/10");
    /// assert_eq!(format!("{}", ctx.from_f64_nice(2.375).unwrap()), "19/8");
    /// ```
    pub fn from_f64_nice(&self, v: f64) -> Result<Ex, SymplexError> {
        self.from_f64_approx(v, 1_000_000_000)
    }

    // ── Exact integers and rationals ───────────────────────────────────

    /// Create an integer expression from a [`BigInt`].
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use num_bigint::BigInt;
    ///
    /// let ctx = Context::new();
    /// let big = BigInt::parse_bytes(b"123456789012345678901234567890", 10).unwrap();
    /// assert_eq!(format!("{}", ctx.from_bigint(big)), "123456789012345678901234567890");
    /// ```
    pub fn from_bigint(&self, n: BigInt) -> Ex {
        let id = self.inner.write().arena.big_int(n);
        ctx_wrap(self, id)
    }

    /// Create a rational expression from a [`Ratio<BigInt>`].
    ///
    /// The value is stored in lowest terms.  A ratio with a zero
    /// denominator (only constructible via `Ratio::new_raw`) maps to
    /// `zoo` (complex infinity), or `nan` for `0/0`, instead of panicking.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use num_bigint::BigInt;
    /// use num_rational::Ratio;
    ///
    /// let ctx = Context::new();
    /// let r = Ratio::new(BigInt::from(6), BigInt::from(-4));
    /// assert_eq!(format!("{}", ctx.from_ratio(r)), "-3/2");
    /// ```
    pub fn from_ratio(&self, r: Ratio<BigInt>) -> Ex {
        let id = ratio_to_id(&mut self.inner.write().arena, r);
        ctx_wrap(self, id)
    }

    /// Create an integer expression from an `i128`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(format!("{}", ctx.from_i128(i128::MIN)), "-170141183460469231731687303715884105728");
    /// ```
    pub fn from_i128(&self, n: i128) -> Ex {
        let id = n.scalar_to_id(&mut self.inner.write().arena);
        ctx_wrap(self, id)
    }

    /// Create an integer expression from a `u64`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(format!("{}", ctx.from_u64(u64::MAX)), "18446744073709551615");
    /// ```
    pub fn from_u64(&self, n: u64) -> Ex {
        let id = n.scalar_to_id(&mut self.inner.write().arena);
        ctx_wrap(self, id)
    }

    /// Parse an exact rational from a string of the form `"p"`, `"p/q"`,
    /// or `"-p/q"` (arbitrary-precision integers, surrounding whitespace
    /// ignored).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the string is not two integers
    /// separated by `/`, or if the denominator is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(format!("{}", ctx.rational_str("22/7").unwrap()), "22/7");
    /// assert_eq!(format!("{}", ctx.rational_str("-6/4").unwrap()), "-3/2");
    /// assert_eq!(format!("{}", ctx.rational_str(" 42 ").unwrap()), "42");
    /// assert!(ctx.rational_str("1/0").is_err());
    /// assert!(ctx.rational_str("0.5").is_err()); // use decimal_str
    /// ```
    pub fn rational_str(&self, s: &str) -> Result<Ex, SymplexError> {
        let bad = |reason: String| SymplexError::InvalidArgument {
            operation: "rational_str",
            reason,
        };
        let s = s.trim();
        let (num_s, den_s) = match s.split_once('/') {
            Some((n, d)) => (n.trim(), Some(d.trim())),
            None => (s, None),
        };
        let numer: BigInt = num_s
            .parse()
            .map_err(|e| bad(format!("invalid numerator {num_s:?}: {e}")))?;
        let denom: BigInt = match den_s {
            Some(d) => d
                .parse()
                .map_err(|e| bad(format!("invalid denominator {d:?}: {e}")))?,
            None => BigInt::from(1),
        };
        if denom.is_zero() {
            return Err(bad("denominator is zero".to_string()));
        }
        Ok(self.from_ratio(Ratio::new(numer, denom)))
    }

    /// Parse a decimal literal **exactly**: `"0.1"` → `1/10`, `"2.5e3"` →
    /// `2500`, `"-1.25e-2"` → `-1/80`.
    ///
    /// Accepts `[+|-] digits [. digits] [(e|E) [+|-] digits]` with
    /// arbitrary length; nothing is rounded.  (The general
    /// [`parse`](Self::parse) also reads decimals exactly, but it accepts
    /// whole expressions; this method rejects anything that is not a plain
    /// number.)
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the string is not a decimal
    /// literal.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(format!("{}", ctx.decimal_str("0.1").unwrap()), "1/10");
    /// assert_eq!(format!("{}", ctx.decimal_str("3.14159").unwrap()), "314159/100000");
    /// assert_eq!(format!("{}", ctx.decimal_str("1e-5").unwrap()), "1/100000");
    /// assert_eq!(format!("{}", ctx.decimal_str("-2.5E3").unwrap()), "-2500");
    /// assert_eq!(format!("{}", ctx.decimal_str(".5").unwrap()), "1/2");
    /// assert!(ctx.decimal_str("1/2").is_err());
    /// assert!(ctx.decimal_str("abc").is_err());
    /// ```
    pub fn decimal_str(&self, s: &str) -> Result<Ex, SymplexError> {
        let r = parse_decimal_exact(s).ok_or_else(|| SymplexError::InvalidArgument {
            operation: "decimal_str",
            reason: format!("{s:?} is not a decimal literal ([+-]digits[.digits][e[+-]digits])"),
        })?;
        Ok(self.from_ratio(r))
    }

    // ── Composite construction ─────────────────────────────────────────

    /// Build the complex number `re + im·I`.
    ///
    /// Both parts may be arbitrary expressions; the result is
    /// canonicalized like any other sum.
    ///
    /// # Panics
    ///
    /// Panics if `re` or `im` belongs to a different context.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let z = ctx.complex(&ctx.int(3), &ctx.int(-4));
    /// assert_eq!(format!("{z}"), "-4*I + 3");
    /// let (re, im) = z.as_real_imag();
    /// assert_eq!((format!("{re}"), format!("{im}")), ("3".to_string(), "-4".to_string()));
    /// ```
    pub fn complex(&self, re: &Ex, im: &Ex) -> Ex {
        let re_id = self.own_id(re);
        let im_id = self.own_id(im);
        let mut inner = self.inner.write();
        let i = inner.arena.i_unit();
        let im_i = inner.arena.mul(&[im_id, i]);
        let id = inner.arena.add(&[re_id, im_i]);
        drop(inner);
        ctx_wrap(self, id)
    }

    /// Create several symbols at once.
    ///
    /// # Panics
    ///
    /// Panics if any name is empty (as [`symbol`](Self::symbol) does).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let v = ctx.symbols(&["x", "y", "z"]);
    /// assert_eq!(v.len(), 3);
    /// assert_eq!(format!("{}", &v[0] + &v[1] + &v[2]), "x + y + z");
    /// ```
    pub fn symbols(&self, names: &[&str]) -> Vec<Ex> {
        names.iter().map(|n| self.symbol(n)).collect()
    }

    /// Create the indexed family `base0, base1, …, base{n-1}`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = ctx.symbols_indexed("a", 3);
    /// assert_eq!(format!("{}", &a[0] * &a[1] * &a[2]), "a0*a1*a2");
    /// assert!(ctx.symbols_indexed("q", 0).is_empty());
    /// ```
    pub fn symbols_indexed(&self, base: &str, n: usize) -> Vec<Ex> {
        (0..n).map(|i| self.symbol(&format!("{base}{i}"))).collect()
    }

    /// Apply a named, otherwise undefined function to arguments: `f(x, y)`.
    ///
    /// This produces the generic `Apply` node.  Nothing is known about
    /// `f`, so:
    ///
    /// * [`eval`](crate::expr::Ex::eval) and
    ///   [`simplify`](crate::expr::Ex::simplify) leave it alone (its
    ///   arguments are still evaluated / simplified);
    /// * [`diff`](crate::expr::Ex::diff) applies the chain rule and yields a
    ///   formal `Derivative(f(…), …)` for the outer function;
    /// * numeric evaluation / [`compile`](crate::expr::Ex::compile) return
    ///   [`SymplexError::NotImplemented`].
    ///
    /// `args` may be a slice of `Ex` or of `&Ex`.
    ///
    /// # Panics
    ///
    /// Panics if `name` is empty or any argument belongs to another context.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = ctx.apply("f", &[&x]);
    /// assert_eq!(format!("{f}"), "f(x)");
    /// assert_eq!(f.eval(), f);
    /// assert_eq!(format!("{}", f.diff(&x)), "Derivative(f(x), x)");
    /// // Chain rule on the argument:
    /// let g = ctx.apply("g", &[x.powi(2)]);
    /// assert_eq!(format!("{}", g.diff(&x)), "2*x*Derivative(g(x^2), x^2)");
    /// ```
    pub fn apply<T: AsRef<Ex>>(&self, name: &str, args: &[T]) -> Ex {
        assert!(!name.is_empty(), "function name cannot be empty");
        let ids: SmallVec<[ExprId; 2]> = args.iter().map(|a| self.own_id(a.as_ref())).collect();
        let mut inner = self.inner.write();
        let sid = inner.arena.symbols.intern(name);
        let id = inner.arena.intern(ExprNode::Apply(sid, ids));
        drop(inner);
        ctx_wrap(self, id)
    }

    // ── Bulk arithmetic ────────────────────────────────────────────────

    /// Sum an iterator of expressions, yielding `0` when it is empty.
    ///
    /// Unlike `iter.sum::<Ex>()`, this never panics on empty input because
    /// the context is known.  Items may be `Ex` or `&Ex`.
    ///
    /// # Panics
    ///
    /// Panics if an item belongs to a different context.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let xs = ctx.symbols_indexed("x", 3);
    /// assert_eq!(format!("{}", ctx.sum(&xs)), "x0 + x1 + x2");
    /// assert_eq!(format!("{}", ctx.sum(Vec::<Ex>::new())), "0");
    /// ```
    pub fn sum<I>(&self, iter: I) -> Ex
    where
        I: IntoIterator,
        I::Item: AsRef<Ex>,
    {
        let ids: Vec<ExprId> = iter.into_iter().map(|e| self.own_id(e.as_ref())).collect();
        if ids.is_empty() {
            return self.zero();
        }
        let id = self.inner.write().arena.add(&ids);
        ctx_wrap(self, id)
    }

    /// Multiply an iterator of expressions, yielding `1` when it is empty.
    ///
    /// # Panics
    ///
    /// Panics if an item belongs to a different context.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let p = ctx.product((1..=4).map(|n| ctx.int(n)));
    /// assert_eq!(format!("{p}"), "24");
    /// assert_eq!(format!("{}", ctx.product(Vec::<Ex>::new())), "1");
    /// ```
    pub fn product<I>(&self, iter: I) -> Ex
    where
        I: IntoIterator,
        I::Item: AsRef<Ex>,
    {
        let ids: Vec<ExprId> = iter.into_iter().map(|e| self.own_id(e.as_ref())).collect();
        if ids.is_empty() {
            return self.one();
        }
        let id = self.inner.write().arena.mul(&ids);
        ctx_wrap(self, id)
    }
}

/// Parse `[+-]digits[.digits][(e|E)[+-]digits]` into an exact rational.
fn parse_decimal_exact(s: &str) -> Option<Ratio<BigInt>> {
    let s = s.trim();
    let (negative, rest) = match s.as_bytes().first()? {
        b'-' => (true, &s[1..]),
        b'+' => (false, &s[1..]),
        _ => (false, s),
    };
    // Split off the exponent.
    let (mantissa, exp) = match rest.find(['e', 'E']) {
        Some(pos) => {
            let e: i64 = rest[pos + 1..].parse().ok()?;
            // Reject "1e+" style junk that i64::parse would accept? It does
            // not: "+"/"" fail to parse, which is what we want.
            (&rest[..pos], e)
        }
        None => (rest, 0),
    };
    let (int_part, frac_part) = match mantissa.split_once('.') {
        Some((i, f)) => (i, f),
        None => (mantissa, ""),
    };
    if int_part.is_empty() && frac_part.is_empty() {
        return None;
    }
    if !int_part.bytes().all(|b| b.is_ascii_digit())
        || !frac_part.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let digits = format!("{int_part}{frac_part}");
    let mut numer: BigInt = if digits.is_empty() {
        BigInt::zero()
    } else {
        digits.parse().ok()?
    };
    if negative {
        numer = -numer;
    }
    // value = numer · 10^(exp − frac_len)
    let scale = exp - frac_part.len() as i64;
    let ten = BigInt::from(10);
    let value = if scale >= 0 {
        Ratio::from_integer(numer * num_traits::pow(ten, usize::try_from(scale).ok()?))
    } else {
        Ratio::new(numer, num_traits::pow(ten, usize::try_from(-scale).ok()?))
    };
    Some(value)
}

// ═══════════════════════════════════════════════════════════════════════════
// Ex — numeric extraction, comparison, evaluation helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Relative tolerance used when a comparison has to fall back to
/// floating-point evaluation (16-digit `evalf`).
const NUMERIC_REL_TOL: f64 = 1e-9;

/// Distance-based "clearly different" test for two complex f64 values.
fn clearly_different((ar, ai): (f64, f64), (br, bi): (f64, f64)) -> bool {
    let scale = 1.0_f64.max(ar.hypot(ai)).max(br.hypot(bi));
    (ar - br).hypot(ai - bi) > NUMERIC_REL_TOL * scale
}

/// SplitMix64 — tiny deterministic PRNG for sample points.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A "random" rational with numerator in `[-64, 64]` and denominator
    /// in `[1, 8]`, never zero.
    fn next_rational(&mut self) -> (i64, i64) {
        let q = (self.next_u64() % 8) as i64 + 1;
        let mut p = (self.next_u64() % 129) as i64 - 64;
        if p == 0 {
            p = 1;
        }
        (p, q)
    }
}

impl Ex {
    // ── Extraction ─────────────────────────────────────────────────────

    /// The exact value if this expression is a numeric literal.
    ///
    /// Returns `None` for anything that is not a plain number node (symbols,
    /// `pi`, `sqrt(2)`, unevaluated sums, …) — call
    /// [`eval`](Self::eval) first if you want constant folding.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use num_bigint::BigInt;
    /// use num_rational::Ratio;
    ///
    /// let ctx = Context::new();
    /// let r = ctx.rational(6, 4).as_rational().unwrap();
    /// assert_eq!(r, Ratio::new(BigInt::from(3), BigInt::from(2)));
    /// assert!(ctx.pi().as_rational().is_none());
    /// assert!((&ctx.int(2).sqrt() * &ctx.int(2).sqrt()).eval().as_rational().is_some());
    /// ```
    #[must_use]
    pub fn as_rational(&self) -> Option<Ratio<BigInt>> {
        let inner = self.inner.read();
        inner.arena.as_num(self.raw_id()).cloned()
    }

    /// The exact value if this expression is an integer literal.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use num_bigint::BigInt;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(ctx.int(-7).as_bigint(), Some(BigInt::from(-7)));
    /// assert_eq!(ctx.rational(1, 2).as_bigint(), None);
    /// ```
    #[must_use]
    pub fn as_bigint(&self) -> Option<BigInt> {
        let inner = self.inner.read();
        let r = inner.arena.as_num(self.raw_id())?;
        r.is_integer().then(|| r.numer().clone())
    }

    /// The value if this expression is an integer literal that fits in `i64`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(ctx.int(42).as_i64(), Some(42));
    /// assert_eq!(ctx.from_u64(u64::MAX).as_i64(), None);
    /// assert_eq!(ctx.symbol("n").as_i64(), None);
    /// ```
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        self.as_bigint()?.to_i64()
    }

    // ── Comparison ─────────────────────────────────────────────────────

    /// Three-valued numeric comparison of `self` and `other`.
    ///
    /// Decision procedure, in order:
    ///
    /// 1. Both are numeric literals → exact rational comparison.
    /// 2. `d = (self − other).eval()` is a literal → exact sign of `d`; if
    ///    `d` is `oo` / `-oo` → `Greater` / `Less`.
    /// 3. The assumption system knows the sign of `d` (e.g. `a − b` with
    ///    `a` positive and `b` negative; `x² + 1` for real `x`).
    /// 4. Both are constants (no free symbols) → 16-digit numeric
    ///    evaluation; decided only if the values differ by more than
    ///    `1e-9` relative and both are real.
    /// 5. Otherwise `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cmp::Ordering;
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(ctx.rational(1, 3).compare_numeric(&ctx.rational(1, 2)), Some(Ordering::Less));
    /// assert_eq!(ctx.pi().compare_numeric(&ctx.int(3)), Some(Ordering::Greater));
    /// assert_eq!(ctx.int(2).sqrt().compare_numeric(&ctx.rational(3, 2)), Some(Ordering::Less));
    ///
    /// let x = ctx.symbol("x");
    /// assert_eq!((&x + 1).compare_numeric(&x), Some(Ordering::Greater));
    /// assert_eq!(x.compare_numeric(&ctx.int(0)), None);
    ///
    /// let p = ctx.symbol_with("p", &[Assumption::Positive]);
    /// assert_eq!(p.compare_numeric(&ctx.int(0)), Some(Ordering::Greater));
    /// ```
    #[must_use]
    pub fn compare_numeric(&self, other: &Ex) -> Option<Ordering> {
        let other_id = self.checked_id(other);
        if self.raw_id() == other_id {
            return Some(Ordering::Equal);
        }
        // 1. Both literals.
        {
            let inner = self.inner.read();
            if let (Some(a), Some(b)) = (
                inner.arena.as_num(self.raw_id()),
                inner.arena.as_num(other_id),
            ) {
                return Some(a.cmp(b));
            }
        }
        // 2. Sign of the evaluated difference.
        let d = (self - other).eval();
        {
            let inner = d.inner.read();
            match inner.arena.node(d.raw_id()) {
                ExprNode::Num(nid) => {
                    let zero = Ratio::zero();
                    return Some(inner.arena.num(*nid).cmp(&zero));
                }
                ExprNode::Infinity => return Some(Ordering::Greater),
                ExprNode::NegInfinity => return Some(Ordering::Less),
                ExprNode::NaN | ExprNode::ComplexInfinity => return None,
                _ => {}
            }
        }
        // 3. Assumption system.
        if d.is_zero() == Some(true) {
            return Some(Ordering::Equal);
        }
        if d.is_positive() == Some(true) {
            return Some(Ordering::Greater);
        }
        if d.is_negative() == Some(true) {
            return Some(Ordering::Less);
        }
        // 4. Numeric fallback for constants.
        if d.is_constant()
            && let (Ok(a), Ok(b)) = (self.eval_complex64(), other.eval_complex64())
        {
            let scale = 1.0_f64.max(a.0.abs()).max(b.0.abs());
            if a.1.abs() > NUMERIC_REL_TOL * scale || b.1.abs() > NUMERIC_REL_TOL * scale {
                return None; // complex values are unordered
            }
            let diff = a.0 - b.0;
            if diff.abs() > NUMERIC_REL_TOL * scale {
                return Some(if diff > 0.0 {
                    Ordering::Greater
                } else {
                    Ordering::Less
                });
            }
        }
        None
    }

    /// Is `self < other`?  Three-valued; see
    /// [`compare_numeric`](Self::compare_numeric) for the decision procedure.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(ctx.int(1).is_less_than(&ctx.int(2)), Some(true));
    /// assert_eq!(ctx.pi().is_less_than(&ctx.int(3)), Some(false));
    /// let x = ctx.symbol("x");
    /// assert_eq!(x.is_less_than(&ctx.int(3)), None);
    /// // Assumptions help: x² ≥ 0 for real x, so x² < -1 is false.
    /// let r = ctx.symbol_with("r", &[Assumption::Real]);
    /// assert_eq!(r.powi(2).is_less_than(&ctx.int(-1)), Some(false));
    /// ```
    #[must_use]
    pub fn is_less_than(&self, other: &Ex) -> Option<bool> {
        match self.compare_numeric(other) {
            Some(Ordering::Less) => Some(true),
            Some(_) => Some(false),
            None => {
                // The assumption system may know "≥ 0" without knowing the sign.
                let d = self - other;
                if d.is_nonnegative() == Some(true) {
                    Some(false)
                } else {
                    None
                }
            }
        }
    }

    /// Is `self > other`?  Three-valued; see
    /// [`compare_numeric`](Self::compare_numeric).
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(ctx.e().is_greater_than(&ctx.int(2)), Some(true));
    /// assert_eq!(ctx.int(2).is_greater_than(&ctx.int(2)), Some(false));
    /// assert_eq!(ctx.symbol("x").is_greater_than(&ctx.int(0)), None);
    /// ```
    #[must_use]
    pub fn is_greater_than(&self, other: &Ex) -> Option<bool> {
        match self.compare_numeric(other) {
            Some(Ordering::Greater) => Some(true),
            Some(_) => Some(false),
            None => {
                let d = self - other;
                if d.is_nonpositive() == Some(true) {
                    Some(false)
                } else {
                    None
                }
            }
        }
    }

    /// Randomized equality test: evaluate both sides at `samples` random
    /// rational points and compare.
    ///
    /// * `Some(false)` — a concrete point was found where the two sides
    ///   differ.  When both sides fold to exact rationals at that point this
    ///   is a proof; when transcendental functions force floating-point
    ///   evaluation the sides differ by more than `1e-9` relative.
    /// * `Some(true)` — [`equals`](Self::equals) proved it symbolically, **or**
    ///   every sample agreed.  The latter is probabilistic: for polynomial
    ///   and rational identities the chance of a false positive is
    ///   negligible after a few samples, but no proof is produced.
    /// * `None` — no sample point could be evaluated (domain errors at
    ///   every point) and the symbolic test was inconclusive.
    ///
    /// Sample points are drawn from a fixed-seed generator keyed on the two
    /// expressions, so results are reproducible.  `samples == 0` is treated
    /// as `1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let lhs = (&x + 1).powi(3);
    /// let rhs = &x.powi(3) + &x.powi(2) * 3 + &x * 3 + 1;
    /// assert_eq!(lhs.probably_equal(&rhs, 5), Some(true));
    /// assert_eq!(x.probably_equal(&ctx.symbol("y"), 5), Some(false));
    /// assert_eq!(x.sin().probably_equal(&x.cos(), 5), Some(false));
    /// ```
    #[must_use]
    pub fn probably_equal(&self, other: &Ex, samples: usize) -> Option<bool> {
        let other_id = self.checked_id(other);
        if let Some(r) = self.equals(other) {
            return Some(r);
        }
        let diff = self - other;
        let syms = diff.free_symbols();
        if syms.is_empty() {
            // `equals` already tried the constant numeric test.
            return None;
        }
        let ctx = self.context();
        let seed = 0x5EED_0000_0000_0000_u64
            ^ (u64::from(self.raw_id().0) << 32)
            ^ u64::from(other_id.0)
            ^ (samples as u64).rotate_left(17);
        let mut rng = SplitMix64(seed);
        let samples = samples.max(1);
        let mut confirmed = 0usize;

        for _ in 0..samples {
            let vals: Vec<Ex> = syms
                .iter()
                .map(|_| {
                    let (p, q) = rng.next_rational();
                    ctx.rational(p, q)
                })
                .collect();
            let pairs: Vec<(&Ex, &Ex)> = syms.iter().zip(vals.iter()).collect();
            let a = self.subs_map(&pairs).eval();
            let b = other.subs_map(&pairs).eval();
            if a == b {
                confirmed += 1;
                continue;
            }
            // Exact path: both folded to rationals.
            if let (Some(ra), Some(rb)) = (a.as_rational(), b.as_rational()) {
                if ra != rb {
                    return Some(false);
                }
                confirmed += 1;
                continue;
            }
            // Numeric path.
            match (a.eval_complex64(), b.eval_complex64()) {
                (Ok(ca), Ok(cb)) => {
                    if clearly_different(ca, cb) {
                        return Some(false);
                    }
                    confirmed += 1;
                }
                _ => continue, // domain problem at this point — skip it
            }
        }
        (confirmed > 0).then_some(true)
    }

    // ── Evaluation ─────────────────────────────────────────────────────

    /// Substitute `(symbol, value)` pairs simultaneously and evaluate.
    ///
    /// Shorthand for `self.subs_map(pairs).eval()`.  The result is exact
    /// and may still be symbolic if not every symbol was bound.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let f = &x.powi(2) + &y;
    /// assert_eq!(format!("{}", f.eval_at(&[(&x, &ctx.int(3)), (&y, &ctx.rational(1, 2))])), "19/2");
    /// assert_eq!(format!("{}", f.eval_at(&[(&x, &ctx.pi())])), "y + pi^2");
    /// ```
    #[must_use]
    pub fn eval_at(&self, pairs: &[(&Ex, &Ex)]) -> Ex {
        self.subs_map(pairs).eval()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::assumptions::Assumption;

    fn ctx_x() -> (Context, Ex) {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        (ctx, x)
    }

    // ── Debug ──────────────────────────────────────────────────────────

    #[test]
    fn debug_prints_display_form() {
        let (ctx, x) = ctx_x();
        assert_eq!(format!("{:?}", &x + 1), "Ex(x + 1)");
        assert_eq!(format!("{:?}", ctx.int(1).gt(&x)), "BoolEx(1 > x)");
        assert_eq!(format!("{:?}", ctx.empty_set()), "SetEx(EmptySet)");
    }

    // ── Scalar operators ───────────────────────────────────────────────

    #[test]
    fn f64_operands_are_exact() {
        let (ctx, x) = ctx_x();
        assert_eq!(&x * 0.5, &x * &ctx.rational(1, 2));
        assert_eq!(2.5 + &x, &x + &ctx.rational(5, 2));
        assert_eq!(&x - 0.25, &x - &ctx.rational(1, 4));
        assert_eq!(1.0 / &x, &ctx.one() / &x);
        assert_eq!(format!("{}", 1.0 / &x), "1/x");
        let tenth = &x * 0.1;
        assert!(format!("{tenth}").contains("3602879701896397"), "{tenth}");
    }

    #[test]
    fn f64_special_values() {
        let (ctx, x) = ctx_x();
        assert_eq!(format!("{}", &x + f64::NAN), "nan");
        assert_eq!(&x * f64::INFINITY, &x * &ctx.infinity());
        assert_eq!(&x + f64::NEG_INFINITY, &x + &ctx.neg_infinity());
    }

    #[test]
    fn integer_scalar_types() {
        let (_ctx, x) = ctx_x();
        let _ctx = &_ctx;
        assert_eq!(format!("{}", &x + 1i32), "x + 1");
        assert_eq!(format!("{}", &x * 2u32), "2*x");
        assert_eq!(format!("{}", 2 * &x), "2*x");
        assert_eq!(format!("{}", 2.0 * &x), "2*x");
        assert_eq!(format!("{}", &x - 3u64), "x - 3");
        assert_eq!(format!("{}", &x + u64::MAX), "x + 18446744073709551615");
        assert_eq!(&x / 2i64, &x * &_ctx.rational(1, 2));
        assert_eq!(&x / 2i64, &x / 2i32);
        assert_eq!(&x + 5i128, &x + 5);
    }

    #[test]
    fn bigint_and_ratio_operands() {
        let (_ctx, x) = ctx_x();
        let _ctx = &_ctx;
        let big = BigInt::from(10).pow(30);
        assert_eq!(
            format!("{}", &x * big.clone()),
            "1000000000000000000000000000000*x"
        );
        assert_eq!(
            format!("{}", big - &x),
            "-x + 1000000000000000000000000000000"
        );
        let r = Ratio::new(BigInt::from(2), BigInt::from(6));
        assert_eq!(format!("{}", &x + r.clone()), "x + 1/3");
        assert_eq!(r / &x, &_ctx.rational(1, 3) / &x);
    }

    #[test]
    fn compound_assignment() {
        let (ctx, x) = ctx_x();
        let mut e = x.clone();
        e += 1;
        e *= &x;
        e -= 0.5;
        e /= 2i64;
        assert_eq!(e, (&(&x + 1) * &x - &ctx.rational(1, 2)) / 2);
        let mut f = ctx.one();
        f += x.clone();
        f *= ctx.int(3);
        assert_eq!(format!("{f}"), "3*x + 3");
    }

    // ── Sum / Product ──────────────────────────────────────────────────

    #[test]
    fn sum_and_product_nonempty() {
        let ctx = Context::new();
        let v: Vec<Ex> = (1..=4).map(|n| ctx.int(n)).collect();
        assert_eq!(format!("{}", v.iter().sum::<Ex>()), "10");
        assert_eq!(format!("{}", v.iter().product::<Ex>()), "24");
        assert_eq!(format!("{}", v.clone().into_iter().sum::<Ex>()), "10");
        assert_eq!(format!("{}", v.into_iter().product::<Ex>()), "24");
    }

    #[test]
    #[should_panic(expected = "empty iterator")]
    fn sum_empty_panics_with_clear_message() {
        let _: Ex = Vec::<Ex>::new().into_iter().sum();
    }

    #[test]
    #[should_panic(expected = "empty iterator")]
    fn product_empty_refs_panics() {
        let v: Vec<Ex> = Vec::new();
        let _: Ex = v.iter().product();
    }

    #[test]
    fn option_sum_product() {
        let ctx = Context::new();
        let v: Vec<Ex> = (1..=3).map(|n| ctx.int(n)).collect();
        let s: Option<Ex> = v.iter().sum();
        assert_eq!(format!("{}", s.unwrap()), "6");
        let p: Option<Ex> = v.into_iter().product();
        assert_eq!(format!("{}", p.unwrap()), "6");
        let none_s: Option<Ex> = Vec::<Ex>::new().into_iter().sum();
        let none_p: Option<Ex> = Vec::<Ex>::new().iter().product();
        assert!(none_s.is_none() && none_p.is_none());
    }

    #[test]
    fn context_sum_product_empty_and_mixed_items() {
        let ctx = Context::new();
        assert_eq!(format!("{}", ctx.sum(Vec::<Ex>::new())), "0");
        assert_eq!(format!("{}", ctx.product(Vec::<Ex>::new())), "1");
        let xs = ctx.symbols(&["a", "b"]);
        assert_eq!(format!("{}", ctx.sum(&xs)), "a + b");
        assert_eq!(format!("{}", ctx.product(xs.iter())), "a*b");
        assert_eq!(format!("{}", ctx.sum(xs)), "a + b");
    }

    #[test]
    #[should_panic(expected = "different contexts")]
    fn context_sum_rejects_foreign_items() {
        let a = Context::new();
        let b = Context::new();
        let _ = a.sum([b.int(1)]);
    }

    // ── Context ingestion ──────────────────────────────────────────────

    #[test]
    fn from_f64_exact_dyadic() {
        let ctx = Context::new();
        assert_eq!(format!("{}", ctx.from_f64(0.75).unwrap()), "3/4");
        assert_eq!(format!("{}", ctx.from_f64(-0.0).unwrap()), "0");
        assert_eq!(
            format!("{}", ctx.from_f64(1e20).unwrap()),
            "100000000000000000000"
        );
        assert_eq!(
            format!("{}", ctx.from_f64(0.1).unwrap()),
            "3602879701896397/36028797018963968"
        );
        assert_eq!(ctx.from_f64(f64::NEG_INFINITY).unwrap(), ctx.neg_infinity());
        assert!(matches!(
            ctx.from_f64(f64::NAN),
            Err(SymplexError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn from_f64_roundtrips_bit_exactly() {
        let ctx = Context::new();
        for v in [
            0.1,
            1.0 / 3.0,
            123.456,
            -9.87e-7,
            f64::MAX,
            f64::MIN_POSITIVE,
        ] {
            let e = ctx.from_f64(v).unwrap();
            let r = e.as_rational().unwrap();
            assert_eq!(numeric::ratio_to_f64(&r).unwrap().to_bits(), v.to_bits());
        }
    }

    #[test]
    fn from_f64_approx_and_nice() {
        let ctx = Context::new();
        assert_eq!(
            format!("{}", ctx.from_f64_approx(0.1, 100).unwrap()),
            "1/10"
        );
        assert_eq!(
            format!("{}", ctx.from_f64_approx(0.3, 1_000_000).unwrap()),
            "3/10"
        );
        assert_eq!(format!("{}", ctx.from_f64_approx(2.0, 1).unwrap()), "2");
        assert_eq!(format!("{}", ctx.from_f64_approx(0.49, 1).unwrap()), "0");
        assert_eq!(format!("{}", ctx.from_f64_nice(1.0 / 3.0).unwrap()), "1/3");
        assert_eq!(format!("{}", ctx.from_f64_nice(-2.5).unwrap()), "-5/2");
        assert_eq!(
            format!("{}", ctx.from_f64_nice(0.123456789).unwrap()),
            "123456789/1000000000"
        );
        assert_eq!(ctx.from_f64_nice(f64::INFINITY).unwrap(), ctx.infinity());
        assert!(ctx.from_f64_approx(0.5, 0).is_err());
        assert!(ctx.from_f64_nice(f64::NAN).is_err());
    }

    #[test]
    fn from_bigint_ratio_i128_u64() {
        let ctx = Context::new();
        assert_eq!(format!("{}", ctx.from_bigint(BigInt::from(-5))), "-5");
        assert_eq!(
            format!(
                "{}",
                ctx.from_ratio(Ratio::new(BigInt::from(-10), BigInt::from(4)))
            ),
            "-5/2"
        );
        assert_eq!(format!("{}", ctx.from_i128(7)), "7");
        assert_eq!(
            format!("{}", ctx.from_i128(i128::MAX)),
            "170141183460469231731687303715884105727"
        );
        assert_eq!(format!("{}", ctx.from_u64(0)), "0");
        assert_eq!(ctx.from_u64(9), ctx.int(9));
        // Same value ⇒ same node (hash-consing through the new entry points).
        assert_eq!(ctx.from_f64(0.5).unwrap(), ctx.rational(1, 2));
        assert_eq!(ctx.from_bigint(BigInt::from(3)), ctx.int(3));
    }

    #[test]
    fn from_ratio_raw_zero_denominator_does_not_panic() {
        let ctx = Context::new();
        let zoo = ctx.from_ratio(Ratio::new_raw(BigInt::from(1), BigInt::zero()));
        assert_eq!(zoo, ctx.complex_infinity());
        let nan = ctx.from_ratio(Ratio::new_raw(BigInt::zero(), BigInt::zero()));
        assert_eq!(nan, ctx.nan());
        let unreduced = ctx.from_ratio(Ratio::new_raw(BigInt::from(4), BigInt::from(-8)));
        assert_eq!(unreduced, ctx.rational(-1, 2));
    }

    #[test]
    fn rational_str_parsing() {
        let ctx = Context::new();
        assert_eq!(ctx.rational_str("3/4").unwrap(), ctx.rational(3, 4));
        assert_eq!(ctx.rational_str("-3 / 4").unwrap(), ctx.rational(-3, 4));
        assert_eq!(ctx.rational_str("10/-4").unwrap(), ctx.rational(-5, 2));
        assert_eq!(ctx.rational_str("7").unwrap(), ctx.int(7));
        assert_eq!(
            format!(
                "{}",
                ctx.rational_str("123456789012345678901234567890/3")
                    .unwrap()
            ),
            "41152263004115226300411522630"
        );
        for bad in ["", "1/0", "a/b", "1/2/3", "1.5", "/2", "2/"] {
            assert!(ctx.rational_str(bad).is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn decimal_str_parsing() {
        let ctx = Context::new();
        let cases = [
            ("0.1", "1/10"),
            ("0.25", "1/4"),
            ("-0.5", "-1/2"),
            ("+1.5", "3/2"),
            ("3", "3"),
            ("3.", "3"),
            (".5", "1/2"),
            ("1e3", "1000"),
            ("1E-3", "1/1000"),
            ("2.5e-1", "1/4"),
            ("-1.25e+2", "-125"),
            ("0.000", "0"),
            ("  42.0  ", "42"),
            ("0.30000000000000004", "7500000000000001/25000000000000000"),
        ];
        for (s, expect) in cases {
            assert_eq!(
                format!("{}", ctx.decimal_str(s).unwrap()),
                expect,
                "input {s:?}"
            );
        }
        for bad in [
            "", ".", "-", "e5", "1e", "1e+", "1/2", "abc", "1.2.3", "1 2", "--1", "0x10",
        ] {
            assert!(ctx.decimal_str(bad).is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn decimal_str_matches_parse_for_decimals() {
        let ctx = Context::new();
        for s in ["0.1", "3.14159", "123.456", "0.000001", "2.5"] {
            assert_eq!(ctx.decimal_str(s).unwrap(), ctx.parse(s).unwrap(), "{s}");
        }
    }

    #[test]
    fn complex_symbols_apply() {
        let ctx = Context::new();
        let z = ctx.complex(&ctx.int(1), &ctx.int(2));
        assert_eq!(z, &ctx.int(1) + &ctx.i_unit() * 2);
        let (re, im) = z.as_real_imag();
        assert_eq!((format!("{re}"), format!("{im}")), ("1".into(), "2".into()));
        let zero_im = ctx.complex(&ctx.int(3), &ctx.zero());
        assert_eq!(zero_im, ctx.int(3));

        let v = ctx.symbols(&["p", "q"]);
        assert_eq!(v, vec![ctx.symbol("p"), ctx.symbol("q")]);
        let idx = ctx.symbols_indexed("t", 2);
        assert_eq!(idx, vec![ctx.symbol("t0"), ctx.symbol("t1")]);

        let x = ctx.symbol("x");
        let f = ctx.apply("f", &[&x, &ctx.int(2)]);
        assert_eq!(format!("{f}"), "f(x, 2)");
        assert_eq!(f.expr_type(), crate::api::expr::ExprType::Apply);
        assert_eq!(f.eval(), f);
        assert_eq!(f.simplify(), f);
        assert!(!f.has_unevaluated());
        assert!(f.diff(&x).has_unevaluated());
        assert!(matches!(
            f.compile(&["x"]),
            Err(SymplexError::NotImplemented(_))
        ));
        // Owned-slice form and hash-consing.
        let f2 = ctx.apply("f", &[x.clone(), ctx.int(2)]);
        assert_eq!(f, f2);
        let g0 = ctx.apply::<Ex>("g", &[]);
        assert_eq!(format!("{g0}"), "g()");
    }

    #[test]
    #[should_panic(expected = "different contexts")]
    fn apply_rejects_foreign_argument() {
        let a = Context::new();
        let b = Context::new();
        let _ = a.apply("f", &[b.symbol("x")]);
    }

    // ── ToEx ───────────────────────────────────────────────────────────

    #[test]
    fn to_ex_conversions() {
        let ctx = Context::new();
        assert_eq!(7i32.to_ex(&ctx), ctx.int(7));
        assert_eq!(7i64.to_ex(&ctx), ctx.int(7));
        assert_eq!(7u32.to_ex(&ctx), ctx.int(7));
        assert_eq!(7u64.to_ex(&ctx), ctx.int(7));
        assert_eq!(7i128.to_ex(&ctx), ctx.int(7));
        assert_eq!(0.25f64.to_ex(&ctx), ctx.rational(1, 4));
        assert_eq!(f64::INFINITY.to_ex(&ctx), ctx.infinity());
        assert_eq!(f64::NAN.to_ex(&ctx), ctx.nan());
        assert_eq!(BigInt::from(9).to_ex(&ctx), ctx.int(9));
        assert_eq!(
            Ratio::new(BigInt::from(1), BigInt::from(3)).to_ex(&ctx),
            ctx.rational(1, 3)
        );
        let x = ctx.symbol("x");
        assert_eq!(x.to_ex(&ctx), x);
        let xr: &Ex = &x;
        assert_eq!(xr.to_ex(&ctx), x);
    }

    #[test]
    #[should_panic(expected = "different contexts")]
    fn to_ex_rejects_foreign_expression() {
        let a = Context::new();
        let b = Context::new();
        let _ = a.symbol("x").to_ex(&b);
    }

    // ── Extraction ─────────────────────────────────────────────────────

    #[test]
    fn as_rational_bigint_i64() {
        let ctx = Context::new();
        assert_eq!(
            ctx.rational(-3, 6).as_rational(),
            Some(Ratio::new(BigInt::from(-1), BigInt::from(2)))
        );
        assert_eq!(ctx.int(5).as_bigint(), Some(BigInt::from(5)));
        assert_eq!(ctx.rational(1, 2).as_bigint(), None);
        assert_eq!(ctx.int(-5).as_i64(), Some(-5));
        assert_eq!(ctx.from_i128(i128::MAX).as_i64(), None);
        assert_eq!(ctx.pi().as_i64(), None);
        assert_eq!(ctx.symbol("x").as_rational(), None);
    }

    // ── compare_numeric & friends ──────────────────────────────────────

    #[test]
    fn compare_numeric_exact_literals() {
        let ctx = Context::new();
        assert_eq!(
            ctx.int(1).compare_numeric(&ctx.int(2)),
            Some(Ordering::Less)
        );
        assert_eq!(
            ctx.rational(7, 3).compare_numeric(&ctx.rational(14, 6)),
            Some(Ordering::Equal)
        );
        assert_eq!(
            ctx.int(-1).compare_numeric(&ctx.rational(-3, 2)),
            Some(Ordering::Greater)
        );
    }

    #[test]
    fn compare_numeric_constants() {
        let ctx = Context::new();
        assert_eq!(
            ctx.pi().compare_numeric(&ctx.int(3)),
            Some(Ordering::Greater)
        );
        assert_eq!(ctx.pi().compare_numeric(&ctx.int(4)), Some(Ordering::Less));
        assert_eq!(
            ctx.int(2).sqrt().compare_numeric(&ctx.rational(141, 100)),
            Some(Ordering::Greater)
        );
        assert_eq!(
            (&ctx.int(2).sqrt() * &ctx.int(2).sqrt()).compare_numeric(&ctx.int(2)),
            Some(Ordering::Equal)
        );
        assert_eq!(
            ctx.infinity().compare_numeric(&ctx.int(10)),
            Some(Ordering::Greater)
        );
        assert_eq!(
            ctx.neg_infinity().compare_numeric(&ctx.int(10)),
            Some(Ordering::Less)
        );
        // Complex values are unordered.
        assert_eq!(ctx.i_unit().compare_numeric(&ctx.int(0)), None);
    }

    #[test]
    fn compare_numeric_symbolic() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        assert_eq!((&x + 1).compare_numeric(&x), Some(Ordering::Greater));
        assert_eq!((&x - 2).compare_numeric(&x), Some(Ordering::Less));
        assert_eq!(x.compare_numeric(&x), Some(Ordering::Equal));
        assert_eq!(x.compare_numeric(&ctx.int(1)), None);
        let p = ctx.symbol_with("p", &[Assumption::Positive]);
        let n = ctx.symbol_with("n", &[Assumption::Negative]);
        assert_eq!(p.compare_numeric(&n), Some(Ordering::Greater));
        assert_eq!(n.compare_numeric(&ctx.zero()), Some(Ordering::Less));
        let r = ctx.symbol_with("r", &[Assumption::Real]);
        assert_eq!(
            (&r.powi(2) + 1).compare_numeric(&ctx.zero()),
            Some(Ordering::Greater)
        );
    }

    #[test]
    fn is_less_greater_than() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        assert_eq!(ctx.int(1).is_less_than(&ctx.int(2)), Some(true));
        assert_eq!(ctx.int(2).is_less_than(&ctx.int(2)), Some(false));
        assert_eq!(ctx.int(3).is_greater_than(&ctx.int(2)), Some(true));
        assert_eq!(x.is_less_than(&ctx.int(2)), None);
        assert_eq!(x.is_greater_than(&ctx.int(2)), None);
        let r = ctx.symbol_with("r", &[Assumption::Real]);
        // r² ≥ 0: not < 0 (known), but > 0 unknown (could be 0).
        assert_eq!(r.powi(2).is_less_than(&ctx.zero()), Some(false));
        assert_eq!(r.powi(2).is_greater_than(&ctx.zero()), None);
        assert_eq!((&r.powi(2) + 1).is_greater_than(&ctx.zero()), Some(true));
    }

    // ── probably_equal ─────────────────────────────────────────────────

    #[test]
    fn probably_equal_polynomial_identity() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let y = ctx.symbol("y");
        let lhs = (&x + &y).powi(2);
        let rhs = &x.powi(2) + &x * &y * 2 + &y.powi(2);
        assert_eq!(lhs.probably_equal(&rhs, 3), Some(true));
        let wrong = &x.powi(2) + &x * &y * 2 + &y.powi(2) + 1;
        assert_eq!(lhs.probably_equal(&wrong, 3), Some(false));
    }

    #[test]
    fn probably_equal_distinguishes_symbols_and_functions() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let y = ctx.symbol("y");
        assert_eq!(x.probably_equal(&y, 4), Some(false));
        assert_eq!(x.probably_equal(&(&x * 2), 4), Some(false));
        assert_eq!(x.sin().probably_equal(&x.cos(), 4), Some(false));
        assert_eq!(x.sin().probably_equal(&x.sin(), 4), Some(true));
    }

    #[test]
    fn probably_equal_transcendental_identity() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        // cos(2x) = 1 - 2 sin²(x) — needs floating-point evaluation.
        let lhs = (&x * 2).cos();
        let rhs = &ctx.int(1) - &x.sin().powi(2) * 2;
        assert_eq!(lhs.probably_equal(&rhs, 3), Some(true));
        // Zero samples is treated as one.
        assert_eq!(lhs.probably_equal(&rhs, 0), Some(true));
    }

    #[test]
    fn probably_equal_constants_defer_to_equals() {
        let ctx = Context::new();
        assert_eq!(ctx.int(1).probably_equal(&ctx.int(1), 3), Some(true));
        assert_eq!(ctx.int(1).probably_equal(&ctx.int(2), 3), Some(false));
        assert_eq!(
            ctx.pi().probably_equal(&ctx.rational(22, 7), 3),
            Some(false)
        );
    }

    // ── eval_at ────────────────────────────────────────────────────────

    #[test]
    fn eval_at_substitutes_simultaneously() {
        let ctx = Context::new();
        let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
        let f = &x + &y * 10;
        // Swap: simultaneous substitution must not chain.
        assert_eq!(format!("{}", f.eval_at(&[(&x, &y), (&y, &x)])), "10*x + y");
        assert_eq!(
            format!("{}", f.eval_at(&[(&x, &ctx.int(1)), (&y, &ctx.int(2))])),
            "21"
        );
        assert_eq!(format!("{}", x.sin().eval_at(&[(&x, &ctx.zero())])), "0");
    }

    // ── parse_decimal_exact edge cases ─────────────────────────────────

    #[test]
    fn parse_decimal_exact_large_exponent() {
        let r = parse_decimal_exact("1e30").unwrap();
        assert_eq!(*r.numer(), BigInt::from(10).pow(30));
        let r = parse_decimal_exact("1e-30").unwrap();
        assert_eq!(*r.denom(), BigInt::from(10).pow(30));
        assert!(parse_decimal_exact("1e99999999999999999999").is_none());
    }
}
