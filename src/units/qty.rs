//! Generic dimensioned quantity wrapper.
//!
//! [`Qty<D>`] pairs a symbolic [`Ex`](crate::api::expr::Ex) expression with a compile-time dimension
//! `D = Dim<L, M, T, I, Th, N, J>`. Arithmetic is forwarded to `Ex` while
//! typenum enforces dimensional correctness:
//!
//! - **Add / Sub**: require identical dimensions ([`SameDim`] bound)
//! - **Mul**: exponents add (`Sum`)
//! - **Div**: exponents subtract (`Diff`)
//! - **Neg / scalar**: preserve dimension
//!
//! All operators are implemented for every ownership combination
//! (`owned × owned`, `owned × &`, `& × owned`, `& × &`) so callers
//! rarely need explicit `.clone()`.

#![allow(clippy::suspicious_arithmetic_impl)]

use std::fmt;
use std::marker::PhantomData;
use std::ops;

use typenum::operator_aliases::{Diff, Sum};

use crate::prelude::Ex;

use super::dim::{Dim, DimName};

// ═══════════════════════════════════════════════════════════════════════════
// IntoEx — accept both Ex and &Ex without explicit .clone()
// ═══════════════════════════════════════════════════════════════════════════

/// Trait for types that can be converted to an owned `Ex`.
/// Allows `from_ex` to accept both `Ex` and `&Ex` without explicit `.clone()`.
pub trait IntoEx {
    /// Convert to an owned `Ex`.
    fn into_ex(self) -> Ex;
}

impl IntoEx for Ex {
    fn into_ex(self) -> Ex {
        self
    }
}

impl IntoEx for &Ex {
    fn into_ex(self) -> Ex {
        self.clone()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Qty<D> — the core dimensioned wrapper
// ═══════════════════════════════════════════════════════════════════════════

/// A symbolic expression with compile-time dimension tracking.
///
/// `D` encodes the SI dimension exponents via typenum type-level integers.
/// The inner value is an [`Ex`] symbolic expression.
///
/// For common dimensions, prefer named newtypes like [`Force`](super::Force)
/// which produce better compiler error messages.
#[derive(Clone)]
pub struct Qty<D> {
    pub(crate) inner: Ex,
    pub(crate) _dim: PhantomData<D>,
}

// ── Inherent methods ───────────────────────────────────────────────────

impl<D> Qty<D> {
    /// Wrap an [`Ex`] with dimension `D`.
    ///
    /// The caller asserts that the expression genuinely has dimension `D`.
    #[inline]
    pub fn from_ex(ex: impl IntoEx) -> Self {
        Qty {
            inner: ex.into_ex(),
            _dim: PhantomData,
        }
    }

    /// Escape hatch: consume the wrapper and return the raw [`Ex`],
    /// dropping all dimension information.
    #[inline]
    pub fn into_inner(self) -> Ex {
        self.inner
    }

    /// Borrow the inner [`Ex`] without consuming the wrapper.
    #[inline]
    pub fn inner(&self) -> &Ex {
        &self.inner
    }

    /// Transform the inner expression while preserving the dimension.
    ///
    /// ```ignore
    /// let f2 = force.map(|ex| ex.simplify());
    /// ```
    #[inline]
    pub fn map(self, f: impl FnOnce(Ex) -> Ex) -> Self {
        Qty::from_ex(f(self.inner))
    }
}

// ── Dimension-preserving symbolic manipulation ─────────────────────────

impl<D> Qty<D> {
    /// Simplify the inner expression (single-pass rewrite rules).
    pub fn simplify(&self) -> Self {
        Self::from_ex(self.inner.simplify())
    }

    /// Algebraic expansion.
    pub fn expand(&self) -> Self {
        Self::from_ex(self.inner.expand())
    }

    /// Evaluate special values (sin(π)→0, etc).
    pub fn eval(&self) -> Self {
        Self::from_ex(self.inner.eval())
    }

    /// Substitute a variable with a value, preserving dimension.
    pub fn subs(&self, var: &impl AsRef<Ex>, val: &impl AsRef<Ex>) -> Self {
        Self::from_ex(self.inner.subs(var.as_ref(), val.as_ref()))
    }

    /// Trigonometric simplification.
    pub fn simplify_trig(&self) -> Self {
        Self::from_ex(self.inner.simplify_trig())
    }

    /// Power simplification.
    pub fn simplify_powers(&self) -> Self {
        Self::from_ex(self.inner.simplify_powers())
    }

    /// Rational simplification (cancel + together).
    pub fn simplify_rational(&self) -> Self {
        Self::from_ex(self.inner.simplify_rational())
    }

    /// Trig expansion (sin(a+b) → sin(a)cos(b)+cos(a)sin(b)).
    pub fn expand_trig(&self) -> Self {
        Self::from_ex(self.inner.expand_trig())
    }

    /// Log expansion (ln(ab) → ln(a)+ln(b)).
    pub fn expand_log(&self) -> Self {
        Self::from_ex(self.inner.expand_log())
    }

    /// Combine logarithms (ln(a)+ln(b) → ln(ab)).
    pub fn log_combine(&self) -> Self {
        Self::from_ex(self.inner.log_combine())
    }

    /// Trig product-to-sum.
    pub fn trig_combine(&self) -> Self {
        Self::from_ex(self.inner.trig_combine())
    }

    /// Factor a polynomial.
    pub fn factor(&self, var: &impl AsRef<Ex>) -> Self {
        Self::from_ex(self.inner.factor(var.as_ref()))
    }

    /// Collect by variable.
    pub fn collect(&self, var: &impl AsRef<Ex>) -> Self {
        Self::from_ex(self.inner.collect(var.as_ref()))
    }

    /// Cancel common polynomial factors.
    pub fn cancel(&self, var: &Ex) -> Self {
        Self::from_ex(self.inner.cancel(var))
    }

    /// Combine fractions over common denominator.
    pub fn together(&self) -> Self {
        Self::from_ex(self.inner.together())
    }

    /// Partial fraction decomposition.
    pub fn partial_fractions(&self, var: &impl AsRef<Ex>) -> Self {
        Self::from_ex(self.inner.partial_fractions(var.as_ref()))
    }

    /// Rationalize the denominator.
    pub fn rationalize_denom(&self) -> Self {
        Self::from_ex(self.inner.rationalize_denom())
    }

    // ── Calculus (returns raw Ex — user wraps in correct output type) ──

    /// Differentiate with respect to a variable. Returns raw `Ex`.
    /// Wrap the result in the appropriate output dimension type.
    pub fn diff(&self, var: &impl AsRef<Ex>) -> Ex {
        self.inner.diff(var.as_ref())
    }

    /// Integrate with respect to a variable. Returns raw `Ex`.
    pub fn integrate(&self, var: &impl AsRef<Ex>) -> Ex {
        self.inner.integrate(var.as_ref())
    }

    // ── Queries (dimension-independent) ──

    /// LaTeX rendering of the inner expression.
    pub fn to_latex(&self) -> String {
        self.inner.to_latex()
    }

    /// Free symbols in the expression.
    pub fn free_symbols(&self) -> Vec<Ex> {
        self.inner.free_symbols()
    }

    /// Check if expression contains a subexpression.
    pub fn contains(&self, other: &impl AsRef<Ex>) -> bool {
        self.inner.contains(other.as_ref())
    }

    /// Number of additive terms.
    pub fn term_count(&self) -> usize {
        self.inner.term_count()
    }

    /// Operation count (for complexity measure).
    pub fn count_ops(&self) -> usize {
        self.inner.count_ops()
    }

    /// Check if structurally zero.
    pub fn is_zero(&self) -> Option<bool> {
        self.inner.is_zero()
    }

    // ── Numerical evaluation ──

    /// Evaluate to f64 (all symbols must be eliminated first).
    pub fn eval_f64(&self) -> Result<f64, crate::base::errors::SymplexError> {
        self.inner.eval_f64()
    }

    /// Evaluate with integer substitutions.
    pub fn eval_f64_with(
        &self,
        subs: &[(&Ex, i64)],
    ) -> Result<f64, crate::base::errors::SymplexError> {
        self.inner.eval_f64_with(subs)
    }

    /// Arbitrary-precision decimal evaluation.
    pub fn eval_decimal(&self, digits: u32) -> Result<String, crate::base::errors::SymplexError> {
        self.inner.eval_decimal(digits)
    }
}

impl<D> AsRef<Ex> for Qty<D> {
    fn as_ref(&self) -> &Ex {
        &self.inner
    }
}

// ── DimName-gated methods ──────────────────────────────────────────────

impl<D: DimName> Qty<D> {
    /// Human-readable dimension name (e.g. `"Force"`, `"Voltage"`).
    #[inline]
    pub fn dim_name(&self) -> &'static str {
        D::dim_name()
    }

    /// Short dimension symbol (e.g. `"N"`, `"V"`).
    #[inline]
    pub fn dim_symbol(&self) -> &'static str {
        D::dim_symbol()
    }
}

impl<D> fmt::Display for Qty<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl<D> fmt::Debug for Qty<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Qty({})", self.inner)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SameDim trait — gates Add / Sub
// ═══════════════════════════════════════════════════════════════════════════

/// Marker trait asserting that two dimension types are identical.
///
/// The blanket impl below makes every `Dim<…>` satisfy `SameDim` with
/// itself.  When the dimensions differ, rustc reports a clear error
/// thanks to `#[diagnostic::on_unimplemented]`.
#[diagnostic::on_unimplemented(
    message = "cannot add or subtract quantities with different physical dimensions",
    label = "incompatible physical dimension",
    note = "addition and subtraction require both sides to have the same dimension",
    note = "use .into_inner() to drop dimension tracking"
)]
pub trait SameDim<Rhs> {}

impl<L, M, T, I, Th, N, J> SameDim<Dim<L, M, T, I, Th, N, J>> for Dim<L, M, T, I, Th, N, J> {}

// ═══════════════════════════════════════════════════════════════════════════
// FromDimExpr trait — used by the dim!(ctx, ) proc macro
// ═══════════════════════════════════════════════════════════════════════════

/// Trait for converting a `Qty<D>` to a named type with compile-time dimension verification.
///
/// Used by the [`dim!`](crate::dim) macro. When the computed dimension doesn't match the target type,
/// the compiler produces a clear error message via `#[diagnostic::on_unimplemented]`.
#[diagnostic::on_unimplemented(
    message = "dimension mismatch: expression does not produce `{Self}`",
    label = "wrong physical dimension",
    note = "the arithmetic in your dim!(ctx, ) expression produces a different dimension than `{Self}`",
    note = "check that your factors multiply/divide to the correct physical dimension"
)]
pub trait FromDimExpr<D> {
    /// Convert from a generic Qty with the matching dimension.
    fn from_dim_expr(qty: Qty<D>) -> Self;
}

// ═══════════════════════════════════════════════════════════════════════════
// Add — all four ownership combos
// ═══════════════════════════════════════════════════════════════════════════

impl<A, B> ops::Add<Qty<B>> for Qty<A>
where
    A: SameDim<B>,
{
    type Output = Qty<A>;

    #[inline]
    fn add(self, rhs: Qty<B>) -> Qty<A> {
        Qty::from_ex(&self.inner + &rhs.inner)
    }
}

impl<A, B> ops::Add<&Qty<B>> for Qty<A>
where
    A: SameDim<B>,
{
    type Output = Qty<A>;

    #[inline]
    fn add(self, rhs: &Qty<B>) -> Qty<A> {
        Qty::from_ex(&self.inner + &rhs.inner)
    }
}

impl<A, B> ops::Add<Qty<B>> for &Qty<A>
where
    A: SameDim<B>,
{
    type Output = Qty<A>;

    #[inline]
    fn add(self, rhs: Qty<B>) -> Qty<A> {
        Qty::from_ex(&self.inner + &rhs.inner)
    }
}

impl<A, B> ops::Add<&Qty<B>> for &Qty<A>
where
    A: SameDim<B>,
{
    type Output = Qty<A>;

    #[inline]
    fn add(self, rhs: &Qty<B>) -> Qty<A> {
        Qty::from_ex(&self.inner + &rhs.inner)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sub — all four ownership combos
// ═══════════════════════════════════════════════════════════════════════════

impl<A, B> ops::Sub<Qty<B>> for Qty<A>
where
    A: SameDim<B>,
{
    type Output = Qty<A>;

    #[inline]
    fn sub(self, rhs: Qty<B>) -> Qty<A> {
        Qty::from_ex(&self.inner - &rhs.inner)
    }
}

impl<A, B> ops::Sub<&Qty<B>> for Qty<A>
where
    A: SameDim<B>,
{
    type Output = Qty<A>;

    #[inline]
    fn sub(self, rhs: &Qty<B>) -> Qty<A> {
        Qty::from_ex(&self.inner - &rhs.inner)
    }
}

impl<A, B> ops::Sub<Qty<B>> for &Qty<A>
where
    A: SameDim<B>,
{
    type Output = Qty<A>;

    #[inline]
    fn sub(self, rhs: Qty<B>) -> Qty<A> {
        Qty::from_ex(&self.inner - &rhs.inner)
    }
}

impl<A, B> ops::Sub<&Qty<B>> for &Qty<A>
where
    A: SameDim<B>,
{
    type Output = Qty<A>;

    #[inline]
    fn sub(self, rhs: &Qty<B>) -> Qty<A> {
        Qty::from_ex(&self.inner - &rhs.inner)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Neg — preserve dimension
// ═══════════════════════════════════════════════════════════════════════════

impl<D> ops::Neg for Qty<D> {
    type Output = Qty<D>;

    #[inline]
    fn neg(self) -> Qty<D> {
        Qty::from_ex(-&self.inner)
    }
}

impl<D> ops::Neg for &Qty<D> {
    type Output = Qty<D>;

    #[inline]
    fn neg(self) -> Qty<D> {
        Qty::from_ex(-&self.inner)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Mul — blanket dimension addition, all four ownership combos
// ═══════════════════════════════════════════════════════════════════════════

impl<L1, M1, T1, I1, Th1, N1x, J1, L2, M2, T2, I2, Th2, N2x, J2>
    ops::Mul<Qty<Dim<L2, M2, T2, I2, Th2, N2x, J2>>> for Qty<Dim<L1, M1, T1, I1, Th1, N1x, J1>>
where
    L1: ops::Add<L2>,
    M1: ops::Add<M2>,
    T1: ops::Add<T2>,
    I1: ops::Add<I2>,
    Th1: ops::Add<Th2>,
    N1x: ops::Add<N2x>,
    J1: ops::Add<J2>,
{
    type Output = Qty<
        Dim<
            Sum<L1, L2>,
            Sum<M1, M2>,
            Sum<T1, T2>,
            Sum<I1, I2>,
            Sum<Th1, Th2>,
            Sum<N1x, N2x>,
            Sum<J1, J2>,
        >,
    >;

    #[inline]
    fn mul(self, rhs: Qty<Dim<L2, M2, T2, I2, Th2, N2x, J2>>) -> Self::Output {
        Qty::from_ex(&self.inner * &rhs.inner)
    }
}

impl<L1, M1, T1, I1, Th1, N1x, J1, L2, M2, T2, I2, Th2, N2x, J2>
    ops::Mul<&Qty<Dim<L2, M2, T2, I2, Th2, N2x, J2>>> for Qty<Dim<L1, M1, T1, I1, Th1, N1x, J1>>
where
    L1: ops::Add<L2>,
    M1: ops::Add<M2>,
    T1: ops::Add<T2>,
    I1: ops::Add<I2>,
    Th1: ops::Add<Th2>,
    N1x: ops::Add<N2x>,
    J1: ops::Add<J2>,
{
    type Output = Qty<
        Dim<
            Sum<L1, L2>,
            Sum<M1, M2>,
            Sum<T1, T2>,
            Sum<I1, I2>,
            Sum<Th1, Th2>,
            Sum<N1x, N2x>,
            Sum<J1, J2>,
        >,
    >;

    #[inline]
    fn mul(self, rhs: &Qty<Dim<L2, M2, T2, I2, Th2, N2x, J2>>) -> Self::Output {
        Qty::from_ex(&self.inner * &rhs.inner)
    }
}

impl<L1, M1, T1, I1, Th1, N1x, J1, L2, M2, T2, I2, Th2, N2x, J2>
    ops::Mul<Qty<Dim<L2, M2, T2, I2, Th2, N2x, J2>>> for &Qty<Dim<L1, M1, T1, I1, Th1, N1x, J1>>
where
    L1: ops::Add<L2>,
    M1: ops::Add<M2>,
    T1: ops::Add<T2>,
    I1: ops::Add<I2>,
    Th1: ops::Add<Th2>,
    N1x: ops::Add<N2x>,
    J1: ops::Add<J2>,
{
    type Output = Qty<
        Dim<
            Sum<L1, L2>,
            Sum<M1, M2>,
            Sum<T1, T2>,
            Sum<I1, I2>,
            Sum<Th1, Th2>,
            Sum<N1x, N2x>,
            Sum<J1, J2>,
        >,
    >;

    #[inline]
    fn mul(self, rhs: Qty<Dim<L2, M2, T2, I2, Th2, N2x, J2>>) -> Self::Output {
        Qty::from_ex(&self.inner * &rhs.inner)
    }
}

impl<L1, M1, T1, I1, Th1, N1x, J1, L2, M2, T2, I2, Th2, N2x, J2>
    ops::Mul<&Qty<Dim<L2, M2, T2, I2, Th2, N2x, J2>>> for &Qty<Dim<L1, M1, T1, I1, Th1, N1x, J1>>
where
    L1: ops::Add<L2>,
    M1: ops::Add<M2>,
    T1: ops::Add<T2>,
    I1: ops::Add<I2>,
    Th1: ops::Add<Th2>,
    N1x: ops::Add<N2x>,
    J1: ops::Add<J2>,
{
    type Output = Qty<
        Dim<
            Sum<L1, L2>,
            Sum<M1, M2>,
            Sum<T1, T2>,
            Sum<I1, I2>,
            Sum<Th1, Th2>,
            Sum<N1x, N2x>,
            Sum<J1, J2>,
        >,
    >;

    #[inline]
    fn mul(self, rhs: &Qty<Dim<L2, M2, T2, I2, Th2, N2x, J2>>) -> Self::Output {
        Qty::from_ex(&self.inner * &rhs.inner)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Div — blanket dimension subtraction, all four ownership combos
// ═══════════════════════════════════════════════════════════════════════════

impl<L1, M1, T1, I1, Th1, N1x, J1, L2, M2, T2, I2, Th2, N2x, J2>
    ops::Div<Qty<Dim<L2, M2, T2, I2, Th2, N2x, J2>>> for Qty<Dim<L1, M1, T1, I1, Th1, N1x, J1>>
where
    L1: ops::Sub<L2>,
    M1: ops::Sub<M2>,
    T1: ops::Sub<T2>,
    I1: ops::Sub<I2>,
    Th1: ops::Sub<Th2>,
    N1x: ops::Sub<N2x>,
    J1: ops::Sub<J2>,
{
    type Output = Qty<
        Dim<
            Diff<L1, L2>,
            Diff<M1, M2>,
            Diff<T1, T2>,
            Diff<I1, I2>,
            Diff<Th1, Th2>,
            Diff<N1x, N2x>,
            Diff<J1, J2>,
        >,
    >;

    #[inline]
    fn div(self, rhs: Qty<Dim<L2, M2, T2, I2, Th2, N2x, J2>>) -> Self::Output {
        Qty::from_ex(&self.inner / &rhs.inner)
    }
}

impl<L1, M1, T1, I1, Th1, N1x, J1, L2, M2, T2, I2, Th2, N2x, J2>
    ops::Div<&Qty<Dim<L2, M2, T2, I2, Th2, N2x, J2>>> for Qty<Dim<L1, M1, T1, I1, Th1, N1x, J1>>
where
    L1: ops::Sub<L2>,
    M1: ops::Sub<M2>,
    T1: ops::Sub<T2>,
    I1: ops::Sub<I2>,
    Th1: ops::Sub<Th2>,
    N1x: ops::Sub<N2x>,
    J1: ops::Sub<J2>,
{
    type Output = Qty<
        Dim<
            Diff<L1, L2>,
            Diff<M1, M2>,
            Diff<T1, T2>,
            Diff<I1, I2>,
            Diff<Th1, Th2>,
            Diff<N1x, N2x>,
            Diff<J1, J2>,
        >,
    >;

    #[inline]
    fn div(self, rhs: &Qty<Dim<L2, M2, T2, I2, Th2, N2x, J2>>) -> Self::Output {
        Qty::from_ex(&self.inner / &rhs.inner)
    }
}

impl<L1, M1, T1, I1, Th1, N1x, J1, L2, M2, T2, I2, Th2, N2x, J2>
    ops::Div<Qty<Dim<L2, M2, T2, I2, Th2, N2x, J2>>> for &Qty<Dim<L1, M1, T1, I1, Th1, N1x, J1>>
where
    L1: ops::Sub<L2>,
    M1: ops::Sub<M2>,
    T1: ops::Sub<T2>,
    I1: ops::Sub<I2>,
    Th1: ops::Sub<Th2>,
    N1x: ops::Sub<N2x>,
    J1: ops::Sub<J2>,
{
    type Output = Qty<
        Dim<
            Diff<L1, L2>,
            Diff<M1, M2>,
            Diff<T1, T2>,
            Diff<I1, I2>,
            Diff<Th1, Th2>,
            Diff<N1x, N2x>,
            Diff<J1, J2>,
        >,
    >;

    #[inline]
    fn div(self, rhs: Qty<Dim<L2, M2, T2, I2, Th2, N2x, J2>>) -> Self::Output {
        Qty::from_ex(&self.inner / &rhs.inner)
    }
}

impl<L1, M1, T1, I1, Th1, N1x, J1, L2, M2, T2, I2, Th2, N2x, J2>
    ops::Div<&Qty<Dim<L2, M2, T2, I2, Th2, N2x, J2>>> for &Qty<Dim<L1, M1, T1, I1, Th1, N1x, J1>>
where
    L1: ops::Sub<L2>,
    M1: ops::Sub<M2>,
    T1: ops::Sub<T2>,
    I1: ops::Sub<I2>,
    Th1: ops::Sub<Th2>,
    N1x: ops::Sub<N2x>,
    J1: ops::Sub<J2>,
{
    type Output = Qty<
        Dim<
            Diff<L1, L2>,
            Diff<M1, M2>,
            Diff<T1, T2>,
            Diff<I1, I2>,
            Diff<Th1, Th2>,
            Diff<N1x, N2x>,
            Diff<J1, J2>,
        >,
    >;

    #[inline]
    fn div(self, rhs: &Qty<Dim<L2, M2, T2, I2, Th2, N2x, J2>>) -> Self::Output {
        Qty::from_ex(&self.inner / &rhs.inner)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Scalar ops — i64
// ═══════════════════════════════════════════════════════════════════════════

// Qty<D> * i64 → Qty<D>
impl<D> ops::Mul<i64> for Qty<D> {
    type Output = Qty<D>;

    #[inline]
    fn mul(self, rhs: i64) -> Qty<D> {
        Qty::from_ex(&self.inner * rhs)
    }
}

impl<D> ops::Mul<i64> for &Qty<D> {
    type Output = Qty<D>;

    #[inline]
    fn mul(self, rhs: i64) -> Qty<D> {
        Qty::from_ex(&self.inner * rhs)
    }
}

// i64 * Qty<D> → Qty<D>
impl<D> ops::Mul<Qty<D>> for i64 {
    type Output = Qty<D>;

    #[inline]
    fn mul(self, rhs: Qty<D>) -> Qty<D> {
        Qty::from_ex(self * &rhs.inner)
    }
}

impl<D> ops::Mul<&Qty<D>> for i64 {
    type Output = Qty<D>;

    #[inline]
    fn mul(self, rhs: &Qty<D>) -> Qty<D> {
        Qty::from_ex(self * &rhs.inner)
    }
}

// Qty<D> / i64 → Qty<D>
impl<D> ops::Div<i64> for Qty<D> {
    type Output = Qty<D>;

    #[inline]
    fn div(self, rhs: i64) -> Qty<D> {
        Qty::from_ex(&self.inner / rhs)
    }
}

impl<D> ops::Div<i64> for &Qty<D> {
    type Output = Qty<D>;

    #[inline]
    fn div(self, rhs: i64) -> Qty<D> {
        Qty::from_ex(&self.inner / rhs)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Scalar ops — raw Ex (dimensionless scaling by arbitrary expression)
// ═══════════════════════════════════════════════════════════════════════════

// Qty<D> * &Ex → Qty<D>
impl<D> ops::Mul<&Ex> for Qty<D> {
    type Output = Qty<D>;

    #[inline]
    fn mul(self, rhs: &Ex) -> Qty<D> {
        Qty::from_ex(&self.inner * rhs)
    }
}

impl<D> ops::Mul<&Ex> for &Qty<D> {
    type Output = Qty<D>;

    #[inline]
    fn mul(self, rhs: &Ex) -> Qty<D> {
        Qty::from_ex(&self.inner * rhs)
    }
}

// &Ex * Qty<D> → Qty<D>
impl<D> ops::Mul<Qty<D>> for &Ex {
    type Output = Qty<D>;

    #[inline]
    fn mul(self, rhs: Qty<D>) -> Qty<D> {
        Qty::from_ex(self * &rhs.inner)
    }
}

impl<D> ops::Mul<&Qty<D>> for &Ex {
    type Output = Qty<D>;

    #[inline]
    fn mul(self, rhs: &Qty<D>) -> Qty<D> {
        Qty::from_ex(self * &rhs.inner)
    }
}

// Qty<D> / &Ex → Qty<D>
impl<D> ops::Div<&Ex> for Qty<D> {
    type Output = Qty<D>;

    #[inline]
    fn div(self, rhs: &Ex) -> Qty<D> {
        Qty::from_ex(&self.inner / rhs)
    }
}

impl<D> ops::Div<&Ex> for &Qty<D> {
    type Output = Qty<D>;

    #[inline]
    fn div(self, rhs: &Ex) -> Qty<D> {
        Qty::from_ex(&self.inner / rhs)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Escape hatch — unchecked dimension assertion
// ═══════════════════════════════════════════════════════════════════════════

/// Assert that a raw expression has dimension `D`.
///
/// This is **unchecked** — the caller is responsible for ensuring that the
/// expression genuinely represents a quantity with dimension `D`.
///
/// # Example
///
/// ```ignore
/// use symplex::units::{assume_dimension, ForceDim};
///
/// let raw_force_expr: Ex = /* ... */;
/// let force: Qty<ForceDim> = assume_dimension(raw_force_expr);
/// ```
#[inline]
pub fn assume_dimension<D>(ex: Ex) -> Qty<D> {
    Qty {
        inner: ex,
        _dim: PhantomData,
    }
}
