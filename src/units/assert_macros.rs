//! Compile-time dimension assertion macros.
//!
//! These macros leverage the Rust type system to verify that expressions
//! have the expected physical dimensions at compile time. If a dimension
//! mismatch occurs, the compiler emits a clear error message showing
//! the expected vs. actual types.

/// Compile-time dimension checkpoint.
///
/// Asserts that an expression has the expected named dimension type.
/// If the type doesn't match, the compiler error shows
/// "expected Force, found Velocity" (or similar clear message).
///
/// The macro evaluates the expression, binds it to a local with the
/// expected type annotation, and returns it — so it can be used inline.
///
/// # Examples
///
/// ```ignore
/// use symplex::prelude::*;
/// use symplex::units::*;
///
/// let ctx = Context::new();
/// let m = Mass::symbol(&ctx, "m");
/// let a = Acceleration::symbol(&ctx, "a");
/// let f = symplex::assert_dim!(m * a, Force);
/// ```
///
/// If you accidentally write:
///
/// ```compile_fail
/// use symplex::prelude::*;
/// use symplex::units::*;
///
/// let ctx = Context::new();
/// let m = Mass::symbol(&ctx, "m");
/// let a = Acceleration::symbol(&ctx, "a");
/// // Wrong! Mass × Acceleration is Force, not Velocity.
/// let v = symplex::assert_dim!(m * a, Velocity);
/// ```
///
/// the compiler will report a type mismatch between `Force` and `Velocity`.
#[macro_export]
macro_rules! assert_dim {
    ($expr:expr, $expected:ty) => {{
        let _check: $expected = $expr;
        _check
    }};
}

/// Compile-time formula dimension verification with custom error messages.
///
/// Uses const evaluation to check that a dimension formula produces
/// the expected result. If wrong, the compiler emits the custom message
/// as a compile-time panic.
///
/// This macro operates on [`ConstDim`](crate::units::dim::ConstDim) values,
/// which carry dimension exponents as plain `i8` values suitable for
/// `const fn` arithmetic.
///
/// # Examples
///
/// ```ignore
/// use symplex::units::ConstDim;
///
/// // Verified at compile time — F = ma:
/// symplex::const_assert_dim!(
///     ConstDim::MASS.mul(ConstDim::ACCELERATION),
///     ConstDim::FORCE,
///     "F = ma: Mass × Acceleration must equal Force"
/// );
///
/// // Energy = Force × Length:
/// symplex::const_assert_dim!(
///     ConstDim::FORCE.mul(ConstDim::LENGTH),
///     ConstDim::ENERGY,
///     "E = F·d: Force × Length must equal Energy"
/// );
/// ```
#[macro_export]
macro_rules! const_assert_dim {
    ($computed:expr, $expected:expr, $msg:literal) => {
        const _: () = {
            if !($computed).eq($expected) {
                panic!($msg);
            }
        };
    };
}
