//! **symplex** — a fast, correct symbolic mathematics library for Rust.
//!
//! Symplex is a symbolic mathematics library designed around seven principles:
//!
//! 1. **Construction is cheap, evaluation is explicit.** Constructors only
//!    canonicalize (flatten, sort, combine). No expansion, no function
//!    evaluation, no identity application. Call `.eval()`, `.expand()`, or
//!    `.simplify()` when *you* choose.
//!
//! 2. **Never silently wrong.** Operations return `Result::Err` instead of
//!    silent wrong answers. Structural substitution by default.
//!
//! 3. **One representation per concept.** One assumption system. One polynomial
//!    type. One number type. One solve function.
//!
//! 4. **Thread-safe from day one.** Expression handles are `Send + Sync`.
//!
//! 5. **No recursive tree walks.** All traversals use explicit stacks.
//!
//! 6. **The compiler is the API contract.** `pub` = stable. `pub(crate)` = internal.
//!
//! 7. **Extensible without inheritance.** Custom functions via registered rules.
//!
//! # Quick Start
//!
//! ```
//! use symplex::prelude::*;
//! use symplex::syms;
//!
//! let ctx = Context::new();
//! syms!(ctx; x, y);
//! let expr = &x * &x + &x * 2 + 1;
//! assert_eq!(format!("{expr}"), "x^2 + 2*x + 1");
//! ```

#![warn(missing_docs)]

// ── Self-referencing extern crate so proc-macro-generated paths
//    (`::symplex::__macro_support::…`) resolve inside the crate itself. ──
extern crate self as symplex;

// ── Directory modules (internal organisation) ──────────────────────────
/// Foundation layer: expression nodes, arena, tree traversal, canonicalization, and core types.
pub mod base;
pub(crate) mod poly;
pub(crate) mod transforms;
pub(crate) mod simplify;
pub(crate) mod calculus;
pub(crate) mod output;
pub(crate) mod plotting;
pub(crate) mod domains;
pub(crate) mod api;
/// Compile-time dimensional analysis for physical quantities.
pub mod units;

// ── Public re-exports (backwards-compatible crate-root paths) ──────────

// base
/// Assumption system for symbolic variables.
pub use base::assumptions;
/// Library-wide configuration knobs.
pub use base::config;
/// Error types used throughout the library.
pub use base::errors;

// poly
/// Sparse multivariate polynomials over ℚ.
pub use poly::multipoly;
/// Gröbner basis computation via Buchberger's algorithm with FGLM order conversion.
pub use poly::groebner;
/// Polynomial system solving via Gröbner bases.
pub use poly::polysys;

// calculus
/// Symbolic Fourier transform.
pub use calculus::fourier_transform;
/// Formal power series representations and algorithms.
pub use calculus::formal_series;
/// Finite difference methods: weights, application, and differentiation.
pub use calculus::finite_diff;
/// Ordinary differential equation solver.
pub use calculus::ode;
/// Z-transform for discrete-time signal analysis.
pub use calculus::z_transform;

// output
/// Runtime expression parser — convert strings to symbolic expressions.
pub use output::parse;
/// Serializable expression tree for interchange (JSON, etc.).
pub use output::tree;

// plotting
/// Data export utilities: CSV, TSV, JSON, Markdown, HTML, LaTeX table output.
pub use plotting::data_export;

// domains
/// Control systems: state-space models, transfer functions, stability analysis.
pub use domains::control;
/// Lagrangian dynamics: equations of motion, mass matrix, Coriolis, gravity.
pub use domains::dynamics;
/// Symbolic matrix type and operations.
pub use domains::matrix;
/// Number theory: primality, factorization, divisors, modular arithmetic.
pub use domains::ntheory;
/// Symbolic quaternion algebra for attitude representation.
pub use domains::quaternion;
/// Robotics kinematics: DH parameters, forward kinematics, rotations.
pub use domains::robotics;
/// Vector calculus: gradient, divergence, curl, laplacian.
pub use domains::vector;

// api
/// The core expression handle and types.
pub use api::expr;
/// Symbolic equation type (`lhs = rhs`).
pub use api::eq;
/// A non-locking, read-only view of an expression node for use in `replace()`.
pub use api::expr_view;
/// Expression context — arena, symbol table, configuration.
pub use api::context;
/// Convenience macros for building expressions.
pub use api::macros;

// ── Proc macro re-exports ──────────────────────────────────────────────
pub use symplex_macros::{eq, expr, matrix, rule};

// ── Macro support (hidden internals used by generated code) ────────────
#[doc(hidden)]
pub mod __macro_support {
    pub use crate::base::node::ExprNode;
    pub use crate::transforms::pattern::{Pattern, Rule, WildId};
    pub use rustc_hash::FxHashMap;
}

// bitflags types don't auto-derive Default; provide it here so
// Assumptions::default() works.
impl Default for base::assumptions::Props {
    fn default() -> Self {
        Self::empty()
    }
}

/// The symplex prelude — one import to get started.
///
/// ```
/// use symplex::prelude::*;
/// ```
pub mod prelude {
    pub use crate::base::assumptions::{Assumption, Assumptions, Props};
    pub use crate::base::config::EvalConfig;
    pub use crate::api::context::Context;
    pub use crate::domains::control::{StateSpace, TransferFunction};
    pub use crate::api::eq::Equation;
    pub use crate::base::errors::SymplexError;
    pub use crate::api::expr::{BoolEx, Boolean, Ex, Expr, ExprType, Numeric, SetEx, SetValued, Sort};
    pub use crate::api::expr_view::ExprView;
    pub use crate::domains::matrix::Matrix;
    pub use crate::transforms::pattern::Step;
    pub use crate::domains::quaternion::Quaternion;
    pub use symplex_macros::{eq, expr, matrix, rule};

    // NOTE: `vars!`, `syms!`, and `sym!` are `#[macro_export]` macros and
    // live at the crate root.  Use `use symplex::{vars, syms, sym};` or
    // `use symplex::prelude::*; use symplex::vars;` to bring them in.
}

// ═══════════════════════════════════════════════════════════════════════════
// Global default context — convenience functions for quick usage
// ═══════════════════════════════════════════════════════════════════════════

use std::sync::OnceLock;

/// The global default context, lazily initialized.
static DEFAULT_CONTEXT: OnceLock<api::context::Context> = OnceLock::new();

/// Returns a reference to the global default context.
///
/// The context is created on first access with default configuration.
/// All expressions created via the free-standing [`symbol`], [`var`],
/// [`int`], and [`rational`] functions share this context.
pub fn default_context() -> &'static api::context::Context {
    DEFAULT_CONTEXT.get_or_init(api::context::Context::new)
}

/// Create a symbolic variable in the global default context.
///
/// This is a convenience shorthand for
/// `symplex::default_context().symbol(name)`.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
///
/// let x = symplex::var("x");
/// let expr = x.powi(2);
/// assert_eq!(format!("{expr}"), "x^2");
/// ```
pub fn var(name: &str) -> api::expr::Ex {
    default_context().symbol(name)
}

/// Create a symbolic variable in the global default context.
///
/// Alias for [`var`].
pub fn symbol(name: &str) -> api::expr::Ex {
    var(name)
}

/// Create an integer expression in the global default context.
///
/// # Examples
///
/// ```
/// let five = symplex::int(5);
/// assert_eq!(format!("{five}"), "5");
/// ```
pub fn int(n: i64) -> api::expr::Ex {
    default_context().int(n)
}

/// Create a rational expression in the global default context.
///
/// # Examples
///
/// ```
/// let half = symplex::rational(1, 2);
/// assert_eq!(format!("{half}"), "1/2");
/// ```
pub fn rational(p: i64, q: i64) -> api::expr::Ex {
    default_context().rational(p, q)
}

/// The constant π in the global default context.
///
/// # Examples
///
/// ```
/// let pi = symplex::pi();
/// assert_eq!(format!("{pi}"), "pi");
/// ```
pub fn pi() -> api::expr::Ex {
    default_context().pi()
}

/// Euler's number e in the global default context.
///
/// # Examples
///
/// ```
/// let e = symplex::e();
/// assert_eq!(format!("{e}"), "E");
/// ```
pub fn e() -> api::expr::Ex {
    default_context().e()
}

/// The imaginary unit i in the global default context.
///
/// # Examples
///
/// ```
/// let i = symplex::i_unit();
/// assert_eq!(format!("{i}"), "I");
/// ```
pub fn i_unit() -> api::expr::Ex {
    default_context().i_unit()
}

/// Positive infinity in the global default context.
///
/// # Examples
///
/// ```
/// let inf = symplex::infinity();
/// assert_eq!(format!("{inf}"), "oo");
/// ```
pub fn infinity() -> api::expr::Ex {
    default_context().infinity()
}

/// Negative infinity in the global default context.
///
/// # Examples
///
/// ```
/// let neg_inf = symplex::neg_infinity();
/// assert_eq!(format!("{neg_inf}"), "-oo");
/// ```
pub fn neg_infinity() -> api::expr::Ex {
    default_context().neg_infinity()
}

/// One-half (1/2) as a symbolic expression.
pub fn half() -> api::expr::Ex {
    default_context().rational(1, 2)
}

/// One-third (1/3) as a symbolic expression.
pub fn third() -> api::expr::Ex {
    default_context().rational(1, 3)
}

/// One-quarter (1/4) as a symbolic expression.
pub fn quarter() -> api::expr::Ex {
    default_context().rational(1, 4)
}

/// Two-thirds (2/3) as a symbolic expression.
pub fn two_thirds() -> api::expr::Ex {
    default_context().rational(2, 3)
}

/// Solve a system of polynomial equations.
///
/// Convenience wrapper around [`polysys::solve_system_ex`].
pub fn solve_system(
    eqs: &[api::expr::Ex],
    vars: &[api::expr::Ex],
) -> Result<Vec<Vec<api::expr::Ex>>, base::errors::SymplexError> {
    poly::polysys::solve_system_ex(eqs, vars)
}
