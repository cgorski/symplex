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

// ── Internal modules (not part of the public API) ──────────────────────
pub(crate) mod apart;
pub(crate) mod arena;
pub(crate) mod canon;
pub(crate) mod codegen;
pub(crate) mod combsimp;
pub(crate) mod compact;
pub(crate) mod complex;
pub(crate) mod convergence;
pub(crate) mod cse;
pub(crate) mod diff;
pub(crate) mod display;
/// Symbolic equation type (`lhs = rhs`).
pub mod eq;
pub(crate) mod eval;
pub(crate) mod evalf;
pub(crate) mod expand;
/// A non-locking, read-only view of an expression node for use in `replace()`.
pub mod expr_view;
pub(crate) mod factor;
pub(crate) mod factor_terms;
pub(crate) mod fourier;
pub(crate) mod gruntz;
pub(crate) mod inequalities;
pub(crate) mod integrate;
pub(crate) mod lambdify;
pub(crate) mod laplace;
pub(crate) mod limit;
pub(crate) mod linalg;
pub(crate) mod log_combine;
pub(crate) mod log_expand;
/// Symbolic matrix type and operations.
pub mod matrix;
pub(crate) mod node;
pub(crate) mod nsimplify;
/// Ordinary differential equation solver.
pub mod ode;
pub(crate) mod pattern;
pub(crate) mod poly;
pub(crate) mod polybridge;
pub(crate) mod powsimp;
pub(crate) mod radsimp;
pub(crate) mod residue;
pub(crate) mod rewrite;
pub(crate) mod separatevars;
pub(crate) mod series;
pub(crate) mod simplify_engine;
pub(crate) mod solve;
pub(crate) mod sort_key;
pub(crate) mod subs;
pub(crate) mod sum_eval;
pub(crate) mod symbol;
pub(crate) mod trig_combine;
pub(crate) mod trig_expand;
pub(crate) mod trig_integ;
pub(crate) mod trigsimp;
/// Vector calculus: gradient, divergence, curl, laplacian.
pub mod vector;
pub(crate) mod walk;

// ── Public modules ─────────────────────────────────────────────────────
/// Runtime expression parser — convert strings to symbolic expressions.
pub mod parse;
/// Serializable expression tree for interchange (JSON, etc.).
pub mod tree;

// ── Public modules (stable API surface) ────────────────────────────────
pub mod assumptions;
pub mod config;
pub mod context;
pub mod errors;
pub mod expr;
mod expr_funcs;
mod expr_ops;
pub mod macros;

// ── Proc macro re-exports ──────────────────────────────────────────────
pub use symplex_macros::{eq, expr, matrix, rule};

// ── Macro support (hidden internals used by generated code) ────────────
#[doc(hidden)]
pub mod __macro_support {
    pub use crate::node::ExprNode;
    pub use crate::pattern::{Pattern, Rule, WildId};
    pub use rustc_hash::FxHashMap;
}

// bitflags types don't auto-derive Default; provide it here so
// Assumptions::default() works.
impl Default for assumptions::Props {
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
    pub use crate::assumptions::{Assumption, Assumptions, Props};
    pub use crate::config::EvalConfig;
    pub use crate::context::Context;
    pub use crate::eq::Equation;
    pub use crate::errors::SymplexError;
    pub use crate::expr::{BoolEx, Boolean, Ex, Expr, ExprType, Numeric, SetEx, SetValued, Sort};
    pub use crate::expr_view::ExprView;
    pub use crate::pattern::Step;
    pub use symplex_macros::{eq, expr, matrix, rule};
}

// ═══════════════════════════════════════════════════════════════════════════
// Global default context — convenience functions for quick usage
// ═══════════════════════════════════════════════════════════════════════════

use std::sync::OnceLock;

/// The global default context, lazily initialized.
static DEFAULT_CONTEXT: OnceLock<context::Context> = OnceLock::new();

/// Returns a reference to the global default context.
///
/// The context is created on first access with default configuration.
/// All expressions created via the free-standing [`symbol`], [`var`],
/// [`int`], and [`rational`] functions share this context.
pub fn default_context() -> &'static context::Context {
    DEFAULT_CONTEXT.get_or_init(context::Context::new)
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
pub fn var(name: &str) -> expr::Ex {
    default_context().symbol(name)
}

/// Create a symbolic variable in the global default context.
///
/// Alias for [`var`].
pub fn symbol(name: &str) -> expr::Ex {
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
pub fn int(n: i64) -> expr::Ex {
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
pub fn rational(p: i64, q: i64) -> expr::Ex {
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
pub fn pi() -> expr::Ex {
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
pub fn e() -> expr::Ex {
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
pub fn i_unit() -> expr::Ex {
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
pub fn infinity() -> expr::Ex {
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
pub fn neg_infinity() -> expr::Ex {
    default_context().neg_infinity()
}
