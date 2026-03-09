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
pub use symplex_macros::{dim, eq, expr, matrix, rule};

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
    pub use symplex_macros::{dim, eq, expr, matrix, rule};

    // NOTE: `vars!`, `syms!`, and `sym!` are `#[macro_export]` macros and
    // live at the crate root.  Use `use symplex::{vars, syms, sym};` or
    // `use symplex::prelude::*; use symplex::vars;` to bring them in.
}
