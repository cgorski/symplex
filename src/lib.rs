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
//! assert_eq!(format!("{expr}"), "1 + x^2 + 2*x");
//! ```

// ── Internal modules (not part of the public API) ──────────────────────
pub(crate) mod arena;
pub(crate) mod canon;
pub(crate) mod diff;
pub(crate) mod display;
pub(crate) mod eval;
pub(crate) mod evalf;
pub(crate) mod expand;
pub(crate) mod node;
pub(crate) mod pattern;
pub(crate) mod poly;
pub(crate) mod polybridge;
pub(crate) mod solve;
pub(crate) mod sort_key;
pub(crate) mod subs;
pub(crate) mod symbol;
pub(crate) mod walk;

// ── Public modules (stable API surface) ────────────────────────────────
pub mod assumptions;
pub mod config;
pub mod context;
pub mod errors;
pub mod expr;
pub mod macros;

// ── Proc macro re-exports ──────────────────────────────────────────────
pub use symplex_macros::{expr, rule};

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
    pub use crate::errors::SymplexError;
    pub use crate::expr::Ex;
    pub use crate::pattern::Step;
    pub use symplex_macros::{expr, rule};
}
