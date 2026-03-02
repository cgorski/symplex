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
//! assert_eq!(format!("{expr}"), "1 + x**2 + 2*x");
//! ```

pub mod arena;
pub mod canon;
pub mod config;
pub mod context;
pub mod display;
pub mod errors;
pub mod expr;
pub mod macros;
pub mod node;
pub mod sort_key;
pub mod symbol;

/// The symplex prelude — one import to get started.
///
/// ```
/// use symplex::prelude::*;
/// ```
pub mod prelude {
    pub use crate::arena::Arena;
    pub use crate::config::EvalConfig;
    pub use crate::context::Context;
    pub use crate::errors::SymplexError;
    pub use crate::expr::Ex;
    pub use crate::node::{CtxId, ExprId, ExprNode, NumId, SymbolId};
}
