//! Tests written alongside the 0.23 changes (`tests/v22/*.rs`): one domain
//! model — a symbol without assumptions is complex and every function takes
//! its principal branch, in the evaluator (`evalf`) and in every rewrite.
//! The findings come from `fuzz_simplify` comparing complex values at real
//! and complex points.

#[path = "v22/v22_evalf.rs"]
mod v22_evalf;
#[path = "v22/v22_integrate.rs"]
mod v22_integrate;
#[path = "v22/v22_simplify.rs"]
mod v22_simplify;
