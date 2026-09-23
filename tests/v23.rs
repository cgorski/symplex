//! Tests written alongside the 0.24 changes (`tests/v23/*.rs`): integration
//! correctness and reach found by the Rubi test suite (`rubi-harness/`), exact
//! evaluation on the principal branch, and `assert!` sites turned into `Result`s.

#[path = "v23/v23_eval.rs"]
mod v23_eval;
#[path = "v23/v23_integrate.rs"]
mod v23_integrate;
#[path = "v23/v23_results.rs"]
mod v23_results;
