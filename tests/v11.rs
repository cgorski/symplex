//! Regression tests for the 0.11.x review fixes (`tests/v11/*.rs`, one
//! module per review track). Run one module with e.g.
//! `cargo test --test v11 v11_special_fixes::`.

#[path = "v11/v11_calculus_algebraic_fixes.rs"]
mod v11_calculus_algebraic_fixes;
#[path = "v11/v11_output_fixes.rs"]
mod v11_output_fixes;
#[path = "v11/v11_review_fixes.rs"]
mod v11_review_fixes;
#[path = "v11/v11_special_fixes.rs"]
mod v11_special_fixes;
