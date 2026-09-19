//! Feature tests written alongside the 0.9 APIs (`tests/v09/*.rs`, one
//! module per feature). Run one module with e.g.
//! `cargo test --test v09 v09_algebraic::`.

#[path = "v09/v09_algebraic.rs"]
mod v09_algebraic;
#[path = "v09/v09_calculus_util.rs"]
mod v09_calculus_util;
#[path = "v09/v09_matrix.rs"]
mod v09_matrix;
#[path = "v09/v09_ntheory_discrete.rs"]
mod v09_ntheory_discrete;
#[path = "v09/v09_output.rs"]
mod v09_output;
#[path = "v09/v09_special.rs"]
mod v09_special;
