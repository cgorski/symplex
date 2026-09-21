//! Feature tests written alongside the 0.13 APIs (`tests/v13/*.rs`, one
//! module per track). Run one module with e.g.
//! `cargo test --test v13 v13_hypothesis::`.

#[path = "v13/v13_agreement.rs"]
mod v13_agreement;
#[path = "v13/v13_data_estimation.rs"]
mod v13_data_estimation;
#[path = "v13/v13_hypothesis.rs"]
mod v13_hypothesis;
#[path = "v13/v13_walkthrough.rs"]
mod v13_walkthrough;
