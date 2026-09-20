//! Feature tests written alongside the 0.11 APIs (`tests/v10/*.rs`, one
//! module per feature). Run one module with e.g.
//! `cargo test --test v10 v10_stats_continuous::`.

#[path = "v10/v10_stats_continuous.rs"]
mod v10_stats_continuous;
#[path = "v10/v10_stats_discrete.rs"]
mod v10_stats_discrete;
#[path = "v10/v10_stats_joint.rs"]
mod v10_stats_joint;
