//! Feature tests written alongside the 0.12 APIs (`tests/v12/*.rs`, one
//! module per feature). Run one module with e.g.
//! `cargo test --test v12 v12_stats_families::`.

#[path = "v12/v12_betainc.rs"]
mod v12_betainc;
#[path = "v12/v12_numeric_summation.rs"]
mod v12_numeric_summation;
#[path = "v12/v12_stats_families.rs"]
mod v12_stats_families;
#[path = "v12/v12_stats_transforms.rs"]
mod v12_stats_transforms;
