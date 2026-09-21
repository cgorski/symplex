//! Feature tests written alongside the 0.18 APIs (`tests/v18/*.rs`, one
//! module per track). Run one module with e.g.
//! `cargo test --test v18 v18_layout::`.

#[path = "v18/v18_layout.rs"]
mod v18_layout;
#[path = "v18/v18_samplers.rs"]
mod v18_samplers;
#[path = "v18/v18_roots.rs"]
mod v18_roots;
