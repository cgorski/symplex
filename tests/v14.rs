//! Feature tests written alongside the 0.14 APIs (`tests/v14/*.rs`, one
//! module per track). Run one module with e.g.
//! `cargo test --test v14 v14_regression::`.

#[path = "v14/v14_multivariate_order.rs"]
mod v14_multivariate_order;
#[path = "v14/v14_regression.rs"]
mod v14_regression;
#[path = "v14/v14_reliability.rs"]
mod v14_reliability;
#[path = "v14/v14_survival.rs"]
mod v14_survival;
