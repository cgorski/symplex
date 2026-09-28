//! Tests written alongside the 0.32 changes (`tests/v31/*.rs`): the bug
//! hunt after 0.31 — the nightly fuzz findings and the open items of the
//! 0.31 hand-off.  Every reference value cites the oracle call that
//! produced it.

#[path = "v31/v31_integrate.rs"]
mod v31_integrate;
#[path = "v31/v31_linalg.rs"]
mod v31_linalg;
#[path = "v31/v31_series.rs"]
mod v31_series;
#[path = "v31/v31_values.rs"]
mod v31_values;
