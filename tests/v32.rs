//! Tests written alongside the 0.33 changes (`tests/v32/*.rs`): the bug
//! hunt after 0.32 — the nightly fuzz findings and the open items of the
//! 0.32 hand-off.  Every reference value cites the oracle call that
//! produced it.

#[path = "v32/v32_norms.rs"]
mod v32_norms;
#[path = "v32/v32_series.rs"]
mod v32_series;
#[path = "v32/v32_simplify.rs"]
mod v32_simplify;
#[path = "v32/v32_undefined.rs"]
mod v32_undefined;
