//! Tests written alongside the 0.35 changes (`tests/v34/*.rs`): the bug
//! hunt after 0.34 — the nightly fuzz findings and the open items of the
//! 0.34 hand-off.  Every reference value cites the oracle call that
//! produced it.

#[path = "v34/v34_evalf.rs"]
mod v34_evalf;
#[path = "v34/v34_undefined.rs"]
mod v34_undefined;
#[path = "v34/v34_values.rs"]
mod v34_values;
