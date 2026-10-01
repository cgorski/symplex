//! Tests written alongside the 0.34 changes (`tests/v33/*.rs`): the bug
//! hunt after 0.33 — the nightly fuzz findings and the open items of the
//! 0.33 hand-off.  Every reference value cites the oracle call that
//! produced it.

#[path = "v33/v33_integrate.rs"]
mod v33_integrate;
#[path = "v33/v33_radicals.rs"]
mod v33_radicals;
