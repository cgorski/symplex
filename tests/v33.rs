//! Tests written alongside the 0.34 changes (`tests/v33/*.rs`): the bug
//! hunt after 0.33 — the nightly fuzz findings and the open items of the
//! 0.33 hand-off.  Every reference value cites the oracle call that
//! produced it.

#[path = "v33/v33_evalf.rs"]
mod v33_evalf;
#[path = "v33/v33_integrate.rs"]
mod v33_integrate;
#[path = "v33/v33_radicals.rs"]
mod v33_radicals;
#[path = "v33/v33_solve.rs"]
mod v33_solve;
#[path = "v33/v33_undefined.rs"]
mod v33_undefined;
