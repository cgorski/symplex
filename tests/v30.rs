//! Tests written alongside the 0.31 changes (`tests/v30/*.rs`): the bug hunt
//! after 0.30 — integration with parameters over ℂ, new differential
//! hunters (systems, sets, sums and transforms of piecewise functions), and
//! the open items of the 0.30 hand-off.  Every reference value cites the
//! oracle call that produced it.

#[path = "v30/v30_hunts.rs"]
mod v30_hunts;
#[path = "v30/v30_integrate.rs"]
mod v30_integrate;
#[path = "v30/v30_poly.rs"]
mod v30_poly;
