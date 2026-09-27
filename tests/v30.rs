//! Tests written alongside the 0.31 changes (`tests/v30/*.rs`): the bug hunt
//! after 0.30 — integration with parameters over ℂ, new differential
//! hunters (systems, sets, sums and transforms of piecewise functions), and
//! the open items of the 0.30 hand-off.  Every reference value cites the
//! oracle call that produced it.

#[path = "v30/v30_evalf.rs"]
mod v30_evalf;
#[path = "v30/v30_hunts.rs"]
mod v30_hunts;
#[path = "v30/v30_integrate.rs"]
mod v30_integrate;
#[path = "v30/v30_integrate2.rs"]
mod v30_integrate2;
#[path = "v30/v30_integrate3.rs"]
mod v30_integrate3;
#[path = "v30/v30_integrate4.rs"]
mod v30_integrate4;
#[path = "v30/v30_linalg.rs"]
mod v30_linalg;
#[path = "v30/v30_parse.rs"]
mod v30_parse;
#[path = "v30/v30_poly.rs"]
mod v30_poly;
#[path = "v30/v30_polylog.rs"]
mod v30_polylog;
