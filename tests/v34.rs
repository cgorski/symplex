//! Tests written alongside the 0.35 changes (`tests/v34/*.rs`): the bug
//! hunt after 0.34 — the nightly fuzz findings and the open items of the
//! 0.34 hand-off.  Every reference value cites the oracle call that
//! produced it.

#[path = "v34/v34_algebra.rs"]
mod v34_algebra;
#[path = "v34/v34_audit.rs"]
mod v34_audit;
#[path = "v34/v34_beyond.rs"]
mod v34_beyond;
#[path = "v34/v34_codegen.rs"]
mod v34_codegen;
#[path = "v34/v34_continuity.rs"]
mod v34_continuity;
#[path = "v34/v34_eval.rs"]
mod v34_eval;
#[path = "v34/v34_evalf.rs"]
mod v34_evalf;
#[path = "v34/v34_integrate.rs"]
mod v34_integrate;
#[path = "v34/v34_limits.rs"]
mod v34_limits;
#[path = "v34/v34_poly_speed.rs"]
mod v34_poly_speed;
#[path = "v34/v34_rewrites.rs"]
mod v34_rewrites;
#[path = "v34/v34_rubi.rs"]
mod v34_rubi;
#[path = "v34/v34_slow.rs"]
mod v34_slow;
#[path = "v34/v34_undefined.rs"]
mod v34_undefined;
#[path = "v34/v34_values.rs"]
mod v34_values;
