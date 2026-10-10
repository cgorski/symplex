//! Tests written alongside the 0.39 changes (`tests/v35/*.rs`): the bug
//! hunt after 0.38.  Every reference value cites the oracle call that
//! produced it.

#[path = "v35/v35_definite.rs"]
mod v35_definite;
#[path = "v35/v35_eval.rs"]
mod v35_eval;
#[path = "v35/v35_integrate_speed.rs"]
mod v35_integrate_speed;
#[path = "v35/v35_solve.rs"]
mod v35_solve;
#[path = "v35/v35_transforms.rs"]
mod v35_transforms;
