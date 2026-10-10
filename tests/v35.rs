//! Tests written alongside the 0.39 changes (`tests/v35/*.rs`): the bug
//! hunt after 0.38.  Every reference value cites the oracle call that
//! produced it.

#[path = "v35/v35_applied.rs"]
mod v35_applied;
#[path = "v35/v35_assumptions.rs"]
mod v35_assumptions;
#[path = "v35/v35_definite.rs"]
mod v35_definite;
#[path = "v35/v35_diff.rs"]
mod v35_diff;
#[path = "v35/v35_eval.rs"]
mod v35_eval;
#[path = "v35/v35_integrate_speed.rs"]
mod v35_integrate_speed;
#[path = "v35/v35_ntheory.rs"]
mod v35_ntheory;
#[path = "v35/v35_numeric.rs"]
mod v35_numeric;
#[path = "v35/v35_poly.rs"]
mod v35_poly;
#[path = "v35/v35_sets.rs"]
mod v35_sets;
#[path = "v35/v35_solve.rs"]
mod v35_solve;
#[path = "v35/v35_special.rs"]
mod v35_special;
#[path = "v35/v35_stats.rs"]
mod v35_stats;
#[path = "v35/v35_transforms.rs"]
mod v35_transforms;
