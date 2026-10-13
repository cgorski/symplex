//! Tests written alongside the 0.49 type-architecture refactor
//! (`tests/v36/*.rs`): one file per workstream.  Every reference value
//! cites the oracle call that produced it.

#[path = "v36/v36_w1_ids.rs"]
mod v36_w1_ids;
#[path = "v36/v36_w2_truth.rs"]
mod v36_w2_truth;
#[path = "v36/v36_w3_solve.rs"]
mod v36_w3_solve;
#[path = "v36/v36_w3b_solve.rs"]
mod v36_w3b_solve;
#[path = "v36/v36_w4_numerics.rs"]
mod v36_w4_numerics;
#[path = "v36/v36_w5_func.rs"]
mod v36_w5_func;
#[path = "v36/v36_w6_facets.rs"]
mod v36_w6_facets;
#[path = "v36/v36_w7_nodes.rs"]
mod v36_w7_nodes;
#[path = "v36/v36_w8_truth.rs"]
mod v36_w8_truth;
#[path = "v36/v36_w9_budgets.rs"]
mod v36_w9_budgets;
