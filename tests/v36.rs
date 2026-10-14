//! Tests written alongside the 0.49 type-architecture refactor
//! (`tests/v36/*.rs`): one file per workstream.  Every reference value
//! cites the oracle call that produced it.

#[path = "v36/v36_b1_definite.rs"]
mod v36_b1_definite;
#[path = "v36/v36_b2_limits.rs"]
mod v36_b2_limits;
#[path = "v36/v36_b3_canon_solve.rs"]
mod v36_b3_canon_solve;
#[path = "v36/v36_s14c_residue.rs"]
mod v36_s14c_residue;
#[path = "v36/v36_s14d_assume.rs"]
mod v36_s14d_assume;
#[path = "v36/v36_s14e_solve.rs"]
mod v36_s14e_solve;
#[path = "v36/v36_w10_poly.rs"]
mod v36_w10_poly;
#[path = "v36/v36_w10b_domains.rs"]
mod v36_w10b_domains;
#[path = "v36/v36_w10c_domains.rs"]
mod v36_w10c_domains;
#[path = "v36/v36_w11_evalf.rs"]
mod v36_w11_evalf;
#[path = "v36/v36_w11b_evalf.rs"]
mod v36_w11b_evalf;
#[path = "v36/v36_w12_perf.rs"]
mod v36_w12_perf;
#[path = "v36/v36_w1_ids.rs"]
mod v36_w1_ids;
#[path = "v36/v36_w2_truth.rs"]
mod v36_w2_truth;
#[path = "v36/v36_w3_solve.rs"]
mod v36_w3_solve;
#[path = "v36/v36_w3b_solve.rs"]
mod v36_w3b_solve;
#[path = "v36/v36_w3c_solve.rs"]
mod v36_w3c_solve;
#[path = "v36/v36_w4_numerics.rs"]
mod v36_w4_numerics;
#[path = "v36/v36_w5_func.rs"]
mod v36_w5_func;
#[path = "v36/v36_w6_facets.rs"]
mod v36_w6_facets;
#[path = "v36/v36_w6b_codegen.rs"]
mod v36_w6b_codegen;
#[path = "v36/v36_w6c_lib.rs"]
mod v36_w6c_lib;
#[path = "v36/v36_w6d_funcs.rs"]
mod v36_w6d_funcs;
#[path = "v36/v36_w7_nodes.rs"]
mod v36_w7_nodes;
#[path = "v36/v36_w8_truth.rs"]
mod v36_w8_truth;
#[path = "v36/v36_w8b_deciders.rs"]
mod v36_w8b_deciders;
#[path = "v36/v36_w8c_truth.rs"]
mod v36_w8c_truth;
#[path = "v36/v36_w9_budgets.rs"]
mod v36_w9_budgets;
