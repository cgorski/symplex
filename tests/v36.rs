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
#[path = "v36/v36_s14f1_constants.rs"]
mod v36_s14f1_constants;
#[path = "v36/v36_s14f2_limits.rs"]
mod v36_s14f2_limits;
#[path = "v36/v36_s14f3_definite.rs"]
mod v36_s14f3_definite;
#[path = "v36/v36_s14f4_solve.rs"]
mod v36_s14f4_solve;
#[path = "v36/v36_s14f5_matrix_ode.rs"]
mod v36_s14f5_matrix_ode;
#[path = "v36/v36_s15g1_undefined.rs"]
mod v36_s15g1_undefined;
#[path = "v36/v36_s15g2_solve.rs"]
mod v36_s15g2_solve;
#[path = "v36/v36_s15g3_quadrature.rs"]
mod v36_s15g3_quadrature;
#[path = "v36/v36_s15g4_float_evidence.rs"]
mod v36_s15g4_float_evidence;
#[path = "v36/v36_s15g5_conventions.rs"]
mod v36_s15g5_conventions;
#[path = "v36/v36_s15h1_infinity.rs"]
mod v36_s15h1_infinity;
#[path = "v36/v36_s15h2_evalf.rs"]
mod v36_s15h2_evalf;
#[path = "v36/v36_s15h3_integration.rs"]
mod v36_s15h3_integration;
#[path = "v36/v36_s15h4_constants.rs"]
mod v36_s15h4_constants;
#[path = "v36/v36_s15h5_float_evidence.rs"]
mod v36_s15h5_float_evidence;
#[path = "v36/v36_s15j1_solve.rs"]
mod v36_s15j1_solve;
#[path = "v36/v36_s15j2_summation.rs"]
mod v36_s15j2_summation;
#[path = "v36/v36_s15j3_evalf.rs"]
mod v36_s15j3_evalf;
#[path = "v36/v36_s15j4_float_evidence.rs"]
mod v36_s15j4_float_evidence;
#[path = "v36/v36_s15j5_integration.rs"]
mod v36_s15j5_integration;
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
