//! Historical bug-hunting rounds and one-off validations (`tests/legacy/*.rs`:
//! the former `round*_*`, `bugfinder*_*`, `math*_*`, `*_validation`,
//! `regular_joe_tests` and `fixpoint_convergence_test` files). Many tests here
//! are intentionally duplicated across rounds — see `tests/README.md`.

mod common;

#[path = "legacy/bugfinder1_edge_cases.rs"]
mod bugfinder1_edge_cases;
#[path = "legacy/bugfinder2_consistency.rs"]
mod bugfinder2_consistency;
#[path = "legacy/crt_overflow_validation.rs"]
mod crt_overflow_validation;
#[path = "legacy/fixpoint_convergence_test.rs"]
mod fixpoint_convergence_test;
#[path = "legacy/infinity_pow_validation.rs"]
mod infinity_pow_validation;
#[path = "legacy/limit_heuristic_validation.rs"]
mod limit_heuristic_validation;
#[path = "legacy/math1_calculus_bugs.rs"]
mod math1_calculus_bugs;
#[path = "legacy/math2_algebra_bugs.rs"]
mod math2_algebra_bugs;
#[path = "legacy/math3_linalg_ntheory_bugs.rs"]
mod math3_linalg_ntheory_bugs;
#[path = "legacy/pythagorean_scaling_validation.rs"]
mod pythagorean_scaling_validation;
#[path = "legacy/regular_joe_tests.rs"]
mod regular_joe_tests;
#[path = "legacy/round2_calculus_deep.rs"]
mod round2_calculus_deep;
#[path = "legacy/round2_codegen_bugs.rs"]
mod round2_codegen_bugs;
#[path = "legacy/round2_domains.rs"]
mod round2_domains;
#[path = "legacy/round2_poly_solver.rs"]
mod round2_poly_solver;
#[path = "legacy/round2_simplify_bugs.rs"]
mod round2_simplify_bugs;
#[path = "legacy/round2_stress.rs"]
mod round2_stress;
#[path = "legacy/round3_api_misuse.rs"]
mod round3_api_misuse;
#[path = "legacy/round3_differential.rs"]
mod round3_differential;
#[path = "legacy/round3_parser_fuzz.rs"]
mod round3_parser_fuzz;
#[path = "legacy/round3_properties.rs"]
mod round3_properties;
#[path = "legacy/round3_untested_modules.rs"]
mod round3_untested_modules;
#[path = "legacy/round4_extreme.rs"]
mod round4_extreme;
