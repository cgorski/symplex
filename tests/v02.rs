//! Feature tests written alongside the 0.2 API (`tests/v02/*.rs`, one module per
//! former `tests/v02_*.rs` file). Run one module with e.g.
//! `cargo test --test v02 v02_sets_algebra::`.

mod common;

#[path = "v02/v02_backends_c.rs"]
mod v02_backends_c;
#[path = "v02/v02_backends_codegen.rs"]
mod v02_backends_codegen;
#[path = "v02/v02_backends_compile.rs"]
mod v02_backends_compile;
#[path = "v02/v02_backends_cse.rs"]
mod v02_backends_cse;
#[path = "v02/v02_basefix_assumptions.rs"]
mod v02_basefix_assumptions;
#[path = "v02/v02_basefix_canon.rs"]
mod v02_basefix_canon;
#[path = "v02/v02_basefix_deep.rs"]
mod v02_basefix_deep;
#[path = "v02/v02_basefix_display.rs"]
mod v02_basefix_display;
#[path = "v02/v02_basefix_eval.rs"]
mod v02_basefix_eval;
#[path = "v02/v02_basefix_evalf.rs"]
mod v02_basefix_evalf;
#[path = "v02/v02_basefix_free_symbols.rs"]
mod v02_basefix_free_symbols;
#[path = "v02/v02_basefix_nroots.rs"]
mod v02_basefix_nroots;
#[path = "v02/v02_basefix_perf.rs"]
mod v02_basefix_perf;
#[path = "v02/v02_basefix_radicals.rs"]
mod v02_basefix_radicals;
#[path = "v02/v02_defint_node.rs"]
mod v02_defint_node;
#[path = "v02/v02_ergonomics_compare.rs"]
mod v02_ergonomics_compare;
#[path = "v02/v02_ergonomics_equation.rs"]
mod v02_ergonomics_equation;
#[path = "v02/v02_ergonomics_ingest.rs"]
mod v02_ergonomics_ingest;
#[path = "v02/v02_ergonomics_plotting.rs"]
mod v02_ergonomics_plotting;
#[path = "v02/v02_ergonomics_traits.rs"]
mod v02_ergonomics_traits;
#[path = "v02/v02_integration_battery.rs"]
mod v02_integration_battery;
#[path = "v02/v02_integration_definite.rs"]
mod v02_integration_definite;
#[path = "v02/v02_integration_residue.rs"]
mod v02_integration_residue;
#[path = "v02/v02_matrices_api.rs"]
mod v02_matrices_api;
#[path = "v02/v02_matrices_control.rs"]
mod v02_matrices_control;
#[path = "v02/v02_matrices_decomp.rs"]
mod v02_matrices_decomp;
#[path = "v02/v02_matrices_quaternion.rs"]
mod v02_matrices_quaternion;
#[path = "v02/v02_matrices_vector.rs"]
mod v02_matrices_vector;
#[path = "v02/v02_nodes_apply_specials.rs"]
mod v02_nodes_apply_specials;
#[path = "v02/v02_nodes_complex.rs"]
mod v02_nodes_complex;
#[path = "v02/v02_nodes_constants.rs"]
mod v02_nodes_constants;
#[path = "v02/v02_nodes_special.rs"]
mod v02_nodes_special;
#[path = "v02/v02_ntheory_factor.rs"]
mod v02_ntheory_factor;
#[path = "v02/v02_ntheory_ntheory.rs"]
mod v02_ntheory_ntheory;
#[path = "v02/v02_ntheory_polyalg.rs"]
mod v02_ntheory_polyalg;
#[path = "v02/v02_numfix_api.rs"]
mod v02_numfix_api;
#[path = "v02/v02_numfix_bessel.rs"]
mod v02_numfix_bessel;
#[path = "v02/v02_numfix_codegen.rs"]
mod v02_numfix_codegen;
#[path = "v02/v02_numfix_logic.rs"]
mod v02_numfix_logic;
#[path = "v02/v02_numfix_macros.rs"]
mod v02_numfix_macros;
#[path = "v02/v02_numfix_parse.rs"]
mod v02_numfix_parse;
#[path = "v02/v02_numfix_radicals.rs"]
mod v02_numfix_radicals;
#[path = "v02/v02_sets_algebra.rs"]
mod v02_sets_algebra;
#[path = "v02/v02_sets_logic.rs"]
mod v02_sets_logic;
#[path = "v02/v02_sets_reduce.rs"]
mod v02_sets_reduce;
#[path = "v02/v02_simplify_gapfill.rs"]
mod v02_simplify_gapfill;
#[path = "v02/v02_simplify_rules.rs"]
mod v02_simplify_rules;
#[path = "v02/v02_simplify_subs.rs"]
mod v02_simplify_subs;
#[path = "v02/v02_solvefix_inequalities.rs"]
mod v02_solvefix_inequalities;
#[path = "v02/v02_solvefix_polysys.rs"]
mod v02_solvefix_polysys;
#[path = "v02/v02_solvefix_rsolve.rs"]
mod v02_solvefix_rsolve;
#[path = "v02/v02_solvefix_series.rs"]
mod v02_solvefix_series;
#[path = "v02/v02_solving_ode_rsolve.rs"]
mod v02_solving_ode_rsolve;
#[path = "v02/v02_solving_solve.rs"]
mod v02_solving_solve;
#[path = "v02/v02_solving_systems.rs"]
mod v02_solving_systems;
#[path = "v02/v02_summation_engine.rs"]
mod v02_summation_engine;
#[path = "v02/v02_summation_series.rs"]
mod v02_summation_series;
#[path = "v02/v02_transforms_fourier.rs"]
mod v02_transforms_fourier;
#[path = "v02/v02_transforms_laplace.rs"]
mod v02_transforms_laplace;
#[path = "v02/v02_transforms_limits.rs"]
mod v02_transforms_limits;
#[path = "v02/v02_transforms_mellin.rs"]
mod v02_transforms_mellin;
#[path = "v02/v02_transforms_series_z.rs"]
mod v02_transforms_series_z;
