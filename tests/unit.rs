//! Unit-style integration tests per module / feature (`tests/unit/*.rs`, one
//! module per former `tests/test_*.rs` file). Run one module with e.g.
//! `cargo test --test unit test_known_answers::`.

mod common;

#[path = "unit/test_advanced.rs"]
mod test_advanced;
#[path = "unit/test_advanced_control.rs"]
mod test_advanced_control;
#[path = "unit/test_alpha_fixes.rs"]
mod test_alpha_fixes;
#[path = "unit/test_api_coverage.rs"]
mod test_api_coverage;
#[path = "unit/test_apply_diff.rs"]
mod test_apply_diff;
#[path = "unit/test_arb_prec_special.rs"]
mod test_arb_prec_special;
#[path = "unit/test_assume_queries.rs"]
mod test_assume_queries;
#[path = "unit/test_bareiss.rs"]
mod test_bareiss;
#[path = "unit/test_bug_regression.rs"]
mod test_bug_regression;
#[path = "unit/test_calculus_parity.rs"]
mod test_calculus_parity;
#[path = "unit/test_calculus_util.rs"]
mod test_calculus_util;
#[path = "unit/test_codegen.rs"]
mod test_codegen;
#[path = "unit/test_codegen_advanced.rs"]
mod test_codegen_advanced;
#[path = "unit/test_codegen_numopt.rs"]
mod test_codegen_numopt;
#[path = "unit/test_codegen_quality.rs"]
mod test_codegen_quality;
#[path = "unit/test_collect_together.rs"]
mod test_collect_together;
#[path = "unit/test_combinatorial.rs"]
mod test_combinatorial;
#[path = "unit/test_compact.rs"]
mod test_compact;
#[path = "unit/test_complex.rs"]
mod test_complex;
#[path = "unit/test_concrete.rs"]
mod test_concrete;
#[path = "unit/test_concurrency.rs"]
mod test_concurrency;
#[path = "unit/test_control.rs"]
mod test_control;
#[path = "unit/test_convenience.rs"]
mod test_convenience;
#[path = "unit/test_correctness_audit.rs"]
mod test_correctness_audit;
#[path = "unit/test_cross_context.rs"]
mod test_cross_context;
#[path = "unit/test_cubic_quartic.rs"]
mod test_cubic_quartic;
#[path = "unit/test_cycle11.rs"]
mod test_cycle11;
#[path = "unit/test_cycle7_8.rs"]
mod test_cycle7_8;
#[path = "unit/test_cycle9_10.rs"]
mod test_cycle9_10;
#[path = "unit/test_data_export.rs"]
mod test_data_export;
#[path = "unit/test_definite_poly.rs"]
mod test_definite_poly;
#[path = "unit/test_delta_laplace_depth.rs"]
mod test_delta_laplace_depth;
#[path = "unit/test_dim_macro.rs"]
mod test_dim_macro;
#[path = "unit/test_display_quality.rs"]
mod test_display_quality;
#[path = "unit/test_dynamics.rs"]
mod test_dynamics;
#[path = "unit/test_dynamics_scale.rs"]
mod test_dynamics_scale;
#[path = "unit/test_equation.rs"]
mod test_equation;
#[path = "unit/test_ergonomics.rs"]
mod test_ergonomics;
#[path = "unit/test_evalf_arb_prec.rs"]
mod test_evalf_arb_prec;
#[path = "unit/test_evalf_gaps.rs"]
mod test_evalf_gaps;
#[path = "unit/test_evalf_special.rs"]
mod test_evalf_special;
#[path = "unit/test_expand_convenience.rs"]
mod test_expand_convenience;
#[path = "unit/test_expr_fraction.rs"]
mod test_expr_fraction;
#[path = "unit/test_expr_view.rs"]
mod test_expr_view;
#[path = "unit/test_factoring_boundaries.rs"]
mod test_factoring_boundaries;
#[path = "unit/test_finite_diff.rs"]
mod test_finite_diff;
#[path = "unit/test_floor_ceil_minmax.rs"]
mod test_floor_ceil_minmax;
#[path = "unit/test_formal_series.rs"]
mod test_formal_series;
#[path = "unit/test_foundations.rs"]
mod test_foundations;
#[path = "unit/test_fu.rs"]
mod test_fu;
#[path = "unit/test_full_simplify.rs"]
mod test_full_simplify;
#[path = "unit/test_gaussian_laplace.rs"]
mod test_gaussian_laplace;
#[path = "unit/test_gc_plus.rs"]
mod test_gc_plus;
#[path = "unit/test_gosper.rs"]
mod test_gosper;
#[path = "unit/test_groebner.rs"]
mod test_groebner;
#[path = "unit/test_hard_math.rs"]
mod test_hard_math;
#[path = "unit/test_heaviside_integ.rs"]
mod test_heaviside_integ;
#[path = "unit/test_hensel.rs"]
mod test_hensel;
#[path = "unit/test_heurisch.rs"]
mod test_heurisch;
#[path = "unit/test_homogeneous_tuples.rs"]
mod test_homogeneous_tuples;
#[path = "unit/test_inequalities.rs"]
mod test_inequalities;
#[path = "unit/test_integrate.rs"]
mod test_integrate;
#[path = "unit/test_integration_advanced.rs"]
mod test_integration_advanced;
#[path = "unit/test_integration_fix.rs"]
mod test_integration_fix;
#[path = "unit/test_integration_gaps.rs"]
mod test_integration_gaps;
#[path = "unit/test_known_answers.rs"]
mod test_known_answers;
#[path = "unit/test_lambertw.rs"]
mod test_lambertw;
#[path = "unit/test_laplace.rs"]
mod test_laplace;
#[path = "unit/test_layering.rs"]
mod test_layering;
#[path = "unit/test_limits.rs"]
mod test_limits;
#[path = "unit/test_limits_infinity.rs"]
mod test_limits_infinity;
#[path = "unit/test_linalg.rs"]
mod test_linalg;
#[path = "unit/test_linear_systems.rs"]
mod test_linear_systems;
#[path = "unit/test_log_to_real.rs"]
mod test_log_to_real;
#[path = "unit/test_logic_ext.rs"]
mod test_logic_ext;
#[path = "unit/test_logic_piecewise.rs"]
mod test_logic_piecewise;
#[path = "unit/test_macro_improvements.rs"]
mod test_macro_improvements;
#[path = "unit/test_macro_new_funcs.rs"]
mod test_macro_new_funcs;
#[path = "unit/test_macros.rs"]
mod test_macros;
#[path = "unit/test_math_depth.rs"]
mod test_math_depth;
#[path = "unit/test_math_rules.rs"]
mod test_math_rules;
#[path = "unit/test_matrix_codegen.rs"]
mod test_matrix_codegen;
#[path = "unit/test_matrix_decomp.rs"]
mod test_matrix_decomp;
#[path = "unit/test_matrix_exp.rs"]
mod test_matrix_exp;
#[path = "unit/test_matrix_linalg.rs"]
mod test_matrix_linalg;
#[path = "unit/test_matrix_ode.rs"]
mod test_matrix_ode;
#[path = "unit/test_multinomial.rs"]
mod test_multinomial;
#[path = "unit/test_multipoly.rs"]
mod test_multipoly;
#[path = "unit/test_new_api.rs"]
mod test_new_api;
#[path = "unit/test_new_math.rs"]
mod test_new_math;
#[path = "unit/test_new_rules.rs"]
mod test_new_rules;
#[path = "unit/test_no_panics.rs"]
mod test_no_panics;
#[path = "unit/test_ntheory.rs"]
mod test_ntheory;
#[path = "unit/test_numerical_validation.rs"]
mod test_numerical_validation;
#[path = "unit/test_ode_advanced.rs"]
mod test_ode_advanced;
#[path = "unit/test_ode_comprehensive.rs"]
mod test_ode_comprehensive;
#[path = "unit/test_ode_coverage.rs"]
mod test_ode_coverage;
#[path = "unit/test_ode_depth.rs"]
mod test_ode_depth;
#[path = "unit/test_ode_expanded.rs"]
mod test_ode_expanded;
#[path = "unit/test_ode_extra.rs"]
mod test_ode_extra;
#[path = "unit/test_ode_public.rs"]
mod test_ode_public;
#[path = "unit/test_ode_systems.rs"]
mod test_ode_systems;
#[path = "unit/test_panel_fixes.rs"]
mod test_panel_fixes;
#[path = "unit/test_parametric_integration.rs"]
mod test_parametric_integration;
#[path = "unit/test_parser.rs"]
mod test_parser;
#[path = "unit/test_perf_infra.rs"]
mod test_perf_infra;
#[path = "unit/test_physical_constants.rs"]
mod test_physical_constants;
#[path = "unit/test_piecewise_integ.rs"]
mod test_piecewise_integ;
#[path = "unit/test_plotting.rs"]
mod test_plotting;
#[path = "unit/test_poly_refactor.rs"]
mod test_poly_refactor;
#[path = "unit/test_polysys.rs"]
mod test_polysys;
#[path = "unit/test_polysys_irrational.rs"]
mod test_polysys_irrational;
#[path = "unit/test_quaternion.rs"]
mod test_quaternion;
#[path = "unit/test_radsimp_limits.rs"]
mod test_radsimp_limits;
#[path = "unit/test_rational_integration.rs"]
mod test_rational_integration;
#[path = "unit/test_recip_trig.rs"]
mod test_recip_trig;
#[path = "unit/test_risch_e2e.rs"]
mod test_risch_e2e;
#[path = "unit/test_robotics.rs"]
mod test_robotics;
#[path = "unit/test_rootof.rs"]
mod test_rootof;
#[path = "unit/test_rothstein_trager.rs"]
mod test_rothstein_trager;
#[path = "unit/test_rule_application.rs"]
mod test_rule_application;
#[path = "unit/test_rule_macro_new.rs"]
mod test_rule_macro_new;
#[path = "unit/test_sampling.rs"]
mod test_sampling;
#[path = "unit/test_sc_fixes.rs"]
mod test_sc_fixes;
#[path = "unit/test_serde_roundtrip.rs"]
mod test_serde_roundtrip;
#[path = "unit/test_series_comprehensive.rs"]
mod test_series_comprehensive;
#[path = "unit/test_series_ext.rs"]
mod test_series_ext;
#[path = "unit/test_series_factor.rs"]
mod test_series_factor;
#[path = "unit/test_sets.rs"]
mod test_sets;
#[path = "unit/test_sign_cancel_bug.rs"]
mod test_sign_cancel_bug;
#[path = "unit/test_simp_depth.rs"]
mod test_simp_depth;
#[path = "unit/test_simp_improvements.rs"]
mod test_simp_improvements;
#[path = "unit/test_simplify.rs"]
mod test_simplify;
#[path = "unit/test_size_assertions.rs"]
mod test_size_assertions;
#[path = "unit/test_solve_cancel.rs"]
mod test_solve_cancel;
#[path = "unit/test_solve_trig.rs"]
mod test_solve_trig;
#[path = "unit/test_solver_utils.rs"]
mod test_solver_utils;
#[path = "unit/test_special_elem.rs"]
mod test_special_elem;
#[path = "unit/test_special_extended.rs"]
mod test_special_extended;
#[path = "unit/test_special_funcs.rs"]
mod test_special_funcs;
#[path = "unit/test_special_values.rs"]
mod test_special_values;
#[path = "unit/test_sprint_integration.rs"]
mod test_sprint_integration;
#[path = "unit/test_sqrt_quadratic.rs"]
mod test_sqrt_quadratic;
#[path = "unit/test_stage1.rs"]
mod test_stage1;
#[path = "unit/test_stage3.rs"]
mod test_stage3;
#[path = "unit/test_stage5.rs"]
mod test_stage5;
#[path = "unit/test_stage6.rs"]
mod test_stage6;
#[path = "unit/test_stage7_8.rs"]
mod test_stage7_8;
#[path = "unit/test_stripper_collector.rs"]
mod test_stripper_collector;
#[path = "unit/test_sturm.rs"]
mod test_sturm;
#[path = "unit/test_sum_closed.rs"]
mod test_sum_closed;
#[path = "unit/test_sympy_cross_validation.rs"]
mod test_sympy_cross_validation;
#[path = "unit/test_trig_hyp.rs"]
mod test_trig_hyp;
#[path = "unit/test_trig_power_integ.rs"]
mod test_trig_power_integ;
#[path = "unit/test_type_gating.rs"]
mod test_type_gating;
#[path = "unit/test_units_comprehensive.rs"]
mod test_units_comprehensive;
#[path = "unit/test_vector_calc.rs"]
mod test_vector_calc;
#[path = "unit/test_workflows.rs"]
mod test_workflows;
#[path = "unit/test_z_transform.rs"]
mod test_z_transform;
