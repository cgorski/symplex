//! Compiles and runs the Rust code blocks of `README.md` and the book
//! (`book/src/**/*.md`) as doctests, so the prose examples cannot drift
//! from the API.  Only built under `cfg(doctest)`; nothing here is part of
//! the library.  A block tagged ```` ```rust,ignore ```` is skipped, one
//! tagged ```` ```rust,no_run ```` is compiled but not executed.
//!
//! One module per Markdown file; the module name is the file's path under
//! `book/src/` with `-`, `/` and `.` replaced by `_`, so a failing test is
//! reported as e.g. `doctests::guide_algebra (line 42)`.  Every chapter that
//! contains a Rust block is listed; a chapter with only `text`/`sh`/`lean`
//! fences needs no entry.

#[doc = include_str!("../README.md")]
mod readme {}

// ── Getting started ──────────────────────────────────────────────────────
#[doc = include_str!("../book/src/getting-started/installation.md")]
mod getting_started_installation {}
#[doc = include_str!("../book/src/getting-started/first-steps.md")]
mod getting_started_first_steps {}
#[doc = include_str!("../book/src/getting-started/key-concepts.md")]
mod getting_started_key_concepts {}

// ── User guide ───────────────────────────────────────────────────────────
#[doc = include_str!("../book/src/guide/algebra.md")]
mod guide_algebra {}
#[doc = include_str!("../book/src/guide/calculus.md")]
mod guide_calculus {}
#[doc = include_str!("../book/src/guide/code-generation.md")]
mod guide_code_generation {}
#[doc = include_str!("../book/src/guide/complex-analysis.md")]
mod guide_complex_analysis {}
#[doc = include_str!("../book/src/guide/definite-integration.md")]
mod guide_definite_integration {}
#[doc = include_str!("../book/src/guide/exact-lp.md")]
mod guide_exact_lp {}
#[doc = include_str!("../book/src/guide/integer-lattices.md")]
mod guide_integer_lattices {}
#[doc = include_str!("../book/src/guide/matrices.md")]
mod guide_matrices {}
#[doc = include_str!("../book/src/guide/number-theory.md")]
mod guide_number_theory {}
#[doc = include_str!("../book/src/guide/numerical-optimization.md")]
mod guide_numerical_optimization {}
#[doc = include_str!("../book/src/guide/polynomials.md")]
mod guide_polynomials {}
#[doc = include_str!("../book/src/guide/response-analysis.md")]
mod guide_response_analysis {}
#[doc = include_str!("../book/src/guide/rule-engine.md")]
mod guide_rule_engine {}
#[doc = include_str!("../book/src/guide/sets-and-logic.md")]
mod guide_sets_and_logic {}
#[doc = include_str!("../book/src/guide/solving.md")]
mod guide_solving {}
#[doc = include_str!("../book/src/guide/statistics.md")]
mod guide_statistics {}
#[doc = include_str!("../book/src/guide/summation.md")]
mod guide_summation {}
#[doc = include_str!("../book/src/guide/transforms.md")]
mod guide_transforms {}
#[doc = include_str!("../book/src/guide/units.md")]
mod guide_units {}

// ── Cookbook ─────────────────────────────────────────────────────────────
#[doc = include_str!("../book/src/cookbook/pid-controller.md")]
mod cookbook_pid_controller {}
#[doc = include_str!("../book/src/cookbook/polynomial-certificates.md")]
mod cookbook_polynomial_certificates {}

// ── Reference ────────────────────────────────────────────────────────────
#[doc = include_str!("../book/src/reference/api-patterns.md")]
mod reference_api_patterns {}
#[doc = include_str!("../book/src/reference/error-handling.md")]
mod reference_error_handling {}
#[doc = include_str!("../book/src/reference/migrating-0.2.md")]
mod reference_migrating_0_2 {}
#[doc = include_str!("../book/src/reference/migrating-0.4.md")]
mod reference_migrating_0_4 {}
#[doc = include_str!("../book/src/reference/migrating-0.7.md")]
mod reference_migrating_0_7 {}
#[doc = include_str!("../book/src/reference/sympy-migration.md")]
mod reference_sympy_migration {}

// ── What's new ───────────────────────────────────────────────────────────
#[doc = include_str!("../book/src/whats-new-0.5.md")]
mod whats_new_0_5 {}
#[doc = include_str!("../book/src/whats-new-0.7.md")]
mod whats_new_0_7 {}
