//! **symplex** — a fast, correct symbolic mathematics library for Rust.
//!
//! Expressions are exact (`Ratio<BigInt>` arithmetic, hash-consed in a
//! [`Context`](prelude::Context) arena) and the library can differentiate,
//! integrate (indefinite, definite, improper, numeric), sum, take limits and
//! series, solve equations, systems, ODEs and recurrences, simplify with a
//! public rewrite-rule engine, work with sets and boolean logic, do exact
//! linear algebra, view expressions as polynomials with symbolic
//! coefficients ([`Poly`](prelude::Poly)), put rational functions into
//! normal form (`ratsimp`), solve linear programs exactly with dual values
//! and Farkas certificates, compute Hermite and Smith normal forms of
//! integer matrices, run numerical root finding and minimisation, apply
//! Laplace/Fourier/Mellin/Z transforms, and generate optimized Rust or C99
//! code. See the [README](https://github.com/cgorski/symplex) and
//! [The Symplex Book](https://cgorski.github.io/symplex/) for a guided tour,
//! and `CHANGELOG.md` for the 0.1 → 0.2 breaking changes and the 0.2 → 0.3
//! behaviour changes.
//!
//! Symplex is designed around seven principles:
//!
//! 1. **Construction is cheap, evaluation is explicit.** Constructors only
//!    canonicalize (flatten, sort, combine). No expansion, no function
//!    evaluation, no identity application. Call `.eval()`, `.expand()`, or
//!    `.simplify()` when *you* choose.
//!
//! 2. **Never silently wrong.** Operations return `Result::Err`, an
//!    unevaluated node, or `None` instead of a guess: `∫₋₁¹ dx/x²` is
//!    `Err(Divergent)`, `solve(x − x)` is `Err(InfiniteSolutions)`, `re(z)` stays
//!    `re(z)` until `z` is known to be real. Structural substitution by default.
//!
//! 3. **One representation per concept.** One assumption system. One polynomial
//!    type. One number type. One solve function.
//!
//! 4. **Thread-safe from day one.** Expression handles are `Send + Sync`.
//!
//! 5. **No recursive tree walks.** All traversals use explicit stacks.
//!
//! 6. **The compiler is the API contract.** `pub` = stable. `pub(crate)` = internal.
//!    `Ex`, `BoolEx` and `SetEx` are distinct types.
//!
//! 7. **Extensible without inheritance.** Custom functions via registered rules
//!    ([`Rule`](prelude::Rule), [`RuleSet`](prelude::RuleSet)).
//!
//! # API model
//!
//! * Operations for which "unevaluated" is a valid answer return
//!   [`Ex`](prelude::Ex) and have a `try_` twin returning `Result`
//!   (`integrate` / `try_integrate`, `integrate_definite` /
//!   `try_integrate_definite`, `summation` / `try_summation`, …).
//! * Numeric boundaries (`eval_f64`, `compile`, `to_rust_fn`, `to_c_fn`,
//!   `integrate_numeric`) and structural preconditions (`Matrix::inv`,
//!   `cholesky`) return `Result`.
//! * Queries (`is_positive`, `equals`, `SetEx::contains`,
//!   `Matrix::is_symmetric`) return `Option<bool>`: yes, no, or unknown.
//!
//! # Quick Start
//!
//! ```
//! use symplex::prelude::*;
//! use symplex::syms;
//!
//! let ctx = Context::new();
//! syms!(ctx; x, y);
//! let expr = &x * &x + &x * 2 + 1;
//! assert_eq!(format!("{expr}"), "x^2 + 2*x + 1");
//!
//! // Differentiate, integrate over an infinite range, solve, compile.
//! assert_eq!(format!("{}", expr.diff(&x)), "2*x + 2");
//! let gauss = (-x.powi(2)).exp().integrate_definite(&x, &ctx.neg_infinity(), &ctx.infinity());
//! assert_eq!(format!("{gauss}"), "sqrt(pi)");
//! let roots = (&x.powi(2) - 4).solve(&x).unwrap();
//! assert_eq!(roots.len(), 2);
//! let f = expr.compile(&["x"]).unwrap();
//! assert_eq!(f(&[2.0]), 9.0);
//! ```
//!
//! # Module map
//!
//! The [`prelude`] re-exports everything most programs need. Domain modules
//! are re-exported at the crate root: [`ntheory`], [`diophantine`],
//! [`combinatorics`], [`mod@matrix`], [`matrix_decomp`], [`normalforms`],
//! [`linprog`], [`optimize`], [`vector`], [`quaternion`], [`control`],
//! [`robotics`], [`dynamics`], [`poly_ex`], [`multipoly`], [`polysys`],
//! [`groebner`], [`factor_zassenhaus`], [`definite`], [`summation`],
//! [`formal_series`], [`finite_diff`], [`fourier_transform`], [`mellin`],
//! [`z_transform`], [`ode`], [`rsolve`], [`sets`], [`logic`], [`parse`],
//! [`tree`], [`codegen`], [`lambdify`], [`units`], [`assumptions`],
//! [`numeric`], [`errors`], [`config`].
//!
//! New in 0.3: [`poly_ex`] (the [`Poly`](prelude::Poly) view of an
//! expression), [`linprog`] (exact simplex), [`normalforms`] (Hermite /
//! Smith normal forms, integer kernels) and [`optimize`] (Brent,
//! Nelder–Mead, differential evolution, least-squares fitting).

#![warn(missing_docs)]

// ── Self-referencing extern crate so proc-macro-generated paths
//    (`::symplex::__macro_support::…`) resolve inside the crate itself. ──
extern crate self as symplex;

// ── Directory modules (internal organisation) ──────────────────────────
pub(crate) mod api;
/// Foundation layer: expression nodes, arena, tree traversal, canonicalization, and core types.
pub mod base;
pub(crate) mod calculus;
pub(crate) mod domains;
pub(crate) mod output;
pub(crate) mod plotting;
pub(crate) mod poly;
pub(crate) mod simplify;
pub(crate) mod transforms;
/// Compile-time dimensional analysis for physical quantities.
pub mod units;

// ── Public re-exports (backwards-compatible crate-root paths) ──────────

// The exact-arithmetic crates whose types appear in the public API
// (`Ratio<BigInt>` from `as_rational`, `linprog::Q`, `poly_fit_exact`,
// `Matrix::from_ratio`, …).  Re-exported so a downstream crate can name and
// manipulate those values without adding — and version-matching — the crates
// itself: `symplex::num_rational::Ratio`, `symplex::num_bigint::BigInt`,
// `symplex::num_traits::{Zero, One, Signed}`, `symplex::num_integer::Integer`.
pub use num_bigint;
pub use num_integer;
pub use num_rational;
pub use num_traits;

// base
/// Assumption system for symbolic variables.
pub use base::assumptions;
/// Library-wide configuration knobs.
pub use base::config;
/// Error types used throughout the library.
pub use base::errors;

// base
/// Exact `f64` ↔ rational conversions (dyadic exact, and best bounded-denominator approximations).
pub use base::numeric;

// poly
/// Univariate factorization over ℤ via Berlekamp–Zassenhaus.
pub use poly::factor_zassenhaus;
/// Gröbner basis computation via Buchberger's algorithm with FGLM order conversion.
pub use poly::groebner;
/// Sparse multivariate polynomials over ℚ.
pub use poly::multipoly;
/// Polynomial system solving via Gröbner bases.
pub use poly::polysys;

// calculus
/// Definite and improper integration.
pub use calculus::definite;
/// Finite difference methods: weights, application, and differentiation.
pub use calculus::finite_diff;
/// Formal power series representations and algorithms.
pub use calculus::formal_series;
/// Symbolic Fourier transform.
pub use calculus::fourier_transform;
/// Mellin transform.
pub use calculus::mellin;
/// Ordinary differential equation solver.
pub use calculus::ode;
/// Symbolic summation and products.
pub use calculus::summation;
/// Z-transform for discrete-time signal analysis.
pub use calculus::z_transform;

// transforms
/// Boolean-logic simplification, normal forms, satisfiability.
pub use transforms::logic;
/// Recurrence-relation solver.
pub use transforms::rsolve;
/// Set algebra on intervals, finite sets, unions.
pub use transforms::sets;

// output
/// Lean 4 / Mathlib rendering (`Ex::to_lean`, `LeanOpts`).
pub use output::lean;
/// Runtime expression parser — convert strings to symbolic expressions.
pub use output::parse;
/// Serializable expression tree for interchange (JSON, etc.).
pub use output::tree;

// plotting
/// Data export utilities: CSV, TSV, JSON, Markdown, HTML, LaTeX table output.
pub use plotting::data_export;

// domains
/// Exact, machine-checkable non-negativity certificates: Handelman (boxes),
/// half-lines, parametric polyhedra and sums of squares, with Lean export.
pub use domains::certificates;
/// Combinatorics: Stirling numbers, multinomial coefficients, partition counting.
pub use domains::combinatorics;
/// Control systems: state-space models, transfer functions, stability analysis.
pub use domains::control;
/// Diophantine equations.
pub use domains::diophantine;
/// Lagrangian dynamics: equations of motion, mass matrix, Coriolis, gravity.
pub use domains::dynamics;
/// Exact linear programming over ℚ (two-phase simplex, duals, Farkas certificates).
pub use domains::linprog;
/// Symbolic matrix type and operations.
pub use domains::matrix;
/// Additional matrix decompositions (QR, Gram–Schmidt) and structure tests.
pub use domains::matrix_decomp;
/// Integer matrix normal forms: Hermite, Smith, unimodular transforms, integer kernels.
pub use domains::normalforms;
/// Number theory: primality, factorization, divisors, modular arithmetic.
pub use domains::ntheory;
/// Numerical optimisation and root bracketing (Brent, Nelder–Mead, polynomial fitting).
pub use domains::optimize;
/// Exact convex polyhedra in ℚⁿ from half-spaces: vertices, volume,
/// containment, cutting.
pub use domains::polytope;
/// Symbolic quaternion algebra for attitude representation.
pub use domains::quaternion;
/// Robotics kinematics: DH parameters, forward kinematics, rotations.
pub use domains::robotics;
/// Vector calculus: gradient, divergence, curl, laplacian.
pub use domains::vector;

// api
/// Expression context — arena, symbol table, configuration.
pub use api::context;
/// Symbolic equation type (`lhs = rhs`).
pub use api::eq;
/// The core expression handle and types.
pub use api::expr;
/// Complex-analysis methods on `Ex` (`re`, `im`, `conjugate`, `arg`, `polar`, …).
pub use api::expr_complex;
/// Definite / improper / numeric integration methods on `Ex`.
pub use api::expr_integrate_ext as integrate_api;
/// Operator overloads and scalar-conversion traits (`ToEx`, `Scalar`).
pub use api::expr_ops;
/// Polynomial-algebra methods on `Ex` (resultant, discriminant, division, numeric roots, …).
pub use api::expr_poly_ext as poly_api;
/// Public rewrite-rule engine: `Rule`, `RuleSet`, `Bindings`, `RewriteOpts`, `Step`.
pub use api::expr_rules_ext as rules;
/// Summation, products, series and formal-power-series methods on `Ex`.
pub use api::expr_series_ext as series_api;
/// Set-algebra and boolean-logic helpers (`reduce_inequalities`).
pub use api::expr_sets_ext as sets_api;
/// Solver entry points beyond `Ex::solve`: `linsolve`, `LinearSolution`, `GeneralSolution`, Newton systems.
pub use api::expr_solve_ext as solvers;
/// Integral transforms (Fourier, Mellin, Laplace helpers) and directional limits on `Ex`.
pub use api::expr_transforms_ext as transforms_api;
/// A non-locking, read-only view of an expression node for use in `replace()`.
pub use api::expr_view;
/// Convenience macros for building expressions.
pub use api::macros;
/// Public sparse polynomial view (`Poly`) over explicit generators.
pub use api::poly_ex;

// output
/// Code-generation options and compiled numeric functions.
pub use output::codegen;
/// Compiled numeric closures (`CompiledFn`, `CompiledFnVec`).
pub use output::lambdify;

// ── Proc macro re-exports ──────────────────────────────────────────────
pub use symplex_macros::{dim, eq, expr, matrix, rule};

// ── Macro support (hidden internals used by generated code) ────────────
#[doc(hidden)]
pub mod __macro_support {
    pub use crate::base::node::ExprNode;
    pub use crate::transforms::pattern::{Pattern, Rule, WildId};
    pub use rustc_hash::FxHashMap;
}

// bitflags types don't auto-derive Default; provide it here so
// Assumptions::default() works.
impl Default for base::assumptions::Props {
    fn default() -> Self {
        Self::empty()
    }
}

/// The symplex prelude — one import to get started.
///
/// ```
/// use symplex::prelude::*;
/// ```
pub mod prelude {
    pub use crate::api::context::Context;
    pub use crate::api::eq::Equation;
    pub use crate::api::expr::{
        BoolEx, Boolean, Ex, Expr, ExprType, Numeric, SetEx, SetValued, SimplifyOpts, Sort,
    };
    pub use crate::api::expr_ops::{Scalar, ToEx};
    pub use crate::api::expr_rules_ext::{
        Bindings, RewriteOpts, RewriteStrategy, Rule, RuleSet, Step,
    };
    pub use crate::api::expr_sets_ext::reduce_inequalities;
    pub use crate::api::expr_solve_ext::{
        GeneralSolution, LinearSolution, NewtonOpts, ZeroForm, linsolve, linsolve_matrix,
        solve_numeric_system, solve_numeric_system_with,
    };
    pub use crate::api::expr_transforms_ext::{Direction, FourierConvention, FourierSeries};
    pub use crate::api::expr_view::ExprView;
    pub use crate::api::poly_ex::Poly;
    pub use crate::base::assumptions::{Assumption, Assumptions, Props};
    pub use crate::base::config::EvalConfig;
    pub use crate::base::errors::SymplexError;
    pub use crate::calculus::definite::QuadOpts;
    pub use crate::calculus::formal_series::FormalPowerSeries;
    pub use crate::domains::control::{StateSpace, TransferFunction};
    pub use crate::domains::linprog::{Feasibility, LpProblem, LpSolution, LpStatus, Q};
    pub use crate::domains::matrix::{Matrix, QMatrix, ZMatrix};
    pub use crate::domains::optimize::{MinimizeOpts, MinimizeResult, RootOpts};
    pub use crate::domains::quaternion::Quaternion;
    pub use crate::domains::robotics::EulerConvention;
    pub use crate::domains::vector::CoordinateSystem;
    pub use crate::output::codegen::{CodegenOptions, MathBackend, Precision};
    pub use crate::output::lambdify::{CompiledFn, CompiledFnVec};
    pub use crate::poly::multipoly::MultiPoly;
    pub use crate::transforms::expand::ExpandOpts;
    pub use symplex_macros::{dim, eq, expr, matrix, rule};

    // NOTE: `vars!`, `syms!`, and `sym!` are `#[macro_export]` macros and
    // live at the crate root.  Use `use symplex::{vars, syms, sym};` or
    // `use symplex::prelude::*; use symplex::vars;` to bring them in.
}
