# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Until 1.0, minor releases may contain breaking changes; they are listed first.

## [0.2.0] - 2026-09-18

A large release: complex analysis, definite/improper/numeric integration,
symbolic summation, a public rewrite-rule engine, sets and logic, a C99
backend, linear-system and general solvers, ODE initial-value problems,
recurrences, Berlekamp–Zassenhaus factoring, and a new `factorint`.  The
guiding rule for every change below is *never silently wrong*: operations
that used to guess now return `Err`, an unevaluated node, or `None`.

Measured at release: 91 `ExprNode` variants, ~10,600 `#[test]` functions
(~148K lines of tests), ~167K lines in `src/`, 520 doctests, and a
1,400-fixture SymPy 1.14 oracle with zero numerical disagreements.

### Breaking

**Core / API model**

- `Ex::compile(&[&str])` returns `Result<CompiledFn, SymplexError>` instead of
  `Option`.  `CompiledFn` is `Clone + Send + Sync`, callable as a closure,
  and has `arity()` and `try_call()` (arity-checked).
- `Ex::definite_integral` is replaced by `Ex::integrate_definite` (returns a
  bounded, unevaluated `DefiniteIntegral` node — displayed
  `Integral(f, x, a, b)` — when it cannot decide) and
  `Ex::try_integrate_definite` (`Err(Divergent)` / `Err(ComputationFailed)`).
  The old method could return a wrong finite number for `∫₋₁¹ dx/x²`.
- `Ex::solve` semantics: identities (`x − x = 0`) return
  `Err(InfiniteSolutions)`; contradictions and range violations (`sin x = 2`,
  `eˣ = −1`, `|x| = −1`) return `Err(NoSolution)`.  Results are `eval`'d
  (`asin(1/2)` → `π/6`).
- `Context::solve_system` returns `Result<LinearSolution>` with
  `Unique` / `Parametric { solution, free }` / `Inconsistent` instead of a
  flat vector.
- `polysys::solve_system_ex` returns algebraic (radical) solutions, not only
  rational ones, and `Err(InfiniteSolutions)` for positive-dimensional systems.
- `has_unevaluated()` no longer counts `RootOf` / `RootSum`: they are complete
  algebraic answers.  `try_*` methods therefore succeed on degree ≥ 5 roots.
- `Ex::equals` may now return `Some(false)` (previously only `Some(true)`/`None`).
- `Debug for Ex` prints the expression (`Ex(x^2 + 1)`) instead of internal ids.
- `re`, `im`, `conjugate`, `arg` of a symbol without a realness assumption
  return unevaluated `re(z)`, `im(z)`, `conjugate(z)`, `arg(z)` nodes.  0.1
  silently assumed every symbol real.
- `Digamma(n)` folds for positive integers and half-integers
  (`ψ(1) = −γ`); `d/dx digamma(x)` is `polygamma(1, x)` (was a formal
  derivative).
- `expand()` no longer splits `(x·y)^a` for symbols of unknown sign (unsound
  over ℂ).  Use `expand_power_base(force)` to opt in.
- `Ex::differentiate_finite(var, points, order)` replaces the previous
  finite-difference signature; `finite_diff::{finite_diff_weights,
  apply_finite_diff, …}` are `Ex`-based.
- `FormalPowerSeries` is `Ex`-based (`from_coefficients`, `coefficient`,
  `general_term`, arithmetic) instead of `Ratio`-based.
- `std::iter::Sum` / `Product` for `Ex` **panic on an empty iterator** (there
  is no context to build `0`/`1` in).  Use `Context::sum` / `Context::product`
  or collect into `Option<Ex>`.
- Plotting methods (`plot_data`, `textplot`, `to_svg`, `to_tikz`, `eval_table`)
  return `Result` instead of panicking or producing empty output.
- `SimplifyOpts::trace` is honoured; use `Ex::simplify_traced` to obtain the
  steps.
- `Assumption` gained `ExtendedReal` and the negated variants (`NotPositive`,
  `NotZero`, …); `match` statements on it need updating.

**Matrices**

- `Matrix::{eigenvals, eigenvects, diagonalize, jordan_form, matrix_exp}` take
  no dummy variable; the eigenvalue symbol is internal.  Use `char_poly(&λ)`
  when you want a named variable.
- `Matrix::is_diagonalizable` and `Matrix::is_symmetric` return `Option<bool>`.
- `Matrix::cholesky` and `Matrix::lu` return `Result` (`InvalidArgument` for a
  non-symmetric / non-square matrix, `ComputationFailed` when not positive
  definite).
- `Matrix::minor(i, j)` returns `Result<Ex>` (the determinant of the minor);
  the sub-matrix is `Matrix::minor_matrix(i, j)`.
- `Matrix::from_i64(ctx, rows)` returns `Result` (ragged rows are an error).
- Removed: `add_elementwise`, `sub_elementwise`, `try_identity`, `try_zeros`
  (use `add`/`sub`/`identity`/`zeros`).
- `StateSpace::poles()` takes no variable.
- `vector::{is_conservative, is_irrotational, is_solenoidal}` return
  `Option<bool>` (three-valued) instead of `bool`.

**Sets & logic**

- `SetEx::contains(&elem)` is set membership returning `Option<bool>`
  (previously structural containment).
- `BoolEx::eval` folds relations through the assumption system
  (`pos > 0` → `True` for a positive-assumed symbol).

**Fixed behaviour that may change results**

- `fourier_series` (and the new `fourier_series_on`) returns correct closed
  forms for `|x|`, `sign(x)` and piecewise inputs (coefficients are exact
  definite integrals).
- One-sided limits fall back to a two-sided `Limit` node when unevaluated.
- Factoring is no longer limited by `MAX_KRONECKER_DEGREE`: `factor` uses
  Berlekamp–Zassenhaus and handles any degree.
- `OdeType` gained `NthOrderLinearConstCoeff`, `Clairaut`, `Riccati`,
  `HomogeneousCoefficient`, `IntegratingFactor` (exhaustive matches break).

### Added

**Core nodes and constants**

- Complex analysis: `Re`, `Im`, `Conjugate`, `Arg` nodes with
  `Ex::{re, im, conjugate, arg, as_real_imag, expand_complex, polar,
  abs_squared, is_real_valued}`; conjugation distributes over `Add`/`Mul`/
  integer powers and commutes with real-analytic functions at construction.
- Constants `EulerGamma`, `Catalan`, `GoldenRatio`
  (`Context::{euler_gamma, catalan, golden_ratio}`) and
  `Context::complex_infinity()` (`zoo`; `1/0` evaluates to it).
- Special functions `si`, `ci`, `ei`, `li`, `zeta`, `polygamma(n, x)`,
  `kronecker_delta(i, j)` with exact values (`ζ(2m)`, `ζ(0)`, `ζ(−n)`,
  `ψ⁽ⁿ⁾(1)`, `Si(∞)`), derivative rules and arbitrary-precision `evalf`.
- `evalf` for `besseli` / `besselk` and orthogonal polynomials of any degree;
  derivative rules for Bessel functions and orthogonal polynomials.
- The parser accepts the new names (`re`, `im`, `conjugate`/`conj`, `arg`,
  `si`, `ci`, `ei`, `li`, `zeta`, `polygamma`, `kronecker`, `zoo`).
- `ExprNode::DefiniteIntegral(body, var, lo, hi)`: a bounded unevaluated
  integral (`Integral(f, x, a, b)`, LaTeX `\int_a^b f\,dx`).  It round-trips
  through Display/parse/JSON, binds its variable for `free_symbols`/`subs`,
  differentiates by the Leibniz rule, evaluates numerically via Gauss–Kronrod
  quadrature in `eval_f64`, and is resolved innermost-out by
  `integrate_definite` / `Ex::eval_integrals`.  Also `Ex::definite_integral_node`,
  `Ex::is_definite_integral`.

**Numeric backends**

- `compile()` covers every numerically evaluable node: Γ, lnΓ, ψ, erf/erfc,
  Lambert W, Beta, factorials and binomials, Bessel J/Y/I/K, orthogonal
  polynomials, integer sequences (`fibonacci`, `lucas`, `harmonic`, …),
  `min`/`max`/`floor`/`ceiling`/`sign`/`heaviside`/`atan2`, piecewise and
  boolean conditions.
- `Ex::compile_many` → `CompiledFnVec` (shared CSE across outputs;
  `call`, `call_vec`, `try_call`, `arity`, `len`).
- `to_rust_fn` embeds a self-contained `mod symplex_rt` runtime with only the
  special-function helpers the expression uses.
- `CodegenOptions::{use_mul_add, checked_domain, emit_runtime}` and
  `CodegenOptions::{runtime_module, c_runtime}` for multi-function files.
- **C99 backend**: `Ex::to_c_fn` / `to_c_fn_with_options`: `#include <math.h>`,
  `static inline symplex_*` helpers, `fma`, `float` precision with
  `f`-suffixed calls, `assert` domain checks, piecewise → ternary chains.
- Deterministic CSE (post-order numbering, cheap-node threshold, no boolean
  temporaries) and `Ex::cse_many`.

**Integration**

- `Ex::integrate_definite` / `try_integrate_definite`: interior
  singularities, infinite bounds, endpoint singularities via one-sided
  limits, symmetry shortcuts, `Piecewise` / `Abs` / `Sign` / `Heaviside` /
  `DiracDelta` integrands, and a ~30-entry table of classical improper
  integrals (Gaussian, Dirichlet, Fresnel, `x/(eˣ−1)`, `ln x`, Γ, …) with
  symbolic parameters under assumptions.
- `Ex::integrate_numeric` / `integrate_numeric_with` (adaptive Gauss–Kronrod
  G7/K15, `QuadOpts`, infinite bounds); `definite::quadrature` for plain
  `Fn(f64) -> f64`.
- Residues at poles of any order; `Ex::residue_at_infinity`.
- ~20 new indefinite-integration families.

**Summation and series**

- `Ex::summation` / `try_summation`, `product_over` / `try_product_over`,
  `hypergeometric_ratio`: Faulhaber sums of any degree, telescoping,
  binomial sums (`Σ P(k)·C(n,k)·xᵏ`), p-series (`ζ(2m)` exact, `zeta(p)` for
  odd `p`, Catalan's constant), power-series recognition (`Σ xᵏ/k! = eˣ`),
  Gosper with a polynomial-time normal form, infinite products.
- `Ex::is_convergent` / `is_absolutely_convergent` (decisive answers only).
- `Ex::series_at_infinity` / `series_at_neg_infinity`.
- `FormalPowerSeries`: lazy exact coefficients, `general_term`, `add`, `mul`,
  `compose`, `inverse`, `reversion`, `derivative`, `integral`.
- `Ex`-based finite differences (`finite_diff_weights`, `apply_finite_diff`,
  `equispaced_grid`, `Ex::differentiate_finite`).

**Solving**

- `Ex::solve_general` → `GeneralSolution` (periodic families with a fresh
  integer parameter, `instance(k)`).
- `linsolve`, `linsolve_matrix`, `LinearSolution`, `ZeroForm` (accepts `Ex`
  or `Equation`), symbolic coefficients, parametric solutions.
- Algebraic solutions in `polysys::solve_system_ex`.
- `solve_numeric_system` / `solve_numeric_system_with` (damped Newton,
  `NewtonOpts`).
- `Ex::solve_ode_ivp`, nth-order constant-coefficient ODEs, Clairaut,
  Riccati (`Ex::solve_riccati`), `ode::solve_ode_system_ivp`.
- `rsolve::rsolve_linear` / `rsolve_first_order` for recurrences.
- Inequalities with absolute values (`|x − 1| < 2`).

**Sets and logic**

- `SetEx::{simplify, eval, difference, symmetric_difference,
  absolute_complement, contains, is_subset, is_superset, is_disjoint,
  is_empty, inf, sup, measure, boundary, closure, interior, is_open,
  is_closed, as_intervals, as_finite_set, to_condition}`; `Ex::is_in`.
- `reduce_inequalities(&[BoolEx], &x) -> Result<SetEx>` and
  `BoolEx::solve_for`.
- `BoolEx::{simplify, to_nnf, to_cnf, to_dnf, is_tautology,
  is_contradiction, satisfiable, atoms, truth_table}` (DPLL with unit
  propagation; declared assumptions respected).
- `Ex::piecewise_simplify`.
- `Props::EXTENDED_REAL`, `Assumptions::implies`, `Assumption::negate`,
  `Display for Assumptions`.

**Matrices, vectors, quaternions, control**

- Eigen family without a dummy variable; `eigenvals_with_multiplicity`,
  `char_poly_coeffs`, `matrix_exp_t`, `matrix_pow_symbolic`, `matrix_sqrt`.
- `RootOf` eigenvalues for irreducible cubics/quartics without a compact
  radical form (exact, numerically evaluable, no Cardano swell);
  `EXPRESSION_BUDGET` swell guard.
- `qr`, `gram_schmidt`, `ldl`, `hessian`, `wronskian`, `adjoint`,
  `is_hermitian`, `is_orthogonal`, `is_unitary`, `is_positive_definite`,
  `is_positive_semidefinite`, `is_nilpotent`, `is_skew_symmetric`,
  `is_upper_triangular`, `is_lower_triangular`, `is_diagonal`, `is_identity`,
  `is_zero`, `norm_1`, `norm_inf`, `norm_p`, `solve_least_squares`,
  `rowspace`, `left_nullspace`.
- Ergonomics: `Index<(usize, usize)>` / `IndexMut`, `TryFrom<Vec<Vec<Ex>>>`,
  scalar operators on both sides (`2 * &m`, `&m * 2`, `m / 2`), `Neg`,
  `col`, `diagonal`, `submatrix`, `set`, `iter`, `to_vec`, `vec`, `eval_f64`,
  `equals`, `map_indexed`, `block_diag`, `hadamard`.
- `Quaternion`: arithmetic operators, `slerp`, `exp`/`ln`/`pow`,
  `rotate_vector`, `to_euler`/`from_euler`, `from_rotation_matrix`,
  `to_axis_angle`.
- `vector::CoordinateSystem` with cylindrical/spherical `gradient_in`,
  `divergence_in`, `curl_in`, `laplacian_in`; `directional_derivative`,
  `line_integral_scalar`, `line_integral_vector`, `scalar_potential`.
- `TransferFunction::to_state_space`, `StateSpace::to_transfer_function`.

**Number theory and polynomials**

- Berlekamp–Zassenhaus `factor` for any degree; multivariate `factor_all`
  (Kronecker substitution); `factor_list`, `factor_list_all`.
- `Ex` polynomial algebra: `resultant`, `discriminant`, `sqf_list`,
  `square_free_part`, `is_squarefree`, `is_irreducible`, `poly_div`,
  `poly_quo`, `poly_rem`, `poly_gcdex`, `decompose`, `content_primitive`,
  `leading_coeff`, `monic`, `poly_compose`, `poly_shift`, `poly_reverse`,
  `poly_interpolate`, `count_real_roots`, `roots_count_real`,
  `real_roots_isolate`, `nroots`.
- `ntheory::factorint` (Pollard–Brent rho with Montgomery `u128` arithmetic
  + ECM for `BigInt`), BPSW `isprime`, `is_probable_prime`,
  `jacobi_symbol`, `kronecker_symbol`, `is_quad_residue`, `sqrt_mod`,
  `sqrt_mod_all`, `discrete_log`, `n_order`/`multiplicative_order`,
  `primitive_root`, `is_primitive_root`, `primepi`, `prime`, `primerange`,
  `carmichael_lambda`, `perfect_power`, `is_mersenne_prime`,
  `continued_fraction`, `continued_fraction_periodic`,
  `continued_fraction_convergents`, `egyptian_fraction`, `digits`,
  `is_palindromic`.
- Sequences: `fibonacci`, `lucas`, `bernoulli`, `euler_number`, `harmonic`
  (ntheory); `bell`, `catalan`, `derangements`, `partitions` iterator
  (combinatorics); symbolic `Ex::{fibonacci, lucas, bell, catalan_number,
  bernoulli_number, euler_number, harmonic, partition_count}`.
- `diophantine::{linear_diophantine, linear_diophantine_n, pell,
  pell_solutions, pell_negative, sum_of_two_squares, sum_of_four_squares,
  pythagorean_triples, frobenius_number}`.

**Simplification and rules**

- Public rewrite-rule engine: `Rule` (template, guarded, closure RHS),
  `RuleSet`, `Bindings`, `RewriteOpts`, `RewriteStrategy`, `Step`;
  `Ex::{rewrite, rewrite_once, rewrite_traced, rewrite_with,
  rewrite_with_traced, simplify_with_rules, simplify_traced}`;
  `RuleSet::standard(&ctx)`.
- AC matching for `Add`/`Mul` with `rest__` sequence wildcards and a
  bounded backtracking budget; `rule!` macro rules usable via
  `Rule::from_macro_rule` / `RuleSet::from_macro_rules`.
- `sqrtdenest`, `signsimp`, `powdenest(force)`, `expand_with(ExpandOpts)`,
  `expand_power_base`, `expand_power_exp`, `expand_multinomial`,
  `log_combine_with`, `expand_log_with`, `nsimplify`,
  `nsimplify_with_constants`, `rcollect`, `collect_const`,
  `separate_vars_additive`, `separate_vars_dict`, `subs_algebraic`.
- 15 trig/hyperbolic identity rules; `vars!` macro (alias of `syms!`).

**Transforms and limits**

- `Ex::{limit_dir, limit_left, limit_right}` + `try_` twins and
  `Direction`; Gruntz work budget; many limits fixed or newly solved.
- `fourier_transform` / `fourier_transform_with` /
  `inverse_fourier_transform[_with]` with `FourierConvention`
  (non-unitary angular, unitary angular, ordinary).
- `mellin_transform` (returns the fundamental strip as a `BoolEx`) and
  `inverse_mellin_transform`.
- Laplace table extensions (`f(t)/t`, Bessel, `t^n e^{−at}`, …), inverse
  extensions (`1/√s`, shifted `e^{−as}F(s)`), `laplace_initial_value`,
  `laplace_final_value` (`Err(Divergent)` for unstable poles).
- `FourierSeries` with `fourier_series_on(var, lower, upper, n)`,
  `coefficient_a/b/c`, `truncate`, `omega0`.
- Z-transform table and inverse extensions.

**Ergonomics**

- `Context::{from_f64 (exact dyadic), from_f64_approx, from_f64_nice,
  from_bigint, from_ratio, from_i128, from_u64, rational_str, decimal_str,
  complex, symbols, symbols_indexed, apply, sum, product}`.
- Operators with `f64`, `i32`, `u32`, `u64`, `i128`, `BigInt`, `Ratio`;
  compound assignment (`+=`, `*=`, …); `ToEx` and `Scalar` traits.
- `Ex::{as_rational, as_bigint, as_i64, compare_numeric, is_less_than,
  is_greater_than, probably_equal, eval_at, subs_map_with}`.
- `Equation` accessors (`lhs`, `rhs`, `swap`, `to_zero_equation`, `to_expr`),
  arithmetic with scalars and equations, `solve` / `solve_for` /
  `solve_or_empty`, `subs`, `is_satisfied`, `is_identity`, `apply`.
- `base::numeric::{f64_to_ratio_exact, f64_to_ratio_approx}`.
- `symplex-wasm`: `Session` (persistent context with `define`) and a full
  stateless API (`integrate_definite`, `to_c_fn`, `eval_decimal`, …).
- `symplex-build`: exact DH parameters via `from_f64_approx`,
  `RobotArmBuilder::generate_fk_matrix`, `"fk_matrix"` in TOML configs.

### Fixed

**Found by the new SymPy oracle and fixed before release**

- Inequality solver: poles are now sign-change points, both-negative
  branches are kept, and the natural domain is intersected in
  (`1/x > 2` → `(0, 1/2)`, `(x−1)/(x+1) ≥ 0` → `(−∞,−1) ∪ [1,∞)`,
  `√x < 2` → `[0, 4)`).  Undecidable cases return `ConditionSet`, never a
  guess.
- `solve_system_ex` returned non-solutions (Cardano emitted `cbrt` of a
  negative radicand, evaluated on the principal branch) and `Ok([])` for
  biquadratic eliminants (Ferrari `0/0`).  Every returned tuple is now
  verified against all equations at 30 digits.
- `rsolve_linear` hung on irrational cubic characteristic roots; roots are
  now `RootOf` values and constant fitting is budgeted.
- `series_at_infinity(atan x)` returned the garbage `atan(zoo)`; constant
  terms are now limits, and unevaluable results are formal `Series` nodes.
- `evalf` Bessel `J`/`Y` were wrong for `x ≳ 12` (doubled leading Hankel
  term, sign error in the recurrence, premature series→asymptotic switch);
  now 25+ digits at any `x`.
- `eval()` of `Piecewise` selected a later `True` branch over an earlier
  undecided one.
- `0 · oo` / `0 · zoo` were order-dependent (`nan` vs `0`).
- Debug-build panic (nested `Mul`) when multiplying numeric radicals such as
  `(√6/3)·(√3/3)`.
- Display of rational/negative bases: `(2/3)^x` printed as `2/3^x`.
- `free_symbols` counted bound index variables of `Sum`/`Product`/`RootOf`/
  `RootSum`/`ConditionSet`/`DefiniteIntegral` as free.
- `eval_decimal` truncated instead of rounding the last digit.
- `eval_f64` on compound expressions with free symbols reported a cache
  miss instead of `FreeSymbol { name }`.
- Assumption lattice: `oo` is positive, extended-real and infinite but not
  real/finite; queries are order-independent; contradictory declarations
  panic with a clear message.
- `nroots` missed real roots of odd/even polynomials (mirror-symmetric Aberth
  start points); real roots are snapped only after an exact Sturm count.
- Expression construction was proportional to tree size (sort keys
  concatenated whole subtrees); keys are now bounded and hashed, and the
  debug canonical-form verifier is iterative (deep expressions no longer
  overflow the stack).
- `sqrt(<large integer>).eval()` trial-divided to `√n` (14 s); square factors
  are now found via bounded `factorint`.  Radical normal form unified:
  `√(1/2) = 1/√2 = √2/2`, `√(4/9) = 2/3`, `∛54 = 3∛2`.
- `Context::rational(p, 0)` panicked; it now returns `zoo` (`nan` for
  `0/0`).
- `abs(3 + 4i)` folds to `5`.
- Generated `no_std` code called `libm::abs` (does not exist); now `fabs`.
  `symplex-build` emitted the `symplex_rt` runtime once per function.
- `expr_type()` reported `RootOf` as unevaluated; `piecewise_simplify`
  ignored assumption-decided conditions; `BoolEx::simplify` gained
  consensus.
- Parser: `binomial`, `beta`, `bessel{j,y,i,k}`, `cot/sec/csc/coth/sech/csch`,
  `min`/`max`, `polygamma`, `Sum`/`Product`, `Integral(f, x[, a, b])`, `n!`.
- `expr!(ctx, 2^10)` (purely numeric bodies) now compiles.

**Other**

- `∫₋₁¹ dx/x²` and other integrals across interior poles no longer return a
  finite value.
- `fourier_series` coefficients for `|x|`, `sign(x)` and piecewise inputs.
- Sign error in shifted alternating half-integer p-series.
- Gosper: dispersion via a bounded gcd scan instead of resultant
  interpolation (`Σ k⁸·2ᵏ` from 23 s to 26 ms); certificate degree cap.
- Gruntz limits: wrong answers for several `exp`/`ln` towers; work budget
  prevents hangs.
- Binomial series for large `|n|`; series at hidden valuations (`1/x` at
  order 1).
- `matrix_exp` for numeric complex eigenvalues (`sin(−1)` parity);
  Jordan chains for repeated eigenvalues; nilpotent blocks.
- Real-root parity in the polynomial root counter.
- `factor_zassenhaus` on non-square-free input.
- Log-to-real exactness guard; polar-form complex powers.
- Definite integrator rejects leaked limit-engine dummies; assumption-decided
  `Piecewise` branches.
- `eval_f64_with` reports `FreeSymbol` for unbound symbols.

### Infrastructure

- CI rewritten: fmt / clippy / test / UI compile-fail (pinned toolchain
  `1.95.0`, `TRYBUILD=overwrite` to refresh snapshots) / sub-crates
  (`symplex-macros`, `symplex-build`, `symplex-wasm` native + `wasm32`, fuzz
  build) / docs (`cargo doc -D warnings` + `mdbook build`) / MSRV `1.93.0` /
  every non-interactive example run.
- GitHub Pages deployment of the mdBook.
- `tests/v02_*` integration suites per area, one concept per test and
  each under a few seconds; SymPy 1.14 oracle (`tests/fixtures/*.json`,
  ~1,400 fixtures, one `#[test]` per subcategory, strict-xfail known-bug
  tables); `tests/README.md` documents the layout and how to regenerate
  fixtures.
- Crate, `symplex-macros`, `symplex-build` and `symplex-wasm` at 0.2.0.

## [0.1.0]

Initial public release: exact arithmetic on `Ratio<BigInt>`, hash-consed
expression arena, differentiation, indefinite integration (Risch,
Rothstein–Trager, Lazard–Rioboo–Trager, heuristics), Gruntz limits,
series, Laplace and Z-transforms, polynomial solving through quartic with
`RootOf`/`RootSum`, Gröbner bases, 13 ODE classes, symbolic matrices with
eigenvalues/Jordan form/matrix exponential, algebraic number fields ℚ(α),
Rust code generation with CSE, compile-time dimensional analysis, and the
`symplex-macros`, `symplex-build` and `symplex-wasm` companion crates.

[0.2.0]: https://github.com/cgorski/symplex/releases/tag/v0.2.0
[0.1.0]: https://github.com/cgorski/symplex/releases/tag/v0.1.0
