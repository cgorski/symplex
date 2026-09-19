# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Until 1.0, minor releases may contain breaking changes; they are listed first.

## [0.7.1] - 2026-09-19

### Changed

- **Hybrid arithmetic in the exact simplex.**  The fraction-free tableau
  is generic over its cell type and runs on `i64` cells first, then `i128`
  cells with exact 256-bit intermediates (a hand-rolled 128×128→256
  multiply, Jebelean exact division by the modular inverse, 256-bit
  comparison), and only on the first value that does not fit does it
  solve the problem again on `BigInt` cells.  Every decision is a sign
  test or an exact comparison of products, so all three take the same
  pivot path and give the same answer — verified byte-for-byte on 4,000
  random LPs and on the pinned Mathlib fixtures.  Measured on a
  downstream generator: the entries of its final tableaux have median
  67 bits and p90 99, so `i64` alone fit 38% of its 18,000 LPs and `i128`
  fits 99%; certificate LP time 8.3 s → 2.3 s, the whole run 33 s → 25 s.
- **Anti-cycling policy.**  Bland's rule used to take over permanently
  after the *first* degenerate pivot; the certificate LPs are degenerate
  from the start (zero right-hand sides), so they walked Bland's slow
  path throughout.  Dantzig's rule now stays in force until twelve
  consecutive degenerate pivots, then Bland's rule runs until the next
  improving pivot — still provably finite.  Handelman degree 8 on a
  2-variable box: 34,763 pivots / 92 s → 3,683 pivots / 11 s.  Optimal
  objectives and statuses are unchanged on the 4,000-LP check; 24 of them
  now report a different (equally optimal) vertex or a different (equally
  valid) Farkas vector, and one facet of the n = 5 floor generator's
  output uses a different hypothesis set — the generated file compiles
  against Mathlib.
- `PolyhedronProver` emits a `tracing::debug!` event per stage LP
  (`symplex::certificates::polyhedron`: degree, rows, cols, status,
  microseconds); `linprog` reports fallbacks to `BigInt` at `debug` and
  final tableau growth at `trace` under `symplex::linprog::growth`.

## [0.7.0] - 2026-09-19

### Breaking

- **One `certificates::Outcome<C, U>` for every prover.**  `BoxOutcome`,
  `HalfLineOutcome`, `PolyhedronOutcome` and `SosOutcome` are now type
  aliases of it (`Proved(C)` / `Refuted { point, value, param_value }` /
  `Unknown(U)`), so `PolyhedronOutcome::Proved(c)` still reads as before.
  What changes: `Refuted.point` is `Vec<(Ex, Q)>` everywhere (was
  `Vec<Q>` for boxes and `Q` for half-lines) and the variant is
  `#[non_exhaustive]` — patterns need `..`; the `Unknown` payloads are the
  structs `BoxUnknown { farkas, degree }`, `HalfLineUnknown {
  max_polya_power }`, `PolyhedronUnknown { degree, lambda_degree, pairwise
  }`, `SosUnknown { reason }` (all `#[non_exhaustive]`, all `Display`) —
  patterns become `Unknown(u)`.
- The Handelman box certificate struct `certificates::Certificate` is
  renamed **`BoxCertificate`** (`CertificateData` → `BoxCertificateData`);
  `certificates::Certificate` is now the **trait** (`goal`, `verify`,
  `to_lean`, `to_lean_with`, `to_json`, `from_json`) implemented by all
  five certificate types.  Inherent methods are unchanged.
- `lean::LeanOpts`, `certificates::PolyhedronOpts` and
  `certificates::SosOpts` are `#[non_exhaustive]`: struct literals
  (including `..Default::default()`) no longer compile outside the crate;
  use `::default()` with the `with_*` builders or assign fields on a `mut`
  default.  Adding an option is no longer a breaking change.
- Functions that could panic on their arguments now return `Result`
  (found by the panic audit below): `StateSpace::{controllability_matrix,
  observability_matrix, discretize_zoh}` (were infallible),
  `StateSpace::{riccati_residual, ackermann}` (were `Option`),
  `robotics::homogeneous`, and `dynamics::{total_time_derivative,
  euler_lagrange, mass_matrix, christoffel_symbols, coriolis_matrix,
  manipulator_equation}`.  `StateSpace::char_poly` and
  `matrix_decomp::wronskian` return NaN instead of panicking on an
  ill-shaped model / empty list (new `try_char_poly`, `try_wronskian`
  return the error); `is_controllable` / `is_observable` are `false` for an
  ill-shaped model; `ode::solve_ode_system{,_nonhomogeneous}` return `None`
  where they could panic.

See `book/src/reference/migrating-0.7.md` for the one-line fix to each.

### Added

- `Outcome::{is_refuted, is_unknown, into_certificate, refutation, unknown,
  map_certificate}`; `Display` for outcomes.
- `RealLineCertificate::{goal, to_data, from_data, to_json, from_json}`,
  `RealLineCertificateData`, and `Display`.
- Builders `PolyhedronOpts::{with_max_degree, with_max_lambda_degree,
  with_pairwise, with_staged}` and `SosOpts::{with_max_basis,
  with_max_iterations, with_rounding_digits, with_max_facial_reductions}`.
- **No-panic policy and ratchet.**  `CONTRIBUTING.md` spells out the
  practical policy (validate at the boundary, `Result` for failure, `Option`
  for absence, `debug_assert!` for invariants, `std`-style `try_` siblings
  for indexing; error plumbing measured at zero cost);
  `tests/unit/test_no_panics.rs` counts `unwrap`/`expect`/`panic!`/
  `unreachable!` in library code and fails on any increase — or on an
  allowlist that is no longer tight.  108 sites removed; the allowlist is
  the two documented logic errors, the arena's `u32` index conversion and
  the compile-time `const_assert_dim!`.

### Fixed

- `matrix_decomp::wronskian` panicked when the derivatives exceeded the
  expression budget (reachable from user input); it now returns NaN and
  `try_wronskian` reports the error.

### Infrastructure

- `symplex` and `symplex-build` at 0.7.0; `symplex-macros` unchanged at
  0.3.0.  Pinned Mathlib-compiled fixtures byte-identical to 0.6.1.

## [0.6.1] - 2026-09-19

### Added

- `PolyhedronCertificate::used_hyps()`: the indices of the hypotheses the
  identity actually uses, so a generated lemma can list exactly those in
  its signature (previously recoverable only by scanning the emitted Lean
  for hypothesis names).
- `PolyhedronProver::prove_poly(&Poly)`: prove a goal that is already an
  exact polynomial, skipping the expression round trip; its generators may
  be any subset of the prover's in any order (a tool's `(j, r, t)` against
  the prover's sorted `(r, t, j)`).
- `MultiPoly::{as_constant, affine_form, eval_var, to_ex}`: the value of a
  constant polynomial; a degree-≤ 1 polynomial as `(coefficients,
  constant)`; substitution of a value for one variable that **keeps** the
  variable count (unlike `substitute`, which drops the variable and shifts
  the indices — the natural operation for instantiating a parameter); and
  the bridge to an `Ex` over named symbols.  `MultiPoly` and the rational
  type `Q` are re-exported from the prelude.
- `Polytope::is_full_dimensional()` and `Polytope::interior_point()`: one
  exact LP (the largest common slack), instead of testing `volume() > 0`;
  defined for unbounded polyhedra too.  `HalfSpace::{value_sign, is_tight,
  is_trivial, normalized, same_hyperplane}` — gcd-free sign tests and the
  canonical hyperplane key that identifies a cut with its flip and its
  rescalings.

### Changed

- **`Polytope::vertices` is 10× faster and cached.**  The enumeration runs
  in integer arithmetic throughout: half-spaces are scaled to integers
  once, only *distinct* hyperplanes are combined, each `n × n` system is
  solved by the fraction-free kernel (which yields the point as `X / D`
  directly) and containment is the sign of `a·X + b·D` — no rational
  reduction until the accepted vertices are returned.  The vertex list is
  cached on the polytope (`Clone` carries it; `PartialEq`/`Debug` ignore
  it).  `Polytope::volume` enumerates vertices once and hands each facet
  its own vertices (those on its hyperplane, projected) instead of
  re-enumerating at every level of the recursion; `is_bounded` recognises a
  description with axis-parallel bounds on every coordinate without LPs.
  `HalfSpace::contains` / `Polytope::contains` use the gcd-free sign test.
  Profiled on a downstream decision-tree generator (three free
  coordinates, 49 leaves): 103 s → 34 s with byte-identical output; the
  remaining time is the certificate LPs.
- `ParametricPolytope::at` instantiates through exact `MultiPoly`
  arithmetic rather than the expression arena (identical results; verified
  against the symbolic route on random families).

### Fixed

- `lean::wrap_lean` measures its continuation indent from the line's
  *tactic column* (past `· ` / `. ` bullets), not from the leading spaces.
  A bullet's tactics sit two columns right of the `·`, so the old `+2`
  put a wrapped `· have … := by tac` continuation at the same column as
  the following tactic — Lean then swallowed that tactic into the inner
  `by` block (`expected '{' or indented tactic sequence`), or rejected a
  wrapped application argument (`unknown tactic`).  Both shapes were
  compiled against Mathlib before and after; plain (non-bullet) lines and
  the pinned certificate fixtures are unchanged.

### Infrastructure

- `symplex` and `symplex-build` at 0.6.1; `symplex-macros` unchanged at
  0.3.0.  Additive over 0.6.0 (`cargo semver-checks`: no semver update
  required).  End-to-end check: a downstream Lean generator built against
  this tree reproduces its 0.6.0 output byte-for-byte apart from the
  `wrap_lean` bullet fix, and the generated file compiles against Mathlib.
- `CONTRIBUTING.md`: no Cargo feature flags by design; the explicit
  context argument of `expr!` and friends is deliberate.

## [0.6.0] - 2026-09-18

### Added

- **Sums-of-squares certificates** (`certificates::prove_sos(goal, &vars,
  &SosOpts)`, `is_sos`): prove `g ≥ 0` on all of ℝⁿ by an exact
  decomposition `g = Σₖ dₖ·pₖ²` with rational `dₖ > 0` and
  rational-coefficient `pₖ` — the class of goals the box, half-line and
  polyhedron certificates could not reach (`(x − 1)² + (y − 1)²`, the
  AM–GM form `x⁴ + y⁴ + z⁴ + 1 − 4xyz`, …).  Outcomes
  `Proved(SosCertificate)` / `Refuted { point, value }` (exact rational
  point, found by a grid and a rationalised numerical minimiser) /
  `Unknown { reason }` (Motzkin's polynomial, odd degree, or a search that
  did not converge — never a wrong `Proved`).
  - The pipeline is Peyrl–Parrilo made exact: the Gram SDP `g = mᵀQm`,
    `Q ⪰ 0` is solved numerically by a small dense primal–dual
    interior-point method (HKM direction, Mehrotra predictor–corrector,
    exact-to-the-boundary steps; no external solver) whose zero objective
    makes it converge to the analytic centre; the solution is rounded,
    projected back onto the coefficient constraints exactly (rational
    least-norm correction) and tested for positive semidefiniteness with
    the rational `QMatrix::ldl_psd`, whose factorisation *is* the
    decomposition.
  - Goals with real zeros have only singular Gram matrices; the search then
    performs **facial reduction**: the numerical kernel is made exact
    either directly (rational kernel) or through its integer relations
    (LLL on the kernel lattice, with Newton-refined zeros of the goal
    providing a double-precision kernel), the problem is restricted to the
    face `Q = B Q' Bᵀ` and re-solved, up to three times.  Sums of two or
    three random squares with irrational common zeros are recovered
    exactly (119 of 120 random cases through degree 6 in two and three
    variables).
  - `SosCertificate::{goal, vars, basis, gram, squares, rank, identity,
    verify, lean_hints, to_lean, to_lean_with, to_data, from_data,
    to_json, from_json}` (`from_*` re-verify), `Display` as
    `goal = d₁·(p₁)² + …`.
  - Lean export: `have h : goal = d₁ * (p₁) ^ 2 + … := by ring` then
    `rw [h]; positivity` — two deterministic steps, no search.  Eight
    shapes (squares with a common zero, positive definite quadratics and
    quartics, a perfect square, a product of squares, univariate, three
    variables, AM–GM) compile against Mathlib (Lean 4.30.0) with
    `linter.style.longLine` on; the emitted text is pinned to that file
    (`tests/fixtures/sos_certificates.lean`).
- `lean::wrap_lean` never breaks between `^` and its exponent.
- `Poly::new` docs point to `try_new` for the failure reason.

### Infrastructure

- `symplex` and `symplex-build` at 0.6.0; `symplex-macros` unchanged at
  0.3.0.  Additive over 0.5.0.

## [0.5.0] - 2026-09-18

### Breaking

- `certificates::PolyhedronOutcome::Refuted` gained the field
  `param_value: Option<Q>` (the sampled parameter value at which the
  counterexample was found).  Patterns must add `..` or bind it.
- `lean::LeanOpts` gained the fields `single_fraction` and `symbol_text`
  (as announced in 0.4.0: use `..Default::default()` or the `with_*`
  builders; literals naming every field break).

### Added

- **`certificates::PolyhedronProver`**: `PolyhedronProver::new(&hyps,
  param, &opts)?` parses the hypotheses and builds every stage's product
  basis once; `.prove(&goal)` / `.prove_empty()` then certify any number
  of goals against them (accessors `hyps`, `gens`, `parameter`, `opts`).
  `prove_nonnegative_on_polyhedron` and `prove_polyhedron_empty` are now
  one-line wrappers over it.  A goal mentioning a symbol absent from the
  hypotheses is an `InvalidArgument`.
- `LeanOpts::symbol_text` (+ `with_symbol_text(name, text)`): render a
  symbol as given Lean text everywhere it occurs — `("J", "(j : ℝ)")` for
  a parameter that is a cast natural in the surrounding proof.  Applied
  by `Ex::to_lean_with` and therefore by every certificate emitter
  (goals, hypotheses, `λ`, `hg`); hypothesis *names* such as `e1J` are
  untouched.  In a full theorem the binder keeps the plain identifier.
- `LeanOpts::single_fraction` (+ `with_single_fraction`): combine over a
  common denominator before rendering, so an `expand`ed rational function
  prints as `(-(8 * j) - 2) / (7 * j + 4)` instead of
  `-(8 * j / (7 * j + 4)) - 2 / (7 * j + 4)`.
- `PolyhedronLeanSteps::to_block(indent)` now re-flows its lines to
  Mathlib's width (indent included) and `to_block_width(indent, width)`
  takes an explicit width; `lean::wrap_lean` never starts a continuation
  line with `:=`, so `have hg : … := by` keeps its `:= by`.
- **`polytope::ParametricPolytope`**: a family `{x : hₖ(j, x) ≥ 0}` with
  half-spaces affine in `x` and polynomial in one parameter.  `at(&j)`
  instantiates exactly; `polytope_at` / `vertices_at` / `volume_at` /
  `is_empty_at` / `contains_at` cache per sample; `clear_cache`.
- `Polytope::volume` works in **any dimension** (was `≤ 3`): exact facet
  decomposition around the vertex centroid, recursing on each facet's
  exact `(n − 1)`-dimensional H-representation; duplicate or rescaled
  facets are counted once.  Verified on hypercubes and simplices up to
  dimension 5 and the 4-D cross-polytope.
- `QMatrix::ldl_psd()` (exact `L·D·Lᵀ` of a PSD matrix, `None` if not
  PSD), `QMatrix::is_positive_semidefinite()`, `QMatrix::is_symmetric()`.

### Infrastructure

- `symplex` and `symplex-build` at 0.5.0; `symplex-macros` unchanged at
  0.3.0.  The three new `lean_steps` skeleton shapes (cast parameter with
  long names and a wrapped `have hg`, emptiness with `K` chains,
  `prefer_subtraction`) compile against Mathlib.

## [0.4.0] - 2026-09-18

### Breaking

- `Ex::roots_count_real` (the 0.3 alias) is removed; call
  `count_real_roots_in` on `Ex` or `Poly`.
- `lean::LeanOpts` gained the field `prefer_subtraction`.  Struct literals
  must add `..Default::default()` (`LeanOpts { real_type: "ℚ".into(),
  ..Default::default() }`) or use the new builders
  `with_real_type` / `with_ascribe_integers` / `with_prefer_subtraction`.
  Further fields may be added in minor releases.

### Added

- **Certificates on a parametric polyhedron**
  (`certificates::prove_nonnegative_on_polyhedron(goal, hyps, Some((&j,
  &j0)), &PolyhedronOpts)` and `prove_polyhedron_empty`): prove `g ≥ 0`
  on `{hₖ(j, x) ≥ 0}` for every real `j ≥ j₀` — or that the set is empty
  — by the exact identity `λ(j)·g = Σ μ·jᵃ(j − j₀)ᵇ·hₖ + Σ μ·jᵃ(j − j₀)ᵇ
  + μ₀` (optionally `+ Σ μ·hₖhₗ`), `λ(j) = 1 + Σ νₐ jᵃ`, all `μ, ν ≥ 0`.
  The polynomial multiplier `λ` on the goal is what makes `j`-dependent
  facets certifiable (their Farkas multipliers are rational functions of
  `j`).  Outcomes `Proved(PolyhedronCertificate)` / `Refuted { point,
  value }` (an exact point of the set, found by sampling `j` and
  minimising an affine goal with the exact LP) / `Unknown { degree,
  lambda_degree, pairwise }`.  The search is **staged** (degree-1
  multipliers with `λ = 1` first, then higher degrees, pairwise products
  last; `PolyhedronOpts::single` for one LP) and every certificate is
  re-verified with exact polynomial arithmetic.  `param = None` gives a
  plain Farkas / pairwise certificate on a fixed polyhedron.
  - `PolyhedronCertificate::{goal, hyps, parameter, terms, lambda,
    lambda_coeffs, lambda_is_one, degree, uses_pairwise,
    proves_emptiness, product_expr, identity, verify}`, `Display` with
    the hypotheses abbreviated `hₖ`.
  - Lean export in the shape a hand-written proof uses: `to_lean(name)`
    emits a theorem with `hj : j₀ ≤ j`, `h0 : 0 ≤ h₀`, …, derives `hJ0`
    / `hK0` when needed, one `have … := mul_nonneg …` per product
    (`h0K`, `h0JK`, `h0xh1`, `pJJ`), and closes with `linarith only […]`
    — through `have hg : 0 ≤ λ * g` and `nonneg_of_mul_nonneg_right`
    when `λ ≠ 1`; emptiness certificates conclude `False`.
    `lean_steps(&PolyhedronLeanNames { hyps, param_nonneg, shift_nonneg },
    &opts)` returns the same `have` lines, hint names and closing block
    (`PolyhedronLeanSteps`, `to_block(indent)`) for an existing proof
    skeleton.  Fifteen shapes (λ = 1; λ of degree 1 and 2 on `j` or on
    `j − j₀`; `j₀ > 0`, `j₀ = 0`, `j₀ < 0`; mixed `J`/`K` chains; pairwise
    products with and without a parameter; emptiness with and without `λ`;
    pure parameter powers `pJJ`, `pJK`, `pKK`; no parameter) compile
    against Mathlib (Lean 4.30.0) with
    `linter.style.longLine` on; the emitted text is pinned to that compiled
    file (`tests/fixtures/polyhedron_certificates.lean`).
- **`symplex::polytope`**: exact convex polyhedra in ℚⁿ from half-spaces
  `a·x + b ≥ 0`.  `Polytope::{new, from_rows, from_exprs, to_exprs,
  halfspaces, contains, with_halfspace, split, is_empty, any_point,
  bounding_box, is_bounded, vertices, irredundant, vertex_centroid,
  volume}` (vertices by `QMatrix::solve` over `n`-subsets, emptiness and
  bounds by the exact LP, volume for `n ≤ 3`); `HalfSpace::{value,
  contains, flipped}`.  Meant for the geometry around certificate
  searches (cells of a decision tree, cuts, containment), not for large
  polyhedra.
- `Poly::try_new(expr, gens) -> Result<Poly, SymplexError>`: `Poly::new`
  with the reason for failure — which generator sits inside a function,
  under a negative power (a rational function), under a fractional or
  symbolic power, or in an exponent.
- `Poly::terms_iter()` (borrowed `(&[u32], &Ex)` pairs, no allocation) and
  `Poly::coeffs_rational() -> Option<Vec<Ratio<BigInt>>>`.
- `LeanOpts::prefer_subtraction`: a sum with exactly one negated term is
  printed as a subtraction with that term last (`(1 / 2 : ℝ) - r` instead
  of `-r + (1 / 2 : ℝ)`), so generated hypotheses match hand-written ones
  syntactically; `with_real_type` / `with_ascribe_integers` /
  `with_prefer_subtraction` builders.
- `HalfLineCertificate::lean_hints(hk, &opts)`: the hint list of the Lean
  proof (`hk`, `pow_nonneg hk n`, `mul_nonneg (sq_nonneg g) (…)`) for a
  caller's proof skeleton, with the caller's name for `0 ≤ k`.
- Certificates cross trust boundaries: `Certificate`,
  `HalfLineCertificate` and `PolyhedronCertificate` have `to_data()` /
  `from_data(&ctx, &data)` (plain serde-derived structs
  `CertificateData`, `HalfLineCertificateData`,
  `PolyhedronCertificateData` built from `ExprTree`s and `"p/q"`
  rationals) and `to_json()` / `from_json(&ctx, json)`.  `from_*`
  **re-verifies** the identity with exact polynomial arithmetic and
  rejects data that does not hold, so a checker can accept a certificate
  produced elsewhere without trusting the producer.
- `linsolve` docs state explicitly that an over-determined but consistent
  system is `Unique` (with an example), a contradictory one
  `Inconsistent`.
- `examples/polyhedron_certificates.rs`; test group `tests/v04.rs`
  (`v04_polyhedron`, `v04_polytope`); book: "What's New in 0.4",
  "Migrating from 0.3 to 0.4", a parametric-polyhedra section in the
  certificates cookbook and a polytope section in the LP guide.

### Infrastructure

- `symplex` and `symplex-build` at 0.4.0; `symplex-macros` unchanged at
  0.3.0.  `cargo-semver-checks --release-type minor` against 0.3.5 reports
  exactly the two breaking changes listed above.

## [0.3.5] - 2026-09-18

### Added

- **Exact matrix core.**  `matrix::{QMatrix, ZMatrix}` (aliases of
  `ExactMatrix<Ratio<BigInt>>` / `ExactMatrix<BigInt>`, both in the
  prelude): dense row-major matrices with no expression arena behind
  them.  Construction (`new`, `from_i64`, `from_fn`, `from_flat`, `zeros`,
  `identity`, `diag`, `row_vector`, `col_vector`), access (`get`,
  `try_get`, `row`, `col`, `diagonal`, `rows`, `iter`, `as_slice`,
  `to_rows`, `into_flat`, indexing), shape ops (`transpose`, `submatrix`,
  `hstack`, `vstack`, `map`), arithmetic (`add`, `sub`, `neg`, `scale`,
  `matmul`, `trace`, operators `+ − *`), `is_zero`, `is_identity`, and
  `Display`/`Debug` in the `Matrix` layout.
  - `QMatrix`: `rref`, `rank`, `nullspace`, `columnspace`, `rowspace`,
    `det`, `inv`, `solve` (square, multiple right-hand sides),
    `clear_denominators`, `to_zmatrix`, `is_integer`, `to_matrix`.  Every
    elimination is **fraction-free** (Bareiss Gauss–Jordan on the
    row-wise integerised matrix): intermediate entries are minors of the
    input, all divisions are exact, and no gcd runs in the inner loop.
  - `ZMatrix`: Bareiss `det`, `rank`, `content`,
    `hermite_normal_form[_with_transform]`, `column_hermite_normal_form`,
    `smith_normal_form[_with_transforms]`, `integer_nullspace`,
    `is_unimodular`, `lattice_determinant`, `to_qmatrix`, `to_matrix`.
  - Conversions: `TryFrom<&Matrix>` for both (constant arithmetic is
    folded first; a symbolic entry is `InvalidArgument`), `From<ZMatrix>
    for QMatrix`.
- `examples/exact_matrices.rs`, `benches/exact_matrix.rs`; book: a
  "0.3.5: the exact matrix core" section on the What's New page, a new
  section in the Matrices guide, performance notes in the LP and lattice
  guides.

### Changed

- `Matrix::{rref, rank, nullspace, columnspace, rowspace, left_nullspace,
  det, inv, solve, solve_least_squares, pinv}`, `linsolve`,
  `linsolve_matrix` and every function in `normalforms` now detect
  all-rational input and run on `QMatrix`/`ZMatrix`, converting back at
  the end.  Results are unchanged (the RREF is unique; parametric
  `linsolve` solutions go through the same `tidy` step and print
  identically); a 30×30 rational `inv` drops from 470 ms to 10 ms, `rref`
  of a 30×36 from 208 ms to 4 ms, `linsolve_matrix` 30×30 from 135 ms to
  1.5 ms.  The `Ex`-based Bareiss determinant that only served numeric
  matrices is gone; symbolic matrices take the same paths as before.
- `linprog`: the simplex tableau pivots on **integers with a common
  denominator** (Bareiss/Edmonds integer pivoting).  Each constraint row
  is scaled once to clear denominators (its artificial gets phase-1 cost
  `1/sᵢ` and the scale is divided back out of the duals and Farkas
  vectors); every pivot keeps the tableau integral; ratio and sign tests
  are cross-multiplied integer comparisons with no gcd in the loop.  The
  entering/leaving choices are made on the same rational values as
  before, so the pivot sequence is the same: on 4,000 random LPs with
  fractional data, degenerate rows, all three relations, free and
  two-sided-bounded variables, `x`, objective, duals and Farkas vectors
  are byte-identical to 0.3.4.  A 40-row × 100-variable program goes from
  1.1 s to 40 ms, 60 × 160 from 2.8 s to 80 ms, and Handelman certificate
  searches run 4–7× faster.
- `normalforms` is now a thin wrapper over `ZMatrix`; error messages and
  conventions are unchanged.

### Infrastructure

- Decision recorded after benchmarking `num-bigint 0.4` against `dashu
  0.6` on the exact-linear-algebra kernels: dashu is ~9× faster on
  Gauss–Jordan over `Ratio` but only 1.2–2× on integer kernels — the gap
  is `Ratio`'s per-operation gcd, not bignum speed.  Fraction-free
  elimination on `num-bigint` beats dashu's rational elimination by 5×
  and the previous code by 40×, so `num-bigint` stays and the public
  `Ratio<BigInt>` types are untouched.
- `tests/v03/v03_exact_matrix.rs`: the exact core against textbook
  Gauss–Jordan, the `Matrix` fast paths against the core, the numeric
  `linsolve` route against the symbolic one, random fractional LPs against
  the exact KKT conditions, Farkas certificates on fractional data, and a
  40×40 rational inverse.  A white-box `linprog` unit test covers the
  tableau's negative common denominator after an artificial is driven out
  on a negative pivot.
- `symplex` and `symplex-build` at 0.3.5; `symplex-macros` unchanged at
  0.3.0.  Additive over 0.3.4.

## [0.3.4] - 2026-09-18

### Added

- Sign facts for univariate polynomials with rational coefficients in a
  real-assumed symbol are now decided exactly (square-free factoring +
  Sturm sequences) when the structural assumption rules are silent:
  `(3u² + 2u + 1).is_positive()` is `Some(true)` for `u ≥ 0`,
  `(x² − 2x + 2).is_positive()` is `Some(true)` for real `x`,
  `(x − 1)²` is `is_nonnegative() == Some(true)` but `is_positive() ==
  None`, `x² − 1` stays `None`.  The symbol's own sign assumptions
  (`Positive`, `NonNegative`, `Negative`, `NonPositive`) restrict the
  domain; an excluded endpoint may be a root (`p² + p > 0` for `p > 0`).
  Everything downstream benefits: `simplify` drops `abs(·)` and folds
  `sqrt(p²)`/`sign(p)`, `ln(p).is_real()`, `compare_numeric`, `BoolEx::eval`
  of `p > 0`.  Guarded to degree ≤ 24; symbols without a `Real` assumption
  (possibly complex) are untouched.
- `lean::wrap_lean(text, width)` and `lean::MATHLIB_LINE_WIDTH`: re-flow
  Lean source at spaces, preferring breaks after commas (hint lists) and
  between binders (signatures), with Lean-compatible continuation
  indentation.  All certificate emitters (`Certificate`,
  `HalfLineCertificate`, `RealLineCertificate`) now produce output that
  passes Mathlib's `linter.style.longLine` — verified by compiling the
  wrapped output with the linter enabled.  `Ex::to_lean` stays single-line
  for embedding.

### Infrastructure

- `symplex` and `symplex-build` at 0.3.4; `symplex-macros` unchanged at
  0.3.0.  Additive over 0.3.3.

## [0.3.3] - 2026-09-18

### Added

- `certificates::prove_nonnegative_on_halfline(goal, var, a, Ray::{AtLeast,
  AtMost}, max_polya_power)`: exact certificates for univariate `goal ≥ 0`
  on `x ≥ a` / `x ≤ a`.  With `k = x − a` the identity is
  `(1 + k)^N · goal = g² · Σ cᵢ kⁱ` with `cᵢ ≥ 0`: `N = 0` is the
  shift-and-read-off-coefficients certificate, `N > 0` a Pólya multiplier
  (always exists for a strictly positive goal), and `g` collects
  even-multiplicity zeros.  Outcomes: `Proved(HalfLineCertificate)`,
  `Refuted { point, value }` (exact, found via root isolation), or
  `Unknown { max_polya_power }`.  `HalfLineCertificate::{verify, identity,
  coefficients, polya_power, square, to_lean}`.
- `certificates::prove_nonnegative_on_reals(goal, var, split, max_polya)`
  → `RealLineCertificate` (two half-lines; Lean proof by
  `rcases le_total split x`).
- Box certificates now handle interior even-multiplicity zeros: when the
  plain Handelman search fails, the goal is factored as `g²·h` (exact
  factoring over ℤ) and `h` is certified; `Certificate::square()` exposes
  `g` and the Lean hints become `mul_nonneg (sq_nonneg g) (…)`.
  `(x − 1/2)²·(1 − xy)` on the unit square is now `Proved`.
- Every Lean theorem emitted by `examples/certificates_to_lean.rs`
  (9 theorems: boxes, half-lines both directions, Pólya exponents 1 and
  12, square factors, the real line) was compiled against Mathlib
  (Lean 4.30.0) with no errors or warnings; the texts are pinned in
  `tests/v03/v03_certificates.rs`.

### Fixed

- `(c·m)^n` with a numeric coefficient `c` and `|n| > 10` was left as an
  opaque power by canonicalisation (the product-power distribution has a
  swell guard at 10), so `(-x)^11` did not become `-x^11`,
  `expand((2 - x)^11)` lost its leading term and `Poly::new` rejected the
  result.  A numeric coefficient is now always pulled out (`(-x)^11 →
  -x^11`, `(2x)^13 → 8192*x^13`); products of symbols keep the guard
  (`(x*y)^12` stays as is) but the polynomial view now recognises such
  powers as monomials (`Poly::new((x*y)^12 + 1)` works).

### Infrastructure

- `symplex` and `symplex-build` at 0.3.3; `symplex-macros` unchanged at
  0.3.0.

## [0.3.2] - 2026-09-18

### Added

- `symplex::certificates` — exact, machine-checkable non-negativity
  certificates on a box.  `prove_nonnegative_on_box(goal, &[(var, lo, hi)],
  degree)` searches for a Handelman certificate
  `goal = Σ λₖ · Π (xᵢ − lᵢ)^a (uᵢ − xᵢ)^b`, `λ ≥ 0`, by exact LP (weighted to
  prefer few, low-degree products), **re-verifies the identity with exact
  `Poly` arithmetic** before returning, and otherwise either refutes the
  claim with an exact counterexample (`BoxOutcome::Refuted { point, value }`)
  or reports `BoxOutcome::Unknown { farkas, degree }` (e.g. when the goal
  touches zero inside the box, where no Handelman certificate exists).
  `Certificate::{terms, product, product_expr, identity, verify, degree}`,
  `is_nonnegative_on_box` (three-valued convenience),
  `Poly::express_as_nonneg_combination(basis)` (the LP step alone), and
  `Ex::prove_nonnegative_on_box`.
- `Certificate::to_lean(name)` / `to_lean_with`: a Lean 4 / Mathlib theorem
  `theorem name (x y : ℝ) (h_x_lo : …) … : 0 ≤ goal := by nlinarith […]`
  whose hints are exactly the certificate's products (`mul_nonneg
  (sub_nonneg.mpr h_x_lo) (sub_nonneg.mpr h_y_hi)`, …); linear certificates
  use `linarith`; unused bounds are underscored for the linter.  Every
  theorem produced by `examples/certificates_to_lean.rs` was compiled
  against Mathlib (Lean 4.30.0) with no errors or warnings.

### Infrastructure

- `symplex` and `symplex-build` at 0.3.2; `symplex-macros` unchanged at
  0.3.0.  Additive over 0.3.1 (`cargo-semver-checks`: 223/223).

## [0.3.1] - 2026-09-18

Driven by field notes from downstream tools built on 0.3.0.  Additive only
(verified with `cargo-semver-checks` against 0.3.0).

### Added

- The exact-arithmetic crates are re-exported — `symplex::num_bigint`,
  `num_rational`, `num_integer`, `num_traits` — so a downstream crate can
  name `Ratio<BigInt>` (from `as_rational`, `linprog::Q`, `Matrix::from_ratio`,
  …) without adding and version-matching those crates itself.
- `Ex::as_ratio_parts() -> Option<(BigInt, BigInt)>` and
  `Ex::as_ratio_i128() -> Option<(i128, i128)>`: numerator and denominator of
  a rational literal (SymPy's `Rational.p` / `.q`).
- `Poly::{count_real_roots, count_real_roots_in(lo, hi), real_roots_isolate,
  is_nonnegative_on(lo, hi), is_positive_on(lo, hi)}` — the Sturm-based
  primitives that were only reachable through `Ex` — and `Poly::shift(gen, a)`
  (Taylor shift; "all coefficients of `p(k + a)` are `≥ 0`" is the
  certificate-style sufficient condition for `p ≥ 0` on `[a, ∞)`).
- `Ex::count_real_roots_in(var, lo, hi)`; the old name `roots_count_real`
  stays as an alias (no deprecation warning in a patch release) and is
  removed in 0.4.
- `Ex::to_lean()` / `to_lean_with(&LeanOpts)` (`symplex::lean`): Lean 4 /
  Mathlib rendering with Mathlib spacing (`2 * j + 1`, `j ^ 2`), ascribed
  rational literals (`(3 / 31 : ℝ)`), single-fraction division
  (`(j - 1) / (2 * j)`), `⁻¹` for negative powers, `Real.sin`/`Real.exp`/
  `Real.sqrt`/`Real.pi`/`|x|`/`⌊x⌋`, relations and connectives for `BoolEx`
  (`0 < x ∧ x < 1`), `if … then … else` for `Piecewise`.  Nodes without a
  standard Mathlib spelling are `Err(NotImplemented)`.

### Behaviour changes

- `Ex::as_numer_denom` follows SymPy: a rational literal splits into
  integers (`3/31` → `(3, 31)`), a rational coefficient splits
  (`2/3·x` → `(2*x, 3)`), and sums are combined over a common denominator at
  every depth (`x/2 + 1/3` → `(3*x + 2, 6)`, `1/x + 1/y` → `(x + y, x*y)`).
  Previously a rational literal was an atom (`(3/31, 1)`) and a sum was
  returned whole.  Still no cancellation (`ratsimp` does that).
- `Ex::together` is deep: fractions nested inside numerators, denominators,
  products and integer powers are flattened into one quotient.  Previously
  only a top-level sum was combined, so `Poly::new` on the numerator of a
  `together()` result could silently see a rational function.

### Fixed

- `eval` was not idempotent on `exp(f)^g`: `(1/exp(-1)).eval()` gave
  `exp(1)`, and only a second `eval` gave `E`.  The rewritten exponent is
  now evaluated in the same pass.
- `laplace_final_value` located poles of `s·F(s)` without cancelling the
  factor `s`, which with the deep `together` would have reported a spurious
  pole at the origin for `F(s) = 3/s − 2/(s + 1)`; it now uses `ratsimp`.

### Infrastructure

- The ~275 integration-test source files are compiled into nine test
  binaries (`tests/{v03,v03_oracle,v02,v02_oracle,unit,legacy,proptests,
  perf}.rs` plus `ui_tests`); every test keeps its name as
  `<module>::<test>`.  Linking dropped from ~6.5 min / 13 GB to seconds, and
  the CI disk-space workaround is gone.  See `tests/README.md` for the
  layout and the `cargo test --test <group> <module>::` / `cargo nextest run
  -E …` invocations.
- `.config/nextest.toml`: `cargo nextest run` executes one process per test
  with per-test wall-clock limits (`default` and `ci` profiles).
- `deny.toml` + a `cargo deny check` CI job enforce the pure-Rust dependency
  policy (no C/C++ or system libraries), a licence allow-list, advisories and
  registry sources.
- Removed the assertion-free `tests/zz_probe_tmp.rs` left over from 0.2.
- `symplex-wasm`: dropped the unused `web-sys` dependency.
- `symplex` and `symplex-build` at 0.3.1; `symplex-macros` is unchanged at
  0.3.0.

## [0.3.0] - 2026-09-18

**Polynomials as data, exact certificates.**  This release makes the
polynomial structure of an expression a first-class object (`Poly`: sparse
terms over explicit generators with symbolic coefficients), gives rational
functions a real normal form (`ratsimp`), and adds three exact
certificate-producing domains: linear programming over ℚ with shadow prices
and Farkas infeasibility vectors, Hermite/Smith normal forms of integer
matrices with unimodular transforms and ℤ-bases of integer kernels, and
Sturm-verified polynomial signs on intervals.  A deterministic `f64`
optimisation toolbox (Brent, Nelder–Mead, differential evolution,
least-squares fits) rounds out the numeric side.  There are no
signature-breaking changes; the behaviour changes below alter the *form* of
some results, never their value.

Measured at release: 92 `ExprNode` variants (unchanged), ~11,000 `#[test]`
functions (~154K lines of tests, 273 integration-test files), ~174K lines in
`src/`, ~600 doctests in the main crate, and a SymPy 1.14 oracle extended
with 455 fixtures for the 0.3 features
(`tests/fixtures/v03_cross_validation.json`).

### Behaviour changes

Not breaking — no signature changed — but results may print differently.

- `Ex::{degree, coeffs, coeff, leading_coeff, is_polynomial}` accept
  symbolic, variable-free coefficients: `(a·x² + x).degree(&x)` is `Some(2)`
  and `coeffs` is `[0, 1, a]` where 0.2 returned `None`.  Rational-coefficient
  results are byte-for-byte unchanged.
- `Ex::solve` on linear and quadratic equations with parametric coefficients
  puts the root (and the quadratic discriminant) into rational normal form:
  `((3r−1)/(j+1) − (r+1)/(2j)).solve(&r)` is `(3*j + 1)/(5*j - 1)` instead of
  a fraction of fractions.
- `Ex::simplify_rational` is now `ratsimp` (one fraction over all variables
  at once, integer-primitive numerator and denominator, positive leading
  denominator coefficient) instead of `together` followed by a per-symbol
  `cancel`.  Same value; nested fractions that 0.2 left uncancelled are
  now cancelled.
- `Display`: a product with a rational coefficient *and* inverse factors is
  printed as one fraction.  `1/2*1/j*(j - 1)` is now `(j - 1)/(2*j)`,
  `4/3*1/pi*sin(3*x)` is `4*sin(3*x)/(3*pi)`, `-1/2*1/(x + 1)` is
  `-1/(2*(x + 1))`, and `x - 3*y*1/z` is `x - 3*y/z`.  Products without an
  inverse factor are unchanged (`1/2*x`), as is `x^(-2)`.

### Added

**Polynomials**

- `poly_ex::Poly` (also `prelude::Poly`) and `Ex::as_poly(&[&gens])`: an
  expression as a sparse polynomial in explicit generators with exact
  rational *or* symbolic coefficients; terms in SymPy's lex-descending
  order.  `Poly::{new, from_terms, zero, one, constant, from_multipoly}`;
  queries `gens`, `num_gens`, `is_zero/is_ground/is_univariate/is_linear/
  is_homogeneous`, `has_rational_coeffs`, `num_terms`, `terms`, `monoms`,
  `coeffs`, `coeff_monomial`, `total_degree`, `degree_in`, `degree_list`,
  `leading_term/leading_coeff/leading_monomial`, `all_coeffs`, `equals`;
  conversion `to_ex`, `to_multipoly`; evaluation `eval` (all generators)
  and `eval_gen` (partial, generator removed); arithmetic `add`, `sub`,
  `mul`, `neg`, `scale`, `pow`, `derivative`, `content_and_primitive`,
  `monic`; `nroots`; `Poly::monomial_basis` and `Poly::coefficient_matrix`
  for turning "goal = Σ λᵢ pᵢ" into an exact linear system; `Display` as
  `Poly(expr, gens…)`.
- `Ex::poly_is_nonnegative_on` / `poly_is_positive_on(var, lo, hi)`: exact
  three-valued sign of a rational-coefficient polynomial on a closed
  interval (square-free part isolates odd-multiplicity roots, Sturm count,
  one sign sample); endpoints may be `±∞`.
- `MultiPoly::{gcd, lcm}` (heuristic GCD, GCDHEU, verified by exact
  division), `integer_content`, `clear_denominators`, `from_terms`,
  `coeff`, `map_coeffs`.

**Simplification**

- `Ex::ratsimp`: rational-function normal form.  `P/Q` over the free
  symbols with every maximal non-rational subexpression (`sin x`, `π`,
  `√x`) treated as an independent indeterminate, `gcd(P, Q)` divided out,
  denominators cleared to integer-primitive parts, leading coefficient of
  `Q` positive.  Idempotent; expressions with `±∞`, `NaN` or unevaluated
  nodes are returned unchanged.

**Linear programming** (`symplex::linprog`)

- `LpProblem` builder (`minimize`/`maximize`, `le`/`ge`/`eq` rows,
  per-variable `bounds`, `free`) and `LpProblem::solve` → `LpSolution
  { status, x, objective, duals, farkas }` with `LpStatus::{Optimal,
  Infeasible, Unbounded}`; `LpSolution::{is_optimal, x_ex}`.  Two-phase
  dense simplex over `Ratio<BigInt>`, Dantzig pivots switching to Bland's
  rule at the first degenerate step (no cycling), pivot cap reported as
  `ComputationFailed`.
- Exact **shadow prices** (`duals`, one per constraint in insertion order,
  `yᵢ = ∂ optimum / ∂ bᵢ`) and exact **Farkas certificates** (`farkas`,
  `Aᵀy`-based inequality proving infeasibility; `None` only when the bounds
  alone are contradictory).  Sign conventions documented in the module docs.
- `linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)` (SciPy-shaped),
  `feasible_nonneg(A, b)` ("is there `x ≥ 0` with `Ax = b`?", exactly),
  `feasible_nonneg_certified` and the column-oriented `nonneg_combination
  (vectors, target)` returning `Feasibility::{Feasible(x), Infeasible {
  farkas }}`, `linprog_matrix(Objective, &c, A_ub, b_ub, A_eq, b_eq)` on
  `Matrix` data with numeric-literal entries, `LpSolution::duals_ex`, a
  one-line `Display` for `LpSolution`, and the literal helpers `q(n, d)`,
  `qi(n)`.
- `prelude` re-exports `LpProblem`, `LpSolution`, `LpStatus`.

**Integer normal forms** (`symplex::normalforms`)

- `hermite_normal_form` (row style, `H = U·A`, unique: positive pivots,
  entries above pivots reduced into `[0, pivot)`, zero rows last),
  `hermite_normal_form_with_transform` → `(H, U)`,
  `column_hermite_normal_form` (`H = A·V`, SymPy / Cohen 2.4.5 convention,
  leading zero columns kept).
- `smith_normal_form` (`diag(d₁, …, dᵣ, 0, …)`, `dᵢ | dᵢ₊₁`) and
  `smith_normal_form_with_transforms` → `(S, U, V)`.
- `integer_nullspace` (a ℤ-basis of `{x ∈ ℤⁿ : Ax = 0}` — generates every
  integer solution, unlike a scaled rational nullspace), `is_unimodular`,
  `lattice_determinant` (index of the column lattice in `ℤᵐ`).
- `Matrix::{hermite_normal_form, smith_normal_form, integer_nullspace}`
  method forms.  Non-integer entries are `InvalidArgument`.

**Matrices**

- Selection: `extract(rows, cols)`, `select_rows`, `select_cols`,
  `delete_row`, `delete_col` (index lists may repeat or reorder; empty or
  out-of-range is `InvalidArgument`).
- Three-valued structure test `is_integer_matrix` (alongside the existing
  `is_zero`).
- Exact numeric conversion: `to_rational_rows`, `to_bigint_rows`,
  `Matrix::from_ratio`, `Matrix::from_bigint`, `Matrix::from_f64_rows`
  (exact dyadic; `NaN`/`∞` rejected).
- `subs_map(&[(&from, &to)])` (simultaneous substitution in every entry),
  `nnz`.

**Number theory**

- `ntheory::{gcd_many, lcm_many}` on `&[BigInt]` (empty list: `0` / `1`;
  early exit at gcd 1; any zero makes the lcm 0), generic `igcd` / `ilcm`
  for any `Into<BigInt>`, and `rational_lcm_of_denominators`.

**Numerical optimisation** (`symplex::optimize`)

- Root finding: `brent_root` (Brent–Dekker), `bisect`, `newton_root`, with
  `RootOpts { xtol, rtol, max_iter }`.
- Minimisation: `nelder_mead`, `minimize_scalar` (Brent), `golden_section`,
  `differential_evolution` (DE/rand/1/bin, Latin-hypercube start,
  Nelder–Mead polish, SplitMix64 seeded by `DeOpts::seed` — fully
  deterministic), with `MinimizeOpts`, `DeOpts` and `MinimizeResult { x,
  fun, iterations, evaluations, converged }`.  An exhausted budget is
  reported through `converged = false`, never by discarding the best point.
- Fitting and helpers: `poly_fit` (column-scaled Householder QR, ascending
  coefficients), `poly_fit_exact` (rational normal equations),
  `linear_fit`, `trapezoid`, `eval_poly`.
- On `Ex`, compiling first: `find_root_bracket[_with]`,
  `minimize_numeric[_with]`, `minimize_scalar_numeric`,
  `minimize_global_numeric`, and `Ex::poly_fit_points` (exact least-squares
  polynomial through rational points).  A stray free symbol is
  `FreeSymbol`, not a silent `NaN`.
- `prelude` re-exports `RootOpts`, `MinimizeOpts`, `MinimizeResult`.

**Examples and tests**

- New examples `polynomials`, `exact_lp`, `integer_lattices`,
  `numeric_optimization`; `readme_snippets` mirrors the new README sections.
- The crate docs list the new modules (`poly_ex`, `linprog`,
  `normalforms`, `optimize`) in the module map.

### Fixed

- `simplify_rational` could not cancel common factors that only appear
  after combining (`(x² − y²)/(x − y)` was returned unchanged) or
  fractions nested inside fractions (`1/(x + 1/y) + 1/(1/x + y)` became
  `(x + 1/x + y + 1/y)/((x + 1/y)*(1/x + y))`); it now returns `x + y` and
  `(x + y)/(x*y + 1)`.
- `solve` returned unsimplified nested fractions for linear and quadratic
  equations whose coefficients are themselves parametric fractions
  (`(1/2*1/j + 1/(j + 1))/(-1/2*1/j + 3/(j + 1))` for the example above).
- `degree`/`coeffs`/`coeff`/`leading_coeff`/`is_polynomial` reported
  "not a polynomial" (`None`/`false`) for polynomials with symbolic
  parameter coefficients such as `a·x² + (a + 1)·x + 3`.

### Infrastructure

- `tests/v03_*` integration suites, one concept per test: `v03_poly_view`,
  `v03_poly_symbolic_coeffs`, `v03_ratsimp`, `v03_linprog` (every `Optimal`
  result checked against the full KKT conditions exactly, every
  `Infeasible` result's Farkas vector verified), `v03_normalforms` (defining
  invariants `H = U·A`, `|det U| = 1`, `S = U·A·V`, divisibility chain,
  `A·k = 0` checked rather than pinned answers), `v03_matrix_ergonomics`,
  `v03_optimize`.
- SymPy 1.14 oracle extended to the 0.3 features (`Poly.terms`/`coeffs`,
  `cancel`/`ratsimp`, `sympy.solvers.simplex`, `hermite_normal_form`,
  `smith_normal_form`) via `tests/fixtures/v03_cross_validation.json`;
  as before, comparisons are numeric or structural, never by printed form.
- Crate, `symplex-macros`, `symplex-build` and `symplex-wasm` at 0.3.0.

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

[0.3.0]: https://github.com/cgorski/symplex/releases/tag/v0.3.0
[0.2.0]: https://github.com/cgorski/symplex/releases/tag/v0.2.0
[0.1.0]: https://github.com/cgorski/symplex/releases/tag/v0.1.0
