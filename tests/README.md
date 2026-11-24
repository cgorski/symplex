# symplex test suite

~590 integration-test source files (~12 100 `#[test]`s) compiled into **36
test binaries**, plus the SymPy oracle described below.  (The crate's inline
unit tests add ~3 000, and `cargo test --doc` runs ~1 370 doctests: the API
examples and the Rust blocks of `README.md` and the book, wired in
`src/doctests.rs`.  Counts at 0.60; `cargo nextest run` runs about 15 130
tests.)  Everything runs with `cargo test --tests`; the recommended fast
path is

```sh
cargo nextest run                       # one process per test, per-test timeouts
                                        # from .config/nextest.toml
cargo nextest run -E 'test(/^v03::/)'   # one binary
cargo nextest run -E 'test(/^unit::test_parser::/)'   # one former file
```

Each former top-level file is now a module of one of the group binaries below,
so a test's full name is `<module>::<test>` inside the `<group>` binary
(`cargo test --test unit test_parser::` runs everything from the former
`tests/test_parser.rs`; `cargo test --test unit poly_` runs every test whose
name contains `poly_`).  Why: 277 separately linked debug binaries took
~6.5 min to link and ~13 GB of `target/`; 36 take seconds.  Up to 0.34 a
new `vNN.rs` group was added per minor release; since then a group spans
several releases, and new tests go into the current one (`v36`) or into
another existing group.

## Layout

| Binary / path | What it is |
|---|---|
| `v09.rs` … `v36.rs` → `vNN/vNN_*.rs` | Feature and regression suites of 0.9 → 0.60, one group per minor release up to `v33` and one per several releases after it (the group number trails the release since 0.15): **v09** `algebraic, calculus_util, matrix, ntheory_discrete, output, special`; **v10** `stats_continuous, stats_discrete, stats_joint`; **v11** `calculus_algebraic_fixes, output_fixes, review_fixes, special_fixes`; **v12** `betainc, numeric_summation, stats_families, stats_transforms`; **v13** `agreement, data_estimation, hypothesis, walkthrough` (the book chapter's numbers); **v14** `multivariate_order, regression, reliability, survival`; **v17** `anova, audit_a/b/c, exact_intervals, pvalues`; **v18** `cox, layout` (the stats placement rules), `loose_ends, poly_fast, roots, samplers`; **v19** `kernel, mnlogit, raters`; **v20** `base, dense, evalf, fits, fp_poly, libfn, matrix_tier, numdist, poly_kernel`; **v21** (0.22) `apart, coverage, integrate, numdist, series, simplify, solve, units` — the regressions of the 0.21 differential audit and tests for public functions no test reached. **v22** (0.23) `evalf, integrate, simplify` — the domain model: principal-branch evaluation, and every rewrite that was right only on the reals or for positive arguments, each pinned at the point where it was wrong; **v23** (0.24) `eval, integrate` — the wrong antiderivatives the Rubi suite (`rubi-harness/`) and `fuzz_integrate` found; **v24** (0.25) `canon, integrate, panics` — the logarithmic part of the rational integrator, memory budgets, panics on caller input; **v25** (0.26) `evalf, radicals` — digits the evaluator can vouch for, canonical radicals; **v26** (0.27) `hypothesis, numdist` — statistics bug hunt round 1; **v27** (0.28) `tails` — far tails of the symbolic distributions; **v28** (0.29) `api, binders, evalf, integrate, perf, stats, subs_matrix` — the known bugs resolved after 0.28 (the integrator's self-check, binder-aware substitution, the evaluator's error bounds, panics on caller input, the statistics hunt continued); **v29** (0.30) `algebra, api_domains, calculus, complex, decisions, evalf, functions, leftovers, leftovers2, limits_series, ode_series, output_ntheory, rewrites, transforms` — the differential hunts (evalf against itself, calculus against numerical oracles, exact algebra against identities, display against parse, compiled code against evalf, rewrites over ℂ, limits against high-precision sequences) and the branch decisions (`loggamma`, compiled roots, the `series` contract); **v30** (0.31) `evalf, hunts, integrate, integrate2, integrate3, integrate4, linalg, parse, polylog, poly` — integration over ℂ and with parameters, integrand normal forms, exact linear algebra with radicals, the systems/sets/sums/transforms hunters, evalf refusals, `polylog` over ℂ; **v31** (0.32) `integrate, linalg, series, values` — the nightly fuzz findings and the 0.31 hand-off; **v32** (0.33) `norms, series, simplify, undefined` — the nightly fuzz findings and the 0.32 hand-off (series through logarithms, complex matrix norms, expressions undefined everywhere); **v33** (0.34) `evalf, integrate, limits, radicals, slow, solve, tiny, undefined` — the nightly fuzz findings and the 0.33 hand-off (huge radicals, branch-cut limits, extreme special-function parameters, values below the exponent range, roots over ℚ(√p…, i), expressions undefined everywhere); **v34** (0.35–0.38) `algebra, audit, beyond, codegen, continuity, continuity2, eval, evalf, integrate, limits, matrices, ode_sums, poly_speed, rewrites, rubi, rubi_trig, slow, undefined, values` — the 0.34 hand-off and the hunts of 0.35–0.38 (zero balls that lost a term, values beyond the exponent range, the code emitters, branch-cut and asymptotic limits, continuous antiderivatives, Rubi coverage, linear algebra, sums and ODEs, rewrite routes); **v35** (0.39–0.46) `applied, assumptions, consistency, decisions, definite, definite2, diff, domains, eval, eval2, eval3, eval4, evalf_complex, evalf_special_complex, evalf_sums, folds, infinity, integrate2, integrate_gaps, integrate_speed, limit_speed, limits, limits2, linalg, linalg2, ntheory, numeric, numint, ode, ode2, poly, quadrature, rational, series, sets, solve, solve2, special, speed, speed2, stats, stats2, sums, sympy_audit, transcendental, transforms, trig_values, units` — transcendental equations, special-function antiderivatives, complex evalf, symbolic eigenvalues, ODEs, evalf of sums and products, numerical integration API, series over ℂ, summation, arbitrary-precision quadrature, rational functions over parameters and number fields, solutions over declared domains, trigonometric values in radicals, consistent decisions about constants, residues and keyhole contours, Green's functions, the cases where symplex does better than SymPy 1.14 that the 0.46 audit of the SymPy claims found unpinned (`sympy_audit`), folds under assumptions, complex limits, decision procedures, infinities, the assumption system, exact special values, control, quaternions, vector calculus, dynamics, root finding, normal forms, units, integrator speed, equation solving and inequalities, integral transforms and definite integrals, number theory and combinatorial numbers, statistics (moments and generating functions), polynomial algebra and systems, solution sets (image sets) hunted. **v36** (0.49–0.60, 87 files, the current group) — 0.49–0.50: `b1_definite, b2_limits, b3_canon_solve, w1_ids, w2_truth, w3_solve, w3b_solve, w3c_solve, w4_numerics, w5_func, w6_facets, w6b_codegen, w6c_lib, w6d_funcs, w7_nodes, w8_truth, w8b_deciders, w8c_truth, w9_budgets, w10_poly, w10b_domains, w10c_domains, w11_evalf, w11b_evalf, w12_perf` — the type-architecture refactor and the bugs it exposed, one file per workstream: identifiers without an order and the checked context boundary, typed memo keys and certified evidence for signs and zeros, explicit solve domains and the `Solutions` outcome, checked casts and the BigFloat boundary (with a property test of astro-float), the `Func` table and its facets, the node-shape derive and canonical construction, deterministic budgets; each test pins a bug the old representation allowed. From 0.51 on a file name starts with the tag of the round of work it comes from (`s15r3_bounds`: round `s15r`, workstream 3), one file per workstream, and every test says what was wrong before and cites its oracle: 0.51 `s14c_residue, s14d_assume, s14e_solve`; 0.52 `s14f1_constants, s14f2_limits, s14f3_definite, s14f4_solve, s14f5_matrix_ode`; 0.53 `s15g1_undefined, s15g2_solve, s15g3_quadrature, s15g4_float_evidence, s15g5_conventions`; 0.54 `s15h1_infinity, s15h2_evalf, s15h3_integration, s15h4_constants, s15h5_float_evidence`; 0.55 `s15j1_solve, s15j2_summation, s15j3_evalf, s15j4_float_evidence, s15j5_integration`; 0.56 `s15k1_silent, s15k2_products, s15k3_stats, s15k4_functions, s15k5_integration`; 0.57 `s15l1_fuzz_stats, s15l2_families, s15l3_tests, s15l4_models, s15l5_numerics, s15l5_oracle` (the statistics hardening: the `fuzz_stats` findings, every family against scipy and mpmath, hypothesis tests and estimators, the models; `s15l5_oracle` is the dumper that writes symplex's values for the nightly statistics oracle, `scripts/stats_oracle/`) and `s15n1_limit_power`; 0.58 `s15m1_value_class, s15m2_divisors, s15m3_alloc, s15m4_perf, s15m5_values` (the five idioms replaced by types and ratchets), `s15n2_fuzz, s15p1_evalf, s15p2_limits, s15p3_queues, s15p4_stats, s15p5_functions, s15q1_silent, s15q2_stats, s15q3_consistency, s15q4_perf, s15q5_functions, s16_fuzz_findings` (values below the exponent range, hidden zeros in limits, limits at tabulated poles, the `numdist` kernels at extreme shapes, the findings of the nightly fuzzing); 0.59 `s15r1_zero_powers` (powers of 0 with symbolic exponents; `0` and `zoo` over step functions), `s15r2_limits` (limits, singularities and root searches; slow limits and series), `s15r3_bounds` (aborts, panics and hangs at huge sizes; the `try_` functions that report a refusal), `s15r4_values` (parts below the exponent range, underflow outcomes, `elliptic_e` at huge arguments, four slow evaluations), `s15r5_stats` (windowed sums and the entropy of large discrete laws, far gamma tails, densities at huge degrees of freedom, integrals of `R(cˣ)` with a complex base), and `s16r6_slow`, a registered stub with no tests yet, for the slow inputs of the nightly fuzzing that are still open (tables, long lattices, cancelling moments); 0.60 `s17a_transforms` (the Laplace, Fourier and Mellin transforms of an expression in which the transform variable also occurs bound — the index of a `Sum`, the variable of an `Integral`, a `Limit` or a `RootOf` — each against mpmath's quadrature of the defining integral, and a test that the four transform modules keep no private copy of the helpers of `calculus/transform_util.rs`), `s17n1_zeros` (10 tests: the digits that `eval_decimal` and `eval_f64` print for `besselj`, `bessely`, `Ci`, `Chi`, the Airy functions and their derivatives, `li`, `besseli` of a negative order and the orthogonal polynomials of degree 90 and more at a dyadic argument next to a zero of the function, each against mpmath at 150 and 230 digits; the values away from the zeros, at a decimal argument next to a zero and next to a zero the recurrence gives exactly are the strings 0.59 printed; an argument beyond the precision budget is refused, not guessed), `s17b_sets` (6 tests: a set of real numbers contains, absorbs and removes only an element proved real — differences and unions with the real line, intersections and complements, `UniversalSet` is not the real line, membership and subsets, a constant whose imaginary part hides below the precision — and an inequality with a side that is not real holds nowhere; the oracle is the definition, with SymPy 1.14 as a second opinion) and `s17b_spectrum` (11 tests: `eigenvals`, `eigenvects`, `diagonalize`, `jordan_form`, `matrix_exp` and `StateSpace::try_poles` of a matrix whose characteristic polynomial has a factor without roots in closed form report an error, never a shorter list (`poles` is empty, `is_diagonalizable` is `None`), and a rational matrix has its whole spectrum, each root of an irreducible quintic with its multiplicity; `StateSpace::is_stable` and `is_routh_stable` decide from proved signs — poles below the range of `f64`, on the imaginary axis, with symbols, a partial spectrum, complex coefficients — against mpmath's roots); the 0.60 tests of limits through a constant at a pole place and of the slow series of the nightly fuzzing are in `s15r2_limits`. Reference values cite scipy / statsmodels / mpmath / SymPy by the call that produced them. |
| `v04.rs` → `v04/v04_*.rs` | Feature tests written alongside the 0.4–0.6 APIs: `v04_polyhedron` (parametric polyhedron certificates and `PolyhedronProver`; the emitted Lean is pinned byte-for-byte to `fixtures/polyhedron_certificates.lean`, compiled against Mathlib with `linter.style.longLine` on), `v04_polytope` (exact and parametric polytopes: vertices, volume in any dimension, cuts, the bridge to the certificate search), `v04_sos` (sums of squares: the exact Gram pipeline, facial reduction on random sums of squares, refutation, Motzkin stays `Unknown`; Lean pinned to `fixtures/sos_certificates.lean`). |
| `v03.rs` → `v03/v03_*.rs` | Feature tests written alongside the 0.3 API (`poly_view`, `poly_symbolic_coeffs`, `ratsimp`, `linprog`, `normalforms`, `matrix_ergonomics`, `optimize`, `certificates`, `assumptions_poly`, `user_notes`; 0.3.5: `exact_matrix` — `QMatrix`/`ZMatrix` against textbook Gauss–Jordan, the `Matrix` fast paths, the numeric `linsolve` route, the integer-pivoting simplex against exact KKT conditions). |
| `v03_oracle.rs` → `v03_oracle/v03_oracle_*.rs` | **SymPy oracle for the 0.3 API**: `v03_oracle_poly` (`Poly`/`as_poly`: `as_dict`, degrees, `LC`, `all_coeffs`, exact `eval`, `nroots`; symbolic-coefficient `degree`/`coeff`; `ratsimp` vs `cancel`; `poly_fit_exact`; Brent roots), `v03_oracle_linprog` (exact LP vs `sympy.solvers.simplex`; objective compared exactly, the point checked for feasibility/optimality since LP vertices are not unique), `v03_oracle_normalforms` (column/row HNF, Smith form, integer kernel, lattice determinant, `gcd_many`/`lcm_many`; handles SymPy's dropped zero columns), `v03_oracle_meta` (consistency of `fixtures/v03_cross_validation.json`: consumers, `fixture_count`, unique keys, size). |
| `v02.rs` → `v02/v02_*.rs` | Feature tests written alongside the 0.2 API (matrices, sets, transforms, …). |
| `v02_oracle.rs` → `v02_oracle/v02_oracle_*.rs` | **SymPy oracle for the 0.2 API** (see below). One `#[test]` per fixture subcategory. `v02_oracle_meta` checks the fixture files (every subcategory has a consumer, ids/keys, size). |
| `v02_oracle_common/mod.rs` | Shared oracle runner (statuses, tolerances, hang guard, strict xfail). Not a test target; declared once in `v02_oracle.rs` and once (as `oracle_common`, via `#[path]`) in `v03_oracle.rs`. |
| `unit.rs` → `unit/test_*.rs` | Unit-style tests per module/feature (170 files). Includes the fourteen ratchets, which parse or scan the sources (`src/`; `test_timing_asserts` also `tests/`) and count what a rule forbids per file against an allowlist that may only shrink (`test_alloc_bounds`, `test_canonical_construction`, `test_decimal_roundtrip`, `test_divisors`, `test_float_casts`, `test_float_evidence`, `test_func_names` — new in 0.60: string literals in the printers, the parser and the sort key that spell a function of the `funcs!` table —, `test_homogeneous_tuples`, `test_id_order`, `test_layering`, `test_no_panics`, `test_special_atoms`, `test_timing_asserts`, `test_truth_values`; CONTRIBUTING.md, "Type rules that keep bug classes out"; when code moves between files, move its rows with it), `test_sympy_cross_validation` (original 263-fixture SymPy oracle for the 0.1 surface, `fixtures/sympy_cross_validation.json`) and `test_correctness_audit` (definite-integral / FTC / Gosper / series audit against `fixtures/new_capabilities.json`). |
| `legacy.rs` → `legacy/{round*_*,bugfinder*_*,math*_*,*_validation,regular_joe_tests,fixpoint_convergence_test}.rs` | Historical bug-hunting rounds and one-off validations. Many tests here are **intentionally duplicated** across rounds (see "Known overlap"). `round3_parser_fuzz` is the parser fuzzer. |
| `proptests.rs` → `proptests/{proptest_*,test_proptest_new,test_quality_props,test_units_proptest}.rs` | Property-based tests (plus `fuzz/` outside this tree). "No panic" *is* the assertion for these. Each module's `<stem>.proptest-regressions` file sits next to its source and is still picked up. |
| `perf.rs` → `perf/{simplify_perf_test,perf_analysis}.rs` | Benchmarks. `#[ignore]`d by default; run with `cargo test --test perf --release -- --ignored --nocapture`. |
| `ui_tests.rs` | `trybuild` compile-fail tests for the macros (~20 s: compiles a crate). Toolchain-pinned; CI runs it separately. |
| `common/mod.rs` | Shared helpers (`assert_math_eq`, …). Declared once per group binary; modules use it via `use super::common;`. |
| `perf_assert/mod.rs` | CPU-time performance assertions (`assert_fast!`, `CpuTimer`, `cpu_time`; budgets scaled by `SYMPLEX_TEST_TIME_SCALE`). Declared in every group root that times something; modules use `crate::perf_assert`. Wall-clock asserts are counted by `unit/test_timing_asserts.rs`. |
| `fixtures/*.json`, `fixtures/*.lean` | Oracle data (JSON, regenerated by `scripts/*.py`, < 2 MB total) and the two Mathlib-compiled Lean pins of `v04`. Modules reach them with `include_str!("../fixtures/…")`. |

### Adding a test file

Create `tests/<group>/<name>.rs` and add
`#[path = "<group>/<name>.rs"] mod <name>;` to `tests/<group>.rs` (keep the
list alphabetical — `cargo fmt` enforces it).  Use `use super::common;`
instead of `mod common;`, and `../fixtures/` in `include_str!`.  Do **not**
add new top-level `tests/*.rs` files: each one is another linked binary.

## The SymPy oracle

Reference values come from SymPy 1.14 in a virtualenv at `.venv` in the
repository root (ignored by git; CONTRIBUTING.md, "SymPy oracle fixtures",
shows how to create it).

### Regenerating fixtures

```sh
PY=.venv/bin/python                        # from the repository root
$PY scripts/generate_v02_fixtures.py       # tests/fixtures/v02_cross_validation.json   (~45 s)
$PY scripts/generate_v03_fixtures.py       # tests/fixtures/v03_cross_validation.json   (~3 s)
$PY scripts/generate_sympy_fixtures.py     # tests/fixtures/sympy_cross_validation.json (~2 s)
$PY scripts/generate_new_fixtures.py       # tests/fixtures/new_capabilities.json       (~2 s)
$PY scripts/gen_new_fixtures.py            # tests/fixtures/new_features_cross_validation.json
# each accepts --check: exit 1 if the committed file would change
```

All five generators are deterministic (fixed seeds, `sort_keys=True`, no
timestamps), so `--check` is a valid CI step.  Every SymPy computation runs
under a per-fixture `SIGALRM` timeout; a fixture whose oracle computation
times out / raises / returns an unevaluated object is **kept** with
`sympy_timeout` / `sympy_error` / `sympy_unevaluated` set, never dropped.
A SymPy value that an independent oracle refutes is corrected in the
fixture with an `oracle_correction` field (SymPy's output and the oracle);
`generate_v02_fixtures.py` re-applies its corrections (`ORACLE_CORRECTIONS`)
so that regenerating keeps them.

### Statuses (honesty policy)

| Status | Meaning | Fails the test? |
|---|---|---|
| `PASS` | symplex agrees with the oracle within tolerance | no |
| `FAIL` | symplex produced a **wrong** value/set/structure | **yes** |
| `KNOWN_BUG` | wrong, but listed in the file's `KNOWN_BUGS` table with a `// BUG:` reason | no — but if it starts passing the test fails (strict xfail) |
| `NOT_IMPLEMENTED` | symplex returned an error / unevaluated form | no (informational) |
| `UNSUPPORTED_API` | no public API for the operation | no (informational) |
| `SKIPPED_ORACLE` | SymPy itself could not produce a reference | no (printed) |

Every fixture runs on its own thread with a 4 s budget; a hang is a `FAIL`
("HANG") so one pathological case cannot stall CI.  Each `#[test]` covers one
`(category, subcategory)` and must finish in well under 10 s.

Comparison rules of thumb used by the consumers:

* constants: complex `f64` with relative tolerance `1e-6` (special functions
  `1e-12`, plus a 18-digit `eval_decimal` check against `N(…, 30)`);
* expressions: evaluated at 3–4 exact rational sample points from the fixture;
* antiderivatives / sums with symbolic bounds: compared through
  differences or brute force at `n = 1..6`, never by string;
* sets / inequalities: membership at ~20 sample points;
* boolean logic: full 16-row truth tables;
* multi-valued results (`sqrt_mod`, `primitive_root`, `nroots`, eigenvalues):
  verified by congruence / membership / sorted multiset, not by SymPy's choice;
* exact results of the 0.3 API (rational coefficients, LP optima, integer
  normal forms) are compared as `Ratio<BigInt>` / `BigInt` **equality**;
  symbolic coefficients at 3 parameter points (`1e-9`).

Oracle caveats found while building the v03 fixtures (worked around in
`scripts/generate_v03_fixtures.py`): SymPy 1.14's `simplex.linprog`
mishandles non-default `bounds` (a negative lower bound or a free variable
is still forced `≥ 0`), so `lpmin`/`lpmax` with explicit relational
constraints are the LP reference; `Poly.nroots` does not converge on
repeated roots, so `all_roots(radicals=False)` is used there.

### Recording a library bug

1. Add `(category, subcategory, key, reason)` to the consumer's `KNOWN_BUGS`
   with a `// BUG:` comment stating the wrong and the correct answer.
2. Add an `#[ignore = "BUG: …"]` reproducer test next to it.
3. When the bug is fixed the oracle test fails with `UNEXPECTED_PASS` —
   remove the entry and un-ignore the reproducer.

Do **not** fix `src/` from a test-campaign branch; report the bug.

## Conventions

* **One concept per test**, named after the mathematical fact it checks.
* **Assert values, not existence.**  `assert!(r.is_ok())` and
  `eprintln!`-only bodies are not tests; compute the expected value
  independently (SymPy in a comment is fine) and assert it.
* **No silent skips.**  `if let Ok(v) = … { assert!(…) }` passes vacuously
  when the operation fails.  Either the operation is expected to succeed
  (`expect`) or the test documents that it is optional and reports the skip.
  (`if let Some(bad) = check() { panic!() }` — failure iff `Some` — is fine.)
* **Display strings** are a legitimate assertion only when the display *is*
  the feature (pretty-printer, LaTeX, canonical form tests).  For
  mathematical results prefer structural equality or a numeric check.
* **Tolerances** should reflect the math: exact results `1e-12`, `f64`
  special functions `1e-10`, truncated series what the remainder bound says.
  `1e-2` needs a comment explaining why.
* **Timeouts.**  Wrap every cargo/python invocation in `timeout` when
  iterating; aim for about a second per `#[test]` in a debug build and keep
  each under 10 s; put exhaustive sweeps and slower cases behind
  `#[ignore]` (with the reason and the time) with a sampled variant that
  runs by default.  nextest reports a test as slow after 30 s and kills it
  after two minutes (`.config/nextest.toml`; 60 s and five minutes in CI).
* **Benchmarks are not tests** — `#[ignore]` them with a reason and run them
  in `--release` when needed.

## Known overlap

The `round*`, `bugfinder*`, `math*`, `test_cycle*` and `test_stage*` files
were written in independent bug-hunting rounds and contain ~130 groups of
byte-identical test bodies (~930 tests, e.g. `test_stage1` ↔ `test_stage3`,
`test_math_rules` ↔ `test_rule_application`).  They are kept deliberately
(cheap, and each round's file is self-contained); do not add new duplicates.
