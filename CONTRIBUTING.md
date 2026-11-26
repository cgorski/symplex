# Contributing to Symplex

Welcome. This document covers everything you need to understand the codebase,
make changes safely, and submit contributions.

**The goal is to be better than SymPy, not to match it.**  SymPy (with
mpmath, SciPy and statsmodels) is our *oracle* and a development aid: it
supplies reference values, a coverage map and a well-tested default for
questions of convention (branch cuts, `expand_log`'s rules, `N`'s digits).
It is not the ceiling.  Where an oracle is wrong (SymPy 1.14's resultant
sign when `deg a < deg b` and `deg a·deg b` is odd, SciPy's `gammaincinv`
in the far tail), weaker than the mathematics allows, slow, or silently
wrong, symplex should do better — provably: an exact argument in the doc
comment, an independent check (mpmath at high precision, a derivative, an
exact identity) in the test, and the oracle's differing answer quoted
beside it.  We owe these projects a great deal; say so where we follow
them (see *Provenance* below), and keep comparisons factual.

### Provenance: mathematics is free, code has a licence

Mathematics — theorems, formulas, algorithms, series coefficients,
conventions such as a branch cut — is not copyrightable, and the values an
oracle computes are facts.  An *implementation* is: its structure,
identifiers and comments are someone's expression.  We want nothing in this
repository that anyone could reasonably flag, so:

* **Oracles.**  Running SymPy, mpmath, SciPy or statsmodels to obtain a
  reference value is always fine; cite the call next to the value.  None of
  their code is copied into the crate.
* **Permissive sources** (BSD, MIT, Apache-2.0, ISC, Zlib, BSL-1.0, public
  domain — SymPy, mpmath, SciPy, statsmodels, Boost.Math, Rubi): reading
  the implementation and following its structure is fine.  Say so in the
  doc comment ("follows SymPy's `gruntz.py`"), cite the publication the
  algorithm comes from, and add a row to
  [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md), with the licence
  text if it is not already there.  Vendored data (e.g. the Rubi test
  suite) keeps its licence file next to it.
* **Restrictive or unclear sources** — GPL / LGPL / AGPL (R's `nmath`,
  GSL, Maxima, PARI/GP, GiNaC, FLINT), the ACM Software License of the
  CALGO / TOMS Fortran codes, Numerical Recipes, "non-commercial" terms,
  code with no licence at all: **do not read the implementation** while
  writing the corresponding symplex code, and do not name it as the
  source.  Work from the publication (the paper, the book, DLMF).  When
  the publication is not enough, use a clean room: one person
  writes a purely mathematical description (formulas, case analysis,
  error bounds, test values — no code and no identifiers beyond the
  paper's notation), and someone who has not seen the restricted code
  implements from that description alone.  Keep the description with
  the change.
* **Be careful with "R's `f`", "as in GSL", "from the TOMS code" in
  comments**: they read as a claim of derivation.  Name the mathematics
  ("Stirling's series", "DiDonato & Morris 1992, §3") instead, and put
  an implementation's name only where it is the *oracle* of a test value.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [The Context System](#the-context-system)
- [The Safety Model](#the-safety-model)
- [The API Model](#the-api-model)
- [Expression Nodes](#expression-nodes)
- [How to Add a New Function](#how-to-add-a-new-function)
- [How to Add a New Node Type](#how-to-add-a-new-node-type-unevaluated-form)
- [Testing](#testing)
- [Continuous Integration](#continuous-integration)
- [Pull Requests](#pull-requests)
- [Code Style](#code-style)
- [Common Pitfalls](#common-pitfalls)

---

## Quick Start

```bash
# Clone and build
git clone https://github.com/cgorski/symplex
cd symplex
cargo build

# Run the test suite (about 15,130 tests; a few minutes in debug)
cargo test
# ... or, recommended: one process per test with per-test timeouts
# (`.config/nextest.toml`; `cargo install cargo-nextest`)
cargo nextest run

# Run the 0.3 / 0.2 feature suites only (one test binary each)
cargo test --test v03
cargo test --test v02

# Run the examples and the book (CI does both)
for f in examples/*.rs; do n=$(basename "$f" .rs); [ "$n" = repl ] || cargo run -q --example "$n"; done
mdbook build book

# Run a single test module (former file `tests/test_known_answers.rs`)
cargo test --test unit test_known_answers::

# Run doc-tests only
cargo test --doc

# Run benchmarks
cargo bench

# Run an example
cargo run --example quickstart

# With tracing output (see simplification steps, integration attempts, etc.)
RUST_LOG=symplex=debug cargo run --example quickstart

# Which steps of a slow call took the time (see "Finding where a slow or
# memory-hungry call spends its time" below)
RUST_LOG=symplex::stage=debug cargo run --release --example …

# Dependency policy check (CI runs this too; `cargo install cargo-deny`)
cargo deny check
```

### Dependency policy

symplex is **pure Rust**: no crate in the graph — including dev- and
build-dependencies — may compile C/C++ or link a system library (no `*-sys`
crates, `cc`, `cmake`, `pkg-config`, GMP/MPFR, BLAS/LAPACK, HiGHS, …), and
every licence must be on the permissive allow-list.  `deny.toml` encodes this
and `cargo deny check` enforces it (bans, licences, advisories, registry
sources).  Adding a dependency means: check it is pure Rust on *every* target
Rust supports (including `wasm32-unknown-unknown`), prefer
`default-features = false`, and extend `deny.toml` deliberately if a new
licence appears.

`Cargo.lock` of the root crate is committed since 0.59, so that CI, the
gate and the golden outputs build the same dependency versions and a
change in a result or a timing is a change in the code.  A release only
bumps the crate's own version in it (`cargo update -w`); the versions of
the dependencies change on purpose, in a commit of their own that passes
the gate, the golden diff and the byte-identity check (the order of long
sums follows `rustc-hash`'s hash: 2.1.0 and 2.1.1 print differently), and
a pull request that changes a dependency commits the lock change with it.
The version floors in the manifests are a promise to users and are tested:
`.github/workflows/dependencies.yml` runs weekly, the suite on the latest
versions (`cargo update`) and `cargo check` on the oldest ones the
manifests allow (`cargo update -Z direct-minimal-versions`, on Rust 1.93
and on stable, all four crates).  When it goes red: latest — fix the code
or pin the dependency with the reason in `Cargo.toml`; minimal — raise the
floor.  The side crates
(`symplex-macros`, `symplex-build`, `symplex-wasm`, `fuzz`, `rubi-harness`,
`probes/*`) keep their locks untracked and resolve on their own
(`.gitignore`).  Users of the library do not get our lock, so when a
dependency version matters, raise its floor in `Cargo.toml`: up to 0.51 a
local lock pinned num-bigint 0.4.6, which allocates for every one-digit
number, and the matrix functions looked 40 % slower than in 0.48 (built
with 0.4.8); `Cargo.toml` now requires 0.4.8.  Compare performance between
versions only with the same lock.

### Feature flags: one crate, no knobs

`symplex` deliberately has **no Cargo features** (only `symplex-wasm` carries
one, for its panic hook).  Do not introduce feature gates for granularity's
sake — `serde`, `certificates`, `sos`, `codegen`, … stay unconditional.
Feature matrices multiply the CI surface, break `--all-features`-free
docs.rs builds and doctests in subtle ways, and, above all, make later
features harder to add (every new module has to decide which gates it lives
behind and which combinations it must compile in).  Compile time and binary
size are not a concern at this stage; capability is.  If a flag ever *does*
appear and starts to constrain a design, removing it is the right call, not
working around it.  Revisit only if a concrete downstream (embedded /
`no_std` consumer of generated code, a size-limited wasm build) asks with
numbers.

---

## Architecture

The source code (about 415,000 lines of Rust at 0.60) is organized into 10
directories under `src/`. Each directory is a layer in the dependency
hierarchy — modules may depend on layers below them but should not reach
upward.

```
src/
├── base/         Expression nodes (node.rs, 96 variants), arena (hash-consing), symbol.rs (name
│                 interning), tree traversal (walk.rs), canonicalization (canon.rs + canon/, see
│                 "The split files"), assumptions, sort keys, compaction,
│                 numeric.rs (exact f64 ↔ rational, the exact rational `Q`),
│                 bigcomplex.rs (BigFloat complex pairs), bernoulli.rs, complex.rs (Complex64
│                 helpers), errors.rs, config.rs (EvalConfig, EXPRESSION_BUDGET), and the shared
│                 value types of 0.15–0.21: interval.rs (Interval/Bounds), extended.rs (Extended),
│                 rng.rs (SplitMix64, XorShift64Star), budget.rs (Budget), stage.rs (stage! tracing), combinatorics.rs,
│                 graph.rs (Tarjan SCC), libfn.rs (the LibFn registry of 50 special functions),
│                 dense_f64.rs (flat f64 Cholesky/SPD/Jacobi kernels), and the 0.49 type
│                 architecture: func.rs (Func/FuncSpec: one identity and fact table for all
│                 101 functions; generates LibFn), truth.rs (Truth, SignSet, Evidence,
│                 Verdict), cast.rs (checked float→integer), bigfloat.rs (the astro-float
│                 boundary), value_class.rs (ValueClass: finite, ±∞, a directed infinity, zoo,
│                 nan); node.rs's shape methods come from symplex-macros' NodeShape derive
├── poly/         traits.rs (Ring/Field/EuclideanDomain hierarchy, the `poly_gcd` hook),
│                 dense.rs/generic.rs (GenPoly<C>), zpoly.rs (ℤ-scaled forms, gcd_via_z),
│                 modpoly.rs (𝔽ₚ[x]: Fp64, PolyIn<R>), interp.rs (Newton divided differences),
│                 factor_zassenhaus.rs (Berlekamp–Zassenhaus over ℤ + Kronecker multivariate),
│                 Gröbner bases, polysys.rs, Sturm sequences, root finding (roots.rs), algebraic
│                 number fields ℚ(α) (algebraic.rs), ratfn.rs, multipoly.rs (sparse multivariate;
│                 heuristic gcd/lcm, content, denominators), polybridge.rs (Ex ↔ polynomial,
│                 incl. symbolic-coefficient term collection)
├── transforms/   diff, expand, subs, inequalities, pattern.rs (AC matcher), realness.rs (realness
│                 of constants); integrate.rs + integrate/ (12 files, one family of rules each)
│                 + heurisch, trig_integ, apart; eval.rs + eval/ (9 files: exact values and zero
│                 tests); evalf.rs + evalf/ (49 files: the numeric kernels); solve.rs + solve/
│                 (9 files: context.rs with SolveCx and Domain, outcome.rs with Solutions, and
│                 the solving strategies) — for these four see "The split files";
│                 sets.rs (set-algebra normal form), logic.rs (boolean simplifier, DPLL,
│                 piecewise), rsolve.rs (recurrences)
├── simplify/     simplify_engine (multi-strategy fixpoint + tracing), rewrite.rs, trig/hyp
│                 (fu.rs, trigsimp, trig_expand, trig_combine), powsimp/radsimp (powdenest,
│                 sqrtdenest), log_expand/log_combine, combsimp, nsimplify, refine, factor_terms,
│                 ratsimp.rs (rational-function normal form over opaque indeterminates)
├── calculus/     definite.rs + definite/ (definite/improper integration, GK15 quadrature) and
│                 ode.rs + ode/ (see "The split files"), summation.rs (+ the sum_eval shim eval()
│                 calls), gosper.rs, convergence.rs, series.rs, formal_series.rs, finite_diff.rs,
│                 limit.rs + gruntz.rs, calculus_util.rs (domain analysis, breakpoints, the zero
│                 tests of constants: hidden_zero, ZeroDecision, NonzeroExpr), residue.rs,
│                 laplace.rs, fourier.rs (series), fourier_transform.rs, mellin.rs,
│                 z_transform.rs, transform_util.rs (0.60: what the four integral transforms
│                 share; the bound-variable rule), risch/ (tower, hermite, rde,
│                 rothstein_trager, log_to_real)
├── output/       display, pretty, latex, mathml, parse, tree (JSON), common.rs (shared print
│                 ordering), cse, lambdify (stack-VM compile), lean.rs (Mathlib rendering) +
│                 lean_proof.rs (structured tactic proofs), codegen.rs (Rust) +
│                 codegen/{codegen_c.rs (C99), codegen_py.rs (Python), numeric_rt.rs (shared f64
│                 special-function runtime), rt_embed.rs (embedding `mod symplex_rt`)}
├── plotting/     Adaptive sampling, textplot, SVG, TikZ, data export, RK4
├── domains/      matrix.rs + matrix/ (zero_test.rs, eigen.rs; see "The split files") +
│                 matrix_decomp.rs (QR, Cholesky, LDL, Gram–Schmidt, structure
│                 tests, norms, hessian, wronskian, 0.3 selection/conversion helpers),
│                 decompositions.rs (named results of factorisations and normal forms),
│                 exact_kernel.rs (0.20: the Cell trait over i64/i128/W256/BigInt, resumable
│                 fraction-free Elimination, Bareiss, Berkowitz, FractionFreeLu — shared by
│                 exact_matrix, polytope and the linprog tableau), exact_matrix.rs (QMatrix/ZMatrix
│                 over Ratio<BigInt>/BigInt, HNF/SNF cores; Matrix routes all-rational input
│                 here and caches the tier), linalg.rs (symbolic rref/linsolve
│                 with the numeric fast path), linprog.rs (exact two-phase simplex on an
│                 integer-pivoting tableau, duals, Farkas certificates), normalforms.rs
│                 (Matrix wrappers over ZMatrix: Hermite/Smith normal forms, integer
│                 nullspace, unimodularity, lattice index), certificates.rs (Handelman box and
│                 univariate half-line certificates) + certificates/polyhedron.rs (0.4:
│                 parametric polyhedron certificates with a λ(j) goal multiplier, staged LP,
│                 Lean export in the mul_nonneg / linarith-only shape) + certificates/sos.rs
│                 (0.6: sums of squares — dense f64 primal–dual SDP, exact rounding/projection,
│                 rational LDLᵀ, LLL-based facial reduction; Lean via ring + positivity),
│                 certificates/outcome.rs (the shared Outcome<C, U>), polytope.rs (0.4: exact
│                 polyhedra from half-spaces: vertices via QMatrix, LP-based emptiness and
│                 bounds, volume in any dimension), optimize.rs (Brent/bisection/Newton,
│                 Nelder–Mead, golden section, differential evolution, least-squares fits),
│                 control, dynamics, robotics, quaternion, vector (coordinate systems), ntheory
│                 (rho/ECM, BPSW, sqrt_mod, dlog, continued fractions, gcd_many/lcm_many),
│                 diophantine, combinatorics, discrete.rs (convolutions, NTT, Walsh–Hadamard,
│                 Möbius), separatevars,
│                 stats/ (0.11–: random variables — family.rs (the Family trait and
│                 Distribution), continuous.rs, discrete.rs, support.rs, rv.rs, events.rs, joint.rs,
│                 wrappers.rs (Truncated/Affine/Transformed/Mixture), order.rs, sample.rs — and
│                 the data layer with one placement rule per module (stats/mod.rs): data,
│                 estimation, hypothesis, anova, agreement, reliability, aggregation, regression,
│                 survival, cox, markov, information, sequential, multivariate, plus common.rs
│                 (shared level checks and exact tails) and numdist.rs (f64 reference
│                 distributions: TOMS 708 incomplete beta, Temme incomplete gamma))
├── units/        Compile-time dimensional analysis, quantity types, conversions, constants
└── api/          context.rs, expr.rs (Expr<S>), expr_funcs.rs (most methods), expr_ops.rs
                  (operators, Scalar/ToEx, Context ingestion), eq.rs (Equation), macros.rs,
                  expr_view.rs, expr_algebraic_ext.rs (0.9: minimal polynomials, multivariate
                  gcd, Gröbner, factoring mod p), expr_calculus_util_ext.rs (0.9: singularities,
                  extrema, monotonicity, periodicity), and the 0.2 extension files:
                  expr_complex.rs (re/im/conjugate/arg + Si/Ci/Ei/li/ζ/polygamma),
                  expr_integrate_ext.rs (definite/numeric integration, residue_at_infinity),
                  expr_series_ext.rs (summation/products/convergence/series at ∞),
                  expr_solve_ext.rs (linsolve, LinearSolution, solve_general, Newton, IVPs),
                  expr_sets_ext.rs (SetEx/BoolEx algebra, reduce_inequalities, piecewise),
                  expr_rules_ext.rs (Rule/RuleSet/rewrite/simplify_traced + gap-fill simplifiers),
                  expr_transforms_ext.rs (directional limits, Fourier/Mellin, FourierSeries),
                  expr_poly_ext.rs (resultant/discriminant/division/roots on Ex, and the 0.3
                  interval sign tests poly_is_nonnegative_on / poly_is_positive_on),
                  and the 0.3 file poly_ex.rs (the public `Poly` view and `Ex::as_poly`)
```

Companion crates: `symplex-macros/` (proc macros), `symplex-build/` (build-time
codegen for `no_std`), `symplex-wasm/` (wasm-bindgen bindings + `Session`),
`rubi-harness/` (the Rubi integration test suite, 72,254 integrands judged by differentiation, with a per-file ratchet: `cd rubi-harness && cargo run --release -- --check`; see its README), `fuzz/` (cargo-fuzz targets that check properties — exact identities, value
preservation, `F′ = f` — run nightly by `.github/workflows/fuzz.yml`; see
`fuzz/README.md` for the replay loop), `probes/` (developer diagnostics, not compiled by
default), `book/` (mdBook), `benches/` (criterion).

**Dependency flow.**  Two types are hubs that every layer may name: the
`ExprId`-level core (`base::{node, arena, canon, …}`) and the handle types
`api::{expr, context}` (`Ex`, `BoolEx`, `SetEx`, `Context`).  Between them
the algorithm layers are ordered, and `output` sits *below* `domains`
(certificates render Lean, matrices generate code, sampling compiles):

```
  api::{expr, context}       handle types — named by every layer except plotting
        ▲
  base::{node, arena, …}     ExprId-level core; `Arena` is also a façade whose
        │                    `expand()/integrate()/…` delegate to the layers below
        ▼
  poly → transforms → simplify → calculus       ExprId in, ExprId out
        ▼
  output                     printers, parser, codegen, lambdify (imports no domain)
        ▼                                                     plotting (f64 leaf)
  domains                    Ex-level algorithms; use output for Lean/codegen/compile
        ▼
  api::*_ext, api::poly_ex   the method surface; re-exports; `poly_ex` is used by
        ▼                    certificates/polytope/stats (an upward edge, allowlisted)
  units                      dimensional analysis over the api
```

A file may depend on its own layer and anything below it in this order
(`base → poly → transforms → simplify → calculus → output → plotting →
domains → api → units`).  The upward edges that exist — the `Arena`
façade, `eval` folding `Sum`/`Integral` through the summation and Risch
engines, `polysys` falling back to the general solver, the certificate
modules working on the public `Poly` view, … — are enumerated with their
reasons in `tests/unit/test_layering.rs`, a ratchet that fails when a file
gains a new upward target module (or when the allowlist is no longer
tight).  Prefer moving code down a layer or routing through the façade to
extending the allowlist.

### The split files

Since 0.60 the eight files that had grown largest (8,500 to 15,400 lines
each at 0.59) are a parent `x.rs` and child modules `x/<child>.rs`.  The
parent keeps the entry points, the dispatcher and the inline `mod tests`; a
child holds one family of rules, one class of equations or one family of
kernels.  A child begins with `use super::*;` and its parent imports it
with a glob, so a path such as `canon::at_infinity` or
`definite::side_value` — in this document, in doc comments and in the code
— resolves as before the split although the item lives in a child file:
`grep -rn "fn side_value" src/calculus/definite*` finds it.

- **`base/canon.rs`** keeps `canon_add`, `canon_mul`, `canon_pow`,
  `canon_neg`, the radicals, the arithmetic of infinities and the
  predicates the constructors need (`may_vanish_on_a_region`,
  `at_complex_infinity`, `known_sign`).  `canon/`: `mul_vanishing` (sums
  and factors that vanish identically; `with_pinned_values`),
  `special_values` (`canon_function`: functions at exact special
  arguments), `assumed` (values under the assumptions on symbols;
  `lib_function_value`), `infinite_args` (library functions at `±∞` and
  `±i·∞`: `at_infinity`), `sets` (the set constructors), `verify`
  (`verify_canonical`).
- **`transforms/integrate.rs`** keeps the entry point `integrate`, the
  budgets, the stages and the acceptance of a candidate, the memo, the
  choice of parts and the dispatcher (`integrate_node`,
  `integrate_node_uncached`).  `integrate/`: `standard_forms` (`(x² ±
  a²)ⁿ`, completed squares, linear over quadratic), `trig_rational`
  (rational functions of `sin` and `cos`: Weierstrass, Bioche),
  `rational_constants` (rational functions with `i` or constants in the
  coefficients), `u_substitution`, `substitution` (logarithms,
  exponentials, radicals; the spiral antiderivatives of `R(e^{λx})`),
  `mobius` (radicals of a Möbius function), `algebraic` (`P(x)·Q(x)^{k/2}`),
  `products` (by parts, product-to-sum), `piecewise` (`|g|`, `sign`, `H`,
  `Piecewise`), `special` (the special-function table), `biquadratic`,
  `verify` (the check of a candidate antiderivative; nested powers over
  the reals).
- **`transforms/eval.rs`** keeps the dispatcher `eval`.  `eval/`:
  `trig_radicals` (trigonometric values in radicals; `eval_exp`, `eval_ln`,
  `eval_abs`, `eval_sign`, the inverse and the hyperbolic functions),
  `integer_sequences` (the factorial family, Fibonacci, Bernoulli,
  harmonic, Stirling and Bell numbers — and `eval_sin`, `eval_cos`,
  `eval_tan`), `lib_values` (`eval_lib_fn`: the library functions),
  `special02` (`Si`, `Ci`, `Ei`, `li`, `ζ`, `ψ`, `ψ⁽ⁿ⁾`, `δᵢⱼ`), `orthopoly`
  (the classical orthogonal polynomials), `digit_guard` (size guards of
  exact values, `eval_factorial`, `eval_gamma`, relations between constants
  and `constant_zero_decision`), `quadratic_tower` (the zero test in towers
  of square roots; `eval_log_gamma`, `eval_erf`, `eval_lambertw`,
  `eval_beta`, `eval_binomial`), and the two older modules `atan_sum` and
  `cyclotomic` (exact zero tests).
- **`transforms/evalf.rs`** keeps the entry points, the adaptive precision
  loop, the `f64` results, `eval_node` and `const_value`.  `evalf/` has 49
  files: to the 31 it had (`wp`, `ball`, `accuracy`, `quad`, `improper`,
  `windowsum`, …) the split added `integral` (definite integrals),
  `subtree` (`Sum`, `Product`, `Piecewise`, `RootOf`, `RootSum`), `celem`
  (complex arithmetic), `f64_special` (`f64` helpers), `gamma` (Stirling's
  series), `constants` (Euler's `γ`), `erf`, `bessel`, `expint` (the
  trigonometric, exponential and logarithmic integrals), `zeta_polygamma`,
  `orthopoly`, `special09`, `incgamma`, `betainc`, `polylog_eta`, `airy`,
  `carlson` (elliptic integrals) and `format` (decimal output).
- **`transforms/solve.rs`** keeps the entry points (`solve_classified`,
  `solve_general`, `known_real_roots`), the checks of candidates
  (`residual_at`, `part_at_a_pole`), `solve_raw` and the polynomials with
  symbolic coefficients.  `solve/`: `context` (`SolveCx`, `Domain`),
  `outcome` (`Solutions`), `closed_forms` (degrees 1 to 4, rational roots),
  `peeling` (inversion, principal branches, families and their real
  members), `change_of_var`, `rational_exp` (rational equations;
  exponential, hyperbolic and half-angle substitutions; sums of
  logarithms), `lambert`, `declared` (declared domains of the unknown),
  `algebraic_roots` (polynomials with algebraic-number coefficients).
- **`calculus/definite.rs`** keeps `integrate_definite`, the closed form,
  the pieces and breakpoints, the poles of rational factors and the residue
  routes.  `definite/`: `branch_cuts` (cut crossings of the
  antiderivative), `endpoints` (endpoint values and one-sided limits),
  `special_integrands` (`δ`, `H`, `|·|`, `sign`, `Piecewise`, `floor`),
  `known_values` (the table of known definite integrals), `gauss_kronrod`
  (the GK15 quadrature: `QuadOpts`, `QuadResult`, `quadrature`).
- **`calculus/ode.rs`** keeps `dsolve`, the normal forms and the elementary
  first- and second-order routes.  `ode/`: `separable`,
  `linear_exact_homog` (first-order linear, exact, homogeneous
  coefficients), `reducible` (no explicit `x`), `bernoulli_euler`
  (Bernoulli, Euler–Cauchy), `variation` (variation of parameters),
  `const_coeff` (linear with constant coefficients, any order),
  `clairaut_riccati`, `classify` (`classify_ode`), `verify`
  (`checkodesol`), `systems` (`solve_ode_system`).
- **`domains/matrix.rs`** keeps the `Matrix` type and its methods.
  `matrix/`: `zero_test` (the three-valued scalar tests, shared with
  `matrix_decomp` and `quaternion`), `eigen` (root formulas for
  characteristic polynomials, multiplicities, Jordan helpers).

**Splitting another file.**  A split moves text and edits nothing.  The
child starts with `use super::*;`, the parent declares `mod <child>;` and
imports it with a glob (`use`, `pub(crate) use` or `pub use`: the widest
visibility a moved item has), so every path that resolved before still
resolves; the inline `mod tests` stays in the parent, so no test changes
its name.  The only edit allowed to a moved line is `pub(super)` on an
item, field or method that the parent or a sibling names (doc links count:
CI builds the private docs with `-D warnings`).  A child must not have the
name of a function or type of its parent, or a doc link to it is ambiguous
(rustdoc: "both a function and a module"): hence
`definite/gauss_kronrod.rs` beside `fn quadrature` and
`canon/infinite_args.rs` beside `fn at_infinity`.  Prove the move before
anything else is run: every top-level item of the old file is found once,
textually equal up to `pub(super)`, in the parent or a child (compare
before rustfmt, and item by item after it); `cargo nextest list` prints
the same test names; and the rows of the ratchet tables move with the
code (see "Type rules that keep bug classes out").

### Key Types

| Type | Location | Purpose |
|------|----------|---------|
| `ExprNode` | `src/base/node.rs` | The expression tree — 96 variants (Add, Mul, Sin, Integral, Re/Im/Conjugate/Arg, Zeta, Polygamma, RootOf, RootSum, Interval, etc.) |
| `Arena` | `src/base/arena.rs` | Hash-consed expression storage. All nodes live here. |
| `ExprId` | `src/base/node.rs` | A handle into the arena (equality and hashing only: no order — see "Identifiers have no order"). This is how expressions are referenced internally. |
| `Func` / `FuncSpec` | `src/base/func.rs` | One identity for every function (dedicated nodes, `LibFn` library functions, internal helpers) and its table of facts: names, arity, sort bytes, the LaTeX, two-dimensional and MathML forms, conjugation, real domain, parity, poles, constant folding, numeric back ends. See "How to Add a New Function". |
| `Truth`, `SignSet`, `Evidence`, `Verdict<T>` | `src/base/truth.rs` | Three-valued decisions, the set of signs a finite value can have, and how a decision was established. See "Decisions: `Truth`, signs and evidence". |
| `SolveCx`, `Domain`, `Solutions` | `src/transforms/solve/{context,outcome}.rs` | The domain of the unknown, passed explicitly through the solvers, and their outcome (`Complete` / `Partial` / `Refused`). See "Solving". |
| `Work`, `WorkScope`, `BudgetKind` | `src/base/budget.rs` | Deterministic work budgets carried by the arena. See "Budgets count work". |
| `Context` | `src/api/context.rs` | User-facing entry point. Owns an arena + assumption cache. |
| `Expr<S>` / `Ex` | `src/api/expr.rs` | User-facing expression handle. Carries a context reference + ExprId. |
| `Poly` (internal) | `src/poly/dense.rs` | Dense univariate polynomial over `Ratio<BigInt>`. Type alias for `GenPoly<Ratio<BigInt>>`; `pub(crate)`. |
| `Poly` (public) | `src/api/poly_ex.rs` | User-facing sparse polynomial view of an `Ex` over explicit generators with symbolic coefficients (`prelude::Poly`). |
| `GenPoly<C>` | `src/poly/generic.rs` | Generic univariate polynomial over any `Ring`/`Field` coefficient type. |
| `RationalFn` | `src/poly/ratfn.rs` | Rational function `p(x)/q(x)` in ℚ(x). Implements `Field`, enabling `GenPoly<RationalFn>`. |
| `AlgExpr` (internal) | `src/poly/algebraic.rs` | Arena-free tree of an algebraic number (rationals, roots, field operations) used by `minimal_polynomial`: resultants eliminate the radicals, the candidate factor is verified at high precision. |
| `MultiPoly` | `src/poly/multipoly.rs` | Sparse multivariate polynomial. |
| `Matrix` | `src/domains/matrix.rs` | Symbolic matrix (Vec of Vec of Ex); its scalar zero tests and eigenvalue helpers in `matrix/`, decompositions in `matrix_decomp.rs`. |
| `Rule` / `RuleSet` | `src/api/expr_rules_ext.rs` | Public rewrite rules over `Pattern` (`src/transforms/pattern.rs`). |
| `CompiledFn` | `src/output/lambdify.rs` | Stack-VM compiled numeric closure (`Clone + Send + Sync`). |
| `LinearSolution` | `src/api/expr_solve_ext.rs` | `Unique` / `Parametric` / `Inconsistent` result of `linsolve`. |
| `FormalPowerSeries` | `src/calculus/formal_series.rs` | Lazy exact power series over `Ex`. |
| `LpProblem` / `LpSolution` | `src/domains/linprog.rs` | Exact LP builder and result (`status`, `x`, `objective`, `duals`, `farkas`). |
| `RootOpts` / `MinimizeOpts` / `DeOpts` / `MinimizeResult` | `src/domains/optimize.rs` | Tolerances, budgets, seeds and results of the `f64` optimisation routines. |

### How Expressions Work

Expressions are stored in an **arena** (a big `Vec<ExprNode>`) with **hash-consing**
(deduplication). When you create `x + y`, the library:

1. Looks up whether `Add([x_id, y_id])` already exists in the arena
2. If yes, returns the existing `ExprId` (O(1) equality!)
3. If no, appends a new node, returns the new `ExprId`

The user never sees `ExprId` directly. They hold `Ex`, which is:

```rust
pub struct Expr<S: Sort> {
    pub(crate) ctx_id: CtxId,           // which context this belongs to
    pub(crate) inner: Arc<RwLock<ContextInner>>,  // pointer to the context
    id: ExprId,                          // PRIVATE — index into the arena
    pub(crate) _sort: PhantomData<S>,    // Numeric, Boolean, or SetValued
}
```

The `id` field is **private to the `api::expr` module**. This is the foundation
of the safety model.

---

## The Context System

Every expression belongs to a `Context`. A context owns:
- An **arena** (expression storage)
- An **assumption cache** (is x positive? is n an integer?)
- Cached **simplification rules**

Users create contexts explicitly:

```rust
let ctx = Context::new();
let x = ctx.symbol("x");
let expr = x.powi(2) + 1;  // lives in ctx's arena
```

Multiple contexts can coexist (e.g., for multi-tenant CAS servers where each
user session has its own context with its own assumptions). Expressions from
different contexts **cannot be mixed** — this is enforced at compile time and
runtime.

### The Units Context

The `units` module has its own shared `OnceLock<Context>` for dimensional
analysis expressions. This is the only hidden static context in the library.
It's documented, justified, and guarded by `checked_id`. See
`src/units/si.rs` for details.

---

## The Safety Model

### Cross-Context Protection

The `id` field on `Expr<S>` is **private to `src/api/expr.rs`**. Code in
other modules literally cannot access it — the compiler rejects `expr.id`.

Two accessor methods exist:

| Method | Purpose | Who can call it |
|--------|---------|----------------|
| `raw_id(&self) -> ExprId` | Get your OWN expression's ID | Any `pub(crate)` code |
| `checked_id<T>(&self, other: &Expr<T>) -> ExprId` | Get ANOTHER expression's ID after validating same context | Any `pub(crate)` code |

`checked_id` panics with a clear message if the two expressions belong to
different contexts — by design: it is a logic error, like indexing a Vec out
of bounds.  (The complete list of remaining panic paths is the two
allowlists in `tests/unit/test_no_panics.rs`; see the next section.)

A panic, an abort, a stack overflow, unbounded memory growth or a call that
does not return on *any* input to the entry points meant for untrusted text
(`Context::parse` and the other parsers, `from_json`, `from_tree`,
`decimal_str`, the wasm `Session`) is a security bug:
[`SECURITY.md`](SECURITY.md) says how to report one and what the library
promises there.

**When adding a new public method** that takes multiple `Ex` arguments, you
MUST use `self.checked_id(other)` for every foreign expression's ID. The
compiler enforces this — you can't access `.id` directly.  A `Context`
method that takes an expression uses `self.own_id(&e)` (generic over the
expression's sort), never `e.raw_id()`: up to 0.48 seven `Context` methods
used a foreign expression's id unchecked, and `a.query(&b.int(-7),
POSITIVE)` answered `Some(true)` about `a`'s `7`.

### No Panics Rule

Library code never calls `.unwrap()`, `.expect()`, `unreachable!()`,
`todo!()` or `panic!()`.  The exceptions are two documented logic errors —
the cross-context guard, and `std::iter::Sum`/`Product` for `Ex` on an
*empty* iterator (there is no context to build `0`/`1` in — users are steered
to `Context::sum`/`product` or `Option<Ex>`) — plus the arena's and symbol
table's `u32` index conversions (an `expect` beyond 4 billion nodes, numbers
or symbols: a panic, not an error value), the arithmetic operators on
matrices of mismatched shapes (`&a + &b`; `add`/`sub`/`matmul` return a
`Result`), `Polytope::with_halfspace` with the wrong number of coefficients
(`try_with_halfspace` returns a `Result`) and the compile-time
`const_assert_dim!` macro.  Every other operation that can fail returns
`Result`, `Option`, or an unevaluated form.  Tests are free to unwrap;
`debug_assert!` is acceptable for internal invariants.

A second, shrinking category is **runtime `assert!`/`assert_eq!`/`assert_ne!`
on caller-supplied shapes and preconditions** — 82 sites in 19 files at 0.22,
27 in 7 at 0.29, 23 in 7 at 0.59, e.g. `Context::symbol("")`, a zero
denominator to `RationalFn`, exponent overflow in `MultiPoly::mul`/`pow` (the
bodies of `*`, whose `try_` twins return `None`), an out-of-range index to
`Matrix::get` (the body of `m[(i, j)]`; `try_get` is its twin).  The
`Matrix` / `ExactMatrix` constructors return `Result` since 0.29.  Each is documented
under `# Panics` on its item.  They
predate point 4 below and are debt, not precedent: new code returns a
`Result` instead, and converting an existing one means removing it from the
allowlist.  Code that multiplies inside an infallible trait operation
(`Ring::mul` of `paramfield::PFrac`) uses `try_mul` and reports an overflow
to its policy (`ParamPolicy::overflow`, since 0.59): the computation is
abandoned — every element counts as zero, so every loop ends — and its
owner discards the result at its next budget or size check (`Budgeted`: the
attempt's budget; `Exact`: a thread-local mark cleared where a computation
starts, read by `ppoly_size_ok`/`papoly_size_ok`).  Up to 0.58 such a
product with an exponent beyond `u32` panicked inside the integrator and
`apart`.

**Both are enforced** by `tests/unit/test_no_panics.rs`, a ratchet over
`src/` with one allowlist per category (`ALLOWLIST`, `ASSERT_ALLOWLIST`, each
entry with its reason).  It fails when a file gains a site beyond its
allowlisted count — and when an allowlisted file *loses* one without the
allowlist being tightened.  The library region of a file ends at its
`#[cfg(test)] mod` test module, not at a `#[cfg(test)]` attribute on a
lone helper `fn` (0.11.1 found twelve panic sites hidden behind those).

#### How to not panic — the practical policy

The three concerns — correctness, performance and honest types — rarely
conflict; when they do, this is the order and the reasoning:

1. **Errors are `SymplexError`, and that is fine.**  It is a 72-byte enum, so
   `Result<Ex, SymplexError>` is 72 bytes against 16 for `Ex`.  Profiling the
   heaviest downstream workload (a certificate search with millions of exact
   operations) shows no measurable time in error plumbing: the cost is in
   `BigInt` arithmetic and the allocations behind it, not in moving a
   `Result`.  Do **not** `Box` errors or invent a second error type for speed;
   do not thread `Result` through *inner loops* either — validate once at the
   boundary (shapes, indices, generator lists), then run the loop on data
   that is known good.

2. **Validate at construction, then rely on the invariant — without a panic
   path.**  `StateSpace::new` checks that `A` is `n×n` and `B` is `n×m`; the
   `matmul`s downstream can therefore not fail, but writing
   `.expect("dimension mismatch")` still leaves a panic in the binary and a
   reviewer who has to re-derive the invariant.  Instead, keep the `Result`
   flowing (the method returns `Result` anyway), or, where the API is
   infallible by contract, use a fallback that is *correct*, not merely
   unreachable — replacing the error by a made-up value (a zero matrix
   where the product should be) is wrong;
   `map_err(|e| internal("state_space", e))?` is right.  Name the invariant in
   a one-line comment where it is established, not where it is used.

3. **`Option` for absence, `Result` for failure, unevaluated form for
   "nothing to do".**  `leading_coeff()` returns `None` for the zero
   polynomial; code that has just checked `!p.is_zero()` should still match
   (`let Some(lc) = p.leading_coeff() else { return p.clone() }`) rather than
   `unwrap()` — the `else` branch documents what the zero case means for that
   algorithm, which is information the `unwrap` throws away.

4. **Indexing is the one place `panic` is idiomatic**, and we follow `std`:
   a plain accessor may panic on an out-of-range index exactly like `Vec`
   (`Matrix::get(i, j)`, `MultiPoly::var(n, i)` with `i ≥ n`) **only if** a
   `try_`/`checked_` sibling returning `Option` exists and the panic is
   documented under `# Panics`.  Anything that is not an index — a shape, a
   degree, a generator list, a parameter that fails to parse — is a `Result`.
   `assert!` on a caller-supplied *shape* is a bug, not a precondition (the
   existing ones are counted by `ASSERT_ALLOWLIST` and may only shrink).

5. **Internal invariants: `debug_assert!`, never `assert!`/`unreachable!`.**
   Release builds must degrade gracefully (return `ComputationFailed` with
   the invariant's name, or the unevaluated form), because a violated
   invariant deep in a simplifier must not take down a CAS server or a
   proof-generation pipeline.  `SymplexError::ComputationFailed { operation,
   reason }` is the right variant; put the invariant in `reason`.

6. **Locks and overflow.**  `Mutex::lock().unwrap()` is a panic on poisoning;
   use `parking_lot` (already a dependency; no poisoning) or
   `unwrap_or_else(PoisonError::into_inner)`.  `u32::try_from(len)` on the
   arena's and the symbol table's indices is the one overflow that still
   panics (`expect("arena overflow: …")` in `Arena::intern`, `intern_num`
   and `SymbolTable::intern`, allowlisted: making them fallible touches
   about 800 call sites).  It takes more than 4 billion nodes to fire; new
   code propagates an overflow as an error or a refusal instead.

7. **Type-level guarantees beat runtime checks when they are free.**  The
   `Sort` phantom on `Expr<S>` (an `Ex` cannot be used where a `BoolEx` is
   needed), sealed `ExactScalar`, and `#[non_exhaustive]` on outcome enums
   and on the option structs that carry it (`LeanOpts`, `PolyhedronOpts`,
   `SosOpts`; most option structs do not yet) are all zero-cost and remove
   whole classes of failure.
   Reach for them before adding a runtime check; do not reach for typestate
   or const generics where a `Result` says the same thing more simply.

When touching a function that violates these, fix it in the smallest
correct way (usually `?` or `let … else`), and tighten the ratchet.

---

## The API Model

Every symbolic operation follows one of these patterns:

### Pattern 1: Always succeeds → returns `Ex`

Operations where "unchanged" or "unevaluated" is a valid result:

```rust
// These always return Ex — never fail
expr.simplify()      // might return input unchanged
expr.expand()        // might return input unchanged
expr.eval()          // might return input unchanged
expr.diff(&x)        // might return Derivative(expr, x) node
expr.integrate(&x)   // might return Integral(expr, x) node
expr.limit(&x, &a)   // might return Limit(expr, x, a) node
expr.laplace(&t, &s) // might return LaplaceTransform(expr, t, s) node
```

### Pattern 2: Might fail → has a `try_` variant returning `Result`

For pipeline/codegen users who need guaranteed closed-form results:

```rust
// CAS-style (always returns something):
let anti = expr.integrate(&x);

// Pipeline-style (Err if unevaluated):
let anti = expr.try_integrate(&x)?;
```

Every method that can produce an unevaluated form has a `try_` twin:
`try_diff`, `try_integrate`, `try_integrate_definite`, `try_limit`,
`try_limit_left/right/dir`, `try_series`, `try_maclaurin`,
`try_series_at_infinity`, `try_summation`, `try_product_over`, `try_laplace`,
`try_inverse_laplace`, `try_residue`, `try_gosper_sum`, `try_solve_ode`,
`try_solve_gt/ge/lt/le`.

`RootOf` and `RootSum` are complete algebraic answers and are **not**
counted by `has_unevaluated()` (changed in 0.2).

The `try_` variant calls the base method, then checks `has_unevaluated()`.
Zero code duplication.

### Pattern 3: Numeric boundary → always `Result`

Operations crossing from symbolic to numeric always return `Result`:

```rust
expr.eval_f64()                 // Err if free symbols remain
expr.eval_complex64()           // Err if can't evaluate
expr.compile(&["x"])            // Err(FreeSymbol / NotImplemented) — Result<CompiledFn>
expr.to_rust_fn("f", &["x"])    // Err if can't generate code
expr.to_c_fn("f", &["x"])       // same, C99
expr.integrate_numeric(&x, &a, &b)   // Err if quadrature does not converge
```

Mathematical outcomes that are facts rather than failures also travel as
`Result`/enum variants: `solve` → `Err(InfiniteSolutions)` for identities and
`Err(NoSolution)` for contradictions; `try_integrate_definite` →
`Err(Divergent)`; `linsolve` → `Ok(LinearSolution::Inconsistent)`.

### Pattern 4: Queries → `Option`

Three-valued queries (yes / no / can't determine):

```rust
expr.is_positive()      // Some(true), Some(false), or None
expr.degree(&x)         // Some(3) or None (not polynomial)
expr.equals(&other)     // Some(true), Some(false), or None
set.contains(&e)        // set membership
matrix.is_symmetric()   // structure tests on matrices are three-valued
```

A new query must return `None` when it cannot decide — never guess. Symbols
without assumptions may be complex; do not assume realness.

A mask of several properties is a conjunction (`Assumptions::query`,
`Ex::query`; since 0.60.1): false as soon as one property is false, true
when all are true, undecided otherwise.  Inside the crate ask one property
at a time, as every call site does (up to 0.60.0 a mask was false only when
all its properties were, so `query(POSITIVE | NEGATIVE)` of a positive
symbol was `None`; nothing in the crate depended on it).  One question has
one implementation: `Context::query` is `Ex::query`, and a new entry point
for an existing question calls the existing one (up to 0.60.0
`Context::query` stopped at the structural rules and was `None` for
`cos(pi/16)`, which `Ex::query` knew to be positive).

### Pattern 5: Structural preconditions → `Result`

Matrix operations that require specific shapes:

```rust
matrix.det()          // Err if non-square
matrix.inv()          // Err if singular
matrix.matmul(&other) // Err if dimensions don't match
matrix.cholesky()     // Err if not symmetric / positive definite
matrix.minor(i, j)    // Err if out of range
```

---

## Expression Nodes

The `ExprNode` enum in `src/base/node.rs` has 96 variants. Its shape methods
(`children`, `for_each_child`, `child_count`, `is_atom`, `map_children`,
`remap_ids`, the binder views) are generated by `#[derive(NodeShape)]`
(`symplex-macros/src/node_shape.rs`, doc-hidden): an `ExprId` field is a
child, a `SmallVec` of ids (or of id pairs) a child list, a field marked
`#[payload]` an id that is not a child, and binders are declared next to their
variant with `#[binder(var = i, scoped(..), outer(..))]` or `#[var_slot(..)]`.
The variants fall into categories:

| Category | Examples | How they work |
|----------|---------|---------------|
| **Atoms** | `Num(NumId)`, `Symbol(SymbolId)`, `Pi`, `E`, `ImaginaryUnit`, `EulerGamma`, `Catalan`, `GoldenRatio`, `Infinity`, `ComplexInfinity`, `NaN` | Leaf nodes, no children |
| **N-ary arithmetic** | `Add(SmallVec)`, `Mul(SmallVec)` | Flattened associative ops |
| **Binary arithmetic** | `Pow(ExprId, ExprId)` | Base, exponent |
| **Functions** | `Sin(ExprId)`, `Exp(ExprId)`, `Gamma(ExprId)`, `Zeta`, `Si`/`Ci`/`Ei`/`Li`, `Polygamma(n, x)`, `KroneckerDelta(i, j)` | Unary or binary |
| **Complex** | `Re`, `Im`, `Conjugate`, `Arg` | Only constructed when realness is unknown |
| **Calculus** | `Derivative(body, var)`, `Integral(body, var)` | Formal/unevaluated |
| **Unevaluated** | `Limit`, `Series`, `Sum`, `Product_`, `LaplaceTransform`, `Residue`, `DSolve`, `ConditionSet`, `Subs(body, var, point)` | Formal results when computation can't produce a closed form (`Subs`: a derivative at a point, `f′(0)` for an undefined `f`) |
| **Algebraic answers** | `RootOf(poly, var, index)`, `RootSum(poly, body, var)` | Exact descriptions of algebraic numbers; *not* unevaluated |
| **Boolean** | `BoolTrue`, `Gt`, `And`, `Or`, `Not` | For inequalities and logic |
| **Sets** | `Interval`, `FiniteSet`, `SetUnion`, `SetIntersection`, `SetComplement`, `EmptySet`, `UniversalSet` | For solution sets; `transforms/sets.rs` computes the normal form |
| **Piecewise** | `Piecewise(Vec<(value, condition)>)` | Conditional expressions |

Every node has a `children()` method that returns its child `ExprId`s. This is
used by tree traversals throughout the codebase.

**Node facts.**  Whole-expression structural questions are answered from
the `NodeFacts` the arena computes when a node is interned: unevaluated
forms, symbols and binders contained, free symbols (a 64-bit mask that
follows the binder rules; symbols beyond the 63rd share a bit and fall back
to a walk), unfolded tree size and height.  Use `walk::has_unevaluated`,
`walk::has_free_symbol`, `walk::free_symbols`, `Arena::is_symbol_free` and
`Arena::tree_size_capped`; do not write a new walk for these.  A new fact is
a flag computed in `facts_of_new` from the children's facts in O(arity),
tested against a walk on every node; a new unevaluated variant must be added
to `is_unevaluated_node` in `arena.rs`.

**Sort keys** are bytes in the arena's pool (`SortKey = [u8]`).  Their
bytes, and the digest of a truncated key (one `FxHasher` write of the full
key, big-endian), are byte-identity critical: a change must show identical
keys on every node of the Rubi arenas.  The leading bytes of a function's
key are its `sort` facet in `funcs!` (`SortSub`); since 0.60 `sort_key.rs`
has no arm per function.

**Internal functions** of an algorithm (the series engine's asymptotic
helpers) are `ExprNode::Internal(InternalFn, args)` nodes declared in the
`internal` block of `func.rs` and built with `Arena::func` — never an
`Apply` with a reserved name, so no user input can reach them.

**Canonical construction.**  Outside `src/base/`, never `intern` a node of a
variant that has a canonical constructor (`Add`, `Mul`, `Pow`, `FiniteSet`,
`Interval`, `SetUnion`, `SetIntersection`, `Piecewise`, `DefiniteIntegral`,
`Subs`, `And`, `Or`, `Not`, `Gt`, `Ge`, `Eq_`, `Ne`): call the constructor
(`arena.add`, `arena.finite_set`, …), or `arena.make(node)` for a node value
(it routes every variant to its constructor; its `match` is exhaustive, so a
new variant must say how it is built).  Up to 0.48 `subs` rebuilt `{x, y}` with
`x, y ↦ 1` as the non-canonical `{1, 1}`.  Enforced by
`tests/unit/test_canonical_construction.rs` (`#[cfg(test)]` code is exempt;
an exception would be allowlisted with its reason and exact count — the list
is empty since 0.52).  `SetUnion` children are sorted by sort key, then the
real intervals among them are put in ascending order of their lower ends
(`canon::order_real_intervals`: `−∞`, exact rationals, constants by 17
digits of `evalf`, ties by sort key) in the places the sort key gave them,
so the set evaluator's normal form (intervals ascending, then the isolated
points) is the canonical union; up to 0.51 `sets.rs` interned that form raw
and `(0, 1) ∪ [2, 3]` was not `==` to its JSON round trip.

---

## How to Add a New Function

Every function — the 43 with a dedicated `ExprNode` variant, the 50 library
functions (nodes `ExprNode::Lib(LibFn, args)`, built by `Arena::lib_apply` /
`Arena::func`, arity-checked; `Apply(name, args)` is a user function), and
the 8 internal asymptotic helpers of the limit engine (`ExprNode::Internal`,
which no name builds) — has one identity, `Func`, and one entry in the
`funcs!` table of `src/base/func.rs`.  Code that treats a family of
functions uniformly asks `Func::elementary()` (sin…atanh, exp, ln) or
`Func::is_special()` through `ExprNode::func_head` (or `ExprNode::func`,
which also returns the arguments), instead of an or-pattern of
variants; a list without a family gets a predicate on `Func` and a test that
it selects the intended set.  The entry is a
`FuncSpec` struct literal with every field mandatory, so a function that
forgets a facet does not compile:

```rust,ignore
/// Imaginary error function `erfi(x)`.
Erfi => FuncSpec {
    name: "erfi", parse: &[], arity: Exactly(1), sort: Named,
    tex: Tex::Call(r"\operatorname{erfi}"),
    pretty: Pretty::Flat,          // no two-dimensional layout of its own
    mathml: Mml::Name,             // `<mi>erfi</mi>`
    conj: Conj::Entire,            // f(conj z) = conj f(z) everywhere (DLMF 7.4.2)
    real_on: RealOn::Reals,        // real for real x
    parity: Odd(Coefficient),      // must match today's sign extraction (a test checks it)
    poles: Poles::None,
    constants: Constants::Evaluated,
    rt: Runtime::helper(Kernel::Unary("erfi"), Backends::VM_RUST_C)
        .scipy("scipy.special.erfi(x)"),
};
```

The data facets — the name, the parser's further names, the argument count,
the sort bytes (node-backed functions keep their historical byte; a test
checks uniqueness), the LaTeX layout, the layout of the two-dimensional
printer (`pretty`), the MathML head (`mathml`), the conjugation rule, the
real-argument domain on which the function is real, parity, poles, constant
folding, and `rt` (which `f64` back ends evaluate it and with which runtime
kernel) — are read from the table by every consumer (`Display`,
`to_latex`, `pretty`, `to_mathml`, the parser, the sort key, `conjugate`,
evalf's conjugate pairing, the assumption system, `compile`, `to_rust_fn`,
`to_c_fn`, the Python printers), so two consumers can no longer disagree
and a new function cannot forget one.

**Names and printed forms (since 0.60).**  The four printers and
`sort_key.rs` have one arm for every function node, `func_nodes!() => …`,
which asks `node.func_head()` for the `Func` and reads `spec()`; up to 0.59
each had an arm per function that repeated the name, the layout or the
byte.  A printed form that is not data is a case keyed by `Func` inside
that arm, with a comment: the postfix `n!` and `lambertw(x, k)` printed as
`W(x, k)` in `display.rs`; the notations marked `Pretty::Own` (`|x|`, an
overline, `ψ⁽ⁿ⁾`) in `pretty.rs` and `Mml::Own` in `mathml.rs`.  The
parser reads a call name with `Func::parsed(name, n)` (the `name`, compared
ignoring ASCII case, or a `parse` name, at an argument count the function
accepts) and builds the call with `Arena::func`.  Do not write a function's
name as a string literal in those six files:
`tests/unit/test_func_names.rs` counts the literals that spell one (see
"A function's name is written once").

**`rt`.**  `Runtime::NONE` (optionally `.scipy("…")` for the Python printer)
when there is no `f64` routine.  Otherwise
`Runtime::helper(Kernel::Unary("name"), Backends::VM_RUST_C)`, which needs
three things: a `pub fn name` in a `// @@begin` section of
`src/output/codegen/numeric_rt.rs` (shared by `compile` and the embedded Rust
runtime), a VM instruction, and a C port (`CHelper { name, deps, src }` in
`codegen_c.rs`, statement for statement, with its dependencies earlier in the
list).  `the_rt_facet_is_what_every_back_end_does` and the two helper tests
fail until all of these agree.
Choose the strongest *true* value and cite DLMF or SymPy in a comment where
it is not obvious; check a `conj` rule in mpmath at two complex points off
the cut; `Conj::No` and `RealOn::Unclaimed` never claim anything.

Code that tells functions apart compares `Func` values from
`ExprNode::func(&arena)` (both families; user functions give `None`) and
builds them with `Arena::func(f, &args)` (arity-checked) — never
`std::mem::discriminant` alone (it is the same for every `Apply`: up to 0.48
evalf took `Shi(z̄)` and `erfi(z)` for conjugates and was off by 2πi) and
never a comparison of `Apply` names.

The algorithmic facets stay where the algorithms are: exact values
(`transforms/eval.rs`, `eval/`), evalf kernels (`transforms/evalf.rs`,
`evalf/`), derivatives (`transforms/diff.rs`), integration tables, series and
limit rules, code generation (`output/codegen.rs`, `codegen/`,
`output/lambdify.rs`).

**A library function** is one `lib` entry in `funcs!` (that also adds the
`LibFn` variant); the compiler then lists the exhaustive matches that need an
arm.  Add its evaluation and derivative where the compiler and the tests
point (its printed forms, parser names and sort key are the entry), and
tests in the current group (`tests/v36/`).

**A node-backed function** needs a `node` entry with an unused `sort` byte,
the `ExprNode` variant (the derive generates its shape), its variant in
`func_nodes!` and its arm in `ExprNode::func_head` (both in `func.rs`; the
compiler checks one against the other), an arm in `Arena::func` (its helper
`node_func`) and in `Arena::make`, and the steps below.

Example of a node-backed function: adding `cot(x)` (cotangent).

### 1. The node already exists

Many functions are represented as compositions. `cot(x)` is `cos(x)/sin(x)`.
Check if the function can be expressed in terms of existing nodes before adding
a new variant.

If it can, add a method to `Expr<Numeric>` in `src/api/expr_funcs.rs`:

```rust
pub fn cot(&self) -> Ex {
    let cos_x = self.cos();
    let sin_x = self.sin();
    &cos_x / &sin_x
}
```

### 2. If a new node IS needed

For functions that need their own identity (e.g., for pattern matching,
display, or special evaluation rules), add to `ExprNode`:

1. **`src/base/node.rs`**: Add variant `Cot(ExprId)` to the enum (the
   `NodeShape` derive generates its shape methods; give binders their
   attribute).  Add its `funcs!` entry in `src/base/func.rs` — the entry is
   also its display name, its parser names, its LaTeX, two-dimensional and
   MathML forms and its sort bytes — its variant in `func_nodes!`, and its
   arms in `ExprNode::func_head`, `Arena::func` and `Arena::make` (all
   exhaustive).

2. **`src/base/sort_key.rs`**: nothing (the `sort` facet: an unused byte;
   the `func.rs` tests check uniqueness).  Sort keys are byte-identity
   critical: never renumber an existing function.

3. **`src/base/walk.rs`, `src/base/compact.rs`**: nothing (generated).

4. **`src/base/assumptions.rs`**: nothing for realness, parity and poles (the
   facets); add a handler only for facts the table cannot express.

5. **`src/output/display.rs`**: nothing (the `name` facet), unless the
   function is not written `name(args)`.

6. **`src/output/latex.rs`**, **`pretty.rs`**, **`mathml.rs`**: nothing (the
   `tex`, `pretty` and `mathml` facets); a notation the facets cannot
   express is a case in the file's `func_nodes!()` arm.

7. **`src/output/tree.rs`**: Add `ExprTree::Cot` variant + serialization.

8. **`src/output/parse.rs`**: nothing (the `name` and `parse` facets;
   `Func::bare_argument` if `parse_implicit` should read `cot x`).  Only a
   name that is not a function of the table — a composition such as `sqrt`
   — is an arm of `call_1` … `call_4` and an entry of `OTHER_CALL_NAMES`.

9. **`src/transforms/diff.rs`**: Add differentiation rule:
   `d/dx cot(f) = -csc²(f) · f'`.

10. **`src/transforms/eval.rs`** (the arm of `eval`) and the fold itself in
    a file of **`src/transforms/eval/`** — the elementary functions in
    `trig_radicals.rs` (`eval_sin`, `eval_cos` and `eval_tan` in
    `integer_sequences.rs`), the library functions in `lib_values.rs`: add
    evaluation for known values (e.g., `cot(π/4) = 1`).

11. **`src/transforms/evalf.rs`** (the arm of `eval_node`) and a kernel in
    a file of **`src/transforms/evalf/`**: add numerical evaluation.

12. **`src/transforms/integrate.rs`** (the dispatcher
    `integrate_node_uncached`) and the rule in the matching file of
    **`src/transforms/integrate/`**: add an integration rule if known.

13. **`src/api/expr_funcs.rs`**: Add the public method on `Expr<Numeric>`.

14. **`src/poly/polybridge.rs`**: Add to the `=> None` arm (not polynomial).

15. **`src/output/codegen.rs`**, **`src/output/codegen/codegen_c.rs`** and
    **`src/output/lambdify.rs`**: Add code generation support (Rust, C99,
    stack-VM) or add to the "unsupported" arm. If a numeric helper is needed,
    add it to `src/output/codegen/numeric_rt.rs` so that `compile`,
    `to_rust_fn` (via `rt_embed.rs`) and the C runtime share one
    implementation.

16. **`src/base/complex.rs`**: nothing for conjugation (the `conj` facet);
    `re`/`im` rules if the function has a closed form for them.

17. **Round trip**: `parse(&ctx, &format!("{expr}"))` must reproduce the
    node; `parse_of_display_and_of_every_name_builds_the_same_node` (in
    `func.rs`) checks it for every function of the table.

18. **Tests**: Add tests in an appropriate test file (see below).

---

## How to Add a New Node Type (Unevaluated Form)

Unevaluated nodes represent computations that couldn't produce a closed form.
For example, `Limit(body, var, point)` represents `lim_{var→point} body`.

Follow the steps for a new node above, with these differences: such a node
is not a function of the table, so it has no `funcs!` entry and its printed
forms, its parser arm and its sort rank are written by hand, as they were
for every node up to 0.59.

1. Add its variant to `non_func_nodes!` in `src/base/func.rs` (the `None`
   arm of `ExprNode::func_head`, whose match is exhaustive: a new variant
   must say whether it is a function), and give it an arm in
   `sort_key.rs`, `display.rs`, `latex.rs` and `mathml.rs` (`pretty.rs`
   falls back to the one-line form).  The arm in `mathml.rs` is a
   presentation, not a refusal, for a node that a stable function
   returns (up to 0.60.0 `to_mathml` refused `RootOf` and `RootSum`,
   which `solve` and `integrate` return and `to_latex` prints).

2. Add it to `is_unevaluated_node` in `src/base/arena.rs` (the per-node
   facts that `walk::has_unevaluated` reads are computed from it).

3. Consider adding it to `ExprType::Unevaluated` in `src/api/expr.rs`.

4. The Display should show the formal notation:
   `Limit(f, x, 0)` or `\lim_{x \to 0} f` in LaTeX.

5. The parser in `src/output/parse.rs` should be able to parse the Display
   output back (round-trip): an arm of `call_2` … `call_4` and an entry of
   `OTHER_CALL_NAMES`.  A binder form that is not a comma list has a
   method of its own, called from `parse_function_call`
   (`parse_piecewise`, `parse_image_set`, `parse_root_sum`), its entry
   in `OTHER_CALL_NAMES` and a line in the inline test
   `known_functions_table_matches_call_tables` (up to 0.60.0
   `RootSum(poly, var -> expr)` was printed and not read).  An optional
   part of a node is printed as one more argument and has its arm in
   the next `call_n` (`Limit(f, x, a, left)` in `call_4`, since 0.60.1).

---

## Testing

### Test Organization

Tests are in `tests/` (integration tests: about 12,100 `#[test]` functions
in about 590 source files) and inline `#[cfg(test)]` modules (about 3,000
unit tests).  At 0.60 `cargo nextest run` runs about 15,130 tests (15 more
are skipped: the `#[ignore]`d ones), and `cargo test --doc` about 1,370
doctests: the API examples and the Rust blocks of `README.md` and the book
(`src/doctests.rs` includes `README.md` and every book chapter, so
the prose examples compile and run under `cargo test --doc`).

The integration-test sources are grouped into **36 test binaries** (linking
277 separate debug binaries took ~6.5 min and ~13 GB of `target/`). Each
former top-level file is a module of its group, so a test is addressed as
`<module>::<test>` inside `--test <group>`.  Up to 0.34 each minor release
got a `vNN` group of its own (`v23` holds the 0.24 tests); since then a
group spans several releases (`v34`: 0.35–0.38, `v35`: 0.39–0.46, `v36`:
0.49–0.60) and new tests go into the newest one.  `tests/README.md` lists
every group's modules:

| Binary (`--test …`) | Sources | What they test |
|---------------------|---------|----------------|
| `v04` | `tests/v04/v04_<area>.rs` | One suite per 0.4–0.6 feature: `polyhedron` (parametric polyhedron certificates; emitted Lean pinned to the Mathlib-compiled `tests/fixtures/polyhedron_certificates.lean`), `polytope`, `sos` (pinned to `tests/fixtures/sos_certificates.lean`) |
| `v09` … `v36` | `tests/vNN/vNN_<area>.rs` (1–87 modules each) | Feature and regression suites of 0.9 → 0.60 (stats families, data statistics, ANOVA, Cox, numdist, the exact kernel, …; from `v21` on mostly regressions found by audits, differential hunts and the nightly fuzzing, each test citing its oracle). `v36` is the current group |
| `v03` | `tests/v03/v03_<area>.rs` (11 modules, ~480 tests) | One suite per 0.3 feature: `poly_view`, `poly_symbolic_coeffs`, `ratsimp`, `linprog` (full KKT check of every optimum, Farkas vector verified), `normalforms` (defining invariants, not pinned answers), `matrix_ergonomics`, `optimize`, `certificates`, `assumptions_poly`, `exact_matrix`, `user_notes` |
| `v03_oracle` | `tests/v03_oracle/v03_oracle_*.rs` | SymPy oracle for the 0.3 API (`tests/fixtures/v03_cross_validation.json`) |
| `v02` | `tests/v02/v02_<area>_<topic>.rs` (62 modules, ~1,050 tests) | One suite per 0.2 feature area: `backends_{c,codegen,compile,cse}`, `basefix_*`, `ergonomics_*`, `integration_{battery,definite,residue}`, `matrices_*`, `nodes_*`, `ntheory_*`, `numfix_*`, `sets_*`, `simplify_*`, `solvefix_*`, `solving_*`, `summation_*`, `transforms_*` |
| `v02_oracle` | `tests/v02_oracle/v02_oracle_*.rs` | SymPy oracle for the 0.2 API (`tests/fixtures/v02_cross_validation.json`); see `tests/README.md` |
| `unit` | `tests/unit/test_*.rs` (170 modules, ~3,990 tests) | Unit-style suites per module / feature, and the fourteen source ratchets (see "Type rules that keep bug classes out"), e.g. `test_known_answers` (256 exact symbolic results against textbook answers), `test_sympy_cross_validation` (263 fixtures against SymPy 1.14, `tests/fixtures/sympy_cross_validation.json`), `test_correctness_audit` (108 fixtures: FTC verification, definite integrals, `tests/fixtures/new_capabilities.json`), `test_cross_context` (cross-context safety guards), `test_ode_comprehensive` (ODE solver classes), `test_rootof` (RootOf solver + numerical evaluation), `test_hard_math` (edge cases and negative tests) |
| `legacy` | `tests/legacy/{round*_*,math*_bugs,bugfinder*,*_validation,…}.rs` (23 modules) | Regression suites from bug-hunting rounds |
| `proptests` | `tests/proptests/{proptest_*,test_proptest_new,test_quality_props,test_units_proptest}.rs` (~150 properties) | Algebraic axioms, idempotence, value preservation, round-trips. `<stem>.proptest-regressions` files live next to the sources |
| `perf` | `tests/perf/{simplify_perf_test,perf_analysis}.rs` | `#[ignore]`d benchmarks (`--release -- --ignored --nocapture`) |
| `ui_tests` | `tests/ui_tests.rs` + `tests/ui/*.rs` | trybuild compile-fail snapshots for the units type system (`Mass + Length` must not compile) and the shape errors of `matrix!` |

Naming convention for new work: `tests/v36/v36_<batch>_<area>.rs` in the
current group (`v36_s15r3_bounds.rs`; the batch tag keeps the files of one
round of work together), or the appropriate `tests/unit/test_*.rs`; one
behaviour per test function, named
for the mathematical fact it checks (`divergent_interior_pole_is_err`, not
`test1`). Register a new file in its group root (`tests/v36.rs`, …) with
`#[path = "v36/<file>.rs"] mod <file>;` — do not add new top-level
`tests/*.rs` files, each one is another linked binary. Inside a module use
`use super::common;` (not `mod common;`) and `include_str!("../fixtures/…")`.

### Running Specific Tests

```bash
# Recommended: cargo-nextest — one process per test, parallel across binaries,
# per-test timeouts from .config/nextest.toml (a hang is killed and reported)
cargo nextest run
cargo nextest run -E 'test(/^v03::/)'                      # one binary
cargo nextest run -E 'test(/^unit::test_known_answers::/)' # one former file

# Run one test binary
cargo test --test v03

# Run one module (the former file tests/test_known_answers.rs)
cargo test --test unit test_known_answers::

# Run one test function (module-qualified, or just a substring)
cargo test --test unit test_known_answers::diff_power_rule
cargo test --test unit poly_

# Run with output visible
cargo test --test unit test_known_answers:: -- --nocapture

# Run only lib unit tests (faster)
cargo test --lib

# Run only doc-tests
cargo test --doc

# Everything the CI Test job runs (UI snapshots excluded — see below)
cargo nextest run --profile ci --lib --bins --tests --examples -E 'not test(units_compile_fail)'
cargo test --benches
cargo test --doc
```

### SymPy oracle fixtures

JSON fixture files under `tests/fixtures/` hold results computed by SymPy
and are consumed by `test_sympy_cross_validation.rs`,
`test_correctness_audit.rs`, the `v02_oracle_*` suites and the 0.3 oracle
(`v03_cross_validation.json`: `Poly`, `cancel`/`ratsimp`,
`sympy.solvers.simplex`, Hermite/Smith normal forms). Regenerate them with
the scripts in `scripts/` inside a Python virtualenv with SymPy installed:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install sympy
python3 scripts/generate_sympy_fixtures.py > tests/fixtures/sympy_cross_validation.json 2> fixture_generation.log
python3 scripts/generate_new_fixtures.py    # new_capabilities.json
python3 scripts/gen_new_fixtures.py          # new_features_cross_validation.json
python3 scripts/generate_v02_fixtures.py    # v02_cross_validation.json
python3 scripts/generate_v03_fixtures.py    # v03_cross_validation.json
```

Each fixture records the SymPy version and generation time; a test compares
symplex's result numerically (at several points) or structurally, never by
string equality with SymPy's printer.

### UI compile-fail snapshots

`tests/ui/*.stderr` are tied to a specific rustc's diagnostic wording, so
they run only on the pinned toolchain (`UI_TOOLCHAIN` in `.github/workflows/ci.yml`,
currently `1.95.0`). When bumping that toolchain, refresh the snapshots and
review the diff:

```bash
TRYBUILD=overwrite cargo +1.95.0 test --test ui_tests
```

### Timeouts and granularity

- A test should finish in about a second in debug mode and must finish in
  under 10 s (the same rule as `tests/README.md`).  A case that needs longer
  is `#[ignore]`d with the reason and both times ("about 20 s in a debug
  build (0.8 s in release)") next to a quicker variant that runs by
  default.  nextest reports a test as slow after 30 s and kills it after
  two minutes (60 s and five minutes in CI's `ci` profile,
  `.config/nextest.toml`); the few tests it reports as slow are debt, not
  precedent.  A test
  that bounds the time of a call — a budget (Gruntz, rewrite, eigenvalue
  swell), a refusal "at once", a fast path — uses the group's `perf_assert`
  module (`tests/perf_assert/mod.rs`, declared in every group root that
  times something; `use crate::perf_assert::{assert_fast, CpuTimer,
  cpu_time};`): `assert_fast!(Duration::from_secs(2), e.series(&x, &zero,
  6))` fails the test if the expression took more than the budget ×
  `SYMPLEX_TEST_TIME_SCALE` of the **thread's CPU time** and returns the
  value; `CpuTimer::start()` … `.assert_within(budget, "what")` for a region
  spanning several statements; `cpu_time(|| …)` to compare two workloads.
  Never `Instant::now()`/`.elapsed()` in a test: a wall-clock bound measures
  the machine (CI failed on 2.02 s against a 2 s bound), and
  `tests/unit/test_timing_asserts.rs` fails on a new one.  A budget should
  be ≥ 5× the CPU time measured alone; CI sets `SYMPLEX_TEST_TIME_SCALE=2`.
  Prefer a structural assertion where one exists; the time bound is the
  backstop.  Work on another thread is invisible to the thread's clock (a
  `CpuTimer` is `!Send`): time it inside that thread.
- Keep one mathematical fact per test function. Long "kitchen sink" tests
  hide which behaviour regressed and defeat `--skip`/filtering.
- Property tests that skip cases must use `common::BailCounter` and call
  `assert_not_vacuous()` / `assert_skip_rate_below(rate)` so a test that
  skips everything fails instead of passing vacuously.
- Use `timeout` when running long suites locally; CI has per-job limits.
- **Run the gate one bounded stage at a time and read the log, never the
  terminal.**  `scripts/gate.sh` runs each stage under `timeout`, writes
  its full output to `$GATE_LOG_DIR/<stage>.log` and prints one summary
  line, so a failing test is named by `grep FAILED` on a log that already
  exists — re-running a twenty-minute stage to learn *which* test failed
  is the one thing not to do.  When something is slow, run the smallest
  unit that reproduces it (`cargo nextest run -E 'test(name)'`, `cargo test
  --doc -- path::to::item`) with a 60 s cap and `--test-threads=1` so the
  harness names the test in flight.
- **The doctest stage is ~1 s of test time, or it is broken.**  Rustdoc
  compiles all ~1,370 doctests (API, README and book) into one merged binary; if *one* doc example
  fails to compile there, it silently falls back to compiling every doctest
  standalone (~3 s each, twenty minutes in all).  A single trivial doctest
  reporting `finished in 3.6s` instead of `0.00s` is the signature; find
  the broken example with `git diff -- src | grep '^[-+].*///'`.
- **Several people on one change:** agree on who edits which files before
  starting, register the new test files as stubs first (so that the tree
  compiles for everyone and `tests/v36.rs` is edited once), and format
  once at the end with `cargo fmt --all`.  Builds that share a `target/`
  directory see each other's half-finished edits: a build break in a file
  you do not own is the owner's to fix.

### Finding where a slow or memory-hungry call spends its time

The expensive steps of an algorithm are wrapped in `stage!`
(`src/base/stage.rs`): a `DEBUG` span named after the step, which
records the time taken and the arena nodes created.  Nothing is computed
unless a subscriber enables the span, so the stages stay in release
builds.  With a subscriber installed (any example or probe that calls
`tracing_subscriber::fmt().with_env_filter(…).init()`, or
`rubi-harness --probe`):

```bash
# The stages that took ≥ 250 ms or created ≥ 200 000 nodes, innermost first,
# each with the chain of stages above it:
RUST_LOG=symplex::stage=debug cargo run --release --example my_probe
#   DEBUG integrate:substitutions:integrate:risch_rational:lrt_prs: …
#         hot stage stage="lrt_prs" ms=23505 new_nodes=0

# Every stage entered (with its expression, clipped) and left:
RUST_LOG=symplex::stage=trace …
```

A call that never returns — a timeout, a memory cap — never logs its
`exit`; the last `enter` lines under `trace` name the step it is stuck in
and the expression it is stuck on.  When the step is not a stage yet,
sample the running process (`sample <name> 1` on macOS, `perf top -p` on
Linux) to find the function, then wrap its call in `stage!` so the next
person sees it directly: `stage!(arena, "name", expr, call(arena, …))`
evaluates to the call's value.

### Test Helpers

`tests/common/mod.rs` provides shared helpers:

| Helper | Purpose |
|--------|---------|
| `assert_math_eq(a, b, var, label)` | Numerical equality at multiple points |
| `assert_math_eq_rational(…)` | Same, at rational points (avoids float artefacts) |
| `assert_ftc_tol(integrand, var, tol, label)` | FTC: d/dx(∫f dx) ≈ f |
| `verify_roots(poly, var, roots, tol)` | Substitute roots back, check ≈ 0 |
| `verify_ode_first_order(…)` | Substitute an ODE solution back |
| `assert_simplify_preserves_value(…)`, `assert_expand_preserves_value(…)` | Value preservation under rewriting |
| `assert_canonical_eq(a, b, label)`, `canonical_eq` | Structural equality after canonicalisation |
| `assert_display_eq`, `assert_display_contains` | Display checks (use sparingly) |
| `approx_eq(a, b, tol)` | Float comparison with NaN/Inf handling |
| `BailCounter` | Skip-rate accounting for property tests |

All helpers extract the context from the expressions passed to them (via
`expr.context()`) rather than using any global. This prevents cross-context
mixing.

### Writing Good Tests

1. **Use `Context::new()` at the top of each test** — don't share contexts
   between tests.

2. **Prefer numerical verification** over string comparison:
   ```rust
   // Good: verifies math
   let result = expr.diff(&x).integrate(&x);
   common::assert_math_eq(&result, &expr, &x, "FTC roundtrip");

   // Fragile: depends on display format
   assert_eq!(format!("{result}"), "x^2 + 1");
   ```

3. **Test the `try_` variant** when testing operations that can fail:
   ```rust
   let result = hard_expr.try_integrate(&x);
   assert!(result.is_err(), "should not find closed form");
   ```

4. **Negative tests**: Verify the library doesn't return wrong answers:
   ```rust
   // If solver returns roots, they MUST satisfy the equation
   let roots = poly.solve_or_empty(&x);
   if !roots.is_empty() {
       common::verify_roots(&poly, &x, &roots, 1e-6);
   }
   // Divergent integrals must not produce a finite number
   assert!(matches!(
       x.powi(-2).try_integrate_definite(&x, &ctx.int(-1), &ctx.int(1)),
       Err(SymplexError::Divergent { .. })
   ));
   ```

5. **Assumptions are part of the test**: a symbol without `Real` may be
   complex. If a result is only true for `a > 0`, declare it
   (`ctx.symbol_with("a", &[Assumption::Positive])`) and add a second test
   showing the unassumed case stays unevaluated.

6. **Examples are tests too**: every `examples/*.rs` runs in CI and must exit
   0 in a few seconds. `examples/readme_snippets.rs` mirrors every code block
   in `README.md` — update it when you change the README.

---

## Continuous Integration

`.github/workflows/ci.yml` runs on every push and pull request to `main`
(and on demand).  Every job has a `timeout-minutes` limit, so a job that
hangs is cancelled and reported:

| Job | Toolchain | Limit | What it does |
|-----|-----------|-------|--------------|
| **Format** | `LINT_TOOLCHAIN` (1.100.0) | 10 min | `cargo fmt --check` for the main crate, `symplex-macros`, `symplex-build`, `symplex-wasm` |
| **Clippy** | `LINT_TOOLCHAIN` | 30 min | `cargo clippy --all-targets -- -D warnings` for all four crates; clippy must reject every call in `probes/clippy_canary` (see "The BigFloat boundary") |
| **Dependency policy (cargo-deny)** | — | 10 min | `cargo deny check` (see "Dependency policy") |
| **Test (stable)** | `stable` | 60 min | `cargo nextest run --profile ci --lib --bins --tests --examples -E 'not test(units_compile_fail)'`: every test in its own process, with the per-test limit of the `ci` profile in `.config/nextest.toml` (a test that hangs is killed and named after five minutes instead of holding the job).  nextest runs neither benches nor doctests, so `cargo test --benches` and `cargo test --doc` follow |
| **UI compile-fail tests** | `UI_TOOLCHAIN` (1.95.0) | 30 min | `cargo test --test ui_tests` |
| **Sub-crates** | `stable` | 45 min | `cargo test` for `symplex-macros`, `symplex-build`, `symplex-wasm` (native); `cargo check --target wasm32-unknown-unknown` for `symplex-wasm`; `cargo check` for `fuzz/` |
| **Docs** | `LINT_TOOLCHAIN` | 30 min | `RUSTDOCFLAGS=-D warnings cargo doc --no-deps --document-private-items` (main crate and `symplex-build`), then `mdbook build book` |
| **MSRV** | 1.93.0 | 30 min | `cargo check --all-targets` for all four crates |
| **Semver** | `LINT_TOOLCHAIN` | 45 min | `cargo semver-checks` of the public API against the newest release tag (see "The semver check" below) |
| **API snapshot** | `API_TOOLCHAIN` (nightly-2026-10-14) | 30 min | `scripts/api_snapshot/api_snapshot.sh --check`: regenerates the listing of the public API from rustdoc's JSON and fails if it differs from `api/*.txt` (see "The API snapshot" below) |
| **Examples run** | `stable` | 45 min | Every `examples/*.rs` except `repl` is run to completion with a 300 s timeout |

All jobs run on `ubuntu-latest` (x86-64 Linux); the only other target CI
builds is `wasm32-unknown-unknown`, for `symplex-wasm`.

**The lint toolchain is pinned** (`LINT_TOOLCHAIN` in `ci.yml`, since 0.59).
rustfmt, clippy and rustdoc change what they accept from one release to the
next: on 2026-11-24 the floating `stable` broke three jobs on unchanged
code.  Format, Clippy, Docs and Semver run on exactly the pinned version;
the tests and the examples stay on the floating `stable`, so a compiler
change that breaks behaviour is still seen at once.  Format and lint with
the pinned version locally too (`cargo +1.100.0 fmt --all`): a newer
rustfmt may lay the same code out differently.  To raise the pin, run the
three checks with the new version on a machine —

```bash
cargo +<version> fmt --all -- --check
cargo +<version> clippy --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo +<version> doc --no-deps --document-private-items
```

(for the side crates too, with `--manifest-path`, as the jobs do) — fix
what they report, and change `LINT_TOOLCHAIN` in the same commit.

**The semver check.**  Until 1.0 a minor release may break the public API,
but only on purpose.  The Semver job runs `cargo semver-checks
--baseline-rev <newest v* tag> --release-type minor`.  When it reports a
break, the job fails unless the part of `CHANGELOG.md` above that release's
heading has a section headed exactly `### Breaking` or `### Breaking (API)`:
list the break there, with what a caller has to change, in the commit that
makes it.  `### Breaking (behaviour)` is for results that changed and does
not satisfy the check.  Auto traits count (a lost `UnwindSafe` is a break).
An addition needs no section for the job, and goes under `### Added`.

**The API snapshot.**  `api/public.txt` lists every public item of the
crate — modules, types with their fields and variants, functions and
methods with their full signatures, trait implementations, re-exports —
one per line, sorted.  It is generated, never edited: after a change to
anything `pub`, run `scripts/api_snapshot/api_snapshot.sh` and commit the
changed file with the code, so that the reviewer sees the API change as a
diff of text.  The job fails, and prints the difference, when the file
does not match the code.  The script needs the nightly pinned as
`API_TOOLCHAIN` in `ci.yml` (`rustup toolchain install nightly-2026-10-14`),
because only a nightly rustdoc emits JSON and the JSON's layout changes
between nightlies; the pin is raised on purpose, as described in
`scripts/api_snapshot/README.md`, which also explains each kind of line
and how to read a diff.  The snapshot and the Semver job complement each
other: `cargo semver-checks` reports a removed item, but not a changed
parameter or return type and not an addition — those are visible only in
this file.

**The rewrite toolkit** (`scripts/codemod/`, Python, standard library
only) rewrites the *user side* of the library — tests, examples, the book,
the doctests — for the mechanical parts of the 1.0 API changes (reads of
yes/no queries, error patterns, number types, paths, renames).  It works
on text, rewrites only what it is sure of and lists the rest for a person;
a dry run is the default.  Its README says how a change of the API uses
it; `python3 -m unittest discover -s scripts/codemod/tests` runs its own
tests.

**The lock file.**  CI builds the dependency versions of the committed
`Cargo.lock` (see "Dependency policy"); new dependency releases are seen
by the weekly `dependencies.yml`, not by `ci.yml`.
`.github/dependabot.yml` proposes updates of the GitHub Actions the
workflows use, once a month; it does not touch the Rust dependencies.

`.github/workflows/deploy-book.yml` builds the mdBook and deploys it to GitHub
Pages on pushes to `main`.

`.github/workflows/fuzz.yml` runs every `fuzz/` target for ten minutes each,
nightly and on demand (`workflow_dispatch`, with the time per target as an
input): one job per target, the corpus cached between nights, a failing
input uploaded as an artifact *before* anything replays it, then printed
as base64 and decoded in the log with every replay bounded (an input that
hangs is reported by libFuzzer after 60 s; up to 0.57 the decoding step ran
first, without a limit, and the job was cancelled with the input lost).  A
manual run and the nightly one do not cancel each other.  A red
nightly is a bug report: reproduce it with `fuzz/README.md`, pin it as a
regression test, fix the cause — and read every failing job, not only the
first: one input that fails every night hides the others (the tree of 0.58
was red for five weeks on two slow inputs, and a report of 2026-11-11, a false
positive of the harness, went unread).  `fuzz_stats` checks the symbolic
distributions by properties decided exactly or from certified digits; a
quadrature that disagrees is re-decided by the certified
`integrate_numeric_decimal`, a refusal is skipped (never a failure), and a
slow refusal is a timeout and a finding.  The harness stops starting new
steps of an input after 2 s (since 0.60.1; a clock, so a slower machine
checks less of the input), which bounds an input only if every single
step is fast; its temporary skips are listed in `fuzz/README.md`, each
to be removed with the fix it names.  A property the harness derives
from a formula must hold on the support, not only for the formula (an even
density needs a symmetric support).

`.github/workflows/stats-oracle.yml` runs the statistics oracle nightly and
on demand (`scripts/stats_oracle/README.md`): a release build evaluates the
seed set in `scripts/stats_oracle/cases/` (families × methods, the
`numdist` kernels, hypothesis tests, least squares), and `compare.py`
checks every value against mpmath at two precisions, with scipy and
statsmodels (pinned in `requirements.txt`) judged at their own inputs.  A
red run is a bug report: `SYMPLEX_WRONG` is a wrong value (pin it as a test,
fix the cause); `SCIPY_WRONG` is scipy's error (add it to `allowlist.tsv`
with the mpmath value and a reason, and keep a symplex test recording it);
`REFUSED` is a refusal where a value exists (fix it or allowlist it with
the reason); `HANG`/`PANIC`/`CRASH` are bugs.  Reproduce locally with
`scripts/stats_oracle/run.sh`.  New statistics code gets cases in the seed
set (data files `cases/<prefix>_*.tsv`, its scipy mapping and mpmath formula
in `refs.py`); run the oracle before a release that touches `stats` or the
special functions it uses (`betainc`, `gammainc`, `erfc`, `erfinv`, `ln Γ`).

`RUSTFLAGS=-D warnings` is set globally, so a new warning anywhere fails CI.
Before opening a PR run at least `cargo fmt --all`, `cargo clippy --all-targets`,
`cargo test`, `cargo doc --no-deps`, and the examples.

**Issues and pull requests.**  The pull-request template
(`.github/PULL_REQUEST_TEMPLATE.md`) asks for what was wrong or what is new
(input, old result, new result), the oracle behind every new expected
value, and the checks you ran, and has the short checklist: no panic and
no recursion over expression trees in library code, an answer that is not
proved is refused, a public API change is listed in `CHANGELOG.md`.  Issues
have two templates (`.github/ISSUE_TEMPLATE/`): *Wrong answer, crash or
hang* (the input as a complete snippet, what symplex returns, what is right
and how you know) and *Refusal or missing feature*.  An input that crashes,
hangs or exhausts memory through the parser or the JSON import is reported
privately, as [`SECURITY.md`](SECURITY.md) describes, not in a public
issue.  Every change is reviewed by the maintainer (`.github/CODEOWNERS`);
what happens to a pull request is described in the next section.

---

## Pull Requests

Every pull request gets an answer, and none is closed in silence.

1. **A reply that thanks you.**  The maintainer comments on every pull
   request from a person, whatever becomes of it.
2. **A check of the work.**  We read the change, run the whole gate with it
   applied (the suite, the lints, the golden outputs when results can
   change) and check each new expected value against its oracle.  A
   library disagreeing with symplex is a reason to look, not yet a proof.
3. **One of three outcomes, each said in a comment:**
   - *Complete* — it does what it says, with a test that names its oracle,
     and the gate is green: it is merged with your authorship, and the
     comment says which release carries it.
   - *Not complete, but we can finish it and it fits the current plan* — we
     finish it on top of your commits, which stay yours, and the comment
     says what was added and why (a wider case, a missing test, a doc
     line).
   - *We cannot complete it, or it does not fit* — the comment says why,
     and what would make it mergeable or where the idea is tracked.  Then
     it is closed.
4. **A note when it ships.**  When the release that contains the change is
   published, the pull request (and the issue it fixes) gets a line with
   the version.

Pull requests opened by bots (the monthly updates of GitHub Actions) are
merged once every workflow passes with them, or closed with the reason;
they get no thanks.

---

## Code Style

### Naming

- Types: `PascalCase` (`ExprNode`, `SortKey`)
- Functions/methods: `snake_case` (`canon_add`, `eval_sin`)
- Constants: `SCREAMING_SNAKE_CASE` (`MAX_KRONECKER_DEGREE`)
- Private fields: no prefix (just `id`, not `_id` or `m_id`)

### Tuples versus structs

A tuple whose positions have the *same type* — `(f64, f64)`, `(Matrix, Matrix)`,
`(&Q, &Q)`, `(Ex, Ex, bool, bool)` — lets nothing but memory say which position
is which.  `(lower, upper)` or `(upper, lower)`?  `(Q, R)` or `(R, Q)`?
`(shape, scale)` or `(shape, rate)`?  Our own `gcdex` returned `(g, s, t)` in
`ntheory` and `(s, t, g)` in `poly` until 0.15.  A struct with named fields
costs nothing at runtime and makes the transposition impossible to write
silently, so **on the public surface (function signatures, `pub` fields,
`pub` type aliases) a repeated element type means a struct** — unless the
tuple is one of these, which stay:

- **universal conventions that are pattern-matched at every use**:
  `(x, y)` points, `(re, im)` for *symbolic* parts (`f64` complex values are
  `num_complex::Complex64`), `(numer, denom)`, `(quotient, remainder)`
  (`num_integer::div_rem`), `(base, exp)`, `(var, value)` substitution pairs
  and other key → value pairs (`Vec<(K, V)>` is the ordered-map idiom);
- **ecosystem conventions**: `shape() -> (rows, cols)`;
- **symmetric positions**: the two squares in `n = a² + b²`, a factor and its
  cofactor;
- **enum tuple variants** (`ExprNode::Pow(base, exp)`): always destructured
  by a pattern that names each field at the site.

The rule is not absolute — the goal is to remove a class of stupid mistake
where a type does it for free, not to wrap every pair.  It is
**enforced** by `tests/unit/test_homogeneous_tuples.rs`, a `syn`-based
ratchet over `src/` that fails when a file's public surface gains a tuple
with a repeated element type beyond its allowlisted count (and when an
allowlisted file loses one without the allowlist being tightened).  The
surface is what `pub` and `pub(crate)` expose; since 0.60 `pub(super)`,
which reaches one module tree (a split file and its children), is as
private as an item of an unsplit file.  Each allowlist entry names its kept
sites and the convention that justifies them.

The shared types this produced live in `src/base/interval.rs` and
`src/base/extended.rs` and are in the prelude:

| Need | Type |
|---|---|
| two finite endpoints, any of `[a, b]`, `(a, b)`, `(a, b]`, `[a, b)` | `Interval<T> { lower, upper, kind }` — a `Vec<Interval<Q>>` can mix kinds (a bisection that hits a root exactly yields `[r, r]` beside `(lo, hi]`) |
| closed constraint side, either end possibly absent (LP variable bounds, bounding boxes) | `Bounds<T> { lower: Option<T>, upper: Option<T> }` — **not** `Interval<Option<T>>`, whose `contains` is wrong because `None < Some` |
| an endpoint that may be `±∞`, as a value | `Extended<T> { NegInf, Finite(T), PosInf }` (`src/base/extended.rs`), ordered `-∞ < a < b < +∞`; `Interval<Extended<T>>` for an unbounded interval whose infinite end has a kind |
| symbolic endpoints, `±∞`, unions, set algebra | `SetEx` (`Context::interval`, `Interval<Ex>::to_set`) |
| the support of a distribution | `stats::Support` (pieces are `Interval<Ex>` and points) |
| oriented limits `∫ₐᵇ`, `Σₐᵇ`, a Fourier period | separate `lower`, `upper` arguments — reversal flips the sign, so it is not a set |

Openness is *behavioural* in this crate — `P(X > 2)` and `P(X ≥ 2)` differ
by a lattice point, `maximum` on an open end is a limit that is not
attained, `to_condition` emits `>` or `>=` — so a conversion that drops or
transposes it breaks things silently.  Carry the `kind`; for a decreasing
map use `Interval::reversed` rather than swapping fields by hand.

### Type rules that keep bug classes out

The 0.49 refactor replaced several conventions that had each produced
silent wrong answers with types, and freezes what is left with ratchet tests
(`tests/unit/`: fourteen at 0.60 — `test_alloc_bounds`,
`test_canonical_construction`, `test_decimal_roundtrip`, `test_divisors`,
`test_float_casts`, `test_float_evidence`, `test_func_names`,
`test_homogeneous_tuples`, `test_id_order`, `test_layering`,
`test_no_panics`, `test_special_atoms`, `test_timing_asserts`,
`test_truth_values` — with exact per-file counts that may only shrink;
twelve parse the sources with `syn`, `test_layering` and `test_no_panics`
scan the text; since 0.60 one more ratchet of this kind is one function of
`test_matrix_linalg.rs`, `partial_root_lists_are_not_read_as_complete`, see
"Solving").  A per-file ratchet table moves with the code (since 0.60,
when the eight largest files were split): when code moves from one file to
another its rows move with it, and the sum over a parent and its children
is conserved in every column by a pure move — a count that changes proves
that the move edited something.  Two ratchets are not sums:
`test_layering` counts distinct upward targets per file (the union over a
parent and its children is conserved), and `test_homogeneous_tuples` looks
at `pub` and `pub(crate)` items only.

**Decisions: `Truth`, signs and evidence** (`src/base/truth.rs`).  An
internal yes/no/undecided answer is a `Truth`, not an `Option<bool>`: it has
no `PartialEq<bool>`, no `unwrap_or`, no ordering, so a branch must say
which reading it takes (`t.is_true()` = proved, `!t.is_false()` = not
refuted, or a `match` over the three cases).  Combine answers with
`and`/`or`/`not`/`all`/`any`; do not write Kleene tables by hand.  The signs a
finite value can have are a `SignSet` ({< 0, 0, > 0, non-real}) with
exhaustively tested `neg`/`add`/`mul`/`recip`/`pow_int`; infinities are not
signs — decide finiteness first.  The public query API keeps `Option<bool>`
(Pattern 4); convert at the boundary.  Since 0.60 that conversion has one
name, `Truth::to_option`: a public query ends in `.to_option()` (the
`SetEx` queries), and an `Option`-returning internal function that
propagates an undecided verdict writes `.to_option()?`.  A value from a
decider that still returns `Option<bool>` (`Assumptions::query`,
`solve::assumptions_hold`, `realness::constant_realness_as_declared`) is
wrapped at the call: `Truth::from(q).is_true()`.  The set deciders answer
in `Truth` (`sets::set_contains`, `is_subset`, `is_empty`, …), and
`transforms/sets.rs`, `logic.rs`, `inequalities.rs` and `solve/context.rs`
have no entry left in `tests/unit/test_truth_values.rs`: combine verdicts
with `and`/`or`/`all`/`any` (an explicit loop with `acc.and(..)` where
every operand must still be evaluated), read a memo of verdicts with
`sets::verdict_of` (an operand not reached is `Unknown`), and test an
interval end with `sets::inside_end`.
Membership in a real set needs the element proved real (since 0.60;
`transforms/sets.rs`, "Real members").  An interval, the real line `(−∞, ∞)`
and the evaluator's normal form (`RankSet`, `SetVal::Exact`) are sets of
real numbers; finite sets, image sets and condition sets may hold members
off the line, and `UniversalSet` (`SetVal::Universe`) holds everything — it
is not the line, and nothing true of one may be concluded for the other.
Whatever a real set does to an element — contain it, absorb it in a union,
remove it in a difference, cut it in an intersection, bound it in an
inequality — goes through a three-valued realness test: `member_realness`
for an element (`±∞` and `zoo` are no real numbers, `nan` is undecided, a
symbol without assumptions is not known to be real), `members_real` for a
whole operand.  Proved real: compare.  Certainly not real: not a member
(`{i} \ ℝ = {i}`, `[0, 1] ∩ {i} = ∅`).  Undecided: the membership is
undecided and the set expression stays as it is (`{x} ∪ ℝ` keeps `x`, `{x} \
ℝ` and `ℝ ∩ {x}` stay) — never simplify on "not known to be non-real", and
never read realness off a comparison (`i + 1/2 − i > 0` orders nothing) or
off a 16-digit value (`numeric_value` ranks a non-rational constant only
when it is proved real).  A shortcut on "the exact part is the whole line"
(`is_real_line()`) must say what it does with the opaque parts
(`parts_beyond_the_line`, `parts_minus_real`, `meet_with_line`).  An order
relation holds only between real numbers: the inequality solvers return the
empty set where a side is certainly not real (`logic::known_nonreal` for a
constant side or bound, `inequalities::real_points` for a side with the
variable), take a plain parameter for real, and refuse a bound whose
realness is undecided.  Up to 0.59 the evaluator gave ℝ and the universal
set one value and asked none of this: `{i} ⊆ ℝ` was true, `{i} \ ℝ = ∅`,
`{i} ∪ ℝ = ℝ`, `U \ {0} = ℝ \ {0}`, and `|t| + i > 0` held on `(−i, ∞) ∪
(−∞, i)`.

Only `Evidence::Exact` or `Evidence::Certified` may justify a claim (a
returned value, a dropped term, a leading coefficient, `Inconsistent`, a
branch that drops cases).  `Approximate` (an `f64`, a value zero only to the
working precision) and `Probabilistic` (random sampling) may only guide a
heuristic whose result is right either way — the choice among pivots that
are proved nonzero, the order of methods — and are read visibly with
`truth_at_least(Evidence::Approximate)`.
**Breakpoints of integrands** (`calculus_util`, `definite`, quadrature) are
exact expressions; their `f64` value is a guide accurate to 2⁻⁴⁶.  Order or
merge them with `cmp_values` (by value only when more than 2⁻⁴⁰ apart,
otherwise by a certified sign), never with an `f64` tolerance; periodic
families are enumerated over the exact bounds (`ScanRange`) from certified
integer parts; a principal root list of a periodic composition sets
`BreakScan.periodic` (one period, not every zero); an `f64` quadrature never
trusts its error estimate across a kink.  Limits: a rule resolves a
reals-only function (`floor`, `ceiling`, `H`, `δ`, `min`, `max`) only for an
argument proved real near the point; the Gruntz `ω` must be positive — split
non-real exponents, and never test non-realness by "contains `i`".  A part
free of the variable is its own limit (`series::locally_real`, since 0.60):
never ask `limit::safe_substitute` for the value of an expression without
the variable, it has none to give (up to 0.59 the constant base of `π^(x +
1)` was therefore not shown real beside a cut, and the series was expanded
three times).
An unevaluated one-sided limit is `Limit(f, x, ray)` with the open ray of
approach (`(-oo, a)`, `(a, oo)`) in the third slot (since 0.60.1): build it
with `limit::unevaluated_limit`, read the slot with `limit::limit_approach`,
and treat the slot as an opaque expression of the outer scope everywhere
else.  The node keeps its layout, its hash and its JSON form; up to 0.60.0
the side was dropped, and the two one-sided limits at a jump were one node
whose difference was 0.

**Primality is a claim like any other** (`Ex::query`, "Primes"; since
0.60.1).  `prime` is `True` only below 3.3·10²⁴, where Miller–Rabin with
the first 13 primes is deterministic; beyond it a witness proves
`composite`, and a probable prime (Baillie–PSW, any number of Miller–Rabin
rounds) is `Unknown`.  `ntheory::isprime` and `Ex::is_prime_value` are value
functions that report the test, and their documentation says so.  A fact
that is there is used: up to 0.60.0 the query was `None` for `1`, for `-7`
and for every integer above 2⁶⁴, and `is_prime_value` was `None` for `1/2`.

Use `evalf_verdict(..).proven()` or `evalf_settled` with
`Settled::Certified` for a numeric proof; `evalf_complex`, `eval_const_f64`
and `expr_to_f64` drop the evidence.  Bernoulli numbers use `B₁ = +1/2`
everywhere; a formula written for `B₁ = −1/2` flips that one sign
explicitly.  Zero tests of constants in limits and
series go through `calculus_util::hidden_zero` (`Verdict<Truth>`): an
undecided coefficient may be kept as a value but never divided by, dropped,
or taken as a leading term; new exact rewrites go in `exact_rewrites`, each
an identity of the principal values wherever both sides are finite.  Since
0.60 `hidden_zero` asks `calculus_util::enclosure` before its ladder: a
certified complex rectangle of the constant, by outward-rounded `f64`
interval arithmetic on the memoised enclosures of the operands of sums,
products, integer powers and rationals, and one `evalf` at 20 digits for
any other node.  A rectangle that excludes 0 is the certified "nonzero";
anything else — 0 inside, no enclosure — goes through the ladder unchanged,
so the fast path never decides a zero and never changes an answer (the
coefficients of a composed series are built from coefficients tested
before; up to 0.59 each test evaluated its whole constant again).  Logs
and powers of a constant are rewritten only where its value is certified on
an axis (`values_on_axes`: `ln`, `√`, `(e^z)^q` by their principal values),
logarithms combine only over certified positive reals (`logs_combined`),
and a coefficient with parameters is decided by its symbol-free parts
(`series::parameter_parts`): `ω·k` is 0 when `k` is proved 0, undecided
otherwise, never "nonzero".  A
symbol-dependent quantity certified nonzero at a sample point that respects
the assumptions (`identically_zero::nonzero_at_a_sample`) is *not
identically zero*: the generic-solution convention may treat it as nonzero.
Do not make a decider stricter alone if its callers read its `false` as a
claim: give it a three-valued result and make the callers refuse on
`Unknown` first.  A public function that must stay a `bool` until its
signature changes reads such a result with `.is_true()` and documents
`false` as "not proved": `ode::ode_solution_verdict` (since 0.60.1) is
the three-valued form of `checkodesol`, and `False` there is a certified
nonzero residual at a point of the declared domain (up to 0.60.0
`check_ode_solution` was `false` for six of 24 correct solutions, with
nothing to tell them from refuted ones).
A value that is not a truth value gets an enum of its own
(`TrigZeros`, `Parity`), not an `Option<bool>` whose `None` means something
else.  `tests/unit/test_truth_values.rs` counts per file the remaining
`== Some(bool)` comparisons, `unwrap_or(bool)` reads, internal functions
returning `Option<bool>`, `Option<…>` sign encodings and `(ExprId, u8)` memo
keys; the counts may only shrink.

**Pivots** (`matrix::pivot_truth` in `src/domains/matrix/eigen.rs`; since
0.60.1).  Whether an entry of a matrix is zero is asked in one place,
`pivot_truth` (a `Truth`), in the generic reading: `True` is a proof that
the entry vanishes for every value of its symbols, `False` a proof that it
does not vanish identically (a constant proved nonzero, an exact
rational-function test, or a certified nonzero value at a sample point that
respects the assumptions), `Unknown` neither — a value that is zero to the
precision reached is `Unknown`.  An elimination takes for the pivot of a
column the first entry proved nonzero; it may clear a row by an entry that
is not proved zero (that is right whatever the entry is), and it must not
divide by one.  When a column has an undecided entry and none proved
nonzero, a function that returns `Result` or `Option` refuses
(`rref_decided`, `rank_decided`, `undecided_pivot`); `rref`, `rank` and
`nullspace`, which return plain values, take the entry for a pivot and
document it; a verdict that is a `bool` is `true` only on a proof
(`StateSpace::is_controllable`).  Do not read the test with `.is_true()` or
`!..is_true()` to choose a pivot: up to 0.60.0 `pivot_is_zero(e)` did, and
`[[sin(1)² + cos(1)² − 1]]` had rank 1.  `is_true()` is right only where the
result is right either way (the block split of the eigenvalues).  The
certified tests are charged to the work budget of the matrix call
(`PivotWork`, 60,000 units; the units are nodes of the entries, never time):
a new proof step charges its cost before it runs and returns `Unknown` when
the budget does not cover it; only the exact tests and `Ex::is_zero()` are
free.  The block split of the eigenvalues keeps the cheap test
(`pivot_is_zero`): it decides which characteristic polynomials are solved,
and a finer split is taken only from a spectrum that the cheap one completed
(the first version of the fix split by the certified test, and `eigenvals`
of a matrix that 0.60.0 refuses in 0.13 s took 38 s).  A certified nonzero
value at one sample point proves "not identically zero"; it proves "nonzero
for generic values" only for an expression that is meromorphic in the
symbols (`analytic_by_form`) — a difference of two eigenvalues in radicals
can vanish on one side of a branch cut and not on the other (`x − (x + 2 −
√((x − 2)²))/2`), so eigenvalues are proved different by the discriminant of
the characteristic polynomial or not at all (`distinct_eigen_pairs`).  A
verdict of a public query that is meant for every value of the symbols
(`is_diagonalizable` → `Some(true)`) needs more than the generic
elimination: a quantity proved finite and nonzero at every point (up to
0.60.0 `[[1, 1], [0, x]]`, a Jordan block at `x = 1`, was `Some(true)`).

**Float evidence** (`tests/unit/test_float_evidence.rs`).  A computed
float compared with a small threshold (`|v| < 1e-12`, `z.im == 0.0`) is a
guess, not a proof: it may steer a heuristic whose result is right either
way, never a claim.  Decide claims with `hidden_zero`,
`evalf_verdict(..).proven()`, `cmp_values` or certified root disks.  The
ratchet counts such comparisons per file (`tol`: an operand with a float
literal ≤ 10⁻³ or a same-file constant bound to one; `zero`: `==`/`!=`
against `0.0`); counts may only shrink, and the table is the review queue.
Poles of rational functions carry the certified sign of their real part
(`PoleView::re_sign`); a residue's pole order counts a coefficient as zero
only when proved (`residue::coefficient_is_zero`).  The calculus files were
reviewed at 0.53: what is left there guides quadrature, refuses, or reads a
certified 16-digit value with a margin far above its error, and each site
says so in a comment.  Which side of the real axis a constant lies on is
`realness::imaginary_part_sign` (a `SignSet`; `REAL` means undecided, so
refuse); `expr_to_f64` refuses a constant whose certified evaluation keeps a
nonzero imaginary part, however small.  Sampled checks certify the sign at
every sample and never skip a sample at 0.  `ln u` next to `Ei/Ci/Chi(u)`
with coefficients summing to 0 does not jump across the cut
(`definite::cancelling_log_steps`).  The solvers, sets, inequalities, the
API layer and `nroots` were reviewed at 0.54: the members of a family `a +
b·n` in a window are listed from an exact index range
(`solve::lattice_indices`, `integers_around`: `eval` of `floor`/`ceiling` of
`(end − a)/b`), never from `f64` quotients cast with `as`; their `f64`
values only drop members far outside the window (`sets::Window`).  A zero is
reduced into a period by a certified floor; two critical points are merged
only when their difference is proved 0; a root from `nroots` is real when
its certified disk shows it; `compare_numeric` orders only constants proved
real and `check_solution` answers from `is_zero`.  The applied domains were
reviewed at 0.55: the `f64` thresholds left in `stats/*`, `optimize`,
`robotics`, `quaternion`, `lambdify` and `codegen` are those `f64` APIs' own
(convergence, route choices, scipy/statsmodels tolerances), and SOS's steer
the search (a certificate needs the exact `L·D·Lᵀ` and `verify`, a
refutation an exact negative value at a rational point).  A symbolic result
built from a candidate found numerically or by `solve` is accepted on exact
evidence: the inverse of a transformed distribution must be real and
continuous on the whole image and invert the map at one point
(`wrappers::inverse_on_support`); its candidates are `solve`'s roots for an
image variable carrying the image's sign (`wrappers::image_symbol`; never
redeclare a caller's symbol), a conditional root counts when its condition
holds on the whole image (`roots_on_image`), and a principal odd root is
replaced by the real one only when its twin is one of the roots
(`real_root_form`); values are ordered by their exact order
(`order::exact_order`).  A decimal read from `evalf` is compared unscaled
(`inequalities::Decimal`): never form `10^exp` of a sample value.
A public verdict needs a certified sign (since 0.60).  A function that
answers yes/no about a mathematical object (stable, positive definite,
inside, convergent) reads signs as `SignSet`s proved by the certified
deciders (`Ex::is_negative`/`is_positive`/`is_zero`/`is_real`,
`evalf::const_value(..).re_sign()`, an exact zero test;
`control::proved_sign` and `real_part_sign` are the pattern), never from
`eval_f64()` or `eval_complex64()` compared with `0.0`: the correctly
rounded `f64` of a nonzero value can be `±0.0`, and a real part that is
exactly 0 comes back as rounding noise of either sign (up to 0.59
`StateSpace::is_stable` called `A = [[−e^(−1000)]]` unstable: `−0.0 >=
0.0`).  "On the boundary" is a proved zero, not a tolerance; an undecided
sign makes the verdict `None`, and a verdict about every element of a set
(every pole) needs the set complete.  A theorem about real polynomials
(Routh–Hurwitz, Descartes, Sturm) is applied only to coefficients proved
real — "not positive and not zero" is not "negative" for a complex number,
and a symbol without assumptions may be complex (`control::routh_verdict`;
up to 0.59 `is_routh_stable` called `i·s + 1 − i`, with the root `1 + i`,
stable).

**Printing complex values.**  A result read as a number (`eval_decimal`,
`eval_f64`, `eval_complex64`) shows a part unless it is 0 within its error
ball: `settle` zeroes a part whose ball contains 0 and certifies the digits
of every other part, however small next to the other; `eval_f64` refuses a
nonzero imaginary part.  Since 0.59 a part that is certainly not 0 but that
the evaluation cannot hold — below the exponent range next to a part within
it — makes the number unprintable (`Adaptive::PartBelow`,
`evalf::part_underflow_error`): it is never dropped as a part whose ball
contains 0 is (up to 0.58 `1 + i·e^(−10¹⁰)` printed `1`).  The certificate
is `accuracy::parts_below`, kept per node: the scaled value's mantissa
(`extended::parts_below`), a whole underflow certainly not 0
(`underflow_nonzero`), and the arithmetic of `accuracy::Part` (sums certify
parts below the range only with one known sign; `f(a + iε)` only where
`f′(a) ≠ 0` is known without evaluation).  A part within the underflow
bound without a certificate is dropped as before — widen the certificate,
never the drop — except in a number that depends on a value only a complex
route's scaled form holds below the range (`E_n(z)`, `Γ(a, z)`, `erfc(z)`
off the real axis: `evalf::held_by_complex_route`), where it is refused
(`untracked_part_error`).  "Negligible next to the other part"
(`is_negligible_part`, `is_real_to_digits`) is an approximate reading for
heuristics, never an output rule.  A part dropped because a joint ball is
wide is a bound to tighten in the route that computed it.  A function on
its branch cut whose parts are real functions of the argument computes them
as such and bounds each one (`cspecial::CutRule`; `cerf::on_axis`); an
argument's rounding moves a part by that part's derivative along the cut.
A route's exact-0 part is the value's only where the function is real there
(`cspecial::real_valued`): an asymptotic expansion on the cut drops an
exponentially small part inside its remainder bound (up to 0.53
`im(expint(1, −160))` was `0`).  The evalf files were reviewed at 0.54: an
integer argument of a function defined only for integers is one only when
its value is an exact integer and its bound is exact; an `f64` simulation
that decides how many terms to sum is followed by a bound on the actual
next term; whether an argument is exact is read from its bound, never from
an `f64` radius.  A function evaluated beyond the domain of its real
routine gets the complex value with the convention mpmath and SymPy use on
the cut (elliptic integrals: the limit from below), and a parameter that is
continuous for the new route (a non-integer degree) is not `discrete` in
`sensitivity`.  Each part of a value has its own bound where the route
computes it to its own precision: an analytic function at an argument that
is not exactly real moves each part by its own Cauchy–Riemann combination
of the radii (`accuracy::analytic_parts`: `exp`, `sin`, `cos`, `sinh`,
`cosh`), a power is bounded along the route `evalf` takes (`power_parts`),
and a part is rounded at its own magnitude only where it is a product of
correctly rounded factors (`own_roundings`).  A result read as a number
pursues a part whose ball contains 0 next to a part with its digits to the
cap before dropping it.  `Π(n | m)` beyond Carlson's half-plane is reached
by exact transformations (a common rotation, the principal value plus half
the residue for a real `n > 1`, the split at `1 − m sin²θ = 0`), never by a
numerical initial integral; what they do not reach is refused.  `K(m)`,
`E(m)` for real `m > 1` use the reciprocal-modulus transformation (DLMF
19.7.3), each part a real integral; Carlson's `R_F`/`R_D`/`R_J` sum the
degree-5 series of the remainder with a proven bound.  An
argument that is infinite or undefined only to the working precision is a
domain error the precision search retries, never a non-convergence.
**Underflow and infinity are outcomes, not zeros.**  A value below the
exponent range is refused by `eval_decimal` ("below the range") and is
`0.0` only through `eval_f64`'s documented rule; a route's 0 from exact
arguments is an exact zero only where the function cannot underflow
(`accuracy::underflowed`/`underflow_nonzero` list the functions that can and
give their sign).  A value with no finite value or limit is
`SymplexError::Infinite` (raised only where the route speaks about its own
value, at exact arguments), read by `ValueClass::Infinite` and
`Arena::meets_infinity`; a solver never claims a candidate whose value is
infinite.  A `log₂` of an argument next to 1 is taken from `1 − t`
(`ln_1p`), never from the rounded `t`.  An `f64` kernel whose argument is a
quotient or product of its inputs takes the tail at the exact value
(recover the rounding with fused multiply-adds and feed it into the
expansion, `numdist::Tails::shifted`).  A continued fraction keeps its
argument apart from 1 at the bits of its parameters and refuses at once
when its calibrated step estimate exceeds the budget.  An expansion that
reads its argument through the distance from the mean takes that distance
formed from the caller's inputs (`numdist::bratio_at`): beyond shapes of
about `10²⁸` one rounding of the argument moves the tail by more than
itself.  Never subtract two logarithms of the size of a huge shape; form the
product or quotient first.  A routine whose series or fraction needs work
that grows with a parameter gets a route whose cost does not (the
incomplete beta near its mode: Laplace's method with the Taylor series of
the integrand and Cauchy's estimate), bounds every truncation in logarithms,
and declines when it cannot certify.  The series of `γ(s, x)` far below the
mean of a huge shape is such a routine (since 0.59): its sum is `∫₀^∞
e^(−ψ)` expanded in `s/(s − x)²` (`evalf/gammatail.rs`), with the error at
most the first omitted term because the coefficient polynomials have
non-negative coefficients; where that term is above the precision the
route declines.  A route that computes one part on its
own next to an exact constant (`Ei = ±iπ − E₁(−z)`) bounds that part by its
own sum's error and refuses when it underflows; the error of a non-real
argument moves each part by its own share (Cauchy–Riemann,
`sensitivity::moved_by`).  Exact accessors (`statistic_exact`,
`p_value_exact`) of an `f64` route are `None`: the value is written as
`exp(ℓ)`, never as the dyadic rational of a float.  A route whose prefactor
`e^(−z)·z^c` underflows while its expansion converges returns 0 within an
underflow bound and gives `extended` the pieces (`cexpint::below_range`: the
exponent and the bracket; since 0.59), never the 0 to a loss loop that
reads it as a total cancellation (up to 0.58 `E1(10¹⁰ + i)` was
`PrecisionExhausted`); the test is on the true modulus, so a value within
the range near its bottom stays the ordinary route's.

**Numeric values of constants** (`evalf::const_value`,
`tests/unit/test_decimal_roundtrip.rs`).  Library code reads the value of a
symbol-free constant as a `ConstValue` — the value `evalf` prints, held as
its settled parts: `re_sign`/`im_sign` (certified for a part that is shown;
`ZERO` only for a certified 0; `REAL` = undecided), `re_decimal` (the printed
digits as an unscaled `Decimal`: compare on digits and exponent, form the
rational only within a cap, `to_ratio(max_exp)`), and `to_f64` (correctly
rounded, real values only, for display and guides).  Never print a decimal
and parse it back: a round trip rounds twice, forms `10^exp` of a sample
value (`exp(1000³)` has a decimal exponent of 4.3·10⁸) and loses what the
text does not show (an imaginary part below the printed digits).  User text
goes through `output::parse` or `Decimal::parse` (written exponent capped at
±10000).  The ratchet counts per file the float parses of text and the
decimal printouts read in functions that do not return the string.

**Poles of rational factors in integrals.**  A pole of a factor
`B(x)^(−k)`, `B ∈ ℚ[x]`, is located by Descartes isolation, never through
the root's closed form: comparing or taking limits at a root in radicals
needs minimal polynomials of unbounded degree (up to 0.51 `∫₁² dx/(x⁴ −
3x³ − 2x² + 3x + 2)` ran for minutes).  Divergence is proved when the other
factors are enclosed away from 0 on the isolating interval
(`definite::pole_divergence`, `evalf::nonzero_on`).  A formal definite
integral whose quadrature refuses is evaluated from its closed form
(`definite::closed_form`, sought once per integral); a closed-form ball is
raised in precision until it certifies the working precision, since
`evalf` reads an integral's ball around 0 as an integral equal to 0.

**Quadrature tails and underflow.**  What an integrand does beyond the last
point a quadrature sampled is never read from the samples.  An `f64` tail is
trusted only when the integrand's constants lie within the samples' reach
(`quad::Integrand::constant_scale`: terms whose constants lie within `2^(±S)`
cross over within `2^(2S)`) and the exact limit of `x·f(x)` at the end
(`improper::tail_decay`) is 0 when a sample measured nothing, and not proved
nonzero otherwise.  The arbitrary-precision walk towards an infinite end
does not stop before its nodes reach `2^(2S)`.  A node value that underflowed
bounds the integrand at that node and nothing else: a rule none of whose
nodes saw a value does not converge, and a quadrature none of whose pieces
did is refused.  `exp` of a ball wholly below the exponent range is bounded
by `e^(Re z + r)`.  A finite piece whose oscillation no bisection could
resolve refuses at once (`Quad::unresolvable`), leaving the closed form its
turn.  Since 0.54 the reach is where the integrand's terms cross over
(`quad::Integrand::crossover_reach`): beyond the size of the constants, terms
whose orders of growth differ by `D < 1` cross near `2^(L/D)`, `2^L` the
ratio of their sizes at `x = 2` (`Integrand::close_points`); samples are
added where such terms meet, and an integrand whose terms cross beyond the
last node is refused.  An `f64` estimate with an error of 0 (every node 0)
is no evidence of a zero integral, in `integrate_numeric` as in `evalf`.
A search that cannot succeed is recognised before it is run, in work units.
Tanh-sinh sums any power `x^(−1 + β)`, `β > 0`, at an end — its terms `cosh
t·e^{−βL}`, `L = π sinh t`, are smooth in `t` — but a *change* of the order
at a depth `L` (a `1/x` plateau that ends at `x = e^{−L}` of the piece) is
`1/L` wide in `t` and needs a step below `π²/(L·T·ln 2)` for the tolerance
`2^−T`; a bisection moves it one octave.  Since 0.60.1 `quad::end_layer`
reads the local order off the level-1 terms, and a piece on which tanh-sinh
did not converge whose layer no level before the last resolves (the last
takes more evaluations than the quadrature has) is neither bisected nor
tried by Gauss–Legendre and has no bound (never a bound of its own size: an
integral whose ball contains 0 within the digits is printed `0`).  Up to
0.60 the moment generating function of `FDistribution(11, 2.3·10⁻²⁵³)` spent
10,775 evaluations on bisections before it was refused.  Do not make such a
rule refuse what a level within the evaluations may still resolve: the first
draft refused every layer deeper than 124 octaves and lost the 20 certified
digits of `∫₀^∞ x^(9/2)(1 + 11·10¹⁰⁰x)^(−11/2 − ε)e^{−x/8} dx` (4 s, level
11); the second compared the level's evaluations with those left
(`Work::left`) and lost three 8-digit values that 0.60 certifies (`∫₂^∞
e^(−3x/10⁹⁴)/x dx`): the comparison came after the piece had spent its
evaluations, and the level is estimated for the piece's own tolerance, which
the digits asked do not need.  A rule that refuses is measured against the
release before it — no value may become a refusal — and measured again
whenever `MAX_LEVEL`, `MAX_EVALS` or the precision ladder changes: the level
is an estimate calibrated on them.

**Cut crossings of antiderivatives.**  A Newton–Leibniz difference is
taken across a crossing of a branch cut only at an exactly located
crossing: a linear argument `u = αx + β` meets the real axis at `c = −im
β/im α`, and when `u(c)` is proved on the cut each cut function of `u` is
continued from the side `u` comes from (`definite::side_value`: `ln`, `Ci`,
`Chi` step by `2πi`, `uᵃ` by `e^{2πia}`, `Ei` takes the mean on the axis);
a crossing found only by sampling refuses.  The multiple of `2πi` that joins
the two sides of a spiral antiderivative is an exact integer part
(`integrate::spiral_turns`), never an `f64`; the singular points of
`R(e^{λx})` are located exactly (`definite::exp_rational_scan`).  A polynomial or rational
argument `u = P/Q` meets the real axis at the real roots of `im(P·Q̄)`
(Sturm isolation; on the cut where `re(P·Q̄) < 0`), located whatever the
samples see (`definite::located_polynomial_crossings`).  An antiderivative of
`R(e^{λx})` for a rate neither real nor imaginary keeps each logarithm in a
form that stays off its cut on each side of `|e^{λx+μ}| = |r|`, joined
continuously (`integrate::spiral_continuous`); the definite integrator splits
a `Piecewise` antiderivative at its conditions in `x`.  On a piece in `[0, ∞)` an
argument built from `x^q` and `ln x` is real and crosses no cut
(`real_for_x_in`), without sampling.  Narrow peaks of a product of powers
and an exponential are split off at the maximum of its logarithm
(`definite::envelope_peak`), found in `f64`: a split point is a guide,
never a claim.  The joining point `x_r = (ln|r| − re μ)/re λ` of
`integrate::spiral_continuous` is read as a number where it is one (`ln √q`
written `½ ln q`, `ln_sqrt_as_half_ln`; since 0.59): at a pole on the path
the turn count is an exact integer only then (up to 0.58 `∫ dx/((1+i)ˣ −
2i)`, with its pole at `x = 2`, stayed unevaluated).

**Poles of powers.**  `b^e` is claimed complex, real, rational,
non-negative or finite only off the pole at `b = 0`: for a symbol-free `b`
that means `b` known nonzero (structure, or a certified value via
`Arena::part_signs`) or `e` known non-negative — a constant has no
"elsewhere".  A base with symbols keeps the generic reading "away from the
poles" (`1/x` is real for a real `x`).  A constant without a value is real
only if known finite.  `δ` has a value only at a real argument (an exact-0
imaginary part).

**Values at infinity.**  The values at `±∞` (and other special values such
as `W_k(0) = −∞`) that `eval` folds are also folded when the node is built
(`canon::lib_function_value`), so that `0/f(∞)` is seen as `0/0` and
`0·f(∞)` as `0·∞` before the canonical `0·x = 0` applies: up to 0.51
`0/besselj(1, ∞)` and `0·airybi(∞)` were 0.  A closed form is not
evaluated at an infinite argument (`uppergamma(2, −∞)` gave `nan` from
`(x + 1)e^{−x}`).  Since 0.53 the constructors also fold the values at `−∞`
of a rational order and `E(±∞)` by the principal branches `evalf` uses: a
directed infinity with its exact finite part where the limit has one
(`E₁(−∞) = −∞ − iπ`, `Γ(1/2, −∞) = √π − i∞`), `zoo` otherwise.  A library
function whose values at `±∞` fold, left unevaluated at `±∞`
(`canon::library_value_unknown_at_infinity`), has no value
the constructors know, so `0·f` and `0/f` are `nan`, as are `0·δ(c)` and
`0·H(c)` for a non-real constant `c`.  Since 0.54 every function of the
`funcs!` table is tabulated at the exact infinities `±∞`, `±i·∞`, argument
slot by argument slot (`canon::at_infinity`): a value where the limit has one
(with its exact finite part; `zoo` where the direction is off the axes or a
part diverges at another rate); `NoFiniteValue` where it is infinite in no
tabulated direction, has no limit (`Ai′(−∞)`) or depends on a parameter
without a value (`hermite(n, ∞)`): then `0·f`, `0/f` are `nan`;
`Untabulated` where a finite value may exist that the table does not build
(the degree slot of an orthogonal polynomial): then `0·f` stays 0.  A
pattern may answer `NoFiniteValue` only when every finite value it can have
is folded.  A finite slot is a number, an argument with symbols (a generic
point) or a constant with a certified value (`finite_argument`); `Π` folds
only where the other argument is known not to be 1 (`off_the_pole`).  The
constructors of `Si`, `Ci`, `Ei`, `li`, `ζ`, `ψ⁽ⁿ⁾` are the `eval_*`
functions of `transforms/eval/special02.rs`; their values at infinity live
there.
`eval` evaluates no closed form at an argument with `±∞` in it.  A fold
that holds for every finite value of one argument when another is special
(`Γ(s, ∞) = 0`, `E_ν(∞) = 0`, `J_ν(∞) = 0`) checks that
argument with `ValueClass::of(arena, a).is_finite_proved()`; `γ(s, 0) = 0`,
listed here up to 0.58, holds only for `re s > 0` and needs that proved
(see "Special values are classified").  A function at `zoo` has no value unless
tabulated, and one left unevaluated at a `zoo` argument (not a user
function, not a degree-0 polynomial) makes `0·f` `nan`
(`canon::unevaluated_at_complex_infinity`).  `Min`/`Max` read their arguments
as real: `±∞ + c` is `±∞` there, and `+∞` absorbs a `Max`, unless an
argument is known not to be real.  An infinity in a direction with
parameters (`sign(k)·∞`) says nothing about a finite part (since 0.59): a
constant factor with parameters keeps the perpendicular part of its
factor's limit (`limit::parametric_scale`: `(y − 1)·Γ(0, x)` at `0⁻` is
`sign(y − 1)·∞ − iπ·(y − 1)`), and two equal one-sided `sign(k)·∞` are a
two-sided limit only when `f/k` has the same finite imaginary part (or a
diverging one) on both sides (`limit::finite_parts_agree`).

**Special values are classified, not compared** (`src/base/value_class.rs`,
`tests/unit/test_special_atoms.rs`).  Whether a value is finite, `±∞`, a
directed infinity, `zoo`, `nan` or unknown is decided by `ValueClass`, never
by comparing with `arena.infinity` and its siblings: a fold at a literal
atom matches on `ValueClass::special_atom(arena, x)` (an O(1) identity test;
the `match` says what happens at each atom, so a fold written for `±∞`
cannot fire at `zoo`); a fold that holds for every finite value of one
argument asks `ValueClass::of(arena, a).is_finite_proved()` (a number, a
named constant, an argument with symbols, or a symbol-free constant known
finite by the assumptions or with a certified value; memoised under
`FactKey::Finite`); "an infinity inside" is `value_class::holds_special_atom`,
`holds_real_infinity` (no closed form is evaluated there) or
`holds_special_atom_outside_formal` (a candidate value; the bounds of formal
integrals, sums and limits are not its infinities).  The ratchet counts the
raw comparisons left per file (identity checks that decide nothing, each
commented, and the calculus files, which migrate with their deciders).
Direct substitution in a limit never stops where the `Poles` facet of the
`funcs!` table puts a pole (`limit::tabulated_pole`): at an integer `≤ 0` (or
a value not proved nonzero) it refuses, and at a `0`/`1` place it accepts
only a value folded to a proved-finite constant or a parameter that makes
the function regular there; `B` at a pole is rewritten as a ratio of `Γ`
(`beta_poles_as_gamma`).  At `zoo` the library functions are tabulated by
their limit in every direction (`canon::at_complex_infinity`), folded when the
node is built and by `eval` alike; `eval` takes no value from a closed form
at a `zoo` argument the table does not cover.  A facet may list both kinds
of place (`Poles::NonPositiveIntegerOrZero`, since 0.59: `γ(s, z)` has
poles at `s = 0, −1, …` and at `z = 0` unless `re s > 0`); a fold at such a
place needs the condition proved (`γ(s, 0) = 0` only for `re s > 0` known:
a positive number, the assumptions, a certified value), never "not known
to fail".  A constant at a pole of its order for every argument (`γ(−n,
c)`) is no limit value.  Since 0.60 the same holds for every constant that
is a function of the table at a pole place of its facet and is not folded
to a value proved finite (`E₁(0)`, `Γ(0, 0)`, `γ(−1/2, 0)`, `K_y(0)`): it
has no value, and an expression that holds one has no limit
(`limit::holds_valueless_part`, asked at the entry of `limit_val_parts`
and by Gruntz at its entries, `refuse_valueless`).  The limit rules
multiply limits, and the canonical `0·c = 0` annihilates such a constant
before a validation of the product can see it (up to 0.59 `limit(x²·E₁(0),
x, 0)` was 0): validate the factors, not the product.  The signed
infinities the constructors fold keep their extended-real reading
(`x²·K₀(0)` is `∞`), and a numeric `B(a, b)` whose pole cancels in its `Γ`
ratio keeps its value (`B(−3, 1) = −1/3`).  What `0` and `zoo` times a
quantity that vanishes on a region are (`0/0^e`, `0/H(y)`) is under
"Piecewise parameters".

**Symbols pinned to a value.**  A symbol whose assumptions allow one value
(`zero`, `even ∧ prime`, a positive integer neither prime nor composite:
`Assumptions::pinned_value`) is that value: the assumption cache gives an
expression in such symbols the properties of its value, and the zero tests
(`factor_vanishes`, `vanishes_by_function_identity`,
`identically_zero::vanishes_by_identity`, `cancel`'s divided-out
denominators) substitute it (`canon::with_pinned_values`).  The residue test
reads a symbol as an indeterminate, so a zero test must substitute before
it.  A new way to pin a symbol goes into `pinned_value` together with its
forward-chain rules.

**Identically zero denominators.**  A quotient whose denominator vanishes
identically is `nan` in every operation.  An operation whose polynomial
arithmetic treats functions as independent unknowns (`cancel`) decides each
denominator it divides out that holds a function
(`simplify::ratsimp::cancel`): residue nonzero or certified nonzero → the
result; zero by an identity → the `1/0` arithmetic; undecided → the input
unchanged.  A new identity between generators goes into
`polybridge::generator_residue`, which must stay a ring homomorphism; that is
how the canonical `0·d⁻¹` and `s·s⁻¹` see the zero.

**Queries: two readings.**  For an expression with symbols, `real`, the
signs, `zero`, `integer`, `rational` and `complex` are about the points where
it is defined (the natural domain, the solvers' reading): `(1/x).is_real()` is
true for a real `x`.  `finite`/`infinite` in the public API (`Ex::query`,
`Ex::is_finite`, `Context::query`) are about every point — the pole detector:
`AssumptionCache::query_public` answers them with `finite_everywhere` (every
symbol finite, every constant part finite, `b^e` with `e ≥ 0` or `b` known
nonzero, every table function off its `Poles`).  Internal code keeps
`AssumptionCache::query`, whose generic reading `0·(1/x) = 0` relies on; a
public finiteness answer must not come from it.

**Refuting by a point** (`Equation::is_identity`, `eq.rs: fails_at_a_point`;
`Ex::probably_equal`, `expr_ops.rs: domain_sample`; since 0.60.1).  A point
refutes an identity only if every value is *proved* to lie in its symbol's
declared domain (all known properties of the symbol proved for the value —
never "not known to be outside"), both sides have a finite value there, and
the difference is certified (`Ex::equals == Some(false)` of two constants).
The points of `is_identity` are fixed, not random: a refutation must not
depend on a seed or on the order in which symbols were created.  A sampler
draws its points from the declared domain in the same way — a candidate
shaped by the assumptions, kept only when `solve::assumptions_hold` proves
every declared property — samples every symbol of either side, also one that
cancels in the difference, and answers `None` when no point of the domain
can be drawn.  Up to 0.60.0 `probably_equal` sampled fractions whatever the
assumptions, and `floor(v/2) + ceiling(v/2)` against `v` for an integer `v`
was `Some(false)`; `Equation::new(x, 3).is_identity()` was `None`.
`equals`' `Some(false)` means "equal nowhere"; do not read it as "not an
identity", and do not read `None` as either.

**Piecewise parameters.**  The generic convention covers analytic
functions of the symbols.  `0^y`, `δ`, `KroneckerDelta`, `H`, `⌊·⌋`, `⌈·⌉`,
and `min`/`max`/`Piecewise` with a vanishing value do vanish on regions:
`canon::may_vanish_on_a_region` (in `base` since 0.59, re-exported by
`identically_zero`) detects them, `ZeroDecision::decide`
and `vanishes_identically` read such a part as undecided (never divided by),
and `cancel` decides such a divided-out denominator.  Being nonzero at a sample point proves nothing for them.  A
Taylor coefficient never comes from differentiating a step: a
`KroneckerDelta(a, b)` of the series variable is constant near the point
only when `a − b` is a nonzero number there.  The constructors obey the
rule too (since 0.59; `0/0^e` since 0.58): `0/d` and `zoo·f` for a `d`, `f`
that may vanish on a region are `Piecewise((0 | zoo, c), (nan, True))`,
where `c` is the conjunction of one condition per such factor
(`canon::push_nonzero_conditions`): the exponent's half plane for `0^e`
(`re(e) < 0 ∨ e = 0`), `a = b` for `KroneckerDelta(a, b)`, `a = 0` for
`δ(a)`, `a ≥ 0` for `H(a)`, and `|f| > 0` for anything else — never `f ≠
0`, which `eval` decides true at a `nan` (`Ne(nan, 0)`); `|nan| > 0` has no
truth value, and a `Piecewise` that reaches such a condition is `nan`.
**Powers of the base 0** have one form, `0^e` (`canon_pow` folds `(0^e)^q`
to `0^(q·e)`), and their exponents add only within one exponent term and
one sign of its coefficient (`canon::push_zero_power_factors`: `0^y·0^(2y)`
is `0^(3y)`, `0^y·0^(−y)` is 1 at `y = 0` and `nan` elsewhere, `0^y·0^z`
stays): never merge or cancel exponents of the base 0 by hand (`powsimp`
takes `0^e` for its own base), and never treat a constant `0^e` as a finite
nonzero factor of a limit (`limit::limit_by_zero_power_cases` takes the
three cases of `re e` for the public limit; the internal generic limit
still reads the power as a generic factor — a known gap: new code that
calls it refuses an expression with a constant `0^e` first).  A
factor `0^e` under a 0 coefficient keeps the generic `0·0^e = 0` unless a
term of `e` has a negative coefficient (`reciprocal_zero_power`).  A sum
of step parts vanishes on a region when it does at a value one of them
takes on a region (`max(c, y)` at `c` and at `y`, `H(y)` at 0 and 1):
`calculus_util::vanishes_at_a_step_value` makes such a constant undecided
(`8·max(0, y)³ − 4·max(0, y)`, `H(y) − 1`; since 0.59).  A
`KroneckerDelta(a, b)` holding the limit variable has its value at the
point as its limit only where `a − b` is a nonzero number there
(`limit::substitute_continuous`); Gruntz rewrites it to 0 where `a − b` has
an eventual sign, and refuses where that leaves the expression at a pole.
Since 0.60 that check follows every rewriting of a step by its value near
the point — `H`, `sign`, `⌊·⌋`, `⌈·⌉`, `min`, `max`, `Piecewise`, `δ`,
`KroneckerDelta`; in Gruntz's `rewrite_tractable`, in `eval` after the
positive variable is substituted, in `refine` under `x > 0` — because each
can put a function at a pole place: ask `limit::holds_valueless_part` on
the rewritten expression, not only for the delta (up to 0.59
`limit(x²·E₁(KroneckerDelta(x + 1, −1)), x, 0, '+')` was 0 while the
two-sided limit was refused).

**Hidden zeros inside functions of parameters.**  `ZeroDecision::decide`
also decides the symbol-free pieces of a constant with parameters (`J` in
`sin(y·J)`): a piece proved 0 is replaced by 0, and a piece 0 only to the
precision reached counts as nonzero only if the constant is nonzero both
with and without it.  A step function (`H`, `sign`) at an argument nonzero
for generic parameters is read as continuous only with its exact condition
attached (`Piecewise((H(y), y ≠ 0))`).

**Divisors carry their proof** (`calculus_util::NonzeroExpr`,
`tests/unit/test_divisors.rs`).  The series, limit and Gruntz engines divide
only by a `NonzeroExpr`, which only the deciders build:
`ZeroDecision::decide` (the one zero test of a coefficient, a leading term,
a derivative value or a direction: the structural test, `hidden_zero` for a
constant, a piecewise part undecided, the parameter parts, the identity test
of `vanishes_identically`), `NonzeroExpr::number`, `structural` (a product of
powers of numbers and the named constants) or `variable` (the dummy of `x =
a ± 1/t`); the quotient is built with `div_by` or `recip`.  An undecided
divisor ends the rule (`series::rule_divisor`: refused at once), never a
division.  The ratchet counts the `.div(…)` and `pow(…, −1)` left in the
three files (quotients of functions whose limit is taken next, and
identities, each commented).  Gruntz's variable stands for `−ln ω` after a
move up: every expansion in `ω`, ramified or not, writes it through the
engine's `l = ln ω` (`variable_as_log_omega`), and the engine never takes an
exponential or power of `l` for a constant (`series::hides_var_power`).

**Binomial coefficients.**  `C(n, k) = 0` for a negative integer `k` and
every `n` (Graham–Knuth–Patashnik, SymPy).  A rewrite may use Pascal's rule
for all integers, the symmetry `C(n, k) = C(n, n − k)` only for `n ≥ 0`;
reading `C(k + c, c)` as `C(k + c, k)` needs `c` not a negative integer
(`summation::negative_integer_shift`: the identity under `c ≥ 0`, 0
otherwise, or a refusal).

**Sums with parameters.**  A sum whose convergence depends on a parameter
the assumptions do not decide is `SumOutcome::Conditional { value, region }`
(the closed form and the exact region, or a sufficient one documented as
such), never `Closed`: the public `summation` writes `Piecewise((value,
region), (Sum, True))`; `eval` of a formal `Sum` does not build it.  The
numeric value of a sum is never truncated on what its terms look like: the
rest beyond the last term summed has a proved bound.  For a weighted sum
over a long range (`evalf/windowsum.rs`, since 0.59: the entropy of
`Binomial(10⁵, p)`) that is an envelope `2^(a + b·j)·(1 + j)^p` of the
summand built bottom-up, in which a hypergeometric weight's rate `2^b`
bounds its step ratio on the whole rest: prove it from the ratio
polynomials (no real root of `P·Q`, and of `ρ·|Q| − c·|P|`, between the edge
and the end: `poly::sturm`), never from "the masses are log-concave" or
from the terms seen.  An `f64` estimate may choose where the window starts
and refuse a window that cannot fit the budget (`MAX_WINDOW_TERMS`,
65,536); it never ends a sum.  A weight moved by its recurrence stops at an
integer root of `P` or `Q` (a pole of a Gamma, a binomial beyond its
support).

When only coprimality or the degree of a gcd is needed, use modular images
(`modgcd::main_gcd_degree_bound`, `coprime_certified`), not a full gcd: an
image gcd of degree 0 proves coprimality (`Exact`), a positive image degree
is `Probabilistic` and may only justify a refusal.

**The constant-facts memo.**  `Arena::constant_facts` is keyed by
`(ExprId, FactKey)` and stores the `SignSet` a decider proved (`UNKNOWN` when
nothing).  Each decider owns its `FactKey` variants; add one for a new
question and never read another decider's key (up to 0.48 two deciders used
colliding hand-picked `u8` keys, and `is_positive` answered according to
what had been computed before).

**A function's name is written once** (`src/base/func.rs`,
`tests/unit/test_func_names.rs`; since 0.60).  How a function is written
and ordered — its name, the names the parser reads, its LaTeX layout, its
layout in the two-dimensional printer, its MathML head, the leading bytes
of its sort key — is data of its `funcs!` entry, and the code that needs it
reads the entry: `display.rs`, `latex.rs`, `pretty.rs`, `mathml.rs` and
`sort_key.rs` have one arm `func_nodes!() =>` for all function nodes (a
second arm for a function node is an unreachable pattern, a warning that
CI's `-D warnings` refuses), and the parser asks `Func::parsed`.  Up to
0.59 each of them had an arm per function (45 in each of `display.rs`,
`latex.rs`, `mathml.rs` and `sort_key.rs`, 36 in `pretty.rs`, 42 in the
parser) and the parser a list of 94 known names, which tests over
`Func::ALL` kept equal to the table; the two-dimensional printer spelled
four functions its own way (`lgamma`, `sgn`, `psi`, `ceil`: kept, and now
visible in the `pretty` facet beside the name).  The ratchet parses those
six files with `syn` and counts the string literals (in expressions,
patterns and macro arguments) that equal a function's name, its name in
lower case or one of its parse names; the allowlist names each literal
that stays and why (Greek letters of symbol names, `log(x, b)`: 14, down
from 274), and may only shrink.  A notation that is not data is a case
keyed by `Func` inside the one arm, commented; a fact the table lacks
becomes a facet (every entry is a struct literal written in full, so the
compiler lists the entries to extend), not a second list.

**Identifiers have no order.**  `ExprId`, `SymbolId` and `NumId` are handles:
equality and hashing only.  Their numbers are allocation order, so sorting by
them made results depend on the order in which symbols were created
(`factor_all(x³ − y³)` with `y` created first).  An order that can reach
output: `arena.canonical_cmp(a, b)` (for symbols, name order).  A fixed
order that never reaches output (dedup, a sorted list used as a set or a
key): `ArenaOrder(id)`, documented as arbitrary.  Hashing and memo keys:
`id.index()`.  `tests/unit/test_id_order.rs` counts `index()` calls and
`ArenaOrder` constructions per file and rejects `index()` inside ordering
closures.  A hash that chooses something that can reach output (a sample
point, an RNG seed) hashes structure — the sort key, a symbol's name — never
`index()`: up to 0.52 `Ex::equals`' sampling depended on the order in which
symbols were created.

**A bound variable is another variable** (`src/calculus/transform_util.rs`;
since 0.60).  The integral transforms (`calculus/{laplace, z_transform,
fourier_transform, mellin}.rs`) keep their shared helpers in
`transform_util.rs` and no copies: `depends_on`, `split_factors` /
`split_independent`, `linear_in` (`a·v + b`), `extract_linear_coeff`, the
entry checks `require_symbol` / `symbol_id`, `param_sign`, `need_sign`,
`fail`.  An occurrence of the transform variable that a binder has bound
(the index of a `Sum`, the variable of a definite `Integral`, of a `Limit`,
of a `RootOf`: `walk::binder`) is another variable: (1) a transform asks
`depends_on` (free occurrences) about its variable, never `walk::contains`;
(2) a pass that rewrites the function being transformed into something
equal only as a function of that variable — `Piecewise` into steps (equal
almost everywhere), a step read on `t > 0` — walks `scope_post_order` and
rebuilds with `rebuild_in_scope`, a search through sub-expressions pushes
`for_each_child_in_scope`, and a test "is there a step in it?" is
`any_in_scope`: none of them enters the operands a binder of the variable
scopes over.  Up to 0.59 the four copies had drifted (Laplace and Z asked
for free occurrences, Fourier and Mellin for any) and every rewriting pass
walked into binders: `laplace(Sum(Piecewise(1 if t < 2, 0 if True), t, 1,
3)*exp(-2*t), t, s)` was `3/(2(s + 2))`, with the sum, which is 1, taken
for 3/2.  `the_transform_modules_share_their_helpers`
(`tests/v36/v36_s17a_transforms.rs`) fails when one of the four modules
defines a helper of that list or gains a call of `walk::contains` (the
four that are left, in `fourier_transform.rs`, look for the imaginary
unit).

**An internal symbol is not a user's symbol** (`Arena::internal_symbol`,
`Arena::positive_symbol`, `Arena::fresh_internal_symbol`; since 0.60.1).  A
dummy an engine makes for itself — the variable of a substitution, a
placeholder, a symbol declared positive for one computation — is made with
`arena.internal_symbol("__name")` (or `positive_symbol`), never with
`arena.symbol("__name")`: the symbol table keeps such symbols in a name
space of its own (`SymbolTable::intern_reserved`), which `Context::symbol`,
the parser and the JSON reader do not read, so the dummy cannot be a symbol
of the expression the user passed in (no capture) and an assumption declared
on it is declared on nothing of the user's.  Two symbols with one name would
have one sort key; `internal_symbol` therefore takes `name_1`, `name_2`, …
where the user's name space has `name`.  A name is converted at all its
sites at once (the dummy is found again by asking for the same name), and a
name that is positive is asked for through `positive_symbol` only.  A test
"is this one of ours?" is `arena.is_internal_symbol(sid)`, not a prefix of
the name.  `compact::Transfer` keeps the name space across arenas.  A
stand-in that an engine *declares* (real, integer, the unknown's
assumptions) and may need again while an outer call holds one is
`arena.fresh_internal_symbol(name, assumptions, avoid)`: declared once, when
made.  A name that is part of a result — the bound variable of a `RootSum` —
stays in the user's name space and is chosen with `subs::fresh_symbol(arena,
name, avoid)`, so that the printed form reads back.  Up to 0.60.0 a limit at
a finite point declared the user's symbol `__lim_r` positive, `((__lim_w*x +
sin(x))/x).limit(x, 0)` was `oo`, and `solve(sqrt(x) + x^(1/3) -
__t_radical, x)` returned three wrong roots.  60 `arena.symbol("__…")` sites
of the integrator, `solve`, the transforms and the Risch code are not
converted yet (the 0.60.1 audit of 136 names in 68 operations found no
capture among them, which shows only what those operations reach); convert a
name when you touch its engine.

**A handle wraps an expression of its sort** (`Expr<S>::args`,
`src/api/expr.rs`; since 0.60.1).  A method of `Expr<S>` that hands out
sub-expressions wraps them in the type they have, not in the type of the
parent: `args()` returns, for a `BoolEx` or a `SetEx`, the operands that are
of that sort by their position (`And`/`Or`/`Not`; union, intersection,
complement) and nothing for a node whose children are numbers (a relation,
an interval).  `Ex` is the untyped handle (`as_ex`).  Up to 0.60.0 `(x >
1).args()[0].not()` built `!x`.

**Checked casts** (`src/base/cast.rs`).  Do not convert an `f64` to an
integer with `as` after `floor`/`ceil`/`round`/`trunc`: the cast saturates
(`1e19 as i64 == i64::MAX`, `NaN as i64 == 0`) and the `± 1` that usually
follows wraps (up to 0.48 `∫₋₁¹ floor(10¹⁹x)/10¹⁹` was 0).  Use
`cast::f64_to_i64` (exact or `None`) or `cast::integer_span` (a float range
widened to integers, with a length cap).  `tests/unit/test_float_casts.rs`
counts these casts and the `(f64 expr) as usize + …` guard-bit pattern per
file.  Differences of `i64` exponents and scales are `saturating_*` or taken
in `f64`.

**Exact arithmetic has size bounds.**  An exact power, or any intermediate
that grows with an exponent, is formed only below a bit bound
(`algebraic::MAX_POWER_BITS`, 2¹⁶; `MAX_POWER_DEGREE` for minimal
polynomials of powers), with checked conversions (`u32::try_from`, never
`as`); beyond the bound the decider returns `None`.  Up to 0.51
`rational_pow` cast the exponent `as u32`: `2^(2³² + 1)` became `2`, so
`2^4294967297 − 2` was called zero, and `1000000007^1000000007 − 2` hung.
Since 0.59 an exact value asked for by the caller is bounded before it is
formed, by a lower bound of its size: `(bits − 1)·degree` for a power
(`MultiPoly::eval`, `try_pow`: `MAX_VALUE_BITS`, 2²⁷), `n + 1` terms for
the `n`-th power of a polynomial of two or more terms (2¹⁶).  Only the
powers of the exponents present are formed (`zpoly::SparsePowers`; a table
of every power up to the degree asked for 68 GB for `x^(2³¹) + 1`), and a
huge value is reduced by a division, not by `Ratio::new` (a binary gcd,
quadratic in the larger number even against 1).

**Allocation lengths from the data** (`src/base/cast.rs`,
`tests/unit/test_alloc_bounds.rs`).  A length read from the input — a
degree, an exponent, a valuation, a pole order, a number of terms — that
sizes a `vec![x; n]`, a `with_capacity(n)`, a `resize(n)` or a loop that
pushes `n` items is bounded first: `cast::bounded_len(n, cap)` (from any
integer type, `&BigInt` or an integral `&Q`; `None` when negative or beyond
the cap), and the routine refuses beyond (an error, `None`, the unevaluated
form); `cast::clamped_len` only where the caller lowers its own bookkeeping
or the length is a capacity hint for a loop bounded elsewhere.  An
allocation failure aborts the process and nothing can catch it: up to 0.57
`x^(9·10¹⁰)`'s dense coefficients (360 GB), the series of `exp(−x^N)`
(137 GB), the pole order of `z/(z − 1)^N` and the harmonics of a Fourier
series did.  The shared caps live in `cast` (`MAX_DENSE_DEGREE` 2¹⁶,
`MAX_SERIES_TERMS` 2¹⁴, `MAX_TERMS` 2²⁰, `MAX_DENSE_LIST` 2²⁴,
`MAX_POWER_BITS`); a routine with a tighter work bound passes its own.  A
public function whose result is a list as long as its input
(`primes_up_to`) has a `try_` twin that reports the refusal.  Since 0.59
the same holds for a count the caller passes: a public function that
returns a list as long as that count (`FormalPowerSeries::coefficients`,
`discrete::convolution_cyclic`) bounds it the same way and has a `try_`
twin; its plain form returns the documented empty list, never a capped
prefix that takes as long as the cap (the first 2¹⁴ coefficients of `sin`
take 20 s).  A function that returns one number whose size grows with an
index (`ntheory::fibonacci`, `lucas`, `primorial`) has a cap on the index,
returns the documented `0` beyond it and has a `try_` twin (up to 0.58
"the caller bounds `n`").  The refusal is documented under `# Refusal` on
the plain function.  A length
computed into a `let` first is not seen by the ratchet: review it.

**The BigFloat boundary** (`src/base/bigfloat.rs`).  astro-float's
`BigFloat::pow` and `BigFloat::log` never return when the exact result is
representable (`4^(1/2)`, `log(8, 2)`; upstream #31) and are disallowed by
`clippy.toml` together with `convert_to_radix`; use `bigfloat::pow`,
`bigfloat::to_decimal_digits`, two `ln`s for a logarithm, and
`bigfloat::to_f64` (finite or `None`) for conversions.  astro-float's
`Up`/`Down`/`ToZero`/`FromZero` are directed roundings (its documentation
says "round half"), `None` truncates, and `BigFloat::from_f64` halves
subnormal inputs.  `probes/clippy_canary` calls each disallowed method (`pow`, `log`,
`convert_to_radix`, `is_positive`) and CI expects clippy to reject every call (a path that stops resolving after an
astro-float release is noticed).  Two more rules since 0.60.1 (issue #8:
`π` to ten million digits took 411 s): **mantissa words** become an
integer, and an integer becomes words, only through
`bigfloat::mantissa_to_biguint` and `bigfloat::words_from_u32_digits`
(linear; an accumulator shifted by a word per word is quadratic, 3.17 s
against 0.7 ms at three million digits); and **the constants cache**
(`π`, `e`, `ln 2`, `ln 10`) of an evaluation comes from
`bigfloat::with_consts(bits, …)`, never from a `Consts::new()` per call,
which recomputes every constant.  `with_consts` keeps one cache per thread
and per band of working precision (astro-float answers a request by
copying the most precise value it holds, so a cache far warmer than the
request is slow too), and its results do not depend on what was evaluated
before — for every rounding mode except `RoundingMode::None`, which
truncates whatever approximation the cache has: code that evaluates in
that mode (`poly::algebraic`, `poly::roots`) keeps a cache of its own.

**Solving** (`src/transforms/solve/`).  The domain of the unknown is a
`SolveCx` (`Domain::{Complex, Real, Integer}`, intersected with the
unknown's declared assumptions), threaded explicitly — never a thread-local
or a parameter whose presence switches the domain.  An inversion arm asks
`cx.range_restricted(f, var)` before using a range restriction.  Solvers
return `Solutions` (`Complete(RootSet)`, `Partial(RootSet, Missing)`,
`Refused(Reason)`; `RootSet` is `Empty`, `All`, `Points` or `Families`) and
merge branches with `union`, which decides once that a refused branch next
to an empty one is a refusal, not a proof.  There is no `is_empty` and no
conversion that collapses "no solution", "every value" and "refused" into an
empty list; callers pick a named policy (`points_if_complete()`,
`known_real_roots()`).  An inversion that takes a principal branch (`f^(p/q)
= c → f = c^(q/p)`, a root `t₀` of `t = x^(1/L)`, `ln`, `asin`, `acos`,
`atan`, the inverse hyperbolics, `W₀`) claims the root only when `c` (or
`t₀`) lies in that branch's range: for a value with parameters the
assumptions decide where they can (`principal_power_preimage`,
`peel_principal_inverse`, from the exact argument range, no evaluation).
Otherwise the public solvers get the root with its exact condition on the
parameters (`Solutions::under` → `Reason::Conditional(Vec<Guarded>)`):
`Ex::solve` refuses, `Ex::solve_as_set` writes `{root} ∩ ConditionSet(x,
cond)`, and `known_real_roots` keeps it as a candidate
(`SolveCx::as_candidates`); where no exact condition exists, the solve is
refused.  A conditional outcome is a refusal for every consumer of lists;
`Missing::merged` keeps an unknown alternative over a conditional one, and
failing principal candidates never prove "no solution" while other branches
exist.  A root of a declared domain is returned only when its membership is
decided or can fail only on a set of measure zero
(`in_domain_only_for_some_parameters`; an exact realness condition such as
`A + C·u^(p/q)` real iff `u ≥ 0` becomes a conditional set).  The constant
factors of a product (symbol-free, or holding an infinity) are decided
first: one proved 0 makes every value a root unless another one has no
value or an undecided zero test (refused); all proved nonzero, the other
factors' roots are the roots.  The solvers substitute the pinned values of
the parameters before solving (`solve::with_pinned_parameters`).  A root,
or the constant of an equation free of the unknown, that holds `±∞`, `zoo`
or `nan` inside a function argument
(`value_class::holds_special_atom_outside_formal`, asked by
`solve::not_known_finite`; not the bounds of formal integrals, sums and
limits) is claimed only when known finite.  A
family not linear in `n` is located in a window by pulling the window back
through a chain of monotone steps (`sets::monotone_members_in`); a step real
only on part of the line (`ln`, `t^q`) may be followed only by steps that
keep non-real values non-real.  A root returned for a real unknown must be proved real; a constant discriminant or leading coefficient is decided by
`constant_zero_decision`, never by how it looks.  A root is a point of the
natural domain: every part of the equation is defined there.  The
evaluator absorbs some poles (`1/zoo = 0`, `zoo^(−k) = 0`), so the check of
a candidate (`solve::residual_at`) also asks whether a part is at a pole
(`part_at_a_pole`: `tan u` at `cos u = 0`, `tanh u` at `cosh u = 0`, a
negative power of a vanishing part, the argument of `W_k`, `k ≠ 0`, at 0;
`acot`, `acoth`, written `atan(1/z)`, `atanh(1/z)`, are defined at 0).  A
candidate whose residual does not evaluate is kept only at a certified
point (a branch point such as `acos` at 1); one whose own value is not
certified is not claimed (the outcome is `Partial`).  An equation `c = 0`
for a constant `c` is "no solution" only when `c` is proved nonzero
(certified digits, a minimal polynomial, or what `Ex::is_zero` decides);
otherwise it is refused.  `e^u`, `a^u`, `sinh`, `cosh`, `tanh` of an
argument not known real are periodic on ℝ: the crate's own root searches
(integration, `singularities`) get their principal roots as `Partial`
(`SolveCx::lists_every_real_root`), callers listing zeros in a window ask
`solve::has_periodic_exponential`, and for a real unknown and a linear
exponent the real roots on every branch are exact
(`real_roots_on_every_branch`); `Ex::solve` keeps the principal roots.  A Lambert-W family of an ODE carries its `W₋₁`
twin, and `check_ode_solution` rewrites `e^{W(z)} = z/W(z)`.  A fitted
initial-value solution needs a proof: its residual at the initial points
proved 0 (`residual_is_zero`: exact tests, `solve::lambert_values`,
certified values) or the candidate solving the equation identically
(`check_ode_solution`); an undecided candidate is refused, never taken for
lack of a better one (Lagrange's formula for linear constant-coefficient
equations is exact by construction and is rejected only on a certified
nonzero residual).  A sector condition `arg y ∈ (a, b]` is written `a <
arg(y) ∧ arg(y) ≤ b`; `Solutions::under` simplifies conditions.  A family
`a + b·n` with a non-real step has at most one real member, at `n = −im
a/im b`, decided exactly (`solve::real_members_of_family`, since 0.59); a
consumer that lists the real members of a family in a window never drops
it because its offset is not real (up to 0.58 `singularities(1/((−2)^x −
4), x, [0, 10])` was empty).  A polynomial in `c^x` for any constant base
`c ≠ 0` is solved in `t = c^x` when every exponential has that base.
A list of roots is complete or says it is not (since 0.60;
`domains::matrix::Spectrum`,
`tests/unit/test_matrix_linalg.rs::partial_root_lists_are_not_read_as_complete`).
The eigenvalues of a matrix travel inside the crate as a `Spectrum` —
`Complete` when the multiplicities sum to `n` (checked by `Spectrum::of`; a
`Spectrum` is built only in `domains/matrix/eigen.rs`), `Partial { found,
missing }` otherwise — which has no `len` and no iterator: a consumer takes
`all(operation)` (every eigenvalue or `ComputationFailed`, naming the ones
found) or matches `Partial` and says why a part is enough
(`StateSpace::is_stable`: one pole proved in the closed right half-plane
decides "not stable"; nothing is ever proved from the poles that are
missing).  A public function returns all the roots or an error / its
documented refusal value (`StateSpace::poles`: none), never a shorter list
that looks complete; up to 0.59 `eigenvals` did, behind a `warn!`.  Do not
call `solve_or_empty` in library code (it turns every refusal into `[]`):
call `Ex::solve` and treat an error, or a number of roots other than the
degree of an irreducible factor, as "this factor's roots are missing".  The
ratchet counts per file the four ways a possibly incomplete list is still
read (`known_real_roots`, `Solutions::claimed_points`, `solve_or_empty`,
`Spectrum::Partial`); the counts are exact and may only shrink.

**Logging is lazy.**  A debug line formats an expression through the macro
(`tracing::debug!(e = %arena.display(e))`), never by building a string
first: up to 0.57 Gruntz spent 79 % of a fuzz input's time formatting lines
that were switched off.

**Budgets count work.**  An engine's limit counts deterministic work — a
`Work` charged per step, or a `WorkScope` that counts interns attempted —
never wall-clock time (only a caller's `Budget` deadline may be one) and
never `node_count()` growth, which depends on what the context already holds
(up to 0.48 a repeated `series` call got slower each time).  A budget that
surfaces as an error is `SymplexError::BudgetExhausted { kind }`; add a
`BudgetKind` per new budget.  Gruntz charges its zero tests to its budget
(`EvaluationScope`), an engine's `expand` is charged by the terms it
distributes (`expand::expansion_terms_bound`, refused beyond its cap), and a
series that needs more than `MAX_WORKING_ORDER` terms is refused at once; a
tree-size bound does not bound an expansion.  An expansion that widens its window does not
repeat a differentiation fallback that refused at a lower order; a special
function whose derivative in a parameter divides by it at the point
(`Π(n | m)` at `n = 0`) gets a structural rule rather than the fallback.
Since 0.59: every `expand` of `gruntz::leadterm_add_by_series` is charged
with `charge_expansion`; a series fallback that needs a derivative asks
first whether the first derivative is formal (`B(a, b)`), and then takes no
limit and makes no pole retry; a loop whose length is a difference of
degrees or a multiplicity is counted and refused beyond a bound
(`MultiPoly::prem_in`: `MAX_PREM_STEPS`); and a verification never repeats
a product once per unit of a multiplicity: monomial factors add exponents,
the others are raised by `try_pow` (`factor_multivariate`), a power of `x`
is a shift (`factor_zassenhaus`).  Since 0.60: a zero test charges its
`EvaluationScope` whether a memo, an enclosure or the ladder answers it,
and a new fast path of `hidden_zero` goes behind the charge.  The charges
are the work accounting of the series engine's compositions: an attempt
that postponed the zero tests removed them, and a 28-term composition was
multiplied out to its end before the budget could refuse it.  An engine
that makes fewer tests charges the work it does instead (`charge_size`,
`charge_expansion`), or the budget no longer bounds it.
Since 0.60.1 a Gröbner basis is budgeted (`poly/groebner.rs`:
`GROEBNER_WORK_CAP`, `BudgetKind::GroebnerBasis`).  The loop that does the
work is charged where it runs: the division of `MultiPoly::reduce` takes the
`Work` (`reduce_charged`; the public `reduce` runs the same loop on a budget
of 2⁶⁴), and Buchberger's rounds, the S-polynomials and FGLM's dense linear
algebra charge the same `Work`.  The units are a cost model fitted to
measured times (`multipoly.rs`, "Work units of a basis computation"): a term
operation, plus the growth with the coefficients' size in words, which is
read from the operands before the operation — never a clock.  A function
that returns a plain list (`groebner_basis`) returns its input, documented
under `# Refusal`, and has a `try_` twin; no caller inside the crate uses
the plain form, and a caller that certifies completeness with a second basis
(`zerodim::Radical::try_new`) propagates the refusal instead of going on
without the certificate.  Up to 0.60.0 Buchberger's loop had no bound, and
`Ex::groebner` of the cyclic system in six variables did not return.  Two
more budgets of 0.60.1 follow the rule without a `BudgetKind` of their own,
which would be a public item: the zero tests of a matrix call (`PivotWork`,
see "Pivots": spent means `Unknown`) and the root search of a quantile (see
"Degenerate data": spent means the documented `ComputationFailed`).

**Finite sums for every integer bound.**  Reversed limits follow Karr's
convention (`Σ_{a}^{b} = −Σ_{b+1}^{a−1}` for `b < a − 1`; the empty sum for
`b = a − 1`) in `summation`, `eval` and `evalf` alike.  A closed form for a
symbolic upper limit must hold for every integer value the assumptions
allow where the sum is defined; each rule says where it holds
(`summation::Holds`): an antidifference holds everywhere; a rule proved only
for `hi ≥ h` (the identities with `C(n, k)`: `n ≥ 0`) says so, and
`finite_sum` decides `hi ≥ h` from the assumptions or returns
`SumOutcome::Conditional`; Gosper's antidifference is checked at the empty
sum, at the first term, and below the lower limit term by term
(`gosper_closed`, `zero_below`).  A closed form undefined as a whole is never
returned for symbolic limits.  `convergence` decides a symbolic geometric
base with `summation::modulus_vs_one`, so `is_convergent` agrees with the
regions `summation` returns.  Products follow Karr's convention like sums
(`Π_a^b = 1/Π_{b+1}^{a−1}` for `b < a − 1`): `product_closed` returns a
`FiniteClosed` (Gamma ratios and rising factorials hold everywhere; a
non-integer power holds while the factors below the lower limit keep their
sign, `sign_run_below`); a rewrite that is an identity of meromorphic
functions but not of their values at the poles (Legendre duplication with
factorials) is used only where the assumptions exclude the poles.  A public
entry point to an algorithm that `summation` checks goes through the checked
path (`gosper_sum_checked`).

**Per-arena state.**  Memos keyed by an arena use `Arena::serial()`, never
its address; zero verdicts of constants are memoised under
`FactKey::HiddenZero` (proofs only, never "no evidence").  An engine budget
that must live in a thread-local (a trait without context, such as `PFrac`
arithmetic) holds `Work` values, set afresh at the outermost entry.  Rule sets built on every call use fixed wilds
(`arena.wild_at(k)`), so repeated calls create no nodes.

**Solvers and parameters.**  A public solver uses `Domain::Complex` (or
`Domain::declared(var)`) with `Params::AsDeclared`: a symbol without
assumptions may be complex, as an unknown and as a parameter.  The crate's
own root searches (sign charts, singularities, integration) use
`Domain::Real` with `Params::Real`.  A range restriction ("no solution
because `f` is real") needs `cx.range_restricted(f, var)` for the very
expression it reasons about, never just the realness of the unknown; a
general solve adds `2nπi` wherever an argument is not certainly real.

**Numerical evaluation (`src/transforms/evalf/`).**  A routine that needs
the working precision or the constants cache takes `w: &mut Wp<'_>` (`&mut
w.at(bits)` for another precision); there is no rounding-mode parameter —
use `RM` (round to nearest even) and write a directed mode out literally
where a bound needs it.  A precision raised by an `f64` estimate goes
through `Prec::guarded(Guard::for_log2(..))` (saturating; no `as usize +
…`).
The working precision of a digit count comes from `evalf::working_bits`
(`u64`, since 0.60.1; `digits·34/10 + 64` in `usize` overflows on a 32-bit
target from 126 million digits on, and a release build wrapped to 128 bits
and passed the limit check), and an entry point of the evaluator validates
the count with `evalf::checked_precision` (zero digits are
`InvalidArgument`, a precision beyond the configured maximum or an eighth of
`usize` is `PrecisionExhausted`): never repeat the formula (up to 0.60.0 it
stood in five places, and `eval_decimal(0)` printed 38 uncertified digits).
A value with its error bound is a `CBall` (`Ball<&Complex>` for a
cached value; a real value with a log₂ radius an `RBall`), never a
`(Complex, Bound)` or `(BigFloat, f64)` pair; evaluated nodes live in one
map of balls (`ball::Balls`) read through the view `ball::Cache`, and a
node's bound rule is a `Ball` method returning `NodeBound::{Final,
Propagated}`; read signs and branch-cut
sides from the ball (`sign_re`, `sign_im`, `cut_side`), not from the
midpoint.  Use `bigfloat::is_strictly_positive`/`is_sign_positive`
(`BigFloat::is_positive` is true for `+0` and is disallowed),
`bigfloat::from_f64` for a computed `f64` (astro-float's halves
subnormals), and `bigfloat::to_f64`/`to_f64_or_inf` as the only conversions
to `f64`.  Decimal output is our exact conversion
(`bigfloat::to_decimal_digits`), not astro-float's `convert_to_radix`.
A route's own rounding belongs in its bound: exact inputs make a node's
*propagated* error 0, not its computation (`b^x` formed as `exp(x·Log b)`
rounds `Log b`; `accuracy::pow_rounding`), and a part is exact only where
the route provably yields an exact 0 (up to 0.51 the rounding of `8π` made
`(−1)^(8 + 2i)` certified non-real).
Since 0.60: a bound read off the result's own size is not a bound for a sum
that cancels.  A real route returns a bare `BigFloat` and the node's bound
gives it two ulps of that value (`accuracy::finish`): true of a value formed
by correctly rounded operations and of a sum of one sign, false of an
alternating series, of `cos ω·P − sin ω·Q`, of `γ + ln x + Σ`, of the end of
a three-term recurrence — formed at `wp` bits from terms of size `2^t` they
are known to `2^(t − wp)` whatever they cancel to, and next to a zero of the
function that is anything (up to 0.59 `besselj`, `bessely`, `Ci`, `Chi`, the
Airy functions, `li`, `besseli` of a negative order and `legendre` of a
large degree printed uncertified trailing digits at a dyadic argument next
to a zero).  Such a route states the absolute error of what it summed and
charges it to the `Wp` it was called with: the terms to a `wp::Summed`
(`term`, `of`, `steps`), the values of a recurrence to a `wp::Recurred`
(`step` with the growth of the family's norm), then `w.charge(seen.error(wp,
scale))` immediately before it returns; `sensitivity::special_error` adds
the charge to the node's propagated error (`cspecial::bound` on the cut),
and `evaluate_adaptive` raises the precision until the digits are certified,
or refuses.  `Wp::at` drops a charge (the value is an ingredient: its user
says what it does to the error), `Wp::carry` keeps it (the value is returned
as it is).  Do not detect zeros, and do not derive the bound from the
result; a route that measures its loss and repeats with the bits made up
(`arb_ei`, `arb_log_gamma`, `arb_jacobi`) needs no charge.
Since 0.60.1 the logarithm of `Γ` is the model of the second kind:
`stirling_log_gamma` charges the absolute error of its sum (`Summed` over
its terms, the first term left out for the tail), and a caller that takes
`exp` of it asks `gamma::log_gamma_abs(x, bits, w)`, which reads that charge
from a sub-`Wp` (`take_own`) and repeats with the missing bits — the first
precision comes from an `f64` estimate, the guarantee from the charge.
**The absolute error of a logarithm is the relative error of its
exponential**: a route that forms `exp(L)` states the absolute error of `L`
(not "`wp` bits of `L`"), with the bits of `abs(L)` on top of its precision.
Up to 0.60 `Γ(x)` took `ln Γ` with 20 guard bits whatever its size and was
`2^(log₂ ln Γ − 21)` times less accurate than its bound from `x ≈ 3·10⁵` on
(600 times at `7·10⁷`); the digits of `Γ(x)` were right and a cancellation
of 50 to 62 bits printed wrong ones.  Hunt such a bound with `f(x) − M·2^E`,
`M` the leading `k` bits of the value, for every `k` from 40 to 75 and every
digit count from 16 to 80: the claim fails only where the cancellation eats
exactly the slack of the working precision (32 digits and `k ≈ 57` for `Γ`),
and direct values never show it.
A zero that the
structure gives exactly keeps a bound of the value's size
(`orthopoly::odd_member_scale`: an odd polynomial next to 0).  A family is
only in this class when every argument is exactly representable: test with
dyadic orders and parameters (a parameter `1/3` carries its own rounding and
hides the route's).  Exact tables that do not depend on
the argument (Borwein's `d_k`, the coefficients of `W` at its branch point)
are built once per thread and on demand (`borwein_d`,
`lambertw::BranchPointCoefficients`; since 0.59), in integer arithmetic
where they are integers: a rational recurrence with a gcd of `wp`-bit
integers per step was 45 % of a `ζ`-heavy evaluation.  A function with a
zero at a rational point takes a rational literal argument from its exact
value (`zeta_next_to_trivial_zero`, as `sin` of a rational reduces it
again): the float argument's rounding is a radius far above a value next
to the zero.  An identity between two terms of a sum (`erf(u) + erfc(u) =
1`) is applied to the sum (`erf_complements`), not left to two rounded
values whose sum is a ball around the constant.  The elliptic routes
pursue a part far below the other as far as the arguments' exponents
explain it, within `MAX_GAP_WORK_BITS` (`elliptic::gap_pursued`).

**Coefficient domains are values.**  A new coefficient ring implements
`poly::domain::CoeffDomain` (`&self` carries its parameters — a modulus, a
minimal polynomial; `inv` returns `None` for non-units) and uses
`DensePoly<D>`: do not hand-write gcd, xgcd, resultant or square-free
decomposition again, and do not put the domain's context into every
element.  A faster method a domain overrides (`dense_gcd`, …) must return
what the generic algorithm returns.

**Integer gcds** go through `base::numeric::gcd_big` (Lehmer), `gcd_u64` or
`gcd_u128`; do not add another.  A graded `MonomialOrd` overrides `cmp_monomials` to use
the stored `Monomial::total_degree`.  Dense multivariate gcds with large
coefficients go to Brown's modular algorithm (`poly/brown.rs`), verified by
trial division.

**Degenerate data** — a rank-deficient design, (quasi-)complete separation,
a constant response, a variance a statistic divides by being zero, no
events — is `SymplexError::DegenerateData { kind: Degeneracy, .. }`, decided
exactly by the exact methods and by the documented numerical threshold in
the `f64` fits; shape and range errors stay `InvalidArgument`, an iteration
that fails on valid data `ComputationFailed`.  An `f64` value derived from an
exact variance is rooted from the rational (`regression::sqrt_to_f64`): a
variance is the square of the data's scale and leaves the range first.
Discrete quantiles are decided on exact masses; an `f64` walk or a lattice
kernel's answer within `2⁻³⁰` of the level only guesses.  A lattice family
whose masses are rationals with a term recurrence forms its tails from the
integer terms over one denominator (`discrete::binomial_mass`,
`hypergeometric_mass`), bounded in bits and in steps, never as the symbolic
sum of the masses (each a quotient of binomials evaluated from scratch, and
a quantile search probes the tail some twenty times); beyond the bounds the
sum stays symbolic.  Whether a numeric point lies in a far tail
(`Distribution::in_far_tail`) is asked of the family's non-cancelling form
of that tail (`cdf_lower`, `sf`) where it has one: the certified `f64` of the
classic CDF next to 0 or 1 costs a working precision of as many digits as
the tail is small.  The question itself costs a certified evaluation and
is asked only of a family that has the other form (since 0.60.1; up to
0.60 every numeric `cdf` of a Student t, F, beta, gamma or χ² law
evaluated its own result once for nothing).  A table keeps what a query
computed per value
(`discrete::TableIndex`, `Finite::atom_f64`): no query evaluates every value
again, and none scans the values before the current one.  A `try_`
constructor decides a symbol-free parameter by the certified queries
(finiteness, realness, sign, integrality by the sign of `x − ⌊x⌋`), never by
`as_rational` alone.  Dense layouts derived from data are bounded
(`solve::MAX_DENSE_POLY_DEGREE`, `series::MAX_DENSE_KNOWN`,
`discrete::BINOMIAL_MASS_MAX_BITS`); beyond them the routine refuses or
keeps the value symbolic.  The keys by which `transformed` tells the images
of a table apart decide only what is certified: two images are merged by
`equals` alone; a key (`wrappers::value_key`: the `f64`, `ln |image|`
beyond the doubles, and since 0.59 the residual `image − f64` for images
that share a double, `residual_key`) only proves two images different.
The bounds between the scales are widened (`HUGE_LOG`, `TINY_LOG`) so that
no rounding puts one number into two scales.  A density whose factors are
powers with a huge shape in the exponent, or Gamma functions of it, is
written at a numeric point with the factors combined in logarithms
(`continuous::huge_shape`, since 0.59: `FDistribution`, `StudentT`), `ln(1
+ y)` as `2 atanh(y/(2 + y))` — a rational power with a half-integer
exponent is split into numerator and denominator powers by the canonical
form, each beyond even the scaled exponent range.  A family whose generic
entropy would sum the masses exactly returns the formal `Sum` above 64
atoms (`Binomial`, `Hypergeometric`).

A root search over certified evaluations has a budget in evaluations and
brackets in the order of the floats (since 0.60.1):
`Distribution::continuous_quantile_f64` takes the support as its bracket
when it has no estimate, grows one by a factor squared at each step
(`BRACKET_MAX_STEPS`), halves the *floats* between the ends
(`narrow_bracket`, `NARROW_MAX_STEPS`) and only then calls Brent's method
(`QUANTILE_ROOT_STEPS`) — a relative tolerance on a bracket of 300 decades
costs a thousand halvings of the width (up to 0.60 12 s for one quartile of
a truncated Cauchy law).  A distribution function built from others is not
simplified at a number (`wrappers::tidy_at`): its callers evaluate it, and
`simplify` of a constant samples it numerically.  An incomplete beta
function whose argument lies within `2⁻¹⁶` of 1 is written in the
complementary argument, which carries the digits
(`continuous::reg_beta_at`).  A table whose values are in a proved order
says so (`Finite::with_order`): its queries bisect (`Finite::locate`) and
sum integers over a common denominator, and evaluate no value to order it;
`transformed` proves the order of the images from the monotonicity of the
map on the closed interval the values span.

Oracles for extreme shapes: mpmath's `betainc` returns `1.0`, `0.5` or a
value wrong in the fourth digit — the same at 60 and 90 digits — for a shape
above about `10⁶`, and `1.0` for a shape `b` below `10⁻³` unless the
precision exceeds `−log₁₀ b` digits; check such a value against a limit with
a proved error (a normal law, an incomplete gamma function) instead.

**Branch on error variants, never on message text.**  A failure callers act
on gets a variant (`BudgetExhausted`, `ExpressionSwell`,
`NotDiagonalizable`, or an internal error enum like quadrature's
`QuadError`); `e.to_string().contains(…)` is not control flow.  A wrapper
that gives its own message to one failure of a callee matches that variant
(`e.degeneracy()`) and returns every other error unchanged
(`reliability::pearson_of`; up to 0.58 it relabelled a shape error as a
constant column).

### Error Handling

| Context | Approach |
|---------|----------|
| Symbolic computation can't complete | Return unevaluated form (`Ex`) |
| Numeric evaluation fails | Return `Err(SymplexError)` |
| Structural precondition violated | Return `Err(SymplexError)` |
| Programming error (cross-context) | Panic with clear message |
| Internal invariant violated | `debug_assert!` (stripped in release) |
| Never use in library code | `.unwrap()`, `.expect()`, `panic!()` |

### Locking

The arena is behind a `parking_lot::RwLock`. Follow these rules:

1. **Hold the lock for the minimum time.** Use the `drop(inner);` pattern:
   ```rust
   let mut inner = self.inner.write();
   let id = inner.arena.some_operation(self.raw_id());
   drop(inner);  // release before wrapping
   self.wrap(id)
   ```

2. **Never nest locks.** The `AssumptionCache` has its own `Mutex` inside
   the `RwLock` — this is the only permitted nesting, and the order is fixed
   (outer `RwLock` first, inner `Mutex` second).

3. **Never hold the lock across a method call on `Ex`.** That would deadlock
   because `Ex` methods acquire the same lock.  A closure of the caller is
   a method call on `Ex` you cannot see: never call one with a lock held.
   Collect what the closure needs under the lock (copy the node:
   `ExprView` holds a copy, not the arena), release, call, and take the
   lock again to write (`Ex::replace`, `Rule::apply_id`).  Up to 0.60.0
   `replace` held the write lock across its closure, and a closure that
   built an expression never returned.  The one exception is
   `Context::with_arena_mut`, whose closure gets the arena itself and is
   documented not to touch handles.

### Proc Macros

The five proc macros (`expr!`, `matrix!`, `eq!`, `dim!`, `rule!`) live in
`symplex-macros/`, beside the doc-hidden `#[derive(NodeShape)]` that
generates `ExprNode`'s shape methods (see "Expression Nodes"); the declarative macros (`syms!`, `sym!`, `vars!`,
`const_assert_dim!`) live in `src/api/macros.rs` and `src/units/assert_macros.rs`.
They all take an explicit first argument:

| Macro | Syntax | First arg |
|-------|--------|-----------|
| `expr!` | `expr!(ctx, x^2 + 1)` | Context variable |
| `matrix!` | `matrix![ctx, [1,2], [3,4]]` | Context variable |
| `eq!` | `eq!(ctx, x^2 = 1)` | Context variable |
| `dim!` | `dim!(ctx, Force: &m * &a)` | Context variable |
| `rule!` | `rule!(arena, "name", LHS => RHS)` | Arena variable (inside `ctx.with_arena_mut`; wrap with `RuleSet::from_macro_rules`) |
| `syms!` / `vars!` | `syms!(ctx; x, y, z)` | Context (semicolon separator) |
| `sym!` | `sym!(ctx; t, Positive)` | Context (semicolon separator) |

Note that `expr!` requires every identifier to be a local `Ex` variable of the
same name.  `matrix!` checks its shape at compile time (non-empty,
rectangular) and expands to an infallible construction; the `trybuild`
snapshots in `tests/ui/` pin the error messages.  `symplex-macros` depends
on `syn` with an explicit minimal feature set (`rule!`'s `if <closure>`
condition needs `full`, the `NodeShape` derive needs `clone-impls`); keep it
explicit rather than relying on feature unification with other proc-macro
crates in the graph — the crate is checked on its own by the gate's
`macros` stage, where unification does not help.

Declaration macros use semicolons. Expression macros use commas.

**The context argument is deliberate and stays.**  `expr!(x^2 + 1)` *could*
recover the context from `x` (`Ex::context()` is cheap), but then the
macro's meaning would depend on the shape of its input: `expr!(2^10)` has
no operand to take a context from, `expr!(x + y)` with `x` and `y` from
different contexts would fail only at run time inside the expansion, and a
reader could no longer tell which arena a literal is interned into.  We
prefer one explicit rule ("the first argument is where the literals live")
over an implicit one that is right most of the time.  The same reasoning
rules out a global or thread-local default context (the units context is the
single documented exception).  Ergonomics improvements to `expr!` must be
explicit in the same way — for example splicing an arbitrary Rust expression
with visible delimiters rather than guessing at identifiers.

---

## Common Pitfalls

### 1. Accessing `.id` directly

```rust
// WRONG — won't compile (id is private):
let id = other_expr.id;

// CORRECT:
let id = self.checked_id(&other_expr);  // validates same context
let my_id = self.raw_id();              // own ID, always safe
```

### 2. Using `.unwrap()` in library code

```rust
// WRONG — will panic on bad input:
let val = expr.eval_f64().unwrap();

// CORRECT — propagate the error:
let val = expr.eval_f64()?;
```

### 3. Creating expressions without a context

```rust
// WRONG — there's no global context:
let x = symplex::var("x");  // doesn't exist

// CORRECT:
let ctx = Context::new();
let x = ctx.symbol("x");
```

### 4. Forgetting `checked_id` in a new method

```rust
// WRONG — uses other.raw_id() without checking context:
pub fn my_method(&self, other: &Ex) -> Ex {
    let id = self.inner.write().arena.op(self.raw_id(), other.raw_id());
    self.wrap(id)
}

// CORRECT — validates context first:
pub fn my_method(&self, other: &Ex) -> Ex {
    let other_id = self.checked_id(other);
    let id = self.inner.write().arena.op(self.raw_id(), other_id);
    self.wrap(id)
}
```

### 5. Recursive tree traversal

```rust
// WRONG — will stack overflow on deep expressions:
fn walk(node: &ExprNode) {
    for child in node.children() {
        walk(arena.node(child));
    }
}

// CORRECT — use explicit stack:
fn walk(arena: &Arena, root: ExprId) {
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        for child in arena.node(id).children() {
            stack.push(child);
        }
    }
}
```

The same holds for every tree a caller can hand in.  `ExprTree` is a public
owned tree and can be as deep as the caller builds it: since 0.59
`tree_to_expr` walks it with an explicit stack (children from
`ExprTree::head_and_children`, results by node address; up to 0.58 a tree
nested 100,000 deep overflowed the stack in `Context::from_tree`).  The
tree's own `Drop`, `Clone` and `PartialEq` are still recursive (derived).

---

## Questions?

Open an issue on GitHub (the templates ask for the input, the result and
the oracle behind the answer you expected). We're happy to help new
contributors find their way around the codebase.  For a vulnerability, see
[`SECURITY.md`](SECURITY.md).