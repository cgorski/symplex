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
  the publication is not enough, use a clean room: one person or agent
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
- [How to Add a New Node Type](#how-to-add-a-new-node-type)
- [Testing](#testing)
- [Continuous Integration](#continuous-integration)
- [Code Style](#code-style)
- [Common Pitfalls](#common-pitfalls)

---

## Quick Start

```bash
# Clone and build
git clone https://github.com/cgorski/symplex
cd symplex
cargo build

# Run the test suite (~14,600 tests; a few minutes in debug)
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

`Cargo.lock` is not committed, so CI and users resolve the newest compatible
versions while a local lock may be old.  Compare performance between
versions only with the same lock, and when a dependency version matters,
raise its floor in `Cargo.toml`: up to 0.51 a local lock pinned num-bigint
0.4.6, which allocates for every one-digit number, and the matrix
functions looked 40 % slower than in 0.48 (built with 0.4.8); `Cargo.toml`
now requires 0.4.8.

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

The source code (~245K lines at 0.23.0) is organized into 10 directories under
`src/`. Each directory is a layer in the dependency hierarchy — modules may
depend on layers below them but should not reach upward.

```
src/
├── base/         Expression nodes (node.rs, 93 variants), arena (hash-consing), symbol.rs (name
│                 interning), tree traversal (walk.rs), canonicalization, assumptions, sort keys,
│                 compaction, numeric.rs (exact f64 ↔ rational, the exact rational `Q`),
│                 bigcomplex.rs (BigFloat complex pairs), bernoulli.rs, complex.rs (Complex64
│                 helpers), errors.rs, config.rs (EvalConfig, EXPRESSION_BUDGET), and the shared
│                 value types of 0.15–0.21: interval.rs (Interval/Bounds), extended.rs (Extended),
│                 rng.rs (SplitMix64, XorShift64Star), budget.rs (Budget), stage.rs (stage! tracing), combinatorics.rs,
│                 graph.rs (Tarjan SCC), libfn.rs (the LibFn registry of 50 special functions),
│                 dense_f64.rs (flat f64 Cholesky/SPD/Jacobi kernels), and the 0.49 type
│                 architecture: func.rs (Func/FuncSpec: one identity and fact table for all
│                 101 functions; generates LibFn), truth.rs (Truth, SignSet, Evidence,
│                 Verdict), cast.rs (checked float→integer), bigfloat.rs (the astro-float
│                 boundary); node.rs's shape methods come from symplex-macros' NodeShape derive
├── poly/         traits.rs (Ring/Field/EuclideanDomain hierarchy, the `poly_gcd` hook),
│                 dense.rs/generic.rs (GenPoly<C>), zpoly.rs (ℤ-scaled forms, gcd_via_z),
│                 modpoly.rs (𝔽ₚ[x]: Fp64, PolyIn<R>), interp.rs (Newton divided differences),
│                 factor_zassenhaus.rs (Berlekamp–Zassenhaus over ℤ + Kronecker multivariate),
│                 Gröbner bases, polysys.rs, Sturm sequences, root finding (roots.rs), algebraic
│                 number fields ℚ(α) (algebraic.rs), ratfn.rs, multipoly.rs (sparse multivariate;
│                 heuristic gcd/lcm, content, denominators), polybridge.rs (Ex ↔ polynomial,
│                 incl. symbolic-coefficient term collection)
├── transforms/   diff, integrate (+ heurisch, trig_integ, apart), eval, evalf, expand, solve,
│                 inequalities, pattern.rs (AC matcher), subs, solve/ (context.rs: SolveCx and
│                 Domain; outcome.rs: Solutions),
│                 sets.rs (set-algebra normal form), logic.rs (boolean simplifier, DPLL,
│                 piecewise), rsolve.rs (recurrences)
├── simplify/     simplify_engine (multi-strategy fixpoint + tracing), rewrite.rs, trig/hyp
│                 (fu.rs, trigsimp, trig_expand, trig_combine), powsimp/radsimp (powdenest,
│                 sqrtdenest), log_expand/log_combine, combsimp, nsimplify, refine, factor_terms,
│                 ratsimp.rs (rational-function normal form over opaque indeterminates)
├── calculus/     definite.rs (definite/improper integration, GK15 quadrature), summation.rs (+ the sum_eval shim eval() calls),
│                 gosper.rs, convergence.rs, series.rs, formal_series.rs, finite_diff.rs,
│                 limit.rs + gruntz.rs, residue.rs, laplace.rs, fourier.rs (series),
│                 fourier_transform.rs, mellin.rs, z_transform.rs, ode.rs, risch/ (tower,
│                 hermite, rde, rothstein_trager, log_to_real)
├── output/       display, pretty, latex, mathml, parse, tree (JSON), common.rs (shared print
│                 ordering), cse, lambdify (stack-VM compile), lean.rs (Mathlib rendering) +
│                 lean_proof.rs (structured tactic proofs), codegen.rs (Rust) +
│                 codegen/{codegen_c.rs (C99), codegen_py.rs (Python), numeric_rt.rs (shared f64
│                 special-function runtime), rt_embed.rs (embedding `mod symplex_rt`)}
├── plotting/     Adaptive sampling, textplot, SVG, TikZ, data export, RK4
├── domains/      matrix.rs + matrix_decomp.rs (QR, Cholesky, LDL, Gram–Schmidt, structure
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

### Key Types

| Type | Location | Purpose |
|------|----------|---------|
| `ExprNode` | `src/base/node.rs` | The expression tree — 93 variants (Add, Mul, Sin, Integral, Re/Im/Conjugate/Arg, Zeta, Polygamma, RootOf, RootSum, Interval, etc.) |
| `Arena` | `src/base/arena.rs` | Hash-consed expression storage. All nodes live here. |
| `ExprId` | `src/base/node.rs` | A handle into the arena (equality and hashing only: no order — see "Identifiers have no order"). This is how expressions are referenced internally. |
| `Func` / `FuncSpec` | `src/base/func.rs` | One identity for every function (dedicated nodes, `LibFn` library functions, internal helpers) and its table of facts: names, arity, sort sub-rank, LaTeX, conjugation, real domain, parity, poles. See "How to Add a New Function". |
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
| `Matrix` | `src/domains/matrix.rs` | Symbolic matrix (Vec of Vec of Ex); decompositions in `matrix_decomp.rs`. |
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
table's `u32` index conversions and the compile-time `const_assert_dim!`
macro.  Every other operation that can fail returns `Result`, `Option`, or an
unevaluated form.  Tests are free to unwrap; `debug_assert!` is acceptable for
internal invariants.

A second, shrinking category is **runtime `assert!`/`assert_eq!`/`assert_ne!`
on caller-supplied shapes and preconditions** — 82 sites in 19 files at 0.22,
27 in 7 at 0.29, e.g. `Context::symbol("")`, a zero denominator to
`RationalFn`, exponent overflow in `MultiPoly::mul`/`pow` (the bodies of
`*`, whose `try_` twins return `None`), an out-of-range index to
`Matrix::get` (the body of `m[(i, j)]`; `try_get` is its twin).  The
`Matrix` / `ExactMatrix` constructors return `Result` since 0.29.  Each is documented
under `# Panics` on its item.  They
predate point 4 below and are debt, not precedent: new code returns a
`Result` instead, and converting an existing one means removing it from the
allowlist.

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
   unreachable — `unwrap_or_else(|_| Matrix::zeros(n, m))` is wrong;
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
   `unwrap_or_else(PoisonError::into_inner)`.  `u32::try_from(len)` on arena
   indices propagates as `ComputationFailed` ("arena overflow"); it will
   never fire, but the code that would be *wrong* if it did should not exist.

7. **Type-level guarantees beat runtime checks when they are free.**  The
   `Sort` phantom on `Expr<S>` (an `Ex` cannot be used where a `BoolEx` is
   needed), sealed `ExactScalar`, and `#[non_exhaustive]` on option structs
   and outcome enums are all zero-cost and remove whole classes of failure.
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

The `ExprNode` enum in `src/base/node.rs` has 93 variants. Its shape methods
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
keys on every node of the Rubi arenas.

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
`Func::is_special()` through `ExprNode::func`, instead of an or-pattern of
variants; a list without a family gets a predicate on `Func` and a test that
it selects the intended set.  The entry is a
`FuncSpec` struct literal with every field mandatory, so a function that
forgets a facet does not compile:

```rust,ignore
/// Imaginary error function `erfi(x)`.
Erfi => FuncSpec {
    name: "erfi", parse: &[], arity: Exactly(1), sort: Named,
    tex: Tex::Call(r"\operatorname{erfi}"),
    conj: Conj::Entire,            // f(conj z) = conj f(z) everywhere (DLMF 7.4.2)
    real_on: RealOn::Reals,        // real for real x
    parity: Odd(Coefficient),      // must match today's sign extraction (a test checks it)
    poles: Poles::None,
    constants: Constants::Evaluated,
    rt: Runtime::helper(Kernel::Unary("erfi"), Backends::VM_RUST_C)
        .scipy("scipy.special.erfi(x)"),
};
```

The data facets — names (display, parser aliases, LaTeX layout), argument
count, sort sub-rank (node-backed functions keep their historical byte; a
test checks uniqueness), conjugation rule, the real-argument domain on which
the function is real, parity, poles, constant folding, and `rt` (which `f64`
back ends evaluate it and with which runtime kernel) — are read from the
table by every consumer (`conjugate`, evalf's conjugate pairing, the
assumption system, LaTeX, `compile`, `to_rust_fn`, `to_c_fn`, the Python
printers), so two consumers can no longer disagree.

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
arm.  Add its evaluation, derivative and printing where the compiler and the
tests point, and tests in the current group (`tests/v36/`).

**A node-backed function** needs a `node` entry with its `sort_key.rs` byte,
the `ExprNode` variant (the derive generates its shape), an arm in
`ExprNode::func`, `Arena::func` and `Arena::make`, and the steps below.

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
   attribute).  Add its `funcs!` entry in `src/base/func.rs` and its arms in
   `ExprNode::func`, `Arena::func` and `Arena::make` (all exhaustive).

2. **`src/base/sort_key.rs`**: Add a sub-rank byte (unique; the `func.rs`
   tests check it) and match arm.  Sort keys are byte-identity critical: never
   renumber an existing function.

3. **`src/base/walk.rs`, `src/base/compact.rs`**: nothing (generated).

4. **`src/base/assumptions.rs`**: nothing for realness, parity and poles (the
   facets); add a handler only for facts the table cannot express.

5. **`src/output/display.rs`**: Add display formatting.

6. **`src/output/latex.rs`**: Add LaTeX rendering.

7. **`src/output/tree.rs`**: Add `ExprTree::Cot` variant + serialization.

8. **`src/output/parse.rs`**: Add `"cot"` to the function name dispatch.

9. **`src/transforms/diff.rs`**: Add differentiation rule:
   `d/dx cot(f) = -csc²(f) · f'`.

10. **`src/transforms/eval.rs`**: Add evaluation for known values
    (e.g., `cot(π/4) = 1`).

11. **`src/transforms/evalf.rs`**: Add numerical evaluation.

12. **`src/transforms/integrate.rs`**: Add integration rule if known.

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

17. **`src/output/parse.rs`** round-trip: `parse(&ctx, &format!("{expr}"))`
    must reproduce the node.

18. **Tests**: Add tests in an appropriate test file (see below).

---

## How to Add a New Node Type (Unevaluated Form)

Unevaluated nodes represent computations that couldn't produce a closed form.
For example, `Limit(body, var, point)` represents `lim_{var→point} body`.

Follow the same steps as adding a new function (above), plus:

1. Add it to `is_unevaluated_node` in `src/base/arena.rs` (the per-node
   facts that `walk::has_unevaluated` reads are computed from it).

2. Consider adding it to `ExprType::Unevaluated` in `src/api/expr.rs`.

3. The Display should show the formal notation:
   `Limit(f, x, 0)` or `\lim_{x \to 0} f` in LaTeX.

4. The parser in `src/output/parse.rs` should be able to parse the Display
   output back (round-trip).

---

## Testing

### Test Organization

Tests are in `tests/` (integration tests, ~10,000 `#[test]` functions in ~360
source files) and inline `#[cfg(test)]` modules (~2,800 unit tests). Total at
0.23.0: **~12,900** tests plus ~1,040 API doctests and ~240 README/book
doctests (`src/doctests.rs` includes `README.md` and every book chapter, so
the prose examples compile and run under `cargo test --doc`).

The integration-test sources are grouped into **23 test binaries** (linking
277 separate debug binaries took ~6.5 min and ~13 GB of `target/`). Each
former top-level file is a module of its group, so a test is addressed as
`<module>::<test>` inside `--test <group>`.  A new `vNN` group is added per
minor release (`v23` holds the 0.24 tests); `tests/README.md` lists every
group's modules:

| Binary (`--test …`) | Sources | What they test |
|---------------------|---------|----------------|
| `v04` | `tests/v04/v04_<area>.rs` | One suite per 0.4–0.6 feature: `polyhedron` (parametric polyhedron certificates; emitted Lean pinned to the Mathlib-compiled `tests/fixtures/polyhedron_certificates.lean`), `polytope`, `sos` (pinned to `tests/fixtures/sos_certificates.lean`) |
| `v09` … `v23` | `tests/vNN/vNN_<area>.rs` (2–9 modules each) | Feature and regression suites of 0.9 → 0.24, one group per minor release (stats families, data statistics, ANOVA, Cox, numdist, the exact kernel, …) |
| `v03` | `tests/v03/v03_<area>.rs` (11 modules, ~480 tests) | One suite per 0.3 feature: `poly_view`, `poly_symbolic_coeffs`, `ratsimp`, `linprog` (full KKT check of every optimum, Farkas vector verified), `normalforms` (defining invariants, not pinned answers), `matrix_ergonomics`, `optimize`, `certificates`, `assumptions_poly`, `exact_matrix`, `user_notes` |
| `v03_oracle` | `tests/v03_oracle/v03_oracle_*.rs` | SymPy oracle for the 0.3 API (`tests/fixtures/v03_cross_validation.json`) |
| `v02` | `tests/v02/v02_<area>_<topic>.rs` (62 modules, ~1,050 tests) | One suite per 0.2 feature area: `backends_{c,codegen,compile,cse}`, `basefix_*`, `ergonomics_*`, `integration_{battery,definite,residue}`, `matrices_*`, `nodes_*`, `ntheory_*`, `numfix_*`, `sets_*`, `simplify_*`, `solvefix_*`, `solving_*`, `summation_*`, `transforms_*` |
| `v02_oracle` | `tests/v02_oracle/v02_oracle_*.rs` | SymPy oracle for the 0.2 API (`tests/fixtures/v02_cross_validation.json`); see `tests/README.md` |
| `unit` | `tests/unit/test_*.rs` (159 modules, ~3,980 tests) | Unit-style suites per module / feature, e.g. `test_known_answers` (256 exact symbolic results against textbook answers), `test_sympy_cross_validation` (263 fixtures against SymPy 1.14, `tests/fixtures/sympy_cross_validation.json`), `test_correctness_audit` (108 fixtures: FTC verification, definite integrals, `tests/fixtures/new_capabilities.json`), `test_cross_context` (cross-context safety guards), `test_ode_comprehensive` (ODE solver classes), `test_rootof` (RootOf solver + numerical evaluation), `test_hard_math` (edge cases and negative tests) |
| `legacy` | `tests/legacy/{round*_*,math*_bugs,bugfinder*,*_validation,…}.rs` (23 modules) | Regression suites from bug-hunting rounds |
| `proptests` | `tests/proptests/{proptest_*,test_proptest_new,test_quality_props,test_units_proptest}.rs` (~180 properties) | Algebraic axioms, idempotence, value preservation, round-trips. `<stem>.proptest-regressions` files live next to the sources |
| `perf` | `tests/perf/{simplify_perf_test,perf_analysis}.rs` | `#[ignore]`d benchmarks (`--release -- --ignored --nocapture`) |
| `ui_tests` | `tests/ui_tests.rs` + `tests/ui/*.rs` | trybuild compile-fail snapshots for the units type system (`Mass + Length` must not compile) |

Naming convention for new work: `tests/v03/v03_<area>[_<topic>].rs` (or the
appropriate `tests/unit/test_*.rs`), one behaviour per test function, named
for the mathematical fact it checks (`divergent_interior_pole_is_err`, not
`test1`). Register a new file in its group root (`tests/v03.rs`, …) with
`#[path = "v03/<file>.rs"] mod <file>;` — do not add new top-level
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

# Everything CI runs on stable (UI snapshots excluded — see below)
cargo test --lib --bins --tests --examples -- --skip units_compile_fail
cargo test --benches
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

- Every test must finish in well under a second in debug mode; suites that
  exercise budgets (Gruntz, rewrite, eigenvalue swell) assert wall-clock bounds
  with `std::time::Instant` (see `tests/v02/v02_transforms_limits.rs`).
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
  compiles all ~1,300 doctests (API, README and book) into one merged binary; if *one* doc example
  fails to compile there, it silently falls back to compiling every doctest
  standalone (~3 s each, twenty minutes in all).  A single trivial doctest
  reporting `finished in 3.6s` instead of `0.00s` is the signature; find
  the broken example with `git diff -- src | grep '^[-+].*///'`.
- **Sub-agents run in parallel only when they are spawned in the same tool
  block.**  A `spawn_agent` call blocks until that agent's final message;
  two calls in consecutive blocks run one after the other, doubling the wall
  time and leaving the coordinator idle.  Commit the skeleton (stubs
  reachable, tree compiles), then issue every `spawn_agent` for the
  milestone in one block with disjoint write sets, and do only
  coordinator work (reviewing the plan, drafting the CHANGELOG) while they
  run.  Agents share `target/`, so one agent's transient build break is
  visible to the others — tell them to wait and retry, not to fix files they
  do not own.  Agents never commit; integrate with one `cargo fmt --all`.

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

`.github/workflows/ci.yml` runs on every push and pull request to `main`:

| Job | What it does |
|-----|--------------|
| **Format** | `cargo fmt --check` for the main crate, `symplex-macros`, `symplex-build`, `symplex-wasm` |
| **Clippy** | `cargo clippy --all-targets -- -D warnings` for all four crates |
| **Test (stable)** | `cargo test --lib --bins --tests --examples -- --skip units_compile_fail`, `cargo test --benches`, `cargo test --doc` |
| **UI compile-fail tests** | `cargo test --test ui_tests` on the pinned `UI_TOOLCHAIN` (1.95.0) |
| **Sub-crates** | `cargo test` for `symplex-macros`, `symplex-build`, `symplex-wasm` (native); `cargo check --target wasm32-unknown-unknown` for `symplex-wasm`; `cargo check` for `fuzz/` |
| **Docs** | `RUSTDOCFLAGS=-D warnings cargo doc --no-deps --document-private-items` (main crate and `symplex-build`), then `mdbook build book` |
| **MSRV** | `cargo check --all-targets` on Rust 1.93.0 for all four crates |
| **Examples run** | Every `examples/*.rs` except `repl` is run to completion with a 300 s timeout |

`.github/workflows/deploy-book.yml` builds the mdBook and deploys it to GitHub
Pages on pushes to `main`.

`.github/workflows/fuzz.yml` runs every `fuzz/` target for ten minutes each,
nightly and on demand (`workflow_dispatch`, with the time per target as an
input): one job per target, the corpus cached between nights, a failing
input uploaded as an artifact with its decoded expression in the log.  A red
nightly is a bug report: reproduce it with `fuzz/README.md`, pin it as a
regression test, fix the cause.

`RUSTFLAGS=-D warnings` is set globally, so a new warning anywhere fails CI.
Before opening a PR run at least `cargo fmt --all`, `cargo clippy --all-targets`,
`cargo test`, `cargo doc --no-deps`, and the examples.

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
allowlisted file loses one without the allowlist being tightened).  Each
allowlist entry names its kept sites and the convention that justifies them.

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
(`tests/unit/`, `syn`-based, exact per-file counts that may only shrink).

**Decisions: `Truth`, signs and evidence** (`src/base/truth.rs`).  An
internal yes/no/undecided answer is a `Truth`, not an `Option<bool>`: it has
no `PartialEq<bool>`, no `unwrap_or`, no ordering, so a branch must say
which reading it takes (`t.is_true()` = proved, `!t.is_false()` = not
refuted, or a `match` over the three cases).  Combine answers with
`and`/`or`/`not`/`all`/`any`; do not write Kleene tables by hand.  The signs a
finite value can have are a `SignSet` ({< 0, 0, > 0, non-real}) with
exhaustively tested `neg`/`add`/`mul`/`recip`/`pow_int`; infinities are not
signs — decide finiteness first.  The public query API keeps `Option<bool>`
(Pattern 4); convert at the boundary.

Only `Evidence::Exact` or `Evidence::Certified` may justify a claim (a
returned value, a dropped term, a leading coefficient, `Inconsistent`, a
branch that drops cases).  `Approximate` (an `f64`, a value zero only to the
working precision) and `Probabilistic` (random sampling) may only guide a
heuristic whose result is right either way — a pivot choice, the order of
methods — and are read visibly with `truth_at_least(Evidence::Approximate)`.
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
non-real exponents, and never test non-realness by "contains `i`".

Use `evalf_verdict(..).proven()` or `evalf_settled` with
`Settled::Certified` for a numeric proof; `evalf_complex`, `eval_const_f64`
and `expr_to_f64` drop the evidence.  Bernoulli numbers use `B₁ = +1/2`
everywhere; a formula written for `B₁ = −1/2` flips that one sign
explicitly.  Zero tests of constants in limits and
series go through `calculus_util::hidden_zero` (`Verdict<Truth>`): an
undecided coefficient may be kept as a value but never divided by, dropped,
or taken as a leading term; new exact rewrites go in `exact_rewrites`, each
an identity of the principal values wherever both sides are finite.  Logs
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
`Unknown` first.  A value that is not a truth value gets an enum of its own
(`TrigZeros`, `Parity`), not an `Option<bool>` whose `None` means something
else.  `tests/unit/test_truth_values.rs` counts per file the remaining
`== Some(bool)` comparisons, `unwrap_or(bool)` reads, internal functions
returning `Option<bool>`, `Option<…>` sign encodings and `(ExprId, u8)` memo
keys; the counts may only shrink.

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
real and `check_solution` answers from `is_zero`.

**Printing complex values.**  A result read as a number (`eval_decimal`,
`eval_f64`, `eval_complex64`) shows a part unless it is 0 within its error
ball: `settle` zeroes a part whose ball contains 0 and certifies the digits
of every other part, however small next to the other; `eval_f64` refuses a
nonzero imaginary part.  "Negligible next to the other part"
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
`sensitivity`.

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

**Cut crossings of antiderivatives.**  A Newton–Leibniz difference is
taken across a crossing of a branch cut only at an exactly located
crossing: a linear argument `u = αx + β` meets the real axis at `c = −im
β/im α`, and when `u(c)` is proved on the cut each cut function of `u` is
continued from the side `u` comes from (`definite::side_value`: `ln`, `Ci`,
`Chi` step by `2πi`, `uᵃ` by `e^{2πia}`, `Ei` takes the mean on the axis);
a crossing found only by sampling refuses.  On a piece in `[0, ∞)` an
argument built from `x^q` and `ln x` is real and crosses no cut
(`real_for_x_in`), without sampling.  Narrow peaks of a product of powers
and an exponential are split off at the maximum of its logarithm
(`definite::envelope_peak`), found in `f64`: a split point is a guide,
never a claim.

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
function in `canon::folds_at_infinity` left unevaluated at `±∞` has no value
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
functions of `transforms/eval.rs`; their values at infinity live there.
`eval` evaluates no closed form at an argument with `±∞` in it.

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
region), (Sum, True))`; `eval` of a formal `Sum` does not build it.

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

**Checked casts** (`src/base/cast.rs`).  Do not convert an `f64` to an
integer with `as` after `floor`/`ceil`/`round`/`trunc`: the cast saturates
(`1e19 as i64 == i64::MAX`, `NaN as i64 == 0`) and the `± 1` that usually
follows wraps (up to 0.48 `∫₋₁¹ floor(10¹⁹x)/10¹⁹` was 0).  Use
`cast::f64_to_i64` (exact or `None`) or `cast::integer_span` (a float range
widened to integers, with a length cap).  `tests/unit/test_float_casts.rs`
counts these casts and the `(f64 expr) as usize + …` guard-bit pattern per
files.  Differences of `i64` exponents and scales are `saturating_*` or taken
in `f64`.

**Exact arithmetic has size bounds.**  An exact power, or any intermediate
that grows with an exponent, is formed only below a bit bound
(`algebraic::MAX_POWER_BITS`, 2¹⁶; `MAX_POWER_DEGREE` for minimal
polynomials of powers), with checked conversions (`u32::try_from`, never
`as`); beyond the bound the decider returns `None`.  Up to 0.51
`rational_pow` cast the exponent `as u32`: `2^(2³² + 1)` became `2`, so
`2^4294967297 − 2` was called zero, and `1000000007^1000000007 − 2` hung.

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
astro-float release is noticed).

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
`A + C·u^(p/q)` real iff `u ≥ 0` becomes a conditional set), and a constant
factor of a product is proved nonzero before the other factors' roots are
claimed.  A root returned for a real unknown must be proved real; a constant discriminant or leading coefficient is decided by
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
otherwise it is refused.  A Lambert-W family of an ODE carries its `W₋₁`
twin, and `check_ode_solution` rewrites `e^{W(z)} = z/W(z)`.

**Budgets count work.**  An engine's limit counts deterministic work — a
`Work` charged per step, or a `WorkScope` that counts interns attempted —
never wall-clock time (only a caller's `Budget` deadline may be one) and
never `node_count()` growth, which depends on what the context already holds
(up to 0.48 a repeated `series` call got slower each time).  A budget that
surfaces as an error is `SymplexError::BudgetExhausted { kind }`; add a
`BudgetKind` per new budget.

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
…`).  A value with its error bound is a `CBall` (`Ball<&Complex>` for a
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

**Branch on error variants, never on message text.**  A failure callers act
on gets a variant (`BudgetExhausted`, `ExpressionSwell`,
`NotDiagonalizable`, or an internal error enum like quadrature's
`QuadError`); `e.to_string().contains(…)` is not control flow.

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
   because `Ex` methods acquire the same lock.

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

---

## Questions?

Open an issue on GitHub. We're happy to help new contributors find their way
around the codebase.