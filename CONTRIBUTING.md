# Contributing to Symplex

Welcome. This document covers everything you need to understand the codebase,
make changes safely, and submit contributions.

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

# Run the test suite (~11,000 tests; a few minutes in debug)
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

---

## Architecture

The source code (~174K lines at 0.3.0) is organized into 10 directories under
`src/`. Each directory is a layer in the dependency hierarchy — modules may
depend on layers below them but should not reach upward.

```
src/
├── base/         Expression nodes (node.rs, 92 variants), arena (hash-consing), tree
│                 traversal (walk.rs), canonicalization, assumptions, sort keys, compaction,
│                 numeric.rs (exact f64 ↔ rational), bernoulli.rs, complex.rs, errors.rs
├── poly/         Dense/sparse/generic polynomials, factor_zassenhaus.rs (Berlekamp–Zassenhaus
│                 over ℤ + Kronecker multivariate), Gröbner bases, polysys.rs, Sturm sequences,
│                 root finding (roots.rs), algebraic number fields ℚ(α) (algebraic.rs), ratfn.rs,
│                 multipoly.rs (sparse multivariate; heuristic gcd/lcm, content, denominators),
│                 polybridge.rs (Ex ↔ polynomial, incl. symbolic-coefficient term collection)
├── transforms/   diff, integrate (+ heurisch, trig_integ, apart), eval, evalf, expand, solve,
│                 inequalities, pattern.rs (AC matcher), subs, sum_eval,
│                 sets.rs (set-algebra normal form), logic.rs (boolean simplifier, DPLL,
│                 piecewise), rsolve.rs (recurrences)
├── simplify/     simplify_engine (multi-strategy fixpoint + tracing), rewrite.rs, trig/hyp
│                 (fu.rs, trigsimp, trig_expand, trig_combine), powsimp/radsimp (powdenest,
│                 sqrtdenest), log_expand/log_combine, combsimp, nsimplify, refine, factor_terms,
│                 ratsimp.rs (rational-function normal form over opaque indeterminates)
├── calculus/     definite.rs (definite/improper integration, GK15 quadrature), summation.rs,
│                 gosper.rs, convergence.rs, series.rs, formal_series.rs, finite_diff.rs,
│                 limit.rs + gruntz.rs, residue.rs, laplace.rs, fourier.rs (series),
│                 fourier_transform.rs, mellin.rs, z_transform.rs, ode.rs, risch/ (tower,
│                 hermite, rde, rothstein_trager, log_to_real)
├── output/       display, pretty, latex, parse, tree (JSON), cse, lambdify (stack-VM compile),
│                 codegen.rs (Rust) + codegen/{codegen_c.rs (C99), numeric_rt.rs (shared f64
│                 special-function runtime), rt_embed.rs (embedding `mod symplex_rt`)}
├── plotting/     Adaptive sampling, textplot, SVG, TikZ, data export, RK4
├── domains/      matrix.rs + matrix_decomp.rs (QR, Cholesky, LDL, Gram–Schmidt, structure
│                 tests, norms, hessian, wronskian, 0.3 selection/conversion helpers),
│                 exact_matrix.rs (0.3.5: QMatrix/ZMatrix over Ratio<BigInt>/BigInt, the
│                 fraction-free Gauss–Jordan kernel and Bareiss determinant, HNF/SNF cores;
│                 Matrix routes all-rational input here), linalg.rs (symbolic rref/linsolve
│                 with the numeric fast path), linprog.rs (exact two-phase simplex on an
│                 integer-pivoting tableau, duals, Farkas certificates), normalforms.rs
│                 (Matrix wrappers over ZMatrix: Hermite/Smith normal forms, integer
│                 nullspace, unimodularity, lattice index), optimize.rs (Brent/bisection/Newton,
│                 Nelder–Mead, golden section, differential evolution, least-squares fits),
│                 control, dynamics, robotics, quaternion, vector (coordinate systems), ntheory
│                 (rho/ECM, BPSW, sqrt_mod, dlog, continued fractions, gcd_many/lcm_many),
│                 diophantine, combinatorics, separatevars
├── units/        Compile-time dimensional analysis, quantity types, conversions, constants
└── api/          context.rs, expr.rs (Expr<S>), expr_funcs.rs (most methods), expr_ops.rs
                  (operators, Scalar/ToEx, Context ingestion), eq.rs (Equation), macros.rs,
                  expr_view.rs, and the 0.2 extension files:
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
`fuzz/` (cargo-fuzz targets), `probes/` (developer diagnostics, not compiled by
default), `book/` (mdBook), `benches/` (criterion).

**Dependency flow** (each layer may only call downward):

```
base → poly → transforms → simplify → calculus
                                         ↓
                              output / plotting / domains → api / units
```

### Key Types

| Type | Location | Purpose |
|------|----------|---------|
| `ExprNode` | `src/base/node.rs` | The expression tree — 91 variants (Add, Mul, Sin, Integral, Re/Im/Conjugate/Arg, Zeta, Polygamma, RootOf, RootSum, Interval, etc.) |
| `Arena` | `src/base/arena.rs` | Hash-consed expression storage. All nodes live here. |
| `ExprId` | `src/base/node.rs` | A `u32` index into the arena. This is how expressions are referenced internally. |
| `Context` | `src/api/context.rs` | User-facing entry point. Owns an arena + assumption cache. |
| `Expr<S>` / `Ex` | `src/api/expr.rs` | User-facing expression handle. Carries a context reference + ExprId. |
| `Poly` (internal) | `src/poly/dense.rs` | Dense univariate polynomial over `Ratio<BigInt>`. Type alias for `GenPoly<Ratio<BigInt>>`; `pub(crate)`. |
| `Poly` (public) | `src/api/poly_ex.rs` | User-facing sparse polynomial view of an `Ex` over explicit generators with symbolic coefficients (`prelude::Poly`). |
| `GenPoly<C>` | `src/poly/generic.rs` | Generic univariate polynomial over any `Ring`/`Field` coefficient type. |
| `RationalFn` | `src/poly/ratfn.rs` | Rational function `p(x)/q(x)` in ℚ(x). Implements `Field`, enabling `GenPoly<RationalFn>`. |
| `AlgNum` | `src/poly/algebraic.rs` | Element of ℚ(α) = ℚ[t]/(m(t)). Implements `Ring` + `Field` with exact zero/sign testing. |
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
different contexts. This is the **only** panic in the symbolic layer (by design —
it's a logic error, like indexing a Vec out of bounds).

**When adding a new public method** that takes multiple `Ex` arguments, you
MUST use `self.checked_id(other)` for every foreign expression's ID. The
compiler enforces this — you can't access `.id` directly.

### No Panics Rule

The symbolic layer has **zero panics** except for two documented logic
errors: the cross-context guard, and `std::iter::Sum`/`Product` for `Ex` on an
*empty* iterator (there is no context to build `0`/`1` in — users are steered
to `Context::sum`/`product` or `Option<Ex>`). Every other operation that can
fail returns `Result`, `Option`, or an unevaluated form. Never use `.unwrap()`,
`.expect()`, `unreachable!()` or `panic!()` in library code (tests are fine);
`debug_assert!` is acceptable for internal invariants.

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

The `ExprNode` enum in `src/base/node.rs` has 91 variants. They fall into categories:

| Category | Examples | How they work |
|----------|---------|---------------|
| **Atoms** | `Num(NumId)`, `Symbol(SymbolId)`, `Pi`, `E`, `ImaginaryUnit`, `EulerGamma`, `Catalan`, `GoldenRatio`, `Infinity`, `ComplexInfinity`, `NaN` | Leaf nodes, no children |
| **N-ary arithmetic** | `Add(SmallVec)`, `Mul(SmallVec)` | Flattened associative ops |
| **Binary arithmetic** | `Pow(ExprId, ExprId)` | Base, exponent |
| **Functions** | `Sin(ExprId)`, `Exp(ExprId)`, `Gamma(ExprId)`, `Zeta`, `Si`/`Ci`/`Ei`/`Li`, `Polygamma(n, x)`, `KroneckerDelta(i, j)` | Unary or binary |
| **Complex** | `Re`, `Im`, `Conjugate`, `Arg` | Only constructed when realness is unknown |
| **Calculus** | `Derivative(body, var)`, `Integral(body, var)` | Formal/unevaluated |
| **Unevaluated** | `Limit`, `Series`, `Sum`, `Product_`, `LaplaceTransform`, `Residue`, `DSolve`, `ConditionSet` | Formal results when computation can't produce a closed form |
| **Algebraic answers** | `RootOf(poly, index)`, `RootSum(poly, body, var)` | Exact descriptions of algebraic numbers; *not* unevaluated |
| **Boolean** | `BoolTrue`, `Gt`, `And`, `Or`, `Not` | For inequalities and logic |
| **Sets** | `Interval`, `FiniteSet`, `SetUnion`, `SetIntersection`, `SetComplement`, `EmptySet`, `UniversalSet` | For solution sets; `transforms/sets.rs` computes the normal form |
| **Piecewise** | `Piecewise(Vec<(value, condition)>)` | Conditional expressions |

Every node has a `children()` method that returns its child `ExprId`s. This is
used by tree traversals throughout the codebase.

---

## How to Add a New Function

Example: adding `cot(x)` (cotangent).

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

1. **`src/base/node.rs`**: Add variant `Cot(ExprId)` to the enum. Update
   `children()`, `child_count()`, `for_each_child()`, `is_atom()` (it's not
   an atom), and `Debug`.

2. **`src/base/sort_key.rs`**: Add a `RANK_COT` constant and match arm.

3. **`src/base/walk.rs`**: Add to `rebuild_with_cache` if there's an exhaustive
   match there.

4. **`src/base/compact.rs`**: Add transfer arm (same pattern as `Sin`).

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

16. **`src/base/complex.rs`**: If the function is real-analytic, teach
    `conjugate`/`re`/`im` how it behaves (conjugation commutes with it).

17. **`src/output/parse.rs`** round-trip: `parse(&ctx, &format!("{expr}"))`
    must reproduce the node.

18. **Tests**: Add tests in an appropriate test file (see below).

---

## How to Add a New Node Type (Unevaluated Form)

Unevaluated nodes represent computations that couldn't produce a closed form.
For example, `Limit(body, var, point)` represents `lim_{var→point} body`.

Follow the same steps as adding a new function (above), plus:

1. Add to `has_unevaluated()` in `src/base/walk.rs` — return `true` for your node.

2. Consider adding it to `ExprType::Unevaluated` in `src/api/expr.rs`.

3. The Display should show the formal notation:
   `Limit(f, x, 0)` or `\lim_{x \to 0} f` in LaTeX.

4. The parser in `src/output/parse.rs` should be able to parse the Display
   output back (round-trip).

---

## Testing

### Test Organization

Tests are in `tests/` (integration tests, ~8,450 `#[test]` functions in ~275
source files) and inline `#[cfg(test)]` modules (~2,580 unit tests). Total at
0.3.0: **~11,000** tests plus ~610 doctests.

The integration-test sources are grouped into **nine test binaries** (linking
277 separate debug binaries took ~6.5 min and ~13 GB of `target/`). Each
former top-level file is a module of its group, so a test is addressed as
`<module>::<test>` inside `--test <group>`:

| Binary (`--test …`) | Sources | What they test |
|---------------------|---------|----------------|
| `v03` | `tests/v03/v03_<area>.rs` (7 modules, ~400 tests) | One suite per 0.3 feature: `poly_view`, `poly_symbolic_coeffs`, `ratsimp`, `linprog` (full KKT check of every optimum, Farkas vector verified), `normalforms` (defining invariants, not pinned answers), `matrix_ergonomics`, `optimize` |
| `v03_oracle` | `tests/v03_oracle/v03_oracle_*.rs` | SymPy oracle for the 0.3 API (`tests/fixtures/v03_cross_validation.json`) |
| `v02` | `tests/v02/v02_<area>_<topic>.rs` (62 modules, ~1,050 tests) | One suite per 0.2 feature area: `backends_{c,codegen,compile,cse}`, `basefix_*`, `ergonomics_*`, `integration_{battery,definite,residue}`, `matrices_*`, `nodes_*`, `ntheory_*`, `numfix_*`, `sets_*`, `simplify_*`, `solvefix_*`, `solving_*`, `summation_*`, `transforms_*` |
| `v02_oracle` | `tests/v02_oracle/v02_oracle_*.rs` | SymPy oracle for the 0.2 API (`tests/fixtures/v02_cross_validation.json`); see `tests/README.md` |
| `unit` | `tests/unit/test_*.rs` (156 modules, ~3,950 tests) | Unit-style suites per module / feature, e.g. `test_known_answers` (256 exact symbolic results against textbook answers), `test_sympy_cross_validation` (263 fixtures against SymPy 1.14, `tests/fixtures/sympy_cross_validation.json`), `test_correctness_audit` (108 fixtures: FTC verification, definite integrals, `tests/fixtures/new_capabilities.json`), `test_cross_context` (cross-context safety guards), `test_ode_comprehensive` (ODE solver classes), `test_rootof` (RootOf solver + numerical evaluation), `test_hard_math` (edge cases and negative tests) |
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
`symplex-macros/`; the declarative macros (`syms!`, `sym!`, `vars!`,
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
same name, and a purely numeric `expr!(ctx, 2^10)` does not compile (the
literals are `i64`); build constants with `ctx.int`/`ctx.rational` instead.

Declaration macros use semicolons. Expression macros use commas.

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