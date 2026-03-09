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
- [Code Style](#code-style)
- [Common Pitfalls](#common-pitfalls)

---

## Quick Start

```bash
# Clone and build
git clone https://github.com/cgorski/symplex
cd symplex
cargo build

# Run the test suite (~5,900 tests)
cargo test

# Run a single test file
cargo test --test test_known_answers

# Run doc-tests only
cargo test --doc

# Run benchmarks
cargo bench

# Run an example
cargo run --example quickstart

# With tracing output (see simplification steps, integration attempts, etc.)
RUST_LOG=symplex=debug cargo run --example quickstart
```

---

## Architecture

The source code is organized into 10 directories under `src/`. Each directory
is a layer in the dependency hierarchy — modules may depend on layers below
them but should not reach upward.

```
src/
├── base/         Expression nodes, arena, tree traversal, canonicalization, assumptions
├── poly/         Dense/sparse polynomials, Gröbner bases, Sturm sequences, root finding
├── transforms/   Differentiation, integration, evaluation, solving, pattern matching
├── simplify/     Trig, power, log, combinatorial simplification, Fu's algorithm
├── calculus/     Series, limits, Laplace, ODE, Gosper summation, formal power series
├── output/       Display, LaTeX, Rust codegen, CSE, JSON serialization, parser
├── plotting/     Adaptive sampling, textplot, SVG, TikZ, data export, RK4
├── domains/      Matrices, control systems, dynamics, robotics, number theory, vectors
├── units/        Compile-time dimensional analysis, quantity types, physical constants
└── api/          Public types (Ex, Context), all public methods, operator overloads
```

**Dependency flow** (each layer may only call downward):

```
base → poly → transforms → simplify → calculus
                                         ↓
                              output / plotting / domains → api / units
```

### Key Types

| Type | Location | Purpose |
|------|----------|---------|
| `ExprNode` | `src/base/node.rs` | The expression tree — ~80 variants (Add, Mul, Sin, Integral, RootOf, etc.) |
| `Arena` | `src/base/arena.rs` | Hash-consed expression storage. All nodes live here. |
| `ExprId` | `src/base/node.rs` | A `u32` index into the arena. This is how expressions are referenced internally. |
| `Context` | `src/api/context.rs` | User-facing entry point. Owns an arena + assumption cache. |
| `Expr<S>` / `Ex` | `src/api/expr.rs` | User-facing expression handle. Carries a context reference + ExprId. |
| `Poly` | `src/poly/dense.rs` | Dense univariate polynomial over `Ratio<BigInt>`. |
| `MultiPoly` | `src/poly/multipoly.rs` | Sparse multivariate polynomial. |
| `Matrix` | `src/domains/matrix.rs` | Symbolic matrix (Vec of Vec of Ex). |

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

The symbolic layer has **zero panics** except for the cross-context guard.
Every operation that can fail returns `Result` or produces an unevaluated form.
Never use `.unwrap()` or `.expect()` in library code (tests are fine).

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
`try_diff`, `try_integrate`, `try_limit`, `try_series`, `try_laplace`,
`try_inverse_laplace`, `try_residue`, `try_gosper_sum`, `try_solve_ode`,
`try_solve_gt/ge/lt/le`.

The `try_` variant calls the base method, then checks `has_unevaluated()`.
Zero code duplication.

### Pattern 3: Numeric boundary → always `Result`

Operations crossing from symbolic to numeric always return `Result`:

```rust
expr.eval_f64()         // Err if free symbols remain
expr.eval_complex64()   // Err if can't evaluate
expr.compile()          // Err if unsupported nodes
expr.to_rust_fn("f")    // Err if can't generate code
```

### Pattern 4: Queries → `Option`

Three-valued queries (yes / no / can't determine):

```rust
expr.is_positive()   // Some(true), Some(false), or None
expr.degree(&x)      // Some(3) or None (not polynomial)
expr.equals(&other)  // Some(true), Some(false), or None
```

### Pattern 5: Structural preconditions → `Result`

Matrix operations that require specific shapes:

```rust
matrix.det()          // Err if non-square
matrix.inv()          // Err if singular
matrix.matmul(&other) // Err if dimensions don't match
```

---

## Expression Nodes

The `ExprNode` enum in `src/base/node.rs` has ~80 variants. They fall into categories:

| Category | Examples | How they work |
|----------|---------|---------------|
| **Atoms** | `Num(NumId)`, `Symbol(SymbolId)`, `Pi`, `E`, `ImaginaryUnit` | Leaf nodes, no children |
| **N-ary arithmetic** | `Add(SmallVec)`, `Mul(SmallVec)` | Flattened associative ops |
| **Binary arithmetic** | `Pow(ExprId, ExprId)` | Base, exponent |
| **Functions** | `Sin(ExprId)`, `Exp(ExprId)`, `Gamma(ExprId)` | Unary or binary |
| **Calculus** | `Derivative(body, var)`, `Integral(body, var)` | Formal/unevaluated |
| **Unevaluated** | `Limit`, `Series`, `LaplaceTransform`, `RootOf`, `DSolve`, `ConditionSet` | Formal results when computation can't produce a closed form |
| **Boolean** | `BoolTrue`, `Gt`, `And`, `Or`, `Not` | For inequalities and logic |
| **Sets** | `Interval`, `FiniteSet`, `SetUnion` | For solution sets |
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

15. **`src/output/codegen.rs`** and **`src/output/lambdify.rs`**: Add code
    generation support or add to the "unsupported" arm.

16. **Tests**: Add tests in an appropriate test file.

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

Tests are in `tests/` (integration tests) and inline `#[cfg(test)]` modules
(unit tests). There are ~5,900 tests total.

| Category | Files | What they test |
|----------|-------|----------------|
| `test_known_answers.rs` | 256 tests | Exact symbolic results against textbook answers |
| `test_sympy_cross_validation.rs` | 263 fixtures | Results compared against SymPy 1.14 |
| `test_correctness_audit.rs` | 108 fixtures | FTC verification, definite integrals |
| `proptest_*.rs` | ~130 properties | Algebraic axioms, idempotence, value preservation |
| `test_cross_context.rs` | 17 tests | Cross-context safety guards |
| `test_ode_comprehensive.rs` | 44 tests | All 13 ODE solver classes |
| `test_rootof.rs` | 13 tests | RootOf solver + numerical evaluation |
| `test_hard_math.rs` | 52 tests | Edge cases and negative tests |

### Running Specific Tests

```bash
# Run one test file
cargo test --test test_known_answers

# Run one test function
cargo test --test test_known_answers diff_power_rule

# Run with output visible
cargo test --test test_known_answers -- --nocapture

# Run only lib unit tests (faster)
cargo test --lib

# Run only doc-tests
cargo test --doc
```

### Test Helpers

`tests/common/mod.rs` provides shared helpers:

| Helper | Purpose |
|--------|---------|
| `assert_math_eq(a, b, var, label)` | Numerical equality at multiple points |
| `assert_ftc_tol(integrand, var, tol, label)` | FTC: d/dx(∫f dx) ≈ f |
| `verify_roots(poly, var, roots, tol)` | Substitute roots back, check ≈ 0 |
| `approx_eq(a, b, tol)` | Float comparison with NaN/Inf handling |

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
   ```

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
`symplex-macros/`. They all take an explicit first argument:

| Macro | Syntax | First arg |
|-------|--------|-----------|
| `expr!` | `expr!(ctx, x^2 + 1)` | Context variable |
| `matrix!` | `matrix![ctx, [1,2], [3,4]]` | Context variable |
| `eq!` | `eq!(ctx, x^2 = 1)` | Context variable |
| `dim!` | `dim!(ctx, Force: &m * &a)` | Context variable |
| `rule!` | `rule!(arena, "name", LHS => RHS)` | Arena variable |
| `syms!` | `syms!(ctx; x, y, z)` | Context (semicolon separator) |
| `sym!` | `sym!(ctx; t, Positive)` | Context (semicolon separator) |

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