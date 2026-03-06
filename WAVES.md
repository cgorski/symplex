# Symplex 0.2.0 — Wave Execution Plan

> **Goal:** Transform symplex from a research prototype with world-class internals
> into a commercially credible product. Fix visible bugs, write narrative documentation,
> rewrite examples to tell stories, and prove the end-to-end pipeline that only symplex
> can deliver.

---

## Current State (Baseline)

| Metric | Value |
|--------|-------|
| Tests | 1,855 passing, 0 failing |
| Source | ~63,000 lines across 77 modules |
| Test code | ~49,000 lines across 126 test files |
| Examples | 1,105 lines across 13 examples |
| Documentation (narrative) | 0 lines (README only: 272 lines) |
| Clippy warnings (`--lib`) | 60 across ~20 src files |
| Clippy warnings (`--all-targets`) | 60 lib (duplicated) + 8 test-only |
| Known Limitations | 25 |

### Bugs Blocking 0.2.0

| ID | Bug | Where | Severity |
|----|-----|-------|----------|
| B1 | **Codegen emits `0.0_f64.powi(2)`, `1.0_f64` as CSE temps** — no constant propagation in CSE pipeline | `src/codegen.rs`, `src/cse.rs` | P0 |
| B2 | **Codegen `Add` emits `+ (-x)` instead of `- x`** — no Neg-child detection in codegen's Add handler | `src/codegen.rs` | P0 |
| B3 | **Codegen emits zero-multiplied terms** — `t0 * t6` where `t6 = 0.0` is not eliminated | `src/codegen.rs` | P0 |
| B4 | **Codegen excessive parenthesization** — `(((-1_f64 / 5_f64) * t0 * t1))` double-wrapping | `src/codegen.rs` | P1 |
| B5 | **Display `+ -N` edge cases** — code handles most cases but Known Limitation #17 still listed | `src/display.rs` | P1 |
| B6a | **Clippy warnings in test files** — `neg_multiply`, `collapsible_if` in test targets | `tests/test_bareiss.rs`, `tests/test_sympy_cross_validation.rs` | P1 |
| B6b | **60 clippy warnings in library source** — collapsible_if, returning let binding, loop variable indexing, div_ceil, is_multiple_of, complex type, etc. across ~20 src files | `src/canon.rs`, `src/eval.rs`, `src/evalf.rs`, `src/matrix.rs`, `src/ntheory.rs`, `src/poly.rs`, `src/ode.rs`, `src/dynamics.rs`, `src/groebner.rs`, `src/integrate.rs`, `src/latex.rs`, `src/control.rs`, `src/z_transform.rs`, `src/fourier_transform.rs`, `src/polysys.rs`, `src/vector.rs`, `src/codegen.rs` | P0 |
| B7 | **`Matrix::to_latex()` listed as missing** — actually exists but not documented/tested enough | `src/matrix.rs` | P2 (verify only) |
| B8 | **Codegen: fractions inline as division** — `-1_f64 / 5_f64` should be precomputed or emitted as `-0.2_f64` | `src/codegen.rs` | P2 |

### Feature Gaps Blocking Commercial Readiness

| ID | Gap | Severity |
|----|-----|----------|
| G1 | Zero narrative tutorial pages (vs SymPy's 3,845-line intro tutorial) | P0 |
| G2 | Examples are toy demos (42-line quickstart, 49-line robotics_codegen) | P0 |
| G3 | No end-to-end robotics workflow showing generated code *compiling and running* | P0 |
| G4 | No `symplex-build` example project (compilable `build.rs` + `main.rs`) | P1 |
| G5 | No ODE solving example | P1 |
| G6 | No complex numbers / Euler's formula example | P1 |
| G7 | No Laplace transform workflow example | P1 |
| G8 | IK example only tests axis-aligned targets, no numerical fallback | P1 |
| G9 | No SymPy vs symplex comparison document | P2 |
| G10 | IMPLEMENTATION_PLAN known limitations list has stale entries | P2 |

---

## Architecture: Five Phases, Strict Ordering

```text
Phase 1: Bug Fixes (codegen, display, clippy — 3 parallel agents)
    ↓ cargo test + cargo clippy gate (0 warnings)
Phase 2: Tutorial Writing (all new files — 4 parallel agents, no conflicts)
    ↓ cargo test gate (tutorial code snippets tested)
Phase 3: Example Rewrites (existing examples — 4 parallel agents, one per file)
    ↓ cargo test gate + cargo run --example gate
Phase 4: New Examples + Docs (all new files — 3 parallel agents)
    ↓ cargo test gate + cargo run --example gate
Phase 5: Verification & Doc Updates (IMPLEMENTATION_PLAN, CHANGELOG, README)
```

Each phase runs to completion before the next begins.
Within each phase, waves can run in parallel with **mutually exclusive file ownership**.

---

## Phase 1: Bug Fixes

### Wave 1A — Codegen Constant Propagation + Dead Code Elimination

**Agent owns exclusively:** `src/codegen.rs`, `tests/test_codegen_quality.rs` (NEW)

**Do NOT touch:** `src/cse.rs`, `src/display.rs`, `src/eval.rs`, any other file

**Bugs addressed:** B1, B2, B3, B4, B8

**Design:**

The root cause of B1 is that `to_rust_fn_with_options` and `matrix_to_rust_fn`
run CSE on the raw expression without first evaluating constant subexpressions.
The CSE pass correctly identifies shared nodes but treats literal `0` and `1`
as hoistable. The fix is a **post-CSE constant propagation pass** inserted into
both `to_rust_fn_with_options` and `matrix_to_rust_fn`:

1. **After CSE, before emission:** Walk the CSE bindings. For each binding,
   check if the value expression contains zero free variables (all leaves are
   `Num`, `Pi`, `E`, or other constants). If so, evaluate it to a float literal
   using `evalf_f64` or direct rational-to-float conversion. Replace the binding
   with an inline constant.

2. **Dead-binding elimination:** After inlining constant bindings, scan remaining
   bindings and final expressions. Any CSE symbol that is no longer referenced
   can be dropped.

3. **Zero-product elimination (B3):** In the emission of `Mul`, if any child
   evaluates to the literal `0.0`, emit `0.0` directly. In `Add`, drop
   zero-valued terms.

4. **Subtraction in codegen Add (B2):** Change the `Add` emission from a naive
   `parts.join(" + ")` to detect `Neg(inner)` children and emit `- inner` instead
   of `+ (-inner)`. Mirror the approach in `display.rs` lines 368-415.

5. **Parenthesization cleanup (B4):** Reduce double-wrapping. When emitting
   a single-child `Mul` or `Add`, omit outer parens. When a child is already
   parenthesized (e.g., a function call), don't add another layer.

6. **Fraction literals (B8):** For `Num` nodes where the rational has a small
   denominator (≤1000), emit the float literal directly (e.g., `-0.2_f64`)
   instead of `(-1_f64 / 5_f64)`. Only for denominators that produce exact
   IEEE 754 representations or within a configurable tolerance.

**Tests to add** (in `tests/test_codegen_quality.rs`):

- `codegen_no_trivial_cse_temps` — Generate 3-DOF planar Jacobian, assert
  output does NOT contain `0.0_f64.powi`, does NOT contain `1.0_f64;` as a
  let binding, does NOT contain `+ (-` pattern.
- `codegen_subtraction_style` — Expression `x + Neg(y)` produces `x - y`.
- `codegen_zero_mul_eliminated` — `0 * sin(x)` emits `0.0_f64`, not a sin call.
- `codegen_dead_binding_removed` — CSE binding only used in zero-product is dropped.
- `codegen_fraction_literal` — `1/5` emits `0.2_f64` not `(1_f64 / 5_f64)`.
- `codegen_3dof_planar_is_clean` — Full robotics pipeline: generate Jacobian
  for a planar 3-DOF arm, assert no `0.0`, `1.0` trivial temps, count total
  `let t` bindings is ≤ 8 (down from current ~12).
- `codegen_output_compiles` — Write generated code to a temp file, invoke
  `rustc --edition 2021` on it, assert exit code 0.

**Expected outcome:** `cargo run --example robotics_codegen` output is clean,
professional, no trivial constants, no `+ (-` patterns.

---

### Wave 1B — Display Edge Cases + Test Clippy Cleanup

**Agent owns exclusively:** `src/display.rs`, `tests/test_bareiss.rs`,
`tests/test_sympy_cross_validation.rs`, `tests/test_hensel.rs`,
`tests/test_workflows.rs`

**Do NOT touch:** `src/codegen.rs`, `src/cse.rs`, any other `src/` file

**Bugs addressed:** B5, B6a

**Tasks:**

1. **Verify Display `+ -N` fix (B5):** The display code at lines 401-411
   already handles negative numeric literals in Add. Write 5 targeted tests:
   - `display_sub_neg_int` — `x + (-3)` displays as `x - 3`
   - `display_sub_neg_frac` — `x + (-1/2)` displays as `x - 1/2`
   - `display_sub_neg_mul` — `x + -1*y` displays as `x - y`
   - `display_leading_neg` — `-3 + x` displays as `-3 + x` (correct)
   - `display_double_neg` — `-x + (-y)` displays as `-x - y`

   If any test reveals a remaining edge case, fix it in `display.rs`.

2. **Fix clippy warnings in test files (B6a):** Fix the specific warnings:
   - `test_bareiss.rs`: Replace `-1.0 * sub_det` with `-sub_det` (neg_multiply)
   - `test_sympy_cross_validation.rs`: Collapse nested `if let` + `if` into
     `if let ... &&` (collapsible_if)
   - `test_hensel.rs`: Fix any duplicated warnings
   - `test_workflows.rs`: Fix any duplicated warnings

3. **Verify `Matrix::to_latex()` exists and works (B7):** The method exists at
   `matrix.rs:985`. Verify the existing test passes. Add one more test:
   `matrix_to_latex_symbolic` — symbolic matrix LaTeX output is correct. If the
   `latex_output` example already exercises this, note it in WAVES.md as resolved.

**Expected outcome:** All display edge cases verified. Test-file clippy warnings
resolved. Known Limitation #17 can be removed.

---

### Wave 1C — Library Source Clippy Cleanup

**Agent owns exclusively:** All `src/*.rs` files **EXCEPT** `src/codegen.rs`
and `src/display.rs` (owned by 1A and 1B respectively).

Specifically, the files with warnings are:
- `src/canon.rs` (~4 warnings: collapsible_if)
- `src/eval.rs` (~5 warnings: collapsible_if with `let` chains)
- `src/evalf.rs` (~2 warnings: collapsible_if)
- `src/matrix.rs` (~8 warnings: loop variable indexing, returning let binding)
- `src/ntheory.rs` (~6 warnings: is_multiple_of, div_ceil, map_or, clone)
- `src/poly.rs` (~3 warnings: while_let, auto-borrow)
- `src/ode.rs` (~2 warnings: too many args, loop variable)
- `src/dynamics.rs` (~2 warnings: loop variable indexing)
- `src/groebner.rs` (~2 warnings: unwrap after is_some)
- `src/integrate.rs` (~2 warnings: collapsible_if)
- `src/latex.rs` (~1 warning: returning let binding)
- `src/control.rs` (~1 warning: function call in expect)
- `src/z_transform.rs` (~6 warnings: collapsible_if, returning let binding)
- `src/fourier_transform.rs` (~2 warnings: collapsible_if)
- `src/polysys.rs` (~1 warning: deref)
- `src/vector.rs` (~2 warnings: complex type)

**Do NOT touch:** `src/codegen.rs`, `src/display.rs`, any `tests/` file,
any `examples/` file.

**Bugs addressed:** B6b

**Tasks:**

These are all mechanical, safe refactors. No behavioral changes.

1. **Collapsible `if`** (~15 instances): Merge `if let Ok(x) = ... { if condition { ... } }`
   into `if let Ok(x) = ... && condition { ... }`.

2. **Returning let binding** (~6 instances): Replace `let result = expr; result` with
   just `expr`.

3. **Loop variable indexing** (~6 instances): Replace `for i in 0..n { arr[i] }` with
   `for item in &arr[..n]` or `for (i, item) in arr.iter().enumerate()`.

4. **`is_multiple_of` / `div_ceil`** (~2 instances in ntheory.rs): Use the standard
   library methods instead of manual reimplementation.

5. **`unwrap` after `is_some`** (~2 instances in groebner.rs): Replace
   `if x.is_some() { x.unwrap() }` with `if let Some(v) = x { v }`.

6. **Complex type** (~1 instance in vector.rs): Factor out a type alias.

7. **Other minor warnings**: `map_or` simplification, `from_ref` instead of `clone`,
   function call inside `expect`, auto-borrow deref.

**Verification:** After all fixes, run:
```sh
cargo clippy --lib -- -D warnings    # Must exit 0
```

**Expected outcome:** Zero clippy warnings on library source.

---

### Wave 1 Commit Gate

Waves 1A, 1B, and 1C run in parallel (mutually exclusive files), then verify:

```sh
cargo test --lib                          # 1,193+ pass
cargo test --all-targets                  # 1,855+ pass
cargo clippy --all-targets -- -D warnings # 0 warnings (was 60+)
cargo run --example robotics_codegen      # Clean output, no trivial temps
```

All four checks must pass before proceeding to Phase 2.

---

## Phase 2: Tutorial Writing

All files in this phase are **new** — they live under `docs/tutorial/`.
No existing file is modified. Agents can run fully in parallel.

### Wave 2A — Introduction + Getting Started

**Agent owns exclusively:**
- `docs/tutorial/01-introduction.md` (NEW, ~300 lines)
- `docs/tutorial/02-getting-started.md` (NEW, ~400 lines)

**Do NOT touch:** Any `src/`, `tests/`, or `examples/` file.

**Content outline for `01-introduction.md`:**

1. **What is symbolic computation?** — The `sqrt(8)` moment. Show that `f64`
   gives `2.8284...` but symplex gives `2*sqrt(2)`. Show that `0.1 + 0.2` in
   f64 is `0.30000000000000004` but in symplex is exactly `3/10`.
2. **A taste of symplex** — 10-line snippet: define `x`, differentiate
   `x³ - 3x² + 2x`, solve for critical points, evaluate numerically. Show the
   output. Each line annotated with a comment.
3. **Why symplex?** — Exact by default, type-safe, thread-safe, Rust-native
   code generation. Position against SymPy: "SymPy helps you understand the
   math. Symplex helps you *ship* the math."
4. **What symplex is NOT** — Not a numerical library (use `nalgebra`, `ndarray`).
   Not a plotting library. Not a SymPy replacement for number theory/geometry.
5. **Who is symplex for?** — Robotics engineers, control systems engineers,
   physics students, numerical algorithm developers. Common thread: derive
   formula symbolically → compile to fast code.

**Content outline for `02-getting-started.md`:**

1. **Installation** — `cargo add symplex`, MSRV, edition 2024.
2. **Your first expression** — `vars!`, `expr!`, building from operators.
3. **Inspecting expressions** — `Display`, `to_latex()`, `expr_type()`.
4. **Substitution** — `subs()`, `subs_i64()`, `subs_map_i64()`.
5. **Numerical evaluation** — `eval_f64()`, `eval_f64_with()`, `eval_decimal()`.
   When to use each. Contrast: `eval_decimal(30)` for π to 30 digits.
6. **The `compile()` function** — `compile()` → `Box<dyn Fn>` for fast loops.
   Example: evaluate sin(x) at 1000 points.
7. **Common patterns** — Method chaining, `_or_empty` / `_or_self` suffixes,
   `Result` vs convenience methods.

**Reference material:** SymPy's `intro.rst` (223 lines) and
`basic_operations.rst` (215 lines) for pedagogical structure.

---

### Wave 2B — Gotchas + Simplification

**Agent owns exclusively:**
- `docs/tutorial/03-gotchas.md` (NEW, ~250 lines)
- `docs/tutorial/04-simplification.md` (NEW, ~400 lines)

**Do NOT touch:** Any `src/`, `tests/`, or `examples/` file.

**Content outline for `03-gotchas.md`:**

1. **`expr!(1/2)` is a compile error** — Rust evaluates `1/2` as integer
   division → `0` before the macro sees it. Use `symplex::half()`,
   `symplex::rational(1, 2)`, or `expr!(1) / expr!(2)`. Show the error
   message, explain why, give the fix.
2. **`Ex` is `Clone`, not `Copy`** — Explain that `Ex` wraps an `Arc`.
   Show the `&` pattern: `&x + &y`. Explain that `expr!` macro hides this.
3. **Structural vs mathematical equality** — `x + 1 == 1 + x` is `true`
   (canonical form), but `(x+1)^2 == x^2 + 2x + 1` is `false`. Use
   `.equals()` or `.expand()` + `==`. Mirror SymPy's gotchas page.
4. **Variables must be declared** — `vars!(x)` before use. Show the error
   without it. Contrast with SymPy where `from sympy.abc import x` works.
5. **`^` is XOR in Rust** — In `expr!` macro, `^` means power. Outside the
   macro, use `.powi()` or `Pow`. Don't write `x ^ 2` outside `expr!`.
6. **Thread safety and the global context** — `symplex::var("x")` uses a
   global context. Safe across threads but variables from different contexts
   cannot be mixed. When to use `Context::new()`.
7. **`simplify()` is heuristic** — It tries multiple strategies and picks the
   "simplest" by `count_ops()`. It may not do what you expect. Use specific
   functions (`expand`, `factor`, `simplify_trig`) when you know what you want.
   Mirror SymPy's simplification caveat.

**Content outline for `04-simplification.md`:**

1. **`simplify()` and `simplify_full()`** — The catch-all. Show cases where
   it works and cases where it doesn't (e.g., `simplify(x^2 + 2x + 1)` stays
   expanded).
2. **`expand()`** — Distribute products, expand powers. Show `(x+1)^3`.
3. **`factor()`** — Factor polynomials. Show `x^3 - 1 = (x-1)(x^2+x+1)`.
4. **`simplify_trig()`** — Pythagorean identity, double-angle. 6 strategies.
5. **`simplify_powers()`** — `x^a * x^b → x^(a+b)`.
6. **`expand_trig()` / `expand_log()`** — Expansion of trig/log compositions.
7. **`log_combine()`** — Inverse of `expand_log()`.
8. **`partial_fractions()`** — Partial fraction decomposition.
9. **`simplify_rational()`** — Cancel + together pipeline.
10. **When to use what** — Decision tree flowchart in text form.

**Reference material:** SymPy's `simplification.rst` (912 lines) for scope and
pedagogical approach. Ours will be ~400 lines — focused on what symplex has.

---

### Wave 2C — Calculus + Solving

**Agent owns exclusively:**
- `docs/tutorial/05-calculus.md` (NEW, ~400 lines)
- `docs/tutorial/06-solving.md` (NEW, ~350 lines)

**Do NOT touch:** Any `src/`, `tests/`, or `examples/` file.

**Content outline for `05-calculus.md`:**

1. **Derivatives** — `diff()`, `diff_n()`, partial derivatives. Show chain rule
   and product rule working. Unevaluated `Derivative` for display.
2. **Higher-order and mixed partials** — `f.diff(&x).diff(&y)`.
3. **Integration** — `integrate()`, linearity, power rule, trig, exp, by-parts,
   u-substitution. Show what symplex CAN integrate and what it can't.
4. **Definite integrals** — `definite_integral()`. FTC verification.
5. **Limits** — `limit()`. Direct substitution, L'Hôpital, Gruntz for ∞.
   Show `sin(x)/x → 1` and `x*exp(-x) → 0` as x→∞.
6. **Series expansion** — `maclaurin()`, `series()`. Show sin(x) and exp(x).
7. **Dependency-aware differentiation** — `diff_with_dependent()` for implicit
   differentiation. Show `d/dx(x² + y²) = 2x + 2y·dy/dx`.

**Content outline for `06-solving.md`:**

1. **Polynomial equations** — Linear through quartic. Show exact Cardano/Ferrari.
2. **Transcendental equations** — `exp(x) = 5`, `sin(x) = 1/2`. Change-of-variable.
3. **Systems via Gröbner bases** — `solve_system()`. Circle-line intersection.
   Explain that only rational roots are found (limitation #20).
4. **Numerical solving** — `solve_numeric()` (Newton's method). Show as fallback
   for transcendental equations.
5. **Inequality solving** — `solve_gt()`, `solve_le()` returning `SetEx`.
6. **ODE solving** — `solve_ode()`. Separable, linear, 2nd-order CC.
   `check_ode_solution()` for verification.
7. **Verified solving** — `solve_verified()`. Solve + filter by back-substitution.
8. **What symplex can't solve** — No Risch integration, no LambertW solver,
   no PDE. Be honest. Point to `solve_numeric()` as fallback.

**Reference material:** SymPy's `solvers.rst` (247 lines) + 11 solving guides
(3,169 lines). Ours will be ~350 lines covering the essentials.

---

### Wave 2D — Matrices + Code Generation (THE KILLER PAGE)

**Agent owns exclusively:**
- `docs/tutorial/07-matrices.md` (NEW, ~300 lines)
- `docs/tutorial/08-code-generation.md` (NEW, ~500 lines)

**Do NOT touch:** Any `src/`, `tests/`, or `examples/` file.

**Content outline for `07-matrices.md`:**

1. **Construction** — `matrix!` macro, `Matrix::from_i64()`, `Matrix::diag()`.
2. **Basic operations** — Add, multiply, transpose, scale.
3. **Determinant and inverse** — `det()`, `inv()`. Show 2×2 and 3×3.
4. **Eigenvalues and characteristic polynomial** — `eigenvals()`, `char_poly()`.
5. **Decompositions** — LU, QR, Cholesky. RREF, rank, nullspace.
6. **Symbolic matrices** — Matrix with `Ex` entries. Differentiate each entry.
7. **Jacobian** — `jacobian(&[&f1, &f2], &[&x, &y])`. This is the bridge to
   the code generation chapter.
8. **LaTeX output** — `m.to_latex()`. Show the bmatrix output.

**Content outline for `08-code-generation.md` (THE KILLER PAGE):**

This is the page that sells symplex. It must be perfect.

1. **The vision** — "Derive a formula symbolically. Generate optimized Rust code.
   Compile it. Run it in production. No Python. No glue scripts. One toolchain."
2. **Simple example** — Generate a function for `sin(x)^2 + cos(x)^2` (should
   constant-fold to `1.0_f64`). Show the before/after of CSE.
3. **The robotics workflow** — Step by step:
   a. Define DH parameters for a 3-DOF planar arm
   b. Compute FK symbolically (show the symbolic expression)
   c. Compute Jacobian symbolically
   d. Generate Rust code with `to_rust_fn()`
   e. **Show the generated code** — clean, no trivial constants
   f. **Call the generated code** — paste it into a program, compute values
   g. **Verify** — compare against numerical FK at same joint angles
4. **Code generation options** — `CodegenOptions::default()`, `::no_std()`,
   `::embedded_f32()`. Math backends: Std, Libm, CfgGated.
5. **Matrix code generation** — `Matrix::to_rust_fn()` for flat-array output.
   Cross-entry CSE across all matrix elements.
6. **The `symplex-build` pipeline** — `build.rs` integration. Show the workflow:
   define robot in build.rs → symplex derives equations → generated .rs file in
   OUT_DIR → firmware crate `include!`s it → cross-compile to target.
7. **What the generated code looks like** — Annotated walkthrough of real output.
   Point out CSE variables, explain why they exist.
8. **Limitations** — No complex expressions, no special functions in codegen,
   no SIMD, no fixed-point. Be honest.

**Reference material:** Nothing in SymPy matches this. This is our unique value
proposition. It must be the best-written page in the entire project.

---

### Wave 2 Commit Gate

```sh
# Verify all tutorial code snippets work
# (Each tutorial should include testable examples;
#  we verify these in Phase 5 with a doc-test script)
ls docs/tutorial/*.md | wc -l  # Should be 8
wc -l docs/tutorial/*.md       # Should total ~2,900 lines
```

---

## Phase 3: Example Rewrites

Each agent owns **specific example files** — no overlaps.
Agents read `src/` files for reference but do NOT modify them.

### Wave 3A — Quickstart + Calculus

**Agent owns exclusively:**
- `examples/quickstart.rs`
- `examples/calculus.rs`

**Do NOT touch:** Any other example file, any `src/` file, any `tests/` file.

**`quickstart.rs` rewrite (42 → 120+ lines):**

- Add section headers as comments (`// ── Expressions ──`, `// ── Calculus ──`, etc.)
- Add narrative comments explaining what each section demonstrates
- Add: partial derivatives, trig simplification, factoring, equation solving,
  matrix operations, LaTeX output, numerical evaluation, `compile()` for fast eval
- End with: "For the full tutorial, see docs/tutorial/"
- Every section should have a `println!` showing both the operation and result
- Add timing for the Jacobian computation to show performance

**`calculus.rs` rewrite (121 → 200+ lines):**

- Add: ODEs (`solve_ode` for y'' + y = 0), Laplace transform, Fourier series
- Add: implicit differentiation example (`diff_with_dependent`)
- Add: convergence testing
- Add: more integration examples (by-parts: ∫ x·eˣ, trig: ∫ sin³x)
- Add narrative comments explaining each technique
- Add numerical verification of results (evaluate at a point, compare)

---

### Wave 3B — Robotics Codegen + Dynamics

**Agent owns exclusively:**
- `examples/robotics_codegen.rs`
- `examples/dynamics.rs`

**Do NOT touch:** Any other example file, any `src/` file, any `tests/` file.

**`robotics_codegen.rs` rewrite (49 → 250+ lines):**

This is **the most important example**. It must be perfect.

- Show the full pipeline end-to-end:
  1. Define a 3-DOF planar arm with DH parameters (with comments explaining DH)
  2. Compute FK, print the symbolic position expressions
  3. Compute Jacobian, print its symbolic entries
  4. Generate Rust code with CSE
  5. Print the generated code
  6. **Manually define the same function inline and call it** to verify correctness
  7. Print numerical values at specific joint angles
  8. Compare symbolic FK vs numerical FK
  9. Show timing for each step
- Add a section computing Lagrangian dynamics for the same arm
- Add a section showing `CodegenOptions::no_std()` for embedded targets
- Heavily commented — a robotics engineer should be able to follow every step

**`dynamics.rs` rewrite (58 → 180+ lines):**

- Expand to **double pendulum** (2-DOF, coupled equations of motion)
- Show: kinetic energy, potential energy, Lagrangian
- Derive: Euler-Lagrange equations, mass matrix, Coriolis matrix, gravity vector
- Verify: mass matrix is symmetric, M(q) is positive definite at a test point
- Numerical evaluation: compute torques at a specific configuration
- Compare single vs double pendulum (show the complexity increase)
- Add narrative comments explaining the physics

---

### Wave 3C — Control System + Equation Solving

**Agent owns exclusively:**
- `examples/control_system.rs`
- `examples/equation_solving.rs`

**Do NOT touch:** Any other example file, any `src/` file, any `tests/` file.

**`control_system.rs` rewrite (62 → 180+ lines):**

- Start from physics: describe the mass-spring-damper system
- Derive state-space model from the equation mẍ + cẋ + kx = F
- Show: eigenvalues (poles), stability analysis, controllability, observability
- Transfer function: derive from state-space, show DC gain
- Routh-Hurwitz stability criterion
- Ackermann pole placement: design a controller
- ZOH discretization: convert to discrete-time
- Laplace transform: show forward and inverse
- Add: a second system (e.g., DC motor, inverted pendulum) for comparison
- Narrative comments throughout

**`equation_solving.rs` rewrite (58 → 150+ lines):**

- Add: transcendental equations (exp, trig, sqrt)
- Add: change-of-variable solving
- Add: complex roots with explanation
- Add: inequality solving returning SetEx
- Add: numerical solving fallback for cos(x) = x
- Add: system solving with explanation of Gröbner bases
- Add: verified solving (solve_verified)
- Show limitations honestly: "This system has irrational roots; the algebraic
  solver returns empty. Use solve_numeric() instead."

---

### Wave 3D — Matrix Algebra + Inverse Kinematics

**Agent owns exclusively:**
- `examples/matrix_algebra.rs`
- `examples/inverse_kinematics.rs`

**Do NOT touch:** Any other example file, any `src/` file, any `tests/` file.

**`matrix_algebra.rs` rewrite (48 → 150+ lines):**

- Add: LU decomposition with pivots
- Add: QR decomposition
- Add: RREF, rank, nullspace
- Add: Kronecker product
- Add: Matrix exponential
- Add: Jacobian computation
- Add: Larger matrix (3×3 or 4×4) for more interesting eigenvalues
- Add: Code generation from a matrix (`to_rust_fn`)
- Add: LaTeX output for matrices

**`inverse_kinematics.rs` rewrite (54 → 150+ lines):**

This example must be **honest about limitations**.

- Keep: axis-aligned targets that work exactly
- **Add: generic target** (e.g., `(1.0, 0.5)`) — show that the algebraic solver
  may return empty for irrational angles
- **Add: numerical fallback** — use `solve_numeric` or `atan2` to find angles
  numerically when algebraic solver can't
- **Add: workspace analysis** — which targets are reachable?
- **Add: explanation** of why Gröbner-based IK finds only rational roots
- **Add: comparison** — show algebraic solution (exact) vs numerical (approximate)
- Narrative comments explaining the robotics context

---

### Wave 3 Commit Gate

```sh
cargo test --all-targets                    # All pass
cargo clippy --all-targets -- -D warnings   # 0 warnings
# Run every example and verify output:
for ex in quickstart calculus robotics_codegen dynamics control_system \
          equation_solving matrix_algebra inverse_kinematics \
          optimization solve_system latex_output number_theory; do
    cargo run --example $ex || exit 1
done
```

---

## Phase 4: New Examples + Comparison Doc

All files in this phase are **new** — no conflicts with existing files.

### Wave 4A — ODE Solving + Complex Numbers

**Agent owns exclusively:**
- `examples/ode_solving.rs` (NEW, ~120 lines)
- `examples/complex_numbers.rs` (NEW, ~100 lines)

**Do NOT touch:** Any existing example, any `src/` file, any `tests/` file.

**`ode_solving.rs`:**
- Separable ODE: dy/dx = xy → y = Ce^(x²/2)
- First-order linear: y' + 2y = e^(-x)
- Second-order constant-coefficient: y'' + 9y = 0 → C₁sin(3x) + C₂cos(3x)
- Verify each solution with `check_ode_solution()`
- Classify each ODE with `classify_ode()`
- Numerical evaluation of solutions at specific points

**`complex_numbers.rs`:**
- Euler's formula: e^(iπ) = -1
- Complex quadratic: x² + 1 = 0 → [i, -i]
- Complex quadratic: x² + 2x + 5 = 0 → [-1+2i, -1-2i]
- Trig-hyperbolic bridge: sin(ix) = i·sinh(x)
- Complex numerical evaluation: `eval_complex64()`
- `re()`, `im()`, `conjugate()`, `arg()`

---

### Wave 4B — Laplace Transforms + Embedded Demo

**Agent owns exclusively:**
- `examples/laplace_transforms.rs` (NEW, ~120 lines)
- `examples/embedded_demo/Cargo.toml` (NEW)
- `examples/embedded_demo/build.rs` (NEW)
- `examples/embedded_demo/src/main.rs` (NEW)

**Do NOT touch:** Any existing example, any `src/` file in main crate.

**`laplace_transforms.rs`:**
- Forward Laplace: L{sin(t)} = 1/(s²+1)
- Forward Laplace: L{t·e^(-at)} = 1/(s+a)²
- Inverse Laplace via partial fractions
- Transfer function derivation from ODE
- Verify: inverse of forward gives back original
- Z-transform example for discrete systems

**`examples/embedded_demo/`:**

A complete, self-contained Cargo project that demonstrates the `symplex-build`
pipeline:

```
examples/embedded_demo/
├── Cargo.toml          # [build-dependencies] symplex-build
├── build.rs            # Derive 2-DOF Jacobian, emit robot_math.rs
├── src/
│   └── main.rs         # include! the generated code, call it, print results
└── README.md           # How to run: cd examples/embedded_demo && cargo run
```

`Cargo.toml` declares:
```toml
[package]
name = "embedded-demo"
edition = "2021"

[build-dependencies]
symplex = { path = "../.." }
symplex-build = { path = "../../symplex-build" }
```

`build.rs`:
- Define 2-DOF planar arm with DH parameters
- Compute FK and Jacobian symbolically
- Use `CodeGen::new().add_matrix_fn(...)` to generate code
- Write to `OUT_DIR/robot_math.rs`

`src/main.rs`:
- `include!(concat!(env!("OUT_DIR"), "/robot_math.rs"));`
- Call the generated `jacobian(theta1, theta2)` function
- Print results at several joint configurations
- Verify against manually computed values

---

### Wave 4C — Comparison Doc + Tutorial Index

**Agent owns exclusively:**
- `docs/sympy-comparison.md` (NEW, ~400 lines)
- `docs/tutorial/index.md` (NEW, ~50 lines)
- `docs/README.md` (NEW, ~30 lines)

**Do NOT touch:** Any `src/`, `tests/`, or `examples/` file.

**`docs/sympy-comparison.md`:**

Side-by-side worked examples showing the same task in SymPy and symplex:

| Task | SymPy | symplex |
|------|-------|---------|
| Define symbols | `x, y = symbols('x y')` | `vars!(x, y)` |
| Differentiate | `diff(sin(x)*exp(x), x)` | `expr!(sin(x) * exp(x)).diff(&x)` |
| Integrate | `integrate(x**2, x)` | `expr!(x^2).integrate(&x)` |
| Solve | `solve(x**2 - 2, x)` | `expr!(x^2 - 2).solve_or_empty(&x)` |
| Matrix | `Matrix([[1,2],[3,4]])` | `matrix![[1,2],[3,4]]` |
| Eigenvalues | `M.eigenvals()` | `m.eigenvals(&x)` |
| LaTeX | `latex(expr)` | `expr.to_latex()` |
| Numerical eval | `expr.evalf()` | `expr.eval_decimal(15)` |
| Code generation | N/A (no Rust codegen) | `expr.to_rust_fn("f", &["x"])` |
| Thread safety | ❌ (GIL) | ✅ (Send + Sync) |
| Type safety | ❌ (duck typing) | ✅ (Ex vs BoolEx vs SetEx) |

Honest about what SymPy does better:
- Risch integration algorithm
- Number theory depth
- Geometry module
- Statistics
- Tensor calculus
- 20+ years of community, thousands of contributors

Where symplex wins:
- Compile-time type safety
- Zero-GIL parallelism
- Native Rust code generation with CSE
- build.rs integration for embedded
- Arena hash-consing with O(1) equality
- No Python runtime required

**`docs/tutorial/index.md`:** Table of contents linking all 8 tutorial pages.

**`docs/README.md`:** Points to tutorial, comparison, and docs.rs.

---

### Wave 4 Commit Gate

```sh
cargo test --all-targets                    # All pass
cargo clippy --all-targets -- -D warnings   # 0 warnings
# Run all examples including new ones:
for ex in quickstart calculus robotics_codegen dynamics control_system \
          equation_solving matrix_algebra inverse_kinematics \
          optimization solve_system latex_output number_theory \
          ode_solving complex_numbers laplace_transforms; do
    cargo run --example $ex || exit 1
done
# Build the embedded demo:
cd examples/embedded_demo && cargo run
```

---

## Phase 5: Verification & Doc Updates

Sequential. One agent. All remaining files.

### Wave 5A — Update Project Documentation

**Agent owns exclusively:**
- `IMPLEMENTATION_PLAN.md` (update Known Limitations, statistics)
- `CHANGELOG.md` (add entries for all fixes and new docs)
- `README.md` (update example count, verify hero example output, add link to tutorial)

**Tasks:**

1. **Update Known Limitations in IMPLEMENTATION_PLAN.md:**
   - Remove #17 (Display `+ -N`) — fixed in Wave 1B
   - Remove #18 (Matrix::to_latex missing) — already exists
   - Update #19 (Codegen constant folding) — fixed in Wave 1A
   - Update example count (13 → 16+)
   - Update line counts

2. **Update CHANGELOG.md:**
   - Add: "Fixed: Codegen constant propagation — trivial CSE temps eliminated"
   - Add: "Fixed: Codegen subtraction style — `+ (-x)` now renders as `- x`"
   - Add: "Fixed: Codegen dead-code elimination for zero-multiplied terms"
   - Add: "Fixed: All clippy warnings on test targets"
   - Add: "Added: 8-page narrative tutorial (docs/tutorial/)"
   - Add: "Added: SymPy comparison document (docs/sympy-comparison.md)"
   - Add: "Added: 3 new examples (ode_solving, complex_numbers, laplace_transforms)"
   - Add: "Added: Embedded demo project (examples/embedded_demo/)"
   - Add: "Rewritten: All 10 existing examples with expanded coverage and narrative"

3. **Update README.md:**
   - Verify hero example output matches actual `cargo run --example robotics_codegen`
   - Add link to tutorial: "📖 **[Tutorial](docs/tutorial/index.md)** — Start here"
   - Update example list with new examples
   - Update comparison table if any entries changed

4. **Re-run SymPy cross-validation:**
   ```sh
   cargo test test_sympy_cross_validation -- --nocapture
   ```
   Record any newly passing fixtures. Update CHANGELOG if count changed.

5. **Run benchmarks:**
   ```sh
   cargo bench
   ```
   Record baseline. Flag any regressions > 20% vs last known good.

---

### Wave 5 Final Gate

```sh
# Everything must pass:
cargo test --all-targets                    # 1,900+ pass
cargo test --doc                            # Doc tests pass
cargo clippy --all-targets -- -D warnings   # 0 warnings
cargo bench                                 # No major regressions

# All examples run cleanly:
for ex in quickstart calculus robotics_codegen dynamics control_system \
          equation_solving matrix_algebra inverse_kinematics \
          optimization solve_system latex_output number_theory \
          ode_solving complex_numbers laplace_transforms; do
    echo "=== $ex ===" && cargo run --example $ex || exit 1
done

# Embedded demo builds and runs:
cd examples/embedded_demo && cargo run

# Documentation exists:
test -f docs/tutorial/index.md
test -f docs/tutorial/08-code-generation.md
test -f docs/sympy-comparison.md
ls docs/tutorial/*.md | wc -l  # 9 (index + 8 chapters)
wc -l docs/tutorial/*.md       # ~2,950 lines total
```

---

## Summary: File Ownership Matrix

| File | Phase | Wave | Agent |
|------|-------|------|-------|
| `src/codegen.rs` | 1 | 1A | Agent 1A only |
| `tests/test_codegen_quality.rs` (NEW) | 1 | 1A | Agent 1A only |
| `src/display.rs` | 1 | 1B | Agent 1B only |
| `tests/test_bareiss.rs` | 1 | 1B | Agent 1B only |
| `tests/test_sympy_cross_validation.rs` | 1 | 1B | Agent 1B only |
| `tests/test_hensel.rs` | 1 | 1B | Agent 1B only |
| `tests/test_workflows.rs` | 1 | 1B | Agent 1B only |
| `src/canon.rs` | 1 | 1C | Agent 1C only |
| `src/eval.rs` | 1 | 1C | Agent 1C only |
| `src/evalf.rs` | 1 | 1C | Agent 1C only |
| `src/matrix.rs` | 1 | 1C | Agent 1C only |
| `src/ntheory.rs` | 1 | 1C | Agent 1C only |
| `src/poly.rs` | 1 | 1C | Agent 1C only |
| `src/ode.rs` | 1 | 1C | Agent 1C only |
| `src/dynamics.rs` | 1 | 1C | Agent 1C only |
| `src/groebner.rs` | 1 | 1C | Agent 1C only |
| `src/integrate.rs` | 1 | 1C | Agent 1C only |
| `src/latex.rs` | 1 | 1C | Agent 1C only |
| `src/control.rs` | 1 | 1C | Agent 1C only |
| `src/z_transform.rs` | 1 | 1C | Agent 1C only |
| `src/fourier_transform.rs` | 1 | 1C | Agent 1C only |
| `src/polysys.rs` | 1 | 1C | Agent 1C only |
| `src/vector.rs` | 1 | 1C | Agent 1C only |
| `docs/tutorial/01-introduction.md` (NEW) | 2 | 2A | Agent 2A only |
| `docs/tutorial/02-getting-started.md` (NEW) | 2 | 2A | Agent 2A only |
| `docs/tutorial/03-gotchas.md` (NEW) | 2 | 2B | Agent 2B only |
| `docs/tutorial/04-simplification.md` (NEW) | 2 | 2B | Agent 2B only |
| `docs/tutorial/05-calculus.md` (NEW) | 2 | 2C | Agent 2C only |
| `docs/tutorial/06-solving.md` (NEW) | 2 | 2C | Agent 2C only |
| `docs/tutorial/07-matrices.md` (NEW) | 2 | 2D | Agent 2D only |
| `docs/tutorial/08-code-generation.md` (NEW) | 2 | 2D | Agent 2D only |
| `examples/quickstart.rs` | 3 | 3A | Agent 3A only |
| `examples/calculus.rs` | 3 | 3A | Agent 3A only |
| `examples/robotics_codegen.rs` | 3 | 3B | Agent 3B only |
| `examples/dynamics.rs` | 3 | 3B | Agent 3B only |
| `examples/control_system.rs` | 3 | 3C | Agent 3C only |
| `examples/equation_solving.rs` | 3 | 3C | Agent 3C only |
| `examples/matrix_algebra.rs` | 3 | 3D | Agent 3D only |
| `examples/inverse_kinematics.rs` | 3 | 3D | Agent 3D only |
| `examples/ode_solving.rs` (NEW) | 4 | 4A | Agent 4A only |
| `examples/complex_numbers.rs` (NEW) | 4 | 4A | Agent 4A only |
| `examples/laplace_transforms.rs` (NEW) | 4 | 4B | Agent 4B only |
| `examples/embedded_demo/**` (NEW) | 4 | 4B | Agent 4B only |
| `docs/sympy-comparison.md` (NEW) | 4 | 4C | Agent 4C only |
| `docs/tutorial/index.md` (NEW) | 4 | 4C | Agent 4C only |
| `docs/README.md` (NEW) | 4 | 4C | Agent 4C only |
| `IMPLEMENTATION_PLAN.md` | 5 | 5A | Agent 5A only |
| `CHANGELOG.md` | 5 | 5A | Agent 5A only |
| `README.md` | 5 | 5A | Agent 5A only |

**Files NEVER touched by any wave (read-only reference):**
- `src/cse.rs` (codegen fix works post-CSE, doesn't modify CSE itself)
- `src/arena.rs`, `src/node.rs`, `src/walk.rs`, `src/sort_key.rs` (core infrastructure)
- `src/expr.rs`, `src/expr_funcs.rs`, `src/expr_ops.rs`, `src/expr_view.rs` (public API)
- `src/diff.rs`, `src/solve.rs`, `src/simplify_engine.rs`, `src/pattern.rs` (transforms)
- `src/assumptions.rs`, `src/config.rs`, `src/errors.rs`, `src/symbol.rs` (types layer)
- All other `src/*.rs` files not assigned to Wave 1C
- `TEAM.md`, `CONTRIBUTING.md`, `Cargo.toml`, `LICENSE-*`
- All existing `tests/` files not listed above
- `symplex-macros/`, `symplex-build/` (except embedded_demo references it)
- `symplex-wasm/`

---

## Estimated Effort

| Phase | Waves | Parallel Agents | Estimated Time |
|-------|-------|-----------------|----------------|
| 1: Bug Fixes | 1A + 1B + 1C | 3 | 4-6 hours |
| 2: Tutorial | 2A + 2B + 2C + 2D | 4 | 6-10 hours |
| 3: Example Rewrites | 3A + 3B + 3C + 3D | 4 | 6-8 hours |
| 4: New Examples + Docs | 4A + 4B + 4C | 3 | 4-6 hours |
| 5: Verification | 5A | 1 | 2-3 hours |
| **Total** | **15 waves** | **max 4 parallel** | **22-33 hours** |

---

## Success Criteria

When all phases complete, the following must be true:

1. **`cargo run --example robotics_codegen`** produces generated code with:
   - Zero trivial constant temps (`0.0_f64`, `1.0_f64`)
   - Clean subtraction (no `+ (-x)` patterns)
   - ≤8 CSE temps for a 3-DOF planar arm
   - Code that compiles with `rustc`

2. **`docs/tutorial/`** contains 8 chapters + index totaling ~2,900+ lines

3. **13 rewritten examples + 3 new examples = 16 examples** totaling 2,500+ lines

4. **`examples/embedded_demo/`** builds and runs standalone

5. **`cargo clippy --all-targets -- -D warnings`** exits 0

6. **All 1,855+ tests pass**

7. **No new Known Limitations introduced**

8. **README hero example output matches actual `cargo run` output**

---

## Principles

1. **No new features.** This plan adds zero new `src/` capabilities. All work is
   fixes, documentation, and examples.

2. **Mutually exclusive file ownership.** No two agents touch the same file in
   the same phase. Verified by the ownership matrix above.

3. **Phase barriers.** Each phase completes and passes its gate before the next
   begins. No skipping.

4. **Don't weaken tests.** If a test fails after a codegen fix, fix the codegen,
   not the test.

5. **Every code snippet in tutorials must work.** If it appears in a `.md` file,
   it must compile and produce the output shown. Phase 5 verifies this.

6. **Page 8 is the product.** `08-code-generation.md` is the most important
   document in the project. It gets the most attention, the most review, and
   the most testing.