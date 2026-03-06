# Symplex Implementation Plan

> **This is the single onboarding document.** A new developer arriving in a fresh
> context window should be able to start productive work from reading this file alone.

---

## 1. Quick Start for New Developers

symplex is a symbolic mathematics library for Rust. It provides an arena-interned
expression DAG with hash-consing, canonical ordering, differentiation, integration,
equation solving, matrix algebra, Laplace transforms, and code generation.

### Build and test

    cargo test          # 4,370+ tests — all must pass
    cargo clippy        # 0 warnings required (1 pre-existing dead_code warning in sturm.rs)
    cargo bench         # Criterion benchmarks (27)

### Architecture: six layers (each layer only calls downward)

    Layer 1: Public API      — expr.rs, expr_funcs.rs, expr_ops.rs, context.rs, eq.rs, matrix.rs
    Layer 2: Macros           — symplex-macros/ (expr!, rule!, matrix!, eq!)
    Layer 3: Transforms       — diff, integrate, expand, eval, simplify, solve, laplace, …
    Layer 4: Canonicalization — canon.rs (add/mul/pow/neg/interval/union/intersection)
    Layer 5: Arena            — arena.rs, node.rs, walk.rs, sort_key.rs, symbol.rs
    Layer 6: Types            — assumptions.rs, config.rs, errors.rs

### Key files to read first

    1. src/node.rs        — ExprNode enum (66 variants), the core data model
    2. src/expr.rs         — Expr<S> struct, Sort system, type aliases
    3. src/expr_funcs.rs   — all 175+ public methods on Ex
    4. src/arena.rs        — hash-consing, intern(), constructors
    5. src/canon.rs        — canonicalization rules (what constructors do)

### Conventions (see §4 for full list)

- No names in source code — see TEAM.md for the no-names policy
- `#[must_use]` on all pure queries and transformations
- Tests: minimum 3 per feature (happy, edge, error)
- Clippy: 0 warnings required

---

## 2. Current Architecture

### Codebase at a glance

| Metric | Value |
|--------|-------|
| Tests | 4,370 passing, 0 failing |
| Source | ~60,400 lines across 76 modules |
| Tests | ~49,000 lines across 125 test files |
| Examples | ~445 lines across 8 examples |
| Companion crates | ~1,045 lines across 2 crates (symplex-build, symplex-wasm) |
| Total | ~111,000 lines |
| ExprNode variants | 66 (including 7 set-valued, 11 boolean) |
| Public methods on `Ex` | 175+ (numeric + boolean + set-valued) |
| Public methods on `Context` | 17 |
| Matrix methods | 50+ (added operators, exp, kronecker, cholesky, pinv, diag, from_i64) |
| Apply functions | 12 + 9 special functions (Bessel, Legendre, etc.) |
| `expr!` macro functions | 65 (54 single-arg + 11 multi-arg) |
| Simplification rules | 24 (with condition guards) |
| Integration forms | 40+ |
| Eval special values | 86+ |
| Criterion benchmarks | 27 |
| Proptest properties | 22+ |
| Source modules | 76 |
| Test files | 125 |
| Examples | 8 |
| Companion crates | 2 (symplex-build, symplex-wasm) |

### Three phantom sorts

```text
Expr<S: Sort>     ← one generic struct, PhantomData<S>, zero runtime cost
  ├── Expr<Numeric>    (type alias: Ex)       — arithmetic, calculus, 150+ methods
  ├── Expr<Boolean>    (type alias: BoolEx)   — comparisons, logic, 12 methods
  └── Expr<SetValued>  (type alias: SetEx)    — intervals, unions, 10+ methods

Cross-sort bridges:
  Ex::gt(&Ex) → BoolEx                       — numeric comparison produces boolean
  Ex::piecewise(&[(&Ex, &BoolEx)]) → Ex      — boolean conditions, numeric values
  Ex::solveset(&Ex) → SetEx                  — solve returning set
  Ex::solve_gt(&Ex) → SetEx                  — inequality solving
```

### Key design patterns

1. **Arena hash-consing** — Every expression is an `ExprId(u32)` index into the arena.
   Structurally identical expressions share the same ExprId. O(1) equality, O(1) hashing.

2. **Canonical form** — Construction always canonicalizes: `Add` children sorted,
   like-terms merged, zeros dropped. `Mul` similar. `Pow(x,0)→1`, `Pow(x,1)→x`.
   `i^n` reduced mod 4. `(-1)^(1/2)→i`. Number×Add distributed.

3. **Phantom-typed sorts** — `Expr<S: Sort>` carries a compile-time sort marker.
   `sin(bool_expr)` is a compile error. Zero runtime cost.

4. **Iterative walks** — All tree traversals use explicit stacks via `walk.rs`.
   No recursion. Stack-safe for arbitrarily deep expressions.

5. **Arena compact()** — Generational GC. Traces live nodes via BitVec, copies
   to a new arena, remaps all ExprIds. Called when arena grows too large.

6. **Multinomial theorem for expand** — `(a+b+c)^n` expanded via incremental
   cross-multiplication, not recursive distribution.

7. **Bareiss for symbolic determinant** — Integer-preserving Gaussian elimination
   for determinants. No division until the final step. LU-based for large matrices.

8. **Stripper-collector for pattern matching** — Classify pattern subterms as
   "strippers" (structurally specific) vs "collectors" (bare wilds absorbing residual).
   O(k·n) for k-term patterns against n-term Add/Mul nodes.

9. **eval-before-evalf pipeline** — `evalf()` calls `eval()` first to reduce exact
   values before numerical approximation. CAS principle: symbolic first, numerical last.

10. **Stirling series for arbitrary-precision Gamma** — Not Lanczos (doesn't scale
    beyond f64). Stirling uses 0.323p terms (fewest), Bernoulli coefficients are
    cacheable, extends to log Γ, ψ, and polygamma.

---

## 3. How to Add Things

### 3.1 Adding a new ExprNode variant (14-file checklist)

1. Add variant to `ExprNode` enum in `src/node.rs`
2. Add `children()` arm in `node.rs`
3. Add `for_each_child()` arm in `node.rs`
4. Add `SortKey` generation in `src/sort_key.rs`
5. Add walk/rebuild arm in `src/walk.rs`
6. Add display formatting in `src/display.rs`
7. Add differentiation rule in `src/diff.rs`
8. Add eval special values in `src/eval.rs`
9. Add evalf numerical evaluation in `src/evalf.rs`
10. Add ExprTree variant in `src/tree.rs` (both to_tree and from_tree)
11. Add expand handling in `src/expand.rs`
12. Add pattern matching arm in `src/pattern.rs`
13. Add polybridge handling in `src/polybridge.rs`
14. Add lambdify case in `src/lambdify.rs`

> The compiler catches every miss — add the variant, then fix all `match` errors.
> Run `cargo test` after each file to catch regressions incrementally.

### 3.2 Adding a new Apply function (4-edit checklist)

Apply functions are for integer-only sequences that need no calculus rules.

1. Add `pub(crate) const FN_NAME: &str = "name"` in `src/arena.rs`
2. Add a constructor method in `src/arena.rs` using the constant
3. Add eval dispatch in `src/eval.rs` matching on the constant
4. Add a convenience method on `Ex` in `src/expr_funcs.rs`

### 3.3 Adding a new simplification rule

1. Choose the identity (e.g., `sin(2w) → 2·sin(w)·cos(w)`)
2. If it fits `outer(inner(w)) → template(w)`, use `unary_compose_rule()`:
   ```
   unary_compose_rule(arena, "my_rule", Arena::sin, Arena::asin, identity)
   ```
3. Otherwise, build manually: `fn rule_xxx(arena: &mut Arena) -> Rule`
4. Add to `basic_rules()` in `src/pattern.rs`
5. Add condition guard if needed: `r.condition = Some(|arena, bindings| { ... })`
6. Add positive test, negative test, and numerical validation test

> **Note:** The `rule!` macro CANNOT be used inside `src/` files (it generates
> `::symplex::` paths that don't resolve within the crate). Use `unary_compose_rule()`
> or manual `Rule::new()` for internal rules.

### 3.4 Adding a new integration form

1. In `src/integrate.rs`, find `integrate_node()`
2. Add a match arm for the new form
3. Return the antiderivative as an `ExprId`
4. If the form needs u-substitution, use `linear_coeff_of()` and existing machinery
5. For trig power forms, add to `src/trig_integ.rs`
6. Add test + SymPy cross-validation fixture

### 3.5 Adding a function to `expr!` macro

1. In `symplex-macros/src/lib.rs`, find the single-arg function match block
2. Add `"name" => quote! { name },` (the method name on `Ex`)
3. For multi-arg functions, add special handling above the match block
4. Add a test in `tests/test_macros.rs`

### 3.6 Adding a function to `rule!` macro

The `rule!` macro shares the same Pratt parser as `expr!`. Functions recognized
by `rule!` are the same as `expr!` single-arg functions. To add a new one:

1. Add the function name to the match arm in the `rule!` code generation section
   of `symplex-macros/src/lib.rs` (look for `Arena::sin` pattern)
2. The function name must match an `Arena` method name
3. Add a test: `rule!(arena, "test", new_fn(w_) => w_)`

---

## 4. Development Conventions

### Code quality

- **No names in source code** — Never put personal names, author attributions, or
  team member references in source files, tests, comments, or doc comments. Use
  `git blame` for attribution. See TEAM.md for the policy.
- **`#[must_use]`** on all pure queries and transformations
- **`#![warn(missing_docs)]`** — concise, precise, no superlatives
- **`expr!` macro updated in same commit** as any feature it should support
- **Tests: minimum 3 per feature** — happy path, edge case, error/empty case
- **Clippy: 0 warnings required** — no `#[allow(clippy::...)]` without justification
- **Proptest** for algebraic invariants on new transformations

### Architecture boundaries

- **Apply vs ExprNode** — ExprNode for calculus-capable functions (need diff/integrate
  rules). Apply for evaluation-only integer sequences (fibonacci, bernoulli, etc.).
  This keeps the ExprNode enum lean.
- **eval-before-evalf** — `evalf()` calls `eval()` first to reduce exact values
  before numerical approximation. Never skip the symbolic reduction step.
- **No recursion** — All tree traversals use explicit stacks. Exception:
  `match_recursive` in pattern matching (patterns are small and bounded).

### Visibility and naming

- New `Arena` methods default to `pub(crate)`. Promote to `pub` only when needed
  by `context.rs`, `expr.rs`, or generated macro code.
- Operator impls use internal macros (`impl_nary_binop`, etc.) — never add manual
  operator impls.
- `SymplexError` is `#[non_exhaustive]`. Every `Result`-returning method must have
  a `# Errors` doc section.

### Tracing

- `tracing::debug!` at algorithm entry points (simplify, integrate, solve, limit)
- `tracing::trace!` for inner loops (per-node processing, rule firings)
- Enable with `RUST_LOG=symplex=debug` or per-module (`symplex::gruntz=trace`)

### File size policy

Split any file exceeding ~4,000 lines. `expr.rs` was already split into `expr.rs`,
`expr_funcs.rs`, `expr_ops.rs`, and `expr_view.rs`.

---

## 5. Parallel Agent / Tool Usage Policy

When dispatching parallel subagents or tools for implementation:

1. **Mutually exclusive file ownership.** Each agent MUST be assigned specific files.
   No two agents may edit the same file in the same phase. Conflicts cause both agents
   to fight each other, producing corrupted or lost work.

2. **List files explicitly in the dispatch.** Every agent message must include:
   "ONLY touch: file1.rs, file2.rs, tests/test_foo.rs. Do NOT touch file3.rs."

3. **Phase barriers.** When Agent A's output is needed by Agent B, serialize them
   across phases. Commit Agent A's work before dispatching Agent B.

4. **Shared-file serialization.** If two waves both need to edit expr_funcs.rs,
   run them in sequential phases, not parallel.

5. **Context for agents.** Each agent message must include:
   - Which files they own exclusively
   - Project conventions (tracing, `#[must_use]`, no names, tests)
   - What the existing code looks like (read X first)
   - Expected test count after their work

6. **Verify after each phase.** Run `cargo test` + `cargo clippy` after every
   phase before proceeding. Never trust an agent's claim of "all tests pass"
   without verification.

7. **Don't weaken tests to make them pass.** If a test fails, fix the root cause.
   The eval-before-evalf fix was discovered because we caught an agent weakening
   a test assertion from `starts_with("24")` to `(val - 24.0).abs() < 1e-8`.

---

## 6. Design Decisions and Expert Guidance

These decisions were made with input from the expert panel (see TEAM.md) and are
recorded here to prevent re-evaluation of the same ideas.

### Q1: Stirling for Gamma (not Lanczos)

**Decision:** Stirling series for arbitrary-precision Γ evaluation.
**Rationale:** Lanczos does not scale beyond f64. Spouge requires ~50% extra working
precision due to catastrophic cancellation. Stirling uses 0.323p terms (fewest),
Bernoulli number coefficients are cacheable across calls, and the same infrastructure
extends to log Γ, ψ, and polygamma by term-by-term differentiation.
**Reference:** Johansson 2021, "Arbitrary-precision computation of the gamma function."

### Q2: Stripper-collector for patterns (not full AC matching)

**Decision:** Three-layer pattern matching: structural → stripper-collector → bipartite.
**Rationale:** AC matching is NP-complete in general, but linear patterns (each wild
appears at most once) are polynomial — O(|s|·|t|³). All 24 simplification rules use
linear or near-linear patterns with k=2. Stripper-collector achieves O(k·n) for the
common case. Full Maude/MatchPy-style discrimination nets are deferred to when we
exceed ~50 rules.
**Reference:** Benanav/Kapur/Narendran 1987, Eker 1995.

### Q3: BitVec liveness for GC (not refcounting)

**Decision:** Arena `compact()` uses BitVec-based liveness tracing, not reference counting.
**Rationale:** Refcounting adds overhead to every construction. BitVec trace is the
same walk `compact()` already does, minus the copy. Heuristic: compact when arena >
100K nodes AND grown 2× since last compact AND liveness < 50%.
**Reference:** Semi-space copying collector design.

### Q4: Buchberger+FGLM for Gröbner (not F5)

**Decision:** Improved Buchberger with Gebauer-Möller criteria + FGLM order conversion.
**Rationale:** F5 is 10-100× faster but significantly more complex and harder to debug.
Buchberger with Gebauer-Möller eliminates ~90% of S-pairs. FGLM converts grevlex→lex
for the system-solving pipeline. Practical for 2-5 variables, degree ≤ 10.
**Status:** Future work (Wave GB).

### Q5: eval-before-evalf pipeline

**Decision:** `evalf()` always calls `eval()` first to reduce exact values.
**Rationale:** CAS principle: symbolic first, numerical last. Without this, `evalf`
would numerically approximate `sin(π)` instead of recognizing it's exactly 0. The
pipeline is: `eval()` → exact symbolic reduction → `evalf()` → arbitrary-precision float.
This was discovered when an agent weakened a test — the failing assertion exposed that
the eval step was being skipped.

### Q6: Number×Add distribution stays (scoped distribution deferred)

**Decision:** `canon_mul` distributes `Number × Add` (e.g., `2*(x+1) → 2*x + 2`).
**Rationale:** Without this, `a - a = 0` fails because `Mul(-1, Add(-1, x))` stays
opaque inside a sum. SymPy makes the same choice. `factor_terms()` returns a
`(gcd, inner)` tuple for display-level factored forms without violating canonicalization.
Scoped distribution (making `factor_terms` preserve through Add.flatten) is a v0.3 plan.

### Other settled decisions

| Decision | Rationale |
|----------|-----------|
| Ex is Clone, not Copy | Arc prevents Copy; expr! macro mitigates `&` noise |
| No Float node type | All exact; `evalf()` returns String |
| Phantom types for sorts | One struct, zero cost, scales to N sorts |
| Iterative display | Explicit work-stack, handles 10K+ depth |
| No feature flags | All capabilities included; feature flags reserved for C deps |
| Single RwLock | Structural lock ordering; deadlock impossible by construction |
| ExprView for replace() | Non-locking view type; deadlock impossible at compile time |
| Sturm sequences for real root counting (not solve-and-filter) | Exact, avoids Wilkinson's polynomial problem, operates entirely in ℚ, O(n²) in degree |
| Dependency parameter for diff (not arena/context storage) | Avoids arena mutation, no hash-consing complications, natural threading through explicit-stack walk |

---

## 7. Current Statistics

| Metric | Value |
|--------|-------|
| Tests | 4,370 passing, 0 failing |
| ExprNode variants | 66 |
| Source modules | 76 |
| Test files | 125 |
| Source | ~60,400 lines across 76 modules |
| Tests | ~49,000 lines across 125 test files |
| Examples | ~445 lines across 8 examples |
| Companion crates | ~1,045 lines across 2 crates (symplex-build, symplex-wasm) |
| Total | ~111,000 lines |
| Public methods on `Ex` | 175+ (numeric + boolean + set-valued) |
| Public methods on `Context` | 17 |
| Matrix methods | 50+ (added operators, exp, kronecker, cholesky, pinv, diag, from_i64) |
| Apply functions | 12 + 9 special functions (Bessel, Legendre, etc.) |
| `expr!` functions | 65 (54 single-arg + 11 multi-arg) |
| Simplification rules | 24 (condition-guarded) |
| Integration forms | 40+ |
| Eval special values | 86+ |
| Criterion benchmarks | 27 |
| Proptest properties | 22+ |
| Solver degree support | 1–4 (Cardano cubic + Ferrari quartic) |
| Examples | 8 (quickstart, calculus, control_system, robotics_codegen, solve_system, dynamics, latex_output, repl) |
| Clippy warnings | 0 (1 pre-existing dead_code warning in sturm.rs) |

---

## 8. Known Limitations

Active limitations (not yet resolved):

1. **`bigint_to_bigfloat` loses precision for integers > i128.** Falls back to f64 intermediate.
2. **`expr!(1/2)` is a compile error.** By design — prevents silent Rust integer division. Use `ctx.rational(1, 2)`.
3. **`expr!(x^2^3)` nested integer powers.** Inner `2^3` evaluates as integer arithmetic, not symbolic.
4. **`factor_terms` undone by Number×Add distribution.** `factor_terms(4x+6y)` extracts 2 but `canon_mul` distributes back. Needs display-only factored form.
5. **`lambdify` does not support complex expressions.** Returns `None` for expressions containing `I`.
6. **Phantom type safety is API-level only.** Internal arena code is untyped. Sort violations caught by `verify_canonical` in debug builds, not at compile time.
7. **No boolean symbols.** All symbols are `Expr<Numeric>`. Boolean-typed symbolic variables not supported.
8. **No arbitrary-precision special function evaluation.** Gamma, erf, beta use f64 fast paths only. — Fix: Wave AP (Stirling series)
9. **Pattern matching limited to linear patterns.** Nonlinear patterns (same wild twice) not supported. — Fix: Wave PM
10. **No Risch integration.** Decision procedure for elementary antiderivatives not implemented.
11. ~~**No multivariate polynomials.** Gröbner bases not yet implemented.~~ — **RESOLVED** (Wave GB: `groebner.rs`, `multipoly.rs`, `polysys.rs`)
12. ~~**No full Hensel/Zassenhaus factoring.** `factor()` uses rational root theorem only.~~ — **PARTIALLY RESOLVED** (Kronecker's method for non-linear factors, square-free decomposition, rational root theorem)
13. **Set types are foundation-only.** Interval merging, membership queries, and set arithmetic are minimal.
14. **Laplace transforms are table-based.** No algorithmic fallback for forms outside the table.
15. ~~**`diff_with_deps` is not exposed in public API.**~~ — **RESOLVED** (exposed as `Ex::diff_with_dependent()`)
16. **Sturm-based inequality solving is a fast-path only.** The Sturm chain detects "no real roots" cases; full Sturm-based interval construction is not yet integrated into the sign-chart builder.
17. **Display: `+ -N` pattern.** Add displays `x + -3` instead of `x - 3`. Fix planned for next release.
18. **Matrix::to_latex() missing.** LaTeX rendering exists for Ex but not Matrix or Quaternion.
19. **Codegen doesn't constant-fold.** Generated code contains `0_f64.cos()` instead of `1.0`.
20. **Gröbner solver only finds rational roots.** Systems with irrational solutions (x²-2=0 → √2) return empty.
21. **No plotting/visualization.** No SVG, no matplotlib integration.
22. **No Python bindings.** Rust-only API; no PyO3 wrapper.
23. **`expr!` doesn't support fraction literals.** `expr!(1/2)` is a compile error (Rust integer division).
24. **2-DOF IK only.** Inverse kinematics limited to planar 2-DOF. General n-DOF IK requires Pieper decomposition (not yet implemented).

---

## 9. Roadmap

### Immediate (0.2.0 release)

| Task | Description | Est. | Priority |
|------|-------------|------|----------|
| Display fix | `+ -N` → `- N` in Add display | 2-3 hrs | Critical |
| Codegen fix | Constant folding, negation style | 3-4 hrs | Critical |
| Matrix::to_latex | Add missing LaTeX method | 30 min | Critical |
| Example overhaul | Rewrite 5 examples, add 4 new | 6-8 hrs | Critical |
| README rewrite | 667 → 300 lines, hero code, accurate tables | 4-6 hrs | Critical |
| CHANGELOG update | Log 20+ unlogged features | 2-3 hrs | High |
| expr! fractions | Recognize half/third/quarter constants | 1-2 hrs | High |

### Post-0.2.0 (audience expansion)

| Task | Description | Audiences Unlocked |
|------|-------------|-------------------|
| Basic 2D plotting (SVG) | Function graphs, Bode plots, pole-zero maps | EE, professors, students |
| Python bindings (PyO3) | `import symplex` for Python users | ML, professors, quants |
| General n-DOF IK (≥7-DOF) | Pieper decomposition, arm-angle parameterization | Robotics startups |
| Sparse matrices (CSR/CSC) | Sparse eigenvalue, sparse LU | Controls engineers |
| WASM demo | Browser-based CAS with LaTeX rendering | Marketing/discovery |

### Version roadmap

| Version | Theme | Key features |
|---------|-------|-------------|
| 0.2.0 | Polish & Ship | Display fixes, API rename, examples, docs, Gröbner bases |
| 0.3.0 | Reach | Python bindings, basic plotting, WASM demo |
| 0.4.0 | Depth | Sparse matrices, n-DOF IK, PDE solving, tensor calculus |
| 1.0.0 | Stability | API freeze, comprehensive testing, documentation |

### Strategic context

symplex is a MIT/Apache-2.0 general-purpose CAS for Rust. Likely early adopters:
robotics/control engineers (symbolic Jacobians → code generation), physics students
(calculus/series/ODEs), compiler/PL researchers (term rewriting), and numerical
algorithm developers (derive formula → compile to fast code). The common thread:
**derive a formula symbolically, then compile it to numerical code.**

---

## 10. Module Reference

### Main crate: `symplex/src/` (76 modules)

| Module | Responsibility |
|--------|----------------|
| `apart.rs` | Partial fraction decomposition |
| `arena.rs` | Expression arena with hash-consing, intern(), constructors, constant pool |
| `assumptions.rs` | Three-valued property inference engine (23 properties, ~40 implication rules) |
| `bernoulli.rs` | Exact Bernoulli number computation (lazy cache, recurrence) |
| `canon.rs` | Canonical-form constructors: add/mul/pow/neg/and/or/not/set operations |
| `codegen.rs` | Rust source code generation (`to_rust_fn` with CSE) |
| `combsimp.rs` | Factorial/binomial simplification |
| `compact.rs` | Arena compaction (generational GC, live-node tracing) |
| `complex.rs` | Complex number decomposition (re/im/arg/conjugate) |
| `config.rs` | `EvalConfig` — max_pow_exponent, max_result_digits, max_evalf_precision |
| `context.rs` | `Context` — user-facing entry point, Arc\<RwLock\<ContextInner\>\> |
| `control.rs` | StateSpace, TransferFunction, Routh-Hurwitz, Ackermann, ZOH discretization |
| `convergence.rs` | Series convergence testing |
| `cse.rs` | Common subexpression elimination |
| `diff.rs` | Symbolic differentiation (all 66 node types, chain rule, n-ary product rule) |
| `display.rs` | Iterative (non-recursive) expression pretty-printer |
| `dynamics.rs` | Euler-Lagrange, mass matrix, Christoffel symbols, Coriolis, gravity vector |
| `eq.rs` | `Equation` type with solve/subs/simplify |
| `errors.rs` | `SymplexError` — non-exhaustive error enum |
| `eval.rs` | Special-value evaluation (86+ values, unit circle, Gamma(n), erf(0)) |
| `evalf.rs` | Arbitrary-precision numerical evaluation via `astro-float` |
| `expand.rs` | Algebraic expansion (distribute, power expand, multinomial) |
| `expr.rs` | `Expr<S>` struct, `Sort` trait, phantom type aliases (Ex, BoolEx, SetEx) |
| `expr_funcs.rs` | All 175+ public methods on Ex/BoolEx/SetEx |
| `expr_ops.rs` | Operator overloads (Add/Sub/Mul/Div/Neg for all Ex/&Ex/i64 combos) |
| `expr_view.rs` | `ExprView` — non-locking view type for `replace()` closures |
| `factor.rs` | Polynomial factoring (rational root theorem, content extraction) |
| `factor_terms.rs` | GCD extraction from sums |
| `fourier.rs` | Fourier series computation via integration |
| `fourier_transform.rs` | Table-based symbolic Fourier transform and inverse |
| `groebner.rs` | Buchberger's algorithm with Gebauer-Möller, FGLM order conversion |
| `gruntz.rs` | Gruntz algorithm for limits at infinity (~1,700 lines) |
| `inequalities.rs` | Polynomial/rational inequality solving → SetEx |
| `integrate.rs` | Symbolic integration (power, trig, exp, by-parts, u-sub, apart pipeline) |
| `lambdify.rs` | Compile expressions to `Box<dyn Fn(&[f64]) -> f64>` closures |
| `laplace.rs` | Forward/inverse Laplace transforms (table + structural rules) |
| `latex.rs` | LaTeX rendering via direct ExprNode matching (no Display parsing) |
| `lib.rs` | Module declarations, prelude, `__macro_support`, proc macro re-exports |
| `limit.rs` | Symbolic limits (direct substitution, L'Hôpital, series, Gruntz dispatch) |
| `linalg.rs` | Linear system solving (Gaussian elimination over exact rationals) |
| `log_combine.rs` | Log combining: ln(a)+ln(b) → ln(ab) |
| `log_expand.rs` | Log expansion: ln(ab) → ln(a)+ln(b) |
| `macros.rs` | `syms!` and `sym!` declarative macros |
| `matrix.rs` | Symbolic matrix: det, inv, eigen, LU, QR, RREF, rank, nullspace, Jacobian, codegen, 50+ methods |
| `multipoly.rs` | Sparse multivariate polynomials with generic monomial ordering MultiPoly\<O\> |
| `node.rs` | `ExprId(u32)`, `ExprNode` enum (66 variants), `children()`, `is_atom()` |
| `nsimplify.rs` | Closed-form detection from floats (PSLQ-lite) |
| `ode.rs` | ODE classification and solving (separable, linear, 2nd-order CC) |
| `parse.rs` | Runtime expression parser (BigInt/Ratio tokens, recursion depth limit) |
| `pattern.rs` | Pattern matching, rewrite rules, `basic_rules()` (24 rules), sub-expr matching |
| `poly.rs` | Dense univariate polynomials over ℚ (arithmetic, Euclidean GCD, Horner eval) |
| `polybridge.rs` | Expression ↔ Poly bridge, cancel(), collect(), together() |
| `polysys.rs` | Polynomial system solving via Gröbner bases + back-substitution |
| `powsimp.rs` | Power simplification (symbolic exponent merging) |
| `quaternion.rs` | Hamilton product, rotation matrix, axis-angle, angular velocity kinematics |
| `radsimp.rs` | Denominator rationalization |
| `residue.rs` | Residue computation via limit |
| `rewrite.rs` | Rewrite protocol: trig↔exp (Euler's formula), trig↔hyp |
| `robotics.rs` | DH parameters, FK chain, rotations, Euler angles, skew3, 2-DOF IK |
| `separatevars.rs` | Variable separation in products |
| `series.rs` | Taylor/Maclaurin series expansion with pole detection |
| `simplify_engine.rs` | Multi-strategy `smart_simplify` (7 strategies, picks lowest count_ops) |
| `solve.rs` | Equation solving: linear through quartic, transcendental, change-of-variable |
| `sort_key.rs` | `SortKey` — compact byte sequences for canonical ordering |
| `sturm.rs` | Sturm sequences for exact polynomial real root counting |
| `subs.rs` | Structural substitution (subs, subs_map) via walk_and_rebuild |
| `sum_eval.rs` | Symbolic sum/product evaluation (finite sums, convergence) |
| `symbol.rs` | Symbol table — string interning + per-symbol assumptions |
| `tree.rs` | `ExprTree` serde type for JSON interchange (to_tree/from_tree round-trip) |
| `trig_combine.rs` | Trig product-to-sum: sin(a)·cos(b) → ½[sin(a+b)+sin(a-b)] |
| `trig_expand.rs` | Trig expansion: sin(a+b), cos(a+b), sin(nx), cos(nx) |
| `trig_integ.rs` | Trig power integration (sin^n, cos^n, sec², csc², reduction formulas) |
| `trigsimp.rs` | Trig simplification (6-strategy choice-set) |
| `vector.rs` | Vector calculus: gradient, divergence, curl, laplacian, conservative/solenoidal |
| `walk.rs` | Shared iterative tree traversal: post_order_ids, walk_and_rebuild |
| `z_transform.rs` | Table-based z-transform and inverse z-transform |

### Proc macro crate: `symplex-macros/` (~1,450 lines)

| Module | Responsibility |
|--------|----------------|
| `lib.rs` | `expr!`, `rule!`, `matrix!`, `eq!` entry points + code generation |
| `parse.rs` | Shared Pratt parser for math expressions (precedence climbing, right-assoc `^`) |

---

## Key Decisions Log

| # | Decision | Alternatives Rejected | Rationale |
|---|----------|----------------------|-----------|
| 13 | **Buchberger over F5B for v1** | F5B (faster but less robust) | Simpler, well-understood, SymPy defaults to it |
| 14 | **MultiPoly\<O\> generic ordering** | Runtime ordering, unsorted dict | O(log n) leading term via BTreeMap, zero-cost phantom type |
| 15 | **Sin/cos ring for IK** (not Weierstrass) | Weierstrass (fewer vars, higher degree) | More robust (no θ=π singularity), cleaner interaction with elimination |
| 16 | **Dense matrices for FGLM** | Sparse/Krylov | D ≤ 1024 for IK; dense is simpler and fast enough |
| 17 | **Delete old API names** (not deprecate) | Keep as #[deprecated] | Pre-1.0, clean break. One name per operation. |
| 18 | **Display: `^` not `**`** | Python-style `**` | Math notation, not programming notation |
| 19 | **build.rs for codegen** (not proc macro) | Heavy proc macro | sqlx acknowledged proc-macro codegen as architecturally flawed |

> Decisions 1–12 are in TEAM.md § Key Decisions Log. Decisions 13+ are added here as the project grows.