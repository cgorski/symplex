# Symplex Implementation Plan

## Overview

Symplex is a symbolic mathematics library for Rust. It provides an arena-interned expression tree with hash-consing, canonical ordering, a three-valued assumption system, symbolic differentiation, pattern matching, algebraic expansion, special-value evaluation, arbitrary-precision numerical evaluation, polynomial algebra with GCD, fraction cancellation, and polynomial equation solving.

This document describes the architecture, design decisions, module responsibilities, data flow, concurrency model, and planned future work. It is intended to be sufficient for a new contributor to understand the codebase and begin development without prior context.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Module Reference](#module-reference)
3. [Data Model](#data-model)
4. [Concurrency Model](#concurrency-model)
5. [Canonicalization Rules](#canonicalization-rules)
6. [Assumption System](#assumption-system)
7. [Design Principles](#design-principles)
8. [Design Decisions and Rationale](#design-decisions-and-rationale)
9. [Dependencies](#dependencies)
10. [Testing Strategy](#testing-strategy)
11. [Public API Surface](#public-api-surface)
12. [Build Stages Completed](#build-stages-completed)
13. [Known Limitations](#known-limitations)
14. [Planned Future Work](#planned-future-work)
15. [File Layout](#file-layout)

---

## Architecture

```
User code
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Public API: Context, Ex, operators, macros          │
│  (context.rs, expr.rs, macros.rs)                    │
├──────────────────────────────────────────────────────┤
│  Assumption Engine: Props, Assumptions, cache        │
│  (assumptions.rs)                                    │
├──────────────────────────────────────────────────────┤
│  Transformations: diff, subs, expand, eval, evalf,   │
│  pattern, solve, cancel                              │
│  (diff.rs, subs.rs, expand.rs, eval.rs, evalf.rs,   │
│   pattern.rs, solve.rs, polybridge.rs)               │
├──────────────────────────────────────────────────────┤
│  Canonicalization: Add, Mul, Pow, Neg                │
│  (canon.rs)                                          │
├──────────────────────────────────────────────────────┤
│  Arena: hash-consing, interning, ExprNode, SortKey   │
│  (arena.rs, node.rs, sort_key.rs, symbol.rs,         │
│   display.rs, walk.rs)                               │
├──────────────────────────────────────────────────────┤
│  Polynomial Algebra: Poly, GCD, bridge               │
│  (poly.rs, polybridge.rs)                            │
└──────────────────────────────────────────────────────┘
```

Each layer only calls downward. There are no circular dependencies between modules.

---

## Module Reference

### Internal modules (`pub(crate)`)

| Module | Lines | Responsibility |
|--------|-------|----------------|
| `arena.rs` | ~900 | Expression arena with hash-consing. Owns all nodes, numbers, sort keys, symbols. Pre-interns constants (0, 1, -1, π, e, i, ∞, NaN). Provides `intern()`, `node()`, `sort_key()`, convenience constructors (`int`, `symbol`, `add`, `mul`, `pow`, `neg`, `sin`, etc.), and delegate methods for all transformations. |
| `node.rs` | ~400 | `ExprId(u32)`, `NumId(u32)`, `SymbolId(u32)`, `CtxId(u32)` newtypes. `ExprNode` enum with 22 variants (Num, Symbol, Add, Mul, Pow, Neg, Sin, Cos, Tan, Exp, Ln, Sqrt, Abs, Apply, Derivative, Integral, Pi, E, ImaginaryUnit, Infinity, NegInfinity, ComplexInfinity, NaN). Methods: `children()`, `is_atom()`. |
| `canon.rs` | ~1150 | Canonical-form constructors: `canon_add`, `canon_mul`, `canon_pow`, `canon_neg`. Flattening via explicit stacks. Like-term collection via `FxHashMap`. Canonical sorting by `SortKey`. Number×Add distribution as final step of `canon_mul`. NaN/infinity propagation. |
| `sort_key.rs` | ~450 | `SortKey` — a compact byte sequence for lexicographic canonical ordering. Class ranks: Num(0) < Symbol(10) < Pow(20) < Mul(30) < Add(40) < Function(50) < Derivative(60) < Integral(70) < Constant(80) < Special(90). Computed once at interning time via `compute_sort_key`. |
| `symbol.rs` | ~75 | `SymbolTable` — string interning for symbol names. Parallel `Vec<Assumptions>` for per-symbol mathematical assumptions. |
| `display.rs` | ~780 | Iterative (non-recursive) expression pretty-printer. Uses an explicit `Vec<WorkItem>` stack where each item is either a literal string or an expression to expand. Handles operator precedence, parenthesization, subtraction rendering (Mul(-1, x) → "- x"), negative/fractional exponent parens. |
| `walk.rs` | ~490 | Shared iterative tree traversal infrastructure. `post_order_ids` — de-duplicated post-order traversal via explicit stack. `walk_and_rebuild` — bottom-up transformation with cache. `rebuild_with_cache` — reconstruct a node with substituted children through canonical constructors. `contains` and `free_symbols` utilities. |
| `diff.rs` | ~660 | Symbolic differentiation. Iterative bottom-up via `post_order_ids`. Rules for all 22 node types. Product rule (n-ary), power rule (constant/variable/general exponent), chain rule for all transcendentals. |
| `subs.rs` | ~340 | Structural substitution. `subs(expr, old, new)` replaces exact ExprId matches. `subs_map` for simultaneous substitution. Uses `walk_and_rebuild`. |
| `expand.rs` | ~580 | Algebraic expansion. Distributes Mul over Add factors via incremental cross-multiplication. Expands `Pow(Add, positive_int)` via repeated multiplication. Iterative bottom-up. |
| `eval.rs` | ~650 | Special-value evaluation. Recognizes rational multiples of π for sin/cos/tan. Known values: sin(0)=0, sin(π/2)=1, cos(0)=1, cos(π)=-1, exp(0)=1, exp(1)=E, ln(1)=0, ln(E)=1, sqrt(perfect squares), abs(numeric). |
| `evalf.rs` | ~760 | Arbitrary-precision numerical evaluation via `astro-float`. Converts expressions to `BigFloat` bottom-up. Uses `Consts` cache for π and e. Integer exponents use `powi`; general exponents use `pow`. Formats via `convert_to_radix(Dec)`. Feature-gated behind `evalf`. |
| `pattern.rs` | ~780 | Pattern matching and rewrite-rule engine. `WildId`, `Pattern`, `match_pattern` (structural, top-down with consistency checking), `instantiate` (template with wild substitution), `Rule` (named LHS→RHS with optional condition), `apply_rules` (bottom-up single-pass with trace), `Step` (trace entry). Built-in rule: Pythagorean identity. |
| `poly.rs` | ~830 | Dense univariate polynomials over ℚ. `Poly` with `Vec<Ratio<BigInt>>` coefficients in ascending degree. Add, Sub, Neg, Mul, scale, div_rem (Euclidean division), GCD (Euclidean, monic-normalized), make_monic, eval (Horner). |
| `polybridge.rs` | ~740 | Bridge between `ExprId` and `Poly`. `expr_to_poly` (iterative, returns None for non-polynomial expressions), `poly_to_expr`, `as_numer_denom` (separate positive/negative exponent factors), `cancel` (GCD-based common factor cancellation). |
| `solve.rs` | ~720 | Equation solver. `solve(expr, var)` converts to Poly, dispatches by degree. Linear: -b/a. Quadratic: discriminant analysis, exact rational roots or symbolic sqrt. Higher degree: Rational Root Theorem with trial division, iterative factor extraction. |

### Public modules

| Module | Lines | Responsibility |
|--------|-------|----------------|
| `context.rs` | ~250 | `Context` — user-facing entry point. Owns `Arc<RwLock<ContextInner>>` containing `Arena` + `Mutex<AssumptionCache>`. Methods: `symbol`, `symbol_with`, `int`, `rational`, `pi`, `e`, `i_unit`, `infinity`, `nan`, `query`, `display`, `node_count`. |
| `expr.rs` | ~740 | `Ex` — user-facing expression handle. 16 bytes: `CtxId + Arc + ExprId`. Clone (not Copy). Implements Add, Sub, Mul, Div, Neg for all combinations of `Ex`, `&Ex`, `i64`. Methods: `pow`, `powi`, `sin`, `cos`, `tan`, `exp_fn`, `ln`, `sqrt`, `abs`, `diff`, `subs`, `subs_map`, `expand`, `simplify`, `simplify_trace`, `eval`, `evalf`, `cancel`, `solve`, `is_zero`, `is_positive`, `query`, `equals`, `is_zero_structural`, `is_one_structural`. All transformation methods have `#[must_use]`. |
| `assumptions.rs` | ~1740 | `Props` (bitflags, 23 properties), `Assumption` enum (user-facing), `Assumptions` struct (known_true + known_false as Props), `forward_chain` (fixpoint bitmask inference), `AssumptionCache` (FxHashMap<ExprId, Assumptions> with property handlers for all node types). |
| `config.rs` | ~36 | `EvalConfig` — `max_pow_exponent`, `max_result_digits`, `max_evalf_precision`. |
| `errors.rs` | ~29 | `SymplexError` — `ContradictoryAssumptions`, `PrecisionExhausted`, `Cancelled`, `DivisionByZero`, `NotImplemented`. |
| `macros.rs` | ~57 | `syms!(ctx; x, y, z)` and `sym!(ctx; t, Positive, Real)` declarative macros. |

---

## Data Model

### ExprId and hash-consing

Every expression is an `ExprId(u32)` — a 32-bit index into the arena's `nodes: Vec<ExprNode>`. Structurally identical expressions share the same `ExprId`, enforced by a deduplication map (`FxHashMap<u64, SmallVec<[ExprId; 2]>>`). This gives O(1) equality and O(1) hashing on expressions.

### ExprNode

The `ExprNode` enum has 22 variants. Numeric values are stored in a side table (`Vec<Ratio<BigInt>>`, indexed by `NumId`) to keep the enum small (~32 bytes). Add and Mul use `SmallVec<[ExprId; 6]>` for inline storage of up to 6 children without heap allocation.

### Numbers

All exact numbers are `Ratio<BigInt>` from `num-rational`. There is no `Float` node type. Inexact arithmetic is only available through `evalf()`, which returns a `String`.

### Sort keys

Each node has a precomputed `SortKey` (a byte sequence) stored in a parallel array. Sort keys encode class rank followed by node-specific data (value bytes for numbers, name bytes for symbols, child sort keys for composites). This makes canonical ordering comparison O(key length) with no tree traversal.

---

## Concurrency Model

```
Arc<RwLock<ContextInner>>
  ├── arena: Arena                         (protected by outer RwLock)
  └── assumptions: Mutex<AssumptionCache>  (interior lock for cache)
```

- **Expression construction** (operators, `add`, `mul`, etc.): acquires outer `write()` lock.
- **Display, structural predicates** (`is_zero_structural`, `Display`): acquires outer `read()` lock.
- **Assumption queries** (`is_positive`, `is_zero`, `query`): acquires outer `read()` lock + inner `assumptions.lock()`.

Lock ordering is structural (outer → inner). Deadlock is impossible by construction.

`Ex` is `Send + Sync`. Multiple threads can share expressions and query assumptions concurrently. Expression construction serializes on the write lock but lock hold times are microseconds (single `intern()` call).

Uses `parking_lot::RwLock` and `parking_lot::Mutex` for performance (no system call for uncontended cases).

---

## Canonicalization Rules

### What constructors do

| Constructor | Does | Does not |
|---|---|---|
| `add()` | Flatten nested Add. Sort by SortKey. Combine like terms (x+x→2x). Evaluate numeric sums (2+3→5). Drop zeros. Propagate NaN. | Expand powers. Evaluate functions. Apply identities. |
| `mul()` | Flatten nested Mul. Sort by SortKey. Combine like bases (x·x→x²). Evaluate numeric products (2·3→6). Drop ones. Propagate NaN/zero. **Distribute Number×Add** (2·(x+y)→2x+2y). | Distribute symbolic×Add (x·(y+z) stays). Expand powers. Evaluate functions. |
| `pow()` | x⁰→1. x¹→x. 1ˣ→1. 0^(positive)→0. numeric^small_int→computed (guarded by EvalConfig). Propagate NaN. | Expand (x+1)². Evaluate x^(1/2)→sqrt(x). |
| `neg()` | Neg(Neg(x))→x. Neg(Num(n))→Num(-n). Neg(Add(...))→distribute. Neg(Mul(c,...))→absorb into coefficient. Otherwise→Mul(-1, x). | Nothing further. |
| `sin()`, `cos()`, etc. | Construct unevaluated node. | Evaluate sin(0)→0. Apply identities. |

### Number×Add distribution

When `canon_mul` produces a result of exactly `[Number, Add]`, the number is distributed over the Add's terms. This is required for correct like-term cancellation (e.g., `a - a = 0`). Symbolic factors are never distributed — that is the job of `.expand()`.

This means `2*(x+1)` → `2*x + 2`, but `y*(x+1)` stays as `y*(x+1)`.

---

## Assumption System

23 mathematical properties tracked as two `bitflags` structs (`known_true` and `known_false`):

```
commutative, complex, real, rational, integer, algebraic, transcendental,
irrational, imaginary, positive, negative, nonnegative, nonpositive,
zero, nonzero, even, odd, prime, composite, finite, infinite,
hermitian, antihermitian
```

`forward_chain()` applies ~40 implication rules as bitmask operations until fixpoint:
- Alpha rules: `integer → rational → real → complex → finite, commutative`
- Contrapositives: `¬complex → ¬real → ¬integer`
- Beta rules: `nonnegative ∧ nonzero → positive`

Property handlers compute assumptions for each node type:
- `Num`: all properties from the rational value (sign, parity, primality)
- `Symbol`: from stored user assumptions
- `Add`: integer if all terms integer, positive if all nonneg with ≥1 positive
- `Mul`: sign from negative count, zero if any factor zero
- `Pow`: positive^real → positive, real^even → nonneg
- Constants: Pi (positive, transcendental), E (positive, transcendental), I (imaginary, algebraic)

---

## Design Principles

1. **Construction is cheap, evaluation is explicit.** Constructors only canonicalize. `.eval()`, `.expand()`, `.simplify()` are explicit calls.
2. **Never silently wrong.** `Result::Err` over silent wrong answers. Structural substitution by default.
3. **One representation per concept.** One assumption system. One polynomial type. One number type.
4. **Thread-safe from day one.** `Arc<RwLock>`, `Ex` is `Send + Sync`.
5. **No recursive tree walks.** All traversals use explicit stacks. Stack overflow is impossible. (Exception: `match_recursive` in pattern matching — patterns are small.)
6. **The compiler is the API contract.** `pub` = stable. `pub(crate)` = internal.
7. **Extensible without inheritance.** Custom functions via name + registered rules.

---

## Design Decisions and Rationale

### Why `Ex` is Clone, not Copy

`Ex` stores an `Arc<RwLock<ContextInner>>` which is not `Copy`. This means operators work without any scoping mechanism (`with_ctx` was removed as dead architecture). The tradeoff: `&x + &y` instead of `x + y`. All operators are implemented for every combination of `Ex`, `&Ex`, and `i64`.

### Why Number×Add distributes in canon_mul

Without this, `Mul(-1, Add(-1, x))` stays opaque inside a sum, preventing `a - a = 0` from cancelling. proptest discovered this bug. The distribution matches SymPy's 20-year-proven `Mul.flatten` approach.

### Why no Float node type

Mixing exact and inexact arithmetic creates precision confusion (a SymPy pain point). All symbolic computation uses exact `Ratio<BigInt>`. Floating-point results are only available through `evalf()`, which returns a `String`.

### Why a single RwLock instead of two

An earlier design used separate `Arc<RwLock<Arena>>` and `Arc<RwLock<AssumptionCache>>`. This was refactored to a single `Arc<RwLock<ContextInner>>` with the assumption cache as an interior `Mutex`. This makes lock ordering structural (outer → inner) and deadlock impossible by construction.

### Why iterative display

The display module was rewritten from recursive `fmt_expr` to an explicit work-stack. This was the last Principle 5 violation. The iterative version handles 10,000-deep nested expressions without stack overflow.

---

## Dependencies

| Crate | Version | License | Purpose |
|-------|---------|---------|---------|
| `num-bigint` | 0.4 | MIT/Apache-2.0 | Arbitrary-precision integers |
| `num-rational` | 0.4 | MIT/Apache-2.0 | Exact rational numbers |
| `num-integer` | 0.1 | MIT/Apache-2.0 | GCD, LCM, div_rem |
| `num-traits` | 0.2 | MIT/Apache-2.0 | Zero, One, Num traits |
| `smallvec` | 1.13 | MIT/Apache-2.0 | Inline small vectors |
| `rustc-hash` | 2.1 | MIT/Apache-2.0 | Fast FxHashMap |
| `bitflags` | 2.11 | MIT/Apache-2.0 | Assumption property flags |
| `parking_lot` | 0.12 | MIT/Apache-2.0 | Fast RwLock/Mutex |
| `thiserror` | 2.0 | MIT/Apache-2.0 | Error type derivation |
| `astro-float` | 0.9 (optional) | MIT | Arbitrary-precision floats for evalf |

Dev dependencies: `criterion` 0.5 (benchmarks), `proptest` 1.10 (property-based testing).

All dependencies are MIT/Apache-2.0. No C bindings. No LGPL.

---

## Testing Strategy

689 tests across these categories:

| Category | Count | Location |
|----------|-------|----------|
| Unit tests (in-module) | ~358 | `src/*.rs` `#[cfg(test)]` modules |
| Property-based (proptest) | 23 | `tests/proptest_canon.rs` |
| Integration (stage 1) | 36 | `tests/test_stage1.rs` |
| Integration (stage 3) | 60 | `tests/test_stage3.rs` |
| Integration (stage 5) | 35 | `tests/test_stage5.rs` |
| Integration (stage 6) | 41 | `tests/test_stage6.rs` |
| Integration (stages 7-8) | 71 | `tests/test_stage7_8.rs` |
| Regression | 5 | `tests/test_bug_regression.rs` |
| Doctests | 14 | Inline in source |

### proptest invariants verified

- Canonicalization: idempotence, commutativity (Add/Mul), associativity (Add), identity elements, zero annihilator, double negation, self-subtraction, x + (-x) = 0.
- Numeric: add/mul/pow correctness for random integers.
- Display: never panics for random expression trees.
- Assumptions: single-assertion consistency, forward-chain idempotence, merge-self is noop, integer property correctness.

### Verification via substitution

Several tests verify correctness by evaluating both the original and transformed expression at specific points and checking they agree.

---

## Public API Surface

### Types

```rust
pub struct Context;           // Entry point. Owns arena + assumption cache.
pub struct Ex;                // Expression handle. Clone, Send, Sync. 16 bytes.
pub struct Props;             // Bitflags for mathematical properties.
pub enum Assumption;          // User-facing assumption specifier.
pub struct Assumptions;       // Three-valued property storage.
pub struct EvalConfig;        // Evaluation guard configuration.
pub enum SymplexError;        // Error type.
pub struct Step;              // Simplification trace entry.
```

### Context methods

```rust
Context::new() -> Self
Context::with_config(EvalConfig) -> Self
ctx.symbol(name) -> Ex
ctx.symbol_with(name, &[Assumption]) -> Ex
ctx.int(n: i64) -> Ex
ctx.rational(p, q) -> Ex
ctx.pi() -> Ex
ctx.e() -> Ex
ctx.i_unit() -> Ex
ctx.infinity() -> Ex
ctx.nan() -> Ex
ctx.query(&ex, Props) -> Option<bool>
ctx.display(&ex) -> String
ctx.node_count() -> usize
```

### Ex methods

```rust
// Construction (all #[must_use])
ex.pow(&exp) -> Ex          ex.powi(n: i64) -> Ex
ex.sin() -> Ex              ex.cos() -> Ex
ex.tan() -> Ex              ex.exp_fn() -> Ex
ex.ln() -> Ex               ex.sqrt() -> Ex
ex.abs() -> Ex

// Transformation (all #[must_use])
ex.diff(&var) -> Ex
ex.subs(&old, &new) -> Ex
ex.subs_map(&[(&old, &new)]) -> Ex
ex.expand() -> Ex
ex.simplify() -> Ex
ex.simplify_trace() -> (Ex, Vec<Step>)
ex.eval() -> Ex
ex.evalf(digits: u32) -> Result<String, SymplexError>
ex.cancel(&var) -> Ex
ex.solve(&var) -> Vec<Ex>

// Queries
ex.is_zero() -> Option<bool>
ex.is_positive() -> Option<bool>
ex.query(Props) -> Option<bool>
ex.equals(&other) -> Option<bool>
ex.is_zero_structural() -> bool
ex.is_one_structural() -> bool
ex.id() -> ExprId
ex.ctx_id() -> CtxId

// Operators: +, -, *, /, unary -
// All for Ex⊕Ex, Ex⊕&Ex, &Ex⊕Ex, &Ex⊕&Ex, Ex⊕i64, i64⊕Ex, &Ex⊕i64, i64⊕&Ex
```

### Macros

```rust
syms!(ctx; x, y, z);                    // Declare multiple symbols
sym!(ctx; t, Positive, Real);           // Declare with assumptions
```

---

## Build Stages Completed

| Stage | Description | Key Deliverables |
|-------|-------------|------------------|
| 1 | Foundation skeleton | Arena, ExprNode, intern, SortKey, Display |
| 2 | Canonicalization | Add/Mul/Pow/Neg with flatten/sort/combine, Number×Add distribution |
| 3 | Context + operators | Context, Ex, +/-/*/÷ operators, syms!/sym! macros, prelude |
| 4 | Assumptions | Props bitflags, forward_chain, 23 properties, handlers for all nodes |
| 5 | Substitution | Structural subs, subs_map, iterative walk_and_rebuild |
| 6 | Differentiation | All rules, chain rule, n-ary product rule, iterative |
| 6.5 | Pattern matching | WildId, Pattern, match_pattern, Rule, apply_rules, Step, simplify_trace |
| 7 | Expand + eval | Algebraic expansion, special-value evaluation table |
| 8 | Numerical evaluation | astro-float integration, evalf(digits) |
| 9 | Polynomial algebra | Poly type, arithmetic, Euclidean GCD, expr↔poly bridge, cancel() |
| 10 | Equation solver | solve() for linear/quadratic/rational-root polynomials |
| Cleanup | Quality | Iterative display, #[must_use], proptest, deduplicated rebuild_with_cache |

---

## Known Limitations

1. **Pattern matching is structural only.** `sin²(w) + cos²(w)` matches a 2-term Add but not a sub-expression inside a larger Add like `3 + sin²(x) + cos²(x)`. Sub-expression matching requires combinatorial search over Add/Mul terms.

2. **Polynomial factoring is not implemented.** `cancel()` and `solve()` use GCD and Rational Root Theorem. Full factoring (Berlekamp, Cantor-Zassenhaus, Hensel lifting) is not yet available.

3. **Complex number support is limited.** `ImaginaryUnit` exists as a node type, but `evalf` returns an error for complex expressions. Complex arithmetic is not implemented.

4. **No integration.** Symbolic antiderivatives are not implemented.

5. **No series expansion.** Taylor/Laurent series are not implemented.

6. **No limit computation.** The Gruntz algorithm is not implemented.

7. **`solve()` returns empty for non-polynomial equations.** Transcendental equations (e.g., `sin(x) = 0`) are not handled.

8. **`simplify()` has one built-in rule** (Pythagorean identity). Additional trig, log, exp, and power identities are needed.

9. **`bigint_to_bigfloat` loses precision for integers larger than i128.** Falls back to f64 conversion.

10. **No `collect()`, `together()`, or `factor_terms()`.** These require polynomial infrastructure that is now available but the expression-level wrappers are not yet written.

---

## Planned Future Work

### High priority (CAS capabilities)

- **Sub-expression matching in Add/Mul.** When a pattern has N terms and the expression has M > N, try C(M, N) subsets. Enables `sin²+cos²=1` inside larger sums.
- **More simplification rules.** Log identities (ln(a*b)=ln(a)+ln(b) for positive a,b), exp/ln inverses, power simplification (x^a * x^b → x^(a+b) inside expressions), conditional rules using assumption system.
- **`collect(var)`.** Group terms by powers of a variable. Uses `expr_to_poly` + `poly_to_expr`.
- **`together()`.** Combine fractions over common denominator using polynomial LCM.
- **Basic integration.** Antiderivatives for polynomials, trig, exp, ln. Risch algorithm for rational functions.
- **Series expansion.** `series(expr, var, point, order)` for Taylor/Laurent expansion.

### Medium priority

- **Polynomial factoring over ℤ.** Berlekamp + Hensel lifting for `factor()`.
- **Limits.** Gruntz algorithm.
- **More solve capabilities.** Systems of equations, transcendental equations via Newton's method.
- **CancelToken.** Cooperative cancellation for long-running computations.
- **Complex number support.** Track real/imaginary parts, complex evalf.

### Low priority / infrastructure

- **Criterion benchmarks.** Measure: 10K-term sum, (a+b)^20 expansion, deep-tree diff, evalf pi to 1000 digits.
- **`compile()` → `CompiledExpr`.** Stack bytecode for fast repeated numerical evaluation.
- **Split `assumptions.rs`.** At 1743 lines, it should become a directory module.
- **LaTeX output.** `expr.latex()` → String.
- **CI configuration.** GitHub Actions with clippy + test + proptest.

---

## File Layout

```
symplex/
├── Cargo.toml
├── README.md
├── IMPLEMENTATION_PLAN.md
├── src/
│   ├── lib.rs              Module declarations, prelude, Props Default impl
│   ├── arena.rs            Expression arena + hash-consing
│   ├── assumptions.rs      Three-valued property inference engine
│   ├── canon.rs            Canonical-form constructors (Add, Mul, Pow, Neg)
│   ├── config.rs           EvalConfig
│   ├── context.rs          Context (user-facing entry point)
│   ├── diff.rs             Symbolic differentiation
│   ├── display.rs          Iterative expression pretty-printer
│   ├── errors.rs           SymplexError
│   ├── eval.rs             Special-value evaluation
│   ├── evalf.rs            Arbitrary-precision numerical evaluation
│   ├── expand.rs           Algebraic expansion
│   ├── expr.rs             Ex (user-facing expression handle)
│   ├── macros.rs           syms! and sym! macros
│   ├── node.rs             ExprId, NumId, SymbolId, CtxId, ExprNode
│   ├── pattern.rs          Pattern matching + rewrite rules
│   ├── poly.rs             Dense univariate polynomials over ℚ
│   ├── polybridge.rs       Expression ↔ Poly bridge + cancel()
│   ├── solve.rs            Polynomial equation solver
│   ├── sort_key.rs         Canonical ordering
│   ├── subs.rs             Structural substitution
│   ├── symbol.rs           Symbol table with string interning
│   └── walk.rs             Shared iterative tree traversal
├── tests/
│   ├── proptest_canon.rs   Property-based canonicalization tests
│   ├── test_bug_regression.rs  Regression tests for proptest-discovered bugs
│   ├── test_stage1.rs      Stage 1 integration tests
│   ├── test_stage3.rs      Stage 3 integration tests
│   ├── test_stage5.rs      Stage 5 integration tests
│   ├── test_stage6.rs      Stage 6 integration tests
│   └── test_stage7_8.rs    Stages 7-8 integration tests
└── benches/
    └── canonicalization.rs (stub)
```
