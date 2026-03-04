# Symplex Implementation Plan

> **For new developers:** Start with [Current Architecture](#current-architecture-020) 
> to understand the codebase structure, then [Roadmap](#roadmap) for what's next.
> The [SymPy Gap Analysis](#sympy-gap-analysis--priority-roadmap) shows where 
> to focus for maximum impact.

## Overview

Symplex is a symbolic mathematics library for Rust. It provides an arena-interned expression tree with hash-consing, canonical ordering, a three-valued assumption system, symbolic differentiation, pattern matching with rewrite rules, algebraic expansion, special-value evaluation, arbitrary-precision numerical evaluation, polynomial algebra with GCD, fraction cancellation, polynomial equation solving, and proc macros for ergonomic expression building and rule definition.

This document describes the architecture, design decisions, module responsibilities, data flow, concurrency model, and planned future work. It is intended to be sufficient for a new contributor to understand the codebase and begin development without prior context.

---

## Current Architecture (0.2.0)

### Codebase at a glance

- **54,240 total lines** across 47 source modules, 50 test files, 2 examples, 1 benchmark
- **2,425 tests** — proptest (131 properties), known-answers (256), bounded exhaustive, numerical cross-validation, SymPy cross-validation (252/263), stress tests
- **46 ExprNode variants** including 11 boolean/logic/piecewise (added in 0.2.0)
- **Phantom-typed expression handles** — `Expr<Numeric>` (aliased `Ex`) and `Expr<Boolean>` (aliased `BoolEx`)

### Six-layer architecture

```text
Layer 1: Public API        expr.rs (Ex/BoolEx methods), context.rs, eq.rs, matrix.rs
Layer 2: Macros            symplex-macros/ (expr!, rule!, matrix!, eq!)
Layer 3: Transformations   diff.rs, integrate.rs, expand.rs, eval.rs, pattern.rs, solve.rs,
                           series.rs, limit.rs, simplify_engine.rs, trig_*.rs, log_*.rs
Layer 4: Canonicalization  canon.rs (canon_add/mul/pow/neg/and/or/not + verify_canonical)
Layer 5: Arena             arena.rs, node.rs (ExprNode enum), walk.rs, sort_key.rs, symbol.rs
Layer 6: Types             assumptions.rs, config.rs, errors.rs
```

Each layer only calls downward. No circular dependencies.

### Key design patterns

1. **Arena hash-consing** — Every expression is an `ExprId(u32)` index into `Arena.nodes: Vec<ExprNode>`. Structurally identical expressions share the same ExprId. O(1) equality, O(1) hashing.

2. **Canonical form** — Construction always canonicalizes: `Add` children sorted, like-terms merged, zeros dropped. `Mul` similar. `Pow(x,0)→1`, `Pow(x,1)→x`. `i^n` reduced mod 4. `(-1)^(1/2)→i`. Enforced by `canon_add/mul/pow/neg` + `verify_canonical` debug assertions.

3. **Phantom-typed sorts** — `Expr<S: Sort>` carries a compile-time sort marker. `Expr<Numeric>` (= `Ex`) supports arithmetic/calculus. `Expr<Boolean>` (= `BoolEx`) supports logic. `sin(bool_expr)` is a compile error. Zero runtime cost.

4. **Iterative walks** — All tree traversals use explicit stacks via `walk.rs` (`post_order_ids`, `walk_and_rebuild`). No recursion. Stack-safe for arbitrarily deep expressions.

5. **Pattern rewriting** — 23 simplification rules in `basic_rules()`. Sub-expression matching in both Add and Mul nodes. Rules can have conditions (`condition: Option<fn(&Arena, &Substitution) -> bool>`).

6. **Smart simplify** — `simplify_engine.rs` tries 7 strategies (eval, expand, factor_terms, trig_expand, logcombine, cancel) and picks the result with lowest `count_ops`.

7. **Gruntz limit algorithm** — Complete implementation of Dominik Gruntz's PhD algorithm for computing limits at infinity. Handles all exp-log functions via MRV (Most Rapidly Varying) set analysis. Falls back to L'Hôpital for finite-point limits. ~1500 lines in `gruntz.rs`.

8. **Comprehensive tracing** — Zero-cost `tracing` instrumentation at debug/trace levels throughout simplification (rule firings), integration (strategy selection, LIATE ordering), limits (L'Hôpital steps, Gruntz MRV/rewrite/leadterm), and smart_simplify (strategy comparison). Enable with `RUST_LOG=symplex=debug`.

### Type system architecture

The expression system uses **phantom-typed sorts** to prevent invalid compositions at compile time:

```text
Expr<S: Sort>     ← one generic struct, PhantomData<S>, zero runtime cost
  │
  ├── Expr<Numeric>    (type alias: Ex)       ← arithmetic, calculus, 103+ methods
  ├── Expr<Boolean>    (type alias: BoolEx)   ← comparisons, logic, 6 methods
  └── Expr<SetValued>  (type alias: SetEx)    ← intervals, unions (planned 0.3.0)

Cross-sort bridges:
  Ex::gt(&Ex) → BoolEx                       ← numeric comparison produces boolean
  Ex::piecewise(&[(&Ex, &BoolEx)]) → Ex      ← boolean conditions, numeric values
  SetEx::contains(&Ex) → BoolEx              ← planned: set membership test
  BoolEx::as_set(&Ex) → SetEx                ← planned: condition-to-interval conversion

Matrix — separate type (not in arena), adequate through 0.2.x
  MatrixExpr in arena planned for 0.4.0+ (symbolic matrix algebra)
```

**Why phantom types (not newtypes, not traits):**
- One `impl<S: Sort> Expr<S>` block for common methods (eval, simplify, subs, display)
- Adding a sort = 3 lines (marker type + `impl Sort` + type alias)
- Zero runtime cost (PhantomData is zero-sized)
- Error messages show sort: `expected Expr<Numeric>, found Expr<Boolean>`

**Why the arena stays untyped (ExprId, not NumericId/BooleanId):**
- Piecewise alternates numeric/boolean children — can't use homogeneous typed IDs
- Every internal module (walk, canon, diff, eval) would need sort generics
- `verify_canonical` debug assertions catch sort violations during testing
- Cost of typed arena >> benefit (API-level phantom types provide 99% of safety)

**Compile-time guarantees we have:**
| Guarantee | Mechanism |
|-----------|-----------|
| Sort safety (no sin(bool)) | Phantom types on Expr<S> |
| Exhaustive node handling | Rust match on ExprNode enum |
| No silent result discarding | #[must_use] on transformations |
| Future-proof errors | #[non_exhaustive] on SymplexError |

**Invariants enforced at debug/test time (not compile time):**
| Invariant | Mechanism |
|-----------|-----------|
| Canonical form (sorted, flattened) | verify_canonical + debug_assert! |
| Boolean children in And/Or | verify_canonical boolean check |
| No boolean children in Add/Mul | verify_canonical numeric check |
| Same-context ExprIds | Test methodology (global vs local context) |

**Invariants NOT enforced (undecidable):**
- Non-zero denominators, positive arguments for ln(), domain correctness

### Mathematical object types — design decisions

| Object | Architecture | Rationale |
|--------|-------------|-----------|
| **Scalars** | ExprNode variants in arena, `Expr<Numeric>` | Core use case, full infrastructure |
| **Booleans** | ExprNode variants in arena, `Expr<Boolean>` | Needed for piecewise, conditionals |
| **Sets** (planned) | ExprNode variants in arena, `Expr<SetValued>` | Algebraic/compositional, fits arena model |
| **Matrices** | Separate `Matrix` struct (Vec<Vec<Ex>>) | Variable dimensions, eager operations |
| **MatrixExpr** (0.4.0+) | ExprNode variants (Det, Transpose, MatMul) | Symbolic matrix algebra, deferred |
| **Vectors** | 1-column Matrix | No separate type needed |
| **Functions (λ)** | Not planned | Requires beta-reduction, too complex |

### ExprNode scaling plan

| Milestone | Variants | Strategy |
|-----------|----------|----------|
| 0.2.0 (current) | 46 | Flat enum, manual match arms |
| 0.3.0 (sets) | ~56 | Same + LatticeOp shared pattern for Union/Intersection |
| 0.4.0 (matrix-expr, special funcs) | ~70 | Consider define_function! macro |
| 0.5.0+ | ~80+ | Consider sub-enum split if unmanageable |

The flat ExprNode enum with compiler-enforced exhaustive matching is viable through ~80 variants. Match compiles to a jump table. Discriminant is 1 byte. Adding a variant requires ~12 file edits (mechanical, compiler catches every miss).

### LatticeOp pattern (for 0.3.0)

Four n-ary operations share the same algebraic structure:

| Operation | Identity | Annihilator | Idempotent | Sort |
|-----------|----------|-------------|------------|------|
| And | True | False | Yes | Boolean |
| Or | False | True | Yes | Boolean |
| Union | EmptySet | Reals | Yes | Set |
| Intersection | Reals | EmptySet | Yes | Set |

All need: flatten nested same-kind, sort by SortKey, deduplicate, drop identity, short-circuit on annihilator. A shared `canon_lattice()` function handles the common logic (~100 lines), with each operation providing its specific identity/annihilator/flatten-check. Add and Mul do NOT fit this pattern (they have numeric coefficient merging and are not idempotent).

### How to add common things

**New ExprNode variant** — Add to `node.rs` enum → compiler errors guide you to ~12 files that need match arms (sort_key, walk, display, diff, eval, evalf, tree, expr, polybridge, lambdify, expand, pattern). The canonical invariant checker (`verify_canonical`) catches structural violations.

**New simplification rule** — Add a `fn rule_xxx(arena) -> Rule` in `pattern.rs`, add it to `basic_rules()`. Use `arena.wild()` for pattern variables. See existing rules for the pattern.

**New integration form** — Add a match arm in `integrate_node()` in `integrate.rs`. Follow the existing patterns for u-substitution (`linear_coeff_of`) or by-parts.

**New eval special value** — Add a case in the relevant `eval_xxx` function in `eval.rs`. Follow the `as_pi_multiple` / `Ratio` comparison pattern.

**New Ex method** — Add to the appropriate impl block in `expr.rs`: `impl<S: Sort> Expr<S>` for sort-preserving ops, `impl Expr<Numeric>` for numeric-only, `impl Expr<Boolean>` for boolean-only.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Module Reference](#module-reference)
3. [Data Model](#data-model)
4. [Concurrency Model](#concurrency-model)
5. [Canonicalization Rules](#canonicalization-rules)
6. [Assumption System](#assumption-system)
7. [Proc Macros](#proc-macros)
8. [Design Principles](#design-principles)
9. [Design Decisions and Rationale](#design-decisions-and-rationale)
10. [Dependencies](#dependencies)
11. [Testing Strategy](#testing-strategy)
12. [Public API Surface](#public-api-surface)
13. [Build Stages Completed](#build-stages-completed)
14. [Known Limitations](#known-limitations)
15. [Detailed Next Steps](#detailed-next-steps)
16. [File Layout](#file-layout)

---

## Architecture

```
User code
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Public API: Context, Ex, operators, macros           │
│  (context.rs, expr.rs, macros.rs)                     │
│  Proc macros: expr!, rule!                            │
│  (symplex-macros/)                                    │
├──────────────────────────────────────────────────────┤
│  Assumption Engine: Props, Assumptions, cache         │
│  (assumptions.rs)                                     │
├──────────────────────────────────────────────────────┤
│  Transformations: diff, subs, expand, eval, evalf,    │
│  pattern, solve, cancel                               │
│  (diff.rs, subs.rs, expand.rs, eval.rs, evalf.rs,    │
│   pattern.rs, solve.rs, polybridge.rs)                │
├──────────────────────────────────────────────────────┤
│  Canonicalization: Add, Mul, Pow, Neg                 │
│  (canon.rs)                                           │
├──────────────────────────────────────────────────────┤
│  Arena: hash-consing, interning, ExprNode, SortKey    │
│  (arena.rs, node.rs, sort_key.rs, symbol.rs,          │
│   display.rs, walk.rs)                                │
├──────────────────────────────────────────────────────┤
│  Polynomial Algebra: Poly, GCD, bridge                │
│  (poly.rs, polybridge.rs)                             │
└──────────────────────────────────────────────────────┘
```

Each layer only calls downward. There are no circular dependencies between modules.

---

## Module Reference

### Main crate: `symplex/src/` (47 modules, ~34,154 lines)

#### Internal modules (`pub(crate)`)

| Module | Lines | Responsibility |
|--------|-------|----------------|
| `arena.rs` | 898 | Expression arena with hash-consing. Owns all nodes, numbers, sort keys, symbols. Pre-interns constants (0, 1, -1, π, e, i, ∞, NaN). Provides `intern()`, `node()`, `sort_key()`, convenience constructors, and delegate methods for all transformations. |
| `node.rs` | 400 | `ExprId(u32)`, `NumId(u32)`, `SymbolId(u32)`, `CtxId(u32)` newtypes. `ExprNode` enum with 22 variants. Methods: `children()`, `is_atom()`. |
| `canon.rs` | 1154 | Canonical-form constructors: `canon_add`, `canon_mul`, `canon_pow`, `canon_neg`. Flattening via explicit stacks. Like-term collection via `FxHashMap`. Canonical sorting by `SortKey`. Number×Add distribution. NaN/infinity propagation. |
| `sort_key.rs` | 453 | `SortKey` — compact byte sequence for lexicographic canonical ordering. Class ranks: Num(0) < Symbol(10) < Pow(20) < Mul(30) < Add(40) < Function(50) < Derivative(60) < Integral(70) < Constant(80) < Special(90). |
| `symbol.rs` | 77 | `SymbolTable` — string interning for symbol names. Parallel `Vec<Assumptions>` for per-symbol mathematical assumptions. |
| `display.rs` | 782 | Iterative (non-recursive) expression pretty-printer. Uses an explicit `Vec<WorkItem>` stack. Handles operator precedence, parenthesization, subtraction rendering, negative/fractional exponent parens. Stack-safe for 10,000+ depth. |
| `walk.rs` | 486 | Shared iterative tree traversal infrastructure. `post_order_ids`, `walk_and_rebuild`, `rebuild_with_cache`. Used by subs, diff, expand, eval, evalf, pattern. |
| `diff.rs` | 658 | Symbolic differentiation. Iterative bottom-up. Rules for all 22 node types. N-ary product rule, power rule with chain rule, all transcendentals. |
| `subs.rs` | 338 | Structural substitution. `subs(expr, old, new)` and `subs_map` for simultaneous replacement. Uses `walk_and_rebuild`. |
| `expand.rs` | 577 | Algebraic expansion. Distributes Mul over Add via incremental cross-multiplication. Expands `Pow(Add, positive_int)` via repeated multiplication. |
| `eval.rs` | 648 | Special-value evaluation. Recognizes rational multiples of π for sin/cos/tan. Known values for exp, ln, sqrt, abs at specific points. |
| `evalf.rs` | 758 | Arbitrary-precision numerical evaluation via `astro-float`. Converts expressions to `BigFloat` bottom-up. |
| `gruntz.rs` | 1500 | Gruntz algorithm for limits at infinity. MRV set computation, expression rewriting in terms of ω, leading term extraction, growth rate comparison. Handles all elementary exp-log functions. |
| `pattern.rs` | 777 | Pattern matching and rewrite-rule engine. `WildId`, `Pattern`, `match_pattern`, `instantiate`, `Rule`, `apply_rules`, `Step`. Built-in Pythagorean identity rule. |
| `poly.rs` | 828 | Dense univariate polynomials over ℚ. Add, Sub, Neg, Mul, scale, div_rem, GCD (Euclidean, monic-normalized), eval (Horner). |
| `polybridge.rs` | 742 | Bridge between `ExprId` and `Poly`. `expr_to_poly`, `poly_to_expr`, `as_numer_denom`, `cancel`. |
| `solve.rs` | 725 | Equation solver. Linear, quadratic, higher-degree via Rational Root Theorem. Returns `Vec<Solution>`. |

#### Public modules

| Module | Lines | Responsibility |
|--------|-------|----------------|
| `context.rs` | 263 | `Context` — user-facing entry point. `Arc<RwLock<ContextInner>>`. Methods: `symbol`, `symbol_with`, `int`, `rational`, constants, `query`, `display`, `with_arena_mut`. |
| `expr.rs` | 739 | `Ex` — expression handle. 16 bytes. Clone, Send, Sync. 28 public methods. All operators for Ex/&Ex/i64 combinations. All `#[must_use]` on transformations. |
| `assumptions.rs` | 1743 | `Props` (bitflags, 23 properties), `Assumption` enum, `Assumptions` struct, `forward_chain`, `AssumptionCache` with handlers for all node types. |
| `config.rs` | 36 | `EvalConfig` — `max_pow_exponent`, `max_result_digits`, `max_evalf_precision`. |
| `errors.rs` | 29 | `SymplexError` — `ContradictoryAssumptions`, `PrecisionExhausted`, `Cancelled`, `DivisionByZero`, `NotImplemented`. |
| `macros.rs` | 57 | `syms!` and `sym!` declarative macros. |
| `lib.rs` | 95 | Module declarations, prelude, `__macro_support`, proc macro re-exports. |

### Proc macro crate: `symplex-macros/` (~810 lines)

| Module | Lines | Responsibility |
|--------|-------|----------------|
| `parse.rs` | 366 | Pratt parser for math expressions over `syn::ParseStream`. `MathExpr` AST, `BinOp` enum, precedence-climbing parser. Handles `+`, `-`, `*`, `/`, `^`, unary `-`, function calls, integer literals, parentheses. Right-associative `^`. |
| `lib.rs` | 443 | Proc macro entry points. `expr!` — syntax sugar for building `Ex` values with auto-borrowing and `^` rewriting. `rule!` — builds `Pattern`/`Rule` from math syntax with wild (`_`-suffix) identifiers. Flat temporary generation to avoid double-mutable-borrow. |

---

## Data Model

### ExprId and hash-consing

Every expression is an `ExprId(u32)` — an index into the arena's `nodes: Vec<ExprNode>`. Structurally identical expressions share the same `ExprId`, enforced by a deduplication map (`FxHashMap<u64, SmallVec<[ExprId; 2]>>`). This gives O(1) equality and O(1) hashing.

### ExprNode

The `ExprNode` enum has 22 variants. Numeric values are stored in a side table (`Vec<Ratio<BigInt>>`, indexed by `NumId`) to keep the enum small (~32 bytes). Add and Mul use `SmallVec<[ExprId; 6]>` for inline storage of up to 6 children without heap allocation.

### Numbers

All exact numbers are `Ratio<BigInt>` from `num-rational`. There is no `Float` node type. Inexact arithmetic is only available through `evalf()`, which returns a `String`.

### Sort keys

Each node has a precomputed `SortKey` (byte sequence) stored in a parallel array. Sort keys encode class rank followed by node-specific data. Canonical ordering comparison is O(key length) with no tree traversal.

---

## Concurrency Model

```
Arc<RwLock<ContextInner>>
  ├── arena: Arena                         (protected by outer RwLock)
  └── assumptions: Mutex<AssumptionCache>  (interior lock for cache)
```

- **Expression construction** (operators, `add`, `mul`, etc.): outer `write()` lock.
- **Display, structural predicates**: outer `read()` lock.
- **Assumption queries**: outer `read()` lock + inner `assumptions.lock()`.

Lock ordering is structural (outer → inner). Deadlock is impossible by construction.

`Ex` is `Send + Sync`. Uses `parking_lot::RwLock` and `parking_lot::Mutex`.

---

## Canonicalization Rules

| Constructor | Does | Does not |
|---|---|---|
| `add()` | Flatten nested Add. Sort by SortKey. Combine like terms. Evaluate numeric sums. Drop zeros. Propagate NaN. | Expand powers. Evaluate functions. Apply identities. |
| `mul()` | Flatten nested Mul. Sort by SortKey. Combine like bases. Evaluate numeric products. Drop ones. Propagate NaN/zero. **Distribute Number×Add**. | Distribute symbolic×Add. Expand powers. Evaluate functions. |
| `pow()` | x⁰→1. x¹→x. 1ˣ→1. 0^pos→0. numeric^small_int (guarded). NaN. | Expand (x+1)². Evaluate x^(1/2)→sqrt. |
| `neg()` | Double-neg cancel. Numeric fold. Distribute over Add. Absorb into Mul coeff. Otherwise Mul(-1, x). | Nothing further. |
| Functions | Construct unevaluated node. | Evaluate. Apply identities. |

Number×Add distribution: `2*(x+1)` → `2*x + 2`. Required for `a - a = 0`. Symbolic products stay: `y*(x+1)` unchanged.

---

## Assumption System

23 properties tracked as two `bitflags` structs (`known_true`, `known_false`):
commutative, complex, real, rational, integer, algebraic, transcendental, irrational, imaginary, positive, negative, nonnegative, nonpositive, zero, nonzero, even, odd, prime, composite, finite, infinite, hermitian, antihermitian.

`forward_chain()` applies ~40 implication rules as bitmask operations until fixpoint. Property handlers compute assumptions for each node type (Num, Symbol, Add, Mul, Pow, constants, functions).

---

## Proc Macros

### `expr!` — Expression builder

Transforms math syntax into Rust code operating on `Ex` values:

```rust
let result = expr!(x^2 + 2*x + 1);
// Expands to: ((&x).powi(2) + (&x) * 2 + 1)
```

- All identifiers → auto-borrowed with `&`
- `^` → `.powi(n)` for integer RHS, `.pow(&rhs)` for expression RHS
- Known functions → method calls: `sin(x)` → `(&x).sin()`
- Integer literals → kept as `i64` (operator impls handle coercion)
- `Int / Int` → compile error (prevents silent Rust integer division)

### `rule!` — Rewrite rule builder

Builds `Pattern`/`Rule` structs from math syntax:

```rust
let r = rule!(arena, "pythagorean", sin(w_)^2 + cos(w_)^2 => 1);
```

- Identifiers ending in `_` are wilds (pattern variables)
- Known constants: `pi`, `E`, `I`, `oo`, `nan`, `zoo`
- Integer literals: `0` → `arena.zero`, `1` → `arena.one`, etc.
- Unknown bare identifiers → compile error with helpful message
- All sub-expressions emitted as flat `let __tN = arena.method(...)` bindings to avoid double-mutable-borrow
- Type references via `::symplex::__macro_support::*`

Shared Pratt parser (~366 lines) handles both macros with standard mathematical precedence.

---

## Design Principles

1. **Construction is cheap, evaluation is explicit.** Constructors only canonicalize. `.eval()`, `.expand()`, `.simplify()` are explicit calls.
2. **Never silently wrong.** `Result::Err` over silent wrong answers. Structural substitution by default.
3. **One representation per concept.** One assumption system. One polynomial type. One number type.
4. **Thread-safe from day one.** `Arc<RwLock>`, `Ex` is `Send + Sync`.
5. **No recursive tree walks.** All traversals use explicit stacks. Exception: `match_recursive` in pattern matching (patterns are small).
6. **The compiler is the API contract.** `pub` = stable. `pub(crate)` = internal.
7. **Extensible without inheritance.** Custom functions via name + registered rules.
8. **No feature flags unless absolutely necessary.** All capabilities are included unconditionally. Feature flags are reserved for dependencies that require a C toolchain, have LGPL licensing, or cause platform-specific build failures.

---

## Design Decisions and Rationale

### Why Ex is Clone, not Copy

`Ex` stores `Arc<RwLock<ContextInner>>` (not Copy). Operators work without scoping. The `expr!` macro mitigates the `&` noise via auto-borrowing.

### Why Number×Add distributes in canon_mul

Without this, `Mul(-1, Add(-1, x))` stays opaque inside a sum, preventing `a - a = 0`. Discovered by proptest. Matches SymPy's approach.

### Why no Float node type

Mixing exact and inexact arithmetic creates precision confusion. All symbolic computation uses exact `Ratio<BigInt>`. `evalf()` returns `String`.

### Why a single RwLock

Earlier design had two Arcs. Refactored to single `Arc<RwLock<ContextInner>>` with interior `Mutex<AssumptionCache>`. Lock ordering is structural. Deadlock impossible.

### Why iterative display

Rewritten from recursive to explicit work-stack. Handles 10,000-deep expressions. Last Principle 5 violation eliminated.

### Why proc macros in a separate crate

Rust requires proc-macro crates to be separate. The `symplex-macros` crate lives inside `symplex/symplex-macros/` as a path dependency. Re-exported via `pub use symplex_macros::*`.

### Why no feature flags

Symplex includes all capabilities unconditionally. Feature flags are avoided because:

1. **Maintenance cost.** Every feature flag multiplies the test matrix and creates conditional compilation branches that can silently diverge.
2. **API clarity.** Users get the full API. No runtime "NotImplemented" errors from missing features.
3. **The `astro-float` precedent.** It's pure Rust, MIT-licensed, adds minimal compile time, and numerical evaluation is a core CAS capability — not optional.

If a future dependency truly warrants a feature flag, document the justification here.

### Why Ex uses Arc (not Copy)

`Ex` is 16 bytes: `CtxId(u32)` + `Arc<RwLock<ContextInner>>(8)` + `ExprId(u32)`. It is `Clone` but not `Copy` because `Arc::clone` performs an atomic refcount increment — a side effect incompatible with `Copy` semantics.

Alternatives evaluated and rejected:

1. **Global arena + Copy index** — `Ex` would be just a `u32`. Breaks thread safety (global `Mutex` contention) or makes `Ex` `!Send` (thread-local arena). Rejected: violates "thread-safe from day one."
2. **Lifetime parameter `Ex<'ctx>`** — Zero overhead, `Copy`-able. Rejected: expressions can't be stored in structs, returned from functions, or used in collections without lifetime gymnastics.
3. **Implicit clone in operators** — Doesn't help: move semantics still consume the value. The `&` is needed to borrow, not to clone.

The `expr!` macro eliminates `&` noise for expression building. Methods take `&self` so no `&` is needed on the receiver. The `&` only appears in binary operators between named variables (`&x + &y`), which is standard Rust.

### Why the core library does not render LaTeX/Markdown/Typst

The core crate computes; rendering belongs in a separate `symplex-format` crate (future). The core provides:

1. **`Display` trait** — plain text output for debugging and simple display.
2. **`ExprTree` (serde)** — the machine-readable interchange format. Any language (JavaScript, Python, a web frontend) can consume the JSON and render it however it wants.

Rationale:
- Separation of concerns: computing and rendering are different responsibilities.
- Dependency footprint: LaTeX/Typst rendering may need template engines, Unicode tables, or format-specific logic that shouldn't inflate the core.
- Versioning independence: a new output format ships as a new `symplex-format` version without touching `symplex`.

### Type system assessment (advanced techniques evaluated)

The following advanced type techniques were evaluated for symplex and rejected or deferred. This section exists to prevent re-evaluation of the same ideas.

| Technique | Verdict | Reason |
|-----------|---------|--------|
| **Typestate for expression categories** (`Ex<Polynomial>`, `Ex<Rational>`) | ❌ Rejected | Classification happens at runtime; operations change categories; users fight the types |
| **Newtype wrappers** (`SymbolEx`, `NumericEx`) | ❌ Rejected | Ergonomic cost exceeds safety benefit; runtime checks are 5ns |
| **Sealed trait for ExprNode extension** | ❌ Rejected | Enum exhaustive matching already provides this guarantee |
| **Const generics for precision** (`EvalResult<50>`) | ❌ Rejected | Precision is a runtime parameter (user input) |
| **Builder with compile-time validation** | ❌ Rejected | Operations are simple single-step; builder adds ceremony |
| **Type-level proof of math properties** (linearity, idempotency) | ❌ Rejected | Rust's type system cannot prove mathematical properties; use proptest |
| **Macro-generated variant exhaustiveness** | 🟡 Deferred | Marginally useful at current scale (<30 variants); revisit at 40+ |
| **Trait for custom user functions** (`MathFunction` trait) | ✅ Future | Genuinely useful for extensibility; design when adding function registration |

What DOES provide compile-time correctness for symplex:
- **Exhaustive `match` on `ExprNode`** — adding a variant without handling it everywhere is a compile error (28 variants × 14 files = ~400 match arms)
- **`#[must_use]` on all transformations** — prevents silently discarding results
- **`#[non_exhaustive]` on `SymplexError`** — allows adding error variants without breaking downstream
- **Serde-derived `ExprTree`** — exhaustive matching on both ExprNode→ExprTree and ExprTree→ExprNode ensures serialization completeness
- **Proptest for algebraic invariants** — 33 properties empirically verify mathematical correctness

### Why phantom types for sort safety (not newtypes, not traits)

Evaluated three approaches for preventing boolean/numeric mixing:

1. **Newtypes** (`BoolEx(Ex)`) — simple but duplicates common methods per sort
2. **Phantom types** (`Expr<S: Sort>`) — chosen: one struct, generic common methods, zero-cost
3. **Traits** (`trait NumericExpr`, `trait BooleanExpr`) — too complex, orphan rules

Phantom types scale to N sorts with O(1) common-method effort. Adding a new sort
(e.g., `SetValued`) requires only a marker type + impl block. Internal modules
(arena, walk, display, diff, eval) are completely unaffected — they operate on
untyped `ExprId`. The sort safety lives exclusively at the public API boundary.

### Tracing and observability plan

The library will use the `tracing` crate for zero-cost diagnostic logging. When no subscriber is registered (the default), each tracing call costs ~1ns (atomic load + predicted branch). When a subscriber captures events, the user opted in.

Instrumentation levels:
- `info` spans: around each public `Ex` method (`simplify`, `expand`, `solve`, `integrate`, `evalf`, `diff`, `factor`, `series`)
- `debug` events: rule firings in simplify, polynomial degree in solve, precision in evalf
- `trace` events: per-node processing in walk/diff/expand (extremely verbose, disabled by default)

`tracing` is MIT-licensed, pure Rust, tiny (~50KB), and unconditionally included (no feature flag) per our "no feature flags" policy.

---

## Dependencies

### Main crate

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
| `astro-float` | 0.9 | MIT | Arbitrary-precision floats |
| `symplex-macros` | 0.1 (path) | MIT/Apache-2.0 | Proc macros |

### Proc macro crate

| Crate | Version | License | Purpose |
|-------|---------|---------|---------|
| `syn` | 2 | MIT/Apache-2.0 | Token parsing |
| `quote` | 1 | MIT/Apache-2.0 | Code generation |
| `proc-macro2` | 1 | MIT/Apache-2.0 | Token manipulation |

Dev dependencies: `criterion` 0.5, `proptest` 1.10.

All MIT/Apache-2.0. No C bindings. No LGPL.

---

## Testing Strategy

732 tests across these categories:

| Category | Count | Location |
|----------|-------|----------|
| Unit tests (in-module) | ~358 | `src/*.rs` |
| Property-based (proptest) | 23 | `tests/proptest_canon.rs` |
| Integration (stages 1,3,5,6,7-8) | 243 | `tests/test_stage*.rs` |
| Proc macro integration | 43 | `tests/test_macros.rs` |
| Regression | 5 | `tests/test_bug_regression.rs` |
| Doctests | 14 | Inline in source |

Proptest invariants: canonicalization idempotence/commutativity/associativity, identity elements, double negation, self-subtraction, numeric correctness, display safety, assumption consistency.

---

## Public API Surface

### Types

```rust
pub struct Context;       // Entry point
pub struct Ex;            // Expression handle (Clone, Send, Sync, 16 bytes)
pub struct Props;         // Bitflags (23 properties)
pub enum Assumption;      // User-facing assumption specifier
pub struct Assumptions;   // Three-valued property storage
pub struct EvalConfig;    // Evaluation guards
pub enum SymplexError;    // Errors
pub struct Step;          // Simplification trace entry
```

### Context methods

```rust
Context::new() -> Self
Context::with_config(EvalConfig) -> Self
ctx.symbol(name) -> Ex
ctx.symbol_with(name, &[Assumption]) -> Ex
ctx.int(n: i64) -> Ex
ctx.rational(p, q) -> Ex
ctx.pi() / ctx.e() / ctx.i_unit() / ctx.infinity() / ctx.nan() -> Ex
ctx.query(&ex, Props) -> Option<bool>
ctx.display(&ex) -> String
ctx.node_count() -> usize
ctx.with_arena_mut(|arena| { ... }) -> R
```

### Ex methods (28 public, all transformations `#[must_use]`)

```rust
// Construction
ex.pow(&exp)  ex.powi(n)  ex.sin()  ex.cos()  ex.tan()
ex.exp_fn()   ex.ln()     ex.sqrt() ex.abs()

// Transformation
ex.diff(&var)              ex.subs(&old, &new)
ex.subs_map(&[...])       ex.expand()
ex.simplify()             ex.simplify_trace()
ex.eval()                 ex.evalf(digits)
ex.cancel(&var)           ex.solve(&var)

// Queries
ex.is_zero() -> Option<bool>     ex.is_positive() -> Option<bool>
ex.query(Props) -> Option<bool>  ex.equals(&other) -> Option<bool>
ex.is_zero_structural() -> bool  ex.is_one_structural() -> bool

// Operators: +, -, *, /, unary -  (all for Ex/&Ex/i64 combinations)
```

### Macros

```rust
syms!(ctx; x, y, z);                    // Declare symbols
sym!(ctx; t, Positive, Real);           // Declare with assumptions
expr!(x^2 + 2*x + 1)                   // Build expression with math syntax
rule!(arena, "name", LHS => RHS)        // Define rewrite rule
```

---

## Build Stages Completed

| Stage | Description |
|-------|-------------|
| 1 | Arena, ExprNode, intern, SortKey, Display |
| 2 | Canonicalization: Add/Mul/Pow/Neg, Number×Add distribution |
| 3 | Context, Ex, operators, syms!/sym! macros, prelude |
| 4 | Assumption engine: 23 properties, forward_chain, handlers |
| 5 | Structural substitution, iterative walk_and_rebuild |
| 6 | Differentiation: all rules, chain rule, n-ary product rule |
| 6.5 | Pattern matching: wilds, rules, simplify, trace |
| 7 | Expand (distribute, power expand) + eval (special values) |
| 8 | Arbitrary-precision numerical evaluation (astro-float) |
| 9 | Polynomial algebra: Poly, arithmetic, Euclidean GCD, expr↔poly bridge, cancel() |
| 10 | Equation solver: linear, quadratic, rational root theorem |
| 11 | Proc macros: expr!, rule! with Pratt parser |
| Cleanup | Iterative display, #[must_use], proptest, deduplication |
| Tier 0 | Foundation cleanup: i64/Ex div, #[non_exhaustive] errors, pub(crate) intern, free_symbols, convenience queries, max_evalf_precision |
| Tier 1 | Test suites (solve/cancel), equals() improvement, error variant breakup (FreeSymbol/Unevaluable), proptest for diff/expand |
| P1–P4 | Simplification rules (exp_ln, ln_exp, abs_abs), sub-expression matching in Add, collect(var), together(), more eval special values |
| P5–P7 | Integration (power, trig, exp, linearity), Taylor series (with eval + pole detection), polynomial factoring (content extraction, multiplicity) |
| Ergonomics | Global default context, vars! macro, subs_i64, maclaurin, display improvements (1/x, negative coefficients) |
| Features | full_simplify (fixpoint), evalf_f64, assume(), sum_of/product_of, sqrt rule, integration by parts, runtime parser, definite integrals, degree/coeffs |
| Math expansion | Inverse trig (asin, acos, atan) + hyperbolic (sinh, cosh, tanh): 6 new ExprNode variants across 17 files |
| Serialization | ExprTree serde type with to_tree/from_tree/to_json/from_json round-trip; removed LaTeX/Markdown formatter (belongs in separate crate) |
| Sprint | Limits (L'Hôpital + series fallback), linear system solving (Gaussian elimination), u-substitution in integration, exposed as_numer_denom/is_polynomial/coeff/limit/solve_system |
| API polish | Return type fix (limit/solve/series → Result), exp_fn→exp rename, removed Context::display, clippy/fmt clean |
| Math sprint 1 | Inverse hyperbolics (asinh/acosh/atanh), nsolve, partial fractions, expand_trig, poly GCD/LCM, odd/even trig eval |
| Sqrt removal | Removed Sqrt node variant, canonicalize Pow(E,x)→Exp(x), cbrt/nthroot convenience, display detection |
| Math sprint 2 | Power-of-power rule, perfect nth root eval, irrational trig values, hyperbolic odd/even, log expansion |
| Infrastructure | Criterion benchmarks (30), GitHub Actions CI, CHANGELOG.md |
| Sprint A-D | Integration completeness (tan/ln/inverse trig-hyp/apart pipeline), 3 new simplify rules + Mul sub-match, complete unit circle eval, ergonomics (zero/one/expr_type/replace/logcombine), calculus example |
| Cycles 4-8 | Complex number support (i²=-1, (-1)^½→I, complex quadratic roots, Euler's formula), transcendental solver (inversion peeling), integer sqrt simplification (√8→2√2), trig-hyp bridge, inverse trig integrals, general linear substitution, 6 new simplify rules, assumption handlers for 9 function types, node rebuilding fixes, SymPy-inspired improvements |
| Cycles 9-13 | General u-sub, trig power integration, trig combine, solver change-of-variable, parser improvements (float/implicit-mul/constants), complex evalf Tier 3, as_real_imag, factor_terms, From<T>/Sum/Product, bounded exhaustive verification, known-answer corpus, numerical cross-validation |
| Cycles 14-18 | Equation type + eq! macro, Factorial/Binomial nodes, canonical invariant checker + canon_mul sort fix, code hardening, symbolic Matrix + matrix! macro + Jacobian, lambdify (expression→closure), CSE, ODE solver (separable/linear/2nd-order), expr! constants/rationals, rule! conditional guards |
| 0.2.0 | Phantom type system (Expr<S: Sort>), BoolEx, relationals (Gt/Ge/Eq_/Ne), logical connectives (And/Or/Not), BoolTrue/BoolFalse atoms, Piecewise expressions, compile-time sort safety, expr! comparison/logic operators |
| Hardening | Parser BigInt/Ratio, ExprView deadlock fix, Piecewise type-safe pairs, smart_simplify GCD fix, by-parts LIATE+depth, pow_pow/acosh_cosh/asin_sin rule corrections, together() LCM, cancel() content factors, expand_trig sin(nx), trig_combine eval pass, exp_log_denest rule, cos_div_sin rule, verify_canonical O(n), rebuild_with_cache macro, 131 proptest properties, concurrency tests, fuzz target |
| Gruntz | Complete Gruntz algorithm for limits at infinity. MRV set computation, expression rewriting in ω, leading term extraction, function-expansion-as-series for Laurent-like expressions. Handles exp(-x)→0, ln(x)/x→0, x·exp(-x)→0. canon_pow flattening for integer exponents. 263-fixture SymPy cross-validation (252 pass, 0 fail). Comprehensive tracing instrumentation. |

---

## Known Limitations

1. ~~**Pattern matching is structural only.**~~ **Resolved** — sub-expression matching in Add finds patterns within larger sums.
2. ~~**No polynomial factoring.**~~ **Resolved** — `factor()` with content extraction and multiplicity handling.
3. ~~**Limited complex number support.**~~ **Significantly improved** — `i²=-1` canonicalization, `(-1)^(1/2)→I`, `(-n)^(1/2)→I√n`, complex quadratic roots, Euler's formula. `evalf` still errors on complex expressions (Tier 3 deferred).
4. ~~**No integration.**~~ **Resolved** — basic antiderivatives for polynomials, trig, exp.
5. ~~**No series expansion.**~~ **Resolved** — Taylor series with pole detection.
6. ~~**No limit computation.**~~ **Resolved** — limits via direct substitution, L'Hôpital's rule, series fallback, and Gruntz algorithm for limits at infinity.
7. ~~**solve() is polynomial-only.**~~ **Improved** — transcendental solving via inversion peeling (exp, ln, sin, cos, tan, sqrt). General transcendental equations still limited.
8. ~~**`simplify()` has limited rules.**~~ **Improved** — 24 rules with condition guards, sub-expression matching in Add and Mul, and fixpoint iteration.
9. **`bigint_to_bigfloat` loses precision for integers > i128.** Falls back to f64.
10. ~~**No `collect()`, `together()`, or `factor_terms()` yet.**~~ **Partially resolved** — `collect()` and `together()` implemented.
11. **`expr!(1/2)` is a compile error.** By design — prevents silent Rust integer division. Use `ctx.rational(1, 2)`.
12. **`expr!(x^2^3)` with nested integer powers causes type errors.** The inner `2^3` evaluates as integer arithmetic, not symbolic.
13. ~~**`replace()` closure cannot call locking methods.**~~ **Resolved** — `replace()` now takes `ExprView` (non-locking view type), making deadlock structurally impossible at the type level.
14. **No `as_real_imag` decomposition.** Expressions cannot be split into real and imaginary parts programmatically.
15. **`factor_terms` undone by Number×Add distribution.** `factor_terms(4x+6y)` extracts 2 but `canon_mul` distributes it back. A display-only factored form is needed.
16. **ODE solver has no public `Ex`-level API.** Must use `ctx.with_arena_mut()` + `dsolve()` directly.
17. **`lambdify` does not support complex expressions.** Returns `None` for expressions containing `I`.
18. **Phantom type safety is API-level only.** Internal arena code is untyped (ExprId). Sort violations in rule implementations are caught by verify_canonical in debug builds, not at compile time.
19. **No boolean symbols.** All symbols are Expr<Numeric>. Boolean-typed symbolic variables (e.g., a proposition `p`) are not supported.
20. ~~**`replace()` closure deadlock.**~~ **Resolved** — `ExprView` type eliminates deadlock at compile time.
21. ~~**`smart_simplify` could return mathematically different expressions.**~~ **Resolved** — GCD factor is now preserved.
22. ~~**Integration by-parts could stack overflow.**~~ **Resolved** — LIATE ordering + depth limit (20).
23. ~~**Parser limited to i64 integers.**~~ **Resolved** — arbitrary-precision BigInt/Ratio parsing.
24. ~~**Limits at infinity couldn't be computed.**~~ **Resolved** — Gruntz algorithm handles all exp-log functions. `lim(exp(-x), x→∞) = 0`, `lim(ln(x)/x, x→∞) = 0`, `lim(x·exp(-x), x→∞) = 0`.
25. ~~**Parser limited to i64 integers and overflowed on 19+ decimal places.**~~ **Resolved** — Parser uses `BigInt`/`Ratio<BigInt>` for all numeric tokens. `0.1 + 0.2 = 3/10` exactly.
26. ~~**`Pow(Pow(a,b),c)` not flattened during canonicalization.**~~ **Resolved** — `canon_pow` now flattens nested integer powers. `(x²)³ = x⁶` at construction time.

---

## Detailed Next Steps

### Priority 1: More Simplification Rules ✅ Completed

**Goal:** Make `simplify()` useful for common mathematical identities.

**Implementation:** Use the `rule!` macro in `pattern.rs`'s `basic_rules()` function. Each rule is one line.

**Rules to add:**

```
// Inverse function pairs
exp(ln(w_)) => w_
ln(exp(w_)) => w_

// Trig at zero/pi (complements eval.rs for expressions inside larger contexts)
sin(0) => 0
cos(0) => 1
tan(0) => 0

// Power identities
w_^0 => 1
w_^1 => w_
1^w_ => 1
0^w_ => 0  (needs condition: w_ positive)

// Double function
sqrt(w_^2) => abs(w_)
abs(abs(w_)) => abs(w_)
```

**Where to edit:** `src/pattern.rs`, function `basic_rules()`. Add each rule using `rule!(arena, ...)`. The proc macro handles all the Pattern/Wild boilerplate.

**Effort:** ~30 minutes. Blocked on nothing.

### Priority 2: Sub-expression Matching in Add/Mul ✅ Completed

**Goal:** Enable `sin²(x) + cos²(x) → 1` inside larger sums like `3 + sin²(x) + cos²(x) → 4`.

**Implementation approach:**

1. In `apply_rules` (pattern.rs), when a rule fails to match an `Add` node, try matching the rule's LHS against all C(N, K) subsets of the Add's K terms, where K is the number of terms in the pattern's Add.
2. For K=2 (the common case), this is O(N²) where N is the number of terms in the expression's Add. Acceptable.
3. If a subset matches, rebuild the Add with the matching terms replaced by the rule's RHS and the remaining terms kept.

**Where to edit:** `src/pattern.rs`, function `apply_rules`. Add a new path after the current `try_apply` fails: if the node is an Add and the rule's pattern root is an Add, iterate over pairs of terms.

**Key data structures:** The `match_pattern` function already handles matching. The new code just needs to try it on different subsets of the Add's children.

**Effort:** ~2 hours. Requires careful handling of the remaining terms.

### Priority 3: `collect(var)` ✅ Completed

**Goal:** `collect(x² + 2xy + y², x)` → `x² + 2y·x + y²` (grouped by powers of x).

**Implementation:**

1. Call `expr_to_poly(arena, expr, var)` to get a `Poly`.
2. Call `poly_to_expr(arena, &poly, var)` to rebuild — this naturally groups by power.
3. If `expr_to_poly` fails (not polynomial), return unchanged.

**Where to edit:** Add `collect` to `polybridge.rs` and wire through `arena.rs` and `expr.rs`.

**Effort:** ~30 minutes. Uses existing Poly infrastructure.

### Priority 4: `together()` ✅ Completed

**Goal:** `1/x + 1/y` → `(x + y) / (x·y)`.

**Implementation:**

1. Decompose each term of an Add via `as_numer_denom`.
2. Compute the LCM of all denominators (as polynomial LCM = product / GCD).
3. Scale each numerator appropriately.
4. Sum the scaled numerators.
5. Rebuild as `new_numer / new_denom`.

**Where to edit:** Add `together` to `polybridge.rs`.

**Prerequisite:** Need polynomial LCM, which is `a * b / gcd(a, b)`. The Poly crate already has GCD and Mul.

**Effort:** ~1.5 hours.

### Priority 5: Basic Integration ✅ Completed

**Goal:** Antiderivatives for polynomials, trig, exp, ln.

**Implementation:**

1. Create `src/integrate.rs`.
2. Walk the expression bottom-up (like diff.rs).
3. Rules:
   - `∫ x^n dx = x^(n+1)/(n+1)` for n ≠ -1
   - `∫ x^(-1) dx = ln(|x|)`
   - `∫ sin(x) dx = -cos(x)`
   - `∫ cos(x) dx = sin(x)`
   - `∫ exp(x) dx = exp(x)`
   - `∫ (f + g) dx = ∫f dx + ∫g dx` (linearity)
   - `∫ c·f dx = c · ∫f dx` (constant factor)
4. For expressions that don't match any rule, return an unevaluated `Integral` node.
5. Chain rule substitution (integration by substitution) for simple cases.

**Where to edit:** New file `src/integrate.rs`. Wire through `arena.rs` and `expr.rs`.

**Effort:** ~4 hours for basic rules. Risch algorithm for rational functions is a separate large project.

### Priority 6: Series Expansion ✅ Completed

**Goal:** `series(exp(x), x, 0, 5)` → `1 + x + x²/2 + x³/6 + x⁴/24 + O(x⁵)`.

**Implementation:**

1. Create `src/series.rs`.
2. Use repeated differentiation + division by factorial: `f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + ...`
3. Each term is computed by `diff` (already implemented) and `subs` (already implemented).
4. Truncate at the requested order.
5. Return as a `Poly` or as an `Ex` expression.

**Where to edit:** New file `src/series.rs`.

**Effort:** ~3 hours for Taylor expansion. Laurent series and asymptotic series are more complex.

### Priority 7: Polynomial Factoring ✅ Completed

**Goal:** `factor(x² - 1)` → `(x - 1)(x + 1)`.

**Implementation:**

1. Find all rational roots via the existing `solve_rational_roots`.
2. For each root r, factor out (x - r).
3. Return the product of linear factors × remaining unfactored polynomial.
4. For complete factoring over ℤ, implement Berlekamp's algorithm + Hensel lifting. This is a large project.

**Where to edit:** Add to `poly.rs` or new `src/factor.rs`.

**Effort:** ~2 hours for rational-root factoring. Berlekamp+Hensel is ~20 hours.

### Priority 8: Equation Solving Improvements

**Goal:** Systems of linear equations, and transcendental equations via Newton's method.

**Implementation for linear systems:**

1. Create `src/linalg.rs` with Gaussian elimination over `Ratio<BigInt>`.
2. Input: list of linear expressions and list of variables.
3. Build coefficient matrix, solve, return substitution map.

**Implementation for Newton's method:**

1. Use `diff` and `evalf` to iteratively refine a numerical root.
2. Return as a numerical `Ex` value.

**Effort:** Linear systems ~4 hours. Newton's method ~2 hours.

### Remaining Feature Priorities

| Priority | Feature | Effort | Status |
|----------|---------|--------|--------|
| F1 | **`tracing` instrumentation** — zero-cost logging for all core operations | 1 hr | ✅ Done |
| F2 | **More simplification rules** — sinh²-cosh²=-1, inverse trig pairs (asin(sin(x))→x) | 30 min | ✅ Done |
| F3 | **MathFunction trait** — user-defined functions with derivative/eval/evalf callbacks | 3 hr | Design only |
| F4 | **symplex-format crate** — LaTeX, Markdown, Typst rendering consuming ExprTree | 4 hr | Planned (separate crate) |
| F5 | **REPL example binary** — `examples/repl.rs` using runtime parser | 30 min | ✅ Done |
| F6 | **More integration rules** — u-substitution for `sin(ax+b)`, `exp(ax)`, etc. | 2 hr | ✅ Done |
| F7 | **Polynomial GCD improvements** — multivariate, sparse representation | 8 hr | Not started (deferred to v0.2.0) |
| F8 | **Limit computation** — basic limits via substitution + L'Hôpital | 4 hr | ✅ Done |
| F9 | **logcombine()** — inverse of expand_log, combines logarithmic terms | 45 min | ✅ Done |
| F10 | **as_real_imag decomposition** — split expressions into real/imaginary parts | 4 hr | Not started |
| F11 | **Complex numerical evaluation** — (real,imag) pair arithmetic in evalf | 8 hr | Not started (deferred) |
| F12 | **Trig power reduction** — ∫ sin^n(x) dx recursive formula | 2 hr | Not started |
| F13 | **General u-substitution** — SymPy-style find_substitutions | 4 hr | Not started |
| F14 | **Limits at infinity** — dominant-term analysis for rational functions | 3 hr | ✅ Done (Gruntz algorithm) |
| F15 | **Series known-coefficient fast paths** — sin/cos/exp without repeated differentiation | 2 hr | Not started |
| F16 | **Completing the square in integration** — ∫ 1/(x²+bx+c) dx | 2 hr | Not started |
| F17 | **Vector calculus** — gradient, divergence, curl on Matrix | 3 hr | Not started |
| F18 | **ODE public API** — Ex-level dsolve with Equation input | 1 hr | Not started |

### Infrastructure Tasks

| Task | Where | Effort | Status |
|------|-------|--------|--------|
| Criterion benchmarks | `benches/canonicalization.rs`, `benches/transforms.rs` | 1 hour | ✅ Done |
| Split `assumptions.rs` (1,743 lines) | Into `assumptions/mod.rs`, `assumptions/inference.rs`, `assumptions/handlers.rs`, `assumptions/cache.rs` | 1 hour | Not started (deferred to v0.2.0) |
| CI configuration | `.github/workflows/ci.yml` — test + clippy + fmt | 30 min | ✅ Done |
| `CancelToken` for timeouts | New `src/cancel.rs`, integrate into expand/solve/simplify | 2 hours | Not started (deferred to v0.2.0) |
| Complex number support | Track re/im parts, complex evalf | 8 hours | Not started (deferred to v0.2.0) |
| Update README | Reflect all new features (inverse trig, hyperbolic, serde, parser, etc.) | 30 min | ✅ Done |
| CHANGELOG.md | For 0.1.0 release | 30 min | ✅ Done |
| Final API surface review | Scan all `pub fn` for consistency, naming, docs | 1 hour | Partially done |

### Release Checklist (0.1.0)

- [x] Benchmarks (30 Criterion benchmarks)
- [x] CI (GitHub Actions: test + clippy + fmt)
- [x] README fully up to date with all 81 public methods
- [x] CHANGELOG.md written
- [x] `cargo clippy` clean (0 warnings)
- [x] `cargo fmt` clean (0 issues)
- [x] All doctests pass in isolation
- [x] Breaking API changes completed (`exp_fn` → `exp`, `Sqrt` removal, `Result` returns)
- [ ] Split `assumptions.rs` (1,743 lines) — deferrable to 0.1.1
- [ ] Cargo.toml metadata polished (documentation link, etc.)
- [ ] Final API surface review — no accidental `pub` on internal types
- [ ] Publish to crates.io

### Current Statistics (Commit 94)

| Metric | Value |
|--------|-------|
| Tests | 2,425 passing, 0 failing, 0 warnings |
| Source | 34,154 lines across 47 modules |
| Tests | 18,086 lines across 50 files |
| Macros | 1,235 lines |
| Total lines | 54,240 |
| Public methods on `Ex` | 103+ (numeric) + 6 (boolean) |
| Public methods on `Context` | 17 |
| Free-standing functions | 5 |
| ExprNode variants | 46 |
| Simplification rules | 24 (with condition guards) |
| Integration forms | 30+ (LIATE-ordered by-parts) |
| Matrix methods | 26 |
| Factorial/Binomial | arbitrary precision (no limit) |
| Eval special values | 86+ (all tan quadrants, full unit circle) |
| Criterion benchmarks | 30 |
| Proptest properties | 131 |
| SymPy cross-validation | 252/263 pass (0 failures) |
| Gruntz algorithm | Complete (~1500 lines) |
| Tracing instrumentation | 6 modules |
| Commits | 94 |

---

## Next Steps for New Contributors

> **If you're reading this in a new context window**, this section tells you
> everything you need to know to continue development.

### Available Macros and Helpers

**Proc macros** (in `symplex-macros/`):
- `expr!(x^2 + sin(x))` — build expressions with natural math syntax
- `rule!(arena, "name", LHS => RHS)` — define rewrite rules with wilds (`w_` suffix)
- `rule!(arena, "name", LHS => RHS if condition)` — conditional rules
- `matrix![[a, b], [c, d]]` — matrix construction
- `eq!(x^2 = 4)` — equation construction

**Note:** The `rule!` macro CANNOT be used inside `src/` files (it generates `::symplex::` paths that don't resolve within the crate). Use `unary_compose_rule()` or manual `Rule::new()` for internal rules.

**Internal helpers** (in `pattern.rs`):
- `unary_compose_rule(arena, "name", outer_fn, inner_fn, template_fn)` — build `outer(inner(w)) → template(w)` rules in one line
- `identity(arena, w)` — template function that returns the wild unchanged (for `f(g(w)) → w`)

**Test helper macros** (in `tests/test_math_rules.rs`):
- `assert_simplifies_to!(expr, "expected")` — verify simplification result
- `assert_simplify_unchanged!(expr)` — verify expression is NOT simplified
- `assert_simplify_preserves_value!(expr, var, point)` — verify simplification preserves numerical value

### Tracing for Debugging

Enable tracing in tests:

```rust
let _ = tracing_subscriber::fmt()
    .with_env_filter("symplex::gruntz=debug")  // or symplex=debug for everything
    .with_test_writer()
    .try_init();
```

Key trace targets:
- `symplex::gruntz` — MRV sets, rewrite steps, leadterm, sign determination
- `symplex::pattern` — rule firings, sub-expression matching
- `symplex::simplify_engine` — strategy comparison in smart_simplify
- `symplex::integrate` — strategy selection, LIATE ordering, depth counter
- `symplex::limit` — form detection, L'Hôpital steps

### SymPy Cross-Validation

**To regenerate fixtures:**
```bash
cd symplex && source .venv/bin/activate
python3 scripts/generate_sympy_fixtures.py > tests/fixtures/sympy_cross_validation.json
```

**To run cross-validation:**
```bash
cargo test --test test_sympy_cross_validation -- --nocapture
```

Current: 252/263 pass, 0 fail, 2 not-implemented, 9 no-API.

### How to Add a New Simplification Rule

1. Choose the rule identity (e.g., `sin(2w) → 2·sin(w)·cos(w)`)
2. If it fits the `outer(inner(w)) → template(w)` pattern, use `unary_compose_rule()`:
   ```rust
   unary_compose_rule(arena, "my_rule", Arena::sin, Arena::asin, identity)
   ```
3. Otherwise, build manually in a `fn rule_xxx(arena: &mut Arena) -> Rule` function
4. Add to `basic_rules()` in `pattern.rs`
5. Add condition guard if needed: `r.condition = Some(|arena, bindings| { ... })`
6. Add positive test, negative test, and numerical validation test

### How to Add a New Integration Form

1. In `src/integrate.rs`, find `integrate_node()`
2. Add a match arm for the new form
3. Return the antiderivative as an `ExprId`
4. If the form needs u-substitution, use `linear_coeff_of()` and the existing machinery
5. Add SymPy cross-validation fixture in `scripts/generate_sympy_fixtures.py`

### Immediate Priorities (to close remaining 11 cross-validation gaps)

#### 1. `∫ sec²(x) dx = tan(x)` — 30 minutes
**File:** `src/trig_integ.rs` or `src/integrate.rs`
**The fix:** In the trig power integration, add a case for `cos(x)^(-2)`:
```rust
if n == -2 {
    // ∫ cos(x)^(-2) dx = tan(x)
    return arena.tan(var);
}
```
Also add `sin(x)^(-2) → -cot(x)` if cot exists, or `-cos(x)/sin(x)`.
**Test:** Add to `scripts/generate_sympy_fixtures.py`, regenerate, run cross-validation.

#### 2. `∫ exp(x)·sin(x) dx` — Cyclic IBP — 2 hours
**File:** `src/integrate.rs`
**The algorithm:** Apply IBP twice:
1. u=exp(x), dv=sin(x)dx → v=-cos(x), du=exp(x)dx
   Result: -exp(x)cos(x) + ∫ exp(x)cos(x)dx
2. u=exp(x), dv=cos(x)dx → v=sin(x), du=exp(x)dx
   Result: -exp(x)cos(x) + exp(x)sin(x) - ∫ exp(x)sin(x)dx
3. Let I = ∫ exp(x)sin(x)dx. Then I = -exp(x)cos(x) + exp(x)sin(x) - I
4. Solve: 2I = exp(x)(sin(x) - cos(x)), I = exp(x)(sin(x) - cos(x))/2

**Implementation:** After two IBP rounds, if the remaining integral equals the original (by ExprId comparison), solve the algebraic equation.

#### 3. Matrix inverse — 3 hours
**File:** `src/matrix.rs`
**Algorithm:** For n×n matrix A:
- Compute det(A)
- Compute cofactor matrix: C[i][j] = (-1)^(i+j) · det(minor(A, i, j))
- Adjugate = C^T (transpose of cofactor matrix)
- A^(-1) = adj(A) / det(A)
**Implementation:** Already have `det()`. Need `minor(i, j)` (matrix with row i and col j removed) and cofactor expansion.

#### 4. Matrix eigenvalues (2×2) — 2 hours
**File:** `src/matrix.rs`
**Algorithm:** For 2×2 matrix [[a,b],[c,d]]:
- Characteristic polynomial: λ² - (a+d)λ + (ad-bc) = 0
- Use existing quadratic solver
**Implementation:** Build the characteristic polynomial symbolically, call `solve()`.

### Medium-Term Priorities

#### 5. Cubic formula (Cardano) — 4 hours
**File:** `src/solve.rs`
**Enables:** 3×3 eigenvalues, irrational cubic roots
**Algorithm:** For x³ + px + q = 0: x = ∛(-q/2 + √(q²/4 + p³/27)) + ∛(-q/2 - √(q²/4 + p³/27))
**Note:** Requires depressed cubic form (substitute x = t - b/(3a) to eliminate x² term)

#### 6. Code generation (`to_rust_fn`) — 4 hours
**File:** New `src/codegen.rs`
**The idea:** Given an expression and a list of variable names, generate a Rust function body:
```rust
let code = expr.to_rust_fn(&["x", "y"]);
// Returns: "pub fn f(x: f64, y: f64) -> f64 { let t0 = x*x; t0*y.sin() + x.cos() }"
```
**Implementation:** Run CSE first, then emit each binding as a `let` statement, emit the final return.

#### 7. Simplify API consolidation
**Current:** `simplify()` (pattern rules only), `full_simplify()` (eval+cancel+expand+rules), `smart_simplify()` (7 strategies)
**Proposed:** Rename `smart_simplify()` → `simplify()`, rename old `simplify()` → `apply_rules()`
**Rationale:** Users expect `simplify()` to be the "do the best you can" function, like SymPy.

#### 8. `rewrite()` protocol — 4 hours
**File:** New `src/rewrite.rs`
**The idea:** `expr.rewrite(RewriteTarget::Exp)` converts trig to exponential form:
- `sin(x) → (exp(ix) - exp(-ix))/(2i)`
- `cos(x) → (exp(ix) + exp(-ix))/2`
**Enables:** Better integration (rewrite in exp form, integrate, convert back)

---

## Next Sprint: Math Depth & Ergonomics (Pre-0.1.0)

> **Status: ✅ COMPLETED** — All 21 items implemented and tested. See commit history.

This section documents the detailed gap analysis and implementation plan for the next development sprint, focused on math completeness and API ergonomics. This plan was produced by the full expert panel and should be executed before 0.1.0 evaluation.

### Sprint A: Integration Completeness

**Gap analysis:** Our integration handles power rule, basic trig (sin/cos), exp, sinh/cosh, u-substitution for `f(ax+b)`, integration by parts for `x*trig` and `x*exp`, and linearity/constant factor. The following standard forms are missing:

| # | Integral | Result | Implementation |
|---|---------|--------|----------------|
| A1 | `∫ tan(x) dx` | `-ln\|cos(x)\|` | Add `Tan` arm in `integrate_node` — rewrite as `sin/cos`, apply u-sub |
| A2 | `∫ ln(x) dx` | `x*ln(x) - x` | Add `Ln` arm — by parts with `u=ln(x), dv=dx` |
| A3 | `∫ 1/(x²+1) dx` | `atan(x)` | Pattern match `Pow(Add([x², 1]), -1)` in Pow arm |
| A4 | `∫ 1/sqrt(1-x²) dx` | `asin(x)` | Pattern match `Pow(Add([1, Neg(Pow(x,2))]), -1/2)` |
| A5 | `∫ 1/sqrt(x²+1) dx` | `asinh(x)` | Similar pattern match |
| A6 | `∫ 1/sqrt(x²-1) dx` | `acosh(x)` | Similar pattern match |
| A7 | `∫ 1/(1-x²) dx` | `atanh(x)` | Pattern match |
| A8 | Wire apart→integrate | Decompose rational, integrate each term | In Mul arm, try `apart()` when integrand is rational |
| A9 | Extended by-parts | `u=ln(x)` with `dv=polynomial` | Allow non-polynomial `u` in by-parts if `du` is simpler |

**Files:** `src/integrate.rs` (all items), tests in `tests/test_integrate.rs`

**Effort:** ~2 hours total

### Sprint B: Simplification Depth

**Gap analysis:** Our 13 simplify rules cover inverse function pairs, Pythagorean identities, and structural rules. Missing:

| # | Rule | Type | Implementation |
|---|------|------|----------------|
| B1 | `sin(w)/cos(w) → tan(w)` | Ratio recognition | New simplify rule matching `Mul([Sin(w), Pow(Cos(w), -1)])` |
| B2 | `sinh(w)/cosh(w) → tanh(w)` | Ratio recognition | Same pattern |
| B3 | `exp(a)*exp(b) → exp(a+b)` | Exp combining | New rule — requires Mul sub-expression matching (B5) |
| B4 | `logcombine()` method | Log collection | `ln(a)+ln(b) → ln(a*b)` — new method `Ex::logcombine()`, reverse of `expand_log()` |
| B5 | Mul sub-expression matching | Infrastructure | Extend `apply_rules` in `pattern.rs` to try rule patterns against pairs of Mul factors (same mechanism as existing Add sub-match) |

**Files:** `src/pattern.rs` (B1-B3, B5), new `src/log_combine.rs` (B4), `src/expr.rs` + `src/arena.rs` (wiring)

**Effort:** ~2 hours total

### Sprint C: Eval Completeness

**Gap analysis:** We evaluate sin/cos at some unit circle angles but not all 16. Missing angles and the strategy to cover them:

**Supplementary angle approach (covers all missing angles with ~10 lines):**
- Add to `eval_sin`: if `coeff > 1/2 && coeff < 1`, compute `sin(π - kπ) = sin((1-k)π)` — i.e., reduce to the first quadrant using `sin(π-x) = sin(x)`.
- Add to `eval_cos`: `cos(π-x) = -cos(x)`.
- With these two symmetry rules plus existing values at 0, π/6, π/4, π/3, π/2, ALL 16 standard angles are covered automatically.

**Additional missing eval values:**
| # | Item | Values |
|---|------|--------|
| C1 | `sin(2π/3)`, `sin(3π/4)`, `sin(5π/4)`, etc. | All covered by supplementary angle rule |
| C2 | `cos(2π/3)`, `cos(3π/4)`, `cos(5π/6)`, etc. | All covered by supplementary angle rule |
| C3 | `tan(π/6) = √3/3`, `tan(π/3) = √3` | Add to `eval_tan` table |

**Files:** `src/eval.rs`

**Effort:** ~45 minutes total

### Sprint D: Ergonomics

| # | Item | Description | Effort |
|---|------|-------------|--------|
| D1 | `Ex::zero()`, `Ex::one()` | Class methods returning 0 and 1 using global default context | 5 min |
| D2 | `Ex::expr_type() -> ExprType` | Structural query enum: `Number, Symbol, Constant, Add, Mul, Pow, Neg, Function, Derivative, Integral, Apply` | 15 min |
| D3 | `Ex::replace(closure)` | `replace(\|e\| Option<Ex>)` — user-provided transformation walk using `walk_and_rebuild` infrastructure | 20 min |
| D4 | `examples/calculus.rs` | Real workflow example showing diff→integrate→series→solve pipeline | 20 min |

**Files:** `src/expr.rs` (D1-D3), `examples/calculus.rs` (D4)

**Effort:** ~1 hour total

### Sprint Summary

| Sprint | Items | Effort | Impact |
|--------|-------|--------|--------|
| A (Integration) | 9 items | 2 hr | Handles all standard calculus textbook integrals |
| B (Simplification) | 5 items | 2 hr | Ratio recognition, exp combining, logcombine, Mul sub-match |
| C (Eval) | 3 items | 45 min | Complete unit circle (all 16 angles) |
| D (Ergonomics) | 4 items | 1 hr | Convenience constructors, structural queries, examples |
| **Total** | **21 items** | **~5.75 hr** | |

After this sprint, the library will:
- Integrate ALL standard forms from a calculus textbook (tan, ln, inverse trig/hyp, rational via apart)
- Simplify trig ratios (`sin/cos → tan`) and exp products (`exp(a)*exp(b) → exp(a+b)`)
- Evaluate the complete unit circle (all 16 standard trig angles)
- Have convenience constructors (`zero`, `one`), structural queries (`expr_type`), and guided transformation (`replace`)

### File Ownership Plan for Next Sprint

```text
BATCH 1 (parallel — mutually exclusive files):
  Agent A: src/integrate.rs           — Sprint A items (all integration)
  Agent B: src/pattern.rs             — Sprint B items B1-B3, B5 (rules + Mul sub-match)
  Agent C: src/eval.rs                — Sprint C items (supplementary angles, tan values)

BARRIER

BATCH 2 (parallel — mutually exclusive NEW files):
  Agent D: src/log_combine.rs (NEW)   — Sprint B item B4 (logcombine)
  Agent E: examples/calculus.rs (NEW) — Sprint D item D4

BARRIER

BATCH 3 (sequential — shared files):
  Self: src/lib.rs                    — register log_combine module
  Self: src/arena.rs                  — delegation methods
  Self: src/expr.rs                   — Sprint D items D1-D3 + wire logcombine

BATCH 4 (parallel — NEW test files):
  Agent F: tests/test_integration_complete.rs (NEW)
  Agent G: tests/test_simplify_advanced.rs (NEW)
```

---

## Roadmap

### 0.2.x series (incremental, non-breaking)

| Version | Theme | Features | Est. |
|---------|-------|----------|------|
| 0.2.0 ✅ | Type safety | Phantom types, BoolEx, relationals, logic, piecewise, 11 new ExprNode variants | Done |
| 0.2.1 | Completeness | Inequality solving (returns BoolEx), floor/ceil/min/max functions | ~500 lines |
| 0.2.2 | Linear algebra | gradient/divergence/curl/laplacian, eigenvalues, characteristic polynomial | ~400 lines |
| 0.2.3 | Discrete math | Symbolic Sum/Product nodes, closed-form evaluation (Gosper) | ~500 lines |
| 0.2.4 | Roots | Cubic formula (Cardano), quartic formula (Ferrari) | ~300 lines |

### 0.3.0 (major)

| Feature | Description | Est. |
|---------|-------------|------|
| Set type | Interval, FiniteSet, Union, Intersection, EmptySet, Reals | ~800 lines |
| Expr<SetValued> sort | Third phantom type marker for set-valued expressions | ~200 lines |
| solveset | Returns Set instead of Vec<Ex> | ~300 lines |
| as_relational ↔ as_set | Bidirectional bridge between boolean conditions and sets | ~200 lines |

### 0.4.0+ (deep algorithms)

| Feature | Description |
|---------|-------------|
| ~~Gruntz algorithm~~ | ✅ **Done** (Commit 94) — ~1500 lines, handles all exp-log functions |
| Risch integration | Decision procedure for elementary antiderivatives |
| Hensel factoring | Full polynomial factoring over ℤ |
| Special functions | Gamma, erf, Bessel node types |
| Multivariate polynomials | Gröbner bases, multivariate GCD |

### Strategic Context

Symplex is the only MIT/Apache-2.0 general-purpose CAS in Rust. There is no direct competitor. The closest alternative is Python interop with SymPy, which has massive overhead. Specialized crates (`num-bigint`, `egg`, `nalgebra`) handle only parts of what a CAS does.

**Likely early adopters:**
1. Robotics/control engineers — need symbolic Jacobians, coordinate transforms, code generation
2. Physics students — need homework-level calculus, series, ODEs
3. Compiler/PL researchers — need term rewriting, optimization
4. Numerical algorithm developers — need to derive formulas then compile to fast code

**The common thread:** Most Rust CAS users want to **derive a formula symbolically, then compile it to numerical code.** This differs from SymPy users who stay in the symbolic world.

### Architecture Decisions for Next Releases

**Formatting:** The `symplex-format` crate will consume `ExprTree` (serde) from the core. It does NOT depend on `Ex` or `Arena` — only on the serialized tree structure. This means:
- Core crate has zero formatting dependencies
- Format crate can evolve independently
- Third-party formatters can be built by anyone consuming ExprTree

**User-defined functions:** The `MathFunction` trait will be registered on `Context` and stored alongside `Apply` nodes. When `diff` encounters `Apply(f, args)`, it checks if `f` has a registered `MathFunction` with a `derivative` callback. This avoids new ExprNode variants.

**CancelToken:** A `Arc<AtomicBool>` threaded through recursive operations. Each loop iteration checks the flag. When set, operations return `Err(SymplexError::Cancelled)`. The token is optional — passing `None` means no timeout.

**Code generation:** `Ex::to_rust_fn(args: &[&str]) -> String` produces a Rust function body that evaluates the expression numerically. This is the bridge between symbolic derivation and numerical execution that Rust users specifically need.

---

## Development Conventions

### Parallel work policy

When multiple implementation tasks are in progress simultaneously (e.g., via parallel agents or contributors), each task **must** operate on a mutually exclusive set of files. If two tasks need to edit the same file, they must be serialized — one completes before the other begins. This prevents merge conflicts, stale-state overwrites, and silent data loss.

When planning parallel work, list the file ownership explicitly:

```text
Task A: src/arena.rs (exclusive)
Task B: src/walk.rs, src/context.rs (exclusive)
Task C: symplex-macros/Cargo.toml, symplex-macros/src/parse.rs (exclusive)
── BARRIER ──
Task D: src/expr.rs (exclusive, depends on A+B)
Task E: symplex-macros/src/lib.rs (exclusive, depends on C)
```

### Visibility defaults

- New `Arena` methods default to `pub(crate)`. Promote to `pub` only when needed by `context.rs`, `expr.rs`, or generated macro code.
- `intern()` and `intern_num()` are `pub(crate)`. External access goes through typed constructors (`int`, `symbol`, `sin`, etc.) or pre-interned constant fields.

### Operator implementation via macros

All `Ex` operator implementations use internal macros (`impl_nary_binop`, `impl_binary_binop`, `impl_nary_binop_i64`, `impl_binary_binop_i64`) to guarantee that every ownership combination (`Ex⊕Ex`, `&Ex⊕Ex`, `Ex⊕&Ex`, `&Ex⊕&Ex`, plus all `i64` variants) is generated uniformly. Never add manual operator impls — always extend or add a macro.

### Error discipline

- `SymplexError` is `#[non_exhaustive]`. New variants can be added without breaking downstream `match` arms.
- `NotImplemented` is reserved for genuinely unimplemented features. Domain errors (free symbols, unevaluable nodes, precision limits) should get dedicated variants.
- Every `Result`-returning method must have a `# Errors` doc section.

### Test organization

- Every new public method ships with at least 3 tests: happy path, edge case, and error/empty case.
- Property-based tests (proptest) should cover algebraic invariants for new transformations.

### No names in source code

Never put personal names, author attributions, or team member references in source files, test files, comments, or doc comments. Use `git blame` for attribution. The only place names appear is `TEAM.md` (the design panel reference document). This keeps the codebase clean, avoids attribution disputes, and ensures automated tools don't flag name strings as PII.

### Mathematical convention documentation

Every non-obvious mathematical choice (e.g., `0^0 = 1`, `ComplexInfinity + finite = NaN`) must be documented in the Design Decisions section with rationale.

---

## SymPy Gap Analysis — Priority Roadmap

Based on comprehensive comparison with SymPy's ~40 modules.

### Critical gaps (typical CAS users expect these)

1. **Sets** (Interval, FiniteSet, Union, Reals) — foundation for solveset and domains
2. **Symbolic sums & products** (Σ, Π with Gosper's algorithm for closed forms)
3. **Inequality solving** — return relational expressions or intervals
4. **Cubic/quartic formulas** — complete polynomial root finding
5. **Definite integrals** (improper, with convergence)

### High-priority gaps (power users and STEM students)

6. **Multivariate polynomials** — needed for systems
7. **Full polynomial factoring** (Hensel, Zassenhaus) — factor over ℤ
8. **Eigenvalues / eigenvectors** — linear algebra courses
9. ~~**Gruntz algorithm** — robust limits at infinity~~ ✅ **Done** (Commit 94, ~1500 lines)
10. **Special functions** (gamma, erf, Bessel) — physics/engineering
11. **Vector calculus** (gradient, divergence, curl) — multivariable calc
12. **Risch/heuristic integration** — handle more integrands

### What symplex does better than SymPy

- Arena hash-consing: O(1) equality, structural sharing
- Compile-time sort safety: Expr<Numeric> vs Expr<Boolean>
- Thread safety: Send + Sync
- No recursion: explicit stacks prevent stack overflow
- Canonical invariant checker: catches bugs at construction time
- Proc macro DSL: expr!, rule!, matrix!, eq!
- Arbitrary-precision parser (0.1+0.2=3/10 exactly)
- Type-safe expression views (ExprView prevents deadlock at compile time)
- Condition-guarded simplification rules (pow_pow, abs_positive)
- LIATE-ordered integration by parts with depth limit

## File Layout

```
symplex/
├── Cargo.toml
├── README.md
├── IMPLEMENTATION_PLAN.md
├── symplex-macros/                    Proc macro crate
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                     expr!, rule! entry points + codegen
│       └── parse.rs                   Shared Pratt parser
├── src/
│   ├── lib.rs                         Module declarations, prelude, __macro_support
│   ├── arena.rs                       Expression arena + hash-consing
│   ├── assumptions.rs                 Three-valued property inference engine
│   ├── canon.rs                       Canonical-form constructors
│   ├── config.rs                      EvalConfig
│   ├── context.rs                     Context (user-facing entry point)
│   ├── diff.rs                        Symbolic differentiation
│   ├── display.rs                     Iterative expression pretty-printer
│   ├── errors.rs                      SymplexError
│   ├── eval.rs                        Special-value evaluation
│   ├── evalf.rs                       Arbitrary-precision numerical evaluation
│   ├── expand.rs                      Algebraic expansion
│   ├── expr.rs                        Ex (expression handle)
│   ├── macros.rs                      syms! and sym! declarative macros
│   ├── node.rs                        ExprId, NumId, SymbolId, CtxId, ExprNode
│   ├── pattern.rs                     Pattern matching + rewrite rules
│   ├── poly.rs                        Dense univariate polynomials over ℚ
│   ├── polybridge.rs                  Expression ↔ Poly bridge + cancel()
│   ├── solve.rs                       Polynomial equation solver
│   ├── sort_key.rs                    Canonical ordering
│   ├── subs.rs                        Structural substitution
│   ├── symbol.rs                      Symbol table
│   └── walk.rs                        Shared iterative tree traversal
├── tests/
│   ├── proptest_canon.rs              Property-based tests
│   ├── test_bug_regression.rs         Regression tests
│   ├── test_macros.rs                 Proc macro integration tests
│   ├── test_stage1.rs                 Stage 1 integration tests
│   ├── test_stage3.rs                 Stage 3 integration tests
│   ├── test_stage5.rs                 Stage 5 integration tests
│   ├── test_stage6.rs                 Stage 6 integration tests
│   └── test_stage7_8.rs              Stages 7-8 integration tests
└── benches/
    └── canonicalization.rs            (stub)
```
