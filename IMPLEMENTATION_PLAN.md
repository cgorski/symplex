# Symplex Implementation Plan

## Overview

Symplex is a symbolic mathematics library for Rust. It provides an arena-interned expression tree with hash-consing, canonical ordering, a three-valued assumption system, symbolic differentiation, pattern matching with rewrite rules, algebraic expansion, special-value evaluation, arbitrary-precision numerical evaluation, polynomial algebra with GCD, fraction cancellation, polynomial equation solving, and proc macros for ergonomic expression building and rule definition.

This document describes the architecture, design decisions, module responsibilities, data flow, concurrency model, and planned future work. It is intended to be sufficient for a new contributor to understand the codebase and begin development without prior context.

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

### Main crate: `symplex/src/` (23 modules, ~12,900 lines)

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

---

## Known Limitations

1. **Pattern matching is structural only.** `sin²(w) + cos²(w)` matches a 2-term Add but not inside `3 + sin²(x) + cos²(x)`.
2. **No polynomial factoring.** GCD and rational root finding only.
3. **Limited complex number support.** `ImaginaryUnit` exists but `evalf` errors on complex expressions.
4. **No integration.** No antiderivatives.
5. **No series expansion.** No Taylor/Laurent.
6. **No limit computation.** No Gruntz algorithm.
7. **`solve()` is polynomial-only.** Transcendental equations not handled.
8. **`simplify()` has limited rules.** Pythagorean identity only via built-in. More rules can be added via `rule!` macro.
9. **`bigint_to_bigfloat` loses precision for integers > i128.** Falls back to f64.
10. **No `collect()`, `together()`, or `factor_terms()` yet.**
11. **`expr!(1/2)` is a compile error.** By design — prevents silent Rust integer division. Use `ctx.rational(1, 2)`.
12. **`expr!(x^2^3)` with nested integer powers causes type errors.** The inner `2^3` evaluates as integer arithmetic, not symbolic.

---

## Detailed Next Steps

### Priority 1: More Simplification Rules (Immediate)

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

### Priority 2: Sub-expression Matching in Add/Mul

**Goal:** Enable `sin²(x) + cos²(x) → 1` inside larger sums like `3 + sin²(x) + cos²(x) → 4`.

**Implementation approach:**

1. In `apply_rules` (pattern.rs), when a rule fails to match an `Add` node, try matching the rule's LHS against all C(N, K) subsets of the Add's K terms, where K is the number of terms in the pattern's Add.
2. For K=2 (the common case), this is O(N²) where N is the number of terms in the expression's Add. Acceptable.
3. If a subset matches, rebuild the Add with the matching terms replaced by the rule's RHS and the remaining terms kept.

**Where to edit:** `src/pattern.rs`, function `apply_rules`. Add a new path after the current `try_apply` fails: if the node is an Add and the rule's pattern root is an Add, iterate over pairs of terms.

**Key data structures:** The `match_pattern` function already handles matching. The new code just needs to try it on different subsets of the Add's children.

**Effort:** ~2 hours. Requires careful handling of the remaining terms.

### Priority 3: `collect(var)` — Group by Powers of a Variable

**Goal:** `collect(x² + 2xy + y², x)` → `x² + 2y·x + y²` (grouped by powers of x).

**Implementation:**

1. Call `expr_to_poly(arena, expr, var)` to get a `Poly`.
2. Call `poly_to_expr(arena, &poly, var)` to rebuild — this naturally groups by power.
3. If `expr_to_poly` fails (not polynomial), return unchanged.

**Where to edit:** Add `collect` to `polybridge.rs` and wire through `arena.rs` and `expr.rs`.

**Effort:** ~30 minutes. Uses existing Poly infrastructure.

### Priority 4: `together()` — Common Denominator

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

### Priority 5: Basic Integration

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

### Priority 6: Series Expansion

**Goal:** `series(exp(x), x, 0, 5)` → `1 + x + x²/2 + x³/6 + x⁴/24 + O(x⁵)`.

**Implementation:**

1. Create `src/series.rs`.
2. Use repeated differentiation + division by factorial: `f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + ...`
3. Each term is computed by `diff` (already implemented) and `subs` (already implemented).
4. Truncate at the requested order.
5. Return as a `Poly` or as an `Ex` expression.

**Where to edit:** New file `src/series.rs`.

**Effort:** ~3 hours for Taylor expansion. Laurent series and asymptotic series are more complex.

### Priority 7: Polynomial Factoring

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

### Infrastructure Tasks

| Task | Where | Effort |
|------|-------|--------|
| Criterion benchmarks | `benches/canonicalization.rs` | 1 hour |
| Split `assumptions.rs` | Into `assumptions/mod.rs`, `assumptions/inference.rs`, `assumptions/handlers.rs`, `assumptions/cache.rs` | 1 hour |
| CI configuration | `.github/workflows/ci.yml` | 30 min |
| `CancelToken` for timeouts | New `src/cancel.rs`, integrate into expand/solve/simplify | 2 hours |
| Complex number support | Track re/im parts, complex evalf | 8 hours |

---

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
