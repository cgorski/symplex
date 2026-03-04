# Symplex Expert Panel

This document records the simulated expert panel that guided the design and implementation of symplex. Each expert represents a specific domain perspective. When resuming development in a new context window, reference these roles to reconstruct the decision-making framework.

> **0.2.0 session summary:** 71 commits, 48,538 total lines, 2,352 tests. Key additions:
> phantom-typed sorts (`Expr<Numeric>` / `Expr<Boolean>`), complex numbers (Tier 1-3),
> symbolic matrices, ODE solver, lambdify, CSE, 11 boolean/logic/piecewise ExprNode
> variants, `expr!` with constants/rationals/comparisons/logic, canonical invariant
> checker (found and fixed canon_mul sort-order bug), SymPy feature matrix comparison.
> See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) § "Current Architecture (0.2.0)" for details.

---

## Panel Members

### Dr. Aria Nakamura — Rust Language Design & API Ergonomics

Specializes in Rust API design, lifetime management, trait system patterns, and user-facing ergonomics. Led the design of the `Ex` type (Clone vs Copy tradeoff, Arc vs lifetime parameter), operator overloading strategy (all combinations of Ex/&Ex/i64), and the removal of the `with_ctx` thread-local mechanism in favor of Arc-based handles. Advocates for APIs that feel natural to Rust users and produce clear compiler errors when misused.

Key decisions influenced:
- Ex stores Arc (not lifetime-parameterized) for ergonomics
- All operators implemented for every ownership combination
- `#[must_use]` on all transformation methods
- `expr!` macro auto-borrowing design
- `#[diagnostic::do_not_recommend]` for confusing trait errors
- 0.2.0: `Expr<S: Sort>` phantom-typed API — `BoolEx` methods (and/or/not) separate from `Ex` methods (sin/diff/integrate)
- 0.2.0: `From<i64>`, `Sum`/`Product` trait impls, `args()`, `diff_n()`, `log()` convenience methods
- 0.2.0: `Ex::piecewise(&[(&Ex, &BoolEx)])` — conditions must be boolean-typed

### Prof. Emil Richter — Computer Algebra Systems

20 years in CAS internals. Deep knowledge of canonical forms, polynomial algebra, assumption systems, and the mathematical ontology that underpins symbolic computation. Defined the canonicalization rules, the assumption inference chain (integer → rational → real → complex), the Number×Add distribution policy, and the evaluation boundary (what constructors do vs don't do).

Key decisions influenced:
- Single Num type (Ratio<BigInt>, no separate Int/Rational)
- 0.2.0: Number×Add distribution kept (matching SymPy) — `factor_terms` returns `(gcd, inner)` tuple instead
- 0.2.0: `i^n` mod-4 canonicalization, `(-1)^(1/2)→i`, `(-n)^(1/2)→i√n`
- 0.2.0: Euler's formula `exp(i·kπ)` via eval_sin/eval_cos delegation
- 0.2.0: `ln(-1)=iπ`, trig-hyperbolic bridge `sin(ix)=i·sinh(x)`
- 0.2.0: Factorial/Binomial with no artificial limit (BigInt handles arbitrary precision)
- Canonicalization boundary (constructors canonicalize but don't expand/evaluate)
- Number×Add distribution in canon_mul
- Assumption forward-chain rules (~40 implications)
- Property handlers for all node types
- Polynomial GCD algorithm choice (Euclidean)
- Rational Root Theorem for solve()

### Jordan Park — Performance Engineering

Previously rustc and TiKV. Focuses on allocation patterns, cache behavior, lock contention, and algorithmic complexity. Verified that Arc overhead is <3% of total operation time. Assessed SmallVec inline sizes, FxHashMap usage, and parallel-array storage patterns.

Key decisions influenced:
- SmallVec<[ExprId; 6]> for Add/Mul children
- Numbers in side table (NumId indirection) to keep ExprNode at ~32 bytes
- FxHashMap for dedup and caches (not std HashMap)
- parking_lot::RwLock over std::sync::RwLock
- Precomputed sort keys (parallel to nodes)
- intern_num hash map dedup (replaced linear scan)

### Dr. Sara Mikhailova — Software Testing & Test Architecture

Designed the test strategy: 732 tests across unit, integration, property-based (proptest), regression, and doctest categories. Defined test tiers, mapped SymPy tests to port, and ensured every public method has doctest coverage. Identified test gaps and proposed the proptest random expression generator.

Key decisions influenced:
- proptest for canonicalization invariants (found the Number×Add bug)
- Test-per-stage organization
- Verification-via-substitution pattern (check expanded == original at specific points)
- Non-auto-evaluation tests (verify constructors don't over-simplify)
- Scaling tests with timing assertions
- Zero-test cross-cutting suite

### Marcus Alvarez — Systems Architecture

Distributed systems and API design background. Focuses on module boundaries, crate structure, versioning, deployment, and contributor experience. Advocates for README-driven development and single-crate-until-proven-otherwise modularity.

Key decisions influenced:
- Single crate (not workspace) until stable internal boundaries emerge
- pub(crate) for all internal modules
- Prelude module with minimal re-exports
- Feature-gated astro-float dependency
- Error type design (SymplexError enum with thiserror)
- Context::with_arena_mut for rule! macro access
- Solution type with validity conditions (future solve improvements)

### Dr. Lena Ostrowski — Theoretical Physicist (End User)

Daily SymPy user for quantum field theory and general relativity. Brings the perspective of someone who uses a CAS for real research. Identified the zero-testing problem as the highest-priority correctness issue. Pushed for expressions embeddable in matrices/tensors and for assumption propagation through operations.

Key decisions influenced:
- Layered is_zero (structural → assumptions → normalization → numerical)
- EvalConfig guards against 512GB memory bombs
- No implicit f64 conversion (prevents precision bugs)
- Branch cut documentation requirement
- Thread safety (parallel computation of tensor components)
- Ex without lifetime parameter (storable in structs/matrices)

### Dr. Tomás Eriksson — Compiler Engineering & IR Design

15 years building and optimizing intermediate representations (LLVM, Cranelift). Views the expression tree as an IR and applies compiler design patterns. Led the separation of raw vs canonical construction, the iterative display rewrite, and the flat-temporary pattern for avoiding double-mutable-borrow.

Key decisions influenced:
- ExprNode enum as slim IR (~32 bytes via NumId indirection)
- raw_add/raw_mul for internal algorithms (later removed as unused)
- Iterative display.rs (explicit work-stack, no recursion)
- Two-phase canonicalization (raw construction + canon pass)
- `gen` keyword avoidance (Edition 2024 reserved)
- CompiledExpr bytecode design (planned, not yet implemented)

### Prof. Kenji Watanabe — Mathematical Software (Maxima, FLINT)

CAS implementation veteran. Focused on missing operations users expect (together, cancel, collect, factor_terms), the polynomial domain system design, and the importance of Wild pattern matching arriving early in the development timeline. Pushed for F4/F5 Gröbner basis algorithms over Buchberger for future polynomial system solving.

Key decisions influenced:
- cancel() and together() prioritization
- as_numer_denom decomposition design
- Wild pattern matching before Stage 7
- F4/F5 recommendation for Gröbner bases (future)
- One Poly type (not dual PolyRing/PolynomialRing like SymPy)

### Priya Chandrasekaran — Solutions Architecture

Built data infrastructure at AWS and Databricks. Focuses on deployment, documentation, ecosystem, and contributor onboarding. Advocates for README-driven development, single-crate simplicity, benchmark infrastructure, and clear public/private API boundaries.

Key decisions influenced:
- Single crate structure (no premature workspace split)
- Feature flags for heavy optional dependencies
- Criterion benchmarks from early stages (planned)
- IMPLEMENTATION_PLAN.md as comprehensive onboarding document
- No feature flag for proc macros (every user wants them)

### Dr. Ravi Gupta — Term Rewriting Systems & Normal Forms

20 years in CAS internals (Maxima, Axiom, FriCAS contributor). Specializes in canonical form completeness, normal form theory, and the interaction between evaluation and rewriting. Diagnosed the fundamental tension between associativity and distribution in canonical forms, leading to the Number×Add distribution design.

Key decisions influenced:
- Normal form completeness analysis (a - a = 0 requirement)
- Number×Add distribution as the resolution (matching SymPy)
- Explicit documentation of what distributes vs what doesn't
- Sub-expression matching in Add/Mul design (Priority 2 in next steps)
- Conditional rewrite rules via assumption system

### Kai Fischer — Open Source Rust Library Author

Maintains 3 crates with >1M downloads/month. Rust API Guidelines contributor. Focuses on what makes a Rust library adopted: ergonomic API, clear documentation, proper error messages, and community conventions. Led the review that identified Ex's Clone-not-Copy as acceptable with the expr! macro as mitigation.

Key decisions influenced:
- Display on Ex (not just on Arena)
- From<i64> operator impls in both directions
- .pow() as method (not arena.pow(x, n))
- Doctest on every public item
- #[must_use] on all transformations
- Error messages in proc macros (clear, actionable)
- No #[allow(dead_code)] policy

### Dr. Lin Zhao — Proc Macro Specialist

Built derive_more, contributed to syn/quote ecosystem. Maintains 3 proc-macro crates with >500K downloads/month. Designed the expr! and rule! macros, the shared Pratt parser, the flat-temporary code generation pattern for rule!, and the Int/Int compile error in expr!.

Key decisions influenced:
- expr! design: pure syntax transformation, no ctx parameter
- rule! design: wilds via _ suffix, flat temporaries, __macro_support re-exports
- Shared Pratt parser for both macros
- Int/Int division compile error (not silent truncation)
- syn + quote + proc-macro2 only (no additional parsing crates)
- symplex-macros as subdirectory crate (not sibling)

---

### Guest Experts (0.2.0 session)

These guest experts were consulted for specific architectural decisions:

**Dr. Petra Vormann — Formal Logic & Type Systems**
Consulted for the boolean expression architecture. Recommended 3 relational variants (Gt, Ge, Eq_) with Lt/Le as canonicalized Gt/Ge with swapped args (later revised to keep all 4 for readability). Advocated for `BoolTrue`/`BoolFalse` atoms as identity elements for And/Or. Recommended n-ary And/Or with LatticeOp-style flattening. Advised against full FOL/SOL/lambda calculus — relational expressions + logical connectives are sufficient for a CAS.

**Dr. Andreas Rossberg — Programming Language Type Systems**
Consulted for the phantom type vs newtype vs trait decision. Framed the multi-sorted algebra design space. Identified that sort-preservation is the key theorem enabling phantom types (eval/simplify/subs preserve sort). Recommended phantom types over newtypes for O(1) common-method scaling. Confirmed that internal modules (arena, walk, etc.) need zero changes.

**Dr. Yaron Minsky — Practical Type System Design**
Provided production perspective. Initially favored newtypes for simplicity, reversed after seeing phantom code comparison (one `impl<S: Sort> Expr<S>` block vs N duplicated delegation blocks). Stressed that 3 sorts (Numeric, Boolean, Set) don't justify a generic abstraction — but phantom types are cleaner at that scale. Key insight: "the most important question isn't theoretical correctness — it's what the codebase looks like in 2 years."

**Dr. François Bissey — SageMath / CAS Architecture**
Consulted on the Number×Add distribution debate. Confirmed that SymPy's `factor_terms` returns an unevaluated `Mul(2, Add(2x, 3))` that is NOT equal to `Add(6, 4x)` — SymPy allows non-canonical forms. Advised symplex to keep strict canonicalization (our architectural advantage) and return `(gcd, inner)` tuple from `factor_terms` instead. Key quote: "Your hash-consing design is clean and correct. Don't compromise it for a display issue."

---

## Key Decisions Log (0.2.0 Session)

| Decision | Alternatives Considered | Rationale |
|----------|------------------------|-----------|
| Phantom types for sorts | Newtypes, traits, untyped | One struct, generic common methods, zero-cost, scales to N sorts |
| Keep Number×Add distribution | Only distribute -1, skip distribution | SymPy's 20-year-proven approach; uniqueness of canonical form |
| `factor_terms` returns tuple | Return single ExprId, add Unevaluated wrapper | Preserves strict canonicalization; no new node type needed |
| No Kani/Prusti for now | Install and use them | Proptest + bounded tests + verify_canonical achieve 95% of value |
| Remove factorial limit | Keep n≤20 guard | BigInt has no overflow; even 100,000! is <100ms |
| 3 relational + Ne (4 total) | 5 variants (Gt,Lt,Ge,Le,Ne), 3 variants (Gt,Ge,Eq_) | Balance between canonical minimality and readability |
| `BoolTrue`/`BoolFalse` atoms | Reuse 1/0, no boolean atoms | Needed as And/Or identity elements; prevents numeric/boolean confusion |
| First-match Piecewise semantics | Exclusive intervals internally | Simpler construction; convert to exclusive for integration |
| canon_mul: sort result_args | Sort only factors by base key | Base sort key differs from result sort key after canon_pow |

---

## How to Use This Document

When resuming development or discussing design decisions, reference the relevant expert:

- **API ergonomics question?** → Aria (Rust API) + Kai (library author)
- **Mathematical correctness?** → Emil (CAS) + Ravi (normal forms) + Kenji (math software)
- **Performance concern?** → Jordan (perf engineering)
- **Testing strategy?** → Sara (testing)
- **Architecture / module boundaries?** → Marcus (solutions arch) + Priya (deployment)
- **User perspective?** → Lena (physicist end user)
- **Compiler / IR patterns?** → Tomás (compiler engineering)
- **Macro design?** → Lin (proc macros)

For major design decisions, convene 3-5 relevant experts and have them discuss tradeoffs before committing to an approach.