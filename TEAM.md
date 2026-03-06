# Symplex Expert Panel

The expert panel is a **design-time framework** — a set of domain-specialist roles
used to guide architectural decisions during development. Panel members are simulated
experts, not real people. They do not appear in source code, tests, or comments.

When resuming development in a new context window, reference these roles to
reconstruct the decision-making framework for specific domains.

---

## No Names in Source Code

**Never put panel member names, author names, or reviewer names in source code,
tests, comments, doc comments, IMPLEMENTATION_PLAN.md, CHANGELOG.md, or README.md.**
The panel is a design-time tool only — it does not appear in shipped artifacts.

- ❌ `// Reviewed by Dr. Vasquez`
- ❌ `//! Author: Prof. Lindqvist`
- ✅ `//! Tests for denominator rationalization and limit edge cases.`

The ONLY file where panel member names may appear is this file (TEAM.md).
Use `git blame` for code attribution.

---

## Panel Members

### Dr. Elena Vasquez — CAS Algorithms & Mathematical Correctness

**Background:** Computational algebraist with deep experience in symbolic
computation engines. Previously worked on canonicalization strategies for
commercial CAS systems. Published on polynomial ideal theory and Gröbner basis
optimizations. Holds strong opinions about the eval-before-evalf pipeline and
canonical form invariants.

**Specialty areas:** Canonicalization rules, assumption inference, polynomial
algebra, special function evaluation, integration algorithms, equation solving
strategy, Gruntz algorithm design, Stirling series for Gamma evaluation.

**Consulting trigger:** Any question about whether a mathematical transformation
is correct, what the canonical form of an expression should be, or how a CAS
algorithm should behave at edge cases.

**Known positions:**
- Insists on exact arithmetic everywhere — no floating-point contamination
- Advocates for eval-before-evalf as inviolable principle
- Pushed for Stirling over Lanczos (Decision #3)
- Skeptical of heuristic simplification — prefers provably correct rewrites

---

### Marcus Lindqvist — Rust API Design & Ergonomics

**Background:** Senior Rust systems engineer focused on API surface design,
trait coherence, and zero-cost abstraction patterns. Extensive experience
designing generic APIs with phantom types and marker traits. Strong background
in operator overloading ergonomics and proc macro design. Believes the best
API is the one users reach for naturally.

**Specialty areas:** Clone vs Copy trade-offs, operator overload combinations,
phantom type systems, `#[must_use]` policy, `expr!` macro UX, the `Sort` trait
hierarchy, `ExprView` design for deadlock-free closures.

**Consulting trigger:** Any question about the public API surface, how users
will interact with types, or whether a method signature feels natural.

**Known positions:**
- Championed `Expr<S: Sort>` phantom types over newtypes (Decision #1)
- Insisted Ex be Clone not Copy — Arc enables thread safety (Decision #8)
- Pushed for `expr!` macro to mitigate clone/reference noise
- Advocates for `pub(crate)` by default, promote to `pub` only when proven needed

---

### Dr. Priya Chandrasekaran — Testing Strategy & Quality Assurance

**Background:** Testing methodologist with expertise in property-based testing,
mutation testing, and test suite quality metrics. Previously designed test
frameworks for safety-critical numerical libraries. Deep experience with
proptest, quickcheck, and fuzzing infrastructure. Believes tests should verify
mathematical invariants, not display formatting.

**Specialty areas:** Test tier design, proptest strategy generation, value-preservation
properties, negative tests, FTC (Fundamental Theorem of Calculus) verification
patterns, silent-bailout detection, test coverage gap analysis.

**Consulting trigger:** Any question about test design, assertion quality,
whether a test actually proves what it claims to prove, or what properties
a new feature should be tested against.

**Known positions:**
- Vocally against `format!("{}")` as a correctness oracle — numerical cross-validation preferred
- Invented the `assert_ftc` pattern used in integration tests
- Caught the eval-before-evalf bug by refusing to accept a weakened test assertion
- Advocates for multi-point numerical evaluation (minimum 3 points) over single-point
- Insists every `if let Ok` in a test is a potential silent pass that must be justified

---

### James Okafor — Performance, Allocation & Concurrency

**Background:** Systems performance engineer specializing in memory layout,
cache behavior, and lock contention in concurrent systems. Background in
garbage collector design and arena allocation strategies. Experienced with
`parking_lot`, lock-free data structures, and `loom` for concurrency testing.
Profiles before optimizing, measures after.

**Specialty areas:** SmallVec sizing decisions, FxHashMap vs BTreeMap trade-offs,
arena compaction heuristics (BitVec liveness), `Arc` overhead analysis,
`RwLock` contention patterns, `ExprView` non-locking design, cache-friendly
node layout, benchmark design.

**Consulting trigger:** Any question about allocation patterns, whether a data
structure choice affects performance, concurrency safety, or GC/compaction design.

**Known positions:**
- Designed BitVec liveness for arena GC over refcounting (Decision #5)
- Pushed for single `RwLock` with structural lock ordering over fine-grained locks
- Advocates for Criterion benchmarks on every performance-sensitive path
- Skeptical of premature optimization — wants profiles before changes

---

### Dr. Sophie Marchand — Architecture, Module Boundaries & Versioning

**Background:** Software architect with experience designing large Rust crate
ecosystems. Focus on module boundary design, visibility policy, error type
design, and crate versioning strategy. Strong opinions about `#[non_exhaustive]`,
semver compliance, and the tension between internal flexibility and public
API stability.

**Specialty areas:** `pub(crate)` vs `pub` boundary decisions, single-crate vs
workspace policy, `SymplexError` design, `ExprNode` variant organization (Apply
vs dedicated variant), file size policy, parallel agent coordination, IMPLEMENTATION_PLAN
maintenance.

**Consulting trigger:** Any question about where code should live, how modules
should interact, what should be public, or how to version breaking changes.

**Known positions:**
- Enforces the six-layer architecture (API → Macros → Transforms → Canon → Arena → Types)
- Designed the Apply vs ExprNode boundary (calculus-capable = ExprNode, eval-only = Apply)
- Pushed for `#[non_exhaustive]` on all error types
- Advocates for the "strangler fig" pattern for legacy test file migration
- Wrote the parallel agent policy (§5 of IMPLEMENTATION_PLAN)

---

### Dr. Tomás Herrera — Compiler Patterns, IR Design & Code Generation

**Background:** Compiler engineer with expertise in intermediate representation
design, tree rewriting systems, and code generation. Background in bytecode
VM design and common subexpression elimination. Experienced with `syn`/`quote`
for proc macro development. Thinks about expression trees the way compiler
engineers think about SSA form.

**Specialty areas:** `ExprNode` sizing and variant layout, iterative display
(non-recursive pretty printing), `lambdify` bytecode VM design, CSE algorithm,
`to_rust_fn()` code generation, `walk.rs` traversal infrastructure, sort key
design for canonical ordering.

**Consulting trigger:** Any question about how expressions are represented
internally, how tree walks should be structured, or how to generate code
from symbolic expressions.

**Known positions:**
- Insisted on iterative (explicit stack) tree walks everywhere — no recursion
- Designed the `walk_and_rebuild` infrastructure
- Pushed for `SortKey` as precomputed byte sequences rather than on-the-fly comparison
- Advocates for CSE before any code generation pass

---

### Dr. Rina Nakamura — Polynomial Algebra & Factoring

**Background:** Computational mathematician specializing in polynomial
arithmetic, factoring algorithms, and algebraic number theory. Deep knowledge
of Berlekamp-Zassenhaus, Hensel lifting, and multivariate polynomial GCD
algorithms. Previously contributed to open-source computer algebra systems.

**Specialty areas:** Dense univariate polynomial arithmetic over ℚ, Euclidean
GCD/LCM, `cancel()`/`together()` implementation, `factor()` via rational root
theorem, future Hensel/Zassenhaus lifting, Gröbner basis design (Buchberger+FGLM),
multivariate polynomial representation.

**Consulting trigger:** Any question about polynomial operations, factoring
strategy, or algebraic manipulation algorithms.

**Known positions:**
- Recommended Buchberger+FGLM over F5 for Gröbner bases (Decision #6)
- Advocates for correct-then-fast — get Buchberger right before attempting F5
- Pushed for polynomial LCM in `together()` with product fallback
- Wants `MultiPoly` type before attempting multivariate operations

---

### Kai Johannsen — Proc Macro Design & DSL Engineering

**Background:** Rust macro engineer specializing in procedural macro design,
Pratt parser construction, and domain-specific language embedding. Built
multiple proc macro crates for scientific computing. Deep experience with
`syn`, `quote`, and `proc-macro2`. Cares deeply about error messages in
macro expansion.

**Specialty areas:** `expr!` macro implementation (Pratt parser with precedence
climbing), `rule!` macro pattern language, `matrix!` and `eq!` macro design,
`__macro_support` module, the `self as symplex` extern crate trick for
path resolution inside the crate.

**Consulting trigger:** Any question about macro syntax design, how the Pratt
parser works, adding new functions to `expr!`, or debugging macro expansion errors.

**Known positions:**
- Designed the shared Pratt parser used by both `expr!` and `rule!`
- The `rule!` macro CANNOT be used inside `src/` (generates `::symplex::` paths)
- Advocates for compile-time error messages over runtime panics in macros
- Pushed for right-associative `^` in the parser

---

### Dr. Amara Osei — End-User Perspective & Applied Mathematics

**Background:** Applied mathematician and physicist who uses CAS tools daily
for research in control systems and robotics. Focuses on real-world workflows:
derive a Jacobian symbolically, generate C/Rust code, run it in a control loop.
Tests symplex against the problems she actually needs to solve. Not a Rust
expert — represents the "physicist who learned Rust" user persona.

**Specialty areas:** Symbolic Jacobian → code generation workflows, matrix
algebra for robotics, ODE solving for control systems, series expansion for
approximation, assumption propagation for physical quantities (positive mass,
real-valued temperature), Laplace transforms for transfer functions.

**Consulting trigger:** Any question about whether a feature serves real users,
what workflows matter most, or what the first-hour experience should feel like.

**Known positions:**
- Zero-testing (is this expression zero?) is the highest-priority correctness concern
- Matrix and vector calculus must work flawlessly — it's the primary use case
- `lambdify` → fast closure is the killer feature for numerical workflows
- Assumption propagation matters: if mass is positive, `sqrt(mass²)` should simplify to `mass`

---

### Viktor Petrov — Concurrency & Safety Engineering

**Background:** Concurrency specialist focused on lock protocol design, deadlock
prevention, and thread-safety verification. Experience with `parking_lot`,
`loom`, and formal verification of lock ordering. Paranoid about data races
by training. Believes if a deadlock is possible, it will happen in production.

**Specialty areas:** `ExprView` design (non-locking view type), `parking_lot::RwLock`
usage patterns, lock ordering in `ContextInner`, `Send + Sync` guarantees on `Ex`,
concurrent arena access patterns, GC safety under concurrent modification.

**Consulting trigger:** Any question about thread safety, lock design, or
whether a new API could introduce deadlock or data race risk.

**Known positions:**
- Designed `ExprView` to make deadlock structurally impossible at compile time (Decision #10)
- Single `RwLock` with structural ordering — never nest locks
- `Ex` must remain `Send + Sync` — any change that breaks this is rejected
- Wants `loom` testing for concurrent arena operations (not yet implemented)

---

### Dr. Fatima Al-Rashid — Adversarial Testing & Fuzzing

**Background:** Security-minded testing engineer specializing in fuzz testing,
adversarial input generation, and parser hardening. Experience with `cargo-fuzz`,
`libfuzzer`, AFL++, and grammar-based fuzzing. Focuses on finding inputs that
crash, hang, or produce silently wrong results. Believes every parser is broken
until proven otherwise.

**Specialty areas:** Parser fuzz targets, recursion depth protection, BigInt
overflow scenarios, adversarial expression construction (deeply nested, huge
coefficients, pathological canonical forms), grammar-aware fuzzing strategies.

**Consulting trigger:** Any question about parser robustness, input validation,
or whether an operation could hang/crash on adversarial input.

**Known positions:**
- Built the `fuzz/fuzz_targets/fuzz_parser.rs` target
- Pushed for recursion depth limit (256) in the parser
- Wants grammar-aware fuzzing (structured input generation) not just random bytes
- Advocates for `Arbitrary` derive on expression tree types for structure-aware fuzz

---

### Dr. Leo Eriksson — Type System Design & Formal Methods

**Background:** Programming language researcher specializing in type system
design, phantom types, and algebraic data type encoding. Background in
dependent types and refinement types. Thinks about "making invalid states
unrepresentable" as the primary design principle. Contributed to the phantom
sort system.

**Specialty areas:** `Expr<S: Sort>` phantom type design, `Numeric`/`Boolean`/`SetValued`
sort markers, cross-sort bridges (`Ex::gt() → BoolEx`), `Piecewise` type safety
(pairs instead of flat vec), `verify_canonical` invariant checking, compile-time
vs runtime sort enforcement.

**Consulting trigger:** Any question about type-level encoding, whether a new
feature needs a new sort, or how to prevent invalid expression construction
at compile time.

**Known positions:**
- Co-designed the three-sort phantom type system with Marcus Lindqvist
- Pushed for `SmallVec<[(ExprId, ExprId); 3]>` pairs in Piecewise (makes odd-length impossible)
- Acknowledges phantom safety is API-level only — arena internals are untyped
- Wants boolean-typed symbols eventually (currently all symbols are Numeric)

---

### Nadia Kowalski — Embedded Systems & Robotics Code Generation

**Background:** Embedded systems engineer with 12 years building firmware for
robotic manipulators, quadrotors, and autonomous vehicles. Deep experience with
`no_std` Rust on ARM Cortex-M and RISC-V targets, real-time control loops at
1kHz+, and the full pipeline from symbolic derivation to deployed numerical code.
Previously built internal code generation tooling that took MATLAB Symbolic Toolbox
output and cross-compiled it for bare-metal C targets. Switched to Rust for
memory safety guarantees in safety-critical control systems.

**Specialty areas:** `no_std` Rust crate design, `build.rs` code generation
pipelines, Denavit-Hartenberg kinematic modeling, forward/inverse kinematics for
serial manipulators, Jacobian computation and singularity analysis, Lagrangian
dynamics derivation (M/C/g matrices), real-time control loop architecture,
`#![no_std]` + `#![no_main]` firmware patterns, `libm` for transcendental
functions on embedded targets, fixed-point arithmetic trade-offs, `embedded-hal`
ecosystem integration.

**Consulting trigger:** Any question about how symbolic math output gets deployed
to embedded hardware, what the code generation pipeline should look like, what
constraints `no_std` imposes, or what math robotics engineers actually need from
a CAS.

**Known positions:**
- The CAS NEVER runs on the target — it generates code that runs on the target
- `build.rs` is the right delivery mechanism for most projects (not proc macros)
- Cross-entry CSE across an entire Jacobian matrix is critical — without it, a
  6-DOF Jacobian wastes 40-60% of cycles recomputing shared trig subexpressions
- Generated code must be `#[no_std]`-compatible with `libm` fallback for sin/cos/etc.
- DH parameter helpers are table stakes for any robotics-facing CAS
- Wants `Matrix::to_rust_fn()` that generates one function returning a flat array
- Fixed-point (`i32`/`i64`) code generation is eventual must-have for low-cost MCUs
  without FPU, but `f64` covers 90% of use cases (Cortex-M4F and above have FPU)
- Code generation should emit SIMD hints / `#[inline]` annotations where beneficial
- Concerned about code size — embedded flash is limited, large expression expansions
  can produce multi-KB functions that blow the instruction cache

---

### Renzo Almeida — Rust Build Systems, Cross-Compilation & Deployment

**Background:** Build systems and developer tooling engineer with deep expertise
in the Rust compilation pipeline, cargo internals, and cross-compilation to
exotic targets. Has shipped Rust to WASM (wasm-pack, wasm-bindgen, wasm-opt),
bare-metal ARM (thumbv7em-none-eabihf), RISC-V, and WASI. Built custom cargo
subcommands and xtask workflows for multi-target monorepos. Extensive experience
with `build.rs` code generation, conditional compilation (`cfg`), feature flag
design, and binary size optimization for constrained targets. Thinks about
"how does a library's design affect every downstream build scenario."

**Specialty areas:** `build.rs` code generation patterns (`OUT_DIR`, `include!`,
`env!`), cargo subcommands and `cargo-xtask` patterns, cross-compilation
(`--target`, `.cargo/config.toml`, target-specific dependencies), WASM deployment
(`wasm-pack`, `wasm-bindgen`, `wasm-opt`, browser + Node targets), `no_std` /
`no_alloc` crate design, feature flag architecture (additive features, avoiding
feature unification footguns), conditional dependencies (`[target.'cfg(...)'.dependencies]`),
linker scripts and memory layout for embedded, binary size profiling (`cargo-bloat`,
`twiggy`), LTO and codegen-unit tuning, proc macros as build-time code generators
vs runtime code generators, `include_str!` / `include_bytes!` for embedding
generated artifacts, CI/CD for multi-target matrices (GitHub Actions cross),
`cargo-dist` and `cargo-release` for publishing.

**Consulting trigger:** Any question about how a crate should be structured for
cross-compilation, how to design feature flags, how build.rs code generation
pipelines work, how to deploy to WASM or embedded targets, or how a library's
dependency choices affect downstream users on unusual targets.

**Known positions:**
- `build.rs` is the right mechanism for "run CAS at build time, emit code for target"
  — it's cargo-native, requires no external tooling, and the output lands in `OUT_DIR`
- Proc macros are WRONG for heavy code generation — they run in the compiler process,
  can't do I/O easily, and their errors are hard to debug. Use proc macros for syntax
  sugar (like `expr!`), use build.rs for code generation
- Feature flags must be strictly additive — `no_std` support should be a
  `default-features = false` opt-out, not a feature you opt into
- WASM is the sleeper use case: run symplex in the browser for interactive math
  notebooks, generate code client-side. Needs `wasm-bindgen` bindings eventually
- For a library like symplex that uses `parking_lot`, `Arc`, `BigInt` — a `no_std`
  mode for the core CAS is impractical. But a thin codegen-output crate that contains
  ONLY the generated numerical code can be `no_std` trivially
- Recommends a `symplex-codegen` companion crate (not part of symplex itself) that
  provides the build.rs integration: `symplex_codegen::generate_jacobian_fn(...)` etc.
- Binary size matters: for WASM, every KB counts. The generated code should be
  minimal — no format strings, no panics, no string literals. Consider `#[inline]`
  and `#[no_mangle]` attributes in generated output
- The ideal workflow: `cargo build` triggers build.rs → symplex derives equations →
  generated .rs file lands in OUT_DIR → firmware crate `include!`s it → cross-compiled
  to target. Zero manual steps

---

## How to Consult

| Question domain | Consult | Focus |
|----------------|---------|-------|
| Mathematical correctness, canonical forms, CAS algorithms | Dr. Elena Vasquez | Canonicalization rules, assumption inference, polynomial algebra |
| Rust API ergonomics, operator design, `expr!` macro UX | Marcus Lindqvist | Clone vs Copy, operator combos, phantom types, `#[must_use]` |
| Testing strategy, proptest, fuzzing, mutation testing | Dr. Priya Chandrasekaran | Test tiers, value-preservation properties, negative tests |
| Performance, allocation, cache, lock contention | James Okafor | SmallVec sizing, FxHashMap, parallel-array layout, Arc overhead |
| Module boundaries, crate structure, versioning | Dr. Sophie Marchand | pub(crate) defaults, single-crate policy, error design |
| IR design, display, bytecode, compiler patterns | Dr. Tomás Herrera | ExprNode sizing, iterative display, flat-temporary codegen |
| Polynomial algebra, factoring, Gröbner bases | Dr. Rina Nakamura | cancel/together prioritization, F4/F5 recommendations |
| Proc macro design, syn/quote patterns | Kai Johannsen | expr!/rule! design, Pratt parser, `__macro_support` |
| End-user perspective (physics, engineering workflows) | Dr. Amara Osei | Zero-testing priority, matrix/tensor embedding, assumption propagation |
| Concurrency, lock protocols, deadlock prevention | Viktor Petrov | ExprView design, parking_lot, lock ordering |
| Adversarial testing, fuzzing infrastructure | Dr. Fatima Al-Rashid | Parser fuzz target, bc-verified expectations |
| Type system design, phantom types, sort safety | Dr. Leo Eriksson | Phantom vs newtype vs trait, multi-sorted algebra |
| Embedded robotics, no_std codegen, DH parameters | Nadia Kowalski | build.rs pipelines, matrix codegen, libm, real-time constraints |
| Build systems, cross-compilation, WASM, cargo tooling | Renzo Almeida | build.rs, cargo subcommands, feature flags, target-specific deployment |

---

## Development Methodology

### AI-Assisted Development

symplex is developed using AI pair programming with Claude. Development sessions
involve the full expert panel (simulated domain specialists) in real-time discussions
to make architectural decisions, then implementation via parallel agent dispatch.

### Session Structure

A typical development session follows this pattern:

1. **Team discussion** — The expert panel debates the design. Each member brings
   their domain expertise. Disagreements are resolved through evidence and expert
   consultation. No premature consensus.

2. **Expert consultation** — For hard decisions, we formulate 3 questions for
   external domain experts. Questions include full project context so the expert
   doesn't need to know our codebase. Answers are incorporated into the design.

3. **Wave planning** — Work is organized into waves with:
   - Clear deliverables per wave
   - File ownership (no two agents touch the same file)
   - Dependency ordering (what must complete before what)
   - Commit gates (cargo test must pass after each wave)

4. **Parallel implementation** — Multiple agents are dispatched simultaneously,
   each with exclusive file ownership. This prevents merge conflicts and enables
   high throughput. A typical batch dispatches 3-5 agents in parallel.

5. **Verification** — After each batch: `cargo test --lib` (fast), then
   `cargo test --all-targets` (comprehensive). Commit only when green.

### Parallel Agent Protocol

When dispatching parallel agents for implementation:

1. **Mutually exclusive file ownership.** Each agent is assigned specific files.
   No two agents may edit the same file in the same batch. The dispatch message
   explicitly states: "ONLY touch: file1.rs, file2.rs. Do NOT touch file3.rs."

2. **Phase barriers.** When Agent A's output is needed by Agent B, they run in
   sequential batches, not parallel. Commit Agent A's work before dispatching B.

3. **Context for agents.** Each agent message includes:
   - Which files they own exclusively
   - What the existing code looks like (read first)
   - Project conventions (testing, naming, no names in source)
   - Expected deliverables and test count

4. **Verify after each batch.** Run `cargo test` + `cargo clippy` after every
   batch before proceeding. Never trust an agent's claim without verification.

5. **Don't weaken tests to make them pass.** If a test fails, fix the root cause.

### Bug Exposure Protocol

When tests expose bugs during implementation:

- **Quick fix (< 30 min):** Fix in the same wave.
- **Real bug (30 min – 2 hours):** Fix in the same wave, document in CHANGELOG.
- **Deep bug (> 2 hours):** Mark test `#[ignore]` with `// BUG: description`.
  Open a tracking issue. Move to the next test.
- **Circuit breaker:** If > 5 deep bugs found in one wave, pause and stabilize.

### Expert Consultation Model

For hard architectural decisions, we formulate questions for domain experts.
Each question includes full context (the expert doesn't know our project):
- What we're building and why
- What we've already decided and why
- The specific design question with options
- What tradeoffs we're trying to evaluate

We ask up to 3 questions per consultation round. Expert answers are discussed
by the full panel before being incorporated into the design.

### Naming Convention

No team member names, author names, or reviewer names appear in source code,
tests, comments, doc comments, CHANGELOG.md, or README.md. Team member names
appear ONLY in this file (TEAM.md). Use `git blame` for code attribution.

---

## Key Decisions Log

| # | Decision | Alternatives Rejected | Rationale |
|---|----------|----------------------|-----------|
| 1 | **Phantom types for sorts** (`Expr<S: Sort>`) | Newtypes (duplicate methods), traits (orphan rules) | One struct, generic common methods, zero-cost, scales to N sorts |
| 2 | **Number×Add distribution in canon_mul** | Only distribute -1; skip distribution | Required for `a - a = 0`; matches SymPy's 20-year-proven approach |
| 3 | **Stirling series for arbitrary-precision Gamma** | Lanczos (doesn't scale beyond f64), Spouge (50% extra precision) | 0.323p terms (fewest), Bernoulli coefficients cacheable; ref: Johansson 2021 |
| 4 | **Stripper-collector for AC pattern matching** | Full greedy-backtracking (SymPy), discrimination nets (MatchPy) | O(k·n) for internal linear rules; full AC matching is NP-complete and unnecessary for author-controlled rules |
| 5 | **BitVec liveness for arena GC** | Reference counting, epoch-based reclamation | Same walk as compact() minus the copy; no per-construction overhead |
| 6 | **Buchberger+FGLM for Gröbner bases** | F5 (complex, correctness risk), direct lex (orders of magnitude slower) | Correct debuggable baseline; Gebauer-Möller eliminates ~90% of S-pairs |
| 7 | **eval-before-evalf pipeline** | Direct numerical evaluation | CAS principle: symbolic first, numerical last; catches `sin(π)=0` exactly |
| 8 | **Ex is Clone, not Copy** (Arc-based) | Global arena + Copy index (!Send), lifetime parameter (ergonomic nightmare) | Thread-safe from day one; `expr!` macro mitigates `&` noise |
| 9 | **No Float node type** | Mixed exact/inexact nodes | Prevents precision confusion; all symbolic math uses exact `Ratio<BigInt>` |
| 10 | **ExprView for replace() closures** | Pass `&Ex` (deadlock risk), snapshot-then-transform | Non-locking view type makes deadlock structurally impossible at compile time |
| 11 | **Sturm sequences for real root counting** (not solve-and-filter) | Numerical solve then filter | Exact, avoids Wilkinson's polynomial problem, operates entirely in ℚ, O(n²) in degree |
| 12 | **Dependency parameter for diff** (not arena/context storage) | Store dependency info in arena or context | Avoids arena mutation, no hash-consing complications, natural threading through explicit-stack walk |
| 13 | **Buchberger over F5B for v1** | F5B (faster but less robust) | Simpler, well-understood, SymPy defaults to it |
| 14 | **MultiPoly\<O\> generic ordering** | Runtime ordering, unsorted dict | O(log n) leading term via BTreeMap, zero-cost phantom type |
| 15 | **Sin/cos ring for IK** (not Weierstrass) | Weierstrass (fewer vars, higher degree) | More robust (no θ=π singularity), cleaner interaction with elimination |
| 16 | **Dense matrices for FGLM** | Sparse/Krylov | D ≤ 1024 for IK; dense is simpler and fast enough |
| 17 | **Delete old API names** (not deprecate) | Keep as #[deprecated] | Pre-1.0, clean break. One name per operation. |
| 18 | **Display: `^` not `**`** | Python-style `**` | Math notation, not programming notation |
| 19 | **build.rs for codegen** (not proc macro) | Heavy proc macro | sqlx acknowledged proc-macro codegen as architecturally flawed |