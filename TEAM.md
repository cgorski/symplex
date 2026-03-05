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

- ❌ `// Reviewed by Dr. X`
- ❌ `//! Author: Prof. Y`
- ✅ `//! Tests for denominator rationalization and limit edge cases.`

The ONLY file where panel member names may appear is this file (TEAM.md).
Use `git blame` for code attribution.

---

## How to Consult

| Question domain | Consult | Focus |
|----------------|---------|-------|
| Rust API ergonomics, operator design, `expr!` macro UX | Rust API specialist | Clone vs Copy, operator combos, phantom types, `#[must_use]` |
| Mathematical correctness, canonical forms, CAS algorithms | CAS specialist | Canonicalization rules, assumption inference, polynomial algebra |
| Performance, allocation, cache, lock contention | Performance specialist | SmallVec sizing, FxHashMap, parallel-array layout, Arc overhead |
| Testing strategy, proptest, fuzzing, mutation testing | Testing specialist | Test tiers, value-preservation properties, negative tests |
| Module boundaries, crate structure, versioning | Architecture specialist | pub(crate) defaults, single-crate policy, error design |
| End-user perspective (physics, engineering workflows) | End-user specialist | Zero-testing priority, matrix/tensor embedding, assumption propagation |
| IR design, display, bytecode, compiler patterns | Compiler specialist | ExprNode sizing, iterative display, flat-temporary codegen |
| Polynomial algebra, factoring, Gröbner bases | Math software specialist | cancel/together prioritization, F4/F5 recommendations |
| Proc macro design, syn/quote patterns | Macro specialist | expr!/rule! design, Pratt parser, `__macro_support` |
| Concurrency, lock protocols, deadlock prevention | Concurrency specialist | ExprView design, parking_lot, lock ordering |
| Adversarial testing, fuzzing infrastructure | Fuzzing specialist | Parser fuzz target, bc-verified expectations |
| Type system design, phantom types, sort safety | Type system specialist | Phantom vs newtype vs trait, multi-sorted algebra |

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