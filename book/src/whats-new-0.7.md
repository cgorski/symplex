# What's New in 0.7

symplex 0.7.0 is a **breaking** release whose theme is the shape of the API rather than new mathematics: one result type for every certificate search, a `Certificate` trait, option structs that can grow without breaking anyone, a library that is verified not to panic, and a much faster exact polytope core. [Migrating from 0.6 to 0.7](./reference/migrating-0.7.md) lists every change with its one-line fix; the [CHANGELOG](https://github.com/cgorski/symplex/blob/main/CHANGELOG.md) has the details.

## One `Outcome` for every prover

The four provers used to return four enums with the same three arms and slightly different payloads (`Refuted { point: Vec<Q> }` here, `Refuted { point: Q }` there, `Unknown { farkas, degree }` versus `Unknown { reason }`). They now all return

```rust
# use symplex::prelude::*;
# use symplex::linprog::Q;
pub enum Outcome<C, U> {
    Proved(C),                                                    // a re-verified certificate
    #[non_exhaustive]
    Refuted { point: Vec<(Ex, Q)>, value: Q, param_value: Option<Q> },
    Unknown(U),                                                   // what the search tried
}
```

under the familiar aliases `BoxOutcome`, `HalfLineOutcome`, `PolyhedronOutcome`, `SosOutcome` — so `PolyhedronOutcome::Proved(c)` still reads as before, a counterexample is always `(variable, value)` pairs, and `is_proved` / `certificate` / `into_certificate` / `refutation` / `unknown` / `map_certificate` are one implementation. The `Unknown` payloads are small `#[non_exhaustive]` structs (`BoxUnknown`, `HalfLineUnknown`, `PolyhedronUnknown`, `SosUnknown`) that implement `Display`, so `println!("{u}")` says what was tried.

## The `Certificate` trait

`BoxCertificate` (the Handelman certificate, until now confusingly named `Certificate`), `HalfLineCertificate`, `RealLineCertificate`, `PolyhedronCertificate` and `SosCertificate` implement `certificates::Certificate` — `goal`, `verify`, `to_lean` / `to_lean_with`, `to_json` / `from_json` (the latter re-verifying). Generic code — a prover that falls back to another, a report over a mixed list — can now treat them alike; the inherent methods are unchanged, so nothing needs the trait in scope to keep working. `RealLineCertificate` gained the JSON round trip and `Display` it was missing.

## Options that can grow

`LeanOpts`, `PolyhedronOpts` and `SosOpts` are `#[non_exhaustive]`: construct them with `::default()` and the `with_*` builders (all three now have one per field), or assign fields on a `mut` default. Adding an option is no longer a breaking change — the 0.4 → 0.5 `LeanOpts` episode does not repeat.

## Verified not to panic

`CONTRIBUTING.md` now spells out a practical no-panic policy (validate at the boundary, `Result` for failure, `Option` for absence, `debug_assert!` for invariants, `std`-style `try_` siblings for indexing) and a **ratchet test** enforces it over `src/`: 108 `unwrap`/`expect`/`unreachable!` sites in library code were removed, one of them a reachable panic (`wronskian` on an expression-budget overflow), and the allowlist is down to the two documented logic errors, the arena's `u32` index conversion and a compile-time assertion macro. The methods that could fail on user-supplied shapes now say so in their types (`StateSpace::{controllability_matrix, observability_matrix, discretize_zoh, riccati_residual, ackermann}`, `robotics::homogeneous`, the `dynamics` functions); `char_poly` and `wronskian` return NaN on an ill-shaped model and have `try_` siblings.

## The polytope core, 3× on a real workload

Profiling a downstream decision-tree generator showed half its time in `Polytope::vertices` — `Ratio<BigInt>` containment tests, a gcd per multiplication — and most of the rest in `volume` re-enumerating vertices at every level of its recursion. `vertices` now runs in integer arithmetic throughout (half-spaces scaled once, distinct hyperplanes only, the fraction-free kernel producing each point as `X / D`, containment as the sign of `a·X + b·D`), is cached on the polytope, and `volume` hands each facet its own vertices. `is_full_dimensional()` (one LP) and `interior_point()` replace `volume() > 0`; `HalfSpace::normalized()` is the key that identifies a cut with its flip. The generator went from 103 s to 34 s with byte-identical output, and the users' own `MultiPoly`-based tooling is served by `PolyhedronProver::prove_poly`, `PolyhedronCertificate::used_hyps`, `MultiPoly::{eval_var, affine_form, as_constant, to_ex}` and `Q` / `MultiPoly` in the prelude (shipped in 0.6.1).

## Lean wrapping and bullets

`wrap_lean` measures its continuation indent from the tactic column past `· ` bullets, so a wrapped `· have … := by tac` no longer swallows the next tactic into its `by` block (both shapes compiled against Mathlib).
