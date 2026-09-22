# Migrating from 0.6 to 0.7

Every breaking change in 0.7.0, with the one-line fix. Code that only pattern-matches `Proved(c)` and `Refuted { point, value, .. }` on `PolyhedronOutcome` / `SosOutcome`, builds options with `::default()` and the `with_*` builders, and calls the inherent certificate methods compiles unchanged.

## Outcomes

The four outcome enums are aliases of one generic `certificates::Outcome<C, U>`.

| 0.6 | 0.7 |
|---|---|
| `BoxOutcome::Refuted { point, value }` with `point: Vec<Q>` | `Refuted { point, value, .. }` with `point: Vec<(Ex, Q)>` — the value of `x` is `point[i].1` |
| `HalfLineOutcome::Refuted { point, value }` with `point: Q` | `Refuted { point, value, .. }` with `point: Vec<(Ex, Q)>` of length 1 — `point[0].1` |
| `PolyhedronOutcome::Refuted { point, value, param_value }` | unchanged fields; the variant is `#[non_exhaustive]`, so write `..` |
| `SosOutcome::Refuted { point, value }` | `Refuted { point, value, .. }` (`param_value` is `None`) |
| `BoxOutcome::Unknown { farkas, degree }` | `Unknown(u)` with `u: BoxUnknown { farkas, degree, .. }` |
| `HalfLineOutcome::Unknown { max_polya_power }` | `Unknown(u)` with `u: HalfLineUnknown { max_polya_power, .. }` |
| `PolyhedronOutcome::Unknown { degree, lambda_degree, pairwise }` | `Unknown(u)` with `u: PolyhedronUnknown { … , .. }` |
| `SosOutcome::Unknown { reason }` | `Unknown(u)` with `u: SosUnknown { reason, .. }`; `u.to_string()` is the reason |

```rust,ignore
// ignore: the "before" lines show the removed API and cannot compile against the current crate
// 0.6
match prove_nonnegative_on_halfline(&g, &j, &ctx.int(3), Ray::AtLeast, 10)? {
    HalfLineOutcome::Refuted { point, value } => println!("false at j = {point}: {value}"),
    HalfLineOutcome::Unknown { max_polya_power } => println!("gave up at N = {max_polya_power}"),
    HalfLineOutcome::Proved(c) => …,
}
// 0.7
match prove_nonnegative_on_halfline(&g, &j, &ctx.int(3), Ray::AtLeast, 10)? {
    HalfLineOutcome::Refuted { point, value, .. } => println!("false at j = {}: {value}", point[0].1),
    HalfLineOutcome::Unknown(u) => println!("gave up: {u}"),
    HalfLineOutcome::Proved(c) => …,
}
```

The helpers `is_proved()` and `certificate()` exist on every outcome as before; `is_refuted()`, `is_unknown()`, `into_certificate()`, `refutation()`, `unknown()` and `map_certificate()` are new.

## The box certificate type

| 0.6 | 0.7 |
|---|---|
| `certificates::Certificate` (struct) | `certificates::BoxCertificate` |
| `certificates::CertificateData` | `certificates::BoxCertificateData` |
| — | `certificates::Certificate` is now the **trait** implemented by all five certificate types |

A `use symplex::certificates::Certificate;` that meant the struct must become `BoxCertificate`; one that is only used for method calls can be deleted (the inherent methods do not need the trait in scope).

## Option structs

`LeanOpts`, `PolyhedronOpts` and `SosOpts` are `#[non_exhaustive]`: a struct literal, including one ending in `..Default::default()`, no longer compiles outside the crate.

```rust,ignore
// ignore: the "before" lines show the removed API and cannot compile against the current crate
// 0.6
let opts = PolyhedronOpts { max_lambda_degree: 0, ..Default::default() };
let lean = LeanOpts { real_type: "ℚ".into(), ..Default::default() };
// 0.7
let opts = PolyhedronOpts::default().with_max_lambda_degree(0);
let lean = LeanOpts::default().with_real_type("ℚ");
// or
let mut opts = PolyhedronOpts::default();
opts.max_lambda_degree = 0;
```

`PolyhedronOpts::single(degree, lambda_degree)` is unchanged. New builders: `PolyhedronOpts::{with_max_degree, with_max_lambda_degree, with_pairwise, with_staged}`, `SosOpts::{with_max_basis, with_max_iterations, with_rounding_digits, with_max_facial_reductions}`.

## Functions that could panic on their arguments now return `Result`

| 0.6 | 0.7 |
|---|---|
| `StateSpace::controllability_matrix() -> Matrix` | `-> Result<Matrix, SymplexError>` |
| `StateSpace::observability_matrix() -> Matrix` | `-> Result<Matrix, SymplexError>` |
| `StateSpace::discretize_zoh(dt, order) -> StateSpace` | `-> Result<StateSpace, SymplexError>` |
| `StateSpace::riccati_residual(p, q, r) -> Option<Matrix>` | `-> Result<Matrix, SymplexError>` (singular `R` and shape mismatches are errors) |
| `StateSpace::ackermann(poles) -> Option<Matrix>` | `-> Result<Matrix, SymplexError>` (`InvalidArgument` for multi-input / wrong pole count, `ComputationFailed` if uncontrollable) |
| `robotics::homogeneous(rotation, position) -> Matrix` | `-> Result<Matrix, SymplexError>` |
| `dynamics::{total_time_derivative, euler_lagrange, mass_matrix, christoffel_symbols, coriolis_matrix, manipulator_equation}` | each returns `Result<_, SymplexError>` |

Behavioural, not signature, changes: `StateSpace::char_poly` and `matrix_decomp::wronskian` return NaN instead of panicking on an ill-shaped model / empty list (use `try_char_poly` / `try_wronskian` for the error); `is_controllable` / `is_observable` return `false` for an ill-shaped model; `ode::solve_ode_system{,_nonhomogeneous}` return `None` where they previously could panic.

## Nothing else

`Polytope`, `ParametricPolytope`, `MultiPoly`, `PolyhedronProver`, the Lean emitters and every certificate's inherent methods are unchanged (and faster); the pinned Mathlib-compiled fixtures are byte-identical to 0.6.1.
