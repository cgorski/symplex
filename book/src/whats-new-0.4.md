# What's New in 0.4

symplex 0.4.0 is a minor release with two mechanical breaking changes. The [CHANGELOG](https://github.com/cgorski/symplex/blob/main/CHANGELOG.md) has the complete list, and [Migrating from 0.3 to 0.4](./reference/migrating-0.4.md) shows the two source changes an upgrading program may need.

The theme is **certificates on parametric polyhedra** — the question a decision procedure asks thousands of times when its cells move with a parameter — together with the exact geometry of the cells themselves. It builds directly on the 0.3.5 exact matrix core and integer-pivoting simplex: each certificate is a few small exact LPs, and each takes milliseconds.

## Certificates on a parametric polyhedron

`certificates::prove_nonnegative_on_polyhedron(goal, hyps, Some((&j, &j0)), &PolyhedronOpts::default())` proves `g ≥ 0` on `{x : hₖ(j, x) ≥ 0}` for every real `j ≥ j₀` by the identity

```text
λ(j)·g = Σ μ · jᵃ (j − j₀)ᵇ · hₖ + Σ μ · jᵃ (j − j₀)ᵇ + μ₀  (+ Σ μ · hₖ hₗ),   λ(j) = 1 + Σ νₐ jᵃ,  μ, ν ≥ 0,
```

whose **polynomial multiplier `λ` on the goal** is what makes `j`-dependent facets certifiable at all. `prove_polyhedron_empty` is the same identity with the goal `−1`, proving a cell empty for every `j`. The search is staged from the smallest basis upwards (`λ = 1` and degree-1 multipliers first; pairwise products of hypotheses last), returns `Proved` / `Refuted { point, value }` (an exact point of the set) / `Unknown`, and re-verifies every certificate with polynomial arithmetic. Without a parameter it is a plain Farkas / pairwise certificate on a fixed polyhedron.

The Lean export writes the proof a person would: `have h0K := mul_nonneg hK0 h0` per product, `linarith only […]` over exactly those facts, `nonneg_of_mul_nonneg_right` when `λ ≠ 1`, `False` for emptiness. `lean_steps` returns the same lines with *your* hypothesis names for an existing proof skeleton. Fifteen distinct shapes were compiled against Mathlib with the long-line linter on, and the emitted text is pinned to that compiled file.

→ [Cookbook: parametric polyhedra](./cookbook/polynomial-certificates.md#parametric-polyhedra-a-multiplier-on-the-goal)

## Exact polytopes

`symplex::polytope::Polytope` is a convex polyhedron in ℚⁿ from half-spaces: exact `vertices` (via `QMatrix::solve`), `volume` (dimension ≤ 3 in 0.4, any dimension since 0.5), `contains`, `is_empty` / `any_point` / `bounding_box` / `is_bounded` (exact LP), `irredundant`, `split` by a hyperplane, and `from_exprs` / `to_exprs` to move between affine `Ex` hypotheses and half-space data — so a cell can be measured, cut and handed to the certificate search.

→ [Exact Linear Programming: polytopes](./guide/exact-lp.md#polytopes-from-half-spaces)

## Certificates as data

`Certificate`, `HalfLineCertificate` and `PolyhedronCertificate` serialise to plain data (`to_data` / `to_json`: expression trees plus `"p/q"` rationals) and back (`from_data(&ctx, …)` / `from_json`). Reconstruction **re-verifies** the identity exactly and rejects anything that does not hold, so a certificate produced by one process can be accepted by another without trusting the producer — the same guarantee the Lean export gives, one step earlier.

## Smaller additions

- `Poly::try_new` — `Poly::new` with the reason for failure (which generator sits inside a function, under a negative power, under a fractional or symbolic power, or in an exponent).
- `Poly::terms_iter()` (borrowed, no allocation) and `Poly::coeffs_rational()`.
- `LeanOpts::prefer_subtraction` (`(1 / 2 : ℝ) - r` instead of `-r + (1 / 2 : ℝ)`) and the `with_*` builders.
- `HalfLineCertificate::lean_hints(hk, &opts)` — the hint list alone, for a proof skeleton.
- `linsolve` documents that an over-determined but consistent system is `Unique`.

## Breaking changes

Two, both mechanical: `Ex::roots_count_real` is gone (use `count_real_roots_in`), and `LeanOpts` struct literals need `..Default::default()`. See [Migrating from 0.3 to 0.4](./reference/migrating-0.4.md).
