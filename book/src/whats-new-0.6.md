# What's New in 0.6

symplex 0.6.0 is an additive release with one theme: **sums of squares**. The [CHANGELOG](https://github.com/cgorski/symplex/blob/main/CHANGELOG.md) has the details.

## `prove_sos`

The certificates of 0.3–0.5 all multiply non-negative *hypotheses* (box bounds, half-line shifts, polyhedron facets). The one class they could not reach is a polynomial that is non-negative on all of ℝⁿ with an interior zero that is not a square factor — `(x − 1)² + (y − 1)²`, or the AM–GM form `x⁴ + y⁴ + z⁴ + 1 − 4xyz`. `certificates::prove_sos(goal, &vars, &SosOpts::default())` proves those by an exact **sum-of-squares decomposition** `g = Σ dₖ·pₖ²` with rational `dₖ > 0` and rational-coefficient `pₖ`, re-verified by expanding it.

Under the hood this is the Peyrl–Parrilo pipeline made exact and self-contained: the Gram semidefinite program `g = mᵀQm, Q ⪰ 0` is solved by a small dense primal–dual interior-point method written for the crate (HKM direction, Mehrotra predictor–corrector, exact steps to the cone boundary; the Gram matrices here have a few dozen rows, so no external solver is needed), the solution is rounded and projected back onto the coefficient constraints exactly, and positive semidefiniteness is decided by the rational `L·D·Lᵀ` of `QMatrix::ldl_psd` — which *is* the decomposition.

Goals with real zeros only have singular Gram matrices, which rounding cannot hit; the search then performs **facial reduction**, reading the kernel off the numerical solution and making it exact — directly when the kernel is a rational subspace, otherwise through its integer relations (LLL on the kernel lattice, after Newton-refining the zeros of the goal to double precision) — before restricting to that face and solving again. Sums of two or three random squares with irrational common zeros come back as exactly those squares (119 of 120 random cases through degree 6).

`Refuted { point, value }` carries an exact point where the goal is negative; a goal that is non-negative but not a sum of squares (Motzkin's polynomial) is `Unknown`, never `Proved`.

## Lean export

`SosCertificate::to_lean` emits

```lean
theorem two_squares (x y : ℝ) : 0 ≤ x ^ 2 + y ^ 2 - 2 * x - 2 * y + 2 := by
  have h : x ^ 2 + y ^ 2 - 2 * x - 2 * y + 2 = (2 : ℝ) * (-(x / 2) - y / 2 + 1) ^ 2 + (1 / 2 : ℝ) *
    (-x + y) ^ 2 := by ring
  rw [h]
  positivity
```

— `ring` checks the identity, `positivity` closes the sum of non-negative terms; two deterministic steps. Eight shapes were compiled against Mathlib with the long-line linter on and the emitted text is pinned to that file. `lean_hints` returns the `sq_nonneg (pₖ)` terms for an `nlinarith` skeleton of your own, and certificates round-trip through JSON with re-verification like the other kinds.

→ [Cookbook: sums of squares](./cookbook/polynomial-certificates.md#sums-of-squares-interior-zeros-without-a-square-factor)
