# What's New in 0.12

**Statistics, redesigned to compose.**  A distribution is now a
[`Family`](guide/statistics.md#continuous-families) — a struct with its
support, density and closed forms — behind an opaque `Distribution`
handle that owns the generic machinery.  That is what makes the new
constructions one-liners:

- `x.given(&event)` — conditioning (`Truncated`): `E[N | N > 0] = √(2/π)`,
  `E[B | B ≥ 2] = 325/131`.
- `x.transform("Y", &g)` — `aX + b` with every closed form transported
  (`Affine`), strictly monotone maps and `X²`/`|X|` by the
  change-of-variables formula (`Transformed`), finite ranges mapped and
  merged.
- `Distribution::mixture(&[(w, F), …])`.
- Your own families: implement `Family`, wrap with
  `Distribution::from_family`.

**Events through the set machinery.**  With numeric bounds any boolean
combination of relations in the variable is accepted — `P(N² < 1)`,
`P(N < −1 ∨ N > 1)`, `E[X | X² > 1]` — and several 0.11 answers that were
wrong are fixed (`P(X = 3 ∧ X > 5)`, `P(X > 1 ∧ X ≥ 2)`, `P(X = ½)` for an
integer variable, `Uniform(0,1).cdf(3)`).

**New closed forms.**  `betainc` / `betainc_regularized` (SymPy's
4-argument form) give `Beta`, `StudentT` and the new `FDistribution`
their CDFs; `erfinv` compiles, so `Normal`/`LogNormal` (and everything
built on them) sample; `Σ C(k+c, k) xᵏ` and the binomial theorem with
symbolic `n` and `p` close, so `NegativeBinomial` needs no polynomial
workaround.

See [Migrating from 0.11 to 0.12](reference/migrating-0.12.md) for the
one-line fixes to code that matched on the old enums.
