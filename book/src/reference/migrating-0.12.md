# Migrating from 0.11 to 0.12

0.12 redesigns the *representation* of distributions in `symplex::stats`;
the query API on `RandomVariable` is unchanged.  Every break has a one-line
fix.

## `Distribution` is a struct, not an enum

`Distribution::Continuous(ContinuousFamily::Normal { mean, std })` and
`Distribution::Discrete(DiscreteFamily::Binomial { n, p })` are gone.  A
`Distribution` is an opaque handle to a `Family` (a trait); the families
are structs.

| 0.11 | 0.12 |
|------|------|
| `match d { Distribution::Continuous(ContinuousFamily::Normal { mean, std }) => … }` | `if let Some(n) = d.downcast_ref::<Normal>() { n.mean, n.std }` |
| `Distribution::Continuous(f) => f.entropy(&ctx)` | `d.family().entropy()` (closed form) or `d.entropy()` (always an answer) |
| `d.mean(&ctx)`, `d.variance(&ctx)`, `d.raw_moment(n, &ctx)` → `Option<Ex>` | `d.family().mean()`, `.variance()`, `.raw_moment(n)` → `Option<Ex>`; `d.mean()`, `d.variance()`, `d.moment(n)` → `Ex` (closed form or generic route) |
| `d.cdf(&x)`, `d.mgf(&t)`, `d.quantile(&p)` → `Option<Ex>` | `d.family().cdf(&x)` … → `Option<Ex>` (closed form on the support); `d.cdf(&x)`, `d.mgf(&t)` → `Ex` |
| `d.is_continuous()` | unchanged (`d.kind() == Kind::Continuous`) |

## `Support` is a typed region

`Support::Continuous { lo: Option<Ex>, hi: Option<Ex> }`,
`Support::Discrete { … }` and `Support::Finite(Vec<Ex>)` are replaced by a
struct with a `Kind` and `Piece`s.

| 0.11 | 0.12 |
|------|------|
| `Support::Continuous { lo: Some(a), hi: Some(b) }` | `Support::interval(a, b)`; unbounded ends are `ctx.neg_infinity()` / `ctx.infinity()` |
| `Support::Discrete { lo: Some(a), hi: None }` | `Support::integers(&ctx, Some(a), None)` |
| `Support::Finite(values)` | `Support::points(values)` |
| `match support { Support::Continuous { lo, hi } => … }` | `let iv = support.as_interval()?;` then `iv.lower`, `iv.upper`, `iv.kind` (an `Interval<Ex>`) |
| `matches!(s, Support::Discrete { .. })` | `s.kind() == Kind::Discrete` |

## `Distribution::finite` takes the context

`Distribution::try_finite(table)` → `Distribution::try_finite(&ctx, table)`
(likewise `finite`): an empty table has no parameter to take a context from.

## `cdf` is clamped to the support

`RandomVariable::cdf(&x)` / `Distribution::cdf(&x)` return the whole-line
distribution function — `0` below the support, the closed form on it, `1`
above it — as SymPy's `cdf(X)(x)` does (`Uniform(0, 1).cdf(3)` is `1`, not
`3`; `Geometric(p).cdf(k)` is a `Piecewise` that is `0` for `k < 1`).  A
test that pinned the unclamped formula should compare
`d.family().cdf(&x)` instead.

## Events that used to be `NotImplemented` now have answers

`P(X² < 1)`, `P(X < −1 ∨ X > 1)`, `E[X | X² > 1]` go through the
inequality solver when the bounds are numeric.  `P(X = 3 ∧ X > 5)` is `0`
(it was `P(X = 3)`), `P(X > 1 ∧ X ≥ 2)` is `P(X ≥ 2)` (the strict bound no
longer wins), and `P(X = ½)` for an integer-valued variable is `0`.
