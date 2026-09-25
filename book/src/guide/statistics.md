# Probability and Statistics

New in 0.11. `symplex::stats` is the counterpart of SymPy's `sympy.stats`: a **random variable** is a symbol together with a **distribution**, and the usual queries — mean, variance, moments, probabilities of events, density, CDF, moment generating function, quantile — are computed *exactly*, as expressions, wherever the distribution's parameters are exact.

```rust
use symplex::prelude::*;
use symplex::stats::{Distribution, RandomVariable};

fn main() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let x = RandomVariable::new(&ctx, "X", Distribution::normal(ctx.int(0), ctx.int(1)));
    println!("{}", x.mean());                                              // 0
    println!("{}", x.variance());                                          // 1
    println!("{}", x.expectation(&(x.symbol().powi(2) + 3 * x.symbol()))); // 1   E[X² + 3X]

    let y = RandomVariable::new(&ctx, "Y", Distribution::binomial(ctx.int(5), ctx.rational(1, 3)));
    println!("{}", y.mean());                                              // 5/3
    println!("{}", y.probability(&y.symbol().gt(&ctx.int(2)))?);           // 17/81
    Ok(())
}
```

## Design

* Every distribution family knows its **support**, its **density** (or probability mass function) as an expression in a free variable, and closed forms for whatever moments it has. The generic machinery — `RandomVariable::expectation`, `probability`, `cdf`, `mgf` — falls back to the crate's exact `integrate_definite` / `summation` over the support, so `E[g(X)]` works for any `g` the integrator can handle. For a polynomial `g` the closed-form raw moments are used directly (exact and cheap).
* **Parameters are expressions.** Rational parameters give exact rational answers; symbolic parameters give symbolic answers (`E[X] = μ`, `Var[Gamma(k, θ)] = kθ²`). Parameter *validity* (`σ > 0`, `0 ≤ p ≤ 1`, `a < b`) is checked for numeric parameters by the `try_` constructors (`Distribution::try_normal(…) -> Result`); the unchecked constructors (`Distribution::normal(…)`) accept anything, and for symbolic parameters validity is the caller's promise.
* Nothing is numerical by default. `RandomVariable::sample` is the only randomness in the symbolic layer; the data modules' bootstrap/permutation tests, `MarkovChain::sample_path` and `MultivariateNormal` sampling take an explicit seeded `Rng`. `stats::Rng` is a seeded, reproducible SplitMix64 so Monte-Carlo sanity checks are deterministic.
* A result that does not exist is never a number. `Cauchy` has no mean: `x.mean()` returns the divergent integral **unevaluated** (`has_unevaluated()` is `true`), and `try_integrate_definite` on `x·f(x)` says `Err(Divergent)`.

| Query | SymPy | Returns |
|-------|-------|---------|
| `mean()`, `variance()`, `std()` | `E(X)`, `variance(X)`, `std(X)` | `Ex` — closed form, or the integral/sum |
| `moment(n)`, `central_moment(n)` | `moment(X, n)`, `cmoment(X, n)` | `Ex` |
| `skewness()`, `kurtosis()` | `skewness(X)`, `kurtosis(X)` (not excess) | `Ex` |
| `expectation(&g)` | `E(g)` | `Ex` — may contain an unevaluated `Integral`/`Sum` |
| `probability(&event)` | `P(cond)` | `Result<Ex>` — `NotImplemented` for events that are not relations/conjunctions in `X` |
| `density(&x)`, `cdf(&x)` | `density(X)(x)`, `cdf(X)(x)` | `Ex` |
| `mgf(&t)`, `characteristic_function(&t)` | `moment_generating_function(X)(t)`, `characteristic_function(X)(t)` | `Ex` |
| `quantile(&p)`, `median()` | `quantile(X)(p)`, `median(X)` | `Option<Ex>` — `None` when there is no closed inverse CDF |
| `sample(n, &mut rng)` | `sample(X, size=n)` | `Result<Vec<f64>>` — exact-in-distribution sampling (inverse transform, cumulative sums, or the family's own algorithm) |

Events are `BoolEx` conditions in the variable's symbol: relations `X < a`, `X ≤ a`, `X > a`, `X ≥ a`, `X = a` with any (also symbolic) bound and their conjunctions, and — with numeric bounds — any boolean combination of relations in `X` (`X² < 1`, `|X| > 2`, `X < −1 ∨ X > 1`), which the crate's inequality solver turns into a set. The event's region is clipped to the support and measured through the closed-form CDF when the family has one, else by exact integration / summation, so `P(X > 1)` for `Exponential(3)` is `exp(-3)`, `P(0 < U < 1/4)` for `Uniform(0, 1)` is `1/4`, and `P(N² < 1)` for a standard normal is `erf(√2/2)`.

`cdf(&x)` is the whole-line distribution function as SymPy prints it: a `Piecewise` that is `0` below the support and `1` above it (`Uniform(0,1).cdf(3) = 1`); the family's own closed form on the support is `distribution().family().cdf(&x)`.  `sf(&x)` is the survival function `P(X > x)` (scipy's `sf`), `1` below the support and `0` above it.

**Far tails keep their digits.**  The classic closed forms cancel in a far tail: for a standard normal, `P(X > 20) = ½ − ½ erf(10√2)` is the difference of two numbers that agree to 89 digits, and it evaluates to `0`.  So a family also supplies a survival function and a lower-tail CDF written without the cancellation (`½ erfc(z/√2)`, `Γ(k, x)/Γ(k)`, `I_{1−x}(β, α)`, `atan(γ/(x − x₀))/π`, `γ(k + 1, λ)/k!` for a Poisson tail, `Σ_{j>k}` of the pmf for a binomial one, `(b − x)²/((b − a)(b − c))` for a triangular one), and a numeric point beyond `2⁻³²` in either tail is measured with them: `P(X > 20)` is `½ erfc(10√2) ≈ 2.75e-89`, an interval inside the upper tail is `S(lo) − S(hi)`, and the wrappers (truncation, affine maps, mixtures, order statistics) carry the same forms through.  A lower tail `1 − e^{−y}` (exponential, Weibull, Pareto with `y = α ln(x/xₘ)`, geometric with `y = −k ln(1 − p)`) is written `2 e^{−y/2} sinh(y/2)`, which has no `expm1` node to rely on but keeps its digits for any tiny `y`, exact or irrational: `Exponential(√2).cdf(10⁻¹⁰⁰) ≈ 1.414e-100`.  Near the median the classic forms are kept, so exact answers read as before.

## Continuous families

Every family is a struct (`stats::Normal`, `stats::Gamma`, …) implementing the `stats::Family` trait — support, density, and the closed forms it has — wrapped in a `Distribution` by a `Distribution::try_<name>(…)` constructor (validates numeric parameters) or its `Distribution::<name>(…)` twin (unchecked). Parameters follow SymPy's order and meaning. `Distribution::downcast_ref::<Normal>()` recovers the struct; `Distribution::from_family(my_family)` admits your own (implement `Family` with a support, a density and `eq_family` via `stats::same_family`, and every query below works).

| Family | Constructor | Support | Closed forms |
|--------|-------------|---------|--------------|
| `Normal(μ, σ)` | `normal(mean, std)` | ℝ | mean, variance, all moments, cdf (`erf`), mgf, quantile (`erfinv`) |
| `Uniform(a, b)` | `uniform(lo, hi)` | `[a, b]` | everything; mgf `(e^{bt} − e^{at})/((b−a)t)` |
| `Exponential(λ)` | `exponential(rate)` | `[0, ∞)` | everything; `E[Xⁿ] = n!/λⁿ` |
| `Gamma(k, θ)` | `gamma(shape, scale)` | `[0, ∞)` | moments `θⁿ (k)ₙ`, cdf `γ(k, x/θ)/Γ(k)` (elementary for integer/half-integer `k`), mgf `(1 − θt)^{−k}`; no quantile |
| `ChiSquared(k)` | `chi_squared(dof)` | `[0, ∞)` | as `Gamma(k/2, 2)` |
| `Beta(α, β)` | `beta(alpha, beta)` | `[0, 1]` | moments `(α)ₙ/(α+β)ₙ`; cdf closed-form via the regularized incomplete beta (a polynomial for integer `α, β`), mgf by integration; no quantile |
| `Cauchy(x₀, γ)` | `cauchy(location, scale)` | ℝ | cdf `½ + atan((x−x₀)/γ)/π`, quantile `x₀ + γ tan(π(p−½))`; **no moments** |
| `Laplace(μ, b)` | `laplace(mean, scale)` | ℝ | all moments (even central moments `n! bⁿ`), piecewise cdf, mgf, quantile |
| `Logistic(μ, s)` | `logistic(mean, scale)` | ℝ | all moments (via Bernoulli numbers), cdf, mgf `e^{μt} B(1−st, 1+st)`, quantile `μ + s ln(p/(1−p))` |
| `LogNormal(μ, σ)` | `log_normal(mu, sigma)` | `(0, ∞)` | moments `e^{nμ + n²σ²/2}`, cdf (`erf`), quantile (`erfinv`); no mgf |
| `StudentT(ν)` | `student_t(dof)` | ℝ | mean `0` (`ν > 1`), variance `ν/(ν−2)` (`ν > 2`), even moments for `n < ν`; cdf closed-form via the regularized incomplete beta (elementary for odd `ν`); no mgf, no quantile |
| `FDistribution(d₁, d₂)` | `f_distribution(d1, d2)` | `(0, ∞)` | mean `d₂/(d₂−2)` (`d₂ > 2`), variance (`d₂ > 4`), moments for `n < d₂/2`; cdf via the regularized incomplete beta `I_{d₁x/(d₁x+d₂)}(d₁/2, d₂/2)`; no mgf, no quantile |
| `Weibull(λ, k)` | `weibull(scale, shape)` | `[0, ∞)` | moments `λⁿ Γ(1 + n/k)`, cdf `1 − e^{−(x/λ)ᵏ}`, quantile `λ(−ln(1−p))^{1/k}`; no mgf |
| `Pareto(x_m, α)` | `pareto(scale, shape)` | `[x_m, ∞)` | moments `α x_mⁿ/(α−n)` for `n < α`, cdf `1 − (x_m/x)^α`, quantile `x_m (1−p)^{−1/α}`; no mgf |
| `Triangular(a, b, c)` | `triangular(lo, hi, mode)` | `[a, b]` | everything, piecewise density/cdf/quantile |

SymPy's `Weibull(alpha, beta)` has `alpha` = scale `λ` and `beta` = shape `k`; symplex names them `scale` and `shape`.

Where a closed form is missing the generic route takes over: `Beta(2, 3).cdf(x)` is the regularized incomplete beta, a polynomial for integer shapes — `3x⁴ − 8x³ + 6x²`; `StudentT(5).cdf(1)` simplifies to `7√5/(27π) + atan(√5/5)/π + ½`; `LogNormal.probability(…)` stays an unevaluated integral because the exact integrator does not close `∫ e^{−(ln x−μ)²/2σ²}/x` (the closed-form `cdf` is available instead). A moment that does not exist — `E[X³]` for `Pareto(1, 3)`, `E[X⁶]` for `StudentT(5)` — is likewise returned as the (divergent) integral, never as a number.

```rust
use symplex::prelude::*;
use symplex::stats::{Distribution, RandomVariable};

fn main() -> Result<(), SymplexError> {
    let ctx = Context::new();
    symplex::syms!(ctx; v, t, p);

    // Exponential(3): rate 3, mean 1/3.
    let x = RandomVariable::new(&ctx, "X", Distribution::try_exponential(ctx.int(3))?);
    println!("{}", x.mean());                                   // 1/3
    println!("{}", x.variance());                               // 1/9
    println!("{}", x.moment(3));                                // 2/9      E[X³] = 3!/3³
    println!("{}", x.skewness());                               // 2
    println!("{}", x.cdf(&v));                                  // -exp(-3*v) + 1
    println!("{}", x.mgf(&t));                                  // 3/(-t + 3)
    println!("{}", x.quantile(&p).unwrap());                    // -1/3*ln(-p + 1)
    println!("{}", x.probability(&x.symbol().gt(&ctx.one()))?); // exp(-3)

    // Gamma(3, 2): the CDF's incomplete gamma closes for integer shape.
    let g = RandomVariable::new(&ctx, "G", Distribution::gamma(ctx.int(3), ctx.int(2)));
    println!("{}", g.mean());                                   // 6
    println!("{}", g.moment(3));                                // 480      2³ · 3·4·5
    println!("{}", g.cdf(&ctx.int(4)));                         // -5*exp(-2) + 1
    println!("{}", g.mgf(&t));                                  // (-2*t + 1)^(-3)

    // Beta(2, 3): no closed CDF on the family, but the integral is a polynomial.
    let b = RandomVariable::new(&ctx, "B", Distribution::beta(ctx.int(2), ctx.int(3)));
    println!("{}", b.mean());                                   // 2/5
    println!("{}", b.cdf(&v).expand());                         // 3*v^4 - 8*v^3 + 6*v^2
    println!("{}", b.probability(&b.symbol().lt(&ctx.rational(1, 2)))?); // 11/16

    // Cauchy(1, 2): a CDF and quantile, but no mean.
    let c = RandomVariable::new(&ctx, "C", Distribution::cauchy(ctx.int(1), ctx.int(2)));
    println!("{}", c.cdf(&ctx.int(3)).simplify());              // 3/4
    println!("{}", c.median().unwrap().simplify());             // 1
    println!("{}", c.mean().has_unevaluated());                 // true  — the divergent integral, unevaluated
    Ok(())
}
```

### Symbolic parameters

Parameters may be symbols; declare their assumptions, as everywhere else in the crate.

```rust
use symplex::prelude::*;
use symplex::stats::{Distribution, RandomVariable};

fn main() {
    let ctx = Context::new();
    let k = ctx.symbol_with("k", &[Assumption::Positive]).unwrap();
    let theta = ctx.symbol_with("theta", &[Assumption::Positive]).unwrap();
    let g = RandomVariable::new(&ctx, "G", Distribution::gamma(k.clone(), theta.clone()));
    println!("{}", g.mean());       // k*theta
    println!("{}", g.variance());   // k*theta^2
    println!("{}", g.moment(2));    // theta^2*rising_factorial(k, 2)   = θ² k(k+1)

    let nu = ctx.symbol_with("nu", &[Assumption::Positive]).unwrap();
    let t = RandomVariable::new(&ctx, "T", Distribution::student_t(nu.clone()));
    println!("{}", t.variance());   // nu/(nu - 2)                        (valid for ν > 2)
}
```

For `StudentT` and `Pareto` the family returns the closed form for a symbolic parameter (its validity — `n < ν`, `n < α` — is the caller's promise) and `None` when a numeric parameter says the moment does not exist, so `Distribution::student_t(ctx.int(2)).family().variance()` is `None` and `RandomVariable::variance` falls through to the divergent integral.

### Differential entropy

`Distribution::entropy()` gives the differential entropy `−∫ f ln f` — a closed form for every continuous family (`Normal`: `½ ln(2πeσ²)`, `Uniform`: `ln(b − a)`, `Exponential`: `1 − ln λ`, `Gamma`: `k + ln θ + ln Γ(k) + (1−k)ψ(k)`, …), the expectation of `−ln f` otherwise.

```rust
use symplex::prelude::*;
use symplex::stats::Distribution;

fn main() {
    let ctx = Context::new();
    println!("{}", Distribution::uniform(ctx.int(2), ctx.int(5)).entropy());   // ln(3)
}
```

### Sampling

`sample(n, &mut rng)` draws from every built-in family, each route exact in distribution (no normal approximations). A continuous family with a closed-form quantile (`Normal`, `Uniform`, `Exponential`, `Cauchy`, `Laplace`, `Logistic`, `LogNormal`, `Weibull`, `Pareto`, `Triangular`) is drawn by inverse transform through the compiled quantile; a family on a finite lattice or table (`Bernoulli`, `Binomial`, `Hypergeometric`, `DiscreteUniform`/`Die`, `Finite`) by cumulative sums of its pmf. The rest have algorithms of their own: `Gamma` by Marsaglia–Tsang (with the `U^{1/k}` boost for shape `< 1`), `ChiSquared` as `Gamma(k/2, 2)`, `Beta` as `X/(X+Y)` of two gammas, `StudentT` as `Z/√(V/ν)`, `FDistribution` as `(U/d₁)/(V/d₂)` of two χ²; `Poisson` by Knuth's multiplication method below `λ = 30` and Hörmann's transformed rejection (PTRS) above; `Geometric` by the closed inversion `⌊ln U / ln(1−p)⌋ + 1`; `NegativeBinomial` as the Poisson–Gamma mixture (so a non-integer `r` is fine). Parameters must be numeric: a symbolic parameter is `Err(Unevaluable)`, a numeric one outside the family's domain `Err(InvalidArgument)`.

```rust
use symplex::prelude::*;
use symplex::stats::{Distribution, RandomVariable, Rng};

fn main() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let x = RandomVariable::new(&ctx, "X", Distribution::exponential(ctx.int(3)));
    let samples = x.sample(20_000, &mut Rng::new(1))?;
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    println!("{mean:.3}");   // 0.329 — within three standard errors of the exact mean 1/3
    Ok(())
}
```

## Conditioning, transformations, mixtures

Distributions compose. `x.given(&event)` is SymPy's `given(X, cond)`: the same symbol with the `Truncated` distribution `f / P(event)` on the event's region (`E[N | N > 0] = √(2/π)`, `E[B | B ≥ 2] = 325/131` for `Binomial(5, ⅓)`). `x.transform("Y", &g)` is the distribution of `g(X)`: an affine `aX + b` transports every closed form exactly (`2N + 1 ~ Normal(1, 2)`, with its mgf, quantile and moments); a strictly monotone `g` on the support (`eˣ`, `ln x`, `1/x`, …) and the even shapes `X²`, `|X|`, `X^{2k}` go through the change-of-variables formula (`N²` has the χ²(1) density `e^{−y/2}/√(2πy)`; `E[eᴺ] = √e` by LOTUS); a finite table or finite integer range has its values mapped and merged (`Die²`). `Distribution::mixture(&[(w₁, F₁), (w₂, F₂)])` is a finite mixture whose every query is the weighted sum of its components'. All of them sample: a truncation by the transported quantile (or rejection), a transformation by mapping inner samples, a mixture by choosing a component.

```rust
# use symplex::prelude::*;
# use symplex::stats::{Distribution, RandomVariable};
# let ctx = Context::new();
# let y = ctx.symbol("y");
# let a = Distribution::normal(ctx.int(0), ctx.int(1));
# let b = Distribution::exponential(ctx.int(2));
let n = RandomVariable::new(&ctx, "N", Distribution::normal(ctx.int(0), ctx.int(1)));
let half = n.given(&n.symbol().gt(&ctx.int(0)))?;         // N | N > 0
half.mean();                                             // √(2/π)
let sq = n.transform("S", &n.symbol().powi(2))?;          // N² ~ χ²(1)
sq.density(&y);                                          // e^{−y/2}/√(2πy)
let m = Distribution::mixture(&[(ctx.rational(1, 4), a), (ctx.rational(3, 4), b)])?;
# assert_eq!(RandomVariable::new(&ctx, "M", m).mean(), ctx.rational(3, 8));
# Ok::<(), SymplexError>(())
```

## Discrete families and finite tables

`Bernoulli(p)`, `Binomial(n, p)`, `Poisson(λ)`, `Geometric(p)` (support `1..`), `NegativeBinomial(r, p)` (failures before the `r`-th success), `Hypergeometric(N, m, n)`, `DiscreteUniform(a, b)` and `Die(sides)` — each with exact mean, variance, raw moments (Stirling-number and factorial-moment formulas, or derivatives of the moment generating function), pmf, cdf where it closes, and `mgf`. Probabilities of `X ≤ a`, `X > a`, `a ≤ X ≤ b` and `X = a` are exact rationals for rational parameters (`Binomial(5, 1/3)`: `P(Y > 2) = 17/81`).

`Distribution::try_finite(&ctx, vec![(value, probability), …])` is SymPy's `FiniteRV`: an explicit table whose values need not be integers. Moments are sums over the table, probabilities are decided by exact comparison of each value with the event's bounds, and `sample` draws by cumulative sums.

`quantile_f64(p)` of a discrete distribution is the smallest atom `k` with `F(k) ≥ p`, `p` read as the exact binary number it is, and decided exactly: `Binomial` and `Poisson` in the `f64` kernel, every other family (and a table, a wrapper, a family of your own) by the sign of `F(k) − p` in arbitrary precision.  Above `½` the same condition is decided on the small tail, `S(k) ≤ 1 − p`, where `F(k)` has no digits left: `Geometric(1/3).quantile_f64(1 − 2⁻⁵³)` is `91`, since `(2/3)⁹⁰ > 2⁻⁵³ ≥ (2/3)⁹¹` (scipy's `geom.ppf` says 90).  A level that equals a jump of `F` only up to rounding is decided by the double's exact value (the double nearest `0.9` exceeds `9/10`).

```rust
# use symplex::prelude::*;
# use symplex::stats::{Distribution, RandomVariable};
# let ctx = Context::new();
let coin = Distribution::try_finite(&ctx, vec![(ctx.int(1), ctx.rational(2, 3)), (ctx.int(0), ctx.rational(1, 3))])?;
let c = RandomVariable::new(&ctx, "C", coin);
c.mean();                                   // 2/3
c.probability(&c.symbol().eq_expr(&ctx.int(1)))?;   // 2/3
# assert_eq!(c.mean(), ctx.rational(2, 3));
# assert_eq!(c.probability(&c.symbol().eq_expr(&ctx.int(1)))?, ctx.rational(2, 3));
# Ok::<(), SymplexError>(())
```

## Several variables

The joint model is **independence**: `stats::expectation(&[&x, &y], &g)` computes `E[g(X, Y)]` for a polynomial `g` from the marginals' raw moments (a product per monomial), and `stats::{variance, covariance, correlation}` follow (`cov(X, 2X) = 2`, `corr(X, 2X + 1) = 1` for a standard normal). `stats::probability(&[&x, &y], &event)` handles rectangles (a conjunction of per-variable relations → product of marginals) and the ordering `X < Y` of two independent normals exactly (`1/2`). `stats::sum_distribution(&x, &y)` returns the closed family of a sum when there is one — Normal + Normal, Binomial + Binomial (same `p`), Poisson + Poisson, NegativeBinomial + NegativeBinomial, Gamma + Gamma (same scale; Exponential and χ² included). `stats::conditional_expectation(&x, &g, &event)` is `E[g · 1_event]/P(event)` (`E[X | X > 0] = √(2/π)` for a standard normal), `conditional_probability(&x, &event, &given)` likewise, and `x.entropy()` is the differential (or Shannon) entropy with closed forms for every continuous family.

## Analysis of variance on data

`stats::anova` holds every analysis of variance — `anova_one_way` (moved here from `hypothesis` in 0.18) and its extensions to factorial and repeated-measures designs — with the same contract as the rest of the data statistics: every quantity that is a rational function of the observations — sums of squares, `F`, `η²`, the sphericity `ε`s, Mauchly's `W` — is an exact `Q`, and p-values are exact expressions (`betainc_regularized` for an `F` tail, `uppergamma` for a χ² tail) evaluated with `eval_f64` when you ask.  The reference implementations named in `tests/v17/v17_anova.rs` are statsmodels' `anova_lm` / `AnovaRM`, pingouin's `rm_anova` / `epsilon` / `sphericity` and scipy's `tukey_hsd`; every number printed below is asserted there.

**Two-way ANOVA.** A `TwoWayData` holds the observations by cell (`cells[a][b]` = replicates; build it from nested vectors, `from_i64`, or long-form `Observation { a, b, y }` rows).  Cell sizes may differ.  A 2 × 3 design with three replicates per cell:

```rust
# use symplex::prelude::*;
# use symplex::linprog::q;
# let ctx = Context::new();
use symplex::stats::anova::{anova_two_way, anova_two_way_with, SsType, TwoWayData};
let data = TwoWayData::from_i64(&[
    &[&[4, 5, 6], &[6, 7, 8], &[9, 10, 12]],     // A = 0: cells for B = 0, 1, 2
    &[&[5, 5, 7], &[8, 9, 11], &[13, 14, 16]],   // A = 1
])?;
let r = anova_two_way(&ctx, &data)?;            // Type II sums of squares
# assert_eq!((r.factor_a.ss.clone(), r.factor_a.df, r.factor_a.f.clone()), (q(49, 2), 1, Some(q(441, 31))));
# assert_eq!((r.factor_b.ss.clone(), r.factor_b.df, r.factor_b.f.clone()), (q(1339, 9), 2, Some(q(1339, 31))));
# assert_eq!((r.interaction.ss.clone(), r.interaction.df, r.interaction.f.clone()), (q(25, 3), 2, Some(q(75, 31))));
# assert_eq!((r.residual.ss.clone(), r.residual.df, r.residual.ms.clone()), (q(62, 3), 12, Some(q(31, 18))));
# assert_eq!((r.total.ss.clone(), r.total.df), (q(3641, 18), 17));
# assert_eq!(r.factor_b.partial_eta_squared, Some(q(1339, 1525)));
# assert!((r.factor_a.p_value_f64()? - 0.0026634776886835334).abs() < 1e-12);
# let _ = anova_two_way_with(&ctx, &data, SsType::TypeI)?;
r.factor_a.ss;                 // 49/2     df 1    F 441/31    p 0.0026634776886835334
r.factor_b.ss;                 // 1339/9   df 2    F 1339/31   p 3.291990727040258e-06
r.interaction.ss;              // 25/3     df 2    F 75/31     p 0.13098893805732253
r.residual.ss;                 // 62/3     df 12   MS 31/18
r.total.ss;                    // 3641/18  df 17
r.factor_b.partial_eta_squared;   // 1339/1525
# Ok::<(), SymplexError>(())
```

Each row is an `AnovaRow { source, ss, df, ms, f, p_value, eta_squared, partial_eta_squared }` (`f`/`p_value` are `None` on the residual and total rows; `p_value_f64()` rounds).  Every sum of squares is the exact difference of the residual sums of squares of two nested least-squares fits on the dummy-coded design, so nothing is lost to floating point even when the design is unbalanced — which is where the *type* of sum of squares matters.  With equal cell sizes the factors are orthogonal and Types I, II and III agree, as above.  With unequal sizes they differ for the main effects: on the same layout with cell sizes 4, 2, 3 / 2, 4, 3, `anova_two_way` (Type II, `SS(A | B)`, statsmodels `anova_lm(typ=2)`) gives `SS_A = 24`, while `anova_two_way_with(&ctx, &data, SsType::TypeI)` (sequential, `SS(A)` first) gives `338/9`.  `SsType::TypeIII` uses sum-to-zero contrasts, the SPSS / `car::Anova(type=3)` convention, and matches statsmodels on a model fit with `C(A, Sum) * C(B, Sum)`; the interaction row is the same under every type.

**Repeated measures.** `anova_repeated_measures(&ctx, &rows)` takes one row per subject over the `k` conditions and returns the conditions / subjects / error / total rows, the exact `F`, and the sphericity machinery: the Greenhouse–Geisser `ε̂ = (tr S̃)²/((k−1) tr S̃²)` computed exactly from the double-centred covariance (no eigenvalues needed), the Huynh–Feldt `ε̃`, the corrected p-values (the `F` tail with both degrees of freedom scaled by `ε`), and Mauchly's `W` with its χ² approximation.

```rust
# use symplex::prelude::*;
# use symplex::linprog::q;
# use symplex::stats::data::from_i64;
# let ctx = Context::new();
use symplex::stats::anova::anova_repeated_measures;
let y = [from_i64(&[5, 7, 9]), from_i64(&[4, 5, 8]), from_i64(&[6, 8, 10]),
         from_i64(&[3, 6, 4]), from_i64(&[7, 9, 13])];       // 5 subjects × 3 conditions
let r = anova_repeated_measures(&ctx, &y)?;
# assert_eq!((r.conditions.ss.clone(), r.conditions.df), (q(542, 15), 2));
# assert_eq!((r.subjects.ss.clone(), r.subjects.df), (q(764, 15), 4));
# assert_eq!((r.error.ss.clone(), r.error.df), (q(178, 15), 8));
# assert_eq!(r.f, q(1084, 89));
# assert_eq!(r.epsilon_gg, q(7921, 14597));
# assert_eq!(r.epsilon_hf, Some(q(4168, 7091)));
# assert!((r.p_value_gg.eval_f64()? - 0.021264365858261566).abs() < 1e-12);
# assert_eq!(r.mauchly.as_ref().map(|m| m.w.clone()), Some(q(1245, 7921)));
r.conditions.ss;    // 542/15   df 2
r.subjects.ss;      // 764/15   df 4
r.error.ss;         // 178/15   df 8
r.f;                // 1084/89 → 12.179775280898877      p 0.003735511033474317
r.epsilon_gg;       // 7921/14597 → 0.5426457491265329
r.epsilon_hf;       // Some(4168/7091) → 0.587787336059794
r.p_value_gg;       // → 0.021264365858261566   (F on 2ε̂ and 8ε̂ degrees of freedom)
r.mauchly;          // Some: W = 1245/7921, χ² 5.551145791696415 on 2 df, p 0.0623137671632364
# Ok::<(), SymplexError>(())
```

**Post hoc.** `tukey_hsd(&ctx, &groups, 0.95)` returns one `PairwiseComparison { i, j, diff, se, statistic, p_adj, ci }` per pair of a one-way design: the difference, the Tukey–Kramer standard error `√(MSE/2·(1/nᵢ + 1/nⱼ))` and the statistic are exact, while `p_adj` and the simultaneous interval come from the studentized range distribution, which has no closed form and is integrated numerically (`studentized_range_cdf` / `_sf` / `_quantile`, agreeing with scipy to about 1e-9).  `pairwise_t_tests(&ctx, &groups, Adjustment::Holm, 0.05)` is the alternative when variances differ: Welch tests for every pair, adjusted by Holm or Bonferroni through the `hypothesis` module.

### Tiny p-values

A p-value is an exact expression, so nothing is lost until you convert it — and `p_value_f64()` (that is, `eval_f64`) converts to an `f64`, which has nothing below about `1e-308`.  A χ² test on `[[9000, 1000], [1000, 9000]]` gives `χ² = 12800` on one degree of freedom; scipy reports `pvalue = 0.0`, and so does `p_value_f64()`, but the expression `uppergamma(1/2, 6400)/Gamma(1/2)` still holds the value.  Every result type with an exact p-value (`TestResult`, `ChiSquareResult`, `AnovaResult`, `RepeatedMeasuresAnova`, `Mauchly`; `AnovaRow` for its effect rows) implements the `stats::PValue` trait and offers the same methods inherently:

```rust
# use symplex::prelude::*;
# use symplex::linprog::qi;
# use symplex::stats::data::from_i64;
# use symplex::stats::PValue;
# let ctx = Context::new();
use symplex::stats::hypothesis::chi_square_independence;
let table = [from_i64(&[9000, 1000]), from_i64(&[1000, 9000])];
let r = chi_square_independence(&ctx, &table, false)?;
r.p_value_f64()?;          // 0.0 — underflow
r.p_value_log10()?;        // -2781.636383026783   (mpmath: -2781.6363830267828509)
r.p_value_ln()?;           // -6404.954469687346
r.p_value_decimal(20)?;    // "2.3100265595063985852e-2782"
r.p_value_decimal(5)?;     // "2.31e-2782"
# assert_eq!(r.p_value_f64()?, 0.0);
# assert!((r.p_value_log10()? - -2781.636383026783).abs() < 1e-9);
# assert_eq!(r.p_value_decimal(5)?, "2.31e-2782");
# assert_eq!(r.statistic, qi(12800));
r.statistic;               // 12800
# Ok::<(), SymplexError>(())
```

`p_value_log10` and `p_value_ln` evaluate the *logarithm* of the expression in arbitrary precision, so they are finite for any positive `p` (an exact `0`, as from a perfectly correlated `pearson_test`, gives `-∞`; an exact `1` gives `0`), and `p_value_decimal(digits)` prints the value itself with an exponent.  `Ols::p_values_log10(&ctx)` does the same for every coefficient of a regression.  The numbers above, and the same accessors on every other result type, are asserted against mpmath in `tests/v17/v17_pvalues.rs`.

## Survival regression: Cox proportional hazards

`stats::survival` (Kaplan–Meier, Greenwood, Nelson–Aalen, the log-rank test — see [Analysing Rater and Response Data](./response-analysis.md)) describes *when* events happen; `stats::cox` explains it with covariates.  `cox_ph(&obs, &x, &opts)` fits `h(t | x) = h₀(t)·exp(xᵀβ)` by Newton–Raphson on Cox's partial likelihood — the same `Observation { time, event }` rows, one `f64` covariate row per observation, no intercept (the baseline hazard absorbs it) — with Efron's tie correction by default (`Ties::Breslow` is the other), the analytic score and information matrix, and step-halving.  Ten subjects, one covariate, three censored:

```rust
# use symplex::prelude::*;
# use symplex::linprog::q;
# let ctx = Context::new();
use symplex::stats::cox::{cox_ph, CoxOpts};
use symplex::stats::survival::Observation;
let obs = Observation::from_i64(&[4, 7, 2, 9, 12, 5, 15, 3, 11, 8],
                                &[true, true, true, false, true, true, false, true, true, false]);
let x: Vec<Vec<f64>> = [3.0, 1.0, 5.0, 2.0, 0.0, 4.0, 1.0, 6.0, 2.0, 3.0].iter().map(|&v| vec![v]).collect();
let fit = cox_ph(&obs, &x, &CoxOpts::default())?;
fit.hazard_ratios();         // [2.401229718322006]       exp(β): each unit of x multiplies the hazard by 2.4
let lr = fit.llr_test(&ctx)?;   // χ² 7.286928045319378 on 1 df, p 0.006945814500177098
fit.score_statistic();       // 7.355733707724492         the log-rank-type score test at β = 0
fit.concordance()?;          // 31/38 — Harrell's C, exactly
&fit.baseline_hazard()[0];   // BaselineHazardRow { stratum: 0, time: 2, hazard: 0.002858840843475405, cumulative: 0.002858840843475405 }
&fit.baseline_hazard()[6];   //   … { time: 12, hazard: 0.2940113085020759, cumulative: 0.4700301513413364 }
# assert!((fit.coefficients[0] - 0.8759809887649096).abs() < 1e-9);
# assert!((fit.standard_errors[0] - 0.3809028296542367).abs() < 1e-9);
# assert!((fit.log_likelihood - -8.465216276861835).abs() < 1e-9);
# assert!((lr.statistic_f64()? - 7.286928045319378).abs() < 1e-9);
# assert!((fit.score_statistic() - 7.355733707724492).abs() < 1e-9);
# assert_eq!(fit.concordance()?, q(31, 38));
fit.coefficients;            // [0.8759809887649096]      statsmodels PHReg(..., ties='efron').fit().params
fit.standard_errors;         // [0.3809028296542367]      bse
fit.p_values;                // [0.021462431348704628]    two-sided normal
fit.log_likelihood;          // -8.465216276861835        llf;  fit.null_log_likelihood = -12.108680299521524
# Ok::<(), SymplexError>(())
```

The fit is numerical (`f64` coefficients, standard errors, residuals), so this is one of the places where the crate is not exact; what *is* exact is kept so: event times are `Q`, and the concordance index is a ratio of pair counts — `(concordant + ½ tied) / usable` over the pairs `tᵢ < tⱼ` with `i` an event — returned as a rational.  `llr_test`, `wald_test` and `score_test` return `TestResult`s (the trio `summary(coxph)` prints; `df = p`), so `p_value_log10` and friends work on them; for a single `0/1` covariate under Breslow ties with no tied event times the score statistic *is* the log-rank statistic of `log_rank_test`, which the tests assert.  `conf_int(c)` gives Wald intervals for `β` and `hazard_ratio_conf_int(c)` the same exponentiated; `baseline_hazard()` is the Breslow estimator `dₜ / Σ_{Rₜ} exp(xⱼᵀβ̂)` with its running sum; `schoenfeld_residuals()` (one row per event) and `martingale_residuals()` (one per subject, summing to zero) are the two diagnostics you plot; `predict_partial_hazard(&x₀)` is `exp(x₀ᵀβ̂)`.  `cox_ph_stratified(&obs, &x, &strata, &opts)` sums the partial likelihood over strata, each with its own baseline hazard.  A monotone likelihood (every subject with the larger covariate value fails before every subject with the smaller one, so `β̂ → ∞`; `coxph` warns "beta may be infinite") is a `ComputationFailed` that names the covariate, as `logit` does for separation.  Every number above is asserted against statsmodels 0.15 in `tests/v18/v18_cox.rs`, including the Freireich 6-MP data (`β̂ = −1.5721` Efron, `−1.5092` Breslow, as in R).

## What is exact and what is not

Everything above is symbolic: rational parameters give rational or closed-form answers, and symbolic parameters stay symbolic (`E[X] = μ`). Two honest gaps: the integrator does not close every density integral (`LogNormal` probabilities stay as an `Integral` although its closed-form `cdf` is available — `probability` uses the `cdf` first), and infinite sums with *symbolic* parameters may stay as a `Sum`. `RandomVariable::sample` is the only randomness in the symbolic layer (seeded through `stats::Rng` so results reproduce); the data modules' bootstrap/permutation tests, `MarkovChain::sample_path` and `MultivariateNormal` sampling take an explicit seeded `Rng`.
