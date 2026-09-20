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
* Nothing is numerical by default. `RandomVariable::sample` is the only place a random number generator appears, and `stats::Rng` is a seeded, reproducible SplitMix64 so Monte-Carlo sanity checks are deterministic.
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
| `sample(n, &mut rng)` | `sample(X, size=n)` | `Result<Vec<f64>>` — inverse-transform sampling through the quantile |

Events are `BoolEx` relations in the variable's symbol: `X < a`, `X ≤ a`, `X > a`, `X ≥ a`, `X = a` and conjunctions of these. The density is integrated over the part of the support where the condition holds, so `P(X > 1)` for `Exponential(3)` is `exp(-3)` and `P(0 < U < 1/4)` for `Uniform(0, 1)` is `1/4`.

## Continuous families

Every family is a variant of `stats::ContinuousFamily` with a `Distribution::try_<name>(…)` constructor (validates numeric parameters) and a `Distribution::<name>(…)` twin (unchecked). Parameters follow SymPy's order and meaning.

| Family | Constructor | Support | Closed forms |
|--------|-------------|---------|--------------|
| `Normal(μ, σ)` | `normal(mean, std)` | ℝ | mean, variance, all moments, cdf (`erf`), mgf, quantile (`erfinv`) |
| `Uniform(a, b)` | `uniform(lo, hi)` | `[a, b]` | everything; mgf `(e^{bt} − e^{at})/((b−a)t)` |
| `Exponential(λ)` | `exponential(rate)` | `[0, ∞)` | everything; `E[Xⁿ] = n!/λⁿ` |
| `Gamma(k, θ)` | `gamma(shape, scale)` | `[0, ∞)` | moments `θⁿ (k)ₙ`, cdf `γ(k, x/θ)/Γ(k)` (elementary for integer/half-integer `k`), mgf `(1 − θt)^{−k}`; no quantile |
| `ChiSquared(k)` | `chi_squared(dof)` | `[0, ∞)` | as `Gamma(k/2, 2)` |
| `Beta(α, β)` | `beta(alpha, beta)` | `[0, 1]` | moments `(α)ₙ/(α+β)ₙ`; cdf and mgf by integration (polynomial cdf for integer `α, β`); no quantile |
| `Cauchy(x₀, γ)` | `cauchy(location, scale)` | ℝ | cdf `½ + atan((x−x₀)/γ)/π`, quantile `x₀ + γ tan(π(p−½))`; **no moments** |
| `Laplace(μ, b)` | `laplace(mean, scale)` | ℝ | all moments (even central moments `n! bⁿ`), piecewise cdf, mgf, quantile |
| `Logistic(μ, s)` | `logistic(mean, scale)` | ℝ | all moments (via Bernoulli numbers), cdf, mgf `e^{μt} B(1−st, 1+st)`, quantile `μ + s ln(p/(1−p))` |
| `LogNormal(μ, σ)` | `log_normal(mu, sigma)` | `(0, ∞)` | moments `e^{nμ + n²σ²/2}`, cdf (`erf`), quantile (`erfinv`); no mgf |
| `StudentT(ν)` | `student_t(dof)` | ℝ | mean `0` (`ν > 1`), variance `ν/(ν−2)` (`ν > 2`), even moments for `n < ν`; cdf by integration (elementary for odd `ν`); no mgf, no quantile |
| `Weibull(λ, k)` | `weibull(scale, shape)` | `[0, ∞)` | moments `λⁿ Γ(1 + n/k)`, cdf `1 − e^{−(x/λ)ᵏ}`, quantile `λ(−ln(1−p))^{1/k}`; no mgf |
| `Pareto(x_m, α)` | `pareto(scale, shape)` | `[x_m, ∞)` | moments `α x_mⁿ/(α−n)` for `n < α`, cdf `1 − (x_m/x)^α`, quantile `x_m (1−p)^{−1/α}`; no mgf |
| `Triangular(a, b, c)` | `triangular(lo, hi, mode)` | `[a, b]` | everything, piecewise density/cdf/quantile |

SymPy's `Weibull(alpha, beta)` has `alpha` = scale `λ` and `beta` = shape `k`; symplex names them `scale` and `shape`.

Where a closed form is missing the generic route takes over: `Beta(2, 3).cdf(x)` integrates the density and returns `3x⁴ − 8x³ + 6x²`; `StudentT(5).cdf(1)` simplifies to `7√5/(27π) + atan(√5/5)/π + ½`; `LogNormal.probability(…)` stays an unevaluated integral because the exact integrator does not close `∫ e^{−(ln x−μ)²/2σ²}/x` (the closed-form `cdf` is available instead). A moment that does not exist — `E[X³]` for `Pareto(1, 3)`, `E[X⁶]` for `StudentT(5)` — is likewise returned as the (divergent) integral, never as a number.

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
    let k = ctx.symbol_with("k", &[Assumption::Positive]);
    let theta = ctx.symbol_with("theta", &[Assumption::Positive]);
    let g = RandomVariable::new(&ctx, "G", Distribution::gamma(k.clone(), theta.clone()));
    println!("{}", g.mean());       // k*theta
    println!("{}", g.variance());   // k*theta^2
    println!("{}", g.moment(2));    // theta^2*rising_factorial(k, 2)   = θ² k(k+1)

    let nu = ctx.symbol_with("nu", &[Assumption::Positive]);
    let t = RandomVariable::new(&ctx, "T", Distribution::student_t(nu.clone()));
    println!("{}", t.variance());   // nu/(nu - 2)                        (valid for ν > 2)
}
```

For `StudentT` and `Pareto` the family returns the closed form for a symbolic parameter (its validity — `n < ν`, `n < α` — is the caller's promise) and `None` when a numeric parameter says the moment does not exist, so `Distribution::student_t(ctx.int(2)).variance(&ctx)` is `None` and `RandomVariable::variance` falls through to the divergent integral.

### Differential entropy

`ContinuousFamily::entropy(&ctx)` gives the closed-form differential entropy `−∫ f ln f` for every family (`Normal`: `½ ln(2πeσ²)`, `Uniform`: `ln(b − a)`, `Exponential`: `1 − ln λ`, `Gamma`: `k + ln θ + ln Γ(k) + (1−k)ψ(k)`, …).

```rust
use symplex::prelude::*;
use symplex::stats::Distribution;

fn main() {
    let ctx = Context::new();
    let Distribution::Continuous(fam) = Distribution::uniform(ctx.int(2), ctx.int(5)) else { return };
    println!("{}", fam.entropy(&ctx).unwrap());   // ln(3)
}
```

### Sampling

`sample(n, &mut rng)` draws by inverse transform sampling for every continuous family with a closed-form quantile (`Uniform`, `Exponential`, `Cauchy`, `Laplace`, `Logistic`, `Weibull`, `Pareto`, `Triangular`; `Normal`/`LogNormal` need `erfinv`, which the numeric compiler does not support yet). Families without a quantile (`Gamma`, `ChiSquared`, `Beta`, `StudentT`) return `Err(NotImplemented)`.

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

## Discrete families

Discrete families (`Binomial(n, p)`, …) live in `stats::DiscreteFamily` with the same shape: a pmf on an integer support, closed forms where they exist, and `summation` over the support otherwise. `probability` rounds non-integer bounds inwards and moves strict bounds by one, so `P(Y > 2)` and `P(Y ≥ 3)` agree. See the API docs for `DiscreteFamily` for the current list.

## Discrete families and finite tables

`Bernoulli(p)`, `Binomial(n, p)`, `Poisson(λ)`, `Geometric(p)` (support `1..`), `NegativeBinomial(r, p)` (failures before the `r`-th success), `Hypergeometric(N, m, n)`, `DiscreteUniform(a, b)` and `Die(sides)` — each with exact mean, variance, raw moments (Stirling-number and factorial-moment formulas, or derivatives of the moment generating function), pmf, cdf where it closes, and `mgf`. Probabilities of `X ≤ a`, `X > a`, `a ≤ X ≤ b` and `X = a` are exact rationals for rational parameters (`Binomial(5, 1/3)`: `P(Y > 2) = 17/81`).

`Distribution::try_finite(vec![(value, probability), …])` is SymPy's `FiniteRV`: an explicit table whose values need not be integers. Moments are sums over the table, probabilities are decided by exact comparison of each value with the event's bounds, and `sample` draws by cumulative sums.

```rust,ignore
let coin = Distribution::try_finite(vec![(ctx.int(1), ctx.rational(2, 3)), (ctx.int(0), ctx.rational(1, 3))])?;
let c = RandomVariable::new(&ctx, "C", coin);
c.mean();                                   // 2/3
c.probability(&c.symbol().eq_expr(&ctx.int(1)))?;   // 2/3
```

## Several variables

The joint model is **independence**: `stats::expectation(&[&x, &y], &g)` computes `E[g(X, Y)]` for a polynomial `g` from the marginals' raw moments (a product per monomial), and `stats::{variance, covariance, correlation}` follow (`cov(X, 2X) = 2`, `corr(X, 2X + 1) = 1` for a standard normal). `stats::probability(&[&x, &y], &event)` handles rectangles (a conjunction of per-variable relations → product of marginals) and the ordering `X < Y` of two independent normals exactly (`1/2`). `stats::sum_distribution(&x, &y)` returns the closed family of a sum when there is one — Normal + Normal, Binomial + Binomial (same `p`), Poisson + Poisson, NegativeBinomial + NegativeBinomial, Gamma + Gamma (same scale; Exponential and χ² included). `stats::conditional_expectation(&x, &g, &event)` is `E[g · 1_event]/P(event)` (`E[X | X > 0] = √(2/π)` for a standard normal), `conditional_probability(&x, &event, &given)` likewise, and `x.entropy()` is the differential (or Shannon) entropy with closed forms for every continuous family.

## What is exact and what is not

Everything above is symbolic: rational parameters give rational or closed-form answers, and symbolic parameters stay symbolic (`E[X] = μ`). Two honest gaps: the integrator does not close every density integral (`LogNormal` probabilities stay as an `Integral` although its closed-form `cdf` is available — `probability` uses the `cdf` first), and infinite sums with *symbolic* parameters may stay as a `Sum`. `RandomVariable::sample` is the only numerical routine, seeded through `stats::Rng` so results reproduce.
