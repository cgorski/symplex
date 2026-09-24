//! Discrete distribution families (probability mass on integers, or on an
//! explicit table of values).  Same shape as [`super::continuous`]: a
//! struct per family, `impl Family` with support, pmf and the closed forms
//! it has.  Raw moments come from one of two exact routes shared by
//! several families: factorial moments through Stirling numbers
//! ([`raw_moment_from_factorial_moments`]) and derivatives of the moment
//! generating function at `0` ([`moment_from_mgf`]).
//!
//! Sampling: a family on a finite lattice leaves [`Family::sampler`] unset
//! and is drawn from the cumulative sums of its pmf
//! ([`Distribution::sampler`]); the three on `0..∞` / `1..∞` — `Poisson`,
//! `Geometric`, `NegativeBinomial` — supply their own exact routes.

use std::fmt;

use num_traits::{One, Zero};

use crate::api::context::Context;
use crate::api::expr::{BoolEx, Ex};
use crate::base::errors::SymplexError;
use crate::base::numeric::Q;
use crate::domains::combinatorics::stirling2;

use super::continuous::sampler_positive;
use super::family::{Distribution, Family, Sampler, family_boilerplate};
use super::sample::{self, Rng};
use super::support::Support;

fn invalid(reason: impl Into<String>) -> SymplexError {
    SymplexError::invalid_argument("stats", reason)
}

/// The exact value of a numeric parameter; `None` for a symbolic one.
fn numeric(e: &Ex) -> Option<Q> {
    e.eval().as_rational()
}

/// Reject a numeric probability outside `[0, 1]`; accept symbolic ones.
fn require_probability(p: &Ex, what: &str) -> Result<(), SymplexError> {
    if let Some(q) = numeric(p)
        && (q < Q::zero() || q > Q::one())
    {
        return Err(invalid(format!("{what} must lie in [0, 1], got `{p}`")));
    }
    Ok(())
}

/// Reject a numeric probability outside `(0, 1]` (`allow_one`) or `(0, 1)`;
/// accept symbolic ones.
fn require_probability_positive(p: &Ex, what: &str, allow_one: bool) -> Result<(), SymplexError> {
    if let Some(q) = numeric(p)
        && (q <= Q::zero() || q > Q::one() || (!allow_one && q == Q::one()))
    {
        let range = if allow_one { "(0, 1]" } else { "(0, 1)" };
        return Err(invalid(format!("{what} must lie in {range}, got `{p}`")));
    }
    Ok(())
}

/// Reject a numeric count that is not a non-negative integer; accept
/// symbolic ones.
fn require_count(n: &Ex, what: &str) -> Result<(), SymplexError> {
    if let Some(q) = numeric(n)
        && (!q.is_integer() || q < Q::zero())
    {
        return Err(invalid(format!(
            "{what} must be a non-negative integer, got `{n}`"
        )));
    }
    Ok(())
}

/// Reject a numeric value that is not an integer; accept symbolic ones.
fn require_integer(e: &Ex, what: &str) -> Result<(), SymplexError> {
    if let Some(q) = numeric(e)
        && !q.is_integer()
    {
        return Err(invalid(format!("{what} must be an integer, got `{e}`")));
    }
    Ok(())
}

/// Reject numeric `a`, `b` with `a > b`; accept when either is symbolic.
fn require_le(a: &Ex, b: &Ex, what: &str) -> Result<(), SymplexError> {
    if let (Some(x), Some(y)) = (numeric(a), numeric(b))
        && x > y
    {
        return Err(invalid(format!("{what}: `{a}` exceeds `{b}`")));
    }
    Ok(())
}

/// A success probability as the `f64` in `(0, 1]` a sampler needs,
/// evaluated once when the sampler is built.
///
/// # Errors
///
/// The evaluation error for a symbolic parameter;
/// [`SymplexError::InvalidArgument`] for a number outside `(0, 1]`.
fn sampler_probability(p: &Ex, what: &str) -> Result<f64, SymplexError> {
    let v = p.eval_f64()?;
    if !(v > 0.0 && v <= 1.0) {
        return Err(invalid(format!(
            "{what} must lie in (0, 1] to sample, got `{p}`"
        )));
    }
    Ok(v)
}

/// The falling factorial `x^{(k)} = x(x−1)⋯(x−k+1)` as an explicit product
/// (a polynomial in `x`, so symbolic parameters stay polynomial).
fn falling_factorial(x: &Ex, k: u32) -> Ex {
    let ctx = x.context();
    let mut acc = ctx.one();
    for j in 0..k {
        acc *= x - ctx.int(i64::from(j));
    }
    acc
}

/// `E[Xⁿ] = Σ_{k=1}^{n} S(n, k) · E[X^{(k)}]`: a raw moment from the
/// factorial moments `E[X^{(k)}] = E[X(X−1)⋯(X−k+1)]` through the Stirling
/// numbers of the second kind (`xⁿ = Σ_k S(n, k) x^{(k)}`).  `None` only if
/// a Stirling number is out of range (it is not, for `u32` arguments).
fn raw_moment_from_factorial_moments(
    n: u32,
    ctx: &Context,
    factorial_moment: impl Fn(u32) -> Ex,
) -> Option<Ex> {
    let mut acc = ctx.zero();
    for k in 1..=n {
        let s = ctx.from_bigint(stirling2(n, k)?);
        acc += s * factorial_moment(k);
    }
    Some(acc.simplify())
}

/// `E[Xⁿ] = M⁽ⁿ⁾(0)`: the `n`-th derivative of the moment generating
/// function at `t = 0`, simplified.  An exact route for families whose mgf
/// is elementary; `None` if the derivative did not evaluate.
fn moment_from_mgf(mgf: impl Fn(&Ex) -> Option<Ex>, n: u32, ctx: &Context) -> Option<Ex> {
    let t = ctx.symbol("_t_mgf");
    let mut m = mgf(&t)?;
    for _ in 0..n {
        m = m.diff(&t);
    }
    let at_zero = m.subs(&t, &ctx.zero()).simplify();
    (!at_zero.has_unevaluated() && !at_zero.contains(&t)).then_some(at_zero)
}

// ═══════════════════════════════════════════════════════════════════════════
// Finite tables
// ═══════════════════════════════════════════════════════════════════════════

/// An explicit finite table `P(X = vᵢ) = pᵢ` (SymPy `FiniteRV`); the values
/// need not be integers.
#[derive(Clone, Debug)]
pub struct Finite {
    /// `(value, probability)` pairs, in the order given.
    pub table: Vec<(Ex, Ex)>,
    ctx: Context,
}

impl PartialEq for Finite {
    fn eq(&self, other: &Self) -> bool {
        self.table == other.table
    }
}

impl Finite {
    /// A table in `ctx` (kept explicitly so that an empty table still has
    /// a context).
    pub fn new(ctx: &Context, table: Vec<(Ex, Ex)>) -> Self {
        Finite {
            table,
            ctx: ctx.clone(),
        }
    }
}

impl Family for Finite {
    family_boilerplate!(Finite, "Finite");

    fn context(&self) -> Context {
        self.ctx.clone()
    }

    fn parameters(&self) -> Vec<(&'static str, Ex)> {
        self.table
            .iter()
            .flat_map(|(v, p)| [("value", v.clone()), ("probability", p.clone())])
            .collect()
    }

    fn support(&self) -> Support {
        Support::points(self.table.iter().map(|(v, _)| v.clone()).collect())
    }

    // Piecewise((pᵢ, k = vᵢ), …, (0, True)): the value 0 off the table.
    fn density(&self, k: &Ex) -> Ex {
        let zero = self.ctx.zero();
        let true_ = self.ctx.bool_true();
        let pairs: Vec<(Ex, BoolEx)> = self
            .table
            .iter()
            .map(|(v, p)| (p.clone(), k.eq_expr(v)))
            .chain(std::iter::once((zero, true_)))
            .collect();
        let refs: Vec<(&Ex, &BoolEx)> = pairs.iter().map(|(a, b)| (a, b)).collect();
        Ex::piecewise(&refs)
    }

    fn mean(&self) -> Option<Ex> {
        Some(
            self.table
                .iter()
                .fold(self.ctx.zero(), |acc, (v, p)| acc + v * p)
                .simplify(),
        )
    }

    fn variance(&self) -> Option<Ex> {
        let mean = self
            .table
            .iter()
            .fold(self.ctx.zero(), |acc, (v, p)| acc + v * p);
        let second = self
            .table
            .iter()
            .fold(self.ctx.zero(), |acc, (v, p)| acc + v.powi(2) * p);
        Some((second - mean.powi(2)).simplify())
    }

    fn raw_moment(&self, n: u32) -> Option<Ex> {
        Some(
            self.table
                .iter()
                .fold(self.ctx.zero(), |acc, (v, p)| {
                    acc + v.powi(i64::from(n)) * p
                })
                .simplify(),
        )
    }

    fn mgf(&self, t: &Ex) -> Option<Ex> {
        Some(
            self.table
                .iter()
                .fold(self.ctx.zero(), |acc, (v, p)| acc + p * (v * t).exp())
                .simplify(),
        )
    }

    fn fmt_display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Finite({{")?;
        for (i, (v, p)) in self.table.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{v}: {p}")?;
        }
        write!(f, "}})")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Bernoulli, Binomial
// ═══════════════════════════════════════════════════════════════════════════

/// `Bernoulli(p)`: `P(X = 1) = p`, `P(X = 0) = 1 − p` on `{0, 1}`.
#[derive(Clone, Debug, PartialEq)]
pub struct Bernoulli {
    /// Success probability `p ∈ [0, 1]`.
    pub p: Ex,
}

impl Family for Bernoulli {
    family_boilerplate!(Bernoulli, "Bernoulli", [p]);

    fn support(&self) -> Support {
        let ctx = self.context();
        Support::integers(&ctx, Some(ctx.zero()), Some(ctx.one()))
    }

    // pᵏ (1−p)^{1−k} is p at k = 1 and 1 − p at k = 0.
    fn density(&self, k: &Ex) -> Ex {
        let ctx = self.context();
        self.p.pow(k) * (ctx.one() - &self.p).pow(&(ctx.one() - k))
    }

    fn mean(&self) -> Option<Ex> {
        Some(self.p.clone())
    }

    fn variance(&self) -> Option<Ex> {
        Some((&self.p * (self.context().one() - &self.p)).simplify())
    }

    // Xⁿ = X on {0, 1}, so E[Xⁿ] = p.
    fn raw_moment(&self, _n: u32) -> Option<Ex> {
        Some(self.p.clone())
    }

    // 0 for k < 0, 1 − p for 0 ≤ k < 1, 1 for k ≥ 1.
    fn cdf(&self, k: &Ex) -> Option<Ex> {
        let ctx = self.context();
        let zero = ctx.zero();
        let one = ctx.one();
        let q = &one - &self.p;
        Some(Ex::piecewise(&[
            (&zero, &k.lt(&zero)),
            (&q, &k.lt(&one)),
            (&one, &k.ge(&one)),
        ]))
    }

    // 1 − p + p eᵗ
    fn mgf(&self, t: &Ex) -> Option<Ex> {
        Some(self.context().one() - &self.p + &self.p * t.exp())
    }
}

/// `Binomial(n, p)`: `P(X = k) = C(n, k) pᵏ (1−p)ⁿ⁻ᵏ` on `0..=n`.
#[derive(Clone, Debug, PartialEq)]
pub struct Binomial {
    /// Number of trials `n`.
    pub n: Ex,
    /// Success probability `p ∈ [0, 1]`.
    pub p: Ex,
}

impl Family for Binomial {
    family_boilerplate!(Binomial, "Binomial", [n, p]);

    fn support(&self) -> Support {
        let ctx = self.context();
        Support::integers(&ctx, Some(ctx.zero()), Some(self.n.clone()))
    }

    fn density(&self, k: &Ex) -> Ex {
        let q = self.context().one() - &self.p;
        self.n.binomial(k) * self.p.pow(k) * q.pow(&(&self.n - k))
    }

    fn mean(&self) -> Option<Ex> {
        Some((&self.n * &self.p).simplify())
    }

    fn variance(&self) -> Option<Ex> {
        Some((&self.n * &self.p * (self.context().one() - &self.p)).simplify())
    }

    // Factorial moments E[X^{(k)}] = n^{(k)} pᵏ.
    fn raw_moment(&self, n: u32) -> Option<Ex> {
        raw_moment_from_factorial_moments(n, &self.context(), |k| {
            falling_factorial(&self.n, k) * self.p.powi(i64::from(k))
        })
    }

    // (1 − p + p eᵗ)ⁿ
    fn mgf(&self, t: &Ex) -> Option<Ex> {
        Some((self.context().one() - &self.p + &self.p * t.exp()).pow(&self.n))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Poisson, Geometric, NegativeBinomial
// ═══════════════════════════════════════════════════════════════════════════

/// `Poisson(λ)`: `P(X = k) = λᵏ e^{−λ} / k!` on `0..∞`.
#[derive(Clone, Debug, PartialEq)]
pub struct Poisson {
    /// Rate `λ > 0`.
    pub rate: Ex,
}

impl Family for Poisson {
    family_boilerplate!(Poisson, "Poisson", [rate]);

    fn support(&self) -> Support {
        let ctx = self.context();
        Support::integers(&ctx, Some(ctx.zero()), None)
    }

    fn density(&self, k: &Ex) -> Ex {
        self.rate.pow(k) * (-&self.rate).exp() / k.factorial()
    }

    fn mean(&self) -> Option<Ex> {
        Some(self.rate.clone())
    }

    fn variance(&self) -> Option<Ex> {
        Some(self.rate.clone())
    }

    // Factorial moments E[X^{(k)}] = λᵏ (Touchard polynomial
    // E[Xⁿ] = Σ_k S(n, k) λᵏ).
    fn raw_moment(&self, n: u32) -> Option<Ex> {
        raw_moment_from_factorial_moments(n, &self.context(), |k| self.rate.powi(i64::from(k)))
    }

    // Γ(⌊k⌋+1, λ) / ⌊k⌋!
    fn cdf(&self, k: &Ex) -> Option<Ex> {
        let kf = k.floor();
        Some(self.rate.uppergamma(&(&kf + self.context().one())) / kf.factorial())
    }

    // γ(⌊k⌋+1, λ) / ⌊k⌋! (the Poisson–gamma duality `P(X > k) = P(G ≤ λ)`,
    // `G ~ Gamma(k + 1, 1)`); `eval` keeps `γ` where its closed form
    // `k! − Γ(k+1, λ)` would cancel.
    fn sf(&self, k: &Ex) -> Option<Ex> {
        let kf = k.floor();
        Some(self.rate.lowergamma(&(&kf + self.context().one())) / kf.factorial())
    }

    // exp(λ(eᵗ − 1))
    fn mgf(&self, t: &Ex) -> Option<Ex> {
        Some((&self.rate * (t.exp() - self.context().one())).exp())
    }

    // Knuth's multiplication method below λ = 30, Hörmann's PTRS above
    // (see `sample::poisson`).
    fn sampler(&self) -> Option<Result<Sampler, SymplexError>> {
        Some(
            sampler_positive(&self.rate, "the rate").map(|lambda| -> Sampler {
                Box::new(move |rng: &mut Rng| sample::poisson(rng, lambda))
            }),
        )
    }
}

/// `Geometric(p)`: the number of trials up to and including the first
/// success, `P(X = k) = (1−p)^{k−1} p` on `1..∞` (SymPy's convention).
#[derive(Clone, Debug, PartialEq)]
pub struct Geometric {
    /// Success probability `p ∈ (0, 1]`.
    pub p: Ex,
}

impl Family for Geometric {
    family_boilerplate!(Geometric, "Geometric", [p]);

    fn support(&self) -> Support {
        let ctx = self.context();
        Support::integers(&ctx, Some(ctx.one()), None)
    }

    fn density(&self, k: &Ex) -> Ex {
        let ctx = self.context();
        (ctx.one() - &self.p).pow(&(k - ctx.one())) * &self.p
    }

    // 1/p
    fn mean(&self) -> Option<Ex> {
        Some((self.context().one() / &self.p).simplify())
    }

    // (1−p)/p²
    fn variance(&self) -> Option<Ex> {
        Some(((self.context().one() - &self.p) / self.p.powi(2)).simplify())
    }

    fn raw_moment(&self, n: u32) -> Option<Ex> {
        moment_from_mgf(|t| Family::mgf(self, t), n, &self.context())
    }

    // 1 − (1−p)^{⌊k⌋}
    fn cdf(&self, k: &Ex) -> Option<Ex> {
        let ctx = self.context();
        Some(ctx.one() - (ctx.one() - &self.p).pow(&k.floor()))
    }

    // (1−p)^{⌊k⌋}
    fn sf(&self, k: &Ex) -> Option<Ex> {
        let ctx = self.context();
        Some((ctx.one() - &self.p).pow(&k.floor()))
    }

    // p eᵗ / (1 − (1−p) eᵗ)
    fn mgf(&self, t: &Ex) -> Option<Ex> {
        let ctx = self.context();
        Some(&self.p * t.exp() / (ctx.one() - (ctx.one() - &self.p) * t.exp()))
    }

    // Inverse transform in closed form: `⌊ln U / ln(1−p)⌋` counts the
    // failures before the first success (`P(≥ j) = (1−p)ʲ`), so the trial
    // count is one more.  `U ∈ (0, 1]`; `p = 1` gives `ln 0 = −∞` in the
    // denominator and the constant 1.
    fn sampler(&self) -> Option<Result<Sampler, SymplexError>> {
        Some(
            sampler_probability(&self.p, "the success probability").map(|p| -> Sampler {
                let ln_q = (-p).ln_1p();
                Box::new(move |rng: &mut Rng| {
                    (sample::positive_uniform(rng).ln() / ln_q).floor() + 1.0
                })
            }),
        )
    }
}

/// `NegativeBinomial(r, p)`: the number of failures before the `r`-th
/// success, `P(X = k) = C(k+r−1, k) pʳ (1−p)ᵏ` on `0..∞` (SymPy's
/// convention).
#[derive(Clone, Debug, PartialEq)]
pub struct NegativeBinomial {
    /// Number of successes `r > 0`.
    pub r: Ex,
    /// Success probability `p ∈ (0, 1)`.
    pub p: Ex,
}

impl Family for NegativeBinomial {
    family_boilerplate!(NegativeBinomial, "NegativeBinomial", [r, p]);

    fn support(&self) -> Support {
        let ctx = self.context();
        Support::integers(&ctx, Some(ctx.zero()), None)
    }

    fn density(&self, k: &Ex) -> Ex {
        let ctx = self.context();
        (k + &self.r - ctx.one()).binomial(k) * self.p.pow(&self.r) * (ctx.one() - &self.p).pow(k)
    }

    // r(1−p)/p
    fn mean(&self) -> Option<Ex> {
        Some((&self.r * (self.context().one() - &self.p) / &self.p).simplify())
    }

    // r(1−p)/p²
    fn variance(&self) -> Option<Ex> {
        Some((&self.r * (self.context().one() - &self.p) / self.p.powi(2)).simplify())
    }

    fn raw_moment(&self, n: u32) -> Option<Ex> {
        moment_from_mgf(|t| Family::mgf(self, t), n, &self.context())
    }

    // (p / (1 − (1−p) eᵗ))ʳ
    fn mgf(&self, t: &Ex) -> Option<Ex> {
        let ctx = self.context();
        Some((&self.p / (ctx.one() - (ctx.one() - &self.p) * t.exp())).pow(&self.r))
    }

    // P(X > k) = I_{1−p}(⌊k⌋ + 1, r) (from P(X ≤ k) = I_p(r, k + 1)), for
    // any real r > 0.  The family has no closed CDF on purpose (see
    // `quantile_f64`'s lattice walk); this measures an upper tail that
    // the generic route left as an unevaluable infinite sum.
    fn sf(&self, k: &Ex) -> Option<Ex> {
        let ctx = self.context();
        Some(
            (ctx.one() - &self.p)
                .betainc_regularized(&(k.floor() + ctx.one()), &self.r, &ctx.zero())
                .eval(),
        )
    }

    // Poisson–Gamma mixture: X | Λ ~ Poisson(Λ) with Λ ~ Gamma(r, (1−p)/p)
    // is NegativeBinomial(r, p) (mean r(1−p)/p, variance r(1−p)/p²), for
    // any real r > 0.
    fn sampler(&self) -> Option<Result<Sampler, SymplexError>> {
        Some(self.build_sampler())
    }
}

impl NegativeBinomial {
    fn build_sampler(&self) -> Result<Sampler, SymplexError> {
        let r = sampler_positive(&self.r, "the number of successes")?;
        let p = sampler_probability(&self.p, "the success probability")?;
        let scale = (1.0 - p) / p;
        Ok(Box::new(move |rng: &mut Rng| {
            let lambda = scale * sample::standard_gamma(rng, r);
            sample::poisson(rng, lambda)
        }))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Hypergeometric, DiscreteUniform
// ═══════════════════════════════════════════════════════════════════════════

/// `Hypergeometric(N, m, n)`: the number of successes in `n` draws without
/// replacement from a population of `N` containing `m` successes,
/// `P(X = k) = C(m, k) C(N−m, n−k) / C(N, n)` on `max(0, n+m−N) ..= min(n, m)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Hypergeometric {
    /// Population size `N`.
    pub population: Ex,
    /// Number of successes `m ≤ N` in the population.
    pub successes: Ex,
    /// Number of draws `n ≤ N`.
    pub draws: Ex,
}

impl Family for Hypergeometric {
    family_boilerplate!(
        Hypergeometric,
        "Hypergeometric",
        [population, successes, draws]
    );

    // max(0, n+m−N) ..= min(n, m); numeric parameters fold the max/min.
    fn support(&self) -> Support {
        let ctx = self.context();
        let zero = ctx.zero();
        Support::integers(
            &ctx,
            Some(
                zero.max_with(&(&self.draws + &self.successes - &self.population))
                    .simplify(),
            ),
            Some(self.draws.min_with(&self.successes).simplify()),
        )
    }

    fn density(&self, k: &Ex) -> Ex {
        self.successes.binomial(k)
            * (&self.population - &self.successes).binomial(&(&self.draws - k))
            / self.population.binomial(&self.draws)
    }

    // nm/N
    fn mean(&self) -> Option<Ex> {
        Some((&self.draws * &self.successes / &self.population).simplify())
    }

    // n (m/N) ((N−m)/N) ((N−n)/(N−1))
    fn variance(&self) -> Option<Ex> {
        let (n, m, big_n) = (&self.draws, &self.successes, &self.population);
        Some(
            (n * (m / big_n)
                * ((big_n - m) / big_n)
                * ((big_n - n) / (big_n - self.context().one())))
            .simplify(),
        )
    }

    // Factorial moments E[X^{(k)}] = n^{(k)} m^{(k)} / N^{(k)}.
    fn raw_moment(&self, n: u32) -> Option<Ex> {
        raw_moment_from_factorial_moments(n, &self.context(), |k| {
            falling_factorial(&self.draws, k) * falling_factorial(&self.successes, k)
                / falling_factorial(&self.population, k)
        })
    }
}

/// `DiscreteUniform(a, b)`: `P(X = k) = 1/(b−a+1)` on the integers `a..=b`.
/// `Die(s)` is `DiscreteUniform(1, s)`.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscreteUniform {
    /// Lowest value `a`.
    pub a: Ex,
    /// Highest value `b ≥ a`.
    pub b: Ex,
}

impl Family for DiscreteUniform {
    family_boilerplate!(DiscreteUniform, "DiscreteUniform", [a, b]);

    fn support(&self) -> Support {
        Support::integers(&self.context(), Some(self.a.clone()), Some(self.b.clone()))
    }

    fn density(&self, _k: &Ex) -> Ex {
        let ctx = self.context();
        ctx.one() / (&self.b - &self.a + ctx.one())
    }

    // (a+b)/2
    fn mean(&self) -> Option<Ex> {
        Some(((&self.a + &self.b) / self.context().int(2)).simplify())
    }

    // ((b−a+1)² − 1)/12
    fn variance(&self) -> Option<Ex> {
        let ctx = self.context();
        Some((((&self.b - &self.a + ctx.one()).powi(2) - ctx.one()) / ctx.int(12)).simplify())
    }

    // Σ_{k=a}^{b} kⁿ/(b−a+1) is a Faulhaber sum the generic summation closes.

    // (⌊k⌋ − a + 1)/(b − a + 1)
    fn cdf(&self, k: &Ex) -> Option<Ex> {
        let ctx = self.context();
        Some((k.floor() - &self.a + ctx.one()) / (&self.b - &self.a + ctx.one()))
    }

    // (e^{at} − e^{(b+1)t}) / ((b−a+1)(1 − eᵗ))
    fn mgf(&self, t: &Ex) -> Option<Ex> {
        let ctx = self.context();
        Some(
            ((&self.a * t).exp() - ((&self.b + ctx.one()) * t).exp())
                / ((&self.b - &self.a + ctx.one()) * (ctx.one() - t.exp())),
        )
    }

    // Smallest k in a..=b with (k − a + 1)/(b − a + 1) ≥ p: a + ⌈p (b − a + 1)⌉ − 1.
    fn quantile(&self, p: &Ex) -> Option<Ex> {
        let ctx = self.context();
        Some((&self.a + (p * (&self.b - &self.a + ctx.one())).ceiling() - ctx.one()).simplify())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Constructors
// ═══════════════════════════════════════════════════════════════════════════

impl Distribution {
    /// A finite distribution from an explicit `(value, probability)` table
    /// in `ctx`.  SymPy: `FiniteRV('X', {v: p, …})`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the table is empty, a numeric
    /// probability is negative, two numeric values coincide, or all
    /// probabilities are numeric and do not sum to exactly `1`.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::{Distribution, RandomVariable};
    ///
    /// let ctx = Context::new();
    /// // A loaded coin: heads (1) with probability 2/3.
    /// let coin = Distribution::try_finite(&ctx, vec![
    ///     (ctx.int(1), ctx.rational(2, 3)),
    ///     (ctx.int(0), ctx.rational(1, 3)),
    /// ])?;
    /// let c = RandomVariable::new(&ctx, "C", coin);
    /// assert_eq!(c.mean(), ctx.rational(2, 3));
    /// assert_eq!(c.variance(), ctx.rational(2, 9));
    /// assert_eq!(c.probability(&c.symbol().eq_expr(&ctx.int(1)))?, ctx.rational(2, 3));
    /// # Ok::<(), SymplexError>(())
    /// ```
    pub fn try_finite(ctx: &Context, table: Vec<(Ex, Ex)>) -> Result<Distribution, SymplexError> {
        if table.is_empty() {
            return Err(invalid("a finite distribution needs at least one value"));
        }
        let mut total = Q::zero();
        let mut all_numeric = true;
        for (v, p) in &table {
            match p.eval().as_rational() {
                Some(q) => {
                    if q < Q::zero() {
                        return Err(invalid(format!("probability of `{v}` is negative: `{p}`")));
                    }
                    total += q;
                }
                None => all_numeric = false,
            }
        }
        if all_numeric && total != Q::one() {
            return Err(invalid(format!("the probabilities sum to {total}, not 1")));
        }
        for (i, (v, _)) in table.iter().enumerate() {
            for (w, _) in &table[i + 1..] {
                if v.equals(w) == Some(true) {
                    return Err(invalid(format!("the value `{v}` is listed twice")));
                }
            }
        }
        Ok(Distribution::finite(ctx, table))
    }

    /// A finite distribution from a table without validation (see
    /// [`try_finite`](Self::try_finite)).
    pub fn finite(ctx: &Context, table: Vec<(Ex, Ex)>) -> Distribution {
        Distribution::from_family(Finite::new(ctx, table))
    }

    /// `Bernoulli(p)`.  SymPy: `Bernoulli('X', p)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `p` is a number outside `[0, 1]`.
    pub fn try_bernoulli(p: Ex) -> Result<Distribution, SymplexError> {
        require_probability(&p, "the success probability")?;
        Ok(Distribution::bernoulli(p))
    }

    /// `Bernoulli(p)` without parameter validation (see
    /// [`try_bernoulli`](Self::try_bernoulli)).
    pub fn bernoulli(p: Ex) -> Distribution {
        Distribution::from_family(Bernoulli { p })
    }

    /// `Binomial(n, p)`.  SymPy: `Binomial('X', n, p)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `n` is a number that is not a
    /// non-negative integer or `p` a number outside `[0, 1]`.
    pub fn try_binomial(n: Ex, p: Ex) -> Result<Distribution, SymplexError> {
        require_count(&n, "the number of trials")?;
        require_probability(&p, "the success probability")?;
        Ok(Distribution::binomial(n, p))
    }

    /// `Binomial(n, p)` without parameter validation (see
    /// [`try_binomial`](Self::try_binomial)).
    pub fn binomial(n: Ex, p: Ex) -> Distribution {
        Distribution::from_family(Binomial { n, p })
    }

    /// `Poisson(λ)` with rate `rate`.  SymPy: `Poisson('X', lamda)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `rate` is a number `≤ 0`.
    pub fn try_poisson(rate: Ex) -> Result<Distribution, SymplexError> {
        super::continuous::require_positive(&rate, "the rate")?;
        Ok(Distribution::poisson(rate))
    }

    /// `Poisson(λ)` without parameter validation (see
    /// [`try_poisson`](Self::try_poisson)).
    pub fn poisson(rate: Ex) -> Distribution {
        Distribution::from_family(Poisson { rate })
    }

    /// `Geometric(p)` on `1..∞` (trials up to the first success).  SymPy:
    /// `Geometric('X', p)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `p` is a number outside `(0, 1]`.
    pub fn try_geometric(p: Ex) -> Result<Distribution, SymplexError> {
        require_probability_positive(&p, "the success probability", true)?;
        Ok(Distribution::geometric(p))
    }

    /// `Geometric(p)` without parameter validation (see
    /// [`try_geometric`](Self::try_geometric)).
    pub fn geometric(p: Ex) -> Distribution {
        Distribution::from_family(Geometric { p })
    }

    /// `NegativeBinomial(r, p)` on `0..∞` (failures before the `r`-th
    /// success).  SymPy: `NegativeBinomial('X', r, p)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `r` is a number `≤ 0` or `p` a
    /// number outside `(0, 1)`.
    pub fn try_negative_binomial(r: Ex, p: Ex) -> Result<Distribution, SymplexError> {
        super::continuous::require_positive(&r, "the number of successes")?;
        require_probability_positive(&p, "the success probability", false)?;
        Ok(Distribution::negative_binomial(r, p))
    }

    /// `NegativeBinomial(r, p)` without parameter validation (see
    /// [`try_negative_binomial`](Self::try_negative_binomial)).
    pub fn negative_binomial(r: Ex, p: Ex) -> Distribution {
        Distribution::from_family(NegativeBinomial { r, p })
    }

    /// `Hypergeometric(N, m, n)`: `draws` without replacement from a
    /// population of `population` with `successes` successes.  SymPy:
    /// `Hypergeometric('X', N, m, n)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if a numeric parameter is not a
    /// non-negative integer, or `successes` / `draws` numerically exceed
    /// `population`.
    pub fn try_hypergeometric(
        population: Ex,
        successes: Ex,
        draws: Ex,
    ) -> Result<Distribution, SymplexError> {
        require_count(&population, "the population size")?;
        require_count(&successes, "the number of successes")?;
        require_count(&draws, "the number of draws")?;
        require_le(
            &successes,
            &population,
            "the number of successes must not exceed the population",
        )?;
        require_le(
            &draws,
            &population,
            "the number of draws must not exceed the population",
        )?;
        Ok(Distribution::hypergeometric(population, successes, draws))
    }

    /// `Hypergeometric(N, m, n)` without parameter validation (see
    /// [`try_hypergeometric`](Self::try_hypergeometric)).
    pub fn hypergeometric(population: Ex, successes: Ex, draws: Ex) -> Distribution {
        Distribution::from_family(Hypergeometric {
            population,
            successes,
            draws,
        })
    }

    /// `DiscreteUniform(a, b)` on the integers `a..=b`.  SymPy:
    /// `DiscreteUniform('X', range(a, b + 1))`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `a` or `b` is a non-integer
    /// number, or both are numbers with `a > b`.
    pub fn try_discrete_uniform(a: Ex, b: Ex) -> Result<Distribution, SymplexError> {
        require_integer(&a, "the lowest value")?;
        require_integer(&b, "the highest value")?;
        require_le(&a, &b, "the lowest value must not exceed the highest")?;
        Ok(Distribution::discrete_uniform(a, b))
    }

    /// `DiscreteUniform(a, b)` without parameter validation (see
    /// [`try_discrete_uniform`](Self::try_discrete_uniform)).
    pub fn discrete_uniform(a: Ex, b: Ex) -> Distribution {
        Distribution::from_family(DiscreteUniform { a, b })
    }

    /// A fair die with `sides` faces: `DiscreteUniform(1, sides)`.  SymPy:
    /// `Die('X', sides)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `sides` is a number that is not
    /// a positive integer.
    pub fn try_die(sides: Ex) -> Result<Distribution, SymplexError> {
        require_integer(&sides, "the number of sides")?;
        super::continuous::require_positive(&sides, "the number of sides")?;
        Ok(Distribution::die(sides))
    }

    /// A fair die with `sides` faces without parameter validation (see
    /// [`try_die`](Self::try_die)).
    pub fn die(sides: Ex) -> Distribution {
        let one = sides.context().one();
        Distribution::from_family(DiscreteUniform { a: one, b: sides })
    }
}
