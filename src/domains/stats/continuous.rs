//! Continuous distribution families.  Each is a struct with its parameters
//! as expressions and an [`impl Family`](Family) holding its support,
//! density and whatever closed forms it has — every formula a textbook
//! identity, cited at the arm.  Where a closed form is absent the generic
//! machinery in [`Distribution`] integrates the density instead.
//!
//! To add a family: a struct, `impl Family` (`family_boilerplate!` for
//! `name`/`context`/`parameters`/`eq_family`, then support, density and
//! the closed forms), and a constructor pair `Distribution::try_name(…)`
//! (validates numeric parameters) / `Distribution::name(…)` (unchecked).
//!
//! Sampling: a family with a closed-form quantile leaves [`Family::sampler`]
//! unset and is drawn by inverse transform ([`Distribution::sampler`]);
//! `Gamma`, `ChiSquared`, `Beta`, `StudentT` and `FDistribution` have no
//! elementary inverse and supply their own exact routes (Marsaglia–Tsang
//! gamma variates and the classical representations through them).

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::errors::SymplexError;

use std::cmp::Ordering;

use super::family::{Distribution, Family, Sampler, family_boilerplate, sign_of};
use super::sample::{self, Rng};
use super::support::Support;

fn invalid(reason: impl Into<String>) -> SymplexError {
    SymplexError::invalid_argument("stats", reason)
}

/// Reject a numeric parameter that is not positive; accept symbolic ones.
pub(crate) fn require_positive(e: &Ex, what: &str) -> Result<(), SymplexError> {
    match e.is_positive() {
        Some(false) => Err(invalid(format!("{what} must be positive, got `{e}`"))),
        _ => Ok(()),
    }
}

/// A parameter as the positive finite `f64` a sampler needs, evaluated
/// once when the sampler is built.
///
/// # Errors
///
/// The evaluation error ([`SymplexError::Unevaluable`], …) for a symbolic
/// parameter; [`SymplexError::InvalidArgument`] for a numeric one that is
/// not positive (the unchecked constructors do not validate).
pub(crate) fn sampler_positive(e: &Ex, what: &str) -> Result<f64, SymplexError> {
    let v = e.eval_f64()?;
    if !(v > 0.0 && v.is_finite()) {
        return Err(invalid(format!(
            "{what} must be a positive number to sample, got `{e}`"
        )));
    }
    Ok(v)
}

/// `θ · Gamma(k, 1)`: the sampler behind `Gamma` and `ChiSquared`.
fn gamma_sampler(shape: &Ex, scale: &Ex) -> Result<Sampler, SymplexError> {
    let k = sampler_positive(shape, "the shape")?;
    let theta = sampler_positive(scale, "the scale")?;
    Ok(Box::new(move |rng: &mut Rng| {
        theta * sample::standard_gamma(rng, k)
    }))
}

/// Reject a numeric parameter that is negative; accept symbolic ones.
fn require_nonnegative(e: &Ex, what: &str) -> Result<(), SymplexError> {
    match e.is_negative() {
        Some(true) => Err(invalid(format!("{what} must be non-negative, got `{e}`"))),
        _ => Ok(()),
    }
}

/// `1 − e^{−y}` for `y ≥ 0`, written `2 e^{−y/2} sinh(y/2)` so that it keeps
/// its relative accuracy for a tiny `y` without an `expm1` node: `sinh` of
/// a tiny argument evaluates to full relative precision, while
/// `1 − exp(−y)` is a difference of two numbers next to `1` that `evalf`
/// returns as `0` once its precision budget (about 256 bits beyond the
/// digits asked for) is spent — `1 − e^{−√2·10⁻¹⁰⁰}` was `0`.  (scipy writes
/// `-expm1(-y)` for the exponential and Weibull CDFs.)  `eval` and
/// `simplify` keep the product as written.
pub(crate) fn one_minus_exp_neg(y: &Ex) -> Ex {
    let ctx = y.context();
    let half = y / ctx.int(2);
    ctx.int(2) * (-&half).exp() * half.sinh()
}

/// `ln(x/xₘ)` as `2 atanh((x − xₘ)/(x + xₘ))`: the difference `x − xₘ` is
/// formed symbolically (exactly for `x = xₘ + 10⁻¹⁰⁰`), and `atanh` of a tiny
/// argument keeps its relative accuracy, where `ln` of a number next to
/// `1` evaluated to `0` (`ln(1 + 10⁻¹⁰⁰)`).
fn ln_ratio(x: &Ex, x_m: &Ex) -> Ex {
    x.context().int(2) * ((x - x_m) / (x + x_m)).atanh()
}

/// Does a moment that exists only for `param > bound` exist?  `false` only
/// when `param − bound` is *known* to be non-positive (a numeric parameter
/// that fails the condition); a symbolic parameter is the caller's promise
/// and the closed form is returned.
fn exceeds(param: &Ex, bound: u32) -> bool {
    (param - i64::from(bound)).is_positive() != Some(false)
}

/// `E[Xⁿ]` for a distribution symmetric about `mean` whose even central
/// moments are `central(2j)` (the odd ones vanish): the binomial expansion
/// `E[(μ + Y)ⁿ] = Σ_j C(n, 2j) μⁿ⁻²ʲ E[Y²ʲ]`.
fn raw_from_even_central(mean: &Ex, n: u32, ctx: &Context, central: impl Fn(u32) -> Ex) -> Ex {
    let mut acc = ctx.zero();
    for j in 0..=n / 2 {
        let binom = ctx.int(i64::from(n)).binomial(&ctx.int(i64::from(2 * j)));
        let c = if j == 0 { ctx.one() } else { central(2 * j) };
        acc += binom * mean.powi(i64::from(n - 2 * j)) * c;
    }
    acc.simplify()
}

// ═══════════════════════════════════════════════════════════════════════════
// Normal
// ═══════════════════════════════════════════════════════════════════════════

/// `Normal(μ, σ)`: density `e^{−(x−μ)²/(2σ²)} / (σ√(2π))` on ℝ.
#[derive(Clone, Debug, PartialEq)]
pub struct Normal {
    /// Mean `μ`.
    pub mean: Ex,
    /// Standard deviation `σ > 0`.
    pub std: Ex,
}

impl Family for Normal {
    family_boilerplate!(Normal, "Normal", [mean, std]);

    fn support(&self) -> Support {
        Support::reals(&self.context())
    }

    fn density(&self, x: &Ex) -> Ex {
        let ctx = self.context();
        let two = ctx.int(2);
        let z = (x - &self.mean) / &self.std;
        (-(z.powi(2)) / &two).exp() / (&self.std * (&two * ctx.pi()).sqrt())
    }

    fn mean(&self) -> Option<Ex> {
        Some(self.mean.clone())
    }

    fn variance(&self) -> Option<Ex> {
        Some(self.std.powi(2))
    }

    // E[Xⁿ] = Σ_{k=0}^{⌊n/2⌋} C(n, 2k) (2k−1)!! μⁿ⁻²ᵏ σ²ᵏ
    fn raw_moment(&self, n: u32) -> Option<Ex> {
        let ctx = self.context();
        let mut acc = ctx.zero();
        for k in 0..=n / 2 {
            let binom = ctx.int(i64::from(n)).binomial(&ctx.int(i64::from(2 * k)));
            // (2k − 1)!! = (2k)! / (2ᵏ k!)
            let dfact = ctx.int(i64::from(2 * k)).factorial()
                / (ctx.int(2).powi(i64::from(k)) * ctx.int(i64::from(k)).factorial());
            acc += binom
                * dfact
                * self.mean.powi(i64::from(n - 2 * k))
                * self.std.powi(i64::from(2 * k));
        }
        Some(acc.simplify())
    }

    // Φ((x−μ)/σ) = ½ + ½ erf((x−μ)/(σ√2))
    fn cdf(&self, x: &Ex) -> Option<Ex> {
        let ctx = self.context();
        let half = ctx.rational(1, 2);
        let arg = (x - &self.mean) / (&self.std * ctx.int(2).sqrt());
        Some(&half + &half * arg.erf())
    }

    // ½ erfc((μ−x)/(σ√2))
    fn cdf_lower(&self, x: &Ex) -> Option<Ex> {
        let ctx = self.context();
        let arg = (&self.mean - x) / (&self.std * ctx.int(2).sqrt());
        Some(ctx.rational(1, 2) * arg.erfc())
    }

    // ½ erfc((x−μ)/(σ√2))
    fn sf(&self, x: &Ex) -> Option<Ex> {
        let ctx = self.context();
        let arg = (x - &self.mean) / (&self.std * ctx.int(2).sqrt());
        Some(ctx.rational(1, 2) * arg.erfc())
    }

    // exp(μt + σ²t²/2)
    fn mgf(&self, t: &Ex) -> Option<Ex> {
        let ctx = self.context();
        Some((&self.mean * t + self.std.powi(2) * t.powi(2) / ctx.int(2)).exp())
    }

    // μ + σ√2 · erfinv(2p − 1)
    fn quantile(&self, p: &Ex) -> Option<Ex> {
        let ctx = self.context();
        Some(&self.mean + &self.std * ctx.int(2).sqrt() * (ctx.int(2) * p - 1).erfinv())
    }

    // ½ ln(2πeσ²)
    fn entropy(&self) -> Option<Ex> {
        let ctx = self.context();
        Some(ctx.rational(1, 2) * (ctx.int(2) * ctx.pi() * ctx.e() * self.std.powi(2)).ln())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Uniform
// ═══════════════════════════════════════════════════════════════════════════

/// `Uniform(a, b)`: density `1/(b − a)` on `[a, b]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Uniform {
    /// Lower end `a`.
    pub lo: Ex,
    /// Upper end `b > a`.
    pub hi: Ex,
}

impl Family for Uniform {
    family_boilerplate!(Uniform, "Uniform", [lo, hi]);

    fn support(&self) -> Support {
        Support::interval(self.lo.clone(), self.hi.clone())
    }

    fn density(&self, _x: &Ex) -> Ex {
        self.context().one() / (&self.hi - &self.lo)
    }

    fn mean(&self) -> Option<Ex> {
        Some(((&self.lo + &self.hi) / self.context().int(2)).simplify())
    }

    // (b − a)² / 12
    fn variance(&self) -> Option<Ex> {
        Some(((&self.hi - &self.lo).powi(2) / self.context().int(12)).simplify())
    }

    // (bⁿ⁺¹ − aⁿ⁺¹) / ((n+1)(b − a))
    fn raw_moment(&self, n: u32) -> Option<Ex> {
        let n1 = i64::from(n) + 1;
        Some(
            ((self.hi.powi(n1) - self.lo.powi(n1))
                / (self.context().int(n1) * (&self.hi - &self.lo)))
                .simplify(),
        )
    }

    fn cdf(&self, x: &Ex) -> Option<Ex> {
        Some((x - &self.lo) / (&self.hi - &self.lo))
    }

    fn sf(&self, x: &Ex) -> Option<Ex> {
        Some((&self.hi - x) / (&self.hi - &self.lo))
    }

    // (e^{bt} − e^{at}) / ((b−a) t)
    fn mgf(&self, t: &Ex) -> Option<Ex> {
        Some(((&self.hi * t).exp() - (&self.lo * t).exp()) / ((&self.hi - &self.lo) * t))
    }

    // a + p(b − a)
    fn quantile(&self, p: &Ex) -> Option<Ex> {
        Some(&self.lo + p * (&self.hi - &self.lo))
    }

    // ln(b − a)
    fn entropy(&self) -> Option<Ex> {
        Some((&self.hi - &self.lo).ln())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Exponential
// ═══════════════════════════════════════════════════════════════════════════

/// `Exponential(λ)`: density `λ e^{−λx}` on `[0, ∞)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Exponential {
    /// Rate `λ > 0` (the mean is `1/λ`).
    pub rate: Ex,
}

impl Family for Exponential {
    family_boilerplate!(Exponential, "Exponential", [rate]);

    fn support(&self) -> Support {
        Support::half_line(self.context().zero())
    }

    fn density(&self, x: &Ex) -> Ex {
        &self.rate * (-(&self.rate * x)).exp()
    }

    fn mean(&self) -> Option<Ex> {
        Some(self.context().one() / &self.rate)
    }

    fn variance(&self) -> Option<Ex> {
        Some(self.context().one() / self.rate.powi(2))
    }

    // n! / λⁿ
    fn raw_moment(&self, n: u32) -> Option<Ex> {
        let ctx = self.context();
        Some((ctx.int(i64::from(n)).factorial() / self.rate.powi(i64::from(n))).simplify())
    }

    // 1 − e^{−λx}
    fn cdf(&self, x: &Ex) -> Option<Ex> {
        Some(self.context().one() - (-(&self.rate * x)).exp())
    }

    // 1 − e^{−λx} as 2 e^{−λx/2} sinh(λx/2) (`one_minus_exp_neg`).  0.28
    // wrote γ(1, λx), which relied on `eval` not folding it into
    // 1 − e^{−λx}: it does fold at an irrational λx, and
    // Exponential(√2).cdf(10⁻¹⁰⁰) was 0.
    fn cdf_lower(&self, x: &Ex) -> Option<Ex> {
        Some(one_minus_exp_neg(&(&self.rate * x)))
    }

    // e^{−λx}
    fn sf(&self, x: &Ex) -> Option<Ex> {
        Some((-(&self.rate * x)).exp())
    }

    // λ / (λ − t)
    fn mgf(&self, t: &Ex) -> Option<Ex> {
        Some(&self.rate / (&self.rate - t))
    }

    // −ln(1 − p) / λ
    fn quantile(&self, p: &Ex) -> Option<Ex> {
        Some(-(self.context().one() - p).ln() / &self.rate)
    }

    // 1 − ln λ
    fn entropy(&self) -> Option<Ex> {
        Some(self.context().one() - self.rate.ln())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Gamma and ChiSquared
// ═══════════════════════════════════════════════════════════════════════════

/// `Gamma(k, θ)`: density `x^{k−1} e^{−x/θ} / (Γ(k) θᵏ)` on `[0, ∞)`
/// (shape–scale parametrisation, SymPy's `Gamma(k, theta)`).
#[derive(Clone, Debug, PartialEq)]
pub struct Gamma {
    /// Shape `k > 0`.
    pub shape: Ex,
    /// Scale `θ > 0` (the rate is `1/θ`).
    pub scale: Ex,
}

impl Family for Gamma {
    family_boilerplate!(Gamma, "Gamma", [shape, scale]);

    fn support(&self) -> Support {
        Support::half_line(self.context().zero())
    }

    fn density(&self, x: &Ex) -> Ex {
        x.pow(&(&self.shape - 1)) * (-(x / &self.scale)).exp()
            / (self.shape.gamma() * self.scale.pow(&self.shape))
    }

    fn mean(&self) -> Option<Ex> {
        Some((&self.shape * &self.scale).simplify())
    }

    fn variance(&self) -> Option<Ex> {
        Some((&self.shape * self.scale.powi(2)).simplify())
    }

    // θⁿ Γ(k+n)/Γ(k) = θⁿ (k)ₙ
    fn raw_moment(&self, n: u32) -> Option<Ex> {
        let n_ex = self.context().int(i64::from(n));
        Some((self.scale.powi(i64::from(n)) * self.shape.rising_factorial(&n_ex)).simplify())
    }

    // γ(k, x/θ) / Γ(k); `eval` closes the incomplete gamma for integer and
    // half-integer `k` (except where the closed form would cancel, which
    // it recognises for a rational x/θ only: at an irrational one the
    // lower tail folds into a cancelling difference, Gamma(5, 1).cdf(√2·10⁻³⁰)
    // evaluates to 0, and no elementary form avoids it).
    fn cdf(&self, x: &Ex) -> Option<Ex> {
        Some(((x / &self.scale).lowergamma(&self.shape) / self.shape.gamma()).eval())
    }

    // Γ(k, x/θ) / Γ(k): closed (a positive sum) for integer and
    // half-integer `k`.
    fn sf(&self, x: &Ex) -> Option<Ex> {
        Some(((x / &self.scale).uppergamma(&self.shape) / self.shape.gamma()).eval())
    }

    // (1 − θt)^{−k}
    fn mgf(&self, t: &Ex) -> Option<Ex> {
        Some((self.context().one() - &self.scale * t).pow(&(-&self.shape)))
    }

    // k + ln θ + ln Γ(k) + (1 − k) ψ(k)
    fn entropy(&self) -> Option<Ex> {
        let ctx = self.context();
        Some(
            &self.shape
                + self.scale.ln()
                + self.shape.gamma().ln()
                + (ctx.one() - &self.shape) * self.shape.digamma(),
        )
    }

    // θ · Gamma(k, 1) by Marsaglia–Tsang (see `sample::standard_gamma`).
    fn sampler(&self) -> Option<Result<Sampler, SymplexError>> {
        Some(gamma_sampler(&self.shape, &self.scale))
    }
}

/// `ChiSquared(k)` = `Gamma(k/2, 2)`: density
/// `x^{k/2−1} e^{−x/2} / (2^{k/2} Γ(k/2))` on `[0, ∞)`.
#[derive(Clone, Debug, PartialEq)]
pub struct ChiSquared {
    /// Degrees of freedom `k > 0`.
    pub dof: Ex,
}

impl ChiSquared {
    fn as_gamma(&self) -> Gamma {
        let ctx = self.context();
        Gamma {
            shape: &self.dof / ctx.int(2),
            scale: ctx.int(2),
        }
    }
}

impl Family for ChiSquared {
    family_boilerplate!(ChiSquared, "ChiSquared", [dof]);

    fn support(&self) -> Support {
        Support::half_line(self.context().zero())
    }

    fn density(&self, x: &Ex) -> Ex {
        self.as_gamma().density(x)
    }

    fn mean(&self) -> Option<Ex> {
        Some(self.dof.clone())
    }

    fn variance(&self) -> Option<Ex> {
        Some((self.context().int(2) * &self.dof).simplify())
    }

    fn raw_moment(&self, n: u32) -> Option<Ex> {
        self.as_gamma().raw_moment(n)
    }

    fn cdf(&self, x: &Ex) -> Option<Ex> {
        self.as_gamma().cdf(x)
    }

    fn sf(&self, x: &Ex) -> Option<Ex> {
        self.as_gamma().sf(x)
    }

    fn mgf(&self, t: &Ex) -> Option<Ex> {
        self.as_gamma().mgf(t)
    }

    fn entropy(&self) -> Option<Ex> {
        self.as_gamma().entropy()
    }

    // χ²(k) = Gamma(k/2, 2).
    fn sampler(&self) -> Option<Result<Sampler, SymplexError>> {
        self.as_gamma().sampler()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Beta
// ═══════════════════════════════════════════════════════════════════════════

/// `Beta(α, β)`: density `x^{α−1} (1−x)^{β−1} / B(α, β)` on `[0, 1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Beta {
    /// First shape `α > 0`.
    pub alpha: Ex,
    /// Second shape `β > 0`.
    pub beta: Ex,
}

impl Family for Beta {
    family_boilerplate!(Beta, "Beta", [alpha, beta]);

    fn support(&self) -> Support {
        let ctx = self.context();
        Support::interval(ctx.zero(), ctx.one())
    }

    fn density(&self, x: &Ex) -> Ex {
        let ctx = self.context();
        x.pow(&(&self.alpha - 1)) * (ctx.one() - x).pow(&(&self.beta - 1))
            / self.alpha.beta(&self.beta)
    }

    fn mean(&self) -> Option<Ex> {
        Some((&self.alpha / (&self.alpha + &self.beta)).simplify())
    }

    // αβ / ((α+β)² (α+β+1))
    fn variance(&self) -> Option<Ex> {
        let s = &self.alpha + &self.beta;
        Some((&self.alpha * &self.beta / (s.powi(2) * (&s + 1))).simplify())
    }

    // (α)ₙ / (α+β)ₙ
    fn raw_moment(&self, n: u32) -> Option<Ex> {
        let n_ex = self.context().int(i64::from(n));
        Some(
            (self.alpha.rising_factorial(&n_ex)
                / (&self.alpha + &self.beta).rising_factorial(&n_ex))
            .simplify(),
        )
    }

    // The regularised incomplete beta I_x(α, β) — `eval` closes it to a
    // polynomial for integer α, β.
    fn cdf(&self, x: &Ex) -> Option<Ex> {
        let ctx = self.context();
        Some(
            x.betainc_regularized(&self.alpha, &self.beta, &ctx.zero())
                .eval(),
        )
    }

    // I_{1−x}(β, α)
    fn sf(&self, x: &Ex) -> Option<Ex> {
        let ctx = self.context();
        Some(
            (ctx.one() - x)
                .betainc_regularized(&self.beta, &self.alpha, &ctx.zero())
                .eval(),
        )
    }

    // ln B(α, β) − (α−1) ψ(α) − (β−1) ψ(β) + (α+β−2) ψ(α+β)
    fn entropy(&self) -> Option<Ex> {
        let s = &self.alpha + &self.beta;
        Some(
            self.alpha.beta(&self.beta).ln()
                - (&self.alpha - 1) * self.alpha.digamma()
                - (&self.beta - 1) * self.beta.digamma()
                + (&s - 2) * s.digamma(),
        )
    }

    // X/(X+Y) for independent X ~ Gamma(α, 1), Y ~ Gamma(β, 1).
    fn sampler(&self) -> Option<Result<Sampler, SymplexError>> {
        Some(self.build_sampler())
    }
}

impl Beta {
    fn build_sampler(&self) -> Result<Sampler, SymplexError> {
        let a = sampler_positive(&self.alpha, "α")?;
        let b = sampler_positive(&self.beta, "β")?;
        Ok(Box::new(move |rng: &mut Rng| {
            // Both gammas underflow to 0 only for tiny shapes; redraw rather
            // than return 0/0.
            loop {
                let x = sample::standard_gamma(rng, a);
                let y = sample::standard_gamma(rng, b);
                if x + y > 0.0 {
                    return x / (x + y);
                }
            }
        }))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cauchy, Laplace, Logistic
// ═══════════════════════════════════════════════════════════════════════════

/// `Cauchy(x₀, γ)`: density `1 / (πγ (1 + ((x−x₀)/γ)²))` on ℝ.  No
/// moments of any order exist.
#[derive(Clone, Debug, PartialEq)]
pub struct Cauchy {
    /// Location `x₀` (the median).
    pub location: Ex,
    /// Scale `γ > 0` (half the interquartile range).
    pub scale: Ex,
}

impl Family for Cauchy {
    family_boilerplate!(Cauchy, "Cauchy", [location, scale]);

    fn support(&self) -> Support {
        Support::reals(&self.context())
    }

    fn density(&self, x: &Ex) -> Ex {
        let ctx = self.context();
        ctx.one()
            / (ctx.pi() * &self.scale * (ctx.one() + ((x - &self.location) / &self.scale).powi(2)))
    }

    // ½ + atan((x−x₀)/γ)/π
    fn cdf(&self, x: &Ex) -> Option<Ex> {
        let ctx = self.context();
        Some(ctx.rational(1, 2) + ((x - &self.location) / &self.scale).atan() / ctx.pi())
    }

    // −atan(γ/(x−x₀))/π for x < x₀ (atan z + atan(1/z) = −π/2 for z < 0),
    // the classic form elsewhere.  The sign of x − x₀ is decided on its
    // value when the symbolic test cannot (`sign_of`).
    fn cdf_lower(&self, x: &Ex) -> Option<Ex> {
        let d = x - &self.location;
        if sign_of(&d) != Some(Ordering::Less) {
            return self.cdf(x);
        }
        Some(-(&self.scale / d).atan() / self.context().pi())
    }

    // atan(γ/(x−x₀))/π for x > x₀ (atan z + atan(1/z) = π/2 for z > 0),
    // ½ − atan((x−x₀)/γ)/π elsewhere.  0.28 decided x > x₀ symbolically
    // only, so Cauchy(√2, π).sf(10¹⁰⁰) (`10¹⁰⁰ − √2 > 0` undecided) took
    // the classic form and was 0.
    fn sf(&self, x: &Ex) -> Option<Ex> {
        let ctx = self.context();
        let d = x - &self.location;
        if sign_of(&d) == Some(Ordering::Greater) {
            return Some((&self.scale / d).atan() / ctx.pi());
        }
        Some(ctx.rational(1, 2) - (d / &self.scale).atan() / ctx.pi())
    }

    // x₀ + γ tan(π(p − ½))
    fn quantile(&self, p: &Ex) -> Option<Ex> {
        let ctx = self.context();
        Some(&self.location + &self.scale * (ctx.pi() * (p - ctx.rational(1, 2))).tan())
    }

    // ln(4πγ)
    fn entropy(&self) -> Option<Ex> {
        let ctx = self.context();
        Some((ctx.int(4) * ctx.pi() * &self.scale).ln())
    }
}

/// `Laplace(μ, b)`: density `e^{−|x−μ|/b} / (2b)` on ℝ.
#[derive(Clone, Debug, PartialEq)]
pub struct Laplace {
    /// Location `μ` (mean and median).
    pub mean: Ex,
    /// Scale `b > 0`.
    pub scale: Ex,
}

impl Family for Laplace {
    family_boilerplate!(Laplace, "Laplace", [mean, scale]);

    fn support(&self) -> Support {
        Support::reals(&self.context())
    }

    fn density(&self, x: &Ex) -> Ex {
        (-((x - &self.mean).abs() / &self.scale)).exp() / (self.context().int(2) * &self.scale)
    }

    fn mean(&self) -> Option<Ex> {
        Some(self.mean.clone())
    }

    fn variance(&self) -> Option<Ex> {
        Some((self.context().int(2) * self.scale.powi(2)).simplify())
    }

    // Even central moments E[(X−μ)ᵏ] = k! bᵏ, odd ones 0.
    fn raw_moment(&self, n: u32) -> Option<Ex> {
        let ctx = self.context();
        Some(raw_from_even_central(&self.mean, n, &ctx, |k| {
            ctx.int(i64::from(k)).factorial() * self.scale.powi(i64::from(k))
        }))
    }

    // ½ e^{(x−μ)/b} for x < μ, 1 − ½ e^{−(x−μ)/b} for x ≥ μ
    fn cdf(&self, x: &Ex) -> Option<Ex> {
        let ctx = self.context();
        let half = ctx.rational(1, 2);
        let z = (x - &self.mean) / &self.scale;
        let below = &half * z.exp();
        let above = ctx.one() - &half * (-z).exp();
        Some(Ex::piecewise(&[
            (&below, &x.lt(&self.mean)),
            (&above, &x.ge(&self.mean)),
        ]))
    }

    // 1 − ½ e^{(x−μ)/b} for x < μ, ½ e^{−(x−μ)/b} for x ≥ μ
    fn sf(&self, x: &Ex) -> Option<Ex> {
        let ctx = self.context();
        let half = ctx.rational(1, 2);
        let z = (x - &self.mean) / &self.scale;
        let below = ctx.one() - &half * z.exp();
        let above = &half * (-z).exp();
        Some(Ex::piecewise(&[
            (&below, &x.lt(&self.mean)),
            (&above, &x.ge(&self.mean)),
        ]))
    }

    // e^{μt} / (1 − b²t²)
    fn mgf(&self, t: &Ex) -> Option<Ex> {
        Some((&self.mean * t).exp() / (self.context().one() - self.scale.powi(2) * t.powi(2)))
    }

    // μ − b sign(p − ½) ln(1 − 2|p − ½|)
    fn quantile(&self, p: &Ex) -> Option<Ex> {
        let ctx = self.context();
        let d = p - ctx.rational(1, 2);
        Some(&self.mean - &self.scale * d.sign() * (ctx.one() - ctx.int(2) * d.abs()).ln())
    }

    // 1 + ln(2b)
    fn entropy(&self) -> Option<Ex> {
        let ctx = self.context();
        Some(ctx.one() + (ctx.int(2) * &self.scale).ln())
    }
}

/// `Logistic(μ, s)`: density `e^{−(x−μ)/s} / (s (1 + e^{−(x−μ)/s})²)` on ℝ.
#[derive(Clone, Debug, PartialEq)]
pub struct Logistic {
    /// Location `μ` (mean and median).
    pub mean: Ex,
    /// Scale `s > 0`.
    pub scale: Ex,
}

impl Family for Logistic {
    family_boilerplate!(Logistic, "Logistic", [mean, scale]);

    fn support(&self) -> Support {
        Support::reals(&self.context())
    }

    fn density(&self, x: &Ex) -> Ex {
        let ctx = self.context();
        let e = (-((x - &self.mean) / &self.scale)).exp();
        &e / (&self.scale * (ctx.one() + &e).powi(2))
    }

    fn mean(&self) -> Option<Ex> {
        Some(self.mean.clone())
    }

    // s²π²/3
    fn variance(&self) -> Option<Ex> {
        let ctx = self.context();
        Some((self.scale.powi(2) * ctx.pi().powi(2) / ctx.int(3)).simplify())
    }

    // Even central moments E[(X−μ)ᵏ] = (−1)^{k/2+1} (2ᵏ − 2) B_k (πs)ᵏ
    // (from the standard logistic mgf πt/sin(πt)), odd ones 0.
    fn raw_moment(&self, n: u32) -> Option<Ex> {
        let ctx = self.context();
        Some(raw_from_even_central(&self.mean, n, &ctx, |k| {
            let sign = if (k / 2) % 2 == 1 { 1 } else { -1 };
            let k_ex = ctx.int(i64::from(k));
            ctx.int(sign)
                * (ctx.int(2).powi(i64::from(k)) - 2)
                * k_ex.bernoulli_number()
                * (ctx.pi() * &self.scale).powi(i64::from(k))
        }))
    }

    // 1 / (1 + e^{−(x−μ)/s})
    fn cdf(&self, x: &Ex) -> Option<Ex> {
        let ctx = self.context();
        Some(ctx.one() / (ctx.one() + (-((x - &self.mean) / &self.scale)).exp()))
    }

    // 1 / (1 + e^{(x−μ)/s})
    fn sf(&self, x: &Ex) -> Option<Ex> {
        let ctx = self.context();
        Some(ctx.one() / (ctx.one() + ((x - &self.mean) / &self.scale).exp()))
    }

    // e^{μt} B(1 − st, 1 + st)
    fn mgf(&self, t: &Ex) -> Option<Ex> {
        let ctx = self.context();
        let st = &self.scale * t;
        Some((&self.mean * t).exp() * (ctx.one() - &st).beta(&(ctx.one() + &st)))
    }

    // μ + s ln(p/(1 − p))
    fn quantile(&self, p: &Ex) -> Option<Ex> {
        Some(&self.mean + &self.scale * (p / (self.context().one() - p)).ln())
    }

    // ln s + 2
    fn entropy(&self) -> Option<Ex> {
        Some(self.scale.ln() + 2)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LogNormal, StudentT, FDistribution
// ═══════════════════════════════════════════════════════════════════════════

/// `LogNormal(μ, σ)`: `ln X ~ Normal(μ, σ)`; density
/// `e^{−(ln x − μ)²/(2σ²)} / (xσ√(2π))` on `(0, ∞)`.
#[derive(Clone, Debug, PartialEq)]
pub struct LogNormal {
    /// Mean `μ` of `ln X`.
    pub mu: Ex,
    /// Standard deviation `σ > 0` of `ln X`.
    pub sigma: Ex,
}

impl Family for LogNormal {
    family_boilerplate!(LogNormal, "LogNormal", [mu, sigma]);

    fn support(&self) -> Support {
        Support::half_line(self.context().zero())
    }

    fn density(&self, x: &Ex) -> Ex {
        let ctx = self.context();
        let two = ctx.int(2);
        (-((x.ln() - &self.mu).powi(2)) / (&two * self.sigma.powi(2))).exp()
            / (x * &self.sigma * (&two * ctx.pi()).sqrt())
    }

    // e^{μ + σ²/2}
    fn mean(&self) -> Option<Ex> {
        Some((&self.mu + self.sigma.powi(2) / self.context().int(2)).exp())
    }

    // (e^{σ²} − 1) e^{2μ + σ²}
    fn variance(&self) -> Option<Ex> {
        let ctx = self.context();
        let s2 = self.sigma.powi(2);
        Some(((s2.exp() - 1) * (ctx.int(2) * &self.mu + &s2).exp()).simplify())
    }

    // e^{nμ + n²σ²/2}
    fn raw_moment(&self, n: u32) -> Option<Ex> {
        let ctx = self.context();
        let n_ex = ctx.int(i64::from(n));
        Some((&n_ex * &self.mu + n_ex.powi(2) * self.sigma.powi(2) / ctx.int(2)).exp())
    }

    // Φ((ln x − μ)/σ)
    fn cdf(&self, x: &Ex) -> Option<Ex> {
        let ctx = self.context();
        let half = ctx.rational(1, 2);
        let arg = (x.ln() - &self.mu) / (&self.sigma * ctx.int(2).sqrt());
        Some(&half + &half * arg.erf())
    }

    // ½ erfc((μ − ln x)/(σ√2))
    fn cdf_lower(&self, x: &Ex) -> Option<Ex> {
        let ctx = self.context();
        let arg = (&self.mu - x.ln()) / (&self.sigma * ctx.int(2).sqrt());
        Some(ctx.rational(1, 2) * arg.erfc())
    }

    // ½ erfc((ln x − μ)/(σ√2))
    fn sf(&self, x: &Ex) -> Option<Ex> {
        let ctx = self.context();
        let arg = (x.ln() - &self.mu) / (&self.sigma * ctx.int(2).sqrt());
        Some(ctx.rational(1, 2) * arg.erfc())
    }

    // exp(μ + σ√2 · erfinv(2p − 1))
    fn quantile(&self, p: &Ex) -> Option<Ex> {
        let ctx = self.context();
        Some((&self.mu + &self.sigma * ctx.int(2).sqrt() * (ctx.int(2) * p - 1).erfinv()).exp())
    }

    // μ + ½ ln(2πeσ²)
    fn entropy(&self) -> Option<Ex> {
        let ctx = self.context();
        Some(
            &self.mu
                + ctx.rational(1, 2) * (ctx.int(2) * ctx.pi() * ctx.e() * self.sigma.powi(2)).ln(),
        )
    }
}

/// `StudentT(ν)`: density
/// `Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) · (1 + x²/ν)^{−(ν+1)/2}` on ℝ.
#[derive(Clone, Debug, PartialEq)]
pub struct StudentT {
    /// Degrees of freedom `ν > 0`.
    pub dof: Ex,
}

impl Family for StudentT {
    family_boilerplate!(StudentT, "StudentT", [dof]);

    fn support(&self) -> Support {
        Support::reals(&self.context())
    }

    fn density(&self, x: &Ex) -> Ex {
        let ctx = self.context();
        let half = ctx.rational(1, 2);
        let nu1 = (&self.dof + 1) * &half;
        nu1.gamma() / ((&self.dof * ctx.pi()).sqrt() * (&self.dof * &half).gamma())
            * (ctx.one() + x.powi(2) / &self.dof).pow(&(-nu1))
    }

    // 0 for ν > 1.
    fn mean(&self) -> Option<Ex> {
        exceeds(&self.dof, 1).then(|| self.context().zero())
    }

    // ν / (ν − 2) for ν > 2.
    fn variance(&self) -> Option<Ex> {
        exceeds(&self.dof, 2).then(|| (&self.dof / (&self.dof - 2)).simplify())
    }

    // Odd moments 0; E[X²ᵐ] = νᵐ Π_{i=1}^{m} (2i−1)/(ν−2i), for n < ν.
    fn raw_moment(&self, n: u32) -> Option<Ex> {
        if !exceeds(&self.dof, n) {
            return None;
        }
        let ctx = self.context();
        if n % 2 == 1 {
            return Some(ctx.zero());
        }
        let m = n / 2;
        let mut acc = self.dof.powi(i64::from(m));
        for i in 1..=m {
            acc *= ctx.int(i64::from(2 * i - 1)) / (&self.dof - i64::from(2 * i));
        }
        Some(acc.simplify())
    }

    // F(t) = 1 − ½ I_{ν/(t²+ν)}(ν/2, ½) for t ≥ 0 and ½ I_{ν/(t²+ν)}(ν/2, ½)
    // for t < 0 (the regularised incomplete beta; DLMF 8.17 / Wikipedia
    // "Student's t-distribution").
    fn cdf(&self, x: &Ex) -> Option<Ex> {
        let ctx = self.context();
        let half = ctx.rational(1, 2);
        let z = &self.dof / (x.powi(2) + &self.dof);
        let tail = &half * z.betainc_regularized(&(&self.dof * &half), &half, &ctx.zero());
        let above = ctx.one() - &tail;
        Some(Ex::piecewise(&[
            (&tail, &x.lt(&ctx.zero())),
            (&above, &x.ge(&ctx.zero())),
        ]))
    }

    // The mirror image: ½ I_{ν/(t²+ν)}(ν/2, ½) for t > 0, 1 − that for t ≤ 0.
    fn sf(&self, x: &Ex) -> Option<Ex> {
        let ctx = self.context();
        let half = ctx.rational(1, 2);
        let z = &self.dof / (x.powi(2) + &self.dof);
        let tail = &half * z.betainc_regularized(&(&self.dof * &half), &half, &ctx.zero());
        let below = ctx.one() - &tail;
        Some(Ex::piecewise(&[
            (&tail, &x.gt(&ctx.zero())),
            (&below, &x.le(&ctx.zero())),
        ]))
    }

    // (ν+1)/2 [ψ((ν+1)/2) − ψ(ν/2)] + ln(√ν B(ν/2, ½))
    fn entropy(&self) -> Option<Ex> {
        let ctx = self.context();
        let half = ctx.rational(1, 2);
        let a = (&self.dof + 1) * &half;
        let b = &self.dof * &half;
        Some(&a * (a.digamma() - b.digamma()) + (self.dof.sqrt() * b.beta(&half)).ln())
    }

    // Z / √(V/ν) for independent Z ~ N(0, 1), V ~ χ²(ν) = 2·Gamma(ν/2, 1).
    fn sampler(&self) -> Option<Result<Sampler, SymplexError>> {
        Some(self.build_sampler())
    }
}

impl StudentT {
    fn build_sampler(&self) -> Result<Sampler, SymplexError> {
        let nu = sampler_positive(&self.dof, "the degrees of freedom")?;
        Ok(Box::new(move |rng: &mut Rng| {
            let z = sample::standard_normal(rng);
            let v = 2.0 * sample::standard_gamma(rng, nu / 2.0);
            z / (v / nu).sqrt()
        }))
    }
}

/// `FDistribution(d₁, d₂)`: density
/// `√((d₁x)^{d₁} d₂^{d₂} / (d₁x + d₂)^{d₁+d₂}) / (x B(d₁/2, d₂/2))` on
/// `(0, ∞)` (SymPy's `FDistribution(d1, d2)`).
#[derive(Clone, Debug, PartialEq)]
pub struct FDistribution {
    /// Numerator degrees of freedom `d₁ > 0`.
    pub d1: Ex,
    /// Denominator degrees of freedom `d₂ > 0`.
    pub d2: Ex,
}

impl Family for FDistribution {
    family_boilerplate!(FDistribution, "FDistribution", [d1, d2]);

    fn support(&self) -> Support {
        Support::half_line(self.context().zero())
    }

    fn density(&self, x: &Ex) -> Ex {
        let ctx = self.context();
        let half = ctx.rational(1, 2);
        let num = (&self.d1 * x).pow(&self.d1) * self.d2.pow(&self.d2)
            / (&self.d1 * x + &self.d2).pow(&(&self.d1 + &self.d2));
        num.sqrt() / (x * (&self.d1 * &half).beta(&(&self.d2 * &half)))
    }

    // d₂ / (d₂ − 2) for d₂ > 2.
    fn mean(&self) -> Option<Ex> {
        exceeds(&self.d2, 2).then(|| (&self.d2 / (&self.d2 - 2)).simplify())
    }

    // 2 d₂² (d₁ + d₂ − 2) / (d₁ (d₂ − 2)² (d₂ − 4)) for d₂ > 4.
    fn variance(&self) -> Option<Ex> {
        exceeds(&self.d2, 4).then(|| {
            let ctx = self.context();
            (ctx.int(2) * self.d2.powi(2) * (&self.d1 + &self.d2 - 2)
                / (&self.d1 * (&self.d2 - 2).powi(2) * (&self.d2 - 4)))
                .simplify()
        })
    }

    // E[Xⁿ] = (d₂/d₁)ⁿ Γ(d₁/2 + n) Γ(d₂/2 − n) / (Γ(d₁/2) Γ(d₂/2)) for 2n < d₂.
    fn raw_moment(&self, n: u32) -> Option<Ex> {
        if !exceeds(&self.d2, 2 * n) {
            return None;
        }
        let ctx = self.context();
        let half = ctx.rational(1, 2);
        let n_ex = ctx.int(i64::from(n));
        let a = &self.d1 * &half;
        let b = &self.d2 * &half;
        Some(
            ((&self.d2 / &self.d1).powi(i64::from(n))
                * (&a + &n_ex).gamma()
                * (&b - &n_ex).gamma()
                / (a.gamma() * b.gamma()))
            .simplify(),
        )
    }

    // I_{d₁x/(d₁x + d₂)}(d₁/2, d₂/2)
    fn cdf(&self, x: &Ex) -> Option<Ex> {
        let ctx = self.context();
        let half = ctx.rational(1, 2);
        let z = &self.d1 * x / (&self.d1 * x + &self.d2);
        Some(z.betainc_regularized(&(&self.d1 * &half), &(&self.d2 * &half), &ctx.zero()))
    }

    // I_{d₂/(d₁x + d₂)}(d₂/2, d₁/2)
    fn sf(&self, x: &Ex) -> Option<Ex> {
        let ctx = self.context();
        let half = ctx.rational(1, 2);
        let w = &self.d2 / (&self.d1 * x + &self.d2);
        Some(w.betainc_regularized(&(&self.d2 * &half), &(&self.d1 * &half), &ctx.zero()))
    }

    // (U/d₁) / (V/d₂) for independent U ~ χ²(d₁), V ~ χ²(d₂).
    fn sampler(&self) -> Option<Result<Sampler, SymplexError>> {
        Some(self.build_sampler())
    }
}

impl FDistribution {
    fn build_sampler(&self) -> Result<Sampler, SymplexError> {
        let d1 = sampler_positive(&self.d1, "the numerator degrees of freedom")?;
        let d2 = sampler_positive(&self.d2, "the denominator degrees of freedom")?;
        Ok(Box::new(move |rng: &mut Rng| {
            // The factors 2 of the two χ² variates cancel.
            let u = sample::standard_gamma(rng, d1 / 2.0);
            let v = sample::standard_gamma(rng, d2 / 2.0);
            (u / d1) / (v / d2)
        }))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Weibull, Pareto, Triangular
// ═══════════════════════════════════════════════════════════════════════════

/// `Weibull(λ, k)`: density `(k/λ) (x/λ)^{k−1} e^{−(x/λ)ᵏ}` on `[0, ∞)`.
/// SymPy's `Weibull(alpha, beta)` has `alpha = λ` (scale) and `beta = k`
/// (shape).
#[derive(Clone, Debug, PartialEq)]
pub struct Weibull {
    /// Scale `λ > 0`.
    pub scale: Ex,
    /// Shape `k > 0`.
    pub shape: Ex,
}

impl Family for Weibull {
    family_boilerplate!(Weibull, "Weibull", [scale, shape]);

    fn support(&self) -> Support {
        Support::half_line(self.context().zero())
    }

    fn density(&self, x: &Ex) -> Ex {
        let z = x / &self.scale;
        (&self.shape / &self.scale) * z.pow(&(&self.shape - 1)) * (-(z.pow(&self.shape))).exp()
    }

    // λ Γ(1 + 1/k)
    fn mean(&self) -> Option<Ex> {
        let ctx = self.context();
        Some((&self.scale * (ctx.one() + ctx.one() / &self.shape).gamma()).simplify())
    }

    // λ² [Γ(1 + 2/k) − Γ(1 + 1/k)²]
    fn variance(&self) -> Option<Ex> {
        let ctx = self.context();
        let g1 = (ctx.one() + ctx.one() / &self.shape).gamma();
        let g2 = (ctx.one() + ctx.int(2) / &self.shape).gamma();
        Some((self.scale.powi(2) * (g2 - g1.powi(2))).simplify())
    }

    // λⁿ Γ(1 + n/k)
    fn raw_moment(&self, n: u32) -> Option<Ex> {
        let ctx = self.context();
        let n_ex = ctx.int(i64::from(n));
        Some((self.scale.powi(i64::from(n)) * (ctx.one() + &n_ex / &self.shape).gamma()).simplify())
    }

    // 1 − e^{−(x/λ)ᵏ}
    fn cdf(&self, x: &Ex) -> Option<Ex> {
        Some(self.context().one() - (-((x / &self.scale).pow(&self.shape))).exp())
    }

    // 1 − e^{−y}, y = (x/λ)ᵏ, as for the exponential: the γ(1, y) of 0.28
    // folded at an irrational y, and Weibull(1, ½).cdf(2·10⁻²⁰⁰) was 0.
    fn cdf_lower(&self, x: &Ex) -> Option<Ex> {
        Some(one_minus_exp_neg(&(x / &self.scale).pow(&self.shape)))
    }

    // e^{−(x/λ)ᵏ}
    fn sf(&self, x: &Ex) -> Option<Ex> {
        Some((-((x / &self.scale).pow(&self.shape))).exp())
    }

    // λ (−ln(1 − p))^{1/k}
    fn quantile(&self, p: &Ex) -> Option<Ex> {
        let ctx = self.context();
        Some(&self.scale * (-(ctx.one() - p).ln()).pow(&(ctx.one() / &self.shape)))
    }

    // γ(1 − 1/k) + ln(λ/k) + 1
    fn entropy(&self) -> Option<Ex> {
        let ctx = self.context();
        Some(
            ctx.euler_gamma() * (ctx.one() - ctx.one() / &self.shape)
                + (&self.scale / &self.shape).ln()
                + 1,
        )
    }
}

/// `Pareto(x_m, α)`: density `α x_mᵅ / x^{α+1}` on `[x_m, ∞)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Pareto {
    /// Scale `x_m > 0` (the minimum).
    pub scale: Ex,
    /// Shape (tail index) `α > 0`.
    pub shape: Ex,
}

impl Family for Pareto {
    family_boilerplate!(Pareto, "Pareto", [scale, shape]);

    fn support(&self) -> Support {
        Support::half_line(self.scale.clone())
    }

    fn density(&self, x: &Ex) -> Ex {
        &self.shape * self.scale.pow(&self.shape) / x.pow(&(&self.shape + 1))
    }

    // α x_m / (α − 1) for α > 1.
    fn mean(&self) -> Option<Ex> {
        exceeds(&self.shape, 1).then(|| (&self.shape * &self.scale / (&self.shape - 1)).simplify())
    }

    // x_m² α / ((α − 1)² (α − 2)) for α > 2.
    fn variance(&self) -> Option<Ex> {
        exceeds(&self.shape, 2).then(|| {
            (self.scale.powi(2) * &self.shape / ((&self.shape - 1).powi(2) * (&self.shape - 2)))
                .simplify()
        })
    }

    // α x_mⁿ / (α − n) for n < α.
    fn raw_moment(&self, n: u32) -> Option<Ex> {
        let n_ex = self.context().int(i64::from(n));
        exceeds(&self.shape, n).then(|| {
            (&self.shape * self.scale.powi(i64::from(n)) / (&self.shape - &n_ex)).simplify()
        })
    }

    // 1 − (x_m/x)^α
    fn cdf(&self, x: &Ex) -> Option<Ex> {
        Some(self.context().one() - (&self.scale / x).pow(&self.shape))
    }

    // 1 − e^{−y} with y = α ln(x/x_m) = 2α atanh((x − x_m)/(x + x_m)): next
    // to x_m the classic form is a difference of two numbers next to 1
    // (Pareto(1, √2).cdf(1 + 10⁻¹⁰⁰) was 0).
    fn cdf_lower(&self, x: &Ex) -> Option<Ex> {
        Some(one_minus_exp_neg(&(&self.shape * ln_ratio(x, &self.scale))))
    }

    // (x_m/x)^α
    fn sf(&self, x: &Ex) -> Option<Ex> {
        Some((&self.scale / x).pow(&self.shape))
    }

    // x_m (1 − p)^{−1/α}
    fn quantile(&self, p: &Ex) -> Option<Ex> {
        let ctx = self.context();
        Some(&self.scale * (ctx.one() - p).pow(&(-(ctx.one() / &self.shape))))
    }

    // ln(x_m/α) + 1/α + 1
    fn entropy(&self) -> Option<Ex> {
        let ctx = self.context();
        Some((&self.scale / &self.shape).ln() + ctx.one() / &self.shape + 1)
    }
}

/// `Triangular(a, b, c)`: density rising linearly from `0` at `a` to
/// `2/(b−a)` at the mode `c` and falling linearly to `0` at `b`, on
/// `[a, b]` with `a ≤ c ≤ b`.
#[derive(Clone, Debug, PartialEq)]
pub struct Triangular {
    /// Lower end `a`.
    pub lo: Ex,
    /// Upper end `b > a`.
    pub hi: Ex,
    /// Mode `c ∈ [a, b]`.
    pub mode: Ex,
}

impl Triangular {
    /// Is the mode decidably at the lower end (`c = a`)?  The rising
    /// piece is then a single point and its formula divides by `c − a = 0`.
    fn mode_at_lo(&self) -> bool {
        (&self.mode - &self.lo).is_zero() == Some(true)
    }

    /// Is the mode decidably at the upper end (`c = b`)?
    fn mode_at_hi(&self) -> bool {
        (&self.hi - &self.mode).is_zero() == Some(true)
    }

    /// `2(x−a)/((b−a)(c−a))` on `[a, c]` and `2(b−x)/((b−a)(b−c))` on
    /// `(c, b]`: the piece that applies, or the `Piecewise` of both.
    fn two_pieces(&self, x: &Ex, rising: Ex, falling: Ex) -> Ex {
        if self.mode_at_lo() {
            falling
        } else if self.mode_at_hi() {
            rising
        } else {
            Ex::piecewise(&[(&rising, &x.le(&self.mode)), (&falling, &x.gt(&self.mode))])
        }
    }
}

impl Family for Triangular {
    family_boilerplate!(Triangular, "Triangular", [lo, hi, mode]);

    fn support(&self) -> Support {
        Support::interval(self.lo.clone(), self.hi.clone())
    }

    // Wikipedia, "Triangular distribution": 2(x−a)/((b−a)(c−a)) on [a, c],
    // 2(b−x)/((b−a)(b−c)) on (c, b].
    fn density(&self, x: &Ex) -> Ex {
        let ctx = self.context();
        let two = ctx.int(2);
        let width = &self.hi - &self.lo;
        let rising = &two * (x - &self.lo) / (&width * (&self.mode - &self.lo));
        let falling = &two * (&self.hi - x) / (&width * (&self.hi - &self.mode));
        self.two_pieces(x, rising, falling)
    }

    fn mean(&self) -> Option<Ex> {
        Some(((&self.lo + &self.hi + &self.mode) / self.context().int(3)).simplify())
    }

    // (a² + b² + c² − ab − ac − bc) / 18
    fn variance(&self) -> Option<Ex> {
        let (a, b, c) = (&self.lo, &self.hi, &self.mode);
        Some(
            ((a.powi(2) + b.powi(2) + c.powi(2) - a * b - a * c - b * c) / self.context().int(18))
                .simplify(),
        )
    }

    // 2 [aⁿ⁺²(b−c) − bⁿ⁺²(a−c) + cⁿ⁺²(a−b)] / ((n+1)(n+2)(a−b)(a−c)(b−c));
    // 0/0 with the mode at an end, where the generic integration of the
    // single linear piece is exact.
    fn raw_moment(&self, n: u32) -> Option<Ex> {
        if self.mode_at_lo() || self.mode_at_hi() {
            return None;
        }
        let ctx = self.context();
        let (a, b, c) = (&self.lo, &self.hi, &self.mode);
        let e = i64::from(n) + 2;
        let num = ctx.int(2) * (a.powi(e) * (b - c) - b.powi(e) * (a - c) + c.powi(e) * (a - b));
        let den = ctx.int(e - 1) * ctx.int(e) * (a - b) * (a - c) * (b - c);
        Some((num / den).simplify())
    }

    // (x−a)²/((b−a)(c−a)) on [a, c], 1 − (b−x)²/((b−a)(b−c)) on (c, b]
    fn cdf(&self, x: &Ex) -> Option<Ex> {
        let ctx = self.context();
        let width = &self.hi - &self.lo;
        let rising = (x - &self.lo).powi(2) / (&width * (&self.mode - &self.lo));
        let falling = ctx.one() - (&self.hi - x).powi(2) / (&width * (&self.hi - &self.mode));
        Some(self.two_pieces(x, rising, falling))
    }

    // With the mode at the lower end the one piece is 1 − (b−x)²/(b−a)²,
    // which cancels next to a: (x−a)(2b−a−x)/(b−a)² instead.  Otherwise the
    // piece next to a is the rising one, already a square.
    fn cdf_lower(&self, x: &Ex) -> Option<Ex> {
        if !self.mode_at_lo() {
            return None;
        }
        let (a, b) = (&self.lo, &self.hi);
        Some((x - a) * (self.context().int(2) * b - a - x) / (b - a).powi(2))
    }

    // (b−x)²/((b−a)(b−c)) on (c, b], 1 − (x−a)²/((b−a)(c−a)) on [a, c]; with
    // the mode at the upper end the one piece is (b−x)(b+x−2a)/(b−a)².  The
    // generic 1 − F was 0 next to b for an irrational end
    // (Triangular(0, √2, 1).sf(√2 − 10⁻³⁰)).
    fn sf(&self, x: &Ex) -> Option<Ex> {
        let ctx = self.context();
        let (a, b, c) = (&self.lo, &self.hi, &self.mode);
        let width = b - a;
        if self.mode_at_hi() {
            return Some((b - x) * (b + x - ctx.int(2) * a) / width.powi(2));
        }
        let rising = ctx.one() - (x - a).powi(2) / (&width * (c - a));
        let falling = (b - x).powi(2) / (&width * (b - c));
        Some(self.two_pieces(x, rising, falling))
    }

    // 2 [(b−c) e^{at} − (b−a) e^{ct} + (c−a) e^{bt}] / ((b−a)(c−a)(b−c) t²);
    // with the mode at an end the limit of that 0/0:
    // c = a: 2 [e^{bt} − e^{at} − (b−a) t e^{at}] / ((b−a)² t²),
    // c = b: 2 [(b−a) t e^{bt} − e^{bt} + e^{at}] / ((b−a)² t²).
    fn mgf(&self, t: &Ex) -> Option<Ex> {
        let ctx = self.context();
        let (a, b, c) = (&self.lo, &self.hi, &self.mode);
        let width = b - a;
        if self.mode_at_lo() {
            let num = ctx.int(2) * ((b * t).exp() - (a * t).exp() - &width * t * (a * t).exp());
            return Some(num / (width.powi(2) * t.powi(2)));
        }
        if self.mode_at_hi() {
            let num = ctx.int(2) * (&width * t * (b * t).exp() - (b * t).exp() + (a * t).exp());
            return Some(num / (width.powi(2) * t.powi(2)));
        }
        let num = ctx.int(2)
            * ((b - c) * (a * t).exp() - (b - a) * (c * t).exp() + (c - a) * (b * t).exp());
        let den = (b - a) * (c - a) * (b - c) * t.powi(2);
        Some(num / den)
    }

    // a + √(p(b−a)(c−a)) for p < (c−a)/(b−a), else b − √((1−p)(b−a)(b−c))
    fn quantile(&self, p: &Ex) -> Option<Ex> {
        let ctx = self.context();
        let width = &self.hi - &self.lo;
        let rising = &self.lo + (p * &width * (&self.mode - &self.lo)).sqrt();
        let falling = &self.hi - ((ctx.one() - p) * &width * (&self.hi - &self.mode)).sqrt();
        if self.mode_at_lo() {
            return Some(falling);
        }
        if self.mode_at_hi() {
            return Some(rising);
        }
        let threshold = (&self.mode - &self.lo) / &width;
        Some(Ex::piecewise(&[
            (&rising, &p.lt(&threshold)),
            (&falling, &p.ge(&threshold)),
        ]))
    }

    // ½ + ln((b − a)/2)
    fn entropy(&self) -> Option<Ex> {
        let ctx = self.context();
        Some(ctx.rational(1, 2) + ((&self.hi - &self.lo) / ctx.int(2)).ln())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Constructors
// ═══════════════════════════════════════════════════════════════════════════

impl Distribution {
    /// `Normal(μ, σ)` with mean `mean` and standard deviation `std`.
    /// SymPy: `Normal('X', mu, sigma)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `std` is a number `≤ 0`.
    pub fn try_normal(mean: Ex, std: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&std, "the standard deviation")?;
        Ok(Distribution::normal(mean, std))
    }

    /// `Normal(μ, σ)`; a non-positive numeric `std` is a programming error
    /// and yields the distribution anyway (use [`try_normal`](Self::try_normal)
    /// to check).
    pub fn normal(mean: Ex, std: Ex) -> Distribution {
        Distribution::from_family(Normal { mean, std })
    }

    /// `Uniform(a, b)` on `[lo, hi]`.  SymPy: `Uniform('X', a, b)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `hi − lo` is a number `≤ 0`.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::Distribution;
    ///
    /// let ctx = Context::new();
    /// assert!(Distribution::try_uniform(ctx.int(0), ctx.int(1)).is_ok());
    /// assert!(Distribution::try_uniform(ctx.int(1), ctx.int(1)).is_err());
    /// ```
    pub fn try_uniform(lo: Ex, hi: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&(&hi - &lo), "the width `b − a`")?;
        Ok(Distribution::uniform(lo, hi))
    }

    /// `Uniform(a, b)` without parameter validation (see
    /// [`try_uniform`](Self::try_uniform)).
    pub fn uniform(lo: Ex, hi: Ex) -> Distribution {
        Distribution::from_family(Uniform { lo, hi })
    }

    /// `Exponential(λ)` with rate `rate`.  SymPy: `Exponential('X', rate)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `rate` is a number `≤ 0`.
    pub fn try_exponential(rate: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&rate, "the rate")?;
        Ok(Distribution::exponential(rate))
    }

    /// `Exponential(λ)` without parameter validation (see
    /// [`try_exponential`](Self::try_exponential)).
    pub fn exponential(rate: Ex) -> Distribution {
        Distribution::from_family(Exponential { rate })
    }

    /// `Gamma(k, θ)` with shape `shape` and scale `scale`.  SymPy:
    /// `Gamma('X', k, theta)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if either parameter is a number
    /// `≤ 0`.
    pub fn try_gamma(shape: Ex, scale: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&shape, "the shape")?;
        require_positive(&scale, "the scale")?;
        Ok(Distribution::gamma(shape, scale))
    }

    /// `Gamma(k, θ)` without parameter validation (see
    /// [`try_gamma`](Self::try_gamma)).
    pub fn gamma(shape: Ex, scale: Ex) -> Distribution {
        Distribution::from_family(Gamma { shape, scale })
    }

    /// `ChiSquared(k)` with `dof` degrees of freedom.  SymPy:
    /// `ChiSquared('X', k)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `dof` is a number `≤ 0`.
    pub fn try_chi_squared(dof: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&dof, "the degrees of freedom")?;
        Ok(Distribution::chi_squared(dof))
    }

    /// `ChiSquared(k)` without parameter validation (see
    /// [`try_chi_squared`](Self::try_chi_squared)).
    pub fn chi_squared(dof: Ex) -> Distribution {
        Distribution::from_family(ChiSquared { dof })
    }

    /// `Beta(α, β)`.  SymPy: `Beta('X', alpha, beta)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if either parameter is a number
    /// `≤ 0`.
    pub fn try_beta(alpha: Ex, beta: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&alpha, "the shape α")?;
        require_positive(&beta, "the shape β")?;
        Ok(Distribution::beta(alpha, beta))
    }

    /// `Beta(α, β)` without parameter validation (see
    /// [`try_beta`](Self::try_beta)).
    pub fn beta(alpha: Ex, beta: Ex) -> Distribution {
        Distribution::from_family(Beta { alpha, beta })
    }

    /// `Cauchy(x₀, γ)` with location `location` and scale `scale`.  SymPy:
    /// `Cauchy('X', x0, gamma)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `scale` is a number `≤ 0`.
    pub fn try_cauchy(location: Ex, scale: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&scale, "the scale")?;
        Ok(Distribution::cauchy(location, scale))
    }

    /// `Cauchy(x₀, γ)` without parameter validation (see
    /// [`try_cauchy`](Self::try_cauchy)).
    pub fn cauchy(location: Ex, scale: Ex) -> Distribution {
        Distribution::from_family(Cauchy { location, scale })
    }

    /// `Laplace(μ, b)` with location `mean` and scale `scale`.  SymPy:
    /// `Laplace('X', mu, b)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `scale` is a number `≤ 0`.
    pub fn try_laplace(mean: Ex, scale: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&scale, "the scale")?;
        Ok(Distribution::laplace(mean, scale))
    }

    /// `Laplace(μ, b)` without parameter validation (see
    /// [`try_laplace`](Self::try_laplace)).
    pub fn laplace(mean: Ex, scale: Ex) -> Distribution {
        Distribution::from_family(Laplace { mean, scale })
    }

    /// `Logistic(μ, s)` with location `mean` and scale `scale`.  SymPy:
    /// `Logistic('X', mu, s)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `scale` is a number `≤ 0`.
    pub fn try_logistic(mean: Ex, scale: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&scale, "the scale")?;
        Ok(Distribution::logistic(mean, scale))
    }

    /// `Logistic(μ, s)` without parameter validation (see
    /// [`try_logistic`](Self::try_logistic)).
    pub fn logistic(mean: Ex, scale: Ex) -> Distribution {
        Distribution::from_family(Logistic { mean, scale })
    }

    /// `LogNormal(μ, σ)`: `ln X ~ Normal(mu, sigma)`.  SymPy:
    /// `LogNormal('X', mu, sigma)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `sigma` is a number `≤ 0`.
    pub fn try_log_normal(mu: Ex, sigma: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&sigma, "the standard deviation of ln X")?;
        Ok(Distribution::log_normal(mu, sigma))
    }

    /// `LogNormal(μ, σ)` without parameter validation (see
    /// [`try_log_normal`](Self::try_log_normal)).
    pub fn log_normal(mu: Ex, sigma: Ex) -> Distribution {
        Distribution::from_family(LogNormal { mu, sigma })
    }

    /// `StudentT(ν)` with `dof` degrees of freedom.  SymPy:
    /// `StudentT('X', nu)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `dof` is a number `≤ 0`.
    pub fn try_student_t(dof: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&dof, "the degrees of freedom")?;
        Ok(Distribution::student_t(dof))
    }

    /// `StudentT(ν)` without parameter validation (see
    /// [`try_student_t`](Self::try_student_t)).
    pub fn student_t(dof: Ex) -> Distribution {
        Distribution::from_family(StudentT { dof })
    }

    /// `FDistribution(d₁, d₂)`.  SymPy: `FDistribution('X', d1, d2)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if either parameter is a number
    /// `≤ 0`.
    pub fn try_f_distribution(d1: Ex, d2: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&d1, "the numerator degrees of freedom")?;
        require_positive(&d2, "the denominator degrees of freedom")?;
        Ok(Distribution::f_distribution(d1, d2))
    }

    /// `FDistribution(d₁, d₂)` without parameter validation (see
    /// [`try_f_distribution`](Self::try_f_distribution)).
    pub fn f_distribution(d1: Ex, d2: Ex) -> Distribution {
        Distribution::from_family(FDistribution { d1, d2 })
    }

    /// `Weibull(λ, k)` with scale `scale` and shape `shape`.  SymPy:
    /// `Weibull('X', alpha, beta)` with `alpha = scale`, `beta = shape`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if either parameter is a number
    /// `≤ 0`.
    pub fn try_weibull(scale: Ex, shape: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&scale, "the scale")?;
        require_positive(&shape, "the shape")?;
        Ok(Distribution::weibull(scale, shape))
    }

    /// `Weibull(λ, k)` without parameter validation (see
    /// [`try_weibull`](Self::try_weibull)).
    pub fn weibull(scale: Ex, shape: Ex) -> Distribution {
        Distribution::from_family(Weibull { scale, shape })
    }

    /// `Pareto(x_m, α)` with minimum `scale` and tail index `shape`.
    /// SymPy: `Pareto('X', xm, alpha)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if either parameter is a number
    /// `≤ 0`.
    pub fn try_pareto(scale: Ex, shape: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&scale, "the scale x_m")?;
        require_positive(&shape, "the shape α")?;
        Ok(Distribution::pareto(scale, shape))
    }

    /// `Pareto(x_m, α)` without parameter validation (see
    /// [`try_pareto`](Self::try_pareto)).
    pub fn pareto(scale: Ex, shape: Ex) -> Distribution {
        Distribution::from_family(Pareto { scale, shape })
    }

    /// `Triangular(a, b, c)` on `[lo, hi]` with mode `mode`.  SymPy:
    /// `Triangular('X', a, b, c)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the parameters are numbers
    /// with `hi ≤ lo` or `mode` outside `[lo, hi]`.
    pub fn try_triangular(lo: Ex, hi: Ex, mode: Ex) -> Result<Distribution, SymplexError> {
        require_positive(&(&hi - &lo), "the width `b − a`")?;
        require_nonnegative(&(&mode - &lo), "`c − a` (the mode must lie in [a, b])")?;
        require_nonnegative(&(&hi - &mode), "`b − c` (the mode must lie in [a, b])")?;
        Ok(Distribution::triangular(lo, hi, mode))
    }

    /// `Triangular(a, b, c)` without parameter validation (see
    /// [`try_triangular`](Self::try_triangular)).
    pub fn triangular(lo: Ex, hi: Ex, mode: Ex) -> Distribution {
        Distribution::from_family(Triangular { lo, hi, mode })
    }
}
