//! 0.18 track: samplers for the families that had none — `Gamma`,
//! `ChiSquared`, `Beta`, `StudentT`, `FDistribution` (continuous, no
//! elementary quantile) and `Poisson`, `Geometric`, `NegativeBinomial`
//! (discrete, infinite lattice).
//!
//! Every check draws from a fixed seed, so each test is a deterministic
//! regression test of the algorithm rather than a flaky statistical one:
//!
//! * the sample mean is within `4σ/√n` of the exact `mean()`, and the
//!   sample variance within `5·√((μ₄ − σ⁴)/n)` of `variance()` (the
//!   asymptotic standard error of `s²`, with `μ₄` the exact fourth central
//!   moment);
//! * every draw lies in `support()`;
//! * for a discrete family, the empirical pmf at the mode is within
//!   `4√(p(1−p)/n)` of the exact `pmf`;
//! * for two continuous families, a Kolmogorov–Smirnov-style statistic
//!   `D = max |F_n(x) − F(x)|` over the empirical quantiles, with `F` the
//!   exact `cdf` evaluated by `eval_f64`, is below `1.63/√n` — the 1 %
//!   critical value of the one-sample KS test for `n > 35` (Massey, "The
//!   Kolmogorov–Smirnov test for goodness of fit", JASA 46 (1951), Table 1);
//! * for the other continuous families, the empirical CDF at
//!   `scipy.stats.<dist>.ppf(0.3)` (scipy 1.18, `symplex/.venv/bin/python`)
//!   is within `4√(0.3·0.7/n)` of `0.3`;
//! * the same seed reproduces the same first ten draws.
//!
//! `n = 20 000` except where a draw costs two gamma variates (`StudentT`,
//! `FDistribution`, `NegativeBinomial`: `n = 5 000`).

use symplex::prelude::*;
use symplex::stats::{Distribution, Kind, Rng, Support};

// ── helpers ──────────────────────────────────────────────────────────────

const N: usize = 20_000;
const N_SMALL: usize = 5_000;

fn f64_of(e: &Ex) -> f64 {
    e.eval_f64()
        .unwrap_or_else(|err| panic!("`{e}` does not evaluate: {err}"))
}

fn draw(d: &Distribution, n: usize, seed: u64) -> Vec<f64> {
    let s = d
        .sample(n, &mut Rng::new(seed))
        .unwrap_or_else(|e| panic!("sampling {d}: {e}"));
    assert_eq!(s.len(), n);
    s
}

struct Moments {
    mean: f64,
    var: f64,
}

fn moments(s: &[f64]) -> Moments {
    let n = s.len() as f64;
    let mean = s.iter().sum::<f64>() / n;
    let var = s.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Moments { mean, var }
}

/// (a) Sample mean within `4σ/√n` of `mean()`; sample variance within
/// `5·√((μ₄ − σ⁴)/n)` of `variance()`.
fn check_moments(d: &Distribution, s: &[f64]) {
    let n = s.len() as f64;
    let m = moments(s);
    let mean = f64_of(&d.mean());
    let var = f64_of(&d.variance());
    let tol_mean = 4.0 * var.sqrt() / n.sqrt();
    assert!(
        (m.mean - mean).abs() <= tol_mean,
        "{d}: sample mean {} vs exact {mean} (4σ/√n = {tol_mean})",
        m.mean
    );
    let mu4 = f64_of(&d.central_moment(4));
    let tol_var = 5.0 * ((mu4 - var * var) / n).sqrt();
    assert!(
        (m.var - var).abs() <= tol_var,
        "{d}: sample variance {} vs exact {var} (5·se = {tol_var})",
        m.var
    );
}

/// (b) Every draw is finite and inside the support: the interval's ends
/// (evaluated once) bound every sample, a lattice family draws integers,
/// and `Support::contains` agrees on the first two hundred.
fn check_support(d: &Distribution, s: &[f64]) {
    let ctx = d.context();
    let support = d.support();
    let iv = support
        .as_interval()
        .unwrap_or_else(|| panic!("{d}: support {support} is not one interval"));
    let discrete = d.kind() == Kind::Discrete;
    // `eval_f64` refuses `±oo`; read those ends as the `f64` infinities.
    let end = |e: &Ex, inf: f64| {
        if *e == ctx.infinity() || *e == ctx.neg_infinity() {
            inf
        } else {
            f64_of(e)
        }
    };
    let lo = end(&iv.lower, f64::NEG_INFINITY);
    let hi = end(&iv.upper, f64::INFINITY);
    for v in s {
        assert!(v.is_finite(), "{d}: non-finite draw {v}");
        assert!(*v >= lo && *v <= hi, "{d}: draw {v} outside [{lo}, {hi}]");
        if discrete {
            assert_eq!(v.fract(), 0.0, "{d}: non-integer draw {v}");
        }
    }
    for v in s.iter().take(200) {
        let point = if discrete {
            ctx.int(*v as i64)
        } else {
            ctx.from_f64(*v).expect("f64 to Ex")
        };
        assert_ne!(
            support.contains(&point),
            Some(false),
            "{d}: sample {v} outside {support}"
        );
    }
}

/// (c) Empirical pmf at `k` within `4√(p(1−p)/n)` of the exact `pmf(k)`.
fn check_pmf_at(d: &Distribution, s: &[f64], k: i64) {
    let ctx = d.context();
    let n = s.len() as f64;
    let exact = f64_of(&d.density(&ctx.int(k)));
    let empirical = s.iter().filter(|v| **v == k as f64).count() as f64 / n;
    let tol = 4.0 * (exact * (1.0 - exact) / n).sqrt();
    assert!(
        (empirical - exact).abs() <= tol,
        "{d}: empirical pmf({k}) = {empirical} vs exact {exact} (tol {tol})"
    );
}

/// (d) Kolmogorov–Smirnov-style statistic over `m` empirical quantiles:
/// with the sample sorted, `F_n(x_(i)) = i/n`, and `D = max_j |j/m −
/// F(x_(⌈jn/m⌉))|` for `j = 1..m` with `F` the exact `cdf` through
/// `eval_f64`.  `D < 1.63/√n` (Massey 1951, 1 % level, `n > 35`).
fn check_ks(d: &Distribution, s: &[f64], m: usize) {
    let ctx = d.context();
    let n = s.len();
    let mut sorted = s.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut dmax = 0.0f64;
    for j in 1..m {
        let p = j as f64 / m as f64;
        let i = ((p * n as f64).ceil() as usize).clamp(1, n);
        let x = sorted[i - 1];
        let f = f64_of(&d.cdf(&ctx.from_f64(x).expect("x")));
        // F_n jumps from (i−1)/n to i/n at x_(i).
        let below = (i - 1) as f64 / n as f64;
        let above = i as f64 / n as f64;
        dmax = dmax.max((above - f).abs()).max((below - f).abs());
    }
    let critical = 1.63 / (n as f64).sqrt();
    assert!(
        dmax < critical,
        "{d}: KS statistic D = {dmax} ≥ 1.63/√n = {critical}"
    );
}

/// The empirical CDF at `x = ppf(p)` (scipy) within `4√(p(1−p)/n)` of `p`.
fn check_fraction_below(d: &Distribution, s: &[f64], x: f64, p: f64) {
    let n = s.len() as f64;
    let frac = s.iter().filter(|v| **v <= x).count() as f64 / n;
    let tol = 4.0 * (p * (1.0 - p) / n).sqrt();
    assert!(
        (frac - p).abs() <= tol,
        "{d}: F_n({x}) = {frac} vs {p} (tol {tol})"
    );
}

/// (e) The same seed gives the same first ten draws; another seed does not.
fn check_reproducible(d: &Distribution, seed: u64) {
    let a = draw(d, 10, seed);
    let b = draw(d, 10, seed);
    assert_eq!(a, b, "{d}: seed {seed} is not reproducible");
    let c = draw(d, 10, seed + 1);
    assert_ne!(a, c, "{d}: seeds {seed} and {} agree", seed + 1);
    // `sample_one` and a reused `sampler` walk the same stream.
    let mut rng = Rng::new(seed);
    let first = d.sample_one(&mut rng).expect("sample_one");
    assert_eq!(first, a[0]);
    let mut rng = Rng::new(seed);
    let mut sampler = d.sampler().expect("sampler");
    let via_sampler: Vec<f64> = (0..10).map(|_| sampler(&mut rng)).collect();
    assert_eq!(via_sampler, a);
}

// ── Gamma and χ² ─────────────────────────────────────────────────────────

#[test]
fn gamma_marsaglia_tsang_shape_above_one() {
    // Gamma(5/2, 3/2): mean 15/4, variance 45/8.
    let ctx = Context::new();
    let d = Distribution::gamma(ctx.rational(5, 2), ctx.rational(3, 2));
    let s = draw(&d, N, 101);
    check_moments(&d, &s);
    check_support(&d, &s);
    // Half-integer shape: the cdf closes through erf, so eval_f64 is exact
    // (and slow enough that 100 quantiles keep the test near half a second).
    check_ks(&d, &s, 100);
    // scipy: stats.gamma.ppf(0.3, 2.5, scale=1.5) = 2.24993109956993
    check_fraction_below(&d, &s, 2.24993109956993, 0.3);
    check_reproducible(&d, 101);
}

#[test]
fn gamma_shape_below_one_uses_the_boost() {
    // Gamma(7/10, 2): mean 7/5, variance 14/5; the `k < 1` branch
    // `Gamma(k+1)·U^{1/k}` — every draw is still non-negative.
    let ctx = Context::new();
    let d = Distribution::gamma(ctx.rational(7, 10), ctx.int(2));
    let s = draw(&d, N, 102);
    check_moments(&d, &s);
    check_support(&d, &s);
    check_reproducible(&d, 102);
}

#[test]
fn gamma_shape_one_is_exponential() {
    // Gamma(1, θ) is Exponential(1/θ): the Marsaglia–Tsang route at the
    // branch point `k = 1` agrees with the inverse-transform route in law.
    let ctx = Context::new();
    let g = Distribution::gamma(ctx.int(1), ctx.int(2));
    let e = Distribution::exponential(ctx.rational(1, 2));
    let sg = draw(&g, N, 103);
    check_moments(&g, &sg);
    check_moments(&e, &sg);
    check_support(&g, &sg);
    // scipy: stats.expon.ppf(0.3, scale=2) = 0.7133498878774649
    check_fraction_below(&g, &sg, 0.7133498878774649, 0.3);
}

#[test]
fn chi_squared_is_gamma_half_dof_scale_two() {
    // χ²(3): mean 3, variance 6.
    let ctx = Context::new();
    let d = Distribution::chi_squared(ctx.int(3));
    let s = draw(&d, N, 104);
    check_moments(&d, &s);
    check_support(&d, &s);
    // scipy: stats.chi2.ppf(0.3, 3) = 1.4236522430352796
    check_fraction_below(&d, &s, 1.4236522430352796, 0.3);
    check_reproducible(&d, 104);
    // The same stream as Gamma(3/2, 2).
    let g = Distribution::gamma(ctx.rational(3, 2), ctx.int(2));
    assert_eq!(draw(&g, 10, 104), draw(&d, 10, 104));
}

// ── Beta ─────────────────────────────────────────────────────────────────

#[test]
fn beta_as_a_ratio_of_gammas() {
    // Beta(2, 5): mean 2/7, variance 5/196.
    let ctx = Context::new();
    let d = Distribution::beta(ctx.int(2), ctx.int(5));
    let s = draw(&d, N, 105);
    check_moments(&d, &s);
    check_support(&d, &s);
    assert!(s.iter().all(|v| (0.0..=1.0).contains(v)));
    // Integer shapes: the cdf is a polynomial, so eval_f64 is exact.
    check_ks(&d, &s, 200);
    // scipy: stats.beta.ppf(0.3, 2, 5) = 0.18180347131894917
    check_fraction_below(&d, &s, 0.18180347131894917, 0.3);
    check_reproducible(&d, 105);
}

#[test]
fn beta_with_shapes_below_one_stays_in_the_unit_interval() {
    // Beta(1/2, 1/2) (arcsine): mean 1/2, variance 1/8; both gammas take
    // the boost branch and the ratio must never be 0/0.
    let ctx = Context::new();
    let d = Distribution::beta(ctx.rational(1, 2), ctx.rational(1, 2));
    let s = draw(&d, N, 106);
    check_moments(&d, &s);
    check_support(&d, &s);
    // scipy: stats.beta.ppf(0.3, 0.5, 0.5) = 0.20610737385376346
    check_fraction_below(&d, &s, 0.20610737385376346, 0.3);
}

// ── Student t and F ──────────────────────────────────────────────────────

#[test]
fn student_t_as_normal_over_root_chi_squared() {
    // t(7): mean 0, variance 7/5, fourth moment 49/5.
    let ctx = Context::new();
    let d = Distribution::student_t(ctx.int(7));
    let s = draw(&d, N_SMALL, 107);
    check_moments(&d, &s);
    check_support(&d, &s);
    // scipy: stats.t.ppf(0.3, 7) = -0.5491096579472854
    check_fraction_below(&d, &s, -0.5491096579472854, 0.3);
    // Symmetric about 0.
    let below = s.iter().filter(|v| **v < 0.0).count() as f64 / N_SMALL as f64;
    assert!((below - 0.5).abs() <= 4.0 * 0.5 / (N_SMALL as f64).sqrt());
    check_reproducible(&d, 107);
}

#[test]
fn f_distribution_as_a_ratio_of_scaled_chi_squareds() {
    // F(5, 12): mean 6/5, variance 27/25 (fourth moment exists for d₂ > 8).
    let ctx = Context::new();
    let d = Distribution::f_distribution(ctx.int(5), ctx.int(12));
    let s = draw(&d, N_SMALL, 108);
    check_moments(&d, &s);
    check_support(&d, &s);
    // scipy: stats.f.ppf(0.3, 5, 12) = 0.6018390885923898
    check_fraction_below(&d, &s, 0.6018390885923898, 0.3);
    check_reproducible(&d, 108);
}

// ── Poisson ──────────────────────────────────────────────────────────────

#[test]
fn poisson_knuth_below_thirty() {
    // Poisson(4): mean = variance = 4; mode 4 (pmf(3) = pmf(4)).
    let ctx = Context::new();
    let d = Distribution::poisson(ctx.int(4));
    let s = draw(&d, N, 109);
    check_moments(&d, &s);
    check_support(&d, &s);
    // scipy: stats.poisson.pmf(4, 4) = 0.19536681481316454
    check_pmf_at(&d, &s, 4);
    assert!((f64_of(&d.density(&ctx.int(4))) - 0.19536681481316454).abs() < 1e-15);
    check_reproducible(&d, 109);
}

#[test]
fn poisson_ptrs_at_and_above_thirty() {
    // Poisson(45): the transformed-rejection branch; mode 45 (= 44).
    let ctx = Context::new();
    let d = Distribution::poisson(ctx.int(45));
    let s = draw(&d, N, 110);
    check_moments(&d, &s);
    check_support(&d, &s);
    // scipy: stats.poisson.pmf(45, 45) = 0.05936077647306439
    check_pmf_at(&d, &s, 45);
    check_reproducible(&d, 110);
    // Exactly at the switch, and well past it.
    let at = Distribution::poisson(ctx.int(30));
    check_moments(&at, &draw(&at, N, 111));
    let big = Distribution::poisson(ctx.int(1000));
    check_moments(&big, &draw(&big, N, 112));
}

#[test]
fn poisson_small_rate_is_mostly_zero() {
    // Poisson(1/20): P(0) = e^{−1/20}.
    let ctx = Context::new();
    let d = Distribution::poisson(ctx.rational(1, 20));
    let s = draw(&d, N, 113);
    check_moments(&d, &s);
    check_pmf_at(&d, &s, 0);
}

// ── Geometric ────────────────────────────────────────────────────────────

#[test]
fn geometric_counts_trials_from_one() {
    // Geometric(3/10) on 1..∞: mean 10/3, variance 70/9, mode 1.
    let ctx = Context::new();
    let d = Distribution::geometric(ctx.rational(3, 10));
    let s = draw(&d, N, 114);
    check_moments(&d, &s);
    check_support(&d, &s);
    assert!(s.iter().all(|v| *v >= 1.0), "a trial count is at least 1");
    // scipy: stats.geom.pmf(1, 0.3) = 0.3, stats.geom.pmf(2, 0.3) = 0.21
    check_pmf_at(&d, &s, 1);
    check_pmf_at(&d, &s, 2);
    // scipy: stats.geom.ppf(0.5, 0.3) = 2.0, i.e. P(X ≤ 2) = 0.51 ≥ 0.5 > P(X ≤ 1)
    check_fraction_below(&d, &s, 2.0, 0.51);
    check_reproducible(&d, 114);
}

#[test]
fn geometric_edge_probabilities() {
    let ctx = Context::new();
    // p = 1: the first trial always succeeds.
    let sure = Distribution::geometric(ctx.int(1));
    assert!(draw(&sure, 100, 115).iter().all(|v| *v == 1.0));
    // A tiny p: mean 1000, and `ln(1−p)` through `ln_1p` keeps the mean.
    let rare = Distribution::geometric(ctx.rational(1, 1000));
    let s = draw(&rare, N, 116);
    check_moments(&rare, &s);
    check_support(&rare, &s);
}

// ── Negative binomial ────────────────────────────────────────────────────

#[test]
fn negative_binomial_as_a_poisson_gamma_mixture() {
    // NegativeBinomial(3, 2/5) on 0..∞: mean 9/2, variance 45/4;
    // pmf(2) = pmf(3) = 0.13824 (scipy: stats.nbinom.pmf(3, 3, 0.4)).
    let ctx = Context::new();
    let d = Distribution::negative_binomial(ctx.int(3), ctx.rational(2, 5));
    let s = draw(&d, N_SMALL, 117);
    check_moments(&d, &s);
    check_support(&d, &s);
    check_pmf_at(&d, &s, 3);
    check_pmf_at(&d, &s, 0);
    check_reproducible(&d, 117);
}

#[test]
fn negative_binomial_with_a_non_integer_r() {
    // r = 5/2, p = 2/5: mean 15/4, variance 75/8 (scipy: nbinom.mean(2.5, 0.4)
    // = 3.75, nbinom.var(2.5, 0.4) = 9.375) — the mixture needs no integer r.
    let ctx = Context::new();
    let d = Distribution::negative_binomial(ctx.rational(5, 2), ctx.rational(2, 5));
    assert_eq!(d.mean(), ctx.rational(15, 4));
    assert_eq!(d.variance(), ctx.rational(75, 8));
    let s = draw(&d, N_SMALL, 118);
    check_moments(&d, &s);
    check_support(&d, &s);
    // Small p sends the gamma rate above 30, so both Poisson branches run.
    let spread = Distribution::negative_binomial(ctx.int(4), ctx.rational(1, 20));
    let s = draw(&spread, N_SMALL, 119);
    check_moments(&spread, &s);
    check_support(&spread, &s);
}

#[test]
fn negative_binomial_with_r_one_is_a_shifted_geometric() {
    // NegativeBinomial(1, p) counts failures; Geometric(p) counts trials:
    // same law up to the shift by one.
    let ctx = Context::new();
    let nb = Distribution::negative_binomial(ctx.int(1), ctx.rational(3, 10));
    let g = Distribution::geometric(ctx.rational(3, 10));
    let s = draw(&nb, N, 120);
    let shifted: Vec<f64> = s.iter().map(|v| v + 1.0).collect();
    check_moments(&g, &shifted);
    check_pmf_at(&g, &shifted, 1);
}

// ── errors and composition ───────────────────────────────────────────────

#[test]
fn symbolic_or_invalid_parameters_are_reported_when_the_sampler_is_built() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let symbolic = [
        Distribution::gamma(k.clone(), ctx.int(1)),
        Distribution::chi_squared(k.clone()),
        Distribution::beta(k.clone(), ctx.int(2)),
        Distribution::student_t(k.clone()),
        Distribution::f_distribution(ctx.int(3), k.clone()),
        Distribution::poisson(k.clone()),
        Distribution::geometric(k.clone()),
        Distribution::negative_binomial(ctx.int(2), k.clone()),
    ];
    for d in &symbolic {
        assert!(
            d.sampler().is_err(),
            "{d}: a symbolic parameter must not sample"
        );
        assert!(d.sample(1, &mut Rng::new(1)).is_err(), "{d}");
    }
    // The unchecked constructors accept anything; the sampler does not.
    let invalid = [
        Distribution::gamma(ctx.int(-1), ctx.int(1)),
        Distribution::gamma(ctx.int(1), ctx.int(0)),
        Distribution::beta(ctx.int(0), ctx.int(2)),
        Distribution::poisson(ctx.int(0)),
        Distribution::geometric(ctx.int(0)),
        Distribution::geometric(ctx.int(2)),
        Distribution::negative_binomial(ctx.int(2), ctx.int(0)),
    ];
    for d in &invalid {
        assert!(
            matches!(d.sampler(), Err(SymplexError::InvalidArgument { .. })),
            "{d}: a non-positive parameter must be rejected"
        );
    }
}

#[test]
fn wrappers_compose_with_the_new_samplers() -> Result<(), SymplexError> {
    let ctx = Context::new();
    // Gamma(2, 1) | X > 1 samples by rejection from the inner sampler.
    let t =
        Distribution::gamma(ctx.int(2), ctx.int(1)).truncated(&Support::half_line(ctx.int(1)))?;
    let s = draw(&t, N, 121);
    assert!(s.iter().all(|v| *v > 1.0));
    let m = moments(&s);
    // E[X | X > 1] = (∫₁^∞ x²e^{−x} dx) / (∫₁^∞ xe^{−x} dx) = 5e⁻¹/2e⁻¹ = 5/2.
    let mean = f64_of(&t.mean());
    assert!((mean - 2.5).abs() < 1e-12, "truncated mean {mean}");
    assert!((m.mean - mean).abs() <= 4.0 * f64_of(&t.variance()).sqrt() / (N as f64).sqrt());
    // 10 − Poisson(3) maps inner draws (a lattice affine map has slope ±1).
    let a = Distribution::poisson(ctx.int(3)).affine(ctx.int(-1), ctx.int(10))?;
    let s = draw(&a, N, 122);
    assert!(s.iter().all(|v| *v <= 10.0 && v.fract() == 0.0));
    check_moments(&a, &s);
    // A mixture picks a component, then samples it.
    let mix = Distribution::mixture(&[
        (
            ctx.rational(1, 3),
            Distribution::beta(ctx.int(2), ctx.int(3)),
        ),
        (ctx.rational(2, 3), Distribution::chi_squared(ctx.int(4))),
    ])?;
    let s = draw(&mix, N, 123);
    check_moments(&mix, &s);
    Ok(())
}
