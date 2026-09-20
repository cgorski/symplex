//! symplex 0.11 — continuous distribution families (`symplex::stats`).
//!
//! Reference values are from SymPy 1.14 (`sympy.stats`), computed with
//! e.g.
//!
//! ```text
//! .venv/bin/python -c "import sympy.stats as st; from sympy import *; \
//!   x,t,p=symbols('x t p'); X=st.Exponential('X', 3); \
//!   print(st.E(X), st.variance(X), st.cdf(X)(x), st.moment_generating_function(X)(t), \
//!         st.quantile(X)(p), st.P(X>1), st.E(X**3), st.skewness(X), st.kurtosis(X))"
//! ```
//!
//! and quoted next to each assertion.  Where SymPy is exact the assertion
//! is exact (`Ex::equals == Some(true)`); where SymPy itself only gives a
//! number (or fails — `E(LogNormal)`, `E(StudentT)` raise in 1.14) the
//! textbook closed form is cited and checked numerically to 1e-9.

use symplex::prelude::*;
use symplex::stats::{Distribution, RandomVariable, Rng, Support};

// ── helpers ──────────────────────────────────────────────────────────────

fn assert_exact(actual: &Ex, expected: &Ex, label: &str) {
    assert_eq!(
        actual.equals(expected),
        Some(true),
        "{label}: got `{actual}`, expected `{expected}`"
    );
}

fn assert_close(actual: &Ex, expected: f64, label: &str) {
    let v = actual
        .eval_f64()
        .unwrap_or_else(|e| panic!("{label}: `{actual}` does not evaluate: {e}"));
    assert!(
        (v - expected).abs() <= 1e-9 * expected.abs().max(1.0),
        "{label}: got {v} (`{actual}`), expected {expected}"
    );
}

/// `∫ density` over the whole support, through the crate's exact
/// integrator (not `expectation(1)`, which is `1` by the polynomial route
/// without touching the density).
fn total_mass(rv: &RandomVariable) -> Ex {
    let ctx = rv.context();
    let x = rv.symbol();
    let (lo, hi) = match rv.support() {
        Support::Continuous { lo, hi } => (lo, hi),
        other => panic!("continuous family with support {other:?}"),
    };
    let lo = lo.unwrap_or_else(|| ctx.neg_infinity());
    let hi = hi.unwrap_or_else(|| ctx.infinity());
    rv.density(x).integrate_definite(x, &lo, &hi).simplify()
}

/// `cdf(quantile(p)) = p` numerically at `p = 0.3`.
fn assert_cdf_quantile_roundtrip(rv: &RandomVariable) {
    let ctx = rv.context();
    let p = ctx.rational(3, 10);
    let q = rv
        .quantile(&p)
        .unwrap_or_else(|| panic!("{}: no quantile", rv.distribution().name()));
    let back = rv.cdf(&q);
    assert_close(
        &back,
        0.3,
        &format!("{}: cdf(quantile(0.3))", rv.distribution().name()),
    );
}

/// Mean of 20 000 samples from `Rng::new(1)` within three standard errors
/// of the exact mean.
fn assert_sample_mean(rv: &RandomVariable) {
    let n = 20_000;
    let samples = rv.sample(n, &mut Rng::new(1)).expect("sampling");
    assert_eq!(samples.len(), n);
    let mean: f64 = samples.iter().sum::<f64>() / n as f64;
    let exact_mean = rv.mean().eval_f64().expect("mean");
    let sd = rv.variance().eval_f64().expect("variance").sqrt();
    let se = sd / (n as f64).sqrt();
    assert!(
        (mean - exact_mean).abs() <= 3.0 * se,
        "{}: sample mean {mean} vs exact {exact_mean} (3 se = {})",
        rv.distribution().name(),
        3.0 * se
    );
}

/// Composite Simpson's rule.
fn simpson(f: impl Fn(f64) -> f64, lo: f64, hi: f64, n: usize) -> f64 {
    let n = n + n % 2;
    let h = (hi - lo) / n as f64;
    let mut acc = f(lo) + f(hi);
    for i in 1..n {
        let w = if i % 2 == 1 { 4.0 } else { 2.0 };
        acc += w * f(lo + i as f64 * h);
    }
    acc * h / 3.0
}

/// Midpoint rule (tolerates integrable log singularities at the ends).
fn midpoint(f: impl Fn(f64) -> f64, lo: f64, hi: f64, n: usize) -> f64 {
    let h = (hi - lo) / n as f64;
    (0..n).map(|i| f(lo + (i as f64 + 0.5) * h)).sum::<f64>() * h
}

// ── Exponential ──────────────────────────────────────────────────────────

#[test]
fn exponential_rate_3() {
    // SymPy: E 1/3, Var 1/9, cdf 1 − exp(−3x), mgf 3/(3 − t),
    // quantile −log(1 − p)/3, P(X>1) exp(−3), E[X³] 2/9, skew 2, kurt 9.
    let ctx = Context::new();
    let x = RandomVariable::new(
        &ctx,
        "X",
        Distribution::try_exponential(ctx.int(3)).unwrap(),
    );
    let s = x.symbol().clone();
    let v = ctx.symbol("v");
    let t = ctx.symbol("t");
    let p = ctx.symbol("p");
    assert_exact(&x.mean(), &ctx.rational(1, 3), "mean");
    assert_exact(&x.variance(), &ctx.rational(1, 9), "variance");
    assert_exact(&x.moment(3), &ctx.rational(2, 9), "E[X³]");
    // E[X² + 3X] = 2/9 + 1 = 11/9 through the closed-form moments.
    assert_exact(
        &x.expectation(&(s.powi(2) + 3 * &s)),
        &ctx.rational(11, 9),
        "E[X² + 3X]",
    );
    assert_exact(&x.skewness(), &ctx.int(2), "skewness");
    assert_exact(&x.kurtosis(), &ctx.int(9), "kurtosis");
    assert_exact(&x.cdf(&v), &(ctx.one() - (-3 * &v).exp()), "cdf");
    assert_exact(&x.mgf(&t), &(ctx.int(3) / (ctx.int(3) - &t)), "mgf");
    assert_exact(
        &x.quantile(&p).unwrap(),
        &(-(ctx.one() - &p).ln() / 3),
        "quantile",
    );
    assert_cdf_quantile_roundtrip(&x);
    assert_exact(
        &x.probability(&s.gt(&ctx.one())).unwrap(),
        &(-ctx.int(3)).exp(),
        "P(X > 1)",
    );
    assert_exact(&total_mass(&x), &ctx.one(), "∫ density");
    assert_exact(
        &x.entropy_closed_form(),
        &(ctx.one() - ctx.int(3).ln()),
        "entropy",
    );
}

// ── Uniform ──────────────────────────────────────────────────────────────

#[test]
fn uniform_0_1() {
    // SymPy: E 1/2, Var 1/12, cdf x on [0, 1], mgf (exp(t) − 1)/t,
    // E[X³] 1/4, skew 0, kurt 9/5, P(0 < U < 1/4) 1/4.
    let ctx = Context::new();
    let u = RandomVariable::new(
        &ctx,
        "U",
        Distribution::try_uniform(ctx.int(0), ctx.int(1)).unwrap(),
    );
    let s = u.symbol().clone();
    let v = ctx.symbol("v");
    let t = ctx.symbol("t");
    let p = ctx.symbol("p");
    assert_exact(&u.mean(), &ctx.rational(1, 2), "mean");
    assert_exact(&u.variance(), &ctx.rational(1, 12), "variance");
    assert_exact(&u.moment(3), &ctx.rational(1, 4), "E[X³]");
    assert_exact(&u.skewness(), &ctx.zero(), "skewness");
    assert_exact(&u.kurtosis(), &ctx.rational(9, 5), "kurtosis");
    assert_exact(&u.cdf(&v), &v, "cdf");
    assert_exact(&u.mgf(&t), &((t.exp() - 1) / &t), "mgf");
    assert_exact(&u.quantile(&p).unwrap(), &p, "quantile");
    assert_cdf_quantile_roundtrip(&u);
    assert_exact(
        &u.probability(&s.gt(&ctx.zero()).and(&s.lt(&ctx.rational(1, 4))))
            .unwrap(),
        &ctx.rational(1, 4),
        "P(0 < U < 1/4)",
    );
    assert_exact(&total_mass(&u), &ctx.one(), "∫ density");
    assert_exact(&u.entropy_closed_form(), &ctx.zero(), "entropy ln(1) = 0");
}

#[test]
fn uniform_2_5() {
    // SymPy: E 7/2, Var 3/4, cdf x/3 − 2/3, mgf (exp(5t) − exp(2t))/(3t),
    // E[X³] 203/4, P(U > 3) 2/3.
    let ctx = Context::new();
    let u = RandomVariable::new(&ctx, "U", Distribution::uniform(ctx.int(2), ctx.int(5)));
    let s = u.symbol().clone();
    let v = ctx.symbol("v");
    let t = ctx.symbol("t");
    let p = ctx.symbol("p");
    assert_exact(&u.mean(), &ctx.rational(7, 2), "mean");
    assert_exact(&u.variance(), &ctx.rational(3, 4), "variance");
    assert_exact(&u.moment(3), &ctx.rational(203, 4), "E[X³]");
    assert_exact(&u.cdf(&v), &(&v / 3 - ctx.rational(2, 3)), "cdf");
    assert_exact(
        &u.mgf(&t),
        &(((5 * &t).exp() - (2 * &t).exp()) / (3 * &t)),
        "mgf",
    );
    assert_exact(&u.quantile(&p).unwrap(), &(3 * &p + 2), "quantile");
    assert_exact(&u.median().unwrap(), &ctx.rational(7, 2), "median");
    assert_exact(
        &u.probability(&s.gt(&ctx.int(3))).unwrap(),
        &ctx.rational(2, 3),
        "P(U > 3)",
    );
    assert_exact(&total_mass(&u), &ctx.one(), "∫ density");
    assert_exact(&u.entropy_closed_form(), &ctx.int(3).ln(), "entropy ln 3");
}

// ── Gamma ────────────────────────────────────────────────────────────────

#[test]
fn gamma_shape_3_scale_2() {
    // SymPy Gamma(3, 2): E 6, Var 12, cdf 1 − (x²/8 + x/2 + 1) exp(−x/2),
    // cdf(4) 1 − 5 exp(−2), mgf (1 − 2t)^(−3), E[X³] 480, skew 2√3/3,
    // kurt 5, P(G > 1) 13 exp(−1/2)/8, no closed quantile.
    let ctx = Context::new();
    let g = RandomVariable::new(
        &ctx,
        "G",
        Distribution::try_gamma(ctx.int(3), ctx.int(2)).unwrap(),
    );
    let s = g.symbol().clone();
    let v = ctx.symbol("v");
    let t = ctx.symbol("t");
    assert_exact(&g.mean(), &ctx.int(6), "mean");
    assert_exact(&g.variance(), &ctx.int(12), "variance");
    assert_exact(&g.moment(3), &ctx.int(480), "E[X³]");
    assert_exact(&g.central_moment(2), &ctx.int(12), "E[(X − 6)²]");
    assert_exact(&g.skewness(), &(2 * ctx.int(3).sqrt() / 3), "skewness");
    assert_exact(&g.kurtosis(), &ctx.int(5), "kurtosis");
    let cdf = g.cdf(&v);
    assert_exact(
        &cdf,
        &(ctx.one() - (v.powi(2) / 8 + &v / 2 + 1) * (-&v / 2).exp()),
        "cdf",
    );
    assert_exact(
        &g.cdf(&ctx.int(4)),
        &(ctx.one() - 5 * (-ctx.int(2)).exp()),
        "cdf(4)",
    );
    assert_exact(&g.mgf(&t), &(ctx.one() - 2 * &t).powi(-3), "mgf");
    assert!(
        g.quantile(&ctx.rational(1, 2)).is_none(),
        "no closed quantile"
    );
    assert_exact(
        &g.probability(&s.gt(&ctx.one())).unwrap(),
        &(ctx.rational(13, 8) * (-ctx.rational(1, 2)).exp()),
        "P(G > 1)",
    );
    assert_exact(&total_mass(&g), &ctx.one(), "∫ density");
}

#[test]
fn gamma_symbolic_shape_moments_use_rising_factorial() {
    // E[Xⁿ] = θⁿ (k)ₙ = θⁿ Γ(k+n)/Γ(k).
    let ctx = Context::new();
    let k = ctx.symbol_with("k", &[Assumption::Positive]);
    let th = ctx.symbol_with("theta", &[Assumption::Positive]);
    let g = RandomVariable::new(&ctx, "G", Distribution::gamma(k.clone(), th.clone()));
    assert_exact(&g.mean(), &(&k * &th), "mean kθ");
    assert_exact(&g.variance(), &(&k * th.powi(2)), "variance kθ²");
    let m2 = g.moment(2);
    // k(k+1)θ² at k = 3, θ = 2 is 3·4·4 = 48.
    assert_exact(
        &m2.subs(&k, &ctx.int(3)).subs(&th, &ctx.int(2)).eval(),
        &ctx.int(48),
        "E[X²] at k=3, θ=2",
    );
}

// ── ChiSquared ───────────────────────────────────────────────────────────

#[test]
fn chi_squared_4_dof() {
    // SymPy ChiSquared(4): E 4, Var 8, cdf 1 − (x/2 + 1) exp(−x/2),
    // cdf(2) 1 − 2 exp(−1), mgf (1 − 2t)^(−2), E[X³] 192, skew √2, kurt 6,
    // P(C < 2) 1 − 2 exp(−1).
    let ctx = Context::new();
    let c = RandomVariable::new(
        &ctx,
        "C",
        Distribution::try_chi_squared(ctx.int(4)).unwrap(),
    );
    let s = c.symbol().clone();
    let v = ctx.symbol("v");
    let t = ctx.symbol("t");
    assert_exact(&c.mean(), &ctx.int(4), "mean");
    assert_exact(&c.variance(), &ctx.int(8), "variance");
    assert_exact(&c.moment(3), &ctx.int(192), "E[X³]");
    assert_exact(&c.skewness(), &ctx.int(2).sqrt(), "skewness");
    assert_exact(&c.kurtosis(), &ctx.int(6), "kurtosis");
    assert_exact(
        &c.cdf(&v),
        &(ctx.one() - (&v / 2 + 1) * (-&v / 2).exp()),
        "cdf",
    );
    let two_over_e = 2 * (-ctx.one()).exp();
    assert_exact(&c.cdf(&ctx.int(2)), &(ctx.one() - &two_over_e), "cdf(2)");
    assert_exact(&c.mgf(&t), &(ctx.one() - 2 * &t).powi(-2), "mgf");
    assert!(
        c.quantile(&ctx.rational(1, 2)).is_none(),
        "no closed quantile"
    );
    assert_exact(
        &c.probability(&s.lt(&ctx.int(2))).unwrap(),
        &(ctx.one() - &two_over_e),
        "P(C < 2)",
    );
    assert_exact(&total_mass(&c), &ctx.one(), "∫ density");
    // Same density as Gamma(2, 2).
    let g = Distribution::gamma(ctx.int(2), ctx.int(2));
    assert_exact(&c.density(&v), &g.density(&v), "density = Gamma(2, 2)");
}

// ── Beta ─────────────────────────────────────────────────────────────────

#[test]
fn beta_2_3() {
    // SymPy Beta(2, 3): E 2/5, Var 1/25, cdf 3x⁴ − 8x³ + 6x² (from the
    // generic integral: the regularised incomplete beta is not in the
    // crate), E[X³] 4/35, E[X⁴] 1/14, skew 2/7, kurt 33/14, P(B < 1/2) 11/16;
    // no mgf (₁F₁) and no closed quantile.
    let ctx = Context::new();
    let b = RandomVariable::new(
        &ctx,
        "B",
        Distribution::try_beta(ctx.int(2), ctx.int(3)).unwrap(),
    );
    let s = b.symbol().clone();
    let v = ctx.symbol("v");
    assert_exact(&b.mean(), &ctx.rational(2, 5), "mean");
    assert_exact(&b.variance(), &ctx.rational(1, 25), "variance");
    assert_exact(&b.moment(3), &ctx.rational(4, 35), "E[X³]");
    assert_exact(&b.moment(4), &ctx.rational(1, 14), "E[X⁴]");
    assert_exact(&b.skewness(), &ctx.rational(2, 7), "skewness");
    assert_exact(&b.kurtosis(), &ctx.rational(33, 14), "kurtosis");
    assert!(
        b.distribution().cdf(&v).is_none(),
        "no closed-form cdf on the family"
    );
    assert_exact(
        &b.cdf(&v),
        &(3 * v.powi(4) - 8 * v.powi(3) + 6 * v.powi(2)),
        "cdf by integration",
    );
    assert!(
        b.quantile(&ctx.rational(1, 2)).is_none(),
        "no closed quantile"
    );
    assert_exact(
        &b.probability(&s.lt(&ctx.rational(1, 2))).unwrap(),
        &ctx.rational(11, 16),
        "P(B < 1/2)",
    );
    assert_exact(&total_mass(&b), &ctx.one(), "∫ density");
}

#[test]
fn beta_symbolic_moments_are_rising_factorial_ratios() {
    let ctx = Context::new();
    let a = ctx.symbol_with("alpha", &[Assumption::Positive]);
    let bb = ctx.symbol_with("beta", &[Assumption::Positive]);
    let b = RandomVariable::new(&ctx, "B", Distribution::beta(a.clone(), bb.clone()));
    assert_exact(&b.mean(), &(&a / (&a + &bb)), "mean α/(α+β)");
    // E[X²] = α(α+1)/((α+β)(α+β+1)); at α = 2, β = 3 that is 6/30 = 1/5.
    let m2 = b
        .moment(2)
        .subs(&a, &ctx.int(2))
        .subs(&bb, &ctx.int(3))
        .eval();
    assert_exact(&m2, &ctx.rational(1, 5), "E[X²] at α=2, β=3");
}

// ── Cauchy ───────────────────────────────────────────────────────────────

#[test]
fn cauchy_1_2_has_cdf_and_quantile_but_no_moments() {
    // SymPy Cauchy(1, 2): cdf atan(x/2 − 1/2)/π + 1/2, cdf(3) 3/4,
    // quantile 2 tan(π(p − 1/2)) + 1, P(Ca > 3) 1/4, ∫ density 1;
    // E, Var, moments undefined.
    let ctx = Context::new();
    let c = RandomVariable::new(
        &ctx,
        "Ca",
        Distribution::try_cauchy(ctx.int(1), ctx.int(2)).unwrap(),
    );
    let s = c.symbol().clone();
    let v = ctx.symbol("v");
    let p = ctx.symbol("p");
    assert!(c.distribution().mean(&ctx).is_none());
    assert!(c.distribution().variance(&ctx).is_none());
    assert!(c.distribution().raw_moment(1, &ctx).is_none());
    // The generic route hands back the divergent integral unevaluated…
    let mean = c.mean();
    assert!(
        mean.has_unevaluated(),
        "E[X] must not be a number, got `{mean}`"
    );
    assert!(
        c.variance().has_unevaluated(),
        "Var[X] must not be a number"
    );
    // …and `try_integrate_definite` names the reason.
    let integrand = &s * c.density(&s);
    assert!(matches!(
        integrand.try_integrate_definite(&s, &ctx.neg_infinity(), &ctx.infinity()),
        Err(SymplexError::Divergent { .. })
    ));
    assert_exact(
        &c.cdf(&v),
        &(ctx.rational(1, 2) + ((&v - 1) / 2).atan() / ctx.pi()),
        "cdf",
    );
    assert_exact(&c.cdf(&ctx.int(3)), &ctx.rational(3, 4), "cdf(3)");
    assert_exact(
        &c.quantile(&p).unwrap(),
        &(ctx.one() + 2 * (ctx.pi() * (&p - ctx.rational(1, 2))).tan()),
        "quantile",
    );
    assert_exact(&c.median().unwrap(), &ctx.one(), "median = x₀");
    assert_cdf_quantile_roundtrip(&c);
    assert_exact(
        &c.probability(&s.gt(&ctx.int(3))).unwrap(),
        &ctx.rational(1, 4),
        "P(Ca > 3)",
    );
    assert_exact(&total_mass(&c), &ctx.one(), "∫ density");
    assert!(c.mgf(&ctx.symbol("t")).has_unevaluated(), "no mgf");
}

// ── Laplace ──────────────────────────────────────────────────────────────

#[test]
fn laplace_1_2() {
    // SymPy Laplace(1, 2): E 1, Var 8, cdf(3) 1 − exp(−1)/2,
    // mgf exp(t)/(1 − 4t²), E[X³] 25, E[X⁴] 433, cmoment 4 = 384, skew 0,
    // kurt 6, P(L > 3) exp(−1)/2.  SymPy has no closed quantile; textbook:
    // μ − b sign(p − ½) ln(1 − 2|p − ½|).
    let ctx = Context::new();
    let l = RandomVariable::new(
        &ctx,
        "L",
        Distribution::try_laplace(ctx.int(1), ctx.int(2)).unwrap(),
    );
    let s = l.symbol().clone();
    let t = ctx.symbol("t");
    assert_exact(&l.mean(), &ctx.one(), "mean");
    assert_exact(&l.variance(), &ctx.int(8), "variance");
    assert_exact(&l.moment(3), &ctx.int(25), "E[X³]");
    assert_exact(&l.moment(4), &ctx.int(433), "E[X⁴]");
    assert_exact(&l.central_moment(4), &ctx.int(384), "E[(X − 1)⁴]");
    assert_exact(&l.skewness(), &ctx.zero(), "skewness");
    assert_exact(&l.kurtosis(), &ctx.int(6), "kurtosis");
    let half_e = (-ctx.one()).exp() / 2;
    assert_exact(
        &l.cdf(&ctx.int(3)).simplify(),
        &(ctx.one() - &half_e),
        "cdf(3)",
    );
    assert_exact(&l.cdf(&ctx.int(-1)).simplify(), &half_e, "cdf(−1)");
    assert_exact(&l.mgf(&t), &(t.exp() / (ctx.one() - 4 * t.powi(2))), "mgf");
    // quantile(0.3) = 1 + 2 ln(0.6)  (below the median, sign = −1).
    assert_exact(
        &l.quantile(&ctx.rational(3, 10)).unwrap().simplify(),
        &(ctx.one() + 2 * ctx.rational(3, 5).ln()),
        "quantile(0.3)",
    );
    assert_exact(&l.median().unwrap().simplify(), &ctx.one(), "median");
    assert_cdf_quantile_roundtrip(&l);
    assert_exact(
        &l.probability(&s.gt(&ctx.int(3))).unwrap(),
        &half_e,
        "P(L > 3)",
    );
    assert_exact(&total_mass(&l), &ctx.one(), "∫ density");
}

// ── Logistic ─────────────────────────────────────────────────────────────

#[test]
fn logistic_1_2() {
    // SymPy Logistic(1, 2): E 1, Var 4π²/3, cdf 1/(exp(1/2 − x/2) + 1),
    // cdf(3) 1/(exp(−1) + 1), mgf exp(t) B(1 − 2t, 2t + 1),
    // quantile 1 − 2 log(−1 + 1/p), P(Lg > 3) 1 − 1/(exp(−1) + 1).
    // Higher moments (SymPy 1.14 does not close them): even central
    // moments (2ᵏ − 2)|B_k|(πs)ᵏ, so E[X³] = 1 + 4π², E[X⁴] = 1 + 8π² + 112π⁴/15.
    let ctx = Context::new();
    let l = RandomVariable::new(
        &ctx,
        "Lg",
        Distribution::try_logistic(ctx.int(1), ctx.int(2)).unwrap(),
    );
    let s = l.symbol().clone();
    let v = ctx.symbol("v");
    let t = ctx.symbol("t");
    let p = ctx.symbol("p");
    let pi = ctx.pi();
    assert_exact(&l.mean(), &ctx.one(), "mean");
    assert_exact(&l.variance(), &(4 * pi.powi(2) / 3), "variance");
    assert_exact(&l.moment(3), &(ctx.one() + 4 * pi.powi(2)), "E[X³]");
    assert_exact(
        &l.moment(4),
        &(ctx.one() + 8 * pi.powi(2) + ctx.rational(112, 15) * pi.powi(4)),
        "E[X⁴]",
    );
    // Check E[X⁴] against quadrature of x⁴·density as well (independent of
    // the Bernoulli-number formula).
    let dens = l.density(&v).compile(&["v"]).unwrap();
    let m4 = simpson(|x| x.powi(4) * dens.call(&[x]), -200.0, 200.0, 100_000);
    let m4_exact = l.moment(4).eval_f64().unwrap();
    assert!(
        (m4 - m4_exact).abs() < 1e-6,
        "E[X⁴] quadrature {m4} vs {m4_exact}"
    );
    assert_exact(
        &l.cdf(&v),
        &(ctx.one() / ((ctx.rational(1, 2) - &v / 2).exp() + 1)),
        "cdf",
    );
    assert_exact(
        &l.mgf(&t),
        &(t.exp() * (ctx.one() - 2 * &t).beta(&(2 * &t + 1))),
        "mgf",
    );
    // μ + s ln(p/(1−p)) is SymPy's 1 − 2 log(−1 + 1/p); the log identity is
    // not decided structurally, so compare at p = 0.3: 1 + 2 ln(3/7).
    assert_close(
        &l.quantile(&p).unwrap().subs(&p, &ctx.rational(3, 10)),
        1.0 + 2.0 * (3.0_f64 / 7.0).ln(),
        "quantile(0.3)",
    );
    assert_cdf_quantile_roundtrip(&l);
    assert_exact(
        &l.probability(&s.gt(&ctx.int(3))).unwrap(),
        &(ctx.one() - ctx.one() / ((-ctx.one()).exp() + 1)),
        "P(Lg > 3)",
    );
    assert_exact(&total_mass(&l), &ctx.one(), "∫ density");
}

// ── LogNormal ────────────────────────────────────────────────────────────

#[test]
fn log_normal_1_2() {
    // SymPy 1.14 `E(LogNormal)` raises (its mgf route); with Y ~ Normal(1, 2)
    // it gives E[e^Y] = e³, E[e^{2Y}] = e¹⁰ (up to erf + erfc = 1), and
    // Var = e⁶(e⁴ − 1).  cdf erf(√2 (log x − 1)/4)/2 + 1/2, cdf(3) ≈
    // 0.519662338497517; SymPy has no closed quantile — textbook
    // exp(μ + σ√2 erfinv(2p − 1)).
    let ctx = Context::new();
    let l = RandomVariable::new(
        &ctx,
        "LN",
        Distribution::try_log_normal(ctx.int(1), ctx.int(2)).unwrap(),
    );
    let v = ctx.symbol("v");
    let p = ctx.symbol("p");
    assert_exact(&l.mean(), &ctx.int(3).exp(), "mean e³");
    assert_exact(&l.moment(2), &ctx.int(10).exp(), "E[X²] = e¹⁰");
    assert_exact(
        &l.variance(),
        &(ctx.int(6).exp() * (ctx.int(4).exp() - 1)),
        "variance",
    );
    assert_exact(&l.moment(3), &ctx.int(21).exp(), "E[X³] = e^{3 + 18}");
    assert_exact(
        &l.cdf(&v),
        &(ctx.rational(1, 2) + (ctx.int(2).sqrt() * (v.ln() - 1) / 4).erf() / 2),
        "cdf",
    );
    assert_close(&l.cdf(&ctx.int(3)), 0.519662338497517, "cdf(3)");
    let q = l.quantile(&p).unwrap();
    assert_exact(
        &q,
        &(ctx.one() + 2 * ctx.int(2).sqrt() * (2 * &p - 1).erfinv()).exp(),
        "quantile",
    );
    assert_exact(
        &l.median().unwrap().simplify(),
        &ctx.one().exp(),
        "median e^μ",
    );
    assert_cdf_quantile_roundtrip(&l);
    assert!(l.distribution().mgf(&ctx.symbol("t")).is_none(), "no mgf");
    // The crate's exact integrator does not close ∫₀^∞ of this density, so
    // the generic `probability` stays an integral; the density does
    // integrate to 1 numerically.
    let mass = total_mass(&l);
    assert!(
        mass.has_unevaluated(),
        "expected an unevaluated integral, got `{mass}`"
    );
    let numeric = l
        .density(&v)
        .integrate_numeric(&v, &ctx.zero(), &ctx.infinity())
        .unwrap();
    assert!(
        (numeric - 1.0).abs() < 1e-9,
        "∫ density numerically = {numeric}"
    );
}

// ── StudentT ─────────────────────────────────────────────────────────────

#[test]
fn student_t_5_dof() {
    // SymPy 1.14 `E(StudentT)` raises (its mgf route); by direct
    // integration of its density 25√5/((x² + 5)³ B(1/2, 5/2)):
    // ∫ density = 3π/(8 B(1/2, 5/2)) = 1, E[X] 0, E[X²] 5π/(8B) = 5/3,
    // E[X⁴] 75π/(8B) = 25; cdf(1) ≈ 0.818391266175439.
    let ctx = Context::new();
    let t5 = RandomVariable::new(&ctx, "T", Distribution::try_student_t(ctx.int(5)).unwrap());
    let s = t5.symbol().clone();
    assert_exact(&t5.mean(), &ctx.zero(), "mean");
    assert_exact(&t5.variance(), &ctx.rational(5, 3), "variance ν/(ν − 2)");
    assert_exact(&t5.moment(3), &ctx.zero(), "E[X³]");
    assert_exact(&t5.moment(4), &ctx.int(25), "E[X⁴] = 3ν²/((ν−2)(ν−4))");
    assert_exact(&t5.kurtosis(), &ctx.int(9), "kurtosis 3(ν−2)/(ν−4)");
    assert!(
        t5.distribution().raw_moment(6, &ctx).is_none(),
        "E[X⁶] does not exist for ν = 5"
    );
    assert!(
        t5.distribution().cdf(&s).is_none(),
        "no closed-form cdf on the family"
    );
    assert!(
        t5.quantile(&ctx.rational(1, 2)).is_none(),
        "no closed quantile"
    );
    assert_exact(
        &t5.probability(&s.gt(&ctx.zero())).unwrap(),
        &ctx.rational(1, 2),
        "P(T > 0)",
    );
    assert_exact(&total_mass(&t5), &ctx.one(), "∫ density");
}

#[test]
fn student_t_5_cdf_closes_by_integration_for_odd_dof() {
    // SymPy: cdf(1) = 8√5 ₂F₁(1/2, 3; 3/2; −1/5)/(15π) + 1/2 ≈ 0.818391266175439;
    // the crate's integrator gives the elementary form
    // 7√5/(27π) + atan(√5/5)/π + 1/2.
    let ctx = Context::new();
    let t5 = RandomVariable::new(&ctx, "T", Distribution::student_t(ctx.int(5)));
    let c1 = t5.cdf(&ctx.int(1));
    assert!(!c1.has_unevaluated(), "cdf(1) should close, got `{c1}`");
    assert_close(&c1, 0.818391266175439, "cdf(1) by integration");
}

#[test]
fn student_t_5_sixth_moment_stays_an_unevaluated_integral() {
    let ctx = Context::new();
    let t5 = RandomVariable::new(&ctx, "T", Distribution::student_t(ctx.int(5)));
    let m6 = t5.moment(6);
    assert!(
        m6.has_unevaluated(),
        "E[X⁶] must not be a number, got `{m6}`"
    );
}

#[test]
fn student_t_low_dof_has_no_mean_or_variance() {
    let ctx = Context::new();
    let t1 = Distribution::student_t(ctx.int(1));
    assert!(t1.mean(&ctx).is_none(), "ν = 1 (Cauchy) has no mean");
    let t2 = Distribution::student_t(ctx.int(2));
    assert_eq!(t2.mean(&ctx), Some(ctx.zero()));
    assert!(t2.variance(&ctx).is_none(), "ν = 2 has infinite variance");
    let nu = ctx.symbol_with("nu", &[Assumption::Positive]);
    let ts = Distribution::student_t(nu.clone());
    assert_exact(
        &ts.variance(&ctx).unwrap(),
        &(&nu / (&nu - 2)),
        "symbolic ν/(ν − 2)",
    );
}

// ── Weibull ──────────────────────────────────────────────────────────────

#[test]
fn weibull_scale_2_shape_3() {
    // SymPy Weibull(alpha=2, beta=3) (alpha = scale, beta = shape): density
    // 3x² exp(−x³/8)/8, E 2Γ(1/3)/3, Var 8Γ(2/3)/3 − 4Γ(1/3)²/9,
    // cdf 1 − exp(−x³/8), cdf(1) 1 − exp(−1/8), quantile 2(−log(1−p))^{1/3}
    // (SymPy writes it with Abs/sign), P(W > 1) exp(−1/8), E[W³] 8,
    // E[W²] 8Γ(2/3)/3.
    let ctx = Context::new();
    let w = RandomVariable::new(
        &ctx,
        "W",
        Distribution::try_weibull(ctx.int(2), ctx.int(3)).unwrap(),
    );
    let s = w.symbol().clone();
    let v = ctx.symbol("v");
    let p = ctx.symbol("p");
    let g13 = ctx.rational(1, 3).gamma();
    let g23 = ctx.rational(2, 3).gamma();
    assert_exact(
        &w.density(&v),
        &(3 * v.powi(2) * (-v.powi(3) / 8).exp() / 8),
        "density",
    );
    assert_exact(&w.mean(), &(2 * &g13 / 3), "mean 2Γ(1/3)/3");
    assert_exact(
        &w.variance(),
        &(8 * &g23 / 3 - 4 * g13.powi(2) / 9),
        "variance",
    );
    assert_exact(&w.moment(2), &(8 * &g23 / 3), "E[W²]");
    assert_exact(&w.moment(3), &ctx.int(8), "E[W³] = λ³ Γ(2) = 8");
    assert_exact(&w.cdf(&v), &(ctx.one() - (-v.powi(3) / 8).exp()), "cdf");
    assert_exact(
        &w.cdf(&ctx.one()),
        &(ctx.one() - (-ctx.rational(1, 8)).exp()),
        "cdf(1)",
    );
    assert_exact(
        &w.quantile(&p).unwrap(),
        &(2 * (-(ctx.one() - &p).ln()).pow(&ctx.rational(1, 3))),
        "quantile",
    );
    assert_cdf_quantile_roundtrip(&w);
    assert_exact(
        &w.probability(&s.gt(&ctx.one())).unwrap(),
        &(-ctx.rational(1, 8)).exp(),
        "P(W > 1)",
    );
    assert_exact(&total_mass(&w), &ctx.one(), "∫ density");
    assert!(
        w.distribution().mgf(&ctx.symbol("t")).is_none(),
        "no closed mgf"
    );
}

// ── Pareto ───────────────────────────────────────────────────────────────

#[test]
fn pareto_xm_1_alpha_3() {
    // SymPy Pareto(1, 3): density 3/x⁴, E 3/2, Var 3/4, cdf 1 − 1/x³,
    // cdf(2) 7/8, quantile (1 − p)^{−1/3} (SymPy writes sign/Abs),
    // P(Pa > 2) 1/8, E[X²] 3; E[X³] does not exist (α = 3).
    // Pareto(2, 5): E 5/2, Var 5/12, E[X³] 20, E[X⁴] 80.
    let ctx = Context::new();
    let pa = RandomVariable::new(
        &ctx,
        "Pa",
        Distribution::try_pareto(ctx.int(1), ctx.int(3)).unwrap(),
    );
    let s = pa.symbol().clone();
    let v = ctx.symbol("v");
    let p = ctx.symbol("p");
    assert_exact(&pa.density(&v), &(3 / v.powi(4)), "density");
    assert_exact(&pa.mean(), &ctx.rational(3, 2), "mean");
    assert_exact(&pa.variance(), &ctx.rational(3, 4), "variance");
    assert_exact(&pa.moment(2), &ctx.int(3), "E[X²]");
    assert!(
        pa.distribution().raw_moment(3, &ctx).is_none(),
        "E[X³] does not exist for α = 3"
    );
    assert!(pa.moment(3).has_unevaluated(), "E[X³] must not be a number");
    assert_exact(&pa.cdf(&v), &(ctx.one() - v.powi(-3)), "cdf");
    assert_exact(&pa.cdf(&ctx.int(2)), &ctx.rational(7, 8), "cdf(2)");
    assert_exact(
        &pa.quantile(&p).unwrap(),
        &(ctx.one() - &p).pow(&ctx.rational(-1, 3)),
        "quantile",
    );
    assert_cdf_quantile_roundtrip(&pa);
    assert_exact(
        &pa.probability(&s.gt(&ctx.int(2))).unwrap(),
        &ctx.rational(1, 8),
        "P(Pa > 2)",
    );
    assert_exact(&total_mass(&pa), &ctx.one(), "∫ density");

    let pa2 = RandomVariable::new(&ctx, "Pb", Distribution::pareto(ctx.int(2), ctx.int(5)));
    assert_exact(&pa2.mean(), &ctx.rational(5, 2), "Pareto(2,5) mean");
    assert_exact(
        &pa2.variance(),
        &ctx.rational(5, 12),
        "Pareto(2,5) variance",
    );
    assert_exact(&pa2.moment(3), &ctx.int(20), "Pareto(2,5) E[X³]");
    assert_exact(&pa2.moment(4), &ctx.int(80), "Pareto(2,5) E[X⁴]");
    // Symbolic α: the formula is returned (valid for n < α).
    let alpha = ctx.symbol_with("alpha", &[Assumption::Positive]);
    let ps = Distribution::pareto(ctx.one(), alpha.clone());
    assert_exact(
        &ps.raw_moment(2, &ctx).unwrap(),
        &(&alpha / (&alpha - 2)),
        "α/(α − 2)",
    );
}

// ── Triangular ───────────────────────────────────────────────────────────

#[test]
fn triangular_0_4_mode_1() {
    // SymPy Triangular(0, 4, 1): E 5/3, Var 13/18, cdf(1/2) 1/16, cdf(2) 2/3,
    // mgf (2 exp(4t) − 8 exp(t) + 6)/(12t²), E[X³] 17/2, P(Tr > 2) 1/3.
    let ctx = Context::new();
    let tr = RandomVariable::new(
        &ctx,
        "Tr",
        Distribution::try_triangular(ctx.int(0), ctx.int(4), ctx.int(1)).unwrap(),
    );
    let s = tr.symbol().clone();
    let t = ctx.symbol("t");
    assert_exact(&tr.mean(), &ctx.rational(5, 3), "mean");
    assert_exact(&tr.variance(), &ctx.rational(13, 18), "variance");
    assert_exact(&tr.moment(3), &ctx.rational(17, 2), "E[X³]");
    assert_exact(
        &tr.cdf(&ctx.rational(1, 2)).simplify(),
        &ctx.rational(1, 16),
        "cdf(1/2)",
    );
    assert_exact(
        &tr.cdf(&ctx.int(2)).simplify(),
        &ctx.rational(2, 3),
        "cdf(2)",
    );
    assert_exact(
        &tr.mgf(&t),
        &((2 * (4 * &t).exp() - 8 * t.exp() + 6) / (12 * t.powi(2))),
        "mgf",
    );
    // quantile: p = 1/16 is the left branch (√(p·4·1) = 1/2), p = 2/3 the right.
    assert_exact(
        &tr.quantile(&ctx.rational(1, 16)).unwrap().simplify(),
        &ctx.rational(1, 2),
        "quantile(1/16)",
    );
    assert_exact(
        &tr.quantile(&ctx.rational(2, 3)).unwrap().simplify(),
        &ctx.int(2),
        "quantile(2/3)",
    );
    assert_cdf_quantile_roundtrip(&tr);
    assert_exact(
        &tr.probability(&s.gt(&ctx.int(2))).unwrap(),
        &ctx.rational(1, 3),
        "P(Tr > 2)",
    );
    assert_exact(&total_mass(&tr), &ctx.one(), "∫ density");
}

// ── Sampling ─────────────────────────────────────────────────────────────

#[test]
fn sampling_exponential_mean_within_three_standard_errors() {
    let ctx = Context::new();
    let x = RandomVariable::new(&ctx, "X", Distribution::exponential(ctx.int(3)));
    assert_sample_mean(&x);
}

#[test]
fn sampling_uniform_mean_within_three_standard_errors() {
    let ctx = Context::new();
    let u = RandomVariable::new(&ctx, "U", Distribution::uniform(ctx.int(2), ctx.int(5)));
    assert_sample_mean(&u);
}

#[test]
fn sampling_weibull_and_logistic_mean_within_three_standard_errors() {
    let ctx = Context::new();
    let w = RandomVariable::new(&ctx, "W", Distribution::weibull(ctx.int(2), ctx.int(3)));
    assert_sample_mean(&w);
    let l = RandomVariable::new(&ctx, "L", Distribution::logistic(ctx.int(1), ctx.int(2)));
    assert_sample_mean(&l);
}

#[test]
fn sampling_needs_a_closed_quantile() {
    let ctx = Context::new();
    let g = RandomVariable::new(&ctx, "G", Distribution::gamma(ctx.int(3), ctx.int(2)));
    assert!(matches!(
        g.sample(10, &mut Rng::new(1)),
        Err(SymplexError::NotImplemented(_))
    ));
}

// ── Entropy ──────────────────────────────────────────────────────────────

/// Every closed-form differential entropy equals `E[−ln f(X)]`, computed
/// by quadrature: through the quantile (`−∫₀¹ ln f(Q(u)) du`) where the
/// family has one, else directly in `x`.
#[test]
fn entropy_closed_forms_match_quadrature() {
    let ctx = Context::new();
    let v = ctx.symbol("v");
    let u = ctx.symbol("u");
    let via_quantile: Vec<(&str, Distribution)> = vec![
        ("Uniform", Distribution::uniform(ctx.int(2), ctx.int(5))),
        ("Exponential", Distribution::exponential(ctx.int(3))),
        ("Cauchy", Distribution::cauchy(ctx.int(1), ctx.int(2))),
        ("Laplace", Distribution::laplace(ctx.int(1), ctx.int(2))),
        ("Logistic", Distribution::logistic(ctx.int(1), ctx.int(2))),
        ("Weibull", Distribution::weibull(ctx.int(2), ctx.int(3))),
        ("Pareto", Distribution::pareto(ctx.int(1), ctx.int(3))),
        (
            "Triangular",
            Distribution::triangular(ctx.int(0), ctx.int(4), ctx.int(1)),
        ),
    ];
    for (name, d) in via_quantile {
        let Distribution::Continuous(fam) = &d else {
            unreachable!()
        };
        let closed = fam.entropy(&ctx).unwrap().eval_f64().unwrap();
        let dens = d.density(&v).compile(&["v"]).unwrap();
        let q = d.quantile(&u).unwrap().compile(&["u"]).unwrap();
        let numeric = midpoint(|p| -dens.call(&[q.call(&[p])]).ln(), 0.0, 1.0, 50_000);
        assert!(
            (closed - numeric).abs() < 1e-3,
            "{name}: closed-form entropy {closed} vs quadrature {numeric}"
        );
    }
    // Direct quadrature in x for the families without a closed quantile
    // (and Normal / LogNormal, whose quantile needs erfinv); the
    // `f > 0` guard avoids 0·ln 0 in the tails.
    let direct: Vec<(&str, Distribution, f64, f64)> = vec![
        (
            "Normal",
            Distribution::normal(ctx.int(1), ctx.int(2)),
            -60.0,
            60.0,
        ),
        (
            "Gamma",
            Distribution::gamma(ctx.int(3), ctx.int(2)),
            0.0,
            200.0,
        ),
        (
            "ChiSquared",
            Distribution::chi_squared(ctx.int(4)),
            0.0,
            200.0,
        ),
        ("Beta", Distribution::beta(ctx.int(2), ctx.int(3)), 0.0, 1.0),
        (
            "StudentT",
            Distribution::student_t(ctx.int(5)),
            -400.0,
            400.0,
        ),
    ];
    for (name, d, lo, hi) in direct {
        let Distribution::Continuous(fam) = &d else {
            unreachable!()
        };
        let closed = fam.entropy(&ctx).unwrap().eval_f64().unwrap();
        let dens = d.density(&v).compile(&["v"]).unwrap();
        let numeric = simpson(
            |x| {
                let f = dens.call(&[x]);
                if f > 0.0 { -f * f.ln() } else { 0.0 }
            },
            lo,
            hi,
            50_000,
        );
        // (`−f ln f` has an infinite slope where `f → 0` at a finite end,
        // so Simpson is only good to ~1e-6 here.)
        assert!(
            (closed - numeric).abs() < 1e-5,
            "{name}: closed-form entropy {closed} vs quadrature {numeric}"
        );
    }
    // LogNormal(1, 2): substitute x = e^y, so the integrand is the
    // Normal(1, 2) density times −ln f(e^y).
    let ln = Distribution::log_normal(ctx.int(1), ctx.int(2));
    let Distribution::Continuous(fam) = &ln else {
        unreachable!()
    };
    let closed = fam.entropy(&ctx).unwrap().eval_f64().unwrap();
    let dens = ln.density(&v).compile(&["v"]).unwrap();
    let phi = Distribution::normal(ctx.int(1), ctx.int(2))
        .density(&v)
        .compile(&["v"])
        .unwrap();
    let numeric = simpson(
        |y| -phi.call(&[y]) * dens.call(&[y.exp()]).ln(),
        -40.0,
        42.0,
        50_000,
    );
    assert!(
        (closed - numeric).abs() < 1e-6,
        "LogNormal: closed-form entropy {closed} vs quadrature {numeric}"
    );
    // The three headline forms, exactly.
    let normal = Distribution::normal(ctx.int(0), ctx.int(2));
    let Distribution::Continuous(fam) = &normal else {
        unreachable!()
    };
    assert_exact(
        &fam.entropy(&ctx).unwrap(),
        &(ctx.rational(1, 2) * (8 * ctx.pi() * ctx.e()).ln()),
        "Normal(0, 2): ½ ln(2πe·4)",
    );
}

// ── Constructors ─────────────────────────────────────────────────────────

#[test]
fn try_constructors_reject_bad_numeric_parameters_and_accept_symbols() {
    let ctx = Context::new();
    let bad = |r: Result<Distribution, SymplexError>, what: &str| {
        assert!(
            matches!(r, Err(SymplexError::InvalidArgument { .. })),
            "{what} should be rejected"
        );
    };
    bad(
        Distribution::try_uniform(ctx.int(1), ctx.int(1)),
        "Uniform(1, 1)",
    );
    bad(
        Distribution::try_uniform(ctx.int(2), ctx.int(1)),
        "Uniform(2, 1)",
    );
    bad(Distribution::try_exponential(ctx.int(0)), "Exponential(0)");
    bad(
        Distribution::try_gamma(ctx.int(-1), ctx.int(2)),
        "Gamma(−1, 2)",
    );
    bad(
        Distribution::try_gamma(ctx.int(1), ctx.int(0)),
        "Gamma(1, 0)",
    );
    bad(Distribution::try_chi_squared(ctx.int(0)), "ChiSquared(0)");
    bad(Distribution::try_beta(ctx.int(0), ctx.int(1)), "Beta(0, 1)");
    bad(
        Distribution::try_cauchy(ctx.int(0), ctx.int(-2)),
        "Cauchy(0, −2)",
    );
    bad(
        Distribution::try_laplace(ctx.int(0), ctx.int(0)),
        "Laplace(0, 0)",
    );
    bad(
        Distribution::try_logistic(ctx.int(0), ctx.rational(-1, 2)),
        "Logistic(0, −1/2)",
    );
    bad(
        Distribution::try_log_normal(ctx.int(0), ctx.int(0)),
        "LogNormal(0, 0)",
    );
    bad(Distribution::try_student_t(ctx.int(-3)), "StudentT(−3)");
    bad(
        Distribution::try_weibull(ctx.int(0), ctx.int(1)),
        "Weibull(0, 1)",
    );
    bad(
        Distribution::try_pareto(ctx.int(1), ctx.int(0)),
        "Pareto(1, 0)",
    );
    bad(
        Distribution::try_triangular(ctx.int(0), ctx.int(4), ctx.int(5)),
        "Triangular(0, 4, 5)",
    );
    bad(
        Distribution::try_triangular(ctx.int(0), ctx.int(4), ctx.int(-1)),
        "Triangular(0, 4, −1)",
    );
    bad(
        Distribution::try_triangular(ctx.int(4), ctx.int(0), ctx.int(2)),
        "Triangular(4, 0, 2)",
    );

    // Symbolic parameters are the caller's promise.
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    assert!(Distribution::try_uniform(a.clone(), b.clone()).is_ok());
    assert!(Distribution::try_exponential(a.clone()).is_ok());
    assert!(Distribution::try_triangular(a.clone(), b.clone(), ctx.symbol("c")).is_ok());
    // Endpoints of the triangle may coincide with the mode.
    assert!(Distribution::try_triangular(ctx.int(0), ctx.int(4), ctx.int(0)).is_ok());
    assert!(Distribution::try_triangular(ctx.int(0), ctx.int(4), ctx.int(4)).is_ok());
}

#[test]
fn display_names_and_parameters() {
    let ctx = Context::new();
    let w = RandomVariable::new(&ctx, "W", Distribution::weibull(ctx.int(2), ctx.int(3)));
    assert_eq!(format!("{w}"), "W ~ Weibull(2, 3)");
    assert_eq!(w.distribution().name(), "Weibull");
    let tr = Distribution::triangular(ctx.int(0), ctx.int(4), ctx.int(1));
    assert_eq!(format!("{tr}"), "Triangular(0, 4, 1)");
    for (d, name) in [
        (Distribution::uniform(ctx.int(0), ctx.int(1)), "Uniform"),
        (Distribution::exponential(ctx.int(1)), "Exponential"),
        (Distribution::gamma(ctx.int(1), ctx.int(1)), "Gamma"),
        (Distribution::chi_squared(ctx.int(1)), "ChiSquared"),
        (Distribution::beta(ctx.int(1), ctx.int(1)), "Beta"),
        (Distribution::cauchy(ctx.int(0), ctx.int(1)), "Cauchy"),
        (Distribution::laplace(ctx.int(0), ctx.int(1)), "Laplace"),
        (Distribution::logistic(ctx.int(0), ctx.int(1)), "Logistic"),
        (
            Distribution::log_normal(ctx.int(0), ctx.int(1)),
            "LogNormal",
        ),
        (Distribution::student_t(ctx.int(1)), "StudentT"),
        (Distribution::pareto(ctx.int(1), ctx.int(1)), "Pareto"),
    ] {
        assert_eq!(d.name(), name);
        assert!(d.is_continuous());
        assert!(format!("{d}").starts_with(name), "{d}");
    }
}

// ── local extension: the family-level entropy through the variable ──────

trait EntropyExt {
    fn entropy_closed_form(&self) -> Ex;
}

impl EntropyExt for RandomVariable {
    /// `ContinuousFamily::entropy` reached through the variable
    /// (`RandomVariable::entropy` is not exposed by `rv.rs` yet).
    fn entropy_closed_form(&self) -> Ex {
        match self.distribution() {
            Distribution::Continuous(f) => f.entropy(self.context()).expect("closed-form entropy"),
            other => panic!("not continuous: {other}"),
        }
    }
}
