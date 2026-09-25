//! symplex 0.11 — discrete distribution families (`stats::DiscreteFamily`):
//! Bernoulli, Binomial, Poisson, Geometric, NegativeBinomial,
//! Hypergeometric, DiscreteUniform / Die.
//!
//! Reference values are from SymPy 1.14 (`sympy.stats`), e.g.
//!
//! ```python
//! import sympy.stats as st; from sympy import *
//! t, k = symbols('t k')
//! X = st.Poisson('X', 3)
//! st.E(X), st.variance(X), st.E(X**3), st.P(X <= 2), st.moment_generating_function(X)(t), st.skewness(X)
//! # (3, 3, 57, 17*exp(-3)/2, exp(3*exp(t) - 3), sqrt(3)/3)
//! ```
//!
//! Each value used below is cited next to its assertion.

use symplex::prelude::*;
use symplex::stats::{Distribution, RandomVariable, Rng, Support};

/// Numeric equality to 1e-9 relative (for results involving `e`, `√`).
fn assert_close(e: &Ex, expected: f64, label: &str) {
    let v = e
        .eval_f64()
        .unwrap_or_else(|err| panic!("{label}: `{e}` is not numeric: {err}"));
    assert!(
        (v - expected).abs() <= 1e-9 * expected.abs().max(1.0),
        "{label}: `{e}` = {v}, expected {expected}"
    );
}

/// `a` and `b` are the same polynomial (their difference expands to 0).
fn assert_poly_eq(a: &Ex, b: &Ex, label: &str) {
    let d = (a - b).expand().simplify();
    assert!(
        d.is_zero_structural(),
        "{label}: `{a}` ≠ `{b}` (difference `{d}`)"
    );
}

fn rv(ctx: &Context, name: &str, dist: Distribution) -> RandomVariable {
    RandomVariable::new(ctx, name, dist)
}

// ── Bernoulli(2/5) ─────────────────────────────────────────────────────
// SymPy: E = 2/5, variance = 6/25, E(B**3) = 2/5, P(B<=0) = 3/5,
// P(B>0) = 2/5, P(Eq(B,1)) = 2/5, mgf = 2*exp(t)/5 + 3/5, mgf(log 2) = 7/5,
// cdf = {0: 3/5, 1: 1}, skewness = sqrt(6)/6, E(B**2 + 3*B) = 8/5.

#[test]
fn bernoulli_moments() {
    let ctx = Context::new();
    let b = rv(&ctx, "B", Distribution::bernoulli(ctx.rational(2, 5)));
    assert_eq!(b.mean(), ctx.rational(2, 5));
    assert_eq!(b.variance(), ctx.rational(6, 25));
    assert_eq!(b.moment(3), ctx.rational(2, 5));
    let s = b.symbol();
    assert_eq!(b.expectation(&(s.powi(2) + 3 * s)), ctx.rational(8, 5));
    assert_close(&b.skewness(), 6f64.sqrt() / 6.0, "skewness(B)");
}

#[test]
fn bernoulli_probabilities() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let b = rv(&ctx, "B", Distribution::bernoulli(ctx.rational(2, 5)));
    let s = b.symbol();
    assert_eq!(b.probability(&s.le(&ctx.int(0)))?, ctx.rational(3, 5));
    assert_eq!(b.probability(&s.gt(&ctx.int(0)))?, ctx.rational(2, 5));
    assert_eq!(
        b.probability(&s.ge(&ctx.int(0)).and(&s.le(&ctx.int(1))))?,
        ctx.int(1)
    );
    assert_eq!(b.probability(&s.eq_expr(&ctx.int(1)))?, ctx.rational(2, 5));
    assert_eq!(b.probability(&s.ge(&ctx.int(0)))?, ctx.int(1), "total mass");
    Ok(())
}

#[test]
fn bernoulli_cdf_and_mgf() {
    let ctx = Context::new();
    let p = ctx.rational(2, 5);
    let b = rv(&ctx, "B", Distribution::bernoulli(p.clone()));
    let t = ctx.symbol("t");
    assert_eq!(b.mgf(&t), ctx.one() - &p + &p * t.exp());
    assert_eq!(b.mgf(&ctx.int(2).ln()).simplify(), ctx.rational(7, 5));
    assert_eq!(b.cdf(&ctx.int(0)).simplify(), ctx.rational(3, 5));
    assert_eq!(b.cdf(&ctx.int(1)).simplify(), ctx.int(1));
    assert_eq!(b.cdf(&ctx.int(-1)).simplify(), ctx.int(0));
    assert_eq!(b.cdf(&ctx.rational(1, 2)).simplify(), ctx.rational(3, 5));
}

// ── Binomial(5, 1/3) ───────────────────────────────────────────────────
// SymPy: E = 5/3, variance = 10/9, E(X**2) = 35/9, E(X**3) = 95/9,
// E(X**4) = 865/27, P(X<=2) = 64/81, P(X>2) = 17/81, P(1<=X<=3) = 200/243,
// P(Eq(X,2)) = 80/243, skewness = sqrt(10)/10, kurtosis = 27/10,
// mgf(log 2) = 1024/243, cdf(2) = 64/81, E(X**2 + 3*X) = 80/9.
// Symbolic (mgf derivatives): E(X**2) = n²p² − np² + np,
// E(X**3) = n³p³ − 3n²p³ + 3n²p² + 2np³ − 3np² + np,
// E(X**4) = n⁴p⁴ − 6n³p⁴ + 6n³p³ + 11n²p⁴ − 18n²p³ + 7n²p² − 6np⁴ + 12np³ − 7np² + np.

#[test]
fn binomial_moments_closed_form() {
    let ctx = Context::new();
    let x = rv(
        &ctx,
        "X",
        Distribution::binomial(ctx.int(5), ctx.rational(1, 3)),
    );
    assert_eq!(x.mean(), ctx.rational(5, 3));
    assert_eq!(x.variance(), ctx.rational(10, 9));
    assert_eq!(x.moment(2), ctx.rational(35, 9));
    assert_eq!(x.moment(3), ctx.rational(95, 9));
    assert_eq!(x.moment(4), ctx.rational(865, 27));
    assert_eq!(
        x.distribution().family().raw_moment(3),
        Some(ctx.rational(95, 9)),
        "the family has a closed form (no summation)"
    );
    let s = x.symbol();
    assert_eq!(x.expectation(&(s.powi(2) + 3 * s)), ctx.rational(80, 9));
    assert_close(&x.skewness(), 10f64.sqrt() / 10.0, "skewness");
    assert_eq!(x.kurtosis(), ctx.rational(27, 10));
}

#[test]
fn binomial_moments_symbolic_parameters() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let p = ctx.symbol("p");
    let x = rv(&ctx, "X", Distribution::binomial(n.clone(), p.clone()));
    let m2 = n.powi(2) * p.powi(2) - &n * p.powi(2) + &n * &p;
    assert_poly_eq(&x.moment(2), &m2, "E[X²]");
    let m3 = n.powi(3) * p.powi(3) - 3 * n.powi(2) * p.powi(3)
        + 3 * n.powi(2) * p.powi(2)
        + 2 * &n * p.powi(3)
        - 3 * &n * p.powi(2)
        + &n * &p;
    assert_poly_eq(&x.moment(3), &m3, "E[X³]");
    let m4 = n.powi(4) * p.powi(4) - 6 * n.powi(3) * p.powi(4)
        + 6 * n.powi(3) * p.powi(3)
        + 11 * n.powi(2) * p.powi(4)
        - 18 * n.powi(2) * p.powi(3)
        + 7 * n.powi(2) * p.powi(2)
        - 6 * &n * p.powi(4)
        + 12 * &n * p.powi(3)
        - 7 * &n * p.powi(2)
        + &n * &p;
    assert_poly_eq(&x.moment(4), &m4, "E[X⁴]");
}

#[test]
fn binomial_probabilities() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let x = rv(
        &ctx,
        "X",
        Distribution::binomial(ctx.int(5), ctx.rational(1, 3)),
    );
    let s = x.symbol();
    assert_eq!(x.probability(&s.le(&ctx.int(2)))?, ctx.rational(64, 81));
    assert_eq!(x.probability(&s.gt(&ctx.int(2)))?, ctx.rational(17, 81));
    assert_eq!(
        x.probability(&s.ge(&ctx.int(1)).and(&s.le(&ctx.int(3))))?,
        ctx.rational(200, 243)
    );
    assert_eq!(
        x.probability(&s.eq_expr(&ctx.int(2)))?,
        ctx.rational(80, 243)
    );
    assert_eq!(x.probability(&s.ge(&ctx.int(0)))?, ctx.int(1), "total mass");
    Ok(())
}

#[test]
fn binomial_cdf_and_mgf() {
    let ctx = Context::new();
    let x = rv(
        &ctx,
        "X",
        Distribution::binomial(ctx.int(5), ctx.rational(1, 3)),
    );
    // No closed-form cdf: the generic summation over 0..=2.
    assert_eq!(x.cdf(&ctx.int(2)).simplify(), ctx.rational(64, 81));
    assert_eq!(x.mgf(&ctx.int(2).ln()).simplify(), ctx.rational(1024, 243));
}

// ── Poisson(3) ─────────────────────────────────────────────────────────
// SymPy: E = 3, variance = 3, E(X**2) = 12, E(X**3) = 57, E(X**4) = 309,
// P(X<=2) = 17*exp(-3)/2, P(X>2) = (-17/2 + exp(3))*exp(-3),
// P(1<=X<=3) = 12*exp(-3), P(Eq(X,2)) = 9*exp(-3)/2,
// mgf = exp(3*exp(t) - 3), mgf(log 2) = exp(3), skewness = sqrt(3)/3,
// kurtosis = 10/3, E(2**X) = exp(3).
// Symbolic: E(X**3) = λ³ + 3λ² + λ, E(X**4) = λ⁴ + 6λ³ + 7λ² + λ.

#[test]
fn poisson_moments_touchard() {
    let ctx = Context::new();
    let x = rv(&ctx, "X", Distribution::poisson(ctx.int(3)));
    assert_eq!(x.mean(), ctx.int(3));
    assert_eq!(x.variance(), ctx.int(3));
    assert_eq!(x.moment(2), ctx.int(12));
    assert_eq!(x.moment(3), ctx.int(57));
    assert_eq!(x.moment(4), ctx.int(309));
    assert_close(&x.skewness(), 1.0 / 3f64.sqrt(), "skewness");
    assert_eq!(x.kurtosis(), ctx.rational(10, 3));

    let lam = ctx.symbol("lambda");
    let y = rv(&ctx, "Y", Distribution::poisson(lam.clone()));
    assert_poly_eq(
        &y.moment(3),
        &(lam.powi(3) + 3 * lam.powi(2) + &lam),
        "E[Y³]",
    );
    assert_poly_eq(
        &y.moment(4),
        &(lam.powi(4) + 6 * lam.powi(3) + 7 * lam.powi(2) + &lam),
        "E[Y⁴]",
    );
}

#[test]
fn poisson_touchard_matches_mgf_derivatives() {
    let ctx = Context::new();
    let lam = ctx.symbol_with("lambda", &[Assumption::Positive]).unwrap();
    let x = rv(&ctx, "X", Distribution::poisson(lam));
    let t = ctx.symbol("t");
    let mut m = x.mgf(&t);
    for n in 1..=4u32 {
        m = m.diff(&t);
        let via_mgf = m.subs(&t, &ctx.zero()).simplify();
        assert_poly_eq(&x.moment(n), &via_mgf, &format!("E[X^{n}] vs M^({n})(0)"));
    }
}

#[test]
fn poisson_probabilities() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let x = rv(&ctx, "X", Distribution::poisson(ctx.int(3)));
    let s = x.symbol();
    let e3 = (-3f64).exp();
    assert_close(&x.probability(&s.le(&ctx.int(2)))?, 8.5 * e3, "P(X ≤ 2)");
    assert_close(
        &x.probability(&s.gt(&ctx.int(2)))?,
        1.0 - 8.5 * e3,
        "P(X > 2)",
    );
    assert_close(
        &x.probability(&s.ge(&ctx.int(1)).and(&s.le(&ctx.int(3))))?,
        12.0 * e3,
        "P(1 ≤ X ≤ 3)",
    );
    assert_close(
        &x.probability(&s.eq_expr(&ctx.int(2)))?,
        4.5 * e3,
        "P(X = 2)",
    );
    // The infinite sum closes exactly: Σ 3ᵏe⁻³/k! = 1.
    assert_eq!(x.probability(&s.ge(&ctx.int(0)))?, ctx.int(1), "total mass");
    Ok(())
}

#[test]
fn poisson_cdf_mgf_and_nonpolynomial_expectation() {
    let ctx = Context::new();
    let x = rv(&ctx, "X", Distribution::poisson(ctx.int(3)));
    let k = ctx.symbol("k");
    let t = ctx.symbol("t");
    // Γ(⌊k⌋+1, 3)/⌊k⌋! at k = 2 is 17e⁻³/2.
    assert_close(&x.cdf(&ctx.int(2)), 8.5 * (-3f64).exp(), "cdf(2)");
    assert!(x.cdf(&k).contains(&k));
    assert_eq!(x.mgf(&t), (ctx.int(3) * (t.exp() - ctx.one())).exp());
    assert_eq!(x.mgf(&ctx.int(2).ln()).simplify(), ctx.int(3).exp());
    // E[2^X] = e^{λ(2−1)} = e³ through the summation engine.
    let e2x = x.expectation(&ctx.int(2).pow(x.symbol()));
    assert!(!e2x.has_unevaluated(), "E[2^X] = `{e2x}` should close");
    assert_eq!(e2x, ctx.int(3).exp());
}

// ── Geometric(1/4) on 1..∞ ─────────────────────────────────────────────
// SymPy: E = 4, variance = 12, E(Y**2) = 28, E(Y**3) = 292, E(Y**4) = 4060,
// cdf(k) = 1 − (3/4)^floor(k) for k ≥ 1, cdf(3) = 37/64, P(Y>3) = 27/64,
// P(Y<=3) = 37/64, P(2<=Y<=4) = 111/256, P(Eq(Y,2)) = 3/16,
// mgf = exp(t)/(4*(1 - 3*exp(t)/4)), mgf(log(5/4)) = 5,
// skewness = 7*sqrt(3)/6.  Symbolic: E(Y**2) = (2 − p)/p².

#[test]
fn geometric_moments_from_mgf() {
    let ctx = Context::new();
    let y = rv(&ctx, "Y", Distribution::geometric(ctx.rational(1, 4)));
    assert_eq!(y.mean(), ctx.int(4));
    assert_eq!(y.variance(), ctx.int(12));
    assert_eq!(y.moment(2), ctx.int(28));
    assert_eq!(y.moment(3), ctx.int(292));
    assert_eq!(y.moment(4), ctx.int(4060));
    assert_eq!(
        y.distribution().family().raw_moment(3),
        Some(ctx.int(292)),
        "closed route through the mgf"
    );
    assert_close(&y.skewness(), 7.0 * 3f64.sqrt() / 6.0, "skewness");

    let p = ctx.symbol("p");
    let z = rv(&ctx, "Z", Distribution::geometric(p.clone()));
    let d = (z.moment(2) - (ctx.int(2) - &p) / p.powi(2)).simplify();
    assert!(d.is_zero_structural(), "symbolic E[Z²]: difference `{d}`");
}

#[test]
fn geometric_probabilities() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let y = rv(&ctx, "Y", Distribution::geometric(ctx.rational(1, 4)));
    let s = y.symbol();
    assert_eq!(y.probability(&s.le(&ctx.int(3)))?, ctx.rational(37, 64));
    assert_eq!(y.probability(&s.gt(&ctx.int(3)))?, ctx.rational(27, 64));
    assert_eq!(
        y.probability(&s.ge(&ctx.int(2)).and(&s.le(&ctx.int(4))))?,
        ctx.rational(111, 256)
    );
    assert_eq!(y.probability(&s.eq_expr(&ctx.int(2)))?, ctx.rational(3, 16));
    assert_eq!(y.probability(&s.ge(&ctx.int(1)))?, ctx.int(1), "total mass");
    // Events below the support are clipped to it.
    assert_eq!(y.probability(&s.ge(&ctx.int(-5)))?, ctx.int(1));
    Ok(())
}

#[test]
fn geometric_cdf_and_mgf() {
    let ctx = Context::new();
    let y = rv(&ctx, "Y", Distribution::geometric(ctx.rational(1, 4)));
    let k = ctx.symbol("k");
    let t = ctx.symbol("t");
    assert_eq!(y.cdf(&ctx.int(3)).simplify(), ctx.rational(37, 64));
    // 0.12: the whole-line cdf is clamped below the support (SymPy:
    // cdf(Geometric(1/4))(k) = Piecewise((1 - (3/4)**floor(k), k >= 1), (0, True))).
    assert_eq!(
        y.distribution().family().cdf(&k),
        Some(ctx.one() - ctx.rational(3, 4).pow(&k.floor()))
    );
    assert_eq!(y.cdf(&ctx.int(0)), ctx.int(0));
    assert_eq!(y.cdf(&ctx.int(2)), ctx.rational(7, 16));
    assert_eq!(
        y.mgf(&t),
        ctx.rational(1, 4) * t.exp() / (ctx.one() - ctx.rational(3, 4) * t.exp())
    );
    // ln(5/4) is inside the region of convergence (t < −ln(3/4)).
    assert_eq!(y.mgf(&ctx.rational(5, 4).ln()).simplify(), ctx.int(5));
}

// ── NegativeBinomial(3, 1/2) on 0..∞ ───────────────────────────────────
// SymPy: E = 3, variance = 6, E(N**3) = 99, E(N**4) = 807, P(N<=2) = 1/2,
// P(N>2) = 1/2, P(1<=N<=3) = 17/32, P(Eq(N,2)) = 3/16,
// mgf = 1/(8*(1 - exp(t)/2)**3), mgf(log(3/2)) = 8, skewness = sqrt(6)/2.
// Symbolic: E(N) = r(1−p)/p.

#[test]
fn negative_binomial_moments_from_mgf() {
    let ctx = Context::new();
    let n = rv(
        &ctx,
        "N",
        Distribution::negative_binomial(ctx.int(3), ctx.rational(1, 2)),
    );
    assert_eq!(n.mean(), ctx.int(3));
    assert_eq!(n.variance(), ctx.int(6));
    assert_eq!(n.moment(3), ctx.int(99));
    assert_eq!(n.moment(4), ctx.int(807));
    assert_close(&n.skewness(), 6f64.sqrt() / 2.0, "skewness");

    let r = ctx.symbol("r");
    let p = ctx.symbol("p");
    let m = rv(
        &ctx,
        "M",
        Distribution::negative_binomial(r.clone(), p.clone()),
    );
    let d = (m.moment(1) - &r * (ctx.one() - &p) / &p).simplify();
    assert!(d.is_zero_structural(), "symbolic E[M]: difference `{d}`");
    assert_poly_eq(
        &(m.moment(2) * p.powi(2)),
        &(&r * (ctx.one() - &p) * (&r * (ctx.one() - &p) + ctx.one())),
        "symbolic E[M²]·p² = r(1−p)(r(1−p) + 1)",
    );
}

#[test]
fn negative_binomial_probabilities() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let n = rv(
        &ctx,
        "N",
        Distribution::negative_binomial(ctx.int(3), ctx.rational(1, 2)),
    );
    let s = n.symbol();
    assert_eq!(n.probability(&s.le(&ctx.int(2)))?, ctx.rational(1, 2));
    assert_eq!(n.probability(&s.gt(&ctx.int(2)))?, ctx.rational(1, 2));
    assert_eq!(
        n.probability(&s.ge(&ctx.int(1)).and(&s.le(&ctx.int(3))))?,
        ctx.rational(17, 32)
    );
    assert_eq!(n.probability(&s.eq_expr(&ctx.int(2)))?, ctx.rational(3, 16));
    assert_eq!(n.probability(&s.ge(&ctx.int(0)))?, ctx.int(1), "total mass");
    Ok(())
}

#[test]
fn negative_binomial_cdf_and_mgf() {
    let ctx = Context::new();
    let n = rv(
        &ctx,
        "N",
        Distribution::negative_binomial(ctx.int(3), ctx.rational(1, 2)),
    );
    assert_eq!(n.cdf(&ctx.int(2)).simplify(), ctx.rational(1, 2));
    assert_eq!(n.mgf(&ctx.rational(3, 2).ln()).simplify(), ctx.int(8));
    // The pmf: C(k+2, k)/8 · (1/2)ᵏ, spelled as a polynomial in k.
    let k = ctx.symbol("k");
    let pmf = n.density(&k);
    assert_eq!(pmf.subs(&k, &ctx.int(0)).simplify(), ctx.rational(1, 8));
    assert_eq!(pmf.subs(&k, &ctx.int(2)).simplify(), ctx.rational(3, 16));
    assert_eq!(pmf.subs(&k, &ctx.int(5)).simplify(), ctx.rational(21, 256));
}

#[test]
fn negative_binomial_symbolic_r_keeps_the_binomial_coefficient() {
    let ctx = Context::new();
    let r = ctx.symbol("r");
    let k = ctx.symbol("k");
    let m = rv(
        &ctx,
        "M",
        Distribution::negative_binomial(r.clone(), ctx.rational(1, 2)),
    );
    let pmf = m.density(&k);
    assert_eq!(
        pmf,
        (&k + &r - ctx.one()).binomial(&k)
            * ctx.rational(1, 2).pow(&r)
            * ctx.rational(1, 2).pow(&k)
    );
    // Substituting r = 3 recovers the numeric family's masses.
    let at3 = pmf.subs(&r, &ctx.int(3));
    assert_eq!(at3.subs(&k, &ctx.int(2)).simplify(), ctx.rational(3, 16));
}

// ── Hypergeometric(10, 5, 3) ───────────────────────────────────────────
// SymPy: E = 3/2, variance = 7/12, E(Z**2) = 17/6, E(Z**3) = 6,
// E(Z**4) = 83/6, P(Z>=2) = 1/2, P(Z<=1) = 1/2, P(1<=Z<=2) = 5/6,
// P(Eq(Z,3)) = 1/12, density = {0: 1/12, 1: 5/12, 2: 5/12, 3: 1/12},
// cdf = {0: 1/12, 1: 1/2, 2: 11/12, 3: 1}, skewness = 0.

#[test]
fn hypergeometric_moments() {
    let ctx = Context::new();
    let z = rv(
        &ctx,
        "Z",
        Distribution::hypergeometric(ctx.int(10), ctx.int(5), ctx.int(3)),
    );
    assert_eq!(z.mean(), ctx.rational(3, 2));
    assert_eq!(z.variance(), ctx.rational(7, 12));
    assert_eq!(z.moment(2), ctx.rational(17, 6));
    assert_eq!(z.moment(3), ctx.int(6));
    assert_eq!(z.moment(4), ctx.rational(83, 6));
    assert_eq!(z.skewness(), ctx.int(0));
    assert_eq!(z.central_moment(2), ctx.rational(7, 12));
}

#[test]
fn hypergeometric_probabilities_and_density() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let z = rv(
        &ctx,
        "Z",
        Distribution::hypergeometric(ctx.int(10), ctx.int(5), ctx.int(3)),
    );
    let s = z.symbol();
    assert_eq!(z.probability(&s.le(&ctx.int(1)))?, ctx.rational(1, 2));
    assert_eq!(z.probability(&s.gt(&ctx.int(1)))?, ctx.rational(1, 2));
    assert_eq!(z.probability(&s.ge(&ctx.int(2)))?, ctx.rational(1, 2));
    assert_eq!(
        z.probability(&s.ge(&ctx.int(1)).and(&s.le(&ctx.int(2))))?,
        ctx.rational(5, 6)
    );
    assert_eq!(z.probability(&s.eq_expr(&ctx.int(3)))?, ctx.rational(1, 12));
    assert_eq!(z.probability(&s.ge(&ctx.int(0)))?, ctx.int(1), "total mass");
    let k = ctx.symbol("k");
    let pmf = z.density(&k);
    for (v, mass) in [(0, (1, 12)), (1, (5, 12)), (2, (5, 12)), (3, (1, 12))] {
        assert_eq!(
            pmf.subs(&k, &ctx.int(v)).simplify(),
            ctx.rational(mass.0, mass.1),
            "P(Z = {v})"
        );
    }
    assert_eq!(z.cdf(&ctx.int(2)).simplify(), ctx.rational(11, 12));
    Ok(())
}

#[test]
fn hypergeometric_support_is_clipped_by_the_population() {
    let ctx = Context::new();
    // 5 draws from 10 with 8 successes: at least 3 successes are drawn.
    let z = Distribution::hypergeometric(ctx.int(10), ctx.int(8), ctx.int(5));
    assert_eq!(
        z.support(),
        Support::integers(&ctx, Some(ctx.int(3)), Some(ctx.int(5)))
    );
    let zr = rv(&ctx, "Z", z);
    assert_eq!(
        zr.probability(&zr.symbol().ge(&ctx.int(0))).unwrap(),
        ctx.int(1)
    );
    // Symbolic parameters keep max/min.
    let (big_n, m, n) = (ctx.symbol("N"), ctx.symbol("m"), ctx.symbol("n"));
    let sym = Distribution::hypergeometric(big_n.clone(), m.clone(), n.clone());
    let support = sym.support();
    let iv = support.as_interval().expect("one integer range");
    assert_eq!(
        iv.lower,
        ctx.zero().max_with(&(&n + &m - &big_n)).simplify()
    );
    assert_eq!(iv.upper, n.min_with(&m).simplify());
    assert_eq!(sym.family().mean(), Some((&n * &m / &big_n).simplify()));
}

// ── DiscreteUniform(2, 7) and Die(6) ───────────────────────────────────
// SymPy DiscreteUniform('U', range(2, 8)): E = 9/2, variance = 35/12,
// E(U**3) = 261/2, P(U<=4) = 1/2, P(U>4) = 1/2, P(3<=U<=5) = 1/2,
// P(Eq(U,3)) = 1/6, mgf = Σ_{j=2}^{7} exp(j t)/6, mgf(log 2) = 42,
// cdf = {2: 1/6, 3: 1/3, 4: 1/2, 5: 2/3, 6: 5/6, 7: 1}, median = {4, 5}.
// SymPy Die('D', 6): E = 7/2, variance = 35/12, E(D**2) = 91/6,
// E(D**3) = 147/2, P(D<=4) = 2/3, P(D>4) = 1/3, P(3<=D<=5) = 1/2,
// P(Eq(D,3)) = 1/6, skewness = 0, kurtosis = 303/175, median = {3, 4}.

#[test]
fn discrete_uniform_moments() {
    let ctx = Context::new();
    let u = rv(
        &ctx,
        "U",
        Distribution::discrete_uniform(ctx.int(2), ctx.int(7)),
    );
    assert_eq!(u.mean(), ctx.rational(9, 2));
    assert_eq!(u.variance(), ctx.rational(35, 12));
    assert_eq!(u.moment(3), ctx.rational(261, 2));
    let s = u.symbol();
    // E[U² + 3U] = 139/6 + 27/2 = 110/3, through the Faulhaber sum.
    assert_eq!(u.expectation(&(s.powi(2) + 3 * s)), ctx.rational(110, 3));
    // Symbolic bounds: mean and variance stay polynomial.
    let (a, b) = (ctx.symbol("a"), ctx.symbol("b"));
    let v = rv(
        &ctx,
        "V",
        Distribution::discrete_uniform(a.clone(), b.clone()),
    );
    assert_poly_eq(&(v.mean() * 2), &(&a + &b), "2E[V]");
    assert_poly_eq(
        &(v.variance() * 12),
        &((&b - &a + 1).powi(2) - 1),
        "12 Var[V]",
    );
}

#[test]
fn discrete_uniform_probabilities() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let u = rv(
        &ctx,
        "U",
        Distribution::discrete_uniform(ctx.int(2), ctx.int(7)),
    );
    let s = u.symbol();
    assert_eq!(u.probability(&s.le(&ctx.int(4)))?, ctx.rational(1, 2));
    assert_eq!(u.probability(&s.gt(&ctx.int(4)))?, ctx.rational(1, 2));
    assert_eq!(
        u.probability(&s.ge(&ctx.int(3)).and(&s.le(&ctx.int(5))))?,
        ctx.rational(1, 2)
    );
    assert_eq!(u.probability(&s.eq_expr(&ctx.int(3)))?, ctx.rational(1, 6));
    assert_eq!(u.probability(&s.ge(&ctx.int(2)))?, ctx.int(1), "total mass");
    // Non-integer bounds round inwards: P(2.5 < U < 5.5) = P(3 ≤ U ≤ 5).
    assert_eq!(
        u.probability(&s.gt(&ctx.rational(5, 2)).and(&s.lt(&ctx.rational(11, 2))))?,
        ctx.rational(1, 2)
    );
    Ok(())
}

#[test]
fn discrete_uniform_cdf_mgf_and_quantile() {
    let ctx = Context::new();
    let u = rv(
        &ctx,
        "U",
        Distribution::discrete_uniform(ctx.int(2), ctx.int(7)),
    );
    for (v, c) in [
        (2, (1, 6)),
        (3, (1, 3)),
        (4, (1, 2)),
        (5, (2, 3)),
        (6, (5, 6)),
        (7, (1, 1)),
    ] {
        assert_eq!(
            u.cdf(&ctx.int(v)).simplify(),
            ctx.rational(c.0, c.1),
            "cdf({v})"
        );
    }
    assert_eq!(u.cdf(&ctx.rational(9, 2)).simplify(), ctx.rational(1, 2));
    let t = ctx.symbol("t");
    let mgf = u.mgf(&t);
    assert_eq!(mgf.subs(&t, &ctx.int(2).ln()).simplify(), ctx.int(42));
    // The same function as SymPy's Σ e^{jt}/6 (a geometric series).
    let sum: Ex = (2..=7).map(|j| (ctx.int(j) * &t).exp() / 6).sum();
    for tv in [ctx.rational(1, 3), ctx.rational(-2, 5), ctx.int(1)] {
        let expected = sum.subs(&t, &tv).eval_f64().unwrap();
        assert_close(&mgf.subs(&t, &tv), expected, "mgf vs Σ e^{jt}/6");
    }
    // Lower quantile: the smallest k with F(k) ≥ q.
    assert_eq!(u.quantile(&ctx.rational(1, 2)), Some(ctx.int(4)));
    assert_eq!(u.quantile(&ctx.rational(1, 6)), Some(ctx.int(2)));
    assert_eq!(u.quantile(&ctx.rational(1, 7)), Some(ctx.int(2)));
    assert_eq!(u.quantile(&ctx.one()), Some(ctx.int(7)));
    assert_eq!(u.median(), Some(ctx.int(4)));
}

#[test]
fn die_is_discrete_uniform_from_one() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let d = rv(&ctx, "D", Distribution::die(ctx.int(6)));
    assert_eq!(
        d.support(),
        Support::integers(&ctx, Some(ctx.int(1)), Some(ctx.int(6)))
    );
    assert_eq!(d.distribution().name(), "DiscreteUniform");
    assert_eq!(format!("{d}"), "D ~ DiscreteUniform(1, 6)");
    assert_eq!(d.mean(), ctx.rational(7, 2));
    assert_eq!(d.variance(), ctx.rational(35, 12));
    assert_eq!(d.moment(2), ctx.rational(91, 6));
    assert_eq!(d.moment(3), ctx.rational(147, 2));
    assert_eq!(d.skewness(), ctx.int(0));
    assert_eq!(d.kurtosis(), ctx.rational(303, 175));
    let s = d.symbol();
    assert_eq!(d.probability(&s.le(&ctx.int(4)))?, ctx.rational(2, 3));
    assert_eq!(d.probability(&s.gt(&ctx.int(4)))?, ctx.rational(1, 3));
    assert_eq!(
        d.probability(&s.ge(&ctx.int(3)).and(&s.le(&ctx.int(5))))?,
        ctx.rational(1, 2)
    );
    assert_eq!(d.probability(&s.eq_expr(&ctx.int(3)))?, ctx.rational(1, 6));
    assert_eq!(d.probability(&s.ge(&ctx.int(1)))?, ctx.int(1));
    assert_eq!(d.median(), Some(ctx.int(3)));
    Ok(())
}

#[test]
fn die_sampling_frequencies_are_uniform() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let d = rv(&ctx, "D", Distribution::die(ctx.int(6)));
    let mut rng = Rng::new(7);
    let n = 60_000;
    let samples = d.sample(n, &mut rng)?;
    assert_eq!(samples.len(), n);
    let mut counts = [0usize; 6];
    for s in &samples {
        let face = *s as i64;
        assert!(
            (1..=6).contains(&face) && (*s - face as f64).abs() < 1e-12,
            "sample {s}"
        );
        counts[(face - 1) as usize] += 1;
    }
    for (face, &c) in counts.iter().enumerate() {
        let freq = c as f64 / n as f64;
        let rel = (freq - 1.0 / 6.0).abs() / (1.0 / 6.0);
        assert!(
            rel < 0.05,
            "face {}: frequency {freq} is off by {rel:.3}",
            face + 1
        );
    }
    Ok(())
}

#[test]
fn bernoulli_and_binomial_sampling_match_the_mean() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let mut rng = Rng::new(11);
    let b = rv(&ctx, "B", Distribution::bernoulli(ctx.rational(2, 5)));
    let s = b.sample(20_000, &mut rng)?;
    let mean = s.iter().sum::<f64>() / s.len() as f64;
    assert!((mean - 0.4).abs() < 0.02, "Bernoulli sample mean {mean}");
    let x = rv(
        &ctx,
        "X",
        Distribution::binomial(ctx.int(5), ctx.rational(1, 3)),
    );
    let s = x.sample(20_000, &mut rng)?;
    let mean = s.iter().sum::<f64>() / s.len() as f64;
    assert!(
        (mean - 5.0 / 3.0).abs() < 0.05,
        "Binomial sample mean {mean}"
    );
    Ok(())
}

// ── Constructors ───────────────────────────────────────────────────────

#[test]
fn constructors_reject_invalid_numeric_parameters() {
    let ctx = Context::new();
    let bad = |r: Result<Distribution, SymplexError>, what: &str| {
        assert!(
            matches!(r, Err(SymplexError::InvalidArgument { .. })),
            "{what} should be rejected"
        );
    };
    bad(
        Distribution::try_bernoulli(ctx.rational(3, 2)),
        "Bernoulli(3/2)",
    );
    bad(Distribution::try_bernoulli(ctx.int(-1)), "Bernoulli(-1)");
    bad(
        Distribution::try_binomial(ctx.rational(5, 2), ctx.rational(1, 2)),
        "Binomial(5/2, 1/2)",
    );
    bad(Distribution::try_poisson(ctx.int(0)), "Poisson(0)");
    bad(Distribution::try_poisson(ctx.int(-2)), "Poisson(-2)");
    bad(Distribution::try_geometric(ctx.int(0)), "Geometric(0)");
    bad(Distribution::try_geometric(ctx.int(2)), "Geometric(2)");
    bad(
        Distribution::try_negative_binomial(ctx.int(0), ctx.rational(1, 2)),
        "NegativeBinomial(0, 1/2)",
    );
    bad(
        Distribution::try_negative_binomial(ctx.int(3), ctx.int(1)),
        "NegativeBinomial(3, 1)",
    );
    bad(
        Distribution::try_hypergeometric(ctx.int(10), ctx.int(11), ctx.int(3)),
        "Hypergeometric(10, 11, 3)",
    );
    bad(
        Distribution::try_hypergeometric(ctx.int(10), ctx.int(5), ctx.int(12)),
        "Hypergeometric(10, 5, 12)",
    );
    bad(
        Distribution::try_hypergeometric(ctx.int(10), ctx.rational(1, 2), ctx.int(3)),
        "Hypergeometric(10, 1/2, 3)",
    );
    bad(
        Distribution::try_discrete_uniform(ctx.int(5), ctx.int(2)),
        "DiscreteUniform(5, 2)",
    );
    bad(
        Distribution::try_discrete_uniform(ctx.rational(1, 2), ctx.int(3)),
        "DiscreteUniform(1/2, 3)",
    );
    bad(Distribution::try_die(ctx.int(0)), "Die(0)");
    bad(Distribution::try_die(ctx.rational(7, 2)), "Die(7/2)");
}

#[test]
fn constructors_accept_valid_and_symbolic_parameters() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let p = ctx.symbol("p");
    let n = ctx.symbol("n");
    assert_eq!(
        Distribution::try_bernoulli(ctx.rational(2, 5))?,
        Distribution::bernoulli(ctx.rational(2, 5))
    );
    assert_eq!(Distribution::try_bernoulli(ctx.int(1))?.name(), "Bernoulli");
    assert_eq!(
        Distribution::try_poisson(ctx.rational(1, 10))?.name(),
        "Poisson"
    );
    assert_eq!(Distribution::try_poisson(n.clone())?.name(), "Poisson");
    assert_eq!(Distribution::try_geometric(ctx.int(1))?.name(), "Geometric");
    assert_eq!(Distribution::try_geometric(p.clone())?.name(), "Geometric");
    assert_eq!(
        Distribution::try_negative_binomial(ctx.rational(5, 2), ctx.rational(1, 3))?.name(),
        "NegativeBinomial"
    );
    assert_eq!(
        Distribution::try_hypergeometric(ctx.int(10), ctx.int(10), ctx.int(10))?.name(),
        "Hypergeometric"
    );
    assert_eq!(
        Distribution::try_hypergeometric(n.clone(), ctx.int(5), ctx.int(3))?.name(),
        "Hypergeometric"
    );
    assert_eq!(
        Distribution::try_discrete_uniform(ctx.int(-3), ctx.int(-3))?.name(),
        "DiscreteUniform"
    );
    assert_eq!(
        Distribution::try_die(n)?,
        Distribution::die(ctx.symbol("n"))
    );
    assert_eq!(
        Distribution::try_die(ctx.int(20))?.name(),
        "DiscreteUniform"
    );
    Ok(())
}

#[test]
fn display_names_the_family_and_parameters() {
    let ctx = Context::new();
    let p = ctx.rational(1, 4);
    let cases = [
        (Distribution::bernoulli(p.clone()), "Bernoulli(1/4)"),
        (
            Distribution::binomial(ctx.int(5), p.clone()),
            "Binomial(5, 1/4)",
        ),
        (Distribution::poisson(ctx.int(3)), "Poisson(3)"),
        (Distribution::geometric(p.clone()), "Geometric(1/4)"),
        (
            Distribution::negative_binomial(ctx.int(3), p.clone()),
            "NegativeBinomial(3, 1/4)",
        ),
        (
            Distribution::hypergeometric(ctx.int(10), ctx.int(5), ctx.int(3)),
            "Hypergeometric(10, 5, 3)",
        ),
        (
            Distribution::discrete_uniform(ctx.int(2), ctx.int(7)),
            "DiscreteUniform(2, 7)",
        ),
    ];
    for (d, text) in cases {
        assert_eq!(format!("{d}"), text);
        assert!(!d.is_continuous());
        assert_eq!(d.support().kind(), symplex::stats::Kind::Discrete);
    }
}

/// `Finite` tables (SymPy `FiniteRV`): non-integer values, exact moments
/// through the table, probabilities by exact comparison, cdf as an
/// indicator sum, sampling by cumulative sums, and constructor checks.
#[test]
fn finite_table_distribution() {
    use symplex::stats::{Distribution, RandomVariable, Rng};
    let ctx = Context::new();
    // SymPy: X = FiniteRV('X', {1/2: 1/4, 2: 1/4, 7/2: 1/2})
    //   E(X) = 19/8, variance(X) = 99/64, E(X**3) = 751/32,
    //   P(X > 1) = 3/4, P(X <= 2) = 1/2, P(Eq(X, 2)) = 1/4, cdf(X)(3) = 1/2
    let table = vec![
        (ctx.rational(1, 2), ctx.rational(1, 4)),
        (ctx.int(2), ctx.rational(1, 4)),
        (ctx.rational(7, 2), ctx.rational(1, 2)),
    ];
    let x = RandomVariable::new(&ctx, "X", Distribution::try_finite(&ctx, table).unwrap());
    assert_eq!(x.mean(), ctx.rational(19, 8));
    assert_eq!(x.variance(), ctx.rational(99, 64));
    assert_eq!(x.moment(3), ctx.rational(751, 32));
    // E[X² − 1] = Var + E[X]² − 1 = 99/64 + 361/64 − 1 = 99/16
    assert_eq!(
        x.expectation(&(x.symbol().powi(2) - 1)),
        ctx.rational(99, 16)
    );
    let s = x.symbol();
    assert_eq!(
        x.probability(&s.gt(&ctx.int(1))).unwrap(),
        ctx.rational(3, 4)
    );
    assert_eq!(
        x.probability(&s.le(&ctx.int(2))).unwrap(),
        ctx.rational(1, 2)
    );
    assert_eq!(
        x.probability(&s.eq_expr(&ctx.int(2))).unwrap(),
        ctx.rational(1, 4)
    );
    assert_eq!(x.probability(&s.eq_expr(&ctx.int(3))).unwrap(), ctx.int(0));
    assert_eq!(x.cdf(&ctx.int(3)), ctx.rational(1, 2));
    assert_eq!(x.probability(&s.ge(&ctx.int(0))).unwrap(), ctx.int(1));
    assert_eq!(x.mgf(&ctx.int(0)).simplify(), ctx.int(1));
    assert_eq!(x.to_string(), "X ~ Finite({1/2: 1/4, 2: 1/4, 7/2: 1/2})");
    // Sampling: the frequency of 7/2 over 40 000 draws is near 1/2.
    let samples = x.sample(40_000, &mut Rng::new(3)).unwrap();
    let f = samples.iter().filter(|v| (**v - 3.5).abs() < 1e-12).count() as f64 / 40_000.0;
    assert!((f - 0.5).abs() < 0.02, "{f}");
    // Constructor checks.
    assert!(Distribution::try_finite(&ctx, vec![]).is_err());
    assert!(Distribution::try_finite(&ctx, vec![(ctx.int(1), ctx.rational(1, 2))]).is_err());
    assert!(
        Distribution::try_finite(
            &ctx,
            vec![(ctx.int(1), ctx.int(2)), (ctx.int(2), ctx.int(-1))]
        )
        .is_err()
    );
    assert!(
        Distribution::try_finite(
            &ctx,
            vec![
                (ctx.int(1), ctx.rational(1, 2)),
                (ctx.int(1), ctx.rational(1, 2))
            ]
        )
        .is_err()
    );
    // A symbolic probability is accepted and carried through.
    let p = ctx.symbol("p");
    let coin = RandomVariable::new(
        &ctx,
        "C",
        Distribution::try_finite(&ctx, vec![(ctx.int(1), p.clone()), (ctx.int(0), 1 - &p)])
            .unwrap(),
    );
    assert_eq!(coin.mean(), p);
}
