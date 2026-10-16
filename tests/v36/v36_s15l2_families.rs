//! Tests of session 15 (s15l2): every distribution family against scipy.stats and mpmath.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use std::time::Duration;

use symplex::prelude::*;
use symplex::stats::Distribution;

use crate::perf_assert::assert_fast;

fn close(actual: f64, expected: f64, rel: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= rel * expected.abs(),
        "{label}: {actual} vs {expected}"
    );
}

/// The first `n` significant characters of a decimal (the digits checked
/// against the oracle).
fn digits(e: &Ex, n: u32) -> String {
    match e.eval_decimal(n) {
        Ok(s) => s,
        Err(err) => panic!("{e} did not evaluate: {err}"),
    }
}

fn rat(ctx: &Context, s: &str) -> Ex {
    match ctx.rational_str(s) {
        Ok(e) => e,
        Err(err) => panic!("{s}: {err}"),
    }
}

/// The Cauchy quantile next to 0 and 1: `x₀ + γ tan(π(p − ½))` puts the
/// argument of `tan` next to its pole, as a difference of two numbers that
/// agree to all but the digits of `p`; up to 0.56 `quantile(10⁻³⁰⁰)` and
/// `quantile_f64(1e-300)` of `Cauchy(9/10, 1)` were refused ("precision
/// exhausted").  Now `x₀ − γ cot(πp)` below ½, `x₀ + γ cot(π(1 − p))`
/// above.  Oracle: mpmath at 40 and 80 digits, `0.9 − cot(π·10⁻³⁰⁰) =
/// −3.1830988618379067153776752674503e299`, `0.9 + cot(π·10⁻²⁰) =
/// 31830988618379067154.676752674503`; scipy 1.18.1 `cauchy(0.9).ppf(1e-300)
/// = -3.183098861837907e+299` agrees.
#[test]
fn cauchy_quantile_in_the_far_tails() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let c = Distribution::cauchy(ctx.rational(9, 10), ctx.int(1));
    let tiny = ctx.int(10).powi(-300);
    let q = c.quantile(&tiny).ok_or(SymplexError::Unevaluable {
        reason: "no closed quantile".into(),
    })?;
    assert_eq!(digits(&q, 30), "-3.18309886183790671537767526745e299");
    close(
        c.quantile_f64(1e-300)?,
        -3.183_098_861_837_906_7e299,
        1e-15,
        "quantile_f64(1e-300)",
    );
    let near_one = 1 - ctx.int(10).powi(-20);
    let q = c.quantile(&near_one).ok_or(SymplexError::Unevaluable {
        reason: "no closed quantile".into(),
    })?;
    assert_eq!(digits(&q, 30), "31830988618379067154.6767526745");
    close(
        c.isf_f64(1e-20)?,
        3.183_098_861_837_906_7e19,
        1e-15,
        "isf_f64(1e-20)",
    );
    // A symbolic level keeps the classic form.
    let p = ctx.symbol("p");
    let pi = ctx.pi();
    let classic = ctx.rational(9, 10) + (&pi * (&p - ctx.rational(1, 2))).tan();
    assert_eq!(c.quantile(&p), Some(classic));
    Ok(())
}

/// A density with a power of huge degree (`x^(9·10¹⁰ − 1)` in
/// `Weibull(51/8, 9·10¹⁰)`, `x^(7.5·10⁹ − 1)` in `F(1.5·10¹⁰, 3/5)`): the
/// breakpoint scan of the moment generating function's integral laid the
/// power out as a dense coefficient vector (`calculus_util::
/// poly_coeffs_symbolic`, `solve::symbolic_poly_coeffs`), asked for 3 GB
/// and 360 GB, and aborted the process up to 0.56.  Now a degree above
/// 2¹⁶ is not laid out densely and the integral is evaluated or refused.
#[test]
fn huge_powers_in_a_density_do_not_exhaust_memory() {
    let ctx = Context::new();
    let w = Distribution::weibull(ctx.rational(51, 8), ctx.int(90_000_000_000));
    let _ = w.mgf(&ctx.rational(80, 51)).eval_decimal(30);
    let f = Distribution::f_distribution(ctx.int(15_000_000_000), ctx.rational(3, 5));
    let _ = f.mgf(&ctx.rational(-1, 5)).eval_decimal(30);
}

/// The lattice walk of a family without a closed CDF decided the quantile
/// in `f64` alone; the compiled masses of `Hypergeometric(10²⁰,
/// 56390476614683887781, 17)` overflow (`C(10²⁰, 17) = ∞`, a mass
/// `x/∞ = 0`), the walk summed zeros to the end of the support, and
/// `quantile_f64(3/998)` was `17` up to 0.56.  Now the walk only guesses
/// and each probe is decided on the exact masses.  Oracle: mpmath (40 and
/// 80 digits) `F(3) = 0.0012839934562482617213`, `F(4) =
/// 0.0062494300577184634168`, so the quantile at `3/998 = 0.003006` is 4.
/// The negative-binomial walk of `NegativeBinomial(5000, 7/54)` at `10⁻²⁰`
/// evaluated tens of thousands of masses exactly and ran beyond 20 s.
/// Oracle: mpmath (continued fraction of `I_p(r, k + 1)`, 40 and 80
/// digits) `F(29059) = 9.9714309420465678925e-21`, `F(29060) =
/// 1.0174304001437641599e-20`; scipy `nbinom(5000, 7/54).ppf(1e-20) =
/// 29060.0`.
#[test]
fn lattice_quantiles_without_a_closed_cdf_are_decided_exactly() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let h = Distribution::hypergeometric(
        rat(&ctx, "100000000000000000000"),
        rat(&ctx, "56390476614683887781"),
        ctx.int(17),
    );
    assert_eq!(h.quantile_f64(3.0 / 998.0)?, 4.0);
    assert_eq!(h.isf_f64(1e-5)?, 17.0);
    let nb = Distribution::negative_binomial(ctx.int(5000), ctx.rational(7, 54));
    assert_eq!(
        assert_fast!(Duration::from_secs(10), nb.quantile_f64(1e-20))?,
        29060.0
    );
    Ok(())
}

/// `B(a, b)` with a huge argument: `evalf` formed `Γ(a)Γ(b)/Γ(a + b)`, and
/// `Γ(1.25·10⁹)` lies beyond the exponent range of the arbitrary-precision
/// floats, so up to 0.56 `B(7/20, 1.25·10⁹)` — the constant of the density
/// of `F(7/10, 2.5·10⁹)` — was refused, and the density with it.  Now
/// `exp(ln Γ(a) + ln Γ(b) − ln Γ(a + b))` with guard bits for the
/// cancellation.  Oracle: mpmath `beta(7/20, 1250000000)` at 40 and 80
/// digits `0.0016671127206155258874571998253835`; the density at `10⁻⁶`
/// from the log-space closed form at 45 and 75 digits
/// `2160.42997175310737711306474333250`; scipy 1.18.1 gives
/// `f(0.7, 2.5e9).pdf(1e-6) = 2160.43799` (wrong from the 6th digit).
#[test]
fn beta_function_and_f_density_with_a_huge_argument() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let b = ctx.rational(7, 20).beta(&ctx.int(1_250_000_000));
    assert_eq!(digits(&b, 30), "0.00166711272061552588745719982538");
    let f = Distribution::f_distribution(ctx.rational(7, 10), ctx.int(2_500_000_000));
    let pdf = f.density(&ctx.int(10).powi(-6));
    assert_eq!(digits(&pdf, 30), "2160.42997175310737711306474333");
    // Ordinary arguments keep the Gamma route.
    assert_eq!(
        digits(&ctx.int(2).beta(&ctx.int(3)), 20),
        "0.083333333333333333333"
    );
    Ok(())
}

/// The F density at 0: `+∞` for `d₁ < 2` (scipy's convention, the limit
/// from the right).  Up to 0.56 the limit through SymPy's form `√((d₁x)^{d₁}
/// d₂^{d₂}/(d₁x + d₂)^{d₁+d₂})/(x B)` was not found for a huge `d₂`, and
/// `F(4/5, 10¹⁰/3).density(0)` was NaN.  The density is now written
/// `(d₁/d₂)^{d₁/2} x^{d₁/2−1} (1 + d₁x/d₂)^{−(d₁+d₂)/2} / B(d₁/2, d₂/2)`.
/// Oracle: the power `x^{−3/5}` at 0; scipy 1.18.1 `f(0.8, 1e10/3).pdf(0) = inf`.
#[test]
fn f_density_at_zero_with_a_huge_denominator_dof() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let f = Distribution::f_distribution(ctx.rational(4, 5), ctx.rational(10_000_000_000, 3));
    assert_eq!(f.density(&zero), ctx.infinity());
    // (2/d₂)/B(1, d₂/2) = 1 (B(1, b) = 1/b, left unevaluated for this b).
    let f = Distribution::f_distribution(ctx.int(2), ctx.int(12_500_000_000));
    assert_eq!(digits(&f.density(&zero), 20), "1");
    let f = Distribution::f_distribution(ctx.int(3), ctx.int(12_500_000_000));
    assert_eq!(f.density(&zero), ctx.int(0));
}

/// Entropies with `ln Γ` and `ln B` of huge shapes: `Γ(5·10²⁹)` and `B(2/5,
/// 5·10⁹/3)` leave the exponent range (or cancel beyond `evalf`'s budget),
/// so up to 0.56 the entropies of `Gamma(5·10²⁹, 8)` and `F(4/5, 10¹⁰/3)`
/// were refused; now `loggamma`.  Oracle: mpmath at 80 digits (40 lose the
/// Gamma one to the cancellation of `(1 − k)ψ(k)` against `k`):
/// `37.690582879515221275593281860316` and (40 and 80 digits)
/// `0.57613782324486914432945788492766`.  scipy 1.18.1:
/// `gamma(5e29, scale=8).entropy() = 37.69058287951522`, but
/// `f(0.8, 1e10/3).entropy() = 0.576141357421875` (wrong from the 6th digit).
#[test]
fn entropies_with_huge_shapes() {
    let ctx = Context::new();
    let g = Distribution::gamma(rat(&ctx, "500000000000000000000000000000"), ctx.int(8));
    assert_eq!(digits(&g.entropy(), 30), "37.6905828795152212755932818603");
    let f = Distribution::f_distribution(ctx.rational(4, 5), ctx.rational(10_000_000_000, 3));
    assert_eq!(digits(&f.entropy(), 30), "0.576137823244869144329457884928");
}

/// Characteristic functions with a phase `e^{iθ}` of a huge exact `θ`, or of
/// a `θ` whose sine is far below the cosine.  `exp(iθ)` takes `θ` at the
/// working precision, where `1.2·10⁹⁹` is inexact, and refused it; and it
/// bounds both parts by the modulus, dropping an imaginary part of
/// `5.7·10⁻³⁰³`.  Up to 0.56 the first four below were refused and the
/// last printed as a real number; now the phase is `cos θ + i sin θ` of the
/// exact `θ` (and the uniform's difference of exponentials is taken about the
/// midpoint).  Oracle: mpmath at 350 and 420 digits (the arguments need
/// their 300 integer digits): `Cauchy(3, 5·10⁻¹⁰⁰)` at `−4·10⁹⁸`:
/// `0.49245660503365583739322260762219 − 0.65406921514038955485880349217016i`;
/// `Laplace(2·10³⁰⁰, 17/2)` at `−1/17`: `0.68841295440017785953360215966507 −
/// 0.40753846961240191493551167721886i`; `Logistic(1, 3·10⁻³⁰⁰)` at
/// `−10³⁰⁰`: `−0.00025587941538061862323539305044737 +
/// 0.0014994743372933101639605530390864i`; `U(10³⁰, 10³⁰ + 1)` at 2 (50 and
/// 90 digits): `0.32016474846461359700395127089099 +
/// 0.77818246710791544206494268798218i`; `Normal(−1, 1.75·10³⁰⁰)` at
/// `4·10⁻³⁰²/7`: `0.99995000124997916692708072918837 −
/// 5.7140000071427380967261755953621e−303i`.
#[test]
fn characteristic_functions_with_huge_or_tiny_phases() {
    let ctx = Context::new();
    let ten = |e: i64| ctx.int(10).powi(e);
    let cases: Vec<(Distribution, Ex, &str)> = vec![
        (
            Distribution::cauchy(ctx.int(3), ctx.int(5) * ten(-100)),
            ctx.int(-4) * ten(98),
            "0.492456605033655837393222607622 - 0.65406921514038955485880349217*i",
        ),
        (
            Distribution::laplace(ctx.int(2) * ten(300), ctx.rational(17, 2)),
            ctx.rational(-1, 17),
            "0.688412954400177859533602159665 - 0.407538469612401914935511677219*i",
        ),
        (
            Distribution::logistic(ctx.int(1), ctx.int(3) * ten(-300)),
            ctx.int(-1) * ten(300),
            "-0.000255879415380618623235393050447 + 0.00149947433729331016396055303909*i",
        ),
        (
            Distribution::uniform(ten(30), ten(30) + ctx.int(1)),
            ctx.int(2),
            "0.320164748464613597003951270891 + 0.778182467107915442064942687982*i",
        ),
        (
            Distribution::normal(ctx.int(-1), ctx.int(175) * ten(298)),
            ctx.rational(4, 7) * ten(-302),
            "0.999950001249979166927080729188 - 5.71400000714273809672617559536e-303*i",
        ),
    ];
    for (d, t, want) in cases {
        assert_eq!(
            digits(&d.characteristic_function(&t), 30),
            want,
            "{} at {t}",
            d.name()
        );
    }
}

/// The Poisson mass `λᵏ e^{−λ}/k!` at a huge rate: `λᵏ` leaves the exponent
/// range of the arbitrary-precision floats (`(10¹⁸)^(10¹⁸)`), so up to 0.56
/// the mass of `Poisson(10¹⁸)` at `10¹⁸` was refused (and that of
/// `Poisson(10⁹)` at `10⁹`); now `exp(k ln λ − λ − ln Γ(k + 1))` for such
/// arguments.  Oracle: mpmath at 80 digits (40 lose 22 digits to the
/// cancellation) `3.9894228040143267790670086990093e-10`; scipy 1.18.1
/// `poisson(1e18).pmf(1e18) = 1.0`, wrong.
#[test]
fn poisson_mass_at_a_huge_rate() {
    let ctx = Context::new();
    let lam = ctx.int(10).powi(18);
    let d = Distribution::poisson(lam.clone());
    assert_eq!(
        digits(&d.density(&lam), 30),
        "3.98942280401432677906700869901e-10"
    );
    // Small arguments keep the classic form.
    let k = ctx.int(3);
    assert_eq!(
        Distribution::poisson(ctx.int(2)).density(&k).eval(),
        ctx.rational(4, 3) * ctx.int(-2).exp()
    );
}

/// Binomial distribution functions beyond the exact sum's budget: up to 0.56
/// `Binomial(10⁷, 1/3).cdf(3333333)` summed `3·10⁶` masses symbolically and
/// was refused ("Sum range too large"), and `Binomial(10⁶, 1 −
/// 10⁻¹⁸).cdf(0)` formed the exact rational `10^(−1.8·10⁷)` (60 million
/// bits) and ran beyond 20 s.  Now the regularised incomplete beta
/// `I_{1−p}(n − k, k + 1)` / `I_p(k + 1, n − k)`.  Oracle: mpmath continued
/// fraction at 40 and 80 digits: `0.50005947080378916057092933834802`,
/// `0.49994052919621083942907066165198`; scipy 1.18.1
/// `binom(10**7, 1/3).cdf(3333333) = 0.5000594708039544` (13 digits).
#[test]
fn binomial_tails_beyond_the_exact_budget() {
    let ctx = Context::new();
    let b = Distribution::binomial(ctx.int(10_000_000), ctx.rational(1, 3));
    let k = ctx.int(3_333_333);
    assert_eq!(digits(&b.cdf(&k), 30), "0.500059470803789160570929338348");
    assert_eq!(digits(&b.sf(&k), 30), "0.499940529196210839429070661652");
    let near_one = 1 - ctx.int(10).powi(-18);
    let b = Distribution::binomial(ctx.int(1_000_000), near_one);
    assert_eq!(
        assert_fast!(Duration::from_secs(10), digits(&b.cdf(&ctx.int(0)), 10)),
        "1e-18000000"
    );
}

/// A level that a jump of the binomial CDF matches to rounding: `F(0) =
/// (1/1000)¹⁰⁰ = 10⁻³⁰⁰` exactly, and the double nearest `1e-300` exceeds it by
/// `2.5·10⁻¹⁷` of itself (exact rational arithmetic), so the smallest `k` with
/// `F(k) ≥ p` is 1 (`F(1) = 9.99·10⁻²⁹⁶`).  The `f64` kernel compared in `f64`
/// and answered 0 up to 0.56 (as does scipy 1.18.1:
/// `binom(100, 0.999).ppf(1e-300) = 0.0`); a kernel answer whose tails lie
/// within `2⁻⁴⁴` of the level is now decided exactly.
#[test]
fn binomial_quantile_at_a_rounding_tie() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let b = Distribution::binomial(ctx.int(100), ctx.rational(999, 1000));
    assert_eq!(b.quantile_f64(1e-300)?, 1.0);
    assert_eq!(b.quantile_f64(0.5)?, 100.0);
    // p = 1 − 10⁻¹⁸ rounds to the double 1.0, for which the kernel's F(9) is 0;
    // truly F(9) = 1 − p¹⁰ = 10⁻¹⁷ (exact), so the quantile at 1e-20 is 9 (it
    // was 10, as scipy's `binom(10, 1 - 1e-18).ppf(1e-20) = 10.0`).
    let near_one = 1 - ctx.int(10).powi(-18);
    let b = Distribution::binomial(ctx.int(10), near_one);
    assert_eq!(b.quantile_f64(1e-20)?, 9.0);
    Ok(())
}

/// Better than scipy: values the differential hunt found scipy 1.18.1 wrong
/// on, and symplex right (each value here agreed before this session; the
/// test records the comparison).  Oracle: mpmath at 30 and 60 digits from
/// the closed forms (exact rationals where noted).
/// * `DiscreteUniform(0, 10¹⁸)`: variance `((10¹⁸ + 1)² − 1)/12` (exact);
///   scipy `randint(0, 10**18 + 1).var() = -2.939291022165716e+17`.
/// * `Weibull(51/8, 9·10¹⁰)`: variance `λ²(Γ(1 + 2/k) − Γ(1 + 1/k)²) =
///   8.25322821710624619861775512258e-21`; scipy
///   `weibull_min(9e10, scale=6.375).var() = -1.80480630440627e-14`.
/// * `Bernoulli(10⁻¹⁸)`: entropy `4.24465316738928223118238461843e-17`;
///   scipy `bernoulli(1e-18).entropy() = 4.1446531673892826e-17`.
/// * `Pareto(2, 7·10⁻¹⁰)`: `F(12) = 1 − 6^(−7·10⁻¹⁰) =
///   1.25423162767309001198331381117e-9`; scipy `1.25423160746152e-09`.
/// * `Geometric(51/146)`: `S(k) = (95/146)ᵏ` is `1.37·10⁻²⁰⁰` at 1071 and
///   `8.9·10⁻²⁰¹` at 1072, so `isf(1e-200) = 1072`; scipy gives `inf`.
#[test]
fn better_than_scipy_family_values() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let big = ctx.int(10).powi(18);
    let u = Distribution::discrete_uniform(ctx.int(0), big.clone());
    let want = ((&big + ctx.int(1)).powi(2) - ctx.int(1)) / ctx.int(12);
    assert_eq!(u.variance(), want.eval());
    let w = Distribution::weibull(ctx.rational(51, 8), ctx.int(90_000_000_000));
    assert_eq!(
        digits(&w.variance(), 30),
        "8.25322821710624619861775512258e-21"
    );
    let b = Distribution::bernoulli(ctx.int(10).powi(-18));
    assert_eq!(
        digits(&b.entropy(), 30),
        "4.24465316738928223118238461843e-17"
    );
    let p = Distribution::pareto(ctx.int(2), ctx.rational(7, 10_000_000_000));
    assert_eq!(
        digits(&p.cdf(&ctx.int(12)), 30),
        "1.25423162767309001198331381117e-9"
    );
    let g = Distribution::geometric(ctx.rational(51, 146));
    assert_eq!(g.isf_f64(1e-200)?, 1072.0);
    Ok(())
}

/// A hypergeometric population of one: the only draw is determined, so
/// the variance is 0; the formula `n (m/N) ((N−m)/N) ((N−n)/(N−1))` is
/// `0/0` there, and up to 0.56 the variance of `Hypergeometric(1, 1, 1)`
/// (and `(1, 1, 0)`, `(1, 0, 1)`) was NaN.  scipy 1.18.1 also gives `nan`
/// (`hypergeom(1, 1, 1).var()`); the exact sum over the one atom is 0.
#[test]
fn hypergeometric_population_of_one_has_zero_variance() {
    let ctx = Context::new();
    for (m, n) in [(1, 1), (1, 0), (0, 1)] {
        let h = Distribution::hypergeometric(ctx.int(1), ctx.int(m), ctx.int(n));
        assert_eq!(h.variance(), ctx.int(0), "Hypergeometric(1, {m}, {n})");
    }
}
