//! Far tails of the symbolic distributions keep their digits.
//!
//! Before 0.28 a tail probability was built as `1 − F` or `F(hi) − F(lo)`
//! from the classic closed forms (`½ + ½ erf(z/√2)`, `1 − Σ pmf`), whose
//! terms agree to hundreds of digits in a far tail; since 0.26 `evalf`
//! returns such a difference as `0` once its precision budget is spent.
//! Every value below was `0.0` (or `"0"`, or an error) in 0.27.0.  Now a
//! family supplies a survival function and a lower-tail CDF that do not
//! cancel (`½ erfc`, `Γ(k, x)/Γ(k)`, `I_{1−x}(β, α)`, `atan(γ/(x − x₀))/π`),
//! the generic machinery uses them in a far tail, and `eval` no longer
//! folds `γ(s, x)` into `Γ(s) − Γ(s, x)` where that difference cancels.
//!
//! Reference values: mpmath 1.3 at `mp.dps = 50` with exact inputs, by the
//! call quoted (`target/scratch/tails_oracle.py` in the 0.28 session).

// Reference values are quoted at the 20 digits mpmath printed them.
#![allow(clippy::excessive_precision)]

use symplex::prelude::*;
use symplex::stats::order::maximum_of;
use symplex::stats::{Distribution, Kind, Piece, Support};

/// `actual` within `rel` of `expected`, relatively.
fn close(actual: f64, expected: f64, rel: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= rel * expected.abs(),
        "{label}: got {actual:e}, expected {expected:e}"
    );
}

/// A value below the `f64` range, compared through its decimal expansion:
/// `eval_decimal(20)` against mpmath's digits, relatively.
fn close_decimal(e: &Ex, expected: &str, rel: f64, label: &str) {
    let got = e.eval_decimal(20).unwrap();
    let split = |s: &str| -> (f64, i64) {
        let (m, x) = s.split_once('e').unwrap_or((s, "0"));
        (m.parse().unwrap(), x.parse().unwrap())
    };
    let ((gm, gx), (em, ex)) = (split(&got), split(expected));
    // Normalise both mantissas to [1, 10).
    let norm = |m: f64, x: i64| -> (f64, i64) {
        let shift = m.abs().log10().floor() as i64;
        (m / 10f64.powi(shift as i32), x + shift)
    };
    let ((gm, gx), (em, ex)) = (norm(gm, gx), norm(em, ex));
    assert!(
        gx == ex && (gm - em).abs() <= rel * em.abs(),
        "{label}: got {got}, expected {expected}"
    );
}

fn above(d: &Distribution, a: &Ex) -> Ex {
    let ctx = d.context();
    d.probability_of(&Support::from_pieces(
        Kind::Continuous,
        vec![Piece::Interval(Interval::open(a.clone(), ctx.infinity()))],
    ))
    .unwrap()
}

fn between(d: &Distribution, a: &Ex, b: &Ex) -> Ex {
    d.probability_of(&Support::from_pieces(
        Kind::Continuous,
        vec![Piece::Interval(Interval::open(a.clone(), b.clone()))],
    ))
    .unwrap()
}

/// `P(N > 20)` for a standard normal was `½ − ½ erf(10√2)`, which
/// evaluated to `0.0`.  It is now `½ erfc(10√2)`, through `probability_of`
/// and through the new `Distribution::sf`.
#[test]
fn normal_upper_tail() {
    let ctx = Context::new();
    let n = Distribution::normal(ctx.int(0), ctx.int(1));
    // mpmath: ncdf(-20) = 2.7536241186062336951e-89
    let want = 2.753_624_118_606_233_695_1e-89;
    close(
        above(&n, &ctx.int(20)).eval_f64().unwrap(),
        want,
        1e-14,
        "P(N > 20)",
    );
    let sf = n.sf(&ctx.int(20));
    assert_eq!(
        sf,
        ctx.rational(1, 2) * (ctx.int(10) * ctx.int(2).sqrt()).erfc()
    );
    close(sf.eval_f64().unwrap(), want, 1e-14, "sf(20)");
    // An interval inside the tail: S(20) − S(21), not F(21) − F(20).
    // mpmath: ncdf(-20) - ncdf(-21) = 2.7536241153269556761e-89
    close(
        between(&n, &ctx.int(20), &ctx.int(21)).eval_f64().unwrap(),
        2.753_624_115_326_955_676_1e-89,
        1e-14,
        "P(20 < N < 21)",
    );
}

/// The lower tail: `Normal(2, 3).cdf(−100)` was `0.0` (`½ + ½ erf(−34/√2)`).
#[test]
fn normal_lower_tail() {
    let ctx = Context::new();
    let n = Distribution::normal(ctx.int(2), ctx.int(3));
    // mpmath: ncdf(-34) = 1.1138987855743793866e-253
    close(
        n.cdf(&ctx.int(-100)).eval_f64().unwrap(),
        1.113_898_785_574_379_386_6e-253,
        1e-14,
        "Normal(2, 3).cdf(-100)",
    );
    let std = Distribution::normal(ctx.int(0), ctx.int(1));
    // mpmath: ncdf(-40) = 3.6558935409150297037e-350 (below the f64 range)
    close_decimal(
        &std.cdf(&ctx.int(-40)),
        "3.6558935409150297037e-350",
        1e-15,
        "cdf(-40)",
    );
}

/// Around the median the classic forms stay: `P(−1 < N < 1)` is
/// `½ erf(√2/2) − ½ erf(−√2/2)` (as in 0.27; `erf(−a)` is not folded to
/// `−erf(a)`), not a mixture of `erf` and `erfc`.
#[test]
fn central_masses_keep_their_classic_form() {
    let ctx = Context::new();
    let n = Distribution::normal(ctx.int(0), ctx.int(1));
    let m = between(&n, &ctx.int(-1), &ctx.int(1));
    let a = ctx.int(2).sqrt() / 2;
    let classic = ctx.rational(1, 2) * a.erf() - ctx.rational(1, 2) * (-&a).erf();
    assert_eq!(m.equals(&classic), Some(true), "{m}");
    // mpmath: erf(sqrt(2)/2) = 0.68268949213708589717
    close(
        m.eval_f64().unwrap(),
        0.682_689_492_137_085_897_17,
        1e-15,
        "P(-1 < N < 1)",
    );
    let t = ctx.symbol("t");
    assert_eq!(
        n.cdf(&t),
        ctx.rational(1, 2) + ctx.rational(1, 2) * (&t / ctx.int(2).sqrt()).erf()
    );
}

/// Log-normal and Cauchy, both tails.  The log-normal tails were `0`
/// (the normal's `½ ± ½ erf`), the Cauchy ones `0` (`½ ± atan(z)/π` with
/// `|z| = 10¹⁰⁰`).
#[test]
fn log_normal_and_cauchy_tails() {
    let ctx = Context::new();
    let ln = Distribution::log_normal(ctx.int(0), ctx.int(1));
    let big = ctx.int(10).powi(20);
    // mpmath: ncdf(-log(mpf(10)**20)) = 2.6329416885455207121e-463
    close_decimal(
        &above(&ln, &big),
        "2.6329416885455207121e-463",
        1e-15,
        "P(LN > 1e20)",
    );
    close_decimal(
        &ln.cdf(&(ctx.one() / &big)),
        "2.6329416885455207121e-463",
        1e-15,
        "LN cdf(1e-20)",
    );
    let c = Distribution::cauchy(ctx.int(0), ctx.int(1));
    let huge = ctx.int(10).powi(100);
    // mpmath: atan(1/mpf(10)**100)/pi = 3.1830988618379067154e-101
    let want = 3.183_098_861_837_906_715_4e-101;
    close(
        above(&c, &huge).eval_f64().unwrap(),
        want,
        1e-14,
        "P(C > 1e100)",
    );
    close(
        c.cdf(&(-&huge)).eval_f64().unwrap(),
        want,
        1e-14,
        "C cdf(-1e100)",
    );
}

/// Gamma: the upper tail of a non-integer shape was `1 − γ(⅓, 800)/Γ(⅓)`
/// (`0`), the lower tail of an integer shape the closed form `1 − e⁻ˣ(1 +
/// x + …)` at `x = 10⁻³⁰` (`0`).  Exponential likewise at `10⁻¹⁰⁰`.
#[test]
fn gamma_and_exponential_tails() {
    let ctx = Context::new();
    let g = Distribution::gamma(ctx.rational(1, 3), ctx.int(1));
    // mpmath: gammainc(mpf(1)/3, 800, inf, regularized=True) = 1.5874391710257242683e-350
    close_decimal(
        &above(&g, &ctx.int(800)),
        "1.5874391710257242683e-350",
        1e-15,
        "P(G > 800)",
    );
    let g5 = Distribution::gamma(ctx.int(5), ctx.int(1));
    let tiny = ctx.one() / ctx.int(10).powi(30);
    // mpmath: gammainc(5, 0, 1/mpf(10)**30, regularized=True) = 8.3333333333333333333e-153
    close(
        g5.cdf(&tiny).eval_f64().unwrap(),
        8.333_333_333_333_333_333_3e-153,
        1e-14,
        "Gamma(5).cdf(1e-30)",
    );
    let e = Distribution::exponential(ctx.int(1));
    // mpmath: -expm1(-1/mpf(10)**100) = 1.0e-100
    close(
        e.cdf(&(ctx.one() / ctx.int(10).powi(100)))
            .eval_f64()
            .unwrap(),
        1e-100,
        1e-14,
        "Exponential(1).cdf(1e-100)",
    );
}

/// Discrete upper tails: Poisson was `1 − e⁻¹ Σ_{k ≤ 60} 1/k!` (`0`), the
/// negative binomial an infinite `Sum` that did not evaluate at all.
#[test]
fn discrete_upper_tails() {
    let ctx = Context::new();
    let p1 = Distribution::poisson(ctx.int(1));
    // mpmath: gammainc(61, 0, 1, regularized=True) = 7.3664939666194517153e-85
    close(
        above(&p1, &ctx.int(60)).eval_f64().unwrap(),
        7.366_493_966_619_451_715_3e-85,
        1e-14,
        "Poisson(1): P(X > 60)",
    );
    let p3 = Distribution::poisson(ctx.rational(1, 3));
    // mpmath: gammainc(61, 0, mpf(1)/3, regularized=True) = 1.116027676521420451e-113
    close(
        above(&p3, &ctx.int(60)).eval_f64().unwrap(),
        1.116_027_676_521_420_451e-113,
        1e-14,
        "Poisson(1/3): P(X > 60)",
    );
    let nb = Distribution::negative_binomial(ctx.int(3), ctx.rational(1, 2));
    // mpmath: betainc(2001, 3, 0, mpf(1)/2, regularized=True) = 2.1850811587270834235e-597
    close_decimal(
        &above(&nb, &ctx.int(2000)),
        "2.1850811587270834235e-597",
        1e-15,
        "NegBin(3, 1/2): P(X > 2000)",
    );
}

/// The wrappers transport the non-cancelling forms: a truncation in the
/// far upper tail, a decreasing affine map (the inner upper tail becomes
/// the lower one), a mixture and an order statistic.  All four were `0`.
#[test]
fn wrappers_keep_the_tails() {
    let ctx = Context::new();
    let n = Distribution::normal(ctx.int(0), ctx.int(1));
    let region = Support::from_pieces(
        Kind::Continuous,
        vec![Piece::Interval(Interval::open(ctx.int(10), ctx.infinity()))],
    );
    let t = n.truncated(&region).unwrap();
    // mpmath: ncdf(-30)/ncdf(-10) = 6.439381326109969604e-175
    close(
        above(&t, &ctx.int(30)).eval_f64().unwrap(),
        6.439_381_326_109_969_604e-175,
        1e-14,
        "N | N > 10: P(X > 30)",
    );
    close(
        t.sf(&ctx.int(30)).eval_f64().unwrap(),
        6.439_381_326_109_969_604e-175,
        1e-14,
        "N | N > 10: sf(30)",
    );
    let m2 = n.affine(ctx.int(-2), ctx.int(0)).unwrap();
    // P(−2N ≤ −40) = P(N ≥ 20); mpmath: ncdf(-20) = 2.7536241186062336951e-89
    close(
        m2.cdf(&ctx.int(-40)).eval_f64().unwrap(),
        2.753_624_118_606_233_695_1e-89,
        1e-14,
        "-2N cdf(-40)",
    );
    let mix = Distribution::mixture(&[
        (ctx.rational(1, 2), n.clone()),
        (
            ctx.rational(1, 2),
            Distribution::normal(ctx.int(1), ctx.int(1)),
        ),
    ])
    .unwrap();
    // mpmath: ncdf(-30)/2 + ncdf(-29)/2 = 1.6448926333524354166e-185
    close(
        above(&mix, &ctx.int(30)).eval_f64().unwrap(),
        1.644_892_633_352_435_416_6e-185,
        1e-14,
        "mixture: P(X > 30)",
    );
    let max3 = maximum_of(&n, 3).unwrap();
    // P(max > 20) = 3p − 3p² + p³, p = ncdf(-20); mpmath: 8.2608723558187010852e-89
    close(
        max3.sf(&ctx.int(20)).eval_f64().unwrap(),
        8.260_872_355_818_701_085_2e-89,
        1e-14,
        "max of 3 normals: sf(20)",
    );
}

/// `eval` folds `γ(s, x)` for an integer or half-integer `s ≤ 64` into
/// `Γ(s) − Γ(s, x)` — except where that difference would cancel: at a
/// rational `x` well below `s`, the fold used to return an exact
/// expression that evaluates to `0` (`γ(5, 10⁻³⁰)`).  Elsewhere the closed
/// form stays, as SymPy writes it.
#[test]
fn lowergamma_is_not_folded_into_a_cancelling_difference() {
    let ctx = Context::new();
    let tiny = ctx.one() / ctx.int(10).powi(30);
    let kept = tiny.lowergamma(&ctx.int(5));
    assert_eq!(kept.eval(), kept, "γ(5, 1e-30) stays");
    // mpmath: gammainc(5, 0, 1/mpf(10)**30) = 2.0e-151 (24 × 8.3333e-153)
    close(kept.eval_f64().unwrap(), 2e-151, 1e-14, "γ(5, 1e-30)");
    let poisson_tail = ctx.int(1).lowergamma(&ctx.int(61));
    assert_eq!(poisson_tail.eval(), poisson_tail, "γ(61, 1) stays");
    // SymPy: lowergamma(3, 1) = 2 - 5*exp(-1)
    let folded = ctx.int(1).lowergamma(&ctx.int(3)).eval();
    assert_eq!(
        folded.equals(&(ctx.int(2) - ctx.int(5) * (-ctx.one()).exp())),
        Some(true)
    );
}
