//! Tests of session 18, package s18h (fuzz_findings): the wrong answers among the 61 inputs that
//! the randomized-test campaign of the statistics module saved (`fuzz/fuzz_targets/fuzz_stats.rs`;
//! the triage is `fuzz_stats_findings.md` of the session).  One test per failed check of the
//! campaign (`a`, `b`, `c`) and one per class of wrong answer found while triaging them.
//!
//! * Every test is a **confirmed** defect, `#[ignore = "known bug: …"]`: the test states what is
//!   right (the right value, or a refusal), fails today at the assertion that states the defect,
//!   and its doc comment says what is returned today.  The fix removes the `ignore`.
//! * "Today" is the release candidate `7c571e5` (0.60.1), release build; 0.60.0 returns the same
//!   value in every case here (`run_ab.sh`, `run_ab2.sh`, `run_hand.sh` of the package's scratch,
//!   run on both trees).
//! * The slow calls of the campaign have no test here: their fixes add CPU-budget tests.
//!
//! Every expected value names its oracle: mpmath 1.3.0 at 40 and 80 digits, SciPy 1.18.1, or an
//! exact argument.  The calls are in `oracle.py` and `oracle2.py` of the package's scratch; their
//! output is quoted here.

#![allow(unused_imports)]
use symplex::definite::QuadRule;
use symplex::prelude::*;
use symplex::stats::{Distribution, RandomVariable, sum_distribution};

/// `∫ g dx` over `[lo, hi]` by `integrate_numeric`; a refusal is `None`.
fn fast(g: &Ex, x: &Ex, lo: &Ex, hi: &Ex) -> Option<f64> {
    g.integrate_numeric(x, lo, hi).ok()
}

/// The same by the certified digits of `integrate_numeric_decimal`; a refusal is `None`.
fn certified(g: &Ex, x: &Ex, lo: &Ex, hi: &Ex, digits: u32) -> Option<f64> {
    g.integrate_numeric_decimal(x, &[lo, hi], digits, QuadRule::default())
        .ok()
        .map(|s| s.parse::<f64>().unwrap())
}

/// A numeric integral is the right value or a refusal.
fn right_or_refused(what: &str, got: Option<f64>, want: f64, tol: f64) {
    if let Some(v) = got {
        assert!(
            (v - want).abs() <= tol,
            "{what} = {v:e}, the integral is {want}"
        );
    }
}

/// The mixture `w·a + (1 − w)·b`.
fn mixture(ctx: &Context, w: (i64, i64), a: Distribution, b: Distribution) -> Distribution {
    let w = ctx.rational(w.0, w.1);
    let rest = ctx.one() - &w;
    Distribution::mixture(&[(w, a), (rest, b)]).unwrap()
}

// ── Numeric integration: a component of the integrand that no node sees ─────

/// Failure (a), fuzz input `crash-890b10d9870e` (check `mgf`).  `X` is the mixture
/// `1/2·Gamma(5/2, θ) + 1/2·Normal(0, 1)` with `θ = 24577/2²⁶⁸ ≈ 5.18·10⁻⁷⁷`, and
/// `Re E[e^{isX}] = ∫ cos(sx)·f(x) dx` over ℝ at `s = −1592262918131443/2⁵³`.  Half of the mass,
/// the gamma component within `10⁻⁷⁵` of 0, is seen by no node.
///
/// Today `integrate_numeric` is `Ok(0.4922482185027043)` (`integrate_numeric_with`: claimed
/// error `1.4·10⁻¹²`), `integrate_numeric_decimal` certifies `0.492248218503` at 12 digits and
/// `0.492248218502704204357193640772` at 30: thirty digits of `exp(−s²/2)/2`, the normal half
/// alone.  The closed `characteristic_function(s)` of the same law is right (`0.9922482185027042`).
/// The harness then splits the range at nine quantiles of the law and gets `0.7422482185032635`
/// certified: one of the two pieces that share the gamma component is missed again.
///
/// Oracle: mpmath, `(exp(-s*s/2) + (1 - 1j*theta*s)**(-mpf(5)/2))/2` with `mp.dps = 40` and `80`
/// = `0.992248218502704204357193640772 − 1.1450515932685846598e-77·i`; the same by `mp.quad`
/// of the two components (the gamma one in `u = x/θ`) = `0.992248218502704204357193640772`.
#[test]
#[ignore = "known bug: the numeric integral of cos(s x)·f for a mixture with a Gamma component of scale 5e-77 is certified as 0.4922…, the value is 0.9922…"]
fn failure_a_the_cf_integral_of_a_mixture_with_a_narrow_gamma_component() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let theta = ctx.parse("24577/2^268").unwrap();
    let d = mixture(
        &ctx,
        (1, 2),
        Distribution::gamma(ctx.rational(5, 2), theta),
        Distribution::normal(ctx.int(0), ctx.int(1)),
    );
    let s = ctx.parse("-1592262918131443/2^53").unwrap();
    let g = (&s * &x).cos() * d.density(&x);
    let (lo, hi) = (ctx.neg_infinity(), ctx.infinity());
    const RE_CF: f64 = 0.992_248_218_502_704_2;
    right_or_refused("integrate_numeric", fast(&g, &x, &lo, &hi), RE_CF, 1e-8);
    right_or_refused(
        "integrate_numeric_decimal, 12 digits",
        certified(&g, &x, &lo, &hi, 12),
        RE_CF,
        1e-8,
    );
}

/// The class of failure (a) with plain parameters: the density of
/// `1/2·Gamma(5/2, 10⁻²⁰) + 1/2·Normal(0, 1)` over ℝ.  Today `integrate_numeric` is `Ok(0.5)`
/// (claimed error `1.2·10⁻¹¹`) and `integrate_numeric_decimal` certifies `0.5` at 12 and at 30
/// digits.  With a scale of `10⁻⁶` only `integrate_numeric` is wrong (`0.5`; the certified digits
/// are `1`); with `Exponential(10²⁰)` in place of the gamma law likewise.
///
/// On a piece next to the component two certified answers contradict each other: over
/// `[10⁻⁷⁶, 10⁻⁷⁰]`, for the law of failure (a), 12 digits are `1.9947094073e-71` (the normal
/// part alone) and 30 digits are `0.284902603577839294091089692946` (right).
///
/// Oracle: the definition (two densities with weights 1/2 and 1/2: total mass 1).  The piece:
/// mpmath, `gammainc(mpf(5)/2, a/theta, b/theta, regularized=True)/2 + (ncdf(b) - ncdf(a))/2`
/// = `0.284902603577839294091089692946` (`mp.dps = 40`, `80`), of which the normal part is
/// `1.99470940733e-71`.
#[test]
#[ignore = "known bug: the density of 1/2·Gamma(5/2, 1e-20) + 1/2·Normal(0, 1) integrates to 0.5, by integrate_numeric and certified"]
fn a_narrow_component_of_a_mixture_is_not_lost_by_the_numeric_integrals() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (lo, hi) = (ctx.neg_infinity(), ctx.infinity());
    let d = mixture(
        &ctx,
        (1, 2),
        Distribution::gamma(ctx.rational(5, 2), ctx.parse("1/10^20").unwrap()),
        Distribution::normal(ctx.int(0), ctx.int(1)),
    );
    let f = d.density(&x);
    right_or_refused("integrate_numeric", fast(&f, &x, &lo, &hi), 1.0, 1e-7);
    right_or_refused(
        "integrate_numeric_decimal, 12 digits",
        certified(&f, &x, &lo, &hi, 12),
        1.0,
        1e-7,
    );

    // The piece [1e-76, 1e-70] of the law of failure (a): 12 and 30 certified digits agree.
    let a = mixture(
        &ctx,
        (1, 2),
        Distribution::gamma(ctx.rational(5, 2), ctx.parse("24577/2^268").unwrap()),
        Distribution::normal(ctx.int(0), ctx.int(1)),
    );
    let fa = a.density(&x);
    let (p, q) = (ctx.parse("1/10^76").unwrap(), ctx.parse("1/10^70").unwrap());
    const PIECE: f64 = 0.284_902_603_577_839_3;
    right_or_refused(
        "integrate_numeric_decimal over [1e-76, 1e-70], 12 digits",
        certified(&fa, &x, &p, &q, 12),
        PIECE,
        1e-9,
    );
    right_or_refused(
        "integrate_numeric_decimal over [1e-76, 1e-70], 30 digits",
        certified(&fa, &x, &p, &q, 30),
        PIECE,
        1e-9,
    );
}

/// Failure (b), fuzz input `crash-c24dc660f094` (check `density`; it fails in the sanitizer
/// build only, see the triage).  The density of `1/2·Laplace(0, 1/10) + 1/2·FDistribution(d₁, 1)`
/// with `d₁ = 2⁻⁹⁰⁰ ≈ 1.2·10⁻²⁷¹`, over ℝ.  The F component has its mass where
/// `ln x ≈ −10²⁷¹`: below every floating-point range, so no quadrature can see it, and the right
/// answer of a numeric integration is a refusal (or 1 from the closed form).
///
/// Today `integrate_numeric` is `Ok(0.25)` (claimed error `1.3·10⁻¹¹`) and
/// `integrate_numeric_decimal` certifies `0.25` at 12 and at 30 digits: the F half is missing
/// and so is half of the Laplace component (the next test).
///
/// Oracle: the definition (total mass 1); mpmath, `quad(lambda x: 5*exp(-10*abs(x)), [-inf, 0,
/// inf])` = `1.0` and `betainc(d1/2, mpf(1)/2, 0, 1, regularized=True)` = `1.0`.
///
/// Since 0.60.1 both calls refuse: the range is split at `−2⁹⁰⁰` (a pole of the F density's
/// formula) and at 0, and the piece between them, no node of which sees the Laplace component
/// next to 0, is no longer summed as 0 (`v36_s18q_quadrature`).  The F half itself, hidden
/// behind the Laplace component on `[0, ∞)`, is still not seen (the two tests above).
#[test]
fn failure_b_the_density_of_a_mixture_with_an_f_law_of_a_tiny_first_parameter() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (lo, hi) = (ctx.neg_infinity(), ctx.infinity());
    let d = mixture(
        &ctx,
        (1, 2),
        Distribution::laplace(ctx.int(0), ctx.rational(1, 10)),
        Distribution::f_distribution(ctx.parse("1/2^900").unwrap(), ctx.int(1)),
    );
    let f = d.density(&x);
    right_or_refused("integrate_numeric", fast(&f, &x, &lo, &hi), 1.0, 1e-7);
    right_or_refused(
        "integrate_numeric_decimal, 12 digits",
        certified(&f, &x, &lo, &hi, 12),
        1.0,
        1e-7,
    );
}

/// Found with failure (b): a range from `−∞` across the point 0, where the F component is
/// singular, loses the part left of 0 as well.  `X ~ 1/2·Normal(0, 1) + 1/2·FDistribution(10⁻²⁷⁰,
/// 1)`, `P(X ≤ 1) = ∫ f` over `(−∞, 1]`.
///
/// Today `integrate_numeric` is `Ok(0.17067237303427146)` (claimed error `1.9·10⁻¹⁵`) and 12
/// certified digits are `0.170672373034`: that is `(Φ(1) − 1/2)/2`, the normal component on
/// `[0, 1]` alone.  The normal component on `(−∞, 0)`, `0.25`, is missing besides the F half;
/// over `(−∞, 0]` and over `[−1, 1]` the same normal parts are found (`0.25`, `0.3413447460685`).
///
/// Oracle: mpmath, `ncdf(1)/2 + betainc(d1/2, mpf(1)/2, 0, d1/(d1 + 1), regularized=True)/2` =
/// `0.9206723730342714742926163` (`mp.dps = 40`, `80`); SciPy, `norm.cdf(1)/2 + f.cdf(1, 1e-270,
/// 1)/2` = `0.9206723730342714`.  `ncdf(1)/2` = `0.42067237303427147429`, `(ncdf(1) − 1/2)/2` =
/// `0.17067237303427147429`.
///
/// Since 0.60.1 both calls refuse (the cause and the fix: `v36_s18q_quadrature`,
/// `a_piece_every_node_of_which_gave_zero_is_not_summed_as_zero`).
#[test]
fn an_integral_from_minus_infinity_across_a_singular_point_keeps_its_left_part() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let d = mixture(
        &ctx,
        (1, 2),
        Distribution::normal(ctx.int(0), ctx.int(1)),
        Distribution::f_distribution(ctx.parse("1/10^270").unwrap(), ctx.int(1)),
    );
    let f = d.density(&x);
    let (lo, hi) = (ctx.neg_infinity(), ctx.int(1));
    const P: f64 = 0.920_672_373_034_271_5;
    right_or_refused("integrate_numeric", fast(&f, &x, &lo, &hi), P, 1e-7);
    right_or_refused(
        "integrate_numeric_decimal, 12 digits",
        certified(&f, &x, &lo, &hi, 12),
        P,
        1e-7,
    );
}

/// Fuzz input `slow-unit-4b2cd2661eb7` (check `density`, verdict "skip"): a density at a huge
/// scale.  Today `integrate_numeric` of the density of `Normal(0, 10⁸⁰)` over ℝ is
/// `Ok(6.130059011750917e-66)` (`integrate_numeric_with`: claimed error `4.8·10⁻⁶²`, which is
/// larger than the value and below the absolute tolerance), and of the input's own law,
/// `Weibull(4·10⁷⁶, 27/8)` over `(0, ∞)`, `Ok(2.6163788076620492e-250)`.  Likewise
/// `Exponential(10⁻⁸⁰)` (`7.7e-66`), `Gamma(2, 10⁸⁰)` (`1.8e-131`), `Weibull(10⁸⁰, 2)`
/// (`3.6e-131`), `Normal(0, 10⁴⁰)` (`6.1e-26`); at a scale of `10²⁰` the value is right.  The
/// certified digits are `1` for the normal and the exponential law and a refusal for the others.
///
/// Oracle: the definition (total mass 1); mpmath, `quad` of the standard density in `u = x/σ`
/// = `1.0` for the normal and the Weibull law (`mp.dps = 40`, `80`).
///
/// Since 0.60.1 both are refused: an estimate that is not larger than its own error estimate is
/// decided by the certified route or not at all (`v36_s18q_quadrature`,
/// `an_estimate_not_larger_than_its_error_estimate_is_not_returned`).
#[test]
fn the_density_of_a_law_of_a_huge_scale_does_not_integrate_to_almost_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let normal = Distribution::normal(ctx.int(0), ctx.parse("10^80").unwrap());
    right_or_refused(
        "integrate_numeric, Normal(0, 1e80)",
        fast(
            &normal.density(&x),
            &x,
            &ctx.neg_infinity(),
            &ctx.infinity(),
        ),
        1.0,
        1e-7,
    );
    let weibull = Distribution::weibull(ctx.parse("4*10^76").unwrap(), ctx.rational(27, 8));
    right_or_refused(
        "integrate_numeric, Weibull(4e76, 27/8)",
        fast(&weibull.density(&x), &x, &ctx.int(0), &ctx.infinity()),
        1.0,
        1e-7,
    );
}

// ── Sums of independent variables ───────────────────────────────────────────

/// Failure (c), fuzz input `crash-38c282dac544` (check `sums`).  `X ~ Gamma(55/1048576, 2·ln 2)`
/// and `Y ~ Exponential(1/(2·ln 2))` are gamma laws of one scale, so `X + Y ~ Gamma(55/1048576 +
/// 1, 2·ln 2)`.  Today `sum_distribution` is `None`: it compares the scales structurally, the
/// scale of the exponential law is `(1/rate).simplify()`, and `1/(1/(2·ln 2))` simplifies to
/// `ln(4)`, not to `2·ln(2)`.  The shape plays no part: every scale `k·ln n` with an integer
/// `k ≥ 2` fails (`3·ln 2`, `2·ln 3`, …), for every shape tried; `ln 2`, `3/2·ln 2`, and the
/// multiples of `√2`, `π`, `e`, `π²` are recognised (`probe/bins/sumc.rs`).
///
/// Oracle: exact (the moment generating functions multiply: `(1 − θt)^(−k)·(1 − θt)^(−1)`);
/// mpmath, `(mpf(55)/1048576 + 1)*2*log(2)` = `1.386367075152885455626491`.
#[test]
fn failure_c_a_gamma_and_an_exponential_law_of_the_scale_two_ln_two_add() {
    let ctx = Context::new();
    let theta = ctx.int(2).ln() * ctx.int(2);
    let rate = (ctx.one() / &theta).simplify();
    let x = RandomVariable::new(
        &ctx,
        "X",
        Distribution::gamma(ctx.rational(55, 1048576), theta),
    );
    let y = RandomVariable::new(&ctx, "Y", Distribution::exponential(rate));
    let sum = sum_distribution(&x, &y);
    assert!(
        sum.is_some(),
        "Gamma(55/1048576, 2·ln 2) + Exponential(1/(2·ln 2)) is not recognised as a gamma law"
    );
    let mean = sum.unwrap().mean().eval_f64().unwrap();
    assert!((mean - 1.386_367_075_152_885_5).abs() < 1e-14, "{mean}");
}
