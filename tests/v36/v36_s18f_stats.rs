//! Tests of session 18. Lane F of the 0.60.1 block: the slow inputs of the nightly `fuzz_stats` job (work budgets
//! on the numeric fallbacks of the statistics code).
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use std::time::Duration;

use symplex::prelude::*;
use symplex::stats::{Distribution, Support};

use crate::perf_assert::CpuTimer;

/// The value of `e` to `digits` digits, or the error text.
fn dec(e: &Ex, digits: u32) -> String {
    e.eval_decimal(digits)
        .unwrap_or_else(|err| format!("refused: {err}"))
}

/// `|a/b − 1|`.
fn rel(a: f64, b: f64) -> f64 {
    (a / b - 1.0).abs()
}

// ── The root search of `quantile_f64` (fuzz input `8ac84c11fc4e`) ───────────

/// `StudentT(1)` (a Cauchy law) truncated to `[−3.183098861837907·10²⁹⁹,
/// 3.0776835371752544]`, the law of the fuzz input `AAoAAADOzs7Ozis=`.
fn cauchy_truncated_over_three_hundred_decades(ctx: &Context) -> Distribution {
    let lo = ctx.from_f64(-3.183_098_861_837_907e299).unwrap();
    let hi = ctx.from_f64(3.077_683_537_175_254_4).unwrap();
    Distribution::student_t(ctx.int(1))
        .truncated(&Support::interval(lo, hi))
        .unwrap()
}

/// The quartile of a Cauchy law truncated to `[−3.2·10²⁹⁹, 3.08]`.  Up to
/// 0.60.0 `quantile_f64(0.25)` took 12.1 s in a release build (minutes for
/// the nine quantiles the fuzzer asks for): without a starting estimate the
/// bracket was grown from the middle of the support, `[−1.2·10²⁹⁹, 0]`, and
/// Brent's method halved its *width* about a thousand times towards the
/// root at `−1.17`, each halving a certified evaluation of the distribution
/// function (12 ms, 10 ms of it a `simplify` of the constant).  The bracket
/// is now narrowed in the order of the floats first (`narrow_bracket`, at
/// most 12 probes), Brent's method has 128 steps, and a constant is not
/// simplified: 0.12 s in a release build.
///
/// Oracle: mpmath at 700 and 900 digits, `tan(π(F − ½))` with `F = F(lo) +
/// p·(F(hi) − F(lo))`, `F(t) = ½ + atan(t)/π` at the exact doubles
/// (`target/scratch/oracle/o1.py`): `p = 0.25`: −1.170849566112539166097647,
/// `p = 0.75`: 0.6128007881399320995128756, `p = 10⁻³⁰⁰`:
/// −1.675315190441003568012332·10²⁹⁹.
#[test]
fn quartile_of_a_cauchy_law_truncated_over_three_hundred_decades() {
    let ctx = Context::new();
    let d = cauchy_truncated_over_three_hundred_decades(&ctx);
    let timer = CpuTimer::start();
    let q1 = d.quantile_f64(0.25).unwrap();
    let q3 = d.quantile_f64(0.75).unwrap();
    let far = d.quantile_f64(1e-300).unwrap();
    timer.assert_within(
        Duration::from_secs(20),
        "three quantiles of a truncated Cauchy law",
    );
    assert!(rel(q1, -1.170_849_566_112_539) < 1e-15, "{q1:e}");
    assert!(rel(q3, 0.612_800_788_139_932_1) < 1e-15, "{q3:e}");
    assert!(rel(far, -1.675_315_190_441_003_6e299) < 1e-15, "{far:e}");
}

/// A level that the tail meets only within one float of a truncation point:
/// `P(X > x) = 10⁻³⁰⁰` for the same law, whose upper end is `3.0776835371752544`.
/// The exact root is `3.077683537175254446793815` (mpmath, 700 and 900 digits,
/// `o1.py`), `2·10⁻²⁹⁹` below the end, so the answer is the end itself, the
/// double nearest to it.  Up to 0.60.0 the search found it by bisecting the
/// floats below the end (0.34 s in a release build); one probe at the float
/// next to the end decides it now.
#[test]
fn a_level_met_within_one_float_of_a_truncation_point() {
    let ctx = Context::new();
    let d = cauchy_truncated_over_three_hundred_decades(&ctx);
    let timer = CpuTimer::start();
    let x = d.isf_f64(1e-300).unwrap();
    timer.assert_within(Duration::from_secs(10), "an upper quantile at the end");
    assert_eq!(x, 3.077_683_537_175_254_4);
}

// ── Student t and F at an extreme shape (fuzz input `facb3dd74510`) ─────────

/// The distribution function of `StudentT(ν)` for `ν = 6.584062344193146·10²³⁶`
/// at `−3.106667654710634` and `−4.169569323349106` (the fuzz input
/// `uymICuU9PQ==`).  The values were right, and each took 0.19 s and 0.11 s to
/// evaluate — twice per call, since `cdf` evaluated the classic form once more
/// only to ask whether the point lies in a far tail, although the family has
/// no other form of that tail: 5.4 s for the fuzzer's comparison with the
/// `f64` kernel.  The incomplete beta function was written `I_z(ν/2, ½)` at
/// `z = ν/(ν + x²) = 1 − 1.5·10⁻²³⁶`, which keeps none of the digits of
/// `1 − z`; it is now `I_{[w, 1]}(½, ν/2)` at `w = x²/(ν + x²)`
/// (`continuous::reg_beta_at`), and no evaluation is made to choose a form
/// that does not exist: 0.03 s for the two calls.
///
/// Oracle: mpmath, the integral of the density `exp(lnΓ((ν+1)/2) − lnΓ(ν/2) −
/// ½ln(νπ) − (ν+1)/2·log1p(t²/ν))` over `(−∞, x]` at 320 and 400 digits, and
/// the normal limit `ncdf(x)` at 60 and 90 digits, which agree to the 28
/// digits printed (`o1.py`): 0.0009460449218750004096101301 and
/// 0.00001525878906249999739876508.
#[test]
fn student_t_distribution_function_at_a_huge_number_of_degrees_of_freedom() {
    let ctx = Context::new();
    let d = Distribution::student_t(ctx.from_f64(6.584_062_344_193_146e236).unwrap());
    let timer = CpuTimer::start();
    let x = ctx.from_f64(-3.106_667_654_710_634).unwrap();
    let (f, s) = (d.cdf(&x), d.sf(&x));
    assert_eq!(dec(&f, 25), "0.0009460449218750004096101301");
    assert_eq!(f.eval_f64().unwrap(), 9.460_449_218_750_004e-4);
    assert_eq!(s.eval_f64().unwrap(), 0.999_053_955_078_125);
    let x = ctx.from_f64(-4.169_569_323_349_106).unwrap();
    let f = d.cdf(&x);
    assert_eq!(dec(&f, 25), "1.525878906249999739876508e-5");
    timer.assert_within(
        Duration::from_secs(10),
        "the distribution function of StudentT(6.6e236) at two points",
    );
    // The argument that carries the digits: x²/(ν + x²), from w to 1.
    let text = f.to_string();
    assert!(
        text.starts_with("1/2*betainc_regularized(1/2, ") && text.ends_with(", 1)"),
        "{text}"
    );
}

/// The classic form is kept where its argument is not next to 1: a symbolic
/// point, and a number with `x²/(ν + x²) ≥ 2⁻¹⁶` (`ν = 5`, `x = −2`:
/// `w = 4/9`).  Oracle: `scipy.stats.t.cdf(-2, 5) = 0.05096973941492914`,
/// mpmath `betainc(5/2, 1/2, 0, 5/9, regularized=True)/2 =
/// 0.050969739414929105…`.
#[test]
fn student_t_classic_form_where_the_argument_is_not_next_to_one() {
    let ctx = Context::new();
    let d = Distribution::student_t(ctx.int(5));
    let f = d.cdf(&ctx.int(-2));
    assert_eq!(f.to_string(), "1/2*betainc_regularized(5/2, 1/2, 0, 5/9)");
    assert!((f.eval_f64().unwrap() - 0.050_969_739_414_929_14).abs() < 1e-16);
    let x = ctx.symbol("x");
    assert!(
        d.cdf(&x)
            .to_string()
            .contains("betainc_regularized(5/2, 1/2, 0, "),
        "{}",
        d.cdf(&x)
    );
}

// ── R6.1: a finite lattice under a map proved strictly monotone ─────────────

/// `√X` for `X` a 100,000-sided die (the fuzz input `33f51d1a95d7`).  Up to
/// 0.60.0 `transformed` formed the certified `f64` of each of the 100,000
/// images to tell them apart (1.4 s in a release build, 5.5 s in the fuzz
/// build, a timeout under the sanitizer: the fuzzer skipped lattices of more
/// than 20,000 atoms).  `√x` is proved strictly increasing on `[1, 100000]`,
/// so the images are distinct and in the order of the lattice: the table is
/// built without evaluating any, and a query finds its point among the
/// images by bisection.
///
/// Oracle: exact arithmetic.  `P(√X ≤ y) = ⌊y²⌋/100000` for `1 ≤ y² ≤
/// 100000`: `y = √30000`: 3/10; `y = 100`: 1/10; `y = 201/2` (`y² =
/// 10100.25`): 101/1000; `y = π·100` (`y² = 98696.04…`, mpmath): 98696/100000
/// = 12337/12500.  The quantile at 0.3 is the 30,000th image, `√30000 =
/// 173.20508075688772` (mpmath `sqrt(30000)`), and the upper quantile at 0.25
/// the 75,000th, `√75000 = 273.8612787525831`.
#[test]
fn square_root_of_a_die_with_a_hundred_thousand_sides() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let die = Distribution::die(ctx.int(100_000));
    let timer = CpuTimer::start();
    let y = die.transformed(&x, &x.sqrt()).unwrap();
    let at = ctx.int(30_000).sqrt();
    assert_eq!(y.cdf(&at), ctx.rational(3, 10));
    assert_eq!(y.sf(&at), ctx.rational(7, 10));
    assert_eq!(y.density(&at), ctx.rational(1, 100_000));
    // Points that are not images.
    assert_eq!(y.cdf(&ctx.int(100)), ctx.rational(1, 10));
    assert_eq!(y.cdf(&ctx.rational(201, 2)), ctx.rational(101, 1000));
    assert_eq!(y.density(&ctx.rational(201, 2)), ctx.int(0));
    assert_eq!(y.cdf(&(ctx.pi() * 100)), ctx.rational(12_337, 12_500));
    assert_eq!(y.cdf(&ctx.rational(1, 2)), ctx.int(0));
    assert_eq!(y.cdf(&ctx.int(1000)), ctx.int(1));
    assert_eq!(y.quantile_f64(0.3).unwrap(), 173.205_080_756_887_72);
    assert_eq!(y.isf_f64(0.25).unwrap(), 273.861_278_752_583_1);
    timer.assert_within(
        Duration::from_secs(60),
        "the table of √X for a 100,000-sided die and eleven queries",
    );
}

/// A strictly decreasing map reverses the order: `Y = e^{−X}` for `X`
/// uniform on `1, …, 1000`.  Oracle: exact arithmetic, `P(Y ≤ e^{−k}) =
/// P(X ≥ k) = (1001 − k)/1000` and `P(Y > e^{−k}) = (k − 1)/1000`; the
/// quantile at the level 0.9995 is the largest value, `e^{−1} =
/// 0.36787944117144233` (`P(Y ≤ e^{−2}) = 999/1000` is below the level), and
/// the upper quantile at 0.0015 is `e^{−2} = 0.1353352832366127` (`P(Y >
/// e^{−2}) = 1/1000 ≤ 0.0015 < 2/1000 = P(Y > e^{−3})`); the doubles by
/// mpmath `exp(-1)`, `exp(-2)`, and the doubles `0.9995` and `0.0015`
/// compared with the fractions exactly (`target/scratch/oracle/o2.py`).
#[test]
fn a_decreasing_map_of_a_lattice_reverses_its_order() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let u = Distribution::discrete_uniform(ctx.int(1), ctx.int(1000));
    let y = u.transformed(&x, &(-&x).exp()).unwrap();
    let image = |k: i64| (-ctx.int(k)).exp();
    assert_eq!(y.cdf(&image(751)), ctx.rational(250, 1000));
    assert_eq!(y.sf(&image(751)), ctx.rational(750, 1000));
    assert_eq!(y.density(&image(751)), ctx.rational(1, 1000));
    assert_eq!(y.cdf(&image(1)), ctx.int(1));
    assert_eq!(y.cdf(&image(1000)), ctx.rational(1, 1000));
    // Between e⁻² and e⁻¹: the 999 values below.
    assert_eq!(y.cdf(&ctx.rational(1, 5)), ctx.rational(999, 1000));
    assert_eq!(y.density(&ctx.rational(1, 5)), ctx.int(0));
    assert_eq!(y.quantile_f64(0.9995).unwrap(), 0.367_879_441_171_442_33);
    assert_eq!(y.isf_f64(0.0015).unwrap(), 0.135_335_283_236_612_7);
}

/// The masses of the lattice need not be equal: `√X` for `X ~ Binomial(400,
/// 1/3)`.  Oracle: `scipy.stats.binom.cdf(130, 400, 1/3) =
/// 0.3839637941008619`, `binom.sf(130, 400, 1/3) = 0.6160362058991381`,
/// `binom.pmf(130, 400, 1/3) = 0.03996739992845445`, `binom.ppf(0.9, 400,
/// 1/3) = 145` (`√145 = 12.041594578792296`); the mpmath sums of the exact
/// masses at 40 digits agree: 0.38396379410086140217, 0.61603620589913859783,
/// 0.039967399928454417818, and `F(144) = 0.8814 < 0.9 ≤ F(145) = 0.9009`
/// (`target/scratch/oracle/o2.py`).
#[test]
fn square_root_of_a_binomial_lattice_keeps_its_masses() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let b = Distribution::binomial(ctx.int(400), ctx.rational(1, 3));
    let y = b.transformed(&x, &x.sqrt()).unwrap();
    let at = ctx.int(130).sqrt();
    assert!(rel(y.cdf(&at).eval_f64().unwrap(), 0.383_963_794_100_861_4) < 1e-14);
    assert!(rel(y.sf(&at).eval_f64().unwrap(), 0.616_036_205_899_138_6) < 1e-14);
    assert!(rel(y.density(&at).eval_f64().unwrap(), 0.039_967_399_928_454_42) < 1e-14);
    assert_eq!(y.quantile_f64(0.9).unwrap(), 12.041_594_578_792_296);
    // The same sums as the lattice itself.
    assert_eq!(y.cdf(&at), b.cdf(&ctx.int(130)));
}

/// A map that is not monotone on the lattice goes the general route, which
/// merges equal images: `X²` for `X` uniform on `−200, …, 200`.  Oracle:
/// exact arithmetic, `P(X² = k²) = 2/401` for `k ≠ 0`, `1/401` at 0, and
/// `P(X² ≤ 100) = 21/401`.
#[test]
fn a_map_that_is_not_monotone_still_merges_equal_images() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let u = Distribution::discrete_uniform(ctx.int(-200), ctx.int(200));
    let y = u.transformed(&x, &x.powi(2)).unwrap();
    assert_eq!(y.density(&ctx.int(49)), ctx.rational(2, 401));
    assert_eq!(y.density(&ctx.int(0)), ctx.rational(1, 401));
    assert_eq!(y.cdf(&ctx.int(100)), ctx.rational(21, 401));
}

// ── The far probes of the root search (a regression of the 0.60.1 release candidate) ──

/// The first of five independent `χ²(60)` variables at the level `10⁻³⁰⁰`
/// of either tail (the fuzz input `timeout-a8786a14e786…`).  The law has no
/// starting estimate, and the first draft of the 0.60.1 root search then
/// bracketed the root by the hull of the support `[0, ∞)`: one probe at the
/// largest double (a certified evaluation of `1 − (1 − F)⁵` where `1 − F`
/// is `exp(−9·10³⁰⁷)`: 1.05 s and 1.54 s in a release build) and probes at
/// the floats halfway to it (`1.7·10¹⁵⁴`: 49 ms, `1.6·10⁷⁷`: 27 ms) — 1.06 s
/// and 1.83 s for the two calls, which took 0.074 s and 0.141 s in 0.60.0.
/// The search now probes near the start of 0.60 first and reads the sign at
/// the far floats from those probes (the miss increases with the point):
/// 0.013 s and 0.155 s, the same doubles.
///
/// Oracle: mpmath at 400 digits, bisection of `P(30, x/2) = 1 − (1 − p)^(1/5)`
/// and of `Q(30, x/2) = q^(1/5)` with `p = q =` the double `10⁻³⁰⁰`
/// (`target/scratch/oracle/o_repro.py`): `2.283072299040228240475092·10⁻⁹` and
/// `447.9396109993127243439717`; SciPy 1.18 `chi2.isf(1e-60, 60)`:
/// `447.9396109993127`.
#[test]
fn far_levels_of_the_first_of_five_chi_squared_are_not_probed_at_the_far_floats() {
    let ctx = Context::new();
    let parent = Distribution::chi_squared(ctx.int(60));
    let d = symplex::stats::order::order_statistic(&parent, 5, 1).unwrap();
    let timer = CpuTimer::start();
    let low = d.quantile_f64(1e-300).unwrap();
    let high = d.isf_f64(1e-300).unwrap();
    timer.assert_within(
        Duration::from_secs(20),
        "two far levels of the first of five chi-squared(60)",
    );
    assert!(rel(low, 2.283_072_299_040_228e-9) < 1e-15, "{low:e}");
    assert!(rel(high, 447.939_610_999_312_7) < 1e-15, "{high:e}");
}

/// The first of three independent `Gamma(25, 12√2/7)` variables (shape,
/// scale) at the level `10⁻³⁰⁰` of either tail (the fuzz input
/// `slow-unit-a1c46ccdfabd…`): 0.057 s and 0.187 s in 0.60.0, 0.43 s and
/// 1.24 s with the first draft of the 0.60.1 root search (the probe at the
/// largest double, as for the `χ²` law above), 0.007 s and 0.176 s now.
///
/// Oracle: mpmath at 400 digits, bisection of `P(25, x/θ) = 1 − (1 − p)^(1/3)`
/// and of `Q(25, x/θ) = q^(1/3)`, `θ = 12√2/7`, `p = q =` the double `10⁻³⁰⁰`
/// (`target/scratch/oracle/o_repro.py`): `2.361234830334826192792173·10⁻¹¹`
/// and `760.0396259498434724735782`; SciPy 1.18 `gamma.isf(1e-100, 25,
/// scale=θ)`: `760.0396259498435`.  (The search stops within `4ε` of the
/// root: the upper quantile is the double above the nearest one, as in the
/// first draft; 0.60.0 returned the nearest.)
#[test]
fn far_levels_of_the_first_of_three_gamma_are_not_probed_at_the_far_floats() {
    let ctx = Context::new();
    let scale = ctx.rational(12, 7) * &ctx.int(2).sqrt();
    let parent = Distribution::gamma(ctx.int(25), scale);
    let d = symplex::stats::order::order_statistic(&parent, 3, 1).unwrap();
    let timer = CpuTimer::start();
    let low = d.quantile_f64(1e-300).unwrap();
    let high = d.isf_f64(1e-300).unwrap();
    timer.assert_within(
        Duration::from_secs(20),
        "two far levels of the first of three gamma(25, 12*sqrt(2)/7)",
    );
    assert!(rel(low, 2.361_234_830_334_826e-11) < 1e-15, "{low:e}");
    assert!(rel(high, 760.039_625_949_843_5) < 1e-15, "{high:e}");
}

/// The median of a law that is symmetric about 0 and has no starting
/// estimate: the middle of three independent Student t variables with
/// `6.6·10²³⁶` degrees of freedom.  The search starts at 0, where the tail
/// is the level exactly, and 0.60.0 answered `0` there (0.1 ms in a release
/// build).  The first draft of the 0.60.1 root search went on over the hull
/// of the support and answered `2.119615247209161·10⁻¹⁷`, the last float of
/// its narrowing whose tail still rounds to `½` (50 ms and 182 ms).
///
/// Oracle: exact — the law is symmetric about 0, so both medians are 0
/// (SciPy 1.18: `t.ppf(0.5, 6.6e236)` is `0.0`).
#[test]
fn the_median_of_a_symmetric_law_without_an_estimate_is_its_centre() {
    let ctx = Context::new();
    let t = Distribution::student_t(ctx.from_f64(6.6e236).unwrap());
    let d = symplex::stats::order::order_statistic(&t, 3, 2).unwrap();
    let timer = CpuTimer::start();
    let lower = d.quantile_f64(0.5).unwrap();
    let upper = d.isf_f64(0.5).unwrap();
    timer.assert_within(Duration::from_secs(10), "two medians of a symmetric law");
    assert_eq!(lower, 0.0);
    assert_eq!(upper, 0.0);
}
