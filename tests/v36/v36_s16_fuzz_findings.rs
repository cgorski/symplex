//! Tests of session 16: what the nightly fuzz runs found on the 0.58
//! development tree (2026-10-17 … 2026-11-23), and issue #1.
//! Every test says what was wrong before and cites its oracle.

use std::time::Duration;

use symplex::prelude::*;
use symplex::stats::Distribution;

use crate::perf_assert::CpuTimer;

/// `|a/b − 1| ≤ 4ε`: an `f64` read from an exact rational against the
/// correctly rounded value of the oracle.
fn close(a: f64, b: f64) -> bool {
    (a / b - 1.0).abs() <= 4.0 * f64::EPSILON
}

// ── Hypergeometric: exact tails by the term recurrence ──────────────────────

/// `C(n, k)` for `n, k ≤ 8` by Pascal's rule.
fn pascal() -> [[i64; 9]; 9] {
    let mut c = [[0_i64; 9]; 9];
    for n in 0..9 {
        c[n][0] = 1;
        for k in 1..=n {
            c[n][k] = c[n - 1][k - 1] + c[n - 1][k];
        }
    }
    c
}

/// `cdf` and `sf` of a hypergeometric law are the sums of its masses: every
/// shape with `N ≤ 8`, every `k` from below the support to above it, and
/// the half-integers between (`P(X ≤ k + ½) = P(X ≤ k)`).  The tails are
/// formed by the recurrence of the integer terms since 0.58
/// (`hypergeometric_mass`); this pins them to the definition.
///
/// Oracle: `Σ_{j ≤ k} C(m, j) C(N − m, n − j) / C(N, n)` from Pascal's
/// triangle.
#[test]
fn hypergeometric_tails_are_the_sums_of_the_masses() {
    let ctx = Context::new();
    let c = pascal();
    let mut compared = 0;
    for big_n in 0..=8_usize {
        for m in 0..=big_n {
            for n in 0..=big_n {
                let int = |v: usize| ctx.int(i64::try_from(v).unwrap());
                let h = Distribution::hypergeometric(int(big_n), int(m), int(n));
                let (lo, hi) = ((n + m).saturating_sub(big_n), n.min(m));
                let den = c[big_n][n];
                let mut acc = 0;
                for k in lo..=hi {
                    acc += c[m][k] * c[big_n - m][n - k];
                    let below = ctx.rational(acc, den);
                    let above = ctx.one() - &below;
                    let at = int(k);
                    assert_eq!(h.cdf(&at), below, "cdf({k}) of H({big_n}, {m}, {n})");
                    assert_eq!(h.sf(&at), above, "sf({k}) of H({big_n}, {m}, {n})");
                    let half = &at + ctx.rational(1, 2);
                    assert_eq!(h.cdf(&half), below, "cdf({k} + 1/2)");
                    assert_eq!(h.sf(&half), above, "sf({k} + 1/2)");
                    compared += 1;
                }
                assert_eq!(acc, den, "Vandermonde for H({big_n}, {m}, {n})");
                let under = int(lo) - ctx.one();
                assert_eq!(h.cdf(&under), ctx.zero());
                assert_eq!(h.sf(&under), ctx.one());
                assert_eq!(h.cdf(&int(hi + 1)), ctx.one());
                assert_eq!(h.sf(&int(hi + 1)), ctx.zero());
            }
        }
    }
    assert_eq!(compared, 495, "tails compared");
}

/// The tails of `Hypergeometric(3000, 1200, 1500)` (1201 atoms; `C(3000,
/// 1500)` has 2994 bits, so the masses overflow as `f64` quotients).  Up to
/// 0.57 each tail was the symbolic sum of the masses, evaluated term by
/// term from three binomials each: `cdf(600)` took 35 ms in a release build
/// and — for a larger population, `Hypergeometric(30000, 9375, 28125)` —
/// `cdf(8789)` 3.7 s and `sf(9300)` 26 s.  They are exact rationals now
/// (2 ms; 9 ms and 14 ms).
///
/// Oracle: exact, Python `fractions.Fraction(sum(comb(1200, j) *
/// comb(1800, 1500 − j) for j ≤ k), comb(3000, 1500))`, rounded by `float`.
#[test]
fn hypergeometric_tails_of_a_large_population_are_exact() {
    let ctx = Context::new();
    let h = Distribution::hypergeometric(ctx.int(3000), ctx.int(1200), ctx.int(1500));
    let at = |k: i64| ctx.int(k);
    let timer = CpuTimer::start();
    let f600 = h.cdf(&at(600));
    assert!(f600.as_rational().is_some(), "an exact rational");
    assert!(close(f600.eval_f64().unwrap(), 0.514_863_778_064_920_8));
    assert!(close(
        h.sf(&at(600)).eval_f64().unwrap(),
        0.485_136_221_935_079_27
    ));
    assert!(close(
        h.cdf(&at(590)).eval_f64().unwrap(),
        0.239_448_816_050_177_73
    ));
    assert!(close(
        h.sf(&at(615)).eval_f64().unwrap(),
        0.123_980_884_961_780_3
    ));
    // The two tails are complementary, and a step of the CDF is the mass.
    assert_eq!(&f600 + h.sf(&at(600)), ctx.one());
    assert_eq!(&f600 - h.cdf(&at(599)), h.density(&at(600)).eval());
    assert!(close(
        h.density(&at(600)).eval_f64().unwrap(),
        0.029_727_556_129_841_484
    ));
    // 0.06 s of CPU in a debug build.
    timer.assert_within(
        Duration::from_secs(2),
        "six tails of Hypergeometric(3000, 1200, 1500)",
    );
}

/// The quantiles of two such laws, at ordinary levels and far in both
/// tails.  Up to 0.57 the search probed the symbolic sum some twenty times:
/// for `Hypergeometric(3000, 1200, 1500)` 0.06–0.46 s at ordinary levels in
/// a release build, and `isf_f64(1e-300)`, whose probes are `1 − Σ` next to
/// 1, **103 s** (for `Hypergeometric(30000, 9375, 28125)`: 7.4 s at the
/// median, 33 s at the quartiles, and the `fuzz_stats` input `XxRfdzAUPA==`
/// at `10⁻³⁰⁰` ran for minutes).  Each probe is an exact rational now:
/// 0.01–0.05 s (0.06–0.21 s for the population of 30000).
///
/// Oracle: exact, Python: the least `k` with `Fraction(Σ_{j ≤ k} T_j, C(N,
/// n)) ≥ Fraction(p)`, and the least `k` with `Fraction(Σ_{j > k} T_j, C(N,
/// n)) ≤ Fraction(q)`.
#[test]
fn hypergeometric_quantiles_of_a_large_population() {
    let ctx = Context::new();
    let timer = CpuTimer::start();
    let h = Distribution::hypergeometric(ctx.int(1200), ctx.int(500), ctx.int(600));
    for (level, quantile, upper) in [
        (0.5, 250.0, 250.0),
        (0.25, 244.0, 256.0),
        (1e-10, 196.0, 304.0),
    ] {
        assert_eq!(
            h.quantile_f64(level).unwrap(),
            quantile,
            "quantile at {level:e}"
        );
        assert_eq!(
            h.isf_f64(level).unwrap(),
            upper,
            "upper quantile at {level:e}"
        );
    }
    // Far in the tails of the larger law (its ends lie below `10⁻³⁰⁰`).
    let h = Distribution::hypergeometric(ctx.int(3000), ctx.int(1200), ctx.int(1500));
    assert_eq!(h.quantile_f64(1e-300).unwrap(), 128.0);
    assert_eq!(h.isf_f64(1e-300).unwrap(), 1072.0);
    // 0.7 s of CPU in a debug build.
    timer.assert_within(
        Duration::from_secs(7),
        "eight quantiles of two hypergeometric laws",
    );
}

/// The `fuzz_stats` case itself, smaller: a hypergeometric law shifted by
/// `x − 9` reads its tails through the inner law's exact ones.
///
/// Oracle: the quantiles above, minus 9; `F(250)` of `Hypergeometric(1200,
/// 500, 600)` by the same Python sum, `0.5233445169833297`.
#[test]
fn shifted_hypergeometric_quantiles() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let h = Distribution::hypergeometric(ctx.int(1200), ctx.int(500), ctx.int(600));
    let shifted = h.transformed(&x, &(&x - 9)).unwrap();
    let timer = CpuTimer::start();
    assert_eq!(shifted.quantile_f64(0.25).unwrap(), 235.0);
    assert_eq!(shifted.isf_f64(1e-10).unwrap(), 295.0);
    assert!(close(
        shifted.cdf(&ctx.int(241)).eval_f64().unwrap(),
        0.523_344_516_983_329_7
    ));
    // 0.2 s of CPU in a debug build.
    timer.assert_within(
        Duration::from_secs(3),
        "two quantiles and a tail of a shifted hypergeometric law",
    );
}

// ── Tables of values that are not rationals ─────────────────────────────────

/// `√X` for a die: a table of roots.  A point is ordered against the
/// values by their certified `f64`s, which the table now forms once (and
/// `transformed` hands over).  Up to 0.57 `cdf`, `sf` and the quantiles
/// evaluated every value at each call: for a 100,000-sided die 0.93 s per
/// tail and 4.0 s per quantile in a release build (0.02 s and 0.13 s now),
/// and the `fuzz_stats` input `/14ASAxI` timed out in 15 nightly runs.
///
/// Oracle: counting.  `√k ≤ 30` for `k ≤ 900` of 1600 faces; `√1000` is
/// an atom; `√k ≤ 30.1` for `k ≤ 906`; `F(√k) = k/1600 ≥ ½` from `k =
/// 800`; `S(√k) = 1 − k/1600 ≤ ¼` from `k = 1200`.
#[test]
fn table_of_roots_orders_points_by_values_formed_once() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = Distribution::die(ctx.int(1600))
        .transformed(&x, &x.sqrt())
        .unwrap();
    let timer = CpuTimer::start();
    assert_eq!(y.cdf(&ctx.int(30)), ctx.rational(9, 16));
    assert_eq!(y.sf(&ctx.int(30)), ctx.rational(7, 16));
    assert_eq!(y.cdf(&ctx.int(1000).sqrt()), ctx.rational(5, 8));
    assert_eq!(y.sf(&ctx.int(1000).sqrt()), ctx.rational(3, 8));
    assert_eq!(y.cdf(&ctx.rational(301, 10)), ctx.rational(453, 800));
    assert_eq!(y.cdf(&ctx.rational(1, 2)), ctx.zero());
    assert_eq!(y.cdf(&ctx.int(41)), ctx.one());
    assert!(close(y.quantile_f64(0.5).unwrap(), 800_f64.sqrt()));
    assert!(close(y.isf_f64(0.25).unwrap(), 1200_f64.sqrt()));
    assert_eq!(y.quantile_f64(1e-300).unwrap(), 1.0);
    assert_eq!(y.density(&ctx.int(1000).sqrt()), ctx.rational(1, 1600));
    assert_eq!(y.density(&ctx.rational(301, 10)), ctx.zero());
    // 0.3 s of CPU in a debug build.
    timer.assert_within(
        Duration::from_secs(3),
        "tails and quantiles of the roots of a 1600-sided die",
    );
}

/// A table built directly (no values handed over) orders the same way.
///
/// Oracle: counting, as above, for the table `{√k: 1/6}`, `k = 1..=6`.
#[test]
fn table_of_roots_built_directly() {
    let ctx = Context::new();
    let table = (1..=6)
        .map(|k| (ctx.int(k).sqrt(), ctx.rational(1, 6)))
        .collect();
    let d = Distribution::finite(&ctx, table);
    assert_eq!(d.cdf(&ctx.int(2)), ctx.rational(2, 3));
    assert_eq!(d.cdf(&ctx.int(5).sqrt()), ctx.rational(5, 6));
    assert_eq!(d.sf(&ctx.rational(17, 10)), ctx.rational(2, 3));
    assert!(close(d.quantile_f64(0.5).unwrap(), 3_f64.sqrt()));
    assert!(close(d.isf_f64(0.5).unwrap(), 3_f64.sqrt()));
    assert!(close(d.quantile_f64(0.9).unwrap(), 6_f64.sqrt()));
}

// ── Far tails are read from the non-cancelling forms ────────────────────────

/// Whether a point lies in a far tail is asked of the family's
/// non-cancelling form of that tail (`½ erfc(−z/√2)` for a normal), not of
/// the classic `½ + ½ erf(z/√2)`, whose certified `f64` next to 0 costs a
/// working precision of as many digits as the tail is small.  Up to 0.57
/// `F(−91)` of a standard normal took 64 ms in a release build (2 ms at
/// `−37`); the values are the same.
///
/// Oracle: mpmath 1.4 at 40 and 80 digits, `erfc(37/sqrt(2))/2` =
/// `5.7255712225245768227e-300`, `erfc(5/sqrt(2))/2` =
/// `2.8665157187919391167e-7`; `erfc(91/√2)/2 ≈ 10⁻¹⁷⁹⁹` is below the
/// doubles, where `eval_f64` reads `0`.
#[test]
fn normal_cdf_far_in_the_lower_tail() {
    let ctx = Context::new();
    let n = Distribution::normal(ctx.int(0), ctx.int(1));
    let timer = CpuTimer::start();
    for _ in 0..20 {
        assert_eq!(n.cdf(&ctx.int(-91)).eval_f64().unwrap(), 0.0);
        assert_eq!(n.sf(&ctx.int(91)).eval_f64().unwrap(), 0.0);
    }
    assert!(close(
        n.cdf(&ctx.int(-37)).eval_f64().unwrap(),
        5.725_571_222_524_577e-300
    ));
    assert!(close(
        n.sf(&ctx.int(37)).eval_f64().unwrap(),
        5.725_571_222_524_577e-300
    ));
    assert!(close(
        n.cdf(&ctx.int(-5)).eval_f64().unwrap(),
        2.866_515_718_791_939e-7
    ));
    // 0.05 s of CPU in a debug build (1.3 s in a release build up to 0.57).
    timer.assert_within(
        Duration::from_secs(2),
        "forty far tails of a standard normal",
    );
}

/// The quantiles of a mixture of two normals far in a tail, where the
/// search probes both components at every step.  Up to 0.57 each probe
/// certified the classic form of each component: `½ N(−73, ½) + ½ N(0, 1)`
/// took 0.24 s at the level `10⁻⁴`, 1.8 s at `10⁻³⁰` and 4.1 s at `10⁻³⁰⁰`
/// in a release build (0.01–0.04 s now), and the `fuzz_stats` inputs
/// `AAAAAED1DDo=` and `CgAbKEArACIAERsr` (12 s and 78 s for the whole
/// check) timed out in 12 nightly runs.
///
/// Oracle: mpmath 1.4 at 40 and 80 digits, `findroot` of `log((erfc(−(x −
/// μ)/(σ√2)) + erfc(−x/√2))/4) − log(p)` for the double `p`:
/// `−78.701928118626693124`, `−91.514197648162730301`; and for `½ N(0,
/// 1/6) + ½ N(0, 1)`, symmetric about 0, `−11.403856237253386249` at
/// `10⁻³⁰`.
#[test]
fn mixture_of_normals_quantiles_far_in_a_tail() {
    let ctx = Context::new();
    let half = ctx.rational(1, 2);
    let std = Distribution::normal(ctx.int(0), ctx.int(1));
    let mix = |mu: i64, sigma: Ex| {
        let other = Distribution::normal(ctx.int(mu), sigma);
        Distribution::mixture(&[(half.clone(), other), (half.clone(), std.clone())]).unwrap()
    };
    let near = |got: f64, want: f64| (got / want - 1.0).abs() < 1e-14;
    let timer = CpuTimer::start();
    let apart = mix(-73, ctx.rational(1, 2));
    for (level, want) in [
        (1e-30, -78.701_928_118_626_7),
        (1e-300, -91.514_197_648_162_73),
    ] {
        let got = apart.quantile_f64(level).unwrap();
        assert!(near(got, want), "quantile at {level:e}: {got}");
    }
    let nested = mix(0, ctx.rational(1, 6));
    let high = nested.isf_f64(1e-30).unwrap();
    assert!(near(high, 11.403_856_237_253_386), "{high}");
    // 0.8 s of CPU in a debug build.
    timer.assert_within(
        Duration::from_secs(8),
        "three quantiles of mixtures of two normals",
    );
}

// ── A uniform law off the origin: the cf is not real ────────────────────────

/// The nightly run of 2026-11-11 reported `Im cf(t) = 0.784…` "for an even
/// density" of `Uniform(−116, −983/9)`: the harness tested the density
/// formula alone, a constant, and not where the support lies.  The value
/// is right: `φ(t) = (e^{itb} − e^{ita}) / (it(b − a))`.
///
/// Oracle: mpmath 1.4 at 30 and 60 digits, `(e**(1j*t*b) -
/// e**(1j*t*a))/(1j*t*(b - a))` = `0.60801044977263860994 +
/// 0.78408904017557322968j`.
#[test]
fn characteristic_function_of_a_uniform_law_off_the_origin() {
    let ctx = Context::new();
    let d = Distribution::uniform(ctx.int(-116), ctx.rational(-983, 9));
    let t = ctx
        .from_f64(-2_301_776_076_851_989.0 / 36_028_797_018_963_968.0)
        .unwrap();
    let z = d.characteristic_function(&t).eval_complex64().unwrap();
    assert!((z.re - 0.608_010_449_772_638_6).abs() < 1e-14, "{z}");
    assert!((z.im - 0.784_089_040_175_573_2).abs() < 1e-14, "{z}");
}

// ── sinc at an argument proved zero (issue #1) ──────────────────────────────

/// `sinc(0) = 1`: up to 0.57 a zero argument built `sin(0)/0 = nan` (issue
/// #1; the literal zero is pinned in `unit::test_recip_trig`).  An argument
/// proved zero counts too; a symbolic one keeps `sin(x)/x`, whose value at
/// 0 is its limit.
///
/// Oracle: `lim_{x → 0} sin(x)/x = 1`; SymPy 1.14 `sinc(0)` = `1`;
/// `Γ(1/3)Γ(2/3) = 2π/√3` (the reflection formula).
#[test]
fn sinc_of_an_argument_proved_zero_is_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let hidden = ctx.parse("gamma(1/3)*gamma(2/3) - 2*pi/sqrt(3)").unwrap();
    assert_eq!(hidden.sinc(), ctx.one());
    assert_eq!(ctx.zero().sinc(), ctx.one());
    assert_eq!(x.sinc(), &x.sin() / &x);
    assert_eq!(x.sinc().limit(&x, &ctx.zero()), ctx.one());
    assert_eq!(ctx.pi().sinc(), ctx.zero());
}
