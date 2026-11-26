//! Tests of session 18, second block of 0.60.1. Lane OS: order statistics, exact tables with
//! huge rationals, sums of gamma laws.
//! Every test says what was wrong before and cites its oracle.
//!
//! "Before" is the base of the block (`e9c50e9`, the release candidate of 0.60.1), release build;
//! 0.60.0 behaves the same.  The oracle calls are in `bench/test_oracle.py` of the lane's scratch
//! (mpmath 1.3.0 at 40 and 80 digits, SciPy 1.18.1, exact rationals by Python's `fractions`);
//! their output is quoted here.

use std::time::Duration;

use crate::perf_assert::CpuTimer;
use symplex::prelude::*;
use symplex::stats::order::{maximum_of, minimum_of, order_statistic};
use symplex::stats::{Distribution, RandomVariable, sum_distribution};

fn parse(ctx: &Context, text: &str) -> Ex {
    ctx.parse(text).unwrap()
}

fn close(got: f64, want: f64, tol: f64) -> bool {
    (got - want).abs() <= tol * want.abs().max(1.0)
}

fn sum_of(ctx: &Context, a: Distribution, b: Distribution) -> Option<Distribution> {
    let x = RandomVariable::new(ctx, "X", a);
    let y = RandomVariable::new(ctx, "Y", b);
    sum_distribution(&x, &y)
}

// ── Class C: the closures of `sum_distribution` ───────────────────────────────────────────────

/// Before: `None` for each of the three sums (the shared parameter was compared structurally,
/// and `2·ln 2` is not the expression `ln 4`).
///
/// Oracle: exact (the moment generating functions multiply); the means by mpmath,
/// `5*log(4)` = `6.931471805599453094172321`, `9*log(3)` = `9.887510598012987222557207`,
/// `5*log(4)/7` = `0.9902102579427790134531887`.
#[test]
fn a_shared_parameter_written_in_two_equal_forms_closes_the_sum() {
    let ctx = Context::new();
    // Gamma(2, ln 4) + Gamma(3, 2·ln 2) = Gamma(5, ln 4): the first form is kept.
    let (t1, t2) = (parse(&ctx, "ln(4)"), parse(&ctx, "2*ln(2)"));
    let s = sum_of(
        &ctx,
        Distribution::gamma(ctx.int(2), t1.clone()),
        Distribution::gamma(ctx.int(3), t2),
    )
    .expect("Gamma(2, ln 4) + Gamma(3, 2 ln 2)");
    assert_eq!(s, Distribution::gamma(ctx.int(5), t1));
    let mean = s.mean().eval_f64().unwrap();
    assert!(close(mean, 6.931_471_805_599_453, 1e-15), "{mean}");

    // Gamma(2, 3·ln 3) + Exponential(1/ln 27) = Gamma(3, 3·ln 3).
    let theta = parse(&ctx, "3*ln(3)");
    let s = sum_of(
        &ctx,
        Distribution::gamma(ctx.int(2), theta.clone()),
        Distribution::exponential(parse(&ctx, "1/ln(27)")),
    )
    .expect("Gamma(2, 3 ln 3) + Exponential(1/ln 27)");
    assert_eq!(s, Distribution::gamma(ctx.int(3), theta));
    let mean = s.mean().eval_f64().unwrap();
    assert!(close(mean, 9.887_510_598_012_987, 1e-15), "{mean}");

    // Binomial(3, ln(4)/7) + Binomial(2, 2·ln(2)/7) = Binomial(5, ln(4)/7).
    let (p1, p2) = (parse(&ctx, "ln(4)/7"), parse(&ctx, "2*ln(2)/7"));
    let s = sum_of(
        &ctx,
        Distribution::binomial(ctx.int(3), p1.clone()),
        Distribution::binomial(ctx.int(2), p2),
    )
    .expect("Binomial(3, ln(4)/7) + Binomial(2, 2 ln(2)/7)");
    assert_eq!(s, Distribution::binomial(ctx.int(5), p1));
    let mean = s.mean().eval_f64().unwrap();
    assert!(close(mean, 0.990_210_257_942_779, 1e-15), "{mean}");

    // NegativeBinomial(2, ln(27)/5) + NegativeBinomial(3, 3·ln(3)/5).
    let (p1, p2) = (parse(&ctx, "ln(27)/5"), parse(&ctx, "3*ln(3)/5"));
    let s = sum_of(
        &ctx,
        Distribution::negative_binomial(ctx.int(2), p1.clone()),
        Distribution::negative_binomial(ctx.int(3), p2),
    )
    .expect("NegativeBinomial(2, ln(27)/5) + NegativeBinomial(3, 3 ln(3)/5)");
    assert_eq!(s, Distribution::negative_binomial(ctx.int(5), p1));
}

/// The other side of the same decision (unchanged, and it must stay): parameters that are not
/// proved equal give `None` — different numbers (`π` and `355/113`; `ln 4` and its 16-digit
/// decimal `1.386294361119891`, which differ by `4·10⁻¹⁶`: `log(4)` = `1.386294361119890618…`,
/// mpmath), and symbols of which nothing is known.
#[test]
fn parameters_that_are_not_proved_equal_do_not_close_the_sum() {
    let ctx = Context::new();
    let two = || ctx.int(2);
    assert!(
        sum_of(
            &ctx,
            Distribution::gamma(two(), ctx.pi()),
            Distribution::gamma(two(), ctx.rational(355, 113)),
        )
        .is_none()
    );
    assert!(
        sum_of(
            &ctx,
            Distribution::gamma(two(), parse(&ctx, "ln(4)")),
            Distribution::gamma(two(), parse(&ctx, "1386294361119891/1000000000000000")),
        )
        .is_none()
    );
    assert!(
        sum_of(
            &ctx,
            Distribution::binomial(ctx.int(3), ctx.rational(1, 3)),
            Distribution::binomial(two(), ctx.rational(1, 2)),
        )
        .is_none()
    );
    assert!(
        sum_of(
            &ctx,
            Distribution::gamma(two(), ctx.symbol("a")),
            Distribution::gamma(two(), ctx.symbol("b")),
        )
        .is_none()
    );
}

// ── Class J: order statistics of a finite lattice, and a table under a monotone map ───────────

/// Before: none of the four calls returned in 70 s (release build): the masses were built by
/// expanding `Σ C(n, j) Fʲ (1 − F)ⁿ⁻ʲ` in the symbolic partial sums of the parent's masses.
/// Now each takes 0.05 s (release) and is the family with the lattice formula, the exact masses
/// being rationals of 79,000 bits and more each.
///
/// Oracle for the value: `P(min > 842) = P(X = 843)² = p¹⁶⁸⁶`; mpmath, `p**1686` with
/// `p = mpf(99999999999991)/10**14` = `0.9999999998482600000115057` (40 and 80 digits).
#[test]
fn the_order_statistic_of_a_binomial_law_with_a_long_fraction_is_built_at_once() {
    let ctx = Context::new();
    let p = parse(&ctx, "99999999999991/100000000000000");
    let parent = Distribution::binomial(ctx.int(843), p);
    for (n, k) in [(2, 1), (2, 2), (3, 2), (5, 3)] {
        let timer = CpuTimer::start();
        let d = order_statistic(&parent, n, k).unwrap();
        timer.assert_within(
            Duration::from_secs(30),
            format!("order_statistic(Binomial(843, 99999999999991/10^14), {n}, {k})"),
        );
        assert_eq!(d.support(), parent.support());
    }
    let min = order_statistic(&parent, 2, 1).unwrap();
    let sf = min.sf(&ctx.int(842)).eval_f64().unwrap();
    assert!(close(sf, 0.999_999_999_848_26, 1e-15), "{sf}");
}

/// The exact tables are the definition's: `P(X_(k) = v) = G(F(v)) − G(F(v⁻))` with
/// `G(u) = Σ_{j≥k} C(n, j) uʲ (1 − u)ⁿ⁻ʲ`, now computed in integers over the common denominator
/// of the tails (before: by a symbolic expansion; the values are the same, `Binomial(843, 1/3)`,
/// 3rd of 6: 1.28 s → 0.52 s in a release build).
///
/// Oracle: the same sums in Python's exact `fractions` (`order_masses` of `test_oracle.py`):
/// `Binomial(4, 1/3)`, 2nd of 3: `54016/531441, 284672/531441, 6464/19683, 17984/531441,
/// 241/531441`; `Hypergeometric(50, 20, 10)`, 3rd of 5: mass at 4 `0.4982020261525355`, mean
/// `3.9846343237560493`; `Binomial(60, 1/3)`, 2nd of 4: mean `18.885148123039528`, variance
/// `4.7887593271087`, `P(X_(2) ≤ 19)` = `0.6116110893835093`.
#[test]
fn the_exact_tables_of_order_statistics_are_those_of_the_definition() {
    let ctx = Context::new();
    let d = order_statistic(
        &Distribution::binomial(ctx.int(4), ctx.rational(1, 3)),
        3,
        2,
    )
    .unwrap();
    let want = [
        (54016, 531_441),
        (284_672, 531_441),
        (6464, 19683),
        (17984, 531_441),
        (241, 531_441),
    ];
    for (v, (num, den)) in (0_i64..).zip(want) {
        assert_eq!(
            d.density(&ctx.int(v)).eval(),
            ctx.rational(num, den),
            "at {v}"
        );
    }

    let h = Distribution::hypergeometric(ctx.int(50), ctx.int(20), ctx.int(10));
    let d = order_statistic(&h, 5, 3).unwrap();
    let mass = d.density(&ctx.int(4)).eval_f64().unwrap();
    assert!(close(mass, 0.498_202_026_152_535_5, 1e-14), "{mass}");
    let mean = d.mean().eval_f64().unwrap();
    assert!(close(mean, 3.984_634_323_756_049_3, 1e-14), "{mean}");

    let b = Distribution::binomial(ctx.int(60), ctx.rational(1, 3));
    let d = order_statistic(&b, 4, 2).unwrap();
    let mean = d.mean().eval_f64().unwrap();
    assert!(close(mean, 18.885_148_123_039_528, 1e-14), "{mean}");
    let variance = d.variance().eval_f64().unwrap();
    assert!(close(variance, 4.788_759_327_108_7, 1e-13), "{variance}");
    let cdf = d.cdf(&ctx.int(19)).eval_f64().unwrap();
    assert!(close(cdf, 0.611_611_089_383_509_3, 1e-14), "{cdf}");
}

/// Before: 12.5 s (release build) for the same table: the 156 images `tanh 1548 … tanh 1703` are
/// all the double `1.0`, their residuals are below the doubles, and every pair went to `equals`.
/// Now 0.014 s: `tanh` is proved strictly increasing on `[1548, 1703]`, so the images of
/// distinct values are distinct.
///
/// Oracle: the values `1548..=1703` are the support of the parent (`max(0, 1703 + 4800 − 4955)`
/// to `min(1703, 4800)`), 156 of them; every image is within `2·e⁻³⁰⁹⁶` of 1 (mpmath,
/// `tanh(1548) - 1` = `0.0` at 80 digits), so the mean is the double `1.0`.
#[test]
fn a_lattice_under_tanh_is_tabulated_without_comparing_its_images_pairwise() {
    let ctx = Context::new();
    let h = Distribution::hypergeometric(ctx.int(4955), ctx.int(4800), ctx.int(1703));
    let x = ctx.symbol("x");
    let timer = CpuTimer::start();
    let t = h.transformed(&x, &parse(&ctx, "tanh(x)")).unwrap();
    timer.assert_within(
        Duration::from_secs(30),
        "Hypergeometric(4955, 4800, 1703).transformed(x, tanh(x))",
    );
    assert_eq!(t.support().as_points().map(|p| p.len()), Some(156));
    let mean = t.mean().eval_f64().unwrap();
    assert!(close(mean, 1.0, 1e-15), "{mean}");
}

// ── Class I: the first of one draw, the minimum and the maximum ───────────────────────────────

/// Before: `order_statistic(Poisson(4), 1, 1)` was a family of its own whose `variance()` took
/// 0.67 s to form a `Sum` that `eval_f64` refused ("cannot evaluate: infinite sum of …"), and
/// whose `mean()` was refused the same way.  The first of one draw is the draw: the parent is
/// returned.
///
/// Oracle: exact (mean and variance of `Poisson(λ)` are `λ`).
#[test]
fn the_first_of_one_draw_is_the_parent() {
    let ctx = Context::new();
    let poisson = Distribution::poisson(ctx.int(4));
    let d = order_statistic(&poisson, 1, 1).unwrap();
    assert_eq!(d, poisson);
    assert_eq!(d.mean(), ctx.int(4));
    assert_eq!(d.variance(), ctx.int(4));
    for parent in [
        Distribution::die(ctx.int(6)),
        Distribution::normal(ctx.int(0), ctx.int(1)),
        Distribution::binomial(ctx.int(843), ctx.rational(1, 3)),
    ] {
        assert_eq!(order_statistic(&parent, 1, 1).unwrap(), parent);
        assert_eq!(minimum_of(&parent, 1).unwrap(), parent);
        assert_eq!(maximum_of(&parent, 1).unwrap(), parent);
    }
    // The ranks are validated as before.
    assert!(order_statistic(&poisson, 1, 2).is_err());
    assert!(order_statistic(&poisson, 0, 0).is_err());
    assert!(order_statistic(&poisson, 1, 0).is_err());
}

/// The maximum is below `x` when every draw is, the minimum above `x` when every draw is:
/// `F(x)ⁿ`, `S(x)ⁿ` in the parent's own tails, and on a lattice the minimum's masses are
/// `S(x−1)ⁿ − S(x)ⁿ` (before: the incomplete beta function of the tails, for the minimum's
/// masses `I_F(1, n)` expanded in powers of `F`; the values at these points were the same, and
/// the refusal of the mean of the minimum of two `NegativeBinomial(36, 1/2)` took 7.1 s, now
/// 0.007 s).  The maximum's masses `F(x)ⁿ − F(x−1)ⁿ` are unchanged.
///
/// Oracle: mpmath at 40 and 80 digits with `F(k) = nsum(exp(-4)*4**j/factorial(j), [0, k])`:
/// `(1-F(2))**2 - (1-F(3))**2` = `0.25953046849112454930473`, `(1-F(3))**2` =
/// `0.3209561045173112638194162`, `F(5)**2` = `0.6164297246385138544393978`,
/// `F(5)**2 - F(4)**2` = `0.2209938335920973990998293`, `F(5)**3 - F(4)**3` =
/// `0.2353130144967315702633362`; `ncdf(1)**3` = `0.595555117931464425193918`; `exp(-3)` =
/// `0.04978706836786394297934242`.  SciPy: `poisson.sf(2,4)**2 - poisson.sf(3,4)**2` =
/// `0.25953046849112443`, `poisson.cdf(5,4)**2 - poisson.cdf(4,4)**2` = `0.22099383359209746`,
/// `norm.cdf(1)**3` = `0.5955551179314644`.  Exact: `(1/2)³ = 1/8`; the dice by `fractions`.
#[test]
fn the_minimum_and_the_maximum_use_the_tails_of_the_parent() {
    let ctx = Context::new();
    let poisson = Distribution::poisson(ctx.int(4));
    let min = minimum_of(&poisson, 2).unwrap();
    let v = min.density(&ctx.int(3)).eval_f64().unwrap();
    assert!(close(v, 0.259_530_468_491_124_55, 1e-14), "{v}");
    let v = min.sf(&ctx.int(3)).eval_f64().unwrap();
    assert!(close(v, 0.320_956_104_517_311_26, 1e-14), "{v}");
    let max = maximum_of(&poisson, 2).unwrap();
    let v = max.cdf(&ctx.int(5)).eval_f64().unwrap();
    assert!(close(v, 0.616_429_724_638_513_9, 1e-14), "{v}");
    let v = max.density(&ctx.int(5)).eval_f64().unwrap();
    assert!(close(v, 0.220_993_833_592_097_4, 1e-14), "{v}");
    let v = maximum_of(&poisson, 3)
        .unwrap()
        .density(&ctx.int(5))
        .eval_f64()
        .unwrap();
    assert!(close(v, 0.235_313_014_496_731_57, 1e-14), "{v}");

    let normal = Distribution::normal(ctx.int(0), ctx.int(1));
    let v = maximum_of(&normal, 3)
        .unwrap()
        .cdf(&ctx.int(1))
        .eval_f64()
        .unwrap();
    assert!(close(v, 0.595_555_117_931_464_4, 1e-14), "{v}");
    let exponential = Distribution::exponential(ctx.int(2));
    let v = minimum_of(&exponential, 3)
        .unwrap()
        .sf(&ctx.rational(1, 2))
        .eval_f64()
        .unwrap();
    assert!(close(v, 0.049_787_068_367_863_944, 1e-14), "{v}");

    let uniform = Distribution::uniform(ctx.int(0), ctx.int(1));
    assert_eq!(
        maximum_of(&uniform, 3).unwrap().cdf(&ctx.rational(1, 2)),
        ctx.rational(1, 8)
    );
    let die = Distribution::die(ctx.int(6));
    let min = minimum_of(&die, 2).unwrap();
    for (v, num) in (1_i64..).zip([11, 9, 7, 5, 3, 1]) {
        assert_eq!(
            min.density(&ctx.int(v)).eval(),
            ctx.rational(num, 36),
            "min at {v}"
        );
    }
    let max = maximum_of(&die, 3).unwrap();
    for (v, num) in (1_i64..).zip([1, 7, 19, 37, 61, 91]) {
        assert_eq!(
            max.density(&ctx.int(v)).eval(),
            ctx.rational(num, 216),
            "max at {v}"
        );
    }
}
