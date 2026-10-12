//! The statistics and plotting re-hunt (0.46): densities at the ends of the support,
//! long negative-binomial CDF sums, the Pareto characteristic function,
//! mixtures with overlapping supports, odd-power transforms, divergent
//! expectations of non-polynomials, joint `E[e^L]`, `E[cos L]`,
//! `E[sin L]` (`symplex::stats`).

use symplex::prelude::*;
use symplex::stats::{self, Distribution, RandomVariable};

fn close(actual: f64, want: f64, rel: f64, label: &str) {
    assert!(
        (actual - want).abs() <= rel * want.abs(),
        "{label}: got {actual}, want {want}"
    );
}

/// At a finite end of the support the density is the one-sided limit
/// from inside, as in scipy.  Before, the formula was evaluated there:
/// `0/0 = NaN` for the log-normal and every F density at 0, and the
/// complex infinity `zoo` for `x^(k−1)` with `k < 1` (Gamma, Beta,
/// Weibull, χ²(1)).  Oracle: scipy 1.x `stats.lognorm(1).pdf(0) = 0.0`,
/// `stats.f(1, 5).pdf(0) = inf`, `stats.f(2, 5).pdf(0) = 1.0`,
/// `stats.f(7, 5).pdf(0) = 0.0`, `stats.gamma(0.5).pdf(0) = inf`,
/// `stats.beta(0.5, 3).pdf(1) = 0.0`, `stats.weibull_min(0.5).pdf(0) = inf`,
/// `stats.chi2(1).pdf(0) = inf`.
#[test]
fn density_at_a_support_end_is_the_one_sided_limit() {
    let ctx = Context::new();
    let r = |a, b| ctx.rational(a, b);
    let (zero, one, inf) = (ctx.int(0), ctx.int(1), ctx.infinity());
    assert_eq!(
        Distribution::log_normal(ctx.int(0), ctx.int(1)).density(&zero),
        zero
    );
    assert_eq!(
        Distribution::log_normal(r(-3, 2), r(1, 4)).density(&zero),
        zero
    );
    assert_eq!(
        Distribution::f_distribution(ctx.int(1), ctx.int(5)).density(&zero),
        inf
    );
    // `2/(5·B(1, 5/2))`: `eval` leaves the beta function (B(1, b) = 1/b).
    let f2 = Distribution::f_distribution(ctx.int(2), ctx.int(5)).density(&zero);
    close(f2.eval_f64().unwrap(), 1.0, 1e-15, "F(2, 5) at 0");
    assert_eq!(
        Distribution::f_distribution(ctx.int(7), ctx.int(5)).density(&zero),
        zero
    );
    assert_eq!(Distribution::gamma(r(1, 2), ctx.int(1)).density(&zero), inf);
    assert_eq!(Distribution::chi_squared(ctx.int(1)).density(&zero), inf);
    assert_eq!(
        Distribution::weibull(ctx.int(1), r(1, 2)).density(&zero),
        inf
    );
    assert_eq!(Distribution::beta(r(1, 2), ctx.int(3)).density(&zero), inf);
    assert_eq!(Distribution::beta(r(1, 2), ctx.int(3)).density(&one), zero);
    // Where the formula is defined it is unchanged, ends included.
    assert_eq!(
        Distribution::chi_squared(ctx.int(2)).density(&zero).eval(),
        r(1, 2)
    );
    assert_eq!(
        Distribution::exponential(ctx.int(3)).density(&zero),
        ctx.int(3)
    );
    assert_eq!(
        Distribution::uniform(ctx.int(0), ctx.int(4)).density(&zero),
        r(1, 4)
    );
    let x = ctx.symbol("x");
    assert!(
        Distribution::log_normal(ctx.int(0), ctx.int(1))
            .density(&x)
            .contains(&x)
    );
}

/// A negative binomial (no closed CDF, a closed survival function) with
/// irrational masses: `P(X ≤ k)` beyond 64 atoms is `1 − I_{1−p}(k+1, r)`
/// rather than the exact sum of `k + 1` irrational terms.  Before,
/// `NegativeBinomial(1/593, 793/870).cdf(1000)` did not finish in 10 s
/// (0.5 s at `k = 200`, quadratic), and the quantile of
/// `NegativeBinomial(15/2, 10⁻⁶)` at `½` — past the 10⁵-atom walk — was
/// refused ("Sum range too large for numerical evaluation").  Oracles:
/// scipy `stats.nbinom.ppf(0.5, 7.5, 1e-6) = 7169422.0`, mpmath
/// `betainc(7.5, 7169423, 0, 1e-6, regularized=True) =
/// 0.500000011852206901712524393837` (and `0.4999998622721716588` at
/// 7169421), `betainc(7.5, 7000001, 0, 1e-6, regularized=True) =
/// 0.474472045797287162681077546406`.
#[test]
fn negative_binomial_long_cdf_sums_and_quantiles() {
    let ctx = Context::new();
    let r = |a, b| ctx.rational(a, b);
    let slow = Distribution::negative_binomial(r(1, 593), r(793, 870));
    let t0 = std::time::Instant::now();
    assert_eq!(slow.cdf(&ctx.int(1000)).eval_f64().unwrap(), 1.0);
    assert!(
        t0.elapsed().as_secs() < 5,
        "cdf(1000) took {:?}",
        t0.elapsed()
    );
    let wide = Distribution::negative_binomial(r(15, 2), r(1, 1_000_000));
    assert_eq!(wide.quantile_f64(0.5).unwrap(), 7_169_422.0);
    close(
        wide.cdf(&ctx.int(7_000_000)).eval_f64().unwrap(),
        0.474_472_045_797_287_16,
        1e-14,
        "cdf(7e6)",
    );
    // Short sums and rational masses stay exact.
    let small = Distribution::negative_binomial(ctx.int(3), r(1, 2));
    assert_eq!(small.cdf(&ctx.int(2)), r(1, 2));
    assert!(small.cdf(&ctx.int(100)).is_rational() == Some(true));
}

/// The Pareto characteristic function in closed form,
/// `α (−i xₘ t)^α Γ(−α, −i xₘ t)` (the upper incomplete gamma at an
/// imaginary argument evaluates since 0.45).  Before, the generic
/// `E[cos tX] + i E[sin tX]` left oscillatory heavy-tail integrals that
/// quadrature refused for small `α` (the hunter: 24 of 75 Pareto cases
/// refused).  Oracle: mpmath `quadosc` of `cos(tx)·f(x)` and
/// `sin(tx)·f(x)` over `[xₘ, ∞)`: Pareto(2, ½) at `t = −½`:
/// `−0.092475228000598347 − 0.28573664632285262i`; Pareto(1, 3) at
/// `t = 1/10`: `0.98577289955213754 + 0.14822052363416666i`;
/// Pareto(1/1000, 5/2) at `t = 100`: `0.98027749294431443 +
/// 0.16221548028088051i`.
#[test]
fn pareto_characteristic_function_in_closed_form() {
    let ctx = Context::new();
    let r = |a, b| ctx.rational(a, b);
    for (xm, a, t, re, im) in [
        (
            r(2, 1),
            r(1, 2),
            r(-1, 2),
            -0.092_475_228_000_598_35,
            -0.285_736_646_322_852_6,
        ),
        (
            r(1, 1),
            r(3, 1),
            r(1, 10),
            0.985_772_899_552_137_5,
            0.148_220_523_634_166_66,
        ),
        (
            r(1, 1000),
            r(5, 2),
            r(100, 1),
            0.980_277_492_944_314_4,
            0.162_215_480_280_880_5,
        ),
    ] {
        let cf = Distribution::pareto(xm, a).characteristic_function(&t);
        assert!(!cf.has_unevaluated(), "{cf}");
        let z = cf.eval_complex64().unwrap();
        close(z.re, re, 1e-12, "re");
        close(z.im, im, 1e-12, "im");
    }
    let p = Distribution::pareto(r(2, 1), r(1, 2));
    assert_eq!(p.characteristic_function(&ctx.int(0)), ctx.int(1));
}

/// A mixture whose components' supports overlap integrates its density
/// over the merged support `[0, 3]`.  Before, the support was the list
/// `[0, 2] ∪ [1, 3]` and the overlap was integrated twice:
/// `E[cos X]` of ½ U(0, 2) + ½ U(1, 3) was 0.08615 (want 0.0522366),
/// the entropy `ln 4 = 1.386` (want 1.0397), `E[X | X > ½]` was `71/28`
/// (want `47/28`), and the variance of ⅓ Gamma(2, 2) + ⅔ Triangular(2, 6, 2)
/// was 14.03 (want `272/81`).  Oracle: exact integration by hand —
/// `E[cos X] = (sin 2 + sin 3 − sin 1)/4 = 0.052236612519413084`, entropy
/// `−(2·¼ ln ¼ + ½ ln ½) = 1.0397207708399179`, `E[X | X > ½] =
/// (½·¾·(5/4) + ½·2)/(½·¾ + ½) = 47/28`; the gamma/triangular moments
/// `E[X²] = ⅓·24 + ⅔·12 = 16`, mean `32/9`.
#[test]
fn mixture_with_overlapping_supports() {
    let ctx = Context::new();
    let r = |a, b| ctx.rational(a, b);
    let m = Distribution::mixture(&[
        (r(1, 2), Distribution::uniform(r(0, 1), r(2, 1))),
        (r(1, 2), Distribution::uniform(r(1, 1), r(3, 1))),
    ])
    .unwrap();
    assert_eq!(m.support().to_string(), "[0, 3]");
    let x = RandomVariable::new(&ctx, "X", m.clone());
    let xs = x.symbol().clone();
    close(
        x.expectation(&xs.cos()).eval_f64().unwrap(),
        0.052_236_612_519_413_084,
        1e-13,
        "E[cos X]",
    );
    close(
        m.entropy().eval_f64().unwrap(),
        1.039_720_770_839_917_9,
        1e-13,
        "entropy",
    );
    let given = x.given(&xs.gt(&r(1, 2))).unwrap();
    assert_eq!(given.mean(), r(47, 28));
    let g = Distribution::mixture(&[
        (r(1, 3), Distribution::gamma(r(2, 1), r(2, 1))),
        (r(2, 3), Distribution::triangular(r(2, 1), r(6, 1), r(2, 1))),
    ])
    .unwrap();
    assert_eq!(g.variance().simplify(), r(272, 81));
    // A discrete pair: ½ DiscreteUniform(−3, 2) + ½ Hypergeometric(5, 1, 3);
    // E[X²] = ½·19/6 + ½·3/5 = 113/60 (it was 34/15).
    let d = Distribution::mixture(&[
        (r(1, 2), Distribution::discrete_uniform(r(-3, 1), r(2, 1))),
        (
            r(1, 2),
            Distribution::hypergeometric(r(5, 1), r(1, 1), r(3, 1)),
        ),
    ])
    .unwrap();
    assert_eq!(d.moment(2), r(113, 60));
}

/// `g(X) = a(X + s)ⁿ + b` with an odd `n` has the real inverse
/// `real_root((y − b)/a, n) − s`; the solver's principal root is complex
/// for negative arguments.  Before, the CDF of `X³` was off by −1 for a
/// t(3) (`−0.1225` at 3), a Cauchy (`−0.5` at 0) and a logistic(0, 2)
/// (`−0.3475` at 2), complex for a Cauchy at −1, and the density had the
/// wrong sign below 0 (`−0.1061` for a Cauchy at −1).  Oracle: scipy
/// `stats.t(3).cdf(3**(1/3)) = 0.8775436557138175`,
/// `stats.cauchy.cdf(-1) = 0.25`, `stats.logistic(scale=2).cdf(2**(1/3))
/// = 0.6524805112905461`, `stats.cauchy.pdf(-1)/3 = 0.05305164769729845`,
/// `stats.t(3).cdf(1) = 0.8044988905221147` (`2X³ + 1 ≤ 3`).
#[test]
fn odd_power_transform_of_a_variable_on_the_line() {
    let ctx = Context::new();
    let r = |a, b| ctx.rational(a, b);
    let t3 = RandomVariable::new(&ctx, "T", Distribution::student_t(ctx.int(3)));
    let cube = t3.transform("W", &t3.symbol().powi(3)).unwrap();
    close(
        cube.cdf(&ctx.int(3)).eval_f64().unwrap(),
        0.877_543_655_713_817_5,
        1e-12,
        "t3 X³",
    );
    let affine = t3.transform("V", &(2 * t3.symbol().powi(3) + 1)).unwrap();
    close(
        affine.cdf(&ctx.int(3)).eval_f64().unwrap(),
        0.804_498_890_522_114_7,
        1e-12,
        "2X³+1",
    );
    let c = RandomVariable::new(&ctx, "C", Distribution::cauchy(r(0, 1), r(1, 1)));
    let cc = c.transform("W", &c.symbol().powi(3)).unwrap();
    assert_eq!(cc.cdf(&ctx.int(0)).eval(), r(1, 2));
    close(
        cc.cdf(&ctx.int(-1)).eval_f64().unwrap(),
        0.25,
        1e-14,
        "Cauchy cdf(-1)",
    );
    close(
        cc.density(&ctx.int(-1)).eval_f64().unwrap(),
        0.053_051_647_697_298_45,
        1e-12,
        "pdf(-1)",
    );
    let l = RandomVariable::new(&ctx, "L", Distribution::logistic(r(0, 1), r(2, 1)));
    let lc = l.transform("W", &l.symbol().powi(3)).unwrap();
    close(
        lc.cdf(&ctx.int(2)).eval_f64().unwrap(),
        0.652_480_511_290_546_1,
        1e-12,
        "logistic",
    );
}

/// A non-polynomial `g` growing in a heavy tail at least like `|x|^r`
/// (`r` the tail order) has a divergent expectation: `+∞`, `−∞`, or NaN
/// for opposite signs at the two ends.  Before, the divergent integral
/// went to quadrature, which returned `E|X − 1| = 225.93` for a standard
/// Cauchy and `E[√(1 + X²)] = 226.37` for a t(1).  Oracle: the integrand
/// `|x|/(π(1 + x²))` decays like `1/(π|x|)`, whose integral diverges
/// (scipy `stats.cauchy.expect(abs)` warns and returns a large number);
/// `E[ln(1 + X²)] = 2 ln 2 = 1.3862943611198906` for a standard Cauchy
/// stays finite.
#[test]
fn divergent_expectations_of_non_polynomials() {
    let ctx = Context::new();
    let r = |a, b| ctx.rational(a, b);
    let c = RandomVariable::new(&ctx, "C", Distribution::cauchy(r(0, 1), r(1, 1)));
    let cs = c.symbol().clone();
    assert_eq!(c.expectation(&(&cs - 1).abs()), ctx.infinity());
    assert_eq!(c.expectation(&(-cs.abs() * &cs)), ctx.nan());
    close(
        c.expectation(&(cs.powi(2) + 1).ln()).eval_f64().unwrap(),
        1.386_294_361_119_890_6,
        1e-13,
        "ln",
    );
    let t1 = RandomVariable::new(&ctx, "T", Distribution::student_t(ctx.int(1)));
    assert_eq!(
        t1.expectation(&(t1.symbol().powi(2) + 1).sqrt()),
        ctx.infinity()
    );
    let p = RandomVariable::new(&ctx, "P", Distribution::pareto(r(1, 1), r(1, 2)));
    assert_eq!(
        p.expectation(&(-p.symbol().abs() * p.symbol())),
        ctx.neg_infinity()
    );
    // Below the order the value stands: E|T − 1| for a t(3) (mpmath
    // `quad(abs(x-1)*t3pdf, [-inf, 1, inf]) = 1.4359911241769174324`).
    let t3 = RandomVariable::new(&ctx, "T3", Distribution::student_t(ctx.int(3)));
    let e = t3.expectation(&(t3.symbol() - 1).abs());
    close(
        e.eval_f64().unwrap(),
        1.435_991_124_176_917_4,
        1e-13,
        "E|T3 - 1|",
    );
    assert_eq!(
        stats::expectation(&[&c], &(cs.powi(2) + 1).sqrt()).unwrap(),
        ctx.infinity()
    );
}

/// `E[cos L]`, `E[sin L]`, `E[e^L]` for `L` linear in independent
/// variables come from the marginal characteristic / generating
/// functions.  Before, the variables were summed out one at a time:
/// `E[cos(X + Y)]` for `X ~ Poisson(½)`, `Y ~ Poisson(3)` did not finish in
/// 10 s (35 hunter hangs).  Oracle: mpmath double sums over `0..80` of
/// `cos(i + j)·p(i)·q(j)` = `−0.19625067437912147784` and of
/// `sin(i − 2j + 1)·p(i)·q(j)` = `−0.01096055378410958856`;
/// `E[e^{X/2 + Y}] = exp(½(e^{1/2} − 1) + 3(e − 1)) = 239.65637360563477`.
#[test]
fn joint_exponentials_of_linear_forms() {
    let ctx = Context::new();
    let r = |a, b| ctx.rational(a, b);
    let x = RandomVariable::new(&ctx, "X", Distribution::poisson(r(1, 2)));
    let y = RandomVariable::new(&ctx, "Y", Distribution::poisson(r(3, 1)));
    let (xs, ys) = (x.symbol().clone(), y.symbol().clone());
    let t0 = std::time::Instant::now();
    let c = stats::expectation(&[&x, &y], &(&xs + &ys).cos()).unwrap();
    close(
        c.eval_f64().unwrap(),
        -0.196_250_674_379_121_47,
        1e-13,
        "E[cos(X + Y)]",
    );
    assert!(t0.elapsed().as_secs() < 5);
    let s = stats::expectation(&[&x, &y], &(&xs - 2 * &ys + 1).sin()).unwrap();
    close(
        s.eval_f64().unwrap(),
        -0.010_960_553_784_109_59,
        1e-12,
        "E[sin(X - 2Y + 1)]",
    );
    let e = stats::expectation(&[&x, &y], &(&xs / 2 + &ys).exp()).unwrap();
    close(
        e.eval_f64().unwrap(),
        239.656_373_605_634_77,
        1e-13,
        "E[exp(X/2 + Y)]",
    );
}

/// Sampling a periodic function with poles over `[−10³⁰⁰, 10³⁰⁰]`: the
/// pole search counted the periods in `i64` and overflowed (a panic in
/// debug builds, "attempt to subtract with overflow" in
/// `calculus_util::push_trig_zeros`, and a wrapped count in release); found
/// by the 0.46 plotting hunter.  The count saturates and the search gives
/// up on a range of more than 10,000 periods.
#[test]
fn plotting_a_tangent_over_a_huge_range_does_not_overflow() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in ["tan(x)", "1/sin(x)", "tan(x)/3 - x"] {
        let f = ctx.parse(src).unwrap();
        // Any `Ok` or `Err` will do; the test is that it returns.
        let _ = f.to_tikz(&x, -1e300, 1e300);
    }
}
