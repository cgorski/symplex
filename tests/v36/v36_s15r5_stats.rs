//! Tests of session 17, wave 16 (s15r5): statistics and integration leftovers.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use std::time::Duration;

use symplex::prelude::*;
use symplex::stats::Distribution;

use crate::perf_assert::CpuTimer;

// ── R5.1: sums over a window around the mode ────────────────────────────────

/// The value of `e` to `digits` digits, or the error text.
fn dec(e: &Ex, digits: u32) -> String {
    e.eval_decimal(digits)
        .unwrap_or_else(|err| format!("refused: {err}"))
}

/// The entropy of a binomial law with more than 10⁴ trials.  Up to 0.58 the
/// `Sum` was refused ("Sum range too large for numerical evaluation (> 10000
/// terms)"); the terms are now summed outward from the mode and both rests
/// are bounded by proved geometric series (`evalf/windowsum.rs`).
///
/// Oracle: mpmath, `−Σ P(k) ln P(k)` with `ln P(k) = loggamma(n+1) −
/// loggamma(k+1) − loggamma(n−k+1) + k ln p + (n−k) ln(1−p)` summed
/// outward from the mode until 8 consecutive terms are below `10^(−dps−8)`
/// of the sum, at 45 and 70 digits (`hunt/o1.py`):
/// `Binomial(20000, 13/100)`: 5.2809206829854938513722305016267046…,
/// `Binomial(100000, 10⁻⁶)`: 0.33367696324050460162455550443604396…
/// (the value Q4 of wave 15 gave to 20 digits).
#[test]
fn entropy_of_a_binomial_law_beyond_ten_thousand_trials() {
    let ctx = Context::new();
    let h = Distribution::binomial(ctx.int(20_000), ctx.rational(13, 100)).entropy();
    assert_eq!(dec(&h, 30), "5.28092068298549385137223050163");
    assert_eq!(h.eval_f64().unwrap(), 5.280_920_682_985_494);
    let h = Distribution::binomial(ctx.int(100_000), ctx.rational(1, 1_000_000)).entropy();
    assert_eq!(dec(&h, 30), "0.333676963240504601624555504436");
    assert_eq!(h.eval_f64().unwrap(), 0.333_676_963_240_504_6);
}

/// The window gives the digits the term-by-term sum gave, in a fraction of
/// the time: `Binomial(10⁴, 13/100).entropy` took 4.2 s in a release build
/// at 0.58 (10,001 terms, each with `ln C(n, k)` from scratch) and takes
/// 0.05 s (about 1,000 terms by the recurrences of the weights).
///
/// Oracle: the 0.58 value itself, `4.93432690480006292947334994521`, and
/// mpmath as above: 4.9343269048000629294733499452052783658…
#[test]
fn window_sum_agrees_with_the_term_by_term_sum() {
    let ctx = Context::new();
    let h = Distribution::binomial(ctx.int(10_000), ctx.rational(13, 100)).entropy();
    let timer = CpuTimer::start();
    assert_eq!(dec(&h, 30), "4.93432690480006292947334994521");
    timer.assert_within(
        Duration::from_secs(20),
        "the entropy of Binomial(10^4, 13/100)",
    );
}

/// Infinite sums whose terms peak far above the lower limit.  Up to 0.58
/// they were summed from the limit and refused beyond 20,000 terms, after
/// 3 to 15 s: the entropy of `Poisson(25000)` and `E[ln(1 + X)]` there
/// ("more than 20000 terms before the bound on the rest is below the
/// precision"), and the entropy of `NegativeBinomial(60000, 599/600)`,
/// whose terms peak at 100 but whose bound needed as many.
///
/// Oracle: mpmath direct sums as above, at 45 and 70 digits (`hunt/o1.py`):
/// 6.4822507517298382643422634003556602…, 10.126651103983676468327539117520…,
/// 3.7223513493262670174580526592045589…
#[test]
fn infinite_sums_that_peak_far_from_the_lower_limit() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let poisson = Distribution::poisson(ctx.int(25_000));
    assert_eq!(
        dec(&poisson.entropy(), 30),
        "6.48225075172983826434226340036"
    );
    let mean_log = poisson.expectation(&(&x + 1).ln(), &x);
    assert_eq!(dec(&mean_log, 30), "10.1266511039836764683275391175");
    let nb = Distribution::negative_binomial(ctx.int(60_000), ctx.rational(599, 600));
    assert_eq!(dec(&nb.entropy(), 30), "3.7223513493262670174580526592");
}

/// The entropy of a hypergeometric law with more than 64 atoms is the
/// formal `Sum` (as a binomial law's is).  Up to 0.58 the generic route
/// summed the masses exactly and merged their logarithms:
/// `Hypergeometric(4196, 3749, 3300)` (448 atoms) did not return within a
/// minute, and `Hypergeometric(50000, 20000, 10000)` (10,001 terms, each
/// with three binomials from scratch) took 8.9 s in a release build; the
/// latter is a window now (0.06 s).
///
/// Oracle: mpmath direct sums at 45 and 70 digits (`hunt/o1.py`):
/// 3.5216831307591035578998748562211…, 5.1989881403010916367123065540628…
#[test]
fn entropy_of_a_hypergeometric_law_is_a_sum() {
    let ctx = Context::new();
    let timer = CpuTimer::start();
    let h = Distribution::hypergeometric(ctx.int(4196), ctx.int(3749), ctx.int(3300)).entropy();
    assert!(h.has_unevaluated(), "{h}");
    assert_eq!(dec(&h, 30), "3.52168313075910355789987485622");
    let h = Distribution::hypergeometric(ctx.int(50_000), ctx.int(20_000), ctx.int(10_000));
    assert_eq!(dec(&h.entropy(), 30), "5.19898814030109163671230655406");
    timer.assert_within(Duration::from_secs(60), "two hypergeometric entropies");
}

/// Other moment sums take the same route: a fractional power of the index
/// (real where the index is not negative), and an irrational success
/// probability, whose constant ratio is recovered as a ball from two
/// consecutive terms.  `E[√X]` for `X ~ Binomial(30000, 1/3)` and for
/// `X ~ Binomial(100000, 1/π)` were refused (more than 10⁴ terms).
///
/// Oracle: mpmath direct sums at 45 and 70 digits (`hunt/o1.py`,
/// `hunt/o1c.py`): 99.999166628468025828960825159377… and
/// 178.41193400002636325376899785190…
#[test]
fn expectation_of_a_root_of_a_binomial_variable() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let d = Distribution::binomial(ctx.int(30_000), ctx.rational(1, 3));
    assert_eq!(
        dec(&d.expectation(&x.sqrt(), &x), 30),
        "99.9991666284680258289608251594"
    );
    let d = Distribution::binomial(ctx.int(100_000), ctx.one() / ctx.pi());
    assert_eq!(
        dec(&d.expectation(&x.sqrt(), &x), 30),
        "178.411934000026363253768997852"
    );
}

/// A tail of a hypergeometric law beyond the exact bounds, as the formal
/// sum of its masses: the window starts at the end of the range next to the
/// mode and sums downward until the rest is proved small (the next wave's
/// use, R6.6).  `Σ_{k=1750000}^{1757000} C(1875000, k) C(125000, 1875000 −
/// k)/C(2000000, 1875000)` was refused (7,001 terms would have been summed
/// one by one; 125,001 for the whole lower tail).
///
/// Oracle: mpmath, the terms from `loggamma` at 45 and 70 digits
/// (`hunt/o1b.py`): 1.2293048815506139178797261178986669324e-23 and
/// 2.527183426959515539839151345528726302562e-6.
#[test]
fn hypergeometric_tail_as_a_window_sum() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let mass = |big_n: i64, m: i64, n: i64| {
        ctx.int(m).binomial(&k) * ctx.int(big_n - m).binomial(&(ctx.int(n) - &k))
            / ctx.int(big_n).binomial(&ctx.int(n))
    };
    let tail = Ex::symbolic_sum(
        &mass(2_000_000, 1_875_000, 1_875_000),
        &k,
        &ctx.int(1_750_000),
        &ctx.int(1_757_000),
    );
    assert_eq!(dec(&tail, 30), "1.2293048815506139178797261179e-23");
    let tail = Ex::symbolic_sum(
        &mass(50_000, 20_000, 10_000),
        &k,
        &ctx.zero(),
        &ctx.int(3800),
    );
    assert_eq!(dec(&tail, 30), "2.52718342695951553983915134553e-6");
}

/// A window that would need more terms than the budget (65,536) is refused
/// at once, by an estimate of its width from the step ratios: the entropy
/// of `Binomial(10⁹, 13/100)` (a standard deviation of 10,634) and of
/// `NegativeBinomial(18141, 23/500)` (2,860), which up to 0.58 was refused
/// only after 13.7 s of summing from the lower limit.
#[test]
fn a_window_beyond_the_budget_is_refused_at_once() {
    let ctx = Context::new();
    let timer = CpuTimer::start();
    let h = Distribution::binomial(ctx.int(1_000_000_000), ctx.rational(13, 100)).entropy();
    assert!(matches!(
        h.eval_f64(),
        Err(SymplexError::Unevaluable { .. })
    ));
    let h = Distribution::negative_binomial(ctx.int(18_141), ctx.rational(23, 500)).entropy();
    assert!(matches!(
        h.eval_f64(),
        Err(SymplexError::Unevaluable { .. })
    ));
    timer.assert_within(Duration::from_secs(10), "two refusals of a wide window");
}

// ── R5.2: the far lower tail of a gamma law with a huge shape ───────────────

/// `F(x)` and `1 − F(x)` of `ChiSquared(3.4·10¹⁰)` and `Gamma(4·10¹⁰, 1)`
/// at their `f64` quantiles of `10⁻³⁰⁰`, and of the first law shifted by
/// `−9` at its own (the `fuzz_stats` slow units of 14–28 s).  Up to 0.58
/// the series of `γ(s, x)` took 170,000–300,000 terms there, 0.3–0.5 s per
/// value in a release build, and the shifted law asked for the tail six to
/// ten times per call (`cdf` 1.9 s, `sf` 2.6 s).  The sum of the series now
/// comes from its expansion in `s/(s − x)²`, whose error is at most the
/// first omitted term (`evalf/gammatail.rs`): 1–4 ms per call, the same
/// digits.
///
/// Oracle: mpmath, `exp(a ln x − x − loggamma(a + 1))·Σ xⁿ/(a + 1)ₙ` on the
/// exact doubles at 70 and 100 digits (`hunt/o2b.py`):
/// 1.00000000013867766500629251064e-300 and
/// 9.99999999318171929309129701147e-301.
#[test]
fn far_lower_tail_of_a_gamma_law_with_a_huge_shape() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let timer = CpuTimer::start();
    let chi = Distribution::chi_squared(ctx.int(34_000_000_000));
    let at = ctx.from_f64(33_990_340_213.531_32).unwrap();
    assert_eq!(dec(&chi.cdf(&at), 20), "1.000000000138677665e-300");
    assert_eq!(dec(&chi.sf(&at), 20), "1");
    let gamma = Distribution::gamma(ctx.int(40_000_000_000), ctx.one());
    let at = ctx.from_f64(39_992_591_037.895_546).unwrap();
    assert_eq!(dec(&gamma.cdf(&at), 20), "9.9999999931817192931e-301");
    // The fuzz unit: the law shifted by −9, at its own quantile.
    let shifted = chi.transformed(&x, &(&x - 9)).unwrap();
    let q = shifted.quantile_f64(1e-300).unwrap();
    assert_eq!(q, 33_990_340_204.531_32);
    let at = ctx.from_f64(q).unwrap();
    assert_eq!(dec(&shifted.cdf(&at), 20), "1.000000000138677665e-300");
    assert_eq!(dec(&shifted.sf(&at), 20), "1");
    timer.assert_within(
        Duration::from_secs(10),
        "far lower tails of huge gamma shapes",
    );
}

// ── R5.3: densities with a huge degree of freedom ───────────────────────────

/// The density of an F law with a huge degree of freedom, at a number.  Up
/// to 0.58 `(d₁/d₂)^{d₁/2}`, `x^{d₁/2−1}` and `(1 + d₁x/d₂)^{−(d₁+d₂)/2}` were
/// separate factors: from `d₁ ≈ 10⁹` or `d₂ ≈ 10¹⁹` on the density was
/// refused ("precision exhausted", or "a power: the value overflows the
/// arbitrary-precision exponent range" at `d₁ = 3·10¹²⁸`, whose first
/// factor is `2^(6·10¹³⁰)`).  The powers are now combined in logarithms,
/// `exp(−d₁·atanh(1/(2u + 1)) − d₂·atanh(u/(u + 2)))/(x·B(d₁/2, d₂/2))` with
/// `u = d₁x/d₂`, an odd `d₁` included (its half-integer power used to be
/// split into numerator and denominator).
///
/// Oracle: mpmath, `exp(−ln x − a·log1p(1/u) − b·log1p(u) − (loggamma(a) +
/// loggamma(b) − loggamma(a + b)))` at 320 and 420 digits (`hunt/h3.py`):
/// 1.06791298073867544819713352663670e-294 for `F(3·10¹²⁸, 1)` at
/// `x = 728/10⁶` (and for `d₁ + 1`, the same 30 digits),
/// 14.780403590541218395987765832… for `F(1, 3·10¹²⁸)` there.
#[test]
fn density_of_an_f_law_with_a_huge_degree_of_freedom() {
    let ctx = Context::new();
    let big = ctx.int(3) * ctx.int(10).powi(128);
    let x = ctx.rational(728, 1_000_000);
    let f = Distribution::f_distribution(big.clone(), ctx.one());
    assert_eq!(
        dec(&f.density(&x), 30),
        "1.06791298073867544819713352664e-294"
    );
    let f = Distribution::f_distribution(&big + 1, ctx.one());
    assert_eq!(
        dec(&f.density(&x), 30),
        "1.06791298073867544819713352664e-294"
    );
    let f = Distribution::f_distribution(ctx.one(), big);
    assert_eq!(dec(&f.density(&x), 30), "14.780403590541218395987765832");
    // Moderate degrees keep the classic form and its value.
    let f = Distribution::f_distribution(ctx.int(1_000_000), ctx.one());
    assert_eq!(
        dec(&f.density(&x), 30),
        "1.70967021245344845829016791813e-294"
    );
}

/// The same structure in Student's t: for a huge `ν` the quotient of the two
/// `Γ` is `√π/B(½, ν/2)` and the power an exponential of an `atanh`.  Up to
/// 0.58 the density of `StudentT(10¹²⁸)` was refused ("a power: the value
/// overflows").
///
/// Oracle: mpmath, `exp(−(ν+1)/2·log1p(x²/ν) − ln(ν)/2 − (loggamma(1/2) +
/// loggamma(ν/2) − loggamma((ν+1)/2)))` at 320 and 420 digits
/// (`hunt/h3.py`); at `ν = 10¹²⁸` it is the normal density to 127 digits,
/// `exp(−9/8)/√(2π) = 0.12951759566589172761409955795…`.
#[test]
fn density_of_a_t_law_with_a_huge_degree_of_freedom() {
    let ctx = Context::new();
    let nu = ctx.int(10).powi(128);
    let t = Distribution::student_t(nu);
    assert_eq!(
        dec(&t.density(&ctx.rational(-3, 2)), 30),
        "0.129517595665891727614099557955"
    );
    assert_eq!(dec(&t.density(&ctx.zero()), 20), "0.39894228040143267794");
}

// ── R5.5: rational functions of cˣ for a complex constant c ─────────────────

/// `∫ dx/((1+i)ˣ − 2i)`: the integrand has a pole at `x = 2` (`(1+i)² =
/// 2i`), and an antiderivative exists on each side of it.  Up to 0.58 the
/// integral stayed unevaluated: the point where `|(1+i)ˣ| = |2i|` was read
/// as `ln 2/ln √2`, a quotient `eval` leaves as it is, and the integer part
/// of the turn count at the pole did not settle.  The joining point is now
/// the number 2 (`ln √2 = ½ ln 2`), and the answer a `Piecewise` at it.
/// A definite integral on one side is its Newton–Leibniz difference; one
/// across the pole is `Divergent`, as before.
///
/// Oracle: `F′ = f` at six points on both sides (differentiation), and
/// mpmath `quad` over 16 subintervals at 30 and 50 digits (`hunt/o5.py`):
/// `∫₀¹` = 0.3197829712935262294048226208 + 0.4362782508807571800497636468i,
/// `∫₃⁴` = −0.3197829712935262294048226208 + 0.06372174911924281995023635317i.
#[test]
fn antiderivative_of_a_complex_power_with_a_pole_on_the_path() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("1/((1+I)^x - 2*I)").unwrap();
    let antiderivative = f.integrate(&x);
    assert!(!antiderivative.has_unevaluated(), "{antiderivative}");
    let derivative = antiderivative.diff(&x);
    for (n, d) in [(0, 1), (1, 1), (3, 2), (5, 2), (3, 1), (7, 1)] {
        let at = ctx.rational(n, d);
        let got = derivative.subs(&x, &at).eval_complex64().unwrap();
        let want = f.subs(&x, &at).eval_complex64().unwrap();
        assert!(
            (got - want).norm() < 1e-13,
            "F′ ≠ f at {n}/{d}: {got} vs {want}"
        );
    }
    let over = |a: i64, b: i64| f.try_integrate_definite(&x, &ctx.int(a), &ctx.int(b));
    assert_eq!(
        dec(&over(0, 1).unwrap(), 25),
        "0.3197829712935262294048226 + 0.4362782508807571800497636*i"
    );
    assert_eq!(
        dec(&over(3, 4).unwrap(), 25),
        "-0.3197829712935262294048226 + 0.06372174911924281995023635*i"
    );
    assert!(matches!(over(0, 3), Err(SymplexError::Divergent { .. })));
}

/// `∫₁² dx/((2−i)^{2x} − 1 + i)` and `∫₁² dx/((2−i)ˣ − 1 + i)`.  Up to 0.58
/// both were refused ("no antiderivative in closed form"): the logarithm
/// `ln((u − 1)·i + 1)` of the answer in `u = (2−i)ˣ` was not read as one of
/// a linear factor (`i·u + 1 − i`), so no form continuous along the spiral
/// was built.
///
/// Oracle: mpmath `quad` over 16 subintervals at 30 and 50 digits
/// (`hunt/o5.py`): 0.02804980577252267864674770828 +
/// 0.1109894591859010007316415973i and 0.4761603746499091389567291818 +
/// 0.2486277001370791003682620351i.
#[test]
fn integral_of_a_complex_power_with_a_gaussian_root() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let over = |text: &str| {
        let f = ctx.parse(text).unwrap();
        assert!(!f.integrate(&x).has_unevaluated(), "{text}");
        dec(
            &f.try_integrate_definite(&x, &ctx.int(1), &ctx.int(2))
                .unwrap(),
            25,
        )
    };
    assert_eq!(
        over("1/((2-I)^(2*x) - 1 + I)"),
        "0.02804980577252267864674771 + 0.1109894591859010007316416*i"
    );
    assert_eq!(
        over("1/((2-I)^x - 1 + I)"),
        "0.4761603746499091389567292 + 0.248627700137079100368262*i"
    );
}

/// The values of the statistics oracle's rows `r5.*` where scipy 1.18.1 is
/// off (`scripts/stats_oracle/allowlist.tsv`): the entropy of
/// `Binomial(10⁵, 13/100)` (scipy 6.0524), the far lower tail of
/// `ChiSquared(3.4·10⁷)` (scipy 6.5·10⁻⁶ off) and the density of
/// `F(7, 10¹⁹)` at 3/2 (scipy 1.0).  Refused (the first and the last) or
/// ten times slower (the tail) up to 0.58.
///
/// Oracle: mpmath at two precisions — the direct sum of `hunt/o1.py`
/// (6.0856557827955500291667413823640633…), the oracle's `refs.py` series
/// (9.533213002300690250006557e-131) and log-space density
/// (0.3490149774598276049984451).
#[test]
fn values_where_scipy_is_off() {
    let ctx = Context::new();
    let h = Distribution::binomial(ctx.int(100_000), ctx.rational(13, 100)).entropy();
    assert_eq!(dec(&h, 30), "6.08565578279555002916674138236");
    let chi = Distribution::chi_squared(ctx.int(34_000_000));
    assert_eq!(
        dec(&chi.cdf(&ctx.rational(67_600_001, 2)), 20),
        "9.53321300230069025e-131"
    );
    let f = Distribution::f_distribution(ctx.int(7), ctx.int(10).powi(19));
    assert_eq!(
        dec(&f.density(&ctx.rational(3, 2)), 20),
        "0.349014977459827605"
    );
}
