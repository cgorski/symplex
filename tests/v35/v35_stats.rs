//! The statistics hunt: moment generating and characteristic functions,
//! moments beyond a heavy tail's order, entropies, and the binomial at
//! large `n` (`symplex::stats`).

use symplex::prelude::*;
use symplex::stats::Distribution;
use symplex::stats::order::{maximum_of, minimum_of};

fn close(actual: &Ex, want: f64, rel: f64, label: &str) {
    let v = actual
        .eval_f64()
        .unwrap_or_else(|e| panic!("{label}: `{actual}` does not evaluate: {e}"));
    assert!(
        (v - want).abs() <= rel * want.abs(),
        "{label}: got {v}, want {want}"
    );
}

fn close_complex(actual: &Ex, re: f64, im: f64, rel: f64, label: &str) {
    let z = actual
        .eval_complex64()
        .unwrap_or_else(|e| panic!("{label}: `{actual}` does not evaluate: {e}"));
    let err = ((z.re - re).powi(2) + (z.im - im).powi(2)).sqrt();
    assert!(
        err <= rel * (re * re + im * im).sqrt(),
        "{label}: got {} + {}i, want {re} + {im}i",
        z.re,
        z.im
    );
}

/// `E[e^{tX}] = +∞` for a real `t` beyond the abscissa of convergence:
/// the integrand is positive and its integral diverges (mpmath:
/// `quad(lambda x: exp(2*x)*exp(-x), [0, inf])` grows without bound).
/// Before, the closed form answered for every `t`, continued past its
/// pole: `Exponential(1).mgf(2)` was `−1`, `Gamma(2, 1).mgf(2)` was `1`,
/// `Laplace(0, 1).mgf(2)` was `−1/3`, `Geometric(1/2).mgf(1)` was
/// `−3.784`, `NegativeBinomial(2, 1/2).mgf(1)` was `1.938`; SymPy 1.14
/// does the same (`moment_generating_function(Exponential('X', 1))(2)` =
/// `-1`).  The heavy right tails (log-normal, Pareto, F, Weibull with shape
/// below 1, Cauchy, Student t) handed their divergent integral to
/// quadrature (the hunter recorded finite log-normal values).  A symbolic
/// `t` gets the `Piecewise`, as `cdf` does for a symbolic `x`.
#[test]
fn mgf_beyond_the_abscissa_of_convergence_is_infinite() {
    let ctx = Context::new();
    let r = |a, b| ctx.rational(a, b);
    let inf = ctx.infinity();
    let e1 = Distribution::exponential(ctx.one());
    assert_eq!(e1.mgf(&ctx.int(2)), inf);
    assert_eq!(e1.mgf(&ctx.one()), inf, "the end of (−∞, λ)");
    assert_eq!(e1.mgf(&r(1, 2)), ctx.int(2), "inside: λ/(λ − t)");
    assert_eq!(
        Distribution::gamma(ctx.int(2), ctx.one()).mgf(&ctx.int(2)),
        inf
    );
    let laplace = Distribution::laplace(ctx.zero(), ctx.one());
    assert_eq!(laplace.mgf(&ctx.int(2)), inf);
    assert_eq!(laplace.mgf(&ctx.int(-1)), inf, "|t| < 1/b on both sides");
    assert_eq!(Distribution::geometric(r(1, 2)).mgf(&ctx.one()), inf);
    assert_eq!(
        Distribution::negative_binomial(ctx.int(2), r(1, 2)).mgf(&ctx.one()),
        inf
    );
    for (name, d) in [
        ("LogNormal", Distribution::log_normal(ctx.zero(), ctx.one())),
        ("Pareto", Distribution::pareto(ctx.one(), ctx.int(3))),
        ("F", Distribution::f_distribution(ctx.int(3), ctx.int(5))),
        ("Weibull", Distribution::weibull(ctx.one(), r(1, 2))),
        ("Cauchy", Distribution::cauchy(ctx.zero(), ctx.one())),
        ("StudentT", Distribution::student_t(ctx.int(3))),
    ] {
        assert_eq!(d.mgf(&r(1, 10)), inf, "{name}");
        assert_eq!(d.mgf(&ctx.zero()), ctx.one(), "{name} at 0");
    }
    // Inside the domain of a heavy right tail: the integral.
    // mpmath: quad(lambda x: exp(-x/3)*npdf(log(x), 1/4, 1/10)/x, [0, 1, 2, inf])
    //   = 0.65101142925647707984
    let ln = Distribution::log_normal(r(1, 4), r(1, 10));
    close(
        &ln.mgf(&r(-1, 3)),
        0.651_011_429_256_477,
        1e-12,
        "LogNormal mgf(−1/3)",
    );
    // A symbolic t.
    let t = ctx.symbol("t");
    let e3 = Distribution::exponential(ctx.int(3));
    let closed = ctx.int(3) / (ctx.int(3) - &t);
    assert_eq!(
        e3.mgf(&t),
        Ex::piecewise(&[(&closed, &t.lt(&ctx.int(3))), (&inf, &ctx.bool_true())])
    );
    assert_eq!(e3.mgf(&t).subs(&t, &ctx.int(4)).eval(), inf);
}

/// A non-real `t` is placed by its real part: inside the strip the closed
/// form continues (`E[e^{iX}] = 1/(1 − i)` for an `Exponential(1)`), on or
/// beyond its edge the integral has no value.  Before, the closed form
/// answered: `Exponential(1).mgf(1 + i)` was `i` (`1/(1 − (1 + i))`);
/// with the real-axis domain alone it became the undecidable
/// `Piecewise(i if 1 > 1 + i, …)`.  (mpmath: `quad(lambda x: exp((1+1j)*x)
/// *exp(-x), [0, inf])` does not converge.)
#[test]
fn mgf_of_a_complex_argument_is_placed_by_its_real_part() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let e1 = Distribution::exponential(ctx.one());
    assert_eq!(e1.mgf(&(ctx.one() + &i)), ctx.nan());
    close_complex(&e1.mgf(&i), 0.5, 0.5, 1e-15, "mgf(i)");
    close_complex(&e1.mgf(&(ctx.int(-1) + &i)), 0.4, 0.2, 1e-15, "mgf(−1 + i)");
}

/// The wrappers carry the domain: `aX + b` scales it by `1/a` (swapping the
/// ends for `a < 0`), a mixture intersects its components', an order
/// statistic scales the rate of each tail.  Before, `(−2·Exponential(1)).
/// mgf(−1)` was `−1` (`M_X(2)` continued); the others were divergent
/// integrals.  Inside: `minimum_of(Exponential(1), 3)` is `Exponential(3)`,
/// mgf `3/(3 − t)`; `maximum_of(Exponential(1), 3).mgf(1/2)` =
/// `Γ(4)Γ(1/2)/Γ(7/2) = 16/5`.
#[test]
fn mgf_of_wrapped_distributions_respects_the_domain() {
    let ctx = Context::new();
    let r = |a, b| ctx.rational(a, b);
    let e1 = Distribution::exponential(ctx.one());
    let neg = e1.affine(ctx.int(-2), ctx.zero()).unwrap();
    assert_eq!(neg.mgf(&ctx.int(-1)), ctx.infinity());
    assert_eq!(neg.mgf(&ctx.one()), r(1, 3), "1/(1 + 2t)");
    let mix = Distribution::mixture(&[
        (r(1, 2), Distribution::normal(ctx.zero(), ctx.one())),
        (r(1, 2), Distribution::pareto(ctx.one(), r(5, 2))),
    ])
    .unwrap();
    assert_eq!(mix.mgf(&r(1, 10)), ctx.infinity());
    // Symbolic rates: the domain t < Min(λ₁, λ₂) (it was none, and the
    // mixture's closed form answered for every t).
    let l1 = ctx.symbol_with("l1", &[Assumption::Positive]).unwrap();
    let l2 = ctx.symbol_with("l2", &[Assumption::Positive]).unwrap();
    let sym_mix = Distribution::mixture(&[
        (r(1, 2), Distribution::exponential(l1.clone())),
        (r(1, 2), Distribution::exponential(l2.clone())),
    ])
    .unwrap();
    let at = |t: Ex| {
        sym_mix
            .mgf(&t)
            .subs(&l1, &ctx.one())
            .subs(&l2, &ctx.int(3))
            .eval()
    };
    assert_eq!(at(ctx.int(2)), ctx.infinity());
    assert_eq!(at(r(1, 2)), r(8, 5), "(2 + 6/5)/2");
    let min3 = minimum_of(&e1, 3).unwrap();
    assert_eq!(min3.mgf(&ctx.int(4)), ctx.infinity());
    assert_eq!(min3.mgf(&ctx.int(2)), ctx.int(3));
    assert_eq!(maximum_of(&e1, 3).unwrap().mgf(&r(1, 2)), r(16, 5));
}

/// A moment whose integral diverges is `+∞` (an even moment, or a heavy
/// tail on one side), undefined (NaN: an odd moment with both tails heavy,
/// or a central moment without a mean), from the family's tail orders —
/// `ν` for a Student t, `α` for a Pareto, `d₂/2` for an F, `1` for a
/// Cauchy.  Before, these were the divergent integrals, unevaluated.
/// scipy: `t.stats(1.5, moments='mv')` = `(0, inf)`, `t.stats(0.5,
/// moments='v')` = `nan`, `pareto.stats(0.5, moments='m')` = `inf`,
/// `f.stats(3, 4, moments='v')` = `inf`, `cauchy.stats(moments='mv')` =
/// `(nan, nan)` (scipy's t mean for ν ≤ 1 is `inf`; it is undefined, as
/// Wikipedia and the Cauchy's `nan` have it).
#[test]
fn moments_beyond_the_tail_order_are_infinite_or_undefined() {
    let ctx = Context::new();
    let r = |a, b| ctx.rational(a, b);
    let (inf, nan) = (ctx.infinity(), ctx.nan());
    let t32 = Distribution::student_t(r(3, 2));
    assert_eq!(t32.mean(), ctx.zero());
    assert_eq!(t32.variance(), inf);
    assert_eq!(t32.moment(3), nan);
    let t12 = Distribution::student_t(r(1, 2));
    assert_eq!(t12.mean(), nan);
    assert_eq!(t12.variance(), nan);
    assert_eq!(t12.moment(2), inf);
    assert_eq!(Distribution::student_t(ctx.int(5)).moment(6), inf);
    let p12 = Distribution::pareto(ctx.one(), r(1, 2));
    assert_eq!(p12.mean(), inf);
    assert_eq!(p12.variance(), inf, "scipy's pareto variance for b ≤ 2");
    assert_eq!(
        Distribution::f_distribution(ctx.int(3), ctx.int(4)).variance(),
        inf
    );
    let c = Distribution::cauchy(ctx.zero(), ctx.one());
    assert_eq!(c.mean(), nan);
    assert_eq!(c.variance(), nan);
    assert_eq!(c.moment(2), inf);
    assert_eq!(c.central_moment(2), nan);
    // E[X² − X] has the divergence of its leading term.
    let x = ctx.symbol("x");
    assert_eq!(c.expectation(&(x.powi(2) - &x), &x), inf);
    // Below the orders, the closed forms are unchanged.
    assert_eq!(Distribution::student_t(r(9, 2)).kurtosis(), ctx.int(15));
}

/// Skewness and kurtosis past the orders: `E[(X − μ)³]` diverges to `+∞`
/// in a right tail alone, so a Pareto with `2 < α ≤ 3` has skewness `+∞`
/// (its variance is finite), and the kurtosis is `+∞` wherever the fourth
/// moment diverges and the variance does not.  Before: the quotients of
/// divergent integrals, unevaluated.  scipy: `t.stats(2.5, moments='sk')`
/// = `(nan, inf)` (kurtosis excess); `pareto.stats(2.5, moments='sk')` =
/// `(nan, nan)` — scipy does not tell `+∞` from undefined there.
#[test]
fn skewness_and_kurtosis_beyond_the_tail_order() {
    let ctx = Context::new();
    let r = |a, b| ctx.rational(a, b);
    let (inf, nan) = (ctx.infinity(), ctx.nan());
    let p52 = Distribution::pareto(ctx.one(), r(5, 2));
    assert_eq!(p52.skewness(), inf);
    assert_eq!(p52.kurtosis(), inf);
    let p32 = Distribution::pareto(ctx.one(), r(3, 2));
    assert_eq!(p32.skewness(), nan, "∞/∞");
    let t52 = Distribution::student_t(r(5, 2));
    assert_eq!(t52.skewness(), nan);
    assert_eq!(t52.kurtosis(), inf);
    assert_eq!(
        Distribution::f_distribution(ctx.int(3), ctx.int(6)).skewness(),
        inf
    );
    assert_eq!(
        Distribution::f_distribution(ctx.int(3), ctx.int(8)).kurtosis(),
        inf
    );
}

/// A symbolic parameter gets the `Piecewise` of the cases.  Before, the
/// closed form answered for every parameter: `StudentT(ν)`'s variance was
/// `ν/(ν − 2)` (`−1/4` at `ν = 2/5`, `−3` at `ν = 3/2`), its kurtosis `−1`
/// at `ν = 5/2`; `Pareto(1, α)`'s mean was `α/(α − 1)` (`−1` at
/// `α = 1/2`), its kurtosis `93/5` at `α = 5/2`.  SymPy 1.14:
/// `integrate(x**2*density(StudentT('X', nu))(x), (x, -oo, oo))` =
/// `Piecewise((…, nu > 2), (Integral(…), True))`.
#[test]
fn symbolic_moments_carry_their_existence_conditions() {
    let ctx = Context::new();
    let r = |a, b| ctx.rational(a, b);
    let (inf, nan) = (ctx.infinity(), ctx.nan());
    let nu = ctx.symbol_with("nu", &[Assumption::Positive]).unwrap();
    let t = Distribution::student_t(nu.clone());
    let at = |e: Ex, v: Ex| e.subs(&nu, &v).eval();
    assert_eq!(at(t.variance(), r(2, 5)), nan);
    assert_eq!(at(t.variance(), r(3, 2)), inf);
    assert_eq!(at(t.variance(), ctx.int(3)), ctx.int(3));
    assert_eq!(at(t.kurtosis(), r(5, 2)), inf);
    assert_eq!(at(t.kurtosis(), ctx.int(5)), ctx.int(9));
    assert_eq!(at(t.mean(), r(1, 2)), nan);
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();
    let p = Distribution::pareto(ctx.one(), a.clone());
    let at = |e: Ex, v: Ex| e.subs(&a, &v).eval();
    assert_eq!(at(p.mean(), r(1, 2)), inf);
    assert_eq!(at(p.mean(), ctx.int(3)), r(3, 2));
    assert_eq!(at(p.kurtosis(), r(5, 2)), inf);
    assert_eq!(at(p.kurtosis(), r(9, 2)), r(1345, 9));
    // A parameter known to exceed the order keeps the closed form.
    let k = ctx.symbol_with("k", &[Assumption::Positive]).unwrap();
    let p3 = Distribution::pareto(ctx.one(), &k + 3);
    assert!(!format!("{}", p3.variance()).contains("Piecewise"));
}

/// A triangular distribution with symbolic ends and mode: the general
/// closed forms divide by `(c − a)(b − c)`, so once a mode at an end was
/// substituted the third moment, skewness, kurtosis and mgf were NaN (`0/0`)
/// — they now carry the end forms in a `Piecewise`.  scipy:
/// `triang(1, loc=0, scale=3).moment(3)` = 10.8, `triang(0, 0, 3).moment(3)`
/// = 2.7, `triang(1, 0, 3).stats(moments='sk')` = (−0.5656854249492381,
/// −0.6) (excess); mgf at `t = 1/30` with `c = b = 3`: `200(1 − 0.9e^{1/10})`
/// = 1.0692347463834275.
#[test]
fn triangular_with_a_symbolic_mode_at_an_end() {
    let ctx = Context::new();
    let (a, b, c) = (ctx.symbol("a"), ctx.symbol("b"), ctx.symbol("c"));
    let d = Distribution::triangular(a.clone(), b.clone(), c.clone());
    let at = |e: Ex, mode: i64| {
        e.subs(&a, &ctx.zero())
            .subs(&b, &ctx.int(3))
            .subs(&c, &ctx.int(mode))
            .eval()
    };
    assert_eq!(at(d.moment(3), 3), ctx.rational(54, 5));
    assert_eq!(at(d.moment(3), 0), ctx.rational(27, 10));
    close(
        &at(d.skewness(), 3),
        -0.565_685_424_949_238,
        1e-14,
        "skewness",
    );
    close(&at(d.kurtosis(), 0), 2.4, 1e-14, "kurtosis");
    let t = ctx.rational(1, 30);
    close(&at(d.mgf(&t), 3), 1.069_234_746_383_427_5, 1e-14, "mgf");
    // The mode inside: the general form, unchanged.
    assert_eq!(
        at(d.moment(3), 1),
        Distribution::triangular(ctx.zero(), ctx.int(3), ctx.one()).moment(3)
    );
}

/// The wrappers transport the tail orders: `aX + b` swaps the sides for
/// `a < 0` (an odd moment then diverges to `−∞`), the maximum of two
/// Cauchy variables has a right tail of order 1 and a left one of order 2
/// (`E = +∞`, not undefined), a mixture takes each side's heaviest
/// component.  Before: divergent integrals, unevaluated.
#[test]
fn wrapped_heavy_tails() {
    let ctx = Context::new();
    let r = |a, b| ctx.rational(a, b);
    let p = Distribution::pareto(ctx.one(), r(3, 2));
    let neg = p.affine(ctx.int(-2), ctx.one()).unwrap();
    assert_eq!(neg.mean(), ctx.int(-5));
    assert_eq!(neg.moment(3), ctx.neg_infinity());
    assert_eq!(neg.variance(), ctx.infinity());
    let c = Distribution::cauchy(ctx.zero(), ctx.one());
    assert_eq!(maximum_of(&c, 2).unwrap().mean(), ctx.infinity());
    assert_eq!(minimum_of(&c, 2).unwrap().mean(), ctx.neg_infinity());
    let mix = Distribution::mixture(&[
        (r(1, 2), Distribution::normal(ctx.zero(), ctx.one())),
        (r(1, 2), Distribution::pareto(ctx.one(), r(5, 2))),
    ])
    .unwrap();
    assert_eq!(mix.variance(), r(83, 36));
    assert_eq!(mix.moment(3), ctx.infinity());
    assert_eq!(mix.skewness(), ctx.infinity());
}

/// Characteristic functions in closed form where they exist (SymPy's
/// `_characteristic_function`), else `E[cos tX] + i·E[sin tX]` by real
/// quadrature.  Before: `E[e^{itX}]` stayed an integral whose integrand
/// does not compile ("cannot compile `ImaginaryUnit`"), and the logistic's
/// continued mgf `B(1 − i, 1 + i)` did not evaluate.  Oracles: `exp(-2)`;
/// `(1 + sqrt(3))*exp(-sqrt(3))` (Student t, ν = 3); `pi/sinh(pi)`;
/// mpmath `quad(lambda u: cos(u**2)*exp(-u), linspace(0, 60, 600) + [inf])`
/// = 0.53487797453351756632 and the `sin` one 0.27051358016221414426
/// (Weibull(1, 1/2), `x = u²`); `quadosc` of the log-normal's
/// = 0.34030108572578101207 + 0.50718984169180596808i.
#[test]
fn characteristic_functions() {
    let ctx = Context::new();
    let r = |a, b| ctx.rational(a, b);
    let one = ctx.one();
    let c = Distribution::cauchy(ctx.zero(), one.clone());
    close(
        &c.characteristic_function(&ctx.int(2)),
        0.135_335_283_236_612_7,
        1e-15,
        "Cauchy",
    );
    close(
        &Distribution::student_t(ctx.int(3)).characteristic_function(&one),
        0.483_357_724_596_507_65,
        1e-14,
        "StudentT(3)",
    );
    let logistic = Distribution::logistic(ctx.zero(), one.clone());
    close(
        &logistic.characteristic_function(&one),
        0.272_029_054_982_133_16,
        1e-14,
        "Logistic",
    );
    let t = ctx.symbol("t");
    let symbolic = logistic.characteristic_function(&t);
    assert_eq!(symbolic.subs(&t, &ctx.zero()).eval(), one, "1 at t = 0");
    close_complex(
        &Distribution::weibull(one.clone(), r(1, 2)).characteristic_function(&one),
        0.534_877_974_533_517_6,
        0.270_513_580_162_214_14,
        1e-12,
        "Weibull(1, 1/2)",
    );
    close_complex(
        &Distribution::log_normal(ctx.zero(), one.clone()).characteristic_function(&one),
        0.340_301_085_725_781,
        0.507_189_841_691_806,
        1e-12,
        "LogNormal(0, 1)",
    );
}

/// The beta's mgf is Kummer's series `₁F₁(α; α + β; t)` (SymPy:
/// `hyper((alpha,), (alpha + beta,), t)`), summed by `evalf` with a bound.
/// Before: `∫₀¹ e^{tx} f(x) dx` by f64 quadrature, which the endpoint
/// singularity of `β = 1/2` left wrong in the 8th digit:
/// `Beta(5, 1/2).mgf(1/10)` was 1.0952385480452327.  mpmath:
/// `hyp1f1(5, 11/2, 1/10)` = 1.0952385618059876010.
#[test]
fn beta_mgf_is_the_kummer_series() {
    let ctx = Context::new();
    let b = Distribution::beta(ctx.int(5), ctx.rational(1, 2));
    close(
        &b.mgf(&ctx.rational(1, 10)),
        1.095_238_561_805_987_6,
        1e-15,
        "Beta(5, 1/2)",
    );
    // mpmath: hyp1f1(1/2, 1, -50) = e^{-25} I_0(25) = 0.080196773547436...
    let arcsine = Distribution::beta(ctx.rational(1, 2), ctx.rational(1, 2));
    close(
        &arcsine.mgf(&ctx.int(-50)),
        0.080_196_773_547_436_7,
        1e-12,
        "Beta(1/2, 1/2)",
    );
}

/// Closed entropies where the generic `−E[ln f]` did not evaluate: the
/// geometric's `(−(1−p) ln(1−p) − p ln p)/p` (an infinite sum before),
/// the F's `ln(d₂/d₁) + ln B(a, b) − (a−1)ψ(a) − (b+1)ψ(b) + (a+b)ψ(a+b)`
/// (quadrature did not converge for a small `d₁`).  scipy:
/// `geom(538/835).entropy()` = 1.0102214491803392, `f(0.1, 7).entropy()`
/// = −13.300982166491304, `f(3, 5).entropy()` = 1.4281442318551711.
#[test]
fn entropy_of_the_geometric_and_the_f() {
    let ctx = Context::new();
    let r = |a, b| ctx.rational(a, b);
    close(
        &Distribution::geometric(r(538, 835)).entropy(),
        1.010_221_449_180_339_3,
        1e-14,
        "Geometric",
    );
    close(
        &Distribution::f_distribution(r(1, 10), ctx.int(7)).entropy(),
        -13.300_982_166_491_306,
        1e-13,
        "F(1/10, 7)",
    );
    close(
        &Distribution::f_distribution(ctx.int(3), ctx.int(5)).entropy(),
        1.428_144_231_855_172,
        1e-13,
        "F(3, 5)",
    );
    assert_eq!(Distribution::geometric(ctx.one()).entropy(), ctx.zero());
}

/// The binomial's tails at a numeric `n` and a rational `p` by the exact
/// term recurrence, its entropy as a `Sum` that `evalf` sums term by term.
/// Before, `Binomial(1000, 167/716).sf(0)` and its entropy each took more
/// than 10 s (the summation added rationals with ever larger
/// denominators; the entropy merged a thousand logarithms).  mpmath:
/// `1 - (549/716)**1000`; `fsum(binomial(1000, k)*q**k*(1-q)**(1000-k) for
/// k in range(301, 1001))` = 5.0712446841012056e-7;
/// `-fsum(P(k)*log(P(k)))` = 4.0120490417561735 (scipy
/// `binom(1000, 167/716).entropy()` = 4.01204904175617).
#[test]
fn binomial_tails_and_entropy_at_large_n() {
    let ctx = Context::new();
    let q = num_rational::Ratio::new(num_bigint::BigInt::from(549), num_bigint::BigInt::from(716));
    let q1000 = q.pow(1000);
    let b = Distribution::binomial(ctx.int(1000), ctx.rational(167, 716));
    assert_eq!(
        b.sf(&ctx.zero()),
        ctx.from_ratio(num_rational::Ratio::from_integer(1.into()) - &q1000)
    );
    assert_eq!(b.cdf(&ctx.zero()), ctx.from_ratio(q1000));
    close(
        &b.sf(&ctx.int(300)),
        5.071_244_684_101_206e-7,
        1e-13,
        "sf(300)",
    );
    close(&b.entropy(), 4.012_049_041_756_174, 1e-13, "entropy");
    assert_eq!(b.cdf(&ctx.int(-1)), ctx.zero());
    assert_eq!(b.sf(&ctx.int(1000)), ctx.zero());
}
