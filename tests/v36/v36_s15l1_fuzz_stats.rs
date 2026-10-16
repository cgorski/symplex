//! Tests of session 15 (s15l1): property fuzzing of the symbolic distribution layer.
//! Every test says what was wrong before and cites its oracle.
//!
//! The findings of `fuzz/fuzz_targets/fuzz_stats.rs`, each pinned as a
//! regression test.

#![allow(unused_imports)]
use symplex::prelude::*;
use symplex::stats::Distribution;

// ── Parameter validation ────────────────────────────────────────────────────

/// `try_normal` accepted an infinite, NaN or non-real mean or standard
/// deviation (0.56: `Normal(∞, 1)`, `Normal(0, NaN)`, `Normal(i, 1)` and
/// `Normal(0, ∞)` were all `Ok`).  Oracle: the definition — a normal
/// distribution has a finite real mean and a finite positive σ.
#[test]
fn normal_with_a_non_finite_or_non_real_parameter_is_rejected() {
    let ctx = Context::new();
    let one = ctx.one();
    for (mean, std) in [
        (ctx.infinity(), one.clone()),
        (ctx.neg_infinity(), one.clone()),
        (ctx.nan(), one.clone()),
        (ctx.i_unit(), one.clone()),
        (ctx.int(1) + ctx.i_unit(), one.clone()),
        (ctx.zero(), ctx.infinity()),
        (ctx.zero(), ctx.nan()),
        (ctx.zero(), ctx.complex_infinity()),
    ] {
        assert!(
            Distribution::try_normal(mean.clone(), std.clone()).is_err(),
            "Normal({mean}, {std}) was accepted"
        );
    }
    assert!(Distribution::try_normal(ctx.int(2).sqrt(), ctx.pi()).is_ok());
}

/// `try_uniform(−∞, 0)` was accepted: its width `0 − (−∞) = ∞` is
/// positive (0.56).  Oracle: a uniform distribution needs finite ends.
#[test]
fn uniform_with_an_infinite_end_is_rejected() {
    let ctx = Context::new();
    assert!(Distribution::try_uniform(ctx.neg_infinity(), ctx.zero()).is_err());
    assert!(Distribution::try_uniform(ctx.zero(), ctx.infinity()).is_err());
    assert!(Distribution::try_triangular(ctx.zero(), ctx.infinity(), ctx.one()).is_err());
    assert!(Distribution::try_cauchy(ctx.nan(), ctx.one()).is_err());
    assert!(Distribution::try_log_normal(ctx.infinity(), ctx.one()).is_err());
    assert!(Distribution::try_exponential(ctx.infinity()).is_err());
    assert!(Distribution::try_poisson(ctx.nan()).is_err());
}

/// The discrete checks read only rational parameters, so a symbol-free
/// irrational one escaped them: `Bernoulli(√2)` (a probability of
/// 1.414…), `Binomial(√2, ½)`, `Geometric(−π/4)` and
/// `DiscreteUniform(0, 5√2)` were accepted (0.56).  Oracle: `√2 > 1`,
/// `−π/4 < 0`, and `√2`, `5√2` are not integers.
#[test]
fn discrete_families_reject_irrational_invalid_parameters() {
    let ctx = Context::new();
    let r2 = ctx.int(2).sqrt();
    let half = ctx.rational(1, 2);
    assert!(Distribution::try_bernoulli(r2.clone()).is_err());
    assert!(Distribution::try_bernoulli(-&r2 / 2).is_err());
    assert!(Distribution::try_binomial(r2.clone(), half.clone()).is_err());
    assert!(Distribution::try_binomial(ctx.int(4), &r2 / 2 + 1).is_err());
    assert!(Distribution::try_geometric(-ctx.pi() / 4).is_err());
    assert!(Distribution::try_negative_binomial(ctx.int(3), ctx.pi() / 2).is_err());
    assert!(Distribution::try_discrete_uniform(ctx.zero(), &r2 * 5).is_err());
    assert!(Distribution::try_hypergeometric(ctx.int(10), &r2 * 5, ctx.int(3)).is_err());
    assert!(Distribution::try_binomial(ctx.infinity(), half.clone()).is_err());
    assert!(Distribution::try_bernoulli(ctx.nan()).is_err());
    // Valid irrational parameters are still accepted.
    assert!(Distribution::try_bernoulli(&r2 / 2).is_ok());
    assert!(Distribution::try_binomial(ctx.int(4), ctx.pi() / 4).is_ok());
    assert!(Distribution::try_geometric(ctx.e().powi(-1)).is_ok());
}

/// A finite table with irrational masses was not checked at all: a
/// negative mass `−√2/2` and masses `√2/2 + 1/7`, `1 − √2/2` summing to
/// `8/7` were accepted (0.56).  Oracle: exact arithmetic.
#[test]
fn finite_table_with_irrational_masses_is_validated() {
    let ctx = Context::new();
    let p = ctx.int(2).sqrt() / 2;
    let negative = vec![(ctx.zero(), -&p), (ctx.one(), ctx.one() + &p)];
    assert!(Distribution::try_finite(&ctx, negative).is_err());
    let too_much = vec![
        (ctx.zero(), &p + &ctx.rational(1, 7)),
        (ctx.one(), ctx.one() - &p),
    ];
    assert!(Distribution::try_finite(&ctx, too_much).is_err());
    let fine = vec![(ctx.zero(), p.clone()), (ctx.one(), ctx.one() - &p)];
    assert!(Distribution::try_finite(&ctx, fine).is_ok());
    let infinite_value = vec![(ctx.infinity(), ctx.one())];
    assert!(Distribution::try_finite(&ctx, infinite_value).is_err());
}

// ── Exact masses that would be enormous rationals ──────────────────────────────────────

/// `Binomial(290000, p).cdf(0)` with `p` the double `1e-9` (an exact
/// dyadic with an 82-bit denominator) ran for minutes (0.56): the exact
/// route formed `(1 − p)ⁿ` over `Dⁿ`, a 24-million-bit rational, and
/// reduced it by a gcd quadratic in that size.  Beyond `2¹⁶` bits the sum
/// stays symbolic for `evalf`.  Oracle: mpmath at 30 and 60 digits,
/// `(1 − p)^290000 = 0.99971004204579050338` and
/// `1 − (1 − p)^290000 = 2.8995795420949662426e-4`.
#[test]
fn binomial_cdf_with_a_long_dyadic_probability_is_fast() {
    let ctx = Context::new();
    let p = ctx.from_f64(1e-9).unwrap();
    let d = Distribution::binomial(ctx.int(290_000), p);
    let t = std::time::Instant::now();
    let c = d.cdf(&ctx.int(0)).eval_f64().unwrap();
    let s = d.sf(&ctx.int(0)).eval_f64().unwrap();
    assert!(t.elapsed().as_secs_f64() < 5.0, "{:?}", t.elapsed());
    // The mpmath values to 20 digits, as the nearest doubles.
    assert!((c / 0.999_710_042_045_790_5 - 1.0).abs() < 1e-15, "{c}");
    assert!((s / 2.899_579_542_094_966e-4 - 1.0).abs() < 1e-14, "{s}");
}

// ── Limits of distribution functions ─────────────────────────────────────────

/// The limit at `0⁺` of the CDF of a mixture of `Weibull(530, N)`,
/// `N = 34278473728`, with a triangular law aborted the process (0.56): the
/// series of `exp(−x^N)` carried the valuation `N` into the dense
/// coefficient vector of `TSeries::compose`, a 137 GB allocation.  Oracle:
/// `exp(−x^N) = 1 + O(x^N)`, so `(1 − exp(−x^N))/x → 0`,
/// `(1 − exp(−x^N))·ln x → 0`, and the mixture's CDF is `0` at `0⁺`
/// (both components vanish there) and `1` at `+∞`.
#[test]
fn limits_with_a_huge_integer_power_do_not_abort() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let n = ctx.int(34_278_473_728);
    let e = (-x.pow(&n)).exp();
    let zero = ctx.zero();
    assert_eq!(((ctx.one() - &e) / &x).limit_right(&x, &zero), zero);
    assert_eq!(((ctx.one() - &e) * x.ln()).limit_right(&x, &zero), zero);
    let w = Distribution::weibull(ctx.int(530), n.clone());
    let r3 = ctx.int(3).sqrt();
    let lo = ctx.rational(25, 9);
    let tri = Distribution::triangular(lo.clone(), &r3 * 3 + &lo, &r3 * ctx.rational(18, 13) + &lo);
    let m = Distribution::mixture(&[(ctx.rational(1, 9), w), (ctx.rational(8, 9), tri)]).unwrap();
    let f = m.family().cdf(&x).unwrap();
    assert_eq!(f.limit_right(&x, &zero), zero);
    assert_eq!(f.limit(&x, &ctx.infinity()), ctx.one());
}

/// The characteristic function of `FDistribution(46517977088, 51087)` at a
/// real `s` aborted the process (0.56): the breakpoint scan of its integral
/// asked `solve` for the zeros of a term `x^23258988544`, whose coefficient
/// list was sized by the degree (a 1.1 TB vector).  A polynomial beyond
/// degree `2¹⁶` is no longer listed densely.  Oracle: the call returns (a
/// value with `|cf| ≤ 1`, or a refusal).
#[test]
fn characteristic_function_with_a_huge_shape_returns() {
    let ctx = Context::new();
    let f = Distribution::f_distribution(ctx.int(46_517_977_088), ctx.int(51087));
    let s = ctx.rational(4_498_272_390_528_213, 281_474_976_710_656);
    if let Ok(z) = f.characteristic_function(&s).eval_complex64() {
        assert!(z.norm() <= 1.0 + 1e-12, "{z}");
    }
}

/// `quantile_f64`/`isf_f64` of every `transformed` law refused (0.56: "the
/// parameter map is symbolic" — the map is an expression in its own bound
/// variable, not a parameter).  Oracle: for `Y = ln X`, `X ~ Exp(1)`,
/// `P(Y ≤ y) = 1 − exp(−e^y)`, so the quantile at `1/4` is
/// `ln(−ln(3/4)) = −1.2458993237072382` and the upper quantile at `10⁻¹⁰` is
/// `ln(ln 10¹⁰) = 3.1366175382420014` (mpmath, 30 digits).
#[test]
fn transformed_quantiles_evaluate() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = Distribution::exponential(ctx.one())
        .transformed(&x, &x.ln())
        .unwrap();
    let q = y.quantile_f64(0.25).unwrap();
    assert!((q / -1.245_899_323_707_238_2 - 1.0).abs() < 1e-14, "{q}");
    let u = y.isf_f64(1e-10).unwrap();
    assert!((u / 3.136_617_538_242_001_4 - 1.0).abs() < 1e-14, "{u}");
    // A symbolic parameter of the inner law is still refused.
    let rate = ctx.symbol_with("lam", &[Assumption::Positive]).unwrap();
    let z = Distribution::exponential(rate)
        .transformed(&x, &x.ln())
        .unwrap();
    assert!(z.quantile_f64(0.5).is_err());
}

/// `Hypergeometric(0, 0, 0)` (the empty population: the point mass at 0)
/// had mean `0·0/0 = NaN` (0.56), and a population of one a NaN variance
/// (`(N − n)/(N − 1) = 0/0`) and NaN factorial moments past `N`.  Oracle:
/// a point mass has mean its value, variance 0, `E[X²]` its square.
#[test]
fn degenerate_hypergeometric_has_finite_moments() {
    let ctx = Context::new();
    let empty = Distribution::try_hypergeometric(ctx.zero(), ctx.zero(), ctx.zero()).unwrap();
    assert_eq!(empty.mean(), ctx.zero());
    assert_eq!(empty.variance(), ctx.zero());
    let one = Distribution::try_hypergeometric(ctx.one(), ctx.one(), ctx.one()).unwrap();
    assert_eq!(one.mean(), ctx.one());
    assert_eq!(one.variance(), ctx.zero());
    assert_eq!(one.moment(2), ctx.one());
    assert_eq!(one.moment(3), ctx.one());
    // An ordinary one is unchanged: n m / N = 5·4/10.
    let h = Distribution::try_hypergeometric(ctx.int(10), ctx.int(4), ctx.int(5)).unwrap();
    assert_eq!(h.mean(), ctx.int(2));
}

/// Two integer overflows panicked (with overflow checks, 0.56): `evalf` of
/// `2^(−2³¹)` negated the exponent `i32::MIN` (the tail of
/// `Exponential(ln 2)` at `2³¹`), and the display order of
/// `x^2678128640·y^2678128640` summed the degrees in a `u32`.  Oracle:
/// `S(2³¹) = 2^(−2³¹)`, far below the smallest double (`0.0`), `F = 1`.
#[test]
fn huge_integer_exponents_do_not_overflow() {
    let ctx = Context::new();
    let e = Distribution::exponential(ctx.int(2).ln());
    let at = ctx.int(2).powi(31);
    if let Ok(s) = e.sf(&at).eval_f64() {
        assert_eq!(s, 0.0);
    }
    if let Ok(c) = e.cdf(&at).eval_f64() {
        assert_eq!(c, 1.0);
    }
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let n = ctx.int(2_678_128_640);
    let p = x.pow(&n) * y.pow(&n) + &x;
    assert!(!format!("{p}").is_empty());
}

// ── Transformed tables ───────────────────────────────────────────────────────────────

/// `g(X)` of a lattice law on many values exhausted memory (0.56):
/// `transformed` compared every new image with every entry by `equals`,
/// each comparison interning a difference, and the table's CDF looked each
/// value's mass up through `density`, which built the whole table as a
/// `Piecewise` per call (`X⁵` of `Hypergeometric(50000, 19531, 30468)`
/// exhausted 2 GB in either step).  Rational images are now merged through
/// a map and a table's mass at a number is a direct lookup.  Oracle:
/// `P(X³ ≤ 1600³) = P(X ≤ 1600) = 4/5` for `X` uniform on `1..=2000`.
///
/// Also: `−X` of a mixture of a lattice law and a table took
/// `P(X ≥ x) = P(X > ⌈x⌉ − 1)`, true on the integers only: with table atoms
/// `9/8`, `5/4`, `P(−X ≤ −5/4)` counted the atom `9/8` (`5/8`, exactly
/// `3/8 = ½·¼ + ½·½`, the binomial's `P(X = 2)` and the table's `P(X = 5/4)`).
#[test]
fn transformed_large_table_merges_without_quadratic_memory() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let u = Distribution::discrete_uniform(ctx.one(), ctx.int(2000));
    let t = std::time::Instant::now();
    let y = u.transformed(&x, &x.powi(3)).unwrap();
    let at = ctx.int(1600).powi(3);
    assert_eq!(y.cdf(&at), ctx.rational(4, 5));
    assert_eq!(y.sf(&at), ctx.rational(1, 5));
    assert_eq!(y.density(&at), ctx.rational(1, 2000));
    assert_eq!(y.density(&(at + 1)), ctx.zero());
    assert!(t.elapsed().as_secs_f64() < 20.0, "{:?}", t.elapsed());
    let half = ctx.rational(1, 2);
    let table = Distribution::try_finite(
        &ctx,
        vec![
            (ctx.rational(9, 8), half.clone()),
            (ctx.rational(5, 4), half.clone()),
        ],
    )
    .unwrap();
    let lattice = Distribution::binomial(ctx.int(2), half.clone());
    let mix = Distribution::mixture(&[(half.clone(), lattice), (half.clone(), table)]).unwrap();
    let image = mix.affine(-ctx.one(), ctx.zero()).unwrap();
    assert_eq!(image.cdf(&ctx.rational(-5, 4)), ctx.rational(3, 8));
    assert_eq!(image.sf(&ctx.rational(-5, 4)), ctx.rational(5, 8));
}

// ── Quantiles ──────────────────────────────────────────────────────────────────────

/// `NegativeBinomial(994, 1/10).isf_f64(1/4)` took 7.5 s (0.56): the
/// starting guess walked the compiled mass function, which is NaN there
/// (`(1/10)^994` underflows, the binomial coefficient overflows), and fell
/// back to the exact mass at each of 9,146 atoms before the exact search
/// began.  A guess now walks only where the compiled function has values.
/// Oracle: mpmath `betainc(k + 1, 994, 0, 9/10, regularized=True)` at 30
/// and 60 digits: `S(9145) = 0.25049905820616046778 > 1/4 ≥ S(9146) =
/// 0.24945146510951962361`.
#[test]
fn negative_binomial_upper_quantile_is_a_fast_exact_search() {
    let ctx = Context::new();
    let nb = Distribution::negative_binomial(ctx.int(994), ctx.rational(1, 10));
    let t = std::time::Instant::now();
    assert_eq!(nb.isf_f64(0.25).unwrap(), 9146.0);
    assert!(t.elapsed().as_secs_f64() < 2.0, "{:?}", t.elapsed());
}
