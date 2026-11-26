//! Tests of session 18, second block of 0.60.1. Lane BETA: the regularized incomplete beta
//! function at extreme parameters.
//! Every test says what was wrong before and cites its oracle.
//!
//! The oracle of the values at huge shapes is `oracle_quad.py` of the lane's notes: mpmath's
//! `quad` of the definition `∫ t^(a−1) (1−t)^(b−1) dt` in the variable `u = (t − t₀)/σ` about
//! the mode `t₀`, the integrand `exp((a−1)·log1p(σu/t₀) + (b−1)·log1p(−σu/(1−t₀)))` evaluated at
//! 1200 digits, the two sides of `x` integrated at 30 and at 45 digits: `I_x = L/(L + R)`.

use std::time::Duration;

use symplex::prelude::*;
use symplex::stats::Distribution;

use crate::perf_assert::assert_fast;

fn num(ctx: &Context, src: &str) -> Ex {
    ctx.parse(src).unwrap()
}

fn reg_beta(ctx: &Context, a: &str, b: &str, x: &str) -> Result<f64, SymplexError> {
    num(ctx, &format!("betainc_regularized({a}, {b}, 0, {x})")).eval_f64()
}

/// Class E of the `fuzz_stats` campaign.  `FDistribution(4·10¹³⁶, 4·10²⁵⁶).cdf(1).eval_f64()`
/// was `Err("continued fraction for the incomplete beta function did not converge in 2014160
/// steps")` after 18.5 s, and forming `cdf(1)` took 62 s, `sf(1)` 55.8 s (three such
/// evaluations): the variance of the beta law, `ab/(a + b)³ = 5·10⁻³⁷⁷`, is 0 as an `f64`, so
/// the estimate of the fraction's steps read 0 where it is `10⁴⁷`, and the route about the
/// mode, which holds this case, declined.  Oracle: `oracle_quad.py 2*10^136 2*10^256
/// "Fr(4*10^136, 4*10^136+4*10^256)"` = `0.5` at 30 and at 45 digits (`x` lies `7.1·10⁻⁶⁹`
/// standard deviations above the mode; the excess over ½ is `1/(3·√(2πa)) = 9.4·10⁻⁷⁰`, the
/// skewness term of the Edgeworth expansion).
#[test]
fn the_f_law_of_huge_degrees_of_freedom_answers_at_its_centre() {
    let ctx = Context::new();
    let f = Distribution::f_distribution(num(&ctx, "4*10^136"), num(&ctx, "4*10^256"));
    let one = ctx.one();
    let cdf = assert_fast!(Duration::from_secs(2), f.cdf(&one));
    assert_eq!(
        assert_fast!(Duration::from_secs(2), cdf.eval_f64()).unwrap(),
        0.5
    );
    let sf = assert_fast!(Duration::from_secs(2), f.sf(&one));
    assert_eq!(
        assert_fast!(Duration::from_secs(2), sf.eval_f64()).unwrap(),
        0.5
    );
}

/// The same cause inside the first shape's envelope: `FDistribution(10¹⁵, 10²⁵⁰).cdf(1)` took
/// 10.6 s (860 000 steps of the continued fraction; the value was right), its `sf(1)` and
/// `betainc_regularized(10¹⁵, 10²⁵⁰, 0, x)` at the mean `x = 10¹⁵/(10¹⁵ + 10²⁵⁰)` more than
/// 12 s.  Oracle: `oracle_quad.py 5*10^14 5*10^249 "Fr(10^15, 10^15+10^250)"` =
/// `0.50000000594708038717590378`, and `oracle_quad.py 10^15 10^250 "Fr(10^15, 10^15+10^250)"`
/// = `0.5000000042052208700336001` (30 digits; the two tails add up to 1).
#[test]
fn a_second_shape_beyond_the_range_of_the_variance_takes_the_central_route() {
    let ctx = Context::new();
    let f = Distribution::f_distribution(num(&ctx, "10^15"), num(&ctx, "10^250"));
    let one = ctx.one();
    let cdf = assert_fast!(Duration::from_secs(2), f.cdf(&one).eval_f64()).unwrap();
    let sf = assert_fast!(Duration::from_secs(2), f.sf(&one).eval_f64()).unwrap();
    assert_eq!(cdf, 0.500_000_005_947_080_4);
    assert_eq!(sf, 0.499_999_994_052_919_6);
    let at_mean = assert_fast!(
        Duration::from_secs(2),
        reg_beta(&ctx, "10^15", "10^250", "10^15/(10^15 + 10^250)")
    );
    assert_eq!(at_mean.unwrap(), 0.500_000_004_205_220_8);
}

/// A continued fraction beyond its budget is refused before its first step.
/// `betainc_regularized(10²⁵⁰, 10¹⁴, 0, x)` at the mean `x = 10²⁵⁰/(10²⁵⁰ + 10¹⁴)` ran
/// 2 050 000 steps at 2060 bits, more than 40 s, to answer "did not converge": `x` lies `10⁻²³⁶` below
/// 1, the evaluation needs 1200 bits, the route about the mode does not reach them, and the
/// estimate from below of the fraction's steps, `1.7·10⁶`, was inside the budget in steps.
/// Beyond the envelope (`a + b > 2⁵⁴`) the budget is now `2²⁵` bit-steps, 16 000 steps here.
/// No oracle: the call refuses (the value is `0.49999…`, about `1.3·10⁻⁸` below ½).
#[test]
fn a_fraction_beyond_its_work_budget_is_refused_at_once() {
    let ctx = Context::new();
    let r = assert_fast!(
        Duration::from_secs(2),
        reg_beta(&ctx, "10^250", "10^14", "10^250/(10^250 + 10^14)")
    );
    assert!(r.is_err(), "{r:?}");
}

/// Inside the envelope nothing is refused that was answered: the symmetric cases at the
/// largest shapes, which the route about the mode holds, and a fraction of a few steps.
/// Oracles: `I_{1/2}(a, a) = 1/2` exactly (the symmetry `I_x(a, b) = 1 − I_{1−x}(b, a)`);
/// mpmath `betainc(1/3, 9*10**15, 0, mpf(10)**-16, regularized=True)` at 30 and 60 digits =
/// `0.889327385905374193523995397289` (the same 30 digits at both).
#[test]
fn the_largest_shapes_inside_the_envelope_still_answer() {
    let ctx = Context::new();
    for a in ["10^8", "10^15", "9007199254740992"] {
        let v = assert_fast!(Duration::from_secs(2), reg_beta(&ctx, a, a, "1/2")).unwrap();
        assert_eq!(v, 0.5, "I_(1/2)({a}, {a})");
    }
    let v = reg_beta(&ctx, "1/3", "9*10^15", "1/10^16").unwrap();
    assert!((v / 0.889_327_385_905_374_2 - 1.0).abs() < 4e-16, "{v:e}");
}

/// Beyond the envelope the route about the mode is tried before the continued fraction,
/// whatever the estimate of the fraction's steps says (it is calibrated for `b/a` between
/// `10⁻⁶` and `10²`).  `betainc_regularized(10²⁵⁰, 10¹¹⁰, 0, x)` three standard deviations below
/// the mode, `x = 1 − (10¹¹⁰ + 3·10⁵⁵)/10²⁵⁰`, took 0.45 s by the fraction (the value was
/// right); the route about the mode takes 0.05 s.  Oracle: `oracle_quad.py 10^250 10^110
/// "Fr(10^250-10^110-3*10^55, 10^250)"` = `0.0013498980316300945266518148` at 30 digits,
/// `…8147675949773778294` at 45 (`Φ(−3) = 0.00134989803163009452665…`: the law is normal to
/// `10⁻⁵⁵` here).
#[test]
fn beyond_the_envelope_the_route_about_the_mode_is_tried_first() {
    let ctx = Context::new();
    let v = assert_fast!(
        Duration::from_secs(5),
        reg_beta(&ctx, "10^250", "10^110", "1 - (10^110 + 3*10^55)/10^250")
    );
    assert_eq!(v.unwrap(), 1.349_898_031_630_094_6e-3);
}
