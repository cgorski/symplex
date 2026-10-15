//! Tests of session 15 (s15g3): quadrature tails and the stats refusals.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;
use symplex::stats::Distribution;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

fn is_divergent<T: std::fmt::Debug>(r: &Result<T, SymplexError>) -> bool {
    matches!(r, Err(SymplexError::Divergent { .. }))
}

/// `∫₀^∞ e^{x − 10¹⁰⁰⁰} dx` diverges (the integrand grows without bound
/// beyond `x = 10¹⁰⁰⁰`; `x·f(x) → ∞`), although it is below `2^(−10³⁰⁰⁰)`
/// out to `x = 2²⁰⁴⁸`, as far as the far-tail samples of an `f64`
/// quadrature reach, and at every node `x ≤ 10⁸` where the
/// arbitrary-precision walk used to stop.  Up to 0.52 every route said
/// `Divergent` only because `exp` of an argument known to `±2⁹` or worse
/// had no error bound at all; once such a ball far below the exponent
/// range gets one (`e^(Re z + r)`), the samples alone would have trusted
/// the `f64` value 0 (an attempt in wave 7 did; a draft of this change gave
/// `eval_f64` = 0 from a quadrature whose nodes all underflowed).
#[test]
fn growth_beyond_every_sample_is_divergent() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = parse(&ctx, "exp(x - 10^1000)");
    let (zero, oo) = (ctx.int(0), ctx.infinity());
    assert!(is_divergent(&f.integrate_numeric(&x, &zero, &oo)));
    let node = f.definite_integral_node(&x, &zero, &oo);
    assert!(is_divergent(&node.eval_f64()), "{:?}", node.eval_f64());
    assert!(is_divergent(&node.eval_decimal(15)));
}

/// `∫₀^∞ e^{−x − 10¹⁰⁰⁰} dx = e^{−10¹⁰⁰⁰}` (exact), far below every
/// representable number: its tiny value (the `f64` it rounds to, `+0.0`)
/// or a refusal, never anything else.
#[test]
fn decay_far_below_the_range_is_tiny_or_refused() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = parse(&ctx, "exp(-x - 10^1000)");
    let (zero, oo) = (ctx.int(0), ctx.infinity());
    if let Ok(v) = f.integrate_numeric(&x, &zero, &oo) {
        assert!((0.0..1e-300).contains(&v), "{v}");
    }
    let node = f.definite_integral_node(&x, &zero, &oo);
    if let Ok(v) = node.eval_f64() {
        assert_eq!(v.to_bits(), 0.0f64.to_bits());
    }
    assert!(node.eval_decimal(15).is_err());
}

/// `∫₁^∞ x⁻²e^{x/10⁷⁰⁰} dx` diverges (`x·f(x) = e^{x/10⁷⁰⁰}/x → ∞`): near
/// every sample and node it looks like `x⁻²`.  Up to 0.52 `integrate_numeric`
/// and `eval_f64` gave `1.0` and `eval_decimal(30)` gave `"1"`.
#[test]
fn growth_set_off_by_a_large_constant_is_divergent() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = parse(&ctx, "x^(-2)*exp(x/10^700)");
    let (one, oo) = (ctx.int(1), ctx.infinity());
    assert!(is_divergent(&f.integrate_numeric(&x, &one, &oo)));
    let node = f.definite_integral_node(&x, &one, &oo);
    assert!(is_divergent(&node.eval_f64()));
    assert!(is_divergent(&node.eval_decimal(30)));
}

/// `∫₀^∞ e^{−x}·e^{x²/10¹⁴⁰⁰ − 10¹⁰} dx` diverges (the exponent `x²/10¹⁴⁰⁰ −
/// x − 10¹⁰ → ∞`); it underflows at every sample out to `2²⁰⁴⁸`.  Up to
/// 0.52 `integrate_numeric` and `eval_f64` gave `0.0`, and without the
/// `10¹⁰` (no underflow) `eval_decimal(30)` gave `"1"`.
#[test]
fn growth_hidden_by_underflow_is_divergent() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (zero, oo) = (ctx.int(0), ctx.infinity());
    let f = parse(&ctx, "exp(-x)*exp(x^2/10^1400 - 10^10)");
    assert!(is_divergent(&f.integrate_numeric(&x, &zero, &oo)));
    let node = f.definite_integral_node(&x, &zero, &oo);
    assert!(is_divergent(&node.eval_f64()));
    let g = parse(&ctx, "exp(-x)*exp(x^2/10^1400)");
    let node = g.definite_integral_node(&x, &zero, &oo);
    assert!(is_divergent(&node.eval_decimal(30)));
}

/// `∫₀¹ sin(10⁶x)·e^{−10⁸(x − ½)²} dx = sin(5·10⁵)·√π·10⁻⁴·e^{−2500}` up to
/// `e^{−2.5·10⁷}` (the Fourier transform of the Gaussian, `∫ e^{iωx −
/// a(x − c)²} dx = e^{iωc}√(π/a)·e^{−ω²/(4a)}`), below `10⁻¹⁰⁸⁹`.  Up to 0.52
/// `eval_f64` gave `−1.9053090980283763·10⁻²⁸` from a quadrature over
/// 159,155 periods that certified a bound of `2⁻¹³⁴`: a piece over which
/// the oscillation cannot be resolved by any bisection now refuses at once.
#[test]
fn unresolvable_oscillation_is_refused_not_misquoted() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = parse(&ctx, "sin(10^6*x)*exp(-10^8*(x - 1/2)^2)");
    let node = f.definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    if let Ok(v) = node.eval_f64() {
        assert!(v.abs() < 1e-300, "{v}");
    }
}

/// `∫₀^∞ cos(5·10⁷x)·e^{−x} dx = 1/(1 + 2.5·10¹⁵)` (exact) `=
/// 3.99999999999999840…·10⁻¹⁶`.  Up to 0.52 the quadrature bisected
/// `[0, 1]` (8·10⁶ periods) until its evaluations ran out and the value was
/// refused; it now gives up at once and the closed form answers.
#[test]
fn oscillation_beyond_resolution_leaves_it_to_the_closed_form() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = parse(&ctx, "cos(5*10^7*x)*exp(-x)");
    let node = f.definite_integral_node(&x, &ctx.int(0), &ctx.infinity());
    assert_eq!(node.eval_decimal(15).unwrap(), "4e-16");
}

/// `Weibull(6·10⁻⁶, 31/2)`: `E[e^{tX}]` at `t = 50000/3` and `E[e^{itX}]`.
/// With `X = λU`, `U ~ Weibull(1, k)`: `∫₀^∞ e^{tλu} k u^{k−1} e^{−u^k} du`
/// (mpmath `quad` split at `½, 1, 3/2, 2, 4`, 30 and 60 digits agree):
/// `1.10152349390803129641522` and `0.9953023726429213829843917 +
/// 0.09651169109748644731815938i`.  Answered at 0.48; refused since ("the
/// integrand has no error bound at a node (x ≈ 5e-1)": the exponent there is
/// `−10⁷⁸ ± 2⁹³`, a ball that `exp` gave no bound although every value in
/// it underflows).
#[test]
fn weibull_mgf_with_a_narrow_peak() {
    let ctx = Context::new();
    let w = Distribution::weibull(ctx.rational(6, 1_000_000), ctx.rational(31, 2));
    let m = w.mgf(&ctx.rational(50000, 3)).eval_f64().unwrap();
    assert!((m - 1.101_523_493_908_031_3).abs() < 1e-14, "{m}");
}

/// The characteristic function of the same distribution at the same `t`
/// (see [`weibull_mgf_with_a_narrow_peak`]).
#[test]
fn weibull_cf_with_a_narrow_peak() {
    let ctx = Context::new();
    let w = Distribution::weibull(ctx.rational(6, 1_000_000), ctx.rational(31, 2));
    let c = w
        .characteristic_function(&ctx.rational(50000, 3))
        .eval_complex64()
        .unwrap();
    assert!((c.re - 0.995_302_372_642_921_4).abs() < 1e-14, "{c:?}");
    assert!((c.im - 0.096_511_691_097_486_45).abs() < 1e-14, "{c:?}");
}

/// `Weibull(8·10⁶/7, 15/2)`: `E[e^{tX}]` at `t = −7/(4·10⁷)` (`tλ = −1/5`)
/// = `0.8291875054655020684343922` (mpmath as above, 30 and 60 digits).
/// Answered at 0.48; `PrecisionExhausted` since.
#[test]
fn weibull_mgf_at_a_negative_argument() {
    let ctx = Context::new();
    let w = Distribution::weibull(ctx.rational(8_000_000, 7), ctx.rational(15, 2));
    let m = w.mgf(&ctx.rational(-7, 40_000_000)).eval_f64().unwrap();
    assert!((m - 0.829_187_505_465_502_1).abs() < 1e-14, "{m}");
}

/// `NegativeBinomial(9, 10⁻¹⁸)`: `P(X ≤ k) = I_p(9, k + 1)`.  The quantiles
/// at `½`, `1/100` and `99/100` are the `f64` atoms `k` with `F(k) ≥ level >
/// F(k − u)`, `u` the `f64` spacing there (mpmath `betainc(9, k + 1, 0, p,
/// regularized=True)` at 50 and 80 digits: `F(k) = 0.50000000000000006323`,
/// `F(k − 1024) = 0.49999999999999992403`; `0.010000000000000007658`,
/// `0.0099999999999999989396`; `0.98999999999999999204`,
/// `0.98999999999999998021`).  `P(X > 8668951184370373000)` =
/// `0.499999999999999883487440935318` (`Σ_{j<9} C(k+9, j) pʲ(1−p)^(k+9−j)` at
/// 40 and 80 digits, and mpmath's `betainc` both ways).  Answered at 0.48;
/// `PrecisionExhausted` since: the bound on `B(8.7·10¹⁸, 9)` in the
/// sensitivity of `betainc_regularized` was `2^(±2.2·10⁷)`.
#[test]
fn negative_binomial_quantiles_at_a_tiny_success_probability() {
    let ctx = Context::new();
    let nb =
        Distribution::negative_binomial(ctx.int(9), ctx.rational(1, 1_000_000_000_000_000_000));
    assert_eq!(nb.quantile_f64(0.5).unwrap(), 8.668951184370373e18);
    assert_eq!(nb.isf_f64(0.5).unwrap(), 8.668951184370373e18);
    assert_eq!(nb.quantile_f64(0.01).unwrap(), 3.507_455_450_586_290_7e18);
    assert_eq!(nb.quantile_f64(0.99).unwrap(), 1.740_265_286_735_253_5e19);
    let s = nb
        .sf(&parse(&ctx, "8668951184370373000"))
        .eval_decimal(20)
        .unwrap();
    assert_eq!(s, "0.49999999999999988349");
}

/// `Weibull(4·10⁻⁹, 1/200)` at `t = −5·10⁷`: the real part of its
/// characteristic function, `∫₀^∞ cos(tx) f(x) dx`, oscillates `8·10⁶` times
/// over `[0, 1]` beyond a heavy tail; the quadrature cannot resolve it.
/// Refused at 0.48; from 0.49 on it bisected until 30,000 evaluations ran
/// out (11 s in a release build for the two parts, beyond the sweep's 10 s
/// limit: case 1552 of its C14 hunt); now refused after one attempt on the
/// piece.
#[test]
fn weibull_cf_with_an_unresolvable_oscillation_is_refused_promptly() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = parse(
        &ctx,
        "cos(-50000000*x)*1250000*(250000000*x)^(-199/200)*exp(-(250000000*x)^(1/200))",
    );
    let node = f.definite_integral_node(&x, &ctx.int(0), &ctx.infinity());
    assert!(node.eval_f64().is_err());
}
