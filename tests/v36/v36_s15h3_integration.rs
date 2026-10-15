//! Tests of session 15 (s15h3): numeric integration: the remaining wrong value and refusals.
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

/// `∫₀^∞ e^{g(x)} dx` with `g = 2x^(1/1000) − x^(1/999) − 10⁵`: `g′ = 0` at
/// `x^(1/999000) = 1.998` (exact), `x = 1.998^999000 ≈ 2^997558`, where `g =
/// 0.002·1.998^999 − 10⁵ ≈ 3.94·10²⁹⁷` (mpmath, 30 digits): the integral
/// exceeds `e^{10²⁹⁷}`, beyond every representable number.  At every point
/// the far-tail samples of the `f64` quadrature reach (`2²⁰⁴⁸`) `g ≈ −10⁵`,
/// and up to 0.53 `integrate_numeric` gave `Ok(0.0)` (an estimate 0 with an
/// error of 0): the terms `2x^(1/1000)` and `x^(1/999)` cross over near
/// `2⁹⁹⁹⁰⁰⁰`, beyond every sample and every node.
#[test]
fn bump_of_close_powers_beyond_every_sample_is_refused() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = parse(&ctx, "exp(2*x^(1/1000) - x^(1/999) - 100000)");
    let (zero, oo) = (ctx.int(0), ctx.infinity());
    assert!(f.integrate_numeric(&x, &zero, &oo).is_err());
    let node = f.definite_integral_node(&x, &zero, &oo);
    assert!(node.eval_f64().is_err());
    assert!(node.eval_decimal(15).is_err());
}

/// `∫₁^∞ x⁻²e^{x^(1/1000)} dx` diverges: `x·f(x) = e^{x^(1/1000)}/x → ∞`
/// (`x^(1/1000) > ln x` from `x ≈ 2^13000` on).  Out to there it decays as
/// `x⁻²`, and up to 0.53 `integrate_numeric`, `eval_f64` and `eval_decimal`
/// gave `2.7210055604834866`: the arbitrary-precision walk stopped at `x ≈
/// 2²⁷`, the scale of the constants alone.
#[test]
fn divergence_set_in_by_a_small_power_is_found() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (one, oo) = (ctx.int(1), ctx.infinity());
    for s in ["x^(-2)*exp(x^(1/1000))", "x^(-2)*exp(x^(1/100))"] {
        let f = parse(&ctx, s);
        assert!(is_divergent(&f.integrate_numeric(&x, &one, &oo)), "{s}");
        let node = f.definite_integral_node(&x, &one, &oo);
        assert!(is_divergent(&node.eval_f64()), "{s}");
        assert!(is_divergent(&node.eval_decimal(15)), "{s}");
    }
}

/// `∫₁^∞ x⁻²e^{5x^(1/100)} dx` diverges too, although `5x^(1/100)` is the
/// larger term from `x = 1` on: `2·ln x` overtakes it and decays the
/// integrand out to `x ≈ 2⁵³²`, where their rates meet, and only beyond
/// `2⁷⁸⁰` does the integrand climb back above its value at 1.  Up to 0.53
/// every route gave `156.3119991254238`.
#[test]
fn divergence_after_a_logarithm_overtakes_is_found() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (one, oo) = (ctx.int(1), ctx.infinity());
    let f = parse(&ctx, "x^(-2)*exp(5*x^(1/100))");
    assert!(is_divergent(&f.integrate_numeric(&x, &one, &oo)));
    let node = f.definite_integral_node(&x, &one, &oo);
    assert!(is_divergent(&node.eval_f64()));
}

/// `∫₀^∞ e^{2x^(1/30) − x^(1/28) − 500} dx = 2.6760439366876599747·10²⁵⁸`
/// (`x = e^t`: mpmath `quad` around the maximum of the exponent at `x =
/// (28/15)^420 ≈ 2³⁷⁸`, 30 and 40 digits): the integrand is below `e^{−470}`
/// at the far-tail samples `2²⁵⁶` and `2⁵¹²`, and up to 0.53
/// `integrate_numeric` gave `1.6108648532346576·10⁻²¹⁴`.  Its value or a
/// refusal now.
#[test]
fn bump_between_the_far_tail_samples_is_seen() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = parse(&ctx, "exp(2*x^(1/30) - x^(1/28) - 500)");
    if let Ok(v) = f.integrate_numeric(&x, &ctx.int(0), &ctx.infinity()) {
        assert!((v / 2.676_043_936_687_66e258 - 1.0).abs() < 1e-8, "{v}");
    }
}

/// The bump of close powers within reach is still integrated:
/// `∫₀^∞ e^{2x^(1/10) − x^(1/9) − 50} dx = 26125694335607661351.05` (mpmath
/// `quad` split around the maximum at `x = 1.8⁹⁰ ≈ 9.43·10²²`, 30 and 60
/// digits; unchanged since 0.53).
#[test]
fn bump_of_close_powers_within_reach_is_integrated() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = parse(&ctx, "exp(2*x^(1/10) - x^(1/9) - 50)");
    let (zero, oo) = (ctx.int(0), ctx.infinity());
    let v = f.integrate_numeric(&x, &zero, &oo).unwrap();
    assert!((v / 2.612_569_433_560_766e19 - 1.0).abs() < 1e-10, "{v}");
}

/// An `f64` run whose every node gives 0 proves nothing.  `∫₀^(10³⁰⁰)
/// e^{−1000} dx = 10³⁰⁰e^{−1000} = 5.075958897549456765·10⁻¹³⁵` and
/// `∫₀^(10¹⁷⁰) e^{−800}x dx = 10³⁴⁰e^{−800}/2 = 1.833937292088843607·10⁻⁸`
/// (exact; mpmath, 40 digits) were `Ok(0.0)` from `integrate_numeric` up to
/// 0.53 (`eval_f64` of the same integrals gave the values), and `∫₀^(10³⁰⁰)
/// e^{−1000}(2 + cos(x/10²⁹⁰)) dx = e^{−1000}(2·10³⁰⁰ + 10²⁹⁰·sin(10¹⁰)) =
/// 1.015191779485145748·10⁻¹³⁴` too.  `∫₀¹ e^{−x² − 1000} dx ≈ 3.8·10⁻⁴³⁵`
/// rounds to `0.0`, as before.
#[test]
fn zero_nodes_are_no_proof_of_a_zero_integral() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let cases = [
        ("exp(-1000)", "10^300", 5.075_958_897_549_456_6e-135),
        ("exp(-800)*x", "10^170", 1.833_937_292_088_843_6e-8),
        (
            "exp(-1000)*(2 + cos(x/10^290))",
            "10^300",
            1.015_191_779_485_145_7e-134,
        ),
    ];
    for (f, hi, want) in cases {
        let v = parse(&ctx, f)
            .integrate_numeric(&x, &zero, &parse(&ctx, hi))
            .unwrap();
        assert!((v / want - 1.0).abs() < 1e-12, "{f}: {v}");
    }
    let tiny = parse(&ctx, "exp(-x^2 - 1000)").integrate_numeric(&x, &zero, &ctx.int(1));
    assert_eq!(tiny.unwrap().to_bits(), 0.0f64.to_bits());
}

/// `Weibull(6·10⁻⁶, 31/2)`: `E[e^{tX}]` at `t = 50000/3` (`tλ = 1/10`) is
/// `Σₙ (tλ)ⁿΓ(1 + n/k)/n!` = `1.1015234939080312859640017159444` (mpmath, 40
/// and 80 digits; `quad` split at 41 points around the peak agrees), and
/// `E[e^{itX}]` = `0.99530237264292138390526852093838 +
/// 0.096511691097486437881013860174156i`.  The density's mass lies in a peak
/// `4·10⁻⁷` wide at `6·10⁻⁶`, next to the end of `[0, ∞)`; up to 0.53
/// `eval_decimal` (16 and 30 digits) of both was `PrecisionExhausted` (only
/// `eval_f64` answered).  (The wave-8 oracle, mpmath `quad` split at 5
/// points, was off in the 17th digit.)
#[test]
fn weibull_mgf_and_cf_with_a_narrow_peak_to_30_digits() {
    let ctx = Context::new();
    let w = Distribution::weibull(ctx.rational(6, 1_000_000), ctx.rational(31, 2));
    let t = ctx.rational(50000, 3);
    assert_eq!(w.mgf(&t).eval_decimal(16).unwrap(), "1.101523493908031");
    assert_eq!(
        w.mgf(&t).eval_decimal(30).unwrap(),
        "1.10152349390803128596400171594"
    );
    assert_eq!(
        w.characteristic_function(&t).eval_decimal(30).unwrap(),
        "0.995302372642921383905268520938 + 0.0965116910974864378810138601742*i"
    );
}

/// `Weibull(8·10⁶/7, 15/2)` at `t = −7/(4·10⁷)` (`tλ = −1/5`):
/// `E[e^{itX}]` = `0.98199665813962589796625379880957 −
/// 0.18656816299727323628834989253723i` and `E[e^{tX}]` =
/// `0.8291875054655020749514337465804` (the moment series as above, 40 and 80
/// digits).  Up to 0.53 `eval_complex64` of the characteristic function and
/// `eval_decimal` of both were `PrecisionExhausted`.
#[test]
fn weibull_cf_with_a_peak_far_from_the_end() {
    let ctx = Context::new();
    let w = Distribution::weibull(ctx.rational(8_000_000, 7), ctx.rational(15, 2));
    let t = ctx.rational(-7, 40_000_000);
    let c = w.characteristic_function(&t).eval_complex64().unwrap();
    assert!((c.re - 0.981_996_658_139_625_9).abs() < 1e-15, "{c:?}");
    assert!((c.im + 0.186_568_162_997_273_25).abs() < 1e-15, "{c:?}");
    assert_eq!(
        w.mgf(&t).eval_decimal(30).unwrap(),
        "0.82918750546550207495143374658"
    );
}

/// `∫₀¹ ln((1 + i)x − 1 − i/2) dx`: the argument crosses the cut of `ln` at
/// `x = ½` (`u(½) = −½`), where the integrand jumps by `2πi`.  mpmath `quad`
/// split at `½` (30 and 60 digits): `−0.58099697944823198508 −
/// 0.24614311400238211447i`; for `√u`: `0.18565507606205290682 −
/// 0.1526628524020776943i`; for `1/√u`: `0.37193781538432823727 −
/// 0.14219391285902608455i`.  Up to 0.52 the first was `−2.1518 −
/// 1.8169i`, at 0.53 all three were refused ("the antiderivative is
/// discontinuous near x = 0.5"); the antiderivative is now continued across
/// the cut at the exact crossing point.
#[test]
fn crossing_of_a_cut_by_a_linear_argument_is_continued() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (zero, one) = (ctx.int(0), ctx.int(1));
    let cases = [
        (
            "log((1+I)*x - 1 - I/2)",
            "-0.58099697944823198508 - 0.24614311400238211447*i",
        ),
        (
            "sqrt((1+I)*x - 1 - I/2)",
            "0.18565507606205290682 - 0.1526628524020776943*i",
        ),
        (
            "1/sqrt((1+I)*x - 1 - I/2)",
            "0.37193781538432823727 - 0.14219391285902608455*i",
        ),
        (
            "x*log((1+I)*x - 1 - I/2)",
            "-0.38893561326401584499 + 0.48272617965822877374*i",
        ),
        (
            "log((1+I)*x - 1 - I/2)^2",
            "-6.4865302228248159446 - 1.3848726818898220555*i",
        ),
    ];
    for (f, want) in cases {
        let v = parse(&ctx, f)
            .try_integrate_definite(&x, &zero, &one)
            .unwrap_or_else(|e| panic!("{f}: {e}"));
        assert_eq!(v.eval_decimal(20).unwrap(), want, "{f}");
    }
    // The formal integral is evaluated from that closed form.
    let node = parse(&ctx, "log((1+I)*x - 1 - I/2)").definite_integral_node(&x, &zero, &one);
    let z = node.eval_complex64().unwrap();
    assert!((z.re + 0.580_996_979_448_232).abs() < 1e-14, "{z:?}");
    assert!((z.im + 0.246_143_114_002_382_12).abs() < 1e-14, "{z:?}");
}

/// `∫₀² e^{(1+i)x}/(x − 1 + 10⁻¹⁴i) dx`: the antiderivative
/// `e^{(1+i)c}·Ei((1 + i)(x − c))`, `c = 1 − 10⁻¹⁴i`, has its argument cross
/// the cut of `Ei` at `x = 1 − 10⁻¹⁴`, `2·10⁻¹⁴` left of the branch point.
/// `4.7265486066168481261 + 2.6664629140770773544i` (the pole subtracted:
/// mpmath `quad` of `(e^{(1+i)x} − e^{(1+i)c})/(x − c)` plus
/// `e^{(1+i)c}·(ln(2 − c) − ln(−c))`, 30 and 60 digits).  Up to 0.52
/// `−9.6453 + 11.8945i` (the jump `2πi·e^{(1+i)c}` left in), refused at
/// 0.53.  With `10⁻³i`: `4.7272190018496913794 + 2.6623288531633174767i`.
#[test]
fn crossing_of_the_cut_of_ei_next_to_its_branch_point_is_continued() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (zero, two) = (ctx.int(0), ctx.int(2));
    let cases = [
        (
            "exp((1+I)*x)/(x - 1 + 10^-14*I)",
            "4.7265486066168481261 + 2.6664629140770773544*i",
        ),
        (
            "exp((1+I)*x)/(x - 1 + 10^-3*I)",
            "4.7272190018496913794 + 2.6623288531633174767*i",
        ),
    ];
    for (f, want) in cases {
        let v = parse(&ctx, f)
            .try_integrate_definite(&x, &zero, &two)
            .unwrap_or_else(|e| panic!("{f}: {e}"));
        assert_eq!(v.eval_decimal(20).unwrap(), want, "{f}");
    }
}
