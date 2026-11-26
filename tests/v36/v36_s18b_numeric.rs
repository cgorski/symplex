//! Tests of session 18. Lane B of the 0.60.1 block: numeric evaluation (the bound of `gamma` at large
//! arguments, `eval_decimal(0)`, `eval_f64` of a definite integral).
//! Every test says what was wrong before and cites its oracle.
//!
//! The values of the cancellation tests come from the hunts of the lane (`gen_scan.py`,
//! `check_scan.py`, `gen_gamma_scan2.py`, `check_gamma.py` of its scratch): `f(x) - M*2^E`, `M`
//! the leading 40 to 90 bits of the value, at 16 to 80 digits, against mpmath 1.3.0 at two
//! working precisions (900 and 1400 bits, or 700 and 1100) that agree to 60 digits.

#![allow(unused_imports)]
use std::time::Duration;

use crate::perf_assert::CpuTimer;
use symplex::definite::QuadRule;
use symplex::prelude::*;
use symplex::stats::Distribution;
use symplex::stats::hypothesis::*;
use symplex::units::Frequency;

/// `src` to `digits` digits.
fn dec(ctx: &Context, src: &str, digits: u32) -> String {
    ctx.parse(src).unwrap().eval_decimal(digits).unwrap()
}

// ── W01: Γ at large arguments ────────────────────────────────────────────────

/// W01, the other users of `Γ` at a large argument: `x!`, `binomial` and `beta` (below `a + b =
/// 2²⁶`) are `Γ(x + 1)` and quotients of three `Γ` values, each `exp` of a `ln Γ` summed with
/// 20 guard bits whatever its size.  Up to 0.60, of 38,480 differences `f - M*2^E` per function
/// (16 argument pairs in `[10⁴, 7·10⁷]`, 36 depths, 16 to 80 digits), 19 (`factorial`), 8
/// (`binomial`) and 5 (`beta`) were more than 0.5625 units of the last digit off:
///
/// * `binomial(88246489/4, 5341416304515333279811583/2^60) - 32410474176390935*2^16358244` at 32
///   digits printed `4.70296222663975769964213658095e4924321` (`…09500`, 1.09 units);
/// * `factorial(63294395/4) - 20930856527228921*2^355601785` at 77 digits ended in `…4981392`
///   (0.67 units: the wrong neighbour);
/// * `beta(9043177758419638655722467/2^60, 5056038714424832470331107/2^60) -
///   19308977781783747*2^-11514007` at 32 digits ended in `…3224926` (0.70 units).
///
/// Since 0.60.1 none of the 115,440 is above 0.502 (`run_scan.sh`).
///
/// Oracle: mpmath, `mp.prec = 900` and `1400` (equal to 60 digits beyond the cancellation):
/// `binomial(mpf(88246489)/4, mpf(5341416304515333279811583)/2**60) -
/// 32410474176390935*mpf(2)**16358244` = `4.702962226639757699642136580950108843e+4924321`;
/// `gamma(mpf(63294395)/4 + 1) - 20930856527228921*mpf(2)**355601785` =
/// `4.0578545554384583452992344024608708939458374287421199098387415604701794981392672e+107046801`;
/// `beta(mpf(9043177758419638655722467)/2**60, mpf(5056038714424832470331107)/2**60) -
/// 19308977781783747*mpf(2)**(-11514007)` = `8.737451634747562803086602322492529606e-3466063`.
#[test]
fn factorial_binomial_and_beta_keep_their_digits_under_a_cancellation() {
    let ctx = Context::new();
    assert_eq!(
        dec(
            &ctx,
            "binomial(88246489/4, 5341416304515333279811583/1152921504606846976) \
             - 32410474176390935*2^16358244",
            32
        ),
        "4.7029622266397576996421365809501e4924321"
    );
    assert_eq!(
        dec(
            &ctx,
            "factorial(63294395/4) - 20930856527228921*2^355601785",
            77
        ),
        "4.0578545554384583452992344024608708939458374287421199098387415604701794981393e107046801"
    );
    assert_eq!(
        dec(
            &ctx,
            "beta(9043177758419638655722467/1152921504606846976, \
             5056038714424832470331107/1152921504606846976) - 19308977781783747*2^-11514007",
            32
        ),
        "8.7374516347475628030866023224925e-3466063"
    );
}

/// W01, the side that was right and stays so: `Γ` left of 0 (the reflection `π/(sin(πd)·Γ(1 −
/// x))`, since 0.60.1 formed with 16 guard bits and rounded once) and under a cancellation
/// there.  38,480 differences over 16 arguments in `[−2·10⁶, −10³]`: no error above 0.5006
/// units before or after.
///
/// Oracle: mpmath, `mp.prec = 700` and `1100`: `gamma(mpf(-100001)/4)` =
/// `-6.485546063294694775919440149798916048736e-99095`; `mp.prec = 900` and `1400`:
/// `gamma(mpf(-121241)/4) + 194549429507494620*mpf(2)**(-407580)` =
/// `-4.457164848101349471030863493897814583e-122695`.
#[test]
fn gamma_left_of_zero_has_its_digits() {
    let ctx = Context::new();
    assert_eq!(
        dec(&ctx, "gamma(-100001/4)", 32),
        "-6.4855460632946947759194401497989e-99095"
    );
    assert_eq!(
        dec(&ctx, "gamma(-121241/4) + 194549429507494620*2^-407580", 32),
        "-4.4571648481013494710308634938978e-122695"
    );
}

// ── W23: zero digits ─────────────────────────────────────────────────────────

/// W23: every public function that takes a digit count refuses zero digits.  Up to 0.60 they
/// printed every digit of the 128-bit working value, of which none was asked for and four
/// bits were certified: `eval_decimal(0)` of `1/3` gave
/// `0.333333333333333333333333333333333333334`.  `Ex::eval_decimal`,
/// `Ex::integrate_numeric_decimal`, the `p_value_decimal` of a test result and the
/// `eval_decimal` of a quantity with a unit all reach the same check; one digit is the least
/// that can be asked.
///
/// Oracle: exact arithmetic (`1/3 = 0.333…`, so one digit is `0.3`); `π = 3.14…`; `∫₀¹ x² dx
/// = 1/3`.
#[test]
fn every_digit_count_refuses_zero_digits() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let third = ctx.rational(1, 3);
    assert!(matches!(
        third.eval_decimal(0),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert_eq!(third.eval_decimal(1).unwrap(), "0.3");
    assert_eq!(ctx.pi().eval_decimal(1).unwrap(), "3");

    let square = x.powi(2);
    let (zero, one) = (ctx.int(0), ctx.int(1));
    for rule in [QuadRule::TanhSinh, QuadRule::GaussLegendre] {
        assert!(matches!(
            square.integrate_numeric_decimal(&x, &[&zero, &one], 0, rule),
            Err(SymplexError::InvalidArgument { .. })
        ));
    }
    assert_eq!(
        square
            .integrate_numeric_decimal(&x, &[&zero, &one], 3, QuadRule::TanhSinh)
            .unwrap(),
        "0.333"
    );

    let table = counts(&[&[3, 1, 4], &[1, 2, 1], &[1, 1, 2]]);
    let test = g_test(&ctx, &table).unwrap();
    assert!(test.p_value_decimal(0).is_err());
    assert!(test.p_value_decimal(1).is_ok());

    let sixty = ctx.int(60);
    assert!(Frequency::rpm(&sixty).eval_decimal(0).is_err());
    assert!(Frequency::rpm(&sixty).eval_decimal(1).is_ok());
}

// ── The working precision of a digit count ───────────────────────────────────

/// A digit count whose working precision exceeds the configured maximum is refused with the
/// count it asked for, at every size of the count.  The precision is `3.4` bits per digit plus
/// 64; up to 0.60 it was computed in `usize`, which on a 32-bit target (`wasm32`) overflows
/// from 126,322,568 digits on: a panic in a debug build, in a release build a precision of 128
/// bits that passed this check, so that the call returned 16 good digits for the hundred
/// million asked.  (The arithmetic itself is tested in 32 bits by the unit test
/// `the_working_precision_of_a_digit_count_does_not_wrap_in_32_bits`; a 64-bit host cannot
/// show the overflow through the public function.)
///
/// Oracle: exact integer arithmetic.  The default maximum is 10,000 bits: `2922·34//10 + 64`
/// = 9998, `2941·34//10 + 64` = 10063; `126322568·34 − 2³²` = 16.
#[test]
fn a_digit_count_beyond_the_configured_precision_is_refused() {
    let ctx = Context::new();
    let pi = ctx.pi();
    for digits in [2941u32, 126_322_567, 126_322_568, 1_263_225_657, u32::MAX] {
        match pi.eval_decimal(digits) {
            Err(SymplexError::PrecisionExhausted { requested, .. }) => {
                assert_eq!(requested, digits);
            }
            other => panic!("{digits} digits: {:?}", other.map(|s| s.len())),
        }
    }
    // (Trailing zeros are not printed: at least 2,900 characters.)
    let most = pi.eval_decimal(2922).unwrap();
    assert!(most.len() > 2900 && most.len() <= 2923, "{}", most.len());
    assert!(most.starts_with("3.14159265358979323846264338327950288419716939937510"));
}

// ── W25: `eval_f64` of a definite integral ───────────────────────────────────

/// W25: what `eval_f64` certifies for an expression with an unevaluated definite integral is 8
/// digits (a relative error below `2⁻³⁰`), not the nearest `f64` its doc promised up to 0.60;
/// the nearest `f64` comes from `eval_decimal(20)`.  The two worst of 416 integrals measured
/// (the 16 of the review and 400 generated ones, `gen_integrals.py`, `check_w25.py`): 1,914 ulp
/// (`3.5·10⁻¹³` relative) and, of the review's, 14.6 ulp; 149 of the 416 are the nearest double.
/// Certifying 16 digits instead costs 7.5 times the total and 500 times the median integral, so
/// the 8 digits stay and the sentence was corrected.
///
/// Oracle: mpmath `quad` over 1,201 equal pieces (and the kinks `kπ/7`) at 40 and 60 digits:
/// `quad(lambda x: sin(74*x)*tanh(mpf(78)/4*(x - mpf(1)/2)), linspace(0, mpf(7)/5, 1201))` =
/// `0.00060010846586109082987418924853193887`, nearest double `0.0006001084658610908`;
/// `Integral(abs(sin(7*x))^(1/2)*exp(x), x, 0, 3)` = `14.918730674259974790840107983334`,
/// nearest double `14.918730674259974` (`check_integrals.py`).
#[test]
fn eval_f64_of_an_integral_has_eight_digits_and_eval_decimal_gives_the_nearest_double() {
    let ctx = Context::new();
    for (src, nearest) in [
        (
            "Integral(sin(74*x)*tanh(78/4*(x - 1/2)), x, 0, 7/5)",
            0.000_600_108_465_861_090_8_f64,
        ),
        (
            "Integral(abs(sin(7*x))^(1/2)*exp(x), x, 0, 3)",
            14.918_730_674_259_974_f64,
        ),
    ] {
        let integral = ctx.parse(src).unwrap();
        let served = integral.eval_f64().unwrap();
        assert!(
            ((served - nearest) / nearest).abs() < 2f64.powi(-30),
            "{src}: {served:e}"
        );
        assert_eq!(
            integral.eval_decimal(20).unwrap().parse::<f64>().unwrap(),
            nearest,
            "{src}"
        );
    }
}

// ── The slow units of `fuzz_evalf`: `Ei` left of 0, `E₁` at a small argument ──

/// `Ei(x)` for `x ≤ −1` summed its power series up to the asymptotic range: it cancels
/// `2·abs(x)·log₂e` bits and takes `e·abs(x)` terms at that many more bits (and Euler's constant
/// at them).  Up to 0.60 `Ei(−691)` at 280 digits took 128 ms in a release build and
/// `li(sin(1 + 10⁻³⁰⁰) − sin 1)` — `Ei` at `−691.3` at the 1,152 bits its argument needs — 145 ms
/// at 16 digits (three inputs of `fuzz_evalf` were saved as slow units for it).  Since 0.60.1
/// `Ei(x) = −E₁(−x)` by Legendre's continued fraction from `x = −max(1, wp/64)` on: 0.5 ms and
/// 1.7 ms.  The values are unchanged (346,320 rows of `Ei`, `li`, `expint(1, x)`,
/// `uppergamma(0, x)`, `Chi`, `Shi` under a cancellation: no error above 0.5075 units before
/// or after, `run_scan_ei.sh`).
///
/// Oracle: mpmath, `mp.dps = 700` and `1000` (equal): `li(sin(1 + mpf(10)**(-300)) - sin(1))`
/// = `-7.80344211866841093813089902954890256133e-304`; `ei(-691)` =
/// `-1.15453689153167835664526223642860247051871552687368770548986e-303`; `ei(mpf(-17)/2)` =
/// `-0.0000216211210434833718699599746614432304849782479087160221990308`; `mp.dps = 300` and
/// `500`: `ei(-200)` = `-6.8852261063076355977108174824557929738368086933302509880664624081348
/// 00621841274909985690580563900548e-90` (100 digits; the series up to 0.60, the fraction now).
#[test]
fn ei_left_of_zero_is_summed_without_cancellation() {
    let ctx = Context::new();
    let timer = CpuTimer::start();
    assert_eq!(
        dec(&ctx, "li(sin(1 + 10^(-300)) - sin(1))", 30),
        "-7.80344211866841093813089902955e-304"
    );
    assert_eq!(
        dec(&ctx, "Ei(-691)", 50),
        "-1.1545368915316783566452622364286024705187155268737e-303"
    );
    assert_eq!(
        dec(&ctx, "Ei(-17/2)", 40),
        "-2.162112104348337186995997466144323048498e-5"
    );
    assert_eq!(
        dec(&ctx, "Ei(-200)", 100),
        "-6.885226106307635597710817482455792973836808693330250988066462408134800621841274909985690580563900548e-90"
    );
    // (0.04 s in a debug build on the build host; a release build 2 ms, 150 ms up to 0.60.)
    timer.assert_within(Duration::from_secs(1), "Ei left of 0");
}

/// `E₁(x) = Γ(0, x)` for `x ≥ 1` was Legendre's continued fraction at every precision; its
/// steps grow as `wp²/x`, beyond its budget of `40·wp + 2000` from about `wp = 1300·x` bits on.
/// Up to 0.60 `expint(1, 4/3)` at 600 digits was refused after 87,440 steps ("continued
/// fraction for the incomplete gamma function did not converge"), and `expint(1, 1)` at 1,000
/// digits after 2.2 s.  Since 0.60.1 the series of `Ei(−x)` below `x = max(1, wp/64)`, where it
/// is the cheaper of the two.
///
/// Oracle: mpmath `expint(1, mpf(4)/3)` with `mp.dps = 640` and `700` (equal to 600 digits):
/// `0.12866430426774742126447263212022104377158848128417207701641554123454…
/// 476756635191897601316890682393386137170567932` (the first 70 and the last 45 characters of
/// `nstr(·, 600)`); `expint(1, 1)` with `mp.dps = 700` and `1000` =
/// `0.219383934395520273677163775460121649031047293406908207577979`.
#[test]
fn expint_one_of_a_small_argument_has_its_digits_at_a_high_precision() {
    let ctx = Context::new();
    let most = dec(&ctx, "expint(1, 4/3)", 600);
    assert_eq!(most.len(), 602);
    assert!(
        most.starts_with("0.12866430426774742126447263212022104377158848128417207701641554123454"),
        "{most}"
    );
    assert!(
        most.ends_with("476756635191897601316890682393386137170567932"),
        "{most}"
    );
    assert_eq!(
        dec(&ctx, "expint(1, 1)", 50),
        "0.21938393439552027367716377546012164903104729340691"
    );
}

// ── Two slow inputs of `fuzz_stats` traced into the numeric evaluator ────────

/// The distribution function of `Poisson(λ)`, `λ ≈ 1.8·10⁹`, at its median is
/// `uppergamma(k + 1, λ)/k!`: `Γ(s, x)` of a huge shape just above `x`.  Up to 0.60 it was
/// `Γ(s) − γ(s, x)` with the series of `γ`, `√(2x·wp·ln 2)` = 630,000 terms: 0.39 s per value in
/// a release build (five slow units of `fuzz_stats`, 3.3 to 3.7 s each in the fuzz build).
/// Since 0.60.1 the continued fraction at a shape just below `x` and the recurrence up to `s`
/// (`evalf/incgamma.rs: Bridge`): 0.04 s.  The values are unchanged (the scan of the lane's
/// notes: shapes `10⁸` to `10¹⁰`, 0 to 20 standard deviations above `x`, against mpmath and
/// against the series of 0.60), and so is the set of arguments that have one: beyond the
/// series' limit (`s ≈ 3·10⁹` here) the quotient is refused at once, as up to 0.60.
///
/// Oracle: mpmath `gammainc(mpf(1780358091), mpf(3733681530275153)/2097152,
/// regularized=True)` with `mp.prec = 400` and `700` = `0.500001274213415293377977073917`, and
/// `1 −` it = `0.499998725786584706622022926083`.
#[test]
fn the_incomplete_gamma_of_a_huge_shape_next_to_its_mode_is_quick() {
    let ctx = Context::new();
    let timer = CpuTimer::start();
    let cdf = ctx
        .parse("uppergamma(1780358091, 3733681530275153/2097152)/1780358090!")
        .unwrap();
    assert_eq!(cdf.eval_decimal(25).unwrap(), "0.5000012742134152933779771");
    assert_eq!(cdf.eval_f64().unwrap(), 0.500_001_274_213_415_3);
    let sf = ctx
        .parse("1 - uppergamma(1780358091, 3733681530275153/2097152)/1780358090!")
        .unwrap();
    assert_eq!(sf.eval_decimal(25).unwrap(), "0.4999987257865847066220229");
    let lower = ctx
        .parse("lowergamma(1780358091, 3733681530275153/2097152)/1780358090!")
        .unwrap();
    assert_eq!(
        lower.eval_decimal(25).unwrap(),
        "0.4999987257865847066220229"
    );
    // (1.9 s in a debug build on the build host; up to 0.60 about nine times that.)
    timer.assert_within(
        Duration::from_secs(5),
        "four incomplete gamma values at s = 1.8e9",
    );
}

/// Beyond the limit of the series the incomplete gamma of a huge shape next to its mode answers
/// at once, value or refusal: the distribution function of `Poisson(3620896395585425/131072)` at
/// `27625247159`, a shape of `2.8·10¹⁰`.  0.60 refuses it in 0.2 ms.  The fraction-and-recurrence
/// route of 0.60.1 has the value, but in 0.1 s in a release build and 1.7 s in a debug build
/// (it was taken there before this test: `v36_s15n2_fuzz::
/// poisson_at_a_huge_rate_refuses_its_symbolic_tails_at_once`, which allows one second, failed),
/// so it is taken only where the series would have run (`evalf/incgamma.rs: Bridge::plan`).
///
/// Oracle (for a value, should one be returned): mpmath, `Q(k + 1, λ)` as
/// `λ^(k+1)e^(−λ)/Γ(k + 1)·∫₀^∞ e^(k ln(1 + v) − λv) dv` (tanh-sinh), 30 and 50 digits:
/// `0.50000231477623983` (the value of the test named above).
#[test]
fn the_incomplete_gamma_beyond_the_limit_of_its_series_answers_at_once() {
    let ctx = Context::new();
    let cdf = ctx
        .parse("uppergamma(27625247160, 3620896395585425/131072)/27625247159!")
        .unwrap();
    let timer = CpuTimer::start();
    if let Ok(p) = cdf.eval_f64() {
        assert!((p - 0.500_002_314_776_239_8).abs() < 1e-15, "{p}");
    }
    timer.assert_within(Duration::from_secs(1), "the incomplete gamma at s = 2.8e10");
}

/// The moment generating function of `FDistribution(11, d₂)`, `d₂ ≈ 2.3·10⁻²⁵³`, is an integral
/// whose integrand is `1/x` from `x ≈ 10⁻²⁵⁴` to 1 and `x^(9/2)` below: its mass lies in a layer
/// `2⁻⁸⁴⁴` of the range from its end, which tanh-sinh resolves only at a level that takes more
/// evaluations than the quadrature has.  Up to 0.60 the pieces were bisected to the last depth
/// first: 10,775 evaluations, 1.5 s in a release build (5.5 s in the fuzz build, the timeout
/// input `20921c72b4f2` of `fuzz_stats`), to be refused.  Since 0.60.1 the layer is recognised
/// on the first piece (`evalf/quad.rs: end_layer`) and the integral is refused at once, in
/// 0.09 s, with the refusal `eval_f64` and `eval_decimal` document: `PrecisionExhausted`.  The
/// same integrand with the layer within reach is evaluated as before.
///
/// Oracle: the refusal is the documented one; `∫₀¹ dx/(x + 10⁻¹⁰) = ln(1 + 10¹⁰)`, mpmath
/// `log(1 + mpf(10)**10)` with `mp.prec = 400` and `700` = `23.0258509300404568401749145468`.
#[test]
fn an_integral_whose_mass_lies_in_an_end_layer_is_refused_at_once() {
    let ctx = Context::new();
    let timer = CpuTimer::start();
    let mgf = ctx
        .parse("Integral(x^(9/2)*(1 + 11*10^250*x)^(-(11 + 1/10^250)/2)*exp(-x/8), x, 0, oo)")
        .unwrap();
    assert!(matches!(
        mgf.eval_f64(),
        Err(SymplexError::PrecisionExhausted { .. })
    ));
    assert!(matches!(
        mgf.eval_decimal(20),
        Err(SymplexError::PrecisionExhausted { .. })
    ));
    // (1.2 s in a debug build on the build host; up to 0.60 about seventeen times that.)
    timer.assert_within(Duration::from_secs(5), "the refusal of an end layer");
    assert_eq!(
        dec(&ctx, "Integral(1/(x + 1/10^10), x, 0, 1)", 25),
        "23.02585093004045684017491"
    );
}

// ── The release blocker of 0.60.1: the slow refusal of a law far narrower than its range ──────

/// The integrals `fuzz_stats` checks a characteristic function against: `cos(sx)·f(x)` and
/// `sin(sx)·f(x)` over `[0, ∞)` for the density `f` of `Gamma(√2/4, θ)`, `θ = 1/(25·10¹⁵⁴)`,
/// and the `s ≈ −5.26·10¹⁵⁴` of the input `431a685e4a17` (`−1/8` over the standard deviation,
/// as a double).  The density underflows at every node of the quadrature from `x = 1` down to
/// `2⁻³³⁶`; what the range holds lies between that node and the next, where no level the
/// quadrature reaches resolves it, and `integrate_numeric` refuses ("every node of the f64
/// quadrature gave 0 … and the arbitrary-precision quadrature failed: … the integrand
/// oscillates about 8.36·10¹⁵³ times over a piece of the range").  Up to 0.60.1 that refusal
/// took 1.9 s in a release build (6.3 s in the fuzz build: the timeout of the nightly run on
/// the release commit) and 5,280 evaluations of the integrand: every node that underflowed was
/// evaluated again at each precision up to 2,752 bits (`evalf/quad.rs: Quad::term`), and the
/// whole refusal was computed twice, once for `integrate_numeric`'s own quadrature and once for
/// the `eval_f64` it asks next (`QuadError::Unresolvable` is now remembered).  The refusal is
/// the same; it takes 0.27 s.
///
/// Oracle: the refusal is the documented one (`ComputationFailed`).  The integrals have
/// values, which the library does not compute at this scale: the real and imaginary part of
/// the law's characteristic function at `s`, `0.9897225671495418 − 0.0726357514890965i` (the
/// closed form `(1 − isθ)^(−√2/4)` as the harness evaluates and prints it for that input).
#[test]
fn the_refusal_of_a_density_far_narrower_than_the_half_line_is_prompt() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let law = Distribution::try_gamma(
        ctx.parse("sqrt(2)/4").unwrap(),
        ctx.parse("1/(25*10^154)").unwrap(),
    )
    .unwrap();
    let s = ctx
        .parse(
            "-52556025953363892372094512341551374542328919970889013859826144814180893194263070339\
             028647983350324472518365953105210059975943266018688001205686601888301056",
        )
        .unwrap();
    let (zero, oo) = (ctx.zero(), ctx.infinity());
    let timer = CpuTimer::start();
    let re = ((&s * &x).cos() * law.density(&x)).integrate_numeric(&x, &zero, &oo);
    let im = ((&s * &x).sin() * law.density(&x)).integrate_numeric(&x, &zero, &oo);
    timer.assert_within(
        Duration::from_secs(30),
        "the refusals of cos(sx) f and sin(sx) f for a gamma law of scale 4e-156",
    );
    for r in [re, im] {
        assert!(
            matches!(r, Err(SymplexError::ComputationFailed { .. })),
            "{r:?}"
        );
    }
}

/// The same integrals at an ordinary scale, `Gamma(√2/4, 1)` and `s = 1/5`, are values, and
/// the same values before and after the change of 0.60.1 to the quadrature's nodes that
/// underflow (`0.990673166768298641` and `0.0692514201107100325` from `integrate_numeric`,
/// whose tolerance is `10⁻¹⁰`).
///
/// Oracle: mpmath 1.3.0, `quad(lambda u: cos(u/5)*u**(k-1)*exp(-u), [0, 1e-20, 1e-8, 0.01, 1,
/// 4, 16, 64, 200])/gamma(k)`, `k = sqrt(2)/4`, at `mp.dps = 60` =
/// `0.990673166774744471963815970531`, the same with `sin` =
/// `0.0692514201107071040604513545001`; the closed form `(1 - 1j/5)**(-k)` =
/// `0.990673166774744471963815970533 + 0.0692514201107071040604513545001j`.
#[test]
fn the_same_integrals_at_an_ordinary_scale_have_their_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let law = Distribution::try_gamma(ctx.parse("sqrt(2)/4").unwrap(), ctx.one()).unwrap();
    let s = ctx.rational(1, 5);
    let (zero, oo) = (ctx.zero(), ctx.infinity());
    let re = ((&s * &x).cos() * law.density(&x))
        .integrate_numeric(&x, &zero, &oo)
        .unwrap();
    let im = ((&s * &x).sin() * law.density(&x))
        .integrate_numeric(&x, &zero, &oo)
        .unwrap();
    assert!((re - 0.990_673_166_774_744_5).abs() < 1e-9, "{re}");
    assert!((im - 0.069_251_420_110_707_1).abs() < 1e-9, "{im}");
}
