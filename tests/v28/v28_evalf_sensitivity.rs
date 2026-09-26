//! The evaluator's bound of a special function scales each argument's error
//! by the function's sensitivity to it (`evalf/sensitivity.rs`), and exact
//! gamma values are bounded by the digit guard (`eval.rs`).
//!
//! Before 0.29 a special function carried its arguments' relative error
//! over, plus 4 bits, whatever its condition number: a rational argument
//! rounded to the working precision (`1 − 3·10⁻³⁰` is not a binary number)
//! gave a value whose error was far beyond the bound, and `evalf` printed
//! wrong digits as certified.  And `gamma(p/q)` was expanded exactly
//! whatever its size: `gamma(3000 + 1/3)` hung in `eval()`.
//!
//! Every reference is mpmath 1.3 with exact rational inputs
//! (`mpf(27)/11`, never a decimal literal), quoted from two working
//! precisions that agree (the dps are given with each value).

// Reference values are quoted at the digits mpmath printed them.
#![allow(clippy::excessive_precision)]

use std::time::{Duration, Instant};

use num_bigint::BigInt;
use num_traits::{Signed, Zero};
use symplex::prelude::*;

/// A decimal string as `m·10^e` with `m` an integer.
fn decimal(s: &str) -> (BigInt, i64) {
    let s = s.trim();
    let (mant, exp) = match s.find(['e', 'E']) {
        Some(i) => (&s[..i], s[i + 1..].parse::<i64>().expect("exponent")),
        None => (s, 0),
    };
    let (neg, mant) = match mant.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, mant),
    };
    let (ip, fp) = mant.split_once('.').unwrap_or((mant, ""));
    let mut m: BigInt = format!("{ip}{fp}").parse().expect("digits");
    if neg {
        m = -m;
    }
    (m, exp - i64::try_from(fp.len()).expect("length"))
}

fn pow10(k: i64) -> BigInt {
    BigInt::from(10u32).pow(u32::try_from(k).expect("non-negative power"))
}

/// `src` evaluated to `digits` digits is within one unit of its last printed
/// digit of `reference` (which carries more digits): the digits printed are
/// right.
fn certified(ctx: &Context, src: &str, digits: u32, reference: &str) {
    let printed = ctx
        .parse(src)
        .expect("parse")
        .eval_decimal(digits)
        .unwrap_or_else(|err| panic!("{src} at {digits} digits: {err}"));
    let (m1, e1) = decimal(&printed);
    let (m2, e2) = decimal(reference);
    assert!(!m1.is_zero(), "{src}: printed 0 for {reference}");
    let e0 = e1.min(e2);
    let a = &m1 * pow10(e1 - e0);
    let b = &m2 * pow10(e2 - e0);
    let lead = e1 + i64::try_from(m1.abs().to_string().len()).expect("length") - 1;
    let ulp = (lead - i64::from(digits) + 1).max(e1);
    assert!(
        ulp >= e0,
        "{src}: the reference {reference} has too few digits"
    );
    assert!(
        (a - b).abs() <= pow10(ulp - e0),
        "{src} at {digits} digits: printed {printed}, mpmath {reference}"
    );
}

/// `eval_f64` of `src` is the double nearest `reference` (to an ulp).
fn f64_certified(ctx: &Context, src: &str, reference: &str) {
    let want: f64 = reference.parse().expect("reference");
    let got = ctx
        .parse(src)
        .expect("parse")
        .eval_f64()
        .unwrap_or_else(|err| panic!("{src}: {err}"));
    assert!(
        (got - want).abs() <= want.abs() * f64::EPSILON,
        "{src}: eval_f64 {got:e}, mpmath {reference}"
    );
}

fn certified_at(ctx: &Context, src: &str, digits: &[u32], reference: &str) {
    for &d in digits {
        certified(ctx, src, d, reference);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// The incomplete beta function near x = 1 (the reported cases)
// ═══════════════════════════════════════════════════════════════════════════

/// `I_x(a, b)` near `x = 1` with a small `b` has `∂I/∂x = x^{a−1}(1−x)^{b−1}/B
/// ≈ b/(1 − x)`, `10²²` here: the rounding of `x = 1 − 3·10⁻³⁰` to 128 bits
/// moved the value in its 11th digit.  Before: `eval_f64` gave
/// `2.5118502017106955e-6` and `eval_decimal(16)` `2.511850201710696e-6`.
#[test]
fn betainc_near_one_with_a_small_second_shape() {
    let ctx = Context::new();
    let src = "betainc_regularized(27/11, 1/26562500, 0, 999999999999999999999999999997/10^30)";
    // mpmath: betainc(mpf(27)/11, mpf(1)/26562500, 0,
    //   mpf(999999999999999999999999999997)/10**30, regularized=True), dps 120 and 240
    let r = "2.51185020171944254712800040938144192478820967288901553716096e-6";
    f64_certified(&ctx, src, r);
    certified_at(&ctx, src, &[16, 30, 50], r);
}

/// The second reported case (`b = 33/(4.75·10²⁶)`, `x = 1 − 5·10⁻²⁸`).
/// Before: `eval_f64` gave `4.199018237177169e-24`.
#[test]
fn betainc_near_one_with_a_tiny_second_shape() {
    let ctx = Context::new();
    let src = "betainc_regularized(116/17, 33/(475*10^24), 0, 1 - 5/10^28)";
    // mpmath: betainc(mpf(116)/17, mpf(33)/(475*10**24), 0, 1 - mpf(5)/10**28,
    //   regularized=True), dps 120 and 240
    let r = "4.19901823717697769336176113655685952438491979720484182550651e-24";
    f64_certified(&ctx, src, r);
    certified_at(&ctx, src, &[16, 30, 50], r);
}

/// The upper tail of large shapes from `x = 1 − 6·10⁻²⁶`: `d ln(tail)/d ln(1−x)
/// ≈ b ≈ 10⁶`.  Before: `eval_decimal(16)` gave `5.318645970393041e-21479656`
/// (wrong from the 8th digit) and the last 4 of 50 digits were wrong.
#[test]
fn betainc_upper_tail_of_large_shapes_near_one() {
    let ctx = Context::new();
    let src = "betainc_regularized(65470000000/999, 10300000/11, 1 - 6/10^26, 1)";
    // mpmath: betainc(mpf(10300000)/11, mpf(65470000000)/999, 0, mpf(6)/10**26,
    //   regularized=True) (the exact complement), dps 120 and 240
    let r = "5.31864604547481966559621231318753081099005529281864071158247e-21479656";
    certified_at(&ctx, src, &[16, 30, 50], r);
}

/// A limit that rounds to 1 at the working precision: the integrand
/// `(1 − t)^{b−1}` is infinite there, and the bound is the integral over the
/// error ball.  Before: `1`, at every precision.
#[test]
fn betainc_with_a_limit_that_rounds_to_one() {
    let ctx = Context::new();
    let src = "betainc_regularized(1/3, 1/7, 0, 1 - 10^-60)";
    // mpmath: 1 - betainc(mpf(1)/7, mpf(1)/3, 0, mpf(10)**-60, regularized=True),
    //   dps 60 and 120
    let r = "0.999999998008883450425455639343213754657067697672155010175193";
    certified_at(&ctx, src, &[16, 30], r);
}

// ═══════════════════════════════════════════════════════════════════════════
// The same class beyond betainc
// ═══════════════════════════════════════════════════════════════════════════

/// Next to a zero the relative error of a value is its argument's error
/// times `|f′/f|`, however modest `f′`: each argument is a 33- or 36-digit
/// rational within `10⁻³³` of the zero.  Before (16 digits): `besselj`
/// `6.719281178695182e-38` (wrong from the 2nd digit), `airyai`
/// `3.089822608156168e-34`, `Ci` `-9.893179153709206e-34`, `Ei`
/// `-5.202811540924496e-34`, `li` `-1.206669993690368e-33` (5th–6th digits).
#[test]
fn special_functions_next_to_their_zeros() {
    let ctx = Context::new();
    let cases = [
        (
            // mpmath: besselj(0, mpf(2404825557695772768621631879326454643)/10**36), dps 120 and 240
            "besselj(0, 2404825557695772768621631879326454643/10^36)",
            "6.45014336340860209398022974859975954115868436534497833841344e-38",
        ),
        (
            // mpmath: airyai(mpf(-2338107410459767038489197252446735)/10**33), dps 120 and 240
            "airyai(-2338107410459767038489197252446735/10^33)",
            "3.08980513257991324692236984894245420615042137859950264854036e-34",
        ),
        (
            // mpmath: ci(mpf(616505485620716233797110404100172)/10**33), dps 120 and 240
            "Ci(616505485620716233797110404100172/10^33)",
            "-9.89318697784191757761494095949544298242465783566967472307164e-34",
        ),
        (
            // mpmath: ei(mpf(372507410781366634461991866580119)/10**33), dps 120 and 240
            "Ei(372507410781366634461991866580119/10^33)",
            "-5.20283854459359999208691804457170589847446849476285652928277e-34",
        ),
        (
            // mpmath: li(mpf(1451369234883381050283968485892027)/10**33), dps 120 and 240
            "li(1451369234883381050283968485892027/10^33)",
            "-1.20666869778723960568312781128037498130439471699335808499538e-33",
        ),
    ];
    for (src, r) in cases {
        certified_at(&ctx, src, &[16, 30, 50], r);
    }
}

/// `ζ` and `η` at a trivial zero, `ln Γ` at its zero `x = 1`: the argument
/// `−2 + √2·10⁻³⁰` (or `1 + √2·10⁻³⁰`) is rounded as a sum, `10⁻³⁸` off, a
/// relative `10⁻⁸` of the distance to the zero.  Before (16 digits):
/// `-4.306062089524848e-32`, `3.014243462667394e-31`,
/// `-8.163062211480588e-31` (wrong from the 9th digit).
#[test]
fn special_functions_at_a_zero_of_a_rounded_argument() {
    let ctx = Context::new();
    let cases = [
        (
            // mpmath: zeta(-2 + sqrt(2)/mpf(10)**30), dps 120 and 240
            "zeta(-2 + sqrt(2)/10^30)",
            "-4.3060620925314558039107397021561118347227915802000714581052e-32",
        ),
        (
            // mpmath: altzeta(-2 + sqrt(2)/mpf(10)**30), dps 120 and 240
            "dirichlet_eta(-2 + sqrt(2)/10^30)",
            "3.01424346477201906273751779150590144235996596827106648892629e-31",
        ),
        (
            // mpmath: loggamma(1 + sqrt(2)/mpf(10)**30), dps 120 and 240
            "loggamma(1 + sqrt(2)/10^30)",
            "-8.16306221717951472723921551200180699330489133212842690113056e-31",
        ),
    ];
    for (src, r) in cases {
        certified_at(&ctx, src, &[16, 30], r);
    }
}

/// Logarithmic and power singularities at `x = 1`, reached by an argument
/// `1 − 10⁻³⁰` rounded to 128 bits: `K(m) ≈ ln(4/√(1−m))` has
/// `∂ ln K/∂ ln(1−m) ≈ 1/(2 ln …)` but `∂K/∂m ≈ 1/(2(1−m))`; `Π(n|m)`,
/// `Li_{1/2}(z)` and `erf⁻¹(y)` blow up like powers.  Before (16 digits):
/// `35.92507075591441`, `2221441468821116`, `1772453850699608`,
/// `8.148616223155713`.
#[test]
fn special_functions_next_to_a_singular_point() {
    let ctx = Context::new();
    let cases = [
        (
            // mpmath: ellipk(1 - mpf(1)/10**30), dps 120 and 240
            "elliptic_k(1 - 10^-30)",
            "35.9250707560305758791043360631905475178565302421218812327571",
            &[16u32, 30, 50][..],
        ),
        (
            // mpmath: ellippi(1 - mpf(1)/10**30, mpf(1)/2), dps 120 and 240
            "elliptic_pi(1 - 10^-30, 1/2)",
            "2221441469079182.27629485570105237096354272695259272462311653",
            &[16, 30, 50][..],
        ),
        (
            // mpmath: erfinv(1 - mpf(1)/10**30), dps 120 and 240
            "erfinv(1 - 10^-30)",
            "8.14861622316986460738456666064810053831286348685420259220632",
            &[16, 30, 50][..],
        ),
        (
            // mpmath: polylog(mpf(1)/2, 1 - mpf(1)/10**30), dps 120 and 240
            "polylog(1/2, 1 - 10^-30)",
            "1772453850905514.566943658673753889179835670562025436344091",
            &[16][..],
        ),
    ];
    for (src, r, digits) in cases {
        certified_at(&ctx, src, digits, r);
    }
}

/// `erfc(x)` has `|∂ ln erfc/∂x| ≈ 2x`: an argument `30000 + 5·10⁻²¹` whose
/// error is that of `√(10⁴⁰ + 1)` at 128 bits (`3·10⁻¹⁹`) moves the value in
/// its 15th digit.  Before: `eval_decimal(16)` gave `3.642312160534615e-390865039`.
#[test]
fn erfc_of_a_large_argument_with_an_error() {
    let ctx = Context::new();
    let src = "erfc(sqrt(10^40+1) - 10^20 + 30000)";
    // mpmath: erfc(sqrt(mpf(10)**40+1) - mpf(10)**20 + 30000), dps 120 and 240
    let r = "3.64231216053461435322969240107051241679722565557576250374287e-390865039";
    certified_at(&ctx, src, &[16, 30, 50], r);
}

/// Exact arguments cost nothing and dyadic ones are exact: the kernels
/// themselves were right (`x = 1 − 2⁻¹⁰⁰`).
#[test]
fn a_dyadic_limit_needs_no_sensitivity() {
    let ctx = Context::new();
    let src = "betainc_regularized(27/11, 1/26562500, 0, 1 - 2^-100)";
    // mpmath: betainc(mpf(27)/11, mpf(1)/26562500, 0, 1 - mpf(2)**-100,
    //   regularized=True), dps 60 and 120
    let r = "2.56213817032109447681848924309806061714829566004830343041622e-6";
    certified_at(&ctx, src, &[16, 30, 50], r);
}

/// A parameter within `10⁻¹²` of an integer was taken for it (an `f64`
/// test in the routines): `binomial(2 + 10⁻¹³, 3)` printed `0`,
/// `besselj(2 + 10⁻¹⁴, 1)`, `besselk(1 + 10⁻¹⁴, 1)` and `besseli(1 +
/// 10⁻¹⁴, 1)` were the integer order's values (wrong from the 15th digit),
/// and `legendre(3 + 10⁻¹³, 1/3)` was `P₃(1/3)`.  Integers are exact now;
/// a polynomial degree that is not one is refused.
#[test]
fn a_near_integer_parameter_is_not_an_integer() {
    let ctx = Context::new();
    let cases = [
        (
            // mpmath: binomial(2 + mpf(1)/10**13, 3), dps 120 and 240
            "binomial(2 + 10^-13, 3)",
            "3.33333333333383333333333335e-14",
            &[16u32][..],
        ),
        (
            // mpmath: besselj(2 + mpf(1)/10**14, 1), dps 60 and 120
            "besselj(2 + 10^-14, 1)",
            "0.114903484931898656573338027378827470972841436542601931196907",
            &[16, 30, 50][..],
        ),
        (
            // mpmath: besselk(1 + mpf(1)/10**14, 1), dps 60 and 120
            "besselk(1 + 10^-14, 1)",
            "0.601907230197238784981922408652984439997279570180189266535709",
            &[16, 30, 50][..],
        ),
        (
            // mpmath: besseli(1 + mpf(1)/10**14, 1), dps 60 and 120
            "besseli(1 + 10^-14, 1)",
            "0.565159103992478385621220479894660365555262637905888809849809",
            &[16, 30, 50][..],
        ),
    ];
    for (src, r, digits) in cases {
        certified_at(&ctx, src, digits, r);
    }
    let e = ctx.parse("legendre(3 + 10^-13, 1/3)").expect("parse");
    assert!(e.eval_decimal(16).is_err());
}

/// Found probing the class, with exact (even dyadic) arguments: the
/// routines' own error went unmeasured.  Before: `digamma(1/3)` was wrong
/// from its 46th digit (the asymptotic series stopped after 12 Bernoulli
/// terms), `digamma` next to its zero `x₀ = 1.4616…` from its 5th (the
/// cancellation near a zero), `loggamma(1 + 2⁻¹⁰⁰)` from its 13th, and
/// `gamma(−3 + √2·10⁻³⁰)` and `polygamma(1, −3 + √2·10⁻³⁰)` (16 digits)
/// were refused as poles (an `f64` test within `10⁻¹²`, `πx` rounded before
/// `sin`).
#[test]
fn gamma_family_routines_measure_their_own_cancellation() {
    let ctx = Context::new();
    let cases = [
        (
            // mpmath: digamma(mpf(1)/3), dps 60 and 120
            "digamma(1/3)",
            "-3.13203378002080632299641907428726885415542829672041806419275",
            &[16u32, 30, 50][..],
        ),
        (
            // mpmath: digamma(mpf(1461632144968362341262659542325721)/10**33), dps 120 and 240
            "digamma(1461632144968362341262659542325721/10^33)",
            "-3.17849556978860714368259581821772495437907118408215890367404e-34",
            &[16, 30, 50][..],
        ),
        (
            // mpmath: loggamma(1 + mpf(2)**-100), dps 60 and 120
            "loggamma(1 + 2^-100)",
            "-4.5534287192197142451711107863344994820178041477325525155035e-31",
            &[16, 30, 50][..],
        ),
        (
            // mpmath: gamma(-3 + sqrt(2)/mpf(10)**30), dps 120 and 240
            "gamma(-3 + sqrt(2)/10^30)",
            "-117851130197757920733474060351.017526158877956360200142971932",
            &[16, 30, 50][..],
        ),
        (
            // mpmath: polygamma(1, -3 + sqrt(2)/mpf(10)**30), dps 120 and 240
            "polygamma(1, -3 + sqrt(2)/10^30)",
            "500000000000000000000000000000000000000000000000000000000003.0",
            &[16, 30][..],
        ),
        (
            // mpmath: besselk(1 + mpf(1)/10**40, 1), dps 60 and 120
            "besselk(1 + 10^-40, 1)",
            "0.601907230197234574737540001535617339261628992411930526851102",
            &[16, 30, 50][..],
        ),
    ];
    for (src, r, digits) in cases {
        certified_at(&ctx, src, digits, r);
    }
}

/// `loggamma` left of 0 (`ln|Γ|`, the compiled back-ends' convention) next
/// to a pole: the reflection took `sin` of a rounded `πx`, and an argument
/// within `10⁻¹²` of a pole (in `f64`) was refused as the pole.  Before:
/// `loggamma(−3 + 2⁻¹²⁶)` was `Unevaluable`.
#[test]
fn loggamma_next_to_a_pole() {
    let ctx = Context::new();
    // mpmath: loggamma(-3 + mpf(2)**-126).real, dps 60 and 120
    certified_at(
        &ctx,
        "loggamma(-3 + 2^-126)",
        &[16, 30, 50],
        "85.5447852813250539857587699453495453048047918294412537601834",
    );
}

/// `expint(ν, x) = x^{ν−1}Γ(1 − ν, x)` far left of 0 took `|1 − ν|` downward
/// steps, `2·steps` overflowing for `ν = 10²⁰/3`: a panic in debug builds,
/// an endless loop in release ones.  The continued fraction takes over
/// beyond 256 steps.
#[test]
fn expint_of_a_huge_order_does_not_panic() {
    let ctx = Context::new();
    let t = Instant::now();
    let r = ctx
        .parse("expint(10^20/3, 1/7)")
        .expect("parse")
        .eval_decimal(16);
    assert!(
        r.is_ok() || matches!(r, Err(SymplexError::PrecisionExhausted { .. })),
        "{r:?}"
    );
    assert!(t.elapsed() < Duration::from_secs(20));
    // mpmath: gammainc(-300 - mpf(1)/3, mpf(1)/7), dps 60 and 120
    let r = "1.86749104572282322624604969628091496187007034480563954996332e+251";
    certified_at(&ctx, "uppergamma(-300 - 1/3, 1/7)", &[16, 30, 50], r);
}

// ═══════════════════════════════════════════════════════════════════════════
// Exact gamma values within the digit guard
// ═══════════════════════════════════════════════════════════════════════════

/// Before: `gamma(1000 + 1/3)` took 2.9 s in `eval()` (debug), `gamma(3000 +
/// 1/3)`, `gamma(−3000 + 1/3)`, `3000!`, `gamma(3000)` and `loggamma(3000)`
/// more than a minute, so `uppergamma(s, x)/gamma(s)` hung before `evalf`
/// ran.  Beyond `max_result_digits` digits the node stays symbolic and is
/// evaluated numerically.
#[test]
fn gamma_of_a_large_rational_is_evaluated_numerically() {
    let ctx = Context::new();
    let t = Instant::now();
    let cases = [
        (
            // mpmath: gamma(3000 + mpf(1)/3), dps 60 and 120
            "gamma(3000 + 1/3)",
            "1.99473015340579839527079495589351702865399098922143028243516e+9128",
        ),
        (
            // mpmath: gamma(-3000 + mpf(1)/3), dps 60 and 120
            "gamma(-3000 + 1/3)",
            "1.26094071895331358303958524196115757997398904594051900026564e-9129",
        ),
        (
            // mpmath: gammainc(3000 + mpf(1)/3, 3000)/gamma(3000 + mpf(1)/3), dps 60 and 120
            "uppergamma(3000 + 1/3, 3000)/gamma(3000 + 1/3)",
            "0.500000047959181177369837623820246796892760679240974947631003",
        ),
        (
            // mpmath: factorial(3000), dps 60 and 120
            "factorial(3000)",
            "4.14935960343785408555686709308661217095111919493180991768947e+9130",
        ),
        (
            // mpmath: loggamma(3000), dps 60 and 120
            "loggamma(3000)",
            "21016.0184854778974546148371304789353967458264644367404117579",
        ),
        (
            // mpmath: gamma(10**5 + mpf(1)/2), dps 60 and 120
            "gamma(10^5 + 1/2)",
            "8.930986400243598515815623500504390675981179177282316594544e+456570",
        ),
        (
            // mpmath: binomial(20000, 10000), dps 60 and 120
            "binomial(20000, 10000)",
            "2.24560266274634554155156943615786314758973696577987824362071e+6018",
        ),
    ];
    for (src, r) in cases {
        certified_at(&ctx, src, &[16, 30], r);
    }
    assert!(t.elapsed() < Duration::from_secs(30), "{:?}", t.elapsed());
    for src in [
        "gamma(3000 + 1/3)",
        "factorial(3000)",
        "binomial(20000, 10000)",
    ] {
        let e = ctx.parse(src).expect("parse").eval();
        assert!(e.as_rational().is_none(), "{src} expanded to {e}");
    }
    // A rising factorial beyond the guard stays symbolic (and evalf has no
    // routine for it): an error, promptly.
    let t = Instant::now();
    let e = ctx.parse("rising_factorial(1/3, 3000)").expect("parse");
    assert!(e.eval_decimal(16).is_err());
    assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
}

/// Within the guard the exact values are unchanged.
#[test]
fn small_gamma_values_stay_exact() {
    let ctx = Context::new();
    let g = ctx.parse("gamma(10 + 1/3)").expect("parse").eval();
    assert_eq!(g.to_string(), "17041024000/59049*Gamma(1/3)");
    // mpmath: gamma(10 + mpf(1)/3), dps 60 and 120
    certified(
        &ctx,
        "gamma(10 + 1/3)",
        30,
        "773118.187682764490695352523222748312856448962358641789641191",
    );
    let cases = [
        ("factorial(20)", "2432902008176640000"),
        ("gamma(6)", "120"),
        ("binomial(10, 3)", "120"),
        ("rising_factorial(-7/2, 5)", "105/32"),
        ("falling_factorial(5, 10)", "0"),
        ("rising_factorial(-5, 10)", "0"),
    ];
    for (src, want) in cases {
        let e = ctx.parse(src).expect("parse").eval();
        assert_eq!(e.to_string(), want, "{src}");
    }
}

/// Before (during 0.29's development): the shape sensitivity of
/// `betainc_regularized(p, q, 0, x)` read the lower limit `0` as positive
/// (astro-float's `is_positive` is the sign bit, true for `+0`) and bounded
/// `|E[ln X | X < x]|` by `|ln 0|`, so `ln` of a small value at an inexact
/// shape had no bound: `p_value_gg_log10` of a repeated-measures ANOVA was
/// `PrecisionExhausted`.
#[test]
fn a_zero_lower_limit_is_not_positive() {
    let ctx = Context::new();
    let ln_of = |s: &str| ctx.parse(s).unwrap().eval_f64().unwrap();
    // mpmath (dps 60): log(betainc(mpf(272861)/11810, mpf(9409)/11810, 0,
    //   mpf(97)/60000030000112, regularized=True)) = -628.09220426642316702
    let v = ln_of("log(betainc_regularized(272861/11810, 9409/11810, 0, 97/60000030000112))");
    assert!((v / -628.092_204_266_423_2 - 1.0).abs() < 1e-14, "{v}");
    // mpmath (dps 60): log(betainc(mpf(70)/3, mpf(1)/2, 0, mpf(2)**-20,
    //   regularized=True)) = -325.62134695677913195
    let v = ln_of("log(betainc_regularized(70/3, 1/2, 0, 2^(-20)))");
    assert!((v / -325.621_346_956_779_13 - 1.0).abs() < 1e-14, "{v}");
}

/// Before: `tan(π(1/2 − 10⁻³⁰⁰)).eval_f64()` was `0.0` (truly `3.18·10²⁹⁹`,
/// mpmath dps 400: `tan(pi*(mpf(1)/2 - mpf(10)**-300))`).  At the precision
/// cap its value was rounding noise near `10⁴⁰` with an error ball of
/// `2³⁹⁶`, which contains 0, and a value whose ball contains 0 was returned
/// as 0.  A zero ball now also has to shrink with the precision, as a true
/// zero's does; near a pole it grows, and the value is refused.
#[test]
fn a_growing_error_ball_is_not_a_zero() {
    let ctx = Context::new();
    let e = ctx.parse("tan(pi*(1/2 - 10^-300))").unwrap();
    assert!(
        matches!(e.eval_f64(), Err(SymplexError::PrecisionExhausted { .. })),
        "{:?}",
        e.eval_f64()
    );
    // Within reach of the precision the value is certified: mpmath (dps 60)
    //   tan(pi*(mpf(1)/2 - mpf(10)**-30)) = 3.1830988618379067154e+29
    let e = ctx.parse("tan(pi*(1/2 - 10^-30))").unwrap();
    assert_eq!(e.eval_decimal(20).unwrap(), "3.1830988618379067154e29");
    // True zeros are still zeros.
    for s in ["sin(pi)", "sqrt(2)^2 - 2"] {
        assert_eq!(ctx.parse(s).unwrap().eval_f64().unwrap(), 0.0, "{s}");
    }
}
