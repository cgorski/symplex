//! Audit round 4: the two engines for the regularised incomplete gamma and
//! beta functions — `symplex::stats::numdist` (`f64`) and `evalf`
//! (arbitrary precision, reached through `Ex`) — cross-checked with mpmath
//! as referee.
//!
//! 1. evalf printed wrong digits as certified ones.  Its error bounds take
//!    a special function's value as exact to the working precision, but
//!    `betainc_regularized` formed `1 − I_{1−x}(b, a)` above the switch
//!    point of its continued fraction assuming at most a bit cancels — for a
//!    small second shape all of them do (`I(⅓, 10⁻¹⁰⁰; 9/10)` came out
//!    `1.3·10⁻⁸²`, truly `4.8·10⁻¹⁰⁰`) —, `uppergamma` formed `Γ(s) − γ(s, x)`
//!    with guard bits for `x` instead of for the cancellation `log₂(1/Q)`
//!    (`uppergamma(10⁻¹⁰⁰, ½) = 0.5649…`, truly `0.5597…`; and `10⁷` guard
//!    bits, a hang, at `x ≈ 7·10⁶`), and the recurrence for `s ≤ 0` reserved
//!    16 bits a step and took an `s` within `10⁻¹²` of an integer for it
//!    (`expint(3 + 10⁻¹⁴, ⅓)` wrong from the 14th digit).  The two-limit
//!    `betainc` returned its third attempt whatever it had lost (a negative
//!    `−8.6·10⁻²⁵⁶` for a positive tail).
//! 2. evalf refused values that exist: `B(a, b)` as `Γ(a)Γ(b)/Γ(a + b)`
//!    overflows the exponent range for shapes near `10⁸`; `Γ(10⁻⁴⁰⁰)` was
//!    taken for the pole at 0 (its `f64` image is 0).
//! 3. numdist formed `(a + b)·x − a` in plain doubles (an error `ε·a`,
//!    `3·10⁻⁸` of a tail 31 standard deviations out at shapes `10¹⁵`, where
//!    its documentation promised `≈ 1e-12`), and Loader's `bd0` switched to
//!    its cancelling direct formula at `|v| = 0.1` (`10⁻¹²` in far gamma
//!    tails, against a documented `1e-15`).
//!
//! Reference values: mpmath 1.3 (`symplex/.venv`), every value at two
//! precisions that agree (the call and both `dps` are cited); where
//! mpmath's hypergeometric route raises `NoConvergence` (both shapes near
//! `10⁸`, or `10¹⁴`), the positive-term series
//! `I_x(a, b) = xᵃ(1−x)ᵇ/(a B(a, b))·Σ_k (a+b)_k/(a+1)_k xᵏ` or
//! `P(s, x) = xˢe⁻ˣ/Γ(s+1)·Σ_k xᵏ/((s+1)…(s+k))`, or Gauss–Legendre
//! quadrature of the density in its standardised variable, in mpmath at the
//! cited precisions.  The differential sweeps behind these (4 600 `f64`
//! parameter sets with shapes from `10⁻³⁰⁰` to `10¹⁶`, 480 exact-rational
//! evalf cases at 50 digits, 120 at 16/30/100 digits, 600 numdist-vs-evalf
//! pairs) ran from `examples/` probes during the session and are not part
//! of the suite.

// Reference values are quoted at the digits mpmath printed them.
#![allow(clippy::excessive_precision)]

use num_bigint::BigInt;
use num_traits::{Signed, Zero};
use std::time::{Duration, Instant};
use symplex::prelude::*;
use symplex::stats::numdist::{self, gammainc_upper_regularized_f64};

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

/// `|printed − reference|` is at most one unit of the last digit printed
/// (of the `digits`-th significant one where the printer pads an integer
/// with zeros): evalf certified the digits it printed.  (The reference
/// carries more digits than are printed.)
fn assert_certified(printed: &str, digits: u32, reference: &str, label: &str) {
    let (m1, e1) = decimal(printed);
    let (m2, e2) = decimal(reference);
    assert!(!m1.is_zero(), "{label}: printed 0 for {reference}");
    let e0 = e1.min(e2);
    let a = &m1 * pow10(e1 - e0);
    let b = &m2 * pow10(e2 - e0);
    let lead = e1 + i64::try_from(m1.abs().to_string().len()).expect("length") - 1;
    let ulp = (lead - i64::from(digits) + 1).max(e1);
    assert!(
        ulp >= e0,
        "{label}: the reference {reference} has too few digits"
    );
    assert!(
        (a - b).abs() <= pow10(ulp - e0),
        "{label}: printed {printed}, mpmath {reference}"
    );
}

/// `src` evaluated to `digits` digits agrees with `reference` to the
/// digits printed.
fn certified(ctx: &Context, src: &str, digits: u32, reference: &str) {
    let e = ctx.parse(src).expect("parse");
    let s = e
        .eval_decimal(digits)
        .unwrap_or_else(|err| panic!("{src} at {digits} digits: {err}"));
    assert_certified(&s, digits, reference, &format!("{src} at {digits} digits"));
}

/// `eval_f64` of `src` within `rel` of `expected`.
fn f64_close(ctx: &Context, src: &str, expected: f64, rel: f64) {
    let v = ctx
        .parse(src)
        .expect("parse")
        .eval_f64()
        .unwrap_or_else(|err| panic!("{src}: {err}"));
    assert!(
        (v - expected).abs() <= rel * expected.abs(),
        "{src}: eval_f64 {v:e}, expected {expected:e}"
    );
}

/// `actual` within `rel` of `expected`, relatively.
fn close(actual: f64, expected: f64, rel: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= rel * expected.abs(),
        "{label}: got {actual:e}, expected {expected:e} (rel {:.1e})",
        (actual / expected - 1.0).abs()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// evalf: certified digits
// ═══════════════════════════════════════════════════════════════════════════

/// `Γ(s, x) = Γ(s) − γ(s, x)` for a tiny `s` cancels `log₂(1/s)` bits
/// (`Q(s, x) ≈ s·E₁(x)`), which 0.28 did not add: `uppergamma(10⁻¹⁰⁰, ½)`
/// printed `0.56498279283493557656…` at 50 digits (`1686384335557091328` at
/// 30) and `eval_f64` gave `3.2·10²⁸`; the regularised
/// `uppergamma(10⁻⁶⁰, ½)/gamma(10⁻⁶⁰)` had its last 6 of 50 digits wrong and
/// `eval_f64` `5.59773594923496e-61`.
#[test]
fn uppergamma_of_a_tiny_shape_is_certified() {
    let ctx = Context::new();
    // mpmath: gammainc(mpf(10)**-100, mpf(1)/2, inf) at dps 60 and 120
    let e1_half = "0.559773594776160811746795939315085235226846890316353515248";
    certified(&ctx, "uppergamma(10^-100, 1/2)", 50, e1_half);
    certified(&ctx, "uppergamma(10^-100, 1/2)", 30, e1_half);
    f64_close(
        &ctx,
        "uppergamma(10^-100, 1/2)",
        0.559773594776160811746795939315,
        4e-16,
    );
    // mpmath: gammainc(mpf(10)**-60, mpf(1)/2, inf, regularized=True) at dps 60 and 120
    let q = "5.59773594776160811746795939315085235226846890316353515248e-61";
    certified(&ctx, "uppergamma(10^-60, 1/2)/gamma(10^-60)", 50, q);
    f64_close(
        &ctx,
        "uppergamma(10^-60, 1/2)/gamma(10^-60)",
        5.59773594776160811746795939315e-61,
        4e-16,
    );
}

/// The downward recurrence for `s ≤ 0` loses `log₂(1/(1 − σ))` bits in its
/// first step when `s` sits just below an integer, and 0.28 reserved 16
/// bits per step; it also took an `s` within `10⁻¹²` of an integer for that
/// integer.  Before: `uppergamma(−2 + 10⁻¹⁵, ½)` printed
/// `0.886417457100713829477…` (wrong from the 16th digit),
/// `expint(3 + 10⁻¹⁴, ⅓)` `0.284893089377292544500…` (from the 14th) and
/// `expint(1 + 10⁻²⁰, ⅓)` `0.828887745348586636111100…` (from the 21st:
/// `E₁(⅓)`).
#[test]
fn uppergamma_just_below_an_integer_is_certified() {
    let ctx = Context::new();
    // mpmath: gammainc(-2 + mpf(10)**-15, mpf(1)/2, inf) at dps 40 and 80
    certified(
        &ctx,
        "uppergamma(-2 + 10^-15, 1/2)",
        30,
        "0.8864174571007135107618991181373214675",
    );
    // mpmath: expint(3 + mpf(10)**-14, mpf(1)/3) at dps 40 and 80
    certified(
        &ctx,
        "expint(3 + 10^-14, 1/3)",
        30,
        "0.284893089377294629022776655945322135",
    );
    // mpmath: expint(1 + mpf(10)**-20, mpf(1)/3) at dps 60 and 120
    certified(
        &ctx,
        "expint(1 + 10^-20, 1/3)",
        50,
        "0.828887745348586636113824033491948718553443365912009484975",
    );
}

/// Above the switch point `(a+1)/(a+b+2)` 0.28 formed `1 − I_{1−x}(b, a)`
/// "losing at most a bit"; for a small `b` the subtracted value is
/// `1 − O(b)`.  Before, at 50 digits: `4.7877819203188910848167778…e-60`
/// (20 digits right) and `1.2765780787981217…e-82` for `4.8·10⁻¹⁰⁰`;
/// `eval_f64` gave `4.97·10⁻⁵¹` for both.
#[test]
fn betainc_with_a_tiny_second_shape_is_certified() {
    let ctx = Context::new();
    // mpmath: betainc(mpf(1)/3, mpf(10)**-60, 0, mpf(9)/10, regularized=True) at dps 60 and 120
    let r60 = "4.78778192031889108481665020076804197532296004895304641609e-60";
    certified(&ctx, "betainc_regularized(1/3, 10^-60, 0, 9/10)", 50, r60);
    f64_close(
        &ctx,
        "betainc_regularized(1/3, 10^-60, 0, 9/10)",
        4.78778192031889108481665020076804e-60,
        4e-16,
    );
    // mpmath: betainc(mpf(1)/3, mpf(10)**-100, 0, mpf(9)/10, regularized=True) at dps 60 and 120
    let r100 = "4.78778192031889108481665020076804197532296004895304641609e-100";
    certified(&ctx, "betainc_regularized(1/3, 10^-100, 0, 9/10)", 50, r100);
    certified(&ctx, "betainc_regularized(1/3, 10^-100, 0, 9/10)", 16, r100);
}

/// The upper tail of a tiny first shape, `1 − I_{1/10}(10⁻³⁰⁰, 1) =
/// 1 − 10^{−10⁻³⁰⁰}`: every digit cancels, and 0.28's two-limit loop
/// returned its third attempt regardless — `0` at 30 digits,
/// `−8.56·10⁻²⁵⁶` (a negative probability) at 50, `3.7·10⁻¹⁶⁰` from
/// `eval_f64`.  Now each tail is computed to its own relative accuracy and
/// the upper one is a single tail, not a difference.
#[test]
fn betainc_upper_tail_of_a_tiny_first_shape() {
    let ctx = Context::new();
    // mpmath: betainc(1, mpf(10)**-300, 0, mpf(9)/10, regularized=True) at dps 60 and 120
    // (= -expm1(mpf(10)**-300 * log(mpf(1)/10)), the same digits)
    let r = "2.30258509299404568401799145468436420760110148862877297603e-300";
    certified(&ctx, "betainc_regularized(10^-300, 1, 1/10, 1)", 30, r);
    certified(&ctx, "betainc_regularized(10^-300, 1, 1/10, 1)", 50, r);
    f64_close(
        &ctx,
        "betainc_regularized(10^-300, 1, 1/10, 1)",
        2.30258509299404568401799145468e-300,
        4e-16,
    );
    // Both tiny shapes: I_{1/3}(10⁻³⁰, 10⁻³⁰) = ½ − 3.47·10⁻³¹.
    // mpmath: betainc(mpf(10)**-30, mpf(10)**-30, 0, mpf(1)/3, regularized=True) at dps 60 and 120
    certified(
        &ctx,
        "betainc_regularized(10^-30, 10^-30, 0, 1/3)",
        50,
        "0.499999999999999999999999999999653426409720027345291383939",
    );
    // Close limits near 0 with a tiny first shape.
    // mpmath: betainc(mpf(10)**-20, mpf(1)/7, mpf(10)**-12, mpf(10)**-10, regularized=True) at dps 60 and 120
    certified(
        &ctx,
        "betainc_regularized(10^-20, 1/7, 10^-12, 10^-10)",
        50,
        "4.60517018607294850941814531714996368852054993713586079716e-20",
    );
}

/// A cancellation beyond `4·wp + 1024` bits (a shape of `10⁻¹⁰⁰⁰`) is
/// refused as `PrecisionExhausted`, never printed: before, the same
/// cancellation printed garbage for `10⁻¹⁰⁰` (above), and for `10⁻¹⁰⁰⁰`
/// `Γ` was refused as a pole.  Where no cancellation arises the tiny shape
/// is simply evaluated (`Γ(10⁻¹⁰⁰⁰, ½) = E₁(½)` to far beyond 50 digits).
#[test]
fn a_cancellation_beyond_the_cap_is_refused() {
    let ctx = Context::new();
    let e = ctx
        .parse("betainc_regularized(1/3, 10^-1000, 0, 9/10)")
        .expect("parse");
    for digits in [16, 50] {
        match e.eval_decimal(digits) {
            Err(SymplexError::PrecisionExhausted { .. }) => {}
            other => panic!("expected a refusal at {digits} digits, got {other:?}"),
        }
    }
    // mpmath: e1(mpf(1)/2) at dps 60 and 120 (Γ(10⁻¹⁰⁰⁰, ½) differs by 10⁻¹⁰⁰⁰·O(1))
    certified(
        &ctx,
        "uppergamma(10^-1000, 1/2)",
        50,
        "0.559773594776160811746795939315085235226846890316353515248",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// evalf: refusals where a value exists, and a hang
// ═══════════════════════════════════════════════════════════════════════════

/// `B(a, b) = Γ(a)Γ(b)/Γ(a + b)` overflows the exponent range for shapes
/// near `10⁸` (`Γ(10⁸) ≈ 2^{2.5·10⁹}`), so 0.28 refused
/// `betainc_regularized(10⁸ + ⅓, 10⁸ + 1/7, 0, ½)` (`PrecisionExhausted`) and
/// the unregularised `betainc` of the same shapes, and
/// `I_{1−10⁻³⁰}(10⁸ + ⅓, 10⁻³⁰)`.  The prefactor is now one exponential of
/// `a ln x + b ln y − ln a − ln B(a, b)`.
#[test]
fn betainc_with_shapes_near_1e8_is_evaluated() {
    let ctx = Context::new();
    // mpmath 1.3 at dps 60 and 120, the positive-term series
    // exp(a*log(x) + b*log(1-x) - log(a) - (loggamma(a) + loggamma(b) - loggamma(a+b)))
    //   * sum_k rf(a+b, k)/rf(a+1, k) x**k   (betainc raises NoConvergence here);
    // Gauss–Legendre quadrature of the density at dps 60 and 80 gives the same digits.
    certified(
        &ctx,
        "betainc_regularized(10^8+1/3, 10^8+1/7, 0, 1/2)",
        50,
        "0.499994626765866338675245041845788585819780753069840613958",
    );
    // The same series times exp(loggamma(a) + loggamma(b) - loggamma(a+b)), dps 40 and 80.
    certified(
        &ctx,
        "betainc(10^8+1/3, 10^8+1/7, 0, 1/2)",
        30,
        "9.384830068263748616970094494308856509e-60206004",
    );
    // mpmath: betainc(mpf(10)**-30, mpf(10)**8+mpf(1)/3, mpf(10)**-30, 1, regularized=True)
    // at dps 110 (dps 55 cancels inside mpmath) — the same tail as I_{1-1e-30}(1e8+1/3, 1e-30)
    certified(
        &ctx,
        "betainc_regularized(10^8+1/3, 10^-30, 0, 1-10^-30)",
        50,
        "5.007965638263413885167828878826298986150914381537289064e-29",
    );
}

/// `Γ(10⁻⁴⁰⁰)` was refused as the pole at 0 (the `f64` image of `10⁻⁴⁰⁰` is
/// 0), and with it every regularised incomplete gamma of such a shape.
#[test]
fn gamma_of_a_shape_below_the_f64_range() {
    let ctx = Context::new();
    // mpmath: gammainc(mpf(10)**-400, 3, inf, regularized=True) at dps 40 and 80
    certified(
        &ctx,
        "uppergamma(10^-400, 3)/gamma(10^-400)",
        30,
        "1.304838109419703741250074582864502295e-402",
    );
    let g = ctx
        .parse("gamma(10^-400)")
        .expect("parse")
        .eval_decimal(30)
        .expect("gamma(10^-400)");
    // mpmath: gamma(mpf(10)**-400) at dps 80 and 160 prints 1.0e+400 (1/ε − γ + O(ε)).
    assert_certified(
        &g,
        30,
        "1.0000000000000000000000000000000000000000e400",
        "gamma(10^-400)",
    );
}

/// For `x < s + 1` 0.28 evaluated `Γ(s) − γ(s, x)` with `1.44·x` guard
/// bits — ten million at `x ≈ 7·10⁶`, which never finished (and 4300 bits,
/// 6.5 s, at `x ≈ 3000`).  Now the guard is the bounded cancellation
/// `log₂(1/min(1, s)) + 5`, and the evaluation costs about what the series
/// of `γ(s, x)` alone costs (compared interleaved, so that machine load
/// affects both sides).
#[test]
fn uppergamma_below_the_mean_of_a_large_shape_does_not_hang() {
    let ctx = Context::new();
    let upper = ctx
        .parse("uppergamma(661000000/97, 3276025437403/484030)")
        .expect("parse");
    let lower = ctx
        .parse("lowergamma(661000000/97, 3276025437403/484030)")
        .expect("parse");
    // mpmath 1.3 at dps 40 and 80: gamma(s)*(1 - P) with the positive series
    // P = x**s e**-x / gamma(s+1) * sum_k x**k/rf(s+1, k) (= 1.0290183207148540406598e-70;
    // gammainc raises NoConvergence here)
    let reference = "1.949490703618700374867759201318328652e+43606475";
    let (mut t_upper, mut t_lower) = (Duration::ZERO, Duration::ZERO);
    for _ in 0..3 {
        let t = Instant::now();
        let s = upper.eval_decimal(30).expect("uppergamma");
        t_upper += t.elapsed();
        assert_certified(&s, 30, reference, "uppergamma(6.8e6, 6.77e6)");
        let t = Instant::now();
        let _ = lower.eval_decimal(30).expect("lowergamma");
        t_lower += t.elapsed();
    }
    assert!(
        t_upper < t_lower * 30 + Duration::from_millis(200),
        "uppergamma {t_upper:?} against lowergamma {t_lower:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// numdist
// ═══════════════════════════════════════════════════════════════════════════

/// `(a + b)·x − a` was formed in plain doubles: an absolute error `ε·a`
/// that the tail's exponent (`≈ λ²/2a`) turns into a relative error `ε·|λ|`.
/// `beta::sf(0.1066…, 3.7·10¹⁴, 3.1·10¹⁵)` (31 standard deviations out)
/// was `1.5453357538008636e-212` (`3.3·10⁻⁸` off, against the `≈ 1e-12` the
/// module documented for large shapes), `beta::sf(0.5347, 3.6·10⁷, 3.2·10⁷)`
/// `1.4863679730141863e-48` (`1.1·10⁻¹¹`).  Now the difference is exact and
/// what remains is the rounding of the tail's logarithm, `≲ 8·ε·|ln P|`.
#[test]
fn numdist_beta_far_tail_of_large_shapes() {
    // mpmath 1.3 at dps 35 and 66: Gauss–Legendre quadrature of the Beta density in
    // its standardised variable (betainc raises NoConvergence); symplex evalf,
    // betainc_regularized of the exact doubles, gives the same 20 digits.
    let p = 1.5453358049401412078594784e-212;
    let got = numdist::beta::sf(0.10661675365881029, 371068858888737.94, 3109335601544169.0);
    close(
        got,
        p,
        8.0 * f64::EPSILON * 488.0,
        "beta::sf(0.1066, 3.7e14, 3.1e15)",
    );
    // mpmath 1.3 at dps 35 and 66: the positive-term series of I_{1-x}(b, a) (and the
    // quadrature, the same digits).
    let p = 1.486367972998267970377776e-48;
    let got = numdist::beta::sf(0.534691970026874, 36154692.055912495, 31575166.46173772);
    close(
        got,
        p,
        8.0 * f64::EPSILON * 110.0,
        "beta::sf(0.5347, 3.6e7, 3.2e7)",
    );
    // The lower tail through the same difference (x below the mean).
    let got = numdist::beta::cdf(
        1.0 - 0.534691970026874,
        31575166.46173772,
        36154692.055912495,
    );
    close(got, p, 8.0 * f64::EPSILON * 110.0, "beta::cdf mirror");
}

/// Loader's `bd0(k, m) = k ln(k/m) + m − k` switched from its series to the
/// direct formula at `|v| = |k − m|/(k + m) = 0.1`, where the direct formula
/// still cancels a factor `1/v`: `Q(24128.47, 30411.25) = 2.7·10⁻³⁰⁶` (the
/// exponent 698.6 from `−5584 + 6283`) was `2.733303614893822e-306`,
/// `10⁻¹²` off against the documented `1e-15`.  The series is used to
/// `|v| = ½` now.
#[test]
fn numdist_gamma_far_tail_uses_the_series_of_bd0() {
    // mpmath: gammainc(a, x, inf, regularized=True) at dps 40 and 80 (and Legendre's
    // continued fraction, the same digits)
    let q = 2.733303614891100805115772803157899352e-306;
    let got = gammainc_upper_regularized_f64(24128.473518962626, 30411.252002906596);
    close(got, q, 5.0 * f64::EPSILON * 704.0, "Q(24128.47, 30411.25)");
}
