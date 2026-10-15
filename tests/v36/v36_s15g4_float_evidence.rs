//! Tests of session 15 (s15g4): float evidence in calculus decisions.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

/// `∫ₐᵇ f dx`: `Ok` with the value to 30 digits, or the error.
fn definite(src: &str, lo: &str, hi: &str) -> Result<(String, String), SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse(src).unwrap();
    let (a, b) = (ctx.parse(lo).unwrap(), ctx.parse(hi).unwrap());
    let v = f.try_integrate_definite(&x, &a, &b)?;
    let d = v.eval_decimal(30).unwrap_or_default();
    Ok((v.to_string(), d))
}

/// The value of `∫ₐᵇ f dx` agrees with the oracle `re + im·i` to `1e-13`
/// (relatively), or the integral is refused, but is never `Divergent`.
fn refused_or(src: &str, lo: &str, hi: &str, re: f64, im: f64) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse(src).unwrap();
    let (a, b) = (ctx.parse(lo).unwrap(), ctx.parse(hi).unwrap());
    match f.try_integrate_definite(&x, &a, &b) {
        Ok(v) => {
            let z = v.eval_complex64().unwrap();
            let scale = re.hypot(im).max(1.0);
            assert!(
                (z.re - re).abs() <= 1e-13 * scale && (z.im - im).abs() <= 1e-13 * scale,
                "∫ {src} on [{lo}, {hi}] = {v} = {z}, the oracle is {re} + {im}i"
            );
        }
        Err(e) => assert!(
            !matches!(e, SymplexError::Divergent { .. }),
            "∫ {src} on [{lo}, {hi}] converges: {e}"
        ),
    }
}

/// The `ln²u` coefficients of an antiderivative at infinity were taken
/// for 0 when their sum was below `10⁻²⁵` at 30 digits
/// (`definite::log_polylog_limit`), and the quadrature check does not run
/// on an infinite range: up to 0.52 `∫₀^∞ (ln(x + 1)/(x² + 1) +
/// e⁻⁶⁰·ln(x + 1)/(x + 2)) dx` was `1.46036211675311954767977573229`, and
/// with `10⁻³⁰` for `e⁻⁶⁰` `…573949`; both diverge.  The sum is now decided
/// by `hidden_zero` (proved 0, or no finite value).
///
/// Oracle: `ln(x + 1)/(x + 2) ≥ ln(x + 1)/(2x)` for `x ≥ 2` and `∫ ln x/x dx =
/// ln²x/2 → ∞` (the coefficient of `ln²u` in the antiderivative is `c/2`
/// for the coefficient `c` of the second term); the convergent part alone
/// is `π·ln 2/4 + G = 1.460362116753119547679776` (mpmath, 30 and 60
/// digits), which is kept.
#[test]
fn a_tiny_ln_squared_coefficient_at_infinity_is_not_zero() {
    for c in ["exp(-60)", "10^-30"] {
        let src = format!("log(x+1)/(x^2+1) + {c}*log(x+1)/(x+2)");
        if let Ok((v, d)) = definite(&src, "0", "oo") {
            panic!("∫₀^∞ {src} dx diverges, not {v} = {d}");
        }
    }
    let (_, d) = definite("log(x+1)/(x^2+1)", "0", "oo").unwrap();
    assert!(d.starts_with("1.4603621167531195476797757"), "{d}");
}

/// A branch cut crossed at a sample point of the definite integrator's
/// cut check (`definite::numeric_cut_crossing`, `x = 1/2` on `[0, 1]`) was
/// skipped: the sample's imaginary part is 0, and the sign change was only
/// looked for between two nonzero samples.  The integrand itself (the
/// principal `ln`, `√`) jumps there, and up to 0.52 the Newton–Leibniz
/// difference of its principal antiderivative was returned:
/// `∫₀¹ ln((1 + i)x − 1 − i/2) dx = −2.1518 − 1.8169i`, `∫₀¹ √(…) dx =
/// −0.0500 − 0.3884i`, `∫₀¹ 1/√(…) dx = 1.7862 + 1.2720i`.  The side of the
/// axis is now certified at every sample, a sample on the cut included.
///
/// Oracle: mpmath `quad` split at `1/2` (30 and 60 digits):
/// `−0.5809969794482319850805529 − 0.2461431140023821144722332i`,
/// `0.1856550760620529068191657 − 0.1526628524020776942954525i`,
/// `0.3719378153843282372678699 − 0.1421939128590260845492663i`.
#[test]
fn a_cut_crossed_at_a_sample_point_is_seen() {
    refused_or(
        "log((1+I)*x - 1 - I/2)",
        "0",
        "1",
        -0.580_996_979_448_232,
        -0.246_143_114_002_382,
    );
    refused_or(
        "sqrt((1+I)*x - 1 - I/2)",
        "0",
        "1",
        0.185_655_076_062_053,
        -0.152_662_852_402_078,
    );
    refused_or(
        "1/sqrt((1+I)*x - 1 - I/2)",
        "0",
        "1",
        0.371_937_815_384_328,
        -0.142_193_912_859_026,
    );
}

/// The same crossing by an imaginary part far below the `10⁻¹²` that
/// counted as 0 (`ε·(x − 1/2)`, `ε = 10⁻²⁰` or `e⁻⁵⁰`): the samples read
/// it as 0 and saw no crossing.  These were refused at 0.52 for want of an
/// antiderivative; with one, the principal difference (`3i` for `1/√u`)
/// would have been returned.
///
/// Oracle (to `O(ε)`): below the axis for `x < 1/2`, above it after, so
/// `∫₀^{3/4} 1/√u dx = i·(3 − 2√2)`, `∫₀^{3/4} ln u dx = ln 4/4 − 3/4 −
/// iπ/4`, `∫₀^{3/4} √u dx = i·(√2/3 − 3/4)`.
#[test]
fn a_cut_crossed_by_a_tiny_imaginary_part_is_seen() {
    refused_or(
        "1/sqrt((1+I*10^-20)*x - 1 - I*10^-20/2)",
        "0",
        "3/4",
        0.0,
        0.171_572_875_253_810,
    );
    refused_or(
        "log((1+I*10^-20)*x - 1 - I*10^-20/2)",
        "0",
        "3/4",
        -0.403_426_409_720_027,
        -std::f64::consts::FRAC_PI_4,
    );
    refused_or(
        "sqrt((1+I*exp(-50))*x - 1 - I*exp(-50)/2)",
        "0",
        "3/4",
        0.0,
        -0.278_595_479_208_968,
    );
}

/// `Ei(p·x + q)` crosses its cut where `im(p·x + q) = 0`; the rewrite
/// `Ei(z) − ln z + ln(x − c)` that keeps the antiderivative continuous was
/// skipped when the crossing was within `10⁻¹²` left of the branch point
/// (`integrate::ei_continuous_on_real_line`).  Up to 0.52
/// `∫₀² e^{(1+i)x}/(x − 1 + 10⁻¹⁴i) dx` was `−9.6453 + 11.8945i`, the
/// value for the pole on the other side of the axis.  The side is now
/// certified.
///
/// Oracle: mpmath, `∫₀² (e^{px} − e^{pc})/(x − c) dx + e^{pc}·(ln(2 − c) −
/// ln(−c))` with `p = 1 + i`, `c = 1 − 10⁻¹⁴i` (30 and 60 digits):
/// `4.726548606616848126120955 + 2.666462914077077354375639i`; for `c = 1 +
/// 10⁻¹⁴i` it is `−9.645328526084759990314832 + 11.89453909810011206078869i`,
/// which is kept.
#[test]
fn ei_crossing_just_left_of_its_branch_point() {
    refused_or(
        "exp((1+I)*x)/(x-1+I*10^-14)",
        "0",
        "2",
        4.726_548_606_616_85,
        2.666_462_914_077_08,
    );
    let (_, d) = definite("exp((1+I)*x)/(x-1-I*10^-14)", "0", "2").unwrap();
    assert!(d.starts_with("-9.6453285260847599903"), "{d}");
    assert!(d.contains("+ 11.894539098100112060"), "{d}");
}

/// A constant whose imaginary part is below the 16 digits `evalf` prints
/// was read as real (`calculus_util::expr_to_f64`): the pole `1 + i·e⁻⁵⁰`
/// of `1/(x − 1 − i·e⁻⁵⁰)` became the real point `1` of `[0, 2]`, and up to
/// 0.52 the integral was `Divergent` ("the antiderivative tends to −∞ as x
/// → exp(−50)·I + 1"); so was `∫₀¹ dx/(e^{(1 + 10⁻²⁰i)x} − 2)`, whose pole
/// is `ln 2/(1 + 10⁻²⁰i)`.  A part `evalf` keeps next to a much larger one
/// has a certified sign, and such a constant is no longer real.
///
/// Oracle: the path keeps off the poles, so the integrals are finite:
/// `ln(1 − i·e⁻⁵⁰) − ln(−1 − i·e⁻⁵⁰) = iπ` to 21 digits; mpmath (the
/// residue part split off) `−0.6654466341020272672 − 1.5707963267948966184i`.
#[test]
fn a_pole_off_the_real_line_by_a_tiny_amount_is_not_on_it() {
    refused_or(
        "1/(x - 1 - I*exp(-50))",
        "0",
        "2",
        0.0,
        std::f64::consts::PI,
    );
    refused_or(
        "1/(x - 1/(1+I*exp(-50)))",
        "0",
        "2",
        0.0,
        -std::f64::consts::PI,
    );
    refused_or(
        "1/(exp((1+I*10^-20)*x) - 2)",
        "0",
        "1",
        -0.665_446_634_102_027,
        // `−π/2 + O(10⁻²⁰)`
        -std::f64::consts::FRAC_PI_2,
    );
}

/// Real roots of the argument of `sign`, `H`, `|·|` closer than `10⁻¹²` to
/// each other were merged (`integrate::try_piecewise_split`): up to 0.52
/// `∫ sign((x − 1)(x − 1 − 10⁻¹⁵)) dx` was `Piecewise((x, x < 1), (x,
/// True))`, missing the interval `(1, 1 + 10⁻¹⁵)` where the sign is `−1`,
/// and so did the `Heaviside` version.  Roots are now compared exactly or
/// by the certified sign of their difference.
///
/// Oracle (exact): `∫₀² sign(…) dx = 2 − 2·10⁻¹⁵` and `∫₀² H(…) dx = 2 −
/// 10⁻¹⁵`.
#[test]
fn roots_closer_than_the_old_tolerance_stay_distinct() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (src, want) in [
        (
            "sign((x-1)*(x-1-10^-15))",
            "999999999999999/500000000000000",
        ),
        (
            "Heaviside((x-1)*(x-1-10^-15))",
            "1999999999999999/1000000000000000",
        ),
    ] {
        let f = ctx.parse(src).unwrap();
        let big_f = f.try_integrate(&x).unwrap();
        let at = |t: i64| big_f.subs(&x, &ctx.int(t));
        let v = (&at(2) - &at(0)).simplify();
        assert_eq!(v.to_string(), want, "∫ {src} dx = {big_f}");
    }
}

/// A sign change of a watched argument exactly at a sample of the
/// definite integrator's guard (`definite::sign_changes_explained`) was
/// skipped, so the guard passed with an incomplete scan.  Up to 0.52
/// `∫₀¹ sign(sin x + x − sin(1/2) − 1/2) dx` came out `0` only because
/// the zero is the midpoint (the single piece was resolved at the zero
/// itself), and `∫₀¹ |sin x + x − sin(1/2) − 1/2| dx` was stopped only by
/// the quadrature check.  The guard now compares the samples around a
/// zero sample.
///
/// Oracle: mpmath `quad` split at `1/2`: `0.4648628179126057148316266`.
#[test]
fn a_sign_change_at_a_guard_sample_is_seen() {
    refused_or(
        "abs(sin(x) + x - sin(1/2) - 1/2)",
        "0",
        "1",
        0.464_862_817_912_606,
        0.0,
    );
    refused_or("sign(sin(x) + x - sin(1/2) - 1/2)", "0", "1", 0.0, 0.0);
}
