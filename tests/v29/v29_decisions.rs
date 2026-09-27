//! Decisions D1 and D2 of the 0.30 hand-off, and the evalf leftovers.
//!
//! * D1 — `loggamma` is SymPy's (mpmath's, Mathematica's `LogGamma`): the
//!   analytic continuation of `ln Γ` from `x > 0`, holomorphic off
//!   `(−∞, 0]`, `ln|Γ(x)| − iπ⌈−x⌉` on the cut.  It was the real `ln|Γ(x)|`
//!   left of 0 and refused off the real axis.  Compiled and emitted code is
//!   NaN left of 0; the real `ln|Γ|` is `ln(abs(gamma(x)))`, compiled to the
//!   overflow-safe `lgamma`.
//! * D2 — compiled and emitted `x^(p/q)` with odd `q` is NaN for `x < 0`
//!   (it was the real root, a different function from the principal power
//!   `evalf` computes); the real root is `real_root`.
//!
//! Every reference value cites the oracle call that produced it (mpmath
//! 1.3 at two precisions that agree, SymPy 1.14).

use num_bigint::BigInt;
use num_traits::Signed;
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

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

/// `(re, im)` of a printed value `a`, `a + b*i`, `a - b*i`, `b*i` (as
/// decimal strings; `"0"` for an absent part).
fn complex_parts(s: &str) -> (String, String) {
    let s = s.trim();
    let im_of = |t: &str| t.strip_suffix("*i").unwrap_or(t).to_string();
    for (sep, neg) in [(" + ", false), (" - ", true)] {
        if let Some(i) = s.rfind(sep) {
            let im = im_of(&s[i + sep.len()..]);
            return (s[..i].to_string(), if neg { format!("-{im}") } else { im });
        }
    }
    if s.ends_with("*i") {
        ("0".to_string(), im_of(s))
    } else {
        (s.to_string(), "0".to_string())
    }
}

/// `log₁₀` of the magnitude of a decimal string's leading digit.
fn lead(s: &str) -> i64 {
    let (m, e) = decimal(s);
    if m == BigInt::from(0) {
        return i64::MIN / 4;
    }
    e + i64::try_from(m.abs().to_string().len()).expect("length") - 1
}

/// `src` evaluated to `digits` digits agrees with the complex reference
/// `(re, im)` (more digits) to one unit of the last digit printed, relative
/// to the larger part.
fn complex_certified(ctx: &Context, src: &str, digits: u32, re: &str, im: &str) {
    let printed = ctx
        .parse(src)
        .expect("parse")
        .eval_decimal(digits)
        .unwrap_or_else(|err| panic!("{src} at {digits} digits: {err}"));
    let (pre, pim) = complex_parts(&printed);
    let scale = lead(re).max(lead(im));
    let ulp = scale - i64::from(digits) + 1;
    for (got, want) in [(&pre, re), (&pim, im)] {
        let (m1, e1) = decimal(got);
        let (m2, e2) = decimal(want);
        let e0 = e1.min(e2).min(ulp);
        let p10 = |k: i64| BigInt::from(10u32).pow(u32::try_from(k).expect("power"));
        let a = &m1 * p10(e1 - e0);
        let b = &m2 * p10(e2 - e0);
        assert!(
            (a - b).abs() <= p10(ulp - e0),
            "{src} at {digits} digits: printed {printed}, mpmath {re} + {im}*i"
        );
    }
}

fn rel_close(got: f64, want: f64, tol: f64) -> bool {
    (got - want).abs() <= tol * want.abs().max(f64::MIN_POSITIVE)
}

// ═══════════════════════════════════════════════════════════════════════════
// D1 — loggamma is the analytic continuation
// ═══════════════════════════════════════════════════════════════════════════

/// Before: `loggamma(−5/2)` evaluated to the real `ln|Γ(−5/2)|` =
/// −0.0562…, `loggamma(−1/2)` to 1.2655…; SymPy's `N(loggamma(-5/2))`
/// (mpmath) is `ln|Γ| − 3πi`, the limit from above the cut, `−iπ⌈−x⌉`.
#[test]
fn loggamma_left_of_zero_is_the_continuation_from_above_the_cut() {
    let ctx = Context::new();
    let cases = [
        // mpmath: mp.dps=60 and 120; loggamma(mpf(-5)/2)
        (
            "loggamma(-5/2)",
            "-0.056243716497674050672594530097654284",
            "-9.4247779607693797153879301498385087",
        ),
        // mpmath: loggamma(mpf(-1)/2)
        (
            "loggamma(-1/2)",
            "1.2655121234846453964889457971347059",
            "-3.1415926535897932384626433832795029",
        ),
        // mpmath: loggamma(mpf(-7)/3)
        (
            "loggamma(-7/3)",
            "0.26678263097664871892178154178005528",
            "-9.4247779607693797153879301498385087",
        ),
    ];
    for (src, re, im) in cases {
        for d in [16, 30] {
            complex_certified(&ctx, src, d, re, im);
        }
    }
    // The principal ln(Γ(−5/2)) is a different branch: ln|Γ| + πi.
    complex_certified(
        &ctx,
        "ln(gamma(-5/2))",
        20,
        "-0.056243716497674050672594530097654284",
        "3.1415926535897932384626433832795029",
    );
    // Not real: `eval_f64` refuses it (it returned −0.0562…).
    assert!(ctx.parse("loggamma(-5/2)").unwrap().eval_f64().is_err());
    let v = ctx.parse("loggamma(5/2)").unwrap().eval_f64().unwrap();
    // mpmath: loggamma(mpf(5)/2)
    assert!(rel_close(v, 0.284_682_870_472_919_2, 1e-15), "{v}");
}

/// Before: every non-real argument was refused ("LogGamma of complex
/// argument not yet supported in evalf").
#[test]
fn loggamma_of_non_real_arguments() {
    let ctx = Context::new();
    // mpmath: mp.dps=60 and 120; loggamma(z) for the z shown
    let cases = [
        (
            "loggamma(I)",
            "-0.65092319930185633888521683150394767",
            "-1.872436647262429817118853349436648",
        ),
        (
            "loggamma(-5/2 + I/10)",
            "-0.10314924404281919776558998305351802",
            "-9.3144442683598381211326697733294024",
        ),
        (
            "loggamma(-5/2 - I/10)",
            "-0.10314924404281919776558998305351802",
            "9.3144442683598381211326697733294024",
        ),
        (
            "loggamma(1/3 + 2*I)",
            "-2.3365598102015009156764267951739905",
            "-0.8614313391401075240040146042510723",
        ),
        (
            "loggamma(10 + 100*I)",
            "-112.39736554967237892570698055002232",
            "374.98942296222949950761542077996698",
        ),
        (
            "loggamma(-100 + I/3)",
            "-362.81672994718079511878052001436642",
            "-314.19334045705490051207743588846015",
        ),
        (
            "loggamma(-1000000 - 1/3 + I)",
            "-12815524.294478256047528819041945236",
            "-3141581.457688022717353221888275096",
        ),
        (
            "loggamma(I*10^20)",
            "-157079632679489661945.23908156071093",
            "4505170185988091368035.197511205331",
        ),
    ];
    for (src, re, im) in cases {
        for d in [16, 30] {
            complex_certified(&ctx, src, d, re, im);
        }
    }
    // Next to the zero at 1 (the terms are 10³⁰ times the value).
    // mpmath: loggamma(1 + 1j/mpf(10)**30)
    complex_certified(
        &ctx,
        "loggamma(1 + I/10^30)",
        30,
        "-8.2246703342411321823620758332301259e-61",
        "-5.7721566490153286060651209008240243e-31",
    );
}

/// The value on the cut is the limit from the upper half-plane, and the
/// lower half-plane approaches its conjugate: `loggamma(−5/2 ± 10⁻⁴⁰i)`.
/// (A real `ln|Γ|` on the axis, as before, was the limit of neither.)
#[test]
fn loggamma_is_continuous_onto_the_cut_from_above() {
    let ctx = Context::new();
    let on = ctx
        .parse("loggamma(-5/2)")
        .unwrap()
        .eval_decimal(25)
        .unwrap();
    let above = ctx
        .parse("loggamma(-5/2 + I/10^40)")
        .unwrap()
        .eval_decimal(25)
        .unwrap();
    let below = ctx
        .parse("loggamma(-5/2 - I/10^40)")
        .unwrap()
        .eval_decimal(25)
        .unwrap();
    assert_eq!(on, above);
    let (re, im) = complex_parts(&on);
    let (bre, bim) = complex_parts(&below);
    assert_eq!(re, bre);
    assert_eq!(im.trim_start_matches('-'), bim.as_str(), "{on} vs {below}");
    // Complex conjugation commutes with loggamma off the cut, so a sum of a
    // conjugate pair is exactly real (mpmath: 2*loggamma(1/3+2j).real).
    let s = ctx
        .parse("loggamma(1/3 + 2*I) + loggamma(1/3 - 2*I)")
        .unwrap()
        .eval_decimal(20)
        .unwrap();
    assert_eq!(s, "-4.6731196204030018314");
}

/// The exact routes are SymPy's: `loggamma(−5/2)` stays unevaluated (a
/// real `ln|Γ|` closed form would now be wrong), the poles fold to `oo`,
/// `loggamma(1) = loggamma(2) = 0`, `loggamma(3) = log(2)`.
/// SymPy: `loggamma(Rational(-5, 2))`, `loggamma(0)`, `loggamma(-3)`,
/// `loggamma(3)`.
#[test]
fn loggamma_exact_values_are_sympys() {
    let ctx = Context::new();
    let show = |s: &str| format!("{}", ctx.parse(s).unwrap().eval());
    assert_eq!(show("loggamma(-5/2)"), "LogGamma(-5/2)");
    assert_eq!(show("loggamma(0)"), "oo");
    assert_eq!(show("loggamma(-3)"), "oo");
    assert_eq!(show("loggamma(1)"), "0");
    assert_eq!(show("loggamma(2)"), "0");
    assert_eq!(show("loggamma(3)"), "ln(2)");
    // d/dx loggamma = digamma everywhere off the cut.
    let x = ctx.symbol("x");
    assert_eq!(format!("{}", x.log_gamma().diff(&x)), "Digamma(x)");
}

/// Compiled `loggamma` is NaN left of 0 (the value is not real there; it
/// was `ln|Γ|`), +∞ at the poles (SymPy's `oo`), `lgamma` right of 0; the
/// real `ln|Γ(x)|` is `ln(abs(gamma(x)))`, compiled to `lgamma`, which
/// does not overflow at x = 200 (`Γ(200)` does: the naive formula was
/// `inf`) nor underflow at x = −401/2 (`-inf`).
#[test]
fn compiled_loggamma_and_ln_abs_gamma() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let lg = x.log_gamma().compile(&["x"]).unwrap();
    for p in [-2.5, -0.5, -1e-300, -1e6 - 0.5] {
        assert!(lg(&[p]).is_nan(), "loggamma({p}) = {}", lg(&[p]));
    }
    assert_eq!(lg(&[0.0]), f64::INFINITY);
    assert_eq!(lg(&[-3.0]), f64::INFINITY);
    // mpmath: loggamma(200)
    assert!(rel_close(lg(&[200.0]), 857.933_669_825_857_4, 1e-15));
    let lag = x.gamma().abs().ln();
    let f = lag.compile(&["x"]).unwrap();
    // mpmath: log(fabs(gamma(200))), log(fabs(gamma(mpf(-5)/2))),
    // log(fabs(gamma(mpf(-401)/2)))
    let cases = [
        (200.0, "200", 857.933_669_825_857_4),
        (-2.5, "-5/2", -0.056_243_716_497_674_05),
        (-200.5, "-401/2", -864.738_287_870_679_7),
    ];
    for (p, exact, want) in cases {
        let got = f(&[p]);
        assert!(rel_close(got, want, 1e-13), "ln|Γ({p})|: {got}");
        let e = lag.subs(&x, &ctx.parse(exact).unwrap()).eval_f64().unwrap();
        assert!(
            rel_close(got, e, 1e-13),
            "ln|Γ({p})|: compiled {got}, evalf {e}"
        );
    }
    // Also when CSE shares `gamma(x)` with another part of the expression.
    let shared = (&lag + &x.gamma().sign()).compile(&["x"]).unwrap();
    assert!(rel_close(shared(&[200.0]), 858.933_669_825_857_4, 1e-15));
    // The code generators: SymPy's loggamma and ln|Γ| are different calls.
    let rust = x.log_gamma().to_rust_fn("f", &["x"]).unwrap();
    assert!(rust.contains("symplex_rt::loggamma(x)"), "{rust}");
    let rust = lag.to_rust_fn("f", &["x"]).unwrap();
    assert!(rust.contains("symplex_rt::lgamma(x)"), "{rust}");
    let c = x.log_gamma().to_c_fn("f", &["x"]).unwrap();
    assert!(c.contains("symplex_loggamma(x)"), "{c}");
    let c = lag.to_c_fn("f", &["x"]).unwrap();
    assert!(c.contains("return lgamma(x);"), "{c}");
    assert_eq!(lag.to_python().unwrap(), "math.lgamma(x)");
    assert!(
        x.log_gamma()
            .to_python()
            .unwrap()
            .contains("math.lgamma(b) if b > 0 else math.inf if b % 1 == 0 else math.nan")
    );
    // A non-real constant is refused, like `asin(2)`.
    let k = ctx.parse("loggamma(-5/2)*x").unwrap();
    assert!(k.compile(&["x"]).is_err());
    assert!(k.to_rust_fn("f", &["x"]).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// D2 — no real-root exception in compiled and emitted code
// ═══════════════════════════════════════════════════════════════════════════

/// Before: `x^(1/3)` compiled to −2 at x = −8, `x^(3/5)` to −3.48,
/// `x^(2/3)` to 4; `evalf` gives the principal values, not real
/// (mpmath: `cbrt(mpc(-8))` = 1 + 1.732j).
#[test]
fn odd_denominator_powers_are_nan_for_negative_bases() {
    let ctx = Context::new();
    for src in [
        "x^(1/3)", "cbrt(x)", "x^(2/3)", "x^(3/5)", "x^(-1/3)", "x^(5/3)",
    ] {
        let e = ctx.parse(src).unwrap();
        let f = e.compile(&["x"]).unwrap();
        assert!(f(&[-8.0]).is_nan(), "{src} at -8: {}", f(&[-8.0]));
        let want = e.subs(&ctx.symbol("x"), &ctx.int(8)).eval_f64().unwrap();
        assert!(rel_close(f(&[8.0]), want, 4e-16), "{src} at 8");
        // evalf is not real at −8.
        let z = e
            .subs(&ctx.symbol("x"), &ctx.int(-8))
            .eval_complex64()
            .unwrap();
        assert!(z.im.abs() > 0.1, "{src}: {z}");
    }
    // The emitted code: no real-root idiom anywhere.
    let e = ctx.parse("x^(3/5) + y^(1/3)").unwrap();
    let rust = e.to_rust_fn("f", &["x", "y"]).unwrap();
    assert!(!rust.contains("abs()"), "{rust}");
    let c = e.to_c_fn("f", &["x", "y"]).unwrap();
    assert!(!c.contains("copysign") && !c.contains("fabs"), "{c}");
    let py = e.to_python().unwrap();
    assert!(!py.contains("copysign"), "{py}");
    assert!(!e.to_numpy().unwrap().contains("cbrt"));
    assert!(!e.to_julia().unwrap().contains("cbrt"));
}

/// `(−8)^(1/3)` is a non-real constant (`1 + √3·i`): refused by `compile`
/// and every emitter, as `asin(2)` was.  Before: compiled and emitted as −2.
#[test]
fn non_real_odd_roots_are_refused_as_constants() {
    let ctx = Context::new();
    let e = ctx.parse("(-8)^(1/3)*x").unwrap();
    assert!(e.compile(&["x"]).is_err());
    assert!(e.to_rust_fn("f", &["x"]).is_err());
    assert!(e.to_c_fn("f", &["x"]).is_err());
    assert!(e.to_python_fn("f", &["x"]).is_err());
    assert!(e.to_numpy_fn("f", &["x"]).is_err());
    assert!(e.to_julia_fn("f", &["x"]).is_err());
}

/// The real root is `real_root`: `sign(x)·|x|^(1/n)` (a `Piecewise` on
/// `im(x) = 0` for a symbol not known to be real, which the back ends now
/// evaluate: every value they compute is real).  Before, the `Piecewise`
/// form did not compile ("cannot compile `im`").
#[test]
fn real_root_compiles_and_emits_the_real_root() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = ctx.symbol_with("r", &[Assumption::Real]).unwrap();
    for (e, var) in [
        (x.real_root(3).unwrap(), "x"),
        (r.real_root(3).unwrap(), "r"),
    ] {
        let f = e.compile(&[var]).unwrap();
        assert_eq!(f(&[-8.0]), -2.0, "{e}");
        assert_eq!(f(&[8.0]), 2.0, "{e}");
        assert!(f(&[f64::NAN]).is_nan());
        assert!(e.to_rust_fn("f", &[var]).is_ok(), "{e}");
        assert!(e.to_c_fn("f", &[var]).is_ok(), "{e}");
        assert!(e.to_python().is_ok(), "{e}");
        assert!(e.to_julia().is_ok(), "{e}");
    }
    let f = x.real_root(5).unwrap().powi(3).compile(&["x"]).unwrap();
    assert!(rel_close(f(&[-32.0]), -8.0, 1e-15));
}

/// A Cardano formula with principal cube roots (SymPy's convention for
/// `roots_cubic`) is a different function from the same formula with real
/// cube roots.  The real root of `x³ + 3x − t` is
/// `real_root(t/2 + √(t²/4 + 1), 3) + real_root(t/2 − √(t²/4 + 1), 3)`;
/// with principal roots the value at t = 2 is not real.  Before, both
/// compiled to the real root, so the compiled formula disagreed with
/// `evalf` of the same expression.
#[test]
fn a_cardano_formula_compiles_to_its_own_value() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = (t.powi(2) / 4 + 1).sqrt();
    let u = &t / 2 + &s;
    let v = &t / 2 - &s;
    let principal = u.cbrt() + v.cbrt();
    let real = u.real_root(3).unwrap() + v.real_root(3).unwrap();
    let fp = principal.compile(&["t"]).unwrap();
    let fr = real.compile(&["t"]).unwrap();
    let z = principal.subs(&t, &ctx.int(2)).eval_complex64().unwrap();
    assert!(z.im.abs() > 0.1, "{z}");
    assert!(fp(&[2.0]).is_nan());
    let root = fr(&[2.0]);
    assert!((root.powi(3) + 3.0 * root - 2.0).abs() < 1e-14, "{root}");
    let e = real.subs(&t, &ctx.int(2)).eval_f64().unwrap();
    assert!(rel_close(root, e, 1e-15), "compiled {root}, evalf {e}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Evalf leftovers
// ═══════════════════════════════════════════════════════════════════════════

/// Before: `erfcinv(sin(1 + 10⁻¹⁵⁰) − sin(1))` took 7.5 s at 60 digits
/// (release; the fuzz unit, 11 s): the bracketed Halley iteration for
/// `erfcinv` converged, but its last step, below an ulp of `x`, rounded
/// onto the end of the bracket and was taken for a step outside it, and
/// bisection ran on for some 260 halvings — 1,104 evaluations of
/// `erfc(18.5)` at 800 bits, each a 1,500-term Taylor series.  Now a step
/// below the tolerance is convergence and `erfc` of a large argument is
/// Legendre's continued fraction.  (16 and 30 digits are refused: the
/// argument, a cancellation of about 500 bits inside `erfcinv`, needs more
/// than twice their working precision plus 256 bits.)
#[test]
fn erfcinv_of_a_deep_cancellation_is_fast() {
    let ctx = Context::new();
    let start = std::time::Instant::now();
    // mpmath: mp.dps = 400 and 800; u = sin(1 + mpf(10)**-150) - sin(1);
    // x = erfinv(1 - u); x, sin(x), cbrt(x)
    let cases = [
        (
            "erfcinv(sin(1 + 10^(-150)) - sin(1))",
            "18.507064015206896979524801523439936880441527661013707143929115",
        ),
        (
            "sin(erfcinv(sin(1 + 10^(-150)) - sin(1)))",
            "-0.33583531062020007272677255286954817008149675360751300774762196",
        ),
        (
            "cbrt(erfcinv(sin(1 + 10^(-150)) - sin(1)))",
            "2.6451228207211060753840918900703700609036630073292723340875748",
        ),
    ];
    for (src, want) in cases {
        complex_certified(&ctx, src, 60, want, "0");
    }
    assert!(
        start.elapsed() < std::time::Duration::from_secs(10),
        "{:?}",
        start.elapsed()
    );
}

/// Before: `uppergamma(10²⁰/3, 10²⁰/3)` ran Legendre's continued fraction
/// for two million steps (20 s in a debug build) and failed ("did not
/// converge"): `x ≥ s + 1` was decided in `f64`, where `s + 1` is `s`.
/// The value, `Q(s, s)·Γ(s)` with `Q ≈ ½`, is about `10^(6.3·10²⁰)`, beyond
/// the exponent range of the arbitrary-precision floats (mpmath 1.3 raises
/// `NoConvergence` for it): refused at once.  Where the value fits, the
/// equal-argument regime evaluates (mpmath: `gammainc(mpf(10)**7,
/// mpf(10)**7)` = 6.01161135698138974179208439491e+65657051).
#[test]
fn incomplete_gamma_of_huge_equal_arguments() {
    let ctx = Context::new();
    let start = std::time::Instant::now();
    for src in [
        "uppergamma(10^20/3, 10^20/3)",
        "lowergamma(10^20/3, 10^20/3)",
    ] {
        let r = ctx.parse(src).unwrap().eval_decimal(16);
        match r {
            Err(SymplexError::Unevaluable { reason }) => {
                assert!(reason.contains("overflows"), "{src}: {reason}");
            }
            other => panic!("{src}: {other:?}"),
        }
    }
    assert!(start.elapsed() < std::time::Duration::from_secs(2));
    let v = ctx
        .parse("uppergamma(10^7, 10^7)")
        .unwrap()
        .eval_decimal(15)
        .unwrap();
    assert_eq!(v, "6.01161135698139e65657051");
}

/// `polylog(s, z)` near `z = 1` for `s` next to an integer.  Before, an
/// order within `10⁻¹²` of an integer was taken for the integer (an `f64`
/// test), so `polylog(2 + 10⁻²⁰, 999/1000)` was `Li₂(999/1000)`, wrong
/// from its 20th digit at 30 and 60 digits (`…11774269579860…`), and
/// `polylog(1 + 10⁻³⁰, 95/100)` from its 29th.  The expansion for a
/// non-integer order then has two terms with opposite poles,
/// `Γ(1 − s)(−ln z)^{s−1}` and `ζ(s − n + 1)·(ln z)^{n−1}/(n−1)!`, which
/// cancel only when both see the same `s` (now `1 − s` is exact) and whose
/// cancellation is now measured and paid for in precision.  (mpmath: mp.dps = 300 and 600 — at 80 digits mpmath itself loses
/// these digits; polylog(2 + mpf(10)**-20, mpf(999)/1000) and
/// polylog(1 + mpf(10)**-30, mpf(95)/100).)
#[test]
fn polylog_next_to_an_integer_order_near_one() {
    let ctx = Context::new();
    let cases = [
        (
            "polylog(2 + 10^-20, 999/1000)",
            "1.6370226052761177426867044024033811127682981866720070528662934",
        ),
        (
            "polylog(1 + 10^-30, 95/100)",
            "2.9957322735539909934352235761389748673030061995463367178015044",
        ),
    ];
    for (src, want) in cases {
        for d in [16, 30, 60] {
            complex_certified(&ctx, src, d, want, "0");
        }
    }
}

/// The exact `bernoulli(n)` comes from the one tangent-number
/// implementation of `evalf` now (there were two); its values are
/// unchanged.  SymPy: `bernoulli(100)`, `bernoulli(30)`.
#[test]
fn bernoulli_numbers_from_one_implementation() {
    let ctx = Context::new();
    let b = |n: &str| format!("{}", ctx.parse(&format!("bernoulli({n})")).unwrap().eval());
    assert_eq!(
        b("100"),
        "-94598037819122125295227433069493721872702841533066936133385696204311395415197247711/33330"
    );
    assert_eq!(b("30"), "8615841276005/14322");
    assert_eq!(b("1"), "-1/2");
    assert_eq!(b("7"), "0");
}

/// A sum with a negative infinity displayed `x*oo + -oo`; now `x*oo - oo`,
/// which parses back to the same expression.
#[test]
fn a_negative_infinity_term_displays_with_a_minus() {
    let ctx = Context::new();
    for (src, shown) in [
        ("x*oo - oo", "x*oo - oo"),
        ("-oo + x*oo + y*oo", "x*oo + y*oo - oo"),
        ("-x*oo - oo", "-x*oo - oo"),
    ] {
        let e = ctx.parse(src).unwrap();
        assert_eq!(format!("{e}"), shown);
        assert_eq!(ctx.parse(shown).unwrap(), e, "{shown}");
    }
}
