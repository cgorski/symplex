//! Cases where symplex does better than SymPy 1.14 that no other test
//! pinned, found by the audit of the repository's SymPy claims and of the
//! SymPy-generated fixtures (0.46).  Each test quotes SymPy's output (re-run
//! in SymPy 1.14.0) and an independent oracle: mpmath 1.3 at 30 digits, a
//! classical identity, or differentiation of the answer.

use symplex::fourier_transform::FourierConvention;
use symplex::prelude::*;

fn close(got: f64, want: f64) -> bool {
    (got - want).abs() <= 1e-12 * want.abs().max(1.0)
}

/// `∫ acosh(x) dx = x·acosh(x) − √(x − 1)·√(x + 1)`.  SymPy 1.14:
/// `integrate(acosh(x), x)` stays `Integral(acosh(x), x)` (the
/// `integrate:acosh` fixture of `new_features_cross_validation.json` is
/// skipped for that reason).  Oracle: differentiation at sample points on
/// both sides of the branch point `x = 1`.
#[test]
fn antiderivative_of_acosh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("acosh(x)").unwrap();
    let big_f = f.integrate(&x);
    assert!(!big_f.has_unevaluated(), "{big_f}");
    let residual = &big_f.diff(&x) - &f;
    for p in ["3/2", "5/2", "7", "1/3", "-2/5"] {
        let r = residual
            .subs(&x, &ctx.parse(p).unwrap())
            .eval_complex64()
            .unwrap();
        assert!(r.norm() < 1e-12, "∫ acosh = {big_f}: F′ − f = {r} at {p}");
    }
}

/// `L{(1 − e^{−t})/t}(s) = ln(s + 1) − ln(s)` (Frullani).  SymPy 1.14:
/// `laplace_transform((1 - exp(-t))/t, t, s)` is `LaplaceTransform(1/t, t,
/// s) - LaplaceTransform(1/t, t, s + 1)`, unevaluated (the
/// `laplace:forward` fixture `(1 - exp(-t))/t` is skipped for that reason).
/// Oracle: mpmath `quad(lambda t: (1 - exp(-t))*exp(-2*t)/t, [0, inf])` =
/// 0.405465108108164381978013115464 = `ln(3/2)`.
#[test]
fn laplace_transform_of_a_frullani_quotient() {
    let ctx = Context::new();
    let (t, s) = (ctx.symbol("t"), ctx.symbol("s"));
    let l = ctx
        .parse("(1 - exp(-t))/t")
        .unwrap()
        .try_laplace(&t, &s)
        .unwrap();
    assert!(!l.has_unevaluated(), "{l}");
    let v = l.subs(&s, &ctx.int(2)).eval_f64().unwrap();
    assert!(close(v, 0.405_465_108_108_164_4), "{l} at 2: {v}");
}

/// `L{1/(t + 1)}(1) = e·E₁(1)`.  SymPy 1.14 `laplace_transform(1/(t + 1), t,
/// s)` gives `(-exp(-s)*Ei(-s), 0, True)`: `−e^{−s}·Ei(−s) = e^{−s}·E₁(s)` has
/// the exponential's sign wrong and is 0.0807 at `s = 1`.  Oracle: mpmath
/// `quad(lambda t: exp(-t)/(t + 1), [0, inf])` =
/// 0.596347362323194074341078499369; symplex's `exp(s)·expint(1, s)` (pinned
/// in `v35_definite2`) has it.
#[test]
fn laplace_transform_of_a_shifted_reciprocal_at_a_point() {
    let ctx = Context::new();
    let (t, s) = (ctx.symbol("t"), ctx.symbol("s"));
    let l = ctx.parse("1/(t + 1)").unwrap().try_laplace(&t, &s).unwrap();
    let v = l.subs(&s, &ctx.int(1)).eval_f64().unwrap();
    assert!(close(v, 0.596_347_362_323_194_1), "{l} at 1: {v}");
}

/// `L{cos t/(t + 1)}` and `L{sin t/(t + 1)}` by the frequency shift of
/// `e^s·E₁(s)` (new in 0.46).  SymPy 1.14 gives `-exp(-s - I)*Ei(-s - I)/2 -
/// exp(-s + I)*Ei(-s + I)/2` and `-I*exp(-s - I)*Ei(-s - I)/2 + I*exp(-s +
/// I)*Ei(-s + I)/2`, the same sign error as for `1/(t + 1)`: 0.91706 and
/// −0.58871 at `s = 1`.  Oracle: mpmath `quad(lambda t: exp(-t)*cos(t)/(t +
/// 1), [0, inf])` = 0.410592543469122494771521946948 and with `sin`
/// 0.262728682711301740567209304802 (also `src/calculus/laplace.rs`, whose
/// doc said SymPy returns a Meijer G).
#[test]
fn laplace_transform_of_cos_and_sin_over_a_shifted_reciprocal() {
    let ctx = Context::new();
    let (t, s) = (ctx.symbol("t"), ctx.symbol("s"));
    for (src, want) in [
        ("cos(t)/(t + 1)", 0.410_592_543_469_122_5),
        ("sin(t)/(t + 1)", 0.262_728_682_711_301_74),
    ] {
        let l = ctx.parse(src).unwrap().try_laplace(&t, &s).unwrap();
        let z = l.subs(&s, &ctx.int(1)).eval_complex64().unwrap();
        assert!(
            close(z.re, want) && z.im.abs() < 1e-12,
            "L{{{src}}} = {l} at 1: {z}"
        );
    }
}

/// The ordinary-frequency Fourier transform of `1/(t² + 1)` is
/// `π·e^{−2π|w|}`, even in `w`.  SymPy 1.14 `fourier_transform(1/(t**2 + 1),
/// t, w)` is `pi*exp(-2*pi*w)` (for a real `w` too), which grows for `w < 0`:
/// 25.51 at `w = −1/3`.  Oracle: mpmath `quad(lambda t: cos(2*pi*t/3)/(t**2 +
/// 1), [-inf, inf])` = 0.386870519626368 = `π·e^{−2π/3}`.
#[test]
fn fourier_transform_at_a_negative_frequency() {
    let ctx = Context::new();
    let (t, w) = (ctx.symbol("t"), ctx.symbol("w"));
    let f = (t.powi(2) + 1).powi(-1);
    let ft = f
        .fourier_transform_with(&t, &w, FourierConvention::Ordinary)
        .unwrap();
    let want = std::f64::consts::PI * (-2.0 * std::f64::consts::PI / 3.0).exp();
    for p in ["-1/3", "1/3"] {
        let v = ft.subs(&w, &ctx.parse(p).unwrap()).eval_f64().unwrap();
        assert!(close(v, want), "{ft} at {p}: {v}");
    }
}

/// `W(3·ln 3) = ln 3`, since `ln 3·e^{ln 3} = 3·ln 3` and `ln 3 > −1`.
/// SymPy 1.14 leaves `LambertW(3*log(3))` (the `lambertw` fixture `3*log(3)`
/// of `new_features_cross_validation.json` records SymPy's unevaluated form
/// and checks the value only).  Oracle: mpmath `lambertw(3*log(3))` =
/// 1.09861228866810969139524523692 = `log(3)`.
#[test]
fn lambert_w_of_three_ln_three() {
    let ctx = Context::new();
    let w = ctx.parse("lambertw(3*log(3))").unwrap().eval();
    assert_eq!(w.to_string(), "ln(3)");
}

/// Decimal exponents in literals: up to 0.46 the parser had none, and
/// `1e-05` was read silently as `1·e − 5` (Euler's number), `2e3` as `2·e3`
/// (a symbol) — found by the 0.47 evalf parity hunt.  SymPy 1.14
/// `sympify("1e-05")` = `1.0e-5`; symplex reads decimal literals exactly.
#[test]
fn decimal_exponents_in_literals() {
    let ctx = Context::new();
    for (src, want) in [
        ("1e-05", ctx.rational(1, 100_000)),
        ("2e3", ctx.int(2000)),
        ("2.5E3", ctx.int(2500)),
        ("1.25e-2", ctx.rational(1, 80)),
        ("3e+2", ctx.int(300)),
    ] {
        assert_eq!(ctx.parse(src).unwrap(), want, "{src}");
    }
    // No digits after the `e`: Euler's number, as before.
    let x = ctx.symbol("x");
    assert_eq!(ctx.parse("2e").unwrap(), 2 * ctx.e());
    assert_eq!(ctx.parse("2e+x").unwrap(), 2 * ctx.e() + &x);
    assert_eq!(ctx.parse("2e3x").unwrap(), 2000 * &x);
    assert!(ctx.parse("1e100000").is_err());
}
