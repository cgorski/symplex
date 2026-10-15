//! Tests of session 15 (s15k5): integration and series leftovers.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

fn definite(ctx: &Context, f: &str, lo: i64, hi: i64) -> String {
    let x = ctx.symbol("x");
    let v = parse(ctx, f)
        .try_integrate_definite(&x, &ctx.int(lo), &ctx.int(hi))
        .unwrap_or_else(|e| panic!("{f} on [{lo}, {hi}]: {e}"));
    v.eval_decimal(20).unwrap()
}

/// A rational function of `u = e^{λx}` (or `cˣ = e^{x·Log c}`) with a rate
/// neither real nor imaginary whose denominator vanishes on the real line,
/// at the point `x_r` where `|u|` meets the root: `cˣ/(1 − cˣ)` at `x = 0`
/// (`polylog(0, cˣ)`), `(1 + i)ˣ/((1 + i)ˣ − 1)` at `x = 0`.  Up to 0.55 the
/// spiral antiderivative read its joining constant off an `f64` value of the
/// gap at `x_r`, which has no value at a pole, and `integrate` refused (the
/// definite integrals away from the pole too).  Oracle: mpmath `quad` split
/// at 64 points, 30 and 45 digits: `∫₁³ polylog(0, Chi(−3/4)ˣ) dx =
/// −1.856597162409392828475 + 0.1198330315223400873763i`, `∫₁⁹ dx/(e^{(1+i)x}
/// − 1) = −0.1082425045618537698699 − 0.2603941133983179240815i`, `∫₁⁴
/// (1 + i)ˣ/((1 + i)ˣ − 1) dx = 2.430906220436339223478 −
/// 0.9765110891855332412603i`.
#[test]
fn spiral_with_a_pole_on_the_real_line() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let anti = parse(&ctx, "exp((1+I)*x)/(exp((1+I)*x) - 1)").integrate(&x);
    assert!(!anti.to_string().contains("Integral"), "{anti}");
    assert_eq!(
        definite(&ctx, "polylog(0, Chi(-3/4)^x)", 1, 3),
        "-1.8565971624093928285 + 0.11983303152234008738*i"
    );
    assert_eq!(
        definite(&ctx, "1/(exp((1+I)*x) - 1)", 1, 9),
        "-0.10824250456185376987 - 0.26039411339831792408*i"
    );
    assert_eq!(
        definite(&ctx, "(1+I)^x/((1+I)^x - 1)", 1, 4),
        "2.4309062204363392235 - 0.97651108918553324126*i"
    );
}

/// `∫₀¹⁰ dx/(e^{(10⁻³⁰+i)x} − 3) = −3.286410275925038131141 −
/// 0.2206747480253583753468i` (mpmath `quad`, 30 and 45 digits).  The spiral
/// meets `|u| = 3` at `x = 10³⁰·ln 3`, after about `1.7·10²⁹` turns; up to
/// 0.55 the turn count was read off an `f64` and refused beyond `10¹²`, so
/// `integrate` returned no antiderivative.  The count is now an exact
/// integer part.
#[test]
fn spiral_with_very_many_turns() {
    let ctx = Context::new();
    assert_eq!(
        definite(&ctx, "1/(exp((10^-30+I)*x)-3)", 0, 10),
        "-3.2864102759250381311 - 0.22067474802535837535*i"
    );
}

/// `∫₀³ e^{(1+i)x}/((e^{(1+i)x} − 2)(e^{(1+i)x} + 2i)) dx =
/// −0.4811866230856852581606 − 0.2200400237803276724187i` and `∫₀³
/// dx/(e^{(1+i)x} + e^{−(1+i)x} − 3) = −0.8916354139495250992657 −
/// 0.5072202224152742078328i` (mpmath `quad`, 30 and 45 digits).  Up to 0.55
/// both were refused: the first had five logarithms after the rational
/// integrator's real forms (`ln(u² + 4)` and `atan(u/2)` both hold the roots
/// `±2i`, counted twice against a cap of four), the second a rate ratio
/// `(1 + i)/(−1 − i)` that `eval` keeps as a quotient, and the definite
/// integrator could not locate the zeros of `e^{(1+i)x} + e^{−(1+i)x} − 3`
/// (now: `u` meets the roots `(3 ± √5)/2` only where `|u|` does, and there
/// `arg u` is certified off them).
#[test]
fn spiral_exponentials_with_two_roots() {
    let ctx = Context::new();
    assert_eq!(
        definite(
            &ctx,
            "exp((1+I)*x)/((exp((1+I)*x)-2)*(exp((1+I)*x)+2*I))",
            0,
            3
        ),
        "-0.48118662308568525816 - 0.22004002378032767242*i"
    );
    assert_eq!(
        definite(&ctx, "1/(exp((1+I)*x)+exp(-(1+I)*x)-3)", 0, 3),
        "-0.89163541394952509927 - 0.50722022241527420783*i"
    );
}

/// An imaginary rate: `u = e^{ix}` runs around the unit circle, and up to
/// 0.55 `∫ e^{ix}/(e^{ix} − 3) dx` was `−i·ln(e^{ix} − 3)`, which jumps by
/// `2π` at `x = π` (`e^{ix} − 3` crosses the cut of `ln` there); the definite
/// integral over `[0, 10]` was refused ("discontinuous near x = 3.141595").
/// Now `−i·ln(1 − e^{ix}/3)` (`|u| < 3`: the argument keeps a positive real
/// part), continuous: its values `10⁻⁸` on either side of `π` differ by
/// `5·10⁻⁹` (mpmath).  `∫₀¹⁰ = 0.1407691722248856065769 −
/// 0.6620242440760751260404i`, and with the root inside the circle `∫₀¹⁰
/// e^{ix}/(e^{ix} − 1/2) dx = 9.810675566709480393612 −
/// 1.061507041964889028054i` (`x + ln(1 − e^{−ix}/2)` times `−i`; mpmath
/// `quad`, 30 and 45 digits).
#[test]
fn circle_exponential_logarithms_are_continuous() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let anti = parse(&ctx, "exp(I*x)/(exp(I*x)-3)").integrate(&x);
    let at = |v: &str| {
        let p = parse(&ctx, v);
        anti.subs(&x, &p).eval_complex64().unwrap()
    };
    let jump = at("pi + 10^-8") - at("pi - 10^-8");
    assert!(jump.norm() < 1e-7, "{anti}: jump {jump}");
    assert_eq!(
        definite(&ctx, "exp(I*x)/(exp(I*x)-3)", 0, 10),
        "0.14076917222488560658 - 0.66202424407607512604*i"
    );
    assert_eq!(
        definite(&ctx, "exp(I*x)/(exp(I*x)-1/2)", 0, 10),
        "9.8106755667094803936 - 1.0615070419648890281*i"
    );
}

/// `K(m) = (π/2)(1 + m/4 + 9m²/64 + 25m³/256 + …)` and `E(m) = (π/2)(1 − m/4 −
/// 3m²/64 − 5m³/256 − …)` (DLMF 19.5.1, 19.5.2).  Up to 0.55 both series at
/// `m = 0` stayed unevaluated: the derivatives of `K` and `E` divide by `m`.
#[test]
fn complete_elliptic_integrals_at_zero_parameter() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let k = parse(&ctx, "elliptic_k(x)").series(&x, &zero, 4);
    let want_k = parse(&ctx, "pi/2 + pi*x/8 + 9*pi*x^2/128 + 25*pi*x^3/512");
    assert!((k.clone() - want_k).expand().is_zero() == Some(true), "{k}");
    let e = parse(&ctx, "elliptic_e(x)").series(&x, &zero, 4);
    let want_e = parse(&ctx, "pi/2 - pi*x/8 - 3*pi*x^2/128 - 5*pi*x^3/512");
    assert!((e.clone() - want_e).expand().is_zero() == Some(true), "{e}");
}

/// `series(elliptic_e(x)·Ei(elliptic_pi(x, y)), x, 0, 4)` took 33 s to stay
/// unevaluated at 0.55 (0.15 s at 0.53): `E(x)` had no series, so the
/// differentiation fallback of the whole product asked the limit engine for
/// its derivatives, whose coefficient sums were sampled at complex `y` again
/// and again.  Now the product of two series.  Oracle: at `y = 3/10`,
/// `E(x)·Ei(Π(x | y))` at `x = 1/100` is `6.25974021826147332712558254228`
/// (mpmath `ellipe`, `ei`, `ellippi`), within the `O(x⁴)` remainder.
#[test]
fn product_with_complete_elliptic_factor_expands() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let s = parse(&ctx, "elliptic_e(x)*Ei(elliptic_pi(x, y))").series(&x, &ctx.int(0), 4);
    assert!(!s.to_string().contains("Series"), "{s}");
    let v = s
        .subs(&y, &parse(&ctx, "3/10"))
        .subs(&x, &parse(&ctx, "1/100"))
        .eval_f64()
        .unwrap();
    assert!((v - 6.259_740_218_261_473).abs() < 1e-7, "{v}");
}

/// `H(g)` near a point where `g` has a non-real value has no expansion (`H`
/// has no value off the reals).  Up to 0.55 the differentiation fallback was
/// tried at every precision and inside the limits of the enclosing
/// fallback: `series(Chi(H(elliptic_pi(x, π))), x, 0, 4)` took 8.7 s to stay
/// unevaluated (0.07 s at 0.53, before `Π(x | m)` had a series; `K(π)` is
/// not real).  It stays unevaluated, now at once.
#[test]
fn heaviside_of_a_non_real_constant_has_no_series() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = parse(&ctx, "Chi(Heaviside(elliptic_pi(x, pi)))").series(&x, &ctx.int(0), 4);
    assert!(s.to_string().starts_with("Series("), "{s}");
}
