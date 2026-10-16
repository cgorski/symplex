//! Tests of session 14 (s14f2): exact zero tests in limits, series hangs.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use std::time::Duration;

use symplex::prelude::*;

use crate::perf_assert::assert_fast;

fn dir(d: &str) -> Direction {
    match d {
        "+" => Direction::Right,
        "-" => Direction::Left,
        _ => Direction::Both,
    }
}

/// `limit(f, x, point, d)` as a string (`ERR …` for a refusal).
fn limit_of(f: &str, point: &str, d: &str) -> String {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse(f).unwrap();
    match f.try_limit_dir(&x, &ctx.parse(point).unwrap(), dir(d)) {
        Ok(v) => v.to_string(),
        Err(e) => format!("ERR {e}"),
    }
}

/// `limit(f, x, point, d)` evaluated to a complex `f64`.
fn limit_value(f: &str, point: &str, d: &str) -> Complex64 {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse(f).unwrap();
    let v = f
        .try_limit_dir(&x, &ctx.parse(point).unwrap(), dir(d))
        .unwrap_or_else(|e| panic!("limit of {f} refused: {e}"));
    v.eval_complex64()
        .unwrap_or_else(|e| panic!("limit {v} of {f} has no value: {e:?}"))
}

fn assert_close(got: Complex64, re: f64, im: f64) {
    let want = Complex64::new(re, im);
    assert!(
        (got - want).norm() <= 1e-12 * want.norm().max(1.0),
        "got {got}, want {want}"
    );
}

fn is_zero(src: &str) -> Option<bool> {
    Context::new().parse(src).unwrap().is_zero()
}

/// `series(f, x, point, order, d)` as a string, `REFUSED` when it stays
/// unevaluated.
fn series_of(f: &str, point: &str, order: u32, d: &str) -> String {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = ctx
        .parse(f)
        .unwrap()
        .series_dir(&x, &ctx.parse(point).unwrap(), order, dir(d));
    if s.has_unevaluated() {
        "REFUSED".into()
    } else {
        s.to_string()
    }
}

/// The leading coefficients of difference quotients at the cuts of inverse
/// functions compare a value with its logarithmic form (Gruntz's rewrite):
/// zeros that canonical forms do not see.  Up to 0.51 (since the 0.50 zero
/// test, which stopped taking "0 to 30 digits" for a proof) none of them
/// was proved and 39 limits of the cut hunter were refused that 0.48
/// answered: `acos(−2) = π − i·ln(2 + √3)` on its cut, `asinh(iy) = iπ/2 +
/// acosh(y)` and `atan(iy) + atan(1/(iy)) = π/2` for `y > 1`.  The
/// logarithms and radicals of constants on an axis are now written by
/// their principal values, and logarithms of positive reals whose product
/// is 1 cancel (`calculus_util::exact_rewrites`).
///
/// Oracle: mpmath at 30 and 60 digits, every constant `0` (`|c| ≤ 10⁻⁶¹`);
/// exactly, `(2 + √3)(2 − √3) = 1` and the identities above.
#[test]
fn hidden_zeros_of_inverse_functions_on_their_cuts_are_proved() {
    for c in [
        "-ln(sqrt(3) + 2)*I - acos(-2) + pi",
        "-asinh(1/2*pi*I) + 1/2*pi*I + ln(sqrt(1/2*pi - 1)*sqrt(1/2*pi + 1) + 1/2*pi)",
        "-atan(1/acos(2)) - atan(acos(2)) + 1/2*pi",
        "-atan(1/2*pi*I) + 1/2*pi + atan(2*I/pi)",
        "-asinh(acos(2)) + pi*I/2 + ln(sqrt(-acos(2)*I - 1)*sqrt(-acos(2)*I + 1) - acos(2)*I)",
        "ln(2 + sqrt(3)) + ln(2 - sqrt(3))",
    ] {
        assert_eq!(is_zero(c), Some(true), "{c}");
    }
}

/// The limits behind the zeros above (the five families of the cut hunter,
/// 39 cases refused at 0.50 and 0.51; 0.48 answered them).
///
/// Oracle: the derivative of the inverse function at the cut point
/// (`1/f′(c)`): `−√3`, `√(1 + acos(2)²) = 0.856958635100106i`, `√(1 −
/// π²/4) = 1.21136332298462i`, `1 + acos(2)² = −0.734378102272636`, `1 −
/// π²/4 = −1.46740110027234`; mpmath at `x = ±10⁻²⁵`, 30 and 60 digits.
#[test]
fn limits_at_the_cuts_of_inverse_functions() {
    let r3 = 3f64.sqrt();
    assert_close(
        limit_value("x/(acos(-2 + I*x) - acos(-2))", "0", "+"),
        -r3,
        0.0,
    );
    assert_close(
        limit_value("x/(asinh(acos(2) + x) - asinh(acos(2)))", "0", "+"),
        0.0,
        0.856_958_635_100_106,
    );
    assert_close(
        limit_value("x/(asinh(acosh(0) + x) - asinh(acosh(0)))", "0", "+"),
        0.0,
        1.211_363_322_984_62,
    );
    assert_close(
        limit_value("x/(atan(acos(2) + x) - atan(acos(2)))", "0", "+"),
        -0.734_378_102_272_636,
        0.0,
    );
    assert_close(
        limit_value("x/(atan(acosh(0) + x) - atan(acosh(0)))", "0", "+"),
        -1.467_401_100_272_34,
        0.0,
    );
}

/// `√(e^z)` and `ln(e^z)` take the principal branch: `e^{z/2}` and `z`
/// when `|Im z| < π`, shifted by a multiple of `2πi` otherwise.  Up to 0.51
/// `√(e^{2i}) − e^i` was not proved 0 and `limit(asinh(e^i·x + e^i) − x,
/// x, −∞)` was refused ("cannot decide whether the coefficient
/// sqrt(exp(2*I)) - exp(I) is 0"; 0.48: `−i + ∞`).
///
/// Oracle: mpmath, `√(e^{2i}) − e^i = 0`, `√(e^{4i}) + e^{2i} = 0` (`|·| ≤
/// 10⁻³¹`), `√(e^{4i}) − e^{2i} = 0.83229 − 1.81859i`, `ln(e^{4i}) =
/// (4 − 2π)i`; the limit at `x = −10¹⁰, −10²⁰, −10⁴⁰`: `x + … − 1.0i`.
#[test]
fn powers_and_logarithms_of_exponentials_take_the_principal_branch() {
    assert_eq!(is_zero("sqrt(exp(2*I)) - exp(I)"), Some(true));
    assert_eq!(is_zero("sqrt(exp(4*I)) + exp(2*I)"), Some(true));
    assert_eq!(is_zero("sqrt(exp(3 + 2*I)) - exp(3/2 + I)"), Some(true));
    assert_eq!(is_zero("sqrt(exp(4*I)) - exp(2*I)"), Some(false));
    assert_eq!(is_zero("log(exp(4*I)) - 4*I + 2*pi*I"), Some(true));
    assert_eq!(is_zero("log(exp(4*I)) - 4*I"), Some(false));
    assert_eq!(
        limit_of("asinh(exp(I)*x + exp(I)) - x", "-oo", ""),
        "-I + oo"
    );
}

/// `atan z + atan(1/z) = ±π/2` for `Re z ≷ 0`; on the imaginary axis the
/// sum is `π/2` for `z = iy`, `y > 1`, and `−π/2` for `0 < y < 1`.  Up to
/// 0.51 none of these was proved 0 when `z` is not real.
///
/// Oracle: mpmath, `atan(2 + i) + atan((2 − i)/5) = 1.5707963267949`,
/// `atan(2i) + atan(−i/2) = 1.5707963267949`, `atan(i/2) + atan(−2i) =
/// −1.5707963267949`.
#[test]
fn atan_of_a_value_and_of_its_reciprocal() {
    assert_eq!(is_zero("atan(2 + I) + atan(1/(2 + I)) - pi/2"), Some(true));
    assert_eq!(
        is_zero("atan(-2 - I) + atan(1/(-2 - I)) + pi/2"),
        Some(true)
    );
    assert_eq!(is_zero("atan(2*I) + atan(-I/2) - pi/2"), Some(true));
    assert_eq!(is_zero("atan(I/2) + atan(-2*I) + pi/2"), Some(true));
    assert_eq!(is_zero("atan(I/2) + atan(-2*I) - pi/2"), Some(false));
}

/// A Gruntz coefficient with the variable `ω` in it is decided by its
/// symbol-free parts: `−ω·(−1)^(5/6) + ω·∛(−1)·i = ω·0`.  Up to 0.51 the
/// sum was sampled as a whole (0 at the samples, not proved) and
/// `limit(x^(4/3)·(tan u − u), x, −∞)`, `u = x^(−5/2)·ln(−x)`, was refused
/// (0.48: `0`); so was `x^(5/2)·(sinh v − v)`, `v = x^(−2/3)·ln(−x)` (0.48:
/// `i·∞`).
///
/// Oracle: `(−1)^(5/6) = e^{5πi/6} = i·e^{πi/3} = i·∛(−1)` exactly; mpmath
/// (30 and 60 digits) at `x = −10⁴, −10⁸, −10¹⁶`: `|f| ≤ 10⁻⁵⁸` for the
/// first, `f = 2·10⁸i, 1.6·10¹⁴i, 1.3·10²⁵i` for the second.
#[test]
fn a_coefficient_with_the_gruntz_variable_is_decided_by_its_parts() {
    assert_eq!(
        limit_of(
            "x^(4/3)*(tan(x^(-5/2)*log(-x)) - x^(-5/2)*log(-x))",
            "-oo",
            ""
        ),
        "0"
    );
    assert_eq!(
        limit_of(
            "x^(5/2)*(sinh(x^(-2/3)*log(-x)) - x^(-2/3)*log(-x))",
            "-oo",
            ""
        ),
        "I*oo"
    );
}

/// The leading coefficient Gruntz finds for `Γ(x)·tanh(ix + x²/2)` at `∞`
/// is `i·sin 2x/D + cos 2x/D` with `D = i·sin 2x + cos 2x` (the phases of
/// `e^{x² + 2ix}`), which is `1`: up to 0.51 its sign was sought as that of
/// a bounded oscillation and the limit was refused (0.48: `∞`).
///
/// Oracle: `Re(ix + x²/2) → +∞`, so `tanh(ix + x²/2) → 1` and the product
/// grows like `Γ(x)`; mpmath at `x = 10, 50, 100`: `362880`, `6.08·10⁶²`,
/// `9.33·10¹⁵⁵` (imaginary parts below `10⁻³⁷`).
#[test]
fn a_sum_of_quotients_with_one_denominator_is_its_quotient() {
    assert_eq!(limit_of("gamma(x)*tanh(I*x + x^2/2)", "oo", ""), "oo");
}

/// `tanh(u)` at a pole of `tanh` (`u₀ = −iπ/2`) is the Laurent series of
/// `coth(u − u₀)`.  Up to 0.51 the node went to the differentiation
/// fallback, which asked the limit engine for derivatives of every order of
/// `tanh(acosh(t − 3i·t³))` and its multiples `tᵏ·…`, and the series was
/// refused after 14 s.
///
/// Oracle: mpmath (30 and 60 digits) at `t = 10⁻³, 10⁻⁵, 10⁻⁷`:
/// `tanh(acosh(t − 3i·t³)) − (−i/t + (3 + i/2)·t) = (1.5 + 9.125i)·t³ + …`.
#[test]
fn tanh_at_a_pole_of_tanh_is_a_laurent_series() {
    assert_eq!(
        series_of("tanh(acosh(-3*I*(x - 1/2)^3 + (x - 1/2)))", "1/2", 3, "+"),
        "-I/(x - 1/2) + (x - 1/2)*(1/2*I + 3)"
    );
}

/// `tanh(Γ(−1 − 3i·x²))` at `0`: `Γ(−1 + δ) = −1/δ + γ − 1 + …`, so the
/// argument is `−i/(3x²) + γ − 1 + …`, which tends to infinity along the
/// imaginary axis, where `tanh` is periodic: no expansion.  Up to 0.51 the
/// series was refused after 20 s (the differentiation fallback asked the
/// limit engine for its derivatives; the series of `Γ(1 + w)` built
/// `w.known + 2` coefficients for a `w` of valuation 6, sums of products of
/// `ζ` values); now at once.
///
/// Oracle: the expansion of `Γ` at `−1` (DLMF 5.7.1 with 5.5.1): the
/// coefficient of `x⁻²` is `1/(3i)`, of real part 0 exactly.
#[test]
fn tanh_of_an_imaginary_pole_has_no_series() {
    assert_eq!(
        assert_fast!(
            Duration::from_secs(30),
            series_of("tanh(gamma(-1 - 3*I*x^2))", "0", 3, "-")
        ),
        "REFUSED"
    );
}

/// `J₀(acos π)` is real: `acos π = i·ln(π + √(π² − 1))` and `J₀(iy) =
/// I₀(y)`.  Up to 0.51 its value had no certified imaginary part 0, its
/// sign was unknown, and `limit(x·J₀(acos π), x, ∞)` was refused (0.48:
/// `∞`).
///
/// Oracle: DLMF 10.27.6 (`Jₙ(iz) = iⁿ·Iₙ(z)` for integer `n`); mpmath
/// `besselj(0, acos(pi))` = `besseli(0, acosh(pi))` = `2.00482519978781`.
#[test]
fn bessel_j0_at_an_imaginary_argument_is_real() {
    assert_eq!(limit_of("x*besselj(0, acos(pi))", "oo", ""), "oo");
    assert_eq!(limit_of("-x*besselj(0, acos(pi))", "oo", ""), "-oo");
}
