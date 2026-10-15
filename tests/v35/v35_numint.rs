//! The `f64` numerical-integration API (`integrate_numeric`,
//! `integrate_numeric_with`, the closure `quadrature`) and the quadrature
//! guard of `integrate_definite`, after 0.43.  Before, the `f64`
//! Gauss–Kronrod quadrature retried a singular integrand under an
//! endpoint-smoothing substitution whose nodes crowded towards a nonzero end
//! (and towards the end `t = 1` of the map of an infinite range), where in
//! `f64` they cannot come closer than half an ulp: the integrand was cut off
//! at `1 − x ≈ 2⁻⁵³` (`x ≈ 10¹⁶`) and the error estimate did not see it.
//! `integrate_numeric` accepted an estimate up to 1000 times its tolerance.
//! Every reference value is a closed form or cites the mpmath call that
//! produced it.

use symplex::definite::{QuadOpts, QuadRule, quadrature};
use symplex::prelude::*;

fn parse3(ctx: &Context, body: &str, lo: &str, hi: &str) -> (Ex, Ex, Ex) {
    (
        ctx.parse(body).unwrap(),
        ctx.parse(lo).unwrap(),
        ctx.parse(hi).unwrap(),
    )
}

/// The default tolerance of `integrate_numeric`, `max(10⁻¹², 10⁻¹⁰·|I|)`.
fn tol(truth: f64) -> f64 {
    let o = QuadOpts::default();
    o.abs_tol.max(o.rel_tol * truth.abs())
}

/// `integrate_numeric` within its tolerance of `truth`, and
/// `integrate_numeric_with` within its own error estimate (plus half an ulp).
fn check(body: &str, lo: &str, hi: &str, truth: f64) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (f, l, h) = parse3(&ctx, body, lo, hi);
    let v = f.integrate_numeric(&x, &l, &h).unwrap();
    assert!(
        (v - truth).abs() <= tol(truth),
        "integrate_numeric of {body} on [{lo}, {hi}]: {v} (truly {truth})"
    );
    let r = f
        .integrate_numeric_with(&x, &l, &h, &QuadOpts::default())
        .unwrap();
    assert!(
        (r.value - truth).abs() <= r.error + truth.abs() * f64::EPSILON / 2.0,
        "integrate_numeric_with of {body} on [{lo}, {hi}]: {} ± {:e} (truly {truth})",
        r.value,
        r.error
    );
}

/// Before: 1.9999999898808887 from `integrate_numeric`, and
/// `integrate_numeric_with` reported an error of `3.2·10⁻¹⁰` for its error
/// of `10⁻⁸`.  Exact value 2.
#[test]
fn inverse_square_root_at_a_nonzero_end() {
    check("1/sqrt(1 - x)", "0", "1", 2.0);
}

/// The Beta(5, ½) moment.  Before: 1.0952385480452325 with an error
/// estimate of `4.4·10⁻¹⁰` (the error is `1.4·10⁻⁸`).  mpmath: `mp.dps =
/// 30; quad(lambda x: x**4*exp(x/10)/(sqrt(1 - x)*beta(5, 0.5)), [0, 1])`
/// = 1.0952385618059876.
#[test]
fn beta_moment_error_estimate_is_honest() {
    check(
        "x^4*exp(x/10)/(sqrt(1 - x)*beta(5, 1/2))",
        "0",
        "1",
        1.0952385618059876,
    );
}

/// Before: `integrate_numeric` returned 15.999983137416310 — an error of
/// `1.7·10⁻⁵`, accepted because its estimate (`2.8·10⁻⁷`) was within 1000
/// times the tolerance.  Exact value 16 (`∫₀¹ ln²u/√u du = 2/(½)³`).
#[test]
fn log_squared_over_square_root_at_a_nonzero_end() {
    check("log(1 - x)^2/sqrt(1 - x)", "0", "1", 16.0);
}

/// Before: `∫₀¹ dx/√(1 − x²)` was 1.5707963196395556 and `∫₀^π dx/√(sin x)`
/// 5.2441150802275507.  Exact values `π/2` and `Γ(¼)²/√(2π)` (mpmath
/// `gamma(0.25)**2/sqrt(2*pi)` = 5.2441151085842396).
#[test]
fn singular_factors_nonlinear_in_x() {
    check("1/sqrt(1 - x^2)", "0", "1", std::f64::consts::FRAC_PI_2);
    check("1/sqrt(sin(x))", "0", "pi", 5.244_115_108_584_239);
}

/// Before: `∫₁^∞ x^(−3/2) dx` was 1.9999999898808887 (the map `x = 1 +
/// t/(1 − t)` cut off at `x ≈ 10¹⁶`, an estimate of `3.2·10⁻¹⁰`), and
/// `∫₁^∞ x^(−101/100) dx` and `∫₁^∞ x^(−5/4) dx` were refused.  Exact values
/// 2, 100 and 4.
#[test]
fn algebraic_tails_on_a_half_line() {
    check("x^(-3/2)", "1", "oo", 2.0);
    check("x^(-5/4)", "1", "oo", 4.0);
    check("x^(-101/100)", "1", "oo", 100.0);
}

/// Before: refused (`∫₀¹ x^(−9/10) dx` did not converge, and an interior
/// singularity was beyond the `f64` bisection).  Exact values 10 and
/// `2(√(1/3) + √(2/3))`.
#[test]
fn strong_and_interior_singularities() {
    check("x^(-9/10)", "0", "1", 10.0);
    check("(x - 1)^(-9/10)", "1", "2", 10.0);
    check(
        "abs(x - 1/3)^(-1/2)",
        "0",
        "1",
        2.0 * ((1.0f64 / 3.0).sqrt() + (2.0f64 / 3.0).sqrt()),
    );
}

/// Narrow peaks no node of a rule over the whole range sees are split off
/// (at their centre and ten half-widths either side).  Before: the Gaussian
/// integrated to `0` with an error estimate of 0.  mpmath (`mp.dps = 40`,
/// split at the peak): `quad(lambda x: exp(-10**6*(x - mpf(2)/3)**2)*(x +
/// 1)**2*cos(4*x/3), [0, mpf(2)/3 - mpf(1)/100, mpf(2)/3, mpf(2)/3 +
/// mpf(1)/100, 2])` = 0.0031031445690331173347, and `quad(lambda x:
/// cos(x)/(1 + 10**6*(x - mpf(2)/7)**2), [0, mpf(2)/7 - mpf(1)/100,
/// mpf(2)/7, mpf(2)/7 + mpf(1)/100, 1])` = 0.0030088115321549025050.
#[test]
fn narrow_peaks_are_split_off() {
    check(
        "exp(-1000000*(x - 2/3)^2)*(x + 1)^2*cos(4*x/3)",
        "0",
        "2",
        0.003_103_144_569_033_117,
    );
    check(
        "cos(x)/(1 + 1000000*(x - 2/7)^2)",
        "0",
        "1",
        0.003_008_811_532_154_902_5,
    );
}

/// A `sech` peak of half-width `10⁻⁴` is split off too.  Before: `0.0`
/// (`6.6·10⁻¹⁵⁹`) with an estimate of `1.2·10⁻¹⁵⁸`.  Exact value
/// `(2/k)(atan(tanh(2k/3)) + atan(tanh(k/3)))`, `k = 10⁴`, which is `π/k`
/// to double precision.
#[test]
fn sech_peak_is_split_off() {
    check(
        "sech(10000*(x - 1/3))",
        "0",
        "1",
        std::f64::consts::PI * 1e-4,
    );
}

/// A strong singularity at an end at 0, which the `f64` bisection resolves:
/// the estimate covers the mass between the outermost node and the end,
/// extrapolated from `|x|^α`.  Before: `3.7·10⁻⁹` off (twice the
/// tolerance) with an estimate of `1.7·10⁻⁹`.  mpmath (`x = u²⁰`):
/// `20*quad(lambda u: sin(u**20/4 + 2)/(1 + 5*u**40), [0, 0.5, 1])` =
/// 17.347029667901519456.
#[test]
fn strong_singularity_at_zero_estimate_is_honest() {
    check(
        "x^(-19/20)*sin(x/4 + 2)/(1 + 5*x^2)",
        "0",
        "1",
        17.347_029_667_901_52,
    );
}

/// The arbitrary-precision quadrature splits a narrow Lorentzian at the real
/// part of its complex poles.  Before: its bisection chased the peak (3.5 s
/// for 20 digits in a debug build), and `∫_{−5/3}^1 (1/(1 + 10⁶(x −
/// 1/9)²))·(1/(1 + 4x²/3) + cos(7x/4)·atan(3x)) dx` was refused at 54
/// digits after 20 s.  Exact value `(atan(8000/9) + atan(16000/9))/1000`;
/// mpmath (`mp.dps = 70`):
/// 0.0031399051541237284136687746597663006595156775549962.
#[test]
fn narrow_lorentzian_to_fifty_digits() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (f, l, h) = parse3(&ctx, "1/(1 + 1000000*(x - 1/9)^2)", "-5/3", "1");
    assert_eq!(
        f.integrate_numeric_decimal(&x, &[&l, &h], 50, QuadRule::TanhSinh)
            .unwrap(),
        "0.0031399051541237284136687746597663006595156775549962"
    );
}

/// A kink at an irrational point is split off: the estimate of a segment
/// holding it understated its error.  Before: `9.0·10⁻¹²` off with an
/// estimate of `6.6·10⁻¹²`.  mpmath: `quad(lambda x: abs(sin(x))*(cos(6*x)
/// + sqrt(x + 4)/(1 + x**2/2)), [0, pi, 4])` = 2.1880567277270403922.
#[test]
fn kink_at_an_irrational_point_is_split_off() {
    check(
        "abs(sin(x))*(cos(6*x) + sqrt(x + 4)/(1 + x^2/2))",
        "0",
        "4",
        2.188_056_727_727_040_4,
    );
}

/// The estimate covers the rounding of a sum that cancels (QUADPACK's
/// `50ε·∫|f|` per segment).  Before: `∫₀^π sin 2x·cos²x dx`, exactly 0,
/// was `−4.4·10⁻¹⁷` with an estimate of `8.9·10⁻²²`.
#[test]
fn estimate_covers_the_rounding_of_a_cancelling_sum() {
    check("sin(2*x)*cos(x)^2", "0", "pi", 0.0);
}

/// Divergent integrals and an oscillating infinite one stay refused.  Since
/// 0.52 a refused quadrature gives way to the definite integrator's closed
/// form: `∫₀^∞ sin x/x dx` is `π/2` (mpmath `quadosc(lambda x: sin(x)/x,
/// [0, inf], omega=1)` = 1.5707963267948966; up to 0.51 refused), so the
/// oscillating case is one without a closed form.
#[test]
fn divergent_integrals_are_refused() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (body, lo, hi) in [
        ("x^(-2)", "-1", "1"),
        ("1/(1 - x)", "0", "1"),
        ("sin(x)/(x + sqrt(x + 1))", "0", "oo"),
    ] {
        let (f, l, h) = parse3(&ctx, body, lo, hi);
        assert!(
            f.integrate_numeric(&x, &l, &h).is_err(),
            "{body} on [{lo}, {hi}]"
        );
    }
    let (f, l, h) = parse3(&ctx, "sin(x)/x", "0", "oo");
    let v = f.integrate_numeric(&x, &l, &h).unwrap();
    assert!((v - std::f64::consts::FRAC_PI_2).abs() < 1e-15, "{v}");
}

/// The closure API (no symbolic body, no arbitrary-precision fallback): its
/// estimate covers the part next to a nonzero end its nodes cannot reach.
/// Before: `∫₀¹ (1 − x)^(−1/2) dx` was 1.9999999898808887 with an estimate
/// of `3.2·10⁻¹⁰`.  An infinite end is resolved (`x = 1/u − 1`): `∫₁^∞
/// x^(−3/2) dx = 2`.
#[test]
fn closure_quadrature_estimate_covers_a_cut_off_end() {
    let o = QuadOpts::default();
    let r = quadrature(&|x: f64| 1.0 / (1.0 - x).sqrt(), 0.0, 1.0, &o).unwrap();
    assert!((r.value - 2.0).abs() <= r.error, "{r:?}");
    let r = quadrature(&|x: f64| x.powf(-1.5), 1.0, f64::INFINITY, &o).unwrap();
    assert!(
        (r.value - 2.0).abs() <= tol(2.0) && r.error <= tol(2.0),
        "{r:?}"
    );
}

/// The guard of `integrate_definite` no longer refuses a correct closed form
/// next to a singularity at a nonzero end.  Before: all three were refused
/// ("the closed form disagrees with adaptive quadrature"; the quadrature
/// said 9.77, 31.5 and 9.76).  Exact values 10, 100 and 10.
#[test]
fn quadrature_guard_keeps_correct_closed_forms() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (body, lo, hi, value) in [
        ("(1 - x)^(-9/10)", "0", "1", "10"),
        ("(1 - x)^(-99/100)", "0", "1", "100"),
        ("(x - 1)^(-9/10)", "1", "2", "10"),
    ] {
        let (f, l, h) = parse3(&ctx, body, lo, hi);
        let v = f.try_integrate_definite(&x, &l, &h).unwrap();
        assert_eq!(v.to_string(), value, "{body} on [{lo}, {hi}]");
    }
}
