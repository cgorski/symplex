//! 0.22 — `integrate` regressions from the 0.21 differential audit.
//!
//! Four confirmed wrong antiderivatives: `∫ sqrt(x²) dx` treated as
//! `∫ x dx`; the Weierstrass substitution applying a numeric factor twice;
//! the same substitution producing bogus `atan(3·tan(x/2))` forms for
//! `∫ cos x·R(sin x) dx` (rational integrals with an irreducible quartic
//! denominator); and `∫ 1/((x²−4)²+1) dx` emitting an `re(…)`/`im(…)`
//! closed form whose derivative is not the integrand.
//!
//! The right assertion for an antiderivative is `d/dx F = f`, so every
//! test differentiates the result and compares with the integrand at
//! four rational points (relative tolerance 1e-10).  The reference value
//! at each point is `f` itself, quoted from sympy 1.14
//! (`symplex/.venv/bin/python`) as `N(f.subs(x, p), 17)`; the sympy
//! antiderivative `integrate(f, x)` is cited alongside for the record.
//! Where the simplifier can do it, `(F′ − f).simplify()` is also checked
//! to be zero.  Nothing below was computed by hand.

// The reference values are quoted with sympy's 17 significant digits.
#![allow(clippy::excessive_precision)]

use symplex::prelude::*;

/// The rational sample points `p = n/d` shared by every table below:
/// `Rational(1,3), Rational(-5,7), Rational(13,11), Rational(-17,5)`.
const POINTS: [(i64, i64); 4] = [(1, 3), (-5, 7), (13, 11), (-17, 5)];

/// Relative closeness, absolute when the reference is (essentially) zero.
fn close(actual: f64, expected: f64, rel: f64, label: &str) {
    if expected.abs() < 1e-15 && actual.abs() < 1e-15 {
        return;
    }
    let scale = expected.abs().max(1e-300);
    assert!(
        (actual - expected).abs() <= rel * scale,
        "{label}: got {actual:e}, expected {expected:e} (relative error {:e})",
        (actual - expected).abs() / scale
    );
}

/// `F = ∫ f dx` must be a closed form with `F′(p) = f(p)` at every point
/// of [`POINTS`]; `f_ref[i]` is sympy's `N(f.subs(x, p_i), 17)` and is
/// also checked against symplex's own evaluation of `f` (so a typo in the
/// integrand would be caught, not silently agreed with).  Returns `F`.
fn assert_antiderivative(f: &Ex, x: &Ex, f_ref: &[f64; 4], label: &str) -> Ex {
    let big_f = f.integrate(x);
    assert!(
        !big_f.has_unevaluated(),
        "{label}: expected a closed form, got {big_f}"
    );
    let d_big_f = big_f.diff(x);
    let ctx = x.context();
    for (&(n, d), &expected) in POINTS.iter().zip(f_ref) {
        let p = ctx.rational(n, d);
        let f_at = f
            .subs(x, &p)
            .eval_f64()
            .unwrap_or_else(|e| panic!("{label}: f({n}/{d}) did not evaluate: {e}"));
        close(
            f_at,
            expected,
            1e-12,
            &format!("{label}: f({n}/{d}) vs sympy"),
        );
        let df_at = d_big_f
            .subs(x, &p)
            .eval_f64()
            .unwrap_or_else(|e| panic!("{label}: F'({n}/{d}) did not evaluate: {e} (F = {big_f})"));
        close(
            df_at,
            expected,
            1e-10,
            &format!("{label}: F'({n}/{d}) vs f, F = {big_f}"),
        );
    }
    big_f
}

/// `(F′ − f).simplify()` is literally zero.
fn assert_derivative_simplifies_to_f(f: &Ex, big_f: &Ex, x: &Ex, label: &str) {
    let residual = (big_f.diff(x) - f).simplify();
    assert_eq!(
        residual.to_string(),
        "0",
        "{label}: (F' - f).simplify() should be 0, F = {big_f}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. sqrt(x²) is |x|, not x
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sqrt_of_x_squared_integrates_as_abs_x() {
    // sympy: integrate(sqrt(x**2), x) = x*sqrt(x**2)/2
    // f = sqrt(x**2): 0.33333333333333333, 0.71428571428571429, 1.1818181818181818, 3.4000000000000000
    // (0.21 returned x**2/2, whose derivative is -5/7 at the second point.)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.pow(&ctx.int(2)).sqrt();
    let big_f = assert_antiderivative(
        &f,
        &x,
        &[
            0.33333333333333333,
            0.71428571428571429,
            1.1818181818181818,
            3.4000000000000000,
        ],
        "∫ sqrt(x^2)",
    );
    assert_ne!(big_f.to_string(), "1/2*x^2", "∫ sqrt(x^2) must not be ∫ x");
}

#[test]
fn neg_sqrt_of_x_squared_integrates_as_neg_abs_x() {
    // sympy: integrate(-sqrt(x**2), x) = -x*sqrt(x**2)/2
    // f = -sqrt(x**2): -0.33333333333333333, -0.71428571428571429, -1.1818181818181818, -3.4000000000000000
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = -x.pow(&ctx.int(2)).sqrt();
    assert_antiderivative(
        &f,
        &x,
        &[
            -0.33333333333333333,
            -0.71428571428571429,
            -1.1818181818181818,
            -3.4000000000000000,
        ],
        "∫ -sqrt(x^2)",
    );
}

#[test]
fn x_squared_to_three_halves_is_abs_x_cubed() {
    // sympy: integrate((x**2)**Rational(3,2), x) = x*(x**2)**(3/2)/4
    // f = (x**2)**(3/2): 0.037037037037037037, 0.36443148688046647, 1.6506386175807663, 39.304000000000000
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.pow(&ctx.int(2)).pow(&ctx.rational(3, 2));
    assert_antiderivative(
        &f,
        &x,
        &[
            0.037037037037037037,
            0.36443148688046647,
            1.6506386175807663,
            39.304000000000000,
        ],
        "∫ (x^2)^(3/2)",
    );
}

#[test]
fn sqrt_of_x_fourth_flattens_to_x_squared() {
    // The flattening (x^m)^n → x^(m·n) stays legitimate when m·n is an even
    // integer: sqrt(x**4) = x**2 for every real x.
    // sympy: integrate(sqrt(x**4), x) = x*sqrt(x**4)/3
    // f = sqrt(x**4): 0.11111111111111111, 0.51020408163265306, 1.3966942148760331, 11.560000000000000
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.pow(&ctx.int(4)).sqrt();
    let big_f = assert_antiderivative(
        &f,
        &x,
        &[
            0.11111111111111111,
            0.51020408163265306,
            1.3966942148760331,
            11.560000000000000,
        ],
        "∫ sqrt(x^4)",
    );
    assert_eq!(big_f.to_string(), "1/3*x^3");
}

#[test]
fn reciprocal_sqrt_flattening_still_reaches_asinh() {
    // The flattening (x²+1)^(1/2) → (x²+1)^(-1/2) that the standard-form
    // handler relies on is unaffected (odd numerator in m = 1/2).
    // sympy: integrate(1/sqrt(x**2+1), x) = asinh(x)
    // f = 1/sqrt(x**2+1): 0.94868329805051380, 0.81373347120673496, 0.64594224146617384, 0.28216632399155017
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.int(1) / (x.pow(&ctx.int(2)) + 1).sqrt();
    let big_f = assert_antiderivative(
        &f,
        &x,
        &[
            0.94868329805051380,
            0.81373347120673496,
            0.64594224146617384,
            0.28216632399155017,
        ],
        "∫ 1/sqrt(x^2+1)",
    );
    assert_eq!(big_f.to_string(), "asinh(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Weierstrass: the constant factor is applied exactly once
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn neg_of_product_sin_plus_one_times_cos() {
    // sympy: integrate(-(sin(x)+1)*cos(x), x) = -sin(x)**2/2 - sin(x)
    // f = -(sin(x)+1)*cos(x): -1.2541418478496062, -0.26060980851463465, -0.73015561900524185, 1.2138548681487652
    // (0.21 returned an F with F' = -f.)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = -((x.sin() + 1) * x.cos());
    let big_f = assert_antiderivative(
        &f,
        &x,
        &[
            -1.2541418478496062,
            -0.26060980851463465,
            -0.73015561900524185,
            1.2138548681487652,
        ],
        "∫ -(sin x + 1) cos x",
    );
    assert_derivative_simplifies_to_f(&f, &big_f, &x, "∫ -(sin x + 1) cos x");
}

#[test]
fn three_times_cos_minus_four_times_sin() {
    // sympy: integrate(3*(cos(x)-4)*sin(x), x) = 3*sin(x)**2/2 + 12*cos(x)
    // f = 3*(cos(x)-4)*sin(x): -2.9987816569492214, 6.3760801515840370, -10.050827316564941, -3.8076632510298883
    // (0.21 returned an F with F' = 3·f.)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (x.cos() - 4) * x.sin() * 3;
    let big_f = assert_antiderivative(
        &f,
        &x,
        &[
            -2.9987816569492214,
            6.3760801515840370,
            -10.050827316564941,
            -3.8076632510298883,
        ],
        "∫ 3 (cos x - 4) sin x",
    );
    assert_derivative_simplifies_to_f(&f, &big_f, &x, "∫ 3 (cos x - 4) sin x");
}

#[test]
fn minus_two_times_cos_minus_four_times_sin() {
    // sympy: integrate(-2*(cos(x)-4)*sin(x), x) = -sin(x)**2 - 8*cos(x)
    // f = -2*(cos(x)-4)*sin(x): 1.9991877712994809, -4.2507201010560247, 6.7005515443766276, 2.5384421673532589
    // (0.21 returned an F with F' = -2·f.)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (x.cos() - 4) * x.sin() * -2;
    let big_f = assert_antiderivative(
        &f,
        &x,
        &[
            1.9991877712994809,
            -4.2507201010560247,
            6.7005515443766276,
            2.5384421673532589,
        ],
        "∫ -2 (cos x - 4) sin x",
    );
    assert_derivative_simplifies_to_f(&f, &big_f, &x, "∫ -2 (cos x - 4) sin x");
}

#[test]
fn three_cos_over_sin_squared_plus_one() {
    // sympy: integrate(3*cos(x)/(sin(x)**2+1), x) = 3*atan(sin(x))
    // f = 3*cos(x)/(sin(x)**2+1): 2.5607285380951193, 1.5860619515432745, 0.61294300386480045, -2.7226050514833956
    // (0.21 returned 9*atan(3*tan(x/2)): both the double constant and the
    // wrong quartic integration at once.)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.cos() * 3 / (x.sin().pow(&ctx.int(2)) + 1);
    let big_f = assert_antiderivative(
        &f,
        &x,
        &[
            2.5607285380951193,
            1.5860619515432745,
            0.61294300386480045,
            -2.7226050514833956,
        ],
        "∫ 3 cos x/(sin²x + 1)",
    );
    assert_eq!(big_f.to_string(), "3*atan(sin(x))");
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. cos x·R(sin x), sin x·R(cos x): u-substitution wins; a·sin² + b
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cos_over_sin_squared_plus_one_is_atan_sin() {
    // sympy: integrate(cos(x)/(sin(x)**2+1), x) = atan(sin(x))
    // f = cos(x)/(sin(x)**2+1): 0.85357617936503976, 0.52868731718109150, 0.20431433462160015, -0.90753501716113186
    // (0.21 returned atan(3*tan(x/2)).)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.cos() / (x.sin().pow(&ctx.int(2)) + 1);
    let big_f = assert_antiderivative(
        &f,
        &x,
        &[
            0.85357617936503976,
            0.52868731718109150,
            0.20431433462160015,
            -0.90753501716113186,
        ],
        "∫ cos x/(sin²x + 1)",
    );
    assert_eq!(big_f.to_string(), "atan(sin(x))");
    assert_derivative_simplifies_to_f(&f, &big_f, &x, "∫ cos x/(sin²x + 1)");
}

#[test]
fn minus_two_cos_over_sin_squared_plus_four() {
    // sympy: integrate(-2*cos(x)/(sin(x)**2+4), x) = -atan(sin(x)/2)
    // f = -2*cos(x)/(sin(x)**2+4): -0.46016263779896368, -0.34117844800441081, -0.15619005457761024, 0.47563421846387038
    // (0.21 returned -2*atan(-3*tan(x/2)); at x = 1, f = -0.2295 but F' = 1.0568.)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.cos() * -2 / (x.sin().pow(&ctx.int(2)) + 4);
    let big_f = assert_antiderivative(
        &f,
        &x,
        &[
            -0.46016263779896368,
            -0.34117844800441081,
            -0.15619005457761024,
            0.47563421846387038,
        ],
        "∫ -2 cos x/(sin²x + 4)",
    );
    assert_eq!(big_f.to_string(), "-atan(1/2*sin(x))");
}

#[test]
fn three_sin_over_scaled_cos_squared_plus_four() {
    // sympy: integrate(3*sin(x)/(Rational(9,4)*cos(x)**2+4), x) = -atan(3*cos(x)/4)
    // f: 0.16334897170264714, -0.37188892601949924, 0.64203134933413098, 0.12561268880521519
    // (0.21 returned an F with F' = 3·f.)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin() * 3 / (x.cos().pow(&ctx.int(2)) * ctx.rational(9, 4) + 4);
    let big_f = assert_antiderivative(
        &f,
        &x,
        &[
            0.16334897170264714,
            -0.37188892601949924,
            0.64203134933413098,
            0.12561268880521519,
        ],
        "∫ 3 sin x/(9/4 cos²x + 4)",
    );
    assert_eq!(big_f.to_string(), "-atan(3/4*cos(x))");
}

#[test]
fn one_over_sin_squared_plus_four() {
    // Pure Weierstrass case: t = tan(x/2) gives (1+t²)/(2t⁴+6t²+2), a
    // biquadratic whose roots in t² are real and negative.
    // sympy: integrate(1/(sin(x)**2+4), x) =
    //   521*sqrt(10)*sqrt(3 - sqrt(5))*(atan(sqrt(2)*tan(x/2)/sqrt(3 - sqrt(5))) + pi*floor((x/2 - pi/2)/pi))/(2880*sqrt(5) + 6440) + … (four such terms)
    // f = 1/(sin(x)**2+4): 0.24348338810226434, 0.22577812476394662, 0.20592339995226665, 0.24598423027398144
    // (0.21 returned sqrt(5)/10*atan(3*sqrt(5)/5*tan(x/2)); at x = 1, f = 0.2124 but F' = 0.1267.)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.int(1) / (x.sin().pow(&ctx.int(2)) + 4);
    assert_antiderivative(
        &f,
        &x,
        &[
            0.24348338810226434,
            0.22577812476394662,
            0.20592339995226665,
            0.24598423027398144,
        ],
        "∫ 1/(sin²x + 4)",
    );
}

#[test]
fn sin_over_cos_squared_plus_one_unchanged() {
    // Was already right in 0.21 (pins the u = cos x route).
    // sympy: integrate(sin(x)/(cos(x)**2+1), x) = -atan(cos(x))
    // f = sin(x)/(cos(x)**2+1): 0.17284967790034225, -0.41701520021118986, 0.80894963134186558, 0.13208314868872689
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin() / (x.cos().pow(&ctx.int(2)) + 1);
    let big_f = assert_antiderivative(
        &f,
        &x,
        &[
            0.17284967790034225,
            -0.41701520021118986,
            0.80894963134186558,
            0.13208314868872689,
        ],
        "∫ sin x/(cos²x + 1)",
    );
    assert_eq!(big_f.to_string(), "-atan(cos(x))");
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Irreducible quartic denominators with nested-radical roots
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn one_over_x_squared_minus_four_squared_plus_one() {
    // sympy: integrate(1/((x**2-4)**2+1), x) — an atan/log combination in
    //   sqrt(-1/136 + sqrt(17)/544), sqrt(33 - 8*sqrt(17)), … (too long to quote).
    // f = 1/((x**2-4)**2+1): 0.062021439509954058, 0.075880159281967006, 0.12858096358877979, 0.017195839982391460
    // (0.21 returned a form in re(…)/im(…) with F'(-2) = 0.98879 for f(-2) = 1,
    //  F'(1) = 0.016535 for f(1) = 0.1.)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.int(1) / ((x.pow(&ctx.int(2)) - 4).pow(&ctx.int(2)) + 1);
    let big_f = assert_antiderivative(
        &f,
        &x,
        &[
            0.062021439509954058,
            0.075880159281967006,
            0.12858096358877979,
            0.017195839982391460,
        ],
        "∫ 1/((x²-4)²+1)",
    );
    let s = big_f.to_string();
    assert!(
        !s.contains("re(") && !s.contains("im("),
        "closed form must not contain opaque re/im constants: {s}"
    );
}

#[test]
fn ln_of_x_squared_minus_four_squared_plus_one() {
    // By parts, then the same quartic: ∫ ln((x²−4)²+1) = x·ln(…) − ∫ 4x²(x²−4)/((x²−4)²+1).
    // sympy: integrate(log((x**2-4)**2+1), x) = x*log((x**2 - 4)**2 + 1) - 4*x
    //   - sqrt(2 + sqrt(17)/2)*log(x**2 - sqrt(2)*x*sqrt(4 + sqrt(17)) - sqrt(8*sqrt(17) + 33) + 4 + 2*sqrt(17)) + … (atan terms)
    // f = log((x**2-4)**2+1): 2.7802751551639376, 2.5786000347877560, 2.0511965062168838, 4.0630877859048051
    // (0.21: F'(-2) = -1.0076 for f(-2) = 0.)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ((x.pow(&ctx.int(2)) - 4).pow(&ctx.int(2)) + 1).ln();
    let big_f = assert_antiderivative(
        &f,
        &x,
        &[
            2.7802751551639376,
            2.5786000347877560,
            2.0511965062168838,
            4.0630877859048051,
        ],
        "∫ ln((x²-4)²+1)",
    );
    let s = big_f.to_string();
    assert!(
        !s.contains("re(") && !s.contains("im("),
        "closed form must not contain opaque re/im constants: {s}"
    );
}

#[test]
fn biquadratic_with_real_negative_roots_in_x_squared() {
    // x⁴ + 6x² + 1 = (x² + 3 − 2√2)(x² + 3 + 2√2).
    // sympy: integrate(1/(x**4+6*x**2+1), x)
    //   = 2*(sqrt(2)/16 + 1/8)*atan(x/(-1 + sqrt(2))) - 2*(1/8 - sqrt(2)/16)*atan(x/(1 + sqrt(2)))
    // f = 1/(x**4+6*x**2+1): 0.59558823529411765, 0.23139938319198150, 0.088254086897815499, 0.0049021145761435653
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.int(1) / (x.pow(&ctx.int(4)) + x.pow(&ctx.int(2)) * 6 + 1);
    assert_antiderivative(
        &f,
        &x,
        &[
            0.59558823529411765,
            0.23139938319198150,
            0.088254086897815499,
            0.0049021145761435653,
        ],
        "∫ 1/(x⁴+6x²+1)",
    );
}

#[test]
fn biquadratic_with_cubic_numerator() {
    // sympy: integrate((x**3+1)/(x**4+3*x**2+1), x)
    //   = (1/4 - 3*sqrt(5)/20)*log(x**2 - 19737*sqrt(5)/980 - …) + (1/4 + 3*sqrt(5)/20)*log(…) - 2*sqrt(…)*atan(…) - …
    // f = (x**3+1)/(x**4+3*x**2+1): 0.77064220183486239, 0.22772720489479182, 0.37119436819099178, -0.22623108834730347
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (x.pow(&ctx.int(3)) + 1) / (x.pow(&ctx.int(4)) + x.pow(&ctx.int(2)) * 3 + 1);
    assert_antiderivative(
        &f,
        &x,
        &[
            0.77064220183486239,
            0.22772720489479182,
            0.37119436819099178,
            -0.22623108834730347,
        ],
        "∫ (x³+1)/(x⁴+3x²+1)",
    );
}

#[test]
fn one_over_x_fourth_plus_one_unchanged() {
    // Was already right in 0.21 (the Rothstein–Trager route stays in charge
    // when its answer verifies).
    // sympy: integrate(1/(x**4+1), x) = -sqrt(2)*log(x**2 - sqrt(2)*x + 1)/8
    //   + sqrt(2)*log(x**2 + sqrt(2)*x + 1)/8 + sqrt(2)*atan(sqrt(2)*x - 1)/4 + sqrt(2)*atan(sqrt(2)*x + 1)/4
    // f = 1/(x**4+1): 0.98780487804878049, 0.79345670852610707, 0.33889634739132448, 0.0074275663727331067
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.int(1) / (x.pow(&ctx.int(4)) + 1);
    let big_f = assert_antiderivative(
        &f,
        &x,
        &[
            0.98780487804878049,
            0.79345670852610707,
            0.33889634739132448,
            0.0074275663727331067,
        ],
        "∫ 1/(x⁴+1)",
    );
    let s = big_f.to_string();
    assert!(
        s.contains("atan") && s.contains("ln"),
        "expected atan/ln form: {s}"
    );
}

#[test]
fn x_cubed_over_shifted_biquadratic() {
    // sympy: integrate(x**3/(x**4-8*x**2+17), x) = log(x**4 - 8*x**2 + 17)/4 + 2*atan(x**2 - 4)
    // f = x**3/(x**4-8*x**2+17): 0.0022970903522205207, -0.027653119271853865, 0.21224070398538633, -0.67586529466791394
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.pow(&ctx.int(3)) / (x.pow(&ctx.int(4)) - x.pow(&ctx.int(2)) * 8 + 17);
    assert_antiderivative(
        &f,
        &x,
        &[
            0.0022970903522205207,
            -0.027653119271853865,
            0.21224070398538633,
            -0.67586529466791394,
        ],
        "∫ x³/(x⁴-8x²+17)",
    );
}

#[test]
fn general_quartic_is_never_silently_wrong() {
    // x⁴ + x + 1 is not biquadratic; sympy answers with a RootSum:
    //   integrate(1/(x**4+x+1), x)
    //   = RootSum(229*_t**4 + 18*_t**2 + 8*_t + 1, Lambda(_t, _t*log(2061*_t**3/64 - 687*_t**2/64 + 391*_t/64 + x + 27/64)))
    // f = 1/(x**4+x+1): 0.74311926605504587, 1.8314263920671243, 0.24198000165275597, 0.0076199997561600078
    // The contract: either an unevaluated Integral, or a form whose derivative is f.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.int(1) / (x.pow(&ctx.int(4)) + &x + 1);
    let big_f = f.integrate(&x);
    if big_f.has_unevaluated() {
        return;
    }
    let d_big_f = big_f.diff(&x);
    let f_ref = [
        0.74311926605504587,
        1.8314263920671243,
        0.24198000165275597,
        0.0076199997561600078,
    ];
    for (&(n, d), &expected) in POINTS.iter().zip(&f_ref) {
        let p = ctx.rational(n, d);
        let df_at = d_big_f
            .subs(&x, &p)
            .eval_f64()
            .unwrap_or_else(|e| panic!("F'({n}/{d}) did not evaluate: {e} (F = {big_f})"));
        close(
            df_at,
            expected,
            1e-10,
            &format!("∫ 1/(x⁴+x+1): F'({n}/{d}), F = {big_f}"),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// u-substitution must not absorb a stray `x`
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn u_substitution_rejects_factor_with_bare_var() {
    // With the wider candidate set (u = sin x inside a sum) the factor
    // x + sin²x must not be turned into x + x²: either unevaluated or right.
    // sympy: integrate((x+sin(x)**2)*cos(x), x) = x*sin(x) + sin(x)**3/3 + cos(x)
    // f = (x+sin(x)**2)*cos(x): 0.41614930888322871, -0.21545486337458458, 0.77289471651060822, 3.2239807196321021
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x + x.sin().pow(&ctx.int(2))) * x.cos();
    let big_f = f.integrate(&x);
    if big_f.has_unevaluated() {
        return;
    }
    let d_big_f = big_f.diff(&x);
    let f_ref = [
        0.41614930888322871,
        -0.21545486337458458,
        0.77289471651060822,
        3.2239807196321021,
    ];
    for (&(n, d), &expected) in POINTS.iter().zip(&f_ref) {
        let p = ctx.rational(n, d);
        let df_at = d_big_f
            .subs(&x, &p)
            .eval_f64()
            .unwrap_or_else(|e| panic!("F'({n}/{d}) did not evaluate: {e} (F = {big_f})"));
        close(
            df_at,
            expected,
            1e-10,
            &format!("∫ (x + sin²x) cos x: F'({n}/{d}), F = {big_f}"),
        );
    }
}

/// Found by `fuzz_integrate` (0.22.3): `∫ |√x| dx` came back as
/// `2/3·x^(3/2)·sign(√x)` — the `|g| = sign(g)·g` route assumed `g` real for
/// real `x`, but `√x` is imaginary for `x < 0` (`|√x| = √|x|`).  At
/// `x = −5/7` the integrand is `√(5/7)` = 0.8451542547285166 (sympy
/// `N(Abs(sqrt(Rational(-5,7))))`) while that `F′` was its negative.  The
/// route now refuses non-real arguments; any closed form returned must
/// differentiate back to the integrand.
#[test]
fn abs_of_sqrt_is_not_integrated_as_a_sign_product() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sqrt().abs();
    let big_f = f.integrate(&x);
    if !big_f.has_unevaluated() {
        let df = big_f.diff(&x);
        for v in [ctx.rational(-5, 7), ctx.rational(1, 3), ctx.rational(13, 4)] {
            let fv = f.subs(&x, &v).eval_f64().unwrap();
            let dv = df.subs(&x, &v).eval_f64().unwrap();
            assert!(
                (fv - dv).abs() < 1e-12,
                "∫|√x| = {big_f}: F′({v}) = {dv}, f = {fv}"
            );
        }
    }
    // The polynomial case keeps its sign-product form: ∫ |x| dx = x·|x|/2.
    let g = x.abs().integrate(&x);
    assert!(!g.has_unevaluated(), "{g}");
    let dg = g
        .diff(&x)
        .subs(&x, &ctx.rational(-5, 7))
        .eval_f64()
        .unwrap();
    assert!((dg - 5.0 / 7.0).abs() < 1e-12, "{g}");
}

/// `∫ F′ = F` check at a few rational points (both sides finite reals).
fn assert_differentiates_back(f: &Ex, big_f: &Ex, x: &Ex) {
    assert!(!big_f.has_unevaluated(), "∫ {f} returned {big_f}");
    let ctx = x.context();
    let df = big_f.diff(x);
    let mut checked = 0;
    for v in [
        ctx.rational(1, 3),
        ctx.rational(7, 5),
        ctx.rational(13, 4),
        ctx.int(5),
    ] {
        // Only points where the integrand is a real number.
        let Ok(fv) = f.subs(x, &v).eval_f64() else {
            continue;
        };
        let dv = df.subs(x, &v).eval_f64().unwrap();
        assert!(
            (fv - dv).abs() <= 1e-12 * fv.abs().max(1.0),
            "∫ {f} = {big_f}: F′({v}) = {dv}, f = {fv}"
        );
        checked += 1;
    }
    assert!(checked >= 2, "∫ {f}: only {checked} real sample points");
}

/// Found by `fuzz_integrate` (0.22.3) as a 20 s timeout: `∫ ln(√x + sin(−2)) dx`
/// searched 524 540 `integrate_node` calls (12.5 s in release) before
/// returning unevaluated.  The by-parts step needed `∫ x/(x + k) dx` for a
/// constant `k` that is not rational (`π`, `√2`, `sin 2`, a symbol), which no
/// route handled.  SymPy: `integrate(x*log(x + sin(2)), x)` =
/// `x**2*log(x + sin(2))/2 - x**2/4 + x*sin(2)/2 - log(x + sin(2))*sin(2)**2/2`.
#[test]
fn polynomial_over_symbolic_linear_denominator() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol_with("a", &[Assumption::Positive]);
    for k in [ctx.pi(), ctx.int(2).sqrt(), ctx.int(2).sin()] {
        for f in [&x / (&x + &k), x.powi(3) / (2 * &x + &k), 1 / (&x - &k)] {
            let big_f = f.integrate(&x);
            assert_differentiates_back(&f, &big_f, &x);
        }
    }
    // A symbolic constant: check at a = 5/3 after integrating symbolically.
    for f in [&x / (&x + &a), x.powi(3) / (2 * &x + &a)] {
        let big_f = f.integrate(&x);
        let at = |e: &Ex| e.subs(&a, &ctx.rational(5, 3));
        assert_differentiates_back(&at(&f), &at(&big_f), &x);
    }
    let f = &x * (&x + ctx.int(2).sin()).ln();
    assert_differentiates_back(&f, &f.integrate(&x), &x);
    let f = (x.sqrt() + ctx.int(-2).sin()).ln();
    assert_differentiates_back(&f, &f.integrate(&x), &x);
}

/// A failing search is bounded: `integrate_node` is capped per top-level
/// `integrate` (every successful integration in the suite takes ≤ ~600
/// calls; the cap is 20 000), so a hopeless integrand returns unevaluated
/// quickly instead of branching for seconds.
#[test]
fn failing_integration_search_is_bounded() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (x.sin() + x.ln()).sqrt() * x.exp().atan();
    let started = std::time::Instant::now();
    let big_f = f.integrate(&x);
    assert!(big_f.has_unevaluated(), "{big_f}");
    assert!(
        started.elapsed().as_secs() < 10,
        "took {:?}",
        started.elapsed()
    );
}

/// Found by `fuzz_integrate` (0.22.3): `∫ |atan(−1)/cos x| dx` came back as
/// `−atan(−1)·ln|sec x + tan x|`.  The `|g|` route took "no real root" to
/// mean "constant sign", but `g = c/cos x` changes sign at its poles; at
/// `x = 13/4` the integrand is 0.79003593021582217 (sympy
/// `N(Abs(atan(-1)/cos(Rational(13,4))), 17)`) and that `F′` was its
/// negative.  The route now requires `g` continuous as well as real.
#[test]
fn abs_of_a_function_with_poles_is_not_a_constant_sign() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (ctx.int(-1).atan() / x.cos()).abs();
    let big_f = f.integrate(&x);
    if !big_f.has_unevaluated() {
        assert_differentiates_back(&f, &big_f, &x);
    }
    // Continuous (polynomial) arguments keep the route: ∫ |x − 2| dx and
    // ∫ x²·|2x + 1| dx.
    let g = (&x - 2).abs();
    assert_differentiates_back(&g, &g.integrate(&x), &x);
    let h = x.powi(2) * (2 * &x + 1).abs();
    assert_differentiates_back(&h, &h.integrate(&x), &x);
}

/// Found by `fuzz_integrate` (0.22.3) as `∫ x²(x − x⁻³ − ¼)·|x| dx = nan`.
/// Two causes.  (1) A reciprocal inside a sum (`x²(x + 1/x)`) was never
/// recognised as a rational function — `as_numer_denom` only lifts
/// negative-power *factors* — so even `∫ x²(x + 1/x) dx` stayed
/// unevaluated; the rational integrator now combines over a common
/// denominator first.  SymPy: `integrate(x**2*(x + 1/x), x)` =
/// `x**4/4 + x**2/2`, `integrate(x*(x + x**-2), x)` = `x**3/3 + log(x)`.
/// (2) The `|g|` route shifted by `G(r)` even where `G` is singular at the
/// root, turning the whole answer into `nan`.
#[test]
fn reciprocals_inside_sums_are_rational_functions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(2) * (&x + x.powi(-1));
    assert_eq!(f.integrate(&x), x.powi(4) / 4 + x.powi(2) / 2);
    let g = &x * (&x + x.powi(-2));
    assert_differentiates_back(&g, &g.integrate(&x), &x);
    let p = x.powi(2) * (&x - x.powi(-3) - ctx.rational(1, 4));
    let h = &p * x.abs();
    assert_differentiates_back(&h, &h.integrate(&x), &x);
    // |x|/x² = 1/|x|: G = ln|x| is singular at the root — never `nan`.
    let k = x.abs() / x.powi(2);
    let big_k = k.integrate(&x);
    assert!(!format!("{big_k}").contains("nan"), "{big_k}");
    if !big_k.has_unevaluated() {
        assert_differentiates_back(&k, &big_k, &x);
    }
}

/// Found by `fuzz_integrate` (0.22.3): `∫ (x + ln x − ¾)·ln x dx` came back
/// as `x·ln²x − 11/4·x·ln x + 11/4·x`, missing `∫ x·ln x = x²ln x/2 − x²/4`.
/// The Risch tower's extraction of the integrand as a polynomial in
/// θ = ln x kept only the first coefficient when two terms had the same power
/// (`x·θ` and `−¾·θ`).  SymPy `integrate((x + log(x) - Rational(3,4))*log(x), x)`
/// = `x**2*log(x)/2 - x**2/4 + x*log(x)**2 - 11*x*log(x)/4 + 11*x/4`.
#[test]
fn log_polynomial_with_repeated_powers_keeps_every_term() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x + x.ln() - ctx.rational(3, 4)) * x.ln();
    let big_f = f.integrate(&x);
    assert_differentiates_back(&f, &big_f, &x);
    let sympy = x.powi(2) * x.ln() / 2 - x.powi(2) / 4 + &x * x.ln().powi(2)
        - ctx.rational(11, 4) * &x * x.ln()
        + ctx.rational(11, 4) * &x;
    assert_eq!((big_f - sympy).expand(), ctx.int(0));
}

/// Found by `fuzz_integrate` (0.22.3): `∫ x^(−5/2)·atan(√x) dx` contained
/// `−⅔·ln|√x|`.  After `x = t²` the rational integrator's `ln|t|` is the
/// antiderivative only for real `t`; for `x < 0` (where the integrand is
/// still real: 2.8732399347178847 at x = −5/7, SymPy
/// `N((x**Rational(-5,2)*atan(sqrt(x))).subs(x, Rational(-5,7)), 17)`)
/// the derivative was 1.94.  The substitution now keeps the analytic `ln t`.
#[test]
fn radical_substitution_keeps_analytic_logarithms() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.pow(&ctx.rational(-5, 2)) * x.sqrt().atan();
    let big_f = f.integrate(&x);
    assert!(!big_f.has_unevaluated(), "{big_f}");
    let df = big_f.diff(&x);
    for v in [ctx.rational(-5, 7), ctx.rational(1, 3), ctx.rational(13, 4)] {
        let fv = f.subs(&x, &v).eval_f64().unwrap();
        let dv = df.subs(&x, &v).eval_f64().unwrap();
        assert!(
            (fv - dv).abs() <= 1e-12 * fv.abs().max(1.0),
            "F′({v}) = {dv}, f = {fv}: {big_f}"
        );
    }
}

/// Found via `fuzz_integrate` (0.22.3, `∫ atan(√x − x³) dx`): a `RootOf` is
/// a constant, but `diff` returned the unevaluated `Derivative(RootOf(…), x)`
/// for every one — even when the polynomial's variable *is* `x`, which is
/// bound there — so an antiderivative containing roots could neither be
/// differentiated back nor evaluated.
#[test]
fn rootof_is_a_constant_under_differentiation() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let roots = (x.powi(5) - &x - 1).solve(&x).unwrap();
    assert_eq!(roots.len(), 5);
    let r = &roots[0];
    assert!(format!("{r}").contains("RootOf"), "{r}");
    assert_eq!(r.diff(&x), ctx.int(0));
    assert_eq!((r * &x).diff(&x), r.clone());
    assert_eq!((r * x.sin()).diff(&x), r * x.cos());
}

/// Found by `fuzz_integrate` (0.22.3) as a timeout: `∫ atan(√x − x³) dx`
/// took 36 s and returned a 32 KB sum.  Three causes: the Lazard–Rioboo–
/// Trager step computed the whole Euclidean remainder sequence over ℚ(t)
/// although it needs only the degree-1 member (the lower ones cost 35 s),
/// every remainder's coefficients grew in `t` (now made primitive over
/// ℚ[t] at each step), and `RootSum` was expanded over twelve `RootOf`
/// placeholders.  The answer must still differentiate back to the
/// integrand.
#[test]
fn arctangent_of_a_radical_polynomial_is_fast_and_compact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (x.sqrt() - x.powi(3)).atan();
    let started = std::time::Instant::now();
    let big_f = f.integrate(&x);
    let elapsed = started.elapsed();
    assert!(!big_f.has_unevaluated(), "{big_f}");
    assert!(
        big_f.to_string().len() < 2_000,
        "{} chars",
        big_f.to_string().len()
    );
    let d = big_f.diff(&x);
    for v in [ctx.rational(1, 3), ctx.rational(7, 5)] {
        let fv = f.subs(&x, &v).eval_f64().unwrap();
        let dv = d.subs(&x, &v).eval_complex64().unwrap();
        assert!(
            (dv.re - fv).abs() < 1e-10 && dv.im.abs() < 1e-10,
            "F′({v}) = {dv}, f = {fv}"
        );
    }
    assert!(elapsed.as_secs() < 20, "took {elapsed:?}");
}
