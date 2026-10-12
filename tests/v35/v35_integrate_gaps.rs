//! Needless refusals of the antiderivative engine (0.44): special-function
//! antiderivatives (`Ei`, `li`, `Si`, `Ci`, `Shi`, `Chi`, `erf`, `erfi`,
//! Fresnel, `Li₂`), integrands that become standard after a logarithmic or
//! exponential substitution, and a normal-form gap.  Every test names what
//! 0.43 returned (all of these stayed unevaluated) and the oracle of its
//! reference; every answer is also checked by differentiation at sample
//! points and, where a substitution has a singular point, by `F(b) − F(a)`
//! against quadrature on an interval across it.

use symplex::prelude::*;

/// `∫ src dx`, with `F′ = f` at the real sample points where `f` is real
/// (`F′ − f` evaluated as one expression, the parameters `a`, `b` at `6/5`,
/// `3/4` substituted after integrating) and the display `expected`.
fn antiderivative(src: &str, expected: &str) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse(src).unwrap();
    let big_f = f.integrate(&x);
    assert!(!big_f.has_unevaluated(), "∫ {src} dx = {big_f}");
    let bind = |e: &Ex| {
        let e = e.subs(&ctx.symbol("a"), &ctx.rational(6, 5));
        e.subs(&ctx.symbol("b"), &ctx.rational(3, 4))
    };
    let residual = bind(&(&big_f.diff(&x) - &f));
    let f = bind(&f);
    let mut checked = 0;
    for point in ["1/3", "7/5", "-5/7", "13/11", "5/2", "-9/4"] {
        let point = ctx.parse(point).unwrap();
        let Ok(fv) = f.subs(&x, &point).eval_complex64() else {
            continue;
        };
        if fv.im.abs() > 1e-12 * fv.norm().max(1.0) {
            continue;
        }
        let r = residual.subs(&x, &point).eval_complex64().unwrap();
        assert!(
            r.norm() < 1e-9 * fv.norm().max(1.0),
            "∫ {src} dx = {big_f}: F′ − f = {r} at {point}"
        );
        checked += 1;
    }
    assert!(checked > 0, "∫ {src} dx = {big_f}: no sample point");
    assert_eq!(big_f.to_string(), expected, "∫ {src} dx");
}

/// `F(b) − F(a) = ∫ₐᵇ f` by quadrature (an interval where `f` is finite).
fn continuous_on(src: &str, a: &str, b: &str) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse(src).unwrap();
    let big_f = f.integrate(&x);
    let (lo, hi) = (ctx.parse(a).unwrap(), ctx.parse(b).unwrap());
    let at = |p: &Ex| big_f.subs(&x, p).eval_complex64().unwrap();
    let diff = at(&hi) - at(&lo);
    let quad = f.integrate_numeric(&x, &lo, &hi).unwrap();
    assert!(
        (diff.re - quad).abs() < 1e-8 * quad.abs().max(1.0) && diff.im.abs() < 1e-8,
        "∫ {src} on [{a}, {b}]: F(b) − F(a) = {diff}, quadrature {quad}; F = {big_f}"
    );
}

/// `∫ e^{12x}/x dx` stayed unevaluated (only `eˣ/x` was in the table), and
/// with it 52 Riccati equations of the ODE solver.  SymPy
/// (`integrate(exp(12*x)/x, x)`): `Ei(12*x)`.
#[test]
fn exponential_integral_of_any_rate() {
    antiderivative("exp(12*x)/x", "Ei(12*x)");
    antiderivative("exp(-x)/x", "Ei(-x)");
    // SymPy: exp(1)*Ei(2*x)
    antiderivative("exp(2*x + 1)/x", "exp(1)*Ei(2*x)");
}

/// `∫ e^{2x}/(x + 1) dx`, `∫ eˣ/x²`, `∫ e^{2x}/x³`, `∫ eˣ/(x(x + 1))` stayed
/// unevaluated.  SymPy: `exp(-2)*Ei(2*x + 2)`, `-exp(x)/x + Ei(x)`,
/// `-exp(2*x)/x - exp(2*x)/(2*x**2) + 2*Ei(2*x)`, `Ei(x) - exp(-1)*Ei(x + 1)`.
#[test]
fn exponential_integral_over_linear_powers_by_parts() {
    antiderivative("exp(2*x)/(x + 1)", "exp(-2)*Ei(2*x + 2)");
    antiderivative("exp(x)/x^2", "-exp(x)/x + Ei(x)");
    antiderivative(
        "exp(2*x)/x^3",
        "-exp(2*x)/x - 1/2*x^(-2)*exp(2*x) + 2*Ei(2*x)",
    );
    antiderivative("exp(x)/(x*(x + 1))", "-exp(-1)*Ei(x + 1) + Ei(x)");
    // An elementary integral stays elementary (the Ei terms cancel).
    antiderivative("exp(x)*(x - 1)/x^2", "exp(x)/x");
}

/// A rate that may vanish: the case `a = 0` comes with the answer.  SymPy
/// (`integrate(exp(a*x)/x, x)`): `Ei(a*x)` without the case (`a` is complex
/// and may be 0 here, decision D4).
#[test]
fn exponential_integral_of_a_parameter_rate() {
    antiderivative(
        "exp(a*x)/x",
        "Piecewise(Ei(a*x) if a != 0, ln(abs(x)) if True)",
    );
    antiderivative(
        "exp(a*x)/(x + b)",
        "Piecewise(exp(-a*b)*Ei(a*b + a*x) if a != 0, ln(b + x) if True)",
    );
}

/// `∫ sin(3x)/x`, `∫ cos(3x)/x`, `∫ sin(x + 1)/x`, `∫ sin x/x²`, `∫ sinh(2x)/x`
/// stayed unevaluated.  SymPy: `Si(3*x)`, `Ci(3*x)`, `sin(1)*Ci(x) +
/// cos(1)*Si(x)`, `Ci(x) - sin(x)/x`, `Shi(2*x)`.
#[test]
fn sine_and_cosine_integrals_of_linear_arguments() {
    antiderivative("sin(3*x)/x", "Si(3*x)");
    antiderivative("cos(3*x)/x", "Ci(3*x)");
    antiderivative("cos(-3*x)/x", "Ci(3*x)");
    antiderivative("sin(x + 1)/x", "sin(1)*Ci(x) + cos(1)*Si(x)");
    antiderivative("sin(x)/x^2", "-sin(x)/x + Ci(x)");
    antiderivative("cos(x)/x^2", "-cos(x)/x - Si(x)");
    antiderivative("sinh(2*x)/x", "Shi(2*x)");
    antiderivative("cosh(2*x)/x", "Chi(2*x)");
    // ∫ sin(x)/x is smooth at 0, and so is Si.
    continuous_on("sin(3*x)/x", "-1", "2");
}

/// `∫ dx/ln(2x + 1)`, `∫ dx/ln(x)²` stayed unevaluated.  SymPy:
/// `li(2*x + 1)/2`, `-x/log(x) + li(x)`.
#[test]
fn logarithmic_integral_of_linear_arguments() {
    antiderivative("1/ln(2*x + 1)", "1/2*li(2*x + 1)");
    antiderivative("1/ln(x)^2", "-x/ln(x) + li(x)");
    antiderivative("(ln(x) - 1)/ln(x)^2", "x/ln(x)");
}

/// `∫ dv/(v·ln v − v)` stayed unevaluated while `∫ dv/(v·(ln v − 1))` worked:
/// the sum `v·ln v − v` was not factored before the substitution `u = ln v`.
/// SymPy: `log(log(v) - 1)`.
#[test]
fn a_common_factor_of_a_sum_is_taken_out() {
    antiderivative("1/(x*ln(x) - x)", "ln(abs(ln(x) - 1))");
    antiderivative("ln(x)/(x*ln(x)^2 + 2*x)", "1/2*ln(1/2*ln(x)^2 + 1)");
}

/// `x = eᵗ` for integrands in `x` and `ln x` alone: `∫ x/ln x`, `∫ x²/ln x`,
/// `∫ sin(ln x)/(x·ln x)` stayed unevaluated.  SymPy: `Ei(2*log(x))`,
/// `Ei(3*log(x))`, `Si(log(x))`.
#[test]
fn logarithmic_substitution_to_special_functions() {
    antiderivative("x/ln(x)", "Ei(2*ln(x))");
    antiderivative("x^2/ln(x)", "Ei(3*ln(x))");
    antiderivative("sin(ln(x))/(x*ln(x))", "Si(ln(x))");
    // Across the pole of 1/ln x at x = 1 the answer may step; on (1/4, 3/4)
    // and (3/2, 3) it must not.
    continuous_on("x/ln(x)", "1/4", "3/4");
    continuous_on("x/ln(x)", "3/2", "3");
}

/// `u = √(2eʸ + 3)`: `∫ dy/√(2eʸ + 3)` stayed unevaluated (the ODE solver
/// worked around it).  SymPy (`manualintegrate`):
/// `(log(sqrt(2*exp(y) + 3) - sqrt(3)) - log(sqrt(2*exp(y) + 3) + sqrt(3)))/sqrt(3)`.
#[test]
fn exponential_substitution_to_radicals() {
    antiderivative(
        "1/sqrt(2*exp(x) + 3)",
        "-1/3*sqrt(3)*ln(sqrt(3) + sqrt(2*exp(x) + 3)) + 1/3*sqrt(3)*ln(sqrt(2*exp(x) + 3) - sqrt(3))",
    );
    continuous_on("1/sqrt(2*exp(x) + 3)", "-3", "2");
    // A real answer on the whole domain e^x < 3/2 (the logarithm of the
    // negative √(3 − 2eˣ) − √3 keeps its absolute value).
    continuous_on("1/sqrt(3 - 2*exp(x))", "-3", "1/4");
    // SymPy: 2*sqrt(exp(x) + 1) + log(sqrt(exp(x) + 1) - 1) - log(sqrt(exp(x) + 1) + 1)
    antiderivative(
        "sqrt(exp(x) + 1)",
        "2*sqrt(exp(x) + 1) - ln(sqrt(exp(x) + 1) + 1) + ln(sqrt(exp(x) + 1) - 1)",
    );
}

/// `u = eˣ` with a non-rational integrand in `u`: `∫ exp(eˣ)`, `∫ sin(eˣ)`
/// stayed unevaluated.  SymPy: `Ei(exp(x))`, `Si(exp(x))`.
#[test]
fn exponential_substitution_to_special_functions() {
    antiderivative("exp(exp(x))", "Ei(exp(x))");
    antiderivative("sin(exp(x))", "Si(exp(x))");
    continuous_on("exp(exp(x))", "-2", "1");
}

/// `∫ e^{−a·x²}` and `∫ x²·e^{−x²}` stayed unevaluated.  SymPy:
/// `sqrt(pi)*erf(sqrt(a)*x)/(2*sqrt(a))` (without the case `a = 0`),
/// `-x*exp(-x**2)/2 + sqrt(pi)*erf(x)/4`.
#[test]
fn gaussians_with_symbolic_coefficients_and_polynomials() {
    antiderivative(
        "exp(-a*x^2)",
        "Piecewise(sqrt(pi)*erf(x*sqrt(a))/(2*sqrt(a)) if a != 0, x if True)",
    );
    antiderivative("x^2*exp(-x^2)", "-1/2*x*exp(-x^2) + 1/4*sqrt(pi)*erf(x)");
    antiderivative("x^3*exp(-x^2)", "-1/2*x^2*exp(-x^2) - 1/2*exp(-x^2)");
    antiderivative("exp(-sqrt(2)*x^2)", "1/4*2^(3/4)*sqrt(pi)*erf(x*2^(1/4))");
}

/// `∫ x·erf x`, `∫ e^{−x²}·erf x`, `∫ erfi(2x)` stayed unevaluated (`erf` was
/// no by-parts `u`).  SymPy: `x**2*erf(x)/2 + x*exp(-x**2)/(2*sqrt(pi)) -
/// erf(x)/4`, `sqrt(pi)*erf(x)**2/4`, `x*erfi(2*x) - exp(4*x**2)/(2*sqrt(pi))`.
#[test]
fn error_functions_by_parts() {
    antiderivative(
        "x*erf(x)",
        "1/2*x^2*erf(x) - (-1/2*x*exp(-x^2) + 1/4*sqrt(pi)*erf(x))/sqrt(pi)",
    );
    antiderivative("exp(-x^2)*erf(x)", "1/4*erf(x)^2*sqrt(pi)");
    antiderivative("erfi(2*x)", "x*erfi(2*x) - exp(4*x^2)/(2*sqrt(pi))");
}

/// Fresnel integrals: `∫ sin(x²)`, `∫ cos(x²)`, `∫ sin(2x² + 1)`, `∫ x²·sin(x²)`
/// stayed unevaluated.  SymPy: `sqrt(2)*sqrt(pi)*fresnels(sqrt(2)*x/sqrt(pi))/2`,
/// `… fresnelc …`, `sqrt(pi)*(sin(1)*fresnelc(2*x/sqrt(pi)) +
/// cos(1)*fresnels(2*x/sqrt(pi)))/2`.
#[test]
fn fresnel_integrals_of_quadratics() {
    antiderivative(
        "sin(x^2)",
        "1/2*sqrt(2)*sqrt(pi)*fresnels(x*sqrt(2)/sqrt(pi))",
    );
    antiderivative(
        "cos(x^2)",
        "1/2*sqrt(2)*sqrt(pi)*fresnelc(x*sqrt(2)/sqrt(pi))",
    );
    antiderivative(
        "sin(2*x^2 + 1)",
        "1/2*sqrt(pi)*(sin(1)*fresnelc(2*x/sqrt(pi)) + cos(1)*fresnels(2*x/sqrt(pi)))",
    );
    antiderivative(
        "x^2*sin(x^2)",
        "-1/2*x*cos(x^2) + 1/4*sqrt(2)*sqrt(pi)*fresnelc(x*sqrt(2)/sqrt(pi))",
    );
    continuous_on("sin(-3*x^2 + x)", "-2", "2");
}

/// Dilogarithms: `∫ ln(1 − x)/x`, `∫ ln(2x + 3)/x`, `∫ ln(x)/(1 − x)` stayed
/// unevaluated (by parts turned one into the other).  SymPy:
/// `-polylog(2, x)`, `log(3)*log(x) - polylog(2, -2*x/3)` (here `ln|x|`),
/// `Li₂(1 − x)` (here `−ln x·ln(1 − x) − Li₂(x)`, the same up to `π²/6`).
#[test]
fn dilogarithms() {
    antiderivative("ln(1 - x)/x", "-polylog(2, x)");
    antiderivative("ln(2*x + 3)/x", "-polylog(2, -2/3*x) + ln(3)*ln(abs(x))");
    antiderivative("ln(x)/(1 - x)", "-ln(x)*ln(-x + 1) - polylog(2, x)");
    // ln(x)/(1 − x) is smooth at x = 1, where ln(1 − x) and Li₂(x) reach
    // their cuts: the answer must be continuous there.
    continuous_on("ln(x)/(1 - x)", "1/2", "3");
}
