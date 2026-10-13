//! Parity with SymPy's integration test suite (0.47): the integrals of
//! `sympy/integrals/tests/` (test_integrals, test_manual, test_risch,
//! test_heurisch, test_meijerint, test_failing_integrals, …) that symplex
//! answered wrongly or refused while SymPy 1.14 answers them, and families
//! around them.  Every antiderivative is checked by differentiation at
//! sample points (`F′ − f` evaluated as one expression), and where the
//! integrand is finite across a singular point of the answer, by
//! `F(b) − F(a)` against quadrature; every definite value against an mpmath
//! quadrature at 30 digits (the oracle named in each test).

use symplex::prelude::*;

/// Declares `decls` (`"a:pos"`, `"k:int"`, …) in `ctx`.
fn declare(ctx: &Context, decls: &[&str]) {
    for d in decls {
        let mut it = d.split(':');
        let name = it.next().unwrap();
        let asm: Vec<Assumption> = it
            .map(|a| match a {
                "pos" => Assumption::Positive,
                "real" => Assumption::Real,
                "int" => Assumption::Integer,
                _ => panic!("unknown assumption {a}"),
            })
            .collect();
        ctx.symbol_with(name, &asm).unwrap();
    }
}

/// `∫ src dx` has a closed form whose derivative is the integrand at the
/// real sample points where the integrand is real, with the parameters
/// `params` bound after integrating.  Returns the antiderivative.
fn antiderivative(decls: &[&str], src: &str, params: &[(&str, &str)]) -> (Context, Ex, Ex) {
    let ctx = Context::new();
    declare(&ctx, decls);
    let x = ctx.symbol("x");
    let f = ctx.parse(src).unwrap();
    let big_f = f.integrate(&x);
    assert!(!big_f.has_unevaluated(), "∫ {src} dx = {big_f}");
    let bind = |e: &Ex| {
        let mut e = e.clone();
        for (k, v) in params {
            e = e.subs(&ctx.symbol(k), &ctx.parse(v).unwrap());
        }
        e
    };
    let residual = bind(&(&big_f.diff(&x) - &f));
    let fb = bind(&f);
    let mut checked = 0;
    // Real values of the integrand first; a complex-valued integrand
    // (`sin x·e^{−ix}/x`) at every point.
    for real_only in [true, false] {
        for point in ["1/3", "7/5", "-5/7", "13/11", "5/2", "-9/4", "3/10"] {
            let p = ctx.parse(point).unwrap();
            let Ok(fv) = fb.subs(&x, &p).eval_complex64() else {
                continue;
            };
            let real = fv.im.abs() <= 1e-12 * fv.norm().max(1.0);
            if !fv.re.is_finite() || !fv.im.is_finite() || real != real_only {
                continue;
            }
            let r = residual.subs(&x, &p).eval_complex64().unwrap();
            assert!(
                r.norm() < 1e-9 * fv.norm().max(1.0),
                "∫ {src} dx = {big_f}: F′ − f = {r} at {point}"
            );
            checked += 1;
        }
        if checked > 0 {
            break;
        }
    }
    assert!(checked > 0, "∫ {src} dx = {big_f}: no sample point");
    (ctx, x, big_f)
}

/// `F(b) − F(a) = ∫ₐᵇ f` by quadrature, for the antiderivative of `src`
/// (no parameters), on an interval where `f` is finite.
fn continuous_on(src: &str, a: &str, b: &str) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse(src).unwrap();
    let big_f = f.integrate(&x);
    assert!(!big_f.has_unevaluated(), "∫ {src} dx = {big_f}");
    let (lo, hi) = (ctx.parse(a).unwrap(), ctx.parse(b).unwrap());
    let at = |p: &Ex| big_f.subs(&x, p).eval_complex64().unwrap();
    let diff = at(&hi) - at(&lo);
    let quad = f.integrate_numeric(&x, &lo, &hi).unwrap();
    assert!(
        (diff.re - quad).abs() < 1e-8 * quad.abs().max(1.0) && diff.im.abs() < 1e-8,
        "∫ {src} on [{a}, {b}]: F(b) − F(a) = {diff}, quadrature {quad}; F = {big_f}"
    );
}

/// `∫ₗₒʰⁱ src dx` with the parameters `params` substituted in the value
/// equals `expected` to 12 digits.
fn definite(decls: &[&str], src: &str, lo: &str, hi: &str, params: &[(&str, &str)], expected: f64) {
    let ctx = Context::new();
    declare(&ctx, decls);
    let x = ctx.symbol("x");
    let f = ctx.parse(src).unwrap();
    let v = f
        .try_integrate_definite(&x, &ctx.parse(lo).unwrap(), &ctx.parse(hi).unwrap())
        .unwrap_or_else(|e| panic!("∫ {src} on [{lo}, {hi}]: {e}"));
    let mut b = v.clone();
    for (k, val) in params {
        b = b.subs(&ctx.symbol(k), &ctx.parse(val).unwrap());
    }
    let z = b.eval_complex64().unwrap();
    assert!(
        (z.re - expected).abs() < 1e-12 * expected.abs().max(1.0) && z.im.abs() < 1e-12,
        "∫ {src} on [{lo}, {hi}] = {v} = {z}, expected {expected}"
    );
}

/// `∫ sin(kx)·sin(mx) dx` for integers `k`, `m` was wrong in its case
/// `m = |k|` (product to sum gave `sin((k − |k|)x)/(2(k − |k|))`, `0/0` for
/// every `k > 0`: the degenerate value came from `solve` as `±√(k²)`), and
/// `∫₀^π sin(kx)·sin(mx) dx` was `0` at `k = m = 2` (it is `π/2`).  Every
/// candidate with an integer parameter also passed the FTC check untested.
/// SymPy (test_integrals): `Piecewise((0, Eq(k, 0) | Eq(m, 0)), (−π/2, Eq(k,
/// −m)), (π/2, Eq(k, m)), (0, True))`; oracle: mpmath quadrature.
#[test]
fn product_of_sines_with_integer_frequencies() {
    antiderivative(
        &["k:int", "m:int"],
        "sin(k*x)*sin(m*x)",
        &[("k", "2"), ("m", "2")],
    );
    antiderivative(
        &["k:int", "m:int"],
        "sin(k*x)*sin(m*x)",
        &[("k", "2"), ("m", "-2")],
    );
    antiderivative(
        &["k:int", "m:int"],
        "sin(k*x)*sin(m*x)",
        &[("k", "2"), ("m", "3")],
    );
    let pi_2 = std::f64::consts::FRAC_PI_2;
    definite(
        &["k:int", "m:int"],
        "sin(k*x)*sin(m*x)",
        "0",
        "pi",
        &[("k", "2"), ("m", "2")],
        pi_2,
    );
    definite(
        &["k:int", "m:int"],
        "sin(k*x)*sin(m*x)",
        "0",
        "pi",
        &[("k", "2"), ("m", "-2")],
        -pi_2,
    );
    definite(
        &["k:int", "m:int"],
        "sin(k*x)*sin(m*x)",
        "0",
        "pi",
        &[("k", "2"), ("m", "3")],
        0.0,
    );
}

/// `∫₀^{2π} dt/(e^{it} − 2)` was `0`: `u = e^{it}` left `ln(e^{it})` and
/// `ln(e^{it} − 2)`, which jump at `t = π` and `t = 0`.  An imaginary rate now
/// goes the trigonometric way (`cos t + i·sin t`), whose routes keep the
/// answer continuous.  SymPy (test_integrals): `−π`; oracle: mpmath.
#[test]
fn exponential_of_an_imaginary_argument_stays_continuous() {
    let pi = std::f64::consts::PI;
    definite(&[], "1/(exp(I*x) - 2)", "0", "2*pi", &[], -pi);
    definite(&[], "1/(exp(I*x) - 1/2)", "0", "2*pi", &[], 0.0);
    definite(&["x:real"], "1/(exp(I*x) - 2)", "0", "2*pi", &[], -pi);
}

/// `∫ x·eˣ·ln x dx` stayed unevaluated (the three-factor rule took only
/// `u = x`, which needs `∫ Ei(x) dx`); also the ODE `y″ + 2y′ + y = ln x`.
/// By parts with `u = ln x` first now.  SymPy (test_integrals): `(x·eˣ −
/// eˣ)·ln x − eˣ + Ei(x)`.
#[test]
fn polynomial_times_exponential_times_logarithm() {
    antiderivative(&[], "x*exp(x)*ln(x)", &[]);
    antiderivative(&[], "x^2*exp(x)*ln(x)", &[]);
    antiderivative(&[], "x*sin(x)*ln(x)", &[]);
    antiderivative(&[], "(x + 1)*exp(2*x)*ln(x)", &[]);
}

/// `∫ Ei(x)`, `∫ Si(x)`, `∫ Ci(u)`, `∫ Shi(x)`, `∫ Chi(u)`, `∫ E₁(x)`,
/// `∫ E₂(x)`, `∫ E_y(x)`, `∫ x·asinh(x/2)` stayed unevaluated (no
/// special function was a by-parts `u`).  SymPy (test_meijerint,
/// test_heurisch): `x·Ei(x) − eˣ`, `x·Si(x) + cos x`, …, `−expint(2, x)`,
/// `−expint(y + 1, x)`.
#[test]
fn special_functions_by_parts() {
    for src in [
        "Ei(x)",
        "Si(x)",
        "Ci(x)",
        "Shi(x)",
        "Chi(x)",
        "Si(2*x + 1)",
        "x*Ei(x)",
        "x^2*Ci(3*x)",
        "expint(1, x)",
        "expint(2, x)",
        "x*asinh(x/2)",
    ] {
        antiderivative(&[], src, &[]);
    }
    antiderivative(&[], "expint(y, x)", &[("y", "6/5")]);
}

/// `∫ acsc x`, `∫ asec x`, `∫ acot x` (and of `a·x`) stayed unevaluated: the
/// parser writes them `asin(1/x)`, …, which had no rule for a non-linear
/// argument; by parts with `dv = dx` takes them now.  SymPy (test_manual):
/// `x·acsc(x) + …`.
#[test]
fn inverse_trigonometric_functions_of_reciprocals() {
    for src in ["acsc(x)", "asec(x)", "acot(x)", "acot(3*x)", "acsc(2*x)"] {
        antiderivative(&[], src, &[]);
    }
}

/// `∫ eˣ·cos x/x dx`, `∫ sin x·e^{−ix}/x dx`, `∫ e^{−x}·sin x/(x² + 1) dx`
/// stayed unevaluated: several exponential and trigonometric factors over
/// a rational function are now written as exponentials, `Ei` at complex
/// points, with the logarithm of `Ei` continued where its argument crosses
/// the cut (the integrand is finite there).  SymPy (test_integrals):
/// `Ei(x − i·x)/2 + Ei(x + i·x)/2`; oracle for the values: mpmath.
#[test]
fn exponential_and_trigonometric_factors_over_rational_functions() {
    antiderivative(&[], "exp(x)*cos(x)/x", &[]);
    antiderivative(&[], "sin(x)*exp(-I*x)/x", &[]);
    antiderivative(&[], "exp(-x)*sin(x)/(x^2 + 1)", &[]);
    antiderivative(&[], "exp(-x)*cos(x)/(x + 1)", &[]);
    antiderivative(&[], "cos(x)^2/x^2", &[]);
    continuous_on("exp(-x)*sin(x)/(x^2 + 1)", "-3", "3");
    continuous_on("exp(-x)*sin(x)/x", "-1", "2");
    continuous_on("sin(2*x)^2/x", "-1", "2");
    // mpmath: quad(lambda x: exp(-x)*sin(x)/(x**2+1), [0, inf])
    definite(
        &[],
        "exp(-x)*sin(x)/(x^2 + 1)",
        "0",
        "oo",
        &[],
        0.263_993_546_979_857_8,
    );
    // mpmath: quad(lambda x: exp(-x)*cos(x)/(x+1), [0, inf])
    definite(
        &[],
        "exp(-x)*cos(x)/(x + 1)",
        "0",
        "oo",
        &[],
        0.410_592_543_469_122_5,
    );
    // SymPy: -Si(3)/4 + 3*Si(1)/4
    definite(&[], "sin(x)^3/x", "0", "1", &[], 0.247_399_170_775_520_2);
}

/// `∫ √(5x² − 6x + 2) dx`, `∫ √(53225x² − 66732x + 23013) dx` stayed
/// unevaluated (only `√(a² ± x²)` had a rule), and with symbolic
/// coefficients `∫ dx/√(a + b·x + c·x²)`, `∫ √(a + b·x + c·x²) dx`,
/// `∫ x·asin(a·x) dx`.  SymPy (test_integrals, test_manual): the reduction to
/// `∫ dx/√Q` and `ln(2√c·√Q + 2c·x + b)/√c`.
#[test]
fn square_roots_of_quadratics() {
    antiderivative(&[], "sqrt(5*x^2 - 6*x + 2)", &[]);
    antiderivative(&[], "sqrt(23013 - 66732*x + 53225*x^2)", &[]);
    antiderivative(&[], "(2*x^2 + 1)*sqrt(3*x^2 + x + 1)", &[]);
    let abc = [("a", "6/5"), ("b", "3/4"), ("c", "7/3")];
    antiderivative(&[], "1/sqrt(a + b*x + c*x^2)", &abc);
    antiderivative(&[], "sqrt(a + b*x + c*x^2)", &abc);
    antiderivative(
        &[],
        "(d + e*x)/sqrt(a + b*x + c*x^2)",
        &[
            ("a", "6/5"),
            ("b", "3/4"),
            ("c", "7/3"),
            ("d", "5/7"),
            ("e", "9/4"),
        ],
    );
    antiderivative(&[], "x*asin(a*x)", &[("a", "3/4")]);
    antiderivative(&[], "x*acos(a*x)", &[("a", "3/4")]);
}

/// `∫ (b + t)^{−a} dt` stayed unevaluated (a symbolic exponent was taken
/// for `xⁿ` only).  SymPy (test_meijerint): `(b + t)^{1−a}/(1 − a)` with the
/// case `a = 1`.
#[test]
fn power_of_a_linear_function_with_a_symbolic_exponent() {
    antiderivative(&[], "(b + x)^(-a)", &[("a", "6/5"), ("b", "3/4")]);
    antiderivative(&[], "(2*x + 1)^n", &[("n", "7/3")]);
}

/// The moments of a normal density with symbolic `μ`, `σ` over `ℝ` were
/// refused: the exponent `−(x − μ)²/(2σ²)` was not seen as a quadratic,
/// and `x²·e^{−(x−μ)²/(2σ²)}` at `±∞` was `∞·0` for the compositional limits,
/// which ignored that the exponential beats every power.  Also `∫₀^∞
/// x⁵·e^{−x}·sin x dx`.  SymPy (test_meijerint): `μ² + σ²`, `μ³ + 3μσ²`,
/// `−15`.
#[test]
fn moments_and_exponential_decay_at_infinity() {
    let pdf = "x^2*exp(-(x - mu)^2/(2*s^2))/(sqrt(2*pi)*s)";
    // μ² + σ² at μ = 6/5, σ = 3/4
    definite(
        &["mu:real", "s:pos"],
        pdf,
        "-oo",
        "oo",
        &[("mu", "6/5"), ("s", "3/4")],
        1.44 + 0.5625,
    );
    let pdf3 = "x^3*exp(-(x - mu)^2/(2*s^2))/(sqrt(2*pi)*s)";
    // μ³ + 3μσ²
    definite(
        &["mu:real", "s:pos"],
        pdf3,
        "-oo",
        "oo",
        &[("mu", "6/5"), ("s", "3/4")],
        1.728 + 3.0 * 1.2 * 0.5625,
    );
    definite(&[], "x*exp(-x)*sin(x)", "0", "oo", &[], 0.5);
    definite(&[], "x^5*exp(-x)*sin(x)", "0", "oo", &[], -15.0);
}

/// `∫ x·e^{x⁴} dx`, `∫ x·sin(x⁴) dx`, `∫ x·sec(x²) dx` stayed unevaluated:
/// `u = x²` is no subexpression of the first two, and for the third only
/// `u = cos(x²)` was tried.  SymPy (test_manual): `√π·erfi(x²)/4`, …,
/// `ln(tan(x²) + sec(x²))/2`.
#[test]
fn power_substitutions() {
    for src in [
        "x*exp(x^4)",
        "x*exp(-x^4)",
        "x*sin(x^4)",
        "x/cos(x^2)",
        "x^3/cos(x^4)",
    ] {
        antiderivative(&[], src, &[]);
    }
}

/// Powers with the variable in the exponent are exponentials:
/// `∫ 2^x/(1 + 2^x) dx`, `∫ dt/(2^π·(1 + 2^t))`, `∫ a^{t−s}/(1 + a^t) dt`,
/// `∫ x^x·(1 + ln x) dx` stayed unevaluated, and `∫₀^∞ dx/(1 + 2^{2x/3}) =
/// 3/2` was refused.  SymPy (test_integrals, test_risch): `log(2^x + 1)/log
/// 2`, …, `x^x`, `3/2`.
#[test]
fn powers_with_variable_exponents() {
    antiderivative(&[], "2^x/(1 + 2^x)", &[]);
    antiderivative(&[], "1/(2^pi*(1 + 2^x))", &[]);
    antiderivative(&[], "a^(x - s)/(1 + a^x)", &[("a", "6/5"), ("s", "3/4")]);
    let (_, _, big_f) = antiderivative(&[], "x^x*(1 + ln(x))", &[]);
    assert_eq!(big_f.to_string(), "x^x");
    definite(&[], "1/(1 + 2^(2*x/3))", "0", "oo", &[], 1.5);
}

/// `∫ x^y·e^{−x} dx`, `∫ xⁿ·e^{−x} dx` (`n` an integer), `∫ x^{1/3}·e^{−x/8} dx`,
/// `∫ e^{−x³} dx`, `∫ erf(x³) dx` stayed unevaluated; the incomplete gamma
/// function `γ(s, z)` with `z^{−s}·γ(s, z)` entire keeps the answer
/// continuous at 0 (with `Γ(s, z)` it would step there).  SymPy
/// (test_integrals, test_manual, test_meijerint): `lowergamma`/`uppergamma`.
#[test]
fn incomplete_gamma_functions() {
    antiderivative(&[], "x^y*exp(-x)", &[("y", "6/5")]);
    antiderivative(&["n:int"], "x^n*exp(-x)", &[("n", "3")]);
    antiderivative(&[], "x^(1/3)*exp(-x/8)", &[]);
    antiderivative(&[], "exp(-x^3)", &[]);
    antiderivative(&[], "erf(x^3)", &[]);
    continuous_on("exp(-x^3)", "-1", "2");
    continuous_on("exp(-2*x^4)", "-1", "1");
}

/// The classical orthogonal polynomials of a symbolic degree were not
/// integrated.  SymPy (test_manual): `(P_{n+1} − P_{n−1})/(2n + 1)`,
/// `T_{n+1}/(n + 1)` for `U_n`, `H_{n+1}/(2(n + 1))`, … (checked by
/// differentiation at `n = 3`).
#[test]
fn orthogonal_polynomials_of_symbolic_degree() {
    for src in [
        "legendre(n, x)",
        "chebyshev_t(n, x)",
        "chebyshev_u(n, x)",
        "hermite(n, x)",
        "laguerre(n, x)",
        "assoc_laguerre(n, 7, x)",
        "gegenbauer(n, 7, x)",
        "jacobi(n, 7, 5/3, x)",
        "x*legendre(n, 2*x + 1)",
        "x*hermite(n, 2*x + 1)",
    ] {
        antiderivative(&[], src, &[("n", "3")]);
    }
}

/// `∫ x·J₀(x) dx`, `∫ x·I₀(x) dx`, `∫ J₁(x) dx`, `∫ J₁(x − 1) dx`,
/// `∫ x·J₁(x²) dx`, `∫ W(x) dx`, `∫ W(2x) dx` stayed unevaluated.  SymPy
/// (test_meijerint, test_heurisch): `x·J₁(x)`, `x·I₁(x)`, `−J₀(x)`, …,
/// `x·W(x) − x + x/W(x)`.
#[test]
fn bessel_functions_and_lambert_w() {
    for src in [
        "x*besselj(0, x)",
        "x*besseli(0, x)",
        "besselj(1, x)",
        "besselj(1, x - 1)",
        "x*besselj(1, x^2)",
        "x^3*besselj(2, 3*x)",
        "lambertw(x)",
        "lambertw(2*x)",
    ] {
        antiderivative(&[], src, &[]);
    }
}

/// `∫ |x² − 3x| dx`, `∫ e^{−|x|} dx`, `∫ e^{|x − 2|} dx`, `∫ min(e^{−x}, eˣ)² dx`
/// stayed unevaluated (one root only, polynomial factors only); they are
/// split at the kinks and joined continuously now.  `∫₋₁₅¹⁵ |x² − 3x| dx`
/// was refused: the quadrature guard missed the kinks.  The Laplace density
/// over `ℝ` with a real `μ` was refused for want of a test point.  SymPy
/// (test_integrals, test_meijerint): Piecewise answers, `2259`, `1`, `μ`.
#[test]
fn kinks_of_absolute_values_and_minima() {
    for src in [
        "abs(x^2 - 3*x)",
        "abs(x^2 - 1)",
        "exp(-abs(x))",
        "exp(abs(x - 2))",
        "min(exp(-x), exp(x))^2",
    ] {
        antiderivative(&["x:real"], src, &[]);
    }
    continuous_on("abs(x^2 - 3*x)", "-2", "5");
    continuous_on("exp(-abs(x))", "-2", "3");
    definite(&[], "abs(x^2 - 3*x)", "-15", "15", &[], 2259.0);
    definite(
        &["b:pos", "mu:real"],
        "exp(-abs(mu - x)/b)/(2*b)",
        "-oo",
        "oo",
        &[("b", "6/5"), ("mu", "3/4")],
        1.0,
    );
    definite(
        &["b:pos", "mu:real"],
        "x*exp(-abs(mu - x)/b)/(2*b)",
        "-oo",
        "oo",
        &[("b", "6/5"), ("mu", "3/4")],
        0.75,
    );
}

/// `∫₀^∞ dx/((x² + a²)(x² + b²))` for positive `a`, `b` was refused: the
/// case `b = a` of the antiderivative (`x/(2a²(a² + x²)) + atan(x/a)/(2a³)`)
/// was `∞·0` at infinity for the compositional limits (leading terms now).
/// `∫₀^∞ ln(x + 1)/(x² + 1) dx` was refused: the antiderivative's `Li₂` and
/// products of logarithms diverge like `ln²x` term by term (their
/// asymptotic expansions now cancel).  SymPy (test_integrals):
/// `π/(2ab(a + b))`; oracle for the second: mpmath, `π·ln 2/4 + G`.
#[test]
fn definite_integrals_at_infinity() {
    let pi = std::f64::consts::PI;
    let (a, b) = (1.2_f64, 0.75_f64);
    definite(
        &["a:pos", "b:pos"],
        "1/((x^2 + a^2)*(x^2 + b^2))",
        "0",
        "oo",
        &[("a", "6/5"), ("b", "3/4")],
        pi / (2.0 * a * b * (a + b)),
    );
    let catalan = 0.915_965_594_177_219;
    definite(
        &[],
        "ln(x + 1)/(x^2 + 1)",
        "0",
        "oo",
        &[],
        pi * 2f64.ln() / 4.0 + catalan,
    );
}

/// `∫_{3x}^{5x} x dx` was refused ("still depends on the integration
/// variable"): the variable of integration is a dummy.  SymPy
/// (test_integrals): `8x²`.
#[test]
fn bounds_in_the_variable_of_integration() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let v = x
        .try_integrate_definite(&x, &ctx.parse("3*x").unwrap(), &ctx.parse("5*x").unwrap())
        .unwrap();
    assert_eq!(v.to_string(), "8*x^2");
}

/// `∫ (ln(−1/y)/2 − ln(1/y)/2)/y − (ln(1 − 1/y)/2 − ln(1 + 1/y)/2)/y dy` came out
/// with a step of `π²/2` at `y = 0`, where the integrand tends to 1 from both
/// sides; it stays unevaluated now (a candidate that jumps at a real pole
/// of the argument of one of its logarithms while the integrand is finite
/// there is rejected).  SymPy 1.14 (test_risch, `integrate`) is wrong too.
#[test]
fn a_dilogarithm_answer_that_jumps_is_refused() {
    let ctx = Context::new();
    let y = ctx.symbol("y");
    let f = ctx
        .parse("(ln(-1/y)/2 - ln(1/y)/2)/y - (ln(1 - 1/y)/2 - ln(1 + 1/y)/2)/y")
        .unwrap();
    let big_f = f.integrate(&y);
    if !big_f.has_unevaluated() {
        let eps = ctx.parse("1/10^6").unwrap();
        let lo = big_f.subs(&y, &(-&eps)).eval_complex64().unwrap();
        let hi = big_f.subs(&y, &eps).eval_complex64().unwrap();
        assert!(
            (hi - lo).norm() < 1e-4,
            "jump {} in {big_f}",
            (hi - lo).norm()
        );
    }
}

/// The logarithmic substitution `w = ln(c·xⁿ)` needs a real `n`: for
/// `n = i` the 0.47 development returned an antiderivative of
/// `x·sec(a + 2·ln(c·x^i))³` that is right for a positive `c` and wrong for
/// a negative one (`F′ − f ≈ −4.7·10⁻⁷ − 2.8·10⁻⁷i` at `a = −6/5`, `c = −5/3`,
/// `x = 7/5`; Rubi 4.7.5 #181, #182, #223, #224 under `--negative-params`).
/// It stays unevaluated, as in 0.46.
#[test]
fn log_substitution_needs_a_real_exponent() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in [
        "x*sec(a + 2*ln(c*x^I))^3",
        "sec(a + 2*ln(c*x^(1/2*I)))^3",
        "x*csc(a + 2*ln(c*x^I))^3",
    ] {
        let f = ctx.parse(src).unwrap();
        let big_f = f.integrate(&x);
        if big_f.has_unevaluated() {
            continue;
        }
        let d = (big_f.diff(&x) - &f)
            .subs(&ctx.symbol("a"), &ctx.rational(-6, 5))
            .subs(&ctx.symbol("c"), &ctx.rational(-5, 3))
            .subs(&x, &ctx.rational(7, 5));
        let z = d.eval_complex64().unwrap();
        assert!(z.norm() < 1e-12, "{src}: {big_f}: F' - f = {z}");
    }
}
