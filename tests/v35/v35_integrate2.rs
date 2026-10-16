//! Special functions at algebraic points, and partial fractions with
//! parameters and radicals together (0.45): `∫ R(x)·f(a·x + b) dx` for `f` =
//! `exp`, `sin`, `cos`, `sinh`, `cosh`, `ln` and a rational `R` whose
//! denominator has a factor irreducible over ℚ of degree ≥ 2, summed over
//! its roots (in radicals, or as a `RootSum`); `apart` over `ℚ(y)(α)`; and
//! the algebraic remainder of the rational integrator, which was integrated
//! by parts.  Every antiderivative is checked by differentiation at sample
//! points and by `F(b) − F(a)` against the value of the integral on an
//! interval free of poles (mpmath's `quad`, quoted with each test).

use std::time::Duration;

use symplex::prelude::*;

use crate::perf_assert::assert_fast;

/// `∫ src dx`: evaluated, `F′ = f` at the real sample points where `f` is
/// real (`F′ − f` as one expression), and displayed as `expected`.
fn antiderivative(src: &str, expected: &str) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse(src).unwrap();
    let big_f = f.integrate(&x);
    assert!(!big_f.has_unevaluated(), "∫ {src} dx = {big_f}");
    let residual = &big_f.diff(&x) - &f;
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

/// `F(b) − F(a)` for `F = ∫ src dx` is the real number `value` (mpmath's
/// `quad` on `[a, b]`, quoted with each test).
fn definite(src: &str, a: &str, b: &str, value: f64) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let big_f = ctx.parse(src).unwrap().integrate(&x);
    let at = |p: &str| {
        big_f
            .subs(&x, &ctx.parse(p).unwrap())
            .eval_complex64()
            .unwrap()
    };
    let d = at(b) - at(a);
    assert!(
        (d.re - value).abs() < 1e-9 * value.abs().max(1.0) && d.im.abs() < 1e-9,
        "∫ {src} on [{a}, {b}]: F(b) − F(a) = {d}, expected {value}; F = {big_f}"
    );
}

/// `∫ e^{2x}/(x² − 2) dx` stayed unevaluated: the partial fractions over ℚ
/// stop at `x² − 2`.  Over its roots `±√2` (residues `±√2/4`) each term is
/// an `Ei`: SymPy (`integrate(exp(2*(u + sqrt(2)))/u, u)` =
/// `exp(2*sqrt(2))*Ei(2*u)` for `u = x − √2`; SymPy 1.14 leaves the
/// integral in `x` unevaluated); `F(1) − F(−1)` = −2.4674466210324260 and
/// `F(3) − F(2)` = 37.351472170676787 (mpmath `quad`).
#[test]
fn exponential_integral_at_irrational_roots() {
    let src = "exp(2*x)/(x^2 - 2)";
    antiderivative(
        src,
        "-1/4*sqrt(2)*exp(-2*sqrt(2))*Ei(2*x + 2*sqrt(2)) + 1/4*sqrt(2)*exp(2*sqrt(2))*Ei(2*x - 2*sqrt(2))",
    );
    definite(src, "-1", "1", -2.467446621032426);
    definite(src, "2", "3", 37.351472170676787);
}

/// `∫ sin x/(x² − 2) dx` and `∫ cosh x/(x² − 2) dx` stayed unevaluated.
/// SymPy (`integrate(sin(u + sqrt(2))/u, u)` = `sin(sqrt(2))*Ci(u) +
/// cos(sqrt(2))*Si(u)`); `∫₋₄⁻² sin x/(x² − 2)` = −0.14177878225722907,
/// `∫₂³ cosh x/(x² − 2)` = 1.509_163_605_816_511 (mpmath `quad`).
#[test]
fn sine_cosine_and_hyperbolic_integrals_at_irrational_roots() {
    let src = "sin(x)/(x^2 - 2)";
    antiderivative(
        src,
        "-1/4*sqrt(2)*(-sin(sqrt(2))*Ci(x + sqrt(2)) + cos(sqrt(2))*Si(x + sqrt(2))) + 1/4*sqrt(2)*(sin(sqrt(2))*Ci(x - sqrt(2)) + cos(sqrt(2))*Si(x - sqrt(2)))",
    );
    definite(src, "-4", "-2", -0.14177878225722907);
    let src = "cosh(x)/(x^2 - 2)";
    antiderivative(
        src,
        "-1/4*sqrt(2)*(1/2*exp(sqrt(2))*Ei(-x - sqrt(2)) + 1/2*exp(-sqrt(2))*Ei(x + sqrt(2))) + 1/4*sqrt(2)*(1/2*exp(sqrt(2))*Ei(x - sqrt(2)) + 1/2*exp(-sqrt(2))*Ei(-x + sqrt(2)))",
    );
    definite(src, "2", "3", 1.509_163_605_816_511);
}

/// A repeated irreducible factor: `∫ eˣ/(x² − 2)² dx` stayed unevaluated.
/// The pole order is lowered first, as Hermite's reduction does
/// (`eˣ·S` with `S = −x/(4(x² − 2))`), then the simple poles give `Ei`s.
/// SymPy (`integrate(exp(x)/(x**2 - 2)**2, x)`): unevaluated;
/// `F(1) − F(−1)` = 1.0111452945726958 (mpmath `quad`).
#[test]
fn repeated_irreducible_factor_lowered_first() {
    let src = "exp(x)/(x^2 - 2)^2";
    antiderivative(
        src,
        "-x*exp(x)/(4*(x^2 - 2)) + (-1/16*sqrt(2) + 1/8)*exp(sqrt(2))*Ei(x - sqrt(2)) + (1/16*sqrt(2) + 1/8)*exp(-sqrt(2))*Ei(x + sqrt(2))",
    );
    definite(src, "-1", "1", 1.0111452945726958);
}

/// The roots of a cubic factor that Cardano's formula writes with complex
/// cube roots although all three are real (`x³ − 3x² + 1`) stay implicit:
/// `RootSum` over the factor, evaluated at its certified (exactly real)
/// roots.  `∫ eˣ/((x + 1)(x³ − 3x² + 1)) dx` stayed unevaluated; the
/// residues are `(2t² − 6t + 1)/9` at the roots `t`, and
/// `∫₋₀.₅^0.5` = 1.6957340602857748, `∫₁^2.5` = −1.4008977033607758 (mpmath
/// `quad`; SymPy 1.14 leaves the integral unevaluated).
#[test]
fn implicit_roots_of_a_cubic_factor() {
    let src = "exp(x)/((x + 1)*(x^3 - 3*x^2 + 1))";
    antiderivative(
        src,
        "-1/3*exp(-1)*Ei(x + 1) + RootSum(__rs_t^3 - 3*__rs_t^2 + 1, __rs_t -> (2/9*__rs_t^2 - 2/3*__rs_t + 1/9)*exp(__rs_t)*Ei(-__rs_t + x))",
    );
    definite(src, "-1/2", "1/2", 1.695_734_060_285_775);
    definite(src, "1", "5/2", -1.400_897_703_360_776);
}

/// A polynomial numerator: the polynomial part times `eˣ` by parts, the
/// proper part over the roots.  `∫ (x³ + 1)·eˣ/(x² − 2) dx` stayed
/// unevaluated (SymPy 1.14 too).
#[test]
fn polynomial_part_and_irreducible_quadratic() {
    antiderivative(
        "(x^3 + 1)*exp(x)/(x^2 - 2)",
        "x*exp(x) - 1/4*sqrt(2)*exp(-sqrt(2))*Ei(x + sqrt(2)) + 1/4*sqrt(2)*exp(sqrt(2))*Ei(x - sqrt(2)) - exp(x) + exp(sqrt(2))*Ei(x - sqrt(2)) + exp(-sqrt(2))*Ei(x + sqrt(2))",
    );
}

/// `∫ ln(L)·R(x) dx` by Hermite's reduction and the residues of `R`:
/// `∫ ln(x)/(x² − 2)`, `∫ ln(x)/(x² + 1)` (a pair of conjugate roots, a real
/// total), `∫ ln(2x + 1)/(x − 3)` stayed unevaluated.  `∫ ln(u)/(u − c) du =
/// ln(u)·ln(1 − u/c) + Li₂(u/c)` (SymPy, `integrate(log(u)/(u - c), u)` for
/// `c ≠ 0`: the same up to a constant).  mpmath `quad`: `∫₀.₅^1.2 ln x/(x² − 2)` =
/// 0.073087085585086414, `∫₂⁴` = 0.35078636944686571 (beyond the pole at
/// `√2`), `∫₀.₂₅³ ln x/(x² + 1)` = −0.094938791145148081, `∫₀² ln(2x + 1)/(x − 3)`
/// = −1.2503676999103733, `∫₄⁶` = 2.5916620051242365.
#[test]
fn logarithm_times_rational_by_residues() {
    let src = "ln(x)/(x^2 - 2)";
    antiderivative(
        src,
        "-1/4*sqrt(2)*(ln(x)*ln(1/2*x*sqrt(2) + 1) + polylog(2, -1/2*x*sqrt(2))) + 1/4*sqrt(2)*(ln(x)*ln(-1/2*x*sqrt(2) + 1) + polylog(2, 1/2*x*sqrt(2)))",
    );
    definite(src, "1/2", "6/5", 0.073_087_085_585_086_4);
    definite(src, "2", "4", 0.350_786_369_446_865_7);
    let src = "ln(x)/(x^2 + 1)";
    antiderivative(
        src,
        "-1/2*(ln(x)*ln(x*I + 1) + polylog(2, -x*I))*I + 1/2*(ln(x)*ln(-x*I + 1) + polylog(2, x*I))*I",
    );
    definite(src, "1/4", "3", -0.094_938_791_145_148_08);
    let src = "ln(2*x + 1)/(x - 3)";
    antiderivative(
        src,
        "ln(2*x + 1)*ln(-2/7*x + 6/7) + polylog(2, 2/7*x + 1/7)",
    );
    definite(src, "0", "2", -1.250_367_699_910_373_3);
    definite(src, "4", "6", 2.591_662_005_124_236_5);
}

/// The roots of an irreducible quintic in a `RootSum`: `∫ ln(x)/(x⁵ − x + 1)
/// dx` stayed unevaluated; `∫₀.₅³` = −0.24318104619781744 (mpmath `quad`).
#[test]
fn logarithm_over_a_quintic_root_sum() {
    let src = "ln(x)/(x^5 - x + 1)";
    antiderivative(
        src,
        "RootSum(__rs_t^5 - __rs_t + 1, __rs_t -> (-320/2869*__rs_t^4 - 400/2869*__rs_t^3 - 500/2869*__rs_t^2 - 625/2869*__rs_t + 256/2869)*(ln(x)*ln(-x/__rs_t + 1) + polylog(2, x/__rs_t)))",
    );
    definite(src, "1/2", "3", -0.243_181_046_197_817_44);
}

/// A quartic factor whose roots Ferrari's formula writes in large radicals
/// stays implicit.  0.44 left `∫ ln(−x)·(−x² + 2x − 2)/(x⁴ − x² + 2x + 1)
/// dx` unevaluated; the residue sum over the explicit roots (while this was
/// written) was a 16 KB answer after 4.3 s (release), 3.4 s of them spent
/// by by parts on nested attempts before the residues were tried.  mpmath
/// `quad`: `∫₋₃⁻²` = −0.45305595746577703 (between the real roots −1.386…
/// and −0.426…: `∫₋₁^{−1/2}` = −1.2701555139941992).
#[test]
fn quartic_factor_with_large_radicals_stays_implicit() {
    let src = "ln(-x)*(-x^2 + 2*x - 2)/(x^4 - x^2 + 2*x + 1)";
    assert_fast!(
        Duration::from_secs(10),
        antiderivative(
            src,
            "RootSum(__rs_t^4 - __rs_t^2 + 2*__rs_t + 1, __rs_t -> (-71/106*__rs_t^3 + 49/106*__rs_t^2 + 23/106*__rs_t - 131/106)*(ln(-x)*ln(-x/__rs_t + 1) + polylog(2, x/__rs_t)))",
        ),
        "the antiderivative and its checks"
    );
    definite(src, "-3", "-2", -0.453_055_957_465_777);
    definite(src, "-1", "-1/2", -1.270_155_513_994_199);
}

/// The algebraic remainder `(−x³/6 + …)/(x⁴ − x³ + x² − x + 2)` of
/// `∫ dx/(x⁵ + x + 2)` went to the heuristic integrator, which took `u` =
/// its numerator and `dv` = `1/(x⁴ − …)` by parts and recursed 8 levels
/// through by parts and sum distribution before it gave up: 0.60 s of CPU
/// for an answer found without it in 0.21 s (release).  The answers are
/// unchanged (and checked by differentiation); the limits are generous for
/// a debug build on a loaded machine.
#[test]
fn rational_functions_are_not_integrated_by_parts() {
    let cases = [
        (
            "1/(x^5 + x + 2)",
            Some(
                "1/6*ln(abs(x + 1)) + RootSum(8376*__rs_t^4 + 1396*__rs_t^3 + 206*__rs_t^2 + 21*__rs_t + 1, __rs_t -> __rs_t*ln(33504/125*__rs_t^3 - 2792/125*__rs_t^2 + 1522/125*__rs_t + x + 16/125))",
            ),
        ),
        ("ln(x + x^(-3) - 1)", None),
        ("(x + x^(-3) - 1)^(-3)", None),
        (
            "x^3/(x^4 - x^3 + 1)",
            Some(
                "RootSum(229*__rs_t^4 - 229*__rs_t^3 + 96*__rs_t^2 - 16*__rs_t + 1, __rs_t -> __rs_t*ln(3664/27*__rs_t^3 - 916/9*__rs_t^2 + 283/9*__rs_t + x - 64/27))",
            ),
        ),
    ];
    for (src, expected) in cases {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let f = ctx.parse(src).unwrap();
        let big_f = assert_fast!(Duration::from_secs(15), f.integrate(&x), "∫ {src}");
        assert!(!big_f.has_unevaluated(), "∫ {src} dx = {big_f}");
        let residual = &big_f.diff(&x) - &f;
        for point in ["1/3", "7/5", "-5/7"] {
            let point = ctx.parse(point).unwrap();
            let r = residual.subs(&x, &point).eval_complex64().unwrap();
            let s = f.subs(&x, &point).eval_complex64().unwrap();
            assert!(
                r.norm() < 1e-9 * s.norm().max(1.0),
                "∫ {src} = {big_f}: F′ − f = {r} at {point}"
            );
        }
        if let Some(expected) = expected {
            assert_eq!(big_f.to_string(), expected, "∫ {src} dx");
        }
    }
}

/// `expr.partial_fractions(x)` is `expected`, and equal to `expr` at sample
/// points (the parameters at `6/5`, `3/4`), and so is the result combined
/// again by `together`.
fn apart_is(src: &str, expected: &str) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse(src).unwrap();
    let parts = f.partial_fractions(&x);
    let combined = parts.together();
    let bind = |e: &Ex, p: &Ex| {
        e.subs(&x, p)
            .subs(&ctx.symbol("a"), &ctx.rational(6, 5))
            .subs(&ctx.symbol("b"), &ctx.rational(3, 4))
            .eval_complex64()
            .unwrap()
    };
    for point in ["1/3", "7/5", "-5/7", "13/11"] {
        let p = ctx.parse(point).unwrap();
        let v = bind(&f, &p);
        for (what, e) in [("apart", &parts), ("together(apart)", &combined)] {
            let w = bind(e, &p);
            assert!(
                (v - w).norm() < 1e-10 * v.norm().max(1.0),
                "{what}({src}) = {e}: {w} against {v} at {point}"
            );
        }
    }
    assert_eq!(parts.to_string(), expected, "apart({src})");
}

/// Parameters and radicals together: `apart(a/((x − √2)(x + 1)), x)` came
/// back unchanged (neither the decomposition over `ℚ(a)` nor the one over
/// `ℚ(√2)` takes both).  Over `ℚ(a)(√2)`: SymPy
/// (`apart(a/((x - sqrt(2))*(x + 1)), x, full=True).doit()` =
/// `(a/(2*sqrt(2) + 3) + sqrt(2)*a/(2*sqrt(2) + 3))/(x - sqrt(2)) +
/// (-sqrt(2)*a/(2*sqrt(2) + 3) - a/(2*sqrt(2) + 3))/(x + 1)`, the same
/// coefficients `±a·(√2 − 1)`; its default `apart` leaves it unchanged).
#[test]
fn apart_over_parameters_and_radicals() {
    apart_is(
        "a/((x - sqrt(2))*(x + 1))",
        "(-a*sqrt(2) + a)/(x + 1) + (a*sqrt(2) - a)/(x - sqrt(2))",
    );
    // SymPy (full=True): (a**2/(a**2 + 2*sqrt(2)*a + 2) + sqrt(2)*a/(a**2 +
    // 2*sqrt(2)*a + 2))/(x - sqrt(2)) + (…)/(a + x), i.e. ±a/(a + √2).
    apart_is(
        "a/((x - sqrt(2))*(x + a))",
        "(-a^2 + a*sqrt(2))/((a^2 - 2)*(a + x)) + (a^2 - a*sqrt(2))/((a^2 - 2)*(x - sqrt(2)))",
    );
    // Two radicals: the coefficients written in the basis 1, √2, √3, √6.
    apart_is(
        "1/((x - sqrt(2))*(x - sqrt(3))*(x + b))",
        "(b^2 - b*sqrt(2) - b*sqrt(3) + sqrt(6))/((b^2 - 2)*(b^2 - 3)*(b + x)) + (-b*sqrt(2) - b*sqrt(3) + sqrt(6) + 2)/((b^2 - 2)*(x - sqrt(2))) + (b*sqrt(2) + b*sqrt(3) - sqrt(6) - 3)/((b^2 - 3)*(x - sqrt(3)))",
    );
    // A repeated factor over ℚ(a)(i).
    apart_is(
        "a/((x - I)^2*(x + 1))",
        "-a*I/(2*(x + 1)) + a*I/(2*(x - I)) + (x - I)^(-2)*(-1/2*a*I + 1/2*a)",
    );
}
