//! Numerical integration of divergent improper integrals, and `eval`/`evalf`
//! parity with SymPy's `test_evalf.py` and `functions/*/tests` and with
//! mpmath (the evalf hunt after 0.46).  Every doc comment gives the value
//! before and the oracle: SymPy 1.14, or mpmath 1.3 at two precisions.

use std::time::Duration;

use symplex::prelude::*;

use crate::perf_assert::{CpuTimer, assert_fast};

fn p(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s)
        .unwrap_or_else(|e| panic!("`{s}` does not parse: {e}"))
}

fn integral(ctx: &Context, body: &str, lo: &str, hi: &str) -> Ex {
    p(ctx, &format!("Integral({body}, x, {lo}, {hi})"))
}

fn divergent<T: std::fmt::Debug>(r: &Result<T, SymplexError>) -> bool {
    matches!(r, Err(SymplexError::Divergent { .. }))
}

/// Before: `eval_f64` of `∫_{−∞}^{∞} |x|/((x² + 1)π) dx` was `225.93…`, of
/// `∫_{−∞}^{∞} 1/(√(x² + 1)·π) dx` `226.37…`, of `∫₀^∞ x/(x² + 1) dx`
/// `354.89…` (the compiled integrand is 0 once `x²` overflows, and the
/// smoothed `f64` quadrature resolved the integral cut off there), and
/// `∫_{−∞}^{∞} (x + 4)/(x² + 5) dx` `5.6198…` (its principal value, both
/// halves cut at the same place); `integrate_numeric` gave the same, and
/// `709.78…` for `|x|/(x² + 1)`, `235.88…` for `∫₂^∞ √x/√(x³ + 1) dx`.  All
/// diverge (logarithmically): `x·f(x)` tends to a nonzero limit.  Now
/// `Divergent`, from every route.
#[test]
fn divergent_improper_integrals_are_not_numbers() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (body, lo, hi) in [
        ("abs(x)/((x^2 + 1)*pi)", "-oo", "oo"),
        ("1/(sqrt(x^2 + 1)*pi)", "-oo", "oo"),
        ("x/(x^2 + 1)", "0", "oo"),
        ("(x + 4)/(x^2 + 5)", "-oo", "oo"),
        ("x^(1/2)/(x^3 + 1)^(1/2)", "2", "oo"),
        ("1/(x*log(x))", "2", "oo"),
    ] {
        let i = integral(&ctx, body, lo, hi);
        let r = i.eval_f64();
        assert!(divergent(&r), "eval_f64 of ∫ {body} on [{lo}, {hi}]: {r:?}");
        let r = i.eval_decimal(20);
        assert!(divergent(&r), "eval_decimal of ∫ {body}: {r:?}");
        let f = p(&ctx, body);
        let r = f.integrate_numeric(&x, &p(&ctx, lo), &p(&ctx, hi));
        assert!(divergent(&r), "integrate_numeric of ∫ {body}: {r:?}");
    }
}

/// Before: `integrate_numeric` of `∫₂^∞ 2x/((x² − x + 2)·ln²x) dx`, which
/// converges like `∫ dx/(x ln²x)`, was `3.0924341688825505`: the `f64`
/// integrand is cut off where `x·ln²x` overflows, and the tail beyond
/// (`≈ 2/ln(10³⁰⁸)`) is missing.  mpmath (x = e^s, s ≤ 4000, dps 35):
/// `3.0760…` (two precisions agree to 4 digits only).  Now refused: no
/// certified quadrature reaches that tail.
#[test]
fn a_logarithmic_tail_beyond_the_f64_range_is_refused() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = p(&ctx, "2*x*log(x)^(-2)/(x^2 - x + 2)");
    assert!(
        f.integrate_numeric(&x, &ctx.int(2), &ctx.infinity())
            .is_err()
    );
    assert!(
        integral(&ctx, "2*x*log(x)^(-2)/(x^2 - x + 2)", "2", "oo")
            .eval_f64()
            .is_err()
    );
}

/// Convergent improper integrals keep their `f64` values and speed.
/// Before: the same.  Oracle: `√π`, `π`, `π⁴/15 = 6.4939394022668291491`
/// (mpmath dps 60, 90), `∫₁^∞ x^(−6/5) dx = 5`, `∫_ℝ (x² + 1)^(−3/4) dx =
/// √π·Γ(1/4)/Γ(3/4) = 5.244115108584239…`.
#[test]
fn convergent_improper_integrals_keep_their_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (body, lo, hi, want) in [
        ("exp(-x^2)", "-oo", "oo", std::f64::consts::PI.sqrt()),
        ("1/cosh(x)", "-oo", "oo", std::f64::consts::PI),
        (
            "x^3/(exp(x) - 1)",
            "0",
            "oo",
            std::f64::consts::PI.powi(4) / 15.0,
        ),
        ("x^(-6/5)", "1", "oo", 5.0),
        ("1/(x^2 + 1)^(3/4)", "-oo", "oo", 5.244_115_108_584_239),
    ] {
        let v = assert_fast!(
            Duration::from_secs(5),
            integral(&ctx, body, lo, hi).eval_f64(),
            "∫ {body}"
        )
        .unwrap();
        assert!((v - want).abs() < 1e-12 * want, "∫ {body} = {v}");
        let v = p(&ctx, body)
            .integrate_numeric(&x, &p(&ctx, lo), &p(&ctx, hi))
            .unwrap();
        assert!(
            (v - want).abs() < 1e-9 * want,
            "integrate_numeric ∫ {body} = {v}"
        );
    }
}

/// Before: `sin(e·10¹⁰⁰)` was refused at every precision
/// (`PrecisionExhausted`: an inexact argument beyond `2⁶⁴`).  SymPy
/// `test_evalf.py`: `NS('sin(E*10**100)', 15)` = `0.409160531722613`;
/// mpmath (dps 200, 300): `0.40916053172261292027243059688 08`.
#[test]
fn sine_of_a_huge_inexact_argument() {
    let ctx = Context::new();
    let s = p(&ctx, "sin(E*10^100)");
    assert_eq!(s.eval_decimal(15).unwrap(), "0.409160531722613");
    assert_eq!(
        s.eval_decimal(30).unwrap(),
        "0.409160531722612920272430596881"
    );
}

/// Before: `erf(x) − 1` was refused for `84 < x < 38000` (beyond the
/// precision search, inside the exponent range).  mpmath (dps 20, 30):
/// `−erfc(100) = −6.405961424921732039e−4346`, `erfc(90) =
/// 1.0276668174027325739e−3520`, `−erfc(38300) =
/// −4.0659797940400019737e−637062238`.
#[test]
fn erf_minus_one_far_from_zero() {
    let ctx = Context::new();
    assert_eq!(
        p(&ctx, "erf(100) - 1").eval_decimal(15).unwrap(),
        "-6.40596142492173e-4346"
    );
    assert_eq!(
        p(&ctx, "1 + erf(-90)").eval_decimal(15).unwrap(),
        "1.02766681740273e-3520"
    );
    let far = p(&ctx, "erf(38300) - 1").eval_decimal(15).unwrap();
    assert!(far.starts_with("-4.06597979404"), "{far}");
    assert!(far.ends_with("e-637062238"), "{far}");
}

/// Before: `erfc(14i)` was `5.34725051942776e83*i` (the asymptotic expansion
/// taken at `−14i`, across the cut of `Γ(1/2, s²)`), and so
/// `uppergamma(-7/2, -401/2)`, which `eval` writes with `erfc(i·√802/2)`,
/// was `2.57061464096165e85*i` — wrong digits and magnitude.  mpmath (dps 20,
/// 30): `erfc(14j)` = `1 − 5.3472505194277552785e83j`, `erfc(-14j)` = `1 +
/// 5.34725…e83j`; `gammainc(-3.5, -200.5)` = `0.27008820585226911 −
/// 5.3264597096208368435e76j` (mpmath's `test_functions2.py`).  Up to 0.53
/// the real parts `1` and `0.27…` were left out (one error bound for both
/// parts, wider than them); since 0.54 `erfc` of an imaginary argument has
/// an exact real part (`erfc(iy) = 1 − i·erfi(y)`).
#[test]
fn erfc_on_the_imaginary_axis_takes_the_right_side() {
    let ctx = Context::new();
    assert_eq!(
        p(&ctx, "erfc(14*I)").eval_decimal(15).unwrap(),
        "1 - 5.34725051942776e83*i"
    );
    assert_eq!(
        p(&ctx, "erfc(-14*I)").eval_decimal(15).unwrap(),
        "1 + 5.34725051942776e83*i"
    );
    assert_eq!(
        p(&ctx, "uppergamma(-7/2, -401/2)")
            .eval_decimal(15)
            .unwrap(),
        "0.270088205852269 - 5.32645970962084e76*i"
    );
}

/// Before: `floor(E)`, `ceiling(exp(2))`, `floor(80782*sqrt(2))` and
/// `floor(50!/e)` stayed.  SymPy `test_integers.py`, `test_evalf.py`:
/// `floor(E)` → 2, `floor(-pi)` → −4, `ceiling(exp(2))` → 8,
/// `floor(80782*sqrt(2))` → 114242, `floor(factorial(50)/E)` →
/// 11188719610782480504630258070757734324011354208865721592720336800
/// (mpmath dps 90).  `ceiling(10*(sin(1)^2 + cos(1)^2))` (exactly 10, not
/// provable here) stays (SymPy: 10).
#[test]
fn floor_and_ceiling_of_real_constants() {
    let ctx = Context::new();
    for (s, want) in [
        ("floor(E)", "2"),
        ("floor(-pi)", "-4"),
        ("ceiling(exp(2))", "8"),
        ("floor(80782*sqrt(2))", "114242"),
        (
            "floor(factorial(50)/exp(1))",
            "11188719610782480504630258070757734324011354208865721592720336800",
        ),
    ] {
        assert_eq!(p(&ctx, s).eval().to_string(), want, "{s}");
    }
    let stays = p(&ctx, "ceiling(10*(sin(1)^2 + cos(1)^2))").eval();
    assert!(stays.to_string().starts_with("ceiling("), "{stays}");
}

/// The integer part of a huge or tiny constant: the fold read the printed
/// value as an exact rational before looking at its size, and `Shi(12!) ≈
/// 5.27·10^208027742` raised 10 to that power — `floor(Shi(factorial(12)))`
/// never finished (the `fuzz_evalf` timeout of the 0.47 release campaign).
/// Now an integer beyond 1,000 digits stays unevaluated at once, and a value
/// below `10⁻⁵` in modulus is settled by its sign: `floor(exp(-10^6))` = 0,
/// `ceiling(exp(-10^6))` = 1, `floor(-exp(-10^6))` = −1 (`e^{−10⁶} ≈
/// 3.3·10^−434295`, mpmath).
#[test]
fn floor_of_huge_and_tiny_constants_is_quick() {
    let ctx = Context::new();
    let t = CpuTimer::start();
    for s in ["floor(Shi(factorial(12)))", "ceiling(exp(10^7))"] {
        let e = p(&ctx, s).eval();
        assert!(
            e.to_string().starts_with("floor(") || e.to_string().starts_with("ceiling("),
            "{s} = {e}"
        );
    }
    for (s, want) in [
        ("floor(exp(-10^6))", "0"),
        ("ceiling(exp(-10^6))", "1"),
        ("floor(-exp(-10^6))", "-1"),
        ("ceiling(-exp(-10^6))", "0"),
    ] {
        assert_eq!(p(&ctx, s).eval().to_string(), want, "{s}");
    }
    t.assert_within(Duration::from_secs(20), "six floors and ceilings");
}

/// Before: `log(8, 2)` was `ln(8)/ln(2)`, `beta(1, 5/2)` and `beta(1/2,
/// 1/2)` stayed, `uppergamma(4, 0)` was `Gamma(4)`.  SymPy
/// `test_exponential.py`, `test_beta_functions.py`, `test_gamma_functions.py`:
/// `log(8, 2)` → 3, `log(9, 27)` → 2/3, `beta(1, 5/2)` → 2/5, `beta(1/2,
/// 1/2)` → pi, `uppergamma(4, 0)` → 6.
#[test]
fn exact_folds_sympy_makes() {
    let ctx = Context::new();
    for (s, want) in [
        ("log(8, 2)", "3"),
        ("log(9, 27)", "2/3"),
        ("beta(1, 5/2)", "2/5"),
        ("beta(1/2, 1/2)", "pi"),
        ("uppergamma(4, 0)", "6"),
    ] {
        assert_eq!(p(&ctx, s).eval().to_string(), want, "{s}");
    }
}

/// Before: `cos(−√2/2)`, `cos(−2)`, `sin(−1)`, `sinh(−1)`, `atanh(−1/2)`,
/// `erf(−2)` kept the sign of a constant argument (`cos(−√2)` lost it).
/// SymPy: `cos(-sqrt(2)/2)` → `cos(sqrt(2)/2)`, `sinh(-1)` → `-sinh(1)`,
/// `atanh(-1/2)` → `-atanh(1/2)`, `erf(-2)` → `-erf(2)`.  A symbolic
/// argument keeps its coefficient (`sin(−2x)`).
#[test]
fn constant_arguments_give_up_their_sign() {
    let ctx = Context::new();
    for (s, want) in [
        ("cos(-sqrt(2)/2)", "cos(1/2*sqrt(2))"),
        ("cos(-2)", "cos(2)"),
        ("sin(-1)", "-sin(1)"),
        ("sinh(-1)", "-sinh(1)"),
        ("atanh(-1/2)", "-atanh(1/2)"),
        ("erf(-2)", "-erf(2)"),
        ("asin(-1/3)", "-asin(1/3)"),
        ("sin(-2*x)", "sin(-2*x)"),
    ] {
        assert_eq!(p(&ctx, s).eval().to_string(), want, "{s}");
    }
}

/// Before: `exp(2*log(x))` stayed (`exp(log(x))` folded to `x`).  SymPy:
/// `exp(2*log(x))` → `x**2`, `exp(log(x)/2)` → `sqrt(x)`, `exp(x +
/// 2*log(y))` → `y**2*exp(x)`; `exp(x*log(2))` stays.
#[test]
fn exponentials_of_multiples_of_a_logarithm_fold_generically() {
    let ctx = Context::new();
    for (s, want) in [
        ("exp(2*log(x))", "x^2"),
        ("exp(log(x)/2)", "sqrt(x)"),
        ("exp(x + 2*log(y))", "y^2*exp(x)"),
        ("exp(x*log(2))", "exp(x*log(2))"),
    ] {
        assert_eq!(p(&ctx, s).eval(), p(&ctx, want).eval(), "{s}");
    }
}

/// Before: `eval` left `nan == -1`, while Boolean `simplify` gave `False`.
/// SymPy: `Eq(nan, -1)` → `False`, `Ne(nan, 2)` → `True`.
#[test]
fn relations_with_nan_are_decided_by_eval() {
    let ctx = Context::new();
    let eq = ctx.parse_bool("Eq(nan, -1)").unwrap();
    assert_eq!(eq.eval().to_string(), "False");
    assert_eq!(eq.simplify().to_string(), "False");
    let ne = ctx.parse_bool("Ne(nan, 2)").unwrap();
    assert_eq!(ne.eval().to_string(), "True");
}

/// Before: `sin(104π/105)`, `cos(106π/105)`, `tan(10π/7)`, `sinh(iπ/105)`,
/// `tan(iπ)`, `exp(23πi/9)` stayed, and `exp(37πi/8) − exp(−11πi/8)` was
/// not 0.  SymPy `test_trigonometric.py`, `test_hyperbolic.py`,
/// `test_exponential.py`: `sin(pi*104/105)` → `sin(pi/105)`,
/// `cos(pi*106/105)` → `-cos(pi/105)`, `tan(pi*10/7)` → `tan(3*pi/7)`,
/// `sec(9*pi/7)` → `-1/cos(2*pi/7)`, `sinh(pi*I/105)` → `I*sin(pi/105)`,
/// `tan(pi*I)` → `I*tanh(pi)`, `exp(I*pi*46/18)` → `exp(5*I*pi/9)`,
/// `exp(2 - 17*I*pi/5)` → `exp(2 + 3*I*pi/5)`,
/// `log(cos(7*pi/12) + I*sin(7*pi/12))` → `7*I*pi/12`.
#[test]
fn rational_multiples_of_pi_are_reduced() {
    let ctx = Context::new();
    for (s, want) in [
        ("sin(104*pi/105)", "sin(pi/105)"),
        ("cos(106*pi/105)", "-cos(pi/105)"),
        ("tan(10*pi/7)", "tan(3*pi/7)"),
        ("sec(9*pi/7)", "-1/cos(2*pi/7)"),
        ("sinh(pi*I/105)", "I*sin(pi/105)"),
        ("cosh(-pi*I/105)", "cos(pi/105)"),
        ("tan(pi*I)", "I*tanh(pi)"),
        ("exp(23*pi*I/9)", "exp(5*pi*I/9)"),
        ("exp(37*pi*I/8) - exp(-11*pi*I/8)", "0"),
        ("exp(2 - 17*pi*I/5)", "exp(2 + 3*pi*I/5)"),
        ("log(cos(7*pi/12) + I*sin(7*pi/12))", "7*pi*I/12"),
    ] {
        assert_eq!(p(&ctx, s).eval(), p(&ctx, want).eval(), "{s}");
    }
}

/// Before (0.46): `solve` of this septic (a quartic factor by Ferrari's
/// formula) took 32 ms in a release build, 1 ms in 0.43: the realness of
/// each radicand went through the minimal polynomial (a Zassenhaus
/// factorisation); now the exact conjugation test decides it first (7 ms).
/// The roots are unchanged; the bound is loose for debug builds.
#[test]
fn ferrari_radicands_are_decided_real_quickly() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let poly = p(
        &ctx,
        "-9*x^7 - 3*x^6 + 99*x^5 - 75*x^4 - 52*x^3 - 180*x^2 + 180*x - 20",
    );
    let roots = assert_fast!(Duration::from_secs(20), poly.solve(&x)).unwrap();
    assert_eq!(roots.len(), 7);
    for r in &roots {
        let v = poly.subs(&x, r).eval_complex64().unwrap();
        assert!(v.norm() < 1e-8, "{r}: {v}");
    }
}

/// `re(g)·∞` and `im(g)·∞` take their directions from one certified value
/// of `g` (0.47; before, from two, and the orientation is unchanged).
/// mpmath (dps 30, 50): `gamma(gamma(1+2j))` = `6.0283701927710527024704 −
/// 0.8287027486742571590456j`: `re(g)·∞ + im(g)·∞ = ∞ − ∞ = nan`,
/// `re(g)·∞ − im(g)·∞ = ∞`.
#[test]
fn directions_of_infinity_from_one_value() {
    let ctx = Context::new();
    let g = "gamma(gamma(1+2*I))";
    assert_eq!(
        p(&ctx, &format!("re({g})*oo + im({g})*oo")).to_string(),
        "nan"
    );
    assert_eq!(
        p(&ctx, &format!("re({g})*oo - im({g})*oo")).to_string(),
        "oo"
    );
    assert_eq!(p(&ctx, &format!("im({g})*x*oo")).to_string(), "-x*oo");
}
