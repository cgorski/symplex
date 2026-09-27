//! Integration of equivalent forms (0.31, third pass): an integrand that
//! integrated in one normal form stayed unevaluated in another —
//! exponentials with an irrational rate written as a sum, a sum factor
//! times an exponential, a parametric denominator with a monomial content,
//! trigonometric powers of a linear argument, products of exponentials,
//! `exp(x)^c`, fractions with a common factor; the `RootSum` of an
//! irreducible parametric denominator; `ratsimp` of a vanishing
//! denominator; the sign of products of non-negative factors.
//!
//! Each test says what was wrong before; every reference value cites the
//! oracle call that produced it (SymPy 1.14, mpmath 1.3.0).  The
//! normal-form hunter (`examples/zz_hunt_forms.rs`) found the classes.

// Reference values are quoted at the digits the oracle printed them.
#![allow(clippy::excessive_precision)]

use symplex::prelude::*;

fn p(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap_or_else(|e| panic!("{s}: {e}"))
}

/// `F′ = f` (to `1e-9`) at the real sample points, with the parameters
/// bound first (as in `v30_integrate2.rs`).
fn assert_antiderivative_at(
    ctx: &Context,
    f: &Ex,
    big_f: &Ex,
    x: &Ex,
    binding: &[(&str, &str)],
    points: &[&str],
) {
    assert!(
        !big_f.has_unevaluated(),
        "∫ {f} stayed unevaluated: {big_f}"
    );
    let bind = |e: &Ex| {
        binding.iter().fold(e.clone(), |acc, (s, v)| {
            acc.subs(&ctx.symbol(s), &p(ctx, v))
        })
    };
    let residual = bind(big_f).diff(x) - bind(f);
    for pt in points {
        let v = residual
            .subs(x, &p(ctx, pt))
            .eval_complex64()
            .unwrap_or_else(|e| panic!("∫ {f} = {big_f} at {binding:?}: residual at {pt}: {e}"));
        assert!(
            v.norm() < 1e-9,
            "∫ {f} = {big_f} at {binding:?}: F′ − f = {v} at x = {pt}"
        );
    }
}

const BINDINGS: [[(&str, &str); 2]; 4] = [
    [("a", "7/5 + 3/11*I"), ("b", "2/3 - 5/7*I")],
    [("a", "-1/2 + 3/2*I"), ("b", "3/4 + 2/5*I")],
    [("a", "6/5"), ("b", "3/4")],
    [("a", "-6/5"), ("b", "-3/4")],
];
const POINTS: [&str; 5] = ["1/3", "7/5", "-5/7", "13/4", "-13/4"];

/// `∫ f dx` evaluates and `F′ = f` at every binding.
fn assert_integrates(ctx: &Context, f: &str) -> Ex {
    let x = ctx.symbol("x");
    let f = p(ctx, f);
    let big_f = f.integrate(&x);
    for binding in &BINDINGS {
        assert_antiderivative_at(ctx, &f, &big_f, &x, binding, &POINTS);
    }
    big_f
}

/// `F(hi) − F(lo)` to `digits` against the oracle's value.
fn assert_difference(big_f: &Ex, var: &Ex, lo: &str, hi: &str, want: &str, ctx: &Context) {
    let d = big_f.subs(var, &p(ctx, hi)) - big_f.subs(var, &p(ctx, lo));
    let got: f64 = d.eval_decimal(25).unwrap().parse().unwrap();
    let want: f64 = want.parse().unwrap();
    assert!(
        (got - want).abs() < 1e-15 * want.abs().max(1.0),
        "F({hi}) − F({lo}) = {got}, want {want} (F = {big_f})"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Item 1: the four normal-form gaps
// ═══════════════════════════════════════════════════════════════════════════

/// `∫ exp(−t·√37/2 − t/2) dt` stayed unevaluated while
/// `∫ exp(−t·(√37 + 1)/2) dt` worked: the rate of a linear argument was
/// read only from a single `c·t` term, and the canonical sum keeps
/// `−t·√37/2` and `−t/2` apart.
///
/// SymPy: `integrate(exp(-t*sqrt(37)/2 - t/2), t)` →
/// `-2*exp(-sqrt(37)*t/2 - t/2)/(1 + sqrt(37))`;
/// `integrate(…, (t, 0, 1)).evalf(30)` → `0.274194346117671461773675164653`.
#[test]
fn exponential_with_an_irrational_rate_written_as_a_sum() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let f = p(&ctx, "exp(-1/2*t*sqrt(37) - 1/2*t)");
    let big_f = f.integrate(&t);
    assert_antiderivative_at(&ctx, &f, &big_f, &t, &[], &POINTS);
    assert_difference(
        &big_f,
        &t,
        "0",
        "1",
        "0.274194346117671461773675164653",
        &ctx,
    );
    for s in [
        "exp(t*sqrt(37) + t)",
        "exp(t*sqrt(2) - t/3 + 1)*sin(t)",
        "t*exp(t*sqrt(5) + 2*t)",
    ] {
        let f = p(&ctx, s);
        let big_f = f.integrate(&t);
        assert_antiderivative_at(&ctx, &f, &big_f, &t, &[], &POINTS);
    }
}

/// `∫ (sin t + cos t)·eᵗ dt` and `∫ t·(sin t + cos t)·eᵗ dt` stayed
/// unevaluated while the distributed sums integrated: no rule split a sum
/// factor of a product (by parts wants a polynomial or a logarithm for
/// `u`; the cyclic rule does not see `(−cos t − sin t)·eᵗ` as `−1` times
/// the integrand).
///
/// SymPy: `integrate((sin(t) + cos(t))*exp(t), t)` → `exp(t)*sin(t)`;
/// `integrate(t*(sin(t) + cos(t))*exp(t), t)` →
/// `t*exp(t)*sin(t) - exp(t)*sin(t)/2 + exp(t)*cos(t)/2`.
#[test]
fn sum_factor_times_an_exponential_is_distributed() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let f = p(&ctx, "(sin(t) + cos(t))*exp(t)");
    let big_f = f.integrate(&t);
    assert_eq!(big_f.to_string(), "sin(t)*exp(t)");
    let f = p(&ctx, "t*(sin(t) + cos(t))*exp(t)");
    let big_f = f.integrate(&t);
    assert_antiderivative_at(&ctx, &f, &big_f, &t, &[], &POINTS);
    let sympy = p(&ctx, "t*exp(t)*sin(t) - exp(t)*sin(t)/2 + exp(t)*cos(t)/2");
    assert_eq!((big_f - sympy).simplify().to_string(), "0");
}

/// `∫ (x² + 1)/(x³ + a·x) dx` stayed unevaluated while
/// `∫ (x² + 1)/(x·(x² + a)) dx` worked: the parametric rational
/// integrator split the denominator only along its written factors, and
/// the logarithmic part of a parametric cubic was written only for a
/// numerator `c·e′`.  The denominator is now factored in `ℚ[x, a]`.
///
/// SymPy: `integrate((x**2 + 1)/(x**3 + a*x), x)` →
/// `(a - 1)*log(a + x**2)/(2*a) + log(x)/a`; at `a = 6/5` over `[1, 2]`
/// (`integrate(…, (x, 1, 2)).subs(a, 6/5).evalf(30)`) →
/// `0.649306089235213716298424766637`.
#[test]
fn parametric_denominator_with_a_monomial_content_is_split() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let big_f = assert_integrates(&ctx, "(x^2 + 1)/(x^3 + a*x)");
    let at = big_f.subs(&ctx.symbol("a"), &p(&ctx, "6/5"));
    assert_difference(&at, &x, "1", "2", "0.649306089235213716298424766637", &ctx);
    // Other factorisations in ℚ[x, a] (SymPy integrates all three).
    for s in ["1/(x^3 + a*x)", "1/(x^3 + a*x^2 + x + a)", "x/(x^3 + a^3)"] {
        assert_integrates(&ctx, s);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Found by the normal-form hunter
// ═══════════════════════════════════════════════════════════════════════════

/// Trigonometric powers were integrated only for the argument `x`:
/// `∫ cos(2x)² dx`, `∫ sin(x/3)³ dx`, `∫ cos(3x/2)² − sin(3x/2)² dx` stayed
/// unevaluated (the last is `cos 3x` written by the double-angle formula).
///
/// SymPy: `integrate(cos(2*x)**2, x)` → `x/2 + sin(2*x)*cos(2*x)/4`;
/// `integrate(sin(x/3)**3, x)` → `cos(x/3)**3 - 3*cos(x/3)`.
#[test]
fn trigonometric_powers_of_a_linear_argument() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let big_f = p(&ctx, "cos(2*x)^2").integrate(&x);
    let sympy = p(&ctx, "x/2 + sin(2*x)*cos(2*x)/4");
    assert_eq!((big_f - sympy).simplify().to_string(), "0");
    for s in [
        "sin(x/3)^3",
        "cos(3/2*x)^2 - sin(3/2*x)^2",
        "sin(2*x + 1)^2*cos(2*x + 1)^3",
        "cos(a*x)^2",
        "x*cos(3/2*x)^2",
        "cosh(2*x)^2",
    ] {
        assert_integrates(&ctx, s);
    }
}

/// A product of exponentials was left as it was written: `∫ eˣ⁻²·e⁻ˣ dx`
/// (the constant `e⁻²`) and `∫ 2x·e^{x/2 + √3}·e^{x/2} dx` stayed
/// unevaluated (the second came out through `u = eˣ` as `ln|eˣ|`-terms
/// when written `e^{x/2}²`); `∫ exp(x)^√37·cos 3x dx` too, while
/// `∫ e^{√37·x}·cos 3x dx` worked (for real `x`, `(eˣ)^c = e^{c·x}`).
///
/// SymPy: `integrate(exp(x - 2)*exp(-x), x)` → `x*exp(-2)`;
/// `integrate(exp(x)**sqrt(37)*cos(3*x), x)` →
/// `3*exp(x)**(sqrt(37))*sin(3*x)/46 + sqrt(37)*exp(x)**(sqrt(37))*cos(3*x)/46`.
#[test]
fn exponential_factors_are_collected() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(
        p(&ctx, "exp(x - 2)*exp(-x)").integrate(&x).to_string(),
        "x*exp(-2)"
    );
    for s in [
        "2*x*exp(x/2 + sqrt(3))*exp(x/2)",
        "exp(x)^sqrt(37)*cos(3*x)",
        "exp(sqrt(5)*x)*exp(-x/3 - 1)*x",
        "exp(a*x)*exp(x + b)*sin(x)",
    ] {
        assert_integrates(&ctx, s);
    }
    // `exp(a·x)^c` is not `exp(a·c·x)` for a complex `a`: kept apart.
    let f = p(&ctx, "exp(a*x)^(1/2)");
    let big_f = f.integrate(&x);
    if !big_f.has_unevaluated() {
        for binding in &BINDINGS {
            assert_antiderivative_at(&ctx, &f, &big_f, &x, binding, &POINTS);
        }
    }
}

/// A fraction with a common factor of numerator and denominator stayed
/// unevaluated: `∫ (x²·cos x + cos x)/(x² + 1) dx` (no stage cancels
/// `x² + 1`), and so did an integrable sum written over a common
/// denominator (`together`).
///
/// SymPy: `integrate((x**2*cos(x) + cos(x))/(x**2 + 1), x)` → `sin(x)`.
#[test]
fn fraction_with_a_common_factor_is_reduced() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(
        p(&ctx, "(x^2*cos(x) + cos(x))/(x^2 + 1)")
            .integrate(&x)
            .to_string(),
        "sin(x)"
    );
    for s in [
        "(x^2*exp(2*x)*sin(x) + exp(2*x)*sin(x))/(x^2 + 1)",
        "((x^2 + 1)*exp(x)*sin(x) + 2*x + 4)/(x^2 + 1)",
        "(-10*ln(a*x + 2)*(x + a)*(x + b) + 3)/((x + a)*(x + b))",
    ] {
        assert_integrates(&ctx, s);
    }
}

/// A trigonometric power times an exponential stayed unevaluated
/// (`∫ e^{x/2}·cos²(3x/2) dx`): by parts and the cyclic rule need single
/// factors.  The powers are reduced to multiple angles first.
///
/// SymPy: `integrate(exp(x/2)*cos(3*x/2)**2, x)` →
/// `36*exp(x/2)*sin(3*x/2)**2/37 + 12*exp(x/2)*sin(3*x/2)*cos(3*x/2)/37
/// + 38*exp(x/2)*cos(3*x/2)**2/37`.
#[test]
fn trigonometric_power_times_an_exponential() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let big_f = assert_integrates(&ctx, "exp(x/2)*cos(3*x/2)^2");
    let sympy = p(
        &ctx,
        "36*exp(x/2)*sin(3*x/2)^2/37 + 12*exp(x/2)*sin(3*x/2)*cos(3*x/2)/37 \
         + 38*exp(x/2)*cos(3*x/2)^2/37",
    );
    // Antiderivatives agree up to a constant: compare differences.
    let d = |e: &Ex| e.subs(&x, &p(&ctx, "2")) - e.subs(&x, &p(&ctx, "-1/3"));
    let diff: f64 = (d(&big_f) - d(&sympy))
        .eval_decimal(25)
        .unwrap()
        .parse()
        .unwrap();
    assert!(diff.abs() < 1e-20, "{big_f} vs SymPy: {diff}");
    for s in [
        "exp(x)*sin(x)^3",
        "x*sin(2*x)^2*cos(2*x)",
        "exp(-x)*sin(x)*cos(x)^2",
    ] {
        assert_integrates(&ctx, s);
    }
}

/// `∫ x/(x² + 2a·x + a²) dx` (and `∫ 4/(a²x² + 2a·x + 1) dx`) stayed
/// unevaluated while `∫ x/(x + a)² dx` worked: the parametric rational
/// integrator left a denominator with one linear factor to the `(a·x +
/// b)ⁿ` routes also when it was multiplied out, which those routes miss.
///
/// SymPy: `integrate(x/(x**2 + 2*a*x + a**2), x)` → `a/(a + x) + log(a + x)`.
#[test]
fn multiplied_out_power_of_a_linear_factor() {
    let ctx = Context::new();
    let big_f = assert_integrates(&ctx, "x/(x^2 + 2*a*x + a^2)");
    assert_eq!(big_f.to_string(), "a/(a + x) + ln(a + x)");
    assert_integrates(&ctx, "4/(a^2*x^2 + 2*a*x + 1)");
}

/// A deterministic sample of the hunter's pairs `(f, g)` with `g` an
/// equivalent form of `f`: before, `∫ f` evaluated and `∫ g` did not (or
/// the reverse).  Both must integrate now (checked at real and complex
/// parameter values).
#[test]
fn equivalent_forms_integrate_alike() {
    let ctx = Context::new();
    for (f, g) in [
        // exp_split / expand (seed 54, 83)
        (
            "exp((-sqrt(2)/2 - 1/2)*x - 1/3)*sin(-3*x - 1/3)",
            "exp(-sqrt(2)/2*x)*exp(-x/2)*exp(-1/3)*sin(-3*x - 1/3)",
        ),
        (
            "(3*sin(x) + 3*cos(x))*exp((1 + sqrt(5))/2*x + sqrt(2))",
            "3*sin(x)*exp(sqrt(2))*exp(x/2)*exp(sqrt(5)*x/2) + 3*cos(x)*exp(sqrt(2))*exp(x/2)*exp(sqrt(5)*x/2)",
        ),
        // exp_pow (seed 0)
        (
            "exp(sqrt(37)*x + b)*cos(3*x)",
            "exp(x)^sqrt(37)*exp(b)*cos(3*x)",
        ),
        // double_angle (seed 24, 161)
        (
            "exp(x/2)*cos(-3*x)",
            "exp(x/2)*(cos(-3/2*x)^2 - sin(-3/2*x)^2)",
        ),
        ("3*cos(3*x)", "3*(cos(3/2*x)^2 - sin(3/2*x)^2)"),
        // numden (seed 17)
        ("2*cos(2*x)", "(2*cos(2*x)*x^2 + 2*cos(2*x))/(x^2 + 1)"),
        // exp_merge (seed 76, a base gap)
        ("exp(x - 2)*exp(-x)", "exp(-2)"),
        // ratsimp (seed 201)
        ("x/(x + a)^2", "x/(a^2 + 2*a*x + x^2)"),
    ] {
        assert_integrates(&ctx, f);
        assert_integrates(&ctx, g);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Item 3: RootSum
// ═══════════════════════════════════════════════════════════════════════════

/// `∫ dx/(x³ + a)` stayed unevaluated: the logarithmic part of an
/// irreducible parametric factor of degree ≥ 3 was written only for a
/// numerator `c·e′`.  It is now the sum over the roots `r` of the factor
/// of `(residue at r)·ln(x − r)` — a `RootSum`, which `eval` writes out
/// by radicals where the roots have them (`x³ + a`), and keeps otherwise
/// (`x³ + a·x + b`).
///
/// SymPy: `integrate(1/(x**3 + a), x)` →
/// `RootSum(27*_t**3*a**2 - 1, Lambda(_t, _t*log(3*_t*a + x)))`.
#[test]
fn irreducible_parametric_cubic_gives_a_root_sum() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_integrates(&ctx, "1/(x^3 + a)");
    assert_integrates(&ctx, "x/(x^4 + a)");
    // Roots without radicals in this form: the RootSum stays.  Checked at
    // real parameter values (the RootSum's roots are found numerically).
    let f = p(&ctx, "1/(x^3 + a*x + b)");
    let big_f = f.integrate(&x);
    assert!(big_f.to_string().contains("RootSum"), "{big_f}");
    for binding in &BINDINGS[2..] {
        assert_antiderivative_at(&ctx, &f, &big_f, &x, binding, &POINTS);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Item 4: ratsimp of a vanishing denominator
// ═══════════════════════════════════════════════════════════════════════════

/// `ratsimp` returned its input when a denominator vanished identically,
/// while `together` gave `zoo`.  Now a negative power of a zero
/// polynomial is a fraction with denominator 0: `P/0` is `zoo`, `0/0` is
/// `nan` (undefined everywhere).
///
/// SymPy: `ratsimp(1/(x*(-x/(x + 1) + x*(-x/(x + 1) + 1))))` → `zoo`
/// (and `ratsimp` of the `0/0` below → `0`, as `cancel` returns a zero
/// numerator before it looks at the denominator).
#[test]
fn ratsimp_of_a_vanishing_denominator_is_zoo() {
    let ctx = Context::new();
    let e = p(&ctx, "1/(x*(-x/(x + 1) + x*(-x/(x + 1) + 1)))");
    assert_eq!(e.ratsimp().to_string(), "zoo");
    assert_eq!(e.together().to_string(), "zoo");
    let e = p(&ctx, "(x + 2)/(x*(x + 1) - x^2 - x) + 1");
    assert_eq!(e.ratsimp().to_string(), "zoo");
    let e = p(&ctx, "(x*(x + 1) - x^2 - x)/(x*(x + 2) - x^2 - 2*x)");
    assert_eq!(e.ratsimp().to_string(), "nan");
    // Unchanged: an ordinary fraction.
    assert_eq!(p(&ctx, "(x^2 - 1)/(x - 1)").ratsimp().to_string(), "x + 1");
}

// ═══════════════════════════════════════════════════════════════════════════
// Item 5: signs of products of non-negative factors
// ═══════════════════════════════════════════════════════════════════════════

/// For real `a`, `b` the assumption system proved `a² ≥ 0` but not
/// `4a² ≥ 0`, `a²b² ≥ 0` or `4a² + b² ≥ 0`: a product's sign was derived
/// only from strictly signed factors.
///
/// SymPy (`a, b = symbols('a b', real=True)`): `(4*a**2).is_nonnegative`,
/// `(4*a**2 + b**2).is_nonnegative`, `(a**2*b**2).is_nonnegative`,
/// `(-3*a**2).is_nonpositive`, `(4*a**2 + b**2 + 1).is_positive` → all
/// `True`; for an unassumed `c`, `(4*c**2).is_nonnegative` → `None`.
#[test]
fn products_of_nonnegative_factors_are_nonnegative() {
    let ctx = Context::new();
    ctx.symbol_with("a", &[Assumption::Real]).unwrap();
    ctx.symbol_with("b", &[Assumption::Real]).unwrap();
    ctx.symbol_with("q", &[Assumption::Positive]).unwrap();
    for s in [
        "4*a^2",
        "4*a^2 + b^2",
        "a^2*b^2",
        "3*a^2*q",
        "a^4 + 2*a^2*b^2",
    ] {
        assert_eq!(p(&ctx, s).is_nonnegative(), Some(true), "{s} ≥ 0");
    }
    for s in ["-3*a^2", "-a^2*q"] {
        assert_eq!(p(&ctx, s).is_nonpositive(), Some(true), "{s} ≤ 0");
    }
    assert_eq!(p(&ctx, "4*a^2 + b^2 + 1").is_positive(), Some(true));
    // Not decided: an unassumed symbol, a factor of unknown sign.
    assert_eq!(p(&ctx, "4*c^2").is_nonnegative(), None);
    assert_eq!(p(&ctx, "a^2*b").is_nonnegative(), None);
    // Unchanged: strict signs.
    assert_eq!(p(&ctx, "3*q^2").is_positive(), Some(true));
}
