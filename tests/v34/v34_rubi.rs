//! Families of the Rubi suite that stayed unevaluated up to 0.36 and now
//! integrate: radicals of a Möbius function (`t = B^{1/q}`, and the pair
//! `t = B₁^{1/q}·B₂^{−1/q}` of Chebyshev's binomial integrals), the
//! binomial `xᵐ·(a + b·xⁿ)ᵖ` through `w = xⁿ`, `e^{k·atanh u}` rewritten
//! as `(1 + u)^{k/2}·(1 − u)^{−k/2}`, and rational functions of
//! exponentials with symbolic rates (`sinh(c + d·x)`).  Every closed form
//! is checked by differentiation at sample points, and — since a
//! derivative check cannot see a jump — `F(b) − F(a)` is compared with
//! mpmath `quad` (`mp.dps = 30`) on an interval where the integrand is
//! real and continuous.  SymPy 1.14 is cited where it has a closed form.

use symplex::prelude::*;

/// `F(t)`, for a closed form `F` in `x` alone.  A principal logarithm of a
/// negative number adds a constant `iπ·c`, which cancels in `F(b) − F(a)`
/// on an interval where it does not jump.
fn value_at(ctx: &Context, big_f: &Ex, t: f64) -> Complex64 {
    let x = ctx.symbol("x");
    big_f
        .subs(&x, &ctx.parse(&format!("{t}")).unwrap())
        .eval_complex64()
        .unwrap()
}

/// `∫ src` has a closed form whose derivative is the integrand at the
/// `points` and whose increment over `[a, b]` is `want`.
fn check(ctx: &Context, src: &str, points: &[f64], a: f64, b: f64, want: f64) {
    let x = ctx.symbol("x");
    let f = ctx.parse(src).unwrap();
    let big_f = f.integrate(&x);
    assert!(!big_f.has_unevaluated(), "∫ {src} = {big_f}");
    let d = big_f.diff(&x);
    for &t in points {
        let tt = ctx.parse(&format!("{t}")).unwrap();
        let lhs = d.subs(&x, &tt).eval_complex64().unwrap();
        let rhs = f.subs(&x, &tt).eval_complex64().unwrap();
        assert!(
            (lhs - rhs).norm() < 1e-9 * rhs.norm().max(1.0),
            "∫ {src}: F′({t}) = {lhs} but f({t}) = {rhs}; F = {big_f}"
        );
    }
    let got = value_at(ctx, &big_f, b) - value_at(ctx, &big_f, a);
    assert!(
        (got.re - want).abs() < 1e-9 * want.abs().max(1.0) && got.im.abs() < 1e-9,
        "∫ {src} over [{a}, {b}]: F(b) − F(a) = {got}, quad = {want}; F = {big_f}"
    );
}

/// One linear radical, `t = √(2 + 3x)` (Rubi 1.1.1.2), and a pair with
/// exponents summing to an integer, `t = √(1 + x)/√(1 − x)` and
/// `t = √x/√(2 − x)` (Rubi 1.1.1.2, 1.1.1.3).  mpmath
/// `quad(lambda t: (2+3*t)**2.5/t, [0.5, 2])` = 97.288763189907242734;
/// `quad(lambda t: sqrt(1+t)/sqrt(1-t), [-0.9, 0, 0.9])` =
/// 2.2395390299972684753; `quad(lambda t: t**2.5*(2-t)**2.5, [0.2, 1.8])`
/// = 0.97229551125100764786.  SymPy 1.14 integrates the first as a
/// `Piecewise` with `acoth(√6·√(x + 2/3)/2)` for `|x + 2/3| > 2/3`, the
/// second as a `Piecewise` of an `asin` form and, for `|x + 1| > 2`, an
/// `acosh` form; symplex keeps `√(1 + x)/√(1 − x)` in one closed form,
/// which is an antiderivative where the integrand is imaginary too
/// (checked at `x = 3/2`).
#[test]
fn radicals_of_linear_functions_are_rationalised() {
    let ctx = Context::new();
    check(
        &ctx,
        "(2+3*x)^(5/2)/x",
        &[0.7, 1.9, -0.5],
        0.5,
        2.0,
        97.288_763_189_907_24,
    );
    check(
        &ctx,
        "sqrt(1+x)/sqrt(1-x)",
        &[-0.6, 0.3, 1.5],
        -0.9,
        0.9,
        2.239_539_029_997_268_5,
    );
    check(
        &ctx,
        "x^(5/2)*(2-x)^(5/2)",
        &[0.4, 1.3],
        0.2,
        1.8,
        0.972_295_511_251_007_6,
    );
}

/// `e^{k·atanh u} = (1 + u)^{k/2}·(1 − u)^{−k/2}` (Rubi 7.3.6), and
/// `e^{acoth u} = e^{atanh(1/u)}` (7.4.2; the harness's translation of
/// `acoth`).  mpmath: `quad(lambda t: exp(3*atanh(t/2)), [-1.9, 0, 1.9])`
/// = 33.640127986705963842; `quad(lambda t: exp(2*atanh(t)), [-0.9,
/// 0.5])` = 1.2700021334646801720; `quad(lambda t: exp(atanh(1/(2*t))),
/// [1, 3])` = 2.6724799044471985226 and over `[-3, -1]`
/// 1.5115490710835402264.  SymPy 1.14 leaves `integrate(exp(3*atanh(x/2)),
/// x)` unevaluated.
#[test]
fn exponentials_of_atanh_are_algebraic() {
    let ctx = Context::new();
    check(
        &ctx,
        "exp(3*atanh(x/2))",
        &[-1.2, 0.4, 1.7],
        -1.9,
        1.9,
        33.640_127_986_705_96,
    );
    check(
        &ctx,
        "exp(2*atanh(x))",
        &[-0.5, 0.2],
        -0.9,
        0.5,
        1.270_002_133_464_680_2,
    );
    check(
        &ctx,
        "exp(atanh(1/(2*x)))",
        &[1.5, 2.5],
        1.0,
        3.0,
        2.672_479_904_447_198_5,
    );
    check(
        &ctx,
        "exp(atanh(1/(2*x)))",
        &[-1.5, -2.5],
        -3.0,
        -1.0,
        1.511_549_071_083_540_2,
    );
}

/// `xᵐ·(a + b·xⁿ)ᵖ` through `w = xⁿ`: `(m + 1)/n` an integer
/// (`x¹¹·√(1 + x⁴)`, `x³·∛(1 + x²)`, `1/(x·√(1 + x³))`) and `(m + 1)/n + p`
/// an integer (`√(2 + 3x²)`, `1/(x²·√(3 + 2x²))`; Rubi 1.1.3.2, 1.1.2.2).
/// The second case maps `x = 0` to `t = 0`, so the answer is continuous
/// across it where the integrand is (`√(2 + 3x²)` over `[−2, 1.5]`).
/// mpmath: `quad(lambda t: t**11*sqrt(1+t**4), [-1.5, 0, 1.2])` =
/// −22.354899523705549689; `quad(lambda t: t**3*cbrt(1+t**2), [-1, 0, 2])`
/// = 5.8193889875461906060; `quad(lambda t: 1/(t*sqrt(1+t**3)), [0.5, 2])`
/// = 0.94411572250607559717; `quad(lambda t: sqrt(2+3*t**2), [-2, 0,
/// 1.5])` = 7.6902940064347113397; `quad(lambda t:
/// 1/(t**2*sqrt(3+2*t**2)), [0.3, 2])` = 1.4286241460659898591.  SymPy
/// 1.14: `x**12*sqrt(x**4 + 1)/14 + x**8*sqrt(x**4 + 1)/70 − …` for the
/// first, `x*sqrt(3*x**2 + 2)/2 + sqrt(3)*asinh(sqrt(6)*x/2)/3` for the
/// fourth.
#[test]
fn binomials_go_through_a_power_of_the_variable() {
    let ctx = Context::new();
    check(
        &ctx,
        "x^11*sqrt(1+x^4)",
        &[-1.1, 0.6],
        -1.5,
        1.2,
        -22.354_899_523_705_55,
    );
    check(
        &ctx,
        "x^3*(1+x^2)^(1/3)",
        &[-0.7, 1.3],
        -1.0,
        2.0,
        5.819_388_987_546_191,
    );
    check(
        &ctx,
        "1/(x*sqrt(1+x^3))",
        &[0.7, 1.6],
        0.5,
        2.0,
        0.944_115_722_506_075_6,
    );
    check(
        &ctx,
        "sqrt(2+3*x^2)",
        &[-1.3, 0.2, 1.1],
        -2.0,
        1.5,
        7.690_294_006_434_711,
    );
    check(
        &ctx,
        "1/(x^2*sqrt(3+2*x^2))",
        &[0.5, 1.7, -0.8],
        0.3,
        2.0,
        1.428_624_146_065_99,
    );
}

/// Hyperbolic functions of `c + d·x` through `u = e^{c + d·x}` (Rubi
/// 6.1.7, 6.7.1): the existing substitution took numeric rates without a
/// shift only.  mpmath: `quad(lambda t: sech(1+2*t)**2*sinh(1+2*t)**3,
/// [-2, 0.7])` = −2.2150439165750275828; `quad(lambda t:
/// cosh(t)**2/(2+sinh(t)**2), [-2, 0, 2])` = 2.8230520972291530317.
/// SymPy 1.14 leaves the first unevaluated
/// (`Integral(sinh(2*x + 1)**3*sech(2*x + 1)**2, x)`).
#[test]
fn hyperbolic_functions_of_a_shifted_argument() {
    let ctx = Context::new();
    check(
        &ctx,
        "sech(1+2*x)^2*sinh(1+2*x)^3",
        &[-1.1, 0.3],
        -2.0,
        0.7,
        -2.215_043_916_575_027_6,
    );
    check(
        &ctx,
        "cosh(x)^2/(2+sinh(x)^2)",
        &[-1.4, 0.5],
        -2.0,
        2.0,
        2.823_052_097_229_153,
    );
}

/// With parameters: the answers are checked by differentiation at the
/// harness's generic values (`a = 6/5`, `b = 3/4`, `c = 5/3`, `d = 2/7`,
/// `A = 19/10`, `B = 23/12`, `x ∈ {1/3, 7/5}`).  The first needs the
/// rational integrator over `ℚ(a, b, A, B)` with the denominator
/// `(t² − a)⁸` of degree 16 (refused above degree 10 up to 0.36; now its
/// square-free decomposition is read off the factors as written); the
/// last two are Rubi 1.1.1.2 and 6.7.1 entries (`coth(a + b·x)·coth(c +
/// b·x)`: two exponentials whose quotient is the constant `e^{a − c}`).
#[test]
fn parametric_families_check_at_generic_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let values = [
        ("a", 6, 5),
        ("b", 3, 4),
        ("c", 5, 3),
        ("d", 2, 7),
        ("A", 19, 10),
        ("B", 23, 12),
    ];
    for src in [
        "(a+b*x)^(5/2)*(A+B*x)/x^8",
        "(a+b*x)^(5/2)*(c+d*x)^(1/2)",
        "coth(a+b*x)*coth(c+b*x)",
        "x^(-7/2)/((a+b*x^2)^2)",
    ] {
        let f = ctx.parse(src).unwrap();
        let big_f = f.integrate(&x);
        assert!(!big_f.has_unevaluated(), "∫ {src} = {big_f}");
        let residual = big_f.diff(&x) - f.clone();
        for (p, q) in [(1, 3), (7, 5)] {
            let mut r = residual.subs(&x, &ctx.parse(&format!("{p}/{q}")).unwrap());
            for (name, n, m) in values {
                r = r.subs(&ctx.symbol(name), &ctx.parse(&format!("{n}/{m}")).unwrap());
            }
            let v = r.eval_complex64().unwrap();
            assert!(v.norm() < 1e-9, "∫ {src}: F′ − f = {v} at x = {p}/{q}");
        }
    }
}
