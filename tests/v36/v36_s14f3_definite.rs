//! Tests of session 14 (s14f3): definite integrals and quadrature hangs.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

/// `D(x) = x⁴ − 3x³ − 2x² + 3x + 2`: irreducible over ℚ, real roots
/// `1.148837482254663…` and `3.274009647488298…` (mpmath `polyroots`), the
/// others `−0.71142… ± 0.16002…i`.
const QUARTIC: &str = "x^4 - 3*x^3 - 2*x^2 + 3*x + 2";

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

fn is_divergent<T: std::fmt::Debug>(r: &Result<T, SymplexError>) -> bool {
    matches!(r, Err(SymplexError::Divergent { .. }))
}

/// `∫₁² dx/D(x)` diverges: `D(1) = 1`, `D(2) = −8`, so `D` has a simple
/// root `c = 1.1488…` inside and the numerator 1 does not vanish there
/// (`1/D ~ 1/(D′(c)(x − c))`).  Up to 0.51 every route ran for more than a
/// minute (the root in Ferrari's radicals was compared with another radical
/// form of itself, and its pole order proved through minimal polynomials of
/// degree in the hundreds): `try_integrate_definite`, `integrate_numeric`
/// and `eval_decimal` of the formal integral.
#[test]
fn pole_at_an_irrational_quartic_root_diverges_at_once() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = parse(&ctx, &format!("1/({QUARTIC})"));
    let (one, two) = (ctx.int(1), ctx.int(2));
    assert!(is_divergent(&f.try_integrate_definite(&x, &one, &two)));
    assert!(is_divergent(&f.integrate_numeric(&x, &one, &two)));
    let node = f.definite_integral_node(&x, &one, &two);
    assert!(is_divergent(&node.eval_decimal(15)));
}

/// The same integrand between its real roots converges and keeps its value:
/// `∫₂³ dx/D(x)` = −0.104117994052790172843486195854918… (mpmath `quad`
/// at 40 and 60 digits agree).
#[test]
fn quartic_without_a_root_in_the_range_converges() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = parse(&ctx, &format!("1/({QUARTIC})"));
    let (two, three) = (ctx.int(2), ctx.int(3));
    let v = f.integrate_numeric(&x, &two, &three).unwrap();
    assert!((v + 0.104_117_994_052_790_17).abs() < 1e-15, "{v}");
    let node = f.definite_integral_node(&x, &two, &three);
    assert_eq!(node.eval_decimal(15).unwrap(), "-0.10411799405279");
    let closed = f.try_integrate_definite(&x, &two, &three).unwrap();
    let s = closed.eval_decimal(20).unwrap();
    assert!(s.starts_with("-0.104117994052790172"), "{s}");
}

/// Other factors that stay away from 0 at the pole do not save the
/// integral (the sweep's integrand families, whose numeric checks hung for
/// more than 25 s each in 67 cases): `2·cosh(−x − 1)/D(x)` on `[1, 2]`
/// (cosh ≥ 1), and `(−x² + 2x − 2)·ln(−x)/(x⁴ − x² + 2x + 1)` on
/// `[−2, −1/2]`, whose denominator has the root `−1.386470913968351…`
/// (mpmath `polyroots`) where `−x² + 2x − 2 ≈ −6.69` and `ln 1.3865 ≈
/// 0.327`.
#[test]
fn pole_with_nonvanishing_cofactor_diverges() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = parse(&ctx, &format!("2*cosh(-x - 1)/({QUARTIC})"));
    assert!(is_divergent(&f.integrate_numeric(
        &x,
        &ctx.int(1),
        &ctx.int(2)
    )));
    let g = parse(&ctx, "(-x^2 + 2*x - 2)*ln(-x)/(x^4 - x^2 + 2*x + 1)");
    let (lo, hi) = (ctx.int(-2), ctx.rational(-1, 2));
    assert!(is_divergent(&g.integrate_numeric(&x, &lo, &hi)));
    assert!(is_divergent(&g.try_integrate_definite(&x, &lo, &hi)));
}

/// A zero of the denominator that the numerator shares is no proof of
/// divergence: `sin(x² − 2)/(x² − 2)` is continuous at `√2` (removable), and
/// `∫₁² sin(x² − 2)/(x² − 2) dx` = 0.869378735711414729257142350602711…
/// (mpmath `quad` split at `√2`, 40 and 60 digits).
#[test]
fn removable_singularity_is_not_a_pole() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = parse(&ctx, "sin(x^2 - 2)/(x^2 - 2)");
    let v = f.integrate_numeric(&x, &ctx.int(1), &ctx.int(2)).unwrap();
    assert!((v - 0.869_378_735_711_414_7).abs() < 1e-14, "{v}");
    let g = parse(&ctx, "(x^2 - 2)/(x^2 - 2)^2");
    assert!(is_divergent(&g.try_integrate_definite(
        &x,
        &ctx.int(1),
        &ctx.int(2)
    )));
}

/// `∫_{−∞}^{∞} −3e^{−3x}/(1 + x + x³) dx` diverges at the real root
/// `−0.682327803828019…` of the cubic (mpmath `polyroots`; the numerator
/// does not vanish), and `∫₁^∞ −e^{x/2}/(1 − x + x³) dx` at `+∞` (the
/// integrand grows like `−e^{x/2}/x³`; the cubic's real root `−1.3247…`
/// is outside).  Both were found divergent, but after 54 s and 25 s (the
/// limits of an antiderivative made of `Ei` at the cubic's roots in
/// radicals).
#[test]
fn exponential_over_cubic_on_infinite_ranges_diverges_at_once() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = parse(&ctx, "-3*exp(-3*x)/(1 + x + x^3)");
    assert!(is_divergent(&f.try_integrate_definite(
        &x,
        &ctx.neg_infinity(),
        &ctx.infinity()
    )));
    let g = parse(&ctx, "-exp(x/2)/(1 - x + x^3)");
    let r = g.try_integrate_definite(&x, &ctx.int(1), &ctx.infinity());
    assert!(is_divergent(&r), "{r:?}");
}

/// `∫ |sin(x/3 + 1/2)| dx` over `[10¹⁰⁰ + 7/10, 10¹⁰⁰ + 100]` =
/// 62.8321240307758659813681402925… (mpmath at 250 digits, from the
/// antiderivative `2⌊u/π⌋ + 1 − cos(u − π⌊u/π⌋)` of `|sin u|` and by `quad`
/// between the zeros).  Up to 0.51 `eval_f64`, `eval_decimal` and
/// `integrate_numeric` refused it (in 16 s at 0.50: the quadrature's nodes
/// need 330 bits before the first fractional one of `x/3`) while
/// `try_integrate_definite` had the closed form: the formal integral now
/// falls back to it.
#[test]
fn periodic_kink_far_out_uses_the_closed_form() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = parse(&ctx, "abs(sin(x/3 + 1/2))");
    let lo = parse(&ctx, "10^100 + 7/10");
    let hi = parse(&ctx, "10^100 + 100");
    let node = f.definite_integral_node(&x, &lo, &hi);
    let v = node.eval_f64().unwrap();
    assert!((v - 62.832_124_030_775_866).abs() < 1e-13, "{v}");
    assert_eq!(node.eval_decimal(25).unwrap(), "62.83212403077586598136814");
    let n = f.integrate_numeric(&x, &lo, &hi).unwrap();
    assert!((n - 62.832_124_030_775_866).abs() < 1e-13, "{n}");
}

/// A guard found while trying to make the quadrature of tails below the
/// exponent range succeed: `∫₀^∞ e^{x − 10¹⁰⁰⁰} dx` diverges (the
/// integrand grows without bound beyond `x = 10¹⁰⁰⁰`, far past the samples
/// `|x| ≤ 2²⁰⁴⁸` of the `f64` tail check).  A tighter bound of `exp` below
/// the range made `integrate_numeric` answer `Ok(0.0)` in a draft.
#[test]
fn integrand_that_grows_beyond_the_tail_samples_is_not_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = parse(&ctx, "exp(x - 10^1000)");
    let (zero, inf) = (ctx.int(0), ctx.infinity());
    assert!(f.integrate_numeric(&x, &zero, &inf).is_err());
    assert!(
        f.definite_integral_node(&x, &zero, &inf)
            .eval_f64()
            .is_err()
    );
}
