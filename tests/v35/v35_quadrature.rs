//! Arbitrary-precision quadrature of unevaluated definite integrals
//! (`evalf` of a `DefiniteIntegral` node by tanh-sinh over `BigFloat`, after
//! mpmath's `quad`).  Before, the node was evaluated by `f64` Gauss–Kronrod
//! quadrature only: more than 16 digits were refused (`NotImplemented`), an
//! endpoint singularity left 8 or 9 digits with a bound its error exceeded,
//! `∫₀¹ x^(−9/10) dx` did not converge, and an integral equal to 0 was
//! refused.  Every reference value cites the mpmath call that produced it;
//! where mpmath's own `quad` loses digits at an endpoint singularity (it gets
//! 28 of the 50 digits of `∫₀¹ dx/√(1 − x)`), the reference is a closed form
//! or `quad` after the substitution that removes the singularity.

use symplex::definite::QuadRule;
use symplex::prelude::*;

fn node(ctx: &Context, body: &str, lo: &str, hi: &str) -> Ex {
    let x = ctx.symbol("x");
    let f = ctx.parse(body).unwrap();
    f.definite_integral_node(&x, &ctx.parse(lo).unwrap(), &ctx.parse(hi).unwrap())
}

/// Before: `eval_decimal(30)` was `NotImplemented` (at most 16 digits from
/// `f64` quadrature).  mpmath: `mp.dps = 60; quad(lambda x: x**x, [0, 1])`
/// = 0.783430510712134407059264386526975469407681990146930958255418.
#[test]
fn x_to_the_x_to_thirty_digits() {
    let ctx = Context::new();
    let n = node(&ctx, "x^x", "0", "1");
    assert_eq!(
        n.eval_decimal(30).unwrap(),
        "0.783430510712134407059264386527"
    );
    assert_eq!(n.eval_decimal(16).unwrap(), "0.7834305107121344");
}

/// Before: `eval_f64` was 1.9999999898808887 (the `f64` endpoint-smoothing
/// estimate, with a bound of about `2·10⁻⁹` its error exceeded), and
/// `eval_decimal(9)` was refused.  Exact value 2 (`∫₀¹ (1 − x)^(−1/2) dx`);
/// mpmath's `quad` at 50 digits gives 1.99999999999999999999999999987…
#[test]
fn inverse_square_root_endpoint_singularity() {
    let ctx = Context::new();
    let n = node(&ctx, "1/sqrt(1 - x)", "0", "1");
    assert_eq!(n.eval_f64().unwrap(), 2.0);
    assert_eq!(n.eval_decimal(9).unwrap(), "2");
    assert_eq!(n.eval_decimal(30).unwrap(), "2");
}

/// Before: refused beyond 16 digits.  Exact value −4 (`∫₀¹ ln x/√x dx`, by
/// `x = u²`: mpmath `quad(lambda u: 2*log(u*u), [0, 1])` = −4.0).
#[test]
fn logarithmic_endpoint_singularity() {
    let ctx = Context::new();
    let n = node(&ctx, "log(x)/sqrt(x)", "0", "1");
    assert_eq!(n.eval_decimal(30).unwrap(), "-4");
}

/// Before: refused beyond 16 digits.  `∫₀^∞ e^(−x²) dx = √π/2`, mpmath
/// `sqrt(pi)/2` = 0.886226925452758013649083741671 (30 digits).
#[test]
fn gaussian_on_a_half_line() {
    let ctx = Context::new();
    let n = node(&ctx, "exp(-x^2)", "0", "oo");
    assert_eq!(
        n.eval_decimal(30).unwrap(),
        "0.886226925452758013649083741671"
    );
}

/// Before: "quadrature of the definite integral did not converge"
/// (`f64` estimate 9.9997 ± 2·10⁻³) at every precision.  Exact value 10;
/// mpmath's `quad(lambda x: x**(-0.9), [0, 1])` at 50 digits gives
/// 9.99996276…, six digits.
#[test]
fn strong_algebraic_endpoint_singularity() {
    let ctx = Context::new();
    let n = node(&ctx, "x^(-9/10)", "0", "1");
    assert_eq!(n.eval_f64().unwrap(), 10.0);
    assert_eq!(n.eval_decimal(30).unwrap(), "10");
}

/// The Beta(5, 1/2) moment `∫₀¹ x⁴e^{x/10}/(√(1 − x)·B(5, ½)) dx = ₁F₁(5;
/// 11/2; 1/10)`.  Before: `eval_f64` was 1.0952385480452327 (wrong in the
/// 9th digit while its bound, below `10⁻⁹`, claimed 9), and digits beyond
/// were refused.  mpmath: `mp.dps = 60; hyp1f1(5, mpf(11)/2, mpf(1)/10)` =
/// 1.09523856180598760095911968287970195671797937015553549…
#[test]
fn beta_moment_with_an_endpoint_singularity() {
    let ctx = Context::new();
    let n = node(&ctx, "x^4*exp(x/10)/(sqrt(1 - x)*beta(5, 1/2))", "0", "1");
    let v = n.eval_f64().unwrap();
    assert!((v - 1.095_238_561_805_987_6).abs() < 4e-16, "{v}");
    assert_eq!(
        n.eval_decimal(30).unwrap(),
        "1.09523856180598760095911968288"
    );
}

/// An integral equal to 0 is 0 to the requested digits (its error ball
/// contains 0 and lies within `10⁻ᵈ`, as `evalf` prints a cancellation that
/// is zero to the precision).  Before: `eval_f64` was `PrecisionExhausted`
/// (while `integrate_numeric` gave 7·10⁻¹⁶).  mpmath: `mp.dps = 60;
/// quad(lambda t: -(-2*sin(t) + 1)*cos(2*t + 4*cos(t)), [0, pi, 2*pi])` =
/// −1.5·10⁻⁶¹.
#[test]
fn an_integral_equal_to_zero_is_zero_to_the_digits() {
    let ctx = Context::new();
    let n = node(&ctx, "-(-2*sin(x) + 1)*cos(2*x + 4*cos(x))", "0", "2*pi");
    assert_eq!(n.eval_f64().unwrap(), 0.0);
    assert_eq!(n.eval_decimal(30).unwrap(), "0");
}

/// A kink inside the interval is split off by the breakpoint scanner.
/// mpmath: `mp.dps = 60; quad(lambda x: abs(x - mpf(1)/3)*exp(x), [0,
/// mpf(1)/3, 1])` = 0.551797574019163978802821482087619509276730665832160…
#[test]
fn interior_kink_is_split_off() {
    let ctx = Context::new();
    let n = node(&ctx, "abs(x - 1/3)*exp(x)", "0", "1");
    assert_eq!(
        n.eval_decimal(40).unwrap(),
        "0.5517975740191639788028214820876195092767"
    );
}

/// The whole real line is split at 0 and each half must converge.
/// `∫ dx/(1 + x²) = π`, mpmath `pi` = 3.14159265358979323846264338328.
#[test]
fn whole_real_line() {
    let ctx = Context::new();
    let n = node(&ctx, "1/(1 + x^2)", "-oo", "oo");
    assert_eq!(
        n.eval_decimal(30).unwrap(),
        "3.14159265358979323846264338328"
    );
    // Reversed bounds negate.
    let r = node(&ctx, "1/(1 + x^2)", "oo", "-oo");
    assert_eq!(r.eval_decimal(20).unwrap(), "-3.1415926535897932385");
}

/// A divergent integral and an oscillatory one on an infinite range are
/// refused, not guessed (`∫₀^∞ sin x/x dx = π/2` needs `quadosc`; mpmath
/// `quadosc(lambda x: sin(x)/x, [0, inf], omega=1)` = 1.5707963267948966).
/// Since 0.52 a refused quadrature gives way to the closed form of the
/// definite integrator, which has `π/2` (up to 0.51 refused); an
/// oscillatory integrand without one stays refused.
#[test]
fn divergent_and_infinite_oscillatory_integrals_are_refused() {
    let ctx = Context::new();
    assert!(node(&ctx, "1/x", "-1", "1").eval_decimal(20).is_err());
    assert!(node(&ctx, "1/x^2", "-1", "1").eval_f64().is_err());
    assert_eq!(
        node(&ctx, "sin(x)/x", "0", "oo").eval_decimal(20).unwrap(),
        "1.5707963267948966192"
    );
    assert!(
        node(&ctx, "sin(x)/(x + sqrt(x + 1))", "0", "oo")
            .eval_decimal(20)
            .is_err()
    );
}

/// `DiracDelta` evaluates to 0 at every point a quadrature samples, so its
/// integral has no quadrature value: only its closed form has one (the
/// exact route of 0.52; `∫₋₁¹ δ(x)·eˣ dx = e⁰ = 1`, sifting).  Before
/// 0.43 `eval_f64` (and `integrate_numeric`) of `∫₋₁¹ δ(x) dx` was `0.0`
/// (truly 1); from 0.43 to 0.51 they were refused.
#[test]
fn dirac_delta_integrand_has_only_its_closed_form() {
    let ctx = Context::new();
    let n = node(&ctx, "diracdelta(x)*exp(x)", "-1", "1");
    assert_eq!(n.eval_f64().unwrap(), 1.0);
    assert_eq!(n.eval_decimal(20).unwrap(), "1");
    let x = ctx.symbol("x");
    let d = ctx.parse("diracdelta(x)").unwrap();
    assert_eq!(
        d.integrate_numeric(&x, &ctx.int(-1), &ctx.int(1)).unwrap(),
        1.0
    );
    // Never the quadrature's 0.
    let g = node(&ctx, "diracdelta(x^3 - x - 1/7)*gamma(x)", "1/2", "2");
    let v = g.eval_f64();
    assert!(!v.as_ref().is_ok_and(|v| *v == 0.0), "{v:?}");
}

/// An integrand that is not real on the interval: `∫₀¹ √(x − 2) dx =
/// (2/3)(2√2 − 1)·i`, mpmath `quad(lambda x: sqrt(x - 2), [0, 1])` =
/// 1.21895141649746006506891829895j.
#[test]
fn complex_valued_integrand() {
    let ctx = Context::new();
    let n = node(&ctx, "sqrt(x - 2)", "0", "1");
    let z = n.eval_complex64().unwrap();
    assert!(
        z.re.abs() < 1e-30 && (z.im - 1.218_951_416_497_460_1).abs() < 1e-15,
        "{z}"
    );
    let s = n.eval_decimal(25).unwrap();
    assert!(s.starts_with("1.218951416497460065068918"), "{s}");
}

/// A special-function integrand without an `f64` runtime is evaluated by
/// `evalf`'s own evaluator (before, `eval_f64` refused it: "cannot be
/// compiled for quadrature").  `∫₀¹ Si(x) dx = Si(1) + cos 1 − 1`, mpmath
/// `mp.dps = 40; si(1) + cos(1) - 1` (and `quad(si, [0, 1])`) =
/// 0.4863853762353227323422899212661562615446.
#[test]
fn special_function_integrand() {
    let ctx = Context::new();
    let n = node(&ctx, "si(x)", "0", "1");
    assert_eq!(n.eval_decimal(25).unwrap(), "0.4863853762353227323422899");
    let v = n.eval_f64().unwrap();
    assert!((v - 0.486_385_376_235_322_7).abs() < 1e-15, "{v}");
}

/// The bounds of a node may themselves be expressions (`π/2`, `√2`), and an
/// interior point the caller supplies splits the range
/// (`integrate_numeric_decimal`).  `∫₀^{π/2} sin x dx = 1`; `∫₀^{√2} x dx =
/// 1`; `∫₀² |x − 1| dx = 1` with the kink at 1 supplied.
#[test]
fn expression_bounds_and_caller_points() {
    let ctx = Context::new();
    assert_eq!(
        node(&ctx, "sin(x)", "0", "pi/2").eval_decimal(30).unwrap(),
        "1"
    );
    assert_eq!(
        node(&ctx, "x", "0", "sqrt(2)").eval_decimal(30).unwrap(),
        "1"
    );
    let x = ctx.symbol("x");
    let f = (&x - 1).abs();
    let v = f
        .integrate_numeric_decimal(
            &x,
            &[&ctx.int(0), &ctx.int(1), &ctx.int(2)],
            30,
            QuadRule::TanhSinh,
        )
        .unwrap();
    assert_eq!(v, "1");
}

/// Gauss–Legendre is available as the alternative rule (mpmath's
/// `method='gauss-legendre'`): exact for this polynomial at its first
/// degrees; `∫₀¹ eˣ dx = e − 1`, mpmath `e - 1` =
/// 1.71828182845904523536028747135.
#[test]
fn gauss_legendre_rule() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let v = x
        .exp()
        .integrate_numeric_decimal(&x, &[&ctx.int(0), &ctx.int(1)], 30, QuadRule::GaussLegendre)
        .unwrap();
    assert_eq!(v, "1.71828182845904523536028747135");
    let p = (&x.powi(3) - &x).integrate_numeric_decimal(
        &x,
        &[&ctx.int(0), &ctx.int(2)],
        25,
        QuadRule::GaussLegendre,
    );
    assert_eq!(p.unwrap(), "2");
}
