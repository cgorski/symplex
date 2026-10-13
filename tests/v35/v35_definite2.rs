//! Definite integrals and integral transforms after the 0.46 hunt
//! (antiderivatives with `Ei`/`Si`/`Ci`/`Li₂` at complex points, improper
//! integrals by residues, transforms of rational functions).  Every test
//! names what was wrong before and the oracle of its reference value
//! (mpmath 1.3.0 `quad`/`quadosc`/`nsum` at `mp.dps = 30`, or SymPy 1.14).

use symplex::fourier_transform::FourierConvention;
use symplex::prelude::*;

fn value(e: &Ex) -> f64 {
    e.eval_f64().unwrap_or_else(|err| panic!("{e}: {err}"))
}

fn close(got: f64, want: f64) -> bool {
    (got - want).abs() <= 1e-12 * want.abs().max(1.0)
}

fn definite(ctx: &Context, f: &str, lo: &Ex, hi: &Ex) -> Ex {
    let x = ctx.symbol("x");
    let f = ctx.parse(f).unwrap();
    f.try_integrate_definite(&x, lo, hi)
        .unwrap_or_else(|e| panic!("{f}: {e}"))
}

/// `∫_{−∞}^{∞} cos 2x/(x² + 2) dx = π e^{−2√2}/√2` was refused (the
/// antiderivative has `Ci`/`Si` at `2x ± 2√2 i`, whose limits at `±∞` were
/// unknown, and the half-line table is not tried on the full line); it is
/// now `2πi·Res_{i√2}` by Jordan's lemma.  mpmath: `quad` on `[−1, 1]` +
/// `quadosc(…, omega=2)` on both tails = 0.131299956473613941960159142767.
#[test]
fn full_line_rational_cosine_by_residues() {
    let ctx = Context::new();
    let v = definite(
        &ctx,
        "cos(2*x)/(x^2 + 2)",
        &ctx.neg_infinity(),
        &ctx.infinity(),
    );
    assert!(close(value(&v), 0.131_299_956_473_613_94), "{v}");
    assert!(!v.to_string().contains('I'), "{v}");
}

/// `∫₀^∞ sin x/(x(x² + 1)) dx = π(1 − e⁻¹)/2` was refused; the odd
/// integrand is half the line, where the simple pole of `1/(x(x² + 1))` at
/// 0 (a zero of `sin`) contributes half its residue.  mpmath: `quad` on
/// `[0, 1]` + `quadosc(…, [1, inf], omega=1)` = 0.992932651899435760276275099983.
#[test]
fn sinc_over_quadratic_half_residue() {
    let ctx = Context::new();
    let v = definite(&ctx, "sin(x)/(x*(x^2 + 1))", &ctx.int(0), &ctx.infinity());
    assert!(close(value(&v), 0.992_932_651_899_435_8), "{v}");
}

/// `∫_{−∞}^{∞} cos x/(x⁴ + 1) dx` was refused; the poles `±√2/2 ± i√2/2`
/// are written out (Ferrari's roots are small), giving
/// `π e^{−1/√2}(cos(1/√2) + sin(1/√2))/√2`.  mpmath: `quad` + `quadosc`
/// = 1.54427600961813566056102325621.
#[test]
fn quartic_denominator_over_the_line() {
    let ctx = Context::new();
    let v = definite(
        &ctx,
        "cos(x)/(x^4 + 1)",
        &ctx.neg_infinity(),
        &ctx.infinity(),
    );
    assert!(close(value(&v), 1.544_276_009_618_135_7), "{v}");
}

/// A double pole (`∫_{−∞}^{∞} cos x/(x² + 1)² dx = π/e`, exact principal
/// part in `ℚ(i)`) and a denominator whose roots stay `RootOf`
/// (`x⁶ + x² + 1`, irreducible, no real root) were refused.  mpmath: `quad`
/// on `[−1, 1]` + `2·quadosc(…, [1, inf], omega=1)` =
/// 1.15572734979092171791009318331 and 1.36771309997303887632316786434.
#[test]
fn double_pole_and_rootof_poles() {
    let ctx = Context::new();
    let (lo, hi) = (ctx.neg_infinity(), ctx.infinity());
    let v = definite(&ctx, "cos(x)/(x^2 + 1)^2", &lo, &hi);
    assert_eq!(v.to_string(), "exp(-1)*pi");
    let v = definite(&ctx, "cos(x)/(x^6 + x^2 + 1)", &lo, &hi);
    assert!(v.to_string().contains("RootOf"), "{v}");
    assert!(close(value(&v), 1.367_713_099_973_039), "{v}");
}

/// `∫₀^∞ sin x/(x² + 1) dx = (e⁻¹ Ei(1) − e Ei(−1))/2` and `∫₀^∞ cos x/(x³ +
/// x + 1) dx` were refused: the antiderivatives have `Si`/`Ci` at the
/// complex poles, `Si(x + c) → π/2` and `Ci(x + c) → 0` were unknown to the
/// limits at `∞` (and the Gruntz engine).  mpmath: `quadosc(sin(x)/(x**2 +
/// 1), [0, inf], omega=1)` = 0.646761122779130071553278590644,
/// `quadosc(cos(x)/(x**3 + x + 1), [0, inf], omega=1)` =
/// 0.531117410128592558393483323085.
#[test]
fn sine_and_cosine_integrals_at_infinity() {
    let ctx = Context::new();
    let v = definite(&ctx, "sin(x)/(x^2 + 1)", &ctx.int(0), &ctx.infinity());
    assert!(close(value(&v), 0.646_761_122_779_130_1), "{v}");
    let v = definite(&ctx, "cos(x)/(x^3 + x + 1)", &ctx.int(0), &ctx.infinity());
    assert!(close(value(&v), 0.531_117_410_128_592_6), "{v}");
}

/// With a frequency declared positive: `∫₀^∞ sin(ax)/(x² + 1) dx` (limits
/// of `Ci(ax ± ai)` decided through the assumptions) and `∫_{−∞}^{∞}
/// cos(ax)/(x² + x + 1) dx = (2π/√3) e^{−a√3/2} cos(a/2)` were refused.
/// mpmath at `a = 2`: `quadosc(sin(2x)/(x² + 1), [0, inf], omega=2)` =
/// 0.515905663339147932870192732881; at `a = 3`: `quad` + `quadosc` =
/// 0.0190957602834805903035686270267.
#[test]
fn positive_frequency_parameter() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();
    let f = (&a * &x).sin() / (x.powi(2) + 1);
    let v = f
        .try_integrate_definite(&x, &ctx.int(0), &ctx.infinity())
        .unwrap();
    assert!(
        close(value(&v.subs(&a, &ctx.int(2))), 0.515_905_663_339_147_9),
        "{v}"
    );
    let f = (&a * &x).cos() / (x.powi(2) + &x + 1);
    let v = f
        .try_integrate_definite(&x, &ctx.neg_infinity(), &ctx.infinity())
        .unwrap();
    assert!(
        close(value(&v.subs(&a, &ctx.int(3))), 0.019_095_760_283_480_59),
        "{v}"
    );
}

/// `∫₀^∞ ln x/((x + 1)(x + 2)) dx = ln²2/2`, `∫₀^∞ ln x/(x⁴ + 1) dx =
/// −√2π²/16` and `∫₀^∞ ln x/(x² + x + 1) dx = 0` were refused (dilogarithms
/// with unknown limits at `∞`); the keyhole contour gives `−½ Σ Res[R(z)
/// ln²(−z)]`.  mpmath: `quad(log(x)/((x + 1)*(x + 2)), [0, 1, inf])` =
/// 0.240226506959100712333551263163 = `ln²2/2`, `quad(log(x)/(x**4 + 1), [0,
/// 1, inf])` = −0.872358024954859941769695117021.  (Up to 0.46 this cited
/// SymPy for `log(2)**2/2`; SymPy 1.14 returns the first and the third
/// integral unevaluated, with `x` positive and with `meijerg=True` too, and
/// takes 80 s for the second.)
#[test]
fn logarithm_over_rational_by_keyhole() {
    let ctx = Context::new();
    let (zero, inf) = (ctx.int(0), ctx.infinity());
    let v = definite(&ctx, "log(x)/((x + 1)*(x + 2))", &zero, &inf);
    assert_eq!(v.to_string(), "1/2*ln(2)^2");
    let v = definite(&ctx, "log(x)/(x^4 + 1)", &zero, &inf);
    assert!(close(value(&v), -0.872_358_024_954_859_9), "{v}");
    let v = definite(&ctx, "log(x)/(x^2 + x + 1)", &zero, &inf);
    assert_eq!(v.to_string(), "0");
}

/// `∫₁^∞ ln x/(x² + 4) dx` was refused; `x = 1/u` turns it into an
/// integral over `[0, 1]`, where the dilogarithms are continuous.  mpmath:
/// `quad(log(x)/(x**2 + 4), [1, inf])` = 0.7880077017231617111802895.
#[test]
fn logarithm_tail_by_inversion() {
    let ctx = Context::new();
    let v = definite(&ctx, "log(x)/(x^2 + 4)", &ctx.int(1), &ctx.infinity());
    assert!(close(value(&v), 0.788_007_701_723_161_7), "{v}");
}

/// `∫₀¹ x·sin(2x)/(2(x² − x − 1)) dx` was refused: its antiderivative has
/// `Ci(2x − √5 − 1)`, complex at both ends (`Ci` of a negative argument), and
/// the test for a negligible imaginary part looked for `I` in a string
/// printed with `i`.  mpmath: `quad(x/2*sin(2x)/(x**2 - x - 1), [0, 1])` =
/// −0.1877164852644586584075553.
#[test]
fn complex_endpoint_values_of_a_real_integral() {
    let ctx = Context::new();
    let v = definite(
        &ctx,
        "x*sin(2*x)/(2*(x^2 - x - 1))",
        &ctx.int(0),
        &ctx.int(1),
    );
    assert!(close(value(&v), -0.187_716_485_264_458_66), "{v}");
}

/// `∫₀^∞ e^{(−1+i)x}/(x + 1) dx`: the antiderivative `e^{1−i} Ei((−1 + i)(x +
/// 1))` needs `Ei(w) → iπ` along a ray into the upper left quadrant.
/// mpmath: `quad(exp((-1+1j)*x)/(x + 1), [0, 1, inf])` =
/// 0.410592543469122494771521946948 + 0.262728682711301740567209304802i.
#[test]
fn exponential_integral_along_a_complex_ray() {
    let ctx = Context::new();
    let v = definite(
        &ctx,
        "exp((-1 + I)*x)/(x + 1)",
        &ctx.int(0),
        &ctx.infinity(),
    );
    let z = v.eval_complex64().unwrap();
    assert!((z.re - 0.410_592_543_469_122_5).abs() < 1e-12, "{v}");
    assert!((z.im - 0.262_728_682_711_301_74).abs() < 1e-12, "{v}");
}

/// The Fourier transform refused every rational function without a table
/// entry, down to `1/(t² + 1)`; it is now `H(−ω)·U(ω) + H(ω)·L(ω)` from the
/// residues in the two half-planes (`L(|ω|)` for an even function), with
/// half residues at simple real poles (principal value).  The ordinary
/// frequency transform of `1/(t² + 1)` is `πe^{−2π|w|}` (mpmath
/// `quad(cos(2πwt)/(t² + 1))`; up to 0.46 this cited SymPy for it, but SymPy
/// 1.14 `fourier_transform(1/(t**2 + 1), t, w)` is `pi*exp(-2*pi*w)` for a
/// plain and for a real `w` alike, wrong for `w < 0`: 25.51 at `w = −1/3`
/// instead of 0.38687, see `v35_sympy_audit`); mpmath (`quad` + `quadosc`):
/// `∫ e^{−it}/(t² + t + 1) dt` = 1.33905116569145104967232157776 +
/// 0.731526985845469784831387754368i, `∫ sin t·e^{−it/2}/(t(t² + 1)) dt` =
/// 1.838364485266392670107542.
#[test]
fn fourier_transform_of_rational_functions() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let w = ctx.symbol("w");
    let f = (t.powi(2) + 1).powi(-1);
    let ft = f.fourier_transform(&t, &w).unwrap();
    assert_eq!(ft.to_string(), "exp(-abs(w))*pi");
    let ord = f
        .fourier_transform_with(&t, &w, FourierConvention::Ordinary)
        .unwrap();
    let at = value(&ord.subs(&w, &ctx.rational(1, 3)));
    assert!(close(
        at,
        std::f64::consts::PI * (-2.0 * std::f64::consts::PI / 3.0).exp()
    ));
    let g = (t.powi(2) + &t + 1).powi(-1);
    let z = g
        .fourier_transform(&t, &w)
        .unwrap()
        .subs(&w, &ctx.int(1))
        .eval_complex64()
        .unwrap();
    assert!(
        (z.re - 1.339_051_165_691_451).abs() < 1e-12
            && (z.im - 0.731_526_985_845_469_8).abs() < 1e-12
    );
    let h = t.sin() / (&t * (t.powi(2) + 1));
    let v = h.fourier_transform(&t, &w).unwrap();
    assert!(
        close(
            value(&v.subs(&w, &ctx.rational(1, 2))),
            1.838_364_485_266_392_7
        ),
        "{v}"
    );
}

/// The Mellin transform knew the binomial forms only; `1/((x + 1)(x + 2))`
/// and `1/(x² + x + 1)` were refused.  Now `π/sin(πs) Σ Res[(−z)^{s−1}R(z)]`
/// on `0 < Re s < 2`.  SymPy: `mellin_transform(1/((x + 1)*(x + 2)), x, s)`
/// = `((-2**s*pi/2 + pi)/sin(pi*s), (0, 2), True)`; mpmath (`x = eᵘ`):
/// `quad(exp(u/2)/(exp(2u) + exp(u) + 1), [-inf, inf])` =
/// 1.81379936423421785059407825764.
#[test]
fn mellin_transform_of_rational_functions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = ctx.symbol("s");
    let f = ((&x + 1) * (&x + 2)).powi(-1);
    let (m, cond) = f.mellin_transform(&x, &s).unwrap();
    let half = ctx.rational(1, 2);
    assert!(
        close(value(&m.subs(&s, &half)), 0.920_151_184_510_610_1),
        "{m}"
    );
    assert_eq!(cond.to_string(), "re(s) > 0 & 2 > re(s)");
    let g = (x.powi(2) + &x + 1).powi(-1);
    let (m, _) = g.mellin_transform(&x, &s).unwrap();
    assert!(
        close(value(&m.subs(&s, &half)), 1.813_799_364_234_217_9),
        "{m}"
    );
}

/// The Laplace transform refused rational functions with poles off 0:
/// `L{1/(t + 1)} = e^s E₁(s)` (mpmath `quad(exp(-t)/(t + 1), [0, inf])` =
/// 0.596347362323194074341078499369 = `e·E₁(1)`; up to 0.46 this cited SymPy
/// for `exp(s)*expint(1, s)`, but SymPy 1.14 gives `-exp(-s)*Ei(-s)`, which is
/// 0.0807 at `s = 1`: wrong, see `v35_sympy_audit`), `L{1/(t² + 1)}`.
/// For a pole with a positive real part (`t² − 2t + 2`, poles `1 ± i`) the
/// continuation `e^{−ρs}(±iπ − Ei(ρs))` keeps the result right at complex
/// `s`.  mpmath: `quad(exp(-2t)/(t**2 + 1), [0, 1, inf])` =
/// 0.399020988594183846892666510762, `quad(exp(-s t)/(t**2 - 2t + 2), [0,
/// …, inf])` at `s = 1/2 + 3i` = −0.10345245615833889771025 −
/// 0.21538120276105854429517i.
#[test]
fn laplace_transform_of_rational_functions() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");
    let l = (&t + 1).powi(-1).try_laplace(&t, &s).unwrap();
    assert_eq!(l.to_string(), "exp(s)*expint(1, s)");
    let l = (t.powi(2) + 1).powi(-1).try_laplace(&t, &s).unwrap();
    assert!(
        close(value(&l.subs(&s, &ctx.int(2))), 0.399_020_988_594_183_85),
        "{l}"
    );
    let l = (t.powi(2) - 2 * &t + 2)
        .powi(-1)
        .try_laplace(&t, &s)
        .unwrap();
    let at = ctx.rational(1, 2) + 3 * ctx.i_unit();
    let z = l.subs(&s, &at).eval_complex64().unwrap();
    assert!((z.re + 0.103_452_456_158_338_9).abs() < 1e-12, "{l}");
    assert!((z.im + 0.215_381_202_761_058_54).abs() < 1e-12, "{l}");
}

/// The z-transform refused `2^(−n)`, `n·3^(−n)` (the table wants `aⁿ`
/// literally), `sinh n`, `C(n + 2, 2)` and `1/(n + 1)`.  mpmath `nsum` of
/// `x[n]·4^(−n)`: `sinh n` → 1.00976291901772464109414118604, `C(n + 2, 2)`
/// → 64/27, `1/(n + 1)` → `4 ln(4/3)` = 1.15072828980712370975687602398.
#[test]
fn z_transform_of_geometric_forms() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let z = ctx.symbol("z");
    let four = ctx.int(4);
    let at = |e: &str| -> f64 {
        let f = ctx.parse(e).unwrap();
        let x = f
            .z_transform(&n, &z)
            .unwrap_or_else(|er| panic!("{f}: {er}"));
        value(&x.subs(&z, &four))
    };
    assert!(close(at("2^(-n)"), 8.0 / 7.0));
    assert!(close(at("n*3^(-n)"), 12.0 / 121.0));
    assert!(close(at("sinh(n)"), 1.009_762_919_017_724_6));
    assert!(close(at("binomial(n + 2, 2)"), 64.0 / 27.0));
    assert!(close(at("1/(n + 1)"), 1.150_728_289_807_123_7));
}
