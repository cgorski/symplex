//! Limits at branch cuts approached from one side, and limits at `∞` of
//! special functions through their asymptotic series (0.37).  Each test
//! says what was refused or wrong before and cites its oracle: SymPy 1.14
//! `limit(...)`, and mpmath 1.3 values of the expression near the point
//! (difference quotients at `x = ±10⁻³⁰` with `mp.dps` 90 and 130, which
//! agree; values at `x = 10², 10³, 10⁴` or more at `∞`).

use symplex::prelude::*;

fn c64(e: &Ex) -> num_complex::Complex64 {
    e.eval_complex64().unwrap()
}

fn lim(ctx: &Context, f: &str, point: &str, dir: Direction) -> Result<Ex, SymplexError> {
    let x = ctx.symbol("x");
    ctx.parse(f)
        .unwrap()
        .try_limit_dir(&x, &ctx.parse(point).unwrap(), dir)
}

fn assert_close(src: &str, dir: Direction, got: &Ex, want: (f64, f64)) {
    let z = c64(got);
    let scale = want.0.hypot(want.1).max(1.0);
    assert!(
        (z.re - want.0).hypot(z.im - want.1) < 1e-12 * scale,
        "{src} {dir:?}: {got} = {z}, want {want:?}"
    );
}

/// Refused before from both sides ("a non-real argument approaches the
/// branch cut"): from the side the argument approaches from, the function
/// is analytic and continues across the cut — `ln u = ln(−u) ± iπ`,
/// `asin u = π/2 ± i·acosh u`, … — and each side has its own limit.  The
/// branch-cut hunter (1,188 one-sided difference quotients at the cuts of
/// `ln`, `√`, `∛`, the inverse trigonometric and hyperbolic functions):
/// 186 refused → 0, all against mpmath.
///
/// Oracles (mpmath at `x = ±10⁻³⁰`; SymPy `limit` where it agrees):
/// `x/(ln(−2 + ix) − ln(−2))`: `2i` / `0` (SymPy `2*I`, `0`);
/// `x/(√(−2 + ix) − √(−2))`: `2√2` / `0`;
/// `x/(asin(−2 + ix) − asin(−2))`: `√3` / `0`;
/// `x/(acosh(1/2 + ix) − acosh(1/2))`: `√3/2` / `0`;
/// `x/(atanh(2 + ix) − atanh 2)`: `0` / `3i`;
/// `x/(∛(−2 + ix) − ∛(−2))`: `3·(−2)^{2/3}/i = 4.12418891099581 +
/// 2.38110157795230i` / `0`;
/// `x/(acosh(acosh 0 + (1 + i)x) − acosh(acosh 0))` from the left: `0`.
/// From the side where the principal value is continuous with the cut the
/// limit is `1/f′`; from the other, the difference tends to the jump and
/// the quotient to `0`.  Two-sided they differ.
#[test]
fn one_sided_limits_at_a_branch_cut() {
    let ctx = Context::new();
    let s3 = 3f64.sqrt();
    type Point = (f64, f64);
    let cases: [(&str, Point, Point); 6] = [
        ("x/(ln(-2 + I*x) - ln(-2))", (0.0, 2.0), (0.0, 0.0)),
        (
            "x/(sqrt(-2 + I*x) - sqrt(-2))",
            (2.0 * 2f64.sqrt(), 0.0),
            (0.0, 0.0),
        ),
        ("x/(asin(-2 + I*x) - asin(-2))", (s3, 0.0), (0.0, 0.0)),
        (
            "x/(acosh(1/2 + I*x) - acosh(1/2))",
            (s3 / 2.0, 0.0),
            (0.0, 0.0),
        ),
        ("x/(atanh(2 + I*x) - atanh(2))", (0.0, 0.0), (0.0, 3.0)),
        (
            "x/(cbrt(-2 + I*x) - cbrt(-2))",
            (4.124_188_910_995_81, 2.381_101_577_952_30),
            (0.0, 0.0),
        ),
    ];
    for (src, right, left) in cases {
        for (dir, want) in [(Direction::Right, right), (Direction::Left, left)] {
            let l = lim(&ctx, src, "0", dir).unwrap_or_else(|e| panic!("{src} {dir:?}: {e}"));
            assert_close(src, dir, &l, want);
        }
        assert!(lim(&ctx, src, "0", Direction::Both).is_err(), "{src}");
    }
}

/// One-sided series at a point of a cut: refused before for an argument
/// leaving the cut (`ln`, powers, `asin`, …), and wrong for a real
/// direction across a cut on the imaginary axis and for `Ei`, `Ci` (the
/// constant term on the cut from both sides: `series_dir(atan(2i + x), x,
/// 0, 3, Left)` had the constant `atan(2i) = π/2 + 0.549i`; it is `−π/2 +
/// 0.549i`).  Now the expansion of the continuous side, mapped by the
/// jump across the cut on the other side (SymPy's `log._eval_nseries`:
/// `−2πi` below; `atan`: `F − π` from `Re < 0`).  A hunter over 864
/// one-sided series at the cuts of `ln`, `√`, `∛`, the inverse functions,
/// `Ei`, `Ci`, `ln Γ` (residual against mpmath at `h = ±10⁻⁶…⁻¹²`): 30 wrong,
/// 194 refused → 0 wrong, 12 refused.  Oracles: mpmath at `x = ±0.001`
/// (30 digits), the series to order 3 within `10⁻⁷`.
#[test]
fn one_sided_series_at_a_branch_cut() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let (r, l) = (Direction::Right, Direction::Left);
    for (src, dir, want) in [
        (
            "ln(-2 + I*x)",
            r,
            (0.693_147_305_559_93, 3.141_092_653_631_46),
        ),
        (
            "ln(-2 + I*x)",
            l,
            (0.693_147_305_559_93, -3.141_092_653_631_46),
        ),
        (
            "atan(2*I + x)",
            l,
            (-1.570_462_993_622_057, 0.549_305_922_111_956),
        ),
        (
            "atan(2*I + x)",
            r,
            (1.570_462_993_622_057, 0.549_305_922_111_956),
        ),
        (
            "Ei(-2 + I*x)",
            r,
            (-0.048_900_459_957_343_3, 3.141_524_985_976_37),
        ),
        (
            "Ei(-2 + I*x)",
            l,
            (-0.048_900_459_957_343_3, -3.141_524_985_976_37),
        ),
        (
            "sqrt(-4 - I*x)",
            r,
            (0.000_249_999_998_046_875, -2.000_000_015_625),
        ),
        (
            "asin(2 + I*x)",
            r,
            (1.570_218_976_621_932, 1.316_958_089_374_848),
        ),
    ] {
        let s = ctx
            .parse(src)
            .unwrap()
            .try_series_dir(&x, &zero, 3, dir)
            .unwrap_or_else(|e| panic!("{src} {dir:?}: {e}"));
        let h = ctx
            .parse(if dir == r { "1/1000" } else { "-1/1000" })
            .unwrap();
        let z = c64(&s.subs(&x, &h));
        assert!(
            (z.re - want.0).hypot(z.im - want.1) < 1e-7,
            "{src} {dir:?}: {s} = {z} at x = {h}, want {want:?}"
        );
    }
    // The two sides differ: no two-sided expansion.
    let f = ctx.parse("ln(-2 + I*x^3)").unwrap();
    assert!(f.try_series(&x, &zero, 3).is_err());
}

/// Arguments that move along the cut, or reach it with a perpendicular
/// part of one sign from both sides (`x²`): refused before, now the value
/// of the continuous side.  mpmath at `x = ±10⁻³⁰`: `x/(asinh(2i + ix) −
/// asinh 2i)` → `√3` from both sides (the argument stays on the cut
/// `i·[1, ∞)`); `x²/(atan(2i + x²) − atan 2i)` → `−3` (SymPy: `-3`);
/// `x²/(asinh(−2i + x²) − asinh(−2i))` → `0` (from `Re > 0`, the side of
/// the lower cut where `asinh` jumps).
#[test]
fn arguments_along_or_on_one_side_of_a_cut() {
    let ctx = Context::new();
    let s3 = 3f64.sqrt();
    for (src, want) in [
        ("x/(asinh(2*I + I*x) - asinh(2*I))", (s3, 0.0)),
        ("x^2/(atan(2*I + x^2) - atan(2*I))", (-3.0, 0.0)),
        ("x^2/(asinh(-2*I + x^2) - asinh(-2*I))", (0.0, 0.0)),
    ] {
        for dir in [Direction::Right, Direction::Left, Direction::Both] {
            let l = lim(&ctx, src, "0", dir).unwrap_or_else(|e| panic!("{src} {dir:?}: {e}"));
            assert_close(src, dir, &l, want);
        }
    }
}

/// The open item of the 0.34 hand-off: `x·(acosh(−1 − 1/x) − acosh(−1 −
/// 2/x))` at `∞` was refused ("a non-real argument approaches the branch
/// cut of ln": Gruntz's `acosh u = ln(u + √(u+1)·√(u−1))` puts a real
/// negative argument under `ln`).  `acosh` of an argument on its cut is
/// now `acosh(−u) + iπ` first: `x·(acosh(1 + 1/x) − acosh(1 + 2/x)) ~
/// (√2 − 2)·√x`.  SymPy: `limit(x*(acosh(-1 - 1/x) - acosh(-1-2/x)), x,
/// oo)` → `-oo`.  The `asinh(atanh(1 − x))` of the 0.34 notes (from
/// `Re < 0`): SymPy `-acosh(pi/2) + I*pi/2` ≈ −1.02322747854755 +
/// 1.5707963267949i (was refused).
#[test]
fn acosh_on_its_cut_at_infinity() {
    let ctx = Context::new();
    let b = Direction::Both;
    let l = lim(&ctx, "x*(acosh(-1 - 1/x) - acosh(-1 - 2/x))", "oo", b).unwrap();
    assert_eq!(l.to_string(), "-oo");
    let l = lim(&ctx, "asinh(atanh(1 - x))", "oo", b).unwrap();
    assert_close(
        "asinh(atanh(1 - x))",
        b,
        &l,
        (-1.023_227_478_547_55, std::f64::consts::FRAC_PI_2),
    );
}

/// Wrong values before, at points of a cut approached from off the axis:
/// the value on the cut was taken from both sides.
///
/// * `limit(asin(2 + ix), x, 0, '+')` was `asin 2 = π/2 − 1.317i`; SymPy
///   `pi - asin(2)` (mpmath `asin(2 + 10⁻²⁵i)` = `1.5708 + 1.31696i`).
/// * `Ei(−2 + ix)` was `Ei(−2)` (real) from both sides; mpmath `ei(-2 ±
///   1e-25j)` = `−0.0489005107 ± 3.14159265i` (SymPy 1.14 gives `Ei(-2)`
///   too, wrongly).
/// * `Ci(−2 + ix)` from the left was `Ci(−2) = Ci 2 + iπ`; mpmath `ci(-2 -
///   1e-25j)` = `0.4229808288 − 3.14159265i`.
/// * `ln Γ(−5/2 + ix)` from the left was `ln Γ(−5/2) = ln(8√π/15) − 3πi`;
///   mpmath `loggamma(-2.5 - 1e-25j)` = `−0.0562437165 + 9.42477796i`.
/// * `Ei(−x + i)` at `∞` was `0`; mpmath `ei(-100 + 1j)` =
///   `−1.96e-46 + 3.14159265i` (SymPy 1.14: `0`, wrongly).
#[test]
fn values_on_a_cut_were_taken_from_the_wrong_side() {
    let ctx = Context::new();
    let pi = std::f64::consts::PI;
    let (r, l) = (Direction::Right, Direction::Left);
    let ach2 = 1.316_957_896_924_816_7;
    let ei2 = -0.048_900_510_708_061_12;
    for (src, point, dir, want) in [
        ("asin(2 + I*x)", "0", r, (pi / 2.0, ach2)),
        ("asin(2 + I*x)", "0", l, (pi / 2.0, -ach2)),
        ("Ei(-2 + I*x)", "0", r, (ei2, pi)),
        ("Ei(-2 + I*x)", "0", l, (ei2, -pi)),
        ("Ci(-2 + I*x)", "0", l, (0.422_980_828_774_865, -pi)),
        (
            "loggamma(-5/2 + I*x)",
            "0",
            l,
            (-0.056_243_716_497_674_05, 3.0 * pi),
        ),
        ("Ei(-x + I)", "oo", Direction::Both, (0.0, pi)),
    ] {
        let v = lim(&ctx, src, point, dir).unwrap_or_else(|e| panic!("{src} {dir:?}: {e}"));
        assert_close(src, dir, &v, want);
    }
    assert_eq!(
        lim(&ctx, "asin(2 + I*x)", "0", r).unwrap().to_string(),
        "1/2*pi + ln(sqrt(3) + 2)*I"
    );
}

/// Special functions with a cut and no continuation formula here gave the
/// value on the cut from both sides; now refused from the side where it is
/// wrong (and, for want of the formula, from the other).  mpmath:
/// `lambertw(-1 - 1e-25j)` = `−0.3181 − 1.3372i` (was `W(−1) = −0.3181 +
/// 1.3372i` from the left); `polylog(2, 3 + 1e-25j)` = `2.3202 + 3.4514i`
/// (was `polylog(2, 3) = 2.3202 − 3.4514i` from the right);
/// `besselk(0, -2 - 1e-25j)` = `0.1139 + 7.1615i` (was `besselk(0, −2)`).
#[test]
fn special_functions_at_their_cut_are_not_taken_on_it() {
    let ctx = Context::new();
    for (src, dir) in [
        ("LambertW(-1 + I*x)", Direction::Left),
        ("polylog(2, 3 + I*x)", Direction::Right),
        ("besselk(0, -2 + I*x)", Direction::Left),
    ] {
        if let Ok(v) = lim(&ctx, src, "0", dir) {
            panic!("{src} {dir:?}: {v}");
        }
    }
}

/// Refused before ("leading coefficients cancel and series expansion
/// failed"): Gruntz's `Γ(z) = e^{ln Γ(z)}` leaves `e^{−ln Γ(1/ω)}·e^{ln Γ(1/ω
/// + 1)}`, two essential singularities whose product is `1/ω`; the
/// exponentials of a product are now combined before the series.  SymPy:
/// `limit(gamma(x+1)/gamma(x) - x, x, oo)` → `0`,
/// `limit(x*(gamma(x+1/2)/(gamma(x)*sqrt(x)) - 1), x, oo)` → `-1/8`
/// (mpmath at `x = 10⁴`: `−0.1249992187`).
#[test]
fn gamma_ratios_at_infinity() {
    let ctx = Context::new();
    let b = Direction::Both;
    for (src, want) in [
        ("gamma(x+1)/gamma(x) - x", "0"),
        ("x*(gamma(x+1/2)/(gamma(x)*sqrt(x)) - 1)", "-1/8"),
    ] {
        assert_eq!(lim(&ctx, src, "oo", b).unwrap().to_string(), want, "{src}");
    }
}

/// Refused before ("a function of unknown growth at the limit of its
/// argument"): the special functions are written at `±∞` with their
/// exponential factor explicit and an internal function whose series in
/// `1/z` the series engine knows (DLMF 6.12, 7.12, 8.11, 8.20, 10.40,
/// 25.2).  SymPy 1.14 and mpmath (at `x = 10², 10³, 10⁴`, 60 digits;
/// `ζ` and `Chi − Shi` at `x = 100, 300, 600` with 700 digits):
///
/// * `zeta(2x) − 1` → `0` (SymPy `0`), `zeta(kx) − 1` (`k > 0`) → `0`
///   (SymPy `0`), `2^x·(zeta(x) − 1)` → `1` (mpmath `1.0`; SymPy unevaluated);
/// * `Chi(x)·x·e⁻ˣ`, `Shi(x)·x·e⁻ˣ`, `Chi(−x)·x·e⁻ˣ` → `1/2` (mpmath
///   `0.50005001` at 10⁴; SymPy `0`, wrongly), `(Chi x − Shi x)·x·eˣ` → `−1`
///   (mpmath `−0.99834` at 600);
/// * `I₀(x)·√x·e⁻ˣ` → `1/√(2π)` (mpmath `0.3989472675`; SymPy times out),
///   `x·(I₁(x)/I₀(x) − 1)` → `−1/2` (`−0.5000125013`);
/// * `K₀(x)·√x·eˣ` → `√(π/2)` (SymPy `sqrt(2)*sqrt(pi)/2`),
///   `x·(K₁(x)/K₀(x) − 1)` → `1/2` (`0.4999875012`);
/// * `erfi(x)·x·e^{−x²}` → `1/√π` (SymPy `1/sqrt(pi)`);
/// * `E₁(x)·x·eˣ` → `1` (mpmath `0.99990002`; SymPy unevaluated),
///   `E₂(x)·x·eˣ` → `1` (`0.99980006`; SymPy `-oo*I`, wrongly),
///   `x·(E₁(x)·x·eˣ − 1)` → `−1` (`−0.99980006`);
/// * `Γ(1/3, x)·x^{2/3}·eˣ` → `1` (`0.9999333444`), `γ(1/2, x)` → `√π`.
#[test]
fn special_functions_at_infinity() {
    let ctx = Context::new();
    let _k = ctx.symbol_with("k", &[Assumption::Positive]).unwrap();
    let x = ctx.symbol("x");
    let oo = ctx.parse("oo").unwrap();
    let f = ctx.parse("zeta(k*x) - 1").unwrap();
    assert_eq!(f.try_limit(&x, &oo).unwrap().to_string(), "0");
    let sqrt_half_pi = (std::f64::consts::PI / 2.0).sqrt();
    let inv_sqrt_pi = 1.0 / std::f64::consts::PI.sqrt();
    for (src, want) in [
        ("zeta(2*x) - 1", 0.0),
        ("2^x*(zeta(x) - 1)", 1.0),
        ("Chi(x)*x*exp(-x)", 0.5),
        ("Shi(x)*x*exp(-x)", 0.5),
        ("Chi(-x)*x*exp(-x)", 0.5),
        ("(Chi(x) - Shi(x))*x*exp(x)", -1.0),
        ("besseli(0,x)*sqrt(x)*exp(-x)", inv_sqrt_pi / 2f64.sqrt()),
        ("x*(besseli(1,x)/besseli(0,x) - 1)", -0.5),
        ("besselk(0,x)*sqrt(x)*exp(x)", sqrt_half_pi),
        ("x*(besselk(1,x)/besselk(0,x) - 1)", 0.5),
        ("erfi(x)*x*exp(-x^2)", inv_sqrt_pi),
        ("expint(1,x)*x*exp(x)", 1.0),
        ("expint(2,x)*x*exp(x)", 1.0),
        ("x*(expint(1,x)*x*exp(x) - 1)", -1.0),
        ("uppergamma(1/3,x)*x^(2/3)*exp(x)", 1.0),
        ("lowergamma(1/2,x)", std::f64::consts::PI.sqrt()),
    ] {
        let b = Direction::Both;
        let v = lim(&ctx, src, "oo", b).unwrap_or_else(|e| panic!("{src}: {e}"));
        assert_close(src, b, &v, (want, 0.0));
    }
}

/// `I₋ν` and `I_ν` have the same expansion in `1/x` to every order (the
/// coefficients depend on `ν²`); their difference is the exponentially
/// smaller `−(2/π)·sin(νπ)·K_ν(x)`, which the expansion cannot see: taken
/// term by term the limit below would be `0`.  It is `−(√3/π)·√(π/2) =
/// −0.690988298942671` (mpmath, 200 digits: `−0.6900387`, `−0.6905110`,
/// `−0.6907490` at `x = 50, 100, 200`; SymPy 1.14 raises `TypeError`), so
/// the pair is refused.
#[test]
fn bessel_i_beside_its_negative_order_is_refused() {
    let ctx = Context::new();
    let src = "(besseli(1/3,x) - besseli(-1/3,x))*exp(x)*sqrt(x)";
    assert!(lim(&ctx, src, "oo", Direction::Both).is_err());
}

/// `ζ(s)` at `s = 1`: `1/(s − 1) + γ` is the expansion to order 1 (exponents
/// below 1); the next coefficient is `−γ₁`, a Stieltjes constant, for which
/// symplex has no node, so order 2 is refused rather than truncated.
/// (SymPy 1.14: `series(zeta(x), x, 1, 2)` raises `PoleError`.)
#[test]
fn zeta_at_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let z = ctx.parse("zeta(x)").unwrap();
    assert_eq!(
        z.try_series(&x, &one, 1).unwrap().to_string(),
        "1/(x - 1) + EulerGamma"
    );
    assert!(z.try_series(&x, &one, 2).is_err());
}
