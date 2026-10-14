//! Tests of the 0.50 refactor (b2): silent wrong limits.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn p(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

fn lim(ctx: &Context, f: &str, point: &str, dir: Direction) -> Result<Ex, SymplexError> {
    let x = ctx.symbol("x");
    p(ctx, f).try_limit_dir(&x, &p(ctx, point), dir)
}

/// The argument of the direction `d` of an infinite value `d·∞`.
fn direction_arg(ctx: &Context, v: &Ex) -> f64 {
    let t = ctx.symbol("t");
    let z = v
        .subs(&ctx.infinity(), &t)
        .subs(&t, &p(ctx, "10^40"))
        .eval_complex64()
        .unwrap();
    z.im.atan2(z.re)
}

/// `Γ(2, z) = (1 + z)·e^{−z}` with `z = c − e·x` and the non-real constant
/// `c = (−3)^{−1/3}` tends at `∞` to the infinity in the direction of
/// `−e^{−c}`, whose argument is `3^{1/6}/2 − π`; with `c = (−3)^{1/3}` and
/// `z = c − x`, of argument `π − 3^{5/6}/2`.  Up to 0.49 both limits were
/// `−∞`: the representative `ω = e^{−(e·x − c)}` of the Gruntz algorithm
/// kept the non-real constant `(−1)^{2/3}`, written without `i`, so `ω` was
/// not positive and the phase was lost.
///
/// Oracle: mpmath (`mp.dps = 30`), `gammainc(2, (-3)**(-1/3) - e*50)` =
/// `−8.3230113e60 − 5.7545283e60j` (argument −2.53666 at `x = 50`;
/// `3^{1/6}/2 − π` = −2.5411241760017918751); `gammainc(2, (-3)**(1/3) - 50)`
/// = `−3.550126e22 + 1.1645238e23j` (argument 1.86670; `π − 3^{5/6}/2` =
/// 1.8925678871063867590).
#[test]
fn upper_gamma_of_a_non_real_constant_minus_x_keeps_its_phase() {
    let ctx = Context::new();
    let v = lim(
        &ctx,
        "uppergamma(2, (-3)^(-1/3) - E*x)",
        "oo",
        Direction::Both,
    )
    .unwrap();
    assert_ne!(v, ctx.neg_infinity());
    let theta = 3f64.powf(1.0 / 6.0) / 2.0 - std::f64::consts::PI;
    assert!((direction_arg(&ctx, &v) - theta).abs() < 1e-12, "{v}");

    let v = lim(&ctx, "uppergamma(2, (-3)^(1/3) - x)", "oo", Direction::Both).unwrap();
    assert_eq!(v, p(&ctx, "-exp(-1/2*3^(5/6)*I)*oo"));
    let theta = std::f64::consts::PI - 3f64.powf(5.0 / 6.0) / 2.0;
    assert!((direction_arg(&ctx, &v) - theta).abs() < 1e-12, "{v}");
}

/// `e^{x + asin 2}` at `∞` is the infinity in the direction `e^{i·Im asin 2}`
/// (`asin 2 = π/2 − 1.3169…i`), and `e^{acos 3 − x}` at `−∞` that of
/// `e^{acos 3}` (`acos 3 = 1.7627…i`, a constant whose real and imaginary
/// parts the complex decomposition does not give exactly).  Up to 0.49 both
/// were `∞`: the non-real constant stayed in the exponent of `ω`.
///
/// Oracle: mpmath (`mp.dps = 30`), `arg(exp(50 + asin(2)))` =
/// −1.31695789692481670862504634731, `arg(exp(acos(3) + 50))` =
/// 1.76274717403908605046521864996.
#[test]
fn a_non_real_constant_without_i_in_a_diverging_exponent_keeps_its_phase() {
    let ctx = Context::new();
    let v = lim(&ctx, "exp(x + asin(2))", "oo", Direction::Both).unwrap();
    assert_ne!(v, ctx.infinity());
    assert!(
        (direction_arg(&ctx, &v) + 1.316_957_896_924_816_7).abs() < 1e-12,
        "{v}"
    );
    let v = lim(&ctx, "exp(acos(3) - x)", "-oo", Direction::Both).unwrap();
    assert_ne!(v, ctx.infinity());
    assert!(
        (direction_arg(&ctx, &v) - 1.762_747_174_039_086).abs() < 1e-12,
        "{v}"
    );
}

/// A limit at `±∞` has a direction only if the phase settles.  `ω^{e₀}` with
/// a non-real `e₀` turns (`x^{−3i} = e^{−3i·ln x}`), and so does `e^{g}`
/// for `g` with a slowly diverging imaginary part.  Up to 0.49
/// `e^{x + i}·((−1)^{2/3} + e·x)^i` at `∞` was `e^{2i}·∞` and
/// `Γ((−1)^{1/7} − x/2)·sinh(√5·i + 2x)` at `−∞` was `−e^{−i√5}·∞` (its
/// `Γ` factor turns as `Im c·ln|x|`); both are refused now, as are
/// `(√5·i + 2x)^x·x^{−3i}` and `(√5·i + 2x)^x·x^{2 − 3i + e·x}` (refused
/// up to 0.49 too; once the settling phase of the first factor is split
/// off, only the rules here keep them from reading `e^{i√5/2}·∞`).  The
/// phase that does settle is kept: `(√5·i + 2x)^x → e^{i√5/2}·∞`, and
/// `sign((atanh 2 − 2x)^x)` at `−∞` (up to 0.49: `1`) tends to `e^{iπ/4}`
/// or is refused, never `1`.
///
/// Oracle: mpmath (`mp.dps = 30`), `arg((sqrt(-5) + 2X)**X * X**(-3j))`
/// at `X = 10³, 10⁴, 10⁵`: −0.756, −1.380, −2.005 (turning by
/// `−3·ln 10`); `arg((sqrt(-5) + 2X)**X)` = 1.11803 (`√5/2`) at all three;
/// `arg(gamma((-1)**(1/7) + T/2)*sinh(sqrt(-5) - 2T))` at `T = 10³, 10⁵`:
/// −2.681, −0.683 (turning); `arg((atanh(2) + 2T)**(-T))` at `T = 10⁴` =
/// 0.785377 (`π/4`; `1` was wrong).
#[test]
fn a_turning_phase_has_no_directed_limit() {
    let ctx = Context::new();
    let b = Direction::Both;
    for f in [
        "(sqrt(-5) + 2*x)^x*x^(-3*I)",
        "(sqrt(-5) + 2*x)^x*x^(2 - 3*I + E*x)",
        "exp(I + x)*((-1)^(2/3) + E*x)^(I)",
    ] {
        assert!(lim(&ctx, f, "oo", b).is_err(), "{f}");
    }
    assert!(
        lim(
            &ctx,
            "gamma((-1)^(1/7) - 1/2*x)*sinh(sqrt(-5) + 2*x)",
            "-oo",
            b
        )
        .is_err()
    );
    assert_eq!(
        lim(&ctx, "(sqrt(-5) + 2*x)^x", "oo", b).unwrap(),
        p(&ctx, "exp(1/2*sqrt(5)*I)*oo")
    );
    if let Ok(v) = lim(&ctx, "sign((atanh(2) - 2*x)^x)", "-oo", b) {
        let z = v.eval_complex64().unwrap();
        assert!(
            (z.im.atan2(z.re) - std::f64::consts::FRAC_PI_4).abs() < 1e-12,
            "{v}"
        );
    }
}

/// A Fresnel integral grows exponentially along a line off the real axis:
/// for `u = c + k·x`, `sin(πu²/2)` grows like `e^{π·|k·Im c|·|x|}`.  Up to
/// 0.49 `eˣ·fresnelc(√5·i − 2x)` at `−∞` was `0` (the Fresnel factor taken
/// for one of lower class, as along the reals); its modulus grows, and the
/// phase of `sin(πu²/2)` turns with `Re u² = 4x² − 5`: no limit, refused.
///
/// Oracle: mpmath (`mp.dps = 30`), `abs(exp(X)*fresnelc(sqrt(-5) - 2*X))`
/// = 3.37824e26, 3.7324e54, 8.84573e110 at `X` = −5, −10, −20.
#[test]
fn fresnel_integral_of_a_non_real_line_is_not_bounded() {
    let ctx = Context::new();
    assert!(
        lim(
            &ctx,
            "exp(x)*fresnelc(sqrt(-5) - 2*x)",
            "-oo",
            Direction::Both
        )
        .is_err()
    );
    // Along the reals it is: `fresnelc(2x)/x → 0`, `eˣ·fresnelc(2x) → 0` at `−∞`.
    assert_eq!(
        lim(&ctx, "exp(x)*fresnelc(-2*x)", "-oo", Direction::Both).unwrap(),
        ctx.zero()
    );
}

/// `⌊z⌋` and `⌈z⌉` have no value off the reals (`ceiling(1 + I/2)` is an
/// error), so neither has a limit where the argument is not real.  Up to
/// 0.49 `lim_{x→0⁺} ⌈ln(x)^(x²)⌉` was `2` and `⌊ln(x)^(x²)⌋` was `1`
/// (`ln(x)^(x²) → 1` with a positive imaginary part), `lim_{x→0⁻} ⌈xˣ⌉` was
/// `2`, and `lim_{x→0} ⌊1/2 + i·x⌋` was `0`.  A real argument is unchanged.
///
/// Oracle: mpmath (`mp.dps = 30`), `log(1e-9)**(1e-18)` = `1.000…0030 +
/// 3.14159e-18j`, `(-1e-9)**(-1e-9)` = `1.00000002 − 3.14159e-9j` (not
/// real); `floor(3/2 + lambertw(-1e-9))` = 1, `1e-9**1e-9` = 0.99999998
/// (`⌈xˣ⌉ → 1` at `0⁺`).
#[test]
fn floor_and_ceiling_of_a_non_real_value_have_no_limit() {
    let ctx = Context::new();
    let r = Direction::Right;
    assert!(lim(&ctx, "ceiling(ln(x)^(x^2))", "0", r).is_err());
    assert!(lim(&ctx, "floor(ln(x)^(x^2))", "0", r).is_err());
    assert!(lim(&ctx, "ceiling(x^x)", "0", Direction::Left).is_err());
    assert!(lim(&ctx, "floor(1/2 + I*x)", "0", Direction::Both).is_err());
    assert!(lim(&ctx, "floor(1/2 + I*x)", "0", r).is_err());
    assert!(p(&ctx, "ceiling(1 + I/2)").eval_complex64().is_err());
    // Real arguments.
    assert_eq!(lim(&ctx, "ceiling(x^x)", "0", r).unwrap(), ctx.one());
    assert_eq!(
        lim(&ctx, "floor(3/2 + LambertW(-x))", "0", r).unwrap(),
        ctx.one()
    );
    assert_eq!(
        lim(&ctx, "floor(1/2 + x)", "0", Direction::Both).unwrap(),
        ctx.zero()
    );
}

/// `δ` is a distribution on the reals: `δ(1 + i)` has no value, so `δ(1 +
/// i) + 1` has neither a sign nor a zero test (`Ex::is_zero` of `δ(i)` is
/// `None`; SymPy: `DiracDelta(1 + I)` raises "defined only for real
/// values").  Up to 0.49 the limit engine took it for 0 (`evalf` does):
/// `lim_{x→∞} x·(δ(1 + i) + 1)` was `∞`; refused now.  A real argument is
/// unchanged: `δ(2) = 0`.
///
/// Oracle: the definition of `δ`.
#[test]
fn dirac_delta_off_the_reals_has_no_value_in_limits() {
    let ctx = Context::new();
    let b = Direction::Both;
    assert!(lim(&ctx, "x*(DiracDelta(1 + I) + 1)", "oo", b).is_err());
    assert!(lim(&ctx, "x*DiracDelta(1 + I) + x", "oo", b).is_err());
    assert_eq!(p(&ctx, "DiracDelta(I)").is_zero(), None);
    assert_eq!(
        lim(&ctx, "x*(DiracDelta(2) + 1)", "oo", b).unwrap(),
        ctx.infinity()
    );
}

/// `acot(c) = atan(1/c)`, `acoth(c) = atanh(1/c)`, `asech(c) = acosh(1/c)`
/// have no nodes of their own; their values at `c = 0` (`π/2`, `iπ/2`, `∞`)
/// were taken for a literal `0` only.  Up to 0.49 a `c` that `eval` folds
/// to 0 lost them: `acot(min(W(π), 0))` evaluated to `atan(zoo)` and
/// compiled to NaN, `acoth(min(W(π), 0))` to `atanh(zoo)`, `asech(min(W(π),
/// 0))` to `zoo`.  The one-sided limits and series of `acot` at 0 were
/// already right (it is discontinuous there).
///
/// Oracle: SymPy 1.14 `acot(0)` = `pi/2`, `acoth(0)` = `I*pi/2`,
/// `asech(0)` = `oo`; mpmath `acot(0)` = 1.5707963267948966;
/// `limit(acot(x), x, 0, '±')` = `±pi/2`, `acot(x).series(x, 0, 4)` =
/// `pi/2 - x + x**3/3`, from the left `-pi/2 - x + x**3/3`.
#[test]
fn reciprocal_functions_keep_their_value_at_a_folded_zero() {
    let ctx = Context::new();
    let half_pi = p(&ctx, "pi/2");
    let c = "min(LambertW(pi), 0)";
    let acot = p(&ctx, &format!("acot({c})"));
    assert_eq!(acot.eval(), half_pi);
    let f = acot.compile(&[]).unwrap();
    assert_eq!(f.call(&[]), std::f64::consts::FRAC_PI_2);
    assert_eq!(p(&ctx, &format!("acot(2*{c})")).eval(), half_pi);
    assert_eq!(p(&ctx, &format!("acoth({c})")).eval(), p(&ctx, "I*pi/2"));
    assert_eq!(p(&ctx, &format!("asech({c})")).eval(), ctx.infinity());
    // `asec`, `acsc`, `acsch` are `zoo` at 0 either way.
    assert_eq!(p(&ctx, &format!("acsc({c})")).eval(), p(&ctx, "zoo"));

    let x = ctx.symbol("x");
    let zero = ctx.zero();
    let a = x.acot();
    assert_eq!(a.limit_right(&x, &zero), half_pi);
    assert_eq!(a.limit_left(&x, &zero), -&half_pi);
    assert!(a.try_limit(&x, &zero).is_err());
    let right = a.try_series_dir(&x, &zero, 4, Direction::Right).unwrap();
    assert_eq!(right, p(&ctx, "pi/2 - x + x^3/3"));
    let left = a.try_series_dir(&x, &zero, 4, Direction::Left).unwrap();
    assert_eq!(left, p(&ctx, "-pi/2 - x + x^3/3"));
}
