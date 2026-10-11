//! Continuity of antiderivatives, second round: the jumps the first one
//! (`v34_continuity.rs`) left.
//!
//! * `RootSum`s whose polynomial has real roots, through `t = tan(x/2)`:
//!   `∫ dx/(2·sin⁵x + 1)` is `Σ ρ·ln(tan(x/2) + β(ρ))` over the roots of a
//!   degree-10 polynomial, some real (the residues at the integrand's real
//!   poles).  Its jump at `x = π` was left out (it was computed only without
//!   real roots), so `F(4) − F(2)` was −1.5391 for 1.9271.  Now `J =
//!   G(+∞) − G(−∞)` is `2πi` times the sum of the residues of the integrand
//!   in `t` at its poles in the upper half-plane, written as a `RootSum` over
//!   all roots whose body vanishes at the real ones.
//! * Real parameters.  `∫ dx/(a + b·sin²x)` (`a`, `b > 0`) went through
//!   `tan(x/2)` to a quartic whose roots' half-planes depend on `a`, `b`:
//!   four logarithms, jumping at every odd multiple of `π`.  Bioche's rule
//!   (`f(x + π) = f(x)`: `t = tan x`) gives `∫ dt/(a + (a + b)·t²)`, whose
//!   jump is decided (the form Rubi gives, plus the floor term).  `∫ dx/(a +
//!   b·cos x)` (`a`, `b > 0`) jumps at `x = π` exactly when `a > b`: the
//!   ends of `ln(tan(x/2) ± √(b² − a²)/(a − b))` are now written with
//!   `sign(im(√(b² − a²))/(a − b))`, right for every real `a ≠ b`.
//!   `atan(a·t/√(a² + 2))` for a real `a` of undecided sign tends to
//!   `±π/2·sign(a)`.
//!
//! Oracle: `F(b) − F(a)` (the parameters substituted after integrating, as
//! the continuity hunter does) against mpmath `quad` over `[a, b]` split
//! every 0.25 (`mp.dps = 30`), on intervals where the integrand is
//! continuous and that contain poles of the substitution.  The values of the
//! 0.37 answers ("before") are cited with each case.

use symplex::prelude::*;

/// `∫ src`, with the parameters `params` declared and then substituted:
/// `F′ − f` at sample points and `F(b) − F(a)` against `want`.
fn check(
    src: &str,
    params: &[(&str, Assumption, (i64, i64))],
    a: (i64, i64),
    b: (i64, i64),
    want: f64,
) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let symbols: Vec<(Ex, Ex)> = params
        .iter()
        .map(|&(name, assumption, (p, q))| {
            (
                ctx.symbol_with(name, &[assumption]).unwrap(),
                ctx.rational(p, q),
            )
        })
        .collect();
    let f = ctx.parse(src).unwrap();
    let big_f = f.integrate(&x);
    assert!(!big_f.has_unevaluated(), "∫ {src} = {big_f}");
    let bind = |e: &Ex| {
        let mut e = e.clone();
        for (s, v) in &symbols {
            e = e.subs(s, v);
        }
        e
    };
    let residual = bind(&(&big_f.diff(&x) - &f));
    for (p, q) in [(3, 10), (17, 10), (-22, 10), (41, 10)] {
        let r = residual
            .subs(&x, &ctx.rational(p, q))
            .eval_complex64()
            .unwrap();
        assert!(
            r.norm() < 1e-10,
            "∫ {src} = {big_f}: F′ − f = {r} at {p}/{q}"
        );
    }
    let bound = bind(&big_f);
    let at = |(p, q): (i64, i64)| {
        bound
            .subs(&x, &ctx.rational(p, q))
            .eval_complex64()
            .unwrap()
    };
    let got = at(b) - at(a);
    assert!(
        (got.re - want).abs() < 1e-12 * want.abs().max(1.0) && got.im.abs() < 1e-12,
        "∫ {src} = {big_f}: F(b) − F(a) = {got}, quadrature {want}"
    );
}

/// `RootSum`s with real roots through `tan(x/2)`, across `x = π` (`x =
/// −3π` for the third).  Before: −1.5390801953, −1.2166234777,
/// 1.1930935876, −0.3504595571 and −0.1747925378 + 0.1303755339i (the
/// principal logarithms of the real roots' terms differ by `iπ·Σ ρ` between
/// `t → +∞` and `t → −∞`).  mpmath: `quad(lambda t: 1/(2*sin(t)**5+1),
/// linspace(2, 4, 9))` = 1.927106362339513110695; `1/(3*sin(t)**5+1)` on
/// `[2, 4]` = 2.004909332915440140171; `1/(1-2*cos(t)**5)` on `[-12, -7]` =
/// 4.659280145212271428313; `1/(2*sin(t)**3+1)` on `[2, 4]` =
/// 2.288312919180059799354; `1/(-2*sin(t)**3+2*cos(t)+1)` on `[2.6, 3.6]`
/// = −1.082095477734278774039 (the integrand's zeros nearest π are 1.3 and
/// 3.84).
#[test]
fn root_sums_with_real_roots_are_continuous() {
    for (src, a, b, want) in [
        ("1/(2*sin(x)^5 + 1)", (2, 1), (4, 1), 1.927_106_362_339_513),
        ("1/(3*sin(x)^5 + 1)", (2, 1), (4, 1), 2.004_909_332_915_44),
        (
            "1/(1 - 2*cos(x)^5)",
            (-12, 1),
            (-7, 1),
            4.659_280_145_212_271,
        ),
        ("1/(2*sin(x)^3 + 1)", (2, 1), (4, 1), 2.288_312_919_180_06),
        (
            "1/(-2*sin(x)^3 + 2*cos(x) + 1)",
            (13, 5),
            (18, 5),
            -1.082_095_477_734_278_8,
        ),
    ] {
        check(src, &[], a, b, want);
    }
}

/// Positive parameters, substituted after integrating.  Before:
/// −2.7125201627 (`a = 2`, `b = 1`), −1.8759006274 (`1/(1 + 2·sin²x)` on
/// `[1, 4]`), −1.3420495159, 6.0764755482, −2.0593874752, −1.0117171216.
/// mpmath: `1/(2+cos(t))` on `[3, 4]` = 0.9150785657603947359163;
/// `1/(1+2*sin(t)**2)` on `[1, 4]` = 1.75169810110731361503;
/// `1/(4*sin(t)**2+cos(t)**2)` on `[1, 5]` = 1.799543137659450150465;
/// `cos(t)/(3+2*cos(t))` on `[-4, 4]` = −2.35330212907694048708;
/// `sin(t)**2/(1+3*cos(t)**2)` on `[0, 7]` = 2.129402729603283326336;
/// `1/(2+cos(t))**2` on `[-4, 4]` = 3.825081182984715311589.
#[test]
fn positive_parameters_are_continuous() {
    let pos = Assumption::Positive;
    for (src, pa, pb, a, b, want) in [
        (
            "1/(a + b*cos(x))",
            (2, 1),
            (1, 1),
            (3, 1),
            (4, 1),
            0.915_078_565_760_394_7,
        ),
        (
            "1/(a + b*sin(x)^2)",
            (1, 1),
            (2, 1),
            (1, 1),
            (4, 1),
            1.751_698_101_107_313_6,
        ),
        (
            "1/(a^2*sin(x)^2 + b^2*cos(x)^2)",
            (2, 1),
            (1, 1),
            (1, 1),
            (5, 1),
            1.799_543_137_659_450_2,
        ),
        (
            "cos(x)/(a + b*cos(x))",
            (3, 1),
            (2, 1),
            (-4, 1),
            (4, 1),
            -2.353_302_129_076_940_5,
        ),
        (
            "sin(x)^2/(a + b*cos(x)^2)",
            (1, 1),
            (3, 1),
            (0, 1),
            (7, 1),
            2.129_402_729_603_283_3,
        ),
        (
            "1/(a + b*cos(x))^2",
            (2, 1),
            (1, 1),
            (-4, 1),
            (4, 1),
            3.825_081_182_984_715_3,
        ),
    ] {
        check(src, &[("a", pos, pa), ("b", pos, pb)], a, b, want);
    }
}

/// `∫ dx/(a + b·cos x)` for `b > a > 0`: no jump at `x = π` (the
/// logarithms' arguments are real there), and the `sign(im(·))` term is 0
/// (0.37 gave the same value).  mpmath: `quad(lambda t: 1/(1+2*cos(t)),
/// linspace(2.2, 4, 9))` = −2.842107220332707756565 (the integrand's poles
/// are `2π/3` and `4π/3` ± 2kπ, so `[2.2, 4]` is between them).
#[test]
fn no_jump_where_the_logarithms_are_real() {
    let pos = Assumption::Positive;
    check(
        "1/(a + b*cos(x))",
        &[("a", pos, (1, 1)), ("b", pos, (2, 1))],
        (11, 5),
        (4, 1),
        -2.842_107_220_332_707_6,
    );
}

/// A real parameter of undecided sign: `∫ dx/(a² + 1 + cos x)` at `a = −3`.
/// Before: −0.5216044900.  mpmath: `quad(lambda t: 1/(10+cos(t)),
/// linspace(3, 4, 5))` = 0.109879393403013406916.
#[test]
fn real_parameter_of_undecided_sign() {
    check(
        "1/(a^2 + 1 + cos(x))",
        &[("a", Assumption::Real, (-3, 1))],
        (3, 1),
        (4, 1),
        0.109_879_393_403_013_4,
    );
}

/// The forms: Bioche's `t = tan x` for `f(x + π) = f(x)` with positive
/// parameters (Rubi's antiderivative plus the floor term), and no branch for
/// a non-real degenerate value of a real parameter (`a = ±i√2`, where 0.37
/// nested two unreachable cases).
///
/// Since 0.42 `√(b²)` is `b` for a positive `b` when it is built, so the
/// solver gives the degenerate value `a = 0` of `√(a² + ab)` exactly (it
/// was `(√(b²) − b)/2`, whose case did not integrate) and the answer
/// carries its branch — unreachable for a positive `a`, as the `b = −a` one
/// already was: the degenerate cases do not consult the sign assumptions of
/// the parameters (a follow-up in `integrate.rs`).  The value is unchanged.
#[test]
fn forms_with_declared_parameters() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let _a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();
    let _b = ctx.symbol_with("b", &[Assumption::Positive]).unwrap();
    let got = ctx.parse("1/(a + b*sin(x)^2)").unwrap().integrate(&x);
    assert_eq!(
        got.to_string(),
        "Piecewise(Piecewise(atan((a + b)*tan(x)/sqrt(a^2 + a*b))/sqrt(a^2 + a*b) + floor(x/pi + 1/2)*pi/sqrt(a^2 + a*b) if b != -a, tan(x)/a if True) if a != 0, -cos(x)/(b*sin(x)) if True)"
    );
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let _a = ctx.symbol_with("a", &[Assumption::Real]).unwrap();
    let got = ctx.parse("1/(a^2 + 1 + cos(x))").unwrap().integrate(&x);
    assert_eq!(
        got.to_string(),
        "Piecewise(2*atan(a*tan(1/2*x)/sqrt(a^2 + 2))/(a*sqrt(a^2 + 2)) + 2*sign(a)*floor(x/(2*pi) + 1/2)*pi/(a*sqrt(a^2 + 2)) if a != 0, tan(1/2*x) if True)"
    );
}
