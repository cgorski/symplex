//! Antiderivatives through `t = tan(x/2)` (the half-angle substitution) and
//! `u = tan(a·x + b)` that jumped where the integrand is continuous.
//!
//! `G(tan w)` passes from `G(+∞)` to `G(−∞)` where `w` crosses
//! `π/2 + kπ`: up to 0.37 `∫ dx/(2 + cos x)` was `2/√3·atan(tan(x/2)/√3)`,
//! which drops by `2π/√3` at every odd multiple of `π`, and
//! `∫ tan x/(tan x + 2) dx` ended in `1/5·atan(tan x)`, which drops by
//! `π/5` at every odd multiple of `π/2`.  The derivative is right away
//! from those points, so the integrator's own check (`F′ = f` at sample
//! points) could not see it; definite integrals taken from such an `F`
//! were wrong.  Now `J·⌊w/π + 1/2⌋` with `J = G(+∞) − G(−∞)` is added
//! (D. J. Jeffrey and A. D. Rich, ACM TOMS 20 (1994); SymPy's
//! `Integral.doit` adds `sign(c)·π·⌊(w − π/2)/π⌋` to an `atan(c·tan w + d)`),
//! and `c·atan(tan w)` is written `c·w − c·π·⌊w/π + 1/2⌋`.
//!
//! Oracle: `F(b) − F(a)` against mpmath `quad` over `[a, b]` split every
//! 0.25 (`mp.dps = 30` and 50 agree to all digits shown), on intervals where
//! the integrand is continuous and that contain poles of the substitution.

use symplex::prelude::*;

/// `∫ f`, `F′ − f` at sample points away from the poles of `tan(x/2)` and
/// `tan x`, and `F(b) − F(a)` against `want`.
fn check_continuous(ctx: &Context, src: &str, a: (i64, i64), b: (i64, i64), want: f64) {
    let x = ctx.symbol("x");
    let f = ctx.parse(src).unwrap();
    let big_f = f.integrate(&x);
    assert!(!big_f.has_unevaluated(), "∫ {src} = {big_f}");
    let residual = &big_f.diff(&x) - &f;
    for (p, q) in [(3, 10), (17, 10), (-22, 10), (41, 10), (-53, 10)] {
        let r = residual.subs(&x, &ctx.rational(p, q)).eval_f64().unwrap();
        assert!(
            r.abs() < 1e-12,
            "∫ {src} = {big_f}: F′ − f = {r} at {p}/{q}"
        );
    }
    let at = |(p, q): (i64, i64)| big_f.subs(&x, &ctx.rational(p, q)).eval_f64().unwrap();
    let got = at(b) - at(a);
    assert!(
        (got - want).abs() < 1e-12 * want.abs().max(1.0),
        "∫ {src} = {big_f}: F(b) − F(a) = {got}, quadrature {want}"
    );
}

/// The half-angle substitution: `F(b) − F(a)` across `x = ±π, 3π, …`.
/// Before (0.37.0) the differences were, in order: −2.7125201627,
/// 1.8539645202, −1.6745881373, −1.3843170321, 0.1418550182,
/// 0.6287120495, 1.9910744387, −2.7774818768 (each off by a multiple of
/// the jump).  mpmath: `quad(lambda t: 1/(2+cos(t)), linspace(3, 4, 5))`
/// = 0.91507856576039473592, `… [-7.3, 9.1]` = 9.1091619771240166991,
/// `1/(2+sin(t))` on `[2, 5]` = 1.953010591127960783, `1/(3+2*cos(t))` on
/// `[-4, 4]` = 4.2355347527179603247, `cos(t)/(2+cos(t))` on `[3, 4]` =
/// −0.83015713152078947183, `1/(1+sin(t)**2)` on `[0, 7]` =
/// 5.0715949876187973071, `1/(5+4*sin(t))` on `[-3, 9]` =
/// 4.0854695410831587193, `1/(1/sin(t)+2)` on `[5.85, 9.82]` =
/// 0.36411077675380943613.
#[test]
fn half_angle_antiderivatives_are_continuous_at_odd_multiples_of_pi() {
    let ctx = Context::new();
    for (src, a, b, want) in [
        ("1/(2+cos(x))", (3, 1), (4, 1), 0.915_078_565_760_394_7),
        ("1/(2+cos(x))", (-73, 10), (91, 10), 9.109_161_977_124_017),
        ("1/(2+sin(x))", (2, 1), (5, 1), 1.953_010_591_127_960_8),
        ("1/(3+2*cos(x))", (-4, 1), (4, 1), 4.235_534_752_717_96),
        (
            "cos(x)/(2+cos(x))",
            (3, 1),
            (4, 1),
            -0.830_157_131_520_789_5,
        ),
        ("1/(1+sin(x)^2)", (0, 1), (7, 1), 5.071_594_987_618_797),
        ("1/(5+4*sin(x))", (-3, 1), (9, 1), 4.085_469_541_083_159),
        (
            "1/(1/sin(x)+2)",
            (585, 100),
            (982, 100),
            0.364_110_776_753_809_44,
        ),
    ] {
        check_continuous(&ctx, src, a, b, want);
    }
}

/// `u = tan w` through a quotient that depends on `x` only through `u`.
/// Before: 0.8586041695 (`… + 1/5·atan(tan x)`), −0.3779188467 and
/// −0.0878045910.  mpmath: `quad(lambda t: tan(t)/(2+tan(t)), linspace(1,
/// 2, 5))` = 1.4869227002247644533, `tan(t)**2/(tan(t)**2+2)` on `[0, 5]` =
/// 2.2246617224100213934, `tan(3*t+1)**2/(tan(3*t+1)**2+3)` on `[-2, 2]` =
/// 1.4453990350374264767.
#[test]
fn tangent_substitution_antiderivatives_are_continuous() {
    let ctx = Context::new();
    for (src, a, b, want) in [
        ("tan(x)/(2+tan(x))", (1, 1), (2, 1), 1.486_922_700_224_764_5),
        (
            "tan(x)^2/(tan(x)^2+2)",
            (0, 1),
            (5, 1),
            2.224_661_722_410_021_4,
        ),
        (
            "tan(3*x+1)^2/(tan(3*x+1)^2+3)",
            (-2, 1),
            (2, 1),
            1.445_399_035_037_426_5,
        ),
    ] {
        check_continuous(&ctx, src, a, b, want);
    }
}

/// The forms: SymPy 1.14 gives `x/5 - 2*log(tan(x) + 2)/5 +
/// log(tan(x)**2 + 1)/5` for the first (the `atan(tan x)` and the
/// correction cancel to `x/5`), and `x - 4*sqrt(3)*(atan(sqrt(3)*tan(x/2)/3)
/// + pi*floor((x/2 - pi/2)/pi))/3` for the third, the same up to a
/// constant (`⌊(x/2 − π/2)/π⌋ = ⌊x/(2π) + 1/2⌋ − 1`).
#[test]
fn continuous_forms() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (src, want) in [
        (
            "tan(x)/(tan(x) + 2)",
            "1/5*x - 2/5*ln(abs(tan(x) + 2)) + 1/5*ln(tan(x)^2 + 1)",
        ),
        (
            "1/(2 + cos(x))",
            "2/3*sqrt(3)*atan(1/3*sqrt(3)*tan(1/2*x)) + 2/3*sqrt(3)*floor(x/(2*pi) + 1/2)*pi",
        ),
        (
            "cos(x)/(2 + cos(x))",
            "x - 4/3*sqrt(3)*atan(1/3*sqrt(3)*tan(1/2*x)) - 4/3*sqrt(3)*floor(x/(2*pi) + 1/2)*pi",
        ),
        // Not integrable at x = π: nothing to make continuous there.
        ("1/(1 + cos(x))", "tan(1/2*x)"),
    ] {
        let got = ctx.parse(src).unwrap().integrate(&x);
        assert_eq!(got.to_string(), want, "∫ {src}");
    }
}

/// Definite integrals over the poles, from the antiderivatives with floor
/// terms (the steps are split like the poles of `tan`).  mpmath:
/// `quad(lambda t: 1/(2+cos(t)), linspace(0, 4, 17))` =
/// 2.5877551032590471531; `2π/√3` = 3.6275987284684357012 over `[0, 2π]`
/// and over `[π, 3π]`; `π/10 + 2/5·ln 2` = 0.59141813758295753 for
/// `tan x/(2 + tan x)` over `[0, π/2]`.
#[test]
fn definite_integrals_across_the_poles() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = ctx.pi();
    let f = ctx.parse("1/(2+cos(x))").unwrap();
    for (lo, hi, want) in [
        (ctx.int(0), ctx.int(4), 2.587_755_103_259_047),
        (ctx.int(0), &ctx.int(2) * &pi, 3.627_598_728_468_435_7),
        (pi.clone(), &ctx.int(3) * &pi, 3.627_598_728_468_435_7),
    ] {
        let v = f.integrate_definite(&x, &lo, &hi);
        let got = v.eval_f64().unwrap();
        assert!((got - want).abs() < 1e-12, "∫[{lo}, {hi}] = {v} = {got}");
    }
    let g = ctx.parse("tan(x)/(2+tan(x))").unwrap();
    let v = g.integrate_definite(&x, &ctx.int(0), &(&pi / 2));
    let got = v.eval_f64().unwrap();
    assert!((got - 0.591_418_137_582_957_5).abs() < 1e-12, "{v} = {got}");
}

/// The rational integrator's `RootSum` over an algebraic factor of the
/// Rothstein–Trager resultant whose log argument has degree 2 in `x`:
/// `Σ α·ln(x² + B(α)·x − 1)` crosses the cut of `ln` at `x = 0` for the
/// non-real `α`, and jumped there.  Before: `∫ (x² + 1)³/((x² + 1)⁴ + 32x⁴)`
/// gave `F(1) − F(−1)` = −1.0599579527 and `integrate_definite` over
/// `[0, 1]` −0.5299789763; `∫ dx/(1 + sin⁴x)` (the same through `tan(x/2)`)
/// gave `F(1) − F(−1)` = −3.0745610724.  Now the sum runs over the roots
/// `ρ` of the denominator's factor with `ln(x − ρ)`, and the jump at
/// `tan(x/2) = ±∞` is `iπ·Σ r(ρ)·sign(Im ρ)`.  mpmath:
/// `quad(lambda t: (t**2+1)**3/((t**2+1)**4+32*t**4), linspace(-1, 1, 9))`
/// = 1.059957952671974463, over `linspace(0, 1, 5)` =
/// 0.52997897633598723148; `1/(1+sin(t)**4)` over `linspace(-1, 1, 9)` =
/// 1.8067638297871783996 and over `linspace(0, 4, 17)` =
/// 3.2437016599870133451; `1/(2+cos(t)**6)` over `linspace(-2, 5, 29)` =
/// 3.1352780376235362144.
#[test]
fn root_sums_are_continuous() {
    let ctx = Context::new();
    for (src, a, b, want) in [
        (
            "(x^2+1)^3/((x^2+1)^4 + 32*x^4)",
            (-1, 1),
            (1, 1),
            1.059_957_952_671_974_5,
        ),
        ("1/(1+sin(x)^4)", (-1, 1), (1, 1), 1.806_763_829_787_178_4),
        ("1/(1+sin(x)^4)", (0, 1), (4, 1), 3.243_701_659_987_013),
        ("1/(2+cos(x)^6)", (-2, 1), (5, 1), 3.135_278_037_623_536_2),
    ] {
        check_continuous(&ctx, src, a, b, want);
    }
    let x = ctx.symbol("x");
    let f = ctx.parse("(x^2+1)^3/((x^2+1)^4 + 32*x^4)").unwrap();
    let v = f.integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    let got = v.eval_f64().unwrap();
    assert!((got - 0.529_978_976_335_987_2).abs() < 1e-12, "{v} = {got}");
}

/// With parameters the jump `G(+∞) − G(−∞)` depends on their values (on
/// `a > |b|` for `∫ dx/(a + b·cos x)`, and on the sides of the cuts its
/// logarithms approach for complex ones): the answer keeps the form
/// continuous between the poles of `tan(x/2)`, as SymPy's and Rubi's do,
/// and is still an antiderivative.
#[test]
fn parametric_answers_stay_antiderivatives() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in ["1/(a + b*cos(x))", "tan(x)/(a + tan(x))"] {
        let f = ctx.parse(src).unwrap();
        let big_f = f.integrate(&x);
        assert!(!big_f.has_unevaluated(), "∫ {src} = {big_f}");
        let residual = (&big_f.diff(&x) - &f)
            .subs(&ctx.symbol("a"), &ctx.int(3))
            .subs(&ctx.symbol("b"), &ctx.int(2))
            .subs(&x, &ctx.rational(7, 10));
        let r = residual.eval_complex64().unwrap();
        assert!(r.norm() < 1e-12, "∫ {src} = {big_f}: F′ − f = {r}");
    }
}
