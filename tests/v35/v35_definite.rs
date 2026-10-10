//! Definite integrals after the 0.39 hunt (parametric and improper
//! integrals).  Every test names what was wrong before and the oracle of
//! its reference value (mpmath 1.3.0 `quad`/`quadosc` at `mp.dps = 30`,
//! or SymPy 1.14).

use symplex::prelude::*;

fn value(e: &Ex) -> f64 {
    e.eval_f64().unwrap_or_else(|err| panic!("{e}: {err}"))
}

/// Up to 0.39 the `[0, ∞)` table returned `(π/2)·sign(b)` for
/// `∫₀^∞ sin(bx)/x dx` and `π|b|/2` for `∫₀^∞ sin²(bx)/x² dx` with `b` not
/// declared real: for `b = 1 + i` both integrals diverge (`|sin(bx)|` grows
/// like `e^x/2`).  An unassumed parameter may be complex (D4), so the table
/// now asks for a real `b`.
#[test]
fn dirichlet_table_needs_a_real_coefficient() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let b = ctx.symbol("b");
    let (zero, inf) = (ctx.int(0), ctx.infinity());
    let sinc = (&b * &x).sin() / &x;
    assert!(sinc.try_integrate_definite(&x, &zero, &inf).is_err());
    let sinc2 = (&b * &x).sin().powi(2) / x.powi(2);
    assert!(sinc2.try_integrate_definite(&x, &zero, &inf).is_err());
    let full = sinc.try_integrate_definite(&x, &ctx.neg_infinity(), &inf);
    assert!(full.is_err(), "{full:?}");

    // Declared real: the classical values.  mpmath:
    // quad(sin(-3x)/x, [0,1]) + quadosc(…, [1, inf], omega=3) = -1.5707963…
    let r = ctx.symbol_with("r", &[Assumption::Real]).unwrap();
    let v = ((&r * &x).sin() / &x)
        .try_integrate_definite(&x, &zero, &inf)
        .unwrap();
    assert!((value(&v.subs(&r, &ctx.int(-3))) + std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    let v = ((&r * &x).sin().powi(2) / x.powi(2))
        .try_integrate_definite(&x, &zero, &inf)
        .unwrap();
    // π·|−3|/2 = 4.71238898…
    assert!((value(&v.subs(&r, &ctx.int(-3))) - 3.0 * std::f64::consts::FRAC_PI_2).abs() < 1e-12);
}

/// `∫₀^∞ e^{−ax} sin(bx) dx = b/(a² + b²)` and `∫₀^∞ e^{−ax} sin(bx)/x dx =
/// atan(b/a)` hold for `a > |Im b|`; up to 0.39 they were returned for
/// `a > 0` and `b` not declared real (at `a = 3/10`, `b = 7/10 + 2i/5` the
/// integrand grows like `e^{x/10}`).  mpmath: quad(exp(-x/2)*sin(2x), [0,
/// inf]) = 0.47058823529411764706 = 8/17.
#[test]
fn damped_sine_table_needs_a_real_frequency() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();
    let b = ctx.symbol("b");
    let (zero, inf) = (ctx.int(0), ctx.infinity());
    let damped = (-(&a * &x)).exp() * (&b * &x).sin();
    assert!(damped.try_integrate_definite(&x, &zero, &inf).is_err());
    let damped_over_x = &damped / &x;
    assert!(
        damped_over_x
            .try_integrate_definite(&x, &zero, &inf)
            .is_err()
    );

    let w = ctx.symbol_with("w", &[Assumption::Real]).unwrap();
    let v = ((-(&a * &x)).exp() * (&w * &x).sin())
        .try_integrate_definite(&x, &zero, &inf)
        .unwrap();
    let at = v.subs(&a, &ctx.rational(1, 2)).subs(&w, &ctx.int(2));
    assert!((value(&at) - 8.0 / 17.0).abs() < 1e-14, "{v}");
}

/// `∫ dx/(a + tan x)` (`a` not declared real) is made continuous across
/// the poles of `tan x` by `J·⌊x/π + 1/2⌋` (0.38).  `J` was taken as for a
/// real `a`, `−iπ/(a² + 1)`; for `Im a < 0`, where `a + tan x` passes the
/// cut of `ln` from below, it is `+iπ/(a² + 1)`, so the antiderivative
/// jumped by `2πi/(a² + 1)` at every pole and `F(−4.1) − F(−7.3)` was
/// `12.54 + 6.31i`.  mpmath: quad(1/(a + tan x), linspace(-7.3, -4.1, 12))
/// = 0.11508933873189061 ± 1.5043964572076968i at a = 1/5 ∓ 11i/10.
#[test]
fn antiderivative_through_tan_is_continuous_at_complex_parameters() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let big_f = (1 / (&a + x.tan())).integrate(&x);
    assert!(!big_f.has_unevaluated(), "{big_f}");
    let i = ctx.i_unit();
    for (im_sign, expected_im) in [(-1, 1.504_396_457_207_696_8), (1, -1.504_396_457_207_696_8)] {
        let a_val = ctx.rational(1, 5) + ctx.rational(11 * im_sign, 10) * &i;
        let at = big_f.subs(&a, &a_val);
        let hi = at
            .subs(&x, &ctx.rational(-41, 10))
            .eval_complex64()
            .unwrap();
        let lo = at
            .subs(&x, &ctx.rational(-73, 10))
            .eval_complex64()
            .unwrap();
        let d = hi - lo;
        assert!(
            (d - Complex64::new(0.115_089_338_731_890_6, expected_im)).norm() < 1e-10,
            "a = {a_val}: F(-4.1) - F(-7.3) = {d}, F = {big_f}"
        );
    }
}

/// `∫ dx/√(x² + a)` through `t = x/√(x² + a)` was `(ln(t + 1) −
/// ln(t − 1))/2`, whose `ln(t − 1)` sits on the cut at `x = 0` (`t = 0`):
/// for every non-real `a`, and for real `a < 0` (the integrand is imaginary
/// on `|x| < √−a`), the antiderivative jumped by `±iπ` there, and so
/// did `F(3/10) − F(−9/10)` against the integral.  mpmath:
/// quad(1/sqrt(x² + a), linspace(-0.9, 0.3, 8)) = 0.68145083495234395 −
/// 0.38711492596526130i at a = 1 + 2i, −0.79643859071671492i at a = −5/2.
#[test]
fn inverse_square_root_antiderivative_has_no_jump_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let big_f = (1 / (x.powi(2) + &a).sqrt()).integrate(&x);
    assert!(!big_f.has_unevaluated(), "{big_f}");
    let cases = [
        (
            1 + 2 * ctx.i_unit(),
            Complex64::new(0.681_450_834_952_343_9, -0.387_114_925_965_261_3),
        ),
        (
            ctx.rational(-5, 2),
            Complex64::new(0.0, -0.796_438_590_716_714_9),
        ),
    ];
    for (a_val, expected) in cases {
        let at = big_f.subs(&a, &a_val);
        let hi = at.subs(&x, &ctx.rational(3, 10)).eval_complex64().unwrap();
        let lo = at.subs(&x, &ctx.rational(-9, 10)).eval_complex64().unwrap();
        assert!(
            (hi - lo - expected).norm() < 1e-12,
            "a = {a_val}: {} vs {expected}; F = {big_f}",
            hi - lo
        );
    }
}

/// Refused before: `∫ dx/(b + 1 + cos x)` with `b > 0` over a period.  The
/// table entry for `1/(a + c·cos x)` wanted a two-term denominator (here
/// the constant part is the sum `b + 1`), and the breakpoint scan's zeros
/// `acos(−b − 1)` (non-real for `b > 0`) were "undecided" relative to the
/// interval.  mpmath at b = 1: quad(1/(2 + cos x), [0, pi]) = π/√3 =
/// 1.8137993642342178506, quad(1/(2 + sin x), [0, 2pi]) = 2π/√3,
/// quad(1/(2 + cos x), [0, 3]) = 1.6726765374986524172.
#[test]
fn cosine_denominators_with_a_parametric_constant_part() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let b = ctx.symbol_with("b", &[Assumption::Positive]).unwrap();
    let one = ctx.int(1);
    let pi = ctx.pi();
    let zero = ctx.int(0);
    let cases = [
        (1 / (&b + 1 + x.cos()), pi.clone(), 1.813_799_364_234_217_8),
        (1 / (&b + 1 + x.sin()), 2 * &pi, 3.627_598_728_468_435_7),
        (1 / (&b + 1 + x.cos()), ctx.int(3), 1.672_676_537_498_652_4),
    ];
    for (f, hi, expected) in cases {
        let v = f
            .try_integrate_definite(&x, &zero, &hi)
            .unwrap_or_else(|e| panic!("∫ {f}: {e}"));
        let at = value(&v.subs(&b, &one));
        assert!((at - expected).abs() < 1e-12, "∫₀^{hi} {f} = {v} = {at}");
    }
}

/// The steps of `floor`/`ceiling` of a non-linear argument were not
/// located, and with no `|·|`/sign/H argument to watch the incomplete scan
/// passed: the single piece was resolved at its midpoint.  `∫₀² a·floor(x²)
/// dx` was `2a` (it is `(5 − √2 − √3)·a`, mpmath quad with breakpoints 1,
/// √2, √3: 1.8537...), `∫₀² floor(x²)e^{−ax} dx` was `(1 − e^{−2a})/a`, and
/// `∫₀^∞ floor(x)e^{−x} dx` (no finite range to place the steps in) was 1
/// (it is `1/(e − 1)` = 0.58197670686932642439, mpmath nsum).  Now they are
/// refused; linear arguments on finite intervals still work.
#[test]
fn floor_steps_must_all_be_located() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let (zero, two) = (ctx.int(0), ctx.int(2));
    let f = &a * x.powi(2).floor();
    assert!(f.try_integrate_definite(&x, &zero, &two).is_err());
    let g = x.powi(2).floor() * (-(&a * &x)).exp();
    assert!(g.try_integrate_definite(&x, &zero, &two).is_err());
    let h = x.floor() * (-&x).exp();
    assert!(
        h.try_integrate_definite(&x, &zero, &ctx.infinity())
            .is_err()
    );
    let c = x.powi(2).ceiling();
    assert!(
        c.try_integrate_definite(&x, &zero, &ctx.rational(3, 2))
            .is_err()
    );

    // Linear arguments: ∫₀^{7/2} floor(x) dx = 0 + 1 + 2 + 3/2 = 9/2,
    // ∫_{−2}^{3/2} x·floor(x) dx = 33/8 (exact sums of the pieces).
    let v = x
        .floor()
        .try_integrate_definite(&x, &zero, &ctx.rational(7, 2))
        .unwrap();
    assert_eq!(v, ctx.rational(9, 2));
    let v = (&x * x.floor())
        .try_integrate_definite(&x, &ctx.int(-2), &ctx.rational(3, 2))
        .unwrap();
    assert_eq!(v, ctx.rational(33, 8));
}
