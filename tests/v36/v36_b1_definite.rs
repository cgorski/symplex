//! Tests of the 0.50 refactor (b1): silent wrong numeric definite integrals at extreme scales.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

/// `∫ f dx` over `[lo, hi]` (strings parsed in `ctx`) by `integrate_definite`,
/// evaluated to 20 digits.
fn definite_20(ctx: &Context, f: &str, lo: &str, hi: &str) -> String {
    let x = ctx.symbol("x");
    let (f, lo, hi) = (
        ctx.parse(f).unwrap(),
        ctx.parse(lo).unwrap(),
        ctx.parse(hi).unwrap(),
    );
    let v = f.try_integrate_definite(&x, &lo, &hi).unwrap();
    v.eval_decimal(20).unwrap()
}

/// `|sin x|` on `[10¹⁵, 10¹⁵ + 4]`: the zero `318309886183791·π` lies 100
/// from neither end, but the comparison of points called two values within
/// `10⁻¹³` of each other (relatively) equal, so it was taken for an end and
/// the integral came out as `∫ sin x dx = −cos(10¹⁵) + cos(10¹⁵ + 4) =
/// 1.4982`.  Oracle: mpmath at 60 and 90 digits, `Σ ±(cos l − cos r)` over
/// the pieces between the multiples of `π/2`: `2.4717950667107758792`.
#[test]
fn abs_sin_near_1e15() {
    let ctx = Context::new();
    assert_eq!(
        definite_20(&ctx, "abs(sin(x))", "10^15", "10^15 + 4"),
        "2.4717950667107758792"
    );
}

/// `|sin(10²⁰x)|` on `[1, 1 + 4·10⁻²⁰]`, a single `f64`: the zeros of the
/// sine were enumerated from the `f64` range and could not be told from the
/// ends (`1.7517·10⁻²⁰`).  Oracle: mpmath (as above, scaled by `10⁻²⁰`):
/// `2.2483374313765516096·10⁻²⁰`.
#[test]
fn abs_sin_scaled_on_a_single_f64() {
    let ctx = Context::new();
    assert_eq!(
        definite_20(&ctx, "abs(sin(10^20*x))", "1", "1 + 4*10^(-20)"),
        "2.2483374313765516096e-20"
    );
}

/// `|cos x|` and `sign(sin x)` on `[10¹⁵, 10¹⁵ + 4]` were `1.0309` and
/// `−4` (the kinks inside taken for the ends).  Oracle: mpmath at 60 and 90
/// digits: `2.6856545582872987132` and `−1.9362109269606387189` (=
/// `636619772367582π − 2000000000000004`, exactly).
#[test]
fn abs_cos_and_sign_sin_near_1e15() {
    let ctx = Context::new();
    assert_eq!(
        definite_20(&ctx, "abs(cos(x))", "10^15", "10^15 + 4"),
        "2.6856545582872987132"
    );
    assert_eq!(
        definite_20(&ctx, "sign(sin(x))", "10^15", "10^15 + 4"),
        "-1.9362109269606387189"
    );
}

/// The families are enumerated between the exact bounds at any scale:
/// `floor(sin x)` on `[10³⁰⁰, 10³⁰⁰ + 10]` (its steps, the zeros of `sin`
/// and `cos`, were not located at all: refused) and `|sin x|` on `[10³⁰⁰ +
/// 1/3, 10³⁰⁰ + 1/3 + 7]`.  Oracle: mpmath at 360 and 400 digits:
/// `−4.8814070016538613940` and `4.6078814667733507346`.
#[test]
fn periodic_kinks_near_1e300() {
    let ctx = Context::new();
    assert_eq!(
        definite_20(&ctx, "floor(sin(x))", "10^300", "10^300 + 10"),
        "-4.881407001653861394"
    );
    assert_eq!(
        definite_20(&ctx, "abs(sin(x))", "10^300 + 1/3", "10^300 + 1/3 + 7"),
        "4.6078814667733507346"
    );
}

/// `integrate_numeric` of `|sin x|` on `[10²⁰, 10²⁰ + 4]`: both bounds
/// round to the same `f64`, and the `f64` quadrature returned `Ok(0.0)`.
/// Now the arbitrary-precision quadrature takes the bounds as they are.
/// Oracle: mpmath at 60 and 90 digits: `2.2483374313765516096`.
#[test]
fn integrate_numeric_bounds_with_one_f64_value() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin().abs();
    let lo = ctx.parse("10^20").unwrap();
    let hi = ctx.parse("10^20 + 4").unwrap();
    let v = f.integrate_numeric(&x, &lo, &hi).unwrap();
    assert!((v - 2.248_337_431_376_551_6).abs() < 1e-14, "{v}");
}

/// `integrate_numeric` on a range of a few thousand ulps: at `10¹⁶` the
/// `f64` nodes of `|sin(x/3 + 1/2)|` on `[10¹⁶, 10¹⁶ + 4]` lie on a grid of
/// spacing 2, K15 and G7 agreed on the few values they saw, and `3.0537`
/// came back as converged (from `eval_f64` of the unevaluated integral too).
/// Oracle: mpmath at 57 and 72 digits: `2.8325014383146011`.
#[test]
fn quadrature_on_a_range_of_few_f64_nodes() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("abs(sin(x/3 + 1/2))").unwrap();
    let lo = ctx.parse("10^16").unwrap();
    let hi = ctx.parse("10^16 + 4").unwrap();
    let v = f.integrate_numeric(&x, &lo, &hi).unwrap();
    assert!((v - 2.832_501_438_314_601).abs() < 1e-12, "{v}");
    let node = f.definite_integral_node(&x, &lo, &hi);
    let w = node.eval_f64().unwrap();
    assert!((w - 2.832_501_438_314_601).abs() < 1e-12, "{w}");
}

/// `Γ(9·10¹⁸x)` on `[−1, 1]` has 1.8·10¹⁹ poles: the symbolic side refused
/// it (0.49), `integrate_numeric` and `eval_f64` of the unevaluated integral
/// spent minutes in the arbitrary-precision quadrature (over 4 minutes).
/// They refuse at once now (the integral diverges: `Γ` has simple poles).
#[test]
fn gamma_of_a_huge_multiple_refuses_fast() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("gamma(9*10^18*x)").unwrap();
    let (lo, hi) = (ctx.int(-1), ctx.int(1));
    let start = std::time::Instant::now();
    assert!(f.integrate_numeric(&x, &lo, &hi).is_err());
    assert!(f.definite_integral_node(&x, &lo, &hi).eval_f64().is_err());
    assert!(start.elapsed().as_secs() < 30, "{:?}", start.elapsed());
}

/// The kinks of `sign(cos x² − 1/7)` on `[0, 3]`: the principal solve gives
/// `±√acos(1/7)` only, its list counted as every zero, and the quadrature
/// behind `eval_f64` ran across the other four (`0.5178`).  Oracle: mpmath
/// `quad` between the zeros (found by bisection at 30 and 45 digits):
/// `0.53597626723185112`.
#[test]
fn sign_of_a_periodic_composition() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("sign(cos(x^2) - 1/7)").unwrap();
    let node = f.definite_integral_node(&x, &ctx.int(0), &ctx.int(3));
    let v = node.eval_f64().unwrap();
    assert!((v - 0.535_976_267_231_851_1).abs() < 1e-12, "{v}");
    let w = f.integrate_numeric(&x, &ctx.int(0), &ctx.int(3)).unwrap();
    assert!((w - 0.535_976_267_231_851_1).abs() < 1e-9, "{w}");
}

/// Steps of `floor` of a non-linear argument, which the scan cannot place,
/// are located numerically: `eval_f64` of `∫₋₂² floor(x³ − x − 10/21) dx`
/// was `−4` (the quadrature ran across them).  Oracle: mpmath `quad`
/// between the crossings of the integers (bisection at 30 and 45 digits):
/// `−3.9534048726733656`.
#[test]
fn floor_of_a_cubic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("floor(x^3 - x - 1/3 - 1/7)").unwrap();
    let node = f.definite_integral_node(&x, &ctx.int(-2), &ctx.int(2));
    let v = node.eval_f64().unwrap();
    assert!((v + 3.953_404_872_673_365_6).abs() < 1e-12, "{v}");
}

/// `sign(re((x + i)²) − 1)` on `[−3, 3]` (`x² − 2` for real `x`): `eval_f64`
/// of the unevaluated integral was `0` (true `6 − 4√2 = 0.34314575…`;
/// exact identity).  The S3 side note `sign(|x + i|² − 4)` (`−0.9375` in
/// 0.48, `6 − 4√3`) is right since 0.49 and stays so.
#[test]
fn sign_of_complex_looking_arguments() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (lo, hi) = (ctx.int(-3), ctx.int(3));
    let f = ctx.parse("sign(re((x + I)^2) - 1)").unwrap();
    let v = f.definite_integral_node(&x, &lo, &hi).eval_f64().unwrap();
    assert!((v - (6.0 - 4.0 * 2f64.sqrt())).abs() < 1e-12, "{v}");
    let g = ctx.parse("sign(abs(x + I)^2 - 4)").unwrap();
    let w = g.definite_integral_node(&x, &lo, &hi).eval_f64().unwrap();
    assert!((w - (6.0 - 4.0 * 3f64.sqrt())).abs() < 1e-12, "{w}");
}

/// `eval_f64` of `∫_{100.7}^{130.7} (x − floor x) dx` was `21` (the `f64`
/// quadrature's estimate across the 30 steps passed for a bound); it is 15
/// (30 unit pieces of mean 1/2: exact).
#[test]
fn fractional_part_across_steps() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("x - floor(x)").unwrap();
    let lo = ctx.parse("100 + 7/10").unwrap();
    let hi = ctx.parse("130 + 7/10").unwrap();
    let v = f.definite_integral_node(&x, &lo, &hi).eval_f64().unwrap();
    assert!((v - 15.0).abs() < 1e-12, "{v}");
}

/// The kinks of `max`: `Max(x − 1/3, 0)` was never split at `1/3` (no
/// breakpoint was pushed for `min`/`max`).  `∫₀¹ = 2/9` (exact).
#[test]
fn max_is_split_at_its_kink() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("Max(x - 1/3, 0)").unwrap();
    let v = f.integrate_numeric(&x, &ctx.int(0), &ctx.int(1)).unwrap();
    assert!((v - 2.0 / 9.0).abs() < 1e-14, "{v}");
}
