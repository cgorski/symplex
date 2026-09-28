//! Rational integrals over the roots of `xⁿ + 1`: the logarithmic part in
//! exact cyclotomic arithmetic (de Moivre), compact answers.

// Reference values are quoted at the digits the oracle printed them.
#![allow(clippy::excessive_precision)]

use symplex::prelude::*;

fn p(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap_or_else(|e| panic!("{s}: {e}"))
}

/// `F′ = f` at real sample points and `F(1) − F(0)` to 15 digits.
fn assert_integral(ctx: &Context, f: &Ex, big_f: &Ex, x: &Ex, definite: f64) {
    assert!(
        !big_f.has_unevaluated(),
        "∫ {f} stayed unevaluated: {big_f}"
    );
    let residual = big_f.diff(x) - f.clone();
    for pt in ["1/3", "7/5", "-5/7", "13/4", "-13/4"] {
        let v = residual
            .subs(x, &p(ctx, pt))
            .eval_complex64()
            .unwrap_or_else(|e| panic!("∫ {f} = {big_f}: residual at {pt}: {e}"));
        assert!(v.norm() < 1e-12, "∫ {f} = {big_f}: F′ − f = {v} at {pt}");
    }
    let d = big_f.subs(x, &ctx.int(1)) - big_f.subs(x, &ctx.int(0));
    let got: f64 = d
        .eval_decimal(25)
        .unwrap_or_else(|e| panic!("{big_f} on [0, 1]: {e}"))
        .parse()
        .unwrap();
    assert!(
        (got - definite).abs() < 1e-15,
        "∫₀¹ {f} = {got}, want {definite}"
    );
}

/// `∫ x⁶/(x⁸ + 1) dx` was correct but 18 KB of text (`x⁴`: 5 KB, `x²`:
/// 1.5 KB): the coefficients of `log_to_real` were the solver's roots
/// `cos((2j+1)π/8) + i·sin((2j+1)π/8)` raised to powers and expanded,
/// de Moivre not applied (`sin²(π/8) + cos²(π/8)` stayed in `∫ 1/(x⁸+1)`).
/// Now every coefficient is computed exactly in `ℚ(ζ₁₆)` and written over
/// the basis `cos(kπ/8)` of its real subfield: about 400 characters each.
///
/// mpmath (dps 30): `quad(lambda t: t**k/(t**8 + 1), [0, 1])` →
/// `0.924651705775538023660718592282` (k = 0),
/// `0.273898219208518613194702791785` (k = 2),
/// `0.15115620388416600496267336442` (k = 4),
/// `0.101520447201492865210749188591` (k = 6).  (SymPy 1.14 returns `0` for
/// these integrals — a SymPy bug.)
#[test]
fn integrals_over_the_roots_of_x8_plus_1_are_compact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (k, want) in [
        (0, 0.924651705775538023660718592282),
        (2, 0.273898219208518613194702791785),
        (4, 0.15115620388416600496267336442),
        (6, 0.101520447201492865210749188591),
    ] {
        let f = p(&ctx, &format!("x^{k}/(x^8 + 1)"));
        let big_f = f.integrate(&x);
        let shown = big_f.to_string();
        assert!(shown.len() < 600, "∫ {f}: {} bytes: {shown}", shown.len());
        assert!(
            !shown.contains("^2*"),
            "unreduced powers of cos/sin: {shown}"
        );
        assert_integral(&ctx, &f, &big_f, &x, want);
    }
}

/// The residues of `xᵏ/(x⁵ + 1)` lie in `ℚ(ζ₁₀)`: the answer is written with
/// `cos(π/5)` (or `√5` where that is smaller) instead of the quartic
/// formula's `√(1 − (−1/4·√5 − 1/4)²)` (about 490 → 245 characters).
///
/// mpmath (dps 30): `quad(lambda t: 1/(t**5 + 1), [0, 1])` →
/// `0.88831357265178863804075522702`.
#[test]
fn integral_over_the_roots_of_x5_plus_1_uses_cos_pi_over_5() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = p(&ctx, "1/(x^5 + 1)");
    let big_f = f.integrate(&x);
    let shown = big_f.to_string();
    assert!(shown.len() < 300, "{} bytes: {shown}", shown.len());
    assert!(shown.contains("cos(1/5*pi)"), "{shown}");
    assert_integral(&ctx, &f, &big_f, &x, 0.88831357265178863804075522702);
}

/// With `xⁿ + 2` the residues are `ρ·ζ` with `ρ` a radical (`2^{1/8}/8·…`):
/// `∫ x⁶/(x⁸ + 2)` printed 25 KB and `∫ x⁸/(x¹⁰ + 2)` 103 KB (the solver's
/// roots expanded).  The coefficients are now computed in `ℚ(ζ)[ρ]`,
/// `ρ^L` rational, and written `Σ ρ^r·(…)` over the cosines.
///
/// mpmath (dps 30): `quad(lambda t: t**6/(t**8 + 2), [0, 1])` →
/// `0.0587484348358736923856811429413`, `quad(lambda t: t**8/(t**10 + 2),
/// [0, 1])` → `0.0455573002779939895611668811952`.
#[test]
fn integrals_over_the_roots_of_xn_plus_2_are_compact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (src, want) in [
        ("x^6/(x^8 + 2)", 0.0587484348358736923856811429413),
        ("x^8/(x^10 + 2)", 0.0455573002779939895611668811952),
    ] {
        let f = p(&ctx, src);
        let big_f = f.integrate(&x);
        let shown = big_f.to_string();
        assert!(shown.len() < 1_500, "∫ {f}: {} bytes", shown.len());
        assert_integral(&ctx, &f, &big_f, &x, want);
    }
}

/// `∫ 1/(x⁶ + 1)` and `∫ x/(x¹² + 1)` stayed unevaluated: the log part of
/// the quartic factor had no exact real/imaginary split of its Ferrari
/// roots.  In `ℚ(ζ₁₂)` it has.
///
/// SymPy: `integrate(1/(x**6 + 1), x)` → `-sqrt(3)*log(x**2 - sqrt(3)*x +
/// 1)/12 + sqrt(3)*log(x**2 + sqrt(3)*x + 1)/12 + atan(x)/3 + atan(2*x -
/// sqrt(3))/6 + atan(2*x + sqrt(3))/6`; mpmath (dps 30):
/// `quad(lambda t: 1/(t**6 + 1), [0, 1])` → `0.903771773748772046842654357987`,
/// `quad(lambda t: t/(t**12 + 1), [0, 1])` → `0.451885886874386023421327178993`.
#[test]
fn integrals_over_the_roots_of_x6_plus_1_evaluate() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = p(&ctx, "1/(x^6 + 1)");
    let big_f = f.integrate(&x);
    assert_eq!(
        big_f.to_string(),
        "-1/12*sqrt(3)*ln(x^2 - x*sqrt(3) + 1) + 1/12*sqrt(3)*ln(x^2 + x*sqrt(3) + 1) + 1/3*atan(x) + 1/6*atan(2*x + sqrt(3)) + 1/6*atan(2*x - sqrt(3))"
    );
    assert_integral(&ctx, &f, &big_f, &x, 0.903771773748772046842654357987);
    let f = p(&ctx, "x/(x^12 + 1)");
    let big_f = f.integrate(&x);
    assert_integral(&ctx, &f, &big_f, &x, 0.451885886874386023421327178993);
}
