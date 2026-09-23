//! 0.23 — antiderivatives found by `fuzz_integrate` after the domain-model
//! change: flattening `(g^m)^n → g^(m·n)` in the integrator assumed the real
//! root / a real base, so `∫ √x·√(1/x) dx` was `x` (the integrand is `−1` at
//! every `x < 0`).  Each case must come back either unevaluated or with
//! `F′ = f` on both sides of 0; nothing here is compared with a printed
//! form.  The values at `x = −5/7` are the principal branch (mpmath:
//! `sqrt(mpf(-5)/7)*sqrt(-mpf(7)/5) = -1`).

use symplex::prelude::*;

/// `∫ f` is unevaluated, or its derivative equals `f` at `x = −5/7` and `7/5`.
fn never_wrong(f: &Ex, x: &Ex) {
    let big_f = f.integrate(x);
    if big_f.has_unevaluated() {
        return;
    }
    let df = big_f.diff(x);
    let ctx = x.context();
    for v in [ctx.rational(-5, 7), ctx.rational(7, 5)] {
        let (a, b) = (
            f.subs(x, &v).eval_complex64().unwrap(),
            df.subs(x, &v).eval_complex64().unwrap(),
        );
        assert!(
            (a - b).norm() < 1e-10 * a.norm().max(1.0),
            "∫ {f} dx = {big_f}, but F′ ≠ f at x = {v}: {a} vs {b}"
        );
    }
}

#[test]
fn sqrt_x_times_sqrt_of_reciprocal() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sqrt() * (1 / &x).sqrt();
    let v = f.subs(&x, &ctx.rational(-5, 7)).eval_complex64().unwrap();
    assert!((v - Complex64::new(-1.0, 0.0)).norm() < 1e-15, "{v}");
    never_wrong(&f, &x);
}

#[test]
fn sqrt_of_cube_over_sqrt() {
    // √(x³)/√x is |x| on the reals (x at x > 0, −x at x < 0).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    never_wrong(&(x.powi(3).sqrt() / x.sqrt()), &x);
}

#[test]
fn constant_over_sqrt_of_reciprocal() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let c = (ctx.int(2).sqrt() * ctx.i_unit()).sin();
    never_wrong(&(c / (1 / &x).sqrt()), &x);
}

#[test]
fn cos_sqrt_over_sqrt_of_inverse_square() {
    // 1/√(x⁻²) = |x| on the reals; the x = s² substitution used to lose it.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    never_wrong(&(x.sqrt().cos() / x.powi(-2).sqrt()), &x);
}

#[test]
fn flattening_still_serves_the_standard_forms() {
    // 1/√(x²+1) = ((x²+1)^(1/2))^(−1) must still flatten to (x²+1)^(−1/2):
    // SymPy: integrate(1/sqrt(x**2 + 1), x) = asinh(x).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = 1 / (x.powi(2) + 1).sqrt();
    let big_f = f.integrate(&x);
    assert!(!big_f.has_unevaluated(), "{big_f}");
    never_wrong(&f, &x);
    // And √(x²) is |x| for the real integration variable: ∫ √(x²) dx = x|x|/2.
    let g = x.powi(2).sqrt();
    let big_g = g.integrate(&x);
    assert!(!big_g.has_unevaluated(), "{big_g}");
    never_wrong(&g, &x);
}
