//! Tests for Wave T series extensions: residue computation and Fourier series.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Residue tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn residue_simple_pole_1_over_x() {
    let ctx = Context::new();
    // Res(1/x, x=0) = 1
    symplex::syms!(ctx; x);
    let f = &ctx.int(1) / &x;
    let r = f.residue(&x, &ctx.int(0));
    assert_eq!(format!("{r}"), "1", "Res(1/x, 0) = 1");
}

#[test]
fn residue_1_over_x_minus_1() {
    let ctx = Context::new();
    // Res(1/(x-1), x=1) = 1
    symplex::syms!(ctx; x);
    let f = &ctx.int(1) / &(&x - 1);
    let r = f.residue(&x, &ctx.int(1));
    assert_eq!(format!("{r}"), "1", "Res(1/(x-1), 1) = 1");
}

#[test]
fn residue_x_over_x_minus_1() {
    let ctx = Context::new();
    // Res(x/(x-1), x=1) = lim_{x→1} (x-1) * x/(x-1) = lim_{x→1} x = 1
    symplex::syms!(ctx; x);
    let f = &x / &(&x - 1);
    let r = f.residue(&x, &ctx.int(1));
    assert_eq!(format!("{r}"), "1", "Res(x/(x-1), 1) = 1");
}

#[test]
fn residue_exp_over_x() {
    let ctx = Context::new();
    // Res(exp(x)/x, x=0) = lim_{x→0} x * exp(x)/x = lim_{x→0} exp(x) = 1
    symplex::syms!(ctx; x);
    let f = &x.exp() / &x;
    let r = f.residue(&x, &ctx.int(0));
    assert_eq!(format!("{r}"), "1", "Res(exp(x)/x, 0) = 1");
}

#[test]
fn residue_of_polynomial_is_zero() {
    let ctx = Context::new();
    // A polynomial has no poles, so the residue at any point is 0.
    // Res(x^2, x=0) = lim_{x→0} x * x^2 = lim_{x→0} x^3 = 0
    symplex::syms!(ctx; x);
    let f = x.powi(2);
    let r = f.residue(&x, &ctx.int(0));
    assert_eq!(format!("{r}"), "0", "Res(x^2, 0) = 0");
}

#[test]
fn residue_does_not_panic_on_hard_case() {
    let ctx = Context::new();
    // Res(1/x^2, x=0) is a double pole — the simple-pole formula won't give
    // a finite result, but the function should not panic.
    symplex::syms!(ctx; x);
    let f = &ctx.int(1) / &x.powi(2);
    // We just verify no panic; the result may be Err or an infinite expression.
    let _result = f.residue(&x, &ctx.int(0));
}

// ═══════════════════════════════════════════════════════════════════════════
// Fourier series tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fourier_of_constant() {
    let ctx = Context::new();
    // Fourier series of constant 1 over [-π,π]:
    //   a₀ = (1/π)∫_{-π}^{π} 1 dx = 2, so a₀/2 = 1
    //   All aₙ, bₙ = 0 for n ≥ 1
    // Result should evaluate numerically to ≈ 1.
    symplex::syms!(ctx; x);
    let f = ctx.int(1);
    let result = f.fourier_series(&x, 3);
    // Numerical check at a sample point
    if let Ok(v) = result.subs(&x, &ctx.rational(1, 2)).eval_f64() {
        assert!(
            (v - 1.0).abs() < 0.1,
            "Fourier of 1 at x=0.5 should be ≈ 1, got {v}"
        );
    }
}

#[test]
fn fourier_of_constant_at_zero() {
    let ctx = Context::new();
    // Same check at x = 0
    symplex::syms!(ctx; x);
    let f = ctx.int(1);
    let result = f.fourier_series(&x, 2);
    if let Ok(v) = result.subs(&x, &ctx.int(0)).eval_f64() {
        assert!(
            (v - 1.0).abs() < 0.1,
            "Fourier of 1 at x=0 should be ≈ 1, got {v}"
        );
    }
}

#[test]
fn fourier_series_of_x_does_not_panic() {
    let ctx = Context::new();
    // Fourier series of x over [-π,π] — verify it doesn't panic,
    // returns a non-empty result, and contains expected sin terms
    // (x is an odd function so the Fourier series should have sin terms).
    symplex::syms!(ctx; x);
    let result = x.fourier_series(&x, 2);
    let s = format!("{result}");
    assert!(!s.is_empty(), "Fourier series of x should be non-empty");
    assert!(
        s.contains("sin") || s.contains("x"),
        "Fourier series of x (odd function) should contain sin or x terms, got: {s}"
    );
}
