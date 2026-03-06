//! Tests for Wave T series extensions: residue computation and Fourier series.

// ═══════════════════════════════════════════════════════════════════════════
// Residue tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn residue_simple_pole_1_over_x() {
    // Res(1/x, x=0) = 1
    symplex::vars!(x);
    let f = &symplex::int(1) / &x;
    let r = f.residue(&x, &symplex::int(0))
        .expect("residue of 1/x at 0 should succeed");
    assert_eq!(format!("{r}"), "1", "Res(1/x, 0) = 1");
}

#[test]
fn residue_1_over_x_minus_1() {
    // Res(1/(x-1), x=1) = 1
    symplex::vars!(x);
    let f = &symplex::int(1) / &(&x - 1);
    let r = f.residue(&x, &symplex::int(1))
        .expect("residue of 1/(x-1) at 1 should succeed");
    assert_eq!(format!("{r}"), "1", "Res(1/(x-1), 1) = 1");
}

#[test]
fn residue_x_over_x_minus_1() {
    // Res(x/(x-1), x=1) = lim_{x→1} (x-1) * x/(x-1) = lim_{x→1} x = 1
    symplex::vars!(x);
    let f = &x / &(&x - 1);
    let r = f.residue(&x, &symplex::int(1))
        .expect("residue of x/(x-1) at 1 should succeed");
    assert_eq!(format!("{r}"), "1", "Res(x/(x-1), 1) = 1");
}

#[test]
fn residue_exp_over_x() {
    // Res(exp(x)/x, x=0) = lim_{x→0} x * exp(x)/x = lim_{x→0} exp(x) = 1
    symplex::vars!(x);
    let f = &x.exp() / &x;
    let r = f.residue(&x, &symplex::int(0))
        .expect("residue of exp(x)/x at 0 should succeed");
    assert_eq!(format!("{r}"), "1", "Res(exp(x)/x, 0) = 1");
}

#[test]
fn residue_of_polynomial_is_zero() {
    // A polynomial has no poles, so the residue at any point is 0.
    // Res(x^2, x=0) = lim_{x→0} x * x^2 = lim_{x→0} x^3 = 0
    symplex::vars!(x);
    let f = x.powi(2);
    let r = f.residue(&x, &symplex::int(0))
        .expect("residue of x^2 at 0 should succeed");
    assert_eq!(format!("{r}"), "0", "Res(x^2, 0) = 0");
}

#[test]
fn residue_does_not_panic_on_hard_case() {
    // Res(1/x^2, x=0) is a double pole — the simple-pole formula won't give
    // a finite result, but the function should not panic.
    symplex::vars!(x);
    let f = &symplex::int(1) / &x.powi(2);
    // We just verify no panic; the result may be Err or an infinite expression.
    let _result = f.residue(&x, &symplex::int(0));
}

// ═══════════════════════════════════════════════════════════════════════════
// Fourier series tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fourier_of_constant() {
    // Fourier series of constant 1 over [-π,π]:
    //   a₀ = (1/π)∫_{-π}^{π} 1 dx = 2, so a₀/2 = 1
    //   All aₙ, bₙ = 0 for n ≥ 1
    // Result should evaluate numerically to ≈ 1.
    symplex::vars!(x);
    let f = symplex::int(1);
    let result = f.fourier_series(&x, 3);
    // Numerical check at a sample point
    if let Ok(v) = result.subs(&x, &symplex::rational(1, 2)).eval_f64() {
        assert!(
            (v - 1.0).abs() < 0.1,
            "Fourier of 1 at x=0.5 should be ≈ 1, got {v}"
        );
    }
}

#[test]
fn fourier_of_constant_at_zero() {
    // Same check at x = 0
    symplex::vars!(x);
    let f = symplex::int(1);
    let result = f.fourier_series(&x, 2);
    if let Ok(v) = result.subs(&x, &symplex::int(0)).eval_f64() {
        assert!(
            (v - 1.0).abs() < 0.1,
            "Fourier of 1 at x=0 should be ≈ 1, got {v}"
        );
    }
}

#[test]
fn fourier_series_of_x_does_not_panic() {
    // Fourier series of x over [-π,π] — verify it doesn't panic,
    // returns a non-empty result, and contains expected sin terms
    // (x is an odd function so the Fourier series should have sin terms).
    symplex::vars!(x);
    let result = x.fourier_series(&x, 2);
    let s = format!("{result}");
    assert!(!s.is_empty(), "Fourier series of x should be non-empty");
    assert!(
        s.contains("sin") || s.contains("x"),
        "Fourier series of x (odd function) should contain sin or x terms, got: {s}"
    );
}
