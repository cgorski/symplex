//! Tests for parametric integration — integrands with symbolic (non-numeric)
//! coefficients such as `∫ sin(a*x) dx`, `∫ exp(a*x) dx`, etc.
//!
//! These exercises the `symbolic_linear_coeff_of` path added to the
//! integration engine.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Verify FTC numerically: d/dx(∫ f dx) ≈ f at a test point, after
/// substituting symbolic parameters with concrete values.
///
/// `param_subs` is a list of `(symbol, f64_value)` for all free parameters.
fn assert_ftc_parametric(
    integrand: &Ex,
    var: &Ex,
    param_subs: &[(&Ex, f64)],
    label: &str,
) {
    let anti = integrand.integrate(var);
    let s = format!("{anti}");
    assert!(
        !s.contains("Integral"),
        "{label}: integration returned unevaluated: {s}"
    );

    let deriv = anti.diff(var);

    // Substitute parameters first, then the variable
    let test_x = 1.0_f64;
    let x_val = symplex::default_context().rational(1, 1);

    let mut integrand_sub = integrand.clone();
    let mut deriv_sub = deriv.clone();
    for &(param, val) in param_subs {
        // Use a rational approximation: val as integer (we pick integer params)
        let val_expr = symplex::default_context().int(val as i64);
        integrand_sub = integrand_sub.subs(param, &val_expr);
        deriv_sub = deriv_sub.subs(param, &val_expr);
    }
    integrand_sub = integrand_sub.subs(var, &x_val);
    deriv_sub = deriv_sub.subs(var, &x_val);

    if let (Ok(orig), Ok(diff)) = (integrand_sub.eval_f64(), deriv_sub.eval_f64()) {
        if orig.is_finite() && diff.is_finite() {
            let err = (orig - diff).abs();
            let tol = 1e-8 * orig.abs().max(1.0);
            assert!(
                err < tol,
                "{label}: FTC failed at x={test_x} — integrand={orig:.12}, deriv={diff:.12}, err={err:.2e}",
            );
        }
    } else {
        panic!("{label}: eval_f64 failed for integrand or derivative");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ∫ sin(a*x) dx = −cos(a*x) / a
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_sin_ax() {
    let x = symplex::default_context().symbol("x");
    let a = symplex::default_context().symbol("a");
    let ax = &a * &x;
    let integrand = ax.sin();
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    assert!(!s.contains("Integral"), "should not be unevaluated: {s}");
    assert!(s.contains("cos"), "should contain cos: {s}");

    // Numerical check at a=2, x=1
    assert_ftc_parametric(&integrand, &x, &[(&a, 2.0)], "∫sin(a*x)dx");
}

#[test]
fn integrate_sin_ax_numeric_check() {
    // Verify the antiderivative equals -cos(a*x)/a numerically at a=2, x=1
    let x = symplex::default_context().symbol("x");
    let a = symplex::default_context().symbol("a");
    let ax = &a * &x;
    let anti = ax.sin().integrate(&x);

    // Substitute a=2, x=1
    let val = anti
        .subs(&a, &symplex::default_context().int(2))
        .subs(&x, &symplex::default_context().int(1))
        .eval_f64()
        .expect("should evaluate");

    // Expected: -cos(2*1)/2 = -cos(2)/2
    let expected = -(2.0_f64).cos() / 2.0;
    assert!(
        (val - expected).abs() < 1e-10,
        "∫sin(2x)dx at x=1: got {val}, expected {expected}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// ∫ cos(a*x) dx = sin(a*x) / a
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_cos_ax() {
    let x = symplex::default_context().symbol("x");
    let a = symplex::default_context().symbol("a");
    let ax = &a * &x;
    let integrand = ax.cos();
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    assert!(!s.contains("Integral"), "should not be unevaluated: {s}");
    assert!(s.contains("sin"), "should contain sin: {s}");

    assert_ftc_parametric(&integrand, &x, &[(&a, 3.0)], "∫cos(a*x)dx");
}

// ═══════════════════════════════════════════════════════════════════════════
// ∫ exp(a*x) dx = exp(a*x) / a
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_exp_ax() {
    let x = symplex::default_context().symbol("x");
    let a = symplex::default_context().symbol("a");
    let ax = &a * &x;
    let integrand = ax.exp();
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    assert!(!s.contains("Integral"), "should not be unevaluated: {s}");
    assert!(s.contains("exp"), "should contain exp: {s}");

    assert_ftc_parametric(&integrand, &x, &[(&a, 2.0)], "∫exp(a*x)dx");
}

#[test]
fn integrate_exp_ax_numeric_check() {
    let x = symplex::default_context().symbol("x");
    let a = symplex::default_context().symbol("a");
    let ax = &a * &x;
    let anti = ax.exp().integrate(&x);

    let val = anti
        .subs(&a, &symplex::default_context().int(3))
        .subs(&x, &symplex::default_context().int(1))
        .eval_f64()
        .expect("should evaluate");

    // Expected: exp(3)/3
    let expected = (3.0_f64).exp() / 3.0;
    assert!(
        (val - expected).abs() < 1e-6,
        "∫exp(3x)dx at x=1: got {val}, expected {expected}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// ∫ 1/(x² + a²) dx = (1/a)·atan(x/a)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_inv_x2_plus_a2() {
    let x = symplex::default_context().symbol("x");
    let a = symplex::default_context().symbol("a");

    // Build 1/(x² + a²) = (x² + a²)^(-1)
    let x2 = x.powi(2);
    let a2 = a.powi(2);
    let denom = &x2 + &a2;
    let integrand = denom.powi(-1);

    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    assert!(
        !s.contains("Integral"),
        "should not be unevaluated: {s}"
    );
    assert!(s.contains("atan"), "should contain atan: {s}");

    // Numerical check: at a=2, x=1
    // Expected: (1/2)*atan(1/2)
    let val = anti
        .subs(&a, &symplex::default_context().int(2))
        .subs(&x, &symplex::default_context().int(1))
        .eval_f64()
        .expect("should evaluate");

    let expected = 0.5 * (0.5_f64).atan();
    assert!(
        (val - expected).abs() < 1e-10,
        "∫1/(x²+4)dx at x=1: got {val}, expected {expected}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// ∫ sin(2*x + 3) dx — mixed numeric (works via numeric path too)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_sin_2x_plus_3() {
    let x = symplex::default_context().symbol("x");
    let two = symplex::default_context().int(2);
    let three = symplex::default_context().int(3);
    let inner = &(&two * &x) + &three;
    let integrand = inner.sin();
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    assert!(!s.contains("Integral"), "should not be unevaluated: {s}");

    // FTC check
    let deriv = anti.diff(&x);
    let test_pt = symplex::default_context().rational(7, 10);
    if let (Ok(o), Ok(d)) = (
        integrand.subs(&x, &test_pt).eval_f64(),
        deriv.subs(&x, &test_pt).eval_f64(),
    ) {
        let err = (o - d).abs();
        assert!(
            err < 1e-8,
            "FTC for ∫sin(2x+3)dx: integrand={o}, deriv={d}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ∫ a*sin(x) dx = a*(-cos(x)) = -a*cos(x)
// (constant factor pulled out — symbolic constant)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_a_times_sin_x() {
    let x = symplex::default_context().symbol("x");
    let a = symplex::default_context().symbol("a");
    let integrand = &a * &x.sin();
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    assert!(!s.contains("Integral"), "should not be unevaluated: {s}");

    // Numerical: at a=3, x=1
    let val = anti
        .subs(&a, &symplex::default_context().int(3))
        .subs(&x, &symplex::default_context().int(1))
        .eval_f64()
        .expect("should evaluate");
    let expected = 3.0 * (-(1.0_f64).cos());
    assert!(
        (val - expected).abs() < 1e-10,
        "∫a·sin(x)dx at a=3,x=1: got {val}, expected {expected}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// ∫ exp(a*x + b) dx = exp(a*x + b) / a
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_exp_ax_plus_b() {
    let x = symplex::default_context().symbol("x");
    let a = symplex::default_context().symbol("a");
    let b = symplex::default_context().symbol("b");
    let inner = &(&a * &x) + &b;
    let integrand = inner.exp();
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    assert!(!s.contains("Integral"), "should not be unevaluated: {s}");
    assert!(s.contains("exp"), "should contain exp: {s}");

    // Numerical: a=2, b=1, x=1
    // Expected: exp(2+1)/2 = exp(3)/2
    let val = anti
        .subs(&a, &symplex::default_context().int(2))
        .subs(&b, &symplex::default_context().int(1))
        .subs(&x, &symplex::default_context().int(1))
        .eval_f64()
        .expect("should evaluate");
    let expected = (3.0_f64).exp() / 2.0;
    assert!(
        (val - expected).abs() < 1e-6,
        "∫exp(2x+1)dx at x=1: got {val}, expected {expected}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// ∫ tan(a*x) dx — parametric tangent
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_tan_ax() {
    let x = symplex::default_context().symbol("x");
    let a = symplex::default_context().symbol("a");
    let ax = &a * &x;
    let integrand = ax.tan();
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    assert!(!s.contains("Integral"), "should not be unevaluated: {s}");

    assert_ftc_parametric(&integrand, &x, &[(&a, 2.0)], "∫tan(a*x)dx");
}

// ═══════════════════════════════════════════════════════════════════════════
// ∫ sinh(a*x) dx — parametric hyperbolic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_sinh_ax() {
    let x = symplex::default_context().symbol("x");
    let a = symplex::default_context().symbol("a");
    let ax = &a * &x;
    let integrand = ax.sinh();
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    assert!(!s.contains("Integral"), "should not be unevaluated: {s}");

    assert_ftc_parametric(&integrand, &x, &[(&a, 2.0)], "∫sinh(a*x)dx");
}

// ═══════════════════════════════════════════════════════════════════════════
// ∫ cosh(a*x) dx — parametric hyperbolic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_cosh_ax() {
    let x = symplex::default_context().symbol("x");
    let a = symplex::default_context().symbol("a");
    let ax = &a * &x;
    let integrand = ax.cosh();
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    assert!(!s.contains("Integral"), "should not be unevaluated: {s}");

    assert_ftc_parametric(&integrand, &x, &[(&a, 3.0)], "∫cosh(a*x)dx");
}

// ═══════════════════════════════════════════════════════════════════════════
// ∫ (a*x + b)^3 dx — parametric power
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_ax_plus_b_cubed() {
    let x = symplex::default_context().symbol("x");
    let a = symplex::default_context().symbol("a");
    let b = symplex::default_context().symbol("b");
    let inner = &(&a * &x) + &b;
    let integrand = inner.powi(3);
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    assert!(!s.contains("Integral"), "should not be unevaluated: {s}");

    // Numerical check: a=2, b=1, x=1: (2+1)^4 / (4*2) = 81/8
    let val = anti
        .subs(&a, &symplex::default_context().int(2))
        .subs(&b, &symplex::default_context().int(1))
        .subs(&x, &symplex::default_context().int(1))
        .eval_f64()
        .expect("should evaluate");
    let expected = 81.0 / 8.0;
    assert!(
        (val - expected).abs() < 1e-10,
        "∫(2x+1)³dx at x=1: got {val}, expected {expected}"
    );
}
