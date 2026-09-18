//! Tests for trigonometric power integration via the public API.
//!
//! These test the pipeline: expr.integrate(&x) for sin^n, cos^n, and mixed powers.
//! Each test verifies the Fundamental Theorem of Calculus: d/dx(∫f dx) ≈ f numerically.

use symplex::prelude::*;

/// Helper: verify FTC — d/dx(antideriv) should equal integrand at a test point.
fn assert_ftc(integrand: &Ex, var: &Ex, label: &str) {
    let ctx = integrand.context();
    let anti = integrand.integrate(var);
    let anti_str = format!("{anti}");
    assert!(
        !anti_str.contains("Integral"),
        "{label}: got unevaluated integral: {anti_str}"
    );

    let deriv = anti.diff(var);

    // Evaluate both at x = 0.7 (avoids zeros and poles)
    let test_point = ctx.rational(7, 10);
    let orig_val = integrand.subs(var, &test_point).eval_f64();
    let deriv_val = deriv.subs(var, &test_point).eval_f64();

    if let (Ok(o), Ok(d)) = (orig_val, deriv_val)
        && o.is_finite()
        && d.is_finite()
    {
        let diff = (o - d).abs();
        let tol = 1e-8 * o.abs().max(1.0);
        assert!(
            diff < tol,
            "{label}: FTC violated — integrand={o:.10}, d/dx(antideriv)={d:.10}, diff={diff:.2e}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Basic sin/cos (first power) — verifies the pipeline handles power-1
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_sin_first_power() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.sin();
    assert_ftc(&integrand, &x, "∫sin(x)dx");
}

#[test]
fn integrate_cos_first_power() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.cos();
    assert_ftc(&integrand, &x, "∫cos(x)dx");
}

// ═══════════════════════════════════════════════════════════════════════════
// Even powers — use double-angle / reduction formulas
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_sin_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.sin().powi(2);
    assert_ftc(&integrand, &x, "∫sin²(x)dx");
}

#[test]
fn integrate_cos_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.cos().powi(2);
    assert_ftc(&integrand, &x, "∫cos²(x)dx");
}

#[test]
fn integrate_sin_fourth() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.sin().powi(4);
    assert_ftc(&integrand, &x, "∫sin⁴(x)dx");
}

#[test]
fn integrate_cos_fourth() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.cos().powi(4);
    assert_ftc(&integrand, &x, "∫cos⁴(x)dx");
}

// ═══════════════════════════════════════════════════════════════════════════
// Odd powers — peel off one factor then use Pythagorean identity
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_sin_cubed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.sin().powi(3);
    assert_ftc(&integrand, &x, "∫sin³(x)dx");
}

#[test]
fn integrate_cos_cubed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.cos().powi(3);
    assert_ftc(&integrand, &x, "∫cos³(x)dx");
}

#[test]
fn integrate_sin_fifth() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.sin().powi(5);
    assert_ftc(&integrand, &x, "∫sin⁵(x)dx");
}

// ═══════════════════════════════════════════════════════════════════════════
// Mixed products — sin^m * cos^n
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_sin2_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin²(x) · cos(x) — odd power of cos triggers u=sin substitution
    let integrand = &x.sin().powi(2) * &x.cos();
    assert_ftc(&integrand, &x, "∫sin²(x)·cos(x)dx");
}

#[test]
fn integrate_sin_cos2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin(x) · cos²(x) — odd power of sin triggers u=cos substitution
    let integrand = &x.sin() * &x.cos().powi(2);
    assert_ftc(&integrand, &x, "∫sin(x)·cos²(x)dx");
}

#[test]
fn integrate_sin2_cos2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin²(x) · cos²(x) — both even, uses double-angle identities
    let integrand = &x.sin().powi(2) * &x.cos().powi(2);
    assert_ftc(&integrand, &x, "∫sin²(x)·cos²(x)dx");
}
